/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types.h"

#include "gemm_gateup.h"

#define QUANTIZE_2D 2

#define QUANTIZE_COMMON 3

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define sg_per_wg (ugemm_wgu_sg_per_wg_m * ugemm_wgu_sg_per_wg_n)
#define wgu_tile_sg_n DIV_UP(ugemm_wgu_wg_tile_n, sg_per_wg)
#define wgu_tile_sg_m DIV_UP(ugemm_wgu_wg_tile_m, sg_per_wg)

typedef ugemm_wgu_c_type s_tile_type;

#ifdef SRC_DT_F16
#define VEC_TYPE1 half
#define VEC_TYPE2 half2
#elif defined(SRC_DT_BF16)
#define VEC_TYPE1 ushort
#define VEC_TYPE2 ushort2
#else
#error "Data type not supported for VEC_TYPE2"
#endif

DECLARE_2D_TILE(wgu_tile_type, uint, SUBGROUP_SIZE, ugemm_wgu_wg_tile_m / 2, 1,
        1, wgu_tile_sg_n)

#ifdef BLOCK_SRC
DECLARE_2D_TILE_BLOCK_OPS(wgu_tile_type, uint, SUBGROUP_SIZE,
        ugemm_wgu_wg_tile_m / 2, 1, 1, wgu_tile_sg_n)
#elif SRC_ALIGN < 4
DECLARE_2D_TILE_LOAD_PACKED_VEC(wgu_tile_type, SRC_DATA_T, VEC_TYPE2,
        SUBGROUP_SIZE, ugemm_wgu_wg_tile_m / 2, 1, 1, wgu_tile_sg_n)
#endif

#if REMAINDER_SRC
#define tile_load_block_rem_src tile_load_block
#define tile_store_block_rem_wgu tile_store_block
#else
#define tile_load_block_rem_src(t, ptr, n, ld, off_r, off_c) \
    tile_load_block(t, ptr, ld, off_r, off_c)
#define tile_store_block_rem_wgu(t, ptr, n, ld, off_r, off_c) \
    tile_store_block(t, ptr, ld, off_r, off_c)
#endif

#define binary_add(x, y) ((x) + (y))
#define binary_mul(x, y) ((x) * (y))

#ifdef ACTIVATION_SWISH

#define unary_activation(x) ((x) / (1.f + exp(-1.f * (x))))

#elif defined ACTIVATION_GELU_ERF

#define sqrt_2_over_2 0.707106769084930419921875f
#define unary_activation(x) (0.5f * (x) * (1.f + erf((x) * sqrt_2_over_2)))

#elif defined ACTIVATION_GELU_TANH

#define sqrt_2_over_pi 0.79788458347320556640625f
#define fitting_const 0.044715f
#define unary_activation(x) \
    (0.5f * (x) \
            * (1.f \
                    + tanh(sqrt_2_over_pi * (x) \
                            * (1 + fitting_const * (x) * (x)))))

#else
#error "Unknown activation function defined"
#endif

#define SG_TILE_BR ugemm_wgu_sg_tile_n
#define SG_TILE_BC 1
#define SG_TILE_NBR 1
#define SG_TILE_NBC ugemm_wgu_sg_tile_m

DECLARE_2D_TILE(s_tile_type_t, float, SUBGROUP_SIZE,
        SG_TILE_BR, SG_TILE_BC, SG_TILE_NBR, SG_TILE_NBC)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_t, float, SUBGROUP_SIZE,
        SG_TILE_BR, SG_TILE_BC, SG_TILE_NBR, SG_TILE_NBC)

DECLARE_2D_TILE(s_tile_type_dst, VEC_TYPE1, SUBGROUP_SIZE,
        SG_TILE_BR, SG_TILE_BC, SG_TILE_NBR, SG_TILE_NBC)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_dst, VEC_TYPE1, SUBGROUP_SIZE,
        SG_TILE_BR, SG_TILE_BC, SG_TILE_NBR, SG_TILE_NBC)

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type_t, SUBGROUP_SIZE,
        ugemm_wgu_c_type_block0, ugemm_wgu_c_type_block1,
        ugemm_wgu_c_type_nblock0, ugemm_wgu_c_type_nblock1, s_tile_type_dst,
        SUBGROUP_SIZE, SG_TILE_BR, SG_TILE_BC, SG_TILE_NBR, SG_TILE_NBC,
        CONVERT_DATA_T)

DECLARE_2D_TILE_SLM_OP_T(s_tile_type, float, SUBGROUP_SIZE,
        ugemm_wgu_c_type_block0, ugemm_wgu_c_type_block1,
        ugemm_wgu_c_type_nblock0, ugemm_wgu_c_type_nblock1, mov, =)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
micro_gated_mlp_horz(const __global SRC_DATA_T *src,
        const __global WTS_GATE_DATA_T *W_gate,
        const __global WTS_UP_DATA_T *W_up,
        const __global WTS_DOWN_DATA_T *W_down, __global DST_DATA_T *dst,
        long MB, long IC, long OC, __global INTER_DATA_T *tmp_reduce_mem,
        const __global WTS_GATE_ATTR_SCALES_DATA_T *wts_gate_scales,
        const __global WTS_GATE_ATTR_ZP_DATA_T *wts_gate_zp,
        const __global WTS_UP_ATTR_SCALES_DATA_T *wts_up_scales,
        const __global WTS_UP_ATTR_ZP_DATA_T *wts_up_zp,
        const __global WTS_DOWN_ATTR_SCALES_DATA_T *wts_down_scales,
        const __global WTS_DOWN_ATTR_ZP_DATA_T *wts_down_zp) {

if ((get_global_id(0) == 0) && (get_global_id(1) == 0) && (get_global_id(2) == 0))
    printf("ugemm_wgu_wg_tile_m = %u, ugemm_wgu_wg_tile_n = %u\n"
           "ugemm_wgu_sg_tile_m = %u, ugemm_wgu_sg_tile_n = %u\n"
           "ugemm_wgu_sg_per_wg_m = %u, ugemm_wgu_sg_per_wg_n = %u\n"
           "ugemm_wgu_c_type_block0 = %u, ugemm_wgu_c_type_block1 = %u\n"
           "ugemm_wgu_c_type_nblock0 = %u, ugemm_wgu_c_type_nblock1 = %u\n"
           , (uint)ugemm_wgu_wg_tile_m, (uint)ugemm_wgu_wg_tile_n
           , (uint)ugemm_wgu_sg_tile_m, (uint)ugemm_wgu_sg_tile_n
           , (uint)ugemm_wgu_sg_per_wg_m, (uint)ugemm_wgu_sg_per_wg_n
           , (uint)ugemm_wgu_c_type_block0, (uint)ugemm_wgu_c_type_block1
           , (uint)ugemm_wgu_c_type_nblock0, (uint)ugemm_wgu_c_type_nblock1
    );

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    uint wg_j0 = get_group_id(0) * ugemm_wgu_wg_tile_m; // OC
    uint wg_i0 = get_group_id(2) * ugemm_wgu_wg_tile_n; // MB

    uint lds = SRC_S0;
    uint ldg = W_GATE_S1;
    uint ldu = W_UP_S1;
    uint ldi = INTER_S0;

#if WTS_GATE_SCALES || WTS_GATE_ZERO_POINTS
    uint ldgq = OC;
#endif
#if WTS_UP_SCALES || WTS_UP_ZERO_POINTS
    uint lduq = OC;
#endif

#if WTS_GATE_SCALES == QUANTIZE_COMMON
    float wg_scale = convert_float(*wts_gate_scales);
#endif
#if WTS_UP_SCALES == QUANTIZE_COMMON
    float wu_scale = convert_float(*wts_up_scales);
#endif

    // TODO: 2 options possible there: either /wg_m <-> /wg_n, or i <-> j
    uint sg_i_wgu = sg_ij % ugemm_wgu_sg_per_wg_n;
    uint sg_j_wgu = sg_ij / ugemm_wgu_sg_per_wg_n;

#define WGU_slm_size (ugemm_wgu_wg_tile_m * ugemm_wgu_wg_tile_n)

    local char slm[MAX(WGU_slm_size * sizeof(float),
            WGU_slm_size * sizeof(SRC_DATA_T) + ugemm_wgu_slm_size)];
    local char *slm_ptr = slm;

    local SRC_DATA_T *wg_slm = (local SRC_DATA_T *)slm_ptr;
    slm_ptr += WGU_slm_size * sizeof(SRC_DATA_T);
    local char *ugemm_gu_slm = slm_ptr;

    wgu_tile_type src_tile;
    uint wgu0_copy = wgu_tile_sg_n * sg_ij;

#ifndef UGEMM_UP_ONLY
    s_tile_type S_WG_tile;
    tile_fill(S_WG_tile, 0.0f);
#endif
    s_tile_type S_WU_tile;
    tile_fill(S_WU_tile, 0.0f);

    for (int k0 = 0; k0 < IC; k0 += ugemm_wgu_wg_tile_m) {

#ifdef BLOCK_SRC
        tile_load_block_rem_src(&src_tile, (global uint *)src, MB, lds >> 1,
                k0 / 2, wg_i0 + wgu0_copy);
#elif SRC_ALIGN >= 4
        tile_load(&src_tile, (global uint *)src, (lds + 1) >> 1, IC, lds >> 1,
                k0 / 2, wg_i0 + wgu0_copy);
#else
        tile_load_packed_vec2(
                &src_tile, src, IC, MB, lds, k0, wg_i0 + wgu0_copy);
#endif

        tile_store_t_sys_src1(src_tile, (local uint *)&wg_slm[0],
                ugemm_wgu_wg_tile_m / 2, wgu0_copy, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

#ifndef UGEMM_UP_ONLY
        s_tile_type FC_G_tile
                = ugemm_wgu(W_gate + k0 / WTS_GATE_ELEMENTS_PER_BYTE, ldg,
                        wg_slm, ugemm_wgu_wg_tile_m, OC, ugemm_wgu_wg_tile_n,
                        ugemm_wgu_wg_tile_m, wg_j0, 0, 0, sg_j_wgu, sg_i_wgu,
                        ugemm_gu_slm
#if WTS_GATE_SCALES == QUANTIZE_2D
                        ,
                        wts_gate_scales + (k0 / WTS_GATE_GROUP_SIZE) * ldgq
#endif
#if WTS_GATE_ZERO_POINTS
                        ,
                        wts_gate_zp
                                + (k0 / WTS_GATE_GROUP_SIZE) * ldgq
                                        / WTS_GATE_ZP_ELEMENTS_PER_BYTE
#endif
#if (WTS_GATE_SCALES == QUANTIZE_2D) || WTS_GATE_ZERO_POINTS
                        ,
                        ldgq
#endif
                );

#if WTS_GATE_SCALES == QUANTIZE_COMMON
#define wg_scale_op(x) ((x) * wg_scale)
        tile_elementwise(FC_G_tile, wg_scale_op);
#endif

        // TODO: S_W[G,U]_tile might end up clobbered at each ukernel call!
        //       The proper solution for now is to acccumulate right to SLM.
        tile_binary(S_WG_tile, FC_G_tile, binary_add);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif // UGEMM_UP_ONLY

        s_tile_type FC_U_tile = ugemm_wgu(W_up + k0 / WTS_UP_ELEMENTS_PER_BYTE,
                ldu, wg_slm, ugemm_wgu_wg_tile_m, OC, ugemm_wgu_wg_tile_n,
                ugemm_wgu_wg_tile_m, wg_j0, 0, 0, sg_j_wgu, sg_i_wgu,
                ugemm_gu_slm
#if WTS_UP_SCALES == QUANTIZE_2D
                ,
                wts_up_scales + (k0 / WTS_UP_GROUP_SIZE) * lduq
#endif
#if WTS_UP_ZERO_POINTS
                ,
                wts_up_zp
                        + (k0 / WTS_UP_GROUP_SIZE) * lduq
                                / WTS_UP_ZP_ELEMENTS_PER_BYTE
#endif
#if (WTS_UP_SCALES == QUANTIZE_2D) || WTS_UP_ZERO_POINTS
                ,
                lduq
#endif
        );

#if WTS_UP_SCALES == QUANTIZE_COMMON
#define wu_scale_op(x) ((x) * wu_scale)
        tile_elementwise(FC_U_tile, wu_scale_op);
#endif
        tile_binary(S_WU_tile, FC_U_tile, binary_add);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#ifndef UGEMM_UP_ONLY
    tile_elementwise(S_WG_tile, unary_activation);
    tile_binary(S_WU_tile, S_WG_tile, binary_mul);
#endif

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_n;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_m;
    size_t k_offset = get_group_id(1) / sg_per_wg * OC * MB;

    tile_slm_mov_t(S_WU_tile, (local float *)slm, ugemm_wgu_wg_tile_m,
            sg_i0_wgu, sg_j0_wgu);
    barrier(CLK_LOCAL_MEM_FENCE);

/*
if ((get_global_id(0) == 0) && (get_global_id(1) == 0))
for (int i = 0; i < 2; i++) {
local float *slm0 = (local float *)slm + 128 * (8 * i + 0);
local float *slm1 = (local float *)slm + 128 * (8 * i + 1);
local float *slm2 = (local float *)slm + 128 * (8 * i + 2);
local float *slm3 = (local float *)slm + 128 * (8 * i + 3);
local float *slm4 = (local float *)slm + 128 * (8 * i + 4);
local float *slm5 = (local float *)slm + 128 * (8 * i + 5);
local float *slm6 = (local float *)slm + 128 * (8 * i + 6);
local float *slm7 = (local float *)slm + 128 * (8 * i + 7);
printf("DST SLM part %d (%ld %ld %ld):\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
, i + 1, get_global_id(0), get_global_id(1), get_global_id(2)
, convert_float(slm0[  0]), convert_float(slm0[  1]), convert_float(slm0[  2]), convert_float(slm0[  3])
, convert_float(slm0[  4]), convert_float(slm0[  5]), convert_float(slm0[  6]), convert_float(slm0[  7])
, convert_float(slm0[  8]), convert_float(slm0[  9]), convert_float(slm0[ 10]), convert_float(slm0[ 11])
, convert_float(slm0[ 12]), convert_float(slm0[ 13]), convert_float(slm0[ 14]), convert_float(slm0[ 15])
, convert_float(slm0[ 16]), convert_float(slm0[ 17]), convert_float(slm0[ 18]), convert_float(slm0[ 19])
, convert_float(slm0[ 20]), convert_float(slm0[ 21]), convert_float(slm0[ 22]), convert_float(slm0[ 23])
, convert_float(slm0[ 24]), convert_float(slm0[ 25]), convert_float(slm0[ 26]), convert_float(slm0[ 27])
, convert_float(slm0[ 28]), convert_float(slm0[ 29]), convert_float(slm0[ 30]), convert_float(slm0[ 31])
, convert_float(slm0[ 32]), convert_float(slm0[ 33]), convert_float(slm0[ 34]), convert_float(slm0[ 35])
, convert_float(slm0[ 36]), convert_float(slm0[ 37]), convert_float(slm0[ 38]), convert_float(slm0[ 39])
, convert_float(slm0[ 40]), convert_float(slm0[ 41]), convert_float(slm0[ 42]), convert_float(slm0[ 43])
, convert_float(slm0[ 44]), convert_float(slm0[ 45]), convert_float(slm0[ 46]), convert_float(slm0[ 47])
, convert_float(slm0[ 48]), convert_float(slm0[ 49]), convert_float(slm0[ 50]), convert_float(slm0[ 51])
, convert_float(slm0[ 52]), convert_float(slm0[ 53]), convert_float(slm0[ 54]), convert_float(slm0[ 55])
, convert_float(slm0[ 56]), convert_float(slm0[ 57]), convert_float(slm0[ 58]), convert_float(slm0[ 59])
, convert_float(slm0[ 60]), convert_float(slm0[ 61]), convert_float(slm0[ 62]), convert_float(slm0[ 63])
, convert_float(slm0[ 64]), convert_float(slm0[ 65]), convert_float(slm0[ 66]), convert_float(slm0[ 67])
, convert_float(slm0[ 68]), convert_float(slm0[ 69]), convert_float(slm0[ 70]), convert_float(slm0[ 71])
, convert_float(slm0[ 72]), convert_float(slm0[ 73]), convert_float(slm0[ 74]), convert_float(slm0[ 75])
, convert_float(slm0[ 76]), convert_float(slm0[ 77]), convert_float(slm0[ 78]), convert_float(slm0[ 79])
, convert_float(slm0[ 80]), convert_float(slm0[ 81]), convert_float(slm0[ 82]), convert_float(slm0[ 83])
, convert_float(slm0[ 84]), convert_float(slm0[ 85]), convert_float(slm0[ 86]), convert_float(slm0[ 87])
, convert_float(slm0[ 88]), convert_float(slm0[ 89]), convert_float(slm0[ 90]), convert_float(slm0[ 91])
, convert_float(slm0[ 92]), convert_float(slm0[ 93]), convert_float(slm0[ 94]), convert_float(slm0[ 95])
, convert_float(slm0[ 96]), convert_float(slm0[ 97]), convert_float(slm0[ 98]), convert_float(slm0[ 99])
, convert_float(slm0[100]), convert_float(slm0[101]), convert_float(slm0[102]), convert_float(slm0[103])
, convert_float(slm0[104]), convert_float(slm0[105]), convert_float(slm0[106]), convert_float(slm0[107])
, convert_float(slm0[108]), convert_float(slm0[109]), convert_float(slm0[110]), convert_float(slm0[111])
, convert_float(slm0[112]), convert_float(slm0[113]), convert_float(slm0[114]), convert_float(slm0[115])
, convert_float(slm0[116]), convert_float(slm0[117]), convert_float(slm0[118]), convert_float(slm0[119])
, convert_float(slm0[120]), convert_float(slm0[121]), convert_float(slm0[122]), convert_float(slm0[123])
, convert_float(slm0[124]), convert_float(slm0[125]), convert_float(slm0[126]), convert_float(slm0[127])
, convert_float(slm1[  0]), convert_float(slm1[  1]), convert_float(slm1[  2]), convert_float(slm1[  3])
, convert_float(slm1[  4]), convert_float(slm1[  5]), convert_float(slm1[  6]), convert_float(slm1[  7])
, convert_float(slm1[  8]), convert_float(slm1[  9]), convert_float(slm1[ 10]), convert_float(slm1[ 11])
, convert_float(slm1[ 12]), convert_float(slm1[ 13]), convert_float(slm1[ 14]), convert_float(slm1[ 15])
, convert_float(slm1[ 16]), convert_float(slm1[ 17]), convert_float(slm1[ 18]), convert_float(slm1[ 19])
, convert_float(slm1[ 20]), convert_float(slm1[ 21]), convert_float(slm1[ 22]), convert_float(slm1[ 23])
, convert_float(slm1[ 24]), convert_float(slm1[ 25]), convert_float(slm1[ 26]), convert_float(slm1[ 27])
, convert_float(slm1[ 28]), convert_float(slm1[ 29]), convert_float(slm1[ 30]), convert_float(slm1[ 31])
, convert_float(slm1[ 32]), convert_float(slm1[ 33]), convert_float(slm1[ 34]), convert_float(slm1[ 35])
, convert_float(slm1[ 36]), convert_float(slm1[ 37]), convert_float(slm1[ 38]), convert_float(slm1[ 39])
, convert_float(slm1[ 40]), convert_float(slm1[ 41]), convert_float(slm1[ 42]), convert_float(slm1[ 43])
, convert_float(slm1[ 44]), convert_float(slm1[ 45]), convert_float(slm1[ 46]), convert_float(slm1[ 47])
, convert_float(slm1[ 48]), convert_float(slm1[ 49]), convert_float(slm1[ 50]), convert_float(slm1[ 51])
, convert_float(slm1[ 52]), convert_float(slm1[ 53]), convert_float(slm1[ 54]), convert_float(slm1[ 55])
, convert_float(slm1[ 56]), convert_float(slm1[ 57]), convert_float(slm1[ 58]), convert_float(slm1[ 59])
, convert_float(slm1[ 60]), convert_float(slm1[ 61]), convert_float(slm1[ 62]), convert_float(slm1[ 63])
, convert_float(slm1[ 64]), convert_float(slm1[ 65]), convert_float(slm1[ 66]), convert_float(slm1[ 67])
, convert_float(slm1[ 68]), convert_float(slm1[ 69]), convert_float(slm1[ 70]), convert_float(slm1[ 71])
, convert_float(slm1[ 72]), convert_float(slm1[ 73]), convert_float(slm1[ 74]), convert_float(slm1[ 75])
, convert_float(slm1[ 76]), convert_float(slm1[ 77]), convert_float(slm1[ 78]), convert_float(slm1[ 79])
, convert_float(slm1[ 80]), convert_float(slm1[ 81]), convert_float(slm1[ 82]), convert_float(slm1[ 83])
, convert_float(slm1[ 84]), convert_float(slm1[ 85]), convert_float(slm1[ 86]), convert_float(slm1[ 87])
, convert_float(slm1[ 88]), convert_float(slm1[ 89]), convert_float(slm1[ 90]), convert_float(slm1[ 91])
, convert_float(slm1[ 92]), convert_float(slm1[ 93]), convert_float(slm1[ 94]), convert_float(slm1[ 95])
, convert_float(slm1[ 96]), convert_float(slm1[ 97]), convert_float(slm1[ 98]), convert_float(slm1[ 99])
, convert_float(slm1[100]), convert_float(slm1[101]), convert_float(slm1[102]), convert_float(slm1[103])
, convert_float(slm1[104]), convert_float(slm1[105]), convert_float(slm1[106]), convert_float(slm1[107])
, convert_float(slm1[108]), convert_float(slm1[109]), convert_float(slm1[110]), convert_float(slm1[111])
, convert_float(slm1[112]), convert_float(slm1[113]), convert_float(slm1[114]), convert_float(slm1[115])
, convert_float(slm1[116]), convert_float(slm1[117]), convert_float(slm1[118]), convert_float(slm1[119])
, convert_float(slm1[120]), convert_float(slm1[121]), convert_float(slm1[122]), convert_float(slm1[123])
, convert_float(slm1[124]), convert_float(slm1[125]), convert_float(slm1[126]), convert_float(slm1[127])
, convert_float(slm2[  0]), convert_float(slm2[  1]), convert_float(slm2[  2]), convert_float(slm2[  3])
, convert_float(slm2[  4]), convert_float(slm2[  5]), convert_float(slm2[  6]), convert_float(slm2[  7])
, convert_float(slm2[  8]), convert_float(slm2[  9]), convert_float(slm2[ 10]), convert_float(slm2[ 11])
, convert_float(slm2[ 12]), convert_float(slm2[ 13]), convert_float(slm2[ 14]), convert_float(slm2[ 15])
, convert_float(slm2[ 16]), convert_float(slm2[ 17]), convert_float(slm2[ 18]), convert_float(slm2[ 19])
, convert_float(slm2[ 20]), convert_float(slm2[ 21]), convert_float(slm2[ 22]), convert_float(slm2[ 23])
, convert_float(slm2[ 24]), convert_float(slm2[ 25]), convert_float(slm2[ 26]), convert_float(slm2[ 27])
, convert_float(slm2[ 28]), convert_float(slm2[ 29]), convert_float(slm2[ 30]), convert_float(slm2[ 31])
, convert_float(slm2[ 32]), convert_float(slm2[ 33]), convert_float(slm2[ 34]), convert_float(slm2[ 35])
, convert_float(slm2[ 36]), convert_float(slm2[ 37]), convert_float(slm2[ 38]), convert_float(slm2[ 39])
, convert_float(slm2[ 40]), convert_float(slm2[ 41]), convert_float(slm2[ 42]), convert_float(slm2[ 43])
, convert_float(slm2[ 44]), convert_float(slm2[ 45]), convert_float(slm2[ 46]), convert_float(slm2[ 47])
, convert_float(slm2[ 48]), convert_float(slm2[ 49]), convert_float(slm2[ 50]), convert_float(slm2[ 51])
, convert_float(slm2[ 52]), convert_float(slm2[ 53]), convert_float(slm2[ 54]), convert_float(slm2[ 55])
, convert_float(slm2[ 56]), convert_float(slm2[ 57]), convert_float(slm2[ 58]), convert_float(slm2[ 59])
, convert_float(slm2[ 60]), convert_float(slm2[ 61]), convert_float(slm2[ 62]), convert_float(slm2[ 63])
, convert_float(slm2[ 64]), convert_float(slm2[ 65]), convert_float(slm2[ 66]), convert_float(slm2[ 67])
, convert_float(slm2[ 68]), convert_float(slm2[ 69]), convert_float(slm2[ 70]), convert_float(slm2[ 71])
, convert_float(slm2[ 72]), convert_float(slm2[ 73]), convert_float(slm2[ 74]), convert_float(slm2[ 75])
, convert_float(slm2[ 76]), convert_float(slm2[ 77]), convert_float(slm2[ 78]), convert_float(slm2[ 79])
, convert_float(slm2[ 80]), convert_float(slm2[ 81]), convert_float(slm2[ 82]), convert_float(slm2[ 83])
, convert_float(slm2[ 84]), convert_float(slm2[ 85]), convert_float(slm2[ 86]), convert_float(slm2[ 87])
, convert_float(slm2[ 88]), convert_float(slm2[ 89]), convert_float(slm2[ 90]), convert_float(slm2[ 91])
, convert_float(slm2[ 92]), convert_float(slm2[ 93]), convert_float(slm2[ 94]), convert_float(slm2[ 95])
, convert_float(slm2[ 96]), convert_float(slm2[ 97]), convert_float(slm2[ 98]), convert_float(slm2[ 99])
, convert_float(slm2[100]), convert_float(slm2[101]), convert_float(slm2[102]), convert_float(slm2[103])
, convert_float(slm2[104]), convert_float(slm2[105]), convert_float(slm2[106]), convert_float(slm2[107])
, convert_float(slm2[108]), convert_float(slm2[109]), convert_float(slm2[110]), convert_float(slm2[111])
, convert_float(slm2[112]), convert_float(slm2[113]), convert_float(slm2[114]), convert_float(slm2[115])
, convert_float(slm2[116]), convert_float(slm2[117]), convert_float(slm2[118]), convert_float(slm2[119])
, convert_float(slm2[120]), convert_float(slm2[121]), convert_float(slm2[122]), convert_float(slm2[123])
, convert_float(slm2[124]), convert_float(slm2[125]), convert_float(slm2[126]), convert_float(slm2[127])
, convert_float(slm3[  0]), convert_float(slm3[  1]), convert_float(slm3[  2]), convert_float(slm3[  3])
, convert_float(slm3[  4]), convert_float(slm3[  5]), convert_float(slm3[  6]), convert_float(slm3[  7])
, convert_float(slm3[  8]), convert_float(slm3[  9]), convert_float(slm3[ 10]), convert_float(slm3[ 11])
, convert_float(slm3[ 12]), convert_float(slm3[ 13]), convert_float(slm3[ 14]), convert_float(slm3[ 15])
, convert_float(slm3[ 16]), convert_float(slm3[ 17]), convert_float(slm3[ 18]), convert_float(slm3[ 19])
, convert_float(slm3[ 20]), convert_float(slm3[ 21]), convert_float(slm3[ 22]), convert_float(slm3[ 23])
, convert_float(slm3[ 24]), convert_float(slm3[ 25]), convert_float(slm3[ 26]), convert_float(slm3[ 27])
, convert_float(slm3[ 28]), convert_float(slm3[ 29]), convert_float(slm3[ 30]), convert_float(slm3[ 31])
, convert_float(slm3[ 32]), convert_float(slm3[ 33]), convert_float(slm3[ 34]), convert_float(slm3[ 35])
, convert_float(slm3[ 36]), convert_float(slm3[ 37]), convert_float(slm3[ 38]), convert_float(slm3[ 39])
, convert_float(slm3[ 40]), convert_float(slm3[ 41]), convert_float(slm3[ 42]), convert_float(slm3[ 43])
, convert_float(slm3[ 44]), convert_float(slm3[ 45]), convert_float(slm3[ 46]), convert_float(slm3[ 47])
, convert_float(slm3[ 48]), convert_float(slm3[ 49]), convert_float(slm3[ 50]), convert_float(slm3[ 51])
, convert_float(slm3[ 52]), convert_float(slm3[ 53]), convert_float(slm3[ 54]), convert_float(slm3[ 55])
, convert_float(slm3[ 56]), convert_float(slm3[ 57]), convert_float(slm3[ 58]), convert_float(slm3[ 59])
, convert_float(slm3[ 60]), convert_float(slm3[ 61]), convert_float(slm3[ 62]), convert_float(slm3[ 63])
, convert_float(slm3[ 64]), convert_float(slm3[ 65]), convert_float(slm3[ 66]), convert_float(slm3[ 67])
, convert_float(slm3[ 68]), convert_float(slm3[ 69]), convert_float(slm3[ 70]), convert_float(slm3[ 71])
, convert_float(slm3[ 72]), convert_float(slm3[ 73]), convert_float(slm3[ 74]), convert_float(slm3[ 75])
, convert_float(slm3[ 76]), convert_float(slm3[ 77]), convert_float(slm3[ 78]), convert_float(slm3[ 79])
, convert_float(slm3[ 80]), convert_float(slm3[ 81]), convert_float(slm3[ 82]), convert_float(slm3[ 83])
, convert_float(slm3[ 84]), convert_float(slm3[ 85]), convert_float(slm3[ 86]), convert_float(slm3[ 87])
, convert_float(slm3[ 88]), convert_float(slm3[ 89]), convert_float(slm3[ 90]), convert_float(slm3[ 91])
, convert_float(slm3[ 92]), convert_float(slm3[ 93]), convert_float(slm3[ 94]), convert_float(slm3[ 95])
, convert_float(slm3[ 96]), convert_float(slm3[ 97]), convert_float(slm3[ 98]), convert_float(slm3[ 99])
, convert_float(slm3[100]), convert_float(slm3[101]), convert_float(slm3[102]), convert_float(slm3[103])
, convert_float(slm3[104]), convert_float(slm3[105]), convert_float(slm3[106]), convert_float(slm3[107])
, convert_float(slm3[108]), convert_float(slm3[109]), convert_float(slm3[110]), convert_float(slm3[111])
, convert_float(slm3[112]), convert_float(slm3[113]), convert_float(slm3[114]), convert_float(slm3[115])
, convert_float(slm3[116]), convert_float(slm3[117]), convert_float(slm3[118]), convert_float(slm3[119])
, convert_float(slm3[120]), convert_float(slm3[121]), convert_float(slm3[122]), convert_float(slm3[123])
, convert_float(slm3[124]), convert_float(slm3[125]), convert_float(slm3[126]), convert_float(slm3[127])
, convert_float(slm4[  0]), convert_float(slm4[  1]), convert_float(slm4[  2]), convert_float(slm4[  3])
, convert_float(slm4[  4]), convert_float(slm4[  5]), convert_float(slm4[  6]), convert_float(slm4[  7])
, convert_float(slm4[  8]), convert_float(slm4[  9]), convert_float(slm4[ 10]), convert_float(slm4[ 11])
, convert_float(slm4[ 12]), convert_float(slm4[ 13]), convert_float(slm4[ 14]), convert_float(slm4[ 15])
, convert_float(slm4[ 16]), convert_float(slm4[ 17]), convert_float(slm4[ 18]), convert_float(slm4[ 19])
, convert_float(slm4[ 20]), convert_float(slm4[ 21]), convert_float(slm4[ 22]), convert_float(slm4[ 23])
, convert_float(slm4[ 24]), convert_float(slm4[ 25]), convert_float(slm4[ 26]), convert_float(slm4[ 27])
, convert_float(slm4[ 28]), convert_float(slm4[ 29]), convert_float(slm4[ 30]), convert_float(slm4[ 31])
, convert_float(slm4[ 32]), convert_float(slm4[ 33]), convert_float(slm4[ 34]), convert_float(slm4[ 35])
, convert_float(slm4[ 36]), convert_float(slm4[ 37]), convert_float(slm4[ 38]), convert_float(slm4[ 39])
, convert_float(slm4[ 40]), convert_float(slm4[ 41]), convert_float(slm4[ 42]), convert_float(slm4[ 43])
, convert_float(slm4[ 44]), convert_float(slm4[ 45]), convert_float(slm4[ 46]), convert_float(slm4[ 47])
, convert_float(slm4[ 48]), convert_float(slm4[ 49]), convert_float(slm4[ 50]), convert_float(slm4[ 51])
, convert_float(slm4[ 52]), convert_float(slm4[ 53]), convert_float(slm4[ 54]), convert_float(slm4[ 55])
, convert_float(slm4[ 56]), convert_float(slm4[ 57]), convert_float(slm4[ 58]), convert_float(slm4[ 59])
, convert_float(slm4[ 60]), convert_float(slm4[ 61]), convert_float(slm4[ 62]), convert_float(slm4[ 63])
, convert_float(slm4[ 64]), convert_float(slm4[ 65]), convert_float(slm4[ 66]), convert_float(slm4[ 67])
, convert_float(slm4[ 68]), convert_float(slm4[ 69]), convert_float(slm4[ 70]), convert_float(slm4[ 71])
, convert_float(slm4[ 72]), convert_float(slm4[ 73]), convert_float(slm4[ 74]), convert_float(slm4[ 75])
, convert_float(slm4[ 76]), convert_float(slm4[ 77]), convert_float(slm4[ 78]), convert_float(slm4[ 79])
, convert_float(slm4[ 80]), convert_float(slm4[ 81]), convert_float(slm4[ 82]), convert_float(slm4[ 83])
, convert_float(slm4[ 84]), convert_float(slm4[ 85]), convert_float(slm4[ 86]), convert_float(slm4[ 87])
, convert_float(slm4[ 88]), convert_float(slm4[ 89]), convert_float(slm4[ 90]), convert_float(slm4[ 91])
, convert_float(slm4[ 92]), convert_float(slm4[ 93]), convert_float(slm4[ 94]), convert_float(slm4[ 95])
, convert_float(slm4[ 96]), convert_float(slm4[ 97]), convert_float(slm4[ 98]), convert_float(slm4[ 99])
, convert_float(slm4[100]), convert_float(slm4[101]), convert_float(slm4[102]), convert_float(slm4[103])
, convert_float(slm4[104]), convert_float(slm4[105]), convert_float(slm4[106]), convert_float(slm4[107])
, convert_float(slm4[108]), convert_float(slm4[109]), convert_float(slm4[110]), convert_float(slm4[111])
, convert_float(slm4[112]), convert_float(slm4[113]), convert_float(slm4[114]), convert_float(slm4[115])
, convert_float(slm4[116]), convert_float(slm4[117]), convert_float(slm4[118]), convert_float(slm4[119])
, convert_float(slm4[120]), convert_float(slm4[121]), convert_float(slm4[122]), convert_float(slm4[123])
, convert_float(slm4[124]), convert_float(slm4[125]), convert_float(slm4[126]), convert_float(slm4[127])
, convert_float(slm5[  0]), convert_float(slm5[  1]), convert_float(slm5[  2]), convert_float(slm5[  3])
, convert_float(slm5[  4]), convert_float(slm5[  5]), convert_float(slm5[  6]), convert_float(slm5[  7])
, convert_float(slm5[  8]), convert_float(slm5[  9]), convert_float(slm5[ 10]), convert_float(slm5[ 11])
, convert_float(slm5[ 12]), convert_float(slm5[ 13]), convert_float(slm5[ 14]), convert_float(slm5[ 15])
, convert_float(slm5[ 16]), convert_float(slm5[ 17]), convert_float(slm5[ 18]), convert_float(slm5[ 19])
, convert_float(slm5[ 20]), convert_float(slm5[ 21]), convert_float(slm5[ 22]), convert_float(slm5[ 23])
, convert_float(slm5[ 24]), convert_float(slm5[ 25]), convert_float(slm5[ 26]), convert_float(slm5[ 27])
, convert_float(slm5[ 28]), convert_float(slm5[ 29]), convert_float(slm5[ 30]), convert_float(slm5[ 31])
, convert_float(slm5[ 32]), convert_float(slm5[ 33]), convert_float(slm5[ 34]), convert_float(slm5[ 35])
, convert_float(slm5[ 36]), convert_float(slm5[ 37]), convert_float(slm5[ 38]), convert_float(slm5[ 39])
, convert_float(slm5[ 40]), convert_float(slm5[ 41]), convert_float(slm5[ 42]), convert_float(slm5[ 43])
, convert_float(slm5[ 44]), convert_float(slm5[ 45]), convert_float(slm5[ 46]), convert_float(slm5[ 47])
, convert_float(slm5[ 48]), convert_float(slm5[ 49]), convert_float(slm5[ 50]), convert_float(slm5[ 51])
, convert_float(slm5[ 52]), convert_float(slm5[ 53]), convert_float(slm5[ 54]), convert_float(slm5[ 55])
, convert_float(slm5[ 56]), convert_float(slm5[ 57]), convert_float(slm5[ 58]), convert_float(slm5[ 59])
, convert_float(slm5[ 60]), convert_float(slm5[ 61]), convert_float(slm5[ 62]), convert_float(slm5[ 63])
, convert_float(slm5[ 64]), convert_float(slm5[ 65]), convert_float(slm5[ 66]), convert_float(slm5[ 67])
, convert_float(slm5[ 68]), convert_float(slm5[ 69]), convert_float(slm5[ 70]), convert_float(slm5[ 71])
, convert_float(slm5[ 72]), convert_float(slm5[ 73]), convert_float(slm5[ 74]), convert_float(slm5[ 75])
, convert_float(slm5[ 76]), convert_float(slm5[ 77]), convert_float(slm5[ 78]), convert_float(slm5[ 79])
, convert_float(slm5[ 80]), convert_float(slm5[ 81]), convert_float(slm5[ 82]), convert_float(slm5[ 83])
, convert_float(slm5[ 84]), convert_float(slm5[ 85]), convert_float(slm5[ 86]), convert_float(slm5[ 87])
, convert_float(slm5[ 88]), convert_float(slm5[ 89]), convert_float(slm5[ 90]), convert_float(slm5[ 91])
, convert_float(slm5[ 92]), convert_float(slm5[ 93]), convert_float(slm5[ 94]), convert_float(slm5[ 95])
, convert_float(slm5[ 96]), convert_float(slm5[ 97]), convert_float(slm5[ 98]), convert_float(slm5[ 99])
, convert_float(slm5[100]), convert_float(slm5[101]), convert_float(slm5[102]), convert_float(slm5[103])
, convert_float(slm5[104]), convert_float(slm5[105]), convert_float(slm5[106]), convert_float(slm5[107])
, convert_float(slm5[108]), convert_float(slm5[109]), convert_float(slm5[110]), convert_float(slm5[111])
, convert_float(slm5[112]), convert_float(slm5[113]), convert_float(slm5[114]), convert_float(slm5[115])
, convert_float(slm5[116]), convert_float(slm5[117]), convert_float(slm5[118]), convert_float(slm5[119])
, convert_float(slm5[120]), convert_float(slm5[121]), convert_float(slm5[122]), convert_float(slm5[123])
, convert_float(slm5[124]), convert_float(slm5[125]), convert_float(slm5[126]), convert_float(slm5[127])
, convert_float(slm6[  0]), convert_float(slm6[  1]), convert_float(slm6[  2]), convert_float(slm6[  3])
, convert_float(slm6[  4]), convert_float(slm6[  5]), convert_float(slm6[  6]), convert_float(slm6[  7])
, convert_float(slm6[  8]), convert_float(slm6[  9]), convert_float(slm6[ 10]), convert_float(slm6[ 11])
, convert_float(slm6[ 12]), convert_float(slm6[ 13]), convert_float(slm6[ 14]), convert_float(slm6[ 15])
, convert_float(slm6[ 16]), convert_float(slm6[ 17]), convert_float(slm6[ 18]), convert_float(slm6[ 19])
, convert_float(slm6[ 20]), convert_float(slm6[ 21]), convert_float(slm6[ 22]), convert_float(slm6[ 23])
, convert_float(slm6[ 24]), convert_float(slm6[ 25]), convert_float(slm6[ 26]), convert_float(slm6[ 27])
, convert_float(slm6[ 28]), convert_float(slm6[ 29]), convert_float(slm6[ 30]), convert_float(slm6[ 31])
, convert_float(slm6[ 32]), convert_float(slm6[ 33]), convert_float(slm6[ 34]), convert_float(slm6[ 35])
, convert_float(slm6[ 36]), convert_float(slm6[ 37]), convert_float(slm6[ 38]), convert_float(slm6[ 39])
, convert_float(slm6[ 40]), convert_float(slm6[ 41]), convert_float(slm6[ 42]), convert_float(slm6[ 43])
, convert_float(slm6[ 44]), convert_float(slm6[ 45]), convert_float(slm6[ 46]), convert_float(slm6[ 47])
, convert_float(slm6[ 48]), convert_float(slm6[ 49]), convert_float(slm6[ 50]), convert_float(slm6[ 51])
, convert_float(slm6[ 52]), convert_float(slm6[ 53]), convert_float(slm6[ 54]), convert_float(slm6[ 55])
, convert_float(slm6[ 56]), convert_float(slm6[ 57]), convert_float(slm6[ 58]), convert_float(slm6[ 59])
, convert_float(slm6[ 60]), convert_float(slm6[ 61]), convert_float(slm6[ 62]), convert_float(slm6[ 63])
, convert_float(slm6[ 64]), convert_float(slm6[ 65]), convert_float(slm6[ 66]), convert_float(slm6[ 67])
, convert_float(slm6[ 68]), convert_float(slm6[ 69]), convert_float(slm6[ 70]), convert_float(slm6[ 71])
, convert_float(slm6[ 72]), convert_float(slm6[ 73]), convert_float(slm6[ 74]), convert_float(slm6[ 75])
, convert_float(slm6[ 76]), convert_float(slm6[ 77]), convert_float(slm6[ 78]), convert_float(slm6[ 79])
, convert_float(slm6[ 80]), convert_float(slm6[ 81]), convert_float(slm6[ 82]), convert_float(slm6[ 83])
, convert_float(slm6[ 84]), convert_float(slm6[ 85]), convert_float(slm6[ 86]), convert_float(slm6[ 87])
, convert_float(slm6[ 88]), convert_float(slm6[ 89]), convert_float(slm6[ 90]), convert_float(slm6[ 91])
, convert_float(slm6[ 92]), convert_float(slm6[ 93]), convert_float(slm6[ 94]), convert_float(slm6[ 95])
, convert_float(slm6[ 96]), convert_float(slm6[ 97]), convert_float(slm6[ 98]), convert_float(slm6[ 99])
, convert_float(slm6[100]), convert_float(slm6[101]), convert_float(slm6[102]), convert_float(slm6[103])
, convert_float(slm6[104]), convert_float(slm6[105]), convert_float(slm6[106]), convert_float(slm6[107])
, convert_float(slm6[108]), convert_float(slm6[109]), convert_float(slm6[110]), convert_float(slm6[111])
, convert_float(slm6[112]), convert_float(slm6[113]), convert_float(slm6[114]), convert_float(slm6[115])
, convert_float(slm6[116]), convert_float(slm6[117]), convert_float(slm6[118]), convert_float(slm6[119])
, convert_float(slm6[120]), convert_float(slm6[121]), convert_float(slm6[122]), convert_float(slm6[123])
, convert_float(slm6[124]), convert_float(slm6[125]), convert_float(slm6[126]), convert_float(slm6[127])
, convert_float(slm7[  0]), convert_float(slm7[  1]), convert_float(slm7[  2]), convert_float(slm7[  3])
, convert_float(slm7[  4]), convert_float(slm7[  5]), convert_float(slm7[  6]), convert_float(slm7[  7])
, convert_float(slm7[  8]), convert_float(slm7[  9]), convert_float(slm7[ 10]), convert_float(slm7[ 11])
, convert_float(slm7[ 12]), convert_float(slm7[ 13]), convert_float(slm7[ 14]), convert_float(slm7[ 15])
, convert_float(slm7[ 16]), convert_float(slm7[ 17]), convert_float(slm7[ 18]), convert_float(slm7[ 19])
, convert_float(slm7[ 20]), convert_float(slm7[ 21]), convert_float(slm7[ 22]), convert_float(slm7[ 23])
, convert_float(slm7[ 24]), convert_float(slm7[ 25]), convert_float(slm7[ 26]), convert_float(slm7[ 27])
, convert_float(slm7[ 28]), convert_float(slm7[ 29]), convert_float(slm7[ 30]), convert_float(slm7[ 31])
, convert_float(slm7[ 32]), convert_float(slm7[ 33]), convert_float(slm7[ 34]), convert_float(slm7[ 35])
, convert_float(slm7[ 36]), convert_float(slm7[ 37]), convert_float(slm7[ 38]), convert_float(slm7[ 39])
, convert_float(slm7[ 40]), convert_float(slm7[ 41]), convert_float(slm7[ 42]), convert_float(slm7[ 43])
, convert_float(slm7[ 44]), convert_float(slm7[ 45]), convert_float(slm7[ 46]), convert_float(slm7[ 47])
, convert_float(slm7[ 48]), convert_float(slm7[ 49]), convert_float(slm7[ 50]), convert_float(slm7[ 51])
, convert_float(slm7[ 52]), convert_float(slm7[ 53]), convert_float(slm7[ 54]), convert_float(slm7[ 55])
, convert_float(slm7[ 56]), convert_float(slm7[ 57]), convert_float(slm7[ 58]), convert_float(slm7[ 59])
, convert_float(slm7[ 60]), convert_float(slm7[ 61]), convert_float(slm7[ 62]), convert_float(slm7[ 63])
, convert_float(slm7[ 64]), convert_float(slm7[ 65]), convert_float(slm7[ 66]), convert_float(slm7[ 67])
, convert_float(slm7[ 68]), convert_float(slm7[ 69]), convert_float(slm7[ 70]), convert_float(slm7[ 71])
, convert_float(slm7[ 72]), convert_float(slm7[ 73]), convert_float(slm7[ 74]), convert_float(slm7[ 75])
, convert_float(slm7[ 76]), convert_float(slm7[ 77]), convert_float(slm7[ 78]), convert_float(slm7[ 79])
, convert_float(slm7[ 80]), convert_float(slm7[ 81]), convert_float(slm7[ 82]), convert_float(slm7[ 83])
, convert_float(slm7[ 84]), convert_float(slm7[ 85]), convert_float(slm7[ 86]), convert_float(slm7[ 87])
, convert_float(slm7[ 88]), convert_float(slm7[ 89]), convert_float(slm7[ 90]), convert_float(slm7[ 91])
, convert_float(slm7[ 92]), convert_float(slm7[ 93]), convert_float(slm7[ 94]), convert_float(slm7[ 95])
, convert_float(slm7[ 96]), convert_float(slm7[ 97]), convert_float(slm7[ 98]), convert_float(slm7[ 99])
, convert_float(slm7[100]), convert_float(slm7[101]), convert_float(slm7[102]), convert_float(slm7[103])
, convert_float(slm7[104]), convert_float(slm7[105]), convert_float(slm7[106]), convert_float(slm7[107])
, convert_float(slm7[108]), convert_float(slm7[109]), convert_float(slm7[110]), convert_float(slm7[111])
, convert_float(slm7[112]), convert_float(slm7[113]), convert_float(slm7[114]), convert_float(slm7[115])
, convert_float(slm7[116]), convert_float(slm7[117]), convert_float(slm7[118]), convert_float(slm7[119])
, convert_float(slm7[120]), convert_float(slm7[121]), convert_float(slm7[122]), convert_float(slm7[123])
, convert_float(slm7[124]), convert_float(slm7[125]), convert_float(slm7[126]), convert_float(slm7[127])
);
}
//*/

    // DST SLM CORRECT

    s_tile_type_t S_tile_t;
    s_tile_type_dst S_tile_dst;

    // TODO: del me
    tile_fill(S_tile_t, 0.0f);

    tile_load(&S_tile_t, (local float *)slm, ugemm_wgu_wg_tile_n,
            ugemm_wgu_wg_tile_m, ugemm_wgu_wg_tile_m, sg_i0_wgu, sg_j0_wgu);

    tile_copy_reblock(S_tile_t, &S_tile_dst);

//*
printf("DST THREAD grp = (%lu %lu %lu), GLB = (%2lu %lu %lu) "
"[sg_i0_wgu = %2u, sg_j0_wgu = %2u, wg_i0 = %2u, wg_j0 = %2u]:\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
, get_group_id(0), get_group_id(1), get_group_id(2)
, get_global_id(0), get_global_id(1), get_global_id(2)
, sg_i0_wgu, sg_j0_wgu, wg_i0, wg_j0
, S_tile_t.x[ 0].s0, S_tile_t.x[ 0].s1
, S_tile_t.x[ 1].s0, S_tile_t.x[ 1].s1
, S_tile_t.x[ 2].s0, S_tile_t.x[ 2].s1
, S_tile_t.x[ 3].s0, S_tile_t.x[ 3].s1
, S_tile_t.x[ 4].s0, S_tile_t.x[ 4].s1
, S_tile_t.x[ 5].s0, S_tile_t.x[ 5].s1
, S_tile_t.x[ 6].s0, S_tile_t.x[ 6].s1
, S_tile_t.x[ 7].s0, S_tile_t.x[ 7].s1
, S_tile_t.x[ 8].s0, S_tile_t.x[ 8].s1
, S_tile_t.x[ 9].s0, S_tile_t.x[ 9].s1
, S_tile_t.x[10].s0, S_tile_t.x[10].s1
, S_tile_t.x[11].s0, S_tile_t.x[11].s1
, S_tile_t.x[12].s0, S_tile_t.x[12].s1
, S_tile_t.x[13].s0, S_tile_t.x[13].s1
, S_tile_t.x[14].s0, S_tile_t.x[14].s1
, S_tile_t.x[15].s0, S_tile_t.x[15].s1
);
//*/

/*
printf("DST THREAD (%lu %lu %lu) [sg_i0_wgu = %u, sg_j0_wgu = %u]:\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n"
, get_global_id(0), get_global_id(1), get_global_id(2), sg_i0_wgu, sg_j0_wgu
, convert_float(S_tile_dst.x[ 0].s0), convert_float(S_tile_dst.x[ 1].s0)
, convert_float(S_tile_dst.x[ 2].s0), convert_float(S_tile_dst.x[ 3].s0)
, convert_float(S_tile_dst.x[ 4].s0), convert_float(S_tile_dst.x[ 5].s0)
, convert_float(S_tile_dst.x[ 6].s0), convert_float(S_tile_dst.x[ 7].s0)
, convert_float(S_tile_dst.x[ 8].s0), convert_float(S_tile_dst.x[ 9].s0)
, convert_float(S_tile_dst.x[10].s0), convert_float(S_tile_dst.x[11].s0)
, convert_float(S_tile_dst.x[12].s0), convert_float(S_tile_dst.x[13].s0)
, convert_float(S_tile_dst.x[14].s0), convert_float(S_tile_dst.x[15].s0)
, convert_float(S_tile_dst.x[16].s0), convert_float(S_tile_dst.x[17].s0)
, convert_float(S_tile_dst.x[18].s0), convert_float(S_tile_dst.x[19].s0)
, convert_float(S_tile_dst.x[20].s0), convert_float(S_tile_dst.x[21].s0)
, convert_float(S_tile_dst.x[22].s0), convert_float(S_tile_dst.x[23].s0)
, convert_float(S_tile_dst.x[24].s0), convert_float(S_tile_dst.x[25].s0)
, convert_float(S_tile_dst.x[26].s0), convert_float(S_tile_dst.x[27].s0)
, convert_float(S_tile_dst.x[28].s0), convert_float(S_tile_dst.x[29].s0)
, convert_float(S_tile_dst.x[30].s0), convert_float(S_tile_dst.x[31].s0)
);
//*/

    tile_store(S_tile_dst, tmp_reduce_mem + k_offset, OC, MB, ldi,
            wg_j0 + sg_i0_wgu, wg_i0 + sg_j0_wgu);
}
