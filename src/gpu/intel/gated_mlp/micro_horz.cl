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
#elif defined(SRC_DT_BF16)
#define VEC_TYPE1 ushort
#else
#error "Data type not supported for VEC_TYPE1"
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

DECLARE_2D_TILE(s_tile_type_dst, VEC_TYPE1, SUBGROUP_SIZE,
        ugemm_wgu_c_type_block0, ugemm_wgu_c_type_block1,
        ugemm_wgu_c_type_nblock0, ugemm_wgu_c_type_nblock1)

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE,
        ugemm_wgu_c_type_block0, ugemm_wgu_c_type_block1,
        ugemm_wgu_c_type_nblock0, ugemm_wgu_c_type_nblock1,
        s_tile_type_dst, SUBGROUP_SIZE,
        ugemm_wgu_c_type_block0, ugemm_wgu_c_type_block1,
        ugemm_wgu_c_type_nblock0, ugemm_wgu_c_type_nblock1,
        CONVERT_DATA_T)

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

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    uint wg_i0 = get_group_id(0) * ugemm_wgu_wg_tile_m; // MB
    uint wg_j0 = get_group_id(2) * ugemm_wgu_wg_tile_n; // OC

#if WTS_GATE_SCALES == QUANTIZE_COMMON
    float wg_scale = convert_float(*wts_gate_scales);
#endif
#if WTS_UP_SCALES == QUANTIZE_COMMON
    float wu_scale = convert_float(*wts_up_scales);
#endif

    uint sg_i_wgu = sg_ij % ugemm_wgu_sg_per_wg_m;
    uint sg_j_wgu = sg_ij / ugemm_wgu_sg_per_wg_m;
/*
#define WGU_slm_size (ugemm_wgu_wg_tile_m * ugemm_wgu_wg_tile_n)

    local char slm[MAX(WGU_slm_size * sizeof(float),
            WGU_slm_size * 2 * sizeof(SRC_DATA_T) + ugemm_wgu_slm_size)];
    local SRC_DATA_T *wg_slm[2] = {(local SRC_DATA_T *)slm,
            (local SRC_DATA_T *)(slm + WGU_slm_size * sizeof(SRC_DATA_T))};
    local char *ugemm_gu_slm = slm + WGU_slm_size * 2 * sizeof(SRC_DATA_T);

#ifndef UGEMM_UP_ONLY
    s_tile_type S_WG_tile;
    tile_fill(S_WG_tile, 0.0f);
#endif
    s_tile_type S_WU_tile;
    tile_fill(S_WU_tile, 0.0f);

    src_to_slm(src, wg_slm[0], SRC_S0, IC, MB, 0, wg_i0, sg_ij);
    int kL = (DIV_UP(IC, ugemm_wgu_wg_tile_m) - 1) * ugemm_wgu_wg_tile_m;
    _Pragma("unroll") for (int k0 = 0; k0 < kL; k0 += ugemm_wgu_wg_tile_m) {
        int curr = (k0 / ugemm_wgu_wg_tile_m) % 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        src_to_slm(src, wg_slm[1 - curr], SRC_S0, IC, MB,
                k0 + ugemm_wgu_wg_tile_m, wg_i0, sg_ij);
#ifndef UGEMM_UP_ONLY
        do_gemm(wg_slm[curr], W_gate, &S_WG_tile, wts_gate_scales, wts_gate_zp,
                ugemm_gu_slm, W_GATE_S1, OC, k0, wg_j0, sg_i_wgu, sg_j_wgu);
#endif
        do_gemm(wg_slm[curr], W_up, &S_WU_tile, wts_up_scales, wts_up_zp,
                ugemm_gu_slm, W_UP_S1, OC, k0, wg_j0, sg_i_wgu, sg_j_wgu);
    }
    int last = (kL / ugemm_wgu_wg_tile_m) % 2;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifndef UGEMM_UP_ONLY
    do_gemm(wg_slm[last], W_gate, &S_WG_tile, wts_gate_scales, wts_gate_zp,
            ugemm_gu_slm, W_GATE_S1, OC, kL, wg_j0, sg_i_wgu, sg_j_wgu);
#endif
    do_gemm(wg_slm[last], W_up, &S_WU_tile, wts_up_scales, wts_up_zp,
            ugemm_gu_slm, W_UP_S1, OC, kL, wg_j0, sg_i_wgu, sg_j_wgu);

#if WTS_UP_SCALES == QUANTIZE_COMMON
#define wu_scale_op(x) ((x) * wu_scale)
    tile_elementwise(S_WU_tile, wu_scale_op);
#endif
#ifndef UGEMM_UP_ONLY
#if WTS_GATE_SCALES == QUANTIZE_COMMON
#define wg_scale_op(x) unary_activation((x) * wg_scale)
    tile_elementwise(S_WG_tile, wg_scale_op);
#else
    tile_elementwise(S_WG_tile, unary_activation);
#endif
    tile_binary(S_WU_tile, S_WG_tile, binary_mul);
#endif // UGEMM_UP_ONLY

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_n;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_m;
    size_t k_offset = get_group_id(1) / sg_per_wg * OC * MB;

    barrier(CLK_LOCAL_MEM_FENCE);
    tile_store_t(S_WU_tile, (local float *)slm, MB, OC, ugemm_wgu_wg_tile_m,
            sg_i0_wgu, sg_j0_wgu);
    barrier(CLK_LOCAL_MEM_FENCE);

    s_tile_type_t S_tile_t;
    s_tile_type_dst S_tile_dst;
    tile_load(&S_tile_t, (local float *)slm, ugemm_wgu_wg_tile_m,
            ugemm_wgu_wg_tile_n, ugemm_wgu_wg_tile_m, sg_j0_wgu, sg_i0_wgu);
    tile_copy_reblock(S_tile_t, &S_tile_dst);
    tile_store(S_tile_dst, tmp_reduce_mem + k_offset, OC, MB, INTER_S0,
            wg_j0 + sg_j0_wgu, wg_i0 + sg_i0_wgu);
//*/

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_m;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_n;

    s_tile_type S_WU_tile = ugemm_wgu(src, SRC_S0, W_up, W_UP_S1,
            MB, OC, IC, wg_i0, wg_j0, 0, sg_i_wgu, sg_j_wgu);
#if WTS_UP_SCALES == QUANTIZE_COMMON
#define wu_scale_op(x) ((x) * wu_scale)
    tile_elementwise(S_WU_tile, wu_scale_op);
#endif

#ifndef UGEMM_UP_ONLY
    s_tile_type S_WG_tile = ugemm_wgu(src, SRC_S0, W_gate, W_GATE_S1,
            MB, OC, IC, wg_i0, wg_j0, 0, sg_i_wgu, sg_j_wgu);
#if WTS_GATE_SCALES == QUANTIZE_COMMON
#define wg_scale_op(x) unary_activation((x) * wg_scale)
    tile_elementwise(S_WG_tile, wg_scale_op);
#else
    tile_elementwise(S_WG_tile, unary_activation);
#endif
    tile_binary(S_WU_tile, S_WG_tile, binary_mul);
#endif // UGEMM_UP_ONLY

    s_tile_type_dst S_tile_dst;
    tile_copy_reblock(S_WU_tile, &S_tile_dst);
    tile_store(S_tile_dst, tmp_reduce_mem, OC, MB, INTER_S0,
            wg_j0 + sg_j0_wgu, wg_i0 + sg_i0_wgu);
}
