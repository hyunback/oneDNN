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

#define BR ugemm_wgu_c_type_block0
#define BC ugemm_wgu_c_type_block1
#define NBR ugemm_wgu_c_type_nblock0
#define NBC ugemm_wgu_c_type_nblock1

DECLARE_2D_TILE(s_tile_type_dst, VEC_TYPE1, SUBGROUP_SIZE, BR, BC, NBR, NBC)
DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, BR, BC, NBR, NBC,
        s_tile_type_dst, SUBGROUP_SIZE, BR, BC, NBR, NBC, CONVERT_DATA_T)

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

    uint wg_i0 = get_group_id(2) * ugemm_wgu_wg_tile_m; // OC
    uint wg_j0 = get_group_id(0) * ugemm_wgu_wg_tile_n; // MB

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    uint sg_i_wgu = sg_ij % ugemm_wgu_sg_per_wg_m;
    uint sg_j_wgu = sg_ij / ugemm_wgu_sg_per_wg_m;

    s_tile_type S_WU_tile = ugemm_wgu(W_up, W_UP_S1, src, SRC_S0,
            OC, MB, IC, wg_i0, wg_j0, 0, sg_i_wgu, sg_j_wgu
#if WTS_UP_SCALES == QUANTIZE_2D
            ,
            wts_up_scales
#endif
#if WTS_UP_ZERO_POINTS
            ,
            wts_up_zp
#endif
#if (WTS_UP_SCALES == QUANTIZE_2D) || WTS_UP_ZERO_POINTS
            ,
            OC
#endif
    );
#if WTS_UP_SCALES == QUANTIZE_COMMON
#define wu_scale_op(x) ((x) * wu_scale)
    float wu_scale = convert_float(*wts_up_scales);
    tile_elementwise(S_WU_tile, wu_scale_op);
#endif

#ifndef UGEMM_UP_ONLY
    s_tile_type S_WG_tile = ugemm_wgu(W_gate, W_GATE_S1, src, SRC_S0,
            OC, MB, IC, wg_i0, wg_j0, 0, sg_i_wgu, sg_j_wgu
#if WTS_GATE_SCALES == QUANTIZE_2D
            ,
            wts_gate_scales
#endif
#if WTS_GATE_ZERO_POINTS
            ,
            wts_gate_zp
#endif
#if (WTS_GATE_SCALES == QUANTIZE_2D) || WTS_GATE_ZERO_POINTS
            ,
            OC
#endif
    );
#if WTS_GATE_SCALES == QUANTIZE_COMMON
#define wg_scale_op(x) unary_activation((x) * wg_scale)
    float wg_scale = convert_float(*wts_gate_scales);
    tile_elementwise(S_WG_tile, wg_scale_op);
#else
    tile_elementwise(S_WG_tile, unary_activation);
#endif
    tile_binary(S_WU_tile, S_WG_tile, binary_mul);
#endif // UGEMM_UP_ONLY

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_m;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_n;

    s_tile_type_dst S_tile_dst;
    tile_copy_reblock(S_WU_tile, &S_tile_dst);
    tile_store(S_tile_dst, tmp_reduce_mem, OC, MB, INTER_S0,
            wg_i0 + sg_i0_wgu, wg_j0 + sg_j0_wgu);
}
