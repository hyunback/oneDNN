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

#include <dnnl_test_common.hpp>
#include <gtest/gtest.h>

#include "test_utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <type_traits>

#include <cassert>
#include <random>

#define DNNL_ARG_WEIGHTS_GATE DNNL_ARG_WEIGHTS_0
#define DNNL_ARG_WEIGHTS_UP DNNL_ARG_WEIGHTS_1
#define DNNL_ARG_WEIGHTS_DOWN DNNL_ARG_WEIGHTS_2

#include "common/gated_mlp_iface.hpp"

// uncomment to dump cpu memory buffers
//#define ENABLE_PRINT_MEM

// uncomment to disable everything except Up
//#define ENABLE_UP_ONLY

namespace dnnl {
namespace impl {

static bool verbose = true; // enable for debug
static const int min_runs = 4;

using tag = memory::format_tag;
using mdt = memory::data_type;

/// Gated MLP (gmlp) internal primitive.
struct gmlp_t : public dnnl::primitive {
    /// Primitive descriptor for a gmlp primitive.
    struct pd_t : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        pd_t() = default;

        /// Constructs a primitive descriptor for a gmlp primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a gmlp primitive.
        pd_t(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::undef) {}

        pd_t(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &W_gate_desc, const memory::desc &W_up_desc,
                const memory::desc &W_down_desc,
                const memory::desc &output_desc, const alg_kind_t &activation,
                const primitive_attr &attr = default_attr()) {
            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_gated_mlp_primitive_desc_create(&pd,
                    aengine.get(), src_desc.get(), W_gate_desc.get(),
                    W_up_desc.get(), W_down_desc.get(), output_desc.get(),
                    activation, attr.get());

            dnnl::error::wrap_c_api(status,
                    "could not create a primitive descriptor for a gmlp "
                    "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    gmlp_t() = default;

    /// Constructs a gmlp primitive.
    /// @param pd Primitive descriptor for a gmlp primitive.
    gmlp_t(const pd_t &pd) : primitive(pd) {}
};

struct mlp_dims_t {
    dim_t mb;
    dim_t ic;
    dim_t oc;

    int gateup_group_size;
    int down_group_size;

    quantize_type qtype;
    dnnl_alg_kind_t activation;

    memory::data_type src_dt;
    memory::data_type dst_dt;

    memory::data_type wgu_wt;
    memory::data_type wgu_s_dt;
    memory::data_type wgu_zp_dt;

    memory::data_type wd_wt;
    memory::data_type wd_s_dt;
    memory::data_type wd_zp_dt;
};

struct gmlp_tensors_t {
    memory m_x, m_w_gate, m_w_up, m_w_down;
    memory m_w_gate_scales, m_w_up_scales, m_w_down_scales;
    memory m_w_gate_zp, m_w_up_zp, m_w_down_zp;
    memory m_fc_gate, m_fc_up, m_fc_down;
    memory m_fc_retn_t;

    memory::dims wgu_groups, wd_groups;
};

std::ostream &operator<<(std::ostream &ss, const dnnl_alg_kind_t &act) {
    switch (act) {
        case dnnl_alg_kind_t::dnnl_eltwise_gelu_erf:
            ss << "_activation_gelu_erf";
            break;
        case dnnl_alg_kind_t::dnnl_eltwise_gelu_tanh:
            ss << "_activation_gelu_tanh";
            break;
        case dnnl_alg_kind_t::dnnl_eltwise_swish:
            ss << "_activation_swish";
            break;
        default: ss << "_activation_unknown"; break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const mlp_dims_t &p) {
    const bool has_quant = p.qtype != quantize_type::no_quantization;

    ss << "mb_" << p.mb;
    ss << "_ic_" << p.ic;
    ss << "_oc_" << p.oc;

    ss << ((has_quant) ? "_quant_" : "_noquant_");
    if (has_quant) {
        ss << "_qtype_" << p.qtype;
        ss << "_gu_group_size_" << p.gateup_group_size;
        ss << "_gd_group_size_" << p.down_group_size;
    }

    ss << "_src_dt_" << dnnl_dt2str(memory::convert_to_c(p.src_dt));
    ss << "_dst_dt_" << dnnl_dt2str(memory::convert_to_c(p.dst_dt));

    ss << "_wgu_wt_" << dnnl_dt2str(memory::convert_to_c(p.wgu_wt));
    if (has_quant) {
        ss << "_wgu_sdt_" << dnnl_dt2str(memory::convert_to_c(p.wgu_s_dt));
        ss << "_wgu_zpdt_" << dnnl_dt2str(memory::convert_to_c(p.wgu_zp_dt));
    }

    ss << "_wd_wt_" << dnnl_dt2str(memory::convert_to_c(p.wd_wt));
    if (has_quant) {
        ss << "_wd_sdt_" << dnnl_dt2str(memory::convert_to_c(p.wd_s_dt));
        ss << "_wd_zpdt_" << dnnl_dt2str(memory::convert_to_c(p.wd_zp_dt));
    }

    ss << p.activation;
    return ss;
}

std::string PrintToString(const ::testing::TestParamInfo<mlp_dims_t> &info) {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
}

template <typename T>
bool maybe_read_from_file(
        std::vector<T> &v, size_t exp_size, const std::string &suffix) {
    auto maybe_path = ::getenv("BUFFER_READ_PREFIX");
    if (!maybe_path) return false;
    std::string path(maybe_path);
    path += "." + suffix + ".bin";

    FILE *file = nullptr;
#ifdef _WIN32
    if (::fopen_s(&file, path.c_str(), "rb")) file = nullptr;
#else
    file = ::fopen(path.c_str(), "rb");
#endif
    if (!file) return false;
    ::fseek(file, 0, SEEK_END);
    size_t total = 0, size = ::ftell(file);
    if (exp_size * sizeof(v[0]) != size) {
        ::fclose(file);
        return false;
    }
    v.resize(exp_size);
    ::fseek(file, 0, SEEK_SET);
    for (size_t read = ~0; read && (total < size); total += read)
        read = ::fread((uint8_t *)v.data() + total, 1, size - total, file);
    ::fclose(file);
    return (total == size);
}

template <typename T>
bool maybe_write_to_file(std::vector<T> &v, const std::string &suffix) {
    auto maybe_path = ::getenv("BUFFER_WRITE_PREFIX");
    if (!maybe_path) return false;
    std::string path(maybe_path);
    path += "." + suffix + ".bin";

    FILE *file = nullptr;
#ifdef _WIN32
    if (::fopen_s(&file, path.c_str(), "wb")) file = nullptr;
#else
    file = ::fopen(path.c_str(), "wb");
#endif
    if (!file) return false;
    size_t total = 0, size = v.size() * sizeof(v[0]);
    for (size_t write = ~0; write && (total < size); total += write)
        write = ::fwrite((uint8_t *)v.data() + total, 1, size - total, file);
    ::fclose(file);
    return (total == size);
}

gmlp_tensors_t get_descriptors(engine &eng, stream &strm, mlp_dims_t p) {
    gmlp_tensors_t out;

    // Prepare input and output shapes to construct the swiglu graph.
    const memory::dims O_proj_sz = {p.mb, p.ic};
    const memory::dims W_gate_sz = {p.ic, p.oc};
    const memory::dims W_up_sz = {p.ic, p.oc};
    const memory::dims W_down_sz = {p.oc, p.ic};
    const memory::dims FC_gate_sz = {p.mb, p.oc};
    const memory::dims FC_up_sz = {p.mb, p.oc};
    const memory::dims FC_down_sz = {p.mb, p.ic};

    const memory::dims gateup_sz = [&]() {
        switch (p.qtype) {
            case quantize_type::no_quantization: return memory::dims {1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        W_gate_sz[0] / p.gateup_group_size, W_gate_sz[1]};
            case quantize_type::per_token:
                return memory::dims {W_gate_sz[0], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1};
            default: return memory::dims {0, 0};
        }
    }();
    const memory::dims down_sz = [&]() {
        switch (p.qtype) {
            case quantize_type::no_quantization: return memory::dims {1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        W_down_sz[0] / p.down_group_size, W_down_sz[1]};
            case quantize_type::per_token:
                return memory::dims {W_down_sz[0], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1};
            default: return memory::dims {0, 0};
        }
    }();
    auto get_gate_dt = [](mdt src, mdt dst, mdt wgu, mdt wd) {
        mdt retn = wd;
        switch (retn) {
            case mdt::f32:
            case mdt::bf16:
            case mdt::f16: break;
            case mdt::u4:
            case mdt::s4:
            case mdt::u8:
            case mdt::s8:
                retn = src;
                if (!utils::one_of(retn, mdt::bf16, mdt::f16)) retn = mdt::f16;
                break;
            default: return mdt::undef;
        }
        return retn;
    };
    auto get_up_dt = [](mdt src, mdt dst, mdt wgu, mdt wd) { return mdt::f32; };

    auto dt = memory::data_type::f16;

    auto src_dt = (p.src_dt == mdt::undef) ? dt : p.src_dt;
    auto dst_dt = (p.dst_dt == mdt::undef) ? dt : p.dst_dt;

    auto wgu_wt = (p.wgu_wt == mdt::undef) ? dt : p.wgu_wt;
    auto wgu_s_dt = (p.wgu_s_dt == mdt::undef) ? dt : p.wgu_s_dt;
    auto wgu_zp_dt = (p.wgu_zp_dt == mdt::undef) ? dt : p.wgu_zp_dt;

    auto wd_wt = (p.wd_wt == mdt::undef) ? dt : p.wd_wt;
    auto wd_s_dt = (p.wd_s_dt == mdt::undef) ? dt : p.wd_s_dt;
    auto wd_zp_dt = (p.wd_zp_dt == mdt::undef) ? dt : p.wd_zp_dt;

    auto fc_gate_dt = get_gate_dt(src_dt, dst_dt, wgu_wt, wd_wt);
    auto fc_up_dt = get_up_dt(src_dt, dst_dt, wgu_wt, wd_wt);

    auto FC_gate_md = memory::desc(FC_gate_sz, fc_gate_dt, tag::ab);
    auto FC_up_md = memory::desc(FC_up_sz, fc_up_dt, tag::ab);
    auto FC_down_md = memory::desc(FC_down_sz, dst_dt, tag::ab);

    auto FC_retn_md_t = memory::desc(FC_down_sz, dst_dt, tag::ab);

    // clang-format off
    auto x_md = memory::desc(O_proj_sz, src_dt, tag::ab);

    auto w_gate_md = memory::desc(W_gate_sz, wgu_wt, tag::ba);
    auto w_up_md   = memory::desc(W_up_sz,   wgu_wt, tag::ba);
    auto w_down_md = memory::desc(W_down_sz,  wd_wt, tag::ba);

    auto w_gate_scales_md = memory::desc(gateup_sz, wgu_s_dt, tag::ab);
    auto w_up_scales_md   = memory::desc(gateup_sz, wgu_s_dt, tag::ab);
    auto w_down_scales_md = memory::desc(down_sz,    wd_s_dt, tag::ab);

    auto w_gate_zp_md = memory::desc(gateup_sz, wgu_zp_dt, tag::ab);
    auto w_up_zp_md   = memory::desc(gateup_sz, wgu_zp_dt, tag::ab);
    auto w_down_zp_md = memory::desc(down_sz,    wd_zp_dt, tag::ab);
    // clang-format on

    // Create memory objects
    out.m_x = memory(x_md, eng);

    out.m_w_gate = memory(w_gate_md, eng);
    out.m_w_up = memory(w_up_md, eng);
    out.m_w_down = memory(w_down_md, eng);

    out.m_w_gate_scales = memory(w_gate_scales_md, eng);
    out.m_w_up_scales = memory(w_up_scales_md, eng);
    out.m_w_down_scales = memory(w_down_scales_md, eng);

    out.m_w_gate_zp = memory(w_gate_zp_md, eng);
    out.m_w_up_zp = memory(w_up_zp_md, eng);
    out.m_w_down_zp = memory(w_down_zp_md, eng);

    out.m_fc_gate = memory(FC_gate_md, eng);
    out.m_fc_up = memory(FC_up_md, eng);
    out.m_fc_down = memory(FC_down_md, eng);

    out.m_fc_retn_t = memory(FC_retn_md_t, eng);

    // Allocate user data.
    std::vector<float> x_data(product(O_proj_sz));

    std::vector<float> w_gate_data(product(W_gate_sz), 1.f);
    std::vector<float> w_up_data(product(W_up_sz), 1.f);
    std::vector<float> w_down_data(product(W_down_sz), 1.f);

    std::vector<float> w_gate_scales_data(
            product(out.m_w_gate_scales.get_desc().get_padded_dims()), 1.f);
    std::vector<float> w_up_scales_data(
            product(out.m_w_up_scales.get_desc().get_padded_dims()), 1.f);
    std::vector<float> w_down_scales_data(
            product(out.m_w_down_scales.get_desc().get_padded_dims()), 1.f);

    std::vector<int> w_gate_zp_data(
            product(out.m_w_gate_zp.get_desc().get_padded_dims()), 0);
    std::vector<int> w_up_zp_data(
            product(out.m_w_up_zp.get_desc().get_padded_dims()), 0);
    std::vector<int> w_down_zp_data(
            product(out.m_w_down_zp.get_desc().get_padded_dims()), 0);

    out.wgu_groups = {};
    out.wd_groups = {};
    switch (p.qtype) {
        case quantize_type::per_token_with_groups: {
            out.wgu_groups = {p.gateup_group_size, 1};
            out.wd_groups = {p.down_group_size, 1};
            break;
        }
        case quantize_type::per_token: {
            // TODO: add
            break;
        }
        case quantize_type::per_tensor: {
            // TODO: add
            break;
        }
        default: break;
    }

    fill_random(x_data, x_md, -.25f, .25f);

    if (p.qtype == quantize_type::no_quantization) {
        if (verbose) printf("no quant init\n");
        fill_random(w_gate_data, w_gate_md, -1.f, 1.f);
        fill_random(w_up_data, w_up_md, -1.f, 1.f);
        fill_random(w_down_data, w_down_md, -1.f, 1.f);
    } else {
        fill_random_quantized(w_gate_data, w_gate_md,
                (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));
        fill_random_quantized(
                w_up_data, w_up_md, (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));
        fill_random_quantized(
                w_down_data, w_down_md, (wd_wt == mdt::u4 || wd_wt == mdt::u8));

        fill_random_scales(w_gate_scales_data, w_gate_scales_md);
        fill_random_scales(w_up_scales_data, w_up_scales_md);
        fill_random_scales(w_down_scales_data, w_down_scales_md);

        bool wgu_zp_unsigned = (wgu_zp_dt == mdt::u4 || wgu_zp_dt == mdt::u8);
        if (verbose) {
            if (wgu_zp_unsigned)
                printf("unsigned gateup quant init\n");
            else
                printf("signed gateup quant init\n");
        }
        fill_random_quantized(w_gate_zp_data, w_gate_zp_md, wgu_zp_unsigned);
        fill_random_quantized(w_up_zp_data, w_up_zp_md, wgu_zp_unsigned);

        bool wd_zp_unsigned = (wd_zp_dt == mdt::u4 || wd_zp_dt == mdt::u8);
        if (verbose) {
            if (wd_zp_unsigned)
                printf("unsigned down quant init\n");
            else
                printf("signed down quant init\n");
        }
        fill_random_quantized(w_down_zp_data, w_down_zp_md, wd_zp_unsigned);
    }

    // Write data to tensor object's handle.
    write_to_dnnl_memory(x_data.data(), out.m_x, eng, strm);

    write_to_dnnl_memory(w_gate_data.data(), out.m_w_gate, eng, strm);
    write_to_dnnl_memory(w_up_data.data(), out.m_w_up, eng, strm);
    write_to_dnnl_memory(w_down_data.data(), out.m_w_down, eng, strm);

    if (p.qtype != quantize_type::no_quantization) {
        write_to_dnnl_memory(w_gate_zp_data.data(), out.m_w_gate_zp, eng, strm);
        write_to_dnnl_memory(w_up_zp_data.data(), out.m_w_up_zp, eng, strm);
        write_to_dnnl_memory(w_down_zp_data.data(), out.m_w_down_zp, eng, strm);

        write_to_dnnl_memory(
                w_gate_scales_data.data(), out.m_w_gate_scales, eng, strm);
        write_to_dnnl_memory(
                w_up_scales_data.data(), out.m_w_up_scales, eng, strm);
        write_to_dnnl_memory(
                w_down_scales_data.data(), out.m_w_down_scales, eng, strm);
    }
    if (verbose)
        printf("memory data types?? %d %d %d\n",
                int(out.m_w_gate_scales.get_desc().get_data_type()),
                int(out.m_w_up_scales.get_desc().get_data_type()),
                int(out.m_w_down_scales.get_desc().get_data_type()));

    return out;
}

void bench_gated_mlp_primitives(std::vector<float> &res, double &avg_time,
        gmlp_tensors_t &t, engine &eng, stream &strm, const mlp_dims_t &p,
        double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);

    // extract memory objects
    auto m_O_proj = t.m_x;
    auto m_FC_gate = t.m_fc_gate;
    auto m_FC_up = t.m_fc_up;
    auto m_FC_down = t.m_fc_down;

    auto O_proj_md = t.m_x.get_desc();
    auto FC_gate_md = t.m_fc_gate.get_desc();
    auto FC_up_md = t.m_fc_up.get_desc();
    auto FC_down_md = t.m_fc_down.get_desc();

    auto m_W_gate = t.m_w_gate;
    auto m_W_gate_scales = t.m_w_gate_scales;
    auto m_W_gate_zp = t.m_w_gate_zp;
    auto m_W_up = t.m_w_up;
    auto m_W_up_scales = t.m_w_up_scales;
    auto m_W_up_zp = t.m_w_up_zp;
    auto m_W_down = t.m_w_down;
    auto m_W_down_scales = t.m_w_down_scales;
    auto m_W_down_zp = t.m_w_down_zp;

    auto m_W_gate_md = t.m_w_gate.get_desc();
    auto m_W_gate_scales_md = t.m_w_gate_scales.get_desc();
    auto m_W_gate_zp_md = t.m_w_gate_zp.get_desc();
    auto m_W_up_md = t.m_w_up.get_desc();
    auto m_W_up_scales_md = t.m_w_up_scales.get_desc();
    auto m_W_up_zp_md = t.m_w_up_zp.get_desc();
    auto m_W_down_md = t.m_w_down.get_desc();
    auto m_W_down_scales_md = t.m_w_down_scales.get_desc();
    auto m_W_down_zp_md = t.m_w_down_zp.get_desc();

    auto m_FC_retn_t = t.m_fc_retn_t;

    auto gen_default_attr = [&](int idx) {
        primitive_attr attr;
        if (p.qtype == quantize_type::no_quantization) return attr;
        int mask = (p.qtype == quantize_type::per_token_with_groups)
                ? (1 << 0) + (1 << 1)
                : 0;
        switch (idx) {
            default: break;
            case DNNL_ARG_WEIGHTS_UP:
                attr.set_scales(DNNL_ARG_WEIGHTS, mask, t.wgu_groups,
                        m_W_up_scales_md.get_data_type());
                attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, t.wgu_groups,
                        m_W_up_zp_md.get_data_type());
                break;
            case DNNL_ARG_WEIGHTS_GATE:
                attr.set_scales(DNNL_ARG_WEIGHTS, mask, t.wgu_groups,
                        m_W_gate_scales_md.get_data_type());
                attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, t.wgu_groups,
                        m_W_gate_zp_md.get_data_type());
                break;
            case DNNL_ARG_WEIGHTS_DOWN:
                attr.set_scales(DNNL_ARG_WEIGHTS, mask, t.wd_groups,
                        m_W_down_scales_md.get_data_type());
                attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, t.wd_groups,
                        m_W_down_zp_md.get_data_type());
                break;
        }
        attr.set_fpmath_mode(
                static_cast<enum fpmath_mode>(fpmath_mode::any), true);
        return attr;
    };

    // fc_up
    auto bmm0_pd = matmul::primitive_desc(eng, O_proj_md, m_W_up_md, FC_up_md,
            gen_default_attr(DNNL_ARG_WEIGHTS_UP));
    auto bmm0_prim = matmul(bmm0_pd);

#ifndef ENABLE_UP_ONLY
    // fc_gate -> swish -> mul
    auto bmm1_attr = gen_default_attr(DNNL_ARG_WEIGHTS_GATE);
    post_ops bmm1_po;
    if (p.activation == dnnl_eltwise_swish) {
        bmm1_po.append_eltwise(algorithm::eltwise_swish, 1.f, 1.f);
    } else if (p.activation == dnnl_eltwise_gelu_erf) {
        bmm1_po.append_eltwise(algorithm::eltwise_gelu_erf, 0.f, 0.f);
    } else if (p.activation == dnnl_eltwise_gelu_tanh) {
        bmm1_po.append_eltwise(algorithm::eltwise_gelu_tanh, 0.f, 0.f);
    }
    bmm1_po.append_binary(algorithm::binary_mul, m_FC_up.get_desc());
    bmm1_attr.set_post_ops(bmm1_po);

    auto bmm1_pd = matmul::primitive_desc(
            eng, O_proj_md, m_W_gate_md, FC_gate_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    auto bmm2_pd = matmul::primitive_desc(eng, FC_gate_md, m_W_down_md,
            FC_down_md, gen_default_attr(DNNL_ARG_WEIGHTS_DOWN));
    auto bmm2_prim = matmul(bmm2_pd);
#endif

    const auto loop = [&](bool print = false) {
#ifdef ENABLE_PRINT_MEM
#define PRINT_MEM(mem) \
    if (print) { print_mem(mem, #mem " "); }
#else
#define PRINT_MEM(mem)
#endif
        if (p.qtype == quantize_type::no_quantization) {
            bmm0_prim.execute(strm,
                    {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_up},
                            {DNNL_ARG_DST, m_FC_up}});
#ifndef ENABLE_UP_ONLY
            bmm1_prim.execute(strm,
                    {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                            {DNNL_ARG_DST, m_FC_gate},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                    m_FC_up}});

            bmm2_prim.execute(strm,
                    {{DNNL_ARG_SRC, m_FC_gate}, {DNNL_ARG_WEIGHTS, m_W_down},
                            {DNNL_ARG_DST, m_FC_down}});
#endif
            PRINT_MEM(m_O_proj)
            PRINT_MEM(m_W_up)
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(m_W_gate)
            PRINT_MEM(m_W_down)
#endif
            PRINT_MEM(m_FC_up)
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(m_FC_gate)
            PRINT_MEM(m_FC_down)
#endif
        } else {
            bmm0_prim.execute(strm,
                    {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_up},
                            {DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES,
                                    m_W_up_scales},
                            {DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_up_zp},
                            {DNNL_ARG_DST, m_FC_up}});
#ifndef ENABLE_UP_ONLY
            bmm1_prim.execute(strm,
                    {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                            {DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES,
                                    m_W_gate_scales},
                            {DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_gate_zp},
                            {DNNL_ARG_DST, m_FC_gate},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                    m_FC_up}});

            bmm2_prim.execute(strm,
                    {{DNNL_ARG_SRC, m_FC_gate}, {DNNL_ARG_WEIGHTS, m_W_down},
                            {DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES,
                                    m_W_down_scales},
                            {DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_down_zp},
                            {DNNL_ARG_DST, m_FC_down}});
#endif
            PRINT_MEM(m_O_proj)
            PRINT_MEM(m_W_up)
            PRINT_MEM(m_W_up_scales)
            PRINT_MEM(m_W_up_zp)
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(m_W_gate)
            PRINT_MEM(m_W_gate_scales)
            PRINT_MEM(m_W_gate_zp)
            PRINT_MEM(m_W_down)
            PRINT_MEM(m_W_down_scales)
            PRINT_MEM(m_W_down_zp)
#endif
            PRINT_MEM(m_FC_up)
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(m_FC_gate)
            PRINT_MEM(m_FC_down)
#endif
        }
#undef PRINT_MEM
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop(true);

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        loop();
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    avg_time = duration.count() / runs;
    if (verbose) {
        std::cout << "primitive runs: " << runs + 1 << "; ";
        std::cout << "avg_time: " << avg_time << " ms" << std::endl;
    }

#ifndef ENABLE_UP_ONLY
    res.resize(product(m_FC_down.get_desc().get_dims()));
    move_data(eng, strm, res, m_FC_down, false);
#else
    res.resize(product(m_FC_up.get_desc().get_dims()));
    move_data(eng, strm, res, m_FC_up, false);
#endif
}

void bench_gated_mlp_internal(std::vector<float> &res, double &avg_time,
        gmlp_tensors_t &t, engine &eng, stream strm, const mlp_dims_t &p,
        double time_limit = 0.) {
    if (verbose) printf("eng?\n");
    const bool quick_test = (time_limit == 0.);

    // Create memory objects
    auto m_O_proj = t.m_x;
    auto m_FC_gate = t.m_fc_gate;
    auto m_FC_up = t.m_fc_up;
    auto m_FC_down = t.m_fc_down;

    auto O_proj_md = t.m_x.get_desc();
    auto FC_gate_md = t.m_fc_gate.get_desc();
    auto FC_up_md = t.m_fc_up.get_desc();
    auto FC_down_md = t.m_fc_down.get_desc();

    // quantization memory
    auto m_W_gate = t.m_w_gate;
    auto m_W_gate_scales = t.m_w_gate_scales;
    auto m_W_gate_zp = t.m_w_gate_zp;
    auto m_W_up = t.m_w_up;
    auto m_W_up_scales = t.m_w_up_scales;
    auto m_W_up_zp = t.m_w_up_zp;
    auto m_W_down = t.m_w_down;
    auto m_W_down_scales = t.m_w_down_scales;
    auto m_W_down_zp = t.m_w_down_zp;

    auto m_W_gate_md = t.m_w_gate.get_desc();
    auto m_W_gate_scales_md = t.m_w_gate_scales.get_desc();
    auto m_W_gate_zp_md = t.m_w_gate_zp.get_desc();
    auto m_W_up_md = t.m_w_up.get_desc();
    auto m_W_up_scales_md = t.m_w_up_scales.get_desc();
    auto m_W_up_zp_md = t.m_w_up_zp.get_desc();
    auto m_W_down_md = t.m_w_down.get_desc();
    auto m_W_down_scales_md = t.m_w_down_scales.get_desc();
    auto m_W_down_zp_md = t.m_w_down_zp.get_desc();

#ifdef ENABLE_UP_ONLY
    const memory::dims FC_retn_sz_t = {p.mb, p.oc};
#else
    const memory::dims FC_retn_sz_t = {p.mb, p.ic};
#endif
    auto FC_retn_md_t
            = memory::desc(FC_retn_sz_t, FC_down_md.get_data_type(), tag::ab);
    auto m_FC_gate_t = memory(FC_retn_md_t, eng);

    primitive_attr attr;
    int mask = 0;
    switch (p.qtype) {
        case quantize_type::per_token_with_groups: mask = (1 << 0) + (1 << 1);
        case quantize_type::per_tensor: {
            attr.set_fpmath_mode(
                    static_cast<enum fpmath_mode>(fpmath_mode::any), true);
            // wts_gate scale+zp
            attr.set_scales(DNNL_ARG_WEIGHTS_GATE, mask, t.wgu_groups,
                    m_W_gate_scales_md.get_data_type());
            attr.set_zero_points(DNNL_ARG_WEIGHTS_GATE, mask, t.wgu_groups,
                    m_W_gate_zp_md.get_data_type());
            // wts_up scale+zp
            attr.set_scales(DNNL_ARG_WEIGHTS_UP, mask, t.wgu_groups,
                    m_W_up_scales_md.get_data_type());
            attr.set_zero_points(DNNL_ARG_WEIGHTS_UP, mask, t.wgu_groups,
                    m_W_up_zp_md.get_data_type());
            // wts_down scale+zp
            attr.set_scales(DNNL_ARG_WEIGHTS_DOWN, mask, t.wd_groups,
                    m_W_down_scales_md.get_data_type());
            attr.set_zero_points(DNNL_ARG_WEIGHTS_DOWN, mask, t.wd_groups,
                    m_W_down_zp_md.get_data_type());
            break;
        }
        default: break;
    }

    auto gmlp_pd = [&]() {
        dnnl_alg_kind_t activation;
        activation = p.activation;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_swish;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_gelu_erf;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_gelu_tanh;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_exp; // should fail
        return gmlp_t::pd_t(eng, O_proj_md, m_W_gate_md, m_W_up_md, m_W_down_md,
                FC_retn_md_t, activation, attr);
    }();

    auto prim_fused_internal = gmlp_t(gmlp_pd);

    const auto loop = [&](bool print = false) {
#ifdef ENABLE_PRINT_MEM
#define PRINT_MEM(mem) \
    if (print) { print_mem(mem, #mem " "); }
#else
#define PRINT_MEM(mem)
#endif
        if (p.qtype == quantize_type::no_quantization) {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC, m_O_proj},
                            {DNNL_ARG_WEIGHTS_GATE, m_W_gate},
                            {DNNL_ARG_WEIGHTS_UP, m_W_up},
                            {DNNL_ARG_WEIGHTS_DOWN, m_W_down},
                            {DNNL_ARG_DST, m_FC_gate_t}});
            if (print) strm.wait();
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(m_O_proj)
            PRINT_MEM(m_W_up)
            PRINT_MEM(m_W_gate)
            PRINT_MEM(m_W_down)
#endif
            PRINT_MEM(m_FC_gate_t)
        } else {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC, m_O_proj},
                            {DNNL_ARG_WEIGHTS_GATE, m_W_gate},
                            {DNNL_ARG_WEIGHTS_UP, m_W_up},
                            {DNNL_ARG_WEIGHTS_DOWN, m_W_down},
                            {DNNL_ARG_DST, m_FC_gate_t},
                            {DNNL_ARG_WEIGHTS_GATE | DNNL_ARG_ATTR_SCALES,
                                    m_W_gate_scales},
                            {DNNL_ARG_WEIGHTS_GATE | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_gate_zp},
                            {DNNL_ARG_WEIGHTS_UP | DNNL_ARG_ATTR_SCALES,
                                    m_W_up_scales},
                            {DNNL_ARG_WEIGHTS_UP | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_up_zp},
                            {DNNL_ARG_WEIGHTS_DOWN | DNNL_ARG_ATTR_SCALES,
                                    m_W_down_scales},
                            {DNNL_ARG_WEIGHTS_DOWN | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_down_zp}});
            if (print) strm.wait();
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(m_O_proj)
            PRINT_MEM(m_W_up)
            PRINT_MEM(m_W_gate)
            PRINT_MEM(m_W_down)
            PRINT_MEM(m_W_up_scales)
            PRINT_MEM(m_W_up_zp)
            PRINT_MEM(m_W_gate_scales)
            PRINT_MEM(m_W_gate_zp)
            PRINT_MEM(m_W_down_scales)
            PRINT_MEM(m_W_down_zp)
#else
            PRINT_MEM(m_W_up)
            PRINT_MEM(m_W_up_scales)
            PRINT_MEM(m_W_up_zp)
#endif
            PRINT_MEM(m_FC_gate_t)
        }
#undef PRINT_MEM
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop(true);

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        loop();
        //print_mem(m_W_gate, "-ilolloopafnternal");
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    avg_time = duration.count() / runs;
    if (verbose) {
        std::cout << "internal gmlp primitive runs: " << runs + 1 << "; ";
        std::cout << "avg_time: " << avg_time << " ms" << std::endl;
    }

    res.resize(product(m_FC_gate_t.get_desc().get_dims()));
    move_data(eng, strm, res, m_FC_gate_t, false);
}

enum class api_kind { primitive, graph, internal_hack };

template <typename T>
void bench(std::vector<T> &res, double &avg_time, gmlp_tensors_t &t,
        api_kind api, engine &eng, stream &strm, const mlp_dims_t &p,
        double time_limit = 0.) {

    try {
        if (api == api_kind::primitive) {
            bench_gated_mlp_primitives(
                    res, avg_time, t, eng, strm, p, time_limit);
            strm.wait();
        } else if (api == api_kind::graph) {
            // TODO: add graph
        } else {
            bench_gated_mlp_internal(
                    res, avg_time, t, eng, strm, p, time_limit);
            strm.wait();
        }
    } catch (error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented) {
            std::cout << "unsupported mlp" << std::endl;
        } else
            throw;
    }
}

class mlp_test_t : public ::testing::TestWithParam<mlp_dims_t> {
public:
    void SetUp() override {
#ifdef DNNL_SYCL_CUDA
        GTEST_SKIP() << "GMLP primitive tests do not support CUDA";
#endif
#ifdef DNNL_SYCL_HIP
        GTEST_SKIP() << "GMLP primitive tests do not support HIP";
#endif
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "GMLP tests require gpus.");
        eng = engine(engine::kind::gpu, 0);
        strm = stream(eng);

        char *maybe_test = nullptr;
        try {
            auto get_type = [](const std::string &tmp) {
                switch (std::stoi(tmp)) {
                    default: return mdt::undef;
                    case 16: return mdt::f16;
                    case -16: return mdt::bf16;
                    case 8: return mdt::u8;
                    case -8: return mdt::s8;
                    case 4: return mdt::u4;
                    case -4: return mdt::s4;
                }
            };
            maybe_test = ::getenv("GMLP_TEST");
            if (!maybe_test) throw std::exception();
            p.gateup_group_size = p.down_group_size = 1;
            p.qtype = quantize_type::no_quantization;
            p.activation = dnnl_eltwise_swish;
            p.src_dt = p.dst_dt = mdt::f16;
            p.wgu_wt = p.wgu_s_dt = p.wgu_zp_dt = mdt::f16;
            p.wd_wt = p.wd_s_dt = p.wd_zp_dt = mdt::f16;

            std::stringstream ss(maybe_test);
            std::string tmp;
            if (!getline(ss, tmp, ' ')) throw std::exception();
            p.mb = std::stoi(tmp);
            if (!getline(ss, tmp, ' ')) throw std::exception();
            p.ic = std::stoi(tmp);
            if (!getline(ss, tmp, ' ')) throw std::exception();
            p.oc = std::stoi(tmp);
            if (getline(ss, tmp, ' ')) p.src_dt = p.dst_dt = get_type(tmp);
            if (getline(ss, tmp, ' ')) p.wgu_wt = p.wd_wt = get_type(tmp);
            if (getline(ss, tmp, ' '))
                p.qtype = (std::stoi(tmp) > 0)
                        ? quantize_type::per_token_with_groups
                        : quantize_type::no_quantization;
            if (getline(ss, tmp, ' '))
                p.gateup_group_size = p.down_group_size = std::stoi(tmp);
            if (getline(ss, tmp, ' ')) p.wgu_s_dt = p.wd_s_dt = get_type(tmp);
            if (getline(ss, tmp, ' ')) p.wgu_zp_dt = p.wd_zp_dt = get_type(tmp);

            printf("GMLP_TEST: (%ld x %ld x %ld, %s x %s, q = %d, gs = %d, "
                   "s = %s, zp = %s)\n",
                    p.mb, p.ic, p.oc,
                    dnnl_dt2str(memory::convert_to_c(p.src_dt)),
                    dnnl_dt2str(memory::convert_to_c(p.wgu_wt)),
                    p.qtype != quantize_type::no_quantization,
                    p.gateup_group_size,
                    dnnl_dt2str(memory::convert_to_c(p.wgu_s_dt)),
                    dnnl_dt2str(memory::convert_to_c(p.wgu_zp_dt)));
        } catch (...) { maybe_test = nullptr; }
        if (!maybe_test) p = GetParam();
        t = get_descriptors(eng, strm, p);
    }

protected:
    mlp_dims_t p;
    engine eng;
    stream strm;
    gmlp_tensors_t t;
};

TEST_P(mlp_test_t, compare) {
    auto tensors = t;
    auto params = p;

    std::vector<float> resph, resih;
    double avg_time_int, avg_time_prim;

    if (verbose) printf("PRIMITIVE\n");
    bench(resph, avg_time_prim, tensors, api_kind::primitive, eng, strm, params,
            2000.0 /*ms*/);

    if (verbose) printf("INTERNAL\n");
    bench(resih, avg_time_int, tensors, api_kind::internal_hack, eng, strm,
            params, 2000.0 /*ms*/);

    if (resih.empty()) {
        if (verbose)
            printf("[WARNING] Empty output: internal kernel failure!\n");
        EXPECT_TRUE(false);
    }
    int n_mismatches = 0, n_matches = 0;
    if (verbose) printf("resih.size() %zu\n", resih.size());
    float max_diff = 0.0f, max_val = -INFINITY, max_gold = -INFINITY;
    for (int i = 0; i < int(resih.size()); ++i) {
        float abs_diff = std::abs(resih[i] - resph[i]);
        float rel_diff = std::abs((resih[i] - resph[i]) / resih[i]);
        if (abs_diff > 1e-4 && rel_diff > 5e-2) {

            if (isfinite(rel_diff) && (abs_diff) > max_diff) {
                max_diff = abs_diff;
                max_val = resih[i];
                max_gold = resph[i];
            }

            n_mismatches++;
            if (verbose && (n_mismatches < 10))
                printf("mismatch @ %d, %f != %f\n", i, float(resih[i]),
                        float(resph[i])); //TODO: improve
        } else {
            if (std::abs(float16_t(resih[i])) > 5e-3) {
                n_matches++;
                if (verbose && (n_matches < 10))
                    printf("vs @ %d, %f == %f\n", i, float(resih[i]),
                            float(resph[i])); //TODO: improve
            }
        }
    }
    int total_size = int(resph.size());
    int threshold = total_size * 0.0006;

    if (verbose) {
        printf("total mismatches: %d, allowed: %d\n", n_mismatches, threshold);
        printf("avg time internal vs primitive: %f vs %f, w/speedup of %f\n",
                avg_time_int, avg_time_prim, avg_time_prim / avg_time_int);
        if (n_mismatches > 0) {
            std::cout << "max diff: " << max_diff << ":  " << max_val
                      << " != " << max_gold << std::endl;
        }
    }
    ASSERT_LE(n_mismatches, threshold) << "out of: " << total_size;
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(VEC, mlp_test_t, ::testing::Values(
    // Test case #12: LLaMA 3.1 8B FP16 no-quant
    mlp_dims_t{1024, 4096, 14336, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    ,
    // Test case: LLaMA 3.1 8B INT4-like shape for real-model debugging.
    // Note: this parameterized harness shares gate/up quantization settings.
    mlp_dims_t{1024, 4096, 14336, 64, 64,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    ,
/*
    // no quantization
    mlp_dims_t {64, 64, 64, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 1

    mlp_dims_t{1024, 3584, 18944, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 2
    mlp_dims_t{1024, 3584, 4864, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 3
    mlp_dims_t{1024, 3584, 14336, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 4
    mlp_dims_t{1024, 3584, 27392, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 5
    mlp_dims_t{1024, 896, 18944, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 6
    mlp_dims_t{1024, 896, 4864, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 7
    mlp_dims_t{1024, 896, 14336, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 8
    mlp_dims_t{1024, 896, 27392, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 9
    mlp_dims_t{1024, 4096, 18944, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 10
    mlp_dims_t{1024, 4096, 4864, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 11
    mlp_dims_t{1024, 4096, 14336, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 12
    mlp_dims_t{1024, 4096, 27392, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 13

    // B = 1024, quantized w=u8
    mlp_dims_t{32, 32, 32, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 14
    mlp_dims_t{1024, 3584, 18944, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 15
    mlp_dims_t{1024, 896, 4864, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 16
    mlp_dims_t{1024, 4096, 14336, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 17
    mlp_dims_t{1024, 4096, 27392, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 18

    mlp_dims_t{1024, 3584, 18944, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 19
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 20
    mlp_dims_t{1024, 4096, 14336, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 21
    mlp_dims_t{1024, 4096, 27392, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u8, mdt::f16, mdt::u8,
            mdt::u8, mdt::f16, mdt::u8}
    , // ^-- 22

    // B = 1024, quantized w=s8
    mlp_dims_t{32, 32, 32, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 23
    mlp_dims_t{1024, 3584, 18944, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 24
    mlp_dims_t{1024, 896, 4864, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 25
    mlp_dims_t{1024, 4096, 14336, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 26
    mlp_dims_t{1024, 4096, 27392, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 27

    mlp_dims_t{1024, 3584, 18944, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 28
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 29
    mlp_dims_t{1024, 4096, 14336, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 30
    mlp_dims_t{1024, 4096, 27392, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s8, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 31

    // B = 1024, quantized w=u4
    mlp_dims_t{32, 128, 32, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    , // ^-- 32
    mlp_dims_t{1024, 3584, 18944, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    , // ^-- 33
    mlp_dims_t{1024, 896, 4864, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    , // ^-- 34
    mlp_dims_t{1024, 4096, 14336, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    , // ^-- 35
    mlp_dims_t{1024, 4096, 27392, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    , // ^-- 36
//*/
    mlp_dims_t{1024, 4096, 27392, 128, 128,
            //quantize_type::no_quantization, dnnl_eltwise_swish,
            //mdt::f16, mdt::f16,
            //mdt::f16, mdt::f16, mdt::f16,
            //mdt::f16, mdt::f16, mdt::f16}
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    ,
    // Test case: Qwen 2.5 7B INT4 (pure u4, group_size=128)
    mlp_dims_t{1024, 3584, 18944, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
/*
    , // ^-- 37
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    , // ^-- 38
    mlp_dims_t{1024, 4096, 14336, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    , // ^-- 39
    mlp_dims_t{1024, 4096, 27392, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
/*
    , // ^-- 40

    // additional 4bit quant
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s4, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 41
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 42
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::s4, mdt::f16, mdt::u8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 43

    // activatioins
    mlp_dims_t{32, 32, 32, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_gelu_tanh,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 44
    mlp_dims_t{32, 32, 32, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_gelu_erf,
            mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16,
            mdt::f16, mdt::f16, mdt::f16}
    , // ^-- 45
    mlp_dims_t{32, 128, 32, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_gelu_tanh,
            mdt::f16, mdt::f16,
            mdt::s4, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 46
    mlp_dims_t{32, 128, 32, 16, 16,
            quantize_type::per_token_with_groups, dnnl_eltwise_gelu_erf,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::s8,
            mdt::s8, mdt::f16, mdt::s8}
    , // ^-- 47
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::bf16, mdt::bf16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    ,
    mlp_dims_t{1024, 896, 4864, 128, 128,
            quantize_type::per_token_with_groups, dnnl_eltwise_swish,
            mdt::f16, mdt::f16,
            mdt::u4, mdt::f16, mdt::u8,
            mdt::u4, mdt::f16, mdt::u8}
    ,
    mlp_dims_t{1024, 896, 4864, 1, 1,
            quantize_type::no_quantization, dnnl_eltwise_swish,
            mdt::f32, mdt::f32,
            mdt::f32, mdt::f16, mdt::u8,
            mdt::f32, mdt::f16, mdt::u8}
//*/
), &PrintToString);
// clang-format on

// === Dump-based test: loads real model data from binary files ===
static bool load_binary_to_memory(const std::string &path, dnnl::memory &mem) {
    FILE *f = ::fopen(path.c_str(), "rb");
    if (!f) return false;
    ::fseek(f, 0, SEEK_END);
    size_t file_size = ::ftell(f);
    size_t mem_size = mem.get_desc().get_size();
    if (file_size != mem_size) {
        ::fprintf(stderr, "Size mismatch %s: file=%zu mem=%zu\n",
                path.c_str(), file_size, mem_size);
        ::fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(file_size);
    ::fseek(f, 0, SEEK_SET);
    size_t rd = ::fread(buf.data(), 1, file_size, f);
    ::fclose(f);
    if (rd != file_size) return false;
    auto *ptr = static_cast<uint8_t *>(mem.map_data());
    memcpy(ptr, buf.data(), file_size);
    mem.unmap_data(ptr);
    ::fprintf(stderr, "Loaded %s (%zu bytes)\n", path.c_str(), file_size);
    return true;
}

TEST(mlp_dump_test, compare_from_dump_int4) {
    const char *dump_dir_env = std::getenv("GMLP_DUMP_DIR");
    if (!dump_dir_env) GTEST_SKIP() << "Set GMLP_DUMP_DIR=/tmp/gmlp_dump";
    std::string dir(dump_dir_env);

    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Requires GPU");
    // Use device index from GMLP_GPU_DEV env var (default: 0).
    // e2e OpenVINO uses GPU.1 which may map to a different physical device.
    // On BMG-03: dev 0 = Arc B580 (BMG), dev 1 = integrated GPU.
    const char *dev_env = std::getenv("GMLP_GPU_DEV");
    int dev_idx = dev_env ? std::atoi(dev_env) : 0;
    engine eng(engine::kind::gpu, dev_idx);
    stream strm(eng);

    // LLaMA 3.1 8B INT4 layer 0 dimensions
    const memory::dim mb = 1024, ic = 4096, oc = 14336;
    const int gate_gs = 64, down_gs = 64;
    const int gate_ngroups = ic / gate_gs;    // 64
    const int down_ngroups = oc / down_gs;    // 224 (down IC=oc=14336, 14336/64=224)
    // Note: down_ngroups = oc/down_gs is wrong. down weight IC = 14336 = oc in gmlp terms.
    // Actually: down weight OpenVINO shape [OC_down=4096, IC_down=14336]
    // In OpenVINO code: ifm = weight.get_dim(1) = 14336, ngroups = scale.get_dim(0) = 224
    // group_size = 14336 / 224 = 64
    const int actual_down_ngroups = 224; // 14336 / 64

    // --- Memory descriptors ---
    auto src_md = memory::desc({mb, ic}, mdt::f16, tag::ab);
    auto dst_md = memory::desc({mb, ic}, mdt::f16, tag::ab);

    // Weights: logical [IC, OC] with ba tag = physical [OC, IC]
    auto wg_md = memory::desc({ic, oc}, mdt::u4, tag::ba);    // gate u4
    auto wu_md = memory::desc({ic, oc}, mdt::u8, tag::ba);    // up u8
    // Down: matmul A[mb,oc] × B[oc,ic] → C[mb,ic], B logical [oc,ic] ba = phys [ic,oc]=[4096,14336]
    auto wd_md = memory::desc({oc, ic}, mdt::u4, tag::ba);

    // Scales (already transposed in dump: [ngroups, OC])
    auto sg_md = memory::desc({gate_ngroups, oc}, mdt::f16, tag::ab);       // [64, 14336]
    auto su_md = memory::desc({1, oc}, mdt::f16, tag::ab);                  // [1, 14336]
    auto sd_md = memory::desc({actual_down_ngroups, ic}, mdt::f16, tag::ab); // [224, 4096]

    // Zero points
    auto zg_md = memory::desc({gate_ngroups, oc}, mdt::u8, tag::ab);
    auto zu_md = memory::desc({1, oc}, mdt::u8, tag::ab);
    auto zd_md = memory::desc({actual_down_ngroups, ic}, mdt::u8, tag::ab);

    // Intermediate buffers
    auto fc_gate_md = memory::desc({mb, oc}, mdt::f16, tag::ab);
    auto fc_up_md = memory::desc({mb, oc}, mdt::f32, tag::ab);

    // --- Create memories ---
    memory m_src(src_md, eng), m_dst_prim(dst_md, eng), m_dst_int(dst_md, eng);
    memory m_wg(wg_md, eng), m_wu(wu_md, eng), m_wd(wd_md, eng);
    memory m_sg(sg_md, eng), m_su(su_md, eng), m_sd(sd_md, eng);
    memory m_zg(zg_md, eng), m_zu(zu_md, eng), m_zd(zd_md, eng);
    memory m_fc_gate(fc_gate_md, eng), m_fc_up(fc_up_md, eng);

    // --- Load from dump files ---
    ASSERT_TRUE(load_binary_to_memory(dir + "/src.bin", m_src));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_gate.bin", m_wg));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_up.bin", m_wu));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_down.bin", m_wd));
    ASSERT_TRUE(load_binary_to_memory(dir + "/scale_gate.bin", m_sg));
    ASSERT_TRUE(load_binary_to_memory(dir + "/scale_up.bin", m_su));
    ASSERT_TRUE(load_binary_to_memory(dir + "/scale_down.bin", m_sd));
    ASSERT_TRUE(load_binary_to_memory(dir + "/zp_gate.bin", m_zg));
    ASSERT_TRUE(load_binary_to_memory(dir + "/zp_up.bin", m_zu));
    ASSERT_TRUE(load_binary_to_memory(dir + "/zp_down.bin", m_zd));

    fprintf(stderr, "All dump files loaded from %s\n", dir.c_str());

    // ========== Primitive reference path (3 separate matmuls) ==========
    fprintf(stderr, "[PROGRESS] Creating up matmul primitive...\n");
    // Up matmul: src[mb,ic] × w_up[ic,oc] → fc_up[mb,oc] (f32), per-channel scale
    primitive_attr up_attr;
    up_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    up_attr.set_scales(DNNL_ARG_WEIGHTS, 2, {}, mdt::f16);
    up_attr.set_zero_points(DNNL_ARG_WEIGHTS, 2, {}, mdt::u8);
    auto up_pd = matmul::primitive_desc(eng, src_md, wu_md, fc_up_md, up_attr);
    auto up_prim = matmul(up_pd);
    fprintf(stderr, "[PROGRESS] Up matmul created.\n");

    fprintf(stderr, "[PROGRESS] Creating gate matmul primitive...\n");
    // Gate matmul: src[mb,ic] × w_gate[ic,oc] → fc_gate[mb,oc] (f16) + swish + mul(fc_up)
    primitive_attr gate_attr;
    gate_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    gate_attr.set_scales(
            DNNL_ARG_WEIGHTS, 3, {gate_gs, 1}, mdt::f16);
    gate_attr.set_zero_points(
            DNNL_ARG_WEIGHTS, 3, {gate_gs, 1}, mdt::u8);
    post_ops gate_po;
    gate_po.append_eltwise(algorithm::eltwise_swish, 1.f, 1.f);
    gate_po.append_binary(algorithm::binary_mul, fc_up_md);
    gate_attr.set_post_ops(gate_po);
    auto gate_pd = matmul::primitive_desc(
            eng, src_md, wg_md, fc_gate_md, gate_attr);
    auto gate_prim = matmul(gate_pd);
    fprintf(stderr, "[PROGRESS] Gate matmul created.\n");

    fprintf(stderr, "[PROGRESS] Creating down matmul primitive...\n");
    // Down matmul: fc_gate[mb,oc] × w_down[oc,ic] → dst[mb,ic] (f16)
    primitive_attr down_attr;
    down_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    down_attr.set_scales(
            DNNL_ARG_WEIGHTS, 3, {down_gs, 1}, mdt::f16);
    down_attr.set_zero_points(
            DNNL_ARG_WEIGHTS, 3, {down_gs, 1}, mdt::u8);
    auto down_pd = matmul::primitive_desc(
            eng, fc_gate_md, wd_md, dst_md, down_attr);
    auto down_prim = matmul(down_pd);
    fprintf(stderr, "[PROGRESS] Down matmul created.\n");

    // Execute primitive path
    // Skip reference path if GMLP_SKIP_REF=1 to test if OOR is caused
    // by memory pressure from reference matmul primitives.
    const bool skip_ref = std::getenv("GMLP_SKIP_REF") != nullptr;
    if (!skip_ref) {
    fprintf(stderr, "[PROGRESS] Executing reference path (3 matmuls)...\n");
    up_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS, m_wu},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_su},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_zu},
                    {DNNL_ARG_DST, m_fc_up}});
    gate_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS, m_wg},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_sg},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_zg},
                    {DNNL_ARG_DST, m_fc_gate},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                            m_fc_up}});
    down_prim.execute(strm,
            {{DNNL_ARG_SRC, m_fc_gate}, {DNNL_ARG_WEIGHTS, m_wd},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_sd},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_zd},
                    {DNNL_ARG_DST, m_dst_prim}});
    strm.wait();
    fprintf(stderr, "[PROGRESS] Reference path done.\n");
    } else {
        fprintf(stderr, "[PROGRESS] Skipping reference path (GMLP_SKIP_REF=1).\n");
    }

    // ========== Internal gated_mlp path ==========
    fprintf(stderr, "[PROGRESS] Creating gated_mlp primitive...\n");
    primitive_attr gmlp_attr;
    gmlp_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    // Use scratchpad:user mode to match OpenVINO e2e behavior.
    // Without this, micro_horz_t hits CL_OUT_OF_RESOURCES (-5) on INT4,
    // while e2e (which uses scratchpad:user) executes without OOR.
    gmlp_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    // Gate: grouped
    gmlp_attr.set_scales(
            DNNL_ARG_WEIGHTS_GATE, 3, {gate_gs, 1}, mdt::f16);
    gmlp_attr.set_zero_points(
            DNNL_ARG_WEIGHTS_GATE, 3, {gate_gs, 1}, mdt::u8);
    // Up: per-channel
    gmlp_attr.set_scales(DNNL_ARG_WEIGHTS_UP, 2, {}, mdt::f16);
    gmlp_attr.set_zero_points(DNNL_ARG_WEIGHTS_UP, 2, {}, mdt::u8);
    // Down: grouped
    gmlp_attr.set_scales(
            DNNL_ARG_WEIGHTS_DOWN, 3, {down_gs, 1}, mdt::f16);
    gmlp_attr.set_zero_points(
            DNNL_ARG_WEIGHTS_DOWN, 3, {down_gs, 1}, mdt::u8);

    auto gmlp_pd = gmlp_t::pd_t(eng, src_md, wg_md, wu_md, wd_md, dst_md,
            dnnl_eltwise_swish, gmlp_attr);
    auto gmlp_prim = gmlp_t(gmlp_pd);

    // Allocate user-managed scratchpad
    auto scratchpad_md = gmlp_pd.scratchpad_desc();
    memory scratchpad_mem(scratchpad_md, eng);
    fprintf(stderr, "[PROGRESS] gated_mlp primitive created (scratchpad: %zu bytes). Executing...\n",
            scratchpad_md.get_size());

    gmlp_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS_GATE, m_wg},
                    {DNNL_ARG_WEIGHTS_UP, m_wu},
                    {DNNL_ARG_WEIGHTS_DOWN, m_wd},
                    {DNNL_ARG_DST, m_dst_int},
                    {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_GATE, m_sg},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_GATE, m_zg},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_UP, m_su},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_UP, m_zu},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_DOWN, m_sd},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_DOWN,
                            m_zd}});
    strm.wait();

    // ========== Compare results ==========
    const size_t n_elems = mb * ic;
    std::vector<float> res_prim(n_elems), res_int(n_elems);
    move_data(eng, strm, res_prim, m_dst_prim, false);
    move_data(eng, strm, res_int, m_dst_int, false);

    int n_mismatches = 0;
    float max_diff = 0.0f, max_val = 0.f, max_gold = 0.f;
    for (size_t i = 0; i < n_elems; i++) {
        float abs_diff = std::abs(res_int[i] - res_prim[i]);
        float rel_diff = (res_prim[i] != 0)
                ? abs_diff / std::abs(res_prim[i])
                : abs_diff;
        if (abs_diff > 1e-4 && rel_diff > 5e-2) {
            if (n_mismatches < 10)
                printf("mismatch @ %zu: internal=%f prim=%f\n", i,
                        res_int[i], res_prim[i]);
            if (abs_diff > max_diff) {
                max_diff = abs_diff;
                max_val = res_int[i];
                max_gold = res_prim[i];
            }
            n_mismatches++;
        }
    }
    int threshold = int(n_elems) * 0.0006;
    printf("total mismatches: %d / %zu (threshold: %d), max_diff: %f (%f vs %f)\n",
            n_mismatches, n_elems, threshold, max_diff, max_val, max_gold);
    ASSERT_LE(n_mismatches, threshold);
}

// Same descriptor as compare_from_dump_int4 (up=u8 per-channel, gate/down=u4 grouped)
// but with random data instead of real model weights.
// This isolates whether CL_OUT_OF_RESOURCES is caused by the descriptor config
// (u8 up + per-channel scales) or by specific data values from the dump.
TEST(mlp_dump_test, random_data_int4_u8up) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Requires GPU");
    engine eng(engine::kind::gpu, 0);
    stream strm(eng);

    const memory::dim mb = 1024, ic = 4096, oc = 14336;
    const int gate_gs = 64, down_gs = 64;
    const int gate_ngroups = ic / gate_gs;
    const int actual_down_ngroups = 224; // 14336 / 64

    // Same descriptors as dump test: up=u8 per-channel, gate/down=u4 grouped
    auto src_md = memory::desc({mb, ic}, mdt::f16, tag::ab);
    auto dst_md = memory::desc({mb, ic}, mdt::f16, tag::ab);
    auto wg_md = memory::desc({ic, oc}, mdt::u4, tag::ba);
    auto wu_md = memory::desc({ic, oc}, mdt::u8, tag::ba);  // u8, NOT u4
    auto wd_md = memory::desc({oc, ic}, mdt::u4, tag::ba);
    auto sg_md = memory::desc({gate_ngroups, oc}, mdt::f16, tag::ab);
    auto su_md = memory::desc({1, oc}, mdt::f16, tag::ab);   // per-channel
    auto sd_md = memory::desc({actual_down_ngroups, ic}, mdt::f16, tag::ab);
    auto zg_md = memory::desc({gate_ngroups, oc}, mdt::u8, tag::ab);
    auto zu_md = memory::desc({1, oc}, mdt::u8, tag::ab);
    auto zd_md = memory::desc({actual_down_ngroups, ic}, mdt::u8, tag::ab);

    // Create memories (GPU-allocated, zero-initialized by default)
    memory m_src(src_md, eng), m_dst(dst_md, eng);
    memory m_wg(wg_md, eng), m_wu(wu_md, eng), m_wd(wd_md, eng);
    memory m_sg(sg_md, eng), m_su(su_md, eng), m_sd(sd_md, eng);
    memory m_zg(zg_md, eng), m_zu(zu_md, eng), m_zd(zd_md, eng);

    // Fill with small random values via host vectors
    auto fill_rand = [&](memory &mem, float lo, float hi) {
        size_t sz = mem.get_desc().get_size();
        std::vector<uint8_t> buf(sz);
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (auto &b : buf) b = dist(rng);
        void *ptr = mem.map_data();
        memcpy(ptr, buf.data(), sz);
        mem.unmap_data(ptr);
    };
    fill_rand(m_src, -1, 1);
    fill_rand(m_wg, 0, 1); fill_rand(m_wu, 0, 1); fill_rand(m_wd, 0, 1);
    fill_rand(m_sg, 0, 1); fill_rand(m_su, 0, 1); fill_rand(m_sd, 0, 1);
    fill_rand(m_zg, 0, 1); fill_rand(m_zu, 0, 1); fill_rand(m_zd, 0, 1);

    // Create gated_mlp primitive with same attrs as dump test
    primitive_attr gmlp_attr;
    gmlp_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    gmlp_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    gmlp_attr.set_scales(DNNL_ARG_WEIGHTS_GATE, 3, {gate_gs, 1}, mdt::f16);
    gmlp_attr.set_zero_points(DNNL_ARG_WEIGHTS_GATE, 3, {gate_gs, 1}, mdt::u8);
    gmlp_attr.set_scales(DNNL_ARG_WEIGHTS_UP, 2, {}, mdt::f16);
    gmlp_attr.set_zero_points(DNNL_ARG_WEIGHTS_UP, 2, {}, mdt::u8);
    gmlp_attr.set_scales(DNNL_ARG_WEIGHTS_DOWN, 3, {down_gs, 1}, mdt::f16);
    gmlp_attr.set_zero_points(DNNL_ARG_WEIGHTS_DOWN, 3, {down_gs, 1}, mdt::u8);

    auto gmlp_pd = gmlp_t::pd_t(eng, src_md, wg_md, wu_md, wd_md, dst_md,
            dnnl_eltwise_swish, gmlp_attr);
    auto gmlp_prim = gmlp_t(gmlp_pd);

    auto scratchpad_md = gmlp_pd.scratchpad_desc();
    memory scratchpad_mem(scratchpad_md, eng);
    fprintf(stderr, "[PROGRESS] random_data_int4_u8up: executing (scratchpad: %zu bytes)...\n",
            scratchpad_md.get_size());

    gmlp_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS_GATE, m_wg},
                    {DNNL_ARG_WEIGHTS_UP, m_wu},
                    {DNNL_ARG_WEIGHTS_DOWN, m_wd},
                    {DNNL_ARG_DST, m_dst},
                    {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_GATE, m_sg},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_GATE, m_zg},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_UP, m_su},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_UP, m_zu},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_DOWN, m_sd},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_DOWN, m_zd}});
    strm.wait();
    fprintf(stderr, "[PROGRESS] random_data_int4_u8up: done.\n");
    // If we get here without CL_OUT_OF_RESOURCES, the descriptor is fine
    // and the OOR in the dump test is data-dependent.
}

TEST(mlp_dump_test, compare_from_dump_fp16) {
    const char *dump_dir_env = std::getenv("GMLP_DUMP_DIR_FP16");
    if (!dump_dir_env)
        GTEST_SKIP() << "Set GMLP_DUMP_DIR_FP16=/tmp/gmlp_fp16_dump_bin";
    std::string dir(dump_dir_env);

    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Requires GPU");
    engine eng(engine::kind::gpu, 0);
    stream strm(eng);

    // LLaMA 3.1 8B FP16 layer dimensions
    const memory::dim mb = 1024, ic = 4096, oc = 14336;

    auto src_md = memory::desc({mb, ic}, mdt::f16, tag::ab);
    auto dst_md = memory::desc({mb, ic}, mdt::f16, tag::ab);

    // Weights: logical [IC, OC] with ba tag = physical [OC, IC]
    auto wg_md = memory::desc({ic, oc}, mdt::f16, tag::ba);
    auto wu_md = memory::desc({ic, oc}, mdt::f16, tag::ba);
    auto wd_md = memory::desc({oc, ic}, mdt::f16, tag::ba);

    // Intermediate buffers
    auto fc_gate_md = memory::desc({mb, oc}, mdt::f16, tag::ab);
    auto fc_up_md = memory::desc({mb, oc}, mdt::f32, tag::ab);

    memory m_src(src_md, eng), m_dst_prim(dst_md, eng), m_dst_int(dst_md, eng);
    memory m_wg(wg_md, eng), m_wu(wu_md, eng), m_wd(wd_md, eng);
    memory m_fc_gate(fc_gate_md, eng), m_fc_up(fc_up_md, eng);

    ASSERT_TRUE(load_binary_to_memory(dir + "/src.bin", m_src));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_gate.bin", m_wg));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_up.bin", m_wu));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_down.bin", m_wd));

    fprintf(stderr, "All FP16 dump files loaded from %s\n", dir.c_str());

    // Primitive reference path (3 separate matmuls)
    auto up_pd = matmul::primitive_desc(eng, src_md, wu_md, fc_up_md);
    auto up_prim = matmul(up_pd);

    primitive_attr gate_attr;
    gate_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    post_ops gate_po;
    gate_po.append_eltwise(algorithm::eltwise_swish, 1.f, 1.f);
    gate_po.append_binary(algorithm::binary_mul, fc_up_md);
    gate_attr.set_post_ops(gate_po);
    auto gate_pd = matmul::primitive_desc(eng, src_md, wg_md, fc_gate_md, gate_attr);
    auto gate_prim = matmul(gate_pd);

    auto down_pd = matmul::primitive_desc(eng, fc_gate_md, wd_md, dst_md);
    auto down_prim = matmul(down_pd);

    up_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS, m_wu}, {DNNL_ARG_DST, m_fc_up}});
    gate_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src},
                    {DNNL_ARG_WEIGHTS, m_wg},
                    {DNNL_ARG_DST, m_fc_gate},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, m_fc_up}});
    down_prim.execute(strm,
            {{DNNL_ARG_SRC, m_fc_gate}, {DNNL_ARG_WEIGHTS, m_wd}, {DNNL_ARG_DST, m_dst_prim}});
    strm.wait();

    primitive_attr gmlp_attr;
    gmlp_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);

    auto gmlp_pd = gmlp_t::pd_t(eng, src_md, wg_md, wu_md, wd_md, dst_md,
            dnnl_eltwise_swish, gmlp_attr);
    auto gmlp_prim = gmlp_t(gmlp_pd);

    gmlp_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src},
                    {DNNL_ARG_WEIGHTS_GATE, m_wg},
                    {DNNL_ARG_WEIGHTS_UP, m_wu},
                    {DNNL_ARG_WEIGHTS_DOWN, m_wd},
                    {DNNL_ARG_DST, m_dst_int}});
    strm.wait();

    const size_t n_elems = mb * ic;
    std::vector<float> res_prim(n_elems), res_int(n_elems);
    move_data(eng, strm, res_prim, m_dst_prim, false);
    move_data(eng, strm, res_int, m_dst_int, false);

    int n_mismatches = 0;
    float max_diff = 0.0f, max_val = 0.f, max_gold = 0.f;
    for (size_t i = 0; i < n_elems; i++) {
        float abs_diff = std::abs(res_int[i] - res_prim[i]);
        float rel_diff = (res_prim[i] != 0) ? abs_diff / std::abs(res_prim[i]) : abs_diff;
        if (abs_diff > 1e-4 && rel_diff > 5e-2) {
            if (n_mismatches < 10)
                printf("fp16 mismatch @ %zu: internal=%f prim=%f\n", i,
                        res_int[i], res_prim[i]);
            if (abs_diff > max_diff) {
                max_diff = abs_diff;
                max_val = res_int[i];
                max_gold = res_prim[i];
            }
            n_mismatches++;
        }
    }
    int threshold = int(n_elems) * 0.0006;
    printf("fp16 total mismatches: %d / %zu (threshold: %d), max_diff: %f (%f vs %f)\n",
            n_mismatches, n_elems, threshold, max_diff, max_val, max_gold);
    ASSERT_LE(n_mismatches, threshold);
}

TEST(mlp_dump_test, compare_from_dump_qwen_int4) {
    const char *dump_dir_env = std::getenv("GMLP_DUMP_DIR");
    if (!dump_dir_env) GTEST_SKIP() << "Set GMLP_DUMP_DIR=/tmp/gmlp_dump";
    std::string dir(dump_dir_env);

    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Requires GPU");
    const char *dev_env = std::getenv("GMLP_GPU_DEV");
    int dev_idx = dev_env ? std::atoi(dev_env) : 0;
    engine eng(engine::kind::gpu, dev_idx);
    stream strm(eng);

    // Qwen 2.5 7B INT4 layer 0 dimensions (all u4, group_size=128)
    const memory::dim mb = 1024, ic = 3584, oc = 18944;
    const int gs = 128;
    const int gate_ngroups = ic / gs;   // 3584/128 = 28
    const int down_ngroups = oc / gs;   // 18944/128 = 148

    // --- Memory descriptors ---
    auto src_md = memory::desc({mb, ic}, mdt::f16, tag::ab);
    auto dst_md = memory::desc({mb, ic}, mdt::f16, tag::ab);

    // All weights are u4, logical [IC, OC] with ba tag = physical [OC, IC]
    auto wg_md = memory::desc({ic, oc}, mdt::u4, tag::ba);
    auto wu_md = memory::desc({ic, oc}, mdt::u4, tag::ba);
    auto wd_md = memory::desc({oc, ic}, mdt::u4, tag::ba);

    // Scales: all grouped [ngroups, OC]
    auto sg_md = memory::desc({gate_ngroups, oc}, mdt::f16, tag::ab);  // [28, 18944]
    auto su_md = memory::desc({gate_ngroups, oc}, mdt::f16, tag::ab);  // [28, 18944]
    auto sd_md = memory::desc({down_ngroups, ic}, mdt::f16, tag::ab);  // [148, 3584]

    // Zero points: all grouped
    auto zg_md = memory::desc({gate_ngroups, oc}, mdt::u8, tag::ab);
    auto zu_md = memory::desc({gate_ngroups, oc}, mdt::u8, tag::ab);
    auto zd_md = memory::desc({down_ngroups, ic}, mdt::u8, tag::ab);

    // Intermediate buffers
    auto fc_gate_md = memory::desc({mb, oc}, mdt::f16, tag::ab);
    auto fc_up_md = memory::desc({mb, oc}, mdt::f32, tag::ab);

    // --- Create memories ---
    memory m_src(src_md, eng), m_dst_prim(dst_md, eng), m_dst_int(dst_md, eng);
    memory m_wg(wg_md, eng), m_wu(wu_md, eng), m_wd(wd_md, eng);
    memory m_sg(sg_md, eng), m_su(su_md, eng), m_sd(sd_md, eng);
    memory m_zg(zg_md, eng), m_zu(zu_md, eng), m_zd(zd_md, eng);
    memory m_fc_gate(fc_gate_md, eng), m_fc_up(fc_up_md, eng);

    // --- Load from dump files ---
    ASSERT_TRUE(load_binary_to_memory(dir + "/src.bin", m_src));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_gate.bin", m_wg));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_up.bin", m_wu));
    ASSERT_TRUE(load_binary_to_memory(dir + "/w_down.bin", m_wd));
    ASSERT_TRUE(load_binary_to_memory(dir + "/scale_gate.bin", m_sg));
    ASSERT_TRUE(load_binary_to_memory(dir + "/scale_up.bin", m_su));
    ASSERT_TRUE(load_binary_to_memory(dir + "/scale_down.bin", m_sd));
    ASSERT_TRUE(load_binary_to_memory(dir + "/zp_gate.bin", m_zg));
    ASSERT_TRUE(load_binary_to_memory(dir + "/zp_up.bin", m_zu));
    ASSERT_TRUE(load_binary_to_memory(dir + "/zp_down.bin", m_zd));

    fprintf(stderr, "All Qwen dump files loaded from %s\n", dir.c_str());

    // ========== Primitive reference path (3 separate matmuls) ==========
    const bool skip_ref = std::getenv("GMLP_SKIP_REF") != nullptr;
    if (!skip_ref) {
    fprintf(stderr, "[PROGRESS] Creating Qwen reference path...\n");

    // Up matmul: src[mb,ic] × w_up[ic,oc] → fc_up[mb,oc] (f32), grouped
    primitive_attr up_attr;
    up_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    up_attr.set_scales(DNNL_ARG_WEIGHTS, 3, {gs, 1}, mdt::f16);
    up_attr.set_zero_points(DNNL_ARG_WEIGHTS, 3, {gs, 1}, mdt::u8);
    auto up_pd = matmul::primitive_desc(eng, src_md, wu_md, fc_up_md, up_attr);
    auto up_prim = matmul(up_pd);

    // Gate matmul: src[mb,ic] × w_gate[ic,oc] → fc_gate[mb,oc] (f16) + swish + mul(fc_up)
    primitive_attr gate_attr;
    gate_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    gate_attr.set_scales(DNNL_ARG_WEIGHTS, 3, {gs, 1}, mdt::f16);
    gate_attr.set_zero_points(DNNL_ARG_WEIGHTS, 3, {gs, 1}, mdt::u8);
    post_ops gate_po;
    gate_po.append_eltwise(algorithm::eltwise_swish, 1.f, 1.f);
    gate_po.append_binary(algorithm::binary_mul, fc_up_md);
    gate_attr.set_post_ops(gate_po);
    auto gate_pd = matmul::primitive_desc(eng, src_md, wg_md, fc_gate_md, gate_attr);
    auto gate_prim = matmul(gate_pd);

    // Down matmul: fc_gate[mb,oc] × w_down[oc,ic] → dst[mb,ic] (f16)
    primitive_attr down_attr;
    down_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    down_attr.set_scales(DNNL_ARG_WEIGHTS, 3, {gs, 1}, mdt::f16);
    down_attr.set_zero_points(DNNL_ARG_WEIGHTS, 3, {gs, 1}, mdt::u8);
    auto down_pd = matmul::primitive_desc(eng, fc_gate_md, wd_md, dst_md, down_attr);
    auto down_prim = matmul(down_pd);

    fprintf(stderr, "[PROGRESS] Executing Qwen reference path...\n");
    up_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS, m_wu},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_su},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_zu},
                    {DNNL_ARG_DST, m_fc_up}});
    gate_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS, m_wg},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_sg},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_zg},
                    {DNNL_ARG_DST, m_fc_gate},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                            m_fc_up}});
    down_prim.execute(strm,
            {{DNNL_ARG_SRC, m_fc_gate}, {DNNL_ARG_WEIGHTS, m_wd},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_sd},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_zd},
                    {DNNL_ARG_DST, m_dst_prim}});
    strm.wait();
    fprintf(stderr, "[PROGRESS] Qwen reference path done.\n");
    }

    // ========== Internal gated_mlp path ==========
    fprintf(stderr, "[PROGRESS] Creating Qwen gated_mlp primitive...\n");
    primitive_attr gmlp_attr;
    gmlp_attr.set_fpmath_mode(
            static_cast<enum fpmath_mode>(fpmath_mode::any), true);
    gmlp_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    // All three are grouped with gs=128
    gmlp_attr.set_scales(DNNL_ARG_WEIGHTS_GATE, 3, {gs, 1}, mdt::f16);
    gmlp_attr.set_zero_points(DNNL_ARG_WEIGHTS_GATE, 3, {gs, 1}, mdt::u8);
    gmlp_attr.set_scales(DNNL_ARG_WEIGHTS_UP, 3, {gs, 1}, mdt::f16);
    gmlp_attr.set_zero_points(DNNL_ARG_WEIGHTS_UP, 3, {gs, 1}, mdt::u8);
    gmlp_attr.set_scales(DNNL_ARG_WEIGHTS_DOWN, 3, {gs, 1}, mdt::f16);
    gmlp_attr.set_zero_points(DNNL_ARG_WEIGHTS_DOWN, 3, {gs, 1}, mdt::u8);

    auto gmlp_pd = gmlp_t::pd_t(eng, src_md, wg_md, wu_md, wd_md, dst_md,
            dnnl_eltwise_swish, gmlp_attr);
    auto gmlp_prim = gmlp_t(gmlp_pd);

    auto scratchpad_md = gmlp_pd.scratchpad_desc();
    memory scratchpad_mem(scratchpad_md, eng);
    fprintf(stderr, "[PROGRESS] Qwen gated_mlp created (scratchpad: %zu bytes). Executing...\n",
            scratchpad_md.get_size());

    gmlp_prim.execute(strm,
            {{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS_GATE, m_wg},
                    {DNNL_ARG_WEIGHTS_UP, m_wu},
                    {DNNL_ARG_WEIGHTS_DOWN, m_wd},
                    {DNNL_ARG_DST, m_dst_int},
                    {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_GATE, m_sg},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_GATE, m_zg},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_UP, m_su},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_UP, m_zu},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_DOWN, m_sd},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_DOWN,
                            m_zd}});
    strm.wait();

    // ========== Compare results ==========
    const size_t n_elems = mb * ic;
    std::vector<float> res_prim(n_elems), res_int(n_elems);
    move_data(eng, strm, res_prim, m_dst_prim, false);
    move_data(eng, strm, res_int, m_dst_int, false);

    int n_mismatches = 0;
    float max_diff = 0.0f, max_val = 0.f, max_gold = 0.f;
    for (size_t i = 0; i < n_elems; i++) {
        float abs_diff = std::abs(res_int[i] - res_prim[i]);
        float rel_diff = (res_prim[i] != 0)
                ? abs_diff / std::abs(res_prim[i])
                : abs_diff;
        if (abs_diff > 1e-4 && rel_diff > 5e-2) {
            if (n_mismatches < 10)
                printf("qwen mismatch @ %zu: internal=%f prim=%f\n", i,
                        res_int[i], res_prim[i]);
            if (abs_diff > max_diff) {
                max_diff = abs_diff;
                max_val = res_int[i];
                max_gold = res_prim[i];
            }
            n_mismatches++;
        }
    }
    int threshold = int(n_elems) * 0.0006;
    printf("Qwen total mismatches: %d / %zu (threshold: %d), max_diff: %f (%f vs %f)\n",
            n_mismatches, n_elems, threshold, max_diff, max_val, max_gold);
    ASSERT_LE(n_mismatches, threshold);
}

} // namespace impl
} // namespace dnnl
