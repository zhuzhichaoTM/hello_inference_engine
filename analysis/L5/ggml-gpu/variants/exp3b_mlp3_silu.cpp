// exp3b: same 3 matmuls, with silu activations in between, to observe inplace reuse
//   t0 = mul_mat(w0, x)
//   s0 = silu(t0)      -> inplace reuse of t0's block (UNARY can be inplace)
//   t1 = mul_mat(w1, s0)
//   s1 = silu(t1)
//   t2 = mul_mat(w2, s1)
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level; (void) user_data;
    fputs(text, stderr); fflush(stderr);
}

struct mlp_model {
    struct ggml_tensor * w0 {}; struct ggml_tensor * w1 {}; struct ggml_tensor * w2 {};
    struct ggml_tensor * x  {};
    ggml_backend_t backend {};
    ggml_backend_t cpu_backend {};
    ggml_backend_sched_t sched {};
    std::vector<uint8_t> buf;
};

static const int K = 64, H1 = 128, H2 = 64, H3 = 128, B = 3;

void init_model(mlp_model & model) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    ggml_backend_load_all();
    model.backend = ggml_backend_init_best();
    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
    model.sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
}

struct ggml_cgraph * build_graph(mlp_model & model) {
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);
    struct ggml_init_params params0 = { buf_size, model.buf.data(), true };
    struct ggml_context * ctx = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    model.w0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K,  H1);
    model.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H1, H2);
    model.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H2, H3);
    model.x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K,  B);

    struct ggml_tensor * t0 = ggml_mul_mat(ctx, model.w0, model.x);
    struct ggml_tensor * s0 = ggml_silu(ctx, t0);
    struct ggml_tensor * t1 = ggml_mul_mat(ctx, model.w1, s0);
    struct ggml_tensor * s1 = ggml_silu(ctx, t1);
    struct ggml_tensor * t2 = ggml_mul_mat(ctx, model.w2, s1);

    ggml_set_name(t0, "t0"); ggml_set_name(s0, "s0");
    ggml_set_name(t1, "t1"); ggml_set_name(s1, "s1");
    ggml_set_name(t2, "t2");

    ggml_build_forward_expand(gf, t2);
    ggml_free(ctx);
    return gf;
}

int main(void) {
    ggml_time_init();
    mlp_model model;
    init_model(model);
    struct ggml_cgraph * gf = build_graph(model);

    ggml_backend_sched_reset(model.sched);
    ggml_backend_sched_alloc_graph(model.sched, gf);

    std::vector<float> ones(32768, 1.0f);
    ggml_backend_tensor_set(model.w0, ones.data(), 0, ggml_nbytes(model.w0));
    ggml_backend_tensor_set(model.w1, ones.data(), 0, ggml_nbytes(model.w1));
    ggml_backend_tensor_set(model.w2, ones.data(), 0, ggml_nbytes(model.w2));
    ggml_backend_tensor_set(model.x,  ones.data(), 0, ggml_nbytes(model.x));

    ggml_backend_sched_graph_compute(model.sched, gf);

    struct ggml_tensor * t2 = ggml_graph_node(gf, -1);
    std::vector<float> out(ggml_nelements(t2));
    ggml_backend_tensor_get(t2, out.data(), 0, ggml_nbytes(t2));
    printf("t2 ne=[%lld,%lld] first=%g\n", (long long) t2->ne[0], (long long) t2->ne[1], out[0]);

    ggml_backend_sched_free(model.sched);
    ggml_backend_free(model.backend);
    ggml_backend_free(model.cpu_backend);
    return 0;
}
