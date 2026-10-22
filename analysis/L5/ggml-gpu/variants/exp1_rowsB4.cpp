// exp1: rows_B = 4 -> ne11 = 4, expect MUL_MAT to leave MMVF and go to cuBLAS
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level; (void) user_data;
    fputs(text, stderr); fflush(stderr);
}

struct simple_model {
    struct ggml_tensor * a {};
    struct ggml_tensor * b {};
    ggml_backend_t backend {};
    ggml_backend_t cpu_backend {};
    ggml_backend_sched_t sched {};
    std::vector<uint8_t> buf;
};

const int rows_A = 4, cols_A = 2;
float matrix_A[rows_A * cols_A] = { 2, 8, 5, 1, 4, 2, 8, 6 };

const int rows_B = 4, cols_B = 2;      // <-- changed 3 -> 4
float matrix_B[rows_B * cols_B] = { 10, 5, 9, 9, 5, 4, 1, 2 };

void init_model(simple_model & model) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    ggml_backend_load_all();
    model.backend = ggml_backend_init_best();
    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
    model.sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
}

struct ggml_cgraph * build_graph(simple_model& model) {
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);
    struct ggml_init_params params0 = { buf_size, model.buf.data(), true };
    struct ggml_context * ctx = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);
    model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
    struct ggml_tensor * result = ggml_mul_mat(ctx, model.a, model.b);
    ggml_build_forward_expand(gf, result);
    ggml_free(ctx);
    return gf;
}

struct ggml_tensor * compute(simple_model & model, struct ggml_cgraph * gf) {
    ggml_backend_sched_reset(model.sched);
    ggml_backend_sched_alloc_graph(model.sched, gf);
    ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));
    ggml_backend_sched_graph_compute(model.sched, gf);
    return ggml_graph_node(gf, -1);
}

int main(void) {
    ggml_time_init();
    simple_model model;
    init_model(model);
    struct ggml_cgraph * gf = build_graph(model);
    struct ggml_tensor * result = compute(model, gf);
    std::vector<float> out(ggml_nelements(result));
    ggml_backend_tensor_get(result, out.data(), 0, ggml_nbytes(result));
    printf("result ne=[%lld,%lld]\n", (long long) result->ne[0], (long long) result->ne[1]);
    for (int j = 0; j < result->ne[1]; j++) {
        for (int i = 0; i < result->ne[0]; i++) printf(" %8.2f", out[j*result->ne[0] + i]);
        printf("\n");
    }
    ggml_backend_sched_free(model.sched);
    ggml_backend_free(model.backend);
    ggml_backend_free(model.cpu_backend);
    return 0;
}
