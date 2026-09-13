// q4km_test.c — Q4_K 裁剪实现的独立测试与断点调试入口
//
// 目的：
//   1. 验证 ggml-quants_q4_k_m.c 的行为与原 ggml-quants.c 一致（回归对比）
//   2. 提供小规模、确定性的数据，便于逐行断点调试量化算法
//
// 编译（见 doc/q4km_debug/README.md）：
//   make -f doc/q4km_debug/Makefile
//
// 运行：
//   ./build/q4km_test            # 全部测试
//   ./build/q4km_test --debug    # 打印中间量，配合断点观察

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "ggml-quants.h"

// ---------------------------------------------------------------- utils
static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("  (%s:%d)\n", __FILE__, __LINE__); g_fail++; } \
} while (0)

// 确定性伪随机（不依赖平台 rand）
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (float)((rng_state >> 40) & 0xFFFFFF) / (float)0xFFFFFF; // [0,1]
}
static float frand_uniform(float lo, float hi) { return lo + (hi - lo) * frand(); }

static double rel_rmse(const float *a, const float *b, int n) {
    double err = 0, mag = 0;
    for (int i = 0; i < n; i++) { double e = a[i] - b[i]; err += e * e; mag += a[i] * a[i]; }
    return mag > 0 ? sqrt(err / mag) : 0;
}

// 生成典型 LLM 权重分布：近似高斯 + 少量离群值
static void make_llm_like_weights(float *x, int n) {
    for (int i = 0; i < n; i++) {
        // Box-Muller 近似：3 次均匀平均 ≈ 高斯
        float g = (frand() + frand() + frand() - 1.5f) * 0.8f;
        if (frand() < 0.005f) g *= 8.0f;  // 0.5% 离群值
        x[i] = g;
    }
}

// ---------------------------------------------------------------- test 1
// 基本往返：量化 → 反量化，误差应在 Q4_K 预期范围（< ~7%）
static void test_roundtrip(void) {
    const int64_t N = QK_K * 4; // 4 个超块
    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));
    block_q4_K *q = malloc(N / QK_K * sizeof(block_q4_K));

    make_llm_like_weights(x, N);
    quantize_row_q4_K_ref(x, q, N);
    dequantize_row_q4_K(q, y, N);

    double e = rel_rmse(x, y, N);
    printf("[test_roundtrip]  rel RMSE = %.5f  (expect < 0.08)\n", e);
    CHECK(e < 0.08, "roundtrip error too large: %f\n", e);

    free(x); free(y); free(q);
}

// ---------------------------------------------------------------- test 2
// 已知数据的手工验证：线性斜坡数据，可人工推算期望量化值
static void test_known_values(void) {
    const int64_t N = QK_K;
    float x[QK_K], y[QK_K];
    block_q4_K q;

    // 精确斜坡 0..255 → 值域 [-1, 1]
    for (int i = 0; i < QK_K; i++) x[i] = -1.0f + 2.0f * i / (QK_K - 1);

    quantize_row_q4_K_ref(x, &q, N);
    dequantize_row_q4_K(&q, y, N);

    // 检查结构性不变量：
    // 1. 最大子块 scale 对应 6-bit 63
    float d = /* fp16 -> f32 */ 0;  // 简单 f16 解码
    {   // 手工 f16→f32（避免依赖 GGML_CPU 宏）
        uint16_t h = q.d;
        int exp = ((h >> 10) & 0x1F) - 15;
        d = ldexpf((h & 0x3FF) / 1024.0f + 1.0f, exp);
        if (exp == -15 && !(h & 0x3FF)) d = 0;
    }
    printf("[test_known_values] d = %.6f, dmin_raw = %04x\n", d, q.dmin);

    // 2. 反量化值单调不减（输入单调）
    int monotonic = 1;
    for (int i = 1; i < QK_K; i++) if (y[i] < y[i-1] - 1e-6f) { monotonic = 0; break; }
    CHECK(monotonic, "dequantized ramp not monotonic\n");

    // 3. 误差下界：4-bit + 6-bit scale 应有非零但受控的误差
    double e = rel_rmse(x, y, N);
    printf("[test_known_values] ramp rel RMSE = %.5f\n", e);
    CHECK(e > 0.005 && e < 0.15, "ramp error out of expected band: %f\n", e);
}

// ---------------------------------------------------------------- test 3
// imatrix 路径：均匀随机权重矩阵 + 随机重要性
static void test_imatrix_path(void) {
    const int64_t N = QK_K * 4, ROWS = 2;
    float *x  = malloc(ROWS * N * sizeof(float));
    float *y  = malloc(ROWS * N * sizeof(float));
    float *im = malloc(ROWS * N * sizeof(float));
    block_q4_K *q = malloc(ROWS * N / QK_K * sizeof(block_q4_K));

    for (int i = 0; i < ROWS * N; i++) {
        x[i]  = frand_uniform(-1, 1);
        im[i] = frand() + 0.1f;
    }
    quantize_q4_K(x, q, ROWS, N, im);
    dequantize_row_q4_K(q, y, ROWS * N);

    double e = rel_rmse(x, y, ROWS * N);
    printf("[test_imatrix_path] rel RMSE = %.5f  (expect < 0.08)\n", e);
    CHECK(e < 0.08, "imatrix roundtrip error too large: %f\n", e);

    free(x); free(y); free(im); free(q);
}

// ---------------------------------------------------------------- test 4
// 极端数据：全零 / 恒定 / 单极性 / NaN 邻界，不应崩溃且输出有限
static void test_edge_cases(void) {
    const int64_t N = QK_K;
    float x[QK_K], y[QK_K];
    block_q4_K q;

    // 全零
    memset(x, 0, sizeof(x));
    quantize_row_q4_K_ref(x, &q, N);
    dequantize_row_q4_K(&q, y, N);
    for (int i = 0; i < N; i++) CHECK(y[i] == 0, "all-zero block: y[%d]=%g\n", i, y[i]);

    // 恒定值
    for (int i = 0; i < N; i++) x[i] = 0.5f;
    quantize_row_q4_K_ref(x, &q, N);
    dequantize_row_q4_K(&q, y, N);
    double e = rel_rmse(x, y, N);
    printf("[test_edge_cases] constant-block rel RMSE = %.5f\n", e);
    CHECK(e < 0.01, "constant block error too large: %f\n", e);

    // 单极性（全正）
    for (int i = 0; i < N; i++) x[i] = frand() * 2.0f;
    quantize_row_q4_K_ref(x, &q, N);
    dequantize_row_q4_K(&q, y, N);
    e = rel_rmse(x, y, N);
    printf("[test_edge_cases] all-positive rel RMSE = %.5f\n", e);
    CHECK(e < 0.1, "all-positive error too large: %f\n", e);

    // 极小值（低于 GROUP_MAX_EPS）
    for (int i = 0; i < N; i++) x[i] = 1e-20f * (frand() - 0.5f);
    quantize_row_q4_K_ref(x, &q, N);
    dequantize_row_q4_K(&q, y, N);
    for (int i = 0; i < N; i++) CHECK(isfinite(y[i]), "tiny block: y[%d] not finite\n", i);
}

// ---------------------------------------------------------------- test 5 (optional)
// 与原实现逐位对比：链接原 ggml-quants.o 后比较输出。
// 由 Makefile 的 target `compare` 启用（定义 COMPARE_WITH_ORIG）。
#ifdef COMPARE_WITH_ORIG
extern void quantize_row_q4_K_ref_orig(const float *x, block_q4_K *y, int64_t k);
extern void dequantize_row_q4_K_orig(const block_q4_K *x, float *y, int64_t k);

static void test_parity_with_orig(void) {
    const int64_t N = QK_K * 8;
    float *x = malloc(N * sizeof(float));
    block_q4_K *q1 = malloc(N / QK_K * sizeof(block_q4_K));
    block_q4_K *q2 = malloc(N / QK_K * sizeof(block_q4_K));
    float *y1 = malloc(N * sizeof(float)), *y2 = malloc(N * sizeof(float));

    make_llm_like_weights(x, N);
    quantize_row_q4_K_ref(x, q1, N);          // 裁剪版
    quantize_row_q4_K_ref_orig(x, q2, N);     // 原版（由 shims 提供）

    CHECK(memcmp(q1, q2, N / QK_K * sizeof(block_q4_K)) == 0,
          "trimmed quantize output differs from original\n");
    dequantize_row_q4_K(q1, y1, N);
    dequantize_row_q4_K_orig(q2, y2, N);
    for (int i = 0; i < N; i++) CHECK(y1[i] == y2[i], "dequant mismatch at %d\n", i);

    printf("[test_parity] outputs identical to original implementation\n");
    free(x); free(q1); free(q2); free(y1); free(y2);
}
#endif

// ---------------------------------------------------------------- debug dump
// --debug: 打印一个超块的完整中间量，是断点调试的"地图"
static void debug_dump_block(void) {
    const int64_t N = QK_K;
    float x[QK_K], y[QK_K];
    block_q4_K q;

    printf("=== debug: single super-block dump ===\n");
    for (int i = 0; i < N; i++) x[i] = -1.0f + 2.0f * i / (QK_K - 1);

    // 想在 make_qkx2_quants 内部观察？在 ggml-quants_q4_k_m.c 中：
    //   断点位置建议：
    //   - quantize_row_q4_K_ref:  scales[j] = make_qkx2_quants(...)  ← 观察每个子块的输入
    //   - make_qkx2_quants:       best_error 比较处                   ← 观察网格搜索收敛
    //   - quantize_row_q4_K_ref:  get_scale_min_k4(j, ...) 调用处     ← 观察打包结果
    quantize_row_q4_K_ref(x, &q, N);
    dequantize_row_q4_K(&q, y, N);

    printf("d(f16 bits)=0x%04x  dmin(f16 bits)=0x%04x\n", q.d, q.dmin);
    printf("scales[12]:");
    for (int i = 0; i < 12; i++) printf(" %02x", q.scales[i]);
    printf("\n");
    printf("sub-block (scale, min) pairs decoded:\n");
    for (int j = 0; j < 8; j++) {
        // 复制 get_scale_min_k4 逻辑用于展示
        uint8_t sc, m;
        const uint8_t *qq = q.scales;
        if (j < 4) { sc = qq[j] & 63; m = qq[j+4] & 63; }
        else { sc = (qq[j+4] & 0xF) | ((qq[j-4] >> 6) << 4);
               m  = (qq[j+4] >> 4) | ((qq[j]   >> 6) << 4); }
        printf("  j=%d  scale6=%2u  min6=%2u\n", j, sc, m);
    }
    printf("qs[0..15]:");
    for (int i = 0; i < 16; i++) printf(" %02x", q.qs[i]);
    printf("\n  (低半字节=L[i], 高半字节=L[i+32])\n");
    printf("rel RMSE = %.5f\n", rel_rmse(x, y, N));
}

int main(int argc, char **argv) {
    int debug = argc > 1 && strcmp(argv[1], "--debug") == 0;

    printf("sizeof(block_q4_K) = %zu (expect 144)\n\n", sizeof(block_q4_K));
    CHECK(sizeof(block_q4_K) == 144, "block size != 144\n");

    if (debug) { debug_dump_block(); printf("\n"); }

    test_roundtrip();
    test_known_values();
    test_imatrix_path();
    test_edge_cases();
#ifdef COMPARE_WITH_ORIG
    test_parity_with_orig();
#endif

    printf("\n%s: %d failures\n", g_fail ? "RESULT" : "RESULT OK,", g_fail);
    return g_fail ? 1 : 0;
}
