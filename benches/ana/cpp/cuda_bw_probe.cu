// ============================================================================
// cuda_bw_probe.cu —— GPU 侧的带宽与 Decode 上界探针
// ----------------------------------------------------------------------------
// 与 cpp/ggml_probe.cpp 互补：那个测 CPU 访存，这个测 GPU 显存带宽，
// 并直接回答「这块卡的 Decode 物理天花板是多少 token/s」。
//
// ⚠️ 本文件需要 nvcc 编译，未在开发机（macOS，无 CUDA）上验证过编译。
//    在目标机上执行：
//      nvcc -O3 -arch=sm_89 cuda_bw_probe.cu -o cuda_bw_probe
//      ./cuda_bw_probe --weight-gib 15.33 --size-gib 4
//
// 三个测量：
//   1) copy  —— device-to-device 大块拷贝（读+写，带宽 = 2×字节/时间）
//   2) read  —— 只读归约（纯读带宽，这是 Decode 的真实上限）
//   3) gemv  —— q4_0 布局的 GEMV，单 token 向量 × 权重矩阵（Decode 主力）
//
// 为什么必须单独测 read 而不是只用 copy？
//   copy 的读写各占一半带宽，而 Decode 是**只读权重**（KV 少量读写）。
//   用 copy 带宽反推 Decode 会低估约 2 倍 —— 这是常见的估算错误来源。
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t e_ = (expr);                                                \
        if (e_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(e_));                               \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// q4_0 布局：32 权重 / 18 字节块（2 字节 fp16 scale + 16 字节 nibble）
// 与 ggml 的 block_q4_0 完全一致，保证访存量可比。
// ---------------------------------------------------------------------------
struct __align__(2) block_q4_0 {
    __half d;
    uint8_t qs[16];
};
static_assert(sizeof(block_q4_0) == 18, "q4_0 block 必须是 18 字节");

// ---------------------------------------------------------------------------
// Kernel 1：纯读带宽（只读归约，向量化 float4）
// ---------------------------------------------------------------------------
__global__ void read_bw_kernel(const float4* __restrict__ in, size_t n4,
                               float* __restrict__ out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n4; i += stride) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // 块内归约后写一个值，避免原子操作成为瓶颈
    __shared__ float smem[256];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// Kernel 2：D2D 拷贝带宽（读+写各一遍）
// ---------------------------------------------------------------------------
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                            size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) out[i] = in[i];
}

// ---------------------------------------------------------------------------
// Kernel 3：q4_0 GEMV —— y[r] = Σ_c x[c] · dequant(W[r,c])
//   每个 warp 处理一行：lane 负责连续的 32 权重块，最后 warp 内归约。
//   这是 llama.cpp 的 mmvq（matrix-vector quantized）的简化参考实现。
// ---------------------------------------------------------------------------
__global__ void gemv_q4_0_kernel(const block_q4_0* __restrict__ W,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int rows, int blocks_per_row) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;
    const block_q4_0* row = W + (size_t)warp_id * blocks_per_row;

    float acc = 0.0f;
    for (int b = lane; b < blocks_per_row; b += 32) {
        const block_q4_0 blk = row[b];
        const float d = __half2float(blk.d);
        const float* xb = x + (size_t)b * 32;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t q = blk.qs[i];
            acc += ((float)(q & 0x0F) - 8.0f) * d * xb[2 * i];
            acc += ((float)(q >> 4) - 8.0f) * d * xb[2 * i + 1];
        }
    }
    // warp 内归约
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

// ---------------------------------------------------------------------------

struct Args {
    double weight_gib = 15.33;
    double size_gib = 4.0;
    int rows = 65536;
    int cols = 16384;
    int iters = 20;
    int warmup = 3;
    int device = 0;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](double def) { return (i + 1 < argc) ? std::atof(argv[++i]) : def; };
        if (k == "--weight-gib") a.weight_gib = next(a.weight_gib);
        else if (k == "--size-gib") a.size_gib = next(a.size_gib);
        else if (k == "--rows") a.rows = (int)next(a.rows);
        else if (k == "--cols") a.cols = (int)next(a.cols);
        else if (k == "--iters") a.iters = (int)next(a.iters);
        else if (k == "--warmup") a.warmup = (int)next(a.warmup);
        else if (k == "--device") a.device = (int)next(a.device);
        else if (k == "-h" || k == "--help") {
            std::printf("cuda_bw_probe —— GPU 带宽与 Decode 上界探针\n"
                        "  --weight-gib G   模型权重体积，用于反推 decode 上界\n"
                        "  --size-gib G     测试缓冲区大小（建议 ≥ 4GiB，超过 L2 才测得到真带宽）\n"
                        "  --rows N --cols N  gemv 形状\n"
                        "  --iters N --warmup N  重复与预热次数\n"
                        "  --device N       GPU 序号\n");
            std::exit(0);
        }
    }
    return a;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) { std::fprintf(stderr, "未检测到 CUDA 设备\n"); return 1; }
    CUDA_CHECK(cudaSetDevice(a.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, a.device));
    std::printf("===============================================================\n");
    std::printf("cuda_bw_probe —— GPU 带宽与 Decode 上界探针\n");
    std::printf("===============================================================\n");
    std::printf("设备          %s (CC %d.%d, %d SMs)\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount);
    std::printf("显存          %.2f GiB\n", prop.totalGlobalMem / 1073741824.0);
    std::printf("标称带宽      %.1f GB/s（按 busWidth × memClockRate 推算）\n",
                2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);
    std::printf("L2 Cache      %.1f MiB\n", prop.l2CacheSize / 1048576.0);
    std::printf("反推用体积    %.2f GiB\n\n", a.weight_gib);

    // ---- 1. 纯读带宽 -------------------------------------------------------
    const size_t bytes = (size_t)(a.size_gib * 1073741824.0);
    const size_t n4 = bytes / sizeof(float4);
    float4 *din = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&din, n4 * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dout, n4 * sizeof(float4)));
    CUDA_CHECK(cudaMemset(din, 1, n4 * sizeof(float4)));
    float* dsum = nullptr;
    CUDA_CHECK(cudaMalloc(&dsum, 4096 * sizeof(float)));

    const int threads = 256;
    const int blocks = std::min<int>(prop.multiProcessorCount * 32, 65535);

    std::vector<double> read_t, copy_t;
    for (int it = 0; it < a.warmup + a.iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        read_bw_kernel<<<blocks, threads>>>(din, n4, dsum);
        CUDA_CHECK(cudaDeviceSynchronize());
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (it >= a.warmup) read_t.push_back(dt);

        t0 = std::chrono::steady_clock::now();
        copy_kernel<<<blocks, threads>>>(din, dout, n4);
        CUDA_CHECK(cudaDeviceSynchronize());
        dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (it >= a.warmup) copy_t.push_back(dt);
    }
    const double gib = (double)bytes / 1073741824.0;
    const double read_gbps = gib / median(read_t);
    const double copy_gbps = 2.0 * gib / median(copy_t);   // 读+写，故乘 2
    std::printf("[1] 纯读带宽    %.2f GB/s（缓冲 %.2f GiB，中位 %.4f s）\n",
                read_gbps, gib, median(read_t));
    std::printf("[2] 拷贝带宽    %.2f GB/s（读+写各一遍）\n", copy_gbps);
    std::printf("    → Decode 只读权重，故应使用**纯读带宽**反推；\n");
    std::printf("      用拷贝带宽会低估约 2 倍（常见估算错误）\n\n");

    // ---- 3. q4_0 GEMV ------------------------------------------------------
    const int blocks_per_row = a.cols / 32;
    const size_t w_blocks = (size_t)blocks_per_row * a.rows;
    const size_t w_bytes = w_blocks * sizeof(block_q4_0);
    block_q4_0* dW = nullptr;
    float* dx = nullptr;
    float* dy = nullptr;
    CUDA_CHECK(cudaMalloc(&dW, w_bytes));
    CUDA_CHECK(cudaMalloc(&dx, a.cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, a.rows * sizeof(float)));
    CUDA_CHECK(cudaMemset(dW, 0x21, w_bytes));            // nibble 与 scale 均为非零
    CUDA_CHECK(cudaMemset(dx, 0, a.cols * sizeof(float)));

    const int gemv_threads = 256;                          // 8 warps/block
    const int gemv_blocks = (a.rows + 7) / 8;
    std::vector<double> gemv_t;
    for (int it = 0; it < a.warmup + a.iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        gemv_q4_0_kernel<<<gemv_blocks, gemv_threads>>>(dW, dx, dy, a.rows, blocks_per_row);
        CUDA_CHECK(cudaDeviceSynchronize());
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (it >= a.warmup) gemv_t.push_back(dt);
    }
    const double w_gib = (double)w_bytes / 1073741824.0;
    const double gemv_gbps = w_gib / median(gemv_t);
    const double gemv_gflops = 2.0 * (double)a.rows * (double)a.cols / median(gemv_t) / 1e9;
    std::printf("[3] q4_0 GEMV   %.2f GB/s，%.1f GFLOPS\n", gemv_gbps, gemv_gflops);
    std::printf("    矩阵 %d × %d，权重 %.3f GiB，中位耗时 %.3f ms\n",
                a.rows, a.cols, w_gib, median(gemv_t) * 1000.0);
    std::printf("    内核效率 = %.1f%%（相对纯读上界）\n\n", gemv_gbps / read_gbps * 100.0);

    // ---- 结论 --------------------------------------------------------------
    std::printf("===============================================================\n");
    std::printf("结论：以纯读带宽反推 Decode 理论上界\n");
    std::printf("===============================================================\n");
    std::printf("  %-24s %-16s %s\n", "模型权重体积", "每步字节(GB)", "理论上界(tps)");
    for (double g : {15.33, 8.0, 4.0, 2.0}) {
        std::printf("  %-24.2f %-16.2f %-10.2f\n",
                    g, g * 1.073741824, read_gbps * 1e9 / (g * 1073741824.0));
    }
    std::printf("\n");
    std::printf("与实测对照的方法：\n");
    std::printf("  1) 记录服务端实测 Decode 速率 tps_measured；\n");
    std::printf("  2) 计算达成率 = tps_measured / 上界；\n");
    std::printf("  3) 达成率 >80%% ⇒ 已触及物理天花板，只能减少每步搬运字节\n");
    std::printf("     达成率 <60%% ⇒ 存在 kernel/调度/回退问题，有明确优化空间\n");

    CUDA_CHECK(cudaFree(din));  CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(dsum)); CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dx));   CUDA_CHECK(cudaFree(dy));
    return 0;
}
