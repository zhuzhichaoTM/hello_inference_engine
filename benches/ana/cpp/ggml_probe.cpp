// ============================================================================
// ggml_probe —— 端侧/边缘大模型推理的带宽探针
// ----------------------------------------------------------------------------
// 目的：把「Decode 是 Memory-bound」这句定性判断，变成一组可测量的数字。
//
// 它回答三个问题：
//   1) 这台设备上，纯顺序读的带宽上界是多少？（stream 模式）
//   2) 按 q4_0 的打包布局（32 权重 / 18 字节块）做 GEMV 时，还能剩多少带宽？
//      （gemv 模式 —— 这才是真实的 Decode 访存形态）
//   3) 用这个实测带宽反推：给定模型体积，Decode 的理论速度上界是多少 token/s？
//
// 为什么不用 fp32 模拟？
//   量化权重的访存形态与 fp32 完全不同：q4_0 每 32 个权重压进 18 字节
//   （16 字节 nibble + 2 字节 fp16 scale）。若用 fp32 测带宽，得到的是
//   「读 4 字节/权重」的带宽，而真实推理是「读 0.5625 字节/权重」。
//   在固定带宽下，后者能喂的 token 数是前者的 7 倍 —— 所以必须按真实布局测。
//
// 编译：
//   clang++ -O3 -std=c++17 -pthread ggml_probe.cpp -o ggml_probe
//   （x86 上可加 -mavx2 -mfma；ARM 上加 -march=armv8.2-a+dotprod）
//
// 用法示例：
//   ./ggml_probe --mode all  --weight-gib 15.33 --threads 8
//   ./ggml_probe --mode gemv --rows 65536 --cols 16384 --threads 4 --iters 20
//   ./ggml_probe --mode stream --size-gib 2
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// 极简参数解析：避免引入任何第三方依赖（端侧交叉编译友好）
// ---------------------------------------------------------------------------
struct Args {
    std::string mode = "all";        // all | stream | stream-file | gemv
    std::string file;                // stream-file 模式的输入文件（通常是模型权重）
    double weight_gib = 15.33;       // 待反推的模型权重体积（默认 Qwen3.8-27B Q4_K_M）
    double size_gib = 2.0;           // stream 模式的缓冲区大小
    int rows = 65536;                // gemv 输出维度（≈ hidden_size × 10）
    int cols = 16384;                // gemv 输入维度（≈ intermediate_size）
    int threads = 0;                 // 0 = 自动取硬件并发数
    int iters = 20;                  // 每个配置的重复次数，取中位数抗抖动
    int warmup = 3;                  // 预热次数（第一遍往往偏慢，含 page fault）
    bool quiet = false;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](double def) -> double {
            if (i + 1 < argc) return std::atof(argv[++i]);
            return def;
        };
        if (k == "--mode")         a.mode = (i + 1 < argc) ? argv[++i] : a.mode;
        else if (k == "--file")        { if (i + 1 < argc) a.file = argv[++i]; }
        else if (k == "--weight-gib") a.weight_gib = next(a.weight_gib);
        else if (k == "--size-gib")   a.size_gib = next(a.size_gib);
        else if (k == "--rows")       a.rows = (int)next(a.rows);
        else if (k == "--cols")       a.cols = (int)next(a.cols);
        else if (k == "--threads")    a.threads = (int)next(a.threads);
        else if (k == "--iters")      a.iters = (int)next(a.iters);
        else if (k == "--warmup")     a.warmup = (int)next(a.warmup);
        else if (k == "--quiet")      a.quiet = true;
        else if (k == "-h" || k == "--help") {
            std::printf(
                "ggml_probe —— 推理带宽探针\n"
                "  --mode all|stream|stream-file|gemv  测哪些（默认 all）\n"
                "  --file PATH              stream-file 模式的输入文件（推荐给模型 .gguf）\n"
                "  --weight-gib G           模型权重体积，用于反推 decode 上界\n"
                "  --size-gib G             stream 模式缓冲区大小\n"
                "  --rows N --cols N        gemv 的矩阵形状\n"
                "  --threads N              线程数（0=自动）\n"
                "  --iters N                重复次数（取中位数）\n"
                "  --warmup N               预热次数\n"
                "  --quiet                  只输出结论\n");
            std::exit(0);
        }
    }
    if (a.threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        a.threads = hc ? (int)hc : 4;
    }
    return a;
}

// ---------------------------------------------------------------------------
// fp16 ↔ fp32（仅用于构造 q4_0 的 block scale，不追求全规格覆盖）
// ---------------------------------------------------------------------------
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;                       // 下溢 → 0
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);          // 上溢 → inf
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) { x = sign; }
        else {                                                 // 次正规数
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            x = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---------------------------------------------------------------------------
// q4_0 块布局：32 个权重 / 块
//   offset 0..1   : fp16 scale (d)
//   offset 2..17  : 16 字节，每字节 2 个 nibble
//   ⇒ 每块 18 字节，即 4.5 bit/权重
//   ⇒ 反量化值 = (nibble - 8) * d
//
// ⚠️ nibble 到权重下标的映射容易写错，ggml 的实际定义是**分半**而非交错：
//     低 4 位 → 该块内的第 j 个权重        (j = 0..15)
//     高 4 位 → 该块内的第 j + 16 个权重
//   写成 2j / 2j+1（交错）会得到一个「字节数一样、但布局不对」的实现 ——
//   带宽数字不受影响，但「按 q4_0 布局访问」这个说法就不成立了。
// ---------------------------------------------------------------------------
static constexpr int QK4_0 = 32;
static constexpr int Q4_0_BLOCK_BYTES = 2 + QK4_0 / 2;   // 18

static inline float dequant_nibble(uint8_t b, int half, float d) {
    int v = half == 0 ? (b & 0x0F) : (b >> 4);
    return (float)(v - 8) * d;
}

// ---------------------------------------------------------------------------
// 计时：取多次重复的**中位数**，避免被偶发调度抖动或热降频带偏
// ---------------------------------------------------------------------------
static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// stream：纯顺序读带宽
//   用 64 位字长读取 —— 逐字节求和会被编译器退化成标量循环，
//   在 CPU 上测出的「带宽」实际是「标量加载速率」，会显著低估
//   （实测本机逐字节 13.1 GB/s vs 64 位字长 21+ GB/s）。
//   这是探针最容易出错的地方：数字看起来合理，但比真值低一半。
// ---------------------------------------------------------------------------
struct StreamResult { double gbps; double seconds; double gib; };

static StreamResult run_stream(const Args& a) {
    const size_t bytes = (size_t)(a.size_gib * 1024.0 * 1024.0 * 1024.0);
    std::vector<uint8_t> buf(bytes);
    // 填充非零值：全零页可能被优化为共享零页，读带宽会虚高
    for (size_t i = 0; i < bytes; i += 4096) buf[i] = (uint8_t)(i * 2654435761u >> 24);

    std::atomic<uint64_t> sink{0};
    std::vector<double> times;
    for (int it = 0; it < a.warmup + a.iters; ++it) {
        std::atomic<uint64_t> acc{0};
        auto t0 = Clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < a.threads; ++t) {
            pool.emplace_back([&, t]() {
                // 按 64 位字长切分，避免跨线程共享 cache line
                const size_t n_words = bytes / sizeof(uint64_t);
                const uint64_t* w = reinterpret_cast<const uint64_t*>(buf.data());
                size_t chunk = n_words / (size_t)a.threads;
                size_t lo = chunk * (size_t)t;
                size_t hi = (t == a.threads - 1) ? n_words : lo + chunk;
                uint64_t local = 0;
                size_t i = lo;
                // 4 路展开：单累加链会串行化，多累加器才能打满访存
                for (; i + 4 <= hi; i += 4) {
                    local += w[i] + w[i + 1] + w[i + 2] + w[i + 3];
                }
                for (; i < hi; ++i) local += w[i];
                acc.fetch_add(local, std::memory_order_relaxed);
            });
        }
        for (auto& th : pool) th.join();
        double dt = secs_since(t0);
        sink.fetch_add(acc.load());
        if (it >= a.warmup) times.push_back(dt);
    }
    double dt = median(times);
    double gib = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    (void)sink;
    return {gib / dt, dt, gib};
}

// ---------------------------------------------------------------------------
// stream_file：直接读一个真实文件（mmap + MADV_WILLNEED）
//   这是最贴近真实 Decode 访存的测量方式：llama.cpp 也是 mmap 权重、
//   按顺序扫一遍。相比 malloc 缓冲区，它避免了分配/清零开销与
//   匿名页的大页缺失，数字通常更高也更接近生产形态。
// ---------------------------------------------------------------------------
static StreamResult run_stream_file(const Args& a, const char* path) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { std::perror("open"); std::exit(1); }
    struct stat st {};
    if (::fstat(fd, &st) != 0) { std::perror("fstat"); std::exit(1); }
    size_t bytes = (size_t)st.st_size;
    void* base = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    ::madvise(base, bytes, MADV_WILLNEED);

    const uint64_t* w = reinterpret_cast<const uint64_t*>(base);
    size_t n_words = bytes / sizeof(uint64_t);

    // 预热：把页表与 cache 层次拉起来，否则首轮测的是 page fault
    {
        volatile uint64_t s = 0;
        for (size_t i = 0; i < n_words; i += 64) s += w[i];
    }

    std::atomic<uint64_t> sink{0};
    std::vector<double> times;
    for (int it = 0; it < a.warmup + a.iters; ++it) {
        std::atomic<uint64_t> acc{0};
        auto t0 = Clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < a.threads; ++t) {
            pool.emplace_back([&, t]() {
                size_t chunk = n_words / (size_t)a.threads;
                size_t lo = chunk * (size_t)t;
                size_t hi = (t == a.threads - 1) ? n_words : lo + chunk;
                // 8 路独立累加器：单个累加链会被延迟串行化，
                // 打不满访存（实测 1 路 vs 8 路差 1.8 倍）。
                uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                uint64_t a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                size_t i = lo;
                for (; i + 8 <= hi; i += 8) {
                    a0 += w[i];     a1 += w[i + 1]; a2 += w[i + 2]; a3 += w[i + 3];
                    a4 += w[i + 4]; a5 += w[i + 5]; a6 += w[i + 6]; a7 += w[i + 7];
                }
                for (; i < hi; ++i) a0 += w[i];
                acc.fetch_add(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7,
                              std::memory_order_relaxed);
            });
        }
        for (auto& th : pool) th.join();
        double dt = secs_since(t0);
        sink.fetch_add(acc.load());
        if (it >= a.warmup) times.push_back(dt);
    }
    double dt = median(times);
    double gib = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    (void)sink;
    ::munmap(base, bytes);
    ::close(fd);
    return {gib / dt, dt, gib};
}


// ---------------------------------------------------------------------------
// gemv：按 q4_0 布局的矩阵-向量乘（Decode 的主力算子）
//   权重矩阵 W: [rows, cols]，行主序，每行 cols 个权重、按 32 分块打包。
//   计算 y[r] = Σ_c x[c] * dequant(W[r, c])
//   这是真实的 Decode 访存形态：每个权重读一次、用一次。
// ---------------------------------------------------------------------------
struct GemvResult {
    double gbps;          // 有效带宽（只计权重字节）
    double seconds;
    double weight_gib;
    double gflops;        // 有效算力，用于和 roofline 拐点对照
    double checksum;      // 结果校验和（用于比对标量与 SIMD 实现是否等价）
};

// ---------------------------------------------------------------------------
// 两种 GEMV 实现：标量 与 SIMD
// ---------------------------------------------------------------------------
// 为什么要同时提供两种，而不是只留一种：
//
//   · **标量**回答「完全不优化能拿到多少」—— 它是**下界**。
//     优点：一套代码在任何 ISA 上都能跑，数字跨架构可比。
//     缺点：它衡量的是「朴素实现」，不能代表生产内核。
//
//   · **SIMD** 回答「一个正常优化的内核能拿到多少」—— 它是**可达参照**。
//     这才是判断「生产内核还有多少空间」的正确基准。
//     只跟标量比会把空间高估到荒谬的程度（标量只有 7% 时，
//     会误判「还有 93% 空间」，而实际上生产内核早已接近上限）。
//
//   · **顺序读**才是硬件上限（常数），两种 GEMV 都是它的下界。
//
// 注意 SIMD 版仍是**参考实现**（未做多行并行、未用整数点积），
// 因此它是「可达」的下界而非上限 —— 生产内核通常还能再高一些。
// ---------------------------------------------------------------------------

// 标量实现（下界参照）
static inline float dot_q4_0_scalar(const uint8_t* qs, const float* xb, float d) {
    float acc = 0.0f;
    for (int i = 0; i < QK4_0 / 2; ++i) {
        // 低 nibble → x[i]，高 nibble → x[i + 16]（ggml 的分半布局）
        acc += dequant_nibble(qs[i], 0, d) * xb[i];
        acc += dequant_nibble(qs[i], 1, d) * xb[i + QK4_0 / 2];
    }
    return acc;
}

#if defined(__x86_64__) || defined(__i386__)
// 用 target 属性把 SIMD 指令限制在单个函数内 —— 其余代码仍是基线 x86-64，
// 这样探针二进制在没有 AVX2 的老机器上也能启动（运行时再决定用哪条路径）。
__attribute__((target("avx2,fma")))
static float dot_q4_0_avx2(const uint8_t* qs, const float* xb, float d) {
    __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(q, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(q, 4), mask);

    const __m256 eight = _mm256_set1_ps(8.0f);
    const __m256 dv = _mm256_set1_ps(d);
    __m256 acc = _mm256_setzero_ps();

    auto fma16 = [&](__m128i nib, const float* xp) {
        // 16 个 int8 → 两组 8 个 int32 → 转 float → (v-8)*d → FMA 到累加器
        __m256i i0 = _mm256_cvtepi8_epi32(nib);
        __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(nib, 8));
        __m256 f0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(i0), eight), dv);
        __m256 f1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(i1), eight), dv);
        acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(xp), acc);
        acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp + 8), acc);
    };
    fma16(lo, xb);          // 低 nibble → x[0..15]
    fma16(hi, xb + 16);     // 高 nibble → x[16..31]

    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc),
                          _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static bool have_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2")
                           && __builtin_cpu_supports("fma");
    return ok;
}
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
static float dot_q4_0_neon(const uint8_t* qs, const float* xb, float d) {
    uint8x16_t q = vld1q_u8(qs);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    uint8x16_t lo = vandq_u8(q, mask);
    uint8x16_t hi = vshrq_n_u8(q, 4);

    const float32x4_t eight = vdupq_n_f32(8.0f);
    const float32x4_t dv = vdupq_n_f32(d);
    float32x4_t acc = vdupq_n_f32(0.0f);

    auto dot8 = [&](uint8x16_t nib, bool upper, const float* xp) {
        uint16x8_t w = upper ? vmovl_u8(vget_high_u8(nib))
                             : vmovl_u8(vget_low_u8(nib));
        float32x4_t f0 = vmulq_f32(
            vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w))), eight), dv);
        float32x4_t f1 = vmulq_f32(
            vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w))), eight), dv);
        acc = vfmaq_f32(acc, f0, vld1q_f32(xp));
        acc = vfmaq_f32(acc, f1, vld1q_f32(xp + 4));
    };
    dot8(lo, false, xb);        // x[0..7]
    dot8(lo, true,  xb + 8);    // x[8..15]
    dot8(hi, false, xb + 16);   // x[16..23]
    dot8(hi, true,  xb + 24);   // x[24..31]
    return vaddvq_f32(acc);
}
static bool have_simd() { return true; }
#else
static bool have_simd() { return false; }
#endif

// 按可用指令集选实现；没有 SIMD 时退回标量
static inline float dot_q4_0_simd(const uint8_t* qs, const float* xb, float d) {
#if defined(__x86_64__) || defined(__i386__)
    if (have_avx2()) return dot_q4_0_avx2(qs, xb, d);
    return dot_q4_0_scalar(qs, xb, d);
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return dot_q4_0_neon(qs, xb, d);
#else
    return dot_q4_0_scalar(qs, xb, d);
#endif
}

static bool simd_available() {
#if defined(__x86_64__) || defined(__i386__)
    return have_avx2();
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return true;
#else
    return false;
#endif
}

static GemvResult run_gemv(const Args& a, int threads, bool use_simd) {
    const int rows = a.rows;
    const int cols = a.cols;
    const int blocks_per_row = cols / QK4_0;
    const size_t row_bytes = (size_t)blocks_per_row * Q4_0_BLOCK_BYTES;
    const size_t total = row_bytes * rows;

    std::vector<uint8_t> W(total);
    // 构造合法的 q4_0 数据：scale 取一个正常量级，nibble 随机分布
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (int r = 0; r < rows; ++r) {
        uint8_t* p = W.data() + (size_t)r * row_bytes;
        for (int b = 0; b < blocks_per_row; ++b) {
            float d = 0.002f + (float)(rnd() % 100) * 0.0001f;
            uint16_t h = f32_to_f16(d);
            std::memcpy(p, &h, 2);
            for (int i = 0; i < QK4_0 / 2; ++i) p[2 + i] = (uint8_t)(rnd() & 0xFF);
            p += Q4_0_BLOCK_BYTES;
        }
    }
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = std::sin((float)i * 0.001f);
    std::vector<float> y(rows, 0.0f);

    std::vector<double> times;
    for (int it = 0; it < a.warmup + a.iters; ++it) {
        auto t0 = Clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t]() {
                int lo = rows * t / threads;
                int hi = rows * (t + 1) / threads;
                for (int r = lo; r < hi; ++r) {
                    const uint8_t* p = W.data() + (size_t)r * row_bytes;
                    float acc = 0.0f;
                    for (int b = 0; b < blocks_per_row; ++b) {
                        uint16_t h;
                        std::memcpy(&h, p, 2);
                        float d = f16_to_f32(h);
                        const float* xb = x.data() + (size_t)b * QK4_0;
                        acc += use_simd ? dot_q4_0_simd(p + 2, xb, d)
                                        : dot_q4_0_scalar(p + 2, xb, d);
                        p += Q4_0_BLOCK_BYTES;
                    }
                    y[r] = acc;
                }
            });
        }
        for (auto& th : pool) th.join();
        double dt = secs_since(t0);
        if (it >= a.warmup) times.push_back(dt);
    }
    // 防止整个计算被优化掉，同时作为标量/SIMD 等价性校验的依据
    double checksum = 0.0;
    for (float v : y) checksum += v;

    double dt = median(times);
    double gib = (double)total / (1024.0 * 1024.0 * 1024.0);
    double flops = 2.0 * (double)rows * (double)cols;
    return {gib / dt, dt, gib, flops / dt / 1e9, checksum};
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    if (!a.quiet) {
        std::printf("===============================================================\n");
        std::printf("ggml_probe —— 推理带宽探针\n");
        std::printf("===============================================================\n");
        std::printf("线程数        %d\n", a.threads);
        std::printf("反推用模型体积 %.2f GiB\n", a.weight_gib);
        std::printf("\n");
    }

    bool do_stream = (a.mode == "all" || a.mode == "stream");
    bool do_stream_file = (a.mode == "all" || a.mode == "stream-file")
                          && !a.file.empty();
    bool do_gemv   = (a.mode == "all" || a.mode == "gemv");

    double stream_gbps = 0.0;
    double stream_file_gbps = 0.0;
    double gemv_best_gbps = 0.0;       // 标量（下界）
    double gemv_best_gflops = 0.0;
    int    gemv_best_threads = 0;
    double gemv_simd_gbps = 0.0;       // SIMD（可达参照）
    double gemv_simd_gflops = 0.0;
    int    gemv_simd_threads = 0;
    double scalar_checksum = 0.0;
    double simd_checksum = 0.0;

    if (do_stream_file) {
        // 优先做这一项：它最贴近真实 Decode 的访存形态
        StreamResult s = run_stream_file(a, a.file.c_str());
        stream_file_gbps = s.gbps;
        std::printf("[stream-file] mmap 顺序读真实文件（最贴近生产形态）\n");
        std::printf("  文件 %s\n", a.file.c_str());
        std::printf("  大小 %.3f GiB，中位耗时 %.4f s\n", s.gib, s.seconds);
        std::printf("  达成带宽  %.2f GB/s\n", s.gbps);
        std::printf("  [metric] stream_file_gbps=%.2f\n", s.gbps);
        std::printf("\n");
    }

    if (do_stream) {
        StreamResult s = run_stream(a);
        stream_gbps = s.gbps;
        std::printf("[stream] 匿名缓冲区顺序读（对照项）\n");
        std::printf("  缓冲 %.2f GiB，中位耗时 %.4f s\n", s.gib, s.seconds);
        std::printf("  达成带宽  %.2f GB/s\n", s.gbps);
        std::printf("  [metric] stream_gbps=%.2f\n", s.gbps);
        std::printf("\n");
    }

    if (do_gemv) {
        std::printf("[gemv] q4_0 布局 GEMV（Decode 主力算子）\n");
        std::printf("  矩阵 %d × %d，权重 %.3f GiB（4.5 bit/权重）\n",
                    a.rows, a.cols, (double)((size_t)(a.cols / QK4_0) * Q4_0_BLOCK_BYTES
                                             * a.rows) / (1024.0 * 1024.0 * 1024.0));
        const bool simd_ok = simd_available();
        std::printf("  SIMD 可用: %s\n", simd_ok ? "是" : "否（该 CPU 无 AVX2/NEON）");

        // 线程扫描：暴露「线程数够不够把带宽喂满」这一端侧常见问题。
        // 标量与 SIMD 各扫一遍 —— 两者的最优线程数可能不同
        // （SIMD 单线程更快，可能更早触及带宽上限）。
        std::vector<int> sweep;
        for (int t = 1; t <= a.threads; t *= 2) sweep.push_back(t);
        if (sweep.back() != a.threads) sweep.push_back(a.threads);

        for (int mode = 0; mode < 2; ++mode) {
            const bool use_simd = (mode == 1);
            if (use_simd && !simd_ok) continue;
            std::printf("  %s\n", use_simd ? "【SIMD 实现】（可达参照）"
                                            : "【标量实现】（下界参照）");
            std::printf("  %-8s %-12s %-12s %-12s\n", "线程", "带宽(GB/s)", "算力(GFLOPS)", "耗时(ms)");
            double best = 0.0, best_gf = 0.0;
            int best_t = 0;
            double ref_checksum = 0.0;
            for (int t : sweep) {
                GemvResult g = run_gemv(a, t, use_simd);
                std::printf("  %-8d %-12.2f %-12.2f %-12.3f\n",
                            t, g.gbps, g.gflops, g.seconds * 1000.0);
                if (g.gbps > best) { best = g.gbps; best_gf = g.gflops; best_t = t; }
                ref_checksum = g.checksum;   // 线程数不影响结果，取最后一次即可
            }
            if (use_simd) {
                gemv_simd_gbps = best; gemv_simd_gflops = best_gf;
                gemv_simd_threads = best_t; simd_checksum = ref_checksum;
            } else {
                gemv_best_gbps = best; gemv_best_gflops = best_gf;
                gemv_best_threads = best_t; scalar_checksum = ref_checksum;
            }
            std::printf("\n");
        }

        // 等价性校验：两种实现必须算出同一个结果。
        // 若不等，说明其中一条路径的 nibble 映射或 FMA 语义写错了 ——
        // 此时带宽数字虽然「看起来正常」，但它衡量的已经不是同一个算子。
        if (simd_ok && scalar_checksum != 0.0) {
            double rel = std::fabs(simd_checksum - scalar_checksum)
                         / std::fabs(scalar_checksum);
            std::printf("  [校验] 标量 vs SIMD 结果相对差 %.3e —— %s\n\n",
                        rel, rel < 1e-5 ? "一致（两条路径等价）"
                                        : "**不一致，实现有误，带宽数字不可信**");
        }
    }

    // ---- 三层参照：硬件上限 / 可达参照 / 下界 ---------------------------
    // 这三个数回答的是**不同的问题**，不能混用：
    //   顺序读      → 这台机器的访存硬件上限（Decode 上界的分母）
    //   SIMD GEMV   → 一个正常优化的内核能拿到多少（判断生产内核空间）
    //   标量 GEMV   → 完全不优化能拿到多少（跨架构可比的下界）
    double ref_bw = stream_file_gbps > 0 ? stream_file_gbps : stream_gbps;
    // 反推 Decode 上界必须用**顺序读**：它才是硬件上限。
    // 用 GEMV 反推会把「内核效率」误当成硬件能力（标量只有 7%，
    // 会得出「上界只有 0.07 t/s」这种荒谬结论）。
    double best_gbps = ref_bw > 0 ? ref_bw
                     : std::max({stream_gbps, gemv_simd_gbps, gemv_best_gbps});

    if (ref_bw <= 0.0 && (gemv_best_gbps > 0.0 || gemv_simd_gbps > 0.0)) {
        // 只跑了 gemv 模式：没有顺序读上界，无法算效率，也无法反推 Decode 上界。
        // 这里必须说清楚，否则读者会拿 GEMV 带宽当硬件上限 —— 那会低估一个数量级。
        std::printf("[kernel] 缺少顺序读基准（本次只测了 gemv 模式）\n");
        std::printf("  没有顺序读值就无法计算内核效率，也无法反推 Decode 上界。\n");
        std::printf("  请用 --mode all（或同时跑 --mode stream-file）重测。\n");
        if (gemv_best_gbps > 0.0)
            std::printf("  [metric] gemv_scalar_gbps=%.2f\n", gemv_best_gbps);
        if (gemv_simd_gbps > 0.0)
            std::printf("  [metric] gemv_simd_gbps=%.2f\n", gemv_simd_gbps);
        std::printf("\n");
    }
    if (ref_bw > 0.0 && (gemv_best_gbps > 0.0 || gemv_simd_gbps > 0.0)) {
        std::printf("[kernel] 内核效率（GEMV 达成带宽 ÷ 顺序读上界）\n");
        std::printf("  %-26s %-10s %-10s %s\n", "实现", "带宽(GB/s)", "占顺序读", "说明");
        if (gemv_best_gbps > 0.0) {
            std::printf("  %-26s %-10.2f %-9.1f%% %s\n", "标量（下界参照）",
                        gemv_best_gbps, gemv_best_gbps / ref_bw * 100.0,
                        "完全不优化时的水平");
        }
        if (gemv_simd_gbps > 0.0) {
            std::printf("  %-26s %-10.2f %-9.1f%% %s\n", "SIMD（可达参照）",
                        gemv_simd_gbps, gemv_simd_gbps / ref_bw * 100.0,
                        "正常优化内核的水平");
        }
        std::printf("  %-26s %-10.2f %-9.1f%% %s\n", "顺序读（硬件上限）",
                    ref_bw, 100.0, "该路径的物理天花板");
        std::printf("\n");

        // 判定要用 **SIMD** 的效率：标量效率低只说明「朴素实现慢」，
        // 不能用来判断「生产内核还有多少空间」。
        double eff = (gemv_simd_gbps > 0.0 ? gemv_simd_gbps : gemv_best_gbps) / ref_bw;
        if (eff < 0.5) {
            std::printf("  判定：**内核受限**。即使加 SIMD 也只用到 %.0f%% 的顺序读带宽，\n",
                        eff * 100.0);
            std::printf("        说明访存路径还有明显空间（预取 / 分块 / 布局 / 多行并行）。\n");
            std::printf("        注意本探针的 SIMD 版仍是**参考实现**（未做多行并行、\n");
            std::printf("        未用整数点积），生产内核通常还能再高一些。\n");
        } else {
            std::printf("  判定：**访存受限**。SIMD 参考内核已用到 %.0f%% 的顺序读带宽，\n",
                        eff * 100.0);
            std::printf("        继续优化内核收益有限，应转向减少每步搬运的字节数\n");
            std::printf("        （量化 / 投机解码）。\n");
        }
        if (gemv_best_gbps > 0.0 && gemv_simd_gbps > 0.0) {
            std::printf("  参考：SIMD 相对标量提速 %.2f× —— 这就是「内核实现」这一层的\n",
                        gemv_simd_gbps / gemv_best_gbps);
            std::printf("        可争取空间；若生产内核已达 SIMD 水平，则这层已无收益。\n");
        }
        // 机器可解析的汇总行（框架用它填报告，不靠解析中文表格）
        if (gemv_best_gbps > 0.0)
            std::printf("  [metric] gemv_scalar_gbps=%.2f gemv_scalar_efficiency=%.4f\n",
                        gemv_best_gbps, gemv_best_gbps / ref_bw);
        if (gemv_simd_gbps > 0.0)
            std::printf("  [metric] gemv_simd_gbps=%.2f gemv_simd_efficiency=%.4f\n",
                        gemv_simd_gbps, gemv_simd_gbps / ref_bw);
        std::printf("\n");
    }

    // ---- 结论：把带宽换算成 token/s --------------------------------------
    std::printf("===============================================================\n");
    std::printf("结论\n");
    std::printf("===============================================================\n");
    if (ref_bw <= 0.0) {
        std::printf("无法反推 Decode 上界：缺少顺序读基准。\n");
        std::printf("  请用 --mode all 重测（顺序读是硬件上限，必须由它作分母）。\n");
        if (gemv_simd_gbps > 0.0)
            std::printf("  已测到的 SIMD GEMV 带宽 %.2f GB/s 只能作为内核参考，\n"
                        "  不能当硬件上限用 —— 它本身只有顺序读的百分之几十。\n",
                        gemv_simd_gbps);
        return 0;
    }
    std::printf("三层参照（用途不同，不可混用）：\n");
    if (stream_file_gbps > 0)
        std::printf("  硬件上限  %-8.2f GB/s  mmap 真实文件顺序读（推荐口径）\n",
                    stream_file_gbps);
    if (gemv_simd_gbps > 0)
        std::printf("  可达参照  %-8.2f GB/s  SIMD GEMV（正常优化内核）\n", gemv_simd_gbps);
    if (gemv_best_gbps > 0)
        std::printf("  下界参照  %-8.2f GB/s  标量 GEMV（完全不优化）\n", gemv_best_gbps);
    std::printf("  反推取用  %-8.2f GB/s  ← 用硬件上限反推 Decode 上界\n", best_gbps);
    std::printf("\n");
    std::printf("以该带宽反推 Decode 理论上界（tps = 带宽 ÷ 每步搬运字节数）：\n");
    std::printf("  %-28s %-14s %s\n", "模型权重体积", "每步字节", "理论上界 tps");
    const double sizes[] = {15.33, 8.0, 4.0, 2.0, 1.11};
    for (double g : sizes) {
        double per_step = g * 1024.0 * 1024.0 * 1024.0;
        std::printf("  %-28.2f GiB %-14.2f GB %-10.2f\n",
                    g, per_step / 1e9, best_gbps * 1e9 / per_step);
    }
    std::printf("\n");
    std::printf("说明：\n");
    std::printf("  · 上界用**顺序读**作分母（硬件天花板），不含 KV 读写与 kernel\n");
    std::printf("    启动开销，因此是**乐观上界**。\n");
    std::printf("  · 若实测 Decode 已达此上界的 80%% 以上，说明已触及物理天花板，\n");
    std::printf("    继续做 kernel 微调收益极小 —— 唯一出路是减少每步搬运的字节数\n");
    std::printf("    （更低比特量化 / 投机解码）。\n");
    std::printf("  · 若实测 Decode 反推的带宽**高于**本探针的顺序读值，说明探针在负载下\n");
    std::printf("    采集（探针与推理共享同一份带宽），应以推理反推值为准。\n");
    std::printf("  · 若线程扫描中带宽未随线程数线性增长，说明是访存受限而非线程不足，\n");
    std::printf("    应优先调内存布局与预取，而不是加线程。\n");
    return 0;
}
