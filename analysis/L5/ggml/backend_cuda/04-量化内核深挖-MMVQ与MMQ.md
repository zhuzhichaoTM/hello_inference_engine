# 模块四：量化内核深挖（MMVQ / vecdotq / MMQ）

> 对标版本：ggml v0.25.3，commit `353b63b4`
> 核心产出：理解 ggml 量化推理的性能本质 —— 为什么"量化点积"能比"反量化+FP 点积"快 4~8 倍
> 主要文件：`mmvq.cu`(1567) / `vecdotq.cuh`(1380) / `mmq.cu`(393) / `mmq.cuh`(1605) / `quantize.cu` / `dequantize.cuh` / `ggml-common.h`

---

## 4.0 一句话概括

**GGML 量化推理的核心思想：把权重压成整数块存储，把激活也量化成整数块，然后用「整数点积 + 两个 scale 相乘」代替「反量化 + 浮点点积」。**

```
传统做法（PyTorch 风格）：          GGML 做法：
  load q4_0 block (18B)              load q4_0 block (18B)
  → 反量化成 32 个 float (128B)      load q8_1 block (36B, 激活已量化)
  → FP32 点积 32 次                  → dp4a 整数点积 8 次（4 路 SIMD）
  = 36B 读 + 32 次 FMA               = 54B 读 + 8 次 IADD4
                                     → × d4 × ds8（2 次乘法）
```
**省在哪**：① 权重从不展开成 float（省 4~8 倍权重带宽）；② 整数点积用 `dp4a` 一条指令算 4 个乘积。

---

## 4.1 量化块的内存布局（一切的基础）

> 文件：`src/ggml-common.h`（**不在 ggml-cuda 里**，是 CPU/CUDA 共享的格式定义）

### 4.1.1 两类量化家族

| 家族 | 代表 | block 大小 | 特点 |
|---|---|---|---|
| **legacy** | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | **32** 权重/block | 单 scale（+可选 min），实现简单 |
| **K-quant** | Q2_K ~ Q6_K | **256** 权重/block（`QK_K`） | 分层 scale/min，精度更高 |
| **i-quant** | IQ1_S ~ IQ4_XS | 混合 | 码本查表（codebook），精度/体积极限 |

```c
#define QK_K 256          // ggml-common.h:89   K-quant 的 block 大小
#define QK4_0 32          // :194
#define QK4_1 32
#define QK5_0 32
#define QK8_0 32
#define QK8_1 32
```

### 4.1.2 Q4_0 —— 最简单的量化块（`:195`）

```c
typedef struct {
    ggml_half d;           // delta: 该 block 的 scale（fp16）
    uint8_t qs[QK4_0 / 2]; // 16 字节 = 32 个 4bit 权重（nibble 打包）
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, ...);  // = 18 字节
```

**反量化公式**（`dequantize.cuh` 的 `dequantize_q4_0`）：
```c
v.x = (qs_nibble_low  - 8.0f) * d;    // 注意 -8 偏移！4bit 存的是 [0,15]，实际值域 [-8,7]
v.y = (qs_nibble_high - 8.0f) * d;
```
> **`-8` 偏移是 Q4_0 的关键**：4bit 无符号数 0~15，减去 8 得到有符号 -8~7。Q4_1 则用 `d*q + m` 形式（多了 min）。

### 4.1.3 Q8_1 —— **激活**的量化块（`:259`）★ 理解 MMVQ 的关键

```c
typedef struct {
    union {
        struct { ggml_half d;  // delta: scale
                 ggml_half s;  // d * sum(qs[i])  ← ★ 预计算的块和！
        };
        ggml_half2 ds;         // 打包成 half2，一条指令读两个 half
    };
    int8_t qs[QK8_1];          // 32 个 int8 权重
} block_q8_1;                  // = 36 字节
```

**`s = d * sum(qs)` 是性能关键**：Q4_0 反量化有 `-8` 偏移，展开点积时会出现一个常数项（`8 * d4 * d8 * Σq8`）。ggml **在量化激活时就把这个和算好存进 `s`**，kernel 里就不用再累加：

```c
// vecdotq.cuh 的 vec_dot_q4_0_q8_1_impl 最后一行：
return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
//                 ↑整数点积  ↑d8        ↑常数 × 预存块和 s
```

> **推导要点（建议自己动手验证一遍）**：Q4_0 实际值 = `(q - 8) * d4`，展开后有
> `Σ(q_i-8)·d4·(q8_i·d8) = d4·d8·Σq_i·q8_i − 8·d4·d8·Σq8_i`
> 第一项 = `sumi * d8`；第二项就是靠预存的 `s` 一次乘出来。
> 系数 `8*vdr/QI4_0` 与「一个线程的 `vdr` 覆盖多少个量化值」有关（`vdr == QI4_0` 时恰好覆盖整块，系数 = 8）。**注意 `QI4_0 = 4`（= block_q4_0 的 `qs[]` 里的 int32 个数），不是 8。**
>
> ⚠️ 另有一个易误读点：`quantize_q8_1` 里 `sum` 的初值是 **`xi`（原始 float）而非量化后的 `q`**（`quantize.cu:83`），所以 `s` 严格说是 `Σx`，近似等于 `d·Σq8`（注释里的写法）。

### 4.1.4 `ggml_cuda_type_traits` —— 类型的元数据表（`common.cuh:982`）

```cpp
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;              // 32   block 元素数
    static constexpr int qr = QR4_0;              // 2    打包比（1 字节存几个量化值）
    static constexpr int qi = QI4_0;              // 4    block 的 qs[] 里的 int32 个数
    static constexpr int bs = sizeof(block_q4_0); // 18   block 字节数
};
```
> 定义在 `ggml-common.h:95-130`，公式统一是 `QI = QK / (4*QR)`：
> Q4_0 → `32/(4*2) = 4`（16 字节 qs = 4 个 int32）；Q8_0 → `32/(4*1) = 8`（32 字节 = 8 个 int32）。

**这三个常量贯穿所有量化 kernel**：
- `qk`（block 元素数）→ 算 `blocks_per_row = ncols_x / qk`；
- `qi`（block 的 int32 个数）→ 算 `blocks_per_iter = vdr * nwarps*warp_size / qi`，也是 `vec_dot` 里跳「q8 block 后半」的 int 偏移（`iqs + QI4_0`）；
- `bs`（block 字节数）→ 指针步进。

---

## 4.2 激活量化：`quantize_q8_1`（`quantize.cu:54`）

**MMVQ 的第一步**：把 F32 激活转成 `block_q8_1`。

```cpp
static __global__ void quantize_q8_1(const float * x_ptr, void * vy_ptr, ...) {
    ggml_cuda_pdl_lc();
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) return;
    ...
    block_q8_1 * y = (block_q8_1 *) vy;
    const int64_t ib  = i_cont / QK8_1;   // block 索引
    const int64_t iqs = i_cont % QK8_1;   // block 内索引

    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[...] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max<QK8_1>(amax);   // ★ 一个 warp 处理一个 block（32 元素）
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;                     // scale: max/127
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d); // 量化到 int8

    y[ib].qs[iqs] = q;
    if (iqs > 0) return;
    y[ib].ds = make_half2(d, sum);   // ★ 由 lane 0 写入 d 和 s（= d*sum(qs)）
}
```

**三个设计要点**：
1. **一个 warp 处理一个 32 元素的 block** —— 归约用 `warp_reduce_max/sum`（shuffle，无需 shared mem / `__syncthreads`）；
2. **对称量化**：`d = amax/127`，值域 `[-127,127]`，**无零点偏移**（区别于 Q4_0 的 -8）；
3. **`s` 在量化时就算好**（`sum` 来自 warp 归约），省得点积时再算。

---

## 4.3 量化点积核心：`vecdotq.cuh`（1380 行）★ 全模块最核心

### 4.3.1 三个 `get_int_b*` —— 从字节流取 int（`:7-31`）

```cpp
static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {  // 逐字节
    const uint8_t * x8 = (const uint8_t *) x;
    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    ...
}
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {  // 2 字节对齐
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
}
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {  // 4 字节对齐
    return ((const int *) x)[i32];
}
```
**为什么三种？** 不同量化类型的 `qs` 数组对齐不同（`int8_t[]`/`int16_t[]`/`int32_t[]`）。用错会触发未对齐访问（AMD 上会崩）。**写自定义量化算子时必须按 block 的实际声明选对。**

### 4.3.2 `get_int_from_table_16` —— 4bit 索引查表（`:34`）★ 最精巧的函数

**问题**：i-quant（如 IQ4_NL）不是线性量化，而是**码本查表**——4bit 索引 → 16 个 float 值之一。需要"用 4bit 索引从 16 字节表里取字节"。

**CUDA 的难点**：没有"用 4bit 索引选字节"的指令。只有 `__byte_perm(a, b, sel)` —— **用 3bit 索引**从 8 字节里选字节。

**解法（`:57`）**：
```cpp
const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);   // 低 8 字节，用低 3 位选
    const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);   // 高 8 字节，用低 3 位选
    tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);     // ★ 用第 4 位选高低
}
return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
```
**三次 `__byte_perm` 嵌套模拟 4bit 查表**：
1. 第 1、2 次：分别用**低 3 位**在表的低半和高半里选字节；
2. 第 3 次：用**第 4 位**（`0x88888888`）在这两个结果间选；
3. 最后 2 次：把字节重排成"偶数索引一组、奇数索引一组"的 `int2`（供 dp4a 用）。

> **这是 CUDA 上"无 4bit 查表指令"的经典 workaround**。你以后实现任何码本量化（IQ 系列）都必须过这一关。AMD 侧用 `__builtin_amdgcn_perm` 一次搞定（`:36-54`）。

### 4.3.3 `vec_dot_q4_0_q8_1` —— 量化点积的标准形态（`:774`）

```cpp
#define VDR_Q4_0_Q8_1_MMVQ 2   // MMVQ 每线程处理 2 个 int（8 个权重）
#define VDR_Q4_0_Q8_1_MMQ  4   // MMQ 每线程处理 4 个 int

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);           // 取 4 字节 = 8 个 4bit 权重
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);           // 取对应 4 个 int8
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);   // 再取 4 个
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}
```

**核心 impl（`:118`）**：
```cpp
template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;   // ★ 拆出 4 个低 nibble（掩码广播）
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;   // ★ 拆出 4 个高 nibble

        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);   // 一条指令算 4 个乘积并累加
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);
    // 第二项等效于给每个量化值减 8
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}
```

**`ggml_cuda_dp4a`（`common.cuh:720`）—— 4 路 SIMD 整数点积**：
```cpp
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP)
    c = __builtin_amdgcn_sdot4(a, b, c, false);     // AMD: sdot4
#else
    c = __dp4a(a, b, c);                            // NVIDIA: DP4A (Pascal+)
#endif
    return c;
}
```
**`__dp4a(a, b, c)` = c + Σ(a[i] × b[i]) for i in 0..3**，一条指令 = 4 次乘加。这是量化推理的算力来源。

**为什么 `- 8*vdr/QI4_0 * ds8f.y`？**
Q4_0 存的是无符号 0~15，实际值 = `(q - 8)*d4`。展开：
```
Σ (q_i - 8) * d4 * (q8_i * d8)  =  d4*d8 * [ Σ q_i*q8_i  -  8 * Σ q8_i ]
                                 =  d4 * [ sumi * d8  -  8 * (d8*Σq8) ]
                                                            ↑ 这就是 ds8f.y = bq8_1->s
```
**所以 `s` 字段（= d8*Σq8）不是冗余，是让偏移项变成一次乘法**。没有它就得在 kernel 里累加 q8。

### 4.3.4 各量化类型的 `VDR` 表（性能调参核心）

```c
#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4
#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4
#define VDR_Q8_0_Q8_1_MMVQ 2
...
```
**VDR (vec dot ratio) = 每次调用处理的 int 数**。MMQ 的 VDR 通常是 MMVQ 的 2 倍（因为 MMQ 用 shared memory tile，访存更划算）。

---

## 4.4 MMVQ：量化 GEMV（`mmvq.cu`）★ decode 阶段主力

### 4.4.1 何时用 MMVQ —— `ggml_cuda_should_use_mmvq`（`:318`）

```cpp
bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) return false;
    // k-quants 解码成本高，MVQ 每列都要重解一遍，所以 MMQ 更早胜出
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {   // RTX 4090 实测
        switch (type) {
            case GGML_TYPE_Q2_K: return ne11 <= 4;
            case GGML_TYPE_Q3_K: return ne11 <= 6;
            default:             return ne11 <= MMVQ_MAX_BATCH_SIZE;   // = 8
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {      // RTX 5090 实测
        switch (type) {
            case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: return ne11 <= 5;
            case GGML_TYPE_Q5_K: return ne11 <= 6;
            case GGML_TYPE_Q6_K: return ne11 <= 7;
            default:             return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    ...
}
```

**决策逻辑**：`ne11` = 一次处理的 token 数（batch）。
- `ne11 == 1`（decode 单 token）→ 几乎总是 MMVQ；
- `ne11` 大（prefill）→ 转 MMQ 或 cuBLAS；
- **k-quant 的阈值更低**（4~7），因为解码成本高。

> **`MMVQ_MAX_BATCH_SIZE = 8`**（`mmvq.cuh:3`）—— 全局硬阈值。

### 4.4.2 kernel 结构（`:599`）

```cpp
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr, ...) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);   // ★ 函数指针（编译期常量）

    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;   // ★ 一轮并行处理多少 block
    ...
```

**并行结构**：
```
grid  = (ceil(nrows_x / rows_per_cuda_block), nchannels_dst, nsamples)
block = (warp_size, nwarps)
```
- **每个 warp 负责若干行**（`rows_per_cuda_block`）；
- **warp 内 32 个线程并行切分 K 维**（点积的累加方向）；
- `blocks_per_iter = vdr * nwarps*warp_size / qi` —— 一轮能覆盖的 block 数。

**关键设计：`vec_dot_q_cuda` 是 `constexpr` 函数指针**（`:41` 的 `get_vec_dot_q_cuda`）。虽然形式上是函数指针，但因为是编译期常量，**编译器能内联**。这是"用统一接口支持 25 种量化类型"而不牺牲性能的手段。

### 4.4.3 融合（`has_fusion`）：`MUL_MAT + bias + GLU` 一次算完（`:669`）

```cpp
if constexpr (has_fusion) {
    use_gate      = fusion.gate      != nullptr;
    use_bias      = fusion.x_bias    != nullptr;
    use_gate_bias = fusion.gate_bias != nullptr && use_gate;
    vgate         = fusion.gate;
    ...
}
...
// 写回时做融合（:818）
float result = tmp[j][i];
result += x_biases[j];
if (use_gate) {
    float gate_value = tmp_gate[j][i];
    gate_value += gate_biases[j];
    switch (active_glu) {
        case GGML_GLU_OP_SWIGLU: result *= ggml_cuda_op_silu_single(gate_value); break;
        case GGML_GLU_OP_GEGLU:  result *= ggml_cuda_op_gelu_single(gate_value); break;
        ...
    }
}
dst[j*stride_col_dst + i] = result;
```

**融合的收益**：FFN 层原本是 `MUL_MAT(gate) → MUL_MAT(up) → GLU`，三次 kernel 启动 + 三次读权重。融合后**一次 kernel 读两份权重、算完直接出 GLU 结果**——省一次全局内存往返 + 两次 kernel 启动。

> 融合的**检测**在 `ggml-cuda.cu:4509` 的 `ggml_backend_cuda_graph_optimize`（模块二提过）。**自定义算子想被融合，必须在那里加模式识别。**

### 4.4.4 参数表：按设备/类型调 nwarps（`calc_nwarps`）

```cpp
enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC, MMVQ_PARAMETERS_TURING, MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2, MMVQ_PARAMETERS_RDNA3_0, MMVQ_PARAMETERS_RDNA4, MMVQ_PARAMETERS_GB10
};
// calc_nwarps 的 return 值（GENERIC 表片段）：
case GGML_TYPE_Q4_0: return 6;   case GGML_TYPE_Q4_1: return 6;
case GGML_TYPE_Q4_K: return 5;   case GGML_TYPE_Q6_K: return 4;
case GGML_TYPE_MXFP4: return 4;  case GGML_TYPE_NVFP4: return 4;
default:             return MMVQ_MAX_BATCH_SIZE;
```
**每 (设备族, 量化类型) 组合都有实测最优的 nwarps**——这是 ggml 团队在真机上跑 benchmark 调出来的。**你优化自己的算子时，也应该建这样的参数表。**

### 4.4.5 主机端流程（`:1421` `ggml_cuda_mul_mat_vec_q`）

```cpp
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                             const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, ...) {
    // ── ① 分配 q8_1 缓冲（池）──
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    // ── ② 量化激活 F32 → q8_1 ──
    quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    // ── ③ 启动 MMVQ kernel ──
    mul_mat_vec_q_switch_type(src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ...);
}
```

**完整数据流**：
```
src0 (量化权重, 显存常驻)  ─────────────────┐
                                            ▼
src1 (F32 激活)  →  quantize_q8_1  →  q8_1  →  mul_mat_vec_q kernel  →  dst (F32)
                    (临时池分配)              (vec_dot_q4_0_q8_1 逐 block 点积)
```

> **注意 `ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)`**（`MATRIX_ROW_PADDING = 512`，`common.cuh:186`）：K 维要 padding 到固定对齐，否则量化块跨行错位。**写量化算子必须处理这个 padding**（含清零 padding 区，见 `:1500` 的 `cudaMemsetAsync`）。

### 4.4.6 分发链（4 层 switch）

```
ggml_cuda_mul_mat_vec_q                       :1421   主机入口
  └─ mul_mat_vec_q_switch_type                :1268   switch(src0->type) → 25 个 case
       └─ mul_mat_vec_q_switch_ncols_dst<type> :1076  switch(ncols_dst) → 1/2/4/8
            └─ mul_mat_vec_q_switch_fusion<type, ncols_dst, ...>  :1011  has_fusion?
                 └─ mul_mat_vec_q<type, ncols_dst, has_fusion, ...>  :599  ← kernel
```
**4 层模板分发**：type × ncols_dst × has_fusion × (small_k, halve_iters)。组合爆炸 → 这就是 `template-instances/` 有 121 个文件的原因。

---

## 4.5 MMQ：量化 GEMM（`mmq.cu` / `mmq.cuh`）

### 4.5.1 何时用 MMQ —— `ggml_cuda_should_use_mmq`（`mmq.cu:266`）

```cpp
bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;                              // 强制走 cuBLAS（调试用）
#endif
    switch (type) { ... mmq_supported = true; ... default: mmq_supported = false; }
    if (!mmq_supported) return false;

    // ★ MMQ tile 需要 ≥48KB shared memory，否则回退 BLAS
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    if (smpbo < 48 * 1024) return false;

    if (turing_mma_available(cc)) return true;   // Turing+ 有 tensor core → 直接用

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;   // 无 dp4a：仅 MoE 用
    }
#ifdef GGML_CUDA_FORCE_MMQ
    return true;                                  // 强制 MMQ（调试/老卡）
#endif
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;  // =64
    }
    ... AMD 分支（mfma/wmma 各自的启发式）...
}
```

**决策矩阵（NVIDIA）**：
| 条件 | 结果 |
|---|---|
| `smpbo < 48KB` | **回退 cuBLAS** |
| Turing+ 且编译了 mma | **MMQ（mma 路径）** |
| 无 dp4a 硬件 | 仅 MoE 用 MMQ |
| 有 fp16 mma 硬件 且 `ne11 >= 64` | 回退 cuBLAS |
| 其他 | MMQ（dp4a 路径） |

> **MMVQ vs MMQ 的分界**：`ne11 <= 8` → MMVQ；`8 < ne11 < 64` → MMQ；`>= 64` → cuBLAS（Ampere+）。

### 4.5.2 `block_q8_1_mmq` —— 为 shared memory 重排的激活块（`mmq.cuh:31`）★

```cpp
static constexpr int QK8_1_MMQ  = 4*QK8_1;   // = 128
static constexpr int QK_FP4_MMQ = 2*QK8_1_MMQ;

struct block_q8_1_mmq {
    union {
        float d4[4];    // 1 个 32bit scale / 32 值，存为 d0,d1,d2,d3
        half2 ds4[4];   // 1 个 16bit scale + 1 个 16bit 部分和 / 32 值
        half  d2s6[8];  // 1 个 scale / 64 值 + 1 个部分和 / 16 值
    };
    int8_t qs[QK8_1_MMQ];   // 128 个 int8
};
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1), ...);   // = 4 个 q8_1 合并
```

**注释里的设计说明（原文翻译）**：
> y 的 float 数据被转换成一种**可以直接作为连续块拷贝到 shared memory** 的布局。先按 128 值分组，然后把这些块当作单个数据值**转置**。为避免 shared memory bank conflict，每个块 padding 16 字节——这个 padding 同时也用来存 block scale / 部分和。

**三个要点**：
1. **4 个 q8_1 合并成 1 个**（128 值）→ 减少指针/索引开销；
2. **转置布局** → 配合 `mma` 指令的 tile 要求（A/B 矩阵的排布）；
3. **padding 16B** → 避开 bank conflict（32 个 bank × 4B = 128B 周期）。

### 4.5.3 `ggml_cuda_mmq_config` —— 每个架构的 tile 参数表（`mmq.cuh:165`）

```cpp
struct ggml_cuda_mmq_config {
    ggml_type type;         // src0->type
    int       nthreads;     // 每 block 线程数
    int       occupancy;    // 目标 occupancy
    int       I;            // SRAM tile 宽度（src0->ne[1]/dst->ne[0] 方向）
    int       J;            // SRAM tile 宽度（src1->ne[1]/dst->ne[1] 方向）
    ggml_cuda_mmq_sram_layout sram_layout;
    int       K_vram;       // VRAM tile 长度（src0->ne[0] 方向，逻辑元素）
    bool      stream_k;     // 是否用 stream-k 分解
    bool      fallback;     // 是否需要 src0->ne[1] 方向的越界检查回退
    ...
};
```

**配置表按架构分文件**（`mmq-config-*.cuh`，共 10 个）：

```cpp
#include "mmq-config-pascal-older.cuh"
#include "mmq-config-pascal-dp4a.cuh"
#include "mmq-config-ampere.cuh"
#include "mmq-config-blackwell.cuh"
#include "mmq-config-gcn.cuh"
#include "mmq-config-cdna.cuh"
#include "mmq-config-rdna2.cuh" ... "mmq-config-rdna4.cuh"
```

**Ampere 的 Q4_0 配置示例**（`mmq-config-ampere.cuh:33`）：
```cpp
CASE(GGML_TYPE_Q4_0, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
CASE(GGML_TYPE_Q4_0, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
CASE(GGML_TYPE_Q4_0, 256, 1, 128,  32, ...);
...
CASE(GGML_TYPE_Q4_0, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
//    type            nthreads occ  I    J   sram_layout                  K_vram    stream_k fallback
```
**每个 `(type, J, fallback)` 组合一条 `CASE`** —— 因为 `J`（batch tile）必须编译期确定，所以枚举 8/16/24/32/40/48/64/80/96/112/128 共 11 档。

> **`CASE` 宏里有 `static_assert`**（`mmq.cuh:213`）：
> ```cpp
> static_assert((nthreads_) % 32 == 0 && (nthreads_) <= 512, "bad nthreads");
> static_assert((occupancy_) <= 8, "bad occupancy");
> static_assert((I_) % 32 == 0, "bad I");
> static_assert((J_) % 8 == 0, "bad J");
> static_assert((K_vram_) % 256 == 0, "bad K_vram");
> ```
> **这些约束来自 mma/shared memory 的硬件要求**——写自己的 GEMM 时要遵守同类约束。

### 4.5.4 MMQ 主 kernel 与 tile 流水线（`mmq.cuh:955`）

```cpp
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
static __global__ void mul_mat_q(const char * __restrict__ x, const int * __restrict__ y, ...) {
    // 跳过快编译的未用特化
    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) { NO_DEVICE_CODE; return; }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);
    const uint32_t nty = (nrows_x + I - 1) / I;   // y 方向 tile 数
    ...
    // ── stream-k 或常规 tile 分解 ──
    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
        const int jt = blockIdx.y, it = blockIdx.x;    // tile 坐标
        ...
        mul_mat_q_process_tile<type, J, fallback, fixup>(x, offset_x, y + offset_y, ids_dst_shared, ...);
    }
}
```

**`mul_mat_q_process_tile` 的 K 维流水线（`mmq.cuh`）**：
```cpp
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);   // ① 搬 x tile 进 smem
    { ... 搬 y tile 进 smem ... }
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);                                     // ② 算
    __syncthreads();
    { ... 搬下一半 y tile ... }                                          // ③ 双缓冲
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);                         // ④ 算
    __syncthreads();
}
```
**这是经典的 double-buffering GEMM 结构**：
- `tile_x` / `tile_y` 在 shared memory；
- 每个 K 迭代搬一次 tile，算两次（`vec_dot` 分两半，为的是和 y 的搬运重叠）；
- `__syncthreads()` 保护 shared memory 的读写顺序。

**函数指针三件套**（`mmq.cuh:12-14`）：
```cpp
typedef void (*ggml_cuda_mmq_load_tiles_t)(...);   // 搬 tile
typedef void (*ggml_cuda_mmq_vec_dot_t)(...);      // 算点积（mma 或 dp4a）
typedef void (*ggml_cuda_mmq_write_back_t)(...);   // 写回（含 scale）
```
**load / dot / writeback 三段解耦** → 同一套骨架支持 mma 和 dp4a 两条路径（`BLACKWELL_MMA_AVAILABLE` / `TURING_MMA_AVAILABLE` 条件编译）。

### 4.5.5 stream-k 分解（`mmq.cuh:1242`）

```cpp
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback)/2, 1)
static __global__ void mul_mat_q_stream_k_fixup(...)
```
**问题**：GEMM 的 M×N tile 数可能远小于 SM 数（如 MoE 小 expert），导致大量 SM 空闲。
**stream-k 解法**：把 K 维切开分给多个 block，最后用一个 `fixup` kernel 归并部分和。
- 配置里的 `stream_k` 字段控制是否启用；
- `tmp_fixup` 参数就是部分和缓冲。

> **这是"负载均衡"优化**——当 tile 数不足以喂满 GPU 时，改按 K 维切分。**自定义 GEMM 遇到小矩阵性能差时，优先考虑这个思路。**

---

## 4.6 MMVQ vs MMQ vs cuBLAS：三路对比

| 维度 | MMVQ | MMQ | cuBLAS |
|---|---|---|---|
| **适用** | `ne11 <= 8`（decode） | `8 < ne11 < 64` | `ne11 >= 64`（prefill） |
| **算法** | GEMV（矩阵×向量） | GEMM tile | 高度优化 GEMM |
| **权重处理** | 直接读量化块点积 | 搬进 smem 再点积 | **需先反量化成 FP16** |
| **激活** | 量化成 q8_1 | 量化成 `block_q8_1_mmq` | 保持 FP16 |
| **算力来源** | `__dp4a`（4 路整数） | `mma` PTX / `dp4a` | Tensor Core |
| **显存** | 最低（无中间缓冲） | 中（smem tile + fixup） | 高（FP16 权重副本） |
| **启动开销** | 单 kernel | 单/双 kernel（stream-k） | 多次 cuBLAS 调用 |
| **源码** | `mmvq.cu` | `mmq.cu/cuh` | `ggml_cuda_mul_mat_cublas` |

**为什么 prefill 不走量化 kernel？** GEMM 是**计算密集**的，tensor core 的 FP16 吞吐远高于 dp4a 整数。量化省的是**带宽**（decode 是访存密集），prefill 不差带宽。

---

## 4.7 学习重点 + 易错点

### 必须记住的 8 条

| # | 结论 | 依据 |
|---|---|---|
| 1 | 量化推理核心 = **整数点积 + 两个 scale 相乘**，权重永不展开成 float | `vecdotq.cuh:118` |
| 2 | `block_q8_1.s = d * Σq` 是**预计算的块和**，让 Q4_0 的 -8 偏移项变成一次乘法 | `ggml-common.h:259` + `vecdotq.cuh:136` |
| 3 | `ggml_cuda_dp4a` = 一条指令 4 次乘加（Pascal+ 的 `__dp4a`） | `common.cuh:720` |
| 4 | `VDR`（vec dot ratio）= 每次调用处理的 int 数，MMQ 通常是 MMVQ 的 2 倍 | `vecdotq.cuh:115` |
| 5 | `MMVQ_MAX_BATCH_SIZE = 8`；`MMQ_DP4A_MAX_BATCH_SIZE = 64` | `mmvq.cuh:3` / `mmq.cuh:7` |
| 6 | MMQ tile 需 **≥48KB shared memory**，不够就回退 cuBLAS | `mmq.cu:314` |
| 7 | `block_q8_1_mmq` = 4 个 q8_1 合并 + **转置 + 16B padding 避 bank conflict** | `mmq.cuh:31` |
| 8 | 每个 (架构 × 量化类型 × J) 都有实测调优的配置表 | `mmq-config-*.cuh` |

### 易错点（源码级）

| 易错 | 后果 | 正解 |
|---|---|---|
| 忘记 Q4_0 的 **-8 偏移** | 结果整体偏移，精度崩 | `(q - 8) * d`，或正确使用 `s` 项 |
| `get_int_b1/b2/b4` 选错 | 未对齐访问（AMD 崩） | 按 block 的 `qs` 数组声明类型选 |
| 忘记 `ne10_padded` padding | 量化块跨行错位 | `GGML_PAD(ne10, MATRIX_ROW_PADDING)` + 清零 padding |
| 自己实现 4bit 查表 | 用 `if/switch` 慢 10 倍 | 用 `get_int_from_table_16`（`__byte_perm` 嵌套） |
| 在 MMVQ 里做跨 warp 归约 | 多一次 `__syncthreads` | MMVQ 一个 warp 独立算若干行，用 `warp_reduce_sum` |
| 小矩阵直接上 GEMM | SM 利用率极低 | 考虑 stream-k 分解（`mmq.cuh:1242`） |
| 认为量化 kernel 永远快 | prefill 反而慢 | `ne11 >= 64` 用 cuBLAS |

---

---

## 4.9 补充：K-quant、IQ、MXFP4/NVFP4 与架构专用路径（审计新增）

前文以 Q4_0/Q8_1 为数学标本，但当前 v0.25.3 的核心量化路径还包括三类必须区分的格式。

### 4.9.1 K-quant：256 元素分层 scale

```text
Q2_K / Q3_K / Q4_K / Q5_K / Q6_K
    └─ QK_K = 256
    └─ 多个 sub-block scale/min
    └─ 解码工作量高于 legacy Q4_0
```

它们不能套用 Q4_0 的单一 `d` 和 `-8` 公式。CUDA 侧必须进入对应的：

```text
vec_dot_q*_K_q8_1              vecdotq.cuh
mmq-load-tiles.cuh             MMQ tile 加载
mmq-config-*.cuh               架构/类型/J 配置
```

**工程含义**：K-quant 的表示精度较高，但每次 dot 的 scale 解码更多，所以 `ggml_cuda_should_use_mmvq` 对 Q2_K/Q3_K/Q4_K 等常常设置更低的 batch 阈值。

### 4.9.2 IQ：码本量化

```text
IQ1_S / IQ1_M / IQ2_* / IQ3_* / IQ4_NL / IQ4_XS
    └─ nibble/bit code
    └─ codebook lookup
    └─ 非线性码本值
```

IQ 路径的核心不是 `(q - zero) * d`，而是：

```text
packed code
    ↓
get_int_from_table_16()
    ↓
码本字节重排成 int2
    ↓
dp4a / 专用 vec_dot
```

CUDA、HIP、MUSA 的查表实现不同：

```text
CUDA  → __byte_perm 嵌套
HIP   → __builtin_amdgcn_perm
MUSA  → generic fallback
```

因此为 IQ 类型添加新 kernel 时，不能简单复制 Q4_0 的 `get_int_b2` 逻辑，必须确认 codebook 布局与厂商分支。

### 4.9.3 MXFP4/NVFP4：Blackwell/FP4 路径

当前 traits 和 MMQ 同时包含：

```text
GGML_TYPE_MXFP4
GGML_TYPE_NVFP4
```

区别：

```text
MXFP4 → E8M0 block scale + E2M1 4-bit value
NVFP4 → UE4M3 分组 scale + E2M1 4-bit value
```

相关定义和实现：

```text
ggml-common.h              block_mxfp4 / block_nvfp4
vecdotq.cuh                 vec_dot_mxfp4_q8_1 / vec_dot_nvfp4_q8_1
mmq.cuh                    block_fp4_mmq
mmq-config-blackwell.cuh   Blackwell 配置
```

NVFP4 的 scale 可能参与 MMVQ/MMQ fusion（`mmvq.cu` 的 `x_scale/gate_scale`），不能把它当成普通 Q4 的单一 `d`。

### 4.9.4 编译架构与运行架构是两件事

```text
__CUDA_ARCH__ / ARCHS       → 编译期决定 device code 是否生成
info.devices[id].cc         → host 运行时决定选择哪条路径
GGML_CUDA_FORCE_MMQ         → 强制 heuristic 选择 MMQ
GGML_CUDA_FORCE_CUBLAS      → 强制回退 cuBLAS
```

检查路径前必须同时确认：

```bash
# CMake 架构
grep CMAKE_CUDA_ARCHITECTURES build/CMakeCache.txt
# backend feature
# 运行时读取 ggml_backend_cuda_get_features()
```

**常见假象**：GPU 本身支持某个 MMA/FP4，但二进制没有编译对应 architecture，`ggml_cuda_highest_compiled_arch()` 仍会让路径回退。

### 练习 1（读码验证，1 小时）
1. 手算验证：给定 `d4 = 0.5`、`bq4_0->qs = {0x12, ...}`、`bq8_1 = {d8, s, qs}`，用 `vec_dot_q4_0_q8_1_impl` 的公式算出结果，再用 `dequantize_q4_0` + FP32 点积验证两者相等（注意 Q4_0 的 nibble 是「低 nibble 存前半、高 nibble 存后半」的分裂布局）。
2. 找到 `QI4_0` 的定义（`ggml-common.h:103`，`QK4_0/(4*QR4_0)`），手算出它的值，并解释它为什么等于 `block_q4_0` 的 `qs[]` 里的 int32 个数。
3. 推导 `- (8*vdr/QI4_0) * ds8f.y` 这个修正项：验证 `vdr == QI4_0`（覆盖整块）时系数恰为 8，并解释 `vdr=2` 时为什么是 4。

### 练习 2（决策链追踪，30 分钟）
给定 A100（cc=80, smpbo=163KB）：
- src0 = `Q4_K`，`ne11 = 1` → 走哪条路径？依据哪个函数？（`ggml-cuda.cu:1848~1877`）
- `ne11 = 32` → ？
- `ne11 = 512` → ？
把三条决策链的**函数调用序列**写出来。

### 练习 3（性能实验，需 CUDA）
```bash
# 对比不同 batch size 下的 pp/tg 性能，观察 MMVQ→MMQ→cuBLAS 的切换点
./build/bin/llama-bench -m model-q4_0.gguf -ngl 99 -p 1,8,32,64,128,512 -n 0
# 强制 cuBLAS 对照
# 重新编译时加 -DGGML_CUDA_FORCE_CUBLAS
```
预期：`p=1,8` 附近性能高（MMVQ），`p=32~64` 出现拐点（MMQ），`p>=128` 由 cuBLAS 主导。

### 练习 4（挑战：实现一个量化点积）
仿照 `vec_dot_q4_0_q8_1`，为 **Q4_1**（`d*q + m` 形式，有 min）写一个 `vec_dot_q4_1_q8_1_impl`，然后用 `test-backend-ops` 验证。
提示：`vec_dot_q4_1_q8_1_impl` 已存在于 `vecdotq.cuh:142`，先读懂再自己重写。

---

## 附：模块四文件索引

| 文件 | 精读重点 |
|---|---|
| `ggml-common.h:194/259` | `block_q4_0` / `block_q8_1` 结构（**在 src/，非 ggml-cuda**） |
| `dequantize.cuh` | `dequantize_q4_0` —— -8 偏移的出处 |
| `quantize.cu:54` | `quantize_q8_1` —— 激活量化 + warp 归约 + `s` 预计算 |
| `vecdotq.cuh:34` | `get_int_from_table_16` —— 4bit 查表的三次 `__byte_perm` |
| `vecdotq.cuh:118` | `vec_dot_q4_0_q8_1_impl` —— **全模块最核心的 10 行**（q4_1 版在 :142） |
| `common.cuh:720` | `ggml_cuda_dp4a` |
| `common.cuh:982` | `ggml_cuda_type_traits`（qk/qi/bs） |
| `mmvq.cu:318` | `ggml_cuda_should_use_mmvq` —— 按设备/类型调阈值 |
| `mmvq.cu:599` | `mul_mat_vec_q` kernel |
| `mmvq.cu:1011` | `mul_mat_vec_q_switch_fusion` —— 融合路径 |
| `mmvq.cu:1421` | `ggml_cuda_mul_mat_vec_q` 主机入口 |
| `mmq.cu:266` | `ggml_cuda_should_use_mmq` —— 含 48KB smem 判据（:314） |
| `mmq.cuh:31` | `block_q8_1_mmq` —— 转置 + padding 布局 |
| `mmq.cuh:165` | `ggml_cuda_mmq_config` —— tile 参数 |
| `mmq.cuh:955` | `mul_mat_q` kernel |
| `mmq-config-ampere.cuh` | 实测调优的配置表 |
