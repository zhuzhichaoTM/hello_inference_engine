# 模块六：FlashAttention 三路实现（Vec / Tile / MMA-F16）

> 对标版本：ggml v0.25.3，commit `353b63b4`
> 主要文件：`fattn.cu`(774) / `fattn-common.cuh`(1302) / `fattn-vec.cuh`(609) / `fattn-tile.cuh`(1355) / `fattn-mma-f16.cuh`(2187)
> 核心目标：理解 GGML CUDA 如何把 Q·Kᵀ、softmax、softmax·V 融合进一个 FlashAttention kernel，并能判断 Vec、Tile、MMA-F16 三路何时被选中。

---

## 6.0 FlashAttention 在 GGML 中解决什么问题

普通 Attention：

```text
S = Q × Kᵀ                         // [Q_len, D] × [D, KV_len]
P = softmax(S + mask)              // 显式生成 [Q_len, KV_len]
O = P × V                          // [Q_len, KV_len] × [KV_len, Dv]
```

问题是中间的 `S/P` 矩阵很大：KV cache 越长，显存读写越重。

GGML 的 FlashAttention 路径把三步融合：

```text
Q tile 常驻寄存器/shared memory
       ↓
分块加载 K/V
       ↓
QKᵀ → online softmax → softmax·V
       ↓
只写最终 O + 少量归一化 metadata
```

**核心收益**：不把完整 attention score/probability 矩阵写回 global memory，减少 KV cache 的中间访存。

在 GGML 中入口是：

```text
ggml_cuda_compute_forward()                    ggml-cuda.cu:2372
  └─ ggml_cuda_flash_attn_ext(ctx, dst)         fattn.cu:755
       └─ ggml_cuda_get_best_fattn_kernel()     fattn.cu:559 附近
            ├─ BEST_FATTN_KERNEL_VEC
            ├─ BEST_FATTN_KERNEL_TILE
            └─ BEST_FATTN_KERNEL_MMA_F16
```

---

## 6.1 张量与维度语义

FlashAttention EXT 的输入：

```text
Q   = dst->src[0]
K   = dst->src[1]
V   = dst->src[2]
mask= dst->src[3]（可选）
```

常用维度：

```text
Q: [D_Q, Q_len, n_heads_Q, n_sequences]
K: [D_Q, KV_len, n_heads_KV, n_sequences]
V: [D_V, KV_len, n_heads_KV, n_sequences]
```

其中：

```text
D_Q       = Q->ne[0]       // Query/Key head dimension
D_V       = V->ne[0]       // Value head dimension
Q_len     = Q->ne[1]       // 本次 query token 数
KV_len    = K->ne[1]       // KV cache 长度
GQA ratio = Q->ne[2] / K->ne[2]
```

GGML 允许 `D_Q != D_V`，但只有部分专门的模板组合支持这种情况，例如：

```text
D_Q=192, D_V=128
D_Q=320, D_V=256
D_Q=576, D_V=512
```

`gqa_ratio` 表示多个 Query head 共享一个 K/V head：

```text
Q heads:  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
K/V:      K0 ----------- K1 -----------
             ↑ gqa_ratio = 4
```

这是 FlashAttention 选择宽 tile 的重要依据。

---

## 6.2 三路实现的职责分工

| 路径 | 核心文件 | 典型场景 | 主要数据类型 |
|---|---|---|---|
| **Vec** | `fattn-vec.cuh` | 单 query / 极小 batch、量化 K/V | F16/BF16/Q4/Q5/Q8 |
| **Tile** | `fattn-tile.cuh` | 通用 tile、无 tensor core 或中小 batch | K/V 通常转 F16 |
| **MMA-F16** | `fattn-mma-f16.cuh` | Turing/Volta/AMD MMA，较大 batch/head | K/V F16 + MMA/MFMA/WMMA |

最重要的事实：

- **Vec 是直接支持多种量化 K/V 的路径**，可以在 KQ 阶段直接调用量化点积；
- **Tile 和 MMA-F16 通常需要 F16 K/V**，如果原始 K/V 不是 F16，运行前会安排转换缓冲；
- 不是简单的“GPU 越新越一定用 MMA”：选择还依赖 query 数、GQA、head size、KV 长度、mask、stride 和编译实例。

---

## 6.3 公共数学：online softmax

FlashAttention 不能等全部 `QKᵀ` 结果生成后再 softmax，因此每个 query row 按 K/V tile 逐块更新：

设当前累计状态为：

```text
m_old = 已处理 score 的最大值
l_old = Σ exp(score - m_old)
o_old = Σ exp(score - m_old) · V
```

新来一个 K tile，得到：

```text
S_tile = Q · K_tileᵀ + mask + ALiBi
m_new = max(m_old, max(S_tile))
alpha = exp(m_old - m_new)
beta  = exp(S_tile - m_new)

l_new = alpha * l_old + Σ beta
o_new = alpha * o_old + beta · V_tile
```

最后：

```text
O = o_new / l_new
```

这样只保留：

```text
m / l / o
```

而不需要保存整个 `Q_len × KV_len` 的 softmax 矩阵。

### GGML 的数值稳定细节（`fattn-common.cuh`）

```cpp
#define HALF_MAX_HALF       __float2half(65504.0f/2)
#define SOFTMAX_FTZ_THRESHOLD -20.0f
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)
```

含义：

1. 使用有限的 half 最大值代替直接 `-INFINITY`，避免后续减法产生 NaN；
2. 小于 `-20` 的 exp 结果可以 flush 到 0，减少数值异常；
3. KQ 最大值上移约 `3*log(2)`，扩大 VKQ 累加器的有效动态范围，代价是更小的值可能被 flush。

这不是普通 softmax 的装饰，而是为了适应：

```text
FP16 中间值
+ 长 KV 序列
+ 大 head dimension
+ tensor core 累加
```

---

## 6.4 公共数据与转换缓冲（`fattn-common.cuh`）

### 6.4.1 kernel 函数指针接口（`:25`）

```cpp
typedef void (*fattn_kernel_t)(
    const char * Q,
    const char * K,
    const char * V,
    const char * mask,
    const char * sinks,
    const int * KV_max,
    float * dst,
    float2 * dst_meta,
    const float scale,
    const float max_bias,
    ...);
```

三条路径都被统一成同一套 host launch 参数：

```text
Q/K/V/mask/sinks 指针
KV_max（可变 KV 长度或 sparse 索引辅助）
dst + dst_meta
scale / ALiBi / logit_softcap
各维 ne/nb
```

这使 `fattn.cu` 可以只负责选择 kernel，具体实现隐藏在三个 `.cuh` 中。

### 6.4.2 F16 转换缓冲（`:46-75`）

```cpp
struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};
```

`ggml_cuda_flash_attn_ext_get_f16_extra_data()` 的策略：

```text
dst->data + ggml_nbytes(dst)
       ↓ 128 字节对齐
临时 F16 K
       ↓ 128 字节对齐
临时 F16 V
```

如果 V 是 K 的 view：

```cpp
if (V_is_K_view) {
    data.V = data.K;
}
```

**避免重复转换和重复占用显存**。

`ggml_cuda_flash_attn_ext_get_alloc_size()`（`fattn.cu:719`）会先调用 kernel selector，提前算出需要的尾部空间，让 backend buffer 一次性分配足够空间。

这是 GGML 一个非常重要的工程模式：

```text
先根据最终 kernel 选择结果估算额外 workspace
→ 把 workspace 放在 dst allocation 尾部
→ 不额外调用 cudaMalloc
```

---

## 6.5 Vec 路径：单 query 和量化 KV 的专用实现

### 6.5.1 kernel 结构（`fattn-vec.cuh:21`）

```cpp
template<int D, int ncols,
         ggml_type type_K, ggml_type type_V,
         bool use_logit_softcap>
__launch_bounds__(128, 1)
static __global__ void flash_attn_ext_vec(...)
```

模板参数：

```text
D                  head size
ncols              一个 block 同时处理的 query 列数（1 或 2）
type_K/type_V      K/V 类型
use_logit_softcap  是否编译 softcap 分支
```

Vec 固定 128 threads：

```cpp
static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
}
```

### 6.5.2 KQ 和 V 的线程布局

```cpp
constexpr int nthreads = 128;
constexpr int nthreads_KQ = type_K == F16/BF16 ? 128/cpy_nb : nthreads_KQ_q;
constexpr int nthreads_V  = type_V == F16/BF16 ? 128/cpy_nb : nthreads_V_q;
```

KQ 阶段：

```text
F16/BF16 K → 向量化 half2/bf162 加载
量化 K     → vec_dot_KQ + q8 query 临时量化
```

V 阶段：

```text
V_rows_per_thread = F16/BF16 ? 2*cpy_ne : 4
```

Q/K/V 的指针按 head、sequence、GQA ratio 重定位：

```cpp
const int sequence = blockIdx.z / ne02;
const int head = blockIdx.z - sequence*ne02;
const int gqa_ratio = ne02 / ne12;

Q += nb03*sequence + nb02*head + nb01*ic0;
K += nb13*sequence + nb12*(head / gqa_ratio);
V += nb23*sequence + nb22*(head / gqa_ratio);
```

### 6.5.3 Vec 直接支持量化 K/V

`fattn-common.cuh` 中有一组专用 KQ 点积函数：

```text
vec_dot_fattn_vec_KQ_f16       F16 K
vec_dot_fattn_vec_KQ_bf16      BF16 K
vec_dot_fattn_vec_KQ_q4_0      Q4_0 K
vec_dot_fattn_vec_KQ_q4_1      Q4_1 K
vec_dot_fattn_vec_KQ_q5_0      Q5_0 K
vec_dot_fattn_vec_KQ_q5_1      Q5_1 K
vec_dot_fattn_vec_KQ_q8_0      Q8_0 K
```

以 Q4_0 为例：

```cpp
const int sumi = ggml_cuda_dp4a(v, u, 0);
const float2 Q_ds = ((const float2 *) Q_ds_v)[...];
sum += __half2float(K_q4_0[ib].d) *
       (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
```

这和模块四的 MMVQ 思路一致：

```text
量化 K + q8_1 query
→ dp4a
→ scale + zero-point 修正
```

**因此 Vec 是 FlashAttention 内部连接“量化点积”和“在线 softmax”的桥梁。**

### 6.5.4 Vec 的 host wrapper

```cpp
template <int D, int cols_per_block,
          ggml_type type_K, ggml_type type_V,
          bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(...) {
    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    fattn_kernel_t kernel = flash_attn_ext_vec<...>;
    launch_fattn<D, cols_per_block, 1>(...);
}
```

`Q->ne[1] == 1` 时：

```text
cols_per_block = 1
```

否则：

```text
cols_per_block = 2
```

而 `logit_softcap` 也被提升到模板参数：

```cpp
if (logit_softcap == 0.0f) {
    use_logit_softcap = false;
} else {
    use_logit_softcap = true;
}
```

这样无 softcap 的常见路径可以完全消除相关分支。

---

## 6.6 Tile 路径：通用 shared-memory tile

### 6.6.1 kernel 结构（`fattn-tile.cuh:793`）

```cpp
template<int DKQ, int DV, int ncols1, int ncols2,
         bool use_logit_softcap>
__launch_bounds__(
    ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, ncols1*ncols2),
    ggml_cuda_fattn_tile_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_tile(...)
```

Tile 的关键模板参数：

```text
DKQ       Q/K head size
DV        V head size
ncols1    当前 block 处理的 query columns
ncols2    GQA 中并行处理的 query heads
```

有效 query tile 宽度：

```cpp
constexpr int ncols = ncols1 * ncols2;
```

### 6.6.2 Shared memory 布局

```cpp
Q_tmp   // kernel 生命周期内保存 Q
KV_tmp  // 迭代 KV 时的 K/V tile
KQ      // KQ 分数 tile
VKQ     // 最终 V·KQ 累加器，通常在寄存器
```

典型定义：

```cpp
__shared__ half2 Q_tmp[ncols * DKQ/2];
__shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];
__shared__ half  KQ[ncols * nbatch_fa];
__align__(16) half2 VKQ[...];
```

`DKQp`、`DVp` 会向上对齐到 `2*warp_size`：

```cpp
constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1);
constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1);
```

原因：

```text
warp 协作加载
half2 向量化
避免最后一行维度不足导致越界
```

### 6.6.3 Tile 主循环

```cpp
const int k_VKQ_max = KV_max ? KV_max[...] : ne11;

for (int k_VKQ_0 = ...; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
    flash_attn_tile_iter(...);
}
```

`ncols2 == 1` 时：

```text
需要 OOB 检查
完整 tile 用 oob_check=false
尾 tile 用 oob_check=true
```

`ncols2 > 1` 时，通常可以走无边界检查的批量路径：

```cpp
constexpr bool oob_check = false;
flash_attn_tile_iter<...>(...);
```

这说明 GGML 通过模板把“完整 tile”和“尾 tile”拆开，避免主循环每次都付出边界判断成本。

### 6.6.4 Tile 与 GQA

```cpp
const int gqa_ratio = ne02 / ne12;
const int sequence = blockIdx.z / (ne02/ncols2);
const int head0 = blockIdx.z*ncols2 - sequence*ne02;
```

一个 block 可以处理多个 query head：

```text
ncols2 越大 → GQA 复用 K/V 越充分
但 shared memory / register 压力也越大
```

因此 `ncols2` 不是越大越好，必须由 head size、GQA ratio、GPU 架构共同决定。

---

## 6.7 MMA-F16 路径：Tensor Core / WMMA / MFMA

### 6.7.1 kernel 结构（`fattn-mma-f16.cuh:1807`）

```cpp
template<int DKQ, int DV, int ncols1, int ncols2,
         bool use_logit_softcap,
         bool V_is_K_view,
         bool use_sparse>
__launch_bounds__(
    ggml_cuda_fattn_mma_get_nthreads(...),
    ggml_cuda_fattn_mma_get_occupancy(...))
static __global__ void flash_attn_ext_f16(...)
```

MMA 路径要求 K/V 已经是 F16，并使用：

```text
Volta tensor core
Turing tensor core
AMD WMMA
AMD MFMA
```

kernel 开头会根据编译能力跳过不可能的特化：

```cpp
if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
    NO_DEVICE_CODE;
    return;
}

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
if (ncols1*ncols2 > 32) {
    NO_DEVICE_CODE;
    return;
}
#endif
```

这是一种**模板组合裁剪**：所有组合可以统一生成，但不适合硬件的组合不生成 device code。

### 6.7.2 stream-K 风格的工作切分

MMA kernel 把连续工作空间拆成：

```text
iter_k      KV 方向 tile 数
iter_j      query tile 数
iter_z_gqa  GQA tile 数
```

然后把线性任务空间分配给 CUDA blocks：

```cpp
int kbc = blockIdx.x * total_work / gridDim.x;
int kbc_stop = (blockIdx.x + 1) * total_work / gridDim.x;
```

如果两个 block 的边界落在同一个 output tile 内：

```text
前一个 block 负责开头
后一个 block 负责末尾
dst_meta / fixup buffer 记录中间状态
```

这和模块四 MMQ 的 stream-k 思路相似：**把 KV/K 方向工作切碎，提高 SM 利用率，再做必要的归并。**

### 6.7.3 `V_is_K_view`

```cpp
constexpr bool V_is_K_view = DKQ == 576;
```

对应 DeepSeek 类结构：V 可以是 K 的 view 或共享布局。若 V 是 K 的 view：

```text
不重复分配 V F16 缓冲
V 指针直接复用 K
stride_V = stride_K
```

这是模型布局和 kernel 特化共同决定的内存复用。

### 6.7.4 Sparse FlashAttention

稀疏路径只在 NVIDIA CUDA 上支持：

```cpp
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
```

第一阶段：`flash_attn_mask_to_sparse_indices`（`fattn.cu:12`）

```text
mask
  ↓
每组 query 检查哪些 KV 位置可见
  ↓
indices[count]
```

然后 MMA kernel 只读取这些 KV index：

```cpp
const int * sparse_indices = use_sparse ? KV_max_ptr : nullptr;
```

适用条件由：

```cpp
ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(...)
```

决定，需要满足：

```text
NVIDIA
Turing MMA 可用
有 mask
无 ALiBi
无 logit softcap
mask 覆盖完整 K 长度
KV 足够长
```

**稀疏路径的收益条件**：mask 真正排除了大量 KV，否则额外构建 index 的成本可能超过收益。

---

## 6.8 三路 kernel 的选择器：`ggml_cuda_get_best_fattn_kernel`

位置：`fattn.cu:559` 附近。

### 6.8.1 第一轮合法性检查

```cpp
const int gqa_ratio = Q->ne[2] / K->ne[2];
GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
```

然后检查 head size 和 V size：

```text
D=40/64/72/80/96/112/128/256 → DV 必须等于 DKQ
D=192 → DV=128 + GQA 优化 + gqa_ratio % 8 == 0
D=320 → DV=256 + GQA 优化 + gqa_ratio % 32 == 0
D=512 → DV=512 + GQA 优化
D=576 → DV=512 + GQA 优化
其他 → 不支持
```

再检查：

```cpp
ggml_cuda_fattn_kv_type_supported(K->type)
ggml_cuda_fattn_kv_type_supported(V->type)
```

当前 vector kernel 直接支持的 K/V 类型包括：

```text
F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
```

### 6.8.2 GQA 优化条件

```cpp
bool gqa_opt_applies =
    gqa_ratio >= 2 &&
    mask &&
    max_bias == 0.0f &&
    K->ne[1] % FATTN_KQ_STRIDE == 0;
```

并且所有非量化 tensor 的 stride 要满足 16 字节对齐。

```text
GQA ratio ≥ 2
+ 有 mask
+ 无 ALiBi
+ KV length 对齐 256
+ stride 对齐
```

才使用更宽的 query tile 复用 K/V。

### 6.8.3 Turing+ 选择逻辑

```cpp
if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
    if (can_use_vector_kernel) {
        ... 小 batch 特判 → BEST_FATTN_KERNEL_VEC
    }
    return BEST_FATTN_KERNEL_MMA_F16;
}
```

Ada/Blackwell 上，对于普通非量化、单 query、合适 head size，Vec 可能优于 MMA：

```cpp
cc >= ADA_LOVELACE
&& Q->ne[1] == 1
&& Q->ne[3] == 1
→ 倾向 Vec
```

但如果 GQA 很高、head 很大或 KV 很长，可能改选 MMA sparse path。

### 6.8.4 Volta 选择逻辑

```cpp
if (volta_mma_available(cc)) {
    if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
        return BEST_FATTN_KERNEL_VEC;
    }
    if (Q->ne[1] * gqa_ratio_eff <= 16) {
        return BEST_FATTN_KERNEL_TILE;
    }
    return BEST_FATTN_KERNEL_MMA_F16;
}
```

Volta 上只有足够大的 batch 才值得 MMA；中小 batch 用 Tile。

### 6.8.5 AMD 选择逻辑

```text
AMD MFMA：head size ≤256，batch×GQA 达到阈值才 MMA
AMD WMMA：需要 GQA 优化且 wide tile 有足够算术强度
否则：Vec（极小 batch）或 Tile（通用）
```

### 6.8.6 无 Tensor Core

```cpp
if (can_use_vector_kernel) {
    if (!quantized K/V && Q_len == 1) return VEC;
    if (quantized K/V && Q_len <= 2) return VEC;
}
return TILE;
```

所以 Vec 并不是只服务 Tensor Core GPU，它也是无 tensor core 设备的小 batch fallback。

---

## 6.9 最终执行与 workspace 生命周期

```cpp
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(...)) {
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}
```

分配阶段已经通过：

```cpp
size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst)
```

知道最终要不要 F16 K/V workspace。

```text
best kernel = VEC
  K/V 类型支持 → 可能不需转换
  F32 K/V → 只转换需要的部分

best kernel = TILE/MMA
  K/V 不是 F16 → 分配并转换 F16 K/V
```

这保证了：

```text
backend 分配 tensor buffer
  └─ dst 正文空间
  └─ 对齐后的 K F16 workspace
  └─ 对齐后的 V F16 workspace
```

而不是在每次 kernel 调用时动态 `cudaMalloc`。

---

## 6.10 三路对比总结

| 维度 | Vec | Tile | MMA-F16 |
|---|---|---|---|
| 目标 | 极小 batch / 单 query | 通用 tile | 高吞吐大 tile |
| 线程 | 固定 128 | 配置表决定 | 架构/形状配置决定 |
| K/V | 可直接量化 | 通常 F16 | F16 |
| 中间数据 | registers + 少量 shared | Q/KV/KQ shared | MMA tile + shared/register |
| Tensor Core | 可不依赖 | 可部分依赖 | 强依赖 MMA/MFMA/WMMA |
| GQA | 支持 | 支持 | 宽 tile / sparse 优化 |
| 长 KV | 单 query 很强 | 稳健 | 大 batch 很强 |
| 模板数量 | type × D × K/V | DKQ × DV × ncols | DKQ × DV × ncols × sparse × softcap |
| 主要代价 | 单 query 并行度有限 | shared memory 压力 | 编译组合爆炸、workspace 转换 |

---

## 6.11 自定义 FlashAttention/Attention kernel 的开发模板

### 第一阶段：先做 Vec 参考版本

```text
1. 只支持 Q/K/V = F16
2. 只支持 D=64/128
3. 只支持 ncols1=1
4. 只支持无 mask、无 ALiBi、无 softcap
5. 在线 softmax + V 累加
6. 与 CPU 参考逐元素比对
```

### 第二阶段：加入 mask 与 GQA

```text
1. 将 mask stride 从 op tensor 读取
2. 添加 gqa_ratio = Q->ne[2] / K->ne[2]
3. 验证多 Query head 共享 K/V
4. 处理 KV cache 未填满的 KV_max
```

### 第三阶段：加入 Tile

```text
1. Q/K/V tile 放 shared memory
2. KQ tile 做块内归约
3. 通过 online softmax 更新 O
4. 对完整 tile 和尾 tile 使用两个模板分支
5. 加入 shared memory bank conflict 检查
```

### 第四阶段：加入 MMA

```text
1. 把 Q/K/V layout 转成 mma 要求的 fragment
2. 配置 nthreads / occupancy / nbatch
3. 为不同 DKQ/DV/ncols 建配置表
4. 处理 block seam / fixup
5. 最后再加入 sparse gather
```

**不要一开始就写 MMA 版本**：先用 Vec 验证数学，再用 Tile 验证 tile 生命周期，最后接 MMA。否则错误可能同时来自数值公式、layout、shared memory 和 warp mapping。

---

## 6.12 调试与性能分析

### 正确性检查

```text
1. Q_len = 1 与 Q_len = 2
2. KV_len = 1、255、256、257
3. mask = nullptr / F16 mask / 不同 stride mask
4. GQA ratio = 1 / 2 / 4 / 8
5. D_Q = D_V 与 D_Q != D_V
6. logit_softcap = 0 与非 0
7. K/V = F16 与 Q4_0
```

### 路径强制/观察

在 `ggml_cuda_get_best_fattn_kernel` 临时打印：

```cpp
GGML_LOG_DEBUG("fattn: cc=%d DQ=%" PRId64 " DV=%" PRId64
               " Qlen=%" PRId64 " KVlen=%" PRId64
               " gqa=%d kernel=%d\n",
               cc, Q->ne[0], V->ne[0], Q->ne[1], K->ne[1], gqa_ratio, kernel);
```

### Nsight Compute 重点

```text
Vec：global load efficiency、DP4A throughput、register pressure
Tile：shared memory throughput、bank conflict、occupancy
MMA：tensor core utilization、shared load pipe、stall barrier、fixup kernel
```

### 端到端测试矩阵

```text
模型：普通 MHA / GQA / 长上下文 / DeepSeek 类 MLA 结构
Q_len：1, 2, 8, 32, 128, 512
KV_len：256, 2048, 8192, 32768, 65536
K/V：F16、Q4_0、Q8_0
指标：tokens/s、kernel ms、显存、转换时间、稀疏索引时间
```

---

## 6.13 模块六必背结论

1. FlashAttention 的核心是 **QKᵀ + online softmax + softmax·V 融合**，不落完整 score/probability 矩阵。
2. Vec、Tile、MMA-F16 是三套不同的并行布局，不是同一个 kernel 的简单开关。
3. Vec 直接支持多种量化 K/V；Tile/MMA 通常先转换到 F16。
4. `ggml_cuda_get_best_fattn_kernel` 同时看 GPU 架构、head size、Q_len、KV_len、GQA、mask、stride、K/V 类型。
5. GQA 的意义不仅是模型语义，也能提高一个 K/V tile 被多个 Query head 复用的算术强度。
6. `ncols1*ncols2` 是 FlashAttention 中最重要的 tile 宽度参数之一。
7. `DKQ/DV/ncols1/ncols2/use_sparse/use_logit_softcap` 造成模板组合爆炸，所以通过模板实例文件拆分编译。
8. workspace 不是每次 kernel 临时申请，而是由 `get_alloc_size` 提前计算并放在 dst allocation 尾部。
9. 稀疏 FlashAttention 只有在 mask 真正排除大量 KV 时才有收益，否则 compact mask 成本可能抵消收益。
10. 自定义实现顺序应该是：Vec 数学验证 → Tile shared memory → MMA tensor core → sparse gather。

---

## 6.14 配套练习

### 练习 1：手算 online softmax
给两个 KV tile，手算：

```text
m_old、l_old、o_old
m_tile、l_tile、o_tile
m_new、l_new、o_new
```

验证最终结果与一次性 softmax 的结果一致。

### 练习 2：路径选择
对以下配置判断最终路径：

```text
A. Ada，D=128，Q_len=1，KV_len=4096，F16 K/V，GQA=1
B. Ada，D=128，Q_len=32，KV_len=4096，F16 K/V，GQA=1
C. Turing，D=128，Q_len=1，Q4_0 K/Q8_0 V
D. Volta，D=128，Q_len=8，GQA=4
E. NVIDIA，D_Q=576，D_V=512，GQA=20，KV_len=65536
```

要求写出：

```text
合法性检查
GQA 优化是否成立
can_use_vector_kernel 是否成立
最终 BEST_FATTN_KERNEL
是否需要 F16 转换 workspace
```

### 练习 3：mask 稀疏收益
比较：

```text
mask 保留 100% KV
mask 保留 25% KV
mask 保留 5% KV
```

记录：

```text
compact mask 时间
sparse MMA kernel 时间
dense MMA kernel 时间
总时间
```

### 练习 4：自定义 D=64 Vec kernel
只支持：

```text
Q/K/V = F16
D=64
Q_len=1
无 mask、无 ALiBi、无 softcap
```

完成：

```text
kernel
host wrapper
best kernel selector 注册
CPU 对照
Q_len=1、KV_len=1/256/4096 正确性测试
```

---

---

## 6.15 补充：FlashAttention 构建开关、类型实例与 fallback 边界（审计新增）

### 6.15.1 编译开关

`src/ggml-cuda/CMakeLists.txt` 会根据：

```text
GGML_CUDA_FA
GGML_CUDA_FA_ALL_QUANTS
GGML_CUDA_FA_QUANTS
```

决定是否编译 FlashAttention 以及哪些量化 K/V 模板实例。`template-instances/fattn-vec-instance-*.cu` 不是逻辑代码，而是把 `D × type_K × type_V` 组合显式实例化。

如果运行时出现：

```text
no FlashAttention vector kernel compiled for K/V types ... converting K and V to f16 instead
```

说明不是 kernel 数学不支持，而是该 `(D, K type, V type)` 组合没有编译进二进制。解决方案是配置 `GGML_CUDA_FA_QUANTS` 并重新 CMake 配置/编译。

### 6.15.2 支持边界与 fallback

`ggml_cuda_flash_attn_ext_supported()`（`fattn.cu:772`）直接复用 best-kernel selector：

```cpp
return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
```

因此以下任一条件不满足，CUDA device 会声明不支持该 op，scheduler 可以选择其他 backend：

```text
不支持的 head size / DV
不支持的 mask 形状
Q/K/V batch/head 不满足 GQA 关系
K/V 类型没有实例且无法合法转换
stride/layout 不满足模板要求
FLASH_ATTN_AVAILABLE 未启用
```

这和“运行时选 Vec/Tile/MMA”是两层逻辑：

```text
supported() == false → scheduler 层面不接管
supported() == true  → CUDA 内部再选三路 kernel
```

### 6.15.3 dst metadata 与尾部 workspace

FlashAttention 除了最终 `dst`，还可能使用 `float2 * dst_meta` 保存 online softmax/fixup 所需的中间元数据。MMA 的跨 block seam 需要它处理：

```text
一个 output tile 被多个 block 分段处理
    ↓
前段/后段分别写 partial 状态
    ↓
fixup / combine
    ↓
最终归一化输出
```

所以不能只验证最终输出指针；自定义路径必须检查：

```text
dst allocation 是否包含 get_alloc_size 返回的尾部
dst_meta 对齐是否正确
MMA 分段时是否存在 data race
```

| 文件/位置 | 精读重点 |
|---|---|
| `ggml-cuda.cu:2372` | `GGML_OP_FLASH_ATTN_EXT` 总分发 |
| `fattn.cu:12` | sparse mask → indices kernel |
| `fattn.cu:106` | compact mask host 入口 |
| `fattn.cu:155` | MMA ncols1 选择 |
| `fattn.cu:194` | MMA ncols2/GQA 选择 |
| `fattn.cu:496` | Vec dispatch |
| `fattn.cu:559` 附近 | `ggml_cuda_get_best_fattn_kernel` |
| `fattn.cu:719` | workspace 预估 |
| `fattn.cu:755` | 最终三路 dispatch |
| `fattn.cu:772` | backend supports 判断 |
| `fattn-common.cuh:25` | 统一 kernel 参数接口 |
| `fattn-common.cuh:46` | F16 K/V workspace 规划 |
| `fattn-vec.cuh:21` | Vec kernel |
| `fattn-vec.cuh:532` | Vec host wrapper |
| `fattn-tile.cuh:793` | Tile kernel |
| `fattn-tile.cuh:1323` | Tile host case |
| `fattn-mma-f16.cuh:1807` | MMA-F16 kernel |
| `fattn-mma-f16.cuh:2017` | MMA-F16 host case |
| `fattn-mma-f16.cuh:2010` 附近 | sparse / fixup / tile launch 参数 |
