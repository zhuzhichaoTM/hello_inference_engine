# 模块五：`mul_mat` 全路径分发链深挖

> 对标版本：ggml v0.25.3，commit `353b63b4`
> 主文件：`src/ggml-cuda/ggml-cuda.cu`
> 关联文件：`mmvf.cu` / `mmf.cu` / `mmvq.cu` / `mmq.cu` / `convert.cu`
> 核心目标：看到一个 `GGML_OP_MUL_MAT`，能准确判断它为什么走 MMVF、MMF、MMVQ、MMQ 或 cuBLAS。

---

## 5.0 先建立总图：`MUL_MAT` 不是一个 kernel

GGML-CUDA 的矩阵乘是一个**运行时决策树**，不是单一实现：

```text
GGML_OP_MUL_MAT
      │
      ▼
ggml_cuda_mul_mat(ctx, src0, src1, dst)                 ggml-cuda.cu:1824
      │
      ├─ Hadamard hint + FWHT 成功                         :1827
      ├─ 坏 padding/view，或输入/输出不是 F32               :1837-1841
      │       └─ cuBLAS fallback
      │
      ├─ should_use_mmvf(...)                              :1845
      │       └─ ggml_cuda_mul_mat_vec_f                    mmvf.cu:634
      │
      ├─ 转置向量特殊路径（ne01==1）                        :1851-1863
      │       └─ 交换 src0/src1 后仍走 MMVF
      │
      ├─ should_use_mmf(...)                               :1865
      │       └─ ggml_cuda_mul_mat_f                        mmf.cu:13
      │
      ├─ should_use_mmvq(...)                              :1869
      │       └─ ggml_cuda_mul_mat_vec_q                     mmvq.cu:1421
      │
      ├─ should_use_mmq(...)                               :1873
      │       └─ ggml_cuda_mul_mat_q                         mmq.cu:85
      │
      └─ cuBLAS                                             :1877
              └─ ggml_cuda_mul_mat_cublas
```

**最重要的阅读结论**：判断顺序本身就是性能策略。先检查专用快速路径，再落到通用路径；**MMVF/MMF 排在量化 MMVQ/MMQ 前面**，因为 `src0` 不一定是量化类型。

---

## 5.1 张量语义：`src0`、`src1`、`dst` 到矩阵维度

`GGML_TENSOR_BINARY_OP_LOCALS` 会把张量维度和 stride 展开到局部变量。典型矩阵乘：

```text
src0（权重 A）: [ne00, ne01, ne02, ne03]
src1（激活 B）: [ne10, ne11, ne12, ne13]
dst             : [ne0,  ne1,  ne2,  ne3]
```

通常满足：

```text
ne10 == ne00          K 维相等
ne0  == ne01          dst 的行/M 维来自 src0 的列维
ne1  == ne11          dst 的列/N 维来自 src1 的列维
```

可以把计算理解为：

```text
A = src0，逻辑尺寸 K × M
B = src1，逻辑尺寸 K × N
C = dst，  逻辑尺寸 M × N
C = Aᵀ × B
```

但 GGML 支持 batch/head 维度和 broadcast，所以不能只用裸指针相乘；每条路径都要处理 `nb[1..3]`、`ne[2..3]` 和 batch ratio。

**进入 CUDA 快速路径前的共同约束**（`ggml_backend_cuda_device_supports_op` :5195 附近）：

```cpp
if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
    return false;  // K 方向不是连续元素，CUDA 当前路径不支持
}
if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
    return false;
}
```

这解释了一个常见现象：**即使 GPU 有对应 kernel，只要 tensor 的最内层 stride 不满足约束，scheduler 也不会把 op 放到 CUDA。**

---

## 5.2 第一层：进入 `ggml_cuda_mul_mat` 前的硬 fallback

### 5.2.1 Hadamard hint（`:1827`）

```cpp
const int32_t hint = ggml_get_op_params_i32(dst, 1);
if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
    return;
}
```

某些表面上是 `MUL_MAT` 的节点，实际是 Hadamard 变换。GGML 用 `op_params` 传 hint，在矩阵乘决策之前旁路到 FWHT。**自定义优化 hint 必须早于普通路径判断**，否则会被错误地当成 GEMM。

### 5.2.2 坏 padding/view（`:1837-1841`）

```cpp
const bool bad_padding_clear =
    ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
    ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
    src0->view_src;

if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
    return;
}
```

为什么 view + padding 会强制 cuBLAS？

- MMVQ/MMQ 可能需要清理 temporary compute buffer 尾部 padding；
- 如果 `src0` 是另一个 tensor 的 view，清理尾部可能覆盖 view 背后的有效数据；
- 因此专用量化 kernel 不能安全执行，选择 cuBLAS 路径。

**排错重点**：看到量化矩阵没有走 MMVQ/MMQ，先查 `bad_padding_clear`，不要马上怀疑 heuristic。

---

## 5.3 MMVF：F32/F16/BF16 的向量矩阵乘

### 5.3.1 入口条件（`:1845`）

```cpp
if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
    return;
}
```

MMVF 的定位是：**矩阵乘中 N 很小，GEMV 比完整 GEMM 更合适**，尤其是 decode 或 attention 中的 thin matrix。

### 5.3.2 `should_use_mmvf` 的硬条件（`mmvf.cu:792`）

```cpp
if (src0_ne[0] % 2 != 0) return false;       // K 至少按 2 对齐
if (src0_nb[0] != ggml_type_size(type)) return false;

for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
    if (src0_nb[i] % (2*ts) != 0) return false; // half2/float2 对齐
}
```

然后按硬件和类型选择 N 阈值：

```text
F32 + NVIDIA Ampere MMA       → ne11 <= 3
F32 + NVIDIA Turing           → ne11 <= 4
F16 + NVIDIA Ampere           → src0_small && ne11 == 1
F16 + NVIDIA Ada              → src0_small && ne11 <= 4
F16 + AMD RDNA3               → ne11 <= 3
```

**为什么 F16 Ampere 反而只接受单列？** Ampere 的完整矩阵路径/张量核心在批量稍大时更有优势，MMVF 只在小矩阵或单 token 场景胜出。

### 5.3.3 MMVF 特殊转置路径（`:1851-1863`）

```cpp
if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
        && src0->type == GGML_TYPE_F32
        && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
        && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, 1)) {
    ggml_tensor dst_vec = *dst;
    dst_vec.ne[0] = ne11;
    dst_vec.ne[1] = 1;
    ...
    ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
    return;
}
```

这是一个**转置向量技巧**：当 `src0` 只有一行但 `src1` 有很多列时，交换两个输入，把问题转换成 MMVF 可高效处理的向量乘。这里只允许 contiguous tensor，因为它直接构造了一个临时的 `dst_vec` 视图。

---

## 5.4 MMF：F16/BF16/F32 的专用矩阵 tile 路径

### 5.4.1 入口（`:1865`）

```cpp
if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size,
                             src0->ne, src0->nb, ne11,
                             /*mul_mat_id=*/false)) {
    ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
    return;
}
```

MMF 不是量化 GEMM，它是 GGML 自己写的 floating-point matrix kernel（`mmf.cu:13`）。定位是：**矩阵规模较小/较窄、cuBLAS 的通用调度开销或矩阵形状不划算时，用定制 tile。**

### 5.4.2 `should_use_mmf` 的筛选条件（`mmf.cu:133`）

```cpp
if (ggml_is_quantized(type)) return false;
const size_t ts = ggml_type_size(type);
if (src0_ne[0] % (warp_size * (4/ts)) != 0) return false;
if (src0_nb[0] != ts) return false;
for (i = 1; i < GGML_MAX_DIMS; ++i) {
    if (src0_nb[i] % (2*ts) != 0) return false;
}
if (src0_ne[1] % mmf_get_rows_per_block(cc) != 0) return false;
```

这些检查分别保证：

1. K 维能被 warp 的向量化工作量整除；
2. 最内层元素连续；
3. 各维 stride 满足 `half2` / `nv_bfloat162` / `float2` 对齐；
4. M 维能完整切成该架构的 block 行数。

### 5.4.3 MMF 适用规模

```text
普通 MUL_MAT：src1_ncols > 16 → 通常拒绝 MMF
MUL_MAT_ID：小矩阵时允许到 128/512，具体取决于 src0_ne[1]
```

硬件能力最后决定是否启用：

```cpp
case GGML_TYPE_F32:
    return ampere_mma_available(cc) || amd_mfma_available(cc);
case GGML_TYPE_F16:
    return volta_mma_available(cc) || turing_mma_available(cc) ||
           amd_wmma_available(cc) || amd_mfma_available(cc);
case GGML_TYPE_BF16:
    return ampere_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
```

**MMF 的关键优化点**：向量化加载 + 固定行 tile + MMA/MFMA。它和 MMVQ/MMQ 的共同点是“形状专用”，不同点是输入不需要量化。

---

## 5.5 MMVQ：量化 GEMV

当 MMVF/MMF 都不适用后，才检查：

```cpp
if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);   // mmvq.cu:1421
    return;
}
```

典型条件：

```text
src0 是量化类型
ne11 较小（默认上限 8，具体由架构/量化类型调整）
src1 / dst 是 F32
src0 没有不能安全清理的 padding view
```

主机端做三步：

```text
1. ctx.pool() 分配 q8_1 临时缓冲
2. quantize_row_q8_1_cuda：F32 激活 → q8_1
3. mul_mat_vec_q_switch_type：按 src0->type 实例化量化点积 kernel
```

**重要**：MMVQ 的“向量”不是直接用 F32 激活点积，而是先把激活量化为 q8_1，再和 q4/q5/K-quant 做整数点积。详细数学见模块四。

---

## 5.6 MMQ：量化 GEMM

```cpp
if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts=*/0)) {
    ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);       // mmq.cu:85
    return;
}
```

MMQ 适用于量化权重 + 中等 batch。它的决策还要检查：

```cpp
1. src0 类型是否在 MMQ 支持列表
2. shared memory 是否至少 48 KiB（mmq.cu:314）
3. 当前 GPU 是否有 mma/dp4a 等硬件
4. 当前 batch 是否值得使用量化 tile
5. 是否编译了对应架构代码
```

MMQ 的激活不是简单的 `block_q8_1`，而是重排成 `block_q8_1_mmq`，合并、转置并 padding 后放进 shared memory。详细布局见模块四 `mmq.cuh:31`。

---

## 5.7 cuBLAS fallback：通用但不是“免费”的路径

### 5.7.1 入口与 compute type 选择（`:1623`）

```cpp
static void ggml_cuda_mul_mat_cublas(...) {
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    }
    ...
    if (dst->op_params[0] == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }
```

量化 src0 不能直接交给普通 cuBLAS GEMM，因此 cuBLAS 路径会选择 F16/F32 作为计算类型，并通过 `convert.cu` 的转换函数把输入转成对应格式。

可以用环境变量强制选择：

```bash
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32   # 或 fp32
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16   # 或 fp16
GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16
GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto
```

### 5.7.2 临时转换和 RAII

```cpp
ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());
```

如果输入类型已经等于 compute type，直接使用原指针；否则通过：

```cpp
traits::convert(...)       // contiguous / contiguously allocated
traits::convert_nc(...)    // non-contiguous
```

把输入转成 F32/F16/BF16 临时缓冲。函数返回后由 pool RAII 自动回收。

### 5.7.3 四种 cuBLAS 调用形态（`:1560` 附近）

```text
1. F32 + 单 batch：
   cublasSgemm

2. F16/BF16/F32 + 单 batch：
   cublasGemmEx

3. 无 broadcast 且跨 batch 连续：
   cublasGemmStridedBatchedEx

4. 普通 broadcast / 非连续 batch：
   先 k_compute_batched_ptrs 构造指针数组
   再 cublasGemmBatchedEx
```

**选择逻辑**：

```cpp
if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
    cublasSgemm(...);
} else if (ne12 == 1 && ne13 == 1) {
    cublasGemmEx(...);
} else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
    cublasGemmStridedBatchedEx(...);
} else {
    k_compute_batched_ptrs<<<...>>>(...);
    cublasGemmBatchedEx(...);
}
```

### 5.7.4 输出处理

F16/BF16 计算可能先写入临时 `dst_temp`，最后：

```cpp
const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(...);
to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
```

所以 `dst` 默认仍然是 F32，但内部计算类型可以是 F16/BF16。**精度控制和存储类型是两个不同概念。**

---

## 5.8 `MUL_MAT_ID`：MoE 专用的另一棵树

`GGML_OP_MUL_MAT_ID` 的 `src[2]` 是 expert/token id。入口：`ggml_cuda_mul_mat_id`（约 `:1884` 起）。

### 5.8.1 快速路径

```text
src1/dst F32 且 batch <= MMVQ_MAX_BATCH_SIZE
  ├─ 量化 src0 且 batch <= get_mmvq_mmid_max_batch
  │    └─ MMVQ + ids
  └─ 非量化 + AMD
       └─ MMVF + ids

否则：
  should_use_mmq(..., n_experts=ne02) → MMQ + ids
  should_use_mmf(..., mul_mat_id=true) → MMF + ids
```

这里传入 `n_experts=ne02` 很重要：**MoE 场景会改变 MMQ heuristic**。即使普通 dense GEMM 不适合 MMQ，专家矩阵数量大时，MMQ 可能因为避免 cuBLAS 的同步/调度开销而获益。

### 5.8.2 慢 fallback：排序 + 一次性计算

当所有专用路径都不满足时，代码会：

```text
1. 把 ids 从 Device 拷到 Host（:1980 附近）
2. Host 端按 expert 排序 token
3. src1 按 expert 排序到 src1_sorted
4. 对每个 expert 做连续矩阵乘
5. 用 ids_from_sorted 把结果写回原 token 顺序
```

这个路径包含：

```cpp
cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

所以源码明确注释：**这个路径不应在 CUDA Graph capture 时到达，因为它需要 stream synchronization。**

`ggml_cuda_mul_mat_id_needs_sync`（`:1882`）就是在执行前判断是否会进入这个慢路径，用于 CUDA Graph 兼容性处理。

---

## 5.9 完整决策表

| 顺序 | 函数 | 典型输入 | 结果 |
|---|---|---|---|
| 0 | Hadamard hint | 特殊 `op_params` | FWHT |
| 1 | `bad_padding_clear` / 类型不符 | view + padding，或非 F32 I/O | cuBLAS |
| 2 | `should_use_mmvf` | F32/F16/BF16，小 N/薄矩阵 | MMVF |
| 3 | 转置向量特例 | `ne01==1`、大 N | 交换输入后 MMVF |
| 4 | `should_use_mmf` | 非量化、对齐、M/N 较小、硬件 MMA | MMF |
| 5 | `should_use_mmvq` | 量化、N 很小（decode） | MMVQ |
| 6 | `should_use_mmq` | 量化、中等 N、smem≥48KB | MMQ |
| 7 | fallback | 以上均不适合 | cuBLAS |

**注意：这不是固定的“batch 1/8/64”常数表。** 最终阈值依赖：

```text
GPU 架构 cc
量化类型
src0 形状/stride
src1 N 维
是否 MUL_MAT_ID / MoE
是否有 tensor core / dp4a / mfma
是否编译了相应架构
shared memory 上限
```

---

## 5.10 三个具体例子

### 例 1：A100 + Q4_K + decode 单 token

```text
src0 = Q4_K，src1 = F32，dst = F32，ne11=1
→ 非 bad padding
→ MMVF：src0 是量化类型，false
→ MMF：量化类型，false
→ MMVQ：量化 + ne11 很小，true
→ quantize_q8_1 + mmvq.cu
```

### 例 2：A100 + Q4_K + prefill 512 tokens

```text
src0 = Q4_K，src1 = F32，dst = F32，ne11=512
→ MMVQ：false（超过小 batch 阈值）
→ MMQ：可能 false/true，取决于 GPU 编译架构与量化配置
→ 若 MMQ heuristic 拒绝 → cuBLAS
→ cuBLAS 路径先把量化权重转换为 F16/F32，再 GEMM
```

### 例 3：F16 attention thin matrix + batch 1

```text
src0 = F16，src1 = F32，dst = F32，ne11=1
→ MMVF：检查 F16 对齐、K 偶数、src0 形状
→ Ampere + 小矩阵时 true
→ mmvf.cu 的向量 kernel
```

---

## 5.11 调试路径选择

### 源码级打印点

在 `ggml_cuda_mul_mat` 中临时加入：

```cpp
GGML_LOG_DEBUG("mul_mat: type=%s ne00=%" PRId64 " ne01=%" PRId64
               " ne11=%" PRId64 " cc=%d\n",
               ggml_type_name(src0->type), ne00, ne01, ne11, cc);
```

在每个分支前打印：

```cpp
GGML_LOG_DEBUG("mul_mat path: MMVF/MMF/MMVQ/MMQ/cuBLAS\n");
```

### 强制对照

```bash
# 强制 cuBLAS（重新配置编译）
cmake -S . -B build -DGGML_CUDA_FORCE_CUBLAS=ON

# 强制 MMQ（适合检查 MMQ 正确性/性能）
cmake -S . -B build -DGGML_CUDA_FORCE_MMQ=ON

# 指定 cuBLAS 内部计算类型
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 ./your_binary ...
```

### 性能实验矩阵

```text
固定：模型、GPU、线程、上下文长度、重复次数
变化：p = 1, 4, 8, 16, 32, 64, 128, 512
观察：
  p 小：MMVQ/MMVF
  p 中：MMQ/MMF
  p 大：cuBLAS
```

用 Nsight Systems 看路径之间的 kernel 数量，用 Nsight Compute 看：

```text
MMVQ：global memory throughput、dp4a 吞吐、warp occupancy
MMQ：shared memory bank conflict、tensor core utilization、occupancy
cuBLAS：Tensor Core utilization、workspace、转换 kernel 时间
```

---

## 5.12 模块五必背结论

1. `MUL_MAT` 是**运行时决策树**，不是一个固定 kernel。
2. 判断顺序是：**特殊 hint → 硬 fallback → MMVF → 转置特例 → MMF → MMVQ → MMQ → cuBLAS**。
3. MMVF/MMF 处理非量化 F32/F16/BF16；MMVQ/MMQ 处理量化权重；cuBLAS 是通用兜底。
4. `supports_op` 只回答“CUDA 能不能接这个 op”；`should_use_*` 决定“接了以后具体走哪条 kernel 路径”。
5. `src0->nb[0]`、各维 stride 对齐和 `ne11` 是路径选择的三大输入。
6. MMVQ 是 decode 小 batch 主力；MMQ 是中等 batch 量化 GEMM；cuBLAS 通常负责大 batch/不规则形状。
7. `MUL_MAT_ID` 额外引入 expert id、排序、回写和 CUDA Graph 同步约束。
8. 强行改变 heuristic 前，必须同时测：正确性、单 kernel 延迟、端到端吞吐、临时显存。

---

## 5.13 配套练习

### 练习 1：手工追踪三条路径
对以下输入逐项判断每个 `should_use_*`：

```text
A. Q4_0 × F32，ne11=1，A100
B. Q4_0 × F32，ne11=32，A100
C. F16 × F32，ne11=1，src0_ne[1]=4096
D. F32 × F32，src0 是 view 且有 padding
E. MUL_MAT_ID，Q4_K，ne2=4，ne12=128，ne02=64
```

要求写出：最终路径、涉及的临时缓冲、是否可能同步、是否兼容 CUDA Graph。

### 练习 2：修改阈值并测量
修改 `mmvq.cu:318` 附近的一个类型阈值：

```text
原始阈值 → 新阈值 → llama-bench → 结果是否改善
```

不要只看 kernel 时间，要看完整 decode `tg` 吞吐。

### 练习 3：识别 cuBLAS 转换成本
对量化模型分别记录：

```text
convert kernel 时间
cuBLAS GEMM 时间
输出转换时间
总时间
```

用 `nsys profile` 检查“cuBLAS 更快”是否被量化→F16 转换成本抵消。

### 练习 4：MoE 路径
比较 `n_experts=0` 与 `n_experts=64` 传给 `ggml_cuda_should_use_mmq` 后的差异，理解为什么 MoE 的 MMQ 阈值不同。

---

## 附：模块五文件索引

| 文件/位置 | 重点 |
|---|---|
| `ggml-cuda.cu:1824` | `ggml_cuda_mul_mat` 总决策树 |
| `ggml-cuda.cu:1410` | cuBLAS GEMM 实现主体 |
| `ggml-cuda.cu:1623` | cuBLAS compute type 选择入口 |
| `ggml-cuda.cu:1882` | `MUL_MAT_ID` 是否需要同步 |
| `ggml-cuda.cu:1903` | `MUL_MAT_ID` 快速路径 |
| `mmvf.cu:792` | MMVF heuristic |
| `mmvf.cu:634` | MMVF 主机入口 |
| `mmf.cu:133` | MMF heuristic |
| `mmf.cu:13` | MMF 主机入口 |
| `mmvq.cu:318` | MMVQ heuristic |
| `mmvq.cu:599` | MMVQ kernel |
| `mmvq.cu:1421` | MMVQ 主机入口 |
| `mmq.cu:266` | MMQ heuristic |
| `mmq.cu:85` | MMQ 主机入口 |
| `convert.cu` | cuBLAS fallback 的量化/非连续输入转换 |
