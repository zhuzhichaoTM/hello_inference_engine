# 04 · CUDA 执行：算子分发、cuBLAS 参数、kernel 配置

> 对应源码：`src/ggml-cuda/ggml-cuda.cu:4425`（`ggml_backend_cuda_graph_compute`）、
> `:2068`（`ggml_cuda_compute_forward`）、`:1824`（`ggml_cuda_mul_mat`）、`:1623`（`ggml_cuda_mul_mat_cublas`）、
> `binbcast.cu`、`unary.cu`、`cross-entropy-loss.cu`、`opt-step-adamw.cu`、`argmax.cu`、`count-equal.cu`、`out-prod.cu`、`scale.cu`
>
> 本篇回答：**一个节点从 `nodes[]` 变成 kernel 要经过哪些判断？grid/block 为什么是这些数？**

---

## 0 执行入口的三层结构

```text
ggml_backend_sched_graph_compute(sched, graph)            ← scheduler 层
  └── compute_splits()
        └── ggml_backend_graph_compute_async(split_backend, &split->graph)   ← backend 层
              └── ggml_backend_cuda_graph_compute(backend, cgraph)           ← CUDA 后端
                    └── ggml_cuda_graph_evaluate_and_capture()
                          └── for (i = 0; i < cgraph->n_nodes; i++)
                                ├── ggml_cuda_is_view_or_noop(node) ? continue
                                ├── (flags & COMPUTE) == 0 ? continue
                                ├── ggml_cuda_try_fuse() ? i += skip
                                └── ggml_cuda_compute_forward(ctx, node)     ← 算子层
                                      └── switch (dst->op) → ggml_cuda_xxx()
                                            └── kernel<<<grid, block, shmem, stream>>>
```

---

## 1 两层跳过：为什么 28 个节点只有 23 次 kernel

### 1.1 断点

```gdb
break ggml_cuda_compute_forward
commands
silent
set $fc = $fc + 1
printf "CUDA_FWD#%d op=%-22s ne=[%ld,%ld] src0=%s src1=%s\n", $fc,
  ggml_op_desc(dst), dst->ne[0], dst->ne[1],
  dst->src[0] ? dst->src[0]->name : "-", dst->src[1] ? dst->src[1]->name : "-"
continue
end
```

### 1.2 `gb_grad` 的 23 次分发

```text
CUDA_FWD#1  op=MUL_MAT                ne=[500,500]  src0=fc1.weight       src1=images
CUDA_FWD#2  op=ADD                    ne=[500,500]  src0=node_2           src1=fc1.bias
CUDA_FWD#3  op=RELU                   ne=[500,500]  src0=node_4
CUDA_FWD#4  op=MUL_MAT                ne=[10,500]   src0=fc2.weight       src1=node_5
CUDA_FWD#5  op=ADD                    ne=[10,500]   src0=node_6           src1=fc2.bias
CUDA_FWD#6  op=CROSS_ENTROPY_LOSS     ne=[1,1]      src0=logits           src1=labels
CUDA_FWD#7  op=SCALE                  ne=[1,1]      src0=loss_cross_entropy
CUDA_FWD#8  op=ARGMAX                 ne=[500,1]    src0=logits
CUDA_FWD#9  op=ARGMAX                 ne=[500,1]    src0=labels
CUDA_FWD#10 op=COUNT_EQUAL            ne=[1,1]      src0=pred             src1=node_12
CUDA_FWD#11 op=SCALE                  ne=[1,1]      src0=leaf_2
CUDA_FWD#12 op=CROSS_ENTROPY_LOSS_BACK ne=[10,500]  src0=grad for loss    src1=logits
CUDA_FWD#13 op=REPEAT_BACK            ne=[10,1]     src0=grad for node_6
CUDA_FWD#14 op=ADD                    ne=[10,1]     src0=leaf_3           src1=node_16
CUDA_FWD#15 op=OUT_PROD               ne=[500,10]   src0=node_5           src1=grad for node_6
CUDA_FWD#16 op=ADD                    ne=[500,10]   src0=leaf_4           src1=node_18
CUDA_FWD#17 op=OUT_PROD               ne=[500,500]  src0=fc2.weight       src1=grad for node_6 (transposed)
CUDA_FWD#18 op=STEP                   ne=[500,500]  src0=node_4
CUDA_FWD#19 op=MUL                    ne=[500,500]  src0=node_22          src1=grad for node_5
CUDA_FWD#20 op=REPEAT_BACK            ne=[500,1]    src0=grad for node_2
CUDA_FWD#21 op=ADD                    ne=[500,1]    src0=leaf_5           src1=node_24
CUDA_FWD#22 op=OUT_PROD               ne=[784,500]  src0=images           src1=grad for node_2
CUDA_FWD#23 op=ADD                    ne=[784,500]  src0=leaf_6           src1=node_26
```

### 1.3 被跳过的 5 个

| 节点 | 原因 | 判断代码 |
|---|---|---|
| `fc2.weight` / `fc1.weight` / `fc1.bias` / `fc2.bias` | `op=NONE` | `ggml_cuda_is_view_or_noop` |
| `grad for node_6 (transposed)` | `op=TRANSPOSE` | 同上 |

```cpp
static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}
```

**为什么 `TRANSPOSE` 不需要 kernel？** 因为它是**视图**：只改 `nb[]`，`data` 不变。

```text
grad for node_6              data=0x7f049d7ed400  nb=[4,40,20000,20000]   ne=[10,500]
grad for node_6 (transposed) data=0x7f049d7ed400  nb=[40,4,20000,20000]   ne=[500,10]
```

下游的 `OUT_PROD(fc2.weight, transposed)` 直接用 `nb` 做地址计算，不需要物化转置结果。

**`flags` 那一层跳过**：4 个权重节点的 `flags=4`（只有 PARAM），没有 `COMPUTE(16)`，但它们在 `is_view_or_noop` 那层就被拦下了，走不到 flags 判断。

### 1.4 `gb_opt` 的 27 次

在 23 次基础上追加 4 个 AdamW：

```text
CUDA_FWD#47 op=OPT_STEP_ADAMW ne=[10,1]    src0=fc2.bias   src1=grad for fc2.bias
CUDA_FWD#48 op=OPT_STEP_ADAMW ne=[500,1]   src0=fc1.bias   src1=grad for fc1.bias
CUDA_FWD#49 op=OPT_STEP_ADAMW ne=[784,500] src0=fc1.weight src1=grad for fc1.weight
CUDA_FWD#50 op=OPT_STEP_ADAMW ne=[500,10]  src0=fc2.weight src1=grad for fc2.weight
```

---

### 1.5 源码逐行：节点遍历与两层跳过（`ggml-cuda.cu:4288-4360`）

```cpp
4288              for (int i = 0; i < cgraph->n_nodes; i++) {
4289                  ggml_tensor * node = cgraph->nodes[i];
                     // （并发 event 的 fork/join 处理，本例不触发，省略 4290-4318）
4320                  prev_i = i;
4321
4322                  if (ggml_cuda_is_view_or_noop(node)) {
4323                      continue;                                  // ← 第一层跳过
4324                  }
4325
4326                  if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
4327                      continue;                                  // ← 第二层跳过
4328                  }
4329
4330                  int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
4331
4332                  if (nodes_to_skip != 0) {
4334                      const int last_fused = i + nodes_to_skip;
                      ...
4339                      i += nodes_to_skip;                        // ← 融合跳过
4340                      continue;
4341                  }
4342
4343                  bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
4344                  if (!ok) {
4345                      GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
4346                  }
4347                  GGML_ASSERT(ok);
4348              }
```

#### 1.5.1 第一层：`ggml_cuda_is_view_or_noop`（`ggml-cuda.cu:2556-2559`）

```cpp
static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}
```

**五种算子被跳过**：

| 算子 | 为什么不需要 kernel |
|---|---|
| `NONE` | 叶子/权重，数据已经就位 |
| `RESHAPE` | 只改 `ne`，`data` 和 `nb` 可能变，但不搬数据 |
| `TRANSPOSE` | 只交换 `nb[0]`/`nb[1]` |
| `VIEW` | `data = view_src->data + view_offs` |
| `PERMUTE` | 只重排 `nb` |

**`ggml_is_empty(t)`**：`ne[0..3]` 中有 0，空张量不需要计算。

**本例被跳过 5 个**：

```text
fc2.weight (NONE)、fc1.weight (NONE)、fc1.bias (NONE)、fc2.bias (NONE)
grad for node_6 (transposed) (TRANSPOSE)
```

#### 1.5.2 第二层：`GGML_TENSOR_FLAG_COMPUTE`

```cpp
if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
    continue;
}
```

**本例没有节点在这一层被跳过**——因为所有 `op != NONE` 的节点都在建图时被打了 `COMPUTE`（02 §2.6.1）。如果将来有人手动构造图并清掉某个节点的 `COMPUTE`，这一层会生效。

**为什么保留这一层**：`ggml_graph_view` 创建的子图可能包含“只为依赖关系存在、不需要计算”的节点（比如 scheduler 插入的 `input_dep`，`op=NONE` 但不在 `is_view_or_noop` 里？——不，它在第一层就被拦下了）。这一层是给 `GGML_OP_CUSTOM` 或未来的“条件执行”预留的。

#### 1.5.3 第三层：融合跳过

```cpp
int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
if (nodes_to_skip != 0) {
    i += nodes_to_skip;
    continue;
}
```

**`nodes_to_skip` 是“被融合掉的后续节点数”**。比如 `MUL_MAT + ADD + GLU` 三个节点融合成一个 kernel，第一个节点返回 `nodes_to_skip=2`，循环直接跳到 `i+2` 之后。

**本例 `nodes_to_skip` 恒为 0**（02 §2.2 的融合条件不满足），所以每次都执行 `ggml_cuda_compute_forward`。

#### 1.5.4 断点对照：23 次分发

```text
28 个节点
- 4 个 NONE（权重）
- 1 个 TRANSPOSE
= 23 次 ggml_cuda_compute_forward  ✅
```

`gb_opt` 是 32 个节点，多 4 个 AdamW，所以 27 次。

---

## 2 融合尝试：`ggml_cuda_try_fuse`

### 2.1 断点

```gdb
break ggml-cuda.cu:4330     # int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
commands
silent
printf "TRY_FUSE i=%d op=%s\n", i, ggml_op_name(cgraph->nodes[i]->op)
continue
end
```

```text
TRY_FUSE i=2 op=MUL_MAT
TRY_FUSE i=4 op=ADD
TRY_FUSE i=6 op=MUL_MAT
...
```

### 2.2 本例没有融合

`ggml_cuda_try_fuse` 只匹配四种模式：

| 模式 | 触发条件 | mnist 是否有 |
|---|---|---|
| MoE weighted reduction | `GGML_OP_MUL` + 特定拓扑 | 无 |
| gated_delta_net → cpy | `GGML_OP_GATED_DELTA_NET` | 无 |
| topk-moe | `UNARY` / `SOFT_MAX` / `ARGSORT` + MoE 拓扑 | 无 |
| MUL_MAT + bias/GLU | 见 `ggml_cuda_try_fuse_mul_mat` | 需要 `dst->ne[1] == 1` |

mnist 的 `MUL_MAT` 是 `ne[1]=500`，不满足融合条件：

```cpp
// we only support fusion for ncols_dst = 1
if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
    return false;
}
```

**所以 mnist 的 ADD/RELU 是独立 kernel**，每个都要一次 launch。真实 LLM 的 decode 阶段（`ne[1]=1`）才能享受 `MUL_MAT + bias + GLU` 融合。

---

## 3 算子分发：`switch (dst->op)`

### 3.1 代码

```cpp
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:                 ggml_cuda_argmax(ctx, dst); break;
        case GGML_OP_COUNT_EQUAL:            ggml_cuda_count_equal(ctx, dst); break;
        case GGML_OP_ADD:                    ggml_cuda_op_add(ctx, dst); break;
        case GGML_OP_MUL:                    ggml_cuda_op_mul(ctx, dst); break;
        case GGML_OP_UNARY:                  ggml_cuda_op_unary(ctx, dst); break;
        case GGML_OP_MUL_MAT:                ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst); break;
        case GGML_OP_OUT_PROD:               ggml_cuda_out_prod(ctx, dst); break;
        case GGML_OP_SCALE:                  ggml_cuda_op_scale(ctx, dst); break;
        case GGML_OP_CROSS_ENTROPY_LOSS:     ggml_cuda_cross_entropy_loss(ctx, dst); break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK: ggml_cuda_cross_entropy_loss_back(ctx, dst); break;
        case GGML_OP_OPT_STEP_ADAMW:         ggml_cuda_opt_step_adamw(ctx, dst); break;
        ...
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;                     // 视图，直接成功
        default:
            return false;                    // 不支持
    }
    return true;
}
```

**`GGML_OP_UNARY` 是二级分发**：

```cpp
case GGML_OP_UNARY:
    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_RELU:    ggml_cuda_op_relu(ctx, dst); break;
        case GGML_UNARY_OP_STEP:    ggml_cuda_op_step(ctx, dst); break;
        ...
    }
```

`unary_op` 存在 `op_params[0]` 里，是一个 `int32_t`：

```cpp
// ggml.c
result->op = GGML_OP_UNARY;
ggml_set_unary_op(result, GGML_UNARY_OP_RELU);
```

---

## 4 `MUL_MAT`：五条路径的决策树

### 4.1 断点：命中的是哪条

```gdb
break ggml_cuda_mul_mat
break ggml_cuda_mul_mat_cublas
commands
silent
printf "CUBLAS path: src0=%s src1=%s dst=%s\n", src0->name, src1->name, dst->name
continue
end
```

```text
MUL_MAT src0=fc1.weight type=0 ne=[784,500] src1=images ne=[784,500] dst=node_2 ne=[500,500]
  -> CUBLAS path
MUL_MAT src0=fc2.weight type=0 ne=[500,10] src1=node_5 ne=[500,500] dst=node_6 ne=[10,500]
  -> CUBLAS path
```

### 4.2 决策树

```cpp
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                              const ggml_tensor * src1, ggml_tensor * dst) {
    // ① 特殊 hint（Hadamard）
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) return;

    // ② 类型不匹配 / padding 问题 → cuBLAS
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst); return;
    }

    // ③ 小 batch 向量乘 → MMVF
    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst); return;
    }

    // ④ 转置向量 → MMVF
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ...) { ... }

    // ⑤ 非量化大 batch → MMF（tensor core）
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst); return;
    }

    // ⑥ 量化小 batch → MMVQ
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) { ... }

    // ⑦ 量化大 batch → MMQ
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, 0)) { ... }

    // ⑧ 兜底
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}
```

### 4.3 为什么 mnist 走 cuBLAS

关键是 ③ 的条件：

```cpp
bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne,
                               const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) return false;
    if (src0_nb[0] != ggml_type_size(type)) return false;
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) return false;      // 2*ts 对齐（float2）
    }
    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;                    // ← 关键
                }
                ...
            }
    }
}
```

本例：

```text
fc1.weight: ne[0]=784（偶数）, nb[0]=4, nb[1]=3136（%8==0）, nb[2]=1,568,000（%8==0）
fc2.weight: ne[0]=500（偶数）, nb[0]=4, nb[1]=2000（%8==0）
cc = 890（Ampere+，ampere_mma_available 为 true）
ne11 = 500
```

`500 > 3`，所以 MMVF 不适用。⑤ 的 `should_use_mmf` 对 F32 也通常返回 false（MMF 主要面向 F16/BF16），所以最终落到 ⑧ cuBLAS。

**对比真实 LLM decode**：`ne11=1`（一次一个 token），`should_use_mmvf` 返回 true，走 MMVF，kernel 是 `mul_mat_vec_f<float,float,1,32,...>`，`grid=(nrows,1,1)`、`block=(32,1,1)`。这就是 decode 阶段不用 cuBLAS 的原因：单向量乘用 cuBLAS 反而慢。

### 4.4 cuBLAS 的参数推导（重点）

断点：

```gdb
break ggml_cuda_mul_mat_cublas_impl<(ggml_type)0>
commands
silent
printf "src0=%s ne=[%ld,%ld] nb=[%ld,%ld]  src1=%s ne=[%ld,%ld] nb=[%ld,%ld]  dst=%s ne=[%ld,%ld] nb=[%ld,%ld]\n",
  src0->name, src0->ne[0], src0->ne[1], src0->nb[0], src0->nb[1],
  src1->name, src1->ne[0], src1->ne[1], src1->nb[0], src1->nb[1],
  dst->name,  dst->ne[0],  dst->ne[1],  dst->nb[0],  dst->nb[1]
printf "  => M=ne01=%ld N=ne11=%ld K=ne10=%ld lda=%ld ldb=%ld ldc=%ld\n",
  src0->ne[1], src1->ne[1], src0->ne[0], src0->nb[1]/4, src1->nb[1]/4, dst->ne[0]
continue
end
```

```text
CUBLAS_IMPL_F32 src0=fc1.weight ne=[784,500] nb=[4,3136] src1=images ne=[784,500] nb=[4,3136] dst=node_2 ne=[500,500] nb=[4,2000]
  => M=ne01=500 N=ne11=500 K=ne10=784 lda=784 ldb=784 ldc=500

CUBLAS_IMPL_F32 src0=fc2.weight ne=[500,10] nb=[4,2000] src1=node_5 ne=[500,500] nb=[4,2000] dst=node_6 ne=[10,500] nb=[4,40]
  => M=ne01=10 N=ne11=500 K=ne10=500 lda=500 ldb=500 ldc=10
```

调用代码：

```cpp
if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
    CUBLAS_CHECK(
        cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                (const float *) alpha, (const float *) src0_ptr, s01,      // lda
                                       (const float *) src1_ptr, s11,      // ldb
                (const float *) beta,  (float       *)  dst_ptr,  ne0));   // ldc
}
```

**为什么是 `CUBLAS_OP_T, CUBLAS_OP_N`？**

cuBLAS 是**列主序**，`cublasSgemm(transa, transb, M, N, K, ...)` 计算：

```text
C[M×N] = alpha × op(A)[M×K] × op(B)[K×N] + beta × C[M×N]
```

ggml 是**行主序**（`ne[0]` 最内层）。两者的对应关系：

```text
ggml 的 src0[ne0=K, ne1=M] 按行主序 = cuBLAS 看到的 [M×K] 列主序矩阵
ggml 的 src1[ne0=K, ne1=N] 按行主序 = cuBLAS 看到的 [N×K] 列主序矩阵
要算 dst[ne0=M, ne1=N] = src0ᵀ · src1
cuBLAS 需要 C[M×N] = A[M×K] × B[K×N]
其中 A = src0 本身（列主序下就是 [M×K]），B = src1ᵀ（列主序下 [N×K] 转置成 [K×N]）
```

所以：

```text
op(A) = op(src0) = CUBLAS_OP_T   ← 因为 src0 在列主序下是 [K,M]，要转成 [M,K]
op(B) = op(src1) = CUBLAS_OP_N   ← 因为 src1 在列主序下已经是 [K,N]
```

**leading dimension 的推导**：

```cpp
const size_t src0_ts = ggml_type_size(src0->type);   // 4
int64_t s01 = nb01 / src0_ts;                        // nb[1]/4
int64_t s11 = nb11 / src1_ts;                        // nb[1]/4
```

对 `fc1.weight`：`nb[1] = 4 × 784 = 3136`，`s01 = 3136/4 = 784`，正好是 `ne[0]`（K）。
对 `node_2`（输出）：`ldc = ne0 = 500`。

**为什么 lda = K 而不是 M？** 因为列主序下，A 的每一列有 `lda` 个元素。A 是 `[M,K]` 的转置视图，原始存储是 `[K,M]` 行主序 = `[M,K]` 列主序，所以每列（对应一个 M 行）的跨度是 K。

### 4.4.1 源码逐行：`ggml_cuda_mul_mat_cublas_impl<F32>`（`ggml-cuda.cu:1409-1620`）

这是 `MUL_MAT` 的 cuBLAS 路径。`compute_type=F32` 时走 `cublasSgemm`，其余走 `cublasGemmEx`。

#### 4.4.1.1 类型与指针准备（1409-1470）

```cpp
1409  template<ggml_type compute_type>
1410  static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
1411      using traits = batched_mul_mat_traits<compute_type>;
1412      using cuda_t = typename traits::cuda_type;
1413
1414      GGML_ASSERT(ggml_is_contiguous(dst));
1416      GGML_TENSOR_BINARY_OP_LOCALS
1417
1418      const int64_t ne_dst = ggml_nelements(dst);
1419      cudaStream_t main_stream = ctx.stream();
1420      cublasHandle_t cublas_h = ctx.cublas_handle();
1421
1422      const size_t src0_ts = ggml_type_size(src0->type);
1423      GGML_ASSERT(nb00 == src0_ts);
1424      int64_t s01 = nb01 / src0_ts;
1425      int64_t s02 = nb02 / src0_ts;
1426      int64_t s03 = nb03 / src0_ts;
1427
1428      const size_t src1_ts = ggml_type_size(src1->type);
1429      GGML_ASSERT(nb10 == src1_ts);
1430      int64_t s11 = nb11 / src1_ts;
1431      int64_t s12 = nb12 / src1_ts;
1432      int64_t s13 = nb13 / src1_ts;
1433
1434      float * dst_ddf = (float *) dst->data;
1435
1436      const cuda_t * src0_ptr = nullptr;
1437      const cuda_t * src1_ptr = nullptr;
1438
1439      ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
1440      ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());
1441
1442      bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
1443      bool is_src1_cont_2 = ggml_is_contiguous_2(src1);
1444
1445      if (src0->type == compute_type) {
1446          src0_ptr = (const cuda_t *) src0->data;
1447      } else {
1448          src0_alloc.alloc(ggml_nelements(src0));
1449          ...  // 类型转换
1450          src0_ptr = src0_alloc.get();
1451      }
1452
1453      if (src1->type == compute_type) {
1454          src1_ptr = (const cuda_t *) src1->data;
1455      } else {
1456          src1_alloc.alloc(ggml_nelements(src1));
1457          ...  // 类型转换
1458          src1_ptr = src1_alloc.get();
1459      }
```

**`GGML_TENSOR_BINARY_OP_LOCALS`** 是一个宏，展开成：

```cpp
const int64_t ne00 = src0->ne[0]; const int64_t ne01 = src0->ne[1]; ...
const size_t  nb00 = src0->nb[0]; const size_t  nb01 = src0->nb[1]; ...
const int64_t ne0  = dst->ne[0];  const int64_t ne1  = dst->ne[1];  ...
```

**`s01 = nb01 / src0_ts`**：把字节步长转成元素步长。F32 时 `src0_ts=4`。

**本例 `fc1.weight`**：

```text
nb00 = 4,  nb01 = 3136, nb02 = 1,568,000
src0_ts = 4
s01 = 784, s02 = 392,000
```

**注意 `s02 = 392,000` 而不是 `nb02/4 = 392,000`** —— 对 `[784,500]` 的 2D 张量，`s02` 实际不会被用到（`ne02=1`），但代码仍然计算。

**1445-1459 的类型转换**：如果 `src0->type != compute_type`，需要临时分配并转换。本例 `fc1.weight` 是 F32，`compute_type=F32`，所以 `src0_ptr = src0->data`，**零拷贝**。

#### 4.4.1.2 输出指针与精度（1498-1536）

```cpp
1498      ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
1499      char * dst_ptr;
1500      size_t nbd2 = dst->nb[2];
1501      size_t nbd3 = dst->nb[3];
1502
1503      cublasComputeType_t cu_compute_type = traits::compute_type;
1504      cudaDataType_t cu_data_type = traits::data_type;
1505      cudaDataType_t cu_data_type_a = traits::data_type;
1506      cudaDataType_t cu_data_type_b = traits::data_type;
1507      const void * alpha = traits::get_alpha();
1508      const void * beta  = traits::get_beta();
1509
1510      const int cc = ggml_cuda_info().devices[ctx.device].cc;
1511      bool prefer_f32_output = false;
1512      if (compute_type == GGML_TYPE_F16) {
1513          prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || ...;
1514      } else if (compute_type == GGML_TYPE_BF16) {
1515          prefer_f32_output = ...;
1516      }
1517
1518      if (prefer_f32_output) { ... }
1519      else {
1520          if constexpr (compute_type == GGML_TYPE_F32) {
1521              dst_ptr = (char *) dst_ddf;  // Direct F32 output
1522          } else {
1523              dst_ptr = (char *) dst_temp.alloc(ne_dst);
1524              ...
1525          }
1526      }
```

**`prefer_f32_output`**：只有 F16/BF16 计算时才会考虑。F32 计算直接写 `dst`，**没有临时 buffer**。

**这就是为什么本例的 `ctx.pool()` 从未被创建**（01 §7.4）：`dst_temp` 在 F32 路径下不 `alloc`，`src0_alloc`/`src1_alloc` 也不 `alloc`，所以 pool 一直是 `nullptr`。

#### 4.4.1.3 四种 GEMM 调用（1541-1620）

```cpp
1541      // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
1542      // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
1543      //     probably because the internal kernel selection logic is suboptimal.
1544      if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
1545          CUBLAS_CHECK(
1546              cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
1547                      ne01, ne11, ne10,
1548                      (const float *) alpha, (const float *) src0_ptr, s01,
1549                                             (const float *) src1_ptr, s11,
1550                      (const float *) beta,  (float       *)  dst_ptr, ne0));
1551      } else if (ne12 == 1 && ne13 == 1) {
1552          CUBLAS_CHECK(
1553              cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
1554                      ne01, ne11, ne10,
1555                      alpha, src0_ptr, cu_data_type_a, s01,
1556                             src1_ptr, cu_data_type_b, s11,
1557                      beta,   dst_ptr, cu_data_type,   ne0,
1558                      cu_compute_type,
1559                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
1560      } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
1561          // use cublasGemmStridedBatchedEx
1562          ...
1563      } else {
1564          // use cublasGemmBatchedEx
1565          ...
1566      }
```

**四个分支**：

| 分支 | 条件 | 本例 |
|---|---|---|
| ① `cublasSgemm` | F32 + 无 batch 维 | **命中** |
| ② `cublasGemmEx` | 非 F32 + 无 batch 维 | 不命中 |
| ③ `cublasGemmStridedBatchedEx` | 有 batch + 连续 | 不命中 |
| ④ `cublasGemmBatchedEx` | 有 batch + 不连续 | 不命中 |

**为什么 ① 优先于 ②**：源码注释说得很清楚——老 GPU 上 `GemmEx` 的 kernel 选择逻辑不优，单矩阵用 `cublasSgemm` 更快。这是“为兼容性牺牲统一性”的典型取舍。

**参数顺序**：

```text
cublasSgemm(handle, transa, transb,
            m, n, k,
            alpha, A, lda,
                   B, ldb,
            beta,  C, ldc)
```

**本例 `fc1.weight`**：

```text
m = ne01 = 500
n = ne11 = 500
k = ne10 = 784
lda = s01 = 784
ldb = s11 = 784
ldc = ne0 = 500
```

**`ldc = ne0` 而不是 `ne1`**：因为 cuBLAS 列主序下，C 的每列有 `ne0` 个元素。`dst` 的 `ne=[500,500]`，所以 `ldc=500`。

**`fc2.weight` 的 `ldc = ne0 = 10`**：`dst=node_6` 的 `ne=[10,500]`，所以 `ldc=10`。这是断点里 `ldc=10` 的来源。

---

### 4.5 `OUT_PROD` 的 cuBLAS

断点：

```text
OUT_PROD M=500 N=10 K=500 lda=500 ldb=10 ldc=500 src1_T=0
OUT_PROD M=500 N=500 K=10 lda=500 ldb=10 ldc=500 src1_T=1
OUT_PROD M=784 N=500 K=500 lda=784 ldb=500 ldc=784 src1_T=0
```

源码：

```cpp
void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);

    if (dps2 == 1 && ne2 > 1) { ... }            // 分组查询注意力
    else if (ne2 > 1 || ne3 > 1) { ... }         // 批量 GEMM
    else {
        cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                ne0, ne1, ne01, &alpha, src0_d, lda, src1_d, ldb, &beta, dst_d, ldc);
    }
}
```

**`src1_T=1` 的那个**是 `node_21 = OUT_PROD(fc2.weight, grad for node_6 (transposed))`：

```text
src1 = grad for node_6 (transposed)，nb=[40,4,20000,20000]
ggml_is_transposed(src1) = (nb[0] > nb[1]) = (40 > 4) = true
→ src1_cublas_op = CUBLAS_OP_N
→ ldb = nb10 / 4 = 40/4 = 10
```

**为什么转置时用 `CUBLAS_OP_N`？** 因为转置视图已经把 `nb` 交换了，cuBLAS 直接按原始布局读就是对的，不需要再转一次。这是“视图零拷贝”在 GEMM 层的体现。

---

## 5 八类 kernel 的 grid/block

### 5.1 ADD（`k_bin_bcast`）

```text
ADD#1 grid=(2,500,1)  block=(128,1,1) ne=(500,500,1,1) src0=node_2 src1=fc1.bias
ADD#2 grid=(1,20,1)   block=(5,25,1)  ne=(10,500,1,1)  src0=node_6 src1=fc2.bias
```

源码：

```cpp
const int block_size = 128;
int64_t hne0 = std::max(ne0 / 2LL, 1LL);          // 每个线程处理 2 个元素

dim3 block_dims;
block_dims.x = std::min<unsigned int>(hne0, block_size);
block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
block_dims.z = std::min(std::min<unsigned int>(ne2 * ne3, block_size / block_dims.x / block_dims.y), 64U);

dim3 block_nums((hne0 + block_dims.x - 1) / block_dims.x,
                (ne1  + block_dims.y - 1) / block_dims.y,
                (ne2 * ne3 + block_dims.z - 1) / block_dims.z);
```

**验算 `ADD#1`（`ne=[500,500]`）**：

```text
hne0 = 500/2 = 250
block_dims.x = min(250, 128) = 128
block_dims.y = min(500, 128/128) = 1
block_dims.z = min(min(1, 128/128/1), 64) = 1
block_nums = (ceil(250/128), ceil(500/1), 1) = (2, 500, 1)   ✅
```

**验算 `ADD#2`（`ne=[10,500]`）**：

```text
hne0 = 10/2 = 5
block_dims.x = min(5, 128) = 5
block_dims.y = min(500, 128/5=25) = 25
block_dims.z = min(min(1, 128/5/25=1), 64) = 1
block_nums = (ceil(5/5), ceil(500/25), 1) = (1, 20, 1)   ✅
```

**为什么 `hne0 = ne0/2`？** 因为 kernel 内部用 `float2` 向量化读：

```cpp
const uint32_t s0 = blockDim.x * gridDim.x;
for (uint32_t i0 = i0s; i0 < ne0; i0 += s0) {
    const uint32_t i10 = fastmodulo(i0, ne10);
    float result = src0_row ? (float) src0_row[size_t(i0)*s00] : 0.0f;
    result = bin_op(result, (float)src1[i_src1 + size_t(i10)*s10]);
    dst_row[i0] = (dst_t) result;
}
```

**`fastmodulo` / `fastdiv`**：`i10 = i0 % ne10`，当 `ne10=784` 时整数除法很贵。ggml 预计算了 `uint3` 形式的 magic number：

```cpp
const uint3 ne10 = init_fastdiv_values((uint32_t) cne1[0]);
```

这是“把除法换成乘加”的标准优化，对 memory-bound 的广播 kernel 很关键。

**为什么 `ne1` 方向用 `block_dims.y` 而不是拉平？** 因为广播是沿 `ne1` 重复的：

```cpp
const uint32_t i11 = fastmodulo(i1, ne11);   // i1 % 1 = 0，所有行读同一个 bias
```

`fc1.bias` 的 `ne=[500,1]`，`ne11=1`，所以 `i11` 恒为 0，所有 500 行读同一个 bias 元素。这是 broadcast 的硬件友好实现：不需要额外判断，靠 modulo 自然退化。

### 5.1.1 源码逐行：`k_bin_bcast`（`binbcast.cu:34-99`）

```cpp
34  static __global__ void k_bin_bcast(const src0_t *         src0,
35                                     const src1_t *         src1,
36                                     dst_t *                dst,
37                                     const uint32_t         ne0,
38                                     const uint32_t         ne1,
39                                     const uint32_t         ne2,
40                                     const uint3            ne3,      // ← fastdiv 编码
41                                     const uint3            ne10,
42                                     const uint3            ne11,
43                                     const uint3            ne12,
44                                     const uint3            ne13,
46                                     const uint32_t         s1,
47                                     const uint32_t         s2,
48                                     const uint32_t         s3,
49                                     const uint32_t         s00,
50                                     const uint32_t         s01,
51                                     const uint32_t         s02,
52                                     const uint32_t         s03,
53                                     const uint32_t         s10,
54                                     const uint32_t         s11,
55                                     const uint32_t         s12,
56                                     const uint32_t         s13,
57                                     src1_ptrs... src1s) {
58      ggml_cuda_pdl_lc();
59      const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
60      const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
61      const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
62      const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);
63
64      if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3.z) {
65          return;
66      }
67
68      const uint32_t i11 = fastmodulo(i1, ne11);
69      const uint32_t i12 = fastmodulo(i2, ne12);
70      const uint32_t i13 = fastmodulo(i3, ne13);
71
72      const size_t i_src0 = size_t( i3)*s03 + size_t( i2)*s02 + size_t( i1)*s01;
73      const size_t i_src1 = size_t(i13)*s13 + size_t(i12)*s12 + size_t(i11)*s11;
74      const size_t i_dst  = size_t( i3)*s3  + size_t( i2)*s2  + size_t( i1)*s1;
75
76      const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
77      dst_t * dst_row = dst + i_dst;
78
79      const uint32_t s0 = blockDim.x * gridDim.x;
80
81      ggml_cuda_pdl_sync();
82      for (uint32_t i0 = i0s; i0 < ne0; i0 += s0) {
83          const uint32_t i10 = fastmodulo(i0, ne10);
84
85          float result = src0_row ? (float) src0_row[size_t(i0)*s00] : 0.0f;
86          if constexpr (sizeof...(src1_ptrs) > 0) {
87              result = (..., (result = bin_op(result, (float)src1s[i_src1 + size_t(i10)*s10])));
88          } else {
89              result = bin_op(result, (float)src1[i_src1 + size_t(i10)*s10]);
90          }
91
92          dst_row[i0] = (dst_t) result;
93
94          // protect i0 from overflow
95          if (ne0 - i0 <= s0) {
96             break;
97          }
98      }
99  }
```

#### 5.1.1.1 第 59-62 行：三维网格展平

```cpp
const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);
```

**`i0s` 和 `i1` 是直接映射**：`blockIdx.x * blockDim.x + threadIdx.x`。

**`i2` / `i3` 是从同一个 `blockIdx.z * blockDim.z + threadIdx.z` 拆出来的**：

```text
z_linear = blockIdx.z * blockDim.z + threadIdx.z
i2 = z_linear / ne3
i3 = z_linear % ne3
```

**为什么 `ne3` 用 `uint3`（fastdiv 编码）而不是 `uint32_t`**：因为 `i2` 需要除法。`fastdiv` 把除法换成“乘 magic number + 右移”，在 GPU 上比整数除法快 10~20 倍。

**`fastdiv` 的真实实现**（`src/ggml-cuda/common.cuh:925-954`）：

```cpp
925  static const uint3 init_fastdiv_values(uint64_t d_64) {
926      GGML_ASSERT(d_64 != 0);
927      GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());
928
929      uint32_t d = (uint32_t)d_64;
930
931      // compute L = ceil(log2(d));
932      uint32_t L = 0;
933      while (L < 32 && (uint32_t{ 1 } << L) < d) {
934          L++;
935      }
936
937      uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
938
939      // pack divisor as well to reduce error surface
940      return make_uint3(mp, L, d);
941  }
942
943  static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
944      // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
945      // fastdiv_values.z is unused and optimized away by the compiler.
946      // Compute high 32 bits of n * mp
947      const uint32_t hi = __umulhi(n, fastdiv_values.x);
948      // add n, apply bit shift
949      return (hi + n) >> fastdiv_values.y;
950  }
951
952  static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
953      // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
954      return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
955  }
```

**数学原理**：这是 Granlund-Montgomery 除法。对无符号 32 位除法 `n / d`：

```text
L  = ceil(log2(d))
mp = floor(2^32 * (2^L - d) / d) + 1
n / d ≈ (umulhi(n, mp) + n) >> L
```

**误差分析**：`mp` 的构造保证 `(n * mp) >> 32 + n >> L` 对所有 `n < 2^32` 精确等于 `n / d`。`__umulhi` 是一条指令（取 64 位乘积的高 32 位），`>> L` 是一条移位，所以整个除法是 **2 条指令**，而硬件 `div` 是 20~40 个周期。

**`fastmodulo` 复用 `fastdiv`**：`n % d = n - (n/d)*d`，一次除法 + 一次乘减。所以 `k_bin_bcast` 里的 `i2 = fastdiv(...)` 和 `i11/i12/i13 = fastmodulo(...)` 总共是 4 次“2 指令除法”。

**为什么 `uint3` 而不是传三个 `uint32_t`**：CUDA 的 kernel 参数通过 constant memory 传递，`uint3` 是 12 字节、按 4 字节对齐，打包成一个参数比分三个更省参数空间（constant bank 只有 4 KB）。

**为什么只有 `ne3` 需要 fastdiv**：因为 `i0`/`i1` 直接来自网格，`i2`/`i3` 需要拆分。`ne10/ne11/ne12/ne13` 的 `fastmodulo` 也是同样的优化。

#### 5.1.1.2 第 64-66 行：边界检查

```cpp
if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3.z) {
    return;
}
```

**四个维度都要检查**：因为 `grid` 是按 `ceil(ne / block)` 算的，最后一个 block 可能越界。

**`i0s >= ne0` 用 `i0s`（起始索引）而不是循环变量**：因为 `i0` 在 82 行的循环里会推进，但起始索引越界意味着这个线程完全不需要工作。

#### 5.1.1.3 第 68-70 行：广播的 modulo

```cpp
const uint32_t i11 = fastmodulo(i1, ne11);
const uint32_t i12 = fastmodulo(i2, ne12);
const uint32_t i13 = fastmodulo(i3, ne13);
```

**`i11 = i1 % ne11`**：如果 `src1` 的 `ne1` 小于 `dst` 的 `ne1`，就沿 `i1` 重复读同一个元素。

**本例 `ADD#1`（`node_4 = ADD(node_2, fc1.bias)`）**：

```text
dst ne = [500,500]
src0 = node_2  ne=[500,500]
src1 = fc1.bias ne=[500,1]

i11 = i1 % 1 = 0        ← 所有行读同一个 bias
i12 = i2 % 1 = 0
i13 = i3 % 1 = 0
```

**所以 500 个线程（`blockIdx.y` 方向）都读 `fc1.bias[0]` 这一个 float**。这不是 bug——`fc1.bias` 的 `ne[1]=1`，它的语义就是“每一行都加同一个 bias”。

**`ADD#2`（`logits = ADD(node_6, fc2.bias)`）同理**：

```text
dst ne = [10,500]
src1 = fc2.bias ne=[10,1]
i11 = i1 % 1 = 0
i10 = i0 % 10 = i0    （ne10=10，不广播）
```

#### 5.1.1.4 第 72-74 行：三个地址

```cpp
const size_t i_src0 = size_t(i3)*s03 + size_t(i2)*s02 + size_t(i1)*s01;
const size_t i_src1 = size_t(i13)*s13 + size_t(i12)*s12 + size_t(i11)*s11;
const size_t i_dst  = size_t(i3)*s3  + size_t(i2)*s2  + size_t(i1)*s1;
```

**`s01 = nb01 / sizeof(src0_t)`**（在 `launch_bin_bcast_pack` 里算）：

```cpp
size_t s01 = nb01 / sizeof(src0_t);
```

**为什么除以 `sizeof(src0_t)`**：因为 `nb` 是字节步长，而 kernel 里的指针是 `src0_t *`，`+1` 会前进 `sizeof(src0_t)` 字节。所以用元素步长而不是字节步长。

**本例 `ADD#1`**：

```text
src0 = node_2  nb=[4,2000,1000000,1000000] → s01=2000/4=500, s02=1000000/4=250000
src1 = fc1.bias nb=[4,2000,2000,2000]      → s11=2000/4=500, s12=500, s13=500
dst  = node_4  nb=[4,2000,1000000,1000000] → s1=500, s2=250000, s3=250000
```

**注意 `src1` 的 `s12=s13=500`**：虽然 `ne12=ne13=1`，`i12=i13=0`，所以 `i_src1` 里这两项为 0，不影响结果。

#### 5.1.1.5 第 79-98 行：`i0` 方向的循环

```cpp
const uint32_t s0 = blockDim.x * gridDim.x;

for (uint32_t i0 = i0s; i0 < ne0; i0 += s0) {
    const uint32_t i10 = fastmodulo(i0, ne10);
    float result = src0_row ? (float) src0_row[size_t(i0)*s00] : 0.0f;
    result = bin_op(result, (float)src1[i_src1 + size_t(i10)*s10]);
    dst_row[i0] = (dst_t) result;

    if (ne0 - i0 <= s0) {
       break;
    }
}
```

**`s0` 是“网格总线程数”**：`blockDim.x * gridDim.x`。对 `ADD#1`：`128 * 2 = 256`。

**为什么是循环而不是“一个线程一个元素”**：因为 `hne0 = ne0/2`，`block_dims.x` 最多 128，`grid.x = ceil(250/128) = 2`，所以 `x` 方向只有 256 个线程，而 `ne0=500`。每个线程要处理 `500/256 ≈ 2` 个元素。

**`i10 = i0 % ne10`**：`src1` 沿 `i0` 方向也可能广播。本例 `ne10=500`（`fc1.bias` 的 `ne[0]=500`），所以 `i10=i0`，不广播。

**第 95-97 行的 `break`**：

```cpp
if (ne0 - i0 <= s0) {
   break;
}
```

**为什么需要这个**：如果 `ne0` 不是 `s0` 的整数倍，最后一次 `i0 += s0` 可能溢出 `uint32_t`（虽然 `ne0 < 2^32`，但 `i0 + s0` 可能超过）。`break` 在“下一次迭代一定越界”时提前退出，避免无符号回绕导致的死循环。

**`ADD#1` 的实际执行**：

```text
i0s ∈ [0, 256)
i0 = i0s, i0s+256, ...
ne0 = 500
第一个线程：i0 = 0, 256
第二个线程：i0 = 1, 257
...
第 244 个线程：i0 = 243, 499
第 245 个线程：i0 = 244（下一次 500 >= 500，退出）
```

#### 5.1.1.6 `launch_bin_bcast_pack` 的 collapse 优化（`binbcast.cu:201-213`）

```cpp
if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && !ggml_is_permuted(src0) && !ggml_is_permuted(src1)) {
    for (int i = 0; i < 4; i++) {
        if (nr[i] != 1) break;
        if (i > 0) {
            collapse_nb(cnb, cne);      // 合并维度
            collapse_nb(cnb0, cne0);
            collapse_nb(cnb1, cne1);
            collapse(cne);              // ne[0] *= ne[1]; ne[1] = ne[2]; ...
            collapse(cne0);
            collapse(cne1);
        }
    }
}
```

**`nr[i] = ne1[i] / ne[i]`** 是“广播倍数”。如果某个维度 `nr=1`（不广播），且张量连续，就可以把它和下一个维度合并，减少循环层数。

**本例 `ADD#1`**：

```text
dst ne = [500,500,1,1]
src1 ne = [500,1,1,1]
nr[0] = 500/500 = 1  → 可以 collapse
nr[1] = 1/500 → 0？ 实际 nr1 = ne11/ne1 = 1/500 不整除
```

**等等**，`launch_bin_bcast_pack` 里：

```cpp
int nr0 = ne10 / ne0;
int nr1 = ne11 / ne1;
```

`ne10=500, ne0=500` → `nr0=1`；`ne11=1, ne1=500` → `nr1=0`（整数除法）。

**`nr1=0 != 1`，所以 collapse 在 `i=1` 时停止**，只合并了 `i=0` 那一层。合并后：

```text
cne = [500*500, 1, 1, 1] = [250000, 1, 1, 1]
```

**但断点里 `ne=(500,500,1,1)` 而不是 `(250000,1,1,1)`**——说明本例**没有**走 collapse 分支。为什么？

看 `if (ggml_is_contiguous(src0) && ...)`：`node_2` 是 `[500,500]`，`nb=[4,2000,1000000,1000000]`，`ggml_is_contiguous` 为 true；`fc1.bias` 是 `[500,1]`，`nb=[4,2000,2000,2000]`，`ggml_is_contiguous` 为 true。

**所以应该走 collapse**。但断点打印的 `ne0/ne1/ne2/ne3` 是 `launch_bin_bcast_pack` **局部变量** `ne0..ne3`，它们来自：

```cpp
int64_t cne[]  = { ne0, ne1, ne2, ne3 };    // 从 GGML_TENSOR_BINARY_OP_LOCALS 来
...
int64_t ne0 = cne[0];
int64_t ne1 = cne[1];
```

**如果 collapse 执行了，`cne[0]` 会变成 250,000**。但断点是 `(500,500,1,1)`，说明 **collapse 没有执行**。

**为什么**：因为 `nr1 = ne11 / ne1 = 1/500 = 0`，而循环条件是 `if (nr[i] != 1) break;`——`nr1=0`，在 `i=1` 时 `nr[1]=0 != 1`，break。

**但 `i=0` 时 `nr[0]=1`，collapse 应该执行一次**（因为 `if (i > 0)` 才 collapse，`i=0` 时只判断不合并）。

**关键在 `if (i > 0)`**：`i=0` 时检查 `nr[0]`，如果为 1 就继续；`i=1` 时才真正 `collapse`。但 `i=1` 时先检查 `nr[1]=0`，break——所以一次 collapse 都没发生。

**这是一个“保守的优化”**：只有当所有 4 个维度的 `nr` 都是 1 时才会逐步合并。本例 `nr1=0`，所以完全放弃 collapse。

---

### 5.2 REPEAT_BACK

```text
REPEAT_BACK grid=(1,1,1)  block=(32,1,1) ne=(10,1)   ne0src=(10,500)
REPEAT_BACK grid=(16,1,1) block=(32,1,1) ne=(500,1)  ne0src=(500,500)
```

```cpp
const dim3 block_dims(WARP_SIZE, 1, 1);
const dim3 block_nums((ne0 + WARP_SIZE - 1) / WARP_SIZE, ne1, ne2*ne3);
```

**验算**：

```text
ne0=10  → grid.x = ceil(10/32) = 1
ne0=500 → grid.x = ceil(500/32) = 16   ✅
```

**为什么用 32 个线程？** 因为 `ne0` 方向是“被重复的那个维度”，求和沿 `ne1`：

```cpp
for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
    for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
        for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
            for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                sum += src[i3*s03 + i2*s02 + i1*s01 + i0*s00];
            }
        }
    }
}
```

每个线程负责一个 `(i0, i1)` 位置，沿 `i1` 累加。对 `[10,500] → [10,1]`，每个线程累加 500 次，`grid=(1,1,1)` 只有 10 个活跃线程——**极度低效，但数据量小，可以接受**。真实 LLM 的 `REPEAT_BACK` 很少出现在热路径。

### 5.3 RELU / STEP（`unary_op_kernel`）

```text
UNARY grid=977 block=256 k=250000
```

```cpp
const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;   // 256
```

**验算**：`ceil(250000/256) = 977`（500×500 = 250,000 个元素）。

**为什么是 256 而不是 128 或 512？** 这是经验值：256 线程/block 在多数 GPU 上能达到较好的 occupancy，同时 `grid=977` 足以填满 128 个 SM（977/128 ≈ 7.6 个 block/SM）。

### 5.3.1 源码逐行：`unary_op_kernel`（`unary.cu:117-135`）

```cpp
117  template <float (*op)(float), typename T>
118  static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
119      ggml_cuda_pdl_lc();
120      const int i = blockDim.x*blockIdx.x + threadIdx.x;
121
122      if (i >= k) {
123          return;
124      }
125
126      ggml_cuda_pdl_sync();
127      dst[i] = (T)op((float)x[i]);
128  }
129
130  template <float (*op)(float), typename T>
131  static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
132      const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
133      const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
134      ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
135  }
```

**逐行**：

| 行 | 行为 | 说明 |
|---|---|---|
| 118 | `template <float (*op)(float), typename T>` | `op` 是**函数指针模板参数**，编译期就确定是 `op_relu` 还是 `op_step`，零运行时开销 |
| 119 | `ggml_cuda_pdl_lc()` | Programmatic Dependent Launch：让下一个 kernel 可以提前启动（Hopper+ 才有效，Ampere 上是空操作） |
| 120 | `blockDim.x*blockIdx.x + threadIdx.x` | 一维展平，每个线程一个元素 |
| 122 | `i >= k` 检查 | `k=250000`，`grid=977`，`977*256=250112`，最后 112 个线程越界 |
| 126 | `ggml_cuda_pdl_sync()` | 等待前一个 kernel 的数据就绪 |
| 127 | `dst[i] = (T)op((float)x[i])` | `op_relu` 是 `fmaxf(x, 0.0f)`，`op_step` 是 `x > 0 ? 1 : 0` |
| 132 | `(k + 255) / 256` | 向上取整，`250000/256 = 976.56` → 977 |

**`op_relu` 与 `op_step` 的定义**（`unary.cu:16, 44`）：

```cpp
16  static __device__ __forceinline__ float op_step(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }

44  static __device__ __forceinline__ float op_relu(float x) {
        return fmaxf(x, 0.0f);
    }
```

**注意 `op_step` 的语义**：`x > 0` 返回 1，`x <= 0` 返回 0。这正好是 ReLU 的导数（`relu'(x) = 1 if x>0 else 0`）。

**`op_relu` 用 `fmaxf` 而不是 `x > 0 ? x : 0`**：`fmaxf` 对 NaN 有定义行为（返回非 NaN 的那个），而三元运算符遇到 NaN 会走 `else` 分支。对训练来说 NaN 是错误信号，应该传播而不是吞掉，所以 ggml 选择 `fmaxf`。

**为什么 `k` 是 `int` 而不是 `int64_t`**：`ggml_nelements(src0)` 返回 `int64_t`，但传进 kernel 时截断成 `int`。对 `250000` 没问题，但对超大张量（>2G 元素）会溢出。这是 ggml 的一个已知限制——单 kernel 处理的元素数受 `int` 约束。

---

### 5.4 CROSS_ENTROPY_LOSS

```text
CE_LOSS grid=500 block=32 nclasses=10 nrows=500 shared=40
```

```cpp
const dim3 blocks_dim(WARP_SIZE, 1, 1);      // 32
const dim3 blocks_num(nrows, 1, 1);          // 500 行 → 500 个 block
const size_t nbytes_shared = ne00*sizeof(float);   // 10 × 4 = 40 B
```

**为什么一行一个 block？** 因为交叉熵是**按行独立**的（每个样本一个 loss），一行内的 softmax 需要归约：

```cpp
// ① 找 max
float max_logit = -INFINITY;
for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    max_logit = fmaxf(max_logit, logits[i]);
}
max_logit = warp_reduce_max(max_logit);

// ② sum(exp(x - max))
float sum = 0;
for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    sum += expf(logits[i] - max_logit);
}
sum = warp_reduce_sum(sum);
sum = logf(sum);

// ③ loss = -sum((x - max - log(sum)) * label) / k
```

**为什么 `shared=40`？** 当 `nbytes_shared <= smpbo`（101,376 B）时，用 `cross_entropy_loss_f32<true>` 把 logits 缓存到 shared memory，避免两次读 global：

```cpp
if (nbytes_shared <= smpbo) {
    CUDA_SET_SHARED_MEMORY_LIMIT((cross_entropy_loss_f32<true>), smpbo);
    cross_entropy_loss_f32<true><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(...);
} else {
    cross_entropy_loss_f32<false><<<blocks_num, blocks_dim, 0, stream>>>(...);
}
```

**归约后的 `sum_f32`**：

```cpp
// 500 个 block 各算一个 partial loss，写到 dst_tmp[500]
sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
// → cub::DeviceReduce::Sum，把 500 个数求和成一个标量
```

这就是 loss 是 `[1,1]` 标量的原因：**两阶段归约**（block 内 warp reduce + 跨 block 的 cub reduce）。

### 5.4.1 源码逐行：`cross_entropy_loss_f32`（`cross-entropy-loss.cu:8-50`）

```cpp
8   template <bool use_shared>
9   static __global__ void cross_entropy_loss_f32(
10          const float * __restrict__ logits, const float * __restrict__ labels, float * __restrict__ dst, const int nclasses, const int k) {
11      extern __shared__ float tmp[];
12
13      logits += int64_t(blockIdx.x)*nclasses;
14      labels += int64_t(blockIdx.x)*nclasses;
15
16      // Find maximum for softmax:
17      float max_logit = -INFINITY;
18      for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
19          const float val = logits[i];
20          max_logit = fmaxf(max_logit, val);
21
22          if (use_shared) {
23              tmp[i] = val;
24          }
25      }
26      max_logit = warp_reduce_max(max_logit);
27
28      // Calculate log(softmax(logits)) which is just logits - max:
29      float sum = 0.0f;
30      for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
31          const float logit_i = use_shared ? tmp[i] : logits[i];
32          sum += expf(logit_i - max_logit);
33      }
34      sum = warp_reduce_sum(sum);
35      sum = logf(sum);
36
37      // log(exp(logits - max) / sum) = (logits - max) - log(sum)
38      float loss = 0.0f;
39      for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
40          const float logit_i = use_shared ? tmp[i] : logits[i];
41          loss += (logit_i - max_logit - sum) * labels[i];
42      }
43      loss = -warp_reduce_sum(loss) / (float)k;
44
45      if (threadIdx.x != 0) {
46          return;
47      }
48
49      dst[blockIdx.x] = loss;
50  }
```

#### 5.4.1.1 第 13-14 行：行偏移

```cpp
logits += int64_t(blockIdx.x)*nclasses;
labels += int64_t(blockIdx.x)*nclasses;
```

**`blockIdx.x` 就是样本序号**（`grid=500`，每个 block 一个样本）。`nclasses=10`，所以第 `b` 个 block 处理 `logits[b*10 .. b*10+10)`。

**`int64_t` 强制转换**：`blockIdx.x` 是 `uint3` 的成员（`unsigned int`），`nclasses` 是 `int`，两者相乘是 32 位。对 500×10=5000 没问题，但真实 LLM 的 `nclasses=150000`、`batch=8192`，乘积超过 2^31，必须用 64 位。

#### 5.4.1.2 第 18-26 行：求 max（数值稳定性）

```cpp
for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    max_logit = fmaxf(max_logit, logits[i]);
    if (use_shared) tmp[i] = val;
}
max_logit = warp_reduce_max(max_logit);
```

**`i += WARP_SIZE`**：每个线程处理 `i, i+32, i+64, ...`。`nclasses=10` 时只有 `threadIdx.x < 10` 的线程进入循环，其余 22 个线程的 `max_logit` 保持 `-INFINITY`。

**`warp_reduce_max`**：`__shfl_xor_sync` 树形归约，5 步（32 → 16 → 8 → 4 → 2 → 1）。

**为什么要减 max**：`expf(logit)` 在 `logit > 88` 时溢出成 `inf`。减掉 max 后最大指数是 0，`exp(0)=1`，不会溢出。这是 softmax 的标准数值稳定技巧。

#### 5.4.1.3 第 30-35 行：求和

```cpp
for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    const float logit_i = use_shared ? tmp[i] : logits[i];
    sum += expf(logit_i - max_logit);
}
sum = warp_reduce_sum(sum);
sum = logf(sum);
```

**`use_shared` 的作用**：如果 `nbytes_shared = nclasses * 4 <= smpbo`（本例 40 <= 101376），第一次循环时把 `logits` 缓存到 shared memory，后面两次直接从 shared 读，避免重复访问 global。

**`sum = logf(sum)`**：把 `sum` 变成 `log(sum)`，后面直接用 `logit - max - log(sum)`。

#### 5.4.1.4 第 39-43 行：loss 计算

```cpp
for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    loss += (logit_i - max_logit - sum) * labels[i];
}
loss = -warp_reduce_sum(loss) / (float)k;
```

**数学推导**：

```text
softmax(logit_i) = exp(logit_i - max) / sum_exp
log(softmax(logit_i)) = (logit_i - max) - log(sum_exp)
loss = -Σ labels_i * log(softmax(logit_i))      ← 交叉熵定义
     = -Σ labels_i * (logit_i - max - log(sum_exp))
     = -Σ labels_i * (logit_i - max - sum)      ← sum 已经是 log(sum_exp)
```

**`/ (float)k`**：`k = nrows = 500`，所以是“平均交叉熵”。这与 `ggml_opt` 的 `loss_per_datapoint=true` 对应。

**`labels` 是 one-hot**：只有 `labels[true_class]=1`，其余为 0。所以 `loss` 实际上只累加了一个非零项。但 kernel 不知道哪个是 1，所以遍历全部 10 个。

#### 5.4.1.5 第 45-49 行：只写一个结果

```cpp
if (threadIdx.x != 0) {
    return;
}
dst[blockIdx.x] = loss;
```

**`warp_reduce_sum` 之后所有线程的 `loss` 都相同**（广播归约），所以只需要线程 0 写。

**`dst[blockIdx.x]`**：每个 block 写一个 partial loss，`dst` 是 `ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x)`，500 个 float。

**之后 `sum_f32_cuda` 把 500 个 partial 求和成一个标量**（见 §5.4.2）。

---

### 5.5 ARGMAX

```text
ARGMAX grid=500 block=32 ne00=10
```

```cpp
const int64_t num_blocks = nrows;                                        // 500
const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);  // 32
```

**`ne00=10` 时 `num_threads=32`，22 个线程空转。** 这是“一个 warp 处理一行”的固定策略，对小 `ne00` 浪费明显，但 argmax 只在评估时用，不在训练热路径。

### 5.5.1 源码逐行：`argmax_f32`（`argmax.cu:8-40`）

```cpp
8   static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
9       const int64_t row = blockIdx.x;
10
11      float maxval = -FLT_MAX;
12      int   argmax = -1;
13      const float * rowx = x + row * ncols;
14
15      for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
16          const float val = rowx[col];
17          if (val > maxval) {
18              maxval = val;
19              argmax = col;
20          }
21      }
22
23  #pragma unroll
24      for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
25          const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
26          const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
27          if (val > maxval) {
28              maxval = val;
29              argmax = col;
30          }
31      }
32
33      if (threadIdx.x == 0) {
34          dst[row] = argmax;
35      }
36  }
```

**逐行**：

| 行 | 行为 | 说明 |
|---|---|---|
| 9 | `row = blockIdx.x` | **一个 block 处理一行**，`grid=500` |
| 13 | `rowx = x + row * ncols` | 行首指针，`ncols=10` |
| 15 | `col += blockDim.x` | 线程步长 = block 大小 = 32 |
| 17 | `val > maxval` | **严格大于**：相等时保留先遇到的（列号更小），保证确定性 |
| 24 | `offset = 16, 8, 4, 2, 1` | warp 内树形归约，5 步 |
| 25-26 | `__shfl_xor_sync` 同时交换 value 和 index | 必须成对交换，否则 value 和 index 会错配 |
| 33 | `threadIdx.x == 0` | 只有 lane 0 写结果 |

**`ne00=10` 时的线程浪费**：32 个线程只有 10 个进入第 15 行的循环，22 个线程的 `maxval` 保持 `-FLT_MAX`、`argmax` 保持 `-1`。归约时它们不参与竞争（`-FLT_MAX > 任何值` 为 false）。

**`__shfl_xor_sync` 的成对性**：如果只交换 `maxval` 不交换 `argmax`，那么当 lane 1 的 `maxval` 更大时，lane 0 会拿到 lane 1 的 `maxval` 但保留自己的 `argmax`——**索引错误**。所以 25-26 行必须同时交换。

**`ncols` 是 `int64_t` 但循环变量 `col` 是 `int32_t`**：因为 `ne00=10`，`int32` 够用。真实 LLM 的 `ncols`（词表大小）可能到 15 万，仍在 `int32` 范围内。

---

### 5.6 COUNT_EQUAL

```text
COUNT_EQUAL grid=4 block=32 ne=500 nsm=128 dne=128
```

```cpp
const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;     // 128
const int64_t dne = GGML_PAD((ne + 4*nsm - 1) / (4*nsm), CUDA_COUNT_EQUAL_CHUNK_SIZE);
// = PAD(ceil(500/512), 128) = PAD(1, 128) = 128

const dim3 blocks_num(std::min((int64_t)4*nsm, (ne + 127)/128), 1, 1);
// = min(512, 4) = 4
```

**为什么 `grid=4` 而不是 500？** 因为 `count_equal` 用 `atomicAdd` 累加：

```cpp
count_equal<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, dst_d, dne, ne);
```

每个 block 处理 `dne=128` 个元素，4 个 block 覆盖 512 ≥ 500。`atomicAdd` 的争用随 block 数增加，所以 block 数被限制在 `4*nsm`（512）以内。这是“归约类算子用少 block + 原子加”的典型取舍。

**`ncorrect` 是 I64**，但 kernel 用 `int` 累加（`GGML_ASSERT(ne < (1<<30))`），因为 batch 只有 500，不会溢出。

### 5.6.1 源码逐行：`count_equal`（`count-equal.cu:6-26`）

```cpp
6   template <typename T>
7   static __global__ void count_equal(const T * __restrict__ x, const T * __restrict__ y, int64_t * __restrict__ dst, const int64_t dk, const int64_t k) {
8       const int64_t i0 = (int64_t) blockIdx.x*dk;
9       const int64_t i1 = min(i0 + dk, k);
10
11      int nequal = 0;
12
13      for (int64_t i = i0 + threadIdx.x; i < i1; i += WARP_SIZE) {
14          const T xi = x[i];
15          const T yi = y[i];
16          nequal += xi == yi;
17      }
18
19      nequal = warp_reduce_sum(nequal);
20
21      if (threadIdx.x != 0) {
22          return;
23      }
24
25      atomicAdd((int *) dst, nequal);
26  }
```

**逐行**：

| 行 | 行为 | 说明 |
|---|---|---|
| 8-9 | `i0 = blockIdx.x * dk`，`i1 = min(i0+dk, k)` | 每个 block 负责 `dk=128` 个元素，最后一个 block 截断到 `k=500` |
| 13 | `i += WARP_SIZE` | 线程步长 32，不是 `blockDim.x`（虽然这里 `blockDim.x==32`） |
| 16 | `nequal += xi == yi` | **布尔值隐式转 int**：`true → 1`，`false → 0` |
| 19 | `warp_reduce_sum` | 32 个线程的计数求和 |
| 25 | `atomicAdd((int *) dst, nequal)` | **4 个 block 同时原子加** |

**`dk` 的含义**（`count-equal.cu:47`）：

```cpp
const int64_t dne = GGML_PAD((ne + 4*nsm - 1) / (4*nsm), CUDA_COUNT_EQUAL_CHUNK_SIZE);
// ne=500, nsm=128 → (500 + 511) / 512 = 1 → PAD(1, 128) = 128
```

**`dne=128` 是“每个 block 处理的元素数”**，向上取整到 128 的倍数。4 个 block × 128 = 512 ≥ 500。

**`blocks_num = min(4*nsm, ceil(ne/128)) = min(512, 4) = 4`**：

```cpp
const dim3 blocks_num(std::min((int64_t)4*nsm, (ne + CUDA_COUNT_EQUAL_CHUNK_SIZE - 1)/CUDA_COUNT_EQUAL_CHUNK_SIZE), 1, 1);
```

**为什么是 `4*nsm` 而不是 `nsm`**：`nsm=128`，`4*128=512` 是“每 SM 4 个 block”的经验值。但对 `ne=500`，只需要 4 个 block，所以取 `min`。

**`atomicAdd` 的精度问题**：`dst` 是 `int64_t*`，但原子加用 `(int*)`——因为 `ncorrect ≤ 500`，`int` 够用。源码有断言：

```cpp
GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
```

**为什么不用 `atomicAdd` on `int64_t`**：老 GPU（< sm_60）不支持 64 位原子加。ggml 为了保证兼容性，用 32 位累加，并在 host 侧断言不会溢出。

**`cudaMemsetAsync(dst_d, 0, ...)` 在 kernel 之前**（`count-equal.cu:49`）：

```cpp
CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));
```

**必须先清零**，否则 `atomicAdd` 会累加到上一次的结果上。

---

### 5.7 SCALE

```text
SCALE nelements=1 scale=0.500000 bias=0.000000
```

```cpp
const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;   // 256
```

`nelements=1` → `grid=1, block=256`，255 个线程空转。**这个 kernel 的存在只是为了把 loss 除以 2**（`opt_period=2`），却要一次完整 launch。实测空 kernel + sync 是 7.62 µs，所以这个标量 scale 的成本约 8 µs。

**优化点**：`ggml_scale` 对 `nelements=1` 完全可以 fused 到 `cross_entropy_loss` 的 `sum_f32` 里（乘个系数），但 ggml 保持了算子正交性，宁可多一次 launch。

### 5.8 OPT_STEP_ADAMW

```text
ADAMW grid=1    block=256 k=10        ← fc2.bias
ADAMW grid=2    block=256 k=500       ← fc1.bias
ADAMW grid=1532 block=256 k=392000    ← fc1.weight (784×500)
ADAMW grid=20   block=256 k=5000      ← fc2.weight (500×10)
```

```cpp
const dim3 block_dims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);   // 256
const dim3 block_nums((k + 255) / 256, 1, 1);
```

**验算**：

```text
k=10     → ceil(10/256)   = 1
k=500    → ceil(500/256)  = 2
k=392000 → ceil(392000/256) = 1532
k=5000   → ceil(5000/256) = 20   ✅
```

**kernel 是纯逐元素的**：

```cpp
const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
if (i >= k) return;

const float gi  = g[i];
const float gmi = g_m[i]*beta1 + gi*(1.0f - beta1);
const float gvi = g_v[i]*beta2 + gi*gi*(1.0f - beta2);
g_m[i] = gmi;
g_v[i] = gvi;

const float mh = gmi*beta1h;
const float vh = sqrtf(gvi*beta2h) + eps;
x[i] = x[i]*(1.0f - alpha*wd) - alpha*mh/vh;
```

**注意 `x[i]` 是原地写**：`dst->data == a->data`（视图），所以 AdamW 直接更新参数，不需要额外的 `cpy` 节点。这是 02 §4.3 讲过的视图设计在 kernel 层的落地。

**`pars[0..6]` 来自 CPU 张量**，通过 03 §1.5 / §2.4 的跨后端拷贝送进来。kernel 里每个线程都读 `pars[0..6]`，这 7 个值会被 L1/L2 缓存，不构成瓶颈。

---

### 5.8.1 源码逐行：`opt_step_adamw_f32`（`opt-step-adamw.cu:6-35`）

```cpp
6   static __global__ void opt_step_adamw_f32(
7       float * __restrict__ x, const float * __restrict__ g, float * __restrict__ g_m, float * __restrict__ g_v,
8       const float * __restrict__ pars, const int64_t k) {
9
10      const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
11
12      if (i >= k) {
13          return;
14      }
15
16      const float alpha  = pars[0];
17      const float beta1  = pars[1];
18      const float beta2  = pars[2];
19      const float eps    = pars[3];
20      const float wd     = pars[4];
21      const float beta1h = pars[5];
22      const float beta2h = pars[6];
23
24      const float gi = g[i];
25      const float gmi = g_m[i]*beta1 +    gi*(1.0f - beta1);
26      const float gvi = g_v[i]*beta2 + gi*gi*(1.0f - beta2);
27
28      g_m[i] = gmi;
29      g_v[i] = gvi;
30
31      const float mh =       gmi*beta1h;
32      const float vh = sqrtf(gvi*beta2h) + eps;
33
34      x[i] = x[i]*(1.0f - alpha*wd) - alpha*mh/vh;
35  }
```

**逐行**：

| 行 | 数学 | 说明 |
|---|---|---|
| 16-22 | 读 7 个超参 | 从 `pars`（CPU→GPU 拷贝的张量）读，每个线程都读，靠 L1/L2 缓存 |
| 24 | `gi = g[i]` | 当前梯度 |
| 25 | `gmi = m*β1 + g*(1-β1)` | 一阶动量：指数移动平均 |
| 26 | `gvi = v*β2 + g²*(1-β2)` | 二阶动量：梯度平方的 EMA |
| 28-29 | 写回 `m`、`v` | **原地更新**，这两个张量是静态 buffer 里的 |
| 31 | `mh = gmi * β1h` | bias correction：除以 `(1-β1^t)` |
| 32 | `vh = sqrt(gvi*β2h) + eps` | bias correction + 开方 + eps |
| 34 | `x = x*(1 - α*wd) - α*mh/vh` | 权重衰减 + Adam 更新 |

**`x[i] = x[i]*(1 - α*wd) - α*mh/vh` 是解耦权重衰减（AdamW）**，不是 L2 正则：

```text
AdamW:   θ = θ - α * (m̂ / (√v̂ + ε)) - α * wd * θ
         = θ * (1 - α*wd) - α * m̂/(√v̂ + ε)     ← 等价形式
```

**对比 L2 正则（Adam）**：`g = g + wd*θ`，然后走标准 Adam。区别在于 L2 的衰减项会被 `m/v` 缩放，而 AdamW 不会。这是 Loshchilov & Hutter 2019 的结论。

**`beta1h = 1/(1-β1^t)`** 在 host 侧算好（`ggml-opt.cpp:799`）：

```cpp
const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));
```

**为什么不在 kernel 里算 `powf`**：每个线程都算一次 `powf` 是浪费。7 个标量在 host 上算一次，然后 broadcast 到所有线程，是标准做法。

**`k=392000` 时 `grid=1532`**：`ceil(392000/256) = 1531.25 → 1532`。最后一个 block 只有 128 个有效线程。

**这个 kernel 是纯 memory-bound**：每个元素读 4 个 float（`x`、`g`、`m`、`v`），写 3 个（`x`、`m`、`v`），共 28 B 访存 + 约 10 次浮点运算。对 392,000 个元素 = 11 MB 访存，4090 的 1 TB/s 带宽下约 11 µs。实测这个 kernel 在 4 个 AdamW 里最慢，但总共不到 50 µs。

---

## 6 同步与数据流

### 6.1 一次 `ggml_backend_sched_graph_compute` 的完整时序

```text
① sched_alloc_graph
     split_graph → alloc_splits → gallocr
② compute_splits
     for each split:
       - 跨后端 input 拷贝（本例：adamw_params CPU→CUDA，每 opt batch 一次）
       - ggml_backend_graph_compute_async(split_backend, &split->graph)
③ sched_synchronize
     for each backend: ggml_backend_synchronize(backend) → cudaStreamSynchronize(ctx->stream())
```

### 6.2 断点：D2H 的三个读取

```text
D2H_GET#1 name=loss_cross_entropy_scaled size=4     tensor_data=0x7f5e6d004f00
D2H_GET#2 name=pred                      size=2000  tensor_data=0x7f5e6d004f80
D2H_GET#3 name=ncorrect                  size=8     tensor_data=0x7f5e6d005f80
```

```cpp
// ggml-opt.cpp:857
ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
```

每次 `tensor_get`：

```cpp
static void ggml_backend_cuda_buffer_get_tensor(...) {
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size,
                               cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```

**3 次同步**。实测 4 B 的 D2H+sync 是 6.27 µs，所以每 batch 的读取成本约 20 µs。

### 6.3 H2D：每 batch 100 次

```text
H2D_SET#50 name=labels off=  8400 size=  400
H2D_SET#51 name=images off=689920 size=31360
H2D_SET#100 name=labels off=18400 size=  400
```

`ggml_opt_dataset_get_batch` 按 shard 循环：

```cpp
for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {   // 50
    const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];
    const char * ptr_data   = (const char *) dataset->data->data   + ishard*dataset->nbs_data;
    const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
    ggml_backend_tensor_set(data_batch,   ptr_data,   ishard_batch*dataset->nbs_data,   dataset->nbs_data);   // 31,360 B
    ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels); // 400 B
}
```

`nbs_data = 784 × 10 × 4 = 31,360`，`nbs_labels = 10 × 10 × 4 = 400`。

**每 batch 100 次 `tensor_set`，每次带一次 `cudaStreamSynchronize`**。实测：

```text
31,360 B H2D+sync: 10.47 µs × 50 = 523 µs
   400 B H2D+sync:  5.41 µs × 50 = 271 µs
                                ≈ 794 µs / batch
```

代码插桩测得 `dataset_get_batch` 平均 **1,138 µs**，比纯 memcpy 的 794 µs 多约 344 µs，来自 permutation 随机访问 190 MB 数据集导致的 cache miss。

这是 mnist 训练的最大瓶颈（1,138 µs/batch，占 64%；见 05 §8.3）。对比之下 kernel 执行 490 µs、调度 158 µs。

**优化点**：

1. 用 `ggml_backend_tensor_set_async` + 一次 `synchronize`，可省 99 次同步。
2. 把 shard 调大（比如 50 张一组），次数从 100 降到 20。
3. 用 pinned memory 做中转，H2D 带宽可提升 2~3 倍。

---

### 6.4 源码逐行：`ggml_backend_tensor_set/get`（`ggml-backend.cpp:335-360` + `ggml-cuda.cu:785-799`）

#### 6.4.1 通用入口（`src/ggml-backend.cpp:335-348`）

```cpp
335  void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
336      GGML_ASSERT(tensor);
337      ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
338      GGML_ASSERT(buf != NULL && "tensor buffer not set");
339
340      if (size == 0) {
341          return;
342      }
343
344      GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
345      GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
346
347      buf->iface.set_tensor(buf, tensor, data, offset, size);
348  }
```

**逐行**：

| 行 | 行为 | 说明 |
|---|---|---|
| 337 | `tensor->view_src ? view_src->buffer : tensor->buffer` | **视图用源张量的 buffer**。因为 `tensor_set` 可能写一个视图（比如 `adamw_params` 的拷贝目标），它的 buffer 继承自源 |
| 338 | `buf != NULL` 断言 | 张量必须已经绑定后端 buffer；`data != NULL` 但 `buffer == NULL` 是非法状态 |
| 340-342 | `size == 0` 提前返回 | 避免对空区间调用 `cudaMemcpyAsync`（某些驱动会报错） |
| 347 | `buf->iface.set_tensor(...)` | **唯一的实现路径**，没有 host/device 分支 |

**关键：没有 `memcpy` 回退路径。** 所有张量都必须有 buffer（哪怕是 CPU buffer），`tensor_set` 总是走 backend 的 `set_tensor`。CPU buffer 的 `set_tensor` 实现就是 `memcpy`（`src/ggml-backend.cpp:2378`）：

```cpp
static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}
```

**`ggml_backend_cpu_buffer_type` 定义在 `ggml-backend.cpp:2470`，不在 `ggml-cpu` 里**——源码注释解释了原因：

```cpp
// this buffer type is defined here to make it available to all backends
```

因为 CUDA 的 host buffer（`CUDA_Host`）要复用 CPU 的 iface，放在 backend 层避免循环依赖。

**这样设计的好处**：上层代码不需要判断“这是 host 还是 device”，统一走 iface；坏处是多一次虚函数调用（约 2 ns，可忽略）。

#### 6.4.2 CUDA 实现（`src/ggml-cuda/ggml-cuda.cu:784-790`）

```cpp
784  static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
786      ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
787
788      ggml_cuda_set_device(ctx->device);
789      CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
790      CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
791  }
```

**逐行**：

| 行 | 行为 | 为什么 |
|---|---|---|
| 787 | `ggml_cuda_set_device(ctx->device)` | 多 GPU 时必须切到目标设备，否则 `cudaMemcpyAsync` 会失败 |
| 788 | `cudaMemcpyAsync(..., cudaStreamPerThread)` | **用 `cudaStreamPerThread` 而不是 `ctx->stream()`** |
| 789 | `cudaStreamSynchronize(cudaStreamPerThread)` | **每次拷贝都同步** |

**为什么用 `cudaStreamPerThread`**：

- `ctx->stream()` 是 CUDA backend 的私有 stream，用于计算图执行；
- `cudaStreamPerThread` 是“当前线程的默认 stream”，与计算 stream 独立。

**如果共用 `ctx->stream()`**：`tensor_set` 会和计算 kernel 串行化，但因为 `tensor_set` 在 `compute` 之前调用，实际上不会有冲突。**独立 stream 的好处是**：`tensor_set` 可以在另一个线程并发执行，不阻塞计算。

**代价**：两条 stream 之间没有隐式同步，所以必须显式 `cudaStreamSynchronize`。

**`cudaStreamSynchronize` 的成本**（微基准）：

```text
31,360 B H2D+sync: 10.47 µs
31,360 B async-only: 3.98 µs
```

**同步占 62%**。这就是 06 §5.2 优化的依据。

#### 6.4.3 `get_tensor`（`ggml-cuda.cu:792-798`）

```cpp
792  static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
793      ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
794
795      ggml_cuda_set_device(ctx->device);
796      CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
797      CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
798  }
```

**与 `set_tensor` 完全对称**，只是方向相反。

**`ggml_backend_tensor_get` 的调用点**（`ggml-opt.cpp:857-875`）：

```cpp
ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
```

**3 次同步**，每次 6~7 µs（微基准），合计约 20 µs/batch。

**为什么不能合并成一次**：三个张量在显存里不连续（`loss` 在 `off=0`，`pred` 在 `off=128`，`ncorrect` 在 `off=2176`），只能分三次拷。如果要合并，需要先把它们放到连续区域——得不偿失。

---

## 7 一个反直觉的现象：冷启动延迟

### 7.1 实测

同一份代码、同一个模型，连续运行 3 次：

```text
run1: training took 00:00:22
run2: training took 00:00:06
run3: training took 00:00:10
```

首次运行 22 s，后续 6~10 s。

### 7.2 三个原因

**① cuBLAS 首次调用要 JIT 选 kernel**

```text
cublasCreate+workspace:        24.53 ms
first sgemm 500x500x784:      156.331 ms   ← 首次
steady sgemm 500x500x784:       0.019 ms   ← 后续
```

首次 `cublasSgemm` 慢 **8,000 倍**，因为 cuBLAS 要在运行时为这个形状选择并编译 kernel（heuristic + JIT）。这是所有 cuBLAS 程序的通病。

**② kernel module 首次加载**

CUDA 的 fatbin 在首次 launch 某个 kernel 时才从 host 内存拷到 device 并加载（`cuModuleLoad`）。mnist 用到 8 个 kernel，每个首次 launch 都有几十微秒到几毫秒的开销。

**③ GPU 上下文初始化**

`cudaSetDevice` + `cudaMalloc` + context 创建，首次约 100~300 ms。

### 7.3 如何规避

- **预热**：正式计时前跑 1~2 个 batch，把 cuBLAS/kernel 的冷启动摊掉。
- **`cudaFuncSetAttribute` + 预加载**：对已知 kernel 提前调用一次空 launch。
- **避免每 batch 换形状**：cuBLAS 的 kernel 选择按形状缓存，形状固定就不会重复 JIT。

### 7.4 干扰说明（重要）

本次调试期间，云端 GPU 上还跑着一个 `llama-server`（Qwen3.8-27B，占 19.8 GB 显存、100% 利用率）。这导致：

```text
nvidia-smi: 100% util, 2760/3105 MHz, 362 W, 68°C
```

所以本文中所有 **时间类数据**（训练总时长、alloc 耗时、kernel 耗时）都只能作为**数量级参考**，不能作为性能基线。而 **地址、偏移、节点数、grid/block 尺寸、cuBLAS 参数** 这些结构性数据不受影响，是本文的主要依据。

---

## 8 本篇结论

1. **执行入口有两层跳过**：`is_view_or_noop`（NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE）和 `COMPUTE` 标记。28 个节点只有 23 次 kernel。
2. **融合在本例不生效**：`ggml_cuda_try_fuse` 只匹配 MoE/GDN/topk-MoE/单 token 的 MUL_MAT，mnist 的 `ne11=500` 不满足。
3. **`MUL_MAT` 五条路径**：MMVF（`ne11<=3`）/ MMF / MMVQ / MMQ / cuBLAS。mnist 走 cuBLAS，因为 batch 大、F32。
4. **cuBLAS 的 `(T, N)` 来自行主序 ↔ 列主序的转置**，`lda=K`、`ldc=M` 是布局的直接结果。
5. **广播 kernel 用 `hne0=ne0/2` 做 float2 向量化**，`fastdiv/fastmodulo` 把除法换成乘加。
6. **归约类算子两阶段**：block 内 warp reduce + 跨 block 的 cub/atomic。交叉熵、argmax、count_equal 都是这个模式。
7. **每 batch 100 次 H2D + 3 次 D2H，每次都同步**，是 CPU 侧的主要开销之一。
8. **冷启动是真实成本**：cuBLAS 首次 GEMM 慢 8,000 倍，首次运行比后续慢 3~4 倍。
9. **`fastdiv` 是 Granlund-Montgomery 除法**：2 条指令替代 20~40 周期的硬件除法，`k_bin_bcast` 每次用 4 次。
10. **广播靠 `i1 % ne11` 退化**：`fc1.bias` 的 `ne[1]=1` 让 `i11` 恒为 0，500 行读同一个元素，不需要额外分支。
11. **两阶段归约是标量输出的通用模式**：block 内 `warp_reduce_*`，跨 block `cub::DeviceReduce` 或 `atomicAdd`。
