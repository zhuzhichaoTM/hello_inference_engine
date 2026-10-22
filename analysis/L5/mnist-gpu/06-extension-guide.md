# 06 · 二次扩展与优化指南

> 本篇把前五篇的断点结论转成“能动手做的事”：
> **怎么加一个算子、怎么换 kernel、怎么改调度、怎么改显存策略、怎么定位瓶颈。**
>
> 每个改动都标注了**要碰的文件**、**验证断点**、**预期收益**。

---

## 0 先建立“改动成本”坐标系

```text
改动范围                涉及文件                       风险      收益
─────────────────────────────────────────────────────────────────────────
新增一个算子            ggml.h / ggml.c / ggml-cuda.*  低        中
替换某算子的 kernel     ggml-cuda/xxx.cu               低        高（热点）
调整 kernel 参数        ggml-cuda/xxx.cu / *.cuh       极低      中
改调度策略              ggml-backend.cpp               中        中（视规模）
改显存分配策略          ggml-alloc.c                   高        中（视规模）
改数据搬运              ggml-opt.cpp / backend         低        高（本例）
```

**原则**：先量化瓶颈，再选层动手。mnist 的实测瓶颈排序（代码插桩，见 05 §8.3）：

```text
① 100 次 H2D + sync（dataset_get_batch）   1,138 µs/batch   64%
② kernel 执行 + D2H（ggml_opt_eval）         490 µs/batch   27%
③ 选图 + split + alloc（ggml_opt_alloc）     158 µs/batch    9%
```

**① 是 ② 的 2.3 倍、③ 的 7.2 倍** —— mnist 的瓶颈**不在 GPU 计算，也不在调度，而在数据搬运**。

> ⚠️ 早期用 gdb 断点计时得出“gallocr 占 4.5 ms、是最大瓶颈”的结论是**错误的**（gdb 断点命中本身有 50~500 µs 开销）。本文所有性能数字以代码插桩和独立微基准为准，详见 05 §8.1-8.3。

---

## 1 新增一个算子：以 `RELU_BACK` 为例

### 1.1 为什么选这个例子

当前 mnist 的 ReLU 反向是**两个节点**：

```text
node_22       = STEP(node_4)                  ← 生成 0/1 mask
grad for node_2 = MUL(node_22, grad for node_5)  ← 逐元素乘
```

两次 kernel launch（实测各 ~8 µs 固定开销 + 计算）。如果合成一个 `RELU_BACK`：

```text
grad for node_2 = RELU_BACK(node_4, grad for node_5)
```

一次 launch，省一次 ~8 µs。对 mnist 是 0.4% 的收益，但对真实 LLM（每层都有激活反向）是可观的。

### 1.2 五处改动

**① `include/ggml.h`：加枚举**

```cpp
enum ggml_op {
    ...
    GGML_OP_OPT_STEP_ADAMW,
    GGML_OP_OPT_STEP_SGD,
    GGML_OP_RELU_BACK,          // ← 新增（注意：加在末尾，避免破坏已有序列化）
    GGML_OP_GLU,
    GGML_OP_COUNT,
};
```

⚠️ **必须加在 `GGML_OP_COUNT` 之前、且不要插在中间**。`ggml_op` 的值会出现在调试输出、可能的序列化格式里；插在中间会让所有后续 op 的数值偏移。

**② `src/ggml.c`：三处**

```cpp
// (a) 名字表
static const char * GGML_OP_NAME[GGML_OP_COUNT] = {
    ...
    "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",
    "RELU_BACK",                // ← 新增
    "GLU",
};

// (b) 符号表（用于 ggml_graph_dump_dot）
static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    ...
    "relu_back(x, dy)",
};

// (c) 算子构造函数
struct ggml_tensor * ggml_relu_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,      // ReLU 的输入
        struct ggml_tensor  * dy) {   // 上游梯度
    GGML_ASSERT(ggml_are_same_shape(x, dy));

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, x->ne, NULL, 0);
    result->op     = GGML_OP_RELU_BACK;
    result->src[0] = x;
    result->src[1] = dy;
    return result;
}
```

**③ `src/ggml.c`：反向规则**

在 `ggml_compute_backward` 的 `switch` 里加：

```cpp
case GGML_OP_RELU_BACK: {
    // d/dx [relu_back(x, dy)] = relu'(x) * dy，但反向的反向很少用，标为不可微
    // 如果只是想让 relu 的反向走这个算子，需要改的是 GGML_OP_UNARY 的 RELU 分支
} break;
```

**更实际的做法**：不改 `ggml_compute_backward` 的 `RELU` 分支语义，而是**在 CUDA 后端把 `STEP + MUL` 融合成一个 kernel**（见 §2.1）。这样不需要动 IR 层，风险最低。

**④ `src/ggml-cuda/ggml-cuda.cu`：能力声明 + 分发**

```cpp
// supports_op
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        ...
        case GGML_OP_RELU_BACK:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]);
        ...
    }
}

// compute_forward
case GGML_OP_RELU_BACK:
    ggml_cuda_relu_back(ctx, dst);
    break;
```

**⑤ `src/ggml-cuda/unary.cu`：kernel**

```cpp
static __global__ void relu_back_kernel(const float * x, const float * dy, float * dst, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    dst[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}

void ggml_cuda_relu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * dy = dst->src[1];
    const int k = (int) ggml_nelements(x);
    const int num_blocks = (k + 255) / 256;
    relu_back_kernel<<<num_blocks, 256, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) dy->data, (float *) dst->data, k);
}
```

**⑥ `src/ggml-cuda/unary.cuh`：声明**

```cpp
void ggml_cuda_relu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

### 1.3 验证断点

```gdb
break ggml_cuda_relu_back
commands
silent
printf "RELU_BACK k=%d grid=%d\n", k, (k+255)/256
continue
end
```

期望看到：原本 `STEP` + `MUL` 两次 `CUDA_FWD` 变成一次 `RELU_BACK`，`gb_grad` 的节点数从 28 降到 27。

### 1.4 一个更省事的替代方案

**不改 IR，改 CUDA 后端的 `ggml_cuda_try_fuse`**：

```cpp
// ggml-cuda.cu:3436
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];

    // 新增：STEP + MUL → relu_back
    if (node->op == GGML_OP_MUL && i > 0 && cgraph->nodes[i-1]->op == GGML_OP_UNARY &&
        ggml_get_unary_op(cgraph->nodes[i-1]) == GGML_UNARY_OP_STEP) {
        // 检查拓扑：node->src[0] == step_node
        if (node->src[0] == cgraph->nodes[i-1]) {
            ggml_cuda_relu_back_fused(*cuda_ctx, node, cgraph->nodes[i-1]->src[0]);
            return 1;   // 跳过 1 个节点
        }
    }
    ...
}
```

**收益**：不改 IR、不改序列化、不影响 CPU 后端。**代价**：融合逻辑分散在 `try_fuse` 里，且只在 CUDA 后端生效。

**推荐**：真实项目里优先用 `try_fuse`（llama.cpp 就是这么做的），只有在“这个算子需要跨后端一致语义”时才改 IR。

---

### 1.5 源码逐行：`ggml_cuda_try_fuse` 的接入点（`ggml-cuda.cu:3436-3530`）

如果要走“不改 IR、只加融合”的路线，必须先理解 `try_fuse` 的结构。

```cpp
3436  static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
3437
3438      static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
3439      if (disable_fusion) {
3440          return 0;
3441      }
3442
3443      ggml_tensor * node = cgraph->nodes[i];
3444
3445      if (node->op == GGML_OP_MUL) {
3446          ggml_cuda_moe_weighted_reduction_match match;
3447          if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
3448              const int output_idx = i + match.node_count - 1;
3449              if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
3450                  ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
3451                  return match.node_count - 1;
3452              }
3453          }
3454      }
3455
3456      if (node->op == GGML_OP_GATED_DELTA_NET) {
3457          ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
3458          const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
3459          if (nodes_to_skip > 0) {
3460              ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
3461              return nodes_to_skip;
3462          }
3463      }
3464
3465      if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
3466              cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
3467          ggml_cuda_topk_moe_args args;
3468          const bool can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
3469          ...
3470      }
3471
3472      if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
3473          const bool can_fuse = ggml_cuda_try_fuse_mul_mat(cuda_ctx, node, ...);
3474          ...
3475      }
3476
3477      return 0;
3478  }
```

**结构**：一系列 `if (node->op == X)` 的“模式匹配”，每个模式返回“被融合掉的后续节点数”。

**返回值语义**：

```text
0          没有融合，调用方继续执行当前节点
N > 0      融合了 N 个后续节点，调用方 i += N 跳过它们
```

**`disable_fusion`**（3438-3441）：`GGML_CUDA_DISABLE_FUSION=1` 时直接返回 0，用于 A/B 对比融合收益。

**`ggml_cuda_check_fusion_memory_ranges`**（3449）：融合前检查输入输出是否有内存重叠。因为 gallocr 可能让相邻节点原地复用，融合后读写顺序变化会导致错误。

**本例为什么全返回 0**：

| 分支 | 条件 | mnist |
|---|---|---|
| MoE weighted reduction | `node->op == MUL` + 特定拓扑 | `node_23 = MUL(node_22, grad)` 不匹配 MoE 拓扑 |
| gated_delta_net | `op == GATED_DELTA_NET` | 无 |
| topk-moe | `op == UNARY/SOFT_MAX/ARGSORT` + MoE 拓扑 | `node_5 = RELU` 不匹配 |
| MUL_MAT + bias/GLU | `dst->ne[1] == 1` | `ne[1]=500`，不满足 |

**加 `RELU_BACK` 融合的位置**：在 3445 的 `if (node->op == GGML_OP_MUL)` 之前插入：

```cpp
if (node->op == GGML_OP_MUL && i > 0) {
    ggml_tensor * step = cgraph->nodes[i-1];
    if (step->op == GGML_OP_UNARY && ggml_get_unary_op(step) == GGML_UNARY_OP_STEP &&
        node->src[0] == step) {
        ggml_cuda_relu_back_fused(*cuda_ctx, node, step->src[0]);
        return 1;   // 跳过 step 节点
    }
}
```

**为什么是 `i-1` 而不是搜索**：因为 `try_fuse` 在遍历时被调用，`i-1` 是刚执行完的上一个节点。如果 STEP 和 MUL 之间隔着别的节点，这个模式就不匹配——这是“局部匹配”的保守设计。

**`return 1` 的语义**：融合了 1 个后续节点（`i+1`）。但这里 STEP 在 `i-1`，MUL 在 `i`——实际上应该是“当前节点 + 前一个节点”，返回值应该表示“跳过多少个后续节点”。如果 STEP 已经执行过了（因为它在 `i-1`），那么这里只需要执行融合 kernel 并 `return 0`（不跳过任何后续节点）。

**正确的做法**：把融合模式改为“STEP 在前，MUL 在后”，在 `i` 指向 MUL 时检查 `i-1` 是否是 STEP，然后：

```cpp
if (/* 匹配 */) {
    ggml_cuda_relu_back_fused(*cuda_ctx, node, cgraph->nodes[i-1]->src[0]);
    return 0;   // STEP 已经执行（浪费了一次 launch），但结果被覆盖
}
```

**这样只省了 MUL 的 launch，没省 STEP 的**。要真正省两次 launch，需要在 STEP 的位置做“前瞻”：

```cpp
if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_STEP &&
    i + 1 < cgraph->n_nodes && cgraph->nodes[i+1]->op == GGML_OP_MUL &&
    cgraph->nodes[i+1]->src[0] == node) {
    ggml_cuda_relu_back_fused(*cuda_ctx, cgraph->nodes[i+1], node->src[0]);
    return 1;   // 跳过 MUL
}
```

**这才是正确的接入点**：在 STEP 处前瞻，一次 kernel 替代两个。

---

## 2 替换/优化 kernel 的四个切入点

### 2.1 切入点一：标量算子的“过度 launch”

**现状**：

```text
SCALE nelements=1 → grid=1, block=256, 255 个线程空转
空 kernel + sync 实测 7.62 µs
每 batch 2 次 SCALE（前向 loss 缩放 + 反向梯度缩放）≈ 15 µs
```

**优化 A：在 kernel 里判断 `nelements == 1`，走特化分支**

```cpp
void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ...
    if (nelements == 1) {
        // 单元素直接用 cudaMemcpyAsync 或者写一个小 kernel
        scale_f32_single<<<1, 1, 0, stream>>>(src0_d, dst_d, scale, bias);
        return;
    }
    scale_f32_cuda(src0_d, dst_d, scale, bias, nelements, stream);
}
```

省的是 255 个线程的调度，但 launch 开销不变。

**优化 B：把 scale 折进上游算子**

```text
loss_scaled = 0.5 × cross_entropy_loss(logits, labels)
```

可以让 `cross_entropy_loss` 的 kernel 直接乘 `0.5`：

```cpp
loss = -warp_reduce_sum(loss) / (float)k * scale;   // scale 从 op_params 读
```

**代价**：破坏算子正交性，`CROSS_ENTROPY_LOSS` 的语义变成“带缩放的交叉熵”。

**优化 C：用 `GGML_OP_CUSTOM` 或 graph_optimize 做常量折叠**

在 `ggml_backend_cuda_graph_optimize` 里检测“SCALE 的输入是标量且 scale 是常量”，把系数写进上游的 `op_params`，删掉这个节点。

### 2.2 切入点二：`REPEAT_BACK` 的极度低效

**现状**：

```text
REPEAT_BACK grid=(1,1,1) block=(32,1,1) ne=(10,1) ne0src=(10,500)
```

`grid=(1,1,1)` 意味着只有 **1 个 block、32 个线程**，其中只有 10 个线程干活，每个线程累加 500 次。总共 5,000 次全局内存读，串行化在 10 个线程上。

**优化：改成 warp-level reduce**

```cpp
// 当前：一个线程沿 ne01 循环
for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) sum += src[...];

// 优化：一个 warp 沿 ne01 并行，然后 warp_reduce_sum
static __global__ void k_repeat_back_warp(
        const float * src, float * dst, int64_t ne00, int64_t ne01, size_t s00, size_t s01) {
    const int64_t i0 = blockIdx.x * blockDim.x + threadIdx.x;   // 每个线程负责一个 i0
    if (i0 >= ne00) return;
    float sum = 0;
    for (int64_t i1 = 0; i1 < ne01; ++i1) {
        sum += src[i1*s01 + i0*s00];
    }
    dst[i0] = sum;
}
```

**收益**：对 `[10,500] → [10,1]`，从 10 线程串行 500 次变成 10 线程并行 500 次（访问模式不变，但循环体内可以向量化）。真正的加速需要**把 i1 维度也并行化**：

```cpp
// 一个 block 处理一个 i0，block 内线程沿 i1 并行，最后 block_reduce
const dim3 grid(ne0);
const dim3 block(256);
k_repeat_back_reduce<<<grid, block, 0, stream>>>(...);
```

**注意**：`REPEAT_BACK` 在 mnist 里只出现 2 次（`[10,500]→[10,1]` 和 `[500,500]→[500,1]`），数据量小。真实 LLM 里它出现在 MoE 的专家聚合，才值得优化。

### 2.3 切入点三：`ARGMAX` 的线程浪费

**现状**：

```text
ARGMAX grid=500 block=32 ne00=10
```

`ne00=10` 却开 32 个线程，22 个空转。而且每行一个 block，500 个 block 里每个只有 10 个有效线程。

**优化：一个 warp 处理多行**

```cpp
// 当前：1 行/block
// 优化：1 行/warp，4 warps/block，则 grid=ceil(500/4)=125
const dim3 block(128);            // 4 warps
const dim3 grid((nrows + 3) / 4);
argmax_f32_multirow<<<grid, block, 0, stream>>>(src0_d, dst_d, ne00, nrows);
```

kernel 内：

```cpp
const int warp_id = threadIdx.x / WARP_SIZE;
const int lane_id = threadIdx.x % WARP_SIZE;
const int64_t row = blockIdx.x * 4 + warp_id;
if (row >= nrows) return;
// ... 每行一个 warp 做 reduce
```

**收益**：block 数从 500 降到 125，launch 开销降低，SM 占用更合理。对 mnist 可忽略（argmax 只在评估时跑），对真实 LLM 的 top-k 采样有意义。

### 2.4 切入点四：`ADD` 的广播路径

**现状**：

```text
ADD#1 grid=(2,500,1) block=(128,1,1) ne=(500,500) src0=node_2 src1=fc1.bias
```

`fc1.bias` 是 `[500,1]`，沿 `ne1` 广播。kernel 里：

```cpp
const uint32_t i11 = fastmodulo(i1, ne11);      // i1 % 1 = 0
result = bin_op(result, src1[i_src1 + i10*s10]);
```

**每个线程都做一次 `fastmodulo`**，虽然已经优化成乘加，但 `ne11=1` 时可以直接跳过：

```cpp
// 特化：ne11 == 1 时，src1 只沿 i0 变化
if (ne11 == 1 && ne12 == 1 && ne13 == 1) {
    // 走一个更简单的 kernel：dst[i] = src0[i] + bias[i % ne10]
}
```

**收益**：省掉 `i11/i12/i13` 三次 modulo 和三次地址乘加。对 memory-bound 的 ADD 是 5~10% 的提升。

---

## 3 调度优化：re-reserve 的真实成本与消除方法

### 3.0 先算清楚值不值得

`gb_grad`(28 节点) 和 `gb_opt`(34 节点) 每个 batch 交替，`ggml_gallocr_needs_realloc` 每次都因“节点数不同”返回 true，触发一次完整的 reserve + alloc。

**成本（代码插桩实测，05 §8.2）**：

```text
split_graph   ≈ 15 µs/batch
alloc_splits  ≈ 42 µs/batch
合计          ≈ 57 µs/batch
3,420 batch × 57 µs ≈ 0.19 s，占 6 s 训练的 3.2%
```

**结论**：可观测，但**不是瓶颈**。相比之下 H2D 占 64%（1,138 µs/batch）。

> 早期用 gdb 断点测出“4.5 ms/batch、占一半时间”是错的——断点命中本身有 50~500 µs 开销。
> 详见 05 §8.1。

所以本节的价值在于**方法**（怎么改调度），而不是**收益**。如果你只是想加速 mnist，直接跳到 §5 优化 H2D。

### 3.1 方案 A：对齐两张图的节点数（最低侵入）

**思路**：让 `gb_grad` 也包含 4 个 AdamW 节点，只是不执行，这样两张图都是 34 节点。

**先看一个错误写法**：

```cpp
// ❌ 错误：ggml_build_forward_expand 对 op=NONE 的节点会放进 leafs[]，nodes[] 不变
struct ggml_tensor * dummy = ggml_view_tensor(opt_ctx->ctx_compute, params[i]);
dummy->op = GGML_OP_NONE;
ggml_build_forward_expand(opt_ctx->gb_grad, dummy);
```

`ggml_visit_parents_graph` 的判定：

```cpp
if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
    cgraph->leafs[cgraph->n_leafs++] = node;       // ← 进 leafs
} else {
    cgraph->nodes[cgraph->n_nodes++] = node;       // ← 进 nodes
}
```

`op=NONE` 且非 PARAM → 进 `leafs[]`，`n_nodes` 不变。所以 `needs_realloc` 仍然会因为“节点数不同”返回 true（虽然 `n_leafs` 也变了，同样触发）。

**正确做法**：总是用 `gb_opt`，在非 opt batch 把 AdamW 节点的 `COMPUTE` 清掉。这需要改 `ggml_opt_alloc` 的选图逻辑：

```cpp
// 当前
case GGML_OPT_BUILD_TYPE_GRAD: graph = opt_ctx->gb_grad; break;
case GGML_OPT_BUILD_TYPE_OPT:  graph = opt_ctx->gb_opt;  break;

// 改成：总是用 gb_opt，但在非 opt batch 把 AdamW 节点的 COMPUTE 清掉
graph = opt_ctx->gb_opt;
for (auto * n : adamw_nodes) {
    if (build_type == GRAD) n->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    else                    n->flags |=  GGML_TENSOR_FLAG_COMPUTE;
}
```

**注意**：`ggml_graph_reset` 和 `gallocr` 都依赖 flags，需要同步验证。

### 3.2 方案 B：显式 `ggml_backend_sched_reserve`（**不解决问题，仅作分析）**

**改动**：`src/ggml-opt.cpp:549`（`ggml_opt_init`）末尾

```cpp
// 用最大图（gb_opt）预留一次
if (opt_ctx->gb_opt) {
    ggml_backend_sched_reserve(opt_ctx->backend_sched, opt_ctx->gb_opt);
}
```

**原理**：

```cpp
bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    ggml_backend_sched_synchronize(sched);
    ggml_backend_sched_split_graph(sched, measure_graph);      // 按最大图 split
    ggml_gallocr_reserve_n(sched->galloc, &sched->graph, ...); // 按最大图规划
    ggml_backend_sched_reset(sched);
    return true;
}
```

之后每次 `alloc_graph` 时：

```text
needs_realloc → 节点数不同 → 仍然返回 true
```

**这没有解决问题**——`needs_realloc` 仍然会因为节点数不同返回 true，每次 `alloc_graph` 还是要重做一遍生命周期分析。

**它能解决的是另一半问题**：保证 buffer 不会被重新 `cudaMalloc`。但本例已经天然满足（`gb_opt` 只比 `gb_grad` 大 128 B，第一次 reserve 后就固定了），所以这个方案在 mnist 上**几乎没有收益**。

**它的适用场景**：图从“小”变“大”时（比如 decode 的序列长度增长），提前 reserve 避免运行时 realloc。

**真正的解法**是让 `needs_realloc` 的判定基于“缓存的规划是否可复用”而不是“节点数是否相同”——见 3.3。

### 3.3 方案 C：gallocr 缓存多张图的规划（治本）

**思路**：给 gallocr 加 2 个缓存槽，按“图指纹”索引。`gb_grad` 和 `gb_opt` 各命中自己的规划，交替时不触发 realloc。

**收益上限**：`alloc_splits` 从 42 µs 降到 ~5 µs，`37 µs × 3,400 ≈ 0.13 s`，占 6 s 训练的 **2.1%**。

**结论**：**在 mnist 上不值得做**。它真正的价值在“图很大、单次 reserve 到毫秒级”的场景。完整改造代码见 §3.5.2。

### 3.4 三种方案的取舍

| 方案 | 改动量 | 收益 | 风险 | 适用 |
|---|---|---|---|---|
| A 对齐节点数 | 中（改 ggml-opt） | 消除 reserve | 中（flags 交互复杂） | 只想改训练流程 |
| B 显式 reserve | 小 | 部分收益 | 低 | 快速验证 |
| C 指纹缓存 | 大（改 gallocr） | 彻底消除 | 高（核心分配器） | 通用优化，可上游 |

**建议路径**：

```text
① 先插桩量化（05 §8.2）——mnist 上只有 42 µs/batch，不值得改
② 如果是大图场景（单次 reserve 到毫秒级），直接上 C
③ B 只适合"图从大变到更大"的序列增长场景
```

---

### 3.5 源码逐行：`ggml_gallocr_needs_realloc` 与指纹缓存改造点

#### 3.5.1 当前实现（`src/ggml-alloc.c:997-1050`）

```cpp
997   static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
998       size_t node_size = 0;
999       if (!node->data && !node->view_src) {
1000          // If we previously had data but don't now then reallocate
1001          if (talloc->buffer_id < 0) {
1002              return false;
1003          }
1004          node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
1005      }
1006      return talloc->size_max >= node_size;
1007  }
1008
1009  static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
1010      if (galloc->n_nodes != graph->n_nodes) {
1011  #ifndef NDEBUG
1012          GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
1013  #endif
1014          return true;
1015      }
1016
1017      if (galloc->n_leafs != graph->n_leafs) {
1018          GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
1019          return true;
1020      }
1021
1022      for (int i = 0; i < graph->n_nodes; i++) {
1023          struct ggml_tensor * node = graph->nodes[i];
1024          struct node_alloc * node_alloc = &galloc->node_allocs[i];
1025
1026          if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
1027              GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
1028              return true;
1029          }
1030
1031          for (int j = 0; j < GGML_MAX_SRC; j++) {
1032              struct ggml_tensor * src = node->src[j];
1033              if (src == NULL) continue;
1034              if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
1035                  GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", ...);
1036                  return true;
1037              }
1038          }
1039      }
1040
1041      return false;
1042  }
```

**三级判定**：

| 行 | 判据 | 语义 |
|---|---|---|
| 1010 | `n_nodes` 不同 | 图结构变了 |
| 1017 | `n_leafs` 不同 | 外部输入变了 |
| 1026/1034 | 某个张量的 `size_max` 不够 | 形状变大 |

**`ggml_gallocr_node_needs_realloc` 的细节**（997-1007）：

```cpp
if (!node->data && !node->view_src) {     // 只有 gallocr 管理的张量才检查
    if (talloc->buffer_id < 0) {
        return false;                     // 上次没分配过 → 需要
    }
    node_size = get_alloc_size(buft, node);
}
return talloc->size_max >= node_size;     // 上次分配的空间够不够
```

**`buffer_id < 0` 返回 false 的含义**：`false` = “需要 realloc”。因为上次这个张量是“预分配/视图”，现在需要 gallocr 分配，所以必须重新规划。

#### 3.5.2 指纹缓存改造

**目标**：让 `gb_grad`（28 节点）和 `gb_opt`（34 节点）各自有一份规划，交替时不触发 realloc。

**改造点 1**：`struct ggml_gallocr` 加缓存槽（`src/ggml-alloc.c:475-495` 附近）

```cpp
struct ggml_gallocr {
    ...
    struct node_alloc * node_allocs;
    int n_nodes;
    struct leaf_alloc * leaf_allocs;
    int n_leafs;

    // 新增
    struct {
        uint64_t fingerprint;
        struct node_alloc * node_allocs;
        struct leaf_alloc * leaf_allocs;
        int n_nodes;
        int n_leafs;
        bool valid;
    } cache[2];          // 2 个槽，覆盖 gb_grad / gb_opt
};
```

**改造点 2**：指纹函数

```cpp
static uint64_t ggml_graph_fingerprint(const struct ggml_cgraph * graph) {
    uint64_t h = 1469598103934665603ULL;   // FNV-1a offset basis

    #define FNV_MIX(v) do { h ^= (uint64_t)(v); h *= 1099511628211ULL; } while (0)

    FNV_MIX(graph->n_nodes);
    FNV_MIX(graph->n_leafs);

    for (int i = 0; i < graph->n_nodes; ++i) {
        const struct ggml_tensor * n = graph->nodes[i];
        FNV_MIX(n->op);
        FNV_MIX(n->type);
        FNV_MIX(n->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT |
                            GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_LOSS));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            FNV_MIX(n->ne[d]);
        }
    }

    #undef FNV_MIX
    return h;
}
```

**为什么要把 `flags` 也纳入**：`INPUT` 标记会影响 `alloc_graph_impl` 的 738-740 行（提前分配），`OUTPUT` 会影响 `free_node` 的 693 行（不释放）。所以 flags 变了，规划就可能不同。

**为什么不纳入 `data`**：`data` 在两次 reserve 之间可能变（因为上次分配了），但它不影响“规划”（offset 布局），只影响“是否需要 init”。

**改造点 3**：`needs_realloc` 改成查缓存

```cpp
static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    const uint64_t fp = ggml_graph_fingerprint(graph);

    for (int c = 0; c < 2; ++c) {
        if (galloc->cache[c].valid && galloc->cache[c].fingerprint == fp) {
            // 命中：把缓存的规划换回当前
            if (galloc->node_allocs != galloc->cache[c].node_allocs) {
                free(galloc->node_allocs);
                free(galloc->leaf_allocs);
                galloc->node_allocs = galloc->cache[c].node_allocs;
                galloc->leaf_allocs = galloc->cache[c].leaf_allocs;
                galloc->n_nodes     = galloc->cache[c].n_nodes;
                galloc->n_leafs     = galloc->cache[c].n_leafs;
            }
            return false;              // ← 不需要 realloc
        }
    }

    // 未命中：走原逻辑
    ... 原有检查 ...
}
```

**改造点 4**：reserve 成功后写缓存

```cpp
static bool ggml_gallocr_reserve_n_impl(...) {
    ...
    // 原有逻辑
    ...

    // 新增：写缓存
    const uint64_t fp = ggml_graph_fingerprint(graph);
    int slot = -1;
    for (int c = 0; c < 2; ++c) {
        if (!galloc->cache[c].valid) { slot = c; break; }
    }
    if (slot < 0) slot = 0;            // 两个都满了，覆盖第一个（LRU 的简化版）

    free(galloc->cache[slot].node_allocs);
    free(galloc->cache[slot].leaf_allocs);
    galloc->cache[slot].node_allocs = (struct node_alloc *)malloc(sizeof(struct node_alloc) * galloc->n_nodes);
    galloc->cache[slot].leaf_allocs = (struct leaf_alloc *)malloc(sizeof(struct leaf_alloc) * galloc->n_leafs);
    memcpy(galloc->cache[slot].node_allocs, galloc->node_allocs, sizeof(struct node_alloc) * galloc->n_nodes);
    memcpy(galloc->cache[slot].leaf_allocs, galloc->leaf_allocs, sizeof(struct leaf_alloc) * galloc->n_leafs);
    galloc->cache[slot].fingerprint = fp;
    galloc->cache[slot].n_nodes     = galloc->n_nodes;
    galloc->cache[slot].n_leafs     = galloc->n_leafs;
    galloc->cache[slot].valid       = true;

    return true;
}
```

**内存代价**：

```text
struct node_alloc = { struct tensor_alloc dst; struct tensor_alloc src[10]; }
struct tensor_alloc = { int buffer_id; struct buffer_address addr; size_t size_max; }
                    ≈ 4 + 16 + 8 = 28 B（对齐后 32 B）
struct node_alloc   ≈ 32 + 10*32 = 352 B

28 节点 × 352 B = 9,856 B
34 节点 × 352 B = 11,968 B
两个槽合计 ≈ 22 KB
```

**对 24 GB 显存微不足道，对 host 内存也可忽略。**

#### 3.5.3 验证方法

```gdb
break ggml_gallocr_needs_realloc
commands
silent
printf "FP=%lx hit=%d\n", fp, (galloc->cache[0].valid && galloc->cache[0].fingerprint == fp) || (galloc->cache[1].valid && galloc->cache[1].fingerprint == fp)
continue
end
```

**期望**：第 3 个 batch 之后全部 `hit=1`，`alloc_splits` 的耗时从 42 µs 降到 ~5 µs。

**收益上限**：`37 µs × 3,400 ≈ 0.13 s`，占 6 s 训练的 2.1%。**仍然不是主要优化点**（主要优化点在 H2D，见 §5）。

---

## 4 显存优化

### 4.1 现状：11 MiB 里有 3.4 MiB 是“可复用的”

```text
静态 buffer：3,158,272 + 4,795,520 = 7,953,792
计算 buffer：3,588,352
cuBLAS workspace：4,194,304
```

### 4.2 优化点一：梯度累加器的 F16 存储

**现状**：5 个梯度累加器共 1,590,400 B（F32）。

**优化**：用 F16 存梯度，AdamW 时转 F32：

```cpp
opt_ctx->grad_accs[i] = ggml_new_tensor(ctx, GGML_TYPE_F16, GGML_MAX_DIMS, node->ne);
```

**收益**：省 50% = 795 KB。

**代价**：反向的 `ADD(leaf, node)` 需要类型转换，多一个 kernel；精度损失对 mnist 可忽略，对大模型需要 loss scaling。

### 4.3 优化点二：AdamW 状态的共享

**现状**：`m` 和 `v` 各 1,590,400 B，共 3,180,800 B。

**优化**：如果多个参数的 `ne` 相同（比如所有 bias 都是 `[N,1]`），可以共享一块大 buffer，用 offset 切片：

```cpp
// 一个 [total_elems] 的大张量，每个参数一个 view
struct ggml_tensor * m_all = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total_elems);
struct ggml_tensor * m_fc1 = ggml_view_1d(ctx, m_all, 392000, 0);
struct ggml_tensor * m_fc2 = ggml_view_1d(ctx, m_all, 5000, 392000*4);
```

**收益**：减少 buffer 数量，降低碎片。**显存总量不变**（数据量一样），但对 gallocr 的 `n_buffers` 有好处。

**真正的省显存**：用 `v` 的低秩近似（如 Adafactor 的 `v ≈ outer(u, 1)`），省 50%。这需要改 AdamW kernel。

### 4.4 优化点三：激活的 checkpointing

**现状**：`node_2`（MUL_MAT 输出，1 MB）和 `node_5`（RELU 输出，1 MB）都要留到反向。

**优化**：不存 `node_5`，反向时从 `node_4` 重算：

```text
前向：只存 node_4（ADD 输出）
反向：需要 node_5 时，重算 relu(node_4)
```

**收益**：省 1 MB。**代价**：多一次 RELU kernel。

**注意**：gallocr 目前不支持这种“重算”语义，需要改 `ggml-alloc.c` 的 `n_children` 逻辑，或者手动在图里插入重算节点。

### 4.5 优化点四：数据集的 mmap

**现状**：190,560,000 B 全量读进内存。

**优化**：

```cpp
// 用 mmap 代替 ifstream + 全量拷贝
void * addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
// dataset->data 直接指向 mmap 区域（去掉 16 字节 header）
```

**收益**：省 190 MB 常驻内存，冷启动更快（按页加载）。**代价**：`ggml_opt_dataset` 需要支持外部 buffer。

---

## 5 数据搬运优化

### 5.1 现状：每 batch 100 次 H2D，每次同步

```text
31,360 B × 50 + 400 B × 50 = 1,588,000 B
实测：523 µs + 271 µs = 794 µs / batch
```

### 5.1.1 源码逐行：`ggml_opt_dataset_get_batch`（`src/ggml-opt.cpp:163-194`）

这是每 batch 100 次 H2D 的来源。

```cpp
163  void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
164      GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
165      GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
166      GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
167      GGML_ASSERT(                   data_batch->type == dataset->data->type);
168      GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);
169
170      const size_t nb_data_batch = ggml_nbytes(data_batch);
171      GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
172      const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;
173
174      if (labels_batch) {
175          const size_t nb_labels_batch = ggml_nbytes(labels_batch);
176          GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
177      }
178
179      GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));
180
181      for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
182          const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];
183
184          const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
185          ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);
186
187          if (!labels_batch) {
188              continue;
189          }
190
191          const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
192          ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
193      }
194  }
```

#### 5.1.1.1 第 170-172 行：算 shard 数

```cpp
const size_t nb_data_batch = ggml_nbytes(data_batch);          // 1,568,000
GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;   // 50
```

**`dataset->nbs_data`**（`ggml-opt.cpp:112`）：

```cpp
result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;
// = 188,160,000 * 10 / 60000 = 31,360
```

**`31,360 = 10 张图 × 784 像素 × 4 字节`**，正好是一个 shard 的大小。

**`1,568,000 / 31,360 = 50`**，所以每 batch 50 个 shard。

#### 5.1.1.2 第 182 行：permutation 间接寻址

```cpp
const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];
```

**`permutation` 是 shard 级别的索引数组**，长度 = `ndata / ndata_shard = 6000`。

**`ggml_opt_dataset_shuffle` 只打乱这个数组**，不动数据本身：

```cpp
void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}
```

**为什么按 shard 打乱而不是按样本**：如果按样本打乱，`get_batch` 就要逐张拷（500 次 H2D）；按 shard 打乱，每次拷 10 张连续的图（50 次 H2D）。这是“shuffle 粒度 vs 拷贝次数”的权衡。

**代价**：同一个 shard 内的 10 张图永远相邻，shuffle 的随机性降低。源码注释承认这一点：

```cpp
// With a shard size of 10 this overhead is greatly reduced at the cost of less shuffling
// (does not seem to have a significant impact).
```

#### 5.1.1.3 第 184-185 行：随机访问的 cache miss

```cpp
const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);
```

**`dataset->data->data` 是 188 MB 的 CPU 内存**（`malloc` 分配）。`ishard` 是随机索引，所以每次读的 31,360 B 都在不同位置。

**cache miss 的代价**：

```text
L3 cache 约 30~60 MB，188 MB 数据集放不下
每次读 31,360 B（约 490 个 cache line）
50 次 × 490 = 24,500 次 cache miss
每次 miss 约 50~100 ns（DDR4）→ 1.2~2.5 ms？
```

**但实测 `dataset_get_batch` 总共 1,138 µs，其中 memcpy 占 794 µs，剩下 344 µs**。所以 cache miss 的实际影响约 344 µs，比理论估算小——因为：

1. 连续 31,360 B 有硬件预取；
2. `cudaMemcpyAsync` 是 DMA，读内存时 CPU 可以继续；
3. 部分 shard 可能命中 L3（6000 个 shard 中，当前 batch 的 50 个有 0.8% 概率命中）。

#### 5.1.1.4 优化方向

**现状**：50 次 `tensor_set`，每次一次 `cudaMemcpyAsync + sync`。

**优化 A**：先在 host 上把 50 个 shard 拼成连续的 1,568,000 B，然后一次 `tensor_set`：

```cpp
// 新增一个 host 缓冲区
std::vector<char> staging(ggml_nbytes(data_batch) + ggml_nbytes(labels_batch));

for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
    const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];
    memcpy(staging.data() + ishard_batch*dataset->nbs_data,
           (const char *)dataset->data->data + ishard*dataset->nbs_data,
           dataset->nbs_data);
}
// 一次 H2D
ggml_backend_tensor_set(data_batch, staging.data(), 0, ggml_nbytes(data_batch));
```

**收益**：

```text
现状：50 × 10.47 µs + 50 × 5.41 µs = 794 µs
优化：1 × 135.62 µs（1,568,000 B）+ 50 次 memcpy（约 50 µs）= 186 µs
省：608 µs/batch
```

**代价**：多一次 1.57 MB 的 host memcpy（约 50 µs），需要额外 1.57 MB host 内存。

**优化 B**：只去掉 sync，保留 50 次 async：

```cpp
// 需要给 ggml_backend_tensor_set 加一个 async 版本
ggml_backend_tensor_set_async(data_batch, ptr_data, offset, size);
...
cudaStreamSynchronize(cudaStreamPerThread);   // 循环外一次
```

**收益**：

```text
现状：50 × 10.47 + 50 × 5.41 = 794 µs
优化：50 × 3.98 + 1 × 5 = 204 µs
省：590 µs/batch
```

**代价**：需要改 `ggml-backend.cpp` 加 async 接口，且要保证 `data` 指针在 sync 之前不被释放（本例中 `dataset->data` 是常驻的，安全）。

**优化 A 和 B 可以叠加**：

```text
A + B：1 × 135.62 µs + 1 × 5 µs ≈ 141 µs
省：653 µs/batch = 2.2 s（3,400 batch）
```

**这就是 06 §9 优化案例里“省 1.37 s”的来源**（保守估计）。

---

### 5.2 优化一：批量拷贝

```cpp
// 当前：50 次 tensor_set，每次一次 cudaMemcpyAsync + sync
for (int64_t ishard_batch = 0; ishard_batch < 50; ++ishard_batch) {
    ggml_backend_tensor_set(data_batch, ptr_data, offset, 31360);
}

// 优化：一次拷完（如果 shard 在内存里连续）
// 但 permutation 打乱了顺序，不连续 → 只能逐 shard
```

**真正的优化**：用 `ggml_backend_tensor_set_async` 或直接调 `cudaMemcpyAsync`，最后统一 sync：

```cpp
// 需要暴露一个 async 接口
ggml_backend_tensor_set_async(data_batch, ptr_data, offset, size);
...
cudaStreamSynchronize(stream);   // 一次
```

**收益**：从 50 次 sync（每次 ~5 µs）降到 1 次，省 ~250 µs。

### 5.3 优化二：pinned memory

**现状**：`dataset->data` 是 `malloc` 的普通内存，H2D 走“staging buffer”路径，带宽受限。

**优化**：

```cpp
// 用 cudaHostAlloc 代替 malloc
float * buf;
cudaHostAlloc(&buf, size, cudaHostAllocDefault);
```

**收益**：H2D 带宽从 ~3 GB/s 提升到 ~10 GB/s（PCIe 3.0 x16）。对 1.57 MB 的 batch，从 523 µs 降到 ~160 µs。

**代价**：pinned memory 不可换页，占用物理内存；需要改 `ggml_backend_cpu_buffer_type` 或给 dataset 加一个 pinned buffer type。

### 5.4 优化三：shard 调大

**现状**：shard=10，每 batch 50 个 shard。

**优化**：shard=50，每 batch 10 个 shard，每个 156,800 B。

```text
实测：156,800 B 的 H2D+sync ≈ 135.62 µs × (1568000/1568000) ≈ 136 µs（整个 batch 一次）
```

等等，实测 1,568,000 B 是 135.62 µs，那么 156,800 B 大约 20 µs。10 次 = 200 µs，比 50 次 × 10.47 µs = 523 µs 好。

**代价**：shuffle 粒度变粗（50 张图绑定在一起），可能影响收敛。源码注释也提到：

```cpp
// With a shard size of 10 this overhead is greatly reduced at the cost of less shuffling
// (does not seem to have a significant impact).
```

所以 shard=10 已经是权衡后的结果，再调大需要实验验证精度。

---

## 6 性能瓶颈清单（按优先级，代码插桩实测）

| # | 瓶颈 | 实测（每 batch） | 占比 | 优化手段 | 预期收益 | 改动文件 |
|---|---|---|---|---|---|---|
| 1 | 100 次 H2D + sync | 1,138 µs | **64%** | async + 单次 sync | 省 ~600 µs | `ggml-opt.cpp` |
| 2 | kernel 执行 + D2H | 490 µs | 27% | 融合/换 kernel | 有限 | `ggml-cuda/*` |
| 3 | 选图 + split + alloc | 158 µs | 9% | 缓存 reserve | 省 ~40 µs | `ggml-alloc.c` |
| 4 | 冷启动（cuBLAS + module） | 180 ms（一次性） | — | 预热 1 个 batch | 首次运行 3× | 调用方 |
| 5 | `REPEAT_BACK` 单 block | 未知（数据小） | — | warp 并行 | 大模型有效 | `binbcast.cu` |
| 6 | `ARGMAX` 线程浪费 | 未知 | — | 1 行/warp | 评估阶段 | `argmax.cu` |
| 7 | `SCALE nelements=1` | ~15 µs | 0.8% | 折进上游 | 极小 | `ggml-opt.cpp` |
| 8 | 3 次 D2H + sync | ~20 µs | 1% | 合并读取 | 极小 | `ggml-opt.cpp` |

**这张表推翻了本文早期版本**（基于 gdb 断点计时，误判“gallocr 是最大瓶颈”）。

**注意**：这是“mnist 规模”的结论。真实 LLM 的瓶颈排序不同：

```text
prefill:  GPU 计算占主导（GEMM），搬运占比 < 5%
decode:   搬运（KV cache 读写）+ kernel launch 开销占比上升
训练:     H2D（batch 数据）+ 反向计算并重
```

但**结构性的优化点（async 拷贝、预热、按形态选 kernel）是通用的**。

---

## 7 从 mnist 映射到真实 LLM

### 7.1 结构映射

| mnist | 真实 LLM（llama.cpp） | 同一套机制 |
|---|---|---|
| `fc1.weight [784,500]` | 每层 `attn_q.weight [4096,4096]` | leaf tensor，`alloc_ctx_tensors` |
| `images [784,500]` | `inp_tokens [n_embd, n_tokens]` | `ggml_set_input` + `tensor_set` |
| `node_2 = MUL_MAT(fc1.weight, images)` | `Qcur = MUL_MAT(wq, cur)` | `ggml_cuda_mul_mat` 决策树 |
| `node_5 = RELU(node_4)` | `SILU` / `GELU` | `GGML_OP_UNARY` |
| `loss_cross_entropy` | `cross_entropy_loss` | 两阶段归约 |
| `grad for fc1.weight` | 每层梯度 | `ggml_build_backward_expand` |
| `adamw step for fc1.weight` | 优化器 step | 原地视图 |
| `n_splits=1` | 整模型在 GPU | 同 |
| 多 split（本例没有） | 部分层 offload | `n_inputs>0` + copy 节点 |
| `n_copies=1` | 单请求 | 同 |
| `n_copies=4`（`parallel=true`） | 多请求并发 | event 流水线 |

### 7.2 关键差异

**① `n_splits`**

mnist 全在 GPU，`n_splits=1`。真实 LLM 如果开了 `--n-cpu-moe` 或显存不够，会有几十个 split，每个 split 之间要 `event_synchronize`。这时候 `compute_splits` 里的同步逻辑（03 §2.4）就成了热点。

**② `n_copies`**

mnist `parallel=false`，`n_copies=1`。llama.cpp 的 server 模式 `parallel=true`，`n_copies=4`，每个 input 有 4 份拷贝，用 event 做流水线。这时候 `ggml_backend_sched_split_graph` 里 `n_copies > 1` 的分支（03 §1.5）才会执行。

**③ kernel 路径**

mnist 的 `MUL_MAT` 是 `ne11=500`，走 cuBLAS。真实 LLM：

```text
prefill:  ne11 = seq_len（几百到几千）→ cuBLAS / MMF / MMQ
decode:   ne11 = 1                     → MMVF / MMVQ
```

**同一个 `ggml_cuda_mul_mat`，在 prefill 和 decode 走完全不同的路径**。这就是为什么 §4.2 的决策树要按 `ne11` 分档。

**④ 融合**

mnist 没有融合（`ne11=500` 不满足 `ne[1]==1`）。真实 LLM 的 decode 有 `MUL_MAT + bias + GLU` 融合（`ggml_cuda_try_fuse_mul_mat`），省 2~3 次 launch。

### 7.3 一个可以直接复用的调试流程

```text
① GGML_SCHED_DEBUG=2 看 split 和 use_counts
② 确认 n_splits / n_inputs / n_copies
③ 如果 n_splits > 1，看哪些节点被分到 CPU（GET_CAUSE 字段）
④ 断点 ggml_cuda_mul_mat，看 prefill/decode 各走哪条路径
⑤ 断点 gallocr，看每 batch 是否 re-reserve
⑥ 断点 tensor_set，看 H2D 次数和大小
```

这套流程在 llama.cpp 上同样适用，只是节点数从 28 变成几千。

---

## 8 性能分析工具链

### 8.1 ggml 自带

```bash
GGML_SCHED_DEBUG=2        # split + node 归属 + use_counts
GGML_SCHED_DEBUG_REALLOC=1 # 只打印意外的 realloc
GGML_CUDA_DISABLE_FUSION=1 # 关闭融合，对比收益
GGML_CUDA_GRAPH_OPT=1      # 打开 CUDA graph 的图优化（需 GGML_CUDA_GRAPHS=ON）
GGML_CUDA_PDL=0            # 关闭 PDL（Hopper+ 才有意义）
```

编译期：

```cpp
#define GGML_ALLOCATOR_DEBUG   // ggml-alloc.c:16，打开后打印每次 alloc/free
#define GGML_CUDA_DEBUG        // 打印 fusion / kernel 选择
```

### 8.2 nsys / ncu

```bash
# nsys：时间线，看 H2D / kernel / sync 的比例
nsys profile -o mnist --trace=cuda,nvtx ./build-debug/bin/mnist-train ...

# ncu：单 kernel 指标（需要 GPU 性能计数器权限）
ncu --set full --launch-count 20 ./build-debug/bin/mnist-eval ...
```

**本次调试遇到权限问题**：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

解决办法（需要 root）：

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiler.conf
update-initramfs -u
reboot
```

或者用 `ncu --metrics` 只采集不需要计数器的指标（如 `launch__grid_size`）。

### 8.3 自建微基准

本文的关键延迟数据（H2D、cublasCreate、空 kernel）都来自独立小程序，因为：

- 在 mnist 里打断点会引入 gdb 开销（每次断点 ~ms 级）；
- 微基准可以精确控制迭代次数、预热、事件计时。

模板：

```cpp
cudaEvent_t a, b;
cudaEventCreate(&a); cudaEventCreate(&b);
// 预热 20 次
for (int i = 0; i < 20; ++i) op();
cudaEventRecord(a);
for (int i = 0; i < 1000; ++i) op();
cudaEventRecord(b);
cudaEventSynchronize(b);
float ms; cudaEventElapsedTime(&ms, a, b);
printf("%.2f us/op\n", ms * 1000.0f / 1000);
```

---

## 9 一个完整的优化案例（可落地）

**目标**：把 mnist 训练从 6 s 降到 4 s 以内。

**基线**（代码插桩，05 §8.3）：

```text
总时间：6.0 s（3,420 个物理 batch）
  H2D：      1,138 µs × 3,420 = 3.89 s   (64%)
  compute：    490 µs × 3,420 = 1.68 s   (27%)
  调度：       158 µs × 3,420 = 0.54 s   (9%)
  合计：                         6.11 s
```

**步骤**：

```text
① 优化 A：H2D 去掉逐次同步（§5.2）
   - 现状：100 次 cudaMemcpyAsync + 100 次 cudaStreamSynchronize ≈ 794 µs
   - 改成：100 次 async + 1 次 sync ≈ 398 µs（微基准：async-only 3.98 µs）
   - 省 ~400 µs/batch = 1.37 s
   - 改动：给 ggml_backend_tensor_set 加 async 版本，或在 ggml_opt_dataset_get_batch 末尾统一 sync

② 优化 B：pinned memory（§5.3）
   - 现状：dataset->data 是 malloc 的普通内存，H2D 走 staging
   - 改成：cudaHostAlloc
   - 省 ~300 µs/batch = 1.03 s
   - 改动：给 dataset 加一个 pinned buffer type

③ 优化 C：预热（§6 #4）
   - 现状：首次 cuBLAS GEMM 156 ms
   - 改成：正式训练前跑 1 个 batch
   - 省 150 ms（一次性，对 30 epoch 影响小，对单次推理影响大）

④ 优化 D：gallocr 指纹缓存（§3.3）
   - 现状：alloc 42 µs/batch
   - 改成：命中缓存后 ~0
   - 省 ~40 µs/batch = 0.14 s（收益很小，优先级最低）
```

**预期结果**：

```text
优化前：6.0 s
优化后：6.0 - 1.37 - 1.03 = 3.6 s
```

**验证方式**：

```bash
# 每次优化后重跑插桩版本，看 [PROF2] 的 h2d_avg 是否下降
[PROF2] n=3000 alloc_avg=158 h2d_avg=1141 comp_avg=496 ...
```

**结论**：mnist 的优化空间在**数据搬运**（64%），不在 kernel（27%），更不在显存规划（9%）。这个排序和直觉相反——直觉会认为“GPU 计算是瓶颈”，但 mnist 的模型太小，计算量被搬运和 launch 开销淹没。

---

## 10 本篇结论

1. **加算子的最小路径是 `try_fuse`**，不是改 IR。改 IR 要同步改 `ggml.h` / `ggml.c` / 名字表 / 符号表 / 反向规则 / `supports_op` / `compute_forward` / kernel，7 处。
2. **mnist 的瓶颈是数据搬运（1,138 µs/batch，64%）**，不是 kernel（490 µs，27%），也不是调度（158 µs，9%）。
3. **re-reserve 的触发有两类**：3,420 次是 `backend_ids_changed`（`graph_copy[0]` 在 CPU/CUDA 之间跳），30 次是 `needs_realloc`（图大小变化）。3,450 次里只有 1 次真正 `cudaMalloc`。每次 reserve 的真实成本是 42 µs，不是 gdb 测出来的 4.5 ms。
4. **gdb 断点计时不可用于性能分析**：断点命中本身有 50~500 µs 开销，会把 42 µs 的代码放大成 4.5 ms。必须用代码插桩或独立微基准。
5. **H2D 的瓶颈是 sync 不是带宽**：31,360 B 带 sync 10.47 µs，不带 3.98 µs。async + 单次 sync 可省 60%。
6. **冷启动是真实成本**：cuBLAS 首次 GEMM 156 ms，是稳态的 8,000 倍。生产环境必须预热。
7. **映射到 LLM 时，变的是 `n_splits` / `n_copies` / kernel 路径，不变的是 split / gallocr / 生命周期分析**。这套调试流程可以直接复用。
