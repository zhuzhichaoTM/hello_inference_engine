# 02 · 建图：从 14 个前向节点到 32 个训练节点

> 对应源码：`examples/mnist/mnist-common.cpp:312-383`（`mnist_model_build`）、
> `src/ggml-opt.cpp:322-547`（`ggml_opt_build`）、`src/ggml.c:7345-7427`（`ggml_build_backward_expand`）、
> `src/ggml.c:6281-6305`（`ggml_opt_step_adamw`）
>
> 核心结论：**整个建图阶段 `ggml_cuda_compute_forward` 一次都没有命中**。三张图只是 IR，不含任何数值。

---

## 0 业务侧：模型结构

```cpp
void mnist_model_build(mnist_model & model) {
    ggml_set_param(model.fc1_weight);      // 打 PARAM 标记
    ggml_set_param(model.fc1_bias);
    ggml_set_param(model.fc2_weight);
    ggml_set_param(model.fc2_bias);

    ggml_tensor * fc1 = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
        ggml_mul_mat(model.ctx_compute, model.fc1_weight, model.images),   // [500,500]
        model.fc1_bias));                                                  // 广播加 [500,1]

    model.logits = ggml_add(model.ctx_compute,
        ggml_mul_mat(model.ctx_compute, model.fc2_weight, fc1),            // [10,500]
        model.fc2_bias);

    ggml_set_name(model.logits, "logits");
    ggml_set_output(model.logits);
}
```

数学上就是：

```text
h      = relu(W1ᵀ · X + b1)     X:[784,500]  W1:[784,500]  b1:[500,1]  → h:[500,500]
logits = W2ᵀ · h + b2           W2:[500,10]   h:[500,500]   b2:[10,1]   → logits:[10,500]
```

注意 `ggml_mul_mat(a, b)` 的语义是 `aᵀ·b`，所以 `W1` 的 `ne=[784,500]` 正好对应 PyTorch 的 `Linear(784,500).weight`（形状 `[500,784]`）。

**为什么这个函数叫 `build` 而不是 `forward`？** 因为它只返回表达式，不计算结果。`model.logits` 此刻是一个 `op=GGML_OP_ADD` 的**描述符**，`data == NULL`。

---

## 1 断点：建图阶段没有任何 CUDA 调用

### 1.1 脚本

```gdb
start mnist-fc ... CUDA0
break ggml_cuda_compute_forward
break ggml_cuda_mul_mat
break ggml_backend_cuda_graph_compute
break ggml_opt.cpp:581       # ggml_opt_init 返回前
run
```

### 1.2 输出

```text
OPT_BUILD_DONE gf_nodes=14      gf_leafs=2  gf_size=2048
OPT_BUILD_DONE gb_grad_nodes=28 gb_opt_nodes=32

ggml_opt_fit: epoch 0001/0030:
CUDA_FWD#1 op=MUL_MAT ...
```

`CUDA_FWD#1` 出现在 `ggml_opt_fit: epoch 0001/0030` **之后**，也就是第一次 `ggml_opt_eval` 里。建图阶段（`mnist_model_build` + `ggml_opt_init`）三个 CUDA 断点全部零命中。

**这证明了什么**：建图是纯 host 操作，只往 `ctx_compute` 的 arena 里追加 336 B 的张量描述符。CUDA 后端直到第一次 `sched_graph_compute` 才被唤醒。

---

## 2 前向图 `gf`：14 个节点

### 2.1 节点表（来自 `ggml_backend_cuda_graph_compute` 的 `cgraph->nodes`）

```text
[ 0] fc2.weight                     op=0   NONE        flags=4   (PARAM)   use=2
[ 1] fc1.weight                     op=0   NONE        flags=4   (PARAM)   use=1
[ 2] node_2                         op=29  MUL_MAT     flags=16  (COMPUTE) use=1
[ 3] fc1.bias                       op=0   NONE        flags=4   (PARAM)   use=1
[ 4] node_4                         op=2   ADD         flags=16            use=2
[ 5] node_5                         op=91  UNARY(RELU) flags=16            use=2
[ 6] node_6                         op=29  MUL_MAT     flags=16            use=1
[ 7] fc2.bias                       op=0   NONE        flags=4   (PARAM)   use=1
[ 8] logits                         op=2   ADD         flags=18  (OUTPUT|COMPUTE) use=3
[ 9] loss_cross_entropy             op=96  CROSS_ENTROPY_LOSS  flags=16    use=1
[10] loss_cross_entropy_scaled      op=32  SCALE       flags=26  (OUTPUT|LOSS|COMPUTE) use=0
[11] pred                           op=18  ARGMAX      flags=18            use=1
[12] node_12                        op=18  ARGMAX      flags=16            use=1
[13] ncorrect                       op=19  COUNT_EQUAL flags=18            use=0
```

（`flags` 是位掩码：`INPUT=1`、`OUTPUT=2`、`PARAM=4`、`LOSS=8`、`COMPUTE=16`）

### 2.2 为什么节点顺序是这样？

`ggml_build_forward_expand` 做的是**后序 DFS**：

```cpp
static size_t ggml_visit_parents_graph(ggml_cgraph * cgraph, ggml_tensor * node, bool compute) {
    if (node->op != GGML_OP_NONE && compute) {
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;      // 打计算标记
    }
    const size_t node_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
    if (ggml_bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
        return node_hash_pos;                          // 已访问，去重
    }
    cgraph->visited_hash_set.keys[node_hash_pos] = node;
    ggml_bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
    cgraph->use_counts[node_hash_pos] = 0;

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        struct ggml_tensor * src = node->src[i];
        if (src) {
            const size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);
            cgraph->use_counts[src_hash_pos]++;        // 父节点被消费一次
        }
    }

    if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
        cgraph->leafs[cgraph->n_leafs++] = node;       // 普通叶子
    } else {
        cgraph->nodes[cgraph->n_nodes++] = node;       // 计算节点
    }
    return node_hash_pos;
}
```

关键点：**`op == GGML_OP_NONE` 且不是 `PARAM` 的进 `leafs[]`，其余进 `nodes[]`。**

`fc1.weight` 是 `op=NONE` 但打了 `PARAM`，所以进 `nodes[]` 而不是 `leafs[]`。这就是为什么前向图 `gf` 的 `leafs=2`（只有 `images` 和 `labels`），但 `nodes[]` 里混着 4 个权重。

**为什么要区分？** 因为反向传播需要遍历参数节点来生成梯度。如果权重进 `leafs[]`，`ggml_build_backward_expand` 就看不到它们。

### 2.3 后序 DFS 的顺序与节点编号

以 `loss_cross_entropy_scaled` 为根，DFS 展开：

```text
loss_scaled
└── loss_cross_entropy
    ├── logits
    │   ├── node_6 (MUL_MAT)
    │   │   ├── fc2.weight        → nodes[0]
    │   │   └── node_5 (RELU)
    │   │       └── node_4 (ADD)
    │   │           ├── node_2 (MUL_MAT)
    │   │           │   ├── fc1.weight  → nodes[1]
    │   │           │   │   └── node_2   → nodes[2]
    │   │           │   └── images (leaf) → leafs[0]
    │   │           └── fc1.bias        → nodes[3]
    │   │           └── node_4          → nodes[4]
    │   │       └── node_5              → nodes[5]
    │   │   └── node_6                  → nodes[6]
    │   └── fc2.bias            → nodes[7]
    │   └── logits              → nodes[8]
    ├── labels (leaf)            → leafs[1]
    └── loss_cross_entropy      → nodes[9]
└── loss_cross_entropy_scaled → nodes[10]
```

但实测 `nodes[0] = fc2.weight`，`nodes[1] = fc1.weight`。为什么 `fc2.weight` 在 `fc1.weight` 前面？

因为 `ggml_build_forward_expand` 被调用了**三次**：

```cpp
ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);      // loss + 上游
ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);      // pred + 上游
ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);  // ncorrect + 上游
```

第一次展开 `loss_scaled` 时，`node_2` 的 `src[0]` 是 `fc1.weight`，但 `fc1.weight` 的 `src` 全空。DFS 先访问 `fc1.weight`（进 `nodes[0]`），再访问 `images`（leaf），再访问 `node_2`（`nodes[1]`）……

实测顺序 `fc2.weight, fc1.weight, node_2` 说明 **`fc2.weight` 在 `fc1.weight` 之前被访问**。原因在 `ggml_mul_mat` 的 `src` 顺序：

```cpp
result->src[0] = a;   // fc2.weight
result->src[1] = b;   // node_5
```

但 `node_5` 依赖 `fc1.weight`，所以如果先 DFS `fc2.weight` 再 DFS `node_5`，顺序就是 `fc2.weight, fc1.weight, node_2`。这是 `GGML_CGRAPH_EVAL_ORDER` 决定的：默认按 `src[0] → src[1]` 顺序（`LEFT_TO_RIGHT`），所以 `fc2.weight`（`node_6.src[0]`）先入栈。

**顺序重要吗？** 对正确性不重要（DFS 保证拓扑序），对内存复用**非常重要**：gallocr 按 `nodes[]` 顺序分配和释放，顺序不同，复用机会不同。

### 2.4 `use_counts` 的含义

`use_counts[]` 是 `ggml_visit_parents_graph` 里累加的“被多少个节点消费”：

```text
fc2.weight  use=2   ← node_6 (前向) + grad for node_5 (反向，用 fc2.weight 算 dW1)
fc1.weight  use=1   ← node_2
node_2      use=1   ← node_4
node_4      use=2   ← node_5 (前向) + node_22 (STEP，反向)
node_5      use=2   ← node_6 (前向) + node_18 (OUT_PROD，反向)
logits      use=3   ← loss_cross_entropy + pred + grad for node_6
loss_scaled use=0   ← 图输出
```

**注意 `use_counts` 是整张图（前向+反向）的计数**，不是前向单独的。`fc2.weight use=2` 就是因为反向里 `node_21 = OUT_PROD(fc2.weight, grad for node_6)` 又用了一次。

**同一个张量在不同图里 `use_counts` 不同**，断点实测：

| 张量 | `gb_grad` | `gb_opt` | 差值来源 |
|---|---|---|---|
| `fc2.weight` | 2 | **3** | +1：`adamw step for fc2.weight` 的 `src[0]` |
| `fc1.weight` | 1 | **2** | +1：AdamW |
| `fc1.bias` / `fc2.bias` | 1 | **2** | +1：AdamW |
| `grad for fc2.weight` | 0 | **1** | +1：AdamW 的 `src[1]` |

这解释了为什么 `gb_opt` 的 `graph_copy` 是 34 个节点而不是 32：多出的 2 个是 `adamw_params` 的依赖和拷贝节点，而 AdamW 本身又让 4 个参数各多一个消费者。

这个计数在 gallocr 的生命周期分析里**不直接使用**——gallocr 用的是自己维护的 `hash_node.n_children`（见 03 §3）。但 `use_counts` 对调试很有用：`use=0` 的节点就是图输出，不会被释放；`use` 突然变大往往意味着有新的消费者被加进来（比如 AdamW）。

### 2.5 `flags` 的语义

```cpp
enum ggml_tensor_flag {
    GGML_TENSOR_FLAG_INPUT   =  1, // ...is an input for the GGML compute graph
    GGML_TENSOR_FLAG_OUTPUT  =  2, // ...is an output for the GGML compute graph
    GGML_TENSOR_FLAG_PARAM   =  4, // ...contains trainable parameters
    GGML_TENSOR_FLAG_LOSS    =  8, // ...defines loss for numerical optimization
    GGML_TENSOR_FLAG_COMPUTE = 16, // ...must be computed
};
```

实测各节点的 `flags`：

| 节点 | flags | 含义 |
|---|---|---|
| `fc1.weight` | 4 | PARAM：参数，进 `nodes[]`，需要梯度 |
| `images` | 1 | INPUT：用户输入，`ggml_set_input` |
| `logits` | 18 = 2+16 | OUTPUT + COMPUTE |
| `loss_cross_entropy_scaled` | 26 = 2+8+16 | OUTPUT + LOSS + COMPUTE |
| `pred` | 18 | OUTPUT + COMPUTE |
| `ncorrect` | 18 | OUTPUT + COMPUTE |
| `node_2` / `node_4` / ... | 16 | COMPUTE：需要执行 |

**`COMPUTE` 标记怎么用的？** CUDA 后端执行时：

```cpp
if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
    continue;                          // 跳过不需要计算的节点
}
```

所以 4 个 `op=NONE` 的权重节点虽然有 `COMPUTE` 吗？没有——`ggml_visit_parents_graph` 只给 `op != GGML_OP_NONE` 的节点打 `COMPUTE`。权重节点 `flags=4`，只有 PARAM，所以 CUDA 执行时被跳过。这就是为什么 28 个节点只有 23 次 `ggml_cuda_compute_forward`。

---

### 2.6 源码逐行：`ggml_visit_parents_graph`（`src/ggml.c:7236-7290`）

这是“建图”的全部逻辑：一次后序 DFS，同时完成三件事——**打 COMPUTE 标记、累加 use_counts、把节点分到 `nodes[]` 或 `leafs[]`**。

```cpp
7236  static size_t ggml_visit_parents_graph(struct ggml_cgraph * cgraph, struct ggml_tensor * node, bool compute) {
7237      if (node->op != GGML_OP_NONE && compute) {
7238          node->flags |= GGML_TENSOR_FLAG_COMPUTE;
7239      }
7240
7241      const size_t node_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
7242      GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);
7243
7244      if (ggml_bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
7245          // already visited
7246
7247          if (compute) {
7248              // update the compute flag regardless
7249              for (int i = 0; i < GGML_MAX_SRC; ++i) {
7250                  struct ggml_tensor * src = node->src[i];
7251                  if (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
7252                      ggml_visit_parents_graph(cgraph, src, true);
7253                  }
7254              }
7255          }
7256
7257          return node_hash_pos;
7258      }
7259
7260      // This is the first time we see this node in the current graph.
7261      cgraph->visited_hash_set.keys[node_hash_pos] = node;
7262      ggml_bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
7263      cgraph->use_counts[node_hash_pos] = 0;
7264
7265      for (int i = 0; i < GGML_MAX_SRC; ++i) {
7266          const int k =
7267              (cgraph->order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
7268              (cgraph->order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
7269              /* unknown order, just fall back to using i */ i;
7270
7271          struct ggml_tensor * src = node->src[k];
7272          if (src == NULL) {
7273              continue;
7274          }
7275
7276          const size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);
7277          cgraph->use_counts[src_hash_pos]++;
7278      }
7279
7280      if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
7281          GGML_ASSERT(cgraph->n_leafs < cgraph->size);
7282          cgraph->leafs[cgraph->n_leafs++] = node;
7283      } else {
7284          GGML_ASSERT(cgraph->n_nodes < cgraph->size);
7285          cgraph->nodes[cgraph->n_nodes++] = node;
7286      }
7287
7288      return node_hash_pos;
7289  }
```

#### 2.6.1 第 7237-7239 行：COMPUTE 标记

```cpp
if (node->op != GGML_OP_NONE && compute) {
    node->flags |= GGML_TENSOR_FLAG_COMPUTE;
}
```

**`compute` 参数的作用**：`ggml_build_forward_expand` 传 `true`，`ggml_build_forward_order` 传 `false`。前者用于“要执行的图”，后者用于“只要拓扑序、不执行”的场景（比如 `ggml_graph_dump_dot` 的可视化图）。

**为什么 `op != NONE` 才打标记**：权重是 `op=NONE`，不应该被执行；`images` 也是 `op=NONE`（它是输入，不是计算节点）。所以 4 个权重 + `images` 都没有 `COMPUTE`，CUDA 执行时被跳过。

#### 2.6.2 第 7241-7258 行：哈希去重

```cpp
const size_t node_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
if (ggml_bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
    // already visited
    ...
    return node_hash_pos;
}
```

**`ggml_hash_find` 返回“该张量的哈希槽位”**，`visited_hash_set.used` 是一个 bitset，标记哪些槽位被占用。如果已访问：

1. **直接返回槽位**，不重复入 `nodes[]`；
2. 但 7247-7255 的循环会**递归给未打 COMPUTE 的 src 补标记**——这是为了处理“同一个张量被多条路径引用”的情况。

**这在 mnist 里的体现**：`logits` 被三个节点消费（`loss_cross_entropy`、`pred`、`grad for node_6`）。第一次 DFS 到 `logits` 时入 `nodes[]`，后面两次直接返回，但 `use_counts` 会累加到 3。

**`ggml_hash_find` 的实现**（`src/ggml-impl.h:266-284`）：

```cpp
266  static inline size_t ggml_hash(const struct ggml_tensor * p) {
267      // the last 4 bits are always zero due to alignment
268      return (size_t)(uintptr_t)p >> 4;
269  }
270
271  static size_t ggml_hash_find(const struct ggml_hash_set * hash_set, const struct ggml_tensor * key) {
272      size_t h = ggml_hash(key) % hash_set->size;
273
274      // linear probing
275      size_t i = h;
276      while (ggml_bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
277          i = (i + 1) % hash_set->size;
278          if (i == h) {
279              // visited all hash table entries -> not found
280              return GGML_HASHSET_FULL;
281          }
282      }
283      return i;
284  }
```

**三个关键设计**：

**① `ggml_hash` 是 `(uintptr_t)p >> 4`**（266-269）。因为 `ggml_tensor` 由 `ggml_new_object` 分配在 arena 里，16 B 对齐（低 4 位为 0）。右移 4 位把这 4 个零去掉，得到“对象槽号”。

**为什么必须右移**：如果直接用指针值，低 4 位全是 0，`h % size` 会在 `size` 为 2 的幂时集中到少数槽位（`0, 16, 32, ...` 的模）。右移后低 4 位是有信息的，配合素数 `size` 分布均匀。

**② `size` 是素数**（`src/ggml.c:6631`）：

```cpp
size_t ggml_hash_size(size_t min_sz) {
    static const size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101, ...
    };
    for (size_t i = 0; i < sizeof(primes)/sizeof(primes[0]); ++i) {
        if (primes[i] >= min_sz) return primes[i];
    }
    ...
}
```

**`ggml_graph_overhead()` 里的 4099 就来自这里**：`ggml_hash_size(2048*2) = 4099`（大于 4096 的下一个素数）。

**③ `GGML_HASHSET_FULL` 是“表满”的信号**（278-281）。如果探测一圈回到起点，说明表里没有空槽——这是致命的，调用方必须 `GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL)`。

**负载因子**：本例 `28 nodes + 7 leafs = 35 / 4099 ≈ 0.85%`，几乎不可能碰撞。真实 LLM 的图有几千节点，`size=8209`，负载因子 30~50%，平均探测 1~2 次。

**没有删除操作**：`ggml_hash_set_reset` 只是 `memset(used, 0)`，一次性清空。这避免了“删除后需要墓碑标记”的复杂度。

#### 2.6.3 第 7265-7278 行：`use_counts` 累加

```cpp
for (int i = 0; i < GGML_MAX_SRC; ++i) {
    const int k = (order == LEFT_TO_RIGHT) ? i : ...;
    struct ggml_tensor * src = node->src[k];
    if (src == NULL) continue;

    const size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);
    cgraph->use_counts[src_hash_pos]++;
}
```

**`use_counts` 在“访问完 src 之后”累加**——此时 src 的槽位一定已分配（递归返回保证）。所以：

```text
父节点访问 → 递归 src → src 入 nodes/leafs → 返回槽位 → use_counts[src]++
```

**`GGML_MAX_SRC = 10`**，但 mnist 的节点最多用 5 个（AdamW）。循环到 10 是固定开销，对 28 个节点 = 280 次迭代，可忽略。

#### 2.6.4 第 7280-7286 行：nodes 还是 leafs

```cpp
if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
    cgraph->leafs[cgraph->n_leafs++] = node;
} else {
    cgraph->nodes[cgraph->n_nodes++] = node;
}
```

**三态判定**：

| `op` | `PARAM` | 结果 | 例子 |
|---|---|---|---|
| `NONE` | 否 | `leafs[]` | `images`、`labels`、`leaf_2..leaf_6`（梯度累加器） |
| `NONE` | 是 | `nodes[]` | `fc1.weight`、`fc1.bias`、`fc2.weight`、`fc2.bias` |
| 非 `NONE` | 任意 | `nodes[]` | `node_2`、`node_4`、... |

**为什么 `PARAM` 要进 `nodes[]`**：反向传播的 `ggml_build_backward_expand` 遍历 `cgraph->nodes[]`，如果参数在 `leafs[]` 里，就看不到它们，也就不会为它们生成梯度。

**为什么梯度累加器进 `leafs[]`**：它们不是“计算”出来的，而是静态分配的外部数据（`ggml_new_tensor` 创建在 `opt ctx_static` 里）。反向图里它们作为 `ADD` 的 `src[0]` 出现，是叶子。

#### 2.6.5 后序 DFS 的顺序保证

**为什么后序能保证拓扑序**：DFS 在“所有 src 都访问完之后”才把自己加入 `nodes[]`，所以任何节点的所有依赖一定排在它前面。这是 gallocr 单遍生命周期分析的前提（见 03 §3.5）。

**但 DFS 不保证“同一层的顺序”**：`LEFT_TO_RIGHT` 只决定 `src[0]` 先于 `src[1]`，不决定兄弟节点之间的全局顺序。这就是为什么断点里 `nodes[0]=fc2.weight` 而 `nodes[1]=fc1.weight`——DFS 先深入 `node_6.src[0]=fc2.weight`，再深入 `node_6.src[1]=node_5`（它依赖 `fc1.weight`）。

**顺序对内存的影响**：gallocr 按 `nodes[]` 顺序分配/释放。如果 `fc1.weight` 排在 `fc2.weight` 前面，`node_2` 的 offset 会不同，但**峰值大小不变**（因为权重都在静态 buffer，不参与 gallocr）。

---

## 3 反向图 `gb_grad`：28 个节点

### 3.1 构建代码

```cpp
// ggml-opt.cpp:489
opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads=*/true);
ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());
```

`ggml_graph_dup` 复制整张前向图（nodes / leafs / hash_set / use_counts），`ggml_build_backward_expand` 在原图后面追加反向节点。

### 3.2 反向节点的来源

`ggml_build_backward_expand` 分两步：

```cpp
// ① 决定哪些节点需要梯度
bool * grads_needed = calloc(hash_set.size, sizeof(bool));
for (int i = 0; i < n_nodes_f; ++i) {
    bool node_needs_grad = (node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS);
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (node->src[j] && grads_needed[hash(src[j])]) { node_needs_grad = true; break; }
    }
    if (!node_needs_grad) continue;
    grads_needed[ihash] = true;
}

// ② 从后往前，对每个需要梯度的节点调用 ggml_compute_backward
for (int i = n_nodes_f - 1; i >= 0; --i) {
    ggml_compute_backward(ctx, cgraph, i, grads_needed);
}
```

`ggml_compute_backward` 是一个巨大的 `switch (tensor->op)`，按算子类型生成对应的反向节点。这就是“静态反向图”的本质：**没有自动微分，只有手写的算子反向规则**。

### 3.3 本例的反向节点

```text
[14] grad for loss_cross_entropy    op=32  SCALE
[15] grad for node_6                op=97  CROSS_ENTROPY_LOSS_BACK
[16] node_16                        op=21  REPEAT_BACK
[17] grad for fc2.bias              op=2   ADD
[18] node_18                        op=31  OUT_PROD
[19] grad for fc2.weight            op=2   ADD
[20] grad for node_6 (transposed)   op=39  TRANSPOSE
[21] grad for node_5                op=31  OUT_PROD
[22] node_22                        op=91  UNARY(STEP)
[23] grad for node_2                op=7   MUL
[24] node_24                        op=21  REPEAT_BACK
[25] grad for fc1.bias              op=2   ADD
[26] node_26                        op=31  OUT_PROD
[27] grad for fc1.weight            op=2   ADD
```

逐条对应数学：

| 节点 | 数学 | 解释 |
|---|---|---|
| `grad for loss_cross_entropy` | `dL/dloss = 0.5` | `loss_scaled = 0.5 × loss`（`opt_period=2`） |
| `grad for node_6` | `dL/dlogits` | 交叉熵反向：`(softmax(logits) - labels) × dL/dloss / nrows` |
| `node_16` | `dL/db2` 的中间量 | `REPEAT_BACK(grad for node_6)` 沿 batch 维求和：`[10,500] → [10,1]` |
| `grad for fc2.bias` | `dL/db2` | `ADD(leaf_3, node_16)`：`leaf_3` 是梯度累加器 |
| `node_18` | `dL/dW2` 的中间量 | `OUT_PROD(node_5, grad for node_6)`：`[500,500] × [10,500]ᵀ = [500,10]` |
| `grad for fc2.weight` | `dL/dW2` | `ADD(leaf_4, node_18)` |
| `grad for node_6 (transposed)` | `(dL/dlogits)ᵀ` | `TRANSPOSE`，只改 stride，不拷贝 |
| `grad for node_5` | `dL/dh` | `OUT_PROD(fc2.weight, dL/dlogitsᵀ)`：`[500,10] × [500,10] = [500,500]` |
| `node_22` | `relu'(node_4)` | `STEP(node_4)`：`x>0 → 1`，`x<=0 → 0` |
| `grad for node_2` | `dL/d(node_4)` | `MUL(node_22, grad for node_5)`：逐元素乘 mask |
| `node_24` | `dL/db1` 的中间量 | `REPEAT_BACK(grad for node_2)`：`[500,500] → [500,1]` |
| `grad for fc1.bias` | `dL/db1` | `ADD(leaf_5, node_24)` |
| `node_26` | `dL/dW1` 的中间量 | `OUT_PROD(images, grad for node_2)`：`[784,500] × [500,500]ᵀ = [784,500]` |
| `grad for fc1.weight` | `dL/dW1` | `ADD(leaf_6, node_26)` |

### 3.4 三个值得注意的设计

**① `relu` 的反向是 `STEP` + `MUL` 两个节点，不是一个融合算子。**

```cpp
case GGML_OP_UNARY: {
    if (uop == GGML_UNARY_OP_RELU) {
        // relu(x) = x * step(x)
        result = ggml_mul(ctx, tensor->src[0], ggml_step(ctx, tensor->src[0]));   // 概念上
    }
}
```

实测是两个独立节点：`node_22 = STEP(node_4)`、`grad for node_2 = MUL(node_22, grad for node_5)`。这是“图层面复用已有算子”的选择：不新增 `RELU_BACK`，而是用 `STEP`（阶跃）和 `MUL`（逐元素乘）组合。好处是 kernel 少，坏处是多一次 launch。真实 LLM 里 silu/gelu 的反向就有专门的 `SILU_BACK`，因为它们是热点。

**② 转置是 `GGML_OP_TRANSPOSE` 节点，但不是计算节点。**

```text
[20] grad for node_6 (transposed)  op=39  data=0x7f049d7ed400  nb=[40,4,20000,20000]  view_src=0x...
[15] grad for node_6               op=97  data=0x7f049d7ed400  nb=[4,40,20000,20000]
```

`data` 指针完全相同，`nb[0]` 和 `nb[1]` 交换（`4,40` → `40,4`），`view_offs=0`。CUDA 后端执行时：

```cpp
if (ggml_cuda_is_view_or_noop(node)) { continue; }   // TRANSPOSE 被跳过
```

所以 28 个节点只有 23 次 kernel 分发：4 个 `NONE` + 1 个 `TRANSPOSE` = 5 个跳过。

**③ 梯度累加器 `leaf_2..leaf_6` 是 `ggml_new_tensor` 创建的，不是视图。**

```cpp
// ggml-opt.cpp:466
opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
```

它们在 `opt ctx_static` 里，由 `ggml_backend_alloc_ctx_tensors` 一次性分配（见 01 §4.4），地址跨 batch 不变。反向图里 `grad for fc2.bias = ADD(leaf_3, node_16)` 的 `leaf_3` 就是它。

### 3.5 `ggml_graph_dup` 的成本

```cpp
struct ggml_cgraph * ggml_graph_dup(struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads) {
    struct ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, result);
    return result;
}
```

`ggml_graph_cpy` 复制 `nodes[]` / `leafs[]` / `hash_set` / `use_counts[]`，**不复制张量本身**。所以 `gb_grad` 的 28 个节点里有 14 个和 `gf` 是**同一批张量指针**。

这就是为什么 `gb_grad` 的 `leafs=7` 而不是 `14`：前向的 2 个叶子（images/labels）+ 5 个梯度累加器 = 7。

---

### 3.6 源码逐行：`ggml_build_backward_expand`（`src/ggml.c:7345-7440`）

这是“手写反向图”的总入口。它做两件事：**决定哪些张量需要梯度**、**从后往前调用 `ggml_compute_backward`**。

```cpp
7345  void ggml_build_backward_expand(
7346          struct ggml_context *  ctx,
7347          struct ggml_cgraph  *  cgraph,
7348          struct ggml_tensor  ** grad_accs) {
7349      GGML_ASSERT(cgraph->n_nodes > 0);
7350      GGML_ASSERT(cgraph->grads);
7351      GGML_ASSERT(cgraph->grad_accs);
7352
7353      const int n_nodes_f = cgraph->n_nodes;
7354
7355      memset(cgraph->grads,     0, cgraph->visited_hash_set.size*sizeof(struct ggml_tensor *));
7356      memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*sizeof(struct ggml_tensor *));
7357      bool * grads_needed = calloc(cgraph->visited_hash_set.size, sizeof(bool));
7358
7359      {
7360          bool any_params = false;
7361          bool any_loss   = false;
7362          for (int i = 0; i < n_nodes_f; ++i) {
7363              struct ggml_tensor * node = cgraph->nodes[i];
7364              any_params = any_params || (node->flags & GGML_TENSOR_FLAG_PARAM);
7365              any_loss   = any_loss   || (node->flags & GGML_TENSOR_FLAG_LOSS);
7366          }
7367          GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
7368          GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
7369      }
7370
7371      for (int i = 0; i < n_nodes_f; ++i) {
7372          struct ggml_tensor * node = cgraph->nodes[i];
7373
7374          if (node->type == GGML_TYPE_I32) {
7375              continue;
7376          }
7377
7378          bool node_needs_grad = (node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS);
7379          bool ignore_src[GGML_MAX_SRC] = {false};
7380          switch (node->op) {
7381              case GGML_OP_IM2COL:
7382              case GGML_OP_IM2COL_BACK:
7383                  ignore_src[0] = true;
7384                  break;
7385              case GGML_OP_UNARY: {
7386                  const enum ggml_unary_op uop = ggml_get_unary_op(node);
7387                  if (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
7388                      ignore_src[0] = true;
7389                  }
7390              } break;
7391              case GGML_OP_CPY:
7392              case GGML_OP_GET_ROWS:
7393              case GGML_OP_GET_ROWS_BACK:
7394              case GGML_OP_ROPE:
7395                  ignore_src[1] = true;
7396                  break;
7397              default:
7398                  break;
7399          }
7400          for (int j = 0; j < GGML_MAX_SRC; ++j) {
7401              if (!node->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, node->src[j])]) {
7402                  continue;
7403              }
7404              GGML_ASSERT(node->src[j]->type == GGML_TYPE_F32 || node->src[j]->type == GGML_TYPE_F16);
7405              node_needs_grad = true;
7406              break;
7407          }
7408          if (!node_needs_grad) {
7409              continue;
7410          }
7411
7412          GGML_ASSERT(!node->view_src || node->op == GGML_OP_CPY || ...);
7413
7414          const size_t ihash = ggml_hash_find(&cgraph->visited_hash_set, node);
7415          GGML_ASSERT(ihash != GGML_HASHSET_FULL);
7416          GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
7417          if (grad_accs && grad_accs[i]) {
7418              cgraph->grad_accs[ihash] = grad_accs[i];
7419              cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
7420          } else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
7421              cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
7422              cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
7423          }
7424          grads_needed[ihash] = true;
7425      }
7426
7427      for (int i = n_nodes_f - 1; i >= 0; --i) {
7428          ggml_compute_backward(ctx, cgraph, i, grads_needed);
7429      }
7430
7431      free(grads_needed);
7432  }
```

#### 3.6.1 第 7355-7357 行：清空梯度表

```cpp
memset(cgraph->grads,     0, hash_size * sizeof(ggml_tensor *));
memset(cgraph->grad_accs, 0, hash_size * sizeof(ggml_tensor *));
bool * grads_needed = calloc(hash_size, sizeof(bool));
```

**`grads` / `grad_accs` 是按哈希槽位索引的数组**，不是按节点序号。这样任意张量（包括不在 `nodes[]` 里的）都能用 `ggml_hash_find` 定位。

`ggml_graph_dup` 已经把这两个数组 `memset` 过一次（`ggml_graph_cpy` 的 7569-7572 行），这里再清一次是为了幂等——如果同一个 `gb_grad` 被重新 build（`ggml_opt_alloc` 每 batch 调用），不会残留旧指针。

#### 3.6.2 第 7378 行：谁需要梯度

```cpp
bool node_needs_grad = (node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS);
```

**初始条件**：参数和 loss 一定需要梯度。

然后 7400-7407 的循环**向前传播**：

```cpp
for (int j = 0; j < GGML_MAX_SRC; ++j) {
    if (!node->src[j] || ignore_src[j] || !grads_needed[hash(src[j])]) continue;
    node_needs_grad = true;
    break;
}
```

**注意这里的遍历顺序**：`i` 从 0 到 `n_nodes_f`（前向顺序），所以 `grads_needed` 是**从参数向前传播**的——只有当某个节点的 src 已经被标记为需要梯度时，它自己才需要梯度。

**这依赖 `nodes[]` 的拓扑序**：参数在 `nodes[]` 里的位置不确定（`fc2.weight` 在 `nodes[0]`，`fc1.weight` 在 `nodes[1]`），但它们的消费者一定在它们之后。所以第一次遍历时：

```text
i=0: fc2.weight → PARAM → needs_grad=true
i=1: fc1.weight → PARAM → needs_grad=true
i=2: node_2 = MUL_MAT(fc1.weight, images)
       src[0]=fc1.weight → grads_needed=true → node_2 needs_grad=true
i=3: fc1.bias → PARAM → true
i=4: node_4 = ADD(node_2, fc1.bias)
       src[0]=node_2 → true → node_4 needs_grad=true
...
```

**`ignore_src` 的作用**：某些算子的某些输入不可微：

- `UNARY(SGN)` / `UNARY(STEP)`：分段常数，导数恒 0；
- `CPY` 的 `src[1]`：目标张量，不参与梯度；
- `ROPE` 的 `src[1]`：位置索引，不是浮点。

**mnist 里命中 `ignore_src[0]` 的**：`node_22 = STEP(node_4)`。`node_4` 是它的 `src[0]`，但因为 STEP 不可微，`grads_needed[node_4]` 不会被它影响——`node_4` 的梯度来自 `node_5`（RELU）的反向。

#### 3.6.3 第 7417-7424 行：挂接梯度累加器

```cpp
if (grad_accs && grad_accs[i]) {
    cgraph->grad_accs[ihash] = grad_accs[i];
    cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
} else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
    cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
    cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
}
grads_needed[ihash] = true;
```

**`grad_accs` 来自 `ggml_opt_build` 的 462-470 行**：

```cpp
opt_ctx->grad_accs.resize(n_nodes);
for (int i = 0; i < n_nodes; ++i) {
    ggml_tensor * node = opt_ctx->gf->nodes[i];
    if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
        opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
    } else {
        opt_ctx->grad_accs[i] = nullptr;
    }
}
```

**只有参数和 loss 有累加器**（`accumulate=true` 时）。中间节点（`node_2`、`node_5` 等）的梯度是临时的，由 gallocr 分配，不需要跨 batch 保持。

**`grads[ihash] = grad_accs[ihash]` 的含义**：初始时梯度指针指向累加器，反向过程中 `ggml_add_or_set` 会在这个累加器上原地累加（见 3.6.4）。

#### 3.6.4 第 7427-7429 行：从后往前展开

```cpp
for (int i = n_nodes_f - 1; i >= 0; --i) {
    ggml_compute_backward(ctx, cgraph, i, grads_needed);
}
```

**为什么从后往前**：反向传播的自然顺序——先算 loss 的梯度，再算 `logits` 的梯度，再算 `node_6` 的梯度……每个节点的梯度依赖它所有消费者的梯度，所以必须后序。

**`ggml_compute_backward` 的骨架**（`src/ggml.c:6760+`）：

```cpp
static void ggml_compute_backward(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int i, const bool * grads_needed) {
    struct ggml_tensor * tensor = cgraph->nodes[i];
    struct ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    if (!grad) {
        return;                                    // 这个节点不需要梯度
    }

    const size_t isrc0 = ggml_hash_find(&cgraph->visited_hash_set, tensor->src[0]);
    const size_t isrc1 = ggml_hash_find(&cgraph->visited_hash_set, tensor->src[1]);

    const bool src0_needs_grads = isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(...used, isrc0) && grads_needed[isrc0];
    const bool src1_needs_grads = ...;

    switch (tensor->op) {
        case GGML_OP_ADD: { ... } break;
        case GGML_OP_MUL_MAT: { ... } break;
        case GGML_OP_UNARY: { ... } break;
        case GGML_OP_CROSS_ENTROPY_LOSS: { ... } break;
        ...
    }
}
```

#### 3.6.5 `ggml_add_or_set`：梯度累加的唯一入口（`src/ggml.c:6682-6696`）

```cpp
6682  static void ggml_add_or_set(
6683          struct ggml_context * ctx,
6684          struct ggml_cgraph  * cgraph,
6685          size_t                isrc,
6686          struct ggml_tensor  * tensor) {
6687      struct ggml_tensor * src = cgraph->visited_hash_set.keys[isrc];
6688      GGML_ASSERT(src);
6689      if (cgraph->grads[isrc]) {
6690          cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
6691      } else {
6692          cgraph->grads[isrc] = tensor;
6693      }
6694      ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
6695      ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
6696  }
```

**三个关键行为**：

1. **第一次**：`grads[isrc]` 为空 → 直接赋值 `grads[isrc] = tensor`。这就是 `grad for loss_cross_entropy`（`tensor` 是 `SCALE` 的结果）的来源。
2. **后续**：`grads[isrc]` 已有值 → `ggml_add_impl(grads, tensor, inplace=grad_accs[isrc])`。`inplace=true` 时复用累加器的内存，避免每 batch 重新分配。
3. **每次都会 `ggml_build_forward_expand`**：把新的梯度节点追加到图里，所以 `gb_grad` 的节点数是动态增长的。

**这解释了断点里 `grad for fc2.weight = ADD(leaf_4, node_18)`**：

- `leaf_4` 是 `grad_accs[fc2.weight]`（静态张量）；
- `node_18` 是本次反向算出的 `OUT_PROD` 结果；
- `ADD(leaf_4, node_18, inplace=true)` 把结果写回 `leaf_4`。

**`inplace` 参数的来源**：

```cpp
ggml_add_impl(ctx, a, b, inplace) → 设置 result->src[0..1] 和 op=ADD
                                   如果 inplace，result 是 ggml_view_tensor(ctx, a)
```

所以 `grad for fc2.weight` 的 `data` 与 `leaf_4` 相同，`view_src` 指向 `leaf_4`——这就是断点里 `[19] grad for fc2.weight data=0x7f049d006000 view_src=0x55db39993440` 的来源。

#### 3.6.6 mnist 三个算子的反向规则

**① ADD（`src/ggml.c:6777-6788`）**

```cpp
case GGML_OP_ADD: {
    if (src0_needs_grads) {
        ggml_add_or_set(ctx, cgraph, isrc0, grad);
    }
    if (src1_needs_grads) {
        struct ggml_tensor * tmp = grad;
        if (!ggml_are_same_shape(src0, src1)) {
            tmp = ggml_repeat_back(ctx, tmp, src1);     // ← 广播的反向
        }
        ggml_add_or_set(ctx, cgraph, isrc1, tmp);
    }
} break;
```

**`node_4 = ADD(node_2, fc1.bias)`**：

- `src0 = node_2`（`[500,500]`），`src1 = fc1.bias`（`[500,1]`）；
- `src0` 的梯度 = `grad` 本身（形状相同）；
- `src1` 的梯度 = `repeat_back(grad, fc1.bias)`——沿 `ne1` 求和，把 `[500,500]` 压成 `[500,1]`。

**这就是断点里 `node_24 = REPEAT_BACK(grad for node_2)` 的来源**：`node_24` 是 `fc1.bias` 的梯度中间量，`[500,500] → [500,1]`。

**② MUL_MAT（`src/ggml.c:6899-6945`）**

```cpp
case GGML_OP_MUL_MAT: {
    // tensor.shape [m,p,qq,rr]
    // src0.shape   [n,m,q1,r1]
    // src1.shape   [n,p,qq,rr]

    if (src0_needs_grads) {
        struct ggml_tensor * tmp = ggml_out_prod(ctx, src1, grad);   // [n,m,qq,rr]
        if (!ggml_are_same_shape(tmp, src0)) { ... repeat_back ... }
        ggml_add_or_set(ctx, cgraph, isrc0, tmp);
    }
    if (src1_needs_grads) {
        ggml_add_or_set(ctx, cgraph, isrc1,
            ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, src0)), grad));
    }
} break;
```

**`node_6 = MUL_MAT(fc2.weight, node_5)`**：

- `src0 = fc2.weight`（`[500,10]`），`src1 = node_5`（`[500,500]`），`dst = node_6`（`[10,500]`）；
- `src0` 的梯度 = `OUT_PROD(node_5, grad)`：`[500,500] × [10,500]ᵀ = [500,10]` → 断点里的 `node_18`；
- `src1` 的梯度 = `MUL_MAT(transpose(fc2.weight), grad)`：`[10,500]ᵀ × [10,500] = [500,500]` → 断点里的 `grad for node_5`。

**注意 `ggml_cont(ggml_transpose(src0))`**：`src0` 是权重，转置后不连续，`ggml_cont` 强制物化。但断点里没看到 `CONT` 节点——因为 `ggml_mul_mat` 内部能直接处理转置的 `src0`（`nb[0] > nb[1]`），`ggml_cont` 被优化掉了。

**`node_18 = OUT_PROD(node_5, grad for node_6)`** 的 `src1` 是 `grad for node_6`，它的 `ne=[10,500]`，`OUT_PROD` 计算 `[500,500] × [10,500]ᵀ = [500,10]`。

**`grad for node_5 = OUT_PROD(fc2.weight, grad for node_6 (transposed))`**：

- `fc2.weight` 是 `[500,10]`；
- `grad for node_6 (transposed)` 是 `[500,10]`（`ne` 交换，`nb` 交换）；
- `OUT_PROD` 计算 `[500,10] × [500,10]ᵀ = [500,500]`。

**这里用了 `TRANSPOSE` 视图而不是 `CONT`**：因为 `OUT_PROD` 的 cuBLAS 路径能直接处理转置的 `src1`（`src1_T=1`，见 04 §4.5）。

**③ UNARY(RELU)（`src/ggml.c:7168-7172`）**

```cpp
case GGML_UNARY_OP_RELU: {
    if (src0_needs_grads) {
        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
    }
} break;
```

**`node_5 = RELU(node_4)`**：

- `src0 = node_4`；
- 梯度 = `STEP(node_4) × grad`；
- 展开成两个节点：`node_22 = STEP(node_4)`、`grad for node_2 = MUL(node_22, grad for node_5)`。

**为什么不用 `RELU_BACK`**：ggml 的算子表里没有它（见 06 §1），`STEP + MUL` 是两个通用算子的组合。对 mnist 多一次 launch（~8 µs），对真实 LLM 的 silu/gelu 有专门的 `SILU_BACK`，因为它们是热点。

**④ CROSS_ENTROPY_LOSS（`src/ggml.c:7200-7205`）**

```cpp
case GGML_OP_CROSS_ENTROPY_LOSS: {
    if (src0_needs_grads) {
        ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
    }
    GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
} break;
```

**`loss_cross_entropy = CROSS_ENTROPY_LOSS(logits, labels)`**：

- `src0 = logits`，`src1 = labels`（one-hot，不可微）；
- 梯度 = `CROSS_ENTROPY_LOSS_BACK(grad, logits, labels)` → 断点里的 `grad for node_6`。

**`grad` 是标量**（`loss_scaled` 的梯度，值 1.0），kernel 里：

```cpp
const float d_by_nrows = *grad/gridDim.x;
dst[i] = (val*sm_scale - labels[i])*d_by_nrows;
```

---

## 4 优化图 `gb_opt`：32 个节点

### 4.1 构建代码

```cpp
// ggml-opt.cpp:504
opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads=*/true);

// 为每个参数追加一个 AdamW 节点
for (int i = 0; i < opt_ctx->gf->n_nodes-1; --i) {   // 实际是从后往前
    struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
    struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);
    if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
        opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
        ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
    }
}
```

### 4.2 追加的 4 个节点

```text
[28] adamw step for fc2.bias        op=98  OPT_STEP_ADAMW   ne=[10,1]
[29] adamw step for fc1.bias        op=98  OPT_STEP_ADAMW   ne=[500,1]
[30] adamw step for fc1.weight      op=98  OPT_STEP_ADAMW   ne=[784,500]
[31] adamw step for fc2.weight      op=98  OPT_STEP_ADAMW   ne=[500,10]
```

**为什么顺序是 bias 在前、weight 在后？** 因为遍历是 `for (i = n_nodes-1; i >= 0; --i)`，从后往前找参数节点。前向图里 `fc2.bias`（`nodes[7]`）在 `fc1.weight`（`nodes[1]`）之后，所以反向遍历时先遇到 `fc2.bias`。

### 4.3 `ggml_opt_step_adamw` 是原地算子

```cpp
struct ggml_tensor * ggml_opt_step_adamw(
        struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * grad,
        struct ggml_tensor * m, struct ggml_tensor * v, struct ggml_tensor * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);   // ← 视图，不是新张量

    result->op     = GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->src[4] = adamw_params;
    return result;
}
```

断点证据：

```text
[30] adamw step for fc1.weight  data=0x7f049cc00000  view_src=0x...   ← 与 fc1.weight 同地址
[ 1] fc1.weight                 data=0x7f049cc00000  view_src=nil
```

**为什么要用视图？** 因为 AdamW 的语义是“就地更新参数”，`result` 必须和 `a` 共享内存。如果 `result` 是新张量，就得再拷一次回去，多一次 kernel 和一块显存。

这也解释了为什么 gallocr 的原地复用列表 `ggml_op_can_inplace` 里**没有** `OPT_STEP_ADAMW`：它已经是视图了，不需要复用。

### 4.4 AdamW 的 7 个参数

```cpp
opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
// 7 个值：alpha, beta1, beta2, eps, wd, beta1h, beta2h
```

`beta1h` / `beta2h` 是 warmup 修正：

```cpp
const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));
```

**为什么在 host 上算？** 因为 `powf` 是标量运算，7 个数，GPU 上算反而更慢（launch 开销 > 计算）。算完写进 CPU 张量，由 scheduler 拷到 GPU（见 03 §1.5）。

### 4.5 `gb_opt` 的 16 个叶子

```text
前向叶子：images, labels                                          = 2
梯度累加器：leaf_2..leaf_6                                        = 5
AdamW 状态：m/v for fc1.weight, fc1.bias, fc2.weight, fc2.bias    = 8
优化器参数：adamw_params                                          = 1
                                                                 = 16
```

对比 `gb_grad` 的 7 个叶子，多了 9 个。这 9 个都在 `opt ctx_static` 和 `opt ctx_cpu` 里，地址固定。

---

### 4.6 源码逐行：`ggml_opt_build` 的三段式（`src/ggml-opt.cpp:322-547`）

`ggml_opt_build` 是训练图的“总装车间”，它在一个函数里完成三件事：建 `gf`、建 `gb_grad`、建 `gb_opt`。

#### 4.6.1 元数据预算（328-382 行）

```cpp
328      const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
329          !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);
330
331      const bool need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
332          opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
```

**`accumulate`**：是否需要梯度累加器。本例 `opt_period=2`，`accumulate=true`。
**`need_momenta`**：AdamW 需要 `m` 和 `v`，所以 `need_momenta=true`。

```cpp
346      if (!opt_ctx->ctx_static) {
354          constexpr size_t n_loss = 1;
355          const size_t tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
356          const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
357          const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
```

**验算 `size_meta = 8,096`**：

```text
n_loss              = 1          （loss 的梯度累加器）
tensors_per_param   = 1 + 2 = 3  （梯度累加器 1 + AdamW m/v 2）
n_param             = 4          （fc1.weight, fc1.bias, fc2.weight, fc2.bias）
tensors_const       = 9          （static_graphs=true：labels, loss, loss_scaled, pred, node_12, ncorrect, ...）

size_meta = (1 + 3*4 + 9) * 368 = 22 * 368 = 8,096  ✅
```

**这 22 个槽位对应断点里 `opt ctx_static` 的 19 个张量**（01 §4.4 枚举了 19 个），差额 3 个是 `ggml_new_tensor` 可能创建的临时张量（比如 `ggml_dup_tensor` 的 labels 视图）。

#### 4.6.2 loss 链（386-446 行）

```cpp
402          case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
403              opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
404              ggml_set_input(opt_ctx->labels);
405              ggml_set_name(opt_ctx->labels, "labels");
406              opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
407              ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
408              if (opt_ctx->opt_period > 1) {
409                  opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
410                  ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
411              }
412              opt_ctx->loss_per_datapoint = true;
413              break;
414          }
```

**逐行**：

| 行 | 行为 | 为什么 |
|---|---|---|
| 403 | `ggml_dup_tensor(outputs)` | 创建一个和 `logits` 同形状的 `labels` 张量。**不是视图**，因为 labels 由用户 `tensor_set` 填入 |
| 404 | `ggml_set_input(labels)` | 打 `INPUT` 标记，scheduler 知道它是外部数据 |
| 406 | `ggml_cross_entropy_loss(logits, labels)` | 创建 `op=96` 的节点，`ne=[1,1]` |
| 408-411 | `opt_period > 1` 时再 `scale` | 本例 `opt_period=2`，所以有 `loss_cross_entropy_scaled` 节点 |
| 412 | `loss_per_datapoint = true` | 告诉 `ggml_opt_result_loss` 这个 loss 是“每样本平均”，不是总和 |

**`ggml_scale` 的实现**（`src/ggml.c`）：

```cpp
struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s) {
    struct ggml_tensor * result = ggml_scale_impl(ctx, a, s, 0.0f);
    return result;
}

static struct ggml_tensor * ggml_scale_impl(...) {
    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);   // 同形状
    ggml_set_op_params_f32(result, 0, s);                    // scale
    ggml_set_op_params_f32(result, 1, bias);
    result->op     = GGML_OP_SCALE;
    result->src[0] = a;
    return result;
}
```

**`scale` 和 `bias` 存在 `op_params` 里**，不是张量。所以 `SCALE` 节点只有一个 `src`，kernel 从 `op_params` 读系数（04 §5.7）。

#### 4.6.3 pred / ncorrect（436-446 行）

```cpp
436      if (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
437          opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
438          ggml_set_name(opt_ctx->pred, "pred");
439          ggml_set_output(opt_ctx->pred);
440          ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);
441
442          opt_ctx->ncorrect = ggml_count_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
443          ggml_set_name(opt_ctx->ncorrect, "ncorrect");
444          ggml_set_output(opt_ctx->ncorrect);
445          ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
446      }
```

**注意 442 行**：`ggml_argmax(labels)` 是**内联创建**的临时张量，没有名字，断点里显示为 `node_12`。

**`ggml_build_forward_expand` 被调用三次**：

```text
434: gf += loss 及其上游
440: gf += pred 及其上游（已访问的去重）
445: gf += ncorrect 及其上游（已访问的去重）
```

**这就是为什么 `gf` 的 `n_nodes=14` 而不是 10**：`pred`、`node_12`、`ncorrect` 是独立的分支，它们的上游（`logits`、`labels`）已在前两次展开时访问过，所以只新增这 3 个节点。

#### 4.6.4 三张图的构建顺序（488-546 行）

```cpp
488      // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
489      opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
490      ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());
491
...
503      // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
504      opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);
505
506      opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
507      ggml_tensor * adamw_params = opt_ctx->opt_step_params;
508      ggml_set_input(adamw_params);
509      const char * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
510      ggml_format_name(adamw_params, "%s_params", optimizer_name);
511      for (int i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
512          struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
513          struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);
514
515          if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
516              struct ggml_tensor * m = nullptr;
517              struct ggml_tensor * v = nullptr;
518              if (need_momenta) {
519                  m = opt_ctx->grad_m[i];
520                  v = opt_ctx->grad_v[i];
521                  ggml_format_name(m, "AdamW m for %s", node->name);
522                  ggml_format_name(v, "AdamW v for %s", node->name);
523              }
524              struct ggml_tensor * opt_step;
525              switch (optimizer) {
526                  case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
527                      opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
528                      break;
...
535              ggml_format_name(opt_step, "%s step for %s", optimizer_name, node->name);
536              ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
537          }
538      }
```

**逐行**：

| 行 | 行为 | 关键点 |
|---|---|---|
| 489 | `ggml_graph_dup(gf, force_grads=true)` | 复制 `gf` 的节点表，但分配 `grads/grad_accs` 数组（`gf` 本身可能没有） |
| 490 | `ggml_build_backward_expand` | 在原 14 个节点后追加 14 个反向节点，共 28 |
| 504 | `ggml_graph_dup(gb_grad)` | 再复制一次，得到 28 个节点 |
| 506 | `ggml_new_tensor_1d(ctx_cpu, F32, 7)` | AdamW 参数放 **CPU context**，7 个 float |
| 508 | `ggml_set_input(adamw_params)` | 打 INPUT 标记，scheduler 会为它创建跨后端拷贝（03 §1.5） |
| 511 | `for (i = n_nodes-1; i >= 0; --i)` | **从后往前**遍历 `gf` 的节点 |
| 512 | `node = gb_opt->nodes[i]` | 取 `gb_opt` 的对应节点（前 14 个与 `gf` 相同） |
| 513 | `ggml_graph_get_grad(gb_opt, node)` | 查这个节点的梯度 |
| 515 | `if (grad && PARAM)` | 只对参数加 AdamW |
| 527 | `ggml_opt_step_adamw(...)` | 创建原地视图节点 |
| 536 | `ggml_build_forward_expand(gb_opt, opt_step)` | 追加到 `gb_opt`，节点数 28 → 32 |

**为什么 511 行从后往前**：为了让 AdamW 节点的顺序是 `fc2.bias → fc1.bias → fc1.weight → fc2.weight`（断点里的 `[28]..[31]`）。如果从前往后，顺序会是 `fc1.weight → fc1.bias → fc2.weight → fc2.bias`。

**顺序重要吗**：对正确性不重要（AdamW 节点之间没有依赖），对调试可读性有影响。从后往前是“反向传播的自然顺序”——先更新最后算梯度的参数。

#### 4.6.5 `ggml_graph_dup` 的成本（`src/ggml.c:7591-7595`）

```cpp
7591  struct ggml_cgraph * ggml_graph_dup(struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads) {
7592      struct ggml_cgraph * result = ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
7593      ggml_graph_cpy(cgraph, result);
7594      return result;
7595  }
```

**`ggml_graph_cpy`（7544-7590）逐段**：

```cpp
7549      dst->n_leafs = src->n_leafs;
7550      dst->n_nodes = src->n_nodes;
7551      dst->order   = src->order;
7553      for (int i = 0; i < src->n_leafs; ++i) {
7554          dst->leafs[i] = src->leafs[i];          // ← 只复制指针
7556      for (int i = 0; i < src->n_nodes; ++i) {
7557          dst->nodes[i] = src->nodes[i];          // ← 只复制指针
7561      for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
7563          if (ggml_bitset_get(src->visited_hash_set.used, i)) {
7564              size_t new_hash_pos = ggml_hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
7565              dst->use_counts[new_hash_pos] = src->use_counts[i];   // ← 重建哈希表
7576          for (int i = 0; i < src->n_nodes; ++i) {
7577              const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
7578              const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);
             ...
7585              dst->grads[igrad_dst]     = src->grads[igrad_src];      // ← 复制梯度指针
7586              dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
```

**关键点**：

1. **张量指针完全共享**：`dst->nodes[i] == src->nodes[i]`。所以 `gb_grad` 的前 14 个节点就是 `gf` 的那 14 个张量。
2. **哈希表要重建**：因为哈希槽位是按插入顺序分配的，`dst` 是新的空表，必须逐个 `ggml_hash_insert`。这是 O(N) 的，但 N=2048，可忽略。
3. **`grads` 数组要复制**：如果 `src->grads` 存在（`gb_grad` 复制到 `gb_opt`），梯度指针被继承。

**成本**：`ggml_graph_dup` 一次约 **10~20 µs**（2048 个哈希槽的 `memset` + 28 个节点的指针复制）。`ggml_opt_build` 每 batch 调用一次（`ggml_opt_alloc` 里），所以每 batch 两次 dup（`gb_grad` + `gb_opt`），约 40 µs。这解释了 05 §8.3 里 `ggml_opt_alloc=158 µs` 的构成：

```text
ggml_opt_build（含 2 次 dup + 反向展开 + AdamW 追加） ≈ 100 µs
split_graph                                            ≈ 15 µs
alloc_splits                                           ≈ 42 µs
                                                     = 157 µs  ✅
```

---

## 5 三张图的对比

| | `gf` | `gb_grad` | `gb_opt` |
|---|---|---|---|
| 节点数 | 14 | 28 | 32 |
| 叶子数 | 2 | 7 | 16 |
| 内容 | 前向 + loss + pred + ncorrect | 前向 + 反向 | 前向 + 反向 + AdamW |
| 何时执行 | 推理 / 验证 | 训练第 1 个物理 batch | 训练第 2 个物理 batch |
| 分配时的 `graph_copy` 节点数 | 14 | 28 | 34（+2 个 input 节点） |
| `n_inputs` | 0 | 0 | 1（`adamw_params`） |

**为什么 `gb_opt` 的 `graph_copy` 是 34 而不是 32？** scheduler 在 split 时插入了两个节点：

```text
graph_copy[0] = adamw_params (view)        ← 依赖源张量，防止提前释放
graph_copy[1] = CUDA0#adamw_params#0       ← CPU→CUDA 的拷贝目标
```

见 03 §1.5 和 §2.4。

---

## 6 一个容易混淆的点：`gf` 到底有没有被单独执行？

训练时 `ggml_opt_fit` 创建了 `gf`，但训练循环里**从来没有单独执行 `gf`**：

```cpp
for (; ibatch < ibatch_split; ++ibatch) {
    ggml_opt_alloc(opt_ctx, /*backward=*/ true);    // 选 gb_grad 或 gb_opt
    ggml_opt_dataset_get_batch(...);
    ggml_opt_eval(opt_ctx, result_train);
}
```

`gf` 只是 `gb_grad` 和 `gb_opt` 的“原型”，`ggml_graph_dup` 复制了它的节点表。真正执行的是 `gb_grad` / `gb_opt`，它们的前 14 个节点就是 `gf` 的节点。

**验证**：第一次 `ggml_backend_cuda_graph_compute` 的 `n_nodes=28`，不是 14。`gf` 只在 `mnist_model_eval`（推理）里单独执行，`n_nodes=13`。

---

## 7 本篇结论

1. **建图 = 后序 DFS + 哈希去重**，不产生任何 kernel 调用，`data` 全为 `NULL`。
2. **`op=NONE` 且非 `PARAM` 的进 `leafs[]`，其余进 `nodes[]`**；这是反向传播能看到参数节点的前提。
3. **`use_counts` 统计的是整张图（含反向）的消费次数**，`use=0` 即图输出。
4. **反向是手写规则，不是自动微分**：`relu'` = `STEP × MUL`，`linear'` = `OUT_PROD`，`cross_entropy'` = `CROSS_ENTROPY_LOSS_BACK`。
5. **转置是视图，不是计算**：`TRANSPOSE` 节点 `data` 不变、`nb` 交换，CUDA 执行时被跳过。
6. **AdamW 是原地视图**：`ggml_opt_step_adamw` 返回 `ggml_view_tensor(a)`，参数地址不变。
7. **三张图共享同一批张量指针**：`ggml_graph_dup` 只复制节点表，不复制张量。
8. **`ggml_add_or_set` 是梯度累加的唯一入口**：第一次 `grads[isrc] = tensor`，后续 `ADD(grads, tensor, inplace=grad_acc)`。
9. **`ggml_opt_build` 的三段式是可重入的**：每 batch 调用一次，两次 `ggml_graph_dup` 约 40 µs，是 `ggml_opt_alloc` 158 µs 的主要构成。
