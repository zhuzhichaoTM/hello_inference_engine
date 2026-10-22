# 02 · build_graph：纯 IR 构图与张量布局

> 源码基线：`/home/ggml`，commit `353b63b`
>
> 对应源码：`examples/simple/simple-backend.cpp:67-96`、`src/ggml.c`（`ggml_init`、`ggml_new_object`、`ggml_new_tensor_impl`、`ggml_mul_mat`、`ggml_build_forward_expand`）
>
> 本文的核心结论：**`build_graph()` 不碰 CUDA、不碰显存、不做任何计算**。它只生产一份与后端无关的 IR（中间表示），以及一份精确的元数据内存账本。

---

## 0 源码与六个功能块

```cpp
struct ggml_cgraph * build_graph(simple_model& model) {
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later
    };

    struct ggml_context * ctx = ggml_init(params0);
    struct ggml_cgraph  * gf  = ggml_new_graph(ctx);

    model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);   // [2,4]
    model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);   // [2,3]

    struct ggml_tensor * result = ggml_mul_mat(ctx, model.a, model.b);  // [4,3]

    ggml_build_forward_expand(gf, result);
    ggml_free(ctx);

    return gf;
}
```

```text
A. buf_size 计算 + model.buf 扩容
B. ggml_init：在外部内存上建立 context（no_alloc=true）
C. ggml_new_graph：创建 cgraph 容器
D. ggml_new_tensor_2d：创建 a、b 的元数据
E. ggml_mul_mat：创建 result 节点 + 表达式
F. ggml_build_forward_expand：后序 DFS，填充 nodes[] / leafs[]
G. ggml_free(ctx)：释放 context 管理状态，元数据留在 model.buf
```

---

## 1 `buf_size`：836,272 字节的精确推导

```cpp
size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
```

### 1.1 两个 overhead

```cpp
size_t ggml_tensor_overhead(void) {
    return GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;   // 32 + 336 = 368
}

size_t ggml_graph_overhead(void) {
    return ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
}
```

gdb 实测：

```text
sizeof(ggml_tensor)    = 336
sizeof(ggml_object)    = 32
ggml_tensor_overhead() = 368
ggml_graph_overhead()  = 82,608
```

### 1.2 `ggml_graph_overhead` 的组成

`ggml_graph_nbytes(2048, false)`（`ggml.c:7451`）：

```cpp
size_t hash_size = ggml_hash_size(size * 2);       // ggml_hash_size(4096) = 4099

incr_ptr_aligned(&p, sizeof(struct ggml_cgraph), 1);                 // 96
incr_ptr_aligned(&p, size * sizeof(struct ggml_tensor *), 8);        // nodes  2048*8 = 16384
incr_ptr_aligned(&p, size * sizeof(struct ggml_tensor *), 8);        // leafs  2048*8 = 16384
incr_ptr_aligned(&p, hash_size * sizeof(int32_t), 4);                // use_counts 4099*4 = 16396
incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), 8);   // hash keys  4099*8 = 32792
incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t), 4); // used 129*4 = 516
```

汇总：

```text
96 + 16,384 + 16,384 + 16,396 + 32,792 + 516 = 82,568
GGML_PAD(82,568, 16) = 82,576
GGML_OBJECT_SIZE(32) + 82,576 = 82,608
```

### 1.3 总计

```text
368 * 2048 = 753,664
753,664 + 82,608 = 836,272
```

断点验证：

```text
=== [BP6] ggml_init (build_graph) mem_size=836272 no_alloc=1 mem_buffer=0x7f0b04f46010 ===
```

### 1.4 这笔账说明了什么

`model.buf` 是**按图容量预留的元数据 arena**：

```text
2048 个 tensor 的元数据槽位（实际只用 3 个）
+ 一张 2048 节点的图容器
```

它不是数据 buffer。三个 tensor 的元素数据（32 + 24 + 48 = 104 B）在 `compute()` 阶段才由 CUDA buffer 分配。

**设计取舍**：按最大图预留，构图期间不需要 realloc；代价是 836 KB 的固定 host 内存。对 llama.cpp 的 2048 节点图是合理上限；对超大图可以调大 `GGML_DEFAULT_GRAPH_SIZE` 或使用 `ggml_new_graph_custom`。

---

## 2 `ggml_init()`：外部内存 + `no_alloc`

```cpp
struct ggml_init_params params0 = {
    .mem_size   = buf_size,
    .mem_buffer = model.buf.data(),
    .no_alloc   = true,
};
```

| 字段 | 值 | 作用 |
|---|---|---|
| `mem_size` | 836,272 | arena 容量上限 |
| `mem_buffer` | `model.buf.data()` | 外部提供的内存，GGML 不负责 free |
| `no_alloc` | true | 创建 tensor 时不分配元素数据 |

### 2.1 arena 分配器

`ggml_init` 后，context 内部维护一条 `ggml_object` 链表：

```cpp
struct ggml_object {
    size_t offs;                 // 在 mem_buffer 中的偏移
    size_t size;
    struct ggml_object * next;
    enum ggml_object_type type;
    char padding[4];
};
```

每次 `ggml_new_object`：

```text
cur_end      = 上一个对象的 offs + size
size_needed  = GGML_PAD(size, 16)
obj_new      = mem_buffer + cur_end
obj_new->offs = cur_end + GGML_OBJECT_SIZE
obj_new->size = size_needed
下一个 cur_end = obj_new->offs + size_needed
```

因为 `GGML_OBJECT_SIZE = 32` 和 `size_needed` 都是 16 的倍数，所有 payload 都保持 16 字节对齐。

对象类型（`include/ggml.h:646`）：

```text
GGML_OBJECT_TYPE_TENSOR
GGML_OBJECT_TYPE_GRAPH
GGML_OBJECT_TYPE_WORK_BUFFER
```

`ggml_free(ctx)` 只销毁链表头，不回收单个对象——arena 是“一次性”的。

### 2.2 `no_alloc` 的精确含义

```text
影响：tensor 的元素数据（data 指向的内存）
不影响：tensor 元数据（ggml_tensor 结构、name）
```

所以 `a`、`b`、`result` 的结构都在 `model.buf` 里，但 `data == NULL`。

如果 `no_alloc=false`，`ggml_new_tensor_impl` 会在 context 的 arena 里为元素数据也分配空间，`data` 指向 arena 内部。这对小图方便，对大模型不可行（权重几百 MB，不能塞进元数据 arena）。

**为什么 `simple-backend` 用 `no_alloc=true`**：它要把数据内存交给 backend（CUDA/CPU）分配，而不是 host arena。这也是所有真实推理框架的做法。

---

## 3 `ggml_new_graph()`：图容器

```cpp
struct ggml_cgraph * gf = ggml_new_graph(ctx);
```

`ggml_new_graph` → `ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false)`：

```text
在 arena 中创建一个 GGML_OBJECT_TYPE_GRAPH 对象
对象大小 = ggml_graph_nbytes(2048, false) = 82,576
```

内部布局：

```text
ggml_cgraph (96 B)
├── nodes[2048]       指针数组，指向 node tensor
├── leafs[2048]       指针数组，指向 leaf tensor
├── use_counts[4099]  每个 tensor 被使用的次数
├── visited_hash_set
│     keys[4099]      开放寻址哈希表
│     used[129]       bitset（uint32_t）
└── order             拓扑序（构建时填充）
```

此时：

```text
gf->n_nodes = 0
gf->n_leafs = 0
```

`nodes[]` / `leafs[]` 只是预留数组，尚未填充。

### 3.1 为什么图需要哈希表

构图时同一 tensor 可能被多个节点引用（例如 `a` 同时参与 `mul_mat` 和 `add`）。DFS 需要：

```text
1. 去重：同一个 tensor 只访问一次
2. 快速查找：判断是否已访问
3. 统一索引：use_counts、order 的下标
```

哈希表用 tensor 指针做 key，`ggml_hash_find_or_insert` 返回槽位下标。

---

## 4 创建 leaf 张量：`ne`、`nb`、`data`

```cpp
model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);  // ne0=2, ne1=4
model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);  // ne0=2, ne1=3
```

断点：

```text
=== [BP7] ggml_new_tensor_impl type=0 n_dims=2 ne=[2,4,...] view_src=(nil) ===
#0 ggml_new_tensor_impl (...) at ggml.c:1772
#1 ggml_new_tensor (...)      at ggml.c:1849
#2 ggml_new_tensor_2d (..., ne0=2, ne1=4) at ggml.c:1865
#3 build_graph (...)          at simple-backend.cpp:84

=== [BP7] ggml_new_tensor_impl type=0 n_dims=2 ne=[2,3,...] view_src=(nil) ===
#3 build_graph (...)          at simple-backend.cpp:85
```

### 4.1 `ne`：元素个数，最内层在前

GGML 的维度顺序与 NumPy 相反：

```text
ne[0] = 最内层（连续）维度
ne[1] = 行数
ne[2] = 第三维（batch/channel）
ne[3] = 第四维（batch/sample）
```

`a` 的逻辑形状：

```text
ne = [2, 4]
     ┌───────┐
     │ 2 列  │  ← 每行 2 个 float
     │ 4 行  │
     └───────┘
```

### 4.2 `nb`：字节步长

`ggml_new_tensor_impl` 中：

```cpp
tensor->ne[0] = ne[0];
tensor->nb[0] = ggml_type_size(type);       // F32 = 4
tensor->nb[1] = tensor->nb[0] * ne[0];      // 4*2 = 8
tensor->nb[2] = tensor->nb[1] * ne[1];      // 8*4 = 32
tensor->nb[3] = tensor->nb[2] * ne[2];      // 32*1 = 32
```

对 `a`：

```text
nb[0] = 4     一个 float
nb[1] = 8     一行（2 个 float）
nb[2] = 32    整个矩阵（4 行）
nb[3] = 32
```

对 `b`：

```text
ne = [2, 3]
nb = [4, 8, 24, 24]
```

断点验证（`ggml_build_forward_expand` 之后）：

```text
model.a->nb = {4, 8, 32, 32}
model.b->nb = {4, 8, 24, 24}
```

### 4.3 内存布局示例

`a` 的 8 个 float 在内存中：

```text
offset  0   4   8  12  16  20  24  28
        a00 a01 a10 a11 a20 a21 a30 a31
        └─row0─┘ └─row1─┘ └─row2─┘ └─row3─┘
```

行主序、连续。`nb[1] = 8` 表示行与行之间隔 8 字节。

`result` 的 `ne=[4,3]`、`nb=[4,16,48,48]`：

```text
offset  0   4   8  12 | 16  20  24  28 | 32  36  40  44
        r00 r01 r02 r03 | r10 r11 r12 r13 | r20 r21 r22 r23
        └──── row0 ────┘ └──── row1 ────┘ └──── row2 ────┘
```

每行 4 个 float，行步长 16 字节。

### 4.4 `GGML_MAX_DIMS = 4`

即使 `a` 只有 2 维，`ne[2]`、`ne[3]` 也被填成 1。所有 tensor 都是 4 维，简化 kernel 实现：

```text
ne[2] = ne[3] = 1
nb[2] = nb[3] = 矩阵总大小
```

这样 kernel 可以用统一的 4 层循环处理 batch/channel，不需要为 1D/2D/3D 写不同代码。

### 4.5 此时 `data` / `buffer` / `view_src`

```text
model.a->data     = NULL
model.a->buffer   = NULL
model.a->view_src = NULL
model.b->data     = NULL
model.b->buffer   = NULL
```

`ggml_nbytes(a) = 32`、`ggml_nbytes(b) = 24` 只是逻辑大小，尚未映射到任何内存。

### 4.6 为什么这里不做 128 字节对齐

`ne/nb` 描述的是**逻辑布局**，不是物理分配。对齐是 buffer type 的职责：

```text
逻辑：a 占 32 B，b 占 24 B，result 占 48 B
物理：CUDA buft 按 128 B 对齐 → 128 + 128 + 128 = 384 B
```

这就是为什么 `ggml_nbytes` 不能直接当显存占用用。

---

## 5 `ggml_mul_mat()`：创建节点表达式

```cpp
struct ggml_tensor * result = ggml_mul_mat(ctx, model.a, model.b);
```

断点（`simple-backend.cpp:91`）：

```text
=== [BP8] after ggml_mul_mat (before build_forward_expand) ===
result->op       = GGML_OP_MUL_MAT
result->ne       = {4, 3, 1, 1}
result->nb       = {4, 16, 48, 48}
result->data     = NULL
result->buffer   = NULL
result->view_src = NULL
ggml_nbytes(result) = 48
gf->n_nodes = 0
gf->n_leafs = 0
```

### 5.1 形状规则

`ggml_mul_mat(a, b)` 计算：

```text
result = a * b^T
result->ne[0] = a->ne[1] = 4
result->ne[1] = b->ne[1] = 3
result->ne[2] = b->ne[2] = 1
result->ne[3] = b->ne[3] = 1
```

源码（`ggml.c:3348`）：

```cpp
const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
```

为什么是 `a * b^T` 而不是 `a * b`：

```text
ggml 的 mul_mat 约定：src0 视为 [K, M]（ne0=K, ne1=M）
                      src1 视为 [K, N]（ne0=K, ne1=N）
结果：[M, N]，即 ne0=M, ne1=N
```

所以 `a[2,4] * b[2,3]^T = [4,3]`。

### 5.2 `src[]` 只是指针

```cpp
result->src[0] = model.a;
result->src[1] = model.b;
```

不是数据拷贝。`src[0]->data` 仍然是 `NULL`。

### 5.3 `op` 只是标记

`result->op = GGML_OP_MUL_MAT` 表示“这个节点将来要做矩阵乘”，不代表已经做了。

### 5.4 为什么 `gf->n_nodes` 仍是 0

`ggml_mul_mat` 只创建 tensor 和表达式，**不会自动把节点挂进图**。挂图必须显式调用：

```cpp
ggml_build_forward_expand(gf, result);
```

这是 ggml 的设计：算子构造与图构建分离。好处：

```text
1. 可以先构造多个输出，再一次性 expand
2. 可以构造不参与最终输出的中间节点（不会进图）
3. 可以多次 expand，把多个子图合并
```

---

## 6 `ggml_build_forward_expand()`：后序 DFS

```cpp
ggml_build_forward_expand(gf, result);
```

调用链：

```text
ggml_build_forward_expand
  → ggml_build_forward_impl(graph, tensor, 0, false)
    → ggml_visit_parents_graph(graph, tensor)
```

### 6.1 递归顺序

```text
visit(result)
  ├── 标记 result 已访问（哈希表去重）
  ├── visit(src[0] = a)
  │     ├── 标记 a 已访问
  │     ├── a 无 src
  │     └── a 是 leaf → leafs[n_leafs++] = a
  ├── visit(src[1] = b)
  │     ├── 标记 b 已访问
  │     ├── b 无 src
  │     └── b 是 leaf → leafs[n_leafs++] = b
  └── result 是 node → nodes[n_nodes++] = result
```

断点（`simple-backend.cpp:93`）：

```text
=== [BP9] after ggml_build_forward_expand (before ggml_free) ===
gf->n_nodes = 1
gf->n_leafs = 2
gf->nodes[0]->name = "node_0"
gf->nodes[0]->op   = GGML_OP_MUL_MAT
gf->leafs[0]->name = "leaf_0"
gf->leafs[1]->name = "leaf_1"
```

### 6.2 为什么是后序

前向执行要求“先算依赖，再算消费者”。后序 DFS 天然产生拓扑序：

```text
leafs 先被访问 → 在数组前面
node 后访问   → 在数组后面
```

`nodes[]` 的顺序就是执行顺序。本例只有 1 个节点，拓扑序平凡；真实模型里，这个顺序决定了 kernel launch 顺序。

### 6.3 哈希去重

```cpp
if (ggml_hash_find(&graph->visited_hash_set, tensor) != GGML_HASH_SET_NULL) {
    return;   // 已访问
}
ggml_hash_insert(&graph->visited_hash_set, tensor);
```

如果 `a` 同时被两个节点引用，第二次访问直接返回，不会重复加入 `leafs[]`。

### 6.4 `COMPUTE` 标志

`ggml_build_forward_expand` 会给参与计算的节点设置：

```text
GGML_TENSOR_FLAG_COMPUTE = 16
```

断点实测：

```text
tensor flags: a=0 b=0 result=16
```

含义：

```text
a、b：只是数据，不需要“计算”
result：必须被计算
```

CUDA 的 node 循环会跳过没有该标志的节点：

```cpp
if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
    continue;
}
```

### 6.5 自动命名

未命名的节点会被赋予：

```text
node_0
leaf_0
leaf_1
```

这些名字出现在：

```text
GGML_SCHED_DEBUG 输出
CUDA debug 日志
gdb 断点打印
```

真实模型中通常用 `ggml_set_name` 命名（如 `blk.0.attn_q.weight`），便于 profiling。

### 6.6 图的完整状态

```text
ggml_cgraph
  n_nodes = 1
  nodes[0] = result
      op     = MUL_MAT
      src[0] = a
      src[1] = b
      data   = NULL
      buffer = NULL
      flags  = COMPUTE

  n_leafs = 2
  leafs[0] = a   (ne=[2,4], nb=[4,8,32,32], data=NULL, flags=0)
  leafs[1] = b   (ne=[2,3], nb=[4,8,24,24], data=NULL, flags=0)
```

---

## 7 `ggml_free(ctx)`：生命周期边界

```cpp
ggml_free(ctx);
return gf;
```

断点（`simple-backend.cpp:95`）：

```text
=== [BP10] after ggml_free(ctx) ===
gf->n_nodes        = 1
gf->nodes[0]->name = "node_0"
gf->nodes[0]->data = NULL
model.a->data      = NULL
model.a->buffer    = NULL
model.buf.size()   = 836272
```

### 7.1 为什么图还能用

```cpp
struct ggml_init_params params0 = {
    .mem_buffer = model.buf.data(),   // 外部内存
};
```

`ggml_free(ctx)` 的行为：

```cpp
void ggml_free(struct ggml_context * ctx) {
    if (ctx == NULL) return;

    if (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}
```

本例 `mem_buffer_owned = false`，所以：

```text
model.buf 不被 free
gf、nodes[]、leafs[]、三个 ggml_tensor 仍然有效
```

### 7.2 两种所有权

| 场景 | `mem_buffer` | `mem_buffer_owned` | `ggml_free` 后 |
|---|---|---|---|
| 外部提供（本例） | `model.buf.data()` | false | 元数据有效，buffer 由调用者管理 |
| GGML 内部分配 | NULL | true | `mem_buffer` 被 free，图指针悬空 |

这是 `simple-backend` 能“先 free context、后 compute”的前提。

### 7.3 为什么要提前 free

```text
1. 释放 ctx 结构本身（几十字节）
2. 明确生命周期边界：构图阶段结束，后续只读图
3. 避免 context 的 arena 继续增长（如果后续还要建别的图）
```

它不释放 `model.buf`，因为 `simple_model` 拥有它。

---

## 8 为什么 `build_graph()` 对 GPU 零参与

| 动作 | 是否涉及 CUDA | 原因 |
|---|---|---|
| `model.buf.resize` | 否 | host `std::vector` |
| `ggml_init` | 否 | 纯 host arena |
| `ggml_new_graph` | 否 | 纯 host 元数据 |
| `ggml_new_tensor_2d` | 否 | 只算 `ne/nb` |
| `ggml_mul_mat` | 否 | 只建表达式 |
| `ggml_build_forward_expand` | 否 | 只做 DFS |
| `ggml_free(ctx)` | 否 | 只销毁 context 结构 |

GPU 的介入点全部在 `compute()`：

```text
split_graph    决定 node → CUDA0
gallocr        CUDA buffer 申请 + tensor 落位
tensor_set     H2D
graph_compute  kernel launch
```

**设计价值**：同一份 `build_graph()` 代码可以跑在 CPU、CUDA、Metal、Vulkan 上，不需要任何 `#ifdef`。后端差异被推迟到 `compute()`，由 scheduler 和 buft 吸收。

### 8.1 业务映射

| 业务场景 | 对应本阶段的机制 |
|---|---|
| GGUF 权重 mmap | 权重 tensor 的 `data` 指向 mmap 地址，`no_alloc=true` 不重复分配 |
| 模型形状推导 | `ggml_mul_mat` 等算子只算 `ne/nb`，不读数据 |
| 图优化 / 量化 | 在 `build_graph` 之后、`compute` 之前改写图 |
| 多后端统一 IR | 同一张 `ggml_cgraph` 可交给任意 backend |
| 图容量规划 | `GGML_DEFAULT_GRAPH_SIZE` 决定 `model.buf` 与图数组大小 |
| 生命周期管理 | 外部 `mem_buffer` 让图在 `ggml_free(ctx)` 后仍有效 |

**一个真实约束**：`build_graph` 阶段不读权重，所以形状推导必须只依赖 `ne`，不能依赖数据。这也是 ggml 算子构造函数只接受 `ggml_tensor *` 和标量参数、不执行计算的原因。

---

## 9 数据表：三个 tensor 的全字段

| 字段 | `a` | `b` | `result` |
|---|---|---|---|
| `name`（expand 前） | 空 | 空 | 空 |
| `name`（expand 后） | `leaf_0` | `leaf_1` | `node_0` |
| `type` | F32 | F32 | F32 |
| `ne` | `[2,4,1,1]` | `[2,3,1,1]` | `[4,3,1,1]` |
| `nb` | `[4,8,32,32]` | `[4,8,24,24]` | `[4,16,48,48]` |
| `nbytes` | 32 | 24 | 48 |
| `op` | NONE | NONE | MUL_MAT |
| `src[0]` | NULL | NULL | `a` |
| `src[1]` | NULL | NULL | `b` |
| `data` | NULL | NULL | NULL |
| `buffer` | NULL | NULL | NULL |
| `view_src` | NULL | NULL | NULL |
| `flags` | 0 | 0 | 16 (COMPUTE) |
| 图中位置 | `leafs[0]` | `leafs[1]` | `nodes[0]` |

---

## 10 扩展：在构图阶段能做什么

### 10.1 标记输入/输出

```cpp
ggml_set_input(model.a);
ggml_set_input(model.b);
ggml_set_output(result);
```

影响：

```text
INPUT：
  - scheduler pass 1 会把它判给 CPU（最后一个 backend），除非已有 buffer
  - gallocr 在分配阶段提前分配它，避免在它第一次被使用前被其他节点的输出占用
  - n_copies>1 时会为它创建多份 copy

OUTPUT：
  - gallocr 永不释放它（即使 n_children 归零）
  - 保证结果在 compute 后仍可读
```

本例没有标记，所以：

```text
a、b 在 node_0 执行后被 gallocr 释放（n_children 归零）
result 因为没有任何节点消费它，不会被 free_node 调用，恰好保留
```

这能工作是因为 alloc 之后没有新的分配；在真实模型中，务必标记输入/输出，否则内存复用可能覆盖输入或输出。

### 10.2 多节点图

```cpp
auto t0 = ggml_mul_mat(ctx, a, b);
auto t1 = ggml_add(ctx, t0, bias);
auto t2 = ggml_silu(ctx, t1);
ggml_build_forward_expand(gf, t2);
```

`build_forward_expand` 会 DFS 出所有依赖，`nodes[]` 顺序为 `t0, t1, t2`。scheduler 会按 node 逐个判 backend，可能在 `t1` 处切 split。

### 10.3 调整图容量

```cpp
struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
```

同时要调大 `buf_size`：

```cpp
size_t buf_size = ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(4096, false);
```

### 10.4 不参与输出的节点

```cpp
auto t = ggml_mul_mat(ctx, a, b);   // 不 expand
```

它不会进图，也不会被计算。这是“惰性图”的直接体现。

---

## 11 常见误区

### 误区 1：`ggml_mul_mat()` 已经执行了矩阵乘

没有。它只创建节点，`data == NULL`。执行在 `compute()`。

### 误区 2：`no_alloc=true` 表示 tensor 没创建

只影响元素数据。`ggml_tensor` 结构、`ne/nb/op/src` 都已创建。

### 误区 3：`ggml_free(ctx)` 后图失效

本例不会。元数据在外部 `model.buf`，`ggml_free` 只销毁 context 结构。

### 误区 4：`ne` 是字节数

`ne` 是元素个数，`nb` 是字节步长。

### 误区 5：`result->src[0]` 是数据拷贝

`src[]` 只是指针，数据在 `compute()` 阶段才写入显存。

### 误区 6：`ggml_nbytes(a) = 32` 就是显存占用

不是。CUDA buft 按 128 B 对齐，实际占用 128 B。

### 误区 7：`gf->n_nodes` 在 `ggml_mul_mat` 后就会增加

不会。必须显式 `ggml_build_forward_expand`。

---

## 12 源码逐行：构图链路

本节逐行拆解 `build_graph()` 触发的 7 个核心函数。行号对应云端 `/home/ggml`（commit `353b63b`）。

### 12.1 `ggml_init()`

`src/ggml.c:1611`

```cpp
struct ggml_context * ggml_init(struct ggml_init_params params) {
    static bool is_first_call = true;

    ggml_critical_section_start();

    if (is_first_call) {
        ggml_time_init();
        is_first_call = false;
    }

    ggml_critical_section_end();

    struct ggml_context * ctx = GGML_MALLOC(sizeof(struct ggml_context));

    // allow to call ggml_init with 0 size
    if (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (struct ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
    };

    GGML_ASSERT(ctx->mem_buffer != NULL);
    GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    return ctx;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1612 | `static bool is_first_call` | 进程内只初始化一次计时系统；多线程由 `critical_section` 保护 |
| 1614-1622 | 临界区 | 只在首次调用时有实际工作，之后是一次加锁/解锁 |
| 1624 | `GGML_MALLOC(sizeof(struct ggml_context))` | **context 结构本身永远在堆上**，不在 arena 里 |
| 1627-1629 | `mem_size == 0 → GGML_MEM_ALIGN` | 允许 `ggml_init({0})` 创建“空 context”，CPU backend 初始化时用到（断点 BP6 的 `mem_size=0`） |
| 1631 | `params.mem_buffer ? params.mem_size : GGML_PAD(...)` | **外部 buffer 时不做 PAD**（调用者负责），内部 malloc 时向上对齐到 16 |
| 1634 | `mem_buffer = params.mem_buffer ? ... : ggml_aligned_malloc(mem_size)` | 关键分支：本程序走前者 |
| 1635 | `mem_buffer_owned = params.mem_buffer ? false : true` | **这一行决定了 `ggml_free` 是否释放 buffer** |
| 1636 | `no_alloc = params.no_alloc` | 本程序为 true |
| 1637-1638 | `objects_begin/end = NULL` | 对象链表为空，第一个 `ggml_new_object` 会设置 `objects_begin` |

**为什么 `mem_size` 在外部 buffer 时不做 PAD**：

```cpp
const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);
```

外部 buffer 的大小由调用者决定，GGML 不应该悄悄改它。断点验证：`mem_size = 836272`，正好是 `368*2048 + 82608`，没有被 PAD 成别的值。

### 12.2 `ggml_new_object()`

`src/ggml.c:1708`

```cpp
static struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size) {
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

    // integer overflow checks
    if (cur_end > SIZE_MAX - size_needed) { ... return NULL; }
    if (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) { ... return NULL; }

    if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", ...);
#ifndef NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#endif
        return NULL;
    }

    *obj_new = (struct ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    return obj_new;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1710-1714 | 计算 `cur_end` | 从链表尾对象算起，**永远追加**，不回收 |
| 1718 | `GGML_PAD(size, GGML_MEM_ALIGN)` | payload 对齐到 16 |
| 1719 | `obj_new = mem_buffer + cur_end` | 对象头直接放在当前末尾 |
| 1722-1733 | 三重溢出/容量检查 | Debug 下容量不足直接 `GGML_ABORT`，Release 下返回 NULL |
| 1735-1740 | 初始化对象头 | **`offs = cur_end + GGML_OBJECT_SIZE`**：payload 紧跟在对象头之后 |
| 1742 | `GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs)` | 因为 `GGML_OBJECT_SIZE = 32` 是 16 的倍数，payload 仍然 16 对齐 |
| 1744-1748 | 链表串接 | 第一个对象设置 `objects_begin` |
| 1750 | `objects_end = obj_new` | 尾指针前移 |

**关键：`.offs` 是相对 `mem_buffer` 的偏移，不是绝对地址**。所以 `ggml_free(ctx)` 之后，只要 `mem_buffer` 还在，所有对象仍然有效。

`ggml_used_mem()` 直接读尾部：

```cpp
size_t ggml_used_mem(const struct ggml_context * ctx) {
    return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}
```

### 12.3 `ggml_new_tensor_impl()`

`src/ggml.c:1766`

```cpp
static struct ggml_tensor * ggml_new_tensor_impl(
        struct ggml_context * ctx,
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t       * ne,
        struct ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    if (view_src != NULL && view_src->view_src != NULL) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    void * data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    if (view_src == NULL && !ctx->no_alloc) {
        obj_alloc_size = data_size;
    }

    GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);

    struct ggml_object * const obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
    GGML_ASSERT(obj_new);

    struct ggml_tensor * const result = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);

    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ NULL,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { NULL },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        /*.padding      =*/ { 0 },
    };

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1774-1775 | 类型/维度断言 | 防止非法枚举和越界写 `ne[]` |
| 1778-1781 | view 链扁平化 | **`view_src->view_src` 非空时，向上找到根 tensor，并累加偏移**。避免多层 view 链导致递归解析 |
| 1783-1786 | `data_size` 计算 | 从 `ggml_row_size(type, ne[0])` 起乘，**而不是 `ne[0]*ne[1]*...*type_size`**。这是量化类型的关键：`ne[0]` 个元素可能不足一个 block，`ggml_row_size` 会向上取整到 block |
| 1788 | view 越界检查 | `data_size + view_offs <= ggml_nbytes(view_src)` |
| 1790-1793 | `data` 指向 view 源 | view tensor **不拥有数据**，`data = view_src->data + view_offs` |
| 1795-1798 | `obj_alloc_size` | **只有“非 view 且 no_alloc=false”才在 arena 里分配数据**。本程序 `no_alloc=true`，所以为 0 |
| 1804 | `ggml_new_object(..., GGML_TENSOR_SIZE + obj_alloc_size)` | 元数据和数据在同一个 object 里连续存放 |
| 1807 | `result = mem_buffer + obj_new->offs` | 元数据地址 |
| 1809-1826 | 聚合初始化 | **`.ne` 初始化为 `{1,1,1,1}`、`.nb` 为 `{0,0,0,0}`**，后面再填 |
| 1823 | `.data = obj_alloc_size > 0 ? (result + 1) : data` | 两种模式：arena 内数据紧跟在 `ggml_tensor` 之后；否则用 view 的 data 或 NULL |
| 1828-1830 | 填 `ne[0..n_dims-1]` | **只填实际维度，其余保持 1** |
| 1832 | `nb[0] = ggml_type_size(type)` | 单个元素（或 block）字节数 |
| 1833 | `nb[1] = nb[0] * (ne[0] / ggml_blck_size(type))` | **注意除以 `blck_size`**：Q4_0 的 `blck_size=32`，`ne[0]=64` 时 `nb[1] = 18 * 2 = 36` |
| 1834-1836 | `nb[i] = nb[i-1] * ne[i-1]` | 逐维累乘 |

**为什么 `nb[1]` 要除以 `blck_size`**：

```text
F32：blck_size = 1  → nb[1] = 4 * ne[0]
Q4_0：blck_size = 32 → 每 32 个元素占 18 字节
       nb[1] = 18 * (ne[0] / 32)
```

这保证了 `nb` 始终是**字节步长**，与元素类型无关。

断点验证：

```text
[BP7] type=0 n_dims=2 ne=[2,4] view_src=(nil)
→ a: ne=[2,4,1,1] nb=[4,8,32,32]
      nb[0]=4, nb[1]=4*(2/1)=8, nb[2]=8*4=32, nb[3]=32*1=32
```

### 12.4 `ggml_mul_mat()`

`src/ggml.c:3341`

```cpp
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 3345 | `ggml_can_mul_mat(a, b)` | 检查 `a->ne[0] == b->ne[0]`、batch 维可广播 |
| 3346 | `!ggml_is_transposed(a)` | **a 不能是转置 view**，否则 `nb[0] > nb[1]`，kernel 的 `vec_dot` 假设失效 |
| 3348 | `ne = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] }` | 输出形状：`M = a->ne[1]`、`N = b->ne[1]`，batch 维来自 `b` |
| 3349 | `ggml_new_tensor(ctx, F32, 4, ne)` | **输出永远是 F32**，即使输入是 F16/量化 |
| 3351-3353 | 设置 op 和 src | 只建表达式，`data/buffer` 仍为 NULL |

**为什么输出固定 F32**：`MUL_MAT` 的语义是“累加”，量化/F16 输入的累加结果用 F32 表示最通用；后续 op 如果需要 F16 再做转换。本例 `a`、`b` 都是 F32，所以输出也是 F32。

**为什么 batch 维来自 `b` 而不是 `a`**：`ggml_can_mul_mat` 要求 `a` 的 batch 维为 1 或与 `b` 相同；结果形状由 `b` 决定，符合 `[K,M] × [K,N] → [M,N]` 的推广。

### 12.5 `ggml_visit_parents_graph()`

`src/ggml.c:7236`

```cpp
static size_t ggml_visit_parents_graph(struct ggml_cgraph * cgraph, struct ggml_tensor * node, bool compute) {
    if (node->op != GGML_OP_NONE && compute) {
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    const size_t node_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
    GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);

    if (ggml_bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
        // already visited
        if (compute) {
            // update the compute flag regardless
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                struct ggml_tensor * src = node->src[i];
                if (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }
        return node_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.keys[node_hash_pos] = node;
    ggml_bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
    cgraph->use_counts[node_hash_pos] = 0;

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const int k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) : i;

        struct ggml_tensor * src = node->src[k];
        if (src) {
            const size_t src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);
        if (strlen(node->name) == 0) {
            ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
        }
        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);
        if (strlen(node->name) == 0) {
            ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }
        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->n_nodes++;
    }

    return node_hash_pos;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 7237-7239 | 设置 `GGML_TENSOR_FLAG_COMPUTE` | **只有非 NONE 的节点才是“要计算的”**；leaf 不设 |
| 7241-7242 | `ggml_hash_find` | 在哈希表里找槽位；找不到返回 `GGML_HASHSET_FULL` |
| 7244-7258 | 已访问分支 | **关键**：即使已访问，如果 `compute=true`，仍会递归给未标记的 src 补 `COMPUTE` 标志。这是 `build_forward_expand` 与 `build_forward_order` 的区别 |
| 7261-7263 | 首次访问登记 | `keys[pos] = node`、`used` 置位、`use_counts[pos] = 0` |
| 7265-7278 | 递归 src 并累加 `use_counts` | **`use_counts` 是“被引用次数”**，后续 gallocr 的 `n_children` 类似但不完全相同（gallocr 自己再统计一次） |
| 7266-7269 | `k` 的选择 | `LEFT_TO_RIGHT` 时按 `src[0], src[1], ...`；`RIGHT_TO_LEFT` 时反向。**这决定了 `leafs[]` 的顺序** |
| 7280-7289 | leaf 分支 | `op == NONE` 且不是 `PARAM` → 放入 `leafs[]`，自动命名 `leaf_%d` |
| 7290-7299 | node 分支 | 其余放入 `nodes[]`，自动命名 `node_%d` |

**为什么 `nodes[]` 是后序**：递归 src 在写 `nodes[]` 之前完成，所以依赖节点先入数组。执行时按 `nodes[0..n-1]` 顺序遍历，天然满足拓扑序。

**本例的递归过程**：

```text
visit(result)  pos=?
  used 置位, use_counts=0
  src[0]=a → visit(a)
      a 是 leaf → leafs[0]=a, 命名 "leaf_0"
      return pos_a
    use_counts[pos_a] = 1
  src[1]=b → visit(b)
      b 是 leaf → leafs[1]=b, 命名 "leaf_1"
      return pos_b
    use_counts[pos_b] = 1
  result 不是 leaf → nodes[0]=result, 命名 "node_0"
```

断点验证：

```text
[BP9] gf->n_nodes=1 gf->n_leafs=2
      nodes[0]=node_0  leafs[0]=leaf_0  leafs[1]=leaf_1
```

### 12.6 `ggml_build_forward_impl()` 与 `ggml_build_forward_expand()`

`src/ggml.c:7304`

```cpp
static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand, bool compute) {
    if (!expand) {
        ggml_graph_clear(cgraph);
    }

    const int n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    const int n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 7305-7308 | `!expand` 时清空 | 保留给“重建整图”的语义；`build_forward_expand` 传 `expand=true`，不清空 |
| 7310 | `n_old = n_nodes` | 用于判断“这次新增了多少节点” |
| 7312 | 递归访问 | `compute=true` |
| 7314 | `n_new = n_nodes - n_old` | 如果全部已访问，`n_new = 0` |
| 7317-7320 | 断言最后一个节点是起点 | **保证 `nodes[n_nodes-1]` 就是传入的输出 tensor**，这是“最后一个 node 是图输出”的约定 |
| 7337-7339 | `build_forward_expand` 直接转调 | `expand=true, compute=true` |

**`expand` 的语义**：多次调用 `ggml_build_forward_expand(gf, t1)`、`ggml_build_forward_expand(gf, t2)` 会把两张子图合并，已访问的节点不会重复加入。这就是为什么 `n_new` 可能为 0。

### 12.7 `ggml_new_graph_custom()`

`src/ggml.c:7477`

```cpp
struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads) {
    const size_t obj_size = ggml_graph_nbytes(size, grads);
    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *) ((char *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);

    void * p = cgraph + 1;

    struct ggml_tensor ** nodes_ptr      = incr_ptr_aligned(&p, size      * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
    struct ggml_tensor ** leafs_ptr      = incr_ptr_aligned(&p, size      * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
    int32_t             * use_counts_ptr = incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t));
    struct ggml_tensor ** hash_keys_ptr  = incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
    struct ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), ...) : NULL;
    struct ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), ...) : NULL;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    assert(obj_size == (size_t)((char *)p - (char *)cgraph));

    *cgraph = (struct ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_keys_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);

    return cgraph;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 7478 | `obj_size = ggml_graph_nbytes(size, grads)` | 先算总大小，再一次性申请 |
| 7479-7480 | 一个 object 容纳整个图 | **图容器是“单块连续内存”**：结构体 + 数组 + 哈希表 |
| 7483 | `hash_size = ggml_hash_size(size * 2)` | 哈希表要容纳 nodes + leafs，所以 `2 * size` |
| 7485 | `p = cgraph + 1` | 从结构体末尾开始切分 |
| 7487-7492 | `incr_ptr_aligned` 依次切分 | **每个数组按自身元素大小对齐**：指针 8、int32 4 |
| 7494 | bitset | 哈希表的 `used` 位图 |
| 7497 | `assert(obj_size == p - cgraph)` | **自检**：切分长度必须等于预算 |
| 7499-7511 | 初始化结构体 | `size` 是容量，`n_nodes/n_leafs` 从 0 开始 |
| 7513 | `ggml_hash_set_reset` | 清空 `used` 位图 |

**为什么用 `incr_ptr_aligned` 而不是分别 `malloc`**：

```text
1. 一次分配，零碎片
2. 与 context 的 arena 语义一致
3. ggml_free(ctx) 时随 arena 一起失效（或随外部 buffer 保留）
4. 缓存局部性好：nodes/leafs/hash 在同一页
```

### 12.8 `ggml_free()`

`src/ggml.c:1663`

```cpp
void ggml_free(struct ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    GGML_FREE(ctx);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1664-1666 | NULL 检查 | `ggml_free(NULL)` 是合法的 no-op；`sched->ctx` 初始为 NULL |
| 1668 | `if (ctx->mem_buffer_owned)` | **外部 buffer 时跳过**；本程序走这一支 |
| 1669 | `ggml_aligned_free(mem_buffer, mem_size)` | 必须传 size，因为 `aligned_alloc` 的实现需要原始指针 |
| 1672 | `GGML_FREE(ctx)` | 只释放 context 结构本身 |

**这就是为什么 `build_graph()` 可以 `ggml_free(ctx)` 后继续用 `gf`**：

```text
ctx->mem_buffer_owned = false（因为 params.mem_buffer != NULL）
→ 不释放 model.buf
→ gf / nodes[] / leafs[] / 三个 tensor 全部仍指向 model.buf 内部
→ model.buf 是 simple_model 的成员，生命周期覆盖 main()
```

### 12.9 完整调用链（带行号）

```text
build_graph()                                    simple-backend.cpp:68
 ├── ggml_tensor_overhead()                      ggml.c:1466
 ├── ggml_graph_overhead()                       ggml.c:7473
 │    └── ggml_graph_overhead_custom(2048,false) ggml.c:7469
 │         └── ggml_graph_nbytes(2048,false)     ggml.c:7451
 │              └── ggml_hash_size(4096)         ggml.c:6631
 ├── model.buf.resize(836272)
 ├── ggml_init({836272, buf, true})              ggml.c:1611
 ├── ggml_new_graph(ctx)                         ggml.c:7522
 │    └── ggml_new_graph_custom(ctx, 2048, false) ggml.c:7477
 │         ├── ggml_graph_nbytes()
 │         └── ggml_new_object(GRAPH, 82576)     ggml.c:1708
 ├── ggml_new_tensor_2d(a)                       ggml.c:1859
 │    ├── ggml_new_tensor()                      ggml.c:1844
 │    │    └── ggml_new_tensor_impl()            ggml.c:1766
 │    │         └── ggml_new_object(TENSOR, 336) ggml.c:1708
 ├── ggml_new_tensor_2d(b)                       同上
 ├── ggml_mul_mat(ctx, a, b)                     ggml.c:3341
 │    └── ggml_new_tensor(F32, 4, ne)            → impl
 ├── ggml_build_forward_expand(gf, result)       ggml.c:7337
 │    └── ggml_build_forward_impl(gf, result, true, true)  ggml.c:7304
 │         └── ggml_visit_parents_graph()        ggml.c:7236
 └── ggml_free(ctx)                              ggml.c:1663
```

### 12.10 三个“为什么这样设计”

**① 为什么 tensor 元数据和数据在同一个 object**

```cpp
ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
...
.data = obj_alloc_size > 0 ? (void *)(result + 1) : data
```

`no_alloc=false` 时，数据紧跟在 `ggml_tensor` 之后，一次分配拿到两者。好处：

```text
1. 元数据和数据在同一缓存行附近，遍历时局部性好
2. 只需一次 ggml_new_object，减少对象链表长度
3. ggml_free(ctx) 时一起失效，无泄漏
```

`no_alloc=true` 时 `obj_alloc_size=0`，数据留给 backend buffer。

**② 为什么 `nb[1]` 要除以 `blck_size`**

```cpp
result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
```

量化类型的一个 block 包含多个元素（Q4_0 = 32），`nb[0]` 是 block 的字节数（18），所以一行 `ne[0]` 个元素占 `nb[0] * (ne[0]/32)` 字节。如果不除，`nb[1]` 会大 32 倍。

**③ 为什么 `use_counts` 在构图时就统计**

```cpp
cgraph->use_counts[src_hash_pos]++;
```

gallocr 后面还要自己统计 `n_children`，看起来重复。区别：

```text
cgraph->use_counts：构图期统计，供调度/调试使用（GGML_SCHED_DEBUG 的 use=N）
galloc->hash_node.n_children：分配期统计，供生命周期释放使用
```

两者独立，因为 scheduler 可能修改图（插入 copy 节点），gallocr 看到的是修改后的图。

---

## 13 断点与变量清单

```gdb
break simple-backend.cpp:79     # ggml_init
break ggml_new_tensor_impl
break simple-backend.cpp:91     # ggml_mul_mat 之后
break simple-backend.cpp:93     # build_forward_expand 之后
break simple-backend.cpp:95     # ggml_free 之后
```

重点：

```gdb
p params0.mem_size
p params0.no_alloc
p params0.mem_buffer
p result->ne
p result->nb
p result->op
p result->data
p result->buffer
p result->flags
p result->src[0]
p result->src[1]
p ggml_nbytes(result)
p gf->n_nodes
p gf->n_leafs
p gf->nodes[0]->name
p gf->leafs[0]->name
p gf->size
p gf->visited_hash_set.size
p model.a->nb
p model.b->nb
p model.a->flags
```

---

## 14 待验证点

1. `p *gf` 打印完整 `ggml_cgraph`，确认 `size`、`visited_hash_set.size`、`use_counts`、`order`。
2. 在 `ggml_new_object` 中观察 `obj->offs`，确认三个 tensor 在 `model.buf` 中的相对偏移（应为 16 字节对齐）。
3. 观察 `ggml_build_forward_impl` 中 `use_counts` 的填充：`a`、`b` 各 1 次，`result` 0 次。
4. 把 `GGML_DEFAULT_GRAPH_SIZE` 改成 16，验证 `buf_size` 与 `ggml_new_graph` 的容量约束。
5. 加 `ggml_set_input/output`，观察 `flags` 变化以及后续 scheduler/gallocr 行为差异。
6. 构造两个输出、共享 `a`，验证哈希去重与 `leafs[]` 不重复。
