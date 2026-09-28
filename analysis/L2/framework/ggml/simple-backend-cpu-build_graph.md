# `simple-backend`：`build_graph()` 分块源码解析

> 源码基线：`353b63b439f27ab2cc19dac97ab1681ba6d2d084`
>
> 示例：`examples/simple/simple-backend.cpp`
>
> 本文只分析 `build_graph()` 及其直接调用链。`init_model()`、backend 注册、scheduler 初始化和具体 backend 执行不在本文展开；文中只在内存边界分析中说明它们与 `build_graph()` 的分工。

---

## 1. 先看整体：`build_graph()` 做了什么

```cpp
struct ggml_cgraph * build_graph(simple_model& model) {
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    struct ggml_tensor * result = ggml_mul_mat(ctx, model.a, model.b);

    ggml_build_forward_expand(gf, result);

    ggml_free(ctx);
    return gf;
}
```

这不是一个“执行函数”，而是一个**计算图描述构建函数**。可以分成六个功能块：

```text
A. 准备构图资源和 context 内存池
B. 创建计算图容器及其管理数组
C. 创建并初始化输入 Tensor 元数据
D. 创建 MUL_MAT 算子表达式
E. 遍历依赖并登记 nodes[] / leafs[]
F. 释放 context 管理状态并返回仍由 model.buf 承载的图
```

完整调用链：

```text
build_graph()
 ├─ ggml_init()
 ├─ ggml_new_graph()
 │   └─ ggml_new_graph_custom()
 │       └─ ggml_new_object()
 ├─ ggml_new_tensor_2d(a)
 │   └─ ggml_new_tensor_impl()
 ├─ ggml_new_tensor_2d(b)
 │   └─ ggml_new_tensor_impl()
 ├─ ggml_mul_mat(a, b)
 └─ ggml_build_forward_expand(result)
     └─ ggml_build_forward_impl()
         └─ ggml_visit_parents_graph()
```

最终得到的不是矩阵乘法结果，而是：

```text
一个 ggml_cgraph
三个 Tensor 元数据对象：a、b、result
一条依赖关系：result = MUL_MAT(a, b)
```

---

# 功能块 A：准备构图资源和 context 内存池

## 2. 预留 `model.buf`

```cpp
size_t buf_size =
    ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
    ggml_graph_overhead();

model.buf.resize(buf_size);
```

`model.buf` 是 `simple_model` 的成员：

```cpp
std::vector<uint8_t> buf;
```

它被作为外部内存池传给 GGML：

```cpp
struct ggml_init_params params0 = {
    .mem_size   = buf_size,
    .mem_buffer = model.buf.data(),
    .no_alloc   = true,
};
```

因此 `ggml_init()` 在这里不是为 Tensor 数据创建独立 backend buffer，而是在 `model.buf` 上建立 GGML context 和后续对象。

这里的容量估算包含两部分：

```text
ggml_tensor_overhead() × GGML_DEFAULT_GRAPH_SIZE
    预留若干 Tensor 对象的元数据空间

ggml_graph_overhead()
    预留 cgraph、nodes、leafs、hash 等图管理空间
```

这是一个按默认图容量进行的保守预留；实际当前示例只创建 `a`、`b`、`result` 三个 Tensor。

## 3. `no_alloc = true` 的关键含义

```cpp
.no_alloc = true
```

它只影响 Tensor 的**实际元素数据**是否在 context 内存池中分配，不影响 Tensor 元数据的创建。

因此本阶段会创建：

```text
ggml_context 管理状态
 ggml_cgraph 及其管理数组
 a、b、result 的 ggml_tensor 元数据
```

本阶段不会创建：

```text
a 的 F32 元素数据
b 的 F32 元素数据
result 的计算数据
CPU/CUDA/Metal backend 数据 buffer
```

后续仍可能计算 `data_size`，但“计算所需大小”和“实际申请数据空间”是两件事。

## 4. 本功能块的断点观察

在 `build_graph()` 和 `ggml_init()` 附近重点观察：

```text
buf_size
model.buf.data()
params0.mem_size
params0.mem_buffer
params0.no_alloc
ctx
```

应建立如下边界认识：

```text
model.buf：构图元数据的宿主内存
context：在这块内存上管理 GGML 对象
no_alloc=true：禁止 context 为 Tensor 元素数据追加空间
```

---

# 功能块 B：创建计算图容器及管理数组

## 5. `ggml_new_graph()` 只是公共包装

```cpp
struct ggml_cgraph * gf = ggml_new_graph(ctx);
```

最终调用：

```cpp
ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
```

当前源码中：

```cpp
GGML_DEFAULT_GRAPH_SIZE = 2048
```

参数含义：

```text
size = 2048：nodes[] 和 leafs[] 的容量

grads = false：不创建 grads[] 和 grad_accs[] 梯度数组
```

## 6. graph object 的一次性内存布局

`ggml_new_graph_custom()` 先计算：

```cpp
const size_t obj_size = ggml_graph_nbytes(size, grads);
```

再调用：

```cpp
struct ggml_object * obj =
    ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
```

然后用 `incr_ptr_aligned()` 在同一块连续内存中依次切出：

```text
ggml_object 管理头
    └─ ggml_cgraph
        ├─ nodes[size]
        ├─ leafs[size]
        ├─ use_counts[hash_size]
        ├─ hash_keys[hash_size]
        ├─ grads[hash_size]       // grads=true 才有
        ├─ grad_accs[hash_size]   // grads=true 才有
        └─ hash_used[bitset_size]
```

这一步是**图管理内存布局**，不是 Tensor 数据内存分配。

### `incr_ptr_aligned()` 的作用

```cpp
static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}
```

它不调用 `malloc()`，只完成：

```text
将当前游标对齐
返回对齐后的区域
把游标向后移动 size 字节
```

`ggml_graph_nbytes()` 和 `ggml_new_graph_custom()` 使用相同的切分顺序，最后通过断言验证估算大小和实际布局一致。

## 7. `ggml_new_object()` 的角色

`ggml_new_object()` 是 context 内部的顺序追加式分配器。它把新对象追加到 `ctx->mem_buffer` 的末尾，并建立 `ggml_object` 链表：

```cpp
struct ggml_object {
    size_t offs;
    size_t size;
    struct ggml_object * next;
    enum ggml_object_type type;
    char padding[4];
};
```

在当前功能块中，对象类型为：

```cpp
GGML_OBJECT_TYPE_GRAPH
```

之后创建 Tensor 时则使用：

```cpp
GGML_OBJECT_TYPE_TENSOR
```

因此 context 内部更像一个顺序追加的对象池，而不是每个图数组单独进行动态分配。

## 8. 图容器创建后的状态

初始化完成后：

```text
cgraph->size       = 2048
cgraph->n_nodes    = 0
cgraph->n_leafs    = 0
cgraph->nodes      = 已分配的节点数组
cgraph->leafs      = 已分配的叶子数组
cgraph->use_counts = 已分配的引用计数数组
cgraph->hash_table = 已分配的 Tensor 访问哈希表
```

此时数组只有容量，还没有写入 `a`、`b`、`result`。

## 9. 本功能块要回答的内存问题

### 是否发生了内存调度？

发生的是**context 内部的顺序空间切分**，不是 backend 内存调度。

### 是否发生了内存复用？

没有。这里是一次性预留图管理数组，不存在计算期 free block 复用。

### `nodes[]` / `leafs[]` 什么时候填充？

本函数只分配数组；具体 Tensor 在功能块 E 的 `ggml_visit_parents_graph()` 中写入。

---

# 功能块 C：创建并初始化 Tensor 元数据

## 10. `ggml_new_tensor_2d()` 的调用链

```cpp
model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
```

调用链：

```text
ggml_new_tensor_2d()
    └─ ggml_new_tensor()
        └─ ggml_new_tensor_impl()
            ├─ 校验 type / n_dims
            ├─ 计算 data_size
            ├─ 处理 view
            ├─ ggml_new_object(TENSOR)
            ├─ 初始化 struct ggml_tensor
            └─ 计算 ne[] / nb[]
```

## 11. 维度和物理步长初始化

当前示例：

```cpp
rows_A = 4; cols_A = 2;
rows_B = 3; cols_B = 2;
```

调用参数是：

```text
model.a：GGML_TYPE_F32, ne0=2, ne1=4
model.b：GGML_TYPE_F32, ne0=2, ne1=3
```

初始化后的逻辑维度：

```text
model.a->ne = { 2, 4, 1, 1 }
model.b->ne = { 2, 3, 1, 1 }
```

GGML 中 `ne[0]` 是最连续维度；不要直接把它当作传统 C 数组的第一维行下标。

`nb[]` 是物理布局步长。对 F32 的 `model.a`，通常为：

```text
nb[0] = 4
nb[1] = 8
nb[2] = 32
nb[3] = 32
```

其中：

```cpp
nb[0] = ggml_type_size(type);
nb[1] = nb[0] * (ne[0] / ggml_blck_size(type));
nb[i] = nb[i - 1] * ne[i - 1];
```

对量化类型，`ggml_blck_size(type)` 会影响第一维物理步长，所以不能把所有类型都简化为 `元素数 × sizeof(element)`。

## 12. `data_size` 与 `obj_alloc_size` 的区别

`ggml_new_tensor_impl()` 会先计算：

```cpp
size_t data_size = ggml_row_size(type, ne[0]);
for (int i = 1; i < n_dims; i++) {
    data_size *= ne[i];
}
```

但是否真的把这段数据空间并入 context object，取决于：

```cpp
if (view_src == NULL && !ctx->no_alloc) {
    obj_alloc_size = data_size;
}
```

当前：

```text
ctx->no_alloc = true
obj_alloc_size = 0
```

所以 `ggml_new_object()` 的大小为：

```text
GGML_TENSOR_SIZE
```

而不是：

```text
GGML_TENSOR_SIZE + data_size
```

这解释了为什么构图结束时通常可以观察到：

```text
model.a->data   = NULL
model.a->buffer = NULL
model.b->data   = NULL
model.b->buffer = NULL
```

这里的 `NULL` 是当前阶段的状态，不表示 Tensor 形状或类型没有初始化，也不表示后续永远没有数据。

## 13. Tensor 初始化字段的含义

`ggml_new_tensor_impl()` 初始化的关键字段包括：

```text
type       数据类型
buffer     backend buffer，初始为 NULL
ne[]       逻辑形状
nb[]       物理字节步长
op         初始为 GGML_OP_NONE
src[]      初始为空
view_src   view 的基础 Tensor
view_offs  view 的累计偏移
data       context 数据地址、view 地址或 NULL
name       初始为空
extra      backend 扩展信息，初始为 NULL
```

### 普通 Tensor

```text
view_src = NULL
view_offs = 0
```

在当前 `no_alloc=true` 下：

```text
data = NULL
buffer = NULL
```

### View Tensor

如果有源 Tensor：

```cpp
data = view_src->data + view_offs;
```

多层 view 会把偏移累加到最底层源 Tensor。View 共享已有存储，不复制元素数据；但如果源 Tensor 当前还没有 `data`，view 的 `data` 也可能暂时为 `NULL`。

## 14. 本功能块的调试重点

在 `ggml_new_tensor_impl()` 断点观察：

```text
type
n_dims
ne
data_size
view_src
view_offs
obj_alloc_size
obj_new
result->type
result->ne
result->nb
result->op
result->data
result->buffer
ctx->n_objects
```

应区分：

```text
Tensor 初始化 = 创建并填充元数据
Tensor 数据分配 = 给 data 提供元素存储
Tensor backend 绑定 = 给 buffer 设置后端 buffer
```

这三者不是同一个动作。

---

# 功能块 D：构建 `MUL_MAT` 算子表达式

## 15. `ggml_mul_mat()` 做什么

```cpp
struct ggml_tensor * result =
    ggml_mul_mat(ctx, model.a, model.b);
```

源码核心逻辑：

```cpp
GGML_ASSERT(ggml_can_mul_mat(a, b));
GGML_ASSERT(!ggml_is_transposed(a));

const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

result->op     = GGML_OP_MUL_MAT;
result->src[0] = a;
result->src[1] = b;
```

该函数完成四件事：

```text
校验矩阵乘法的形状约束
根据输入形状推导输出形状
创建结果 Tensor 元数据
记录 op 和 src[] 依赖
```

它不做：

```text
读取 matrix_A / matrix_B
执行 CPU/CUDA/Metal 矩阵乘法
生成 result 的实际元素
```

## 16. 当前结果 Tensor

当前输入形状：

```text
A：逻辑 4 × 2
B：逻辑 3 × 2
```

表达式为：

```text
result = A × Bᵀ
```

结果逻辑形状为：

```text
4 × 3
```

GGML 的 `ne[]` 通常表示为：

```text
result->ne = { 3, 4, 1, 1 }
```

依赖关系为：

```text
result->op     = GGML_OP_MUL_MAT
result->src[0] = model.a
result->src[1] = model.b
```

这是一个**表达式节点**，不是已经计算完成的结果。

## 17. 本功能块的调试问题

### 算子结果是否已经有数据？

没有。当前 `no_alloc=true`，结果 Tensor 也只是元数据。

### `src[]` 是否等于数据拷贝？

不是。`src[]` 只是 Tensor 指针依赖关系：

```text
result 依赖 a 和 b
```

### `op` 是否表示已经执行？

不是。`op` 描述后续需要执行的算子类型。

---

# 功能块 E：遍历依赖并登记图节点

## 18. 前向构图入口

```cpp
ggml_build_forward_expand(gf, result);
```

调用链：

```text
ggml_build_forward_expand(gf, result)
    └─ ggml_build_forward_impl(gf, result, true, true)
        └─ ggml_visit_parents_graph(gf, result, true)
```

参数含义：

```text
expand=true
    追加到现有图，不清空已有图

compute=true
    对需要执行的算子设置 GGML_TENSOR_FLAG_COMPUTE
```

## 19. `ggml_build_forward_impl()` 的职责

```cpp
const int n_old = cgraph->n_nodes;
ggml_visit_parents_graph(cgraph, tensor, compute);
const int n_new = cgraph->n_nodes - n_old;
```

它是流程控制层，负责：

```text
处理 expand
记录构图前 node 数量
调用递归遍历
计算新增计算节点数
检查目标 Tensor 是否是最后加入的 node
```

`n_new` 只统计新增的 `nodes[]`，不统计 `leafs[]`。当前示例访问三个 Tensor，但新增计算 node 只有一个。

## 20. `ggml_visit_parents_graph()` 的递归顺序

函数首先为当前 Tensor 找到哈希槽：

```cpp
const size_t node_hash_pos =
    ggml_hash_find(&cgraph->visited_hash_set, node);
```

首次访问时登记：

```cpp
cgraph->visited_hash_set.keys[node_hash_pos] = node;
ggml_bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
cgraph->use_counts[node_hash_pos] = 0;
```

然后递归访问 `src[]`：

```cpp
for (int i = 0; i < GGML_MAX_SRC; ++i) {
    struct ggml_tensor * src = node->src[k];
    if (src) {
        const size_t src_hash_pos =
            ggml_visit_parents_graph(cgraph, src, compute);
        cgraph->use_counts[src_hash_pos]++;
    }
}
```

最后才把当前 Tensor 分类写入数组。因此是后序遍历：

```text
result
 ├─ 先访问 a → 写入 leafs[]
 ├─ 再访问 b → 写入 leafs[]
 └─ 最后 result → 写入 nodes[]
```

## 21. Tensor 放入哈希表的收益

这里的哈希表保存的是 Tensor 元数据指针，不是元素数据。

### 去重

共享输入只会进入 `nodes[]` 或 `leafs[]` 一次：

```text
第一次访问：登记 key、设置 used
再次访问：发现 used，直接返回 hash position
```

### 快速查找

`ggml_hash()` 使用 Tensor 指针地址：

```cpp
return (size_t)(uintptr_t)p >> 4;
```

哈希冲突通过线性探测处理。这样不需要扫描整个 `nodes[] + leafs[]`。

### 统一索引

同一个哈希位置可以关联：

```text
keys[hash_pos]
used[hash_pos]
use_counts[hash_pos]
grads[hash_pos]
grad_accs[hash_pos]
```

这比只用 `nodes[]` 下标更通用，因为普通输入通常位于 `leafs[]`。

### 引用计数

访问 `src` 返回其哈希槽后：

```cpp
cgraph->use_counts[src_hash_pos]++;
```

当前示例的逻辑关系是：

```text
use_counts[model.a] = 1
use_counts[model.b] = 1
use_counts[result]  = 0
```

## 22. `leafs[]` 和 `nodes[]` 的真正填充入口

真正写入数组的位置都在 `ggml_visit_parents_graph()`：

```cpp
if (node->op == GGML_OP_NONE &&
    !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
    cgraph->leafs[cgraph->n_leafs] = node;
    cgraph->n_leafs++;
} else {
    cgraph->nodes[cgraph->n_nodes] = node;
    cgraph->n_nodes++;
}
```

分类规则：

```text
无算子且不是训练参数 → leafs[]
其他情况             → nodes[]
```

当前示例：

```text
model.a、model.b → leafs[]
result            → nodes[]
```

要特别区分两个阶段：

```text
ggml_new_graph_custom()
    只分配 nodes[] / leafs[] 的容量

ggml_visit_parents_graph()
    才填入具体 Tensor 指针
```

## 23. `COMPUTE` 标志的作用

递归开始时：

```cpp
if (node->op != GGML_OP_NONE && compute) {
    node->flags |= GGML_TENSOR_FLAG_COMPUTE;
}
```

当前 `compute=true`，所以：

```text
result->flags 包含 GGML_TENSOR_FLAG_COMPUTE
```

这表示该算子节点应参与后续计算图执行；它仍不等于“此刻已经执行”。

## 24. 本功能块的内存和调度边界

本阶段使用：

```text
visited_hash_set
nodes[]
leafs[]
use_counts[]
Tensor 的 src[] 指针
```

这些都是图调度元数据。

本阶段不负责：

```text
backend Tensor buffer 分配
Tensor 在 CPU/GPU 间迁移
跨 backend copy
free block 复用
算子执行顺序的 backend 实现
```

这些属于后续 scheduler / allocator / backend compute 阶段。

---

# 功能块 F：返回图与生命周期边界

## 25. `ggml_free(ctx)` 的主要作用

```cpp
ggml_free(ctx);
return gf;
```

`ggml_free()` 负责释放：

```text
ggml_context 管理对象
以及在 context 自己拥有 mem_buffer 时释放该内存池
```

源码逻辑：

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

关键不是函数名本身，而是：

```cpp
ctx->mem_buffer_owned
```

它决定 `ggml_free()` 是否释放 context 的底层内存池。

## 26. 当前示例为什么释放 context 后仍能计算

当前 `build_graph()` 传入的是外部 buffer：

```cpp
struct ggml_init_params params0 = {
    /*.mem_size   =*/ buf_size,
    /*.mem_buffer =*/ model.buf.data(),
    /*.no_alloc   =*/ true,
};
```

`ggml_init()` 中相应设置：

```cpp
.mem_buffer_owned = params.mem_buffer ? false : true
```

因此当前状态是：

```text
ctx->mem_buffer       = model.buf.data()
ctx->mem_buffer_owned = false
```

执行：

```cpp
ggml_free(ctx);
```

实际只释放：

```text
ggml_context 管理对象
```

不会释放：

```text
model.buf
cgraph
model.a 的 Tensor 元数据
model.b 的 Tensor 元数据
result 的 Tensor 元数据
```

内存关系为：

```text
model
 └─ model.buf
     ├─ ggml_cgraph
     ├─ model.a
     ├─ model.b
     └─ result

ctx
 └─ 只负责在 model.buf 上创建和管理这些对象
```

因此 `ctx` 可以理解为构图阶段的“工厂和管理器”，而 `gf`、Tensor 元数据是构图产物。构图完成后不再需要 `ctx`，但产物仍然由 `model.buf` 承载。

## 27. 两种 context 内存所有权场景

### 外部提供 `mem_buffer`

当前示例：

```cpp
.mem_buffer = model.buf.data()
```

结果：

```text
mem_buffer_owned = false
ggml_free(ctx) 不释放 model.buf
```

### 由 GGML 内部分配 `mem_buffer`

如果：

```cpp
.mem_buffer = NULL
```

则 `ggml_init()` 会自己分配对齐内存：

```text
mem_buffer_owned = true
```

此时：

```cpp
ggml_free(ctx);
```

会同时释放：

```text
ggml_context
ctx->mem_buffer
```

如果 `cgraph` 和 Tensor 元数据位于这块内部 buffer 中，它们也会随之失效。

## 28. `ggml_free(ctx)` 后哪些对象有效

当前外部 buffer 模式下，只要 `model` 和 `model.buf` 仍然存活：

```text
gf
model.a
model.b
result
gf->nodes
gf->leafs
gf->visited_hash_set
```

仍然指向有效的 `model.buf` 区域。

但以下对象已经失效，不能继续访问：

```text
ctx
ctx->objects_begin
ctx->objects_end
ctx->mem_size
ctx->no_alloc
```

计算阶段不需要 `ctx`，因为 `compute()` 使用的是：

```text
ggml_cgraph
Tensor 元数据
model.sched
backend buffer
```

而不是 context 的对象链表。

## 29. 生命周期的关键前提

```text
model.buf 不能在 gf 使用期间 resize
model 必须保持存活
```

如果：

```cpp
model.buf.resize(new_size);
```

导致 `std::vector` 更换底层地址，则：

```text
gf、model.a、model.b、result
```

都会变成悬空指针，即使 `ctx` 之前已经正确释放也一样。

因此当前流程是：

```text
build_graph()
 ├─ model.buf 提供稳定存储
 ├─ ggml_init() 创建 ctx
 ├─ 创建 graph 和 Tensor
 ├─ 构建依赖
 ├─ ggml_free(ctx)   // 释放 context 管理状态
 └─ 返回仍位于 model.buf 中的 gf

compute()
 ├─ 使用 gf 和 Tensor 元数据
 ├─ 分配/绑定 backend 数据
 └─ 执行计算
```

---

# 专题：`ggml_tensor` 结构体布局与字段关系

`build_graph()` 直接创建和使用 `ggml_tensor`，因此需要把 Tensor 元数据本身与后端数据存储区分开。

在当前 64 位环境中，断点可能显示：

```text
sizeof(struct ggml_tensor) = 336
alignof(struct ggml_tensor) = 8
隐式 padding = 4
```

相关常量：

```cpp
#define GGML_MAX_DIMS        4
#define GGML_MAX_SRC         10
#define GGML_MAX_OP_PARAMS   64
#define GGML_MAX_NAME        64
```

典型布局如下：

| 字段 | 偏移 | 大小 |
|---|---:|---:|
| `type` | 0 | 4 |
| 隐式 padding | 4 | 4 |
| `buffer` | 8 | 8 |
| `ne[4]` | 16 | 32 |
| `nb[4]` | 48 | 32 |
| `op` | 80 | 4 |
| `op_params[16]` | 84 | 64 |
| `flags` | 148 | 4 |
| `src[10]` | 152 | 80 |
| `view_src` | 232 | 8 |
| `view_offs` | 240 | 8 |
| `data` | 248 | 8 |
| `name[64]` | 256 | 64 |
| `extra` | 320 | 8 |
| `padding[8]` | 328 | 8 |
| 结构体结束 | 336 | — |

计算过程：

```text
显式字段大小合计 = 332 字节
type 后为 buffer 对齐插入 = 4 字节隐式 padding
────────────────────────────
sizeof(ggml_tensor) = 336 字节
```

结构体整体对齐为 8，是因为其中的指针、`size_t` 和 `int64_t` 在当前 64 位 ABI 下通常要求 8 字节对齐：

```text
336 % 8 == 0
```

`char padding[8]` 是源码中明确声明的成员，已经包含在 336 字节内；它不是调试器报告的 4 字节隐式 padding。两者需要区分：

```text
隐式 padding：编译器为字段对齐自动插入
padding[8]：源码显式声明的尾部布局预留
```

更准确地说，不能仅凭 `padding[8]` 断言它专门为某种 SIMD 指令或 backend 设计；Tensor 对象起始地址的对齐还由 `GGML_MEM_ALIGN` 和 context allocator 保证。

## 31. `buffer`、`data` 与 Tensor 实际数据

```cpp
struct ggml_backend_buffer * buffer;
void * data;
```

二者不是同一个概念：

```text
buffer
    管理 Tensor 数据所在的 backend 存储区域

data
    当前 Tensor 在该存储区域中的具体起始地址
```

可以抽象为：

```text
一个 backend buffer
    ├─ Tensor A：data = buffer_base + offset_A
    ├─ Tensor B：data = buffer_base + offset_B
    └─ Tensor C：data = buffer_base + offset_C
```

所以多个 Tensor 可能共享同一个 `buffer`，但拥有不同的 `data` 偏移。

在当前 `build_graph()` 阶段，由于：

```cpp
.no_alloc = true
```

普通 Tensor 通常为：

```text
data   = NULL
buffer = NULL
```

这表示 Tensor 元数据已创建，但实际 backend 存储尚未绑定；它不表示 Tensor 初始化失败。

## 32. `nb[]` 的作用

```cpp
size_t nb[GGML_MAX_DIMS];
```

`nb[]` 是各维度的物理字节步长，描述沿某个维度移动一个逻辑位置时需要跨过多少字节：

```text
地址 = data
      + i0 * nb[0]
      + i1 * nb[1]
      + i2 * nb[2]
      + i3 * nb[3]
```

它解决的问题包括：

```text
支持非连续 Tensor
支持 view、切片、转置和重排
描述量化 block 的物理布局
让不同 backend 使用统一的布局信息
```

例如 `model.a` 的 F32 布局：

```text
ne = { 2, 4, 1, 1 }
nb = { 4, 8, 32, 32 }
```

`ne[]` 是逻辑形状，`nb[]` 是物理步长，二者不能混为一谈。

## 33. `view_src` 和 `view_offs`

```cpp
struct ggml_tensor * view_src;
size_t view_offs;
```

它们描述当前 Tensor 是否是另一个 Tensor 的数据视图：

```text
view_src
    源 Tensor

view_offs
    相对于源 Tensor data 起点的字节偏移
```

View 不复制数据，而是共享源 Tensor 存储：

```cpp
view->data = view_src->data + view_offs;
```

多级 view 会将偏移折叠到最底层源 Tensor：

```text
base
 └─ view1(offset=32)
     └─ view2(offset=64)

view2.view_src  = base
view2.view_offs = 96
```

创建 View 时还会检查：

```text
view_offs + view data_size <= ggml_nbytes(view_src)
```

典型应用包括：

```text
子矩阵和切片
共享存储的形状重解释
带步长的非连续区域
避免不必要的数据复制
```

---

# 36. 专题分析一：`build_graph()` 到底涉及哪些内存

## 26.1 当前阶段确实使用的内存

```text
A. context 对象管理内存
   ggml_context、ggml_object 链表信息

B. graph 管理内存
   ggml_cgraph
   nodes[] / leafs[]
   use_counts[]
   hash_keys[] / hash_used[]

C. Tensor 元数据内存
   model.a、model.b、result 的 struct ggml_tensor

D. 指针关系内存
   result->src[]
   cgraph->nodes[] / leafs[] 中的 Tensor 指针
```

## 26.2 当前阶段不申请的内存

```text
A. Tensor 元素数据
B. backend buffer
C. GPU/Metal/CUDA 设备存储
D. Tensor 副本和跨 backend copy buffer
E. 执行期工作区
```

`data_size` 被计算并不代表数据存储已经申请；`no_alloc=true` 使 `obj_alloc_size` 保持为 0。

---

# 37. 专题分析二：内存调度、内存复用发生在哪里

## 27.1 `build_graph()` 阶段的“调度”是什么

这里的调度主要是：

```text
组织 context 内存布局
组织图节点和依赖元数据
为 Tensor 建立图内索引
```

不是：

```text
将 Tensor 分配到 CPU/GPU
规划 backend buffer 地址
安排跨设备传输
```

## 27.2 `build_graph()` 阶段是否发生内存复用

严格来说没有发生计算内存复用。

`ggml_new_object()` 是 context 内部的顺序追加分配：

```text
旧对象末尾 → 新对象
```

`incr_ptr_aligned()` 只是同一块 graph object 的切片器：

```text
cgraph → nodes → leafs → use_counts → hash
```

它们都不是基于 Tensor 生命周期释放后再复用的 free-block allocator。

## 27.3 Tensor 数据内存的规划和复用属于后续阶段

后续调用：

```cpp
ggml_backend_sched_alloc_graph(model.sched, gf);
```

才会针对具体计算图处理 backend Tensor 存储、地址规划和可能的 buffer 复用。执行阶段还可能根据节点生命周期和后端能力复用临时空间。

因此不能把以下机制归入 `build_graph()`：

```text
free block 复用
Tensor 数据 offset 规划
backend buffer 绑定
跨 backend Tensor copy
执行期工作区
```

`build_graph()` 提供的是后续 allocator 所需的图结构和 Tensor 元数据。

---

# 38. 专题分析三：Tensor 初始化问题

## 28.1 普通 Tensor 的初始化

当前 `a`、`b` 的初始化顺序是：

```text
确定 type 和 ne[]
计算 data_size
创建 ggml_tensor object
初始化 op/src/view/data/buffer
填写 ne[]
计算 nb[]
```

在 `no_alloc=true` 下：

```text
type、ne、nb 已有效
op = GGML_OP_NONE
src[] 为空
buffer = NULL
data = NULL
```

## 28.2 View Tensor 的初始化

View 记录：

```text
view_src
view_offs
```

并把数据地址解释为：

```text
view_src->data + view_offs
```

它共享源 Tensor 的存储，不是复制数据。多级 view 的偏移会被折叠到最底层源 Tensor。

## 28.3 `ne[]`、`nb[]`、`data`、`buffer` 的区别

```text
ne[]
    逻辑维度和元素数量

nb[]
    物理存储步长

data
    元素数据地址，当前可能为 NULL

buffer
    GGML backend buffer 对象，当前可能为 NULL
```

因此调试时不能因为 `data=NULL` 就认为 Tensor 没有初始化；它可能已经完成了完整的形状、类型和布局初始化，只是尚未绑定实际存储。

## 28.4 算子结果 Tensor 的初始化

`result` 先通过通用 Tensor 创建逻辑建立元数据，再由 `ggml_mul_mat()` 设置：

```text
op = GGML_OP_MUL_MAT
src[0] = a
src[1] = b
```

它仍然没有计算数据。`op + src[]` 是表达式描述，不是执行结果。

---

# 39. 专题分析四：图拓扑、节点登记和哈希表

## 29.1 为什么需要哈希表

哈希表解决的是：

```text
Tensor 元数据去重
Tensor → hash position 的映射
use_counts / grads / grad_accs 的统一索引
```

它不解决：

```text
Tensor 元素数据存储
backend buffer 分配
内存复用
```

## 29.2 为什么采用后序遍历

递归先访问 `src[]`，再登记当前节点：

```text
输入/依赖先登记
当前算子后登记
```

这让 `nodes[]` 的新增顺序天然具有依赖优先特征。不过后续 scheduler 仍可能根据 backend、copy 和 split 重新组织执行计划，不能把 `nodes[]` 顺序简单等同于最终所有设备上的执行序列。

## 29.3 `nodes[]` 和 `leafs[]` 的入口总结

```text
外部入口：
    ggml_build_forward_expand()

流程控制：
    ggml_build_forward_impl()

递归和实际写入：
    ggml_visit_parents_graph()

分类：
    op == NONE 且非 PARAM → leafs[]
    其他                 → nodes[]
```

---

# 40. 当前示例构图后的状态

```text
Tensor 元数据：
    model.a
    model.b
    result

依赖关系：
    result->src[0] = model.a
    result->src[1] = model.b

哈希集合：
    model.a
    model.b
    result

leafs[]：
    model.a
    model.b

nodes[]：
    result

use_counts[]：
    model.a  = 1
    model.b  = 1
    result   = 0

result->op：
    GGML_OP_MUL_MAT

result->flags：
    包含 GGML_TENSOR_FLAG_COMPUTE

result->data / result->buffer：
    当前尚未提供 backend 实际数据存储
```

---

# 41. 调试变量清单：按功能块观察

## A：内存池

```text
buf_size
model.buf.data()
params0.mem_size
params0.mem_buffer
params0.no_alloc
ctx
```

## B：图容器

```text
obj_size
hash_size
cgraph
cgraph->size
cgraph->n_nodes
cgraph->n_leafs
cgraph->nodes
cgraph->leafs
cgraph->use_counts
cgraph->visited_hash_set
```

## C：Tensor 初始化

```text
type
n_dims
ne
data_size
obj_alloc_size
view_src
view_offs
result->type
result->ne
result->nb
result->op
result->data
result->buffer
```

## D：算子表达式

```text
a->ne
b->ne
result->ne
result->op
result->src[0]
result->src[1]
```

## E：图遍历

```text
expand
compute
node
node->op
node->src[0]
node->src[1]
node_hash_pos
used[node_hash_pos]
keys[node_hash_pos]
use_counts[node_hash_pos]
n_old
n_new
```

建议验证：

```text
第一次访问 Tensor：used 从 0 变为 1
重复访问 Tensor：不重复写入 nodes[] / leafs[]
model.a、model.b：进入 leafs[]
result：进入 nodes[]
result->flags：包含 COMPUTE
```

---

# 42. 常见误区

### 误区一：进入哈希表就分配了 Tensor 数据

错误。哈希表只保存 Tensor 元数据指针。

### 误区二：`ggml_new_graph_custom()` 已经填充了节点

错误。它只分配 `nodes[]`、`leafs[]` 等数组；具体填充在 `ggml_visit_parents_graph()`。

### 误区三：`ggml_mul_mat()` 已经执行矩阵乘法

错误。它只设置 `op` 和 `src[]`。

### 误区四：`data=NULL` 表示 Tensor 初始化失败

错误。在 `no_alloc=true` 下，这是尚未分配 Tensor 实际数据的正常状态。

### 误区五：`build_graph()` 已经完成内存复用

错误。它只组织 context 和图元数据；backend 数据 buffer 的规划和复用属于后续 allocator/scheduler 阶段。

### 误区六：`n_new` 是访问到的 Tensor 总数

错误。`n_new` 只统计新增的 `nodes[]` 计算节点，不包括 `leafs[]`。

---

# 43. 源码位置索引

| 功能 | 源码位置 |
|---|---|
| `build_graph()` | `examples/simple/simple-backend.cpp:68-95` |
| `ggml_new_object()` | `src/ggml.c:1708-1764` |
| `ggml_new_tensor_impl()` | `src/ggml.c:1766-1842` |
| `ggml_mul_mat()` | `src/ggml.c:3333-3356` |
| `ggml_visit_parents_graph()` | `src/ggml.c:7236-7302` |
| `ggml_build_forward_impl()` | `src/ggml.c:7304-7321` |
| `ggml_build_forward_expand()` | `src/ggml.c:7337-7339` |
| `incr_ptr_aligned()` | `src/ggml.c:7444-7449` |
| `ggml_graph_nbytes()` | `src/ggml.c:7451-7467` |
| `ggml_new_graph_custom()` | `src/ggml.c:7477-7520` |
| hash set 结构和实现 | `src/ggml-impl.h:233-331` |
| `GGML_DEFAULT_GRAPH_SIZE` | `include/ggml.h:233` |
| `struct ggml_tensor` | `include/ggml.h:685-717` |

---

# 44. 总结：用一张图理解 `build_graph()`

```text
功能块 A：资源准备
    model.buf + ggml_init(no_alloc=true)
        ↓
功能块 B：图容器
    cgraph + nodes[] + leafs[] + hash + use_counts[]
        ↓
功能块 C：Tensor 元数据
    a、b：type/ne/nb 已初始化，data/buffer 暂为空
        ↓
功能块 D：算子表达式
    result：op=MUL_MAT，src[0]=a，src[1]=b
        ↓
功能块 E：依赖遍历
    哈希去重、引用计数、COMPUTE 标志、leaf/node 分类
        ↓
功能块 F：返回
    ggml_free(ctx)，但 model.buf 继续承载图和 Tensor 元数据
```

最终结论：

> `build_graph()` 的本质是把 Tensor 元数据组织成一个可执行的计算图描述。它负责 context 元数据布局、Tensor 初始化、算子依赖表达和图节点登记；它不负责 Tensor 实际数据分配、backend 内存调度、计算期内存复用或矩阵乘法执行。后续 scheduler/allocator 阶段会消费这里构建出的 `ggml_cgraph`、Tensor `src[]` 关系和节点元数据，完成真正的 backend 存储规划与执行。
