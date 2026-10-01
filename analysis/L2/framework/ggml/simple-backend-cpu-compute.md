# simple-backend.cpp：compute 之后的 GGML 执行链路

> 本文只分析 `simple-backend.cpp` 从 `compute()` 开始到结果读取、资源释放的代码。重点不是按断点顺序罗列代码，而是按运行时职责划分为：**调度准备、数据装载、split 执行与依赖、结果回读、资源释放**五个功能域。

源码范围：

- `examples/simple/simple-backend.cpp`
- `src/ggml-backend.cpp`
- `include/ggml-backend.h`
- `src/ggml-backend-impl.h`

---

## 1. 先建立整体模型：五个功能域

`compute()` 的代码很短，但背后包含五个职责不同的阶段：

```cpp
struct ggml_tensor * compute(simple_model & model, struct ggml_cgraph * gf) {
    // A. 调度状态和 graph 存储准备
    ggml_backend_sched_reset(model.sched);
    ggml_backend_sched_alloc_graph(model.sched, gf);

    // B. 外部输入装载
    ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));

    // C. split 输入搬运、backend 执行、完成同步
    ggml_backend_sched_graph_compute(model.sched, gf);

    // 返回原始 graph 最后一个 node，即 result
    return ggml_graph_node(gf, -1);
}
```

完整运行时职责：

```text
域 A：Scheduler preparation
    reset → split → allocation

域 B：Input materialization
    host matrix → backend tensor buffer

域 C：Split execution
    split 间依赖 → 跨 backend copy → backend graph compute

域 D：Output materialization
    backend result → host out_data

域 E：Lifetime cleanup
    scheduler → backend
```

最重要的边界：

| 阶段 | 主要函数 | 做什么 | 不做什么 |
|---|---|---|---|
| 调度准备 | `ggml_backend_sched_alloc_graph` | 决定 backend、切 split、准备 buffer | 不执行 `MUL_MAT` |
| 内存分配 | `ggml_backend_sched_alloc_splits` | reserve/alloc tensor 存储 | 不写入 `matrix_A/B` |
| 输入装载 | `ggml_backend_tensor_set` | 把外部数据写入已分配 tensor | 不建图、不执行算子 |
| 计算提交 | `ggml_backend_sched_compute_splits` | 搬运 split input、提交 backend graph | scheduler 不实现算子循环 |
| 完成同步 | `ggml_backend_sched_synchronize` | 等待 backend 异步工作完成 | 不负责读回 host |
| 输出读取 | `ggml_backend_tensor_get` | backend tensor → host | 不重新计算 |

---

## 2. 示例运行前提：本例实际上使用单 copy

`init_model()` 中 scheduler 的创建方式是：

```cpp
model.sched = ggml_backend_sched_new(
    backends,
    nullptr,
    2,
    GGML_DEFAULT_GRAPH_SIZE,
    false,  // parallel = false
    true);
```

`parallel = false` 导致：

```cpp
sched->n_copies = 1;
```

因此当前示例中：

```cpp
sched->cur_copy  = 0;
sched->next_copy = 0;
```

`cur_copy/next_copy` 的 round-robin 逻辑仍然存在，但在本例中退化为单 copy。多 copy、event 和流水线重叠是 `parallel = true` 时才真正体现的机制。

此外，当前 graph 通常非常小：

```text
a ─┐
   ├─ MUL_MAT → result
b ─┘
```

一般是：

```text
gf->n_nodes = 1
gf->n_leafs = 2
```

如果所有节点最终使用同一个 backend，通常：

```text
sched->n_splits = 1
split->n_inputs = 0
```

跨 backend copy、多个 split 和 MoE 分支是 scheduler 为复杂模型保留的通用路径，simple-backend 不一定会触发。

---

# 3. 功能域 A：调度准备与内存落位

## 3.1 重置 scheduler：开始一轮新的 allocation

```cpp
ggml_backend_sched_reset(model.sched);
```

`reset` 的作用是清理上一轮 graph 的调度状态，主要涉及：

```cpp
sched->is_reset
sched->is_alloc
sched->hv_tensor_backend_ids
sched->hv_tensor_copies
```

逻辑结果：

```text
is_reset = true
is_alloc = false
```

它不是销毁 scheduler，也不是立即释放所有 backend buffer。应区分：

```text
reset：重置下一轮 graph 调度元数据
alloc：为当前 graph 建立/复用存储布局
free：最终销毁 scheduler/backend 资源
```

---

## 3.2 allocation 总入口：`ggml_backend_sched_alloc_graph`

```cpp
bool ggml_backend_sched_alloc_graph(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int) sched->hash_set.size >=
                graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy =
        (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;
    return true;
}
```

这个函数不是“执行 graph”，而是把准备工作组合起来：

```text
1. 选择本轮 copy slot
2. split_graph：确定 backend、切分 graph、建立 copy 描述
3. alloc_splits：使用 galloc 为 tensor 分配/复用 buffer
4. is_alloc = true
```

### `split_graph` 产出的核心对象

```cpp
struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    struct ggml_cgraph graph;
};
```

可理解为：

```text
split.backend_id：这段 graph 由哪个 backend 执行
[i_start, i_end)：原 graph 的 node 范围
split.inputs：从其他 backend 获取的原始 source
split.graph：该范围对应的 graph view
```

### 跨 backend copy 只在此阶段“描述”，不在此阶段“搬运”

当 source 无法被目标 backend 直接使用时，scheduler 创建 layout copy：

```cpp
struct ggml_tensor * tensor_copy =
    ggml_dup_tensor_layout(sched->ctx, src);
```

然后：

```cpp
split->inputs[n_inputs] = src;
node->src[j] = tensor_copy;
```

二者职责不同：

```text
split->inputs：运行时要从哪里取数据
node->src[j]：目标 backend 计算时实际读取什么
```

`ggml_dup_tensor_layout` 只复制 tensor 元数据和 layout，不代表真实数据已经复制；真实搬运在 `compute_splits` 中完成。

---

## 3.3 `alloc_splits`：从 graph 描述到实际 buffer

### 3.3.1 方案变化检查

`alloc_splits` 首先比较本轮和上一轮的：

```cpp
sched->node_backend_ids
sched->prev_node_backend_ids
sched->leaf_backend_ids
sched->prev_leaf_backend_ids
```

并检查对应 `bufts` 是否变化：

```cpp
if (current_backend_id != previous_backend_id &&
    current_buft != previous_buft) {
    backend_ids_changed = true;
}
```

这里真正关心的是 buffer type，而不是 backend ID 名字本身。不同 backend 可能使用同一种 buffer type，因此 backend ID 改变并不必然要求重新 reserve。

### 3.3.2 尝试复用已有 allocation

```cpp
if (backend_ids_changed ||
    !ggml_gallocr_alloc_graph(
        sched->galloc,
        &sched->graph)) {
```

如果方案没有变化，先尝试使用已有 reservation 完成 graph allocation。失败的可能原因包括：

- graph size 增大；
- tensor 生命周期或布局变化；
- split input 数量变化；
- 现有 reservation 无法容纳本轮 graph。

### 3.3.3 必要时同步后重新 reserve

如果必须重分配：

```cpp
for (int i = 0; i < sched->n_backends; i++) {
    ggml_backend_synchronize(sched->backends[i]);
}
```

原因是新的 reservation 可能移动 tensor 地址，必须先让旧的异步工作结束。这里直接同步各 backend，而不是调用 scheduler synchronize，是为了不改变多 copy 模式下的 copy 状态。

然后：

```cpp
if (!ggml_gallocr_reserve_n(
        sched->galloc,
        &sched->graph,
        sched->node_backend_ids,
        sched->leaf_backend_ids)) {
    return false;
}

if (!ggml_gallocr_alloc_graph(
        sched->galloc,
        &sched->graph)) {
    return false;
}
```

两者的职责：

```text
reserve_n：重新计算各 backend 需要的容量和复用布局
alloc_graph：按新布局给 tensor 设置实际 buffer/data
```

成功后通常可观察到：

```cpp
tensor->buffer != nullptr
tensor->data   != nullptr
```

但这只代表存储已准备好，不代表：

```text
matrix_A/B 已写入
MUL_MAT 已执行
结果已同步到 CPU
```

### 3.3.4 `alloc_splits` 的输出

```text
true：scheduler graph 的 backend 存储已经准备好
false：reserve 或 allocation 失败
```

外层随后设置：

```cpp
sched->is_alloc = true;
```

---

# 4. 功能域 B：输入数据装载

## 4.1 `ggml_backend_tensor_set` 的职责

```cpp
ggml_backend_tensor_set(
    model.a,
    matrix_A,
    0,
    ggml_nbytes(model.a));

ggml_backend_tensor_set(
    model.b,
    matrix_B,
    0,
    ggml_nbytes(model.b));
```

公共实现：

```cpp
void ggml_backend_tensor_set(
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    GGML_ASSERT(tensor);

    ggml_backend_buffer_t buf =
        tensor->view_src ?
        tensor->view_src->buffer :
        tensor->buffer;

    GGML_ASSERT(buf != nullptr);

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != nullptr);
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    buf->iface.set_tensor(
        buf,
        tensor,
        data,
        offset,
        size);
}
```

功能分解：

```text
1. 找到目标 buffer
2. 处理 view 的底层 buffer
3. 检查 data/size/range
4. 转发到具体 backend 的 set_tensor
```

它不创建 tensor、不建图、不分配 graph、不执行算子。

### 4.1.1 普通 tensor 与 view

```cpp
tensor->view_src == nullptr：使用 tensor->buffer
tensor->view_src != nullptr：使用 tensor->view_src->buffer
```

view 共享底层存储，`tensor->data` 通常是：

```text
view_src->data + view_offs
```

### 4.1.2 buffer 与 data

```text
tensor->buffer：管理对象，描述 backend buffer
 tensor->data：该 tensor 在 buffer 中的实际数据地址
```

最终具体数据搬运由：

```cpp
buf->iface.set_tensor(...)
```

完成。CPU、CUDA、Metal 等 backend 可以实现不同的上传路径。

---

## 4.2 Tensor 大小和布局验证

### `model.a`

```cpp
model.a = ggml_new_tensor_2d(
    ctx,
    GGML_TYPE_F32,
    cols_A,   // 2
    rows_A);  // 4
```

因此：

```text
ne = [2, 4, 1, 1]
nelements = 8
ggml_nbytes(a) = 32 bytes
nb = [4, 8, 32, 32]
```

### `model.b`

```cpp
model.b = ggml_new_tensor_2d(
    ctx,
    GGML_TYPE_F32,
    cols_B,   // 2
    rows_B);  // 3
```

因此：

```text
ne = [2, 3, 1, 1]
nelements = 6
ggml_nbytes(b) = 24 bytes
nb = [4, 8, 24, 24]
```

调试时查看 b 只能读 24 字节：

```gdb
x/6f model.b->data
x/24bx model.b->data
```

不能按 32 字节读取后把越界内容误认为 b 的有效数据。

---

## 4.3 已观察内存的解释

### `model.a->data`

观察到：

```text
00 00 00 40   00 00 00 41   00 00 a0 40   00 00 80 3f
00 00 80 40   00 00 00 40   00 00 00 41   00 00 c0 40
```

按小端 IEEE-754 F32 解码：

```text
2, 8, 5, 1, 4, 2, 8, 6
```

与 `matrix_A` 一致：

```cpp
float matrix_A[rows_A * cols_A] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};
```

对应 GGML 逻辑矩阵：

```text
[ 2, 8 ]
[ 5, 1 ]
[ 4, 2 ]
[ 8, 6 ]
```

第 0 维连续，偏移公式为：

```cpp
offset = i0 * tensor->nb[0] + i1 * tensor->nb[1];
```

### `model.b->data` 的异常观察

正确的 `matrix_B` 是：

```cpp
float matrix_B[rows_B * cols_B] = {
    10, 5,
    9,  9,
    5,  4
};
```

正确的 b 字节应为：

```text
00 00 20 41   00 00 a0 40   00 00 10 41
00 00 10 41   00 00 a0 40   00 00 80 40
```

如果 b dump 出现 a 的 8 个 float，首先检查：

```gdb
p/x model.a->data
p/x model.b->data
p ggml_nbytes(model.a)
p ggml_nbytes(model.b)
x/6f matrix_B
x/6f model.b->data
```

可能原因：

1. 查看的是错误指针；
2. `a->data` 与 `b->data` 意外重叠；
3. 第二次 `set_tensor` 的源指针不是 `matrix_B`；
4. backend 的 `set_tensor` 目标地址计算错误；
5. 读取 b 时超出了 24 字节有效范围。

---

## 4.4 为什么查看 `model.a->buffer` 不是在看数据

`model.a->buffer` 指向：

```cpp
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;
    ggml_backend_buffer_type_t buft;
    void * context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};
```

其中 `iface` 是函数指针表，所以原始 dump 开头出现大量代码地址是正常的。

在 64 位系统中，`iface` 的字段大致是：

```text
0x00  free_buffer
0x08  get_base
0x10  init_tensor
0x18  memset_tensor
0x20  set_tensor
0x28  get_tensor
0x30  set_tensor_2d
0x38  get_tensor_2d
0x40  cpy_tensor
0x48  clear
0x50  reset
0x58  buft
0x60  context
0x68  size
0x70  usage
```

此前 dump 可解释为：

```cpp
buffer->size  = 0x80 = 128 bytes
buffer->usage = 0x02 = GGML_BACKEND_BUFFER_USAGE_COMPUTE
```

`buffer->size` 是整个 backend buffer 容量，不是 `model.a` 的 32 字节数据大小。查看实际 tensor 内容应使用：

```gdb
x/8f model.a->data
```

---

# 5. 功能域 C：split 执行、数据依赖与同步

## 5.1 同步入口：`ggml_backend_sched_graph_compute`

```cpp
enum ggml_status ggml_backend_sched_graph_compute(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph) {
    enum ggml_status err =
        ggml_backend_sched_graph_compute_async(sched, graph);

    ggml_backend_sched_synchronize(sched);

    return err;
}
```

它是同步执行入口：

```text
graph_compute_async：提交计算
synchronize：等待所有 backend
return：返回 async 阶段的状态
```

### async 内部逻辑

```cpp
if (!sched->is_reset && !sched->is_alloc) {
    ggml_backend_sched_reset(sched);
}

if (!sched->is_alloc) {
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        return GGML_STATUS_ALLOC_FAILED;
    }
}

return ggml_backend_sched_compute_splits(sched);
```

simple-backend 前面已经显式 allocation，因此正常情况下这里直接进入 `compute_splits`。

### synchronize 的语义

```cpp
for (int i = 0; i < sched->n_backends; i++) {
    ggml_backend_synchronize(sched->backends[i]);
}
```

函数返回时，计算结果已经完成，可以安全进行 `tensor_get`。但它不会自动把 device memory 复制到 host。

---

## 5.2 `compute_splits` 的执行框架

```cpp
static enum ggml_status ggml_backend_sched_compute_splits(
        ggml_backend_sched_t sched) {
    struct ggml_backend_sched_split * splits = sched->splits;
    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;
    int prev_backend_id = -1;

    for (int split_id = 0;
         split_id < sched->n_splits;
         split_id++) {
        ...
    }

    return GGML_STATUS_SUCCESS;
}
```

按职责可分为：

```text
C1. 选择当前 split/backend
C2. 处理 split 间同步
C3. 处理跨 backend 输入
C4. 提交 split graph 计算
C5. 记录 event 并推进 split 状态
```

MoE 的 `ids/used_ids` 只服务于特殊的 expert 局部 copy，不影响 simple-backend 普通 `MUL_MAT` 路径。

---

## 5.3 C1：选择当前 split/backend

```cpp
struct ggml_backend_sched_split * split =
    &splits[split_id];

int split_backend_id = split->backend_id;
ggml_backend_t split_backend =
    sched->backends[split_backend_id];
```

当前 split 的所有 node 都交给 `split_backend`。

---

## 5.4 C2：split 间同步与 buffer 生命周期

```cpp
if (split->n_inputs == 0 &&
    prev_backend_id >= 0 &&
    prev_backend_id != split_backend_id) {
    if (sched->events[prev_backend_id][sched->cur_copy] != nullptr) {
        ggml_backend_event_synchronize(
            sched->events[prev_backend_id][sched->cur_copy]);
    } else {
        ggml_backend_synchronize(
            sched->backends[prev_backend_id]);
    }
}
```

这段同步不只为数值依赖服务，还为了防止 allocator 复用仍在异步使用的内存：

```text
split0 backend A 仍在写 buffer X
split1 backend B 准备复用 buffer X
    ↓
必须先等待 split0
```

触发条件：

```text
当前 split 没有显式跨 backend input
存在前一个 split
前后 backend 不同
```

有 event 时等待特定 event；没有 event 时同步整个前一 backend。simple-backend 的 `n_copies=1` 通常没有 event，因此多见后者。

---

## 5.5 C3：跨 backend input 的统一数据路径

```cpp
for (int input_id = 0;
     input_id < split->n_inputs;
     input_id++) {
    ggml_backend_t input_backend =
        ggml_backend_sched_get_tensor_backend(
            sched,
            split->inputs[input_id]);

    struct ggml_tensor * input =
        split->inputs[input_id];

    struct ggml_tensor * input_cpy =
        tensor_copy(
            input,
            split_backend_id,
            sched->cur_copy);
```

数据关系：

```text
input：生产者 backend 中的原始 tensor
input_cpy：消费者 split backend 中的目标 tensor
```

split 间依赖的完整链路：

```text
生产 split 产生 input
    ↓
split->inputs 保存 input
    ↓
找到 input_cpy
    ↓
等待或安排跨 backend copy
    ↓
消费者 node->src[j] 已指向 input_cpy
    ↓
执行消费者 split
```

### C3-a：用户输入

```cpp
if (input->flags & GGML_TENSOR_FLAG_INPUT) {
    if (event != nullptr) {
        ggml_backend_event_synchronize(event);
    } else {
        ggml_backend_synchronize(split_backend);
    }

    ggml_backend_tensor_copy(input, input_cpy);
}
```

用户输入可能在下一轮被覆写，因此先等待目标 copy slot 的旧使用完成，再执行 copy。

### C3-b：普通 tensor

```cpp
} else {
    if (event != nullptr) {
        ggml_backend_event_wait(split_backend, event);
    } else {
        ggml_backend_synchronize(split_backend);
    }
```

这是为了保证目标 `input_cpy` 可被覆盖。`event_wait` 通常是让目标 backend 的后续工作等待 event，不必阻塞 CPU。

### C3-c：普通异步 copy 与 fallback

优先：

```cpp
split_backend->iface.cpy_tensor_async(
    input_backend,
    split_backend,
    input,
    input_cpy);
```

如果不支持或返回失败，则：

```text
synchronize 源 backend
    ↓
synchronize 目标 copy slot
    ↓
ggml_backend_tensor_copy(input, input_cpy)
```

这就是“上一个 split 的结果如何传给下一个 split”的实际实现。

### C3-d：MoE 局部权重 copy

当 source 是 host 上的 WEIGHTS，当前节点是 `GGML_OP_MUL_MAT_ID`，scheduler 会：

```text
读取 expert ID
    ↓
构建 used expert bitset
    ↓
将连续 expert 合并为区间
    ↓
只异步拷贝被使用的 expert
```

当前 simple-backend 使用普通 `GGML_OP_MUL_MAT`，不会进入此分支。

---

## 5.6 C4：提交当前 split graph

### 无 callback：整张 split 一次提交

```cpp
if (!sched->callback_eval) {
    enum ggml_status ec =
        ggml_backend_graph_compute_async(
            split_backend,
            &split->graph);

    if (ec != GGML_STATUS_SUCCESS) {
        return ec;
    }
}
```

这一步才把 graph 交给具体 backend 执行。scheduler 不直接实现 `MUL_MAT` 的底层循环，具体执行由 CPU/CUDA/Metal 等 backend 完成。

### 有 callback：切成 graph view 分段执行

```cpp
bool need = sched->callback_eval(t, true, user_data);
```

scheduler 先询问节点是否需要观察；连续的不需要观察的节点会合并到同一个 `[j0, j1]` 区间：

```cpp
struct ggml_cgraph gv =
    ggml_graph_view(&split->graph, j0, j1 + 1);

ggml_backend_graph_compute_async(split_backend, &gv);
ggml_backend_synchronize(split_backend);
```

随后以 `ask=false` 通知用户节点已经执行完成。如果 callback 返回 false，后续计算停止。

当前 simple-backend 没有设置 callback，走整张 split 一次提交的路径。

---

## 5.7 C5：记录 event，推进 split 状态

```cpp
if (sched->events[split_backend_id][sched->cur_copy] != nullptr) {
    ggml_backend_event_record(
        sched->events[split_backend_id][sched->cur_copy],
        split_backend);
}

prev_backend_id = split_backend_id;
```

event 记录当前 backend/copy slot 的执行位置，供后续 copy slot 复用时等待。

本例：

```cpp
parallel = false
n_copies = 1
```

通常没有创建 event，因此事件相关代码为空操作，实际主要由 backend synchronize 保证安全。

---

## 5.8 `compute_splits` 的输出

成功：

```cpp
return GGML_STATUS_SUCCESS;
```

表示各 split 的 copy 和 graph compute 已成功提交或执行。

失败：

```cpp
return ec;
```

表示具体 backend 在执行某个 split 时返回错误。

需要区分：

```text
compute_splits 成功：计算请求已成功提交
sched_graph_compute 返回：因为随后 synchronize，计算也已完成
```

---

# 6. 功能域 D：结果回读与布局验证

## 6.1 `ggml_graph_node(gf, -1)`

`compute()` 返回：

```cpp
return ggml_graph_node(gf, -1);
```

这只是取得原始 graph 的最后一个 node，即 `result`。它不执行读取，也不复制数据。

---

## 6.2 `ggml_backend_tensor_get`

```cpp
std::vector<float> out_data(ggml_nelements(result));

ggml_backend_tensor_get(
    result,
    out_data.data(),
    0,
    ggml_nbytes(result));
```

公共接口逻辑与 `tensor_set` 对称：

```cpp
void ggml_backend_tensor_get(
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    GGML_ASSERT(tensor);

    ggml_backend_buffer_t buf =
        tensor->view_src ?
        tensor->view_src->buffer :
        tensor->buffer;

    GGML_ASSERT(buf != nullptr);

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != nullptr);
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    buf->iface.get_tensor(
        buf,
        tensor,
        data,
        offset,
        size);
}
```

数据路径：

```text
backend result tensor
    ↓
buf->iface.get_tensor
    ↓
host out_data
```

必须在同步计算接口返回后调用，或者在直接使用 async 接口时先手动 synchronize。

---

## 6.3 矩阵乘法结果

输入矩阵：

```text
A =
[ 2  8 ]
[ 5  1 ]
[ 4  2 ]
[ 8  6 ]

B =
[ 10  5 ]
[ 9   9 ]
[ 5   4 ]
```

`ggml_mul_mat(a, b)` 的数学含义是：

```text
A × B^T
```

数学矩阵结果：

```text
[ 60  90  42 ]
[ 55  54  29 ]
[ 50  54  28 ]
[110 126  64 ]
```

GGML 第 0 维是连续维，因此线性内存顺序为：

```text
60, 55, 50, 110,
90, 54, 54, 126,
42, 29, 28, 64
```

源码按：

```cpp
out_data[j * result->ne[0] + i]
```

打印为：

```text
[ 60.00 55.00 50.00 110.00
  90.00 54.00 54.00 126.00
  42.00 29.00 28.00 64.00 ]
```

---

# 7. 功能域 E：资源释放

释放顺序：

```cpp
ggml_backend_sched_free(model.sched);
ggml_backend_free(model.backend);
ggml_backend_free(model.cpu_backend);
```

必须先释放 scheduler，再释放 backend，因为 scheduler 内部仍持有 backend 指针。

### scheduler 释放的对象

主要包括：

- events；
- galloc；
- scheduler context；
- hash set；
- split 和 split input 数组；
- graph input 数组；
- tensor copy 映射；
- node/leaf backend ID 数组；
- context buffer；
- scheduler graph 的 node/leaf 数组。

### backend 释放

随后释放具体 CPU/GPU backend 及其私有资源。此阶段不再执行计算，也不再读取 tensor。

---

# 8. 断点调试路线：按功能域验证

## 8.1 调度准备域

在 `alloc_graph` / `alloc_splits`：

```gdb
p sched->n_copies
p sched->cur_copy
p sched->next_copy
p sched->n_splits
p sched->graph.n_nodes
p sched->graph.n_leafs
p sched->is_alloc
```

观察：

```text
n_copies = 1
cur_copy = 0
next_copy = 0
```

## 8.2 输入装载域

在 `ggml_backend_tensor_set`：

```gdb
p tensor->name
p tensor->type
p tensor->ne[0]
p tensor->ne[1]
p tensor->nb[0]
p tensor->nb[1]
p tensor->data
p tensor->buffer
p tensor->view_src
p data
p offset
p size
p ggml_nbytes(tensor)
```

预期：

```text
a: ne=[2,4], nbytes=32
b: ne=[2,3], nbytes=24
```

## 8.3 split 执行域

在 `compute_splits`：

```gdb
p sched->n_splits
p sched->n_copies
p sched->cur_copy
p split_id
p split->backend_id
p split->i_start
p split->i_end
p split->n_inputs
p split->graph.n_nodes
```

处理 input 时：

```gdb
p input->name
p input->flags
p input->data
p input_cpy->data
p input_backend
p split_backend
```

执行前：

```gdb
p ggml_op_name(split->graph.nodes[0]->op)
```

## 8.4 结果域

```gdb
p result->ne[0]
p result->ne[1]
p result->nb[0]
p result->nb[1]
p ggml_nbytes(result)
x/12f out_data.data()
```

## 8.5 资源释放域

确认顺序：

```text
sched_free → backend_free
```

不要在 backend 已释放后继续调用 scheduler 的 tensor/buffer API。

---

# 9. 常见误区集中说明

## 9.1 allocation 成功不等于计算完成

```text
alloc_splits：准备 buffer/data
compute_splits：提交 copy 和算子计算
synchronize：等待完成
```

## 9.2 tensor_set 不执行算子

`ggml_backend_tensor_set` 只是把 host 输入写进已分配 tensor。

## 9.3 scheduler 不直接实现 MUL_MAT

`compute_splits` 负责调度，具体 `GGML_OP_MUL_MAT` 由实际 backend 执行。

## 9.4 async 返回不等于硬件完成

直接调用：

```cpp
ggml_backend_sched_graph_compute_async(...)
```

后必须根据需要调用：

```cpp
ggml_backend_sched_synchronize(...)
```

而同步入口 `ggml_backend_sched_graph_compute` 已经内置了这一步。

## 9.5 `buffer` 不是 tensor 数据

```text
buffer：管理对象，含函数指针、buffer type、context、容量、usage
data：tensor 的实际数据地址
```

## 9.6 `model.b` 不是 32 字节

```text
b->ne = [2,3,1,1]
b->nbytes = 24
```

调试器读取超过 24 字节会把其他区域误认为 b 数据。

## 9.7 split 之间不是简单共享指针

跨 backend 且 buffer 不兼容时必须：

```text
识别 source
    ↓
创建目标 backend copy
    ↓
运行前搬运数据
    ↓
让消费者 node->src 指向 copy
```

## 9.8 split 同步不只用于数值依赖

没有显式 input copy 时，仍可能因 allocator buffer reuse 而需要等待前一个 backend。

---

# 10. 一页式总结

```text
[调度准备]
reset
  ↓
split_graph：backend assignment、split、copy 描述
  ↓
alloc_splits：galloc reserve/alloc

[输入装载]
ggml_backend_tensor_set(a, matrix_A)
ggml_backend_tensor_set(b, matrix_B)

[执行]
graph_compute_async
  ↓
compute_splits
  ├─ split 间同步
  ├─ 跨 backend input copy
  ├─ backend graph compute async
  └─ event record
  ↓
synchronize 所有 backend

[输出]
ggml_graph_node(gf, -1) 得到 result
  ↓
ggml_backend_tensor_get
  ↓
host out_data

[释放]
sched_free
  ↓
backend_free
```

当前 simple-backend 的核心路径是：

```text
单 copy
单个简单 MUL_MAT split
输入写入 backend buffer
backend 执行 graph
同步后读取 result
```

更复杂模型才会明显展示：

```text
多 split
跨 backend copy
event 依赖
多 copy pipeline
MoE expert 局部权重搬运
callback 分段执行
```

最终职责边界：

```text
ggml_backend_sched_alloc_graph：决定并准备 graph 存储
ggml_backend_sched_alloc_splits：通过 galloc 完成 buffer allocation
ggml_backend_tensor_set：装载外部输入
ggml_backend_sched_compute_splits：搬运 split input 并提交 backend 计算
ggml_backend_sched_synchronize：等待异步计算完成
ggml_backend_tensor_get：读回 backend 结果
ggml_backend_sched_free/backend_free：释放运行时资源
```

---

# 11. 后续验证清单

1. 确认本例 `sched->n_splits`、每个 split 的 `backend_id` 和 `n_inputs`。
2. 确认 `model.a->data`、`model.b->data` 不重叠。
3. 确认第二次 `tensor_set` 的源为 `matrix_B`，size 为 24。
4. 进入具体 backend 的 `buf->iface.set_tensor`，确认实际上传实现。
5. 进入具体 backend 的 `ggml_backend_graph_compute_async`，定位 `MUL_MAT` backend 实现。
6. 在 `tensor_get` 中确认 result 的读取路径。
7. 若研究 pipeline，将 `parallel` 改为 `true`，观察 `n_copies`、events、`cur_copy`、`next_copy`。
8. 若研究跨 backend split，使用真正不同且 buffer 不兼容的 backend，观察 `split->inputs`、`input_cpy` 和 `node->src[j]`。

---

# 12. CPU backend 的内存分配入口

本节补充 CPU backend 内存申请的真实调用链。需要区分“调度层入口”和“底层实际申请入口”。

## 12.1 两个入口

从 `simple-backend.cpp` 的 scheduler 视角看，入口是：

```cpp
ggml_backend_sched_alloc_graph(model.sched, gf);
```

它内部经过：

```text
ggml_backend_sched_alloc_graph
  → ggml_backend_sched_alloc_splits
    → ggml_gallocr_reserve_n
    → ggml_gallocr_alloc_graph
```

从 CPU backend 真正向系统申请一块对齐内存的视角看，入口是：

```cpp
static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {
    void * data = ggml_aligned_malloc(size);
    return ggml_backend_buffer_init(
        buft,
        ggml_backend_cpu_buffer_i,
        data,
        size);
}
```

所以：

```text
scheduler 入口：ggml_backend_sched_alloc_graph
CPU 实际申请入口：ggml_backend_cpu_buffer_type_alloc_buffer
真正申请动作：ggml_aligned_malloc(size)
```

## 12.2 完整分配调用链

```text
ggml_backend_sched_alloc_graph
  → ggml_backend_sched_alloc_splits
    → ggml_gallocr_reserve_n
      → graph allocator 计算 buffer 需求
        → ggml_backend_buft_alloc_buffer
          → buft->iface.alloc_buffer
            → ggml_backend_cpu_buffer_type_alloc_buffer
              → ggml_aligned_malloc(size)
            → ggml_backend_buffer_init
```

`ggml_gallocr` 负责 graph 生命周期分析、buffer 需求计算和 tensor placement；CPU buffer type 负责实际申请底层 host 内存。

## 12.3 一整块 buffer 与多个 tensor

CPU backend 通常不是给 `a`、`b`、`result` 各自单独调用一次系统 `malloc`，而是申请一整块 backend buffer，再由 graph allocator 在其中放置多个 tensor：

```text
CPU backend buffer
+------------------------------+
| a data | b data | result ... |
+------------------------------+
```

随后通过 `ggml_tallocr`/`ggml_gallocr_init_tensor` 设置：

```cpp
tensor->buffer
tensor->data
```

因此：

```text
buffer->size：整块 backend buffer 容量
ggml_nbytes(tensor)：单个 tensor 的逻辑数据大小
tensor->data：该 tensor 在整块 buffer 中的具体地址
```

## 12.4 推荐断点

```gdb
break ggml_backend_sched_alloc_graph
break ggml_backend_sched_alloc_splits
break ggml_gallocr_reserve_n
break ggml_gallocr_alloc_graph
break ggml_backend_buft_alloc_buffer
break ggml_backend_cpu_buffer_type_alloc_buffer
break ggml_aligned_malloc
break ggml_backend_buffer_init
```

在 CPU 实际申请函数中观察：

```gdb
p size
p ggml_backend_buft_name(buft)
```

在 tensor 落位阶段观察：

```gdb
p tensor->name
p tensor->buffer
p tensor->data
p ggml_nbytes(tensor)
```

---

# 13. chunk：graph allocator 的分段内存

## 13.1 chunk 的概念

在 `src/ggml-alloc.c` 中，chunk 是 graph allocator 管理的一个逻辑/物理 backend 内存分段。相关定义：

```cpp
#define GGML_VBUFFER_MAX_CHUNKS 16

struct buffer_address {
    int chunk;
    size_t offset;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};
```

需要区分：

```text
tallocr_chunk：allocator 侧的规划信息
vbuffer->chunks[i]：实际 backend buffer 对象
```

一个 tensor 的地址不是单独的 offset，而是：

```text
(chunk_id, offset_in_chunk)
```

例如：

```text
tensor A → chunk 0, offset 0x0000
tensor B → chunk 0, offset 0x0020
tensor C → chunk 1, offset 0x0000
```

## 13.2 chunk 的作用

chunk 主要解决：

1. backend 单个 buffer 有最大容量时，将整体需求拆分；
2. 将多个 backend buffer 统一抽象成一个虚拟 buffer；
3. 在每个 chunk 内维护 free block；
4. 支持 tensor 生命周期结束后的内存复用；
5. 通过 `(chunk, offset)` 定位 tensor；
6. 满足 backend 的对齐和 allocation 限制。

逻辑上可以表示为：

```text
虚拟 vbuffer
+-------------+-------------+-------------+
| chunk 0     | chunk 1     | chunk 2     |
| backend buf | backend buf | backend buf |
+-------------+-------------+-------------+
```

当前 simple-backend 的小型 CPU graph 通常只有一个 chunk；复杂模型或受 backend 最大容量限制时才会出现多个 chunk。

## 13.3 chunk 的创建

```cpp
static int ggml_dyn_tallocr_new_chunk(
        struct ggml_dyn_tallocr * alloc,
        size_t min_size) {
    struct tallocr_chunk * chunk =
        calloc(1, sizeof(struct tallocr_chunk));

    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    chunk->free_blocks[0].size =
        MAX(min_size, alloc->max_chunk_size);

    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    return alloc->n_chunks - 1;
}
```

新 chunk 初始只有一个 free block：

```text
offset = 0
size = max(min_size, max_chunk_size)
```

真正创建对应 backend buffer 的位置：

```cpp
static struct vbuffer * ggml_vbuffer_alloc(...) {
    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size =
            talloc->chunks[n]->max_size;

        buf->chunks[n] =
            ggml_backend_buft_alloc_buffer(
                buft,
                chunk_size);
    }
}
```

CPU 路径最终仍然是：

```text
chunk_size
  → ggml_backend_buft_alloc_buffer
  → ggml_backend_cpu_buffer_type_alloc_buffer
  → ggml_aligned_malloc(chunk_size)
```

## 13.4 chunk 中的 tensor 分配和复用

动态分配函数：

```cpp
static struct buffer_address ggml_dyn_tallocr_alloc(
        struct ggml_dyn_tallocr * alloc,
        size_t size,
        const struct ggml_tensor * tensor)
```

逻辑分三步：

```text
1. 将 size 按 alignment 对齐
2. 在已有 chunk 的 free block 中寻找合适区域
3. 如果没有空间，则创建新 chunk
```

它返回：

```cpp
struct buffer_address {
    int chunk;
    size_t offset;
};
```

tensor 最终地址由：

```cpp
void * base = ggml_backend_buffer_get_base(
    buf->chunks[buf_addr.chunk]);

void * addr = (char *) base + buf_addr.offset;
```

得到：

```text
tensor->data = chunk backend buffer base + chunk 内 offset
```

tensor 生命周期结束后，空间进入 `free_blocks`，后续 tensor 可以复用。同一 chunk 内的相邻 free block 还会被合并，以减少碎片。

## 13.5 当前例子的典型状态

当前输入和结果规模很小：

```text
a      = 32 bytes
b      = 24 bytes
result = 约 48 bytes
```

通常可由一个 chunk 承载：

```text
chunk 0：
  a      offset 0
  b      offset 32
  result offset 64 或复用已经结束的区域
```

此前观察到：

```cpp
model.a->buffer->size = 128
```

可以理解为整个 chunk/backend buffer 的容量，而不是 `a` 的数据大小。`a` 的有效数据仍是：

```cpp
ggml_nbytes(model.a) = 32
```

---

# 14. CPU 矩阵计算的最终入口和完整调用链

## 14.1 应用层入口

`simple-backend.cpp` 中：

```cpp
ggml_backend_sched_graph_compute(model.sched, gf);
```

这是同步 scheduler 入口，但不直接执行矩阵乘法。

## 14.2 scheduler 到具体 backend

```text
ggml_backend_sched_graph_compute
  → ggml_backend_sched_graph_compute_async
    → ggml_backend_sched_compute_splits
      → ggml_backend_graph_compute_async(
            split_backend, &split->graph)
```

`compute_splits` 负责：

- 遍历 split；
- 处理 split 间同步；
- 处理跨 backend input copy；
- 把 split graph 转发给目标 backend。

## 14.3 通用 backend 分发

```cpp
enum ggml_status ggml_backend_graph_compute_async(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    return backend->iface.graph_compute(
        backend,
        cgraph);
}
```

`backend->iface.graph_compute` 是 backend 虚接口：

```text
CPU → ggml_backend_cpu_graph_compute
CUDA → CUDA graph compute
Metal → Metal graph compute
其他 → 相应 backend 实现
```

因此先检查：

```gdb
p split->backend_id
p ggml_backend_name(split_backend)
```

只有当前 split 确实使用 CPU backend，才会进入下面的 CPU C/C++ 执行链。

## 14.4 CPU backend graph compute

```cpp
static enum ggml_status ggml_backend_cpu_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx =
        (struct ggml_backend_cpu_context *) backend->context;

    struct ggml_cplan cplan =
        ggml_graph_plan(
            cgraph,
            cpu_ctx->n_threads,
            cpu_ctx->threadpool);

    // 必要时申请 CPU 临时 workspace
    ...

    return ggml_graph_compute(cgraph, &cplan);
}
```

这里的 `cplan.work_data` 是算子执行 workspace，不是 `a/b/result` 的 tensor buffer。tensor buffer 已由 graph allocator 准备。

## 14.5 CPU graph executor

```cpp
ggml_graph_compute(cgraph, &cplan)
```

负责：

- 设置/复用 threadpool；
- 绑定当前 graph 和 plan；
- 启动 worker；
- 等待 worker 完成；
- 返回执行状态。

worker 进入：

```cpp
ggml_graph_compute_thread(void * data)
```

## 14.6 CPU node 遍历和 op 分发

```cpp
for (int node_n = 0;
     node_n < cgraph->n_nodes;
     node_n++) {
    struct ggml_tensor * node =
        cgraph->nodes[node_n];

    if (ggml_op_is_empty(node->op)) {
        continue;
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        continue;
    }

    ggml_compute_forward(&params, node);
}
```

当前 graph 的 node 是：

```text
node = result
node->op = GGML_OP_MUL_MAT
```

因此进入：

```cpp
case GGML_OP_MUL_MAT:
    ggml_compute_forward_mul_mat(params, tensor);
    break;
```

## 14.7 最终矩阵计算入口

CPU backend 下，最接近具体矩阵乘法算法的入口是：

```cpp
void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst)
```

进一步通常进入：

```cpp
ggml_compute_forward_mul_mat_one_chunk(...)
```

最后通过：

```cpp
vec_dot(...)
```

执行向量点积/乘加 kernel。

完整 CPU 调用链：

```text
ggml_backend_sched_graph_compute
  → ggml_backend_sched_graph_compute_async
    → ggml_backend_sched_compute_splits
      → ggml_backend_graph_compute_async
        → backend->iface.graph_compute
          → ggml_backend_cpu_graph_compute
            → ggml_graph_plan
            → ggml_graph_compute
              → ggml_graph_compute_thread
                → ggml_compute_forward
                  → GGML_OP_MUL_MAT
                    → ggml_compute_forward_mul_mat
                      → ggml_compute_forward_mul_mat_one_chunk
                        → vec_dot
```

## 14.8 `MUL_MAT` 内部数据访问

`ggml_compute_forward_mul_mat` 从：

```cpp
const struct ggml_tensor * src0 = dst->src[0];
const struct ggml_tensor * src1 = dst->src[1];
```

获取输入。在本例中：

```text
src0 = model.a
src1 = model.b
dst  = result
```

它根据：

```cpp
ne[]
nb[]
data
```

计算行/列的实际字节地址，并通过 `vec_dot` 计算点积。对于当前 F32 示例，不涉及量化权重解码；量化类型则会使用 `type_traits_cpu` 中相应的 `vec_dot` 和转换路径。

## 14.9 推荐矩阵计算断点

```gdb
break ggml_backend_sched_compute_splits
break ggml_backend_graph_compute_async
break ggml_backend_cpu_graph_compute
break ggml_graph_compute
break ggml_graph_compute_thread
break ggml_compute_forward
break ggml_compute_forward_mul_mat
break ggml_compute_forward_mul_mat_one_chunk
```

重点观察：

```gdb
p ggml_backend_name(split_backend)
p split->graph.n_nodes
p node->op
p node->src[0]
p node->src[1]
p dst->ne[0]
p dst->ne[1]
p params->ith
p params->nth
p dst->data
```

---

# 15. 计算入口的最终结论

当前 CPU 路径有三个层次的“入口”：

```text
应用/调度入口：
ggml_backend_sched_graph_compute

CPU backend graph 入口：
ggml_backend_cpu_graph_compute

MUL_MAT 算法入口：
ggml_compute_forward_mul_mat
```

真正执行本例矩阵乘法的完整链路是：

```text
scheduler 接收 graph
  → scheduler 遍历 split
    → 通用 backend 接口分发
      → CPU backend 生成 cplan
        → CPU threadpool 遍历 graph node
          → 根据 node->op 分发
            → ggml_compute_forward_mul_mat
              → 分块计算
                → vec_dot
```

因此：

```text
ggml_backend_sched_graph_compute：负责调度和同步
 ggml_backend_cpu_graph_compute：负责 CPU backend 执行准备
 ggml_graph_compute_thread：负责遍历并执行 node
 ggml_compute_forward_mul_mat：负责 MUL_MAT 算法
 vec_dot：负责底层向量点积
```
