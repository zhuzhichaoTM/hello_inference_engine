# 03 · compute：调度、显存分配、CUDA kernel 全链路

> 源码基线：`/home/ggml`，commit `353b63b`
>
> 对应源码：`examples/simple/simple-backend.cpp:98-112`、`src/ggml-backend.cpp`（scheduler）、`src/ggml-alloc.c`（gallocr）、`src/ggml-cuda/ggml-cuda.cu`、`src/ggml-cuda/mmvf.cu`
>
> 本文回答四个问题：
>
> 1. **调度**：为什么 `MUL_MAT` 被判给 CUDA0，而不是 CPU？
> 2. **内存**：为什么只申请 384 B，三个 tensor 为什么在 0/128/256？
> 3. **执行**：为什么走 MMVF 而不是 cuBLAS，kernel 参数怎么来的？
> 4. **复用**：为什么 `a`/`b` 在分配后就被标记为空闲，为什么仍然安全？

---

## 0 全景

### 0.1 `compute()` 源码

```cpp
struct ggml_tensor * compute(simple_model & model, struct ggml_cgraph * gf) {
    ggml_backend_sched_reset(model.sched);
    ggml_backend_sched_alloc_graph(model.sched, gf);

    ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));

    ggml_backend_sched_graph_compute(model.sched, gf);

    return ggml_graph_node(gf, -1);
}
```

### 0.2 五个职责域

| 域 | 入口 | 做什么 | 不做什么 |
|---|---|---|---|
| A 调度 | `sched_alloc_graph` | 决定 node 归属、切 split | 不执行算子 |
| B 分配 | `sched_alloc_splits` / `gallocr` | 申请显存、放置 tensor | 不写数据 |
| C 装载 | `tensor_set` | host → device | 不建图、不计算 |
| D 执行 | `sched_graph_compute` | 遍历 split、launch kernel、同步 | scheduler 不实现 kernel |
| E 回读/释放 | `tensor_get` / `sched_free` | D2H、释放 | — |

### 0.3 断点时序

```text
[A1] sched_alloc_graph      nodes=1 leafs=2 n_splits(before)=0
[A2] sched_split_graph      node_0 MUL_MAT
[A3] sched_alloc_splits     n_splits=1 backend_id=0 n_inputs=0
[A4] gallocr_alloc_graph    galloc n_nodes=0 → 失败
[A5] gallocr_reserve_n      nodes=1 leafs=2
[A6] buft_alloc_buffer      CUDA0 size=384
[A7] cuda buft alloc        device=0
[A8] cudaMalloc             0x7f5600c00000
[A9] buffer_init            size=384
[A4] gallocr_alloc_graph    成功（n_nodes=1）
[B1] tensor_set a           32 B
[B2] cuda set_tensor        dst=base+0x000
[B1] tensor_set b           24 B
[B2] cuda set_tensor        dst=base+0x080
[C1] graph_compute          n_splits=1 is_alloc=1
[C2] compute_splits         split[0] backend_id=0 n_inputs=0
[C3] graph_compute_async
[C4] cuda graph_compute     node_0 MUL_MAT
[C5] evaluate_and_capture   use_cuda_graph=0
[D1] compute_forward        op=MUL_MAT
[D2] mul_mat                F32/F32/F32
[H2] mul_mat_vec_f
[H1] launch                 grid=(4,1,1) block=(32,1,1) shared=128
[I3] synchronize
[E1] tensor_get             48 B
[E2] cuda get_tensor
[F3] sched_free
[F1] cuda buffer free
[F2] cuda backend free
```

---

## 1 `ggml_backend_sched_reset()`

```cpp
void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(int));
        memset(sched->hv_tensor_copies, 0, ...);
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}
```

本次运行：

```text
is_reset = 1（init_model 里已经 reset 过）
→ 只执行 is_alloc = false
```

为什么 reset 不释放 buffer：

```text
buffer 归 gallocr 所有，生命周期跨多次 compute
reset 只重置“调度状态”（backend 归属、copy、is_alloc）
下一次 alloc_graph 会重新 split；如果 backend 归属和上一轮一致，
gallocr 可以复用同一块 buffer（needs_realloc == false）
```

这是“每次 compute 重新决策，但尽量复用内存”的设计。

---

## 2 `ggml_backend_sched_alloc_graph()`：调度决策

```cpp
bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;
    return true;
}
```

断点：

```text
[A1] graph nodes=1 leafs=2
     n_splits(before)=0 is_reset=1 is_alloc=0 n_copies=1 cur_copy=0
```

`hash_set.size = 2053 >= 1 + 2`，断言通过。

### 2.1 `split_graph` 的五个 pass

`src/ggml-backend.cpp:1066`。五个 pass 的顺序不是随意的，每一步都在缩小“未知”：

```text
pass 1：已有 buffer 的 tensor → 直接定 backend（最确定的）
pass 2：沿相邻节点扩散（利用图拓扑）
pass 3：仍未定的 node → 按 backend 优先级 + 输入兼容性选择
pass 4：把 src 拉到 dst 所在 backend（依赖方向）
pass 5：按 backend id 切 split，插入跨 backend copy
```

#### pass 1：预分配优先

```cpp
for (int i = 0; i < graph->n_leafs; i++) {
    struct ggml_tensor * leaf = graph->leafs[i];
    int * leaf_backend_id = &tensor_backend_id(leaf);
    if (*leaf_backend_id == -1) {
        *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
    }
}
```

`ggml_backend_sched_backend_id_from_cur()` 的判定：

```text
1. tensor->buffer 已存在 → 用 buffer 的 buft 所属 backend
2. view_src->buffer 已存在 → 用 view_src 的 backend
3. flags & GGML_TENSOR_FLAG_INPUT → 最后一个 backend（CPU）
4. src 中有 WEIGHTS buffer → 用权重 backend（可选 op_offload）
5. 否则 -1
```

本例：

```text
a、b、result 都没有 buffer
flags：a=0, b=0, result=16（COMPUTE，不是 INPUT）
没有权重
→ 全部 -1
```

**设计意图**：已经分配内存的 tensor 不能移动，必须尊重现有位置；用户显式标记的输入默认留在 CPU（避免每轮 H2D）；权重决定计算位置（权重在哪，计算尽量在哪，避免搬运）。

#### pass 2：拓扑扩散

```cpp
// expand gpu down
// expand gpu up
// expand rest down
// expand rest up
```

规则：

```text
GPU 向下/向上扩散：遇到 CPU 节点就停止（CPU 是最后优先级）
rest 向下/向上：任何已定节点都可以扩散
```

本例所有 id 都是 -1，没有起点，pass 2 全部跳过。

#### pass 3：能力 + 优先级决策（本例的关键）

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    int * node_backend_id = &tensor_backend_id(node);
    if (*node_backend_id == -1) {
        int n_supported_best = -1;
        for (int b = 0; b < sched->n_backends; b++) {
            if (ggml_backend_supports_op(sched->backends[b], node)) {
                int n_supported = 0;
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) continue;
                    if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1)
                        && ggml_backend_sched_buffer_supported(sched, src, b)) {
                        n_supported++;
                    }
                }
                if (n_supported > n_supported_best) {
                    n_supported_best = n_supported;
                    *node_backend_id = b;
                }
            }
        }
    }
}
```

代入本例：

```text
node_0 = MUL_MAT
src[0] = a (backend_id = -1)
src[1] = b (backend_id = -1)
```

对每个 backend：

```text
b = 0 (CUDA0):
    supports_op(MUL_MAT) = true
    src a/b 的 backend_id 都是 -1 → 条件不满足 → n_supported = 0
    0 > -1 → 选中 CUDA0，n_supported_best = 0

b = 1 (CPU):
    supports_op(MUL_MAT) = true
    n_supported = 0
    0 > 0 不成立 → 不覆盖
```

**结论：CUDA0 胜出的原因是“下标更小 + 严格大于”，不是“CUDA 更快”**。如果 CPU 放在 index 0，CPU 会胜出。这就是为什么 backend 数组顺序是调度优先级。

`ggml_backend_supports_op` 对 CUDA 的实现（`ggml-cuda.cu:5136`）会检查：

```cpp
case GGML_OP_MUL_MAT:
    if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b))
        return false;
    if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16)
        return false;
    switch (a->type) {
        case F32: case F16: case Q4_0: ... case BF16: return true;
        default: return false;
    }
```

本例 `a->nb[0] = 4 == sizeof(float)`，`b->nb[0] = 4`，类型 F32，返回 true。

**设计意图**：`supports_op` 只做“能力粗筛”（类型、布局、设备），不做“性能判断”。性能判断在 kernel 内部（`should_use_*`）。

#### pass 4：src 跟随 dst

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    int * cur_backend_id = &tensor_backend_id(node);
    ...
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * src = node->src[j];
        if (src == NULL) continue;
        int * src_backend_id = &tensor_backend_id(src);
        if (*src_backend_id == -1) {
            if (src->view_src != NULL) {
                *src_backend_id = tensor_backend_id(src->view_src);
            } else {
                *src_backend_id = *cur_backend_id;
            }
        }
    }
}
```

本例：

```text
node_0 backend = 0
a、b 都没有 view_src → 都赋为 0
```

断点验证：

```text
node_backend_ids[0] = 0
leaf_backend_ids[0] = 0
leaf_backend_ids[1] = 0
```

**设计意图**：数据应该去计算所在的地方，而不是计算去数据所在的地方（计算移动成本远高于数据移动，且数据可以预取）。跨 backend 的情况在 pass 5 处理。

#### pass 5：切 split + 插入 copy

```cpp
struct ggml_backend_sched_split * split = &sched->splits[0];
split->backend_id = tensor_backend_id(node);
split->i_start = 0;
split->n_inputs = 0;
int cur_backend_id = split->backend_id;

for (; i < graph->n_nodes; i++) {
    ...
    if (node_backend_id != cur_backend_id || need_new_split) {
        split->i_end = i;
        i_split++;
        split = &sched->splits[i_split];
        split->backend_id = node_backend_id;
        split->i_start = i;
        split->n_inputs = 0;
        cur_backend_id = node_backend_id;
    }

    // 跨 backend 输入：创建 copy tensor
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * src = node->src[j];
        ...
        if (src_backend_id != cur_backend_id && !buffer_supported(...)) {
            if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                ...
                split->inputs[split->n_inputs++] = src;
            }
            node->src[j] = tensor_id_copy(...);
        }
    }
}
split->i_end = graph->n_nodes;
sched->n_splits = i_split + 1;
```

本例：

```text
n_splits = 1
split[0].backend_id = 0
split[0].i_start = 0
split[0].i_end = 1
split[0].n_inputs = 0
split[0].graph.n_nodes = 1
```

断点：

```text
[A3] sched_alloc_splits: n_splits=1 sched_graph nodes=1 leafs=2
split[0]: backend_id=0 i_start=0 i_end=1 n_inputs=0 graph.nodes=1
```

**为什么 `n_inputs = 0`**：`a`、`b` 都被判给 CUDA0，和 `node_0` 同一个 backend，不需要跨 backend copy。`split->inputs[]` 记录的是“需要在 split 开始前从其他 backend 拷进来的 tensor”，不是“图输入”。

#### 图副本的尺寸

```cpp
int graph_size = std::max(graph->n_nodes, graph->n_leafs)
               + total_inputs * 2 * sched->n_copies
               + n_dep_nodes;
```

本例：

```text
max(1, 2) = 2
total_inputs = 0
n_dep_nodes = 0
→ sched->graph.size = 2
```

断点：

```text
model.sched->graph.size = 2
model.sched->graph.n_nodes = 1
model.sched->graph.n_leafs = 2
```

为什么是 `max` 而不是 `n_nodes + n_leafs`：`nodes[]` 和 `leafs[]` 是两个独立数组，各自需要 `size` 个槽位，不是共享。

`split[0].graph` 是 `ggml_graph_view(graph, i_start, i_end)`：

```cpp
struct ggml_cgraph cgraph = {
    /*.size             =*/ 0,
    /*.n_nodes          =*/ i1 - i0,
    /*.n_leafs          =*/ 0,
    /*.nodes            =*/ cgraph0->nodes + i0,
    ...
};
```

断点：

```text
split[0].graph.size = 0
split[0].graph.n_nodes = 1
split[0].graph.n_leafs = 0
```

`size = 0` 是 view 的约定：它不拥有数组，只是窗口；`n_leafs = 0` 是因为 split 只关心节点区间，叶子通过原图的 buffer 访问。

### 2.2 跨 backend 的反事实

假设 `a` 被显式放到 CPU buffer（`ggml_backend_alloc_ctx_tensors_from_buft` 或手动分配）：

```text
pass 1: a 有 buffer → backend_id = 1 (CPU)
pass 3: node_0 的 src a 已定，但 buffer_supported(CPU buft, CUDA) = false
        → n_supported 仍为 0 → node_0 可能仍选 CUDA（下标优先）
pass 4: b → 0
pass 5: node_0 在 CUDA，src a 在 CPU，buffer 不兼容
        → 创建 a 的 CUDA copy（在 sched->ctx 中）
        → split[0].n_inputs = 1
        → node_0->src[0] = copy
compute_splits:
        → 在 split 开始前把 a 从 CPU 拷到 CUDA copy
        → 如果是异步 copy，用 event 同步
```

这就是真实 LLM 中“权重在 CPU、计算在 GPU”或“部分层 offload”的机制。

> **实验验证**：[04-gdb-trace.md](04-gdb-trace.md) §12 构造了 `a` 在 CPU、`b` 在 CUDA 的场景，实测：
>
> ```text
> SPLIT #0: CUDA0 # 1 inputs: [leaf_0]
> sched->graph.n_nodes = 3
> nodes = [leaf_0 (view), CUDA0#leaf_0#0, node_0]
> node_backend_ids = [1, 0, 0]
> ```
>
> 其中 `leaf_0 (view)` 是保活依赖，`CUDA0#leaf_0#0` 是真实 copy tensor；执行期 `cpy_tensor_async` 因 src 不是 CUDA buffer 被拒绝，自动降级为 `tensor_copy` → `cudaMemcpyAsync(H2D)`。

---

## 3 `ggml_backend_sched_alloc_splits()`：显存分配

### 3.1 两次尝试

```cpp
static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    // 比较 node/leaf backend id 与上一轮

    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
        // 同步所有 backend
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph,
                                    sched->node_backend_ids, sched->leaf_backend_ids)) {
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            return false;
        }
    }
    return true;
}
```

本次是首次分配：

```text
galloc->n_nodes = 0
graph->n_nodes  = 1
→ needs_realloc = true
→ n_buffers = 2 > 1，不能自动 realloc
→ 返回 false
```

日志：

```text
ggml_gallocr_needs_realloc: graph has different number of nodes
ggml_gallocr_alloc_graph: cannot reallocate multi buffer graph automatically, call reserve
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
```

**这不是错误**，而是设计好的两阶段流程：

```text
reserve：做生命周期分析、算峰值、申请 buffer、记录每个 tensor 的 (buffer_id, chunk, offset)
alloc  ：把记录好的地址写进 tensor->data / tensor->buffer（不重新分析）
```

为什么要两阶段：

```text
1. reserve 可能很慢（O(V+E) 生命周期分析），但只需在图结构变化时做
2. alloc 很快（O(V) 指针赋值），每次 compute 都可以做
3. 多 buffer type 时无法自动 realloc，必须显式 reserve
```

### 3.2 同步：第一次创建 CUDA stream

```cpp
for (int i = 0; i < sched->n_backends; i++) {
    ggml_backend_synchronize(sched->backends[i]);
}
```

CUDA：

```cpp
static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
}
```

断点（调用前打印）：

```text
=== [I3] cuda synchronize: backend=0x55b542e52030 stream=(nil) ===
```

进入后 `cuda_ctx->stream()` 惰性创建：

```cpp
cudaStream_t stream(int device, int stream) {
    if (streams[device][stream] == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
    }
    return streams[device][stream];
}
```

第二次同步（compute 之后）：

```text
stream = 0x55b54372f9f0（非空）
```

**为什么同步**：reserve 可能释放旧 buffer（`ggml_vbuffer_free`）。如果旧 buffer 上还有异步 kernel 在跑，`cudaFree` 会导致未定义行为。同步保证所有 backend 的工作完成后再动内存。

### 3.3 `reserve_n`：生命周期分析与 buffer 申请

```cpp
bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                            const int * node_buffer_ids, const int * leaf_buffer_ids) {
    return ggml_gallocr_reserve_n_impl(..., /*no_alloc =*/ false);
}
```

`reserve_n_impl` 的步骤：

```text
1. min_hash_size = (n_nodes + n_leafs) = 3
   加 25% 余量：3 + 3/4 = 3
   galloc->hash_set = ggml_hash_set_new(3) → size = 3
2. 重置所有 tallocr
3. ggml_gallocr_alloc_graph_impl：逐个 tensor 分配 (buffer_id, chunk, offset)
4. 把结果写入 node_allocs[] / leaf_allocs[]
5. 对每个 buffer type，如果新峰值 > 当前 buffer 大小 → ggml_vbuffer_alloc
```

断点：

```text
[A5] gallocr_reserve_n: graph nodes=1 leafs=2
galloc->hash_set.size = 3
```

为什么是 3 而不是质数表里的更大值：`ggml_hash_size(3)` 返回 3（质数表里 3 在列）。

### 3.4 tallocr：按 buffer type 的分配器

```cpp
struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[16];
    int n_chunks;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};
```

创建（`ggml_dyn_tallocr_new`）：

```cpp
alignment      = 128 (CUDA) / 32 (CPU)
max_chunk_size = MIN(max_buffer_size, SIZE_MAX/2) = 2^63-1
n_chunks       = 1
chunk[0].free_blocks[0] = { offset = 0, size = max(min_size, 2^63-1) }
chunk[0].max_size = 0
```

**为什么 chunk 初始大小是“几乎无穷”**：tallocr 不知道最终需要多少，先用一个虚拟的大 free block 记录 offset；每次分配只推进 offset，并更新 `chunk->max_size`。真正的物理 buffer 在分析完成后按 `max_size` 一次性申请。

**为什么最多 16 个 chunk**：某些后端有单块 buffer 上限（例如 32 位索引、显存碎片）；超过时开新 chunk。`GGML_VBUFFER_MAX_CHUNKS = 16`。

### 3.5 分配顺序与地址

`ggml_gallocr_alloc_graph_impl`：

```text
1. 分配所有 leaf（按 leaf_backend_ids）
2. 统计每个 tensor 的 n_children / n_views
3. 按节点顺序，先分配 node 的 src，再分配 node
4. 每处理完一个 node，把 src 的 n_children--，归零则 free
```

本例：

```text
leafs: a, b
nodes: node_0

allocate a   → tallocr_alloc(32)  → aligned 128 → offset 0
allocate b   → tallocr_alloc(24)  → aligned 128 → offset 128
allocate node_0 → tallocr_alloc(48) → aligned 128 → offset 256
```

断点（`ggml_vbuffer_tensor_alloc`）：

```text
[G1] tensor=leaf_0 chunk=0 offset=0   size=32 base=0x7f6088c00000 -> addr=0x7f6088c00000
[G1] tensor=leaf_1 chunk=0 offset=128 size=24 base=0x7f6088c00000 -> addr=0x7f6088c00080
[G1] tensor=node_0 chunk=0 offset=256 size=48 base=0x7f6088c00000 -> addr=0x7f6088c00100
```

### 3.6 384 字节的推导

```text
aligned_size(a)      = ceil(32/128)  * 128 = 128
aligned_size(b)      = ceil(24/128)  * 128 = 128
aligned_size(result) = ceil(48/128)  * 128 = 128
chunk.max_size       = 0 + 128 + 128 + 128 = 384
```

注意 `ggml_backend_buft_get_alloc_size()` 对 F32 返回的是逻辑大小：

```text
alloc_size: a=32 b=24 result=48
```

对齐发生在 `ggml_dyn_tallocr_alloc`：

```cpp
size = aligned_offset(NULL, size, alloc->alignment);
```

**为什么需要对齐**：

```text
1. float4 / half2 / int4 向量化加载要求 16 B 对齐
2. 量化块（Q4_0 18 B、QK_K 256 元素）需要块级对齐
3. tensor core ldmatrix 对行首有 16/32/128 B 要求
4. 同一 buffer 内多 tensor 时，每个 tensor 的 data 都要满足上述要求
```

代价：小 tensor 的填充比例很高（104 B 有效数据占 384 B）。对大 tensor 可以忽略。

### 3.7 物理 buffer 申请

```cpp
static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft,
                                           const struct ggml_dyn_tallocr * talloc,
                                           enum ggml_backend_buffer_usage usage) {
    struct vbuffer * buf = calloc(1, sizeof(struct vbuffer));

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}
```

断点：

```text
[A6] backend_buft_alloc_buffer: buft=CUDA0 size=384
[A7] cuda_buffer_type_alloc_buffer: size=384 device=0
[A8] cuda alloc done: dev_ptr=0x7f5600c00000
[A9] backend_buffer_init: buft=CUDA0 size=384 context=...
```

CUDA 实现：

```cpp
static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    ...
    ggml_backend_cuda_buffer_context * ctx =
        new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}
```

`ggml_cuda_device_malloc`：

```cpp
if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
    err = cudaMallocManaged(ptr, size);
} else {
    err = cudaMalloc(ptr, size);
}
```

本次没有设置环境变量，走 `cudaMalloc`。

**为什么不是 VMM**：tensor buffer 分配固定走 `ggml_cuda_device_malloc`；VMM（`cuMemCreate/cuMemMap`）是 compute pool 的临时 workspace 路径，两套机制。本例也没有创建 pool。

### 3.8 CPU tallocr 为什么是空的

断点：

```text
talloc[1]: alignment=32 max_chunk_size=2^63-1 n_chunks=0
vbuffer[1]: chunk0=(nil)
```

因为没有任何 tensor 的 `buffer_id == 1`。`ggml_vbuffer_alloc` 对 CPU 也会被调用（`buffers[1]` 非空），但 `n_chunks = 0`，循环不执行，不申请任何 host 内存。

**设计价值**：scheduler 总是为所有 backend 建 tallocr，但只有实际使用的才产生物理 buffer。这避免了“为每个 backend 预分配”的浪费。

### 3.9 第二次 `alloc_graph`：只写指针

```cpp
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) { ... }

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    for (int i = 0; i < graph->n_leafs; i++) {
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }
    return true;
}
```

`ggml_gallocr_init_tensor`：

```cpp
if (tensor->view_src != NULL) {
    ggml_backend_view_init(tensor);
} else if (tensor->data == NULL) {
    ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
} else {
    // 外部已分配，跳过
}
```

**关键**：第二次 alloc 不做任何生命周期分析，只把 `leaf_allocs/node_allocs` 里记录的 `(chunk, offset)` 转成 `tensor->data`。

### 3.10 分配结果的完整数据

断点（`simple-backend.cpp:104`，alloc 之后）：

```text
=== [K1] after alloc_graph: scheduler/galloc internals ===
sched->graph.size=2 n_nodes=1 n_leafs=2
split[0].graph.size=0 n_nodes=1 n_leafs=0
galloc->n_buffers=2 n_nodes=1 n_leafs=2 hash_set.size=3

talloc[0]: alignment=128 max_chunk_size=2^63-1 n_chunks=1
           chunk0.max_size=384 chunk0.n_free_blocks=2
talloc[1]: alignment=32 n_chunks=0

vbuffer[0]: chunk0=0x55d202423a90 size=384
vbuffer[1]: chunk0=(nil)

leaf_alloc[0]: buffer_id=0 chunk=0 offset=0   size_max=32
leaf_alloc[1]: buffer_id=0 chunk=0 offset=128 size_max=24
node_alloc[0].dst:    buffer_id=0 chunk=0 offset=256 size_max=48
node_alloc[0].src[0]: buffer_id=0 chunk=0 offset=0   size_max=32
node_alloc[0].src[1]: buffer_id=0 chunk=0 offset=128 size_max=24

alloc_size: a=32 b=24 result=48
alignment: cuda=128 cpu=32
```

### 3.11 为什么 `n_free_blocks = 2`

这是本例最容易被忽略、也最能说明 gallocr 设计的一组数据：

```text
chunk0.max_size      = 384
chunk0.n_free_blocks = 2
free[0] = { offset = 0,   size = 256 }
free[1] = { offset = 384, size = 2^63-1 - 384 }
```

推导（发生在 **reserve 的 `ggml_gallocr_alloc_graph_impl`** 中，不是第二次 alloc）：

```text
初始：free[0] = {0, huge}

alloc a      → {0,128} 被占用，free[0] = {128, huge}
alloc b      → {128,128} 被占用，free[0] = {256, huge}
alloc result → {256,128} 被占用，free[0] = {384, huge}
  此时 n_free_blocks = 1，max_size = 384

node_0 分配完成后，处理 src：
  a.n_children-- → 0，无 view，allocated → free_node(a)
      free_bytes({0,128}) → free = {0,128}, {384,huge}
  b.n_children-- → 0，无 view，allocated → free_node(b)
      free_bytes({128,128}) → 与 {0,128} 合并 → {0,256}
      free = {0,256}, {384,huge}       ← n_free_blocks = 2
```

**为什么 `a`、`b` 会被释放**：

```text
node_0 是它们唯一的消费者
node_0 处理完后，n_children 归零，且没有 view
它们不是 OUTPUT（flags 都是 0）
→ ggml_gallocr_free_node 被调用
```

注意：`ggml_gallocr_free_node` **只检查 `GGML_TENSOR_FLAG_OUTPUT`**，不检查 `INPUT`。即使标记了 `INPUT`，tensor 在最后一个消费者之后仍然会被释放。`INPUT` 影响的是：

```text
1. scheduler pass 1 的默认 backend 选择（CPU）
2. gallocr 的提前分配顺序
3. n_copies>1 时的多 copy 创建
```

它**不保证** tensor 的内存一直保留。

**为什么仍然安全**：

```text
1. 释放只发生在 reserve 的静态分析阶段，不改变 tensor->data
2. 后续没有新的分配（图只有 1 个 node）
3. tensor_set 写的是 a/b 原来的地址，不会被覆盖
4. kernel 读的也是这些地址
```

**但这在真实模型中是危险的**：如果图里还有后续节点需要分配内存，`a`/`b` 的地址可能被复用，而 `tensor_set` 在 alloc 之后才写入，就会覆盖新 tensor。正确做法是：

```cpp
// 方案 A：把输入放在独立 buffer，不参与 gallocr 的复用
ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors_from_buft(ctx_in, buft);

// 方案 B：标记 OUTPUT，阻止 gallocr 释放（只适用于结果，不适用于输入）
ggml_set_output(result);

// 方案 C：确保输入在 compute 前写入，且 alloc 后不再发生新分配
//         （本例的情况，依赖图小这个隐含前提）
```

`ggml_set_input` 本身不会阻止释放；它只是让 scheduler 知道“这是外部输入”，从而做正确的 backend 归属和 copy 规划。这是 ggml 内存模型里最容易误解的一点。

本例没有标记，恰好因为图小、无后续分配而正确工作。这是一个很好的“最小示例的隐含假设”案例。

### 3.12 内存复用算法

> **实验验证**：[04-gdb-trace.md](04-gdb-trace.md) §13 用三层 MLP 记录了完整的 ALLOC/FREE 时序、块合并、尾部块保留与 inplace 复用（`t0`/`s0` 同地址）。

gallocr 的复用不是引用计数，而是**静态生命周期分析**：

```text
hash_node {
    int n_children;   // 被多少 node 消费
    int n_views;      // 有多少 view
    int buffer_id;
    buffer_address addr;
    bool allocated;
}
```

流程：

```text
1. 分配 leaf
2. 统计所有 src 的 n_children、所有 view 的 n_views
3. 按 node 顺序：
     allocate src（如果还没分配）
     allocate node
     for each src:
         src.n_children--
         if src.n_children == 0 && src.n_views == 0:
             free_node(src)   → 加入 free list
4. 后续 allocate 从 free list 按 best-fit 取块
```

三种复用：

| 复用方式 | 条件 | 代码 |
|---|---|---|
| free-list 复用 | tensor 生命周期结束，后续 tensor 需要空间 | `ggml_dyn_tallocr_alloc` best-fit |
| inplace 复用 | 输出与某个输入 layout 相同，且该输入只有一个消费者 | `ggml_op_can_inplace` + `ggml_are_same_layout` |
| view 共享 | `view_src` 指向已有 tensor | `ggml_backend_view_init` |

`MUL_MAT` 的输出 layout 与输入不同（`[M,N]` vs `[K,M]`），不能 inplace。`ADD`、`MUL`、`SCALE` 等 element-wise op 可以。

反事实：三层图

```text
t0 = mul_mat(a, b)    // 128 B
t1 = add(t0, c)       // 128 B
t2 = silu(t1)         // 128 B

t0 被 t1 消费后 n_children=0 → free
t2 可以复用 t0 的块
峰值 = 2 × 128 = 256 B（而不是 384 B）
```

这就是 gallocr 对 LLM 的价值：几百层 transformer，峰值只取决于同时存活的 tensor，而不是总 tensor 数。

---

## 4 `ggml_backend_tensor_set()`：H2D

### 4.1 通用入口

```cpp
void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data,
                             size_t offset, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL);
    GGML_ASSERT(tensor->data != NULL);
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}
```

**走 buffer interface，不是 backend interface**：

```text
buffer->iface.set_tensor     同步语义（CUDA 内部 sync）
backend->iface.set_tensor_async  异步语义（调用者负责同步）
```

这是设计上的分工：buffer 知道“这块内存怎么读写”，backend 知道“这条 stream 怎么排队”。

### 4.2 断点与数据

```text
[B1] tensor_set: tensor=leaf_0 ne=[2,4] nb=[4,8] offset=0 size=32
     data=0x7f5600c00000 buffer=0x556de43981d0 buffer->size=384
[B2] cuda_buffer_set_tensor: dst=0x7f5600c00000 src=0x556de3167020 device=0

[B1] tensor_set: tensor=leaf_1 ne=[2,3] nb=[4,8] offset=0 size=24
     data=0x7f5600c00080 buffer=0x556de43981d0 buffer->size=384
[B2] cuda_buffer_set_tensor: dst=0x7f5600c00080 src=0x556de3167040 device=0
```

| 值 | 含义 |
|---|---|
| `dst=base+0x000` | `a` 在显存中的地址 |
| `dst=base+0x080` | `b` 在显存中的地址 |
| `size=32` | `2*4*4` |
| `size=24` | `2*3*4` |
| `src=matrix_A` | host 全局数组 |

### 4.3 CUDA 实现

```cpp
static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor * tensor,
                                                const void * data,
                                                size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size,
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```

三个设计点：

1. **`cudaStreamPerThread`**：per-thread default stream，不是 `ctx->stream()`。
2. **立即同步**：保证 `tensor_set` 返回时数据已到显存。
3. **pageable host 内存**：`matrix_A` 是普通全局数组，`cudaMemcpyAsync` 对 pageable 内存实际上会走内部暂存 + 同步，异步收益有限。

### 4.4 为什么同步

```text
正确性：tensor_set 返回后，调用者可能立即修改 matrix_A；
        如果拷贝还在异步进行，会读到新数据。
简单性：不需要调用者管理 event。
```

代价：

```text
每次 tensor_set 都有一次 host 等待
真实推理中，输入拷贝通常很小，影响不大；
但大权重加载时，应使用 pinned + 异步 + 批量。
```

### 4.5 与 kernel stream 的关系

```text
tensor_set  : cudaStreamPerThread
kernel      : ctx->stream()（cudaStreamNonBlocking）
```

两个不同 stream。正确性由“拷贝后立即同步 + host 侧顺序”保证：

```text
tensor_set(a) 返回 → a 已在显存
tensor_set(b) 返回 → b 已在显存
graph_compute → kernel launch 在 ctx->stream()
```

没有 event，因为 `tensor_set` 已经同步，不存在跨 stream 依赖。

---

## 5 `ggml_backend_sched_graph_compute()`：提交

```cpp
enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched,
                                                  struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}
```

断点：

```text
[C1] sched_graph_compute: n_splits=1 is_alloc=1 cur_copy=0
```

### 5.1 `compute_splits`

```cpp
for (int split_id = 0; split_id < sched->n_splits; split_id++) {
    struct ggml_backend_sched_split * split = &splits[split_id];
    ggml_backend_t split_backend = sched->backends[split->backend_id];

    // 上一个 split 的异步工作完成
    if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
        if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(...);
        } else {
            ggml_backend_synchronize(sched->backends[prev_backend_id]);
        }
    }

    // 跨 backend 输入拷贝
    for (int input_id = 0; input_id < split->n_inputs; input_id++) { ... }

    // 提交
    ggml_backend_graph_compute_async(split_backend, &split->graph);
}
```

本例：

```text
n_splits = 1
→ 无 split 间同步
→ n_inputs = 0，无拷贝
→ 直接提交 split[0].graph
```

断点：

```text
[C2] split[0]: backend_id=0 n_nodes=1 n_inputs=0 graph.nodes=1 graph.leafs=0
```

**设计意图**：scheduler 只负责“切分 + 同步 + 分发”，不实现任何算子。它把每个 split 变成一个独立的 `ggml_cgraph`，交给 backend 的 `graph_compute`。这样：

```text
1. backend 可以用自己的方式执行（CPU 线程池、CUDA stream、Metal command buffer）
2. scheduler 不需要知道算子语义
3. 跨 backend 的同步逻辑集中在一处
```

### 5.2 通用分发

```cpp
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend,
                                                  struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}
```

断点：

```text
[C3] backend=0x56469a662030 device=0x56469a664560 cgraph nodes=1 leafs=0
     node[0]=node_0 op=MUL_MAT data=0x7f8704c00100 buffer=0x56469af401d0
```

---

## 6 `ggml_backend_cuda_graph_compute()`：CUDA 后端入口

```cpp
static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend,
                                                        ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    ... // capture / warmup / instantiate
#endif

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph,
                                         cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}
```

断点：

```text
[C4] cgraph nodes=1 leafs=0 device=0
node[0]=node_0 op=MUL_MAT ne=[4,3] nb=[4,16] data=0x7f8704c00100
src0=leaf_0 src0.data=0x7f8704c00000
src1=leaf_1 src1.data=0x7f8704c00080
```

### 6.1 `ggml_cuda_set_device()`

```text
每次进入 CUDA backend 都 cudaSetDevice(device)
```

作用：保证后续 stream、cuBLAS、kernel 都在正确设备上。单卡时是 no-op；多卡时是必要的上下文切换。

### 6.2 CUDA graph 未启用

构建配置：

```text
GGML_CUDA_GRAPHS = OFF
```

断点：

```text
[C5] use_cuda_graph=0 update_required=0 nodes=1 device=0
```

CUDA graph 的设计意图：

```text
把一串 kernel launch 录制成一张图，之后一次 cudaGraphLaunch 提交
减少 CPU 端 launch 开销（每个 kernel 约几微秒）
适合：kernel 多、图结构稳定、重复执行的场景（LLM decode）
不适合：单 kernel、图结构变化、需要 CPU 干预的场景
```

本例只有 1 个 kernel，开启 CUDA graph 反而增加 capture/instantiate 开销。

### 6.3 Node 循环

```cpp
for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor * node = cgraph->nodes[i];

    if (ggml_op_is_empty(node->op)) {
        continue;
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        continue;
    }

    int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
    if (nodes_to_skip != 0) {
        i += nodes_to_skip;
        continue;
    }

#ifndef NDEBUG
    assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ...);
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        if (node->src[j] != nullptr) {
            assert(node->src[j]->buffer);
            assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(...) || ...);
        }
    }
#endif

    bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
    GGML_ASSERT(ok);
}
```

三个检查：

```text
1. ggml_op_is_empty：跳过 NONE/VIEW 等空操作
2. COMPUTE 标志：跳过不需要计算的节点
3. Debug 断言：所有 tensor 必须在 CUDA buffer 上
```

第 3 点很重要：如果 scheduler 把某个 node 判给 CUDA，但它的 src 还在 CPU buffer，会在这里断言失败。scheduler 的 pass 5 已经保证不会发生（要么 copy，要么切 split）。

### 6.4 Fusion

`ggml_cuda_try_fuse()` 会尝试把连续节点融合成一个 kernel：

```text
MUL_MAT + ADD(bias) + GLU → 一个 fused kernel
MoE weighted reduction
top-k + softmax + argsort
gated delta net cache
```

本例单节点，没有 fusion。

**设计意图**：fusion 减少 kernel launch 和中间结果读写，是 decode 性能的关键。但它要求节点相邻、layout 兼容，所以放在 CUDA backend 内部，而不是 scheduler。

---

## 7 `ggml_cuda_compute_forward()`：算子分发

```cpp
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX: ...
        case GGML_OP_GET_ROWS: ...
        ...
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        ...
    }
}
```

断点：

```text
[D1] node=node_0 op=MUL_MAT device=0
dst ne=[4,3,1,1] nb=[4,16,48,48] data=0x7f8704c00100 type=f32
src0 type=f32 ne=[2,4] nb=[4,8] data=0x7f8704c00000
src1 type=f32 ne=[2,3] nb=[4,8] data=0x7f8704c00080
```

**设计意图**：`switch(op)` 是最直接的静态分发，编译期生成跳转表，零查找开销。相比虚函数或注册表：

```text
优点：快、简单、可内联
缺点：新增 op 必须改这个 switch（见 05-extension-guide.md）
```

---

## 8 `ggml_cuda_mul_mat()`：多路径选择

### 8.1 入口

```cpp
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
    ...
}
```

断点：

```text
[D2] device=0 src0.type=f32 src1.type=f32 dst.type=f32
src0 ne=[2,4,1,1] nb=[4,8]
src1 ne=[2,3,1,1] nb=[4,8]
dst  ne=[4,3,1,1] nb=[4,16]
src0 cont=1 cont2=1 | src1 cont=1 cont2=1 | dst cont=1
```

### 8.2 完整决策树

```cpp
// 1. Hadamard hint
const int32_t hint = ggml_get_op_params_i32(dst, 1);
if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
    return;
}

// 2. padding 清理安全性
const bool bad_padding_clear =
    ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
    && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0)
    && src0->view_src;

// 3. 类型不满足 → cuBLAS
if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
    return;
}

const int cc        = ggml_cuda_info().devices[ctx.device].cc;   // 890
const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size; // 32

// 4. MMVF：小 N
if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
    return;
}

// 5. 转置向量 + 大 batch → MMVF
if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ...) { ... }

// 6. MMF：非量化、大 N
if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, false)) {
    ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
    return;
}

// 7. MMVQ：量化、小 N
if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
    return;
}

// 8. MMQ：量化、大 N
if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, 0)) {
    ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
    return;
}

// 9. 兜底
ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
```

### 8.3 本例为什么命中 MMVF

`ggml_cuda_should_use_mmvf`（`mmvf.cu:792`）：

```cpp
if (src0_ne[0] % 2 != 0) return false;                 // 2 % 2 = 0 ✓
const size_t ts = ggml_type_size(type);                // 4
if (src0_nb[0] != ts) return false;                    // 4 == 4 ✓
for (i = 1..3) if (src0_nb[i] % (2*ts) != 0) return false;
                                                        // 8%8=0, 32%8=0, 32%8=0 ✓

case GGML_TYPE_F32:
    if (NVIDIA) {
        if (ampere_mma_available(cc)) return ne11 <= 3; // 3 <= 3 ✓
        if (cc >= TURING) return ne11 <= 4;
        return ne11 <= 3;
    }
```

`ampere_mma_available(890)`：

```cpp
GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE(800)
```

`GGML_CUDA_ARCHITECTURES=89`，所以最高编译架构是 890 ≥ 800，返回 true。

最终：

```text
should_use_mmvf = true → ggml_cuda_mul_mat_vec_f
```

### 8.4 路径切换的临界点

| 变化 | 结果路径 | 原因 |
|---|---|---|
| `ne11 = 4` | cuBLAS | `should_use_mmvf` 要求 `ne11 <= 3`；`should_use_mmf` 要求 `ne0 % 32 == 0`，本例 `ne0=2` 不满足 |
| `src0` 改成 Q4_0，`ne11=1` | MMVQ | 量化 + 小 batch |
| `src0` 改成 Q4_0，`ne11=64` | MMQ | 量化 + 大 batch |
| `src0` 改成 F16，`ne11=32`，`ne0` 是 64 的倍数 | MMF | 非量化 + 大 batch + 对齐（F16 的 `4/ts = 2`，要求 `ne0 % 64 == 0`） |
| `GGML_CUDA_FORCE_CUBLAS=1` | cuBLAS | 环境变量强制 |
| `src1` 不是 F32 | cuBLAS | 类型兜底 |
| `bad_padding_clear = true` | cuBLAS | 量化 compute buffer 的 padding 不能安全清理 |

> **实验验证**：`ne11 = 4` 这一行已在 [04-gdb-trace.md](04-gdb-trace.md) §11 实测：MMVF 不命中，落到 `cublasSgemm`，并首次创建 `cublasCreate_v2` 与 4 MiB workspace。

**设计哲学**：

```text
没有“万能最优 kernel”，只有“当前形态下最优”
- decode（N 小、K 大）：MMVF / MMVQ，内存带宽受限
- prefill（N 大、K 大）：MMF / MMQ，计算受限
- 量化权重：MMVQ / MMQ，减少带宽
- 特殊布局/兜底：cuBLAS
```

### 8.5 cuBLAS 路径（本例未走）

如果走 cuBLAS：

```cpp
static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, ...) {
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    }
    ...
    switch (compute_type) {
        case GGML_TYPE_F32: ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(...); break;
        ...
    }
}
```

关键调用（`ggml-cuda.cu:1546`）：

```cpp
cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            ne01, ne11, ne10,
            alpha, src0_ptr, s01,
                   src1_ptr, s11,
            beta,  dst_ptr, ne0);
```

参数含义：

```text
m = ne01 = 4       (a 的行数 = 输出行数)
n = ne11 = 3       (b 的行数 = 输出列数)
k = ne10 = 2       (内积维度)
lda = s01 = 2      (a 的行步长，单位元素)
ldb = s11 = 2      (b 的行步长)
ldc = ne0 = 4      (输出行步长)
op(A) = T, op(B) = N
```

**为什么 `op(A)=T`**：ggml 是行主序，cuBLAS 是列主序。把 ggml 的 `a[K,M]` 按列主序看，等价于 `a^T[M,K]`；所以 `CUBLAS_OP_T` 正好还原成 `M×K` 矩阵。

cuBLAS 首次调用会创建：

```text
cublasCreate
cublasSetMathMode(TF32_TENSOR_OP_MATH)
cublasSetStream
cudaMalloc(4 MiB)   // cc < Hopper；Hopper 为 32 MiB
cublasSetWorkspace
```

本例 MMVF 不需要这些，所以 `cublas_handles[0][0] == NULL`、`pools[0][0] == NULL`。

---

## 9 MMVF kernel：逐参数拆解

### 9.1 入口

```cpp
void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx,
                             const ggml_tensor * src0,
                             const ggml_tensor * src1,
                             const ggml_tensor * ids,
                             ggml_tensor * dst,
                             const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);   // 4
    ...
    const int64_t s01 = src0->nb[1] / ts_src0;           // 8/4 = 2
    const int64_t s11 = src1->nb[1] / ts_src1;           // 8/4 = 2
    const int64_t s1  = dst->nb[1] / ts_dst;             // 16/4 = 4
    ...
    const int64_t ncols_dst = ids ? ne2 : ne1;           // 3
}
```

### 9.2 模板实例

```text
ggml_cuda_mul_mat_vec_f
  → mul_mat_vec_f_cuda<float>
    → mul_mat_vec_f_cuda_switch_ncols_dst<float, float>
      → ncols_dst == 3
        → launch_mul_mat_vec_f_cuda<float, float, 3, false>
          → mul_mat_vec_f_switch_fusion<float, float, 3, 32, false>
            → ggml_cuda_kernel_launch(mul_mat_vec_f<float,float,3,32,false,false>, ...)
```

模板参数：

| 参数 | 值 | 作用 |
|---|---|---|
| `T` | float | src0 元素类型 |
| `type_acc` | float | 累加类型 |
| `ncols_dst` | 3 | 输出列数，编译期展开 `sumf[3]` |
| `block_size` | 32 | 线程数，编译期确定循环步长 |
| `has_fusion` | false | 是否有 bias/gate |
| `is_multi_token_id` | false | 是否 MUL_MAT_ID 多 token |

### 9.3 Launch 参数推导

断点：

```text
[H1] ncols(ne00)=2 nrows(ne01)=4 ncols_dst=3 block_size_best=32
     block_nums=(4,1,1) block_dims=(32,1,1) nbytes_shared=128
     warp_size=32 has_fusion=0
```

代码（`mmvf.cu:418`）：

```cpp
int64_t block_size_best = warp_size;                                  // 32
int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);  // (2+63)/64 = 1

for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
    const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
    if (niter < niter_best) {
        niter_best      = niter;
        block_size_best = block_size;
    }
}

const int nbytes_shared = warp_size*sizeof(float)
                        + (has_fusion ? warp_size*sizeof(float) : 0);  // 128

const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);     // (4,1,1)
const dim3 block_dims(block_size_best, 1, 1);                         // (32,1,1)
```

逐项：

```text
ncols = ne00 = 2          内积维度 K
nrows = ne01 = 4          输出行数 M
nchannels_dst = 1         batch/channel
nsamples = 1              sample
```

**为什么 block_size 是 32**：`ncols=2`，每个线程处理 `float2`（2 个元素），32 个线程一轮就能覆盖 64 个元素，远大于 K=2。更大的 block 不会减少迭代次数（已经是 1），所以保留最小值 32。

**为什么 shared=128 但实际不用**：`buf_iw` 只在 `block_size > warp_size` 时用于跨 warp 归约。本例 `block_size == warp_size`，编译期分支不执行，但动态共享内存仍然按 128 B 申请。

### 9.4 Grid/Block 语义

```cpp
const int row         = blockIdx.x;    // 0..3，输出行
const int channel_dst = blockIdx.y;    // 0
const int tid         = threadIdx.x;   // 0..31
```

每个 block 负责一行输出：

```text
block 0 → result[0][0..2]
block 1 → result[1][0..2]
block 2 → result[2][0..2]
block 3 → result[3][0..2]
```

### 9.5 计算核心

```cpp
const float2 * x2 = (const float2 *) x;   // a 的一行
const float2 * y2 = (const float2 *) y;   // b 的所有行

for (int col2 = tid; col2 < ncols2; col2 += block_size) {
    const float2 tmpx = x2[col2];
    for (int j = 0; j < ncols_dst; ++j) {
        const float2 tmpy = y2[j*stride_col_y2 + col2];
        ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
    }
}
```

本例：

```text
ncols2 = ncols/2 = 1
→ 只有 tid=0 进入循环
→ x2[0] = {a[row][0], a[row][1]}
→ j=0: y2[0] = {b[0][0], b[0][1]}
   j=1: y2[1] = {b[1][0], b[1][1]}
   j=2: y2[2] = {b[2][0], b[2][1]}
→ sumf[j] = a[row] · b[j]
```

31 个线程的 `sumf[j]` 都是 0，但仍参与 warp reduction。

### 9.6 归约与写回

```cpp
for (int j = 0; j < ncols_dst; ++j) {
    sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
}

if (tid >= ncols_dst) {
    return;
}

dst[tid*stride_col_dst + row] = sumf[tid];
```

本例：

```text
stride_col_dst = s1 = 4
ncols_dst = 3

tid=0 → dst[0*4 + row] = result[row][0]
tid=1 → dst[1*4 + row] = result[row][1]
tid=2 → dst[2*4 + row] = result[row][2]
```

与 `result->nb = {4,16,48,48}` 一致：列步长 16 B = 4 个 float。

### 9.7 性能分析

```text
总线程数 = grid.x * block.x = 4 * 32 = 128
有效计算线程 = 4（每个 block 的 tid=0）
有效乘加 = 4 行 × 3 列 × 2 维 = 24 次
```

利用率极低，但这是**小矩阵的固有形态**：K=2 没有并行度。真正的收益是：

```text
1. 一个 kernel launch 完成整个 MUL_MAT，没有 cuBLAS handle/workspace 开销
2. float2 加载，减少指令数
3. ncols_dst 编译期展开，3 个累加器常驻寄存器
4. warp reduction 不需要 shared memory
```

在真实 LLM decode 中：

```text
K = 4096（hidden size），M = 4096（输出维度），N = 1（单 token）
每个 block 处理一行，32 个线程各处理 4096/32 = 128 个元素（64 个 float2）
内存带宽受限，MMVF 接近峰值
```

### 9.8 如果要优化这个小例子

```text
1. 换更大矩阵（K/M/N 都放大），否则 launch 开销主导
2. 用 pinned host memory + 异步拷贝，减少 tensor_set 的同步
3. 用 CUDA graph 合并多次 launch（需要循环 compute）
4. 量化权重（Q4_0/Q8_0）→ MMVQ/MMQ，减少带宽
5. 把 a/b 预先放在显存，避免每轮 H2D
6. 合并多个小矩阵乘（batching）
```

---

## 10 同步与回读

### 10.1 `sched_synchronize`

```cpp
void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
}
```

CUDA：

```cpp
cudaStreamSynchronize(cuda_ctx->stream());
```

断点：

```text
[I3] stream=0x55b54372f9f0
#0 ggml_backend_cuda_synchronize
#1 ggml_backend_synchronize
#2 ggml_backend_sched_synchronize
#3 ggml_backend_sched_graph_compute
```

**语义**：host 阻塞，直到 `ctx->stream()` 上所有 kernel 完成。

### 10.2 `tensor_get`

```cpp
static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor * tensor,
                                                void * data, size_t offset, size_t size) {
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size,
                               cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```

断点：

```text
[E1] tensor=node_0 ne=[4,3] offset=0 size=48 data=0x7f8704c00100
[E2] src=0x7f8704c00100 dst=0x56469c2647f0 size=48 device=0
```

结果：

```text
[ 60 55 50 110
  90 54 54 126
  42 29 28  64 ]
```

验证：

```text
a = [2 8; 5 1; 4 2; 8 6]      (4x2)
b = [10 5; 9 9; 5 4]          (3x2)
a * b^T:
row0: [2*10+8*5, 2*9+8*9, 2*5+8*4] = [60, 90, 42]
row1: [5*10+1*5, 5*9+1*9, 5*5+1*4] = [55, 54, 29]
row2: [4*10+2*5, 4*9+2*9, 4*5+2*4] = [50, 54, 28]
row3: [8*10+6*5, 8*9+6*9, 8*5+6*4] = [110, 126, 64]
```

输出按列主序打印（`out_data[j * ne0 + i]`），所以看到的矩阵是转置的：

```text
[60 55 50 110]
[90 54 54 126]
[42 29 28  64]
```

---

## 11 资源释放

### 11.1 顺序

```cpp
ggml_backend_sched_free(model.sched);    // 先释放 buffer
ggml_backend_free(model.backend);        // 再释放 CUDA context
ggml_backend_free(model.cpu_backend);
```

断点：

```text
[F3] sched_free: n_splits=1
     galloc buffers[0]=0x56469af410c0 buffers[1]=0x56469af40b90
[F1] cuda_buffer_free_buffer: buffer=0x56469af401d0 size=384 dev_ptr=0x7f8704c00000
     #2 ggml_vbuffer_free
     #3 ggml_gallocr_free
     #4 ggml_backend_sched_free
[F2] backend_cuda_free: backend=0x56469a662030 device=0
     #1 ggml_backend_free
```

### 11.2 CUDA buffer 释放

```cpp
struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr;

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_backend_cuda_buffer_context *)buffer->context;
}
```

`delete ctx` → `cudaFree(0x7f8704c00000)`。

### 11.3 Backend 释放

```cpp
static void ggml_backend_cuda_free(ggml_backend_t backend) {
    delete (ggml_backend_cuda_context *)backend->context;
    delete backend;
}
```

析构函数：

```cpp
~ggml_backend_cuda_context() {
    // 等待所有 capture 完成
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter == 0; });

    if (copy_event) cudaEventDestroy(copy_event);

    for (i, j) {
        if (streams[i][j])         cudaStreamDestroy(streams[i][j]);
        if (cublas_handles[i][j])  cublasDestroy(cublas_handles[i][j]);
        if (cublas_workspaces[i][j]) cudaFree(cublas_workspaces[i][j]);
    }
}
```

本次：

```text
streams[0][0]        → 销毁
cublas_handles[0][0] = NULL → 跳过
cublas_workspaces    = NULL → 跳过
```

### 11.4 为什么顺序不能反

```text
buffer 的 cudaFree 需要 device 上下文有效
backend 的析构会销毁 stream/cuBLAS
如果先 backend_free，再 sched_free：
    buffer 仍然持有 dev_ptr，但 stream 已销毁
    cudaFree 本身不需要 stream，通常仍能工作，
    但如果有异步操作未完成，行为未定义
正确顺序：先 buffer（数据），后 backend（执行环境）
```

---

## 12 设计总结

### 12.1 三层决策

```text
scheduler：拓扑 + 能力 → 哪个 backend
gallocr  ：生命周期 + buft → 哪块内存、哪个 offset
backend  ：形态 + 类型 → 哪个 kernel
```

### 12.2 本例的完整决策链

```text
a,b,result 无 buffer、无 INPUT
  → pass 3：CUDA 和 CPU 都支持 MUL_MAT，n_supported 都是 0
  → CUDA 下标小，严格大于 → CUDA0
  → pass 4：a,b → CUDA0
  → pass 5：n_splits=1，n_inputs=0

gallocr：
  buft CUDA0，alignment 128
  a: 32→128 @ 0
  b: 24→128 @ 128
  result: 48→128 @ 256
  chunk.max_size = 384 → cudaMalloc(384)
  a,b 在 node_0 后释放 → free list {0,256},{384,...}

CUDA backend：
  F32/F32/F32，cc=890，ne11=3
  → MMVF
  → grid=(4,1,1)，block=(32,1,1)，shared=128
  → float2 + warp reduce
  → 4 个 block，每 block 32 线程
```

### 12.3 扩展钩子（详见 05）

| 想改什么 | 改哪里 |
|---|---|
| 新增算子 | `ggml.h` 枚举、`ggml.c` 构造函数、`ggml_cuda_compute_forward` case、`supports_op`、kernel |
| 新增 kernel 路径 | `ggml_cuda_mul_mat` 的决策树 + `should_use_*` |
| 新增后端 | `ggml_backend_reg_i / device_i / backend_i` |
| 新增内存类型 | `ggml_backend_buffer_type_i` |
| 调度策略 | `backends[]` 顺序、`op_offload`、`parallel` |
| 内存优化 | `ggml_set_input/output`、`ggml_gallocr_reserve`、alignment、inplace |
| 性能 | pinned memory、CUDA graph、fusion、量化、batch |

### 12.4 业务映射

| 真实场景 | 本例中的对应物 | 关键机制 |
|---|---|---|
| decode（逐 token） | `ne11 = 3` 的小 N | MMVF，内存带宽受限 |
| prefill（整段 prompt） | `ne11` 很大 | MMF / MMQ，计算受限 |
| 权重常驻 | `a`、`b` 常驻显存 | `tensor_set` 只在初始化时调用 |
| 激活复用 | free list `{0,256}` | `n_children` 生命周期分析 |
| 层 offload | 多 split + `n_inputs>0` | 跨 backend copy + event |
| CPU 权重按 batch 提升 | `op_offload` | `offload_op` + 阈值 |
| KV cache 更新 | 输入 tensor 每轮重写 | `ggml_set_input` + `tensor_set` |
| logits 回读 | `result` 的 D2H | `tensor_get` |
| 逐层调试 | eval callback | 分段执行，牺牲整图优化 |
| 显存预算 | 384 B 的 chunk | gallocr reserve + alignment |
| 多流流水线 | `n_copies=1`（本例未用） | `parallel=true` + event |

**一句话**：真实 LLM 只是把本例的“1 个 MUL_MAT、3 个 tensor、1 个 split”放大成“几百个算子、上千个 tensor、多个 split”，但 `split → alloc → H2D → kernel → sync → D2H` 的骨架完全不变。

---

## 13 性能优化清单（从本例到真实模型）

### 13.1 本例的开销

```text
cudaMalloc 384 B              一次
cudaMemcpyAsync H2D 32 B      + sync
cudaMemcpyAsync H2D 24 B      + sync
kernel launch                 一次（4 个 block）
cudaStreamSynchronize         一次
cudaMemcpyAsync D2H 48 B      + sync
```

固定开销远大于 24 次乘加。

### 13.2 真实模型的优化顺序

```text
1. 权重常驻显存（不要每轮 H2D）
2. 输入用 pinned memory + 异步拷贝
3. 用 CUDA graph 消除 launch 开销（decode 场景）
4. 量化权重（Q4_0/Q8_0）→ MMVQ/MMQ
5. 合并小 kernel（fusion）
6. 多 stream 并行（n_copies > 1 / 多 backend）
7. 调整 gallocr reserve，避免每轮 realloc
8. 用 GGML_SCHED_DEBUG / nsys / ncu 定位瓶颈
```

### 13.3 什么时候不该用 GPU

```text
1. 矩阵太小（本例）：launch + 拷贝开销 > 计算
2. 频繁 host-device 同步：tensor_set/get 的 sync
3. 图结构频繁变化：CUDA graph 失效
4. 显存不足：频繁 realloc / OOM
```

---

## 14 源码逐行：执行链路

本节按 `compute()` 的实际执行顺序，逐行拆解调度、分配、拷贝、计算、回读、释放六段代码。行号对应云端 `/home/ggml`（commit `353b63b`）。

### 14.1 `ggml_backend_sched_reset()`

`src/ggml-backend.cpp:1949`

```cpp
void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1951-1954 | `if (!is_reset)` | **只有“上一轮 split 过”才需要清空**。首次调用时 `calloc` 已经把数组清零，`is_reset=0` 会执行一次 |
| 1955 | `ggml_hash_set_reset` | 清 `used` 位图；`keys[]` 不清理（内容无意义） |
| 1956 | `memset(hv_tensor_backend_ids, -1, ...)` | **-1 表示“未分配”**，这是 pass 1 的判定依据 |
| 1957 | `memset(hv_tensor_copies, 0, ...)` | copy 指针清空 |
| 1958 | `is_reset = true` | 标记“可以开始新一轮 split” |
| 1960 | `is_alloc = false` | **无条件**：上一轮的 buffer 归属可能失效，必须重新 alloc |

**为什么 reset 不释放 buffer**：

```text
buffer 由 gallocr 持有，跨多次 compute 复用
reset 只重置“调度状态”，不动内存
真正的释放发生在 gallocr_free（sched_free 时）
```

**为什么 `is_alloc=false` 在 if 外面**：

```cpp
if (!sched->is_reset) { ...; sched->is_reset = true; }
sched->is_alloc = false;   // 无论是否刚 reset，都要失效
```

因为即使 `is_reset` 已经是 true（同一轮内多次 reset），上一轮 alloc 的地址也可能因为图变化而失效。

### 14.2 `ggml_backend_sched_alloc_graph()`

`src/ggml-backend.cpp:1992`

```cpp
bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1994 | `hash_set.size >= n_nodes + n_leafs` | 哈希表必须能装下所有 tensor，否则 `hash_find_or_insert` 会满 |
| 1995 | `!is_alloc` | 不允许重复 alloc；重复调用必须先 reset |
| 1997-1998 | copy 轮转 | `n_copies=1` 时 `cur_copy = next_copy = 0`，无实际作用 |
| 2000 | `split_graph` | 决策 + 切分 + 插入 copy |
| 2002-2004 | `alloc_splits` | 分配显存 |
| 2006 | `is_alloc = true` | 标记“已分配”，`graph_compute_async` 会跳过再次分配 |

**为什么先 split 再 alloc**：

```text
alloc 需要知道每个 tensor 的 buffer_id
buffer_id 由 split_graph 的 pass 3/4/5 决定
所以顺序不能反
```

### 14.3 `ggml_backend_sched_split_graph()`：五个 pass

`src/ggml-backend.cpp:1066`

#### 14.3.1 初始化（1066-1085）

```cpp
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1068-1069 | 清零 split 计数 | 每轮重新切分 |
| 1071-1075 | 在 `context_buffer` 上建 context | **复用同一块 41 MiB 内存**，不重新 malloc |
| 1077 | `ggml_free(sched->ctx)` | 释放上一轮的 context 结构（外部 buffer，不释放 `context_buffer`） |
| 1079 | `ggml_init(params)` | 重建 context，对象链表归零 |
| 1084 | `graph->uid = ...` | 给图分配唯一 id，用于 CUDA graph 的 key |

#### 14.3.2 pass 1：预分配优先（1087-1121）

```cpp
for (int i = 0; i < graph->n_leafs; i++) {
    struct ggml_tensor * leaf = graph->leafs[i];
    int * leaf_backend_id = &tensor_backend_id(leaf);
    // do not overwrite user assignments
    if (*leaf_backend_id == -1) {
        *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1091 | `if (*leaf_backend_id == -1)` | **不覆盖用户显式赋值**。用户可能通过 `ggml_backend_sched_set_tensor_backend` 指定 |
| 1092 | `backend_id_from_cur` | 判定顺序：已有 buffer → view_src buffer → INPUT 标志 → 权重 → -1。注意“已有 buffer”不是“用 buffer 所属 backend”，而是 `ggml_backend_sched_backend_from_buffer` 找**支持该 buft 且支持该 op 的最高优先级 backend** |

`ggml_backend_sched_backend_id_from_cur()`（921）的关键分支。先看它依赖的 `ggml_backend_sched_backend_from_buffer()`（888）：

```cpp
static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched,
                                                  const struct ggml_tensor * tensor,
                                                  const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, "
                   "the weight will need to be copied\n", ...);
    return -1;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 889 | view 用源的 buffer | view 不拥有内存 |
| 890-892 | 无 buffer → -1 | 交给后续 pass 处理 |
| 895-900 | **从 0 开始找第一个满足的 backend** | “第一个满足”= 最高优先级；同一 buft 被多个 backend 支持时（如 BLAS 和 CPU 都用 host 内存），排在前面者胜出 |
| 902-905 | 找不到时打警告 | 说明权重 buft 与所有 backend 都不兼容，后续会触发 copy |

再看 `backend_id_from_cur()` 的关键分支：

```cpp
// 1. tensor 已有 buffer
int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
if (cur_backend_id != -1) return cur_backend_id;

// 2. view_src 已有 buffer
if (tensor->view_src != NULL) {
    cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
    if (cur_backend_id != -1) return cur_backend_id;
}

// 3. 预分配但 backend 不支持 → abort
if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
    GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", ...);
}

// 4. INPUT 标志 → 最后一个 backend（CPU）
if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
    return sched->n_backends - 1;
}

// 5. src 有 WEIGHTS → 跟随权重
for (int i = 0; i < GGML_MAX_SRC; i++) {
    const struct ggml_tensor * src = tensor->src[i];
    if (src && src->buffer && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
        if (sched->op_offload && src_backend_id == sched->n_backends - 1
            && ggml_backend_buffer_is_host(src->buffer)) {
            for (int b = 0; b < src_backend_id; b++) {
                if (ggml_backend_supports_op(sched->backends[b], tensor)
                    && ggml_backend_offload_op(sched->backends[b], tensor)) {
                    return b;
                }
            }
        }
        return src_backend_id;
    }
}

return -1;
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 923-927 | 已有 buffer → `backend_from_buffer` | **不是“用 buffer 所属 backend”**，而是“支持该 buft 且支持该 op 的最高优先级 backend” |
| 930-936 | view_src 有 buffer → 同上 | view 必须与源在同一 buft 上 |
| 938-942 | 有 buffer 但无 backend 支持 → abort | 宁可崩溃也不静默产生错误结果 |
| 945-949 | INPUT → CPU | 输入默认留在 host，避免每轮 H2D |
| 962-977 | WEIGHTS → 跟随权重 | 权重在哪，计算尽量在哪 |
| 968-976 | `op_offload` 提升 | CPU 权重 + GPU 支持 + `offload_op` → 提升到 GPU |

#### 14.3.3 pass 2：拓扑扩散（1124-1201）

```cpp
// expand gpu down
{
    int cur_backend_id = -1;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) continue;
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id != -1) {
            if (*node_backend_id == sched->n_backends - 1) {
                cur_backend_id = -1;      // CPU 不扩散
            } else {
                cur_backend_id = *node_backend_id;
            }
        } else if (cur_backend_id != -1) {
            ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
        }
    }
}
```

四个循环（gpu down / gpu up / rest down / rest up）的区别：

| 循环 | 方向 | 是否跳过 CPU | 目的 |
|---|---|---|---|
| gpu down | 正向 | 是 | 从 GPU 节点向下扩散 |
| gpu up | 反向 | 是 | 从 GPU 节点向上扩散 |
| rest down | 正向 | 否 | 任意 backend 向下扩散 |
| rest up | 反向 | 否 | 任意 backend 向上扩散 |

`ggml_backend_sched_set_if_supported()`（1058）：

```cpp
static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
    }
}
```

**为什么 CPU 不参与 gpu 扩散**：CPU 是兜底，如果让它扩散，几乎每个节点都会被 CPU “感染”，GPU 就永远用不上。

#### 14.3.4 pass 3：能力 + 优先级（1204-1263）

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    if (ggml_is_view_op(node->op)) continue;
    int * node_backend_id = &tensor_backend_id(node);
    if (*node_backend_id == -1) {
        int n_supported_best = -1;
        for (int b = 0; b < sched->n_backends; b++) {
            if (ggml_backend_supports_op(sched->backends[b], node)) {
                int n_supported = 0;
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) continue;
                    if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1)
                        && ggml_backend_sched_buffer_supported(sched, src, b)) {
                        n_supported++;
                    }
                }
                if (n_supported > n_supported_best) {
                    n_supported_best = n_supported;
                    *node_backend_id = b;
                }
            }
        }
    } else {
        // 已分配节点：尝试升级到更高优先级
        for (int b = 0; b < *node_backend_id; b++) {
            if (sched->bufts[b] == sched->bufts[*node_backend_id]
                && ggml_backend_supports_op(sched->backends[b], node)) {
                bool supported = true;
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) continue;
                    if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                        supported = false;
                        break;
                    }
                }
                if (supported) {
                    *node_backend_id = b;
                    break;
                }
            }
        }
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1209 | `if (*node_backend_id == -1)` | 只处理 pass 2 没分配的节点 |
| 1212 | `n_supported_best = -1` | 初始值必须小于任何可能的 `n_supported`（0 起） |
| 1214 | `supports_op` | **能力粗筛**：类型、布局、设备 |
| 1216-1224 | 统计兼容的 src 数 | `tensor_backend_id(src) != -1` 要求 src 已分配；`buffer_supported` 要求 buft 兼容 |
| 1225 | `n_supported > n_supported_best` | **严格大于**，所以相同分数时保留下标更小的 backend |
| 1233-1258 | 已分配节点升级 | 如果两个 backend 用同一个 buft，且高优先级也支持，就升级 |

**本例的关键**：

```text
node_0: CUDA 和 CPU 都 supports_op = true
        src a/b 的 backend_id 都是 -1 → n_supported = 0
        b=0: 0 > -1 → 选中 CUDA0，best=0
        b=1: 0 > 0  → 不成立
```

**结论：CUDA 胜出靠的是“下标小 + 严格大于”，不是“CUDA 更快”**。

#### 14.3.5 pass 4：src 跟随 dst（1265-1295）

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    int * cur_backend_id = &tensor_backend_id(node);

    if (node->view_src != NULL && *cur_backend_id == -1) {
        *cur_backend_id = tensor_backend_id(node->view_src);
    }

    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * src = node->src[j];
        if (src == NULL) continue;
        int * src_backend_id = &tensor_backend_id(src);
        if (*src_backend_id == -1) {
            if (src->view_src != NULL) {
                *src_backend_id = tensor_backend_id(src->view_src);
            } else {
                *src_backend_id = *cur_backend_id;
            }
        }
    }

    // 仍未分配的 node：第一个支持的 backend
    for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
        ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
    }
    GGML_ASSERT(*cur_backend_id != -1);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1270-1272 | view 跟随 view_src | view 不能跨 backend |
| 1275-1285 | src 跟随 dst | **数据去计算所在的地方**，而不是计算去数据所在的地方 |
| 1287-1290 | 兜底 | 如果 node 还没分配，选第一个支持的 backend |
| 1291 | 断言 | 所有节点必须有归属，否则后面无法切 split |

#### 14.3.6 pass 5：切 split + 插入 copy（1297-1429）

```cpp
int i_split = 0;
struct ggml_backend_sched_split * split = &sched->splits[0];

int i = 0;
for (; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    if (!ggml_is_view_op(node->op)) {
        split->backend_id = tensor_backend_id(node);
        break;
    }
}
split->i_start = 0;
split->n_inputs = 0;
int cur_backend_id = split->backend_id;

for (; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    if (ggml_is_view_op(node->op)) continue;

    const int node_backend_id = tensor_backend_id(node);
    GGML_ASSERT(node_backend_id != -1);

    bool need_new_split = false;
    if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) continue;
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = tensor_backend_id(src);
                if (src_backend_id != cur_backend_id
                    && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    need_new_split = true;
                    break;
                }
            }
        }
    }

    if (node_backend_id != cur_backend_id || need_new_split) {
        split->i_end = i;
        i_split++;
        if (i_split >= sched->splits_capacity) {
            int old_cap = sched->splits_capacity;
            sched->splits_capacity *= 2;
            sched->splits = realloc(sched->splits, sched->splits_capacity * sizeof(...));
            for (int k = old_cap; k < sched->splits_capacity; k++) {
                memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
            }
        }
        split = &sched->splits[i_split];
        split->backend_id = node_backend_id;
        split->i_start = i;
        split->n_inputs = 0;
        cur_backend_id = node_backend_id;
    }

    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * src = node->src[j];
        if (src == NULL) continue;

        size_t src_id = hash_id(src);
        const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
        GGML_ASSERT(src_backend_id != -1);

        if (src_backend_id != cur_backend_id
            && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
            if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[cur_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                    ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                    if (sched->n_copies > 1) {
                        ggml_set_input(tensor_copy);
                        ggml_set_output(tensor_copy);
                    }
                    tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                }
                int n_inputs = split->n_inputs++;
                if (n_inputs >= split->inputs_capacity) {
                    ggml_backend_sched_split_inputs_grow(split);
                }
                split->inputs[n_inputs] = src;
            }
            node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
        }
    }
}
split->i_end = graph->n_nodes;
sched->n_splits = i_split + 1;
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1301-1308 | 找第一个非 view 节点 | **view 不决定 split 边界**，它跟随源 |
| 1310-1311 | 初始化 split[0] | `i_start=0`，`n_inputs=0` |
| 1319 | `node_backend_id != cur_backend_id` | **backend 变化 → 新 split** |
| 1322-1334 | `need_new_split` | 权重在不同 backend 且 buft 不兼容时，强制切分以便复用权重内存 |
| 1336-1353 | 关闭当前 split、开新 split | 倍增扩容，`memset` 新槽位 |
| 1357-1359 | `hash_id(src)` | 在 scheduler 哈希表中定位 src |
| 1363 | `src_backend_id != cur_backend_id` | 跨 backend |
| 1364 | `!buffer_supported` | buft 不兼容才需要 copy；如果 buft 相同（例如都是 host 内存），可以直接用 |
| 1366 | `tensor_id_copy(...) == NULL` | **每个 (tensor, backend, copy) 只创建一次 copy** |
| 1368-1375 | `ggml_dup_tensor_layout` | 创建同 shape 的新 tensor，名字 `CUDA0#leaf_0#0` |
| 1376-1379 | `n_copies > 1` 时标记 INPUT/OUTPUT | 多 copy 时每个副本独立保活 |
| 1380-1383 | 记录到 `split->inputs[]` | compute 时按这个列表拷贝 |
| 1385 | `node->src[j] = copy` | **改写节点输入**，后续计算直接用 copy |
| 1427-1428 | 收尾 | 最后一个 split 的 `i_end`，总数 |

**本例的路径**：

```text
所有 node/leaf backend_id = 0
→ 循环中 node_backend_id == cur_backend_id
→ 不切新 split
→ src_backend_id == cur_backend_id
→ 不创建 copy
→ n_splits = 1, split[0].n_inputs = 0
```

### 14.4 `ggml_backend_sched_alloc_splits()`

`src/ggml-backend.cpp:1591`

```cpp
static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph,
                                    sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1593-1602 | 比较 node backend id | **只有 backend 变了且 buft 也变了才需要 realloc**；同一 buft 的不同 backend（例如两个 CUDA 卡）可以复用 |
| 1603-1613 | 比较 leaf backend id | 同上 |
| 1616 | `backend_ids_changed \|\| !alloc_graph` | 两个触发条件：归属变化，或 alloc 失败 |
| 1618 | `GGML_LOG_DEBUG` | 就是日志 `failed to allocate graph, reserving` |
| 1625-1627 | **同步所有 backend** | reserve 可能释放旧 buffer，必须先等异步工作完成 |
| 1629 | `reserve_n` | 生命周期分析 + 计算峰值 + 申请 buffer |
| 1633 | `alloc_graph`（第二次） | 只写 `tensor->data`，不重新分析 |

**为什么第一次 `alloc_graph` 会失败**：

```cpp
ggml_gallocr_alloc_graph:
  if (ggml_gallocr_needs_realloc(galloc, graph)) {
      if (galloc->n_buffers == 1) { 自动 reserve }
      else { return false; }         ← 本例 n_buffers = 2
  }
```

`n_buffers = 2`（CUDA + CPU），不能自动 realloc，必须显式 `reserve_n`。

### 14.5 `ggml_gallocr_alloc_graph_impl()`：生命周期分析

`src/ggml-alloc.c:718`

```cpp
static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                                          const int * node_buffer_ids, const int * leaf_buffer_ids) {
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

    // allocate leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) continue;

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) continue;
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) continue;

            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_impl_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                } else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
        }
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 720-721 | 清空哈希 | 每轮重新统计 |
| 725-728 | 先分配所有 leaf | leaf 是数据源，必须最早有地址 |
| 739-742 | 统计 view | view 不拥有内存，但会阻止源被释放 |
| 744-746 | INPUT 节点提前分配 | 避免在第一次使用前被其他节点的输出占用 |
| 754 | `n_children += 1` | **每个 src 被引用一次就 +1** |
| 757-759 | INPUT src 提前分配 | 同上 |
| 764-766 | 按节点顺序遍历 | 拓扑序，保证依赖先分配 |
| 769-774 | 先分配 parent | parent 可能是上一个节点的输出，此时还没分配 |
| 776 | 再分配 node | node 自己的输出 |
| 781 | `n_children -= 1` | **消费后减一** |
| 783 | `n_children == 0 && n_views == 0` | **生命周期结束** |
| 785-793 | view 特殊处理 | view 归零后还要检查 view_src 的 `n_views` |
| 794-796 | 普通 tensor 释放 | `allocated` 为真才释放（外部 buffer 不释放） |

**本例的执行序列**：

```text
leafs: a, b
nodes: node_0

1. allocate a  → tallocr_alloc(32)  → aligned 128 → offset 0
2. allocate b  → tallocr_alloc(24)  → aligned 128 → offset 128
3. 统计: node_0 的 src a、b → n_children 各 +1
4. 遍历 node_0:
   - allocate a（已分配，跳过）
   - allocate b（已分配，跳过）
   - allocate node_0 → tallocr_alloc(48) → aligned 128 → offset 256
   - a.n_children-- → 0, n_views=0, allocated → free_node(a)  → free [0,128)
   - b.n_children-- → 0 → free_node(b) → 合并为 [0,256)
```

**为什么 `a`、`b` 被释放**：

```text
它们不是 OUTPUT
node_0 是唯一消费者
n_children 归零
```

**为什么仍然安全**：见 §3.11 的详细说明——释放只更新 free list，不改 `tensor->data`；本例 alloc 后没有新分配。

### 14.6 `ggml_dyn_tallocr_alloc()`：best-fit

`src/ggml-alloc.c:202`

```cpp
static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size,
                                                    const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    if (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunk's size)
        int64_t best_reuse = INT64_MIN;
        for (int c = 0; c < alloc->n_chunks; ++c) {
            struct tallocr_chunk * chunk = alloc->chunks[c];
            if (chunk->n_free_blocks > 0) {
                struct free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                if (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    if (best_fit_block == -1) {
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }

    struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    struct free_block    * block = &chunk->free_blocks[best_fit_block];
    struct buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset};
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 203 | `aligned_offset(NULL, size, alignment)` | 向上对齐到 128/32 |
| 211 | `i < n_free_blocks - 1` | **不搜索最后一个块**：它是“可增长尾部”，留给扩展 |
| 215-219 | best-fit | 选最小的足够块，减少碎片 |
| 226-247 | 尾部块兜底 | 前面的块都不够时，才考虑尾部；`reuse_factor` 衡量浪费 |
| 251-253 | 开新 chunk | 所有块都不够，创建新 chunk |
| 261-267 | 切分块 | `offset += size`、`size -= size`；用尽则删除 |
| 270 | `max_size = MAX(max_size, offset + size)` | **峰值记录**，最终决定物理 buffer 大小 |

**实验三验证**（[04-gdb-trace.md](04-gdb-trace.md) §13）：

```text
t1 需要 768 B
free list = [65536, 33536], [100608, huge]
非尾部块只有 [65536, 33536] → t1 @65536
尾部块 [100608, huge] 不参与 best-fit
```

这就是为什么 `t1` 没有落在更靠后的位置。

### 14.7 `ggml_dyn_tallocr_free_bytes()`：合并

`src/ggml-alloc.c:312`

```cpp
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];

        // 与后一个块相邻
        if (block->offset + block->size == addr.offset) {
            block->size += size;
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }

        // 与前一个块相邻
        if (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }

    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 313 | 对齐 | 与 alloc 一致，保证 free 的尺寸与占用一致 |
| 318-333 | 向后合并 | 释放的块紧接在某个块之后 → 扩展它，并尝试再合并下一个 |
| 336-350 | 向前合并 | 释放的块紧接在某个块之前 → 前移起点，并尝试合并前一个 |
| 353 | 都不相邻 → 插入 | 保持按 offset 排序，便于后续合并 |

**实验三验证**：

```text
FREE w0 @65536 (32768)
FREE x  @98304 (768)
→ x 紧接在 w0 之后 → 合并为 [65536, 33536]
```

### 14.8 `ggml_backend_tensor_set()` 与 CUDA 实现

`src/ggml-backend.cpp:335`

```cpp
void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 337 | `view_src ? view_src->buffer : tensor->buffer` | **view 没有自己的 buffer**，用源的 |
| 338 | `buf != NULL` | 防止未分配就写 |
| 341-343 | `size == 0` 直接返回 | 空 tensor 是合法的 |
| 345 | `tensor->data != NULL` | alloc 之后才允许写 |
| 346 | 越界检查 | `offset + size <= ggml_nbytes` |
| 348 | **`buf->iface.set_tensor`** | 走 **buffer** 接口，不是 backend 接口 |

CUDA 实现（`ggml-cuda.cu:785`）：

```cpp
static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size,
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 789 | `ggml_cuda_set_device(ctx->device)` | 多卡时必须切到正确设备 |
| 790-791 | `cudaMemcpyAsync(..., cudaStreamPerThread)` | **per-thread default stream**，不是 `ctx->stream()` |
| 792 | `cudaStreamSynchronize(cudaStreamPerThread)` | **立即同步**，保证返回时数据已到显存 |

**为什么用 `cudaStreamPerThread`**：`tensor_set` 是同步语义的公开 API，调用者不管理 stream；用 per-thread stream + 立即同步，语义最简单。

**为什么 kernel 用 `ctx->stream()`**：计算需要与 backend 的 stream 绑定，以便 cuBLAS、event、CUDA graph 都在这条 stream 上。

### 14.9 `ggml_backend_cuda_graph_compute()`

`src/ggml-cuda/ggml-cuda.cu:4425`

```cpp
static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    ... // capture / warmup / instantiate
#endif

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph,
                                         cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 4426 | 取 `cuda_ctx` | 执行状态（stream、pool、cublas） |
| 4428 | `ggml_cuda_set_device` | 每次进入都切设备，不依赖调用者 |
| 4430-4432 | 三个 CUDA graph 变量 | `USE_CUDA_GRAPH` 未定义时保持 false |
| 4436-4440 | 捕获/执行 | 未启用 CUDA graph 时直接执行 |

`ggml_cuda_graph_evaluate_and_capture()`（4185）的核心循环：

```cpp
while (!graph_evaluated_or_captured) {
    if (!use_cuda_graph || cuda_graph_update_required) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];

            if (ggml_op_is_empty(node->op)) {
                continue;
            }

            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }

            int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
            if (nodes_to_skip != 0) {
                i += nodes_to_skip;
                continue;
            }

            bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
            GGML_ASSERT(ok);
        }
        graph_evaluated_or_captured = true;
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 4218 | `!use_cuda_graph \|\| update_required` | 捕获阶段也要真正执行一次 |
| 4221 | `ggml_op_is_empty` | 跳过 NONE/VIEW 等空操作 |
| 4225 | `COMPUTE` 标志 | 跳过不需要计算的节点 |
| 4230 | `ggml_cuda_try_fuse` | 尝试融合，返回要跳过的节点数 |
| 4231-4235 | 跳过已融合节点 | `i += nodes_to_skip` |
| 4237 | `ggml_cuda_compute_forward` | 真正的算子分发 |
| 4238 | `GGML_ASSERT(ok)` | 不支持的 op 直接崩溃，而不是静默跳过 |

**Debug 断言**（在 `compute_forward` 之前）：

```cpp
assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ...);
for (int j = 0; j < GGML_MAX_SRC; j++) {
    if (node->src[j] != nullptr) {
        assert(node->src[j]->buffer);
        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(...) || ...);
    }
}
```

**为什么需要这个断言**：scheduler 理论上保证所有 tensor 都在 CUDA buffer 上；但如果有 bug（例如 copy 没插入），这里会立刻发现，而不是产生错误结果。

### 14.10 `ggml_cuda_mul_mat()`：决策树逐行

`src/ggml-cuda/ggml-cuda.cu:1824`

```cpp
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0)
        && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }

    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }

    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }

    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1829 | `GGML_TENSOR_BINARY_OP_LOCALS` | 宏，展开出 `ne00..ne03`、`ne10..ne13`、`nb00..` 等局部变量 |
| 1831-1834 | Hadamard hint | 特殊融合路径，由 `ggml_mul_mat_set_hint` 设置 |
| 1836-1839 | `bad_padding_clear` | 量化 compute buffer 有 padding 且是 view 时，不能安全清零 → 必须 cuBLAS |
| 1840 | 类型兜底 | `src1`/`dst` 不是 F32 → cuBLAS（cuBLAS 支持更多类型组合） |
| 1845 | `cc` / `warp_size` | 从设备信息取，避免每次重新查询 |
| 1848-1851 | **MMVF** | F32/F16/BF16 小 N，向量化 |
| 1853-1866 | 转置向量 + 大 batch | `ne01==1` 时把 `src1` 当权重、`src0` 当向量，交换后走 MMVF |
| 1868-1871 | MMF | 非量化、大 N、tensor core |
| 1873-1876 | MMVQ | 量化、小 N |
| 1878-1881 | MMQ | 量化、大 N |
| 1883 | **cuBLAS 兜底** | 所有路径都不满足时的正确性保证 |

**本例代入**：

```text
hint = 0（未设置）
bad_padding_clear = false（F32 没有 padding）
src1.type = F32, dst.type = F32 → 不兜底
cc = 890, warp_size = 32
should_use_mmvf(F32, 890, ne00=2, nb00=4, ne11=3) = true
→ ggml_cuda_mul_mat_vec_f
```

### 14.11 `ggml_cuda_should_use_mmvf()` 逐行

`src/ggml-cuda/mmvf.cu:792`

```cpp
bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc,
                               const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_F16:
            ...
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 793 | `src0_ne[0] % 2 != 0` | **K 必须是偶数**，kernel 用 `float2` 加载 |
| 797 | `ts = ggml_type_size(type)` | F32 = 4，F16 = 2 |
| 798 | `src0_nb[0] != ts` | 要求连续：行内元素步长等于元素大小 |
| 803-807 | `nb[i] % (2*ts) != 0` | **行步长必须是 2 元素对齐**，否则 `float2` 加载会越界 |
| 809-825 | 按类型和架构选阈值 | F32 + Ampere → `ne11 <= 3` |
| 812 | `ampere_mma_available(cc)` | `cc >= 800` 且编译了 Ampere 架构 |

**本例代入**：

```text
2 % 2 = 0 ✓
ts = 4
4 == 4 ✓
8 % 8 = 0, 32 % 8 = 0, 32 % 8 = 0 ✓
F32 + NVIDIA + ampere_mma_available(890) = true
ne11 = 3 <= 3 ✓
→ true
```

**为什么 Ampere 的阈值比 Turing 更小（3 vs 4）**：Ampere 有更高效的 MMA 路径，`ne11=4` 时 MMF 更快；Turing 没有，所以 MMVF 覆盖到 4。这是**按架构调参**的典型例子。

### 14.12 MMVF launch 逐行

`src/ggml-cuda/mmvf.cu:418`

```cpp
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, ..., 
        const int64_t ncols, const int64_t nrows, ...,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);

    const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);

    switch (block_size_best) {
        case 32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, ..., block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        ...
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 425-427 | 三个断言 | `ncols`、`stride_row`、`stride_col_y` 都必须是偶数（`float2` 要求） |
| 432 | `block_size_best = warp_size` | 初始值 32 |
| 433 | `niter_best = ceil(ncols / (2*warp))` | 每个线程处理 `float2`，所以除以 `2*warp` |
| 438-446 | 尝试更大的 block | 只有在能减少迭代次数时才增大 |
| 454 | `nbytes_shared = warp_size * 4` | **只在 `block_size > warp_size` 时真正使用**；本例 128 B 申请但不用 |
| 455 | `block_nums(nrows, nchannels_dst, nsamples)` | `grid.x = M`，每个 block 算一行 |
| 456 | `block_dims(block_size_best, 1, 1)` | 32 线程 |
| 458-490 | 按 `block_size_best` 分派模板 | **模板参数 `block_size` 编译期展开**，循环步长常量 |

**本例代入**：

```text
ncols = 2, warp_size = 32
niter_best = (2 + 63) / 64 = 1
尝试 block_size = 64: niter = (2+127)/128 = 1，不 < 1，不更新
...
block_size_best = 32
block_nums = (4, 1, 1)
block_dims = (32, 1, 1)
nbytes_shared = 128
```

### 14.13 MMVF kernel 逐行

`src/ggml-cuda/mmvf.cu:8`

```cpp
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, ...,
        const int ncols2, ...) {
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int row         = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    ggml_cuda_pdl_sync();

    ...
    x   += int64_t(sample_x)*stride_sample_x + channel_x*stride_channel_x + row*stride_row;
    y   += int64_t(sample_y)*stride_sample_y + channel_y*stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;

    const float2 * x2 = (const float2 *) x;
    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }
            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    dst[tid*stride_col_dst + row] = sumf[tid];
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 14-17 | `__restrict__` 指针 | 告诉编译器无别名，允许重排加载 |
| 18-20 | `blockIdx.x/y`、`threadIdx.x` | 一行一个 block，线程按列分 |
| 28 | `ggml_cuda_pdl_sync()` | Hopper+ 的 programmatic dependent launch 同步；本例空操作 |
| 47-49 | 指针偏移 | `row*stride_row` 定位到本行；channel/sample 用于 batch |
| 51-52 | 转 `float2 *` | **一次加载 2 个 float**，减少内存事务 |
| 54-55 | 动态共享内存 | 只在 `block_size > warp_size` 时用于跨 warp 归约 |
| 60-66 | 初始化 `buf_iw` | 同上 |
| 68 | `float sumf[ncols_dst]` | **编译期数组**，`ncols_dst=3` 时 3 个累加器常驻寄存器 |
| 72 | `for (col2 = tid; col2 < ncols2; col2 += block_size)` | 每个线程处理 `float2` 列，步长 = block 大小 |
| 73 | `x2[col2]` | 加载 `x` 的 2 个元素 |
| 76-79 | 内层 `j` 循环 | **一次加载 `x`，复用到所有 `y[j]`**，减少重复读取 |
| 83 | `ggml_cuda_pdl_lc()` | 提前触发 launch completion；本例空操作 |
| 86-102 | warp 归约 | `warp_reduce_sum` 用 shuffle；`block_size > warp` 时再走 shared |
| 104-106 | `tid >= ncols_dst` 提前返回 | **只有前 3 个线程写出**，其余线程的 `sumf` 是冗余的 |
| 108 | `dst[tid*stride_col_dst + row]` | 列步长 `stride_col_dst = s1 = 4`，与 `nb[1]=16` 一致 |

**本例代入**：

```text
ncols2 = ncols/2 = 1
block_size = 32
只有 tid=0 进入 col2 循环，加载 x2[0]
j=0..2：加载 y2[0]、y2[1]、y2[2]，累加
warp_reduce_sum：31 个线程的 0 参与归约，结果不变
tid=0/1/2 写出 dst[0*4+row]、dst[1*4+row]、dst[2*4+row]
```

**为什么 `sumf[ncols_dst]` 用模板参数**：`ncols_dst` 是编译期常量，编译器可以把 3 个 `float` 分配在寄存器里，内层循环完全展开，没有数组索引开销。

### 14.14 `ggml_backend_tensor_get()` 与 CUDA 实现

`src/ggml-backend.cpp:350`

```cpp
void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}
```

与 `tensor_set` 完全对称，只是方向相反。

CUDA 实现（`ggml-cuda.cu:793`）：

```cpp
static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor * tensor,
                                                void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size,
                               cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```

同样使用 `cudaStreamPerThread` + 立即同步。

### 14.15 释放链路逐行

#### `ggml_backend_sched_free()`

`src/ggml-backend.cpp:1920`

```cpp
void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) return;

    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1925-1929 | 释放 event | `n_copies=1` 时 event 为 NULL，`ggml_backend_event_free(NULL)` 是安全的 |
| 1930 | `ggml_gallocr_free` | **释放所有 CUDA/CPU buffer** |
| 1931 | `ggml_free(sched->ctx)` | 释放 split 用的 context（外部 `context_buffer`，不释放 buffer） |
| 1933-1935 | 释放每个 split 的 `inputs[]` | 每个 split 单独 malloc 的数组 |
| 1936-1946 | 释放各数组 | 与 `sched_new` 的分配一一对应 |

**注意顺序**：先 `gallocr_free`（释放显存），再释放元数据。如果先释放元数据，gallocr 就找不到 buffer 指针了。

#### `ggml_gallocr_free()` → `ggml_vbuffer_free()`

`src/ggml-alloc.c:538`

```cpp
void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (galloc == NULL) return;

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_free(galloc->buffers[i]);
        }
    }
    ...
}
```

`ggml_vbuffer_free()`（402）：

```cpp
static void ggml_vbuffer_free(struct vbuffer * buf) {
    if (buf == NULL) return;
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    free(buf);
}
```

`ggml_backend_buffer_free()`（116）：

```cpp
void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) return;
    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}
```

CUDA 实现（`ggml-cuda.cu:742`）：

```cpp
static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
    delete ctx;
}
```

`delete ctx` 触发析构：

```cpp
~ggml_backend_cuda_buffer_context() {
    CUDA_CHECK(cudaFree(dev_ptr));
}
```

**完整释放链**：

```text
ggml_backend_sched_free
  → ggml_gallocr_free
      → ggml_vbuffer_free
          → ggml_backend_buffer_free
              → buffer->iface.free_buffer   (CUDA)
                  → delete buffer_context
                      → cudaFree(dev_ptr)
              → delete buffer
```

#### `ggml_backend_cuda_free()`

`src/ggml-cuda/ggml-cuda.cu:2442`

```cpp
static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    delete cuda_ctx;
    delete backend;
}
```

析构函数（`common.cuh`）：

```cpp
~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 1-2 | 等待 CUDA graph 捕获完成 | 捕获中的 stream 不能销毁 |
| 5-7 | 销毁 copy event | 只有 `n_copies>1` 时非空 |
| 9-12 | 遍历 8×8 | 销毁所有已创建的 stream |
| 13-15 | 销毁 cuBLAS handle | 只有走过 cuBLAS 路径才非空 |
| 16-18 | 释放 workspace | 同上 |

**本例的实际状态**：

```text
streams[0][0]        → cudaStreamDestroy
cublas_handles[0][0] = NULL → 跳过
cublas_workspaces    = NULL → 跳过
copy_event           = NULL → 跳过
```

### 14.16 完整调用链（带行号）

```text
compute()                                        simple-backend.cpp:99
 ├── ggml_backend_sched_reset                    ggml-backend.cpp:1949
 ├── ggml_backend_sched_alloc_graph              ggml-backend.cpp:1992
 │    ├── ggml_backend_sched_split_graph         ggml-backend.cpp:1066
 │    │    ├── pass 1                            :1087
 │    │    │    └── backend_id_from_cur          :921
 │    │    ├── pass 2                            :1124
 │    │    ├── pass 3                            :1204
 │    │    ├── pass 4                            :1265
 │    │    └── pass 5                            :1297
 │    │         └── ggml_dup_tensor_layout       :749
 │    └── ggml_backend_sched_alloc_splits        ggml-backend.cpp:1591
 │         ├── ggml_gallocr_alloc_graph          ggml-alloc.c:1052
 │         │    └── ggml_gallocr_needs_realloc   ggml-alloc.c:1010
 │         ├── ggml_backend_synchronize ×2       ggml-backend.cpp:425
 │         │    └── ggml_backend_cuda_synchronize ggml-cuda.cu:2548
 │         ├── ggml_gallocr_reserve_n            ggml-alloc.c:962
 │         │    └── reserve_n_impl               ggml-alloc.c:825
 │         │         ├── alloc_graph_impl        ggml-alloc.c:718
 │         │         │    ├── ggml_gallocr_allocate_node  ggml-alloc.c:623
 │         │         │    │    └── ggml_dyn_tallocr_alloc  ggml-alloc.c:202
 │         │         │    └── ggml_gallocr_free_node      ggml-alloc.c:681
 │         │         │         └── free_bytes             ggml-alloc.c:312
 │         │         └── ggml_vbuffer_alloc      ggml-alloc.c:424
 │         │              └── ggml_backend_buft_alloc_buffer  ggml-backend.cpp:39
 │         │                   └── cuda_buffer_type_alloc_buffer  ggml-cuda.cu:884
 │         │                        └── ggml_cuda_device_malloc   ggml-cuda.cu:141
 │         └── ggml_gallocr_alloc_graph（第二次） ggml-alloc.c:1052
 │              └── ggml_gallocr_init_tensor     ggml-alloc.c:975
 │                   └── ggml_vbuffer_tensor_alloc  ggml-alloc.c:442
 │                        └── ggml_backend_tensor_alloc  ggml-backend.cpp:2134
 ├── ggml_backend_tensor_set(a)                  ggml-backend.cpp:335
 │    └── cuda_buffer_set_tensor                 ggml-cuda.cu:785
 ├── ggml_backend_tensor_set(b)                  同上
 ├── ggml_backend_sched_graph_compute            ggml-backend.cpp:2011
 │    ├── graph_compute_async                    ggml-backend.cpp:2017
 │    │    └── compute_splits                    ggml-backend.cpp:1646
 │    │         └── ggml_backend_graph_compute_async  ggml-backend.cpp:461
 │    │              └── backend->iface.graph_compute
 │    │                   └── ggml_backend_cuda_graph_compute  ggml-cuda.cu:4425
 │    │                        └── graph_evaluate_and_capture  ggml-cuda.cu:4185
 │    │                             └── ggml_cuda_compute_forward  ggml-cuda.cu:2068
 │    │                                  └── ggml_cuda_mul_mat  ggml-cuda.cu:1824
 │    │                                       └── ggml_cuda_mul_mat_vec_f  mmvf.cu:634
 │    │                                            └── mul_mat_vec_f_cuda  mmvf.cu:610
 │    │                                                 └── launch_mul_mat_vec_f_cuda  mmvf.cu:418
 │    │                                                      └── mul_mat_vec_f_switch_fusion  mmvf.cu:386
 │    │                                                           └── kernel<<<>>>  mmvf.cu:8
 │    └── sched_synchronize                      ggml-backend.cpp:2032
 │         └── ggml_backend_cuda_synchronize     ggml-cuda.cu:2548
 └── (返回 result)
```

### 14.17 四个“为什么这样设计”

**① 为什么 split 在 alloc 之前**

```text
alloc 需要 buffer_id
buffer_id 由 split 的 pass 3/4/5 决定
→ 顺序不可换
```

**② 为什么 reserve 和 alloc 分成两次**

```text
reserve：O(V+E) 生命周期分析 + 物理分配，慢，只在图结构变化时做
alloc  ：O(V) 写 tensor->data，快，每次 compute 都做
```

**③ 为什么 `tensor_set` 用 buffer 接口而不是 backend 接口**

```text
buffer 知道“这块内存怎么读写”（CUDA buffer → cudaMemcpy；CPU buffer → memcpy）
backend 知道“这条 stream 怎么排队”
set_tensor 是同步语义，不需要 stream 管理 → 走 buffer
```

**④ 为什么 cuBLAS 是兜底而不是首选**

```text
cuBLAS 需要 handle + 4 MiB workspace + 每次调用的 kernel 选择开销
对小矩阵（decode、投机解码）不划算
ggml 自研 kernel 在特定形态下更快
→ 能用专用 kernel 就用，否则兜底
```

---

## 15 常见误区

1. **`alloc_graph` 成功 = 计算完成**：不是。它只做 split + 分配。
2. **`tensor_set` 触发计算**：不是。只做 H2D。
3. **scheduler 实现 MUL_MAT**：不是。只分发。
4. **`cudaMemcpyAsync` 返回 = 拷贝完成**：不是。CUDA 实现紧接着 `cudaStreamSynchronize`。
5. **F32 MUL_MAT 一定走 cuBLAS**：不是。本例走 MMVF。
6. **`ggml_nbytes` = 显存占用**：不是。按 128 B 对齐。
7. **`n_inputs=0` = 没有输入**：不是。它指“跨 backend 拷贝的输入”。
8. **`a`/`b` 被 free 就危险**：本例安全，因为 alloc 后无新分配；真实模型应标记 INPUT/OUTPUT。
9. **CUDA graph 总是更快**：不是。单 kernel 反而更慢。
10. **`cudaMalloc` 走 VMM**：不是。VMM 是 compute pool 的路径。

---

## 16 待验证点

1. `GGML_SCHED_DEBUG=2`：确认 `## SPLIT #0: CUDA0 # 0 inputs`；注意 `SET_CAUSE` 在当前源码被 `#if 0` 编译掉，cause 列为空。
2. `GGML_CUDA_DEBUG` 是编译期宏，环境变量无效；需要 `-DGGML_CUDA_DEBUG` 重编译。
3. 把 `rows_B` 改成 4：观察 MMVF → cuBLAS，`cublasCreate`、4 MiB workspace。
4. `GGML_CUDA_FORCE_CUBLAS=1`：强制 cuBLAS，对比 launch 参数。
5. 把 `a` 放到 CPU buffer：观察 `n_inputs=1`、copy tensor、event。
6. 加 `ggml_set_input/output`：观察 flags、scheduler 归属、gallocr free 行为。
7. 构造 3 层图：观察 free list 复用与 `ggml_gallocr_free_extra_space`。
8. `GGML_CUDA_GRAPHS=ON` + 循环 compute：观察 capture/instantiate/launch。
9. 用 `nsys profile` / `ncu` 对比 MMVF 与 cuBLAS 的 launch/执行时间。
