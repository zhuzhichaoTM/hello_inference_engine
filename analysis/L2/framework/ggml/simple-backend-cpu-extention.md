# simple-backend.cpp：CPU 执行扩展原理

本文是 `simple-backend-cpu-compute.md` 的扩展文档，聚焦三个用于闭合“graph 已提交后，CPU kernel 为什么这样执行”这一底层链路的主题：

1. CPU graph plan 与 threadpool；
2. graph allocator 的 tensor 生命周期和内存复用；
3. `type_traits_cpu`、量化类型与 `vec_dot` kernel。

核心闭环：

```text
CPU backend graph dispatch
    → ggml_graph_plan
    → ggml_graph_compute / threadpool
    → ggml_compute_forward
    → GGML_OP_MUL_MAT
    → type_traits_cpu / vec_dot_type
    → ggml_compute_forward_mul_mat_one_chunk
    → vec_dot
```

配套主文档：

```text
docs/simple-backend-cpu-compute.md
```

---

# 16. CPU graph plan：从 graph 到可执行计划

## 16.1 `ggml_graph_plan` 的位置

CPU backend 收到 split graph 后，首先进入：

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

    ...

    return ggml_graph_compute(cgraph, &cplan);
}
```

`ggml_graph_plan` 不执行数值计算，而是把一张静态 graph 转换为 CPU executor 使用的 execution plan：

```text
cgraph：执行哪些 node、每个 node 的 op 和依赖
cplan：使用多少线程、每个 op 需要多少 task/workspace、如何启动执行
```

这一步连接了：

```text
backend graph dispatch
    → CPU 执行规划
    → threadpool 实际执行
```

## 16.2 plan 的主要输出

`ggml_graph_plan` 主要确定：

```text
n_threads：参与执行的线程数
max_tasks：graph 中单个 op 所需的最大任务数
work_size：所有临时 workspace 的最大需求
threadpool：可复用的 CPU 线程池
```

函数首先处理线程数：

```cpp
if (n_threads <= 0) {
    n_threads = threadpool
        ? threadpool->n_threads
        : GGML_DEFAULT_N_THREADS;
}
```

在不支持 pthread 的 WASI/Emscripten 配置下还会强制：

```text
n_threads = 1
```

随后遍历 graph node：

```cpp
for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor * node = cgraph->nodes[i];
    const int n_tasks = ggml_get_n_tasks(node, n_threads);
    max_tasks = MAX(max_tasks, n_tasks);
    ...
}
```

`ggml_get_n_tasks` 根据 op、tensor shape 和线程数决定当前 node 如何切分任务。它不是简单地为每个 node 固定创建 `n_threads` 个任务；某些小 op 可能只需要一个任务，矩阵乘法则通常可以并行切分。

## 16.3 workspace 与 tensor buffer 的区别

CPU plan 会计算临时工作区：

```cpp
size_t work_size = 0;
```

不同 op 可能需要不同 workspace：

```text
量化/非量化转换
F16/BF16/F32 中间转换
MUL_MAT 的输入打包
IQ panel scratch
特殊 reduction 或 copy 临时区
```

例如 `MUL_MAT` 会根据：

```cpp
src0->type
src1->type
vec_dot_type
```

判断 `src1` 是否需要转换为 dot kernel 需要的类型；某些 IQ 路径还会按线程增加 scratch 空间。

这块 `cplan.work_data` 与 graph allocator 分配的 tensor buffer 不同：

```text
tensor buffer：保存 a、b、result 等 tensor 的持久数据
cplan.work_data：算子执行期间的临时 workspace
```

CPU backend 中：

```cpp
if (cpu_ctx->work_size < cplan.work_size) {
    delete[] cpu_ctx->work_data;
    cpu_ctx->work_data =
        new uint8_t[cplan.work_size];
    cpu_ctx->work_size = cplan.work_size;
}

cplan.work_data =
    (uint8_t *) cpu_ctx->work_data;
```

因此 workspace 会在 CPU backend context 中按需增长，并可以在后续 graph compute 中复用。

## 16.4 plan、executor、kernel 的边界

```text
ggml_graph_plan：估算任务和 workspace，不算数值
ggml_graph_compute：启动/协调线程，不决定具体数学公式
ggml_graph_compute_thread：遍历 node、调用 op dispatch
ggml_compute_forward_mul_mat：执行 MUL_MAT 算法
```

---

# 17. CPU threadpool：node 级同步与 op 内并行

## 17.1 worker 线程执行入口

`ggml_graph_compute` 将当前 graph/plan 绑定到 threadpool 后，worker 进入：

```cpp
ggml_graph_compute_thread(void * data)
```

线程构造当前计算参数：

```cpp
struct ggml_compute_params params = {
    /* .ith        = */ state->ith,
    /* .nth        = */ atomic_load(&tp->n_graph)
                         & GGML_THREADPOOL_N_THREADS_MASK,
    /* .wsize      = */ cplan->work_size,
    /* .wdata      = */ cplan->work_data,
    /* .threadpool = */ tp,
    /* .use_ref    = */ cplan->use_ref,
};
```

关键字段：

```text
ith：当前线程编号，从 0 到 nth-1
nth：当前参与该 graph 的线程总数
wsize/wdata：共享的 CPU 临时 workspace
threadpool：线程同步和调度上下文
```

## 17.2 graph node 的顺序执行

worker 按 `cgraph->nodes[]` 顺序遍历：

```cpp
for (int node_n = 0;
     node_n < cgraph->n_nodes;
     node_n++) {
    struct ggml_tensor * node = cgraph->nodes[node_n];

    if (ggml_op_is_empty(node->op)) {
        continue;
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        continue;
    }

    ...
}
```

这体现了两个过滤条件：

```text
GGML_OP_NONE/空 op：跳过
没有 COMPUTE flag：跳过
```

在构图阶段，真正的计算节点通常会被标记为 `GGML_TENSOR_FLAG_COMPUTE`；leaf 只提供输入数据，不作为算子执行。

## 17.3 同一个 node 的多线程协作

node 的执行由所有 worker 共同进入 `ggml_compute_forward`，每个线程通过：

```cpp
params.ith
params.nth
```

选择自己的输出工作区间。

因此需要区分两种并行：

```text
CPU op parallelism：多个 CPU 线程共同计算一个 MUL_MAT
scheduler pipeline copies：多轮 graph 使用不同 copy slot 并行提交
```

当前 simple-backend 使用：

```cpp
parallel = false
n_copies = 1
```

因此没有 scheduler 多 copy pipeline，但 CPU `MUL_MAT` 仍可能使用多个 CPU worker。

## 17.4 node 之间的 barrier

一个 node 执行完后，线程通常在下一个 node 前进行：

```cpp
ggml_barrier(state->threadpool);
```

含义是：

```text
所有线程完成当前 node
    ↓
所有线程再进入下一个 node
```

这保证后继 node 读取前驱 node 的输出时，前驱已经由所有参与线程完成。

## 17.5 fused op

worker 在普通 dispatch 前会尝试：

```cpp
const int n_fused =
    ggml_cpu_try_fuse_ops(
        cgraph,
        node_n,
        &params,
        cplan);
```

如果返回大于 0：

```cpp
node_n += n_fused;
```

表示多个相邻 op 已经融合执行，跳过已被融合的 node。否则进入：

```cpp
ggml_compute_forward(&params, node);
```

融合的意义是减少中间 tensor 写回和重复内存访问，但不改变 graph 的数学依赖。

---

# 18. Graph allocator 生命周期：从 use count 到内存复用

## 18.1 allocator 不是逐 tensor malloc

`ggml_gallocr` 的工作是：

```text
根据 graph 拓扑和 tensor 生命周期
    ↓
规划每个 tensor 的 (buffer_id, chunk, offset)
    ↓
复用已经不再使用的区域
```

其内部记录：

```cpp
struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};
```

核心生命周期字段：

```text
n_children：还有多少 graph node 会读取该 tensor
n_views：还有多少 view 依赖该 tensor
allocated：该 tensor 当前是否占据 allocator 区域
addr：实际 (chunk, offset)
```

## 18.2 初始统计阶段

`ggml_gallocr_alloc_graph_impl` 首先处理 leaf，然后统计依赖：

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];

    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * src = node->src[j];
        if (src == NULL) {
            continue;
        }

        ggml_gallocr_hash_get(galloc, src)
            ->n_children += 1;
    }
}
```

如果 node 是普通 view，还会增加源 tensor 的：

```cpp
n_views += 1;
```

这两个计数共同决定 tensor 是否可以释放：

```text
n_children == 0 && n_views == 0
```

## 18.3 按 graph 顺序分配 node

随后遍历 graph：

```cpp
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];

    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * parent = node->src[j];
        if (parent != NULL) {
            ggml_gallocr_allocate_node(
                galloc,
                parent,
                buffer_id);
        }
    }

    ggml_gallocr_allocate_node(
        galloc,
        node,
        buffer_id);

    ...
}
```

分配顺序是：

```text
先确保 parent/source 可用
    ↓
再分配当前 node 的输出
```

`ggml_gallocr_allocate_node` 可能从现有 free block 中复用空间，也可能在 chunk 尾部扩展。

## 18.4 执行当前 node 后减少 parent 引用

当前 node 已经被安排后，allocator 更新其 parent 的剩余消费者数：

```cpp
p_hn->n_children -= 1;
```

如果 parent 不再有 child，且没有 view 依赖：

```cpp
if (p_hn->n_children == 0 &&
    p_hn->n_views == 0) {
    ggml_gallocr_free_node(
        galloc,
        parent);
}
```

释放并不意味着立即调用系统 `free`；它通常只是把这段区域放回动态 tallocr 的 `free_blocks`，供后续 tensor 复用。

## 18.5 view 对生命周期的影响

对于 view：

```text
view 不拥有独立 storage
view_src 不能在 view 仍存在时释放
```

因此 allocator 同时维护：

```text
view_src->n_views
view 的 n_children
```

当 view 自身不再被使用时，先减少源 tensor 的 `n_views`；只有源 tensor 同时满足：

```text
n_views == 0
n_children == 0
```

才允许释放源区域。

## 18.6 输出不会被复用

```cpp
if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
    return;
}
```

graph output 被保留到 graph 使用结束，不能像普通中间 tensor 一样在 graph 内被提前回收。这保证调用方在计算完成后仍能读取 result。

## 18.7 allocation dependency 的作用

backend 可能因为异步 stream 或内部并发，让某个 tensor 在拓扑关系上看似用完，但实际仍需存活。scheduler 会通过 `GGML_OP_NONE` dependency node 保留它：

```text
backend add_alloc_dep(tensor, until)
    ↓
插入引用 tensor 的 dependency node
    ↓
allocator 增加可见生命周期
    ↓
阻止过早复用
```

因此内存复用必须同时服从：

```text
graph children
view dependencies
backend allocation dependencies
output flag
```

## 18.8 allocation 与初始化的分离

galloc 的实际执行分成两层：

```text
reserve/规划阶段：记录每个 tensor 的 tensor_alloc
初始化阶段：ggml_gallocr_init_tensor
```

`ggml_gallocr_init_tensor` 中：

```cpp
if (tensor->view_src != NULL) {
    ggml_backend_view_init(tensor);
} else if (tensor->data == NULL) {
    ggml_vbuffer_tensor_alloc(
        galloc->buffers[buffer_id],
        tensor,
        tensor_alloc->addr);
}
```

普通 tensor 最终得到：

```text
tensor->buffer = chunk 对应 backend buffer
tensor->data   = buffer base + chunk offset
```

view 则通过 `ggml_backend_view_init` 继承源 tensor 的 buffer，并使用 view offset。

---

# 19. CPU `type_traits` 与 `vec_dot`：从 tensor type 到 kernel

## 19.1 `type_traits_cpu` 的职责

CPU backend 使用：

```cpp
static const struct ggml_type_traits_cpu
    type_traits_cpu[GGML_TYPE_COUNT];
```

它是按 `ggml_type` 索引的 CPU 类型行为表，提供类似：

```text
vec_dot：向量点积 kernel
vec_dot_type：该 kernel 期望的计算/输入类型
from_float：F32 到该类型的转换函数
nrows：kernel 一次处理的行数
```

访问方式：

```cpp
const struct ggml_type_traits_cpu * traits =
    ggml_get_type_traits_cpu(type);
```

`MUL_MAT` 中通常直接使用：

```cpp
type_traits_cpu[src0->type].vec_dot_type
type_traits_cpu[type].vec_dot
```

## 19.2 F32 本例的路径

当前 simple-backend 使用：

```cpp
GGML_TYPE_F32
```

因此通常是：

```text
src0 type = F32
vec_dot_type = F32
vec_dot = F32 对应点积 kernel
```

不需要量化块解码，也通常不需要把 activation 转为量化格式。

## 19.3 量化 `MUL_MAT` 的路径

如果 `src0` 是量化权重，例如：

```text
Q4_0
Q4_K
Q5_K
Q6_K
Q8_0
IQ 系列
```

`MUL_MAT` 不会先把整份权重展开成 F32 再计算，而是通过量化类型对应的 `vec_dot` kernel：

```text
量化 tensor data
    ↓
按 block 读取 scale/min/quant values
    ↓
在 vec_dot 内解码或反量化
    ↓
与 activation 做点积
    ↓
写入 F32 result
```

这同时降低：

```text
权重存储空间
内存带宽
整模型反量化的额外开销
```

## 19.4 `MUL_MAT` 如何选择 dot 类型

`ggml_compute_forward_mul_mat` 中：

```cpp
enum ggml_type vec_dot_type =
    type_traits_cpu[src0->type].vec_dot_type;

ggml_vec_dot_t vec_dot =
    type_traits_cpu[type].vec_dot;
```

随后检查：

```cpp
if (src1->type != vec_dot_type) {
    // 需要临时转换/打包
}
```

这解释了 CPU plan 为什么可能为 `MUL_MAT` 计算 workspace：

```text
src1 类型不符合 vec_dot_type
    ↓
先转换或打包到 cplan.work_data
    ↓
vec_dot 使用目标类型计算
```

## 19.5 `from_float` 的作用

当 kernel 需要特定输入类型时：

```cpp
ggml_from_float_t from_float =
    type_traits_cpu[vec_dot_type].from_float;
```

它把 F32 临时值转换成 dot kernel 期望的格式，例如某些 Q8 中间表示。这个转换通常写入 `cplan.work_data`，而不是修改原始 activation tensor。

## 19.6 量化 block 与 tensor stride

量化 tensor 的 `nb[]` 不是简单的：

```text
ne[i] * sizeof(float)
```

第 0 维以量化 block 为基本单位，具体行大小由：

```cpp
ggml_row_size(type, ne[0])
```

决定。

因此量化 kernel 同时依赖：

```text
tensor->type
ne[]
nb[]
ggml_type_size(type)
ggml_blck_size(type)
```

这也是 `MUL_MAT` 中存在 layout 断言和 `ggml_row_size` 计算的原因。

## 19.7 量化与 GGUF 的关系

GGUF 加载的权重 tensor 会保留其量化 `ggml_type` 和 block layout。运行时通常直接将量化权重放入 WEIGHTS buffer：

```text
GGUF tensor type = Q4_K 等
    ↓
GGML tensor->type 保留量化类型
    ↓
CPU/GPU backend 选择对应 kernel
    ↓
vec_dot 边解码边计算
```

当前手工构造的 F32 `simple-backend` 不体现这条路径，但它是实际 LLM 推理中最重要的性能机制之一。

## 19.8 观察 type_traits 和量化 kernel 的断点

```gdb
break ggml_compute_forward_mul_mat
break ggml_compute_forward_mul_mat_one_chunk
```

观察：

```gdb
p src0->type
p src1->type
p vec_dot_type
p src0->nb[0]
p src1->nb[0]
p params->wdata
```

如需验证类型表：

```gdb
p type_traits_cpu[src0->type].vec_dot
p type_traits_cpu[src0->type].vec_dot_type
p type_traits_cpu[src0->type].nrows
```

当前 F32 示例预期：

```text
src0->type = GGML_TYPE_F32
src1->type = GGML_TYPE_F32
vec_dot_type = GGML_TYPE_F32
```

---

# 20. 从 graph 提交到 CPU kernel 的闭环

现在可以把“graph 已经提交后，CPU kernel 为什么这样执行”完整串起来：

```text
1. scheduler 选择 CPU split
       ↓
2. ggml_backend_cpu_graph_compute
       ↓
3. ggml_graph_plan
       ├─ 计算每个 op 的 task 数
       ├─ 计算 workspace
       └─ 准备 cplan
       ↓
4. ggml_graph_compute
       ├─ 绑定 threadpool
       └─ 启动 worker
       ↓
5. ggml_graph_compute_thread
       ├─ 按 cgraph->nodes 顺序遍历
       ├─ 跳过空/非 COMPUTE node
       ├─ 尝试 op fusion
       └─ ggml_compute_forward
       ↓
6. GGML_OP_MUL_MAT 分支
       ↓
7. ggml_compute_forward_mul_mat
       ├─ 读取 src0/src1/dst 的 ne/nb/type
       ├─ 选择 vec_dot_type 和 vec_dot
       ├─ 必要时使用 work_data 转换输入
       ├─ 根据 ith/nth 划分输出区间
       └─ 调用分块计算
       ↓
8. ggml_compute_forward_mul_mat_one_chunk
       ↓
9. vec_dot
       └─ F32 直接点积，量化类型边解码边点积
```

这条链路闭合了三类核心信息：

```text
graph topology：决定 node 顺序和依赖
allocator lifetime：决定数据在哪里、何时可复用
type_traits/kernel：决定具体使用哪种数学实现
```

---

# 21. 新增后的调试重点

## 21.1 CPU graph plan

```gdb
break ggml_backend_cpu_graph_compute
break ggml_graph_plan
p cgraph->n_nodes
p n_threads
p threadpool
p cplan.work_size
p cplan.max_tasks
```

## 21.2 threadpool

```gdb
break ggml_graph_compute_thread
p state->ith
p params.ith
p params.nth
p node_n
p node->name
p node->op
```

观察不同线程是否进入同一个 node，以及 `ith/nth` 如何变化。

## 21.3 allocator lifetime

```gdb
break ggml_gallocr_allocate_node
break ggml_gallocr_free_node
p node->name
p node->data
p node->view_src
p hn->n_children
p hn->n_views
p hn->allocated
p hn->addr.chunk
p hn->addr.offset
```

重点观察：

```text
parent 的 n_children 变为 0
view 的 n_views 变为 0
free_node 后区域进入 free_blocks
后继 tensor 复用同一 chunk/offset
```

## 21.4 type_traits 和 vec_dot

```gdb
break ggml_compute_forward_mul_mat
break ggml_compute_forward_mul_mat_one_chunk
p src0->type
p src1->type
p vec_dot_type
p params->ith
p params->nth
p params->wdata
```

这套结构共同实现：

```text
低碎片
低峰值内存
跨 backend placement
零拷贝 view
中间 tensor 复用
异步执行安全
```

---

# 31. 管理面、控制面、数据面：GGML 内存系统的分层

为了避免把“描述内存的对象”和“真正存放数据的空间”混为一谈，可以把 GGML 内存系统分为三层：

```text
管理面：拥有/提供内存资源
控制面：描述、规划、调度内存对象
数据面：承载 tensor 的真实字节和执行数据
```

这不是源码中直接定义的三个模块，而是理解 GGML 数据结构职责的分析模型。

## 31.1 三个平面的核心问题

| 平面 | 核心问题 | 典型对象 |
|---|---|---|
| 管理面 | 这块存储由谁拥有、属于哪个 backend、何时创建/释放 | `ggml_backend_buffer_type`、`ggml_backend_buffer`、`vbuffer`、CPU context |
| 控制面 | 哪个 tensor 需要多少空间、放在哪、何时可以复用 | `ggml_object`、`ggml_context`、`ggml_tensor`、`ggml_cgraph`、`ggml_gallocr`、`hash_node`、`tensor_alloc` |
| 数据面 | 实际字节如何分布、如何按 offset/stride 访问、哪些区间可复用 | backend buffer base、`tensor->data`、`ne[]`/`nb[]`、`tallocr_chunk`、`free_block`、chunk offset |

需要注意：某些结构跨越多个平面。例如：

```text
vbuffer：管理面聚合多个 backend buffer，同时为数据面提供 chunk 容器

ggml_tensor：控制面描述逻辑 tensor，同时通过 data/buffer 指向数据面

tallocr_chunk：控制面记录分配区间，数据面对应一个真实 buffer 分段
```

## 31.2 一条完整的平面关系

```text
管理面：CPU backend / buffer type 创建 buffer
    ↓
控制面：galloc 根据 graph 规划 tensor placement
    ↓
数据面：dyn_tallocr 在 chunk/free_block 中选择具体区间
    ↓
管理面：backend buffer 提供真实 base address
    ↓
控制面：ggml_backend_tensor_alloc 设置 tensor->buffer/data
    ↓
数据面：CPU/GPU kernel 通过 data + nb[] 读写真实字节
```

---

# 32. 管理面：内存资源的拥有、创建和后端归属

## 32.1 `ggml_backend_buffer_type`

`ggml_backend_buffer_type` 描述一种 backend 存储类型，主要提供：

```text
get_name
alloc_buffer
get_alignment
get_max_size
get_alloc_size
is_host
```

它回答：

```text
如何申请这种内存？
需要什么对齐？
单个 buffer 最大多大？
一个 tensor 需要分配多少实际空间？
该内存是否可被 CPU 直接访问？
```

CPU 类型中：

```cpp
.alloc_buffer = ggml_backend_cpu_buffer_type_alloc_buffer
```

最终：

```cpp
ggml_aligned_malloc(size)
```

`buffer_type` 本身通常不保存某个具体 tensor 的数据，它是分配策略和能力描述。

## 32.2 `ggml_backend_buffer`

```cpp
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;
    ggml_backend_buffer_type_t buft;
    void * context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};
```

它是某一次实际分配出的存储实例：

```text
buffer_type：采用哪种存储规则
context：底层 CPU/GPU 私有对象
size：整块 buffer 容量
usage：WEIGHTS、COMPUTE 等用途
iface：set/get/copy/free 等具体操作
```

`ggml_backend_buffer` 拥有或包装实际 storage，但不负责决定 graph 中哪个 tensor 放在哪里；placement 由控制面 allocator 决定。

## 32.3 `vbuffer`

```cpp
struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};
```

`vbuffer` 是管理面和数据面之间的聚合层：

```text
一个逻辑 buffer
    = 多个实际 backend buffer chunk
```

它允许 graph allocator 在 backend buffer 有容量上限时将逻辑空间拆开，同时仍用统一的 `buffer_address {chunk, offset}` 描述 tensor 位置。

## 32.4 CPU backend context 与 workspace

```cpp
struct ggml_backend_cpu_context {
    int n_threads;
    struct ggml_threadpool * threadpool;
    uint8_t * work_data;
    size_t work_size;
    ...
};
```

CPU context 管理两类资源：

```text
threadpool：执行控制资源
work_data：CPU kernel 临时数据资源
```

`work_data` 不是 graph tensor 的长期存储；它由 `ggml_graph_plan` 的 `work_size` 驱动按需扩展。

---

# 33. 控制面：元数据、图关系和内存调度

## 33.1 `ggml_object`：控制对象在 context 中的记录

```cpp
struct ggml_object {
    size_t offs;
    size_t size;
    struct ggml_object * next;
    enum ggml_object_type type;
};
```

它控制的是：

```text
context buffer 中有哪些 object
每个 object 占用哪段 metadata 空间
object 的遍历顺序和类型
```

它不管理 backend tensor data。

## 33.2 `ggml_context`：metadata arena 的控制器

```cpp
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool mem_buffer_owned;
    bool no_alloc;
    int n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

`ggml_context` 管的是：

```text
构图对象的生命周期
metadata 内存池
是否允许在 context 中直接分配 data
```

simple-backend 的：

```cpp
.no_alloc = true
```

把“构图对象控制”与“backend 数据存储管理”明确分开。

## 33.3 `ggml_tensor`：逻辑对象的控制描述

`ggml_tensor` 处于控制面和数据面的边界：

```text
控制面字段：type、op、src[]、flags、view_src、view_offs、ne[]、nb[]
数据面指针：buffer、data
```

它控制：

```text
这个对象是什么类型
形状和 stride 是什么
由哪个 op 产生
依赖哪些 source
是否 input/output/view/compute
```

同时通过：

```cpp
tensor->buffer
tensor->data
```

指向数据面。

因此 `ggml_tensor` 不是“数据块本身”，而是“数据块的逻辑句柄 + 计算语义描述”。

## 33.4 `ggml_cgraph`：控制计算顺序和使用关系

`ggml_cgraph` 的内存相关作用不是存储 tensor 数据，而是提供：

```text
nodes[]：哪些 tensor node 要执行
leafs[]：哪些 tensor 是外部输入/常量
src[]：依赖边
use_counts[]：图上的使用计数
```

它是控制面中把“数据何时需要、何时可以释放”转化为可计算关系的关键。

## 33.5 `ggml_gallocr`：控制面总调度器

`ggml_gallocr` 把 graph 控制关系转化成 placement 计划：

```text
输入：graph、node/leaf backend ID、tensor size/lifetime
输出：buffer_id、chunk、offset、size_max
```

它不直接执行 kernel，也不直接处理每个 byte，而是调用地址层 `dyn_tallocr`。

## 33.6 `hash_node`：每个逻辑 tensor 的生命周期状态机

```cpp
struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};
```

控制面逻辑：

```text
首次被需要：allocated = true
每消费一个 source：n_children--
每创建/销毁一个 view：n_views +/- 1
children/views 都为 0：允许释放
```

## 33.7 `tensor_alloc`：控制面 placement 记录

```cpp
struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max;
};
```

它不是实际 storage，而是 allocation plan：

```text
buffer_id：使用哪个 buffer type
addr：chunk 和 offset
size_max：规划时预留的最大空间
```

---

# 34. 数据面：具体字节的存储和地址复用

## 34.1 `tallocr_chunk`：数据分段的空间模型

```cpp
struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};
```

从数据面角度，它描述：

```text
一个 chunk 当前有哪些可写区间
每个区间从哪里开始、长度多少
chunk 当前实际需要多大
```

## 34.2 `free_block`：可复用物理区间

```cpp
struct free_block {
    size_t offset;
    size_t size;
};
```

`free_block` 是数据面最直接的复用单位：

```text
[free block offset, offset + size)
```

当 tensor 最后一个消费者完成后，该范围回收到 free block；后续 tensor 可以直接占用，不需要重新申请底层 CPU/GPU buffer。

## 34.3 `buffer_address`：跨 chunk 的数据地址

```cpp
struct buffer_address {
    int chunk;
    size_t offset;
};
```

它将物理地址抽象为：

```text
第几个 chunk + chunk 内偏移
```

在最终 placement 时转换为：

```cpp
base = ggml_backend_buffer_get_base(
    vbuffer->chunks[addr.chunk]);

data = (char *) base + addr.offset;
```

## 34.4 `tensor->data`：kernel 实际使用的地址

最终 CPU/GPU kernel 不读取 `free_block`、`hash_node` 或 `tensor_alloc`；它只通过：

```cpp
tensor->data
```

结合：

```cpp
tensor->ne[]
tensor->nb[]
tensor->type
```

访问真实数据。

因此：

```text
控制面：决定 data 应该指向哪里
数据面：保存 data 指向的字节
kernel：使用 data + nb[] 解释字节
```

## 34.5 `free_block` 与 `tensor->data` 的关系

一个典型过程：

```text
t0 分配：chunk 0, offset 0, size 64
    ↓
t0 最后一个 child 完成
    ↓
free_block = { offset 0, size 64 }
    ↓
t1 请求 48 bytes
    ↓
t1 复用 chunk 0, offset 0
    ↓
t1->data = 同一底层地址
```

这并不意味着 `t0` 和 `t1` 同时拥有同一块有效数据；复用的前提是 t0 已经不再被 graph、view、output 或 backend dependency 使用。

---

# 35. 三个平面如何协同完成一次 allocation

以普通中间 tensor `t` 为例：

```text
【控制面：对象产生】
ggml_new_tensor
    → 创建 t 的 metadata
    → t->data/buffer 尚未设置

【控制面：图关系产生】
ggml_build_forward_expand
    → t 进入 graph node
    → src/use relationship 被记录

【控制面：生命周期规划】
ggml_gallocr
    → 统计 n_children/n_views
    → 选择 buffer_id
    → 调用 dyn_tallocr

【数据面：区间选择】
ggml_dyn_tallocr
    → 查找 free_block
    → 选择 chunk + offset
    → 记录 tensor_alloc.addr

【管理面：资源实例化】
ggml_vbuffer_alloc
    → 创建 ggml_backend_buffer
    → CPU: ggml_aligned_malloc

【控制面：绑定句柄】
ggml_gallocr_init_tensor
    → ggml_backend_tensor_alloc
    → 设置 t->buffer/t->data

【数据面：kernel 使用】
CPU kernel
    → 通过 t->data + t->nb[] 读写

【控制面：生命周期结束】
n_children/n_views 归零
    → ggml_gallocr_free_node

【数据面：空间回收】
ggml_dyn_tallocr_free_bytes
    → free_block
    → 后继 tensor 复用
```

## 35.1 三个平面之间的边界

```text
管理面不决定 graph 中哪个 tensor 复用哪个 offset
控制面不直接执行 memcpy/vec_dot
数据面不理解 op 依赖和 tensor 生命周期
```

这三个边界是理解 GGML 内存系统的关键：

```text
管理面提供资源
控制面做决策
数据面承载字节
```

---

# 36. 数据结构分类速查表

| 数据结构 | 所属平面 | 承载内容 | 是否直接存放 tensor 数据 | 核心作用 |
|---|---|---|---:|---|
| `ggml_object` | 控制面 | metadata object 位置/大小/类型 | 否 | 管理 context 对象 |
| `ggml_context` | 控制面 | metadata arena 和 object 链 | 否 | 管理构图内存 |
| `ggml_tensor` | 控制面 + 数据面句柄 | shape/op/src/flags/data/buffer | 通过 `data` 指向 | 描述逻辑 tensor |
| `ggml_cgraph` | 控制面 | nodes/leafs/use counts | 否 | 描述执行拓扑 |
| `ggml_gallocr` | 控制面 | buffer、plan、生命周期状态 | 否 | 图级内存调度 |
| `hash_node` | 控制面 | children/views/address/allocated | 否 | tensor 生命周期状态 |
| `tensor_alloc` | 控制面 | buffer_id/chunk/offset/size | 否 | placement 计划 |
| `ggml_dyn_tallocr` | 控制面 → 数据面 | chunks/free 区间分配器 | 否 | 地址选择和复用 |
| `tallocr_chunk` | 数据面管理 | free blocks/max size | 间接 | 描述一段可用空间 |
| `free_block` | 数据面 | offset/size 空闲区间 | 否 | 最小复用单位 |
| `buffer_address` | 控制面 + 数据面 | chunk/offset | 否 | 逻辑物理地址 |
| `vbuffer` | 管理面 + 数据面 | backend buffer chunks | 间接 | 聚合多个实际 buffer |
| `ggml_backend_buffer` | 管理面 | backend storage/context/interface | 是，底层拥有 | 具体 backend 存储实例 |
| `ggml_cplan.work_data` | 管理面/执行数据面 | 临时 workspace | 是，临时数据 | 算子执行 scratch |
| `tensor->data` | 数据面入口 | 实际数据地址 | 是 | kernel 直接读写 |

---

# 37. 最终内存逻辑图

```text
                 【管理面】
  buffer_type / backend / CPU context / vbuffer
                       │
                       │ 创建真实存储
                       ▼
                 backend buffer
                       │
                       │ 提供 base/size/interface
                       ▼
                 【控制面】
  context → object → tensor → cgraph → gallocr
                                      │
                                      │ 生命周期/placement
                                      ▼
             hash_node / tensor_alloc / buffer_address
                                      │
                                      ▼
                 【数据面】
       dyn_tallocr → chunk → free_block → offset
                                      │
                                      ▼
                 tensor->data
                                      │
                         ne[] / nb[] / type
                                      ▼
                    CPU/GPU kernel 读写
```

最简洁的职责判断方法：

```text
看到 context/object：问“这个 metadata 对象如何管理？”
看到 tensor/cgraph/gallocr：问“哪个逻辑对象何时需要、放哪里？”
看到 chunk/free_block/data：问“真实字节在哪个区间、如何复用？”
看到 backend buffer：问“这块存储由哪个 backend 创建和拥有？”
```

---

# 38. 基于三平面的调试路径

## 38.1 管理面断点

```gdb
break ggml_backend_cpu_buffer_type_alloc_buffer
break ggml_backend_buffer_init
break ggml_vbuffer_alloc
p buft
p size
p buffer->context
p buffer->size
```

确认：

```text
谁创建了真实 backend buffer
申请了多大容量
属于 CPU/其他 backend
usage 是 WEIGHTS 还是 COMPUTE
```

## 38.2 控制面断点

```gdb
break ggml_new_object
break ggml_gallocr_allocate_node
break ggml_gallocr_free_node
break ggml_gallocr_init_tensor
break ggml_graph_plan
```

确认：

```text
metadata 如何追加
node/source 如何建立生命周期计数
tensor 的 placement plan 是什么
何时允许释放
```

## 38.3 数据面断点

```gdb
break ggml_dyn_tallocr_alloc
break ggml_dyn_tallocr_free_bytes
break ggml_dyn_tallocr_new_chunk
break ggml_backend_tensor_alloc
break ggml_compute_forward_mul_mat_one_chunk
```

确认：

```text
具体 chunk/offset
free block 如何变化
tensor->data 的最终地址
kernel 如何用 data + nb[] 访问
```

---

# 39. 分层总结

GGML 内存系统可以按以下方式理解：

```text
管理面：
    负责 backend storage 的拥有、创建、销毁和能力接口
    代表：buffer_type、backend_buffer、vbuffer、CPU context

控制面：
    负责 metadata、graph topology、tensor placement 和生命周期决策
    代表：ggml_object、ggml_context、ggml_tensor、cgraph、gallocr、hash_node、tensor_alloc

数据面：
    负责真实字节的布局、地址访问、区间释放与复用
    代表：tensor->data、ne/nb、chunk、free_block、buffer_address、workspace
```

其中用户特别关心的两个例子可以准确归类为：

```text
ggml_object / ggml_context / ggml_tensor：
    主要属于控制面，描述和调度内存对象；
    tensor 的 data/buffer 字段是连接数据面的句柄。

tallocr_chunk / free_block：
    主要属于数据面地址管理，描述真实 backend buffer 中的可用/已释放区间；
    但它们由控制面的 gallocr/dyn_tallocr 驱动。
```

最终核心原则：

```text
n_children 和 n_views 决定“能不能回收”，
chunk/free_block 决定“回收后哪段物理空间可以复用”，
tensor->data 则是 kernel 最终真正访问的地址。
```


---

# 22. GGML 内存高效分配与调度：数据结构全景

本节从“谁拥有内存、谁规划内存、谁分配 tensor、谁回收空间、谁保证生命周期”五个问题出发，系统梳理 GGML 的内存数据结构。

## 22.1 先区分三类内存

GGML 中容易混淆的“内存”至少有三类：

```text
A. context metadata 内存
   保存 ggml_object、ggml_tensor、ggml_cgraph 等元数据

B. backend tensor buffer
   保存 a、b、result、权重和中间 tensor 的实际数据

C. execution workspace
   算子执行期间的临时空间，如类型转换、打包、scratch
```

对应关系：

```text
ggml_context
    └─ A：构图元数据

ggml_backend_buffer
    └─ B：backend 数据存储

CPU ggml_cplan.work_data
    └─ C：CPU 执行临时空间
```

三者的生命周期和分配器不同：

```text
context metadata：context 创建/释放
backend buffer：galloc/tallocr/backend buffer type
workspace：CPU backend context 按 graph plan 按需增长
```

## 22.2 从上到下的数据结构层次

```text
ggml_context
  └─ ggml_object 链表
      ├─ ggml_tensor metadata
      └─ ggml_cgraph metadata

ggml_gallocr
  ├─ buffer type 数组 bufts[]
  ├─ vbuffer buffers[]
  │   └─ backend buffer chunks[]
  ├─ dyn_tallocr buf_tallocs[]
  │   └─ tallocr_chunk[]
  │       └─ free_block[]
  ├─ hash_node hash_values[]
  ├─ tensor_alloc node_allocs[]
  └─ leaf_alloc leaf_allocs[]

CPU backend
  ├─ ggml_backend_cpu_context
  │   └─ work_data/work_size
  └─ ggml_cplan
```

可以把它们归纳为四层：

| 层 | 代表结构 | 主要职责 |
|---|---|---|
| 元数据层 | `ggml_context`、`ggml_object`、`ggml_tensor` | 描述 tensor/graph，不负责 backend 数据 placement |
| 图规划层 | `ggml_cgraph`、`ggml_gallocr`、`hash_node` | 根据拓扑和生命周期规划 allocation |
| 地址分配层 | `ggml_dyn_tallocr`、`tallocr_chunk`、`free_block` | 在 chunk 中寻找、释放、复用地址 |
| backend 存储层 | `vbuffer`、`ggml_backend_buffer` | 创建真实 CPU/GPU buffer 并管理底层数据 |

---

# 23. 元数据层：`ggml_object`、`ggml_context`、`ggml_tensor`

## 23.1 `ggml_object`：context 中的对象记录

`src/ggml.c`：

```cpp
struct ggml_object {
    size_t offs;
    size_t size;
    struct ggml_object * next;
    enum ggml_object_type type;
    char padding[4];
};
```

它不是 tensor 数据本身，而是 context 内存池中的一条对象记录：

```text
offs：对象内容相对于 context buffer 的偏移
size：对象占用的对齐后空间
next：下一个 object
 type：TENSOR、GRAPH 等对象类型
```

`ggml_new_object()` 采用 bump-pointer 风格从 context buffer 尾部追加对象：

```text
context buffer
+---------+---------+---------+
| object  | object  | object  |
| tensor  | graph   | tensor  |
+---------+---------+---------+
```

这种分配方式的特点：

```text
优点：简单、快速、低碎片、适合构图
缺点：单个 tensor metadata 不能独立释放
回收方式：通常随 context 整体释放或 reset
```

## 23.2 `ggml_context`：构图 metadata arena

```cpp
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool mem_buffer_owned;
    bool no_alloc;
    int n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

关键字段：

```text
mem_buffer：context 管理的连续区域
mem_size：容量
mem_buffer_owned：是否由 context 自己申请并负责释放
no_alloc：禁止在 context 内为 tensor data 分配实际数据
objects_begin/end：object 链表
```

simple-backend 构图时：

```cpp
.no_alloc = true
```

这意味着 `ggml_new_tensor_2d` 只创建 tensor metadata，不在 context 中申请 tensor 数据区；真正的数据空间由后续 backend allocator 分配。

## 23.3 `ggml_tensor`：逻辑 tensor 与物理 placement 的连接点

内存相关字段：

```cpp
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    enum ggml_op op;
    struct ggml_tensor * src[GGML_MAX_SRC];
    struct ggml_tensor * view_src;
    size_t view_offs;
    void * data;
    int32_t flags;
    void * extra;
};
```

字段职责：

```text
type：元素/量化类型，决定 size、block 和 kernel
ne[]：各维元素数量
nb[]：各维字节 stride
buffer：归属哪个 backend buffer
data：实际数据地址
view_src/view_offs：view 的共享存储关系
src[]：计算依赖，不等同于内存 ownership
flags：INPUT、OUTPUT、COMPUTE 等生命周期/执行标记
extra：backend 私有附加信息
```

特别要区分：

```text
tensor->data：数据位置
node->src[]：计算依赖
view_src：物理存储共享
buffer：backend 存储容器
```

## 23.4 普通 tensor、view、预分配 tensor

### 普通 tensor

```text
tensor->view_src == NULL
tensor->data == buffer base + allocated offset
```

### view tensor

```text
view->view_src = source
view->data = source->data + view_offs
view->buffer = source->buffer
```

view 不新申请 data，但会延长源 tensor 的生命周期。

### 预分配 tensor

如果 tensor 已经有：

```cpp
tensor->buffer != NULL
tensor->data != NULL
```

allocator 可能把它视为外部/预分配对象，不再为它重新申请独立区域。

---

# 24. 图规划层：`ggml_cgraph` 与 `ggml_gallocr`

## 24.1 `ggml_cgraph`：决定使用关系

计算图保存：

```text
nodes[]：需要执行的计算 node
leafs[]：输入/常量等叶子
use_counts[]：图中的使用计数
visited_hash_set：构图去重
```

对内存分配来说最重要的是：

```text
graph 的 node 顺序
每个 node 的 src[]
leaf/node 的 flags
```

它们决定：

```text
tensor 何时首次需要
tensor 最后一次被谁使用
tensor 何时可以回收
```

## 24.2 `ggml_gallocr`：图级 allocation planner

```cpp
struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts;
    struct vbuffer ** buffers;
    struct ggml_dyn_tallocr ** buf_tallocs;
    int n_buffers;

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values;

    struct node_alloc * node_allocs;
    int n_nodes;
    struct leaf_alloc * leaf_allocs;
    int n_leafs;
};
```

它不直接等于某一块内存，而是管理：

```text
不同 buffer type 的 allocator
各 backend 的 vbuffer
每个 tensor 的规划地址
每个 tensor 的生命周期状态
```

`ggml_gallocr_new_n()` 还会检查多个 buffer type 是否相同；如果相同，则共享同一个 `dyn_tallocr`，避免对相同存储类型重复维护 allocator。

## 24.3 `hash_node`：每个 tensor 的 allocation 状态

```cpp
struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};
```

这是 graph allocator 的关键运行时状态：

```text
n_children：仍会读取该 tensor 的 node 数
n_views：仍依赖该 tensor 的 view 数
buffer_id：使用哪个 buffer type/backend allocation
addr：chunk + chunk 内 offset
allocated：当前是否已占据空间
```

可释放条件通常是：

```cpp
n_children == 0 && n_views == 0 && allocated
```

但 output、allocation dependency、外部预分配等情况会改变这个判断。

## 24.4 `tensor_alloc` / `node_alloc` / `leaf_alloc`

```cpp
struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max;
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};
```

它们保存的是“规划结果”，不是 tensor 本身：

```text
buffer_id：放到哪个 backend buffer
addr：放到哪个 chunk/offset
size_max：该 tensor 规划时需要的最大 allocation size
```

之后 `ggml_gallocr_init_tensor()` 使用这些规划结果设置真实 tensor placement。

---

# 25. 地址分配层：`dyn_tallocr`、chunk、free block

## 25.1 `ggml_dyn_tallocr`

```cpp
struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;
};
```

它负责在 graph allocator 给出的某一种 buffer type 内：

```text
按 alignment 分配地址
选择已有空闲块
创建新 chunk
释放并合并空闲块
```

## 25.2 `buffer_address`

```cpp
struct buffer_address {
    int chunk;
    size_t offset;
};
```

这是跨多个实际 backend buffer 的逻辑地址：

```text
chunk：第几个 backend buffer
offset：该 buffer 内的偏移
```

它替代了单一连续 buffer 中的裸 offset，使 allocator 能管理多个 chunk。

## 25.3 `tallocr_chunk`

```cpp
struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};
```

一个 chunk 的核心状态是：

```text
free_blocks：可复用的空闲区间
n_free_blocks：空闲区间数量
max_size：当前需要的 chunk 逻辑容量
```

## 25.4 `free_block`

```cpp
struct free_block {
    size_t offset;
    size_t size;
};
```

allocator 释放 tensor 时，并不立即释放整个 backend buffer，而是：

```text
tensor 区域
    ↓
插入 free_blocks
    ↓
尝试与相邻 block 合并
    ↓
供后继 tensor 复用
```

这就是 graph 内存复用的直接数据结构基础。

## 25.5 chunk 的真实 backend 映射

```cpp
struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};
```

`vbuffer` 把多个实际 backend buffer 聚合成一个逻辑 buffer。创建时：

```cpp
for (int n = 0; n < talloc->n_chunks; n++) {
    size_t chunk_size =
        talloc->chunks[n]->max_size;

    buf->chunks[n] =
        ggml_backend_buft_alloc_buffer(
            buft,
            chunk_size);
}
```

CPU 路径：

```text
vbuffer chunk
  → ggml_backend_buft_alloc_buffer
    → ggml_backend_cpu_buffer_type_alloc_buffer
      → ggml_aligned_malloc
```

GPU 路径则由 CUDA/Metal/Vulkan 的 buffer type 提供底层 allocation。

## 25.6 分配策略

`ggml_dyn_tallocr_alloc()` 会：

```text
1. 将 size 按 alignment 对齐
2. 在已有 chunk 的非尾部 free block 中寻找 best fit
3. 如果没有合适 block，尝试 chunk 尾部增长
4. 如果仍不够，创建新 chunk
5. 返回 {chunk, offset}
```

释放时：

```cpp
ggml_dyn_tallocr_free_bytes(
    alloc,
    addr,
    size);
```

会将空间放回 free block，并尽可能和前后相邻块合并。

---

# 26. Backend 存储层：`ggml_backend_buffer`

## 26.1 buffer 对象

```cpp
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;
    ggml_backend_buffer_type_t buft;
    void * context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};
```

它是 backend 存储对象，不是单个 tensor：

```text
iface：free、set/get、copy、clear 等操作
buft：CPU/CUDA/Metal 等 buffer type
context：backend 私有上下文
size：整块 buffer 容量
usage：WEIGHTS 或 COMPUTE 等用途
```

## 26.2 `ggml_backend_tensor_alloc`

普通 tensor 最终通过：

```cpp
ggml_backend_tensor_alloc(
    buffer,
    tensor,
    addr);
```

设置：

```cpp
tensor->buffer = buffer;
tensor->data = addr;
```

然后调用：

```cpp
ggml_backend_buffer_init_tensor(buffer, tensor);
```

让 backend 为该 tensor 做额外初始化。

## 26.3 `ggml_backend_view_init`

view 不重新分配 data：

```cpp
tensor->buffer = tensor->view_src->buffer;
tensor->data =
    (char *) tensor->view_src->data
    + tensor->view_offs;
```

之后同样调用 backend 的 tensor initialization。这个过程体现：

```text
storage alias
≠
独立 allocation
```

---

# 27. 一次 tensor allocation 的完整生命周期

以中间 tensor `t` 为例：

```text
1. ggml_new_tensor 创建 metadata
   t->data 可能为 NULL
   t->buffer 为 NULL

2. graph 构建登记 t 的 op/src

3. galloc 扫描 graph
   统计 t 的 children/views

4. dyn_tallocr 规划：
   buffer_id + chunk + offset + size_max

5. vbuffer/backend buffer 创建
   必要时触发 aligned_malloc/cudaMalloc 等

6. ggml_gallocr_init_tensor
   调用 ggml_backend_tensor_alloc

7. t->buffer/t->data 有效

8. backend 执行 graph

9. 最后一个 consumer 执行后：
   n_children--

10. n_children == 0 且 n_views == 0：
    free_node

11. 区域回到 free_blocks

12. 后继 tensor 复用同一 chunk/offset
```

如果 `t` 是 view：

```text
第 5/6 步不申请独立数据
view_init 继承 view_src 的 buffer/data
并通过 n_views 延长源 tensor 生命周期
```

---

# 28. 内存高效性的核心机制总结

GGML 的内存效率不是由单一 allocator 带来的，而是多个结构协同：

```text
ggml_context / ggml_object
    低成本管理构图 metadata

no_alloc context
    将 metadata 和 backend data allocation 解耦

ggml_cgraph
    提供拓扑和使用关系

ggml_gallocr / hash_node
    进行 graph 级生命周期规划

ggml_dyn_tallocr / free_block
    在地址层复用已释放区域

chunk / vbuffer
    适配多 buffer、容量上限和 backend allocation

ggml_tensor view_src/view_offs
    零拷贝共享 storage

allocation dependency
    在异步 backend 下延长必要生命周期

CPU cplan.work_data
    独立管理算子执行临时空间
```

可以概括为：

```text
元数据 arena
+ graph lifetime analysis
+ best-fit free block reuse
+ chunked backend buffers
+ zero-copy views
+ backend-aware placement
+ execution workspace reuse
```

---

# 29. 内存调试建议

## 29.1 观察 context metadata

```gdb
p ctx->mem_size
p ctx->mem_buffer
p ctx->no_alloc
p ctx->n_objects
p ctx->objects_begin
p ctx->objects_end
```

遍历 object 时观察：

```gdb
p obj->offs
p obj->size
p obj->type
p obj->next
```

## 29.2 观察 tensor placement

```gdb
p tensor->name
p tensor->data
p tensor->buffer
p tensor->view_src
p tensor->view_offs
p ggml_nbytes(tensor)
```

## 29.3 观察 graph allocator

```gdb
p galloc->n_buffers
p galloc->n_nodes
p galloc->n_leafs
p hn->n_children
p hn->n_views
p hn->buffer_id
p hn->addr.chunk
p hn->addr.offset
p hn->allocated
```

## 29.4 观察 chunk/free block

```gdb
p alloc->n_chunks
p alloc->alignment
p alloc->max_chunk_size
p chunk->n_free_blocks
p chunk->max_size
p chunk->free_blocks[i].offset
p chunk->free_blocks[i].size
```

## 29.5 推荐断点

```gdb
break ggml_gallocr_allocate_node
break ggml_gallocr_free_node
break ggml_dyn_tallocr_alloc
break ggml_dyn_tallocr_free_bytes
break ggml_dyn_tallocr_new_chunk
break ggml_backend_buft_alloc_buffer
break ggml_backend_tensor_alloc
break ggml_backend_view_init
```

重点验证：

```text
同一 tensor 的规划地址何时产生
最后一个 consumer 执行后何时释放
后继 tensor 是否复用同一 free block
view 是否延长源 tensor 生命周期
是否因为 output/alloc_dep 阻止释放
```

---

# 30. 内存系统的最终闭环

```text
context
  → object
    → tensor metadata
      → cgraph topology
        → gallocr lifetime analysis
          → hash_node use/view counts
            → dyn_tallocr allocation
              → chunk/free_block
                → vbuffer/backend buffer
                  → tensor->buffer/tensor->data
                    → CPU/GPU kernel 使用
                      → children/views 归零
                        → free_block
                          → 后继 tensor 复用
```

对 GGML 来说，“高效内存分配”不是简单的 `malloc/free`，而是：

```text
图级生命周期驱动的静态/动态混合规划
```

其中：

```text
ggml_object：管理构图对象
chunk：承载分段 buffer
free_block：表示可复用区间
hash_node：记录 tensor 生命周期
tensor_alloc：记录规划地址
vbuffer：聚合实际 backend buffer
ggml_backend_buffer：提供 backend 存储和操作接口
```

这套结构共同实现：

```text
低碎片
低峰值内存
跨 backend placement
零拷贝 view
中间 tensor 复用
异步执行安全
```



---

# 40. 架构师视角：Tensor 布局、核心算子与计算图

## 40.1 四个执行域与两个横向支撑域

GGML 可以按职责划分为四个执行域：

```text
Tensor Layout Domain
    ↓
Operator Expression Domain
    ↓
Compute Graph Domain
    ↓
Backend Execution Domain
```

横向支撑域：

```text
Memory Planning Domain
Scheduling / Synchronization Domain
```

完整关系：

```text
Tensor Layout
    ├─ 描述数据是什么、如何排布、如何访问
    ↓
Core Operator
    ├─ 描述对 Tensor 做什么数学操作
    ↓
Compute Graph
    ├─ 描述多个 Operator 的依赖、顺序和生命周期
    ↓
Memory Planner / Scheduler
    ├─ 决定 Tensor 放在哪里、何时分配、何时复用
    ↓
Backend Kernel
    └─ 使用具体硬件上的高性能实现执行
```

simple-backend 的矩阵乘法路径：

```text
model.a / model.b
    ↓
ggml_mul_mat(a, b)
    ↓
result tensor，op = GGML_OP_MUL_MAT
    ↓
ggml_build_forward_expand
    ↓
ggml_cgraph
    ↓
scheduler / allocator
    ↓
CPU backend
    ↓
ggml_compute_forward_mul_mat
    ↓
vec_dot
```

`ggml_mul_mat()` 是创建计算表达式的函数，不是立即执行矩阵乘法的 kernel。

## 40.2 三层为什么必须拆开

| 层次 | 解决的问题 | 典型对象 |
|---|---|---|
| Tensor Layout | 数据是什么、占多少空间、如何寻址 | `ggml_tensor` |
| Core Operator | 对输入执行什么数学操作 | `ggml_op`、`ggml_mul_mat` |
| Compute Graph | 多个操作如何依赖、排序和调度 | `ggml_cgraph` |

如果在 `ggml_mul_mat()` 中立即完成内存申请、backend 选择、线程规划、kernel 执行和同步，就无法有效完成：

```text
全局内存复用、算子融合、跨 backend 调度、异步执行、量化适配、workspace 规划
```

GGML 因此采用：

```text
先表达计算意图
    ↓
收集完整 graph
    ↓
统一规划内存和 backend
    ↓
交给最合适的 kernel
```

---

# 41. Tensor Layout Domain：数据如何表示和访问

## 41.1 `ggml_tensor` 是逻辑对象与物理数据的连接点

```cpp
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    enum ggml_op op;
    struct ggml_tensor * src[GGML_MAX_SRC];
    struct ggml_tensor * view_src;
    size_t view_offs;
    void * data;
    int32_t flags;
    void * extra;
};
```

```text
type/ne/nb：逻辑形状与物理布局
op/src/flags：计算语义与依赖
buffer/data：backend 存储和实际地址
view_src/view_offs：零拷贝别名关系
```

## 41.2 `ne[]` 与 `nb[]`

`ne[]` 表示逻辑维度元素数量；`nb[]` 表示沿维度移动时的字节 stride。地址公式为：

```cpp
(char *) tensor->data
    + i0 * tensor->nb[0]
    + i1 * tensor->nb[1]
    + i2 * tensor->nb[2]
    + i3 * tensor->nb[3];
```

因此：

```text
ne[] 决定逻辑形状
nb[] 决定物理布局
```

GGML 第 0 维通常是连续维，这有利于 SIMD、cache locality、block tiling 和 GPU coalescing。

## 41.3 `type` 与量化布局

`GGML_TYPE_F32/F16/BF16` 按元素类型解释数据；Q4/Q5/Q6/IQ 等量化类型则以量化 block 为基本单元。实际数据大小和行跨度由：

```cpp
ggml_type_size(type)
ggml_blck_size(type)
ggml_row_size(type, ne[0])
```

共同决定，不能统一简化为 `nelements * sizeof(float)`。

## 41.4 view 和零拷贝布局变换

```cpp
view->buffer = view_src->buffer;
view->data = (char *) view_src->data + view_offs;
```

view 只改变 data 起点、shape 和 stride，不复制底层数据；allocator 用 `n_views` 保证源 storage 在 view 使用结束前不被回收。

---

# 42. Core Operator Domain：算子是计算意图

`ggml_mul_mat()` 主要完成：

```text
检查形状
推导输出 shape
创建 result metadata
设置 result->op = GGML_OP_MUL_MAT
设置 result->src[0/1]
```

它不读取输入数据、不分配 backend output、不调用 CPU/GPU kernel。它建立的是：

```text
result = MUL_MAT(a, b)
```

类似编译器中的 AST/IR 节点，实际执行被延迟到 graph/backend 阶段。

延迟的原因是调用时还不知道：

```text
使用 CPU 还是 GPU
是否需要跨 backend copy
输出是否可以复用中间 buffer
是否可以融合后续 op
是否需要量化转换
workspace 和异步依赖如何安排
```

---

# 43. Compute Graph Domain：图是全局优化基础

`ggml_cgraph` 保存：

```text
nodes[]：需要执行的计算节点
leafs[]：输入/常量
src[]：依赖边
use_counts[]：使用关系
```

simple-backend 的图：

```text
a ─┐
   ├─ MUL_MAT → result
b ─┘
```

构图阶段仍不计算数值。完整 graph 让 allocator 能计算 Tensor 的最后一个 consumer：

```text
a → node1
a → node2

初始：a.n_children = 2
node1 完成：a.n_children = 1
node2 完成：a.n_children = 0
```

若同时 `a.n_views == 0`，a 的物理空间才能回收到 `free_block` 并复用。

---

# 44. 为什么不直接执行高性能算子

直接执行模型会把：

```text
检查 layout → 选择 backend → 申请 output → 执行 kernel → 同步
```

压在一次算子调用内。它简单，但看不到完整 graph，无法做好全局内存复用、跨 backend 调度、算子融合、异步依赖和 workspace 复用。

GGML 的分层执行模型是：

```text
表达式构建
    ↓
Graph 收集
    ↓
Memory planning
    ↓
Backend scheduling
    ↓
Kernel dispatch
```

额外的控制层换来更低的分配次数、更低的峰值内存、跨硬件执行和全局优化。

---

# 45. GGML 如何通过布局和调度实现高性能

## 45.1 布局效率

连续维优先、显式 stride 和量化 block 布局使 kernel 能够：

```text
使用 SIMD/向量化加载
保持 cache locality
进行 block tiling
实现 GPU coalesced access
避免整模型预先反量化
```

量化路径通常是：

```text
量化权重 block
    ↓
vec_dot 读取 scale/min/quant values
    ↓
边解码边计算 dot product
    ↓
写入 F32/F16 result
```

## 45.2 Backend-aware scheduling

scheduler 综合判断：

```text
backend 是否支持 op
backend 是否支持 buffer type
source buffer 是否能被目标 backend 直接消费
权重在哪个 backend
是否适合 offload
```

据此生成 `node_backend_id` 并切分为多个 split。

## 45.3 Split 与异步 copy

```text
生产 split
    ↓
异步或同步 tensor copy
    ↓
消费 split
```

多 copy 模式还可以使用不同 copy slot，配合 `events[backend][copy]` 防止不同轮次覆盖仍在使用的 buffer。

## 45.4 Graph allocator 降低峰值内存

```text
按 graph 生命周期分配
    ↓
最后一个 consumer 完成
    ↓
区域回收到 free_block
    ↓
后续 Tensor 复用
```

## 45.5 CPU plan 复用 workspace

`ggml_graph_plan()` 提前计算 task 数和 workspace 需求，执行阶段复用 `cplan.work_data`，避免 kernel 内频繁 malloc。

---

# 46. 六类核心职责边界

```text
Tensor Layout：type、ne、nb、data、view
    描述数据是什么以及如何访问

Operator：op、op_params、src[]
    描述数学变换

Compute Graph：nodes、leafs、use_counts
    描述全局依赖和执行顺序

Memory Planner：gallocr、hash_node、tensor_alloc、chunk、free_block
    规划 placement、生命周期和复用

Scheduler：splits、backend ids、events、tensor copies
    决定 backend、copy 和同步

Backend Kernel：compute_forward、MUL_MAT、vec_dot
    在具体硬件上执行数学计算
```

---

# 47. 从表达式到高性能 kernel 的闭环

```text
Tensor Layout
  → Operator Expression
    → Compute Graph
      → Memory Planner
        → Backend Scheduler
          → CPU graph plan / GPU command queue
            → operator dispatch
              → specialized kernel
                → vec_dot / SIMD / GPU instruction
```

CPU simple-backend 路径：

```text
ggml_backend_sched_graph_compute
  → ggml_backend_sched_compute_splits
    → ggml_backend_graph_compute_async
      → ggml_backend_cpu_graph_compute
        → ggml_graph_plan
          → ggml_graph_compute_thread
            → ggml_compute_forward
              → GGML_OP_MUL_MAT
                → ggml_compute_forward_mul_mat
                  → vec_dot
```

最终性能来自：

```text
布局效率
+ 图级内存复用
+ backend-aware scheduling
+ 专用 kernel
+ 量化计算
+ 多线程/异步执行
```

这就是 GGML 将“表达计算”和“执行计算”拆开的根本原因。
