# examples/mnist × ggml-cuda：GPU 断点调试与设计思想

> 源码基线：云端 `/home/ggml`，commit `353b63b4`（ggml 0.25.3）
>
> 运行环境：RTX 4090（`cc=890`、`nsm=128`、`warp_size=32`、`sharedMemPerBlockOptin=101,376 B`、VMM 可用）、
> Ubuntu 22.04、CUDA 12.8、驱动 570.124.04、gdb 12.1、Debug 构建（`CMAKE_BUILD_TYPE=Debug`，`GGML_CUDA=ON`）
>
> 编译期开关：`GGML_CUDA_GRAPHS=OFF`（无 `GGML_CUDA_USE_GRAPHS`）、`GGML_CUDA_FORCE_CUBLAS=OFF`、
> `GGML_CUDA_FORCE_MMQ=OFF`、`GGML_CUDA_FA=OFF`、`GGML_SCHED_MAX_COPIES=4`（但运行期 `n_copies=1`）
>
> 本文档不是源码导读，而是**用断点数据反推设计**：每看到一个数值，回答三个问题——它是什么、为什么是这个值、如果改掉会发生什么。

---

## 0 为什么用 mnist 研究 ggml-cuda 的训练链路

`simple-backend`（见 `analysis/L5/ggml-gpu/`）只有 1 个 `MUL_MAT`，能覆盖“推理”的最小闭环，但覆盖不到下面这些东西：

| 能力 | simple-backend | mnist-fc |
|---|---|---|
| 前向 | 1 个 MUL_MAT | 2 个 MUL_MAT + ADD + RELU |
| 损失 | 无 | CROSS_ENTROPY_LOSS + SCALE |
| 反向 | 无 | 11 个梯度节点（OUT_PROD / REPEAT_BACK / STEP / MUL） |
| 优化器 | 无 | 4 个 OPT_STEP_ADAMW（原地更新参数） |
| 多图交替 | 无 | `gb_grad`(28) ↔ `gb_opt`(32) 每个 batch 切换 |
| 跨后端拷贝 | 无 | `adamw_params` CPU→CUDA，`n_inputs=1` |
| 静态常驻显存 | 无 | 权重 + 输入 + 梯度累加器 + AdamW m/v 共 7.95 MB |
| 数据流 | 一次 tensor_set | 每 batch 100 次 H2D（shard 化） |
| 内存复用 | 3 个 tensor | 28 个节点、offset 复用链、原地链 |

mnist-fc 的参数规模（`784→500→10`）小到可以在 gdb 里逐个节点检查，但训练语义完整：前向、交叉熵、反向传播、AdamW 更新、验证集评估，一个都不少。它把 ggml 的**训练图机制**（`ggml-opt` + `ggml_build_backward_expand`）和**CUDA 后端机制**（split、gallocr、cuBLAS、kernel 选择）串在了一起。

---

## 1 一次训练的真实数据流

```text
host                                                    device (RTX 4090)
────                                                    ─────────────────
dataset->data (190,560,000 B, CPU buffer, 常驻)
  images [784, 60000]  labels [10, 60000]

每个物理 batch（500 张图）：
  ggml_opt_alloc(backward=true)  ── 选图：gb_grad 或 gb_opt
      sched_reset
      split_graph     → n_splits=1, backend=CUDA0
      alloc_graph     → graph_copy[0] 在 CPU/CUDA 间跳 ⇒ backend_ids_changed ⇒ reserve_n
      （无 cudaMalloc，buffer 已在第 2 次 reserve 后固定为 3,588,352 B；
        每 batch 调度总成本 ~57 µs，占总时间 3%，不是瓶颈）
  dataset_get_batch  ── 50 次 images shard + 50 次 labels shard
      cudaMemcpyAsync(H2D, cudaStreamPerThread) + cudaStreamSynchronize
                                                  ──────► images/labels (CUDA buffer)
  ggml_opt_eval
      sched_graph_compute
          compute_splits → cuda_graph_compute
              23~27 次 ggml_cuda_compute_forward
                  MUL_MAT   → cublasSgemm（T,N）
                  OUT_PROD  → cublasSgemm
                  ADD/RELU → k_bin_bcast / unary_op_kernel
                  CE_LOSS   → cross_entropy_loss_f32 + sum_f32
                  ADAMW     → opt_step_adamw_f32
          sched_synchronize（cudaStreamSynchronize）
      tensor_get(loss) / tensor_get(pred) / tensor_get(ncorrect)
                                                  ◄────── 4 B + 2,000 B + 8 B
```

实测：30 epoch 全量训练 **6 s**（代码插桩，3,420 个训练物理 batch），约 **1.8 ms/batch**（H2D 1,138 µs + compute 490 µs + 调度 158 µs）；
验证集准确率 98.03%，最终保存的 GGUF 在 `mnist-eval` 上 10,000 张图 97.88%。

---

## 2 六层对象模型（mnist 版本）

```text
┌──────────────────────────────────────────────────────────────────────┐
│ ggml_backend_registry（进程级 static）                                │
│   CUDA reg → device CUDA0；CPU reg → device CPU                       │
└───────────────┬──────────────────────────────────────────────────────┘
                │ ggml_backend_dev_by_name("CUDA0")
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ggml_backend_device (CUDA0)                                          │
│   get_buffer_type() → CUDA0 buft（alignment=128, max_size=SIZE_MAX）  │
│   supports_op()     → 决定 node 能否落在这个后端                       │
└───────────────┬──────────────────────────────────────────────────────┘
                │ ggml_backend_dev_init()
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ggml_backend (×3: CUDA0, CUDA0, CPU)                                 │
│   context: ggml_backend_cuda_context                                 │
│     streams[8][8] / cublas_handles[8][8] / pools[8][8]  —— 全部惰性    │
└───────────────┬──────────────────────────────────────────────────────┘
                │
     ┌──────────┴───────────┐
     ▼                      ▼
┌─────────────────┐   ┌──────────────────────────────────────────────┐
│ ggml_backend_sched │ │ ggml_gallocr                                │
│  决定 node 归属     │ │  决定 tensor 在哪个 buffer 的哪个 offset      │
│  切 split、插 copy  │ │  生命周期分析 + 复用                          │
└─────────────────┘   └──────────────────────────────────────────────┘
     │                      │
     ▼                      ▼
┌─────────────────┐   ┌──────────────────────────────────────────────┐
│ ggml_backend_buffer │ │ 静态 buffer：ctx_static / ctx_gguf / dataset │
│  计算图临时显存      │ │  生命周期 = 整个程序                          │
└─────────────────┘   └──────────────────────────────────────────────┘
```

mnist 里同时存在 **5 个 buffer**：

| buffer | 后端 | 大小 | 内容 | 生命周期 |
|---|---|---|---|---|
| dataset->buf | CPU | 190,560,000 | 60,000 张图 + one-hot 标签 | 进程 |
| `model.buf_static` | CUDA0 | 3,158,272 | 4 个权重 + `images` | 进程 |
| `opt_ctx->buf_static` | CUDA0 | 4,795,520 | labels、loss、pred、ncorrect、grad_accs、AdamW m/v | 进程 |
| `opt_ctx->buf_cpu` | CPU | 32 | `adamw_params`（7 个 float） | 进程 |
| `galloc` compute buffer | CUDA0 | 3,588,352 | 中间激活与梯度 | 图生命周期内复用 |

**为什么静态 buffer 和计算 buffer 要分开？**
静态张量的地址在整个训练过程中必须稳定（优化器要原地更新权重、`images` 每步被覆写、`grad_accs` 跨 batch 累加），所以它们不能交给会重排 offset 的 gallocr；而中间激活每步都可丢弃重算，交给 gallocr 才能复用。这个边界就是 `ggml_set_input` / `ggml_set_output` / `GGML_TENSOR_FLAG_PARAM` 三个标记。

---

## 3 关键数值总表（gdb 实测 + 源码推导）

### 3.1 结构体与元数据

| 项目 | 值 | 推导 / 来源 |
|---|---|---|
| `sizeof(ggml_tensor)` | 336 B | gdb `print sizeof` |
| `sizeof(ggml_object)` | 32 B | gdb |
| `ggml_tensor_overhead()` | 368 B | 32 + 336 |
| `sizeof(ggml_cgraph)` | 96 B | gdb |
| `ggml_graph_overhead()` | 82,608 B | 32 + pad16(96 + 2×2048×8 + 4099×4 + 4099×8 + 516)；`hash_size = ggml_hash_size(2×2048) = 4099` |
| `ctx_static` mem_size | 376,832 B | 1024 × 368 |
| `ctx_compute` mem_size | 1,001,488 B | 2048 × 368 + 3 × 82,608 |
| `opt ctx_static` mem_size | 8,096 B | 22 × 368 |
| scheduler `context_buffer_size` | 41,370,288 B | 2048 × 30 × 2 × 336 + 82,608 |
| CUDA buft alignment | 128 B | `ggml_backend_cuda_buffer_type_get_alignment` |
| CUDA buft max_size | `SIZE_MAX` | iface 中为 NULL，取默认 |
| `GGML_MEM_ALIGN` | 16 B | context 对象池对齐 |

### 3.2 张量

| 张量 | type | ne | nb | nbytes |
|---|---|---|---|---|
| `fc1.weight` | F32 | `[784,500,1,1]` | `[4,3136,1568000,1568000]` | 1,568,000 |
| `fc1.bias` | F32 | `[500,1,1,1]` | `[4,2000,2000,2000]` | 2,000 |
| `fc2.weight` | F32 | `[500,10,1,1]` | `[4,2000,20000,20000]` | 20,000 |
| `fc2.bias` | F32 | `[10,1,1,1]` | `[4,40,40,40]` | 40 |
| `images` | F32 | `[784,500,1,1]` | `[4,3136,1568000,1568000]` | 1,568,000 |
| `labels` | F32 | `[10,500,1,1]` | `[4,40,20000,20000]` | 20,000 |
| `logits` | F32 | `[10,500,1,1]` | `[4,40,20000,20000]` | 20,000 |
| `loss_cross_entropy` | F32 | `[1,1,1,1]` | `[4,4,4,4]` | 4 |
| `pred` | I32(26) | `[500,1,1,1]` | `[4,2000,2000,2000]` | 2,000 |
| `ncorrect` | I64(27) | `[1,1,1,1]` | `[8,8,8,8]` | 8 |

### 3.3 图与调度

| 项目 | 训练 | 推理 |
|---|---|---|
| 前向图 `gf` | 14 节点 / 2 叶子 | 13 节点 / 2 叶子 |
| 梯度图 `gb_grad` | 28 节点 / 7 叶子 | 无 |
| 优化图 `gb_opt` | 32 节点 / 16 叶子 | 无 |
| scheduler `graph_copy` | 28 / 34 节点 | 13 节点 |
| `n_splits` | 1（CUDA0） | 1（CUDA0） |
| `n_inputs` | 0（grad）/ 1（opt，`adamw_params`） | 0 |
| `n_copies` | 1 | 1 |
| compute buffer | 3,588,352 B | 1,020,160 B |
| `ggml_cuda_compute_forward` 次数 | 23（grad）/ 27（opt） | 11 |

### 3.4 kernel

| 节点 | kernel | grid | block | 备注 |
|---|---|---|---|---|
| `node_4` ADD [500,500] | `k_bin_bcast` | (2,500,1) | (128,1,1) | 原地复用 `node_2` |
| `logits` ADD [10,500] | `k_bin_bcast` | (1,20,1) | (5,25,1) | 原地复用 `node_6` |
| `node_5` RELU | `unary_op_kernel` | 977 | 256 | 原地复用 |
| `node_2`/`node_6` MUL_MAT | `cublasSgemm` | — | — | `(M,N,K)=(500,500,784)` / `(10,500,500)` |
| `loss_cross_entropy` | `cross_entropy_loss_f32` | 500 | 32 | shared=40 B + `sum_f32` |
| `pred` / `node_12` | `argmax_f32` | 500 | 32 | 每行一个 block |
| `ncorrect` | `count_equal` | 4 | 32 | `nsm=128`，`dne=128` |
| `grad for node_6` | `cross_entropy_loss_back_f32` | 500 | 32 | |
| `node_16` / `node_24` | `k_repeat_back` | (1,1,1) / (16,1,1) | (32,1,1) | |
| `node_18` / `node_21` / `node_26` | `cublasSgemm` | — | — | OUT_PROD |
| `node_22` STEP | `unary_op_kernel` | 977 | 256 | 原地 |
| AdamW ×4 | `opt_step_adamw_f32` | 1 / 2 / 1532 / 20 | 256 | 原地更新参数 |

---

## 4 六条设计原则（由断点数据归纳）

### 4.1 构图与执行彻底分离

`mnist_model_build()` + `ggml_opt_init()` 期间，`ggml_cuda_compute_forward` **一次都没有命中**。建图只做三件事：

```text
ggml_new_tensor_*  → 从 context arena 切出 336 B 元数据
ggml_xxx()         → 设置 op / op_params / src[]
ggml_build_forward_expand → 后序 DFS 填 nodes[]/leafs[]，打 COMPUTE 标记
```

`data` 全部是 `NULL`，`buffer` 全部是 `NULL`。直到第一次 `ggml_backend_sched_alloc_graph` 才把地址填进去。

### 4.2 后端能力决定调度，优先级决定归属

`ggml_backend_sched_split_graph` 分五趟：先看张量已经在哪个 buffer 上，再按“权重所在后端优先”，再用 GPU 后端向上下扩展，最后按能力兜底。本例 3 个 backend（CUDA0、CUDA0、CPU），**所有节点都支持 CUDA，所以 `n_splits=1`，CPU 一次都没用到**。

### 4.3 内存按“生命周期”复用，不按引用计数

gallocr 在 reserve 阶段做一次静态分析：

```text
n_children（被多少节点消费）+ n_views（被多少视图引用）
节点执行完 → 父节点 n_children-- → 归零且无 view → free → 进入 free list
```

训练图 28 个节点，最终只需要 **3.42 MiB** 而不是全部张量之和（≈ 6.6 MiB），靠的就是这条链：

```text
node_2 (1,000,064) → node_4 → node_22 → grad for node_2   （同一块 1 MB）
node_5 (1,000,064) → grad for node_5
node_6 (20,096)   → logits
grad loss (128)   → node_16
```

### 4.4 kernel 按形态选路，不按“GPU 就用最快”

`MUL_MAT` 有五条路径（MMVF / MMF / MMVQ / MMQ / cuBLAS），选择依据是 `src0->type`、`cc`、`ne11`（batch 列数）。本例 `ne11=500`，远超 `should_use_mmvf` 的阈值（Ampere+ 为 `ne11<=3`），所以走 cuBLAS。反过来，真实 LLM decode（`ne11=1`）会走 MMVF。

### 4.5 惰性：能不做就不做

```text
CUDA stream    首次 ctx.stream() 才 cudaStreamCreateWithFlags
cuBLAS handle  首次 ctx.cublas_handle() 才 cublasCreate + 4 MiB workspace
CUDA pool      首次 ctx.pool() 才创建
kernel module  首次 launch 才加载
```
断点证据：第一个 `MUL_MAT` 命中时 `cublas_handles[0][0] == NULL`、`cublas_workspace_sizes[0] == 0`；第二个 `MUL_MAT` 时已变成 `0x...` 和 `4,194,304`。

---

### 4.6 瓶颈排序（反直觉但实测）

代码插桩（见 05 §8.3）：

```text
① 100 次 H2D + sync（dataset_get_batch）   1,138 µs/batch   64%
② kernel 执行 + D2H（ggml_opt_eval）         490 µs/batch   27%
③ 选图 + split + alloc（ggml_opt_alloc）     158 µs/batch    9%
```

**mnist 的瓶颈不在 GPU 计算，也不在调度，而在数据搬运。**

这与“训练是计算密集型”的直觉相反，原因是模型太小：500×784 的 GEMM 在 4090 上只要 19 µs，而 100 次 H2D 的同步开销就要 794 µs。这个结论不能直接外推到真实 LLM——后者的计算量把搬运摊销掉了——但它揭示了一个通用事实：**小模型/小 batch 场景下，调度和搬运的固定开销会主导总时间**。

---

## 5 源码逐行解读索引

每篇文档都包含“源码逐行”小节，把断点数据对应到具体代码行。索引如下：

| 文档 | 逐行解读的关键函数 | 覆盖的问题 |
|---|---|---|
| 01 | `ggml_init`（`ggml.c:1611`） | context arena 怎么分配、`no_alloc` 的作用 |
| 01 | `ggml_new_object`（`ggml.c:1708`） | 对象链表、16 B 对齐、为什么不用数组 |
| 01 | `ggml_new_tensor_impl`（`ggml.c:1766`） | `ne/nb` 推导、视图扁平化、三条 `data` 路径 |
| 01 | `ggml_backend_alloc_ctx_tensors_from_buft_impl`（`ggml-alloc.c:1168`） | 静态 buffer 的逐张量对齐、3,158,272 的推导 |
| 02 | `ggml_visit_parents_graph`（`ggml.c:7236`） | 后序 DFS、哈希去重、`use_counts`、nodes/leafs 三态 |
| 02 | `ggml_build_backward_expand`（`ggml.c:7345`） | `grads_needed` 传播、`ignore_src`、`ggml_add_or_set` |
| 02 | `ggml_opt_build`（`ggml-opt.cpp:322`） | 三段式建图、元数据预算、AdamW 追加、`ggml_graph_dup` 成本 |
| 03 | `ggml_backend_sched_split_graph`（`ggml-backend.cpp:1066`） | 五趟分析、`tensor_backend_id` 哈希索引、跨后端拷贝改写 |
| 03 | `ggml_backend_sched_compute_splits`（`ggml-backend.cpp:1646`） | split 间同步、INPUT 强制同步、event 记录 |
| 03 | `ggml_gallocr_alloc_graph_impl`（`ggml-alloc.c:718`） | 两遍统计、单遍分配、`n_children` 归零释放 |
| 03 | `ggml_gallocr_allocate_node`（`ggml-alloc.c:623`） | 原地复用的五个门槛、`free_extra_space` |
| 03 | `ggml_dyn_tallocr_alloc`（`ggml-alloc.c:202`） | 三段式搜索、best-fit、末尾块增长 |
| 04 | CUDA 节点遍历（`ggml-cuda.cu:4288`） | 两层跳过 + 融合跳过的完整顺序 |
| 04 | `ggml_cuda_mul_mat_cublas_impl<F32>`（`ggml-cuda.cu:1409`） | `(T,N)` 的由来、lda/ldb/ldc、四个 GEMM 分支 |
| 04 | `k_bin_bcast`（`binbcast.cu:34`） | 三维网格展平、`fastdiv`、广播 modulo、collapse 优化 |
| 04 | `unary_op_kernel`（`unary.cu:117`） | 函数指针模板、`op_relu` vs `op_step` |
| 04 | `cross_entropy_loss_f32`（`cross-entropy-loss.cu:8`） | 两阶段归约、shared memory 缓存、数值稳定 |
| 04 | `argmax_f32`（`argmax.cu:8`） | warp 树形归约、value/index 成对交换 |
| 04 | `count_equal`（`count-equal.cu:6`） | `dk` 分块、`atomicAdd` 的 32 位限制 |
| 04 | `opt_step_adamw_f32`（`opt-step-adamw.cu:6`） | AdamW 数学、bias correction、解耦权重衰减 |
| 04 | `ggml_backend_tensor_set/get`（`ggml-backend.cpp:335`） | 为什么用 `cudaStreamPerThread`、同步代价 |
| 05 | — | 断点 ↔ 源码行对照表（§11） |
| 06 | `ggml_cuda_try_fuse`（`ggml-cuda.cu:3436`） | 融合的接入点、返回值语义、`RELU_BACK` 的正确位置 |
| 06 | `ggml_gallocr_needs_realloc`（`ggml-alloc.c:1009`） | 三级判定、指纹缓存改造的完整代码 |
| 06 | `ggml_opt_dataset_get_batch`（`ggml-opt.cpp:163`） | shard 计算、permutation 间接寻址、cache miss |

---

## 6 文档地图

| 文档 | 内容 | 关键断点 |
|---|---|---|
| [01-init-and-alloc.md](01-init-and-alloc.md) | 3 个 backend、5 个 context、静态 buffer 账本、GGUF 加载、CUDA 惰性资源 | `ggml_init`、`ggml_backend_alloc_ctx_tensors`、`ggml_backend_cuda_init` |
| [02-build-graph.md](02-build-graph.md) | 前向表达式、loss/pred/ncorrect、反向展开、AdamW 图、节点编号与 flags | `ggml_opt.cpp:581`、`ggml_backend_cuda_graph_compute` 节点表 |
| [03-schedule-and-memory.md](03-schedule-and-memory.md) | split 决策、graph_copy、gallocr offset 规划、原地复用、realloc 现象 | `ggml_backend_sched_alloc_splits`、`ggml_dyn_tallocr_alloc` |
| [04-cuda-execution.md](04-cuda-execution.md) | 算子分发、cuBLAS 参数推导、8 类 kernel 的 grid/block、同步与数据流 | `ggml_cuda_compute_forward`、`ggml_cuda_mul_mat_cublas_impl` |
| [05-gdb-trace.md](05-gdb-trace.md) | 原始 gdb 脚本与输出、逐段解读、调试踩坑与方法论 | 全部 |
| [06-extension-guide.md](06-extension-guide.md) | 新增算子/kernel、调度与显存优化、性能瓶颈清单、映射到 LLM | — |

---

## 7 与 simple-backend 文档的关系

`analysis/L5/ggml-gpu/` 回答“最小推理图在 CUDA 上怎么跑”；本目录回答“**训练图**在 CUDA 上怎么跑”。

不变的部分：

```text
ggml_context / ggml_tensor / ggml_cgraph / gallocr 生命周期算法 / split 五趟分析 / kernel 选择框架
```

新增的部分：

```text
ggml-opt 的三张图（gf / gb_grad / gb_opt）
反向展开（OUT_PROD、REPEAT_BACK、STEP、CROSS_ENTROPY_LOSS_BACK）
AdamW 原地更新与 CPU→CUDA 参数拷贝
多图交替导致的 per-batch re-reserve
静态常驻显存（grad_accs / m / v）与计算 buffer 的边界
```

---

## 8 复现命令

```bash
# 云端
cd /home/ggml
cmake --build build-debug --target mnist-train mnist-eval -j 32

# 训练（30 epoch，约 7 s）
./build-debug/bin/mnist-train mnist-fc mnist-fc-f32.gguf \
  examples/mnist/data/MNIST/raw/train-images-idx3-ubyte \
  examples/mnist/data/MNIST/raw/train-labels-idx1-ubyte CUDA0

# 推理
./build-debug/bin/mnist-eval mnist-fc-f32.gguf \
  examples/mnist/data/MNIST/raw/t10k-images-idx3-ubyte \
  examples/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte CUDA0

# 调度可视化
GGML_SCHED_DEBUG=2 ./build-debug/bin/mnist-eval ... CUDA0
```

> **调试前提**：`mnist-train` 链接 `libggml.so` / `libggml-cpu.so` / `libggml-cuda.so`，
> 启动时这些 so 尚未加载。必须 `start` 到 `main` 之后再下 `ggml.c:xxx` / `ggml-cuda.cu:xxx` 断点，
> 否则报 `No source file named ggml.c`。
