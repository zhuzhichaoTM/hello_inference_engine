# simple-backend.cpp GPU 断点调试：GGML-CUDA 设计思想与全链路

> 源码基线：云端 `/home/ggml`，commit `353b63b`（ggml 0.25.3，2026-09-24）
>
> 运行环境：RTX 4090（sm_89 / `cc=890`）、Ubuntu 22.04、gdb 12.1、Debug 构建、`GGML_CUDA=ON`、`GGML_CUDA_GRAPHS=OFF`、`GGML_BACKEND_DL=OFF`
>
> 本文档不是源码导读，而是**用断点数据反推设计**：每看到一个数值，都回答三个问题——它是什么、为什么是这个值、如果改掉会发生什么。

---

## 0 为什么用 simple-backend 研究 ggml-cuda

`simple-backend.cpp` 只有 153 行、1 个算子、3 个 tensor，却完整穿过了 ggml 的每一层抽象：

```text
backend registry / device / backend / buffer_type / buffer
        ↓
scheduler（split、backend 归属、跨 backend copy）
        ↓
graph allocator（生命周期分析、显存申请、地址落位、复用）
        ↓
CUDA backend（stream、cuBLAS、pool、kernel 选择）
        ↓
具体 kernel（MMVF / MMF / MMQ / MMVQ / cuBLAS）
```

它像一颗“最小可运行种子”：图小到可以在 gdb 里逐值检查，路径却完整到可以解释真实 LLM 推理的每一个环节。把它的每个断点值吃透，再看 llama.cpp 的 `llm_build_*`，只是把 1 个 `MUL_MAT` 放大成几百个节点的同一套机制。

---

## 1 五层对象模型

ggml 的后端体系不是“一个 backend 包打天下”，而是把**身份、能力、执行状态、内存策略、内存实例**拆成五层：

| 层 | 类型 | 本程序中的实例 | 职责 | 生命周期 |
|---|---|---|---|---|
| 模块/注册表 | `ggml_backend_reg` | CUDA reg、CPU reg | 枚举设备、提供动态库入口 | 进程级，函数内 static |
| 设备 | `ggml_backend_device` | `CUDA0`、`CPU` | 身份 + 能力：类型、显存、`supports_op`、默认 buft、`init_backend` | 进程级 |
| 执行实例 | `ggml_backend` | `model.backend`、`model.cpu_backend` | 执行状态：stream、cuBLAS handle、pool、函数表 | `init_model` 创建，`main` 末尾释放 |
| 内存类型 | `ggml_backend_buffer_type` | `CUDA0`、`CPU` | 内存策略：alignment、alloc/free、alloc_size、是否 host | 进程级 static |
| 内存实例 | `ggml_backend_buffer` | 384 B 的 CUDA buffer | 一块真实显存 + tensor 读写接口 + usage | `alloc_graph` 申请，`sched_free` 释放 |

再往上叠两个“决策层”：

```text
ggml_backend_sched   决定「哪个 node 在哪个 backend 跑、图怎么切、跨 backend 怎么拷」
ggml_gallocr         决定「每个 tensor 放在哪块 buffer 的哪个 offset、何时可复用」
```

### 1.1 为什么这样分层

**设备与执行实例分离**：`CUDA0` 是身份（名字、显存、算力），`ggml_backend` 是执行状态（这个 stream、这个 cuBLAS handle）。同一个设备可以创建多个 backend 实例，各自拥有独立 stream/pool，用于多流并行或不同线程。

**buffer type 与 buffer 分离**：`CUDA0` 这个 buffer type 描述“CUDA 显存怎么分配、对齐多少、量化行怎么补齐”；384 B 的 buffer 是它的一次具体分配。同一设备可以有多种 buft：设备显存（`CUDA0`）、页锁定 host 内存（`CUDA_Host`）、host-mapped 内存。scheduler 判断“这个 node 能不能放到这个 backend”时，看的是 **buft 兼容性**，不是 backend 名字。

**scheduler 与 gallocr 分离**：scheduler 只关心“拓扑 + 后端能力”，不关心字节；gallocr 只关心“生命周期 + 字节”，不关心算子语义。两者通过 `node_backend_ids[] / leaf_backend_ids[]` 交接。

**backend 不实现内存分配**：CUDA backend 的 iface 里没有 `alloc`；分配在 buft 的 iface 里。这样 gallocr 可以脱离 backend 独立工作，甚至在没有 backend 实例时做纯规划（`ggml_gallocr_reserve_n_size`）。

### 1.2 所有权关系

```text
registry (static)
 ├── backends[]  ── CUDA reg ── devices[] ── CUDA0 device ── device_context
 │                                  │
 │                                  └── init_backend() ── ggml_backend{ iface, device, context }
 │                                                          └── cuda_context{ streams, cublas, pools }
 ├── bufts[]     ── CUDA0 buft (static, alignment 128)
 │
scheduler
 ├── backends[2] = { cuda_backend, cpu_backend }       // 不拥有，只引用
 ├── bufts[2]    = { CUDA0, CPU }                      // 不拥有
 ├── galloc ── buf_tallocs[2] ── chunks[] ── free_blocks
 │             buffers[2]      ── vbuffer ── chunks[] ── ggml_backend_buffer (owned)
 ├── graph / splits / ids / context_buffer (owned)
 └── ctx (owned, 元数据 arena)
```

关键点：**scheduler 拥有 buffer，但不拥有 backend**；`ggml_backend_sched_free()` 会释放显存，`ggml_backend_free()` 才释放 backend 的 stream/cuBLAS。所以 `main()` 里的顺序必须是 `sched_free` → `backend_free`。

---

## 2 一次运行的数据流

```text
host                                          device
────                                          ──────
model.buf (836,272 B)                         无
  ├── ggml_context
  ├── ggml_cgraph
  └── a / b / result 的元数据

compute()
  ├── split_graph
  │     a,b,result → backend_id = 0 (CUDA0)
  ├── alloc_splits
  │     cudaMalloc(384) ────────────────────► base
  │     a      @ base+0x000 (32 B)
  │     b      @ base+0x080 (24 B)
  │     result @ base+0x100 (48 B)
  ├── tensor_set(a) ─── cudaMemcpyAsync ────► base+0x000
  ├── tensor_set(b) ─── cudaMemcpyAsync ────► base+0x080
  ├── graph_compute
  │     MMVF kernel (grid 4, block 32) ─────► base+0x100
  ├── synchronize ───── cudaStreamSynchronize
  └── tensor_get(result) ─ cudaMemcpyAsync ◄─ base+0x100
```

---

## 3 三阶段各自证明了什么

| 阶段 | 断点结论 | 证明的设计 |
|---|---|---|
| `init_model` | `load_all` 空转；registry 在 `init_best` 首次构造；CUDA 先于 CPU 注册 | 后端是**可插拔 + 惰性注册**的；静态链接与动态加载是两条路径 |
| `init_model` | `cublas_handles[0][0] == NULL`、`pools[0][0] == NULL`，直到运行结束 | CUDA 资源**按需创建**，初始化 backend ≠ 初始化 cuBLAS |
| `init_model` | `context_buffer_size = 41,370,288`，sched 元数据合计约 41.4 MiB | scheduler **预留最坏情况**，避免 split 期间 malloc，保证延迟可预测 |
| `build_graph` | 三个 tensor 的 `data/buffer == NULL`，`flags = 0/0/16` | 构图是**纯 IR 描述**，与后端、内存完全解耦 |
| `build_graph` | `gf->n_nodes=1`、`gf->n_leafs=2`、节点名 `node_0/leaf_0/leaf_1` | 图是**后序 DFS + 哈希去重**的结果，`COMPUTE` 标志决定是否执行 |
| `compute` | `split[0].backend_id=0`、`n_inputs=0`、`n_splits=1` | 调度是**能力 + 优先级**决策，不是“GPU 优先”的硬编码 |
| `compute` | 只 `cudaMalloc(384)` 一次，三 tensor 偏移 0/128/256 | 分配是**按 buffer type 整块 + 对齐 + 生命周期复用**，不是 per-tensor |
| `compute` | `free_blocks = {0..256}, {384..}`，`a/b` 已被 gallocr 释放 | 复用基于**消费者计数**；未标 `INPUT` 的 leaf 在最后一个消费者后即视为可复用 |
| `compute` | `MUL_MAT` 命中 `MMVF`，`cublasSgemm` 从未命中 | kernel 选择是**多路径 + 启发式**，不是“F32 就走 cuBLAS” |
| `compute` | `grid=(4,1,1)`、`block=(32,1,1)`、`shared=128`，K=2 时 31 个线程空转 | MMVF 为**decode 形态（大 K、小 N）**设计；小 K 时 launch 开销主导 |

---

## 4 关键数值总表

| 项目 | 值 | 来源 / 推导 |
|---|---|---|
| `GGML_DEFAULT_GRAPH_SIZE` | 2048 | `include/ggml.h:233` |
| `sizeof(ggml_tensor)` | 336 B | gdb |
| `sizeof(ggml_object)` | 32 B | gdb |
| `ggml_tensor_overhead()` | 368 B | 32 + 336 |
| `sizeof(ggml_cgraph)` | 96 B | gdb |
| `ggml_graph_overhead()` | 82,608 B | 32 + pad16(96 + 2×2048×8 + 4099×4 + 4099×8 + 516) |
| `model.buf` | 836,272 B | 368×2048 + 82,608 |
| scheduler hash size | 2053 | 大于等于 2048 的下一个质数 |
| `context_buffer_size` | 41,370,288 B | 2048×30×2×336 + 82,608 |
| scheduler 元数据合计 | ≈ 41.4 MiB | 见 01 §8.6 |
| CUDA buft alignment | 128 B | `ggml-cuda.cu:903` |
| CPU buft alignment | 32 B | `ggml-cpu` |
| `a` 逻辑 / 分配 | 32 B / 128 B | `ne=[2,4]`，F32 |
| `b` 逻辑 / 分配 | 24 B / 128 B | `ne=[2,3]` |
| `result` 逻辑 / 分配 | 48 B / 128 B | `ne=[4,3]` |
| CUDA buffer 总大小 | 384 B | 3 × 128 |
| `split[0].backend_id` | 0（CUDA0） | pass 3 决策 |
| `MUL_MAT` 路径 | MMVF | `should_use_mmvf(F32, 890, ne11=3)` |
| kernel 模板 | `mul_mat_vec_f<float,float,3,32,false,false>` | `ncols_dst=3`，`block_size=32` |
| grid / block / shared | (4,1,1) / (32,1,1) / 128 B | `nrows=4`，`warp=32` |
| cuBLAS handle | 未创建 | MMVF 路径不需要 |
| CUDA pool | 未创建 | MMVF 无临时 buffer |
| CUDA stream | 1 个，`alloc_splits` 中创建 | 惰性 |

---

## 5 设计原则提炼

### 5.1 分层：每层只回答一个问题

```text
reg      ：这个库提供哪些设备？
device   ：这个设备能跑什么 op、用什么内存？
backend  ：这次执行用哪条 stream / 哪个 handle？
buft     ：这种内存怎么分配、对齐多少、量化行怎么补？
buffer   ：这块内存的基址、大小、usage 是什么？
sched    ：这个 node 归谁？
galloc   ：这个 tensor 放哪、什么时候能复用？
```

任何一层都可以被替换而不影响其他层。这是 ggml 能同时支持 CUDA / Metal / Vulkan / CPU / CANN 的根本原因。

### 5.2 惰性：能不做就不做

```text
registry       首次 get_reg() 才构造
CUDA 设备枚举  首次 ggml_cuda_info() 才执行
stream         首次 cuda_ctx->stream() 才创建
cuBLAS handle  首次 cublas_handle() 才创建 + workspace
pool           首次 pool() 才创建
event          n_copies > 1 才创建
tensor 数据    alloc_graph 才分配
```

好处：小图不付出大初始化成本；大模型才承担 cuBLAS workspace / pool 的开销；失败点后移，便于定位。

### 5.3 按 buffer type 分配，而不是按 tensor

gallocr 的分配单位是 **buffer type**，不是 tensor，也不是 backend：

```text
每个 buft 一个 ggml_dyn_tallocr
每个 tallocr 有若干 chunk（默认 1 个，上限 16）
每个 chunk 是一个 backend buffer，内部用 free list 管理 offset
```

这样同一 backend 的多个 tensor 共享一块显存，减少 `cudaMalloc` 次数；跨 backend 的 tensor 自动落到不同 buft 的 tallocr，不需要特殊判断。

### 5.4 生命周期复用，而不是引用计数

gallocr 不用引用计数，而是在构图时做一次**静态生命周期分析**：

```text
每个 tensor 记录 n_children（被多少 node 消费）和 n_views
node 执行完后，父 tensor 的 n_children--
归零且无 view → 立即 free，加入 free list
后续 tensor 按 best-fit 复用
```

这是“单次前向、无动态控制流”场景下最优的方案：O(V+E) 一次分析，运行时零开销。

### 5.5 多路径 kernel，按形态选择

`MUL_MAT` 不是只有一个实现，而是五条路径：

```text
MMVF   小 N（decode / 投机解码少量 token），向量化 + warp reduce
MMF    非量化、大 N，tensor core / MMA
MMVQ   量化权重、小 N
MMQ    量化权重、大 N
cuBLAS 兜底 + 特殊布局
```

选择由 `src0->type`、`cc`、`ne`、`nb`、`ne11` 共同决定。没有“最优实现”，只有“当前形态下最优”。

### 5.6 可观测性

```text
GGML_SCHED_DEBUG=1/2      打印 split 与 node 归属
GGML_ALLOCATOR_DEBUG      打印每次 alloc/free 与峰值
GGML_CUDA_DEBUG（编译期） 打印 fusion、kernel 选择
GGML_CUDA_DISABLE_FUSION  关闭 CUDA 图融合
GGML_CUDA_FORCE_CUBLAS    强制走 cuBLAS
GGML_CUDA_FORCE_MMQ       强制走 MMQ
GGML_CUDA_NO_PINNED       禁用页锁定内存
GGML_CUDA_ENABLE_UNIFIED_MEMORY  用 managed memory
```

### 5.7 从 1 个 MUL_MAT 映射到真实 LLM

| simple-backend | 真实 LLM | 同一套机制 |
|---|---|---|
| `a`、`b` | 权重矩阵、激活 | leaf tensor，由 gallocr 放置 |
| `result` | logits / 下一层激活 | node tensor，可能被后续节点消费 |
| 1 个 `MUL_MAT` | 每层 QKV / FFN 的矩阵乘 | 同一个 `ggml_cuda_mul_mat` 决策树 |
| `a`/`b` 常驻显存 | 权重常驻显存 | `ggml_backend_alloc_ctx_tensors` |
| `tensor_set` 每轮写输入 | token embedding / KV cache 更新 | `ggml_set_input` + `tensor_set` |
| 单 split | 整模型在 GPU | `n_splits=1` |
| 多 split | 部分层 offload 到 CPU | `n_inputs>0` + copy 节点 |
| free list 复用 | 激活内存复用 | `n_children` 生命周期分析 |
| MMVF（N=3） | decode（N=1） | 内存带宽受限，向量化 kernel |
| MMF/MMQ（N 大） | prefill（N=seq_len） | 计算受限，tensor core/量化 kernel |
| CUDA graph off | decode 循环开启 | 减少 launch 开销 |
| `n_copies=1` | 流水线并行 | `parallel=true` |
| `op_offload` | CPU 权重按 batch 提升到 GPU | `offload_op` + 阈值 |
| eval callback | 逐层调试/可视化 | `ggml_backend_sched_set_eval_callback` |

**为什么 ggml 敢用“先规划、后执行”**：

```text
LLM 单步推理的图结构是静态的（decode 每步拓扑相同，prefill 形状由 seq_len 决定）
没有动态控制流（无 if/while 依赖运行时数据）
→ 可以一次性完成 split、生命周期分析、显存规划
→ 运行期零分配、零决策，只有 kernel launch
```

这就是 scheduler + gallocr 的设计前提。如果图是动态的（例如带数据依赖分支的 agent 工作流），这套静态规划的收益会下降，需要更频繁的 re-split/re-alloc（`GGML_SCHED_DEBUG_REALLOC` 就是为此准备的观测手段）。

---

## 6 文档地图

| 文档 | 内容 | 适合谁 |
|---|---|---|
| [01-init_model.md](01-init_model.md) | registry/device/backend/buft/buffer/scheduler/gallocr 的初始化，含内存账本、惰性资源证据、**§11 源码逐行（register_backend / cuda_init / sched_new / gallocr_new_n）** | 想理解后端体系与调度器结构 |
| [02-build_graph.md](02-build_graph.md) | context arena、tensor 布局、图构建、生命周期边界，**§12 源码逐行（ggml_init / new_object / new_tensor_impl / visit_parents_graph / new_graph_custom）** | 想理解 ggml IR 与内存模型 |
| [03-compute.md](03-compute.md) | split 决策、gallocr 分配与复用、H2D、kernel 选择、MMVF 实现、同步与释放，**§14 源码逐行（五个 pass / alloc_graph_impl / tallocr_alloc / free_bytes / MMVF kernel）** | 想理解 GPU 执行全链路 |
| [04-gdb-trace.md](04-gdb-trace.md) | 原始断点输出 + 逐条数值解读 + 三组追加实验（§11–§13）+ **§15 断点与源码行号对照** | 想自己复现或改代码验证 |
| [05-extension-guide.md](05-extension-guide.md) | 新增算子/后端/buffer type、调度与 kernel 优化、profiling，**§10 五套 iface 逐字段解读** | 想做二次开发与性能优化 |
| [variants/](variants/) | 三组实验的可编译源码与构建说明 | 想直接复跑实验 |

---

## 7 与 CPU 版文档的关系

CPU 版（`analysis/L5/ggml/`）回答“host 上怎么建图、分配、执行”；本目录回答“同一份代码在 CUDA 上，哪些层被替换、哪些层不变”。

不变的部分：

```text
ggml_context / ggml_tensor / ggml_cgraph / 构图 API / gallocr 生命周期算法
```

被替换的部分：

```text
registry 注册内容、device/backend/buft/buffer 的实现、compute 入口、MUL_MAT 实现、同步方式
```

这正是分层设计的价值：**上层 IR 与调度逻辑完全复用，只有最底下三个接口被替换**。

---

## 8 三组实验验证（已完成）

三组实验的源码在 [`variants/`](variants/)，原始断点数据在 [04-gdb-trace.md](04-gdb-trace.md) §11–§13。

### 8.1 实验一：`ne11` 3 → 4，MMVF 切 cuBLAS

```text
改动：rows_B = 4
结果：ne11=4 时 should_use_mmvf 返回 false（要求 <= 3）
      should_use_mmf 返回 false（ne0=2 不满足 32 对齐）
      → 落到 cuBLAS

实测：
  cublasCreate_v2
  cublasSetWorkspace_v2 size = 4,194,304（4 MiB）
  cublasSgemm m=4 n=4 k=2, lda=2 ldb=2 ldc=4
  A/B/C = base+0x000/+0x080/+0x100
```

证明：kernel 选择是**多路径 + 形态阈值**，不是“F32 就走 cuBLAS”；cuBLAS 的 handle/workspace 是**首次使用才创建**。

### 8.2 实验二：`a` 在 CPU、`b` 在 CUDA，跨 backend copy

```text
改动：build_graph 后手动分配
      a → CPU buffer
      b → CUDA buffer

实测：
  SPLIT #0: CUDA0 # 1 inputs: [leaf_0]
  sched->graph.size = 4, n_nodes = 3
  nodes = [leaf_0 (view), CUDA0#leaf_0#0, node_0]
  node_backend_ids = [1, 0, 0]
  split[0].n_inputs = 1

执行：
  cpy_tensor_async(CPU→CUDA) 被拒绝（src 不是 CUDA buffer）
  → 降级 tensor_copy → cudaMemcpyAsync(H2D)
  copy tensor @ galloc buffer +0x000，result @ +0x080
```

证明：**数据位置与计算位置解耦**；copy 是图里的显式节点（含一个 view 依赖用于保活）；异步失败会自动降级同步路径。

### 8.3 实验三：三层 MLP 的复用与 inplace

```text
图：t0 = mul(w0,x) [128,3]
    t1 = mul(w1,t0) [64,3]
    t2 = mul(w2,t1) [128,3]

实测（free-list）：
  w2@0  w1@32768  w0@65536  x@98304  t0@99072
  free w0, x  → 合并 [65536, 33536]
  t1@65536（尾部块不参与 best-fit）
  free w1, t0 → [32768,32768] + [66304,huge]
  t2@32768（同样避开尾部块）

  峰值 = 100608 B
  朴素求和 = 102912 B

实测（inplace，exp3b）：
  t0@99072 → s0@99072   （UNARY 复用输入）
  t1@65536 → s1@65536
  free_extra_space(s0,t0) / free_extra_space(s1,t1) 命中
```

证明：gallocr 的复用是**静态生命周期分析 + best-fit + 尾部保留**；element-wise 激活可以 inplace 复用；真实 LLM 的显存峰值由“权重 + 少量存活激活”决定，而不是所有 tensor 之和。

### 8.4 三组实验的共同结论

```text
ggml 的执行链 = 静态规划 + 动态执行
  规划：split / buffer 归属 / offset / 生命周期，在 alloc 阶段一次完成
  执行：只做 copy、launch、sync，运行期不再决策
```

---

## 9 待后续验证点

- `GGML_BACKEND_DL=ON` + 正确命名/路径下的动态加载与重复注册。
- `GGML_CUDA_GRAPHS=ON` 下的 capture / instantiate / launch，以及第二次 compute 的行为。
- `ggml_set_input()` / `ggml_set_output()` 对调度与生命周期的影响（实验二已展示 input 不阻止释放）。
- 多 GPU（`GGML_CUDA_DEVICES`）下的 device context、虚拟设备与 tensor split。
- 量化权重（Q4_0/Q8_0）下 MMVQ/MMQ 的路径与 512 元素行补齐。
- `GGML_ALLOCATOR_DEBUG=ON` 下的逐次 alloc/free 日志（与实验三的 gdb 时序对拍）。
