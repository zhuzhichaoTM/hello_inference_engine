# 04 · gdb 断点原始数据与逐条解读

> 本文件保存本次调试的全部原始输出，并在每段后给出“这组数据说明了什么”。
>
> 复现命令：

```bash
cd /home/ggml
gdb -q -batch -x /tmp/trace.gdb ./build-debug/bin/simple-backend
```

---

## 0 通用脚本模板

```gdb
set pagination off
set confirm off
set breakpoint pending on
set print elements 0

break <function>
commands
silent
printf "\n=== [TAG] ... ===\n"
bt 4
continue
end

run
quit
```

要点：

| 指令 | 原因 |
|---|---|
| `set breakpoint pending on` | 共享库符号在 `run` 后才解析 |
| `silent` | 不打印 “Breakpoint N, ...” |
| `continue` | 自动跑完 |
| `bt 4` | 确认调用者，区分同名函数的多次调用 |
| `printf` + 强制转换 | 访问不透明结构体的字段 |

---

## 1 阶段一：init_model + build_graph

### 1.1 脚本

```gdb
break ggml_backend_load_all
break ggml_backend_init_best
break ggml_backend_init_by_type
break ggml_backend_cuda_init
break ggml_backend_sched_new
break ggml_init
break ggml_new_tensor_impl
break simple-backend.cpp:91
break simple-backend.cpp:93
break simple-backend.cpp:95
break ggml_free
```

### 1.2 原始输出

```text
Breakpoint 1 at 0x4420
Breakpoint 2 at 0x4380
Breakpoint 3 at 0x4410
Function "ggml_backend_cuda_init" not defined.
Breakpoint 4 (ggml_backend_cuda_init) pending.
Breakpoint 5 at 0x4550
Breakpoint 6 at 0x4400
Function "ggml_new_tensor_impl" not defined.
Breakpoint 7 (ggml_new_tensor_impl) pending.
Breakpoint 8 at 0x485a: file /home/ggml/examples/simple/simple-backend.cpp, line 91.
Breakpoint 9 at 0x486d: file /home/ggml/examples/simple/simple-backend.cpp, line 93.
Breakpoint 10 at 0x4879: file /home/ggml/examples/simple/simple-backend.cpp, line 95.
Breakpoint 11 at 0x4300
warning: Error disabling address space randomization: Operation not permitted
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/x86_64-linux-gnu/libthread_db.so.1"

=== [BP1] ggml_backend_load_all ===
#0  ggml_backend_load_all () at /home/ggml/src/ggml-backend-reg.cpp:575
#1  init_model (model=...) at /home/ggml/examples/simple/simple-backend.cpp:58
#2  main () at /home/ggml/examples/simple/simple-backend.cpp:118

=== [BP2] ggml_backend_init_best ===
#0  ggml_backend_init_best () at /home/ggml/src/ggml-backend-reg.cpp:383
#1  init_model (model=...) at /home/ggml/examples/simple/simple-backend.cpp:60
#2  main () at /home/ggml/examples/simple/simple-backend.cpp:118
[New Thread 0x7fce6e1ff000 (LWP 56706)]
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24091 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24091 MiB
register_backend: registered backend CUDA (1 devices)
register_device: registered device CUDA0 (NVIDIA GeForce RTX 4090)

=== [BP6] ggml_init mem_size=0 no_alloc=0 mem_buffer=(nil) ===
#0  ggml_init (params=...) at /home/ggml/src/ggml.c:1614
#1  ggml_cpu_init () at /home/ggml/src/ggml-cpu/ggml-cpu.c:3871
#2  ggml_backend_cpu_reg () at /home/ggml/src/ggml-cpu/ggml-cpu.cpp:705
#3  ggml_backend_registry::ggml_backend_registry (...) at /home/ggml/src/ggml-backend-reg.cpp:172

=== [BP11] ggml_free ctx=0x56407cb6eda0 ===
#0  ggml_free (ctx=0x56407cb6eda0) at /home/ggml/src/ggml.c:1664
#1  ggml_cpu_init () at /home/ggml/src/ggml-cpu/ggml-cpu.c:3872
#2  ggml_backend_cpu_reg () at /home/ggml/src/ggml-cpu/ggml-cpu.cpp:705
#3  ggml_backend_registry::ggml_backend_registry (...) at /home/ggml/src/ggml-backend-reg.cpp:172
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (Intel(R) Xeon(R) Platinum 8470Q)

=== [BP4] ggml_backend_cuda_init device=0 ===
#0  ggml_backend_cuda_init (device=0) at /home/ggml/src/ggml-cuda/ggml-cuda.cu:5843
#1  ggml_backend_cuda_device_init_backend (...) at /home/ggml/src/ggml-cuda/ggml-cuda.cu:5122
#2  ggml_backend_dev_init (...) at /home/ggml/src/ggml-backend.cpp:612
#3  ggml_backend_init_best () at /home/ggml/src/ggml-backend-reg.cpp:389
#4  init_model (...) at /home/ggml/examples/simple/simple-backend.cpp:60
#5  main () at /home/ggml/examples/simple/simple-backend.cpp:118

=== [BP3] ggml_backend_init_by_type type=0 ===
#0  ggml_backend_init_by_type (type=GGML_BACKEND_DEVICE_TYPE_CPU, params=0x0) at /home/ggml/src/ggml-backend-reg.cpp:375
#1  init_model (...) at /home/ggml/examples/simple/simple-backend.cpp:61
#2  main () at /home/ggml/examples/simple/simple-backend.cpp:118

=== [BP5] ggml_backend_sched_new n_backends=2 graph_size=2048 parallel=0 op_offload=1 ===
backends[0]=0x5579e80e8030 backends[1]=0x5579e80f0730
$1 = (ggml_backend_dev_t) 0x5579e80ea560
$2 = (ggml_backend_dev_t) 0x7f0b5f4e69c0 <ggml_backend_cpu_reg_get_device(...)::ggml_backend_cpu_device>
#0  ggml_backend_sched_new (...) at /home/ggml/src/ggml-backend.cpp:1854
#1  init_model (...) at /home/ggml/examples/simple/simple-backend.cpp:64
#2  main () at /home/ggml/examples/simple/simple-backend.cpp:118

=== [BP6] ggml_init (build_graph) mem_size=836272 no_alloc=1 mem_buffer=0x7f0b04f46010 ===
#0  ggml_init (params=...) at /home/ggml/src/ggml.c:1614
#1  build_graph (model=...) at /home/ggml/examples/simple/simple-backend.cpp:79
#2  main () at /home/ggml/examples/simple/simple-backend.cpp:120

=== [BP7] ggml_new_tensor_impl type=0 n_dims=2 ne=[2,4,...] view_src=(nil) ===
#0  ggml_new_tensor_impl (...) at /home/ggml/src/ggml.c:1772
#1  ggml_new_tensor (...) at /home/ggml/src/ggml.c:1849
#2  ggml_new_tensor_2d (..., ne0=2, ne1=4) at /home/ggml/src/ggml.c:1865
#3  build_graph (...) at /home/ggml/examples/simple/simple-backend.cpp:84

=== [BP7] ggml_new_tensor_impl type=0 n_dims=2 ne=[2,3,...] view_src=(nil) ===
#3  build_graph (...) at /home/ggml/examples/simple/simple-backend.cpp:85

=== [BP7] ggml_new_tensor_impl type=0 n_dims=4 ne=[4,3,1,1] view_src=(nil) ===
#2  ggml_mul_mat (..., a=0x7f0b04f5a2e0, b=0x7f0b04f5a450) at /home/ggml/src/ggml.c:3349
#3  build_graph (...) at /home/ggml/examples/simple/simple-backend.cpp:88

=== [BP8] after ggml_mul_mat (before build_forward_expand) ===
$3 = '\000' <repeats 63 times>          # result->name 为空
$4 = GGML_OP_MUL_MAT
$5 = {4, 3, 1, 1}                       # result->ne
$6 = {4, 16, 48, 48}                    # result->nb
$7 = (void *) 0x0                       # result->data
$8 = (ggml_backend_buffer *) 0x0        # result->buffer
$9 = (ggml_tensor *) 0x0                # result->view_src
$10 = '\000' <repeats 63 times>         # src[0]->name 为空
$11 = '\000' <repeats 63 times>         # src[1]->name 为空
$12 = 48                                # ggml_nbytes(result)
$13 = 0                                 # gf->n_nodes
$14 = 0                                 # gf->n_leafs

=== [BP9] after ggml_build_forward_expand (before ggml_free) ===
$15 = 1                                 # gf->n_nodes
$16 = 2                                 # gf->n_leafs
$17 = "node_0"
$18 = GGML_OP_MUL_MAT
$19 = (void *) 0x0
$20 = (ggml_backend_buffer *) 0x0
$21 = "leaf_0"
$22 = "leaf_1"
$23 = {4, 8, 32, 32}                    # model.a->nb
$24 = {4, 8, 24, 24}                    # model.b->nb

=== [BP11] ggml_free ctx=0x5579e8025e90 ===
#0  ggml_free (...) at /home/ggml/src/ggml.c:1664
#1  build_graph (...) at /home/ggml/examples/simple/simple-backend.cpp:93
#2  main () at /home/ggml/examples/simple/simple-backend.cpp:120

=== [BP10] after ggml_free(ctx) ===
$25 = 1                                 # gf->n_nodes
$26 = "node_0"
$27 = (void *) 0x0                      # gf->nodes[0]->data
$28 = (void *) 0x0                      # model.a->data
$29 = (ggml_backend_buffer *) 0x0       # model.a->buffer
$30 = 836272                            # model.buf.size()
```

### 1.3 解读

| 观察 | 说明 |
|---|---|
| BP1 前后无注册日志 | `load_all` 没有找到任何 `libggml-<name>-*.so` |
| BP2 后出现 CUDA/CPU 注册 | registry 在 `ggml_backend_dev_by_type` 中首次构造 |
| CUDA 先、CPU 后 | registry 构造函数按源码顺序注册 |
| BP4 栈 | `init_best → dev_init → cuda_device_init_backend → cuda_init` |
| BP6 两次 | 第一次是 CPU backend 初始化（`mem_size=0`），第二次是 `build_graph`（`836272`） |
| BP7 三次 | a、b、result；result 由 `ggml_mul_mat` 创建，`n_dims=4` |
| BP8 | `result` 的 `data/buffer` 为 NULL；`gf` 仍为空 |
| BP9 | expand 后 `n_nodes=1`、`n_leafs=2`，自动命名 |
| BP10 | `ggml_free` 后元数据仍可读；`model.buf` 836,272 |

---

## 2 阶段二：调度与分配

### 2.1 脚本

```gdb
break ggml_backend_sched_alloc_graph
break ggml_backend_sched_split_graph
break ggml_backend_sched_alloc_splits
break ggml_gallocr_alloc_graph
break ggml_gallocr_reserve_n
break ggml_backend_buft_alloc_buffer
break ggml_backend_cuda_buffer_type_alloc_buffer
break ggml-cuda.cu:898
break ggml_backend_buffer_init
break ggml_backend_tensor_set
break ggml_backend_cuda_buffer_set_tensor
break ggml_backend_sched_graph_compute
break ggml_backend_sched_compute_splits
```

### 2.2 原始输出

```text
=== [A1] sched_alloc_graph: graph nodes=1 leafs=2 | sched n_splits(before)=0
     is_reset=1 is_alloc=0 n_copies=1 cur_copy=0 ===
#0  ggml_backend_sched_alloc_graph (...) at /home/ggml/src/ggml-backend.cpp:1993
#1  compute (...) at /home/ggml/examples/simple/simple-backend.cpp:101
#2  main () at /home/ggml/examples/simple/simple-backend.cpp:123

=== [A2] sched_split_graph: graph nodes=1 leafs=2 ===
node[0]=node_0 op=MUL_MAT leaf[0]=leaf_0 leaf[1]=leaf_1
#0  ggml_backend_sched_split_graph (...) at /home/ggml/src/ggml-backend.cpp:1066
#1  ggml_backend_sched_alloc_graph (...) at /home/ggml/src/ggml-backend.cpp:2000
#2  compute (...) at /home/ggml/examples/simple/simple-backend.cpp:101

=== [A3] sched_alloc_splits: n_splits=1 sched_graph nodes=1 leafs=2 ===
split[0]: backend_id=0 i_start=0 i_end=1 n_inputs=0 graph.nodes=1
node_backend_ids[0]=0 leaf_backend_ids[0]=0 leaf_backend_ids[1]=0
bufts[0]=0x7f56767500c0 bufts[1]=0x7f56770b81a0
galloc->n_buffers=2 galloc->buffers[0]=(nil) buffers[1]=(nil)

=== [A4] gallocr_alloc_graph: graph nodes=1 leafs=2
     galloc n_nodes=0 n_leafs=0 n_buffers=2 ===
ggml_gallocr_needs_realloc: graph has different number of nodes
ggml_gallocr_alloc_graph: cannot reallocate multi buffer graph automatically, call reserve
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
[New Thread 0x7f560f88a000 (LWP 58062)]
[New Thread 0x7f560f089000 (LWP 58063)]

=== [A5] gallocr_reserve_n: graph nodes=1 leafs=2 ===
#0  ggml_gallocr_reserve_n (...) at /home/ggml/src/ggml-alloc.c:963
#1  ggml_backend_sched_alloc_splits (...) at /home/ggml/src/ggml-backend.cpp:1633
#2  ggml_backend_sched_alloc_graph (...) at /home/ggml/src/ggml-backend.cpp:2002
#3  compute (...) at /home/ggml/examples/simple/simple-backend.cpp:101

=== [A6] backend_buft_alloc_buffer: buft=0x7f56767500c0 size=384 ===
#0  ggml_backend_buft_alloc_buffer (...) at /home/ggml/src/ggml-backend.cpp:40
#1  ggml_vbuffer_alloc (...) at /home/ggml/src/ggml-alloc.c:432
#2  ggml_gallocr_reserve_n_impl (...) at /home/ggml/src/ggml-alloc.c:939
#3  ggml_gallocr_reserve_n (...) at /home/ggml/src/ggml-alloc.c:963

=== [A7] cuda_buffer_type_alloc_buffer: size=384 device=0 ===
#0  ggml_backend_cuda_buffer_type_alloc_buffer (...) at /home/ggml/src/ggml-cuda/ggml-cuda.cu:884
#1  ggml_backend_buft_alloc_buffer (...) at /home/ggml/src/ggml-backend.cpp:45
#2  ggml_vbuffer_alloc (...) at /home/ggml/src/ggml-alloc.c:432
#3  ggml_gallocr_reserve_n_impl (...) at /home/ggml/src/ggml-alloc.c:939

=== [A8] cuda alloc done: dev_ptr=0x7f5600c00000 size=384 ===

=== [A9] backend_buffer_init: buft=0x7f56767500c0 size=384 context=0x556de4398b50 ===
#0  ggml_backend_buffer_init (...) at /home/ggml/src/ggml-backend.cpp:107
#1  ggml_backend_cuda_buffer_type_alloc_buffer (...) at /home/ggml/src/ggml-cuda/ggml-cuda.cu:900
#2  ggml_backend_buft_alloc_buffer (...) at /home/ggml/src/ggml-backend.cpp:45

=== [A4] gallocr_alloc_graph: graph nodes=1 leafs=2
     galloc n_nodes=1 n_leafs=2 n_buffers=2 ===

=== [B1] ggml_backend_tensor_set: tensor=leaf_0 ne=[2,4] nb=[4,8] offset=0 size=32 ===
tensor->data=0x7f5600c00000 tensor->buffer=0x556de43981d0 buffer->size=384
#0  ggml_backend_tensor_set (...) at /home/ggml/src/ggml-backend.cpp:336
#1  compute (...) at /home/ggml/examples/simple/simple-backend.cpp:104

=== [B2] cuda_buffer_set_tensor: tensor=leaf_0 offset=0 size=32
     dst=0x7f5600c00000 src=0x556de3167020 device=0

=== [B1] ggml_backend_tensor_set: tensor=leaf_1 ne=[2,3] nb=[4,8] offset=0 size=24 ===
tensor->data=0x7f5600c00080 tensor->buffer=0x556de43981d0 buffer->size=384
#1  compute (...) at /home/ggml/examples/simple/simple-backend.cpp:105

=== [B2] cuda_buffer_set_tensor: tensor=leaf_1 offset=0 size=24
     dst=0x7f5600c00080 src=0x556de3167040 device=0

=== [C1] sched_graph_compute: n_splits=1 is_alloc=1 cur_copy=0 ===
#0  ggml_backend_sched_graph_compute (...) at /home/ggml/src/ggml-backend.cpp:2012
#1  compute (...) at /home/ggml/examples/simple/simple-backend.cpp:108

=== [C2] sched_compute_splits: n_splits=1 ===
split[0]: backend_id=0 n_nodes=1 n_inputs=0 graph.nodes=1 graph.leafs=0
```

### 2.3 解读

| 数据 | 含义 |
|---|---|
| `n_splits=1` | 整图只有一个 backend 段 |
| `backend_id=0` | CUDA0 |
| `i_start=0, i_end=1` | 覆盖 `nodes[0]` |
| `n_inputs=0` | 无跨 backend 拷贝 |
| `galloc->buffers[0]=(nil)` | 尚未分配 |
| `galloc n_nodes=0` | 首次分配，与图不一致 |
| `size=384` | 三个 tensor 对齐后总和 |
| `dev_ptr=0x7f5600c00000` | 显存基址 |
| `buffer->size=384` | 整块 buffer 大小 |
| `tensor->data=base+0x00/0x80` | a/b 的落位 |
| `graph.leafs=0` | split view 只含节点 |

---

## 3 gallocr 内部状态（关键）

### 3.1 脚本

```gdb
break simple-backend.cpp:104
commands
silent
printf "sched->graph.size=%d n_nodes=%d n_leafs=%d\n", model.sched->graph.size, ...
printf "talloc[0]: alignment=%zu max_chunk_size=%zu n_chunks=%d chunk0.max_size=%zu n_free_blocks=%d\n", ...
printf "free[0]: offset=%zu size=%zu\n", ...
printf "free[1]: offset=%zu size=%zu\n", ...
printf "leaf_alloc[0]: buffer_id=%d chunk=%d offset=%zu size_max=%zu\n", ...
printf "node_alloc[0].dst: ...\n"
printf "alloc_size: a=%zu b=%zu result=%zu\n", ...
printf "tensor flags: a=%d b=%d result=%d\n", ...
continue
end
```

### 3.2 原始输出

```text
=== [K1] after alloc_graph: scheduler/galloc internals ===
model.sched->graph.size=2 n_nodes=1 n_leafs=2
split[0].graph.size=0 n_nodes=1 n_leafs=0
galloc->n_buffers=2 n_nodes=1 n_leafs=2 hash_set.size=3
talloc[0]: alignment=128 max_chunk_size=9223372036854775807 n_chunks=1
           chunk0.max_size=384 chunk0.n_free_blocks=2
talloc[1]: alignment=32 max_chunk_size=9223372036854775807 n_chunks=0
vbuffer[0]: chunk0=0x55d202423a90 size=384 | vbuffer[1]: chunk0=(nil)
leaf_alloc[0]: buffer_id=0 chunk=0 offset=0   size_max=32
leaf_alloc[1]: buffer_id=0 chunk=0 offset=128 size_max=24
node_alloc[0].dst:    buffer_id=0 chunk=0 offset=256 size_max=48
node_alloc[0].src[0]: buffer_id=0 chunk=0 offset=0   size_max=32
node_alloc[0].src[1]: buffer_id=0 chunk=0 offset=128 size_max=24
alloc_size: a=32 b=24 result=48
alignment: cuda=128 cpu=32

=== [K3] free blocks and tensor flags ===
chunk0 max_size=384 n_free_blocks=2
free[0]: offset=0 size=256
free[1]: offset=384 size=9223372036854775423
max_chunk_size=9223372036854775807
tensor flags: a=0 b=0 result=16
```

### 3.3 解读

**`sched->graph.size = 2`**

```text
max(n_nodes=1, n_leafs=2) + total_inputs(0) + n_dep_nodes(0) = 2
```

`nodes[]` 和 `leafs[]` 是独立数组，各需 2 个槽位。

**`split[0].graph.size = 0`**

`ggml_graph_view` 不拥有数组，只持有窗口指针，`size` 固定为 0。

**`galloc->hash_set.size = 3`**

```text
min_hash_size = (1 + 2) * 1.25 = 3
ggml_hash_size(3) = 3
```

**`talloc[0].max_chunk_size = 2^63-1`**

```cpp
MIN(SIZE_MAX, SIZE_MAX/2)   // 防止 offset 溢出
```

**`chunk0.max_size = 384`**

```text
128 (a) + 128 (b) + 128 (result) = 384
```

**`n_free_blocks = 2`，`free[0] = {0, 256}`，`free[1] = {384, huge}`**

推导：

```text
alloc 后：{384, huge}
free a({0,128}) → {0,128}, {384,huge}
free b({128,128}) → 合并 → {0,256}, {384,huge}
```

**为什么 a/b 被 free**：

```text
flags: a=0, b=0（没有 INPUT）
node_0 是唯一消费者
node_0 分配后 n_children 归零
→ free_node
```

**为什么安全**：

```text
1. free 只更新 free list，不改 tensor->data
2. 之后没有新的分配
3. tensor_set 写的是原地址
```

**`result flags=16`**：

```text
GGML_TENSOR_FLAG_COMPUTE = 16
```

注意不是 4（4 是 PARAM），不是 1（INPUT），不是 2（OUTPUT）。

**`talloc[1].n_chunks = 0`**：

```text
没有 tensor 判给 CPU
→ CPU vbuffer 不申请任何内存
```

---

## 4 阶段三：CUDA compute

### 4.1 脚本

```gdb
break ggml_backend_graph_compute_async
break ggml_backend_cuda_graph_compute
break ggml_cuda_graph_evaluate_and_capture
break ggml_cuda_compute_forward
break ggml_cuda_mul_mat
break ggml_cuda_mul_mat_vec_f
break ggml_cuda_mul_mat_f
break ggml_cuda_mul_mat_vec_q
break ggml_cuda_mul_mat_q
break ggml_cuda_mul_mat_cublas
break cublasSgemm
break ggml-cuda/mmvf.cu:455
break ggml_backend_cuda_synchronize
break ggml_backend_tensor_get
break ggml_backend_cuda_buffer_get_tensor
break ggml_backend_sched_free
break ggml_backend_cuda_buffer_free_buffer
break ggml_backend_cuda_free
```

### 4.2 原始输出

```text
=== [C3] backend_graph_compute_async: backend=0x56469a662030 device=0x56469a664560
     cgraph nodes=1 leafs=0 ===
node[0]=node_0 op=MUL_MAT data=0x7f8704c00100 buffer=0x56469af401d0
#0  ggml_backend_graph_compute_async (...) at /home/ggml/src/ggml-backend.cpp:462
#1  ggml_backend_sched_compute_splits (...) at /home/ggml/src/ggml-backend.cpp:1799
#2  ggml_backend_sched_graph_compute_async (...) at /home/ggml/src/ggml-backend.cpp:2029
#3  ggml_backend_sched_graph_compute (...) at /home/ggml/src/ggml-backend.cpp:2012

=== [C4] backend_cuda_graph_compute: cgraph nodes=1 leafs=0 device=0 ===
node[0]=node_0 op=MUL_MAT ne=[4,3] nb=[4,16] data=0x7f8704c00100
src0=leaf_0 src0.data=0x7f8704c00000 src1=leaf_1 src1.data=0x7f8704c00080
#0  ggml_backend_cuda_graph_compute (...) at /home/ggml/src/ggml-cuda/ggml-cuda.cu:4425
#1  ggml_backend_graph_compute_async (...) at /home/ggml/src/ggml-backend.cpp:463
#2  ggml_backend_sched_compute_splits (...) at /home/ggml/src/ggml-backend.cpp:1799

=== [C5] cuda_graph_evaluate_and_capture: use_cuda_graph=0 update_required=0
     nodes=1 device=0 ===

=== [D1] cuda_compute_forward: node=node_0 op=MUL_MAT device=0 ===
dst ne=[4,3,1,1] nb=[4,16,48,48] data=0x7f8704c00100 type=f32
src0 type=f32 ne=[2,4] nb=[4,8] data=0x7f8704c00000
src1 type=f32 ne=[2,3] nb=[4,8] data=0x7f8704c00080

=== [D2] cuda_mul_mat dispatch: device=0 src0.type=f32 src1.type=f32 dst.type=f32 ===
src0 ne=[2,4,1,1] nb=[4,8]
src1 ne=[2,3,1,1] nb=[4,8]
dst  ne=[4,3,1,1] nb=[4,16]
src0 cont=1 cont2=1 | src1 cont=1 cont2=1 | dst cont=1

=== [D3a] HIT ggml_cuda_mul_mat_vec_f (mmvf.cu) ===
#0  ggml_cuda_mul_mat_vec_f (...) at /home/ggml/src/ggml-cuda/mmvf.cu:635
#1  ggml_cuda_mul_mat (...) at /home/ggml/src/ggml-cuda/ggml-cuda.cu:1848
#2  ggml_cuda_compute_forward (...) at /home/ggml/src/ggml-cuda/ggml-cuda.cu:2261
```

未命中：

```text
ggml_cuda_mul_mat_f
ggml_cuda_mul_mat_vec_q
ggml_cuda_mul_mat_q
ggml_cuda_mul_mat_cublas
cublasSgemm
```

### 4.3 解读

| 数据 | 含义 |
|---|---|
| `cgraph.leafs=0` | split view 只含节点 |
| `node->data=base+0x100` | result 的显存地址 |
| `src0.data=base+0x000` | a |
| `src1.data=base+0x080` | b |
| `use_cuda_graph=0` | 编译时 `GGML_CUDA_GRAPHS=OFF` |
| `cont=1` | 三个 tensor 都连续 |
| 只命中 MMVF | 决策树在 MMVF 处返回 |

---

## 5 MMVF launch 与资源状态

### 5.1 脚本

```gdb
break ggml-cuda/mmvf.cu:455
commands
silent
printf "ncols=%lld nrows=%lld ncols_dst=%lld block_size_best=%lld\n", ...
printf "block_nums=(%u,%u,%u) block_dims=(%u,%u,%u) nbytes_shared=%d\n", ...
printf "warp_size=%d has_fusion=%d\n", ...
bt 4
continue
end

break ggml-cuda/mmvf.cu:635
commands
silent
printf "device=%d stream=%p cublas=%p pool=%p\n", ...
continue
end
```

### 5.2 原始输出

```text
=== [H1] mmvf launch params ===
ncols(ne00)=2 nrows(ne01)=4 ncols_dst=3 block_size_best=32
block_nums=(4,1,1) block_dims=(32,1,1) nbytes_shared=128
warp_size=32 has_fusion=0
#0  launch_mul_mat_vec_f_cuda<float, float, 3, false> (...) at mmvf.cu:455
#1  mul_mat_vec_f_cuda_switch_ncols_dst<float, float> (...) at mmvf.cu:562
#2  mul_mat_vec_f_cuda<float> (...) at mmvf.cu:628
#3  ggml_cuda_mul_mat_vec_f (...) at mmvf.cu:710

=== [J1] at mul_mat_vec_f: device=0
stream[dev][0]=0x5624e018e8c0
cublas_handle[dev][0]=(nil)
pool[dev][0]=(nil)
copy_event=(nil) curr_stream_no=0
```

### 5.3 解读

**`block_size_best=32`**

```text
ncols=2
初始 niter_best = ceil(2/64) = 1
更大的 block 无法把 niter 降到 0
→ 保持 32
```

**`grid=(4,1,1)`**

```text
blockIdx.x = nrows = 4
blockIdx.y = nchannels_dst = 1
blockIdx.z = nsamples = 1
```

**`shared=128`**

```text
warp_size * sizeof(float) = 32 * 4 = 128
has_fusion=false → 不加第二块
```

**`cublas_handle=(nil)`、`pool=(nil)`**

```text
MMVF 不需要 cuBLAS handle，也不需要临时 pool
→ 惰性资源从未创建
```

**`copy_event=(nil)`**

```text
n_copies=1，不需要 event
```

---

## 6 `supports_op` 的实际调用

### 6.1 原始输出

```text
=== [K2] cuda device supports_op: op=MUL_MAT ===
op->src[0]=leaf_0 type=f32 nb0=4 elem=4 | src[1]=leaf_1 type=f32 nb0=4 elem=4
#0  ggml_backend_cuda_device_supports_op (...) at ggml-cuda.cu:5136
#1  ggml_backend_dev_supports_op (...) at ggml-backend.cpp:636
#2  ggml_backend_supports_op (...) at ggml-backend.cpp:468
```

### 6.2 解读

```text
nb0 == element_size → 连续
type F32 → 在支持列表
→ 返回 true
```

这次调用发生在 `split_graph` pass 3：

```text
for b in {CUDA, CPU}:
    ggml_backend_supports_op(backends[b], node_0)
```

CUDA 返回 true，CPU 也返回 true；最终 CUDA 因下标小胜出。

---

## 7 `GGML_SCHED_DEBUG=2` 输出

```bash
GGML_SCHED_DEBUG=2 ./build-debug/bin/simple-backend
```

```text
## SPLIT #0: CUDA0 # 0 inputs
node #  0 (   MUL_MAT):               node_0 (   0K) [CUDA0         ] use=0,c=1:
                        leaf_0 (   0K) [CUDA0         ]
                        leaf_1 (   0K) [CUDA0         ]
```

解读：

| 字段 | 值 | 含义 |
|---|---|---|
| `SPLIT #0` | 1 个 split | 整图在 CUDA0 |
| `# 0 inputs` | 无跨 backend 输入 | `n_inputs=0` |
| `[CUDA0]` | 每个 tensor 的归属 | pass 3/4 结果 |
| `use=0` | result 没有被后续节点使用 | 图输出 |
| `c=1` | COMPUTE 标志 | 会被执行 |
| `0K` | 大小四舍五入 | 104 B → 0K |

注意：`SET_CAUSE` 在当前源码被 `#if 0` 编译掉，所以方括号里第二列（cause）为空。源码位置 `ggml-backend.cpp:910`。

---

## 8 关键数值汇总

| 项目 | 值 |
|---|---|
| `sizeof(ggml_tensor)` | 336 B |
| `sizeof(ggml_object)` | 32 B |
| `ggml_tensor_overhead()` | 368 B |
| `sizeof(ggml_cgraph)` | 96 B |
| `ggml_graph_overhead()` | 82,608 B |
| `model.buf` | 836,272 B |
| scheduler hash size | 2,053 |
| `context_buffer_size` | 41,370,288 B |
| `sizeof(ggml_backend_sched)` | 1,064 B |
| `sizeof(ggml_backend_sched_split)` | 128 B |
| `sizeof(ggml_backend_cuda_context)` | 4,336 B |
| CUDA alignment | 128 B |
| CPU alignment | 32 B |
| a/b/result 逻辑 | 32/24/48 B |
| CUDA buffer | 384 B |
| offsets | 0/128/256 |
| free blocks | `{0,256}`, `{384,∞}` |
| n_splits | 1 |
| backend_id | 0 |
| n_inputs | 0 |
| path | MMVF |
| grid/block/shared | (4,1,1)/(32,1,1)/128 |
| cuBLAS handle | NULL |
| pool | NULL |
| CUDA graph | off |

---

## 9 复现注意事项

1. **gdb 会改变时序**：断点暂停进程，异步操作可能被同步化；耗时不可信。
2. **地址每次不同**：不要用固定地址判断。
3. **`@plt` 命中**：第一次可能是 PLT stub，用 `bt` 确认真实实现。
4. **`ggml_free` 多次命中**：CPU backend 初始化、scheduler context、build_graph 都会调用；用 `bt` 区分。
5. **`ggml_init` 多次命中**：CPU backend 的 `mem_size=0`，`build_graph` 的是 836,272。
6. **`GGML_CUDA_DEBUG` 是编译期宏**：环境变量无效，需要 `-DGGML_CUDA_DEBUG` 重编译。
7. **`SET_CAUSE` 被编译掉**：`GGML_SCHED_DEBUG` 只能看到归属，看不到原因字符串。
8. **Debug 日志**：`ggml_gallocr_needs_realloc` 只在非 `NDEBUG` 构建可见。

---

## 10 追加实验命令

> 其中前三项已在 §11–§13 完成并记录结果：
>
> - §11 `rows_B = 4` → MMVF 切 cuBLAS
> - §12 `a` 在 CPU、`b` 在 CUDA → 跨 backend copy
> - §13 三层 MLP → free-list 复用与 inplace
>
> 下面的命令用于继续扩展验证。

```bash
# 1. 强制 cuBLAS
GGML_CUDA_FORCE_CUBLAS=1 ./build-debug/bin/simple-backend

# 2. 强制 MMQ（量化类型才有意义）
GGML_CUDA_FORCE_MMQ=1 ./build-debug/bin/simple-backend

# 3. 关闭 fusion
GGML_CUDA_DISABLE_FUSION=1 ./build-debug/bin/simple-backend

# 4. 禁用 pinned memory（本例未用，但可观察 host buffer 回退）
GGML_CUDA_NO_PINNED=1 ./build-debug/bin/simple-backend

# 5. managed memory
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build-debug/bin/simple-backend

# 6. 调度 debug
GGML_SCHED_DEBUG=2 ./build-debug/bin/simple-backend

# 7. 分配器 debug（需要 -DGGML_ALLOCATOR_DEBUG 重编译）
cmake -DGGML_ALLOCATOR_DEBUG=ON ...
```

对应的 gdb 断点：

```gdb
break cublasCreate
break cublasSetWorkspace
break ggml_cuda_pool_vmm::ggml_cuda_pool_vmm
break ggml_backend_cuda_host_buffer_type_alloc_buffer
break ggml_cuda_device_malloc
break ggml_cuda_try_fuse
```

---

## 11 追加实验一：`rows_B = 4`，MMVF → cuBLAS 的路径切换

### 11.1 实验目的

验证 `ggml_cuda_mul_mat` 决策树中 `ne11` 阈值的作用：`ne11 = 3` 走 MMVF，`ne11 = 4` 应当离开 MMVF 并落到 cuBLAS，同时观察 cuBLAS 的初始化成本。

### 11.2 改动

只改一个常量：

```cpp
const int rows_B = 4, cols_B = 2;      // 原为 3
float matrix_B[rows_B * cols_B] = { 10, 5, 9, 9, 5, 4, 1, 2 };
```

对应 `result->ne = [4, 4]`、`ne11 = 4`。源码见 `variants/exp1_rowsB4.cpp`。

### 11.3 运行结果

```text
result ne=[4,4]
    60.00    55.00    50.00   110.00
    90.00    54.00    54.00   126.00
    42.00    29.00    28.00    64.00
    18.00     7.00     8.00    20.00
```

第 4 列由新增的 `b[3] = [1,2]` 计算：

```text
row0: 2*1 + 8*2 = 18
row1: 5*1 + 1*2 = 7
row2: 4*1 + 2*2 = 8
row3: 8*1 + 6*2 = 20
```

### 11.4 路径断点

```text
=== [E1] ggml_cuda_mul_mat: src0.type=f32 ne=[2,4] src1.ne=[2,4] ne11=4 cc=890 ===

=== [E1i] HIT cublas_impl<F32>: ne12=1 ne13=1 ===

=== [E1f] cublasCreate_v2 ===

=== [E1g] cublasSetWorkspace_v2: workspace=0x7f2d95800000 size=4194304 ===

=== [E1j] cublasSgemm_v2 -> caller frame ===
m=ne01=4 n=ne11=4 k=ne10=2 lda=s01=2 ldb=s11=2 ldc=ne0=4
A=0x7f59a8c00000 B=0x7f59a8c00080 C=0x7f59a8c00100
alpha=0x7f5a1c056fa4 beta=0x7f5a1c056fa8
cublas_h=0x55a8080db850 stream=0x55a807de2700
```

未命中：

```text
ggml_cuda_mul_mat_vec_f
ggml_cuda_mul_mat_f
ggml_cuda_mul_mat_vec_q
ggml_cuda_mul_mat_q
```

### 11.5 逐条解读

| 观察 | 含义 |
|---|---|
| `ne11=4` | 输出列数从 3 变 4 |
| 未命中 MMVF | `should_use_mmvf` 对 F32/Ampere 要求 `ne11 <= 3` |
| 未命中 MMF | `should_use_mmf` 要求 `src0_ne[0] % (warp_size * 4/ts) == 0`，即 `2 % 32 != 0` |
| 未命中 MMVQ/MMQ | 只支持量化 `src0` |
| 命中 cuBLAS wrapper | 决策树兜底 |
| `cublasCreate_v2` | 第一次创建 handle（注意是 `_v2` 版本化符号） |
| `cublasSetWorkspace_v2` size=4,194,304 | cc=890 < Hopper，workspace 为 4 MiB（Hopper 为 32 MiB） |
| `cublasSgemm_v2` m=4 n=4 k=2 | `m=ne01`、`n=ne11`、`k=ne10` |
| `lda=2 ldb=2 ldc=4` | `s01=nb01/4`、`s11=nb11/4`、`ne0` |
| A/B/C = base+0/0x80/0x100 | 与实验零相同的 128 B 对齐布局 |
| `alpha/beta` 指向常量 | 1.0 / 0.0，来自 traits |

### 11.6 为什么 `op(A)=T`

cuBLAS 是列主序，GGML 是行主序。GGML 的 `a` 是 `[K, M]`（行主序），按列主序解释等价于 `a^T [M, K]`。所以调用时：

```text
CUBLAS_OP_T 作用于 A：把内存中的 [K,M] 当作 [M,K]^T 使用
CUBLAS_OP_N 作用于 B：b 是 [K,N]，列主序下就是 [K,N]
m = M, n = N, k = K
```

这解释了为什么代码里 `m=ne01, n=ne11, k=ne10`，而不是直觉上的 `M×N×K` 顺序。

### 11.7 设计解读：cuBLAS 的代价与收益

代价：

```text
1. cublasCreate + cublasSetMathMode + cublasSetStream
2. cudaMalloc 4 MiB workspace（Hopper 32 MiB）
3. 每次调用的 kernel 选择开销
```

收益：

```text
1. 对大矩阵，cuBLAS 的 tensor core kernel 通常比 ggml 自研路径更快
2. 支持任意布局/转置/批量，是正确性兜底
3. 维护成本低
```

因此 ggml 的策略是：

```text
能用专用 kernel（MMVF/MMF/MMVQ/MMQ）就用专用 kernel
否则退回 cuBLAS
```

`ne11=4` 这个临界点说明专用 kernel 的收益区间很窄——decode 的 `ne11=1`、投机解码的 2~3 个 token 才在 MMVF 的甜点区。

### 11.8 验证了文档中的哪条结论

`03-compute.md` §8.4 的路径切换表：

| 变化 | 预测 | 实测 |
|---|---|---|
| `ne11 = 4` | cuBLAS | ✓ |
| MMVF 不再命中 | ✓ | ✓ |
| cuBLAS handle / 4 MiB workspace 创建 | ✓ | ✓ |

---

## 12 追加实验二：跨 backend 输入（`a` 在 CPU，`b` 在 CUDA）

### 12.1 实验目的

模拟真实 LLM 的典型场景：**权重在 GPU、输入在 CPU**。验证：

1. scheduler 是否把 `MUL_MAT` 判给 CUDA；
2. `split->n_inputs` 是否变为 1；
3. 跨 backend copy 如何被描述、如何被执行；
4. gallocr 如何为 copy tensor 分配显存。

### 12.2 改动

在 `build_graph()` 之后、`compute()` 之前，手动把两个 leaf 放到不同 buffer：

```cpp
// b -> CUDA buffer（权重）
model.buf_b = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0), ggml_nbytes(model.b));
ggml_backend_tensor_alloc(model.buf_b, model.b, ggml_backend_buffer_get_base(model.buf_b));

// a -> CPU buffer（输入）
model.buf_a = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(model.a));
ggml_backend_tensor_alloc(model.buf_a, model.a, ggml_backend_buffer_get_base(model.buf_a));
```

源码见 `variants/exp2_cross_backend.cpp`。

### 12.3 运行结果

```text
a buffer = CPU, b buffer = CUDA0
result ne=[4,3]
    60.00    55.00    50.00   110.00
    90.00    54.00    54.00   126.00
    42.00    29.00    28.00    64.00
```

数值正确，说明 copy 生效。

### 12.4 `GGML_SCHED_DEBUG=2`

```text
## SPLIT #0: CUDA0 # 1 inputs: [leaf_0 (   0K)]
node #  0 (   MUL_MAT):               node_0 (   0K) [CUDA0] use=0,c=1:
                                      CUDA0#leaf_0#0 (   0K) [ NULL ]   leaf_1 (   0K) [CUDA0]
```

三个信息：

```text
1. 仍然是 1 个 split，backend 是 CUDA0
2. split 有 1 个 input：leaf_0（即 a）
3. node_0 的 src[0] 已经被替换成 CUDA0#leaf_0#0，它在 split 阶段还没有 buffer（NULL）
```

### 12.5 split 后的完整状态

```text
=== [E2a] after split ===
n_splits=1 | sched->graph.size=4 n_nodes=3 n_leafs=2
split[0]: backend_id=0 i_start=0 i_end=1 n_inputs=1
split[0].inputs[0]=leaf_0 buffer=CPU
node_backend_ids[0]=1 [1]=0 [2]=0 leaf[0]=1 leaf[1]=0
copy node[0]: name=leaf_0 (view) op=NONE
copy node[1]: name=CUDA0#leaf_0#0 op=NONE buffer=NULL
real node[2]: name=node_0 op=MUL_MAT src0=CUDA0#leaf_0#0 src1=leaf_1
```

这是本次实验最重要的数据，逐条解释：

| 数据 | 含义 |
|---|---|
| `sched->graph.size = 4` | `max(1,2) + total_inputs(1) * 2 * n_copies(1) = 4` |
| `sched->graph.n_nodes = 3` | 原始 1 个节点 + 2 个额外节点（依赖 view + copy） |
| `node_backend_ids[0] = 1` | 依赖 view 属于 CPU（源 backend） |
| `node_backend_ids[1] = 0` | copy 属于 CUDA |
| `node_backend_ids[2] = 0` | 真正的 MUL_MAT 属于 CUDA |
| `leaf[0] = 1` | `a` 被显式放到 CPU buffer，pass 1 直接采用 |
| `leaf[1] = 0` | `b` 被显式放到 CUDA buffer，pass 1 直接采用 |
| `node[0] = leaf_0 (view)` | `ggml_view_tensor(ctx, input)`，`src[0] = input`，用于保活 |
| `node[1] = CUDA0#leaf_0#0` | `ggml_dup_tensor_layout` 创建的真实 copy tensor |
| `node[2].src[0] = CUDA0#leaf_0#0` | 原节点的 `src[0]` 被改写成 copy |

### 12.6 为什么图副本里多出两个节点

源码（`ggml-backend.cpp:1498`）：

```cpp
for (int j = 0; j < split->n_inputs; j++) {
    struct ggml_tensor * input = split->inputs[j];
    struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

    // 1) 依赖节点：保证 input 在 copy 完成前不被 gallocr 释放
    struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
    input_dep->src[0] = input;
    sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
    graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

    // 2) copy 节点：让 gallocr 在 split 开始时为它分配显存
    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
    graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
}
```

两个节点的作用完全不同：

```text
input_dep（view）：
    - 是 view，gallocr 不为它分配内存（dst buffer_id = -1）
    - 但它把 input 作为 src[0]，使 input 的 n_children +1
    - 结果：input 不会在 copy 之前被 free

input_cpy（copy）：
    - 普通 tensor，gallocr 在 CUDA buffer 中分配
    - 作为 node[2] 的 src[0] 被消费
```

断点验证 `input_dep` 的 gallocr 记录：

```text
node[0]=leaf_0 (view) dst{buf=-1,off=18446744073709551615,sz=0}
node[1]=CUDA0#leaf_0#0 dst{buf=0,off=0,sz=32}
node[2]=node_0 dst{buf=0,off=128,sz=48} src0{buf=0,off=0}
```

`buf=-1`、`off=SIZE_MAX` 正是 view 的标记。

### 12.7 执行期的 copy 路径

```text
=== [E2b] compute_splits ===
n_splits=1 split[0].n_inputs=1
input=leaf_0 | split graph node[0]=node_0 op=MUL_MAT src0=CUDA0#leaf_0#0

=== [E2c] cuda_cpy_tensor_async: src=leaf_0(CPU) dst=CUDA0#leaf_0#0(CUDA0)

=== [E2d] tensor_copy fallback: leaf_0(CPU) -> CUDA0#leaf_0#0(CUDA0) nbytes=32
#0 ggml_backend_tensor_copy (...) at ggml-backend.cpp:488
#1 ggml_backend_sched_compute_splits (...) at ggml-backend.cpp:1792
#2 ggml_backend_sched_graph_compute_async (...)
#3 ggml_backend_sched_graph_compute (...)

=== [E2f] cuda set_tensor: tensor=CUDA0#leaf_0#0 dst=0x7f188cc00200 size=32
```

执行顺序：

```text
1. scheduler 取 input_backend = CPU，input_cpy = CUDA0#leaf_0#0
2. 尝试 split_backend->iface.cpy_tensor_async(CPU, CUDA, a, copy)
3. CUDA 实现检查 src/dst buffer 是否都是 CUDA → src 是 CPU → 返回 false
4. 降级：
     ggml_backend_synchronize(CPU)
     ggml_backend_synchronize(CUDA)
     ggml_backend_tensor_copy(a, copy)
5. tensor_copy 内部走 buffer 接口：CPU 侧 get + CUDA 侧 set
6. 最终落到 cudaMemcpyAsync(H2D) + cudaStreamSynchronize
```

这就是真实 LLM 中“输入在 CPU、计算在 GPU”的数据通路。

### 12.8 gallocr 只为 copy 和结果分配

```text
=== [E2g] gallocr after alloc (cross-backend) ===
sched->graph n_nodes=3 | galloc n_nodes=3 n_leafs=2
talloc[0] chunk0.max_size=256 n_free_blocks=2
cuda vbuffer base=0x7f70b4c00200 size=256
cpu  vbuffer=0x555b52c3b9f0
leaf_0(a) data=0x555b52e16440 buffer=CPU
leaf_1(b) data=0x7f70b4c00000 buffer=CUDA0
node[1] data=0x7f70b4c00200
node[2] data=0x7f70b4c00280
```

关键点：

| 观察 | 含义 |
|---|---|
| `chunk0.max_size = 256` | 只有 copy(32→128) + result(48→128) 参与 gallocr |
| `a`、`b` 未参与 | 它们已有 buffer，`ggml_gallocr_init_tensor` 跳过 |
| CUDA vbuffer base = `...0200` | 与 `b` 的 buffer base `...0000` 不同 |
| `b` 占 `0000..0018` | 24 字节，但 cudaMalloc 实际按 512 B 粒度返回 |
| copy @ `+0x00`、result @ `+0x80` | gallocr 内部偏移 |

`0x200` 的间隔不是 gallocr 造成的，而是 **CUDA 分配器的粒度**：小分配按 512 B 对齐，所以第二次 `cudaMalloc` 从 `+0x200` 开始。这一点在分析显存碎片时很重要——`ggml_nbytes` 和对齐之外，还有驱动层的取整。

### 12.9 为什么仍然是 1 个 split

直觉上“CPU 一个 split、CUDA 一个 split”才对。实际不是，因为：

```text
1. 输入的搬运由 scheduler 在 split 开始前完成（copy 节点）
2. 计算全部在 CUDA，不需要在 CPU 上执行任何 op
3. split 的划分依据是「node 的 backend 归属」，不是「数据位置」
```

只有当某个 **node** 被判给 CPU（例如 CUDA 不支持该 op）时，才会出现第二个 split。

### 12.10 设计解读

```text
1. 数据位置与计算位置解耦：
   node 归属看能力，输入位置只影响是否需要 copy
2. copy 是显式节点：
   进入图副本，参与 gallocr 的生命周期分析，避免 input 被提前释放
3. 异步优先、同步兜底：
   cpy_tensor_async 失败后自动降级到 get+set，保证正确性
4. copy 在 split 内完成：
   不产生额外的 CPU split，减少 split 间同步
```

---

## 13 追加实验三：三层 MLP 与 gallocr 复用

### 13.1 实验目的

验证：

1. 多节点图的 split 与执行；
2. free-list 复用的真实时序；
3. 峰值显存 vs 朴素求和；
4. inplace 复用（`GGML_OP_UNARY` 可以复用输入内存）。

### 13.2 图结构

```text
K = 64, H1 = 128, H2 = 64, H3 = 128, B = 3

w0 [64,128]   x [64,3]
t0 = mul_mat(w0, x)    -> [128,3]   1536 B
w1 [128,64]
t1 = mul_mat(w1, t0)   -> [64,3]     768 B
w2 [64,128]
t2 = mul_mat(w2, t1)   -> [128,3]   1536 B
```

源码见 `variants/exp3_mlp3.cpp`；带 silu 的版本见 `variants/exp3b_mlp3_silu.cpp`。

### 13.3 DFS 顺序

`ggml_build_forward_expand(gf, t2)` 的后序 DFS：

```text
visit(t2)
 ├── visit(w2)  → leaf_0
 └── visit(t1)
      ├── visit(w1) → leaf_1
      └── visit(t0)
           ├── visit(w0) → leaf_2
           ├── visit(x)  → leaf_3
           └── t0 → node_0
      └── t1 → node_1
 └── t2 → node_2
```

因此：

```text
leafs = [w2, w1, w0, x]
nodes = [t0, t1, t2]
```

这个顺序决定了分配顺序，也解释了后面所有偏移。

### 13.4 运行结果

```text
## SPLIT #0: CUDA0 # 0 inputs
node #  0 (MUL_MAT): t0 (1K) [CUDA0] use=1,c=1: w0 (32K) [CUDA0]  x  (0K) [CUDA0]
node #  1 (MUL_MAT): t1 (0K) [CUDA0] use=1,c=1: w1 (32K) [CUDA0]  t0 (1K) [CUDA0]
node #  2 (MUL_MAT): t2 (1K) [CUDA0] use=0,c=1: w2 (32K) [CUDA0]  t1 (0K) [CUDA0]

t2 ne=[128,3] first=524288 expected=524288
```

数值验证：

```text
w 全 1，x 全 1
t0 = sum(64 个 1) = 64
t1 = sum(128 个 64) = 8192
t2 = sum(64 个 8192) = 524288
```

### 13.5 分配/释放时序（关键）

```text
ALLOC w2  req=32768 | n_chunks=0
ALLOC w1  req=32768 | n_chunks=1
        [0] off=32768  size=huge
ALLOC w0  req=32768 | n_chunks=1
        [0] off=65536  size=huge
ALLOC x   req=768   | n_chunks=1
        [0] off=98304  size=huge
ALLOC t0  req=1536  | n_chunks=1
        [0] off=99072  size=huge
FREE  addr={chunk=0, off=65536} size=32768        # w0
FREE  addr={chunk=0, off=98304} size=768          # x
ALLOC t1  req=768   | n_chunks=1
        [0] off=65536  size=33536                 # w0 + x 合并
        [1] off=100608 size=huge
FREE  addr={chunk=0, off=32768} size=32768        # w1
FREE  addr={chunk=0, off=99072} size=1536         # t0
ALLOC t2  req=1536  | n_chunks=1
        [0] off=32768  size=32768
        [1] off=66304  size=huge
FREE  addr={chunk=0, off=0}     size=32768        # w2
FREE  addr={chunk=0, off=65536} size=768          # t1

=== final: chunk0.max_size=100608 n_free_blocks=2 ===
free[0] off=0      size=32768
free[1] off=34304  size=huge
```

### 13.6 逐条解读

**1）为什么 `w0` 和 `x` 在 `t0` 之后立刻被释放**

```text
w0 只被 t0 消费（n_children = 1）
x  只被 t0 消费（n_children = 1）
t0 处理完 → 两者 n_children 归零 → free_node
```

**2）为什么 `w0` 和 `x` 的块合并成 33536**

```text
w0 @65536..98304
x  @98304..99072
地址相邻 → ggml_dyn_tallocr_free_bytes 合并为 [65536, 33536]
```

**3）为什么 `t1` 落在 65536 而不是 98304**

`t1` 需要 768 B，合并后的块 [65536, 33536] 足够。此时 free list 只有两个块：

```text
[65536, 33536]  ← 非尾部
[100608, huge]  ← 尾部（可增长）
```

`ggml_dyn_tallocr_alloc` 的 best-fit **只搜索“非尾部块”**：

```cpp
for (int i = 0; i < chunk->n_free_blocks - 1; i++) { ... }
```

尾部块被视为“可增长区”，留给需要扩展的情况。所以 `t1` 只能落在 65536。

**4）为什么 `t2` 落在 32768 而不是 66304**

`t2` 分配前：

```text
FREE w1 @32768 (32768)
FREE t0 @99072 (1536) → 与尾部 [100608, huge] 合并
free list = [32768, 32768], [66304, huge]
```

同样，尾部块 [66304, huge] 不参与 best-fit，`t2` 只能选 [32768, 32768]，落在 32768。

**这条规则是 gallocr 的一个隐含策略**：

```text
保留最后一个 free block 作为“可增长尾部”
优点：避免为了一个稍大的 tensor 频繁开新 chunk
代价：可能放弃更贴合的尾部空间，产生轻微碎片
```

**5）峰值 100608 的来源**

```text
t0 分配后：99072 + 1536 = 100608
后续 t1/t2 都复用了已释放的块，没有超过这个值
```

**6）朴素求和 vs 实际峰值**

```text
朴素求和 = w0 + w1 + w2 + x + t0 + t1 + t2
         = 32768*3 + 768 + 1536 + 768 + 1536
         = 102912 B
实际峰值 = 100608 B
节省     = 2304 B（约 2.2%）
```

小图节省比例不高，因为三个权重一直存活。真实 LLM 中权重也常驻，但**激活**的复用比例很高：

```text
每层的 QKV 输出、attention 输出、FFN 中间结果
生命周期只覆盖相邻几个节点
峰值 ≈ 权重 + 少数几层激活
而不是 权重 + 所有层激活之和
```

### 13.7 inplace 复用（exp3b）

在 `t0 → t1`、`t1 → t2` 之间插入 `silu`：

```text
t0 = mul_mat(w0, x)
s0 = silu(t0)      ← UNARY 可以 inplace
t1 = mul_mat(w1, s0)
s1 = silu(t1)
t2 = mul_mat(w2, s1)
```

分配结果：

```text
[E3b] place: w2   off=0      size=32768
[E3b] place: w1   off=32768  size=32768
[E3b] place: w0   off=65536  size=32768
[E3b] place: x    off=98304  size=768
[E3b] place: t0   off=99072  size=1536
[E3b] place: s0   off=99072  size=1536     ← 与 t0 相同
[E3b] place: t1   off=65536  size=768
[E3b] place: s1   off=65536  size=768      ← 与 t1 相同
[E3b] place: t2   off=32768  size=1536
```

同时命中：

```text
[E3c] free_extra_space: node=s0 parent=t0
[E3c] free_extra_space: node=s1 parent=t1
```

`ggml_gallocr_free_extra_space` 的作用：

```text
inplace 复用时，如果 parent 比 node 大，
把 parent 块末尾多余的部分归还给 free list
本例 s0 与 t0 大小相同，extra_size = 0，函数被调用但不释放字节
```

inplace 的条件（`ggml_gallocr_allocate_node`）：

```text
1. ggml_op_can_inplace(node->op) 为 true（UNARY/ADD/MUL/SOFT_MAX 等）
2. parent 由 gallocr 分配（不是外部 buffer）
3. parent 不是 OUTPUT
4. node 与 parent layout 相同（ggml_are_same_layout）
5. parent 只有一个消费者且没有 view
```

`MUL_MAT` 不满足条件 4（输出 `[M,N]` 与输入 `[K,M]` 不同），所以 t0/t1 不能 inplace 到权重。

### 13.8 最终 free list

```text
chunk0.max_size = 100608
n_free_blocks  = 2
free[0] off=0     size=32768      # w2 的块
free[1] off=34304 size=huge       # t2 之后的所有空间
```

`34304 = 32768 + 1536`，即 t2 的末尾。t2 是图输出，没有被 free（虽然也没有标 `OUTPUT`，但没有消费者，`free_node` 不会被调用）。

### 13.9 设计解读

**1）静态生命周期分析的价值**

```text
O(V + E) 一次遍历，得到每个 tensor 的 (buffer_id, offset)
运行期零分配决策
对 LLM 这种“图结构稳定、重复执行”的场景最优
```

**2）free list 的两个策略**

```text
best-fit（非尾部）：尽量复用已释放的块，减少峰值
尾部保留：允许增长，避免频繁开新 chunk
```

**3）inplace 是“零成本”优化**

```text
不增加显存，不增加拷贝
只要求 op 满足 layout 与消费者条件
对激活链（silu/gelu/add/scale）效果显著
```

**4）权重常驻，激活复用**

```text
权重：整个 compute 期间存活，不参与复用
激活：生命周期短，复用率高
这就是为什么 LLM 的显存峰值 ≈ 权重 + 少量激活
```

---

## 14 三组实验的横向结论

| 实验 | 改了什么 | 观察到的机制 | 对应文档章节 |
|---|---|---|---|
| 一 | `ne11` 3→4 | 决策树阈值、cuBLAS 初始化、列主序映射 | `03-compute.md` §8.4 |
| 二 | `a` 放 CPU、`b` 放 CUDA | 跨 backend copy、input_dep 保活、异步降级 | `03-compute.md` §2.2、§12.4 |
| 三 | 3 层 MLP | free-list 复用、尾部保留策略、inplace | `03-compute.md` §3.11、§13 |

三组实验共同验证了一件事：

```text
ggml 的执行链是「静态规划 + 动态执行」
  规划：split、buffer 归属、offset、生命周期 —— 在 alloc 阶段一次完成
  执行：只做拷贝、launch、同步 —— 运行期不再决策
```

这也是为什么同一个 `simple-backend.cpp` 骨架可以放大到真实 LLM：

```text
1 个 MUL_MAT   → 几百个算子
3 个 tensor    → 上千个 tensor
384 B buffer   → 几 GB 显存
1 个 split     → 多个 split
但 split → alloc → copy → launch → sync 的流程完全不变
```

---

## 15 断点与源码行号对照

本节把本次调试用到的每个断点映射到具体源码行，并给出该行的作用。行号对应云端 `/home/ggml`（commit `353b63b`）。

### 15.1 对照表

| 断点标签 | 函数 | 文件:行 | 这一行在做什么 |
|---|---|---|---|
| BP1 | `ggml_backend_load_all` | `ggml-backend-reg.cpp:574` | 转调 `load_all_from_path(nullptr)` |
| BP2 | `ggml_backend_init_best` | `ggml-backend-reg.cpp:382` | 按 GPU → IGPU → CPU 选设备 |
| BP3 | `ggml_backend_init_by_type` | `ggml-backend-reg.cpp:374` | 按类型选设备并初始化 |
| BP4 | `ggml_backend_cuda_init` | `ggml-cuda.cu:5763` | 创建 CUDA backend 实例 |
| BP5 | `ggml_backend_sched_new` | `ggml-backend.cpp:1848` | 分配 scheduler 全部元数据 |
| BP6 | `ggml_init` | `ggml.c:1611` | 建立 context |
| BP7 | `ggml_new_tensor_impl` | `ggml.c:1766` | 创建 tensor 元数据 + 计算 nb |
| BP8 | `simple-backend.cpp:91` | 源码 | `ggml_mul_mat` 返回后 |
| BP9 | `simple-backend.cpp:93` | 源码 | `ggml_build_forward_expand` 后 |
| BP10 | `simple-backend.cpp:95` | 源码 | `ggml_free(ctx)` 后 |
| BP11 | `ggml_free` | `ggml.c:1663` | 释放 context 结构 |
| A1 | `ggml_backend_sched_alloc_graph` | `ggml-backend.cpp:1992` | split + alloc 入口 |
| A2 | `ggml_backend_sched_split_graph` | `ggml-backend.cpp:1066` | 五个 pass |
| A3 | `ggml_backend_sched_alloc_splits` | `ggml-backend.cpp:1591` | 尝试 alloc，失败则 reserve |
| A4 | `ggml_gallocr_alloc_graph` | `ggml-alloc.c:1052` | 写 tensor->data |
| A5 | `ggml_gallocr_reserve_n` | `ggml-alloc.c:962` | 生命周期分析 + 申请 |
| A6 | `ggml_backend_buft_alloc_buffer` | `ggml-backend.cpp:39` | 分发到 buft iface |
| A7 | `ggml_backend_cuda_buffer_type_alloc_buffer` | `ggml-cuda.cu:884` | `cudaMalloc` |
| A8 | `ggml-cuda.cu:898` | `ggml-cuda.cu:898` | `new buffer_context` |
| A9 | `ggml_backend_buffer_init` | `ggml-backend.cpp:96` | 组装 buffer 结构 |
| B1 | `ggml_backend_tensor_set` | `ggml-backend.cpp:335` | 走 buffer 的 set_tensor |
| B2 | `ggml_backend_cuda_buffer_set_tensor` | `ggml-cuda.cu:785` | `cudaMemcpyAsync` H2D + sync |
| C1 | `ggml_backend_sched_graph_compute` | `ggml-backend.cpp:2011` | async + synchronize |
| C2 | `ggml_backend_sched_compute_splits` | `ggml-backend.cpp:1646` | 遍历 split |
| C3 | `ggml_backend_graph_compute_async` | `ggml-backend.cpp:461` | 走 backend iface |
| C4 | `ggml_backend_cuda_graph_compute` | `ggml-cuda.cu:4425` | CUDA 后端入口 |
| C5 | `ggml_cuda_graph_evaluate_and_capture` | `ggml-cuda.cu:4185` | node 循环 |
| D1 | `ggml_cuda_compute_forward` | `ggml-cuda.cu:2068` | `switch(op)` |
| D2 | `ggml_cuda_mul_mat` | `ggml-cuda.cu:1824` | 路径决策树 |
| D3 | `ggml_cuda_mul_mat_vec_f` | `mmvf.cu:634` | MMVF 入口 |
| E1 | `ggml_backend_tensor_get` | `ggml-backend.cpp:350` | 走 buffer 的 get_tensor |
| E2 | `ggml_backend_cuda_buffer_get_tensor` | `ggml-cuda.cu:793` | `cudaMemcpyAsync` D2H + sync |
| F1 | `ggml_backend_cuda_buffer_free_buffer` | `ggml-cuda.cu:742` | `delete ctx` → `cudaFree` |
| F2 | `ggml_backend_cuda_free` | `ggml-cuda.cu:2442` | `delete cuda_ctx` |
| F3 | `ggml_backend_sched_free` | `ggml-backend.cpp:1920` | 释放 buffer + 元数据 |
| G1 | `ggml_vbuffer_tensor_alloc` | `ggml-alloc.c:442` | `base + offset` 写入 tensor |
| H1 | `mmvf.cu:455` | `mmvf.cu:455` | 计算 grid/block/shared |
| H2 | `ggml_cuda_mul_mat_vec_f` 入口 | `mmvf.cu:634` | 取 stride、ncols_dst |
| I1 | `simple-backend.cpp:65` | 源码 | init_model 结束 |
| I3 | `ggml_backend_cuda_synchronize` | `ggml-cuda.cu:2548` | `cudaStreamSynchronize` |

### 15.2 几个关键行的源码展开

#### `ggml-backend-reg.cpp:574`：`load_all`

```cpp
void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}
```

只有一行转调。`nullptr` 意味着使用默认搜索路径（可执行文件目录 + cwd）。断点 BP1 在这里命中，然后一路走到 `load_best("cuda", ...)`，发现文件名不匹配而返回。

#### `ggml-backend.cpp:1066`：`split_graph` 入口

```cpp
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
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
    ...
}
```

断点 A2 命中在函数首行，此时 `n_splits` 还没清零（打印出来是 0，因为上一轮也是 0）。真正清零是 1068 行。

#### `ggml-alloc.c:1052`：`alloc_graph`

```cpp
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
            return false;
        }
    }
    ...
}
```

断点 A4 第一次命中时 `galloc->n_nodes = 0`，`needs_realloc` 返回 true，`n_buffers = 2`，直接 `return false`。这就是日志 `cannot reallocate multi buffer graph automatically` 的来源。

第二次命中时 `galloc->n_nodes = 1`，与图一致，跳过 realloc，进入 `ggml_gallocr_init_tensor`。

#### `ggml-cuda.cu:884`：CUDA buffer 分配

```cpp
static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx =
        (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", ...);
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}
```

断点 A7 命中时 `size = 384`、`device = 0`。A8 断在 `new buffer_context` 这一行（898），此时 `dev_ptr` 已经有效。

#### `mmvf.cu:455`：launch 参数计算

```cpp
const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);
const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
const dim3 block_dims(block_size_best, 1, 1);
```

断点 H1 停在这里，此时三个值都已算好：`nbytes_shared=128`、`block_nums=(4,1,1)`、`block_dims=(32,1,1)`。

### 15.3 断点为什么会“偏移”

调试 CUDA 源码时经常遇到“断点行号与源码不符”：

```text
gdb 显示 ggml_backend_cuda_init at ggml-cuda.cu:5843
源码定义在                             5763
```

原因：

```text
1. nvcc 生成的行号信息指向“实际指令”，不是函数首行
2. 模板实例化时行号可能来自模板定义处
3. #include 展开后行号对应展开后的位置
4. 内联函数被折叠到调用者
```

应对：

```text
1. 用 `break function_name` 而不是 `break file:line`
2. 用 `info line *$pc` 确认实际位置
3. 用 `bt` 看调用栈，确认进入的是哪个重载/模板
```

### 15.4 断点顺序与源码执行顺序的对应

```text
源码顺序                                   断点顺序
─────────────────────────────────────────────────────────
ggml_backend_load_all()                    BP1
ggml_backend_init_best()                   BP2 → BP4
ggml_backend_init_by_type(CPU)             BP3
ggml_backend_sched_new()                   BP5
build_graph()
  ggml_init()                              BP6
  ggml_new_tensor_impl() ×3                BP7 ×3
  ggml_mul_mat() 返回                      BP8
  ggml_build_forward_expand() 返回         BP9
  ggml_free(ctx)                           BP11 → BP10
compute()
  sched_reset()
  sched_alloc_graph()                      A1
    split_graph()                          A2
    alloc_splits()                         A3
      gallocr_alloc_graph()                A4（失败）
      gallocr_reserve_n()                  A5
        buft_alloc_buffer()                A6
          cuda buft alloc                  A7 → A8
        buffer_init()                      A9
      gallocr_alloc_graph()                A4（成功）
  tensor_set(a)                            B1 → B2
  tensor_set(b)                            B1 → B2
  sched_graph_compute()                    C1
    compute_splits()                       C2
      graph_compute_async()                C3
        cuda graph_compute()               C4
          evaluate_and_capture()           C5
            compute_forward()              D1
              mul_mat()                    D2
                mul_mat_vec_f()            D3 / H2
                  launch                   H1
    synchronize()                          I3
  tensor_get(result)                       E1 → E2
sched_free()                               F3
  gallocr_free()                           F1
backend_free()                             F2
```

**读这张表的方法**：先看源码顺序，再看断点顺序，任何“断点顺序与源码顺序不一致”的地方都值得深究。本例唯一的不一致是 `BP11 → BP10`（`ggml_free` 在函数内，`BP10` 是它返回后的行）。

---

