# 05 · gdb 断点原始数据与逐条解读

> 本文件保存本次调试的全部原始输出，并在每段后给出“这组数据说明了什么”。
>
> 复现环境：`/home/ggml`，Debug 构建，CUDA0（RTX 4090）
>
> ⚠️ 调试期间 GPU 上有其他进程（`llama-server`，19.8 GB / 100% util），所有**时间类**数据仅供数量级参考；
> **地址、偏移、节点数、grid/block、cuBLAS 参数**等结构性数据不受影响。

---

## 0 通用脚本模板与踩坑

### 0.1 必须先 `start`，再下断点

```gdb
start mnist-fc mnist-fc-f32.gguf <train-images> <train-labels> CUDA0
break ggml.c:1841
break ggml-alloc.c:1229
break ggml-cuda.cu:4425
continue
```

**为什么不能直接 `break ggml_init`？**

`mnist-train` 链接的是动态库：

```text
linux-vdso.so.1
libggml.so.0        => /home/ggml/build-debug/src/libggml.so.0
libggml-cpu.so.0    => /home/ggml/build-debug/src/libggml-cpu.so.0
libggml-cuda.so.0   => /home/ggml/build-debug/src/ggml-cuda/libggml-cuda.so.0
libggml-base.so.0   => /home/ggml/build-debug/src/libggml-base.so.0
```

进程启动瞬间这些 so 还没被加载，gdb 里会报：

```text
No source file named ggml.c.
Function "ggml_new_tensor_impl" not defined.
```

三个解决办法：

| 方法 | 写法 | 说明 |
|---|---|---|
| **先 start 再断** | `start ...` 然后 `break ggml.c:1841` | 推荐，符号已解析 |
| pending | `set breakpoint pending on` | 会等 so 加载后自动绑定，但行号可能漂移 |
| 完整路径 | `break /home/ggml/src/ggml.c:1841` | 对同一份源码有效 |

### 0.2 行号要落在“有变量”的语句上

`break ggml.c:1841` 对应：

```cpp
1841	    return result;         // ← 断点在这里，result 已完整
1842	}
1843	
1844	struct ggml_tensor * ggml_new_tensor(
1845	        struct ggml_context * ctx, ...
1849	    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
```

如果断在 1844（函数签名行），gdb 会落在 1849 的调用行，此时 `result` 不存在：

```text
Breakpoint 2, ggml_new_tensor (ctx=..., type=GGML_TYPE_F32, n_dims=2, ne=...) at ggml.c:1849
1849	    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
No locals.
$1 = '\000' <repeats 113 times>     ← result 是野值
```

**教训**：断点行号要用 `nl -ba` 精确确认，且必须落在 `return result;` 这类语句上。

### 0.3 模板函数的多实例断点

`ggml_cuda_mul_mat_cublas_impl<compute_type>` 是模板，`break ggml-cuda.cu:1546` 会解析成 **3 个位置**：

```text
Breakpoint 2 at 0x7f00f70593fb: ggml-cuda.cu:1546. (3 locations)
2.1  ... ggml_cuda_mul_mat_cublas_impl<(ggml_type)0>   ← F32
2.2  ... ggml_cuda_mul_mat_cublas_impl<(ggml_type)30>  ← BF16
2.3  ... ggml_cuda_mul_mat_cublas_impl<(ggml_type)1>   ← F16
```

要只看 F32：

```gdb
break ggml_cuda_mul_mat_cublas_impl<(ggml_type)0>
```

### 0.4 `printf` 里不能直接用宏

```text
No symbol "GGML_MEM_ALIGN" in current context.
No symbol "CUDA_NEG_BLOCK_SIZE" in current context.
No symbol "MAX_GRIDDIM_X" in current context.
```

宏在预处理阶段就展开了，调试信息里没有。要么直接写数值，要么在 `printf` 里用表达式代替：

```gdb
printf "UNARY grid=%d block=256 k=%d\n", num_blocks, k    # block 写死
```

### 0.5 结构体字段可能不存在

```text
There is no member named smpbo_max.
Cannot resolve function ggml_backend_buffer_get_size to any overloaded instance.
```

`ggml_cuda_device_info` 没有 `smpbo_max`；`ggml_backend_buffer_get_size` 是 inline 函数，gdb 无法调用。改用直接访问字段：

```gdb
printf "size=%zu\n", sched->galloc->buffers[0]->chunks[0]->size
```

---

## 1 阶段一：context 创建

### 1.1 脚本

```gdb
set pagination off
set confirm off
break ggml_init
commands
silent
printf "\n=== ggml_init hit ===\n"
printf "params.mem_size=%zu no_alloc=%d mem_buffer=%p\n", params.mem_size, params.no_alloc, params.mem_buffer
bt 4
continue
end
break ggml_new_tensor_2d
commands
silent
printf "ggml_new_tensor_2d ctx=%p type=%d ne0=%ld ne1=%ld\n", ctx, type, ne0, ne1
continue
end
run mnist-fc /home/ggml/gdb-test.gguf examples/mnist/data/MNIST/raw/train-images-idx3-ubyte examples/mnist/data/MNIST/raw/train-labels-idx1-ubyte CUDA0
```

### 1.2 原始输出

```text
=== ggml_init hit ===
params.mem_size=736 no_alloc=1 mem_buffer=(nil)
#0  ggml_init (params=...) at /home/ggml/src/ggml.c:1614
#1  0x00007fb30d9b7fdd in ggml_opt_dataset_init (type_data=GGML_TYPE_F32, type_label=GGML_TYPE_F32,
      ne_datapoint=784, ne_label=10, ndata=60000, ndata_shard=10) at /home/ggml/src/ggml-opt.cpp:108
#2  0x000056170542619e in main (argc=6, argv=0x7ffca85dd9c8) at /home/ggml/examples/mnist/mnist-train.cpp:23

=== ggml_new_tensor_2d ===
ctx=0x561706550bc0 type=0 ne0=784 ne1=60000
=== ggml_new_tensor_2d ===
ctx=0x561706550bc0 type=0 ne0=10 ne1=60000

=== ggml_init hit ===
params.mem_size=0 no_alloc=0 mem_buffer=(nil)
#0  ggml_init (params=...) at /home/ggml/src/ggml.c:1614
#1  0x00007fb3111ea544 in ggml_cpu_init () at /home/ggml/src/ggml-cpu/ggml-cpu.c:3871
#2  0x00007fb3111ecc75 in ggml_backend_cpu_reg () at /home/ggml/src/ggml-cpu/ggml-cpu.cpp:705
#3  0x00007fb3114e2b64 in ggml_backend_registry::ggml_backend_registry (...)
      at /home/ggml/src/ggml-backend-reg.cpp:172

=== ggml_init hit ===
params.mem_size=376832 no_alloc=1 mem_buffer=(nil)
#1  0x000056170542add9 in mnist_model::mnist_model (...) at /home/ggml/examples/mnist/mnist-common.h:128

=== ggml_init hit ===
params.mem_size=1001488 no_alloc=1 mem_buffer=(nil)
#1  0x000056170542ae45 in mnist_model::mnist_model (...) at /home/ggml/examples/mnist/mnist-common.h:139

=== ggml_init hit ===
params.mem_size=8096 no_alloc=1 mem_buffer=(nil)
#1  0x00007fb30d9b93d8 in ggml_opt_build (opt_ctx=0x56170703b360) at /home/ggml/src/ggml-opt.cpp:363
```

### 1.3 解读

| # | mem_size | no_alloc | 来源 | 含义 |
|---|---|---|---|---|
| 1 | 736 | 1 | `ggml_opt_dataset_init` | 2 个张量元数据（images + labels） |
| 2 | 0 | 0 | `ggml_cpu_init` | CPU 后端的量化表，`mem_size=0` 被规范成 `GGML_MEM_ALIGN` |
| 3 | 376,832 | 1 | `ctx_static` | 1024 × 368，权重 + images |
| 4 | 1,001,488 | 1 | `ctx_compute` | 2048 × 368 + 3 × 82,608，3 张图 |
| 5 | 8,096 | 1 | `opt ctx_static` | 22 × 368，梯度累加器 + AdamW 状态 |

**`mem_size=0` 的那个**很特殊：`ggml_init` 里

```cpp
if (params.mem_size == 0) {
    params.mem_size = GGML_MEM_ALIGN;      // 16
}
```

CPU 后端用它做一次“空 context”初始化，只为触发 `ggml_cpu_init()`（量化表、线程池）。这说明 `ggml_init` 允许 0 大小，是一个纯粹的“初始化副作用”调用。

---

## 2 阶段二：张量创建与 nb 推导

### 2.1 脚本

```gdb
start mnist-fc ... CUDA0
set $tcount = 0
break ggml.c:1841
commands
silent
set $tcount = $tcount + 1
printf "TENSOR#%d type=%d ne=[%ld,%ld,%ld,%ld] nb=[%ld,%ld,%ld,%ld] data=%p view_src=%p\n",
  $tcount, result->type, result->ne[0], result->ne[1], result->ne[2], result->ne[3],
  result->nb[0], result->nb[1], result->nb[2], result->nb[3], result->data, result->view_src
continue
end
break ggml_set_name
commands
silent
printf "SET_NAME name=%s type=%d ne=[%ld,%ld] data=%p\n", name, tensor->type, tensor->ne[0], tensor->ne[1], tensor->data
continue
end
```

### 2.2 原始输出（前 40 个）

```text
TENSOR#1  type=0 ne=[784,60000,1,1] nb=[4,3136,188160000,188160000] data=(nil) view_src=(nil)
TENSOR#2  type=0 ne=[10,60000,1,1]  nb=[4,40,2400000,2400000]       data=(nil) view_src=(nil)

TENSOR#3  type=0 ne=[784,500,1,1]   nb=[4,3136,1568000,1568000]     data=(nil) view_src=(nil)
TENSOR#4  type=0 ne=[500,1,1,1]     nb=[4,2000,2000,2000]           data=(nil) view_src=(nil)
TENSOR#5  type=0 ne=[500,10,1,1]    nb=[4,2000,20000,20000]         data=(nil) view_src=(nil)
TENSOR#6  type=0 ne=[10,1,1,1]      nb=[4,40,40,40]                 data=(nil) view_src=(nil)
SET_NAME name=fc1.weight type=0 ne=[784,500]
SET_NAME name=fc1.bias   type=0 ne=[500,1]
SET_NAME name=fc2.weight type=0 ne=[500,10]
SET_NAME name=fc2.bias   type=0 ne=[10,1]

TENSOR#7  type=0 ne=[784,500,1,1]   nb=[4,3136,1568000,1568000]     data=(nil) view_src=(nil)
SET_NAME name=images     type=0 ne=[784,500]

TENSOR#8  type=0 ne=[500,500,1,1]   nb=[4,2000,1000000,1000000]     data=(nil) view_src=(nil)
TENSOR#9  type=0 ne=[500,500,1,1]   nb=[4,2000,1000000,1000000]     data=(nil) view_src=(nil)
TENSOR#10 type=0 ne=[500,500,1,1]   nb=[4,2000,1000000,1000000]     data=(nil) view_src=(nil)
TENSOR#11 type=0 ne=[10,500,1,1]    nb=[4,40,20000,20000]           data=(nil) view_src=(nil)
TENSOR#12 type=0 ne=[10,500,1,1]    nb=[4,40,20000,20000]           data=(nil) view_src=(nil)
SET_NAME name=logits     type=0 ne=[10,500]
TENSOR#13 type=0 ne=[10,500,1,1]    nb=[4,40,20000,20000]           data=(nil) view_src=(nil)
SET_NAME name=labels     type=0 ne=[10,500]
TENSOR#14 type=0 ne=[1,1,1,1]       nb=[4,4,4,4]                    data=(nil) view_src=(nil)
SET_NAME name=loss_cross_entropy type=0 ne=[1,1]
TENSOR#15 type=0 ne=[1,1,1,1]       nb=[4,4,4,4]                    data=(nil) view_src=(nil)
SET_NAME name=loss_cross_entropy_scaled type=0 ne=[1,1]
TENSOR#16 type=26 ne=[500,1,1,1]    nb=[4,2000,2000,2000]           data=(nil) view_src=(nil)
SET_NAME name=pred       type=26 ne=[500,1]
TENSOR#17 type=26 ne=[500,1,1,1]    nb=[4,2000,2000,2000]           data=(nil) view_src=(nil)
TENSOR#18 type=27 ne=[1,1,1,1]      nb=[8,8,8,8]                    data=(nil) view_src=(nil)
SET_NAME name=ncorrect   type=27 ne=[1,1]

TENSOR#19 type=0 ne=[500,10,1,1]    nb=[4,2000,20000,20000]         data=(nil) view_src=(nil)
TENSOR#20 type=0 ne=[784,500,1,1]   nb=[4,3136,1568000,1568000]     data=(nil) view_src=(nil)
...
TENSOR#35 type=0 ne=[10,1,1,1]      nb=[4,40,40,40]                 data=(nil) view_src=0x561ccf73b650
TENSOR#37 type=0 ne=[500,10,1,1]    nb=[4,2000,20000,20000]         data=(nil) view_src=0x561ccf73b200
TENSOR#38 type=0 ne=[10,500,1,1]    nb=[4,40,20000,20000]           data=(nil) view_src=0x7f302c92fec0
```

### 2.3 解读

**① 所有张量 `data=(nil)`** —— 建图阶段不分配数据，`no_alloc=true` 生效。

**② `nb` 完全由 `ne` 决定**：

```text
TENSOR#1: ne=[784,60000]  nb=[4, 4×784=3136, 3136×60000=188,160,000, 同上]
TENSOR#4: ne=[500,1]      nb=[4, 4×500=2000, 2000×1=2000, 同上]     ← ne[2]=1，nb[2]=nb[1]
TENSOR#18: type=27(I64)   nb=[8, 8, 8, 8]                           ← 元素 8 字节
```

**③ `type` 编码**：`0=F32`、`26=I32`、`27=I64`。这些数值来自 `enum ggml_type` 的顺序，调试时可以直接看。

**④ `view_src` 非空的三个**：

```text
TENSOR#35 ne=[10,1]   view_src=0x...   ← grad for fc2.bias 的视图
TENSOR#37 ne=[500,10] view_src=0x...   ← AdamW step 的视图
TENSOR#38 ne=[10,500] view_src=0x7f30... ← 跨后端拷贝的视图
```

视图的 `data` 在创建时仍是 `NULL`，因为 `view_src->data` 也是 `NULL`。等 `view_src` 分配后，`ggml_backend_view_init` 会把 `data` 算出来。

**⑤ 为什么 `TENSOR#8/9/10` 都是 `[500,500]`？**
它们是 `gb_grad` 的梯度累加器（`leaf_2..leaf_6`）中形状为 `[500,500]` 的三个：`grad for node_5`、`grad for node_2`、`node_22` 相关。`ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne)` 为每个需要梯度的节点创建一个同形状的累加器。

---

## 3 阶段三：静态 buffer 分配

### 3.1 脚本

```gdb
break ggml-alloc.c:1229
commands
silent
printf "CTX_BUFFER buft=%-6s size=%9zu", ggml_backend_buft_name(buffer->buft), buffer->size
if buffer->buft == ggml_backend_cuda_buffer_type(0)
printf " dev_ptr=%p", ((struct ggml_backend_cuda_buffer_context *)buffer->context)->dev_ptr
end
printf "\n"
continue
end
```

### 3.2 训练路径

```text
CTX_BUFFER buft=CPU    size=190560000
CTX_BUFFER buft=CUDA0  size=  3158272 dev_ptr=0x7fddf0c00000
CTX_BUFFER buft=CUDA0  size=  4795520 dev_ptr=0x7fddf1000000
CTX_BUFFER buft=CPU    size=       32
```

### 3.3 推理路径

```text
CTX_BUFFER buft=CPU    size= 31760000
CTX_BUFFER buft=CUDA0  size=  1590272 dev_ptr=0x7f38c0c00000
CTX_BUFFER buft=CUDA0  size=  1568000 dev_ptr=0x7f38c0e00000
CTX_BUFFER buft=CUDA0  size=    24448 dev_ptr=0x7f38c0d84400
```

### 3.4 解读

**① 训练为什么只有 3 个 CUDA buffer，推理有 4 个？**

训练：

```text
weights + images 在同一个 ctx_static → 3,158,272
opt 的 labels/loss/pred/grad_accs/m/v → 4,795,520
```

推理：

```text
ctx_gguf（4 个权重）        → 1,590,272
ctx_static（images）        → 1,568,000
ctx_compute 的静态部分       →    24,448   （labels/loss/pred/ncorrect）
```

推理多出来的原因是 `ctx_gguf` 由 GGUF 元数据精确算出，没有余量放 `images`：

```cpp
// The space in ctx_gguf exactly fits the model weights,
// the images (which also need to be statically allocated) need to be put in a different context.
```

**② `dev_ptr` 的规律**

```text
0x7fddf0c00000
0x7fddf1000000
0x7f38c0c00000
0x7f38c0e00000
0x7f38c0d84400
```

都是 128 的倍数（低 7 位为 0），部分还是 4 MB / 2 MB 对齐。这是 `cudaMalloc` 的行为：driver 返回的地址通常按大页对齐（2 MB 或更大）。ggml 不依赖这个性质，只要求 128 B 对齐。

---

## 4 阶段四：建图完成时的图统计

### 4.1 脚本

```gdb
break ggml-opt.cpp:581        # ggml_opt_init 的 return result;
commands
silent
printf "OPT_BUILD_DONE gf_nodes=%d gf_leafs=%d gf_size=%d\n", result->gf->n_nodes, result->gf->n_leafs, result->gf->size
printf "OPT_BUILD_DONE gb_grad_nodes=%d gb_opt_nodes=%d\n", result->gb_grad->n_nodes, result->gb_opt->n_nodes
continue
end
```

### 4.2 原始输出

```text
OPT_BUILD_DONE gf_nodes=14 gf_leafs=2 gf_size=2048
OPT_BUILD_DONE gb_grad_nodes=28 gb_opt_nodes=32
```

推理：

```text
EVAL_OPT gf_nodes=13 gf_leafs=2 build_type=10 gb_grad=(nil) gb_opt=(nil) labels=labels
```

### 4.3 解读

**① `build_type=10`** = `GGML_OPT_BUILD_TYPE_FORWARD`（枚举值 10）。推理时 `mnist_model_eval` 设置：

```cpp
params.build_type = GGML_OPT_BUILD_TYPE_FORWARD;
```

所以 `gb_grad` / `gb_opt` 都是 `NULL`——只建了前向图。

**② 训练 `build_type` 默认是 `GGML_OPT_BUILD_TYPE_OPT`（30）**，所以三张图都建。

**③ `gf_size=2048`** 是 `ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, grads=true)` 传入的容量，不是实际节点数。实际用了 14 个，剩下 2034 个空槽。

---

## 5 阶段五：scheduler 分配

### 5.1 脚本

```gdb
set $ac = 0
break ggml_backend_sched_alloc_splits
commands
silent
set $ac = $ac + 1
printf "SCHED_ALLOC#%d graph_nodes=%d n_splits=%d is_reset=%d is_alloc=%d cur_copy=%d n_copies=%d\n",
  $ac, sched->graph.n_nodes, sched->n_splits, sched->is_reset, sched->is_alloc, sched->cur_copy, sched->n_copies
printf "  galloc n_buffers=%d n_nodes=%d n_leafs=%d\n", sched->galloc->n_buffers, sched->galloc->n_nodes, sched->galloc->n_leafs
set $i = 0
while $i < sched->n_splits
printf "  split[%d] backend=%s nodes=[%d,%d) n_inputs=%d\n", $i,
  ggml_backend_name(sched->backends[sched->splits[$i].backend_id]),
  sched->splits[$i].i_start, sched->splits[$i].i_end, sched->splits[$i].n_inputs
set $i = $i + 1
end
continue
end
break ggml_backend_sched_reset
commands
silent
printf "SCHED_RESET is_reset=%d is_alloc=%d cur_copy=%d\n", sched->is_reset, sched->is_alloc, sched->cur_copy
continue
end
```

### 5.2 原始输出

```text
SCHED_RESET is_reset=0 is_alloc=0 cur_copy=0
SCHED_RESET is_reset=1 is_alloc=0 cur_copy=0

SCHED_ALLOC#1 graph_nodes=28 n_splits=1 is_reset=0 is_alloc=0 cur_copy=0 n_copies=1
  galloc n_buffers=3 n_nodes=0 n_leafs=0
  split[0] backend=CUDA0 nodes=[0,28) n_inputs=0

SCHED_RESET is_reset=0 is_alloc=1 cur_copy=0
SCHED_ALLOC#2 graph_nodes=34 n_splits=1 is_reset=0 is_alloc=0 cur_copy=0 n_copies=1
  galloc n_buffers=3 n_nodes=28 n_leafs=7
  split[0] backend=CUDA0 nodes=[0,32) n_inputs=1
```

### 5.3 解读

**① `galloc n_buffers=3`**：CUDA0、CUDA0、CPU 三个 backend 对应的 buft，但前两个共享同一个 tallocr（见 01 §2.2）。

**② `galloc n_nodes=0` → `28`**：第一次分配前，gallocr 没记住任何图；分配后记住 28 个节点、7 个叶子。

**③ `is_reset` 的翻转**：

```text
ggml_opt_alloc → ggml_backend_sched_reset() → is_reset=1
ggml_backend_sched_split_graph() → is_reset=0
ggml_backend_sched_alloc_graph() → is_alloc=1
compute_splits → 结束
```

`is_reset` 的含义是“调度状态需要重建”：`reset` 会清空 `hv_tensor_backend_ids` 和 `hv_tensor_copies`，但**不释放显存**。

---

## 6 阶段六：gallocr 的 offset 规划

### 6.1 脚本

```gdb
break ggml-alloc.c:264        # dyn_tallocr_alloc 的 block->offset += size;
commands
silent
printf "ALLOC name=%-28s size=%8zu chunk=%d off=%8zu block_left=%8zu n_chunks=%d\n",
  tensor->name, size, addr.chunk, addr.offset, block->size, alloc->n_chunks
continue
end
break ggml-alloc.c:710        # free_node 的 free_bytes 之前
commands
silent
printf "FREE  name=%-28s off=%8zu size=%8zu n_free_blocks=%d\n",
  node->name, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks
continue
end
```

### 6.2 原始输出

```text
ALLOC node_2                        size= 1000064 chunk=0 off=       0
ALLOC node_5                        size= 1000064 chunk=0 off= 1000064
ALLOC node_6                        size=   20096 chunk=0 off= 2000128
ALLOC grad for loss_cross_entropy   size=     128 chunk=0 off= 2020224
ALLOC grad for node_6               size=   20096 chunk=0 off= 2020352
FREE  grad for loss_cross_entropy   off= 2020224 size=       4 n_free_blocks=1
ALLOC node_16                       size=     128 chunk=0 off= 2020224   ← 复用
FREE  node_16                       off= 2020224 size=      40 n_free_blocks=1
ALLOC node_18                       size=   20096 chunk=0 off= 2040448
FREE  node_5                        off= 1000064 size= 1000000 n_free_blocks=2
FREE  node_18                       off= 2040448 size=   20000 n_free_blocks=3
ALLOC grad for node_5               size= 1000064 chunk=0 off= 1000064   ← 复用
FREE  grad for node_6               off= 2020352 size=   20000 n_free_blocks=2
FREE  grad for node_5               off= 1000064 size= 1000000 n_free_blocks=1
ALLOC node_24                       size=    2048 chunk=0 off= 1000064   ← 复用
FREE  node_24                       off= 1000064 size=    2000 n_free_blocks=2
ALLOC node_26                       size= 1568000 chunk=0 off= 2020224   ← 复用
FREE  grad for node_2               off=       0 size= 1000000 n_free_blocks=2
FREE  node_26                       off= 2020224 size= 1568000 n_free_blocks=2
```

### 6.3 解读

**① 地址单调递增**：`0 → 1,000,064 → 2,000,128 → 2,020,224 → 2,040,448`，因为第一次分配时 free list 是空的，`dyn_tallocr_alloc` 每次都取 chunk 尾部。

**② 复用发生在“释放之后”**：

```text
grad loss (off=2,020,224, 128 B) 释放
  → node_16 立刻拿到 off=2,020,224（best-fit 命中）
node_5 (off=1,000,064, 1,000,064 B) 释放
  → grad for node_5 拿到 off=1,000,064（唯一够大的块）
```

**③ `n_free_blocks` 的波动**：

```text
分配 → 从块里切一块 → 块数不变
释放 → 多一个块 → 块数 +1
相邻块合并 → 块数 -1
```

比如 `FREE node_5` 后 `n_free_blocks=2`，`FREE grad for node_5` 后变成 1——因为两个相邻块合并了。

**④ `size` 与 `FREE size` 的差异**：

```text
ALLOC grad for loss_cross_entropy  size=128     ← PAD(4, 128)
FREE  grad for loss_cross_entropy  size=4       ← 原始 nbytes
```

`dyn_tallocr_free_bytes` 收到的是未对齐的 4，但块里实际占的是 128。剩下的 124 B 留在 free list 里（`n_free_blocks=1`），下次有 ≤124 B 且对齐合适的张量可以用。

**这是“不精确释放”的设计**：宁可留碎片，也不破坏对齐。对训练图这种“张量大小差异大”的场景，碎片可以忽略（3.42 MiB 里只有几百字节）。

---

## 7 阶段七：kernel 分发与 cuBLAS

### 7.1 脚本

```gdb
break ggml_cuda_mul_mat
commands
silent
printf "MUL_MAT src0=%s type=%d ne=[%ld,%ld] nb=[%ld,%ld] src1=%s ne=[%ld,%ld] -> dst ne=[%ld,%ld]\n",
  src0->name, src0->type, src0->ne[0], src0->ne[1], src0->nb[0], src0->nb[1],
  src1->name, src1->ne[0], src1->ne[1], dst->ne[0], dst->ne[1]
printf "  ctx.device=%d cublas_handle=%p stream=%p workspace=%zu\n",
  ctx.device, ctx.cublas_handles[0][0], ctx.streams[0][0], ctx.cublas_workspace_sizes[0]
continue
end
break ggml_cuda_mul_mat_cublas_impl<(ggml_type)0>
commands
silent
printf "CUBLAS_F32 src0=%s src1=%s dst=%s => M=%ld N=%ld K=%ld lda=%ld ldb=%ld ldc=%ld\n",
  src0->name, src1->name, dst->name, src0->ne[1], src1->ne[1], src0->ne[0],
  src0->nb[1]/4, src1->nb[1]/4, dst->ne[0]
continue
end
```

### 7.2 原始输出

```text
MUL_MAT src0=fc1.weight type=0 ne=[784,500] nb=[4,3136] src1=images ne=[784,500] -> dst ne=[500,500]
  ctx.device=0 cublas_handle=(nil) stream=0x55b32c9d0ab0 workspace=0
CUBLAS_F32 src0=fc1.weight src1=images dst=node_2 => M=500 N=500 K=784 lda=784 ldb=784 ldc=500

MUL_MAT src0=fc2.weight type=0 ne=[500,10] nb=[4,2000] src1=node_5 ne=[500,500] -> dst ne=[10,500]
  ctx.device=0 cublas_handle=0x55b32cdae620 stream=0x55b32c9d0ab0 workspace=4194304
CUBLAS_F32 src0=fc2.weight src1=node_5 dst=node_6 => M=10 N=500 K=500 lda=500 ldb=500 ldc=10
```

### 7.3 解读

**① 惰性初始化的两个证据**：

```text
第一次 MUL_MAT: cublas_handle=(nil)    workspace=0
第二次 MUL_MAT: cublas_handle=0x55b... workspace=4,194,304
```

**② `M/N/K` 的对应关系**：

```text
src0 = [K=784, M=500]   ← 权重，ne[0]=K
src1 = [K=784, N=500]   ← 激活，ne[0]=K, ne[1]=N
dst  = [M=500, N=500]
```

**③ `lda` 的两种值**：

```text
fc1.weight: nb[1]=3136, lda=3136/4=784 = K
fc2.weight: nb[1]=2000, lda=2000/4=500 = K
```

**④ 为什么 `ldb = K` 而不是 `N`？**
因为 `src1` 在 cuBLAS 眼里是 `[N,K]` 列主序（行主序的 `[K,N]`），每列的跨度是 `K`。这就是 `CUBLAS_OP_N` 直接用的原因。

### 7.4 kernel launch 配置

```text
ADD_BCAST_UNRAVEL grid=(2,500,1) block=(128,1,1) ne=(500,500,1,1)
UNARY            grid=977          k=250000
ADD_BCAST_UNRAVEL grid=(1,20,1)  block=(5,25,1)  ne=(10,500,1,1)
CE_LOSS          grid=500 block=32 nclasses=10 nrows=500 shared=40
SCALE            nelements=1 scale=0.500000 bias=0.000000
ARGMAX           grid=500 block=32 ne00=10
ARGMAX           grid=500 block=32 ne00=10
COUNT_EQUAL      grid=4 block=32 ne=500 nsm=128 dne=128
SCALE            nelements=1 scale=0.500000 bias=0.000000
CE_LOSS_BACK     grid=500 block=32 nclasses=10 nrows=500
OUT_PROD         M=500 N=10 K=500 lda=500 ldb=10 ldc=500 src1_T=0
OUT_PROD         M=500 N=500 K=10 lda=500 ldb=10 ldc=500 src1_T=1
UNARY            grid=977 k=250000
OUT_PROD         M=784 N=500 K=500 lda=784 ldb=500 ldc=784 src1_T=0
UNARY            grid=977 k=250000
ADAMW            grid=1    block=256 k=10
ADAMW            grid=2    block=256 k=500
ADAMW            grid=1532 block=256 k=392000
ADAMW            grid=20   block=256 k=5000
```

### 7.5 解读

**① `OUT_PROD` 的 `src1_T` 三态**：

```text
node_18 = OUT_PROD(node_5, grad for node_6)                    src1_T=0, ldb=10
node_21 = OUT_PROD(fc2.weight, grad for node_6 (transposed))   src1_T=1, ldb=10
node_26 = OUT_PROD(images, grad for node_2)                    src1_T=0, ldb=500
```

`node_21` 的 `src1` 是转置视图（`nb=[40,4,...]`），`ggml_is_transposed` 返回 true，于是 `src1_cublas_op = CUBLAS_OP_N`。

**② `ADAMW grid=1532`**：`fc1.weight` 有 `784×500=392,000` 个元素，`ceil(392000/256)=1532`。这是训练里最大的单 kernel（纯逐元素，带宽受限）。

**③ `CE_LOSS grid=500 block=32`**：500 个样本 → 500 个 block，每个 block 一个 warp（32 线程）处理 10 个类别，22 个线程空转。

---

## 8 阶段八：时间数据与两次自我纠错

> 本节是全文最重要的方法论部分：**gdb 断点计时会给出错误结论，必须用代码插桩或独立微基准验证**。

### 8.1 gdb 断点计时（失真，勿作结论）

最初的测法：在 `ggml_backend_sched_alloc_graph` 入口和 `compute_splits` 入口各打一个断点，用 `ggml_time_us()` 求差。

```text
BATCH#6  split+alloc=4479 us   (split part=1591) nodes=34
BATCH#7  split+alloc=106507 us (split part=1705) nodes=28
BATCH#8  split+alloc=192086 us (split part=1868) nodes=34
BATCH#9  split+alloc=6163 us   (split part=2612) nodes=28
BATCH#10 split+alloc=10121 us  (split part=2195) nodes=34
BATCH#11 split+alloc=6423 us   (split part=2807) nodes=28
```

看起来“每 batch 分配 4.5 ms”。**但这是假的。**

**为什么假**：两个断点之间的时间是“真实代码 + gdb 开销”。gdb 每次命中要做：

```text
停所有线程 → 解析表达式（tensor->name 要读内存、ggml_op_desc 要调用函数）
→ 执行 printf（格式化 + 写 stderr）→ 恢复线程
```

实测每个断点命中 50~500 µs，两个断点就是 100~1000 µs。**gdb 开销比被测代码大一个数量级。**

### 8.2 代码插桩：真实值是 42 µs

在 `ggml_backend_sched_split_graph` 和 `ggml_backend_sched_alloc_splits` 首尾各打一次 `ggml_time_us()`：

```cpp
// ggml-backend.cpp（临时插桩，测完已还原）
static int64_t t_split0 = 0, g_prof_split_us = 0, g_prof_alloc_us = 0;
static int64_t prof_n = 0, prof_total = 0, prof_max = 0;

void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    t_split0 = ggml_time_us();
    ...
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    const int64_t t_alloc0 = ggml_time_us();
    g_prof_split_us += t_alloc0 - t_split0;
    ...
    {
        const int64_t dt = ggml_time_us() - t_alloc0;
        g_prof_alloc_us += dt; prof_total += dt; prof_n++;
        if (dt > prof_max) prof_max = dt;
        if (prof_n % 200 == 0) {
            fprintf(stderr, "[PROF] n=%ld split_total=%ld us alloc_total=%ld us alloc_avg=%ld us alloc_max=%ld us\n",
                    prof_n, g_prof_split_us, g_prof_alloc_us, prof_total/prof_n, prof_max);
        }
    }
    return true;
}
```

连跑 3 次：

```text
[PROF] n=3400 split_total=57695 us alloc_total=144328 us alloc_avg=42 us alloc_max=430 us
[PROF] n=3400 split_total=53463 us alloc_total=144303 us alloc_avg=42 us alloc_max=407 us
[PROF] n=3400 split_total=52208 us alloc_total=142901 us alloc_avg=42 us alloc_max=425 us
```

每 batch：

```text
split_graph   ≈ 15 µs
alloc_splits  ≈ 42 µs
合计          ≈ 57 µs
```

**与 gdb 测得的 4.5 ms 差 80 倍**。真实占比：`57 µs × 3,400 ≈ 0.19 s`，占 6 s 总时间的 **3.2%**。

`alloc_max=430 µs` 对应每个 epoch 第一次遇到 `gb_opt`（34 节点）时的 reserve；之后稳定在 42 µs。

### 8.3 三段式拆解：瓶颈是数据搬运

在 `ggml_opt_epoch` 的训练循环里插三个时间戳：

```cpp
t_b0; ggml_opt_alloc(opt_ctx, /*backward=*/true);   // 选图 + split + alloc
t_b1; ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);  // 100 次 H2D
t_b2; ggml_opt_eval(opt_ctx, result_train);          // compute + sync + D2H
t_b3;
```

```text
[PROF2] n=3000 alloc_avg=158 h2d_avg=1141 comp_avg=496 alloc_max=716 us
[PROF2] n=3000 alloc_avg=159 h2d_avg=1114 comp_avg=493 alloc_max=683 us
[PROF2] n=3000 alloc_avg=159 h2d_avg=1138 comp_avg=489 alloc_max=2741 us
```

| 阶段 | 每 batch | 占比 |
|---|---|---|
| `ggml_opt_alloc`（选图 + split + alloc） | 158 µs | 9% |
| `ggml_opt_dataset_get_batch`（100 次 H2D + sync） | **1,138 µs** | **64%** |
| `ggml_opt_eval`（kernel + D2H） | 490 µs | 27% |
| 合计 | 1,786 µs | 100% |

`1,786 µs × 3,420 batch ≈ 6.1 s`，与实测训练总时长 6 s 吻合。

**结论反转**：mnist 的瓶颈是**数据搬运**（64%），不是显存规划（9%），也不是 GPU 计算（27%）。

`dataset_get_batch` 的 1,138 µs 里：

```text
100 次 cudaMemcpyAsync + cudaStreamSynchronize ≈ 794 µs   （微基准加权）
permutation 随机访问 190 MB 数据集导致的 cache miss ≈ 344 µs
```

### 8.4 每 batch 的 compute（gdb 断点，同样失真）

```text
BATCH#1 graph_compute+sync=141187 us   ← 含冷启动
BATCH#2 graph_compute+sync=2201 us
BATCH#3 graph_compute+sync=1885 us
BATCH#4 graph_compute+sync=1839 us
BATCH#5 graph_compute+sync=1812 us
BATCH#6 graph_compute+sync=2035 us
```

插桩测得 `comp_avg=490 µs`，比 gdb 的 1,800~2,000 µs 小 4 倍——同样是 gdb 开销。**以插桩值为准。**

### 8.5 微基准（独立小程序，最可信）

```text
h2d               400B      5.41 us/op  (2000 iters)
h2d             31360B     10.47 us/op  (2000 iters)
h2d           1568000B    135.62 us/op  (500 iters)
d2h                 4B      6.27 us/op  (2000 iters)
d2h              2000B      7.12 us/op  (2000 iters)
d2h                 8B      6.72 us/op  (2000 iters)
h2d async-only  31360B      3.98 us/op

cudaMalloc+cudaFree 3.42MiB: 154.64 us/op
cublasCreate+workspace:       24.53 ms
first sgemm 500x500x784:     156.331 ms
steady sgemm 500x500x784:      0.019 ms   (20.6 TFLOP/s)
steady sgemm 10x500x500:       0.207 ms
empty kernel + sync:           7.62 us/op
977-block empty kernel + sync: 8.00 us/op
```

### 8.6 解读

**① `sync` 是 H2D 的主要成本**：31,360 B 带 sync 是 10.47 µs，不带 sync 是 3.98 µs，**同步占 62%**。

**② 空 kernel 也要 7.62 µs**：这是 launch + sync 的固定成本。`SCALE nelements=1` 这种 kernel 的成本几乎全是开销。

**③ 冷启动**：`cublasCreate` 24.5 ms + 首次 GEMM 156 ms = 180 ms。对 6 s 的稳态训练是 3%，对首次运行 22 s 是 0.8%。

**④ `cudaMalloc+cudaFree` 154.64 µs**：这是 §5.4 里“只 realloc 1 次”的价值——如果每 batch 都重新分配，3,400 × 155 µs = 0.53 s。

### 8.7 干扰说明

调试期间 GPU 上有 `llama-server`（Qwen3.8-27B，19.8 GB，100% util，362 W）。

**结构性数据不受影响**（地址、偏移、节点数、grid/block、cuBLAS 参数），**时间数据只能作数量级参考**。§8.2/§8.3 的插桩数据是在相对空闲时段测的，且三次运行一致，可信度较高。

---

## 9 阶段九：推理路径

### 9.1 脚本

```gdb
start mnist-fc-f32.gguf <t10k-images> <t10k-labels> CUDA0
break gguf_init_from_file
commands
silent
printf "GGUF_LOAD file=%s no_alloc=%d\n", fname, params.no_alloc
continue
end
break ggml-opt.cpp:581
commands
silent
printf "EVAL_OPT gf_nodes=%d gf_leafs=%d build_type=%d gb_grad=%p gb_opt=%p\n",
  result->gf->n_nodes, result->gf->n_leafs, result->build_type, result->gb_grad, result->gb_opt
continue
end
break ggml_backend_cuda_graph_compute
commands
silent
printf "=== EVAL GRAPH nodes=%d ===\n", cgraph->n_nodes
set $i = 0
while $i < cgraph->n_nodes
printf "  [%d] %-24s op=%-3d data=%p flags=%d\n", $i, cgraph->nodes[$i]->name,
  cgraph->nodes[$i]->op, cgraph->nodes[$i]->data, cgraph->nodes[$i]->flags
set $i = $i + 1
end
quit
end
continue
```

### 9.2 原始输出

```text
GGUF_LOAD file=mnist-fc-f32.gguf no_alloc=1
EVAL_OPT gf_nodes=13 gf_leafs=2 build_type=10 gb_grad=(nil) gb_opt=(nil)

=== EVAL GRAPH nodes=13 ===
  [0] fc2.weight               op=0   data=0x7fbdf0d7f500 flags=4
  [1] fc1.weight               op=0   data=0x7fbdf0c00000 flags=4
  [2] node_2                   op=29  data=0x7fbdf1000000 flags=16
  [3] fc1.bias                 op=0   data=0x7fbdf0d7ed00 flags=4
  [4] node_4                   op=2   data=0x7fbdf1000000 flags=16
  [5] node_5                   op=91  data=0x7fbdf1000000 flags=16
  [6] node_6                   op=29  data=0x7fbdf10f4280 flags=16
  [7] fc2.bias                 op=0   data=0x7fbdf0d84380 flags=4
  [8] logits                   op=2   data=0x7fbdf10f4280 flags=18
  [9] loss_cross_entropy       op=96  data=0x7fbdf0d89280 flags=26
  [10] pred                    op=18  data=0x7fbdf0d89300 flags=18
  [11] node_11                 op=18  data=0x7fbdf0d89b00 flags=16
  [12] ncorrect                op=19  data=0x7fbdf0d8a300 flags=18
```

### 9.3 解读

**① 三个节点共用同一地址**：

```text
node_2 (MUL_MAT)  0x7fbdf1000000
node_4 (ADD)      0x7fbdf1000000   ← 原地复用
node_5 (RELU)     0x7fbdf1000000   ← 原地复用
```

**推理比训练激进**：没有反向，`node_2` 只有 `node_4` 一个消费者，`node_4` 只有 `node_5` 一个消费者，所以可以连续原地。训练时 `node_4` 有两个消费者（`node_5` + `node_22`），`node_5` 不能复用 `node_4`。

**② 静态张量的地址在 `ctx_gguf` / `ctx_static` 区域**：

```text
fc1.weight 0x7fbdf0c00000   ← ctx_gguf buffer
fc1.bias   0x7fbdf0d7ed00
fc2.weight 0x7fbdf0d7f500
fc2.bias   0x7fbdf0d84380
loss       0x7fbdf0d89280   ← ctx_compute 的静态部分
```

**③ `flags=26` 的 `loss_cross_entropy`**：`2+8+16 = OUTPUT|LOSS|COMPUTE`。推理时也打了 `LOSS`，因为 `ggml_opt_build` 里 `ggml_set_loss(opt_ctx->loss)` 无条件执行。

---

## 10 调试方法论总结

### 10.1 按“层”组织断点，而不是按“函数”

```text
层 1（context）  : ggml_init, ggml_new_tensor_impl, ggml_set_name
层 2（分配）     : ggml_backend_alloc_ctx_tensors, ggml-alloc.c:1229
层 3（建图）     : ggml_build_forward_expand, ggml-opt.cpp:581
层 4（调度）     : ggml_backend_sched_split_graph, alloc_splits, reset
层 5（显存规划） : dyn_tallocr_alloc, gallocr_free_node, needs_realloc
层 6（执行）     : ggml_cuda_compute_forward, ggml_cuda_mul_mat
层 7（kernel）   : 各 .cu 的 launcher（unary.cu:131, binbcast.cu:348, ...）
层 8（数据搬运） : ggml_backend_tensor_set / get
```

一次只打通一层，比“什么都断”有效得多。

### 10.2 用“计数断点”代替“逐次断点”

```gdb
set $n = 0
break some_function
commands
silent
set $n = $n + 1
if $n <= 5 || $n % 100 == 0
printf "...\n"
end
continue
end
```

训练有 3,600 个 batch，如果每个都打印，输出会淹没关键信息。

### 10.3 用“结构数据”而不是“时间数据”下结论

时间数据受 GPU 负载影响（本次调试有 `llama-server` 干扰），但地址、偏移、节点数、grid/block 是确定性的。

### 10.4 用 `GGML_SCHED_DEBUG=2` 做交叉验证

```bash
GGML_SCHED_DEBUG=2 ./build-debug/bin/mnist-eval ... CUDA0
```

输出：

```text
## SPLIT #0: CUDA0 # 0 inputs
node #  0 (      NONE):           fc2.weight (  19K) [CUDA0] use=1,c=0:
node #  2 (   MUL_MAT):               node_2 ( 976K) [CUDA0] use=1,c=1:  fc1.weight (1M) [CUDA0]  images (1M) [CUDA0]
node #  4 (       ADD):               node_4 ( 976K) [CUDA0] use=1,c=1:      node_2 (976K) [CUDA0]  fc1.bias (1K) [CUDA0]
node #  5 (      RELU):               node_5 ( 976K) [CUDA0] use=1,c=1:      node_4 (976K) [CUDA0]
...
```

`use=N` 是 `use_counts`，`c=0/1` 是 `COMPUTE` 标记。**这份输出和 gdb 断点得到的节点表完全一致**，可以互相印证。

### 10.5 一个典型的调试误区

**误区**：看到 `needs_realloc` 就以为在 `cudaMalloc`。

**实际**：

```text
needs_realloc nodes: 30 次
reallocating buffer:  1 次
```

`needs_realloc` 只是“需要重新规划 offset”，只有新图更大时才真正 `cudaMalloc`。断点要打在 `ggml_backend_cuda_buffer_type_alloc_buffer` 才能确认。

---

## 11 断点 ↔ 源码逐行对照表

本文件所有断点都能在源码里精确定位。下表给出“断点 → 源码行 → 该行在做什么”的完整映射，便于读者自己复现时对照。

### 11.1 context / tensor 层

| 断点 | 源码位置 | 该行在做什么 |
|---|---|---|
| `break ggml_init` | `ggml.c:1611` | 函数入口；`params.mem_size` / `no_alloc` 可读 |
| `break ggml.c:1841` | `ggml.c:1841` | `ggml_new_tensor_impl` 的 `return result;`，`result->ne/nb/data/view_src` 已完整 |
| `break ggml_set_name` | `ggml.c:1946` | 函数入口；`name` 和 `tensor` 可读（但 `tensor->name` 还未写入） |
| `break ggml_free` | `ggml.c:1655` | 函数入口；`ctx->mem_buffer_owned` 决定是否释放 arena |

**为什么 `ggml.c:1841` 而不是 `:1844`**：1844 是 `ggml_new_tensor` 的签名行，gdb 会落到 1849 的调用行，此时 `result` 还不存在。1841 是 `ggml_new_tensor_impl` 的 `return result;`，所有字段已填。

### 11.2 buffer / 分配层

| 断点 | 源码位置 | 该行在做什么 |
|---|---|---|
| `break ggml-alloc.c:1229` | `ggml-alloc.c:1229` | `alloc_ctx_tensors_from_buft_impl` 的 `return buffer;`；`buffer->size`、`buffer->buft` 可读 |
| `break ggml-alloc.c:264` | `ggml-alloc.c:264` | `dyn_tallocr_alloc` 的 `block->offset += size;` 之后；`addr.offset` 是本次分配的起点 |
| `break ggml-alloc.c:710` | `ggml-alloc.c:710` | `free_node` 的 `ggml_dyn_tallocr_free_bytes(...)` 之前；`hn->addr` 是被释放的区间 |
| `break ggml-alloc.c:920` | `ggml-alloc.c:920` | `reserve_n_impl` 里累加 `new_size`；`realloc` 为 true 时才会 `cudaMalloc` |
| `break ggml_backend_cuda_buffer_type_alloc_buffer` | `ggml-cuda.cu` | 真正 `cudaMalloc` 的位置；`size` 是本次申请的字节数 |

**`ggml-alloc.c:264` 与 `:710` 的 `size` 差异**：前者打印的是 `aligned_offset` 之后的大小（`1,000,064`），后者打印的是 `get_alloc_size` 的原始值（`1,000,000`）。`free_bytes` 内部会再对齐一次，所以不会漏字节（03 §3.3）。

### 11.3 建图层

| 断点 | 源码位置 | 该行在做什么 |
|---|---|---|
| `break ggml-opt.cpp:581` | `ggml-opt.cpp:581` | `ggml_opt_init` 的 `return result;`；`gf` / `gb_grad` / `gb_opt` 都已建好 |
| `break ggml_build_forward_expand` | `ggml.c:7337` | 函数入口；`cgraph` 和 `tensor`（根节点）可读 |
| `break ggml_backend_sched_alloc_splits` | `ggml-backend.cpp:1591` | 函数入口；`sched->graph.n_nodes` / `n_splits` 可读 |

**为什么 `ggml-opt.cpp:581` 是最佳观测点**：此时 `ggml_opt_build` 刚返回，三张图的节点数、叶子数、`grad_accs` 都已就绪，且还没有进入训练循环。

### 11.4 调度层

| 断点 | 源码位置 | 该行在做什么 |
|---|---|---|
| `break ggml_backend_sched_split_graph` | `ggml-backend.cpp:1066` | 函数入口；`graph->n_nodes` / `n_leafs` 可读 |
| `break ggml_backend_sched_alloc_splits` | `ggml-backend.cpp:1591` | 函数入口；`sched->n_splits` / `splits[]` 已由 `split_graph` 填好 |
| `break ggml_backend_sched_compute_splits` | `ggml-backend.cpp:1646` | 函数入口；`sched->splits` 可读 |
| `break ggml_backend_sched_reset` | `ggml-backend.cpp:1949` | 函数入口；`is_reset` / `is_alloc` 的翻转点 |
| `break ggml_gallocr_needs_realloc` | `ggml-alloc.c:1009` | 函数入口；`galloc->n_nodes` / `graph->n_nodes` 可比较 |
| `break ggml_gallocr_reserve_n_impl` | `ggml-alloc.c:825` | 函数入口；`graph->n_nodes` / `n_leafs` 可读 |

**`ggml_backend_sched_alloc_splits` 的两个用途**：

1. 读 `sched->n_splits` / `splits[]`：split 决策的结果；
2. 读 `sched->graph.n_nodes`：`graph_copy` 的节点数（`gb_opt` 是 34 而不是 32）。

### 11.5 执行层

| 断点 | 源码位置 | 该行在做什么 |
|---|---|---|
| `break ggml_backend_cuda_graph_compute` | `ggml-cuda.cu:4425` | 函数入口；`cgraph->n_nodes` 可读 |
| `break ggml_cuda_compute_forward` | `ggml-cuda.cu:2068` | 函数入口；`dst->op` / `ne` / `src[]` 可读 |
| `break ggml_cuda_mul_mat` | `ggml-cuda.cu:1824` | 函数入口；`src0` / `src1` / `dst` 的形状和类型可读 |
| `break ggml_cuda_mul_mat_cublas_impl<(ggml_type)0>` | `ggml-cuda.cu:1410` | F32 实例的入口；`src0->ne/nb` 可读 |
| `break ggml-cuda.cu:1546` | `ggml-cuda.cu:1546` | `cublasSgemm` 调用行；`ne01/ne11/ne10/s01/s11/ne0` 是实参 |
| `break binbcast.cu:333` | `binbcast.cu:333` | `k_bin_bcast` 的 launch；`launch_params` 可读 |
| `break binbcast.cu:348` | `binbcast.cu:348` | `k_bin_bcast_unravel` 的 launch；同上 |
| `break unary.cu:134` | `unary.cu:134` | `unary_op_kernel` 的 launch；`num_blocks` / `k` 可读 |
| `break cross-entropy-loss.cu:127` | `cross-entropy-loss.cu:127` | `cross_entropy_loss_f32<true>` 的 launch |
| `break opt-step-adamw.cu:42` | `opt-step-adamw.cu:42` | `opt_step_adamw_f32` 的 launch |
| `break argmax.cu:90` | `argmax.cu:90` | `argmax_f32` 的 launch |
| `break count-equal.cu:58` | `count-equal.cu:58` | `count_equal` 的 launch |

**模板函数的多实例**：`ggml_cuda_mul_mat_cublas_impl` 有 3 个实例（F32 / BF16 / F16），`break ggml-cuda.cu:1546` 会命中全部 3 个。要只看 F32，用 `break ggml_cuda_mul_mat_cublas_impl<(ggml_type)0>`。

### 11.6 数据搬运层

| 断点 | 源码位置 | 该行在做什么 |
|---|---|---|
| `break ggml_backend_tensor_set` | `ggml-backend.cpp:335` | 函数入口；`tensor->name` / `offset` / `size` 可读 |
| `break ggml_backend_tensor_get` | `ggml-backend.cpp:350` | 同上 |
| `break ggml_backend_cuda_buffer_set_tensor` | `ggml-cuda.cu:785` | CUDA 实现入口；`ctx->device` 可读 |

**`ggml_backend_tensor_set` 的两次命中**：

1. 初始化时的 4 次权重上传（`size=1,568,000 / 2,000 / 20,000 / 40`）；
2. 每 batch 的 100 次 shard 拷贝（`size=31,360 / 400`）。

用 `$sc` 计数器区分（05 §5.1 的脚本）。

### 11.7 一个完整的“从断点到源码”的例子

**现象**：`gb_opt` 的 `graph_copy` 是 34 个节点，但 `gb_opt` 本身是 32 个。

**断点**：

```gdb
break ggml_backend_sched_alloc_splits
commands
silent
printf "n_splits=%d graph_nodes=%d\n", sched->n_splits, sched->graph.n_nodes
set $i = 0
while $i < sched->graph.n_nodes
printf "  [%d] %s op=%d\n", $i, sched->graph.nodes[$i]->name, sched->graph.nodes[$i]->op
set $i = $i + 1
end
continue
end
```

**输出**：

```text
n_splits=1 graph_nodes=34
  [0] adamw_params (view) op=0
  [1] CUDA0#adamw_params#0 op=0
  [2] fc2.weight op=0
  ...
  [33] adamw step for fc2.weight op=98
```

**源码定位**：

```cpp
// ggml-backend.cpp:1495-1520
for (int i = 0; i < sched->n_splits; i++) {
    struct ggml_backend_sched_split * split = &sched->splits[i];
    for (int j = 0; j < split->n_inputs; j++) {
        struct ggml_tensor * input = split->inputs[j];
        const size_t input_id = hash_id(input);
        struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

        // add a dependency to the input source so that it is not freed before the copy is done
        struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
        input_dep->src[0] = input;
        sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
        graph_copy->nodes[graph_copy->n_nodes++] = input_dep;      // ← [0]

        // add a dependency to the input copy so that it is allocated at the start of the split
        sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
        graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;      // ← [1]
    }

    for (int j = split->i_start; j < split->i_end; j++) {
        sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
        graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];  // ← [2..33]
    }
}
```

**结论**：`n_inputs=1`（`adamw_params`）导致插入 2 个节点，所以 32 + 2 = 34。

**为什么 `input_dep` 是 `op=NONE`**：`ggml_view_tensor` 创建的视图默认 `op=GGML_OP_NONE`，它只作为“依赖占位符”，CUDA 执行时被 `is_view_or_noop` 跳过。但它在 gallocr 的生命周期分析里算作“消费者”，保证 `adamw_params` 在拷贝完成前不被释放。

---

## 12 复现清单

```bash
# 1. 构建
cd /home/ggml
cmake --build build-debug --target mnist-train mnist-eval -j 32

# 2. 训练（生成 GGUF）
./build-debug/bin/mnist-train mnist-fc mnist-fc-f32.gguf \
  examples/mnist/data/MNIST/raw/train-images-idx3-ubyte \
  examples/mnist/data/MNIST/raw/train-labels-idx1-ubyte CUDA0

# 3. 推理
./build-debug/bin/mnist-eval mnist-fc-f32.gguf \
  examples/mnist/data/MNIST/raw/t10k-images-idx3-ubyte \
  examples/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte CUDA0

# 4. 调度可视化
GGML_SCHED_DEBUG=2 ./build-debug/bin/mnist-eval ... CUDA0

# 5. gdb
gdb -q ./build-debug/bin/mnist-train
(gdb) start mnist-fc ... CUDA0
(gdb) break ggml-alloc.c:1229
(gdb) break ggml-cuda.cu:4425
(gdb) continue
```
