# 01 · 初始化与内存账本：从 5 个 context 到 5 块 buffer

> 对应源码：`examples/mnist/mnist-train.cpp:14-39`、`examples/mnist/mnist-common.h:65-155`、
> `examples/mnist/mnist-common.cpp:18-130, 241-310, 131-239`
>
> 对应 ggml 源码：`src/ggml.c:1611`（`ggml_init`）、`src/ggml.c:1766`（`ggml_new_tensor_impl`）、
> `src/ggml-alloc.c:1168/1239/1247`（`alloc_ctx_tensors`）、`src/ggml-backend.cpp:1848`（`sched_new`）、
> `src/ggml-cuda/ggml-cuda.cu:5842`（`ggml_backend_cuda_init`）
>
> 本文所有数值来自 gdb 断点实测。

---

## 0 业务侧：这一步在训练流程里的位置

```cpp
// mnist-train.cpp
int main(int argc, char ** argv) {
    // ① 数据集：60000 张 28×28 图 + one-hot 标签，放在 CPU 内存
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTRAIN, /*ndata_shard=*/ 10);

    mnist_image_load(argv[3], dataset);   // 47 MB idx 文件 → dataset->data
    mnist_label_load(argv[4], dataset);   // 60 KB idx 文件 → dataset->labels

    // ② 模型：3 个 backend + 3 个 context + 静态 buffer
    mnist_model model = mnist_model_init_random(argv[1], "CUDA0", MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);

    // ③ 构图（只建图，不计算）
    mnist_model_build(model);

    // ④ 训练：30 epoch，5% 验证集
    mnist_model_train(model, dataset, 30, 0.05f);

    // ⑤ 保存
    mnist_model_save(model, argv[2]);
}
```

三个宏决定了整个内存账本的形状：

```cpp
#define MNIST_NBATCH_LOGICAL  1000   // 一次梯度更新用 1000 张图
#define MNIST_NBATCH_PHYSICAL  500   // 一次前向/反向并行 500 张图
#define MNIST_NHIDDEN          500
```

**为什么逻辑 batch 是物理 batch 的 2 倍？** 这是**梯度累加**（gradient accumulation）：

```text
opt_period = nbatch_logical / nbatch_physical = 2
物理 batch #1 → 前向 + 反向，梯度累加到 grad_accs（不更新参数）
物理 batch #2 → 前向 + 反向，梯度继续累加，然后执行 AdamW step
```

在 ggml 里它体现为两种图交替执行：`gb_grad`（28 节点，前向+反向）与 `gb_opt`（32 节点，前向+反向+AdamW）。
这个交替是后面 §3.5「每 batch 重新 reserve」现象的直接原因。

**为什么 shard 是 10？** 源码注释写得很清楚：

```cpp
// The MNIST model is so small that the overhead from data shuffling is non-negligible, especially with CUDA.
// With a shard size of 10 this overhead is greatly reduced at the cost of less shuffling.
```

数据在 CPU 上，每个物理 batch 要拷 500 张图到 GPU。如果逐张拷，就是 500 次 `cudaMemcpyAsync` + 500 次 `cudaStreamSynchronize`；按 10 张一组（一个 shard = 10×784×4 = 31,360 B），只需 50 次。实测单次 31,360 B 的 H2D+sync 是 10.47 µs，400 B 是 5.41 µs（见 04 §6.3），所以 shard 化把每 batch 的拷贝次数从 500 降到 50。实测 `dataset_get_batch` 平均 1,138 µs/batch（含 cache miss），仍是训练的最大瓶颈（见 05 §8.3）。

---

## 1 第一层：dataset 的 context 与 CPU buffer

### 1.1 断点

```gdb
break ggml_init
commands
silent
printf "params.mem_size=%zu no_alloc=%d mem_buffer=%p\n", params.mem_size, params.no_alloc, params.mem_buffer
bt 3
continue
end
```

### 1.2 原始输出

```text
=== ggml_init hit ===
params.mem_size=736 no_alloc=1 mem_buffer=(nil)
#0  ggml_init (params=...) at /home/ggml/src/ggml.c:1614
#1  0x00007fb30d9b7fdd in ggml_opt_dataset_init (...) at /home/ggml/src/ggml-opt.cpp:108
#2  0x000056170542619e in main (...) at /home/ggml/examples/mnist/mnist-train.cpp:23
```

### 1.3 为什么是 736？

```cpp
// ggml-opt.cpp:102
struct ggml_init_params params = {
    /*.mem_size   =*/ 2*ggml_tensor_overhead(),   // 2 × 368 = 736
    /*.mem_buffer =*/ nullptr,
    /*.no_alloc   =*/ true,
};
result->ctx = ggml_init(params);
result->data   = ggml_new_tensor_2d(result->ctx, type_data,  ne_datapoint, ndata); // [784, 60000]
result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label,     ndata); // [10, 60000]
```

这个 context 里**只放两个张量的元数据**，正好 736 B。`no_alloc=true` 意味着张量 `data=NULL`，真正的 190 MB 在下一行分配：

```cpp
result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());
```

### 1.4 断点：静态 buffer 分配

```gdb
break ggml-alloc.c:1229        # alloc_ctx_tensors_from_buft_impl 的 return buffer
commands
silent
printf "CTX_BUFFER buft=%-6s size=%9zu\n", ggml_backend_buft_name(buffer->buft), buffer->size
continue
end
```

```text
CTX_BUFFER buft=CPU    size=190560000
```

**190,560,000 的推导**：

```text
images: 784 × 60000 × 4 B = 188,160,000
labels:  10 × 60000 × 4 B =   2,400,000
                          = 190,560,000   （正好，无 padding）
```

CPU buffer 的 alignment 是 64，两个张量大小都是 64 的倍数，所以没有补齐。

### 1.5 为什么数据集放在 CPU 而不是显存？

三个原因：

1. **容量**：60,000 张图 190 MB，RTX 4090 放得下，但真实数据集（ImageNet 140 GB）放不下。
2. **随机访问**：`ggml_opt_dataset_shuffle` 只打乱 `permutation` 索引，`get_batch` 按 shard 做**非连续**读取。GPU 侧只保留当前 500 张（1.57 MB），是“流式”而不是“全量常驻”。
3. **CPU buffer 是 host 内存**：`ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type())` 分配的是 `malloc` 出来的普通内存，`dataset->data->data` 可以直接被 `memcpy` / `ifstream` 写。CUDA buffer 的 `data` 是设备指针，host 不能直接解引用。

断点证据：`mnist_image_load` 里

```cpp
float * buf = ggml_get_data_f32(images);   // CPU buffer → 合法 host 指针
buf[iex*MNIST_NINPUT + i] = image[i] / 255.0f;
```

---

### 1.6 源码逐行：`ggml_init`（`src/ggml.c:1611-1651`）

```cpp
1611  struct ggml_context * ggml_init(struct ggml_init_params params) {
1612      static bool is_first_call = true;
1613
1614      ggml_critical_section_start();
1615
1616      if (is_first_call) {
1617          // initialize time system (required on Windows)
1618          ggml_time_init();
1619
1620          is_first_call = false;
1621      }
1622
1623      ggml_critical_section_end();
1624
1625      struct ggml_context * ctx = GGML_MALLOC(sizeof(struct ggml_context));
1626
1627      // allow to call ggml_init with 0 size
1628      if (params.mem_size == 0) {
1629          params.mem_size = GGML_MEM_ALIGN;
1630      }
1631
1632      const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);
1633
1634      *ctx = (struct ggml_context) {
1635          /*.mem_size           =*/ mem_size,
1636          /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
1637          /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
1638          /*.no_alloc           =*/ params.no_alloc,
1639          /*.n_objects          =*/ 0,
1640          /*.objects_begin      =*/ NULL,
1641          /*.objects_end        =*/ NULL,
1642      };
1643
1644      GGML_ASSERT(ctx->mem_buffer != NULL);
1645      GGML_ASSERT_ALIGNED(ctx->mem_buffer);
1646
1647      GGML_PRINT_DEBUG("%s: context initialized\n", __func__);
1648
1649      return ctx;
1650  }
```

逐行说明：

| 行 | 行为 | 为什么 |
|---|---|---|
| 1612 | 函数内 `static` 变量 | 全进程只做一次时间系统初始化；线程安全由 1614 的临界区保证 |
| 1614-1623 | 临界区包住 `is_first_call` | `ggml_init` 可能被多个线程同时调用（多 backend 注册） |
| 1625 | `GGML_MALLOC(sizeof(struct ggml_context))` | **只分配 context 结构体本身**（约 56 B），不是 `mem_size` 字节 |
| 1628-1630 | `mem_size == 0` → 16 | 允许“空 context”调用，仅为触发 `ggml_cpu_init()` 的副作用（断点里看到的 `mem_size=0` 就是这条路径） |
| 1632 | `GGML_PAD(mem_size, 16)` | 只有**自己分配**时才向上取整；外部传入的 `mem_buffer` 完全信任调用方 |
| 1636 | `mem_buffer ? mem_buffer : ggml_aligned_malloc` | 支持“用户提供 arena”（如 `simple-backend` 的 `model.buf`）和“ggml 自管”两条路径 |
| 1637 | `mem_buffer_owned` | 决定 `ggml_free` 是否 `ggml_aligned_free`；外部 buffer 不释放 |
| 1639-1641 | `n_objects=0`、`objects_begin/end=NULL` | 对象链表为空，第一个对象在首次 `ggml_new_object` 时插入 |
| 1645 | `GGML_ASSERT_ALIGNED` | 整个 arena 的基址必须 16 B 对齐，否则后续所有对象的 `offs` 都会错位 |

**关键结论**：`ggml_init` 是 **O(1)** 的，无论 `mem_size` 是 736 B 还是 1 MB，都只做一次 `malloc` + 一次结构体赋值。这也解释了为什么 `ctx_compute` 的 1,001,488 B 在 `ggml_init` 返回时就已经从操作系统拿到了——它是一整块 arena，不是按需增长。

**`ggml_free` 的对称性**（`src/ggml.c:1663-1674`）：

```cpp
1663  void ggml_free(struct ggml_context * ctx) {
1664      if (ctx == NULL) {
1665          return;
1666      }
1667
1668      if (ctx->mem_buffer_owned) {
1669          ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
1670      }
1671
1672      GGML_FREE(ctx);
1673  }
```

只释放 `mem_buffer`（如果自己拥有）和 `ctx` 本身。**它不释放任何 `ggml_backend_buffer`**——那是 `ggml_backend_buffer_free` 的职责。这就是为什么 `mnist_model` 的析构函数要分别调用 `ggml_free(ctx_*)` 和 `ggml_backend_buffer_free(buf_*)`，顺序不能反（先释放 buffer 再释放 ctx 也可以，但反过来会让 buffer 的 `context` 里保存的指针悬空）。

### 1.7 源码逐行：`ggml_new_object`（`src/ggml.c:1708-1764`）

这是整个 context arena 的核心：**顺序追加 + 16 B 对齐**。

```cpp
1708  static struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size) {
1709      // always insert objects at the end of the context's memory pool
1710      struct ggml_object * obj_cur = ctx->objects_end;
1711
1712      const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
1713      const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
1714      const size_t cur_end  = cur_offs + cur_size;
1715
1716      // align to GGML_MEM_ALIGN
1717      GGML_ASSERT(size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
1718      size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);
1719
1720      char * const mem_buffer = ctx->mem_buffer;
1721      struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);
1722
1723      // integer overflow checks
1724      if (cur_end > SIZE_MAX - size_needed) { ... return NULL; }
1728      if (cur_end + size_needed > SIZE_MAX - GGML_OBJECT_SIZE) { ... return NULL; }
1734      if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
1735          GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", ...);
1737  #ifndef NDEBUG
1738          GGML_ABORT("not enough space in the context's memory pool");
1739  #endif
1740          return NULL;
1741      }
1742
1743      *obj_new = (struct ggml_object) {
1744          .offs = cur_end + GGML_OBJECT_SIZE,
1745          .size = size_needed,
1746          .next = NULL,
1747          .type = type,
1748      };
1749
1750      GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);
1751
1752      if (obj_cur != NULL) {
1753          obj_cur->next = obj_new;
1754      } else {
1755          // this is the first object in this context
1756          ctx->objects_begin = obj_new;
1757      }
1758
1759      ctx->objects_end = obj_new;
1760
1761      return obj_new;
1762  }
```

逐行说明：

| 行 | 行为 | 为什么 |
|---|---|---|
| 1710-1714 | 从 `objects_end` 算出当前 arena 的尾部 | **单向链表 + 尾指针**，追加 O(1)；遍历从头（`objects_begin`）走 |
| 1718 | `GGML_PAD(size, 16)` | 每个对象的**载荷**按 16 B 对齐；`ggml_object` 本身 32 B 也是 16 的倍数，所以下一个对象也自然对齐 |
| 1721 | `obj_new = mem_buffer + cur_end` | **对象头放在当前尾部**，紧跟着才是载荷 |
| 1744 | `.offs = cur_end + GGML_OBJECT_SIZE` | 载荷偏移 = 对象头之后 32 B。注意 `offs` 是**相对 arena 基址**，不是相对对象头 |
| 1745 | `.size = size_needed` | 只记载荷大小（已对齐），不含对象头 |
| 1750 | 断言载荷地址 16 B 对齐 | 这是 `ggml_tensor` 能被安全 `memcpy`/SIMD 访问的前提 |
| 1734-1741 | 越界检查 | Debug 构建直接 `GGML_ABORT`；Release 返回 NULL（调用方负责检查） |

**断点可验证的量**：`ctx_static` 里 5 个张量，每个占 `32 + 336 = 368 B`，5 个共 1,840 B。但注意 `GGML_TENSOR_SIZE = 336` 已经是 16 的倍数，所以 `size_needed == 336`，没有 padding。如果某个对象大小不是 16 的倍数（比如 `ggml_new_buffer` 的任意字节数），才会有 padding。

**为什么用链表而不是数组**：因为 context 的生命周期内对象只增不减（`ggml_reset` 会把链表头清空但不释放内存），数组需要扩容/搬迁，而 `ggml_tensor*` 指针会被图、scheduler、backend 长期持有，**地址绝不能变**。

---

## 2 第二层：mnist_model 的 3 个 backend 与 3 个 context

### 2.1 构造函数源码

```cpp
mnist_model::mnist_model(const std::string & backend_name, int nbatch_logical, int nbatch_physical)
    : nbatch_logical(nbatch_logical), nbatch_physical(nbatch_physical) {

    // ① 主后端
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    backends.push_back(backend);
    devices.push_back(dev);

    // ② 所有可用后端作为 fallback（注意：会再创建一个 CUDA0）
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        backends.push_back(backend);
        devices.push_back(dev);
    }

    // ③ 调度器
    backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(),
                                           GGML_DEFAULT_GRAPH_SIZE, /*parallel=*/false, /*op_offload=*/true);

    // ④ 静态 context（权重 + images）
    const size_t size_meta = 1024*ggml_tensor_overhead();
    ctx_static = ggml_init({size_meta, nullptr, /*no_alloc=*/true});

    // ⑤ 计算 context（3 张图）
    const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
    ctx_compute = ggml_init({size_meta, nullptr, /*no_alloc=*/true});
}
```

### 2.2 断点：三个 backend 的真实身份

```gdb
break ggml_backend_sched_alloc_splits
commands
silent
printf "N_BACKENDS=%d names=[%s][%s][%s]\n", sched->n_backends,
  ggml_backend_name(sched->backends[0]), ggml_backend_name(sched->backends[1]), ggml_backend_name(sched->backends[2])
continue
end
```

```text
N_BACKENDS=3 names=[CUDA0][CUDA0][CPU]
```

**为什么 CUDA0 出现两次？** 源码注释：

```cpp
// A "backend" is a stream on a physical device so there is no problem with adding
// multiple backends for the same device.
```

- `backends[0]`：主后端，权重、输入、所有计算节点都在这里。
- `backends[1]`：同设备的第二个执行实例（第二个 `ggml_backend_cuda_context`，独立的 stream / cuBLAS handle / pool），作为 fallback。
- `backends[2]`：CPU，`ggml_backend_sched_new` 强制要求最后一个必须是 CPU：

```cpp
GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);
```

断点证据：`ggml_backend_cuda_init` 被命中 **2 次**（`device=0`），对应两个 CUDA 执行实例：

```text
CUDA_BACKEND_INIT device=0
CUDA_BACKEND_INIT device=0
```

**这个设计对 mnist 有副作用**：gallocr 按 **buffer type** 建分配器，两个 CUDA0 backend 共享同一个 `CUDA0 buft`，所以 `n_buffers=3` 但只有 **2 个独立的 tallocr**：

```text
buf[0] buft=CUDA0 vbuf=0x55f377815260 chunk0=0x55f377815ef0 size0=3588224 usage=2
buf[1] buft=CUDA0 vbuf=0x55f377815260 chunk0=0x55f377815ef0 size0=3588224 usage=2   ← 同一个 vbuffer
buf[2] buft=CPU   vbuf=0x55f377815f70 chunk0=(nil)               size0=0
```

`buf[0]` 与 `buf[1]` 的 `vbuf` 指针完全相同，这是 `ggml_gallocr_new_n` 的去重逻辑：

```cpp
for (int j = 0; j < i; j++) {
    if (bufts[i] == bufts[j]) {
        galloc->buf_tallocs[i] = galloc->buf_tallocs[j];   // 共用同一个分配器
        break;
    }
}
```

### 2.3 断点：两个 context 的 mem_size

```text
=== ggml_init hit ===
params.mem_size=376832 no_alloc=1
#1  mnist_model::mnist_model (...) at mnist-common.h:128

=== ggml_init hit ===
params.mem_size=1001488 no_alloc=1
#1  mnist_model::mnist_model (...) at mnist-common.h:139
```

**376,832 = 1024 × 368**：`ctx_static` 最多容纳 1024 个张量元数据。本例只用了 5 个（4 权重 + images），但留足余量，因为 `mnist_model_init_from_file` 的 CNN 分支需要 6 个权重 + images。

**1,001,488 = 2048 × 368 + 3 × 82,608**：

```text
2048 × 368  = 753,664   → 最多 2048 个张量元数据
3 × 82,608  = 247,824   → 3 个 cgraph（gf / gb_grad / gb_opt）
            = 1,001,488
```

**为什么恰好 3 张图？** 源码注释：

```cpp
// The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
```

`ggml_opt_build` 的流程：

```text
gf      = ggml_new_graph_custom(ctx, 2048, grads=true);  ggml_build_forward_expand(gf, outputs);
gb_grad = ggml_graph_dup(ctx, gf, force_grads=true);     ggml_build_backward_expand(...);
gb_opt  = ggml_graph_dup(ctx, gb_grad, force_grads=true); + 4 个 AdamW 节点
```

三张图都在 `ctx_compute` 的 arena 里，所以 `mem_size` 必须预先把三份 graph 对象算进去。

### 2.4 断点：context 对象池的内部布局

`ggml_context` 不是显存分配器，而是一个**线性对象池**：

```cpp
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    int    n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

每次 `ggml_new_tensor_impl` 调用 `ggml_new_object`，从池尾追加一个对象：

```cpp
size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);          // 16 字节对齐
struct ggml_object * obj_new = (struct ggml_object *)(mem_buffer + cur_end);
obj_new->offs = cur_end + GGML_OBJECT_SIZE;                    // 32
obj_new->size = size_needed;
```

所以 `ctx_static` 里 5 个张量占用的空间是：

```text
5 × (32 + 336) = 5 × 368 = 1,840 B
```

而 `mem_size` 是 376,832 B —— 这说明 **context 的 mem_size 是“容量”，不是“用量”**。`ggml_used_mem(ctx)` 才返回实际用量。

---

## 3 第三层：张量布局与 nb 的自动推导

### 3.1 断点：每次张量创建后的 ne/nb

```gdb
start mnist-fc ... CUDA0
break ggml.c:1841              # ggml_new_tensor_impl 的 return result;
commands
silent
printf "TENSOR type=%d ne=[%ld,%ld,%ld,%ld] nb=[%ld,%ld,%ld,%ld] data=%p view_src=%p\n",
  result->type, result->ne[0], result->ne[1], result->ne[2], result->ne[3],
  result->nb[0], result->nb[1], result->nb[2], result->nb[3], result->data, result->view_src
continue
end
```

```text
TENSOR#1 type=0 ne=[784,60000,1,1] nb=[4,3136,188160000,188160000] data=nil view_src=nil   ← dataset images
TENSOR#2 type=0 ne=[10,60000,1,1]  nb=[4,40,2400000,2400000]       data=nil view_src=nil   ← dataset labels

TENSOR#3 type=0 ne=[784,500,1,1] nb=[4,3136,1568000,1568000] data=nil   ← fc1.weight
TENSOR#4 type=0 ne=[500,1,1,1]   nb=[4,2000,2000,2000]       data=nil   ← fc1.bias
TENSOR#5 type=0 ne=[500,10,1,1]  nb=[4,2000,20000,20000]     data=nil   ← fc2.weight
TENSOR#6 type=0 ne=[10,1,1,1]    nb=[4,40,40,40]             data=nil   ← fc2.bias
TENSOR#7 type=0 ne=[784,500,1,1] nb=[4,3136,1568000,1568000] data=nil   ← images
```

### 3.2 nb 的推导代码

```cpp
// ggml.c:1832
result->nb[0] = ggml_type_size(type);                                 // F32: 4
result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));   // F32 blck=1
for (int i = 2; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
}
```

逐项验算：

| 张量 | ne | nb[0] | nb[1] | nb[2] | nb[3] |
|---|---|---|---|---|---|
| `fc1.weight` | `[784,500]` | 4 | 4×784=3,136 | 3,136×500=1,568,000 | 同 nb[2]（ne[2]=1） |
| `fc2.weight` | `[500,10]` | 4 | 4×500=2,000 | 2,000×10=20,000 | 同 nb[2] |
| `logits` | `[10,500]` | 4 | 4×10=40 | 40×500=20,000 | 同 nb[2] |
| `ncorrect` (I64) | `[1,1]` | 8 | 8 | 8 | 8 |

**为什么 `nb[2] == nb[3]`？** 因为 `ne[2]=1`，`nb[3] = nb[2] × ne[2] = nb[2]`。这是 ggml 的“4 维统一表示”：1D/2D 张量也按 4 维存，高维 `ne=1` 时 stride 退化。

**为什么 `fc1.weight` 是 `[784,500]` 而不是 `[500,784]`？**
ggml 的 `ne[0]` 是**最内层维度**，与 PyTorch 的 `[out, in]` 相反。`ggml_mul_mat(a, b)` 计算的是 `aᵀ · b`：

```text
a: [K, M]  （K=784 输入维，M=500 输出维）
b: [K, N]  （N=500 batch）
→ dst: [M, N] = [500, 500]
```

所以 `fc1.weight` 的 `ne=[784,500]` 对应 PyTorch 的 `Linear(784, 500).weight` 的 `[500,784]` 转置存储。这也是后面 cuBLAS 调用 `CUBLAS_OP_T` 的原因（见 04 §4.4）。

### 3.3 业务视角：这些张量分别代表什么

| 张量 | 数学形状 | 业务含义 |
|---|---|---|
| `fc1.weight` | `[784,500]` | 第一层全连接权重，784 像素 → 500 隐层 |
| `fc1.bias` | `[500,1]` | 第一层偏置，广播加到 500 个隐单元 |
| `fc2.weight` | `[500,10]` | 第二层权重，500 隐层 → 10 类 |
| `fc2.bias` | `[10,1]` | 第二层偏置 |
| `images` | `[784,500]` | 一个物理 batch 的 500 张 28×28 图（已归一化到 [0,1]） |
| `labels` | `[10,500]` | 500 个 one-hot 标签 |
| `logits` | `[10,500]` | 500 个样本在 10 类上的原始打分 |

注意 `bias` 的 `ne[1]=1`：它不是 `[500]` 向量，而是 `[500,1]` 矩阵。这样 `ggml_add` 的广播规则可以统一处理：`[500,500] + [500,1]` 沿 `ne[1]` 广播 500 次。

---

### 3.4 源码逐行：`ggml_new_tensor_impl`（`src/ggml.c:1766-1842`）

这是所有张量创建的**唯一入口**，`ggml_new_tensor_1d/2d/3d/4d` 都是它的薄包装。

```cpp
1766  static struct ggml_tensor * ggml_new_tensor_impl(
1767          struct ggml_context * ctx,
1768          enum   ggml_type      type,
1769          int                   n_dims,
1770          const int64_t       * ne,
1771          struct ggml_tensor  * view_src,
1772          size_t                view_offs) {
1773
1774      GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT);
1775      GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);
1776
1777      // find the base tensor and absolute offset
1778      if (view_src != NULL && view_src->view_src != NULL) {
1779          view_offs += view_src->view_offs;
1780          view_src   = view_src->view_src;
1781      }
1782
1783      size_t data_size = ggml_row_size(type, ne[0]);
1784      for (int i = 1; i < n_dims; i++) {
1785          data_size *= ne[i];
1786      }
1787
1788      GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));
1789
1790      void * data = view_src != NULL ? view_src->data : NULL;
1791      if (data != NULL) {
1792          data = (char *) data + view_offs;
1793      }
1794
1795      size_t obj_alloc_size = 0;
1796
1797      if (view_src == NULL && !ctx->no_alloc) {
1798          // allocate tensor data in the context's memory pool
1799          obj_alloc_size = data_size;
1800      }
1801
1802      GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size);
1803
1804      struct ggml_object * const obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
1805      GGML_ASSERT(obj_new);
1806
1807      struct ggml_tensor * const result = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);
1808
1809      *result = (struct ggml_tensor) {
1810          /*.type         =*/ type,
1811          /*.buffer       =*/ NULL,
1812          /*.ne           =*/ { 1, 1, 1, 1 },
1813          /*.nb           =*/ { 0, 0, 0, 0 },
1814          /*.op           =*/ GGML_OP_NONE,
1815          /*.op_params    =*/ { 0 },
1816          /*.flags        =*/ 0,
1817          /*.src          =*/ { NULL },
1818          /*.view_src     =*/ view_src,
1819          /*.view_offs    =*/ view_offs,
1820          /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data,
1821          /*.name         =*/ { 0 },
1822          /*.extra        =*/ NULL,
1823          /*.padding      =*/ { 0 },
1824      };
1825
1826      for (int i = 0; i < n_dims; i++) {
1827          result->ne[i] = ne[i];
1828      }
1829
1830      result->nb[0] = ggml_type_size(type);
1831      result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
1832      for (int i = 2; i < GGML_MAX_DIMS; i++) {
1833          result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
1834      }
1835
1836      ctx->n_objects++;
1837
1838      return result;
1839  }
```

#### 3.4.1 视图扁平化（1777-1781）

```cpp
if (view_src != NULL && view_src->view_src != NULL) {
    view_offs += view_src->view_offs;
    view_src   = view_src->view_src;
}
```

**为什么只追一层？** 因为构造时保证：任何视图的 `view_src` 一定指向**基础张量**（`view_src->view_src == NULL`）。如果出现“视图的视图”，这里会把它压平——但只压一层，因为调用方（`ggml_view_impl`）已经保证了传入的 `view_src` 是基础张量。

**这在 mnist 里的体现**：`grad for node_6 (transposed)` 的 `view_src` 指向 `grad for node_6`（基础张量），`view_offs=0`。如果对这个转置再做一次 view，新张量的 `view_src` 仍然指向 `grad for node_6`，`view_offs` 累加。

#### 3.4.2 `data_size` 的计算（1783-1786）

```cpp
size_t data_size = ggml_row_size(type, ne[0]);
for (int i = 1; i < n_dims; i++) {
    data_size *= ne[i];
}
```

**`ggml_row_size(type, ne[0])` 而不是 `ne[0] * type_size`**——这个区别对量化类型至关重要：

```cpp
// ggml.c
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}
```

对 F32：`blck_size=1`，`row_size = 4 * ne0`。
对 Q4_0：`blck_size=32`，`type_size=18`，`row_size = 18 * ne0 / 32`——**每 32 个元素占 18 字节**（2 B 的 fp16 scale + 16 B 的 4-bit 数据）。

所以 `fc1.weight` 的 `data_size = row_size(F32, 784) * 500 = 3136 * 500 = 1,568,000`，与断点完全一致。

#### 3.4.3 三条 `data` 路径（1790-1800）

```cpp
void * data = view_src != NULL ? view_src->data : NULL;
if (data != NULL) {
    data = (char *) data + view_offs;
}

size_t obj_alloc_size = 0;

if (view_src == NULL && !ctx->no_alloc) {
    obj_alloc_size = data_size;
}
```

| 场景 | `view_src` | `ctx->no_alloc` | `data` | `obj_alloc_size` |
|---|---|---|---|---|
| 普通张量 + `no_alloc=false` | NULL | false | `result + 1`（arena 内） | `data_size` |
| 普通张量 + `no_alloc=true`（本例） | NULL | true | **NULL** | 0 |
| 视图 + 源已分配 | 非 NULL | 任意 | `view_src->data + view_offs` | 0 |
| 视图 + 源未分配 | 非 NULL | 任意 | **NULL**（源 data 也是 NULL） | 0 |

**本例全部落在第 2 行**：`ctx_static` / `ctx_compute` / `opt ctx_static` 都是 `no_alloc=true`，所以断点里所有张量的 `data=(nil)`。真正的地址由 `ggml_backend_alloc_ctx_tensors` 或 `gallocr` 后填。

**视图在第 4 行**：`TENSOR#35`、`#37`、`#38` 的 `view_src` 非空且 `data=(nil)`，因为它们的源张量还没分配。等 `ggml_backend_view_init` 时：

```cpp
// src/ggml-backend.cpp:2122
tensor->buffer = tensor->view_src->buffer;
tensor->data   = (char *)tensor->view_src->data + tensor->view_offs;
```

#### 3.4.4 对象大小（1804）

```cpp
struct ggml_object * const obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR,
                                                     GGML_TENSOR_SIZE + obj_alloc_size);
```

`GGML_TENSOR_SIZE = 336`，`obj_alloc_size = 0`，所以每个张量对象是 `GGML_PAD(336, 16) = 336` 字节载荷 + 32 字节对象头 = **368 字节**。这就是 `ggml_tensor_overhead()` 的由来。

`ctx_static` 的 `mem_size = 1024 * 368 = 376,832`，正好对应 1024 个张量。断点里 `ctx_static` 只用了 5 个（4 权重 + images），浪费了 1019 个槽位——**这是“预留容量”的典型代价**，但 376 KB 对 24 GB 显存微不足道。

#### 3.4.5 结构体初始化（1809-1824）

```cpp
*result = (struct ggml_tensor) {
    /*.type         =*/ type,
    /*.buffer       =*/ NULL,        // ← 后端 buffer 由 alloc_ctx_tensors 填
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
```

**`data = result + 1` 的含义**：如果 `no_alloc=false` 且不是视图，张量数据紧跟在 `ggml_tensor` 结构体之后（同一个 `ggml_object` 载荷里）。这是“元数据 + 数据”合并分配的优化——一次 `ggml_new_object` 同时拿到 336 B 结构体和 `data_size` 字节的数据。

**本例不走这条路径**，因为 `no_alloc=true`。

**`buffer = NULL`**：此时张量还没绑定后端。`ggml_backend_alloc_ctx_tensors` / `ggml_gallocr_init_tensor` 会调用：

```cpp
// src/ggml-backend.cpp:2134
tensor->buffer = buffer;
tensor->data   = addr;
return ggml_backend_buffer_init_tensor(buffer, tensor);
```

#### 3.4.6 `nb` 推导（1829-1834）

```cpp
for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
}

result->nb[0] = ggml_type_size(type);
result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
for (int i = 2; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
}
```

**注意 1829 的循环只到 `n_dims`**：如果创建 2D 张量，`ne[2]` 和 `ne[3]` 保持初始化的 1。

**但 `nb` 的循环从 `i=2` 到 `GGML_MAX_DIMS-1`（即 3）**，所以 4 个 `nb` 都会被填：

```text
nb[2] = nb[1] * ne[1]
nb[3] = nb[2] * ne[2]   ← ne[2]=1，所以 nb[3] == nb[2]
```

这解释了断点里为什么 2D 张量的 `nb[2] == nb[3]`。

**逐项验算 `fc1.weight`（`ne=[784,500]`，F32）**：

```text
nb[0] = 4
nb[1] = 4 * (784 / 1) = 3136
nb[2] = 3136 * 500 = 1,568,000
nb[3] = 1,568,000 * 1 = 1,568,000
```

**`n_dims` 与 `ne` 的关系**：`ggml_new_tensor_2d(ctx, F32, 784, 500)` 里 `n_dims=2`、`ne[0]=784`、`ne[1]=500`。但 `ggml_tensor` 的 `ne` 数组固定 4 维，`ne[2]=ne[3]=1` 由初始化填。

**这就是 ggml 的“4 维统一表示”**：所有张量都是 4 维，1D/2D 只是高维退化为 1。好处是 kernel 不需要按维度分叉，坏处是 2D 张量的 `nb[2]`、`nb[3]` 是冗余的（但被 `nb[3] == nb[2]` 的规律一致填充）。

---

## 4 第四层：静态 buffer 分配（`ggml_backend_alloc_ctx_tensors`）

### 4.1 调用位置

```cpp
// mnist-common.cpp:296
model.buf_static = ggml_backend_alloc_ctx_tensors(model.ctx_static, model.backends[0]);

// mnist-common.cpp:541 (ggml-opt.cpp)
opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(...));

// ggml-opt.cpp:546
opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
```

### 4.2 断点：4 次分配

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

```text
CTX_BUFFER buft=CPU    size=190560000                                    ← dataset
CTX_BUFFER buft=CUDA0  size=  3158272 dev_ptr=0x7fddf0c00000             ← model.ctx_static
CTX_BUFFER buft=CUDA0  size=  4795520 dev_ptr=0x7fddf1000000             ← opt ctx_static
CTX_BUFFER buft=CPU    size=       32 host_ptr=0x5608e8327c00            ← opt ctx_cpu
```

### 4.3 3,158,272 的推导

`alloc_ctx_tensors_from_buft_impl` 遍历 context 中所有 `data==NULL && view_src==NULL` 的张量：

```cpp
size_t this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);  // alignment = 128
cur_buf_size += this_size;
```

| 张量 | nbytes | pad 到 128 | 累计 |
|---|---|---|---|
| `fc1.weight` | 1,568,000 | 1,568,000（已对齐） | 1,568,000 |
| `fc1.bias` | 2,000 | 2,048 | 1,570,048 |
| `fc2.weight` | 20,000 | 20,096 | 1,590,144 |
| `fc2.bias` | 40 | 128 | 1,590,272 |
| `images` | 1,568,000 | 1,568,000 | 3,158,272 |

**注意 `fc2.bias` 从 40 B 涨到 128 B**：单个 float4 只要 40 B，但 CUDA 的 128 B 对齐要求把它补齐到 128 B。这是“按 buffer type 对齐”而不是“按张量紧凑打包”的直接代价。对 40 B 的张量，浪费 3.2 倍；对 1.5 MB 的张量，几乎无损。

### 4.4 4,795,520 的推导

`opt ctx_static` 里有多少个非视图张量？断点枚举：

```gdb
break ggml-alloc.c:1229
commands
silent
printf "=== CTX buft=%s size=%zu ===\n", ggml_backend_buft_name(buffer->buft), buffer->size
set $t = ggml_get_first_tensor(ctx)
while $t != 0
if $t->view_src == 0
printf "   %-34s ne=[%ld,%ld] nbytes=%zu\n", $t->name, $t->ne[0], $t->ne[1], ggml_nbytes($t)
end
set $t = ggml_get_next_tensor(ctx, $t)
end
continue
end
```

```text
=== CTX buft=CUDA0 size=4795520 tensors: ===
   labels                         ne=[10,500]  nbytes=20000
   loss_cross_entropy             ne=[1,1]     nbytes=4
   loss_cross_entropy_scaled      ne=[1,1]     nbytes=4
   pred                           ne=[500,1]   nbytes=2000
   node_12                        ne=[500,1]   nbytes=2000
   ncorrect                       ne=[1,1]     nbytes=8
   leaf_4                         ne=[500,10]  nbytes=20000     ← grad_acc for fc2.weight
   leaf_6                         ne=[784,500] nbytes=1568000   ← grad_acc for fc1.weight
   leaf_5                         ne=[500,1]   nbytes=2000      ← grad_acc for fc1.bias
   leaf_3                         ne=[10,1]    nbytes=40        ← grad_acc for fc2.bias
   leaf_2                         ne=[1,1]     nbytes=4         ← grad_acc for loss
   AdamW m for fc2.weight         ne=[500,10]  nbytes=20000
   AdamW v for fc2.weight         ne=[500,10]  nbytes=20000
   AdamW m for fc1.weight         ne=[784,500] nbytes=1568000
   AdamW v for fc1.weight         ne=[784,500] nbytes=1568000
   AdamW m for fc1.bias           ne=[500,1]   nbytes=2000
   AdamW v for fc1.bias           ne=[500,1]   nbytes=2000
   AdamW m for fc2.bias           ne=[10,1]    nbytes=40
   AdamW v for fc2.bias           ne=[10,1]    nbytes=40
   SUM=4794140
```

对齐后：

```text
labels 20,000 → 20,096
loss 4 + loss_scaled 4 → 128 + 128
pred 2,000 → 2,048
node_12 2,000 → 2,048
ncorrect 8 → 128
leaf_4 20,000 → 20,096
leaf_6 1,568,000 → 1,568,000
leaf_5 2,000 → 2,048
leaf_3 40 → 128
leaf_2 4 → 128
m/v × 4 组：20,096×2 + 1,568,000×2 + 2,048×2 + 128×2
= 4,795,520  ✅
```

**为什么梯度累加器和 AdamW 状态要放在静态 buffer？**
因为它们必须**跨 batch 保持**：

- `grad_accs`（`leaf_2..leaf_6`）：梯度累加，batch #1 的梯度不能丢。
- AdamW `m` / `v`：一阶、二阶动量，跨所有 iteration 累积。

如果交给 gallocr，它们的地址每个 batch 都可能变，数据就丢了。`GGML_TENSOR_FLAG_PARAM` 让 gallocr 把参数当作“永不复用”的节点，`ggml_set_output` 让 loss/pred/ncorrect 不被复用。

### 4.5 为什么 `adamw_params` 在 CPU？

```cpp
opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
```

每个 iteration 的 AdamW 超参数（`alpha, beta1, beta2, eps, wd, beta1h, beta2h`）由 `get_opt_pars` 在 host 上算出来，直接写进 CPU 张量：

```cpp
float * adamw_par_data = ggml_get_data_f32(opt_ctx->opt_step_params);
adamw_par_data[0] = opt_pars.adamw.alpha;
...
```

然后由 scheduler 在 split 阶段插入一次 CPU→CUDA 拷贝（见 03 §1.5）。**为什么不做成 `op_params`？** 因为 `op_params` 是编译期常量，而 AdamW 超参每个 iteration 都可能变（warmup 学习率、beta 衰减）。用张量表达，就能复用同一条“输入张量 → 跨后端拷贝”的通路，不需要给算子加特殊参数通道。

---

### 4.6 源码逐行：`ggml_backend_alloc_ctx_tensors_from_buft_impl`（`src/ggml-alloc.c:1168-1230`）

这是“把 context 里所有未分配张量一次性绑定到某个 backend buffer”的完整实现。

```cpp
1168  static ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft_impl(
1169          struct ggml_context * ctx, ggml_backend_buffer_type_t buft, size_t * nbytes_total, bool no_alloc) {
1170      GGML_ASSERT(ggml_get_no_alloc(ctx) == true);
1171
1172      size_t alignment = ggml_backend_buft_get_alignment(buft);
1173      size_t max_size = ggml_backend_buft_get_max_size(buft);
1174
1175      ggml_backend_buffer_t * buffers = NULL;
1176      size_t n_buffers = 0;
1177      *nbytes_total = 0;
1178
1179      size_t cur_buf_size = 0;
1180      struct ggml_tensor * first = ggml_get_first_tensor(ctx);
1181      for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
1182          size_t this_size = 0;
1183          if (t->data == NULL && t->view_src == NULL) {
1184              this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
1185          }
1186
1187          if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
1188              // allocate tensors in the current buffer
1189              if (!no_alloc && !alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
1190                  return NULL;
1191              }
1192              first = t;
1193              *nbytes_total += cur_buf_size;
1194              cur_buf_size = this_size;
1195          } else {
1196              cur_buf_size += this_size;
1197          }
1198      }
1199
1200      // allocate remaining tensors
1201      if (cur_buf_size > 0) {
1202          *nbytes_total += cur_buf_size;
1203          if (!no_alloc && !alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
1204              return NULL;
1205          }
1206      }
1207
1208      if (no_alloc) {
1209          return NULL;
1210      }
1211
1212      if (n_buffers == 0) {
1213  #ifndef NDEBUG
1214          GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
1215  #endif
1216          GGML_ASSERT(!buffers);
1217          return NULL;
1218      }
1219
1220      ggml_backend_buffer_t buffer;
1221      if (n_buffers == 1) {
1222          buffer = buffers[0];
1223      } else {
1224          buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
1225      }
1226      if (buffers) {
1227          free(buffers);
1228      }
1229      return buffer;
1230  }
```

逐行说明：

| 行 | 行为 | 为什么 |
|---|---|---|
| 1170 | 断言 `no_alloc == true` | 这个函数**只能**用于“元数据已创建、数据未分配”的 context；如果 `no_alloc=false`，张量数据已经在 arena 里了，不需要再分配 |
| 1172 | `alignment = buft->get_alignment()` | CUDA0 = 128，CPU = 64。**每个张量的分配大小都要按这个对齐** |
| 1173 | `max_size = buft->get_max_size()` | CUDA0 的 iface 里是 NULL → 默认 `SIZE_MAX`，所以永不切分；某些后端（如 Vulkan）有单 buffer 上限，需要多 buffer |
| 1183 | `t->data == NULL && t->view_src == NULL` | 只统计“普通张量”，视图不占空间（它共享源的 data） |
| 1184 | `GGML_PAD(get_alloc_size(buft, t), alignment)` | `get_alloc_size` 可能大于 `ggml_nbytes`（量化行补齐、flash-attn padding），再按 buffer 对齐 |
| 1187 | `cur_buf_size + this_size > max_size` | 只有超过后端上限才切 buffer；本例永不触发 |
| 1196 | `cur_buf_size += this_size` | 顺序累加，**不排序、不优化**——张量按 context 创建顺序排列 |
| 1203 | `alloc_tensor_range(ctx, first, NULL, ...)` | 一次性分配整块 buffer，然后在里面顺序切给每个张量 |
| 1221-1225 | 单 buffer 直接返回，多 buffer 包装 | 多 buffer 场景下返回一个 `ggml_backend_multi_buffer`，对上层透明 |

#### 4.6.1 `alloc_tensor_range`：从 buffer 切给张量（`src/ggml-alloc.c:1127-1166`）

```cpp
1127  static bool alloc_tensor_range(struct ggml_context * ctx,
1128          struct ggml_tensor * first, struct ggml_tensor * last,
1129          ggml_backend_buffer_type_t buft, size_t size,
1130          ggml_backend_buffer_t ** buffers, size_t * n_buffers) {
1131
1132      ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
1133      if (buffer == NULL) { ... return false; }
1134
1139      *buffers = realloc(*buffers, sizeof(ggml_backend_buffer_t) * (*n_buffers + 1));
1140      (*buffers)[(*n_buffers)++] = buffer;
1141
1142      struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);
1143
1144      for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
1145          enum ggml_status status = GGML_STATUS_SUCCESS;
1146          if (t->data == NULL) {
1147              if (t->view_src == NULL) {
1148                  status = ggml_tallocr_alloc(&tallocr, t);
1149              } else if (t->buffer == NULL) {
1150                  status = ggml_backend_view_init(t);
1151              }
1152          } else {
1153              if (t->view_src != NULL && t->buffer == NULL) {
1154                  // view of a pre-allocated tensor
1155                  status = ggml_backend_view_init(t);
1156              }
1157          }
1158          if (status != GGML_STATUS_SUCCESS) { ... return false; }
1159      }
1160
1161      return true;
1162  }
```

**这是线性 tallocr，不是 gallocr**：

```cpp
// src/ggml-alloc.c:76
enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) { ... abort ... }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;                    // ← 只增不减

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}
```

**关键差异**：

| | `ggml_tallocr`（静态） | `ggml_dyn_tallocr`（动态，gallocr 用） |
|---|---|---|
| 分配策略 | 顺序 bump pointer | best-fit free list |
| 释放 | **不支持** | 支持，按 `n_children` 归零释放 |
| 用途 | `alloc_ctx_tensors`（权重、输入、梯度累加器） | 计算图的中间张量 |
| 本例大小 | 3,158,272 / 4,795,520 | 3,588,352 |

**为什么静态张量不用 free list**：它们的生命周期是整个程序（权重、`images`、`grad_accs`、AdamW m/v），永远不会释放。用 bump pointer 最简单、最快，且地址稳定。

#### 4.6.2 `3,158,272` 的逐张量推导（断点验证）

| 顺序 | 张量 | `ggml_nbytes` | `get_alloc_size` | `PAD(,128)` | 累计 offset |
|---|---|---|---|---|---|
| 0 | `fc1.weight` | 1,568,000 | 1,568,000 | 1,568,000 | 0 → 1,568,000 |
| 1 | `fc1.bias` | 2,000 | 2,000 | 2,048 | 1,568,000 → 1,570,048 |
| 2 | `fc2.weight` | 20,000 | 20,000 | 20,096 | 1,570,048 → 1,590,144 |
| 3 | `fc2.bias` | 40 | 40 | 128 | 1,590,144 → 1,590,272 |
| 4 | `images` | 1,568,000 | 1,568,000 | 1,568,000 | 1,590,272 → 3,158,272 |

**注意 `fc1.bias` 从 2,000 涨到 2,048**：`2000 % 128 = 80`，补齐 `128 - 80 = 48`。这不是浪费，而是**下一个张量必须从 128 的倍数开始**。

**`fc2.bias` 从 40 涨到 128**：`40 % 128 = 40`，补齐 88。对 40 B 的张量浪费 220%，但它是最后一个（在 `images` 之前），后面紧跟 1.5 MB 的 `images`，所以这 88 B 完全被淹没。

#### 4.6.3 为什么 `nbytes_total` 与 `buffer->size` 一致

断点里 `nbytes_total=3158272`，`buffer->size=3158272`，完全相等。因为：

```cpp
// 1184 行：每个 this_size 已经 PAD 过
this_size = GGML_PAD(get_alloc_size(buft, t), alignment);

// 1196 行：直接累加
cur_buf_size += this_size;

// 1202 行：传给 alloc_tensor_range
*nbytes_total += cur_buf_size;
alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, ...);
```

所以 `alloc_tensor_range` 收到的 `size` 就是所有对齐后大小之和，`ggml_backend_buft_alloc_buffer(buft, size)` 一次 `cudaMalloc` 正好这个大小。**没有额外的 buffer 头开销**——CUDA buffer 的 `ggml_backend_buffer` 结构体是 host 侧 `new` 出来的，不占显存。

---

## 5 第五层：权重初始化与 H2D

### 5.1 随机初始化

```cpp
std::normal_distribution<float> nd{0.0f, 1e-2f};
for (ggml_tensor * t : init_tensors) {
    const int64_t ne = ggml_nelements(t);
    std::vector<float> tmp(ne);
    for (int64_t i = 0; i < ne; ++i) tmp[i] = nd(gen);
    ggml_backend_tensor_set(t, tmp.data(), 0, ggml_nbytes(t));
}
```

`std::normal_distribution(0, 0.01)` 是经典的权重初始化：均值 0、标准差 0.01，配合 ReLU 在 MNIST 上足够收敛。对比 He 初始化 `σ=√(2/fan_in)=√(2/784)=0.05`，这里故意取更小，避免初始 logits 过大导致 softmax 饱和。

### 5.2 断点：4 次 H2D

```gdb
break ggml_backend_tensor_set
commands
silent
printf "H2D_SET name=%-10s off=%8zu size=%8zu tensor_data=%p\n", tensor->name, offset, size, tensor->data
continue
end
```

```text
H2D_SET#1 name=fc1.weight off=0 size=1568000 tensor_data=0x7f5e6cc00000
H2D_SET#2 name=fc1.bias   off=0 size=   2000 tensor_data=0x7f5e6cd7ed00
H2D_SET#3 name=fc2.weight off=0 size=  20000 tensor_data=0x7f5e6cd7f500
H2D_SET#4 name=fc2.bias   off=0 size=     40 tensor_data=0x7f5e6cd84380
```

CUDA 后端的实现：

```cpp
static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size) {
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size,
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```

**注意三点**：

1. 用的是 `cudaStreamPerThread`，**不是** `ctx->stream()`。所以权重上传与计算图执行不在同一条 stream 上，靠显式 `cudaStreamSynchronize` 保证完成。
2. 每次 `tensor_set` 都带一次同步。4 次权重上传 = 4 次同步，初始化阶段可接受；但训练时每 batch 100 次就是 100 次同步（见 04 §6.3）。
3. `tensor->data` 此时已经是 CUDA 设备指针（`0x7f5e...`），`buffer` 已经绑定。

### 5.3 为什么初始化用 H2D 而不是直接在 GPU 上生成随机数？

- ggml 没有“GPU 随机数生成”算子；
- `std::mt19937` + `std::normal_distribution` 是标准库实现，host 上生成 392,000 个 float 只要几百微秒；
- 权重上传只发生一次，不在训练热路径上。

实测 1,568,000 B 的 H2D+sync 是 **135.62 µs**，4 个权重合计约 0.3 ms，占 6.9 s 训练的 0.004%。

---

## 6 推理路径：GGUF 加载的另一种初始化

`mnist-eval` 走的是 `mnist_model_init_from_file`，与训练路径有三处关键差异。

### 6.1 断点：GGUF 以 `no_alloc=true` 打开

```gdb
break gguf_init_from_file
commands
silent
printf "GGUF_LOAD file=%s no_alloc=%d\n", fname, params.no_alloc
continue
end
```

```text
GGUF_LOAD file=mnist-fc-f32.gguf no_alloc=1
```

```cpp
struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ &model.ctx_gguf };
ctx = gguf_init_from_file(fname.c_str(), params);
```

`no_alloc=true` 让 gguf 只创建**张量元数据**（name / ne / nb / type / data_offset），不分配数据。数据仍在文件里，由 `load_from_gguf` 分块读入：

```cpp
const size_t buf_size = 4*1024*1024;
for (size_t pos = 0; pos < nbytes; pos += buf_size) {
    fread(buf, 1, nbytes_cpy, f);
    ggml_backend_tensor_set(tensor, buf, pos, nbytes_cpy);
}
```

**为什么要 4 MB 分块？** 因为 `tensor_set` 是同步拷贝，一次性读 1.5 MB 没问题，但真实 LLM 的权重是 GB 级，必须分块避免 host 内存峰值。这个函数在源码里明确标注是临时方案：

```cpp
// Temporary util function for loading data from GGUF to a backend != CPU until GGML itself provides this functionality
```

### 6.2 断点：推理的 5 块 buffer

```text
CTX_BUFFER buft=CPU    size=31760000   ← 10000 张测试图 + 标签
CTX_BUFFER buft=CUDA0  size= 1590272   ← 4 个权重（1,568,000 + 2,048 + 20,096 + 128）
CTX_BUFFER buft=CUDA0  size= 1568000   ← images（独立 ctx_static）
CTX_BUFFER buft=CUDA0  size=   24448   ← labels + loss + pred + node_11 + ncorrect
```

**为什么推理要两个 CUDA buffer？** 源码注释：

```cpp
// The space in ctx_gguf exactly fits the model weights,
// the images (which also need to be statically allocated) need to be put in a different context.
```

`ctx_gguf` 是由 gguf 元数据精确算出来的，没有余量放 `images`，所以 `images` 单独放在 `model.ctx_static`。

### 6.3 训练与推理的 context 对比

| | 训练 | 推理 |
|---|---|---|
| 权重来源 | `ggml_new_tensor_2d` + 随机数 | `gguf_init_from_file` + `load_from_gguf` |
| `images` 所在 ctx | `ctx_static`（与权重同 ctx） | 独立 `ctx_static` |
| 输出张量 | loss / pred / ncorrect | loss / pred / ncorrect（同样，用于评估） |
| 图数量 | 3（gf / gb_grad / gb_opt） | 1（gf） |
| 图节点数 | 14 / 28 / 32 | 13 |

**推理图为什么是 13 而不是 14？** 训练图有一个 `loss_cross_entropy_scaled`（`GGML_OP_SCALE`，把 loss 除以 `opt_period=2`）。推理时 `opt_period=1`，不需要缩放：

```cpp
if (opt_ctx->opt_period > 1) {
    opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
}
```

---

## 7 CUDA 后端的惰性初始化（断点证据）

### 7.1 断点

```gdb
start mnist-fc ... CUDA0
printf "DEVICE_INFO cc=%d nsm=%d warp=%d smpbo=%zu\n",
  ggml_cuda_info().devices[0].cc, ggml_cuda_info().devices[0].nsm,
  ggml_cuda_info().devices[0].warp_size, ggml_cuda_info().devices[0].smpbo

break ggml_cuda_mul_mat
commands
silent
printf "MUL_MAT src0=%s src1=%s dst=%s\n", src0->name, src1->name, dst->name
printf "  ctx.device=%d cublas_handle=%p stream=%p workspace_size=%zu\n",
  ctx.device, ctx.cublas_handles[0][0], ctx.streams[0][0], ctx.cublas_workspace_sizes[0]
continue
end
```

### 7.2 原始输出

```text
DEVICE_INFO cc=890 nsm=128 warp=32 smpbo=101376
DEVICE vmm=1 integrated=0

MUL_MAT src0=fc1.weight src1=images dst=node_2
  ctx.device=0 cublas_handle=(nil) stream=0x55b32c9d0ab0 workspace_size=0

MUL_MAT src0=fc2.weight src1=node_5 dst=node_6
  ctx.device=0 cublas_handle=0x55b32cdae620 stream=0x55b32c9d0ab0 workspace_size=4194304
```

### 7.3 解读

**第一次 `MUL_MAT`**：

- `stream` 已经存在（`0x55b32c9d0ab0`）—— 它在 `alloc_splits` 里被创建，因为 `ggml_backend_synchronize` 需要 stream。
- `cublas_handle == NULL` —— cuBLAS 还没初始化。
- `workspace_size == 0` —— 还没分配 workspace。

**第二次 `MUL_MAT`**：

- `cublas_handle = 0x55b32cdae620` —— 第一次调用时 `ctx.cublas_handle()` 惰性创建。
- `workspace_size = 4,194,304` —— 4 MiB，因为 `cc=890 < Hopper(900)`：

```cpp
cublas_workspace_sizes[device] = (cc >= GGML_CUDA_CC_HOPPER) ? 32*1024*1024 : 4*1024*1024;
```

**为什么惰性？** 三个理由：

1. **小图不该付出大初始化成本**：`cublasCreate + workspace` 实测 **24.53 ms**（见 04 §6.3）。如果程序只跑 CPU 算子，这 24 ms 完全浪费。
2. **多 stream 场景**：`streams[8][8]`、`cublas_handles[8][8]`、`pools[8][8]`，如果全部预创建，8×8=64 个 handle，每个 4 MiB workspace = 256 MiB，不可接受。
3. **失败点后移**：初始化阶段不碰 CUDA runtime，便于在没有 GPU 的环境下做纯构图/纯规划。

### 7.4 这张表总结了所有惰性资源

| 资源 | 创建时机 | 断点证据 |
|---|---|---|
| `ggml_backend_registry` | 首次 `ggml_backend_dev_by_name` / `dev_count` | `register_backend: registered backend CUDA` |
| CUDA device 枚举 | 首次 `ggml_cuda_info()` | `ggml_cuda_init: found 1 CUDA devices` |
| `ggml_backend_cuda_context` | `ggml_backend_dev_init` | `CUDA_BACKEND_INIT device=0` ×2 |
| CUDA stream | 首次 `ctx.stream()` | 第一次 `MUL_MAT` 前已存在 |
| cuBLAS handle | 首次 `ctx.cublas_handle()` | 第一次 `MUL_MAT` 时为 NULL |
| cuBLAS workspace | 同上 | 4 MiB |
| CUDA pool | 首次 `ctx.pool()` | 本例未创建（cuBLAS 路径不需要临时 buffer） |
| kernel module | 首次 launch | 见 04 §7 的冷启动现象 |

---

## 8 内存账本总表

### 8.1 训练

```text
host                                                         device
────                                                         ──────
dataset->buf        190,560,000  (images 188,160,000 + labels 2,400,000)
model.buf_static      3,158,272  ─────────────────────────►  dev_ptr 0x7fddf0c00000
   fc1.weight 1,568,000
   fc1.bias       2,048
   fc2.weight    20,096
   fc2.bias         128
   images     1,568,000
opt.buf_static        4,795,520  ─────────────────────────►  dev_ptr 0x7fddf1000000
   labels 20,096 / loss 128+128 / pred 2,048 / node_12 2,048 / ncorrect 128
   grad_accs 20,096 + 1,568,000 + 2,048 + 128 + 128
   AdamW m/v 4×2 个
opt.buf_cpu                  32  ◄──── 7 个 float（28 B，pad 到 32）
galloc compute buffer 3,588,352  ─────────────────────────►  dev_ptr 0x7ff295600000
```

合计显存：`3,158,272 + 4,795,520 + 3,588,352 = 11,542,144 B ≈ 11.0 MiB`
加上 cuBLAS workspace 4 MiB，总计约 **15 MiB**。对 24 GB 的 4090 来说微不足道，但这个账本在真实 LLM 里就是 24 GB 的去向。

### 8.2 一个反直觉的对比

| 项目 | 字节 | 占比 |
|---|---|---|
| 参数（4 个权重） | 1,590,040 | 13.8% |
| 梯度累加器 | 1,590,400 | 13.8% |
| AdamW m + v | 3,180,800 | 27.6% |
| 输入/输出/标量 | 1,610,000 | 13.9% |
| 计算临时（峰值） | 3,588,352 | 31.1% |

**训练显存 ≈ 参数 × 7**，这是经典结论在 ggml 里的具体形态：

```text
参数 1.0× + 梯度 1.0× + AdamW m/v 2.0× + 激活/临时 ~3×  ≈ 7×
```

mnist 的 1.59 MB 参数对应 11 MB 显存，正好符合。真实 LLM 的 7B 参数（F32 = 28 GB）在单卡上放不下，所以才有混合精度、LoRA、ZeRO 这些技术 —— 它们本质上都是在压缩这张表里的某一行。

---

## 9 本篇结论

1. **context 是元数据池，不是显存分配器**：`mem_size` 是容量，`data` 全部为 `NULL`，直到 `alloc_ctx_tensors` 才绑定。
2. **静态 buffer 与计算 buffer 的边界由三个 flag 决定**：`INPUT` / `OUTPUT` / `PARAM`。前者地址必须稳定，后者可复用。
3. **CUDA alignment = 128 B**：小张量（40 B）会被放大 3.2 倍，这是“整块 buffer + 对齐”的代价，换来的是 `cudaMalloc` 次数从 N 降到 1。
4. **CUDA 资源全部惰性**：stream 在 alloc 阶段创建，cuBLAS handle 在第一次 `MUL_MAT` 时创建，workspace 4 MiB。
5. **3 个 backend 里有 2 个是 CUDA0**：这是“backend = stream”的设计，gallocr 按 buft 去重，所以只多一个 stream，不多一块显存。
6. **训练显存 ≈ 参数 × 7**：参数 1× + 梯度 1× + AdamW 2× + 激活 ~3×。
