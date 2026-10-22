# 05 · 二次扩展与优化指南

> 本文把前四篇的断点结论转成“可操作的改造手册”。
>
> 所有代码骨架都对应 `/home/ggml`（commit `353b63b`）的真实接口，省略号表示与本文无关的字段。

---

## 0 扩展地图

| 目标 | 需要改的层 | 难度 | 本文小节 |
|---|---|---|---|
| 新增一个算子 | op 枚举 + builder + CPU/CUDA 实现 + supports_op | 中 | §1 |
| 新增 MUL_MAT 的 kernel 路径 | `ggml_cuda_mul_mat` 决策树 + 新 kernel | 中高 | §2 |
| 新增一个后端（新硬件） | reg/device/backend/buft/buffer 五套 iface | 高 | §3 |
| 新增一种内存类型 | buffer type iface | 低 | §4 |
| 改变调度优先级/切分 | scheduler 参数或 split 逻辑 | 低中 | §5 |
| 减少显存/复用 | `set_input/output`、reserve、alignment | 低 | §6 |
| 提升 kernel 性能 | 向量化、模板、fusion、CUDA graph | 高 | §7 |
| 定位瓶颈 | 环境变量、gdb、nsys/ncu | 低 | §8 |

---

## 1 新增算子

以一个“逐元素 `y = a * scale + bias`”的自定义算子 `GGML_OP_MY_SCALE_ADD` 为例。虽然 `ggml_scale_bias` 已存在，但它的实现路径与本例完全同构，适合作为模板。

### 1.1 第一步：枚举与名字表

`include/ggml.h`：

```cpp
enum ggml_op {
    ...
    GGML_OP_MY_SCALE_ADD,
    ...
    GGML_OP_COUNT,
};
```

`src/ggml.c`：

```cpp
static const char * GGML_OP_NAME[GGML_OP_COUNT] = {
    ...
    [GGML_OP_MY_SCALE_ADD] = "MY_SCALE_ADD",
};

static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    ...
    [GGML_OP_MY_SCALE_ADD] = "my_scale_add(x)",
};
```

注意：

```text
1. 新枚举必须加在 GGML_OP_COUNT 之前
2. NAME/SYMBOL 必须用指定初始化器，否则数组顺序错位
3. 如果算子有特殊属性（可 inplace、可量化、可 fusion），还要改：
   ggml_op_can_inplace()
   ggml_op_is_empty()
   ggml_op_alloc_size_may_expand()
```

### 1.2 第二步：builder

```cpp
struct ggml_tensor * ggml_my_scale_add(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 scale,
        float                 bias) {
    GGML_ASSERT(ggml_is_contiguous(a));

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_f32(result, 1, bias);

    result->op     = GGML_OP_MY_SCALE_ADD;
    result->src[0] = a;

    return result;
}
```

要点：

```text
1. 输出形状 = 输入形状（逐元素）
2. 标量参数放 op_params，不要放全局变量
3. 只建表达式，不计算
4. 命名（ggml_set_name）可选，但对 profiling 很有用
5. ggml_set_op_params_f32 是 ggml-impl.h 的内部 helper；
   外部扩展直接用 memcpy 写 tensor->op_params
```

### 1.3 第三步：CPU 实现（fallback）

即使只打算跑 GPU，也建议实现 CPU 版本：

```text
1. scheduler 的最后一个 backend 是 CPU，CUDA 不支持的 op 会 fallback
2. 便于数值对拍
3. 便于在没有 GPU 的环境跑测试
```

`src/ggml-cpu/ggml-cpu.c`：

```cpp
static void ggml_compute_forward_my_scale_add(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    float scale, bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;
    const int64_t n = ggml_nelements(src0);

    const float * x = (const float *) src0->data;
    float * y = (float *) dst->data;

    for (int64_t i = ith; i < n; i += nth) {
        y[i] = scale * x[i] + bias;
    }
}
```

在 `ggml_compute_forward` 的 switch 中加：

```cpp
case GGML_OP_MY_SCALE_ADD:
    ggml_compute_forward_my_scale_add(params, tensor);
    break;
```

### 1.4 第四步：CUDA kernel

新建 `src/ggml-cuda/my-scale-add.cu` / `.cuh`：

```cpp
// my-scale-add.cu
#include "my-scale-add.cuh"

static __global__ void my_scale_add_f32(
        const float * __restrict__ x,
        float * __restrict__ y,
        const float scale,
        const float bias,
        const int64_t n) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        y[i] = scale * x[i] + bias;
    }
}

void ggml_cuda_op_my_scale_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    float scale, bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    const int64_t n = ggml_nelements(dst);
    const int block = 256;
    const int grid  = (int) MIN((n + block - 1) / block, (int64_t) 65535);

    const ggml_cuda_kernel_launch_params launch_params(
        dim3(grid), dim3(block), 0, ctx.stream());

    ggml_cuda_kernel_launch(my_scale_add_f32, launch_params,
        (const float *) src0->data, (float *) dst->data, scale, bias, n);
}
```

要点：

```text
1. 用 ctx.stream()，不要用 cudaStreamPerThread
2. 用 ggml_cuda_kernel_launch 而不是 <<<>>>，以兼容 PDL/未来扩展
3. 检查连续性和类型；不连续时要么 fallback，要么写 strided kernel
4. 用 ggml_nelements / ggml_nrows 而不是手算
```

### 1.5 第五步：接入 `compute_forward`

`src/ggml-cuda/ggml-cuda.cu`：

```cpp
#include "my-scale-add.cuh"

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        ...
        case GGML_OP_MY_SCALE_ADD:
            ggml_cuda_op_my_scale_add(ctx, dst);
            break;
        ...
    }
}
```

### 1.6 第六步：`supports_op`

```cpp
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        ...
        case GGML_OP_MY_SCALE_ADD:
            return op->src[0]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[0])
                && ggml_is_contiguous(op);
        ...
    }
}
```

**这一步决定 scheduler 是否会把节点判给 CUDA**。如果返回 false，节点会 fallback 到 CPU，图会被切成两个 split，产生跨 backend copy。

### 1.7 第七步：测试

```cpp
// tests/test-my-scale-add.cpp
auto y_cpu = ...;   // CPU backend
auto y_gpu = ...;   // CUDA backend
for (i) assert(fabs(y_cpu[i] - y_gpu[i]) < 1e-5);
```

调试：

```gdb
break ggml_cuda_op_my_scale_add
break ggml_cuda_compute_forward
p dst->ne
p dst->nb
p src0->data
p dst->data
```

如果节点没有进入 CUDA：

```text
1. 检查 supports_op 是否返回 true
2. 检查 scheduler pass 3 是否选中 CUDA
3. 用 GGML_SCHED_DEBUG=2 看 split
```

---

## 2 新增或修改 MUL_MAT 路径

### 2.1 何时需要新路径

现有五条路径（MMVF / MMF / MMVQ / MMQ / cuBLAS）覆盖：

```text
F32/F16/BF16/量化 × 小 N/大 N × 对齐/不对齐
```

可能需要新路径的场景：

```text
1. 新的量化类型（Q3_0、MXFP8 等）
2. 新的硬件特性（Hopper TMA、Blackwell 新 MMA）
3. 特定 shape 的专用 kernel（如 K 很小、M 很大）
4. 融合前的预处理（转置、padding）
```

### 2.2 修改决策树

`ggml-cuda.cu:1824` 的 `ggml_cuda_mul_mat`：

```cpp
if (ggml_cuda_should_use_my_path(src0->type, cc, src0->ne, src0->nb, ne11)) {
    ggml_cuda_mul_mat_my_path(ctx, src0, src1, nullptr, dst);
    return;
}
```

放置顺序很重要：

```text
越靠前，优先级越高
新路径应放在“能处理它的最宽泛路径”之前，
放在“更专用的路径”之后
```

例如：如果新路径只处理“F32 + ne11 == 5”，应放在 MMVF（ne11<=3）之后、cuBLAS 之前。

### 2.3 谓词函数

```cpp
bool ggml_cuda_should_use_my_path(
        enum ggml_type type, int cc,
        const int64_t * src0_ne, const size_t * src0_nb,
        int64_t ne11) {
    if (type != GGML_TYPE_F32) return false;
    if (ne11 != 5) return false;
    if (src0_ne[0] % 16 != 0) return false;
    if (!ggml_cuda_is_aligned(...)) return false;
    return true;
}
```

要点：

```text
1. 谓词必须便宜（在每次 MUL_MAT 调用时执行）
2. 只做“能否用”判断，不做“是否更快”判断
3. 性能判断放在 benchmark 和默认策略里
```

### 2.4 兜底

无论加多少路径，最后一行必须保留：

```cpp
ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
```

这是正确性保证。

---

## 3 新增一个后端

以“假想的 NPU”为例。需要实现五套接口。

### 3.1 reg

```cpp
struct ggml_backend_reg_i {
    const char * (*get_name)(ggml_backend_reg_t);
    size_t       (*get_device_count)(ggml_backend_reg_t);
    ggml_backend_dev_t (*get_device)(ggml_backend_reg_t, size_t);
    void *       (*get_proc_address)(ggml_backend_reg_t, const char *);
};

ggml_backend_reg_t ggml_backend_npu_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ {
            /* .get_name         = */ npu_reg_name,
            /* .get_device_count = */ npu_reg_device_count,
            /* .get_device       = */ npu_reg_device,
            /* .get_proc_address = */ NULL,
        },
        /* .context = */ nullptr,
    };
    return &reg;
}
```

### 3.2 device

```cpp
struct ggml_backend_device_i {
    const char * (*get_name)(ggml_backend_dev_t);
    const char * (*get_description)(ggml_backend_dev_t);
    void         (*get_memory)(ggml_backend_dev_t, size_t *, size_t *);
    enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t);
    void         (*get_props)(ggml_backend_dev_t, struct ggml_backend_dev_props *);
    ggml_backend_t (*init_backend)(ggml_backend_dev_t, const char *);
    ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t);
    ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t);
    ggml_backend_buffer_t (*buffer_from_host_ptr)(...);
    bool (*supports_op)(ggml_backend_dev_t, const struct ggml_tensor *);
    bool (*supports_buft)(ggml_backend_dev_t, ggml_backend_buffer_type_t);
    bool (*offload_op)(ggml_backend_dev_t, const struct ggml_tensor *);
    ggml_backend_event_t (*event_new)(ggml_backend_dev_t);
    void (*event_free)(ggml_backend_dev_t, ggml_backend_event_t);
    void (*event_synchronize)(ggml_backend_dev_t, ggml_backend_event_t);
};
```

`supports_op` 的粒度决定调度质量：

```text
太严格 → 大量 fallback，split 变多，性能差
太宽松 → 运行时断言/错误
```

### 3.3 backend

```cpp
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t);
    void (*free)(ggml_backend_t);
    void (*set_tensor_async)(...);
    void (*get_tensor_async)(...);
    bool (*cpy_tensor_async)(...);
    void (*synchronize)(ggml_backend_t);
    enum ggml_status (*graph_compute)(ggml_backend_t, struct ggml_cgraph *);
    void (*event_record)(...);
    void (*event_wait)(...);
    void (*graph_optimize)(ggml_backend_t, struct ggml_cgraph *, struct ggml_backend_graph_optimize_params *);
};
```

最小实现：

```cpp
ggml_backend_t npu_init(int device) {
    npu_context * ctx = new npu_context(device);
    return new ggml_backend {
        /* .guid    = */ npu_guid(),
        /* .iface   = */ npu_backend_iface,
        /* .device  = */ npu_reg_device(ggml_backend_npu_reg(), device),
        /* .context = */ ctx,
    };
}
```

### 3.4 buffer type

```cpp
static ggml_backend_buffer_type_t npu_buffer_type(int device) {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name      = */ npu_buft_name,
            /* .alloc_buffer  = */ npu_buft_alloc_buffer,
            /* .get_alignment = */ npu_buft_alignment,   // 例如 256
            /* .get_max_size  = */ NULL,
            /* .get_alloc_size= */ npu_buft_alloc_size,
            /* .is_host       = */ NULL,
        },
        /* .device  = */ npu_reg_device(ggml_backend_npu_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}
```

### 3.5 buffer

```cpp
static ggml_backend_buffer_t npu_buft_alloc_buffer(buft, size) {
    void * ptr = npu_malloc(size);
    return ggml_backend_buffer_init(buft, npu_buffer_iface, ptr, size);
}
```

`npu_buffer_iface` 至少实现：

```text
free_buffer / get_base / set_tensor / get_tensor / clear
```

### 3.6 注册

静态链接：在 `ggml_backend_registry()` 构造函数中加：

```cpp
#ifdef GGML_USE_NPU
    register_backend(ggml_backend_npu_reg());
#endif
```

动态加载：实现 `GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)`，编译为 `libggml-npu-<score>.so`。

### 3.7 接入 scheduler

```cpp
ggml_backend_t backends[2] = { npu_backend, cpu_backend };
model.sched = ggml_backend_sched_new(backends, nullptr, 2,
                                     GGML_DEFAULT_GRAPH_SIZE, false, true);
```

注意：

```text
1. CPU 必须最后
2. NPU 的 buft alignment 决定 gallocr 的对齐
3. NPU 的 supports_buft 决定跨 backend copy 是否可行
```

---

## 4 新增 buffer type

CUDA 已经有两个 buft：

```text
CUDA0      设备显存，alignment 128
CUDA_Host  页锁定 host 内存，alignment 32（复用 CPU）
```

### 4.1 页锁定内存的正确用法

**不要**把 `CUDA_Host` 作为 CUDA backend 的默认 buft：

```cpp
// 错误：离散 GPU 的 supports_buft 只接受 CUDA0
ggml_backend_sched_new({cuda, cpu}, {cuda_host_buft, cpu_buft}, ...);  // 断言失败
```

源码（`ggml-cuda.cu:5601`）：

```cpp
return (ggml_backend_buft_is_cuda(buft) && buft->device == dev)
    || (integrated && ggml_backend_buft_is_cuda_host(buft));
```

离散 GPU 的 `integrated = false`，所以 `CUDA_Host` 不被接受为 scheduler 的 buft。

**正确用法**：把输入 tensor 单独分配到 `CUDA_Host`，然后让 scheduler 看到“已有 buffer”：

```cpp
// 1. 创建输入 context
struct ggml_init_params p = { ... no_alloc = true ... };
struct ggml_context * ctx_in = ggml_init(p);

struct ggml_tensor * input = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, K, M);

// 2. 在 pinned buffer 中分配（需要 #include "ggml-alloc.h"）
ggml_backend_buffer_t buf_in =
    ggml_backend_alloc_ctx_tensors_from_buft(ctx_in, ggml_backend_cuda_host_buffer_type());

// 3. 把 input 挂到图里
struct ggml_tensor * y = ggml_mul_mat(ctx, weight, input);
```

scheduler pass 1 会看到 `input->buffer`，用其 buft 所属 backend；由于 `CUDA_Host` 不是 `CUDA0`，pass 5 可能创建 copy 到 CUDA 显存。

### 4.2 自定义 buft 的最小实现

```cpp
static ggml_backend_buffer_t my_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = my_alloc(size);
    return ggml_backend_buffer_init(buft, my_buffer_iface, ptr, size);
}

static size_t my_alignment(ggml_backend_buffer_type_t buft) { return 256; }

static size_t my_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    return GGML_PAD(ggml_nbytes(t), 256);
}
```

`get_alloc_size` 必须 ≥ `ggml_nbytes`，且是 alignment 的倍数；否则 gallocr 的断言会失败。

---

## 5 调度策略扩展

### 5.1 backend 顺序

```cpp
ggml_backend_t backends[3] = { cuda0, cuda1, cpu };
```

pass 3 从 index 0 开始找，低索引优先。多卡时可以用顺序表达偏好。

### 5.2 `op_offload`

```cpp
ggml_backend_sched_new(backends, bufts, n, size, false, /*op_offload=*/true);
```

作用：当 op 的权重在 CPU（`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`）时，如果 GPU 支持且 `offload_op` 返回 true，则把 op 提升到 GPU。

CUDA 的 `offload_op`：

```cpp
return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
```

`op_offload_min_batch_size` 默认 32，可由 `GGML_OP_OFFLOAD_MIN_BATCH` 调整。

### 5.3 `parallel`

```cpp
ggml_backend_sched_new(..., /*parallel=*/true);
```

效果：

```text
n_copies = 4
为每个输入创建 4 份 copy
events[b][c] 创建
compute 时 round-robin 使用不同 copy
```

适用：连续多次 compute，且输入拷贝与计算可以重叠。

### 5.4 评估回调

```cpp
ggml_backend_sched_set_eval_callback(sched, callback, user_data);
```

作用：在 split 内按节点分段执行，允许宿主在中间读取 tensor（调试、可视化、逐层推理）。

代价：破坏整图提交，失去 CUDA graph 和 fusion 的收益。

### 5.5 自定义 split

scheduler 的 split 逻辑在 `ggml_backend_sched_split_graph`。如果要实现自定义策略（例如“每 N 层切一次以限制显存”），可以在 pass 3 之后、pass 5 之前插入自己的 backend id 赋值，或直接修改 pass 5 的切分条件。

不建议修改上游源码；更稳妥的方式是：

```text
1. 用多个 scheduler，每个负责一段
2. 用 eval callback 分段
3. 在 build_graph 阶段插入 GGML_OP_NONE 依赖节点
```

---

## 6 内存优化

### 6.1 标记输入/输出

```cpp
ggml_set_input(a);
ggml_set_input(b);
ggml_set_output(result);
```

效果：

```text
INPUT：
  - pass 1 默认判给 CPU（除非已有 buffer）
  - gallocr 提前分配，避免第一次使用前被其他节点的输出占用
  - n_copies>1 时创建多份 copy
  - 注意：INPUT 不阻止最后一个消费者之后的释放

OUTPUT：
  - gallocr 永不释放
```

**本例的隐含假设**：没有标记，`a`/`b` 在 node_0 后被 free。因为 alloc 后没有新分配，所以安全；真实模型应使用独立 buffer 或确保 alloc 后无新分配，结果则标记 OUTPUT。

### 6.2 预分配（reserve）

```cpp
ggml_backend_sched_reserve(sched, measure_graph);
```

或只测量大小：

```cpp
size_t sizes[2];
ggml_backend_sched_reserve_size(sched, measure_graph, sizes);
```

作用：提前做生命周期分析并申请 buffer，后续 `alloc_graph` 不再 realloc。

适用：

```text
1. 图结构稳定（LLM decode）
2. 想避免首次 compute 的分配抖动
3. 想控制峰值显存
```

### 6.3 对齐与填充

```text
CUDA alignment = 128
小 tensor 的填充比例高
```

优化：

```text
1. 合并小 tensor（减少对齐浪费）
2. 用 view 代替独立分配
3. 对大 tensor 影响可忽略
```

### 6.4 inplace

`ggml_op_can_inplace(op)` 为 true 的算子（ADD、MUL、SCALE 等）可以复用输入内存：

```text
条件：
  - 输入只有一个消费者
  - 输入不是 OUTPUT
  - 输出与输入 layout 相同
  - 输入由 gallocr 分配（不是外部 buffer）
```

效果：峰值显存降低。`MUL_MAT` 不满足 layout 条件，不能 inplace。

### 6.5 chunk

`GGML_VBUFFER_MAX_CHUNKS = 16`。当单块 buffer 达到后端上限时，tallocr 会开新 chunk。

优化：

```text
1. 如果后端有单块上限，确保 get_max_size 正确
2. 过多 chunk 会增加 cudaMalloc 次数和碎片
3. 可以通过 reserve_size 观察每个 chunk 的大小
```

---

## 7 kernel 优化

### 7.1 向量化

```text
F32：float2 / float4
F16：half2
BF16：nv_bfloat162
```

要求：

```text
1. 元素数能被向量宽度整除
2. 行步长能被向量宽度整除
3. data 指针对齐
```

MMVF 的谓词就是在检查这些条件。

### 7.2 block size 选择

MMVF 的策略：

```text
从 warp_size 开始，尝试 2*warp、3*warp ... 直到 256
选择使 K 迭代次数最少的 block
```

通用原则：

```text
K 小 → 小 block，减少空转
K 大 → 大 block，增加并行度
shared memory / 寄存器压力 → 限制上限
```

### 7.3 模板特化

```text
ncols_dst 作为模板参数 → 编译期展开 sumf[ncols_dst]
block_size 作为模板参数 → 编译期确定循环步长
类型作为模板参数 → 编译期选择指令
```

代价：模板实例膨胀，编译时间增加。需要限制组合（例如 ncols_dst 只展开 1..8）。

### 7.4 shared memory 与 warp reduce

```text
block_size == warp_size：只用 warp shuffle，不用 shared
block_size > warp_size：warp 内归约 → shared → 第一个 warp 再归约
```

优化：

```text
1. 尽量让 block_size 等于 warp_size，避免 shared 和 __syncthreads
2. 归约前尽量让每个线程都有有效数据
```

### 7.5 fusion

CUDA backend 的 fusion 在 `ggml_cuda_try_fuse` / `ggml_backend_cuda_graph_optimize`：

```text
MUL_MAT + ADD(bias) + GLU
MoE weighted reduction
top-k + softmax + argsort
gated delta net cache
```

新增 fusion 的步骤：

```text
1. 写匹配函数：检查节点序列、类型、layout
2. 写 fused kernel
3. 在 compute 循环中跳过被融合的节点
4. 在 graph_optimize 中添加 alloc_dep，防止中间 tensor 被提前复用
```

风险：

```text
1. 融合改变数值顺序 → 精度变化
2. alloc_dep 缺失 → 中间 tensor 被覆盖
3. 融合条件太宽 → 回退路径错误
```

### 7.6 CUDA graph

```bash
cmake -DGGML_CUDA_GRAPHS=ON
```

流程：

```text
第一次：warmup，直接执行
第二次：capture，录制 kernel
之后：cudaGraphInstantiate + cudaGraphLaunch
```

适用：

```text
1. kernel 多（>20）
2. 图结构稳定
3. 重复执行
```

不适用：

```text
1. 单 kernel（本例）
2. 图结构变化
3. 需要 CPU 中间干预
```

### 7.7 多 stream

```text
1. n_copies > 1：流水线
2. concurrent events：把独立分支放到不同 stream
3. 多 backend：多个 CUDA backend 实例，各自 stream
```

风险：

```text
1. 同步复杂
2. 显存竞争
3. 小 kernel 并发收益低
```

### 7.8 量化

```text
Q4_0 / Q8_0 / Q4_K / Q6_K → MMVQ（小 N）/ MMQ（大 N）
```

收益：

```text
1. 权重带宽降低 4-8 倍
2. decode 场景接近内存带宽上限
```

代价：

```text
1. 精度损失
2. 需要量化工具链
3. 某些 shape 需要 padding 到 512
```

---

## 8 调试与 profiling

### 8.1 环境变量

| 变量 | 作用 | 备注 |
|---|---|---|
| `GGML_SCHED_DEBUG=1/2` | 打印 split 与归属 | cause 列当前被编译掉 |
| `GGML_SCHED_DEBUG_REALLOC=1` | 检测意外 realloc | |
| `GGML_CUDA_FORCE_CUBLAS=1` | 强制 cuBLAS | |
| `GGML_CUDA_FORCE_MMQ=1` | 强制 MMQ | |
| `GGML_CUDA_DISABLE_FUSION=1` | 关闭 fusion | |
| `GGML_CUDA_NO_PINNED=1` | 禁用 pinned | |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` | managed memory | |
| `GGML_CUDA_DEVICES` | 虚拟设备 | |
| `GGML_OP_OFFLOAD_MIN_BATCH` | offload 阈值 | 默认 32 |

编译期开关：

```text
GGML_CUDA_DEBUG
GGML_ALLOCATOR_DEBUG
GGML_CUDA_GRAPHS
GGML_CUDA_USE_PDL
```

### 8.2 gdb

关键断点：

```gdb
break ggml_backend_sched_split_graph
break ggml_backend_sched_alloc_splits
break ggml_gallocr_reserve_n
break ggml_gallocr_alloc_graph
break ggml_backend_cuda_graph_compute
break ggml_cuda_mul_mat
break ggml_cuda_mul_mat_vec_f
break cublasSgemm
```

关键变量：

```gdb
p sched->splits[0]
p sched->node_backend_ids[0]
p sched->galloc->buf_tallocs[0]->chunks[0]->n_free_blocks
p sched->galloc->node_allocs[0].dst
p cgraph->nodes[0]->ne
p cgraph->nodes[0]->nb
```

### 8.3 nsys / ncu

```bash
# 时间线
nsys profile -o simple ./build-debug/bin/simple-backend

# kernel 级指标
ncu --set full -o simple ./build-debug/bin/simple-backend
```

关注：

```text
1. kernel 时间 vs memcpy 时间
2. SM 利用率、内存带宽
3. launch 间隔
4. 是否有隐式同步
```

### 8.4 分配器调试

```bash
cmake -DGGML_ALLOCATOR_DEBUG=ON ...
```

会打印：

```text
max_size[0] = ... MB: tensors: ...
allocating node_0 (48 bytes) - block ...
freeing leaf_0 at {chunk=0, offset=0}
```

---

## 9 实战改造示例

### 9.1 三层 MLP

```cpp
auto t0 = ggml_mul_mat(ctx, w0, x);   // [H, B]
auto t1 = ggml_silu(ctx, t0);
auto t2 = ggml_mul_mat(ctx, w1, t1);  // [O, B]
ggml_build_forward_expand(gf, t2);
```

预期：

```text
nodes = [t0, t1, t2]
leafs = [w0, w1, x]
splits = 1（全部 CUDA）
alloc：
  w0/w1 大，x 小，t0/t1/t2 中等
  t0 被 t1 消费后 free
  t1 被 t2 消费后 free
  峰值 ≈ max(w0+w1+x+t2, w0+w1+t0+t1)
```

观察点：

```gdb
p sched->galloc->buf_tallocs[0]->chunks[0]->max_size
p sched->galloc->buf_tallocs[0]->chunks[0]->n_free_blocks
```

### 9.2 跨 backend 输入

```cpp
// x 分配在 CPU buffer
auto y = ggml_mul_mat(ctx, w_gpu, x_cpu);
```

预期：

```text
pass 1：x 有 CPU buffer → backend 1
pass 3：node 选 CUDA
pass 5：src x 不在 CUDA → 创建 copy，n_inputs=1
compute_splits：CPU → CUDA 拷贝
```

观察：

```gdb
p sched->splits[0].n_inputs
p sched->splits[0].inputs[0]->name
p sched->hv_tensor_copies[...]
```

### 9.3 batch 从 3 变 4

```cpp
const int rows_B = 4;   // 原为 3
```

预期：

```text
ne11 = 4
should_use_mmvf(F32, 890, 4) = false   // 要求 <= 3
should_use_mmf(F32, ..., ne0=2) = false // 要求 ne0 % 32 == 0
MMVQ/MMQ 只支持量化 → false
→ cuBLAS
```

观察：

```gdb
break cublasCreate
break cublasSgemm
```

会看到：

```text
cublasCreate
cudaMalloc(4 MiB)   // workspace
cublasSgemm(m=4, n=4, k=2)
```

### 9.4 量化权重

```cpp
// w 用 Q4_0
auto y = ggml_mul_mat(ctx, w_q4, x);
```

预期：

```text
src0.type = Q4_0
ne11 小 → MMVQ
ne11 大 → MMQ
```

观察：

```gdb
break ggml_cuda_mul_mat_vec_q
break ggml_cuda_mul_mat_q
```

---

## 10 接口源码逐行：扩展点到底长什么样

前九节讲“怎么改”，本节讲“为什么接口是这样设计的”。逐字段读五套 iface，理解每个字段被谁调用、不实现会怎样。

### 10.1 `ggml_backend_reg_i`

`src/ggml-backend-impl.h:230`

```cpp
struct ggml_backend_reg_i {
    const char * (*get_name)(ggml_backend_reg_t reg);

    // enumerate available devices
    size_t             (*get_device_count)(ggml_backend_reg_t reg);
    ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

    // (optional) get a pointer to a function in the backend
    void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
};
```

| 字段 | 谁调用 | 必须实现 | 不实现的后果 |
|---|---|---|---|
| `get_name` | `register_backend` 打日志、`ggml_backend_reg_by_name` | 是 | 日志为 NULL，按名字查找失败 |
| `get_device_count` | `register_backend` 循环、`ggml_backend_dev_count` | 是 | 设备不被注册 |
| `get_device` | `register_device`、`ggml_backend_reg_dev_get` | 是 | 崩溃 |
| `get_proc_address` | 宿主通过 `ggml_backend_reg_get_proc_address` 取扩展函数 | 否 | 返回 NULL，只影响高级功能 |

**为什么 `get_device_count` 和 `get_device` 分开**：registry 需要先知道数量才能分配数组/循环；有些后端（如 RPC）枚举设备代价高，可以缓存。

**调用链**：

```text
register_backend(reg)
  → ggml_backend_reg_dev_count(reg)      // get_device_count
  → for i in 0..count:
        ggml_backend_reg_dev_get(reg, i) // get_device
        register_device(dev)
```

### 10.2 `ggml_backend_device_i`

`src/ggml-backend-impl.h:176`

```cpp
struct ggml_backend_device_i {
    const char * (*get_name)(ggml_backend_dev_t dev);
    const char * (*get_description)(ggml_backend_dev_t dev);

    void         (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);

    enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

    void (*get_props)(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);

    ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, const char * params);

    ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

    ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

    ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

    bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

    bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

    bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

    ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
    void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
    void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
};
```

按“被谁调用”分组：

**① 身份类（registry 用）**

```cpp
get_name / get_description / get_type
```

`register_device` 打印 `CUDA0 (NVIDIA GeForce RTX 4090)`；`ggml_backend_dev_by_type` 靠 `get_type` 选设备。

**② 资源类（宿主查询用）**

```cpp
get_memory / get_props
```

`get_memory` 返回 free/total，用于显存预算；`get_props` 返回 caps（async、host_buffer、events、mmap）。

**③ 创建类（scheduler 用）**

```cpp
init_backend          → 创建 ggml_backend 实例
get_buffer_type       → scheduler 的默认 buft
get_host_buffer_type  → pinned host buffer
buffer_from_host_ptr  → mmap 模型
```

**④ 决策类（scheduler 的 pass 3/5 用）**

```cpp
supports_op     → 这个 op 能不能跑
supports_buft   → 这个 buft 能不能用
offload_op      → 权重在 CPU 时要不要提升到 GPU
```

这三个是**调度质量的核心**。太严格 → 大量 fallback；太宽松 → 运行时断言。

**⑤ 同步类（scheduler 的并行模式用）**

```cpp
event_new / event_free / event_synchronize
```

`n_copies > 1` 时，每个 backend × copy 创建一个 event，用于跨 copy 流水线同步。

**CUDA 的 `supports_op` 逐行**（`ggml-cuda.cu:5136`）：

```cpp
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // 1. 检查所有源是否在本设备
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx =
                (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false;   // 非连续，不支持
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    ...
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        ...
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 5140-5147 | 多卡源检查 | 张量在其他 CUDA 设备上时，本设备不能直接算（除非 peer copy） |
| 5198-5202 | `nb[0] != element_size` | **非连续布局不支持**，kernel 假设行内连续 |
| 5203-5205 | F16 `b` 要求 `a` 也是 F16 | 混合精度的 kernel 未实现 |
| 5207-5209 | `MUL_MAT_ID` + `PREC_F32` | 该组合不支持 |
| 5211-5244 | 类型白名单 | 只列出有 kernel 的类型 |

**注意它没有检查的**：`ne11`、`cc`、对齐。这些是**性能/路径选择**，放在 `should_use_*` 里；`supports_op` 只回答“能不能算”。

**这是 ggml 的一个重要分工**：

```text
supports_op  → 正确性（能不能算）
should_use_* → 性能（走哪条路径）
```

### 10.3 `ggml_backend_i`

`src/ggml-backend-impl.h:121`

```cpp
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);

    void (*free)(ggml_backend_t backend);

    // (optional) asynchronous tensor data access
    void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    void (*set_tensor_2d_async)(...);
    void (*get_tensor_2d_async)(...);
    bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    // (optional) complete all pending operations
    void (*synchronize)(ggml_backend_t backend);

    // (optional) graph plans (not used currently)
    ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
    void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
    enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

    // compute graph (always async if supported by the backend)
    enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

    // (optional) event synchronization
    void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
    void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

    // (optional) sort/optimize the nodes in the graph
    void (*graph_optimize)(ggml_backend_t backend, struct ggml_cgraph * cgraph,
                           struct ggml_backend_graph_optimize_params * params);
};
```

| 字段 | 谁调用 | 必须 | 说明 |
|---|---|---|---|
| `get_name` | 日志 | 是 | `"CUDA0"` |
| `free` | `ggml_backend_free` | 是 | 销毁 backend 与 context |
| `set_tensor_async` | `ggml_backend_tensor_set_async` | 否 | 无则降级为同步 |
| `get_tensor_async` | `ggml_backend_tensor_get_async` | 否 | 同上 |
| `cpy_tensor_async` | scheduler 跨 backend copy | 否 | **返回 false 时降级** |
| `synchronize` | `ggml_backend_synchronize` | 是（若支持异步） | 等待所有工作完成 |
| `graph_plan_*` | 当前未使用 | 否 | 历史遗留 |
| `graph_compute` | scheduler 提交 split | 是 | 核心 |
| `event_record` / `event_wait` | `n_copies > 1` | 否 | 流水线同步 |
| `graph_optimize` | `split_graph` 之后 | 否 | fusion 的钩子 |

**`cpy_tensor_async` 的返回值语义**（实验二验证）：

```cpp
static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                               const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ...
}
```

**两个 `return false`**：只要 src 或 dst 有一端不是 CUDA，就拒绝。scheduler 收到 false 后：

```cpp
if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(...)) {
    ggml_backend_synchronize(input_backend);
    ggml_backend_synchronize(split_backend);
    ggml_backend_tensor_copy(input, input_cpy);   // 同步降级
}
```

这就是实验二里 `cuda_cpy_tensor_async` 被调用、返回 false、然后走 `tensor_copy` 的完整原因。

**`graph_optimize` 的设计**（fusion 钩子）：

```cpp
void (*graph_optimize)(ggml_backend_t backend, struct ggml_cgraph * cgraph,
                       struct ggml_backend_graph_optimize_params * params);
```

`params->add_alloc_dep` 是关键：fusion 后中间 tensor 不再被图引用，但 kernel 仍然需要它们；通过 `add_alloc_dep` 告诉 gallocr “别提前释放”。

### 10.4 `ggml_backend_buffer_type_i`

`src/ggml-backend-impl.h:17`

```cpp
struct ggml_backend_buffer_type_i {
    const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
    size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
    size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
    size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
};
```

| 字段 | 谁调用 | 影响 |
|---|---|---|
| `get_name` | 日志、`ggml_backend_sched_print_assignments` | 调试可见性 |
| `alloc_buffer` | `ggml_vbuffer_alloc` | 真正申请内存 |
| `get_alignment` | `ggml_dyn_tallocr_new` | **决定所有 tensor 的偏移** |
| `get_max_size` | `ggml_dyn_tallocr_new` | 单块上限，超过则开新 chunk |
| `get_alloc_size` | `ggml_dyn_tallocr_alloc`、`ggml_gallocr_init_tensor` | **决定每个 tensor 占多少字节** |
| `is_host` | `ggml_backend_scheduler_buffer_supported`、跨 backend copy | 判断是否 host 内存 |

**`get_alignment` 的连锁影响**：

```text
get_alignment → ggml_dyn_tallocr_new(alignment)
              → ggml_dyn_tallocr_alloc: size = aligned_offset(NULL, size, alignment)
              → chunk->max_size 变大
              → cudaMalloc 的 size 变大
```

本例 128 B 对齐让 104 B 有效数据变成 384 B 分配。如果把 alignment 改成 32，会变成 128 B；但 float4/MMA 的对齐要求就不满足了。

**`get_alloc_size` 的量化补齐**（`ggml-cuda.cu:911`）：

```cpp
static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 913-915 | FLASH_ATTN 特殊处理 | FA kernel 需要额外 workspace |
| 916 | `ne0 = tensor->ne[0]` | 行内元素数 |
| 919-923 | 量化补齐到 512 | **MMQ/MMVQ 按 512 元素行块读取**，最后一行不足会越界 |
| 920 | `nb[0] == element_size` | 量化块必须连续 |

**这就是为什么量化模型的显存占用会略大于 `ggml_nbytes` 之和**。

### 10.5 `ggml_backend_buffer_i`

`src/ggml-backend-impl.h:46`

```cpp
struct ggml_backend_buffer_i {
    void         (*free_buffer)  (ggml_backend_buffer_t buffer);
    void *       (*get_base)     (ggml_backend_buffer_t buffer);
    enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    void         (*memset_tensor)(...);
    void         (*set_tensor)   (...);
    void         (*get_tensor)   (...);
    void         (*set_tensor_2d)(...);
    void         (*get_tensor_2d)(...);
    bool         (*cpy_tensor)   (...);
    void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
    void         (*reset)        (ggml_backend_buffer_t buffer);
};
```

| 字段 | 谁调用 | 必须 |
|---|---|---|
| `free_buffer` | `ggml_backend_buffer_free` | 是 |
| `get_base` | `ggml_vbuffer_tensor_alloc`、`ggml_backend_tensor_alloc` | 是 |
| `init_tensor` | `ggml_backend_tensor_alloc` | 否 |
| `set_tensor` | `ggml_backend_tensor_set` | 是 |
| `get_tensor` | `ggml_backend_tensor_get` | 是 |
| `cpy_tensor` | `ggml_backend_tensor_copy`（同 backend 内） | 否 |
| `clear` | 调试/清零 | 是 |
| `reset` | `ggml_vbuffer_reset` | 否 |

**为什么 `set_tensor` 在 buffer 上而不是 backend 上**：

```text
buffer 知道“这块内存是什么类型、怎么读写”
backend 知道“这条 stream 怎么排队”

tensor_set 是同步语义，不需要 stream 管理
→ 放在 buffer 上最自然
```

CUDA 的两个实现（`ggml-cuda.cu:785` / `793`）分别做 H2D 和 D2H，都用 `cudaStreamPerThread` + 立即同步。

### 10.6 模板分派：以 MMVF 为例

扩展 kernel 路径时，最常改的是模板实例。理解 MMVF 的分派结构：

```text
ggml_cuda_mul_mat_vec_f                      mmvf.cu:634
  └── mul_mat_vec_f_cuda<float>              mmvf.cu:610
       └── mul_mat_vec_f_cuda_switch_ncols_dst<float, float>  mmvf.cu:513
            └── switch (ncols_dst) {
                  case 1: launch_mul_mat_vec_f_cuda<T, type_acc, 1>  :548
                  case 2: launch_mul_mat_vec_f_cuda<T, type_acc, 2>  :555
                  case 3: launch_mul_mat_vec_f_cuda<T, type_acc, 3>  :562
                  case 4: ...                                        :569
                  case 5..8: ...
                  default: 通用路径（运行时 ncols_dst）
                }
```

`launch_mul_mat_vec_f_cuda`（418）再做一次分派：

```text
switch (block_size_best) {
    case  32: mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, ...>
    case  64: ... 64 ...
    ...
    case 256: ... 256 ...
}
```

`mul_mat_vec_f_switch_fusion`（386）再做一次：

```text
if constexpr (ncols_dst == 1) {
    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, ...);
        return;
    }
}
ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, ...);
```

**三层模板特化的目的**：

| 层 | 模板参数 | 收益 |
|---|---|---|
| 1 | `ncols_dst` | `sumf[ncols_dst]` 编译期定长，寄存器分配 |
| 2 | `block_size` | 循环步长常量，展开优化 |
| 3 | `has_fusion` | 去掉 bias/gate 分支 |

**代价**：实例数量 = `ncols_dst(8) × block_size(8) × has_fusion(2) × 类型(3)`，编译时间和二进制体积都会涨。所以 ggml 用 `if constexpr` 和 `default` 兜底来控制。

**扩展时的取舍**：

```text
如果新路径只在少数形态下用 → 用运行时分支，不模板化
如果新路径是热点 → 模板化，但限制组合（例如 ncols_dst 只展开 1..4）
```

### 10.7 新增后端时最容易踩的三个坑

**① `supports_buft` 写错导致断言**

```cpp
static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev)
        || (integrated && ggml_backend_buft_is_cuda_host(buft));
}
```

离散 GPU 的 `integrated = false`，所以只接受 `CUDA0` buft。如果把 `CUDA_Host` 当成默认 buft，`sched_new` 的断言会失败。

**② `get_alignment` 与实际分配不匹配**

如果 `get_alignment` 返回 128，但 `alloc_buffer` 返回的指针只有 16 对齐，kernel 的 `float4` 加载会崩溃。

**③ `get_alloc_size` 小于 `ggml_nbytes`**

`ggml_gallocr_init_tensor` 有断言：

```cpp
assert(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
```

如果 `get_alloc_size` 算小了，分配时按小值，使用时按 `ggml_nbytes` 写，就会越界。

### 10.8 扩展检查表（源码级）

```text
[ ] reg_i: get_name / get_device_count / get_device 全部实现
[ ] device_i: get_type 返回正确的枚举（GPU/IGPU/CPU/ACCEL/META）
[ ] device_i: supports_op 只做正确性判断，不做性能判断
[ ] device_i: supports_buft 与实际 buft 一致
[ ] backend_i: graph_compute 是异步的（返回时 kernel 可能还在跑）
[ ] backend_i: synchronize 真的等待所有工作
[ ] backend_i: cpy_tensor_async 失败时返回 false（不要 abort）
[ ] buft_i: get_alignment 是 2 的幂
[ ] buft_i: get_alloc_size >= ggml_nbytes
[ ] buft_i: get_max_size 正确（单块上限）
[ ] buffer_i: get_base 返回分配时的基址
[ ] buffer_i: set_tensor/get_tensor 处理 offset
[ ] buffer_i: free_buffer 释放所有资源
```

---

## 11 改造检查清单

### 新增算子

```text
[ ] 枚举加在 GGML_OP_COUNT 之前
[ ] NAME/SYMBOL 用指定初始化器
[ ] builder 设置 op/src/op_params
[ ] CPU fallback 实现
[ ] CUDA compute_forward case
[ ] CUDA supports_op case
[ ] 类型/连续性断言
[ ] 数值对拍测试
[ ] GGML_SCHED_DEBUG 确认节点归属
```

### 新增后端

```text
[ ] reg / device / backend / buft / buffer 五套 iface
[ ] supports_op 粒度合理
[ ] supports_buft 正确
[ ] alignment / alloc_size 正确
[ ] graph_compute 异步 + synchronize
[ ] 注册（静态或动态）
[ ] CPU 作为最后一个 backend
[ ] 跨 backend copy 测试
```

### 性能优化

```text
[ ] 确认瓶颈（计算 / 带宽 / launch / 拷贝）
[ ] 权重常驻显存
[ ] 输入 pinned + 异步
[ ] 合适的 kernel 路径
[ ] fusion 是否命中
[ ] CUDA graph 是否适用
[ ] 显存峰值是否可接受
[ ] 数值精度是否达标
```

---

## 12 最后一句话

`simple-backend` 的价值不在于它做了什么，而在于它**没有隐藏任何一层**：

```text
1 个 MUL_MAT = registry + device + backend + buft + buffer
             + scheduler + gallocr + CUDA context + kernel 选择
```

把这 9 层逐一拆开，你就具备了在任意一层做扩展和优化的能力：

```text
想加算子   → §1
想加路径   → §2
想加硬件   → §3
想加内存   → §4
想调调度   → §5
想省内存   → §6
想提性能   → §7
想定位问题 → §8
```
