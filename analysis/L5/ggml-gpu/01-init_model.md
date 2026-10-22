# 01 · init_model：CUDA 后端体系与调度器的构造

> 源码基线：`/home/ggml`，commit `353b63b`
>
> 对应源码：`examples/simple/simple-backend.cpp:57-65`、`src/ggml-backend-reg.cpp`、`src/ggml-backend.cpp:1848-1975`、`src/ggml-alloc.c:498-532`、`src/ggml-cuda/ggml-cuda.cu`
>
> 本文所有数值来自 gdb 断点实测（见 [04-gdb-trace.md](04-gdb-trace.md) §1、§2）。

---

## 0 这一阶段到底在做什么

```cpp
void init_model(simple_model & model) {
    ggml_log_set(ggml_log_callback_default, nullptr);   // ① 日志出口
    ggml_backend_load_all();                            // ② 动态后端发现（本环境空转）
    model.backend     = ggml_backend_init_best();       // ③ 创建 CUDA0 backend
    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr); // ④ 创建 CPU backend
    ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
    model.sched = ggml_backend_sched_new(backends, nullptr, 2,   // ⑤ 创建 scheduler
                                         GGML_DEFAULT_GRAPH_SIZE, false, true);
}
```

五步对应五个设计问题：

| 步骤 | 设计问题 | 答案的位置 |
|---|---|---|
| ① | 日志如何从 ggml 内部流出 | 函数指针 + `stderr` |
| ② | 后端是“链接进来”还是“运行时加载” | 两条路径并存，本环境走静态注册 |
| ③ | 如何从多个设备中选一个，并创建执行实例 | device → backend，资源惰性 |
| ④ | 为什么必须有 CPU backend | scheduler 的 fallback 约定 |
| ⑤ | 调度器需要预先知道什么 | 后端数组、buffer type、图容量 |

---

## 1 五层对象模型（先建立坐标系）

后面所有断点值都要放回这张图里看：

```text
┌─────────────────────────────────────────────────────────────────────┐
│ ggml_backend_registry (函数内 static)                               │
│   backends: [ CUDA reg, CPU reg ]                                   │
│   devices : [ CUDA0, CPU ]                                          │
└───────────────┬─────────────────────────────────────────────────────┘
                │ ggml_backend_dev_by_type(GPU)
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ggml_backend_device (CUDA0)                                         │
│   iface: init_backend / get_buffer_type / supports_op / get_props   │
│   context: ggml_backend_cuda_device_context { device=0, name, ... } │
└───────────────┬─────────────────────────────────────────────────────┘
                │ device->iface.init_backend()
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ggml_backend (model.backend)                                        │
│   guid / iface / device / context                                   │
│   context: ggml_backend_cuda_context (4336 B)                       │
│       streams[8][8] / cublas_handles[8][8] / pools[8][8] ...        │
└───────────────┬─────────────────────────────────────────────────────┘
                │ ggml_backend_get_default_buffer_type()
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ggml_backend_buffer_type (CUDA0)                                    │
│   alignment = 128                                                   │
│   alloc_buffer / get_alloc_size / get_max_size / is_host            │
└───────────────┬─────────────────────────────────────────────────────┘
                │ ggml_backend_buft_alloc_buffer(size)
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ggml_backend_buffer (384 B)                                         │
│   buft / iface / context{device, dev_ptr} / size / usage            │
└─────────────────────────────────────────────────────────────────────┘
```

再叠两个决策层：

```text
ggml_backend_sched  ── 拓扑 → backend 归属 → split
ggml_gallocr        ── 生命周期 → buffer type → offset
```

### 1.1 为什么“设备”和“backend”要分开

- `device` 是**能力描述**：它回答“我支持哪些 op、我有多少显存、我的默认内存类型是什么”。它没有 stream，也不持有执行状态。
- `backend` 是**执行实例**：它回答“这次计算用哪条 stream、哪个 cuBLAS handle、哪个内存池”。同一个 device 可以有多个 backend 实例。

分开的直接收益：

```text
1. scheduler 只需要 device 的 supports_op 做决策，不需要创建 backend
2. 多线程可以各自创建 backend（各自 stream），共享同一个 device
3. backend 释放（destroy stream）不影响 device 的注册信息
```

### 1.2 为什么“buffer type”和“buffer”要分开

`buffer type` 是**内存策略**，`buffer` 是**内存实例**。

```text
CUDA0 buft        : alignment=128, alloc_buffer→cudaMalloc, get_alloc_size→量化行补齐
CUDA_Host buft    : alignment=32（复用 CPU），alloc_buffer→cudaMallocHost（页锁定）
CPU buft          : alignment=32, alloc_buffer→aligned malloc
```

scheduler 在判断“node 能不能放在 backend b”时，用的是 **buft 兼容性**：

```cpp
ggml_backend_sched_buffer_supported(sched, src, b)
  → buft = src->buffer ? src->buffer->buft : sched->bufts[backend_id(src)]
  → ggml_backend_supports_buft(sched->backends[b], buft)
```

而不是比较 backend 指针。这为“同一个 backend 支持多种内存”留出了空间。

---

## 2 日志出口：`ggml_log_set()`

```cpp
static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}
```

设计要点：

- ggml 内部不直接 `printf`，而是通过函数指针 `ggml_log_internal` 分发，允许宿主接管日志（例如 llama.cpp 把它接到自己的日志系统）。
- `level` 被忽略，意味着 Debug 构建下 `GGML_LOG_DEBUG` 也会输出。这就是为什么 `ggml_gallocr_needs_realloc` 这类调试日志可见。
- `fflush(stderr)` 保证日志与崩溃/断点顺序一致，便于调试。

---

## 3 `ggml_backend_load_all()`：本环境为什么“什么都没做”

### 3.1 动态后端的设计

```cpp
void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas",   silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann",   silent, dir_path);
    ggml_backend_load_best("cuda",   silent, dir_path);
    ggml_backend_load_best("hip",    silent, dir_path);
    ggml_backend_load_best("metal",  silent, dir_path);
    ggml_backend_load_best("rpc",    silent, dir_path);
    ggml_backend_load_best("sycl",   silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu",silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("hexagon",silent, dir_path);
    ggml_backend_load_best("musa",   silent, dir_path);
    ggml_backend_load_best("openvino",silent, dir_path);
    ggml_backend_load_best("cpu",    silent, dir_path);
}
```

`ggml_backend_load_best(name, silent, dir)` 的匹配规则：

```text
文件名前缀：libggml-<name>-
文件扩展名：.so
例：libggml-cuda-89.so、libggml-cpu-haswell.so
```

找到候选后：

```text
dlopen → dlsym("ggml_backend_score") → score() → 选分数最高者
→ dlsym("ggml_backend_reg") → 注册
```

**为什么用 score**：同一个后端可能有多个编译变体（CPU 的 AVX2/AVX512、CUDA 的不同 arch），运行时根据当前机器选择最优变体，而不是在编译期写死。

搜索路径（`user_search_path == nullptr` 时）：

```text
1. GGML_BACKEND_DIR（编译期定义，若有）
2. 可执行文件所在目录
3. 当前工作目录
```

### 3.2 本环境的实际情况

产物：

```text
/home/ggml/build-debug/src/ggml-cuda/libggml-cuda.so.0.25.3
/home/ggml/build-debug/src/libggml-cpu.so.0.25.3
```

两个不匹配：

```text
1. 目录：文件在 build-debug/src/，搜索路径是 build-debug/bin/ 和 /home/ggml
2. 命名：搜索前缀是 libggml-cuda-，产物是 libggml-cuda.so（没有 -<score> 后缀）
   版本化文件 libggml-cuda.so.0.25.3 的 extension 是 .3，也不等于 .so
```

断点：

```text
=== [BP1] ggml_backend_load_all ===
#0  ggml_backend_load_all () at ggml-backend-reg.cpp:575
#1  init_model (model=...) at simple-backend.cpp:58
```

进入前后都没有任何 `loaded ... backend from ...` 日志。**`load_all()` 是一次完整的空转**。

### 3.3 为什么空转不影响 CUDA 可用

构建配置：

```text
GGML_BACKEND_DL = OFF
BUILD_SHARED_LIBS = ON
```

`GGML_BACKEND_DL=OFF` 意味着 backend 不是插件，而是**编译期链接进 `libggml.so` 的依赖**：

```text
simple-backend
  → libggml.so
      → libggml-base.so
      → libggml-cpu.so
      → libggml-cuda.so
```

`ldd` 可以直接看到 `libggml-cuda.so.0`。因此 CUDA 的注册不依赖 `dlopen`，而是 `ggml_backend_registry` 构造函数里的静态调用。

### 3.4 两条路径的取舍

| | 静态链接（本环境） | 动态加载（`GGML_BACKEND_DL=ON`） |
|---|---|---|
| 启动 | 无文件查找 | 扫描目录、dlopen、score |
| 体积 | 所有后端都链进来 | 按需加载 |
| 多变体 | 编译期选择 | 运行时按 score 选择 |
| 部署 | 简单 | 需要管理 `.so` 布局 |
| 调试 | 符号直接可用 | 需要 `set breakpoint pending on` |

断点技巧：动态加载时，函数符号在 `run` 之后才出现，gdb 必须 `set breakpoint pending on`。

---

## 4 registry：惰性构造与注册顺序

### 4.1 函数内 static

```cpp
static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}
```

三个设计考量：

1. **避免静态初始化顺序问题**：跨翻译单元/共享库的全局对象构造顺序未定义。函数内 static 在首次调用时构造，顺序确定。
2. **惰性**：如果程序从不使用 ggml backend，就不构造 registry。
3. **线程安全**：C++11 起，函数内 static 的初始化是线程安全的（magic static）。

析构在进程退出时执行。注意 `~ggml_backend_registry()` 里对动态库 handle 做了 `release()`（故意不卸载），因为 backend 线程可能仍在运行。

### 4.2 构造顺序

`src/ggml-backend-reg.cpp:119`：

```cpp
ggml_backend_registry() {
#ifdef GGML_USE_CUDA
    register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
    register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
    register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
    ...
#endif
#ifdef GGML_USE_CPU
    register_backend(ggml_backend_cpu_reg());
#endif
}
```

顺序是**源码书写顺序**：CUDA 在前，CPU 在后。日志验证：

```text
register_backend: registered backend CUDA (1 devices)
register_device: registered device CUDA0 (NVIDIA GeForce RTX 4090)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (Intel(R) Xeon(R) Platinum 8470Q)
```

**注册顺序不等于调度优先级**。调度优先级由 `ggml_backend_sched_new(backends, ...)` 的数组顺序决定：

```cpp
ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
//                             CUDA0 (index 0)   CPU (index 1)
```

所以即使 registry 里 CPU 先注册，scheduler 仍然把 CUDA 放在 index 0。

### 4.3 为什么日志出现在 `init_best` 而不是 `load_all`

`ggml_backend_dev_by_type()`（`ggml-backend-reg.cpp:355`）：

```cpp
ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {   // ← 首次调用 get_reg()
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}
```

`ggml_backend_load_all()` 在文件搜索阶段就返回了，没有触碰 registry；`init_best()` 第一次调用 `ggml_backend_dev_count()`，才触发 `get_reg()` 构造。

断点顺序：

```text
[BP1] load_all        无注册日志
[BP2] init_best       随后出现 CUDA/CPU 注册日志
[BP4] cuda_init       device=0
```

### 4.4 `register_backend()` 与 `register_device()`

```cpp
void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
    if (!reg) return;

    for (auto & entry : backends) {
        if (entry.reg == reg) return;   // 去重
    }

    GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
        __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));

    backends.push_back({ reg, std::move(handle) });

    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
        register_device(ggml_backend_reg_dev_get(reg, i));
    }
}
```

两个细节：

- `ggml_backend_reg_dev_count(reg)` 会调用 CUDA reg 的 `get_device_count()`，这正是触发 `ggml_cuda_init()` 的位置。
- 去重基于 `reg` 指针。静态注册 + 动态加载同一个库时，若两者返回不同 `reg` 实例，仍可能重复；这是 `GGML_BACKEND_DL` 与静态链接混用时的已知风险。

`register_device()` 把 device 追加到 `devices` 数组，并打印：

```text
register_device: registered device CUDA0 (NVIDIA GeForce RTX 4090)
```

---

## 5 CUDA 设备发现：`ggml_cuda_info()`

### 5.1 惰性单例

```cpp
const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}
```

`ggml_cuda_init()` 只在第一次调用时执行：

```text
cudaGetDeviceCount
→ 对每个物理设备 cudaGetDeviceProperties
→ 填充 info.devices[id]：name、cc、nsm、smpb、warp_size、totalGlobalMem、vmm
→ 计算 default_tensor_split
→ 打印 "ggml_cuda_init: found N CUDA devices ..."
```

### 5.2 断点日志逐字段解读

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24091 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24091 MiB
```

| 字段 | 值 | 含义 | 后续影响 |
|---|---|---|---|
| `found N devices` | 1 | 物理设备数（或虚拟设备数） | registry 为每个设备建一个 device |
| `Total VRAM` | 24091 MiB | 所有物理设备总显存 | 日志展示 |
| `compute capability` | 8.9 | 显示值 | 内部 `cc = 100*major + 10*minor = 890` |
| `VMM` | yes | 支持虚拟内存管理 | compute pool 可走 `cuMemCreate/cuMemMap` |
| `VRAM` | 24091 MiB | 单设备显存 | 分配上限、tensor split |

内部 `cc` 的计算（`ggml-cuda.cu:352`）：

```cpp
info.devices[id].cc = 100*prop.major + 10*prop.minor;   // 8.9 → 890
```

`cc` 的量纲与常量一致：

```text
GGML_CUDA_CC_AMPERE        = 800
GGML_CUDA_CC_ADA_LOVELACE  = 890
```

后续 `should_use_mmvf(F32, 890, ...)`、`ampere_mma_available(890)` 都用这个值。

### 5.3 设备信息的用途

```cpp
info.devices[id].integrated = false;   // 独立显卡
info.devices[id].nsm        = prop.multiProcessorCount;
info.devices[id].smpb       = prop.sharedMemPerBlock;
info.devices[id].warp_size  = prop.warpSize;   // 32
info.devices[id].vmm        = !!device_vmm;
info.devices[id].default_tensor_split[id] = ...
```

- `nsm`（SM 数）用于决定 grid 上限和 split-K 策略。
- `smpb`（每 block 共享内存）用于判断 kernel 能否使用更大的 tile。
- `warp_size` 用于 MMVF 的 block 大小、warp reduction。
- `vmm` 用于 compute pool 选择实现。
- `integrated` 影响 buffer 断言（APU 的 host-visible buffer）。

### 5.4 虚拟设备

```text
GGML_CUDA_DEVICES=2  → 在 1 张物理卡上虚拟出 2 个设备
```

代码里：

```cpp
info.devices[id].physical_device = id % info.physical_device_count;
info.devices[id].virtual_index  = physical_share_count[...]++;
```

这用于测试多卡逻辑（tensor split、NCCL）而不需要多卡。日志会额外打印：

```text
emulating N virtual device(s) on M physical device(s) (GGML_CUDA_DEVICES)
```

---

## 6 `ggml_backend_init_best()`：从设备到执行实例

### 6.1 选择顺序

```cpp
ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) return nullptr;
    return ggml_backend_dev_init(dev, nullptr);
}
```

```text
GPU  →  IGPU  →  CPU
```

为什么 IGPU 单独一档：集成 GPU 与 CPU 共享物理内存，buffer type / host buffer / `integrated` 语义都不同，不能简单归为 GPU。

### 6.2 调用栈（断点实测）

```text
=== [BP4] ggml_backend_cuda_init device=0 ===
#0  ggml_backend_cuda_init (device=0)              at ggml-cuda.cu:5843
#1  ggml_backend_cuda_device_init_backend          at ggml-cuda.cu:5122
#2  ggml_backend_dev_init                          at ggml-backend.cpp:612
#3  ggml_backend_init_best                         at ggml-backend-reg.cpp:389
#4  init_model                                     at simple-backend.cpp:60
#5  main                                           at simple-backend.cpp:118
```

三层：

```text
ggml_backend_dev_init()       通用分发：device->iface.init_backend(device, params)
  → CUDA device 实现           ggml_backend_cuda_device_init_backend
    → 创建 backend 实例        ggml_backend_cuda_init(0)
```

### 6.3 backend 实例的组成

```cpp
ggml_backend_t ggml_backend_cuda_init(int device) {
    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}
```

```cpp
struct ggml_backend {
    ggml_guid_t guid;
    struct ggml_backend_i iface;
    ggml_backend_dev_t device;
    void * context;
};
```

- `guid`：类型标识，用于 `ggml_backend_is_cuda()` 之类的快速判断。
- `iface`：函数表。后续 `graph_compute`、`set_tensor_async`、`synchronize`、`free` 全部从这里分发。
- `device`：指回 registry 中的 CUDA0 device。
- `context`：`ggml_backend_cuda_context *`，4336 B。

### 6.4 资源惰性：本次断点最值得记住的证据

`ggml_backend_cuda_context` 创建时只设置 `device` 和 `name`：

```cpp
struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    void * cublas_workspaces[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    size_t cublas_workspace_sizes[GGML_CUDA_MAX_DEVICES] = {0};

    int curr_stream_no = 0;

    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];
    ...
};
```

断点在 `ggml_cuda_mul_mat_vec_f` 入口（此时 kernel 即将 launch）：

```text
=== [J1] at mul_mat_vec_f: device=0
stream[dev][0]=0x5624e018e8c0
cublas_handle[dev][0]=(nil)
pool[dev][0]=(nil)
copy_event=(nil) curr_stream_no=0
```

也就是说：

```text
stream        已创建（在 alloc_splits 的同步步骤中）
cuBLAS handle 从未创建
memory pool   从未创建
event         从未创建（n_copies=1）
```

**为什么这样设计**：

1. **cuBLAS 初始化昂贵**：`cublasCreate` + 4 MiB workspace（Hopper 为 32 MiB）。小图/非 cuBLAS 路径不应付这个成本。
2. **pool 按需**：只有需要临时 buffer 的 kernel（类型转换、split-K、cuBLAS fallback）才创建 pool。
3. **stream 按需**：每个 (device, stream_no) 一条，最多 8×8；全部预创建会浪费资源。
4. **event 按需**：只有 `n_copies > 1` 的流水线模式才需要跨 copy 同步。

### 6.5 默认 buffer type

```cpp
sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
```

CUDA：

```cpp
static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cuda_buffer_type(ctx->device);
}
```

`ggml_backend_cuda_buffer_type(0)` 返回静态数组元素：

```text
name       = "CUDA0"
alignment  = 128
alloc_buffer → ggml_backend_cuda_buffer_type_alloc_buffer
get_alloc_size → 量化类型按 MATRIX_ROW_PADDING=512 补齐
get_max_size → 默认 SIZE_MAX
is_host    → NULL（false）
```

`get_alloc_size` 的关键逻辑：

```cpp
size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
    ? ggml_cuda_flash_attn_ext_get_alloc_size(...)
    : ggml_nbytes(tensor);

if (ggml_is_quantized(tensor->type)) {
    if (ne0 % MATRIX_ROW_PADDING != 0) {
        size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
    }
}
return size;
```

为什么量化要补齐：MMQ/MMVQ kernel 会按 512 元素的行块读取，最后一行若不足会越界；补齐到 512 的倍数保证安全。本例 F32 不触发，所以 `get_alloc_size == ggml_nbytes`。

### 6.6 alignment = 128 的原因

CUDA buft 返回 128。这个值不是随便选的：

```text
1. float4（16 B）向量化加载要求 16 B 对齐；128 是其倍数
2. 量化块（Q4_0 32 元素 × 18 B = 18 B，QK_K 256 元素块）需要更大对齐
3. tensor core / MMA 的 ldmatrix 对行首地址有 16 B / 32 B / 128 B 对齐要求
4. cudaMalloc 本身返回 256 B 对齐，128 是保守且足够的约束
5. 多 tensor 共享一块 buffer 时，128 B 对齐保证每个 tensor 的 data 都满足上述要求
```

对比 CPU buft 的 32 B：CPU 侧 32 B 足以覆盖 AVX2 的向量宽度；AVX-512 的更宽向量由 kernel 内部按需处理。不同 buft 不同 alignment，是 gallocr 按 buft 建 tallocr 的直接原因。

---

## 7 CPU backend：为什么必须存在

```cpp
model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
```

断点：

```text
=== [BP3] ggml_backend_init_by_type type=0 ===
#0  ggml_backend_init_by_type (type=GGML_BACKEND_DEVICE_TYPE_CPU, params=0x0) at ggml-backend-reg.cpp:375
```

CPU backend 的创建会触发 `ggml_cpu_init()`，其中有一个内部的 `ggml_init()`/`ggml_free()` 对：

```text
#0 ggml_free (ctx=0x5579e80ecfe0) at ggml.c:1664
#1 ggml_cpu_init () at ggml-cpu.c:3872
#2 ggml_backend_cpu_init () at ggml-cpu.cpp:219
```

这个 context 用于初始化 CPU 的指令集特性、线程池等，与 `build_graph()` 的 context 无关。

### 7.1 三个作用

```text
1. 断言要求：scheduler 强制最后一个 backend 必须是 CPU
2. fallback：CUDA supports_op 不支持的 op 判给 CPU
3. 跨 backend copy 的 host 落脚点：权重在 CPU、计算在 GPU 时，
   split 之间需要 CPU 侧 buffer 作为中转
```

### 7.2 为什么“最后一个必须是 CPU”

```cpp
GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1]))
            == GGML_BACKEND_DEVICE_TYPE_CPU);
```

scheduler 的分配策略假设：

```text
index 0..n-2：加速器，优先级从高到低
index n-1   ：CPU，最终兜底
```

pass 3 的 `for (int b = 0; b < sched->n_backends; b++)` 从 0 开始找“支持该 op 且输入兼容”的 backend，天然偏好低索引（加速器）。如果 CPU 不在最后，fallback 逻辑会失效。

---

## 8 `ggml_backend_sched_new()`：调度器的静态结构

### 8.1 调用与参数

```cpp
model.sched = ggml_backend_sched_new(backends, nullptr, 2,
                                     GGML_DEFAULT_GRAPH_SIZE, false, true);
```

断点：

```text
=== [BP5] ggml_backend_sched_new n_backends=2 graph_size=2048 parallel=0 op_offload=1 ===
backends[0]=0x5579e80e8030   (CUDA0)
backends[1]=0x5579e80f0730   (CPU)
```

| 参数 | 值 | 设计含义 |
|---|---|---|
| `backends` | `{CUDA0, CPU}` | 优先级由数组顺序决定 |
| `bufts` | `nullptr` | 用各 backend 默认 buft |
| `n_backends` | 2 | 最多 `GGML_SCHED_MAX_BACKENDS` |
| `graph_size` | 2048 | 所有容量数组的上限 |
| `parallel` | false | `n_copies = 1`，无流水线 |
| `op_offload` | true | CPU 权重 op 可按 `offload_op` 提升到 GPU |

### 8.2 `n_copies` 与 event

```cpp
sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;   // 4 : 1

if (sched->n_copies > 1) {
    for (int b = 0; b < n_backends; b++)
        for (int c = 0; c < sched->n_copies; c++)
            sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
}
```

`parallel=true` 的设计意图：为同一份图准备多个输入/输出 copy，让第 N+1 次计算与第 N 次的数据拷贝重叠（流水线）。`simple-backend` 只算一次，`parallel=false` 足够。

`n_copies=1` 的直接后果：

```text
cur_copy = next_copy = 0
tensor_copy(t, backend_id, 0) 是唯一副本
events[b][0] == NULL
```

### 8.3 哈希表：2053 从哪来

```cpp
sched->hash_set = ggml_hash_set_new(graph_size);
```

```cpp
size_t ggml_hash_size(size_t min_sz) {
    static const size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, ...
    };
    // 二分找第一个 >= min_sz 的质数
}
```

`min_sz = 2048` → 第一个 ≥ 2048 的质数是 **2053**。

为什么用质数：scheduler 用开放寻址哈希表（`keys[]` + `used` bitset），质数表长可以降低聚集（clustering）。

为什么容量只需要 2048 而不是 2×：

```cpp
// FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
```

源码自己承认这是一个待修的点：理论上节点 + 叶子可能达到 4096，但 `split_graph` 会重新分配更大的 hash（如果需要）。本例 1+2=3，远小于 2053。

### 8.4 backend id 数组

```cpp
const size_t ggml_sched_max_splits = graph_size;                 // 2048
const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
                         // 2048 + 2048*30*2 = 124,928

sched->node_backend_ids      = calloc(nodes_size, sizeof(int));
sched->leaf_backend_ids      = calloc(nodes_size, sizeof(int));
sched->prev_node_backend_ids = calloc(nodes_size, sizeof(int));
sched->prev_leaf_backend_ids = calloc(nodes_size, sizeof(int));
```

为什么 `nodes_size` 远大于 `graph_size`：

```text
最坏情况：每个 node 自成一个 split（2048 个 split）
每个 split 最多 30 个跨 backend 输入
每个输入在 graph_copy 中产生 2 个条目（原 tensor + copy tensor）
→ 2048 * 30 * 2 = 122,880 个额外条目
```

`prev_*` 数组用于比较“这一轮和上一轮的 backend 归属是否变化”，决定是否需要重新 reserve。

### 8.5 scheduler 的元数据 context

```cpp
sched->context_buffer_size =
    ggml_sched_max_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(struct ggml_tensor)
    + ggml_graph_overhead_custom(graph_size, false);
```

代入：

```text
2048 * 30 * 2 * 336 = 41,287,680
ggml_graph_overhead_custom(2048, false) = 82,608
合计 = 41,370,288
```

断点验证：

```text
context_buffer_size = 41370288
```

这个 context 的用途：

```text
split_graph 时复制图（ggml_graph_cpy）
跨 backend copy 时 ggml_dup_tensor_layout 创建 copy tensor
分配依赖节点（GGML_OP_NONE）的创建
```

它**不是** tensor 数据 buffer，也**不是** `model.buf`。

### 8.6 元数据内存账本

| 项 | 计算 | 字节 |
|---|---|---|
| `struct ggml_backend_sched` | gdb `sizeof` | 1,064 |
| `hash_set.keys` | 2053 × 8 | 16,424 |
| `hash_set.used` | ceil(2053/32) × 4 | 260 |
| `hv_tensor_backend_ids` | 2053 × 4 | 8,212 |
| `hv_tensor_copies` | 2053 × 2 × 1 × 8 | 32,848 |
| `node_backend_ids` | 124,928 × 4 | 499,712 |
| `leaf_backend_ids` | 124,928 × 4 | 499,712 |
| `prev_node_backend_ids` | 124,928 × 4 | 499,712 |
| `prev_leaf_backend_ids` | 124,928 × 4 | 499,712 |
| `context_buffer` | 公式 | 41,370,288 |
| `splits` | 16 × 128 | 2,048 |
| `graph_inputs` | 30 × 8 | 240 |
| **合计** | | **≈ 43,430,232（41.42 MiB）** |

**为什么预留这么多**：scheduler 的目标是“图一旦开始 split/alloc，就尽量不再 malloc”。主要容量（hash、ids、context buffer）在 `sched_new` 一次分配完，split 期间只做指针操作；`splits[]` 和 `split->inputs[]` 仍保留倍增扩容路径，用于极端情况（split 数超过 16、单 split 输入超过 30）。代价是初始化 41 MiB 的固定开销——对 llama.cpp 这种长期运行的进程可以忽略，对一次性小工具则显得重。这是“延迟可预测性 vs 启动内存”的典型权衡。

### 8.7 gallocr 的创建

```cpp
sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
```

```cpp
for (int i = 0; i < n_bufs; i++) {
    galloc->bufts[i] = bufts[i];
    galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
}
galloc->n_buffers = n_bufs;
```

断点：

```text
galloc->n_buffers = 2
talloc[0].alignment = 128   (CUDA0)
talloc[1].alignment = 32    (CPU)
```

每个 buft 一个 tallocr，是“按 buffer type 分配”的结构基础。CPU tallocr 在本例中 `n_chunks = 0`，因为没有任何 tensor 判给 CPU。

### 8.8 `ggml_backend_sched_reset()`

```cpp
void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, ...);
        memset(sched->hv_tensor_copies, 0, ...);
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}
```

`calloc` 后 `is_reset == 0`，所以第一次 reset 会执行清零；之后 `is_reset == 1`，只重置 `is_alloc`。

### 8.9 结束状态

```text
sched=0x55b542e86330 n_backends=2 n_copies=1 op_offload=1 is_reset=1 is_alloc=0
backends[0]=0x55b542e52030   (CUDA0)
backends[1]=0x55b542e5a730   (CPU)
bufts[0]=0x7f67fb8980c0      name=CUDA0
bufts[1]=0x7f67fc2001a0      name=CPU
galloc=0x55b542e95120        n_buffers=2  alignment={128, 32}
hash_set.size=2053
context_buffer_size=41370288
```

---

## 9 断点数据逐条解读

| 断点 | 数值 | 含义 | 为什么是这个值 |
|---|---|---|---|
| BP1 | 无日志 | `load_all` 空转 | 文件名/目录不匹配 |
| BP2 | 出现 CUDA/CPU 注册 | registry 首次构造 | `dev_by_type` → `get_reg()` |
| BP4 | `device=0` | 第一个 GPU | `dev_by_type(GPU)` 返回 CUDA0 |
| BP3 | `type=0` | CPU 枚举值 | `GGML_BACKEND_DEVICE_TYPE_CPU = 0` |
| BP5 | `n_backends=2` | CUDA + CPU | 用户显式数组 |
| BP5 | `graph_size=2048` | 容量基准 | `GGML_DEFAULT_GRAPH_SIZE` |
| BP5 | `parallel=0` | 单 copy | 只算一次 |
| BP5 | `op_offload=1` | 允许 CPU→GPU 提升 | 为权重 offload 预留 |
| I1 | `hash_set.size=2053` | 质数表长 | 下一个 ≥2048 的质数 |
| I1 | `context_buffer_size=41370288` | 元数据 arena | 2048×30×2×336 + 82608 |
| I1 | `alignment={128,32}` | 两种内存策略 | CUDA float4/MMA vs CPU SIMD |

---

## 10 设计思想与扩展点

### 10.1 本阶段体现的四个原则

1. **注册与执行分离**：registry/device 是静态能力，backend 是动态实例。
2. **惰性初始化**：registry、设备枚举、stream、cuBLAS、pool、event 全部按需。
3. **按 buffer type 组织内存**：buft 是 alignment/alloc_size/alloc 的载体，gallocr 按 buft 建分配器。
4. **预留最坏情况**：scheduler 元数据一次分配，运行期零 malloc。

### 10.2 可以在这里扩展什么

| 扩展 | 入口 | 影响 |
|---|---|---|
| 新增 backend | 实现 `ggml_backend_reg_i / device_i / backend_i`，在 registry 构造中注册 | 新硬件 |
| 新增 buffer type | 实现 `ggml_backend_buffer_type_i`，在 device 的 `get_buffer_type` 暴露 | 页锁定内存、host-mapped、压缩内存 |
| 多 GPU | `GGML_CUDA_DEVICES` 或多次 `ggml_backend_cuda_init(i)` | tensor split、NCCL |
| 流水线 | `parallel=true` | `n_copies=4`，event 同步 |
| 调整调度优先级 | `backends[]` 顺序 | 谁先被 pass 3 选中 |
| 自定义日志 | `ggml_log_set` | 接入宿主日志系统 |

### 10.3 业务映射

| 业务场景 | 对应本阶段的机制 |
|---|---|
| 推理服务启动 | `init_model`：一次初始化，长期复用 |
| 多卡部署 | `backends[]` 放多个 CUDA backend，CPU 最后 |
| 显存预算评估 | gallocr 按 buft 建分配器；alignment 决定填充 |
| 首 token 延迟 | stream/cuBLAS/pool 惰性创建，小图不付大初始化成本 |
| 模型热切换 | registry/device 是进程级，backend/scheduler 可按模型重建 |
| 插件式后端 | `GGML_BACKEND_DL=ON` + score 选择 |

---

## 11 源码逐行：init_model 链路

本节把 `init_model()` 触发的四个关键函数逐行拆开，行号对应云端 `/home/ggml`（commit `353b63b`）。阅读方式：先看代码，再看「为什么这一行必须存在」，最后看断点验证。

### 11.1 `ggml_backend_registry::register_backend()`

`src/ggml-backend-reg.cpp:186`

```cpp
void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
    if (!reg) {
        return;
    }

    for (auto & entry : backends) {
        if (entry.reg == reg) {
            return;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
        __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
    backends.push_back({ reg, std::move(handle) });
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
        register_device(ggml_backend_reg_dev_get(reg, i));
    }
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 187 | `if (!reg) return;` | 动态加载失败时 `ggml_backend_*_reg()` 可能返回空，静默跳过而不是崩溃 |
| 190-194 | 遍历 `backends` 去重 | 静态链接 + 动态加载同一个库时，避免注册两次导致设备重复枚举 |
| 197-200 | `GGML_LOG_DEBUG(...)` | **这一行就是日志 `register_backend: registered backend CUDA (1 devices)` 的来源** |
| 201 | `backends.push_back({reg, handle})` | `handle` 是 `dlopen` 句柄；静态注册时为 `nullptr`，析构时不会被 `dlclose` |
| 202-204 | 循环注册设备 | 注意：`ggml_backend_reg_dev_count(reg)` 被调用了两次（日志一次、循环一次），CUDA 的 `get_device_count` 内部走 `ggml_cuda_info()`，第二次是缓存命中 |

**关键点：日志中的设备数来自 `ggml_backend_reg_dev_count(reg)`，它触发了 CUDA 的惰性初始化**。这就是为什么 `load_all()` 没有任何日志，而 `init_best()` 之后突然出现两行注册日志。

### 11.2 `ggml_backend_registry::register_device()`

`src/ggml-backend-reg.cpp:207`

```cpp
void register_device(ggml_backend_dev_t device) {
    for (auto & dev : devices) {
        if (dev == device) {
            return;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: registered device %s (%s)\n",
        __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
    devices.push_back(device);
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 208-212 | 指针去重 | device 是进程级静态对象，指针唯一 |
| 215-216 | 打印 name + description | `name = "CUDA0"`，`description = "NVIDIA GeForce RTX 4090"`，正好对应日志 |
| 217 | `devices.push_back(device)` | **只存指针，不拥有对象**；device 由 backend reg 自己持有 |

**设计要点**：registry 是「索引」而不是「所有者」。`devices` 里的指针生命周期由各 backend 的静态对象保证。

### 11.3 `ggml_backend_cuda_init()`

`src/ggml-cuda/ggml-cuda.cu:5763`

```cpp
ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 5764 | 设备号范围检查 | 防止越界访问 `info.devices[]` |
| 5769 | `new ggml_backend_cuda_context(device)` | **只分配结构体**（4336 B），构造函数不创建 stream/cuBLAS/pool |
| 5775-5780 | 聚合初始化 `ggml_backend` | 四个字段：`guid` 用于类型判断，`iface` 是函数表，`device` 指回 registry，`context` 是执行状态 |
| 5778 | `ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device)` | 再次调用 `ggml_backend_cuda_reg()`；由于是函数内 static，返回同一实例 |

**断点验证**：

```text
[BP4] ggml_backend_cuda_init device=0
#0 ggml_backend_cuda_init            ggml-cuda.cu:5843
#1 ggml_backend_cuda_device_init_backend  ggml-cuda.cu:5119
#2 ggml_backend_dev_init             ggml-backend.cpp:610
#3 ggml_backend_init_best            ggml-backend-reg.cpp:389
```

注意 gdb 显示的行号是 `5843`，而源码定义在 `5763`——这是因为 `ggml-cuda.cu` 里有 `#include` 展开或宏，gdb 的 `at file:line` 指向的是**实际指令行**而非函数首行。这是读 CUDA 源码时常见的小陷阱。

### 11.4 `ggml_backend_dev_init()`

`src/ggml-backend.cpp:610`

```cpp
ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}
```

这一行是整个后端体系的**分发点**：

```text
device->iface.init_backend
  ├── CUDA0 → ggml_backend_cuda_device_init_backend
  │             → ggml_backend_cuda_init(ctx->device)
  ├── CPU   → ggml_backend_cpu_device_init_backend
  │             → ggml_backend_cpu_init()
  └── 其他   → 各自实现
```

**设计价值**：`ggml_backend_init_best()` 完全不知道 CUDA 的存在，它只调用 `dev_init`；具体创建逻辑由 device 的 iface 决定。新增后端不需要改这一层。

### 11.5 `ggml_backend_sched_new()`

`src/ggml-backend.cpp:1848`

```cpp
ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (struct ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));
```

逐段：

**① 三个断言（1854-1856）**

```cpp
GGML_ASSERT(n_backends > 0);
GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);          // 16
GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1]))
            == GGML_BACKEND_DEVICE_TYPE_CPU);
```

第三个断言是**调度算法的契约**：pass 3 从 index 0 开始找 backend，index 越大优先级越低，最后一个必须是兜底的 CPU。没有它，`n_supported` 全为 0 时可能选中一个不能执行任意 op 的加速器。

**② `calloc` 而不是 `malloc`（1858）**

```cpp
struct ggml_backend_sched * sched = calloc(1, sizeof(struct ggml_backend_sched));
```

`calloc` 保证所有指针为 NULL、所有 int 为 0。后面 `ggml_backend_sched_reset()` 依赖 `is_reset == 0` 来判断“是否第一次 reset”。

**③ 调试开关（1860-1869）**

```cpp
const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;
```

`GGML_SCHED_DEBUG=1` 打印 split 概要，`=2` 额外打印每个 node 的 backend 归属和 `use_counts`。这就是 §7 中 `## SPLIT #0: CUDA0` 的来源。

**④ 哈希表容量（1875）**

```cpp
sched->hash_set = ggml_hash_set_new(graph_size);
```

注意传入的是 `graph_size = 2048`，不是 `2 * graph_size`。源码注释自己承认：

```cpp
// FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
```

理论上一张图最多有 2048 nodes + 2048 leafs = 4096 个 tensor，2053 个槽位可能不够。这是一个已知的待修点；`split_graph` 里通过 `ggml_hash_find_or_insert` 的扩容或断言兜底。

**⑤ `nodes_size` 的推导（1880-1884）**

```cpp
const size_t ggml_sched_max_splits = graph_size;                                  // 2048
const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
```

`GGML_SCHED_MAX_SPLIT_INPUTS = 30`，`*2` 是因为每个跨 backend input 会在图副本里产生**两个**节点（依赖 view + copy tensor）。所以：

```text
2048 + 2048 * 30 * 2 = 124,928
```

这是 `node_backend_ids[]` 等四个数组的长度。**为什么按最坏情况**：如果每个 node 自成一个 split，每个 split 最多 30 个跨 backend 输入，就要这么多槽位。

**⑥ `context_buffer_size`（1889-1890）**

```cpp
sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor)
                           + ggml_graph_overhead_custom(graph_size, false);
```

```text
2048 * 30 * 2 * 336 = 41,287,680
+ 82,608
= 41,370,288
```

与断点实测的 `41370288` 完全一致。这笔内存用于：

```text
1. split 时 ggml_graph_cpy 复制节点/叶子指针
2. 跨 backend copy 时 ggml_dup_tensor_layout 创建 copy tensor
3. 分配依赖节点（GGML_OP_NONE）的创建
```

**⑦ `splits` 与 `graph_inputs` 用倍增扩容（1892-1898）**

```cpp
const int initial_splits_capacity = 16;
sched->splits = calloc(initial_splits_capacity, ...);
sched->splits_capacity = initial_splits_capacity;
```

与前面几个数组不同，`splits` 没有按最坏情况预留，而是初始 16、按需倍增。原因：`ggml_backend_sched_split` 有 128 B，2048 个就是 256 KB，预留意义不大；而实际 split 数通常远小于节点数。

### 11.6 `ggml_gallocr_new_n()`

`src/ggml-alloc.c:498`

```cpp
ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));

    galloc->bufts       = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    galloc->buffers     = calloc(n_bufs, sizeof(struct vbuffer *));
    galloc->buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size  = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 505 | `buffers = calloc(...)` | **初始全 NULL**，表示还没有申请任何物理 buffer |
| 512 | `for j < i` 复用 tallocr | 如果两个 backend 用同一个 buft（例如 BLAS + CPU 都用 host 内存），共享一个 tallocr，避免同一块内存被分配两次 |
| 521-523 | `get_alignment` / `get_max_size` | **alignment 来自 buft**，这是 CUDA 128 / CPU 32 差异的源头 |
| 524 | `ggml_dyn_tallocr_new(alignment, max_size)` | 创建一个“只知道对齐和上限”的分配器，此时 `n_chunks = 0` |

断点验证：

```text
galloc->n_buffers = 2
talloc[0].alignment = 128   (CUDA0)
talloc[1].alignment = 32    (CPU)
talloc[1].n_chunks = 0      (没有 tensor 判给 CPU)
```

### 11.7 `ggml_dyn_tallocr_new()`

`src/ggml-alloc.c:366`

```cpp
static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {NULL},
        /*.n_chunks       = */ 0,
    };

    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}
```

逐行：

| 行 | 代码 | 为什么 |
|---|---|---|
| 371 | `MIN(max_buffer_size, SIZE_MAX/2)` | 防止 `offset + size` 溢出；`SIZE_MAX/2 = 2^63-1`，与断点实测的 `9223372036854775807` 一致 |
| 379 | `ggml_dyn_tallocr_reset(alloc)` | 清空 chunk 数组，`n_chunks = 0` |

注意：`reset` 之后 `n_chunks = 0`，**没有创建任何 chunk**。第一个 chunk 是在第一次 `ggml_dyn_tallocr_alloc` 时惰性创建的（`ggml_dyn_tallocr_new_chunk`）。这就是为什么 `talloc[1]`（CPU）从未使用就一直是 `n_chunks = 0`。

### 11.8 完整调用链（带行号）

```text
init_model()                                  simple-backend.cpp:57
 ├── ggml_backend_load_all()                  ggml-backend-reg.cpp:574
 │    └── ggml_backend_load_best("cuda", ...) ggml-backend-reg.cpp:481
 │         └── (无匹配文件，返回 nullptr)
 ├── ggml_backend_init_best()                 ggml-backend-reg.cpp:382
 │    ├── ggml_backend_dev_by_type(GPU)       ggml-backend-reg.cpp:355
 │    │    └── ggml_backend_dev_count()
 │    │         └── get_reg()                 ggml-backend-reg.cpp:292
 │    │              └── ggml_backend_registry::ggml_backend_registry()  :119
 │    │                   ├── register_backend(cuda_reg)  :186
 │    │                   │    ├── register_device(CUDA0) :207
 │    │                   │    └── ggml_cuda_info() → ggml_cuda_init()
 │    │                   └── register_backend(cpu_reg)
 │    └── ggml_backend_dev_init(dev)          ggml-backend.cpp:610
 │         └── ggml_backend_cuda_device_init_backend  ggml-cuda.cu:5119
 │              └── ggml_backend_cuda_init(0)  ggml-cuda.cu:5763
 ├── ggml_backend_init_by_type(CPU)           ggml-backend-reg.cpp:374
 └── ggml_backend_sched_new()                 ggml-backend.cpp:1848
      ├── ggml_hash_set_new(2048)             ggml.c:6613
      ├── ggml_graph_overhead_custom()        ggml.c:7469
      └── ggml_gallocr_new_n()                ggml-alloc.c:498
           └── ggml_dyn_tallocr_new() ×2      ggml-alloc.c:366
```

**这张调用链的价值**：从 `init_model()` 到 `ggml_dyn_tallocr_new()`，中间隔了 7 层。任何一层都可以被替换或扩展，而调用者不需要知道。这就是 §1 五层对象模型的代码依据。

---

## 12 断点与变量清单

```gdb
break ggml_backend_load_all
break ggml_backend_init_best
break ggml_backend_init_by_type
break ggml_backend_dev_init
break ggml_backend_cuda_device_init_backend
break ggml_backend_cuda_init
break ggml_backend_sched_new
break ggml_backend_sched_reset
break cublasCreate
break cudaStreamCreateWithFlags
```

重点变量：

```gdb
p device
p backends[0]
p backends[1]
p n_backends
p graph_size
p parallel
p op_offload
p sched->n_copies
p sched->bufts[0]
p sched->bufts[1]
p ggml_backend_buft_name(sched->bufts[0])
p sched->galloc->n_buffers
p sched->galloc->buf_tallocs[0]->alignment
p sched->galloc->buf_tallocs[1]->alignment
p sched->context_buffer_size
p sched->hash_set.size
p ((ggml_backend_cuda_context*)backend->context)->streams[0][0]
p ((ggml_backend_cuda_context*)backend->context)->cublas_handles[0][0]
```

---

## 13 待验证点

1. **动态加载**：把 `libggml-cuda.so` 按 `libggml-cuda-89.so` 命名放到 `build-debug/bin/`，或 `GGML_BACKEND_PATH=/path/to/libggml-cuda.so`，观察 `loaded CUDA backend from ...` 与去重行为。
2. **VMM**：在 `ggml_cuda_pool_vmm` 构造函数下断点，确认大模型下 pool 走 VMM；本例 384 B tensor buffer 走 `cudaMalloc`。
3. **stream 创建时机**：在 `cudaStreamCreateWithFlags` 下断点，确认首次创建发生在 `alloc_splits` 的 `ggml_backend_synchronize()`。
4. **cuBLAS 惰性**：在 `cublasCreate` 下断点，确认本例不命中；把 `rows_B` 改成 4 或设 `GGML_CUDA_FORCE_CUBLAS=1`，确认 handle + workspace 创建。
5. **多 GPU**：`GGML_CUDA_DEVICES=2`，观察虚拟设备、`physical_device`、`default_tensor_split`。
6. **parallel=true**：改 `ggml_backend_sched_new` 的 `parallel` 参数，观察 `n_copies=4`、`events[2][4]` 的创建。
7. **内存账本**：用 `p sizeof(struct ggml_backend_sched)`、`p sched->hash_set.size` 验证 §8.6 的表格。
