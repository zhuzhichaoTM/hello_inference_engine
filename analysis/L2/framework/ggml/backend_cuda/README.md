# GGML-CUDA 源码深度学习笔记 · 总纲与导航

> 对标版本：**ggml v0.25.3**，commit `353b63b4`
> 源码根：`/Users/zhuzhichao/Documents/ai_interfence/clion/ggml/`
> 核心模块：`src/ggml-cuda/`（162 文件 / 45293 行）
> 终极目标：读懂全量核心源码 → 掌握 CUDA 推理运行机制 → 独立扩展/改写/优化 CUDA 算子

---

## 一、章节索引

| 模块 | 文件 | 一句话内容 | 篇幅 |
|---|---|---|---|
| 一 | [01-目录结构与核心数据结构.md](01-目录结构与核心数据结构.md) | 五层架构坐标、162 文件分工、核心结构体、backend 注册与 buffer 生命周期 | 480 行 |
| 二 | [02-运行全流程与显存流机制.md](02-运行全流程与显存流机制.md) | 初始化时序、前向全链路、显存三通道、多流并发、同步点、CUDA Graph、多卡通信 | 484 行 |
| 三 | [03-算子开发模板与实战.md](03-算子开发模板与实战.md) | 三段式算子模板、unary/softmax 逐行精读、自定义算子全流程、全算子族地图、优化与排错 | 798 行 |
| 四 | [04-量化内核深挖-MMVQ与MMQ.md](04-量化内核深挖-MMVQ与MMQ.md) | 量化块布局、K-quant/IQ/FP4、vecdotq、MMVQ、MMQ、架构编译边界 | 708 行 |
| 五 | [05-MUL_MAT全路径分发链.md](05-MUL_MAT全路径分发链.md) | `MUL_MAT` 决策树、MMVF/MMF/MMVQ/MMQ/cuBLAS、`MUL_MAT_ID`/MoE | 616 行 |
| 六 | [06-FlashAttention三路实现.md](06-FlashAttention三路实现.md) | online softmax、Vec/Tile/MMA-F16、GQA、稀疏、workspace 与 fallback | 1005 行 |

### 配套题库（以考促学）✅ 已完成 1000 题

| 卷 | 文件 | 题量 | 配比 |
|---|---|---|---|
| 总纲 | [题库/README.md](题库/README.md) | — | 分卷配比、进度、使用方式 |
| 卷一 | [题库/01-架构-300题.md](题库/01-架构-300题.md) | 300 | 单选 180 / 多选 120 |
| 卷二 | [题库/02-量化-350题.md](题库/02-量化-350题.md) | 350 | 单选 210 / 多选 140 |
| 卷三 | [题库/03-算子-200题.md](题库/03-算子-200题.md) | 200 | 单选 120 / 多选 80 |
| 卷四 | [题库/04-调优-150题.md](题库/04-调优-150题.md) | 150 | 单选 90 / 多选 60 |

> 共 **1000 题**（单选 600 / 多选 400；基础 300 / 进阶 500 / 高阶 200），每题五段式，解析均锚定源码行号。
> 建议用法：读完一个模块 → 做对应卷 → 错题回到笔记对应 §。

### 配套题库（原理业务版）—— 以考促学 · 工程决策向 ✅ 已完成 1000 题

| 卷 | 文件 | 题量 | 侧重 |
|---|---|---|---|
| 总纲 | [题库-原理业务版/README.md](题库-原理业务版/README.md) | — | 与源码版区别、配比、使用方式 |
| 卷一 | [题库-原理业务版/01-架构原理-300题.md](题库-原理业务版/01-架构原理-300题.md) | 300 | 张量语义、图执行、异构调度、显存业务行为 |
| 卷二 | [题库-原理业务版/02-量化原理-350题.md](题库-原理业务版/02-量化原理-350题.md) | 350 | 量化数学、误差传播、选型决策 |
| 卷三 | [题库-原理业务版/03-算子原理-200题.md](题库-原理业务版/03-算子原理-200题.md) | 200 | 算子数学、CUDA 执行模型、Tensor Core |
| 卷四 | [题库-原理业务版/04-性能调优-150题.md](题库-原理业务版/04-性能调优-150题.md) | 150 | 瓶颈识别、显存/KV 优化、多卡、调优方法论 |

> 两版题库互补：**源码版**练「读得懂代码」（含文件/行号锚点），**原理业务版**练「用得上」（严禁源码定位题，全部围绕原理、业务现象、工程取舍）。

---

## 二、核心知识点总表（按重要度分级）

**分级标准**：

| 级别 | 含义 | 不掌握会怎样 |
|---|---|---|
| **P0 骨架** | 决定“代码怎么跑起来”，所有其他知识都挂在它上面 | 完全读不懂调用链，无法定位任何问题 |
| **P1 主力** | 决定“性能好不好”，占实际推理耗时 90%+ | 能跑但调不动、优化无从下手 |
| **P2 机制** | 决定“边界在哪”，影响正确性、并发、多卡 | 遇到并发/多卡/长上下文问题无法排查 |
| **P3 专项** | 特定模型/算子族，用到再深入 | 不影响主干能力，按需查阅 |

### P0 · 骨架级（必须最先吃透）

| # | 知识点 | 为什么重要 | 导航 | 源码锚点 |
|---|---|---|---|---|
| 1 | **五层架构与 backend 抽象** | ggml-cuda 只是 `ggml-backend` 的一个实现；不懂这层就找不到入口 | 模块一 §0 | `src/ggml-backend-impl.h:17/46/121/176/230` |
| 2 | **算子总分发 `compute_forward`** | 所有 CUDA 算子的唯一入口 switch，~150 个 case | 模块一 §3、模块三 §3.10 | `ggml-cuda.cu:2068` |
| 3 | **`supports_op` 能力声明** | 决定 op 是否被调度到 CUDA；漏改则算子静默失效 | 模块一 §3、模块三 §3.4 | `ggml-cuda.cu:5136` |
| 4 | **`ggml_backend_cuda_context`** | 所有算子入口的第一个参数，持有流/句柄/显存池 | 模块一 §2.2 | `common.cuh:1455` |
| 5 | **三段式算子结构** | `.cuh` 声明 / `.cu` 实例化 / `ggml_cuda_op_xxx` 入口 | 模块三 §3.1 | `unary.cu:117/129/138` |
| 6 | **算子入口签名铁律** | `void ggml_cuda_op_xxx(ggml_backend_cuda_context &, ggml_tensor *)` | 模块三 §3.1 | `unary.cu:138` |
| 7 | **前向执行全链路** | 图 → sched → graph_compute → compute_forward → kernel | 模块二 §2.2 | `ggml-cuda.cu:4425/4190/2068` |
| 8 | **异步语义** | `graph_compute` 返回 ≠ 算完，同步在别处 | 模块二 §2.2/§2.4 | `ggml-cuda.cu:2548` |

### P1 · 主力级（决定推理性能）

| # | 知识点 | 为什么重要 | 导航 | 源码锚点 |
|---|---|---|---|---|
| 9 | **`MUL_MAT` 决策树** | 推理耗时最大头；6 条路径的运行时选择 | 模块五 §5.0/§5.9 | `ggml-cuda.cu:1824` |
| 10 | **量化点积 `vec_dot_*_q8_1`** | 量化推理的算力核心：整数点积 + scale | 模块四 §4.3 | `vecdotq.cuh:118` |
| 11 | **`dp4a` 4 路整数乘加** | 一条指令算 4 个乘积，量化加速的物理来源 | 模块四 §4.3.3 | `common.cuh:720` |
| 12 | **量化块布局 Q4_0 / Q8_1** | `-8` 偏移与预计算块和 `s` 是全部量化的基础 | 模块四 §4.1 | `src/ggml-common.h:195/259` |
| 13 | **MMVQ（decode 主力）** | 单 token 推理的主要路径 | 模块四 §4.4 | `mmvq.cu:318/599/1421` |
| 14 | **MMQ（量化 GEMM）** | 中等 batch 的量化矩阵乘，tile + stream-k | 模块四 §4.5 | `mmq.cu:266` / `mmq.cuh:31/165/955` |
| 15 | **激活量化 `quantize_q8_1`** | MMVQ/FlashAttention 的第一步；warp 归约 + `s` 预计算 | 模块四 §4.2 | `quantize.cu:54` |
| 16 | **cuBLAS fallback 与转换成本** | 大 batch 走这里；量化→F16 转换有隐藏开销 | 模块五 §5.7 | `ggml-cuda.cu:1623/1410` |
| 17 | **FlashAttention 三路（Vec/Tile/MMA）** | 长上下文的主要开销 | 模块六 §6.2/§6.8 | `fattn.cu:559/755` |
| 18 | **online softmax** | FlashAttention 的数学核心，不落完整 score 矩阵 | 模块六 §6.3 | `fattn-common.cuh:9-19` |
| 19 | **算子融合（图级）** | `MUL_MAT+ADD+GLU` 等模式的旁路优化 | 模块三 §3.5、模块二 §2.5 | `ggml-cuda.cu:4509` |
| 20 | **kernel 启动与 PDL** | 统一走 `ggml_cuda_kernel_launch`，含 Hopper 重叠启动 | 模块三 §3.1/§3.2 | `common.cuh:1699` |

### P2 · 机制级（正确性 / 并发 / 多卡）

| # | 知识点 | 为什么重要 | 导航 | 源码锚点 |
|---|---|---|---|---|
| 21 | **显存池（VMM / LEG）与 RAII** | 临时显存必须走池；`actual_size` 易错 | 模块一 §2.3 | `ggml-cuda.cu:419/536`、`common.cuh:1215` |
| 22 | **多流并发 concurrent_events** | Attention 里 Q/K/V 三路并行的唯一机制 | 模块二 §2.4 | `ggml-cuda.cu:4770/4211` |
| 23 | **同步点地图** | 分清热/冷路径同步，是排查卡顿的基础 | 模块二 §2.4 | `ggml-cuda.cu:2548/790/2537` |
| 24 | **CUDA Graph 录制/重放/热身** | 小 batch 提速 10~30%，但有严格兼容边界 | 模块二 §2.2/§2.8 | `ggml-cuda.cu:4425-4480` |
| 25 | **多卡通信（NCCL/internal/butterfly）** | 多卡推理与 tensor split 的基础 | 模块二 §2.8 | `ggml-cuda.cu:969-1273`、`allreduce.cu` |
| 26 | **虚拟设备 → 物理设备映射** | 多卡代码的第一道坑 | 模块二 §2.1 | `ggml-cuda.cu:113/250` |
| 27 | **`MUL_MAT_ID` / MoE 路径** | MoE 模型的专用分发 + 排序 fallback + graph 约束 | 模块五 §5.8 | `ggml-cuda.cu:1882/1903` |
| 28 | **`ggml_cuda_pool_alloc` 参数单位** | 参数是**元素个数**，误用导致错位 | 模块一 §2.3、模块三 §3.3 | `common.cuh:1215` |
| 29 | **backend 注册与 feature 暴露** | 排查“同二进制不同机器表现不同” | 模块一 §6 | `ggml-cuda.cu:5787/5756/5701` |
| 30 | **buffer / host buffer 生命周期** | device / pinned host / pageable host 三种内存 | 模块一 §6.3 | `ggml-cuda.cu:869/1273/4910` |
### P3 · 专项级（按需查阅）

| # | 知识点 | 导航 | 源码锚点 |
|---|---|---|---|
| 31 | K-quant（Q2_K~Q6_K，256 元素分层 scale） | 模块四 §4.9.1 | `vecdotq.cuh` / `mmq-config-*.cuh` |
| 32 | IQ 码本量化（`__byte_perm` 嵌套查表） | 模块四 §4.3.2/§4.9.2 | `vecdotq.cuh:34` |
| 33 | MXFP4 / NVFP4（Blackwell FP4 路径） | 模块四 §4.9.3 | `mmq.cuh:31` / `mmq-config-blackwell.cuh` |
| 34 | 编译架构 vs 运行架构 | 模块四 §4.9.4 | `ggml_cuda_highest_compiled_arch()` |
| 35 | FlashAttention 稀疏 gather | 模块六 §6.7.4 | `fattn.cu:12/128` |
| 36 | FlashAttention workspace / `dst_meta` | 模块六 §6.4/§6.15 | `fattn.cu:719` / `fattn-common.cuh:46` |
| 37 | RoPE / 位置编码 | 模块三 §3.10 | `rope.cu:696` |
| 38 | Norm 归约族 | 模块三 §3.10 | `norm.cu:435/478` |
| 39 | 卷积 / im2col | 模块三 §3.10 | `conv2d.cu:337` / `im2col.cu:81` |
| 40 | top-k / argsort | 模块三 §3.10 | `top-k.cu:213` / `argsort.cu:254` |
| 41 | SSM / 线性注意力族 | 模块三 §3.10 | `ssm-scan.cu:767` / `ssm-conv.cu:160` |
| 42 | 训练算子（AdamW / SGD / CE loss） | 模块三 §3.10 | `opt-step-adamw.cu` / `cross-entropy-loss.cu` |
| 43 | copy / get_rows / set_rows | 模块三 §3.10 | `cpy.cu:429` / `getrows.cu:442` / `set-rows.cu:376` |

---

## 三、重要度排序说明（为什么是这个顺序）

### 排序依据：**依赖关系 > 耗时占比 > 出错概率**

```
┌────────────────────────────────────────────────────────────┐
│ P0 骨架：不掌握 = 读不懂代码                              │
│   架构分层 → 分发 switch → context → 三段式 → 执行链路      │
│   （缺任一环，后面所有内容都无处挂靠）                      │
└────────────────────────────────────────────────────────────┘
                          ↓ 依赖
┌────────────────────────────────────────────────────────────┐
│ P1 主力：不掌握 = 性能调不动                              │
│   MUL_MAT 决策树 → 量化点积 → MMVQ/MMQ → FlashAttention    │
│   （占实际推理耗时 90%+，是性能的全部来源）                 │
└────────────────────────────────────────────────────────────┘
                          ↓ 依赖
┌────────────────────────────────────────────────────────────┐
│ P2 机制：不掌握 = 边界踩坑                                │
│   显存池 → 多流 → 同步点 → CUDA Graph → 多卡通信           │
│   （平时不出问题，一出就是正确性/并发/多卡难题）            │
└────────────────────────────────────────────────────────────┘
                          ↓ 依赖
┌────────────────────────────────────────────────────────────┐
│ P3 专项：不掌握 = 特定模型受限                            │
│   K-quant / IQ / FP4 / 稀疏 / SSM / 卷积 / 训练算子         │
│   （用到再查，不影响主干工程能力）                          │
└────────────────────────────────────────────────────────────┘
```

### 三条关键判断

**1. 为什么 `supports_op` 排在 P0 第 3 位（而不是埋在细节里）？**
因为它是**最隐蔽的失效点**：只改 `compute_forward` 不改 `supports_op`，算子会**静默不被调度**，没有报错、没有日志，只是走了 CPU 或压根不执行。这是新手第一大坑。

**2. 为什么量化点积排在 P1 而不是 P0？**
因为它是**性能问题而非功能问题**：不理解 `dp4a` 和 `s` 预计算，代码依然能跑对，只是慢 4~8 倍。但它决定了“能不能做量化推理优化”，所以是 P1 首位。

**3. 为什么 FlashAttention 排在 P1 靠后？**
因为**短上下文下它不是瓶颈**。只有在长上下文（KV > 8K）或高并发推理时，它才成为主要开销。但一旦涉及长上下文，它的三路选择和 workspace 管理就无法回避。

### 学习投入建议

| 级别 | 建议投入 | 目标 |
|---|---|---|
| P0 | 总时间的 **30%** | 能画出完整调用链，能定位任意 op 的入口 |
| P1 | 总时间的 **40%** | 能解释任意一次推理走哪条路径、为什么 |
| P2 | 总时间的 **20%** | 能排查并发/多卡/显存/图捕获问题 |
| P3 | 总时间的 **10%** | 用到时能快速定位对应文件 |

---

## 四、导航指引

### 4.1 按学习阶段导航

```
S1 基础认知 ──→ S2 机制吃透 ──→ S3 源码精读 ──→ S4 算子改写 ──→ S5 自定义开发
   模块一          模块二          模块三/四         模块三实战       模块三实战
   P0 骨架         P2 机制        P1 主力           改参数/向量化    新算子全流程
```

| 阶段 | 读什么 | 完成后应具备 |
|---|---|---|
| **S1 基础认知** | 模块一全篇 | 能说清 ggml-cuda 在架构里的位置、核心结构体、文件分工 |
| **S2 机制吃透** | 模块二全篇 | 能画出初始化 → 前向 → 同步的完整时间轴 |
| **S3 源码精读** | 模块三 §3.2-3.3 + 模块四全篇 | 能逐行讲清 unary/softmax/vecdotq |
| **S4 算子改写** | 模块三 §3.5-3.7 + 模块五 | 能改阈值、向量化、测性能 |
| **S5 自定义开发** | 模块三 §3.4 + 模块六 §6.11 | 能从零加一个算子并跑通测试 |

### 4.2 按目标导航

| 我想…… | 直接看 |
|---|---|
| 搞清一次推理的完整调用链 | 模块二 §2.2 时间轴 |
| 知道某个 op 走哪条 kernel | 模块五 §5.9 决策表 |
| 加一个自定义算子 | 模块三 §3.4 检查清单（11 项） |
| 理解量化为什么快 | 模块四 §4.0/§4.3 |
| 调 decode 性能 | 模块四 §4.4（MMVQ）、模块五 §5.11 |
| 调 prefill 性能 | 模块五 §5.7（cuBLAS）、模块四 §4.5（MMQ） |
| 调长上下文性能 | 模块六 §6.8（三路选择）、§6.12（分析） |
| 排查 illegal memory access | 模块三 §3.6 排错手册 |
| 排查多卡问题 | 模块二 §2.8 |
| 排查 CUDA Graph 不生效 | 模块二 §2.8 边界 |
| 知道某算子是否有 CUDA 实现 | 模块三 §3.10 算子族地图 |

### 4.3 按问题类型导航（排错索引）

| 症状 | 首查 | 次查 |
|---|---|---|
| 算子从不被调度 | `supports_op`（模块三 §3.4） | backend feature（模块一 §6.4） |
| 结果错乱无报错 | `actual_size` / stride（模块一 §2.3） | 量化 `-8` 偏移（模块四 §4.1.2） |
| illegal memory access | 异步报错滞后（模块三 §3.6） | `compute-sanitizer` |
| 推理卡顿（首次慢） | CUDA Graph 热身（模块二 §2.2） | 首次 kernel 加载 |
| 多卡结果错 | 虚拟/物理设备（模块二 §2.1） | `ggml_cuda_set_device` |
| 量化模型精度异常 | MMVQ/MMQ 路径（模块五 §5.9） | 试 `-ngl 0` 对照 CPU |
| 显存逐渐增长 | 池 vs `cudaMalloc`（模块一 §2.3） | `compute-sanitizer --tool memcheck` |
| MoE 模型性能差 | `MUL_MAT_ID` 路径（模块五 §5.8） | `n_experts` 阈值 |
| 长上下文慢 | FlashAttention 路径（模块六 §6.8） | 是否走到 Tile/MMA |

---

## 五、开发铁律（背下来）

1. 算子入口签名固定：`void ggml_cuda_op_xxx(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
2. 新算子必须改 **`compute_forward` + `supports_op` 两处**（漏一处静默失效）
3. 一律 `ctx.stream()`，绝不自建流或写死流号
4. 一律 `ggml_cuda_kernel_launch`，不直接 `<<<>>>`（丢 PDL）
5. 临时显存用 `ggml_cuda_pool_alloc<T>`（参数是**元素个数**）
6. 架构判据读 `ggml_cuda_info().devices[i].cc`，不用 `__CUDA_ARCH__`
7. 不 `#include <cuda_runtime.h>`，靠 `vendors/*.h` 保持跨厂商
8. 量化算子必须处理 `ne10_padded` 对齐与 padding 清零
9. 涉及 MoE/排序/通信时，必须检查 CUDA Graph 兼容性
10. 修改 heuristic 后，必须同时测正确性 + 单 kernel 延迟 + 端到端吞吐 + 显存

---

## 六、源码锚点速查表

| 想找什么 | 位置 |
|---|---|
| 算子总分发 | `ggml-cuda.cu:2068` `ggml_cuda_compute_forward` |
| 能力声明 | `ggml-cuda.cu:5136` `supports_op` |
| 设备属性（cc/warp_size/smpbo） | `common.cuh:1177` `ggml_cuda_device_info` |
| backend context | `common.cuh:1455` |
| 临时显存 RAII | `common.cuh:1215` `ggml_cuda_pool_alloc` |
| kernel 启动（含 PDL） | `common.cuh:1699` `ggml_cuda_kernel_launch` |
| 类型元数据（qk/qr/qi/bs） | `common.cuh:982` `ggml_cuda_type_traits` |
| `dp4a` | `common.cuh:720` |
| 显存池 LEG / VMM | `ggml-cuda.cu:419` / `:536` |
| 设备探测 | `ggml-cuda.cu:221` `ggml_cuda_init` |
| 图执行 | `ggml-cuda.cu:4190` / `:4425` |
| 图级融合 | `ggml-cuda.cu:4509` `graph_optimize` |
| 后端注册 | `ggml-cuda.cu:5787` / `:5756` |
| 多卡通信 | `ggml-cuda.cu:969-1273` / `allreduce.cu` |
| **`MUL_MAT` 决策树** | `ggml-cuda.cu:1824` |
| cuBLAS 路径 | `ggml-cuda.cu:1410` / `:1623` |
| MMVQ 决策 / kernel / 入口 | `mmvq.cu:318` / `:599` / `:1421` |
| MMQ 决策 / 入口 / 配置 | `mmq.cu:266` / `:85` / `mmq.cuh:165` |
| MMVF / MMF 决策 | `mmvf.cu:792` / `mmf.cu:133` |
| 量化点积核心 | `vecdotq.cuh:118` `vec_dot_q4_0_q8_1_impl` |
| 4bit 查表 | `vecdotq.cuh:34` `get_int_from_table_16` |
| 激活量化 | `quantize.cu:54` `quantize_q8_1` |
| 量化块定义 | `src/ggml-common.h:195`(q4_0) / `:259`(q8_1) |
| FlashAttention 入口 / 选择 | `fattn.cu:755` / `:559` |
| FlashAttention Vec / Tile / MMA | `fattn-vec.cuh:21` / `fattn-tile.cuh:793` / `fattn-mma-f16.cuh:1807` |
| 三段式算子标本 | `unary.cu` |
| 归约算子标本 | `softmax.cu` / `norm.cu` |
| CPU 参考实现 | `src/ggml-cpu/ops.cpp` |
| 后端一致性测试 | `tests/test-backend-ops.cpp` |

---

## 七、调试速查

```bash
# 1. 只编 CUDA 后端快速迭代
cmake --build build --target ggml-cuda -j

# 2. 运行时详细日志（含 stream/concurrent 信息）
GGML_CUDA_DEBUG=1 ./binary ...

# 3. 内存/竞态检查（定位 illegal memory access 最有效）
compute-sanitizer --tool memcheck ./binary ...
compute-sanitizer --tool racecheck ./binary ...

# 4. 后端一致性 + 单算子性能
./build/bin/test-backend-ops test -b CUDA0
./build/bin/test-backend-ops test -b CUDA0 -o AFFINE

# 5. 端到端性能（pp 看 GEMM，tg 看 GEMV+启动开销）
./build/bin/llama-bench -m model.gguf -ngl 99 -p 512 -n 128 -r 5

# 6. 强制路径对照（重新 cmake 配置）
cmake -S . -B build -DGGML_CUDA_FORCE_CUBLAS=ON
cmake -S . -B build -DGGML_CUDA_FORCE_MMQ=ON
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 ./binary ...

# 7. 确认编译架构与 backend feature
grep CMAKE_CUDA_ARCHITECTURES build/CMakeCache.txt
```

---

## 八、审计覆盖清单

已确认覆盖（对标 v0.25.3 / `353b63b4`）：

```text
✅ backend registry / device / buffer / pinned host buffer / feature 暴露
✅ CUDA 初始化与设备探测、虚拟设备→物理设备映射
✅ stream / event / synchronization 全清单
✅ 显存池（LEG / VMM）/ RAII workspace / actual_size 陷阱
✅ CUDA Graph 录制 / 重放 / 热身状态机 / LRU 淘汰 / 兼容边界
✅ concurrent streams / fork-join / copy_event
✅ NCCL / internal allreduce / butterfly 三层通信
✅ 全量算子 dispatch 总表（~150 case）
✅ supports_op / supports_buft / offload_op
✅ 三段式算子模板 + 自定义算子 11 项检查清单
✅ CPU fallback 与 test-backend-ops
✅ CMake / CUDA architectures / template-instances
✅ legacy quant / K-quant / IQ 码本 / MXFP4 / NVFP4
✅ MMVQ / MMQ / MMVF / MMF / cuBLAS 五路
✅ MUL_MAT_ID / MoE / expert sorting / graph 同步约束
✅ FlashAttention Vec / Tile / MMA-F16 + GQA + sparse + online softmax
✅ host/device 转换与 workspace 布局
✅ PDL / cp.async / MMA / dp4a
✅ 调试、sanitizer、Nsight、benchmark 方法
```

**已登记但属“后续专项精读”**（不影响主干能力，按需展开）：

```text
ssm-scan.cu / gated_delta_net.cu / gla.cu / lightning-indexer.cu / dsv4-hc.cu
conv2d / conv3d / pool / upscale
top-k / argsort / argmax
opt-step-adamw / opt-step-sgd / cross-entropy-loss
```

这些算子的底层机制（stride / view / workspace / stream / reduction / template dispatch / supports_op / CPU fallback）已在模块一~三中完整覆盖。
