# NVIDIA GPU 推理优化：硬件架构与运行原理系统指南

> **面向读者**：需要在 NVIDIA 最新架构（Hopper H100/H200、Blackwell B200/RTX 50/RTX PRO）上进行高性能算子开发、算子库/推理框架开发与推理优化的工程师。
> **定位**：不是 CUDA 语法教程，而是"硬件如何工作 → 映射到推理优化决策"的系统梳理。所有关键参数附官方文档/白皮书参考链接。
> **硬件锚点**：以 Hopper（GH100）与 Blackwell（GB202 消费 / B200 数据中心）为主线，向上兼容 Ada/Ampere 的差异点单独标注。

---

## 目录

1. [GPU 执行模型总览](#一gpu-执行模型总览)
2. [SM 微架构详解](#二sm-微架构详解)
3. [内存层次结构](#三内存层次结构)
4. [Tensor Core 演进与矩阵指令](#四tensor-core-演进与矩阵指令)
5. [Hopper 专有特性](#五hopper-专有特性)
6. [Blackwell 专有特性](#六blackwell-专有特性)
7. [精度格式与量化硬件基础](#七精度格式与量化硬件基础)
8. [推理负载的硬件特征画像](#八推理负载的硬件特征画像)
9. [算子开发核心优化技术](#九算子开发核心优化技术)
10. [推理框架/引擎层面优化](#十推理框架引擎层面优化)
11. [性能分析与调优工具链](#十一性能分析与调优工具链)
12. [按架构的优化检查清单](#十二按架构的优化检查清单)
13. [附录：关键硬件参数速查与参考](#十三附录)

---

## 一、GPU 执行模型总览

### 1.1 三级执行层次

```
Kernel（内核函数，由 Host 启动）
└── Grid（网格）
    └── Thread Block / CTA（线程块，≤1024 线程，常驻于一个 SM 上）
        └── Warp（线程束，32 线程，SM 调度与执行的基本单位）
            └── Thread（线程）
```

推理优化的第一性原理：**GPU 是吞吐型处理器，靠海量并发隐藏延迟**。所有优化最终落在两个问题上：
- 并发够不够（occupancy / parallelism）
- 数据供给够不够快（memory hierarchy 是否喂饱计算单元）

### 1.2 SIMT 与 Warp 语义

- SM 一次取指/发射以 **warp（32 线程）** 为单位，warp 内线程执行同一条指令（SIMT = Single Instruction Multiple Threads）。
- **分支发散（divergence）**：warp 内 if/else 不同分支会串行执行两遍，推理 kernel 中处理变长序列、MoE 路由、稀疏 mask 时是首要性能杀手。
- **warp 内通信**：`__shfl_sync`/`__reduce_add_sync` 等 shuffle 指令可在寄存器级做 warp 归约，是 softmax/rmsnorm/attention reduce 的标准原语（比 shared memory 快一个数量级）。
- **独立线程调度（Volta 起）**：每线程有独立 PC，warp 内线程可交错执行不同分支——但性能上仍需按 warp 聚合思考。

### 1.3 调度视角：一个 SM 在干什么

- 每个 SM 有 **4 个 warp scheduler**（处理块，sub-partition），每周期每 scheduler 可发射 1 条指令 → 单 SM 每周期最多发射 4 warp 指令。
- 常驻 warp 数上限：Hopper/Blackwell 每 SM 最多 **64 warp（2048 线程）**；由寄存器数、shared memory、block 数三者共同限制。
- **延迟隐藏模型**：指令延迟（如 FP32 FMA ~4 cycle、L2 命中 ~200 cycle、HBM ~400–800 cycle）需要足够的可发射 warp 来覆盖——这就是 occupancy 的本质。推理 kernel 常因寄存器占用过高而 occupancy 不足，导致访存延迟暴露。

### 1.4 关键量化参考（H100 SXM）

| 资源 | 数值 | 推理优化含义 |
|---|---|---|
| SM 数量 | 132 | Grid 至少 132×若干 才能填满 |
| 每 SM 寄存器 | 65536 × 32bit | 每线程 >255 寄存器会 spill 到 local memory（致命） |
| 每 SM shared mem | 228 KB（可配） | FlashAttention 的 tile 尺寸由它决定 |
| 每 SM 常驻线程 | 2048 | occupancy 上限 |
| 每 SM 每周期 | 4 warp 发射 | 指令级并行（ILP）可补 occupancy 不足 |
| 时钟 | ~1.98 GHz boost | 用于把 cycle 数换算成时间 |

---

## 二、SM 微架构详解

### 2.1 SM 内部结构（Hopper SM 为例）

每个 SM = 4 个 processing block（sub-partition），每个 block 含：

| 单元 | Hopper 每 SM 合计 | 说明 |
|---|---|---|
| FP32 CUDA core | 128 | 每 block 32 个 |
| INT32 | 64 | 与 FP32 共享发射端口的一半（注意 INT 计算会挤占 FP32 吞吐） |
| FP64 | 64 | 仅数据中心 SM；消费级 1/64（Ada SM FP64 = 2 个） |
| Tensor Core | 4（每 block 1） | 第 4 代；Blackwell 第 5 代 |
| LD/ST 单元 | 32 | 决定访存指令吞吐 |
| SFU（特殊函数） | 16 | exp/log/rsqrt——softmax、激活函数瓶颈所在 |
| Warp scheduler | 4 | 每 scheduler 16 个常驻 warp 槽位 |
| L1/Shared | 228 KB 统一 | L1 与 shared 物理共享，可配置比例 |

**推理要点**：
- **INT/FP32 发射竞争**：index 计算、mask、地址生成走 INT 单元；在 FP32 密集 kernel 中减少 INT 指令（如用向量访存减少地址计算、预计算 stride）可直接提升吞吐。
- **SFU 是隐性瓶颈**：softmax 的 exp、gelu 的 tanh/erf、rsqrt 都走 SFU（每 SM 16 个，吞吐是 FP32 的 1/8）。大 kernel 中归一化/激活占比高时，SFU 吞吐先于 Tensor Core 成为瓶颈——这是"融合算子"收益的一大来源（减少往返 + 分摊）。

### 2.2 发射与流水线

- **指令发射**：scheduler 每周期从就绪 warp 选 1 条指令发射。warp "就绪"要求源操作数可用（记分板 scoreboard 机制）。
- **流水线深度**：访存 → L2 → L1 → 寄存器，各级延迟逐级降低；计算（FMA/Tensor Core）吞吐远高于访存——**算力利用率 = 计算指令占比 × 数据复用度**。
- **异步执行**：`cp.async`（Ampere）/ TMA（Hopper）让访存与计算真正重叠；Hopper 起 Tensor Core 指令（wgmma）本身也是异步的——"发射即返回，fence 才等待"，这与 Ampere 的同步 mma 是编程模型上的本质差异。

### 2.3 Occupancy 的推理场景修正

教科书追求高 occupancy，但推理 kernel 有反例：
- **FlashAttention 类 kernel**：为复用数据需要大 shared tile + 大量寄存器，典型 occupancy 只有 1–2 个 CTA/SM，靠 **ILP（指令级并行）+ 异步流水线**而非 occupancy 隐藏延迟。
- **正确判据**：不是 occupancy 越高越好，而是"任一时刻每个 scheduler 都有可发射的就绪 warp"。低 occupancy + 深流水线（cp.async 多级 buffer）同样可行。

---

## 三、内存层次结构

推理性能 ≈ 90% 是内存问题。层次从快到慢：

### 3.1 层次全景（H100 数值）

| 层级 | 容量 | 带宽 | 延迟 | 控制方式 | 作用与功能 |
|---|---|---|---|---|---|
| 寄存器 | 256KB/SM（64K×32b） | ~最高 | ~1 cycle | 编译器分配 | 指令操作数空间，每线程私有；spill 即性能灾难 |
| Shared Memory | 228KB/SM | ~19 TB/s | ~20–30 cycle | 程序员显式管理 | 片上可控 cache；tile 暂存/CTA 内数据复用的核心载体；32 bank，冲突即串行 |
| L1 Cache | 与 shared 共享 228KB | 同上 | ~30 cycle | 硬件自动 | SM 级自动缓存；与 shared 共享物理 SRAM；兜非规则访存局部性 |
| L2 Cache | 50MB（H100）/~96MB（B200） | ~5.5 TB/s | ~200 cycle | 硬件自动，可设 persistence | 全 GPU 统一 LLC，跨 GPU 访存落点；热权重/KV 常驻提升有效带宽 |
| HBM3/HBM3e | 80–141GB | 3.35–4.8 TB/s | ~400–800 cycle | 物理显存 | 数据最终归宿；容量决定能否装下，带宽决定 decode 吞吐上限 |
| NVLink | — | 900 GB/s（H100） | μs 级 | NCCL/NVSHMEM | GPU 间直连；TP/EP 通信能否留在 NVLink 域是分布式推理生死线 |
| PCIe | — | 64 GB/s（Gen5 x16） | μs 级 | cudaMemcpy/UVM | 主机↔设备、无 NVLink 跨卡兜底；带宽低一个量级 |

### 3.2 Shared Memory：推理 kernel 的核心战场

- **组织**：32 个 bank，每 bank 4B/周期。warp 访问时若多线程命中同一 bank 不同地址 → **bank conflict**，访问串行化。
- **经典规避**：swizzling / XOR 映射 / padding。Tensor Core 的 ldmatrix/wgmma 输入要求特定布局，CuTe/flash-attention 中的 `Swizzle<3,3,3>` 类布局函数就是为解决此问题。
- **编程角色**：global→shared 做数据暂存（tiling），是所有 GEMM/attention kernel 的基本骨架；Ampere 起 `cp.async` 可绕过寄存器直达 shared（省寄存器、支持异步流水）。
- **Hopper 起**：TMA（见 §5）进一步把"拷贝"本身硬件化，shared memory 申请（`cudaFuncAttributeMaxDynamicSharedMemorySize`）可吃满 227KB。

### 3.3 L2 Cache 的推理价值

- 50MB L2 是推理的隐形加速器：**同一 batch 内不同 token 复用权重**时，热权重常驻 L2 可使有效带宽远超 HBM 标称。
- `cudaAccessPolicyWindow`（persistence）可 pinning 热数据（如 embedding 表、常用权重段）到 L2——对 MoE 模型路由不均的热点 expert 尤其有效。
- Blackwell L2 进一步增大且支持更细粒度策略；推理框架中权重布局应配合 L2 亲和性（expert 权重连续存放）。

### 3.4 HBM 与带宽利用率

- **有效带宽 ≠ 标称带宽**：连续 128B 对齐访问才能吃满；随机访问/stride 大时利用率可跌至 <30%。
- **合并访存（coalescing）**：warp 的 32 线程访问应尽量落在连续的 128B（理想）或最少 cache line 数内——布局转换（如 KV cache 的 `[B, H, S, D]` vs `[B, S, H, D]`）本质是改变访存合并模式。
- **推理特例——decode 是纯带宽题**：逐 token 生成时每步读全量权重一次，tokens/s ≈ 带宽 ÷ 模型字节数（batch=1 时）。H100 80GB BF16 模型：3.35TB/s ÷ 160GB ≈ 21 tok/s 理论上限——**这就是为什么量化（FP8→字节减半→吞吐翻倍）在 decode 阶段近乎线性收益**。

### 3.5 统一内存与主机内存

- `cudaMallocManaged` 统一寻址适合原型；生产推理用 pinned memory（`cudaHostAlloc`）+ 显式 `cudaMemcpyAsync` + stream 重叠 H2D/D2H。
- GH200 的 NVLink-C2C（900GB/s CPU↔GPU 一致性互连）使 CPU 侧 480GB LPDDR5X 可作为 GPU 的"二级显存"——超大模型分层推理的新玩法（把冷 expert/KV cache 放 CPU 内存按需换入）。

---

## 四、Tensor Core 演进与矩阵指令

### 4.1 五代 Tensor Core 一览

| 代际 | 架构 | 指令 | 支持精度 | 关键能力 |
|---|---|---|---|---|
| 1 | Volta | `mma.sync` (HMMA) | FP16→FP32 | 首次引入，4×4 矩阵片 |
| 2 | Turing | `mma.sync` | +INT8/INT4/INT1 | 多精度推理起点 |
| 3 | Ampere | `mma.sync` | +TF32、BF16、**结构化稀疏 2:4** | 稀疏 2×；ldmatrix 加载 |
| 4 | Hopper | **`wgmma.mma_async`** | +**FP8 (E4M3/E5M2)** | warpgroup（128线程）级指令、异步、TMA 配合、SS 模式（双 shared 源） |
| 5 | Blackwell | **`tcgen05.mma`** | +**FP4/FP6（微缩放 MXFP）** | CTA pair（2 SM 协作）、tensor memory（TMEM）暂存累加器、更大指令粒度 |

### 4.2 指令编程模型演进（写 kernel 必须理解的范式转移）

- **Ampere `mma.sync`**：warp（32线程）粒度，同步执行，操作数经 `ldmatrix` 从 shared 载入寄存器。累加器在寄存器。
- **Hopper `wgmma`**：**warpgroup（4 warp = 128 线程）** 粒度异步指令；A 矩阵可在寄存器或 shared，**B 矩阵必须在 shared**（SS 模式双 shared 源是主流），配合 TMA 自动填充；`wgmma.commit_group`/`wait_group` 控制流水线。**寄存器压力骤降**（B 不占寄存器），这是 Hopper GEMM 能做大 tile 的原因。
- **Blackwell `tcgen05.mma`**：**CTA pair（2 个 SM 组成 cluster pair）** 协同执行一条 MMA；引入 **Tensor Memory（TMEM）**——片上专用累加器存储，取代寄存器存累加结果，配合 `tcgen05.ld/st` 搬运。**cuBLAS/CUTLASS 已封装**，手写 kernel 建议直接基于 CUTLASS 3.x 的 SM100 抽象，裸写 tcgen05 成本极高。

### 4.3 结构化稀疏（2:4）

Ampere 起 Tensor Core 支持"每 4 元素保留 2 个非零"的硬件稀疏，dense 吞吐 ×2。推理应用：权重可离线做 2:4 稀疏化（精度损失需逐模型验证），`mma.sp` 指令；实际收益受限于稀疏格式元数据开销与精度，业界采用率低于量化——**优先级：量化 > 稀疏**。

### 4.4 算力账（dense 峰值，选型/kernel 设计基准）

| GPU | BF16 | FP8 | FP4 | 说明 |
|---|---|---|---|---|
| A100 SXM | 312 TF | — | — | 稀疏 624 |
| H100 SXM | 990 TF | 1,979 TF | — | Transformer Engine |
| H200 SXM | 990 TF | 1,979 TF | — | 算力同 H100，带宽 4.8TB/s |
| B200 | 2,250 TF | 4,500 TF | 9,000 TF | 第 5 代 TC |
| RTX 5090 | ~105 TF（FP32口径）/ TC 峰值 ~838 | ~1,676 | ~3,352 AI TOPS | 消费级 FP8/FP4 可用 |
| RTX PRO 6000 | — | ~500 TF | ~2,000 TF | 96GB GDDR7 |

> kernel 设计目标不是打满峰值（GEMM 实测 MFU 天花板 ~80–90%，推理端到端更低），而是理解**算术强度（arithmetic intensity = FLOPs/bytes）阈值**：当 AI < 峰值算力/带宽（H100 ≈ 990e12/3.35e12 ≈ 295 FLOP/B）时访存受限。decode（AI≈2）永远访存受限，prefill（AI 随 batch/seq 增长）可算力受限——**同一模型不同阶段瓶颈相反**，优化策略完全不同。

---

## 五、Hopper 专有特性

### 5.1 TMA（Tensor Memory Accelerator）

- **本质**：独立的硬件 DMA 引擎，由单线程发起多维 tensor 的 global↔shared 块拷贝，异步执行。
- **对比 `cp.async`**：cp.async 由 warp 内每线程负责一段地址计算；TMA 由描述符（`CUtensorMap`，host 侧预建）描述多维布局 + 边界处理 + swizzle，**一条指令搬一个 tile**，零线程参与地址计算。
- **推理价值**：权重/KV tile 加载零开销化；配合 mbarrier 实现 producer-consumer 流水线（`mbarrier.arrive.expect_tx`）。CUTLASS 3.x、TRT-LLM 的 Hopper kernel 全面基于 TMA。
- **cluster 支持**：TMA 可向 cluster 内其他 SM 的 shared 写（multicast），多 SM 共享同一权重 tile——大 batch GEMM 的 L2/shared 两级复用。

### 5.2 Thread Block Cluster + Distributed Shared Memory (DSM)

- **cluster**：最多 8（H100 可移植上限）/16（不可移植）个 CTA 组成 cluster，**同驻一个 GPC**，可直接读写彼此的 shared memory（`cluster.map_shared_rank`）。
- **推理价值**：跨 SM 的权重 tile 复用、split-K GEMM 的中间结果交换不走 global memory；FlashDecoding 的 split-K 归约、MoE 的 token 分发可用 DSM 降延迟。
- **代价**：cluster 启动开销大、调度受限（必须整 cluster 驻留），小 kernel 慎用。

### 5.3 Transformer Engine 与 FP8

- **硬件**：第 4 代 TC 原生 E4M3/E5M2；Transformer Engine 是"硬件 FP8 + 软件动态 scaling"的组合——每层维护 scale/amax 历史，per-tensor（或 Hopper 后期 per-block）缩放。
- **推理侧**：FP8 GEMM 吞吐 = BF16 ×2，权重量化后显存/带宽减半；精度上 weight FP8 + KV cache FP8 是主流甜点（attention 中 QK^T/PV 也可 FP8，需 per-head/per-block scale）。
- **编程入口**：不建议手写 FP8 GEMM——用 cuBLASLt/TE/TRT-LLM kernel，scale 管理是正确性重灾区（amax 溢出、延迟更新策略）。

### 5.4 其他 Hopper 特性

- **异步事务屏障 mbarrier**：比 `__syncthreads()`/`cuda::barrier` 更细粒度，TMA 流水线的配套同步原语。
- **程序依赖启动（PDL, programmatic dependent launch）**：`cudaTriggerProgrammaticLaunchCompletion` 让后继 kernel 在前者未完全结束时提前启动——**kernel 间气泡消除**，对逐层小 kernel 串行的推理路径收益明显（TRT-LLM 已使用）。
- **DPX 指令**（动态规划加速）：基因组比对类，LLM 无关。
- **HBM3e（H200）**：同 SM/算力，纯带宽升级——推理 decode 场景的性价比升级点。

---

## 六、Blackwell 专有特性

### 6.1 第 5 代 Tensor Core + FP4/FP6

- **微缩放格式**：NVFP4（E2M1 + FP8 scale，block size 16）、MXFP4/MXFP6（E3M2/E2M3，block 32）。相比 FP8 再减半权重字节，decode 吞吐理论再翻倍。
- **精度现实**：FP4 权重量化需 GPTQ/AWQ 类校准 + 敏感层回退（首末层/embedding 常保 FP8/BF16）；NVFP4 比 MXFP4 精度更好（scale 粒度细）。推理引擎已内置（TRT-LLM/vLLM FP4 支持 SM100+）。
- **编程**：tcgen05 指令族 + TMEM；CUTLASS SM100 kernel、cuBLASLt 自动选择。**手写 FP4 kernel 的成本/收益比极低，优先走库**。

### 6.2 双 die 架构（B200）

- B200 = 2 个 die，10 TB/s die-to-die（NV-HBI）互联，对 CUDA 程序**呈现为单一 GPU**（192 个 SM 逻辑统一、统一 L2 地址空间）。
- 对开发者的影响主要在**性能调优细节**：跨 die 访存延迟略高于 die 内（透明但存在），kernel 无需感知拓扑，但 profiling 时 `die` 维度指标可用于定位 NUMA 类效应。

### 6.3 NVLink5 + NVL72

- 单 GPU NVLink5 1.8 TB/s；NVL72 机柜内 72 GPU 全互联（13.5 TB/s 聚合），**一个 NVLink 域 = 一个"大 GPU"**。
- 推理含义：TP（tensor parallel）可扩到 72 卡而通信仍在 NVLink 域内（不降级到 IB）；MoE 模型的 expert parallel（EP）大规模部署的硬件基础——DeepSeek 类 671B MoE 在 NVL72 上可做 EP=72。
- 编程：NCCL 自动利用；自定义 kernel 可用 NVSHMEM/设备端 NVLink P2P。

### 6.4 消费级 Blackwell（RTX 50）

- GB202 单 die；GDDR7 1792GB/s（5090）；同代 Tensor Core（FP4）但**无 NVLink、无 TMA cluster 部分特性裁剪**（TMA 存在，cluster/DSM 可用但域窄）。
- 本地推理优化要点同 Hopper 主线；显存带宽仍高（GDDR7 1.8TB/s > H100 HBM2e PCIe 版 2TB/s 量级）使 decode 性能可观——5090 32GB 跑 70B FP4 是性价比极致。

### 6.5 新增/增强指令与特性

- `tcgen05` 全套（mma/cp/shift/ld/st 围绕 TMEM）；CTA pair 执行模型。
- **Tensor Memory（TMEM）**：每 SM 256KB 专用累加器存储，与寄存器/shared 并列的第三层片上存储——SM100 GEMM 的累加器不再占寄存器堆，解放出寄存器给 epilogue（融合操作）使用。
- 增强 TMA（im2col、更大描述符）、MMA 与 epilogue 更深的硬件流水。

---

## 七、精度格式与量化硬件基础

### 7.1 格式速查

| 格式 | 位宽 | 结构 | 硬件支持 | 推理用途 |
|---|---|---|---|---|
| FP32 | 32 | 1+8+23 | 全代 | 参考精度 |
| TF32 | 19 | 1+8+10 | Ampere+ | 训练过渡 |
| FP16 | 16 | 1+5+10 | Volta+ | 推理传统主力 |
| BF16 | 16 | 1+8+7 | Ampere+ | **推理当前默认**（动态范围大） |
| FP8 E4M3 | 8 | 1+4+3 | Hopper+ | 权重/激活/KV cache（精度好） |
| FP8 E5M2 | 8 | 1+5+2 | Hopper+ | 梯度用，推理少用 |
| INT8 | 8 | — | Turing+ | 经典量化（SmoothQuant/W8A8） |
| NVFP4 | 4 | E2M1+FP8 scale(16) | Blackwell | 权重（精度最优 FP4） |
| MXFP4/FP6 | 4/6 | E2M1/E3M2+scale(32) | Blackwell | 权重（OCP 标准） |
| INT4 | 4 | — | Turing+ | W4A16（AWQ/GPTQ） |

### 7.2 量化方案的硬件映射

- **W8A8（SmoothQuant）**：权重+激活 INT8/FP8，计算走 Tensor Core 低精度——**吞吐收益最大**，需处理激活离群值。
- **W4A16（AWQ/GPTQ）**：仅权重 INT4，计算反量化到 FP16 走 FP16 TC——**显存收益大、decode 带宽收益大**（权重字节减半以上），prefill 计算无加速。Marlin/ExLlamaV2 kernel 是典范。
- **FP8 KV cache**：KV cache 占显存随 context 线性增长；FP8 化省一半显存且 Hopper+ 原生支持，P99 延迟收益显著。
- **FP4（Blackwell）**：权重 NVFP4 + KV FP8 是 Blackwell 推理甜点组合。

### 7.3 量化正确性要点（开发自验）

- scale 粒度：per-tensor < per-channel < per-group(64/128) < per-block(MX)——越细越准但反量化开销越大。
- 累加器永远是 FP32（TC 硬件如此），低精度只影响输入/存储。
- KV cache 量化对长上下文模型精度最敏感（误差随序列累积），长上下文场景慎用 INT4 KV。

---

## 八、推理负载的硬件特征画像

### 8.1 Prefill vs Decode：两种相反负载

| | Prefill（提示词处理） | Decode（逐 token 生成） |
|---|---|---|
| 计算形态 | 大 GEMM（M=batch×seq） | GEMV/扁 GEMM（M=batch） |
| 瓶颈 | 算力受限（AI 高） | **带宽受限**（AI≈2） |
| 优化方向 | TC 利用率、tiling、FP8 算力 | 权重量化、KV 管理、kernel 融合降 launch 开销 |
| 指标 | TTFT（首 token 延迟） | TPOT/ITL（逐 token 延迟）、吞吐 |

**系统含义**：两阶段对硬件需求相反 → **PD 分离（prefill/decode disaggregation）** 成为主流部署：prefill 用算力型节点（H100/B200），decode 用带宽/显存型节点（H200/H20/RTX PRO 6000），经 KV cache 传输衔接。

### 8.2 Attention 的硬件压力点

- **QK^T**：GEMM，但 K 需转置视角访问 → KV cache 布局决定访存效率。
- **Softmax**：SFU（exp）+ 归约——长序列时不可忽略；online softmax（FlashAttention 核心）把 3 趟访存压成 1 趟。
- **PV GEMM**：依赖 softmax 结果，online 化后与 QK^T 融合成单 kernel。
- **访存量级**：朴素 attention O(S²×D) 访存；FlashAttention 压到 O(S×D²/tile)——**attention 优化的本质是显存 IO 复杂度**，不是 FLOPs。

### 8.3 其他高频算子的特征

| 算子 | 特征 | 优化抓手 |
|---|---|---|
| RMSNorm/LayerNorm | 纯带宽 | 融合进相邻 GEMM epilogue/prologue；warp shuffle 归约 |
| RoPE | 带宽+三角函数 | 融合进 QK 投影 epilogue 或 attention prologue |
| SiLU/GELU/激活 | 带宽+SFU | GEMM epilogue 融合 |
| MoE 路由/dispatch | 不规则访存+all-to-all | grouped GEMM、expert 并行、top-k 用 bitonic/radix |
| KV cache 读写 | decode 每步全读 | PagedAttention 块管理、FP8 KV、分块 attention |
| Sampling | 小算子串行 | 融合 logit 处理、减少 sync |

### 8.4 小 batch 推理的"kernel launch 税"

decode 每步数十个 kernel，每个 kernel 5–10μs launch + 同步开销在低负载下占比可达 20–40%：
- **CUDA Graph**：把整步推理 capture 成一张图重放，消除 launch 开销——vLLM/TRT-LLM decode 路径标配；限制是 shape 必须固定（需 padding 到 bucket）。
- **PDL**（Hopper+）：kernel 间提前启动，图之外的补充。
- **融合**：persistent kernel 把多个算子合一（TRT-LLM 的 fused MHA、allreduce+gemm 融合）。

---

## 九、算子开发核心优化技术

### 9.1 GEMM 类（MLP/投影/MoE expert）

优先级路径：
1. **首选 cuBLASLt / CUTLASS**：官方库对 SM90/SM100 的调优远胜手写（TMA+wgmma+cluster 流水线封装）；CUTLASS 3.x 提供 CuTe DSL 可定制 epilogue 融合（bias、激活、scale）。
2. **手写场景**：非标准 shape（极小 M 的 decode GEMM）、特殊融合（GEMM+allreduce、GEMM+router）、新精度组合库未覆盖。
3. 手写骨架：TMA 载入 tile → mbarrier 流水（3–5 级 stage）→ wgmma/tcgen05 → epilogue（寄存器内融合反量化/激活/residual）→ TMA 写出。

### 9.2 Attention 类

- **直接用现成实现**：FlashAttention-3（Hopper 专用，TMA+wgmma+pingpong）、TRT-LLM fused MHA、cuDNN attention（SDPA 后端）。FA3 在 H100 上 MFU ~75%。
- **手写/改造场景**：新 mask 结构（sliding window、ALiBi 变体）、KV 量化、MLA（DeepSeek 的 latent attention）、cross-attention 变体。
- **decode 专用**：FlashDecoding/FlashDecoding++——沿 sequence 维 split-K 增加并行度（decode 时 M=1 导致 SM 空转），split 结果在 shared/DSM 归约。
- **MLA/GQA/MQA**：KV head 数 < Q head 数 → K/V tile 广播复用，访存减半以上；MLA 的 low-rank 压缩（KV → 512 维 latent）彻底改变布局，需专门 kernel。

### 9.3 归约/归一化类

- 模式：块内 warp shuffle 归约 → shared 跨 warp 归约 → 单值写回；向量化访存（float4/16B）打满带宽。
- 融合：RMSNorm + residual add + （量化 cast） 是标准三合一小 kernel；也可并入 GEMM epilogue（TRT-LLM 做法）。

### 9.4 MoE 专项

- **Grouped GEMM**：不同 expert 的 token 数不等 → 变长 GEMM 组；CUTLASS grouped GEMM / TRT-LLM moe kernel 用 device 侧 schedule。
- **Dispatch/Combine**：token→expert 的 scatter/gather + all-to-all（EP 部署）——NVLink/NVSHMEM 域内做，NVL72 使 EP≤72 通信不出柜。
- **热点**：expert 负载不均 → L2 persistence pinning 热 expert 权重、动态容量因子调优。

### 9.5 KV cache 管理

- **PagedAttention**（vLLM 范式）：KV 按 block（如 16 token/块）池化管理，逻辑页→物理页映射，消除碎片 + 支持 prefix sharing（system prompt 共享）。
- **布局**：`[num_blocks, num_kv_heads, head_dim, block_size]` 类布局让单 token 的 KV 连续；FP8 化 + TMA 批量载入。
- **chunked prefill**：长 prompt 分块 prefill 与 decode 混跑，平滑延迟。

### 9.6 融合策略总览

收益排序（推理场景经验）：
1. **GEMM + epilogue 全家桶**（bias/residual/激活/量化 cast）——收益大且库支持好（CUTLASS/cuBLASLt epilogue）。
2. **Attention 全融合**（QK^T+softmax+PV+RoPE+mask）——FA 系列已解决。
3. **小算子纵向融合**（norm+add+quant、rope+permute）——省带宽+省 launch。
4. **跨层/通信融合**（GEMM+allreduce、attention+allgather）——TRT-LLM/自研 persistent kernel 领域，收益大但复杂度高。

### 9.7 反模式清单

- kernel 内 `__syncthreads()` 与异步拷贝/TMA 混用不当 → 死锁或数据竞争（用 mbarrier/cuda::pipeline）。
- 为 occupancy 减小 tile → 反而降低数据复用，GEMM 变慢。
- decode GEMV 用 GEMM kernel → M=1 时 TC 利用率 <5%；需 GEMV 专用（split-K + 向量化加载）。
- shared memory 超 48KB 未设 `cudaFuncAttributeMaxDynamicSharedMemorySize` → 启动失败。
- 寄存器 >255/线程 → local memory spill，性能崩溃；用 `__launch_bounds__` + `-maxrregcount` 控制，配合 PTXAS `-v` 检查。

---

## 十、推理框架/引擎层面优化

### 10.1 引擎架构认知（以 TRT-LLM/vLLM/SGLang 为参照）

```
调度层：continuous batching（in-flight batching）、PD 分离、prefix cache 命中
  └─ 执行层：CUDA Graph 重放、多流并行（计算/通信/KV传输分离）
      └─ kernel 层：cuBLASLt/CUTLASS/自研 fused kernel
          └─ 通信层：NCCL（TP/EP）、NVSHMEM、IB/RoCE（跨节点）
```

### 10.2 并行策略与硬件对应

| 策略 | 通信原语 | 硬件要求 | 适用 |
|---|---|---|---|
| TP（张量并行） | allreduce/allgather 每步多次 | **必须在 NVLink 域内**（8/72 卡） | 大模型低延迟 |
| PP（流水线并行） | 层间 P2P | 跨节点可行 | 超大模型、气泡需调度掩盖 |
| EP（专家并行） | all-to-all | NVLink 域最佳（NVL72→EP72） | MoE |
| DP（数据并行） | 无（推理） | — | 纯吞吐扩展 |
| CP/SP（上下文/序列并行） | KV allgather | 长上下文 | >32K context |

### 10.3 Continuous Batching 的硬件含义

- 动态拼接不同长度请求 → attention 需 varlen kernel（`cu_seqlens` 模式）；GEMM 部分天然合 batch。
- batch 越大 decode 越接近算力受限拐点（H100 上 70B 模型 batch~数百时 decode 也可打满 TC）——**吞吐最大化部署应让 decode batch 尽量大**，延迟敏感场景反之。
- 调度器按"迭代"而非"请求"调度：每步调度决定下一步 batch 组成——speculative decoding（投机采样）、chunked prefill 都实现在这一层。

### 10.4 Speculative Decoding 的硬件视角

- 草稿模型（小）+ 目标模型验证（并行）→ 把 decode 从"逐 token 带宽受限"变成"小 batch GEMM 验证"，**等效提升算术强度**。
- Medusa/EAGLE/自投机变体均增加少量算力换带宽——在带宽受限的 decode 阶段是免费午餐；验证 GEMM 的 M=候选数 很小，仍是带宽敏感，但接受率>50% 时端到端加速 1.5–2.5×。

### 10.5 部署形态选择

| 形态 | 适用 | 关键优化 |
|---|---|---|
| 单卡（≤70B Q4/FP4） | 5090 32GB / RTX PRO 6000 96GB | CUDA Graph + FP4/FP8 + FA3 |
| 单机多卡 TP | 70B–200B | TP≤8 在 NVLink 域，L40S×8 或 HGX |
| NVL72 机柜 | 400B+ / MoE 高并发 | EP+TP 混合、NVLink 全域、KV 分层（HBM+LPDDR） |
| 多节点 | 超大集群推理 | PD 分离、跨节点 KV 传输、负载均衡 |

---

## 十一、性能分析与调优工具链

### 11.1 工具地图

| 工具 | 用途 | 关键用法 |
|---|---|---|
| **Nsight Compute (ncu)** | kernel 级微观分析 | `ncu --set full ./app`；看 SM throughput、memory throughput、warp stall reasons、L1/L2/HBM 命中率 |
| **Nsight Systems (nsys)** | 时间线/系统级 | `nsys profile`；看 kernel 间隙、stream 重叠、NCCL 通信占比、H2D/D2H |
| **PyTorch Profiler / nsys 插件** | 框架层 | 关联 PyTorch op ↔ CUDA kernel |
| **cuBLASLt/CUTLASS profiler** | GEMM 选型 | `cutlass_profiler` 枚举 tile 配置 |
| **PTXAS 输出** | 编译诊断 | `nvcc -Xptxas -v`：寄存器数、shared 用量、spill 警告 |
| **NVML/nvidia-smi** | 运行时状态 | SM 利用率、显存、功耗、温度节流 |

### 11.2 标准调优流程

1. **nsys 先看全局**：kernel 间隙（launch 税）、通信/计算是否重叠、哪类 kernel 占时多。
2. **ncu 定位单 kernel**：判定瓶颈象限（compute vs memory vs latency vs launch-bound）；看 `Speed Of Light` 和 `Warp State Statistics`（stall on memory barrier/long scoreboard/sfu…）。
3. **对照算术强度**：kernel 实测 FLOPs ÷ 实测 bytes 对比 roofline，判定该往哪侧优化。
4. **改一处测一处**：tile 大小、stage 数、向量宽度、融合范围；自动化用 CUTLASS profiler / 自研 autotune。

### 11.3 常见 stall 诊断速查

| ncu 现象 | 诊断 | 处方 |
|---|---|---|
| `stall_long_scoreboard` 高 | 等 global memory | 增加并行度/cp.async 预取/改善合并访存 |
| `stall_short_scoreboard` 高 | 等 shared memory | bank conflict（改 swizzle/padding） |
| `stall_barrier` 高 | 同步开销 | 减少 `__syncthreads`、改 mbarrier 流水 |
| `stall_wait`/`stall_math` | 计算管线饱和 | 已接近算力上限，或换低精度 |
| SM busy 低 + 无 stall | launch-bound | CUDA Graph、融合、加大 batch |
| TC pipe 利用率低 | 未走 Tensor Core | 检查精度（是否 FP32 路径）、对齐（16B 倍数）、M/N/K 太小 |

---

## 十二、按架构的优化检查清单

### 12.1 Hopper（H100/H200）必做

- [ ] GEMM 走 cuBLASLt/CUTLASS 3.x（wgmma+TMA），确认实际走的是 SM90 kernel 而非 Ampere fallback（ncu 中看指令是否为 wgmma.mma_async）
- [ ] Attention 用 FA3 或 TRT-LLM fused kernel（SM90 特化版）
- [ ] FP8 权重 + FP8 KV cache（Transformer Engine 校准）
- [ ] CUDA Graph 覆盖 decode 全步
- [ ] PDL 开启（TRT-LLM `--enable-pdl` 类似开关）
- [ ] TP 通信不出 NVLink 域；NCCL 拓扑验证（`nvidia-smi topo -m`）
- [ ] H200 场景：利用 141GB 增大 KV cache 池 / 提升最大并发

### 12.2 Blackwell（B200/GB200/RTX 50/RTX PRO）增量项

- [ ] FP4（NVFP4）权重量化评估——先敏感层分析（首末层回退 FP8）
- [ ] kernel 走 SM100 路径（tcgen05/TMEM）；cuBLAS/TRT-LLM 自动，自研 kernel 需 CUTLASS SM100
- [ ] NVL72：EP/TP 跨卡仍处 NVLink 域，验证 NCCL/NVSHMEM 拓扑
- [ ] CTA pair/cluster 利用（库 kernel 已封装；自研需评估 cluster launch 开销）
- [ ] RTX 50 本地推理：FA 后端选择（SM100 消费卡无部分 DC 特性，验证 kernel 兼容性）

### 12.3 通用清单（全架构）

- [ ] 权重布局：GEMM 输出维连续、16B 对齐；量化权重打包紧凑（int4×8/32b）
- [ ] KV cache：FP8、分块、paged、prefix 共享
- [ ] 小算子全融合（norm+add+quant+rope）
- [ ] batch 策略：continuous batching、chunked prefill、PD 分离评估
- [ ] speculative decoding 评估（带宽受限场景）
- [ ] 显存预算审计：权重 + KV 池 + activation + 框架开销 < 物理显存 ×0.95

---

## 十三、附录

### 13.1 关键硬件参数速查

| 参数 | A100 | H100 SXM | H200 | B200 | RTX 5090 | RTX PRO 6000 BW |
|---|---|---|---|---|---|---|
| SM 数 | 108 | 132 | 132 | ~148×2 die | 170 | 188 |
| Shared/SM | 164KB | 228KB | 228KB | 228KB | ~100KB | ~100KB |
| L2 | 40MB | 50MB | 50MB | ~96MB | 96MB | 96MB+ |
| 显存 | 80GB HBM2e | 80GB HBM3 | 141GB HBM3e | 192GB HBM3e | 32GB GDDR7 | 96GB GDDR7 |
| 带宽 | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8 TB/s | 1.79 TB/s | 1.79 TB/s |
| BF16 dense | 312 TF | 990 TF | 990 TF | 2,250 TF | — | ~125 TF FP32 |
| FP8 dense | — | 1,979 TF | 1,979 TF | 4,500 TF | ~1,676 TOPS | ~500 TF |
| FP4 dense | — | — | — | 9,000 TF | ~3,352 TOPS | ~2,000 TF |
| NVLink | 600 GB/s | 900 GB/s | 900 GB/s | 1.8 TB/s | 无 | 无 |
| TDP | 400W | 700W | 700W | 1,000W | 575W | 600W |
| 编程模型 | mma.sync+cp.async | wgmma+TMA+cluster | 同 H100 | tcgen05+TMEM+pair | tcgen05（裁剪） | tcgen05 |

### 13.2 参考文档（官方一级来源）

**架构白皮书/指南**
- Hopper 架构深度解析：https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Blackwell 架构：https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- Ampere 调优指南：https://docs.nvidia.com/cuda/ampere-tuning-guide/
- Hopper 调优指南：https://docs.nvidia.com/cuda/hopper-tuning-guide/
- CUDA C++ Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA 各架构计算能力表：https://developer.nvidia.com/cuda-gpus （A100=8.0, H100/H200=9.0, B200/GB202=10.0/12.0 注意区分 SM90/SM100/SM120）

**指令级文档**
- PTX ISA（mma/wgmma/tcgen05/cp.async/TMA/mbarrier 全套）：https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Math API：https://docs.nvidia.com/cuda/cuda-math-api/

**库与框架**
- CUTLASS（含 CuTe、SM90/SM100 kernel 模板）：https://github.com/NVIDIA/cutlass
- cuBLASLt：https://docs.nvidia.com/cuda/cublaslt/
- Transformer Engine：https://github.com/NVIDIA/TransformerEngine
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM
- FlashAttention（含 FA3 Hopper 实现）：https://github.com/Dao-AILab/flash-attention
- vLLM：https://github.com/vllm-project/vllm ；SGLang：https://github.com/sgl-project/sglang
- NCCL：https://docs.nvidia.com/deeplearning/nccl/ ；NVSHMEM：https://docs.nvidia.com/nvshmem/

**工具**
- Nsight Compute：https://developer.nvidia.com/nsight-compute
- Nsight Systems：https://developer.nvidia.com/nsight-systems
- CUDA Graphs：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs

**精度与量化**
- FP8 格式论文（E4M3/E5M2）：https://arxiv.org/abs/2209.05433
- OCP Microscaling Formats（MXFP4/6/8）：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- SmoothQuant：https://arxiv.org/abs/2211.10438 ；AWQ：https://arxiv.org/abs/2306.00978 ；GPTQ：https://arxiv.org/abs/2210.17323

### 13.3 置信度说明

- SM 结构、指令语义、内存层次机制：高置信度（官方白皮书/PTX ISA 长期稳定）。
- 各代峰值算力/带宽/TDP：高置信度（官方 datasheet）。
- B200 SM 数/L2/消费级 Blackwell shared memory 精确值：中置信度（不同口径有差异，已标"约"）。
- kernel 实测 MFU、speculative decoding 加速比等经验值：中置信度（随实现与负载波动）。

---

*文档完 · 生成于 2026-09-20*
