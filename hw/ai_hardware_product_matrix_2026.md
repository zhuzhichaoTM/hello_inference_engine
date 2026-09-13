# AI 算力硬件产品参数与应用场景分析报告（2026）

> **报告日期**：2026-09-13
> **报告定位**：产品级参数目录 + 应用场景映射，供选型、对标与技术规划使用
> **⚠️ 重要说明**：本次会话中厂商官网/arXiv 抓取被网络策略拦截，**所有参数均无法现场复核**，来源于公开资料与我掌握的厂商发布信息，按"公开资料口径"标注。**采购与招标决策请务必对照官方 datasheet 复核**。
> **口径警示**：不同厂商的 TOPS/TFLOPS **定义不同**（INT8 / FP8 / FP4 / 是否含稀疏 / 是否含 scale 开销），**跨厂商直接比较算力数字是行业最常见的误用**。本报告在每个参数处标注口径。
> **关联文档**：
> - `knowledge/ai_hardware_industry_insights_2026.md`（行业洞察）
> - `knowledge/on_device_quantization_hotspots_hardware_2026.md`（端侧量化与硬件）
> - `knowledge/on_device_llm_survey_2021_2026.md`（端侧文献综述）

---

# 第一部分 · 产品全景地图

## 1.1 产品分类总览

| 类别 | 产品形态 | 典型功耗 | 代表产品 | 主战场 |
|---|---|---|---|---|
| **旗舰训练 GPU** | 板卡 / SXM / 机架 | 700W - 1400W+ | NVIDIA B200/GB300、AMD MI355X | 大模型预训练 |
| **推理优化 GPU** | 板卡 | 300W - 600W | NVIDIA L40S、H200、AMD MI325X | 在线推理服务 |
| **AI ASIC（数据中心）** | 板卡 / Pod | 300W - 700W | Google TPU、AWS Trainium/Inferentia | 自家模型 / 云服务 |
| **数据中心 NPU** | 板卡 / 超节点 | 300W - 400W | 华为昇腾 910C、寒武纪 MLU | 国产合规算力 |
| **GPGPU（国产）** | 板卡 | 300W - 400W | 海光 DCU、摩尔线程 | 国产替代 |
| **AI PC SoC** | 封装 SoC | 15W - 60W | Apple M4、高通 X Elite、Intel Lunar Lake | 端侧 PC |
| **移动 SoC** | 封装 SoC | 3W - 15W | 高通 8 Elite、联发科 D9400、Apple A18 | 手机端侧 |

## 1.2 代际演进时间轴

```
数据中心 GPU
2020 ─── A100 (Ampere)
2022 ─── H100 (Hopper)
2023 ─── H200 (Hopper refresh, HBM3e)
2024 ─── B200 / GB200 (Blackwell)
2025 ─── GB300 / B300 (Blackwell Ultra)
2026 ─── Rubin (规划) ⚠️

AMD Instinct
2021 ─── MI250X (CDNA2, 128GB)
2023 ─── MI300X (CDNA3, 192GB)
2024 ─── MI325X (CDNA3, 256GB)
2025 ─── MI350X / MI355X (CDNA4, 288GB) ⚠️

华为昇腾
2023 ─── 910B
2024 ─── 910C (双芯合封) + CloudMatrix 384 ⚠️
2025+ ── UB/UB-Mesh 架构 (Hot Chips 2025) ⚠️

Google TPU
2023 ─── v5e / v5p
2024 ─── v6e (Trillium)
2025+ ── v7 (Ironwood) ⚠️

端侧
2023 ─── Apple M3 / A17 Pro
2024 ─── Apple M4 / A18, 高通 8 Gen3, 联发科 D9300
2025 ─── 高通 8 Elite, 联发科 D9400, Intel Lunar Lake
```

---

# 第二部分 · 主流厂商产品参数矩阵

## 2.1 NVIDIA

### 产品线参数表

| 产品 | 架构 | 显存 | 显存带宽 | 关键精度算力（口径） | 互联 | 功耗 | 定位 |
|---|---|---|---|---|---|---|---|
| **A100** | Ampere | 40/80GB HBM2e | 1.55 / 2.0 TB/s | FP16 312 TFLOPS（稀疏） | NVLink3 600GB/s | 250-400W | 上一代训练标品 |
| **H100 SXM** | Hopper | 80GB HBM3 | 3.35 TB/s | FP8 ~1,979 TFLOPS（稀疏）/ ~989 稠密 | NVLink4 900GB/s | 700W | 训练主力（存量） |
| **H200** | Hopper | **141GB HBM3e** | **4.8 TB/s** | FP8 同 H100 | NVLink4 900GB/s | 700W | 推理/微调，显存性价比高 |
| **B200** | Blackwell | 192GB HBM3e | **8 TB/s** | **FP4 ~18 PFLOPS（密集口径见注）** | NVLink5 1.8TB/s | ~1000W | 训练主力 |
| **GB200 NVL72** | Blackwell | 72×192GB | 聚合 ~576TB/s 级 | 机架级聚合 | NVLink5 全互联 | 机柜 ~120kW | 超节点训练 |
| **GB300 / B300** | Blackwell Ultra | **288GB HBM3e** | 8 TB/s | FP4 增强 | NVLink5 | ~1400W | 超节点训练（新一代） |
| **L40S** | Ada | 48GB GDDR6 | 864 GB/s | FP8 733 TFLOPS（稀疏） | PCIe | 350W | 推理/图形混合 |
| **Rubin** | 新架构 | 规划中 | — | 规划中 | NVLink6（预期） | — | 下一代 ⚠️ |

**注**：B200 的 FP4 算力厂商常报"20 PFLOPS（稀疏）"、"10 PFLOPS（密集）"等不同口径，**必须核对是否含稀疏加速、是否含 scale 换算**。此为我所见公开资料中最容易混淆的数字之一 ⚠️。

### 应用场景映射

| 场景 | 推荐产品 | 理由 |
|---|---|---|
| 大模型预训练（千卡级） | GB200/GB300 NVL72 | 机架级单逻辑加速器，消除跨节点通信开销 |
| 大模型微调（百卡级） | H200 / B200 | 显存容量决定可微调模型规模（141-192GB） |
| 在线推理（高吞吐） | H200 + FP8/FP4 量化 | HBM3e 带宽 4.8TB/s 对 batch 推理最实用 |
| 推理（成本敏感） | L40S / 量化 A100 | 单位美元吞吐更优 |
| 长上下文推理 | GB300（288GB） | 超大显存直接容纳长 KV Cache |

### 软件栈
CUDA → TensorRT-LLM（推理）→ cuDNN/cuBLAS（算子）→ NCCL（通信）→ Nsight（调优）→ Triton（服务化）。生态完整度是护城河核心。

---

## 2.2 AMD Instinct

### 产品线参数表

| 产品 | 架构 | 显存 | 显存带宽 | 关键精度 | 互联 | 功耗 | 定位 |
|---|---|---|---|---|---|---|---|
| **MI250X** | CDNA2 | 128GB HBM2e | 3.2 TB/s | FP16/BF16 383 TFLOPS | Infinity Fabric | 560W | 上一代 HPC |
| **MI300X** | CDNA3 | **192GB HBM3** | 5.3 TB/s | FP8 ~2,615 TFLOPS（稀疏） | IF 896GB/s | 750W | 推理大显存标杆 |
| **MI300A** | CDNA3 | 128GB 统一 | — | 同 MI300X | IF | 760W | APU 形态，HPC |
| **MI325X** | CDNA3 | **256GB HBM3e** | **6.0 TB/s** | FP8 增强 | IF | 1000W | **单卡显存容量领先** |
| **MI350X/MI355X** | CDNA4 (gfx950) | **288GB HBM3e** | ~8 TB/s | **FP4 / FP8** | IF | ~1000W+ | 对标 B200 ⚠️ |

**⚠️ 全部为公开资料口径，未现场复核。ROCm 10 官方支持列表覆盖 MI355X/MI350X/MI350P（gfx950）、MI325X/MI300X/MI300A（gfx942）、MI250/MI210（gfx90a）、MI100（gfx908）** ⚠️。

### 应用场景映射

| 场景 | 推荐产品 | 理由 |
|---|---|---|
| 大显存推理 | **MI325X（256GB）/ MI355X（288GB）** | 单卡容量可容纳更大模型与更长 KV |
| 训练（成本敏感） | MI355X | FP4 支持 + 容量优势，需预留 kernel 调优人力 |
| HPC 混合负载 | MI300A | CPU+GPU 统一内存 |

### 软件栈与迁移成本（关键）
ROCm 10（Linux + Windows）⚠️ → HIP（CUDA 移植层）→ MIOpen/RCCL → vLLM 0.27.0 / SGLang 0.5.15（推理，官方验证）⚠️ → PyTorch 2.11-2.13 / JAX 0.10-0.11（训练）⚠️

**已验证的迁移风险** ✅（ROCm 官方 HIPIFY 文档，3-0 票）：
1. HIP **不是** CUDA 的完全替代，**无 HIP 等价物的 CUDA 库或第三方库完全无法翻译**；
2. **翻译成功 ≠ 性能对齐**，NVIDIA 上优化的代码移植后需额外重做。

→ **选型含义**：AMD 路线的真实成本 = 硬件成本 + **kernel 调优人力成本**。

---

## 2.3 华为昇腾

### 产品线参数表

| 产品 | 形态 | 显存 | 关键精度 | 互联 | 功耗 | 定位 |
|---|---|---|---|---|---|---|
| **昇腾 310** | 边缘推理 | — | INT8 | — | 低功耗 | 边缘推理 |
| **昇腾 910B** | 训练卡 | 64GB HBM ⚠️ | FP16/INT8 | HCCS | ~350W | 训练主力 |
| **昇腾 910C** | 双芯合封 | HBM ⚠️ | FP16/INT8 | HCCS | ~350W+ | 对标 H100 级 ⚠️ |
| **CloudMatrix 384** | 超节点 | 384 卡聚合 | — | 自研光互联 ⚠️ | 机柜级 | 超节点训练/推理 |
| **UB/UB-Mesh** | 互联架构（Hot Chips 2025） | — | — | 统一取代 PCIe/NVLink/RoCE ⚠️ | — | 下一代超节点 |

**UB/UB-Mesh 目标指标** ⚠️：单芯片带宽 100 Gbps → **10 Tbps（1.25 TB/s）**；跳跃延迟 微秒 → **~150 纳秒**；MTBF 提升 100 倍；语义从异步 DMA → **NVLink 式同步 load/store**；目标规模 **100 万处理器** 单超节点。

### 应用场景映射

| 场景 | 推荐 | 理由 |
|---|---|---|
| 国产合规训练 | CloudMatrix 384 | 系统规模对冲单卡差距 |
| 国产合规推理 | vllm-ascend | 走 vLLM 插件而非直接写 CANN，降低适配成本 |
| 边缘推理 | 昇腾 310 | 低功耗 |

### 生态要点
CANN（对标 CUDA）⚠️；**vLLM Hardware Pluggable RFC（vLLM 社区 + 华为昇腾团队联合开发）使昇腾可模块化接入 vLLM** ⚠️，vllm-ascend 已成官方验证后端 ⚠️。

---

## 2.4 Google TPU

| 产品 | 代际 | 显存 | 关键精度 | 互联 | 定位 |
|---|---|---|---|---|---|
| **TPU v5e** | 2023 | — | INT8/BF16 | ICI | 性价比推理 |
| **TPU v5p** | 2023 | — | BF16 | ICI | 大规模训练 |
| **TPU v6e (Trillium)** | 2024 | — | BF16 | ICI | 训练+推理 |
| **TPU v7 (Ironwood)** | 2025+ | — | — | ICI | 新一代 ⚠️ |

架构特征：脉动阵列（Systolic Array）+ HBM + ICI 光缆 2D Torus 拓扑 ⚠️；软件栈 JAX/XLA/Pax，对自家用例优化最深，**外部生态封闭**。

**场景**：Google 内部模型训练、Vertex AI 云推理、Anthropic 等大客户租用。

---

## 2.5 Intel

| 产品 | 形态 | 参数 | 定位 |
|---|---|---|---|
| **Gaudi 3** | 训练/推理加速卡 | 128GB HBM2e，~3.7TB/s，FP8 ⚠️ | 数据中心，市场存在感弱 |
| **Jaguar Shores** | 机架级（规划） | — | 转向机架方案 ⚠️ |
| **Lunar Lake NPU** | AI PC SoC | ~40-48 TOPS ⚠️ | AI PC |
| **Arrow Lake NPU** | AI PC SoC | ~13-48 TOPS（型号差异大）⚠️ | AI PC |
| **Xeon 6** | 服务器 CPU | 内置 AMX，INT8/BF16 加速 | 通用计算 + 轻量推理 |

软件栈：OpenVINO（AI PC 成熟）、oneAPI。

---

## 2.6 Apple Silicon

| 产品 | NPU | 统一内存 | 带宽 | 定位 |
|---|---|---|---|---|
| **M4** | 16 核 NE，**38 TOPS** ⚠️ | 最高 32GB | 120 GB/s | 轻薄本 |
| **M4 Pro** | 16 核 NE | 最高 64GB | 273 GB/s | 高性能本 |
| **M4 Max** | 16 核 NE | 最高 128GB | ~410-546 GB/s ⚠️ | 移动工作站 |
| **M4 Ultra** | 16 核 NE | 最高 512GB ⚠️ | ~819 GB/s ⚠️ | 桌面工作站 |
| **A18 Pro** | 16 核 NE | 8GB | — | iPhone |

**关键架构特性**：统一内存（Unified Memory）——CPU/GPU/NPU 共享同一内存池；**MLX 框架下 CPU↔GPU 执行算子无需数据拷贝** ⚠️，消除了 CUDA 式的 host-device 传输开销。

**关键限制** ⚠️：**MLX 支持的计算设备仅 CPU 与 GPU，不含 Apple Neural Engine（ANE）**；ANE 算力需由 Core ML 调用。形成双轨制：MLX 走 GPU（灵活），Core ML 走 ANE（能效）。

**量化生态** ⚠️：affine / **mxfp4 / nvfp4 / mxfp8** 四种模式；混合比特配方 mixed_2_6/3_4/3_6/4_6；量化 KV Cache；量化模型上直接 QLoRA。

**应用场景**：Mac 本地跑 7B-70B 量化模型、端侧隐私敏感推理、开发者本地微调。

---

## 2.7 高通

| 产品 | NPU | 制程 | 定位 |
|---|---|---|---|
| **Snapdragon 8 Gen 3** | Hexagon NPU | 4nm | 2024 旗舰手机 |
| **Snapdragon 8 Elite** | Hexagon，**45-80 TOPS** ⚠️ | 3nm | 2025 旗舰，端侧 GenAI |
| **Snapdragon X Elite** | Hexagon NPU 45 TOPS ⚠️ | 4nm | Windows AI PC |
| **Snapdragon X Plus** | 简化版 NPU | 4nm | 主流 AI PC |

**低比特支持**：int4 / int8 / w4a16 ⚠️。**生态**：QNN / AI Engine SDK；**llama.cpp 官方 Hexagon backend** ⚠️。

**场景**：手机端侧 LLM（3B-7B int4）、Android AI 应用、Windows on ARM AI PC。

---

## 2.8 联发科

| 产品 | NPU | 官方亮点 | 定位 |
|---|---|---|---|
| **Dimensity 9300** | APU 790 | 上代旗舰 | 2024 旗舰 |
| **Dimensity 9400** | **NPU 890（APU 890）** ⚠️ | **业界首个端侧 LoRA 训练** ⚠️；支持 MoE LLM、Diffusion Transformer、投机解码、SLM+LLM、多模态 | 2025 旗舰 |

**性能宣传** ⚠️：LLM prefill +80%、扩散模型生成 +100%（vs 上代）、MLLM >50 tokens/s、能效 +35%。

**生态**：NeuroPilot SDK。

---

## 2.9 其他重要玩家

| 厂商 | 产品 | 定位 | 备注 |
|---|---|---|---|
| **AWS** | Trainium2 / Inferentia2 | 云自研 ASIC | 训练+推理，Neuron SDK ⚠️ |
| **海光** | DCU（K100-AI 等） | 国产 GPGPU | **ROCm 派生** ⚠️，hipify 适用；迁移难度 40%零改/40%小改/20%大改 ⚠️ |
| **寒武纪** | MLU370 / MLU590 | 国产 NPU | 自研指令集，生态追赶 |
| **摩尔线程** | MTT S 系列 | 国产 GPU | 自研 MUSA 架构 |
| **燧原** | 邃思系列 | 国产训练卡 | 自研 |
| **昆仑芯** | R200/R300 | 百度自研 | 自研 XPU |
| **特斯拉** | Dojo D1 | 自研训练 ASIC | 内部使用 |

---

# 第三部分 · 参数口径陷阱（选型必读）

## 3.1 算力数字的四种"注水"方式

| 陷阱 | 说明 | 后果 |
|---|---|---|
| **稀疏 vs 稠密** | 厂商常报含结构化稀疏的算力（可达稠密 2x） | 实际业务若无法利用稀疏，算力打对折 |
| **精度口径不一** | INT8 / FP8 / FP4 的 TOPS 数量级差异巨大 | 跨精度比较无意义 |
| **是否含 scale 开销** | NVFP4 实际约 4.5 bit/value（含 scale）⚠️ | 标称 4-bit 实际压缩比低于 4x |
| **峰值 vs 实际利用率** | 厂商报理论峰值，实际 kernel 利用率常 30-60% | 需看实测 benchmark 而非纸面参数 |

## 3.2 显存带宽才是推理的真瓶颈

对自回归解码（batch=1、逐 token），**算力利用率极低，性能几乎完全由显存带宽决定**：

| 产品 | 带宽 | 相对解码速度参考 |
|---|---|---|
| A100 80GB | 2.0 TB/s | 1.0x |
| H100 | 3.35 TB/s | ~1.7x |
| H200 | 4.8 TB/s | ~2.4x |
| B200 | 8.0 TB/s | ~4.0x |
| MI325X | 6.0 TB/s | ~3.0x |

→ **选型含义**：推理场景优先看 HBM 带宽与容量，而非 FP4 峰值算力。

## 3.3 端侧 TOPS 与云端的不可比性

端侧 NPU 的 38-80 TOPS ⚠️ 与数据中心 GPU 的 1000+ TFLOPS 不是同一语境：
- 端侧 TOPS 多为 **INT8 稀疏口径**，实际 LLM 推理利用率常 <20%；
- 端侧受**内存带宽与 TDP** 双重限制，峰值算力几乎无法达到；
- **端侧真正该看的是能效（tokens/Joule）与首 token 延迟**，而非 TOPS。

---

# 第四部分 · 场景化产品选型对照

## 框 4-1 · 按场景的产品推荐矩阵

| 场景 | 规模 | 首选 | 次选 | 关键指标 | 量化策略 |
|---|---|---|---|---|---|
| **大模型预训练** | 千卡+ | GB200/GB300 NVL72 | TPU v7 / 昇腾 CloudMatrix | 互联带宽、FP8 训练算力 | FP8 训练 |
| **大模型微调** | 百卡级 | H200 / B200 | MI355X | 显存容量（141-192GB） | QLoRA / FP8 |
| **在线推理（高吞吐）** | 集群 | H200 | MI325X | **HBM 带宽 > 峰值算力** | FP8 / NVFP4 |
| **推理（成本敏感）** | 中小 | MI325X | L40S | 容量/美元 | NVFP4 |
| **长上下文推理** | 集群 | GB300（288GB） | MI355X（288GB） | 显存容量 | 量化 KV Cache |
| **国产合规训练** | 千卡 | 华为 CloudMatrix 384 | 海光 DCU | 合规 + 系统规模 | INT8 / W8A8 |
| **国产合规推理** | 集群 | 昇腾 + vllm-ascend | 海光 DCU | 生态适配成本 | INT8 |
| **AI PC（Mac）** | 单机 | Apple M4 Max/Ultra | — | 统一内存带宽 | mxfp4/nvfp4 + 量化KV |
| **AI PC（Win）** | 单机 | 高通 X Elite | Intel Lunar Lake | NPU TOPS + 生态 | int4 / w4a16 |
| **手机端侧** | 单机 | 高通 8 Elite | 联发科 D9400 | 能效 tokens/Joule | int4 |
| **边缘推理** | 边缘节点 | 昇腾 310 / Jetson | — | 功耗 | INT8 |

## 框 4-2 · 三大阵营对比（决策速览）

| 维度 | NVIDIA 阵营 | AMD 阵营 | 国产阵营 |
|---|---|---|---|
| **硬件竞争力** | 最强（代际领先） | 强（容量领先） | 中（系统级对冲） |
| **软件生态** | CUDA（壁垒最高） | ROCm（追赶中） | CANN / ROCm派生 |
| **迁移成本** | — | 中（kernel 调优人力）✅ | 高（华为）/ 低（海光）⚠️ |
| **供给风险** | HBM/封装受限 ⚠️ | 同上 | 制程受限 ⚠️ |
| **合规性** | 受限场景不可用 | 受限场景不可用 | 可满足合规 |
| **性价比** | 溢价最高 | 较优 | 政策性优势 |

---

# 第五部分 · 选型决策 Checklist

选型时建议逐项验证：

1. **算力口径**：厂商报的是稀疏还是稠密？什么精度？是否含 scale 开销？
2. **显存容量**：目标模型 + KV Cache + 激活峰值是否装得下？（这是最硬的约束）
3. **显存带宽**：解码阶段的实际瓶颈（不是算力）
4. **互联能力**：Scale-up（NVLink/IF/UB）与 Scale-out（IB/RoCE/以太网）是否满足集群拓扑
5. **软件栈成熟度**：目标框架（PyTorch/vLLM/SGLang）是否官方支持该硬件
6. **迁移成本**：是否需重写自定义 kernel？需多少 GPU 性能工程师人力？
7. **供给可得性**：HBM/先进封装产能是否锁定
8. **功耗与散热**：机柜功率密度、是否需液冷改造
9. **端侧额外项**：能效（tokens/Joule）、首 token 延迟、内存容量上限、动态图支持度
10. **合规**：出口管制与国产化要求

---

# 第六部分 · 数据可信度说明

## 6.1 本次会话的验证状态

| 状态 | 内容 |
|---|---|
| ✅ **三票对抗验证确认**（5 条，均来自 NVIDIA/AMD 官方文档） | ①NVFP4 格式定义（E2M1 + 每 16 值 FP8 scale + 每张量 FP32 二级）②DeepSeek-R1-0528 量化到 NVFP4 退化 ≤1%、AIME 2024 高 2% ③NVFP4 内存压缩 3.5x vs FP16 / 1.8x vs FP8 ④HIP 非 CUDA 完全替代、部分库无法翻译 ⑤翻译成功 ≠ 性能对齐 |
| ⚠️ **公开资料口径，未现场复核** | 本报告**绝大多数产品参数**（各 GPU/NPU 的显存、带宽、算力、功耗、TOPS）均属此类 |

## 6.2 关键局限（务必注意）

1. **参数未复核**：本次会话中 arXiv、厂商官网、HuggingFace 等抓取均被网络策略拦截，**所有硬件规格均来自公开资料与既有知识，未逐条对照官方 datasheet**。特别是 2025-2026 年新发布产品（Blackwell Ultra、MI355X、TPU v7、Rubin）的规格存在时效性风险。
2. **口径不统一**：跨厂商算力数字**不可直接比较**（见第三部分）。
3. **性能数字为厂商自报**：1.8TB/s、10Tbps、38 TOPS、+80% 等均为厂商宣传口径，**未见独立第三方复现**。
4. **产品路线图不确定**：Rubin、Jaguar Shores、TPU v7 等尚在规划或早期发布阶段，规格可能变化。

## 6.3 建议的复核方式

对报告中的具体参数，建议通过以下官方渠道复核：
- NVIDIA：官方产品页 + NVIDIA Developer Blog
- AMD：AMD Instinct 产品页 + ROCm 官方文档
- 华为：昇腾社区 + 华为官方发布会材料
- Google：Google Cloud TPU 文档
- Apple：Apple Newsroom + Apple Developer（MLX/Core ML 文档）
- 高通/联发科：官方产品页 + 开发者文档

---

# 附：主要参考来源

- [NVIDIA NVFP4 官方博客](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) ✅
- [AMD HIPIFY 官方文档](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html) ✅
- [AMD ROCm 文档](https://rocm.docs.amd.com/en/latest/index.html) ⚠️
- [vLLM Hardware Plugin 博客](https://blog.vllm.ai/2025/05/12/hardware-plugin.html) ⚠️
- [华为 UB/UB-Mesh（Hot Chips 2025 转述）](https://blog.csdn.net/fastboy_abc/article/details/158705107) ⚠️
- [海光 DCU 迁移指南](https://www.zdyedu.cn/blog/cuda-to-hip-dcu-migration-guide/) ⚠️
- [MLX (GitHub)](https://github.com/ml-explore/mlx) ⚠️
- [TrendForce](https://trendforce.com/) ⚠️
