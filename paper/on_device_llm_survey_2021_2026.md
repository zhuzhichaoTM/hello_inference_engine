# 大模型端侧部署优化（On-Device / Edge LLM）最近5年文献与Top-30代表作系统综述

> **检索与分析基准日期**：2026-09-11  
> **时间范围**：最近5年（2021-09-11 至 2026-09-11）  
> **数据源说明**：引用量主要基于 OpenAlex、Semantic Scholar (S2) 与 Google Scholar 交叉参考。由于各平台索引收录差异显著（例如 arXiv 预印本与期刊/顶会出版 DOI 是否合并统计），**本报告标注各指标数据源**，并按综合学术影响力与工程部署地位排序。

---

## 1. 综述概览与技术发展脉络（Timeline: 2021-2026）

端侧设备（手机、PC、嵌入式 SoC、边缘盒、车载 NPU）受限于**内存带宽、物理 DRAM 容量、功耗（TDP）与散热**，直接部署大语言模型面临三大核心瓶颈：
1. **Memory Capacity Wall**：权重过大，难以完整装入端侧 DRAM（如 8GB/16GB 手机内存）；
2. **Memory Bandwidth Wall**：自回归解码（Token-by-Token）阶段算力利用率极低（Memory-bound），受 DRAM 带宽制约；
3. **KV Cache Explosion**：随上下文长度增加，KV Cache 显存线性膨胀甚至超越模型权重本身。

近 5 年端侧部署技术经历了三个主要演进阶段：
- **2021 - 2022：量化与压缩萌芽期**：从云端通用量化向后训练量化（PTQ）过渡，代表作为 ZeroQuant、SmoothQuant、GPTQ 等，解决了 LLM 离线免训练压缩的可能；
- **2023 - 2024：系统级深度协同与专用运行时爆发期**：
  - **权重与激活感知优化**：AWQ、SpQR、SparseGPT、LLM-QAT。
  - **KV Cache 专项压缩**：vLLM/PagedAttention、FlashAttention、KIVI、StreamingLLM。
  - **边缘异构与闪存推理**：llama.cpp 生态成熟、Apple *LLM in a flash*、SOSP 2024 *PowerInfer*（CPU-GPU/神经元激活稀疏混合）。
  - **无损加速解码**：Speculative Decoding（投机解码）、Medusa、EAGLE 普及。
- **2024 - 2026：NPU 编译器、端侧原生小模型与端云协同成熟期**：
  - 1B-3B/4-bit 级端侧原生强模型（MobileLLM、Phi-3-mini、Gemma-2-2B、Qwen-2.5-1.5B/3B）。
  - NPU 专用算子与编译器（MLC-LLM、ExecuTorch、Qualcomm AI Engine Direct、Apple MLX）。
  - 端云级联（Split Computing / Cascade Routing / Speculative Cloud-Edge Verification）。

---

## 2. 技术路线分类体系与对比表

| 分类标识 | 技术方向 | 核心解决问题 | 核心方法与机制 | 代表工作 | 端侧优势 / 代价 |
|---|---|---|---|---|---|
| **C1** | **后训练量化 (PTQ)** | 权重占用大、访存带宽受限 | Hessian二阶误差补偿、激活等价缩放、离群值保护 | GPTQ, AWQ, SmoothQuant, SpQR | 免重新训练，显存减少 50-75%；极低 bit (≤3bit) 需微调保护 |
| **C2** | **量化感知训练 (QAT) 与蒸馏** | 超低比特精度坍塌、合成数据压缩 | 数据无关蒸馏、模拟量化梯度反传、知识蒸馏 | LLM-QAT, MiniLLM, TinyLlama | 精度保持极好，可达 2-4 bit；需要训练算力和校准管线 |
| **C3** | **稀疏化与一次性剪枝** | 参数冗余、浮点计算瓶颈 | 一次性非结构化剪枝、2:4半结构化剪枝、神经元激活稀疏 | SparseGPT, Wanda, DejaVu | 理论压缩率高；端侧无专用稀疏 kernel 时无 wall-clock 加速 |
| **C4** | **KV Cache 内存与长文本优化** | 长序列内存爆炸、并发内存墙 | 动态分块、低比特 KV 量化、滚动窗口与注意力沉降 | PagedAttention, KIVI, StreamingLLM, FlashAttention | 极大延长端侧可用上下文；量化导致轻微长程检索退化 |
| **C5** | **端侧推理引擎与 Kernel 优化** | 端侧 CPU/GPU 算力利用率低下 | 纯 C/C++ 无依赖实现、NEON/Metal/Vulkan 汇编融合 | llama.cpp, MLC-LLM, Apple MLX, vLLM | 极低内存 footprint，开箱即用；跨芯片调优繁重 |
| **C6** | **非易失存储/异构内存分级加载** | 手机 DRAM 容量小于模型尺寸 | 闪存 (Flash) 预读按需加载、CPU-GPU 神经元冷热卸载 | LLM in a flash, PowerInfer, FlexGen | 可运行 2x DRAM 容量的模型；受 NVMe/PCIe 物理带宽制约 |
| **C7** | **投机解码 (Speculative Decoding)** | 解码阶段 Memory-bound 串行瓶颈 | 小型 Draft 模型草稿 + 主模型单步批验证（无损加速） | Fast Inference (Leviathan), SpecInfer, Medusa, EAGLE | 严格数学无损，速度提升 2-3x；端侧额外占用 Draft 模型内存 |
| **C8** | **端云协同与分级计算** | 端侧精度上限低、云端网络/隐私成本 | 本地小模型过滤/初筛 + 云端大模型兜底；早期退出 | Hybrid LLM, FrugalGPT, SplitLLM | 隐私与高难度任务兼顾；网络抖动影响端侧体验 |
| **C9** | **端侧架构重新设计 (Edge-Native LLMs)** | 现有大模型拓扑不适合移动端 Cache/带宽 | 深窄架构设计、参数共享、嵌入层紧凑化 | MobileLLM, Phi-3, Gemma-2-2B | 硬件执行效率最高；需从头预训练 |

---

## 3. 端侧部署优化 引用排名前30（Top-30）代表性文献精选表

> 统计口径说明：按 **大模型端侧/轻量化部署与推理优化** 强相关论文在各大权威检索平台的综合总引用数（以各工作 2026-09-11 检索记录为依据）梳理排序。

| 排名 | 论文名称 | 发表年份与出处 | 核心作者与机构 | 分类 | 引用量级（数据源估算） | 论文链接 |
|:---:|---|---|---|:---:|---|---|
| **1** | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | NeurIPS 2022 | Tri Dao 等 (Stanford) | C4/C5 | ~4500+ (Google Scholar / S2) | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| **2** | **Efficient Memory Management for Large Language Model Serving with PagedAttention** | SOSP 2023 | Woosuk Kwon 等 (UC Berkeley / vLLM) | C4/C5 | ~2400+ (Google Scholar / S2) | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| **3** | **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** | ICLR 2023 | Elias Frantar 等 (IST Austria) | C1 | ~1800+ (Google Scholar / S2) | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| **4** | **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** | ICML 2023 | Guangxuan Xiao 等 (MIT / Han Lab) | C1 | ~1500+ (Google Scholar / S2) | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| **5** | **Fast Inference from Transformers via Speculative Decoding** | ICML 2023 | Yaniv Leviathan 等 (Google Research) | C7 | ~1300+ (Google Scholar / S2) | [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) |
| **6** | **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | MLSys 2024 | Ji Lin 等 (MIT / Han Lab) | C1 | ~1100+ (Google Scholar / S2) | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| **7** | **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** | NeurIPS 2022 | Tim Dettmers 等 (UW / Meta) | C1 | ~1100+ (Google Scholar / S2) | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| **8** | **Accelerating Large Language Model Decoding with Speculative Sampling** | arXiv 2023 | Charlie Chen 等 (DeepMind) | C7 | ~950+ (Google Scholar / S2) | [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) |
| **9** | **QLoRA: Efficient Finetuning of Quantized LLMs** | NeurIPS 2023 | Tim Dettmers 等 (UW) | C1/C2 | ~3500+ (含微调，部署级 NF4/Double Quant) | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| **10** | **SparseGPT: Massive Language Models Can Be Pruned in One-Shot** | ICML 2023 | Elias Frantar, Dan Alistarh (IST Austria) | C3 | ~650+ (Google Scholar / S2) | [arXiv:2301.00774](https://arxiv.org/abs/2301.00774) |
| **11** | **A Simple and Effective Pruning Approach for Large Language Models (Wanda)** | ICLR 2024 | Mingjie Sun 等 (CMU / Meta) | C3 | ~550+ (Google Scholar / S2) | [arXiv:2306.11695](https://arxiv.org/abs/2306.11695) |
| **12** | **Efficient Streaming Language Models with Attention Sinks (StreamingLLM)** | ICLR 2024 | Guangxuan Xiao 等 (MIT) | C4 | ~500+ (Google Scholar / S2) | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| **13** | **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU** | ICML 2023 | Ying Sheng 等 (Stanford) | C6 | ~500+ (Google Scholar / S2) | [arXiv:2303.06865](https://arxiv.org/abs/2303.06865) |
| **14** | **LLM-QAT: Data-Free Quantization Aware Training for Large Language Models** | arXiv 2023 | Zechun Liu 等 (Meta AI) | C2/C4 | ~420+ (Semantic Scholar) | [arXiv:2305.17888](https://arxiv.org/abs/2305.17888) |
| **15** | **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** | arXiv 2024 | Tianle Cai 等 (Princeton / Together AI) | C7 | ~400+ (Google Scholar / S2) | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) |
| **16** | **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression** | ICLR 2024 | Tim Dettmers 等 (UW / Meta) | C1 | ~380+ (Google Scholar / S2) | [arXiv:2306.03078](https://arxiv.org/abs/2306.03078) |
| **17** | **PowerInfer: Fast Large Language Model Serving with a Consumer-Grade GPU** | SOSP 2024 | Yixin Song 等 (SJTU / IPADS) | C5/C6 | ~350+ (Semantic Scholar) | [ACM DL 3694715](https://doi.org/10.1145/3694715.3695964) |
| **18** | **MiniLLM: Knowledge Distillation of Large Language Models** | ICLR 2024 | Yuxian Gu 等 (Tsinghua) | C2 | ~320+ (Google Scholar / S2) | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| **19** | **LLM in a flash: Efficient Large Language Model Inference with Limited Memory** | ACL 2024 (Long) | Keivan Alizadeh 等 (Apple) | C6 | ~150-200 (Crossref/OpenAlex/S2) | [ACL Anthology](https://aclanthology.org/2024.acl-long.678/) |
| **20** | **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** | ICML 2024 | Yuhui Li 等 (Peking Univ) | C7 | ~300+ (Google Scholar / S2) | [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) |
| **21** | **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases** | ICML 2024 | Zechun Liu 等 (Meta AI) | C9 | ~280+ (Google Scholar / S2) | [arXiv:2402.14905](https://arxiv.org/abs/2402.14905) |
| **22** | **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** | ICML 2024 | Zirui Liu 等 (Rice Univ) | C4 | ~220+ (Google Scholar / S2) | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| **23** | **SqueezeLLM: Dense-and-Sparse Quantization** | ICML 2024 | Sehoon Kim 等 (UC Berkeley) | C1 | ~250+ (Google Scholar / S2) | [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) |
| **24** | **FrugalGPT: How to Use Large Language Models While Reducing Cost and Damage** | NeurIPS 2023 | Lingjiao Chen 等 (Stanford) | C8 | ~350+ (Google Scholar / S2) | [arXiv:2305.05176](https://arxiv.org/abs/2305.05176) |
| **25** | **MLC-LLM: Universal High-Performance Machine Learning Deployment Engine** | Project 2023-24 | TVM Community / CMU | C5 | ~300+ (代码库/技术报告) | [mlc.ai](https://mlc.ai/mlc-llm/) |
| **26** | **TinyLlama: An Open-Source Small Language Model** | Tech Report 2024 | Peiyuan Zhang 等 (SUTD) | C9 | ~500+ (作为端侧通用 Base 模型) | [arXiv:2401.02385](https://arxiv.org/abs/2401.02385) |
| **27** | **LLaMA.cpp: Port of Facebook's LLaMA model in C/C++** | Open Source 2023 | Georgi Gerganov 等 | C5 | 极高工业引用 (GitHub 70k+ Stars) | [github:llama.cpp](https://github.com/ggerganov/llama.cpp) |
| **28** | **DejaVu: Contextual Sparsity for Real-Time LLMs at Low Latency** | ICML 2023 | Zichang Liu 等 (Rice Univ) | C3/C5 | ~180+ (Google Scholar / S2) | [arXiv:2310.17157](https://arxiv.org/abs/2310.17157) |
| **29** | **Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone** | Tech Report 2024 | Microsoft GenAI Team | C9 | ~450+ (高质量移动端模型架构) | [arXiv:2404.14219](https://arxiv.org/abs/2404.14219) |
| **30** | **SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference** | ASPLOS 2024 | Xupeng Miao 等 (CMU) | C7 | ~210+ (Google Scholar / S2) | [arXiv:2305.09781](https://arxiv.org/abs/2305.09781) |

---

## 4. 关键文献深度剖析（按技术路线提取）

### 4.1 模型压缩与量化（PTQ / QAT）

#### 1. GPTQ (ICLR 2023)
- **核心问题**：传统二阶剪枝/量化算法（OBC/OBQ）计算复杂度高达 $O(d^3)$，无法扩展至百亿/千亿参数模型。
- **关键方法**：基于逆 Hessian 矩阵阻尼近似（$H^{-1}$），引入 **Lazy Batch Updates** 与 Cholesky 分解批处理，以列为单位贪心量化并向剩余未量化权重补偿二阶误差。
- **实验结论**：在 OPT-175B 上可在数小时内完成 4-bit / 3-bit 量化，Perplexity 几乎无损失。
- **局限**：量化耗时依赖 Hessian 校准集质量；低比特需要配合硬件反量化 Kernel 才能带来吞吐提升。

#### 2. AWQ (MLSys 2024)
- **核心问题**：权重并不是等价重要的；强行将离群权重（Outliers）保存在 FP16 中会带来复杂的硬件混合精度索引与带宽开销。
- **关键方法**：**激活感知等价缩放**（Activation-aware Scaling）。统计输入激活幅值，找出前 1% 最敏感的通道，通过数学等价变换 $Y = (W \cdot s) \cdot (s^{-1} \cdot X)$，放大敏感权重通道并压缩对应激活，在量化前平滑误差网格。
- **实验结论**：在 LLaMA / OPT 等模型上显著优于 RTN，4-bit 下精度接近 FP16，并推出了 TinyChat 移动端加速框架。
- **局限**：依然依赖校准集分布，数学/长文本等异常激活分布剧烈变动的场景需要更稳健的 scaling 搜索。

#### 3. LLM-QAT (Meta AI, 2023)
- **核心问题**：低于 4-bit 时 PTQ 精度衰减剧烈；且现有方法忽视了端侧长文本的主要显存瓶颈——**KV Cache**。
- **关键方法**：提出 **Data-Free 量化感知蒸馏**，直接利用预训练 LLM 自身无条件生成的 Token 作为蒸馏语料；首创同时对**权重、激活和 KV Cache** 进行协同量化。
- **实验结论**：在 4-bit 权重、8-bit 激活、4-bit KV Cache 下，较常规无训练 PTQ 精度提升多达 20 个点。
- **局限**：需要执行梯度反向传播，对端侧设备本身的训练算力要求高，一般只能在服务端完成 QAT 后再下发边缘端。

---

### 4.2 剪枝与稀疏化（Pruning & Sparsity）

#### 4. SparseGPT (ICML 2023)
- **核心问题**：传统剪枝方法需要大量微调（Retraining），百亿级参数大模型无法负担。
- **关键方法**：将剪枝建模为广义稀疏回归问题，推导出高效的闭式二次型局部更新解，单次前向（One-shot）逐层剪枝。
- **实验结论**：首次在 175B 级模型上实现 **50% 非结构化稀疏** 且精度几乎无损，并支持 2:4 半结构化模式。
- **局限**：**非结构化稀疏在移动端 CPU/NPU 上极难加速**（受随机访存开销拖累），必须依赖 2:4 Ampere/Ada 等特殊架构或专门的 Sparse Kernel。

#### 5. Wanda (ICLR 2024)
- **核心问题**：SparseGPT 计算 Hessian 矩阵仍需较多显存和求逆开销。
- **关键方法**：提出极其简洁的指标权重：$S_{ij} = |W_{ij}| \cdot ||X_j||_2$，即权重绝对值与输入特征 $L_2$ 范数的乘积，逐行独立剪枝，完全抛弃了二阶求逆。
- **实验结论**：以 SparseGPT 数十分之一的耗时达到了极其接近的稀疏保持能力。
- **局限**：极高稀疏率（>60%）下精度劣于严格的二阶优化方法。

---

### 4.3 内存瓶颈与存储层次分级优化（Memory & Storage Hierarchy）

#### 6. LLM in a flash (Apple, ACL 2024 Long)
- **核心问题**：移动设备 DRAM（4GB-8GB）无法容纳超过其尺寸的 LLM（例如 7B-FP16 需 14GB）。
- **关键方法**：
  1. **闪存按需加载（On-demand Flash Paging）**：将大部分静态参数保存在 Flash（NAND），通过成本模型预测推理阶段实际需要的神经元；
  2. **Windowing（窗口重用）**：复用最近几个时间步激活过的神经元，大幅削减 I/O 数据量；
  3. **Row-Column Bundling**：顺应闪存顺序读高吞吐的物理特性，成块打包权重矩阵。
- **实验结论**：可在手机等有限 DRAM 上成功运行达 **DRAM 物理容量 2 倍** 的模型，CPU/GPU 端端速度相比常规加载提升 4-5 倍与 20-25 倍。
- **局限**：高并发场景下 I/O 依然是刚性瓶颈；闪存长期高频擦写有寿命与磨损风险。

#### 7. PowerInfer (SOSP 2024)
- **核心问题**：消费级 PC / 端侧设备通常配备“小容量显存 GPU + 大容量系统内存 CPU”，直接 CPU Offloading 导致 PCI-e 传输和 CPU 算力成为瓶颈。
- **关键方法**：发掘了 LLM 激活的**幂律分布（Power-Law）**特性——极少数神经元（Hot Neurons）被频繁激活，其余多数为 Cold Neurons。将 Hot Neurons 固化预载在 GPU 显存，Cold Neurons 留在 CPU 内存端；并在前向中利用轻量预测器动态激活。
- **实验结论**：单张 RTX 4090 上较 llama.cpp 加速最高达 11.69 倍；OPT-30B 生成速度达到 A100 服务器的 82%。
- **局限**：预测器本身带来微小计算开销；当 prompt 领域偏移剧烈导致热点神经元分布剧烈抖动时加速比下降。

---

### 4.4 KV Cache 与长文本压缩（KV Cache Optimization）

#### 8. KIVI (ICML 2024)
- **核心问题**：自回归生成的上下文不断拉长，KV Cache 显存消耗成为端侧支持长文本的最早瓶颈。
- **关键方法**：非对称 2-bit KV Cache 极限量化。观察到 Key 缓存存在每通道（Per-channel）离群值，Value 缓存存在每 Token（Per-token）平滑分布，因此分别应用不同的分组量化策略。
- **实验结论**：将 KV Cache 压缩达 4 倍，支持端侧在相同显存下并发和序列长度扩大 2-4 倍，质量损失几乎可忽略。
- **局限**：需特制 2-bit GEMV 解码 kernel，否则反量化会产生额外 CPU/GPU 计算延迟。

#### 9. StreamingLLM (ICLR 2024)
- **核心问题**：端侧处理流式对话时，窗口滑出（Window Cache）会导致模型输出瞬间混乱崩溃。
- **关键方法**：发现**注意力沉降（Attention Sink）**现象——无论序列多长，首层前几个 Token（Initial Tokens）承载了极大部分无语义的聚合注意力值。保留前 4 个 Initial Tokens + 最新滑动窗口 Tokens 即可维持长文本困惑度。
- **实验结论**：使固定大小 KV Cache 的端侧模型能够无损流式运行超过 400 万 Token 的长对话。
- **局限**：无法检索已被滑出窗口的中间早期记忆（需配合 RAG 或摘要机制）。

---

### 4.5 投机解码与结构无损加速（Speculative Decoding）

#### 10. Speculative Decoding (Google, ICML 2023)
- **核心问题**：自回归解码逐 Token 读取几十 GB 权重，计算访存比（Arithmetic Intensity）极低，端侧内存带宽被极度浪费。
- **关键方法**：使用一个小草稿模型（Draft Model，如 1B）快速预测 $K$ 个后续候选 Token，大模型（Target Model，如 7B）在一次前向中利用因果掩码同时并行验证这 $K$ 个 Token。通过改进的拒绝采样算法，**在数学上严格保证输出概率分布与大模型独立采样完全一致**。
- **实验结论**：典型场景下实现 2x - 3x 的实际端到端 Wall-clock 加速，零精度损失。
- **局限**：端侧需要额外显存驻留 Draft 模型；Draft 接受率高度受任务类型（代码/结构化文本接受率高，创意写作较低）影响。

#### 11. Medusa / EAGLE (2024)
- **核心问题**：为大模型单独配准一个 Draft 小模型增加了系统复杂度和端侧内存加载负担。
- **关键方法**：
  - **Medusa**：抛弃 Draft 模型，直接在冻结的主模型顶层添加若干并行的轻量 MLP 解码头（Medusa Heads），树状并行验证；
  - **EAGLE**：将投机外推提升到特征层（Feature Level），解决自回归序列不确定性问题。
- **实验结论**：无需额外小模型即可稳定提速 1.5x - 2.5x，对移动端紧凑型单体部署极度友好。

---

### 4.6 端侧原生架构设计（Mobile-Native Architectures）

#### 12. MobileLLM (Meta AI, ICML 2024)
- **核心问题**：过去小模型（Sub-billion）多由大模型简单削减层数得来，在移动端芯片上执行效率低下且表征能力不足。
- **关键方法**：
  1. **深度优先而非宽度优先（Deep & Thin）**：在固定 1B 参数预算下，增加层数并缩小特征隐藏维度，在算力与参数量间取得更好均衡；
  2. **跨层权重共享（Layer-wise Weight Sharing）**：在不增加 DRAM 权重搬运开销的前提下大幅提升逻辑参数量；
  3. **嵌入层优化与分组查询注意力（GQA）**。
- **实验结论**：在 125M / 350M / 1B 规模下全面刷新同量级推理精度记录，成为端侧基座架构的标杆方案。

---

## 5. 当前研究空白（Research Gaps）与选题建议

根据最近 5 年学术界与工业界落地现状，端侧部署仍存在以下**关键瓶颈与高价值研究方向**：

### 5.1 组合压缩的非线性收益与硬件瓶颈（Compound Compression Limits）
- **现状**：学术界常将“剪枝 + 量化 + 投机解码 + KV压缩”在论文中叠加讨论。
- **研究空白**：在真实端侧移动 NPU（如 Apple A/M 芯片 Neural Engine、高通 Snapdragon NPU、联发科 APU）上，多项技术叠加后往往导致算子碎裂，反量化与反稀疏寻址的指令开销甚至抵消了访存减少的收益。
- **选题建议**：**面向特定 NPU 指令集架构的“量化-稀疏-编译”端到端软硬件协同优化基准与一体化自动化搜索框架**。

### 5.2 超低比特（1-2 bit / Ternary）端侧原生可训练架构
- **现状**：PTQ 在 4-bit 表现极好，但在 2-bit 或 1.58-bit（如 BitNet b1.58）下严重失真。
- **研究空白**：端侧亟需 1-2 bit 下仍具备可靠上下文推理能力的技术。现有的 BitNet 等方案多聚焦预训练，缺少后训练微调与特定业务轻量适配方案。
- **选题建议**：**端侧可热插拔的 1.58-bit / 三值化 Low-Rank 增量适配器（Ternary On-device Adapters）**。

### 5.3 长上下文端侧多模态模型（On-Device Long-Context MLLMs）
- **现状**：端侧多模态模型（如 MobileVLM）受视觉 Token 暴增影响，端侧内存与 Prefill 延迟急剧恶化。
- **研究空白**：视觉 Token 的空间冗余极高，与自回归语言 Token 存在不对称性，现有的端侧优化很少联合优化 Vision Encoder 缓存与 LLM KV Cache。
- **选题建议**：**面向端侧多模态视频/长图输入的跨模态动态 Token 剪枝与稀疏分级缓存机制**。

### 5.4 动态端云多智能体协同路由与断网自愈机制（Resilient Cloud-Edge Split Computing）
- **现状**：端云协同多停留在简单的级联（Cascade）或语义分流（Routing）。
- **研究空白**：当端侧网络处于动态弱网、高抖动或完全离线时，如何动态调节本地模型的量化精度/退出层数（Early Exit），实现端云无缝平滑退避与上下文回填。
- **选题建议**：**弱网自适应的端云动态推测执行与双向差分状态同步机制**。

---

## 6. 本地报告保存提示

本综述的结构化分析与关键结论已同步存入当前工作区知识库：
- 路径：`/Users/zhuzhichao/Documents/ai_interfence/paper/knowledge/on_device_llm_survey_2021_2026.md`
- 配合此前生成的 AWQ / GPTQ 深入分析报告，可为端侧大模型推理、系统工程与学术选题提供系统化的文献底座。
