# 端侧大模型量化部署：研究热点与硬件平台现状分析报告

> **报告日期**：2026-09-13
> **数据来源**：deep-research 多智能体检索（4 条高置信确认 claim + 多条未经三票核验的来源内容），结合 OpenAlex/Crossref/arXiv/官方厂商资料。检索期间部分验证请求因 API 限流（429）失败，凡未通过对抗验证的条目均标注 `[未核验]`，引用时请以官方原始页面为准。
> **关联文档**：`knowledge/on_device_llm_survey_2021_2026.md`（近5年文献与Top-30综述）、`knowledge/awq_gptq_analysis.md`（GPTQ/AWQ深度对比）

---

## 一、研究热点领域（未来 2-3 年的 8 大主攻方向）

### 热点 1：超低比特量化（1-bit / 1.58-bit / 2-bit）——从 PTQ 走向原生态训练

- **核心事实**：`BitNet b1.58 2B4T` 是**首个开源的原生 1-bit（1.58-bit 三值权重）LLM**，参数规模 2B（arXiv:2504.12285，该"首个"表述经 1-1 票确认，属前沿方向代表性工作）。
- **趋势判断**：超低比特的重心正从"训练后量化（PTQ）"转向"量化感知的原生预训练（native QAT-from-scratch）"。RTN/AWQ/GPTQ 在 ≤2-bit 时精度断崖几乎不可避免，三值权重 {−1,0,+1} 的原生训练成为主战场。
- **端侧意义**：1.58-bit 将 7B 模型的权重占用从 ~14GB（FP16）压到 ~1.75GB，首次让 3B-7B 级模型在 8GB 手机上全量驻留 DRAM 成为可能。
- **未解问题** [未核验]：三值模型在内存/能耗/解码延迟上的优势数据均来自微软自报；其训练成本（需要完整预训练）使开源社区难以复刻。

### 热点 2：Mixture-of-Quantization（混合比特量化）——从论文走向工程

- **学术前沿**：MoQAE（ACL 2025）把 MoE 范式引入量化本身——把 FP16/INT4/INT2 当作"专家"，用 MLP router 按 token 块选择比特宽度。KV Cache 混合精度方向有 PM-KVQ（arXiv:2505.18610，渐进式按块降比特，长思维链场景宣称吞吐提升 2.73-5.18x）[均未核验，来自摘要]。
- **工业落地（已证实）**：Apple 官方 `mlx-lm` 已内置 `mixed_2_6 / mixed_3_4 / mixed_3_6 / mixed_4_6` 混合比特配方（`--quant-predicate`），设计明确对标 llama.cpp 的 Q4_K_M；llama.cpp 的 GGUF Q2_K/IQ2/TQ1_0 系列本质也是逐张量混合比特。**混合量化已不是"是否"而是"默认"**。
- **研究空白**：学术 router 是动态 per-token 的（硬件不友好），工业实现是静态 per-layer 的（精度次优）；"能在 NPU 上零开销执行的动态混合比特"仍是空白。

### 热点 3：KV Cache 量化与长上下文（端侧长文本的真正瓶颈）

- **已证实**：mlx-lm 实现了可配置比特宽的量化 KV Cache（QuantizedKVCache，`--kv-bits`/`--kv-group-size`/`--quantized-kv-start`，默认序列达 5000 token 后才开始量化，to_quantized 默认 bits=4）。
- **热点细分**：
  - 非对称量化（K 轴 per-channel、V 轴 per-token，KIVI 路线）；
  - 长思维链（long-CoT）下的累积量化误差（PM-KVQ 动机）；
  - 与 Attention Sink / 滑动窗口（StreamingLLM）联合的混合保留策略。
- **矛盾点**：MoQAE 在 H20-NVLink（96GB 数据中心 GPU）上评测、softmax 前需把 Key 反量化回 FP16——**说明 KV 量化学术前沿仍瞄准服务器，端侧 NPU 上的 KV 量化 kernel 基本空白** [未核验]。

### 热点 4：量化 + LoRA / PEFT 的协同（端侧个性化微调）

- **已证实**（ScienceDirect 低比特 LLM 综述，3-0 票）：量化与 LoRA 的朴素组合效果次优，催生了 **LoftQ、LQ-LoRA** 等量化感知初始化方法——把量化误差分解为 LoRA 可捕获的低秩分量。
- **工业侧信号** [未核验，厂商官方页]：联发科 Dimensity 9400 宣称"业界首个端侧 LoRA 训练"（on-device LoRA training）；mlx-lm 支持在量化模型上直接做 QLoRA/LoRA/DoRA/全参微调（`--model` 指向量化模型时自动切换 QLoRA）。
- **研究空白**：端侧训练的内存（optimizer state）、热（持续计算）、隐私（本地数据回传）三角约束下的"亿级参数级个性化适配器即时训练"尚无成熟方案。

### 热点 5：W4A8 / W4A4 激活量化（从 weight-only 走向整图整型）

- **机制辨析** [未核验，机理表述]：weight-only 量化（GPTQ/AWQ）的加速主要来自降低内存带宽；只有权重+激活同时量化（W4A8/W4A4）才能用低比特矩阵乘内核带来**计算加速**——这决定了不同硬件（带宽受限 vs 算力受限）的收益差异。
- **趋势**：NPU 天然偏爱整型输入，端侧部署正在从"4-bit 权重 + FP16 激活"过渡到"全整型图"（激活 int8 + per-tensor/per-block scale）。SmoothQuant 的等价迁移思想是该方向的基石。
- **空白**：离群值（outliers）使 W4A4 精度难以守住，激活侧的在线量化（dynamic quant）与离群值通道保留是当前攻坚点。

### 热点 6：稀疏-量化联合压缩

- SparseGPT 论文已证明剪枝与权重量化可叠加（2:4 半结构化 + 量化）；但端侧移动 NPU 普遍**不提供非结构化稀疏加速指令**，稀疏的 wall-clock 收益在端侧经常为负。
- **方向**：块稀疏（block sparsity）+ 低比特的联合表示、与编译器的算子融合协同。

### 热点 7：多模态端侧量化（On-Device MLLM）

- 视觉 Token 数量庞大且冗余极高，端侧 VLM 的 Prefill 延迟与 KV 膨胀远比纯文本模型严重。
- 联发科官方宣称 Dimensity 9400 支持 MLLM 生成 >50 tokens/s、Diffusion Transformer 端侧运行 [未核验]。
- **空白**：视觉 Token 剪枝 + 非对称跨模态 KV 压缩 + Vision Encoder 量化的联合优化基本空白。

### 热点 8：编译器与硬件感知量化（HALQ / 编译协同）

- 量化布局（group size、scale 存储位置、对称/非对称）必须匹配目标硬件的指令集（int4 dot product、MXFP4 block format），否则反量化开销吃掉收益。
- 生态现状（已证实）：llama.cpp 已提供专用 **Hexagon backend**（高通 NPU）与 **OpenCL backend**（Adreno GPU）——开源推理栈正在直接吃进移动 NPU/DSP。
- **空白**：面向量化感知的端到端自动搜索（量化参数 × 算子调度 × 内存布局的联合 NAS/autoscheduling）在端侧 NPU 上还没有类似 TVM/Ansor 级别的成熟系统。

---

## 二、端侧各硬件平台发展现状

> 算力与带宽数字均为厂商官方宣传口径，实际可用算力受功耗墙与算子支持限制；`[未核验]` 表示该数字来自厂商页面，未能三票核验。

### 2.1 Apple（A 系列 / M 系列 + Neural Engine + MLX）

| 项目 | 现状 |
|---|---|
| 算力 | M4 Neural Engine 16 核，最高 **38 TOPS**（Apple 宣称"快于任何 AI PC 的 NPU"）[未核验，官方新闻稿] |
| 架构特点 | 统一内存（Unified Memory）架构，CPU/GPU/NPU 共享高带宽内存池；CPU 性能核与能效核均内置 ML 加速器 [未核验] |
| 量化格式 | MLX/`mlx-lm` 生态：affine（int 系）、**mxfp4、nvfp4、mxfp8** 四种量化模式；混合比特配方（mixed_2_6/3_4/3_6/4_6）；量化 KV Cache；量化模型上直接 QLoRA [未核验，官方 GitHub] |
| 生态 | MLX（研究友好）+ Core ML / Core ML LLM（系统级）+ llama.cpp Metal backend；Apple 强调推理完全在设备端、隐私保护、对内存/响应性/续航影响小 [未核验] |
| 短板 | Neural Engine 对第三方推理框架基本封闭（Core ML 之外难以直达 ANE）；MLX 仅覆盖 Apple Silicon |

### 2.2 Qualcomm（Snapdragon 8 系 + Hexagon NPU）

| 项目 | 现状 |
|---|---|
| 算力 | 8 Gen 3/8 Elite Hexagon NPU 官宣 45-80 TOPS 区间（INT8 口径，厂商宣传）[未核验] |
| 量化支持 | QNN（Qualcomm AI Engine Direct）支持 int4/int8/fp16/w4a16 等；文档强调 LLM 权重 int4 布局 |
| 生态 | AI Engine SDK / QNN；**llama.cpp 官方 Hexagon backend 已落地**（开源栈直连 NPU）[未核验] |
| 定位 | 端侧 GenAI 生态最激进的 Android 阵营玩家；Hexagon 融合 DSP+NPU，含专用低比特矩阵加速单元 |

### 2.3 MediaTek（Dimensity 9400 + APU 890）

| 项目 | 现状（均来自官方页，[未核验]） |
|---|---|
| 官方亮点 | 第 8 代 NPU（NPU 890）：**业界首个端侧 LoRA 训练**；支持 MoE LLM、Diffusion Transformer、投机解码、SLM+LLM 组合、多模态 |
| 性能宣传 | LLM prefill +80%、扩散生成 +100%（vs 上代旗舰）、MLLM >50 tokens/s、能效 +35% |
| 生态 | NeuroPilot SDK；对 MoE 与投机解码的官方支持是差异化卖点 |

### 2.4 ARM CPU（Cortex + NEON / DotProd / I8MM）

- **定位**：所有端侧设备的兜底算力；llama.cpp 的 ARM 后端（NEON、i8mm dot product）是 CPU 推理的主力路径。
- **现状**：int8 dot product（SDOT/UDOT）与 int8mm（I8MM）已在 Cortex-A7x 普及；int4 仍主要靠 unpack-to-int8 模拟，真正 int4 原生指令只在部分微架构出现。
- **意义**：GGUF 的 K-quant / i-quant 位打包格式就是为 CPU SIMT 吞吐设计的——**llama.cpp 1.5~8-bit 全谱系量化是 CPU 路线的直接产物** [未核验]。

### 2.5 x86 PC 侧 NPU（AI PC）

| 平台 | 现状 |
|---|---|
| Intel Lunar Lake / Arrow Lake | NPU 算力 ~40-48 TOPS（官方宣传）[未核验]；OpenVINO 支持低比特 LLM |
| AMD XDNA (Ryzen AI) | XDNA2 NPU ~50 TOPS 级（官方宣传）[未核验]；Ryzen AI Software（ONNX Runtime + Vitis AI） |
| 共性 | AI PC NPU 算力已接近手机旗舰，但 LLM 生态成熟度落后于 Apple/高通；内存带宽（LPDDR5X）成为共同瓶颈 |

### 2.6 移动 GPU（Adreno / Mali / Apple GPU）

- 定位：NPU 不支持算子时的回退算力；llama.cpp 的 OpenCL backend（Adreno）与 Vulkan 后端覆盖 Mali/Adreno [未核验]。
- 弱点：移动 GPU 缺少 int4/int8 矩阵乘专用单元（与服务器 GPU 的 Tensor Core 不同），低比特收益主要还是访存减少。

### 2.7 Samsung Exynos NPU

- Exynos 2400/2500 内置 NPU，官方口径与联发科/高通同代竞争 [未核验]；开源生态支持度（llama.cpp 无专用 backend）明显弱于前三者，端侧 GenAI 存在感较低。

### 2.8 跨平台横向对比表

| 平台 | 典型 NPU 算力（宣传口径） | 内存带宽 | 量化类型支持 | 官方/主流推理栈 |
|---|---|---|---|---|
| Apple M4/A17-A18 | 35-38 TOPS | ~100-273 GB/s（统一内存，M 系 Pro/Max 更高） | int4/int8, **mxfp4/nvfp4/mxfp8**（MLX） | MLX, Core ML, llama.cpp(Metal) |
| Snapdragon 8 Elite | 45-80 TOPS [未核验] | LPDDR5X ~77-107 GB/s | int4/int8/w4a16 | QNN/AI Engine SDK, llama.cpp(Hexagon) |
| Dimensity 9400 | 官方未公布绝对 TOPS | LPDDR5X | int4/int8 + 官方 MoE/投机解码支持 | NeuroPilot |
| AI PC NPU（Intel/AMD） | 40-50 TOPS [未核验] | DDR5/LPDDR5X | int4/int8/fp8（视平台） | OpenVINO, Ryzen AI, ONNX Runtime |
| ARM CPU（无 NPU） | —（SIMD 算力） | 受 LPDDR 带宽限制 | 1.5~8-bit GGUF 全谱系 | llama.cpp(NEON/I8MM) |
| 移动 GPU | — | 同 SoC | 主要 int8/FP16 | llama.cpp(OpenCL/Vulkan) |

---

## 三、学术界热点与工业界落地的差距分析

### 差距 1：评测硬件错位——论文在 A100/H100 上，部署在手机上

MoQAE（ACL 2025，KV 混合量化代表作）在 96GB 的 H20-NVLink 上评测 [未核验]。量化领域的学术 SOTA 大量依赖服务器 GPU 验证，而端侧 NPU 的指令约束（支持的 group size、scale 布局、激活量化方式）在论文里几乎从不出现。**"论文 SOTA 无法编译到 NPU"是常态而非例外。**

### 差距 2：量化格式分裂——学术 INT 仿射 vs 工业 MX/浮点块格式

- 学术界默认 int4/int8 仿射量化（AWQ/GPTQ 语义）；
- 工业界快速转向块浮点格式：Apple MLX 已支持 mxfp4/nvfp4/mxfp8，NVIDIA 主推 NVFP4，AMD/高通阵营推 MXFP8/MXINT8。
- **后果**：论文新方法大多没有 MX 格式版本，落地时需重新设计 scale 语义，精度结论不能直接迁移。

### 差距 3：动态策略的硬件开销被忽略

- 学术热点（per-token 动态混合比特、动态路由、动态投机解码深度）全部引入运行时 router/控制流；
- 端侧 NPU 是静态图执行模型，动态分支意味着 CPU 回退与算子重编译。MoQAE 自己也承认 router 与反量化带来额外内存与延迟 [未核验]。
- **工业解法是"编译期静态化"**（如 mlx-lm 的 mixed_* 静态配方），与学术的"运行时动态化"方向相反。

### 差距 4：端侧训练（On-Device Learning）能力缺口

- 工业（联发科端侧 LoRA 训练）已把卖点做进芯片宣传，但学术上端侧训练的内存优化（梯度 checkpointing、optimizer state 量化）、热约束下的训练调度、跨设备联邦化个性化仍是早期阶段。

### 差距 5：加速比口径不一致，缺乏端侧统一基准

- 厂商数字（prefill +80%、>50 tokens/s、38 TOPS）各自选择对手与负载；学术数字（2-3x speedup）多为服务器 batch 场景。
- 端侧缺少类似 MLPerf-Mobile 但**针对 LLM 量化推理**（prefill/decode/首 token 延迟/长上下文/多模态分开报告）的权威基准。能效（tokens/Joule）报告更是普遍缺失。

### 差距总结表

| 维度 | 学术界热点 | 工业界现实 | 差距性质 |
|---|---|---|---|
| 评测平台 | 数据中心 GPU | 手机 NPU / AI PC NPU | 生态错位 |
| 量化格式 | INT 仿射（int4/int8） | MXFP4/NVFP4 块浮点 | 格式分裂 |
| 比特选择策略 | 运行时动态（MoE-router） | 编译期静态（mixed_3_6 等） | 执行模型冲突 |
| 训练侧 | 云端 QAT/QLoRA | 芯片级端侧 LoRA 训练 | 角色反转 |
| 加速指标 | tokens/s（batch 吞吐） | 首 token 延迟、能效、内存峰值 | 指标口径不一 |
| 稀疏化 | 50% 非结构化稀疏 | 端侧无稀疏指令，基本不用 | 可行性鸿沟 |

---

## 四、结论与选题优先级建议

1. **最高优先级**：面向 MX 块浮点格式（mxfp4/nvfp4）与 NPU 指令约束的**硬件感知量化方法学**（HALQ）——把 group size/scale 布局/激活量化点纳入量化目标函数，使论文方法可直接编译到 MLX/QNN。
2. **高优先级**：端侧 KV Cache 量化 + 视觉 Token 剪枝的联合优化（多模态长上下文），当前两条线各自为战。
3. **高优先级**：量化模型上的端侧 LoRA 训练系统（内存-热-隐私三约束），呼应芯片厂商已给出的硬件能力。
4. **中优先级**：W4A4 全整型图的离群值治理（激活侧动态量化 + 通道保留）。
5. **中优先级**：建立**端侧 LLM 量化统一基准**（能效 + 首 token 延迟 + 长上下文多维报告），补齐学术-工业评测口径鸿沟。
6. **谨慎跟进**：1-bit 原生训练路线（BitNet）——学术上限高但需要完整预训练资源，端侧团队更适合跟进其开源 2B 权重的后训练适配研究。

---

## 五、主要来源

- [BitNet b1.58 2B4T (arXiv:2504.12285)](https://arxiv.org/abs/2504.12285)
- [低比特 LLM 综述（ScienceDirect, 2025）](https://www.sciencedirect.com/science/article/pii/S0893608025007361)
- [MoQAE (ACL 2025)](https://aclanthology.org/2025.acl-long.531/)
- [PM-KVQ (arXiv:2505.18610)](https://arxiv.org/abs/2505.18610)
- [Apple M4 官方新闻稿](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)
- [MediaTek Dimensity 9400 官方页](https://www.mediatek.com/products/smartphones-2/mediatek-dimensity-9400)
- [mlx-lm (GitHub)](https://github.com/ml-explore/mlx-lm)
- [llama.cpp (GitHub)](https://github.com/ggml-org/llama.cpp)
- [Awesome-Model-Quantization](https://github.com/AI-Efficiency/Awesome-Model-Quantization)
