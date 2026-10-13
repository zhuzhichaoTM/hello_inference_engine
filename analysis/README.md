# 剖析篇 · 端侧推理引擎、推理框架与源码级特征分析

> **定位**：从原理到源码的深度剖析，聚焦**端侧**推理优化实现，基于代码、可视化图解与运行报告讲清端侧大模型推理的每一步。

| [顶层](../README.md) | [文档索引](../docs/index.md) | [路线图](../docs/roadmap.md) | [写作规范](../docs/writing-style.md) |

## 如何使用本导航

本目录围绕[端侧大模型推理业务地图](端侧大模型推理业务地图.md)组织资料。业务地图**三图一体**，本目录主要承载其中「第一部分 · 基础知识地图」的 L0–L8 分层，并在各层 README 中标注与「业务流程地图」「工程闭环地图」的对应关系：

| 部分 | 视角 | 回答的问题 | 本目录对应 |
| ---- | ---- | ---------- | ---------- |
| 第一部分 基础知识地图 | 知识维度（纵向分层） | 我要掌握什么 | L0–L8 分层目录 |
| 第二部分 业务流程地图 | 流程维度（横向流动） | 事情怎么跑起来 | 各层 README「在业务地图中的位置」 |
| 第三部分 工程闭环地图 | 闭环维度（迭代反馈） | 团队如何持续交付 | 各层实践产出与验收标准 |

> **边界**：只覆盖**端侧**（手机 / PC / 平板 / 嵌入式 / IoT / 车载 / 机器人等单设备或近设备场景）。数据中心的多卡多机并行、集群调度、K8s 平台、多租户云服务、按 token 计费不在范围内；端云协同仅保留「卸载边界判定」这一个决策点。

```text
L0 数学、深度学习与 Transformer 基础（端侧视角）
  → L1 端侧模型资产、权重、Tokenizer 与模型格式
  → L2 端侧 Tensor / 算子 / 计算图 / 精度与量化
  → L3 端侧推理运行时 / 内存 / KV Cache / 生成
  → L4 端侧 CPU / GPU / NPU / DSP 与 Kernel 优化
  → L5 端侧推理引擎与框架
  → L6 端侧部署形态与工程化
  → L7 端侧多模态 / Agent / 业务编排
  → L8 评测 / 可观测性 / 功耗 / 安全 / 成本
```

完整的学习目标、能力依赖、实践产出和验收标准请先阅读[端侧大模型推理业务地图](端侧大模型推理业务地图.md)；业务流程（01–13）与工程闭环（0–6）见该文档第二、三部分。每个层级目录都是对应业务特性的资料承载入口；已有量化资料暂不移动，通过下表按职责映射。

## 内容

| 层级 | 主题 | 入口 | 内容 |
| ---- | ---- | ---- | ---- |
| L0 | 数学、深度学习与 Transformer 基础（端侧视角） | [L0/](L0/) | 线性代数、概率、数值精度、Attention、RoPE、FFN、Prefill/Decode 与端侧内存/带宽估算 |
| L1 | 端侧模型资产、权重、Tokenizer 与模型格式 | [L1/](L1/) · [GGUF 格式解析](L1/gguf/) | 模型配置、权重加载、Tokenizer、Chat Template、GGUF/ONNX/TFLite/Core ML 等格式 |
| L2 | 端侧 Tensor、算子、计算图、精度与量化 | [L2/](L2/) · [ggml 框架与 CUDA 后端](L2/framework/ggml/) · [ggml 量化源码](L2/quant/ggml/ggml-quants源码详细分析.md) · [Q4_K_M 分析](L2/quant/ggml/ggml-quants_q4_k_m_analysis.md) · [可视化调试](L2/quant/ggml/q4km_debug/) | Tensor 布局、核心算子、图优化、低比特量化、量化误差与一致性验证 |
| L3 | 端侧推理运行时、内存、KV Cache 与生成 | [L3/](L3/) · [量化执行流程](L2/quant/llama_cpp/llama-quantize-impl-flow.md) · [工具分析](L2/quant/llama_cpp/llama-quantize-tool-analysis.md) | 模型加载、图执行、内存规划、KV Cache、Prefill/Decode、采样与流式生成 |
| L4 | 端侧 CPU、GPU、NPU、DSP 与 Kernel 优化 | [L4/](L4/) · [CUDA 后端剖析](L2/framework/ggml/backend_cuda/) · [端侧量化系统设计](L2/quant/端侧大模型量化系统设计文档.md) | ARM/x86/RISC-V SIMD、移动 GPU、NPU/DSP、Kernel、带宽、异构调度与 Profile |
| L5 | 端侧推理引擎与框架 | [L5/](L5/) | llama.cpp/ggml、MNN、ncnn、TFLite、Core ML/MLX、OpenVINO、TensorRT、QNN、RKNN、ExecuTorch 等引擎的选型、后端覆盖与调优 |
| L6 | 端侧部署形态与工程化 | [L6/](L6/) | App/固件集成、模型打包与分发、OTA、冷启动、包体积、多设备适配与端云边界判定 |
| L7 | 端侧多模态、Agent 与业务编排 | [L7/](L7/) | 端侧多模态、本地 RAG、端侧 Agent、工具调用与业务工作流 |
| L8 | 评测、可观测性、功耗、安全与成本 | [benches/](../benches/) · [L8/](L8/) | 质量与性能评测、功耗与能效、日志追踪、故障、安全隐私、审计与成本闭环 |

## 推荐学习路线

1. **读懂并运行模型：L0 → L1**。先理解 Token、Transformer、模型配置、Tokenizer、权重和 Greedy Sampling，并建立端侧内存/带宽估算。
2. **实现推理核心：L2 → L3**。从 Tensor、MatMul、Norm、RoPE、Attention 入手，逐步实现计算图、内存规划、KV Cache 和流式生成。当前仓库可优先阅读 [Q4_K_M 可视化调试](L2/quant/ggml/q4km_debug/)、[ggml 量化源码分析](L2/quant/ggml/ggml-quants源码详细分析.md) 和 [ggml CUDA 后端剖析](L2/framework/ggml/backend_cuda/)。
3. **进行硬件与量化优化：L4**。比较 CPU/GPU/NPU 的 Kernel、布局、精度、带宽和功耗，用 Profile 验证收益。
4. **选定引擎并调优：L5**。基于设备画像与业务约束完成引擎选型、后端适配与自定义算子扩展。
5. **落地部署：L6**。把引擎集成进 App/固件，处理分发、OTA、冷启动、体积与多设备适配。
6. **落地业务并形成闭环：L7 → L8**。构建端侧多模态、Agent 或本地 RAG 应用；质量、性能、功耗、安全与成本专题统一进入 [benches/](../benches/)，并用端侧指标验收。

## 资料归档约定

- 新资料按**主要负责的业务层级**放入对应 `Lx/` 目录；跨层系统设计选择一个主层归档，并在相关层级 README 中互链。
- 每层优先沉淀四类材料：**原理说明、源码分析、可复现实践、验证/评测报告**。
- 目录 README 负责说明边界和学习入口，并标注该层在业务流程（01–13）与工程闭环（0–6）中的位置；具体文档负责记录问题、实现、数据和验收结论。
- 量化资料同时涉及 L2（算子与精度）、L3（运行时与 KV Cache）和 L4（硬件优化）时，以主题主责层为入口，避免复制同一份文档。

## 状态

✅ 已建立 L0–L8 学习导航和资料承载目录，对齐[端侧大模型推理业务地图](端侧大模型推理业务地图.md)；现有量化、ggml/CUDA 与 llama.cpp 分析资料持续补充中。新增框架或业务主题请按归档约定提 [内容提议](../.github/ISSUE_TEMPLATE/content_proposal.yml)。
