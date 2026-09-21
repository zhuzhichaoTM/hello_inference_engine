# Hello Inference Engine

> **端侧与边缘场景的大模型推理工程化实践库：理论 → 量化 → 硬件 → 部署 → 评测，全链路可复现。**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs Lint](https://github.com/zhuzhichaoTM/hello_inference_engine/actions/workflows/docs-lint.yml/badge.svg)](https://github.com/zhuzhichaoTM/hello_inference_engine/actions/workflows/docs-lint.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zhuzhichaoTM/hello_inference_engine)
[![GitHub stars](https://img.shields.io/github/stars/zhuzhichaoTM/hello_inference_engine?style=social)](https://github.com/zhuzhichaoTM/hello_inference_engine/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/zhuzhichaoTM/hello_inference_engine)](https://github.com/zhuzhichaoTM/hello_inference_engine/commits/main)

| **[文档索引](docs/index.md)** | **[路线图](docs/roadmap.md)** | **[部署实战](deploy/)** | **[量化工具链](deploy/gguf_quant/)** | **[硬件洞察](hw/)** | **[论文综述](paper/)** | **[贡献指南](CONTRIBUTING.md)** | **[安全策略](SECURITY.md)** |

## 何时用本仓库

- 你要在**单卡 / 端侧 / 边缘**环境部署 LLM，需要量化选型 + 框架选型 + 实测数据
- 你想搞懂 **AWQ / GPTQ / GGUF（Q4_K_M 等）**量化原理，最好有源码级剖析和可视化演示
- 你要对比 **NVIDIA GPU / 华为 NPU / 端侧芯片**，需要架构分析与产业洞察
- 你要用 **llama.cpp / vLLM / TensorRT-LLM** 做部署，需要可一键复现的脚本与报告

如果只是调 OpenAI 兼容 API 写应用、不关心推理内部实现，本仓库对你太重——直接看各模型官方文档即可。

**内容成熟度一览：**

| 篇章 | 内容 | 实测 | 代码 | 状态 |
|------|:----:|:----:|:----:|:----:|
| [basics](basics/) 基础篇 | 6 章系统文档 | — | — | ✅ 完整 |
| [paper](paper/) 学术篇 | 3 篇综述 | — | — | ✅ 完整 |
| [analysis](analysis/) 剖析篇 | ggml / llama.cpp 源码级分析 + Q4_K_M 动效 | ✅ | ✅ | ✅ 亮点 |
| [hw](hw/) 硬件篇 | GPU / NPU / 产业洞察 5 篇 | — | — | ✅ 完整 |
| [deploy](deploy/) 部署篇 | Qwen3.8-27B 实战 + GGUF 量化框架 | ✅ | ✅ | ✅ 核心 |
| [benches](benches/) 性能篇 | 压测方法与评测报告 | 🚧 | 🚧 | 🚧 建设中 |
| [hotspot](hotspot/) 热点篇 | 行业动态追踪 | — | — | 🚧 建设中 |
| [extention](extention/) 扩展篇 | 推理框架二次开发 | 🚧 | 🚧 | 🚧 建设中 |

> 实测速览（环境：单卡 RTX 4090 24G / llama.cpp CUDA / Qwen3.8-27B UD-Q4_K_M ~15.5G / 64K 上下文，详见部署报告）：SSH 隧道约 **86 tok/s**，公网转发约 **80 tok/s**。数字随硬件与版本变化，请以报告内复现命令为准。

## 本仓库做什么

大多数推理教程只讲单点（调参或跑通 demo）。本仓库是**面向落地的全栈地图**——从理论地基、量化压缩、硬件选型，到多框架部署实战与压测评测，每一步都要求可复现、可引用。

```
hello_inference_engine/
├── basics/      # 基础篇 —— 推理系统设计与核心理论（6 章 + 57 图）
├── paper/       # 学术篇 —— AWQ/GPTQ、端侧 LLM 综述与研究热点
├── analysis/    # 剖析篇 —— ggml Q4_K_M 源码、llama.cpp 量化工具链、可视化动效
├── hw/          # 硬件篇 —— NVIDIA GPU、华为 NPU、端侧芯片矩阵与产业洞察
├── deploy/      # 部署篇 —— Qwen3.8-27B 实战、GGUF 低比特量化框架与脚本
├── benches/     # 性能篇 —— 压测方法、数据集与评测报告（建设中）
├── hotspot/     # 热点篇 —— 行业动态与社区追踪（建设中）
├── extention/   # 扩展篇 —— 推理框架二次开发（建设中）
├── docs/        # 门户 —— 索引、路线图、写作规范
└── .github/     # 治理 —— Issue/PR 模板、CI 工作流
```

完整内容地图见 **[docs/index.md](docs/index.md)**，版本规划见 **[docs/roadmap.md](docs/roadmap.md)**。

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/zhuzhichaoTM/hello_inference_engine.git
cd hello_inference_engine
```

### 选项 A：部署一个 27B 大模型（llama.cpp，约 5 分钟读懂链路）

```bash
# 1. 看实战报告（环境、参数、压测结果全在里面）
open deploy/Qwen3.8-27B/Qwen3.8-27B部署方案与实战报告.md

# 2. 启动脚本在云端 GPU 机器上执行（先按报告 §5 配置好模型路径）
bash deploy/Qwen3.8-27B/start_qwen3.8_27b_server.sh

# 3. 验证
curl http://127.0.0.1:8000/v1/models
```

也可用：[GGUF 量化工具链](deploy/gguf_quant/)（从 BF16 一键生成 Q4_K_M 等档位）。

### 选项 B：生成 GGUF 量化模型（1-bit ~ 16-bit）

```bash
cd deploy/gguf_quant

# 一键全流程（示例：Q4_K_M + 数据集校准）
python -m quant_framework.quant_cli \
  --model-path <BF16_GGUF或safetensors目录> \
  --quants Q4_K_M \
  --imatrix-mode dataset \
  --calib-file calibration/final.txt

# 仅生成 imatrix / 仅执行量化（分步）
python -m quant_framework.quant_cli --imatrix-only --help
python -m quant_framework.quant_cli --quant-only --help
```

详见 [deploy/gguf_quant/](deploy/gguf_quant/) 与 [calibration/](deploy/gguf_quant/calibration/)。

### 选项 C：系统学习推理基础

| 我想…… | 入口 |
|---------|------|
| 建立推理全景认知 | [basics/chapters/01_第1章_大模型推理概述.md](basics/chapters/01_第1章_大模型推理概述.md) |
| 理解量化原理（源码级） | [analysis/quant/ggml/ggml-quants源码详细分析.md](analysis/quant/ggml/ggml-quants源码详细分析.md) |
| 看懂 Q4_K_M（含动效演示） | [analysis/quant/ggml/q4km_debug/](analysis/quant/ggml/q4km_debug/) |
| 选硬件（GPU/NPU/端侧） | [hw/GPU推理优化_硬件架构与运行原理指南.md](hw/GPU推理优化_硬件架构与运行原理指南.md) |
| 读量化综述 | [paper/awq_gptq_analysis.md](paper/awq_gptq_analysis.md) |

## 本地写作与校验

```bash
# Markdown 格式检查（与 CI 一致）
npx prettier --check "**/*.md"

# 大文件红线：>5MB 走 GitHub Releases，不进 git（CI 会拦截）
```

写作规范见 [docs/writing-style.md](docs/writing-style.md)。

## 社区与贡献

本仓库采用 OSS-first 协作：Issue 先行、PR 小步、CI 必过。欢迎纠错、补实测数据、新增硬件/框架实战。

- **[贡献指南](CONTRIBUTING.md)** —— 分支、提交、评审规范
- **[行为准则](CODE_OF_CONDUCT.md)** —— 社区基本约
- **[安全策略](SECURITY.md)** —— 漏洞报告方式（请勿公开提 Issue 披露）
- **提 Issue**：[Bug / 纠错](.github/ISSUE_TEMPLATE/bug_report.yml) · [内容提议](.github/ISSUE_TEMPLATE/content_proposal.yml)

## 最新动态

- `fd80b90` 新增 GPU 推理优化：硬件架构与运行原理指南
- `448478d` 新增英伟达和华为 AI 算力芯片发展现状洞察
- `14142e7` 补充量化过程中的数据集
- `55fb8bb` GGUF 量化模型生成工具：低比特量化执行链路
- `9a4b397` 基于 llama.cpp 部署 Qwen3.8-27B 并做性能测试

## 📄 License

本项目采用 [Apache License 2.0](LICENSE)。
