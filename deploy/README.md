# 部署篇 · 多框架部署实战与量化工具链

> **定位**：全仓的工程核心。汇总从小模型到 27B 大模型、多模态、VLA 的本地化部署实战，覆盖本地、云端、算力集群，llama.cpp / vLLM / TensorRT-LLM 多框架。

| [顶层](../README.md) | [文档索引](../docs/index.md) | [路线图](../docs/roadmap.md) | [写作规范](../docs/writing-style.md) |

## 内容

| 主题 | 入口 | 内容 |
|------|------|------|
| Qwen3.8-27B 部署方案与实战报告 | [Qwen3.8-27B/Qwen3.8-27B部署方案与实战报告.md](Qwen3.8-27B/Qwen3.8-27B部署方案与实战报告.md) | 单卡 RTX 4090 + llama.cpp，Q4_K_M 主模型 + MTP 投机解码 + 64K 上下文全链路 |
| 启动脚本 | [Qwen3.8-27B/start_qwen3.8_27b_server.sh](Qwen3.8-27B/start_qwen3.8_27b_server.sh) | 一键启动 llama-server（含混合 KV Cache、Flash Attention） |
| GGUF 低比特量化工具链 | [gguf_quant/](gguf_quant/) | 1-bit ~ 16-bit 一键量化框架（CLI + pipeline + 校准集） |
| Qwen2.5-1.5B 量化指导 | [quant/qwen2.5-1.5b-量化指导文档.md](quant/qwen2.5-1.5b-量化指导文档.md) | 小模型量化完整指导 |
| 量化优化方案验证报告 | [quant/quant_verification/量化优化方案验证报告.md](quant/quant_verification/量化优化方案验证报告.md) | 10 组档位对照（PPL + smoke，A0/B1~B9/Z），附原始日志与解析脚本 |

工具链速览（`gguf_quant/`）：`quant_framework/`（CLI / pipeline / imatrix 生成）、`calibration/`（WikiText / Alpaca / 自生成校准集）、`docs/`（4 篇设计与使用文档）、`tests/`。大模型权重（`.gguf`）与运行日志（`logs/`）按 `.gitignore` 不应进仓，见路线图迁移计划。

## 动手顺序

1. 先读 Qwen3.8-27B 实战报告建立端到端认知。
2. 再用 `gguf_quant` 工具链复现量化流程。
3. 用 `quant_verification` 的 10 组对照数据理解档位取舍。
4. 原理不懂回查 [剖析篇](../analysis/)，硬件选型回查 [硬件篇](../hw/)。

## 状态

✅ 核心完整（vLLM / TensorRT-LLM 实战待补充，见 [roadmap](../docs/roadmap.md)）。新增模型实战请先提 [内容提议](../.github/ISSUE_TEMPLATE/content_proposal.yml)，注明归属 `deploy`。
