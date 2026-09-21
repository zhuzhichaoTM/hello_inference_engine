# 性能篇 · 压测方法、数据集与评测报告

> **定位**：用数据说话的评测中枢。汇总主流开源模型的评测方法、数据集，以及各推理关键特征在不同部署引擎上的质量与性能对照。

| [顶层](../README.md) | [文档索引](../docs/index.md) | [路线图](../docs/roadmap.md) | [写作规范](../docs/writing-style.md) |

## 内容

| 主题 | 入口 | 内容 |
|------|------|------|
| Q4_K_M 量化 10 组档位对照 | [../deploy/quant/quant_verification/](../deploy/quant/quant_verification/) | A0 基线 + B1~B9 + Z 交付档，PPL 与 smoke 双维度 |
| Qwen3.8-27B 部署压测 | [../deploy/Qwen3.8-27B/Qwen3.8-27B部署方案与实战报告.md](../deploy/Qwen3.8-27B/Qwen3.8-27B部署方案与实战报告.md) | Prefill/Decode 基准与在线吞吐 |
| GGUF 工具链压测日志 | [../deploy/gguf_quant/logs/](../deploy/gguf_quant/logs/) | bench / quant 各阶段原始日志 |

> 本目录为评测门户：原始数据沉淀在各实战目录，本篇负责收敛为可引用的报告。新增报告落在本目录，原始日志留在实战目录。

## 评测规范（必读）

1. 每个数字必须附：硬件 / 驱动 / 框架版本 / 模型版本 / 复现命令，详见 [写作规范](../docs/writing-style.md)。
2. PPL 与 smoke 结果需同时保留原始日志与解析脚本（如 `paired_test.py` / `parse_smoke.py`）。
3. 跨引擎对比需注明对齐方式（batch / 并发 / 精度档位）。

## 状态

🚧 建设中：首份收敛性评测报告待撰写（见 [roadmap](../docs/roadmap.md) v0.3）。认领请提 [内容提议](../.github/ISSUE_TEMPLATE/content_proposal.yml)，注明归属 `benches`。
