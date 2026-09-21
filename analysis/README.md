# 剖析篇 · 量化原理与源码级特征分析

> **定位**：从原理到源码的深度剖析。聚焦 GGUF / llama.cpp 量化实现，用代码 + 可视化讲清低比特量化的每一步。

| [顶层](../README.md) | [文档索引](../docs/index.md) | [路线图](../docs/roadmap.md) | [写作规范](../docs/writing-style.md) |

## 内容

| 主题 | 入口 | 内容 |
|------|------|------|
| ggml Q4_K_M 源码详细分析 | [quant/ggml/ggml-quants源码详细分析.md](quant/ggml/ggml-quants源码详细分析.md) | 量化函数逐行解读 |
| Q4_K_M 量化分析（含 C 参考实现） | [quant/ggml/ggml-quants_q4_k_m_analysis.md](quant/ggml/ggml-quants_q4_k_m_analysis.md)、[ggml-quants_q4_k_m.c](quant/ggml/ggml-quants_q4_k_m.c) | 原理 + 可运行代码对照 |
| Q4_K_M 可视化动效与数学深挖 | [quant/ggml/q4km_debug/](quant/ggml/q4km_debug/) | HTML 动态演示 + VERIFICATION |
| llama.cpp 量化工具链分析 | [quant/llama_cpp/](quant/llama_cpp/) | quantize 流程、k-quant 选型指南 |
| 端侧大模型量化系统设计 | [quant/端侧大模型量化系统设计文档.md](quant/端侧大模型量化系统设计文档.md) | 系统级设计视角 |

`export/` 为导出物暂存目录（构建产物不进仓）。

## 阅读建议

1. 先看 `q4km_debug/` 的动效建立直觉，再读源码分析。
2. 选量化档位时配合 [部署篇](../deploy/) 的 `quant_verification` 实测数据一起看。
3. 动手验证：`q4km_debug/` 内含 Makefile 与测试，可本地复现。

## 状态

✅ 完整。补充新框架剖析请先提 [内容提议](../.github/ISSUE_TEMPLATE/content_proposal.yml)，注明归属 `analysis`。
