# 扩展篇 · 推理框架二次开发实战

> **定位**：从"会用"到"会改"。沉淀基于 llama.cpp 等推理框架的二次开发实战：新增算子、自定义量化档位、接入新硬件后端。

| [顶层](../README.md) | [文档索引](../docs/index.md) | [路线图](../docs/roadmap.md) | [写作规范](../docs/writing-style.md) |

## 内容

| 主题 | 入口 | 内容 |
|------|------|------|
| 二次开发实战（待建） | — | 算子新增 / 量化档位定制 / 新后端接入 |
| 可参考的现有代码 | [../analysis/quant/ggml/ggml-quants_q4_k_m.c](../analysis/quant/ggml/ggml-quants_q4_k_m.c) | C 参考实现可作为改码起点 |
| 可参考的调试脚手架 | [../analysis/quant/ggml/q4km_debug/](../analysis/quant/ggml/q4km_debug/) | Makefile + 单测 + VERIFICATION |

> 命名说明：本目录当前拼写为 `extention`（历史原因），v0.3 路线图中计划修正为 `extension` 并保留重定向，见 [roadmap](../docs/roadmap.md)。

## 实战规范

1. 每个实战必须说明：基于的框架版本 + 改动点 + 回归验证方式。
2. 改动代码优先以最小 diff 呈现，完整补丁放附件或 Release。
3. 性能类改动必须有 before/after 对照数据，遵循 [写作规范](../docs/writing-style.md)。

## 状态

🚧 建设中：首个实战待撰写（见 [roadmap](../docs/roadmap.md) v0.3）。认领请提 [内容提议](../.github/ISSUE_TEMPLATE/content_proposal.yml)，注明归属 `extention`。
