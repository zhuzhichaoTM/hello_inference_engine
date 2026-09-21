# 技术路线图

## 愿景

**构建极速本地 AI 创作环境，探究具体业务本地加速经验，实现极致 AI 体验。**

目前聚焦自身需求场景的效率提升：**设计、编程、创作**。

## 项目目的

从 0 到 1 探究大模型**微调 / 部署 / 量化 / 优化**及具体业务 AI 应用的全过程：

- 熟悉 **llama.cpp / vLLM / TensorRT-LLM / dynamo** 等模型推理框架
- 掌握**模型量化、算子融合、图优化**等模型优化技术
- 通过 **PyTorch** 等工具将开源基座模型微调出特定业务模型
- 实现**端侧部署**针对特定应用 / 模型的完整工程能力

---

## 阶段路线（每阶段一个 MVP，可演示、可复现）

| 阶段 | 主题 | MVP（最小功能产品） | 关键能力 | 状态 |
|------|------|---------------------|----------|------|
| **v0.2** | 仓库治理与门面 | 一套专业开源仓库骨架 | Issue/PR/CI/文档门户 | ✅ 完成 |
| **v0.3** | 部署闭环：单框架通吃 | `deploy/` 任一模型「下载 → 量化 → 启动 → 压测」脚本链 | llama.cpp + Q4_K_M 量化 | ✅ 已具备（Qwen3.8-27B） |
| **v0.4** | 量化进阶：多档位+验证 | `gguf_quant` 工具链一键出 1~16bit 模型 + `quant_verification` 对照报告 | 档位选型、imatrix、PPL/smoke 回归 | ✅ 已具备（10 组对照） |
| **v0.5** | 优化剖析：源码级 | `analysis/` 从原理到算子的可运行剖析包 | Q4_K_M 动效 + C 参考实现 + llama.cpp 工具链拆解 | ✅ 已具备 |
| **v0.6** | 多框架部署 | vLLM + TensorRT-LLM 部署实战各 1 篇，与 llama.cpp 同基准对比 | 三框架部署差异、显存/吞吐对照 | 🚧 建设中 |
| **v0.7** | 微调入门 | `finetune/` PyTorch LoRA/SFT 最小可跑通示例（基座→业务模型） | 数据构造、训练、合并权重 | 📋 规划中 |
| **v0.8** | 优化进阶 | 算子融合/图优化专项（`extention/`）：给 llama.cpp 提交一个最小可验证改动 | 二次开发、回归验证、before/after | 📋 规划中 |
| **v0.9** | 业务闭环：设计 | 设计场景 demo：本地模型辅助素材/文案/配色生成 | 端侧部署 + 场景封装 | 📋 规划中 |
| **v0.10** | 业务闭环：编程 | 编程场景 demo：本地代码补全/解释器（对接 IDE） | OpenAI 兼容 API + 本地模型 | 📋 规划中 |
| **v0.11** | 业务闭环：创作 | 创作场景 demo：本地写作/润色/脑暴助手 | 提示词工程 + 模型选型 | 📋 规划中 |
| **v1.0** | 对外发布 | 英文 README + 全链路一键复现 + 三场景完整闭环 | 可推广的知识库 | 📋 规划中 |

---

## v0.2 · 仓库治理与门面（已完成）

- [x] LICENSE（Apache-2.0）
- [x] CONTRIBUTING / CODE_OF_CONDUCT / SECURITY
- [x] `.github/`（Issue 模板 ×2、PR 模板、docs-lint CI）
- [x] `.gitignore`（拦截 `.gguf` / `logs` / `.DS_Store`）
- [x] 顶层 README 重写（对标 dynamo：导航表 + Quick Start + 能力矩阵）
- [x] 8 篇章 README 统一格式

**MVP**：访客打开仓库即可按导航找到任意篇章入口。

## v0.3 · 部署闭环：单框架通吃（已完成）

- [x] `deploy/Qwen3.8-27B/`：llama.cpp 单卡部署 + 启动脚本 + 完整实战报告
- [x] `deploy/gguf_quant/`：从 BF16 GGUF 一键生成 Q4_K_M 等档位
- [x] `deploy/quant/quant_verification/`：10 组档位 PPL/smoke 对照 + 解析脚本

**MVP**：一条命令启动 27B 模型并拿到压测数据。

## v0.4 · 量化进阶：多档位 + 验证（已完成）

- [x] `quant_framework/`：CLI + pipeline + imatrix 生成（dataset/self_gen/existing）
- [x] `calibration/`：WikiText / Alpaca / 自生成校准集
- [x] `docs/`：量化方案 × 4

**MVP**：指定档位 + 校准策略，一键产出可复现的 GGUF。

## v0.5 · 优化剖析：源码级（已完成）

- [x] `analysis/quant/ggml/`：Q4_K_M 源码分析 + 动效演示 + C 参考实现
- [x] `analysis/quant/llama_cpp/`：quantize 流程 + k-quant 选型指南

**MVP**：看懂量化每一步，并能本地复现验证。

## v0.6 · 多框架部署（当前重点）

- [ ] vLLM 部署实战：`deploy/vllm-<model>/`（同模型、同基准与 llama.cpp 对照）
- [ ] TensorRT-LLM 部署实战：`deploy/trtllm-<model>/`
- [ ] dynamo 编排实战：至少一个 disaggregated serving 或 KV-aware routing demo
- [ ] 三框架对照表：显存 / TTFT / 吞吐 / 易用性

**MVP**：同一模型在 3 个框架上跑通，输出一份可引用的对照报告。

## v0.7 · 微调入门

- [ ] `finetune/` 目录 + LoRA/SFT 最小示例（建议 Qwen2.5-1.5B，成本低）
- [ ] 数据构造模板：业务指令集格式规范
- [ ] 合并 + 转 GGUF + 回到 `deploy/` 部署验证的完整链路

**MVP**：用自有数据把基座模型微调出一个可用的业务模型。

## v0.8 · 优化进阶（二次开发）

- [ ] `extention/` 首个实战：llama.cpp 新增一个最小可验证改动（算子/档位/后端接入其一）
- [ ] 回归验证脚本 + before/after 性能对照
- [ ] `extention` 拼写修正为 `extension`，保留重定向 README

**MVP**：一份可被上游 review 的改动 diff + 验证报告。

## v0.9 ~ v0.11 · 业务闭环（设计 / 编程 / 创作）

- [ ] 设计场景：本地模型辅助素材/文案/配色生成 demo
- [ ] 编程场景：本地代码补全/解释器（对接 IDE，OpenAI 兼容 API）
- [ ] 创作场景：本地写作/润色/脑暴助手 demo
- [ ] 每个场景沉淀一份「配置模板 + 复现脚本」

**MVP（每阶段一个）**：在对应场景里，本地模型产出可用结果，延迟 < 1s 级。

## v1.0 · 对外发布

- [ ] 全部性能数字可一键复现（含环境说明）
- [ ] 英文 README（`README.en.md`）
- [ ] 固定季度更新行业洞察与硬件矩阵
- [ ] 至少 3 个业务场景达到「微调 → 部署 → 评测」完整闭环
- [ ] 大文件迁移：已进仓的 `.gguf` / `logs` 迁往 GitHub Releases

**MVP**：对外可被引用、可被 star、可被 fork 的专业知识库。

---

## 维护说明

- 每完成一个 MVP 即在顶部表格更新状态，并同步 [docs/index.md](index.md)。
- 新增阶段请先提 [内容提议](../.github/ISSUE_TEMPLATE/content_proposal.yml)，注明归属篇章与目标 MVP。
- 性能类条目遵循 [写作规范](writing-style.md)：必须有环境、复现命令、before/after。
