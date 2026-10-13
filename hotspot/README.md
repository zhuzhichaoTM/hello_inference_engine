# 热点篇 · 行业动态与社区信息追踪

> **定位**：雷达站。追踪大模型推理相关的行业动态、关键技术论坛、社区讨论与视频资源，沉淀为可回溯的热点档案。

| [顶层](../README.md) | [文档索引](../docs/index.md) | [路线图](../docs/roadmap.md) | [写作规范](../docs/writing-style.md) |

## 内容

| 文件名 | 入口 | 内容简介 |
|---|---|---|
| `llm-inference-hotspots-2026-10-13.md` | [2026-10 月度档案](2026-10/llm-inference-hotspots-2026-10-13.md) | 2026-10-12 大模型推理优化每日热点，涵盖 vLLM 的 MoRI-IO KV 请求状态机与 HiSparse/稀疏 MLA/EPLB、SGLang 的 HiCache MTP/Mamba 状态完整性与 PD 失败语义、TensorRT-LLM 的 KV cache manager V2 转默认（破坏性变更）、llama.cpp 的 KV 维度对齐与图重分配可观测性。 |
| `llm-inference-hotspots-2026-10-02.md` | [2026-10 月度档案](2026-10/llm-inference-hotspots-2026-10-02.md) | 2026-10-01 大模型推理优化每日热点，涵盖 vLLM 的 NIXL KV 生命周期与 HiSparse/NVFP4、SGLang 的 PD/HiCache/DSpark 与 AMD 路径、TensorRT-LLM 的 sparse KV offload 与 retirement deadline、llama.cpp 的 MTP/NVFP4 与多后端稳定性。 |
| `llm-inference-hotspots-2026-10-01.md` | [2026-10 月度档案](2026-10/llm-inference-hotspots-2026-10-01.md) | 2026-09-30 大模型推理优化每日热点，涵盖 vLLM 的 Mamba/MTP 与 ROCm/MLA、SGLang 的 EAGLE3/DFlash/DSpark、TensorRT-LLM 的 KVCacheManager V2 与分离 KV 复用、llama.cpp 的 DFlash 与 batch 顺序。 |
| `llm-inference-hotspots-2026-09-30.md` | [2026-09 月度档案](2026-09/llm-inference-hotspots-2026-09-30.md) | 2026-09-29 大模型推理优化每日热点，涵盖 vLLM 的 KV 传输与前缀缓存约束、SGLang 的 Dspark/P-D 协同、TensorRT-LLM 的 warmup/KV transfer/FP8 稳定性，以及累计趋势与执行清单。 |
| `llm-inference-hotspots-2026-09-25.md` | [2026-09 月度档案](2026-09/llm-inference-hotspots-2026-09-25.md) | 2026-09-24 大模型推理优化每日热点，涵盖 vLLM、SGLang、TensorRT-LLM、Hugging Face Optimum、KV Cache、投机解码、量化、异构硬件及可追溯来源。 |
| `system_prompt.md` | [当前任务提示词](system_prompt.md) | 当前任务的提示词整理版，包含资料采集标准、热点档案格式、月度归档规则、README 刷新规则与远端同步要求。 |

> 追踪渠道：关键技术论坛、X（Twitter）、YouTube 等。每期注明日期与来源链接，观点与事实分离。

## 收录规范

1. 每条热点必须附来源链接与日期，禁止无源转述。
2. 按月份归档热点档案，例如 `YYYY-MM/`；本目录保留索引。
3. 观点必须标注为观点，数据必须标注出处与年份。

## 状态

✅ 已完成首期追踪：见 [2026-09 月度档案](2026-09/llm-inference-hotspots-2026-09-25.md)。

✅ 最新一期：2026-10-13 档案（覆盖 2026-10-12），见 [2026-10 月度档案](2026-10/llm-inference-hotspots-2026-10-13.md)。

> **归档空档说明：** 2026-10-03 至 2026-10-12 未产生每日档案（自动化任务未执行），本期不补写中间日期内容，仅按各自关注窗口采集。如需回溯该区间，请以各引擎官方提交 API 按日期区间重新采集。