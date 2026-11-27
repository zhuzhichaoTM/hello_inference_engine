# 大模型推理优化每日热点档案

> 档案日期：2026-11-27（本自动化于该日执行）
> 关注窗口：2026-11-26（周四）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，四家单页取全无截断）+ releases API + tag/release 精确查询 + **47 个关键 PR 正文经官方 issues API 抓取 markdown（无 HTML 降级）** + llama.cpp `v1.1.0` 官方发布说明全文（45,395 字符）+ HN Algolia（HTTPS，`numericFilters` 按 2026-11-26 epoch 窗口 1795651200–1795737600）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 75 / SGLang 83 / TensorRT-LLM 23 / llama.cpp 41 合计 **222** 条（2026-11-26 为周四，工作日正常量，较上一期 221 基本持平）；**窗口内有新正式版本：llama.cpp `v1.1.0` 于 2026-11-26T09:16:47Z 发布（prerelease=false，tag target commit `074f1f3f`，发布说明 45,395 字符全文核验），对齐上一期「v1.1.0 tag/release 均 404、不记为发布」的悬案——本期正式落地**；其余无新正式版/无新 prerelease（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、TRT-LLM v1.3.0 11-20 与滚动 RC v1.4.0rc4 11-18 均落窗口前；llama.cpp 另推进 nightly b12774–b12800 与窗口内 `ggml` bump 1.1.1 提交 `#2a59da5bf`）。本期头号事件为 **llama.cpp v1.1.0 正式发布 + KV 卸载/分层生命周期簇（淘汰顺序 × 状态检查点边界 × 每请求行宽）成为 agentic/多轮缓存命中率的一等公民**，据此新增累计结论第 27 条；次主线为 vLLM `#67064` Mamba P/D 前缀缓存切点重建（D 命中 0/123,032→59.4%）、SGLang `#51196` int32 行偏移溢出与 `#53032` DSpark 高并发 1024 行边界。HN 引擎官方更新类故事无（引擎相关帖均二手线索/未核验）。
> 来源拼接规则：下文 `#NNNNN` 为官方 PR/提交编号，vLLM → `https://github.com/vllm-project/vllm/pull/NNNNN`、SGLang → `https://github.com/sgl-project/sglang/pull/NNNNN`、TRT-LLM → `https://github.com/NVIDIA/TensorRT-LLM/pull/NNNNN`、llama.cpp → `https://github.com/ggml-org/llama.cpp/pull/NNNNN`；官方发布说明见 `https://github.com/ggml-org/llama.cpp/releases/tag/v1.1.0`。

---

## 1. 版本状态与本期头号事件：llama.cpp v1.1.0 正式发布 + KV 卸载生命周期簇（新增第 27 条）

### 1.1 头号事件：llama.cpp v1.1.0 正式发布（官方一手，release，累计悬案落地）

- **主题**：llama.cpp v1.1.0 于 2026-11-26T09:16:47Z 正式发布（非预发布）。
- **内容详情（事实）**：tag `v1.1.0`，target commit `074f1f3f730c1d17f34645d62c1859c74bedbdd8`，发布说明 45,395 字符全文核验。核心：新增 Inkling 2 家族（base / Mini / DSpark）、Domyn Edge 4B、Zaya1 8B、MiniMax-M3.1、RAON-1-mini（30B-A4B MoE）、Hy4 flash（220B/12B active，DSA iSWA cache）；**DFlash 投机解码拆分为 `draft-dflash` / `draft-dflash2` / `draft-dspark` / `draft-lilicorr` 四种 draft 方法**；**弃用旧 `llama_batch` API，改推 `llama_batch_ext` + `llama_process`**；新增 `target_bpw`/`target_size` 按张量自动量化（可断点续算 state file）；`ggml` 更新到 v1.1.0 与 v1.1.1；chat token-aware 解析、named `tool_choice`、slot scheduling 重构为 `server_slot_sched`、Llama Chat Web UI 改版（首屏 JS −63%）；内置 GDN replay 替代 rollback（#31669）、`.scale` 自动应用（#32669）、ON_DEVICE 状态恢复修复（#30318）等此前各期逐条跟踪的 PR 全部收口进正式版。
- **应用场景**：端侧/本地部署升级基线；Inkling 2 / Hy4 flash 新模型开箱；投机解码多 draft 方法选型；需要按 BPW/尺寸预算自动量化的场景。
- **研判（观点）**：这是自 v1.0.0（11-13）以来首个正式版，且把过去十余期的静默正确性修复（GDN replay、`.scale`、ON_DEVICE、prompt cache 双会话交错 #28992）打包交付，**升级优先级高于任一单独 PR**；但 `llama_batch` 弃用与 DFlash 方法拆分属 API 级变更，集成方需按第 16/20 条做破坏性审计。窗口内 `ggml` 已 bump 1.1.1（`#2a59da5bf`），说明主干已继续前进，1.1.0 是稳定快照而非终点。
- **来源**：GitHub releases API `/releases/tags/v1.1.0`，发布日期 2026-11-26T09:16:47Z，官方一手。
- **证据等级**：官方一手。

### 1.2 新增累计结论第 27 条：KV 卸载/分层的「淘汰顺序 × 状态检查点边界 × 每请求行宽」成为缓存命中率与正确性的一等公民

- **主题**：KV 分层栈的优化焦点从「能不能 offload」升级为「以什么粒度/顺序/边界 offload 与回填」。
- **内容详情（事实）**：vLLM `#67952`（SimpleCPUOffload CPU 前缀尾部优先 LRU，opt-in `cpu_lru_eviction` 默认 false）在公开 AgentX 语料（393 Claude Code sessions / 98,827 requests）回放中，CPU 层小于工作集时命中相对 +10%，下游 MiniMax-M3.1-Flash TP4/MI355X 二次 gsm8k 回放 CPU-hit tokens **5,632→23,552（4.2×）**；`#70026` 让 LRU/ARC 优先淘汰非 retained CPU chunk；`#67064` 让 Mamba P/D 在 prefix-cache checkpoint 处截断、D 自行重算尾部，D 命中 **0/123,032→59.4%**；`#68836` offload 每个 attention group 的 prompt tail；`#70068`/`#69626` 修 Mamba 状态块缓存时序与降权；`#65123`/`#68660`/`#69656` 分别修 CPU 淘汰后 runner cache 重发、fetch 跨步保留未消费 lookup、UMBP 同步加载块与清零竞态。SGLang 侧 `#53249`/`#53314`/`#53296` 把 `req_to_token` 行宽收归 pool 单一字段（上一期第 26 条的架构收口），`#52956`/`#52959` 修 SWA 窗口错配与流式会话越窗 KV 回收。llama.cpp v1.1.0 发布说明收录 `#28992`（prompt cache 双会话交错永久漏查，30 天生产证据）。
- **应用场景**：长会话/agentic 多轮、P/D 解耦 + Mamba/线性注意力混合模型、CPU/NVMe 分层 KV 池、PD 跨 TP 尺寸。
- **研判（观点）**：成立条件为「工作集 > CPU/磁盘层容量 + 存在可复用前缀」；观察指标：CPU/NVMe hit tokens、prefix 命中率、prompt tokens recomputed、同会话二次回放一致性；停止条件：按行宽/检查点单一来源派生容量并统一淘汰顺序；验证动作：用公开 AgentX 类语料做「首轮 vs 二次回放」A/B，并显式测 `context_len` 非块整数倍边界。证据等级整体为官方一手，但下游 4.2× 为单次运行、需自硬件复验。
- **来源**：上述各 PR 官方正文 + llama.cpp v1.1.0 发布说明，2026-11-26。
- **证据等级**：官方一手（性能数字多为作者/单次运行，观点部分为研判）。

### 1.3 次主线：静默正确性与数值边界跨引擎延续（第 19/22/24 条）

- **事实**：SGLang `#51196`（per-token int8 quant 的 int32 行偏移在 ≥2^31 元素时回绕越界，83890×25600=21.47 亿元素；条件式 int64 修复后 H100 gsm8k 0.73 vs main 0.725、吞吐 236.35→235.96 tok/s 即 0.998×）；TRT `#22873`（LongRoPE 未设 `max_seq_len` 时对所有请求用长因子，GSM8K flexible 74.68→77.56、strict 37.60→36.92，改静态选择为一次性告警）；vLLM `#68814`（mxfp8 activation scales 在 DP/EP dispatch 后需 swizzle 才能匹配 FlashInfer CUTLASS，2×RTX5090 SM120，CI GPU 不支持 mxfp8 仅验证 skip）；llama.cpp `#33044`（MMF matrix offsets int32 溢出改 64 位，同 `#32965` 的 MMVF 先例）、`#33045`（DKQ 576 在 GQA ratio 非 4 倍数时误选 tile FA kernel）。
- **研判（观点）**：整型边界（≥2^31 元素 / 大 row offset）在多个引擎的量化与 KV kernel 中反复出现，已不构成偶发；建议把「大输入偏移 64 位」纳入升级检查表（第 19 条口径扩为「优化 × 量化 × 并发 × 大规模输入」四类交叉）。
- **来源**：各 PR 官方 issues API 正文，2026-11-26。
- **证据等级**：官方一手。

### 1.4 版本状态与破坏性默认值

- llama.cpp v1.1.0：`llama_batch` 弃用、`llama_process` 引入、DFlash 四种 draft 方法拆分、`--ui-agent` 新增、`media_io` 相关默认（见 #22334 上一期）——按第 16/20 条审计。
- vLLM `#69675` 弃用 `cudagraph_copy_inputs` 并移除编译器管理的输入拷贝；`#70133` 移除 DBRX 模型；`#60505` 新增 `/health_generate` 生成就绪检查（呼应此前 `/health` 在挂起时仍 200 的问题）。

---

## 2. vLLM：2026-11-26 提交（75 条）—— QSA prefill FlashInfer TTFT −8~16%(#69735) + Mamba P/D 切点重建(#67064) + CPU 层 LRU 4.2×(#67952) + sparse_disk 权重热更(#69112) + KV Offload 簇

### 2.1 `#69735` [Perf][Qwen4Exp] Run QSA prefill steps with FlashInfer on Blackwell（官方一手，perf，prefill/TTFT）

B200 上 QSA attention 占 Qwen4Exp（Qwen3.8-Flash-Next）8K prefill 步 **26% GPU 时间**。本 PR 把含 prefill 的步改用 FlashInfer `QToken-KvBlock-Sparse` kernel：该 kernel 一次处理同一请求 8 相邻行、对其选中 block 并集共享 K/V 加载（原 Triton 逐行），选择与计算不变。**TTFT −8%~−16%，生产 c32 +4.2%**；FlashInfer 比重调的 Triton（48 配置最优）快 2.4–3.7×。Prior work：GB10 上 QSA 仅占 12%，同类想法 #55394/#55430 曾 −1.6% TTFT 而关闭。**应用场景**：Blackwell 上 Qwen3.8-Flash-Next 长 prompt。研判（观点）：收益依赖 QSA 在 prefill 中的占比，需按硬件/模型分层验收；decode 仍走 Triton。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 2.2 `#67064` [KV Connector][NIXL] Cut P/D loads at the Mamba prefix-cache checkpoint（官方一手，fix/perf，P/D + Mamba）

Mamba `align` 前缀缓存 + P/D 下，P 算到 N-1 并在块网格外发状态，保留策略只留 prompt checkpoint C（在 P 侧），导致下一次直接路由到 D 的短跟进请求整段历史重算：#63610 在 Qwen3.5-9B（block 576, GSM8K 多轮）测得 **D 命中 0/123,032 prompt tokens，拿到 checkpoint 后 59.4%**。本 PR 让 P 在 C 处截断 prefill、D 自算尾部 `[C,N)`：无需改传输协议、无需 P 侧 retained-block 生命周期、无需 P/D 对 C 一致。代价：Qwen3.5-122B TP2 block 2112 下尾部 ≤1024 token ≈24ms / 2112 ≈35ms / 4224 ≈60ms，建模负载约 0.7–1.0% D step time，`FULL_DECODE_ONLY` 步占比 98.9%→98.5%。**应用场景**：条件 P/D + Mamba/混合线性注意力多轮对话。研判（观点）：以重算 <block 的尾部换取协议与生命周期简化，工程上优于搬运 checkpoint；数字为作者测量/建模、需自有硬件复验。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 2.3 `#67952` [Bugfix][KV Offload] Requeue CPU prefixes tail first in SimpleCPUOffload（官方一手，perf，CPU 分层，opt-in）

为 eager `SimpleCPUOffloadConnector` CPU 层加 opt-in LRU 淘汰（`kv_connector_extra_config: {"cpu_lru_eviction": true}`，默认 false 保持原序；取代改默认的 #65561）。公开 AgentX 语料（393 sessions / 98,827 requests）CPU 回放：CPU 层小于工作集时 prompt tokens recomputed 少 0.3–1.7 点、CPU hits 相对 +10%，容纳下工作集后两者持平；下游 MiniMax-M3.1-Flash TP4/MI355X（GPU 524K token、CPU 16GiB/rank）二次 gsm8k 回放 **CPU-hit 5,632→23,552（4.2×）**、flexible 0.9621→0.9651（各一次）。MI355X 216 passed/11 skipped。**应用场景**：长会话 agentic 负载 + 小 CPU 池。研判（观点）：默认关闭说明作者谨慎，升级须显式开启并用真实流量回放。来源：官方 PR 正文，2026-11-26。证据等级：官方一手（单次运行）。

### 2.4 `#69112` [RL][Weight Transfer] Add sparse_disk backend for sparse deltas through storage（官方一手，perf，权重热更新）

RL 在线权重热更新的 sparse-delta 经共享存储下发后端。8×H20 单机、并发 32 加载下，patch 修改 94 个 checkpoint shard 中 3 个（含 45 个 GDN conv1d 权重）；同节点对比 **sparse NCCL 2.0–2.4s、dense NCCL 0.74–0.90s**（dense 用 `--gpu-memory-utilization 0.95`，默认 1GiB 接收缓冲会受限）。**应用场景**：RLHF/在线权重更新、稀疏增量同步。研判（观点）：sparse_disk 解决存储路径贯通而非追求比 NCCL 快，与第 21 条「权重热更新时序竞态」配套审计。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 2.5 其他要点（官方一手）

- **`#70026` [KV Offload] Prefer eviction of non-retained CPU chunks**：LRU 在淘汰堆中同时记录 retention 与 recency，ARC 先扫 T1/T2 中非 retained chunk。第 27 条配套。
- **`#70068`/`#69626`/`#65123`/`#68660`/`#69656`**：修 Mamba decode resume 缓存持有后继状态的块（RTX3050Ti WSL2 实测）、降权被取代的 Mamba prefix 状态、CPU 淘汰后重发 runner cache hit、P2P fetch 跨步保留未消费 lookup answer、UMBP 同步加载块不参与清零（MI355X 378 passed，并移除测试自身竞态）。
- **`#66805` [KV Offload] GDS kv offloading support** / **`#68836`** offload 每个 attention group 的 prompt tail / **`#69870`** Mooncake 拒绝缺失的 draft context KV。
- **`#70004` [ROCm][HY4] fp8_ds_mla with sinks, ShardQ and iHC SP**：8×MI350X（gfx950）补齐 HY4 在 ROCm 与 NVIDIA 的差距；2048 in/128 out decode on/off 1.000/0.998、0.997，运行内离散 ≤0.11%。
- **`#70178` [Perf][DCP] Size sparse-indexer decode logits by DCP-local context**：按各 DCP rank 本地 KV 上下文而非全模型上下文分配 decode logits，释放 CUDA graph 显存给 KV，不改全局 top-k。
- **`#68814`** mxfp8 scales swizzle（见 1.3）；**`#70116`** 保留 HF processor 前置 BOS（Emu3/Molmo2 纯文本）；**`#69621`** Kimi-K3 透明图按 checkpoint 背景合成；**`#65741`** `/v1/completions` top_logprobs 键冲突保留最可能候选（不改采样/文本，temperature=0 边界未复测）。
- **`#69852`/`#69967`/`#69969` [Security]**：保留 chat 请求中的 HF tokenizer options、约束 Jina ranking 提示预处理、限制 Cohere serving 内联请求模板——第 24 条服务安全延续。
- **结构性/标题级**：`#69675` 弃用 `cudagraph_copy_inputs`、`#70133` 移除 DBRX、`#60505` `/health_generate`；Rust 前端约 15 条（`#70031` `/v1/decisions`、`#70092`/`#70090`/`#70091`/`#70089` 请求校验与媒体错误分类、`#70093`/`#70096`/`#70095` 词表/布尔/abort logprobs、`#69614`/`#69382`/`#68933` 解析器与校验合并、`#67945` ZMQ 恢复、`#67700` gRPC Render RPC 等）；多条 ROCm/CI/XPU 变更。

---

## 3. SGLang：2026-11-26 提交（83 条）—— 行宽单一来源(#53249/#53314) + int8 int64 行偏移(#51196) + DSpark 1024 行边界(#53032) + HiCache Engram(#46248) + diffusion 大簇

### 3.1 `#53249`/`#53314`/`#53296` 行宽单一来源收口（官方一手，refactor，上一期第 26 条）

`#53249` 把 `req_to_token` 行宽收归 pool 上的 `row_width` 单一字段（原文：除一个被修的值外所有值不变）；`#53314` 让每个注意力后端从 pool 一次派生 KV table bounds（无值变化）；`#53296` 在 decode Mamba pool 测试中改用 `row_width`（test-only，CPU-only 运行）。**研判（观点）**：这是上一期第 26 条「请求长度边界 × 每请求表宽静默越界」的结构性收口——把行宽变为单一事实来源，消除「按 `context_len` 还是按行宽分配」的分歧。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 3.2 `#51196` [Fix] kernels/quantization: Use int64 row offsets in the per-token int8 quant kernel（官方一手，fix，数值边界）

`_per_token_quant_int8` 用 int32 `tl.program_id(0)` 计算 `row_id*stride`，输入 ≥2^31 元素时回绕为负偏移、越界读写；fp8 同类已用 int64、#51148 已修 group kernel。本 PR 复用 `USE_INT64_OFFSETS` constexpr，仅在 `x.numel() >= 2**31` 时升 64 位，小形状编译结果与 main 完全一致。测试用 83890×25600（21.47 亿元素）；H100 `TestW8A8Int8` gsm8k 0.73 vs main 0.725（200 例，阈值 0.69），吞吐 236.35→235.96 tok/s（0.998×）。未验证：CI 注册测试、其他 GPU、ROCm/XPU。**应用场景**：W8A8-INT8 线性层与 channel-wise int8 fused MoE 的超大 activation。研判（观点）：属第 19 条整型边界跨量化路径的又一铁证；吞吐无损以条件式修复为前提。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 3.3 `#53032` [Perf][NVIDIA] dsv41: Lift 1024-row decode limits in fused mHC post and FlashInfer MXFP8 autotune（官方一手，perf，投机解码高并发）

DSv4.1-Flash 高并发投机解码（DEP16、`cuda-graph-max-bs-decode=192`、DSpark block 5 → 192×6=**1,152 verify rows/rank**）跨 1024 行触发两处回退：fused mHC post `MAX_ROWS=1024` 导致拆成 tilelang+独立量化（seam kernels **2.70ms→1.25ms**），FlashInfer MXFP8 autotune 只调 ≤1024、运行时桶 1280 落 1-CTA。修复：`MAX_ROWS` 1024→2048，并按实际 verify token 数（而非请求数）触发 autotune 且向上取整到混合桶。合计 **−1.2ms/step**（各 −0.6ms）。副作用：模型自有 prefill autotune hook 每次启动都跑。研判（观点）：收益限于高并发投机 verify 行 >1024 的场景，低并发无感；属第 23 条语义/边界治理的性能侧。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 3.4 `#46248` [Feat] hicache: Add unified Engram store and support Qwen4-Exp（官方一手，feat/perf，KV 分层）

统一 Engram store：首次 eager lookup 发布匿名不可变快照后，HiCache embedding adapter 释放原 CPU 权重存储（保留参数对象）。受控配置下 medians（native→HiCache）吞吐变化 **−0.92%~+0.34%**、请求延迟 −0.34%~+0.94%；11.92GiB TP shard 被 pinned allocator 进位到 16GiB，Mooncake 用表大小 mmap 并释放原 CPU 存储。**注意**：为 closed-wave Engine 计时，非持续到达 serving。研判（观点）：功能与内存收益明确，吞吐中性，不能据此宣称加速。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 3.5 其他要点（官方一手）

- **`#52452`** DFlash committed block 一次 launch（预期不改输出、未附端到端精度）；**`#52451`** Prims-TS 稀疏 MLA kernel 在后端构建时预编译（GB300 输出 bitwise 一致）。
- **`#52956`** hicache 在 buffer 模式下存储存储查询发现缺失的 SWA 窗口；**`#52959`** 流式会话扩展自身行时回收窗口后的 SWA KV；**`#51519`** encoder 在同一 tick 先中止等待请求再 drain embedding（原中止被静默忽略，断连客户端仍跑完 prefill+decode）。
- **`#51234`** PD 在不同 attention TP 尺寸间转移 NVFP4 page-row KV；**`#47462`** 把 MoriTransport 抽到统一 Transport 接口（第 17 条）；**`#53153`** 在 prefill server 上直接回答 one-token 请求；**`#53392`** K3.1 DCP 下允许 fake PD 传输后端；**`#48172`** Inkling-2-Mini 支持；**`#50613`** GigaChat 4 支持；**`#49502`** Ascend A2 上 DSA MLA KV 存 INT8。
- **`#51786` [Fix] diffusion/serving**：保护请求隐私、把 media job 限定到租户（第 24 条安全延伸到 diffusion serving）；**`#40872`** AMD aiter NEXTN draft extend 在 CK kernel 崩溃（重跑待定，标注未闭合）。
- **diffusion 大簇约 15 条**：`#53305`/`#53429`/`#43242`/`#52480`/`#52528`/`#35014`/`#52918`/`#52657`/`#52119`/`#52601`/`#52064`/`#51505`/`#51784`；另有 NPU/MLU/XPU CI 与 cookbook 同步。

---

## 4. TensorRT-LLM：2026-11-26 提交（23 条）—— DSv4.1 decode 稳态准备(#22370) + SM107 解码路径(#22694) + PrimTS MLA opt-in(#22806) + LongRoPE 告警(#22873) + NIXL agent 恢复(#22530)

### 4.1 `#22370` [TRTLLM-16622][perf] DeepSeek-V4.1: steady decode-step prepare, row-bounded CSA2 page tables and cross-rank tactic merge（官方一手，perf）

稳态 decode-step prepare 对未变 batch 跳过 staging launch；CSA2 page table 按行有界，draft shrink 时页数每新块只变一次（原约 0.8 次/128-token 块）；跨 rank tactic 合并统一 padding/count exchange/投机元数据步骤。研判（观点）：与 SGLang `#53032` 同属「把投机解码高并发边界从回退路径拉回融合路径」的跨引擎收敛。来源：官方 PR 正文，2026-11-26。证据等级：官方一手（具体吞吐数字未在正文给出，按描述级处理）。

### 4.2 `#22694` [perf] Enable qualified SM107 DeepSeek-V4.1 decode paths（官方一手，perf）/ `#22806` [fix] dense PrimTS MLA decode opt-in on SM100/SM103

`#22694` 移植 SM107 小 batch DSv4.1 生成路径（CSA2 direct decode、block selection、fused Reindex、capped GVR、low-M query、shared-expert projection、replicated mHC、paired Engram），来源候选 job 1131468/1131469，SM107 回归 job 1133308 报 45 passes。`#22806` 把 dense PrimTS MLA decode 默认只在 SM107（Rubin）启用，SM100/SM103 需显式 `TLLM_FMHA_LIBS=+prims_ts`（B300 实测）。研判（观点）：SM100/103 上「默认开」曾带来回退风险，改为 opt-in 属第 16 条默认性能化的反向校正。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 4.3 `#22873` [fix] Warn when LongRoPE uses the long factors for every request（官方一手，fix，静默精度）

128k Phi-3/3.5、Phi-4-mini 的 PyTorch backend 加载时建一张 RoPE 表；`max_seq_len` 未设（默认）或大于 `original_max_position_embeddings`(4096) 时对**所有**请求用长上下文因子，牺牲短请求精度（#22655）。本 PR 保持静态选择（KV cache 不混因子集）但加一次性告警并写文档。GSM8K（1319 例，GB300）：不设 `max_seq_len` flexible 74.68 / strict 37.60，`--max_seq_len 4096` 分别 77.56 / 36.92。逐请求选因子被否决（融合 RoPE 路径无 per-request 表偏移、KV 块复用会混因子、PD 需携带选择）。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 4.4 其他要点（官方一手/描述级）

- **`#22530` [fix] Recover NIXL agents and fail stalled requests**：agent 失败关闭新收发准入、跨 rank 传播失败意图并用会话取消停发布；3 个回归在旧源码失败、修后通过；测试文件 1297→300 行（17 cases）。
- **`#22662` [fix]** 保留被 end id 停止请求的最后一行 generation logits（1.4.0rc3/rc4，L40S）；**`#21111` [fix]** Kimi-K3 chat prompt ids 取 renderer、`count_tokens` 与 usage 一致；**`#22710` [fix]** Qwen3 tool parser 保留与 tool-call opener 同一流式块中的文本。
- **`#22614` [feat]** DeCo-Attention 作为 VisualGen Cute DSL backend 的 opt-in 量化注意力 recipe（QK 用 NVFP4 操作数、无需校准/改 checkpoint）；**`#22957`/`#22846`** cache-isolation 与 Helix cache metadata 测试修复；**`#22830`** VR200 QA selection profile。
- 标题级/基础设施：`#22339` Qwen3.6 长生成 QA、`#22773` 恢复解聚覆盖、`#21899` fat sqsh 按 commit 缓存、`#22565` Blackwell/Rubin 长跑启用 nvMatmulHeuristics。

---

## 5. llama.cpp：2026-11-26 提交（41 条）—— v1.1.0 正式发布 + Inkling 2 五链融合(#33060) + MoE 权重范数融合(#32943) + int64 MMF 偏移(#33044) + Hexagon 2.0(#32582)

### 5.1 `#33060` CUDA: fuse the Inkling 2 short conv, depth mix, loop controller, L2 clip and situ chains（官方一手，perf，端侧 decode）

Inkling 2 Mini decode 每 token 约 5,745 个 kernel（多为 n_embd 行上的小算子）。本 PR 融合五条链（short conv 192/token、depth mix 120、loop controller 96、L2 clip 144、situ 48），**kernel launches 5,745→2,584**。RTX PRO 6000 Blackwell、IQ4_NL、`-fa 1 -r 3`：pp4 **+58~61%**（222.8→357.8 t/s）、pp512 +16~22%、tg128 **+28~31%**（100.4→131.6）。精度：conv/clip/situ 保序 KLD 0；depth mix 与 loop controller 求和顺序变化使 mean KLD 0.00205、PPL ratio 0.9996±0.0013（对比 master 自己 `-ub 256` 的 0.00225）。**应用场景**：端侧 Inkling 2 Mini 低并发 decode。研判（观点）：收益显著且数值等价在噪声内，属第 16 条「算子融合 × 精度微回退」的标准验收样例。来源：官方 PR 正文，2026-11-26。证据等级：官方一手。

### 5.2 `#32943` CUDA: fuse the top-k MoE weight norm with an additive sum bias（官方一手，perf，量化 MoE）

融合 kernel 把分母算作 `max(sum, clamp) + bias`，每层 MoE 去掉 4 个 kernel（sum_rows、bias scale、div、routed scale）；CUDA graph 下 tg128 在共享机上处于运行间噪声内。KLD vs master（40 chunks×512）0.0038（同 top-p 97.8%），对照 master `-ub 256` 自身 0.0040。来源：官方 PR 正文，2026-11-26。证据等级：官方一手（吞吐收益标注为噪声内）。

### 5.3 `#33044`/`#33045`/`#33063`/`#32533` 数值边界与 kernel 覆盖（官方一手）

`#33044` 把 MMF matrix offsets 改 64 位计算（同 `#32965` 对 MMVF 的修复，`#32175` 的 x 改动在 #31047 后不再适用而遗漏 y/dst）；`#33045` 修 ROCm CI：DKQ 576 仅当 GQA ratio 为 4 的倍数才选 tile FA kernel，其余报不支持并回退；`#33063` ggml/CUDA+HIP 支持带 indexed bias 的 Flash Attention（并入 v1.1.1 的 FA 数据加载重构）；`#32533` CPU/CUDA 的 MUL_MAT/MUL_MAT_ID 支持 BF16 src1。

### 5.4 `#32582` ggml-hexagon 2.0（官方一手，perf，NPU 异构）

支持 int-hmx、自动 f32→f16 lowering、通用 DMA pipeline；作者称需对每 op/kernel/solver/fusion 做大改以支持混合类型。已在 Android/Linux/Windows（v73–v85，单/双/四 NPU）与 Gemma4/Qwen3.x/LFM2.x/3 等主流模型族测试。示例日志：prompt eval 约 5,614→8,279 tok/s（同一构建不同环节）、eval 35.19 tok/s。研判（观点）：数字来自作者日志、非配对 A/B，仅作能力证明（第 16 条口径）。来源：官方 PR 正文，2026-11-26。证据等级：官方一手（性能数字为作者报告）。

### 5.5 其他要点（官方一手）

- `#33073` DFlash2 支持 Inkling-2-Mini；`#33098` OpenCL 多 token f32 GEMV 用于 spec verify；`#30606` SYCL 把 BMG D=256 单 query decode 路由到 FA vector kernel；`#492d23cc9` bailingmoe3 融合 ssm_conv+silu 去冗余拷贝；`#0e4e925e8` CUDA 融合 Inkling 2 short conv/depth mix/loop controller/L2 clip/situ 链（见 5.1）。
- `#28148b5f0` 拒绝 Hadamard-folded 模型的 `SPLIT_MODE_TENSOR`；`#33127` CI 放松 spec decode 接受长度下限、`#33141` 放松 DEPTH_MIX 容差；UI 簇（slash 命令、NVFP4 4-bit 展示、agent trace 等）；`#2a59da5bf` ggml bump 1.1.1。

---

## 6. 趋势研判与累计结论

累计结论继承第 1–26 条（见上一期档案第 6 节），**本期新增第 27 条**；本期铁证按既有条目归类并强化如下：

1. **【新增第 27 条】KV 卸载/分层的淘汰顺序 × 状态检查点边界 × 每请求行宽成为 cache 命中率与正确性的一等公民**：vLLM `#67952`（CPU 命中 4.2×，AgentX 回放 +10%）/`#70026`/`#67064`（Mamba P/D 切点，D 命中 0→59.4%）/`#68836`/`#70068`/`#69656` + SGLang `#53249`/`#53314`/`#52956`/`#52959` + llama.cpp `#28992`。成立/观察/停止/验证条件见 1.2。
2. **版本交付里程碑**：llama.cpp v1.1.0 正式发布（上期悬案落地），把十余期静默正确性修复打包；`llama_batch` 弃用与 DFlash 拆分为 API 级破坏变更（第 16/20 条）。
3. **静默正确性（第 4/19/22/24 条）继续最高频，整型边界成跨引擎通病**：SGLang `#51196`（≥2^31 元素 int32 回绕）、llama.cpp `#33044`（MMF 偏移）、TRT `#22873`（LongRoPE 静态长因子静默掉精度）、vLLM `#68814`（mxfp8 scale 布局）、`#65741`（top_logprobs 键冲突）。建议把「大偏移 64 位 + 默认配置掉精度」列入升级检查表。
4. **投机解码高并发边界跨引擎收敛**：SGLang `#53032`（1,152 verify rows，−1.2ms/step）、`#52452`/`#52451` 与 TRT `#22370`/`#22694` 同属「把 >1024 行从回退路径拉回融合路径」；llama.cpp DFlash 四方法拆分（v1.1.0）。
5. **默认性能化的反向校正**：TRT `#22806` 把 SM100/103 上 PrimTS MLA decode 改为 opt-in；vLLM `#67952` LRU 淘汰默认 off——第 16 条「默认开」并非单向，回退/ opt-in 因硬件或工作集不确定而回归。
6. **服务安全（第 24 条）扩散到 diffusion 与请求模板**：vLLM `#69852`/`#69967`/`#69969`、SGLang `#51786`（租户隔离）。
7. **观点**：本期无「跨引擎同 bug 同日爆发」的戏剧性事件，主线是**结构收口**（行宽单一来源、版本打包、分层生命周期），符合周四工作日「修复与重构多于新特性」的节奏；升级决策应按「先版本、再生命周期、后边界」排序。

---

## 7. 可执行清单（面向工程落地）

1. **版本基线（P0）**：llama.cpp 生产环境升级到 v1.1.0，同步审计 `llama_batch` 弃用与 DFlash 四种 draft 方法（原 `draft-dflash` 调用点需重新选型）；记录 ggml 1.1.0/1.1.1 依赖。
2. **KV 分层/agentic 缓存（新增，P0）**：启用 vLLM `#67952` 需显式 `cpu_lru_eviction=true` 并用真实多轮流量回放验证 hit tokens；P/D + Mamba 升级 `#67064`（验证短跟进前缀命中）；按 `#53249`/`#53314`/`#53296` 对齐 SGLang 行宽单一来源；`#70026`/`#65123`/`#68660`/`#69656` 一并升级。
3. **整型/数值边界**：W8A8-INT8 大 activation 升级 SGLang `#51196`；llama.cpp 升级含 `#33044`/`#33045`（MMF 64 位、DKQ 576 GQA 非 4 倍数）；TRT 使用 Phi-3/Phi-4-mini 时按 `#22873` 设 `max_seq_len<=4096` 并在日志中确认告警消失。
4. **投机解码高并发**：DSv4.1-Flash DEP16 + DSpark 评估 SGLang `#53032`（>1024 verify rows 才有收益）；TRT SM107 评估 `#22694`，SM100/103 上 PrimTS MLA 按 `#22806` 显式 opt-in。
5. **性能验收（按硬件分层）**：Qwen3.8-Flash-Next B200 评估 vLLM `#69735`（TTFT −8~16%，GB10 不适用）；Inkling 2 Mini 端侧评估 `#33060`（pp4 +58~61%，KLD 0.00205）；RL 权重热更评估 `#69112`（sparse NCCL 仍快于 sparse_disk）；HiCache Engram `#46248` 按吞吐中性验收。
6. **谨慎采用（验证未闭合）**：AMD aiter `#40872`（重跑待定）、llama.cpp `#32582`（作者日志非配对 AB）、vLLM `#67064`/`#67952` 的下游数字（单次运行）、SGLang `#52452`（无端到端精度）、TRT `#22370`（正文无吞吐数字）、HN 线索（见第 8 节，均二手/未核验）。

---

## 8. 来源、数据质量与限制

- **一手来源**：GitHub commits API（vLLM 75 / SGLang 83 / TensorRT-LLM 23 / llama.cpp 41，均 HTTP 200、单页取全无截断）；releases API 与 `/releases/tags/v1.1.0` 精确查询（45,395 字符全文）；47 个关键 PR 正文经 issues API 抓取 markdown（vLLM 18 / SGLang 13 / TRT 8 / llama.cpp 8，配额 60→4，无 HTML 降级）。
- **证据分级**：官方仓库/API/文档（最高）> 官方博客/发布页 > 可信二手报道 > 社区转述。本期所有性能数字均标注运行环境与来源；作者报告、单次运行、非配对 A/B、CI 未全绿、CPU-only/描述级均已在正文显式标注，不作确定事实。
- **官方博客**：vLLM 官博索引窗口内无新文章（最新为 11-24 ERNIE 5.5 / GR announcement，上一期已补录 ERNIE）；LMSYS SGLang 博客索引窗口内无可核验新条目。不编造内容。
- **社区线索（HN Algolia，二手线索/未核验）**：窗口内 `Ask HN: OpenRouter equivalent to launch in 2026`（1 分）、`RustingStrata – Pure Rust LLM GGUF Engine, Runs Models > VRAM on CUDA and Vulkan`（3 分，`github.com/RustingStudio/RustingStrata`）、`AMD-specific fork of llama.cpp`（275 分，`devnull.land/amd-specific-llama-cpp`，为该帖最高热度但未取得可核验的一手实现细节）、`DFly2: Speculative decoding that reuses the target's sparse KV`（1 分，转推腾讯 AI News）、`bare-metal OS running Qwen3-0.6B 54% faster than llama.cpp`（3 分，推特链接）、`SemShareKV`（1 分，arXiv 2509.24832，2025 年论文）。`sglang`/`tensorrt-llm`/`paged attention`/`FlashAttention` 关键词 0 命中。以上均未当作确定事实。
- **局限**：未对每个 PR 跑 CI 或复现实验；部分 PR 正文仅为标题级或模板；llama.cpp v1.1.0 发布说明中的大量新模型能力属官方声明、未独立验证；`v1.1.1` 仅为窗口内 `ggml` bump 与 nightly，尚未正式发布。
- **下一窗口**：2026-11-27（周五）UTC 全天，关注 llama.cpp 1.1.x 后续 RC/正式版、TRT-LLM v1.4.0 滚动 RC、以及第 27 条相关 PR 的复验与跟进。

---
