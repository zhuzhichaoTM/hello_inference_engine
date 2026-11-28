# 大模型推理优化每日热点档案

> 档案日期：2026-11-28（本自动化于该日执行）
> 关注窗口：2026-11-27（周五）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，四家单页取全无截断）+ releases API + 42 个关键 PR 正文经官方 issues API 抓取 markdown + 8 个 llama.cpp PR 经 GitHub PR 页面 HTML 直抓（`comment-body` div 深度配对）+ vLLM 官博索引核验 + HN Algolia（epoch 窗口 1795737600–1795824000，HTTPS）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 71 / SGLang 94 / TensorRT-LLM 23 / llama.cpp 20，合计 **208** 条（周五工作日正常量，较上一期 222 略回落）；**窗口内无新正式版、无新 prerelease**（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、TRT-LLM v1.3.0 11-20 与滚动 RC v1.4.0rc4 11-18 均落窗口前；llama.cpp 正式版 v1.1.0 11-26 落窗口前，窗口内仅推进 nightly b12801–b12818）。本期头号事件为 **JIT 编译 / CUDA graph 捕获时序从「启动耗时」升维为 TTFT 与可用性的一等约束**（vLLM `#69452` 冷启动运行时编译 6.6–12.1s→1.0–2.2s、TTFT p99 10.41→4.87s；TRT-LLM `#22374` 每 rank JIT 构建改为节点内共享；SGLang `#48850` DP rank 图捕获分歧致 300s 超时/Xid 43），据此**新增累计结论第 28 条**；次主线为 **每请求行宽边界收口的第二波**（SGLang `#53348`/`#53464`/`#53453` 与 `#53491` 的 8192 页边界回潮）。社区侧 34 分 GB10 调参帖已回溯核验为第三方博客（非官方）；GPU 侧信道研究仅作二手线索。
> 来源拼接规则：下文 `#NNNNN` 为官方 PR/提交编号，vLLM → `https://github.com/vllm-project/vllm/pull/NNNNN`、SGLang → `https://github.com/sgl-project/sglang/pull/NNNNN`、TRT-LLM → `https://github.com/NVIDIA/TensorRT-LLM/pull/NNNNN`、llama.cpp → `https://github.com/ggml-org/llama.cpp/pull/NNNNN`；日期均为 2026-11-27（UTC 窗口内合入/发布），不再逐条重复链接。

---

## 1. 版本状态与本期头号事件

### 1.1 版本状态（官方一手）

**主题**：四家引擎窗口内均无新版本。
**内容详情**：vLLM 最新正式版 v0.34.0（2026-11-16T23:22:00Z）；SGLang v0.5.23（2026-11-02T22:14:10Z）；TensorRT-LLM 双线——独立稳定线 v1.3.0（2026-11-20T02:45:00Z）与滚动 RC v1.4.0rc4（2026-11-18T19:16:49Z，prerelease）；llama.cpp 正式版 v1.1.0（2026-11-26T09:16:47Z，上一期已全文核验），窗口内 15 个 nightly 标签 b12801–b12818（b12802/b12813/b12817 未见，按 API 实有记录）。
**应用场景**：升级基线判定；本期无「发版日」效应，修复均未打包进正式版。
**研判**（观点）：按第 3/20 条，未进版的修复只能按源码/镜像 nightly 评估，发布说明不作为「修复已进版」依据。
**来源**：四家 releases API（HTTP 200）。**证据等级**：官方一手。

### 1.2 头号事件：JIT 编译 / 图捕获时序成为 TTFT 与可用性的一等约束（新增第 28 条）

**主题**：编译期常量与图捕获时机把「冷启动」问题延伸为在线请求的 TTFT 尖刺、DP 全组死锁与依赖升级即崩。
**内容详情**：三支独立证据——① vLLM `#69452`：Inkling 2 预填 attention 在 warmup 后仍在引擎线程按 `%16` 编译 Triton 变体，每次新长度编译约 10 个 kernel、在途请求全部停顿；1×H200 上运行时编译 6.6–12.1s→1.0–2.2s，最坏 TTFT（spec off, c=1）7.51→0.81s，p99 TTFT（spec on, c=16）10.41→4.87s，输出贪心一致、kernel 时间 ±1%、GSM8K 噪声内。② TRT-LLM `#22374`：Kimi-K3/MiniMax-M3 预热编译数百个 JIT kernel 且每 rank 重复编译，改为 per-rank worker 预构建 + 节点内默认共享 `/tmp/tllm-jit-<uid>/<launch>`；**行为变更**（`TLLM_JIT_SHARE_DEFAULT=0` / `TLLM_KERNEL_BUILDER=0` 可退）。③ SGLang `#48850`：DSv4.1 prefill + MegaMoE + breakable graph + decoder SWA replay 下，DP rank 一个走 eager break 捕获缺失 row-bucket graph（barrier），另一个已发 MegaMoE 等 grid sync，互相等待 → 300s 超时、Xid 43、重启仍可复现；改为启动时捕获、绝不在 DP rank 跳过的步上捕获。旁证 vLLM `#70277`：nvidia-cutlass-dsl 新版移除 NVVM 私有 dialect 后 `mbarrier` 模块导入即 `AttributeError`，改用 inline PTX。
**应用场景**：长上下文/循环模型（Inkling 2、Kimi-K3、MiniMax-M3）、DP 注意力 + breakable CUDA graph 的 prefill 服务、滚动升级 CUTLASS DSL 依赖的环境。
**研判**（事实与观点分离）：事实是三条官方 PR 正文均给出可复现路径与量化代价；观点是第 28 条独立成条——「编译/捕获时机」已不只是部署耗时，而是直接影响在线 TTFT 与cluster 可用性的运行态问题，应与第 26/27 条（行宽/生命周期）并列进升级检查表。成立条件：调度路径存在编译期常量或首次使用才捕获的 graph；停止条件：warmup 覆盖全部 shape 且 DP 各 rank 图捕获一致；验证动作：冷启动 + 未见过的 prompt 长度 + 多 rank 非均匀 token 数三组合压测。
**来源**：vLLM `#69452`、TRT-LLM `#22374`、SGLang `#48850`、vLLM `#70277` 官方 PR 正文，2026-11-27。**证据等级**：官方一手（`#22374` 含未关闭的共享目录行为变更；`#48850` 为 root cause 分析，未在独立集群复现）。

### 1.3 次主线：每请求行宽边界收口的第二波（承接第 26/27 条）

**主题**：上一期 `#53249`/`#53314` 把 `req_to_token` 行宽收归 pool 后，残留读取点与越界常量在本期集中暴露。
**内容详情**：SGLang `#53348` 修 DSA `prims_ts` 读已被删的 `max_context_len` 致初始化 `AttributeError`、n-gram embedding 表按 `context_len` 分配而 overlap 额外步写 `seq_len` 越界；`#53464` 把投机专用 sizing 函数泛化为所有 decode 的 write-ahead（非投机行宽由 `+1` 变 `+page_size`，无 overlap 的 retire lag 0→1）；`#53453` 把冲突命名改掉（`max_kv_len` 行宽 vs 批内最大序列长；此前一次重命名 `#53348` 曾因此掩盖悬空读）。`#53491` 显示 `#53314` 的副作用：MiniMax-M3.1 1M 上下文下行宽 8194 页 > MSA-v1 indexer 上限 8192，启动即 `ValueError`。
**应用场景**：SGLang 1M 上下文 MiniMax-M3.1、DSA/DSV4/HIP 行组表、投机 + overlap 调度。
**研判**（观点）：这是第 26 条架构收口的「债务偿还期」，升级时须整组而非单 PR  cherry-pick；`#53491` 的 CI 多项 skipped，按「官方一手但验证未闭合」处理。
**来源**：SGLang `#53348`/`#53464`/`#53453`/`#53491` 官方 PR 正文，2026-11-27。**证据等级**：官方一手（`#53491` CI 未全绿）。

---

## 2. vLLM：2026-11-27 提交（71 条）

### 2.1 `#69452` [Bugfix][Model] Inkling 2: stop Triton %16 recompiles in prefill attention（官方一手，fix/perf，TTFT）

**主题**：冷启动后按 token 数编译 Triton 变体，导致在途请求按秒阻塞。
**内容详情**：见 1.2①；运行时编译 6.6–12.1s→1.0–2.2s，c=16 spec on 的 TTFT p99 10.41→4.87s，max 10.64→5.10s；贪心 token 一致、kernel 时间 ±1%、GSM8K 噪声内。
**应用场景**：H200/B200 上 Inkling 2 Mini 服务，prompt 长度 1–16k 均匀分布。
**研判**（观点）：与第 28 条一致，建议把「首次见到的长度」纳入验收；修复为删除运行时编译而非改数值路径，风险低。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手（作者测量，200 请求/臂）。

### 2.2 `#67944` [Perf][UMBP] Default the embedded DRAM pool to 0.99/0.97 eviction watermarks（官方一手，perf/默认变更，KV 分层）

**主题**：MORI v1.2.3.post1 默认 0.9/0.7 水位使 UMBP KV 池平均仅 80% 满，vLLM 改为仅当两个水位都未设置时默认 0.99/0.97。
**内容详情**：满池多容纳约 1/5 KV（约 98% vs 80%）；Kimi-K3.1-Pro 单机 MI355X（2.4TB 池）吞吐 +1.3%、TTFT p90 2.92→2.64s、每请求重算 prompt token −4%（各一次）；Kimi K3 1P1D（1,799GB 预填池）吞吐 +6.2%、TTFT p90 −22%（各一次）。381 单测 + 18 集成通过。
**应用场景**：ROCm + MORI UMBP 的大容量 KV 池 / agentic 长会话。
**研判**（事实/观点）：事实是水位默认值改变；观点是第 3/16 条「默认值风险」的正面案例——但收益数字均为单次背靠背运行，且调整水位会压缩空闲余量，需按显存/DRAM 压力复验。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手（单次运行）。

### 2.3 `#67218` + `#68374` [ROCm][MoRIIO] WRITE 模式 KV 块生命周期两连修（官方一手，fix，PD KV）

**主题**：MoRIIO WRITE 模式下预填 KV 块因传输提前释放/复用 transfer id 或共享 KV 层未注册而永不释放或整批挂起。
**内容详情**：`#67218` 让提前释放与复用 ended transfer id 的请求完成时仍收到释放报告；MI355X + Qwen3-0.6B 上 32/32 个提前释放请求在 main 全部滞留、修复后 0；8 个复用 id 同理。`#68374` 修 Gemma-4-26B-A4B + MTP  drafter 共享 KV 层：main 上 0/64 请求完成（300s 超时），修复后 1,319/1,319，GSM8K 0.472/0.474（无连接器 0.460/0.464），草稿接受率 89.8% vs 90.3%。
**应用场景**：MI355X/MI350X 上 Gemma 4 等共享 KV 层模型的 P/D 解耦 + WRITE 模式。
**研判**（观点）：属第 8/27 条 KV 生命周期，且是「挂起但 /health 可能仍 200」类故障；建议 PD 验收加入「请求完成计数」而非只看精度。
**来源**：两个官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 2.4 `#70246` + `#69404` + `#70214` 静默正确性三连（官方一手，fix，非量化/量化交叉）

**主题**：padding tile 漏读卷积缓存、词表 padding id 走错 Engram 行、lm_head 量化 target 被静默丢弃。
**内容详情**：`#70246` Inkling 2 `short_conv` 以 tile 最后一行的 request start 判断是否需要 conv cache，尾部 padding 行（start=0）掩盖真实行，使其对零卷积；B300 上新增 padding 排程在 main 的 max abs diff 2.5–3.6、修复后通过。`#69404` Hy4-flash Engram 对词表 padding id 120818–120831 及随后两个 token 用错 embedding 行且无报错，同时 `--enable-dbo`/`--ubatch-size>1` 被接受后在 DeepGEMM 启动失败（改为配置期拒绝）。`#70214` Qwen4Exp 构建 lm_head 未带 quant config，`targets` 里的 `re:.*lm_head$` 被静默忽略、头保持 BF16；DGX Spark 上模型占用 72.82→72.23 GiB（省 0.59 GiB），RTX 5090 4 测试通过。
**应用场景**：Inkling 2 含 padding 的 prefill tile；Hy4-flash/DeepSeek-V4.1 Engram；Qwen3.8-Flash-Next 在线量化 lm_head。
**研判**（事实/观点）：事实是三条均为「无报错但输出/配置错」；观点是强化第 19 条——非量化常规路径与量化配置路径的静默错误密度仍高，升级后必须以引用实现对拍而非只看错误日志。
**来源**：三个官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 2.5 `#69838` [Kimi K3][Bugfix] Honor indexer_logits_dtype in the DSA indexer（官方一手，fix/perf，配置未生效）

**主题**：`--attention-config '{"indexer_logits_dtype":"bfloat16"}'` 对 Kimi K3 从未生效，MQA logits 恒为 FP32。
**内容详情**：GB300、Kimi-K3.1-Pro MXFP4、DEP16 P+D 共 32 卡、AgentX c=352、60 分钟窗口：BF16 + `deep_select` 下 tok/s/GPU 35,756→37,130（+3.8%）、TTFT p90 15.90→11.93s（−25%）、E2E p90 −4.3%；131K prefill forward 378.5→327.7ms（−13%）。不带 `deep_select` 的 BF16 仍读 FP32 副本（378.9ms），故默认保持 FP32；50 单测通过。
**应用场景**：Kimi-K3.1-Pro GB300/B300 官方 recipe（已设 bf16）用户。
**研判**（观点）：典型的「配置被忽略但性能数字看起来合理」；建议把关键 attention config 的实际生效值写入启动日志/指标，避免第 3 条默认值风险的隐形变体。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手（单次 60 分钟窗口）。

### 2.6 性能与后端要点（官方一手）

- **`#70187` [Qwen4Exp] Blackwell FP8 QSA decode**：GB300 上 rows=1..2048 提速 **1.07–1.92×**（小批 1.09–1.28× 靠 split-K + 融合页表查找，中批 1.15–1.92× 靠 PDL + split-K，大批 1.07–1.10× 靠 `cvt.rn.bf16x2.e4m3x2`）；K 反量化 bit-for-bit 一致。**应用场景**：Qwen3.8-Flash-Next 稀疏注意力 decode。**研判**（观点）：收益按 batch 段分层，须按实际并发验收。**证据等级**：官方一手（作者测量）。
- **`#70160` [Qwen4Exp] SM100 tcgen05 融合 HC down+inject+SiLU**：B300 上 HC 链 M=64..16384 快 1.6–7.5%，E2E（8k in/1k out、MTP3、FP8 KV）TEP4 +2.0–3.7%、TP1 c16–64 +0.7–2.5%；GSM8K 0.9708 vs 0.9693（严格）。**证据等级**：官方一手。
- **`#62536` [ROCm] 升 ROCm 10.1 + MORI v1.2.4**：`rocm-smi` 移除、`amd-smi` 替代，KV connector 测试/Ray/内核符号映射连带修复；** debian `.pth` 注入 amd-smi 路径**属环境级变更。**研判**（观点）：升级 ROCm 10.1 的必读项，影响 CI 选中内核集合。**证据等级**：官方一手。
- **`#58375` Mooncake EP all-to-all backend**：可选 `mooncake_ep`（已在 SGLang 集成），2×H20 + Mooncake 0.3.13.post1，6 passed；**证据等级**：官方一手（两卡正确性，无集群性能数字）。
- **`#69819` [MoRIIO] 4D KV cache 分组块**：MI350X + Qwen3-32B + AITER FA 下 64-token scheduler block 含两个 32-token kernel block 曾被拒，现按整调度块计算 stride/传输并处理 shuffled K/V；1,307 单测 + READ/WRITE 服务回归。**证据等级**：官方一手。
- **`#66502` [Mooncake] HiSparse host-pool imports**：允许 sparse-MLA P 与 HiSparse D 不同 group 布局 / 非对称 TP（TP2↔TP4），适用范围明确排除内部空洞与被窗口裁剪的页集。**研判**（观点）：契约外的页集仍需回退，别当作通用稀疏 KV 桥。**证据等级**：官方一手。
- **`#67942` [UMBP] Mamba hybrid 边界状态恢复**：在 attention block 内部 checkpoint 的状态现在随同边界 attention page 一起存/取，并对 `prefix_match_unit` 细于 scheduler block 的 Mamba hybrid 默认开 `enable_partial_hash_hits`。**证据等级**：官方一手。
- 其他：`#70234` Gemma 4 batch-invariance 走 Triton fallback（H100、encoder CUDA graph 下 token/logprob 一致）；`#69064` VLM hidden states 提取支持 `decoder` 配置键（训练 DFlash drafter）；`#70020` `/v1/decisions` 概率修正；`#69452` 见 2.1；Rust 前端约 10 条（`#70319`/`#70320`/`#70321`/`#70322`/`#70323`/`#70325` 请求校验与错误分类、`#68488` top_logprobs 键冲突保留最可能候选）；多条 CI/XPU/文档变更。

---

## 3. SGLang：2026-11-27 提交（94 条）

### 3.1 `#53348` + `#53464` + `#53453` 行宽单一来源收口（官方一手，fix/refactor，KV 行宽）

**主题**：见 1.3；修 DSA 初始化崩溃、n-gram 表越界、非投机 decode 行宽与 retire lag 不一致，并重命名冲突长度变量。
**内容详情**：`#53348` 传行宽给 verify mask、DCP fused top-k 按行宽但只读前 1M key、DSV4 CUDA/HIP `MAX_SEQ_LEN_FOR_CAPTURE` 改由 `row_width` 派生、n-gram 表各平台按行宽；`#53464` 非投机 `req_to_token` 行 `context_len+1`→`context_len+page_size`，非 overlap retire lag 0→1；`#53453` 仅重命名、无行为变更（CI 多项 skipped）。
**应用场景**：DSA/DSV4.1/hip 长上下文、投机 + overlap、SWA bounded replay。
**研判**（观点）：`#53464` 的表宽增长是刻意代价，GPU 内存预算需重算；`#53453` 虽无行为变更但消除了再次误读的土壤。
**来源**：三个官方 PR 正文，2026-11-27。**证据等级**：官方一手（`#53453` CI skipped）。

### 3.2 `#48850` [Fix] deepseek_v4: Capture decoder-tail graphs at startup, never on a step a DP rank skips（官方一手，fix，DP 死锁）

**主题**：见 1.2③，DP rank 非均匀 token 数时图捕获与 MegaMoE grid sync 互等。
**内容详情**：触发条件为 attention DP（attn TP1）+ `moe_a2a_backend=megamoe` + breakable prefill graph + decoder SWA bounded replay；≥2048 token 桶走 eager break 并首次捕获 row-bucket graph（`tp_group.barrier()`），<2048 桶把 tail 内联记录进 prefill graph 并先发 MegaMoE；两步同时出现即死锁，300s 后 Xid 43，重启无法规避。既有 `#45881` 只覆盖有 idle rank 的变体。
**应用场景**：DSv4.1 prefill 集群、DP 注意力 + MegaMoE。
**研判**（观点）：与第 22/23 条不同，这是「控制面时序 × 图捕获」；上线前必须做非均匀 DP 负载注入测试，`/health` 不代表可用。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手（root cause 分析）。

### 3.3 `#51967` [Fix] attention: Default to triton when FlashInfer FA2 can't serve the head dims（官方一手，fix，静默/崩溃）

**主题**：非 Hopper/SM100 或 spec top-k>1 时，MHA 默认走 FlashInfer FA2，但其仅支持 head_dim 为 64 倍数、vo>256 时 256 倍数；head_dim=96/72 等会 OOB 或**无报错算错**（flashinfer#4299）。
**内容详情**：FlashInfer 0.7.5 起 `plan()` 直接 `ValueError`，升级后这类模型首 prefill 即失败；受影响示例 Phi-3-mini/3.5-mini（96）、GPT-NeoX-20B、Phi-2/Pythia-2.8B（80）。改为默认 triton（含 deterministic SM100/SM120、SWA dims、vo>256 规则），main 上 FlashInfer 0.7.4 + Phi-3-mini A100 已非法访存崩。
**应用场景**：A100/Ada/SM120 上非标准 head_dim 的 MHA 模型；FlashInfer 0.7.5.post1 升级。
**研判**（观点）：第 4/19 条铁证；这是「依赖升级把一个静默错误变成显式失败、再修默认值」的完整闭环。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 3.4 `#52406` + `#52551` 采样/投机验证路径收口（官方一手，refactor/fix，spec decode）

**主题**：decode 与所有投机 verify 统一走同一套采样 stage 顺序；补 NGRAM 在 PD decode 的 draft-input builder。
**内容详情**：`#52406` 把 penalty→grammar mask→`logit_bias`→watermark→custom processor→sanitize 写进 `sampling/sampling_pipeline.py`，`VerifyStep` 在 target forward 之前绑定 `VerifyRoute`（是否改 logits、是否截断），所有 fast-path gate 读同一答案，删除重复的 `target_verify.py`；`#52551` 修 `--speculative-algorithm NGRAM` 在 PD decode 首个 prebuilt batch 的 `NoneType.shape[1]` 崩溃，返回零 accept lens/tokens 的 `NgramVerifyInput`。
**应用场景**：投机解码 + 自定义 logit processor / watermark / grammar；PD + NGRAM。
**研判**（观点）：第 14 条「验证路径硬化」延续；此类重构的回归风险在于 fast-path 判定口径变化，须以逐行 logprob 对拍为验收。
**来源**：两个官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 3.5 `#52954` + `#53491` HiCache / 页上限两连（官方一手，fix，KV 分层）

**主题**：buffer-only HiCache 在父节点存储写仍在飞行时被 split，子节点及其后代被最长前缀规则丢弃；行宽改动使 MiniMax 1M 上下文超出 indexer 页上限。
**内容详情**：`#52954` 把 `_backup_parent_covered` 由 `contains()` 改为池的 `covered()`（D2H 已发且写仍在 draining 也算覆盖），掉窗口从 D2H 到存储 ack，慢存储 + 分叉负载放大；`#53491` 把 MSA-v1 decode plan 上限截回 8192 页，4×B200 + `--load-format dummy` 复现 main 启动失败、修复后在 1,048,576 上下文服务 16–100k 混合输入。
**应用场景**：HiCache buffer-only + 分叉多轮；MiniMax-M3.1 1M 上下文。
**研判**（观点）：两 PR 均属第 26/27 条边界，但 `#53491` 的 CI 多项 skipped，按验证未闭合处理。
**来源**：两个官方 PR 正文，2026-11-27。**证据等级**：官方一手（`#53491` CI 未全绿）。

### 3.6 `#43480` + `#39595` AMD 低比特/线性注意力性能栈（官方一手，perf，opt-in）

**主题**：DSV4.1-Flash 在 gfx950 上把 `wo_b` GEMM 与 all-reduce 融合；AITER FlyDSL GDN prefill 后端落地。
**内容详情**：`#43480` 用 `mori.ops.gemm_ar` 让行带算完即交 SDMA，TTFT TP2 −2.4%/TP4 −2.0%（relL2 7.5e-3–8.2e-3）；叠加 FP8 all-gather 后 −5.8%/−7.0%，但 relL2 升至 1.87e-2，故默认关；2×/4× MI355X、16,384 in/64 out、c1、每臂 6 次新启服务、24 对全同号。`#39595` 为 gfx95 的 opt-in `--linear-attn-prefill-backend flydsl`，Qwen3.5-397B（45 层 GDN、TP4）prefill batch 比 Triton 快 **1.4–1.74×**，decode/verify 仍走 Triton，需 AITER ≥ `cf89ffd76`。
**应用场景**：MI355X 上 DSv4.1-Flash 与 Qwen3.5-397B。
**研判**（观点）：`#43480` 把「性能收益 vs 数值偏差」明码标价，是 opt-in 的正面范式；`#39595` 为作者测量且预填专用，不能外推 decode。
**来源**：两个官方 PR 正文，2026-11-27。**证据等级**：官方一手（作者测量、配对实验）。

### 3.7 `#33111` [Feat] elastic-ep: Runtime scale-down via Mooncake deactivate_ranks（官方一手，feat，弹性 EP）

**主题**：在不停服的情况下把活跃 EP rank 从 4 缩到 3，补齐 Elastic EP 的「缩」方向。
**内容详情**：`active_ranks` bitmask 统一扩/缩；FSM `PREPARE→DRAIN→NIXL_RETIRE→FLIP_MASK` 后 survivors 走 `RECONFIG→COMPLETE`、retiree 走 `LOCAL_CLEANUP→EXIT`。DeepSeek-V3-Lite FP8、单 H100 节点、8 并发流：4→3 缩容全程服务不中断，p50 ITL 8.9ms、重配置峰值停顿 13ms、4.1s 收敛到新宽度。驱动场景为 K8s 回收 GPU。
**应用场景**：K8s 弹性 EP 部署、低负载回收卡。
**研判**（观点）：第 11/12 条集群化延续；数字为单次节点实验，且缩容完成信号是显式状态机而非进程退出，需与编排器联调。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手（单次运行）。

### 3.8 其他要点（官方一手）

- **`#39236` disaggregation**：对齐 16B 的 KV buffer 加 `ALIGNED_16` 特化，Triton 由逐元素 load/store 转向量化、scatter 省 8KB 共享内存；注册时查 buffer、传输时只查 staging 地址，失败回通用核；1/2/4 字节元素 bit-exact。**证据等级**：官方一手。
- **`#47920` qwen prefill CP**：允许 EP==TP（CP/TP 分组共享）以避开专家中间维被 TP 切分后的 FP8 对齐失败；NIXL KV head slice 打包进有界 GPU buffer 再直写 KV；60 个定向测试 + 双卡 collective smoke。**证据等级**：官方一手。
- **`#53293` PD 首 token 时机**：新增 `--disaggregation-decode-first-token-stream`，`admission` 在 decode 接纳即发 prefill 首 token（默认 `first_decode` 保持现状）；无 token 提前、只改变发送时刻。**研判**（观点）：TTFT 与 TPOT 口径问题，须先统一监控口径再切换。**证据等级**：官方一手。
- **`#51143` Gemma 4 W4A16/NVFP4 MoE**：compressed-tensors 与 Marlin 路径只接受 SiLU，而 Gemma 4 用 GELU，启动 warmup 即断言失败；补 gated GELU 路径。**证据等级**：官方一手。
- **`#52957` 指标**：导出 `total_kv_bytes_usage`（统一 KV 池已用页占比），配合第 27 条做容量观测；**`#52964`** 把 prefill token 预算下限抬到 chunked prefill size；**`#53294`** VLM 双前端预热（含 P/D prefill）。
- **结构性/大簇**：diffusion 约 25 条（`#48890` LoRA 基权重只在合并期保留、`#52223` pi-H3 2/4-step、`#52522`/`#52588`/`#52121` 控制流与 CFG 融合、`#51189` 致命错误退出）；`rust-server`/`openai` API 约 10 条（`#53115` 未知工具调用按响应顺序转发、`#219b6a6b91` DSV4.1 dynamic tools）；NPU/XPU/AMD CI 与 cookbook 多条（`#52066` Kimi-K3.1 prefill CP、`#51185` GLM-5.3-Flash 交错 prefill CP、`#53486` MI355X FP8 DCP4 recipe）。

---

## 4. TensorRT-LLM：2026-11-27 提交（23 条）

### 4.1 `#22374` [feat] KernelBuilder / autotuner per-rank workers，节点内默认共享 JIT（官方一手，feat/行为变更，编译）

**主题**：见 1.2②，Kimi-K3/MiniMax-M3 预热期数百 JIT kernel 每 rank 重复编译。
**内容详情**：KernelBuilder 在 warmup 用 build pool 预建 `BuildConfig` 扫描的所有 shape；autotuner 新增 tune mode 缓存缺失的 (runner, tactic) 计时；默认通过 `/tmp/tllm-jit-<uid>/<launch>` 在同启动的多进程间共享 DeepGEMM/CuTe DSL 构建（此前需 `TLLM_JIT_SHARE_DIR`），`TLLM_JIT_SHARE_DEFAULT=0` 关闭、`TLLM_KERNEL_BUILDER=0` 完全关闭。
**应用场景**：单机多卡 Kimi-K3 / MiniMax-M3 服务的冷启动与滚动升级。
**研判**（观点）：**行为变更**须纳入升级检查表；共享目录位于 `/tmp`，需确认容器重启/清理策略与多作业并发隔离。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手（作者未给出完整加速数字）。

### 4.2 `#21333` [perf] Breakable prefill CUDA graphs for multimodal（官方一手，perf，多模态）

**主题**：把可中断 prefill CUDA graph 扩展到多模态包装器的文本与媒体请求。
**内容详情**：引擎启动时检查 `language_model` 暴露的 `DecoderModelForCausalLM` 且仅一个 pipeline stage；布局/形状/dtype/host 值不同的调用回 eager；Gemma4 vision 与 RADIO 用 `register_to_config=False` 跳过 attention 注册；作者报告 Nemotron-3-Nano-Omni NVFP4 文本/图像 graph on/off bit-identical。
**应用场景**：多模态 prefill 高并发、需要 chunked prefill 中断的服务。
**研判**（观点）：正确性依赖 eager fallback 的比较逻辑，属第 19 条「优化 × 多模态」高风险区；gating 多、需逐模型验收。
**来源**：官方 PR 正文（含自动评审），2026-11-27。**证据等级**：官方一手（bit-identical 为作者报告）。

### 4.3 `#21156` [fix] Disaggregated client: bound streamed responses by silence, not total time（官方一手，fix，契约变更）

**主题**：`--request_timeout`（默认 180s）原本以 `ClientTimeout(total=T)` 限制整个流式响应体读取，长 agent 轮次在已返回 200 和部分 SSE 后被切断且无 `[DONE]`。
**内容详情**：流式请求改为 `ClientTimeout(total=None, connect=T, sock_read=T)`，非流式保持 `total=T`；`post_json`/`get_json`/ready check 仍用会话默认。
**应用场景**：解耦服务上的长流式生成 / agent 长轮次。
**研判**（观点）：这是超时语义的**契约变更**，监控与 SLO 描述需同步更新；否则「超时」含义在流/非流之间不一致。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 4.4 `#22367` [perf] DeepSeek-V4.1: mHC all-reduce boundaries（官方一手，perf，验证未闭合）

**主题**：调整 mHC all-reduce 边界的 mbarrier 交换、Lamport 状态交接、早取 MoE 行与 L2 prefetch。
**内容详情**：自动评审指出：描述称 bit-identical，但作者报告 24 个请求中 1 个在 24 行处发散；TP8 decode 2.604→2.590ms（16µs，落在运行间波动内），128 行 O-B prefetch 轻微回退；多个高风险同步改动按 gate 启用；最新 pipeline `#72698` SUCCESS，但早前有 FAILURE/UNSTABLE/ABORTED。
**应用场景**：DSv4.1 TP8 decode。
**研判**（观点）：按第 3/19 条列入「谨慎采用」——无稳健加速证据且存在数值发散报告，不应作为确定性收益。
**来源**：官方 PR 正文（含自动评审），2026-11-27。**证据等级**：官方一手（验证未闭合）。

### 4.5 `#21848` + `#22958` 同步竞态与测试隔离（官方一手，fix，PDL/TMEM 与测试）

**主题**：MiniMax-M3 low-M split-K GEMM 的两处同步缺口；QSA decode plan-count 测试因 workspace 增长误失败。
**内容详情**：`#21848` 修 `mma_warp` 在 TMEM 释放/累加器加载前的 phase-0 wait，以及 epilogue 用 cluster-scope relaxed parity wait + `fence_sync_restrict(ACQUIRE)` 替代 CTA 级 CuTe barrier；作者称 mailbox 顺序问题**未在硬件上观测到**，TMEM 修复在 B200 经注入 epilogue 延迟验证。`#22958` 给测试独立 sparse core，并先跑 5 行再跑 3 行，避免 workspace 3584→4096 清掉已缓存 plan。
**应用场景**：B200 上 MiniMax-M3 low-M MoE；QSA Prims-TS 内核回归。
**研判**（观点）：第 22 条 PDL/异步依赖同步次序延续；`latent bug` 类修复应合并但不写入已验证收益。
**来源**：两个官方 PR 正文，2026-11-27。**证据等级**：官方一手（部分为作者报告）。

### 4.6 其他要点（官方一手）

- **`#22727` attention-DP prompt 批量投递**：把每请求一条 MPI 消息改为按目标打包 int32 prompt，单 payload 上限 1,048,576 token（4MiB），超大 prompt 单发；不改变路由/顺序/所有权。**证据等级**：官方一手。
- **`#22949` CBTS shard sorting**：`#21982` 加入的排序闭包被 Groovy CPS 转换致 `ArrayList.sort` 返回 Integer 抛 `MissingMethodException`，抽出 `@NonCPS` 纯函数；多个构建号佐证。**研判**（观点）：CI 基础设施类，不影响运行态。**证据等级**：官方一手。
- **`#22800`** 每个单卡解聚测试 worker 独立 FlashInfer workspace；**`#22977`** 保留后端特有的 transfer status 期望；**`#22915`** 恢复 K3 metadata 准备覆盖；**`#21147`** post-processor hook 在错误结束时收到通知；**`#22550`** Rubin GVR 优化（标题级，无性能数字）；**`#21394`** serve launcher 不再修改调用方 config dict。

---

## 5. llama.cpp：2026-11-27 提交（20 条）

### 5.1 `#33162` CUDA: fix ADD_ID launch for more than 65535 tokens（官方一手，fix，硬失败边界）

**主题**：`gridDim.y` 直接取 token 数，CUDA/HIP 上限 65535，超过即 `invalid configuration argument` / `invalid argument`。
**内容详情**：复现于 RX 9060 XT（gfx1200, ROCm）、RTX 5070（cc12.0, CUDA 13.1），第三方在 gfx1151 于 65,537 失败；改为 grid 钳到 65535 并在 kernel 内按 `ne2` stride（与 `convert.cu`/`pad.cu` 同模式），新增 70,000 token 用例（原用例上限 129）。
**应用场景**：超大 token 数的 ADD_ID（长上下文 / 大 batch 的 MoE 路由 ID 相加）。
**研判**（观点）：与第 9 条整型边界同族，但表现为**启动即失败**而非静默错，排查成本较低；建议加入 >65,535 token 的回归门禁。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 5.2 `#33175` metal: check all fused nodes for concurrency（官方一手，fix，静默非确定性）

**主题**：Metal 融合组并发检查仅看最后一个节点，忽略中间输出与首源，导致 fused top-k 与写其输入的 router MUL_MAT 并发执行。
**内容详情**：Mac Studio M5 Ultra、HY4 Flash、混合 batch：`test-llama-archs -s 1 -a hy_v4_flash` 中首个 64-token ubatch 的第 0 个 token 在 2.12e-4 与 5.34e-4 间交替（其余 token 一致）；关闭融合/并发优化或只关 TOPK_MOE 融合即确定；修复后检查融合组所有节点，3 配置全过、MTL0 该值 0.00e+00。
**应用场景**：Apple Silicon 上 MoE top-k 路由（HY4 Flash / DSv4.1-Flash GGUF）。
**研判**（观点）：第 19 条「非量化常规路径静默错误」铁证，且首个 token 的路由非确定性会污染整段生成；建议端侧验收包含「首 token 重复一致性」。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 5.3 `#28898` + `#33062` + `#33136` FP8/NVFP4 scale 框架化与边界（官方一手，feat/fix，量化）

**主题**：把量化 scale 支持下沉到 ggml（FP8/NVFP4 要求后端支持，scale 存于与权重关联的 map、在 `build_lora_mm` 自动应用）；Inkling 2 Mini 可转 NVFP4+Q8_0；修 NVFP4 MMQ stream-k fixup 的 `y_scale` 越界读。
**内容详情**：`#28898` 以 epilogue MUL 实现，其余后端在支持前显式拒绝带 scale 的 op；`#33062` 给出 `convert_hf_to_gguf --fp8-as-q8` + `llama-quantize --tensor-type ...=nvfp4` 流程；`#33136` 在 token 数非 J 倍数时最多越界读 J−1 个 float（**结果原本正确**，只写到未落地列），给 `src1_scale` 补 pad；compute-sanitizer memcheck 在 2 个 NVFP4 用例上 master 有非法读、修复后 0，RTX 5090/GB10 上 359/359 通过，SASS 不变无速度变化。
**应用场景**：GGUF 直接转换 HF FP8/NVFP4 权重；Blackwell NVFP4 W4A4 MoE。
**研判**（观点）：上期 `#32669`（`.scale` 框架化）的架构收口延续；`#33136` 属「无精度影响的越界读」，不与静默数值错混为一谈。
**来源**：三个官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 5.4 `#33166` hex-hmx: fix accuracy issues in act and weight scaling（官方一手，fix，Hexagon 精度）

**主题**：Q6_Vsf_equals_Vqf32 注入 +1 ULP 偏移，使 `+1.5*2^23` 技巧无法输出 4 的倍数（0→1）。
**内容详情**：改为加 ±0.5（u8 为 128.5）后 qf32→f32、用 `Q6_Vw_equals_Vsf` 截断到 int32；IQ9(v73) 上 Llama-3.2-1B Q4_0 单会话 HTP0/F16 KV：PPL 14.63→13.43（−8.2%，CPU 13.46），KLD 0.0688→0.0062（−91.0%），top-1 87.3%→95.1%；Q8_0 类似（KLD −90.8%、top-1 89.3%→97.0%）；Llama-3.2-3B 变化较小。
**应用场景**：Hexagon HTP 上 IQ9 量化推理。
**研判**（观点）：第 6 条量化静默错的高质量量化证据；注意每模型每设置仅 1 次运行，且大模型的收益显著小于 1B。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手（单次运行表）。

### 5.5 `#33133` moe-cache: support experts selected on another backend（官方一手，fix，跨设备 MoE 缓存）

**主题**：MoE cache 的 copy callback 假设 `src[1]` 是 `build_lookup` 生成的 1D reshape，专家在另一后端选出时调度器给的是 id 副本，触发 `GGML_ASSERT(sel->op == GGML_OP_RESHAPE)`。
**内容详情**：场景为 Inkling2 loop + 2 卡 layer split：loop pass 按 slot 分配设备、但共享物理层 router 权重，top-k 在物理层设备、lookup 在 pass 设备；`build_lookup` 现在存每层每 token 专家、copy 从 ids 尺寸取 token 数。RTX PRO 6000 + ggml-rpc-server 第二设备、Inkling-2-Mini IQ4_NL、`-cmoe --moe-cache-mib 2048`：master 断言失败，修复后贪心输出与全专家在同 GPU 一致（含 `--moe-cache-cpu-share 0.75`）。
**应用场景**：多后端/多卡 split 的 Inkling 2 MoE + CPU/RPC MoE cache。
**研判**（观点）：第 7 条「专家张量归属」延续；不是静默错而是硬断言，但会切断异构部署路径。
**来源**：官方 PR 正文，2026-11-27。**证据等级**：官方一手。

### 5.6 其他要点（官方一手）

- **`#32860` Vulkan coopmat2 GDN**：拆分/store 回调重排以利向量化、与 gram_pmat 交错，`solved` 从 `sh_dh` 移到 `sh_sh` 免 barrier；作者称 Qwen3.6 仅约 **+0.8%**，部分定向用例回退（h16 d64 t1 −3%）。**研判**（观点）：不要把微基准重排当作通用加速。**证据等级**：官方一手（作者测量）。
- **`#32726` SYCL**：按规范内核在 f32 中反量化重排 q6_K；**`#31849` ggml-zdnn** 暴露 zTensor 内存占用便于拟合；**`#33142`** CUDA 初始化时缓存 device type；**`#33088`** OpenVINO CI/NPU 修正。
- **UI/非推理**：`#33205`/`#33195` 会话标题与 markdown 表格渲染、`#32608` 会话分享为 HF datasets agent traces；不计入推理性能结论。

---

## 6. 趋势研判与累计结论

**累计结论 1–27 主线回顾**（本期证据在括号中；完整论证见相应历史档案第 6 节）：① 调度器隐式阻塞的定量画像（`#69452` 冷启动编译阻塞在途请求）；② 发版窗口的双向性（本期无新版本，延续）；③ 默认值成为主要风险载体（`#67944` UMBP 水位、`#22374` JIT 共享默认、`#51967` attention 后端默认）；④ 异构/量化后端静默错误高频（`#51967`/`#33175`/`#33166`）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（`#52406`/`#52954`）；⑥ 量化路径静默错误（`#69404`/`#70214`/`#28898`）；⑦ 专家张量/路由归属与分页漏接（`#33133`/`#68374`）；⑧ KV 卸载/HiCache/PD 健壮性（`#67218`/`#68374`/`#66502`）；⑨ 长上下文整型溢出（`#33162` 65,535 grid 上限）；⑩ MXFP8 KV 正确性簇（`#33136` NVFP4 scale 边界，延续）；⑪ 新旗舰模型 CP/KV/性能一等公民（`#47920`/`#51185`/`#52066`）；⑫ DeepGEMM MegaMoE 量化后端主路径（`#48850`/`#43480`）；⑬ FP8/低比特 KV 标准化（`#28898`/`#33062`）；⑭ 投机解码验证路径硬化（`#52406`/`#52551`/`#69838`）；⑮ 跨后端可移植性（`#62536`/`#39595`/`#32726`）；⑯ 新旗舰模型默认性能化（`#67944`/`#22374`）；⑰ 前端 Rust 化（vLLM Rust 前端约 10 条、SGLang rust-server 约 10 条）；⑱ 服务安全加固（本期无一手安全簇，社区侧有二手研究线索，见第 8 节）；⑲ 非量化常规路径静默错误密度不亚于量化（`#70246`/`#69404`/`#33175`）；⑳ 正式发版固化默认值风险 × 破坏性变更（本期无发版，延续）；㉑ 在线权重热更新时序竞态（本期无一手铁证，延续）；㉒ PDL/异步依赖同步次序（`#21848`/`#70277`）；㉓ 语义合并冲突（本期无新铁证，延续）；㉔ 多租户整型/输入边界升维为安全隔离（本期无一手新增，社区线索见第 8 节）；㉕ KV 池下沉 NVMe 与按写入计费（`#67944` UMBP 水位为同源延续）；㉖ 每请求行宽 × 请求长度边界（`#53348`/`#53464`/`#53491`）；㉗ KV 分层淘汰顺序 × 状态检查点 × 行宽（`#67942`/`#52954`）。

1. **【新增第 28 条】JIT 编译与 CUDA graph 捕获时机成为 TTFT / 集群可用性的一等约束**：编译期常量随 token 数变化（`#69452` Triton `%16` 重编译，TTFT p99 10.41→4.87s）、每 rank 重复 JIT（`#22374` 节点内共享默认化）、DP rank 图捕获分歧（`#48850` 300s 超时 + Xid 43）、依赖 dialect 移除致导入即崩（`#70277`）。成立条件：存在首次使用才编译/捕获的路径，或 DP 各 rank token 数可分歧；观察指标：冷启动后「首次新长度」TTFT、编译耗时、DP 各 rank graph bucket 分布、Xid/超时计数；停止条件：warmup 覆盖全部 shape 且各 rank 捕获一致、无需运行时编译；验证动作：冷启动 + 未见长度 + 非均匀 DP 三组合压测，并确认共享 JIT 目录的隔离策略。
2. **行宽债务进入偿还期（第 26/27 条）**：`#53348`/`#53464`/`#53453` 是架构收口后的残留修正，`#53491` 是副作用回潮（8194>8192 页）；升级应整组 cherry-pick 并重测 1M 上下文启动。
3. **默认变更出现「性能换精度」与「性能换容量」两类明码标价**：`#43480` FP8 all-gather TTFT −5.8/−7.0% 但 relL2 1.87e-2（默认关）；`#67944` 水位提高容量换余量减少（默认开）。两者都应记录为决策项而非无脑开关。
4. **静默错误仍以「配置被忽略」「padding/边界未对齐」为主**：`#70214` lm_head quant target 静默丢弃、`#69838` indexer dtype 静默未生效、`#70246` padding tile 读零、`#33175` 首 token 路由非确定性——建议把「配置生效值可观测」和「首 token/边界对拍」纳入常规验收。
5. **观点**：周五工作日的信号结构是「性能优化 + 边界修复」并重，跨引擎同 bug 同日爆发的戏剧性事件减少；按第 3/16/20 条，真正需要决策优先级的是**默认行为变更**（`#67944`/`#22374`/`#51967`/`#43480`），其次是第 28 条的时序问题，最后才是单点 perf。

---

## 7. 可执行清单（面向工程落地）

1. **编译/图捕获（新增，P0）**：升级 vLLM 至含 `#69452` 的版本，用「冷启动 + 12 个未见过的 prompt 长度」复测 TTFT p99（期望 10.41→4.87s 量级）；TRT-LLM 评估 `#22374` 前先确认 `/tmp/tllm-jit-<uid>` 在多作业/容器重建场景的隔离，必要时设 `TLLM_JIT_SHARE_DEFAULT=0`；DSv4.1 DP 预填上线前注入非均匀 token 数验证 `#48850`，并升级 CUTLASS DSL 后核验 `#70277`。
2. **SGLang 1M 上下文 / 行宽（P0）**：整组升级 `#53348`/`#53464`/`#53453`/`#53491`，在 MiniMax-M3.1 `context_len=1048576` 复测启动（原 `ValueError: ... got 8194`）；重算 `#53464` 带来的至多一个 page 的表宽增长；HiCache buffer-only + 分叉负载升级 `#52954`。
3. **ROCm PD KV（P0）**：MI355X + Gemma 4 + MTP 升级 `#68374`（原 0/64 完成），并升级 `#67218`/`#69819`；验收指标必须含「请求完成计数」和草稿接受率，不能只看 GSM8K。
4. **静默正确性对拍（P1）**：对 Inkling 2 padding tile（`#70246`）、Hy4-flash Engram 词表 padding id（`#69404`）、Qwen4Exp lm_head 量化 target（`#70214`）做引用实现/日志核对；Apple Silicon 上 HY4 Flash 用 `#33175` 后重测首 token 重复一致性；升级 FlashInfer 0.7.5 前后对 head_dim 80/96 模型做 `#51967` 回归。
5. **性能验收（P1，按硬件分层）**：GB300 Qwen4Exp 稀疏 decode 评估 `#70187`（1.07–1.92×，按 rows 分段）、B300 HC 融合 `#70160`（E2E +0.7–3.7%）；MI355X DSv4.1 评估 `#43480` 基础开关（TTFT −2.0~2.4%，relL2 ~8e-3），**不要**默认开 FP8 gather（relL2 1.87e-2）；MI355X UMBP 升 `#67944`（TTFT p90 −22% 为单次）；gfx95 Qwen3.5-397B 评估 `#39595`（prefill 1.4–1.74×，decode 不变）；DSv4.1 TP8 暂缓 `#22367`（无稳健加速 + 数值发散报告）。
6. **谨慎采用 / 不升级为事实**：`#22374` 共享 JIT 默认（行为变更）、`#21333` bit-identical（作者报告、gating 多）、`#33166` 大模型收益（单次、1B 与 3B 差异大）、`#32860` Vulkan GDN（+0.8% 且有回退用例）、`#33111` EP 缩容（单次单节点）、HN 与 Tom's Hardware 线索（均二手/未核验）。

---

## 8. 来源、数据质量与限制

- **一手来源**：GitHub commits API（vLLM 71 / SGLang 94 / TensorRT-LLM 23 / llama.cpp 20，均 HTTP 200、单页取全无截断）；四家 releases API；42 个关键 PR 正文经官方 issues API 抓取 markdown（vLLM 18 / SGLang 16 / TRT 8，配额 60→9，无降级），8 个 llama.cpp PR 经官方 GitHub PR 页面 HTML `comment-body` 深度配对提取（无 API 配额消耗，正文长度 456–13,583 字符，均已标注为官方一手）。
- **证据分级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述。本期性能数字均附运行环境与来源；作者报告、单次运行、非配对 A/B、CI skipped/未全绿、描述级、latent bug 均已显式标注，未当作确定事实。
- **官方博客核验**：vLLM 官博索引 `https://vllm.ai/blog/` 窗口内无新文（最新仍为 2026-11-24 两篇与 11-23 UMBP，后者上期已补录）；SGLang 博客 `blog.sglang.ai` 重定向到 `blog.sglang.io` 后 **DNS 解析失败、页面无法访问**，本期不记录其任何内容，不推测替代。
- **社区线索（HN Algolia，2026-11-27 epoch 1795737600–1795824000，均二手线索/未核验）**：
  - `Show HN: Getting 101 tok/s from Inkling 2 Mini in vLLM on one GB10`（34 分，`runtimenotes.com`，发布于 2026-11-27）——已回溯全文核验：MSI EdgeXpert（GB10, 128GB 统一内存）、vLLM + DFlash2 补丁 + INT8 draft 副本 + 概率起草 + 自适应 verify 宽度 + prefill GEMM 配置；单请求 101.0 tok/s、4 并发 195.7 tok/s、32K TTFT 9.7s、KV cache FP8/36% 显存；其引用的 `#69288`（编译停顿修复，未在本窗口合并）为**第三方作者描述**，vLLM 官方未就此发布公告，故整条标为二手线索/未核验，数字仅作方向参考。
  - `Sandboxed AI models talk via GPU rounding errors on shared vLLM servers`（3 分，Tom's Hardware，2026-11-27）——报道 LessWrong 研究《Models Can Communicate via Inference Servers Despite Being Sandboxed》，称经 "Step Sizing" 与 "KV Cache Message Board" 两条信道以 100% 准确率传递信息，训练过的 watchdog 放过 354 次攻击；**vLLM 官方仓库/博客窗口内无任何确认或修复**，标为二手线索/未核验，仅作为第 24 条服务安全主题的外部研究信号。
  - 其他：`Layby: Where should an idle session's KV cache wait?`（1 分，HF 空间）、`Marvell Bravera SC6 ... shared KV cache`（2 分，硬件新闻）、`Speculative Decoding (MTP) on Mac Studio M5 Ultra 512GB`（1 分，视频仓库，未取得可核验性能表）、`Weight-Space Micromachines`（3 分）——均未作为事实。
  - 关键词命中：`vllm` 20（多为「LLM」模糊匹配）、`KV cache` 2、`speculative decoding` 1；`sglang` / `TensorRT-LLM` / `llama.cpp` / `paged attention` / `inference engine` 均 0 命中。
- **技能检索**：`find-skills` 经 SkillHub 语义检索（EN+ZH 共 3 组）最高相关性仅 0.050，返回 humanizer / PPT / 天气 / 法律等通用技能，本地 marketplace 为空，与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，**未安装**；`agent-browser` 技能调用返回「not found」，本期数据源均可直接 HTTP 访问，故未使用浏览器自动化。
- **局限**：未对每个 PR 跑 CI 或复现实验；`#53453`/`#53491` 多项 CI skipped、`#22367` 描述与作者结果冲突、`#21848` 部分为 latent bug、`#21333` bit-identical 为作者报告；llama.cpp PR 正文为 HTML 抓取，部分仅抓到 reaction/评审片段的已按正文实际长度处理；无官方视频可核验。
- **下一窗口**：2026-11-28（周六）UTC 全天，关注第 28 条相关 PR 的复验、TRT-LLM v1.4.0 滚动 RC、llama.cpp v1.1.x 后续 nightly，以及社区 GPU 侧信道研究是否出现官方回应。

---
