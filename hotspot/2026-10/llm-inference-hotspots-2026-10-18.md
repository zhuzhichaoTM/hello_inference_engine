# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-18（北京时间）
- **关注窗口：** 2026-10-17 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-17 08:00 – 2026-10-18 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-17 UTC 窗口，分别返回 **48 / 47 / 11 / 12** 条，均 HTTP 200）、四家官方 releases API，以及 57 个 PR 页面正文（含上一期欠账补取 5 个）。**本期改进：** PR 正文改由 `github.com/<repo>/pull/<n>` 页面直接抓取，不消耗 GitHub API 配额，因此**四节均含 PR 正文级细节，不再出现上一期「仅基于提交标题」的降级**；上一期列出的 5 项欠账（TensorRT-LLM `#18599`/`#19981`/`#20290`、llama.cpp `#30613`/`#30449`）已在本文第 4、5 节补齐。**窗口特征：** 2026-10-17 为**周六**，TensorRT-LLM（11 条）与 llama.cpp（12 条）提交量明显低于工作日，属正常节律，**不构成「活动减少」的结论**。HN Algolia 公开 API 本期可访问（HTTP 200），但与推理引擎主题直接相关的讨论几乎为零。Reddit 未纳入采集范围；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** 本期四家**均无新的正式版本发布**。vLLM 最新公开 GitHub Release 仍为 **v0.31.0（2026-10-05）**，但 `#61989` 正文提及 **v0.32.0rc2 镜像**与「v0.32.0 的 3–5% 解码回退」，说明 **v0.32.0 已在镜像层面进入 RC**，只是尚未发布 GitHub Release（此项以 PR 正文为据，非发布页确认）；SGLang 仍为 **v0.5.21（2026-10-02）**；TensorRT-LLM 仍为 **v1.4.0rc0（prerelease，2026-10-14）**；llama.cpp 正式版仍为 **v0.7.0**，窗口内滚动 nightly 共 **8 个标签（b11709–b11716）**，最新 b11716 发布于 2026-10-17T21:11:34Z。

---

## 1. 补录：vLLM 官方博客《Hyper Optimizing Kimi K3 in vLLM》（2026-10-16 发布，前一期未收录）

**主题**  vLLM 官方博客发布《Hyper Optimizing Kimi K3 in vLLM: Moving Fast and Responsibly》，作者 **Benjamin Chislett（NVIDIA）**，12 min read。它正是前一期档案中 `#61867`（Kimi K3 GB300 专用解码）的**配套说明文**：前半部分讲清 23 kernel → 13 kernel 的融合路径与四项优化，后半部分讲**为什么要先建流程再收代码**——提出「fast-track 模型优化 RFC」与四条原则，用于承接**由 coding agent 生成的推理内核**。

**内容详情**  官方博客原文要点：
- **单层时间线：** stock vLLM 每层 **130 微秒 / 23 个 kernel**，专用解码路径 **69 微秒 / 13 个 kernel**。
- **覆盖范围：** NVIDIA Blackwell、8 或 16 GPU、decode batch **1–64**、可选 decode context parallelism（DCP）处理长上下文；GPU 处于同一 NVLink 域且支持 multicast。
- **四项优化：**
  1. **按负载调形状**——attention 输入投影在 batch 1–2 用**专用小矩阵 kernel**，更大 batch 用 **split-K**；KDA 使用专用 CuTe DSL decode kernel，按 batch 与每 GPU head 数调调度；长上下文下 DCP 把 KV cache 切分到各 GPU，本地算注意力后由**融合 kernel** 结合 v-proj 与 gating。
  2. **计算通信重叠**——把 attention 输出投影**直接融进 all-reduce 与随后的 attention-residual**；kernel 把部分结果发布到 NVLink 可访问缓冲，接收方合并后继续；部分小 batch 配置把生产者与消费者融进**同一个 kernel**，数据交换采用 **fence-less** 方式；更大 batch 切换为**两阶段归约**（先合并各自分配的输出部分，再分发结果）以减少跨 GPU 数据移动。
  3. **MoE 调度与重叠**——基线在**每个 GPU 上都算完整 router 分数与 latent 输入投影**；改为按 TP rank 切分这些计算，再把各 rank 份额合并为一次矩阵乘，结果经 NVLink 交换；shared expert 跑在**独立 stream**，与 routing 或 routed-expert 计算重叠，调度器控制各操作占用的 GPU 资源以避免争抢。
  4. **权重预取**——小 batch 常受限于**权重从内存加载的速度**；在通信等无法吃满带宽的阶段，用**旁路 stream 上的专用 kernel** 预热下一操作的权重，使通信结束后命中率显著提高。
- **性能（官方表格，均为 TP 组总吞吐，不含 prefill，无投机解码）：**
  - **有投机（DSpark，DL7，合成 AL=6，ISL=1/OSL=1k，batch 1，TP16）：842 tok/s vs stock 488 tok/s，1.73×**；博客摘要另称「batch 1 上超过 840 tok/s」。
  - **无投机：** TP16 在 batch 16–64 保持 **1.62–1.71×**，TP8 同区间 **1.31–1.43×**。逐点：TP8 B1 **193.3（1.58×）**、B8 **820.1（1.45×）**、B64 **2,375.9（1.31×）**；TP16 B1 **256.4（1.88×）**、B8 **1,157.9（1.79×）**、B64 **3,606.1（1.62×）**。
  - **AgentX：** 并发 1 时 P90 交互性 **403.2 vs 223.7 tokens/s/user，1.8×**。**口径注意（原文自述）：该对比是「优化实现在 16 张 GB300」对「较早的 vLLM 结果在 8 张 GB300」——即两边 GPU 数不同**。
- **流程部分（本期最值得记录的软性变化）：** 博客明说「**agent 优化的推理代码正在跑在我们的 review 与维护流程前面**」，逐行读 kernel 的人工评审无法持续。四条原则为：(1) **隔离改动**——模型专属优化留在模型代码内，且运行时也要可追溯（何时进入优化路径、改了哪些状态、如何回退）；**明确排除**输入准备、采样、KV cache 管理三类（这些是多数部署共享组件，为单一负载改动会破坏其他稳定配置，应由核心维护者介入）；(2) **限定范围但保留用户必需特性**；(3) **系统性验证正确性**——在所有支持配置上「严格更好」；(4) **扩展验证**——贡献者自带 GPU 跑标准评测并提交可复现结果。
- **下一步（官方明示）：** `#61868` 按同一模式做 **DeepSeek-V4.1-Flash**（同一个 opt-in `decode_backend: latency_specialized` 开关，支持最多 64 并发、prefill 与混合 batch 无缝回退、投机解码；融合 all-reduce + mHC、用 CuTe DSL 替换大 batch MoE/GEMM、indexer 去重分片、把每步 CPU 上的 eager 工作录成 CUDA graph）；**Muse Spark Open** 在发布两天内已超 **950 tokens/s/user（K=7，AL=6）**；并将发布 **GLM5.3 低并发 decode 延迟**的同类机制。

**应用场景**  小 batch 低延迟解码的 kernel 融合路径（23→13 kernel）、MoE 的「按 rank 切分 router/latent」以消除跨卡重复计算、all-reduce 与 attention-residual 的融合与两阶段归约选择、通信窗口内的权重预取、agent 生成内核的工程治理（隔离范围 + 可复现验证）。

**研判**  **事实：** 博客与 `#61867` 同源，给出了前一期缺失的**完整 batch 扫描表**与**四项优化的机制描述**；同时公开了「fast-track 模型优化 RFC」这一流程性提案。**观点：** 三点——(1) **本期最有迁移价值的不是 1.73×，而是「按 rank 切分 router 与 latent 投影」这一条**：它与前一期 MI355X 博客的 `#51253`（MoE 输出坐标划分）是**同一根因的两次独立复现**——基线在每张 GPU 上重复计算同一份投影，再靠集合通信对齐。两个不同厂商、不同引擎、不同硬件（GB300 / MI355X）在同一个月内分别修掉它，说明这是**TP 部署的通用浪费**，应直接进自查清单；(2) **1.8× P90 交互性必须打折看**：官方自己写明是 16 卡 vs 8 卡，虽以「每 GPU」口径归一，但并发 1 时的单请求交互性主要由**单卡 kernel 时间**决定，卡数差异会影响 DCP/TP 划分从而影响单步延迟，**不能当作纯 kernel 收益**；(3) 真正的信号在第二部分：**当优化代码由 agent 产出时，瓶颈从「写 kernel」转移到「验证与维护」**，而官方选择的解法是**收缩改动半径**（禁止碰 KV cache 管理/采样/输入准备）而非加强评审——这对使用 vLLM 的团队有直接含义：**未来一段时间的性能改动会更多出现在模型目录内，而非 core**，core 的稳定性反而更有保障。

**来源**
- vLLM 官方博客：https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel （2026-10-16 发布，作者 Benjamin Chislett / NVIDIA）
- 配套 PR：`#61867` https://github.com/vllm-project/vllm/pull/61867 ；后续 `#61868`（DeepSeek-V4.1-Flash 同款执行器）
- 同日未收录的另一篇（仅列线索，不展开）：《Serving Reflection Beam on vLLM from Day 0》https://vllm.ai/blog/2026-10-16-reflection-beam

**证据等级：** 官方一手（官方博客 + 配套 PR）；842/488 tok/s、1.31–1.92×、1.8× P90 均为官方在其自述配置下测得；**AgentX 的 1.8× 为 16 卡 vs 8 卡对比，官方已注明**，不可作纯 kernel 收益解读。

---

## 2. vLLM：10-17 提交（48 条）——MLA 上下文 prefill 的 CuTe DSL 后端、KV 传输结果收集前移、投机解码三处一致性修复

**主题**  vLLM 在 10 月 17 日的 48 条提交集中在：**MLA 上下文 prefill 新增 opt-in 的 CuTe DSL 后端**、**MRV2 把 KV 传输结果收集提前到 drafter 之前**、**投机解码三条一致性修复**（fixed-K proposer 崩溃、陈旧共享 block-table 行、EAGLE group 启发式改显式标签）、**SimpleCPUOffload 允许显式块数**（修复 PP 各 stage 池大小不一致导致的越界拷贝）、**恢复单节点 MNNVL all-reduce**（修 v0.32.0 的 3–5% 解码回退），以及 MoRIIO 对混合模型+DSpark 的 WRITE 传输支持。

**内容详情**  官方提交 API 与 PR 正文显示（节选可核验项）：
- **注意力与 MLA：**
  - **`[Attention] Add CuTe DSL backend selection for MLA context prefill`**（`90d6d871d6`，PR `#62069`）：为 **MLA 对已缓存上下文的注意力**新增 **opt-in** CuTe DSL kernel，默认分派与新 token prefill 不变。启用方式为 `--attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true,"trtllm_ragged_backend":"cute-dsl"}'`。**PR 正文未给出任何性能数字**。
  - **`[Feature] Skip MLA weight absorption when weights are pre-processed`**（`c61a1e73c6`，PR `#57320`）：在 `--load-format ipc_cache` 下，权重缓存守护进程**已把 `kv_b_proj` 吸收进 `W_UK_T` / `W_UV`** 并导出；引擎再跑一次吸收会**写进守护进程的共享分配**。改为在 `is_weights_pre_processed()` 且 `W_UV` 来自守护进程时跳过。
  - **`[Bugfix] Fix NaN output of the Mamba2 chunk scan kernel on sm90`**（`783380af39`，PR `#61718`）；**`[Bugfix][Kernel] hadacore: don't load past the input on a partial last warp`**（`08a65e287e`，PR `#61944`）；**`[Bugfix] Fix Gemma 4 encoder CUDA graph capture OOM at startup`**（`ec8b20fbf0`，PR `#61365`）。
- **KV 传输与 P/D 解耦：**
  - **`[MRV2] Collect KV transfer results before the drafter runs`**（`0c5dd18e7a`，PR `#62041`）：有 speculator 时，在**采样之后立刻**收集 KV connector 的传输结果，只把 `wait_for_save()` 与 `clear_connector_metadata()` 延后到 drafter 之后（新增 `finalize_step()`）。效果是**已完成的 load 不必等 drafter 才上报**，异步 load 也**能在 drafter 之前启动**。PR 明确**不涉及 KVConnector 接口本身**。
  - **`[Feature][MoRIIO] Support hybrid WRITE transfers with DSpark`**（`41b8c15809`，PR `#60544`）：把 MoRIIO 的 **WRITE** 路径扩展到**混合注意力 + Mamba/KDA 模型**（含 DSpark），接续 `#51052`/`#57700` 的 hybrid READ 支持。要点：按 group 传 attention KV、conv 与 SSM state 并做**分组完成记账**；解析**存活的 final prefill state**、**排除投机 scratch 槽**并发布匹配的 **N-1 token 记账**；用原生 `wait_all` 配**有限 deadline**，旧版 MoRI 走有界回退。验证：MoRIIO 全套 **274** 测试 + 3 个 block-state + 12 个相邻 scheduler 测试，共 **289 passed**。
  - **`[NIXL] Drop unreachable producer-side notif counting in push mode`**（`7879f3687b`，PR `#62210`）：push 模式下生产端分支**任何代码路径都到不了**（`<req_id>:<n>` 形式的 notif 只由 `pull_worker.py` 填充；push 模式下发往 P 的只有 `PUSH_REG` 与 `HB:`），故删除该分支及其所需的 `transfer_topo` 断言与计数清理。**26 insertions / 59 deletions**；`test_nixl_push_connector` + `test_nixl_heartbeat` **103 passed**、`test_nixl_connector` **182 passed**；作者在删除分支处埋 **tripwire**，在 GB200 上跑 TP1→TP1、TP2→TP1、TP1→TP2 端到端 push 均未触发。**一个边界行为变化：** `kv_both` 实例上同一请求既 load 又 save 时，其完成 notif 现在计为 **load** 而非 save。
  - **`[Bugfix][EPLB] Wait for pending expert buffer copies before NIXL READs`**（`9c778e9ff8`，PR `#62081`）：修复 MI300 上 `test_rearrange_expert_weights_with_redundancy[nixl-*]` 的间歇失败——冗余专家权重校验读到 `[2100.0, 2101.0, …]` 而参考为 `[1000.0, 1001.0, …]`，即 **NIXL READ 早于专家 buffer 拷贝完成**。
  - **`[Bugfix][KV Offload] Allow explicit SimpleCPUOffload block counts`**（`ce8853d27a`，PR `#61674`）：允许显式设置共享 `num_cpu_blocks`，使**不均匀 pipeline stage 使用同一大小的 offload 池**。端到端 A/B（Kimi K3，两节点各 8×B200，TP8/DCP8/PP2，fp8 KV，TOKENSPEED MLA，MRV2，RDMA，每 rank 128 MiB CPU 预算）：自动定尺寸给出 **25/25/23**（PP0/PP1 各 stage 不一致），显式设为 23 后统一为 **23/23/23**；**越界 native copy 调用从 40 次（覆盖全部 8 个 PP1 rank）降到 0（共 1,088 次拷贝）**；GPU 前缀缓存重置后的强制 CPU 重载，基线未请求，修复后 **16 个 rank 全部重载且输出 16 token 完全一致**。PR 明确：该结果**只确立定尺寸/拷贝缺陷**，不把历史崩溃全部归因于此，**也不声明吞吐或精度提升**。
- **投机解码（本期三条，均属「静默错误/崩溃」类）：**
  - **`[Bugfix][Spec Decode] Disable dynamic speculative decoding for fixed-K proposers instead of crashing`**（`763e3f7c91`，PR `#46399`）：动态投机（`num_speculative_tokens_per_batch_size`，源自 `#32374`）会按 batch 把运行时 draft 数**降到 0**，而四个早于该特性的 proposer（`ngram_proposer_gpu.py`、`extract_hidden_states.py`、`suffix_decoding.py`、`medusa.py`）**硬断言运行时数等于配置上限**，于是调度器合法地选出更小的数时，会在 `LLM.generate()` 深处抛 `AssertionError`——**且是在配置已被接受、目标模型已跑过之后**。改为在 config init 阶段对这四个方法**禁用动态投机并告警**，回落到静态 `num_speculative_tokens`（与 `#45963` 对数据并行的处理同构）。
  - **`[Bugfix][Spec Decode] Clear stale shared block-table rows`**（`61bb590877`，PR `#61174`）：**更大的 MTP decode 图在目标 batch 收缩后会读到陈旧的共享 block-table 行**，使 padding 出去的工作**寻址到另一个请求的 cache 或 state**。改为在 gather 与 dummy 准备阶段用**既有缓冲与 gather kernel** 清理先前活跃的行，保留活跃行、返回的视图与独立 PCP 输出缓冲。验证（H20，基线 `92dc3dd39332`）：无修复时新回归 **7 failed**，修复后 BlockTables/speculator/PCP 套件 **140 passed**、模型/cache/offload 兼容套件 **656 passed**；`GLM-5.3-Flash TP8/MTP3, stream=true` 在 **47/47** 请求上正确无截断（1,281–32,769 输入 token，CUDA Graph + 原生 CPU offload，CPU 恢复阶段记录 56,960 个外部 cache-hit token，MTP 接受 1,549/1,573 draft token）。PR 明确「**这是正确性验证，不是吞吐对比**」。
  - **`[Core][Spec Decode] Replace EAGLE group heuristics with explicit draft-layer tags`**（`c3de0b37e2`，PR `#61455`）：`is_eagle_group` 决定**哪些 KV cache group 承担 EAGLE 的 last-block drop**，原实现靠 `non_causal_multi_token_decode` 与「持有最后注册层的 group」**猜**，且只在部分分组路径上生效。改为 **runner 上报的逐层标签**：group 当且仅当**持有 draft 层**时被标记。修复面：DFlash/DSpark drafter 在混合模型上**不再让所有 group 都做 drop**，Mamba group 恢复前缀缓存复用（`#57032`/`#54360` 所报的「投机解码静默禁用 hybrid GDN 模型 prefix cache」）；跨多 group 的 drafter（DeepSeek-V4/V4.1 DSpark）**每个 group 都被标记**；Gemma4 MTP 不再误标无关目标 group；**标签能穿过流水并行**（此前 `PP >= 2` 时最后 stage 的 draft group 从不被标记，协调器只好「全标」）。
- **采样器（一 perf 一 bugfix）：**
  - **`[Perf][Sampler] Speed up the small-batch split top-p pipeline`**（`7b6310c109`，PR `#61996`）：小 batch split top-p（`#54651`）把 grid 上限设为**每 SM 一个 program**，该占用下 step kernel 受延迟约束——在 148 SM 的 GPU 上 **S 在 38 行时降到 2**，于是 38–64 行（如 8 请求 × 8 投机验证 token）的 top-p 调用**耗时约为 37 行时的两倍**。两项改动：**在每切片仍覆盖 ≥ 7,168 词表项的前提下继续跨 wave 切行**；**MRV2 在没有 top-p-only 行时跳过整条流水线**（此前仍启动 7 个 kernel，program 全部立即退出）。实测（B300 148 SM / H200 132 SM，词表 120K–262K 及 32K）：top-p-only 行 **B300 1.22–2.16×、H200 1.05–1.90×**；top-k 与 top-p-only 混合且 B ≥ 32、V ≥ 120K 时 **1.43–2.70×**；**top-p-only 行越少收益越大**，仅一行时在 V=151,936 达 **3.7×**、V=262,144 达 **4.4×**。
  - **`[Bugfix][Sampler] Carry the boundary tie count across blocks in the split top-p mask kernel`**（`ce49c47cbb`，PR `#62046`）：`_topp_sb_mask_kernel` 在**每个 8192 项 block 处重置**边界并列计数，导致超过一个 block 的切片**保留了过多份边界值**（新测试在未修复内核上 `kept [4160], expected [2300]`）。改为跨 block 携带计数，与单体内核一致。
- **分布式与通信：**
  - **`[Perf][Bugfix] Prefer MNNVL for single-node FlashInfer allreduce again`**（`2b85d9cb65`，PR `#61989`）：修复 **v0.32.0 的 3–5% 解码回退**，根因是 `#56891` 把自动选择改为**所有节点内进程组都用 trtllm**（此前单节点用 mnnvl，仅在无 NVSwitch multicast 的节点回退 trtllm），而该 PR **未附任何性能数据**。perf-eval nightly 显示跳变发生在 10-03 与 10-04 之间的 14 个提交里，`#56891` 是其中唯一的分布式改动，且**只有 TP>1 负载变化**：Qwen3.5-397B NVFP4 8×B200 **~5650 → ~5460 tok/s（TPOT 20.8 → 21.6 ms）**、gpt-oss-120b TP4 H200 **10883 → 10571 tok/s**、GLM-5.3-Flash TP4 B200 **~2770 → 2650–2720 tok/s**、MiniMax-M3 TP8 H200 **TPOT 35.9 → 36.8 ms**；**gpt-oss-120b TP1 不变**。A/B（v0.32.0rc2 镜像强制后端）：Qwen3.5-397B 8k/1k 并发 128，trtllm **5462 tok/s / TPOT 21.78 ms** vs mnnvl **5728 tok/s / 20.22 ms**；GLM-5.3-Flash TP4 三次重复 **2533/2753/2771** vs **2581/2811/2845**。改动：节点内组**回到 mnnvl + `allow_trtllm_fallback=True`**；多节点选择不变（Blackwell 用 mnnvl，Hopper 禁用）。
- **MoE / LoRA / 依赖：**
  - **`[Feature][LoRA][MoE] Expert LoRA for FlashInfer TRTLLM MXFP4 experts`**（`837e87e1c5`，PR `#61979`）；**`[FlashInfer] Migrate autotune warmup to flashinfer.autotune_v2 managed cache`**（`71c20f73f2`，PR `#55671`）——仅在**单 rank、单节点、PP=1、非弹性**部署下自动选托管 v2，多 rank 仍用上游旧缓存同步，PR 明确「**不声称通用死锁预防、零预热 profiling 或吞吐提升**」。
- **服务与核心正确性：**
  - **`[Bugfix][Core] Keep resumable sessions that stop while preempted`**（`f98ae6a9fd`，PR `#58778`）：异步调度下，可恢复（streaming-input）请求可能在 step 在途时被抢占，该 step 的输出作为**陈旧输出**投递，若命中停止条件请求就**停在 PREEMPTED 状态**。两个后果：**会话从所有队列消失**（`_handle_stopped_request` 重新入队，但 stopped-while-preempted 清理又把它从 waiting / kv_holding_waiting / deferred_waiting 中删掉，包括那条新记录）；**fold 删掉 prompt**（`_update_request_as_session` 保留 `all_token_ids[num_prompt_tokens:num_computed_tokens]`，抢占后 `num_computed_tokens == 0` 于是整段被删，而 `prompt_token_ids` 是另一个列表只做追加，`num_prompt_tokens` 与 `all_token_ids` 不再匹配，**上一轮输出丢失**）。改为清理后**重新入队未完成请求**，并在 fold 中保留到 `max(num_computed_tokens, num_tokens - 1)`。
  - **`[Bugfix] Enable inductor batch-invariant mode under VLLM_BATCH_INVARIANT`**（`03e8300d2c`，PR `#61118`）；**`[Model] DeepSeek-V4/V4.1 VL: stream language-model weights instead of sorting the checkpoint`**（`026f6ab268`，PR `#58857`）；**`[ROCm][Perf] Use AITER tuned configs to select a4w4 gemm kernel instead of env var`**（`2067297b2f`，PR `#61278`）；**`[Perf][Parser] Stream Qwen3 tool argument values incrementally (12 KB call: 2.1 s to 11 ms)`**（`5c1c903501`，PR `#62027`）；**`[Bugfix][Rust Frontend] Fix macOS build: read inherited listener family from its address`**（`255f14599e`，PR `#62177`）。
  - **文档与流程：** `[Docs] Add train-inference consistency agent skill`（`2d8efee9a7`，`#61551`）与紧随其后的 **`Revert`**（`237d99b4af`，`#62186`）——同日添加又回滚；`[Doc] Host model recipes in recipes/`（`b515648641`，`#61302`）；`[Doc] Sync DeepSeek-V4.1-Flash recipe with vllm-project/recipes#1136, #1137`（`21e09ea959`）。

**应用场景**  MLA 模型长上下文的 prefill 后端选择（opt-in，风险可控）、P/D 解耦下 KV 传输与投机 draft 的时序优化、混合注意力 + Mamba/KDA 模型的 KV/state 传输、大规模 PP 部署的 CPU offload 池定尺寸、单节点 NVSwitch 系统的 all-reduce 后端选择、小 batch 高词表采样的延迟治理、可恢复流式会话在抢占下的状态保持。

**研判**  **事实：** 本期 vLLM 有**三条彼此独立、但同属一类**的投机解码修复（崩溃 → 改为禁用、陈旧 block-table 行 → 清空、EAGLE group 猜测 → 显式标签），且都附有可复现的失败证据（断言位置、`kept [4160] expected [2300]`、`7 failed → 140 passed`）。**观点：** 四点——(1) `#61455` 是本期**最值得记住的一条**：它说明「**投机解码静默禁用 hybrid 模型的前缀缓存**」这类问题**不会报错、不会降精度、只会在账单上体现**（缓存命中率归零），而根因是规划器**猜** group 归属。把「谁持有 draft 层」变成 runner 上报的显式标签，是把**隐式约定升格为契约**——与 `#53558`（KV cache 分组可插拔）方向完全一致；(2) `#61989` 提供了一个**教科书式的回退归因**：靠 nightly 的跳变定位到 14 个提交、靠「只有 TP>1 变化」锁定唯一的分布式改动、再用强制后端的 A/B 定量，并且**公开质疑原改动是否修了一个未记录的 bug**（PR 末尾直接 @ 原作者提问）——这类「**没有性能数据就合入的默认值变更**」是推理引擎最常见的性能回退来源，值得作为 review 检查项；(3) `#61674` 的价值不在性能而在**暴露了一个计数不一致**：自动定尺寸让 PP0/PP1 得到 25/25/23 三个不同的池大小，进而产生 **40 次越界 native copy**——而基线「完成了响应且没有段错误」，说明**这类缺陷可以长期潜伏**；(4) `#57320` 揭示了一个新的耦合面：**权重缓存守护进程与引擎对同一块共享内存各做一次「权重预处理」**，`ipc_cache` 这类跨进程复用方案会引入「谁有权改写共享分配」的问题，**任何自建权重缓存的项目都应检查同类重复变换**。

**来源**
- vLLM 官方提交 API（2026-10-17 UTC，48 条）：https://api.github.com/repos/vllm-project/vllm/commits?since=2026-10-17T00:00:00Z&until=2026-10-18T00:00:00Z&per_page=100
- MLA context prefill CuTe DSL：https://github.com/vllm-project/vllm/pull/62069 （commit `90d6d871d6`）
- 跳过 MLA 权重吸收：https://github.com/vllm-project/vllm/pull/57320 （commit `c61a1e73c6`）
- KV 传输结果前移：https://github.com/vllm-project/vllm/pull/62041 （commit `0c5dd18e7a`）
- MoRIIO hybrid WRITE + DSpark：https://github.com/vllm-project/vllm/pull/60544 （commit `41b8c15809`）
- NIXL push 模式删除不可达分支：https://github.com/vllm-project/vllm/pull/62210 （commit `7879f3687b`）
- EPLB 等专家 buffer 拷贝：https://github.com/vllm-project/vllm/pull/62081 （commit `9c778e9ff8`）
- SimpleCPUOffload 显式块数：https://github.com/vllm-project/vllm/pull/61674 （commit `ce8853d27a`）
- fixed-K proposer 禁用动态投机：https://github.com/vllm-project/vllm/pull/46399 （commit `763e3f7c91`）
- 清理陈旧共享 block-table 行：https://github.com/vllm-project/vllm/pull/61174 （commit `61bb590877`）
- EAGLE group 改显式标签：https://github.com/vllm-project/vllm/pull/61455 （commit `c3de0b37e2`）
- split top-p 小批量加速：https://github.com/vllm-project/vllm/pull/61996 （commit `7b6310c109`）
- 边界 tie 计数跨 block：https://github.com/vllm-project/vllm/pull/62046 （commit `ce49c47cbb`）
- 单节点 MNNVL all-reduce 回退修复：https://github.com/vllm-project/vllm/pull/61989 （commit `2b85d9cb65`）
- 可恢复会话在抢占时停止：https://github.com/vllm-project/vllm/pull/58778 （commit `f98ae6a9fd`）
- 最新 GitHub Release 仍为 v0.31.0：https://github.com/vllm-project/vllm/releases

**证据等级：** 官方一手；`#61996` 的 1.22–2.16× / 4.4×、`#61989` 的 5462 vs 5728 tok/s、`#61674` 的 40 → 0 次越界拷贝等均为 PR 作者自述，绑定其所述硬件与配置；`#55671` 明确**未验证新 GPU 依赖栈**；`#62069` **未给出性能数字**。

---

## 3. SGLang：10-17 提交（47 条）——HiCache 写回排序与 MLA Mamba 按 rank 键、unified memory 联合回收、DSv4.1 默认开启 decoder bounded replay

**主题**  SGLang 在 10 月 17 日的 47 条提交集中在：**HiCache 的两条正确性修复**（写回 host 拷贝必须排在产生 KV 的 forward 之后、MLA 模型上 Mamba state 必须按 TP rank 加键）、**unified memory 的联合回收与无 scratch 压缩**（`#39294` + `#44097` 成对合入）、**prefill CUDA graph replay 前必须等待逐层 cache load**、**DeepSeek-V4.1 默认开启 decoder bounded replay**，以及 **MegaMoE 在 SM100 上的 EPLB 再平衡**与**任意 32 对齐 block 的 shared-expert 融合**。

**内容详情**  官方提交 API 与 PR 正文显示（节选可核验项）：
- **HiCache（三条，两条为静默错误）：**
  - **`[Fix] hicache: Order write-back host copies after the forward that produced the KV`**（`f75232cb42`，PR `#44800`）：在 `--hicache-write-policy write_back` 下，KV 页**可能在被产生它的 forward 完成之前就拷到 host**，于是 host 副本保留的是**提交时刻**的页面内容，后续 load-back 会把**陈旧 KV** 喂给另一个请求。两条路径都没有依赖 forward stream：**insert 时的 SWA window 备份**（write-back 把 Full/Mamba 备份推迟到驱逐，但树核仍会在请求节点插入时**发布未备份的 SWA window**，而该请求自己的 forward 可能仍在写这个 window）；**阻塞式 write-back 排空**（驱逐把本 tick 排队的全部内容提交到 D2H stream 而**没有 fence**，被驱逐的 SWA 页可能属于「产生它的 forward 尚未完成」的 checkpoint）。改法：write-back 下**插入阶段不再返回 BackupKV 动作**（Python `UnifiedTreeCore` 与 Rust `unified_tree_core.rs` 同步）；`HiCacheController.start_writing(wait_for_forward=...)` 在配置了 `write_fence_stream` 时让 D2H stream **先等 forward stream**，调度器把 `write_fence_stream` 设为 model runner 的 forward stream（与既有 `load_fence_stream` 并列）。
  - **`[Fix] hicache: Key MLA Mamba state per TP rank in the HF3FS backend`**（`5fcb07da34`，PR `#43817`）：**MLA 在 TP rank 间共享 KV 存储键**，但 **Mamba checkpoint 是分片的**——于是 MLA 模型上所有 HF3FS rank 都用 rank 0 的文件与元数据，**不同 rank 的 Mamba 页拿到同一个键**。改为给 Mamba 键加 **TP rank/size 后缀**，其他池保持原键；**旧的未加 rank 的 Mamba 条目不再复用**（归属 rank 不明）。与 `#43594` 对 NIXL 的修复同源。
  - **`[Feat] server-args: Add --enable-hicache alias for --enable-hierarchical-cache`**（`3fc08b939a`，PR `#44849`）；**`[Doc] hicache: Explain HiCache host pool sizing with a storage backend`**（`9590dc80cc`，PR `#44790`）。
- **unified memory / unified cache（本期最集中的一组）：**
  - **`[Fix] unified-memory: Bound joint reclaim and preserve cached prefixes`**（`ee30dc4d40`，PR `#39294`，已 squash 合入）：FULL/SWA 联合分配**可能驱逐超过所需的已缓存前缀**；FULL/SWA/Mamba 分配**可能明明有可回收容量却失败**，或在拒绝一个不可能满足的需求前**先丢弃整个缓存**。改法：让分配驱动的驱逐停在**只读的充分回收谓词**上，在有界阶段边界准备并校验物理布局；**先用字节或 ID 上下界拒掉可证不可能的需求**，再限定 token/state 回收配额；为统一 SWA 分配器计算**精确的未分片 paged-extend 需求**（其余位置保持保守）。
  - **`[Fix] unified-memory: Avoid gather scratch during Mamba and Conv compaction`**（`025655068c`，PR `#44097`）：统一 Mamba/Conv 压缩在搬移槽位时使用 **gathered state tensor**，**恰恰在分配器回收内存的时刻可能 OOM**。改为用既有 `copy_pages` kernel **直接拷贝不相交的物理 state envelope**，去掉主 state 的 gather 临时量；保留兄弟 state 拷贝与常规 copy-on-write。两条合计在 H200 上通过 **87 测试 / 211 子测试**（覆盖 dLLM KV 复用、prefill 预算、联合回收、SWA 页估算、Mamba/Conv 压缩与视图、相邻 HiCache 排序回归）。
  - **`[Feat] unified_cache: Support hybrid SWA + Mamba models in the Mooncake direct linker`**（`05c1539098`，PR `#41343`）：unified-cache 的外部 linker（**direct L3，无 host 层**）此前只支持 Full 与 SWA 树组件，**混合 SWA + Mamba 模型（如 Inkling）在启动时被 Python wrapper 与 Rust 树核双双拒绝**。新增 FULL+SWA+MAMBA 的设备池组（Mamba 池是**槽位寻址**，conv-only 模型的空时序状态跳过）；**启动期明确拒绝**：`--enable-unified-memory` 与 int8 Mamba checkpoint、Mamba side state（short conv / ngram）、scaled KV cache 的组合。依赖 `#44220`。
  - **`[Fix] unified_cache: Prevent misaligned Mamba checkpoints after speculative truncation`**（`86e67f7d93`，PR `#43387`）：Mamba `no_buffer` + 投机解码下，**verify 在 CPU 侧的 grammar/finish 截断之前就推进了 live state**，把这个 state 以**更短的 cache key** 捐赠出去会污染后续前缀复用。改为要求 live state 长度匹配 `owned_kv_len()` 才允许捐赠，否则跳过该已完成请求的插入（保留既有前缀与正常清理，**不增加 GPU 同步**）。验证（GB300，TP1，Qwen3.5-0.8B，NEXTN 3 步 4 draft，no_buffer，page size 1，关闭 overlap/CUDA graph）：`D_KL(cold‖warm)` 在 count 场景 **0.876971 → 0.000235300**、story 场景 **4.107031 → 0.000941570**（阈值 0.005，全 248,320 词表项）；warm cache 长度 51 → 42、27 → 22（即**不再复用错位的 checkpoint**）。
- **CUDA graph（本期最危险的一条静默错误）：**
  - **`[Fix] cuda_graph: Wait for in-flight layer-wise cache loads before prefill graph replay`**（`ae28306a74`，PR `#44220`）：HiCache L2 load-back、HiCache storage load-back（如 `buffer_only`）与 unified-cache 外部 linker 都是**逐层**把 KV/SWA/Mamba state 载入设备池并与 prefill forward 重叠，forward 在池访问器（`get_key_buffer`、`mamba2_layer_index` 等）内部通过注册的 `layer_transfer_counter` **等每一层**。**prefill CUDA graph replay 从不执行这些 Python 层等待**——图是整体启动的，于是**图可能重放一个 load 仍在途的 batch**，forward 读到尚未落地的层，**静默产生错误 logits**；后续页的 KV 又建立在被污染的状态上，**还会以内容寻址键被缓存与备份**。`scheduler.py` 里那句 `# todo: disable cuda graph execution if hicache loading triggered` 自 **2025-06** 就挂着。当前 main 上**任何对 loading batch 重放 prefill 图的模型都可能触发**；Inkling 默认用 Full prefill CUDA graph 后端，其 HiCache 路径受影响。**现有 Inkling HiCache 逐位一致测试之所以通过，只是因为 H2D 拷贝通常跑赢了 replay**。改法：`PrefillCudaGraphRunner.execute` 在 Full replay 前，若层传输计数器仍有活跃消费者就**等到最后一层落地**（同时支持 HiCache `LayerDoneCounter` 的 stream 等待与 linker `LayerWiseLoadCounter` 的 CPU 等待）。
- **DeepSeek-V4.1（五条，含默认值变更）：**
  - **`[Perf] DSv4.1: Enable decoder bounded replay by default`**（`3603824541`，PR `#44403`）：在**受支持的独立 CUDA DSv4.1 配置**上默认开启 decoder SWA bounded replay，同时保留显式全 prefill 退出开关。配对精度（**作者明确称之为有限、随机的精度测量，不是数值等价或普遍加速的证明**）：AIME 2026 不设输出上限时两边**各 469/480**；DeepSWE 相差 1 个解决任务、MMMU-Pro 相差 5 个正确回答。**PD、context parallelism、encoder replay、其他平台与不兼容配置不自动开启**；新增 `--disable-decoder-bounded-replay`。**bounded replay 与全 prefill 数值不同，且不支持 prompt logprobs**。另含投机验证越过 **1,048,576 token** 边界所需的候选表容量修复（按页表容量选 16 KiB 或扩展 32 KiB 位图，**拒绝 > 2M token 的容量**，不放大模型上下文上限）。
  - **`[Perf] deepseek-v4.1: Skip wasted layer 20 work on earlier prompt tokens under decoder bounded replay`**（`05ce0e980f`，PR `#42901`）：开启后**最后几层只处理每请求最后 128 个 prompt token**，但该优化从**第 21 层**才开始，**第 20 层仍处理全部 token**。第 20 层做两件事：写解码要读的**压缩 KV cache**（需要全部 token）、跑自己的 **indexer/attention/MoE** 产出（而第 21 层只读该输出的最后 128 行）。因此第 20 层在更早 token 上的 indexer/attention/MoE **全部被丢弃**，且 **indexer 扫描整个上下文，浪费随 prompt 长度增长**。改为第 20 层**先为所有 token 写压缩 KV**，再只对最后 128 个 token 跑该层其余部分；**ROCm 保持旧行为**（其融合层路径一次调用跑完第 20 层）。
  - **`[Perf] DSv4.1: Reduce BCG metadata memory`**（`e4512a5a17`，PR `#44920`）：encoder bounded replay 关闭时，DSv4.1 的可断 prefill 图在**各 capture bucket 与压缩比之间**保留**过大的页表与重复索引缓冲**。改为用既有 prefill 图上下文上限**约束低压缩比 BCG 的页表宽度**、**跨串行 capture bucket 复用 full-metadata 索引存储**（保持 c1/c2、raw/page、full/tail 缓冲独立）、**在 prefill chunk 内 c1/c2 之间共享被完全覆写的逐层 combined-index scratch**（c0/c128 输出仍独立）。
  - **`[Fix] [AMD] deepseek-v4.1: Reject attention DP with encoder SWA bounded replay on ROCm`**（`563cd60345`，PR `#44934`）：ROCm attention 后端**把前缀命中放在自己的 forward 里重放**而不折进 extend，而只有命中的 attention-DP rank 会进入该 forward——用「每 rank 一个 count 的 MLP-sync gather」时会在**首个前缀命中**处 `_set_execution_counts` 抛错，用「一个 count 的 gather」时则**只在命中的 rank 上跑集合通信**。改为**启动即失败**；CUDA 无行为变化。
  - **`[Fix] deepseek-v4.1: Serve prompt-logprob requests under decoder SWA bounded replay`**（`a2953d7b7f`，PR `#44850`）；**`[Fix] deepseek-v4.1: Keep the cached KV of folded encoder replay rows`**（`ef26f6ad49`，PR `#44857`）。
- **MoE 与量化（DSv4 主线，均为「消除重复/浪费」）：**
  - **`[Fix] megamoe: Support EPLB rebalance on SM100`**（`9259553f09`，PR `#44847`）：SM100 的 MegaMoE 把**专家参数的副本**（`build_mega_moe_experts_weights` 里 UTCCP 转置的 L1/L2 scale、`_build_mega_moe_weights` 里的 NVFP4 `mega_l1_alphas`/`mega_l2_alphas`/`mega_l2_act_scales`）注册为 experts module 的参数——EPLB 搬动专家的每个 routed-expert 参数切片（`get_moe_weights`）时，**这些副本会随之搬走**；此前被搬动的专家**仍沿用前一个专家的 scale**。实测（DeepSeek-V4.1-Flash，DEP2，2×GB300，`--tp 2 --dp 2 --ep-size 2 --enable-dp-attention --moe-a2a-backend megamoe`，GSM8K 200 例 greedy）：main + EPLB **72.50%**（142 次再平衡，平均输出 465.4 token）→ 本 PR + EPLB **97.00%**（35 次，127.2 token）→ 本 PR 关 EPLB **97.50%**（0 次，127.6 token）。
  - **`[Perf] deepseek_v4: fuse block-FP8 shared experts into MegaMoE for any 32-aligned block size`**（`14c4dda24e`，PR `#44837`）：profiling 显示 MegaMoE 以 `kNumSharedExperts=0` 实例化、shared expert 退化为独立 MLP（每层 2 个 dense MXFP8 GEMM、2 次 activation 量化、1 个 `silu_mul_clamp`、1 个 `add_`）。根因：`should_fuse_mega_moe_shared_experts` 要求 `shared_experts_weight_block_size == [128,128]`，而 DeepSeek-V4.1-Flash 的 checkpoint 是 **[32,32] block-FP8 + ue8m0**，于是**开关默认开着却静默回落**。改为接受任意 `block_k % 32 == 0`，并把 checkpoint block scale 展开为内核的 per-row 1×32 UE8M0 布局。量化收益（GB300 DEP2，每 rank 每步 16K token 冷 prefill）：每步 **1,812 个 kernel launch**、其中 **210 个 dense MXFP8 GEMM（80 个属 shared-expert，约 18 ms）**、shared-expert 的量化/silu/add 各 80/40/40 次；**该 eager step 是 dispatch-bound（kernel 间中位空隙 41 µs，每次 launch 约 65 µs 空转），因此省下的约 200 次 launch 与 kernel 时间同样重要**。
  - **`[Perf] deepseek_v4: fold the MegaAttention query layout into wq_b for block-FP8-as-MXFP8 checkpoints`**（`a126e8d640`，PR `#44838`）：`setup_mega_attention` 只在 `wq_b` 是**原生 MXFP8 linear** 时才把 query head 布局折进 `wq_b` 的行；DSv4.1-Flash 的 [32,32] block-FP8 走 **block-FP8-as-MXFP8 视图**，导致 `mega_wq_b_permuted` 恒为 false，**每层都在 forward 时现打包 query**。实测（GB300，DEP2，每 rank 一个 16,384 token 冷 prefill chunk，torch profiler）：每步 **40 次 `aten::new_zeros`（每次 141 µs，合计 5.7 ms）+ 40 次 `aten::copy_`（每次 691 µs，合计 27.6 ms）= 33 ms / 322 ms 的 step（约 10%）、80 次 launch**；每层写约 **1.2 GB 零** 并搬约 **2.4 GB** 非连续数据。
  - **`[Perf] deepseek_v4: run FlashInfer autotune for block-FP8 models served as MXFP8`**（`bce7b27aa6`，PR `#44842`）；**`[Refactor] nvfp4_online: Use explicit FlashInfer 4over6 config`**（`29a7e19de1`，PR `#41745`）。
- **稀疏注意力：**
  - **`[Perf] [NVIDIA] qsa: Keep sparse decode scratch in FP8 for FP8 KV pools`**（`3b34d9f8b6`，PR `#44114`）：`#38851` 修 QSA 稀疏解码损坏时做了三件事，其中**把 FP8 池行反量化进 BF16 scratch 是保守选择**（使 TRTLLM-gen decode 恒为 BF16-q + BF16-KV）。而 FlashInfer 的 `trtllm_batch_decode_with_kv_cache` **支持 BF16-q + FP8-E4M3-KV**；FP8 池下让 packed scratch 保持 FP8 可**把 gather 与 attention 的 KV 流量减半**并**去掉 `fused_kv_prepare` 里的反量化**。精度（Qwen3.8-Flash-Next-NVFP4 on VR200，TP1，FP8 KV，FP8 QSA indexer，NEXTN MTP 3 步 4 draft）：AIME24+25 60×16 **90.52% → 92.08%（+1.56 pp，95% CI [−0.47, +3.59]）**、GPQA 100×8 **87.25% → 88.62%**、GSM8K 200 例 **198 → 198**；对 FP32 参考的注意力输出误差两条路径同为约 **2.1e-3**，**到 64K 以上不随 KV 长度增长**。
- **并行、通信与前端：**
  - **`[Perf] distributed: skip the DLPack re-wrap of custom all-reduce results`**（`400c1dfe6e`，PR `#44251`）；**`[Perf] kernels: Enter a JIT kernel's device guard only when the device differs`**（`56a20d7e34`，PR `#44644`）。
  - **Rust 前端补齐四个端点：** `[Feat] rust-server: Serve /v1/tokenize and /v1/detokenize`（`0d13efb2b8`，`#44767`）、`[Feat] rust-server: Report Python's launch-time /server_info fields`（`5cbe2371a5`，`#44764`）、`[Feat] sglang-server: Serve /flush_cache from the Rust frontend`（`b62a48ee15`，`#44766`）、`[Fix] gateway: Drop unused npyz dev-dependency to unbreak builds after zip yank`（`86cc2e02ec`，`#44720`）。
  - **重构：** **`[Refactor] TCPCG: Consolidate attention eager regions and isolate Inkling policy (4/9)`**（`b234b5c32e`，PR `#42288`）——把 attention 的 TCPCG split-op 适配器与 layer-registry 查表换成**类自有 eager 方法**，BCG replay 时通过 step 2 的 runner 作用域 `get_forward_batch()` 读活跃 batch；这是从 `#41634` 拆出的 9 步中的第 4 步（1–3 已合入，5–9 为 `#42289`–`#42293`）。
  - **NPU / AMD：** `[Fix] [NPU] attention: Use FIA TND for MLA prefix attention and add Gemma norm fallback`（`2d3dd3168d`，`#40131`）；`[Fix] [NPU] qwen: Fix shared-expert routing with Ascend FuseEP`（`7a8804edb5`，`#40516`）；`[Fix] megamoe: Respect request-pool limits in decode validation`（`ff979e0816`，`#44816`）；`[Refactor] dsv41: Shard Engram tables on hash-column boundaries`（`01dc08e23f`，`#42787`）。
  - **Diffusion 子线（本期 8 条，非 LLM 推理主线，仅记存在）：** `#44634`（pi0.5 completion-boundary action batching）、`#40171`（FLUX.2 FP8/NVFP4 fast path 在 LoRA attach 后复验）、`#43732`（ZMQ ingress 端口）、`#44493`（FLUX.3 serving 稳定化）、`#44662`（per-task prompt enhancer + FLUX.3 Klein upsampler 协议）、`#44845`（多模态 padding 策略改 helper）、`#43823`（VSA-H3 支持 sm80/A100 的裸指针 Triton 加载）、`#44376`（LoRA merge 决策只在首次评估打日志）。

**应用场景**  write-back 型 HiCache 的 KV 正确性（尤其 SWA 窗口与驱逐排空）、MLA + Mamba 混合模型的多 rank 共享存储键设计、unified memory 在「回收与压缩同时发生」时的 OOM 防护、混合 SWA+Mamba 模型接入 L3 direct linker、prefill CUDA graph 与异步逐层加载的竞态、DSv4.1 的默认 replay 策略与精度验收、MegaMoE 的 EPLB 与 block-FP8 量化布局、FP8 稀疏注意力的 scratch dtype 选择。

**研判**  **事实：** 本期 SGLang 有**四条「不报错、只出错」的正确性修复**（write-back 拷贝早于 forward、MLA 下 Mamba 键冲突、speculative truncation 后 checkpoint 错位、prefill graph 重放读到未落地的层），每条都给出了机制级因果链与量化验证（KL 0.877 → 0.000235、`7 failed → 140 passed` 等）。**观点：** 四点——(1) `#44220` 是本期**全库最值得读的一条**：它暴露的不是某个 kernel 的 bug，而是**「逐层异步加载」与「图整体重放」两种执行模型的语义冲突**，而且**现有测试通过只是因为拷贝跑赢了 replay**——这类「靠时序侥幸通过的测试」在推理引擎里会系统性掩盖竞态；同类风险适用于任何把 `cudaGraphLaunch` 与异步 H2D 混用的栈（TensorRT-LLM 的 CUDA graph + offload、vLLM 的 CUDA graph + KV offload 同理）；(2) `#44837`/`#44838` 合起来说明 **DSv4.1-Flash 的性能瓶颈已从 kernel 算术转到 dispatch 与打包开销**：一个 16K chunk 的 eager step 有 **1,812 次 launch、kernel 间中位空隙 41 µs**，而单是 query 重打包就吃掉 **33 ms / 322 ms（10%）**——**在 dispatch-bound 的 prefill 上，「少 launch」比「快 kernel」更值钱**；(3) `#44403` 把 bounded replay 设为默认，是本期**唯一一处「默认行为变更」**，作者自己把精度证据限定为「有限、随机测量」并保留退出开关——**采用方应把它当作需要自行复验的默认值**，尤其在需要 prompt logprobs 的场景（该路径不支持）；(4) `#41343` 的**拒绝清单**本身是信息：`--enable-unified-memory` 与 int8 Mamba checkpoint、Mamba side state、scaled KV cache **不兼容**，说明 unified-cache 的能力边界仍在「FP16/BF16 KV + 无 side state」这一区间。

**来源**
- SGLang 官方提交 API（2026-10-17 UTC，47 条）：https://api.github.com/repos/sgl-project/sglang/commits?since=2026-10-17T00:00:00Z&until=2026-10-18T00:00:00Z&per_page=100
- HiCache 写回排序：https://github.com/sgl-project/sglang/pull/44800 （commit `f75232cb42`）
- MLA Mamba 按 TP rank 键：https://github.com/sgl-project/sglang/pull/43817 （commit `5fcb07da34`）
- unified memory 联合回收：https://github.com/sgl-project/sglang/pull/39294 （commit `ee30dc4d40`）
- Mamba/Conv 压缩去 gather scratch：https://github.com/sgl-project/sglang/pull/44097 （commit `025655068c`）
- Mooncake direct linker 支持混合 SWA+Mamba：https://github.com/sgl-project/sglang/pull/41343 （commit `05c1539098`）
- speculative truncation 后 checkpoint 错位：https://github.com/sgl-project/sglang/pull/43387 （commit `86e67f7d93`）
- prefill CUDA graph 等待逐层加载：https://github.com/sgl-project/sglang/pull/44220 （commit `ae28306a74`）
- DSv4.1 默认开启 decoder bounded replay：https://github.com/sgl-project/sglang/pull/44403 （commit `3603824541`）
- 第 20 层跳过浪费工作：https://github.com/sgl-project/sglang/pull/42901 （commit `05ce0e980f`）
- BCG metadata 内存：https://github.com/sgl-project/sglang/pull/44920 （commit `e4512a5a17`）
- ROCm 拒绝 attention DP + encoder SWA：https://github.com/sgl-project/sglang/pull/44934 （commit `563cd60345`）
- MegaMoE EPLB on SM100：https://github.com/sgl-project/sglang/pull/44847 （commit `9259553f09`）
- shared expert 融合任意 32 对齐 block：https://github.com/sgl-project/sglang/pull/44837 （commit `14c4dda24e`）
- MegaAttention query layout 折进 wq_b：https://github.com/sgl-project/sglang/pull/44838 （commit `a126e8d640`）
- QSA FP8 scratch：https://github.com/sgl-project/sglang/pull/44114 （commit `3b34d9f8b6`）
- TCPCG 4/9：https://github.com/sgl-project/sglang/pull/42288 （commit `b234b5c32e`）
- 最新正式版仍为 v0.5.21：https://github.com/sgl-project/sglang/releases

**证据等级：** 官方一手；`#43387` 的 KL 数值、`#44847` 的 GSM8K 72.50→97.00、`#44838` 的 33 ms/322 ms、`#44114` 的 AIME +1.56 pp 等均为 PR 作者在自述配置下测得；`#44403` 作者自述精度差异「非数值等价证明」；`#44847` 与 `#44837` 的 CI 状态在 PR 页面显示为失败/跳过（`PR Test (Base)` 🚫、`(Extra)` ❌），**合入前状态需以维护者确认为准**。

---

## 4. TensorRT-LLM：10-17 提交（11 条）——Kimi K3 在 Rubin（SM107）上的 23 提交优化栈、Qwen3.5 GB300 agentic serving 文档

**主题**  TensorRT-LLM 在 10 月 17 日仅 11 条提交（周六），但其中 **`#19960` 是单个 23 提交的堆叠**，把 Kimi K3 在 **Rubin（SM107）** 上的路径系统性重做；其余为 Qwen3.5 的 GB300 agentic serving 文档、HF 流式 detokenizer 收尾、CUDA 前向兼容驱动保持、多节点 perf-sanity 的 NVLink 域约束与若干 infra 变更。**本期同时补齐上一期欠账 3 条**（`#18599`/`#19981`/`#20290`）。

**内容详情**  官方提交 API 与 PR 正文显示：
- **`[None][feat] Kimi K3 optimizations on Rubin`**（`e88d19ca3b`，PR `#19960`，基于 main `ff7ea92dfb`，**23 个提交**）：官方正文列出的改动（节选）：
  - **MXFP8 FP8 weight-read linear**——SM107 + Rubin CuTe DSL 下 `_Fp8BlockScaleWeightReadLinear` 打包 **K32 R128c4 UE8M0 scale** 并跑 `cute_dsl_mxfp8_gemm_rubin`，新增 **K3 细粒度 low-M 调优桶（`fine_grained_m`）**；**SM100/SM103 继续用 DeepGEMM**；新增 `forward_prequantized`。
  - **CUTEDSL_FC12 SiTU**——Rubin 的融合 FC1+FC2 kernel 在 **trace 时**为 gated epilogue 特化 SiTU；runner/op/leaf 传递 SiTU soft-cap 并以 epilogue 作为 kernel 与 autotuner 的键；CuteDSL runner 只对 locality-domain 路径预热 inner tactics（`prime_inner_tactics`），使以 epilogue 为键的 FC12 runner **不再在 autotuner 预热时抛 `TypeError`**。Kimi K3 接受 `moe_config.backend: CUTEDSL_FC12`。
  - **SM107 上的 CuteDSL NVFP4 SiTU**——撤销 `#19003` 加入的 SM107 拒绝（Rubin 的 act-fusion kernel 已实现 SiTU）。
  - **CuTe DSL MLA 的 FP8 KV scale 放到设备上**——FP8 decode kernel 与 op 把 BMM1/BMM2 scale 作为**设备张量**，**去掉逐层 host 读取**。
  - **Helix 的逐 token 边界覆盖**用于 CuTe DSL MLA decode（kernel 侧已由 `#19273` 在 main 上）。
  - **DWDP for Kimi K3 on CUTEDSL / CUTEDSL_FC12**，`submit_dwdp.py` 支持 `compact_packing`（依赖 `#19183`）。
  - **tensor-core（tcgen05）KDA MTP verify kernel**——可选 **bf16 tensor-core 操作数路径**（`TRTLLM_KDA_MTP_BF16_MATH=1`，默认关）；replay 缓存拆为 `kda_k_cache`(bf16) + `kda_g_cache`(fp32) + bf16 的 `kda_v_cache`/`kda_beta_cache`/conv 缓存，**bf16 与 fp32 状态池都接受**；SM107 启用优化后的 KDA kernel；mixer 改用 kernel 的**融合 attention-residual epilogue**（打包 token 布局、epilogue 内 gated RMSNorm `fuse_output_norm`），SM107 上对**可断 CUDA graph 之外的 verify-only batch** 用 MXFP8 core（R128c4）直连 `o_proj.forward_prequantized`；缺 output-gate 段或未定稿 o_norm 权重时**保留 Python output gate 的旧启动作为回退**。
  - **KDA suspend 在 tensor-core kernel 上**——被挂起请求的 pending KDA draft 通过 verify kernel 的 **REPLAY_ONLY 变体**提交（chunk 即已接受 draft，只存状态）；`kda_mtp_commit_pending_drafts` 等读拆分后的缓存。**官方明确：resume 后的 verify 开启新的 WY-form chunk，因此与未挂起的运行「按 kernel 精度一致」而非逐位一致；conv 窗口与 replay 缓存保持逐位一致。**
  - **CuTe DSL MLA decode 的 split-KV 定尺寸**——约 **8 个 CTA wave** 覆盖打包的 128 行 query tile（`ceil(H * S / 128)`），仅非持久化；CUDA-graph workspace 按 batch bucket **一次定尺寸**。
  - **K3 MoE routing prologue**——tensor-core 的 **896 专家 router GEMM**、基于 redux 的小 batch `noAuxTc` top-k、post-topK routing 分级与 launch 选择、**异步 NVFP4 输入量化**（CuteDSL 的 MoE aux stream，`TLLM_MOE_ASYNC_INPUT_QUANT`）。
  - **routed RMSNorm 融进 routed all-reduce**——`RMS_NORM` 获得**无残差**的 MNNVL 支持，其他策略回退。
  - **MegaMoE 启动死锁**——FC12 scheduler prologue 修复；TP 下 fused-comm `MEGAMOE_CUTEDSL` 的 **shared-expert all-reduce 排在 routed MoE 之后**。
  - **融合 router + latent-down GEMM**（`trtllm::dsv3_router_latent_gemm_op`，SM107，TP `MEGAMOE_CUTEDSL`）；**latent down-projection 与 routing 重叠**（`input_ready_event`，`ConfigurableMoE` 与调度器在首次读 `x` 时 join；FC12 op 用一次 fill 清零计数）。
  - **KDA 独立 QKVG codes 在融合时释放**——各投影融合后立即释放独立 Q/K/V/G FP8 codes，在 **DEP8 的 context rank 上避免约 22.6 GiB 峰值加载内存**。
  - **KDA replay 槽位更新不再有阻塞 HtoD 拷贝**；**DSpark draft KV 页大小**——DFlash 与独立 DSpark 的 draft KV 池拥有**自己的 32-token 页大小**，128-token 的目标页大小不再破坏 trtllm-gen MLA drafter；**DSpark MLA drafter 默认走 CuTe DSL block decode（AUTO），drafter 区域约快 36%**；**draft KV GPU 预算上限**（`TLLM_DRAFT_KV_GPU_BUDGET_GIB`，opt-in，多余容量给目标池，不低于 draft 分配器下限）；**Helix CP GEN autotuner 预热**改为在一个 generation-only batch 上执行，PARALLEL tactics 按 TP×CP 组切分（每 rank 只 profile 1/(TP×CP)）；**CuTe DSL MLA decode 的 split-KV 按请求规划**（一次 launch 的 Triton kernel 依 KV 长度定 split 数，单 split 请求直接存输出、reducer 跳过其行）；**FP8 H=96 MLA verify group 在 SM100/SM103 走 CuTe DSL**（非 Helix 的 64 < H < 128 多 token FP8-KV MLA 生成 batch 此前落到 FlashInfer TRTLLM-Gen，**该 head 范围被拒导致 executor 预热失败**）。
  - **明确不含：** BMM/GEMM cubin 更新；**把 FP8 KV 设为 K3 MLA 默认路径**（其 hook 已被 `#18653` 移除）；**K3 的 FP4（NVFP4 KV）MLA**——main 在混合线性注意力模型与分离式服务上拒绝 FP4 MLA，相关 context/Helix/split-KV/CTX-cache 工作**另开 PR**。
  - **评审状态（需注意）：** PR 页面上的自动评审仅检视到**三个 Python 文件的 import 格式化改动**，并写明「所给目标中的 Kimi K3 与 Rubin 特性声明**未被该 diff 支持**」——这说明**评审看到的是局部 diff**，正文所列运行时改动面远大于此。
- **`[None][doc] Document Qwen3.5 GB300 agentic serving`**（`5de29bb28f`，PR `#20514`，作者标注由 Codex 生成）：为 GB300 NVL72 上的长多轮 Dynamo + TensorRT-LLM 部署补 **NVFP4-V2 recipe**——固定运行时版本、**prefill/decode 分离 YAML**、worker 环境、可运行的 **colocated 1P1D** 命令，以及 InferenceX `#4195` 中**八种 topology/batch 变体**的调优表；含 **MTP6、FP8 KV cache、host offload、conversation-scoped block reuse、session affinity**。**作者明确：源码扫描的 9 个性能作业与 6 个评测作业使用合成投机接受率，生产 recipe 去掉了该控制项；Docker/MPI 命令与自然接受率吞吐「尚未在 GB300 上运行」**。
- **其他：**
  - **`[fix] Flush the HF streaming detokenizer on the final response; fix the AgentX aiperf install check`**（`2f5bb049f3`，PR `#20424`）；**`[fix] Keep the 26.08 CUDA forward-compat driver on the 26.09 image`**（`8b2ac2d08e`，PR `#20252`，nvbugs 6919405）；**`[fix] Keep multi-node aggregated perf-sanity jobs in one NVLink domain`**（`20d5469559`，PR `#20418`）；**`[fix] disable bolt`**（`c962eddaa4`，PR `#20496`）；**`[infra] Check in most recent lock file from nightly pipeline`**（`700d9f4e29`）；**`[TRTLLM-10804][infra] Run guardwords scan in nightly releases`**（`acb83ac268`，PR `#20135`）；三批 **waive failed cases**（`#20505`/`#20500`/`#20478`）。
- **上一期欠账补取（3 条，本期补齐）：**
  - **`#18599`（96 head MLA fmha autotune）**：正文极简，仅「**fmha lib / autotuner 已更新、移除若干 H96 情形的旧 WAR、更新文档**」，**无性能数字**。→ 上一期按提交标题记录的「96 head MLA autotune」**不宜再作性能结论引用**。
  - **`#19981`（CUTLASS workspace 回收）**：为每个 runner/设备/stream 增加 CUTLASS workspace 跟踪，**三次合格 forward 后可回收高于成功预热基线的容量**；提供 native owner scope 与 Python decoder 集成，**多 stream 执行与 CUDA stream capture 绕过该 scope**，`TRTLLM_RECLAIM_WORKSPACE=0` 可整体关闭。**评审明确：作者报告的全模型负载中「没有发生任何 MoE 回收事件」，因此不构成真实模型的内存收益证据**；且所列 `L0_MergeRequest_PR` 流水线全部失败。
  - **`#20290`（GVR V2 top-K 溢出开销）**：修复确定性 GVR V2 top-K 在 **K=1024/2048** 下的过高开销（接 `#20100`）。根因：clustered logits 下少数 warp 拿到大部分候选，其**局部缓冲即使共享缓冲总量充足也会溢出**，触发**整行昂贵扫描**。改法：按**精确的 per-warp 计数并行回收溢出候选**、增加候选缓冲余量（不改采样阈值）、移除未用共享内存分配。实测（BS=512，N≈131,000，同一 B200，nsys + 冷 L2）：**附加开销从 208–221 µs 降到 1.05–2.76 µs**；V4 Pro K=1024（512×131,075）**kernel 300.57 → 94.25 µs，开销占比 +225.04% → +1.13%**；V4 Pro K=1024（512×32,771）**53.01 → 30.23 µs，+83.63% → +6.02%**；V3.2 K=2048（512×32,783）**63.21 → 33.80 µs，+106.67% → +11.31%**。

**应用场景**  Rubin（SM107）平台的 FP8/MXFP8/NVFP4 weight-read 与 MoE 路径选型、Kimi K3 类混合线性注意力模型的 MLA + KDA verify、分离式服务下的 draft KV 预算控制、GB300 NVL72 上 Qwen3.5 的 1P1D agentic 部署、长词表采样的 top-K 溢出治理。

**研判**  **事实：** `#19960` 是本期四家中**单个 PR 体量最大**的改动（23 提交），官方列出约 25 项子改动，并明确列出 4 项「不含」；`#20514` 是**文档类变更**且作者自述关键命令未在目标硬件验证。**观点：** 四点——(1) **本期最值得跟踪的是 `#19960` 里「host 侧逐层读取被搬到设备」的两处**（MLA FP8 KV scale 作为设备张量、KDA replay 槽位更新去掉阻塞 HtoD）——这与前一期 SGLang `#44598`（预构建 decode batch 输入改 `async_h2d_stack`，**每个被替换拷贝原先都执行 `cudaStreamSynchronize`**）是同一类问题：**decode 每步的 host↔device 同步点，是低延迟场景下最稳定的收益来源**；(2) `#19960` 的**回退设计**比优化本身更值得学：SM107 的 MXFP8 路径在缺 output-gate 段或 o_norm 未定稿时**保留旧启动**，SM100/SM103 继续用 DeepGEMM——**按平台分叉而不按版本分叉**，是把新硬件优化风险限制在单一平台的正确做法；(3) `#19981` 是一个**必须读评审意见的反例**：PR 描述了完整的 workspace 回收机制，但评审指出**全模型负载里没有触发过 MoE 回收**、且 L0 流水线全失败——**「机制存在」不等于「收益存在」**，这正好呼应第 1 节 vLLM 博客强调的「可复现的验证证据」；(4) `#20514` 的作者披露方式值得记录：**明确区分「源码扫描得到的合成接受率结果」与「未在 GB300 上跑过的 Docker/MPI 命令与自然接受率吞吐」**——这类披露应当成为引用官方 recipe 时的默认前提。

**来源**
- TensorRT-LLM 官方提交 API（2026-10-17 UTC，11 条）：https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits?since=2026-10-17T00:00:00Z&until=2026-10-18T00:00:00Z&per_page=100
- Kimi K3 on Rubin：https://github.com/NVIDIA/TensorRT-LLM/pull/19960 （commit `e88d19ca3b`）
- Qwen3.5 GB300 agentic serving 文档：https://github.com/NVIDIA/TensorRT-LLM/pull/20514 （commit `5de29bb28f`）
- HF 流式 detokenizer 收尾：https://github.com/NVIDIA/TensorRT-LLM/pull/20424 （commit `2f5bb049f3`）
- CUDA 前向兼容驱动：https://github.com/NVIDIA/TensorRT-LLM/pull/20252 （commit `8b2ac2d08e`）
- 多节点 perf-sanity NVLink 域：https://github.com/NVIDIA/TensorRT-LLM/pull/20418 （commit `20d5469559`）
- 上一期欠账：https://github.com/NVIDIA/TensorRT-LLM/pull/18599 、https://github.com/NVIDIA/TensorRT-LLM/pull/19981 、https://github.com/NVIDIA/TensorRT-LLM/pull/20290
- 最新版本仍为 v1.4.0rc0（prerelease）：https://github.com/NVIDIA/TensorRT-LLM/releases

**证据等级：** 官方一手（PR 正文）；`#19960` 的 36%（drafter）、22.6 GiB（DEP8 峰值）为**作者自述**，**未给出端到端吞吐对比**；`#19981` 官方评审明确**无真实模型收益证据**；`#18599` **无性能数字**；`#20290` 的 208–221 µs → 1.05–2.76 µs 有逐 shape 表格与测量方法，属较可信的 PR 级证据。

---

## 5. llama.cpp：10-17 提交（12 条）——低比特 MMVQ、grouped experts top-k 融合、HIP 的 `amdgcn_perm` 解包

**主题**  llama.cpp 在 10 月 17 日的 12 条提交里，**性能类改动密度为四家最高**：CUDA 低比特 **MMVQ** 重写（附七卡逐类型表格）、**grouped experts 的 top-k 融合**（补回分组带来的损失）、HIP 上用 **`__builtin_amdgcn_perm`** 替换 `__byte_perm` 做 q1_0 解包（MI50 上 pp/tg 大幅提升）、修复 **SYCL 在首次 decode 后的 prompt-processing 回退**；另有 Hexagon 的 Q1_0/Q2_0 支持、UI 的 HF Hub 模型发现与下载、Kolibri-1 模型支持。

**内容详情**  官方提交 API 与 PR 正文显示：
- **`CUDA: faster MMVQ for low-bit types`**（`2744e48d1e`，PR `#30517`）：`test-backend-ops` 在 **60 个不同的 4096×14336 矩阵**链上测（**权重不进 L2**），base 与 PR 交替跑 2–3 轮。**`q4_K` 与 `q8_0` 不变，被作者用作噪声基线**。ne11=1 时的加速比（节选）：`q2_K` **RTX 3090 1.59 / A16 1.71 / RTX 5090 1.13 / gfx1151 1.16**；`q3_K` **3090 1.67 / A16 1.59**；`q6_K` **3090 1.06 / A16 1.19**；`iq1_s` **3090 1.27 / A16 1.25**；`q1_0` **3090 1.17 / A16 1.21**；`iq2_s`/`iq3_s` 在多数卡上**持平**（GB10 上 iq2_s 1.15、iq3_s 1.11）。ne11=4 时 `q2_K` 在 3090/A16 分别 **1.38 / 1.40**。**注意：RTX 4090 与 GB10 在多数类型上接近 1.00**，即收益**强依赖具体卡型与低比特类型**。
- **`cuda : add grouped experts top-k fusion`**（`f2f667b02d`，PR `#29181`）：为 **grouped experts** 增加 CUDA top-k MoE 融合，作者自述「**补回分组专家损失的大部分性能**」。bailingmoe2 16B.A1B Q8_0：`tg128` **359.26 → 432.82 t/s（1.20×）**、`tg256` **360.21 → 435.93（1.21×）**、`tg512` **357.53 → 430.90（1.20×）**、`tg1024` **356.57 → 429.42（1.20×）**。
- **`ggml-cuda: HIP: use __builtin_amdgcn_perm for q1_0 unpacking`**（`9d1878a435`，PR `#29927`）：把 `vec_dot_q1_0_q8_1` 与 `ggml_cuda_mmq_load_tiles_q1_0` 里的 `__byte_perm` 换成 `__builtin_amdgcn_perm`。**AMD Instinct MI50（gfx906），m=4096、k=14336**：`pp2048` **2512.65 → 3570.25 t/s（+42.09%，1.421×）**、`tg128` **66.99 → 128.96 t/s（+92.51%，1.925×）**；小 n 的 TFLOPS 提升 **6.24×–10.65×**（n=1 时 0.263 → 2.26 TFLOPS），n=512 时 6.94 → 10.81 TFLOPS（1.56×）。
- **`sycl: fix prompt-processing regression after Q4_0/Q8_0 weight reorder`**（`f74e2fa6a3`，PR `#30374`）：**SYCL 上 Q4_0/Q8_0 权重一旦被 decode 重排，后续每个 prompt batch 都会先经 `dequantize_row_*_sycl_reorder` 反量化到 f16 再做 GEMM**，而这些 kernel 给每个 work-item 分配整个 32 值块、work-group 宽 16，导致**子组内 load 间隔 16 或 32 字节、store 间隔 64 字节**。后果：**首次 decode 后 Q4_0 的 prompt processing 掉约 40%、Q8_0 掉约 25%，并持续整个会话**（即**每次多轮对话都在慢速运行**）。改为与 `dequantize_block<...>` 相同的**逐元素映射**（Q4_0 每个打包字节一个 work-item、Q8_0 每个 work-item 两个值）+ **256 宽 work-group**，用 `ceil_div + downsample_sycl_global_range` 启动并循环覆盖全局范围；**数学保持 float，f16/fp32 两条路径反量化结果逐位相同**（刻意不用以 half 计算的 `dequantize_q*_reorder` 辅助函数）。
- **`hexagon: add support for Q1_0 and Q2_0`**（`797de63b2f`，PR `#30122`）：HVX 上的原生 Q1_0/Q2_0 `MUL_MAT`（对 tiled Q8_0 激活直接做 **{−1,+1} 展开**）、32x1/32x2/2D/4D/matvec worker、精确 host 重打包与任意字节范围读回、2D HMX 的 Q1_0/Q2_0 权重路径。**`GET_ROWS` 支持留待后续**。验证：v73/v75/v79/v81 构建通过，**SM8850/v81 Android** 上 Q1_0 与 Q2_0 **各 128/128 测试通过**。
- **其他：**
  - **`ui : Discover & Download Models from HF Hub`**（`c0fd786264`，PR `#29584`）：模型管理器新增「Discover Models」视图——HF GGUF 浏览/搜索（精选目录、防抖搜索、尺寸区间过滤）、**按量化显示下载 chip 并附内存适配提示**、下载状态实时化、每文件取消/删除；**下载主量化时自动拉取匹配的 draft sidecar（优先 DFlash）**。
  - **`model : add Kolibri-1 (Aleph-Alpha) support`**（`62318cd57d`，PR `#30673`）：德英双语 MoE 推理模型，**78B 参数、每 token 3.46B 激活**，50 层、**384 routed experts + 1 shared**；注意力为**每 4 个 513 窗口的滑动层 + 1 个全层**，滑动层用 NeoX RoPE、全层 NoPE；GQA **48 query head vs 4 KV head，head_dim 128 + QK-RMSNorm**；每 block 四个 RMSNorm（attention 与 MLP 输出在汇入残差前各自归一）；路由用 `logits + expert_bias` 选专家、以**无偏 logits 的 sigmoid** 加权且**不 renorm**。
  - **`ci : added ABI reports as github artifact`**（`83f9389419`，PR `#30619`）；**`tests: give the random model generator a per-tensor filler table`**（`10b8b75534`，PR `#30674`）；**`cuda : fix -Wtautological-compare in fattn vec dispatch`**（`9d1e03616e`，PR `#30519`）；**`model : Granite switch decision 256 options`**（`1268edac57`，PR `#30693`）；**`skills: add slice-model`**（`ea0c4c2c85`，PR `#30677`）。
- **上一期欠账补取（2 条，本期补齐）：**
  - **`#30613`（k-pool 视图共享）**：根因是**调度器为 host 输入的每个视图各自上传**，于是 `new_pool_idxs`（以及 `glm5-next` 的 `sel_mask`）的逐层 reshape **被复制了「层数」次**。→ 上一期按标题记录的「k-pool 输入视图跨层共享」应更正为「**消除 host 输入视图的逐层重复上传**」。
  - **`#30449`（pinned host buffer 上限 4 GiB）**：在 `--moe-cache-mib + --load-mode none` 下，**驻留 CPU 的 MoE 权重必须放在 pinned 内存**，否则向 GPU 流式传专家显著变慢。问题：无 mmap 时 CPU 侧权重进入**单个 `cudaMallocHost` 缓冲**，Qwen3.8-Flash-Next 配 `-cmoe` 达 **56 GiB**，**在 Windows 上失败**，ggml 于是**静默回退到 pageable 内存**。改法：CUDA pinned host buffer 类型**上限 4 GiB**，由既有分配器把大权重集**拆成多个 pinned 缓冲**；`--mlock` 跳过这些拆分缓冲（无单一基址，否则触发断言；它们本就是 pinned 即已锁定）。测试环境为 **RTX 5090 / Windows**。

**应用场景**  边缘与消费级 GPU 的低比特量化推理（q2_K/q3_K/q6_K/iq1 系列）、AMD 老卡（gfx906/MI50）与 gfx1151 的 q1_0 路径、MoE 模型在 llama.cpp 上的分组专家调度、SYCL/Intel 与 Snapdragon Hexagon 后端的 PP 稳定性、桌面端模型发现与 draft sidecar 自动匹配、Windows 上大 MoE 模型的 CPU 专家驻留。

**研判**  **事实：** 本期 llama.cpp 的四条性能改动都附**可复现的对照数据与测试方法**（交替跑、权重不进 L2、逐卡逐类型表格），并且**主动把 `q4_K`/`q8_0` 作为噪声基线**。**观点：** 四点——(1) **`#30517` 的方法论比数字更值得借鉴**：它没有只报最好的一张卡，而是**逐卡逐类型全表公开，并保留接近 1.00 的格子**——读者由此能看出「低比特 MMVQ 的收益集中在 3090/A16 这类**显存带宽受限的老卡**，而 4090/GB10 上多数类型无变化」。**任何引用「低比特提速 N 倍」的说法，都应要求这种粒度**；(2) `#30374` 是**最容易被误判为「模型慢」的 bug**：SYCL 上 Q4_0/Q8_0 在**首次 decode 之后**PP 掉 40%/25% 且**持续整个会话**——如果只测首轮 PP，或只测单轮问答，**永远发现不了**；这提示基准测试必须包含**「多轮之后重新长 prompt」**这一场景；(3) `#29927` 的 1.925×（tg128）与 10× 级（小 n TFLOPS）来自**把一个通用置换指令换成本架构专用指令**，属「编译器内建函数级」优化——这类收益**不可跨架构外推**（NVIDIA 侧无对应指令），但说明 **AMD 老卡的 q1_0 路径此前基本未优化**；(4) `#30449` 揭示的**「静默回退到 pageable」**是 llama.cpp 里反复出现的模式（内存不足→降级→变慢但能跑），与本期 vLLM `#61989`（默认值变更无性能数据）、SGLang `#44837`（融合开关默认开着却因 block size 不匹配静默回落）**同属一类：默认路径的静默降级**，应作为部署前的必查项。

**来源**
- llama.cpp 官方提交 API（2026-10-17 UTC，12 条）：https://api.github.com/repos/ggml-org/llama.cpp/commits?since=2026-10-17T00:00:00Z&until=2026-10-18T00:00:00Z&per_page=100
- 低比特 MMVQ：https://github.com/ggml-org/llama.cpp/pull/30517 （commit `2744e48d1e`）
- grouped experts top-k 融合：https://github.com/ggml-org/llama.cpp/pull/29181 （commit `f2f667b02d`）
- HIP q1_0 解包：https://github.com/ggml-org/llama.cpp/pull/29927 （commit `9d1878a435`）
- SYCL PP 回退修复：https://github.com/ggml-org/llama.cpp/pull/30374 （commit `f74e2fa6a3`）
- Hexagon Q1_0/Q2_0：https://github.com/ggml-org/llama.cpp/pull/30122 （commit `797de63b2f`）
- UI 模型发现与下载：https://github.com/ggml-org/llama.cpp/pull/29584 （commit `c0fd786264`）
- Kolibri-1：https://github.com/ggml-org/llama.cpp/pull/30673 （commit `62318cd57d`）
- 上一期欠账：https://github.com/ggml-org/llama.cpp/pull/30613 、https://github.com/ggml-org/llama.cpp/pull/30449
- nightly 标签 b11709–b11716（2026-10-17）：https://github.com/ggml-org/llama.cpp/releases

**证据等级：** 官方一手；`#30517`/`#29927`/`#29181`/`#30374` 的加速比均为 PR 作者在自述硬件与测试方法下测得，**不可跨卡型/跨后端外推**；`#30122` 仅有功能测试（128/128）**无性能数字**；`#30673` 为模型支持**无性能声明**。

---

## 6. 当前会话已形成的累计内容结论

1. **KV Cache 的竞争焦点已从「容量与分层」推进到「规则归属、生命周期与跨 rank 一致性」**：本期新增三条——(1) SGLang 发现 **MLA 在 TP rank 间共享存储键、而 Mamba checkpoint 是分片的**，导致不同 rank 的 Mamba 页**共用同一个键并互相覆盖**（`#43817`，与 `#43594` 对 NIXL 的修复同源）；(2) SGLang 发现 **write-back 型 HiCache 会把 KV 拷到 host 的时刻早于产生它的 forward**，从而把**陈旧 KV 供给另一个请求**（`#44800`，涉及 insert 时发布未备份 SWA window 与无 fence 的驱逐排空两条路径）；(3) vLLM 的 **EAGLE group 由「猜」改为 runner 上报的显式 draft-layer 标签**（`#61455`），修掉「投机解码**静默禁用** hybrid 模型前缀缓存」。加上前两期的 `KVCacheConfigBuilder`（`#53558`）、`page_unified` 布局、KVCacheManagerV2，可确认：**KV cache 的语义（谁拥有、谁可写、键怎么构造）正在被逐条显式化，而不再是内核实现的副产品**。
2. **「执行模型冲突」取代「单点 bug」成为最危险的缺陷类型**：本期 SGLang `#44220` 是最清晰的案例——**逐层异步加载（Python 层等待）与 CUDA graph 整体重放（无 Python 等待）在语义上不兼容**，于是图可能重放一个 load 仍在途的 batch，**静默产生错误 logits 并被内容寻址缓存备份**；而**现有逐位一致测试通过，只因 H2D 拷贝通常跑赢 replay**，且 `scheduler.py` 里相关 TODO 自 **2025-06** 挂到今天。同类结构风险适用于任何「异步加载 + 图重放」的组合（vLLM KV offload、TensorRT-LLM host offload、unified memory 的层传输）。
3. **「重复计算」与「无消费者/静默降级」仍是跨厂商通用根因，本期三度复现**：vLLM 博客（第 1 节）确认 Kimi K3 基线**在每个 GPU 上都算完整 router 分数与 latent 投影**，改为按 TP rank 切分后合并；SGLang 发现 **MegaMoE 的 shared-expert 融合开关默认开着，却因 checkpoint 是 [32,32] block-FP8 而静默回落**（`#44837`，代价是每步约 200 次多余 launch）；llama.cpp 发现**调度器为 host 输入的每个视图各自上传，逐层 reshape 被复制「层数」次**（`#30613`）。三者都不是算法问题，而是**「配置/形状/视图」层面的默认路径失配**。
4. **投机解码的风险面已从「接受率」扩展到「状态归属与图尺寸变化」**：本期 vLLM 三条同族修复——fixed-K proposer 在动态投机把 draft 数降到 0 时**硬断言崩溃**（`#46399`）、**更大的 MTP 图在 batch 收缩后读到陈旧共享 block-table 行并寻址到别的请求的 cache/state**（`#61174`）、EAGLE group 归属靠猜（`#61455`）；SGLang 则发现 **verify 在 CPU 侧截断之前推进 live state，以更短 cache key 捐赠后污染前缀复用**（`#43387`，KL 从 0.877 降到 0.000235）。**共同点：问题都出现在「投机 + 批/图尺寸变化 + 缓存复用」三者交汇处**，且**都不会通过端到端精度测试暴露**。
5. **MoE 推理的优化重心正在从「kernel 快不快」转向「launch 少不少、副本对不对」**：SGLang DSv4.1 的 profiling 给出量化证据——一个 16K token 的 eager prefill step 有 **1,812 次 launch、kernel 间中位空隙 41 µs**，仅 **query 重打包**就占 **33 ms / 322 ms（10%）**；而 **EPLB 搬动专家时漏搬 scale 副本**会让 GSM8K 从 **97.5% 掉到 72.5%**（`#44847`）。vLLM 侧对应的是 **SM100 FP4 DeepGEMM 用 prefix-sum scheduler 跳过 padding**（前一期 `#59044`）与 **EPLB 等专家 buffer 拷贝再发 NIXL READ**（`#62081`）。**「专家参数的副本一致性」已成为与路由算法同等重要的正确性问题。**
6. **量化结论必须绑定硬件、内核、布局与「是否静默回落」四要素**：本期 llama.cpp `#30517` 用**逐卡逐类型全表**说明同一低比特类型在 3090/A16 上提速 1.6–1.7×、在 4090/GB10 上基本持平；SGLang `#44114` 说明 **FP8 池下把 scratch 从 BF16 换成 FP8** 可减半 gather/attention 的 KV 流量，且**对 FP32 参考的误差到 64K 以上不随长度增长**；TensorRT-LLM `#19960` 则明确 **SM100/SM103 继续 DeepGEMM、只有 SM107 走 Rubin CuTe DSL**。**「同一 dtype 在不同引擎/设备/布局上不可互推」这一条，本期证据最充分。**
7. **「默认值变更」与「未验证合入」是性能回退的主要来源，且本期两侧都有反例**：vLLM `#61989` 复盘了一次**无性能数据合入的默认值变更**（`#56891` 把节点内 all-reduce 全改为 trtllm），造成 **v0.32.0 的 3–5% 解码回退**，靠 nightly 跳变 + 「只有 TP>1 变化」定位、再用强制后端 A/B 定量（5462 vs 5728 tok/s）；TensorRT-LLM `#19981` 的评审直接指出**作者报告的全模型负载中从未触发 MoE 回收事件**、L0 流水线全失败。**「机制存在」不等于「收益存在」，「默认开启」不等于「路径生效」。**

---

## 7. 今日可执行清单

1. **对任何「异步逐层加载 + CUDA graph」的组合做一次时序审查**：确认图重放前**显式等待最后一片层传输落地**（SGLang `#44220` 的做法：在 `PrefillCudaGraphRunner.execute` 检查层传输计数器的活跃消费者）。自查信号：**测试通过但结果偶发不一致**、或「只在 H2D 较慢的机器上出错」。特别注意**现有逐位一致测试可能只因拷贝跑赢 replay 而通过**。
2. **HiCache / 多级缓存部署检查三件事**：(a) write-back 模式下 **D2H 拷贝必须排在产生 KV 的 forward 之后**（`write_fence_stream`，`#44800`），否则会把陈旧 KV 供出去；(b) **MLA 模型的 Mamba/分片 state 必须按 TP rank 加键**（`#43817`），否则跨 rank 覆盖；(c) 旧的未加 rank 的 Mamba 条目**必须视为不可复用**（归属不明），升级时应主动清理。
3. **投机解码做「批/图尺寸变化」专项回归**：构造「大 MTP 图 → batch 收缩 → 继续 decode」的序列，确认共享 block-table 行已被清理（`#61174`）；确认动态投机与 `ngram_gpu`/`suffix`/`medusa`/`extract_hidden_states` 的组合**已改为禁用而非崩溃**（`#46399`）；对 hybrid（Mamba/GDN）模型**实测前缀缓存命中率是否非零**（`#61455`，此前会被静默清零）。
4. **检查所有「融合开关」是否真的生效**：SGLang 的 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_FUSE_SHARED_EXPERTS` 默认开但会因 **block size 不匹配**静默回落（`#44837`）；llama.cpp 在 pinned 内存不足时会**静默回退 pageable**（`#30449`）；vLLM 的 `#61989` 是**默认值选错后端**。做法：在启动日志中显式打印「哪个后端/哪条路径被选中」，并对关键开关做一次 **A/B 而非默认信任**。
5. **PP 部署的 CPU offload 池定尺寸改为显式**：vLLM `#61674` 显示自动定尺寸会让各 pipeline stage 得到 **25/25/23** 三个不同大小，产生 **40 次越界 native copy**（基线「完成响应且无段错误」）。建议显式设置共享 `num_cpu_blocks`，并在升级后**核对各 stage 块数与最高目标 ID 是否一致**。
6. **单节点 NVSwitch 系统复核 FlashInfer all-reduce 后端**：确认节点内组使用 **mnnvl + `allow_trtllm_fallback`**（`#61989`），并复跑一次 TP>1 的 decode 基准；若此前在 v0.32.0rc 上观测到 3–5% 回退，可直接归因。
7. **基准测试必须包含「多轮之后重新长 prompt」场景**：llama.cpp `#30374` 的 SYCL 回退**只在首次 decode 之后出现并持续整个会话**（Q4_0 −40%、Q8_0 −25%），**只测首轮 PP 或单轮问答永远发现不了**。同理，KV cache 复用类 bug 也只在多轮后才显形。
8. **DSv4.1 用户注意默认值变更**：SGLang `#44403` 已**默认开启 decoder bounded replay**，该路径**数值上与全 prefill 不同、且不支持 prompt logprobs**；需要严格复现或需要 prompt logprob 的场景应显式加 `--disable-decoder-bounded-replay`，并把 AIME/DeepSWE/MMMU-Pro 的配对差异（官方：AIME 各 469/480，DeepSWE 差 1 题，MMMU-Pro 差 5 题）纳入自己的验收基线。
9. **本期已补齐上一期全部欠账**：TensorRT-LLM `#18599`（**无性能数字**，仅「fmha lib/autotuner 更新 + 移除 H96 旧 WAR」）、`#19981`（**评审确认无真实模型收益证据**）、`#20290`（GVR V2 top-K 开销 **208–221 µs → 1.05–2.76 µs**）、llama.cpp `#30613`（**消除 host 输入视图的逐层重复上传**）、`#30449`（**pinned host buffer 4 GiB 上限**，修 Windows 上 56 GiB 分配失败导致的静默降级）。**下一期无需再补。**

---

## 8. 社区与视频资料

- **HN Algolia 公开 API 本期可访问（HTTP 200）。** 在 2026-10-17 00:00–24:00 UTC 窗口内按 “vllm”“sglang”“llama.cpp”“tensorrt”“KV cache”“speculative decoding”“quantization inference”“GPU inference”“inference” 检索，**与本期主题直接相关的条目接近于零**，如实记录如下：
  - **“Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression”**（HN 2026-10-17T15:11:13Z，**1 point / 0 comments**，https://arxiv.org/abs/2609.36322 ）——本期唯一与 KV Cache 主线相关的条目。**arXiv 摘要页（官方一手页面）显示：提交于 2026-09-28**，作者发现**分块 KV 压缩引入「相位」这一新位置坐标**（token 相对压缩窗口边界的位置），使**同一信息在不同相位下检索难度不同**；在大型开放权重模型上，**长上下文检索准确率跨相位可相差最多 40 个百分点**，而**平均基准分会把这种周期性弱点掩盖掉**。**注意：该论文为预印本，未经同行评审；本档案仅核验其摘要页，未核验实验细节，故整体标注为「二手线索 / 未核验」**。它与本期 SGLang 的 chunked/compressed KV 工作（`#44920`、`#44837`）方向相关但**无直接因果**。
  - “Apple releases RSI-Inference: evaluating agents for self-improving LLM inference”（HN 2026-10-17T19:23:11Z，**2 points / 0 comments**，来源为 **X/Twitter 链接**）——**未核验**，无可访问的一手发布页，**不构成事实**，仅记录「有该讨论」。
  - 其余命中（“GoRoute – LLM Gateway in Go”“InBox Wall”“Tensor Golf”“Three LLM judges couldn't tell an AI agent's real memory…”“Are You Sure Your LLM Is Sure?”等）**均非推理引擎性能主题，不计入本期结论**。
  - **检索未命中：** “sglang”“llama.cpp”“speculative decoding”“quantization inference”“GPU inference” 在窗口内**均返回 0 条**，说明当日社区讨论未聚焦推理引擎，**不构成「无动态」的结论**（官方提交仍在密集进行，本期四家合计 118 条）。
- **官方博客：** vLLM 官方博客**本期无 2026-10-17 新文**（最新仍为 2026-10-16 三篇）。其中《Hyper Optimizing Kimi K3 in vLLM》已作为**补录**列入本文第 1 节；同日另一篇《Serving Reflection Beam on vLLM from Day 0》（https://vllm.ai/blog/2026-10-16-reflection-beam ）为**模型 Day-0 支持**类文章，**不属推理优化主线，仅列线索不展开**。
- **Reddit 未纳入本期采集范围**（往期记录其 API 受限），本期不据 Reddit 内容形成结论。
- **视频资料：** 本期未核验到可确认发布日期与内容的官方视频，故不列入。llama.cpp `#29584` 的 PR 页面内嵌了一段屏幕录制（`Screen.Recording.2026-10-16.at.12.24.58.mp4`），属**功能演示**而非技术讲解，不计入视频资料。
- **技能检索：** 本期沿用 WorkBuddy 技能检索结论（候选为「AI 热点日报」「GitHub AI趋势追踪」「GitHub热门项目」「开源雷达 · GitHub 热门榜」「LLM推理加速方案制定与优化配置」等**媒体式聚合或方案输出**类技能），与本任务所需的「**官方提交/发布 API 按日期窗口采集 + 证据分级热点归档 + hotspot 目录与远端同步**」不匹配，**未安装任何外部技能**。已加载的 `agent-browser`、`find-skills` 本期**未产生实际调用需求**（全部数据源均可直接 HTTP 访问；PR 正文改用页面直抓后，也不再需要浏览器自动化）。

---

**归档状态：** 已按 `hotspot/2026-10/` 月度目录归档。
**证据边界：** 事实与观点已分开；官方提交/PR 中的性能数字（842 tok/s / 1.31–1.92× / 1.8× P90、1.22–2.16× / 4.4×、KL 0.877→0.000235、GSM8K 72.5→97.0、33 ms/322 ms、208–221 µs→1.05–2.76 µs、1.59–1.71× / 1.20× / 1.925× 等）均按原文配置范围记录，**不推导为普遍结论**；TensorRT-LLM `#19960` 的 36% 与 22.6 GiB 为作者自述且无端到端对比，`#19981` 经评审确认**无真实模型收益证据**，`#18599` **无性能数字**；社区条目一律标注为二手线索且未核验；**本期与上一期（覆盖 2026-10-16）连续无空档，且上一期列出的 5 项欠账已全部补齐**。
