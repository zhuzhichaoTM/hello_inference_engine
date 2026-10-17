# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-17（北京时间）
- **关注窗口：** 2026-10-16 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-16 08:00 – 2026-10-17 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-16 UTC 窗口，分别返回 49 / 79 / 36 / 29 条，均 HTTP 200）、四家官方 releases API，以及 vLLM 官方博客原文（2026-10-16 发布）。本期与上一期（2026-10-16 档案，覆盖 2026-10-15）**连续无空档**。**限制说明：** 本期在批量抓取 GitHub PR 正文时触发未认证 API 速率上限（60 次/小时，剩余 0），因此 vLLM / SGLang 的 PR 正文细节完整，**TensorRT-LLM 与 llama.cpp 的条目仅基于官方提交标题**（仍为官方一手来源，但未取到 PR 正文的性能数字与适用范围声明），已在对应小节标注。HN Algolia 公开 API 本期可访问；Reddit 未纳入采集范围；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** 本期四家**均无新的正式版本发布**。vLLM 最新公开 release 仍为 v0.31.0（2026-10-05）；SGLang 仍为 v0.5.21（2026-10-02）；TensorRT-LLM 仍为 v1.4.0rc0（prerelease，2026-10-14）；llama.cpp 正式版仍为 v0.7.0（2026-10-15），窗口内滚动 nightly 构建标签为 b11701 / b11700 / b11699 / b11696。

---

## 1. vLLM 官方博客：Kimi K3 在 AMD MI355X 上追平 B300 每 GPU 吞吐（本期头号一手材料）

**主题**  vLLM 官方博客发布《Scaling Kimi K3 on AMD Instinct MI355X: Matching NVIDIA HGX B300 Throughput per GPU on AgentX》，由 **AMD 与 Embedded LLM** 联合撰写，用六类优化（缓存划分、MoE 去重、访存融合、MLA 汇编路径、KDA 并行化、投机深度选择）把 8 卡 MI355X 的 Kimi K3 服务吞吐做到**超过同卡数 NVIDIA HGX B300**。

**内容详情**  官方博客原文要点：
- **发布信息：** 2026-10-16 发布，作者 AMD and Embedded LLM，标签 `models` `performance` `rocm` `amd` `speculative-decoding`，24 min read。数据快照为 SemiAnalysis InferenceX 于 **2026-10-09** 检索；提交日期 AMD 为 **2026-09-30**、NVIDIA 为 **2026-08-16**。
- **头条结果（8 卡对比，50 tokens/s/user P90 交互性）：**
  - 每 GPU 总吞吐 **约 7,567 tokens/s vs B300 的 6,083 tokens/s，高 24.4%**；
  - 同一成本模型下**每美元 token 数高 87.4%**（18.16M vs 9.69M），建模 TCO **$1.50 vs $2.26 /GPU-hour**，50-token 目标下**每 token 成本低 46.6%**。
  - **口径（原文）：** 交互性 = P90 响应 inter-token latency 的倒数，**不含 TTFT**；总 token 包含**复用的输入 token 与生成输出**。
- **硬件规格对比（每 GPU，官方已发布规格）：** 密集低精度矩阵峰值 MI355X **10.1 PFLOP/s MXFP4** vs B300 **13.5 PFLOP/s FP4**；两者均为 **288 GB HBM3E / 8 TB/s**。即 **AMD 峰值算力更低但服务吞吐更高**。
- **工作负载（AgentX）：** 以合成 token 回放真实编码 Agent 会话轨迹，保留请求长度、时序与前缀共享模式；无限缓存下前缀复用率**超过 96%**；投机运行使用**固定合成接受参考**。
- **Kimi K3 结构（原文）：** 24 个 MLA 层、69 个 KDA 层、96 个 MLA query heads（TP8 下每 GPU 12 个）、92 个 MoE 层、**每 token 从 896 个专家中选 16 个**。
- **六项优化（逐项，含官方数字）：**
  1. **Cache / DCP（Decode Context Parallelism）：** TP8 下每 GPU 只算 12 个 query heads 却要存**全部**压缩 KV 历史；DCP8 让每 GPU 只存 1/8 缓存位置，解码时交换 queries、本地算注意力、按 softmax 权重合并部分结果。PR **#51705** 把 ROCm AITER DCP 路径扩展到 multi-token verification。**内存测算：** 142,000-token 对话的 MLA payload ≈ 24 层 × 142,000 位置 × 576 FP8 bytes ≈ **1.96 GB/GPU**，DCP8 降至 **约 245 MB**。DCP 只切分 MLA 历史，KDA 状态保持原有跨 head 划分。
  2. **Cache / CPU offload 恢复点对齐：** 恢复点必须**跨所有必需层兼容**。示例：MLA 历史已到 token **3,072**，但最新兼容 KDA checkpoint 在 **2,048** → 请求必须从 2,048 恢复并**重算 1,024 个前缀位置**。PR **#54736** 把 attention-block tail 与 KDA checkpoint 一起保存，使恢复位置对齐。
  3. **MoE / 消除跨 GPU 重复计算：** 原 ROCm 路径下早期 reduction 与 normalization 让 8 个 GPU 拿到相同 latent 输入且各持完整权重，**同一投影被算 8 次**。PR **#51253** 把**输出坐标**划分给各 GPU，每卡只把自己的 routed 输出加到 shared-expert partial 上，由**已有的 all-reduce** 组装，**不新增集合通信**。实测：8 卡 MI355X / TP8、并发 8–128、随机 8K-in/1K-out、关闭 prefix caching，输出吞吐 **+3.05%–4.91%**。
  4. **MoE / 释放显存换缓存（PR #59591）：** 只加载各 rank 实际使用的行，TP8 的 92 个 MoE 层上每 GPU 释放**约 3.85 GiB**。独立 AgentX 实验：并发 32/48 时吞吐提升 **<1%**，并发 **70** 时提升 **28.9%**，prefix-cache 命中率 **74.8% → 80.1%**。**官方明确该改动晚于头条结果所用镜像**，且未报告重复运行的不确定性。
  5. **访存 / 融合算子与去冗余拷贝：** PR **#50654** 把单 token KDA 路径从**3 个 kernel + 2 次输出拷贝**融合为 1 次 launch；PR **#50593** 将 AttnRes 与 RMSNorm 合并（临时混合值留在寄存器，running residual 保持 BF16 rounding，混合值全程 FP32），在 8 卡 MI355X / TP8 / 100K-in/1K-out / 关闭 prefix caching 下输出吞吐 **+14.2%**，长 prefill 延迟基本不变；PR **#58566** 指出 ROCm 矩阵乘 dispatcher **在选择 kernel 之前就打包输入**，选中 fallback 时拷贝无人消费——profile 显示**每个 decode step 有 69 次无谓拷贝**（每个 KDA 层一次），单次仅搬运 1,792 bytes 却耗时约 **4.8 微秒**（launch 开销主导），被移除的拷贝占每 step copy-kernel 时间 **0.339 ms**。
  6. **MLA / 用 AMD AITER 汇编加速长上下文解码（PR #50578）：** TP8 下每 GPU 12 个 query heads **填充到 16** 以启用 AITER 汇编解码路径。**每输出 token 时间从 43.8 ms 降至 23.6 ms**（32K/1K），1K/1K 时 22.9 → 22.8 ms（几乎不变）。配置：8 卡 MI355X、TP8、无 DCP、并发 1、MXFP4 权重、BF16 attention。原理：每个 query head 独立对缓存做注意力，**增加 head 不改变原 head 的 softmax**，只填充 query、不动已存的压缩 KV；被检视的汇编策略选择 **16 个分区**。
  7. **KDA / prefill 与投机验证并行化：** PR **#54038** 用有序 scan 确定各 token group 起始状态后并行 replay，**代价是把融合 chunk 路径从 2 次 launch 扩到 4 次**（序列或 head 太少、并行度不足时收益可能不抵额外 pass）；PR #56036 显示 Fused KDA 在 4K 与 16K 输入下比 Triton **提升 5.3%–5.6% 输入吞吐**。PR **#58045** 把每 head 的 128 个 value columns 切成 **4 个 32-column tile** 时只启动 **48 个 workgroups**（256 个 compute units），改为 **four-column tiles** 后提升到 **384 个 workgroups**，单卡 graph-replay kernel 测试中每层时间 **16.20 → 5.61 微秒（2.89×）**。
  8. **投机解码 / draft 深度选择：** AgentX 固定接受参考下，**3 drafts → 每周期 3.00 期望 committed tokens**；**4 drafts → 3.36**（含 recovery 或 bonus token），即**每周期多 12% committed 输出**。**官方阈值表述：**「四个 draft 只有在完整周期耗时比三个 draft 增加不到 12% 时才能提升吞吐」。公开 AMD recipe 用 **3–7 drafts + DCP1**，并在更大的 DCP8 并发设置下**禁用投机**。
- **量化：** 权重 **FP4**、KV cache **FP8**、attention **BF16**；PR **#53940** 的 **A4W4** 把 activations 从 8 bit 降到 **4 bit**（权重保持 4-bit）。KDA prologue 使用 AMD 原生 **FP32 matrix instructions（16×16×4）** 做 64×64 within-chunk 依赖矩阵求逆，保留 FP32 输入与累加、仅把完成的逆矩阵舍入为 BF16；官方用标量例子说明为何必须先保留 FP32：`256.5 − 256 = 0.5`，若先舍入到 BF16 则 `256 − 256 = 0`。
- **并行与配置：** 所有对比配置均为 8 GPU / TP8；AMD 用 **SimpleCPU offload**、NVIDIA 用 **Mooncake offload**；host 缓存预算分别 **224.875 GB 与 281 GB per GPU**；vLLM 版本 AMD `36768d1bf` / NVIDIA `5894fdf98`；回放数据集 `cc-traces-weka-062126`，固定版本 AIPerf harness，排除 warmup，在固定合成接受曲线下 profiling **1 小时**。正在推进 prefill/decode 解耦。
- **贡献规模：** 该工作涵盖来自 AMD、Embedded LLM 及社区的 **50 余个已合并 AMD/ROCm 相关 vLLM PR**，外加 AITER 开发、集成、review 与 benchmarking。
- **官方给出的三条「下一步该优化什么」检查法：** (1) **Compare ranks**——是否有多个 GPU 在集合通信前算了同样的结果？(2) **Follow buffers**——每个临时 tensor 在所选路径上是否真有消费者、数据能否留在片上？(3) **Count independent work**——实际的 heads / tokens / tiles 是否产生足够 workgroup 分散负载。
- **官方明示的限制（原文要点）：** 公开曲线比较的是**完整服务栈**，而每个优化旁边的测量来自**独立测试**，只在其所述配置下解释单个机制；#59591 **晚于头条结果所用镜像**；高并发下更大增益**与缓存压力一致**，官方建议「在等缓存容量下比较两种实现以隔离该效应」；投机解码的 **12% 是成本预算，本期未测量完整周期增长**；交互性**排除 TTFT**，且「并发、DCP 与 draft 深度可能在括号运行的两次之间变化」；闭环回放可能产生**不同的已完成请求组合**；**下载的数据集 revision 未记录**；**部分资本与利用率假设未公开**；LDS 预算中 workgroup 请求 **79.5 KiB 向上取整到 80 KiB**，为 AMD CDNA 4 计算单元 **160 KiB LDS 的一半**；完整 AMD 免责声明（按 AS IS 提供、不作担保）。

**应用场景**  AMD MI355X / ROCm 上的长上下文 Agent 服务、DCP 与 TP 组合的 KV 内存规划、MoE 输出坐标划分以避免跨 rank 重复计算、MLA 汇编路径与 query head padding、KDA 类线性注意力模型的 prefill/verify 并行化、投机深度与交互性目标的联合调参、TCO 与每美元 token 的成本决策。

**研判**  **事实：** 这是本期唯一一篇官方一手博客，也是四家引擎中**唯一给出跨厂商等卡数服务对比**的材料，且官方主动列出了 12 项限制（含数据集 revision 未记录、资本假设未公开、per-PR 数字来自独立测试）。**观点：** 三点——(1) 本期最具可迁移价值的是**三条检查法**而非 24.4% 这个数字：其中「多个 rank 在集合通信前算了同一份结果」正是 #51253 的根因（原路径下 8 卡各算一遍同一投影），「临时 tensor 是否有消费者」正是 #58566 的根因（dispatcher 先打包再选 kernel，fallback 时拷贝无人读）——这两类问题**与厂商无关**，在任何 TP 部署上都可以用同样的方法自查；(2) 24.4% / 87.4% 必须严格限定在「8 卡 / TP8 / AgentX / 50 tokens/s/user P90 / 各自已发布 offload 序列内插值」这一条件，官方自己承认括号运行之间并发与 draft 深度可能不同，因此**不能读作硬件优劣结论**，更不能外推到 prefill 密集或低前缀复用负载；(3) 真正的工程信号是「**同一份 MLA 压缩历史在不同并行策略下的可复现性**」——DCP 切分位置、CPU offload 的恢复点对齐（3,072 vs 2,048 的例子）、以及 3.4 节 KDA 状态精度，说明长上下文 Agent 服务的瓶颈已经转移到**状态一致性与恢复点语义**，而不是单纯算力。

**来源**
- vLLM 官方博客：https://vllm.ai/blog/2026-10-16-kimi-k3-mi355x （发布日期 2026-10-16）
- 相关 PR：#51705（AITER DCP multi-token verification）、#54736（attention-block tail 与 KDA checkpoint 同存）、#51253（MoE 输出坐标划分）、#59591（按 rank 加载专家行，释放约 3.85 GiB/GPU）、#50654（KDA 单 token 路径融合）、#50593（AttnRes × RMSNorm 融合）、#58566（避免未被消费的输入打包）、#50578（MLA query head padding 到 16）、#54038（KDA fused prefill 并行化）、#58045（KDA verify four-column tiles）、#53940（A4W4）
- 上一期同源材料（KV 传输与投机解码，早于窗口）：https://vllm.ai/blog/2026-10-14-nixl-push-connector 、https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator

**证据等级：** 官方一手（官方博客，含 AMD 免责声明）；24.4% / 87.4% / 2.89× / 14.2% / +3.05–4.91% 等均为官方在各自独立配置下测得，**不可外推**；其中 #59591 官方自述晚于头条镜像。

---

## 2. vLLM：10-16 提交（49 条）——KV cache 规划可插拔化、DP wave 唤醒归属修正、Kimi K3 GB300 专用解码

**主题**  vLLM 在 10 月 16 日的 49 条官方提交集中在：**KV cache 规划的可插拔化**（新增实验性 `KVCacheConfigBuilder`）、**DP wave 唤醒职责从 front-end 收归引擎**（修复一个 DP 挂起）、Kimi K3 在 GB300 上的 **CuTeDSL 专用解码（1.5–2×）**、弹性 EP 的外部 LB 模式、投机解码 bonus-logit 冗余计算消除、ROCm 侧 DSpark 自适应验证与 FlashInfer 0.7.2。

**内容详情**  官方提交 API 与 PR 正文显示（节选可核验项）：
- **KV cache 规划抽象（本期最具结构性）：**
  - **`[Core][Feat] Pluggable KVCacheConfigBuilder`**（`523d74e2ac`，PR `#53558`，RFC `#62100`，merged 2026-10-16T23:16:48Z）：vLLM 原先用 core 里**单一启发式** `get_kv_cache_groups()` 决定所有模型的 KV cache 分组，导致新混合模型 / drafter / cache 精度 / 平台每次都要改 core，且 **vllm-ascend 直接 monkey-patch core 规划**以使用自己的 DSV4 plan。本 PR 新增**实验性**可插拔 `KVCacheConfigBuilder`（`vllm/v1/core/kv_cache_config_builder.py`）：平台选择 builder 决定分组与放置，模型可返回自己的 groups，**core 保留单一规划流程并校验每个 plan**。官方明确「接口可能变化，尤其是当 #61452 改变 KV cache group 数据结构时」。
- **数据并行（DP）正确性：**
  - **`[DP] Engines own wave wake-ups; drop the front-end FIRST_REQ`**（`a94c315711`，PR `#59559`）：修复一个 DP 挂起。原设计下 front-end 只要认为引擎空闲就发 `FIRST_REQ`，coordinator 随后广播 `START_DP_WAVE`，收到请求的 rank **假定 peer 已被唤醒**——但**仍处于 paused 的 rank 会丢弃 `START_DP_WAVE`**。PR 给出双 API server（A/B）+ 双 rank（X/Y）的完整竞态序列：B 向 X 发 ADD 并给 coordinator 发 `FIRST_REQ` → coordinator 向仍 paused 的 Y 发 `START_DP_WAVE`，Y 丢弃 → A 恢复两个 rank，但两者都无工作、保持空闲 → B 的 ADD 到达 X，其 wave 已是最新，X 不再触发唤醒。改后**由引擎独占启动 DP wave**，front-end 不再发送 `FIRST_REQ`。
- **Kimi K3 专用解码路径：**
  - **`[Perf] Kimi K3 Specialized Decode: CuTeDSL Optimized GB300 (1.5-2x)`**（`642c18d64c`，PR `#61867`，merged 2026-10-16T02:50:42Z）：NVIDIA 开发的 GB300 专用实现，支持 **TP8 与 TP16**、**NoSpec（最高 B64）与 DSpark 验证（3–7 draft，最高 64 tokens，DL7 为 B8、DL3 为 B16）**、以及 **DCP=TP**（也支持关闭 DCP）。通过 `--additional-config '{"decode_backend": "latency_specialized"}'` **选择性启用**，改动局限在 `vllm/models/kimi_k3/nvidia/`。**官方声明（FP8 KV cache、FP32 SSM cache、模型提供方权重）：** TP16 / ISL 1 / 无投机时，B1–B8 上 decode step **比 stock 快 1.79–1.92×**（B1 为 **3.90 vs 7.38 ms**）；TP16 / DSpark DL7 / 合成接受 6 时 **decode tok/s 提升 1.53–1.71×**。
- **弹性专家并行（EEP）：**
  - **`[Feature] Add External LB Mode Support for Elastic EP Scaling`**（`dd9e07a25f`，PR `#43202`，RFC `#42515`，merged 2026-10-16T01:29:02Z）：部分在线服务部署使用**外部负载均衡器 + 外部管理的 DP rank 生命周期**，此时 vLLM 不应创建/删除新的 Ray actor；本 PR 为**外部启动的 rank** 提供加入既有 EEP scale-up / scale-down 转换的控制路径，复用既有 worker 侧状态机。
- **投机解码：**
  - **`[Perf][Spec Decode] Skip bonus-logit reduction for short drafts`**（`663e535640`，PR `#61451`）：改用**请求边界**（而非 `local_pos >= num_speculative_steps`）识别 bonus logit，使 `num_drafts < num_speculative_steps` 的请求不再在 bonus 行上执行**从未被读取**的全词表 max/sum-exp 与 residual-mass reduction（`_compute_local_logits_stats_kernel` / `_compute_local_residual_mass_kernel`）；`num_speculative_steps` kernel 参数被 `num_logits` 取代。为 #60945 的后续。
  - **`[ROCm][Kimi-K3] Support DSpark adaptive verification on MI355X`**（`5bf3d9f665`，PR `#59824`）：依赖 **#55504**（零 draft 投机步上保留 GDN recurrent state；否则 1-token/请求的 varlen capture dummy 会记录非投机 KDA 路径）与 **#56831**（**跨 TP rank 同步自适应 DSpark 决策**）。PR 明确指出：**若不同步，rank 本地 confidence 可能在不同 TP rank 上选出不同 draft 预算或请求边界，导致集合通信错配并挂起**；Kimi-K3 运行在 TP8，**并发 ≥ 8 时约 97% 的 step 会触发裁剪**。
  - **`[Attention][Spec Decode] Enable varlen Mamba2 decode for adaptive verification`**（`3dc6def360`，PR `#53036`）：自适应验证需要**每请求 query 长度**的 Mamba decode batch；审计结论是 Mamba2 decode kernel 已能处理来自 device offsets 的 ragged batch，故本 PR **仅做接线**；打开 Mamba2 的 `supports_device_cpu_query_lens_mismatch`（ReplaySSM 仍退出），保留 `UNIFORM_BATCH`，并用 `get_varlen_cudagraph_max_query_len = max_decode_query_len` 界定 varlen FULL decode 图。
- **上下文并行与注意力：**
  - **`[Feature][PCP] Add DeepSeek-V4.1-Flash support`**（`fa75d3456d`，PR `#59857`）：为 Model Runner V2 + FlashMLA 稀疏注意力新增**纯文本 DSv4.1-Flash 的 Prefill Context Parallelism**；**有意限定为同址单模型实例**，**不支持 P/D 解耦与 KV 传输**；实现保持 compressed KV、indexer、SWA、Engram lookback、MoE routing 输入与 padding 在 PCP rank 间一致，并用 gathered predecessor rows 闭合被 PCP chunk 切开的 ratio-2 对。对应 DSv4.1 ratio-1/2 cache 路径，**不含 V4 的 C4/C128 或 MegaMoE 工作**。
  - **`[Perf][DSv4.1] Split fused_wqa_wkv by token across TP ranks on SM90`**（`05ad0a8ace`）；**`[ROCm][Perf][Kimi-K3] Use AITER Gluon chunk_kda for KDA prefill`**（`28e92eeff2`）；**`[ROCm][Perf] Any-shape low-latency MoE router GEMM on MI355`**（`b466fbacd2`）；**`[Model][Hardware][AMD][Kernel] Part 2/2 → Enable e2e QK Norm + RoPE + KV Cache runtime fusion for Qwen3-30B-A3B on ROCM_ATTN`**（`ef38e8d006`）；**`[ROCm][Qwen4Exp] Support HC and MoE sequence parallelism on ROCm`**（`29345f42a6`）；**`[ROCm][fusion] support ROCm aiter fusion of per tensor quant + rmsnorm`**（`cc2e4fecf2`）。
- **MoE 与量化：**
  - **`[Perf][MoE] Skip padded work in SM100 FP4 DeepGEMM experts`**（`68ebea789a`，PR `#59044`）：Blackwell FP4 DeepGEMM MoE 原先在 GEMM 与 activation 中**都对 padding 做计算**（主要影响 DeepEPv2+ 下的 decoder）。改用 **DeepGEMM 的 prefix-sum scheduler**，让 packed SiLU+mul+FP8-quant kernel 只处理 **live rows**；端点 = 对齐后的专家起点 + 实际 token 数，下一专家从前一端点向上取整到 GEMM 对齐处开始，**无需 count readback、额外 launch 或修改 DeepGEMM 源码**；每行布局元数据从 **112 KiB 降到 128 bytes**（32 个本地专家、28,672 个已分配行）。
  - **`[Dependency] Upgrade FlashInfer to 0.7.2 and restore KDA auto backend`**（`af28c171f5`，PR `#61913`）：把 `flashinfer-python==0.7.2` 固定进 `requirements/cuda.txt`，并让 `_flashinfer_kda_prefill` 传 `backend="auto"`（#60948 曾固定 0.7.1 并强制 `cute-dsl`）。**PR 自述：未运行 GPU 测试、端到端服务、模型评测与基准，也未做任何精度或性能声明**（该主机无 NVIDIA GPU）。
  - **`[Bugfix][MoE] Keep the gate's hash_indices_table out of RoutedExperts parameters`**（`a956a8126e`）；**`[Bugfix][MoE] Restore TP-sharded routed-expert capture for DP=1`**（`a8be1500ba`）；**`[Bugfix] remove duplicate hash-routing capability method`**（`8f93c64866`）；**`[Bugfix][Kernel] silu_and_mul_quant: run the vectorized loop`**（`fb3fec970e`）；**`[XPU][Linear][MXFP4] enable torch as mxfp4 linear backend on xpu`**（`07f3857fb9`）。
- **依赖与工程：** `[Dependency] Upgrade TokenSpeed MLA to 2.0.3`（`e365792997`）；`[XPU][Docker] Bump nixl to 1.5.0`（`3ddc6702a8`）；`[Perf] Share Triton JIT warmup compilation pool across registered kernels`（`de7eeaa87b`，PR `#61422`）；`[Refactor] Remove dead code multiple places`（`8e25bbca19`）；`[Build] Remove unused tools/install_deepgemm.sh`（`61265`）；`Update DeepGEMM commit`（`bee95149e2`）；`[Bugfix] Keep VLLM_RAY_BUNDLE_INDICES out of the compile cache key`（`9731f30a03`）。
- **服务与前端正确性（本期较集中的一类）：** `[Bugfix][Mooncake] Fix MultiConnector stats and proxy SSE responses`（`9a16d43ef0`）；`[Bugfix][ROCm][GLM-5.2] Stop forcing skip_final_all_reduce on AMD DSA model path`（`8a6c0449b3`）；`[Bugfix][Model] Honor quantization config in Inkling MTP`（`76081b9ee4`）；`[Bugfix][Parser] Honor a prompt-closed reasoning block in non-streaming parse and reasoning token counts`（`ae8f41238c`）；`[Bugfix][Structured Output] Mask grammar-invalid implicit reasoning ends`（`4ac1f8d2ac`）；`[Frontend] Resolve logprobs on the parsed streaming derender path`（`024c5188bf`）；`[Bugfix][Frontend] Keep the parser's tool call IDs in batch derender`（`64175d3827`）；`[Bugfix][Frontend] Honor the constructor chat template in offline scoring`（`4ab99ff136`）；`[Bugfix] Honor code_revision in config-based tokenizer loading`（`49cb2bbf71`）；`[Bugfix] Honor OPT token embedding width for prompt embeddings`（`92dc3dd393`）；`[Bugfix][Multimodal] Honor RGB PNG transparency when loading images`（`2268d83bd9`）；`[Bugfix][LoRA] Fix garbage/NaN lora_expand output on SM90 with fp32 i…`（`523d74e2ac` 同批）。
- **其他：** `[Perf][Parser] Stream MiniMax M2 tool argument values incrementally (16 KB call: 5.9 s → 26 ms)`（`b088eba944`）；`[Frontend] Add the Beam reasoning and tool parsers`（`153a17166e`）；`[Kernel] Add router GEMM kernels for (5120, 256) and BF16 output to ll_bf16_gemm`（`596e530675`）；`[CI][AMD] Move 17 test groups from MI300 to MI355 and test large DeepSeek/Nemotron evals on AITER`（`706791d516`）；`[ROCm][CI] Move Spec Decode Speculators + MTP to MI355`（`e7a5bdb454`）。

**应用场景**  多模型/多平台的 KV cache 分组策略定制（含 Ascend 等平台不再 monkey-patch core）、大规模 DP 部署的 wave 唤醒正确性、GB300 上 Kimi K3 的低延迟解码、外部 LB 场景下的弹性 EP 扩缩、长 draft 与短 draft 混合负载的投机采样开销、PCP 下的 DSv4.1-Flash 长 prefill、SM100 FP4 MoE 的 padding 开销治理。

**研判**  **事实：** 本期 vLLM 的结构性改动有两处——**KV cache 规划从 core 单一启发式改为可插拔 builder**（且 core 保留校验），以及 **DP wave 的启动权从 front-end 收归引擎**；两者都伴随明确的动机（前者是 vllm-ascend 的 monkey-patch，后者是完整的双 server / 双 rank 竞态复现）。**观点：** 四点——(1) `KVCacheConfigBuilder` 是本期最值得跟踪的一项：官方措辞为**实验性**且预告接口会随 #61452 变化，因此**不宜据此刻实现有平台插件**，但它把「KV cache 分组/放置」正式定义为**平台与模型可协商的契约**，方向与 TensorRT-LLM 的 KV cache V2、SGLang 的 `page_unified` 一致，即各引擎都在把 KV 布局从内部启发式升格为一等公民；(2) `#59559` 是典型的「**唤醒信号与暂停状态竞态**」——它的价值不在性能而在**避免静默挂起**，此类 bug 极难通过精度测试发现，建议在任何 DP>1 部署中关注该修复；(3) `#61867` 的 1.5–2× 是 NVIDIA 为 GB300 写的**专用路径**（`latency_specialized`），需显式开启，且其声明绑定 FP8 KV + FP32 SSM cache + 特定 batch 区间，**不应视为通用收益**，但它是「**同一模型在同一厂商内做延迟特化实现**」的一个明确案例；(4) `#61913` 的 PR 自述明确写了「未运行 GPU 测试、未做任何精度或性能声明」——这类**无验证的依赖升级**在升级时应视同新引入变更，而非安全补丁。

**来源**
- vLLM 官方提交 API（2026-10-16 UTC，49 条）：https://api.github.com/repos/vllm-project/vllm/commits?since=2026-10-16T00:00:00Z&until=2026-10-17T00:00:00Z&per_page=100
- Pluggable KVCacheConfigBuilder：https://github.com/vllm-project/vllm/commit/523d74e2ac （PR #53558，RFC #62100）
- DP 引擎独占 wave 唤醒：https://github.com/vllm-project/vllm/commit/a94c315711 （PR #59559）
- Kimi K3 GB300 专用解码：https://github.com/vllm-project/vllm/commit/642c18d64c （PR #61867）
- 弹性 EP 外部 LB 模式：https://github.com/vllm-project/vllm/commit/dd9e07a25f （PR #43202，RFC #42515）
- 短 draft 跳过 bonus-logit reduction：https://github.com/vllm-project/vllm/commit/663e535640 （PR #61451）
- MI355X DSpark 自适应验证：https://github.com/vllm-project/vllm/commit/5bf3d9f665 （PR #59824）
- varlen Mamba2 decode 支持 AV：https://github.com/vllm-project/vllm/commit/3dc6def360 （PR #53036）
- DSv4.1-Flash PCP：https://github.com/vllm-project/vllm/commit/fa75d3456d （PR #59857）
- SM100 FP4 DeepGEMM 跳过 padding：https://github.com/vllm-project/vllm/commit/68ebea789a （PR #59044）
- FlashInfer 0.7.2：https://github.com/vllm-project/vllm/commit/af28c171f5 （PR #61913）
- 最新公开 release 仍为 v0.31.0（2026-10-05）：https://github.com/vllm-project/vllm/releases

**证据等级：** 官方一手；`#61867` 的 1.5–2×、`#59044` 的 112 KiB→128 bytes 等为 PR 自述声明，绑定特定配置；`#61913` 官方自述**未做验证与性能声明**。

---

## 3. SGLang：10-16 提交（79 条）——unified memory 预热、DSv4.1 丢弃行跳过、PD 延迟释放的重复释放修复

**主题**  SGLang 在 10 月 16 日的 79 条官方提交集中在：**unified memory 池首次压缩的启动预热**（修复满池下调度器崩溃）、DSv4.1 窗口提交**跳过将被丢弃的 KV 行**、dp-attention 的 MLP sync 行改用 **gather + broadcast**、custom all-reduce 新增 **2shot_push**、PD KV checksum 对**不可比 peer** 可跳过、机密计算下改用 **pageable host staging**，以及一条**严重的 PD 重复释放**修复。

**内容详情**  官方提交 API 与 PR 正文显示（节选可核验项）：
- **unified memory（两条，其一为崩溃修复）：**
  - **`[Perf] unified-memory: Prewarm the compaction move at startup and copy whole page envelopes by page id`**（`36cf028346`，PR `#44592`）：官方动机写得很直接——**第一次压缩会在服务过程中付出首次使用成本**（move kernel 需编译与加载、commit 的 `.all()` reduction 与 `_assert_async` 首次执行、kernel 首次 launch 可能需要池外设备内存即上下文为 spilling kernel 预留的 local memory）。**当这次首次压缩是满池下的紧急压缩时，launch 会以 `Triton Error [CUDA]: out of memory` 失败，调度器随之宕掉**；即使不失败，首次使用工作也会拖住调度器循环。改法：`MultiEndedAllocator.prewarm_compaction()` 在**启动时**用保留的 sink page 0 跑一次两元素自拷贝，完成首次使用工作；并按 page id 拷贝整页 envelope。
  - **`[Fix] unified-memory: enabling trtllm_mha with decode graphs and speculative decoding`**（`8d229d96e0`）。
- **KV 与 HiCache：**
  - **`[Perf] DSv4.1: Skip discarded KV rows during window commit`**（`8e3f11b219`，PR `#44783`）：request-window commit 原先**拷贝每一条输入 KV 行**，把被丢弃的行路由到共享 sink，但**只有最终窗口会存活**，因此长 prefill 搬运了大量永不被读的 KV 字节。新增 Triton commit kernel：对**被丢弃的行跳过 KV load/store**、计算保留下来的 ring 目的地、并在同一次 launch 内写 position tags；历史校验与既有窗口保留策略不变。**注意：launch grid 仍覆盖全部输入行**——本改动消除的是**被丢弃行的访存流量**，而非缩小 grid。官方称在 GB300 上通过 22 项本地测试（4 项 packed-copy/commit GPU、5 项 encoder replay CPU、13 项 compressed-pool CPU）。
  - **`[Fix] hicache: Bound the storage prefetch timeout from the first blocked admission poll`**（`453b251dbd`，PR `#44614`）：在 `timeout` 停止策略下，一次 storage prefetch 从**操作自身启动时刻**计时，因此对同一请求**每次重新发起的 fetch 都会重新开始一个预算**（且按当前可能已缩短的 hit chain 定尺寸），结果是**请求可以跨重试无限等待**；同时 per-request 等待指标 `storage_prefetch_wait` 与策略预算由两套互不相关机制计时。另两个问题：**被截断的读被报成 `storage_transfer`**，从而错误地「治愈」了 store 其实仍持有页面的存在性信念；以及尚未到达自身 admission poll 的 rank 可能重新发起 peer 已同意放弃的 prefetch。
  - **`[Perf] [NVIDIA] scheduler: Pageable host staging and inline hidden-state/DSpark readback under Confidential Computing`**（`bda6ef7e01`，PR `#44568`）：CC 下 host↔device 拷贝经过 bounce buffer，本改动通过 `is_pin_memory_available()` 返回 `False` **改选 pageable staging**；另有两类 batch **不能走 readback worker**——返回 hidden states 的 batch 与 DSpark batch 读的是 CUDA graph buffer 的视图，**下一次 replay 会重写它们**，只有在 forward 之后立刻在 `copy_stream` 上排队的拷贝才能及时读到，晚于该时点由 worker 线程发起的拷贝会与下一次 replay 竞争。
  - **`[Fix] deepseek-v4.1: Keep the DSpark draft SWA out of the radix tree`**（`fad7d95a7f`）；**`[Fix] deepseek-v4.1: Keep the full SWA window for short bounded-replay extends`**（`ed8c3139f1`，PR `#41303`）；**`[Fix] [AMD] hicache: Serve encoder-replay prefix hits with a paged DSpark draft SWA`**（`80c54e6ac5`，PR `#42716`）；**`[Test] mem_cache: Free SWA peers below the eviction floor in the SWA repair test`**（`a18a908066`）；**`[CI] tests: Let test_hicache_storage size its host cache to the runner`**（`a31a8c2bbd`）。
- **P/D 解耦（本期含一条严重正确性修复）：**
  - **`[Fix] scheduler: Keep PD decode non-idle while deferred KV releases are held`**（`f9df33494a`，PR `#44386`）：PD decode 上一个失败或被中止的传输会被**延迟释放**（#35049，默认开启），其 KV 页、请求行、tree lock 与元数据槽会一直保留，直到每个 prefill rank 确认写入已排空或超时（**默认 30 s**）。而 `Scheduler.is_fully_idle` **从不检查这些 hold**——其 decode 分支只查 prealloc / retracted / transfer 队列，而被 hold 的请求**已经离开 transfer 队列**。后果链：`flush_cache` 在 hold 未结清时执行 → 清空 tree / request pool / allocator，被 hold 的页回到空闲链 → hold 结清时 `_do_release` → `release_kv_cache` **再次释放同一批页** → **allocator 中这些页各出现两次** → 后续两个请求拿到相同页。
  - **`[Feat] disaggregation: Let a KV transfer backend skip the checksum for an incomparable peer`**（`989ccf4863`，PR `#44605`）：可选 PD KV checksum（`--disaggregation-enable-kv-checksum`）会摘要 prefill rank 的**完整本地 KV**，decode rank 在请求进入 batch 时再摘要自己的本地 KV。该比较**只在两侧持有相同字节时成立**：在 prefill 与 decode **TP 宽度不匹配**时，decode rank 收到的是一个 prefill rank 所摘要内容的 head 切片、或来自多个 prefill rank 的切片拼接，于是**每个请求都被报为损坏并中止，尽管传输本身是正确的**。本 PR 让**知道每个 prefill peer 布局是否匹配**的传输后端来表达该判断：`BaseKVManager.should_verify_kv_checksum(bootstrap_addr) -> bool`，**默认 `True`**。
  - **`[Perf] pd: Copy prebuilt decode batch inputs to the GPU without host syncs`**（`d14fa5db9b`，PR `#44598`）：新增 `async_h2d_stack(tensors, *, device)`，把逐请求 host tensor 直接 stack 进 pinned memory 并用**一次非阻塞拷贝**搬运；`build_eagle_disagg_draft_input` 的 `topk_p` / `topk_index` / `hidden_states` / DSA seed rows 与 `prepare_for_prebuilt` 的 `req_pool_indices` / `seq_lens` / `orig_seq_lens` 改走该路径。**PR 明确：每个被替换的拷贝原先都会在 schedule stream 上执行 `cudaStreamSynchronize`**。
  - **`[Perf] pd: Stage decode-side KV preallocation lengths through pinned memory`**（`3a5d46e946`，PR `#38955`）；**`[Fix] engram: Avoid blocking history copies during PD admission`**（`71b60fa46c`）；**`[Feat] dsv4.1: Support encoder SWA bounded replay with CUDA DP attention`**（`aaf5f397a5`）；**`[Fix] DSv4.1: Fix BCG image handling and add multimodal bounded replay coverage`**（`06934575b9`）。
- **并行与通信（本期性能密度最高的一组）：**
  - **`[Perf] [NVIDIA] custom-all-reduce: Add 2shot_push all-reduce and retune SM100 TP4/TP8`**（`07950be745`，PR `#42811`）：`CustomAllReduceV2` 在 Blackwell 的 TP4/TP8 上**在约 0.5 MB 到 8 MB 之间没有好选择**——`1shot_push` 每 rank 搬 `W` 份 tensor 而力竭，`2shot_pull` 要付两次 barrier 与远端 load。本 PR 新增 **push-plane two-shot（Lamport markers、无 barrier）**，并基于 B300 sweep 重调 SM100 的 world 4/8 行；与 #35192 设计的差异在于**plane 布局**：两半 chunk 留在同一个既有 push slot 中，因此 push plane、其 workspace 与 `PushPlane` 构造不变，也未新增 multicast 变体。
  - **`[Perf] dp-attention: Exchange the scheduler's MLP sync rows with gather + broadcast on gloo`**（`2948662026`，PR `#44622`）：DP attention 下每个调度步都要在 CPU gloo group 上交换一行 `MLPSyncBatchInfo`（token 数、attention pairs、CUDA-graph 投票、forward mode）。**gloo 把 `all_gather` 实现为 `world_size - 1` 次串行 hop 的 ring，因此该交换的延迟随 attention world 线性增长**；改为 **gather 到 group rank 0 + broadcast** 只需一跳加一个 log 深度树，且填充同一 buffer。`exchange_sync_rows` 现在按 group-rank 顺序填充 flat buffer，行为与 `all_gather_single` 一致。
  - **`[Fix] [NVIDIA] allreduce-fusion: Decline MNNVL fusion across NVLink cliques`**（`83940471b4`）；**`[Fix] monokernels: Honor resolved CUDA graph configuration`**（`51faf52305`）；**`[Perf] [XPU] usp: Enable packed Ulysses QKV all-to-all`**（`1e33d2c4bd`）。
- **AMD / ROCm（本期提交数量最大的一簇，含 dsv4 八件套）：**
  - dsv4 系列 **1/8–7/8**：`#44016` fp8 `unified_kv` 在 FlyDSL v4 nm decode 上的 decode/verify（含 latency-model split plan）、`#44003` fp8 verify 在融合 norm/rope kernel 中写 SWA ring、`#44004` 小 batch sort（histogram + bitonic，**32 < P ≤ 256**）、`#44006` aiter row-group kernel 上的 FP4 indexer（decode 与 target-verify）、`#44007` 稠密 128×128 fp8 linear 的可选 e8m0 mxscale GEMM、`#44008` 携带 aiter DSv4 GEMM tuned rows 与 mHC `tile_m=16` 直到 aiter 上游发布、`#41903` 复用 q_lora fp8 量化给 C4 indexer 的 `wq_b`；另有 **`[Perf] [AMD] dsv4: Add a separate MegaMoE decode instance`**（`#44657`）。
  - **`[Feat] [AMD] disagg: Support DSV4 decode radix cache on ROCm unified KV (bf16/fp8, DSpark)`**（`f05da43d44`，PR `#30929`）；**`[Perf] [AMD] dspark: No per-step host sync in decode (1/4)`**（`bdc4ec15ce`，PR `#44405`）；**`[Perf] [AMD] qwen3: Fuse QK-norm + RoPE + KV cache write for dense Qwen3 decode`**（`a448fdf316`，PR `#44267`）；**`[CI] [AMD] aiter: pin aiter commit to 2bb549 for rocm10.1 gfx1250`**（`9246acd462`）；**`[Feat] [AMD] docker: Install the sgl-amd-mono monokernel plugin in the ROCm image`**（`536cdb4b04`）。
- **内核与量化：** **`[Feat] FlashInfer: Add NVFP4 W4A16 and SiTU support to MegaMoE`**（`7098ec387c`，PR `#44036`）；**`[Fix] compressed-tensors: Serve NVFP4 W4A4 linears via FP4 Marlin when --fp4-gemm-backend marlin is set on SM100`**（`5117bfbaeb`，PR `#44213`）；**`[Chore] deps: Bump FlashInfer to 0.7.2`**（`44be30d216`）；**`[Perf] deepseek-v4.1: Integrate FlashMLA MegaAttention with NVFP4 KV`**（`a93f16f0cf`，PR `#43922`）；**`[Feat] kernels: Add muse spark operators and collective GEMM (2/5)`**（`456231e251`）。
- **调度与前端：** **`[Fix] sgl-router: ...`** 系列（`#42674` native gRPC ChatComplete、`#42681` 其余原生 gRPC 推理 RPC、`#44532` 由 Rust 前端提供 `/v1/loads`、`#44531` 在 `meta_info` 上报 `cached_tokens` 与 `reasoning_tokens`、`#44533` 在 `/generate` 上遵守 `cache_salt` 与 `require_reasoning`）；**`[Fix] sglang-server: Keep the matched stop token in output_ids`**（`2957ff3165`，PR `#42637`）；**`[Fix] imports: Bind FlashInfer lazily at four import sites so the server starts without it`**（`4e85e8ed37`，PR `#44430`）；**`[Fix] sglang-server: ...`**；**`[Fix] sglang: Apply --gc-threshold in every server process and freeze every tokenizer worker`**（`d8dd49d21d`，PR `#44785`）；**`[Fix] tokenizer_manager: re-report only emitted customized_info keys on cumulative streams`**（`d393298736`）；**`[Fix] dsa: Fix ragged seed indices before MTP draft decode`**（`7c99ace127`，PR `#42565`）；**`[Fix] DSA: skip-topk pipeline handoff for owed outputs`**（`d5d87899f9`，PR `#43832`）。
- **模型、文档与 CI：** `MiniCPM-V-4.7` 模型支持（`#41993`）；muse spark CUDA 镜像与 DeepGEMM patch（`#44664`）；多篇 cookbook（Beam 1.0 部署、SemiAnalysis AgentX launches、Muse Spark Open SWA ratio calculator、RTX Spark Qwen3.6 NVFP4、XPU Docker 0.5.22、NPU 依赖刷新与 qwen3-235b-a22b 镜像冻结、DeepSeek V4 AgentX PD recipe）；`[Doc] blogs: Sync LMSYS SGLang blog cards`（`c15edf314d`）；`[Fix] utils: Allowlist torch.bcomplex32 in SafeUnpickler`（`e3961eb6fa`）。

**应用场景**  unified memory / 统一 KV 池在满池紧急压缩下的稳定性、DSv4.1 长 prefill 的窗口提交访存治理、HiCache storage 超时语义与指标口径、PD 延迟释放的页双重释放、TP 宽度不匹配的 P/D 部署下 KV 校验、Blackwell TP4/TP8 的 all-reduce 选择带、DP attention 的 CPU 同步延迟、AMD MI355X/gfx1250 上的 DSv4 全链路优化。

**研判**  **事实：** 本期 SGLang 出现三类高信号改动——(1) **一条明确的崩溃修复**（`#44592`：满池下首次紧急压缩以 `Triton Error [CUDA]: out of memory` 失败并带崩调度器）；(2) **一条明确的正确性修复**（`#44386`：PD 延迟释放导致同一批 KV 页被释放两次，allocator 中出现重复页）；(3) **一批针对「重试语义」的修正**（`#44614` 的 prefetch 每次重试重置预算导致无限等待、`#44605` 的 checksum 在 TP 宽度不匹配下把正确传输报成损坏）。**观点：** 四点——(1) `#44386` 与 `#44592` 共同说明 SGLang 当前的主要风险面已经从「吞吐」转向「**生命周期与首次使用路径**」：前者是页归属的生命周期，后者是 kernel/allocator 的首次使用成本——后者尤其值得警惕，因为**它把一个平时无害的启动开销，变成满池高负载下的致命失败**，属于典型的「只在最坏时刻触发」的缺陷；(2) `#44614` 与 `#44605` 属于同一类：**重试/校验机制自身成为故障源**——超时按操作而非按请求计时、校验在异构布局下失去可比性。这类问题难以通过端到端精度测试发现，建议把 `storage_prefetch_wait` 与 PD checksum 的中止计数直接接入监控；(3) `#42811` 的 2shot_push 与 `#44622` 的 gather+broadcast 都指向同一个工程事实——**通信原语的实现细节（gloo 的 ring all_gather、push plane 的 slot 布局）在中等规模与中等消息尺寸下会主导延迟**，因此「换算法」的收益可能大于「换硬件」；(4) `#44783` 的注释值得单独记住：**它消除的是被丢弃行的访存流量，而非缩小 launch grid**——即优化针对的是带宽而非 launch 开销，二者不可混同。

**来源**
- SGLang 官方提交 API（2026-10-16 UTC，79 条）：https://api.github.com/repos/sgl-project/sglang/commits?since=2026-10-16T00:00:00Z&until=2026-10-17T00:00:00Z&per_page=100
- unified memory 启动预热压缩：https://github.com/sgl-project/sglang/commit/36cf028346 （PR #44592）
- DSv4.1 窗口提交跳过丢弃 KV 行：https://github.com/sgl-project/sglang/commit/8e3f11b219 （PR #44783）
- PD 延迟释放导致重复释放：https://github.com/sgl-project/sglang/commit/f9df33494a （PR #44386）
- hicache storage prefetch 超时上界：https://github.com/sgl-project/sglang/commit/453b251dbd （PR #44614）
- PD KV checksum 对不可比 peer 可跳过：https://github.com/sgl-project/sglang/commit/989ccf4863 （PR #44605）
- CC 下 pageable host staging 与内联 readback：https://github.com/sgl-project/sglang/commit/bda6ef7e01 （PR #44568）
- custom all-reduce 2shot_push：https://github.com/sgl-project/sglang/commit/07950be745 （PR #42811）
- dp-attention MLP sync 改 gather+broadcast：https://github.com/sgl-project/sglang/commit/2948662026 （PR #44622）
- PD 预构建 decode batch 输入免 host sync：https://github.com/sgl-project/sglang/commit/d14fa5db9b （PR #44598）
- FlashMLA MegaAttention × NVFP4 KV：https://github.com/sgl-project/sglang/commit/a93f16f0cf （PR #43922）
- 最新公开 release 仍为 v0.5.21（2026-10-02）：https://github.com/sgl-project/sglang/releases
- SGLang 官方博客（窗口内无新文章，最新仍为 2026-10-14）：https://www.sglang.io/blog/step-5-preview-day0-support

**证据等级：** 官方一手（提交标题 + 对应 PR 正文）；PR 中的百分比与测试结论均绑定其自述配置；`#43922` 的 PR 正文为占位内容（仅「Integrate MegaAttention」），**其具体实现细节未从 PR 正文获得**。

---

## 4. TensorRT-LLM：10-16 提交（36 条）——KVCacheManagerV2 池再平衡可配置、MiniMax-M3 稠密层改 paged attention、96 head MLA autotune

**主题**  TensorRT-LLM 在 10 月 16 日的 36 条官方提交继续围绕 **KV cache V2 的可配置化与测试迁移**、MiniMax-M3 / MiniMax-H3 的模型支持、**96 head MLA 的 fmha autotune**、投机解码的 per-request seed、SM107 NVFP4 CuTe DSL dispatch 与 MoE workspace 回收。

**内容详情**  官方提交 API 显示（节选可核验项；**受本期 GitHub API 速率上限影响，本节仅基于官方提交标题，未取到 PR 正文**）：
- **KV cache 与分离式部署：**
  - `[None][feat] make KVCacheManagerV2 pool rebalancing parameters configurable`（`2e9c7c378f`，PR `#19077`）；
  - `[https://nvbugs/6888251][test] Disable THP for KV cache V2 host tiers in disagg tests`（`9e8e5a270e`，PR `#20032`）——即 **KV cache V2 的 host tier 会受透明大页影响**，测试中显式关闭 THP；
  - `[https://nvbugs/6902355][fix] Keep TopK rows in the DSA sparse MLA test KV pool`（`b519164514`，PR `#20283`）；
  - `[https://nvbugs/6893965][fix] Extend disagg server start timeout in multi-node disagg test`（`3ac4e61255`，PR `#20453`）；`[https://nvbugs/6913517][fix] Drop launcher spawn-proxy env in the disagg cancel marathon`（`2df58914a1`，PR `#20420`）；`[https://nvbugs/6836097][test] Restrict UCX TCP to loopback in single-node disagg tests`（`84f14da455`，PR `#20031`）。
- **注意力与 MLA：**
  - `[https://nvbugs/6691772][fix] Update trtllm-gen fmha autotuner to enable 96 head MLA`（`4dee061ea6`，PR `#18599`）——与本期 vLLM 博客中 Kimi K3 的 **96 MLA query heads**（TP8 下每卡 12 个）形成对照；
  - `[None][perf] primsTS dense bf16, qk-bf16/pv-fp8, fp8 fmha context bringup`（`c4afddb48c`，PR `#20006`）；
  - `[TRTLLM-16992][perf] Use paged attention for MiniMax-M3 dense layers`（`4271011718`，PR `#20159`）；
  - `[https://nvbugs/6900756][fix] Bound packed indexer prefill logits by grouping context requests`（`dac84b01a6`，PR `#20164`）——限制 packed indexer prefill logits 的内存上界；
  - `[https://nvbugs/6713231][fix] Guard Whisper cross-KV reuse without encoder tokens`（`7ebcf4debc`，PR `#19302`）。
- **MoE 与量化：**
  - `[TRTLLM-16972][fix] Reclaim idle eager CUTLASS MoE workspace`（`a0835b3d4f`，PR `#19981`）——回收空闲的 eager CUTLASS MoE workspace；
  - `[None][fix] Enable SM107 NVFP4 CuTe DSL dispatch`（`c264cd1823`，PR `#20204`）；
  - `[https://nvbugs/6920456][fix] ...`、`[TRTLLM-16992]` 等；
  - `[None][doc] Add MoE Performance Guide and Perf Dashboard`（`45a126a65b`，PR `#19546`）。
- **投机解码：**
  - `[TRTLLM-16780][feat] Honor per-request seeds in one-model speculative decoding's acceptance step`（`7fafb022ee`，PR `#20147`）——投机解码**接受步骤遵守逐请求 seed**，与上一期 vLLM 修复「跨 block 验证轮次复用随机抽样」属同一类**可复现性**问题；
  - `[None][chore] Remove is_first_draft dead code`（`c226e2d77c`，PR `#19867`）。
- **模型与多模态：**
  - `[None][feat] Add opt-in MiniMax-H3 sleep/wake support`（`19604`）——**选择性启用**的 sleep/wake（与 vLLM sleep mode 同类能力）；
  - `[None][feat] VisualGen: token-sharded tensor parallelism (tp_layout) for Wan`（`38dc88e996`，PR `#19713`）；
  - `[https://nvbugs/6926547][fix] Route FLUX.2 linear_in quantization sca…`（`778c6c1b1a`，PR `#20403`）。
- **平台与基础设施：**
  - `[TRTLLMINF-439][infra] Add GR100 post-merge single-GPU stage`（`e95cdbdadc`，PR `#19894`）——新增 **GR100**（Rubin 代）后合并单卡阶段；
  - `[None][fix] Reduce deterministic GVR V2 top-K overflow overhead`（`0c1882b28b`，PR `#20290`）；
  - `[https://nvbugs/6836304][fix] Forward-port stale MPI worker env fix to main and unwaive multimodal tests`（`bb1600f99a`，PR `#20151`）——与上一期 v1.4.0rc0 Known Issue「多 GPU streaming/RPC 复用 MPI worker pool 时可能在 worker 初始化挂起」**同源**；
  - `[None][infra] Check in most recent lock file from nightly pipeline`（5 次）；`[TRTLLM-15891][infra] Refactor CBTS Collection`（`#18299`）；`[None][infra] Add blossom-ci authorized users`（`#20461`）。
- **文档：** `[TRTLLM-16619][doc] Remove stale execution backend guidance`（`#19986`）与 `[https://nvbugs/6920456][fix] Describe PyTorch as the sole execution backend in the architecture overview`（`e921fa39b0`，PR `#20350`）——**官方文档统一把 PyTorch 描述为唯一执行后端**，与上一期「移除 KVCacheManagerV2 的 Python 后端」属同一条迁移主线。
- **本期无新 release：** 最新仍为 **v1.4.0rc0**（prerelease，published 2026-10-14T16:35:44Z）。

**应用场景**  KV cache V2 的池再平衡调参与 host tier 的 THP 敏感性、多节点分离式部署的启动与取消路径、96 head MLA 的 fmha 调优、MiniMax-M3 / MiniMax-H3 服务化、SM107 NVFP4 内核路径、投机解码的逐请求可复现性、GR100 平台准备。

**研判**  **事实：** 本期 TRT-LLM 无新版本，提交以 KV cache V2 的**可配置化与测试加固**、模型支持与内核/autotune 修复为主；其中两条与上一期直接衔接——**MPI worker env 修复被 forward-port 到 main 并解 waive 多模态测试**（对应 v1.4.0rc0 的挂起类 Known Issue），以及**文档统一声明 PyTorch 为唯一执行后端**（延续 KV cache V2 的破坏性迁移）。**观点：** 三点——(1) `#19077` 把 KV cache V2 的**池再平衡参数**变为可配置，说明 V2 已进入「按部署调参」阶段而非仅功能落地，配合 `#20032` 关闭 THP 的测试改动，提示 **V2 host tier 对操作系统内存策略敏感**，自建部署应把 THP 设置纳入核对项；(2) `#20147`（逐请求 seed）与上一期 vLLM 的「跨 block 验证轮次复用随机抽样」修复指向同一个问题域——**投机解码的随机性来源在多请求/多轮次下会互相污染**，凡依赖可复现评测或 RL 一致性的场景都应重点验证；(3) 本期**受速率上限影响未取到 PR 正文**，因此对 `#18599`（96 head MLA autotune）、`#19981`（MoE workspace 回收）、`#20290`（GVR V2 top-K 溢出开销）三项**只能记录其提交标题所声明的事实，不记录任何未取得的性能数字**——下一期应在配额充足时补取这三项的正文以确认适用范围。

**来源**
- TensorRT-LLM 官方提交 API（2026-10-16 UTC，36 条）：https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits?since=2026-10-16T00:00:00Z&until=2026-10-17T00:00:00Z&per_page=100
- KVCacheManagerV2 池再平衡可配置：https://github.com/NVIDIA/TensorRT-LLM/commit/2e9c7c378f （PR #19077）
- 96 head MLA fmha autotune：https://github.com/NVIDIA/TensorRT-LLM/commit/4dee061ea6 （PR #18599）
- MiniMax-M3 稠密层 paged attention：https://github.com/NVIDIA/TensorRT-LLM/commit/4271011718 （PR #20159）
- 逐请求 seed 的投机接受步骤：https://github.com/NVIDIA/TensorRT-LLM/commit/7fafb022ee （PR #20147）
- SM107 NVFP4 CuTe DSL dispatch：https://github.com/NVIDIA/TensorRT-LLM/commit/c264cd1823 （PR #20204）
- 回收空闲 CUTLASS MoE workspace：https://github.com/NVIDIA/TensorRT-LLM/commit/a0835b3d4f （PR #19981）
- 关闭 KV cache V2 host tier 的 THP（测试）：https://github.com/NVIDIA/TensorRT-LLM/commit/9e8e5a270e （PR #20032）
- MPI worker env 修复 forward-port：https://github.com/NVIDIA/TensorRT-LLM/commit/bb1600f99a （PR #20151）
- 最新 release 仍为 v1.4.0rc0（2026-10-14）：https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc0

**证据等级：** 官方一手（官方提交标题与 PR 编号）；**本期因 GitHub API 速率上限未取到 PR 正文**，故不记录任何 PR 内的性能数字与适用边界。

---

## 5. llama.cpp：10-16 提交（29 条）——k-pool 输入视图跨层共享、CUDA pinned host buffer 上限 4 GB、多后端内核修缮

**主题**  llama.cpp 在 10 月 16 日的 29 条官方提交集中在：**KV 相关的 k-pool 输入视图跨层共享**、**CUDA pinned host buffer 上限限制为 4 GB**、CUDA/HIP `rms_norm` 的 float4 变体、Hexagon/SYCL/Vulkan/OpenCL 多后端的内核与内存修复，以及 Mellum2.1 的 MTP 支持。

**内容详情**  官方提交 API 显示（节选可核验项；**受本期 GitHub API 速率上限影响，本节仅基于官方提交标题**）：
- **KV / 内存路径（本期与本档案主题最相关的两条）：**
  - `llama : share the k-pool input views between layers`（`6afa9a3134`，PR `#30613`）——**k-pool 的输入视图在层间共享**，属 KV 缓冲视图/别名管理类改动；
  - `Limit max CUDA pinned host buffer size to 4 GB`（`b6334e37e6`，PR `#30449`）——为 CUDA pinned host buffer **设 4 GB 上限**，与上一期 v0.7.0 引入的 host 专家缓存（`--moe-cache-mib`）与 `--fit` 显存/内存规划同属**主机侧缓冲的容量治理**。
- **CPU 与 GGML：** `cpu : reduce AVX2 tiled accumulators together`（`3dbcb7fca5`，PR `#30296`）；`ggml-cpu: refactor s309x repack helper functions`（`c804754529`，PR `#30588`）；`gguf-py : add numpy Q1_0 and Q2_0 de/quantization support`（`495c5805ce`，PR `#30630`）。
- **CUDA / HIP：** `CUDA: HIP: rms_norm: add variants that load float4 at a time and store x in registers`（`623c16393c`，PR `#29720`）；`Fix failing RocM tests`（`f12aaa344c`，PR `#30611`）。
- **Hexagon（本期出现次数最多的后端）：** `hexagon: tensor repacking rework and support for tiled and flat quantized and fp weights`（`b797b9ca81`，PR `#30635`）；`hexagon: correct HMX FA sinks`（`ceefd16daa`，PR `#30581`）——**修正 HMX flash attention 的 sink**；`hexagon: vectorize softplus with HVX`（`27c7a769da`，PR `#30435`）；`ci: update device matrix for Snapdragon jobs by removing SM8750`（`6420cf5dd5`，PR `#30620`）。
- **OpenCL / Vulkan：** `opencl: Add MxFP4 MoE bin kernel for Adreno X1 devices`（`2f2763db6e`，PR `#30691`）——**Adreno X1 上的 MxFP4 MoE bin kernel**；`vulkan: fix gpu-visible host memory issues`（`a8d9a7e1e5`，PR `#30468`）。
- **SYCL：** `[SYCL]: fix memory errors in mul_mat, split buffer, host pool`（`a30fb4924d`，PR `#29889`）；`sycl: support reordered Q8_0 MoE weights`（`e707790c73`，PR `#29337`）；`sycl: wide stores for the q4_K/q5_K weight dequant`（`13de11e78f`，PR `#29696`）；`sycl : reduce the build threads to half`（`c0083e3052`，PR `#30236`）；`[SYCL] Enable ssl in build script and CI`（`43d57fb8ca`，PR `#30456`）。
- **模型与转换：** `model : MTP support for Mellum2.1`（`9a16f1dbe4`，PR `#30612`）；`graph : add precision handling for MELLUM architecture`（`5e03a1ff45`，PR `#30584`）；`model: add Beam (Reflection)`（`01fc3aae84`，PR `#30670`）；`model : Granite Switch Decision`（`23f7aabc16`，PR `#30676`）；`convert : speed up model endian conversion and add more datatype support`（`b84ea5cf5e`，PR `#30647`）；`convert : respect Mistral format for DeepseekV2Model`（`2a667032dc`，PR `#30669`）；`convert : support text-only checkpoints in pplx-decider`（`0f4376268f`，PR `#30575`）。
- **common / 服务：** `common : read ngram cache parts by reference instead of copying`（`30061`）——ngram cache 分片**按引用读取而非拷贝**；`common: honor HTTP_PROXY/HTTPS_PROXY env vars in http client`（`7182ef71e6`，PR `#21752`）。
- **测试与基准：** `tests : catch vanished rows in the fusion baseline`（`88aaf325a7`，PR `#30618`）；`Add more benchmarks to llama-eval`（`f3363593f4`，PR `#26174`）。
- **本期无新正式版本：** 正式版仍为 **v0.7.0**（2026-10-15T17:13:19Z）；窗口内滚动 nightly 构建标签为 **b11701**（2026-10-16T23:20:45Z）、**b11700**（18:33:07Z）、**b11699**（17:57:12Z）、**b11696**（11:26:26Z）；**b11702 发布于 2026-10-17T00:21:45Z，已落在本档案关注窗口之外**，故仅记录不纳入本期。

**应用场景**  本地/端侧 MoE 与长上下文部署的 KV 缓冲视图与主机缓冲容量、CUDA/HIP 归一化内核、Snapdragon/Adreno（Hexagon、OpenCL）与 SYCL 异构后端的正确性、Mellum2.1 / Beam / Granite Switch Decision 等新模型本地推理、ngram 缓存与 HTTP 代理环境。

**研判**  **事实：** 本期 llama.cpp 无新正式版本，提交以**多后端内核修缮与内存上限**为主；其中 `k-pool 输入视图跨层共享` 与 `CUDA pinned host buffer 上限 4 GB` 是两条与本档案 KV/内存主线相关的改动，其余集中在 Hexagon（4 条）、SYCL（5 条）、OpenCL/Vulkan（2 条）等异构后端。**观点：** 三点——(1) 两条 KV/内存改动的方向与上一期 v0.7.0 的 `--moe-cache-mib` 一脉相承：**主机侧缓冲开始被显式设界**（4 GB 上限）并**共享视图**（k-pool 跨层复用），这与 vLLM / SGLang 在同一时期对 host staging、pinned memory 与池分配的密集改动是同一趋势——**「host 内存」正从「免费资源」变成需要显式规划与限额的资源**；(2) `common : read ngram cache parts by reference instead of copying` 与 `Limit max CUDA pinned host buffer size to 4 GB` 都是**避免无谓拷贝与无界增长**的小改动，但它们与本期 vLLM 博客中的第 3 类优化（去冗余拷贝、只对使用输入的 kernel 做拷贝）**根因完全相同**，可作为跨项目验证「减少拷贝」这一方向的独立证据；(3) 本期 llama.cpp 的 Hexagon/SYCL/OpenCL 修缮密度明显高于 CUDA，说明**端侧异构后端的正确性收敛仍在进行中**，官方 release notes 级别的稳定性结论尚不适用，本地部署应针对具体后端单独回归。

**来源**
- llama.cpp 官方提交 API（2026-10-16 UTC，29 条）：https://api.github.com/repos/ggml-org/llama.cpp/commits?since=2026-10-16T00:00:00Z&until=2026-10-17T00:00:00Z&per_page=100
- k-pool 输入视图跨层共享：https://github.com/ggml-org/llama.cpp/commit/6afa9a3134 （PR #30613）
- CUDA pinned host buffer 上限 4 GB：https://github.com/ggml-org/llama.cpp/commit/b6334e37e6 （PR #30449）
- CUDA/HIP rms_norm float4 变体：https://github.com/ggml-org/llama.cpp/commit/623c16393c （PR #29720）
- Hexagon HMX FA sink 修正：https://github.com/ggml-org/llama.cpp/commit/ceefd16daa （PR #30581）
- Hexagon 张量重打包改造：https://github.com/ggml-org/llama.cpp/commit/b797b9ca81 （PR #30635）
- OpenCL Adreno X1 MxFP4 MoE kernel：https://github.com/ggml-org/llama.cpp/commit/2f2763db6e （PR #30691）
- Vulkan gpu-visible host memory 修复：https://github.com/ggml-org/llama.cpp/commit/a8d9a7e1e5 （PR #30468）
- ngram cache 分片按引用读取：https://github.com/ggml-org/llama.cpp/commit/13b8048920 （PR #30061）
- Mellum2.1 MTP：https://github.com/ggml-org/llama.cpp/commit/9a16f1dbe4 （PR #30612）
- 最新正式 release v0.7.0（2026-10-15）：https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0

**证据等级：** 官方一手（官方提交标题与 PR 编号）；**本期因 GitHub API 速率上限未取到 PR 正文**，故不记录任何 PR 内的性能数字与适用边界；nightly b-tag 为滚动构建，不构成稳定性承诺。

---

## 6. 当前会话已形成的累计内容结论

1. **KV Cache 的竞争焦点已从「容量与分层」转移到「规划权、生命周期与重试语义」**：本期新增三类证据——(1) vLLM 把 KV cache 分组/放置从 core 单一启发式改为**可插拔 `KVCacheConfigBuilder`**（`#53558`，RFC `#62100`，动机是 vllm-ascend 的 monkey-patch），并在 core 保留校验；(2) SGLang 修复 PD **延迟释放导致同一批 KV 页被释放两次**（`#44386`）、并把 HiCache storage prefetch 的**超时基准从「操作自身」改为「首次阻塞的 admission poll」**（`#44614`，此前每次重试都重置预算、请求可无限等待）；(3) TensorRT-LLM 把 **KVCacheManagerV2 的池再平衡参数变为可配置**（`#19077`）并显式关闭 host tier 的 THP（`#20032`）。三条合起来说明：KV cache 已经从「怎么放」推进到「**谁来定规则、页归谁、失败后怎么收**」。
2. **「协调与同步」仍是 P/D 解耦与并行部署的第一瓶颈，且本期证据集中在 CPU 侧同步原语**：SGLang 把 dp-attention 每步的 `MLPSyncBatchInfo` 交换从 **gloo ring `all_gather`（world_size−1 次串行 hop）** 改为 **gather + broadcast**（`#44622`）；PD 预构建 decode batch 输入改走 `async_h2d_stack`，官方明确指出**原先每个被替换的拷贝都会在 schedule stream 上执行 `cudaStreamSynchronize`**（`#44598`）；vLLM 修复 **DP wave 唤醒的竞态**（paused rank 丢弃 `START_DP_WAVE` 导致全体空闲，`#59559`）。这与上一期 vLLM 官方博客把 P/D 瓶颈改写为「协调串行化 + 关键路径 CPU 工作」完全一致。
3. **投机解码的风险从「接受率」转向「跨 rank / 跨轮次的一致性」**：本期 vLLM `#59824` 明确指出**若不同步自适应 DSpark 决策，rank 本地 confidence 会让不同 TP rank 选出不同 draft 预算，导致集合通信错配并挂起**（Kimi-K3 在并发 ≥ 8 时约 **97% 的 step 触发裁剪**）；TensorRT-LLM `#20147` 让接受步骤**遵守逐请求 seed**；vLLM `#61451` 用**请求边界**而非位置判断识别 bonus logit，消除从未被读取的全词表归约。加上上一期的「跨 block 验证轮次复用随机抽样」修复，可确认：**投机解码在多请求/多 rank 下的随机性与预算决策必须显式同步**。
4. **「重复计算」与「无消费者拷贝」被官方明确列为跨厂商通用的可优化根因**：vLLM 博客给出的三条检查法中，前两条（Compare ranks / Follow buffers）分别对应两个真实案例——MoE 路径下**8 个 GPU 各算一遍同一投影**（`#51253`，输出坐标划分，输出吞吐 +3.05%–4.91%）、ROCm 矩阵乘 dispatcher **先打包输入再选 kernel 导致 69 次/step 无谓拷贝**（`#58566`，单次 1,792 bytes 却约 4.8 µs，launch 开销主导）。llama.cpp 同期出现同构证据（k-pool 视图跨层共享、ngram cache 按引用读取）。**这类问题与厂商无关，可直接作为自查清单。**
5. **MoE 推理继续沿「专家放置与访存去冗余」演进，而非单纯 kernel 优化**：本期新增 vLLM **SM100 FP4 DeepGEMM 用 prefix-sum scheduler 跳过 padding**（`#59044`，每行布局元数据 **112 KiB → 128 bytes**）、Kimi K3 按 rank 加载专家行**释放约 3.85 GiB/GPU**（`#59591`，官方标注晚于头条镜像）、SGLang **MegaMoE 新增 NVFP4 W4A16 与 SiTU**（`#44036`）、TensorRT-LLM **回收空闲 eager CUTLASS MoE workspace**（`#19981`）。加上上一期 llama.cpp 的 host 专家 GPU LRU 缓存，**「专家行/列的布局与驻留」已成为一等公民**。
6. **量化结论必须绑定硬件、内核与并行布局，本期再次被强化**：vLLM 博客明确 Kimi K3 在 MI355X 上为**权重 FP4 + KV FP8 + attention BF16**，并把 **A4W4（activations 8→4 bit）** 列为独立改动；SGLang 在 SM100 上按 `--fp4-gemm-backend marlin` 选择 **NVFP4 W4A4 的 FP4 Marlin 路径**（`#44213`）；TensorRT-LLM 新增 **SM107 NVFP4 CuTe DSL dispatch**（`#20204`）与 **96 head MLA fmha autotune**（`#18599`）。**同一 dtype 在不同引擎/设备/布局上仍不可互推。**
7. **「首次使用成本」与「静默失效」成为新的高风险面**：SGLang `#44592` 揭示**满池下的首次紧急压缩会因 kernel 首次 launch 需要池外设备内存而以 `Triton Error [CUDA]: out of memory` 失败并带崩调度器**——即一个平时无害的启动开销在最坏时刻变成致命故障；`#44605` 揭示 **TP 宽度不匹配时 PD KV checksum 会把正确传输报为损坏并中止每个请求**；vLLM `#61913`（FlashInfer 0.7.2 + 恢复 KDA auto backend）PR 自述**未运行 GPU 测试、未做任何精度或性能声明**。三者共同点：**不会通过端到端精度测试暴露**，需靠启动预热、监控指标与升级审查发现。

---

## 7. 今日可执行清单

1. **把 vLLM 博客的三条检查法作为自有部署的自查清单**：(a) **Compare ranks**——用 profiler 检查集合通信前是否有多个 rank 算了相同结果（对应 MoE shared-expert 投影的 8 次重复）；(b) **Follow buffers**——逐个确认临时 tensor 在**所选路径**上是否真有消费者、能否留在片上（对应 dispatcher 先打包后选 kernel 的 69 次/step 无谓拷贝）；(c) **Count independent work**——核对实际 heads / tokens / tiles 是否产生足够 workgroup（对应 KDA verify 从 48 提升到 384 个 workgroup 才换来 2.89×）。三项均不依赖 AMD 硬件，可在任何 TP 部署上执行。
2. **长上下文 Agent 服务的 KV 规划做一次状态一致性审查**：核对 DCP 与 TP 的组合是否让每卡只存 1/DCP 的压缩历史；确认 **CPU offload 的恢复点在所有必需层上兼容**（官方示例：MLA 历史 3,072 而 KDA checkpoint 只在 2,048，则必须从 2,048 重算 1,024 个前缀位置）；若使用 KDA/Mamba 类状态模型，确认 attention-block tail 与 checkpoint 同存（vLLM `#54736`）或等价机制已到位。
3. **PD 部署必须验证两类「延迟释放/校验」路径**：对 SGLang，确认 `flush_cache` 与延迟释放的互斥已包含在修复中（`#44386`，默认 30 s 超时窗口内被 hold 的页不得被 flush 归还）；若启用 `--disaggregation-enable-kv-checksum`，**在 prefill 与 decode TP 宽度不一致时必须跳过校验**（`#44605`），否则每个请求都会被误判损坏并中止；同时把 `storage_prefetch_wait` 接入监控并确认超时按**首次 admission poll** 计时（`#44614`）。
4. **DP 与 DP-attention 部署做唤醒与同步回归**：vLLM 侧确认已含 `#59559`（引擎独占 wave 唤醒、front-end 不再发 `FIRST_REQ`），并针对「多 API server + 多 rank、部分 rank 暂停」的组合做长时压测；SGLang 侧确认 dp-attention 的 MLP sync 交换已改为 gather + broadcast（`#44622`），并观察 attention world 增大时每步 CPU 同步延迟是否仍随 world 线性增长。
5. **投机解码做一次「跨 rank / 跨轮次一致性」专项检查**：确认自适应验证/预算决策在 **所有 TP rank 上同步**（vLLM `#59824` 明确不同步会导致集合通信错配与挂起，Kimi-K3 并发 ≥ 8 时约 97% 的 step 触发裁剪）；确认接受步骤遵守**逐请求 seed**（TensorRT-LLM `#20147`）；按 drafter 类型分别记录接受率与吞吐，并注意 Kimi K3 类模型的 draft 深度存在明确阈值——**4 drafts 只有在完整周期耗时比 3 drafts 增加不到 12% 时才划算**。
6. **把「首次使用成本」纳入上线前预热清单**：对 SGLang unified memory，确认 `prewarm_compaction()` 在启动时执行（否则满池下的首次紧急压缩可能直接以 CUDA OOM 带崩调度器，`#44592`）；对采用新的依赖或后端切换（如 FlashInfer 0.7.2 + KDA auto backend）的改动，按「未验证的新变更」而非「安全补丁」对待，先在小规模灰度上跑 GPU 测试与精度回归。
7. **AMD/异构路径单独记录设备、dtype 与并行布局，不做跨厂商外推**：本期 MI355X（gfx950 同代）、gfx1250（ROCm 10.1 / aiter pin `2bb549`）、SM100/SM107、Snapdragon/Adreno、SYCL 各自的 FP4/MXFP4/NVFP4/FP8 路径互不可推；引用 vLLM 博客的 24.4% / 87.4% 时必须同时带上「8 卡 / TP8 / AgentX / 50 tokens/s/user P90 / 各自已发布 offload 序列内插值 / 不含 TTFT」这一组限定条件。
8. **本期欠账（下一期补取）**：TensorRT-LLM `#18599`（96 head MLA autotune）、`#19981`（CUTLASS MoE workspace 回收）、`#20290`（GVR V2 top-K 溢出开销）与 llama.cpp `#30613`（k-pool 视图共享）、`#30449`（pinned host buffer 4 GB 上限）的 **PR 正文**因本期 GitHub API 速率上限（60 次/小时）未能取得，需在配额充足时补取以确认适用范围与性能声明。

---

## 8. 社区与视频资料

- **HN Algolia 公开 API 本期可访问（HTTP 200）。** 在 2026-10-16 00:00–24:00 UTC 窗口内按 “LLM inference”“vLLM”“SGLang”“llama.cpp”“KV cache”“speculative decoding”“TensorRT-LLM”“inference engine” 检索，与本期主题直接相关的仅下列几条，**除注明为官方来源者外均标注为二手线索/未核验**：
  - **“Llama.cpp on a jailbroken PS5: run any GGUF model natively on the console”**（HN 2026-10-16T18:32:35Z，**173 points / 45 comments**，https://github.com/cobanov/PS5LM ）——本期社区热度最高的推理相关条目；来源为个人 GitHub 仓库，**二手线索，未核验**其性能与实现细节；
  - **“Local and Distributed Large LLM Inference with MLX and RDMA on Mac Studio”**（HN 2026-10-16T03:34:45Z，7 points / 1 comment，https://machinelearning.apple.com/research/mlx-distributed-inference ）——来源为 **Apple Machine Learning Research 官方研究页**（官方一手页面），但**本档案未核验其正文与性能数字**，故仅列为线索；与上一期同一条目重复出现；
  - **“AbridgeKV: Compressing Long Reasoning into the KV Cache of Learned Pause Tokens”**（HN 2026-10-16T05:57:08Z，1 point / 0 comments，https://github.com/JackCai1206/abridgekv ）——**二手线索，未核验**，主题为把长推理压缩进 KV cache，与本档案 KV 主线相关但为个人项目；
  - “Outlines – Structured Outputs for LLMs”（3 points，https://github.com/dottxt-ai/outlines ）、“A Statistical Analysis of LLM Benchmaxing”（1 point）、“Client-Side LLMs Product Demo: PDF to AudioBook”（5 points，YouTube）等——均非推理引擎性能主题，**不计入本期结论**。
  - **检索未命中：** “SGLang”“speculative decoding”“TensorRT-LLM” 三个关键词在窗口内**均返回 0 条**，说明当日社区讨论未聚焦这三者，**不构成「无动态」的结论**（官方提交仍在密集进行）。
- **Reddit 未纳入本期采集范围**（往期记录其 API 受限），本期不据 Reddit 内容形成结论。
- **视频资料：** 本期未核验到可确认发布日期与内容的官方视频，故不列入。
- **技能检索：** 本期通过 WorkBuddy 技能检索（关键词：“LLM inference optimization”“推理引擎热点追踪”“github trending hotspot archive”“KV cache optimization”“AI news daily digest”）返回的候选仍为面向**媒体式热点聚合**或**方案输出**的技能（如「AI 热点日报」「GitHub AI趋势追踪」「GitHub热门项目」「开源雷达 · GitHub 热门榜」「LLM推理加速方案制定与优化配置」），与本任务所需的「**官方提交/发布 API 按日期窗口采集 + 证据分级热点归档 + hotspot 目录与远端同步**」不匹配，故**未安装任何外部技能**；本期采集与归档由内置 HTTP 抓取、本地文件与 Git 能力完成。已加载的 `agent-browser`、`find-skills` 两个技能本期未产生实际调用需求（本期全部数据源均为可直接 HTTP 访问的公开 API 与官方页面，无需浏览器自动化）。

---

**归档状态：** 已按 `hotspot/2026-10/` 月度目录归档。
**证据边界：** 事实与观点已分开；官方提交/发布中的性能标题（24.4% / 87.4% / 2.89× / 14.2% / 1.5–2× / 3.05–4.91% 等）均按原文配置范围记录，不推导为普遍结论；TensorRT-LLM 与 llama.cpp 两节**仅基于官方提交标题**（因 GitHub API 速率上限未取到 PR 正文），已显式标注；社区条目一律标注为二手线索且未核验；本期与上一期（覆盖 2026-10-15）之间**无归档空档**。
