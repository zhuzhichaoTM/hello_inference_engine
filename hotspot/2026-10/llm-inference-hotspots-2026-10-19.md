# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-19（北京时间）
- **关注窗口：** 2026-10-18 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-18 08:00 – 2026-10-19 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-18 UTC 窗口，分别返回 **22 / 68 / 4 / 5** 条，均 HTTP 200）、四家官方 releases API、vLLM **v0.32.0 发布说明全文**（39,727 字符），以及 **61 个 PR 页面正文**（HTML 直抓，不消耗 API 配额）。**窗口特征：** 2026-10-18 为**周日**，TensorRT-LLM（4 条）与 llama.cpp（5 条）提交量处于周末低位，属正常节律，**不构成「活动减少」的结论**；SGLang 的 68 条中有 **31 条集中在 12:31–12:42 UTC 的 11 分钟窗口内**，是一个 10 个 PR 的技术栈被集中合入，不是当日独立发生的 31 次开发活动（详见第 3 节）。vLLM 官方博客窗口内**无新文**（最新仍为 2026-10-16）。HN Algolia 公开 API 可访问（HTTP 200），但 `sglang`、`speculative decoding`、`inference optimization` 三个查询在窗口内**均 0 命中**，仅 `KV cache` 有 2 条低分条目。Reddit 未纳入采集范围；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** **vLLM v0.32.0 正式发布**（2026-10-18T06:50:48Z，814 commits / 363 contributors / 121 位新贡献者），这是本期最重要的事实。SGLang 仍为 **v0.5.21（2026-10-02）**；TensorRT-LLM 仍为 **v1.4.0rc0（prerelease，2026-10-14）**；llama.cpp 正式版仍为 **v0.7.0**，窗口内滚动 nightly 共 **5 个标签（b11717–b11721）**，最新 b11721 发布于 2026-10-18T18:04:19Z。

---

## 1. vLLM v0.32.0 正式发布（窗口内唯一正式版本，本期头号事件）

**主题**  vLLM 发布 **v0.32.0** 稳定版（`published_at` 2026-10-18T06:50:48Z，非 prerelease），规模为 **814 个提交、363 位贡献者（含 121 位新贡献者）**。这是自 v0.31.0（2026-10-05）后 13 天的累积版本，也是前一档档案中 `#61989` 反复提到的「v0.32.0 已在镜像层进入 RC」的最终落地。

**内容详情**  与推理优化直接相关的官方条目（均取自发布说明原文）：

- **低精度 KV Cache 全面铺开（本节最值得读的一条主线）：**
  - **NVFP4 KV cache** 扩到 **SM8x 与 SM12x**（`#46963`）、**SM90**（`#60620`）、**SM12x 的 XQA 解码**（`#60452`），以及 **SM100 上 MiniMax-M3 的稀疏注意力**（`#59300`）。
  - **GLM-5.x 在 SM100 上默认 FP8 KV cache**（`#60140`、`#60281`）：官方口径为**每 rank KV 内存减半，吞吐 +2.3–5.5%，代价是 TTFT +7–11%**（该数字出自 PR 自测）。
  - **SM90 的 `fp8_ds_mla`** 让 **GLM-5.3-Flash 输出吞吐最多 +31%**（`#59246`）。
  - **ROCm 新增 4-bit `ultraquant_4bit` KV cache**，官方称密度约为 FP8 的 **2 倍**（`#57057`）。
- **Model Runner V2（MRV2）：** GPU n-gram 投机器支持**变长 draft**（`#40704`）、**Elastic EP**（drain-then-commit 扩缩，`#53934`）、**MLA 模型的 PP + prefill context parallelism**（`#59139`）、`STOCK_TORCH_COMPILE` 模式（`#59079`）；**MRV2 在 ROCm 上转默认**，并移除模型特定的 MRV1 例外（`#56360`）。
- **投机解码：** FlashInfer trtllm-gen 上的**融合多步 draft decode**（`#58371`）、**Mamba2 MTP 的 FlashInfer ReplaySSM**（`#52928`）、DFlash/DSpark draft 的**图内元数据**（`#60692`）。
- **GLM-5.3 / GLM-5.3-Flash 专项：** 稀疏 MLA index 转换**跨层复用**（该 kernel 3.5–3.9×，`#59464`）、**融合多步 draft decode 在并发 1 时 +13.3%**（`#57443`）、**HiSparse 按请求驻留缓存**（B200 上 GLM-5.3-NVFP4 长上下文全驻留 + 高并发时**最多 +58% 吞吐**，`#60083`）、MHA 跳过 qlnorm（TTFT 4.4–7.7%，`#58845`）、长上下文 indexer prefill 按 TP 分片（1M 上下文 TTFT 最多 −25%，`#54951`）。
- **MoE 通信与大规模型服务：** `--moe-backend flashinfer_cutedsl` 支持更多模型（`#54049`）、FlashInfer 单侧 all2all 的 **FP8 combine（DEP8 下 +4%）**（`#57995`）、**NCCL symmetric-memory MoE reduce-scatter（延迟降 9.6–16%）**（`#49194`）、**DeepEPv2 混合 NVLink+RDMA 模式转默认**（`#57991`）、Kimi-K3 shared experts 走 DeepEP v2（`#56875`）。
- **KV 连接器 / PD：** NIXL host-buffer 拷贝合并（`#54483`）、混合 block size（`#58860`）、Mooncake 合并 hybrid/MLA RDMA region（`#57952`）、**DeepSeek-V4.1 P/D 对已加载 KV 跳过 bounded replay**（`#59197`）。
- **硬件专项：** SM100 原生 **64-head FlashMLA 稀疏 prefill**（TTFT 1.16–1.18×，`#60029`）、ROCm **AITER MegaMoEv2**（`#59685`）与可选 shared-expert 融合（3–8%，`#61004`）、MI355X 上 DeepSeek-V4.1-Flash 可选 mono decode（`#60397`）、Kimi-K3 prefill checkpoint（TTFT 约 +20%，`#58344`）、GLM-5.3-Flash split-K 稀疏 MLA 解码（ITL 降 5–15%，`#58584`）、原生 `merge_attn_states`（2.4×，`#60159`）。
- **其他性能条目：** MoE weighted-sum 调优（1.17–2.37×，`#59731`）、更快的 KV block 空闲队列（`#60269`）、回退 detokenization 在 32K 时 CPU 降 69%（`#60497`）、`--api-server-count > 1` 时短请求 req/s 提升 43–85%（Qwen3-0.6B，32 输入/1 输出，`#60723`）。
- **破坏性变更（与推理部署直接相关者）：** 移除 **cascade attention**（`#61126`，`--disable-cascade-attn` 仍可解析但仅告警，v0.33.0 移除）；移除 **`mamba_cache_mode="all"`**（`#58997`，`--mamba-cache-mode all` 现在校验失败）；**`max_tokens` 不再从上下文窗口预留**，改为输出上限（`#57035`）；移除 **torch.compile 的 sequence parallelism 与 AsyncTP pass**（`#60517`）；**放弃 Python 3.10**（`#60402`）；默认 CORS 从 `*` 改为无（`#61009`）。

**应用场景**  低精度 KV 的硬件矩阵选型（NVFP4 已覆盖 SM8x/SM90/SM100/SM12x，FP8 在 SM100 上成为 GLM-5.x 默认）、长上下文全驻留 KV（HiSparse）、PD 场景下已加载 KV 的重算规避、MoE 集合通信的对称内存与 FP8 combine、ROCm 侧的 4-bit KV 与 MegaMoE。

**研判**
- **事实：** v0.32.0 是一个「低精度 KV + MRV2 默认化」的版本；官方明确给出了 GLM-5.x FP8 KV 的**双向代价**（吞吐 +2.3–5.5%，TTFT +7–11%），这比此前只讲收益的表述更诚实。
- **事实（本期最需要注意的一条，已用 commit 祖先关系核验）：** **`#61989`（把单节点 FlashInfer all-reduce 重新偏好 MNNVL，修 `#56891` 引入的 3–5% 解码回退）没有进入 v0.32.0。** 核验方式：取 `#61989` 的 merge commit `2b85d9cb`，与 tag `v0.32.0` 做 compare，返回 `status=diverged, ahead_by=345, behind_by=16`；对照组 `#61126`（`e1a50862`）与 `#60140`（`fba31f31`）均返回 `status=behind, ahead_by=0`，即确实在版内。**结论：v0.32.0 是带着已知的单节点 TP>1 解码回退发布的**，该修复只存在于 main，需要等 v0.32.1 或自行 backport。这一点无法从发布说明看出——`#61989` 与 `#56891` 均未被写入 v0.32.0 的 release notes。
- **观点：** 三点。(1) **对单节点 NVSwitch 机器的 TP>1 部署，不要直接升级到 v0.32.0**，或升级后立刻用 `VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl` 做 A/B；`#61989` 给出的 A/B 是 Qwen3.5-397B NVFP4 8×B200 从 5462 → 5728 tok/s、TPOT 21.78 → 20.22 ms，这个量级与「3–5%」的官方描述一致。(2) **GLM-5.x 的 FP8 KV 默认化是「用 TTFT 换吞吐」的默认值**，长 prompt、低并发场景（如 RAG、文档问答）会直接吃到 7–11% 的 TTFT 惩罚，这类负载应显式设 `--kv-cache-dtype` 覆盖默认。(3) **cascade attention 的移除**值得单独记一笔：它此前是长上下文 + 多请求共享前缀场景的一种显存/算力折中，移除意味着官方判定其维护成本高于收益，依赖该路径的部署需要在 v0.33.0 前迁移。

**来源**
- Release：https://github.com/vllm-project/vllm/releases/tag/v0.32.0 （2026-10-18T06:50:48Z）
- 未入版修复：`#61989` https://github.com/vllm-project/vllm/pull/61989 （merged 2026-10-17T09:55:28Z，merge commit `2b85d9cb65c1f56c347cb7f56f1596a6b8b28373`）；被修对象 `#56891`
- 核验命令：`GET /repos/vllm-project/vllm/compare/v0.32.0...2b85d9cb65c1f56c347cb7f56f1596a6b8b28373`

**证据等级：** 官方一手（release 页面 + GitHub compare API）。吞吐/TTFT/ITL 等百分比均为**各 PR 自测口径**，非官方统一基准；「未入版」结论为本次基于 commit 祖先关系的**独立核验**，非官方声明。

---

## 2. vLLM：10-18 提交（22 条）——DBO × DeepEP 的跨流缓冲复用崩溃、投机解码语法掩码改由 GPU 驱动

窗口内 22 条提交，其中 16 个 PR 抓取到正文。发布当日的提交以 **bugfix + CI/构建** 为主，符合「发版日」节律。

### 2.1 `#60735` DBO 与 DeepEP 高吞吐后端之间的跨流缓冲复用（本期最值得读的 bug）

**主题**  `DeepEPHTPrepareAndFinalize` 在 DBO 下让 DeepEP 把 dispatch 输出分配在 **comm stream** 上、却在 **compute stream** 上读；combine 的输入反向跨流。**缓存分配器从未被告知这些跨流读取**，于是另一个 microbatch 的下一次 DeepEP 调用可能复用该 block 并在其被读取前覆盖。配上 DeepGEMM FP8 MoE 时，引擎在启动阶段即因 `ep_scatter` 非法访存崩溃。

**内容详情**  修复方式是对每个跨流张量调用 `record_stream`。官方给出的代价：DBO 打开时**没有 host sync、没有额外 kernel**，每 MoE 层每 microbatch **至多 9 次 `record_stream`**，外加每个被记录 block 释放时一个 CUDA event；**吞吐代价未测量**。验证在 1×8 H200、GLM-5.3 FP8、DP4×TP2、EP8、`--all2all-backend deepep_high_throughput --enable-dbo` 上做，GSM8K 5-shot / 1319 题：

| Linears | DBO | 本 PR | GSM8K flexible / strict | 每 worker microbatch 步数 |
|---|---|---|---|---|
| CUTLASS | on | 否 | 崩溃，2/2 次 | 死在第一步（2×4096 token） |
| CUTLASS | on | 是 | **0.927 / 0.927** | 100–149 |
| DeepGEMM | on | 是 | 0.912 / 0.913 | 100–149 |
| DeepGEMM | off | 是（空操作） | 0.923 / 0.923 | 0 |

崩溃时的核心转储定位到 `_fwd_kernel_ep_scatter_2` 的 `tl.load(expert_map + expert_id)`：从 DeepEP topk ids 经 `torch.where` 得到的 `expert_id` 越界，SASS 在 `expert_id != -1` 检查之后仍加载 `expert_map + expert_id*4`。在 0.29.0 上做的隔离实验显示：加 `CUDA_LAUNCH_BLOCKING=1`、`PYTORCH_NO_CUDA_MEMORY_CACHING=1`、或强制 host sync 都能「绕过」崩溃（GSM8K 0.920/0.945/0.943），说明**根因确实是异步跨流复用**而非 kernel 逻辑错误。文中另指出 `#52177`（给 DeepEP v2 加 DBO 支持）引入了同样的 comm stream 切换，**可能需要同样处理**。

**应用场景**  DBO（Dual-Batch Overlap）+ DeepEP HT + DeepGEMM FP8 MoE 的 GB300/H200 大规模 MoE 部署。

**研判**  **事实：** 这是一个**只在「DBO 打开 + DeepEP HT + DeepGEMM FP8 MoE 默认值」三者同时成立时出现**的启动崩溃，而这三者恰好是当前大 MoE 高性能部署的常见组合；`#46990` 此前为 ROCm 加过一个 HT dispatch/combine 后的 host sync，很可能就是同一隐患的局部补丁。**观点：** 这类「分配器不知道跨流读取」的问题在 CUDA 生态里反复出现，特征是**症状随机、绕过手段（关缓存/串行化）都有效、但都会吃掉性能**。判断一个引擎是否真修好了，标准应该是「修完后 host sync 数量不增加」，本 PR 的 `record_stream` 方案符合这一点（DBO 下 0 host sync），因此比 host sync 类补丁更值得跟进。

**来源**  `#60735` https://github.com/vllm-project/vllm/pull/60735 （提交 2026-10-18T22:47:48Z，作者 aniravich）

**证据等级：** 官方一手（PR 正文 + 完整 GSM8K 表与 core dump 栈）；**吞吐代价官方未测量**。

### 2.2 `#52477` 结构化输出的语法掩码改由 GPU logit 计数驱动

**主题**  让 GPU 的 per-request logit 计数决定施加哪些语法掩码行，取代 CPU 侧构建的 mask→logit 映射。

**内容详情**  scheduler 为每个结构化输出请求发出**固定宽度**的 `num_spec_tokens + 1` 行掩码；MRV2 kernel 每个语法请求启动一次，**只施加设备端 `cu_num_logits` 中活跃的位置**（例如自适应验证在零预算步后只剩 bonus token）。这**删除了 `_build_grammar_mapping`**，worker 不再依赖 `cu_num_logits_np` 或已调度的 draft 数。batch-sharded sampling 按整块切片，MRV1 把块解包进自己的 packed 布局。单测 102 passed / 1 xfailed，**GPU 测试标记为 pending**。

**应用场景**  投机解码 + 结构化输出（JSON schema / 正则 / 语法）同时开启的场景，尤其是**自适应验证**会在运行中改变每步的有效 draft 数。

**研判**  **事实：** 这是对 `#52436`（已合入的 CPU 侧修复）的替代方案，两者同作者协作完成。**观点：** 值得记录的原因不在性能，而在**「哪一侧是真相源」的方向性选择**：把「本步有几个有效 logit」的判定从 CPU 挪到 GPU，消除了 CPU 侧对 `cu_num_logits` 的复制与推断。随着自适应验证、变长 draft 这类「每步形状不定」的特性普及，**这类 CPU/GPU 计数不一致会成为一个反复出问题的类别**，本 PR 是把它一次性收敛掉的做法。

**来源**  `#52477` https://github.com/vllm-project/vllm/pull/52477 （提交 2026-10-18T18:29:21Z，作者 LucasWilkinson）

**证据等级：** 官方一手（PR 正文）；**GPU 测试未完成，正确性仅有单测覆盖**，这一点官方已自述。

### 2.3 `#62129` MRV2 投机解码在流水并行下的启动死锁

**主题**  修复 `#59260` 引入的 PP 启动死锁：只有最后一个 PP stage 持有 rejection sampler，而额外的 greedy warmup pass 以「是否有 rejection sampler」为门控，于是前面的 stage 跳过该 pass、最后一个 stage 在等它们的激活。

**内容详情**  改为以 `speculative_config` 为门控，保证所有 PP stage 执行**匹配的 forward 次数**。回归测试 34 passed；GLM-4.7-Flash 冒烟（MRV2、TP2、PP2、BF16、MTP 单 draft token、JIT warmup + CUDA graph）生成恰好 1024 token，**记录 532 个 draft token、492 个被接受**，进程正常退出。官方明确**未做精度基准**。

**应用场景**  PP > 1 且开启 MTP/投机解码的 MRV2 部署。

**研判**  **事实：** 这是「warmup 次数必须跨 stage 对齐」这一类约束被破坏后的典型症状——**不是死锁在通信原语上，而是死锁在「有人在等别人不执行的 forward」**。**观点：** 492/532 ≈ 92.5% 的接受率只在 `temperature=0` + 固定 prompt 的冒烟条件下成立，不能当接受率基准；但「532 draft / 492 accepted」这个记录方式本身值得借鉴——**它让「warmup 通过」与「投机真的在工作」两件事同时可验证**。

**来源**  `#62129` https://github.com/vllm-project/vllm/pull/62129 （提交 2026-10-18T06:51:32Z，作者 chenjr97）

**证据等级：** 官方一手（PR 正文 + 单测结果）；精度未测。

### 2.4 其余 vLLM 提交（按主题归类）

| PR | 提交时间(UTC) | 主题 | 关键内容 | 证据等级 |
|---|---|---|---|---|
| `#61819` | 10-18 14:39 | ROCm / DSv4 性能 | 把**未融合的 shared expert 放到 aux stream**、与 router + AITER MegaMoE 并行，沿用 `VLLM_DISABLE_SHARED_EXPERTS_STREAM` / `..._TOKEN_THRESHOLD` 约定（默认 ≤256 token 的步生效）。DeepSeek-V4-Pro on 8×MI355X、DP8 attention + EP8：**P90 交互性 c128 27.13→27.87（+2.7%）、c64 48.26→51.33（+6.4%）**，c128 输出 tok/s/GPU 335→346（+3.3%）；c64 total tok/s/GPU **−1.5%**；gsm8k 0.9697/0.9704 → 0.9727/0.9735 | 官方一手（单次运行） |
| `#61829` | 10-18 12:12 | 构建 / SM103 | 镜像里 DeepEP 只编了 `9.0a 10.0a`，**SM103（B300/GB300）无法运行**。根因：`(a)` cubin 只在其精确计算能力上加载，且 DeepEP 构建**不嵌 PTX**，故 sm_100a 代码在 10.3 上不加载，表现为 `cudaErrorUnknown (999)`。加 `10.3a`（需 CUDA ≥12.9）。8×B300 上 `test_intranode.py` **24/24 dispatch/combine 检查通过**，最佳 dispatch FP8 0.41 ms、combine 0.67 ms。**官方撤回**了早前版本中「DeepEP 有性能收益」的说法——那次对比被重复 prompt 的 prefix cache 命中放大了 | 官方一手（含主动撤稿） |
| `#60896` | 10-18 07:14 | 构建 / GDN | `FUSED_GDN_DECODE_ARCHS` 只在 CUDA ≥13.0 分支里设置，导致**所有 CUDA 12.x 构建（含已发布的 cu129 镜像）都没有 `fused_gdn_decode_post_conv_mtp`**，GDN 模型回退到 Triton 投机解码路径。补 8.0/8.6/8.9/9.0a。A100 上单测 71 passed/29 skipped/2 failed → **102 passed**；CUDA-graph 重放时每 rank 形状 H=2/HV=6/K=V=128/4 MTP token，**kernel 时间为 Triton 路径的 0.48–0.76×（1–32 请求）** | 官方一手 |
| `#62072` | 10-18 16:49 | 前端正确性 | beam search 只把 `tokenizer.eos_token_id` 当结束，**不认 generation config 里的额外 eos**，导致 chat 模型答完继续写下一轮直到 max_tokens。GLM-Edge-1.5B-Chat 上：main 输出 48 token 的「自问自答」，修后 8 token 正常停止，`stop_reason: 59253`；Phi-3-mini 上 main 每 beam 都以 `<|end|><|endoftext|>` 结束（`stop_reason 32000`），修后以 `<|end|>` 结束（`32007`） | 官方一手 |
| `#62274` | 10-18 18:45 | ROCm 稳定性 | `#61118` 无条件设 `torch._inductor.config.batch_invariant`，但 ROCm CI 镜像的 torch 无此选项且对未知 key 抛错，导致 **10 个 AMD nightly 用例在 init 即失败**（MI250/MI300/MI355 的 Batch Invariance、LoRA、Spec Decode Draft Model、MI250 E2E Scheduling Accuracy）。改为「torch 有该选项时才设」 | 官方一手 |
| `#61892` | 10-18 20:29 | Rust 前端 | Rust 前端 gRPC 把所有结构化输出约束降级到 guidance 后端，而 guidance 只解析旧版 `triggers/structures`，**发 xgrammar 格式 structural tag 的 gRPC 客户端每个请求都 INTERNAL**（`KeyError: 'triggers'`）。改为 gRPC 路径统一走 xgrammar，并让 xgrammar 后端自己编译 `choice` | 官方一手 |
| `#62221` | 10-18 07:48 | 发版流水线 | `#61302` 把 `/recipes` 加进 `.dockerignore`，但那 188 个文件是被跟踪的，`.git` 以 bind mount 进构建，`git diff --quiet` 于是判定「Repo is dirty」，**CUDA x86_64/aarch64 与 ROCm 的 wheel 构建全部失败**，连带所有下游镜像与发布步骤 | 官方一手 |
| `#58761` | 10-18 02:43 | ROCm 默认切换 | **ROCm 10.0（TheRock）成为默认 ROCm 镜像与 wheel**，ROCm 7.2 保留为 `-rocm72` 过渡 | 官方一手 |
| `#60877` | 10-18 11:30 | 前端 | `/score` 使用 Sentence Transformers CrossEncoder 声明的 chat template | 官方一手 |
| `#61454` | 10-18 11:38 | Qwen3.8-Flash-Next | 清理 HC 与 PLE 实现：删冗余 custom-op 注册/包装/fake impl，直接调 Triton kernel；`lora_rank`→`hc_lowrank` | 官方一手 |
| `#61759` | 10-18 10:37 | 多模态 | 修 OpenCV 视频后端对 WebM 输入的帧数缺失 | 官方一手 |
| `#62305`/`#62267`/`#62302` | 10-18 09:26–15:26 | 文档 | `/inference/v1/generate` 请求/响应字段与 DP rank header；RL 训练侧**不依赖 vLLM 时的权重传输 HTTP 协议**；MoE routing replay（R3）的 routed experts 返回 | 官方一手 |
| `#56549`/`#62279`/`#62213`/`#62161` | 10-18 02:49–12:13 | 构建 / CI | 同时请求带后缀的 CUDA arch 时保留 plain arch；Language Models (Extended Pooling) 逐测试超时；`test_rocm_aiter_sampler_defers_import` 顺序依赖；给 AITER JIT 构建留出 MoE 并行启动 watchdog 余量 | 官方一手 |

---

## 3. SGLang：10-18 提交（68 条）——一个 10-PR 的 `--enable-unified-memory` 技术栈在 11 分钟内整体落地

窗口内 68 条提交，其中 38 个 PR 抓取到正文。**必须先说明结构**：其中 **31 条集中在 12:31–12:42 UTC**，是同一个 10-PR 技术栈（`#43168` → `#44973`）被顺序合入，**不是当日独立发生的 31 次开发活动**。剔除这 31 条后，当日实际独立变更为约 37 条。

### 3.1 头号事件：`--enable-unified-memory` 技术栈整体合入（10 个 PR）

**主题**  一个以「统一内存池」（unified memory pool：每个 token 的 entry 内并排存放所有层的 K/V、SWA、Mamba 状态）为核心的技术栈，在 12:31–12:42 UTC 的 11 分钟内从栈底到栈顶依次合入，共 10 个 PR。栈内顺序（由旧到新）：`#43168` → `#43170` → `#43171` → `#43325` → `#43326` → `#44969` → `#44970` → `#44971` → `#44972` → `#44973`。

**内容详情**  逐条要点：

1. **`#43168`（Fix，SWA 保留整窗）：** SWA 窗口非页对齐时，解码驱逐会在 prefix-cache 插入点所在页之下留下**不足一整窗**的残留（127 token 窗口、页大小 64 时 `pre_len = 63 mod 64` 只保留 64 token），导致插入的前缀**无法被复用**。改为把 `max(window, page)` **向上取整到整页**再推导驱逐边界；页对齐窗口与页大小 1 行为不变，非对齐窗口每请求至多多留一个 SWA 页。
2. **`#43170`（Perf，去 host sync）：** `UnifiedSWAAllocatorBase` 用布尔掩码 `reps[swa_v2p > 0]` 过滤每个 segment 释放，而**掩码索引会走 `nonzero`，把计数拷回 host 并阻塞调度器直到在飞 forward 排空**。gpt-oss-20b（SM90、页大小 256）服务轨迹中，`free_swa_segment` 与 `_release_page_reps` **每个 prefill 约发 2.6 次此类 sync**；prefill 周期因此比 paged SWA 分配器长 **+1.3..+3.3%（投机）/ +2.5..+7.0%（非投机）**，全部是 GPU 空转。改为固定形状释放 + 新增两条 allocator 不变式（`SGLANG_INVARIANT_CHECK` 下异步、从 warn 起变响）。
3. **`#43171`（Perf，GDN prefill 后端）：** `flashinfer_gdn_prefill_default` 在 `enable_page_major_kv_layout` 下直接拒绝，而 `--enable-unified-memory` 正是 page-major，于是 Qwen3.5 的 GDN prefill 停在 Triton chunk kernel。SM90（Qwen3.5-9B、页 256、无投机）上 **Triton chunk 每周期 +17.8 ms，FlashInfer GDN 为 10.7 ms**，端到端每个 attention backend 都 **−1.1..−1.5% 吞吐**。关键论证：prefill 阶段 `forward_extend` 本就会把非连续池的 conv/SSM 状态 gather 成连续 per-sequence 拷贝再 scatter 回去（`needs_state_gather`），所以**布局对 prefill 无影响**。
4. **`#43325`（Feat，非对称 KV 行）：** MiMo-V2-Flash 的 K 头 192 宽、V 头 128 宽，`--enable-unified-memory` 启动时直接拒绝（"does not yet admit asymmetric K/V rows"），只能退回默认池；而默认池在启动时按两类层切分内存，**长解码时会在另一类层仍有空间的情况下撤回请求**。统一池把每个 token 的 K 与 V 行并排放在同一 entry、各用自己的偏移，因此池层面无需拒绝；剩下的是后端属性——FlashInfer 的 `paged_kv_t` 有独立 K/V stride 但两 tile 共用一个 `head_dim`，sglang 的 MHA plan 路径从不传 `head_dim_vo`，trtllm_mha 拒绝不等 QK/V。故非对称模型可用后端收窄为 **triton / fa3 / fa4**；**fa4 是 SM100 上非对称模型的解析默认**，不纳入会让该模型在自身默认配置下无法启动。
5. **`#43326`（Feat，draft 融合进 SWA 与 recurrent 子池）：** 此前只有「全 full-attention 层」的 draft 能融合，带 SWA 层的 MTP 头（MiMo-V2-Flash）或带 recurrent state 的（Inkling 的 per-depth conv-chain 头）会被拒绝，混合模型因此无法在 `--enable-unified-memory` 下跑 NEXTN。改为按**子池种类注册表**放置（先 `full`，再 `swa`、`mamba`）：SWA 子池要保留 draft 会读的范围（`CacheInitParams.swa_retain_window` 取 target 窗口与 draft 触及范围的较大者），树的锁、前缀匹配、backup/load-back、请求保留都用同一跨度；Mamba 子池把 draft 的 recurrent state 排在 target 自身状态之后，`clear_slots`/`copy_from`/`move_kv_cache` 以整 entry 为单位操作，于是 radix checkpoint、slot 重置、compaction 搬迁都会带着 draft 状态走。
6. **`#44969`（Refactor，解码元数据与池所有权分离）：** 解码元数据 kernel 此前伸手进 pool 对象选静态 SWA 映射，并复制了融合 kernel 主体。改为运行时解析映射张量、kernel 只吃张量与 stride。18 个静态 GPU 微基准（页大小 1/128、batch 1/16/64、FULL/SWA/lengths-only）交替 CUDA-graph 采样对比：**输出完全一致，中位延迟 +0.29%，最大增幅 0.036 µs（+2.18%）**。
7. **`#44970` / `#44971` / `#44972` / `#44973`（Perf）：** 分别降低分配与 SWA 退役开销、融合 strided 状态更新与 replay 索引翻译、融合 draft 写窗口 gather 与 KV 翻译、捕获 DSPARK 的 FA4 verify 元数据。**这四个 PR 与 `#44969` 官方均声明「不单独归因吞吐增益」**，性能只在合并栈上报告（见下）。

**技术栈端到端数据（官方，`#44973` 正文，合并栈 `fe3202b8`，基于 main `3671503e`）：** Inkling-Small-NVFP4 + Inkling-Small-DSpark，**4×B300 / TP4**，FA4 target 与 draft，CUDA graph，页大小 128，greedy，48 请求 × 固定 512 输入 / 2048 输出：
- **精确输出 token ID：48/48 全部一致**；**每请求 verify 次数：48/48 一致**；**聚合 accept length：default 4.43470023 vs unified 4.43470023（完全一致）**。
- **输出吞吐两轮：default 8486.2 / 8484.6 tok/s，unified 8548.8 / 8552.7 tok/s（约 +0.75%）**。
- 官方明确：这是**受控负载上的合并栈验证**，不是单个优化的前后对比，也不是任务精度评测。

**统一池相对默认池的服务级数据（官方，`#43326` 正文，bench_serving，default 每配置跑两次取 A/A 噪声底）：**

| 模型 | 请求 × 输入/输出 | default req/s | unified req/s | 变化 | A/A | accept length default / unified |
|---|---|---|---|---|---|---|
| MiMo-V2-Flash | 100 × 512 / 4096 | 0.509 / 0.510 | **0.841** | **+65.1%** | 0.2% | 2.79 / 2.78 |
| MiMo-V2-Flash | 192 × (4096 共享 + 128) / 2048 | 0.555 / 0.554 | **0.837** | **+51.1%** | 0.2% | 3.19 / 3.24 |
| Inkling-Small | 100 × 512 / 4096 | 0.442 / 0.442 | 0.440 | −0.5% | 0.1% | 1.29 / 1.29 |

**应用场景**  混合 SWA / Mamba / MLA 模型（MiMo-V2-Flash、Inkling、Qwen3.5 GDN、GLM-5.x）在**长解码 + 投机解码**下，默认池因「两类层各占一半内存、互不借用」而提前撤回请求；统一池让所有层共享同一 token entry，从而把这种碎片化消掉。MiMo 的 +65% 不是 kernel 变快，而是**池不再因内存切分而缩批**。

**研判**
- **事实：** 这个技术栈解决的是一个**结构性**问题（层类型各自占池 → 长解码下互相饿死），而非某个 kernel 慢。官方给出的对照实验设计是本期质量最高的：**A/A 噪声底、精确输出 token ID 全比对、accept length 一致性检查、合并栈而非单点归因**。
- **观点：** 三点。(1) **MiMo 的 +65%/+51% 必须理解成「默认池在该负载下的容量利用率问题」**，不是 unified memory 的通用加速比；同一栈在 Inkling-Small 上是 **−0.5%**，这正是「没有碎片化可回收时，统一池只是多了一点间接层」的证据。看到 +65% 就去全量切换是误读。(2) **`#43170` 的 2.6 次 host sync/prefill 是本期最可迁移的诊断**：布尔掩码过滤在 overlap scheduler 下会隐式变成「阻塞到 GPU 空闲」，而这个代价在 profiler 里表现为「prefill 周期变长」，很容易被误判为 kernel 慢。**凡是分配器里出现 `mask` / `nonzero` / `item()` 的释放路径，都值得按这个模式查一遍**。(3) **`#43326` 把 draft 状态塞进 target 的子池**，意味着 draft 与 target 的 KV 生命周期彻底绑死——好处是 checkpoint/compaction 天然一致，代价是**任何一侧的几何变更（窗口宽度、v_head_dim、层数）都会变成启动期拒绝**；官方用「注册表 + 按子池 keying」来管理这个组合爆炸，是这类系统目前最实用的工程解法。

**来源**  10 个 PR 链接：`#43168` https://github.com/sgl-project/sglang/pull/43168 、`#43170` .../43170 、`#43171` .../43171 、`#43325` .../43325 、`#43326` .../43326 、`#44969` .../44969 、`#44970` .../44970 、`#44971` .../44971 、`#44972` .../44972 、`#44973` .../44973 （合入时间 2026-10-18 12:31–12:42 UTC）

**证据等级：** 官方一手（PR 正文 + 官方基准表）；所有百分比为作者自测，`#44973` 明确标注「合并栈验证、非单点归因」。

### 3.2 HiCache：三个静默数据损坏级问题被修

**主题**  窗口内 HiCache / host pool 一侧有 5 个 PR，其中 3 个修的是**不报错但结果错**的问题。

**内容详情**
- **`#40917`（host pool 改由 buffer info 构建）** —— 三个行为变化，每个都是一个静默错误：
  1. **MHA host 现在在构造时拒绝非 MXFP8 的 KV scale 布局**。main 上 **FP4 KV 池会得到一个「只存 packed payload、不存 scales」的 host**，**load back 时静默恢复出错误数值**。
  2. Aibrix storage 改为从 host pool 取 dtype 而非 device pool。main 上 **FP8 KV 时，经 Aibrix SDK 的 L1 写入再读回，每 block 6656 字节里有 416 字节被损坏**；本 PR 逐字节不变。
  3. **prefill CP 层切分下的 packed MTP draft 层**现在映射到本 rank target 层之后的 host 行。main 上 GLM-5.2 PD prefill worker 配 `--enable-dsa-cache-layer-split` + EAGLE 时，**第一次 write-through 就报 `index 78 is out of bounds for dimension 0 with size 40`**。
- **`#43333`（unified memory 与 `page_first_direct` 互斥）** —— unified 池下每层设备 KV 行是 **strided** 的（一个 entry 含所有层的 K/V），而 `page_first_direct` 的直接拷贝仍把一页当作**一整段连续 `page_size * row_stride`** 搬。Inkling 上这段是**真实页大小的 8 倍**：backup 会写穿到相邻 host 页，load-back 会重写该行其他层与后续行；越过分配末尾时 `cudaMemcpyBatchAsync failed ... error=invalid argument`，调度器直接死。`page_first_direct` 会强制 direct io 后端，而 **Mooncake 等 storage 后端会解析到它**，所以普通 HiCache 参数就能触发。修复：unified 时把 `page_first_direct` 解析为 `page_first` + kernel io（JIT kernel 按 stride 寻址），并让直接拷贝路径 `_require_packed_device_rows` 显式失败。**官方明确说明「没有展示错误输出的复现」**——因为 unified 的 HiCache KL 测试没有从 host load 回任何东西，看不到损坏。
- **`#41907` + `#43816`（MLA Mamba 状态按 TP rank 加键，file / SeaweedFS 后端）** —— MLA 在各 TP rank 间共享 KV 存储键，但 **Mamba checkpoint 是分片的**，于是 rank 之间互相覆盖。GLM-5.3-Flash TP4 + file L3、16K 前缀仅从 L3 恢复：**main 上 544 个 Mamba payload 里有 408 个取到了错误 rank 的数据，修后 544/544 正确**。同一修复此前已用于 NIXL（`#43594`），HF3FS 在 `#43817`。
- **`#44968`（Mamba host 传输改由 pool info 路由）** —— 混合 Mamba 栈此前 controller 路由取自 `req_to_token_pool.mamba_map`、而 host 池取自 `DevicePoolInfo`，**两个真相源**；改为 assembler 捕获一次 info 后两者同源。KL 验证：Inkling（FULL+SWA+Mamba，确定性）L2/L3 **avg_kl_div 均为 0.0**，device/host cached tokens 10496/15104 与 main 一致。

**应用场景**  HiCache（L2 host / L3 存储）在 MLA + Mamba 混合模型上的正确性；unified memory 与 Mooncake/`page_first_direct` 的组合；FP8/FP4 KV 经 Aibrix、TensorCast 等 storage 的往返。

**研判**  **事实：** 三个问题都属于「**不崩溃、不报错、精度悄悄掉**」类别，且**现有测试都没覆盖到**（`#43333` 官方自述、`#40917` 的 Aibrix 416 字节损坏也是靠逐字节比对才发现）。**观点：** 这是本期最应该进自查清单的一组。**共性根因是「同一份布局信息存在多个真相源」**：host 池自己读 device 池的私有字段重建布局、page_first_direct 假设「一页 = 一段连续内存」、Mamba 键假设「KV 键共享则状态也共享」。**判据很简单：只要一个系统里同时存在「池的布局描述」和「消费方对布局的假设」，就一定会在某个新池布局上错位**——`#40917` 的解法（设备池导出 typed `MHABufferInfo`/`MLABufferInfo`，host 只吃这份 info）是正确方向。另外 `#41907` 的 **408/544** 这个数字值得记住：**在 MLA + TP 的组合下，「键共享但值分片」是一个默认会踩的坑**。

**来源**  `#40917` https://github.com/sgl-project/sglang/pull/40917 、`#44968` .../44968 、`#43333` .../43333 、`#41907` .../41907 、`#43816` .../43816 （提交时间 2026-10-18 07:07–15:21 UTC）

**证据等级：** 官方一手（PR 正文含逐字节/逐 payload 对比）；`#43333` 的「错误输出复现」官方标注为未提供。

### 3.3 DeepSeek-V4.1 与 PD 解耦：SWA bounded replay 的四处连带修复

**主题**  DSv4.1 的 encoder/decoder SWA bounded replay 在 PD 场景铺开后，窗口内出现 4 个连带问题，全部由同一作者（hnyls2002）在当日修完，并在 GB300 上给出 GSM8K 数据。

**内容详情**
- **`#45035`（默认值）** —— **CUDA 上 PD 的 prefill 与 decode worker 均默认开启 decoder SWA bounded replay**（与 standalone server 一致）；ROCm 上 PD 仍需显式 `--enable-decoder-swa-bounded-replay`（官方称 replay 在 ROCm 的 PD 下未验证）。GB300 + DSv4.1-Flash、DEP2 prefill + DEP2 decode、DSpark block 5、MegaAttention、MegaMoE、Mooncake：两个 worker 均记录 `resolved to True`，**GSM8K 97.12%，零 KV 传输失败**。
- **`#44952`（PD prefill 跑 encoder replay）** —— 解除 PD prefill worker 在 CUDA 上的拒绝；decode 从不 replay（prefill 把最后一个 SWA 窗口发过去）。代价：**DSpark draft ring 向上取整到整页**，每请求 slot 从 `window + draft tokens` 变为 `ceil(window + draft, page)`，页 256 时 **134 → 256 行，每 rank 多约 55 MB**。**已知限制（官方自述）：** 前缀命中且新增不足 128 token 时，decode 的 paged draft SWA 会收到 prefill draft ring 从未写过的行，**会降低接受率**（输出不变，因为 target 会验证每个 draft token）。
- **`#45037`（PD decode draft 窗口起点对齐 prefill 的 valid-from）** —— 修的就是 `#44952` 那条已知限制。draft 池按请求 slot 记 `swa_valid_from`，PD 把它作为一个新的 `REQUEST_ROW` 状态分量 slot-to-slot 传给 decode（Mooncake 与 NIXL 都支持），decode 在自己 draft 池里收到该标记；`ComputeDsparkWindowGather` 增加可选 `valid_from`（torch 与 triton 两版）。GB300 + DSv4.1-Flash、DEP2 prefill + DEP2 decode、DSpark block 5：**2048 缓存前缀 + 43 token 问题、64 新 token、128 请求**，第 1 轮 accept length **3.16 → 3.85**，第 2 轮 **2.87 → 3.80**（该负载每次命中留 85 行缺口）。GSM8K 在「共享 8-shot 前缀、命中落在页边界、无缺口」的负载上：main 两轮 97.12% / 96.89%，本 PR 96.74%。
- **`#45040`（prefill chunk 给 encoder replay 行留空间）** —— 有 encoder SWA bounded replay 时，前缀命中会在同一步里跑最多 **128 个缓存 token**；scheduler 先按剩余 chunk 定 extend 长度、**之后**才扣这 128 行，于是「填满 chunk 的那个请求」会超出 chunk 最多 128 行。当 MegaMoE token 上限设为 per-rank chunk 时，该步会回退到 Triton fused MoE，而它在 DSv4.1-Flash 的 block scales 上断言失败（`assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]`）。GB300 上：**main 在第二个请求（24576 prompt + 24576 追加）的第一块为 16384 新 token + 128 replay 行，服务崩溃**；修后为 **16128 + 128**，请求完成。
- **`#44948`（前向前交换 late-layer DP 行数）** —— DP attention + MegaMoE 下，一个 rank 可能 replay breakable prefill graph、而对端 idle rank 走 eager。decoder SWA bounded replay 在模型 forward 内部的 CPU group 上 gather late-layer 行数，**graph replay 永远走不到那里**：idle rank 阻塞在 gather、prefilling rank 阻塞在 MegaMoE（DeepGEMM grid sync 超时）。**每个「一 prefill + 一 idle」的步都挂**，包括多模态启动 warmup。改为在 `ForwardBatch.prepare_mlp_sync_batch` 里交换。GB300、DP2/EP2：`--cuda-graph-backend-prefill breakable` 下服务能起（此前挂在多模态 warmup），GSM8K **96.97%**；关 prefill graph 为 **97.12%**。
- **`#44963`（限住 encoder BCG 元数据显存）** —— 把已有的 low-ratio BCG 元数据上限与 full-metadata index buffer 共享**扩展到 encoder replay 也生效**。GB300×4、DEP4 上 42 个回归测试通过，52 个记录请求的完整输出 token 序列与 usage/cache/rank/finish 元数据**与 main 完全一致**，KV 容量保持 39,928,064 token。
- **`#43760`（PD 角色切换复用启动期 decode graph）** —— decode 模式启动时在 `init_cuda_graphs()` 里捕获了 decode graph，但**该路径没有置 `decode_cuda_graph_captured`、也没填 `decode_cuda_graph_capture_bs`**，于是 decode→prefill→decode 往返后 `ensure_decode_cuda_graphs()` 会重新捕获。官方称在 v0.5.21 源码上复现出**首次回到 decode 的负载即 CUDA 非法访存**（两次独立并发运行）。

**应用场景**  PD 解耦 + DSv4.1-Flash + DSpark + MegaMoE 的 GB300 部署；DP attention 下 prefill/decode 混跑；PD 角色切换（同一进程先 prefill 后 decode）。

**研判**
- **事实：** 这 6 个 PR 共同描述了一个模式：**bounded replay（用缓存前缀替代重算）在 PD 下会把「谁写了哪一段 KV」变成一个跨进程的契约**，而这个契约在 main 上是隐式的（靠「窗口从哪开始」的默认假设），于是出现了「接受率下降但不报错」和「chunk 超限导致 MoE 回退断言」两类问题。
- **观点：** (1) **`#45037` 的 accept length 3.16→3.85 是最能说明问题性质的数字**——它修的不是吞吐，而是**「decode 在读 prefill 从没写过的行」**；输出仍然正确（target 会验证），所以任何只看 GSM8K 的验收都发现不了，只有看 accept length 才能看见。**这提示 PD 场景的验收指标必须包含接受率，不能只看精度与吞吐。** (2) `#45040` 揭示了一个更隐蔽的耦合：**chunk 大小、MegaMoE token 上限、replay 行数三者相等时，任何一处「事后扣减」都会把请求挤出快速路径并触发断言**。这类「三个预算相等」的配置在调优时很常见，属于高风险配置。(3) `#44948` 是「graph replay 跳过 host 侧集合通信」的经典陷阱——**任何放在 forward 内部的 CPU collective，在 breakable graph 下都会变成单向等待**。

**来源**  `#45035` https://github.com/sgl-project/sglang/pull/45035 、`#44952` .../44952 、`#45037` .../45037 、`#45040` .../45040 、`#44948` .../44948 、`#44963` .../44963 、`#43760` .../43760 （提交时间 2026-10-18 03:55–22:54 UTC）

**证据等级：** 官方一手（PR 正文 + GB300 GSM8K/accept length 表）；`#45037` 的 97.12→96.74 为**同一作者自测的单次对比**，官方已注明该负载「命中落在页边界、无缺口」，与接受率提升的负载不同。

### 3.4 投机解码与调度器：两条「只在特定并行组合下才崩」的修复

| PR | 主题 | 关键内容 | 证据等级 |
|---|---|---|---|
| `#45043` | 删除 `--speculative-skip-dp-mlp-sync` | 该 flag 在**非 PD 的 attention DP** 下允许一个 DP rank prefill 而另一个 decode，把 decoding rank 的 batch 送进 prefill 路径，抛 `AttributeError` 并**把服务打挂**（GLM-4.7-Flash + MTP draft、DP2 复现）。删除后该 flag 变成 unrecognized-argument 错误；`SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1` 可安全跳过同一 sync。官方称**在其他所有配置下该 flag 不改变任何运行路径** | 官方一手 |
| `#45041` | `prepare_for_decode` 丢弃 prefill 的 embedding 覆盖 | `replace_embeds` / `replace_positions` 未随 `input_embeds` 一起清空，导致请求的 `positional_embed_overrides` **在 prefill 之后继续挂在 running batch 上进入每个 decode 步**。DP attention 下持有该请求的 rank 会走 eager decode，从而进入一个对端从未加入的集合通信 → **挂死**（Qwen3-30B-A3B、TP2、DP2 复现） | 官方一手 |
| `#43559` | lazy Mamba prefix lock 在 donation 重试前释放 | lazy overlap 下，下一个 decode 可能在前一个 prefill donate checkpoint **之前**消费掉临时 tracking slot；donation 此时需要一个替换 slot，而**被消费的前缀状态仍被锁着**，池耗尽并断言。GB300 上红绿验证：base `f05da43d` + 新用例 → `AssertionError: Can not alloc mamba cache`；PR head `c2950396` → 通过。反向对照：只移除本 PR 的释放块 → 同样断言 | 官方一手（含反向对照） |

**研判**  **事实：** 两条修复（`#45043`、`#45041`）都只在「DP attention + 投机解码」的组合下触发，且**症状都是服务挂死而非报错**。**观点：** `#45041` 值得单独记一笔——**「prefill 期的临时状态没在 prefill 结束时清掉」是一个会随 batch 复用一直传播的 bug**，而且它的表现（一个 rank 走 eager、进入无人加入的 collective）会让人误判为通信库问题。`#43559` 的反向对照（只删自己的修复看是否复现）是本期方法论上最规范的一次验证。

**来源**  `#45043` https://github.com/sgl-project/sglang/pull/45043 、`#45041` .../45041 、`#43559` .../43559

**证据等级：** 官方一手。

### 3.5 其余 SGLang 提交（按主题归类）

| PR | 主题 | 关键内容 | 证据等级 |
|---|---|---|---|
| `#44601` | AMD monokernel 跳过无用元数据 | `sgl-amd-mono` 的 DSv4.1 forward 用 K2i 给每个 index source 打分，**每步都构建 aiter decode schedule 却无人读**，且该构建被录进 decode CUDA graph、每次重放都跑。新增 `StepMetadata` 枚举 + `MonoSpec.unread_metadata`（默认空，插件 API 版本保持 1）。MI350X、TP4、DSpark 模拟接受率 3.51、1024 token 中位 3 轮：8k/1 请求 1.600→1.593（−0.5%），**128k/1 请求 1.643→1.595（−2.9%）** | 官方一手 |
| `#44740` | NIXL 预构建 dlist | 混合模型状态（Mamba/GDN、SWA、DSA indexer、QSA、MiniMax、MXFP8 block scales）此前每请求 `initialize_xfer` 重建并重校验描述符。改为 peer 注册时按 decode peer 建一对 strided 源/目的 dlist 覆盖所有整 slot 状态分量，每请求一次 `make_prepped_xfer` + int32 选择子；decode 通过新的注册帧 23 广播状态池长度；stride 不匹配在注册时记录并让该 peer 的所有传输失败（而非用错 stride 寻址或杀 bootstrap 线程） | 官方一手 |
| `#44404` | Mooncake 升到 v0.3.14-rc1 | `01d1eb2a` 下**跨节点请求走了 Mooncake 的单节点 GPU-IPC 传输**，prefill→decode 的 KV 传输失败（`hipIpcOpenMemHandle failed (201)`，随后 "remote mooncake session is not alive"）。ROCm 10 镜像 + DSv4-Flash FP4、1 prefill + 1 decode on 2×MI355X：**GSM8K 0.926，并发 1–256 全部完成，零传输错误**；未修镜像首个请求即失败 | 官方一手 |
| `#44412` | mori 用配对 dst 索引查 elided DSA 条目 | `#41769` 加了 `is_elided_entry(src_len, dst_lens, dst_index)`，`#37358` 把同一循环改成 `for i, j in pairs`，但**合并后仍传 `i`**。当 prefill PP stage 与 decode 条目配对位置不同（`i != j`）时读错 decode 条目：**要么跳过必须发送的条目（decode 永远拿不到该层 index-K），要么发往 `descriptor` 为 None 的 elided 条目**。官方明确：`i != j` 需要 per-layer id，而 DSA 状态分量目前不注册，故该路径**当前不可达**，本 PR 是为将来铺路 | 官方一手 |
| `#43967` | bench_serving 增加 goodput 与 SLO 达成率 | 引入 vLLM 的 `--goodput`：`--goodput ttft:2000 tpot:100`，报告 request goodput（达标请求/秒）、SLO attainment（达标/发出）与各 SLO 单独达成率。示例输出 28.37 req/s、63.48% | 官方一手 |
| `#44642` | custom all-reduce v2 按 size 缓存 eager launch | H200（x86）上 eager TP all-reduce 在 **5.2 µs launch 之上还有约 3.8 µs 的 Python 开销**，拆分为：kernel wrapper（`get_all_reduce_module` 查找、`comm.world_size`、结果检查）1.18 µs、`CustomAllReduceV2.custom_all_reduce`（字节数、再次 `_pick_config`）1.04 µs、`should_custom_ar`（`_pick_config`）0.62 µs | 官方一手（逐层剖析） |
| `#44158` | eager 下直接调用 `register_custom_op` 注册的 op | 每个注册 op 都经 `torch.ops.sglang.<name>` 调用，普通张量下 dispatcher 只做参数装箱再回到同一 Python 实现，每次几微秒。py-spy 显示 FLUX.2 / Qwen-Image / Ideogram（H200、2 GPU、eager、host 受限尺寸）下 `torch/_ops.py __call__` **占 host 时间 5–8%**；Grace 主机（GB200/GB300）多 TP rank 时该开销进入端到端延迟 | 官方一手（profiler 归因） |
| `#44075` | sgl-router 重发与熔断器开关 | **SGLang 5 s 关空闲 keep-alive，sgl-router 却复用至 90 s**，引擎关闭连接瞬间发出的请求在收到任何响应前失败；每角色一个 worker 时 hyper 与 `--retry-max-attempts` 都不重发，客户端拿到 502；**三次这样的重置就会打开 worker 熔断器**，此后该角色请求持续 502/503 而 worker 其实在服务。改为 3 s 过期 + 发送失败且无响应时向同一 worker 重发一次 + 新增 `--disable-circuit-breaker` | 官方一手 |
| `#43043` | Kimi 图像缓存身份加固 | 多模态 padding 把 item hash **压缩到 30 bit**，不同图像因此可共享 radix 前缀键并复用错误 KV。Kimi K2.5/K3 还需要区分预处理配置与 patch grid。改为**四个 62 bit 哨兵**；官方解释了为何不用四个 30 bit：radix 匹配可**在第一个哨兵后即停止**，Python 与 Rust 的 radix 测试在**页大小 1**、以及**页大小 16 且前面有 15 个文本 token** 时都复现了假命中，后续哨兵无法阻止这种部分匹配 | 官方一手（含反例论证） |
| `#44666` | Muse Spark runtime（3/5 层） | 82 文件 +9,175/−224；**B300 的 `test_muse_spark_12_b300_pd.py` 注册被临时关闭至 2026-11-01**，因为 B300 job 117611788593 在服务启动阶段失败、零测试执行（日志显示 tokenizer 加载前发生整模型 HF 缓存清理，符合并发缓存清理竞态） | 官方一手（含自述未通过项） |
| `#42833` | tokenized 多模态生成 | Rust gRPC 监听器接受预分词请求（`input_ids`），但 `GenerateRequest` 无 media 字段。新增 `image_data`/`video_data`/`audio_data`（repeated string，URL/路径/base64，顺序与 `input_ids` 中占位符一致）。**官方明确未跑模型输出精度测试**，仅验证请求转发 | 官方一手（精度未测） |
| `#44716` | Inkling 约束输出按原生消息 token 框定 | `--reasoning-parser inkling` 下 `response_format`/regex/ebnf 不遵循 Inkling 的线格式（每条消息用 `<|message_model|><|content_text|>…<|end_message|>` 框定）。实测：答案头缺失、JSON 前后夹散文、空白跑到 max_tokens、regex payload 重复（`Yesyes`）、`continue_final_message` + 约束 **54/54 失败** | 官方一手 |
| `#40941`/`#39475`/`#34385`/`#44471`/`#44643`/`#44196`/`#44761`/`#44180`/`#44478`/`#44606` | diffusion 路径性能 | dLLM decode graph 按默认最大运行请求数定尺寸；Cosmos Dreams/Multiview 支持；LingBot 因果注意力的可选序列并行策略；Triton launch 重放去掉 runner 的 per-launch hook 元数据；按 linear 缓存 fused linear-GELU 判定；Qwen-Image 2.1 的 fused gate/up + SwiGLU MLP on comfy_kitchen INT8；FLUX.3 动态 LoRA 走 PrimsTS GEMM；FP32 AdaLN 表的 fused indexed modulation；`residual_gate_add` 操作数每次调用只分类一次；导入 runtime 时推迟 full GC | 官方一手 |
| `#39264`/`#44950`/`#44951`/`#43592`/`#44937`/`#44903`/`#44693`/`#44947`/`#44892`/`#44938`/`#45004`/`#44946` | 权重同步 / 路由 / 文档 / CI | 从元数据分配 packed 接收缓冲；`--max-in-flight` 改名 `--max-inflight-requests`；`TextGenerate`→`Generate` 转换补 media 字段；sgl-router README 重写 + 架构指南；DSv4.1-Flash 1017 cookbook；AMD cookbook 1016；popular-models 用 Muse Spark Open 替换 Kimi-K3；**revert 掉 Muse Spark Open 的 SWA ratio 计算器文档**；GLM-5.3-Flash 服务配方拆成独立测试；重新启用 Inkling unified cache linker KL 测试；Inkling 缓存 linker KL 测试注册到 extra-b；BCG replay 契约测试设置 KV 池 | 官方一手 |

---

## 4. TensorRT-LLM：10-18 提交（4 条）——overlap scheduler 的「退役请求占座」把 ctx_only 吞吐拉低 7.5%

**主题**  窗口内仅 4 条提交（周日），但其中 `#20250` 是本期**诊断质量最高的一份性能回归复盘**，值得完整记录。

**内容详情**
- **`#20250`（NVBug 6919819，非 attention-DP 的 overlap 运行补上退役请求的座位余量）：**
  - **现象：** GB200 `ctx_only qwen3-235b-fp4 8k1k con1024` 的 perf-sanity 输出吞吐从 **6.03 降到 5.58（−7.5%）**。
  - **关键判断：** 该 post-merge 序列**不是斜坡，而是三个离散台阶**，且每个台阶都同时出现在共享同一 context-worker 配置的两个兄弟用例（con64、con1）上。本 PR 修的是最大的那个台阶（build 2935–2938，**−2.6%**）。
  - **根因：** `#18346` 停止对 perf-sanity 的 ctx_only 用例强制 `disable_overlap_scheduler=True`，这些用例于是带着 overlap scheduler 运行。ToT 的 CI worker 日志（`max_batch_size=4`、TP=1、无 attention DP、8k prompt）显示：**`num_scheduled_requests` 直方图为 `{1: 2561, 3: 2560}`，节奏是 `3, 1, 3, 1, …`，从不出现 4**。原因是 overlap 下发出最后一个 token 的请求**晚一个迭代才被拆除**（`_process_previous_batch` 在下一迭代取批调度之后才跑），它继续占着序列槽并被计入准入；而 context 请求在其 prefill 步就结束，**于是每个被调度的请求在紧接着的 fetch 中都在退役**。`_fetch_new_requests` 的非 attention-DP 分支把 `len(active_requests)`（含退役者）从 `max_num_active_requests` 里减掉，**每迭代只能补上上一迭代没用掉的座位：3 → 1 → 3 → 1，平均批 2，而配置是 4**。
  - **为什么之前没修到：** `#18457`（NVBug 6627795）**正是修这个的**——`ADPRouter.exclude_retiring_requests` + `should_enable_overlap_headroom` 给座位池多一个 micro-batch。但**两半都以 `mapping.enable_attention_dp` 为门控**，于是纯 TP 的 context server 与所有非 ADP 的聚合 overlap 运行都留着这个 bug。索引池（`KVCacheManagerV2.max_admissible_sequences`）早已对任意非 PP 的 overlap 运行放宽，**缺的只是座位门控与准入计数**。
  - **改动与影响面：** `should_enable_overlap_headroom` 去掉 `enable_attention_dp` 项（PP、overlap 关、KV cache manager V1、hybrid/SSM、MRoPE-delta 模型的排除不变）；非 ADP 分支在 router 报告 `exclude_retiring_requests` 时从准入计数里扣掉 `count_retiring_requests`，但**退役请求在「是否空闲等待」的判断里仍算活跃**（与 ADP 分支一致）。**受影响范围：** 任何非 PP + overlap 开 + KVCacheManagerV2 + 非 attention-DP 的运行，即**聚合 TP 服务器与 PD 的 context / generation worker**；GB200/H200 perf-sanity 的 PD 配置中有 **14 个**带这样的 context worker，另有 **116 个聚合 perf-sanity server 配置**（B200 等）现在也过这个门控。
  - **内存代价：** 每座位缓冲从 B 行翻到 2B 行——sampler 状态、penalty 计数、seed/top-p 数组、guided-decoder 位掩码、投机 draft-prob 缓冲；**词表大小的那些是懒分配、仅在使用该特性时分配**，且与 attention-DP 运行自 `#18457` 起就一直承担的一致。PD context worker 的发送方在 `start_transfer`（KV 传输前）就释放座位，**故仍在传 KV 的请求不占座位，多出的 micro-batch 不会被它们吃掉**。
- **`#20163`（perf-sanity 用例更新）：** 新增 11 个用例——Kimi-K3 的 4 个 `gen_only_no_context` AgentX 配置（con138/63/88/220，GB300 1–4 节点 8–16 GPU，均 `fp4-dspark_agentx_ctx16_dwdp16`）、GB300 MiniMax-M3、DeepSeek-V4 DSpark 的 `ctx_only`（con1156、ctx2、DEP8、EPLB0、dspark3）。同时**删掉 gpt-oss 与 time_breakdown 用例**。Kimi-K3 的 AgentX 配置需要 harness 里新增 **DWDP ctx 启动路径**（`submit.py`、`slurm_launch_draft.sh`、`dwdp_ctx_rank.sh`、`test_perf_sanity.py`）；**Kimi-K3 的 e2e 与 ctx_only 用例尚未加入，因为 main 上 Kimi K3 的 MoE 不支持 DWDP**。
- **`#20523` / 锁文件提交：** 为 main 的 pre-merge 64813 豁免 1 个失败用例；从 nightly 流水线检入最新锁文件。

**应用场景**  聚合 TP 服务与 PD 的 context/generation worker 上的 overlap scheduler；perf-sanity 基准的横向可比性。

**研判**
- **事实：** 这是一个**被 `#18457` 修过、但因门控条件过窄而只修了一半**的 bug；且它的表现（吞吐缓慢下降）极易被误判为「模型变慢」或「驱动回退」。
- **观点：** 三点。(1) **「不是斜坡，是三个离散台阶」这个判断方法是本期最值得复用的**——它把「性能缓慢劣化」从「需要 bisect 全区间」压缩到「只需查台阶处的提交」，并且用兄弟用例（con64/con1 共享配置）交叉确认了配置而非模型的问题。(2) **`{1: 2561, 3: 2560}` 这个直方图是整个复盘的核心证据**：它把「平均批 2」这个模糊现象变成了「批大小在 1 与 3 之间振荡」的确切描述，从而直接指向「每迭代只能补上一迭代的空座」。**遇到「吞吐低于配置预期」时，先看批大小直方图，而不是先看 kernel。** (3) **影响面从「ADP」扩到「所有非 PP overlap」意味着这是一个普适修复**：116 个聚合配置 + 14 个 PD context worker 同时受益，但代价是**每座位缓冲翻倍**——在 `max_batch_size` 很大、guided decoding 或投机开启的部署里，这部分显存需要重新核算。

**来源**
- `#20250` https://github.com/NVIDIA/TensorRT-LLM/pull/20250 （提交 2026-10-18T03:20:41Z，NVBug 6919819）
- `#20163` https://github.com/NVIDIA/TensorRT-LLM/pull/20163 （提交 2026-10-18T12:03:08Z）
- 关联：`#18346`（停用强制 disable overlap）、`#18457`（NVBug 6627795，ADP 侧同一修复）

**证据等级：** 官方一手（PR 正文含 CI 日志直方图与逐台阶归因）；`#20250` 的性能验证基于 GB200 上 post-merge CI wheel 的单次对照，官方已注明两臂安装同一 wheel、修复臂再叠加补丁。

---

## 5. llama.cpp：10-18 提交（5 条）——Qwen4Exp 的 HC 门控链融合、非连续 SUM 的四处对齐

窗口内 5 条提交（周日），全部附具体数字或明确的能力边界。

| PR | 主题 | 内容详情 | 应用场景 | 证据等级 |
|---|---|---|---|---|
| `#30602` | CUDA 融合 Qwen4Exp 的 `SCALE → SIGMOID → SCALE → hc_post` | 移植 Vulkan 的 `HC_POST_GATE` 融合（`#29520`）。`hc_combine` 把 hc_post 权重构造成 `2*sigmoid(inject/hc)`；`dsv4_hc_post_f32` 现在接受 gated 模板参数，直接从原始权重算 `scale_out*sigmoid(scale_in*p)`，**四 kernel 变一 kernel**。官方口径：RTX Pro 6000 上 Qwen4-next **约 3% TG** | Qwen4Exp 系列在消费级/工作站 Blackwell 上的 decode | 官方一手（作者自测，未给逐项表） |
| `#28851` | 非连续 SUM（CPU / CUDA / Vulkan / WebGPU） | `test-backend-ops` 含 `SUM(f32, ne=[33,256,1,1], permute=[1,0,2,3])`，CUDA 拒绝、CPU 参考实现崩溃。**根因是门控与 kernel 实际能力不匹配**：CUDA 的 `supports_op` 要求 `ggml_is_contiguous_rows`，而 kernel 只要求 `ggml_is_contiguously_allocated`（`ggml_cuda_op_sum` 里已断言），CUB 的扁平归约对任何无空洞视图都成立（求和与顺序无关）；该检查是 `#16559` 为保 CI 绿而加的，**并非基于 kernel 真实能力**。CPU 侧 `ggml_compute_forward_sum_f32` 断言 `nb[0] == sizeof(float)`，故只要 CPU 当参考/回退就中止，改为保留 `ggml_float` 累加的分步路径。**Vulkan 与 WebGPU 的门控接受任意「行连续」输入，但 kernel 把张量当单一扁平块求和，切片视图会返回错误结果**，改为同样用 `ggml_is_contiguously_allocated` | 多后端一致性与正确性；`#14909`/`#16559`/`#28175` 的后续 | 官方一手（RTX 3070 Ti Laptop + CUDA 13.3、Vulkan 双厂商实测；WebGPU 未本地测、由 CI 验证） |
| `#30572` | FlashAttention 的越界检查剪枝 | 当 `k_VKQ_0 == k_VKQ_max - nbatch_fa` 时不再做 oob 检查。例：`k_VKQ_max=256`、`nbatch_fa=128` 时，第二块 128→255 此前仍走 `oob_check=true`，而**两块都是满的、不需要检查** | CUDA FlashAttention 的 decode/verify 路径 | 官方一手（逻辑论证，未附性能数字） |
| `#30239` | 在 FFN 融合路径上恢复 ESIMD kernel | 给新增的 FFN 融合路径补上既有 ESIMD kernel 支持。B70 上 `tg128`：gemma3 1B Q8_0 **246.92±1.34 → 253.49±1.40（+2.7%）**、qwen2 3B Q4_K_M **169.45±0.24 → 173.81±0.36（+2.6%）** | Intel Arc / B70 上的 decode | 官方一手（含误差棒的逐模型表） |
| `#30627` | Hexagon 支持 Hadamard `MUL_MAT` 与 `GET_ROW`（flat/tiled Q1_0、Q2_0） | 承接此前把 baseline Q2_0 与 Q1_0 一起合入的结果，补齐 Hadamard 乘与取行。官方称 **Bonsai-2 Q1/Q2 现已完全可用**，但要更好利用 DDR 占用优势（相对 Q4 及以上）仍需进一步优化 | 端侧 Qualcomm Hexagon NPU | 官方一手（作者声明用了 AI 辅助推理、测试与验证） |

**研判**
- **事实：** 5 条中有 2 条（`#28851`、`#30572`）是**纯正确性/门控对齐**，2 条是性能（+2.6~3%），1 条是端侧能力补齐。
- **观点：** **`#28851` 是本期 llama.cpp 侧最有价值的一条**，因为它暴露了一个跨后端系统性问题：**`supports_op` 的门控比 kernel 实际能力更严（CUDA）或更松（Vulkan/WebGPU）**。前者导致「明明能算却被拒绝」，后者导致「接受了但算错」——**后者更危险，而且它在 main 上一直是错的**。这类「门控与 kernel 各写各的」的问题在支持 8+ 后端的项目里几乎必然出现，判据是：**`supports_op` 里的条件应当逐条对应 kernel 里的一条断言**，对不上就是隐患。`#30239` 的 +2.7%/+2.6% 说明**「新增融合路径时忘了把老的后端特化 kernel 接回去」是一个会反复发生的回归**——B70 这类小市场设备最容易在融合重构中被静默降级。

**来源**  `#30602` https://github.com/ggml-org/llama.cpp/pull/30602 、`#28851` .../28851 、`#30572` .../30572 、`#30239` .../30239 、`#30627` .../30627 （提交时间 2026-10-18 02:17–17:48 UTC）

**证据等级：** 官方一手。

---

## 6. 当前会话已形成的累计内容结论

按主题回溯本目录自 2026-09-25 起的连续档案，截至本期已沉淀的判断如下（**仅列已被多期独立证据支持者**）：

1. **KV Cache 的低精度化已成默认路径，且进入「按硬件分档」阶段。** 从 09-25 的探索、10-13 的 KV 维度对齐、到本期 v0.32.0 的「NVFP4 覆盖 SM8x/SM90/SM100/SM12x、FP8 成为 GLM-5.x 在 SM100 的默认」，主线是**同一件事在四种硬件上分别落地**。同时官方开始公开代价（FP8 KV：吞吐 +2.3–5.5%，TTFT +7–11%），说明**低精度 KV 已从「优化项」变成「需要按负载取舍的默认值」**。
2. **KV 分层与迁移的瓶颈从「带宽」转向「每次命中的固定开销」。** 本期 arXiv 2610.21500 给出了可直接用的判据：远端 KV 层只有在交付带宽超过 `B* = sR`（该模型/GPU 上 2.53 GB/s）时命中才划算，且**存在约 0.33 s 的每次命中固定引擎开销**，使**短于约 17k token 的 prompt 在任何带宽下都无法胜过重算**。这与 10-14 的 NIXL Push、10-12 的双向 KV 传输构成同一条线索：**KV 复用的收益由「prompt 长度 × 命中率 × 层带宽」三者共同决定，任何单看命中率的评估都会高估收益**。
3. **PD 解耦的验收指标必须包含接受率，不能只看精度与吞吐。** 10-13（PD 失败语义）、10-17（PD 延迟释放导致 KV 页重复释放）、到本期 `#45037`（decode 读 prefill 从未写的行，accept length 3.16→3.85，而 GSM8K 反而从 97.12 降到 96.74）——**三期独立证据指向同一结论：PD 下的「正确」与「高效」是两套指标**。
4. **「静默错误」是本领域最稳定的故障类别。** 累计已记录：SGLang `#44220`（prefill CUDA graph 重放与逐层异步加载冲突，产出错误 logits 并被内容寻址缓存备份）、`#44847`（EPLB 漏搬 expert scale，GSM8K 97.5%→72.5%）、本期 `#40917`（FP4 KV host 丢 scales、Aibrix FP8 KV 每块坏 416/6656 字节）、`#41907`（MLA Mamba 408/544 取错 rank）、`#43333`（unified + page_first_direct 写穿 8 倍页）。**共性：都不崩溃、都不报错，且现有测试都不覆盖。**
5. **「同一份布局/状态信息存在多个真相源」是上述静默错误的第一大根因。** 本期 `#40917`（设备池私有字段 vs host 重建）、`#43333`（页假设 vs strided 行）、`#41907`（KV 键共享 vs 状态分片）、`#44968`（`mamba_map` vs `DevicePoolInfo`）四条同源。**工程解法已收敛为「导出 typed buffer info，消费方只吃这一份」**。
6. **调度器的「隐式阻塞」比 kernel 慢更常见。** 本期 `#43170`（布尔掩码释放 → 每 prefill 2.6 次 host sync，prefill 周期 +1.3..7.0%）、TRT-LLM `#20250`（退役请求占座 → 批大小在 1/3 间振荡，吞吐 −7.5%）。**两者的 profiler 表现都是「某阶段变长」，但根因都不在被变长的阶段里。**
7. **发版节奏与「已修但未发」之间存在可观测的窗口，且会实际影响用户。** 本期首次用 commit 祖先关系证实：vLLM v0.32.0 **不含** `#61989`（修 3–5% 单节点解码回退），该修复在发版前 21 小时合入 main 但未进发布分支。**结论：不要假设「最新 release 含最新修复」。**
8. **agent 生成推理内核正在改变维护方式。** 10-16 的 vLLM megakernel 博客提出 fast-track RFC 四原则（禁止改 KV cache 管理/采样/输入准备），本期 v0.32.0 与多个 PR 都出现 `[CC]`/`codex/claude generated` 标注。**趋势是：性能改动更多落在模型目录内，core 的稳定性反而更有保障。**

---

## 7. 今日可执行清单

**P0（今天就能做，风险低）**

1. **若在单节点 NVSwitch 机器上跑 TP>1 的 vLLM：先别升 v0.32.0，或升级后立刻 A/B。** 用 `VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl` 与默认值对比，指标看 **TPOT 与 output tok/s**（`#61989` 的实测是 Qwen3.5-397B NVFP4 8×B200：5462→5728 tok/s、TPOT 21.78→20.22 ms）。若确认回退，cherry-pick `2b85d9cb` 或等 v0.32.1。
2. **升级 v0.32.0 前逐条核对破坏性变更：** `--mamba-cache-mode all` 会校验失败、`cascade attention` 已移除、`max_tokens` 语义变为输出上限（**长 prompt 场景下实际可用输出可能变多**）、Python 3.10 不再支持、torch.compile 的 SP/AsyncTP pass 已删。
3. **若用 GLM-5.x + SM100 + 长 prompt 低并发（RAG/文档问答）：显式覆盖 `--kv-cache-dtype`。** 新默认 FP8 KV 会带来 **+7–11% TTFT**，换来的 2.3–5.5% 吞吐在该负载下不划算。
4. **若跑 DBO + DeepEP high-throughput + DeepGEMM FP8 MoE：确认已含 `#60735`。** 否则启动阶段会以 `ep_scatter` 非法访存崩溃；临时绕过手段是 `CUDA_LAUNCH_BLOCKING=1` 或 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`（**两者都会吃掉性能，只能当诊断手段**）。

**P1（本周内，需排期）**

5. **按第 6 节第 5 条做一次「多真相源」审计。** 重点查三处：host 池是否直接读 device 池私有字段重建布局；`page_first_direct` 类「一页 = 一段连续内存」的假设在 strided 池下是否成立；MLA/TP 组合下「键共享但值分片」的状态是否有独立键。**验收方式：强制命中从 host/L3 返回，逐字节或逐 payload 比对**，而不是只看 GSM8K。
6. **PD 部署补上接受率监控。** 采集 accept length（completion tokens / verify count）与 cached tokens，而不是只采吞吐与精度。基线参考：`#44973` 的 4.43470023（Inkling-Small-NVFP4 + DSpark、4×B300、TP4）与 `#45037` 的 3.16→3.85。
7. **若用 TRT-LLM 的聚合 TP 或 PD context worker：评估 `#20250`。** 收益是恢复被占座吃掉的吞吐（参考 −7.5% 的回归），代价是**每座位缓冲翻倍**（sampler 状态、penalty 计数、seed/top-p、guided-decoder 位掩码、投机 draft-prob；词表大小的懒分配）。
8. **若用 SGLang `--enable-unified-memory`：按模型分档决定是否开启。** 判据是「默认池是否因层类型切分而提前撤回请求」——MiMo-V2-Flash（非对称 KV + SWA draft）**+65.1%/+51.1%**，Inkling-Small（draft 状态可融合但无碎片化）**−0.5%**。**不要按「unified memory 更快」来推。**

**P2（观察项，不急于行动）**

9. **跟踪 `#52177`（DeepEP v2 的 DBO 支持）是否补上 `record_stream`** —— 它引入了与 `#60735` 相同的 comm stream 切换，官方已提示可能需要同样处理。
10. **跟踪 LMCache 上游是否合入 arXiv 2610.21500 的四项改动**（按 PCP rank 分片 MLA KV 存储、按 rank 分区远端层使 GET 带宽 6.3× 到 10.76 GB/s、异步加载、三个客户端 bug），以及论文给出的 `B* = sR` 判据是否被其他团队复现。**该文为预印本，未经同行评审。**
11. **跟踪 SGLang `#44666` 的 B300 PD 测试恢复情况**（注册关闭至 2026-11-01，`#44975` 为其恢复 PR），以及 `#44412` 所依赖的 DSA per-layer id 注册何时落地。

---

## 8. 社区与视频资料

**本期可确认的事实：**

- **vLLM 官方博客：窗口内无新文。** 最新仍为 2026-10-16 的三篇（`2026-10-16-kimi-k3-megakernel`、`2026-10-16-kimi-k3-mi355x`、`2026-10-16-reflection-beam`），其中前两篇已分别收录于 2026-10-18 与 2026-10-17 档案。**本期不重复收录。**
- **Hacker News（Algolia 公开 API，HTTP 200）：** 窗口内 `vllm` 关键词 28 条命中，但**与推理引擎实现直接相关的为零**；`sglang`、`speculative decoding`、`inference optimization` **均 0 命中**。`KV cache` 有 2 条，均为低分、零评论：
  - **《When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill》**（2026-10-18T14:23:22Z，2 points，0 评论，指向 arXiv 2610.21500）。**已直接核验 arXiv 摘要页**，见下。
  - 《A Token-by-Token Guide to Prompt Caching》（2026-10-18T18:35:20Z，2 points，0 评论，指向 `sithankanna.com` 个人博客）。**内容未核验**，仅记录线索。
- **官方视频：未核验到可确认发布日期与内容的条目。** 本期不以推测替代。

**二手线索（标注为未核验）**

- **arXiv 2610.21500《When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill》**（2026-10-15 提交，预印本，**未经同行评审**）。摘要页可直接核验的内容：
  - 场景：**GLM-5.3（FP8 MoE + MLA）在 8×B300 节点上跑 prefill context parallelism（PCP8）**的 PD 部署，远端 KV 层为 **LMCache + Valkey**。
  - 现象：开启远端层后 **median TTFT 不变，但 16k token 的 p99 从 3.96 s 升到 17.3 s**；在隔离引擎上重放生产请求流，**一次远端命中耗时 0.88 s，而重算 prompt 只需 0.31 s**；同步加载把命中密集的流变成不稳定队列，**p99 83.32 s 而 GPU 只有 7% 忙**。
  - 成本模型（两个可迁移的参数）：**命中只有在层交付带宽超过 `B* = sR` 时才划算（该模型与 GPU 上为 2.53 GB/s）**；**每次命中有约 `T0 ≈ 0.33 s` 的固定引擎开销，使得短于约 17k token 的 prompt 在任何带宽下都无法胜过重算**。
  - 四项改动：(i) 把 MLA KV 存储**按 PCP rank 分片**，并证明现有的「跨 rank 取最小」查询返回最长缓存前缀、给出**永不使集合通信失步的加载协议**；(ii) 远端层**按 rank 分区**（每 rank 一个服务实例、无协调），**GET 带宽提升 6.3× 到 10.76 GB/s**；(iii) 为分片存储加**异步加载**；(iv) 修三个导致无界停顿的客户端 bug。
  - 结果：命中密集流上（**98.5% 的 prompt token 由远端层服务**）**p99 从 83.32 s 降到 0.67 s**；分片存储使 **prefill 吞吐 +14%**，配合一处上游 kernel 修复达 **+34%**。所有改动已提交给 LMCache 上游。
  - **证据等级：** 一手（论文摘要页已直接核验）；**性能数字为作者自测，未经第三方复现，且论文未经同行评审**。HN 上仅 2 points、0 评论，**不构成社区共识**。

**采集限制说明：** Reddit 未采集；YouTube 等视频平台未采集（无法在不引入推测的前提下确认发布日期与内容）；X/Twitter 未采集。以上缺口不影响第 1–5 节的官方一手结论，仅影响「社区声量」这一维度。
