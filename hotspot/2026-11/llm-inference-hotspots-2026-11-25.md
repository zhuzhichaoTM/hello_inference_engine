# 大模型推理优化每日热点档案

> 档案日期：2026-11-25（本自动化于该日执行）
> 关注窗口：2026-11-24（周二）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，四家单页取全无截断）+ releases API + 52 个关键 PR 正文（vLLM 24/SGLang 20 经官方 issues API 抓取 markdown；TRT-LLM 8/llama.cpp 16 因配额 60/h 耗尽改 GitHub PR 页面 HTML `comment-body` div 深度配对提取，官方一手）+ vLLM 官方博客全文（HTTP 200，41K 字符）+ HN Algolia（HTTPS，`numericFilters` 按 2026-11-24 epoch 窗口 1795478400–1795564800）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 96 / SGLang 73 / TRT-LLM 41 / llama.cpp 43 合计 **253** 条（2026-11-24 为周二，工作日正常量）；**窗口内无新正式版本、无新 prerelease**（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、llama.cpp 正式版 v1.0.0 11-13、TRT-LLM v1.3.0 11-20 与滚动 RC v1.4.0rc4 11-18 均落窗口前不变；llama.cpp 仅推进 nightly，releases 前 8 条 b12676–b12686 落 11-24）。本期头号事件为 **vLLM 官方博客《Skip the DRAM Tier: MoRI UMBP Runs the Agentic KV Cache on NVMe》（2026-11-23 发布，经 HN 窗口 1 分线索回溯核验全文）**——同容量 1.8 TB NVMe KV 池在 MI355X/Kimi-K3 上吞吐与 DRAM 池相差 ±1%（48/70/96 并发），且无 DRAM 层（仅约 80 GB 作写缓冲与读暂存），成本从「按持有 TB 计费」变为「按写入 TB 计费」（约 5.9 万美元可存一天 KV，对比 7.6 万美元 DRAM 池仅存 31 分钟）。据此立累计结论第 25 条。次主线为 **服务安全加固簇延续 + 静默正确性/路由精度铁证**：SGLang `#44415`（GLM-5.3-Flash on MI355X，288 专家修正 bias 被存成 bf16 致 59% token top-8 改变）、vLLM `#66052`（DSA ShardQ 复用 pinned host 缓冲异步竞态，MI355X 10 次中 9 次 GPU hang）。HN 引擎官方更新类故事无（引擎相关帖均二手线索/未核验）。

---

## 1. 版本状态与本期头号事件：无新版本 + vLLM 官方博客「跳过 DRAM 层、KV 池直挂 NVMe」+ 服务安全加固簇 + 路由精度静默错误

**提交计数（2026-11-24 UTC，均 HTTP 200；周二）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 96 | Triton per-token-head KV 向量化 +33~48%(#69170) + renderer 默认 4 workers(#69041) + DSA ShardQ hang(#66052) + 安全加固三连(#69098/#61430/#68934) + GDN checkpoint 状态 #69401 + Qwen4Exp SM100 skinny GEMM #69529 |
| SGLang | 73 | diffusion API 键(#52929) + MoE 修正 bias bf16 静默(#44415) + GDN deterministic batch-invariance(#44194) + mem_cache 统一内存 decode 清零(#52904) + EGM Mooncake 池(#51416) + linker host-sync 簇(#49462~#49468) + AsyncPP 1/9(#52196) |
| TensorRT-LLM | 41 | logits_bias×投机(#22581) + KV 预算报告(#22795) + cluster-scope 屏障两则(#22712/#21850) + NVFP4 layout 错配(#22666) + PrimTS 预热编译(#22274) + Qwen3.8-Flash-Next sm_107(#22552) |
| llama.cpp | 43 | MMVF int32 溢出(#32965) + Metal top-k scratch 8GiB→2MiB(#32942) + NVFP4 dequant launch shape(#32875) + prompt cache 漏查(#28992) + Vulkan/Metal/OpenCL 多后端 + Inkling 2 DSpark(#32218) |
| **合计** | **253** | 周二工作日正常量；与上一期连续，无空档 |

### 1.1 头号事件：vLLM 官方博客《Skip the DRAM Tier》（MoRI UMBP，KV 池直挂 NVMe，官方一手）

- **来源核验**：`https://vllm.ai/blog/2026-11-23-skip-the-dram-tier`，标题日期 November 23, 2026，39 min read，标签 `performance / rocm / amd / prefix caching`；经 HN Algolia 窗口内 1 分线索「Skip the DRAM Tier: MoRI UMBP Runs the Agentic KV Cache on NVMe」回溯抓取，HTTP 200、正文约 41K 字符全文核验。**注意**：博客发布日期 2026-11-23 落在本期关注窗口（11-24）前一天，属「窗口内社区线索回溯核验到的官方一手资料」，此前 11-24 档案（覆盖 11-23）未收录，本期补录。
- **核心结论（事实，官方实测）**：在 AMD MI355X + Kimi-K3 上，用 MoRI 开源库 UMBP（unified memory and bandwidth pool）把 vLLM 的 KV 池直接放在 NVMe 上，**同容量 1.8 TB 与 1.8 TB DRAM 池吞吐相差 ±1%**——48 并发 DRAM 12,240 → NVMe 12,275 tok/s/GPU（+0.3%）、70 并发 13,380 → 13,492（+0.8%）、96 并发 13,313 → 13,284（−0.2%）；KV cache 命中率两组分别为 96.62%/96.67%、95.57%/95.65%、94.50%/94.48%。更长 prompt 的第二组负载上 NVMe 也在同容量下与 DRAM 相差 1% 以内。NVMe 池**没有 DRAM 层**：全部 1.8 TB 在盘上，DRAM 仅 64 GB 写缓冲 + 16 GiB 读暂存（约 80 GB，占池 <5%）；70 并发下无写缓冲亦零代价。
- **成本模型（事实，官方按 2026-11-13 中位报价测算）**：服务器 DDR5 约 $42/GB（≈$42,000/TB 持有），企业级 SSD 约 $666/TB（按每日一次全盘写入、5 年 1,825 次算，每写入 TB 约 $0.36）。Kimi-K3 每秒写入 2.3–4.6 TB/小时、96 并发 3.5 TB/小时（83 TB/天）；**按写入计费」的 NVMe：可存一天的 83 TB 池与 1.8 TB 池五年成本均约 $5.9 万（后者按磨损更换）；1.8 TB DRAM 池约 $7.6 万仅存 31 分钟，存一天的 DRAM 池约 $350 万**。吞吐由命中率决定：42 次 Kimi-K3 跑中命中率单独解释 96% 吞吐方差（R²=0.96），每多 1% prompt token 命中约值 1,280 tok/s/GPU。
- **三条实操教训（事实，官方）**：① **小 segment 文件**——UMBP 默认 4 GiB 文件淘汰时连带丢弃仍活跃 KV，改为 256 MiB 使命中率 94.45%→95.65%、TTFT p90 63.0s→21.3s、吞吐 11,672→13,492 tok/s/GPU（+15.6%，从落后 DRAM 12.8% 变为领先 0.8%）；② **touch（向池报告 GPU 已命中的前缀）**——避免池误淘汰 GPU 自身服务的热前缀，DRAM+NVMe 池吞吐 +2.1%/+3.9%、TTFT p99 从 70–77s 降至 35–43s（MoRI #779、charxwu/vllm-upstream#19）；③ **混合模型更频繁做 checkpoint**。
- **限制（官方明示，观点/边界）**：重读场景未对比（同容量实验池最多服务 16% prompt token，DeepSeek-V4-Pro 超 64 会话时服务 64–86%、DRAM 池读峰值近 27 GB/s，本文对比止于 64）；轻负载无队列隐藏时 NVMe 加载使 128K token 前缀 TTFT +约 0.4s；多数结果为单次运行；最大池 4.6 TB（跨两节点），「存一天」为算术外推；所有实验用 UMBP + MI355X，DeepSeek-V4.1-Flash 未跑 NVMe；部分改动仍在 upstream 途中。**Kimi-K3/DSv4-Pro 每 token KV 很小（约 14 KB / 5.5 KB，FP8），dense 无 MLA 模型需要成倍盘带宽**。
- **落地（事实）**：`UMBPStoreConnector` 已在 vLLM main（10 连 PR，起于 #61789，11-13 收尾）；prefill 端把它与 `MoRIIOConnector` 并列加入 KV 传输配置，decode 端只保留 MoRIIO consumer；关键 flag 含 `--prefix-cache-retention-interval 7168`、`enable_partial_hash_hits`。**研判（观点）**：这是继 KV Offload/分层之后把 KV 容量层的「最后一层」从 DRAM 下沉到 NVMe 的完整工程+成本论证；对本追踪主线而言，它把「KV 容量自检/命中率」（第 13 条）从正确性话题扩展到**存储层经济学**——值得单独列为第 25 条。

### 1.2 次主线一：服务安全加固簇延续（第 24 条，官方一手）

- **SGLang `#52929` [Feat] diffusion server 要求 API key（官方一手，安全加固）**：diffusion server 此前无 API key，控制路由（`/update_weights_from_disk`、`/v1/set_lora`、显存释放、profiling）对任何可达端口者开放；本 PR 移植 LLM server 的 `--api-key`/`--admin-api-key`（admin key 回退 `SGLANG_ADMIN_API_KEY`），25 条路由在 `auth_levels.json` 声明级别：权重/LoRA/显存/profile 为 `admin_force`（无 admin key 返回 403），`/liveness` 与 `/health`/`/ready`/`/metrics` 同为免键（K8s 存活探针所需）。CPU host 上 diffusion `test_http_server_auth`、`test_diffusion_metrics` 等通过。**研判（观点）**：与 11-15 档案 `#49896`（LLM admin 端点需 admin key）同源，把安全边界从 LLM server 扩到 diffusion server，延续第 24 条。
- **vLLM `#69098` [Security] Bound media preprocessing per prompt（官方一手，安全加固，正文 25K）**：新增 `VLLM_MAX_MEDIA_PREPROCESSING_BYTES`（默认 715,827,880 B ≈ 683 MiB），独立于下载/解码预算；超限请求返回 HTTP 400。整合 `#68375`(FunAudioChat)、`#68767`(Nano Nemotron VL)、`#68898`(DeepSeek-VL2/OCR) 三 PR（均关闭为 superseded），并覆盖 Molmo2、Qwen3-VL、LLaVA-OneVision、Gemma 4、EmbeddingGemma2 视频路径；Qwen3-VL/LLaVA-OneVision 默认每 prompt 至多 32 视频。接受的媒体输出不变，仅改变超额请求的准入。
- **vLLM `#61430` [Security] Contain Sentence Transformers module files（官方一手，安全加固）**：`modules.json` 可把 Pooling/Dense 加载器指向模型目录外文件（含符号链接）；本 PR 对每条模块路径做包含性校验（含解析 symlink 目标），失败即 fail-closed，覆盖 `/v1/embeddings`/`/invocations`/`/pooling` 与离线 pooling。macOS CPU 29 测试通过。**注意**：11-21 Buildkite 重试命中与 upstream main 相同的 AMD Unlimited-OCR-2 抢占断言（非本 PR 引入）。
- **SGLang `#50709` [Fix] io_struct: Reject list RIDs for single requests（官方一手，fix）**：单请求传 list `rid` 导致 `TypeError: unhashable type: 'list'` → HTTP 500；改为在 `GenerateReqInput`/`EmbeddingReqInput` 归一化时拒绝非字符串 `rid`，HTTP 层返回 400；批量 `rid` list 与 `n>1` 展开不变。

### 1.3 次主线二：路由精度与静默正确性铁证（第 4/6/7/9/19 条）

- **SGLang `#44415` [Fix][AMD] GLM-5.3-Flash 保持 MoE correction bias 为 fp32（官方一手，fix，**本期最强静默错误铁证**，强化第 4/7 条）**：MI355X + aiter + FP8/Quark 权重下，`MoEGate` 把每层 288 个专家修正 bias 从 checkpoint 的 fp32 存成 bf16（该 bias 在约 0.4 宽的区间内，bf16 步长 0.03–0.06），**59% 真实 token 的 top-8 专家发生改变**，而 CUDA 路径保持 fp32 不受影响。修法按 `model_type=="glm5_next_text"` 保持 fp32（`#44797` 起 aiter 已用 fp32 logits 路由 fp32 bias），`topk.py` 与 CUDA 不动，fp32 路由 6 token 时多 2.7 µs/次。评测基于 upstream main `0cf683ea8d` + `#41615` + `#38764`（GLM-5.3-Flash 跑 DSA k-pool 所需），1319 GSM8K + 150k wikitext token。**研判（观点）**：这是「量化权重下路由元数据 dtype 降级→专家选择静默改变」的教科书案例——不崩溃、精度只以路由错位形式体现，AMD 独有，须作为异构后端精度回归必测项。
- **vLLM `#66052` [Bugfix][DSA] ShardQ 每步用新 host 内存暂存 indexer indices（官方一手，fix，强化第 4/19 条并发竞态）**：DSA indexer ShardQ 路径（`#63956`）每步的 gather indices 与重编号 query range 从复用的 pinned host buffer 上传，异步调度可在该步拷贝前覆写。DeepSeek-V4.1-Flash、MI355X TP4、FP8 indexer、1M-token prompt：main builder **10 次中 9 次 GPU hang**（4 节点，无报错），`--no-async-scheduling` 8/8 通过，改用每步新 host 内存 16/16 通过；CUDA 无硬件未复现但路径共享。
- **SGLang `#44194` [Fix] deterministic GDN varlen prefill 固定输出 tile（官方一手，fix，强化 batch-invariance 主线）**：`chunk_fwd_o` 中 `BT=min(chunk_size, max(16, next_pow2(T)))`，`T` 是整批 token 数，同一短 prompt 单独 prefill 得 16-tile、与别的请求同批得 32/64-tile，logprob 末位改变。MI350X Qwen3.8-27B：同一 prompt 重复 1–50 次，token id 总相同但 TP2 下 1,275 条响应仅 33–48 条 logprob 匹配；修法在 batch-invariant 模式下 varlen 输入固定用 64 chunk tile，定长批保持小 tile。**注意**：同样 Triton kernel 在 NVIDIA deterministic 模式下也运行，非 ROCm 独有。
- **llama.cpp `#32965` cuda: fix int overflow of MMVF matrix offsets（官方一手，fix，强化第 9 条整型溢出）**：`mul_mat_vec_f` 把 block index×stride 算成 `int*int`，超过 2^31 值回绕致越界非法访存——F8_E4M3 专家张量（DeepSeek-V3 256×2048×7168≈3.8B 值）中 147–255 号专家起点越过 2^31；单矩阵 1 token 也触发（Nemotron-4-340B 256000×18432≈4.7B 值）。改 64 位；RTX 5090 临时用例从 `illegal memory access` 变为 pass，`test-backend-ops -b CUDA0 -o "MUL_MAT.*"` 5628/5628。源自 `#32533` 评审。
- **vLLM `#62268` [Bugfix] Guard Conch W4A16 preconditions and pad non-tile-multiple outputs（官方一手，fix，强化第 6 条量化正确性）**：`--linear-backend conch` 在 `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` 首次 forward 即 GPU 内存访问错误；48 层 `in_proj_ba`（N=96）中 Conch W4A16 kernel 以 64 列 tile 加载权重与 scale 且无边界 mask，第二个 tile 越界 128B 权重 + 64B scale/行。修法在 wrapper 层拒绝无法寻址的 K、把 N pad 到 tile 宽度；gfx950(MI355X) 上 48 层全部可跑，decode TPOT 相对默认 Triton **−31.7%/−29.4%/−22.5%**（c=1/8/32，8k-in/1k-out）；Triton 仍默认、Conch 仍 opt-in。
- **vLLM `#68934` [Core][KV Cache] prefix_match_unit 显式值失败即启动拒绝（官方一手，fix，第 13 条 KV 容量/对齐）**：显式 `--prefix-match-unit` 若因 manager 需块对齐（如 `(768,6144,768 SWA)` 配 128）而降级到整块命中，此前只在运行时打易被忽略的 warning，现启动时抛 `ValueError` 点名违规 manager；未设置时保持 GCD 默认行为。8 passed。

### 1.4 其他结构性观察

- **KV 层/容量与 PD 资源生命周期**：SGLang `#52904` 修 `--enable-unified-memory` 下 full/SWA 共享 arena 时把请求 decode 预算按名义 `max_total_num_tokens` 裁剪，导致 150K（名义）~520K（实际）间的 240K prompt 被准入后**返回 0 token 且无报错**（4×RTX PRO 6000 Beam 1.0 NVFP4 实测）；vLLM `#67185` HiSparse 在分配下一 prefill chunk 前先回收驻留页以解 streaming-resume 停顿（650 passed/2 xfailed，但**保留的 streaming-resume 回归在 upstream 上仍失败**、GPU 端到端/精度/性能待验）；SGLang `#47618` PD PP prefill 由 PP0 汇总各 stage 状态后统一释放（`#45085` rebase，修「某 stage Failed 即全放、仍在 Transferring 的 stage 释放其 worker 仍读的元数据槽」）。
- **host-sync 移出热路径延续**：SGLang linker 簇 `#49462`/`#49464`/`#49465`/`#49466`/`#49468`（Mooncake 分层加载一次 numpy 建 range 元数据、fence 前建好所有 layer group 的 range reads、submit 间隙完成已落地 group、向量化 restorable-prefix mask、DSpark draft KV 与最后传输层同批）——`#49468` 记录单次 range-get 在 load thread 提交期间 2.15 ms p50、提交结束后 1.40 ms（慢 1.45×/p90 2.5×），nsys 中 linker 引入的 GPU idle 81% 落在 group0 submit 到其 reads 落地之间；`#49464` 的加速数字为**离线日志估计、非实测 A/B**。
- **编译态特化/热路径停顿（第 20/22 条）**：TRT-LLM `#22712` 把 GLM-5.2 TEP4 融合 top-k routing 8 个 `mbarrier.try_wait`（默认 CTA-scope acquire 与 peer 的 cluster-scope release 不同步→读陈旧 `all_counts`/`base`→专家总数或置换索引错）改为 relaxed cluster-scope try-wait + `fence.sync.restrict(shared::cluster)`；`#21850` 给 scheduler slot 在 CTA 间交接加 cluster-scope acquire/release（Rubin FC1/FC2，**作者报未在 sm_107 运行、QA 判 needs follow-up**）；`#22274` PrimTS sparse MLA decoder warmup 预编译（21 文件 +1532/−1936，coderabbit 指静默 warmup fallback、vendor lock 未合并上游、多条 pipeline 失败、当前 head CI 不可用）。

---

## 2. vLLM：2026-11-24 提交（96 条）—— KV 向量化 +33~48%(#69170) + renderer 默认 4 workers(#69041) + GDN checkpoint #69401 + Qwen4Exp skinny GEMM #69529 + 安全三连 + DSA hang(#66052)

### 2.1 `#69170` [Perf][Attention] Triton attention 向量化 per-token-head KV tile 载入（官方一手，perf，强化第 14/16 条）

从启动时的实际 stride/offset 推算编译器可依赖的最宽对齐，按该字节数预取载入（`head_dim=128` 时 K 一次读 8 字节、V 读 4 字节，原来各 1 字节）；仅改生成 kernel 代码，不改 KV cache 布局/页大小/内存占用。`vllm bench serve` 总吞吐（每 build 两次均值，main→本 PR）：Qwen3-14B int8 A100 **3,536→4,701（+33%）@8k prefill**、2,970→3,995（+34%）@256 decode；int4 +16%/+18%；Qwen3-8B int8 +36%/+41%；int4 +10%/+11%；Qwen3-0.6B fp8 RTX 5060 Ti **+48%/+48%**。精度：GSM8K 16 组 main/PR 中 14 组差在 stderr 内、最大差 −0.0228（重复后 +0.0083，同 build 两次差至 0.0159）；kernel 输出逐位一致。与 `#62113`/`#56164` 的布局改动独立兼容。

### 2.2 `#69041` [Frontend][Perf] 异步 renderer 的 `--renderer-num-workers` 默认改为 4（官方一手，perf，默认化，第 16 条）

长 prompt（~100k token）时单线程 tokenize 成为瓶颈、引擎停顿。B200 上 Qwen3.8-Flash-NVFP4 DP8 + EP、随机 prefix 99,201 token、c=1024：吞吐 **26,816→46,975 tok/s（×1.75）**、中位 TTFT **21.4 s→0.47 s**、P99 TTFT 32.8→11.1 s（16 workers 时 44,193/6.1 s 更优 P99 但吞吐略低）；代价是中位 TPOT 14.3→19.8 ms（并发上升所致）。仅对非 offline serving 生效，offline 保持 1。**研判（观点）**：长 prefix agentic 负载下前端 tokenize 是隐形瓶颈，属「默认性能化」正面案例，升级须同时看 TPOT 与 TTFT 权衡。

### 2.3 `#69401` [Perf][GDN] 只写被使用的 FlashInfer prefill checkpoint 状态（官方一手，perf）

FlashInfer GDN prefill（`#60050`）每 N token 吐一个 state，`--prefix-match-unit 128` 下混合 prefill 步 N=128–256、kernel 写约 60–130 个 state 而只用约 3 个。本 PR 传 FlashInfer 0.7.4 的 `checkpoint_state_indices`。Qwen3.8-Flash-Next-FP8、1×B300 TP1：GDN prefill kernel 每步时间 **−34% 以上**（15.80→10.48 ms 缓存前缀加载场景；服务 13.78/16.10→9.04/7.75 ms），输出逐位一致、GSM8K 与 main 相同；每步约省 300 ms prefill 中的 2%，端到端吞吐 ±10% run-to-run 噪声内（41.2k vs 41.9k tok/s）。

### 2.4 `#69529` [Perf][Qwen4Exp] SM100 TP=2/TP=8 skinny GEMM plans（官方一手，perf）

B200 上 Qwen3.8-Flash-Next TP2/TEP2 或 TP8/TEP8 的所有 BF16 decode projection 此前走 cuBLAS；本 PR 增 B200 实测 cuteDSL skinny-GEMM plans（6 个 TP2 + 6 个 TP8 局部形状，纯表、仅 SM100）。TP2 kernel vs cuBLAS：GDN B/A 2.44–2.78×、shared-expert gate/up 1.54–2.00×（M=1–8）；端到端 TEP2 median TPOT **−10.4%/−7.4%/−6.6%/−3.0%**（c=1/2/4/8），GSM8K 0.968/0.967（PR）vs 0.968/0.971（main）。仅当 sweep 中胜 cuBLAS ≥10% 才加表项、M=16 不加。

### 2.5 其他要点（官方一手，perf/feat/fix）

- `#62004` [Kernel] ReplaySSM GDN 标准 decode：复用近期 SSM 输入 ring buffer + 周期 checkpoint 重建状态，免每步把 recurrent state 写回 HBM（Dao AI Lab × NVIDIA，Mamba2 标准 decode `#48018` 之后第二子 PR）。
- `#66749` [Kernel][MoE] FlashInfer CuTe-DSL **非量化** MoE 后端（SM90 bf16/fp16），并成为 Hopper 默认非量化 MoE 后端。确定性两阶段 finalize fp32 固定序（逐位可复现）比 fused finalize 慢约 1.6%；`#41306` 的 8×H100 用例上全面优于 Triton（Mixtral-8x7B TP8 −12% TPOT/−8% TTFT/+13% 吞吐@ c=1）。
- `#68321` SM90 HYV4/V4-Flash 8/16/32-head 原生 BF16 FlashMLA（不再 pad 到 64）：4×H20-3e TP4、8192/128 c=16 吞吐 **141.77→193.52 tok/s（+36.5%）**、TTFT 5,276→3,635 ms（−31.1%）；ABBA 两轮、128/128 基本不变。
- `#65474` ROCm Qwen4Exp QSA sparse GQA + indexer tuning（gfx942/gfx950）：sparse GQA geomean prefill 1.57×/decode 1.65×（gfx942）、1.24×/1.45×（gfx950）；indexer prefill 1.3–6.5×@8K–60K；GSM8K 不变、RULER NIAH 64K/128K = 1.000。
- `#66342` [Feature][Quantization] 加载时按 rename mapper 重写 compressed-tensors `re:` 目标（如 `re:^layers\.(0|1)\.ffn\.shared_experts\.w2` → `re:^model\.layers\.(0|1)\.ffn\.shared_experts\.down_proj`），checkpoint 不改、旧 vLLM 仍见具体名。
- `#63002` [ROCm][UMBP] DSv4-Pro 支持 UMBP standalone（仅压缩 KV 入池、SWA 靠 bounded replay 重建）：8×MI355X GSM8K strict-match 0.9515→0.9538、AgentX 吞吐 +14.4%；`#69752`/`#69749` 为 DSv4.1-Flash/Kimi-K3.1-Pro MI355X AgentX 配方对齐 InferenceX #5195/#5196（第三方 run/描述级）。
- `#69564` 移除 LM Format Enforcer structured outputs backend（v0.33.0 弃用、计划 v0.35 移除，当前 0.11.3 已因 tokenizer 模块缺失不工作）；`#66720` 修 n>1 prefill 统计误报 sibling cache hits；`#56239` 避免在解析 AllReduce+RMSNorm 融合配置时初始化 CUDA；`#e2617756` Rust frontend 对无法服务的多模态模型启动即失败；多条 `[CI]`/`[Governance]` 变更。

---

## 3. SGLang：2026-11-24 提交（73 条）—— diffusion API 键(#52929) + MoE bias 静默(#44415) + GDN deterministic(#44194) + mem_cache(#52904) + EGM Mooncake(#51416) + linker host-sync 簇 + AsyncPP 1/9

### 3.1 `#44415` / `#44194` / `#52904` / `#50709` / `#52929` 静默正确性与安全 —— 见第 1.2/1.3 节

### 3.2 `#51416` [Feat] mooncake: opt-in EGM-backed Mooncake Store pool over NVLink（官方一手，feat，架构，第 11/17 条）

GB200/GB300 NVL72 上 GPU 可经 EGM（Extended GPU Memory，`cuMemCreate` + fabric handle）读写同 NVLink 域其他 tray 的 host DRAM；Mooncake 0.3.14 支持 `enable_egm_store_pool`。本 PR 加 opt-in `enable_egm_store_pool`/`egm_numa_nodes`，命中时改走 `setup(config: dict)` 传 EGM 键；关闭时行为不变。GB300 两 tray 实测：一 tray 的 HiCache L3 命中由另一 tray 的 EGM 池服务（7,351 prompt token 中 7,296 来自存储），跨 tray 消费逐字节一致。**性能未 benchmark**（作者仅给单次粗读速率）。**研判（观点）**：继 SGLang Mooncake TENT/#51416 后，KV 三层存储的数据面在 NVLink 域内进一步摆脱 NIC，与 vLLM 博客的「KV 层下沉/跨节点池化」同向。

### 3.3 linker host-sync 簇与 KV 分配批处理（官方一手，perf）

- `#49468`/`#49464`/`#49465`/`#49466`/`#49462` 见 1.4；`#49468` 把 group0 读尾延迟与 GPU idle 归因到 GIL 争用，fence 前预建所有 range reads；`#49464` 速度数字为离线估计。
- `#47578` [Perf] kv-shard `PageInterleavePoolAllocator` 批式 pop/free：`alloc_extend` 改为每 owner class 一次 slice pop + numpy host plan + 一次 H2D/gather/mul-add，设备操作数不再随 batch 增长；`free()` 用 `scatter_add_` + 一次 `.tolist()` 把每 class 一次 host sync 降为一次；`free_segment()` 走 stride 快路径免 `torch.unique`。分配结果与现实现逐页同序一致。
- `#47576` [Perf] cp zigzag CP metadata 用一次 pinned non-blocking 上传替代 12 次 pageable H2D（避免 overlap/PP 下每次 prefill-CP batch 的 host 停顿），Qwen3-30B-A3B-FP8 4×B300 GSM8K 0.9543→0.9536（一题差、噪声内）。

### 3.4 AMD/NVIDIA 后端与异步 PP（官方一手，perf/feat/fix）

- `#49566` [Perf][AMD] gfx1250 fused RoPE + KV write 在 T≥256 时 tile 化（16 token×8 query head）：gpt-oss-120b TP1 T=16384 kernel **586→135 µs**，逐位一致；服务 A-B-A 数字见正文。
- `#50685` [Perf][AMD] Muse Spark FP8 joint router/pooler GEMM 用 Hyperloom kernelforge 重写（更宽 pooler tile、独立 router tile、XCD-aware program-id remap），MI355X 上约 2 TB/s，partials 逐位一致。
- `#52643` [Feat][NVIDIA] FlashInfer KDA prefill 放宽到 SM107（VR200）：Kimi-K3.1-Pro TP8/DCP8/EP2 冷 prefill TTFT −7.8~−10.2%（1K–32K），AgentX 4 会话总吞吐 +3%、首 token 早 4%；prompt logprob 平均差 0.26（mean NLL 3.108 vs 3.110），GSM8K 97.50% vs main 97.50/97.19%。
- `#52644` [Fix][AMD] gfx1250 上 FP8/BF16 MoE 回退 Triton runner：aiter 只有 MXFP4 interleave 的 FlyDSL grouped GEMM，block-FP8 模型（Qwen3.5-397B/35B-FP8）启动即崩、BF16 MTP draft 在 CUDA graph capture 崩；`auto` 现在选 Triton，免手动 `--moe-runner-backend triton`。
- `#48154` gfx1250 MLA 复用 gfx950 的 aiter RoPE/RMSNorm/store_mla 路径；`#52196` [AsyncPP 1/9] PP 输出消息 device logprob 行先拷回 host 再 pickle（否则 device tensor 在 CPU group 元数据里被跨设备 unpickle）；`#52570` sampler params 复用、`#50622` QSA extend row map 免 device 读回、`#44194` deterministic 等见对应小节。

---

## 4. TensorRT-LLM：2026-11-24 提交（41 条）—— logits_bias×投机(#22581) + KV 预算报告(#22795) + cluster 屏障(#22712/#21850) + NVFP4 layout(#22666) + PrimTS 预热(#22274)

### 4.1 `#22581` [feat] Support logits_bias with speculative decoding（官方一手，feat）

支持 `logits_bias`/`bad_words` 与投机解码并用（greedy + rejection sampling）；logit 修改不作用于 drafter，接受率（AL）可能因此下降（官方明示预期）。Qwen3-27B + 新单测；14 文件 +1551/−118，新增 `speculative/logit_edits.py`。**研判（观点）**：属功能补齐，但「logit 编辑只作用于 target 不作用于 drafter」是接受率口径变化，须在 A/B 中把 AL 列为验收指标（呼应 11-23 档案 PD 验收含接受率的结论）。

### 4.2 `#22795` [feat] Report the profiled KV cache sizing budget（官方一手，feat，可观测性）

full-mode 显存报告现在展示各 rank profiling 测量如何转成 KV cache 字节预算（PyTorch 峰值、估算非 PyTorch、临时 KV pool 抵扣、free-memory fraction 保留、用户上限扣减、最终容量上限），复用启动期 sizing 值、不需新采样。1×B200 free fraction=0.2：Qwen2-0.5B 峰值 3.413 GiB→最终 KV 预算 35.014 GiB、Phi-4-mini 10.934→33.702、Llama-3.1-8B 19.074→32.074；39 单测通过。**研判（观点）**：与 vLLM 的 KV 容量自检（第 13 条）同向，把 KV 预算从「黑盒」变为可审计数字。

### 4.3 `#22666` / `#22712` / `#21850` / `#22274` 布局与同步正确性（官方一手，fix/perf，验证未闭合）

- `#22666` [fix] sparse-offload NVFP4 layout 错配：`#22080` 的 NVFP4 DSA layout `auto` 是每 worker 本地选择，驻留 worker 选 inline-scale `fp8rope`、offload worker 用 separate-scale `residual` 时 `PeerRegistrar.get_pool_mapping` 无法配对 block-scale view（同 GPU 同软件亦可发生）。修法在测试的 child-process 环境显式 `TRTLLM_NVFP4_MLA_KV_CACHE_LAYOUT=residual`、移除 GB300 waiver；**产品默认选择未改**，不同 residual-quantization 设置的互操作性未解决。
- `#22712` / `#21850` 见 1.4（cluster-scope 屏障，`#21850` QA needs follow-up）。
- `#22274` [perf] PrimTS sparse MLA 在 decoder warmup 预编译并复用 decode views：coderabbit 指静默 warmup fallback（worker 错误仅 warning、首次 forward 仍可能编译）、prepared-launch cache 无运行时校验、vendor lock 分支未合并上游、多条 pipeline 失败/当前 head CI 不可用，按「官方一手但验证未闭合」处理。

### 4.4 其他要点（标题级/描述级）

`#22552` Qwen3.8-Flash-Next modeling_v2 服务 sm_107（NVFP4 tp2/tep2、dep2/dep8 跨节点；`layout()` 去 `layouts` 参数 + 模型类重命名属 API 变更、coderabbit 指测试 key 可能拼错导致静默跳过 cell、作者未在 sm_103 跑 dep8、CI 失败）；`#22645` skip unread grouped DSA TopK copies（省每层一次 GPU copy，38 CPU differential cases、GPU/E2E 未跑）；`#22777` fused mHC tactics 跨 TP rank 共享（无测试守卫）；`#22821` FP8 MLP 精度参考匹配选定 tactics（仅改单测）；`#22794` MXFP8 tactic tuning bounds debug 日志（标题级）；`#22057` one-model spec worker 词表并行温度采样（标题级）；`#21404`/`#22682` refactor（标题级）；多条 `[infra]` lock file/工作流加固。

---

## 5. llama.cpp：2026-11-24 提交（43 条）—— MMVF int32 溢出(#32965) + Metal top-k scratch 8GiB→2MiB(#32942) + NVFP4 dequant(#32875) + prompt cache 漏查(#28992) + 多后端

### 5.1 `#32965` int32 溢出 —— 见第 1.3 节

### 5.2 `#32942` metal: reserve the top-k scratch only for short rows（官方一手，perf/内存，第 13 条 KV/缓冲区容量）

bitonic top-k 仅在短行（≤128 列）更快，超过则 radix 路径更快且原地选择无需 scratch；但 buffer 类型始终按 `2*sizeof(int32)*元素数` 预留——524288×2048 分数张量即 **8 GiB**。改为在 128 列阈值处预留 bitonic 最坏情况（`2*sizeof(int32)*128*nrows`，不低于输出张量）：同一 524288×2048 稀疏注意力分数 k=64 仅 2 MiB、k=2048 仅 16 MiB。M5/M2 Ultra 微基准：16 列 bitonic 12.8×/11.2×、128 列 1.88×/2.7×、256 列 radix 1.79×/bitonic 1.14×(M2)、512–1024 列 radix 3.29–6.71×。**研判（观点）**：与第 13 条「KV/缓冲区容量自检」同族——预留按最坏情况无脑放大是隐形显存杀手，按运行时路径阈值收紧收益数量级。

### 5.3 `#32875` CUDA NVFP4 dequant launch shape（官方一手，perf，描述级微基准）

PR 收敛为仅改 launch shape（`convert.cu` 2 行，每 CTA 打包 8 块）。V100 sm_70、dequant 路径、nsys：`MUL_MAT type_a=nvfp4,type_b=f32,m=4096,n=512,k=14336` 的 dequant kernel 2,293→1,196 µs，op 级 1,436→343 µs；`test-backend-ops -b CUDA0 -p nvfp4` 通过。作者称 Volta 上仍成立（launch shape 获得约 95% op 级收益）。**注意**：形状特定微基准、非端到端。

### 5.4 `#28992` server: fix skipped prompt cache lookups（官方一手，fix，KV 前缀缓存可用性）

真实生产 30+ 天证据（b11206，`--parallel 1 --cache-ram 16384`）：两个 agent 会话共享约 12.4K-token 前缀、大小 97K(A)/17K(B) 交替轮次时，A 返回时 `f_sim=0.128>0.1`、`f_keep=0.73≥0.5` → `update_cache=false` → `prompt_load` 从不调用 → A 每轮重算约 87K（约 2 分钟）；B 亦从未被保存，两侧永久全量重算。单会话 100% 命中（`f_keep=1.0`）。临时规避 `--slot-prompt-similarity 0` 强制走 LRU 路径恢复存/取（有每次切换一次 RAM 存+取的代价），本 PR 用 `has_better()` 定向修复门控。**研判（观点）**：这是「缓存准入/淘汰门控在多会话交错下把两边都饿死」的真实案例，与服务端 prefix cache 命中率优化直接相关。

### 5.5 其他要点（官方一手，feat/fix，标题级/描述级）

- `#32218` Inkling 2 DSpark support；`#32721` Vulkan chunked GDN replay（基于 `#31669`，叠加 961d7ca，draft）；`#6fb42833` CUDA NVFP4 dequant pack 8 blocks/CTA（提交标题）。
- `#32897` CUDA cooperative softmax 在 nsm>256 时归并所有 partial（正文仅讨论 `#pragma unroll` 无法展开，性能/精度数字未在抓取片段呈现）；`#32691` WebGPU softmax kernel 加 `dst_rows` 越界检查；`#32182` SpacemiT FA null mask/ALiBi/softcap 修复（正文仅评审对话）；`#32747` OpenCL GDN QK offsets 修复（标题级）。
- `#30997` server 清 idle slot 后重试 prompt cache restore（正文仅自动 PR 模板提示）；`#32500` CUDA 融合 RMS_NORM+MUL+MROPE/IMROPE 并调 block size（正文仅一句 `@am17an 请 review`，记为标题级）；`#32142` ARMv8.0 无 dotprod 目标 Q4_0/Q8_0 repack（标题级）；`#30867`/`#32909` HTML 抓取正文为空，标题级。

---

## 6. 趋势研判与累计结论

累计结论继承第 1–24 条（见上一期档案第 6 节），**本期新增第 25 条**；本期铁证按既有条目归类并强化如下：

1. **【新增第 25 条】KV 池层可下沉到 NVMe 且同容量吞吐等价，成本模型从「按持有容量」转为「按写入量」**：vLLM 官方博客在 MI355X/Kimi-K3 上以 1.8 TB NVMe 池（无 DRAM 层）在 48/70/96 并发下与同容量 DRAM 池相差 ±1%，每 1% 命中约值 1,280 tok/s/GPU，命中率解释 96% 吞吐方差；按写入计费使「存一天的 KV」约 $5.9 万（DRAM 需约 $350 万）。成立条件：MLA/低 KV-per-token 模型（Kimi-K3 约 14 KB/token）、异步加载可藏延迟、segment 文件调小 + touch；停止条件/观察：重读场景（池服务 >16% token）、轻负载 TTFT（+0.4s/128K）、盘磨损（1.8 TB 池约 11 次写入/天、5.5 个月耗尽）。
2. **静默正确性（第 4/6/7/9/19 条）持续最高频，「路由元数据 dtype 降级」成新显眼子项**：SGLang `#44415`（bf16 bias 致 59% token top-8 改变，AMD 独有）、`#44194`（varlen tile 致 batch-invariance 破坏）、vLLM `#66052`（pinned host 复用竞态致 9/10 GPU hang）、`#62268`（Conch W4A16 N=96 越界）、llama.cpp `#32965`（MMVF int32 溢出越界）——覆盖路由/元数据/并发/布局/整型五类。
3. **服务安全加固（第 24 条）从 LLM server 扩到 diffusion server**：SGLang `#52929`（admin key 门槛）、vLLM `#69098`（媒体预处理 683 MiB 上限）、`#61430`（Sentence Transformers 路径包含性）、SGLang `#50709`（rid 类型校验 500→400）。
4. **KV 容量/命中率治理（第 13 条）扩到缓冲预留**：SGLang `#52904`（统一内存 decode 被名义容量裁成 0 token）、llama.cpp `#32942`（top-k scratch 8 GiB→2 MiB）、TRT `#22795`（KV 预算可审计）、vLLM `#68934`（prefix_match_unit 降级启动拒绝）。
5. **host-sync 移出调度器热路径（既有主线）延续**：SGLang linker 五连 `#49462`–`#49468`、`#47578`、`#47576`、`#50622`（QSA extend row map 免 device 读回，16k prefill 每轮省约 90 ms scheduler 停顿）——「pageable H2D/repeat_interleave 读回/逐 class 掩码」是三类主要同步源。
6. **默认性能化与代价（第 16 条）**：vLLM `#69041` renderer workers 1→4（长 prefix 吞吐 ×1.75、TTFT 21.4s→0.47s，但 TPOT 14.3→19.8 ms）；`#66749` FlashInfer CuTeDSL MoE 成 Hopper 默认（确定性路径慢 1.6%）；`#69564` 移除 LM Format Enforcer backend。升级须同时看吞吐、延迟分布与移除项。
7. **KV 存储层/跨节点池化（第 11/17 条）**：SGLang `#51416`（EGM-backed Mooncake 跨 tray NVLink 池）、vLLM `#63002`（DSv4-Pro UMBP standalone +14.4%）、`#69752`/`#69749` 配方——与本年第 25 条共同指向「KV 层从 GPU→DRAM→NVMe/跨节点」的连续谱。

---

## 7. 可执行清单（面向工程落地）

1. **KV 分层/容量规划（新增）**：agentic 长会话服务评估 vLLM 官方博客的 UMBP NVMe 池（`UMBPStoreConnector` 已在 main），先测写 TB/小时并据停留时长定容量；采用 256 MiB segment 文件 + touch；规避重读>16% token、轻负载 TTFT 敏感和 MLA 缺失的 dense 大模型。
2. **AMD/量化路由精度回归**：MI355X + GLM-5.3-Flash（FP8/Quark）升级 SGLang `#44415`（修正 bias 保持 fp32，否则 59% token top-8 改变）；deterministic 模式升级 `#44194`（GDN varlen tile）；ROCm 下 Qwen4Exp 评估 `#65474`。
3. **并发 hang / 整型溢出排查**：DSA ShardQ + 异步调度在 MI355X 上出现无报错 GPU hang 升级 vLLM `#66052`；大专家张量/大输出层（>2^31 元素）升级 llama.cpp `#32965`；Conch W4A16 在 N 非 64 倍数层升级 `#62268`（性能收益同时成立）。
4. **服务安全边界**：diffusion server 配置 `--api-key`/`--admin-api-key` 升级 SGLang `#52929`；多模态服务评估 vLLM `#69098`（默认 683 MiB）、`#61430`；单请求 rid 校验升级 `#50709`（500→400）。
5. **KV 缓存可用性/命中率**：统一内存服务升级 SGLang `#52904`（否则长 prompt 返回 0 token）；多 agent 会话交错场景评估 llama.cpp `#28992`（否则两侧永久重算）；HiSparse streaming-resume 评估 vLLM `#67185`（注意其保留回归在 upstream 仍失败、GPU 端到端待验）。
6. **吞吐/延迟优化**：Triton attention per-token-head KV dtype 服务评估 vLLM `#69170`（+33~48% 且逐位一致）；长 prefix 服务评估 `#69041`（×1.75，注意 TPOT 上升）；GDN prefill 评估 `#69401`（kernel −34%）；Qwen4Exp B200 TP2/TP8 评估 `#69529`（TPOT −10.4%@c1）；HYV4/V4-Flash SM90 评估 `#68321`（+36.5%）。
7. **显存/缓冲预留**：Metal 大分数张量 top-k 升级 llama.cpp `#32942`（8 GiB→2 MiB）；TRT KV 预算排障用 `#22795` 报告。
8. **Rubin/sm_107 与 cluster 同步谨慎采用**：TRT `#21850`（作者未在 sm_107 运行、QA needs follow-up）、`#22274`（CI 失败/当前 head 不可用）、`#22552`（API 变更/测试 key 可能拼错）、SGLang `#52643`（prompt logprob 平均差 0.26）均按官方一手但验证未闭合处理，先在目标硬件做精度/稳定性回归。

---

## 8. 来源、数据质量与限制

- **一手来源（官方仓库/API/文档）**：vLLM 96 / SGLang 73 / TRT-LLM 41 / llama.cpp 43 条提交均经 `api.github.com/repos/<repo>/commits?since=2026-11-24T00:00:00Z&until=2026-11-25T00:00:00Z` 取得（全部 HTTP 200、单页取全无截断）；releases API 确认窗口内无新正式版/无新 prerelease（llama.cpp releases 前 8 条 b12676–b12686 为 prerelease nightly）；52 个关键 PR 正文中，vLLM 24 + SGLang 20 经官方 issues API 抓取 markdown（配额 60→8，无降级为纯标题），TRT-LLM 8 + llama.cpp 16 因配额耗尽改 GitHub PR 页面 `comment-body` div 深度配对提取（官方一手，但 `#22785`/`#22794`/`#22057`/`#21404`/`#22682`/`#32182`/`#32500`/`#32747`/`#32142` 等仅抓到评审对话/reaction 片段，按标题级处理）。vLLM 官方博客全文 `https://vllm.ai/blog/2026-11-23-skip-the-dram-tier` HTTP 200、约 41K 字符核验。性能数字如无特别说明均来自 PR 正文/官方博客（作者实测）。
- **二手线索/未核验**：HN Algolia 窗口内 `vllm` 37 命中（多为「LLM」模糊匹配）、`sglang` 2、`inference engine` 4、`KV cache` 1、`TensorRT-LLM`/`llama.cpp`/`speculative decoding`/`paged attention` 均 0。引擎相关线索——1 分「Skip the DRAM Tier」（**已回溯核验为 vLLM 官方一手**）、1 分「A live peek inside your LLM inference engine (SGLang, vLLM) as it runs」(`gauravjain14.github.io/inference-observe/`)、4 分「Can agents write inference engines from scratch in 12 hours and beat vLLM?」(`github.com/humanityslastsystem/hls`)、1 分「FastPath Agentic Recipe: 50% more agents at equal P90 interactivity on SGLang」(`fastpathlabs.ai/blog/fastpath-agentic-recipe`)——后三者均二手线索/未核验，未作为确定事实采纳。
- **未采集/无可用数据**：Reddit、YouTube、X 本期未采集；窗口内未核验到可确认的官方视频发布。
- **证据等级口径**：官方一手=官方提交/releases/issues API 正文、官方 PR 页面正文、官方博客；官方学习资料=vLLM 官方博客与 recipe（性能多为第三方 InferenceX run，标「描述级/第三方基准」）；二手线索=HN 故事与社区转述；未核验=社区 API/视频页不可达或无一手确认的内容。
- **描述级/未闭合项清单（本期）**：TRT-LLM `#21850`（Rubin FC1/FC2 未在 sm_107 运行、needs follow-up）、`#22274`（静默 warmup fallback、vendor lock 未合并上游、多条 pipeline 失败、当前 head CI 不可用）、`#22552`（API 变更、测试 key 可能拼错、作者未在 sm_103 跑 dep8、CI 失败）、`#22666`（产品默认 layout 未改、不同 residual-quantization 互操作未解决）、`#22645`（GPU/E2E 未跑）、`#22777`（无测试守卫）、`#22794`/`#22057`/`#21404`/`#22682` 标题级；SGLang `#49464`（速度为离线日志估计、非实测 A/B）、`#51416`（性能未 benchmark）、`#52929`（CPU host 测试）、`#51787`（Base/AMD CI ✗、Extra 跳过）、`#52643`（prompt logprob 平均差 0.26、decode 读数原因未明）；vLLM `#67185`（保留的 streaming-resume 回归 upstream 仍失败、GPU 端到端/精度/性能待验）、`#66052`（CUDA 无硬件未复现）、`#61430`（Buildkite 命中与 upstream 相同的 AMD 抢占断言）、`#69752`/`#69749`（配方性能为第三方 InferenceX run/描述级）、`#69041`（TPOT 上升权衡）；llama.cpp `#32875`/`#32942`（形状特定微基准）、`#32721`（draft）、`#32897`/`#32500`/`#32747`/`#32142`/`#32182`/`#30997`（正文仅评审对话/模板提示，标题级）、`#30867`/`#32805`/`#32909`（HTML 正文为空，标题级）。以上均按「官方一手但验证未闭合/描述级」处理，不作为确定结论。

---

> **复核说明（本期）**：GitHub commits API 与 releases API 均 HTTP 200 且四家提交在窗口内单页取全（最高 96 < per_page=100，无截断）；PR 正文抓取在 issues API 配额（60/h）耗尽后切换 PR 页面 HTML 提取，两种来源均属官方一手，已在第 8 节逐条标注正文质量差异。vLLM 官方博客发布日期为 2026-11-23（窗口前一天），系经窗口内 HN 线索回溯核验后补录，已显式说明以免与关注窗口混淆。
