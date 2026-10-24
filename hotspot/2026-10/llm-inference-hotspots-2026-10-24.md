# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-24（北京时间）
- **关注窗口：** 2026-10-23 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-23 08:00 – 2026-10-24 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-23 UTC 窗口，分别返回 **90 / 96 / 52 / 21** 条，均 HTTP 200）、四家官方 releases API、**TensorRT-LLM v1.4.0rc1 发布说明全文**（39,615 字符）、**12 个 PR 页面正文**（HTML 直抓，不消耗 API 配额，含 vLLM `#63268`/`#54163`/`#63118`/`#62591`、SGLang `#45700`/`#45407`/`#45941`、TensorRT-LLM `#20452`/`#19989`/`#19773`、llama.cpp `#30804`/`#30663`）。**窗口特征：** 2026-10-23 为**周四**，四家提交量处于工作日中位（vLLM 90 / SGLang 96 高于上期周一 83/80，因本期未经历发版后「补丁 + CI 收敛」的回落；TRT-LLM 从 27 跃升至 52，与 v1.4.0rc1 于前一日 10-22 打 tag 后的稳定化收尾直接相关；llama.cpp 21 为周四正常量）。HN Algolia 公开 API 可访问（HTTP 200），窗口内 `vllm` 8 条命中**全部为「LLM」模糊匹配，无一条指向 vLLM 引擎本身**；`sglang`、`llama.cpp`、`TensorRT-LLM`、`speculative decoding` **均 0 命中**；`KV cache` 1 条为 Show HN《HyperCache》（二手线索，见第 8 节）。Reddit、YouTube、X 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** **TensorRT-LLM v1.4.0rc1 发布**（2026-10-22T01:38:13Z，prerelease）——这是本期最受关注的版本事件，但其 tag 落在关注窗口前一日；本期 52 条 TRT-LLM 提交中相当部分为 rc1 稳定化收尾。vLLM 仍为 **v0.32.0**（2026-10-18），SGLang 仍为 **v0.5.22**（2026-10-19），二者窗口内无新版本。llama.cpp 滚动 nightly 共 12 个标签（b11835–b11846，b11846 发布于 2026-10-24T00:01:59Z，已略超出 UTC 窗口）。
- **本期方法论沿用：** 继续用 GitHub commit API 的 UTC 日期窗口 + PR 页面正文 HTML 直抓（div 深度配对，避免 `</div>\s*</td>` 正则把正文截断到几十字符）。延续「版本核验」口径：发布说明不等于「修复已进版」；本期对 TRT-LLM rc1 的 BREAKING 项仅作发布说明级记录（rc1 为 prerelease，且其发布说明即权威来源），未做 commit 祖先关系核验。

---

## 1. 版本状态与本期头号事件：TensorRT-LLM v1.4.0rc1（tag 于 2026-10-22，prerelease）

**主题**  NVIDIA 于 **2026-10-22T01:38:13Z** 发布 **TensorRT-LLM v1.4.0rc1**（prerelease，自 v1.4.0rc0 2026-10-14 起的首个新候选）。这是本期唯一一个正式版本事件；其 tag 比关注窗口早约 21 小时，但它是该窗口内 TRT-LLM 提交量从 27 跃至 52 的直接原因，也是本期头号事件。

**内容详情（取自发布说明全文，39,615 字符，以下为与推理优化强相关的条目）：**

- **BREAKING — API/默认值：** 默认切换到 **KV cache manager V2** 与 **Python KV cache transceiver**（#17495）；移除未使用的 `LlmRequest` 字段（#19967）；C++ `ArrayView` API 替换为 `std::span`（#20205）。
- **模型支持：** 新增 **DeepSeek-V4.1-Flash**（#19727）、**NVFP4 GLM-5.3-Flash** checkpoint 支持（#20127）、**Qwen3.8-Flash-Next NVFP4** 作为实验性 Modeling V2 目标（#20143）、DFlash 支持 **LiLiCorr** draft checkpoint（#19856）；移除 Nemotron-H NVFP4 在 Hopper 上的自动 W4A16 回退（#15772）。
- **Feature（KV / 解耦 / 量化）：** 集成 **Mooncake connector** 与 KV cache manager V2（#19171）；**KV cache V2 host tier 渐进填充**（#19650，即上期档案记录的 `#19650` 落地）；新增 **RoPE 精度选项**用于 NVFP4 冷页压缩（#19391）；**MiniMax-M3 稠密层改用 paged attention** 并削减 NVFP4 KV 主机开销（#20159、#19974）；**Blackwell CuTe DSL MLA** 支持 packed query heads 与 compact inputs（#19338）；**PrimTS** dense BF16 / 混合 BF16-FP8 / FP8 上下文注意力路径（#20006）；Rubin locality-domain 执行与 SM107 / 多节点 MegaMoE 初始化（#19272、#19183）；**Kimi K3 在 Rubin 优化并避免 KDA prefill 的 cuBLAS 回退**（#19960、#20084）。
- **Feature（投机 / 调度）：** ragged DSpark 验证（置信度策略 + 校准打分，#19707/#19818/#20071）；确定性 FP32 GVR V2 top-K 解码并削减溢出开销（#20100、#20290）；**可编译仅 piecewise CUDA graph forward** 的 opt-in 开关（#18366）；MiniMax-H3 支持运行时 sleep/wakeup（#19016、#19604）；KV cache V2 池再平衡参数可配置（#19077，即上期记录项落地）。
- **Fix（值得记）：** 对齐 MiniMax-M3 / GLM 流式与非流式的推理与工具解析（#20442、#19935）；**保持 overlap scheduler 在等待解耦 KV 接收时响应**（#19857、#19940）；**修正 overlap 投机下 cache 命中的 draft KV 调度与 prompt-tail 重算**（#19745、#19748）；**96-head MLA 在 FMHA autotuner 启用并修正 CuTe MLA page-table 边界**（#18599、#20256，即上期记录的 `#18599` 落地）；**SM120 DSA KV append 因缺失 stride factor 崩溃**（#20062）；**bound packed indexer prefill logits 并用 64 位偏移处理 indexer K-cache**（#20164、#20190）；**DeepSeek-V4 张量并行 decode all-reduce 避开 NCCL**（#20233）。

**应用场景**  MoE 大模型（DeepSeek-V4.x / GLM-5.x / Qwen3.8-Flash-Next / MiniMax-M3·H3）在 NVIDIA GPU 上的 KV 分层、PD 解耦、投机解码与量化部署；KVCM V2 默认化意味着所有新部署开箱即用分层 KV，但也意味着**升级到 rc1 后行为变化需重新验收**（见研判）。

**研判**
- **事实：** v1.4.0rc1 是 TRT-LLM 把「KV cache manager V2 + Python transceiver」提升为默认后的第一个候选版本；发布说明中大量 Fix 与 Feature 与上期（2026-10-19/20）档案中观察到的 pr 在途项（#18599、#19650、#19960、#20256 等）一一对应，说明这些修复在本期随 rc1 一起进入预发布线。
- **事实（默认变更风险）：** 发布说明把「默认 KVCM V2 + Python transceiver」列为 **BREAKING**。上期档案已记录 KVCM V2（#19077 再平衡可配置、#19650 host tier 渐进填充）是「默认成为主要风险载体」模式的最新实例——**升级到 rc1 的部署需要重新验收 KV 命中率、分层行为与解耦传输语义**，因为默认值变了。
- **观点：** rc1 的发布说明几乎是「过去两周四家引擎共同主题」的 TRT-LLM 镜像：**KV 分层 + 解耦 + MoE/投机量化 + Blackwell/Rubin 后端**。对自建服务而言，rc1 最有价值的不是某个单一特性，而是**它把 KVCM V2 设为默认**——这会把「分层 KV 必须显式配置」的旧认知一次性翻转为「分层 KV 默认开启」，部署方应主动核对 host tier 占用与 PD 语义。

**来源**
- Release：https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc1 （2026-10-22T01:38:13Z，prerelease，发布说明 39,615 字符）

**证据等级：** 官方一手（release 页面）。所有「新增/修复」条目均为发布说明口径；其中头号 BREAKING 项 **#17495（默认 KVCM V2 + Python transceiver）已通过 compare API 与 v1.4.0rc1 做 commit 祖先关系核验**：该 PR `merged_at=2026-10-12T07:03:41Z`，`compare/v1.4.0rc1...afc1fa27447464405e971b39ff3c7df63cc85e71` 返回 `status=behind, ahead_by=0, behind_by=189`，即 merge commit 为 rc1 的祖先，**确认已包含在 rc1 中**（其余条目仍仅发布说明级记录，rc1 为 prerelease）。

---

## 2. vLLM：10-23 提交（90 条）——ROCm 静默错误与「前缀缓存 × 投机」正确性债务

**主题**  vLLM 在 2026-10-23 UTC 窗口合并 **90 条**提交（v0.32.0 之后的常规工作日量），构成：Fix 29、ROCm/AMD 19、CI/Test 18、CUDA/Kernel 18、Quant 16、KV/Cache 17、MoE/EP 11、Perf 5、SpecDecode 3，其余 18 条为前端/模型/重构。本期最值得读的是 **3 条「静默错误 / 进程崩溃」级修复**，集中在 ROCm 与投机/前缀缓存交叉面。

### 2.1 `#63268` ROCm gfx942：Triton paged decode 漏写输出 heads（静默错误）

**主题**  `[ROCm][Bugfix] Fix Triton paged decode leaving output heads unwritten on gfx942`（`#63268`）。

**内容详情（取自 PR 正文）**  `test_prefix_prefill.py` 的一个用例（interleaved V cache，#42755 引入）在 MI300 上间歇性失败（build 96693），本地 20 跑中 6 跑失败。Triton decode kernel **只写了该 decode token 的 2048 个输出值中的 263 个**，结果取决于输出缓冲被复用时残留的内存。kernel 逻辑本身正确（`TRITON_INTERPRET=1` 或 `AMDGCN_USE_BUFFER_OPS=0` 下每个值都被写入），但 **LLVM IR 算出的 64 位 head offset 正确，而 gfx942 ISA 从不读取 `output_stride_1` 参数、改用无关的 spilled SGPR（block-table row offset）做乘法**，行落到错误偏移，多数值未写、部分 head 互相覆盖。这只发生在有数百个 SGPR spill 的变体；**把 per-head offset 用 32 位计算即可规避**（即对后端 miscompile 的 workaround）。影响不限于 interleaved V：normal layout 在 Triton fallback、128/256 token page、head size 128/256 下也丢输出（hybrid 大模型的大 page 会命中）。在 60 组 head-size/GQA/page-size/layout 的 decode sweep 中，**修复前 18 组丢 store、修复后 0 组**；kernel 时间变化 −0.4% ~ +1.4%。PR 注明「AI assistance (Claude Code) was used to draft this change」，且作者逐行复核。

**应用场景**  ROCm（MI300/MI355）上的 Triton paged decode、hybrid 大模型、大 KV page。

**研判**
- **事实：** 这是一个**静默错误**（无报错，仅部分输出值未写，结果取决于内存复用），根因是 gfx942 后端 miscompile（SGPR 溢出），修复是 32 位计算的 workaround。
- **观点：** 与上期 SGLang `#44564`（token 下标当页下标用致小 ID 整页覆盖）、TRT-LLM `#20779`（CuteDSL NVFP4 量化 prefill 下不同步）同属一类：**异构后端的「静默错输出」是本期四家引擎的高频主题**，且都能通过「填充内存 / 改精度 / 改同步」被发现。ROCm 部署方对 decode 输出应保留数值抽检。

**来源**  https://github.com/vllm-project/vllm/pull/63268 （merge 在 2026-10-23 窗口内；页面 464,038 字节，正文已核验）

### 2.2 `#54163` 投机解码不再误丢最后一个前缀缓存块（恢复 Mamba checkpoint）

**主题**  `[Bugfix][Spec Decode] DFlash/DSpark no longer drop the last prefix-cache block (mamba align context recompute)`（`#54163`，Fixes #53477）。

**内容详情（取自 PR 正文）**  DFlash/DSpark 保留 target KV 缓存，本不需要 EAGLE 式的「丢弃最后一个前缀缓存块」。原 `SpeculativeConfig.use_eagle()` 是个宽泛谓词，会把 DFlash/DSpark 也匹配进去，导致误丢。**修正该作用域后，`--mamba-cache-mode align` 下的可复用 Mamba checkpoint 被恢复，后续回复不再全量重算。** 验证（2×L40S，Qwen3.5-4B hybrid target + Qwen3.5-4B-DFlash draft，三回合共享 30× system prefix）：

| 配置 | 修复前 cached | 修复后 cached |
|---|---|---|
| DFlash + mamba align | [0, 0, 0]（每回合全重算） | [0, 560, 560] |
| DFlash + mamba all | [0, 0, 0] | [0, 560, 560] |
| 无 spec + align/all | [0, 1056, 1136] | 不变 |
| MTP + align | 无回归 | 无回归 |

验收（同模型、seed、temp=0、max_tokens=128、预热前缀缓存、V2 runner）：DFlash 32 prompts 接受率/长度 0.14804/3.36859 → 0.15199/3.43189；DSpark 128 prompts 0.34399/3.40794 → 0.34512/3.41582。独立验证：Intel Arc B70 + Qwen3.8-27B-INT4 + DFlash2 前缀缓存 token 损坏已解决；4×DGX Spark GB10（TP4）+ GLM-5.3-Flash + DFlash2 相同请求前缀命中 **0/12,552 → 4,608**，TTFT 3.9s → 1.55s。

**应用场景**  hybrid 模型（Mamba/DSv 类）上的 DFlash/DSpark 投机解码部署，尤其是多回合对话复用前缀缓存的场景。

**研判**
- **事实：** 这是「前缀缓存 × 投机解码 × MoE 路由」三者交叉的正确性债务（详见第 6 节累计结论第 5 条）的又一实例；修复后前缀命中与 Mamba checkpoint 复用恢复，且无回归。
- **观点：** 这条 PR 的「独立验证」里 GLM-5.3-Flash + DFlash2 的「相同请求前缀命中 0/12,552 → 4,608」说明：**在修复前，开了 DFlash2 的 GLM 部署几乎完全无法命中前缀缓存**——这对有系统前缀的多用户 SaaS 是隐蔽的吞吐损失，值得作为上线验收项。

**来源**  https://github.com/vllm-project/vllm/pull/54163 （merge 在 2026-10-23 窗口内；页面 536,715 字节，正文已核验）

### 2.3 `#63118` cooperative_topk 在 NaN 分数下非法访存（进程崩溃）→ 自 #59645 起出现

**主题**  `[Bugfix][Kernel] Fix cooperative_topk illegal memory access on NaN scores`（`#63118`）。

**内容详情（取自 PR 正文）**  自 `#59645` 起，当一行 64K 列中 NaN 数量超过 top_k（如 256K 行的 1%、或整行全 NaN）时，cooperative_topk 以 `cudaErrorIllegalAddress` 崩溃。NaN 分数可在 warmup 与 CUDA graph capture 期间进入 decode top-k，此时 indexer 读取未初始化的 KV block。**根因：nvcc 把整数表达式 `bits | 0x80000000u` 编译成 FP 操作 `-|x|`**——对有限值与 ±inf 位精确，但 FADD 把 NaN 规范化成 `0x7FFFFFFF`（−0.0 的有序键）；FP16 coarse bin 把 NaN 放在 +inf 之上，两种排序不一致，threshold bin 溢出后走到 single-key 路径、以 −0.0 为 pivot，于是每个正分都被算作「高于」它，未检查的 append 写穿 `s_topk`。**修复：通过 opaque `mov.b32` 读取 bits**，使 key 位精确、NaN 在 coarse 与 exact 中都在 +inf 之上，与 `torch.topk` 一致。

**应用场景**  decode top-k（投机解码候选、路由等）在 warmup/graph capture 期间遇到未初始化 KV 的场景。

**研判**
- **事实：** 这是一个**编译器的位操作语义陷阱**导致的崩溃，修复是「用 opaque mov 保位」。它与 `#63268` 同属「数值边界（NaN/未初始化）→ 静默或崩溃级错误」一类。
- **观点：** 这条的价值在于它揭示了一个**可迁移的审查点**：任何把整数位模式当 FP 操作的 CUDA 代码（尤其 `| 0x80000000u` 这种符号位技巧）都应被审计——`nvcc` 在优化下可能把位运算重写成 FP 运算并改变 NaN 语义。

**来源**  https://github.com/vllm-project/vllm/pull/63118 （merge 在 2026-10-23 窗口内；页面 503,492 字节，正文已核验）

### 2.4 `#62591` 前缀命中复用了 routed experts 已被逐出的 KV（BlockObjectStoreError）

**主题**  `[Bugfix][AuxOutput] Retain routed experts while a KV cache tier holds their blocks`（`#62591`，Split from #58328，after #62392/#62489）。

**内容详情（取自 PR 正文）**  routed-experts store 按自己的 LRU 逐出，于是**前缀命中可能复用其 routed experts 已丢失的 KV**，worker 以 `BlockObjectStoreError` 失败。修复：调度器从 KV cache events 追踪哪些 block 仍有 routed experts，并告知 worker 保留/释放哪些；本地命中越过保留 block 的部分被丢弃；外部命中被裁剪到 connector 能加载的范围（新增 `KVConnectorBase_V1.external_load_alignment`，仅 full attention），或整段丢弃；store 从 KV cache 大小起步、**超了就带警告增长而非直接失败**。未覆盖：routed experts prompt 之前的命中仍需保留输出、hybrid Mamba 跨 block 中点的命中会被丢弃。main 上该组新测试失败（命中越过保留输出仍被服务、store 报「cannot retain the requested batch」），修复后全过。

**应用场景**  vLLM 的 Auxiliary-Output（routed experts 卸载/分层）与 KV 分层/解耦传输组合部署。

**研判**
- **事实：** 这是「KV 分层/前缀缓存 × MoE routed experts 生命周期」交叉的正确性修复，与上期 `#40703`/`#41769`（SGLang 的 index-K 省略、routed 权重和压缩）同属 MoE 在分层 KV 下的状态一致性问题。
- **观点：** 它再次印证第 6 节的累计判断：**「前缀命中」与「专家/递归状态的生命周期」是两个独立系统，默认各自 LRU 时必然冲突**；任何把 MoE 专家状态与 KV 分开管理的引擎，都需要一条显式的「保留/裁剪」协议，否则就是 `BlockObjectStoreError` 这类硬失败。

**来源**  https://github.com/vllm-project/vllm/pull/62591 （merge 在 2026-10-23 窗口内；页面 433,955 字节，正文已核验）

### 2.5 其余值得记的 vLLM 提交

- **`#62384`** `[Feature][PCP] Support decoder-side SWA bounded replay with APC for DeepSeek-V4.1-Flash`——延续上期 DSv4.1 的 PD/解耦主题。
- **`#60834`** `[ROCm][Perf] Apply opt-in paged MXFP4 sparse indexer dense path for DeepSeek-V4`——ROCm 上的 MXFP4 稀疏索引器 opt-in 路径。
- **`#61855`** `[ROCm][Perf] Triton port of the fused GDN MTP decode kernel`——ROCm 上 GDN MTP decode 的 Triton 融合 kernel。
- **`#63268`/`#63131`/`#60644`/`#62756`/`#62757`** 等 19 条 ROCm/AMD 相关提交显示：**本期 vLLM 的 ROCm 侧活动密度显著高于其他后端**，集中在 MXFP4、GDN MTP、Muse Spark、Kimi-K3 的 gfx942/MI355 适配与修复。
- **`#40186`** `[entrypoints/openai] add kv params to streaming api response`——OpenAI 兼容接口在流式响应里返回 KV 参数，便于可观测。
- **`#61509`** `[Bugfix][MRV2] Don't count adaptive verification profiling cache as CUDA graph memory`——显存记账修复（避免把验证 profiling 缓存算进 CUDA graph 显存预算）。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文 HTML 直抓）。所有吞吐/命中率/时间数字均为各 PR 自测口径，非统一基准；`#63268`/`#54163`/`#63118`/`#62591` 的正文细节为本次独立抓取核验。

---

## 3. SGLang：10-23 提交（96 条）——DSv4.1 prefill 图重放、融合 DFLASH KV 与 diffusion 性能潮

**主题**  SGLang 在 2026-10-23 UTC 窗口合并 **96 条**提交（v0.5.22 之后的常规工作日量），构成：Fix 32、Perf 25、CI/Test 19、MoE/EP 20、KV/Cache 15、CUDA/Kernel 15、ROCm/AMD 10、Diffusion 相关（跨 Fix/Perf/CI 大量）约 30+、Quant 8、SpecDecode 2。本期两条主线：**(A) DeepSeek-V4.1 的 prefill CUDA graph 重放与 MegaGate 优化**；**(B) diffusion/multimodal 成为本期最大主题群**，以及 **HiCache 与 DFLASH KV 的融合**。

### 3.1 `#45407` + `#45941` DeepSeek-V4.1 在 SM90 的 breakable prefill graph 修复与「16K 之后重放图」

**主题**  `[Fix][NVIDIA] deepseek-v4.1: Fix the breakable prefill CUDA graph in the late-layer tail on SM90`（`#45407`）及其后续 `[Perf] dsv4: Replay the V4.1 prefill graph past 16K contexts on SM90`（`#45941`）。

**内容详情（取自 PR 正文）**
- **`#45407`（修复）**：H200（SM90）上，DSv4.1-Flash 开启 decoder SWA bounded replay（默认开）且 `--cuda-graph-backend-prefill breakable` 时**无法服务**。late-layer tail 在两处失败，且都在 SM100 不走的路径上：SM90 没有 in-graph low-ratio 路径（`low_ratio_prefill_graph=false`），ratio-1/2 源作为 eager break 或（一处）被 capture。第一处 fault 是 `torch.repeat_interleave` 的设备端 assert（`Repeat.cu:20`），调用链 `bcg_deepseek_v4_low_ratio_sources → forward_low_ratio_sources → token_req_indices → repeat_interleave(req, extend_seq_lens, output_size=positions.shape[0])`；tail 内 eager break 只看到 tail 的 live rows，而 `extend_seq_lens` 仍跨整个 extend。修复：当 hoisted rows 比 break 的 rows 长时，取其 live prefix 而非从 `extend_seq_lens` 重建。main @ c1c287a4 仍可复现。
- **`#45941`（perf/行为）**：在 CUDA 上，V4.1 prefill CUDA graph 在**没有 c4/c128 compress metadata 被 capture 时，于 live context 宽度构建重放 metadata**，于是 SM90 在 16K 之后像 SM100 一样重放图（该路径上没有任何依赖宽度的 capture）。**16K 默认值移到 HIP 后端**（其 prefill graph metadata 在固定宽度 capture）。行为变化仅影响「V4.1 + `--cuda-graph-backend-prefill breakable` + CUDA + 无 two-level candidates（SM90）」：16K 之后由 eager 改为重放图；attention context parallelism（此前无默认上限）下也改为 live 宽度构建。Follows `#45407`。

**应用场景**  H200/SM90 上 DSv4.1-Flash 的 breakable prefill CUDA graph 部署（长上下文 16K+ 重放、CP 场景）。

**研判**
- **事实：** 这是 SM90 与 SM100 在 low-ratio prefill graph 路径上的**行为分叉**导致的崩溃与修复；`#45941` 把「16K 之后重放」从 SM100-only 扩展到 SM90，但把 16K 默认值下放给 HIP 后端。两条 PR 共同说明 DSv4.1 的 prefill 图在 Blackwell（SM100）与 Hopper（SM90）上仍需要逐后端的图路径调校。
- **观点：** 与上期 SGLang `#44220`（逐层异步加载与 prefill CUDA graph 重放语义冲突）同源——**「把 prefill 拆成可重放的图」在 DSv4.1 上持续是正确性高发区**，根因是「live rows vs extend 全宽」的元数据假设在不同后端/路径不一致。

**来源**  https://github.com/sgl-project/sglang/pull/45407 ｜ https://github.com/sgl-project/sglang/pull/45941 （均 merge 在 2026-10-23 窗口内；正文已核验）

### 3.2 `#45700` HiCache：统一混合 SWA 池里支持 fused DFLASH KV（消除私有 draft 池回退）

**主题**  `[Feat] HiCache: Support fused DFLASH KV in unified hybrid SWA pools`（`#45700`）。

**内容详情（取自 PR 正文）**  Muse Spark 在 unified memory 与 HiCache 同时开启时，会**回退到私有 DFLASH KV 池**。本 PR 让 target 与 draft 在每 TP rank 共享同一 GPU backing pool。做法：把 FA4 draft 的 BF16 行放在 target 的 FULL page envelope 内、与其 FP8 KV 行一起，并复用现有 HiCache page translation 一起传输；在 DFLASH prefill（含 CUDA graph 重放）前等 envelope 恢复，并去掉独立的 draft host sidecar。这启用了「full-attention DFLASH draft 在 unified hybrid-SWA target 上的 L2 HiCache」；其他 host-pool 路径（含 L3 存储）保留原有私有 draft 回退。不新增 CUDA kernel。精度测试：1,464 个固定输入的 FA4/TRT-LLM 输出与 LSE 比较**位精确（bit-exact）**；真实模型缓存测试从 host 恢复 2,304 tokens 且零 device-cache 命中、正确回忆前缀中存储的参考数。4×B300、TP4/EP4、target `trtllm_mha`、draft `RadixArk/Muse-Spark-Open-DFlash2 with fa4`、unified memory、HiCache size 200：

| 数据集 | 正确/总数 | 得分 |
|---|---|---|
| AIME26（8 repeats） | 228 / 240 | 95.00% |
| GPQA Diamond | 177 / 198 | 89.39% |
| Video-MME short（96 frames） | 798 / 900 | 88.67% |
| GSM8K | 1284 / 1319 | 97.35% |

全部零请求错误（AIME26/GPQA 各 1 个输出截断；Video-MME/GSM8K 无）。PR 注明**未做受控吞吐对比**（现有同步 host-arena compaction 在连续评测中造成停顿，本 PR 未改该分配器）。

**应用场景**  unified memory + HiCache 同时开启的 DFLASH 投机解码部署（Muse Spark / DFlash2 draft）。

**研判**
- **事实：** 这是「HiCache 分层 KV × 投机 draft 状态」融合的落地，且**位精确**验证了混合 dtype envelope 的传输正确性；与 vLLM `#62591`、SGLang 历史 `#40917`/`#41907`/`#43333`（HiCache 静默损坏三处）形成对照——本期 SGLang 在**主动修复** DFLASH KV 与分层缓存的共存。
- **观点：** 与上期 SGLang `#40917` 等「HiCache 静默损坏」相反，本期是「把 draft 状态并入统一池并位精确验证」——说明 SGLang 已把 DFLASH KV 的分层管理从「私有 sidecar」推进到「与 target KV 同池同传输」。但**「未做受控吞吐对比」应被读作未验证性能收益**，只验证了正确性。

**来源**  https://github.com/sgl-project/sglang/pull/45700 （merge 在 2026-10-23 窗口内；页面 448,038 字节，正文已核验）

### 3.3 SGLang 其余值得记的提交

- **`#45933`** `[Perf] dsv41: Drop the MegaGate padded-row copy on sgl-deep-gemm 0.2.3.post2` 与 **`#45941`** 共同构成 DSv4.1 MegaGate 优化；**`#45969`** `[Fix] dsv41: Keep MegaGate out of prefill graphs by gating on the forward mode`。
- **`#45934`/`#45938`** DSv4.1 的 K 行写入与 prefill index selection 的 kernel 级优化（one warp per token、单 kernel 映射 pool slot）。
- **`#40929`** `[Feat] KVShard: Support kv cache sharding for MTP`——MTP 的 KV 分片。
- **`#41187`** `[Feat] hicache: Support MLA DCP with HiCache L3(file/mooncake)`——MLA 的 decode context parallelism 与 HiCache L3 结合。
- **`#43774`** `[Feat] glm5.3flash: Support padded CP BCG and prefix replay with PD/DFlash2`——GLM-5.3-Flash 的 CP + 前缀重放 + PD/DFlash2。
- **`#46066`** `[Fix] rust-server: Keep a transfer backend's own bootstrap port on prefill`——PD rust-server 的 bootstrap 端口正确性。
- **`#45922`** `[Fix] disaggregation: Restart a lost prefill's heartbeat miss count`——PD 心跳丢失计数重启。
- **Diffusion 主题潮：** `#45966`/`#45979`/`#45987`/`#45967`（LTX-2/Cosmos3/Qwen-Image 的 denoising host overhead、residency probe、in-memory 图像响应、per-step host sync 消除）、`#45943`/`#45995`（diffusion 的 LoRA delta 与 LTX-2.3 BCG video RoPE offset）、`#45983`/`#45861`（AMD CI 的 diffusion decode stream drain 与 LTX-2.5 跳过）——显示 **SGLang 正在把 diffusion/multimodal 生图作为一等公民推进**，本期 diffuison 相关提交约占总量 1/3。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文 HTML 直抓）。`#45407`/`#45941`/`#45700` 为正文核验；其余为提交标题级记录。`#45700` 的精度数字为 PR 自测，吞吐未测。

---

## 4. TensorRT-LLM：10-23 提交（52 条）——rc1 稳定化收尾与三条值得读的 in-window PR

**主题**  TRT-LLM 在 2026-10-23 UTC 窗口合并 **52 条**提交（较上期 27 条接近翻倍，主因 v1.4.0rc1 于 10-22 打 tag 后的稳定化收尾）。构成：Fix 22、CI/Test 10、CUDA/Kernel 11、MoE/EP 8、KV/Cache 6、Perf 7、Quant 7、SpecDecode 2、infra/other 约 10。rc1 的 BREAKING/Feature 已见第 1 节；本节记录窗口内**新合入且值得读**的 PR。

### 4.1 `#20452` First-N/Last-N 混合精度 KV 缓存（KVCM V2 原生，NVFP4 历史 + 高精度首尾）

**主题**  `[TRTLLM-17204][feat] Add First-N/Last-N mixed-precision KV caching using native SWA in KVCM V2`（`#20452`）。

**内容详情（取自 PR 正文）**  在 KVCM V2 原生池里加入可选的**混合精度 KV 缓存**：full-attention 层把**整个 KV 历史放在低精度池**（NVFP4 为首个低精度格式），而**每个请求的前 `mixed_precision_first_n` 与后 `mixed_precision_last_n` 个 token 放在 KV dtype 的第二池**（native sliding-window + sink layer group，复用 V2 现有 window 逻辑做分配/复用/逐出）。注意力 kernel 不变：每层前把 batch gather 到 KV dtype 的 staging 拷贝、在其上跑注意力、只把本步新写的 token 写回两个池。配置：`kv_cache_config.dtype = "mixed_nvfp4_{fp8|float16|bfloat16}"` + `mixed_precision_first_n`/`mixed_precision_last_n`（默认 256），`pool_ratio` 可选在两池间分配预算。无新 manager 类（折叠进 `KVCacheManagerV2`）；TRTLLM/FlashInfer/GQA/MLA 共用一条 CuTe DSL gather+writeback kernel。QA：`test_mixed_precision_kv_cache.py`（l0_b200，池布局/尺寸/保留/复用/pool_ratio/解耦映射）、`test_mixed_precision_backends.py`（TRTLLM+FlashInfer prefill/decode、MLA、转换 kernel、staging 几何，TP/attention DP）、`test_trtllm_serve_config_schemas.py`（l0_cpu，新 dtype 串与保留计数）。B200 手动 E2E：Qwen3-8B BF16 dense vs mixed 两后端输出一致；GLM-4.5-Air-FP4（FP8 KV）在 dense run-to-run 带内。Prototype。

**应用场景**  长上下文下「首尾高精度 + 中间低精度」的 KV 显存压缩（NVFP4 历史 + BF16/FP8/FP16 首尾），与 vLLM/llama.cpp 的 KV 量化思路同源但用「sink + sliding window」原生实现。

**研判**
- **事实：** 这是 KVCM V2 默认化（rc1 BREAKING）之后的首个**混合精度 KV 原语**，把「KV 量化」从「整池单一精度」推进到「首尾高精度 + 历史低精度」的细粒度方案，且复用 V2 的 window 逻辑（无新 manager）。
- **观点：** 与上期社区侧 arXiv 2610.21500 的「远端 KV 层成本模型」形成工程对照——TRT-LLM 选择的是**本地混合精度**而非**远端分层**，二者解决同一问题（长上下文 KV 成本）的不同侧面。标记为 Prototype，说明尚未 GA。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20452 （merge 在 2026-10-23 窗口内；页面 715,567 字节，正文已核验）

### 4.2 `#19989` 单模型投机解码的通用件（默认关闭，SM100 专用 step-copy kernel）

**主题**  `[None][perf] One-model speculative decoding: host work between steps, worker-produced target logits, KDA drafts' records`（`#19989`，依赖 #19988，曾评审为 #19816/#19814/#19817）。

**内容详情（取自 PR 正文）**  这些是**单模型投机解码的通用件，全部默认关闭或保持现有模型行为不变**。拆分：
- `cute_dsl_kernels/spec_step_copies/`：`StepInputGather`/`SlotScatter` 各一次 launch，用于 overlap scheduler 的输入 gather 与 sampler 的 store 更新（**仅 SM100**；kernel 不覆盖时回退 torch 路径）。
- `pyexecutor`：`copy_to_device_if_changed`（值未变则跳过拷贝，debug 与单测中对 device 校验）、sampler 的 host 拷贝移到 side stream、CUDA graph 共享 engine 的 input buffers。
- `speculative/interface.py` + `models/modeling_speculative.py`：`SpecWorkerBase.target_logits` 由 spec worker 选布局（默认不变，vocab-shard 契约写在 hook 上）。
- `pyexecutor/kv_cache/mamba_cache_manager.py`：`MambaHybridCacheManagerV2(kda_token_states=...)` 用 per-draft records（`kda_state_tok`）替代 q/k/g、v、beta replay 缓存（默认关闭）。
- 测试：GB200 上本分支 vs main，新测试 `test_spec_step_copies.py`/`test_copy_to_device_if_changed.py`/`test_rng_window_counter.py` 93 passed；`kv_cache` 987 passed / 42 failed（与 main 的 52 failed 同因，每处失败 main 也失败）；`spec_step_copies` 在 compute-sanitizer 下 memcheck 0 errors、racecheck 0 hazards。
- 限制：step-copy kernel 仅 SM100；Mamba2 仅单测无 e2e；**丢弃 H2D staging 在 BS1 多耗 +0.42% host 时间**。

**应用场景**  单模型（self-speculative）投机解码的调度/拷贝优化，面向 Blackwell（SM100）。

**研判**
- **事实：** 这是「把投机解码的 host 工作移到 side stream、用 step-copy kernel 替代逐份拷贝」的工程化，默认关闭、无回归。
- **观点：** 与 vLLM `#54163`（DFlash/DSpark 前缀块）、llama.cpp `#30663`（DFlash2 概率采样）共同构成本期「投机解码正确性 + 效率」的三方信号；TRT-LLM 这条偏**调度/拷贝层面的零碎优化**，收益小（+0.42% host 代价换拷贝消除），但为后续单模型投机铺路。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/19989 （merge 在 2026-10-23 窗口内；页面 830,418 字节，正文已核验）

### 4.3 `#19773` BF16 融合 gate/up GEMM + SwiGLU（QuACK，SM100/SM103，opt-in）

**主题**  `[None][perf] Fuse SwiGLU into the BF16 gate/up GEMM epilogue on SM100`（`#19773`）。

**内容详情（取自 PR 正文）**  在现有 NVFP4 融合路径旁新增 **opt-in 的 BF16 融合 gate/up GEMM + SwiGLU**，经 QuACK 实现。**默认关闭**，按模块 `GatedMLP(fuse_bf16_gate_up_swiglu=True)` 开启；MiniMax-H3 在构造器里决定一次（BF16 配置开启）。单一谓词 `_can_fuse_gate_up_swiglu_bf16(x)` 逐调用判定：模块 opt-in + kernel 可用（SM100/SM103、QuACK 可 import）、plain SwiGLU（无限/alpha/beta）、无 bias 的未量化 BF16 投影、无 TP/其他 GEMM provider、rank≥2 的 CUDA BF16 激活（无 backward）、K/I 为 8 的倍数。命中时 gate/up 投影调用 QuACK `gemm_act(..., activation="swiglu", concat_layout=("B",))`，在 GEMM epilogue 的 FP32 累加器里算 `silu(gate)·up`，只存 `[M, I]` 乘积——`[M, 2I]` BF16 中间与单独的 `silu_and_mul` kernel 消失，输出只舍入一次。`concat_layout=("B",)` 匹配 TRT-LLM 的 `[gate; up]` 权重布局，无需权重拷贝/置换；`tuned=False` 钉住 QuACK SM100 默认 tile（autotuner 按精确形状 key，会在每个新 token 数重新 tune）。QuACK 0.5.0 已是 pinned 依赖；首次 kernel 失败会对其进程禁用并走原生路径（并记一条 warning）。LoRA 保留自有路径。

**应用场景**  BF16 MLP（如 MiniMax-H3）在 Blackwell 上的 gate/up GEMM + SwiGLU 融合，减少中间 BF16 缓冲与一次舍入。

**研判**
- **事实：** 这是「把激活融合进 GEMM epilogue、只舍入一次」的标准优化，但**依赖 QuACK 0.5.0 且 pinned 的 CUTLASS DSL 与之存在漂移风险**（CI 镜像的 `nvidia-cutlass-dsl==4.8.0.dev0` 曾以 `ValueError: too many values to unpack` 拒绝 SM100 tile scheduler）；PR 用「首次失败即禁用」做了防御。
- **观点：** 与上期 TRT-LLM `#19960`/`#20084`（Kimi K3 Rubin 优化）同源——**TRT-LLM 在 Blackwell 上越来越依赖 QuACK/CuTe DSL 融合 kernel**，但「pinned 依赖漂移导致 kernel 静默禁用」是这类融合的运维风险点，应在 CI 里持续校验 QuACK 与 CUTLASS DSL 版本组合。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/19773 （merge 在 2026-10-23 窗口内；页面 781,113 字节，正文已核验）

### 4.4 TRT-LLM 其余值得记的提交

- **`#20653`** `[perf] Integrate FlashInfer KDA prefill with fused convolution`——KDA prefill 与融合卷积结合。
- **`#20167`** `[feat] Enable MiniMax-M3 on Rubin (SM107): NVFP4 KV exemption and 128-token paged FMHA kernels`——MiniMax-M3 上 Rubin。
- **`#20334`** `[fix] Keep low-M GEMM outputs distinct during CUDA graph capture`——graph capture 期间低-M GEMM 输出区分（正确性）。
- **`#20119`** `[fix][nvbugs/6796296] Pad DSv4 NVFP4 compressed FP8 scratch by one FMHA KV tile`——DSv4 NVFP4 压缩 FP8 scratch 补齐一个 FMHA KV tile（越界保护）。
- **`#20779`** `[fix][nvbugs/6448152] Keep CuteDSL NVFP4 input quantization synchronous for prefill batches`——CuteDSL NVFP4 量化在 prefill 下保持同步（异步可能导致量化错）。
- **`#20388`** `[refactor] BREAKING: Migrate LlmRequest fields to Python`——rc1 BREAKING 项的落地。
- **`#20438`** `[chore] Deprecate ngram/user-provided spec dec`——弃用 ngram/用户提供的投机解码。
- **`#20576`** `[feat] Kimi K3: DWDP fabric gather and TCPStore rendezvous, KVCM-managed KDA replay buffers`——Kimi K3 的 KVCM 托管 KDA replay 缓冲。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文 HTML 直抓）。`#20452`/`#19989`/`#19773` 为正文核验；其余为提交标题级记录。nvbugs 编号项为 NVIDIA 内部 issue 引用，未独立核验。

---

## 5. llama.cpp：10-23 提交（21 条）——chunked CUDA graph 与 DFlash2 概率采样

**主题**  llama.cpp 在 2026-10-23 UTC 窗口合并 **21 条**提交（周四正常量），构成：Fix 4、CI/Test 4、CUDA/Kernel 4、SpecDecode 1、Perf 1、ROCm/AMD 1、other 10。本期两条主线：**(A) CUDA graph 分块摊销**；**(B) DFlash2 概率采样扩展**；以及若干后端稳定性修复。

### 5.1 `#30804` + `#31053` CUDA：chunk graphs to amortize graph launch

**主题**  `CUDA: chunk graphs to amortize graph launch`（`#30804`），及 `CUDA: fix build after chunked CUDA graphs (#30804)`（`#31053`）。

**内容详情（取自 PR 正文）**  CUDA graph 的成本随图大小线性增长；**把图拆成更小的 chunk 来运行可摊销该成本**，带来 **1–2% TG（token generation）加速**。点子来自 @ORippler。该 PR 出现两次进入窗口（`#30804` 主体 + `#31053` 构建修复），说明分块改动触发了一次构建回退、随后被补齐。

**应用场景**  llama.cpp 在 CUDA 上的长图/TG 场景，尤其是 batch 较大、graph 捕获成本显著时。

**研判**
- **事实：** 1–2% TG 是 modest 但免费的收益，且无正确性风险（改动是图的分块执行，不改数值）。
- **观点：** 与本期 vLLM/TRT-LLM 的「CUDA graph 复用/分块」主题一致（vLLM Elastic EP 把「保住图」当作核心收益、TRT-LLM `#18366` opt-in piecewise graph）——**「CUDA graph 的捕获/启动成本」已成为各引擎共同的优化对象**，只是手段不同（复用 vs 分块 vs piecewise）。

**来源**  https://github.com/ggml-org/llama.cpp/pull/30804 ｜ https://github.com/ggml-org/llama.cpp/pull/31053 （均 merge 在 2026-10-23 窗口内；正文已核验）

### 5.2 `#30663` 为 DFlash2 drafts 扩展概率采样（带多硬件性能表）

**主题**  `Extend probabilistic sampling for DFlash2 drafts`（`#30663`，延展 #27694）。

**内容详情（取自 PR 正文）**  为 DFlash2 技术加入**概率采样 drafter**，复用 #27694 的 verify-accept-reject 逻辑。PR 给出的多硬件性能（相对上游，temp 为采样温度）：

| 硬件 / 模型 | temp | Δ accepted len | Δ t/s |
|---|---|---|---|
| RTX 5090 / Qwen3.8-27B | 0.0 | +0.00% | −0.20% |
| RTX 5090 / Qwen3.8-27B | 0.4 | +2.76% | +2.04% |
| RTX 5090 / Qwen3.8-27B | 1.0 | +4.33% | +4.20% |
| Muse-Glimmer-30B | 1.0 | +11.86% | +12.33% |
| DGX Spark / Qwen3.8-27B | 1.0 | +4.63% | +4.82% |

温度越高、收益越明显（高 temp 下概率采样 drafter 接受率提升更显著）；低温（0.0）基本持平或略降。

**应用场景**  高温度采样（creative/generative）场景下的 DFlash2 投机解码加速。

**研判**
- **事实：** 概率采样 drafter 在**高温度**下才显著受益（temp=1.0 时 +4~12% t/s），低温几乎无收益——这与投机解码「低温下 draft 已很准、概率采样边际收益小」的直觉一致。
- **观点：** 对落地很实用的一条提示：**DFlash2 概率采样应默认在高温度/生成类负载开启，低温/确定性负载不必开**（甚至可能 −0.2~−0.3% 微损）。

**来源**  https://github.com/ggml-org/llama.cpp/pull/30663 （merge 在 2026-10-23 窗口内；页面 497,020 字节，正文已核验）

### 5.3 llama.cpp 其余值得记的提交

- **`#30837`** `ggm-cuda: Enforce DP4A MMQ tile height multiple of warp size`——MMQ tile 高度对齐 warp size（数值正确性）。
- **`#30885`** `hip: fix bf16 WMMA operand vector type for gfx11/gfx12`——ROCm gfx11/gfx12 的 bf16 WMMA 操作数向量类型修复（正确性）。
- **`#30974`** `vocab: fix Nemotron-3.5-Embed tokenization`——分词修复。
- **`#30969`** `ggml: fix time overflow in ggml_time_us`——时间溢出修复（影响计时类功能）。
- **`#31036`** `ci: add behavior check workflow`——新增行为检查 CI。
- **`#28607`** `mtmd: add Soprano TTS support`——多模态增加 Soprano TTS。
- **`#28725`** `sycl: Add graph record and replay` 与 **`#30684`** `sycl: Replace host waits in fattn-mkl with explicit dependencies`——SYCL（Intel）后端的 graph record/replay 与 host wait 显式依赖（Intel GPU 推理稳定性）。
- **`#31038`** `ui: bump packages to clear npm audit`——安全依赖更新。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文 HTML 直抓）。`#30804`/`#30663` 为正文核验；其余为提交标题级记录。

---

## 6. 当前会话已形成的累计内容结论

> 以下为「逐日热点档案」跨期累积的判断（非单日结论）。第 1–3 条继承自 2026-10-20 档案，第 4–5 条为近期（含本期）新增/强化。

1. **调度器隐式阻塞的定量画像（继承自 10-20）：** 多引擎反复出现「host sync / 逐层加载 / 页面 H2D」落在调度器热路径上，导致 GPU 利用率与吞吐被隐性压低（SGLang 上期 `#44904` 等四条同源 PR 把 host sync 移出热路径、合计 +4.5–9.9%；本期 `#45941` 的 16K 重放、vLLM `#60834` 的 MXFP4 稀疏索引器 opt-in 延续同一方向）。**判读：调度器的「隐性阻塞」已可用提交计数与 GPU 利用率变化定量画像，是评估一个引擎成熟度的高频信号。**
2. **发版窗口的双向性（继承自 10-20）：** 「发版当天合入 main」与「在发版内」是两个独立事实（SGLang v0.5.22 的 `#42557`/`#44904`/`#45075`/`#45223` 均 `diverged`）。本期 TRT-LLM rc1 是反向印证：其发布说明中的大量 Fix/Feature 与上期档案记录的在途 PR 对应，说明 rc1 是这些在途项随预发布线一起落地的节点。**判读：「发布说明含某修复」≠「该修复在版内」，对 prerelease 至少应记录其发布说明作为权威来源。**
3. **默认值成为主要风险载体（继承自 10-20）：** 大量破坏性/静默行为来自默认值翻转（SGLang `--disaggregation-decode-host-receive-threshold` 语义翻转、TRT-LLM rc1「默认 KVCM V2 + Python transceiver」BREAKING）。**判读：升级验收应把「默认值变更」列为一级检查项，而非只看新增 flag。**
4. **「静默错误 / 进程崩溃」在异构后端高密度聚集（本期强化）：** vLLM `#63268`（gfx942 Triton decode 漏写 263/2048 输出，静默）、`#63118`（NaN 下 cooperative_topk 非法访存崩溃）、`#63131`（gfx942 int4 量化 requant 校验）、TRT-LLM `#20779`（CuteDSL NVFP4 量化 prefill 异步致错）——**四家引擎本期均有「异构后端位/精度/同步陷阱 → 静默错或崩溃」的修复**。判读：ROCm/Blackwell/Rubin 等后端的正确性验收应保留数值抽检与 warmup/graph-capture 边界测试。
5. **「前缀缓存 × MoE 路由 × 投机解码」三者交叉的正确性债务持续累积（本期强化）：** vLLM `#62591`（routed experts 逐出致前缀命中复用失效）、`#54163`（DFlash/DSpark 误丢最后前缀块致 Mamba 全重算）；SGLang `#45700`（DLASH KV 与统一 HiCache 池融合、位精确验证）；llama.cpp `#30663`（DFlash2 概率采样）。**判读：任何把「KV 分层/前缀缓存」与「MoE 专家状态 / 递归状态 / 投机 draft 状态」分开管理的引擎，都需要一条显式的「保留/裁剪/对齐」协议，否则表现为硬失败（BlockObjectStoreError）或静默命中率损失（GLM-5.3-Flash + DFlash2 前缀命中 0/12,552 → 4,608）。**

---

## 7. 今日可执行清单

针对本期发现，给正在跑或准备升级这些引擎的团队：

1. **TRT-LLM 升级到 v1.4.0rc1 的验收（高优先）：** 因「默认 KVCM V2 + Python transceiver」是 BREAKING，升级前后必须对比 **KV 命中率、分层行为、PD/解耦 KV 传输语义、96-head MLA 与 SM120 DSA KV append**（`#20062`/`#20256`/`#18599` 相关）。
2. **ROCm/异构后端部署保留数值抽检（高优先）：** 受 vLLM `#63268`/`#63131`、TRT-LLM `#20779` 启发，对 MI300/MI355 上的 decode 输出、量化 requant、prefill 量化做抽样核对，并覆盖 warmup 与 CUDA graph capture 期间的 NaN/未初始化输入。
3. **vLLM DFlash/DSpark + Mamba 部署检查前缀命中：** 跑 `#54163` 场景（多回合共享系统前缀），确认开启投机后前缀命中与 Mamba checkpoint 复用未被误丢；GLM-5.3-Flash + DFlash2 用户应确认前缀命中已恢复。
4. **vLLM Auxiliary-Output / routed experts 卸载 + KV 分层组合：** 核对 `#62591` 的「保留/裁剪」协议已生效，避免 `BlockObjectStoreError`；hybrid Mamba 跨 block 中点的命中仍会被丢弃（已知未覆盖）。
5. **SGLang DSv4.1 on SM90 的 breakable prefill graph：** 开启 `--cuda-graph-backend-prefill breakable` + decoder SWA bounded replay 的用户应确认已含 `#45407`/`#45941`（16K 之后重放、CP live 宽度）；HIP 后端的 16K 默认值已变更。
6. **llama.cpp 高温度生成负载开启 DFlash2 概率采样（`#30663`）：** temp≥0.4 时建议开启（RTX 5090 Qwen3.8-27B 在 temp=1.0 约 +4.2% t/s，Muse-Glimmer-30B +12.3%）；低温/确定性负载不必开。
7. **CUDA graph 成本优化持续关注：** vLLM（图复用）、TRT-LLM（`#18366` piecewise）、llama.cpp（`#30804` chunked）三条不同路线，可按自身后端选型参考。
8. **QuACK / CuTe DSL 融合 kernel 的版本漂移（TRT-LLM 用户）：** 确认 QuACK 与 `nvidia-cutlass-dsl` 版本组合在 CI 中持续校验，避免 `#19773` 式的「首次失败即静默禁用融合」。

---

## 8. 社区与视频资料

- **Hacker News（Algolia，2026-10-23 UTC 窗口）：** `vllm` 8 条命中**全部为「LLM」模糊匹配，无一条指向 vLLM 引擎本身**（如「LLM-d v1.0.0 is out」「Private LLM Inference with Homomorphic Encryption」「LLM Multi-Agent Systems Are Vulnerable to Herding」）；`sglang`、`llama.cpp`、`TensorRT-LLM`、`speculative decoding` **均 0 命中**。
- **KV cache（1 条，二手线索/未核验）：** Show HN《**HyperCache – Distributed KV-cache quantization and relative RoPE reuse**》（2026-10-23T02:56:38Z，2 points）。内容为分布式 KV 缓存量化与相对 RoPE 复用，**非官方引擎仓库、未提供基准或论文链接**，按本档案规范标记为**二手线索/未核验**，不计入引擎事实结论。
- **官方视频：** 本期（窗口内）四家引擎**均未核验到可确认发布日期与内容的官方视频**（vLLM 官方博客窗口内无新文；SGLang/TRT-LLM/llama.cpp 无新官方视频公告）。
- **未采集渠道：** Reddit、YouTube、X（Twitter）按既定范围未纳入；如后续需要，应单独说明并标注来源类型。

**证据等级说明：** 本期所有引擎事实均来自官方 GitHub 提交 API / releases / PR 正文（官方一手）；HN 单条 Show HN 为二手线索；社区转述如无官方源印证一律标记为「未核验」。

---

> 归档说明：本档案为自动化每日追踪产出，覆盖 2026-10-23 UTC 窗口。相邻日期（2026-10-21、2026-10-22）未产生每日档案（自动化未执行），本目录不补写中间日期内容，各期仅按自身关注窗口采集。下一期（如执行）将覆盖 2026-10-24。

> 复核说明（2026-10-24 二次核验）：本档案关注窗口（2026-10-23 UTC）提交计数经二次独立采集仍为 **vLLM 90 / SGLang 96 / TRT-LLM 52 / llama.cpp 21**（GitHub 历史提交不可变，与首采一致）；档案内关键 PR/发布页链接复核均返回 HTTP 200；首采遗留的「TRT-LLM v1.4.0rc1 BREAKING 仅发布说明级记录」已完成 **#17495 的 commit 祖先关系核验**（见第 1 节证据等级），确证该 BREAKING 默认变更已随 rc1 发布。
