# 大模型推理优化每日热点档案

> 档案日期：2026-11-07（本自动化于该日执行）
> 关注窗口：2026-11-06（周五）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（vLLM/SGLang 部分经 issues API 取 markdown；其余经 PR HTML 页面 `comment-body` 提取，不消耗 API 配额）+ HN Algolia（HTTPS，numericFilters 按 2026-11-06 epoch 窗口 1793923200–1794009600）
> 数据质量：四家提交 API 均 HTTP 200；vLLM **105**（page1=100 命中 per_page 上限、page2=5，API 窗口实计数 105，未做 GitHub 网页交叉核验）、SGLang **88**、TensorRT-LLM **44**、llama.cpp **17**；四家 releases API：vLLM 仍 v0.33.0rc3、SGLang 仍 v0.5.23、**TRT-LLM 新增 v1.4.0rc3（2026-11-06T21:38:32Z，prerelease，窗口内发布）**、llama.cpp 仍 v0.9.0（11-04）+ nightly `b12174–b12180`（11-06，prerelease）；PR 正文级/HTML 级细节共 **32 个**（vLLM 12 + SGLang 8 + TRT-LLM 8 + llama.cpp 4）；HN Algolia 窗口内引擎相关高信号弱（vllm 431 命中多为 LLM 网关/工具类、sglang 仅「Who wants to be hired」、KV cache 仅 Simon Willison Qwen 帖；llama.cpp 命中含「Sleep Mode for Llama.cpp」博客 1 分、Ripple-65B Preview、Qwen3.8 生态帖、TieredQ 量化论文，均二手线索/未核验）。

## 1. 版本状态与本期头号事件：Kimi K3.1 跨引擎 day-0 支持落地 + 静默正确性/投机解码正确性双主线延续 + TRT-LLM v1.4.0rc3 发布

**提交计数（2026-11-06 UTC，均 HTTP 200；2026-11-06 为周五工作日，vLLM 105 较上一窗口 11-05 的 105 持平、SGLang 88 较 90 略降、TRT-LLM 44 较 30 回升、llama.cpp 17 较 43 显著回落）**：vLLM **105** / SGLang **88** / TensorRT-LLM **44** / llama.cpp **17**。

**发布状态（窗口内 TRT-LLM 有新版，其余无正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-06 全部 105 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-06 含 Kimi-K3.1 DSA/MegaMoE、HiCache、disaggregation 密集修复/性能。
- **TensorRT-LLM**：**窗口内新增 `v1.4.0rc3`（2026-11-06T21:38:32Z，prerelease）**；上版为 v1.4.0rc2（2026-10-28）。rc3 为 v1.4 正式发布前的重要候选，含大量模型支持与 3 项 BREAKING（详见第 4 节头号事件）。
- **llama.cpp**：仍为 **v0.9.0（2026-11-04T13:08:06Z 正式发布）**；11-06 仅推进 nightly `b12174–b12180`（15:44Z–23:18Z，prerelease），无新正式版本。

**本期头号事件（Kimi K3.1 跨引擎 day-0 支持 + 静默/投机解码正确性双主线延续 + TRT-LLM v1.4.0rc3 发布 + Qwen3.8 家族落地 + SGLang MLU 新后端）**：
1. **Kimi K3.1 跨引擎 day-0 支持同窗口落地（模型支持事件）**：vLLM `#66030`（split 1/5 模型文件 + NVIDIA specialized decode，含 DSA 每 MLA 层、DSA indexer kernels、packed KDA/DSA hybrid caches、CuTe DSL DSA/KDA/MLA/MoE/GEMM kernels）、`#66028`（split 2/5 Rust renderer）、`#65725`（Kimi K3.1-Pro TPU 服务 via vllm-torchtpu）；SGLang `#48721`（DSA decode 上下文并行运行时，port #48414）、`#48714`（opt-in trtllm DSA backend，TP8 B300 trtllm vs flashmla_kv 精度 0.953/0.949、TTFT −6~−12%、KV 容量 +11%）、`#48713`（shared-expert TP 跨 interleave prefill CP）；llama.cpp `#31779`（Kimi-K3.1-Pro 文本模型，DSA 替换全注意力 MLA 层）。**一个新模型在同窗口进入三家开源引擎，属「模型 day-0 支持」协同事件**（趋势第 1 条）。
2. **vLLM 投机解码正确性簇（强化第 5 条）**：`#65229`（自适应验证接受估计器在 drafter 高置信时因 fp32 灾难性抵消把所有置信压成常数 `40.0`，temp=0.5 源码上 **11.5% 的 draft 位置置信被破坏**）、`#66048`（`_insert_resampled_kernel` 跨 warp 读写 `num_sampled` 竞争致越界 lookup + 致命 CUDA 异常，RTX PRO 6000 Blackwell 两次 coredump）、`#65804`（standalone draft model 的 FULL CUDA graph 把 draft prefill 误分类为 prefill、UNIFORM_BATCH 仅保证 decode 安全 → GB200 崩溃，关联 #60989/#65598）、`#65710`（speculative Guidance 批次中 EOS 处理）、`#66010`（Sleep Mode Level 2 仅丢弃参数，修唤醒后非参数状态丢失致 GSM8K→0）。
3. **vLLM KV-sharing / 可扩展 KV cache 正确性（强化第 4/8 条，延续 #64095 跨层 KV 共享）**：`#64986`（异步调度下 `--kv-sharing-fast-prefill` 把 D2H 未完成的 `query_start_loc_cpu` 交给 attention metadata builder，trtllm-gen decode kernel 收到垃圾 query length 挂起；B200 Gemma-4 E4B 上 4649 请求切片 base 4/4 挂起 → 本 PR 20s 完成）、`#65554`（FlashMLA sparse prefill + 可扩展 KV cache #56492 启动非法访存，DeepSeek-V3.2 on B300 复现，扩展 #65225 段对齐）、`#65463`（可扩展 KV cache 下 `--max-model-len auto` 在 warmup 后容量不足时启动失败，改为对 `auto` 重新拟合）。
4. **SGLang HiCache / PD 健壮性簇（强化第 5/8 条）**：`#48522`（分层 cache + L3 后端下，hybrid SWA 模型 L3 prefetch 跨度越过 radix 匹配键，约 **49.6% 的 prompt 长度**会令已付费 RDMA 预取的 KV 被静默丢弃、整 prompt 重算）、`#48693`（PD 传输等待时不计 staged-prefetch 延迟）、`#48450`/`#48496`（unified-pool 页复用/回收排在 forward 之后）、`#48451`（PD 按 attention-DP shard 轮询 prefill 结果并泵 HiCache 存储事件）、`#48231`（Mooncake 在 peer 注册时比对 prefill/decode 的 KV item 长度，不匹配（如 DSA 656 vs 576 B/token）显式拒绝而非静默拷贝错位行）。
5. **TRT-LLM v1.4.0rc3 发布（窗口内） + Qwen3.8 家族落地（强化第 3 条 BREAKING 清理）**：rc3（2026-11-06T21:38:32Z）含模型支持（GLM-5.2 NVFP4 Modeling V2 GB300/SM103 #20524/#20662、MiniMax-M3 NVFP4 低延迟 decode #20306/#20922、Kimi K3 native + NVFP4 KV #20893/#20888、Qwen3.5 dense VLM for ModelExpress #20764、DeepSeek-V4.1-Flash on SM120 #20769）；3 项 BREAKING（移除 LlmRequest 字段 #21036、移除 TensorRT 兼容面 #21041、默认 8 服务前端 #19379）；**Known Issues 明示 DSA 模型（含 GLM-5.2）在 NVFP4 KV cache 下可能产出错误输出（静默正确性！）、Kimi K3 DSpark + MegaMoE CuTe DSL TEP + CUDA graphs 在 GB300 warmup 可能挂起**——与往期静默正确性主线直接呼应。Qwen3.8 家族：`#20714`（PrimTS sparse core backend for Qwen3.8-Flash-Next QSA attention，SM100/103/107 运行时选择）、`#21372`（Qwen3.8-Flash-Next NVFP4 modeling_v2 共享 sm_103 目标，287 文件；当前 head CI 未全绿，记为描述级/needs follow-up）。
6. **量化 / MoE 路由静默正确性（强化第 4/6/7 条）**：SGLang `#44437`（FP4 MLA `move_kv_cache` 只拷贝 `kv_buffer`、不拷 `kv_scale_buffer` → 移动后的 FP4 latent 用目标槽（常是 rejected draft）的 block scales 反量化，**静默错且无报错**、无测试拦截，强化第 4 条）、`#47892`（`--quantization-param-path` 的 KV scales 以 Python float 存储致 aiter/triton 后端启动失败，已改为 device tensor）、vLLM `#57823`（Inductor 产生 padded 物理行 router logits 时原生 top-k 用 `blockIdx.x*num_experts` 定位、**静默改路由**、input-row offset 改 int64 防 32 位回绕，强化第 7 条）、`#64410`（w4afp8 checkpoint 经 load-time translation 落到 W4A8 FP8 MoE 路径，PhalaCloud/GLM-5.3-W4AFP8 GSM8K 0.9113）。
7. **新硬件后端：SGLang MLU (Cambricon) 初始支持（#45149）**：设备管理、CNCL 通信、融合 RMSNorm、CI 测试、安装 extras（`srt_mlu/all_mlu/dev_mlu`）、pin torch-mlu；MLU RMSNorm 显式拒绝 deterministic 模式（确定性推理推迟）。

---

## 2. vLLM：2026-11-06 提交（105 条）—— Kimi K3.1 day-0 支持 + 投机解码正确性簇 + KV-sharing/可扩展 KV cache 正确性 + 量化

### 2.1 `#66030` [Model] Kimi K3.1：模型文件 + NVIDIA specialized decode（split 1/5，模型支持）
- **主题**：Kimi K3.1（`KimiK31ForConditionalSequence`）模型文件与 NVIDIA specialized decode，置于 `vllm/models/kimi_k3`；K3.1 = Kimi-K3 在每 MLA 层带 DSA（DeepSeek sparse attention），含 DSA indexer kernels、packed KDA/DSA hybrid caches、dense/GQA drafts、CuTe DSL DSA/KDA/MLA/MoE/GEMM kernels。
- **内容详情**：5-split 栈（1/5 本 PR 模型文件；2/5 #66028 Rust renderer；3/5 #66037 KV cache/attention 改动以 TP8 服务；4/5 #65685 ROCm/P-D/indexer top-k 后端；5/5 #66027 DCP 优化 opt-in）。本 PR 不碰 `v1/attention`；在 3/5 落地前 K3.1 可 import 但不能服务，Kimi-K3 不受影响。
- **应用场景**：Kimi K3.1 多引擎 day-0 支持（vLLM 侧）。
- **研判**：【事实】PR 给 5-split 拆分与 K3.1=DSA 每 MLA 层的结构。【观点】属「新模型 day-0 支持」协同事件（趋势第 1 条）；在 3/5 落地前仅 import 不服务，运维须等齐栈。**证据等级**官方一手（PR 描述级/多 split 栈）。

### 2.2 `#64986` [Bugfix][Attention] 修 KV-sharing fast prefill 因 host offsets 未同步挂起（KV 正确性，强化第 4/8 条）
- **主题**：修 #64985。异步调度下 `--kv-sharing-fast-prefill` 可能把 D2H 拷贝尚未完成的 `query_start_loc_cpu` 交给 attention metadata builder，trtllm-gen decode kernel 收到垃圾 query length 挂起。
- **内容详情**：现 `query_start_loc_cpu` 始终持最终值；单 logit（无投机）情形用 `torch.arange` 而非拷贝，common path 无新 sync；回归测试 base 2 failed/6 passed → 本 PR 8 passed。E2E：B200、Gemma-4 E4B fine-tune (bf16)、`--kv-sharing-fast-prefill --enable-prefix-caching -cc '{"cudagraph_mode":"PIECEWISE"}'`、MRV2、异步调度；4649 请求切片 base 4/4 挂起 → 本 PR 20s 完成；全量 31649 请求 base 挂起 → 完成。
- **应用场景**：`--kv-sharing-fast-prefill` + 异步调度 + 可扩展 KV cache。
- **研判**：【事实】PR 给根因（D2H 未同步）+ 单测对照 + B200 E2E 对照（挂起→完成）。【观点】属「异步调度下 host/device offset 同步」崩溃型 bug（与往期 CUDA graph replay 生命周期同源），是 #64095 跨层 KV 共享的连带正确性。**证据等级**官方一手（含单测与 E2E 对照）。

### 2.3 `#65554` [Bugfix] 修可扩展 KV cache 未对齐段导致的启动非法访存（KV 正确性，强化第 8 条）
- **主题**：修 FlashMLA sparse prefill + 可扩展 KV cache（#56492）在 DeepSeek-V3.2 on B300 复现的启动非法访存；FlashInfer autotuning 在 warmup 扩展 cache 前跑 FlashMLA，sparse gather 含无效（padding）index 时读越选中的行、可能触及层预留 KV view 内未映射内存。
- **内容详情**：把 #65225 的段对齐扩展到无 RDMA connector 的 cache，并为 block-outermost 布局在分配时提交第二块（Nick Hill 的 follow-up commit）。覆盖 block-compact / layer-outermost / block-outermost 布局，逻辑 KV 容量与 block strides 不变。E2E：DeepSeek-V3.2 (FP8) 8×B300，EP+TP2+DP2+PP2、eager、关前缀缓存，`max_model_len=8192`。
- **应用场景**：FlashMLA sparse prefill + 可扩展 KV cache 的启动健壮性。
- **研判**：【事实】PR 给根因（padding index 读越界）+ 验证配置。【观点】属「可扩展 KV cache 内存映射对齐」正确性（第 8 条），与 #65463 同属可扩展 KV cache 收口。**证据等级**官方一手（含复现与验证，部分截断）。

### 2.4 `#65463` [Bugfix][KV Cache] 可扩展 KV cache 收缩后重新拟合 `--max-model-len auto`（KV 容量自检，强化第 8 条）
- **主题**：可扩展 KV cache（默认 CUDA/ROCm + model runner V2，#56492）下 `--max-model-len auto` 在规划期按排除 CUDA graphs/warmup 的预算拟合；当实测容量装不下拟合长度时启动失败。本 PR 仅对 `auto` 重新拟合到实测容量并记日志。
- **内容详情**：显式 `--max-model-len` 装不下仍失败；`auto` 仅向下搜索，容量足够时保持不变；+24 行（`kv_cache_planning.py`+19、`engine/core.py`+5）+ 单测（66 passed，缺失修复时启动 `ValueError`）。
- **应用场景**：可扩展 KV cache 下 `--max-model-len auto` 启动健壮性。
- **研判**：【观点】属「KV 容量自检/重新拟合」（第 8 条，与 #65413 容量低估同源），降低 auto 长度在 warmup 后装不下导致的启动失败。**证据等级**官方一手（单测级）。

### 2.5 `#65229` [Bugfix][Spec Decode] 避免接受估计器的灾难性抵消（投机解码正确性，强化第 5 条）
- **主题**：#52228 加的自适应验证接受特征在 drafter 高置信时因 fp32 灾难性抵消丢失全部信号——每个比 `1-6e-8` 更自信的 draft 行都报同一接受特征 `40.0`。
- **内容详情**：temp=0.5 源码上 **11.5% 的 draft 位置**置信被毁、替换为常数；原特征定义为 log-odds `logit(q(m*))`，用 `1 - sum(含 max 单位项)` 恢复补数，fp32 下毁灭被测量（无 NaN、无警告、仅 clamp）。本 PR 直接计算补数修复数值问题。
- **应用场景**：无 confidence head 的自适应验证投机解码。
- **研判**：【事实】PR 给数值分析（11.5% 位置被破坏、~23-nat 跨度塌缩到单值）。【观点】属「投机解码数值正确性」（第 5 条）——静默（无报错）地把不同置信塌缩成常数，影响接受决策；直接算补数修复。**证据等级**官方一手（含数值分析，len=17005）。

### 2.6 `#66048` [BugFix] 修投机解码 insert resampled 竞争（投机解码正确性，强化第 5 条）
- **主题**：`_insert_resampled_kernel` 以 `num_warps=1` 运行，避免读写 `num_sampled` 的跨 warp 竞争——该竞争可致投机解码越界 lookup 与致命 CUDA 异常。
- **内容详情**：两次 RTX PRO 6000 Blackwell coredump 同机制：不同 warp 读到 `num_sampled=4/5`，反汇编映射到 `num_sampled` 的 load；环境 vLLM 0.32.0、PyTorch 2.13.0+cu130、Triton 3.7.1、Gemma 4 MTP 4 speculative tokens；main 分支仍含该读写模式。仅改 launch 配置，无性能/质量声称。
- **应用场景**：Gemma 4 MTP 等投机解码的 resampled kernel 竞争。
- **研判**：【事实】PR 给 coredump 分析（两次崩溃同机制、Warp Out of range Address）。【观点】属「投机解码 kernel 竞争」崩溃型 bug（第 5 条）。**证据等级**官方一手（含崩溃分析）。

### 2.7 `#66010` [Sleep Mode] Level 2 仅丢弃参数（特征/正确性，强化第 5 条旁支）
- **主题**：Level 2 释放 GPU 内存但不保留可重建项；此前丢弃模型构建期分配的一切，唤醒后非参数状态丢失。现 `sleep(level=2)` 声明调用方重写的内容、其余原位保留。
- **内容详情**：未修复前 GB200 实测——GEMM-RS/AR peer pointer table 读全零→`CUDA_ERROR_ILLEGAL_ADDRESS`（Kimi-K3 占位权重、DeepSeek-V4.1-Flash + GEMM-RS）；DeepSeek-V4.1 `permute_on_load` 索引→GSM8K 0.90→0.0；MXFP4 TRT-LLM permute cache→gpt-oss-20b GSM8K 0.375→0.0。属直接权重重载工作（#54477, F11）。
- **应用场景**：权重热更新/睡眠唤醒场景。
- **研判**：【事实】PR 给三类非参数状态丢失的逐状态对照（GSM8K 0.90→0.0 / 0.375→0.0 / CUDA_ILLEGAL_ADDRESS）。【观点】属「权重重载后非参数状态」正确性（第 5 条旁支），与社区「llama.cpp Sleep Mode」博客同主题。**证据等级**官方一手（含逐状态对照）。

### 2.8 `#65804` [Spec Decode] standalone draft model prefill 按 decode 处理（投机解码正确性，强化第 5 条）
- **主题**：自 #60989，standalone draft model（`method:"draft_model"`）为 draft prefill 捕获 FULL CUDA graph；该 prefill 是每请求 `num_speculative_tokens+2` 的均匀批，但 `max_decode_query_len()` 返回 `1+num_speculative_tokens`，使每个 spec-as-decode attention builder 的 reorder 阈值少一 token、draft-prefill 被分类为 **prefill**、FULL graph 捕获 prefill 路径。
- **内容详情**：后果——正确性（UNIFORM_BATCH 仅保证均匀 *decode* 批安全，FlashInfer TRTLLM prefill 非 graph-safe，致 GB200 崩溃，#65598 修了该分配但 prefill 路径仍在 UNIFORM_BATCH 之外）；性能（draft prefill 跑 prefill 注意力核虽为 decode 形）。MRV2 上 `max_decode_query_len()` 加 `speculative_config.max_num_new_slots_for_drafting`（standalone draft=1，其余=0）。
- **应用场景**：standalone draft model + FULL CUDA graph 的投机 prefill。
- **研判**：【事实】PR 给根因（分类为 prefill → FULL graph 捕获 prefill 路径）+ 关联 #60989/#65598。【观点】属「投机 draft prefill 分类」正确性（第 5 条）。**证据等级**官方一手（根因级）。

### 2.9 `#57823` [Bugfix][MoE] 尊重 top-k 路由的 padded 行 stride（MoE 路由正确性，强化第 7 条）
- **主题**：修 #56900。Inductor 可产生带 padded 物理行的 router logits（如形状 `[M,60]` stride `(64,1)`），原生 MoE top-k 用 `blockIdx.x*num_experts` 定位每行，stride≠num_experts 时读错 logits、**编译下静默改路由**。
- **内容详情**：传播 `gating_output.stride(0)` 到原生 top-k launcher；padded 行走 unfused softmax/sigmoid + top-k；分配 padded 2 的幂专家数所需 workspace；packed 行走专用路径；回归覆盖 60/64、64/72 逻辑/物理行宽。input-row offset 改 `int64_t` 防 32 位回绕（与 #53131 仅重叠 offset 宽度）。
- **应用场景**：Compiled/torch.compile 下 MoE top-k 路由。
- **研判**：【事实】PR 给根因（padded stride 误定位）+ 回归覆盖 + int64 防回绕。【观点】属「MoE 路由静默错」（第 7 条）——编译下静默改专家路由、无报错，强铁证。**证据等级**官方一手（含回归覆盖；部分截断）。

### 2.10 `#65746` [GLM 5.3 Perf] sm100 fp8 moe tma 扩展，1.03~1.60× kernel 性能（性能，强化第 6 条）
- **主题**：#63820 follow-up，sm100 GLM-5.3 本地 FP8 MoE 的 Triton pointer 与 B-TMA 加载对比 kernel 性能 1.03~1.60×。
- **内容详情**：附 AI 生成对比脚本（覆盖 TOKENS 1–8192 的本地 MoE kernel 计时，非模型服务）。单测覆盖 ACC。
- **应用场景**：GLM-5.3 在 sm100 的 FP8 MoE kernel。
- **研判**：【观点】属「GLM-5.3 FP8 MoE kernel」性能（第 6 条），1.03~1.60× 为 kernel 级数字、非端到端；记为官方一手（perf 脚本级，未给 E2E）。**证据等级**官方一手（性能脚本级）。

### 2.11 `#64410` [Quantization] W4A8 FP8 MoE 路径支持 w4afp8 checkpoint（量化，强化第 6 条）
- **主题**：新增 `w4afp8` 量化方法，使 SGLang 式 W4AFP8 MoE checkpoint（`PhalaCloud/GLM-5.3-W4AFP8` 等）落到 #58427 的 compressed-tensors W4A8 FP8 MoE 路径；仅 load-time translation，无 kernel/model/`fused_moe` 改动。
- **内容详情**：经 `QuantizationConfig.get_checkpoint_weight_mapper()` 翻译 checkpoint 布局；覆盖 Hub 两种 input_scale 命名与两种 ignore-list key；routed experts 走 `CompressedTensorsW4A8Fp8MoEMethod`（Humming/CUTLASS W4A8）、dense 走原生 FP8 block-128。
- **应用场景**：GLM-5.3-W4AFP8 等 SGLang 式 checkpoint 在 vLLM 服务。
- **研判**：【事实】PR 给等价性验证（2×H200 TP2+EP2，贪婪 4 prompts×24，新/旧/新 一致、max abs logprob diff 见原文）。【观点】属「W4AFP8→W4A8 量化互通」（第 6 条）。**证据等级**官方一手（含 GSM8K/logprob 对照）。

### 2.12 本期其余 vLLM 要点（标题/正文级，未逐条深采）
- **Kimi K3.1 / 模型**：`#66030`（见 2.1）、`#66028`（Rust renderer）、`#65725`（Kimi K3.1-Pro TPU via vllm-torchtpu）、`#65826`/`#65827`/`#65828`（LFM3-Text/VL/Audio）、`#66097`（Qwen3-ASR 用上游 config）。
- **投机解码簇**：`#63795`（DSpark sliding-window MLA draft layers）、`#65270`（MRV2 mixed draft-model warmup token budget）、`#65710`（Structured Output EOS in speculative Guidance batches）、`#66048`/`#65804`/`#65229`/`#66010`（见 2.5–2.8）。
- **KV / 可扩展 KV cache**：`#64986`/`#65554`/`#65463`（见 2.2–2.4）、`#65667`（MRV2 从 unkeyed full blocks 读 routed experts）、`#65831`（KV block 分配/缓存每 block CPU 成本 −45~50%，140K token 请求，scheduler-only 时序）、`#65554`/`#65249`（ROCm DSv4.1 prefix-cache SWA replay on gfx942）。
- **ROCm / 异构**：`#61448`（Muse Glimmer AMD fusion）、`#62718`/`#63365`（ROCm 构建提速）、`#63890`（AITER all-gather 小 TP）、`#65249`/`#65267`/`#65478`/`#65481`/`#65530`（MiniMax-M3.1 ROCm perf 调优）、`#65517`（ROCm MoE tkw1 拒 count-free expert metadata）、`#65722`（AITER FA 不跳过 1-token prefill）、`#65937`（Bump AITER v0.1.28.post1）、`#65971`（gfx942 MLA Triton verify follow-up）。
- **量化 / 模型加载**：`#64410`（见 2.11）、`#60438`（nvcc<12.9 时退 DeepGEMM）、`#64352`（Direct reload 重置 MoE expert padding）、`#60047`（注册 aten::mm out_dtype 重载保 batch invariance）、`#66012`（offline 模式不重试 safetensors metadata）、`#65955`（解析量化 config 时避免 import humming）、`#65930`（Upgrade FlashInfer 0.7.3）。
- **安全 / 前端**：`#63574`（抬高 patched runtime/test 依赖下限）、`#60308`（Pillow bomb check 拒图返回 400）、`#65697`（Anthropic count_tokens 计超 max_model_len 的 prompt）、`#66016`（CUDA arch 不向更低 minor 回退）。

---

## 3. SGLang：2026-11-06 提交（88 条）—— Kimi K3.1 DSA/MegaMoE + HiCache/PD 健壮性 + FP4 MLA KV 静默错误 + MLU 新后端

### 3.1 `#48721` [Feat] kimi-k3.1：DSA decode 上下文并行运行时（模型支持，强化趋势第 1 条）
- **主题**：port #48414。Kimi-K3.1 Day-0 分支在 decode 上下文并行下把 MLA KV 与 indexer 的 index-K cache 按 token 分片，每 rank 选 local top-k 再合并为 global top-k，decode attend 各 rank 分片并按 LSE 合并 partials。main 上 K3.1 仍拒 `--dcp-size>1`，本 PR 在 #48501/#48588 内核上 port 该运行时，使 `--dcp-size>1` 服务普通 prefill/decode。
- **内容详情**：相对 raw port——merge 用 `DcpTopKMerge.create`（#48588）、top-k 取 `dcp_size/dcp_rank/deterministic`（#48501）、FlashMLA schedule=`windowed`、经 `flash_mla_interface` 与 backend `_call_*` hook；丢弃原生 head FlashMLA 分支与 prefix-only candidate gather；DCP 下 K3.1 override 拒投机解码/PD/HiCache/HiSparse/prefill CP。
- **应用场景**：Kimi-K3.1 在 `--dcp-size>1` 的 decode 上下文并行。
- **研判**：【事实】PR 给 port 范围与 DCP 下拒绝集合。【观点】属「Kimi K3.1 day-0 支持」协同事件（趋势第 1 条），DCP 运行时在 main 落地。**证据等级**官方一手（port/内核级）。

### 3.2 `#48714` [Feat] kimi-k3.1：opt-in trtllm DSA backend（模型支持/性能）
- **主题**：port #48414 的 opt-in trtllm DSA backend（radixark/sglang-k31#31），独立于 prefill-CP 栈。`--dsa-prefill-backend`/`--dsa-decode-backend trtllm`，默认 `flashmla_kv`、ROCm 保持 `triton`；两阶段读同一 KV pool、PD peer 交换之，故须匹配；trtllm 暂拒 `--dcp-size>1`。
- **内容详情**：Mooncake 拒非 DCP 的 MLA P/D 对，其每页 KV item 长度不同（flashmla_kv 656 B vs trtllm 576 B/token）而非静默拷贝错位行。TP8 B300，trtllm vs flashmla_kv：gsm8k 0.953/0.949；passkey 32K/128K 均 5/5；TPOT bs1/16/64 8.45/15.09/21.12 vs 8.60/15.26/21.30 ms；TTFT −6~−12%；KV 容量 +11%。
- **应用场景**：Kimi-K3.1 的 trtllm DSA backend 选项。
- **研判**：【事实】PR 给 TP8 B300 精度（0.953/0.949）+ speed（TTFT −6~−12%、KV +11%）+ Mooncake 长度不匹配显式拒绝。【观点】属「Kimi K3.1 DSA backend」模型支持/性能（趋势第 1 条）。**证据等级**官方一手（含精度/speed 对照）。

### 3.3 `#48586` [Perf] kimi-k3：decode MoE 路由 1–256 行融合为一 kernel（性能，强化第 7 条）
- **主题**：port #48414。每个 Kimi-K3/K3.1 decode MoE 层在 router 与 routed experts 间跑串行小 kernel 链（`route_quant_fused` → padded-row mask → trtllm-gen routing kernel）；本 PR 以一个 1–256 行 kernel 替之，routed MoE 仅跑 GEMM。
- **内容详情**：K3 与 K3.1 共享该 MoE（896 专家、top-16 sigmoid+bias 无 group、3584 latent、MXFP4 trtllm-gen SiTU），故命名为 `kimi_k3_fused_routing`；移除 kill switch `SGLANG_K31_FUSED_ROUTING`，无法服务的情况回退 unfused。
- **应用场景**：Kimi-K3/K3.1 decode MoE 路由融合。
- **研判**：【观点】属「MoE 路由融合」（第 7 条性能），把串行小 kernel 链合成一 kernel；记为官方一手（port/内核级）。**证据等级**官方一手（port/内核级，len=9304）。

### 3.4 `#44437` [Fix] mem_cache：在 `move_kv_cache` 中随 KV 移动 FP4 MLA block scales（KV 静默正确性，强化第 4/6 条）
- **主题**：投机解码用 `move_kv_cache` 重定位 KV 行（verify 后移动 accepted draft、page_size>1 时把 prefix tail page 拷到各 branch）。`MLATokenToKVPoolFP4` 继承的 `move_kv_cache` 只拷 `kv_buffer` 行，移动后的 FP4 latent 用目标槽（常为 rejected draft）的 block scales 反量化，**无任何报错**。
- **内容详情**：其余量化池（MHA/MHA-MXFP8/DSA）已随数据移 scales；本 PR 在 latent move 后同索引拷 `kv_scale_buffer` 行；CPU `MLATokenToKVPoolFP4` 测试移动两槽验证 data+scales 到每层。
- **应用场景**：FP4 MLA + 投机解码（EAGLE topk>1 / NGRAM）的 KV 行移动。
- **研判**：【事实】PR 给根因（scales 不随 KV 移动→用目标槽 scales 静默反量化错）+ CPU 测试。【观点】属「FP4 KV scale 漏接」静默错误（第 4/6 条）——往期 SGLang #46928/#45486 为 AMD aiter/CTA fp8 静默错，本条为 FP4 MLA scales 不随 KV 移动的静默错铁证。**证据等级**官方一手（根因级，静默错误）。

### 3.5 `#47892` [Fix] quantization：`--quantization-param-path` 的 KV scales 按 checkpoint 方式存储（量化 KV 正确性，强化第 4/6 条）
- **主题**：`--kv-cache-dtype fp8_e4m3 --quantization-param-path <json>` 启动失败——aiter 后端 `reshape_and_cache_flash` 报 `Unsupported ptr type ... float32 in tl.load`，triton 后端 `decode_attention_fwd` 报 `TypeError: float * None`。根因：`load_kv_cache_scales` 把 Python float 赋给 `attn.k_scale/v_scale`、`k_scale_float/v_scale_float=None`，而 attention 后端期望 `BaseKVCacheMethod` 存的 device tensor + host float。
- **内容详情**：`load_model_utils.load_kv_cache_scales` 在加载后跑 `_kv_cache_scales_to_tensors`——每个 scale 为 float 的 `RadixAttention` 获 0 维 float32 device tensor，float 保留在 `k_scale_float/v_scale_float`；`ModelRunner` 传 `device=self.device`。含 CPU 测试。
- **应用场景**：fp8 KV cache + 外部量化参数 JSON 的服务启动。
- **研判**：【事实】PR 给根因（float vs device tensor）+ 两类后端报错 + 修复。【观点】属「KV scale 存储格式」静默/启动失败（第 4/6 条）。**证据等级**官方一手（根因级）。

### 3.6 `#48522` [Fix] hicache：在匹配键上限处停止 L3 prefetch（HiCache 健壮性，强化第 5/8 条）
- **主题**：分层 cache + L3 后端下，prefix 已存仍可能在成功 prefetch 后 `#cached-token: 0`——预取数据已付费（RDMA 读）却被静默丢弃、整 prompt 重算。
- **内容详情**：hybrid SWA 模型（DeepSeek-V4 page_size=256、sliding_window=128、swa_gates_prefix_match=True）当 L3 prefetch 跨度越过 radix 匹配键时发生：`Match key` 把键 cap 在 `L - swa_reprefill_tail_tokens()`，页对齐结束于 `E_match`；L3 prefetch 用 `L-1`，页对齐结束于 `E_pre`；当 `L mod P=r` 且 `1<=r<=W-1`（P=256/W=128 时约 **49.6% 的 prompt 长度**）`E_pre>E_match` 且该页在 L3 中，prefetch 越过 `E_match` 一页后 matcher 停在 `E_match`——边界节点带全 KV host 值但无 SWA host 值。修复：L3 prefetch 停在匹配键。
- **应用场景**：HySWA 模型 + HiCache + L3 后端的 prefix 缓存命中。
- **研判**：【事实】PR 给根因（匹配键 vs prefetch 跨度错位，49.6% 概率）+ 静默丢弃现象。【观点】属「HiCache L3 prefetch 静默丢弃」（第 5/8 条）——已付费预取被静默丢、整 prompt 重算，强铁证。**证据等级**官方一手（根因级，含 49.6% 概率）。

### 3.7 `#48231` [Fix] disaggregation：Mooncake 注册时拒 KV page stride 不同的 decode peer（PD 健壮性，强化第 5/8 条）
- **主题**：修 #48230。Mooncake sender 用 prefill 的 item_len 在 `dst_ptr + page*item_len` 写 decode 页；当两角色分配不同 KV 布局，每次传输静默损坏 decode KV、池尾页越层 buffer（"Requested address not found"）。触发例：DSA 模型 prefill/decode 不同 DSA backend（656 vs 576 B/token）；或两角色不同 `--kv-cache-dtype`（#19355）。
- **内容详情**：MooncakeKVManager 在 peer 注册时比对 prefill/decode 的 KV item 长度（sender 配对方式）；MLA/layer-id pool 复用 PP-aware `_state_stride_mismatch`；plain MHA pool 同 attn TP 经 `get_mha_kv_ptrs_with_pp` 逐层比 K/V。不匹配时 peer 进 `state_layout_rejections`，每请求失败带明确原因（两长度、page size、可能成因），不写任何字节。
- **应用场景**：Mooncake 解聚传输的 KV 布局一致性。
- **研判**：【事实】PR 给根因（item_len 错位→静默损坏）+ 显式拒绝机制。【观点】属「PD-KV 布局不一致」由静默损坏转为显式拒绝（第 5/8 条），与往期 PD-KV 正确性主线同源。**证据等级**官方一手（根因级）。

### 3.8 `#45149` [Feat][MLU] 初始支持 Cambricon MLU（新硬件后端）
- **主题**：加 Cambricon MLU 初始支持——平台检测与设备管理、CNCL 通信、融合 RMSNorm + residual、CI 测试、安装 extras（`srt_mlu/all_mlu/dev_mlu`）、pin torch-mlu/torch-mlu-ops/Triton 版本；MLU RMSNorm 显式拒 deterministic 模式（确定性推理推迟）。
- **内容详情**：复用通用 all-reduce/all-gather 路径（CNCL + MLU custom-op）；RMSNorm 支持 1D/高维/转置、原生 fallback；含 `--hw mlu` 测试注册、单设备 RMSNorm + 双设备通信套件（eager 与 torch.compile(fullgraph=True) collectives）、`/rerun-test` MLU 识别、CI 覆盖与组件版本记录。
- **应用场景**：SGLang 在 Cambricon MLU 硬件的服务。
- **研判**：【观点】属「新硬件后端」（国产 MLU），扩 SGLang 异构支持版图；记为官方一手（新后端级，未给精度/性能数字）。**证据等级**官方一手（新后端级）。

### 3.9 `#48971` [Perf] dsv4：wo_b 与 Engram KV MXFP8 GEMM 在 CC 10.7 >8K 行路由到 cuDNN（量化性能，强化第 10 条）
- **主题**：workaround（待 flashinfer #9020 落地后删表）。CC 10.7 上 DeepSeek-V4.1 的 dense MXFP8 linears 经 FlashInfer CuTe-DSL kernel，但 FlashInfer "auto" 按优先级选单后端、cuDNN 的 MXFP8 GEMM 从未被计时；对两 prefill 形状 cuDNN 更快且 bitwise 相等。
- **内容详情**：`flashinfer_mxfp8_blockscaled_linear` 在 device=CC10.7、M>=8192、(N,K) 为 `wo_b (N=5120,K=8192)` 或 `Engram KV (N=25600,K=6144)` 时以 `backend="cudnn"` 调用；需 cuDNN>=9.23.1 支持 override-shape graphs。无 scale 处理变更。
- **应用场景**：DSv4.1 在 CC10.7 的 MXFP8 线性（wo_b / Engram KV）。
- **研判**：【观点】属「MXFP8 GEMM 后端选择」性能（第 10 条），cuDNN 对这些形状更快且 bitwise 相等；记为官方一手（workaround/perf 级）。**证据等级**官方一手（workaround/perf 级）。

### 3.10 `#48751` [Fix] eplb：把 draft forwards 排除出 target expert 窗口（MoE 路由正确性，强化第 7 条）
- **主题**：闭 #48603。target 与 draft runner 共享一个 expert-distribution recorder；draft forward 仍进入 `with_forward_pass()`、reset 计数、append 空记录（即便 draft model 禁用了计数 hook），这些空记录占据 EPLB 用的 target 统计窗口。
- **内容详情**：`ModelRunner.forward` 在 `is_draft_worker` 时于 `with_forward_pass()` 前进入 `disable_this_region()`；所有权为 runner 身份非 ForwardMode（DSpark draft batch 用 TARGET_VERIFY）。CUDA_VISIBLE_DEVICES=99 SGLANG_IS_IN_CI=true 下 `test_expert_distribution_disabled_pass.py` 7 passed on H200。
- **应用场景**：EPLB 统计在投机解码（含 DSpark）下的准确性。
- **研判**：【事实】PR 给根因（空 draft 记录污染 EPLB 窗口）+ H200 7 passed。【观点】属「EPLB/路由统计」正确性（第 7 条），draft 空记录污染 target 统计。**证据等级**官方一手（含测试）。

### 3.11 本期其余 SGLang 要点（标题/正文级）
- **Kimi K3.1 / 模型**：`#48721`/`#48714`/`#48713`（见 3.1–3.2、shared-expert TP 跨 interleave prefill CP）、`#48547`（ERNIE 5.5 Flash）、`#48586`（见 3.3）、`#48723`（mega_moe 为 SM107 decode+target-verify 预留 18 SMs）、`#48733`（dsv41 Index-Q RoPE + fp4 pack 每 fp4 block 一线程）、`#48941`（layernorm rmsnorm_mxfp8 2 warps / hc_combine_norm）。
- **HiCache / PD / mem_cache**：`#48522`/`#48231`（见 3.6–3.7）、`#48693`（PD 传输等待时不计 staged-prefetch 延迟）、`#48450`（lazy unified-pool 页复用排在 forward 后）、`#48496`（recycled unified-page writers 排在 forward 后）、`#48451`（PD 按 attention-DP shard 轮询 prefill 结果 + 泵 HiCache 存储事件）、`#48447`（dp-attention 把 rank 的 ShutdownReq 经 MLP-sync gather 带到 DP peers）、`#48458`（disaggregation 把非最终 KV 发送保留在 protected prefix 内）。
- **调度 / 稳定性**：`#48757`（scheduler abort beam groups on retract pause、补三处 request-drop 缺口）、`#48654`（scheduler 每 PP stage 匹配到达前缀）、`#48762`（SchedulerInvariantChecker 贯穿每状态检查）、`#46200`（SWA KV pool 在 PP 下自第一层起）、`#48619`/`#48692`/`#48722`（CI/AMD 多项）。
- **性能**：`#46597`（minimax-sparse decode indexer CP on Hopper TP4/TP8）、`#47354`（glm5 小 M BF16 GEMM 路由 FlashInfer warp split-K + PDL on SM107）、`#48174`（重落地 JIT moe_finalize_fuse_shared 在 PDL wait 后）、`#48761`（mhc DeepGEMM prenorm slices 减半当 grid 溢出一波）、`#48784`（eplb 静态 per-rank dispatch map 向量化）、`#47964`（AMD JIT custom all-reduce v2 on ROCm）、`#48468`（Bump FlashInfer 0.7.3、NVLink one-sided all-to-all 迁 flashinfer.moe_ep）。
- **依赖清理**：`#48829`（dropped flash-attn-4 默认依赖）、`#48853`/`#48789`（移除 kernels-hub FA3 loader / vendored FlashMLA 构建）、`#48828`（Rust `PoolName` 仅留读池名）。

---

## 4. TensorRT-LLM：2026-11-06 提交（44 条）—— v1.4.0rc3 发布（窗口内）+ Qwen3.8 家族落地 + BREAKING 清理 + KV 卸载/Helix

### 4.1 `v1.4.0rc3` [Release] 窗口内发布（2026-11-06T21:38:32Z，prerelease）—— 头号事件
- **主题**：v1.4 正式发布前最后一档重要候选；含模型支持、3 项 BREAKING、特性与 Known Issues。
- **内容详情（Highlights）**：
  - **模型支持**：Muse Spark Open #21042/#20643、LTX-2.3 text-to-video #18111、GLM-5.2 NVFP4 Modeling V2 GB300/SM103 #20524/#20662、MiniMax-M3 NVFP4 低延迟 decode #20306/#20922、Kimi K3 native + NVFP4 KV #20893/#20888、MiniMax-H3 CPU offload + TP #21301/#19807/#20176、Qwen3.5 dense VLM for ModelExpress #20764、DeepSeek-V4.1-Flash on SM120 #20769。
  - **API BREAKING**：移除 LlmRequest 未用字段 #21036、移除 obsolete TensorRT-backend 兼容面 #21041、默认 8 服务前端（单前端 fallback）#19379。
  - **Feature**：统一 KV cache manager 基础 + sparse attention arenas + host-backed pools #20178/#21129、aggregated Helix KV sizing/preemption/phase-dependent MLA heads #20759/#21122/#20833、attention DP connectors + per-view KV-head transfers + owner-only Helix MLA writes #19369/#20709/#21245、native sparse MLA for DeepSeek-V4 #20209、native FP8 MoE LoRA #18555/#19646、DSpark ragged graphs + FlashInfer ring-buffer replay for GDN MTP #20072/#20008。
  - **Known Issues（静默正确性主线呼应）**：DSA 模型（含 GLM-5.2）在 NVFP4 KV cache 下可能产出**错误输出**；Kimi K3 DSpark + MegaMoE CuTe DSL TEP + CUDA graphs 在 GB300 warmup 可能**挂起**；DeepSeek-V4-Pro 初始化在 MPI worker 复用后可能 OOM；TP serving + RPC 在跨线程复用 MPI worker pool 时初始化可能挂起。
- **应用场景**：TRT-LLM v1.4 正式发布候选。
- **研判**：【事实】PR/release 给 rc3 模型支持、BREAKING、Feature 与 Known Issues 清单。【观点】rc3 把 DSA+NVFP4 KV 静默错误、Kimi K3 DSpark warmup 挂起列为 Known Issues，与往期「静默正确性/默认值风险」主线直接呼应；升级须先核对 Known Issues。**证据等级**官方一手（release notes 级）。

### 4.2 `#20703` [refactor] BREAKING：checkpoint formats 用字符串枚举（BREAKING/配置，强化第 3 条）
- **主题**：用字符串枚举 `CheckpointFormat`（HF/MX/mistral 等）替代内置 checkpoint formats 的字符串；自定义格式仍可用字符串；serve schemas 同步；移除字符串值的死代码；简化 checkpoint I/O policy 解析（合并 fallback 到 `_checkpoint_io_incompatibility_reason()`）；`_construct_checkpoint_loader()` 公开为 `construct_checkpoint_loader()`。
- **内容详情**：3 commits（枚举 / 要求 LoadFormat / 简化 policy）；1 test added、12 tests changed。
- **应用场景**：TRT-LLM checkpoint 格式/IO 配置。
- **研判**：【观点】属「配置类型化 BREAKING」（第 3 条）——与 vLLM 侧 #20498 同属 v1.4 配置面清理。**证据等级**官方一手（BREAKING 重构级）。

### 4.3 `#20714` [feat] Qwen3.8-Flash-Next QSA attention 的 PrimTS sparse core backend（模型支持/性能）
- **主题**：Qwen3.8-Flash-Next QSA attention 的 PrimTS kernels，运行时选择（同 glm_kpool/deepseek_v4）：SM100/103/107 输入合格走 PrimTS，否则 Triton、小 decode batch 也走 Triton。
- **内容详情**：sparse core（flashinfer QToken-KvBlock-Sparse kernel，flashinfer #5351/v0.20.0，vendored aa709061）按 geometry/layer 缓存 plan、直接消费 block IDs、每调用 FP8 Q 量化且 scale 折入 sm_scale；indexer scorer（tcgen05 task-scheduled 替 triton_qsa_paged_index_scores）；main 锁 flashinfer-python==0.7.2 prims_ts vendored，metadata 侧 pieces 等下个 pin；小 batch decode PrimTS 在 CUDA graphs 下 1 行平 Triton、2 行起领先。
- **应用场景**：Qwen3.8-Flash-Next QSA attention 的稀疏核心。
- **研判**：【观点】属「Qwen3.8 家族落地」（趋势第 6 条）+ 稀疏注意力内核；部分 metadata 侧等下个 FlashInfer pin，记为官方一手（含 sparse core 表，部分截断）。**证据等级**官方一手（含 sparse core 表，部分截断）。

### 4.4 `#21372` [feat] Qwen3.8-Flash-Next NVFP4 modeling_v2 共享 sm_103 目标（模型支持，描述级）
- **主题**：把 TP1-only 的 Qwen3.8-Flash-Next NVFP4 target 替换为 TP1/TP4/TEP4/DEP4/DEP2 共享 sm_103 目标；加 rank-aware 权重加载、pool 校验、routing admission、attention/SSM 路径、expert-window 执行、vision 支持；把共享算子/测试从 TP1 树迁到共享 qwen4_exp 包。
- **内容详情**：改 287 文件；移除多个 TP1-only 测试与 5 个共享 Qwen4 测试。Dev review 提示核对 rank 特定权重切片、cache/recurrent-state offsets、layout/capacity 边界、routing rejection；supplied history 报更早 commit CI 通过、current head 无 completed checks；QA 判结果作者报告、未独立验证。
- **应用场景**：Qwen3.8-Flash-Next 在多 TP 配置下的 NVFP4 服务。
- **研判**：【事实】PR 给范围（287 文件、共享 sm_103 目标、rank-aware 加载）。【观点】属「Qwen3.8 家族落地」（趋势第 6 条）；current-head CI 未全绿、QA 未独立验证，记为**描述级/needs follow-up**。**证据等级**官方一手（描述级，CI 未全绿/未独立验证）。

### 4.5 `#21400` [perf] Kimi-K3 MoE decode front：routing + moe_sort 一 launch + NVFP4 latent_down epilogue（性能，描述级）
- **主题**：Kimi K3 decode 路径改 router/latent-down 发射顺序、shared-expert 调度、residual handoff、SM reservation；加 router-logit 与 attention-residual 的 CuTe DSL kernels；Low-M GEMM 加 gated/sigmoid/prequantized-input/NVFP4-output 变体。
- **内容详情**：objectives 报 Rubin/GB300 性能增益、item 1–4 bitwise、item 5 SM107 latent FP32 rounding 变更、GSM8K 95.95 vs 95.98（作者报告、未独立验证）；4 个 SM107 failures 也在 main（未独立验证）；QA 报 CI 混合（含 aborted/failed runs）。
- **应用场景**：Kimi K3 在 Rubin/GB300 的 MoE decode 性能。
- **研判**：【观点】属「Kimi K3 MoE decode 融合」性能（趋势第 1 条旁支）；GSM8K 与 bitwise 为作者报告、未独立验证，记为**描述级/作者报告**。**证据等级**官方一手（性能/精度作者报告，未独立验证）。

### 4.6 `#21596` [feat] DSpark confidence scheduling 的设备端重排序（默认开，强化第 5 条）
- **主题**：加 DSpark confidence scheduling（#20078）的设备端重排序，无 attention DP 也可用，默认开。`enable_fused_confidence_scheduler`（prototype，默认 True）传 `DSparkVerifyPlanner(device_windows=...)`；仅 `enable_confidence_scheduling` 开时生效，关 overlap scheduler 时回退 host 排序。
- **内容详情**：confidence scheduling 从上一 step 置信副本在 host 选 verifier budget（host 不等 GPU）；device 重排序保留 budget 但按被验证 block 的置信重排 GPU 上验证的 drafted 位置。无 attention DP 时每 TP rank 验证同批、各 rank 从自身置信选 verify 行。
- **应用场景**：DSpark 投机解码的置信调度。
- **研判**：【观点】属「投机解码置信调度」性能（第 5 条旁支），默认开；记为官方一手（config 级）。**证据等级**官方一手（config 级）。

### 4.7 `#21380` [feat] Staircase：Rubin GLM 优化低延迟（Conc up to 56，模型/内核）
- **主题**：把 GLM-5.2-NVFP4 Rubin target 从 TP16 扩到 TP4/TP8/TP16 + 4 种 batch/draft 配置；重命名为 `glm_5_2_nvfp4__sm_107__low_latency`，C++ kernels 与 Torch bindings 迁到 `glmMoeDsaRubin`；一个 GLM router 继续选 Blackwell/Rubin target；shared router registry/family init/README/CI discovery 不变。
- **内容详情**：Blackwell 保留现有实现与精度覆盖；Rubin 模型/kernel 更新覆盖小 batch MoE routing/G…（len=16581，截断）。
- **应用场景**：GLM-5.2-NVFP4 在 Rubin 的低延迟服务。
- **研判**：【观点】属「GLM Rubin 低延迟」模型/内核支持；记为官方一手（模型/内核级，部分截断）。**证据等级**官方一手（模型/内核级）。

### 4.8 `#20664` [feat] DSpark Unified KVCache standalone（KV 正确性/卸载，强化第 8 条）
- **主题**：为 DFlash 与 standalone DSpark 加共享 target/draft KV-cache 分配（block-width-aware 分配 + cache-cost 估计）、disaggregated peer 兼容性检查、launcher 选择逻辑、MLA page-index rebasing（per-layer V2 caches）；加公共 `get_draft_block_width` 导出与 `shared_draft_block_width` 参数。
- **内容详情**：改 21 测试文件 + 新 `test_dflash_ctx_block_tables.py`；覆盖 shared-cache 资格、draft-block 分配、cache 估计、DFlash block-table、DSpark block widths、MLA page indices、high-layer pool head counts。CI 状态混合（含 aborted/failed runs）。
- **应用场景**：DFlash/standalone DSpark 的共享 KV cache。
- **研判**：【观点】属「KV 共享/卸载」正确性（第 8 条）；CI 混合，记为**描述级（CI 混合）**。**证据等级**官方一手（描述级，CI 混合）。

### 4.9 `#21455` [feat] aggregated Helix 在主机层挂起（KV 卸载/主机层，强化第 8 条）
- **主题**：KV 压力下 aggregated Helix 把请求挂起到 host tier，host tier 满后以 recompute-pause 为 fallback；disaggregated Helix 不变（生成 server 不能重放 prefill，仍 raise）。行为变更：host tier 对 aggregated Helix 默认开、按自动 host tier 定大小——既有部署将用主机内存、压力下调度可能不同；`kv_cache_config.host_cache_size:0` 保 GPU-only。
- **内容详情**：scheduler——Helix 下 generation 路径仅当存在低于 GPU 的 tier 时才驱逐并自挂起；context 路径在 decode 请求持 GPU 页时等待，否则 recompute-pause 同 pass 内更晚启动的请求；绝不取同 pass 已挂起/recompute-pause 的请求；mid-prefill 挂起请求无法恢复页时同做。retention ranges 透传不变。
- **应用场景**：aggregated Helix 在 KV 压力下的主机层卸载。
- **研判**：【观点】属「Helix 主机层卸载」（第 8 条），默认开 host tier 是默认值风险（第 3 条）的旁支——既有部署行为可能变；记为官方一手（KV 卸载/主机层 级）。**证据等级**官方一手（KV 卸载/主机层 级）。

### 4.10 本期其余 TRT-LLM 要点（标题/正文级）
- **模型 / Qwen3.8**：`#20714`/`#21372`（见 4.3–4.4）、`#21012`（Kimi-K3 context server 的 Dense DWDP + balanced DWDP expert partition）、`#21301`/`#19807`/`#20176`（MiniMax-H3 CPU offload + TP）。
- **Helix / KV**：`#21455`/`#20664`（见 4.9–4.8）、`#21428`（Merge PoolingRunner 进 EncoderRunner）、`#20759`/`#21122`/`#20833`（aggregated Helix KV sizing/preemption/phase-dependent MLA heads）、`#19369`/`#20709`/`#21245`（attention DP connectors / per-view KV-head transfers / owner-only Helix MLA writes）。
- **DSpark / 投机**：`#21596`（见 4.6）、`#20072`/`#20008`（DSpark ragged graphs / FlashInfer ring-buffer replay for GDN MTP）、`#20005`（VisualGen dense FP8 attention + fused QK norm/RoPE/FP8 for Rubin）。
- **基础设施 / waive**：`#21710`（移除 DeepSeek R1 多节点 perf 测试）、`#21730`/`#21772`/`#21765`/`#21771`（DGX_H100/ MiniMax M3 B300 / main post-merge 3056 / pre-merge 68497 多项 waive）、`#21774`（Unwaive #21597 修的 mixed-precision disagg 测试）、`#21776`/`#21782`/`#21783`（CBTS 测试范围 / CPU 测试选择 / blossom-ci 授权用户）。
- **BREAKING / refactor**：`#20703`（见 4.2）、`#21024`（Component-local C++ Tests Part 5）、`#21375`（transport typed Python request inputs）。

---

## 5. llama.cpp：2026-11-06 提交（17 条）—— Kimi-K3.1-Pro 模型支持 + MoE-cache LRFU 混合 + 安全/Hexagon/OpenCL

### 5.1 `#31779` [model] 加 Kimi-K3.1-Pro 文本模型（模型支持，强化趋势第 1 条）
- **主题**：K3.1 复用 K3 大部分，仅把全注意力 MLA 层替换为 DSA；复用 build_attn 的 top-k mask 实现；模型有时带 -Pro 后缀（HF 名）但不总带（config.json 的 arch），代码用 non-pro 后缀。
- **内容详情**：已测 BF16/MXFP4 转换 + q2/q2/q3 gate/up/down 量化无问题；代码审查用 llama.cpp 与 DeepSeek Harness 三 agent 并行（默认 -np 4、unified KV）各 ~250K seq len、多 tool calls 无错误；AI 使用披露 yes；Closes #31778。
- **应用场景**：Kimi-K3.1-Pro 在 llama.cpp 的服务（day-0 协同）。
- **研判**：【事实】PR 给结构（DSA 替全注意力 MLA）+ 转换/量化测试。【观点】属「Kimi K3.1 day-0 支持」协同事件（趋势第 1 条）在 llama.cpp 落地。**证据等级**官方一手（模型支持级，描述简）。

### 5.2 `#31233` [moe-cache] LRFU/MQ-like 混合缓存 ClassLRU（MoE 缓存性能，强化第 8 条）
- **主题**：用 8 类（MQ 启发）带衰减 CRF（Combined Recency+Frequency）分数的缓存替换 SLRU，优先保留高频使用专家。
- **内容详情**：新 `CRF = old*2^(-elapsed_tokens/128)+1`，half-life 128 token 对应最高类 ID 7（8 类覆盖全程）；warm cache 下 QFN **+6% decode**、GLM **+7% decode** vs SLRU，uploads 少 8%；启用 llama_moe_cache 的 auto 模式。
- **应用场景**：llama.cpp MoE 专家缓存淘汰策略。
- **研判**：【事实】PR 给算法（8 类 CRF）+ 逐模型 +6%/+7% decode、−8% uploads 对照。【观点】属「MoE 专家缓存」性能/复用（第 8 条），LRFU 比 SLRU 更稳地保留高频专家。**证据等级**官方一手（含逐模型对照）。

### 5.3 `#31730` [common] 把 HF token 限制到 huggingface.co 主机（安全，强化第 3 条旁支）
- **主题**：此前 `export HF_TOKEN` 足以把 token 发往任何 server；现限制到 huggingface.co 主机，`MODEL_ENDPOINT` 作为显式信任决策仍可用。
- **内容详情**：安全加固——避免 token 泄漏到非 HF 服务器；AI 使用披露未填（按 AGENTS.md 限制，记为描述级）。
- **应用场景**：llama.cpp 的 HF_TOKEN 安全处理。
- **研判**：【观点】属「凭据安全」（第 3 条旁支），与 vLLM #63574 抬高依赖下限同属安全加固；记为官方一手（安全 级）。**证据等级**官方一手（安全 级）。

### 5.4 `#31758` [opencl] q6_k 多列 tiled GEMV 用于投机验证（OpenCL 性能/投机，强化第 5 条）
- **主题**：加 q6_k 多列（2–4 列）变体用于投机验证，含部分重构。
- **内容详情**：`llama-speculative-simple` 下——Qwen3.5-4B-Q4_K_M：master decode 22.532 t/s → PR 32.270 t/s；Qwen3.5-9B-Q4_K_M：15.628 → 23.116 t/s；accept ~66%/62%、n_draft=3。
- **应用场景**：OpenCL 后端 q6_k 的投机验证性能。
- **研判**：【事实】PR 给逐模型 t/s 对照（4B 22.5→32.3、9B 15.6→23.1）与 accept 率。【观点】属「OpenCL 投机验证」性能（第 5 条旁支）。**证据等级**官方一手（含 t/s 对照）。

### 5.5 本期其余 llama.cpp 要点（标题/正文级）
- **模型 / 支持**：`#31779`（见 5.1）、`#31782`（OpenCL 支持 repeat for f16/i32/i16/bf16）、`#31787`（CUDA 修 Maxwell build）、`#31788`（CI bump CUDA 13 release 到 13.5）、`#31806`（docker 加显式 /usr/local/nvidia 搜索路径）、`#31814`（Hexagon conv_2d/conv_3d 优化）。
- **MoE / 缓存**：`#31233`（见 5.2）。
- **安全 / 工具**：`#31730`（见 5.3）、`#31563`（cli 修无效 UTF-8 prompt/文件崩溃）、`#31718`（cmake 直接生成 RC 文件）、`#31757`（vendor BoringSSL 更新到 0.20261105.0）、`#31791`/`#31797`（tests 限定 isnan/signbit / 覆盖 hardware_concurrency）。
- **后端**：`#31658`（Vulkan 检测 ReBAR 支持、不可用时避免 BAR 分配）、`#31744`（Hexagon set_tensor_n + repacking/tiling 健壮性）、`#31473`（WebGPU batched tensor API get/set_tensor_n）。

---

## 6. 累计结论（继承 1–10，本期更新）

- **第 1 条（调度器隐式阻塞的定量画像）**：本期 vLLM `#65831`（KV block 分配/缓存每 block CPU 成本 −45~50%）继续补强「释放/分配路径」；沿用，无新增铁证。
- **第 2 条（发版窗口的双向性：发版当天合入 ≠ 在版内）**：本期 TRT-LLM v1.4.0rc3（2026-11-06）为窗口内新发布，但 rc3 为 prerelease、非稳定版；llama.cpp 仍 v0.9.0（11-04）。沿用。
- **第 3 条（默认值成为主要风险载体）**：本期以 BREAKING/配置清理与默认值旁支为主——TRT-LLM `#20703`（checkpoint formats 枚举化，BREAKING）、`#21455`（aggregated Helix 默认开 host tier，既有部署行为可能变）、vLLM `#65463`（auto 重拟合）、llama.cpp `#31730`（HF token 限主机，安全默认值）。**默认值/配置/BREAKING 清理仍是高频风险载体**，但本期无「影响面极广的默认值迁移」（如 11-05 的 vLLM block-size 16→64）。
- **第 4 条（异构后端静默错误高频）**：**本期新增 SGLang `#44437`（FP4 MLA `move_kv_cache` 不随 KV 移动 block scales → 静默反量化错、无报错）铁证**，与往期 #46928/#45486/#46833（AMD aiter/CTA fp8）同源；vLLM `#64986`（KV-sharing fast prefill host offsets 未同步挂起）为异步调度下 host/device 同步崩溃；SGLang `#47892`（KV scales 存储格式致启动失败）、`#48522`（HiCache L3 prefetch 静默丢弃已付费 KV）亦强化。**异构/量化静默错误持续为最密集主线之一**。
- **第 5 条（前缀缓存 × MoE 路由 × 投机解码交叉债务）**：**本期 vLLM 投机解码正确性簇为最密集子项**——`#65229`（接受估计器 fp32 灾难性抵消，11.5% draft 位置置信被毁）、`#66048`（resampled kernel 跨 warp 竞争致越界 CUDA 异常）、`#65804`（standalone draft prefill 误分类为 prefill → FULL graph 捕获 prefill 路径）、`#65710`（speculative Guidance EOS）、`#66010`（Sleep Mode 唤醒后非参数状态丢失）；SGLang `#48751`（EPLB draft 空记录污染 target 统计）；llama.cpp `#31758`（OpenCL q6_k 投机验证 +43~48% t/s）。**投机解码正确性跨引擎延续**。
- **第 6 条（量化路径静默错误，与异构后端并列的第二类高频静默错误）**：SGLang `#44437`（FP4 MLA KV scales 不随 KV 移动，静默错）、`#47892`（KV scales 存储格式致启动失败）；vLLM `#64410`（w4afp8→W4A8 互通）、`#57823`（MoE 路由 padded stride 静默改路由，第 7 条同源）；llama.cpp `#31233`（MoE-cache LRFU）。**量化静默债跨引擎延续**。
- **第 7 条（专家张量/路由归属与分页漏接静默错误）**：vLLM `#57823`（padded 物理行致原生 top-k 静默改路由，int64 offset 防回绕）、SGLang `#48751`（EPLB draft 空记录污染 target 统计）、`#48586`（Kimi-K3 decode MoE 路由融合）相关；**本期 vLLM `#57823` 为「编译下静默改路由」铁证**，强化第 7 条。
- **第 8 条（KV 容量/复用智能：容量自检、准入、传输/卸载正确性）**：vLLM `#64986`/`#65554`/`#65463`（KV-sharing/可扩展 KV cache 挂起/非法访存/auto 重拟合）、SGLang `#48522`/`#48231`（HiCache L3 prefetch 静默丢弃 / Mooncake KV item 长度不匹配显式拒绝）、TRT `#21455`（aggregated Helix 主机层挂起）、`#20664`（DSpark Unified KVCache）、llama.cpp `#31233`（MoE-cache LRFU）。**KV 正确性/复用跨引擎延续**。
- **第 9 条（长上下文 int 溢出为静默正确性新子项）**：本期 vLLM `#57823` 以 int64 防 32 位 input-row offset 回绕（强化第 9 条）；其余无新增 int 溢出铁证。沿用。
- **第 10 条（量化/MXFP8 KV 正确性簇）**：SGLang `#48971`（DSv4.1 wo_b/Engram KV MXFP8 GEMM 在 CC10.7>8K 行路由 cuDNN，bitwise 相等）、`#48723`（mega_moe SM107 预留 18 SMs）；vLLM `#65746`（GLM-5.3 sm100 fp8 moe tma 1.03~1.60×）；沿用并强化。

---

## 7. 社区侧（HN Algolia，2026-11-06 窗口）—— 引擎相关高信号弱，模型/量化生态帖为主

HN Algolia（2026-11-06 窗口，epoch 1793923200–1794009600）引擎关键词高信号弱：
- `vllm` 431 命中多为 LLM 网关/工具/招聘类（如「I Hate Emacs」「Who is hiring」），**无一条指向引擎更新**；
- `sglang` 仅 1 条「Ask HN: Who wants to be hired? (November 2026)」；
- `KV cache` 仅 1 条 Simon Willison 博客《Qwen 3.8 27B on my laptop can interpret model weights, too》（2026-11-06T19:45:51Z，二手线索/未核验）；
- `TensorRT-LLM` 0 命中；
- `llama.cpp` 4 命中中含 **《Swapping LLMs in Under Two Seconds: A Sleep Mode for Llama.cpp》**（zaage.it，2026-11-06T22:59:16Z，1 point，第三方博客，与 vLLM `#66010` / llama.cpp 睡眠唤醒主题呼应，二手线索/未核验）、**Ripple-65B Preview: U.S. Open-Weight SOTA on Agentic Coding**（ricursive.com，2026-11-06T18:51:43Z，关联往期 llama.cpp Ripple 模型支持 #31692/#31749，二手线索/未核验）、Qwen3.8 生态帖、TieredQ 量化论文（GitHub，2026-11-06T05:22:43Z，1 point，矢量量化研究，二手线索/未核验）；
- `quantization` 命中含 **TieredQ: Space Efficient Vector Quantization（Beating RaBitQ and TurboQuant）**（GitHub，1 point，量化研究，二手线索/未核验）；
- `LLM inference` 命中含 **Ripple-65B Preview**、Astra 读写权重改进 Qwen（rikiyat.github.io，weight-space programming，二手线索/未核验）、Tencent Hy4（hy.tencent.ai，二手线索/未核验）。

**研判**：【事实】上述均为 2026-11-06 当日 HN 发布的博客/公告链接与标题。【观点】窗口内无引擎官方更新类高信号；与本期主线相关的仅有「llama.cpp Sleep Mode」博客与「Ripple-65B / Qwen3.8 / TieredQ」等模型/量化生态帖——**均为二手线索/未核验**，未核验其对主流引擎的可用性、精度或性能声称。**证据等级**二手线索/未核验（HN 公告 + 外部博客/GitHub，未独立验证）。

---

## 8. 趋势判断与可执行清单

**趋势判断**：
1. **Kimi K3.1 跨引擎 day-0 支持成为协同事件**（新观察）：vLLM `#66030`/#66028/#65725、SGLang `#48721`/#48714/#48713、llama.cpp `#31779` 在**同一 UTC 窗口**把 Kimi K3.1（DSA 每 MLA 层）推进到三家开源引擎——新模型落地速度已进入「跨引擎同窗口」节奏（趋势第 1 条）。
2. **静默正确性跨引擎延续且新增铁证**（第 4/6 条）：SGLang `#44437`（FP4 MLA KV scales 不随 KV 移动 → 静默反量化错、无报错）是异构/量化静默错误的教科书级新增；vLLM `#64986`（KV-sharing host offsets 未同步挂起）、SGLang `#47892`（KV scales 存储格式启动失败）、`#48522`（HiCache L3 prefetch 静默丢弃已付费 KV）；**TRT-LLM v1.4.0rc3 Known Issues 明示 DSA+NVFP4 KV 可能静默错误**——静默正确性主线从「bug 修复」升维到「release 已知问题」。
3. **投机解码正确性簇最密集**（第 5 条）：vLLM `#65229`（接受估计器 fp32 灾难性抵消，11.5% draft 位置置信被毁）、`#66048`（resampled kernel 跨 warp 竞争）、`#65804`（standalone draft prefill 误分类）、`#65710`、`#66010`（Sleep 唤醒非参数状态丢失）共构「投机解码 × 数值/竞争/分类/状态」正确性收口。
4. **MoE 路由静默正确性**（第 7 条）：vLLM `#57823`（padded 物理行致原生 top-k 静默改路由、int64 防回绕）为编译下静默改路由铁证；SGLang `#48751`（EPLB draft 空记录污染 target 统计）。
5. **KV 容量/复用/传输正确性**（第 8 条）：vLLM `#65554`/`#65463`、SGLang `#48522`/`#48231`、TRT `#21455`/`#20664` 共构 KV 卸载/HiCache/PD/Mooncake 正确性收口，且 Mooncake 把 KV 布局不一致由静默损坏转为显式拒绝。
6. **TRT-LLM v1.4.0rc3 发布 + Qwen3.8 家族落地**（第 3 条旁支）：rc3 含大量模型支持与 3 项 BREAKING，`#20714`/`#21372` 把 Qwen3.8 家族推进到 TRT-LLM；rc3 Known Issues 须优先核对。
7. **新硬件后端**：SGLang `#45149` 初始支持 Cambricon MLU（国产 MLU），扩异构版图。

**可执行清单（建议）**：
- **P0（rc3 升级门禁）**：升级 TRT-LLM 到 v1.4.0rc3 前，核对 Known Issues——DSA 模型（含 GLM-5.2）在 NVFP4 KV cache 下可能静默错误、Kimi K3 DSpark + MegaMoE CuTe DSL TEP + CUDA graphs 在 GB300 warmup 可能挂起、DeepSeek-V4-Pro MPI worker 复用后可能 OOM、TP+RPC 复用 MPI worker pool 可能初始化挂起（对应 rc3 Known Issues）。
- **P0（静默错误自查）**：SGLang 开启 FP4 MLA + 投机解码（EAGLE topk>1 / NGRAM）时核对 `#44437`（move_kv_cache 须随 KV 移动 block scales）；vLLM 启用 `--kv-sharing-fast-prefill` 前核对 `#64986`（host offsets 同步）与可扩展 KV cache `#65554`/`#65463`；SGLang `--quantization-param-path` + fp8 KV 核对 `#47892`（KV scales 存储格式）；SGLang HiCache + L3 后端 + HySWA 模型核对 `#48522`（L3 prefetch 匹配键上限，约 49.6% prompt 长度会静默丢预取）。
- **P1（投机/PD 验收）**：vLLM 投机解码验收含 `#65229` 接受估计器数值精度、`#66048` resampled race、`#65804` standalone draft prefill 分类、`#66010` Sleep 唤醒后非参数状态；SGLang Mooncake PD 验收含 `#48231`（KV item 长度不匹配显式拒绝）、EPLB 含 `#48751`（draft 空记录污染）。
- **P1（路由/编译）**：torch.compile 下 MoE top-k 路由核对 vLLM `#57823`（padded 行 stride）；关注 SGLang Kimi-K3 decode MoE 路由融合 `#48586` 的 fallback 覆盖。
- **P2（跟踪）**：Kimi K3.1 day-0 支持在三家引擎的实测精度/吞吐对照（趋势第 1 条）；SGLang MLU (Cambricon) 后端在目标国产机型的可用性（#45149）；llama.cpp MoE-cache LRFU（#31233）在目标负载下的 decode 提升复现。

> 证据等级说明：本档案「官方一手」指来自四家官方仓库的提交/PR/发布元数据与 PR 正文/HTML `comment-body`；「二手线索/未核验」指来自社区公告/HN/外部博客/GitHub 且未经一手来源确认的内容（如 Sleep Mode 博客、Ripple-65B、Qwen3.8 生态帖、TieredQ）。所有性能/精度数字均标注来源 PR 或注明「截断/未完整呈现/描述级/作者报告未独立验证」，未外推未呈现数字。
