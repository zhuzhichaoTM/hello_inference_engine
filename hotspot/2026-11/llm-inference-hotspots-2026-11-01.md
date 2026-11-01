# 大模型推理优化每日热点档案

> 档案日期：2026-11-01（本自动化于该日执行）
> 关注窗口：2026-10-31（周五）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + PR 正文（pulls API，markdown）+ HN Algolia（HTTPS）
> 数据质量：四家提交 API 均 HTTP 200；PR 正文 14 篇已抓取；HN 窗口内 KV cache 4 条研究帖（二手线索/未核验），引擎相关无高信号故事。

## 1. 版本状态与本期头号事件：静默正确性主线延续且范围扩大（int32 溢出跨引擎再现 + 量化/MXFP8 KV 正确性簇）

**提交计数（2026-10-31 UTC，均 HTTP 200）**：vLLM **39** / SGLang **71** / TensorRT-LLM **12** / llama.cpp **13**。

**发布状态**（四家窗口内均无新正式版本）：
- vLLM：最新正式版仍为 **v0.32.0**（2026-10-18T06:50:48Z）；nightly 持续。
- SGLang：最新正式版 **v0.5.22**（2026-10-19T02:55:19Z）；窗口内无新 tag（0.5.23 仍未发布）。
- TensorRT-LLM：最新仍为 **v1.4.0rc2**（2026-10-28T15:51:34Z，prerelease），窗口内无更新。
- llama.cpp：窗口内发布 nightly 标签 **b12019–b12024**（2026-10-31，均为 prerelease 自动构建）。

**本期头号事件（静默正确性主线延续 + 范围扩大）**：
1. **int32 溢出类跨引擎再现（强化累计结论第 9 条）**：vLLM `#64820` 证实 Qwen4Exp 的 HyperConnection / PLE gate 核把行偏移 `ptr + row*stride` 在 int32 下回绕，在 row 209,716 / 167,774 即触发（main 上两测试直接 **SIGSEGV**），改 int64 后通过；llama.cpp `#31442` 修复 Hexagon DMA 在 `stride > 24 bit` 的整型溢出。**int 溢出静默/崩溃类已从「长上下文 Qwen3.5」扩展到 Qwen4Exp 与端侧 Hexagon，成为跨引擎共性问题**。
2. **量化 / MXFP8 KV 正确性簇集中爆发（SGLang，新增累计结论第 10 条）**：同一日内 `#47337`（MXFP8 dummy weight 的 UE8M0 scale 为 0 → 反量化≈0 → NaN 向下游传播）、`#47543`（HiCache 钉 host buffer 未 mlock，ATS GPU 上页老化致 page-fault）、`#46936`（MXFP8 KV 在 HiCache L3 原 `NotImplementedError`，现存 payload+scales）三连——**量化 KV 与 KV 卸载的正确性成为新的集中暴露面**。
3. **默认值风险本期以 opt-in / safe 默认为主**：SGLang `#46827` 的 `joint_lru` 淘汰策略为 **opt-in**（`--radix-eviction-policy joint_lru`），vLLM `#8ecf86ba83` 把 `num_redundant_experts` 默认到最小有效值——本期待行为变更默认（如 10-31 的 `#19379`/`#21041`）明显收敛，无重大静默行为翻转。

---

## 2. vLLM：2026-10-31 提交（39 条）——int32 溢出家族 + 量化/投机确定性 + ROCm 前缀缓存回放

### 2.1 `#64820` Qwen4Exp HyperConnection / PLE gate 行偏移 int32 回绕（静默正确性，头号）
- **主题**：Qwen4Exp 的 HC 核（`nvidia/ops/hc.py`、`amd/ops/hc.py`）与 PLE gate 核（`common/ops/ple.py`）按 `ptr + row*stride`（`row = tl.program_id(0)`）算每行地址，乘积在 int32 下求值，一旦 `row*stride >= 2^31-1` 即回绕。
- **内容详情**：把 11 个核的行偏移统一拓宽到 int64（列/组/流/瓦片索引仍 int32）。2560-wide hidden + 4 HC stream 时 HC row stride=10240、PLE K/V row stride=12800，分别在 row 209,716 / 167,774 回绕；`max_num_batched_tokens >= 167,774` 的启动 profile 即命中。新增 `test_row_offsets.py`：在第三行起点 `2^31 + 8192` 元素处校验，CPU 下 `TRITON_INTERPRET=1` 时 main 两测试均 **SIGSEGV**、分支通过；sm90/Triton 3.6.0 下 main 生成 `mul.lo.s32`、分支生成 int64 乘法。
- **应用场景**：Qwen4Exp（含 HC/PLE）长序列 prefill/生成；serving 方长上下文易踩。
- **研判**：【事实】PR 含逐核根因与新回归测试，main 现崩溃、分支修好。【观点】这是 `#63761`（Qwen3.5 int32 溢出）同族，但表现为**硬崩溃**而非静默错输出——印证第 9 条「int 溢出类横跨静默错与崩溃」，建议把 int32 边界（~167K–524K token）纳入默认门禁（强化第 4、6、8、9 条）。
- **来源**：vLLM PR [#64820](https://github.com/vllm-project/vllm/pull/64820)（merge 2026-10-31T02:41:53Z）。
- **证据等级**：官方一手（含逐核回归 + SIGSEGV 复现）。

### 2.2 `#62192` Batch invariance：钉死 FlashInfer FP8 bmm 与 W4A16 NVFP4 核（确定性 / 静默正确性）
- **主题**：`VLLM_BATCH_INVARIANT=1` 下，同一行结果不应随 batch 变化；但 FlashInfer FP8 `bmm` 默认按 M bucket 在 cutlass/cuBLAS/cuDNN 间 autotune，W4A16 NVFP4 此前被强制到仅 W4A4 的 Cutlass 核（直接 `ValueError`）。
- **内容详情**：`flashinfer_scaled_fp8_mm` 在 BI 下传 `backend="cutlass"`；W4A16 NVFP4 线性在 BI 下选 `HummingNvFp4LinearKernel`（已读 `use_batch_invariant`）。新增 SM100+ 的 `test_flashinfer_fp8_mm_batch_invariant_after_autotune` 与 CPU 的 `test_batch_invariant_w4a16_selects_humming`。关闭 BI 行为不变。
- **应用场景**：可复现/确定性推理、A/B 对比与回归基准。
- **研判**：【事实】PR 给出 autotune 导致跨 batch 后端不一致的根因与两处测试。【观点】属「非确定性被 autotune 悄悄引入」类静默正确性（第 8 条），批量一致性应作为发布基准之一。
- **来源**：vLLM PR [#62192](https://github.com/vllm-project/vllm/pull/62192)（merge 2026-10-31T15:25:10Z）。
- **证据等级**：官方一手（含确定性测试）。

### 2.3 `#63416` NIXL KV Connector：每次传输按层配对 KV region（PD/KV 传输健壮性）
- **主题**：NIXL 此前按「index / PP 窗口 / 层名 / packed-row 拆分」四种方式配对对端 KV region；本 PR 统一为一条规则——**每次传输都按双方持有的层配对 region**。
- **内容详情**：每个 worker 广播其各 region 的层（`region_members`）与每层的 cache group（`layer_group_ids`）；握手时 `_pair_regions` 按层在两侧查找、去重记录（peer region, local region, cache group）每对一个；`_compute_desc_ids` 各模式复用。覆盖 PP=1、PP pull/push、HMA/packed 等场景。
- **应用场景**：PP/TP 混合并行下的 NIXL PD 与 KV 卸载；多 cache group 模型。
- **研判**：【事实】PR 描述统一配对规则与各模式收敛。【观点】属 KV 连接器层健壮性重构，降低「层/region 错配导致错误 KV 复用」的风险，与跨引擎 KV 正确性主线同源（第 4/8 条）。
- **来源**：vLLM PR [#63416](https://github.com/vllm-project/vllm/pull/63416)（merge 2026-10-31T05:51:20Z）。
- **证据等级**：官方一手（PR 正文）。

### 2.4 `#63098` DSv4.1 启用 PCP + PP / DP（特征，已实测）
- **主题**：放宽 DSv4.1-Flash（text-only，NVIDIA CUDA）在启用 PCP 时的模型级兼容检查，开放既有的 PCP+PP 与 PCP+DP 路径（底层 MRV2 来自 `#59139`/`#57075`）。
- **内容详情**：移除「PCP 开启时 PP=DP=1」的全局限制；TP=1 或禁用专家并行时允许 PP+DP 组合；DCP 必须保持 1；在 `test_deepseek_v41_pcp.py` 增加 PP2+PCP2 / DP2+PCP2 真实权重服务+生成用例。
- **应用场景**：DSv4.1-Flash 的多维并行部署（PP/DP 与 PCP 组合）。
- **研判**：【事实】8×H20(96G) 实测，新增真实权重测试。【观点】多维并行组合增加，PCP 与 PP/DP 的交叉 KV 路由需回归覆盖（第 5 条交叉债务）。
- **来源**：vLLM PR [#63098](https://github.com/vllm-project/vllm/pull/63098)（merge 2026-10-31T04:41:30Z）。
- **证据等级**：官方一手（含 8×H20 验证）。

### 2.5 `#64094` ROCm DSv4.1 前缀缓存 SWA replay 在 gfx950 重新开启（性能，含验证）
- **主题**：恢复 DSv4.1 前缀缓存 sliding-window bounded replay 在 gfx950 的路径（#59039 已开 decoder replay；本 PR 仅恢复来自 #57906 的前缀缓存路径）。gfx942 仍强制关闭，其余 ROCm 维持原 force-off。
- **内容详情**：AITER sparse MLA 路径消费已被 clamp 到 `replay_start` 的 slot id；gfx950 上 `--no-swa-bounded-replay` 才关，默认仍开。MI355X 最佳配置下端到端吞吐 **concurrency 128 +13.5%、256 +12%**；GSM8K（0.904 vs 0.902）与 RULER NIAH 1.0/1.0 不变。
- **应用场景**：ROCm MI355X(gfx950) 上 DSv4.1 长上下文前缀复用。
- **研判**：【事实】PR 给出 GSM8K/RULER 表与吞吐提升，精度无退化。【观点】ROCm 前缀缓存 replay 收益明确且精度不变，属安全收益；gfx942 仍关，ROCm 部署方需按 GPU 核对。
- **来源**：vLLM PR [#64094](https://github.com/vllm-project/vllm/pull/64094)（merge 2026-10-31T02:46:49Z）。
- **证据等级**：官方一手（含 GSM8K/RULER 验证表）。

### 2.6 其余 vLLM 要点（官方一手，标题/正文级）
- `#25fc60723c` **sparse-indexer prefill 融合 scoring+top-k 单发射核**（一次 launch 替代分拆，Perf Kernel）；`#63842` Rust 前端计算 prompt block hash（Perf Frontend）；`#63174` 跳过冗余权重张量 clone（loader 产出 owned tensor 时）；`#64841` ModelRunner V2 降 chunked prompt logprobs 峰值显存。证据等级：官方一手（标题/正文）。
- `#64792`/`#64805`/`#64804` ROCm 配方（Mistral Large 4 TP4、MiniMax-M3.1-Flash MI355X TP4 AgentX、DSv4.1-Flash MI355X TP8 c1 AgentX 预设）；`#62956` SM121 TP=2 GLM-5.3-Flash skinny-GEMM、`#61614` FlashKDA pin 直读 row-strided QKV、`#64650` SM103 Qwen3.8 TP=1、`#64368` ROCm DSpark drafter 的 `draft_lm_head_quantization`（Spec Decode）。证据等级：官方一手（标题/正文级）。
- `#61542` HYV4 参数名保留空白（Tool Parser bugfix）、`#64757` 批量 chat completions 上报 `prompt_tokens_details`、`#54399` CohereASR FP16 attention bias dtype、`#64338` PP>1 下 FlashInfer attention warmup 死锁、`#62967` gRPC 在 `--shutdown-timeout` 内排空在途 RPC、`#64579` NIXL bump 1.5.2、`#63297` ROCm Kimi-K3 gluon 后端保留 KDA prefill checkpoint、`#63116` ROCm AITER MLA persistent metadata 按 decode query 长度、`#64269` ColBERT reranking 批化 token heads、`#64653` MoE-LoRA 扩至 64 token、`#64830` V2-only 检查移入 V1、`#61837` Step-5-Preview 上 ROCm、`#64675`/`#64673`/`#64674`/`#64575` 前端 parser 性能/bugfix。证据等级：官方一手（标题级为主）。

---

## 3. SGLang：2026-10-31 提交（71 条）——量化/MXFP8 KV 正确性簇 + 统一内存淘汰 + PD/HiCache 健壮性

### 3.1 `#47337` MXFP8 dummy weight 的 UE8M0 scale 为 0 → 反量化≈0 → NaN（静默正确性，头号）
- **主题**：`initialize_dummy_weights` 只初始化浮点张量；MXFP8 层的 UE8M0 block scale（`weight_scale_inv`）分配为 `uint8` 零，在 `--load-format dummy` 下保持 0。UE8M0 中 0 表示 2^-127，于是每个 MXFP8 权重反量化≈0，输出近乎零并可向下游传播成 NaN。
- **内容详情**：把 `*weight_scale_inv` 的 uint8 填 127（即 2^0，与已存在的 FP4 neutral 全局 scale 填 1.0 对应）；新增 `test_ue8m0_dummy_block_scales_are_neutral`（紧邻 `test_nvfp4_dummy_scales_are_neutral`）。
- **应用场景**：`--load-format dummy` 的占位/结构验证；真实 checkpoint 自带 scale 不受影响。
- **研判**：【事实】PR 给出根因（UE8M0 0=2^-127）与单测。【观点】又一处「量化 scale 未正确初始化 → 静默 NaN」实例，强化第 6 条（量化路径静默错误）与第 10 条（MXFP8/量化 KV 正确性面）。
- **来源**：SGLang PR [#47337](https://github.com/sgl-project/sglang/pull/47337)（merge 2026-10-31T04:23:23Z）。
- **证据等级**：官方一手（含单测）。

### 3.2 `#47543` HiCache：mlock 钉死 host buffer，防页老化扫描致 page-fault（静默正确性）
- **主题**：`cudaHostRegister` 让 host pool 页常驻 DMA，但仍挂在内核可驱逐 LRU 上；主动回收（page aging）会清除其 accessed 标志。在经 CPU 页表读 host 内存的 GPU（ATS）上，被老化页下次访问触发 page fault，host→device 拷贝逐页出错。
- **内容详情**：`_cuda_host_register` 在所有 range 注册后 `mlock`；所有 `alloc_with_host_register` 钉住的 buffer（CUDA/ROCm 的 HiCache host pool 及其余用户）随之锁定；`_cuda_host_unregister` 配对 munlock；锁定前把 `RLIMIT_MEMLOCK` 软限提到硬限（无需特权）。
- **应用场景**：HiCache host pool（CUDA/ROCm）在 ATS GPU 上的稳定 KV 卸载。
- **研判**：【事实】PR 描述根因（可驱逐 LRU vs 钉死 LRU）与 mlock 修复。【观点】属「KV 卸载页管理静默错误」类（与 vLLM KV-Offload #63266/#61019 同源），强化第 4/8 条与第 10 条。
- **来源**：SGLang PR [#47543](https://github.com/sgl-project/sglang/pull/47543)（merge 2026-10-31T04:22:25Z）。
- **证据等级**：官方一手（PR 正文）。

### 3.3 `#46936` HiCache：MXFP8 KV 页以 payload + scales 存 L3（正确性 + 能力解锁）
- **主题**：MXFP8 KV cache 下 HiCache L3 完全不可用——`MHATokenToKVPoolMXFP8Host` 因每页 UE8M0 scale 在 `kv_buffer` 之外的独立 host buffer 而对每存储页抛 `NotImplementedError`。
- **内容详情**：每页拆成两个 multi-buffer 对象（key 下 K payload 后接 K scales，V 同理），key schema 与其余非 MXFP8 pool 不变；`get_page_buffer_meta` 返回对应指针/尺寸对，`get_hybrid_pool_buffer` 暴露两 scale buffer 供注册。新增 `test_storage_objects_round_trip_payload_and_scales`（NotImplemented→pass）；`test_mxfp8_mha_pool_host_unit.py` 5 passed；PD + MXFP8 KV 端到端每轮 logprobs 与无缓存参考一致（最大漂移 4.4e-16）。
- **应用场景**：MXFP8 KV 的 HiCache L3 卸载（此前不可用）。
- **研判**：【事实】PR 给出 round-trip 测试与端到端漂移 4.4e-16。【观点】补齐 MXFP8 KV 的 L3 卸载能力，同时消除一处「静默 NotImplementedError」，强化第 10 条。
- **来源**：SGLang PR [#46936](https://github.com/sgl-project/sglang/pull/46936)（merge 2026-10-31T15:25:20Z）。
- **证据等级**：官方一手（含 e2e 漂移验证）。

### 3.4 `#46827` unified-memory：`joint_lru` 统一 Full/SWA KV 淘汰（opt-in 默认，非错误）
- **主题**：`--enable-unified-memory` 下 Full/SWA 混合模型把两类 KV 放同一共享字节缓冲；原分配期回收各自按配额驱逐，压力下一侧被抽干而另一侧仍保留很久未用的条目。
- **内容详情**：新增 opt-in `--radix-eviction-policy joint_lru`——按一个共享访问时间序淘汰 Full 叶子与 SWA 条目（每次取最久未用者）；其余策略/位置行为不变。正确性由 `uses_joint_lru` 在构造期判定（需 policy=joint_lru、cache 启用、树含 SWA 组件、allocator 为双端 ChainAllocator）。
- **应用场景**：Full/SWA 混合模型在统一内存下的 KV 淘汰公平性。
- **研判**：【事实】PR 描述 opt-in 策略与判定条件。【观点】属「默认值风险」主线的**正面收敛**——以 opt-in 引入，未静默改变行为；混合模型部署方可按需开启。
- **来源**：SGLang PR [#46827](https://github.com/sgl-project/sglang/pull/46827)（merge 2026-10-31T17:13:58Z）。
- **证据等级**：官方一手（PR 正文）。

### 3.5 其余 SGLang 要点（官方一手，标题/正文级）
- `#47452` kimi-k3 允许 residual bank shards 经 NCCL（Perf）；`#47557` HiCache 仅扣留 protected bytes 作为 page-cache credit（Fix）；`#47652` PD 忽略已 scatter chunk 的迟到 STAGING_REQ（Fix）；`#47638` PD 注册时把 `aux_item_lens` 紧邻 `aux_ptrs` 发送（Fix）；`#40107` PD 跨阶段保留 sampler `customized_info`（Fix）；`#47546` 已发 abort 的请求不再 dispatch（Fix）；`#47662` M3 拒绝枚举外 `reasoning_effort`（Fix）；`#45174` MiniMax-M3 CUDA radix decode top-k 确定性 tie-breaking（Fix）；`#a2cafa5676` sampling 在 `tp_size>1` 时确定性 top-p/top-k renorm（DFlash TP rank 分歧，Fix，标题级）。证据等级：官方一手（标题/正文）。
- `#47159` observability 在 `meta_info` 报告 TTFT 相位分解（Feat）；`#46908` qwen3 decode 复用融合 QK-norm/RoPE 核（Perf NVIDIA）；`#46724` muse-spark-open 把 MoE finalize 与 2shot_push all-reduce 融合（Perf AMD）；`#37367` AMD MoE 从 AITER 激活契约解析 MoRI dispatch（Perf）；`#45876` DFLASH draft bf16 GEMM 在 GB300 走 tuned dispatch（Perf）；`#45302` NPU DSA 稀疏卸载 index-cache 预算（Fix）。证据等级：官方一手。
- 配方/文档：`#47520`/`#47615`/`#47623`/`#47660` AMD cookbook（GLM-5.2 MI355X、MiniMax-M3.1-Flash、Qwen3.8-Flash-Next B300 NVFP4 / MI355X AgentX）。diffusion 性能簇（MiniMax-H3 等）约占当日提交 1/4+，但与本档案推理引擎主线关联弱，未逐条展开。证据等级：官方一手（标题级）。

---

## 4. TensorRT-LLM：2026-10-31 提交（12 条）——MiniMax-M3 NVFP4 低延迟 decode megakernel + 一项 BREAKING + PD/视频新能力

### 4.1 `#20306` MiniMax-M3 NVFP4 低延迟 decode megakernel（Blackwell modeling_v2，性能/特征）
- **主题**：为 MiniMax-M3 在 Blackwell 上的 modeling_v2 引入 NVFP4 低延迟 decode megakernel。
- **内容详情**：commit `f405e10b73`（merge 2026-10-31T09:56:52Z，PR #20306）描述为「MiniMax-M3 NVFP4 low-latency decode megakernel for modeling_v2 for Blackwell」。属 Blackwell(SM100) 上低比特 decode 的融合内核落地。
- **应用场景**：MiniMax-M3 在 Blackwell 的低延迟 NVFP4 decode 服务。
- **研判**：【事实】commit message 明确为 NVFP4 低延迟 decode megakernel（Blackwell）。【观点】与 10-31 `#203xx` 同族的 Blackwell 低比特 decode 加速方向一致；**性能数字未在 commit message 中给出，记为官方提交（描述级/无独立性能数字），不计入量化结论**——如需采用须以 PR 正文或 release note 实测为准。
- **来源**：TRT-LLM commit [f405e10b73](https://github.com/NVIDIA/TensorRT-LLM/commit/f405e10b73) / PR [#20306](https://github.com/NVIDIA/TensorRT-LLM/pull/20306)。
- **证据等级**：官方提交（描述级，无独立性能数字）。

### 4.2 `#21036` BREAKING：移除未使用的 LlmRequest 字段（默认值/API 风险主线）
- **主题**：从原生 request API、Python 绑定与序列化中移除 prompt-tuning 配置与 prompt-table 状态，以及原生 `Result` 的 logprob 字段与序列化表示。
- **内容详情**：Python sampler logprobs 仍支持；原生 `tllm.Result` 跳过被移除字段；`prompt_adapter_request` 作为 deprecated 兼容参数保留（非 None 时告警并拒绝）。coderabbit 评审指出「未提供当前评审发现，severity 计数不可用、未建立完整测试运行结论」。
- **应用场景**：TRT-LLM 原生 API / 序列化下游。
- **研判**：【事实】PR 为 BREAKING API 移除，需兼容库与绑定重编。【观点】属「API 破坏性变更」主线的常规项；下游须移除引用并针对新头文件重编——与 10-31 `#21041` 同类，升级时需审计。
- **来源**：TRT-LLM PR [#21036](https://github.com/NVIDIA/TensorRT-LLM/pull/21036)（merge 2026-10-31T08:52:34Z）。
- **证据等级**：官方一手（BREAKING API 移除，coderabbit 评审未给完整测试结论）。

### 4.3 能力 / 性能 / PD 新特性（官方一手）
- `#20787` **Sol-Attn 可选 Morton(Z-order)/Hilbert token ordering**（Wan 视频）：把 64-token block 的 3D 局部性从「沿宽长条」提升到「~4×4×4 紧邻邻域」，置换仅作用于 token 序列与 RoPE 表，密集数学不变；对齐 NVlabs/Sana 的 `install_wan_morton_forward`。证据等级：官方一手（含 locality 量化）。
- `#20491` **PD 取消配对 GEN**：generation-first serving 在配对请求失败/取消时取消并排空对侧待办，保留已完成 GEN 响应、不吞掉取消；同时 SSE 观察、HTTP 诊断（2,048 字节流式错误前缀）、委派放置清理均做相应改动。证据等级：官方一手（PD 正确性）。
- `#21142` cache reset 事件不再等待推理完成即发布（fix）；`#21125` 给 OpenEngine KV 事件加 image routing metadata（feat）。证据等级：官方一手（标题/正文）。
- `#21292` 把构建产物硬链进 Python 包、`#21345` 隔离 modeling_v2 测试、`#21175` 更新 modeling_v2 代码/测试归属、`#20017` util_convert_type 返回 owning fragment、`#20911` flashinfer-python 0.6.18→0.7.2。证据等级：官方一手（chore/refactor）。

---

## 5. llama.cpp：2026-10-31 提交（13 条）——共享 KV 层注意力旋转修正 + SYCL 量化舍入 + Hexagon 整型溢出

### 5.1 `#31406` 共享 KV 层注意力张量旋转修正（正确性，头号）
- **主题**：复用其他层 KV cache 的层，其注意力张量须使用源层的旋转，否则 Q 与 KV head 不在同一设备。
- **内容详情**：为 `test-llama-archs` 启用 Gemma 4（#31401）所需；原实现在 `llama-model.cpp` 的 Gemma 特定逻辑被移入 Gemma 专属文件。PR 由 Opus 5.5 辅助定位（作者重度改写）。
- **应用场景**：Gemma 4 等多层共享 KV 的架构；含共享 KV 的模型加载与推理。
- **研判**：【事实】PR 给出根因（旋转须用源层）与 Gemma 4 测试失败→修复对照。【观点】属「注意力/异构层 KV 一致性」静默正确性（第 4 条），共享 KV 架构须在开启前验证 head 设备一致性。
- **来源**：llama.cpp PR [#31406](https://github.com/ggml-org/llama.cpp/pull/31406)（merge 2026-10-31T22:30:50Z）。
- **证据等级**：官方一手（含 Gemma 4 测试）。

### 5.2 `#31120` SYCL：f32→q2_0/q4_0/q5_0 舍入修复（量化正确性，强证据）
- **主题**：SYCL 下 `f16/bf16 → q2_0, q4_0, q5_0` 的 CPY 随机失败（测试数据随机，故「随机」）；设备侧三处与 CPU 差 1 ulp：1.0f/d 的倒数、x*id+8.5f(q4_0)/16.5f(q5_0) 的 fused multiply-add、`(int)(x+0.5f)` 舍入。
- **内容详情**：`ggml/src/ggml-sycl/cpy.hpp` 改为正确舍入除法、向零外的 round-half-away、no-op `fmin` 阻止融合；仅 f32→qX_0 block 函数改动。`test-backend-ops -o CPY`：master **70/100 失败 → 0/100 通过**（Arc B580 12GB / i5-13400 / Ubuntu 24.04 / icpx 2025.3.3）。review 同时给 Metal `quantize_q2_0` 补 `#pragma METAL fp math_mode(safe)`。
- **应用场景**：Intel SYCL（Arc GPU）上的低比特量化 CPY；量化正确性回归。
- **研判**：【事实】100-run 实测 master 70 失败→修复 0 失败，证据强。【观点】量化舍入 1 ulp 差异即致随机失败，印证第 6 条（量化路径静默错误）跨后端存在；SYCL/Intel 部署方应合入。
- **来源**：llama.cpp PR [#31120](https://github.com/ggml-org/llama.cpp/pull/31120)（merge 2026-10-31T01:54:42Z）。
- **证据等级**：官方一手（含 100-run 对比）。

### 5.3 `#31442` Hexagon：修复 stride > 24 bit 的 DMA 传输（整型溢出，跨引擎印证）
- **主题**：Hexagon DMA 传输在 stride 超过 24 bit 时因整型溢出而出错。
- **内容详情**：修正 stride 的整型处理，覆盖大 stride 张量。属端侧 Hexagon 后端的低层 DMA 正确性修复。
- **应用场景**：高通 Hexagon NPU 上的大张量大 stride 传输。
- **研判**：【事实】commit 描述 stride > 24 bit 溢出修复。【观点】与 vLLM `#64820`、Qwen3.5 `#63761` 同属「整型溢出」家族（第 9 条跨引擎扩展），端侧后端亦需把整型边界纳入回归。
- **来源**：llama.cpp PR [#31442](https://github.com/ggml-org/llama.cpp/pull/31442)（merge 2026-10-31T23:49:23Z）。
- **证据等级**：官方一手（提交级）。

### 5.4 其余 llama.cpp 要点（官方一手）
- `#30886` mtmd 修复 InternVL CLS token 顺序；`#30922` mtmd Gemma 4 图像去 letterbox 缩放；`#31401` 测试启用 Gemma 4；`#31431` LFM3 启用 tensor split；`#30831` LFM 多模态张量 reshape 2D 替代 3D；`#31459` vendor miniaudio 0.11.26；`#31434` OpenCL 加 fwht；`#31380` qwen3-coder `reasoning_format none` 剥离 think 块；`#30710` server 把共享 decision 前缀截断在首个输出；`#30959` server `/v1/responses` 接受 `extra_params`。证据等级：官方一手（标题/正文级）。
- nightly：窗口内发布 **b12019–b12024**（2026-10-31，prerelease）。

---

## 6. 累计结论与研判（沿用 + 新增）

**沿用 2026-10-31 档案（覆盖 10-30）第 1–9 条：**
1. 调度器隐式阻塞的定量画像（host sync 移出热路径的 +4.5%~9.9% 画像）；
2. **发版窗口的双向性**：发版当天合入 main ≠ 在版内（compare API 祖先关系核验）；
3. **默认值成为最主要的风险/收益载体**（默认后端/量化/NUMA 静默改变行为）；
4. **异构后端静默错误高频**（AMD ROCm / Metal / 异构 kernel 不报错只掉精度）；
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**（草稿/索引 KV 所有权一致性）；
6. **量化路径静默错误**（scale 未应用/参数丢失/4D 量化漏算/舍入 1 ulp → 输出偏离且无报错）；
7. **专家张量/路由归属与分页漏接静默错误**（paged-experts 不被 paging、路由行覆盖 → 静默用错专家值）；
8. **静默正确性升维为跨引擎常态主线**（高频失效模式已从「崩溃」转为「不报错、精度静默掉/长上下文静默塌方」）；
9. **长上下文 / 大张量 int32 溢出为静默正确性新子项**（Triton 将 `<2^31` 整数参数按 int32 传递，Qwen3.5 #63761 在 ~209K token 回绕）。

**本期新增第 10 条（量化 KV 与 KV 卸载的正确性成为新集中暴露面）：**
- SGLang 同日三连 `#47337`（MXFP8 dummy weight scale=0→NaN）、`#47543`（HiCache host buffer 未 mlock 致 ATS 页老化 page-fault）、`#46936`（MXFP8 KV 在 HiCache L3 原 `NotImplementedError`、现存 payload+scales）显示**量化 KV（MXFP8/FP8 KV cache）与 HiCache 卸载的正确性**正成为新的集中爆发面。建议把「MXFP8/FP8 KV 的 scale 初始化、host pool 钉页/卸载、L3 存储 round-trip」纳入默认发布门禁（强化第 4、6、8 条）。

**主线强化（本期的进一步铁证）：**
- **第 9 条（int 溢出跨引擎）**：vLLM `#64820`（Qwen4Exp HC/PLE 行偏移 int32 回绕、main 直接 SIGSEGV）+ llama.cpp `#31442`（Hexagon DMA stride>24bit 溢出）证明「整型溢出」家族已从长上下文 Qwen3.5 扩到 Qwen4Exp 与端侧 Hexagon，横跨云/边。
- **第 6 条（量化路径静默错误）**：SGLang `#47337`（MXFP8 scale=0→NaN）+ llama.cpp `#31120`（SYCL 量化舍入 1 ulp、master 70/100 CPY 失败）新增两端证据。
- **第 3 条（默认值风险收敛）**：本期 SGLang `#46827` joint_lru、vLLM `#8ecf86ba83` 最小有效默认均为 opt-in / safe 默认，无 10-31 `#19379`/`#21041` 级的行为变更默认——默认值风险主线本期趋缓，但仍是升级审计项。

---

## 7. 趋势判断与可执行清单

**趋势判断**
1. **静默正确性主线继续主导且范围扩大**：本期跨 vLLM/SGLang/llama.cpp 的 silent-error 与确定性修复数量仍居首；int 溢出家族（#64820、#31442）与量化/MXFP8 KV 正确性簇（#47337、#47543、#46936、#31120）是新增高频点。
2. **量化 KV / KV 卸载正确性成为新独立面**：SGLang 单日三连指向 MXFP8 KV + HiCache 卸载的正确性，建议作为独立回归门禁（第 10 条）。
3. **默认值风险本期收敛为 opt-in / safe 默认**：`joint_lru`、最小有效默认等均以不静默改行为的方式引入，与 10-31 的 BREAKING 默认变更形成对比——升级方本期可相对放心，但 TRT-LLM `#21036` 的 BREAKING API 移除仍需审计。
4. **PD / KV 连接器走「按层配对 + 失败清理」健壮性路线**：vLLM `#63416`（NIXL 按层配对）、SGLang `#47652`/`#47638`/`#40107`（PD 修复）、TRT-LLM `#20491`（取消配对 GEN）共同指向 PD 解耦的失败语义与 KV 一致性加固。

**可执行清单（建议 serving/研发方）**
- [ ] **int 溢出回归门禁**：把 int32 边界（~167K–524K token / stride >24bit）纳入默认发布门禁，覆盖 Qwen4Exp/llama.cpp Hexagon 等（第 9 条、第 4 条）。
- [ ] **量化 KV / HiCache 卸载门禁**：MXFP8/FP8 KV 的 scale 初始化、host pool mlock、L3 round-trip 加回归（#47337、#47543、#46936，第 10 条）。
- [ ] **SYCL/Intel 量化**：合入 llama.cpp `#31120`（f32→q2_0/q4_0/q5_0 舍入，70/100→0/100 失败），量化 CPY 正确性回归（第 6 条）。
- [ ] **共享 KV 架构**：启用 Gemma 4 等共享 KV 模型前验证 #31406 的 head 设备一致性（第 4 条）。
- [ ] **TRT-LLM 升级审计**：`#21036` 移除 prompt-tuning/Result logprob 字段（下游重编）、`#20306` MiniMax-M3 NVFP4 低延迟 decode megakernel（性能数字待 PR/release 实测，记描述级）。
- [ ] **SGLang 默认/策略**：评估 `#46827` joint_lru（混合 Full/SWA 模型公平淘汰，opt-in）；PD 评估 `#47652`/`#47638`/`#40107` 的失败语义修复。
- [ ] **vLLM ROCm / 并行**：`#64094` gfx950 前缀缓存 SWA replay（+13.5%@c128，精度不变，可采纳）；`#63098` DSv4.1 PCP+PP/DP 多维并行组合加 KV 路由回归。

---

## 8. 来源索引与数据质量

**官方一手（提交 API + 发布说明 + PR 正文）**
- vLLM：https://github.com/vllm-project/vllm （**39 commits @2026-10-31**；v0.32.0 2026-10-18，窗口内无新正式版）；PR `#64820`/`#62192`/`#63416`/`#63098`/`#64094` 正文已抓。
- SGLang：https://github.com/sgl-project/sglang （**71 commits @2026-10-31**；v0.5.22 2026-10-19，窗口内无新 tag）；PR `#47337`/`#47543`/`#46936`/`#46827` 正文已抓（`#a2cafa5676` 仅标题级）。
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM （**12 commits @2026-10-31**；**v1.4.0rc2 2026-10-28T15:51:34Z** prerelease，窗口前一日发布、仍为最新）；PR `#21036`/`#20787`/`#20491` 正文已抓，`#20306`（MiniMax-M3 NVFP4 decode megakernel）以 commit `f405e10b73` 正文级记录。
- llama.cpp：https://github.com/ggml-org/llama.cpp （**13 commits @2026-10-31**；nightly **b12019–b12024** 2026-10-31，prerelease）；PR `#31406`/`#31120`/`#31442` 正文已抓。

**社区/外部（二手线索，未核验）**
- HN（2026-10-31 窗口，Algolia HTTPS，story 过滤）：引擎相关**无高信号官方更新故事**。`KV cache` 4 条均为研究帖（疑似 arXiv）：`GAAC: Lossless KV Cache Compression`（2 分）、`MCF: Move queries, not KV caches`（1 分）、`Gandalf: Extending SSD Lifespan for LLM KV Cache Offloading`（1 分）、`UlaCache: Unified Logical Addressing for Sparse LLM Serving`（3 分）；推理相关另有 `Non-autoregressive NC LLM spits tokens 21x faster than text diffusion`（3 分）、`Local Inference with LLMs`（3 分）。均记为**二手线索/未核验**（研究帖非官方引擎更新，未核验发布日期与可复现性），不计入结论。`vllm`/`sglang`/`llama.cpp`/`TensorRT-LLM`/`speculative decoding` 故事数均为 0 或无关（恶意软件用 LLM 排名受害者等）。
- Reddit / YouTube / X：未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

**数据质量与限制**
- 计数口径：GitHub commit API 的 `since`/`until` UTC 日期窗口（vLLM 39 / SGLang 71 / TRT-LLM 12 / llama.cpp 13，均 HTTP 200）；按 commit date 过滤。**vLLM 当日 39 条显著低于前一日（10-30 为 97）**，为周五实际提交量（无第二页），非采集异常；如需交叉核验可改查 PR merge 口径。
- 证据等级：带验证表的 silent-error/性能 PR 记为「官方一手（含验证）」；仅提交标题或正文未全抓的记为「标题级」（如 SGLang `#a2cafa5676`）；TRT-LLM `#20306` 记「官方提交（描述级，无独立性能数字）」；社区文章记「二手线索/未核验」。
- 未调用外部浏览器（agent-browser）：本期数据源（GitHub API / releases / HN Algolia）均可直接 HTTP 访问，沿用 2026-10-18 起确立的「API + PR 正文直抓」方法，未触发浏览器需求（与历次一致）。
- `find-skills`：经 SkillHub 语义检索（EN+ZH 各 1 组）返回腾讯文档等通用生产力/办公类技能，与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，**未安装**（与历次一致）。

> **归档空档说明**：本档案覆盖 **2026-10-31** UTC 窗口；**2026-10-29** UTC 窗口因 2026-10-30 当日自动化未执行而无档案（即上一期 2026-10-29 覆盖 10-28、2026-10-31 档案覆盖 10-30、本期 2026-11-01 档案覆盖 10-31 之间，仅 2026-10-29 窗口一日空档）；**自 2026-10-30 窗口起已连续两期（2026-10-31 与 2026-11-01）无空档**。本目录不补写中间日期内容，各期仅按自身关注窗口采集。如需回溯 2026-10-29 窗口，请以各引擎官方提交 API 按日期区间重新采集。
> 本档案仅覆盖 2026-10-31 UTC 窗口；相邻日期请以官方提交 API 按日期区间重新采集。
