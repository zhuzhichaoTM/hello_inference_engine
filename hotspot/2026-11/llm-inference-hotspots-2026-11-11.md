# 大模型推理优化每日热点档案

> 档案日期：2026-11-11（本自动化于该日执行）
> 关注窗口：2026-11-10（周二）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（issues/pulls API，markdown，官方一手）+ HN Algolia（HTTPS，numericFilters 按 2026-11-10 epoch 窗口）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 77 / SGLang 93 / TRT-LLM 48 / llama.cpp 42 合计 260 条；四家单页均未触 per_page=100 上限（无截断）；releases API 确认窗口内**无新正式版本**（vLLM 仍为 v0.33.0rc3、SGLang v0.5.23、TRT-LLM v1.4.0rc3；llama.cpp 仅推进 nightly `b12259`/`b12261`/`b12262`，均 prerelease）；57+58+49+42 个合并 PR 经 pulls API 抓取正文（markdown 干净文本，含标题级/测试级/描述级标注）；HN Algolia 引擎关键词高信号弱（无四家官方更新类故事，仅社区/Show HN，二手线索/未核验）。本窗口为周二，提交量较周一（11-09：99/92/35/29）回落至 77/93/48/42，属正常工作日波动。

## 1. 版本状态与本期头号事件：FP8/低比特 KV cache 跨引擎标准化落地 + SGLang mem_cache 统一内存重构收尾 + 静默/投机解码正确性

**提交计数（2026-11-10 UTC，均 HTTP 200；2026-11-10 为周二，量较周一回落；vLLM 77 / SGLang 93 / TRT-LLM 48 / llama.cpp 42）**：
- **vLLM 77** / **SGLang 93** / **TensorRT-LLM 48** / **llama.cpp 42**。

**发布状态（窗口内无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-10 全部 77 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-10 含 mem_cache 统一内存重构收尾、HiCache 存储健壮性、DSv4.1 MegaMoE decode perf、AMD 量化栈、投机解码 dspark 密集提交。
- **TensorRT-LLM**：仍为 **v1.4.0rc3（True/prerelease，2026-11-06T21:38:32Z）**；11-10 含 Rubin 大合并回灌（#21067）、FP8 KV 扩展、Helix/DSpark 正确性、GEN 抢占。
- **llama.cpp**：仍为 **v0.9.0（2026-11-04T13:08:06Z 正式发布）**；11-10 推进 nightly `b12259`（22:11Z）/ `b12261`（22:50Z）/ `b12262`（23:04Z，prerelease），无新正式版本。

**本期头号事件（FP8/低比特 KV cache 跨引擎标准化落地 + SGLang mem_cache 重构收尾 + 静默/投机解码正确性）**：
1. **FP8/低比特 KV cache 跨引擎标准化落地（新增第 13 条观察）**：llama.cpp `#28392` 首秀 **OCP-FP8 KV-cache** 支持（运行时静态 per-tensor scales，可选，附转换/加载/context shift/正确性 E2E 测试）；TRT-LLM `#19396`（非-MLA `PrimsTSFmha` 的 FP8 E4M3 KV，context/decode/mixed 全支持、scale 在 CUDA Graph replay 中保持 live）+ `#20958`（跨注意力 FP8 KV 静默错 token / 非法访存修复）；vLLM `#66892`（MiniMax-M3 NVFP4 KV 兼容 `BLHNC` 布局，使 SimpleCPUOffload 按 block 而非 per-layer page 注册）。FP8 KV 从「MLA 专属」走向「跨架构/跨引擎通用 KV 压缩」主线确立。
2. **SGLang mem_cache 统一内存重构收尾（强化第 8 条）**：`#49300`/`#49301`/`#49299`/`#49309`/`#49310`/`#49311`/`#49312` 栈把统一内存的**容量决策从标量「最大适配搜索」改为按具体 demand 计价**（prefill 准入 / PD decode 预分配 / state slots / host load-back 全部走 `demand`），并让 **PP 各 stage 用同一组 simulated snapshot 回答容量问题**以消除 stage 间决策分歧；配套 **double-free 防护**（#49311，`SGLANG_INVARIANT_CHECK` 下每释放页设备端校验 bound）与**统一内存文档**（#49312）。属 KV 内存管理的架构级收口。
3. **HiCache 存储健壮性（强化第 8 条）**：SGLang `#47617`（HiCache 四个 storage worker 线程此前无人看守，异常逃逸导致**静默冻结**——host pool 填满、命中 parked 永不释放；本 PR 改为 rank 崩溃 fail-fast）+ `#47626`（storage backup 128-page 批次失败时仍按 `completed_tokens` 提前 ack，致未写尾段被误信已存、后续 prefetch 在该 batch 截断；改为**按 rank 约定的 written prefix 统一 settle**，避免 rank 间 ack 序列失同步）。均属 KV 复用/容量正确性的静默债务清理。
4. **静默 / 投机解码正确性铁证再现（强化第 4/5/6 条）**：llama.cpp `#32015`（图采样链未全部 `ggml_set_output`，导致 `-np 2` 并发下**草稿 token 被别名缓冲静默损坏**、eagle3 接受率 0.15 vs 0.40）；TRT-LLM `#20958`（跨注意力 FP8 KV 池按 16-bit 地址、实际按 8-bit 分配，池不足 2× 即非法访存或**静默错 token**）；TRT-LLM `#22051`（Helix + DSpark 下 draft KV block reuse 与 drafter 私有 ctx arena 错配 → **静默掉接受长度**）；SGLang `#48117`（语法拒绝的草稿 token 触发 `ValueError` 未捕获 → **崩溃整个 decode 引擎**，单条结构化输出请求拖垮 PD 部署）；TRT-LLM `#21923`（Triton FP8 MoE align 仅支持 2 幂专家数，非 2 幂被拒，72 专家等形状正确性修复）。
5. **AMD/ROCm 与量化 MoE 后端延续（强化第 4/12 条）**：vLLM `#66644`（DeepSeek-V4.1 在 ROCm 接 `aiter_mega_moe` 后端，MI355X DP8/EP8 GSM8K 0.9735 持平 aiter 0.9704，强化第 12 条 AMD MegaMoE 路径）+ `#63893`/`#64153`/`#61498`（ROCm DSv4.1 Mono decode 调优）+ `#61942`（ROCm 短 dense-MHA prefill 跳过 DSA indexer 计分，GLM-5.1-FP8 TTFT −10.7%@c1）；SGLang `#47398`（gpt-oss W4A8 MoE 在 gfx1250：Gluon 在 <320 token 快 1.4–2.3×、FlyDSL 在 ≥1024 token 快 1.7×）+ `#47371`（muse-spark ROCm day0）+ `#42119`（ROCm 启用 GPTQ/AutoRound INT4 检查点，新增 Triton W4A16 GPTQ kernel）+ `#49659`（MXFP8 权重 JIT 量化）。

---

## 2. vLLM：2026-11-10 提交（77 条）—— FP8 KV 布局兼容 + ROCm DSv4.1 MegaMoE/Mono decode + GLM-5.3 专用 decode + 量化(Humming SiLU)/投机解码 PD

### 2.1 `#66644` [ROCm][DSv4.1] Wire AITER MegaMoE into the V4.1 model（AMD MegaMoE 接入，强化第 12 条）
- **主题**：在 ROCm（gfx950，DP+EP + MoRI）上为 DeepSeek-V4.1 启用 `--moe-backend aiter_mega_moe`；V4 MoE 基类已建 AITER MegaMoE experts、`config_utils` 已接受 `DeepseekV41FP8Config`，但 V4.1 模型硬编码 `use_mega_moe=False` 使该后端无法使用。
- **内容详情**：仅在该 backend 下走新路径，其他 backend 不变。验证：MI355X×8、DSv4.1-Flash、DP8/EP8、`--all2all-backend mori`、DSpark 5 draft tokens + block rejection；GSM8K 5-shot：`aiter` 0.9704±0.0047 / `aiter_mega_moe` 0.9735±0.0044，acceptance 3.93 vs 3.95。
- **应用场景**：DeepSeek-V4.1-Flash 在 AMD MI355X 的 DP+EP 量化 MoE 服务。
- **研判**：【事实】PR 给两 backend GSM8K 对照（0.9735 vs 0.9704）与 acceptance（3.95 vs 3.93）。【观点】属「新旗舰 MoE 量化后端 AMD 落地」（第 12 条）——把 AITER MegaMoE 接到 V4.1，精度持平；描述含 FlyDSL 32-bit buffer 限制等历史包袱，记为官方一手（含 GSM8K 对照）。**证据等级**官方一手（含 GSM8K 对照）。
- **来源**：vllm-project/vllm #66644（2026-11-10T05:21:40Z 合并）。

### 2.2 `#65822` [Perf][KV Connector][ROCm] MoRIIO WRITE: skip blocks the decoder prefix cache already holds（KV Offload/Prefix 复用，强化第 8 条）
- **主题**：MoRIIO WRITE 模式下 prefiller 把每个 prompt block 重写进 decoder（含 D 已持有的 prefix cache 副本）；P 现跳过这些块（与 NIXL push / Mooncake 一致；#62190/#48471 仅改 READ 模式）。
- **内容详情**：仅 single-group WRITE。MiniMax-M3.1-Flash 1P1D MI355X：每请求 P 写入 1,650 MB→130 MB、P write-thread busy 79%→32%；3600 s agentic replay @c48：35,639→38,741 tok/s/GPU（+8.7%）、p90 interactivity 157.8→149.6。跳过块为 D 的本地 prefix 副本，无精度评测。
- **应用场景**：vLLM MoRIIO WRITE 模式下的 P/D KV 传输去重。
- **研判**：【事实】PR 给 P 写入 1650→130 MB、busy 79%→32%、tok/s/GPU +8.7%。【观点】属「KV 卸载/Prefix 复用去重」（第 8 条），跳过 D 已缓存块省传输与线程占用；单 D 时省下的时间变成更大 decode batch，故不主张 1P1D Pareto。**证据等级**官方一手（含 MI355X 对照，无精度评测）。
- **来源**：vllm-project/vllm #65822（2026-11-10T05:23:59Z 合并）。

### 2.3 `#63102` [Perf][GLM-5.3-Flash] Specialized decode for GB300 TP4（GLM-5.3 专用 decode， perf）
- **主题**：为 GLM-5.3-Flash 在 GB300 TP4 加 opt-in 专用 decode 路径（同 Kimi K3 #61867 / DSV4.1 #61868）：每子层 CuTe DSL kernel、融合 all-reduce + mHC boundary、按 token 数选实现的 policy 表；decode 与 uniform MTP verify（≤64 token）走专用路径。
- **内容详情**：`--additional-config '{"decode_backend":"latency_specialized"}'`。GB300×4、TP4、FP8 KV、`FLASHINFER_MLA_SPARSE`、MRV2、无 prefix caching：median ITL c1 5.15→2.93 ms（1.76×）、c64 13.92→12.00 ms（1.16×）；MTP3 c1 1.33×、c16 1.23×。GSM8K 0.918（stock 0.914）、MTP3 0.915（stock 0.912）。不满足要求（GB300 族/TP4 无 PP/DP/EP/DCP/MRV2 FULL decode graph/FP8 KV）时启动即拒绝。
- **应用场景**：GLM-5.3-Flash 在 GB300 的低延迟 decode。
- **研判**：【事实】PR 给 ITL 表（c1 1.76×、c64 1.16×）与 GSM8K 对照。【观点】属「GLM-5.3 专用 decode 提速」，纯 perf、精度在 stock 范围内；强约束开关，不影响其他部署。**证据等级**官方一手（含 ITL/GSM8K 对照）。
- **来源**：vllm-project/vllm #63102（2026-11-10T07:08:41Z 合并）。

### 2.4 `#66652` [Perf][DSv4.1] CuTe DSL mHC boundary for decode batches up to 336 tokens（DSv4.1 mHC boundary，perf）
- **主题**：DSv4.1 decode 在 shifted mHC boundary 上每步约 80 次 Mega mHC launch（residual/Sinkhorn/RMSNorm 等），DeepGEMM `sm100_mega_mhc_impl` 每调用 13–19 µs 远超流量需求；本 PR 加 CuTe DSL kernel 覆盖最多 336 token 的 boundary。
- **内容详情**：16-CTA cluster 每 cluster 持 token tile 并分 hidden；CTA 经分布式共享内存交换 partial mix/sum-of-squares，整 boundary 单 kernel；mixer GEMM 的 A 片段从 L2 以 coalesced 布局预取 4 k-step，每 CTA 占半 SM，14 cluster 同跑、一波覆盖 336 token。每层注册非持久 buffer（`hc_attn_boundary_weight`/`hc_ffn_boundary_weight`），在 `process_weights_after_loading`（每 load/reload 路径均跑）in-place 填充，使 captured graph 读更新后权重。
- **应用场景**：DeepSeek-V4.1-Flash 在 Blackwell 的 mHC 边界融合 decode。
- **研判**：【观点】属「DSv4.1 decode 边界融合」，移植自 TRT-LLM #21438 的 mHC boundary 并扩到 336 token、单 kernel 化；性能数字描述级未给完整表，记为官方一手（perf 级，描述级）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：vllm-project/vllm #66652（2026-11-10T08:26:48Z 合并）。

### 2.5 `#66660` [Perf][EPLB] Copy rebalanced experts back from the buffer in one launch per weight（EPLB 重排 copy-back，perf）
- **主题**：异步 EPLB 在重排时每 MoE 层 `move_from_buffer` 对每移动 expert 行发一个 `copy_`（含复制行），多行时每层数 ms host 时间且走 copy-engine 带宽；改用一个 Triton launch 拷贝某 weight 张量的所有移动行。
- **内容详情**：GB200 上每层 copy-back host 时间 0.44–6.1 ms→0.20–0.30 ms（≥8 行移动时）、GPU 时间 1.8–4.7×；copy-back 后权重与 main 字节一致；无服务期 JIT（`EplbState.add_model` 编译模型权重布局 kernel）；CPU/XPU 张量保留逐行 `copy_` 路径。
- **应用场景**：vLLM 异步 EPLB 重排时的 expert 权重回拷。
- **研判**：【事实】PR 给 host 0.44–6.1→0.20–0.30 ms、GPU 1.8–4.7× 与字节一致验证。【观点】属「EPLB 重排性能」，单 launch 合并回拷降 host 时间与 GPU 时间；记为官方一手（含 GB200 对照）。**证据等级**官方一手（含 GB200 对照）。
- **来源**：vllm-project/vllm #66660（2026-11-10T08:27:35Z 合并）。

### 2.6 `#61942` [ROCm][Perf] Skip DSA indexer scoring for short dense-MHA prefills（ROCm DSA indexer 跳过，perf）
- **主题**：ROCm `deepseek_v32`（DeepSeek-V3.2 / GLM-5.x）对纯短 prefill（dense MHA）跳过 sparse indexer 的 Q 投影与计分；CUDA 已跳过（#48407/#51298），现两平台同判据。
- **内容详情**：GLM-5.1-FP8、MI355X、TP4：1K-token prefill median TTFT −10.7%@c1、−5.0%@c32，attention 输出与 KV 位逐位一致；gsm8k 不变、GPQA-Diamond 0.691（main/PR 各 8 runs）；GLM-5.2-MXFP4（21/78 层开 indexer）无显著变化。长于 `index_topk`/MTP/FULL cudagraph/capture 仍计分；其他平台不变。
- **应用场景**：ROCm 上 GLM 系列短 prefill 的 DSA indexer 去冗余。
- **研判**：【事实】PR 给 GLM-5.1-FP8 TTFT −10.7%@c1、KV 位逐位一致、GPQA 0.691 持平。【观点】属「ROCm DSA 短 prefill 提速」，位精确、无回归；记为官方一手（含 MI355X 对照）。**证据等级**官方一手（含 MI355X 对照）。
- **来源**：vllm-project/vllm #61942（2026-11-10T12:14:01Z 合并）。

### 2.7 `#65975` [Perf][Quantization] Use fast SiLU intrinsics in Humming MoE activations（Humming MoE 量化激活提速，强化第 12 条）
- **主题**：在 vLLM 传给 Humming `process_input`（融合 MoE w2 激活 + 输入量化的 kernel）的 SiLU 族激活表达式里用 fast SiLU（`__fdividef` + flush-to-zero `ex2.approx.ftz.f32` exp）；Humming 该 kernel 关 fast math，SiLU 原编为精确 `expf` + IEEE 除法。仅改激活表达式，Humming 的量化/Hadamard/scale 数学保持精确 flag。
- **内容详情**：B300 上 prefill 尺寸 w2 activation+quant kernel 快 16–30%，DeepSeek-V4.1-Flash clamped SwiGLU 至多 ~40%；gpt-oss-20b 端到端 prefill +1.9%（MXFP8）/+1.7%（FP8 per-token）。覆盖 SILU/SILU_NO_MUL/clamped SiLU/SITU/SWIGLUSTEP；GELU/tanhf 不变。GSM8K 噪声内不变；量化输出 MXFP4/MXFP8 位一致、FP8 差 ~1e-6 字节。
- **应用场景**：Humming 量化 MoE 后端（MXFP4/MXFP8/FP8）的 w2 激活+量化提速。
- **研判**：【事实】PR 给 B300 kernel 表（bf16 137.6→115.4 µs、fp8 block-128 156.5→118.4 µs）与 gpt-oss 端到端 +1.9%/+1.7%。【观点】属「量化 MoE 后端激活提速」（第 12 条），用 fast math 换 16–40% kernel 提速且量化字节位一致；记为官方一手（含 B300 kernel 表 + 端到端）。**证据等级**官方一手（含 kernel + 端到端对照）。
- **来源**：vllm-project/vllm #65975（2026-11-10T15:42:00Z 合并）。

### 2.8 `#65217` [Spec Decode][MRV2] Add prefill_context_kv_only for P/D prefill instances（MRV2 投机解码 P/D，强化第 5 条）
- **主题**：P/D prefill 实例上基于 DFlash 的 drafter（DFlash/DSpark/DFlash2/XPress）只需存 decode 实例读取的 context K/V；本 PR 加 `prefill_context_kv_only`：prefill 实例不排草稿、不预留 lookahead KV 槽、不 capture 草稿 CUDA graph，并对命名 context K/V 模块的 draft 模型释放仅起草读取的权重。该实例上 decode 的请求不带投机。
- **内容详情**：显式设置（因 `kv_role` 不识别 prefill 实例：`kv_both` 也经 MultiConnector 加载/卸载 KV store）。沿用 #62784（per-step K=0）与 #56957 的 lookahead/speculator-flag 处理；保留 `num_speculative_tokens` 不变（其还标定 decode 实例加载的 state 如 Mamba state）。DFlash/DFlash2/DSpark/XPress 等声明支持。
- **应用场景**：MRV2 + 投机解码的 P/D prefill 实例瘦身。
- **研判**：【观点】属「投机解码 × P/D 资源瘦身」（第 5 条），prefill 实例不再为草稿预留 KV/权重/图，降显存与启动开销；记为官方一手（feature/config 级）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #65217（2026-11-10T17:01:37Z 合并）。

### 2.9 `#65704` [Bugfix][GLM-5.3] Build the MTP eh_proj low-latency GEMM plan（GLM-5.3 MTP eh_proj，强化第 4/5 条）
- **主题**：CUDA 上 `DeepseekV32MultiTokenPredictorLayer`（glm_moe_dsa 的 MTP draft 层）从不构建 `eh_proj` 低延迟 GEMM plan——其检查 draft config 的 `model_type`，而 `SpeculativeConfig.hf_config_override` 把它从 `glm_moe_dsa` 改写为 `deepseek_mtp`；应改判 target model type（同 `DeepseekV32MTP`）。
- **内容详情**：SM100/SM103/SM107 + BF16 dtype 下，draft 的 6144×12288 `eh_proj` 现得 GLM 低延迟 plan：SM100/SM107 上 M 1–4 走 CuTe skinny GEMM、SM103 上 M 1–3；其他 M 走 cuBLAS。修前 plan 从不构建、`eh_proj` 恒走 `nn.Linear`（cuBLAS）。加 CPU 单测（mock config）验证 `build_glm52_plan` 被调用。
- **应用场景**：GLM-5.2/GLM-5.3 的 MTP 投机解码 `eh_proj` 低延迟路径。
- **研判**：【事实】PR 给「修前 plan 从不构建、eh_proj 恒走 cuBLAS；修后 SM100/103/107 得 GLM plan」与单测 5 passed/1 failed→pass。【观点】属「MTP/投机解码低延迟路径修复」（第 4/5 条），模型类型改写导致的 plan 漏建，非静默错误但是性能/正确性漏配；记为官方一手（fix 级，含单测）。**证据等级**官方一手（fix 级，含单测）。
- **来源**：vllm-project/vllm #65704（2026-11-10T18:19:37Z 合并）。

### 2.10 `#66892` [Attention][MiniMax-M3] Allow the BLHNC KV cache layout with an NVFP4 KV cache（MiniMax NVFP4 KV 布局兼容，强化第 13 条）
- **主题**：让 MiniMax-M3 MSA NVFP4 backend 支持 `BLHNC` KV 布局（此前仅声明 `LBHNC`，使 NVIDIA 上 `--kv-cache-dtype nvfp4` + `VLLM_KV_CACHE_LAYOUT=BLHNC` 在布局解析失败）。
- **内容详情**：`MiniMaxM3SparseMSANvfp4Backend` 接受 `BLHNC`（并列 `LBHNC`，默认不变）。`BLHNC` 下 SimpleCPUOffloadConnector 每 rank 注册一个 block-strided region、每 block 拷一行，而非每层一页（MiniMax-M3.1-Flash TP4 约 120 页 9–18 KB）；GB200 TP4 + NVFP4 KV + `BLHNC` + SimpleCPUOffload：GSM8K 两遍（存 CPU 95.53/95.38、从 CPU 载 95.30/95.07），第二遍 93% 外部 prefix 命中。
- **应用场景**：MiniMax-M3.1-Flash 在 NVFP4 KV + `BLHNC` 布局下的 KV 卸载。
- **研判**：【事实】PR 给 GB200 GSM8K 两遍（存/载）与 `BLHNC` 注册方式。【观点】属「低比特 KV 布局兼容扩展」（第 13 条），`BLHNC` 使 NVFP4 KV 能走更省的块级卸载；记为官方一手（含 GB200 对照）。**证据等级**官方一手（含 GB200 对照）。
- **来源**：vllm-project/vllm #66892（2026-11-10T22:45:40Z 合并）。

### 2.11 `#66370` [Bugfix][Core] Re-prepare model state after extensible KV cache extension（Kimi-K3 AMD 专用 decode，强化第 4 条）
- **主题**：启用 extensible KV cache（#56492，MRV2 默认开）后，Kimi-K3 在 AMD 上 `decode_backend=latency_specialized` 的首个 eager decode step 启后即崩；本 PR 加 `ModelState.rebind_kv_cache` hook，`extend_kv_cache` 重绑 KV views 后调用，Kimi-K3 AMD model state 在其内重跑 `prepare_kv()`。
- **内容详情**：在验证 #63486（该 PR 在 Kimi-K3 引擎测试里禁 extensible KV 直到本 PR 落地）时发现。修前每 TP rank 抛 `RuntimeError: Kimi-K3 specialized decode: the KV cache was re-bound without prepare_kv`；修后 hook 仅默认空操作、Kimi-K3 重写 `prepare_kv`。加真实 Qwen3-0.6B 引擎测试（hook 恰好跑一次）。
- **应用场景**：Kimi-K3 在 AMD（MI355X）上 extensible KV cache + 专用 decode。
- **研判**：【事实】PR 给修前 RuntimeError 与修后 hook 恰跑一次 + 8×MI355X Kimi-K3 e2e parity 测试。【观点】属「AMD/异构后端 + extensible KV 的正确性耦合」（第 4 条），KV 重绑后未重跑 prepare 致崩溃，属架构衔接漏配；记为官方一手（fix 级，含 e2e 测试）。**证据等级**官方一手（fix 级，含 e2e）。
- **来源**：vllm-project/vllm #66370（2026-11-10T17:18:20Z 合并）。

### 2.12 `#66875` [ROCm][Bugfix] Key the FSE MXFP8 scale-layout test check on the selected backend（AMD MXFP8 scale 测试修正，测试级）
- **主题**：修 `test_fse_online_quantization_routed_shared` 使其校验**实际选中 backend** 的 MXFP8 scale 布局；3 个 `routed_mxfp8` case 在 gfx942（MI300）失败，因测试假设每个 MXFP8 backend 都存扁平 scales。仅测试改动。
- **内容详情**：gfx942 用 pytest 插件**模拟**（非真实 MI300 跑），`supports_mx()=False` 时 3 failed→pass；AITER_MXFP8(FlyDSL) 路径保留扁平布局校验。不改运行时行为；使 #66672 的 3 个 `--deselect` 可移除。
- **应用场景**：vLLM ROCm MXFP8 量化测试在 MI300 的修正。
- **研判**：【观点】属「AMD 量化测试修正」（测试级），解开 MI300 上 MXFP8 scale 布局假设；gfx942 为模拟非真实硬件跑，记为官方一手（测试级，模拟）。**证据等级**官方一手（测试级，gfx942 模拟）。
- **来源**：vllm-project/vllm #66875（2026-11-10T21:51:27Z 合并）。

### 2.13 本期其余 vLLM 要点（标题/正文级）
- **ROCm / AMD**：`#63893`（ROCm DSv4.1 Mono: tuned K1 at TP4）、`#64153`（ROCm Mono: batch TP8 K2 的 X8 copies、late push peer bases）、`#61498`（ROCm Kimi-K3 Mono decode layer MI355X vanilla decode）、`#66596`（ROCm AITER bump v0.1.29）、`#66737`（Revert ZenCPU zen5 测试）、`#66798`/`#66854`（ROCm CI：重启用 ROCM_AITER_UNIFIED_ATTN / 加 `attention/backend.py` 到 MI355 DPX MLA test group）。
- **量化 / MoE / 激活**：`#65177`（INT8 MoE per-channel weight scales 独立于 act quant 索引，#58570 修复）、`#63753`（Fast Start：D2D transport 播种编译缓存）、`#59025`（注册 cudnn backend for FlashInfer BF16 mm）、`#66654`（Deprecate：v0.37.0 移除 legacy grpc_server/api_server/run_batch）、`#66661`（Clean up MRv1 残留）。
- **KV Offload / HiCache / PD / 调度**：`#66544`（KV Offload 重试 lookup 后保留 confirmed junction）、`#66605`（KVConnector 让外部连接器声明 KV cache 内存注册）、`#65877`（Mooncake PD 与 Store 连接器拆为 sibling 包）、`#66678`（CI：offline examples 转 pytest + sharding）、`#61424`（Core Metrics：调度期收集 iteration details）、`#61498`（见 ROCm）。
- **GLM / MTP / 安全 / Frontend**：`#66693`（Misc：filename typing→types 归一）、`#66257`（Security：Kimi K3 模式要求 native chat encoding）、`#66380`（Rust Frontend：不重开 reasoning）、`#66528`/`#66529`（Rust Frontend：tool-call 参数 JSON 风格写入 / DeepSeek DSML 流式）、`#66744`/`#66798`/`#66187`/`#64742`/`#65028`（测试/CI/Rust frontend 多项）、`#55387`（CI/Build：默认 CUDA build 13.0→13.2）、`#58795`（CI：pinned mamba-ssm/causal-conv1d 预编译）、`#62345`（Security：muse tool-call 二次属性扫描 quadratic 修复）、`#62791`（Benchmark：openai-audio input tokens 从 server usage 取）。

---

## 3. SGLang：2026-11-10 提交（93 条）—— mem_cache 统一内存重构收尾 + HiCache 存储健壮性 + DSv4.1 MegaMoE decode + AMD 量化栈 + 投机解码 dspark

### 3.1 `#49321` [Perf][NVIDIA] dsv41: Fork the decode attention mHC mix at the MegaAttention core（DSv4.1 MegaMoE decode，强化第 12 条）
- **主题**：SM100 上 next attention input 的侧流 mHC mix 在子层输入处分叉，与全网格 wqkv_a/wq_b GEMM 串行；MegaAttention core（每 CTA 一行）使 SM 空转。本 PR（从 #49262 拆出，与 MegaMoE-input epilogue 独立）让 mix 的分叉等到 attention core 启动；无 core 时仍 fallback 分叉。
- **内容详情**：FlashMLA 下延迟 mix 会在 core 后串行（TP4 aggregated 曾丢 ~10% interactivity，故 fork 限 MegaAttention）。GSM8K DSv4.1-Flash GB300 1P1D DEP16：96.66/97.04（本改 + #49262 叠 fused decode post + BF16x3 side mix #48456）vs 96.66/96.82（基线）。GB300 disagg 6 DEP2 prefill + DEP32 decode 8 nodes、AgentX c3018、3600 s：总 token 吞吐/GPU 472.2k→474.1k（+0.4%）、interactivity 329.3→337.7 tok/s/user（+2.6%）。
- **应用场景**：DeepSeek-V4.1-Flash 在 Blackwell 的 MegaMoE decode 侧流 mix 融合。
- **研判**：【事实】PR 给 GSM8K 96.66/97.04（持平偏优）与 GB300 吞吐 +0.4%/interactivity +2.6%。【观点】属「DSv4.1 MegaMoE decode 融合」（第 12 条），把侧流 mix 错峰到 attention core 启动后、消除 SM 空转；记为官方一手（含 GSM8K + GB300 对照）。**证据等级**官方一手（含 GSM8K + GB300 对照）。
- **来源**：sgl-project/sglang #49321（2026-11-10T21:16:41Z 合并）。

### 3.2 `#49595` [Perf] dspark: opt-in MXFP8 draft lm_head on SM10x（dspark 量化 draft lm_head，强化第 12/5 条）
- **主题**：DSpark 每 decode 步把 draft hidden 经 target bf16 lm_head 投影（DeepSeek-V4.1 每 rank 224 req 时每步一个 672×129280×5120 bf16 GEMM，cuBLAS，约步的 1.6%）；draft logits 仅引导提案、target verify logits 另行计算，故该 GEMM 精度不影响输出。
- **内容详情**：新 server arg `--speculative-dspark-lm-head-dtype {fp8,mxfp8}`：`mxfp8`（Blackwell，lm_head 权重一次性量化 MXFP8 缓存、每调用 FlashInfer CuTe DSL `mm_mxfp8` 量化 hidden 行）；`fp8`（gfx95x，既有 `torch._scaled_mm`，原由 `SGLANG_ROCM_OPT_DSPARK_FP8_LM_HEAD` 门控）。启动期校验平台（非 Blackwell 要 mxfp8 / 非 gfx95x 要 fp8 即 raise）；移除旧 env 改 deprecated warning。运行时门：floating-point head、K%32==0（mxfp8）/K%16==0（fp8），否则走 plain matmul。
- **应用场景**：DSpark 投机解码在 Blackwell/ROCm 的 draft lm_head 量化。
- **研判**：【观点】属「投机解码 × 量化」（第 5/12 条），draft lm_head 精度不影响输出、量化仅提速；新增 Blackwell MXFP8 路径、统一定义 ROCm fp8 路径；记为官方一手（feature/perf 级，性能描述级）。**证据等级**官方一手（feature 级，性能描述级）。
- **来源**：sgl-project/sglang #49595（2026-11-10T21:22:08Z 合并）。

### 3.3 `#42855` [Perf][NVIDIA] dspark: Opt-in fused int8 Markov walk kernel for bs <= 64 on Hopper（dspark int8 Markov walk，强化第 5 条）
- **主题**：DSpark 一次并行前向提 γ 个 draft token，再以串行 Markov walk 校正；folded draft graph 把 walk 跑成 γ 个独立 step（每步 bf16 GEMM 重读全部 W2 自 HBM）+ `MarkovSampleStep`。H100 上 γ=7 时 walk 每轮 269 µs@bs1、453 µs@bs64。
- **内容详情**：加 opt-in 替身（每 draft graph bucket ≤bs64）：单 persistent cooperative launch 跑全部 γ step；W2 的 int8 副本铺在 SM 上（多数驻 shared memory、余量从 L2 流）；bias 用精确 int32 tensor-core 点积；每步 token 跨 SM 以单次 atomic `{key,count}` 交换选出。walk 本身 3.3–6.5× 快于默认；端到端 client output 吞吐在 6 个真实 target/drafter 对上 +1.2–9.0%；greedy 输出不变、sampled 输出 TV ~1e-5/draft。
- **应用场景**：DSpark 投机解码在 Hopper 的 Markov walk 提速。
- **研判**：【事实】PR 给 walk 3.3–6.5× 与端到端 +1.2–9.0%（6 对）。【观点】属「投机解码 Markov walk 提速」（第 5 条），int8 副本 + cooperative launch 降 HBM 重读；opt-in，记为官方一手（含 Hopper 对照）。**证据等级**官方一手（含 Hopper 对照）。
- **来源**：sgl-project/sglang #42855（2026-11-10T14:04:15Z 合并）。

### 3.4 `#47398` [Feat][AMD] gpt-oss: Mixed W4A8 MoE on gfx1250（AMD W4A8 MoE，强化第 12 条）
- **主题**：在 gfx1250 上用 Mixed（FlyDSL + Gluon）MoE 策略，按 token 数可配置阈值切换。
- **内容详情**：gpt-oss-120b、TP1、layer 18 真实权重、CUDA graph 内（HBM 流式权重）逐层 MoE 时间（µs，3 pass 中位）：Gluon 在 <~320 token 快 1.4–2.3× FlyDSL（1 token 36=Gluon=Triton，128 229 vs 345）；FlyDSL 从 1024 token 起快，3072 时 758 vs Gluon 1015（1.7×）；Triton 仅 1 token 与 Gluon 持平、4 token 起慢 1.5–2.3×。TP2 同形。
- **应用场景**：gpt-oss 在 AMD gfx1250（MI455X 等）的 W4A8 MoE 内核选型。
- **研判**：【事实】PR 给 Gluon/FlyDSL/Triton 三路逐 token 数时延表（Gluon 小 batch 胜、FlyDSL 大 batch 胜）。【观点】属「AMD 量化 MoE 内核选型」（第 12 条），按 token 数阈值切换 Gluon/FlyDSL；记为官方一手（含 gfx1250 时延表）。**证据等级**官方一手（含 gfx1250 时延表）。
- **来源**：sgl-project/sglang #47398（2026-11-10T14:06:15Z 合并）。

### 3.5 `#47371` [Feat][AMD] muse-spark: Add the ROCm day0 implementation（muse-spark ROCm day0，feature）
- **主题**：muse-spark 的 ROCm day0 实现（#44667 与 main 0f3f6198 合并、堆叠 ROCm 优化、解冲突）。
- **内容详情**：PR 正文为模板（Motivation/Modifications/Accuracy/Speed/Checklist 均为空占位），无具体性能/精度数字——**记为描述级/模板级（空正文）**。
- **应用场景**：Muse Spark Open 模型在 AMD ROCm 的 day0 支持。
- **研判**：【观点】属「AMD/国产后端 day0 支持」，但 PR 正文为空模板、未给实现细节或数字，按描述级/标题级记录，不将其能力声称当作已验证事实。**证据等级**官方一手（描述级/空模板，标题级）。
- **来源**：sgl-project/sglang #47371（2026-11-10T20:49:27Z 合并）。

### 3.6 `#42119` [Feat][AMD] quantization: enable GPTQ and AutoRound INT4 checkpoints（AMD GPTQ/AutoRound INT4，强化第 12 条）
- **主题**：AutoRound/GPTQ 检查点（如 `Intel/Qwen3.6-27B-int4-AutoRound`）在 ROCm 加载失败（降显存、本地服务常用）；本 PR 在 ROCm RDNA/CDNA（gfx1151/gfx950 验证）启用。
- **内容详情**：两处阻塞：① ROCm device capability 误选 CUDA-only Marlin 路径；② 跳过 Marlin 后无支持的 dense GPTQ kernel。修复：Marlin 选择限 CUDA；ROCm dense GPTQ 走新 Triton 路径、MoE 走既有 `MoeWNA16`；加 Triton W4A16 GPTQ kernel（含 load-time repacking 与 gfx1151 tuned config）、`GPTQTritonLinearKernel`（对称/非对称 zero-point，GPTQ v1/v2）；传播 AutoRound `sym`；加 load-path 与 GEMM 测试。支持 4-bit、fp16/bf16、`desc_act=False`、group −1/32/64/128/256。
- **应用场景**：ROCm 上 GPTQ/AutoRound INT4 检查点（本地 LLM 服务）加载。
- **研判**：【事实】PR 给根因（Marlin 误选 + 缺 dense GPTQ kernel）与新增 Triton W4A16 GPTQ kernel/测试。【观点】属「AMD 量化检查点支持」（第 12 条），补齐 ROCm 的 INT4 dense GPTQ 路径；记为官方一手（feature 级，含测试）。**证据等级**官方一手（feature 级，含测试）。
- **来源**：sgl-project/sglang #42119（2026-11-10T22:42:30Z 合并）。

### 3.7 `#47617` [Fix] hicache: keep the storage ack sequence on backend failures（HiCache 存储健壮性，强化第 8 条）
- **主题**：HiCache 四个 storage worker 线程（prefetch query/IO/sync、backup）每 rank 启动且无人看守；最近特性（hybrid path #20457 及其 follow-ups）在 transfer 前后加了调用点——unified host layout lease（acquire 中毒、release 抛）、trailing sidecar chain（`_sync_trailing_keys`/`_resolve_sidecar_nonkv_derived_pool_transfers`）、#37976 的 trace 初始化——使异常可逃逸某 worker 且无人察觉。
- **内容详情**：症状为**静默冻结**（host pool 填满、命中 parked 永不释放、默认 `timeout` 策略下请求持续准入无 L3 收益；`wait_complete` 下永不准入），无任何日志；hybrid controller 上跳过的 sidecar ack 还使本 rank ack 序列与 peer 失同步。响应改为 **crash the rank**（逃逸异常属罕见事件，保活 worker 靠臆造更糟）。属 KV 复用/容量正确性主线。
- **应用场景**：HiCache 存储 worker 异常逃逸的 fail-fast 加固。
- **研判**：【观点】属「HiCache/KV 复用健壮性」（第 8 条），worker 异常逃逸致静默冻结、无日志，改为 rank 崩溃 fail-fast 以避免静默 hang；无 field report、属防护性加固，记为官方一手（fix 级，描述级）。**证据等级**官方一手（fix 级，描述级）。
- **来源**：sgl-project/sglang #47617（2026-11-10T18:12:07Z 合并）。

### 3.8 `#47626` [Fix] hicache: settle storage backups at the rank-agreed written prefix（HiCache 存储 settle，强化第 8 条）
- **主题**：HiCache storage backup 按 128-page 批次写；批次失败则 `_page_backup` 停、操作按 `completed_tokens` 不足 ack（#39444 两侧均现）。tree 的 ack drain 把操作当完成：buffer 模式喂每 key 存在信念并释放 staging；cache 模式释放 node host lock。未写尾段被信为已存、无人重写，后续该 prefix 的 prefetch 在失败 batch 截断。
- **内容详情**：为何必须 rank-agreed：backup 提交与 ack 无自身 collective；drain 按 ack-queue 大小 MIN 跨 rank 配对操作，write-through 触发读各 rank 自身 tree 状态；某 rank 写失败（各 rank 写自身 KV shard、replicated MLA 仅 writer rank 写）若各按自身 `completed_tokens` settle 则信念分歧、write-through 触发不一致、positional ack 配对永久失同步。本 PR 范围刻意小：drain 时每 rank 以单一 packed M… 同意每 acked backup 的 written prefix。
- **应用场景**：HiCache 存储 backup 失败的命中率修复（非可靠性）。
- **研判**：【观点】属「HiCache 存储正确性」（第 8 条），backup 批次失败致尾段被误信已存、命中率受损；改为 rank-agreed written prefix settle，属 hit-rate 修复；记为官方一手（fix 级，描述级）。**证据等级**官方一手（fix 级，描述级）。
- **来源**：sgl-project/sglang #47626（2026-11-10T18:12:23Z 合并）。

### 3.9 `#49355` [Feat] linker: Support Kimi-K3.1 DSA, MLA draft and DCP pools（Kimi-K3.1 linker 池，强化第 5/11 条）
- **主题**：Kimi-K3.1 配 DSA attention（MLA KV + indexer）与 KDA linear attention；`main` 的 direct external linker 的 FULL+MAMBA 池组仅接受纯 MLA 池、拒 DCP 且无 MLA draft sidecar，使 `--enable-unified-cache-external-linker` 拒 Kimi-K3.1；本 PR 移植 Day-0 分支 #48414 的 linker 池支持。
- **内容详情**：DSA 的 indexer 作为独立池（每 KV page 一行；shared-topk 层留零行占位并跳过）；MLA draft KV 池（DFLASH draft）作 `DRAFT` sidecar；跨 attention TP 字节一致的池（MLA KV/indexer/MLA draft）去 TP rank 键、仅 TP0 写；TP-sharded KDA state 保 per-rank 键；DCP 的 KV/indexer/DCP-collapsed draft 映射本 rank 份额、加 DCP rank 键、首 DCP block 写。
- **应用场景**：Kimi-K3.1 的 DSA/MLA/DCP 跨节点 KV 外部 linker。
- **研判**：【观点】属「Kimi-K3.1 上下文并行/解码栈 + PD linker」（第 5/11 条），把 DSA/indexer/DCP 池接入外部 linker；DFLASH draft KV ring 与 `cutedsl_mla` draft backend 留待 #49110 移植，记为官方一手（feature 级，部分未齐）。**证据等级**官方一手（feature 级，部分未齐）。
- **来源**：sgl-project/sglang #49355（2026-11-10T22:57:05Z 合并）。

### 3.10 `#48117` [Fix] spec: Prune the draft branch instead of crashing（投机解码 语法拒绝 健壮性，强化第 5 条）
- **主题**：`traverse_tree`（DSPARK/DFLASH 等投机解码建草稿树 grammar bitmask）对每个过父 mask 行的 draft token 调 `grammar.accept_token()`；matcher 仍可拒一个（mask/matcher 失同步，如 PD retracted-request double-accept），`XGrammarGrammar.accept_token` 抛 `ValueError: Tokens not accepted`——**未捕获 → 传播出调度器事件循环 → 杀引擎**。
- **内容详情**：实测于 PD 解聚 DeepSeek 部署跑 DSPARK + xgrammar JSON-schema 结构化输出：单条结构化输出请求拖垮整个 decode 引擎（Scheduler hit an exception → 进程退出 → liveness probe 失败 → pod 重建）。修复：`dfs` 中捕获 `ValueError`——在**父 mask 行清除该 token**（使 verify 步不接受、mask 与 matcher 一致）、**剪枝该 branch**（不 fill/children，无需 rollback）；兄弟分支继续验证、请求保持存活。
- **应用场景**：DSpark/DFLASH 等投机解码在结构化输出下的语法拒绝健壮性。
- **研判**：【事实】PR 给复现栈（ValueError 未捕获 → 杀引擎）与修法（清父 mask + 剪枝）。【观点】属「投机解码 × 结构化输出健壮性」（第 5 条），语法拒绝的 draft token 本就是投机解码已处理的情形，未捕获异常致引擎崩溃属高危缺陷；记为官方一手（fix 级，含复现）。**证据等级**官方一手（fix 级，含复现）。
- **来源**：sgl-project/sglang #48117（2026-11-10T22:17:37Z 合并）。

### 3.11 `#46548` [Fix] PD/PP: Fix consensus checker failure while using PP prefill（PD/PP 共识检查，fix）
- **主题**：开启 consensus checker 时，PP prefill + TP decode 部署在 **warmup 期间** prefill 进程失败（报错含 `max_new_tokens` 的 rank divergence）。
- **内容详情**：PR 正文为模板（Motivation 仅给失败消息样例、Modifications 空占位），无具体根因/修复数字——**记为描述级/标题级**。
- **应用场景**：SGLang PD 解聚 + PP prefill 的共识检查器 warmup 修复。
- **研判**：【观点】属「PD/PP 部署健壮性」，但 PR 正文为空模板、未给根因与修复数字，按描述级/标题级记录。**证据等级**官方一手（描述级/空模板，标题级）。
- **来源**：sgl-project/sglang #46548（2026-11-10T23:05:20Z 合并）。

### 3.12 mem_cache 统一内存重构收尾栈 `#49299`/`#49300`/`#49301`/`#49309`/`#49310`/`#49311`/`#49312`（统一内存 KV 架构收口，强化第 8 条）
- **主题**：延续 2026-11-10 档案（覆盖 11-09）的头号事件 `#48011`–`#48014` 统一内存 layout planner——本期把「容量决策」从标量「最大适配搜索」改为**按具体 demand 计价**，并补完 PP stage 一致性与防护。
- **内容详情**：① `#49300` 使 `--enable-unified-memory` 下每个容量决策（prefill 准入 / PD decode 预分配 / state slots / host load-back）都为一个具体 demand 计价，移除 largest-fit 搜索及其算术捷径/memo；② `#49301` 让 PP 每个 stage 在 PP loop 起点 all-gather 几何（arena 字节、各 band unit 字节、page/id 计数），每个 demand 问题对**所有 stage 的 simulated snapshot** 提问（同一组共享逻辑单元），使各 stage 决策一致、无需额外通信（同 TP rank）；③ `#49299` 把 reused request row 的 state 在下一 forward 重置；④ `#49309` 让调度器在 forward 声明共享读结束后才改请求表与 state slot 虚→物映射——Mamba commit（ReplaySSM/KDA fold/EAGLE post-verify graph/Inkling conv-state/Qwen4-Exp PLE）现读 `forward_state_slots`，避免读到新 owner 而写错 state；⑤ `#49310` 在 PP loop 与每个 unified-memory device 上施加 write-after-read barrier；⑥ `#49311` 在 `SGLANG_INVARIANT_CHECK` 下对每释放页设备端校验 bound，防 **double free / 从未分配页释放** 静默进 free list（unified bands 懒释放、无 host 读）；⑦ `#49312` 补 `--enable-unified-memory` 的 usage + design 文档（置于 Advanced Features 邻 HiCache）。
- **应用场景**：SGLang 统一内存 KV 的容量/准入/分配/回收一致性与可观测性。
- **研判**：【观点】属「统一内存 KV 重构收口」（第 8 条）——把分散的容量标量搜索收为 demand 计价 + PP stage 一致 + double-free 防护 + 文档，消除跨 stage/跨 sub-pool 决策分歧与静默释放债务；`#49311` 直接堵「double free 静默进 free list」一类静默错误，记为官方一手（refactor/fix 级，含单测）。**证据等级**官方一手（refactor/fix 级，含单测）。
- **来源**：sgl-project/sglang #49299/#49300/#49301/#49309/#49310/#49311/#49312（2026-11-10 17:16–17:24Z 合并）。

### 3.13 本期其余 SGLang 要点（标题/正文级）
- **DSv4.1 / MegaMoE / 量化 perf**：`#45262`（fp8：SM12x 用 FlashInfer b12x 做 block-fp8-as-MXFP8 linears）、`#45264`（mhc：SM120 用 BLOCK_K=32 for hc_mix）、`#48731`（dsv41：SM103/SM107 每 program 聚 16 local Engram 行）、`#49139`（unquant：FlashInfer CuTe DSL mm_bf16 重落地 low-M BF16 autotune）、`#48928`（DSv4.1-Flash 用 segmented fastokens 做 prompt 编码）、`#49659`（量化：MXFP8 权重 JIT kernel 替代 triton_kernels）、`#49642`（megakernel CI 跑 SGLang GB300 runners）、`#49593`（diffusion H3-Image 部署性能文档）。
- **AMD / NPU / 量化**：`#49024`（wan2.2：FlyDSL norm/scale-shift 融合 high quality 门控）、`#47917`（NPU longcat2.0 arch22）、`#49004`（rocprofiler-sdk：gfx1250-rocm1010 TP 挂起修复）、`#49443`（AMD nightly-benchmark：InferenceX AgentX DSv4.1-Flash MI35x 曲线）、`#49410`（docker：Mega main nightly + pin releases）、`#49573`（bench：benchmark 加载的 tokenizer 查 chat template）。
- **健壮性 / 调度 / 其他**：`#48615`（diffusion：LTX breakable CUDA graph 内保 cuDNN attention）、`#41293`（gemma4：尊 OOT attention 默认）、`#49263`（cuda-graph：CUDA graph capture 期间暂停自动 GC）、`#49392`（sgl-router：StepFun 经 sglang-processor）、`#49192`/`#49606`（sgl-router：typed gRPC 媒体计数 / async workers 渲染分词）、`#49496`（调度器：单次调用在 owner list 间移请求）、`#49623`（调度器：每请求记其 holder list）、`#49405`（diffusion sana-video2：sequence parallelism）、`#49552`（diffusion：MiniMax-H3-Image）、`#49081`（cli：CLI Guide）、`#49577`（ci：jit-kernel Run 测试超时 90 min）、`#49423`（diffusion cancellation backpressure 确定性）、`#48776`/`#48777`（kimi-k3：thinking 对象映射 reasoning effort / chat 响应塑形为 K3 wire format）、`#43032`（Qwen3.8-Flash-Next：避免 padded PLE prefill 激活）、`#49647`（test：pin speculative_dspark_lm_head_dtype 到 resolvable-field 白名单）。

---

## 4. TensorRT-LLM：2026-11-10 提交（48 条）—— Rubin 大合并回灌 + FP8 KV 扩展（非-MLA/cross-attention）+ Helix/DSpark 正确性 + GEN 抢占 + 量化/AMD

### 4.1 `#21067` [feat] Rubin merge back: MLA, DSpark, DWDP, Helix（Rubin 大合并，架构级）
- **主题**：把 Rubin（SM107）相关的 MLA、DSpark、DWDP、Helix 等改动合并回主线（含 draft-cache 支持跨 cache managers/speculative workers/executor state/disaggregated transfer、DSpark 以 manager-owned pages 替代 worker-owned rolling windows 并加 paged attention + FP8 projection、FP4 MLA 加 packed-query/chunked prefill/wave-based planning、Helix 改 V-projection 放置与通信、FP8 block-scale all-reduce 融合、cache-domain API 变更、MoE event ordering/autotuner tactic 分发）。
- **内容详情**：PR 正文为 coderabbit 自动生成的 Dev/QA review（列出需核验项：draft-history lifecycle/slot reuse/position limits/cache sizing、DSpark page-table mapping/capacity/scale layouts/legacy fallback、FP4 MLA split bounds/task ordering/sync/output merge、Helix rank agreement/collective selection/shared-mem sync/CUDA-graph、FP8 block-scale enum/scale packing/padding/launch、cache-domain C++/binding/Python/type-stub 默认与枚举一致性、MoE waits 先于依赖工作 + rank 收到预期 tactic）。**未提供测试执行结果或 review-finding 计数**，且为 auto-generated，记为描述级/CI 未确立（大合并、需 main CI 全绿）。
- **应用场景**：NVIDIA Rubin（SM107）上 MLA/DSpark/DWDP/Helix 系列特性的主线收口。
- **研判**：【观点】属「Rubin 架构特性大合并回灌」，覆盖面极广（投机/draft cache/DSpark/FP4 MLA/Helix/FP8 block-scale/cache-domain）；正文为自动 review、无执行结果，风险在于大合并的跨模块一致性，记为官方一手（描述级/CI 未确立，大合并）。**证据等级**官方一手（描述级，CI 未确立，大合并）。
- **来源**：NVIDIA/TensorRT-LLM #21067（2026-11-10T02:39:52Z 合并）。

### 4.2 `#19396` [feat] Support FP8 KV cache for non-MLA PrimsTS attention（非-MLA FP8 KV，强化第 13 条）
- **主题**：为常规非-MLA `PrimsTSFmha` 的 context/decode/mixed 批次启用 FP8 E4M3 KV cache；融合预处理量化 Q，PrimTS 消费 FP8 Q/K/V 并写原生 BF16/FP16 输出；有效 QK/output scales 留 GPU 并在 CUDA Graph replay 中保持 live。
- **内容详情**：基于 PrimTS refresh #21354。context 预处理清未用 FP8 V 行（packed FP8 stores）；标准 dense 与 block-sparse PrimTS 共享 SM100/SM103/SM107 架构表；MLA 保留 SM100/SM103 路由。vendor lock/兼容 patch/sources 与 main 一致（canonical source `yuxianq/flashinfer:trtllm-prims-ts-dev-20261107`、Merged FlashInfer #8093 供 device global scales）。已与 `paged_v_tail_is_zero=True` 兼容。
- **应用场景**：TensorRT-LLM 非-MLA 模型（如标准 decoder-only）的 FP8 KV cache。
- **研判**：【事实】PR 给 FP8 E4M3 KV 跨 context/decode/mixed、scale 在 CUDA Graph replay 保持 live、复用 FlashInfer #8093 device global scales。【观点】属「FP8 KV 跨架构扩展」（第 13 条），把 FP8 KV 从 MLA 扩到通用 PrimsTS attention；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：NVIDIA/TensorRT-LLM #19396（2026-11-10T16:16:07Z 合并）。

### 4.3 `#20958` [#19861][fix] Fix FP8 KV cache for cross attention on the PyTorch backend（跨注意力 FP8 KV 静默错误，强化第 4/6 条）
- **主题**：encoder-decoder 模型在 PyTorch backend + `KvCacheConfig(dtype="fp8")` 下，cross-attention KV 池被越界使用——除非池持 2× 所需 block 否则进程以非法访存死或**静默返回错 token**。
- **内容详情**：两处缺陷：① `CrossAttention` 建 attention backend 不带 `quant_config`、建权重也不调 `update_quant_config`，使 op 的 quant mode 无 FP8-KV 位、`getKvCacheElemSizeInBits` 返 16，每 token 按 2 字节寻址，而池由同 config 按 1 字节/元素分配；读写共享翻倍 stride，故 2× block 的池可用、更小则越界。现 `CrossAttention` 传 `quant_config`/`skip_create_weights_in_init` 并 `update_quant_config`。② 修正后，cross attention 融合 context FMHA 从 paged cross cache 读 encoder K/V（`Q_PAGED_KV` 对 cross attention 强制），但 `fmhaParams.dataTypeKv` 仅在 paged-context FMHA 开时从 cache quant mode 设；16-bit 模型把 FP8 池当 bf16 读、解垃圾。现 `AttentionOp::initialize` 对 cross attention + 量化 KV cache 且非 paged-context FMHA 时 fallback 非融合 context 路径（从 16-bit workspace 读 K/V）。
- **应用场景**：TRT-LLM encoder-decoder 模型在 PyTorch backend 的 FP8 跨注意力 KV cache。
- **研判**：【事实】PR 给两处根因（KV 池 1 字节分配 vs 2 字节寻址、cross attention 非 paged-context 误读 FP8 为 bf16）与修复。【观点】属「量化 KV 静默错误」（第 4/6 条），FP8 KV 池地址位宽不一致致越界/静默错 token，典型量化静默债；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：NVIDIA/TensorRT-LLM #20958（2026-11-10T18:11:17Z 合并，Closes #19861）。

### 4.4 `#21582` [fix] Fix Helix MLA decode kernel selection（Helix MLA decode kernel 选择，正确性）
- **主题**：受支持的多 token Helix verify groups 会落入 CuTe DSL perf gate 之后到无法执行的 backend；单 token Helix decode 同问题。FlashInfer TRTLLM-Gen 与 PrimTS 不支持 Helix，留 fallback op；其 TRTLLM-Gen 生成把 33→63 头 pad 到 64-row Q tile（仅 precompiled kernel、32-token pages 存在）。Kimi K3 TP2/CP4（48 local heads）+ 64-token pages 在 SM100/SM103 上 FP8 或 BF16 KV 触发。
- **内容详情**：把这些正确性 case 准入 CuTe DSL（含 mixed context/generation batch 的 generation 部分）；其他单 token Helix head 数保留 perf gate（fallback JIT 任意 page size）；保留 exclusive-page-size fallback、metadata/dtype checks、最终 kernel capability check；kernel math 与已测性能表不变。
- **应用场景**：TRT-LLM Helix + Kimi K3 的 MLA decode kernel 选择。
- **研判**：【观点】属「Helix MLA decode 正确性」，perf gate 误把正确性 case 落到不可执行 backend；扩 CuTe DSL 准入范围修复，记为官方一手（fix 级，含 CPU routing 测试）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：NVIDIA/TensorRT-LLM #21582（2026-11-10T23:11:51Z 合并）。

### 4.5 `#22051` [feat] Guard DSpark block reuse under aggregated Helix（DSpark block 复用护栏，强化第 5 条）
- **主题**：聚合 Helix 下的 DSpark：block reuse 与 host KV tier 护栏（TRTLLM-17645）。
- **内容详情**：**Block reuse**——DSpark drafter 在 Helix CP 下当其 draft KV cache manager 加入 block reuse 但 drafter fallback 到私有 ctx arena 时 raise（此状态下 scheduler 把两池配在 reused prefix 上、drafter 读不到，静默掉接受长度）；`_check_helix_schedule_consistency`（`TLLM_HELIX_AGG_CHECK_SCHEDULE=1`）也比较 paired draft pool 的 committed depth（draft pool 非 CP-sharded，不同 rank 复用不同 draft prefix 会从不同 K/V 起草）。**Host KV tier**——recompute-pause 仅在仍有 GPU page 可驱逐时挂起 victim（context `_try_preempt_for_pages` + generation `_try_recompute_pause_for_gen`）；有 host tier 时挂起请求的页在 tier 有空时移至 GPU 下，撕掉它不释放 GPU 内存只费一次 re-prefill；检查（`KVCacheManagerV2.has_evictable_gpu_pages`）适用于任何带 host tier 的 V2 配置；移除 `_skips_auto_host_tier`，使聚合 Helix+DSpark 同其他配置得自动 host tier。
- **应用场景**：TRT-LLM 聚合 Helix + DSpark 投机解码的 block 复用与 host KV tier。
- **研判**：【观点】属「投机解码 × Helix 正确性/护栏」（第 5 条），draft KV block reuse 与 drafter 私有 arena 错配致静默掉接受长度、host tier 下 recompute-pause 误撕页；加 raise/check 与统一 host tier，记为官方一手（feature/fix 级，含单测）。**证据等级**官方一手（feature/fix 级，含单测）。
- **来源**：NVIDIA/TensorRT-LLM #22051（2026-11-10T23:35:09Z 合并）。

### 4.6 `#21923` [fix] Support non-power-of-2 expert counts in Triton FP8 MoE align（非 2 幂专家数 MoE 对齐，强化第 7 条）
- **主题**：`_moe_prefix_kernel` 原扫 `triton.next_power_of_2(num_experts)` lane，对非 2 幂专家数 Triton 拒绝；现 mask 超 `NUM_EXPERTS` 的 lane 的 load/store（那些 lane 对 padded-count sum 贡献 0），移除非 2 幂 range。
- **内容详情**：测试 shape 加 72-expert case（SM120 MoE correctness）；alignment reference 测试覆盖非 2 幂专家数；runtime/hardware 验证结果未提供（QA 判覆盖 sufficient，H100 CI 列该 selector）。
- **应用场景**：TRT-LLM Triton FP8 block-scale MoE 的非 2 幂专家数对齐。
- **研判**：【事实】PR 给根因（next_power_of_2 扫 lane 致 Triton 拒非 2 幂）与 masked load/store 修复、72-expert 测试。【观点】属「MoE 专家数正确性」（第 7 条），非 2 幂专家数被拒属专家路由/对齐正确性；runtime 验证未提供，记为官方一手（fix 级，CI 未完全验证）。**证据等级**官方一手（fix 级，CI 未完全验证）。
- **来源**：NVIDIA/TensorRT-LLM #21923（2026-11-10T05:07:11Z 合并）。

### 4.7 `#21190` [fix] CuteDSL MoE: keep inplace FC1 outputs out of the Rubin autotune key（CuteDSL MoE autotune，perf）
- **主题**：Rubin（SM107）下 locality-domain NVFP4 MoE FC1 op 的 autotune 以 12 输入调 `Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner`；输入 10/11 为两 partition 同写的共享 inplace 输出与 output scale-factor buffer。
- **内容详情**：runner 的 `TuningConfig` bucket 输入 7（`permuted_idx_to_expanded_idx`）并约束输入 0/2/5/6；未约束 10/11。`AutoTuner._find_nearest_profile` 把未约束维度逐字拷入 cache key（假设静态）。但 10/11 非静态——调用方按实际 permuted-token 数分配 `fc1_out`/`fc1_out_sf`，原始 count 重入 key 使 bucketing 失效：① 调优扫为每个（bucket, output-buffer size）对存一项而非每项；② 推理时同 bucket 仅当 buffer size 恰匹配才命中，否则 fallback 默认 tactic、调优结果废置；③ 同 bucket 重复项独立调优致分歧。修复：加 `ConstraintSpec(10, 0, infer_shape_max_num_permuted_tokens)` 与输入 11 的 `ConstraintSpec`（驱动新 `GatherGroupedGemmInputsHelper.infer_shape_locality_domain_output_sf_numel`），两维变为输入 7 bucket 的纯函数。
- **应用场景**：Rubin NVFP4 MoE FC1 融合的 autotune 命中率。
- **研判**：【观点】属「MoE autotune 正确性/perf」，inplace buffer 尺寸非静态却入 autotune key 致命中失败、调优结果废置（perf 回归风险，非静默错 token）；记为官方一手（fix 级，描述级）。**证据等级**官方一手（fix 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21190（2026-11-10T08:30:52Z 合并）。

### 4.8 `#21334` [feat] GEN slot preemption: pause decode requests in place（GEN 抢占，调度）
- **主题**：在 #20736 的 TTFT-percentile 准入队列之上加 opt-in GEN slot 抢占：当每个 decode slot 占满且某 waiter 有错过 TTFT 目标风险时，控制器暂停一 resident decode 请求并准入 waiter。
- **内容详情**：暂停请求保留 KV/recurrent state/DFlash/DSpark slots 而离 scheduled batch；resident pools 与内存估计保留暂停上限；CUDA-graph batch 上限不变。`tpot_budget` 偏好最长 resident 请求、控 mean pause-induced TPOT 膨胀；`target_output_tps` 偏好有输出率 slack 的请求；resume 优先于新准入；max-pause 过期强制 resume；PEFT pages 为暂停/被挤 decode 保留。固定 3/8 暂停驻留预算界内存开销（batch 128 时 48 暂停 resident）。配置：GEN worker 上恰设一字段（非空即开）。PP/attention-DP 拒绝留配置校验、同步限制文档化；3/8 暂停驻留预算界内存。
- **应用场景**：TRT-LLM 高并发下 decode slot 抢占以保 TTFT 目标。
- **研判**：【观点】属「调度器抢占/准入」，opt-in 的 GEN slot 抢占，属调度器能力扩展（与第 1 条调度器隐式阻塞互补而非冲突）；记为官方一手（feature 级，描述级）。**证据等级**官方一手（feature 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21334（2026-11-10T06:23:43Z 合并）。

### 4.9 `#21793` [feat] KVCM2: pool SINK_REQUIRED_ONLY lifecycles like REQUIRED ones（KVCM2 池生命周期，强化第 8 条）
- **主题**：`effectivePoolReusePolicy` 把 `SINK_REQUIRED_ONLY` 映射为 `REQUIRED` 用于 GPU 与 cold-tier 池分组；`OPTIONAL` 等保持独立分组键。GPU 分组仍计 per-pool 稀疏、cold-tier 仍计 cold-page size。
- **内容详情**：加两 C++ 单测（`SinkRequiredOnlyWindowsShareRequiredPools` GPU 分组、`SinkRequiredOnlyWindowSharesTheRequiredColdPoolGroup` cold-tier 分组 + 三 lifecycle 成两组）；无 integration test-list 变更（C++ 单测不需）。
- **应用场景**：TRT-LLM KVCacheManagerV2 的 sink-only 窗口池复用分组。
- **研判**：【观点】属「KVCM2 池生命周期/复用」（第 8 条），把 `SINK_REQUIRED_ONLY` 与 `REQUIRED` 同组以正确复用 GPU/cold 池；记为官方一手（feature 级，含 C++ 单测）。**证据等级**官方一手（feature 级，含单测）。
- **来源**：NVIDIA/TensorRT-LLM #21793（2026-11-10T09:21:43Z 合并）。

### 4.10 `#21208` [perf] Quantize large bf16 activations of static-FP8 linears with a Triton kernel（static-FP8 激活量化，强化第 12 条）
- **主题**：加 Triton E4M3 量化器（连续 BF16 输入）；`FP8QDQLinearMethod.apply` 仅在 PyTorch 非编译且输入 ≥ `1<<23` 元素时用之，其他输入保留 C++ 量化器。
- **内容详情**：源描述 reciprocal rounding + saturation；测试与 C++ op 逐位比（跨 shapes/scales、saturation/特殊值、BF16 vs FP16 阈值 dispatch、未变 `Linear` 输出）；测试列 `l0_h100.yml`。runtime/硬件验证未确立。
- **应用场景**：TRT-LLM 静态 FP8 线性层的大 BF16 激活量化提速。
- **研判**：【观点】属「量化激活提速」（第 12 条），大 BF16 激活的 Triton E4M3 量化替代 C++ path；逐位等价测试，runtime 验证未确立，记为官方一手（perf 级，CI 未完全验证）。**证据等级**官方一手（perf 级，CI 未完全验证）。
- **来源**：NVIDIA/TensorRT-LLM #21208（2026-11-10T17:14:24Z 合并）。

### 4.11 `#20848` [perf] MiniMax-M3 sparse attention: trim per-step host work（MiniMax 稀疏注意力 host 工作，perf）
- **主题**：减 MiniMax-M3 sparse attention 的 host 侧工作：slot mapping 与 MSA metadata 构造改用 NumPy；block-ID rows 拷 page-index views 并以零替代 `BAD_PAGE_INDEX` padding；NVFP4 sparse API resolver 与 fused-producer validation plans 缓存。
- **内容详情**：移除 `block_ids_from_views` 选项并更新其配置调用点；decode 测试加 BF16 zero-length-row + CUDA-graph replay 检查；MSA backend 测试加 CPU 动态 KV 长度/query-row metadata 覆盖；runtime-policy 测试比 block-ID rows 与 cache-index rows（含 padding/dtype/pinning）；MiniMax-M3 测试加 producer-plan reuse/invalidation/unsupported-layout/capacity/cache-bound；`l0_b200.yml` 加 4 horizontal-producer plan-cache 测试。性能测量仅覆盖 slot mapping 与 MSA metadata 构造；block-ID 拷贝与 producer-plan 缓存的隔离测量未提供；current-head CI/build/test 未提供。
- **应用场景**：TRT-LLM MiniMax-M3 稀疏注意力的 host 侧开销缩减。
- **研判**：【观点】属「稀疏注意力 host 工作优化」，NumPy + 缓存降 host 开销；部分隔离性能测量未提供、current-head CI 未提供，记为官方一手（perf 级，描述级/CI 未提供）。**证据等级**官方一手（perf 级，描述级，CI 未提供）。
- **来源**：NVIDIA/TensorRT-LLM #20848（2026-11-10T19:26:09Z 合并）。

### 4.12 `#21841` [perf] Reduce Qwen4-Exp PLE padding and workspace（Qwen4-Exp PLE 缩减，perf）
- **主题**：加独立的 context 与 generation row 宽度；投机解码中拆分 mixed context/generation batch 并按 generation 宽度定 rollback candidates；加 guarded BF16 packed short-convolution path（context chunks）；不支持输入保留既有卷积 path。
- **内容详情**：packed-kernel guard 查 tensor 几何与长度，但**不校验** `req_indices`/`token_offsets`/`lengths` 描述同一 packed layout 或 state slots 互异（API 文档要求调用方保）；两单测文件改（uniform context-row 宽度、mixed spec workspace + accepted-prefix rollback、model-prepared mixed-step 等价、packed-kernel parity、不支持输入处理），列 `l0_b300.yml`，QA 判 sufficient；测试执行结果未确立。
- **应用场景**：TRT-LLM Qwen4-Exp 的 PLE（ple/engram）padding 与 workspace 缩减。
- **研判**：【观点】属「Qwen4-Exp PLE 优化」，独立 context/generation 宽度降 padding/workspace；guard 不校验 packed layout 一致性留隐患，记为官方一手（perf 级，CI 未确立）。**证据等级**官方一手（perf 级，CI 未确立）。
- **来源**：NVIDIA/TensorRT-LLM #21841（2026-11-10T19:53:34Z 合并）。

### 4.13 `#21491` [fix] Separate generation cancellation from receive retirement（解聚 生成取消 vs 接收退役，健壮性）
- **主题**：及时取消 model-idle GEN 请求，同时保留其 KV/AUX/本地 CUDA 资源至安全退役；逻辑取消绝不授权提前内存复用。
- **内容详情**：Transceiver/coordinator 保留已发布 receives 与未发布 CUDA fences、保留早前失败、保固定退役 deadline；Scheduler/executor 把保留内存暴露给本地死锁检查、保 TP 宽 idle/drain/shutdown 决策一致、把 KV-moving connector saves 推到物理退役后；失败取消通知重试但不重发成功投递；保留 #21508 的 model-active cancellation guards。新逻辑 GEN 取消限既有 generation-first FP4-MLA bridge；无新配置/context-first GEN 激活/重试策略/超时预算。
- **应用场景**：TRT-LLM 解聚（disaggregated）服务的生成请求取消与资源退役分离。
- **研判**：【观点】属「解聚请求取消健壮性」，把逻辑取消与物理退役解耦以避免提前内存复用（与 #21508/#20903 同族）；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：NVIDIA/TensorRT-LLM #21491（2026-11-10T21:42:35Z 合并）。

### 4.14 本期其余 TRT-LLM 要点（标题/正文级）
- **Rubin / Helix / DSpark / FP4 MLA**：`#21067`（见 4.1）、`#21423`（CuTe DSL MLA：SM107 准入 FP8 wide-head verify groups）、`#21708`/`#21914`（multimodal encoder runner 仅备多模态输入 / Kimi K3 单测按 scope 分）、`#21749`（doc：列 Kimi K3.1 Pro feature-branch 支持）、`#21546`（修 draft KV chunked prefill 过度分配）、`#21793`（见 4.9）。
- **量化 / MoE / 模型**：`#21949`（low-M GEMM max_m_for 测试 SM107-aware）、`#21610`（autotuner：线程池驱 DeepGEMM JIT warmup、停重走 excluded-op sweeps）、`#21712`（modeling_v2：LLM arg 默认 auto）、`#20648`（FastH3 + VSA 支持）、`#21514`（CUDA graph 配置 guard）、`#21232`（scheduler 准入 + CUDA graph 资格 guard）、`#21204`（CODEOWNERS 随代码布局更新）、`#18637`（stream multimodal content hashing）、`#21775`（C++ logger keyed log-once）、`#21908`（CuteDSL MoE FC1 autotune，见 4.7）。
- **解聚 / 调度 / 健壮性**：`#21334`（见 4.8）、`#21491`（见 4.13）、`#21963`/`#21989`/`#22011`/`#22010`/`#22023`/`#22042`（pre-merge/retired 测试 waive 与 license 多项）、`#21204`（CODEOWNERS）、`#20238`（Enable PPLX decider 支持）、`#21911`（Kimi K3 单测按 scope 分）。
- **KV / 内存 / 安全**：`#21893`（doc：deployment YAML 用 nested KV cache memory settings）、`#21767`（Clamp CGA reduction split cap silently，fix）、`#21315`（bound context logits memory + CPU fallback）、`#20908`（CUDA graph capture 期间允 NCCL window 注册）、`#21975`（Tool parsers 报 stream 末增量带来的 calls）、`#21790`（Pin sentencepiece <0.2.3）、`#21582`（见 4.4）。

---

## 5. llama.cpp：2026-11-10 提交（42 条）—— FP8 KV-cache 首秀 + Kimi K3.1 视觉 + CUDA VMM 减半 + Vulkan OOB + SYCL 设备端 MoE 调度 + 图采样链别名静默错误

### 5.1 `#28392` [feat] Add FP8 KV-cache support（OCP-FP8 KV-cache 首秀，强化第 13 条）
- **主题**：加 **OCP-FP8 KV-cache** 支持；运行时静态 per-tensor scales 支持但可选。
- **内容详情**：用 graph-level scaling/multiplication 而非改 `ggml_flash_attn_ext`（匹配 NVFP4/FP8 的 ownership-model，便于 FP8 逻辑在 FA 与标准 attention 间共享）；加 E2E 测试（转换+加载 / context shifts / FP8-scaled KV 一般正确性）；示例模型 `nvidia/Mistral-Medium-3.5-128B-NVFP4`（非单位 scale 的 FP8-KV-cache）。
- **应用场景**：llama.cpp 的 FP8 KV cache（OCP E4M3），降低 KV 显存。
- **研判**：【事实】PR 给 OCP-FP8 KV-cache 支持 + graph-level scaling + 三类 E2E 测试 + 示例模型。【观点】属「FP8 KV cache 跨引擎标准化落地」（第 13 条）——llama.cpp 首次引入 FP8 KV（此前仅 NVFP4/FP8 权重），与 vLLM/TRT-LLM 同期推进 FP8 KV 主线；记为官方一手（feature 级，含 E2E 测试）。**证据等级**官方一手（feature 级，含 E2E 测试）。
- **来源**：ggml-org/llama.cpp #28392（2026-11-10T16:08:33Z 合并）。

### 5.2 `#31980` [feat] mtmd: add kimi K3.1 vision（Kimi K3.1 视觉多模态，feature）
- **主题**：修 #31963；多为同 K2.5 的实现，并含若干修复使结果与 vLLM 对齐。
- **内容详情**：测过文本模型切片 + vision tower 全权重；AI 使用披露 yes。无逐模型精度/性能数字——记为官方一手（feature 级，描述级）。
- **应用场景**：llama.cpp 的 Kimi K3.1 视觉多模态（mtmd）。
- **研判**：【观点】属「多模态模型支持」，Kimi K3.1 视觉接入、与 vLLM 对齐；无数字，记为官方一手（feature 级，描述级）。**证据等级**官方一手（feature 级，描述级）。
- **来源**：ggml-org/llama.cpp #31980（2026-11-10T10:28:48Z 合并，Fixes #31963）。

### 5.3 `#30542` [fix] cuda: halve VMM pool reservation and retry on cuMemAddressReserve failure（CUDA VMM 池减半重试，fix）
- **主题**：`ggml_cuda_pool_vmm` 原固定预留 32 GB 虚拟地址空间；统一内存/集显设备上此远超其容量，跑 gemma4:26b（LLM+MTP+mmproj 三 VMM 池）时 GPU 返 OOM CUDA error；若初始 32 GB 预留失败，本变更减半 `max_size` 并重试至成功。
- **内容详情**：修 #29142（亦 #24232/#23094）；在 aarch64 iGPU 上以 gemma4:26b（main+MTP draft+vision mmproj）验、确认分配 6 个并发 VMM 池；AI 使用披露 YES（Gemini Argon 定位失败点并本地 patch 测试）。
- **应用场景**：llama.cpp CUDA VMM 池在 iGPU/统一内存设备上的预留失败重试。
- **研判**：【事实】PR 给根因（固定 32 GB 预留超 iGPU 容量、gemma4:26b 三池 OOM）与减半重试修复、aarch64 iGPU 验 6 并发池。【观点】属「CUDA 显存管理健壮性」，VMM 预留按设备容量自适应；记为官方一手（fix 级，含验证）。**证据等级**官方一手（fix 级，含验证）。
- **来源**：ggml-org/llama.cpp #30542（2026-11-10T11:36:29Z 合并，Fixes #29142）。

### 5.4 `#31890` [fix] vulkan: Fix out of bounds reads/writes on MoltenVK（Vulkan OOB 修复，强化第 4 条）
- **主题**：修 #31892——Mac 上以 MoltenVK 编译时，部分 Vulkan shader 从不检查其 row 是否存在，可越张量读写，触发 `test-backend-ops` 的 `sentinel mismatch` 错误。
- **内容详情**：KosmicKrisp 不受影响；AI 使用披露 YES（troubleshoot + fix）。
- **应用场景**：llama.cpp Vulkan 后端在 MoltenVK（Mac）的越界读写修复。
- **研判**：【事实】PR 给复现（MoltenVK 下 shader 不查 row 存在 → sentinel mismatch #31892）。【观点】属「Vulkan 后端越界写入」（第 4 条），特定平台 shader 不查 row 存在致 OOB，修复前无报错；记为官方一手（fix 级，含复现）。**证据等级**官方一手（fix 级，含复现）。
- **来源**：ggml-org/llama.cpp #31890（2026-11-10T07:19:12Z 合并，Fixes #31892）。

### 5.5 `#30156` [feat] sycl: Add device-side MoE scheduler（SYCL 设备端 MoE 调度，perf）
- **主题**：为 SYCL backend 加 device-side MoE 调度，扩展 #29245 使 CPU 侧调度可选，GPU graph 不被打断（助 #28725）；核心：在 GPU 上建 MMID schedule（而非 CPU）、建所需 slot 上界、空则早返（免 host↔GPU 拷形状）、跑 #29245 的 Grouped GEMM。
- **内容详情**：**解除 Multi-GPU 的 graph 限制**（作者未遇到问题，提醒若破坏告知）；禁部分 OneDNN/MKL 功能以保 graph 兼容。主要影响 PP；TG 在极小输入略降（kernel 过近似非免费）。B60 GPUs：Qwen3.8-Flash-Next UD-IQ4_XS 3 卡 128 prompt TTFT 648.9→628.8 ms（−3.1%）、PP 197.50→203.83 tok/s（+3.20%）；2048 prompt TTFT 4142.3→3727.2 ms（−10.0%）、PP 494.57→549.68（+11.14%）。
- **应用场景**：llama.cpp SYCL（Intel GPU）后端的设备端 MoE 调度。
- **研判**：【事实】PR 给 B60 多 shape 的 TTFT/PP 对照（PP +3.20%~+11.14%、TTFT −3.1%~−10.0%）。【观点】属「SYCL MoE 调度性能」，device-side 调度保 GPU graph 连续、PP 提速；TG 小输入略降，记为官方一手（含 B60 对照）。**证据等级**官方一手（含 B60 对照）。
- **来源**：ggml-org/llama.cpp #30156（2026-11-10T07:42:26Z 合并）。

### 5.6 `#30952` [fix] hexagon: fix 64-bit FastRPC handle truncation on close（Hexagon 64 位句柄截断，强化第 9 条）
- **主题**：修 `ggml/src/ggml-hexagon/htp-drv.cpp` 中 64 位 FastRPC 句柄截断致 `remote_handle64_close()` 失败（0x2c，AEE_EBADHANDLE）。
- **内容详情**：错误函数指针签名由 2026-01-29 commit `ce38a4db`（PR #19150，hexagon Windows on Snapdragon offload）引入并一直未变；FastRPC API 期望 `remote_handle64`，动态解析的指针却声明为 `remote_handle`，wrapper 自身正确用 `remote_handle64`；经错误类型指针传参在测试平台截到 32 位。修复仅改 typedef：`-remote_handle` → `+remote_handle64`。无需 DSP 侧改动。验于 Snapdragon 8 Elite Gen 5（Hexagon v81）有效 64 位句柄。
- **应用场景**：llama.cpp Hexagon（Qualcomm）后端的 FastRPC 64 位句柄关闭。
- **研判**：【事实】PR 给根因（函数指针签名 `remote_handle` 截 64 位句柄到 32 位，`remote_handle64_close` 失败 0x2c）与修复（typedef 改 `remote_handle64`）、Snapdragon 8 Elite Gen 5 验证。【观点】属「64 位句柄/整型截断」（第 9 条 int 溢出家族，但为指针截断而非算术溢出），属平台后端健壮性；记为官方一手（fix 级，含验证）。**证据等级**官方一手（fix 级，含验证）。
- **来源**：ggml-org/llama.cpp #30952（2026-11-10T00:10:14Z 合并）。

### 5.7 `#32015` [fix] graph: mark every sampling chain as an output to prevent aliasing（图采样链别名静默错误，强化第 4/5 条）
- **主题**：采样链在 `ggml_backend_graph_compute` 后于 host 读，但仅当前 graph build 选中的链标了 `ggml_set_output`；未选中的 per-(sequence,row) 链在其最后消费者（`ggml_get_rows`，建 top-k logits）后被 ggml-alloc 释放、buffer 被下一链复用。token-generation reserve graph 每 sequence 一行输出、地址按 node index 被实际 graph 复用（`needs_realloc==0`），致 block draft（每 sequence `n_max` 行）继承别名 buffer：每行列最后一行 top-k candidate ids、**腐蚀 draft token 并使受影响 sequence 接受率崩**。
- **内容详情**：以 `-np 2`（非统一 KV）两并发请求复现 `draft-dspark`/`draft-dflash`/`draft-eagle3`，不同 prompt 时一 sequence 接受崩（如 eagle3 0.15 vs 0.40）；单请求/统一 KV/相同 prompt 均掩盖。修复：`build_sampling` 记每 per-(sequence,row) chain 于 `llm_graph_result::t_sampling_all`，`set_outputs` 全标 `ggml_set_output`，使任一链不被释放/别名。复现命令：`./bin/llama-server -hf ggml-org/spec-for-testing-GGUF:Q4_K_M --spec-type draft-eagle3 -c 8192 -lv 4 -np 2`。
- **应用场景**：llama.cpp 投机解码（eagle3/dflash/dspark）在并发请求下的采样链别名错误。
- **研判**：【事实】PR 给复现（eagle3 接受率 0.15 vs 0.40 于 `-np 2` 不同 prompt）与修复（全采样链 `ggml_set_output`）。【观点】属「图采样链别名静默错误」（第 4/5 条）——未标 output 的采样链 buffer 被复用，静默腐蚀 draft token、崩接受率，典型异构/图后端静默正确性债；记为官方一手（fix 级，含复现）。**证据等级**官方一手（fix 级，含复现）。
- **来源**：ggml-org/llama.cpp #32015（2026-11-10T15:12:28Z 合并）。

### 5.8 `#32004` [cuda] skip graph compat check when graph is being reused（CUDA graph 复用跳检查，perf）
- **主题**：大 graph 上通过跳过 compat check 降 host 时间（该检查非必要，非零匹配 uid 保证匹配）。
- **内容详情**：AI 使用披露 YES；无性能数字表——记为官方一手（perf 级，描述级）。
- **应用场景**：llama.cpp CUDA graph 复用时的 host 时间优化。
- **研判**：【观点】属「CUDA graph host 开销优化」，复用 graph 跳过 compat check；无数字，记为官方一手（perf 级，描述级）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：ggml-org/llama.cpp #32004（2026-11-10T10:53:47Z 合并）。

### 5.9 `#31834` [fix] common/json-schema: fix allOf with plain types generating empty object grammar（json-schema allOf 空对象 grammar 修复，fix）
- **主题**：修 #31805——含 integer 期望的 JSON schema 输出为空（`allOf` 由 plain types（如 integer/number）组成时 grammar root 沦为 `"{" space "}"`）。
- **内容详情**：根因 `_visit_all_of`（common/json-schema-to-grammar.cpp）在查 enum/object/`$ref` 后把 integer 因非 `$ref`/object/enum 丢弃，无 collected fields 时建空 `{}`；children 非 `$ref`/object/enum 被静默丢。修复：加 `leaves` 列表收集除 `$ref`/enum/object 外的所有类型、跳过空 `{}`、仅取 allOf 直接子（不含 anyOf 嵌套）；`properties` 空但 `leaves` 非空时渲染之；单 item 直接渲染、多 item 且均 integer/number 则并为其交集整数范围（string+integer 等组合仍按旧）。现路径渲染 direct non-structural children（string/boolean/const 等）。macOS/cmake 构建、gemma-3-1b-it-Q4_K_M.gguf 验。
- **应用场景**：llama.cpp 结构化输出（JSON schema / grammar）的 allOf 处理。
- **研判**：【事实】PR 给根因（`_visit_all_of` 静默丢非结构 children → 空 `{}`）与修复（leaves 列表 + 交集整数范围）、gemma-3-1b 验证。【观点】属「结构化输出正确性」，allOf 静默丢非结构类型致空 grammar（功能静默失效）；记为官方一手（fix 级，含验证）。**证据等级**官方一手（fix 级，含验证）。
- **来源**：ggml-org/llama.cpp #31834（2026-11-10T07:57:36Z 合并，Fixes #31805）。

### 5.10 `#32009` [feat] llama: reuse backend scheduler when sched_params are compatible（backend scheduler 复用，perf/健壮性）
- **主题**：把定义 backend scheduler 实例的参数（`max_nodes`/`parallel`/`op_offload`）包进 `llama_context::sched_params`，当新参数可由当前 scheduler 服务时跳过 scheduler reset。
- **内容详情**：当前 `sched_reserve()` 触发即重建 scheduler——如设/卸 LoRA adapter、backend sampler、切 `embeddings` 标志；重建会 dealloc/realloc 内部 buffer 与 reserved compute buffer，即便新最坏 graph 同尺寸或更小。scheduler 可服务任意 ≤`max_nodes` 节点的 graph，故新 `max_nodes` 不大且调度行为不变（同 `parallel`/同 `op_offload`/同 callbacks）时复用现有 scheduler。以 `test-llama-context`（含 `GGML_SCHED_DEBUG_REALLOC=1` 遇非预期 graph realloc 即 abort）+ API 级测试验（toggle `embeddings` 复用、加 backend sampler 重建、去 sampler 再复用）。
- **应用场景**：llama.cpp 切换 LoRA/backend sampler/embeddings 时的 scheduler 重建开销削减。
- **研判**：【观点】属「调度器资源复用」，sched_params 兼容时跳过重建、省 dealloc/realloc；含 `GGML_SCHED_DEBUG_REALLOC` 验证，记为官方一手（feature/fix 级，含测试）。**证据等级**官方一手（feature/fix 级，含测试）。
- **来源**：ggml-org/llama.cpp #32009（2026-11-10T15:10:12Z 合并）。

### 5.11 本期其余 llama.cpp 要点（标题/正文级）
- **量化 / CUDA / 多后端**：`#31918`/`#32018`/`#32001`（CUDA refactor graph + 去 unused quantize_cuda_t typedef / 去 DIAG_MASK）、`#31896`（MUL_MAT src1 precision tests + gating）、`#31946`（移除 DIAG_MASK_INF/ZERO）、`#31860`（gpt-oss gguf conversion workaround for p-e-w）、`#30829`（vulkan：Qualcomm Adreno 大 head 标量子注意力 ≤8 行）、`#31930`（vulkan：GROUP_NORM workgroup 拆 2D 避 maxComputeWorkGroupCount）、`#30415`（ggml-rpc：不可达端点/握手失败优雅失败）、`#31574`（sycl：Alchemist head_dim=64 解 gating oneDNN fused SDPA）、`#31540`（ci：oneAPI 2026.2/oneDNN 2026.1/Level Zero 1.34.0）、`#31701`（FreeBSD Vulkan build 设 _XOPEN_SOURCE=700）、`#31270`（cann：buffer type 不支持早失败）。
- **多模态 / UI / 其他**：`#31945`（mtmd：subcontext 间共享 compute buffer）、`#31982`（server：realtime api 输出语言修复）、`#31937`（common：tool argument 字符串转义控制字符）、`#31956`（chat：response_format 设 tools 时不懒 grammar）、`#31984`（model：PLaMo2 停重命名 recurrent cache tensors）、`#31957`（llama：mmap off 时才用 iGPU lazy-mode fallback）、`#32022`/`#32012`/`#32010`/`#32030`/`#32035`/`#32040`/`#32006`/`#31789`（ui：sidebar/MCP/Settings/editor/README banner 等多项）、`#31837`（ui：mmproj 有 audio encoder 的模型在 Discover 显音频）。

---

## 6. 累计结论（继承 1–12，新增第 13 条）

- **第 1 条（调度器隐式阻塞的定量画像）**：本期无新增铁证。沿用。
- **第 2 条（发版窗口的双向性：发版当天合入 ≠ 在版内）**：本期窗口内无新发布；vLLM 仍为 v0.33.0rc3（2026-11-02）、SGLang v0.5.23（2026-11-02）、TRT-LLM v1.4.0rc3（2026-11-06，prerelease）、llama.cpp v0.9.0（2026-11-04）+ nightly b12259/b12261/b12262（2026-11-10，prerelease）。沿用。
- **第 3 条（默认值成为主要风险载体）**：本期无往期级「默认变更/移除」头号事件；窗口内多为正确性/性能/重构（vLLM #66654 弃 legacy server 在 v0.37.0、#65217 opt-in spec config；TRT #21334 opt-in GEN 抢占；llama.cpp #32004 跳过 graph check）。**默认值/移除风险主线本期无新增铁证，节奏暂缓**。
- **第 4 条（异构后端静默错误高频）**：本期四处铁证——llama.cpp `#32015`（图采样链别名致 draft token 静默腐蚀、eagle3 接受率 0.15 vs 0.40）、`#31890`（Vulkan MoltenVK shader 不查 row 存在 → OOB 读写 sentinel mismatch）；TRT-LLM `#20958`（跨注意力 FP8 KV 池地址位宽不一致 → 非法访存/静默错 token）；vLLM `#66370`（Kimi-K3 AMD extensible KV 重绑后未 prepare_kv 致启动即崩）。**异构/量化静默错误持续密集，本期以「已修复的静默偏差/崩溃」为主**。
- **第 5 条（前缀缓存 × MoE 路由 × 投机解码交叉债务）**：本期以 PD/投机正确性为主——SGLang `#48117`（语法拒绝 draft token 未捕获 → 崩溃整个 decode 引擎）、`#49355`（Kimi-K3.1 DSA/MLA/DCP 池接入外部 linker）；TRT-LLM `#22051`（Helix+DSpark draft KV block reuse 与私有 arena 错配 → 静默掉接受长度）、`#21582`（Helix MLA decode kernel 选择落到不可执行 backend）；vLLM `#65217`（MRV2 prefill_context_kv_only 瘦 P/D prefill 实例）、`#65704`（GLM-5.3 MTP eh_proj plan 漏建）；llama.cpp `#32015`（见第 4 条）。**投机解码/PD 正确性跨引擎延续（含一例崩溃级缺陷）**。
- **第 6 条（量化路径静默错误，与异构后端并列的第二类高频静默错误）**：TRT-LLM `#20958`（跨注意力 FP8 KV 池 1 字节分配 vs 2 字节寻址 → 越界/静默错 token）；vLLM `#65177`（INT8 MoE per-channel weight scales 独立索引）、`#65975`（Humming MoE SiLU fast math 提速、量化字节位一致）；SGLang `#42119`（ROCm GPTQ/AutoRound INT4 路径补齐）、`#49659`（MXFP8 权重 JIT 量化）；TRT `#21208`（static-FP8 激活 Triton 量化）。**量化静默债跨引擎延续（本期多为已修复/已适配项）**。
- **第 7 条（专家张量/路由归属与分页漏接静默错误）**：TRT-LLM `#21923`（Triton FP8 MoE align 仅支持 2 幂专家数、非 2 幂被拒，72 专家正确性修复）属专家数/对齐正确性，与第 7 条呼应。本期无往期级「专家分页静默用错值」铁证。沿用并补充。
- **第 8 条（KV 容量/复用智能：容量自检、准入、传输/卸载正确性）**：SGLang `#49299`–`#49312` 统一内存重构收尾（容量决策按 demand 计价 + PP stage 一致 + **double-free 防护**）+ `#47617`/`#47626` HiCache 存储健壮性（worker 异常逃逸静默冻结 / backup 批次失败误信已存）；vLLM `#65822`（MoRIIO WRITE 跳过 D 已缓存 prefix 块，1650→130 MB、+8.7% tok/s/GPU）、`#66605`（外部连接器声明 KV 内存注册）、`#66892`（MiniMax NVFP4 KV 兼容 BLHNC 卸载）；TRT `#21793`（KVCM2 SINK_REQUIRED_ONLY 与 REQUIRED 同组复用）。**KV 复用/容量正确性跨引擎延续且深化（统一内存 demand 化 + double-free 防护 + HiCache fail-fast）**。
- **第 9 条（长上下文 int 溢出为静默正确性新子项）**：本期 llama.cpp `#30952`（Hexagon 64 位 FastRPC 句柄截断致 `remote_handle64_close` 失败）属 64 位句柄/指针截断家族（非算术溢出，但同「64 位量被截到 32 位」根因类），记为第 9 条旁支（指针截断）。无往期级算术 int 溢出新增。**第 9 条新增「指针/句柄 64→32 位截断」旁支**（llama.cpp #30952）。
- **第 10 条（量化/MXFP8 KV 正确性簇）**：vLLM `#65975`（Humming MoE SiLU fast math）、`#66892`（MiniMax NVFP4 KV BLHNC）；SGLang `#42119`（ROCm INT4）、`#49659`（MXFP8 权重 JIT）、`#49595`（dspark MXFP8 draft lm_head）；TRT `#19396`（非-MLA FP8 KV）、`#20958`（跨注意力 FP8 KV）、`#21208`（static-FP8 激活）。llama.cpp `#28392`（OCP-FP8 KV-cache）。沿用并强化。
- **第 11 条（新旗舰模型的上下文并行/解码栈 CP 成为一等公民）**：本期 SGLang `#49355`（Kimi-K3.1 DSA/MLA/DCP 池接入外部 linker，CP/DCP 池映射）延续 CP 主线。属已确立观察，本期无新增强化，节奏暂缓。
- **第 12 条（DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 模型落地的主路径）【上期新增，本期强化】**：vLLM `#66644`（DeepSeek-V4.1 在 ROCm 接 `aiter_mega_moe`，GSM8K 0.9735 持平 aiter）+ `#65975`（Humming MoE SiLU fast math 16–40% kernel 提速、量化字节位一致）；SGLang `#49321`（DSv4.1 MegaMoE decode mHC mix 错峰到 MegaAttention core，GB300 吞吐 +0.4%/interactivity +2.6%）+ `#47398`（gpt-oss W4A8 MoE Gluon/FlyDSL 阈值）+ `#42119`（ROCm INT4）；TRT `#21208`（static-FP8 激活）。两引擎 + llama.cpp 同期推进「新旗舰 MoE × 量化 × MegaMoE/JIT」深化，确认该路径从「能跑」向「融合/阈值选型/自动量化」演进。**第 12 条维持并强化（AMD MegaMoE 落地 + 量化激活/权重 JIT 提速）**。
- **第 13 条（FP8/低比特 KV cache 跨引擎标准化落地）【本期新增】**：llama.cpp `#28392`（**首秀 OCP-FP8 KV-cache**，graph-level scaling + E2E 测试）；TRT-LLM `#19396`（非-MLA PrimsTS FP8 E4M3 KV，跨 context/decode/mixed、scale 在 CUDA Graph replay 保持 live）+ `#20958`（跨注意力 FP8 KV 静默错 token/非法访存修复）；vLLM `#66892`（MiniMax-M3 NVFP4 KV 兼容 BLHNC 卸载布局）；SGLang 本期未见对应 FP8 KV 主干（仍以 HiCache/MXFP8 权重为主）。FP8 KV 从「MLA 专属压缩」走向「跨架构/跨引擎通用 KV 压缩」主线确立——llama.cpp 的加入标志四家中三家（vLLM/TRT/llama.cpp）已支持 FP8 KV，SGLang 侧以 MXFP8 权重/HiCache 为主。**属新兴观察，需后续窗口复现确认是否成为稳定范式**。

---

## 7. 社区侧（HN Algolia，2026-11-10 窗口）—— 引擎相关高信号弱，仅社区/Show HN 推理生态帖

HN Algolia（2026-11-10 窗口，epoch 1794268800–1794355200，HTTPS）引擎关键词高信号弱：
- `vllm` 46 命中均为评论/模糊「LLM」匹配，0 条 story；`sglang`/`llama.cpp`/`TensorRT-LLM`/`KV cache`/`speculative decoding`/`quantization inference` 均 0 story；`LLM inference` 6 命中、`GPU inference` 2 命中，均为社区/Show HN。
- 与四家官方更新相关的 story 0 条。窗口内出现的推理生态帖（均**二手线索/未核验**）：
  - **「LLM inference, visually explained: why a 13B model reads 26 GB per token」**（YouTube，3 分）：可视化讲解 LLM 推理的 KV/激活显存开销，属教学视频，与推理机制相关，未核验其技术准确性。
  - **Show HN: Warmctl – Fast cold starts for vLLM inference**（github.com/TensorRails/warmctl，1 分）：面向 vLLM 的快速冷启动工具，第三方，未核验其对 vLLM 的可用性与效果。
  - **Show HN: Tokspur – a 3x faster open source inference engine for Macs**（github.com/san-digital/tokspur，4 分）：Mac 端开源推理引擎，声称 3× 更快，未核验其基准与精度。
  - **FlashStates – Unlocking Flexible Hybrid State Management for Agentic LLM Inference**（flashstates.ai，2 分）：混合状态管理（agentic 推理），未核验。
  - **CUDA+ROCm dual-backends LLM inference for mixed-vendor GPUs (up to 48% faster)**（github.com/mw00/project-maya/pull/879，2 分）：异构 GPU 双后端推理，声称最多 48% 更快，未核验。
- 其余 `vllm`/`LLM inference` 命中为 Ask HN 学习类、Show HN 无关工具（博客/ReAct 实现/VLM 训练数据/VW embedding 等），与四家引擎官方更新无直接关系。

**研判**：【事实】上述为 2026-11-10 当日 HN 发布的标题/链接与命中数（HTTPS 抓取，epoch 窗口经 `date`-等价生成）。【观点】窗口内无引擎官方更新类高信号；仅通用推理生态/工具 Show HN 与四家相关（vLLM 冷启动、Mac 引擎、异构 GPU）——**均为二手线索/未核验**，未核验其对主流引擎的可用性、精度或性能声称。**证据等级**二手线索/未核验（HN 公告 + 外部 GitHub/视频，未独立验证）。

---

## 8. 趋势判断与可执行清单

**趋势判断**：
1. **FP8/低比特 KV cache 跨引擎标准化落地（新观察，第 13 条）**：llama.cpp `#28392` 首秀 OCP-FP8 KV（vLLM/TRT/llama.cpp 三家已支持）、TRT `#19396`（非-MLA PrimsTS FP8 KV）+ `#20958`（跨注意力 FP8 KV 静默错 token 修复）、vLLM `#66892`（MiniMax NVFP4 KV 兼容 BLHNC 卸载）。FP8 KV 从「MLA 专属」走向「跨架构通用 KV 压缩」主线确立。
2. **SGLang mem_cache 统一内存重构收尾 + HiCache 存储健壮性（第 8 条）**：`#49299`–`#49312` 把容量决策从标量搜索改 demand 计价 + PP stage 一致 + double-free 防护 + 文档；`#47617`/`#47626` 堵 HiCache 静默冻结与 backup 尾段误信。KV 复用/容量正确性从「能复用」推进到「demand 计价 + fail-fast + 防 double-free」。
3. **静默 / 量化正确性铁证再现（第 4/6 条）**：llama.cpp `#32015`（图采样链别名致 draft token 静默腐蚀、eagle3 接受率 0.15 vs 0.40）、`#31890`（Vulkan MoltenVK OOB）；TRT `#20958`（跨注意力 FP8 KV 静默错 token/非法访存）；vLLM `#66370`（Kimi-K3 AMD extensible KV 启动崩）。**量化/异构后端静默错误持续密集，本期多为已修复项**。
4. **投机解码 / PD 正确性（第 5 条）**：SGLang `#48117`（语法拒绝未捕获 → 崩溃整个 decode 引擎）、`#49355`（Kimi-K3.1 DSA/MLA/DCP 池接入 linker）；TRT `#22051`（Helix+DSpark draft KV block reuse 静默掉接受长度）、`#21582`（Helix MLA decode kernel 选择落到不可执行 backend）；vLLM `#65217`（MRV2 prefill_context_kv_only 瘦 P/D 实例）。
5. **DeepGEMM MegaMoE / 量化 MoE 后端延续（第 12 条）**：vLLM `#66644`（DSv4.1 ROCm `aiter_mega_moe`，GSM8K 0.9735）/ `#65975`（Humming MoE SiLU fast math 16–40%）；SGLang `#49321`（DSv4.1 MegaMoE decode mHC mix +2.6% interactivity）/ `#47398`（gpt-oss W4A8 Gluon/FlyDSL 阈值）/ `#42119`（ROCm INT4）；TRT `#21208`（static-FP8 激活）。
6. **TRT-LLM Rubin 大合并回灌 + Helix/DSpark 正确性（架构级）**：`#21067` Rubin MLA/DSpark/DWDP/Helix 合并回主线（auto-generated review、CI 未确立、大合并风险）；`#21582` Helix MLA decode kernel 选择修复、`#22051` DSpark block reuse 护栏 + 自动 host tier；`#21334` GEN slot 抢占（调度）；`#19396`/`#20958` FP8 KV 扩展。
7. **llama.cpp 多后端健壮性 + 量化/图正确性**：`#28392` FP8 KV-cache 首秀、`#31980` Kimi K3.1 视觉、`#30542` CUDA VMM 减半重试、`#31890` Vulkan OOB、`#30156` SYCL 设备端 MoE 调度、`#30952` Hexagon 64 位句柄截断、`#32015` 图采样链别名、`#31834` json-schema allOf 空对象 grammar。

**可执行清单（建议）**：
- **P0（静默正确性自查）**：llama.cpp 投机解码并发核对 `#32015`（未标 output 的采样链 buffer 别名、eagle3 接受率崩）；Vulkan MoltenVK 越界核对 `#31890`（shader 查 row 存在）；TRT 跨注意力 FP8 KV 核对 `#20958`（KV 池 1 字节分配 vs 2 字节寻址、cross attention 非 paged-context 误读 FP8 为 bf16）；vLLM Kimi-K3 AMD extensible KV 核对 `#66370`（KV 重绑后 prepare_kv 必须重跑）；llama.cpp json-schema allOf 核对 `#31834`（非结构 children 被静默丢致空 grammar）。
- **P1（FP8/低比特 KV）**：llama.cpp FP8 KV-cache 核对 `#28392`（OCP-FP8、graph-level scaling、E2E 测试）；TRT 非-MLA FP8 KV 核对 `#19396`（PrimsTS scale 在 CUDA Graph replay 保持 live）/ `#20958`（跨注意力修复）；vLLM MiniMax NVFP4 KV BLHNC 核对 `#66892`（SimpleCPUOffload block 级注册）。
- **P1（KV 复用 / HiCache / 统一内存）**：SGLang 统一内存 demand 计价 + double-free 防护核对 `#49299`–`#49312`（PP stage 一致、invariant check）；HiCache 存储健壮性核对 `#47617`（worker 异常逃逸 fail-fast）/ `#47626`（rank-agreed written prefix settle）；vLLM MoRIIO WRITE 跳过 prefix 核对 `#65822`（1650→130 MB、+8.7% tok/s/GPU）。
- **P2（跟踪）**：DeepGEMM MegaMoE 量化后端实测（#66644 无端到端 t/s、#49321 GB300 +0.4%/+2.6%、#47398 gfx1250 Gluon/FlyDSL 阈值）；TRT `#21067` Rubin 大合并主线 CI 状态（auto-generated review、CI 未确立）；TRT `#21208` static-FP8 激活 runtime 验证（CI 未完全验证）；TRT `#21841` Qwen4-Exp PLE packed-kernel guard 不校验 packed layout 一致性（留隐患）；llama.cpp SYCL 设备端 MoE 调度 Multi-GPU graph 限制解除后的回归（#30156）。

> 证据等级说明：本档案「官方一手」指来自四家官方仓库的提交/PR/发布元数据与 PR 正文（经 GitHub pulls API 抓取的 markdown）；「二手线索/未核验」指来自社区公告/HN/外部 GitHub 且未经一手来源确认的内容（如 Warmctl/Tokspur/project-maya/FlashStates/LLM-inference 视频）。所有性能/精度数字均标注来源 PR 或注明「描述级/CI 未确立/作者报告未独立验证/无 t/s 对照/部分截断/模拟」，未外推未呈现数字。本窗口为 2026-11-10（周二），提交量较周一（11-09：99/92/35/29）回落至 77/93/48/42（合计 260）；llama.cpp 在 2026-11-10 仅推进 nightly b12259/b12261/b12262（prerelease），无新正式版本。
