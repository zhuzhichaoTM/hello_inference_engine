# 大模型推理优化每日热点档案

> 档案日期：2026-11-08（本自动化于该日执行）
> 关注窗口：2026-11-07（周六）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（issues API，markdown，官方一手）+ HN Algolia（HTTPS，numericFilters 按 2026-11-07 epoch 窗口）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 35 / SGLang 87 / TRT-LLM 10 / llama.cpp 16 合计 148 条；vLLM/SGLang/TRT-LLM 单页均未触 per_page=100 上限（无截断），llama.cpp 16 亦单页；releases API 确认窗口内**无新正式版本**（llama.cpp 仅推进 nightly `b12192` 21:28Z / `b12193` 23:45Z）；23 个关键 PR 正文经 issues API 抓取（markdown 干净文本）；HN Algolia 引擎关键词高信号弱（仅 3 分第三方「TokiRecur」KV 回收启动器与 3 分「Jacq」CUDA DSL，均为二手线索/未核验）。本窗口为周六，提交量较工作日显著回落，与往期周六（如 10-17、10-24、10-31）同形。

## 1. 版本状态与本期头号事件：SGLang Kimi-K3.1 上下文并行（CP）全栈落地 + vLLM 移除 Model Runner V1

**提交计数（2026-11-07 UTC，均 HTTP 200；2026-11-07 为周六，量低于工作日；vLLM 35 / SGLang 87 / TRT-LLM 10 / llama.cpp 16）**：
- **vLLM 35** / **SGLang 87** / **TensorRT-LLM 10** / **llama.cpp 16**。

**发布状态（窗口内无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-07 全部 35 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-07 含 Kimi-K3.1 CP 全栈、rust-server、AMD/量化密集提交。
- **TensorRT-LLM**：仍为 **v1.4.0rc3（True/prerelease，2026-11-06T21:38:32Z）**，窗口内无更新；rc3 收口后小幅清理。
- **llama.cpp**：仍为 **v0.9.0（2026-11-04T13:08:06Z 正式发布）**；11-07 仅推进 nightly `b12192`（21:28Z）/ `b12193`（23:45Z，prerelease），无新正式版本。

**本期头号事件（SGLang Kimi-K3.1 上下文并行全栈 + vLLM 移除 MRV1 + 默认值/静默正确性延续）**：
1. **SGLang Kimi-K3.1 上下文并行（CP）全栈落地（#48709/#48710/#48739/#48740/#48584/#48877/#48638）**：在**同一 UTC 窗口**把 Kimi-K3.1 的 prefill CP、MegaMoE + breakable prefill CUDA graph（BCG）+ DFlash under prefill CP、DSPARK/DFLASH under decode CP（DCP）、TP-shard decode MoE front/latent-up tail、HiCache under decode CP、opt-in 持久化 MXFP4 decode MoE 内核全部并入 `main`。这是「Kimi K3.1 day-0 支持」（往期趋势第 1 条）之后的**架构级深化**——新旗舰 MoE 模型以「CP 为一等公民」的节奏落地，从「模型能跑」推进到「长上下文/多节点解码栈原生 CP」。gsm8k PP2×PCP4 0.954 / PP2×PCP8（2 节点）0.958；PD（PP4×PCP2 prefill + DFLASH → TP8 DCP8 decode）passkey 32K/128K 5/5（详见第 3 节）。
2. **vLLM 移除 Model Runner V1（#63373）**：删 `gpu_model_runner.py`（7.2K 行）、V1 投机解码 proposers（eagle/dflash/draft_model/ngram*/suffix_decoding/custom_class_proposer 等）与 Triton 内核、V1 sampler/rejection sampler、V1 自定义 logits-processor API；移除 `VLLM_USE_V2_MODEL_RUNNER` 与 `VllmConfig.use_v2_model_runner`，启动时若设为非 `1` 直接失败；`parallel_drafting` 废弃（设 `True` 对非 DFlash/DSpark 方法仍 raise）。属「默认值/移除风险」（第 3 条）的收口——升级须清理所有 V1 专属配置。
3. **TRT-LLM 默认 dense NVFP4 → Marlin on Ada/Hopper（#21434，修复 #20969）**：默认设置下 `RadixArk/Qwen3.8-27B-NVFP4` 在 H200 加载后 warmup 失败（"Arch unsupported for CUTLASS FP4 GEMM"）；本 PR 在 SM89–99 默认改走 Marlin 后端，server 启动且 GSM8K 92/100。属「默认值变更」（第 3 条）的修复类。
4. **静默/量化正确性跨引擎延续（第 4/6 条）**：llama.cpp `#31465`（SYCL CPY 转 Q4_1/Q5_1/Q8_0 舍入与 CPU 静默偏差，20/50 B70、19/50 B580 失败，强化第 6 条）、`#31852`（Metal `GROUP_NORM` 忽略 `ne03` 致静默垃圾，强化第 4/6 条）；SGLang `#48128`（Muse-Glimmer-NVFP4 MXFP8 lm_head 默认加载失败 → block-FP8 转换）。
5. **PD/投机解码正确性（第 5 条）**：SGLang `#48375`（PD 发布 `spec_kv_layout`，DSpark/DFLASH prefill 与 decode 的算法/draft entry 数不匹配直接拒绝，221 + 175 subtests 通过）；vLLM `#66000`（MTP draft config 跳过重校验修 Step5 MTP + PP>1 启动 `NotImplementedError`，含 CUDA graph 内存/KV 容量权衡）。
6. **KV 容量/复用智能（第 8 条）**：vLLM `#60130`（SimpleCPUOffload 每 rank 一 host buffer + opt-in HugeTLB，NUMA 策略，pin 128 GiB 0.19s vs 普通页 2.5s）、`#64994`（DCP MLA 先 up-project 再 A2A，每 head 搬运 514→130 bf16，MI355X +0.6%~+1.1% 吞吐）；SGLang `#48877`（HiCache under decode CP，CI 红，详见第 3 节）。

---

## 2. vLLM：2026-11-07 提交（35 条）—— 移除 MRV1 + DCP/投机/ROCm 性能 + KV Offload

### 2.1 `#63373` [Core] Remove Model Runner V1（V1→V2 迁移收口，强化第 3 条）
- **主题**：删除 Model Runner V1（MRV1），MRV2 已是默认；仅经显式 `VLLM_USE_V2_MODEL_RUNNER=0` 或回退路径可达的 MRV1 被整体删除，仅 MRV1 独有的特性被丢弃而非移植；Dual Batch Overlap 保留并默认跑在 MRV2 上。
- **内容详情**：删除 `gpu_model_runner.py`（7.2K 行）、`gpu_input_batch.py`、`gpu_ubatch_wrapper.py`、`cpu_model_runner.py`、KV/EC connector runner mixins、V1 `BlockTable`；删除 V1 投机解码 proposers（eagle/dflash/draft_model/ngram*/suffix_decoding/custom_class_proposer 等）及其 Triton 内核；删除 V1 sampler 与 rejection sampler；删除 V1 自定义 logits-processor API（`vllm.v1.sample.logits_processor` 现为唯一接口）；移除 `VLLM_USE_V2_MODEL_RUNNER` 与 `VllmConfig.use_v2_model_runner`，启动若在 `1` 之外即失败；`parallel_drafting` 不再被读取，设 `True` 对非 DFlash/DSpark 方法仍 raise，其余方法仅打 deprecation warning。
- **应用场景**：vLLM 运行时默认路径与升级兼容性。
- **研判**：【事实】PR 给删除清单与失败语义。【观点】属「V1→V2 迁移 / 移除风险」（第 3 条）——删 7.2K 行 + 全部 V1 投机 proposers，影响所有依赖 V1 专属特性 / 环境变量的部署；升级到含本 PR 的版本前须移除 `VLLM_USE_V2_MODEL_RUNNER=0` 与 V1 自定义 logits-processor、V1 投机配置。【证据等级】官方一手（PR 正文，已合入 main）。

### 2.2 `#66000` [Bugfix][Spec Decode] Skip VllmConfig re-validation for the MTP draft config（投机解码正确性，强化第 5 条）
- **主题**：修 NVIDIA Step5 MTP 在 PP>1 下的启动，构造 draft config 视图时保留 target 的 `cudagraph_mode` 与 `use_breakable_cudagraph`，复用 #58212 的 derived-view 处理。
- **内容详情**：Step5 MTP 现在 PP>1 下不再对 `Step5MTP` 架构相对 target 的 pipeline size 做校验；视图保留 draft 的 HF config 与 target 共享 runtime configs 且不改原 `draft_model_config`。验证用 `stepfun-ai/Step-5-Preview`（FP8）于 B300、MRV2、EP、3 个投机 token、关前缀缓存：PP2×TP2 修复前 `NotImplementedError`、修复后 OK 且 GSM8K 0.931、MTP accept 0.765、mean accept length 3.294；PP1×TP4 前后 GSM8K 0.929；CUDA graph 由 `FULL_DECODE_ONLY` 变为 `FULL_AND_PIECEWISE`，graph 内存 0.69→4.74 GiB、KV 容量 983,723→925,755 tokens。
- **应用场景**：Step5 MTP + 流水线并行部署。
- **研判**：【事实】PR 给 PP2×TP2/PP1×TP4 前后 GSM8K/accept 对照与 CUDA graph 内存/KV 容量权衡。【观点】属「投机解码 × PP 正确性」（第 5 条），修复恢复 piecewise prefill graph 但牺牲 KV 容量，运维须按负载评估。【证据等级】官方一手（含逐配置对照表）。

### 2.3 `#64994` [Perf][DCP] Up-project the MLA decode output before the A2A merge（DCP/MLA 性能，强化第 8 条）
- **主题**：`--dcp-comm-backend a2a` 下，MLA decode 各 rank 的部分 latent 输出先 all-to-all 再 LSE merge 后施加 `W_UV`；本 PR 改为先对全部 DCP-gathered heads 施加 `W_UV`，使 A2A 每 head 仅搬运 130 而非 514 bf16 值（Kimi-K3，512/128）。merge 对每 head 线性，仅舍入变化。
- **内容详情**：仅作用于无 PCP 的 NCCL A2A（`v_up_proj_combine` 仅设于此；symmetric-memory 与 `ag_rs` 路径不变）；Kimi-K3 在 `W_UV` 未量化时对 MLAAttention 与 DSpark draft 生效，默认开；每 MLA 层成本：gathered `W_UV`（Kimi-K3 TP8/DCP8 +12 MiB/0.32 GiB per GPU）+ N×heads 的 BMM；低于 8 解码 token 时 BMM 比 A2A 省得更多。Kimi-K3 1P1D MI355X TP8/DCP8 decode-bound C16/C32：吞吐 +0.6%/+1.1%、90 分位交互 +1.4%/+2.8%；24/32 running requests 下 decode step −1.1%/−1.7%；prefill-bound C72 交互 +4%、吞吐持平。fp64 相对误差 2.30e-3，两顺序一致。
- **应用场景**：Kimi-K3 DCP 解码通信/计算重叠。
- **研判**：【事实】PR 给 MI355X 逐配置吞吐/交互对照与 fp64 误差。【观点】属「DCP 通信体积优化」（第 8 条旁支），默认开、收益温和（<1.1% 吞吐），仅在大 batch decode 显著。【证据等级】官方一手（含 MI355X 对照）。

### 2.4 `#65508` [ROCm][Perf][MiniMax-M3.1] Emit the o_proj MXFP8 input from the NVFP4 sparse decode kernel（ROCm 量化融合，强化第 6 条）
- **主题**：ROCm 上每 MiniMax-M3.1 decode 层跑 NVFP4 sparse decode attention 后单独做 MXFP8 激活量化（60 launches/step、~4.4–4.6 us 各），再 `o_proj` GEMM；本 PR 给 FlyDSL sparse decode kernel 加 `fp8_out` 变体，末存直接写 E4M3 + E8M0 scales 而非 bf16，以 `QuantizedActivation` 交 `o_proj`。
- **内容详情**：输出布局——末存 lane 拥有 head `n` 的 32 元素 MX block，量化无需跨 lane；split-K 时末 workgroup 合并 partials 并在合并 O 上做 quant。算术：量化的是 bf16-rounded 值，scale 公式 `clamp(exponent_field - 8 + (mantissa > 0.75), 0, 254)` 为 `ceil(log2(amax/448)) + 127` 的精确实现（无 log2/除法），故输出与同 kernel bf16 输出的 `mxfp8_e4m3_quantize` **位精确**；scale 公式已在 CPU 对照 `_mxfp8_e4m3_quantize` 校验。
- **应用场景**：ROCm MiniMax-M3.1 解码层激活量化延迟融合。
- **研判**：【事实】PR 给位精确声明与 CPU 对照。【观点】属「ROCm NVFP4 解码量化融合」（第 6 条），消除 60 launches/step 的量化开销，且不改变数值（非近似）。【证据等级】官方一手（含位精确声明）。

### 2.5 `#60130` [KV Offload] SimpleCPUOffload: one host buffer per rank, opt-in HugeTLB pages（KV 卸载优化，强化第 8 条）
- **主题**：`SimpleCPUOffloadConnector` 每 rank 的 CPU KV cache 作为单一 host mapping 的视图（按解析出的 KV blocks 上取整到所选页大小）；新增 `cpu_hugepage_block_size` 键（"1GB" 或 "2MB"）以显式 HugeTLB 页分配；映射遵循 worker 的 NUMA 内存策略；预算仍为 `cpu_bytes_to_use_per_rank`；从 `fork()` 排除（`MADV_DONTFORK`）避免子进程 COW 钉页。
- **内容详情**：注册按至多 256 GiB 分块（HiSparse 已用上限，不撕裂 KV block）；页太小致预算不足时**直接报错**而非回退 4 KiB 页。早期修订实证：`vllm serve`（TP=2，64 GiB/rank，普通页）引擎 init 19.0s vs 前 20.2s；1 GiB 大页下 pin 128 GiB 0.19s vs 普通页 2.5s；Host↔CPU 拷贝带宽约 42 GB/s 不变。
- **应用场景**：vLLM CPU KV offload 的初始化/钉页性能与 NUMA 亲和。
- **研判**：【事实】PR 给 init/pin 对照（19.0s vs 20.2s、0.19s vs 2.5s）。【观点】属「KV 卸载 host 层优化」（第 8 条），HugeTLB 显著降低 pin 延迟；预算不足 fail-fast 比静默回退安全。【证据等级】官方一手（含对照）。

### 2.6 `#64476` [Spec Decode][DSpark] Fuse the Markov bias and block combine into Gumbel sampling（投机采样融合，强化第 5 条）
- **主题**：加速 DSpark 的采样循环而不改变输出：Gumbel kernel 自带 bias 加法，再用一个 kernel 做 argmax + gather；建立在 #6532（前文）之上。
- **内容详情**：device 端单步采样耗时 42–50% 降低（1 行 15.4→7.8 us、64 行 91.2→52.9 us；GB300、BF16 logits+bias、vocab 200,000、temp 0.8）；每步 kernel 9→7；draft token 与 draft logits 与「先加 bias」**逐位一致**（验证无损）。
- **应用场景**：DSpark 投机解码 Gumbel 采样。
- **研判**：【事实】PR 给逐行数对照（49.6%@1行、42.2%@64行）与逐 token 对齐。【观点】属「投机解码采样融合」（第 5 条），纯性能、输出无损。【证据等级】官方一手（含 GB300 对照表）。

### 2.7 本期其余 vLLM 要点（标题/正文级）
- **DCP / 模型**：`#66273`（Kimi-K3 在 sleep 模式保留 latent-MoE tail fusion）、`#66165`（level-2 sleep 丢弃 draft weights 时告警）、`#66228`（MiniMax-M3 升 fmha_sm100 子仓）、`#61813`（MiniMaxM3VLConfig 改用 upstream Transformers）、`#59767`（Aria 迁 Transformers modeling 后端）、`#61796`（UMBP 第 8/10 项独立 runtime 覆盖）。
- **MoE / DeepGEMM**：`#66015`（DeepGEMM 的 MLOPart helper 在 MegaMoE 缓存 SM map 后停手）、`#66080`（DeepGEMM 放松 warmup M 值匹配 vendored SM100 启发）、`#63540`（fused_inv_rope_fp8_quant 每程序 4 heads）、`#66227`（服务前 warm up DeepGEMM NVFP4 MegaMoE kernels）、`#66021`（稀疏权重 patch 单次校验）。
- **ROCm / 调度 / 启动**：`#63166`/`#66297`/`#65957`/`#61816`/`#66205`/`#66217`（ROCm AITER/CI 修复）、`#65792`（DP 引擎初始化前对并行配置做 hash 校验）、`#65589`（启动期不再从 `vllm.v1.structured_output` import `vllm.parser`）、`#49459`（移除每 load 完整性检查）、`#65743`（删死代码 kernel）、`#64528`（ruff 升级）、`#66100`/`#64829`（CPU ops 重构 / GDN 投机 recurrent update 加速）、`#64588`/`#66188`/`#56246`（Recipe / XPU / Rust frontend CI）。

---

## 3. SGLang：2026-11-07 提交（87 条）—— Kimi-K3.1 CP 全栈 + rust-server + AMD/量化

### 3.1 `#48709` [Feat] kimi-k3.1: Prefill context parallelism (CP-TP group sharing)（CP 头号，新增第 11 条）
- **主题**：Kimi-K3.1 的 prefill 上下文并行——`--enable-prefill-cp --cp-strategy interleave --attn-cp-size N`，基于 layer-boundary prefill CP 框架（#42044），共享 CP 与 TP 组。`#48414` 的拆分（#48504 的子拆分），栈：#48709（core）→ #48710（MegaMoE/BCG/DFlash）→ #48711（fused kernels）→ #48712（PP × PCP BCG）。
- **内容详情**：Config 把 `KimiK31ForConditionalGeneration` 加入 CP-TP group sharing（interleave 策略，仅 prefill；decode 节点与 Kimi-K3 prefill CP 被拒，DFLASH 允许在 PD prefill 节点）；KDA 按 `linear_attn_parallel_group()` 分 heads、interleave↔sequence 行重排；Latent MoE 在 attention CP 下离开 residual stream，reduce-scatter 到本 rank CP 行；model 在每 rank CP shard 跑层并 gather hidden states。
- **应用场景**：Kimi-K3.1 长上下文/多节点 prefill 横向扩展。
- **研判**：【事实】PR 给 gsm8k（PP2×PCP4 0.954、PP2×PCP8 2 节点 0.958、TP2×PP4 0.949）、PD passkey 32K/128K 5/5、TTFT 表（8K 0.95s / 32K 2.25s / 128K 5.22s / 256K 9.67s @ PP4×PCP2，bs=1）。【观点】属「新旗舰模型的上下文并行落地」（第 11 条），标志 Kimi-K3.1 以 CP 为一等公民进入 main。【证据等级】官方一手（含 gsm8k/TTFT 对照表）。

### 3.2 `#48710` [Feat] kimi-k3.1: MegaMoE, breakable prefill graphs and DFlash under prefill CP（CP 续，第 11 条）
- **主题**：Kimi-K3.1 prefill CP 下的 MegaMoE、breakable prefill CUDA graph（BCG）、DFlash。依赖于 core PR #48709。
- **内容详情**：MegaMoE——静态 graph bucket 的 padding 行 route 到 −1 并零化输出、EP 对称 workspace 在 load 时分配（计入 KV 预算）、Kimi-K3.1 加入 `MEGA_MOE_NO_FALLBACK_ARCHS`、token budget 约束 eager decode/verify 行与 prefill 行；BCG——Kimi-K3.1 进入 interleave BCG；DFlash——允许 integrated 与 PD prefill，Mooncake 允许不同物理 TP 的 PD 对交换 replicated single-entry MLA draft KV；trtllm DSA 把 65–127 query heads pad 到 128。
- **应用场景**：Kimi-K3.1 prefill CP 下的 MoE/图捕获/DFlash 协同。
- **研判**：【观点】属「Kimi-K3.1 CP 全栈」（第 11 条）的中段，把 MegaMoE/BCG/DFlash 都拉到 prefill CP 下；EP workspace 计入 KV 预算是容量侧影响。【证据等级】官方一手（描述级/部分截断）。

### 3.3 `#48739` / `#48740` [Feat] kimi-k3.1: DSPARK and DFLASH under decode CP + Faster DCP decode reduce（解码栈 CP，第 11 条）
- **主题**：在 Kimi-K3.1 DCP decode runtime（#48721）之上，把投机解码也拉到 DCP 下。DSA target verify 在各 rank shard 上做注意力并返回 `(o, lse)` partials 供既有 DCP merge；indexer 给每个 verify 行按其 causal 长度的本地份额打分；SM100 上 DG-native paged MQA logits 对每个 draft 行只读一次 index-K。DCP override 接纳 DSPARK（static ragged verify，decode attention 模式）与 DFLASH（cutedsl_mla draft backend，decode attention 模式）；DFLASH MLA drafts 在 DCP 下不再被拒。#48740 为 Faster DCP decode reduce（push-combine、V pre-projection、direct exchange）。
- **内容详情**：draft KV 预算按 target 行而非 widened 位置计（每 draft pool 行每 target 行一次）；仅 `cutedsl_mla` 作为 MLA-only DFLASH draft backend 被接纳（`trtllm_mla` 需 sliding-window draft table 工作）；`draft_loc_space_scale` 改用 main 规则。
- **应用场景**：Kimi-K3.1 解码栈上下文并行下的投机解码与 DCP reduce。
- **研判**：【观点】属「Kimi-K3.1 解码栈 CP 深化」（第 11 条）——把 DSPARK/DFLASH 从 prefill CP 延伸到 decode CP，是旗舰模型 decode 横向扩展的关键一步；部分子项（sliding-window MLA drafts、EAGLE3 MLA under DCP、PD+HiCache under DCP）留待后续端口。【证据等级】官方一手（描述级）。

### 3.4 `#48584` [Perf] kimi-k3.1: TP-shard the decode MoE front and latent-up tail（解码 MoE 性能，第 11 条）
- **主题**：main 上每个 TP rank 读完整 router（12.9 MB）与 latent down-proj（51.4 MB）及完整 latent up_proj（51.4 MB）；DFLASH verify batch 下这些 GEMM 受权重读取约束。本 PR 把 MoE front 与 latent-up tail 按 TP 分片（端口 #48414），适用 18–256 行的 front 与 13–256 行的 tail。
- **内容详情**：`18 <= M <= 256` 时，4480 `[router | latent]` 行被切成每 rank 一块（TP8 下 560 行）；`_merge_front_weights` 把 rank 块放到合并权重缓冲前，形成单 fp32-out GEMM；新 fp32 multicast all-gather 组装各块；shared experts + all-reduce（16 blocks）排到 side stream 先于 routed `gemm1`。低于 18 行时 gather + 分块 GEMM 不划算。
- **应用场景**：Kimi-K3.1 DFLASH verify 批的 MoE 权重读取瓶颈。
- **研判**：【观点】属「Kimi-K3.1 解码 MoE TP 分片」（第 11 条），把权重读取从「全量复制」降为「每 rank 本块」，属 CP/TP 协同的性能收口；窗口仅由 M 决定，每个捕获 graph 用同一 front。【证据等级】官方一手（性能/架构级）。

### 3.5 `#48877` [Feat] kimi-k3.1: HiCache under decode context parallelism（HiCache under DCP，第 8 条，CI 红）
- **主题**：Kimi-K3.1 在 DCP 下按 token 分片 DSA index-K cache（#48721），HiCache 须像既有的分片 MLA KV 那样存 index-K；本 PR 移植该路径并解除 #48721 对 HiCache 的 DCP 拒绝。
- **内容详情**：L2——DSA indexer host pool 取 `dcp_size`/`dcp_rank`，按 widened logical page 分页；L3——indexer pool 经 `_storage_page_index`（从 MLA host 移到基类）寻页，storage backends 按 `l3_page_span` 键页；`check_dcp_l3_extra_pool` 在 MLA anchor 旁接纳分片 indexer。CI：Latest PR Test (Base) 与 (Extra) 均 :x:（未全绿）。
- **应用场景**：Kimi-K3.1 DCP 下的 HiCache L2/L3 复用。
- **研判**：【观点】属「HiCache under DCP」（第 8 条），把 KV 复用智能扩展到 decode CP；**CI 红，记为描述级/CI 未全绿**，GPU 验证（L2 + Mooncake L3 under DCP8 多轮前缀命中对照）仍 owed。【证据等级】官方一手（描述级，CI 红）。

### 3.6 `#48638` [Perf][NVIDIA] moe: Add an opt-in persistent MXFP4 decode MoE kernel for Kimi-K3.1（Kimi-K3.1 量化解码内核，第 10/11 条）
- **主题**：opt-in（`SGLANG_K31_DECODE_MOE=1`，默认关）的 Kimi-K3.1 解码 MoE 路径，面向 SM100/SM103。TP-MoE rank（896 local experts、intermediate shard 384、top-16、MXFP4 权重 flashinfer_mxfp4 布局、MXFP8 激活、SiTU）、最多 256 token 下，两个 JIT kernel 替换 flashinfer 的 trtllm-gen MXFP4 routing+FC1+FC2 调用。
- **内容详情**：routing kernel（一个 CTA 建 per-expert token 表、一个 reset work counters）+ 一个持久 tcgen05 kernel（chain 到 routing 经 PDL：FC1 + SiTU + MXFP8 requant 到中间缓冲、再 FC2，动态 claim work）；直接读 flashinfer_mxfp4 权重（无 load-time repacking）；`k31_decode_moe_supported` 在权重准备后一次性决定每层是否走该路径。与 #48586（fused decode routing）交互。
- **应用场景**：Kimi-K3.1 在 Blackwell 的解码 MoE 低延迟。
- **研判**：【观点】属「Kimi-K3.1 MXFP4 解码 MoE 内核」（第 10/11 条），opt-in 默认关、需 SM100/SM103 与完整 896 experts，属增量性能件而非默认路径。【证据等级】官方一手（opt-in/config 级）。

### 3.7 `#48375` [Feat] PD: Publish spec_kv_layout so DSpark/DFlash prefill and decode must match（PD 正确性，强化第 5 条）
- **主题**：经既有 bootstrap `PUT /route` 发布 `spec_kv_layout`（算法 + `num_draft_entries`），仅注入 draft KV 的 DSpark/DFLASH prefill 发布；decode 比对发布布局与自身——不同算法/draft entry 数或 target-only decode 在注册 prefill 目的地前即被拒；老/无字段的 prefill 保持既有行为。
- **内容详情**：两框架（Python + Rust registry）保留首个非 `None` 布局，允许 draftless PP stage 任意注册顺序并拒冲突已知布局；block size 不比（DSpark/DFLASH 按 prompt 位置写 draft KV，gamma 仅塑形 decode 侧）；普通 PD/EAGLE/NEXTN 的 bootstrap payload 不变。
- **应用场景**：DSpark/DFLASH PD 的 draft KV 布局协商与拒绝。
- **研判**：【事实】PR 给验证——221 passed + 175 subtests passed（10 文件），对 merge base 与旧 head 的差分 oracle 共 399 bootstrap PUT/GET + 4788 decode 场景，新-on-新仅在布局不兼容时 raise。【观点】属「PD draft KV 正确性别名」（第 5 条）——把「不兼容 KV 被静默消费」转为「启动前显式拒绝」。【证据等级】官方一手（含验证）。

### 3.8 `#47695` [Feat][AMD] quark: Quark int4 W4A16 on ROCm + gfx1151 Strix Halo（AMD 量化后端，国产后端版图）
- **主题**：main 上 `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` 因 Quark 无 int4 weight-only scheme 而 `NotImplementedError` 无法加载；本 PR 加 Quark int4 W4A16 scheme 与 gfx1151（Strix Halo APU，Radeon 8060S，RDNA3.5，256 GB/s LPDDR5X）的 decode/prefill kernel。
- **内容详情**：`use_gfx115_kernels()` 单一网关（同 #46327 的 RDNA3.5 检查 + `SGLANG_DISABLE_GFX115_KERNELS` 开关），kernel 仅承认实测形状、其余回退 main 路径；`QuarkW4A16Int4` 在 HIP 上对 int4 对称 per-group 32/64/128 选入、CUDA 上返回 False（行为不变）；Strix Halo 上 M<=8 时 W4A16 GEMM 权重读取 40–70 GB/s（低于内存屋顶）。
- **应用场景**：AMD Strix Halo / RDNA3.5 上的 Quark AWQ int4 模型服务。
- **研判**：【事实】PR 给加载失败根因与定性带宽（40–70 GB/s）。【观点】属「AMD/国产后端量化版图扩展」（延续往期 #45149 Cambricon MLU）；无 before/after t/s 对照，仅定性带宽，记为官方一手（含定性带宽，无 t/s 对照）。【证据等级】官方一手（含定性带宽，无 t/s 对照）。

### 3.9 `#48128` [Fix] quantization: Load Muse-Glimmer-NVFP4's MXFP8 lm_head（量化加载正确性，强化第 6 条）
- **主题**：`RadixArk/Muse-Glimmer-NVFP4` 默认 flag 下无法启动——MXFP8 `lm_head` 加载失败（任意 GPU），Hopper 上 MXFP8 dense layers 在 DeepGEMM warmup 失败。
- **内容详情**：`fp8.py` 读 `deepgemm_mxfp8_input` 属性；`modelopt_quant` 在 MIXED_PRECISION + CUDA 非 Blackwell 时把 MXFP8 dense layers（dense + `lm_head`）转 block-FP8 [128,128] 于加载期（MoE 层保持原 MXFP8 config）；Marlin 路径从 linear method 取 block size（[128,128] 而非 checkpoint 的 [1,32]）。1x H200：修复前加载失败，修复后 serve 且 few_shot_gsm8k 200 题 5-shot 0.955（无 BF16/Blackwell 参照，仅 sanity check）。
- **应用场景**：Muse-Glimmer-NVFP4 在 Hopper/默认路径的加载。
- **研判**：【事实】PR 给 0.955 gsm8k 与 CPU 单测（36 tests 通过）。【观点】属「MXFP8 lm_head 加载正确性」（第 6 条），离 Blackwell 默认转 block-FP8 解决加载失败；SGLANG_FORCE_MXFP8_BLOCK_CONVERT 仍 opt-in 且含 MoE，不提供可用默认。【证据等级】官方一手（含 gsm8k 0.955）。

### 3.10 本期其余 SGLang 要点（标题/正文级）
- **Kimi-K3.1 CP 栈其余**：`#48711`/`#48712`（fused kernels / PP×PCP BCG）、`#48713`（shared-expert TP across PCP）、`#48714`（trtllm DSA）、`#48715`（Linker PCP L3）、`#49037`（megakernel DCP push-combine warmup stub）、`#48393`（cap cached_tokens at prompt_tokens）、`#49022`/`#49087`（Kimi-K3.1/ Muse Spark cookbook 同步）。
- **AMD / 量化**：`#47862`（aiter in-tree tuned rows 每模型一目录）、`#49001`（custom-all-reduce carv2 on MI450，FABRIC handles + gfx1250 表）、`#48438`（gfx950 HD128 ASM prefill test）、`#48966`/`#48978`（AMD CI/ jit 平台 warp mask）、`#48044`（gfx950 Q/K GemmaRMSNorm+RoPE Triton 融合）、`#49034`（hicache `_storage_page_index` 返 int 不 DCP）、`#48128`（见 3.9）。
- **性能 / 内核**：`#47247`（DeepSeek-v4.1 CP MoE output reduce-scatter）、`#48563`（gemma4 TP all-reduce 融合 layer epilogue + FP8 quant on NVLink multicast）、`#47893`（glm5_next fused KDA decode 8/16 heads/rank）、`#48531`（Hopper block GEMV 取 K=6144 for MiniMax M3.1）、`#43703`（unquant BF16 GEMM FlashInfer mm_bf16 autotune）、`#44486`（per-tensor FP8 linear cuBLASLt epilogue 加 bias）、`#45497`（avoid concat MLA queries for zigzag CP）、`#47129`（muse-spark B200 projection dispatch 限界）。
- **rust-server / 调度 / 安全**：`#48457`/`#48455`/`#49023`（crash-dump / 日志 / JSON lines）、`#48518`（redact access keys from /server_info/logs）、`#48512`/`#48753`（refuse ignored flags / GenerateValidator）、`#47538`（SageMaker /invocations 路由）、`#48992`/`#49032`（PP 下 pause_generation / multimodal relay）、`#49092`（kv-canary warmup dummy forward token-oracle ids）、`#48896`（LMSYS blog cards 同步）、`#48999`/`#48983`/`#48990`（docker/CI 清理）。

---

## 4. TensorRT-LLM：2026-11-07 提交（10 条）—— rc3 收口后小幅：GLM Rubin 自包含 + NVFP4 融合 + 默认 Marlin + Gemma4 variable-window

### 4.1 `#21434` [#20969][fix] Qwen3.5/3.8: default dense NVFP4 checkpoints to Marlin on Ada/Hopper（默认值变更，强化第 3 条）
- **主题**：修复 #20969——默认设置下 `RadixArk/Qwen3.8-27B-NVFP4` 在 H200 加载权重后 warmup 失败（"Arch unsupported for CUTLASS FP4 GEMM"），同因适用于 SM89–99 的 dense Qwen3.5。本 PR 在 #17293 同样条件下把 dense 模型默认 Marlin 后端，`_get_qwen35_dense_model` 设 Marlin linear default，MoE helper 调它并加 `moe_config.backend = "MARLIN"`（MoE 不变）。
- **内容详情**：1x H200（1.4.0rc1 容器，替换 modeling_qwen3_5.py 为含本改的副本、无额外配置）：server 启动、NVFP4 linears 跑 `MarlinFP4Method`、小客户端在首 100 题 GSM8K（5-shot greedy）得 92/100；同命令无改则如上失败。未跑：H100/Ada/TP>1/图像视频输入/BF16 基线对照。
- **应用场景**：Ada/Hopper（SM89–99）上 dense Qwen3.5/3.8 NVFP4 默认路径。
- **研判**：【事实】PR 给复现（warmup 失败）+ 修复后 GSM8K 92/100。【观点】属「默认值变更修复」（第 3 条）——将原本加载即失败的默认行为改为走 Marlin；rc3 仍为 prerelease，SM89–99 跑 dense Qwen3.5/3.8 NVFP4 部署应核对本改。【证据等级】官方一手（含 GSM8K 92/100）。

### 4.2 `#21273` [perf] Fuse NVFP4 activation quant into SwiGLU (SM100+) and add+RMSNorm (SM12x)（量化融合性能，强化第 6 条）
- **主题**：把 SwiGLU + NVFP4 量化融合进 `GatedMLP`——dense NVFP4 MLPs 经新 `swiglu_nvfp4_quantize` op 直接从 `down_proj` 输出产生量化输入（复用 CuteDSL MoE 激活 kernel，`gate_first` 布局，无 tile 元数据）；column split 让 kernel 在 decode batch 填满 GPU。对 SM>=100 的 static-NVFP4 dense/shared-expert `GatedMLP` 默认开。SM12x 上 `add + RMSNorm + NVFP4 quantize` 融合（Qwen3-Next/Qwen3.5 把 dense `gate_up_proj` 的 static NVFP4 输入量化折进前置 add+RMSNorm）。
- **内容详情**：B200（SM100）SwiGLU+NVFP4 quantize 融合相对 `swiglu()+fp4_quantize`：per-rank I 在 M=1/8/64/512/4096/16384 与 I=1152/1792/3584/7168 组合下加速 1.49×–2.29×（小 M 最高 ~2.29×，大 M 收敛 ~1.15×）。SM12x 的 `fused_add_rms_norm_quant` 因需 ~192 KiB shared mem（SM12x 限 ~99 KiB）保持禁用。
- **应用场景**：Blackwell/Rubin dense NVFP4 MLP 与 dense gate_up_proj 输入量化。
- **研判**：【事实】PR 给 B200 多 I×M 加速表（最高 2.29×）。【观点】属「NVFP4 激活量化融合」（第 6 条），默认开、decode 小 batch 收益最大；大 M 收敛到 ~1.15×。【证据等级】官方一手（含 B200 加速表，部分截断）。

### 4.3 `#21812` [refactor] Make GLM Rubin modeling v2 target self-contained（GLM Rubin 自包含，refactor 级）
- **主题**：解决 #21380 后 GLM-5.2 SM107 低延迟 target 的两处缺口：① 模型结构/权重准备/prefill 仍依赖 V1 模块——以 target 自有实现替换 V1 MLA 与 speculative-model 继承，target 现拥有 MLA 权重绑定与 post-load 变换、DSA prefill/indexer、embedded-MTP 生命周期；② 把 51 个原生文件从 `cpp/.../glmMoeDsaRubin/` 移到 `modeling_v2/models/glm_moe_dsa/.../csrc/`。
- **内容详情**：保留既有 GLM 权重加载器；通用 serving/cache/attention-backend/checkpoint 接口仍共享；用既有 family-library build/loader、target 专属 Torch namespace 与 target 自有 fake registrations；CUDA kernel body 与既有 `100f` 编译保留。
- **应用场景**：GLM-5.2 NVFP4 Rubin 低延迟 target 的可维护性/自包含。
- **研判**：【观点】属「GLM Rubin target 自包含重构」，为后续低延迟迭代铺路，无性能数字，记为官方一手（refactor 级）。**证据等级**官方一手（refactor 级）。

### 4.4 `#18912` [feat] Add Gemma4 variable-window attention（Gemma4 多模态注意力，feature 级）
- **主题**：为 SM100/SM103 上的 Gemma4 多模态请求加 TRTLLM-Gen variable-window context attention——为连续 image/video token block 建含入式 per-query KV bounds 并贯穿 attention backend/THOP/C++ runtime/FMHA dispatcher；sliding-window 层用这些 bounds，full-attention 层保持 causal。
- **内容详情**：FlashInfer context 请求在 variable bounds 下回退 TRTLLM-Gen，eligible generation 请求保留 FlashInfer；runtime 校验成对的 context-only int32 bound 张量并对 variable-window 请求保持 FMHA 选择缓存独立；capability probe 用 attention layer 实际 output dtype/量化决策/context 维度，init 失败报 unsupported 而非终止进程；vision bounds 超 retained sliding cache window 时走既有 dense-mask 路径。
- **应用场景**：Gemma4 多模态（图/视频）在 Blackwell 的 variable-window 注意力。
- **研判**：【观点】属「Gemma4 多模态注意力支持」，feature 级、无性能数字，记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。

### 4.5 本期其余 TRT-LLM 要点（标题/正文级）
- ** disaggregation / warmup / 测试**：`#21538`（port 原生 disaggregation KV result report token，#20723）、`#20813`（跨 warmup phase 复用 sampling 同步与 sequence slots）、`#21692`（关 GLM-5.2 SM103 TEP4 GSM8K 测试的 HF 权重缓存）、`#21823`（unwaive MiniMax H3 两卡 Ulysses LPIPS 测试）、`#21671`（修 MiniMax 测试 subprocess imports）、`#20723`（原生 disaggregation KV result report token 原 PR）。
- **量化 / 模型**：`#21434`（见 4.1）、`#21273`（见 4.2）；其余窗口内无新正式模型支持（rc3 已含 GLM-5.2 NVFP4 V2/MiniMax-M3 NVFP4 decode/Kimi K3/DeepSeek-V4.1-Flash SM120）。

---

## 5. llama.cpp：2026-11-07 提交（16 条）—— SYCL 舍入/VRAM 安全 + Metal GROUP_NORM 修复 + MoE-cache CPU + 后端修复

### 5.1 `#31465` [sycl] exact rounding in CPY to Q4_1, Q5_1 and Q8_0（SYCL 量化舍入正确性，强化第 6 条）
- **主题**：#31120 的 follow-up，修 #31140。SYCL 上把 f32 转 Q4_1/Q5_1/Q8_0 偶尔比 CPU 差一步，致 `test-backend-ops -o CPY` 时好时坏（B70 20/50、B580 19/50 失败）。
- **内容详情**：Intel 编译器默认 fast float——除法成近似倒数偶有 ulp 偏差、Q8_0 tie rounding 在精确 .5 走反；`x*id+0.5` 可能被融成 FMA 致舍入不同。修法同 #31120：`div_correctly_rounded()` 先除再用精确 FMA residual 与两邻居比较取近者；Q4_1/Q5_1 用其并在 `+0.5f` 前以 `fmin` clamp（无 mul-add 可融）；Q8_0 用 `rint`。Q8_0 对齐 x86 AVX2 路径（非 reference）；`ggml-base` 在 `icx` 下以 `-fp-model=precise` 构建。
- **应用场景**：SYCL（Intel GPU）后端量化拷贝的数值一致性。
- **研判**：【事实】PR 给复现率（20/50、19/50）与修法（FMA residual 比较）。【观点】属「量化路径舍入静默偏差」（第 6 条）——此前测试偶发失败即静默数值偏离；修后对齐 CPU/AVX2。无性能数字（仅正确性）。【证据等级】官方一手（含复现率）。

### 5.2 `#31852` [metal] fix GROUP_NORM for ne03 > 1 and non-contiguous inputs（Metal 静默正确性，强化第 4/6 条）
- **主题**：Metal `GROUP_NORM` kernel 忽略 batch 维（`ne03`）且假定完全连续输入——`ne03 > 1` 时大部分输出未写、padded 输入产出垃圾。
- **内容详情**：使新 `GROUP_NORM(type=f32, ne=[3,1,1,65537], num_groups=1)` 测试（来自 #31490）在 Metal 通过（修复前 `ERR = 4.0`）。测试硬件 Mac Studio M5 Ultra 256GB、GLM-5.3-Flash-GGUF。
- **应用场景**：Metal 后端 GROUP_NORM（GLM-5.3-Flash 等 `ne03>1` 模型）。
- **研判**：【事实】PR 给测试名/形状与修复前 ERR=4.0、测试硬件。【观点】属「Metal 后端静默数值错误」（第 4/6 条）——`ne03>1` 或 padded 输入静默产出垃圾，修复前无报错；GLM-5.3-Flash 等模型须核对本修。【证据等级】官方一手（含 Metal 测试）。

### 5.3 `#31285` [sycl] refuse device buffers that would exceed VRAM（SYCL 内存安全，健壮性）
- **主题**：临时补丁——SYCL 过度分配显存曾在 B70（32,656 MiB）跑 `Nemotron-3.5-Super-VL-120B-A12B UD-IQ2_XXS`（45,469 MiB）时致 kernel panic（此前仅按 `max_mem_alloc_size` 校验，未按已分配总量）。现跟踪每设备已分配字节，总量超全局显存时拒新 buffer 并返回 `nullptr`（同 Vulkan 后端）；`GGML_SYCL_ALLOW_VRAM_OVERCOMMIT=1` 保留旧行为。
- **应用场景**：SYCL（Intel GPU）后端大模型的显存越界保护。
- **研判**：【观点】属「显存越界安全默认」（第 3 条旁支）——默认拒绝 overcommit（避免系统崩溃），opt-in 恢复旧行为；临时补丁，记为官方一手（安全/健壮性 级）。**证据等级**官方一手（安全/健壮性 级）。

### 5.4 `#30966` / `#31853` [llama_moe_cache] run missed experts on CPU, add tuning script（MoE 缓存 CPU 卸载，强化第 8 条）
- **主题**：依 `--moe-cache-cpu-share` 把未命中专家放到 CPU 执行（CPU 读专家速度影响 decode）；新增 `scripts/moe-cache-tune.py` 调优 `--moe-cache-mib` 与 `--moe-cache-cpu-share`。
- **内容详情**：Qwen3.8-Flash-Next-IQ4_NL 实测——6464 MiB / share 0：24.2 t/s、hit 73.3%；share 0.95：47.6 t/s、hit 61.5%；19392 MiB / share 0：65.5 t/s、hit 93.5%；share 0.95：54.1 t/s、hit 71.6%。调优脚本前置于 `llama-server` 命令自动测 ideal 配置。
- **应用场景**：llama.cpp MoE 专家缓存的 CPU 协同与容量调优。
- **研判**：【事实】PR 给逐配置 t/s 与 hit rate 对照（share 升 → t/s 升但 hit 降）。【观点】属「MoE 专家缓存容量/复用」（第 8 条），CPU 协同以命中率换吞吐，须按负载调优。【证据等级】官方一手（含 t/s 对照）。

### 5.5 `#31803` [CUDA] fix shared memory bank conflicts in the lightning indexer（CUDA 内核性能）
- **主题**：profiling CUDA 后端发现 `lightning_indexer_kernel_mma`（`LIGHTNING_INDEXER` op）因 bank conflict 损失不少 shared memory 带宽，两处修复。
- **内容详情**：① `qk_chunk` 声明 `float[16][32]`（每行 128 B、行首均 bank 0），半 warp 四行落同 8 bank → 每 8 B store 占 4 wavefront；以 `K_VECS_PADDED` 每行补 8 float，使半 warp 行首隔 8 bank、单次通过。② Q staging 每线程四 floats 转 half 后四次 2 B store（线程隔 8 B 致 bank 冲突），改单次 8 B store（`ggml_cuda_memcpy_1<8>`）。代价：每 block +512 B shared mem。
- **应用场景**：CUDA 后端 `LIGHTNING_INDEXER` 内核带宽。
- **研判**：【事实】PR 给 Nsight Compute 数字——shared wavefronts 5,292,032 → 3,588,096（ideal 3,784,704），全部 19 项指标达 ideal。【观点】属「CUDA 内核 bank-conflict 修复」性能件，无端到端 t/s（仅 kernel 级 wavefront 数）。【证据等级】官方一手（含 Nsight 数字）。

### 5.6 本期其余 llama.cpp 要点（标题/正文级）
- **构建 / 后端**：`#31785`（llama 补缺失 pthread 链接）、`#31447`（meta 提高少张量 buffer 的 compute view 上限）、`#30857`（sycl 加 SWIGLU_OAI 的 f16 支持）、`#31490`（webgpu 实现 group norm kernel）、`#31829`（opencl 加 arange）、`#31822`（hexagon 把 post-norm residual add 融进 RMS_NORM_MUL）、`#31857`（hexagon n_hvx/n_threads 清理）。
- **多模态 / 模型**：`#31808`（mtmd 把 log-mel 计算移到 cgraph）、`#31819`（mtmd 支持 MageVL-Cowork 模型）、`#30966`（见 5.4）、`#31285`（见 5.3）。

---

## 6. 累计结论（继承 1–10，新增第 11 条）

- **第 1 条（调度器隐式阻塞的定量画像）**：本期无新增铁证（窗口为周六低活跃）。沿用。
- **第 2 条（发版窗口的双向性：发版当天合入 ≠ 在版内）**：本期窗口内无新发布；TRT-LLM 仍为 v1.4.0rc3（2026-11-06 发布，prerelease），llama.cpp 仍为 v0.9.0（2026-11-04）+ nightly b12192/b12193；vLLM v0.33.0rc3、SGLang v0.5.23。沿用。
- **第 3 条（默认值成为主要风险载体）**：本期以「移除/默认变更修复」为主——vLLM `#63373`（删除 MRV1 与全部 V1 投机 proposers、移除 `VLLM_USE_V2_MODEL_RUNNER`，升级须清理 V1 配置，属于「移除风险」）、TRT-LLM `#21434`（默认 dense NVFP4 → Marlin on Ada/Hopper，修复 H200 warmup 失败）、llama.cpp `#31285`（默认拒绝 SYCL VRAM overcommit，`GGML_SYCL_ALLOW_VRAM_OVERCOMMIT=1` opt-in 恢复旧行为，安全默认）。**默认值/移除/配置清理仍是高频风险载体**。
- **第 4 条（异构后端静默错误高频）**：本期无往期级「整段垃圾无报错」铁证，但 llama.cpp `#31852`（Metal `GROUP_NORM` 忽略 `ne03` 致 `ne03>1` 静默垃圾、修复前 ERR=4.0）属 Metal 后端静默数值错误；SGLang `#48877`（HiCache under DCP，CI 红，未独立验证）为架构扩展侧。**异构/量化静默错误持续密集，但本期以「已修复的静默偏差」为主**。
- **第 5 条（前缀缓存 × MoE 路由 × 投机解码交叉债务）**：本期以 PD/投机正确性为主——SGLang `#48375`（PD 发布 `spec_kv_layout`，DSpark/DFLASH prefill/decode 布局不匹配启动前显式拒绝，221+175 subtests 通过）、vLLM `#66000`（MTP draft config 跳过重校验修 Step5 MTP + PP>1 启动 `NotImplementedError`，含 CUDA graph 内存/KV 容量权衡）、vLLM `#64476`（DSpark Gumbel 采样融合无损）。**投机解码/PD 正确性跨引擎延续**。
- **第 6 条（量化路径静默错误，与异构后端并列的第二类高频静默错误）**：llama.cpp `#31465`（SYCL CPY 转 Q4_1/Q5_1/Q8_0 舍入与 CPU 静默偏差，20/50 B70、19/50 B580 失败，已修对齐）、`#31852`（见第 4 条）；SGLang `#48128`（Muse-Glimmer-NVFP4 MXFP8 lm_head 默认加载失败 → block-FP8 转换）；TRT `#21273`（融合 NVFP4 量化）。**量化静默债跨引擎延续（本期多为已修复项）**。
- **第 7 条（专家张量/路由归属与分页漏接静默错误）**：本期无新增铁证。沿用。
- **第 8 条（KV 容量/复用智能：容量自检、准入、传输/卸载正确性）**：vLLM `#60130`（SimpleCPUOffload 每 rank 一 host buffer + opt-in HugeTLB + NUMA，pin 128 GiB 0.19s vs 2.5s）、`#64994`（DCP MLA 先 up-project 再 A2A，每 head 514→130 bf16，MI355X +0.6%~+1.1% 吞吐）；SGLang `#48877`（HiCache under decode CP，CI 红）；llama.cpp `#30966`/`#31853`（MoE-cache CPU 协同 + 调优脚本，share 升 → t/s 升但 hit 降）。**KV 正确性/复用跨引擎延续**。
- **第 9 条（长上下文 int 溢出为静默正确性新子项）**：本期无新增 int 溢出铁证。沿用。
- **第 10 条（量化/MXFP8 KV 正确性簇）**：SGLang `#48638`（opt-in 持久化 MXFP4 decode MoE kernel for Kimi-K3.1，SM100/SM103）、`#47695`（AMD Quark int4 W4A16 ROCm + Strix Halo gfx1151）；TRT `#21273`（融合 NVFP4 量化）。沿用并强化。
- **第 11 条（新旗舰模型的上下文并行/解码栈 CP 成为一等公民）【本期新增】**：SGLang 在**同一 UTC 窗口**把 Kimi-K3.1 的 prefill CP（#48709）、MegaMoE+BCG+DFlash under prefill CP（#48710）、DSPARK/DFLASH under decode CP（#48739/#48740）、TP-shard decode MoE（#48584）、HiCache under DCP（#48877）、opt-in MXFP4 decode MoE kernel（#48638）全部并入 main。这是往期「Kimi K3.1 day-0 支持」（趋势第 1 条）之后的**架构级深化**——新旗舰 MoE 模型落地从「能跑」推进到「长上下文/多节点解码栈原生 CP」。属新兴观察，需后续窗口复现确认是否成为稳定范式。

---

## 7. 社区侧（HN Algolia，2026-11-07 窗口）—— 引擎相关高信号弱，第三方 KV 回收工具与 CUDA DSL

HN Algolia（2026-11-07 窗口，epoch 1794009600–1794096000）引擎关键词高信号弱：
- `vllm` 14 命中多为无关（「Jacq, a DSL that compiles model and kernel descriptions to CUDA」3 分、「TokiRecur – llama.cpp LLM launcher with KV-cache recycling」3 分、Rust/LLM 理论帖）；
- `sglang` 0 命中；`TensorRT-LLM` 0 命中；`KV cache` 仅 1 条（即 TokiRecur，3 分）；`speculative decoding` 0 命中；`quantization` 0 命中；`LLM inference` 1 条（Jacq DSL）；`llama.cpp` 1 条（TokiRecur）。
- **TokiRecur – llama.cpp LLM launcher with KV-cache recycling**（2026-11-07T18:08:43Z，3 points，第三方工具，GitHub）：llama.cpp 启动器，主打 KV-cache 回收复用；与推理引擎 KV 复用主题呼应，但属第三方工具，**二手线索/未核验**（未核验其可用性、精度或性能声称）。
- **Jacq, a DSL that compiles model and kernel descriptions to CUDA**（2026-11-07T17:58:49Z，3 points，Show HN）：把模型/内核描述编译到 CUDA 的 DSL，与内核/编译相关但非特定推理引擎更新，**二手线索/未核验**。

**研判**：【事实】上述为 2026-11-07 当日 HN 发布的标题/链接与命中数。【观点】窗口内无引擎官方更新类高信号；仅 TokiRecur（KV 回收启动器）与 Jacq（CUDA DSL）与推理/内核生态沾边——**均为二手线索/未核验**，未核验其对主流引擎的可用性、精度或性能声称。**证据等级**二手线索/未核验（HN 公告 + 外部 GitHub，未独立验证）。

---

## 8. 趋势判断与可执行清单

**趋势判断**：
1. **Kimi-K3.1 上下文并行（CP）全栈在 SGLang 单窗口落地（新观察，第 11 条）**：prefill CP + MegaMoE + BCG + DFlash/DSpark under CP + TP-shard decode MoE + HiCache under DCP + opt-in MXFP4 decode MoE kernel 一次并入 main；gsm8k PP2×PCP4 0.954 / PP2×PCP8（2 节点）0.958，PD passkey 32K/128K 5/5，TTFT 8K 0.95s → 256K 9.67s（PP4×PCP2）。这是「新旗舰 MoE 模型以 CP 为一等公民」的落地节奏，标志 day-0 支持之后进入架构级深化。
2. **vLLM 移除 Model Runner V1（#63373）收口 V1→V2 迁移（第 3 条）**：删 7.2K 行 + 全部 V1 投机解码 proposers（eagle/dflash/draft_model/ngram*/suffix_decoding）+ V1 sampler/rejection sampler + V1 自定义 logits-processor API；移除 `VLLM_USE_V2_MODEL_RUNNER`，启动非 `1` 即失败；`parallel_drafting` 废弃。运维升级须移除 V1 专属配置。
3. **默认值/配置清理延续为风险载体（第 3 条）**：TRT `#21434` NVFP4 → Marlin on Ada/Hopper（修复 H200 warmup 失败，默认变更）、llama.cpp `#31285` 安全默认（SYCL overcommit 默认拒绝，opt-in 恢复）。
4. **量化/低比特静默正确性跨引擎延续（第 4/6 条）**：llama.cpp `#31465`（SYCL CPY 舍入静默偏差，已修对齐）、`#31852`（Metal GROUP_NORM `ne03>1` 静默垃圾，已修）；SGLang `#48128`（Muse-Glimmer-NVFP4 MXFP8 lm_head 加载失败 → block-FP8 转换）；TRT `#21273`（融合 NVFP4 量化，默认开）。
5. **PD/投机解码正确性（第 5 条）**：SGLang `#48375`（spec_kv_layout 必须匹配，拒绝不兼容 DSpark/DFLASH KV，221+175 subtests 通过）、vLLM `#66000`（MTP draft config 跳过重校验修 PP>1 启动，含 CUDA graph 内存/KV 容量权衡）、vLLM `#64476`（DSpark Gumbel 采样无损融合）。
6. **AMD/国产后端扩版图（延续）**：SGLang `#47695`（Quark int4 W4A16 ROCm + Strix Halo gfx1151）、`#47862`（aiter in-tree tuned rows）、`#49001`（custom-all-reduce carv2 on MI450）。
7. **TRT-LLM v1.4.0rc3 收口后小幅**：`#21812` GLM Rubin 自包含、`#21273` NVFP4 融合 SwiGLU、`#21434` 默认 Marlin、`#18912` Gemma4 variable-window；rc3 仍为 prerelease。

**可执行清单（建议）**：
- **P0（vLLM 升级）**：升级到含 `#63373` 的版本前，移除所有 `VLLM_USE_V2_MODEL_RUNNER=0` 设置与 V1 专属特性（V1 投机 proposers、V1 自定义 logits-processor API）；`parallel_drafting` 已废弃（设 `True` 对非 DFlash/DSpark 方法仍 raise）；确认投机解码走 DFlash/DSpark。
- **P0（TRT-LLM 默认）**：在 Ada/Hopper（SM89–99）跑 dense Qwen3.5/3.8 NVFP4 时核对 `#21434`——默认改走 Marlin，避免 H200 warmup "Arch unsupported for CUTLASS FP4 GEMM" 失败；rc3 仍为 prerelease，勿用于生产稳定版。
- **P1（静默/量化自查）**：llama.cpp SYCL 后端核对 `#31465`（CPY Q4_1/Q5_1/Q8_0 舍入与 CPU 对齐，已修但需确认测试稳定）；Metal 后端 GLM-5.3-Flash 等 `ne03>1` GROUP_NORM 核对 `#31852`（修复前静默垃圾）；SGLang Muse-Glimmer-NVFP4 核对 `#48128`（MXFP8 lm_head 加载，离 Blackwell 默认 block-FP8 转换）。
- **P1（PD/投机验收）**：SGLang DSpark/DFLASH PD 验收含 `#48375`（spec_kv_layout 匹配，拒绝不兼容）；vLLM Step5 MTP + PP>1 核对 `#66000`（启动修复 + CUDA graph 内存 0.69→4.74 GiB / KV 容量 983,723→925,755 tokens 权衡）。
- **P2（跟踪）**：Kimi-K3.1 CP 全栈在 SGLang 的实测精度/吞吐（#48709 gsm8k 0.954–0.958、TTFT 表）与后续端口（sliding-window MLA drafts/EAGLE3 MLA under DCP、PD+HiCache under DCP）；AMD Strix Halo/MI450 量化后端可用性与实测（#47695/#49001）；llama.cpp MoE-cache CPU share 调优（#30966 `moe-cache-tune.py`）。

> 证据等级说明：本档案「官方一手」指来自四家官方仓库的提交/PR/发布元数据与 PR 正文（经 GitHub issues API 抓取的 markdown）；「二手线索/未核验」指来自社区公告/HN/外部 GitHub 且未经一手来源确认的内容（如 TokiRecur、Jacq DSL）。所有性能/精度数字均标注来源 PR 或注明「描述级/CI 红/作者报告未独立验证/无 t/s 对照」，未外推未呈现数字。
