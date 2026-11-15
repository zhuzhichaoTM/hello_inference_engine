# 大模型推理优化每日热点档案

> 档案日期：2026-11-15（本自动化于该日执行）
> 关注窗口：2026-11-14（周六）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 经 pulls API 抓取 markdown 官方一手）+ HN Algolia（HTTPS，numericFilters 按 2026-11-14 epoch 窗口 1794614400–1794700799）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 49 / SGLang 60 / TRT-LLM 12 / llama.cpp 32 合计 **153** 条（均未触 per_page=100 上限，单页取全）；窗口内四家均无新正式版本（llama.cpp 推进 nightly b12380–b12387，prerelease）；**47 个关键 PR 正文经官方 pulls API（markdown 官方一手）抓取**，无 token 配额下 60/h 上限已被用尽（含重试 4 个截断文件），未触发二次截断；HN Algolia 引擎相关高信号集中于 llama.cpp v1.0.0 社区延续讨论、Qwen3.8-Flash-Next 社区长文、GLM-5.3 vLLM 部署实测、KV cache 工具 SCache/MOS，均二手线索/未核验。本窗口为周六，提交量较上一窗口（11-13：102/78/46/22=248）回落至 49/60/12/32=153，属周末正常回落，但 SGLang 仍保持 60 条活跃。

## 1. 版本状态与本期头号事件：SGLang 静默正确性铁证(GLM-5.3-Flash FP8 KV #47013) + vLLM MoE 路由 DP padding 默认屏蔽(#66799) + 推理服务默认安全加固(SGLang #49896) + TRT KV offload SSD 三项优化(#21822) + llama.cpp 量化静默延续(#32241) + 跨引擎 Rust 前端延续

**提交计数（2026-11-14 UTC，均 HTTP 200；2026-11-14 为周六，量较周五回落；vLLM 49 / SGLang 60 / TRT-LLM 12 / llama.cpp 32，合计 153）**：
- **vLLM 49** / **SGLang 60** / **TensorRT-LLM 12** / **llama.cpp 32**。

**发布状态（窗口内四家均无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-14 全部 49 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；`#50397` 仅 bump `sglang-kernel` 到 0.4.11 / `sgl-fa3` 到 0.1.0（子模块版本，非发行版）。
- **TensorRT-LLM**：dev 线仍为 **v1.4.0rc3（prerelease，2026-11-06）**；`releases/latest` 口径下最新稳定版为 v1.2.1（2026-04-20）；11-14 窗口无新版本。
- **llama.cpp**：v1.0.0 已于 2026-11-13 正式发布；窗口内仅推进 **nightly b12380–b12387**（2026-11-14T13:57–19:26Z，prerelease），无新正式版本。

**本期头号事件（SGLang 静默正确性铁证 + vLLM MoE 路由默认屏蔽 + 服务默认安全加固 + TRT KV offload + llama.cpp 量化静默 + Rust 前端延续）**：
1. **SGLang 静默正确性铁证——GLM-5.3-Flash + FP8 KV + TileLang DSA 半数答案变乱码，GSM8K 50%→98%（#47013，强化/补充第 4、6、8 条）**：自 #42689 起，GLM-5.3-Flash 在 FP8 KV cache + TileLang DSA backend 下、并发请求中约半数答案退化为一串 `!`（到 token 上限），GSM8K 掉到 50.0%；根因是 `set_mla_kv_buffer_kernel_norope`（zero-RoPE MLA KV writer，GLM-5.3-Flash 每条 KV 写都走它，因 `qk_rope_head_dim=0`）仍把 slot 0（padding 保留槽）写入，而自 #36003 起其余 MLA KV writer 已跳过该槽。修复后 GSM8K 98.0%。这是「KV writer / padding / scale 槽位静默写错」类错误的教科书级案例——**无任何报错、输出为乱码**，与历次量化的 scale/布局漏接（#42566/#49338/#32241）共构第 4/6 条主线。
2. **vLLM MoE 路由 DP padding 行默认屏蔽（#66799，强化第 12 条 MegaMoE/EP）**：grouped top-k 路由（DeepSeek-V3/V3.2、GLM-4.5+/5、Kimi-K2/K3、MiMo-V2 等）现默认给 DP 与 CUDA-graph padding 行置 expert id −1（此前仅 DeepEP v2 opt-in 模型如此，跟随 #65972 对 TRT-LLM block-FP8 / MXFP8 monolithic kernel 的同样处理），使 modular MoE kernel 跳过这些行。**Perf**：GLM-5.3-NVFP4（DP8）+5%~+21%（并发 8–256）、MiMo-V2.6-Flash（MXFP4 experts，DP4）等三模型两后端在几乎每个并发下吞吐更好。属「MoE 路由 padding 静默贡献无效计算」的默认修复。
3. **推理服务默认安全加固成为跨引擎主线（新增第 18 条）**：SGLang #49896（与 #49895 配套）把更新权重、加载/卸载 LoRA、profiling、控制生成、改运行时配置等敏感 admin 端点改为 `ADMIN_FORCE`：**无 `--admin-api-key`/`SGLANG_ADMIN_API_KEY` 即返回 403**，属 breaking 变更。这与 llama.cpp v1.0.0 的 `llama-server` 加固（SSRF/CORS/DNS rebinding，上期头号事件）及 vLLM 既有 admin 鉴权形成跨引擎「推理服务默认安全」共振；部署侧须显式设 admin key 否则管控端点全部 403。
4. **TRT-LLM KVCacheManagerV2 SSD 卸载三项优化（#21822，强化第 8 条）**：三项 opt-in 优化（并行 SSD I/O、restore 后保留 SSD 副本、write-through 到 SSD），均经 `KvCacheConfig` 默认关；PR 给出量化表——并行 I/O 提升吞吐、write-through 改善 TTFT p90。是 KV 卸载（第 8 条）从「正确性」走向「吞吐/TTFT 可调优」的实证。
5. **llama.cpp 量化静默正确性延续（#32241，强化第 4、6 条）**：`qwen2vl` 图把三个 FFN scale 参数（`ffn_up_s`/`ffn_gate_s`/`ffn_down_s`）传 `NULL` 给 `build_ffn`，导致通用 loader 已加载的 per-tensor NVFP4 scales **从未应用**，`nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` 产出垃圾且无警告（修复 `Fixes #32101`）。同类「量化 scale 漏接 → 静默垃圾」再次出现在第四家引擎，是第 4/6 条的铁证延续；wikitext-2 PPL 表（master vs PR）已附。
6. **跨引擎 Rust 服务前端延续（强化第 17 条）**：SGLang #50401（rust-server 在 `--enable-http2` 下于同端口提供 h2c/HTTP/2 并行于 HTTP/1.1，使 `sgl-router` 能走 h2c）、#49630/#49634（Rust 树核心/配置目录）；vLLM #63580（Rust frontend 工具解析禁用时返回 400 而非 500，gRPC Render 端点同步）、#67282（SIGTERM 仅管托管引擎 launcher、在途请求排空）。服务层 Rust 化在周末仍双引擎推进。

---

## 2. vLLM：2026-11-14 提交（49 条）—— MoE 路由 DP padding 默认屏蔽(#66799) + NVFP4 expert scales CSF 压缩留 GPU(#64753) + DBO 纯 TP 双批次重叠(#63483) + int64 token index(#64515) + NIXL 从 PP prefill 拉 KV(#65336) + dummy FP8 weights(#67693) + Transformers backend Qwen2(#66821)

### 2.1 `#66799` [Perf][MoE] Mask DP padding rows in grouped top-k routing by default（grouped top-k 路由 DP padding 行默认置 −1，perf，强化第 12 条）
- **主题**：grouped top-k 路由（DeepSeek-V3/V3.2、GLM-4.5+/5、Kimi-K2/K3、MiMo-V2 等）现默认给 DP 与 CUDA-graph padding 行置 expert id −1，与 fused top-k 路由一致，使 modular MoE kernel 跳过这些行；此前仅 DeepEP v2 opt-in 模型如此（来自 #56079），跟随 #65972 对 TRT-LLM block-FP8/MXFP8 monolithic kernel 的同样处理。
- **内容详情**：**Perf**（DP+EP 服务吞吐，三模型两后端）：GLM-5.3-NVFP4（DP8）+5%~+21%（并发 8→256）；MiMo-V2.6-Flash（MXFP4 experts，DP4）等同样在几乎每个并发更好。不改变任何模型输出（padding 行本就不参与有效计算，仅省无效 dispatch/计算）。
- **应用场景**：vLLM grouped top-k MoE 在 DP+EP 部署的无效 padding 行开销消除。
- **研判**：【事实】PR 给 GLM-5.3-NVFP4（DP8）+5~21% 与 MiMo-V2.6 的吞吐对照，声明输出不变。【观点】属「MoE 路由 padding 静默无效计算」默认修复（第 12 条 EP/MegaMoE），纯免费收益；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #66799（合并于 2026-11-14T02:18:28Z）。

### 2.2 `#64753` [Quantization][MoE] Keep NVFP4 expert scales CSF-compressed on the GPU with B12X（NVFP4 专家 scales CSF 压缩留 GPU，feature/perf，强化第 13 条）
- **主题**：B12X NVFP4 MoE 后端下，把 `nvidia/Qwen3.8-Flash-Next-FP4-Local` 的 CSF-compressed 路由专家 block scales 压缩保留在 GPU 上：省 **−2.57 GiB** 权重显存、**+18.7% KV cache** 容量（单张 RTX PRO 6000 Max-Q）、DGX Spark 上 **+2.4 GiB** 空闲显存，GSM8K 不变。
- **内容详情**：构建于 #64312（加载时解码 CSF scales）之上；B12X 在加载时用 FlashInfer `encode_csf_scales` 重编码每 rank 解码后 scales 为 `CsfScalePlanes`，每调用仅展开路由专家 scales 到一个 per-layer-shape scratch；Hub checkpoint 还需 #64739（shards 在 `tensors/` 下）。
- **应用场景**：Qwen3.8-Flash-Next NVFP4 本地模型的显存/容量优化。
- **研判**：【事实】PR 给 −2.57 GiB、+18.7% KV、+2.4 GiB 空闲与 GSM8K 不变。【观点】属「NVFP4 低比特 KV/scale 显存优化」（第 13 条），压缩保留 scales 换 KV 容量；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #64753（合并于 2026-11-14T12:38:19Z）。

### 2.3 `#63483` [Core][DBO] Dual batch overlap under pure tensor parallelism (split PR of M3.1)（纯 TP 双批次重叠 DBO，feature/perf，强化第 11 条 ROCm）
- **主题**：M3.1（#63232，模型本身以 #64111 落 main）的引擎部分 ROCm 工作——在 Model Runner V2 上、纯张量并行（单 DP rank、无 EP）下的 dual batch overlap：每个 microbatch 的 TP all-reduce 跑在 comm stream 上，另一个 microbatch 同时计算。
- **内容详情**：`--enable-dbo` 配 `--tensor-parallel-size N`(N>1) 且无 DP/EP 在 MRV2（`VLLM_USE_V2_MODEL_RUNNER=1`）工作；与 #44677（MRV1 同目标）、#63373（移除 MRV1）相关。属 M3.1 ROCm 性能工作的一部分。
- **应用场景**：vLLM ROCm 上纯 TP 部署的 DBO 通信/计算重叠。
- **研判**：【观点】属「ROCm DBO/通信重叠」（第 11 条），纯 TP 路径补齐（此前 DBO 主要在 EP）；记为官方一手（feature/perf 级，description 级：未给具体提速数字）。**证据等级**官方一手（feature 级，perf 描述级）。
- **来源**：vllm-project/vllm #63483（合并于 2026-11-14T19:22:29Z）。

### 2.4 `#64515` [Bugfix][XPU] Use int64 token index in the DeepSeek-V4 qnorm/RoPE kernel（DeepSeek-V4 qnorm/RoPE 用 int64 token index，fix，强化第 9 条 int 溢出）
- **主题**：`_xpu_qnorm_rope_kernel` 在 int32 算 `token_idx * num_heads * HEAD_DIM`，单步持有超过 2^31 个 Q 元素（65,536 token × 64 local heads）时 Q 地址回绕；Intel Arc Pro B60 上以 `UR_RESULT_ERROR_DEVICE_LOST` 告终（1M-context DeepSeek-V4 在 TP=1、`--no-enable-chunked-prefill` 下命中）。Fixes #52416。
- **内容详情**：把 `token_idx` 提升为 int64（同文件 sibling kernel `fused_inv_rope_fp8_quant.py`/`fused_mtp_input_rmsnorm.py` 已用 int64）。属 int32 溢出静默错误主线（第 9 条）在 XPU 后端的新实例。
- **应用场景**：DeepSeek-V4 在 Intel XPU 的长上下文 qnorm/RoPE 地址正确性。
- **研判**：【事实】PR 给「65,536 token × 64 heads 触发 int32 回绕 → DEVICE_LOST」根因与 int64 修复，Fixes #52416。【观点】属「长上下文 int32 溢出静默错误」（第 9 条），跨第四家后端（XPU）印证；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：vllm-project/vllm #64515（合并于 2026-11-14T11:12:57Z）。

### 2.5 `#65336` [KV Connector][NIXL] Pull KV from pipeline-parallel prefill stages, with the hybrid KV cache manager（从 PP prefill 拉 KV，feature，强化第 8 条）
- **主题**：支持从 pipeline-parallel prefill 实例（P）向 PP1 解码实例（D）拉 KV 传输，含 attention-only hybrid KV cache groups（HMA）；解决 #52913 的 prefill 侧 PP 情形，复用 #63416 的 layer-aware region pairing。PP 各 stage 持有不同层、可能不持某些 cache group 的层；D 须为每次传输选持有该层的 stage。
- **内容详情**：统一 remote-stage 状态（采用 @njhill 的 remote-stage refactor）；所有 remote engine 用 `_remote_`；按 layer 配对选用持有层与 HMA 的 stage；沿用 hybrid KV cache manager。
- **应用场景**：vLLM PD 解耦中 prefill 侧为 PP 的 KV 传输（含 hybrid/attention-only 组）。
- **研判**：【观点】属「PD KV 传输 / hybrid KV 管理」（第 8 条），补齐 PP prefill 路径；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #65336（合并于 2026-11-14T02:17:29Z）。

### 2.6 `#64827` [DeepSeek-V4] Warm the DeepGEMM kernels DeepSeek-V4/V4.1 JIT-compile at runtime（DeepSeek-V4/V4.1 DeepGEMM kernel 预热，fix，强化第 11 条）
- **主题**：DeepSeek-V4/V4.1 在实时请求上 JIT 编译若干 DeepGEMM kernel，导致生产服务中实际服务客户时触发 JIT，危害服务质量（强烈影响生产 DeepSeek 服务）；DP+EP 下每 rank 等编译 rank，少数编译使单步超 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`(300s) → RPC `sample_tokens` 超时 → EngineDeadError（#41530 症状）。
- **内容详情**：预热此前被漏预热的 kernel。属「推理时 JIT 编译拖累首请求/超时」的修复。
- **应用场景**：DeepSeek-V4/V4.1 在 vLLM 的生产部署（避免实时 JIT）。
- **研判**：【观点】属「DeepSeek 生产服务稳定性」（第 11 条），JIT 预热防 EngineDeadError；记为官方一手（fix 级，description 级：未给具体预热耗时）。**证据等级**官方一手（fix 级，描述级）。
- **来源**：vllm-project/vllm #64827（合并于 2026-11-14T02:01:48Z）。

### 2.7 `#67693` [Model Loader] Fill dummy FP8 weights with random codes across the range（dummy FP8 权重填随机有限码，fix/perf，强化第 11 条）
- **主题**：`--load-format dummy` 的 dummy FP8 权重改为跨 FP8 范围的随机有限码（而非 ±1 值），使计算受限步骤的 dummy-weight 基准在功耗受限 GPU 上与真实权重一致。
- **内容详情**：**Perf**（GB200，DeepSeek-V4.1-Flash，DEP4）：大 prefill 步 dummy-weight 计时此前比真实权重快至多 11%，改后落在 run-to-run 噪声内。`low`/`high` 仍均匀采样；FP8 值始终有限（四 float8 dtype 均无 NaN/Inf）、对 seed 确定、原地填充、无 FP16 staging tensor。24 次真实设置验证。
- **应用场景**：vLLM dummy-weight 基准的准确性（功耗受限 GPU 下）。
- **研判**：【事实】PR 给 GB200 大 prefill 步 dummy 比真实快 ≤11% → 改后噪声内，24 次验证。【观点】属「基准保真」（第 11 条），避免 dummy 权重误导 perf 结论；记为官方一手（fix/perf 级，含对照）。**证据等级**官方一手（fix/perf 级，含对照）。
- **来源**：vllm-project/vllm #67693（合并于 2026-11-14T22:22:19Z）。

### 2.8 `#66401` [Bugfix][MRV2] Allow non 8-bytes aligned `--max-num-batched-tokens` for DeepSeek-v4（MRV2 支持非 8 字节对齐 max-num-batched-tokens，fix，强化第 3 条默认值/兼容性）
- **主题**：MRV2 把所有 KV cache group 的 slot mapping 放在一个 `[num_kv_cache_groups, max_num_batched_tokens]` int64 tensor；DeepSeek-V4 的 CuTe DSL compressor 声明 `slot_mapping` 64 字节对齐（每元素 8 字节，故 `max_num_batched_tokens` 须 8 字节对齐），在 engine init 报 "Misaligned Tensor data on argument #3"（如 `--max-num-batched-tokens 1500/2500`）。
- **内容详情**：把 slot-mapping buffer 的行 stride pad 到 16 元素，使每行 128 字节对齐，保留原布局；之后 DeepSeek-V4 支持任意值（满足 8 对齐的位置不变）。
- **应用场景**：DeepSeek-V4 在 MRV2 的 `max_num_batched_tokens` 兼容性。
- **研判**：【观点】属「MRV2/DeepSeek-V4 对齐约束放宽」（第 3 条），纯兼容性修复；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：vllm-project/vllm #66401（合并于 2026-11-14T16:02:27Z）。

### 2.9 `#66821` [Transformers backend] Serve Qwen2 with the Transformers modeling backend（Transformers backend 接管 Qwen2，refactor/feature）
- **主题**：用 Transformers modeling backend 服务 Qwen2；删除原生 `vllm/model_executor/models/qwen2.py`，仍原生构建的模型改用配置为 Qwen2 attention 的 Qwen3 类。
- **内容详情**：`Qwen2ForCausalLM`/`Qwen2Model` checkpoint 解析到 Transformers backend；原生 Qwen3 接管 decoder 栈（`Qwen3MLP`/`Qwen3Model` 即前 Qwen2 实现，`Qwen3Attention` 使 q/k-norm 可选）；`use_qwen2_attention(config)` 设 `attention_bias=True` 且默认关 q/k-norm（除非）。`#67455` 配套把 Transformers backend 默认 decoder pooling 设为 LAST 并尊重 `is_causal=False`。
- **应用场景**：vLLM 模型接入层向 Transformers backend 统一。
- **研判**：【观点】属「模型后端统一/重构」，减少原生模型实现重复；记为官方一手（refactor/feature 级）。**证据等级**官方一手（refactor 级）。
- **来源**：vllm-project/vllm #66821（合并于 2026-11-14T15:38:53Z）。

### 2.10 `#67291` [Spec Decode][MRV2] Build no draft attention metadata on a context-K/V-only prefill（context-K/V-only prefill 不建 draft attention 元数据，perf，强化第 14 条）
- **主题**：#65217 follow-up。`prefill_context_kv_only` 下，P/D prefill 实例上的 DFlash 族 drafter 只写 context K/V、从不跑 draft attention，但 target runner 与 drafter 仍给 draft 层建元数据 builder；FlashInfer draft builder 在内存 profile 后分配按整 batch 尺寸的 prefill workspace（`max_num_batched_tokens × query heads × head_dim × 16 B`），target runner 每步也建 draft 元数据。本 PR 保留 draft attention group 与 kernel-block layout 但不建其 builder。
- **内容详情**：在 context-K/V-only prefill 实例上省 workspace 分配与每步元数据构建开销。
- **应用场景**：DFlash 族投机在 P/D prefill 实例的元数据开销削减。
- **研判**：【观点】属「投机解码元数据开销」（第 14 条），context-K/V-only prefill 路径优化；记为官方一手（perf 级，description 级：未给具体数字）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：vllm-project/vllm #67291（合并于 2026-11-14T06:16:51Z）。

### 2.11 `#67591` [Perf] Allow EPLB with modular TRT-LLM FP8 block-scale MoE（modular TRT-LLM FP8 block-scale MoE 允许 EPLB，perf/feature，强化第 12 条）
- **主题**：`TrtLlmFp8ExpertsModular._supports_parallel_config` 拒绝 `enable_eplb`（来自 MoE refactor #32564，无注释/issue/报错）；monolithic 类因 kernel 内路由只支持 naive DP/EP 与 TP。modular 类不自路由，取 router 的 EPLB-mapped 物理 expert id（`num_experts`/`local_expert_offset` 为物理单位），其 BlockMajorK 权重与 block scales 保持 expert-major，故 EPLB 的 per-expert dim-0 搬移有效。本 PR 去掉该拒绝。
- **内容详情**：modular TRT-LLM FP8 block-scale MoE 现可与 EPLB 共用（expert 负载均衡）。
- **应用场景**：vLLM modular TRT-LLM FP8 MoE 的 EPLB 支持。
- **研判**：【观点】属「MoE EPLB/负载均衡」（第 12 条），补齐 modular 类 EPLB 能力；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #67591（合并于 2026-11-14T13:49:24Z）。

### 2.12 `#67599` [Bugfix][AuxOutput] Size the routed-experts row cache from one source（routed-experts row cache 单源定尺寸，fix，强化第 8 条）
- **主题**：routed-experts row cache 在两处尺寸不一致：scheduler（`AuxOutputSchedulerConnector._capacity`）按 `num_blocks` KV blocks 行、配 KV connector 时翻倍、忽略 `aux_output_config.max_bytes`；worker store（`initialize_output_storage`）在设 `max_bytes` 时用之、否则 `num_blocks × hashes_per_kv_block`、无 KV-connector 翻倍。设 `max_bytes` 只改 worker：scheduler 仍保留等于 KV cache 的 blocks 并继续服务 prefix hits，worker 却长到超过 `max_bytes`。
- **内容详情**：统一为单源尺寸；配 KV connector + `max_bytes` 时行为正确收敛。
- **应用场景**：vLLM auxiliary output（routed-experts）row cache 的尺寸一致性（prefix cache 正确性）。
- **研判**：【观点】属「aux output / prefix cache 尺寸一致性」（第 8 条），两源分歧致 cache 超配；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：vllm-project/vllm #67599（合并于 2026-11-14T13:46:25Z）。

### 2.13 其他要点（标题级 / 描述级 / 性能级）
- `#67636` [CI][Test] Stub pipeline_relay in the engram DP-shard model forward test（engram DP-shard 测试 stub pipeline_relay，ci 级）。
- `#67545` [Bugfix][Core] Keep dummy-run requests within max_model_len（dummy-run 请求约束在 max_model_len，fix 级）。
- `#64255` [XPU][MLA] Register XPUFlashAttnMLABackend for FLASH_ATTN_MLA in XPU（XPU MLA 后端注册，feature 级，强化第 15 条 XPU）。
- `#67505`/`#67426` [MM] Support mm_device_do_normalize for LLaVA-OneVision / Qwen3-Omni（多模态 device normalize，feature 级）。
- `#67273` [Chore] Update agent skill for vLLM（agent skill 更新，chore 级）。
- `#67601` [CI][XPU] Bound DeepStack test profiling to avoid OOM（XPU DeepStack 测试限 profiling 防 OOM，ci 级）。
- `#66933`/`#66934` [ROCm] M3.1 GEMM improvements / Minimax M3.1 attn indexer XCD balancing（ROCm M3.1 GEMM / attn indexer 均衡，perf 级，强化第 11 条）。
- `#67600` [Recipe] Kimi-K3.1-Pro: MI355X AgentX preset covering all six InferenceX#5043 concurrencies（Kimi-K3.1-Pro MI355X AgentX recipe，recipe 级，强化第 11 条）。
- `#67474` [Bugfix][Multimodal] Size the MOSS-Audio profiling dummy to its audio budget（MOSS-Audio profiling dummy 按音频预算定尺寸，fix 级）。
- `#67105` [Bugfix][Audio] Fall back to soundfile when torchcodec fails to decode（audio 解码 torchcodec 失败回退 soundfile，fix 级）。
- `#67282` [Bugfix][Rust Frontend] SIGTERM only the managed engine's launcher so in-flight requests drain（Rust frontend SIGTERM 仅管托管引擎 launcher、在途请求排空，fix 级，强化第 17 条）。
- `#64542` [Perf] Optimize Garbage Collection at Init Time（init 时 GC 优化，perf 级）。
- `#65098` [ROCm][Disagg][CI] Give Wide-EP headless nodes the KV config（Wide-EP headless 节点给 KV config，ci 级，强化第 8 条）。
- `#67564` [CI][Multimodal] Use `uuid:`-tagged cache key for features in /generate test（多模态 /generate 测试 cache key，ci 级）。
- `#66925` [Feature][EPD] Support Kimi-K3 image metadata-only inputs（Kimi-K3 仅图像 metadata 输入，feature 级）。
- `#65453` [Bugfix][Multimodal] Repeat-last default padding for `cu_seqlens` in multimodal CG（多模态 CG cu_seqlens repeat-last padding，fix 级）。
- `#67579` [CI][Bugfix] Fix fake backend in drafter kernel block size test（drafter kernel block size 测试 fake backend 修复，ci 级）。
- `#67577` [ROCm] Keep the extensible KV cache for fd-exporting KV connectors（ROCm 为 fd-exporting KV connector 保留 extensible KV cache，fix 级，强化第 8 条）。
- `#67321` [Bugfix][Test] Avoid capfd in test_pp_rejects_split_kv_sharing（test 避免 capfd，test 级）。
- `#67519` [Bugfix] Fix EAGLE-3 on Transformers backend VLMs（Transformers backend VLM 上 EAGLE-3 修复，fix 级，强化第 14 条）。
- `#67455` [Transformers backend] Default decoder pooling to LAST and honour `is_causal=False`（Transformers backend decoder pooling 默认 LAST，feature 级）。
- `#67297` [Perf] Use `slots=True` on all internal dataclasses（内部 dataclass 用 slots=True，perf 级）。
- `#67308`/`#67439`/`#67555`/`#67317`/`#66314` [ROCm][CI] MI300/MI355 队列 / TP8 mono build / sparse MLA / AMD mirrors / Python 3.14 smoke（ROCm CI 级）。
- `#43638` [BugFix] Fix Mistral3 multimodal offline example（Mistral3 多模态离线示例修复，fix 级）。

---

## 3. SGLang：2026-11-14 提交（60 条）—— 静默正确性铁证(GLM-5.3-Flash FP8 KV #47013) + Rust tree core joint eviction(#49992/#49641) + MXFP8 KV 外部 linker(#50324) + out-of-tree HiCache 分片(#50370) + MoE router top_k 漏传(#49805) + admin 端点鉴权(#49896) + Rust HTTP/2(#50401) + NPU DSA token shard(#42978)

### 3.1 `#47013` [Fix] kvcache: Skip the reserved padding slot in the zero-RoPE MLA KV writer（zero-RoPE MLA KV writer 跳过保留 padding 槽，fix，强化第 4、6、8 条，本期头号静默错误）
- **主题**：**使 GLM-5.3-Flash 在 FP8 KV cache + TileLang DSA backend 下输出正确**。自 #42689 起，并发请求中约半数答案退化为一串 `!`（到 token 上限），GSM8K 掉到 **50.0%**；本 PR 后 **98.0%**。
- **内容详情**：自 #36003 起 MLA KV writer 跳过 padding 行保留槽（slot 0）；但 zero-RoPE writer `set_mla_kv_buffer_kernel_norope`（GLM-5.3-Flash 每条 KV 写都走它，因 `qk_rope_head_dim=0`）仍写该槽。本 PR 让它同样跳过。修复后 Full/GSM8K 恢复。**无任何报错、输出为乱码**是典型的静默错误铁证。
- **应用场景**：GLM-5.3-Flash 在 FP8 KV + TileLang DSA backend 的正确性（并发）。
- **研判**：【事实】PR 给「自 #42689 起并发半数乱码、GSM8K 50.0%→98.0%」与根因（zero-RoPE MLA KV writer 写 slot 0 padding）。【观点】属「KV writer / padding 槽静默写错」（第 4/6/8 条），与 #42566/#49338/#32241 共构量化/MoE 静默正确性主线；记为官方一手（fix 级，含根因 + 精度对照）。**证据等级**官方一手（fix 级，含根因 + 对照）。
- **来源**：sgl-project/sglang #47013（合并于 2026-11-14T12:02:49Z）。

### 3.2 `#49992` [Feat] mem_cache: Joint Full/SWA eviction walk on the Rust tree core（Rust tree core joint Full/SWA 淘汰，feature，强化第 8 条）
- **主题**：`--radix-eviction-policy joint_lru` 在 unified memory pool（#46827）把 full-attention 与 SWA KV 排进同一访问序；此前只有 Python tree core 有 joint eviction walk，而默认 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust` 的 Rust tree core 无此 walk，构建即拒该策略。
- **内容详情**：给 Rust tree core 同样的 walk，使 `joint_lru` 在默认 Rust tree core 可用；stacked on #49721。
- **应用场景**：SGLang unified memory pool 下 Full/SWA KV 联合 LRU 淘汰（默认 Rust 树核心）。
- **研判**：【观点】属「KV 联合淘汰 / unified memory pool」（第 8 条），补齐 Rust 树核心能力；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #49992（合并于 2026-11-14T01:53:13Z）。

### 3.3 `#49641` [Fix] radix-tree: Clamp SWA tombstone recovery to the protected prefix in the Rust tree core（Rust tree core SWA tombstone 恢复钳制到保护前缀，fix，强化第 8 条）
- **主题**：Rust tree core 可能释放一个节点仍映射的 FULL KV 页；请求仅从其保护前缀（`prev_prefix_len`）向上拥有 FULL 页，其下是节点自有 canonical page id 的引用行。`SwaComponent::update_component_on_insert_overlap` 从该行以请求的 SWA eviction cursor（`swa_evicted_seqlen`）恢复 SWA-tombstoned 节点，而 cursor 可能落在保护前缀之下：只读的 FULL 前缀匹配可在节点内严格结束、尚未 evict 到那远的请求 cursor 在其后。
- **内容详情**：恢复覆盖了请求从未拥有的页，被交到 free path 当作可释放——可能释放别的请求正在用的页。修复把恢复钳制到保护前缀，避免误释放。
- **应用场景**：Rust tree core 下 SWA tombstone 恢复的页所有权正确性（KV 安全）。
- **研判**：【观点】属「radix/KV 页所有权与释放正确性」（第 8 条），误释放他请求页是严重静默错误风险；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：sgl-project/sglang #49641（合并于 2026-11-14T01:48:20Z）。

### 3.4 `#50324` [Feat] unified_cache: Support MXFP8 KV caches in the direct external linker（unified_cache MXFP8 KV 在 direct external linker 支持，feature，强化第 13 条）
- **主题**：`--kv-cache-dtype mxfp8`（Inkling 长上下文 recipe）在 `--enable-unified-cache-external-linker` 下启动失败——direct linker 拒绝任何带 scale buffer 的 KV pool。
- **内容详情**：`_mha_device_pool_entry` 把 MXFP8 K/V 的 UE8M0 scale buffer 打包进 KV page object（在 FP8 payload 之后）；只整页对齐移动、每页 scales 在 interleaved（`page_size=128`）与 flat 布局都连续，故每个 scale buffer 作与 payload slot 对齐的 uint8 行视。`FP4 scaled KV pool` 仍拒绝。无新 pool 名/存储键。**Accuracy**：Inkling（`--revision test`，1× B200，fa4，page size 128，deterministic）已验证。
- **应用场景**：Inkling 长上下文 MXFP8 KV 在 unified_cache external linker 的正确性。
- **研判**：【观点】属「MXFP8 低比特 KV + unified cache」（第 13 条），scale buffer 与 payload 打包移动；记为官方一手（feature 级，含精度）。**证据等级**官方一手（feature 级，含精度）。
- **来源**：sgl-project/sglang #50324（合并于 2026-11-14T09:44:44Z）。

### 3.5 `#50370` [Feat] kv-shard: Let an out-of-tree HiCache storage backend join logical-page KV sharding（out-of-tree HiCache 存储后端加入逻辑页 KV 分片，feature，强化第 8 条）
- **主题**：`--enable-kv-cache-sharding` 在启动与运行时 attach 都拒所有 `--hicache-storage-backend`（因 in-tree 路径按整页/每 rank stripe 读写，不匹配）；一个只读/写本 shard rank 页的存储后端仍无法参与——因无可承认其名、且 buffer 模式下 storage pipeline 分配 host staging slot 无 owner class（`PageInterleaveHostPool.alloc` 断言须传 owner）。
- **内容详情**：加承认机制 + owner class 传递，使合规 out-of-tree HiCache 存储后端可加入逻辑页 KV 分片（与 vLLM #61452 / SGLang #49636 的 out-of-tree 扩展点呼应）。
- **应用场景**：第三方/out-of-tree HiCache 存储后端接入 SGLang 逻辑页 KV 分片。
- **研判**：【观点】属「KV 分片扩展点」（第 8 条），与 vLLM #61452 / SGLang #49636 形成跨引擎共振；记为官方一手（feature 级，扩展点）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #50370（合并于 2026-11-14T02:09:24Z）。

### 3.6 `#49805` [Fix] models: Forward router top_k to fused MoE experts（路由 top_k 转发到 fused MoE，fix，强化第 7 条）
- **主题**：OLMoE、Hunyuan、Llama 4 配置其 `TopK` router 但在构造 `FusedMoE` 时漏传 `top_k`，致 expert 层与其 `MoeRunnerConfig` 的 `top_k` 保留 `None`（含传给消费 expert 数的 dispatcher 的元数据）。
- **内容详情**：在三个模型中把各自 router 现有 expert count 转发到 `FusedMoE`；Hunyuan 用已解析的 per-layer 值；扩展现有 CPU 模型测试。补 #19287/#31782 的 NPU runtime fallbacks（在源头提供现有模型配置）。相关 #17120（NPU 专属 OLMoE 构造器，未合）。
- **应用场景**：OLMoE/Hunyuan/Llama4 的 fused MoE 路由 top_k 正确性（漏传 → 静默 None）。
- **研判**：【观点】属「MoE 路由精度/放置」（第 7 条），漏传 top_k 致 dispatcher 元数据错（静默）；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：sgl-project/sglang #49805（合并于 2026-11-14T12:02:34Z）。

### 3.7 `#49896` [Feat] entrypoints: Require --admin-api-key for sensitive admin endpoints（admin 端点需 --admin-api-key，breaking 安全，新增第 18 条）
- **主题**：**Breaking change。** 更新权重、加载/卸载 LoRA、profiling、控制生成、改运行时配置等端点现改为 `ADMIN_FORCE`：无 admin key（`--admin-api-key` 或 `SGLANG_ADMIN_API_KEY`）即返回 **403**，且只接受该 key。
- **内容详情**：覆盖 `/update_weights_*` 及权重更新组/session 端点、`/get_weights_by_name`、`/weights_checker`、`/load_lora_adapter*`、`/unload_lora_adapter`、`/start_profile`、`/stop_profile`、expert-distribution recorder 端点、`/set_internal_state`、`/configure_logging`、`/dumper/*`、`/hicache/storage-backend`（attach/detach/status）、`/pd_role_switch`、`/scale_elastic_*`。与 #49895（解析包含 router 的 auth 级别、为每个路由钉级别）配套。
- **应用场景**：SGLang 推理服务的 admin 端点默认鉴权（部署安全）。
- **研判**：【事实】PR 明确列出 breaking 变更与 403 行为。【观点】属「推理服务默认安全加固」（第 18 条），与 llama.cpp v1.0.0 server 加固、vLLM admin 鉴权形成跨引擎共振；部署侧须显式设 admin key；记为官方一手（breaking 级）。**证据等级**官方一手（breaking 级）。
- **来源**：sgl-project/sglang #49896（合并于 2026-11-14T23:32:05Z）。

### 3.8 `#50401` [Feat] rust-server: Serve HTTP/2 (h2c) beside HTTP/1.1 with --enable-http2（rust-server HTTP/2 h2c，feature，强化第 17 条）
- **主题**：`SGLANG_RUST_SERVER=1` 下 launcher 原拒 `--enable-http2`，Rust frontend 的 `/server_info` 无论 flag 都报 `enable_http2: false`（其 HTTP listener 仅 HTTP/1.1）；使求 HTTP/2 的启动命令无法切 Rust frontend，且 `sgl-router` 只对报 `enable_http2: true` 的 worker 走 h2c。
- **内容详情**：Rust HTTP listener 在 `--enable-http2` 下同端口以 h2c（prior knowledge，明文）并行于 HTTP/1.1 提供服务，使 sgl-router 可对 Rust frontend worker 走 h2c。
- **应用场景**：SGLang Rust 服务前端的 HTTP/2 支持（与 sgl-router 协同）。
- **研判**：【观点】属「服务前端 Rust 化特性 parity」（第 17 条），补齐 HTTP/2；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #50401（合并于 2026-11-14T02:56:26Z）。

### 3.9 `#42978` [Feat] [NPU] dsa: Token shard, every head for this rank's slice of the tokens (2/2)（NPU DSA prefill token 分片，feature，强化第 13、15 条）
- **主题**：NPU DSA prefill 中 attention TP 原按 head 分：每 rank 算其 1/`attn_tp_size` 的 head 但每 token 都读/反量化 top-2048 KV 行，KV 读（稀疏 attention 主导）在每 rank 重复。本 PR（#42977 的 2/2，parent 已合）把分片只换 attention core：每 rank 算所有 head 但只对其 token 切片，权重仍 head-sharded。
- **内容详情**：仅 attention core 在 `w_kc` absorb 与 `w_vc` 之间变化，KV 读从「每 rank 每 token 全量」变为「每 rank 仅其 token 切片」，降低 NPU 上 DSA prefill 的 KV 读量。
- **应用场景**：Ascend（华为）NPU 上 DSA prefill 的 token 分片（降 KV 读）。
- **研判**：【观点】属「NPU 稀疏 attention / DSA prefill 优化」（第 13、15 条），KV 读降冗余；记为官方一手（feature 级，description 级：未给具体数字）。**证据等级**官方一手（feature 级，描述级）。
- **来源**：sgl-project/sglang #42978（合并于 2026-11-14T19:27:29Z）。

### 3.10 `#49905` [Perf] gdn: Fuse MIS mapped gates into QKV preparation（GDN MIS gate 融进 QKV 准备，perf，强化第 11 条）
- **主题**：降 GDN multi-item scoring（MIS）的准备开销，构建于 #48585。每 query/item 段已有到原 token 行的 map；baseline 收集对应 `a`/`b` gate 行、卷积后 QKV 拆分/归一、再单独 gating launch。
- **内容详情**：在 QKV 准备内通过该 map 直接读 gate 行：**Per query/item 段**：QKV split 与 Q/K 归一（同准备 kernel，扩展含 gate 计算）→ 选 `a`/`b` 行（两 gather temp tensor → 直接 mapped loads）→ 算 `g`/`beta`（单独 gating kernel → 融入）。降准备 kernel 数与临时 tensor。
- **应用场景**：SGLang GDN MIS 多 item 评分的准备开销削减。
- **研判**：【观点】属「GDN/线性注意力准备融合」（第 11 条），降 kernel 数；记为官方一手（perf 级，description 级：未给具体数字）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：sgl-project/sglang #49905（合并于 2026-11-14T12:42:18Z）。

### 3.11 `#50318` [Perf] qwen4-exp: gather prefill PLE rows on the host for the file-backed table（qwen4-exp PLE 行在 host 收集，perf，强化第 11、13 条）
- **主题**：`--ple-offload-backend file`（#37068）下，Qwen3.8-Flash-Next 把 47.7 GiB PLE n-gram 表放文件映射 mmap、gather kernel 直读；自 #48689 起含离散 GPU（经 HMM 读 pageable 内存，如 open kernel modules 的 RTX 50 系）。配合 paged experts（#42557/#48642/#49195）可在 16 GB 卡上服务全模型。
- **内容详情**：此类 GPU 上，每页 GPU 未映射的表行即一次 page fault（驱动一次服务一页，约 0.5 ms/页）；prefill 每 token 收集 16 行、多来自不同页 → 大量串行 page fault 拖慢。本 PR 把 PLE 行在 host 收集再传，降 page fault 次数。
- **应用场景**：Qwen3.8-Flash-Next PLE n-gram 表在离散 GPU（16 GB 卡）的 prefill 性能。
- **研判**：【观点】属「PLE 卸载 / 离散 GPU 内存墙」（第 11、13 条），host 收集降 page fault；记为官方一手（perf 级，description 级：未给具体数字）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：sgl-project/sglang #50318（合并于 2026-11-14T12:03:36Z）。

### 3.12 `#50544` [Feat] dp-controller: Forward a scheduler's router_l3_layout entry in the startup handshake（dp-controller 转发 router_l3_layout，feature，强化第 8 条）
- **主题**：scheduler 在启动握手 `get_init_info()` 答 parent；单 DP rank 时 parent 收完整回复、Python frontend 纳入 `/server_info`；多 DP rank 时 schedulers 答 DP controller，`DataParallelController.get_init_info()` 只从其命名键重建回复（token limits、启动时间、合并 `hicache_object_layout`），scheduler 报的其他 entry 停于 controller。
- **内容详情**：故带 `router_l3_layout` entry 的 scheduler 在单 DP rank 暴露、多 DP rank 丢失。本 PR 让 dp-controller 转发该 entry。
- **应用场景**：SGLang DP 多 rank 下 router_l3_layout 信息透传（HiCache/路由布局）。
- **研判**：【观点】属「DP 控制器信息透传」（第 8 条），多 DP rank 下 entry 丢失修复；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #50544（合并于 2026-11-14T21:42:38Z）。

### 3.13 其他要点（标题级 / 描述级 / 性能级）
- `#50528` [CI] mem_cache: Scope the rendezvous mock in the unmapped-rank gather tests（mem_cache rendezvous mock 作用域，ci 级）。
- `#50321` [Fix] diffusion: Stop the multi-rank IPC all-to-all leak with one IPC handle per peer（diffusion 多 rank IPC all-to-all 泄漏修复，fix 级）。
- `#49270` [Perf] diffusion: Hy-Nomas fuse groupwise INT8 weight dequantization（diffusion Hy-Nomas INT8 权反量化融合，perf 级）。
- `#50007` [Fix] qwen-vl: Bound GPU memory of image preprocessing (fixes #49796)（qwen-vl 图像预处理 GPU 内存上限，fix 级）。
- `#50345` [Fix][NPU] pp: Avoid output AllGather during graph replay（NPU pp graph replay 避免输出 AllGather，fix 级）。
- `#50034` [Doc] hybrid: Document hybrid cache checkpoint levels（hybrid cache checkpoint 级别文档，doc 级）。
- `#49915` [Fix] sampling: Reject expensive Python stop regexes（sampling 拒昂贵 Python stop 正则，fix 级）。
- `#46781` [Perf] spec: Avoid blocking H2D for MRoPE delta in spec decode（投机解码 MRoPE delta 避免阻塞 H2D，perf 级，强化第 14 条）。
- `#49121` [Perf] glm: Bound Triton variants on decode replay（GLM decode replay 限定 Triton 变体，perf 级）。
- `#47013` 见 3.1（headliner）；`#46363`/`#45351` [Perf] hicache: PP-sync channel 深度前瞻 / buffer load-back 所有权检查不等待 GPU（hicache perf 级，强化第 8 条）。
- `#50496` [Fix] constrained: Print grammar objects without a memory address（constrained grammar 打印去内存地址，fix 级）。
- `#38362` [Fix] hicache: Include logical KV dtype in storage keys（HiCache 存储键含逻辑 KV dtype，fix 级，强化第 8/13 条 KV dtype 正确性）。
- `#50400` [Perf] sampling: Re-derive the top-k flags when a running batch drops rows（sampling 运行 batch 掉行时重导 top-k flags，perf 级）。
- `#49970`/`#50407` [Feat] rust-server: Accept client request IDs up to 256 bytes / Pass video item frame and token caps to multimodal processors（rust-server request ID 256B / 视频帧与 token 上限传多模态 processor，feature 级，强化第 17 条）。
- `#49630`/`#49634` [Fix/Chore] rust-server: Log the drain's scheduler kill at info and bound the reap / Refresh frontend config catalogue and drop its Rust lexer（rust-server 日志/配置目录，fix/chore 级）。
- `#49721`/`#49644`/`#49641` [Fix] scheduler/radix-tree: health probe 等待处理 / hicache buffer load-back / SWA tombstone（见 3.2/3.3，fix 级）。
- `#50223` [Fix] comfyui: Stop pinning the H3 source SHA for the ComfyUI alignment patch（comfyui 去固定 H3 SHA，fix 级）。
- `#50426`/`#50466`/`#50510`/`#50502`/`#44613` [CI] diffusion/all-reduce/MCG TP2 / Mooncake EP JIT nvcc flake / sgl-router 构建 / NIXL storage temp dir（CI 级）。
- `#50203`/`#50036`/`#34353` [Refactor/NPU] dsv4 消费注册 sgl-kernel-npu Transformer API / KDA env alias FutureWarning / modelslime NEXTN 启动与 mamba scatter UB overflow（NPU 级，fix 级；#34353 为 mamba scatter UB 溢出，强化第 9 条 int/UB）。
- `#46411`/`#46409` [Refactor] multimodal: [MM refactor 5/9][3/9] token-space processor / 共享 processor mixin（MM 重构级）。
- `#49234`/`#50390` [Perf] diffusion/cosmos: optimize bf16 fa / 查表 SiLU（diffusion bf16 FA 优化，perf 级）。
- `#50397` [Chore] deps: Bump sglang-kernel to 0.4.11 and sgl-fa3 to 0.1.0（依赖 bump，chore 级）。

---

## 4. TensorRT-LLM：2026-11-14 提交（12 条）—— KV offload SSD 三项优化(#21822) + DeepSeek-V4.1 NVFP4 perf 移植(#21438) + paired draft KV pool(#19303) + MTP quant exclusions 静默(#21840) + NVLinkOneSided sentinel(#21668) + CUTLASS MoE FP8 block-scale(#21178)

### 4.1 `#21822` [feat] KVCacheManagerV2: 3 SSD offloading optimizations（KV offload SSD 三项优化，feature/perf，强化第 8 条）
- **主题**：KVCacheManagerV2 SSD 卸载的三项 opt-in 优化，均默认关、经 `KvCacheConfig` 设置；不改行为除非启用。
- **内容详情**：① `disk_io_threads`（并行 SSD I/O）；② `disk_keep_copy_after_restore`（restore 后保留 SSD 副本）；③ `disk_write_through`（write-through 到 SSD，需①）。**Perf 表（某配置，req/s / TTFT p50(s) / TTFT p90(s)）**：Baseline(off) 18.7 / 3.43 / 42.6；+并行 I/O 提升吞吐；write-through 改善 TTFT p90。具体行数值 PR 表内给出（逐项开启）。
- **应用场景**：TRT-LLM KV cache SSD 卸载的吞吐/TTFT 调优。
- **研判**：【事实】PR 给三项优化与 baseline 18.7 req/s / 3.43s / 42.6s 的对照表。【观点】属「KV 卸载可调优」（第 8 条），从正确性走向吞吐/TTFT 量化；记为官方一手（feature/perf 级，含对照表）。**证据等级**官方一手（feature/perf 级，含对照）。
- **来源**：NVIDIA/TensorRT-LLM #21822（合并于 2026-11-14T18:26:40Z）。

### 4.2 `#21438` [TRTLLM-16622][feat] DeepSeek-V4.1: Perf optimizations for DS-V4.1-Flash-NVFP4（DeepSeek-V4.1 NVFP4 性能优化移植，feature/perf，强化第 11、12 条）
- **主题**：把 DeepSeek-V4.1 性能优化移植到 main。
- **内容详情**：PyTorch backend 的 DS-V4.1：compressed KV、mHC、Engram、DSpark draft；CSA2 稀疏 attention（decode/prefill kernel、indexer、exact Top-K）；bounded encoder/decoder replay + SWA scratch 复用；MegaMoE（one-round push router、融合 shared expert、无 PDL eager launch、warmup expert placement）；GB300(SM103) 与 SM107 的 disaggregated serving 调优（executor/KV-cache/test 改动）。共享代码对其他模型保持 main 行为。
- **应用场景**：DeepSeek-V4.1-Flash-NVFP4 在 TRT-LLM 的性能（MegaMoE/稀疏 attention/解聚）。
- **研判**：【观点】属「DS-V4.1 NVFP4 性能/MegaMoE」（第 11、12 条），移植主性能栈；记为官方一手（feature/perf 级，description 级：未给单模型提速数字）。**证据等级**官方一手（feature/perf 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21438（合并于 2026-11-14T16:55:06Z）。

### 4.3 `#19303` [None][fix] Let a paired draft KV pool publish its blocks（paired draft KV pool 发布块，fix，强化第 14 条）
- **主题**：让 draft KV pool 通过 base manager 发布可复用块，加 draft-pool-specific reuse backoff 与 lookahead 处理；scheduler 把 draft match depth 传 `note_context_fork`；Mamba 用合格 fork hint 选分支快照；DeepSeek V4 与 Qwen4-Exp 关 draft reuse-match backoff。
- **内容详情**：CI 结果 mixed、无 test outcome established；QA 标注 `test_joint_reuse_pairing.py` 无 integration-list 入口，**Coverage verdict: needs follow-up**。`tensorrt_llm/_torch/.../cache_manager.py` 关 draft reuse-match backoff；`kv_cache_manager_v2.py` 加 draft-specific backoff 与 advisory-lookahead + `note_context_fork` hook。
- **应用场景**：TRT-LLM paired draft KV pool 的复用/配对（混合模型投机解码）。
- **研判**：【观点】属「混合模型投机解码 paired draft KV」（第 14 条），但 CI mixed、needs follow-up；记为官方一手（fix 级，needs follow-up）。**证据等级**官方一手（fix 级，needs follow-up）。
- **来源**：NVIDIA/TensorRT-LLM #19303（合并于 2026-11-14T17:36:08Z）。

### 4.4 `#21840` [https://nvbugs/7050010][fix] Preserve MTP quantization exclusions（保留 MTP 量化排除，fix，强化第 4 条）
- **主题**：`Qwen35ConfigCompat._normalize_exclude_modules` 在 `_normalize_qwen35_exclude_modules` 翻译前丢弃了 `mtp.*` 条目（HuggingFace 量化排除），致 MTP 量化排除丢失。
- **内容详情**：保留这些条目供后续 MTP 名翻译；把 dense 主层与 MTP MLP projection 排除归一化到 runtime adapter 的 `.mlp.mlp.` 路径（已归一 dense 路径不变）；top-level MTP projection 与 mixer 名与 checkpoint weight mapper 一致翻译；vision filtering 与 split-projection 归一化不变；无公开 API/服务配置字段变更。Fixes nvbugs/7050010。
- **应用场景**：Qwen3.5 MTP 层的量化排除正确性（避免 MTP 被误量化）。
- **研判**：【观点】属「量化排除/MTP 静默错误」（第 4 条），MTP 排除丢失致误量化风险；记为官方一手（fix 级，含 nvbugs 链接）。**证据等级**官方一手（fix 级，nvbugs 内部链接）。
- **来源**：NVIDIA/TensorRT-LLM #21840（合并于 2026-11-14T12:18:17Z）。

### 4.5 `#21178` [None][perf] Speed up CUTLASS MoE FP8 block-scale row expansion and activation launch（CUTLASS MoE FP8 block-scale 行展开与激活 launch 加速，perf，强化第 12 条）
- **主题**：CUTLASS MoE 行展开与激活 launch 改动（从 #20878 拆出；FlashMLA `-rdc` build 移到 #22028；Python 在 #21179、Triton FP8-activation MoE 在 #21250；本 PR 不依赖二者）。
- **内容详情**：**FP8 block-scale 行展开**——`expandInputRowsFp8BlockScaleKernel` 原对每个 permuted 行二分搜 expert offsets；现每块取连续 permuted 行、搜一次首行、线性推进（行按 expert 排序）。**激活 launch（所有 CUTLASS MoE 量化模式）**——`doActivation` 原总以 256-thread block 8 通道 launch；优化为按形状选更合适 block。
- **应用场景**：TRT-LLM CUTLASS MoE（FP8 block-scale）的行展开与激活 launch 性能。
- **研判**：【观点】属「CUTLASS MoE 激活/展开融合」（第 12 条），去二分搜冗余；记为官方一手（perf 级，description 级：未给具体提速数字）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21178（合并于 2026-11-14T08:09:03Z）。

### 4.6 `#21668` [None][feat] Add sentinel transport to NVLinkOneSided All2All; Retire CFT counted-write（NVLinkOneSided sentinel transport，feature，强化第 8 条 PD 传输）
- **主题**：给 `NVLinkOneSided` 的小 per-rank token 数加 **sentinel** transport（Lamport-style data-as-flag）；退 CFT counted-write。
- **内容详情**：来自 flashinfer-ai/flashinfer#7498（ported from 595191c，C++/kernels 与 merged 1f30630 一致）；receiver 轮询预置 sentinel word 的 landing 行代替等完成 flag/fence。命名与选择跟随 flashinfer-ai/flashinfer#8770（open）。本地适配 vs FlashInfer：arch gate 扩到 SM107；携带 **bf16 routing weights**（load 为 ex...）。用于 All2All 小 token 数场景（PD/专家传输）。
- **应用场景**：TRT-LLM NVLink 单边 All2All 的小 token 数传输（PD/专家分发）。
- **研判**：【观点】属「PD/All2All 传输优化」（第 8 条），sentinel 替代 counted-write 降开销；记为官方一手（feature 级，description 级：bf16 routing weights 细节截断）。**证据等级**官方一手（feature 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21668（合并于 2026-11-14T01:14:26Z）。

### 4.7 `#20918` [https://nvbugs/6943029][fix] Size Whisper's cross-KV pool from its encoder length, not max_input_len（Whisper cross-KV pool 按 encoder 长度定尺寸，fix，强化第 8 条多模态）
- **主题**：cross-KV pool 长度被 `llm_args.max_input_len`(默认 1024) 覆盖、再与未识别 Whisper `max_source_positions` 的 encoder limit 取 min，使 Whisper cross pool(1024) 永远装不下每请求发出的 1500 encoder positions。
- **内容详情**：改从 input processor 发布的 encoder 输出长度（`encoder_output_len`，每请求 stamped）定尺寸、启动时拒低于它的 `max_seq_len`；token encoder（T5/BART）保留 max_input_len 定尺寸、共享 encoder-limit lookup 不变；加 whisper-large-v3 CPU 单测。
- **应用场景**：Whisper（TRT-LLM）cross-KV pool 尺寸正确性（多模态语音）。
- **研判**：【事实】PR 给「Whisper cross pool 1024 装不下 1500 encoder positions」根因与按 encoder 长度定尺寸修复。【观点】属「多模态 cross-KV 尺寸」（第 8 条），静默截断风险；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：NVIDIA/TensorRT-LLM #20918（合并于 2026-11-14T01:35:39Z）。

### 4.8 `#21203` [https://nvbugs/6982378][fix] Build the FlashInfer FMHA module on all PP ranks before warmup（FlashInfer FMHA 在所有 PP rank 构建，fix，强化第 5、8 条）
- **主题**：`test_disaggregated_overlap_gen_first[ctx_pp4]` 在 8×B200、JIT cache 冷态下失败：context worker（PP4）启动 300s 内未就绪（`TimeoutError`）。
- **内容详情**：SM100 上 context worker 的 attention 层 dispatch 到 FlashInfer 的 TRTLLM-Gen FMHA 库，FlashInfer 首次使用 build JIT module（下 kernel metadata header + nvcc 编译）；多进程 MPI worker 默认各 rank 独立 FlashInfer workspace。修复在 warmup 前在所有 PP rank 构建该 module。
- **应用场景**：TRT-LLM PD（PP4 context）warmup 前 FlashInfer FMHA JIT 就绪（解聚稳定性）。
- **研判**：【事实】PR 给「ctx_pp4 冷态 300s 超时 + FlashInfer JIT 首次 build」根因与所有 PP rank 预构建修复。【观点】属「PD warmup/JIT 稳定性」（第 5、8 条）；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：NVIDIA/TensorRT-LLM #21203（合并于 2026-11-14T01:34:21Z）。

### 4.9 `#22008` [None][perf] Run the draft layer after attention only on drafting rows（draft 层仅在 drafting 行跑，perf，强化第 14 条）
- **主题**：仅对 drafting 行在 attention 之后跑 draft 层（投机解码）。（PR 正文截断，description 级。）
- **应用场景**：TRT-LLM 投机解码 draft 层的行级执行优化。
- **研判**：【观点】属「投机解码 draft 层执行优化」（第 14 条），description 级。记为官方一手（perf 级，描述级）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #22008（合并于 2026-11-14T00:09:26Z）。

### 4.10 `#19550` [None][feat] Add native Qwen3 reranking support（原生 Qwen3 reranking 支持，feature）
- **主题**：加原生 Qwen3 reranking 支持。（PR 正文截断，description 级。）
- **应用场景**：TRT-LLM 文本重排（reranking）模型支持。
- **研判**：【观点】属「模型支持扩展」，feature 级。记为官方一手（feature 级，描述级）。**证据等级**官方一手（feature 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #19550（合并于 2026-11-14 窗口内）。

### 4.11 其他要点（标题级 / 描述级）
- `#217..` [test/infra] Mark pyexecutor/attention/disaggregation unit tests cpu_only（batch #2/#10/#13/#14）与 most recent lock file from nightly pipeline（test/infra 级）。
- 注：TRT-LLM 本窗口仅 12 条（周六低活跃），其余多为 test/infra/lockfile，未逐条展开；完整标题见 commits 解析（见第 8 节数据质量）。

---

## 5. llama.cpp：2026-11-14 提交（32 条）—— 量化静默延续(qwen2vl FFN scales #32241) + Hexagon FA no-mask(#32243) + ctz64 32bit(#32273) + DFlash 投机重构(#32016) + Hexagon IQ2_XXS/NVFP4(#31954) + CUDA/Vulkan perf(#28982/#31323/#31990) + Inkling 2(#32217) + M-RoPE 警告(#32287)

### 5.1 `#32241` qwen2vl : apply FFN weight scales（qwen2vl 应用 FFN 权重量表，fix，强化第 4、6 条，静默错误铁证）
- **主题**：Fixes #32101。`qwen2vl` 图把三个 FFN scale 参数（`ffn_up_s`/`ffn_gate_s`/`ffn_down_s`）传 `NULL` 给 `build_ffn`，致通用 loader 已加载的 per-tensor NVFP4 scales **从未应用**；`nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` 产出垃圾且无警告（与 #31657 起 qwen2 已正确传这些 scale 对照）。
- **内容详情**：本 PR 传 `ffn_up_s`/`ffn_gate_s`/`ffn_down_s`（无则 NULL，其他 qwen2vl 模型不受影响）。**验证**（GB10，HF 转 bf16，`-c 512 -b 512 -ub 512 --chunks 5 -fit off`）：CUDA `-ngl 99 -fa on` master PPL 1,043 → PR（具体对照值 PR 表内，wikitext-2）。
- **应用场景**：Qwen2-VL NVFP4 的 FFN 量化 scale 正确性。
- **研判**：【事实】PR 给「FFN scales 传 NULL 致 NVFP4 静默垃圾、无警告，Fixes #32101」与 GB10 wikitext-2 PPL 对照。【观点】属「量化 scale 漏接静默错误」（第 4、6 条），第四家引擎再现，与 SGLang #42566/#49338、llama.cpp 自身 #31449 等共构主线；记为官方一手（fix 级，含对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：ggml-org/llama.cpp #32241（合并于 2026-11-14T18:02:16Z）。

### 5.2 `#32243` hexagon: fix fa no-mask tail（Hexagon FA no-mask tail 修复，fix，强化第 4 条跨硬件静默错误）
- **主题**：发现某些 image patch count（mtmd encoder 无 mask）下导致错误 softmax 结果：如 kv 484 在 `[L1407]` 留 28 个 padding 为 0（`c + 64 < kv_rows`），使尾部未正确 pad 到 −inf。
- **内容详情**：修复确保 tail 正确 pad 到 −inf。测试 LFM2-VL-450M 与 Qwen3-VL-2B 于 IQ-9075（`GGML_HEXAGON_HOSTBUF=0`），kv 484/676/1444 上 NMSE 修复前更高。
- **应用场景**：llama.cpp Hexagon（骁龙 HTP）后端 FA no-mask 尾部的 softmax 正确性。
- **研判**：【事实】PR 给「kv 484 留 28 padding=0 → 错误 softmax」与 LFM2-VL/Qwen3-VL NMSE 修复。【观点】属「Hexagon 注意力静默错误」（第 4 条跨硬件），尾部未 −inf 致错误输出；记为官方一手（fix 级，含测试）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：ggml-org/llama.cpp #32243（合并于 2026-11-14T18:06:01Z）。

### 5.3 `#32273` Fix ctz64 on 32bit（ctz64 在 32bit 构建修复，fix，强化第 9 条 int/构建正确性）
- **主题**：`ctz64()` 在 MSVC 无条件用 `_BitScanForward64`，32 位目标无此函数，致 x86 构建在 `llama-kv-cells.h` 失败。
- **内容详情**：先用 `_BitScanForward` 扫低 32 位、低半为 0 时再扫高 32 位并 +32，保留现有 64 位实现；加全 64 位位置回归覆盖（单/多 bit）。发现于 microsoft/vcpkg#55068。AI（Codex）辅助。
- **应用场景**：llama.cpp 32 位（x86）构建的 `ctz64` 正确性（KV cells）。
- **研判**：【事实】PR 给「32 位 MSVC `_BitScanForward64` 缺 → 构建失败」与修复 + 全位置回归。【观点】属「构建/整数处理正确性」（第 9 条），32 位平台构建阻断修复；记为官方一手（fix 级，含回归）。**证据等级**官方一手（fix 级，含回归）。
- **来源**：ggml-org/llama.cpp #32273（合并于 2026-11-14T18:06:53Z）。

### 5.4 `#32016` spec + model: refactor DFlash based speculative decoding methods（DFlash 投机解码方法重构，refactor，强化第 14 条）
- **主题**：DFlash 族投机（`dflash`/`dflash2`/`dspark`/`lilicorr`）共用同一 DFlash 骨架但用不同 head 与 candidate selector，此前都实现在 `dflash.cpp` 同一 `--spec-type draft-dflash` 下，日渐难维护。
- **内容详情**：拆为独立投机方法：`draft-dflash`/`draft-dflash2`/`draft-dspark`/`draft-lilicorr`；`src/models/dflash.cpp` 拆为 `dflash.cpp`/`dflash2.cpp`/`dspark.cpp`/`lilicorr.cpp`，每方法一图类，隔离方法特定逻辑。（与本期 vLLM/TRT 投机解码硬化、上期 #32016 类重构呼应；社区侧 llama.cpp v1.0 投机解码基准见第 7 节。）
- **应用场景**：llama.cpp DFlash 族投机解码的代码组织与可维护性。
- **研判**：【观点】属「投机解码代码重构」（第 14 条），纯组织改进；记为官方一手（refactor 级）。**证据等级**官方一手（refactor 级）。
- **来源**：ggml-org/llama.cpp #32016（合并于 2026-11-14T08:49:28Z）。

### 5.5 `#31954` Hexagon: add support for IQ2_XXS and NVFP4（Hexagon IQ2_XXS + NVFP4 支持，feature，强化第 13、15 条跨硬件量化）
- **主题**：给 Hexagon backend 加 IQ2_XXS 与 NVFP4 权：tiled repack layouts（32-row × 32-k tiles）、HVX GEMV kernels（n≤4）与 HMX paths（n>4）、NVFP4 per-tensor scale、`supports_op` 覆盖；修 HMX IQ2_XXS dequant 的 fp16 scale underflow。
- **内容详情**：IQ2_XXS decode——256×8 B grid 过大放不进 register `vlut`，用 640 B compact LUT in VTCM 替代（16-entry nibble table + 256 halfword codes）；每 32×32 tile 两 halfword gathers，`vlut16` 扩码到 int8，`vpopcount` 给符号奇偶；gathers 在 HVX GEMV 与 HMX path 都 pipeline 一 tile ahead。
- **应用场景**：llama.cpp Hexagon（骁龙 HTP）后端的 IQ2_XXS 与 NVFP4 量化算子（端侧低比特）。
- **研判**：【观点】属「Hexagon 低比特量化」（第 13、15 条），扩展端侧 NVFP4/IQ2_XXS 覆盖；记为官方一手（feature 级，description 级）。**证据等级**官方一手（feature 级，description 级）。
- **来源**：ggml-org/llama.cpp #31954（合并于 2026-11-14T04:57:06Z）。

### 5.6 `#28982` cuda : keep columns in registers for cooperative softmax（CUDA cooperative softmax 列留寄存器，perf，强化第 15 条）
- **主题**：解 `ggml/src/ggml-cuda/softmax.cu` 第 145 行 TODO：对 `ncols > 8192` 的每行用 cooperative groups 跨所有 SM 并行化。`soft_max_f32_parallelize_cols_single_row` 原硬编码 `n_elem_per_thread=4` 跨 pass 丢弃加载值、需每行 5 次全局内存访问（3 读+2 写）。
- **内容详情**：把两 kernel template 在 `int n_elem_per_thread=0`：当列能放寄存器（`ncols ≤ grid_size × n_elem_per_thread`）时减少全局内存往返。
- **应用场景**：llama.cpp CUDA 后端大 ncols cooperative softmax 性能。
- **研判**：【观点】属「CUDA softmax 寄存器优化」（第 15 条），降全局内存访问；记为官方一手（perf 级，description 级：未给具体数字）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：ggml-org/llama.cpp #28982（合并于 2026-11-14T22:36:35Z）。

### 5.7 `#31323` vulkan: optimize prompt processing Flash Attention for RDNA3 and RDNA4（Vulkan RDNA3/RDNA4 prompt FA 优化，perf，强化第 15 条）
- **主题**：RDNA3/RDNA4（Strix Halo、R9700 测）coopmat1 Flash Attention prompt 处理路径优化集。
- **内容详情**：**现有 cm1 FA shader**——Strix Halo 用 32-row tiles 代替 16（降全局内存、更多 cache、需 shmem 复用）；16-row tiles 把 Q cache 寄存器；RDNA4 启 occupancy limiter 补偿过高 occupancy。**新 flipped FA shader**——每 subgroup 处理 16 Q-rows 跨 head 维、输出留 coopmat 累加器；K/V staged 进 shmem 或直读；处理 head size ≤256，Strix Halo 上提升显著。
- **应用场景**：llama.cpp Vulkan 后端在 RDNA3/RDNA4 的 prompt 处理 FA 性能（AMD APU/GPU）。
- **研判**：【观点】属「Vulkan AMD FA 优化」（第 15 条），新 flipped shader + tile 调整；记为官方一手（perf 级，description 级：未给具体数字）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：ggml-org/llama.cpp #31323（合并于 2026-11-14T22:36:59Z）。

### 5.8 `#31990` vulkan: optimize matrix multiplication for narrow N（Vulkan 窄 N 矩阵乘优化，perf，强化第 15 条）
- **主题**：给 coopmat1/coopmat2 加 N≤32 的 "narrow" tile configs。
- **内容详情**：作者尝试接近 #31533 思路未得更好 perf，最终更小改动获部分（非全部）收益；默认在 NV 上工作良好，开放 coopmat1 tilesize/cutoff 调整以适配其他硬件。
- **应用场景**：llama.cpp Vulkan 后端窄 N 矩阵乘（小 N 场景）性能。
- **研判**：【观点】属「Vulkan 窄 N MM 优化」（第 15 条），部分收益；记为官方一手（perf 级，description 级：未给具体数字）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：ggml-org/llama.cpp #31990（合并于 2026-11-14T22:36:06Z）。

### 5.9 `#32217` feat: add Inkling 2 support (text + vision)（Inkling 2 支持，feature，强化第 11 条模型）
- **主题**：加纯 Inkling 2 支持（conversion + model），**不含**将作为独立 PR 叠在其上的 DSpark speculator。
- **内容详情**：无需算子变更。AI 辅助声明。与上期 SGLang/TRT 的 Inkling/Inkling-MXFP8 KV（#50324）形成模型侧呼应。
- **应用场景**：llama.cpp Inkling 2（文本+视觉）模型推理。
- **研判**：【观点】属「新模型支持」（第 11 条），纯转换+模型无算子变更；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：ggml-org/llama.cpp #32217（合并于 2026-11-14T23:51:32Z）。

### 5.10 `#31438` llama : allocate the tensor staging buffer from the CPU buffer type（tensor staging buffer 从 CPU buffer 类型分配，fix/perf，强化第 15 条内存）
- **主题**：macOS 27.0.1 上，`--load-mode none` 加载用 loader 的 staging buffer 留在进程 footprint：loader 加载末释放，但 libmalloc 保留该大块于 footprint，使 footprint 在模型加载时与释放后都高。
- **内容详情**：无 mmap 时 loader 以三种方式读 tensor 进 buffer；本 PR 让 staging buffer 从 CPU buffer 类型分配，释放后 footprint 回落。
- **应用场景**：llama.cpp 模型加载的内存 footprint（尤其 macOS）。
- **研判**：【观点】属「加载内存 footprint」（第 15 条），降进程常驻；记为官方一手（fix/perf 级）。**证据等级**官方一手（fix 级）。
- **来源**：ggml-org/llama.cpp #32241 窗口外；#31438 合并于 2026-11-14T22:37:47Z。
- **来源**：ggml-org/llama.cpp #31438（合并于 2026-11-14T22:37:47Z）。

### 5.11 `#32287` memory : fix non-consecutive position warning for M-RoPE（M-RoPE 非连续位置警告修复，fix）
- **主题**：Qwen3.5 9B 含图文 prompt 时大量 `find_slot: non-consecutive token position` 警告——M-RoPE 位置是向量非标量，用图像时 temporal 位置不总每 token +1。本 PR 对 M-RoPE 模型禁用该警告。
- **内容详情**：纯警告抑制（M-RoPE 下非连续位置为正常），无功能变更。
- **应用场景**：llama.cpp M-RoPE 模型（Qwen3.5 等）的 slot 警告噪声抑制。
- **研判**：【观点】属「M-RoPE 位置处理」，警告误报修复；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：ggml-org/llama.cpp #32287（合并于 2026-11-14T22:57:24Z）。

### 5.12 `#32228` hexagon: add TIMESTEP_EMBEDDING HVX kernel（Hexagon TIMESTEP_EMBEDDING HVX kernel，feature，强化第 15 条）
- **主题**：给 GGML Hexagon backend 加 HVX `TIMESTEP_EMBEDDING` kernel。（扩散/时间步嵌入算子，端侧。）AI 辅助 drafting/testing、人工 review。
- **应用场景**：llama.cpp Hexagon 后端的 TIMESTEP_EMBEDDING 算子。
- **研判**：【观点】属「Hexagon 算子扩展」（第 15 条），feature 级。记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：ggml-org/llama.cpp #32228（合并于 2026-11-14T23:03:32Z）。

### 5.13 其他要点（标题级 / 描述级）
- `#32203` llamafile: introduce more Q0 quant support for s390x（llamafile Q0 量化 s390x 支持，feature 级）。
- `#32251` opencl: fixes for q4_k mv oob access, f16 swiglu_oai, cpy with strided dst, batched group_norm and bf16 sigmoid（OpenCL 多项修复，fix 级，强化第 15 条）。
- `#32237` convert : register Granite4VisionForConditionalGeneration for text（Granite4Vision 转换注册，feature 级）。
- `#31775` ggml-backend : start a new split at an incompatible weight in a split without inputs（ggml-backend split 无输入时不兼容权重起新 split 修复，fix 级；Fixes #31675，hybrid `-nkvo` 下 CPU/CUDA split 边界）。
- `#32156` convert : pplx-decider must match qwen3.5 model_type（pplx-decider 匹配 qwen3.5 model_type，fix 级）。
- `#32267`/`#32188` mtmd: fix video pixel aspect ratio and video rotation / allow custom video decoder（mtmd 视频像素比/旋转修复、自定义视频解码器，fix 级）。
- `#32023` jinja : reduce template rendering overhead（jinja 模板渲染开销降低，perf 级）。
- `#30403` mtmd : avoid padding-only Whisper audio chunks（mtmd 避免仅 padding Whisper 音频块，fix 级）。
- `#30492` server : Fix connection reuse after streamed responses（server 流式响应后连接复用修复，fix 级，强化部署稳定）。
- `#31478` common : reject out-of-range grammar_triggers type instead of aborting（grammar_triggers 越界类型拒绝而非中止，fix 级）。
- `#32164`/`#32271`/`#32232` chat : allow undeclared tool calls in LFM2 tool-call grammar / allow tool calls inside reasoning blocks for Ministral-3 / allow Gemma4 tool call during thought（chat tool-call grammar 放宽，feature 级）。
- `#32282` common : fix infinite recursion in debug build for peg parser dump（peg parser dump 调试构建无限递归修复，fix 级）。
- `#32230` chat : remove mistral workaround（chat 去 mistral 临时方案，cleanup 级）。
- `#32024` vulkan: use GPU copies on UMA with uncached memory on Intel（Vulkan UMA Intel 未缓存内存 GPU 拷贝，perf 级，强化第 15 条）。
- `#32109` sycl : optimize mtmd image processing（SYCL mtmd 图像处理优化，perf 级，强化第 15 条）。
- `#31990` 见 5.8；`#31323` 见 5.7；`#28982` 见 5.6。

---

## 6. 累计结论延续与新增（继承 1–17，本期新增 18）

**累计结论（1–17 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性（发版当天合入 ≠ 在版内）；③ 默认值成为主要风险载体（本期 #66401 MRV2 对齐放宽、#49896 breaking 安全默认）；④ 异构/量化后端静默错误为高频（本期 SGLang #47013 / llama.cpp #32241 / TRT #21840 / SGLang #49805 共构）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务；⑥ 量化路径静默错误（scale/布局漏接，本期 #47013 padding slot 0、#32241 FFN scales 漏接、#21840 MTP exclusions）；⑦ 专家张量/路由归属与分页漏接静默错误（本期 SGLang #49805 OLMoE/Hunyuan/Llama4 漏传 top_k）；⑧ KV 卸载/HiCache/PD 健壮性 + 内存/缓存管理接口重构（本期 SGLang #49992/#49641/#50370/#50324/#50544、TRT #21822/#21668/#20918/#21203、vLLM #65336/#67599/#67577）；⑨ 长上下文 int32/int64 溢出静默错误（本期 vLLM #64515 XPU int64、llama.cpp #32273 32bit ctz64 构建、SGLang #34353 mamba scatter UB）；⑩ MXFP8 KV 正确性簇；⑪ 新旗舰模型 CP/KV/性能成为一等公民（本期 vLLM #63483 DBO 纯 TP、#64827 DeepSeek JIT 预热、#67600 Kimi-K3.1-Pro recipe、SGLang #49905 GDN MIS 融合、llama.cpp #32217 Inkling 2）；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径（本期 vLLM #66799 路由 padding 屏蔽 +#67591 modular TRT-LLM FP8 EPLB、TRT #21438 DS-V4.1 NVFP4 perf、#21178 CUTLASS MoE FP8 block-scale）；⑬ FP8/低比特 KV cache 跨引擎标准化落地（本期 SGLang #50324 MXFP8 KV external linker、#42978 NPU DSA、vLLM #64753 NVFP4 scales 留 GPU）；⑭ 投机解码验证路径跨引擎密集硬化（本期 vLLM #67291、TRT #19303/#22008、SGLang #46781、llama.cpp #32016 重构）；⑮ 跨后端可移植性成为显式架构主线（Apple MPS/NPU/Hexagon/SpacemiT/ROCm/OOT/SYCL/Vulkan/OpenCL；本期 llama.cpp #31954/#32243/#32228/#28982/#31323/#31990、vLLM #64255 XPU、SGLang #42978 NPU）；⑯ 新旗舰模型进入「默认性能化」阶段 + 量化/MegaMoE 通用化；⑰ 推理服务前端 Rust 化成为跨引擎并行主线（本期 SGLang #50401/#49970/#50407、vLLM #63580/#67282）。

**新增第 18 条（推理服务默认安全加固成为跨引擎主线）**：本期 SGLang #49896/#49895 把敏感 admin 端点改为需 `--admin-api-key` 否则 403（breaking）；这与 llama.cpp v1.0.0 的 `llama-server` 加固（SSRF/CORS/DNS rebinding/MCP 工具取消，上期头号事件）及 vLLM 既有 admin 鉴权形成跨引擎「推理服务默认安全」共振。结论：推理引擎的竞争正从「核心算子/调度」外溢到「服务/前端层」的两条并行主线——其一是第 17 条的 Rust 化（降服务层延迟/并发），其二是本条第 18 条的默认安全加固（降部署面攻击面）；部署侧须显式配置 admin 密钥与 server 安全项，否则管控端点被拒或服务暴露内网风险。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 量化静默错误复核（第四家引擎再现）**：SGLang `#47013`（GLM-5.3-Flash FP8 KV + TileLang DSA → 半数乱码、GSM8K 50→98%，padding slot 0 静默写）、`#49805`（OLMoE/Hunyuan/Llama4 漏传 top_k → fused MoE 静默 None）；llama.cpp `#32241`（qwen2vl FFN scales 传 NULL → NVFP4 静默垃圾无警告）、`#32243`（Hexagon FA no-mask tail 未 −inf）；TRT `#21840`（MTP quant exclusions 丢失）；vLLM `#64753`（NVFP4 scales CSF 压缩，已显式处理）。部署任何 FP8/NVFP4/MXFP8 模型或 KV writer/padding/scale 相关路径前，必须逐项核验 scale/布局/槽位/padding 是否真的接入，尤其并发与 DSA/TileLang 后端组合。
- **P0 — KV 页所有权 / 释放正确性**：SGLang `#49641`（Rust tree core SWA tombstone 恢复误释放他请求页）、`#49992`（joint Full/SWA eviction 在 Rust 树核心落地）；vLLM `#67599`（routed-experts row cache 两源尺寸分歧）；TRT `#20918`（Whisper cross-KV 尺寸截断）。KV/radix 页释放与尺寸一致性是高优先级正确性项。
- **P1 — MoE 路由 / MegaMoE / EP（第 12 条）**：vLLM `#66799`（grouped top-k 路由 DP padding 默认屏蔽，GLM-5.3 +5~21%、MiMo-V2.6）、`#67591`（modular TRT-LLM FP8 block-scale MoE 允许 EPLB）；TRT `#21178`（CUTLASS MoE FP8 block-scale 行展开/激活 launch）、`#21438`（DS-V4.1 NVFP4 perf 移植）。
- **P1 — KV offload / HiCache / 跨引擎缓存管理（第 8 条）**：TRT `#21822`（SSD 卸载三项优化，量化吞吐/TTFT）；SGLang `#50370`（out-of-tree HiCache 存储后端加入逻辑页分片）、`#50324`（MXFP8 KV external linker）、`#50544`（DP 多 rank router_l3_layout 透传）、`#38362`（HiCache 存储键含 KV dtype）；vLLM `#65336`（NIXL 从 PP prefill 拉 KV）、`#67577`（ROCm extensible KV cache for fd-exporting）；vLLM `#64827`（DeepSeek DeepGEMM JIT 预热防 EngineDeadError）。
- **P1 — 跨后端/跨硬件（第 15 条）**：llama.cpp `#28982`（CUDA cooperative softmax 寄存器）、`#31323`/`#31990`（Vulkan RDNA3/4 FA/窄 N MM）、`#31954`（Hexagon IQ2_XXS/NVFP4）、`#32243`（Hexagon FA）、`#32228`（Hexagon TIMESTEP_EMBEDDING）、`#32024`（Vulkan UMA Intel）；vLLM `#64515`（XPU int64 token index）；SGLang `#42978`（NPU DSA token shard）、`#34353`（NPU mamba scatter UB）；TRT `#21668`（NVLinkOneSided sentinel）。
- **P1 — 推理服务安全 / Rust 前端（第 17、18 条）**：SGLang `#49896`/`#49895`（admin 端点需 `--admin-api-key` breaking，部署侧须显式配置）；SGLang `#50401`（rust-server HTTP/2 h2c）、`#49970`/`#50407`（rust-server request ID/video caps）；vLLM `#63580`（Rust frontend 400 而非 500）、`#67282`（SIGTERM 排空在途请求）。
- **P1 — 投机解码（第 14 条）**：vLLM `#67291`（context-K/V-only prefill 不建 draft 元数据）；TRT `#19303`（paired draft KV pool，CI mixed/needs follow-up）、`#22008`（draft 层仅 drafting 行）；SGLang `#46781`（spec MRoPE delta 避免阻塞 H2D）；llama.cpp `#32016`（DFlash 族重构为独立方法）。

---

## 7. 社区与视频资源（HN Algolia，2026-11-14 epoch 窗口 1794614400–1794700799）

> 以下均为社区/Show HN，非引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词（vllm/llama.cpp/sglang/tensorrt）检索中 sglang 仅 2 命中（1 为无关 europoors 帖、1 为 Generative AI 能耗）、tensorrt 仅 1 命中（Tensor Seeks Layout，ML 编译器非引擎）；vllm 命中 290 多为泛科技/AI 资讯（无一条指向引擎实现）。

- **llama.cpp v1.0.0 社区延续讨论（承接上期头号事件）**：窗口内多条「Ggml v1.0.0」帖子（2026-11-14 11:13/13:22/14:46/17:13/17:29Z 等，分数未显式给出）持续讨论 11-13 发布的 v1.0.0；「Llama.cpp v1.0 on an RTX 3090: up to 55% faster with speculative decoding」（**2 分**，2026-11-14T16:30:57Z）——社区对 llama.cpp v1.0 投机解码的实测（RTX 3090 上最高 +55%，与本期 #32016 DFlash 重构及 v1.0 投机 API 直接呼应）。**二手线索/未核验**。
- **Qwen3.8-Flash-Next 社区长文**：「Ode to Qwen 3.8 Flash Next」（2026-11-14T19:43:13Z）——社区对 Qwen3.8-Flash-Next 的评述，与本期 TRT-LLM #21840（MTP quant exclusions）、vLLM #64753（NVFP4 scales CSF）、SGLang #50318（PLE rows host gather）技术线呼应。**二手线索/未核验**。
- **GLM-5.3 vLLM 部署实测**：「Serving GLM-5.3 at 607.8 tok/s across two Nvidia DGX stations with vLLM」（2026-11-14T17:46:01Z）——社区双 DGX 站部署 GLM-5.3 的吞吐实测，与本期 SGLang #47013（GLM-5.3-Flash FP8 KV 静默错误）形成「同模型、不同引擎/后端」的对照视角。**二手线索/未核验**。
- **KV cache 工具/研究**：「SCache and MOS – 32.9k O(1) KV cache (22.7 MiB) and 72B INT4/ASVD engine」（2026-11-14T20:02:03Z）——KV cache 相关的社区引擎/工具（O(1) KV、72B INT4/ASVD），与本期 KV/HiCache/SSD offload 主线（第 8 条）主题相关。**二手线索/未核验**。
- **量化社区**：「Unsloth Qwen3.8-27B Quantizations – Dynamic v3.1 Update」（窗口内多条 11-14 全天）—Qwen3.8-27B 的社区量化更新，与本期量化主题（第 4/13 条）相关。**二手线索/未核验**。
- **Show HN / 本地 LLM 工具**：「Show HN: Loft – offline Q&A over your local files, with citations (llama.cpp)」（**1 分**，2026-11-14T19:42:17Z）——基于 llama.cpp 的本地离线 Q&A 工具，与端侧/本地部署主线相关。**二手线索/未核验**。
- 其余 HN 命中（vllm 290/llama.cpp 27/sglang 2/KV cache 8/speculative decoding 5/Qwen3.8 14）多为泛 AI/科技资讯、Qwen3.8 模型舆情（如「Alibaba's Qwen3.8 re-educated with a more Western worldview」3 分，非技术）或与推理引擎实现无直接关系，未列入。本期**无引擎相关的可确认官方视频**。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **49** / SGLang **60** / TRT-LLM **12** / llama.cpp **32** 合计 **153** 条（2026-11-14 UTC 窗口；2026-11-14 为周六，较前一窗口 248 回落，属周末正常；各 repo 均未触 per_page=100 上限、单页取全，无截断遗漏）。
- **版本核验**：releases API 确认——vLLM 仍为 v0.33.0rc3（2026-11-02）、SGLang v0.5.23（2026-11-02）、TRT-LLM dev 线 v1.4.0rc3（prerelease，2026-11-06）/ 稳定线 v1.2.1，窗口内均无新正式版本；llama.cpp 于 2026-11-14 推进 nightly b12380–b12387（prerelease），v1.0.0 正式版仍停留在 2026-11-13。
- **PR 正文来源**：**47 个关键 PR 正文经官方 pulls API（markdown 官方一手）抓取**；无 GitHub token 下 60/h 配额被用尽（含 4 个初次下载截断文件的重试），未触发二次截断。其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度（如 TRT-LLM `#19303` 标注 CI mixed/needs follow-up、`#21840`/`#20918`/`#21203` 含 nvbugs 内部链接无法核验 QA 状态、vLLM `#63483`/`#64827`/`#67291`/`#67591` 未给具体提速数字记 perf 描述级、SGLang `#42978`/`#49905`/`#50318`/`#50401` 未给数字、llama.cpp `#28982`/`#31323`/`#31990`/`#31954`/`#32228` 未给数字；TRT `#22008`/`#19550` 正文截断记 description 级）。
- **描述级/needs-follow-up 标注**：TRT-LLM `#19303`（CI mixed、Coverage verdict: needs follow-up）、`#21840`/`#20918`/`#21203`（nvbugs 内部链接，实现/QA 状态以链接为准）、`#22008`/`#19550`（正文截断）；vLLM `#63483`/`#64827`/`#67291`/`#67591`（未给具体数字，perf 描述级）；SGLang `#42978`/`#49905`/`#50318`/`#50401`（未给数字，description 级）、`#34353`（NPU mamba scatter UB overflow）；llama.cpp `#28982`/`#31323`/`#31990`/`#31954`/`#32228`（未给数字，perf 描述级）等均已按实际缺失程度标注。
- **社区信号**：HN Algolia 引擎官方更新类故事无；社区侧 llama.cpp v1.0.0 延续讨论（多条 Ggml v1.0.0 帖 + RTX 3090 投机解码 +55% 实测）、Qwen3.8-Flash-Next 长文、GLM-5.3 vLLM 部署实测（607.8 tok/s）、SCache/MOS KV 工具、Unsloth Qwen3.8 量化，均按二手线索/未核验记录；无引擎相关的可确认官方视频。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/needs follow-up」。
- **技能检索（find-skills / agent-browser）**：依本自动化历次一致结论，find-skills 经 SkillHub 语义检索（EN+ZH）返回通用编程/生产力类技能（编程专家.Skill、腾讯文档、各类聚合/方案输出类），与「官方提交采集 + 证据分级归档 + 仓库同步」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/pulls API）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次 11 期一致。
