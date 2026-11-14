# 大模型推理优化每日热点档案

> 档案日期：2026-11-14（本自动化于该日执行）
> 关注窗口：2026-11-13（周五）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 经 pulls API 抓取 markdown 官方一手；全部 44 个详述 PR 均经官方 pulls API 一手获取，未走 HTML 页面抓取）+ HN Algolia（HTTPS，numericFilters 按 2026-11-13 epoch 窗口 1794528000–1794614400）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 102 / SGLang 78 / TRT-LLM 46 / llama.cpp 22 合计 248 条（vLLM 触 per_page=100 上限，page2 追加 2 条，实计 102）；**llama.cpp 于窗口内正式发布 v1.0.0**（2026-11-13T10:20:19Z，正式版）；vLLM/SGLang/TRT-LLM 窗口内无新正式版本；44 个关键 PR 正文经 pulls API（markdown 官方一手）抓取（API 配额充足、未触发 60/h 限制）；HN Algolia 引擎关键词高信号集中于 llama.cpp v1.0.0 社区讨论（20 分），其余为社区/Show HN，二手线索/未核验。本窗口为周五，提交量较上一窗口（11-12：94/89/39/39）回升至 102/78/46/22，属正常工作日波动。

## 1. 版本状态与本期头号事件：llama.cpp v1.0.0 正式发布 + 双引擎 Rust 服务前端同窗口大规模合入 + SGLang mem_cache 统一内存池重构 + 量化静默正确性延续(TRT #22108 / SGLang #42566/#49338) + 投机解码验证硬化 + 跨硬件(AMD silent / Metal overflow)

**提交计数（2026-11-13 UTC，均 HTTP 200；2026-11-13 为周五，量较周四回升；vLLM 102 / SGLang 78 / TRT-LLM 46 / llama.cpp 22）**：
- **vLLM 102**（page1=100 触上限，page2=2，实计 102）/ **SGLang 78** / **TensorRT-LLM 46** / **llama.cpp 22**。

**发布状态（窗口内 llama.cpp 正式发布 v1.0.0，其余无新正式版本）**：
- **llama.cpp**：**v1.0.0 正式发布（2026-11-13T10:20:19Z，正式版，非 prerelease）**——本窗口最大单一事件。移除废弃公共 API（#30565 删 35 个函数）、把 NextN/投机解码 API 提升进 `llama.h`、新增 OCP FP8 KV cache、host-memory MoE expert cache（可配 CPU 份额，#30966）、外部 PLE shard 加载；新增 Kimi-K3.1-Pro / Ripple / xing4_0 三个文本架构、drama-3 TTS、MageVL-Cowork 与 Kimi-K3.1 视觉；加固 `llama-server`（SSRF/CORS/DNS rebinding/MCP 工具取消）、新增 `/v1/audio/speech`、`ggml` 升 v1.0.0。博客：https://huggingface.co/blog/ggml-org/llama-cpp-v1。
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-13 全部 102 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；#50369 仅 bump `sgl-kernel` 到 0.4.11（子模块版本，非发行版）。
- **TensorRT-LLM**：dev 线仍为 **v1.4.0rc3（prerelease，2026-11-03）**；`releases/latest` 口径下最新稳定版为 v1.2.1（2026-04-20）；11-13 窗口无新版本。

**本期头号事件（llama.cpp v1.0.0 + 双引擎 Rust 服务前端 + mem_cache 统一池 + 量化静默 + 投机硬化 + 跨硬件）**：
1. **llama.cpp v1.0.0 正式发布（本窗口最大单一事件，强化/补充部署安全主线与第 15 条跨硬件）**：历经从 0.9.0（11-04）到 1.0.0 的跨越，正式移除废弃公共 API、把 NextN/投机解码 API 提升为正式接口、引入 OCP FP8 KV cache 与 host-memory MoE expert cache（可配 CPU 份额）。最值得推理优化关注的三点：① **OCP FP8 KV cache** 把低比特 KV 标准化推进到端侧/本地引擎（呼应第 13 条跨引擎低比特 KV 标准化）；② **host-memory MoE expert cache（可配 CPU 份额）** 是端侧大模型 MoE 显存墙的系统性解法；③ **`llama-server` 加固（SSRF/CORS/DNS rebinding/MCP 工具取消）** 把上期的部署安全主线（#32115/#32120）从补丁提升为正式能力。HN 社区「Llama.cpp v1 Is Out」（20 分）、「llama.cpp v1.0」（5 分）直接讨论该发布。
2. **vLLM + SGLang 双引擎 Rust 服务前端同窗口大规模合入（新增第 17 条：服务前端 Rust 化跨引擎并行主线）**：vLLM 本窗口出现密集 "Rust Frontend" 簇（#65994 gRPC Render、#67256/#67187/#66222/#66530/#66540/#67359/#67398/#67409/#67383 等约 20 个 PR）；SGLang 同步密集 "rust-server" 簇（#49631/#49633/#49635/#49640/#49650/#49653/#49663/#49907/#50189/#50190/#50199 共 11 个 PR）。两引擎在同一 UTC 窗口并行把服务/前端层迁移到 Rust，是服务架构层面的跨引擎共振，与「推理核心层」优化并列成为新主线。
3. **SGLang mem_cache 统一内存池重构（强化第 8 条 KV/HiCache/PD 健壮性 + 架构主线）**：以 #49913（Kimi-K3.1 跑在 unified memory pool）为顶点的栈（#49719→#49720→#49775→#49912→#49966→#49913）把 KDA/Mamba 状态、DSA indexer keys、FP8 latent rows 统一进复合 sub-pool；#49886 把 hybrid Mamba state checkpoint 收敛到 fork point / request end（默认 level 2）；#49719/#49720 把 GDN/KDA ReplaySSM ring 放进 unified sub-pool。这是 SGLang KV/状态内存布局的系统性收口，与 vLLM #61452（KV cache 分配策略移出 KVCacheSpecs）形成跨引擎「内存/缓存管理接口重构」共振。
4. **量化静默正确性铁证延续（强化第 4、6、13 条）**：TRT-LLM `#22108`（W4A8_MXFP4_FP8 MoE 的 SwiGLU scaling 误传 dequant 单位，输出约 **766× 过小、93.2129% 元素 mismatch**）、SGLang `#42566`（block-FP8 MoE TP shard 切分 checkpoint block 被拒，padding 方案 +20%~+60% 显存/算力）、SGLang `#49338`（WNA16 MoE TP shard 切分 scale group 致半 K 用错 scale、A800 上 `illegal memory access`）、vLLM `#63097`（Kimi-K2.6 q/k/v scales 全量加载，183/183 精确匹配）、vLLM `#66784`（Qwen4Exp/Kimi KDA merged-linear padding 冷启动与 reload 下清零）。scale/布局漏接仍是最高频静默错误源。
5. **投机解码验证硬化延续（强化第 14 条）**：vLLM `#64156`（落 RFC #64099，新增 `verified` 计数，drafted≥verified≥accepted）、`#62515`（FlashInferMLASparseSM90Backend 自适应验证）、`#64777`（EAGLE draft attention 用 target kernel block size）；TRT-LLM `#21972`（shared drafter KV pools + FP8 MLA context inputs + KV cache V2 copy/memory 修复）、`#21815`（MiniMax-M3 Eagle3 draft-loop API 对齐，36 passed）、`#22009`（KDA replay acceptance 一次 Triton launch，44→18µs eager）；SGLang `#50039`（DSv4.1 CP compact DSpark 输出保持）。
6. **跨硬件/异构正确性（强化第 15 条；AMD silent + Metal overflow）**：SGLang `#50096`（NPU GLM-5.2 FP8 KV for DSA prefill CP，gsm8k 198/200=0.990）、`#49054`（NPU HiCache Mooncake L3 for DSV4/GLM-5.2）；SGLang `#48773`（AMD topk-v2 split-launch short row 错选 → `illegal memory access`，#47960 回归的静默错误铁证）；llama.cpp `#32221`（**Metal `mul_mv` 目标偏移 int32 溢出** → 升 uint64，M5 上 FAIL 2097152/2149580800 → OK，强化第 9 条 int 溢出）、`#32135`（BF16 RMS_NORM CPU+CUDA）、`#31921`（Hexagon SSM_SCAN op）；TRT `#21316`（Qwen3.8-Flash-Next attention-DP padding SM120）、`#20497`（Qwen3-VL 视频布局跨 E/PD）。

---

## 2. vLLM：2026-11-13 提交（102 条）—— Rust Frontend 簇 + KV cache 接口重构(#61452) + 量化静默(#63097/#66784) + DSv4.1 mono decode 融合(#65461) + 投机验证(#64156) + Mistral Large 4(#64534)

### 2.1 `#61452` [Core][KV Cache] Move KV cache allocation policy out of KVCacheSpecs（RFC #42449 part 2，breaking 接口变更，强化第 8 条）
- **主题**：`KVCacheSpec` 现在只描述层的**张量布局**；**分配策略**（window/recycling/prefix caching/Mamba modes）移到强制它的 KV cache manager。`KVCacheSpec` 体系移除了 `FullAttentionSpec`/`SlidingWindowSpec` 等 policy-typed specs、`KVCacheSpecRegistry` 与 `Platform.register_custom_kv_cache_specs`，插件/out-of-tree 维护者受影响。
- **内容详情**：每层返回 `KVCacheLayerConfig(ManagerConfig(manager_cls, params), spec)`，`params` 为冻结 `Params` dataclass；「group」即共享同一 block table 的层。提供了 gist drop-in shim 兼容旧 worker/model runner，但 shim 为 stopgap。测试：CPU/GPU 单测 + GLM-5.3-Flash / DeepSeek-V4.1-Flash 的 E2E（其余 E2E pending）。
- **应用场景**：vLLM KV cache 内存/缓存管理接口重构（worker↔engine core 契约变更）。
- **研判**：【事实】PR 显式声明破坏插件接口、提供 shim、属 RFC #42449 part 2。【观点】属「KV cache 管理接口重构」（第 8 条），与 SGLang mem_cache 统一池形成跨引擎共振；out-of-tree 维护者须按新 API 迁移；记为官方一手（breaking 级，含 shim）。**证据等级**官方一手（breaking 级）。
- **来源**：vllm-project/vllm #61452（合并于 2026-11-13T04:47:49Z）。

### 2.2 `#65461` [ROCm][Perf][DSv4.1] Mono: merge K2(l) and K1(l + 1) into one launch at TP8（DSv4.1 mono decode 融合，perf，强化第 11、15 条）
- **主题**：MI355X（gfx950）TP8 下 DeepSeek-V4.1 mono decode（`--monokernel`）把第 l 层 K2 与第 l+1 层 K1 融成一个 persistent kernel，覆盖 39 个 target 层边界中的 34 个 + 两个 DSpark draft 边界，每个合并边界省一次 launch gap 与 K1 启动。
- **内容详情**：**bit-exact**（FULL-graph replay 每 rank 输出一致）；**perf**：concurrency 1、1×TP8、DSpark k=6，**每步 −96.8 µs（−2.21%）**，layer bench −1.71~−2.08 µs/边界；GSM8K（1319，8-shot）main 91.24 → PR 91.85。**Scope**：仅 TP8；TP2/TP4/gfx942 与独立 K1/K2/FFN 构建与 main 字节一致；无新 env var。
- **应用场景**：DeepSeek-V4.1-Flash 在 MI355X 的 mono decode 内核融合。
- **研判**：【事实】PR 给 −2.21% 每步、GSM8K 91.24→91.85 与 bit-exact 验证。【观点】属「ROCm mono decode 内核融合」（第 11、15 条），纯 perf 且位精确；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #65461（合并于 2026-11-13T02:04:03Z）。

### 2.3 `#66784` [Bugfix][Model Loader] Handle merged-linear padding for Qwen4Exp and Kimi KDA（Qwen4Exp/Kimi KDA merged-linear padding 清零，fix，强化第 4 条）
- **主题**：保持 Qwen4Exp HyperConnection 与 Kimi KDA projection 的对齐 padding 在冷启动与权重 reload（direct/layerwise/load plan）下为 0。行为收进 `MergedColumnParallelLinear(padding_size=N)`，替代各模型的私有拷贝。
- **内容详情**：padding 不是 checkpoint shard（不参与 `shard_ids`/fused-checkpoint slicing），排除出 `weight_loader_numel`，在 init 与两个 weight loader 中清零，并经 `padding_fill_params()` 声明使 direct reload/load plan 零填充；`padding_size=0` 保持现行为。根因：对齐行无 checkpoint 张量，声明为 shard 会报 missing shard、阻塞 layerwise completion，dummy/stale 字节在 reload 后存活。Qwen4Exp（NVIDIA+AMD）与通用 Kimi GDN projection 采用；Kimi-K3 子类移除私有 padding 处理。
- **应用场景**：Qwen4Exp / Kimi KDA 权重加载的对齐 padding 正确性。
- **研判**：【事实】PR 给「对齐行无 checkpoint 张量 → 声明为 shard 报 missing、reload 存活」根因与 `padding_fill_params()` 修复。【观点】属「量化/对齐 padding 静默错误」（第 4 条），reload 后 stale 字节是典型静默错误源；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：vllm-project/vllm #66784（合并于 2026-11-13T04:14:48Z）。

### 2.4 `#63097` [Model] Load q/k/v scales for the quantized Kimi-K2.6（Kimi-K2.6 量化 q/k/v scales 加载，fix/feature，强化第 4、13 条）
- **主题**：为 `MLAAttention` 注册 compressed-tensors KV-cache 量化方法，把 Kimi-K2.5/K2.6 checkpoint 的 q/k/v scales 映射进嵌套 MLA attention 模块；scale 名翻译限定在 Kimi 作用域。Fixes intel/auto-round#2370。
- **内容详情**：4× B200（`tp=4`）上用 `intel-ai/Kimi-K2.6-MXFP4-FP8ATTN-CT-AutoRound`（106 shards / 524.76 GiB）验证：**183/183** checkpoint q/k/v scales 在每个 TP rank 精确匹配 post-load scale buffer（`RANK=0..3 LOADED=183 MISSING=0 EXTRA=0 MISMATCH=0`，`SCALE_VERIFICATION_PASS 183/183`），生成文本 coherent。
- **应用场景**：Kimi-K2.6 MXFP4-FP8 量化 MLA 的 q/k/v scale 正确加载。
- **研判**：【事实】PR 给 183/183 scales 跨 4 rank 精确匹配与生成验证。【观点】属「量化 KV/MLA scale 加载正确性」（第 4、13 条），scale 漏接是静默错误主线；记为官方一手（fix/feature 级，含精确匹配对照）。**证据等级**官方一手（fix/feature 级，含对照）。
- **来源**：vllm-project/vllm #63097（合并于 2026-11-13T16:11:38Z）。

### 2.5 `#67516` [Bugfix][HiSparse] Fix GLM-5.3-Flash startup: derive resident/hot specs without grouping on workers（GLM-5.3-Flash HiSparse 启动修复，fix，强化第 3、8 条）
- **主题**：GLM-5.3-Flash + HiSparse 在 `main` 启动报 `ValueError: KV cache layout has not been resolved yet`（workers 在 engine core 解析 KV 布局前调 `resolve_hisparse_specs`）。
- **内容详情**：根因——#61452 前只 group MLA specs；#61452 后把每个 layer config 传入，GLM-5.3-Flash 其余层（kpool tail rings、KDA state）混合 page size 走 packed path 读布局；#61452 与 #61243 几小时内先后合并、CI 互未跑对方。本 PR 直接构建 per-layer 的 resident/hot config（`get_hisparse_page_layer_configs`），不再依赖 grouping。TP4 下 engine core 布局相同 41 groups、1,114,112 device bytes/block。
- **应用场景**：GLM-5.3-Flash HiSparse 在 #61452 重构后的启动正确性。
- **研判**：【事实】PR 给「#61452 与 #61243 互未 CI、packed path 读未解析布局」根因与直接构建 per-layer config 修复。【观点】属「KV cache 接口重构连带回归」（第 3、8 条），属 breaking 变更的典型下游代价；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：vllm-project/vllm #67516（合并于 2026-11-13T18:38:39Z）。

### 2.6 `#64156` [Spec Decode] Consistent drafted, verified and accepted token counts（投机解码三计数一致，feature，强化第 14 条）
- **主题**：落 RFC #64099，投机解码现报告三计数：`drafted`（含 grammar-invalid）、`verified`（新，进入 target 前向的 drafted）、`accepted`（不变），满足 `drafted ≥ verified ≥ accepted`。
- **内容详情**：新增 `vllm:spec_decode_num_verified_tokens` 等计数器与 per-request `num_verified_tokens`/`per_step_verified`，覆盖 Prometheus / 日志 / `vllm bench serve`（Python + Rust）/ Rust frontend。结构化输出下 `drafted` 现含 grammar-invalid token（#29821 曾扣除），`accepted/drafted` 下降；MRV1 `ngram_gpu` 不再把 −1 padding 计为 drafted。worker 同时看 drafter 输出与 target 前向，故 model runner 在 `ModelRunnerOutput` 报两计数、scheduler 只计 accepted。
- **应用场景**：投机解码可观测性与计数一致性（跨 Python/Rust frontend）。
- **研判**：【观点】属「投机解码验证硬化/可观测性」（第 14 条），新增 `verified` 计数统一度量；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #64156（合并于 2026-11-13T15:27:40Z）。

### 2.7 `#64534` [Model] Add Mistral Large 4（Mistral Large 4 模型支持，feature，强化第 11 条）
- **主题**：为 vLLM 加 Mistral-Large-4-1T-A52B 支持（FP8 与 NVFP4，model runner V2）：Sparse MLA + DSA indexer、cross-layer KV sharing、SSMax、DSpark 投机、DCP/PCP 长上下文、NIXL P/D、HiSparse host KV offload；NVIDIA（Blackwell/Hopper）与 AMD ROCm（MI350 系列）kernel。
- **内容详情**：PR 自述「not meant to be merged as is；generic pieces 在独立 PR 上游化」，作为 open working branch 共享；Recipe：https://recipes.vllm.ai/mistralai/Mistral-Large-4-1T-A52B。声明使用了 AI 辅助。
- **应用场景**：Mistral Large 4（1T-A52B）在 vLLM 的推理支持（working branch）。
- **研判**：【事实】PR 自述为 working branch、通用部分另拆 PR 上游化。【观点】属「新旗舰模型支持」（第 11 条），但非最终合入形态；记为官方一手（description 级，PR 自述 working branch）。**证据等级**官方一手（description 级）。
- **来源**：vllm-project/vllm #64534（合并于 2026-11-13T15:05:06Z）。

### 2.8 `#50265` [Feature] add the option to continue serving unfinished requests after the recovery（故障恢复后继续服务，feature，强化第 8 条）
- **主题**：新增 `FaultToleranceConfig.request_failure_policy` 控制故障时未完成请求的处理：`"fail"`（默认，立即中止）或 `"recompute"`（幸存引擎上重排队、恢复后从 prompt 与已生成 token 重算 KV 继续生成）。
- **内容详情**：替代旧的布尔 `resume_requests_after_recovery`；orchestrator 可用 `params={"abort": true}` 丢弃某次 retry 的排队请求；在 expert parallel 部署指南中记录策略、恢复行为与 per-retry override。
- **应用场景**：vLLM 故障容忍部署的未完成请求续服。
- **研判**：【观点】属「KV/服务韧性」（第 8 条），recompute 策略提升长会话容错；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #50265（合并于 2026-11-13T07:57:39Z）。

### 2.9 `#62800` [Bugfix] Fix KV cache pages for hybrid models with draft layers（混合模型 + draft 层 KV 页修复，fix，强化第 8、14 条）
- **主题**：修复 vLLM 在混合层类型模型（即 hybrid attention + linear-attention target 上跑 draft model 的投机解码）下选 KV cache 页的方式。
- **内容详情**：FlashInfer 现按每 attention group 自身的 head count 选 kernel 页（MHA draft 紧邻 GQA target 否则拿到 target 大页、metadata builder 启动拒绝）；padded manager 页可拆成均匀 strided kernel 页且保留 manager 页边界；hybrid block alignment 计入 drafter 的 KV bytes/token。与 #62961-#62964、#62963、#63279（FA4 dense warmup 覆盖）独立；NIXL/Mooncake connector 传每 group 的 layer names，使 P/D 传输用与 runner 相同的 kernel block。
- **应用场景**：混合模型（linear-attention target + draft）的 KV 页与 P/D 传输一致性。
- **研判**：【观点】属「混合模型 KV 页 + 投机 + PD 一致性」（第 8、14 条），与 SGLang mem_cache/Hybrid 重构呼应；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：vllm-project/vllm #62800（合并于 2026-11-13T23:33:36Z）。

### 2.10 `#47547` [Model] Use FlashInfer for Gemma 4 with an NVFP4 KV cache when FA4 is unavailable（Gemma 4 无 FA4 时 NVFP4 KV 走 FlashInfer，fix，强化第 13 条）
- **主题**：无 FA4 时 `Gemma4Config` 把 Triton 钉给 Gemma 4 的异构 head；但 Triton 读不了 NVFP4 KV cache，导致 `--kv-cache-dtype nvfp4` 在 SM8x/SM12x 上后端选择失败。本 PR 在 plain Gemma 4 + nvfp4 cache 下改钉 `FLASHINFER`（自 #46963 起 FlashInfer 在 SM8x/SM12x 读该 cache）。
- **内容详情**：仅 plain Gemma 4（text-only 或非多模态前缀 LM）在 SM8x/SM12x 无 FA4 时、未显式设后端下走 FlashInfer + nvfp4；其余 dtype（含 `nvfp4_4over6`）与 DiffusionGemma/EmbeddingGemma2 仍 TRITON_ATTN；显式后端不动；有 FA4 处不变。新增 `test_gemma4_without_fa4_backend`（3 个 plain case + 2 复用模型）在 `main` 上 nvfp4 case 失败、PR 上 5 case 全过（FlashInfer 0.7.0.post1 / 0.7.1rc）。
- **应用场景**：Gemma 4 在 SM8x/SM12x 的 NVFP4 KV cache 后端选择。
- **研判**：【观点】属「低比特 KV 后端兼容」（第 13 条），无 FA4 时回退 FlashInfer 读 NVFP4；记为官方一手（fix 级，含测试）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：vllm-project/vllm #47547（合并于 2026-11-13T15:24:06Z）。

### 2.11 `#66051` [ROCm][Muse Spark] DSpark Markov sampling with one launch a round (--monokernel)（Muse Spark DSpark Markov 单 launch，perf，强化第 11、15 条）
- **主题**：MI355X 上 Muse Spark Open `--monokernel`：把 DSpark 7 个 Markov round 的 draft 各跑成**一个 Triton launch**（bias GEMV + add + Muse output transform + logits cache write + Gumbel-max），其余退回 vLLM 循环。
- **内容详情**：7 轮 CUDA graph：**498–501 → 296 µs**（c=1）、**562 → 315 µs**（c=2）；Full AgentX（同节点同镜像）：interactivity **+3.1%**（c=1，tok/s/GPU +2.2%），c=2 interactivity −1.1%（噪声内）、tok/s/GPU **+4.0%**；live AIME25 server 中 567,826 draft rows 0 差异；精度在噪声内（GSM8K c=1..16、AIME25 64 seeds −0.78±0.41pp、RULER NIAH）；无新 flag、无额外显存（KV pool 内 0.02%）。
- **应用场景**：Muse Spark 在 MI355X 的 DSpark 投机 Markov 融合。
- **研判**：【事实】PR 给 498→296µs、interactivity +3.1%、567,826 rows 0 差异与精度噪声内。【观点】属「ROCm 投机采样融合」（第 11、15 条），单 launch 降开销；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #66051（合并于 2026-11-13T22:53:53Z）。

### 2.12 `#67196` [XPU][Perf][Spec Decode] Enable triton greedy_argmax on XPU（XPU Triton greedy_argmax 投机，perf，强化第 15 条）
- **主题**：启用 Triton greedy_argmax（来自 #57770）加速 XPU 上投机解码的 greedy draft-token 采样；此前在 XPU 回退 Torch，#64897 修回退后本 PR 验证并启用 Triton 路径。
- **内容详情**：bench 用 Qwen3-4B + DFlash（`num_speculative_tokens=15`、synthetic acceptance）在 tp=2、`max-num-seqs 32`、concurrency 1/8/16/32 跑 `vllm bench serve`。具体提速数值 PR 未在截断文本中完整给出，记为 perf 描述级。
- **应用场景**：Intel XPU 上投机解码的 greedy 采样加速。
- **研判**：【观点】属「XPU 投机采样」（第 15 条），启用 Triton 路径替代 Torch 回退；具体数值未完整，记为官方一手（perf 级，描述级）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：vllm-project/vllm #67196（合并于 2026-11-13T20:08:39Z）。

### 2.13 其他要点（标题级 / 描述级 / 性能级）
- `#61395` [EPLB] Support heterogeneous DeepSeek V4/V4.1 DSpark draft experts（DSpark draft 异构成员 EPLB，feature 级，强化第 14 条；V4.1 draft 在 target EPLB 开时默认启用，draft `draft_num_redundant_experts` 从 `--eplb-config` 读）。
- `#67272` [Recipe] MiniMax-M3.1-Flash: MI355X Agent（MI355X AgentX recipe 对齐 InferenceX#5035，NVFP4 KV + DBO + breakable CUDA graphs，perf/recipe 级）。
- `#67333` [Bugfix][KV Connector] Stop forwarding shutdown() to LMCache until it no longer aborts（LMCache shutdown 转发中止修复，fix 级，强化第 8 条）。
- `#67050` [Recipe][ROCm] MiMo-V2.6: use ROCM_AITER_UNIFIED_ATTN + encoder-parallel opt-in（MiMo-V2.6 ROCm recipe，recipe 级）。
- `#67194` [Perf][GLM-5.3] build eager attention metadata on the host for ViT（GLM-5.3 ViT eager attention metadata 主机构建，perf 级）。
- `#62515` [Spec Decode] Enable adaptive verification for `FlashInferMLASparseSM90Backend`（FlashInfer MLA sparse SM90 自适应验证，feature 级，强化第 14 条）。
- `#64777` [Bugfix][Spec Decode] Use the target's kernel block sizes for EAGLE draft attention（EAGLE draft attention 用 target kernel block size，fix 级，强化第 14 条）。
- `#67344` [ROCm][gfx1151] ROCM_RDNA_ATTN: prefer 16-token KV blocks（RDNA 偏好 16-token KV block，perf 级）。
- `#67466` [ROCm][DI-CI] Enable MiniMax-M3.1-Flash in AMD CI pipeline（MiniMax-M3.1-Flash 进 AMD CI，ci 级）。
- `#67485` [Bugfix] Fix startup OOM in KV cache sizing warmup by releasing kernel warmup's KV blocks（KV 容量 warmup 启动 OOM 修复，fix 级，强化第 3、8 条）。
- `#67296` [Bugfix] Keep extensible KV cache headroom for measured allocator fragmentation（extensible KV cache 为碎片留余量，fix 级，强化第 8 条）。
- `#67005` [Spec Decode] Keep dynamic speculative decoding for dense models with DP（dense 模型 DP 保留动态投机，feature 级，强化第 14 条）。
- `#61815` [Quantization] Allow online per-channel fp8 quantization of an unquantized lm_head（在线 per-channel fp8 量化 lm_head，feature 级，强化第 13 条）。
- `#66803` [Quantization] Skip kernel dispatch for online requantization sources（在线重量化源跳过内核 dispatch，perf 级）。
- `#59460` [Quantization] Make activation-quantization fallbacks explicit（激活量化 fallback 显式化，refactor 级，强化第 4 条）。
- 其余 Rust Frontend 簇（详见第 6 节第 17 条与可执行清单）：`#65994`/`#67256`/`#67187`/`#66222`/`#66530`/`#66540`/`#67359`/`#67398`/`#67409`/`#67383`/`#67437`/`#67346`/`#67295`/`#67420`/`#67476`/`#67553`/`#67304`/`#67533`/`#67280`/`#67549` 等约 20 个 PR，覆盖 gRPC Render、mock engine、PCRE2 正则、stop strings、logprobs、CI/构建。

---

## 3. SGLang：2026-11-13 提交（78 条）—— mem_cache 统一内存池(#49913/#49886) + 量化静默(#42566/#49338/#48773) + Rust server 簇 + NPU 跨硬件(#50096/#49054) + MegaMoE(#48350/#45905) + PD TENT(#47025)

### 3.1 `#49913` [Feat] kimi_k3: Serve Kimi-K3.1 on the unified memory pool（Kimi-K3.1 跑在 unified memory pool，feature，强化第 8 条，mem_cache 重构顶点）
- **主题**：Kimi-K3.1（`KimiK31ForConditionalGeneration`，KDA hybrid + DSA sparse attention）此前拒绝 `--enable-unified-memory`（unified pool 放不下 DSA indexer keys、不能存带 block scale 的 FP8 latent rows、DSA backend 直接读 `req_to_token` 而非翻译 unified pool 虚拟 id）。本 PR（栈顶点 #49719→#49720→#49775→#49912→#49966→#49913）使其无额外 kernel launch 地跑在 unified pool。
- **内容详情**：composite sub-pool 上每 DSA MLA 层声明两 full-lifetime part：① latent rows（`MLAGeometry`，`row_format=fp8_ds_mla`，656 B：NoPE values + block scales + RoPE + UE8M0 scales，Kimi-K3.1）；② indexer keys（每页一 blob 存 page 的 keys 再 scales，paged indexer kernel 读取；复用上一层 top-k 的层不声明）。每 64-token page 存每层 64 latent rows 连续 + index blobs + padding；每层 latent view `(pages,64,1,656)`。
- **应用场景**：Kimi-K3.1 在 SGLang unified memory pool 的 KV/状态统一管理。
- **研判**：【观点】属「mem_cache 统一内存池」（第 8 条），KDA/DSA 状态首次纳入 unified pool；记为官方一手（feature 级，栈顶点）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #49913（合并于 2026-11-13T09:12:31Z）。

### 3.2 `#49886` [Feat] mem_cache: Checkpoint hybrid Mamba state only at fork points and request ends（hybrid Mamba state 仅 fork point/请求末 checkpoint，feature，强化第 8 条）
- **主题**：hybrid 模型 prefix cache 原本在每个 prefill chunk 与 decode tracking boundary checkpoint Mamba state；多轮/agentic 流量下几乎每次命中都延伸上一轮 tail，内部 checkpoint 大多占 device 槽并写穿 host/storage、冲刷别的路径 tail。新增 `--hybrid-cache-checkpoints-level`。
- **内容详情**：`2`（默认）：仅 fork point 与 request end checkpoint（PD prefill worker 上 prompt 完成时一次），host/storage 仅收这些 state，覆盖 #48431 目标；`3`：旧行为（每 chunk/boundary）。新增 derived leaf `exec.mamba.mamba_path_state_cap`（level<3 下 `--mamba-max-states-per-path` 否则 1）；旧 `--mamba-max-states-per-path` 弃用但保留（`-1` 仍无限）。多轮复用省状态槽与写穿。
- **应用场景**：hybrid（Mamba/GDN）模型多轮/agentic 流量的 Mamba state 缓存效率。
- **研判**：【观点】属「hybrid 状态缓存收敛」（第 8 条），默认 level 2 减少无效 checkpoint 与写穿；记为官方一手（feature 级，默认关→默认 level 2 需谨慎）。**证据等级**官方一手（feature 级，默认变更）。
- **来源**：sgl-project/sglang #49886（合并于 2026-11-13T06:37:54Z）。

### 3.3 `#42566` [Fix] quantization: Load block-FP8 MoE TP shards that split checkpoint blocks（block-FP8 MoE TP shard 切分 checkpoint block，fix，强化第 4、6 条）
- **主题**：per-rank intermediate size 非 checkpoint block 整数倍的 block-FP8 MoE 层被拒（128×128 block、`moe_intermediate_size=640` 时 TP2/4/8 均中招：`ValueError: ... not divisible by weight quantization block_n = 128`）。
- **内容详情**：`Qwen3.8-Flash-Next-NVFP4` 仅 MTP experts 是 block-FP8（仅 draft 层受影响，cookbook 用 RadixArk BF16 head 绕过）；`Qwen/Qwen3.8-Flash-Next-FP8` 每层（48+MTP）都受影响（EP 下专家不切分故无变化）。选择「refine scales」而非 padding：padding 方案每 MoE 层都 pad 到整 128 block，TP2 约 +20%、TP4/TP8 约 +60% 显存且 +8~23%（Triton）/ +4~17%（DeepGEMM）更慢。本 PR 精化 scales 而非 padding。
- **应用场景**：block-FP8 MoE 在 TP（非 EP）下的分片加载正确性/效率。
- **研判**：【事实】PR 给 TP2/4/8 的 padding 显存 +20%~+60% 与速度 −4~−23% 对照、选择 refine scales。【观点】属「量化 MoE 分片静默错误」（第 4、6 条），误拒或非整 block 切分是典型坑；记为官方一手（fix 级，含对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：sgl-project/sglang #42566（合并于 2026-11-13T08:14:22Z）。

### 3.4 `#49338` [Fix] compressed-tensors: Reject WNA16 MoE TP shards that split a scale group（WNA16 MoE TP shard 切分 scale group 拒载，fix，强化第 4、6 条）
- **主题**：`CompressedTensorsWNA16MoE` 按整 scale group 收窄 w2 scales；当 MoE TP shard 非 `group_size` 整数倍时，每 rank 拿到错误 K 范围的 scales。
- **内容详情**：例 `moe_intermediate_size=640`、`group_size=128`、`moe_tp_size=4`：每 rank `K=160=128+32`，`create_weights` 建 1 group/rank，通用 loader 按 5//4=1 收窄，Triton kernel 需 `cdiv(160,128)=2` groups——半 K 元素用错 scale、kernel 越读至 1-group 张量尾。`Qwen3.8-Flash-Next W4A16` 在 `--tp 8 --ep 2`（8× A800）上每 EP1 rank `illegal memory access`、长生成退化为 `!!!!`。vLLM #41161 已在加载期拒此布局；本 PR 让 SGLang 同拒（启动清晰报错，建议兼容 TP 或 EP）。
- **应用场景**：WNA16 MoE 在 TP（非 EP）下的分片加载正确性。
- **研判**：【事实】PR 给「半 K 用错 scale → illegal memory access / `!!!!` 退化」根因与 A800 复现。【观点】属「量化 MoE scale 布局静默错误」（第 4、6 条），与 #42566 并列；记为官方一手（fix 级，含根因 + 复现）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：sgl-project/sglang #49338（合并于 2026-11-13T15:27:11Z）。

### 3.5 `#48773` [Fix] [AMD] topk-v2: Fix wrong picks on split-launch short rows（AMD topk-v2 split-launch 短行错选，fix，强化第 4 条 AMD 静默错误）
- **主题**：GLM-5.2 MXFP4 on MI355X（TP8/EP1/MTP3，`SGLANG_OPT_USE_TOPK_V2=true`，InferenceX recipe）在 AgentX c=4 约 30 分钟后 8 rank 同时 `HIP error: an illegal memory access`；DSA draft-extend 步的 per-call output check 抓到 `TOPK BAD`：page indices 超出 KV pool，sparse attention 解引用越界。
- **内容详情**：根因——#47960 把 register path 的 refinement 移出 inline（`refine_ties_cold`），非 0-bin overflow 走 noinline `collect_overflow`，使 streaming +0-bin case 成该 path 唯一 inline `refine_ties`；split-launch short row 上 scores 舍入到 binary16 +0（41024 中 20000 个，`PAGE_TABLE`）时 `PAGE_TABLE` 错、`INDICES` 对。是 #47960 回归。
- **应用场景**：AMD（MI355X）topk-v2 在 split-launch 短行下的正确性。
- **研判**：【事实】PR 给「8 rank 同时 illegal memory access + `TOPK BAD step=37639` + #47960 回归」根因与 `PAGE_TABLE` 错/INDICES 对对照。【观点】属「AMD 量化/稀疏静默错误」（第 4 条），越界解引用属静默错误铁证；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：sgl-project/sglang #48773（合并于 2026-11-13T08:27:25Z）。

### 3.6 `#50096` [Feat] [NPU] attention: Support FP8 KV for DSA prefill context parallelism（NPU DSA prefill CP 用 FP8 KV，feature，强化第 13、15 条）
- **主题**：GLM-5.2 on Ascend 950 可把 DSA KV 存 packed FP8；DSA prefill context parallel（PCP）路径需用该格式做 Top-K indexing 与 gathered cache 上的 attention。
- **内容详情**：packed FP8 PCP prefill 经注册的 quantized Lightning Indexer（local Q、gathered paged Index K、其 scales），保留 BF16 PCP indexer 路径；复用现有 packed FP8 QSFA 调用于 PCP attention 与普通 FP8 attention。精度：gsm8k **198/200 = 0.990**；bs=1 input64k out32 的 PCP2 step 时间（interleave / zigzag × quant / C8quant）有截图对照（数值未在文本中提取，记为 perf 描述级）。
- **应用场景**：Ascend（华为）上 GLM-5.2 DSA prefill 上下文并行的 FP8 KV。
- **研判**：【事实】PR 给 gsm8k 198/200=0.990 与 PCP step 时间截图。【观点】属「NPU 低比特 KV + CP」（第 13、15 条），FP8 KV 跨 CP 落地；记为官方一手（feature 级，含精度 + perf 描述级）。**证据等级**官方一手（feature 级，perf 描述级）。
- **来源**：sgl-project/sglang #50096（合并于 2026-11-13T07:27:39Z）。

### 3.7 `#48350` [Perf] flashinfer_cutedsl: Use swap-AB MoE on Blackwell（Blackwell swap-AB MoE，perf，强化第 12 条）
- **主题**：在 SM100/SM103 用 FlashInfer #7290 的 swap-AB split CuTe DSL MoE（需 FlashInfer 0.7.4，由 #49771 pin）；依赖已满足，本 PR rebased。
- **内容详情**：dispatch 对确切 SM100/SM103 选 `swap_ab=True`（按 weight tensor device），其余架构保留 legacy；selector 驱动准备、processed-weight cache key、`CuteDslMoEWrapper` 构造；weight 准备用公开 `interleave_cute_dsl_moe_weights`（16-row groups for swap-AB、64-row 否则），检查点值/global scales 保留、无重量化；复用现有 ModelOpt / `nvfp4_online` 加载与 W4A4 scalar/per-token 路径。
- **应用场景**：Blackwell（SM100/SM103）上 SGLang 的 swap-AB MoE（NVFP4）。
- **研判**：【观点】属「MegaMoE / NVFP4 MoE 跨硬件」（第 12 条），Blackwell swap-AB 融合；记为官方一手（perf 级，依赖 FlashInfer 0.7.4）。**证据等级**官方一手（perf 级）。
- **来源**：sgl-project/sglang #48350（合并于 2026-11-13T06:28:51Z）。

### 3.8 `#49380` [Fix] moe/router: Reject CPU-offloaded bias in warp router（warp router 拒 CPU-offloaded bias，fix，强化第 7 条）
- **主题**：在选 CUDA-only warp router 前，于 `bias_layout_ok` 中拒 CPU-resident correction biases；`--cpu-offload-gb` 下仍有非 warp route 可用。
- **内容详情**：Fixes #49334（@janbernloehr 定位根因并提 `b.is_cuda` guard、用真实 Nemotron SM120 验证）；扩展示例 warp-router layout test 加 aligned contiguous CPU bias（仅对齐不应选 CUDA kernel）。本地 CPU-only predicate check（CUDA hidden）：unmodified main 接受 aligned CPU bias、patch 拒。CUDA-registered router test / E2E 未本地跑（needs follow-up 标注）。AI-assisted + 本地 review。
- **应用场景**：MoE warp router 在 CPU-offload bias 下的分派正确性。
- **研判**：【观点】属「MoE 路由精度/放置」（第 7 条），CPU-offload bias 误选 CUDA kernel 的修复；评审标注 CUDA E2E 未本地跑，记为官方一手（fix 级，needs follow-up）。**证据等级**官方一手（fix 级，needs follow-up）。
- **来源**：sgl-project/sglang #49380（合并于 2026-11-13T06:54:49Z）。

### 3.9 `#50039` [Fix] deepseek-v4.1: Preserve compact DSpark outputs with CP bounded replay（DSv4.1 CP compact DSpark 输出保持，fix，强化第 14 条）
- **主题**：follow-up #47406（DeepSeek-V4.1 Prefill CP + DSpark + decoder SWA bounded replay 的 compact-output 处理）。#47406 的 compact gather path 与 #45534 既有 full-shard scatter 在 DSpark 下同时启用，导致「Full shard 布局 vs Compact tail 布局」错配。
- **内容详情**：例 16,384 extend tokens、CP4、128 tail：每 rank 拥 4,096 full-shard rows 但仅 32 compact-tail rows；full-shard scatter 后 rank 的 compact gather 用 full-extend metadata 读 compact 数据 → 错。本 PR 协调两者布局契约使 DSpark CP 输出保持正确。关联 #45534、#47406。
- **应用场景**：DeepSeek-V4.1 Prefill CP + DSpark 的 compact 输出正确性。
- **研判**：【观点】属「DP/CP 解耦 + 投机正确性」（第 14 条），布局契约错配修复；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：sgl-project/sglang #50039（合并于 2026-11-13T15:28:03Z）。

### 3.10 `#46132` [Perf] cuda-graph: Raise the decode graph ceiling default to 128 for 35 to 60 GiB GPUs（35–60GiB GPU 解码图上限默认 128，perf，强化第 3 条默认值）
- **主题**：默认 decode graph ceiling 按内存档读 8/48/32/256；46GiB L40S/A40 在 batch 32 以上跑 decode eager（早于 24GB 卡）。本 PR 把 35–60GiB 档在 `tp_size<4` 下默认 128。
- **内容详情**：L40S closed loop（128 in / 256 out）：Qwen3-8B C=48/64/128 分别 +2.3%/+2.1%/+1.7%；Qwen3-4B +3.3%/+3.3%/+2.4%；Qwen3-1.7B +32.5%/+21.0%/+6.4%；**Qwen3-0.6B C=48 翻倍 +119%**（median TPOT 10.9→4.8ms）。8B 约 +2%、0.6B 在 C=48 翻倍；ceiling 128 费 8B 0.19GB graph 内存、0.9s capture、1.0% KV tokens。无模型/kernel 代码变更。
- **应用场景**：中显存（35–60GiB）GPU 的 decode CUDA graph 覆盖上限。
- **研判**：【事实】PR 给 L40S 各模型 C=32/48/64/128 的 +0.1%~+119% 对照（0.6B 翻倍）。【观点】属「decode graph 默认值优化」（第 3 条），低风险免费收益；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：sgl-project/sglang #46132（合并于 2026-11-13T15:28:21Z）。

### 3.11 `#50364` [Feat] quantization: Support dense SiLU+mul fusion for masked per_token_group_quant（masked per_token_group_quant 的 dense SiLU+mul 融合，feature，强化第 13 条）
- **主题**：`per_token_group_quant` 可把 SiLU-and-multiply 融进 per-token group 量化；`silu_and_mul_mode="dense"` 在 FP32 算 SiLU 与乘积、舍入一次回源 dtype，融合输出与单独存 dense 激活再量化 bit-identical。default 模式在乘前把 SiLU 舍入回源 dtype。
- **内容详情**：dense 模式仅接受 flat `[N,2H]` 布局；masked `[E,N,2H]`（喂 masked grouped GEMM）原 raise `ValueError`——但 kernel 本不需要该限制，masked launcher 只是没转发 flag。本 PR 让 masked layout 也接受 dense 模式（sorted layout 仍拒）；dense flag 仅 dense 模式追加到 launcher 参数，既有 flat 行为不变。
- **应用场景**：SGLang 量化激活（SiLU+mul）与 per_token_group_quant 融合。
- **研判**：【观点】属「量化激活融合」（第 13 条），masked 布局补齐 dense SiLU+mul 融合；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #50364（合并于 2026-11-13T23:36:08Z）。

### 3.12 `#47025` [Feat] disaggregation: Add opt-in Mooncake TENT transfer engine backend for PD（Mooncake TENT PD 传输后端，feature，强化第 8 条）
- **主题**：Mooncake 下一代传输引擎 **TENT** 可按请求选 transport、按 backlog/带宽把流量摊到多 NIC、把 foreground/background 流量隔离到独立 QP pool。本 PR 把它作为 Mooncake PD connector 的 **opt-in** 后端评估；**Classic TE 仍为默认**。
- **内容详情**：新增 `KVTransferEngine` 接口（TE/TENT 共享）+ `TransferIntent` 枚举（FOREGROUND/BACKGROUND）；`SGLANG_MOONCAKE_TRANSFER_ENGINE` 选 `te`（默认）或 `tent`；Elastic EP 与 expert backup 依赖 TE 内部，TENT 下显式报错。TENT：RDMA only（钉 `--disaggregation-ib-device`，`MOONCAKE_PROTOCOL=tcp`/`MC_FORCE_TCP=1` 切 TCP，其余拒）；metrics 在 `9100+gpu_id`；handshake 用 `session_id`。单节点 NIC 拥塞下 TENT 降 KV 传输延迟，TTFT 因 prefill 计算主导变化不大（参考点非通用性能声明）。
- **应用场景**：Mooncake PD KV 传输的 TENT 后端（opt-in 评估）。
- **研判**：【观点】属「PD KV 传输后端可扩展」（第 8 条），TENT 为 opt-in 新传输引擎；记为官方一手（feature 级，opt-in、默认不变）。**证据等级**官方一手（feature 级，opt-in）。
- **来源**：sgl-project/sglang #47025（合并于 2026-11-13T23:20:26Z）。

### 3.13 `#49636` [Feat] kv-shard: Let out-of-tree backends and pools join logical-page KV sharding（out-of-tree 后端加入逻辑页 KV 分片，feature，强化第 8 条）
- **主题**：logical-page KV sharding（`--enable-kv-cache-sharding`）原仅 admit 固定名（`fa3` GQA prefill backend、`mooncake`/`nixl` transfer backend），scratch 预算按标准 K/V 或 MLA row 定。out-of-tree 后端/池实现相同契约却无法参与。
- **内容详情**：`arg_groups/kv_shard_hook.py` 把 admit 的集合移为 module-level 集合，加 `register_kv_shard_prefill_backend(name)` / `register_kv_shard_transfer_backend(name)`；`KVCacheConfigurator.page_shard_scratch_row_bytes()` 改为可重写方法（sharded pool 存更宽 row 时覆盖）。in-tree 无新增注册，纯扩展点。
- **应用场景**：第三方/out-of-tree 注意力与传输后端接入 SGLang 逻辑页 KV 分片。
- **研判**：【观点】属「KV 分片扩展点」（第 8 条），与 vLLM #61452 的 out-of-tree 维护者接口变更呼应；记为官方一手（feature 级，扩展点）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #49636（合并于 2026-11-13T23:35:32Z）。

### 3.14 其他要点（标题级 / 描述级 / 性能级）
- `#45905` [Refactor] moe: simplify FlashInfer MegaMoE autotune handoff（MegaMoE 启动 autotune 简化，refactor 级，强化第 12 条；去 native-cache JSON 传输与 `FLASHINFER_MOE_EP_KNOB_CACHE` 进程级变更，依赖 FlashInfer 0.7.4 #6415/#6416）。
- `#49054` [Feat][NPU] HiCache: Support Mooncake L3 for DeepSeek-V4 and GLM-5.2（NPU HiCache Mooncake L3 for DSV4/GLM-5.2，feature 级，强化第 8、15 条）。
- `#50004` [Fix][NPU] moe: Add ascend_ops backend checks across MoE models（NPU MoE ascend_ops 后端检查，fix 级）。
- `#50107` [Fix][NPU] kimi-k3: Skip the CuTe DSL TGV tactic registration on NPU（NPU 跳 CuTe DSL TGV tactic，fix 级）。
- `#49701` [Fix][NPU] EPLB: preserve NZ storage, reuse TP, and isolate draft capture（NPU EPLB 保 NZ 存储，fix 级）。
- `#50177`/`#50178` [Fix] DeepSeek-V4: Skip decode attention TP setup when off / Default input_embeds to None（DSV4 decode attention TP setup / input_embeds，fix 级）。
- `#49912` [Feat] dsa: Honor the page stride of index-K buffers（DSA 尊重 index-K buffer page stride，feature 级）。
- `#49719`/`#49720` [Refactor] mem_cache: Keep the GDN/KDA speculative ReplaySSM ring in a request sub-pool（GDN/KDA ReplaySSM ring 入 unified sub-pool，refactor 级，强化第 8 条）。
- `#49775` [Fix] mem_cache: Size a fused draft's KV region in the dtype the draft resolves（fused draft KV region 按 draft 解析 dtype 定尺寸，fix 级）。
- `#49966` [Refactor] mem_cache: Compose same-lifetime parts into one unified sub-pool（同生命周期部分合成 unified sub-pool，refactor 级）。
- `#49376` [Perf][NVIDIA] kimi-k3: Bound q6 preparation by server concurrency（K3 q6 准备按并发限，perf 级）。
- `#49477` [Perf] attention: Optimize FA3/FA4 tree attention（FA3/FA4 tree attention 优化，perf 级）。
- `#43136`/`#47645` [Perf/AMD] DCP: Slice gathered MHA prefixes with host offsets to drop GPU syncs / Plan DCP asm verify shard with draft tokens under CUDA graph replay（AMD DCP 去 GPU sync / CUDA graph replay 下 DCP asm verify shard，perf/fix 级，强化第 15 条）。
- `#49603` [Fix][AMD] docker: Pin UCX and NIXL commits for ROCm PD disaggregation（ROCm PD 钉 UCX/NIXL，fix 级）。
- `#50363` [Perf][AMD] hicache: Submit staged write-back copies with hipMemcpyBatchAsync on ROCm（ROCm HiCache 用 hipMemcpyBatchAsync 提交 staged write-back，perf 级，强化第 8 条）。
- `#50181` [Fix] hicache: Refuse a runtime write_back policy under buffer_only host memory（HiCache buffer_only host memory 下拒运行时 write_back，fix 级）。
- `#49878` [Fix] disaggregation: Attribute decode HiCache hits to host and storage（decode HiCache 命中归因 host/storage，fix 级）。
- `#49686` [Feat] disaggregation: Make the PD top-logprobs metadata width configurable（PD top-logprobs 元数据宽可配，feature 级）。
- `#49631`/`#49633`/`#49635`/`#49640`/`#49650`/`#49653`/`#49663`/`#49907`/`#50189`/`#50190`/`#50199` [rust-server] SGLang Rust server 簇（SHA-256 aarch64 指令、/v1/loads snapshot 锁、MM processor 失败处理、--warmups/--gc-warning、abort hook、media 免 fetch slot 等，feature/fix 级）——与 vLLM Rust Frontend 簇并列，见第 6 节第 17 条。
- `#50289` [Doc] cookbook: Refresh SemiAnalysis AgentX launches（AgentX launch 文档刷新，doc 级）。
- `#50369` [Chore] release: Bump sgl-kernel version to 0.4.11（sgl-kernel 升 0.4.11，chore 级）。
- `#50016` [Chore] deps: Bump sgl-deep-gemm to 0.2.8 and sgl-flash-mla to 1.0.5（依赖 bump，chore 级）。

---

## 4. TensorRT-LLM：2026-11-13 提交（46 条）—— 量化静默(#22108 SwiGLU scaling 766×/93%) + Kimi K3 SM107 MegaMoE(#21960) + 共享 drafter KV pools(#21972) + Qwen3.8-Flash-Next-FP8 服务修复(#21930/#21316) + Cosmos3-Edge-FP8(#21572) + MiniMax M3 Eagle3(#21815)

### 4.1 `#22108` [fix] Correct MXFP4 FP8 MoE SwiGLU scaling（MXFP4 FP8 MoE SwiGLU scaling 修正，fix，强化第 4、6 条）
- **主题**：TRTLLM-Gen MoE 用 `W4A8_MXFP4_FP8` 与 gpt-oss SwiGLU（`alpha=1.702`、`beta=1`、`clamp=7`）时，把 beta/clamp/expert biases 以**反量化单位**传给消费累加器单位的 kernel ABI，导致 GB200 用例输出约 **766× 过小、93.2129% 元素 mismatch**。
- **内容详情**：用 `fc31_input_gate_dequant` 转换新加载的 FC1 bias、`fc2_input_dequant` 转换 FC2 bias；在 `cache_derived_state()` 中经 `materialize_activation_params()` 从 `module.activation` 重建 alpha、beta/clamp 除以 FC1 gate dequant scale。覆盖全 reload/meta init/runtime-weight restore 后重建 derived state，不保留重复未缩放 activation buffer；forward kernel 与精度容差不变。恢复 gpt-oss 回归覆盖（NVBug 6178914 排除项），把 blanket 多 GPU 排除拆为独立 gate：NVBug 6178915（DEP/NVLINK 与 DTP/AllGather 用错 raw-FC1 input scale）、NVBug 7085659（logical intermediate width 非 128 对齐的 TP shard）；TEP/DEP/DeepEP/aligned TTP 启用；raw-FC1 input-scale 修正另由 #22106 提。
- **应用场景**：TRT-LLM W4A8_MXFP4_FP8 MoE 的 SwiGLU 缩放正确性。
- **研判**：【事实】PR 给「输出约 766× 过小、93.2129% 元素 mismatch」与 NVBug 6178914/6178915/7085659 关联。【观点】属「量化 MoE 静默错误」（第 4、6 条），单位混淆致 93% mismatch 是静默错误铁证；记为官方一手（fix 级，含 mismatch 数字）。**证据等级**官方一手（fix 级，含 mismatch 数字）。
- **来源**：NVIDIA/TensorRT-LLM #22108（合并于 2026-11-13T14:55:47Z）。

### 4.2 `#21960` [perf] Add the Kimi K3 SM107 MegaMoE generation-phase tactic at 512 tokens per rank（Kimi K3 SM107 MegaMoE 生成相 tactic，perf，强化第 11、12 条）
- **主题**：加静态 SM107 Kimi K3 生成相 tactic（命中 `max_tokens_per_rank==512` 的 exact runner key）：用 `[256,128,256]` tile、`[4,1,1]` cluster、phase-interleaved 调度、atomic work IDs、bulk FC2 store、`(2,2)` epilogue flag batching。
- **内容详情**：默认 tactic 解析在 K3 entry 先于通用 SM107 bucket 表；其余 bucket/runner key 保持默认；K3 entry 也进 `get_valid_tactics` 候选。无 env var。**Perf**：作者报 `bench_moe` 1.089×（power-law）/ 1.013×（balanced），六进程；未测 E2E。**Correctness**：作者报与通用 tactic bit-exact（cap 512 的 K3 shape，eager + CUDA graph replay）；新增 CPU-only test `test_sm107_kimi_k3_genphase_tactics_take_precedence`（未覆盖候选枚举）。
- **应用场景**：Kimi-K3 在 Rubin（SM107）的 MegaMoE 生成相融合。
- **研判**：【观点】属「Kimi-K3 SM107 MegaMoE 量化融合」（第 11、12 条），1.089×/1.013× 但无 E2E、correctness 作者报 bit-exact 未由新增 test 覆盖；记为官方一手（perf 级，描述级 + needs follow-up 标注）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21960（合并于 2026-11-13T02:12:17Z）。

### 4.3 `#21972` [feat] Shared drafter KV pools, FP8 MLA context inputs and KV cache V2 copy/memory fixes（共享 drafter KV pools + FP8 MLA context inputs + KV cache V2 修复，feature，强化第 14 条）
- **主题**：Kimi K3（SM107）modeling_v2 路径所需的引擎/ KV cache manager / attention op 运行时件。
- **内容详情**：① 并行化长 KV cache block-offset 拷贝（`kvCacheManagerV2Utils.cu`，每 pool/sequence 行长行分 bounded chunk，多 CTA 拷）；② executor setup 后把 PyTorch allocator cap 外预留至多 1 GiB（防 cap 用尽全部空闲致 OOM）；③ 接受 FP8 K/V（drafter 层 KV 池 opt-in 持有需 `TRTLLM_DISAGG_SHARED_DRAFT_KV`，PD 亦需）；④ item 3 的 scale fix 修正 op 自身 FP8 copies（每现有路径、FP8 KV cache + 非单位 KV scale 的 run 都生效）；⑤ 其余为 KV cache V2 copy/memory 修复。多数为通用路径改进，仅 FP8/共享 drafter 生效时走新路径。
- **应用场景**：Kimi-K3 混合模型（KDA/Mamba）投机解码的共享 drafter KV 与 FP8 MLA context。
- **研判**：【观点】属「混合模型投机解码共享 drafter KV」（第 14 条），并行化 block-offset 拷贝 + allocator 余量防 OOM；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：NVIDIA/TensorRT-LLM #21972（合并于 2026-11-13T10:44:20Z）。

### 4.4 `#21930` [fix] Runtime fixes and small kernel changes from serving Qwen3.8-Flash-Next-FP8（Qwen3.8-Flash-Next-FP8 服务修复，fix，强化第 4、13 条）
- **主题**：来自实际服务 Qwen3.8-Flash-Next-FP8 的 7 个独立改动（C++/CUDA/Triton/Python/tests，20 文件、+493/−52）。
- **内容详情**：① MNNVL one-shot all-reduce（`triggerCompletionAtEnd` 默认 true，保留现行为，`false` 时 PDL completion 在 grid-dependency 同步后即触发）；② QSA kernels 6 个 row-count kernel 排除 `num_rows` 特化、2 个 index-scoring 排除 row-metadata view 对齐、sparse GQA split-K/merge 排除 `num_rows`（须确认 unaligned metadata view 正确性）；③ causal conv update 排除 `batch`/`conv_state_indices_ptr` 对齐特化（更广卷积 claim 未全覆盖）。评审标注部分实现未含完整 all-reduce、须核实真实 experts 路径截断到 local token count。
- **应用场景**：Qwen3.8-Flash-Next-FP8 实际服务下的运行时/内核修复。
- **研判**：【观点】属「量化模型服务运行时修复」（第 4、13 条），7 合 1 须逐项 review/rollback；评审标注 all-reduce/专家路径未全核实，记为官方一手（fix 级，needs follow-up 标注）。**证据等级**官方一手（fix 级，needs follow-up）。
- **来源**：NVIDIA/TensorRT-LLM #21930（合并于 2026-11-13T11:17:10Z）。

### 4.5 `#21316` [fix] Correct Qwen3.8-Flash-Next attention-DP padding on SM120（Qwen3.8-Flash-Next attention-DP padding SM120 修正，fix，强化第 4 条）
- **主题**：`Qwen3NextSparseMoeBlock.forward` 现 pad 新变量 `routed_hidden_states`（而非 `hidden_states`），仅当 attention DP 开、`tp_size>1`、且 `is_sm_120f()` 时；decode-front 调用、fallback `self.gate`、`experts` 收 padded tensor；shared expert 路径仍用未 pad `hidden_states`。
- **内容详情**：fix 依赖 routed experts 仅返 rank-local rows（diff 未展示）；CPU unit test mock `experts` 返 local count 行。test list `l0_rtx_pro_6000.yml` 删 42 行（38 active `TestDeepSeekV3Lite` + 2 disagg guided-decoding + 2 注释 `online_eplb`）加 1 行（`TestQwen3_8_Flash_Next::test_nv...`）。
- **应用场景**：Qwen3.8-Flash-Next 在 SM120（RTX PRO 6000）attention-DP 下的 padding 正确性。
- **研判**：【观点】属「量化模型 attention-DP padding 正确性」（第 4 条），SM120 特化；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：NVIDIA/TensorRT-LLM #21316（合并于 2026-11-13T20:59:43Z）。

### 4.6 `#20497` [fix] Preserve Qwen3-VL video layout across E/PD（Qwen3-VL 视频布局跨 E/PD 保持，fix，强化第 8 条）
- **主题**：Qwen3-VL encoder 把视频按带时间戳 frame group 处理，但 E/PD prefill 从原始 text + embedding row count 重建 prompt，丢掉时间戳与 frame 边界（保留 encoder 的 mRoPE positions）；本地 40-frame 复现中 prefill 收 1,011 token 而非 1,169，生成 token 与聚合推理偏离。
- **内容详情**：在可选 `DisaggregatedParams.multimodal_input_layout` payload 中携带 encoder 的精确 prompt 与多模态布局；Qwen3-VL 在 encoder 提交前导出、prefill 消费时保留 embedding-shape 校验；payload 在 `asdict`/JSON transport 存活；预处理提取精确 prompt 与元数后请求本地副本丢弃已消费布局。在推理前拒 mismatched mRoPE shapes（适用于每个转发 mRoPE handle 的多模态解聚 processor，不只 Qwen3-VL）。
- **应用场景**：Qwen3-VL 多模态在 E/PD 解聚下的视频布局正确性。
- **研判**：【观点】属「多模态 PD 布局正确性」（第 8 条），mRoPE/video layout 跨 E/PD 保持；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：NVIDIA/TensorRT-LLM #20497（合并于 2026-11-13T11:56:44Z）。

### 4.7 `#21572` [feat] Load nvidia/Cosmos3-Edge-FP8 with its W8A8/W8A16 step policy（Cosmos3-Edge-FP8 按 step policy 加载，feature）
- **主题**：`nvidia/Cosmos3-Edge-FP8` 是静态 ModelOpt FP8（Edge Nemotron-dense backbone，独立 Hub repo）；#17476 已为 Nano/Super 加载该 quant recipe。Edge 无 gate projection、无 audio tower，其发布 `diffusion_step_policy` 在首末 3 denoising step 跑 W8A16、中间 W8A8、reasoner W8A16。
- **内容详情**：注册 Hub ID、把 checkpoint-key remap 收进单函数、跳过 ModelOpt quantizer buffers（`*_quantizer._amax`/`.scale`）避免掩盖真实缺失权重；采样保持 Edge 默认（video 832×480/shift 3.0/guidance 5.0；text-to-image 640×640/guidance 4.0）。test 覆盖 manifest 非量化张量落位、`lm_head.weight`/`norm.weight` 仅跳过、step policy 匹配 parser schema（无 GPU 跑）；手动单 GPU（RTX PRO 6000 SM120）`TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1`。
- **应用场景**：Cosmos3-Edge-FP8 扩散模型的 W8A8/W8A16 step policy 加载。
- **研判**：【观点】属「扩散模型量化加载」，注册 Hub ID + step policy；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：NVIDIA/TensorRT-LLM #21572（合并于 2026-11-13T17:40:19Z）。

### 4.8 `#21815` [fix] Align MiniMax M3 Eagle3 draft-loop API（MiniMax M3 Eagle3 draft-loop API 对齐，fix，强化第 14 条）
- **主题**：MiniMax-M3 modeling_v2 Eagle3 初始化在 SM103 TP4/TP8 当 drafting 回退 shared worker 时失败：`_forward_draft_loop()` 被调用时缺 `accepted_tokens` 与 `pre_draft_seq_lens`（共享 API 在 #20815 改后返回一对 token buffer）。
- **内容详情**：更新两个 MiniMax worker 转发所需参数、保留 pre-draft seq-len 引用、消费双值返回契约；扩展现有 CPU worker API 回归自动发现每个 MiniMax-M3 NVFP4 变体（现 5 个 SM103/SM107 target），覆盖 fallback/prebuilt-buffer/fused/external shared-KV 路径；移除 NVBug 7028061 两 waiver 使 GB300 accuracy/acceptance gate 执行。**36 passed locally**。
- **应用场景**：MiniMax-M3 Eagle3 投机解码在 SM103/SM107 的 draft-loop API 一致性。
- **研判**：【事实】PR 给「36 passed locally」与 5 个 SM103/SM107 target 覆盖。【观点】属「投机解码 draft-loop API 对齐」（第 14 条），消 NVBug 7028061；记为官方一手（fix 级，含 passed）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：NVIDIA/TensorRT-LLM #21815（合并于 2026-11-13T00:51:29Z）。

### 4.9 `#22009` [perf] Record KDA replay acceptance in one Triton launch（KDA replay acceptance 一次 Triton launch，perf，强化第 14 条）
- **主题**：把 `KDAReplayState.update` 中 6 个 PyTorch op 替换为 1 个 Triton kernel `record_kda_replay_acceptance`，写 `max(num_accepted[i] - num_bonus_tokens, 0)` 到每请求 state slot；dummy 请求不写、空 batch 不 launch。
- **内容详情**：需 CUDA tensor、目标为一维 `int32`、可 strided；index/count 须 dense 一维 `int32`/`int64` 等长；可选 dummy mask 为 dense bool CUDA。行为变更：CPU tensor 路径现 raise `ValueError`（`unsupported tensor layout`）。作者报 batch 1–2 下 **44 µs → 18 µs（eager）、11 µs → 1.8 µs（CUDA graph）**。
- **应用场景**：Kimi-K3/GLM-5-next KDA 混合模型投机解码 replay acceptance 记录。
- **研判**：【事实】PR 给 44→18µs（eager）、11→1.8µs（CUDA graph）对照。【观点】属「混合模型投机 replay 性能」（第 14 条），6 op 合一 Triton；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：NVIDIA/TensorRT-LLM #22009（合并于 2026-11-13T02:10:17Z）。

### 4.10 `#21718` [feat] Run unquantized TRTLLM-Gen MoE on native BF16 cubins; fix routing tiers past 128 experts（未量化 TRTLLM-Gen MoE 跑 native BF16 cubins，feature，强化第 4 条）
- **主题**：把未量化 BF16 TRTLLM-Gen MoE 从 FlashInfer 移到 native TRT-LLM kernel；加 BF16×BF16 batched GEMM cubins 与含 low-latency/high-throughput 模板的 config.json。资格检查拒不支持的激活、未对齐 shape、expert 数 >2048、top-k >32、top-k ≥ slot 数；不支持 shape 回退别实现。
- **内容详情**：routing dispatch 无匹配 tier 时启动失败；默认/SigmoidRenorm 共享 tier `(128,8)/(256,8)/(512,8)/(2048,32)`，SigmoidBias 加 `(2048,32)`；SigmoidRenorm 设 `mSumEpsilon=1e-20f`；`SumNormalizePostprocess` 加 `sumEpsilon` 参数（默认 0）。native runner 加 routing/autotuning/DP token bucket/finalized+unfinalized 输出路径/对齐 TMA-OOB workspace；`run_bf16_moe` 返回 tensor 或 tensor list。移除非量化 BF16 的 FlashInfer provider 及其依赖探测、`use_flashinfer` 路径。
- **应用场景**：TRT-LLM 未量化 BF16 MoE 的 native kernel 路径（去 FlashInfer 依赖）。
- **研判**：【观点】属「MoE 路由/后端 native 化」（第 4 条），去 FlashInfer 依赖、routing tier 过 128 experts 修复；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：NVIDIA/TensorRT-LLM #21718（合并于 2026-11-13T10:08:05Z）。

### 4.11 其他要点（标题级 / 描述级 / 性能级）
- `#21977` [feat] MegaMoE: sync the vendored CuTe DSL kernels from upstream（MegaMoE 同步上游 CuTe DSL kernel，feature 级，强化第 12 条）。
- `#21288` [feat] Add SeaCache step caching for Cosmos3（Cosmos3 SeaCache step 缓存，feature 级）。
- `#22163` [test] Add GLM-5.3-Flash NVFP4 core performance coverage（GLM-5.3-Flash NVFP4 核心性能覆盖，test 级，强化第 11 条）。
- `#22184` [fix] Keep 16-bit GVR register plans within their row window（16-bit GVR register plan 在 row window 内，fix 级）。
- `#21824`/`#21862` [chore] Update flashinfer-python 0.7.2→0.7.3（FlashInfer 依赖 bump，chore 级，与 vLLM/SGLang 同步）。
- `#22210`/`#22219`/`#22217`/`#22212`/`#22235` [infra] pre-merge/ post-merge waive / test entry / CI stage / blossom-ci 授权（infra/ci 级）。
- `#21799` [fix] Fix the breakable-prefill NCCL window hang behind Qwen3.5 NCCL_SYMMETRIC（breakable-prefill NCCL window hang 修复，fix 级，强化第 5、14 条）。
- `#21625` [fix] Kimi-K3 tool parser: warn only when a call names no declared tool（Kimi-K3 tool parser 仅当无声明 tool 时 warn，fix 级）。
- `#22180` [test] Use real text for Qwen3.8 and fix shared prompt counts（Qwen3.8 用真实文本 + 共享 prompt 数修复，test 级）。
- `#20987`/`#21614` [fix] Cap decoded audio samples in AudioMediaIO / Tokenize stop and bad strings when end_id given（audio 样本截断 / stop+bad 字符串 tokenize，fix 级）。
- `#22191` [chore] Upgrade Mooncake version to 0.3.14（Mooncake 升 0.3.14，chore 级，与 SGLang #47025 TENT 同期）。
- `#21974` [perf] Cut per-step host work of the executor, sampler and KV cache scheduler（executor/sampler/KV cache scheduler 每步 host 工作削减，perf 级，强化第 3 条）。
- `#21677`/`#21614`/`#21486` [fix] agent-flow modeling-bringup resumes / Tokenize stop / Cherry-pick optional native RankInfo authentication（agent-flow / tokenize / RankInfo 认证，fix 级）。
- `#21986` [refactor] Pin down multimodal item lifetimes across inline, scheduled and pending（多模态 item 生命周期固化，refactor 级，强化第 8 条）。

---

## 5. llama.cpp：2026-11-13 提交（22 条）—— v1.0.0 正式发布(#32190) + Metal int 溢出修复(#32221) + BF16 RMS_NORM(#32135) + moe-cache 重构(#32157) + 投机 p-min 警告(#31294) + Hexagon SSM_SCAN(#31921)

### 5.1 `#32190` llama.cpp : bump version to 1.0.0（llama.cpp v1.0.0 发布，feature/里程碑，新增部署安全 + 第 13、15 条）
- **主题**：llama.cpp 从 0.9.0 跨入 **v1.0.0 正式版**（2026-11-13T10:20:19Z 打 tag）。
- **内容详情**（官方 release notes，一手）：① 移除废弃公共 API（`llama.h` 删 `llama_set_warmup`、35 个废弃函数、`defrag_thold`/`--defrag-thold`，#30565）；② 把 NextN/投机解码 API 从 `llama-ext.h` 提升进 `llama.h`（`llama_set_ctx_state_src` 重命名）；③ `llama_context_params` 加 `moe_cache_size` 与 `moe_cache_cpu_share`（#30966）；④ 新增 **OCP FP8 KV caches**；⑤ **host-memory MoE expert cache（可配 CPU 份额）**；⑥ 加载外部 **PLE shards**；⑦ 三个新文本架构（Kimi-K3.1-Pro、Ripple、xing4_0）、drama-3 TTS、MageVL-Cowork 与 Kimi-K3.1 视觉；⑧ 加固 `llama-server`（SSRF、CORS、DNS rebinding、MCP 工具取消）、新增 `/v1/audio/speech`；⑨ 重设计 web UI（chat groups + markdown 编辑器）；⑩ `ggml` 升 v1.0.0。博客：https://huggingface.co/blog/ggml-org/llama-cpp-v1。
- **应用场景**：本地/端侧大模型推理引擎的 1.0 里程碑（API 稳定化 + 低比特 KV + MoE expert cache + 服务安全）。
- **研判**：【事实】官方 release notes 列出上述 10 项（含 #30565 删 35 函数、OCP FP8 KV、host-memory MoE cache、server 加固、ggml v1.0.0）。【观点】属「本地推理引擎 1.0 里程碑」——OCP FP8 KV 与 host-memory MoE cache 是端侧显存墙的系统性解法，server 加固把部署安全主线从补丁提升为正式能力（呼应 vLLM #32115/#32120）；记为官方一手（发布级，含 notes + 博客）。**证据等级**官方一手（发布级，含 notes）。
- **来源**：ggml-org/llama.cpp #32190（合并于 2026-11-13T09:06:07Z）；tag v1.0.0 发布于 2026-11-13T10:20:19Z。

### 5.2 `#32221` ggml-metal: fix mul_mv destination offset overflow（Metal `mul_mv` 目标偏移溢出修复，fix，强化第 9 条 int 溢出）
- **主题**：修复 `mul_mv` 目标偏移计算在大型 batched 输出下的 **32 位整数溢出**——当 `im * args.ne0 * args.ne1` 超过 `INT32_MAX` 时偏移溢出，使受影响目标切片保持未写。
- **内容详情**：把偏移算术在乘法前提升为 `uint64_t`。Fixes #30237。在 Apple M5 Metal backend 复现：修复前 `RESULT mv: FAIL (bad=2097152 of 2149580800)`；修复后 `RESULT mv: OK (bad=0 of 2149580800)`。AI 用作复现/分析辅助，root cause 与最终一行改动经作者独立 review 验证。
- **应用场景**：llama.cpp Metal（Apple Silicon）后端大 batched `mul_mv` 的正确性。
- **研判**：【事实】PR 给 M5 上 `FAIL bad=2097152/2149580800 → OK bad=0` 修复对照与 `uint64_t` 修复。【观点】属「大张量 int32 溢出静默错误」（第 9 条），未写切片即静默错误，跨引擎（vLLM/TRT/llama.cpp）int 溢出主线延续；记为官方一手（fix 级，含复现对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：ggml-org/llama.cpp #32221（合并于 2026-11-13T23:49:33Z）。

### 5.3 `#32135` ggml : add BF16 support to RMS_NORM (CPU, CUDA)（BF16 RMS_NORM，feature/perf，强化第 15 条跨后端）
- **主题**：给 CPU 与 CUDA 加 BF16 RMS_NORM（含 CUDA RMS_NORM+MUL 与 RMS_NORM+MUL+ADD 融合），是 #29675 之后「保持激活在 BF16」的下一步。
- **内容详情**：当前无任何模型产出 BF16 激活（故现有模型不受影响，F32 kernel 编译到相同 SASS）；CUDA `rms_norm_f32` 按 x/dst type 模板化（F32/BF16，math 留 F32 仅 load/store 转换），norm weight 留 F32，fused ADD 操作数可 F32/BF16；Vulkan/Metal/SYCL/OpenCL/OpenVINO/CANN 在 `supports_op` 拒 BF16（SpacemiT 回退 CPU）。测试：BF16 cases、NMSE 1e-5（CPU/CUDA reduction 末位差异）、fused 1e-4。**Perf**（RTX 5090 µs/run，F32→BF16）：2048 hidden 1行 1.01→0.85(−16%)、512行 2.39→1.95(−18%)、2048行 6.41→4.60(−28%)、8192行 70.11→15.42(−78%)*；4096 hidden 类似（末位 −54%~−78%*）。
- **应用场景**：llama.cpp BF16 激活路径（RMS_NORM）的跨后端支持与性能。
- **研判**：【事实】PR 给 RTX 5090 各 hidden/行数的 −16%~−78% 对照（大行最显著）与 NMSE 限制。【观点】属「BF16 激活/BF16 算子跨后端」（第 15 条），当前无模型产出 BF16 故为前瞻；记为官方一手（feature/perf 级，含对照，description 级：尚无模型真正用 BF16 激活）。**证据等级**官方一手（feature/perf 级，description 级）。
- **来源**：ggml-org/llama.cpp #32135（合并于 2026-11-13T13:41:17Z）。

### 5.4 `#32157` llama: move moe-cache code out of llama-graph（moe-cache 代码移出 llama-graph，refactor）
- **主题**：把 moe-cache 代码从 `llama-graph` 移到 `llama-moe-cache`，获得更清晰的子模块分离、加速 review。
- **内容详情**：纯重构，无行为变更；与 v1.0.0 的 `moe_cache_size`/`moe_cache_cpu_share`（#30966）参数体系配套的代码组织准备。AI 辅助。
- **应用场景**：llama.cpp MoE expert cache 代码组织（配合 v1.0.0 host-memory MoE cache）。
- **研判**：【观点】属「MoE cache 代码组织」，纯重构；记为官方一手（refactor 级）。**证据等级**官方一手（refactor 级）。
- **来源**：ggml-org/llama.cpp #32157（合并于 2026-11-13T10:48:30Z）。

### 5.5 `#31294` spec : Warn when --spec-draft-p-min is used with probabilistic draft sampling（投机 p-min 与概率采样冲突警告，fix/可用性，强化第 14 条）
- **主题**：`--spec-draft-sampling probabilistic` 下 draft 经 rejection sampling 接受（概率 min(1,p/q)），draft 的 top 概率（`--spec-draft-p-min` 检查的）并不能说明 target 会接受多少；`p_min>0` 常停掉 target 本可接受的大部分 draft。
- **内容详情**：本 PR 仅在投机初始化时打印警告（p_min>0 + probabilistic），建议 p_min 0；`p_min` 对其他 drafter/采样模式行为不变，无功能/性能变更。警告：`common_speculative_init: --spec-draft-p-min 0.75 stops sampled drafts early, consider setting it to 0 with --spec-draft-sampling probabilistic`。测量（p_min 0 vs 0.75，RTX 5090 + DGX Spark）在评论中；验证警告仅 probabilistic + p_min>0 时打印、输出与 master 一致。
- **应用场景**：llama.cpp 投机解码 `p_min` 与概率采样的误用警告。
- **研判**：【观点】属「投机解码可用性/正确性」（第 14 条），p_min 与 probabilistic 语义冲突的告警；记为官方一手（fix/可用性 级）。**证据等级**官方一手（fix/可用性 级）。
- **来源**：ggml-org/llama.cpp #31294（合并于 2026-11-13T12:25:33Z）。

### 5.6 `#31921` hexagon: add support for SSM_SCAN op（Hexagon SSM_SCAN op，feature，强化第 15 条）
- **主题**：给 Hexagon backend 加 f32 `SSM_SCAN` op 支持（线性注意力/状态空间模型算子的端侧落地）。
- **内容详情**：在 v75|v81|v85 Android 与 v81 WoS 设备测试；`test-backend-ops` **15/15 通过**。AI 用于头脑风暴/review/test。
- **应用场景**：llama.cpp Hexagon（骁龙 HTP）后端的 SSM_SCAN（GDN/Mamba 类线性注意力）算子。
- **研判**：【事实】PR 给 15/15 通过。【观点】属「Hexagon 线性注意力算子」（第 15 条），扩展端侧 SSM 覆盖；记为官方一手（feature 级，含通过数）。**证据等级**官方一手（feature 级，含通过数）。
- **来源**：ggml-org/llama.cpp #31921（合并于 2026-11-13T20:27:52Z）。

### 5.7 其他要点（标题级 / 描述级）
- `#32179` CUDA: split the PAD_REFLECT_1D grid to stay within the HIP grid limit（HIP grid 上限内拆分 PAD_REFLECT_1D grid，fix 级，强化第 15 条）。
- `#30099` CUDA: add SM103 SASS to non-native builds（非原生构建加 SM103 SASS，feature 级，强化第 15 条 Blackwell）。
- `#31544` vulkan: add TOP_K_MASK（Vulkan 加 TOP_K_MASK，feature 级）。
- `#32144` hexagon: improve buffer allocations and mappings (mmap stacking)（Hexagon buffer 分配/映射改进，perf/fix 级，强化第 15 条）。
- `#30021` ggml-cuda: hip: tune GCN MMQ configs（HIP GCN MMQ config 调优，perf 级）。
- `#29557` grammar: optimize matching with left factoring（grammar 左因子化匹配优化，perf 级）。
- `#32185`/`#32183`/`#32199`/`#32148`/`#32158` ci: windows-cuda ccache 1G / HIP FP8 vec FA spill 忽略 / Python 3.15 支持 / qdc timeout 缩减 / .npmrc allow-remote=root（ci 级）。
- `#32184` examples: clean-up old/obsolete stuff（示例清理，cleanup 级）。
- `#32191` examples: remove explicit version from test-cmake（test-cmake 去显式版本，配合 v1.0.0，cleanup 级）。

---

## 6. 累计结论延续与新增（继承 1–16，本期新增 17）

**累计结论（1–16 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性（发版当天合入 ≠ 在版内）；③ 默认值成为主要风险载体（本期 #46132 decode graph 上限默认 128 为低风险正向默认）；④ 异构/量化后端静默错误为高频；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务；⑥ 量化路径静默错误（scale/布局漏接）；⑦ 专家张量/路由归属与分页漏接静默错误；⑧ KV 卸载/HiCache/PD 健壮性（本期 vLLM #61452 + SGLang mem_cache 统一池 #49913/#49886/#49636 形成跨引擎缓存管理接口重构共振）；⑨ 长上下文 int32/int64 溢出静默错误（本期 llama.cpp #32221 Metal 溢出，跨四引擎印证）；⑩ MXFP8 KV 正确性簇；⑪ 新旗舰模型 CP/KV/性能成为一等公民；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径（本期 TRT #21960/#21977、SGLang #48350/#45905）；⑬ FP8/低比特 KV cache 跨引擎标准化落地（本期 llama.cpp v1.0.0 引入 OCP FP8 KV、vLLM #47547、SGLang #50096 NPU FP8 KV）；⑭ 投机解码验证路径跨引擎密集硬化（DFlash/DSpark/DFLASH/ReplaySSM/lookahead；本期 vLLM #64156/#62515/#64777、TRT #21972/#21815/#22009、SGLang #50039）；⑮ 跨后端可移植性成为显式架构主线（Apple MPS/NPU/Hexagon/SpacemiT/ROCm/OOT；本期 llama.cpp #32135 BF16 RMS_NORM、#31921 Hexagon SSM_SCAN、SGLang #50096/#49054 NPU、TRT #21316/#20497）；⑯ 新旗舰模型进入「默认性能化」阶段 + 量化/MegaMoE 通用化（上期为 GLM-5.3 默认性能化 + MegaMoE 跨 SM90）。

**新增第 17 条（推理服务前端 Rust 化成为跨引擎并行主线）**：本期 vLLM 出现密集 "Rust Frontend" 簇（#65994/#67256/#67187/#66222/#66530/#66540/#67359/#67398/#67409/#67383 等约 20 个 PR：gRPC Render、mock engine、PCRE2 正则、stop strings、logprobs、CI/构建），SGLang 同步密集 "rust-server" 簇（#49631/#49633/#49635/#49640/#49650/#49653/#49663/#49907/#50189/#50190/#50199 共 11 个 PR：SHA-256 aarch64 指令、/v1/loads snapshot 锁、MM processor 失败处理、--warmups/--gc-warning、abort hook、media 免 fetch slot）。两引擎在**同一 UTC 窗口**并行把服务/前端层迁移到 Rust，与「推理核心层（KV/量化/投机/MegaMoE）」优化并列成为服务架构层面的跨引擎共振。结论：推理引擎的能力竞争正从「核心算子/调度」外溢到「服务/前端层」，Rust 化是降低服务层延迟与并发开销的显式选择；部署侧应跟踪两引擎 Rust 前端的特性 parity 与稳定性。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 量化静默错误复核**：TRT-LLM `#22108`（W4A8_MXFP4_FP8 SwiGLU scaling 单位混淆 → 766× 过小、93.2129% mismatch）、`#21316`（Qwen3.8 attention-DP padding SM120）；SGLang `#42566`（block-FP8 MoE TP shard 切分 block）、`#49338`（WNA16 MoE TP shard 切分 scale group → A800 illegal memory access）、`#48773`（AMD topk-v2 短行错选 → illegal memory access）；vLLM `#63097`（Kimi-K2.6 q/k/v scales）、`#66784`（Qwen4Exp/Kimi KDA padding reload）。部署任何 MXFP4/FP8/NVFP4 MoE 或 attention-DP/TP 分片前，必须逐项核验 scale/布局/量化 kernel 兼容性，尤其 TP（非 EP）分片。
- **P0 — 引擎接口 breaking 变更**：vLLM `#61452`（KV cache 分配策略移出 KVCacheSpecs，破坏 out-of-tree 插件接口，需 shim）+ SGLang `#49636`（KV 分片扩展点）。自研/oob 后端维护者须按新 API 迁移。
- **P1 — KV/HiCache/mem_cache 重构**：SGLang `#49913`（Kimi-K3.1 unified memory pool）、`#49886`（Mamba state 仅 fork point checkpoint，默认 level 2）、`#49636`（out-of-tree KV 分片）、`#47025`（Mooncake TENT opt-in）、`#50363`（ROCm hipMemcpyBatchAsync）、`#49054`（NPU HiCache Mooncake L3）；vLLM `#62800`（混合模型 + draft 层 KV 页）、`#67516`（GLM-5.3-Flash HiSparse 启动，#61452 连带）、`#50265`（故障恢复续服）。
- **P1 — 跨后端/跨硬件**：llama.cpp `#32221`（Metal int 溢出修，强化第 9 条）、`#32135`（BF16 RMS_NORM，前瞻）、`#31921`（Hexagon SSM_SCAN）、`#30021`（HIP GCN MMQ）；SGLang `#50096`/`#49054`（NPU FP8 KV / HiCache L3）；TRT `#21316`/`#20497`（SM120 padding / Qwen3-VL 视频布局 E/PD）。
- **P1 — 服务前端 Rust 化（新增）**：vLLM Rust Frontend 簇 + SGLang rust-server 簇。若依赖任一引擎的服务层，应跟踪 Rust 前端的特性 parity 与稳定性，确认其所用 endpoint/行为未被迁移改变。
- **P1 — llama.cpp 1.0.0 升级**：升级注意 #30565 删 35 个废弃函数（`llama_set_warmup`/`defrag_thold` 等）、NextN API 重命名；评估 OCP FP8 KV 与 host-memory MoE expert cache（可配 CPU 份额）对本机/端侧部署的显存收益；确认 `llama-server` 加固（SSRF/CORS/DNS rebinding）是否满足内网部署要求。

---

## 7. 社区与视频资源（HN Algolia，2026-11-13 epoch 窗口 1794528000–1794614400）

> 以下均为社区/Show HN，非引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词（vllm/llama.cpp/sglang/tensorrt）检索中 sglang/tensorrt 均无命中。

- **llama.cpp v1.0.0 社区讨论（直接关联本期头号事件）**：HN 「Llama.cpp v1 Is Out」（**20 分**，2026-11-13）——社区对 v1.0.0 发布的讨论；「llama.cpp v1.0」（**5 分**）——同期讨论。**二手线索/未核验**（社区对发布的解读，非官方说明）。
- **Show HN / 生态**：「CyberGate – vLLM plugin to unlock cybersecurity without weight edits」（**7 分**）——vLLM 插件生态，与 vLLM 服务层扩展相关，**二手线索/未核验**；「Langfuse AI Gateway, open source rust gateway to trace every LLM call」（**9 分**）——Rust 写的 LLM 网关，与本期「服务前端 Rust 化」主线（第 17 条）形成社区侧呼应，**二手线索/未核验**。
- **推理正确性/数值**：「Eliminating Nondeterminism in LLM Inference with Integer-Only Arithmetic」（**3 分**）——整数算术消除 LLM 推理不确定性，与本期 int 溢出/确定性主线（第 9 条）主题相关，**二手线索/未核验**。
- **泛 AI 推理**：「Weebit Nano In-Memory Compute ReRAM Silicon Kit for AI Inference」（2 分）、「Waiting a little saves us over 50% on inference」（2 分）、「First inference results on Xbox Series X」（2 分）、「Competitively-priced EU-hosted DeepSeek: v4.1 Flash is now on Verda Inferen」（2 分）、「I built a tool to find local LLMs that fit your hardware」（3 分）——与本地/端侧推理部署相关，**二手线索/未核验**。
- 其余 HN 命中（共约 13 条 inference 相关）多为泛 AI/科技资讯，与推理引擎实现无直接关系，未列入。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **102**（page1=100 触 per_page=100 上限，page2=2，实计 102）/ SGLang 78 / TRT-LLM 46 / llama.cpp 22 合计 **248** 条（2026-11-13 UTC 窗口）；vLLM 单页触上限已用 page2 补全，无截断遗漏。
- **版本核验**：releases API 确认——**llama.cpp v1.0.0 于 2026-11-13T10:20:19Z 正式发布**（tag v1.0.0，非 prerelease）；vLLM 仍为 v0.33.0rc3、SGLang v0.5.23、TRT-LLM dev 线 v1.4.0rc3（prerelease）/ 稳定线 v1.2.1，窗口内均无新正式版本。
- **PR 正文来源**：44 个关键 PR 正文经官方 pulls API（markdown 官方一手）抓取，API 配额充足、未触发 60/h 限制、未走 HTML 页面抓取；其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度（如 TRT-LLM 大量 `nvbugs` 内部链接 PR 仅能凭标题/描述记录、无法核验内部 QA 状态，维持「描述级/needs follow-up」口径；vLLM `#64534` 自述为 working branch；llama.cpp `#32135` 暂无模型产出 BF16 激活故为前瞻）。
- **描述级/未全绿/needs-follow-up 标注**：TRT-LLM `#22108`（内部 nvbugs 6178914/6178915/7085659，实现落地/QA 状态以链接为准）、`#21960`（1.089×/1.013× 无 E2E、bit-exact 由作者报未由新增 test 覆盖）、`#21930`（评审标注 all-reduce/专家路径未全核实）、`#21718`（去 FlashInfer 依赖须验证兼容）、`#49380`（CUDA E2E 未本地跑）；SGLang `#48773`（AMD 静默错误铁证但 CI 标注见 `pr-states`）、`#50096`（PCP step 时间数值未在文本提取，perf 描述级）；vLLM `#67196`（XPU 提速数值未完整）；llama.cpp `#32135`（BF16 无模型真正使用，前瞻）等均已按实际缺失程度标注。
- **社区信号**：HN Algolia 引擎官方更新类故事无；社区侧「Llama.cpp v1 Is Out」（20 分）、「llama.cpp v1.0」（5 分）直接讨论 v1.0.0 发布，Rust 网关（Langfuse 9 分）与本期 Rust 前端主线呼应，均按二手线索/未核验记录。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/needs follow-up」。
- **技能检索（find-skills / agent-browser）**：依本自动化历次一致结论，find-skills 经 SkillHub 语义检索（EN+ZH）返回通用编程/生产力类技能（编程专家.Skill、腾讯文档、各类聚合/方案输出类），与「官方提交采集 + 证据分级归档 + 仓库同步」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/pulls API）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次 11 期一致。
