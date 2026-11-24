# 大模型推理优化每日热点档案

> 档案日期：2026-11-24（本自动化于该日执行）
> 关注窗口：2026-11-23（周一）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，四家单页取全无截断）+ releases API + 48 个关键 PR 正文（vLLM 17/SGLang 11/TRT-LLM 10/llama.cpp 10，均经官方 issues API 抓取 markdown，官方一手；配额 60 起、抓取后剩余 3，无降级）+ HN Algolia（HTTPS，`numericFilters` 按 2026-11-23 epoch 窗口 1795392000–1795478400）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 68 / SGLang 25 / TRT-LLM 41 / llama.cpp 34 合计 **168** 条（2026-11-23 为周一，较上一窗口周日 98 大幅回升，符合工作日规律）；**窗口内无新正式版本、无新 prerelease**（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、llama.cpp 正式版 v1.0.0 11-13、TRT-LLM v1.3.0 11-20 与滚动 RC v1.4.0rc4 11-18 均落窗口前不变；llama.cpp 仅推进 nightly b12638–b12655，prerelease）。本期头号主线为 **量化/数值边界的静默正确性铁证集中爆发**——vLLM `#69002`（torchao AWQ 融合层每个 shard 用自己的 `act_pre_scale`，此前全用最后一个 shard 的 scale，GSM8K 0.000→0.858、TP2 无法加载）、vLLM `#69085`（ROCm 非对称 int8 把零点在 matmul 结果舍入到 bf16 后才减，精度丢失使 Meta-Llama-3-8B-INT8-asym 从勉强通过跌到阈值下，0.696→0.732 on ROCm 10.1）、SGLang `#51138`（Grok-1 softcap router `exp(2x/c)` 溢出→NaN top-k 权重且 top-k id 也错）、llama.cpp `#32566`（few-row MMQ 全零块 amax=0 除零→NaN→整数转换 UB）。此外 vLLM `#69433` 揭示 MLA 分块上下文 LSE 原地合并的**竞态**致跨块 token 注意力输出错误（80 用例 7 次失败→0）。HN 引擎官方更新类故事无；124 分 NariQ 量化帖等均二手线索/未核验。

---

## 1. 版本状态与本期头号事件：无新版本 + 量化/数值边界静默正确性集中爆发 + MLA 合并竞态

**提交计数（2026-11-23 UTC，均 HTTP 200；周一）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 68 | 量化静默修（#69002/#69085/#67413/#69446）+ TRITON_MLA 1.7×(#69029) + ROCm SWA 38×(#63867) + DCP ShardQ(#66027) + GLM fused AR +8.7%(#69001) + KV Connector 静默注册地址(#59586) |
| SGLang | 25 | softcap router NaN(#51138) + CPU GPTQ 零点(#46559) + linker 栅栏(#49463/#49467) + AMD QSA FlyDSL(#45196) + deterministic decode(#47410) + runtime core 静默清理 1/4(#50803) + RUSTSEC(#52663) |
| TensorRT-LLM | 41 | KDA replay 编译键(#22811) + 多模态调度器假死锁(#19898) + TMEM 释放次序(#21851) + GLM-5.2 FC1 barrier(#21847) + Mamba2 strided(#22596) + DSV4.1 KVCache V2 OPTIONAL(#22352) + MXFP8 FC12 Rubin(#21589) |
| llama.cpp | 34 | GDN replay 替代 rollback(#31669) + few-row MMQ 零块(#32566) + Metal NVFP4(#32825) + BF16/F32 src1(#32677) + 自动最优量化(#15550) + OpenCL spec verify(#32716) + iPhone 命令缓冲(#31815) |
| **合计** | **168** | 工作日回升（上一窗口周日 98）；与上一期连续，无空档 |

### 1.1 头号事件一：量化/数值边界的静默正确性铁证（跨 vLLM/SGLang/llama.cpp，强化第 6/9/19 条）

- **vLLM `#69002` [Bugfix][Quantization] torchao AWQ 融合层逐 shard 应用 `act_pre_scale`（官方一手，fix，强化第 6 条量化 scale 静默错误）**：torchao AWQ checkpoint 给每个 linear 自己的 `act_pre_scale`，vLLM 把 q/k/v、gate/up 融合成一个权重后只保留一个 `act_pre_scale`，导致 `qkv_proj`/`gate_up_proj` 每个 shard 都用最后一个 shard 的 scale——**输出错误，且 TP=2 时直接加载失败**（`ValueError: Not supported args for copy_`）。修法按 shard 逐个应用。A100 40GB 实测：Qwen3-8B AWQ（tinygemm int4）GSM8K 5-shot strict-match **0.000→0.858**；新单测 main 失败、修后通过（torchao 0.18.0 CI 版与 main 版均验证）。**研判（观点）**：这是「融合权重的量化元数据未按 shard 保留」的典型静默错误——权重融合是常规优化、不是量化路径专属，说明第 19 条「非量化常规路径静默错误密度不亚于量化路径」应扩展为「**优化路径与量化路径的交叉处**」。

- **vLLM `#69085` [ROCm][Bugfix] 修非对称 int8 linear 在 ROCm 的精度丢失（官方一手，fix，强化第 6 条）**：ROCm 上非对称 int8（W8A8）走 `TritonInt8ScaledMMLinearKernel`，在 matmul 结果**舍入到 bf16 之后**才减 activation zero-point，丢掉大部分输出精度。`Meta-Llama-3-8B-Instruct-INT8-compressed-tensors-asym` 在 MI300 上原本只靠一道题通过，ROCm 10.1 升级（#62536）后跌到阈值下。修法在 Triton kernel 内从精确的 int32 累加器里减零点（仅 ROCm）。MI300X 实测 GSM8K strict-match（阈值 0.703）：ROCm 10.0 **0.708→0.740**、ROCm 10.1 **0.696→0.732**；153 passed/1 skipped，其他平台无变化。**研判（观点）**：量化零点减法必须发生在累加精度域，否则「数值正确但精度悄悄退化」会被 LM-eval 阈值直接吞掉——这类问题不会崩溃、只会让模型在临界模型上悄悄不达标。

- **SGLang `#51138` [Fix] softcap router 大 logits 下权重保持有限（官方一手，fix，强化第 9 条整型/数值边界）**：`fused_moe_router_shim`（Grok-1 router，`moe_softcapping=30`）把 softcap `c·tanh(x/c)` 算成 `(exp(2x/c)−1)/(exp(2x/c)+1)·c`，当 `x/c>≈44` 时 fp32 `exp` 溢出为 `inf`，`inf/inf` 得 NaN——**两个 kernel 都返回 NaN top-k 权重，top-k id 也错**（复现中 `[7,6]` 而非 `[0,7]`）。修法把 `x/c` 钳到 `[-20,20]`（`tanh(20)` 在 fp32 恰为 1，钳位精确且 `exp(40)` 不溢出）。L4 实测：1 token（CUDA-core）与 512 token（tensor-core）两条路径均从 `[nan,nan]/ids[7,6]` 变为 `[1.0,9.06e-11]/ids[0,7]`，与 float64 参考一致；普通 logits（softcap 30 随机输入）新旧差 4.5e-08/9.5e-07，仅 fp32 舍入。**研判（观点）**：`exp` 溢出→`inf/inf`→NaN 是数值边界的经典静默错误；MoE router 的 top-k id 一旦错，后面整条专家路径都错且无任何报错，须对大 logits 做专项回归。

- **llama.cpp `#32566` ggml-cuda few-row MMQ 量化处理零块（官方一手，fix，强化第 6 条）**：`quantize_mmq_few_q8_1` 量化全零块时 `amax=0` 导致除零，随后 `0×Inf=NaN`，C++ 中 NaN 转整数是未定义行为——`q` 未定义、`sum` 不保证。修法让全零块产出合法零量化值/零 scale/零修正和，非块保持原算术。few-row MMQ 会在输入列数不匹配 tile 宽度时用零填充未用列，故该路径实际可达；标准 `quantize_q8_1()` 早已修过、`quantize_mmq_few_q8_1` 漏接。

### 1.2 头号事件二：vLLM `#69433` MLA 分块上下文 LSE 原地合并竞态（强化第 19 条「非量化常规路径静默错误」）

- **vLLM `#69433` [Bugfix][MLA] 修分块上下文原地 LSE 合并的竞态（官方一手，fix）**：`merge_attn_states` 原生 kernel 可能在某个 head 的所有线程读完 LSE 之前就覆写它；当请求上下文跨两个 chunk、跨边界 token 合并时产出**错误注意力输出**。该原地合并由 `#50613` 引入。修法在 `accumulate_mla_context_chunk` 合并续块前先 clone prefix LSE。1×MI355X(gfx950) 实测：新测试在 main 上 3/3 失败、修后 5/5 通过；CI 对比用例 80 个里 main 7 次失败、修后 0 次；原有不匹配元素 3310/294912（1.1%）、最大绝对差 1.921875。**代价**：仅续块，CI 用例 `_compute_prefill_context` 282.0→288.8 µs。**研判（观点）**：这是本期唯一「并发竞态」型静默错误（其余为数值/量化/元数据），说明静默正确性主线已从「算错」扩到「算得不稳定」；跨 chunk 前缀合并路径须做多次重复运行的一致性测试。

### 1.3 其他静默正确性与服务安全铁证（强化第 4/6/9/19/24 条）

- **vLLM `#59586` [Bugfix][KVConnector] NIXL block 大于 kernel row 时拆成 dense page（官方一手，fix，强化第 8 条 KV 传输布局正确性）**：runner 按 KV cache group 各自选 kernel block，NIXL 却跨所有 group 选一个公共 block 并套用统一比例。当 draft 等 group 的 backend 只支持更小 block 时，某 group 会保留比 NIXL 更大的 kernel block（如 128 vs 64），其 view `[num_blocks,H,128,C]` 的 `stride(0)` 跨两个 NIXL page、而 NIXL 的 `num_blocks` 翻倍——注册时要么断言失败，要么**在 `num_blocks*block_stride` 恰好匹配存储跨度时静默注册整行，传输寻址到错误字节**。修法在布局允许时把这些行注册为 dense NIXL page，否则报可操作错误。**影响任意 attention 类型，不止 MLA**。
- **vLLM `#69285` [Bugfix] elastic EP 扩缩容中间件尊重 root_path（官方一手，fix，第 24 条服务隔离族）**：`--root-path` 下 `ScalingMiddleware` 在 elastic EP 扩缩容时对 `/api/health`、`/api/is_scaling_elastic_ep` 返回 503，而这些路径本应保持可用——与 `#52794`（认证中间件）同根因、同修法。main 上 2 个 root_path 用例失败（`/api/health should stay up`，503≠200），修后 6 passed。**研判（观点）**：可观测性路径在扩缩容期间误报 503 会让自动扩缩容器误判实例不健康而反复重启，属服务可用性边界问题。
- **SGLang `#46559` [Fix][CPU] Qwen3.5-35B-A3B-GPTQ-Int4 量化规则与精度修复（官方一手，fix，强化第 6 条）**：CPU 路径把本应排除量化的 `in_proj_ba` 误判为 GPTQ 量化；且反量化沿用旧 GPTQ 约定给对称零点错误地 +1。修法委派给 `get_linear_quant_method`、CPU GPTQ 后端把打包零点规范到 7（解包时 +1 得正确 8）。CPU `--tp 3` GSM8K 100 例 ACC 98.75%。**注意**：Base CI ✗、AMD ROCm CI ✗（新测试未覆盖改动，按「官方一手但 CI 未全绿」处理）。
- **SGLang `#52663` [Fix] sgl-router 升级 url 至 2.5.9 修 RUSTSEC-2026-0592（官方一手，安全修复）**：`url 2.5.8` 的 `Url::origin()` 对每个嵌套 `blob:` 前缀递归一次，构造恶意 URL 可栈溢出（公告 2026-11-20）。修法仅改 Cargo.lock（`url` 2.5.8→2.5.9、`form_urlencoded` 1.2.2→1.2.3），无源码变更；`cargo test --lib` 1246 passed、clippy clean、Base CI ✓。
- **SGLang `#50803` [Refactor] srt runtime core 静默失败清理 1/4（官方一手，fix+重构）**：四部曲第 1 部（runtime core/entrypoints/managers/tools），**244 文件、+2916/−2405**（约 2200 行为测试、68 个测试文件）。修复一批「没有测试、默认启动不触发」的静默失败——如 Responses API/gpt-oss 流式 builtin tool 后首 chunk 丢失（`HarmonyError`）、`input_tokens` 恒为 0（读错 meta_info key）、部分 VLM 模板的 `stop` 触发 HTTP 500、`/v1/rerank` 问题等，每个修复配一个 main 上失败的回归测试。四部曲 2/4(#50804 模型与层)、3/4(#50805 kernels)、4/4(#50806 diffusion) 未在本窗口。**研判（观点）**：这是把「静默失败清理」工程化的样本——按包拆分、每修复配失败测试、四 PR 文件互不相交可任意顺序合并。

### 1.4 本期其他结构性观察

- **默认/后端切换的性能化（第 16 条延续）**：vLLM `#69001` 把 fused all-reduce + mHC + RMSNorm 在 Hopper TP4 decode 配置下默认开启（`#67099` 后续）；vLLM `#69446` 修 DSv4.1 在 H100 TP1（如 DP8/EP8）MXFP8 linear 走 `#66497` 的 BF16+FP8-GEMV 路径致每 GPU 多占约 12.4 GiB 的**内存回归**（47.67→60.21 GiB；修后 reduced reproducer 28.37→17.52 GiB，decode 代价约 1.0%/step@1 req、0.6%@8 req）。
- **编译态特化与热路径停顿**：TRT-LLM `#22811` KDA replay acceptance kernel 对 batch view 的 16 字节对齐做 Triton 特化，warmup 只跑纯 generation（`num_contexts==0`），首个混合 prefill/decode batch 在 forward 路径上编译 2 个新变体、所有在途请求等待——Kimi K3 TP8 实测服务健康后编译 4 次、其中一次落在 15.3 s token 流空档；修法对三个 view 指针加 `do_not_specialize_on_alignment`。TRT-LLM `#21207` 新增 opt-in `TRTLLM_GDN_FUSED_CONV` 融合 GDN prefill conv（默认关）。
- **KV 卸载/调度栅栏优化（第 8 条延续）**：vLLM `#69048` 修混合模型从 CPU tier 加载 partial-tail checkpoint 后，包含命中点的 chunk 从未被 store（新测试 main 失败、修后通过）；`#67957` Mooncake 加载失败后不再反复从 store 重载。SGLang `#49463` 把 Mooncake layer-wise load 的 slot 位置解析移到 forward fence 之前（与 read-done 依赖配合，作者明示两个改动未单独隔离、读为 read-done 证据而非本 PR 增益）；`#49467` 让 prefetch 发布的 KV 在请求排队期间也带 session ref（prefetched prefix 被逐出 1.5%→0%、TTFT mean 2.27→2.08s，但同批还含其他 prefetch 改动）。

---

## 2. vLLM：2026-11-23 提交（68 条）—— 量化静默四连(#69002/#69085/#67413/#69446) + TRITON_MLA 1.7×(#69029) + ROCm SWA 38×(#63867) + DCP ShardQ(#66027) + GLM fused AR +8.7%(#69001) + KV 地址静默注册(#59586)

### 2.1 `#69002` / `#69085` / `#69046`(实为#69446) / `#67413` 量化与数值边界 —— 见第 1.1 节

- `#67413` [Bugfix][Quantization] `moe_wna16` 下 checkpoint 排除的 routed experts 保持未量化（官方一手，fix）：`--quantization moe_wna16` 给所有 routed-experts 层分配 WNA16 方法，包括 GPTQ checkpoint 声明不量化的层，导致加载部分量化 checkpoint 的 MTP draft 时 `AttributeError: 'RoutedExperts' object has no attribute 'w2_weight'`。修法让 `MoeWNA16Config` 同时读 checkpoint 的 `dynamic` 规则与 `modules_in_block_to_quantize`（AutoGPTQ 自 #49539 起的判定），排除层用 `UnquantizedFusedMoEMethod`，并保留 AutoGPTQ 优先级（正向规则列在通配 `-:` 规则之前仍生效）。33 passed/9 skipped；main 上 2 个新用例失败。

### 2.2 `#69029` [Perf][Kernel] TRITON_MLA 合并 K_Buffer 载入并协同调度 head block（官方一手，perf，强化第 12/14 条）

优化 `_fwd_grouped_kernel_stage1`：把 head block 调度到 `grid[0]` 改善 L2 KV cache 局部性（此前 grid 按 `(batch, head_groups, NUM_KV_SPLITS)`，调度器先遍历 batch 维导致各 query head group 被跨步调度）；`K_Buffer` 改按行主序 `[BLOCK_N, BLOCK_DMODEL]` 载入以 128 位向量化合并、消除共享内存 bank conflict。**18 个 `(batch,seq_len)` 配置（DeepSeek-V2-Lite 几何）**：H100 **1.06–1.71×**（长上下文与 batched decode 形状 1.49–1.71×）、A100 1.05–1.47×（bs=128 时 1.41–1.47×）、L40S 1.06–1.38×。H100 Nsight Compute（64q1s8k）：L2 命中率 **55%→74%**、共享内存 store bank conflict 4.4-way/8.06%→0、动态共享内存 63.49→56.32 KB。约 15 行 kernel/launcher diff，保留 FP8 KV 反量化、SWA、GQA/MHA 分组路径与 Stage-2 reduction。

### 2.3 `#63867` [ROCm][Perf] Triton prefix-prefill 跳过 SWA 模型窗口外 KV tile（官方一手，perf，强化第 16 条）

ROCm 上滑动窗口模型（Gemma 2/3、Mistral/Zephyr、Cohere2、Granite、GLM-5.x DSpark draft）的 prefix prefill 跳过窗口外 KV tile 读取。gfx942(MI300X/MI325X)、ROCm 10.0、Triton 3.8.0：32,000 token 上下文 + 512 窗口 + head size 256（MI300X，KV cache>2 GB）kernel 时间最高 **38.37×**；32K prompt TTFT：zephyr-7b-beta **1.69×**、gemma-3-1b-it **2.21×**；GLM-5.2-FP8/GLM-5.3 + DSpark draft（8×MI325X TP8，每层 verify 步都跑该 kernel）ITL 降 **1.39×/1.38×**。所有配置逐位一致；无跳过时 kernel 0.99–1.05×、2K prompt TTFT 不变。402 passed/64 skipped，无新选项/环境变量/kernel 变体。

### 2.4 `#66027` [Attention][DCP] Sparse MLA DCP CuTe DSL A2A reduce + DCP ShardQ（Kimi K3.1 6/6，官方一手，feat，强化第 11/17 条）

Kimi K3.1 稀疏 MLA 优化最后一片（1–5 片已合入 #66030/#66028/#66037/#66532/#65685），本 diff 仅 DCP 部分，无 DCP 时行为不变：① CuTe DSL 直接 A2A LSE reduce（SM100），2 次 launch 替代 4 次；② DCP ShardQ——把 TP ShardQ(#63956) 扩到 DCP==TP（FlashInfer sparse MLA，SM100），每 rank 把本步 prefill 请求的 KV shard（MLA 与 indexer page）推入对称内存 workspace，再按 head 处理一段 prefill token；prefill token 跳过 query all-gather、LSE combine 与 top-k merge，decode token 仍走 DCP 路径。workspace 容纳 `max_model_len` token，装不下的 step 回退 main 的 DCP 路径，无调度器改动；由 `attention_config.sparse_mla_shard_q` 控制（默认开，同 TP ShardQ），需 fp8 indexer K cache。`nvfp4_ds_mla + DCP` 现也可接受。**研判（观点）**：延续「新旗舰模型上下文并行成一等公民」（第 11 条），把 prefill 从 all-gather TP 路径分流到 shard-Q 路径；性能数字来自同 8×B300 官方 checkpoint 的端到端 TTFT 测试（正文含 108k token 结果，描述级）。

### 2.5 `#69001` [Perf][GLM-5.3-Flash] Hopper 上默认开 fused all-reduce + mHC + RMSNorm（官方一手，perf，默认化）

`#67099` 后续，扩展至 Hopper，对受支持的 TP4 decode 配置默认开启。4×H100 服务测试（GLM-5.3-Flash + MTP 5 token，random 2→1024，并发 1）：输出吞吐 **391.10→425.10 tok/s（+8.7%）**、benchmark 时长 83.78→77.08 s，TTFT/TPOT 相近；aime25 exact_match 0.7667（融合后）。**注意**：正文只给了融合后 aime25 一个值，未给融合前对照精度数字，故「精度无损」为作者口径（描述级）。

### 2.6 其他要点（标题级 / 描述级）

- **MoE 量化**：`#66456` MXFP8 加入 FlashInfer CuteDSL batched experts（DeepEP low-latency），使 MXFP8 MoE checkpoint 可在 SM100/SM103 用 `--all2all-backend deepep_low_latency`/`nixl_ep` 服务；B200 + flashinfer 0.7.3 上 19 passed（11 个新用例）；不可跑的 MXFP8 配置启动期拒绝，NVFP4 与默认值不变。
- **GDN/模型**：`#68368` 复用 Qwen GDN 做 OLMo Hybrid prefill checkpoint；`#68368` 之外还有 `#57011` 给 model loader 加 sleep/wake/reload 生命周期钩子、`#67349` input embeddings/final norms/LM heads 只在用到的 PP stage 构建、`#67998` 向 connector 暴露逐 stage KV cache 元数据、`#66930` EPD receiver cache 排除 metadata-only item。
- **投机解码**：`#69274` 允许 DFly 跑在任意 compressed-tensors target scheme（NVFP4）；`#68651` DecodeBenchConnector 与合成 DFly 接受；`#60961` 拒绝 decode topK 超出 [1,8192]（修后 RTX 2000 Ada 55 passed、pre-fix 8193 不报错，MI50/ROCm 与完整 serving 未跑）。
- **KV Offload / 服务**：`#69048`/`#67957` 见 1.4；`#68674` OTLP tracing 开启时收集 request stats；`#68408` Rust frontend `allowed_tools` tool_choice。
- **多模态/其他**：`#68802` 视频加载套用 HF video processor 帧数上限；`#69468` 仅 Cohere tokenizer 模式注入 `cohere_format` 默认模板 kwarg；`#4a1620a0`(`#69285`) root_path 中间件见 1.3；`#f17a0e40` LoRA worker-local tensor adapter 私有加载。
- **Recipe/ROCm**：`#67443` Step-5-Preview MI325X/MI355X recipe；`#67857` MiniMax-M3.1-Flash MI355X AgentX PD over Mooncake（InferenceX run 六并发点全过、GSM8K 95.98–97.19%，功率测量无效故无能效声明——第三方基准/描述级）。
- **CI/治理**：多条 `[CI]`/`[Governance]`（`#69497`/`#69416`/`#69056`/`#68884` 等）；`#69066` 弃用 TorchAO 量化后端。

---

## 3. SGLang：2026-11-23 提交（25 条）—— softcap router NaN(#51138) + CPU GPTQ(#46559) + linker 栅栏(#49463/#49467) + AMD QSA FlyDSL(#45196) + deterministic decode(#47410) + runtime core 清理(#50803) + RUSTSEC(#52663)

### 3.1 `#51138` / `#46559` / `#50803` / `#52663` 静默正确性与安全 —— 见第 1.1/1.3 节

### 3.2 `#45196` [Perf][AMD] Qwen3.8 QSA decode 用 AITER 融合 FlyDSL sparse decode on gfx950（官方一手，perf，强化第 12/17 条）

ROCm 上 Qwen3.8 QSA decode 与投机 verify 原走两遍（Triton gather 把选中 KV 拷到 scratch + Triton decode 每 `(row,KV head)` 一个 program），decode batch 下多一次选中 KV 往返；32k 上下文 + MTP 单 MI355X 上 QSA attention 占 decode GPU 时间约 9.4%（每层调用约 120 µs）。AITER 新提供融合核 `aiter.ops.flydsl.qsa_sparse_decode`（AITER #6670），直接从 pool 经 `req_to_token` 读选中 KV、按 CTA 切分每行的 selection。本 PR 在 HIP 下优先走该核（条件：gfx950、bf16 q、bf16/fp8 KV、head_dim 256、GQA group≤16），覆盖 decode、target verify 与 draft extend；`recommended_num_splits` 按 batch size 固定 split 数以适配 CUDA graph，workspace 按捕获最大值一次分配。GSM8K 精度与 AMD 后端行为见正文（描述级），需 AITER 带该算子。

### 3.3 `#47410` [Perf] deterministic 模式大 split 预算下启动固定数量 Triton decode block（官方一手，perf，强化第 4/16 条）

deterministic 模式下 Triton decode attention 随 batch 增大显著变慢：MI350X gpt-oss-20b 在 64 并发吞吐仅 normal 模式的 0.33×、256 并发 0.21×。根因：为 batch invariance 固定 split size=256 token、按最长请求（gpt-oss 131k 上下文=512 splits）定预算，kernel 每个 split 启一个 block，1k–2k token 请求约 99% block 空转。修法让分组 kernel 跑更少 block、每 block 循环本请求的 splits（足够覆盖 GPU、每请求/head tile 至少 32 个）；merge 也在本请求 split 数处停止，每个 split 计算与合并顺序不变。**实测**：deterministic gpt-oss-20b 从 0.33×/0.21× 升至 **0.63×/0.50×**（64/256 并发）；gpt-oss-120b +63%/2.1×。normal 模式逐位不变（MI350X MLA/GQA batch 1–64 为 main 的 0.98–1.05×）；deterministic GQA 逐位一致，MLA 因 merge 提前结束可差几 ULP、仍保持 batch-invariant。

### 3.4 `#49463` / `#49467` linker 栅栏与 session 覆盖 —— 见第 1.4 节

### 3.5 其他要点（标题级 / 描述级）

- **调度/请求生命周期**：`#52510` 在 observer 延迟响应前释放已完成的 prefill 请求（修 SpecForge capture 插件每个 prefill-only 请求触发 `ValueError: response still in DECODING` 使 server 退出；main 上回归用例失败、修后通过，无模型输出变化）；`#e2d88a64` deterministic 固定 Triton decode block 数（见 3.3）。
- **AMD**：`#52234` QK-norm 权重 dtype 与 q/k 不同时 ROCm 回退非融合 norm（`#39071` 开启的融合 JIT 只按 `q.dtype` gate，fp32 权重 + bf16 q 触发 `Dtype value [float32] not in the allowed options`，main 上 2026-11-22 起两次 scheduled run 失败）；`#48649` ROCm QSA decode MQA logits 加速（padded Q、group-strided launch）；`#45196` 见 3.2。
- **kernel_plugins**：`#52438` 让不满足规格的 draft megakernel 优雅 decline（此前 draft spec 在 `install_megakernel` 时经 `_refuse` 直接停服，导致为某 TP 构建的 draft spec 使其他 TP 全不可用）；`#1f713857` DFLASH worker 绑定自身 round phase。
- **NPU/CPU/其他**：`#50968` DSA-TAIL 支持 PD-transfer 与 decode-retraction；`#51747` hicache 去 AICPU MemFabric 大页 flag；`#7c21d8c7` CPU 上 Qwen3.5-35B-A3B-GPTQ-Int4 量化规则失败与精度退化（见 1.3）；`#563813c0` cookbook 加 Ling-3.2-air；`#52373` AgentX linker options cookbook 渲染修复。

---

## 4. TensorRT-LLM：2026-11-23 提交（41 条）—— KDA 编译键(#22811) + 多模态假死锁(#19898) + TMEM 释放次序(#21851) + GLM FC1 barrier(#21847) + Mamba2 strided(#22596) + DSv4.1 KVCM V2 OPTIONAL(#22352) + MXFP8 FC12 Rubin(#21589)

### 4.1 `#22811` [TRTLLM-16624][fix] KDA replay acceptance kernel 不再按 batch-view 对齐特化（官方一手，fix，强化第 20 条「编译态默认值/性能化风险」）

`_record_kda_replay_acceptance_kernel`（#22009 引入）记录每个 generation 请求被接受的 draft 供下次 KDA replay；其 per-request 输入是整 batch 张量从 `num_contexts` 起的 view。Triton 按指针 16 字节对齐设编译键，这些 view 随 batch 改变：int32 slots/counts 随 `num_contexts%4`、bool dummy mask 随 `num_contexts%16` 变键。warmup 只跑 generation-only batch（`num_contexts==0`），首个混合 prefill/decode batch 在 forward 路径编译 2 个新变体、所有在途请求等待。Kimi K3 TP8 服务 main 上健康后该 kernel 编译 4 次（两节点各 2 变体），其中一次落在 15.3 s token 流空档；另一次 run 在 3.9 s/2.3 s。修法对三个 view 指针加 `do_not_specialize_on_alignment`（`num_requests` 已在 `do_not_specialize`），kernel 不因失去对齐变慢。**研判（观点）**：与 TRT #21666（spec-decode Triton kernel 不再按 batch 特化消除 5–12 s 停顿）同族——「warmup 形状未覆盖线上形状」导致热路径即时编译停顿，是服务延迟长尾的常见来源。

### 4.2 `#19898` [fix] 修多模态调度器下的假 KV 死锁与被忽略的 prefill demotion（官方一手，fix，强化第 8 条）

多模态 encoder item 调度时 executor 的调度器是 `MultimodalScheduler` 包装器；`PyExecutor` 只在调度器有 `set_async_transfer_manager` 时把 `AsyncTransferManager` 交给它，包装器没有→被包装的 `KVCacheV2Scheduler` 拿不到 manager。解聚 context server 上完成的 KV send 在落地前仍占 KV page，V2 死锁检查把该等待误判为 stall 抛**假死锁**。修法给 `RequestScheduler` 加空实现 `set_async_transfer_manager` 并在 `MultimodalScheduler` 转发。同一问题：`PyExecutor` 用 `getattr` 读 `ctx_demote_uncached_tokens`，包装器遮蔽使长 prefill demotion 在多模态下无效，现加 0 默认并转发。4 个新 CPU 测试 main 上全 AttributeError 失败、修后通过。**研判（观点）**：包装器未转发子组件的方法/属性是「默认路径失效」类静默问题——不崩、只是优化/保护悄悄不生效。

### 4.3 `#21851` / `#21847` / `#21589` TMEM / barrier / MXFP8 FC12（官方一手，fix/feat，验证未闭合）

- `#21851` [fix] paged MQA-logits kernel（Blackwell FP4/FP8）在所有 math warp 读完 TMEM 后再 dealloc：加 256 线程 named barrier（FP4 ID 3、FP8 ID 2）+ `fence_view_async_tmem_load()`。coderabbit 提示：FP8 的 ID 2 唯一性未完全确认、任一 math 线程提前退出会挂住 kernel、硬件验证（B200 注入 stall/TMEM 复用）仅在 PR 描述中、当前 head 无完成检查。**按「官方一手但验证未闭合」处理**。
- `#21847` [fix] GLM-5.2 routed MoE FC1 operand barrier 改 cluster scope：relaxed cluster-scope polling + acquire fence，移除 acquire-qualified PTX wait helper。作者报告 `sm_103a` 三种 launch 变体编译、但**该 kernel 未在硬件运行**；QA 判 needs follow-up（新测试为 CPU-only 源码检查）。
- `#21589` [feat] Rubin(SM107) MXFP8 fused FC1+FC2 CuTe DSL MoE leaf：把 routing gather、FC1、SwiGLU、MXFP8 requant、FC2、finalize 融合，新增 Rubin block-scaled MMA helper、MXFP8 权重交错与 partial-refit；locality-domain 重构把 planning/权重生命周期/output-memset 从 `CuteDslFusedMoEBase` 移到 mixin。coderabbit 提示新 Rubin kernel 用 `ld_acquire_gpu` 做 FC1 ready、但两个 Rubin fused FC12 kernel 都**没有 `fence.proxy.async.global`**，FC1→FC2 次序需 follow-up；当前 head CI/build/test 不可用，仅报告早前运行结果。

### 4.4 `#22352` / `#22596` / `#22434` KVCM V2 / Mamba2 / FP4 MLA 重构（官方一手，feat/perf/refactor）

- `#22352` [TRTLLM-16622][feat] DSV4.1 恢复更早 encoder checkpoint 而不缩短前缀复用：给 `KvCacheManagerV2` 加 OPTIONAL endpoints（`optionalReuseMaxBacktrack=0` 默认、`optionalReuseEndpoints=nullopt`、`completeOptionalRecovery=false`），`createKvCache` 默认改为带 optional-match 收集调用 `match`。多处 C++/Python API 签名变化（有默认值，旧调用可编译），coderabbit 提醒确认零 backtrack 路径复用结果与之前一致、Python negative 校验等价。
- `#22596` [perf] Mamba2 decode state update 传 stride x/B/C view（非 MTP 单 token decode 分支在 `_flashinfer_row_views_ok` 时跳过 `.contiguous()`）：作者报告 Nemotron-3-Nano-Omni-30B-A3B NVFP4（TP1、单 GB200）eager decode GPU-op 时间 batch 8 降 6.1–6.5%、batch 32 降 2.2–2.9%（Nsight Systems，共享节点、非端到端）；368/368 与 1173/1173 次 state update 逐位一致（flashinfer 0.7.2/0.6.18）。**注意**：helper 只查行 stride 与行内稠密，未查完整张量的列（内）stride；Dynamo 编译路径仍走 copy；CUDA graph 下逐位性未完全确立。
- `#22434` [refactor] 清理 FP4 MLA 热路径分配、cache layout mixin、graph resolvers、CuTe DSL primitives：多处公开 API 移除/重命名（`decode_kv_regime`/`decode_packed_split`→`decode_graph_variant`、`CUDAGraphRunner.KeyType` 增 `variants`、manager getter 移到 `fp4_mla_layout`），若干 `getattr(...,None)` 改直接读使缺失属性从静默变 `AttributeError`（fail-fast），`check_kernel_argv` 安全检查移除——out-of-tree 调用方需跟进。

### 4.5 其他要点（标题级 / 描述级）

- `#22494` PrimsTS VC-Attention preparation 修复并接收 grouped token range（vendor lock 从 `yuxianq/flashinfer` 换到 `harrisonzhy/flashinfer` fork，vendored `quantization.py` 141 行变更未被评审；`grouped_tokens=None` 保持整序列分组、默认行为不变）；`#22643` 跳过未使用的 full-window DSA schedule；`#21207` opt-in fused GDN prefill conv（默认关）；`#22576` 预编译 multi-row gated RMSNorm 的 batch 依赖变体；`#20972` per-step device copy 改走 model-declared copy plan；`#20971` Gemma3/4 TRTLLM attention 启用 SWA scratch 复用；`#22344` Modeling V2 stated disaggregated deployments；`#22134` 多输入 embedding request all-or-nothing enqueue；`#22646` Qwen2-VL tied lm_head；`#22767` trtllm-bench 上报 engine 解析的 KV dtype；`#20308` VisualGen SymmMemPool 对称内存池。

---

## 5. llama.cpp：2026-11-23 提交（34 条）—— GDN replay 替代 rollback(#31669) + few-row MMQ 零块(#32566) + Metal NVFP4(#32825) + BF16/F32 src1(#32677) + 自动最优量化(#15550) + OpenCL spec verify(#32716)

### 5.1 `#32566` 静默正确性与安全 —— 见第 1.1 节

### 5.2 `#31669` GDN 用 replay 替代 rollback（官方一手，架构/性能，强化第 8/14 条）

移除 GDN（Gated DeltaNet）里的 rollback 机制，改用 replay（理想是各后端协调一次性移除所有 rollback）。TODO 已完成 CUDA/Vulkan/Metal/Hexagon/OpenCL/WebGPU，SYCL 未完成。RTX PRO 6000 + Qwen3.8-27B Q8_0 实测：**recurrent state 内存**随 `n_rs_seq` 3/7/15 从 598.5/1197.0/2394.0 MiB 降至 163.2/176.8/203.9 MiB（省 **73%/85%/91%**），32 序列 `n_rs_seq=7` 时 37.4→5.5 GiB；**速度** DFlash2 投机解码单序列基本持平（−0.1%~+0.5%）、llama-server 4/16/32 slots +2.1/+7.9/+3.8%，batched verify step（27B、8 token/seq、`n_rs_seq=7`）1/2/4/8/16/32 序列 **+0.9/+5.9/+8.6/+17.8/+31.9/+34.5%**。**研判（观点）**：以重算换内存的大幅节省，对长上下文/多 slot 服务的 KV+状态容量是数量级改善；verify step 加速随 batch 增大（replay 省去 rollback 的状态拷贝），与第 14 条「投机验证状态正确性/效率」主线呼应。

### 5.3 `#32825` / `#32677` Metal 量化与精度扩展（官方一手，feat）

- `#32825` Metal 支持 NVFP4（`GGML_TYPE_NVFP4` 此前回落 CPU）：按 MXFP4 路径加 UE4M3 sub-block scale 的 `ue4m3_to_fp32`、`dequantize_nvfp4`/`_t4`、专用 mat-vec kernel 及 `MUL_MAT_ID`/小 batch/few-row MMA 变体、`mul_mm`/`mul_mm_id` 与 tensor-op 变体、`get_rows`/`cpy`、在 `supports_op` 启用类型。AI harness：2×DGX Spark（RDMA、TP=2）、DeepSeek-V4.1-Flash-1017-GGUF；**正文未给逐卡加速数字**，属「新增后端类型支持」。
- `#32677` Metal `MUL_MAT`/`MUL_MAT_ID` 支持 BF16/F32 src1：`GGML_PREC_BF16` 让 MoE down-proj 保留 BF16 激活免 rescale，`GGML_PREC_F32` 让 libmtmd audio log-mel `MUL_MAT` 在 GPU 上跑全 F32；`LLAMA_MOE_AMAX_SKIP` 为临时 workaround（禁 amax trick、经 precision policy 请求 BF16 src1，直到各后端都支持 BF16）。拆 `mul_mm.metal`/`mul_mm_tensor.metal`、引入 `mul_mm_tensor` library kind 与 `_tensor` host-name infix。

### 5.4 `#15550` quantize 自动选择最优量化类型以达目标文件/bpw（官方一手，feat，非推理核心）

新增 `target_bpw_type()`，解约束优化问题在给定文件大小（`--target-size 1.5g`）或全局 bpw（`--target-bpw 4.5678`）预算下最小化量化误差：① 对每个适用 tensor 用 Monte Carlo sampling + imatrix 加权估计误差；② Pareto frontier 丢弃「增大体积但误差降得不够」的类型；③ Lagrangian relaxation 在全模型分配 bit；④ `--state-file` 可断点续算。34 模型 238 测试评估中 **93%** 情况下 KLD 指标达到或优于标准量化。属量化工具链增强，非推理路径。

### 5.5 `#32716` / `#31815` / `#32828` / `#32861` 其他要点（官方一手，perf/fix）

- `#32716` OpenCL 加 q4_k coop-k c8 GEMM（最多 8 token，供 5–8 token 的投机 verify 分派）：gemma-4-E4B-it-Q4_K_M + `--spec-draft-n-max 7` decode **27.158→32.294 t/s**、接受率 40.512%→43.498%（同模型但接受 token 数 190→194，非严格配对 A/B）。
- `#31815` iOS 族 Metal 构建默认命令缓冲数从 1 增至 8（其他 Apple 平台保持 1），意在减少大 prompt evaluation 的长命令缓冲超时；仅 Linux host 预处理/源码检查，**未在 iPhone 实机验证**（贡献者报告的 ubatch 128 规避也未在本 patch 上测试）。
- `#32828` server 在 restore 全成功前不清除 prompt-cache entry（`#25649` 后逐项清理已冗余、且 main restore 成功但 draft restore 失败时会留下「目标字节已丢」的坏状态）；`#32861` ggml-openvino CI 修复 + Muse-Glimmer-30B DFlash/Ling-3.0-tiny 新模型 + 图转换缓存 LRU 8 shapes + Qwen3.5/3.6 GDN NPU 翻译修复等；`#31759` Windows on Snapdragon configure 时拉取 Adreno 二进制 kernel 库并校验 SHA256（失败仅警告、hash 不符报错）。

---

## 6. 趋势研判与累计结论

累计结论继承第 1–24 条（见上一期档案第 6 节），**本期无新增条目**；本期铁证按既有条目归类并强化如下：

1. **静默正确性（第 4/6/9/19 条）持续为最高频主线，且「优化×量化交叉处」成新高频区**：本期五类铁证——① 融合权重量化元数据按 shard 保留（vLLM `#69002` GSM8K 0.000→0.858、TP2 加载失败）；② 零点减法落在舍入后精度域（vLLM `#69085` ROCm int8 0.696→0.732）；③ `exp` 溢出→`inf/inf`→NaN 的 MoE router（SGLang `#51138`）；④ 除零→NaN→整数 UB（llama.cpp `#32566`）；⑤ **并发竞态**（vLLM `#69433` MLA LSE 原地合并，80 用例 7→0 失败）。第 19 条口径建议扩展为：静默错误密度在「常规优化路径」「量化路径」及二者交叉处均高，且覆盖数值/整型/元数据/状态/布局/并发六类。
2. **KV 传输/布局正确性与地址注册（第 8 条）再添铁证**：vLLM `#59586` 揭示 kernel block 与 NIXL block 不一致时可能**静默注册整行、寻址到错误字节**（非 MLA 专属）；TRT `#19898` 包装器未转发 AsyncTransferManager 导致**假 KV 死锁**与长 prefill demotion 失效；TRT `#22352` 用 KVCacheManagerV2 OPTIONAL endpoints 恢复更早 encoder checkpoint 而不牺牲前缀复用。
3. **编译态特化与热路径停顿（第 20 条延伸）**：TRT `#22811`（KDA replay 按 batch-view 对齐特化，健康后编译 4 次、15.3 s 空档）与 `#21207`（opt-in fused GDN conv）、vLLM `#60961`（decode topK 宽度校验）共同说明：warmup 未覆盖的运行时形状/参数会在 forward 路径即时编译或越界，须把「编译键稳定性」纳入服务延迟评审。
4. **默认性能化（第 16 条）与内存回归并存**：vLLM `#69001` Hopper TP4 fused AR+mHC+RMSNorm 默认开 +8.7% E2E；`#69446` 修 DSv4.1 H100 TP1 走 BF16+FP8-GEMV 多占 12.4 GiB/卡的内存回归。性能化默认既是收益来源也是回归来源，升级须同时看吞吐与显存。
5. **KV 状态内存的数量级优化（第 14 条）**：llama.cpp `#31669` GDN replay 替代 rollback，recurrent state 内存省 73–91%（32 序列 37.4→5.5 GiB），verify step 随 batch 增大 +0.9~34.5%；以重算换内存的取舍在多 slot 服务下收益显著。
6. **服务安全/鲁棒性（第 24 条）延伸**：vLLM `#69285` root_path 下健康/扩缩容探测路径误报 503（自动扩缩容器误判风险）、SGLang `#52663` 修 RUSTSEC url 栈溢出、SGLang `#50803` 把「静默失败清理」工程化（244 文件、每修复配失败测试）。
7. **架构性 KV/attention 抽象收敛延续（第 11/17 条）**：vLLM `#66027` DCP ShardQ + CuTe DSL A2A LSE reduce（Kimi K3.1 6/6）、TRT `#22352` KVCacheManagerV2 OPTIONAL endpoints、TRT `#22434` FP4 MLA layout/graph key 收敛——新旗舰模型 CP 与 KV 布局声明继续向类型级治理演进。

---

## 7. 可执行清单（面向工程落地）

1. **量化权重融合审查**：使用 torchao AWQ 且启用 vLLM 融合 qkv/gate-up 或 TP>1 的场景，升级 `#69002`（GSM8K 0.000→0.858、TP2 加载失败）；ROCm 非对称 int8 升级 `#69085`；部分量化 GPTQ checkpoint 的 MTP draft 升级 `#67413`；CPU Qwen3.5-GPTQ-Int4 升级 `#46559`（注意其 CI 未全绿）。
2. **MoE router / 大 logits 回归**：Grok-1 类 softcap router 升级 SGLang `#51138`，并对 `x/c>44` 的大 logits 加 top-k id 与权重的专项测试；不要把 NaN 权重当作「偶发」。
3. **MLA 分块上下文与 KV 传输**：跨 chunk 前缀合并的多请求服务升级 vLLM `#69433`（竞态致错注意力输出）；使用 NIXL + 异构 block size（尤其含 draft group）时升级 `#59586`，否则可能静默传输到错误字节；检查 kernel block 与 NIXL 公共 block 的一致性。
4. **编译态停顿排查**：Kimi K3/KDA replay 服务在健康后出现秒级 token 空档时，检查 `#22811` 的对齐特化编译；启用 fused GDN prefill conv 用 TRT `#21207`（默认关、`TRTLLM_GDN_FUSED_CONV=1`）。
5. **内存回归与默认性能化**：H100/TP1 的 DSv4.1 MXFP8 服务升级 `#69446`（省约 12.4 GiB/卡、decode +约 0.6–1.0%）；Hopper TP4 GLM-5.3-Flash decode 评估 `#69001`（+8.7% E2E，精度无损为作者口径）。
6. **解码/verify 吞吐**：ROCm SWA 模型（Gemma/Mistral/GLM-5.x DSpark draft）长 prompt 评估 vLLM `#63867`（kernel 最高 38.37×、TTFT 1.69–2.21×、bit-identical）；TRITON_MLA decode 评估 `#69029`（H100 1.06–1.71×）；deterministic 模式评估 SGLang `#47410`（gpt-oss-20b 0.33×→0.63×）；多 slot GDN 模型评估 llama.cpp `#31669`（state 省 73–91%、verify +0.9~34.5%）。
7. **KV 卸载运维**：混合模型 partial-tail CPU 命中场景升级 vLLM `#69048`；Mooncake 失败加载用 `#67957`；SGLang PD/linker 场景评估 `#49463`/`#49467`（注意作者明示未单独隔离该改动增益）。
8. **服务可用性边界**：`--root-path` + elastic EP 部署升级 vLLM `#69285`（否则健康探测 503、自动扩缩容误判）；sgl-router 升级 `#52663` 修 RUSTSEC-2026-0592；VLM 多模态服务关注 TRT `#19898`（假死锁/demotion 失效）。
9. **Rubin/SM103 新后端谨慎采用**：TRT `#21589`（MXFP8 fused FC12，FC1→FC2 同步存疑、当前 head CI 不可用）、`#21847`（GLM-5.2 FC1 barrier 未在硬件运行）、`#21851`（TMEM 释放次序、FP8 barrier ID 唯一性未确认）均按「验证未闭合」处理，先在目标硬件做精度/稳定性回归。

---

## 8. 来源、数据质量与限制

- **一手来源（官方仓库/API/文档）**：vLLM 68 / SGLang 25 / TRT-LLM 41 / llama.cpp 34 条提交均经 `api.github.com/repos/<repo>/commits?since=2026-11-23T00:00:00Z&until=2026-11-24T00:00:00Z` 取得（全部 HTTP 200、单页取全无截断）；releases API 确认窗口内无新正式版/无新 prerelease（llama.cpp nightly b12638–b12655 为 prerelease）；48 个关键 PR 正文经官方 issues API 抓取 markdown（配额 60→3，无降级），PR 号均已在窗口内提交标题中二次确认。性能数字如无特别说明均来自 PR 正文（作者/合作者实测）。
- **二手线索/未核验**：HN Algolia 窗口内 `vllm` 29 命中（多为「LLM」模糊匹配）、`llama.cpp` 2、`sglang`/`TensorRT-LLM`/`KV cache`/`speculative decoding`/`paged attention` 均 0、`inference engine` 2。高信号但未核验：124 分「NariQ: BF16-level Qwen3.8-27B in a 10 GB file」（指向 HuggingFace `nari-labs/Qwen3.8-27B-NariQ-GGUF`，未核验量化细节与精度）、2 分「Eliminating Nondeterminism in LLM Inference with Integer-Only Arithmetic」(`nathan.rs/posts/detllm/`)、1 分「In Search of Perfection: aci-cpp vs. llama.cpp v1.0.0」(`actual.inc/blog/in-search-of-perfection`)、2 分「Show HN: Device kernels for gfx900 (Vega) in the radiance inference engine」、1 分「Show HN: Ignis – A browser-native LLM and multimodal inference engine」——均为二手线索/未核验，未作为确定事实采纳。
- **未采集/无可用数据**：Reddit、YouTube、X 本期未采集；窗口内未核验到可确认的官方视频发布。
- **证据等级口径**：官方一手=官方提交/releases/issues API 正文；官方学习资料=vLLM/SGLang 官方文档与 recipe（性能多为第三方 InferenceX run，标「描述级/第三方基准」）；二手线索=HN 故事与社区转述；未核验=社区 API/视频页不可达或无一手确认的内容。
- **描述级/未闭合项清单（本期）**：TRT-LLM `#21851`（FP8 barrier ID 2 唯一性未确认、硬件验证仅 PR 描述、当前 head 无检查）、`#21847`（GLM-5.2 FC1 barrier 未在硬件运行、QA needs follow-up）、`#21589`（FC1→FC2 无 `fence.proxy.async.global`、当前 head CI 不可用）、`#22494`（vendored quantization.py 141 行未评审）、`#21207`（性能数字未在当前 head 验证）、`#22596`（列 stride 未查、CUDA graph 逐位性未确立）、`#22352`（零 backtrack 复用等价性待确认）；SGLang `#46559` Base/AMD CI ✗、`#45196` 精度见正文、`#49463`/`#49467` 增益未单独隔离、`#46559` 新测试未覆盖改动；vLLM `#69001` 融合前精度对照缺失、`#67857` 功率测量无效（无能效声明）、`#60961` MI50/ROCm 与完整 serving 未跑；llama.cpp `#32825` 无逐卡加速数字、`#31815` 未在 iPhone 实机验证、`#32716` 非严格配对 A/B、`#15550` 为量化工具链非推理。以上均按「官方一手但验证未闭合/描述级」处理，不作为确定结论。
