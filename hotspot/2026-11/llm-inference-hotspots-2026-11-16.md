# 大模型推理优化每日热点档案

> 档案日期：2026-11-16（本自动化于该日执行）
> 关注窗口：2026-11-15（周日）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 经 issues API 抓取 markdown 官方一手）+ HN Algolia（HTTPS，numericFilters 按 2026-11-15 epoch 窗口 1794700800–1794787199）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 26 / SGLang 59 / TRT-LLM 8 / llama.cpp 23 合计 **116** 条（均未触 per_page=100 上限，单页取全）；窗口内四家均无新正式版本（llama.cpp 推进 nightly b12419–b12427，prerelease；b12428 落在 11-16T00:12Z 窗口外）；**37 个关键 PR 正文经官方 issues API（markdown 官方一手）抓取**，其中 #65992 首次传输中断后重试成功，无 token 配额下 60/h 上限被用尽；HN Algolia 引擎相关高信号弱（仅 1 分「2-bit ternary KV-cache」与 2 分「Hardware-Agnostic Models in vLLM」，均二手线索/未核验）。本窗口为周日，提交量较上一窗口（11-14：49/60/12/32=153）回落至 116，属周末正常回落，但 SGLang 仍保持 59 条活跃。

---

## 1. 版本状态与本期头号事件：静默正确性铁证密集爆发（SGLang GQA KV head #47802 + PD tail replay #49170 + vLLM K-cache int32 越界 #65926 + Mamba prefix 状态 #67717）+ 量化静默（SGLang SiTU #46392 + vLLM two-sided expert id #67592）+ AITER/FlyDSL 性能（#67377/#65992） + llama.cpp VNNI 3×（#26348）

**提交计数（2026-11-15 UTC，均 HTTP 200；2026-11-15 为周日，量较周六回落；vLLM 26 / SGLang 59 / TRT-LLM 8 / llama.cpp 23，合计 116）**：
- **vLLM 26** / **SGLang 59** / **TensorRT-LLM 8** / **llama.cpp 23**。

**发布状态（窗口内四家均无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-15 全部 26 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；窗口内无新版本。
- **TensorRT-LLM**：dev 线仍为 **v1.4.0rc3（prerelease，2026-11-06T21:38:32Z）**；`releases/latest` 口径下最新稳定版为 v1.2.1（2026-04-20）；11-15 窗口无新版本。
- **llama.cpp**：v1.0.0 于 2026-11-13 正式发布；窗口内推进 **nightly b12419–b12427**（2026-11-15T17:08–23:23Z，prerelease），b12428 落在 11-16T00:12Z（窗口外），无新正式版本。

**本期头号事件（静默正确性铁证密集爆发 × 7 条）**：

1. **SGLang Triton decode GQA 分组读错 KV head（#47802，强化第 4、6 条）**：`glm-4-32b-0414` 在 Triton 后端 decode 时 head 24–31 读成 kv head 0，GSM8K 从 ~93% 掉到 **50.87%（patched 93.48%）**；`--kv-cache-dtype fp8_e5m2` 下 **49.51%→92.87%**。根因 `_fwd_grouped_kernel_stage1` 用 BLOCK_H=16 铺 tile，group=24（48/2）时 tile 1 跨两个 group。因 Triton 也是 Hopper 上 fp8_e5m2 KV 的默认选择，影响面扩大。**CI 未全绿**（Base/Extra/AMD 三测试均 ✗）。这是「注意力 tile 划分静默写错 KV head」的铁证，与量化 KV 路径耦合。

2. **SGLang PD/DP 部署 active decode rank 被 idle tail replay 误路由（#49170，强化第 5、8 条）**：colocated PD + DP attention 下，一步内一个 rank 跑 EXTEND、另一个跑 active DECODE，但都置 `is_extend_in_batch=True`，tail graph 把 active decode rank 送进 `run_idle`。4× GB200 复现：inventory 续写从 `007 | iron | ready` 退化成 `007 reading reading reading …`，JSON key `"id"` 被损坏为 `"id reading reading"`。GSM8K（1319 题）**非思考 67.93%→97.27%、思考 78.70%→96.82%**，且修复后 `\ufffd` 替换字符与 50K token 截断消失。这是 PD 解耦 × DP × eager tail graph 组合的教科书级静默错误。

3. **vLLM DSv4.1 K-cache gather int32 行偏移越界（#65926，强化第 9 条）**：gather kernel 用 int32 算输出行偏移，prefill chunk 超过 2^31 元素后把行写到 workspace 外约 4 GiB。MI355X TP8 + DSpark adaptive verification 启动 **3/3 次 `HSA_STATUS_ERROR_MEMORY_FAULT`**（415–551 s 后）；修复后 2/2 ready、GSM8K 0.920。更隐蔽的是：约 140+ 请求共享 ~30K-token prefix 的 burst 会产生 `(140,30159,512)` chunk，越界行落进 KV-cache 分配区而**输出无可见变化**。根因是 Triton 把 `out_stride0 < 2**31` 特化为 int32。这是第 9 条在 ROCm K-cache 路径的又一实证。

4. **vLLM Mamba `align` prefix cache 发布错误 SSM 状态（#67717，强化第 5、6、8 条）**：从 Mamba block 中途 resume 且跑过 block 末端的 prefill，会让该 block 持有 resume-point state，但 `cache_blocks()` 仍以 full-block hash 发布；下一共享前缀请求全块命中后加载截断 state 生成垃圾。GB300 最小复现（`ibm-granite/granite-4.0-tiny-preview`，block size 832）：正确答案 `pillow`，main 输出 `'access'`（hit 832 stale），修复后 hit 16、输出 `' pillow.'` 正确。属「KV block 状态哈希错位 → 静默错误」，与前缀缓存主线直接相关。

5. **SGLang Kimi-K3 在 SM90/SM120 用 flashinfer_mxfp4 跑错激活函数（#46392，强化第 4、6 条）**：显式 `--moe-runner-backend flashinfer_mxfp4` 在 SM90/SM120 上加载并服务**无任何报错**，但 routed experts 跑的是 GPT-OSS clamped SwiGLU 而非 K3 的 SiTU。H100 NVL 张量级核验：对 SiTU cosine 仅 0.63/0.90，对 GPT-OSS SwiGLU cosine 1.000（bf16 精度吻合）。修复为在参数解析期与层构造期双重拒绝。这是「量化 kernel 功能不匹配但静默服务」的典型。

6. **vLLM FlashInfer NVLink two-sided all2all expert id 哨兵值错误（#67592，强化第 7 条）**：two-sided prepare 把 padding 行标 `invalid_expert_id`，但 manager 设为 `num_experts`，而 `moe_sum`/`apply_expert_map` 只跳过 `<0` 的 id → `--moe-backend triton` two-sided **输出 garbage（GSM8K 0.0/0.0）**。修复后 two-sided+triton GSM8K .878/.886、two-sided+deep_gemm 正常启动 .935/.927。one-sided 路径 GLM-5.3 FP8 on GB300 `deep_gemm` 比默认 TRT-LLM FP8 +2.4%。

7. **跨引擎「非量化常规路径」静默错误密度不亚于量化路径（新增第 19 条）**：本期 7 条铁证中 #47802（GQA tile）、#49170（PD tail replay）、#67717（Mamba block hash）、#65926（K-cache 索引）、#49803（GDN chunk-state 索引）均非量化引入，而是注意力/缓存/路由的索引、分片、状态归属错误。过去静默错误多归因于量化/异构，本期表明**常规优化路径（PD/DP/投机/tail graph/chunked prefill 的正交组合）同样是高发区**。

**性能与能力类要点**：vLLM `#67377`（DSv4.1-Flash dense MXFP8 GEMM 改用 AITER 调优 FlyDSL，prefill 189.8→106.7 ms=1.78×、decode 4544→2822 µs=1.61×、GSM8K 不变）、`#65992`（gfx950 fused mHC seams 改 AITER 单 launch HIP Mega-MHC，per-seam 16–41% 但服务端 A/B 在噪声内）、`#67805`（MI355X 94 GiB Engram 表按 offset 排序，单次 lookup 110 ms→3.9–5.5 ms、AgentX 吞吐 +3.8%/TTFT p90 −17%，但 intvty p90 +20% 需并列看）、`#65658`（PP2 SDMA hand-off，`--pp-sdma-handoff` opt-in，sender GEMM +411%→+1.4%、端到端 +3.6%/TTFT −3.4%）、`#65974`（SM107 flashinfer_moe_ep_cutedsl NVFP4 experts，GSM8K 噪声内、明确无 perf claim）；SGLang mem_cache 统一内存栈 `#50306`–`#50311` 六连（削 host work / 批量 hand-out / 无 host read  relief，#50309 c256 +2.2%、#50306 修首次 c256 冷启动 unified 5200→5630 tok/s）；llama.cpp `#26348`（x86 VNNI Q2_0 dot product，Bonsai 1.7B–27B **3.00×–3.64×**，但 PPL 有 small-but-repeatable 差异）/ `#32296`（Metal LFM3 融合 workaround tg128 +3.3%）/ `#32301`（MiniMax-M3.1，dspark 66% 接受率约 1.5×、PPL 差 <0.3%）；TRT-LLM `#21778`（`MoeConfig.tactic` 按 id pin MoE 实现、不 fallback）/ `#21585`（GLM-5.2 NVFP4 Rubin SM107 高吞吐 target，描述级）。

---

## 2. vLLM：2026-11-15 提交（26 条）—— K-cache int32 越界(#65926) + Mamba prefix 状态哈希(#67717) + two-sided expert id(#67592) + DSv4.1 MXFP8 FlyDSL(#67377) + Mega-MHC(#65992) + Engram 排序(#67805) + PP SDMA(#65658)

### 2.1 `#65926` [Bugfix][DSv4.1] K-cache gather 用 int64 行偏移（int32 越界 4 GiB，fix，强化第 9 条）
- **主题**：DeepSeek-V4.1 的 fp8/fp8_one_pass/MXFP8 K-cache gather kernel 用 int32 算输出行偏移，`batch_idx * out_stride0` 超过 2^31 元素后把行写到 workspace 外约 4 GiB。
- **内容详情**：MI355X TP8 + DSpark adaptive verification 启动 profile 把 511 个 8K 请求塞进一个 chunk `out=(511,8335,512)`，末 7 行越界；base 3/3 `HSA_STATUS_ERROR_MEMORY_FAULT`（415–551 s），修复后 2/2 ready、GSM8K 250 5-shot 0.920、256/256 请求 @c64。内核测试 `test_v41_gather_rows_past_int32_elements` 修复前 3 failed/修复后全过。另记录：约 140+ 请求共享 ~30K prefix 的 burst 产生 `(140,30159,512)` chunk，越界行落入 KV-cache 分配区而输出无可见变化。NVIDIA 无影响（CUDA 走 CuteDSL gather）。
- **应用场景**：1M 上下文 DeepSeek-V4.1 在 ROCm MI355X 的 prefill gather（含 DSpark adaptive verification 与前缀缓存 burst）。
- **研判**：【事实】PR 给出根因（Triton 特化 `out_stride0 < 2**31` 为 int32）、3/3→2/2 崩溃对照、内核测试与「输出无变化但写越界」的记录。【观点】属「长上下文 int32 溢出静默错误」（第 9 条）在 ROCm K-cache 路径的新实例；记为官方一手（fix 级，含根因与复现）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：vllm-project/vllm #65926（合并于 2026-11-15T15:27:43Z）。

### 2.2 `#67717` [Bugfix][Mamba] 中途 resume 跑过 block 末端的 block 不再发布（prefix 状态错位，fix，强化第 5、6、8 条）
- **主题**：修复 Mamba `align` 模式 prefix caching 服务错误 SSM state——从 block 中途 resume 且跑过 block 末端的 prefill 使 block 持有 resume-point state，却仍以 full-block hash 发布，下一共享请求命中后加载截断 state 生成垃圾。
- **内容详情**：`MambaManager` 记录每个 chunk 的范围，`cache_blocks` 中 `_record_mid_block_states` 标记持有非 block-end state 的 block，经新增 hook `_refine_block_mask` 从 full-block hash 中剔除。最小复现（`granite-4.0-tiny-preview`，mamba_block_size=832，GB300）：main 以 hit 832 输出 `'access'`（正确答案 `pillow`），修复后 hit 16 输出正确；单元测试 3/3 main 失败、修复后通过。
- **应用场景**：vLLM Mamba/混合线性注意力模型开启 prefix caching 且 `mamba_cache_mode="align"` 的部署。
- **研判**：【事实】PR 给出 block 0 状态错位（16 vs 832）的逐步推演、GSM8K 类复现与 3/3 单测。【观点】属「前缀缓存 × SSM 状态归属」静默错误（第 5/6/8 条交叉债务）；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：vllm-project/vllm #67717（合并于 2026-11-15T22:59:35Z）。

### 2.3 `#67592` [Perf] 允许 DeepGEMM/Triton block-FP8 MoE 配 FlashInfer NVLink all2all（two-sided expert id 修复，perf/fix，强化第 7 条）
- **主题**：`--moe-backend deep_gemm` 映射的 `TritonOrDeepGemmExperts` 此前拒绝 FlashInfer NVLink one-/two-sided all2all；本 PR 放开并修复 two-sided expert-id 错误。
- **内容详情**：根因是 two-sided prepare 用 `invalid_expert_id` 标记 padding 槽，manager 设为 `num_experts`，但 `moe_sum`/`apply_expert_map` 只跳过 `<0` → two-sided+triton 输出 garbage（GSM8K 0.0/0.0）。修复后 two-sided+triton .878/.886、two-sided+deep_gemm 正常启动 .935/.927（709 tok/s/GPU，单节点 g32）。性能：GLM-5.3 FP8 on GB300 DEP8 one-sided，`deep_gemm` 852.4/856.7 vs 默认 TRT-LLM FP8 832.2/836.9（+2.4%）。原始 IMA 未复现，其他模型/量化未覆盖。
- **应用场景**：Blackwell/GB300 上 DeepSeek/GLM 系 FP8 MoE 配 FlashInfer NVLink all2all 的部署。
- **研判**：【事实】PR 给 two-sided 0.0→.878/.935 对照、one-sided +2.4% 与明确的根因（哨兵值 `<0` 语义）。【观点】属「MoE 路由 / expert id 归属静默错误」（第 7 条）与 MegaMoE 后端扩展（第 12 条）；记为官方一手（perf/fix 级，含对照）。**证据等级**官方一手（perf/fix 级，含对照）。
- **来源**：vllm-project/vllm #67592（合并于 2026-11-15T11:58:19Z）。

### 2.4 `#67377` [Perf][ROCm][DSv4.1] dense MXFP8 GEMM 用 AITER 调优 FlyDSL kernel（perf，强化第 11/13 条）
- **主题**：8× MI355X 上 DeepSeek-V4.1-Flash（DP8 attention + EP8 MoE，TP1 dense linear）的 32×32 block-scaled MXFP8 linear 从无 TP1 调优的 Triton block32 GEMM 改用 AITER 调优的 FlyDSL kernel。
- **内容详情**：离线 dense GEMM（按 engine step 调用加权）prefill **189.8→106.7 ms/step（1.78×）**、decode **4544→2822 µs/step（1.61×）**；MXFP8 activation quant 357→172 µs @M=32768,K=8192；权重原地预 shuffle、显存仍 52.1 GiB/GPU。端到端（InferenceX aiperf，CONC=256，DSpark 5 token）26,714 tok/s/GPU、P90 4.61 tok/s/user（与上一修订在噪声内）；GSM8K 0.9735→0.9727（strict，±0.0045）。FlyDSL 用 32-bit offset，行按 2 GiB 分块。
- **应用场景**：ROCm gfx950（MI355X）上 DeepSeek-V4.1-Flash 的 dense linear 与激活量化加速。
- **研判**：【事实】PR 给出逐 GEMM 的 prefill/decode 对照表、端到端与 GSM8K。【观点】属「异构后端性能优化」（第 11 条）与 MXFP8 栈（第 13 条）落地；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #67377（合并于 2026-11-15T01:46:26Z）。

### 2.5 `#65992` [ROCm][DSv4.1][Perf] fused mHC seams 改 AITER 单 launch HIP Mega-MHC（perf，强化第 11 条）
- **主题**：gfx950 上 DeepSeek-V4.1 的 fused delayed mHC seams 从 AITER 两 launch Triton seam 改为单 launch HIP Mega-MHC（AITER v0.1.27 起、main 已 pin v0.1.29）。
- **内容详情**：kernel 级每 seam 节省 **16–41%**（MI350X layer-30 weights，HIP-graph replay）；关键路径上约占 C256 decode step 的 1%、4096-token prefill step 的 1.6–1.9%。服务端 A/B（n=3/arm，decode C1–C256 与 prefill-heavy）**均在 run-to-run 噪声内**——与节省量级预测一致。
- **应用场景**：ROCm MI355X/MI350X 上 DeepSeek-V4.1 的 mHC seam 融合。
- **研判**：【事实】PR 给出 per-seam 微基准与「服务端噪声内」的诚实结论。【观点】属「异构后端性能优化」（第 11 条），提示微基准提升未必转化为服务级收益；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #65992（合并于 2026-11-15T12:56:28Z；首次抓取传输中断后重试成功）。

### 2.6 `#67805` [ROCm][Perf][DSv4.1] Engram host lookup 按 table offset 排序（perf，强化第 11 条）
- **主题**：MI355X 上 DP 共享 Engram 表（94 GiB `/dev/shm`）的 prefill 每 Engram layer 花 36–110 ms 在 `_engram_lookup_kernel`，启用按 table offset 排序路径。
- **内容详情**：微基准（94 GiB 表，589,824 行）unsorted 54 ms → sorted 5.1 ms（8 GPU）；DEP8 prefill profile 单次 >1 ms lookup 从 110.1/109.9… ms → 5.5/5.1/3.9/3.9 ms，窗口内每 rank 合计 ~586 ms→27–37 ms。AgentX c256 1 小时对照（单次运行，baseline commit 略旧）：吞吐/GPU 175,582→182,270（+3.8%）、TTFT p50 2.06→1.85 s（−10%）、p90 4.41→3.68 s（−17%），但 **intvty p90 96.4→115.6（+20%）**、p50 +6.7%。GSM8K 0.9757 vs baseline 0.9727（输出预期不变，仅行访问顺序变化）。
- **应用场景**：ROCm MI355X 上 host-offloaded DP 共享大 Engram 表的 prefill lookup。
- **研判**：【事实】PR 给出微基准/profile/AgentX 全表与「TTFT 改善但 interactivity p90 变差」的并列数据。【观点】属「异构性能优化 + 指标权衡」（第 11 条），单指标改善不等于体验改善；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #67805（合并于 2026-11-15T23:44:04Z）。

### 2.7 `#65658` [ROCm][PP] SDMA copy engine 做 stage hand-off（`--pp-sdma-handoff`，perf，强化第 11 条）
- **主题**：ROCm PP 的 stage 间 peer copy 以 blit kernel 占发送端 CU；新增 opt-in `--pp-sdma-handoff` 用 SDMA copy engine 搬运（gfx950、PP2×TP1、eager、同节点）。
- **内容详情**：接收端分配 3 slot 经 CUDA IPC 共享，发送线程在双 SDMA engine 上拷贝，`/dev/shm` 双计数器同步。发送端与之并行的 GEMM：torch peer copy **+411%**、本 PR SDMA **+1.4%**（64.5 GB/s）。端到端 MiniMax-M3.1-Flash TP1×PP2 prefill（16384-token steps）吞吐 3.18→3.29 req/s（+3.6%）、TTFT mean 2427→2345 ms（−3.4%）、p99 −3.7%，GSM8K 193/200 不变。默认关，无 flag 无变化。
- **应用场景**：ROCm MI355X PP2 eager 部署的 stage 间张量搬运去 CU 占用。
- **研判**：【事实】PR 给出 copier 对照表（含 `hipMemcpyDeviceToDeviceNoCU` 静默跳过 copy 的反例）、端到端 4 轮一致增益与 GSM8K。【观点】属「ROCm 通信/计算重叠」（第 11 条）；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #65658（合并于 2026-11-15T07:55:59Z）。

### 2.8 `#65974` [MoE][NVIDIA] flashinfer_moe_ep_cutedsl 支持 SM107 的 NVFP4 experts（feature，强化第 12 条）
- **主题**：让 `flashinfer_moe_ep_cutedsl` MoE 后端在 SM107（Rubin，compute capability 10.7）上跑 NVFP4 experts（FlashInfer ≥0.7.3 的 `Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig`）。
- **内容详情**：4× SM107 单节点 DP4+EP 端到端验证 DeepSeek-V4.1-Flash NVFP4（ModelOpt checkpoint），GSM8K 0.9007 vs 默认后端 0.9067（噪声内）。旧版 FlashInfer（含 pin 的 0.7.2.post1）以明确错误拒绝；SM107 上拒绝 MXFP4 experts / `flashinfer_moe_ep_mega_deep_gemm` / EPLB / `reload_weights`；SM100/SM103 行为不变。**明确无 perf claim**（serving benchmark 仅 informational）；CuTe DSL 打印 `TYPE_INT_LITERAL_OUT_OF_RANGE` 警告指向 FlashInfer 自身。
- **应用场景**：NVIDIA Rubin（SM107）上 NVFP4 MoE 专家的 FlashInfer CuTe DSL 后端。
- **研判**：【观点】属「DeepGEMM/MegaMoE 量化后端成为旗舰 MoE 主路径」（第 12 条）在 SM107 的扩面；明确无性能数字，记为官方一手（feature 级，描述级 perf）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #65974（合并于 2026-11-15T22:46:05Z）。

### 2.9 其他要点（标题级 / 描述级 / 性能级）
- `#62807` [XPU][LoRA] 支持非量化 MoE LoRA（feature 级，强化第 15 条 XPU）。
- `#67469` [Bugfix] 把 warmup 与 CUDA graph 尺寸封顶到 `max_num_seqs * max_model_len`（fix 级）。
- `#66842` [ROCm][DeepSeek-V4.1] 为 DeepSeek-V4.1-Flash 启用 eager prefill context parallel（feature 级，强化第 11 条 CP，正文抓取为空）。
- `#63976` [Fast Start] 支持 compressed-tensors CUTLASS block FP8 缓存重启（IPC weight-cache，B300 16 passed + 真实模型 118 s，feature 级）。
- `#65534` [Core][MM] 多模态 encoder cache 尺寸可配置（feature 级）。
- `#63524` [Bugfix][KV Offload] fs tier store job 只上报实际写入块（transfer_bytes 计 0 避免喂给 BackpressureDetector EMA，fix 级，588 passed，第 8 条）。
- `#66494` [XPU] 在 XPU MXFP8 linear kernel 展平 N-D 输入（fix 级）；`#64255` 前一期已注册 XPU MLA 后端。
- `#67691` [Bugfix][Muse Spark] 延迟加载 model class 避免 import 时初始化 CUDA（fix 级）。
- `#4388bec`/`#67611` [Bugfix][Audio] 零 decode limit 视为 disabled（fix 级）。
- `#67607` [KV Offload][Bugfix] P2P e2e 测试端口保留到 tier 绑定（test 级）；`#67742` 要求 batch invariance 用于 derender parity 测试（CI 级）；`#67391`/`#67574`/`#67716`/`#67752`/`#67753` 均 CI/Test 级。
- `#66878`/`#67670` [Rust Frontend] `/inference/v1/generate` 支持 n>1、bump llm-multimodal（第 17 条，feature/chore 级）。

---

## 3. SGLang：2026-11-15 提交（59 条）—— GQA KV head 静默(#47802) + PD tail replay 静默(#49170) + SiTU 静默(#46392) + GDN int64 溢出(#49803) + mem_cache 统一内存六连(#50306–50311) + AMD dsa/RCCL(#33506/#49955) + KV dtype 插件(#49717)

### 3.1 `#47802` [Fix] Triton decode 对 GQA group>16 且非 16 整除读错 KV head（静默正确性铁证，强化第 4、6 条）
- **主题**：`glm-4-32b-0414` 在 Triton 后端 decode 时 head 24–31 读成 kv head 0；Triton 也是 Hopper 上 fp8_e5m2 KV cache 的默认选择。
- **内容详情**：`_fwd_grouped_kernel_stage1` 以 BLOCK_H=16 铺 tile（`cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)`），但 tile 按 query head 连续铺，group=24（48/2）时 tile 1 跨两个 group，head 24–31 读 kv head 0。修复让每个 KV head 获得整数个 16-head tile，非整除时编译期分支按 group 起点索引。GSM8K（1319 题，H100）`triton` **50.87%→93.48%**、`fp8_e5m2` **49.51%→92.87%**，与 fa3 93.48% 对齐；无损 shape 位精确、48/2 kernel 因 4 tile 慢 ~1.35×（旧快是因结果错），端到端 decode +1.0%@b32。CI Base/Extra/AMD 三测试均 ✗。
- **应用场景**：GQA group 非 16 整除的模型（GLM-4-32B）在 Triton / fp8_e5m2 KV 下的 decode。
- **研判**：【事实】PR 给出 tile 划分根因、GSM8K 三后端对照表、repro gist 与「旧快是因结果错」的诚实说明。【观点】属「KV head/tile 索引静默写错」（第 4/6 条，且非量化路径，强化第 19 条）；CI 未全绿须复核。**证据等级**官方一手（fix 级，含根因与对照；CI 未全绿）。
- **来源**：sgl-project/sglang #47802（合并于 2026-11-15T07:06:04Z）。

### 3.2 `#49170` [Fix] active DP decode rank 不被 idle tail replay 吞掉（PD 静默正确性铁证，强化第 5、8 条）
- **主题**：colocated PD + DP attention 部署中，一步内一个 rank 跑 EXTEND、另一个跑 active DECODE，两者都置 `is_extend_in_batch=True`，tail graph 把 active decode rank 送进 idle replay。
- **内容详情**：在 4× GB200（DP4/EP4、flashmla_mega、无投机）复现：inventory 续写从 `007 | iron | ready` 退化为 `007 reading reading reading …`，JSON key `"id"` 被覆写为 `"id reading …"`。基线 158 个混合步中 158 个 active decode rank 被误送 `run_idle`；修复后 0。GSM8K（1319 题）非思考 **67.93%→97.27%**、思考 **78.70%→96.82%**，基线 `\ufffd` 替换字符（73 非思考/9 思考）与 50K token 截断消失。修复把 tail-graph 资格标志搭载到已有 host collective 的行计数中，要求所有 rank 合格才 replay。基于 v0.5.23 构建实测，upstream main 未跑 GPU。
- **应用场景**：SGLang PD 混合部署 + DP attention + eager tail graph 的 DeepSeek-V4.1-Flash。
- **研判**：【事实】PR 给出 mixed-step 计数、内容损坏实例、GSM8K 双模式对照与 `\ufffd`/截断消失。【观点】属「PD 解耦 × DP × tail graph 交叉静默错误」（第 5/8 条，强化第 19 条）；upstream main 未实测须标注。**证据等级**官方一手（fix 级，含根因与对照；非 upstream main 实测）。
- **来源**：sgl-project/sglang #49170（合并于 2026-11-15T03:39:10Z）。

### 3.3 `#46392` [Fix] Kimi-K3 在 SM90/SM120 拒绝 flashinfer_mxfp4 SiTU experts（量化静默，强化第 4、6 条）
- **主题**：修复 #45755——SM90/SM120 上显式 `--moe-runner-backend flashinfer_mxfp4` 加载并服务 Kimi-K3 **无任何报错**，但 routed experts 跑 GPT-OSS clamped SwiGLU 而非 K3 的 SiTU。
- **内容详情**：参数解析期（`kimi_k3.py`）与层构造期（`mxfp4.py` 的 `create_moe_runner`）双重拒绝。H100 NVL 张量级核验（8 experts/hidden 512/intermediate 256/top-2）：对 SiTU `SituAndMul(4.0,25.0)` cosine 0.63/0.93/0.90（三档 scale），对 GPT-OSS SwiGLU cosine 1.000（bf16 精度），确认 SM90 路径匹配 SwiGLU 而非 SiTU。SM100 的 `trtllm_sm100` 实现 SiTU、不受影响；把 SiTU 透传进 CUTLASS kernel 待 FlashInfer #4209。
- **应用场景**：Kimi-K3 MXFP4 检查点在 SM90/SM120（H100/RTX 6000 等）的服务。
- **研判**：【事实】PR 给出公式级张量对照（cosine 0.63 vs 1.000）与「无报错静默服务」的根因。【观点】属「量化 kernel 功能不匹配静默错误」（第 4/6 条）；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：sgl-project/sglang #46392（合并于 2026-11-15T09:35:34Z）。

### 3.4 `#49803` [Fix] GDN chunk-state offset 乘前拓宽 int64（int32 溢出静默，强化第 9 条）
- **主题**：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（及 XPU copy）用 int32 做 `(boh*H+i_h)*V*K` 乘法、cast 在回绕后，chunk-state slot 达到 4096 时 `h` 偏移过 2^31 变负写到 buffer 前，8192 时回绕到 0 静默覆写首个序列的 chunk state。
- **内容详情**：H=32/K=V=128 时每 slot 2^19 元素，约 262k token 前置（H=64 时 131k）即触发；只影响 packed batch 中非首序列。修复把 `bos`/`boh` 先转 int64；新增测试把两相同序列放 slot 0 与 8192 比对（8 GiB buffer 填 NaN 防未写 slot 蒙混）。MI300X/MI355X：修复前新测试失败、修复后 31 pass/1 skip。其他 FLA prefill kernel（`chunk_o.py` 等）仍用 int32 `cu_seqlens`，阈值约 2×，待 follow-up。CI Base ✓、Extra/AMD ✗。
- **应用场景**：大/disabled `chunked_prefill_size` 下 Qwen3.5 等 GDN/线性注意力混合模型的长 packed prefill。
- **研判**：【事实】PR 给出回绕阈值、slot 8192 覆写机制、两卡测试对照与未覆盖 kernel 清单。【观点】属「长上下文 int32 溢出静默错误」（第 9 条）在 GDN chunk-state 的实例；CI 未全绿须复核。**证据等级**官方一手（fix 级，含根因；CI 未全绿）。
- **来源**：sgl-project/sglang #49803（合并于 2026-11-15T03:39:29Z）。

### 3.5 `#50306`–`#50311` mem_cache 统一内存六连（perf，强化第 8/11 条）
- **主题**：同一 6-PR 栈在 12:09–12:12 UTC 的约 3 分钟内整体落地，围绕 `--enable-unified-memory` 削减 scheduler/host 侧开销：`#50311` 削减 admission host work（demand/可行性答案按 pass 缓存复用，GSM8K 0.87–0.875）、`#50310` 批量 hand-out state slot/page（8 请求 prefill 从绑定 9 次→1 次，GSM8K 0.873）、`#50309` relief 无 host read（设备端 pack ends，c256 +2.2%/c16 +0.2%/shared prefix +0.6%）、`#50308` word-wide 清/拷状态、`#50307` GDN prefill 原地跑在 unified state pool（bit-identical，int32 track slot 一次转 int64）、`#50306` 消除 GDN prefill host stall + FlashInfer 四变体 warmup。
- **内容详情**：`#50306` 首次 c256 run（启动后）default pool 5391/5351/5372→5860/5863/5878 tok/s、unified 5200/5172/5195→5630/5630/5596，稳态 default pool c256 5821.4→5857.4；但 steady-state 对照表中 `#50306` c256 −0.1%/c16 −0.4%（在噪声内）。所有 arm GSM8K 0.873–0.877 不变。
- **应用场景**：SGLang 统一内存池下的调度 admission、state/page 分配、GDN prefill 与 relief。
- **研判**：【事实】PR 栈给出逐 PR 的 tok/s 增量表、bit-identical 核验与 GSM8K 不变。【观点】属「统一内存/调度 host 开销」（第 8/11 条）；注意增益主要来自冷启动与特定负载，稳态接近噪声，记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：sgl-project/sglang #50306–#50311（合并于 2026-11-15T12:09–12:12Z）。

### 3.6 `#33506` [Fix][AMD] DSA aiter backend 修复 Triton cache flooding（feature/fix）
- **主题**：DSA 模型（DeepSeek-V3.2、GLM-5.x FP8）用 `--dsa-*-backend aiter` 时，`/tmp/torchinductor_root/triton/` 无界增长（benchmark sweep 中写入数万 kernel 目录），拖慢容器文件系统。
- **内容详情**：根因 `_get_valid_kv_indices_kernel` 把 batch size 声明为 `tl.constexpr`，每个 batch size 触发一次编译。tilelang 后端与非 DSA 模型不复现。
- **应用场景**：AMD aiter 后端服务 DSA 稀疏注意力模型的长时服务/压测。
- **研判**：【观点】属「异构后端工程健壮性」（第 11/15 条），非数值错误但影响长期服务稳定性；记为官方一手（fix 级，描述级：未给编译数抑制后的定量）。**证据等级**官方一手（fix 级，描述级）。
- **来源**：sgl-project/sglang #33506（合并于 2026-11-15T23:26:03Z）。

### 3.7 `#49955` [Feat][AMD] gfx1250 新增 ring all-reduce 与非展开 2shot_push（feature，强化第 15 条）
- **主题**：gfx1250 上 carv2 的 2shot_push 大张量两过 fabric 且每 scoped store 后等待，超过数 MB 输给 RCCL；新增 `ring_simple` ring all-reduce（每 CU 一 channel、2 MB slice、双 slot FIFO + 64-bit ready/drained 计数）与 unrolled 2shot_push（warp 先发整 hunk load 再使用）。
- **内容详情**：全部 push kernel/ring 走 uncached symmetric memory，用 `ld_relaxed_16B`（device scope，避免 CU cache 陈旧）与 `st_relaxed_16B`/新 `st_relaxed_run_16B`（8× `global_store_b128` 后一次 `s_wait_xcnt`）；gfx1250 TP2–8 尺寸表以 1shot/2shot/ring 覆盖到 4 GB、无 RCCL 窗口。默认表（`HIP_FORCE_DEV_KERNARG!=1`）TP8 分界 16 MB，显式 `=1` 表 32 MB。`=0`/`=1` 表在三个 8× gfx1250 节点的服务中验证（正文给出节点/表对应）。
- **应用场景**：AMD MI350/MI355 系（gfx1250）TP2–8 的大 all-reduce（prefill chunk、大 MoE 通信）。
- **研判**：【观点】属「跨后端可移植性/异构通信」（第 15 条）的 AMD 原生通信栈深化；正文以设计与验证节点描述为主，未给端到端吞吐数字，记为官方一手（feature 级，描述级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #49955（合并于 2026-11-15T18:27:45Z）。

### 3.8 其他要点（标题级 / 描述级 / 性能级）
- `#49717` [Feat] 允许 platform plugin 注册 KV cache dtype（feature 级，正文抓取为空，第 13 条相关）。
- `#50122` [Feat] `mem_cache` opt-in 纯 state、cost-based 淘汰顺序（`--radix-eviction-policy state_cost`，feature 级）。
- `#40893` [Fix] 投机 rejection sampling 支持缩减 draft vocabulary（fix 级，第 14 条）。
- `#49089` [Feat] glm5.3flash 允许 PD prefill CP 配 native MTP 或不用投机（feature 级，第 11/14 条）。
- `#50559` [Refactor] 单一表声明每个 HTTP route 的 auth level，Python/Rust server 共读；行为变更：Rust server 无 admin key 时从 400→403 与 Python 对齐（第 18 条延续，538 lib tests + clippy）。
- `#49717`/`#50586`/`#47060`/`#50044` KV dtype 插件 / auth 方法按路由校验 + `/internal/peer` allowlist / STANDALONE metrics 快照 / gRPC instance identity（均 feature/refactor/fix 级）。
- `#47802` 之外的 attention 修复：`#50197` 消除 GDN MIS state reset 的 host sync、`#9a0e6ea` PD poll/grammar PP-sync 按 pipeline depth 而非单轮限定（perf 级）。
- `#33506`/`#46392`/`#49170`/`#49803`/`#47802` 五条静默/溢出修复构成本期 SGLang 正确性主线；`#38751` causal Conv1D out-of-tree dispatch、`#46412` MM refactor 6/9 token-space processor、`#50358`/`#50357`/`#50184`/`#50182`/`#50530`/`#50441` 等 diffusion 修复多为描述级（无性能数字）。
- CI/Rerun 基建 `#50603`/`#50604`/`#50606`/`#50592`/`#50593`/`#50584`/`#50580`/`#50562`/`#50560`/`#50550` 等均 CI 级；`#49759` tokenizer 入口校验 `token_indices_to_pool`、`#50234` 拒绝不匹配 manifest、`#50586` auth 路由校验均 fix 级。

---

## 4. TensorRT-LLM：2026-11-15 提交（8 条）—— GLM-5.2 NVFP4 Rubin target(#21585) + MoE tactic pin(#21778) + Relu2 NVFP4 解 waive(#22104) + CI 基建为主

本窗口 TRT-LLM 仅 8 条（其中 4 条 infra/test），无新版本、无重大性能数字。

- **`#21585` [feat] StairCase: GLM-5.2 NVFP4 Rubin 高吞吐 target（强化第 11/12 条）**：为 `glm_moe_dsa` 增加 `glm_5_2_nvfp4__sm_107__high_throughput` target（modeling/weights/MLA/indexer/DSA/MTP + target CUDA kernels + fake ops），含 MLA Q-length 量化 grid opt-in、per-call KV split 限制、 honoring target NVFP4 KV capability。**正文为目录结构，无性能数字，描述级**。来源：NVIDIA/TensorRT-LLM #21585（2026-11-15T04:20:09Z）。
- **`#21778` [feat] 新增 `MoeConfig.tactic` 按 id pin MoE 实现（强化第 12 条）**：`moe_config.backend` 命名 family 且可能 fallback 到 CUTLASS；新 `tactic` 按 canonical id `provider.technique.kernel_name.quant`（或其点分子集）pin 具体实现且**从不 fallback**——未知/无匹配的 tactic 在模型加载前失败并列出原因。配置解析期与每层构建期均校验，draft/MTP 用 `speculative_config.moe_tactic`。来源：NVIDIA/TensorRT-LLM #21778（2026-11-15T10:18:13Z）。
- **`#22104` [test] Unwaive Relu2 NVFP4 MoE backend 测试（量化正确性验证）**：移除 nvbugs 5989912 对 Relu2/bfloat16/`e60_k4_h2048_i1408`/seq=8/Renormalize 的 waive。B200 本地 1 passed；B300 早前 20/20 新 pytest 进程 × 188 tactics（3,760 检查）全过；L0 #69913 通过。正文明确「keep 5989912 open until post-merge stability verified」。来源：NVIDIA/TensorRT-LLM #22104（2026-11-15T23:44:58Z）。
- **其余（infra/test 级）**：`#22307`（拆分 GB300 与 DGX B200 pre-merge 测试）、`#22244`（Rocky Linux 8 CI 镜像禁用 UBI repos）、`#22055` [fix]（稳定 MPI 启动/readiness/CI）、`#22124`（预构建 x86 C++ 测试、合并 GPU CI 阶段）、`4737591`（nightly lock file 检入）。均 CI/infra 级，非引擎功能变更。

---

## 5. llama.cpp：2026-11-15 提交（23 条）—— x86 VNNI Q2_0 3×(#26348) + MiniMax-M3.1(#32301) + DFlash2 Ripple 元数据(#32363) + Metal LFM3 融合(#32296) + ROCm sync builtins(#31617)

### 5.1 `#26348` ggml-cpu x86 VNNI Q2_0 dot product（perf，强化第 15 条）
- **主题**：为 AVX-VNNI / AVX-512 VNNI+VL 构建新增 `ggml_vec_dot_q2_0_q8_0` 优化实现。
- **内容详情**：`test-quantize-fns` 通过、14,000 次随机比对一致。AMD EPYC 9645（8 核，纯 CPU `-t 8 -ngl 0 -fa off`）Bonsai Q2_0 GGUF：pp/tg 加速 **3.00×–3.64×**（1.7B pp512 14.07→50.47 tok/s=3.59×、tg128 10.22→33.28=3.26×；27B pp128 0.793→2.846=3.59×、tg32 0.716→2.374=3.32×）。作者诚实记录：算子隔离等价，但 PPL 有 small-but-repeatable 差异（mean KLD 差 `0.0003…`，正文截断）。
- **应用场景**：x86 VNNI 兼容 CPU 上的纯 CPU / 混合推理 group-64 Q2_0 量化模型。
- **研判**：【事实】PR 给出逐模型 pp/tg 对照表与 PPL 差异自述。【观点】属「跨后端 CPU 优化」（第 15 条），加速显著但需注意 PPL 微差；记为官方一手（perf 级，含对照；KLD 值正文截断）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：ggml-org/llama.cpp #26348（合并于 2026-11-15T15:19:33Z）。

### 5.2 `#32301` model: support MiniMax-M3.1（feature，强化第 11 条）
- **主题**：独立于 #31192 的 MiniMax-M3.1 实现（将最终取代之），强调充分验证并精简冗余代码。
- **内容详情**：视觉对 SGLang 3D RoPE 相对 L2 0.24%、无需 resize 图 per-token cosine ≥0.999996，需 resize 时两方 38×58 grid 一致、cosine ≥0.9992（resample 差异）。dspark 代码 66% 接受率、约 1.5×（54 vs 36 tok/s）、散文 42%。8× H200 真实模型 PPL SGLang 401.10 vs llama.cpp 399.75（差 <0.3%）。
- **应用场景**：llama.cpp 上 MiniMax-M3.1（含视觉与 DFlash/DSpark 投机）的部署。
- **研判**：【事实】PR 给出与 SGLang 的逐项数值对齐（L2/cosine/PPL）与接受率/速度。【观点】属「新旗舰模型多引擎落地」（第 11 条）；记为官方一手（feature 级，含对照）。**证据等级**官方一手（feature 级，含对照）。
- **来源**：ggml-org/llama.cpp #32301（合并于 2026-11-15T16:39:59Z）。

### 5.3 `#32363` convert: 修复 Ripple DFlash2 NoPE 元数据（fix）
- **主题**：修正 #31904（误按 granite 格式），依据 Ripple-65B-Preview-DFlash2 的 config。
- **内容详情**：转换后模型在 Strix Halo 真机 server 验证，draft acceptance 0.27（378/1400，mean len 2.89），acc per pos 从 0.720 递减到 0.015。作者注明低接受率或因测试题太简单、模型在其硬件上太慢。
- **应用场景**：Ripple-65B-Preview-DFlash2 的 GGUF 转换与 DFlash2 投机解码。
- **研判**：【观点】属「投机解码模型适配」（第 14 条），修元数据正确性；接受率是单一简单任务实测、不具一般性，记为官方一手（fix 级，描述级）。**证据等级**官方一手（fix 级，描述级）。
- **来源**：ggml-org/llama.cpp #32363（合并于 2026-11-15T23:47:27Z）。

### 5.4 `#32296` / `#31617` / 其他要点
- **`#32296`** [model][Metal] LFM3 融合 workaround：RMS_NORM+MUL+ADD 融合要求 chain 在 src[0]、对 LFM3 从未触发，修复后 M5 Max 上 LFM3-1.7B-QAD-Q4_0 tg128 293.3→302.9 t/s（+3.3%）。官方一手（perf 级，含对照）。
- **`#31617`** [CUDA/HIP] 重构 `_sync` warp primitives：ROCm 要求提到 6.3，用 ROCm 原生 builtin 修缺失的 sync，使 mask 生效（此前被忽略）、wave64 上强制 64-bit mask；无硬件支持变化。官方一手（fix 级）。
- **`#32349`** 从 tensor 读取 qwen3vl output head size（fix 级，正文仅标题）；`#1719dbb/#32359` server router 子节点跳过 MCP discovery、`#56fd8a3/#32093` 忽略空 stop string、`#4e5116f/#32321` 修 `/v1/messages` 的 `output_config.effort` 丢失、`#58a4a6b/#30367` 全局 LoRA 配置变更清 slot cache（均 server fix 级）。
- **`#429460d/#32154`** server 重构新增 `server_slot_sched` 与 prompt running hash；`#fc504f9/#32329` mtmd 支持 higgs v3 STT 与 vibevoice-ASR；`#7ae3c6b/#32020` 新增 zonos3-mini TTS；`#de32b26/#31904` Recursive Ripple-65B-Preview 支持 DFLASH2；`#18b74fa/#32349` 与 `#dd099a6/#32294`（inkling 2 图像 token 边界）均 model 级。
- **`#6d3568e/#30949`** OpenCL 新增 A7 Q6_K 非 MoE binary kernel；`#900f01b/#32331` CI/HIP 更新 fattn symbol names；`#278a89c/#31617` CUDA/HIP `_sync` 重构；`#5a4afd8/#30279` 测试命令行缺类型/后端时报错退出（均 fix/test 级）。
- `#a681a53`/`#f4d0673` 为同日「jinja: add 4 missing features」的提交与回滚（`#32352` 保留），记为回滚对，不单独计功能。

---

## 6. 累计结论延续与新增（继承 1–18，本期新增 19）

**累计结论（1–18 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性（发版当天合入 ≠ 在版内）；③ 默认值成为主要风险载体（本期 vLLM #67469 尺寸封顶、SGLang #50311–#50306 统一内存 opt-in 栈）；④ 异构/量化后端静默错误为高频（本期 SGLang #47802/#46392、vLLM #67592/#65926 共构）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（本期 vLLM #67717 Mamba prefix 状态哈希、SGLang #49170 PD tail replay）；⑥ 量化路径静默错误（scale/布局漏接，本期 SGLang #46392 SiTU 跑错激活、vLLM #67592 expert id 哨兵）；⑦ 专家张量/路由归属与分页漏接静默错误（本期 vLLM #67592 two-sided `num_experts` 哨兵致 garbage）；⑧ KV 卸载/HiCache/PD 健壮性 + 内存/缓存管理接口重构（本期 vLLM #63524 fs tier 只报写入块、SGLang #49170/#50306–#50311、TRT #21778）；⑨ 长上下文 int32/int64 溢出静默错误（本期 vLLM #65926 K-cache 越界 4 GiB、SGLang #49803 GDN chunk-state slot 8192 覆写）；⑩ MXFP8 KV 正确性簇（本期 SGLang #47802 fp8_e5m2 KV 触发条件）；⑪ 新旗舰模型 CP/KV/性能成为一等公民（本期 vLLM #65974 SM107 NVFP4 MoE、#67377 DSv4.1 FlyDSL、#66842 eager prefill CP、llama.cpp #32301 MiniMax-M3.1）；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径（本期 vLLM #67592/#65974、TRT #21778/#21585）；⑬ FP8/低比特 KV cache 跨引擎标准化落地（本期 SGLang #49717 KV dtype 插件、vLLM #67377 MXFP8 dense）；⑭ 投机解码验证路径跨引擎密集硬化（本期 SGLang #40893 缩减 draft vocab、llama.cpp #32363 DFlash2 元数据）；⑮ 跨后端可移植性成为显式架构主线（Apple MPS/NPU/Hexagon/SpacemiT/ROCm/OOT/SYCL/Vulkan/OpenCL；本期 llama.cpp #26348 VNNI/#31617 ROCm sync/OpenCL Q6_K、vLLM #62807 XPU、SGLang #33506/#49955 AMD）；⑯ 新旗舰模型进入「默认性能化」阶段 + 量化/MegaMoE 通用化；⑰ 推理服务前端 Rust 化成为跨引擎并行主线（本期 SGLang #50559 Python/Rust 共读 auth 表、vLLM #66878 Rust frontend n>1）；⑱ 推理服务默认安全加固成为跨引擎主线（本期 SGLang #50559 Rust 无 admin key 400→403 与 Python 对齐）。

**新增第 19 条（非量化的常规优化路径静默错误密度不亚于量化路径）**：本期 7 条静默正确性铁证中，#47802（GQA tile 划分读错 KV head）、#49170（PD/DP tail replay 误路由 active decode）、#67717（Mamba block 状态哈希错位）、#65926（K-cache 行偏移 int32 越界）、#49803（GDN chunk-state 偏置回绕）**均非量化引入**，而是注意力/缓存/路由的索引、分片、状态归属错误；另外 #46392（SiTU 跑成 SwiGLU）、#67592（expert id 哨兵语义）才是量化/路由路径。结论：随着 PD 解耦、DP、投机、chunked prefill、tail graph、统一内存等优化的正交组合成为默认配置，**正确性风险已从「量化 scale/布局漏接」扩散到「常规优化路径的状态/索引/归属」**；验证矩阵必须覆盖这些组合（并发、混合 EXTEND/DECODE、长上下文 packed batch、非整除 GQA group、pad 行语义），且不能只看吞吐/TTFT——本期 #49170（GSM8K 67.93→97.27%）、#47802（50.87→93.48%）均以准确率而非性能指标暴露问题。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 注意力 / KV head 正确性复核（非量化路径）**：SGLang `#47802`（GQA group>16 非 16 整除读错 KV head，GLM-4-32B GSM8K 50.87→93.48%、fp8_e5m2 49.51→92.87%，CI 未全绿）→ 部署 GLM-4-32B 等 group 非整除模型于 Triton/fp8_e5m2 前必须核验 KV head 映射与准确率。
- **P0 — PD/DP/投机组合静默错误复核**：SGLang `#49170`（PD+DP tail replay 吞 active decode，GSM8K 67.93→97.27%、`\ufffd` 损坏，非 upstream main 实测）、vLLM `#67717`（Mamba align prefix 状态哈希错位，输出 `access` vs `pillow`）；凡启用 PD、DP attention、eager tail graph、Mamba prefix caching 的组合，必须加入 acceptance/准确率验收而非只看吞吐。
- **P0 — 长上下文 int32/64 溢出（第 9 条）**：vLLM `#65926`（DSv4.1 K-cache gather 超 2^31 越界 4 GiB，MI355X 3/3 崩、burst 下静默写越界）、SGLang `#49803`（GDN chunk-state slot 8192 覆写，MI300X/MI355X，CI 未全绿）；1M 上下文/大 packed batch 部署须核验所有 offset 计算的 int64 拓宽。
- **P0 — 量化 kernel 功能/哨兵语义**：SGLang `#46392`（Kimi-K3 SiTU 在 SM90/SM120 跑成 GPT-OSS SwiGLU，cosine 0.63 vs 1.000）、vLLM `#67592`（two-sided `invalid_expert_id=num_experts` 致 garbage，GSM8K 0.0）；MXFP4/NVFP4 MoE 部署须核验激活函数实现与 padding 哨兵值是否被 kernel 正确跳过。
- **P1 — AITER/FlyDSL 性能与指标权衡（第 11/13 条）**：vLLM `#67377`（DSv4.1 MXFP8 FlyDSL prefill 1.78×/decode 1.61×）、`#65992`（Mega-MHC per-seam 16–41% 但服务端噪声内）、`#67805`（Engram 排序 TTFT p90 −17% 但 intvty p90 +20%）、`#65658`（PP SDMA +3.6%，opt-in）；关注「微基准 ≠ 服务级收益」与单指标权衡。
- **P1 — 统一内存 / KV 管理栈（第 8 条）**：SGLang `#50306`–`#50311`（冷启动首次 c256 5200→5630 tok/s，稳态接近噪声）、vLLM `#63524`（fs tier 只报实际写入块，避免喂给 BackpressureDetector）；统一内存增益需区分冷启动与稳态。
- **P1 — 新模型 / 跨硬件（第 11/15 条）**：llama.cpp `#32301`（MiniMax-M3.1 vs SGLang PPL 差 <0.3%、dspark ~1.5×）、`#26348`（x86 VNNI Q2_0 3×，注意 PPL 微差）、`#32296`（Metal LFM3 +3.3%）；vLLM `#65974`（SM107 NVFP4 MoE，无 perf claim）；TRT `#21585`（GLM-5.2 NVFP4 Rubin target，描述级）。
- **P1 — 服务安全 / 一致性（第 17/18 条）**：SGLang `#50559`（Python/Rust 共读 auth 表、无 admin key 400→403）、vLLM `#66878`（Rust frontend n>1）。

---

## 7. 社区与视频资源（HN Algolia，2026-11-15 epoch 窗口 1794700800–1794787199）

> 以下均为社区/Show HN，非引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词检索中 `SGLang`/`llama.cpp`/`speculative decoding` 均 **0 命中**；`TensorRT` 1 条（1MB Tenstorrent driver，1 分，非本引擎）；`KV cache` 1 条；`vLLM` 20 条多为泛 AI 资讯（仅 1 条沾边引擎）。

- **KV cache 研究/工具**：「Show HN: Vantage-V-AI – 2-bit ternary KV-cache with attention sinks」（**1 分**，窗口内）——社区 2-bit 三值 KV-cache + attention sinks 方案，与本期 KV 低比特/卸载主线（第 8/13 条）主题相关。**二手线索/未核验**。
- **vLLM 相关**：「Hardware-Agnostic Models in vLLM」（**2 分**，2026-11-15T02:25:51Z）——社区 vLLM 硬件无关模型文章，与本期跨后端主线（第 15 条）主题相关，但无法由一手来源确认内容。**二手线索/未核验**。
- 其余 HN 命中（`vLLM` 20 条如「Understanding Positional Embeddings in Modern LLMs」195 分、「LeCun: If you're interested in human-level AI...」6 分等，`KV cache` 1 条）多为泛 AI/LLM 资讯、模型舆情或本地工具，与推理引擎实现无直接关系，未列入。本期**无引擎相关的可确认官方视频**；Reddit / YouTube / X 未采集。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **26** / SGLang **59** / TRT-LLM **8** / llama.cpp **23** 合计 **116** 条（2026-11-15 UTC 窗口；2026-11-15 为周日，较前一窗口 153 回落，属周末正常；各 repo 均未触 per_page=100 上限、单页取全，无截断遗漏）。与上一期（2026-11-15 档案覆盖 11-14）连续，无空档。
- **版本核验**：releases API 确认——vLLM 仍为 v0.33.0rc3（2026-11-02）、SGLang v0.5.23（2026-11-02）、TRT-LLM dev 线 v1.4.0rc3（prerelease，2026-11-06）/ 稳定线 v1.2.1，窗口内均无新正式版本；llama.cpp 于 2026-11-15 推进 nightly b12419–b12427（prerelease），b12428 落在 11-16T00:12:45Z（窗口外），v1.0.0 正式版仍停留在 2026-11-13。
- **PR 正文来源**：**37 个关键 PR 正文经官方 issues API（markdown 官方一手）抓取**；vLLM #65992 首次 `IncompleteRead(32768 bytes)` 传输中断，单线程重试成功（正文 55,787 字符）。无 GitHub token 下 60/h 配额被用尽（起始 52→用尽）。其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度（如 TRT-LLM 8 条中 4 条 infra/test、llama.cpp `#32349`/`#32359`/`#32093` 等正文仅标题、SGLang `#49717` 正文抓取为空、vLLM `#66842` 正文抓取为空）。
- **描述级/未核验标注**：SGLang `#47802`/`#49803` CI 未全绿（Base/Extra/AMD ✗）、`#49170` 非 upstream main 实测（基于 v0.5.23 构建 + backport）、`#33506`/`#49955` 未给端到端数字（描述级）、`#49717` 正文为空；vLLM `#65974` 明确无 perf claim、`#65992` 服务端 A/B 噪声内、`#67805` interactivity p90 +20% 已并列记录、`#66842` 正文为空；TRT-LLM `#21585` 目录结构无数字、`#22104` 保留 nvbugs 5989912 open；llama.cpp `#26348` PPL 的 mean KLD 值正文截断、`#32363` 接受率来自单一简单任务、llama.cpp 多处 model/server fix 仅标题。
- **社区信号**：HN Algolia 引擎官方更新类故事无；社区侧 2-bit ternary KV-cache（Vantage-V-AI，1 分）、Hardware-Agnostic Models in vLLM（2 分）均按二手线索/未核验记录；无引擎相关的可确认官方视频。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/CI 未全绿/非 upstream main 实测」。
- **技能检索（find-skills / agent-browser）**：find-skills 经 SkillHub 语义检索（EN+ZH 各 1–3 组，含「LLM inference github commit monitoring」「github trending hotspot digest」「大模型推理 GitHub 提交采集 热点归档」）最高相关性仍为 0.050，返回编程大神.Skill / 育儿大师 / 腾讯文档 / 电子印章等通用编程/办公/育儿类技能，与「官方提交采集 + 证据分级归档 + 仓库同步」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/issues API）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次一致。
