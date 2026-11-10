# 大模型推理优化每日热点档案

> 档案日期：2026-11-10（本自动化于该日执行）
> 关注窗口：2026-11-09（周一）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（issues API，markdown，官方一手）+ HN Algolia（HTTPS，numericFilters 按 2026-11-09 epoch 窗口）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 99 / SGLang 92 / TRT-LLM 35 / llama.cpp 29 合计 255 条；四家单页均未触 per_page=100 上限（无截断）；releases API 确认窗口内**无新正式版本**（vLLM 仍为 v0.33.0rc3、SGLang v0.5.23、TRT-LLM v1.4.0rc3；llama.cpp 仅推进 nightly `b12236` 22:39Z / `b12237` 23:43Z，均 prerelease）；31 个关键 PR 正文经 issues API 抓取（markdown 干净文本，含标题级/截断标注）；HN Algolia 引擎关键词高信号弱（仅 1 条 Show HN「Flide」决策模型加速，119 分，二手线索/未核验）。本窗口为周一，提交量较周六/周日显著回升（vLLM 99 / SGLang 92），与往期周一（如 11-02、11-03）同形。

## 1. 版本状态与本期头号事件：vLLM Qwen4Exp 量化 MoE 后端深化 + KV Offload 跨引擎扩面 + 静默正确性延续（Metal/Granite 铁证）

**提交计数（2026-11-09 UTC，均 HTTP 200；2026-11-09 为周一，量较周末回升；vLLM 99 / SGLang 92 / TRT-LLM 35 / llama.cpp 29）**：
- **vLLM 99** / **SGLang 92** / **TensorRT-LLM 35** / **llama.cpp 29**。

**发布状态（窗口内无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-09 全部 99 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-09 含 mem_cache 统一内存重构栈、Kimi-K3/DSv4.1 正确性修复、AMD 量化融合密集提交。
- **TensorRT-LLM**：仍为 **v1.4.0rc3（True/prerelease，2026-11-06T21:38:32Z）**，窗口内无更新；rc3 收口后小幅稳定化（多为 infra/CI/bugfix）。
- **llama.cpp**：仍为 **v0.9.0（2026-11-04T13:08:06Z 正式发布）**；11-09 推进 nightly `b12236`（22:39Z）/ `b12237`（23:43Z，prerelease），无新正式版本。

**本期头号事件（vLLM Qwen4Exp 量化 MoE 后端 + KV Offload 扩面 + 静默正确性延续 + SGLang mem_cache 重构 + llama.cpp Granite/Metal 静默错误）**：
1. **vLLM Qwen4Exp 量化 MoE 后端深化（#64315 + #66512，新增第 12 条观察）**：`#64315` 把 Qwen4Exp（Qwen3.8-Flash-Next）routed experts 跑在 **DeepGEMM MegaMoE** 后端（`--kernel-config '{"moe_backend":"deep_gemm_mega_moe"}'`），BF16/FP8/MXFP8/MXFP4 checkpoint 自动 requantize，EP 必需；`#66512` 新的 fp8 QSA kernel 在 Blackwell（sm_100/103）提速 **2.63×（MTP decode/verify 形状）/ 1.37–1.42×（prefill）**、端到端 **−4.4% median TPOT**，且 QK/PV 与 split-K 逐位一致（仅 fp32 求和分组不同）。属「新旗舰模型量化 MoE 落地路径」主线。
2. **KV Offload / HiSparse 跨引擎扩面（第 8 条）**：vLLM `#56192`（KV offload OBJ 二级层支持 NIXL S3-over-RDMA 加速引擎，钉 NIXL 1.5.2，4 GiB 分块 DRAM 注册规避 `CUOBJ_MAX_MEMORY_REG_SIZE`）、`#65888`（HiSparse P/D import 在 GPU 池有富余时**拷贝进 resident GPU 页**，无需 NIXL 改动，GLM-5.3-NVFP4 1P1D 2×GB300：ITL p50 10.40ms vs main 11.36ms、NIXL bytes 1×、GSM8K 0.915 vs 0.906）、`#65559`（ROCm `SimpleCPUOffload` 用 `hipMemcpyFlagExtPreferCE` 把 host↔device 批量拷贝改走 SDMA 而非 CU blit，decode 旁拷贝由 +122% 慢→+0.4~3.9%）；SGLang `#27267`（HiCache host pool hugepage 预检/分配模式 off/prefer/required）、`#45727`（HiSparse 共享 KV 源支持，HiSparse V2）、`#48405`（AMD `vectorized_5d` fp8 KV 量化融合进 5D 写入，去 4 个 elementwise launch）。
3. **静默 / 量化正确性铁证再现（第 4/6 条）**：llama.cpp `#31449`（Granite NVFP4 FFN weight scales 未前向应用，PPL **4,899,652→18.24**，无报错，静默垃圾，强化第 6 条）；llama.cpp `#31662`（Metal `flash-attn` vec output scratch OOB 写，Metal shader validation 报 **109→0** 越界，强化第 4 条）；vLLM `#60531`（Triton unified attention SM100 batch-invariance 在 head-size-256 致 10 failures→20 passes、Gemma3 逐位一致，强化第 4 条）、`#65406`（MoE `w2_bias` 在 TP>1 被计 `tp_size` 次 → 正确 next token 错，TP=2/4 修前 token 4 胜）、`#66486`（MoE LoRA 在 DP+EP 三处「仅见本地批次」假设 → EP dispatch 送错 token-slot）。
4. **SGLang mem_cache 统一内存布局规划器重构（#48011–#48014 栈）**：`--enable-unified-memory` 下每个 arena 的 chain 把容量/准入/回收各自重算易不一致；`#48011` 引入 `plan()`（对任意 chain 形状给出是否满足 demand 及布局）与 demand 类型（纯函数、尚未接生产调用），为后续切换打底；配套 `#48010`/`#48009`（unified request rows 单一生命周期与 reset、PD 传输按行跟踪复用）。属 KV 内存管理的架构级清理。
5. **SGLang Kimi-K3 / DeepSeek-V4.1 正确性修复（第 5/7 条）**：`#43675`（Kimi-K3 派生权重缓冲跨 CUDA graph reload 保持 ——MLA 吸收投影 / 合并 MoE front / ROCm KDA 输入投影 / AttnRes 缓存乘积原地刷新，修 reload 后旧存储读取）、`#47406`（DSv4.1 `--enable-decoder-swa-bounded-replay` + CP + DSpark 在 prefill 输出 gather 时按原全 chunk 元数据期望 4096 行、实际返 32 尾行 → 加 `postprocess_cp_output` hook 修）、`#49262`（DSv4.1 MegaMoE input 由 fused decode mHC post 直写 epilogue，GSM8K 96.66/97.04 vs 96.82 基线、位精确）。

---

## 2. vLLM：2026-11-09 提交（99 条）—— Qwen4Exp 量化 MoE 后端 + QSA fp8 提速 + KV Offload 扩面 + 静默正确性

### 2.1 `#64315` [Model][Qwen4Exp] Run routed experts on DeepGEMM MegaMoE (MXFP8 or MXFP4)（量化 MoE 后端，新增第 12 条）
- **主题**：把 Qwen4Exp（Qwen3.8-Flash-Next）routed experts 跑在 DeepGEMM MegaMoE 后端（`--kernel-config '{"moe_backend":"deep_gemm_mega_moe"}'`，基于 #61538 的 `DeepGemmMegaMoEExperts`），EP 必需。
- **内容详情**：routed experts 按 checkpoint 格式加载，MegaMoE 能跑则原样、否则 requantize——BF16 官方（fused gate_up/down）原样；128×128 block FP8 官方 requantize 带警告；MXFP8（ModelOpt）原样；MXFP4（Quark）原样。可用 `--quantization` 强制 mxfp8/mxfp4。MXFP8/MXFP4 公开 checkpoint 尚未发布，仅单元测试覆盖（格式检测测 ModelOpt MXFP8；Quark MXFP4 走既有 Quark config 查找）。
- **应用场景**：Qwen4Exp 在 Blackwell 上的 DP+EP 量化 MoE 服务。
- **研判**：【事实】PR 给 checkpoint→MegaMoE 格式映射表与 requantize 规则。【观点】属「新旗舰模型量化 MoE 落地路径」（第 12 条）——把 MXFP8/MXFP4 接到 DeepGEMM MegaMoE，是 Qwen4Exp 在 vLLM 量化服务的主路径雏形；公开量化 checkpoint 缺位使路径暂靠单测覆盖，记为官方一手（格式/映射级，无端到端 t/s）。**证据等级**官方一手（映射级，无端到端对照）。

### 2.2 `#66512` [Perf][Qwen4Exp] Faster QSA kernel for large fp8 batches on Blackwell（QSA fp8 提速，第 12 条）
- **主题**：Qwen4Exp QSA 稀疏注意力在 sm_100/103（B200/B300）fp8 KV 上的新单趟 kernel（`vllm/models/qwen4_exp/nvidia/ops/qsa_fp8.py`），替代 split-K。
- **内容详情**：`rows × KV heads per rank > 64` 的 decode/verify 与 prefill 走新单趟 kernel；bf16 缓存 / 小 batch / 其他 GPU 保持 split-K 且**逐位不变**。QK 留 bf16、PV 转 fp16（e4m3→bf16/fp16 精确），仅 fp32 求和分组不同。验证：82 passed（GB200）、新 `test_qsa_fp8_kernel_correctness` 在全部 TP head 布局逐位（rtol=atol=2e-2）。
- **应用场景**：Qwen4Exp 在 Blackwell 的 QSA 稀疏注意力。
- **研判**：【事实】PR 给 **2.63×（MTP decode/verify）/ 1.37–1.42×（prefill，GB200 kernel time）**、端到端 **−4.4% median TPOT** 与逐位验证。【观点】属「Qwen4Exp 量化注意力提速」（第 12 条），纯性能、数值不变（非近似）。**证据等级**官方一手（含 GB200 提速表与逐位测试）。

### 2.3 `#56192` [Feature][KV Offload] Add RDMA-capable OBJ tier (#55855)（KV Offload RDMA 层，强化第 8 条）
- **主题**：KV offload OBJ 二级层支持 NIXL 加速（S3-over-RDMA）object-store 引擎，钉 NIXL 1.5.2；`accelerated/type/use_virtual_addressing/req_checksum/resp_checksum` 透传 `store_config.backend_params`，不新增 `ObjStoreConfig` 字段。
- **内容详情**：主 CPU DRAM 池按至多 `4 GiB - 64 KiB` 分块注册（cuObject 引擎拒更大注册，`CUOBJ_MAX_MEMORY_REG_SIZE` 在 CUDA 13.2/13.3 分别为 4 GiB / 4 GiB-64 KiB）；超 4 GiB 的 CPU tier 启动失败、单块超限显式 `ValueError`。NIXL 1.5.2 仅 ship vendor 引擎（type: dell），`accelerated:true` 无 type 时仍走 plain S3；通用 S3-RDMA 引擎在 NIXL main/1.6。**PR 描述存在未填占位（"Not duplicating #65956: that PR is about <fill...>"、"Scope note... <Confirm before posting.>"），记为描述级/待确认**。
- **应用场景**：vLLM KV offload 经 RDMA 加速的 OBJ 远程层。
- **研判**：【事实】PR 给分块注册规避与参数透传。【观点】属「KV 卸载远程层扩面」（第 8 条）——把 NIXL S3-over-RDMA 接入 OBJ tier；描述含未填占位，记为官方一手（描述级/待确认，无实测带宽）。**证据等级**官方一手（描述级，含未填占位）。

### 2.4 `#65888` [HiSparse] promote P/D imports into GPU pages without NIXL changes（HiSparse P/D 入 GPU 页，强化第 8 条）
- **主题**：HiSparse P/D import 落地仍入 host tier（同 main）；当 import 完成且 GPU 池高于 transition watermark 富余时，HiSparse 把导入页拷贝进新的 resident GPU 页（每步一个批量 host→device 拷贝）。
- **内容详情**：GLM-5.3-NVFP4，1P1D 2×GB300（TP4 each），decoder 跑 `MultiConnector[NixlConnector, HiSparseConnector]`、NIXL over RDMA。对照 main / #55398 / 本 PR（同节点）：Multi-turn 64K ITL p50 11.36 / 10.49 / **10.40 ms**，TTFT p50 1469 / 2086 / 1954 ms，GSM8K 0.906/0.906/0.915（wall 58.0/43.7/42.4 s），NIXL bytes 1×/3.4×/**1×**。施加 #65607 后 ITL p50 11.34/10.36/10.31 ms；8K–100K needle retrieval 每次正确。
- **应用场景**：HiSparse P/D 导入的 GPU 驻留加速。
- **研判**：【事实】PR 给三变体逐配置 ITL/TTFT/GSM8K/NIXL-bytes 对照（本 PR ITL 10.40ms、NIXL bytes 1×、GSM8K 0.915）。【观点】属「HiSparse P/D 入 GPU 页」（第 8 条）——在零 NIXL 改动下把导入页搬进 resident GPU 页，省去 #55398 的 NIXL bytes 膨胀（3.4×→1×）且 ITL 更好。**证据等级**官方一手（含 GB300 对照表）。

### 2.5 `#60531` [Bugfix][Kernel] Preserve batch invariance in Triton unified attention on SM100（SM100 注意力正确性，强化第 4 条）
- **主题**：在 Triton unified attention 中把 batch-invariant 执行排除在 SM100 家族 large-head tuning 路径之外，恢复 batch invariance 下的逐位注意力一致。
- **内容详情**：回归比较「单独 causal query」与「同 query 在多 query block 内」的输出；head-size-256 SM100 BF16 配置恢复一致；head-size-512 作回归对照；新测试需 SM100、上游 CI 跳过。B200 上：新 kernel 回归 10 failures/10 passes → 20 passes；Gemma3 decode vs teacher-prefill 选定 log-prob eager/FULL CUDA graph/单 batch vs 批量均「exact for all 3」。不声称完整投机服务正确性。
- **应用场景**：SM100（Blackwell）Triton unified attention 的 batch invariance。
- **研判**：【事实】PR 给 Gemma3 三上下文逐位一致与 10 failures→20 passes。【观点】属「SM100 注意力逐位正确性」（第 4 条）——batch invariance 回归修复，非静默垃圾但属数值一致性主线；head-size-256 为受影响形状。**证据等级**官方一手（含 B200 逐位对照）。

### 2.6 `#65559` [ROCm][KV Offload] SimpleCPUOffload: batch copies on SDMA with hipMemcpyFlagExtPreferCE（ROCm KV 卸载性能，强化第 8 条）
- **主题**：ROCm 镜像升 HIP 7.16；其上 `hipMemcpyBatchAsync` 默认把 `SimpleCPUOffloadConnector` 的 host↔device 批量拷贝作为 blit kernel 跑在 CU 上，拖慢相邻 decode；加 `hipMemcpyFlagExtPreferCE`（0x100）把整批交给 SDMA 引擎、无 kernel。
- **内容详情**：`build_params` 在 HIP≥7.16 设该 flag（旧运行时 flag 0，CUDA 不动）。MI355X 单卡：decode 旁拷贝无 flag 时中位 **+122% 慢**，有 flag 时 **+0.4%~+3.9%**；拷贝带宽不变（44–56 GB/s）；round trip 逐字节一致。
- **应用场景**：vLLM ROCm KV offload 批量拷贝的 SDMA 卸载。
- **研判**：【事实】PR 给 MI355X decode 旁拷贝 +122%→+0.4~3.9% 与带宽不变。【观点】属「ROCm KV 卸载性能」（第 8 条）——用 SDMA 避免 CU blit 占满相邻 decode 的计算，属已验证性能件。**证据等级**官方一手（含 MI355X 对照）。

### 2.7 `#65406` [Bugfix][MoE] Count the expert down-projection bias once under TP（MoE 正确性，强化第 6 条）
- **主题**：`RoutedExperts.weight_loader` 把完整 `w2_bias` 拷到每个 TP rank，使 row-parallel down projection 在 all-reduce 前计 `tp_size` 次；现仅 TP rank 0 保留（同 gpt-oss 原生 loader）。Fixes #65405。
- **内容详情**：TP=1/EP（每 rank tp_rank=0）/ gpt-oss 原生 loader 不变。新测试 `test_moe_layer_tp_expert_bias`：TP=1 pass、TP=2 修前 1882/2048 超出容差（max abs err 1.85）→ pass；端到端 synthetic Mixtral-style biased checkpoint 经 `LLM(model_impl="transformers")`：TP=1 token 3、TP=2/4 修前 token 4（错）→ 修后 token 3。
- **应用场景**：带 expert bias 的 MoE 在 TP>1。
- **研判**：【事实】PR 给 TP=2/4 修前 token 4 胜、修后 token 3 的端到端复现与 max abs err 1.85。【观点】属「MoE TP 正确性」（第 6 条）——biased experts 在通用 loader 下 TP>1 静默错 token，修后正确。**证据等级**官方一手（含 TP=2/4 对照）。

### 2.8 `#66486` [Bugfix][LoRA] Fix MoE LoRA correctness under DP + EP（MoE LoRA 正确性，强化第 5/6 条）
- **主题**：DP+EP 下 MoE LoRA 跑在来自其他 DP rank 分发的 token 上，但三处假设「专家只见本地批次」出错。
- **内容详情**：① `LoRAKernelMeta.prepare_tensors` 在无 LoRA 批次时先于拷贝 token→slot 映射返回，使 EP dispatch 送上一 batch 的 slot；② `add_lora_fused_moe`/`moe_lora_align_block_size`（含 TRTLLM MXFP4 LoRA experts）以本 rank 的 no-LoRA 标志与 active-slot 统计门控分发批次；③ 空闲 DP rank 的 dummy batch 激活 warmup LoRA 后清空该 rank 全部 adapter。合并 #61714/#63526，适配 Model Runner V1 移除（#63373），加 DP=2+EP 端到端测试。
- **应用场景**：MoE LoRA 在 DP+EP 部署。
- **研判**：【观点】属「MoE LoRA × DP+EP 正确性」（第 5/6 条）——三处本地批次假设在跨 DP 分发下致 EP dispatch 错 slot / warmup LoRA 误清；合并既有修复并补 DP=2+EP 测试。**证据等级**官方一手（fix 级，部分截断）。

### 2.9 `#66268` [Recipe][ROCm] MiniMax-M3.1-Flash MI355X AgentX PD disaggregation over MoRI-IO（AMD PD 配方，第 8 条）
- **主题**：为 MiniMax-M3.1-Flash 加 opt-in MI355X `pd_cluster` preset（`mi355x_agentx_pd`）：1 个 TP4 prefill 经 MoRI-IO RDMA WRITE 喂 2 个 TP4 decode（1P2D，12 GPU），AgentX 并发 40–128。
- **内容详情**：6 个并发点（40/48/64/96/112/128），各自 decode batch cap / CUDA graph 尺寸 / prefill host-DRAM KV 大小；等 p90 交互下比 SemiAnalysisAI/InferenceX#4866 聚合 TP4 lane **多 6–32% tok/s/GPU**。无该配置精度评测（仅 `validate-source.mjs` 通过、单 3600s agentx-harness 跑 ROCm/ATOM CI）。
- **应用场景**：MI355X 上 MiniMax-M3.1-Flash 的 AgentX PD 解聚。
- **研判**：【事实】PR 给 6 并发点 tok/s/GPU +6–32% vs #4866（同 p90 交互）。【观点】属「AMD PD 解聚配方」（第 8 条），作者报告未独立验证（无精度评测），记为官方一手（作者报告，无精度验证）。**证据等级**官方一手（作者报告，无精度验证）。

### 2.10 `#66546` [Transformers backend] Support EAGLE3 aux hidden states with PP and layer 0（EAGLE3 + PP，强化第 5 条）
- **主题**：在 Transformers modeling backend 支持 EAGLE3 aux hidden states 跨 PP 与 layer 0，匹配原生 Llama（服务 Llama 的 #66547 一部分）。
- **内容详情**：backend 模型设 `supports_aux_hidden_states_over_pp` 并经 `EagleModelMixin` helper 在 PP rank 间传 aux hidden states；末 PP rank 在 spec decode 需时保留 target input embedding（供无 `embed_tokens` 的 drafter 共享）；修 `_target_class` 在 rank 0 变 `PPMissingLayer` 致 aux recorder 钩到 PP 占位而非 decoder layer；aux hidden layer 0 即 backend 上的 input embeddings（修 `_layer_names[-1]` 越界）。多模态 backend 仍拒 EAGLE3+PP。
- **应用场景**：Transformers backend 上 Llama 的 EAGLE3 投机解码 + PP。
- **研判**：【观点】属「EAGLE3 投机解码 × PP 支持」（第 5 条），backend 侧对齐原生 Llama 的 aux hidden 传递与 layer 0 处理；多模态仍拒。**证据等级**官方一手（feature 级）。

### 2.11 本期其余 vLLM 要点（标题/正文级）
- **量化 / MoE / 注意力**：`#61666`（compressed-tensors MXFP6 `mxfp6-pack-quantized`）、`#65972`（TRT-LLM monolithic MoE 经 vLLM top-k 跳过 DP padding rows）、`#64315`（见 2.1）、`#63847`（ROCm persistent_topk）、`#66537`（H200 FP8 config GLM-5.3-Flash TP8）、`#61630`（XPU tuned large-head prefill）、`#66321`（ROCm CI PCP for GLM-5.2 sparse MLA）。
- **KV Offload / HiSparse / PD**：`#56192`（见 2.3）、`#65888`（见 2.4）、`#65559`（见 2.5 实为 2.6）、`#60390`（CPU-offloaded Qwen4Exp PLE table 跨 co-located DP 副本共享）、`#64693`（ROCm MoRIIO 支持 PD 解聚 prefill 实例的 PP + MoRI WRITE mode）、`#66397`（ROCm 在 warmup 建 DSv4.1 replay builders、load 建 mono runner）。
- **ROCm / 调度 / 启动 / Rust Frontend**：`#66312`/`#66307`/`#66338`/`#66316`/`#66310`/`#66308`/`#66548`/`#66611`（ROCm AITER block-FP8 / HC kernel / Fault Tolerance / LM Eval / Model Tests / Router GEMM / MXFP4 padding / nemotron GSM8K CI）、`#66319`（ROCm Mamba cache stochastic rounding gfx950）、`#66398`（ROCm DiffusionGemma logprobs 设备端存免 host sync）、`#66224`（简化 `prefill_schedule_interval` 出 DP）、`#65349`（warm up 仅长 prefill/eager drafters/constrained decoding 达的 JIT kernel）、`#59981`（V2 runner decode 优化 for NVIDIA Confidential Computing）、`#63337`（AsyncLLM shutdown 区分 stopped vs dead）、`#65629`/`#66119`/`#66478`/`#65864`/`#66105`/`#66103`/`#66107`/`#66106`/`#66104`（Rust Frontend 输出通道内存/log-level/scale-out/routed experts/metrics/render 等多项）、`#65156`（frontend 启动 Rust frontend 后关父监听 socket）、`#66081`（spec decode 非延迟 structured-output draft retrieval）、`#64897`（spec decode XPU greedy_argmax NaN→-inf torch fallback）。
- **Transformers backend / 多模态 / 其他**：`#66545`（batch-sharded sampling）、`#66559`（Migrate Phi3）、`#65735`（transcription 建模后端）、`#66413`（LoRA on lm_head 与任意 input embedding）、`#66417`（Pooling 保 CLIP/SigLIP special tokens）、`#66432`（Qwen2-VL video budget 补到 context length）、`#65818`（Granite 4.1 Vision deepstack handoff 修复）、`#64041`（chat derender usage 报 reasoning/cached tokens）、`#65744`（Chat Completions tools/tool_choice 校验）、`#66289`（Qwen3.5 FlashInfer GemmaRMSNorm JIT→Triton for arch<sm80）、`#65995`（ROCm AITER preload FlyDSL INT4 QuickReduce）。

---

## 3. SGLang：2026-11-09 提交（92 条）—— mem_cache 统一内存布局规划器重构 + Kimi-K3/DSv4.1 正确性 + AMD 量化融合

### 3.1 `#48011` [Refactor] mem_cache: Add a layout planner and request demand for the unified chain（mem_cache 重构头号，架构级）
- **主题**：`--enable-unified-memory` 下每个 arena 的 chain 把容量/准入/回收各自按规则重算易不一致；本 PR（栈 `#48014`→`#48011`→`#48010`→`#48009` 的顶层）引入两个纯函数——`plan(snapshot, demands, step)`（对任意 chain 形状给出是否满足 demand 及布局，含 `NO_MOVE`/`FLUSH_ENDS`/`RELAYOUT`）与 demand 类型（命名每个 sub-pool 的分配取用与 request estimate），**尚无生产调用方**。
- **内容详情**：置于 `mem_cache/kv/alloc/pricing.py` 旁，与既有 per-shape 函数并存；bands in place 为不动点、floats 间重定位；下一步 PR 才把 chain 切到它们。
- **应用场景**：SGLang 统一内存 KV 缓存的容量/准入/回收一致性。
- **研判**：【观点】属「HiCache/统一内存 KV 架构重构」——把分散的规则收为单一 layout planner，消除 capacity/admission/reclaim 各算各的歧义；纯函数、尚未接线，记为官方一手（refactor 级/接线前的可评审切片）。**证据等级**官方一手（refactor 级，未接线）。

### 3.2 `#43675` [Fix] Kimi-K3: Fix derived weight buffers across CUDA graph reloads（Kimi-K3 权重缓冲正确性，强化第 7 条）
- **主题**：CUDA graph 捕获后 reload Kimi-K3 权重会留 captured 操作读旧 derived-weight 存储；AttnRes 也在底层权重变更后保留缓存的 norm/projection 乘积（影响 eager）。
- **内容详情**：保持 MLA 吸收投影缓冲跨 reload；复用合并 MoE front 与 CUDA BFA 缓冲（恢复参数视图进合并存储）、保持 dequantized BFA 投影缓冲；复用 ROCm fused KDA 输入投影缓冲（保整缓冲/BFA-tail/参数视图地址）；原地刷新 CUDA KDA fused-decode tensor 参数；原地刷新 BF16/FP32 AttnRes 缓存乘积；加 repeated-reload 回归。Reload 契约假定 model config/shapes/dtypes/devices 不变；原生 ROCm kernel 与所有量化 loader 变体未覆盖；不声称每个 K3 热更新路径已验证。
- **应用场景**：Kimi-K3 权重热更新 / CUDA graph reload。
- **研判**：【观点】属「Kimi-K3 派生权重/专家缓冲跨 reload 正确性」（第 7 条）——修 reload 后旧存储/缓存乘积被读；ROCm 与量化 loader 变体验证未覆盖，记为官方一手（fix 级，部分未覆盖）。**证据等级**官方一手（fix 级，部分未覆盖）。

### 3.3 `#47406` [Fix] deepseek-v4.1: Fix bounded replay with CP and DSpark（DSv4.1 正确性，强化第 5 条）
- **主题**：开 `--enable-decoder-swa-bounded-replay` + CP + DSpark 时 prefill 输出 gather 可能失败——bounded replay 仅留所需 SWA tail，DSpark aux hidden capture 使模型返紧凑尾输出而非全 chunk；`EagerRunner._execute_extend_cp` 绕过模型 tail-aware 后处理、按原全 chunk CP 元数据 gather，期望行数多于实际返回。
- **内容详情**：16,384-token chunk、CP4、128-token tail 时每 rank 返 32 尾行、原元数据期望 4096 行（`got=32, expected_at_least=4096`）。加可选模型特定 `postprocess_cp_output` hook 给 eager CP runner，使 DeepSeek V4/V4.1 处理紧凑输出。CP 配置此前默认关 bounded replay，但显式开启仍能到该不兼容路径。
- **应用场景**：DSv4.1 bounded replay + CP + DSpark 的 prefill 输出 gather。
- **研判**：【事实】PR 给复现（16K chunk/CP4/128 tail：got 32 vs expected 4096）与 `postprocess_cp_output` hook。【观点】属「DSv4.1 长上下文 + CP + DSpark 正确性」（第 5 条）——显式开启 bounded replay 与 CP/DSpark 组合会触 gather 行数不符，属配置组合下的正确性陷阱。**证据等级**官方一手（含复现）。

### 3.4 `#49262` [Perf] [NVIDIA] dsv41: Write the MegaMoE input from the fused decode mHC post（DSv4.1 MegaMoE 输入融合，第 12 条）
- **主题**：fused decode mHC post（`#46088`/`#47949`）写 normed FFN 输入行；与 MegaMoE 时 `mega_moe_pre_dispatch` 每层单独 kernel 仅把这些行量化进 MegaMoE input buffer。本 PR 给 mHC post 加 MegaMoE-input epilogue（直接写 fp8 rows + routed UE8M0 scales + shared BLOCK_M-paged UE8M0 scales，数学同 `mega_moe_pre_dispatch`），MegaMoE 跳过 pre-dispatch；post kernel 加 PDL。
- **内容详情**：epilogue 仅用于 MegaMoE all-to-all + fp8×fp4 MoE 权重；否则不变。准确性：`test_mhc_post_combine_norm_decode.py` 验 epilogue 输出与 post+pre-dispatch **逐位相等**（1–257 行、block_m 64/128）；GSM8K DSv4.1-Flash GB300 1P1D DEP16：96.66/97.04（本改 + #49321 + fused decode post + BF16x3 side mix #48456）vs 96.66/96.82（Mega-mHC decode 基线）。性能：GB300 DSv4.1-Flash 解聚（6 DEP2 prefill + DEP32 decode on 8 nodes）未给具体数字（描述级）。
- **应用场景**：DSv4.1-Flash 在 Blackwell 的 MegaMoE 解码输入融合。
- **研判**：【事实】PR 给逐位相等验证与 GSM8K 96.66/97.04 vs 96.82（#49321 之上）。【观点】属「DSv4.1 MegaMoE 输入融合」（第 12 条），消除每层 pre-dispatch kernel、位精确、GSM8K 持平偏优；性能数字描述级未给，记为官方一手（含逐位+GSM8K，性能描述级）。**证据等级**官方一手（含逐位+GSM8K，性能描述级）。

### 3.5 `#49135` [Perf] kda: shorten the MTP verify phase-1 critical path（KDA MTP verify 提速，第 5 条旁支）
- **主题**：KDA MTP verify kernel（`kda_decode_mtp.py`）phase-1 conv precompute 在 block barrier 前的临界路径上；重构去掉 q/k/g producer loop 仅省 18–19% kernel（K3.1 生产形状 H=12, num_spec=7, ring+fused output norm, N=32/64）。
- **内容详情**：SASS（sm_107a）显示每步在 store 后立刻消费 load、ptxas 无法把 plain load 移过不可证 alias 的 store；每步独立 basic block。改法：① 每 job 一个直线块（把 `p1_job` 分支提出 producer-step loop、`i_t<T_LOOP` 在 `T_LOOP%P1_JOB_WARPS==0` 时编译期丢）；② 窄块上 invariant loads 只读。
- **应用场景**：Kimi-K3.1 KDA MTP verify 内核临界路径。
- **研判**：【事实】PR 给 SASS 级归因（load 不能移过 store）与重构法。【观点】属「KDA MTP verify 提速」（第 5 条旁支），-11%@N=32 / -12%@N=64 on VR（描述级，未给端到端），记为官方一手（kernel 级，端到端描述级）。**证据等级**官方一手（kernel 级，端到端描述级）。

### 3.6 `#27267` [Feat] mem_cache: Add hugepage preflight and allocation modes for host pools（HiCache host hugepage，强化第 8 条）
- **主题**：设 `SGLANG_HUGEPAGE_SIZE` 时 HiCache host-pool 预检只看 `psutil.virtual_memory().available`（不含 reserved hugetlb），启动可能报「Not enough host memory」即便大 free hugetlb 池可用；本 PR 加 hugepage 预检/分配模式 off/prefer/required。
- **内容详情**：`mmap_allocator.py` 加 `hugepage_size_requested()`/`hugepage_mode()`/`hugetlb_pool_free_bytes()`（sysfs free-resv，受 hugetlb cgroup 限）/per-tensor backend 跟踪；`alloc_mmap` 尊重模式（required 失败即 raise、prefer 回退普通页）；`host_memory.py` 泛化 cgroup headroom 共享内存/hugetlb 控制器。
- **应用场景**：HiCache host pool 的 hugepage 预检与分配。
- **研判**：【观点】属「HiCache host 内存 hugepage 支持」（第 8 条），让 host pool 正确计入 hugetlb 池、避免误报内存不足；安全默认（prefer 回退）。**证据等级**官方一手（feature 级）。

### 3.7 `#49205` [Feat] [AMD] custom-all-reduce: DPX hierarchical 2shot_push to carv2 on MI450 (gfx1250)（AMD 自定义 all-reduce，国产后端版图）
- **主题**：基于 #49003（carv2 在 TP16 跨两 MI455X 主机 DPX 模式，把所有 16 rank 当 flat peers 致 2shot_push 几乎整张量过 fabric 两次）；本 PR 利用同 SoC 两分区共享 HBM、其间流量便宜，做层次化 2shot_push。
- **内容详情**：TP2 同 SoC vs 跨两 SoC（bf16, CUDA graph, us）：4KB 2.3/5.2、4MB 13.7/25.9、8MB 24.8/44.6（1shot_push）；2shot_push 同 SoC 3.1/18.4/19.5、跨 SoC 7.5/18.0/28.7。本 PR 减少同 SoC 内重复 fabric 传输。
- **应用场景**：MI450（gfx1250）上 DPX 模式的 carv2 自定义 all-reduce。
- **研判**：【事实】PR 给同/跨 SoC 各级别 1shot/2shot 延迟表。【观点】属「AMD/国产后端 all-reduce 层次化」（延续 #49003/#49001 MI450/UALink），利用 SoC 内 HBM 共享降 fabric 流量；量化口径为微秒级 kernel 延迟，记为官方一手（含延迟表）。**证据等级**官方一手（含延迟表）。

### 3.8 `#48405` [Perf] [AMD] kvcache: Fuse fp8 quantization into the vectorized_5d KV-cache write（AMD fp8 KV 写入融合，强化第 6/10 条）
- **主题**：`vectorized_5d` KV 布局 + fp8 KV 缓存下，`MHATokenToKVPool.set_kv_buffer` 每 4 个 elementwise launch 做 K/V 除以 scale 转 fp8 再跑 5D writer；本 PR 把 divide+cast 融进 5D writer（边写边量化）。
- **内容详情**：5D writer 接可选 K/V scales（device tensor 读内核免 host sync、host tensor 或 float）；fp8 缓存时 fp32 除、饱和到 fp8 范围、cast；gfx950 head-dim-256 新鲜 fp8 prefill 改回读刚写入的 K/V（同 head-dim-128 ASM 分支）而非 cast 原始 K/V 用 descales。
- **应用场景**：AMD gfx950 `vectorized_5d` + fp8 KV 缓存的写入。
- **研判**：【观点】属「AMD fp8 KV 写入量化融合」（第 6/10 条），去 4 个 elementwise launch、device scale 免 host sync；无 before/after t/s（仅架构），记为官方一手（性能/架构级）。**证据等级**官方一手（性能/架构级，无 t/s 对照）。

### 3.9 `#45727` [Feat] hisparse: Support shared KV sources in HiSparse MLA swap kernels（HiSparse 共享 KV 源，强化第 8 条）
- **主题**：HiSparse MLA swap kernels 假设 request-private Host 存储 + hot buffer；支持共享 HiCache 存储需直接访问 resident Device KV 与 Hot rows（不同内存布局）。本 PR（#32314 的拆分）加 kernel 支持（HiSparse V2）。
- **内容详情**：加可选 `device_locs` 直接读 resident KV 免拷进 hot buffer；加可选 `host_binding`。（描述级，正文未给性能/精度数字。）
- **应用场景**：HiSparse V2 的共享 KV 源（HiCache 共享存储）swap。
- **研判**：【观点】属「HiSparse 共享 KV 源支持」（第 8 条），为 HiSparse V2 做 kernel 层准备；描述级无数字，记为官方一手（feature 级，描述级）。**证据等级**官方一手（feature 级，描述级）。

### 3.10 `#49388` [Fix] [NVIDIA] mega_moe: Accept MXFP4 activations at DeepSeek-V4.1-Flash's 2304 intermediate size（MegaMoE 形状适配，第 12 条）
- **主题**：`--enable-w4a4-mxfp4-megamoe`（`mxf4xmxf4` MMA）在 DSv4.1-Flash 起不来——`check_mega_moe_shapes` 要求 `moe_intermediate_size % 512 == 0`，而 V4.1-Flash 为 2304（512 来自 16-byte token SF rows：16 UE8M0 × 32 elements）。
- **内容详情**：MXFP4 路径把 intermediate SFs 存/载 MN-major（同 FP8×FP4），intermediate 只需 `mxf4xmxf4` 与 `fp8xfp4`/`fp8xfp8` 一致的 `intermediate % 128`；input SF rows 仍需 `hidden % 512`（V4.1-Flash 5120 满足）。`test_mega_moe_deepgemm_api.py` 加 `test_shape_check_mxf4_takes_intermediate_multiples_of_128`（CPU）：接受 (5120,2304)、拒非 128 倍数 intermediate 与非 512 倍数 hidden；main 失败、本 PR 通过。
- **应用场景**：DSv4.1-Flash 的 MXFP4 MegaMoE（W4A4）启用。
- **研判**：【事实】PR 给 (5120,2304) 接受、非 128 倍数拒，及 main 失败→本 PR 通过的测试。【观点】属「MegaMoE MXFP4 形状适配」（第 12 条）——解 DSv4.1-Flash 的 2304 intermediate 起不来，使 W4A4 MegaMoE 可用。**证据等级**官方一手（含形状校验测试）。

### 3.11 本期其余 SGLang 要点（标题/正文级）
- **mem_cache 重构栈**：`#48014`/`#48013`/`#48012`/`#48010`/`#48009`（unified chain 的 layout planner 接线 / unified request rows 单一生命周期与 reset / PD 传输按行跟踪复用）、`#48013`（unified memory 经单一 layout search 规划/准入/回收）。
- **AMD / 量化融合**：`#47585`（quark 拓宽 Quark MXFP4 W4A16 MoE gate 到 consumer RDNA、无 aiter 时清晰失败）、`#48536`（AMD 融合 SiLU-and-mul + dynamic per-tensor fp8 quant of dense MLPs on gfx95）、`#48406`（AMD fused QK-norm + RoPE + KV write 服务 vectorized_5d）、`#48583`（Intel xpu_kernel 升 0.6.0）、`#49003`（carv2 跨 gfx1250 UALink 节点）、`#48165`（Hopper group32 FP8 量化用 FP32 scales 提速）、`#47658`（SM120 honour DEEP_GEMM_HC_PRENORM for DSv4.1 compensated mhc projection）、`#49182`/`#49247`（dsv4 恢复 SM120/121 到 DeepGEMM indexer gates / 从 dp_attention import get_eager_max_batch_size）。
- **性能 / 内核**：`#49262`（见 3.4）、`#49135`（见 3.5）、`#48984`（mhc one-pass DeepGEMM prenorm GEMM；回退 sm107 decode MegaMoE SM reserve）、`#47627`（dspark 把 DSv4.1 长上下文 target verify metadata 旁置 draft）、`#46062`（DSv4.1 每层保一个 zeroed MegaAttention packed-query buffer 替代逐层清零）、`#46003`（dsv4.1 仅对 indexer consumers 的 candidate 位置打分）、`#48585`（gdn 减 MIS output 与 QK 准备开销）、`#49088`（minimax-sparse decode indexer CP on Hopper at TP8 with attention DP2）、`#46577`（inkling 跳过 CUDA-graph padding tokens 的 fused attn prologue KV stores）、`#46663`（gemma4 修 Bolt E4B 与 Gemma4 prefill CUDA graphs）、`#47493`（dp-attention 在 draft extend 前从 extend outputs 裁掉 MLP-sync pad rows）、`#46934`（hicache 恢复 mid-tree anchor 下的 sub-window SWA L3 hits）、`#48786`（HiCache 加 unified layout 的 tma copy kernel）、`#46168`/`#45106`（mooncake 层加载 host 时间削减/分组并行化）、`#46552`（VLM 复用 MultiModalStaticCache.get 的预计算 combined_hash）。
- **正确性与健壮性**：`#43675`（见 3.2）、`#47406`（见 3.3）、`#47529`（RL 在 DCP/KV sharding 下返 routed experts 崩溃修复）、`#49350`（去 removed SGLANG_HICACHE_TMA_TRANSFER 出 page-unified HiCache TMA 测试）、`#49253`（scheduler pause 时 retract 每个 KV holder 并排序少量 in-flight 后释放）、`#49218`（unregister 失败的 staging room 再释放其 KV）、`#46214`（flashinfer ragged prefill 对 hybrid-SWA 模型应用 sliding window）、`#48871`（modelopt_fp4 在 Marlin 路径跳过 swizzled NVFP4 MoE blockscales）、`#48812`（weight_checker 在 live weight 设备比较）、`#46586`（srt 客户端断开即 abort running requests）、`#44477`（metrics 处理 restore 后 process CPU clock resets）。
- **模型 / 多模态 / 其他**：`#49399`（ernie55 从 sgl-flash-mla wheel import FlashMLA）、`#48656`（comfyui 加载 FastH3/H3 FP8/Qwen-Image 2.1 Turbo/INT8 单文件）、`#48364`（CPU 支持 Intern-S2-Mobius 模型）、`#49318`（NPU/glm5.3 2P2D 32P PD-separation perf case）、`#48014`/`#49362`/`#49398`/`#49401`/`#49260`/`#49239`/`#49237`/`#49163`/`#49171`（cookbook/CI/doc/cleanup 多项）。

---

## 4. TensorRT-LLM：2026-11-09 提交（35 条）—— rc3 稳定化：FP8 GDN/block-scale 量化 + 解聚健壮性 + KVCacheManagerV2 析构安全

### 4.1 `#21287` [perf] Support FP8 rowwise GDN projections in Qwen3.5 and fuse their norm + per-token FP8 quantize（FP8 GDN 量化融合，强化第 6 条）
- **主题**：Qwen3.5 的 per-channel FP8 GDN projections（FP8_PER_CHANNEL_PER_TOKEN 的 `in_proj_qkv`/`in_proj_z` 融合为一个 FP8 rowwise `in_proj_qkvz`）；Dense MLP quant keys 重路由到 `.mlp.mlp.` 模块树（覆盖所有 quant 算法，不只 NVFP4）；融合 (add+)RMSNorm + per-token FP8 quantize（Triton）。
- **内容详情**：gemma add+RMSNorm 直喂 `in_proj_qkvz` 保 BF16 输出给 unquantized `in_proj_ba`、Gated RMSNorm 直喂 `out_proj`；`FP8RowwiseLinearMethod` 用 `vectorized_per_token_fp8_quant` 替代 `quantize_e4m3_activation` + scale cast。性能（RTX PRO 6000 / SM120，BF16 激活，Qwen3.5-4B 形状 hidden 2560、GDN 32 heads×128）：device time per call 数字描述级（正文未给具体倍数）。
- **应用场景**：Qwen3.5（VL）在 SM120 的 FP8 GDN 投影与激活量化。
- **研判**：【事实】PR 给 FP8 rowwise GDN 融合与 RMSNorm+quantize 融合的改动。【观点】属「FP8 激活/投影量化融合」（第 6 条），覆盖所有 quant 算法（不只 NVFP4）；device time 描述级未给倍数，记为官方一手（含改动，性能描述级）。**证据等级**官方一手（含改动，性能描述级）。

### 4.2 `#20766` [feat] Qualify FP8 block-scale weights for MX (Qwen3 dense)（FP8 block-scale 权重准入，强化第 6/10 条）
- **主题**：runtime qualification registry 加 SM90-only Qwen3 FP8 block-scale profile（要求 checkpoint 声明 (128,128) weight blocks、专用 transform-layout ABI）；loader 把检测到的 SM 版本供 qualification；BF16 Qwen3 准入仍架构无关。
- **内容详情**：`SOURCE_IDENTITY_FORMAT_VERSION` 3→4；quantization fingerprints 把 `exclude_modules` patterns 当 set（global 与 per-layer 均）；改动 12 文件（+1186/-53）。**当前 head 无完成测试/CI 结果**（提供的 CI 评论为更早 commit 的混合结果，不确立当前 head 状态）。
- **应用场景**：Qwen3 dense 的 FP8 block-scale 权重服务（SM90）。
- **研判**：【事实】PR 给 SM90-only profile 与版本/指纹变更；CI **未全绿/未确立当前 head 状态**（明确注明）。【观点】属「FP8 block-scale 权重准入」（第 6/10 条），SM90-only；CI 状态未确立，记为官方一手（描述级/CI 未确立）。**证据等级**官方一手（描述级，CI 未确立）。

### 4.3 `#21795` [fix] Make KVCacheManagerV2 teardown safe without a CUDA context（KVCacheManagerV2 析构安全，健壮性）
- **主题**：`KVCacheManagerV2` 期望显式 `shutdown()`；当 manager 被最后引用持有者释放（pytest 下 cyclic GC 在任意线程）时，`~KvCacheManager()` 跑 `shutdown()`、`~KvCache()` 跑 `close()` 在可能无 CUDA context 的线程 → 每个 driver 调用失败 `CUDA_ERROR_INVALID_CONTEXT`(201)；析构只能经进程级 poison latch 上报，一个泄漏 manager 变成之后每次 KVCacheManagerV2 调用的 `CorruptedError`，host tier 停止 unmapping、终以 `HostOOMError`/`CuOOMError`（每 pytest 进程约 180 连续失败）。#20816/#21158 治标，本 PR 治本。
- **内容详情**：新 `CudaContextScope`（push/pop CUDA context，已当前则 no-op）；manager 记录构造时的 context 并拒无 context 构造。
- **应用场景**：TRT-LLM KVCacheManagerV2 在引用计数释放/无 CUDA context 线程下的安全析构。
- **研判**：【观点】属「KV cache manager 析构健壮性」——修无 CUDA context 线程下 teardown 级联失败（约 180 连续失败/进程），治本而非逐路径补；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。

### 4.4 `#21508` [fix] Stop scheduling cancelled disaggregated requests（解聚取消请求调度，健壮性）
- **主题**：停止 eligible cancelled CTX/GEN 请求进入另一个 model batch；保留先前 model/transfer 资源受保护直到正常清理安全退役。
- **内容详情**：scheduling gate 把 cancelled work 排除出 scheduler candidates（非所有权）；cancellation boundaries 保留 previous-batch 请求（含无 sampler 输出的 context chunks）并重用正常 post-batch 清理与 terminal responses；overlap bookkeeping 仅清当前 forward 缺席请求的 harvested markers，deferred GEN 从悬挂点续跑。
- **应用场景**：TRT-LLM 解聚（disaggregated）服务的请求取消处理。
- **研判**：【观点】属「解聚请求取消健壮性」——避免 cancelled 请求再进 batch、保资源清理有序；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。

### 4.5 `#20903` [fix] Fail disagg gen requests whose grammar matcher cannot be created（解聚 grammar matcher 创建失败，健壮性）
- **主题**：解聚中 gen worker 在 `GuidedDecoder._init_disagg_gen_requests` 于 model forward 前为每个 gen-only 请求建 grammar matcher，原 `grammar_matcher_factory.create()` 无 per-request 错误处理 → 工厂异常逃出 executor loop 终止它、之后该 worker 所有请求失败；context 请求在 `_build` 建 matcher 已 catch。现逐请求 catch（同 `_build`）。
- **内容详情**：slot 不设 matcher、rollback counters 清空使 rollback 跳过它；请求经既有 `(request_id, error_msg)` failed-requests 列表上报，executor 转 error response；同 batch 其他请求保 matcher 仍 guided。`init_failed_requests` 依赖每次 `init_disagg_gen_requests` 后跟 `_build`。
- **应用场景**：TRT-LLM 解聚 gen 请求的 guided decoding（grammar）创建失败隔离。
- **研判**：【观点】属「解聚 guided decoding 健壮性」——单请求 grammar 创建失败隔离，避免拖垮整个 worker；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。

### 4.6 本期其余 TRT-LLM 要点（标题/正文级）
- **量化 / 模型 / 性能**：`#21287`（见 4.1）、`#20766`（见 4.2）；`#20984`（Cosmos3 per-request sigma grid selection feat）；`#21026`（pass chat_template_kwargs in JSON-schema serve example body，fix）；`#21631`（RADIO video_embedder weights 在加载时而非每视频校验，fix）；`#21876`（Update Qwen3.5 checkpoint loader test API，fix）。
- **解聚 / 调度 / 健壮性**：`#21508`（见 4.4）、`#20903`（见 4.5）、`#21597`（unittest/disaggregated fixed by #21597 的 unwaive，test）、`#21213`（保持编译 Helix NCCL all-to-all 输入 contiguous，fix）、`#21249`（disagg test server 绑 kernel-assigned port，fix）、`#21798`（GLM tool parsers 交付 schema 未类型化的参数，fix）、`#21938`（waive test_executor_loop_fails...，infra）。
- **文档 / 构建 / infra**：`#21910`（Clarify Modeling V2 rules for V1/native/routing）、`#21603`（Relax Modeling V2 naming 并按变更范围选 V2 测试）、`#21813`（scripts/bump_version.py 设版本并查每条版本记录）、`#21325`（引入 Pyrefly 做 type check）、`#21354`（Refresh PrimTS source onto FlashInfer main 2026-11-07）、`#21817`/`#21810`/`#21918`/`#21920`/`#21905`/`#21898`/`#21938`（pre/post-merge waive 与 JUnit/artifactory/OpenSearch 迁移等 infra 多项）。

---

## 5. llama.cpp：2026-11-09 提交（29 条）—— Granite NVFP4 scales 静默错误铁证 + Metal flash-attn OOB + MTP device buffer + 多模态/PLE

### 5.1 `#31449` [models] Attach Granite NVFP4 weight scales（Granite NVFP4 静默错误铁证，强化第 4/6 条）
- **主题**：修 #31448——`llama_model_granite` 在 `build_ffn()` 把 FFN weight scales（`ffn_{up,gate,down}_s`）传 NULL，致 NVFP4 checkpoint 的 per-tensor scales 被加载但未应用，FFN 输出被全局 scale 偏置、产出百万级垃圾（实测 perplexity）。
- **内容详情**：模型 `ibm-granite/granite-4.2-8b-nvfp4`（convert_hf_to_gguf.py 转换）。PPL：NVFP4 修前 **4,899,652.6** → 修后 **18.2424**；Q8_0 参考 17.9163、Q4_K_M 参考 18.5038。CUDA 与 CPU 均得相同垃圾结果。类似 Mistral3 的 #23629 修法。
- **应用场景**：Granite NVFP4 GGUF 的 FFN 量化正确性。
- **研判**：【事实】PR 给 PPL 4,899,652.6 → 18.24 的量化铁证（修前百万级垃圾、无报错）。【观点】属「量化 scales 漏接静默错误」（第 4/6 条）——per-tensor NVFP4 scales 未前向应用致输出偏离参考两个数量级，纯静默错误；与往期 MoE/量化静默债同族。**证据等级**官方一手（含 PPL 铁证）。

### 5.2 `#31662` [metal] guard flash-attn vec output scratch zeroing（Metal flash-attn OOB 写，强化第 4 条）
- **主题**：`kernel_flash_attn_ext_vec` 中 `so4` 指每 SIMD group 的 threadgroup scratch buffer；zero-init 循环前指针按 lane index 偏移，但 32 lanes 全跑该循环。当 `DK=DV=64, Q=1, NE=4` 时某 lane 可能访问 index 39，而其 SIMD group 仅含 32 元素。
- **内容详情**：加条件 `(DV4/NL % NW == 0) || tiisg < NL` 约束 zero-init 循环，与后续 `so4` 更新条件一致。M2 Mac / macOS 27.2 的 Metal shader validation：head-size 64 下报 OOB 写 **109 → 0**。
- **应用场景**：Metal 后端 flash-attn vec 输出的 scratch 零初始化。
- **研判**：【事实】PR 给 Metal shader validation 报 OOB 写 109→0（head-size 64）。【观点】属「Metal 后端静默越界写」（第 4 条）——特定形状下 scratch buffer OOB 写，修复前无报错；Granite/Metal 等模型须核对本修。**证据等级**官方一手（含 Metal validation 数字）。

### 5.3 `#31927` [mtp] read the target hidden states from a device buffer（MTP device buffer 提速，第 5 条旁支）
- **主题**：接 #31506 把 MTP 纳入；target 把未掩码 `h_nextn` 行写 device buffer 而非 host 内存，MTP draft 经 `get_rows` 按 per-token state rows 读；仅每序列 pending 行过 host。
- **内容详情**：所有测试模型 prefill **快 4–10%**。（描述级，无逐模型表。）
- **应用场景**：llama.cpp MTP 投机解码的 target hidden states 读取。
- **研判**：【事实】PR 给「所有测试模型 prefill 4–10% 快」。【观点】属「MTP 投机解码提速」（第 5 条旁支），device buffer 替代 host 内存减少 host 往返；无逐模型表，记为官方一手（含笼统 4–10%）。**证据等级**官方一手（含笼统提速，无逐模型表）。

### 5.4 `#31185` [feat] external PLE file loading（PLE/engram 外部文件加载，feature）
- **主题**：新 `--shard` 参数并支持从多 shard 位置加载模型（常见用例：从独立 GGUF 加载 PLE/engram 文件）；转换支持构建 PLE-less 与 PLE-only GGUF。
- **内容详情**：支持 Gemma4/Qwen3.8Next/GLM5.3Flash 等所有支持 PLE 的模型；多 shard 可组合不同提供方的模型分片与公共模型片段。用法：`llama-server -m ...-00001-of-00003.gguf --shard ...-00003.gguf` 或 `--ple ...-engrams.gguf`（`--shard`/`--ple` 为别名）。可选 flag 覆盖 shard 文件数检查。
- **应用场景**：llama.cpp 多 shard / PLE（engram）文件加载。
- **研判**：【观点】属「模型分片与 PLE 外部加载」feature，方便跨盘/跨提供方组装模型；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。

### 5.5 `#31965` [mtmd] support gemma 4 input audio stream（Gemma4 音频流多模态，feature）
- **主题**：Gemma4 音频 encoder 为 causal，复用 `mtmd_audio_stream` 基建；初用 `step_ms=320ms`、后改 `step_ms=480ms` 提速。
- **内容详情**：Worst-row cosine（stream vs non-stream）：Clip ≤0.3s 各后端 1.000000；3s random pushes Metal 0.999985 / CUDA 0.999913 / CPU 1.000000；30s random pushes Metal 0.999593 / CUDA 0.999555 / CPU 1.000000；76s random pushes Metal 0.992487 / CUDA 0.968441 / CPU 1.000000。
- **应用场景**：llama.cpp Gemma4 音频输入流式多模态。
- **研判**：【事实】PR 给 worst-row cosine 对照（76s CUDA 0.968441 最低）。【观点】属「多模态流式音频」feature，长流（76s）下 CUDA 最低 0.968、仍有偏差；记为官方一手（含 cosine 表）。**证据等级**官方一手（含 cosine 表）。

### 5.6 `#26145` [server] Responses API: Convert text.format JSON schema（Responses API JSON schema 转换，feature）
- **主题**：把 Responses API 的 `text.format` JSON schema 转换为对应的 Chat Completions `response_format`。
- **内容详情**：`test-chat.cpp` 加测试通过；AI 使用披露：ChatGPT 用于代码探索与逻辑分析，所有代码编辑与验证手工完成。
- **应用场景**：llama.cpp server 的 Responses API ↔ Chat Completions schema 兼容。
- **研判**：【观点】属「server Responses API 兼容」feature；测试覆盖转换；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。

### 5.7 本期其余 llama.cpp 要点（标题/正文级）
- **构建 / 后端 / 多模态**：`#31978`（realtime api 用 common_chat_input）、`#31763`（oai-compat `/v1/audio/speech` endpoint）、`#31964`（mtmd_helper_resampler）、`#31952`/`#31949`/`#31936`（ui sidebar retry/fold/preview）、`#31940`（tests keep dummy ffn.act scales away from zero）、`#31874`（tests bound test_rope tolerance by f32 angle resolution）、`#31824`（ggml-zdnn 接受非 zdnn tensors、每 op 后失效 compute tensors）、`#31905`（ggml-zdnn 修 release binary 不检 backend lib）、`#31935`（opencl include climits）、`#31783`（opencl f32 MUL_MAT_ID）、`#31567`（ci gfx1151 ROCm job on Strix Halo Devlab）、`#26014`（Windows 非缓冲模型加载）、`#30712`（ggml TRI backward pass）、`#30802`（ggml YaRN truncation on openvino/sycl/vulkan/webgpu/rpc）、`#31126`（sycl cache device-id）、`#31419`（ggml-backend 不上传未读输入）。
- **正确性 / 量化 / MTP**：`#31449`（见 5.1）、`#31662`（见 5.2）、`#31927`（见 5.3）、`#28697`（server fix response_format json_schema 非嵌套包装）、`#31027`（server new spec dec functionality test）、`#30711`（metal ggml_tri 选而非按比较缩放）、`#31655`（Pocket TTS 经 API 设 temperature 通用支持）。

---

## 6. 累计结论（继承 1–11，新增第 12 条）

- **第 1 条（调度器隐式阻塞的定量画像）**：本期无新增铁证。沿用。
- **第 2 条（发版窗口的双向性：发版当天合入 ≠ 在版内）**：本期窗口内无新发布；vLLM 仍为 v0.33.0rc3（2026-11-02）、SGLang v0.5.23（2026-11-02）、TRT-LLM v1.4.0rc3（2026-11-06，prerelease）、llama.cpp v0.9.0（2026-11-04）+ nightly b12236/b12237（2026-11-09，prerelease）。沿用。
- **第 3 条（默认值成为主要风险载体）**：本期无往期级「默认变更/移除」头号事件；窗口内多为正确性/性能/重构（vLLM #66546 EAGLE3+PP feature、TRT #21508/#20903 解聚健壮性 fix）。**默认值/移除风险主线本期无新增铁证，节奏暂缓**。
- **第 4 条（异构后端静默错误高频）**：本期两处铁证——llama.cpp `#31449`（Granite NVFP4 FFN scales 未应用，PPL 4,899,652→18.24，静默垃圾）、`#31662`（Metal flash-attn vec output scratch OOB 写，Metal validation 109→0）；vLLM `#60531`（SM100 Triton unified attention batch invariance，head-size-256 10 failures→20 passes、Gemma3 逐位一致）。**异构/量化静默错误持续密集，本期以「已修复的静默偏差」为主**。
- **第 5 条（前缀缓存 × MoE 路由 × 投机解码交叉债务）**：本期以 PD/投机正确性为主——SGLang `#47406`（DSv4.1 bounded replay + CP + DSpark prefill 输出 gather 行数不符，显式开启组合触陷阱）、vLLM `#66486`（MoE LoRA 在 DP+EP 三处本地批次假设 → EP dispatch 错 slot）、`#66546`（EAGLE3 aux hidden + PP）、llama.cpp `#31927`（MTP device buffer 提速 4–10%）。**投机解码/PD 正确性跨引擎延续**。
- **第 6 条（量化路径静默错误，与异构后端并列的第二类高频静默错误）**：llama.cpp `#31449`（Granite NVFP4 scales 未应用，PPL 百万级垃圾）、`#31662`（见第 4 条）；vLLM `#64315`（Qwen4Exp 在 DeepGEMM MegaMoE 的 MXFP8/MXFP4 requantize）、`#65406`（MoE `w2_bias` 在 TP>1 计 tp_size 次 → token 错）；SGLang `#49388`（MegaMoE MXFP4 2304 intermediate 形状适配）、`#48405`（AMD fp8 KV 写入融合）；TRT `#21287`（FP8 GDN 投影/激活量化融合）、`#20766`（Qwen3 FP8 block-scale 准入，CI 未确立）。**量化静默债跨引擎延续（本期多为已修复/已适配项）**。
- **第 7 条（专家张量/路由归属与分页漏接静默错误）**：SGLang `#43675`（Kimi-K3 派生权重缓冲跨 CUDA graph reload 保持，修 reload 后旧存储读取；ROCm/量化 loader 变体未覆盖）属权重/专家缓冲跨 reload 正确性，与第 7 条呼应。本期无往期级「专家分页静默用错值」铁证。沿用并补充。
- **第 8 条（KV 容量/复用智能：容量自检、准入、传输/卸载正确性）**：vLLM `#56192`（KV offload OBJ 层 RDMA-capable，NIXL 1.5.2 S3-over-RDMA）、`#65888`（HiSparse P/D import 入 GPU resident 页，零 NIXL 改动，ITL 10.40ms/NIXL bytes 1×）、`#65559`（ROCm SDMA 批量拷贝，decode 旁 +122%→+0.4~3.9%）；SGLang `#27267`（HiCache host pool hugepage 预检/分配）、`#45727`（HiSparse 共享 KV 源）、`#48405`（AMD fp8 KV 写入融合）；llama.cpp `#31185`（PLE 外部文件加载）。**KV 卸载/复用跨引擎延续且扩面（RDMA OBJ 层 / GPU 驻留页 / hugepage / 共享 KV 源）**。
- **第 9 条（长上下文 int 溢出为静默正确性新子项）**：本期无新增 int 溢出铁证。沿用。
- **第 10 条（量化/MXFP8 KV 正确性簇）**：vLLM `#64315`（Qwen4Exp MXFP8/MXFP4 on DeepGEMM MegaMoE）、`#65559`（ROCm fp8 KV 卸载）；SGLang `#48405`（AMD fp8 KV 写入融合）、`#49388`（MegaMoE MXFP4 2304）；TRT `#21287`（FP8 GDN 投影）、`#20766`（Qwen3 FP8 block-scale 准入）。沿用并强化。
- **第 11 条（新旗舰模型的上下文并行/解码栈 CP 成为一等公民）**：本期无往期级「新 CP feature 全栈落地」铁证（2026-11-08 头号事件为 SGLang Kimi-K3.1 CP 全栈）；本期 Kimi-K3 以 `#43675` 派生权重缓冲跨 reload 正确性修复为主（巩固而非新增 CP 能力）。属已确立观察，本期无新增强化，节奏暂缓。
- **第 12 条（DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 模型落地的主路径）【本期新增】**：vLLM `#64315`（Qwen4Exp routed experts 跑 DeepGEMM MegaMoE，MXFP8/MXFP4 自动 requantize，EP 必需）+ SGLang `#49262`（DSv4.1 MegaMoE input 由 fused decode mHC post 直写 epilogue、位精确、GSM8K 96.66/97.04）+ `#49388`（MegaMoE MXFP4 2304 intermediate 形状适配启用 W4A4）。两引擎在同一窗口把「新旗舰 MoE 模型 × 量化（MXFP4/MXFP8/W4A4）× DeepGEMM MegaMoE」推进到可用/融合，标志量化 MoE 后端从「能跑」到「输入融合 + 形状适配 + 自动 requantize」的深化。属新兴观察，需后续窗口复现确认是否成为稳定范式。

---

## 7. 社区侧（HN Algolia，2026-11-09 窗口）—— 引擎相关高信号弱，仅 1 条 Show HN 决策模型加速

HN Algolia（2026-11-09 窗口，epoch 1794182400–1794268800，HTTPS）引擎关键词高信号弱：
- `vllm` 570 命中均为评论/模糊「LLM」匹配，0 条 story；`sglang` 3 命中、**1 条 story**；`llama.cpp` 3 命中 0 story；`TensorRT-LLM` 0；`KV cache` 4 命中 0 story；`speculative decoding` 4 命中 0 story；`LLM inference` 49 命中 2 story（均为 Ask HN 学习类提问，低信号）；`quantization inference` 1 命中 0 story；`GPU inference` 37 命中 0 story。
- **Show HN: Flide: Make Jev-like decision models up to 9.5x faster**（2026-11-09T04:43:12Z，119 points）：把「Jev-like」决策模型加速最多 9.5× 的第三方工具/方法；与推理引擎 KV 复用主题无直接关系，属通用推理/决策模型加速，**二手线索/未核验**（未核验其可用性、精度或性能声称）。
- 其余 `sglang`/`llama.cpp`/`KV cache`/`speculative decoding` 命中均为评论级，无指向四家引擎的官方更新。

**研判**：【事实】上述为 2026-11-09 当日 HN 发布的标题/链接与命中数（HTTPS 抓取，epoch 窗口经 `date -u -j -f` 生成）。【观点】窗口内无引擎官方更新类高信号；仅「Flide」决策模型加速（119 分 Show HN）与推理/内核生态沾边——**均为二手线索/未核验**，未核验其对主流引擎的可用性、精度或性能声称。**证据等级**二手线索/未核验（HN 公告 + 外部 GitHub，未独立验证）。

---

## 8. 趋势判断与可执行清单

**趋势判断**：
1. **DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 模型落地的主路径（新观察，第 12 条）**：vLLM `#64315`（Qwen4Exp routed experts 跑 DeepGEMM MegaMoE，MXFP8/MXFP4 自动 requantize）+ SGLang `#49262`（DSv4.1 MegaMoE input 由 fused decode mHC post 直写、位精确、GSM8K 96.66/97.04）+ `#49388`（MegaMoE MXFP4 2304 intermediate 适配）。两引擎同一窗口把「新旗舰 MoE × 量化 × DeepGEMM MegaMoE」推进到可用/融合，标志量化 MoE 后端深化。
2. **KV Offload / HiSparse 跨引擎扩面（第 8 条）**：vLLM `#56192`（RDMA-capable OBJ tier，NIXL S3-over-RDMA）、`#65888`（HiSparse P/D import 入 GPU resident 页、零 NIXL 改动、ITL 10.40ms/NIXL bytes 1×）、`#65559`（ROCm SDMA 批量拷贝 +122%→+0.4~3.9%）；SGLang `#27267`（HiCache host hugepage）、`#45727`（HiSparse 共享 KV 源）、`#48405`（AMD fp8 KV 写入融合）。KV 卸载从「能卸载」推进到「RDMA 远程层 / GPU 驻留页 / hugepage / 共享 KV 源」。
3. **静默 / 量化正确性铁证再现（第 4/6 条）**：llama.cpp `#31449`（Granite NVFP4 scales 未应用，PPL 4,899,652→18.24，静默垃圾）、`#31662`（Metal flash-attn OOB 写 109→0）；vLLM `#60531`（SM100 batch invariance 头-size-256 10 failures→20 passes）、`#65406`（MoE `w2_bias` TP>1 计 tp_size 次 → token 错）、`#66486`（MoE LoRA 在 DP+EP 正确性）。**量化/异构后端静默错误持续密集，本期多为已修复项**。
4. **PD / 投机解码正确性（第 5 条）**：SGLang `#47406`（DSv4.1 bounded replay + CP + DSpark prefill 输出 gather 行数不符，显式开启组合触陷阱）、vLLM `#66486`（MoE LoRA 在 DP+EP）、`#66546`（EAGLE3 aux hidden + PP）、llama.cpp `#31927`（MTP device buffer 提速 4–10%）。
5. **SGLang mem_cache 统一内存布局规划器重构（架构级）**：`#48011`–`#48014` 栈把分散的容量/准入/回收规则收为单一 layout planner（纯函数、尚未接线），为统一内存 KV 管理的后续切换打底。
6. **TRT-LLM v1.4.0rc3 稳定化（infra/CI 为主）**：`#21287`（FP8 GDN 投影/激活量化融合）、`#20766`（Qwen3 FP8 block-scale 准入，CI 未确立）、`#21795`（KVCacheManagerV2 无 CUDA context 析构安全，治本）、`#21508`/`#20903`（解聚取消/grammar 创建失败隔离）；rc3 仍为 prerelease。
7. **llama.cpp 多模态 / PLE / 量化正确性**：`#31449`（Granite NVFP4 scales 铁证）、`#31662`（Metal OOB）、`#31965`（Gemma4 音频流，76s CUDA 最低 0.968）、`#31185`（PLE 外部文件加载）、`#31927`（MTP device buffer）。

**可执行清单（建议）**：
- **P0（量化静默自查）**：llama.cpp Granite/NVFP4 类模型核对 `#31449`（FFN weight scales 必须前向应用，修前 PPL 百万级垃圾）；Metal 后端 head-size 64 flash-attn 核对 `#31662`（scratch OOB 写，修复前 Metal validation 报 109 越界）；vLLM Qwen4Exp 量化 MoE 核对 `#64315`（MXFP8/MXFP4 自动 requantize 路径，公开 checkpoint 缺位靠单测覆盖）；SGLang DSv4.1-Flash W4A4 MegaMoE 核对 `#49388`（2304 intermediate 形状适配后启用）。
- **P1（KV Offload / HiSparse）**：vLLM KV offload OBJ 远程层核对 `#56192`（NIXL 1.5.2 钉版、4 GiB 分块 DRAM 注册）；HiSparse P/D 入 GPU 页核对 `#65888`（零 NIXL 改动、ITL 10.40ms、NIXL bytes 1×）；ROCm KV offload 批量拷贝核对 `#65559`（HIP≥7.16 走 SDMA，decode 旁拷贝 +122%→+0.4~3.9%）；SGLang HiCache host hugepage 核对 `#27267`（prefer/required 模式，避免误报内存不足）。
- **P1（PD / 投机 / MoE 正确性）**：SGLang DSv4.1 显式开启 bounded replay + CP + DSpark 核对 `#47406`（prefill 输出 gather 行数陷阱）；vLLM MoE LoRA 在 DP+EP 核对 `#66486`（三处本地批次假设）、MoE `w2_bias` 在 TP>1 核对 `#65406`（修前 token 4 胜）；vLLM EAGLE3 + PP 核对 `#66546`（aux hidden 跨 PP 传递与 layer 0）。
- **P2（跟踪）**：DeepGEMM MegaMoE 量化后端在两引擎的实测吞吐/精度（#64315 无端到端 t/s、#49262 GSM8K 96.66/97.04 含 #49321、#49388 形状适配）；SGLang mem_cache layout planner 接线后的容量/准入一致性（#48011–#48014）；TRT-LLM `#20766` Qwen3 FP8 block-scale 准入 CI 状态（当前 head 未确立）；llama.cpp MTP device buffer 逐模型提速表（#31927 仅笼统 4–10%）。

> 证据等级说明：本档案「官方一手」指来自四家官方仓库的提交/PR/发布元数据与 PR 正文（经 GitHub issues API 抓取的 markdown）；「二手线索/未核验」指来自社区公告/HN/外部 GitHub 且未经一手来源确认的内容（如 Flide）。所有性能/精度数字均标注来源 PR 或注明「描述级/CI 未确立/作者报告未独立验证/无 t/s 对照/部分截断」，未外推未呈现数字。本窗口为 2026-11-09（周一），提交量较周末回升（vLLM 99 / SGLang 92）；llama.cpp 在 2026-11-09 仅推进 nightly b12236/b12237（prerelease），无新正式版本。
