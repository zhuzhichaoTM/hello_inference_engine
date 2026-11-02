# 大模型推理优化每日热点档案

> 档案日期：2026-11-02（本自动化于该日执行）
> 关注窗口：2026-11-01（周日）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + PR 正文（pulls API，markdown）+ HN Algolia（HTTPS）
> 数据质量：四家提交 API 均 HTTP 200；PR 正文 21 篇已抓取；HN 窗口内引擎相关无高信号故事（vllm 14 条为「LLM」模糊匹配、非引擎更新）。

## 1. 版本状态与本期头号事件：静默正确性主线在周日低活跃下延续（SGLang int64 KV offset 修复 + 跨引擎 KV dtype 量化正确性 + PD 解耦健壮性簇）

**提交计数（2026-11-01 UTC，均 HTTP 200；2026-11-01 为周日，vLLM / TRT-LLM / llama.cpp 显著低于工作日、无第二页）**：vLLM **10** / SGLang **48** / TensorRT-LLM **7** / llama.cpp **3**。

**发布状态**（四家窗口内均无新正式版本）：
- vLLM：最新正式版仍为 **v0.32.0**（2026-10-18T06:50:48Z）；nightly 持续。
- SGLang：最新正式版 **v0.5.22**（2026-10-19T02:55:19Z）；窗口内无新 tag（0.5.23 仍未发布）。
- TensorRT-LLM：最新仍为 **v1.4.0rc2**（2026-10-28T15:51:34Z，prerelease），窗口内无更新。
- llama.cpp：窗口内发布 nightly 标签 **b12028–b12032**（2026-11-01，均为 prerelease 自动构建）。

**本期头号事件（静默正确性主线延续，周日低活跃但仍有铁证级修补）**：
1. **int32 溢出家族再落 SGLang extend attention（强化累计结论第 9 条）**：SGLang `#47029` 证实 `extend_attention.py` 的 `_fwd_kernel` / `_fwd_kernel_unified` 用 **int32 `kv_indices`** 算 KV slot × stride，乘积在 `slot >= 2^31 / stride`（8 KV heads × 128 head dim 下为 **2,097,152 tokens**）回绕——轻则静默读错 KV、重则 `_fwd_kernel` 内存错误；**aiter 的 CUDA-graph `cuda_graph_kv_indices`（int32）被 legacy target-verify 路径（NGRAM、EAGLE topk>1）直接使用，长时运行的 aiter 服务 KV 池超限后必崩**。MI355X（Llama-3.1-8B-FP8-KV + NGRAM）在 GSM8K 第 430–485 题间反复 `Memory Fault`；main 两子测试 max abs diff 1.09/1.04，分支将 slot 先 cast int64 后通过。**int 溢出静默/崩溃类已从 vLLM（Qwen3.5 #63761、Qwen4Exp #64820）、llama.cpp（Hexagon #31442）扩展到 SGLang extend attention 与 aiter 投机 verify 路径，横跨云/边/投机解码。**
2. **跨引擎 KV dtype 量化正确性（强化累计结论第 4 / 6 / 8 条）**：vLLM `#56155` 在 attention runtime **逐层解析 KV-cache 编码**（FP8 vs NVFP4，新增 `QuantizationConfig.get_kv_cache_dtype(prefix)` hook、alias 归一与组内不兼容编码拒绝）；TRT-LLM `#19864` 修 **encoder-decoder 跨注意力 KV 在 FP8 KV 下越界**——cross-attn 写未量化 K/V，但其 pool 继承了 self-attn 的 KV 量化，导致分配与写入 element size 不一致（CUDA 错误或错误输出），修后 cross-cache 保持 model dtype、decoder self-attn 保留 FP8/NVFP4。**量化 KV 在「逐层 dtype 解析」与「enc-dec 跨池 dtype」两个维度同时暴露，是 #6（量化路径静默错误）与 #8（静默正确性升维）的同源铁证。**
3. **SGLang 解耦（PD）健壮性簇 + AMD gfx950 MiniMax 低比特栈落地**（强化累计结论第 5 / 8 条）：`#46395` 让 PD decode radix cache 把 decode KV 写入 linker 并**跳过 prefill 已存前缀**（Kimi Linear 4-turn e2e 验证），`#47716` 修 overlap 调度循环在「下一 batch 已排程后才决定提前处理上一结果」导致已完成请求仍被启动——与 10 月密集的 PD 修复同源；AMD 侧 `#46690`/`#46691`/`#46692` 把 MiniMax-M3.1-Flash 的 sparse attention / NVFP4 KV / DSpark verify、MoE+block-fp8 GEMM、gfx950 TP4 JIT quick all-reduce 落到 main（perf 数字随 `#46693` 后续合入，本组记为描述级）。

---

## 2. vLLM：2026-11-01 提交（10 条）——逐层 KV dtype 解析 + DSv3.2 跨 PP indexer top-k + KV Offload GC 优化

### 2.1 `#56155` 在 attention runtime 逐层解析 KV-cache 编码（量化 KV dtype 正确性，头号相关）
- **主题**：把 KV-cache 编码解析从全局收拢到 attention runtime 的逐层粒度，让 FlashInfer 在全局 dtype=`auto` 时由已加载的 attention 层推断编码，显式全局设置仍优先；为 `#56156`/`#56116` 的前置。
- **内容详情**：新增默认 `None` 的 `QuantizationConfig.get_kv_cache_dtype(prefix)` hook；复用已有 storage-`dtype`/`kv_quant_mode` 分组（FP8 vs NVFP4）；归一等价读编码（`fp8`/`fp8_e4m3`、NVFP4 scale-search 变体），组内拒绝不兼容编码；滑动窗口 MLA spec 合并时保留物理量化模式。GPU runner 与 `AttentionGroup` 均不变。**GPU-backed 构造 pytest 用例尚未运行（local full pytest 停在缺失 `tblib`）。**
- **应用场景**：多模型 / 混合 KV dtype（FP8/NVFP4）服务；逐层 KV 量化一致性。
- **研判**：【事实】PR 给出 hook 设计、alias 归一与组内拒绝，含 CPU 级 resolver/merge 36 组合测试、sliding-window MLA 回归（base 失败、本 PR 通过）。【观点】属「KV dtype 量化静默正确性」主线（第 4/6/8 条）的基础设施级修补——逐层解析可避免全局 dtype 误套到个别层；但 GPU 用例未跑，落地前应补 GPU 验证。
- **来源**：vLLM PR [#56155](https://github.com/vllm-project/vllm/pull/56155)（merge 2026-11-01T15:03:45Z）。
- **证据等级**：官方一手（含 CPU 级 resolver 测试，GPU-backed 构造用例未跑）。

### 2.2 `#59961` DSv3.2：把共享 indexer top-k 跨流水线阶段传递（MoE / indexer 正确性，跨 PP）
- **主题**：GLM-5.x / DeepSeek-V3.2 类模型复用上一层 indexer top-k；当某 PP stage 起始层恰为「复用层」时，该 stage 读到**陈旧 top-k**，导致 GSM8K 掉点。
- **内容详情**：随中间张量一并下发 top-k（共享 helper，同时修 HY-V4、拒绝不支持的 MiniMax-M3 AMD 情形）。`GLM-5.3-NVFP4 PP2×TP2`：GSM8K **0.879 → 0.908**；带 gather-based DCP（`#59916`）更明显 **0.820 → 0.908**；ROCm 8×MI350X `PP2×TP4` **0.886 → 0.916–0.931**。根因：`index_topk_freq=4 offset 3`，默认 PP2 切分点在 layer 39（落在组内）。
- **应用场景**：GLM-5.x / DSv3.2 在 PP>1（含 DCP）部署；indexer top-k 复用模型的跨 stage 一致性。
- **研判**：【事实】4×B300 + 8×MI350X 双平台实测 GSM8K 表，掉点修复明确。【观点】属「前缀缓存 × MoE 路由 × 跨 PP/DP 交叉债务」（第 5 条）的 indexer 维度实例——跨 stage 状态传递遗漏是隐蔽精度退化的典型来源，PP 切分点应避开 top-k 复用组边界（强化第 5、8 条）。
- **来源**：vLLM PR [#59961](https://github.com/vllm-project/vllm/pull/59961)（merge 2026-11-01T06:47:36Z）。
- **证据等级**：官方一手（含 4×B300 与 8×MI350X 实测 GSM8K 表）。

### 2.3 `#63927` KV Offload：把 offload 事件元数据移出循环 GC（性能，含实测）
- **主题**：`OffloadingEventsTracker` 为每个驻留 chunk 保留一个 `_OffloadEventMetadata`，每个条目挂 3 个被 Python cyclic GC 跟踪的对象；大二级层时这些条目占 EngineCore 长寿命堆主体，每次 full GC 让调度器停顿 >100 ms。
- **内容详情**：把每个条目从 3 个 GC-tracked 对象降到 0（不进 cyclic GC）。冷缓存 KV-offload 服务：full-collection 停顿 **106 → 31 ms**、ITL p99 **119 → 55 ms**、req/s **+7%**（与禁用 GC 仅差 1.5% 内）。驻留簿记仍为 `set` 语义、事件不变。
- **应用场景**：大二级层（KVCR / 对象存储）的 KV Offload 在线服务，降低 GC 引起的 ITL p99 抖动。
- **研判**：【事实】PR 给出冷缓存实测表（req/s 13.5→14.5、ITL p99 119→55ms、扫描对象 252,147→49,209），模型 forward 与 GPU↔CPU 拷贝为 mock。**mock 不含真实 GPU**，但 GC 路径与事件不变，结论可信。【观点】属 KV 卸载健壮性/性能（第 8 条同源），KV Offload 上线前建议把「二级层规模下的 GC 停顿」纳入验收。
- **来源**：vLLM PR [#63927](https://github.com/vllm-project/vllm/pull/63927)（merge 2026-11-01T18:27:47Z）。
- **证据等级**：官方一手（含冷缓存实测表，模型 forward/GPU 拷贝为 mock）。

### 2.4 `#64955` MoE：把 padding 跳过融合进 grouped 与 Muse prepared routing（性能）
- **主题**：在 Muse Spark 的 prepared NVFP4 routing 路径跳过 CUDA-graph padding，并把已有 grouped-routing padding mask 融合进 CUDA top-k 核——padding 产生的 `-1` expert id 与零权重不再需要独立的 CUDA masking launch。
- **内容详情**：Muse 的 TRT-LLM 路径传入原生 padding mask 给融合 router；grouped routing 在全部 4 个 CUDA kernel 族处理 padding（XPU/AITER 接口兼容）。8 个 isolated grouped-routing 用例把 **5 个 kernel launch 降为 1**，median span **−43.34%~−70.06%**（非整模型加速）；GB200 上 `test_grouped_topk.py`+`test_muse_spark_router.py` **632 passed**，真实权重下 16 个活跃输出与 2,880 个 routing 张量 byte-identical。
- **应用场景**：MoE grouped routing / Muse Spark prepared routing 的 launch 开销削减。
- **研判**：【事实】PR 给出 byte-identical 验证与 632 passed、span 降幅表。【观点】属投机/MoE 路由路径的确定性性能优化，无精度风险（输出逐字节一致），可采纳。
- **来源**：vLLM PR [#64955](https://github.com/vllm-project/vllm/pull/64955)（merge 2026-11-01T07:06:26Z）。
- **证据等级**：官方一手（含 GB200 632 passed 与 byte-identical 验证）。

### 2.5 `#64999` GDN：恢复 CUDA 上 `in_proj_ba` 与 `in_proj_qkvz` 的 overlap（回归修复）
- **主题**：`#64425` 用 `default_unquantized_gemm` 判断把 `in_proj_ba` 与 `in_proj_qkvz` 重叠；但 `#62956`（10-31 合入）把 CUDA 全部 dispatch 改为 `cuda_unquantized_gemm`，导致 overlap 实际未执行——一行回归。
- **内容详情**：把判断改回匹配新 dispatch；新增 `test_gdn_ba_overlap.py` 防止未来回归。
- **应用场景**：Qwen3.5 GDN 模型在 CUDA 上的 projection 融合（省一次 GEMM）。
- **研判**：【事实】PR 描述回归链路（`#62956` 改 dispatch → overlap 失效）与新增测试。【观点】属「默认值/dispatch 变更静默改变行为」（第 3 条）的轻量实例——一行 dispatch 改名即让优化失效，强化「发版/合入须核对行为是否仍生效」。
- **来源**：vLLM PR [#64999](https://github.com/vllm-project/vllm/pull/64999)（merge 2026-11-01T18:27:26Z）。
- **证据等级**：官方一手（含回归测试，1 行修复）。

### 2.6 其余 vLLM 要点（官方一手，标题/正文级）
- `#63695` **KV Offload 在原始存储事件上携带 `ReqContext`**（CPU/FS/OBJ 三层贯通，154+51 passed，transport-only 首步）。证据等级：官方一手（含测试）。
- `#64607` **Qwen3.5-397B-A17B B200 NVFP4 TP2 agentic preset**（Recipe）；`#64910`/`#64909` **MiniMax-M3.1-Flash GB300/GB200 AgentX preset 文档更新**（Doc）。证据等级：官方一手（标题/正文级）。
- 提交计数仅 10 条（周日），其余为 preset/doc 类，无第二页；如需交叉核验可改查 PR merge 口径。

---

## 3. SGLang：2026-11-01 提交（48 条）——int64 KV offset 修复 + AMD gfx950 MiniMax 低比特栈 + PD 解耦健壮性

### 3.1 `#47029` extend attention：用 int64 计算 KV offset（静默正确性，头号）
- **主题**：`extend_attention.py` 的 `_fwd_kernel` / `_fwd_kernel_unified` 从 `kv_indices` 取 KV slot 后乘 KV buffer stride，**int32 `kv_indices` 在 `slot >= 2^31 / stride` 回绕**（8 KV heads × 128 head dim 下为 2,097,152 tokens），核随后 fault，或回绕地址被映射时**静默读错 KV**。
- **内容详情**：在地址运算前把 slot cast 到 int64（覆盖 `PAGE_SIZE==1` 与 paged 分支、任意调用方 dtype）。aiter 的 CUDA-graph `cuda_graph_kv_indices`（int32）被 legacy target-verify 路径（NGRAM、EAGLE topk>1）直接使用；triton 后端自身 buffer 为 int64 不受影响。新增 `test_extend_attention_int32_kv_indices_above_2g`：两核在 int32 slot 高于 `2^31/stride` 下对照低位同数据，main 两子测试失败（max abs diff 1.09/1.04），分支通过。
- **应用场景**：长上下文（>~2.1M token）extend attention；aiter + 投机解码（NGRAM/EAGLE）长时服务。
- **研判**：【事实】PR 含边界测试与 MI355X GSM8K 在 q430–485 反复 `Memory Fault` 复现、main 失败/分支通过对照。【观点】与 vLLM `#64820`/`#63761`、llama.cpp `#31442` 同族，把「int 溢出静默/崩溃」主线（第 9 条）扩展到 **SGLang extend attention 与 aiter 投机 verify 路径**——建议把 2.1M token 级 int32 边界纳入默认发布门禁（强化第 4、6、8、9 条）。
- **来源**：SGLang PR [#47029](https://github.com/sgl-project/sglang/pull/47029)（merge 2026-11-01T04:13:21Z）。
- **证据等级**：官方一手（含 int32 边界测试 + MI355X GSM8K fault 复现）。

### 3.2 `#46690` / `#46691` / `#46692` AMD gfx950 MiniMax-M3.1-Flash 低比特栈（特征 / 性能，描述级）
- **主题**：MiniMax-M3.1-Flash 的 AMD 系列（替代已关的 `#45936`）：`#46690` 注意力、`#46691` MoE/GEMM、`#46692` 分布式，均落 `main`，随后 `#46693`（M3.1 call sites）与 `#46694`（indexer CP）接线；沿用 DeepSeek-V4.1 AMD layout（`#41018`–`#41021`/`#41308`）。
- **内容详情**：
  - `#46690`：**NVFP4 KV（MSA-v1 layout）** 的 Gluon 反量化、`nvfp4_decode_gfx950` 稀疏 decode（每 row 一 program、per-warp online softmax）、verify indexer（fused prefill indexer + 流式 top-k on MFMA）、fp8 O epilogue、QK-norm+RoPE fused、`HiCache` NVFP4 页行拷贝。
  - `#46691`：MoE sort+MXFP4 量化（bit-for-bit 对齐 aiter 多相路径）、shared-expert fold 进 aiter stage-2 归约、router GEMV 升到 512 rows 并与 shared-expert fp8 SwiGLU 一 launch 融合、aiter asm block-fp8 split-K GEMM（`out_zeroed`）、fp8 producers、Gemma RMSNorm producers。
  - `#46692`：sgl-kernel quick all-reduce 的 JIT 副本（wave-major 线布局、TP4 gfx950 仅 INT4/INT8 编解码），`jit_serves()` 仅 gfx950 验证；INT4 在 <`SGLANG_ROCM_QUICK_REDUCE_INT8_BELOW_MB`（默认 64 MiB）时改 INT8（约 1/16 INT4 误差）。**perf 数字来自 `#46693`，本组为 kernel/分布式落地，记为描述级。**
- **应用场景**：MiniMax-M3.1-Flash 在 AMD MI355X(gfx950) 的 sparse attention / NVFP4 KV / DSpark verify / MoE / 低比特 all-reduce 全栈。
- **研判**：【事实】三 PR 给出逐 kernel 落地与 bit-for-bit 对齐 aiter/router 的声明，JIT 仅 gfx950 验证。【观点】属「异构硬件（AMD ROCm）低比特推理加速」主线（第 4 条）——与 10 月 SGLang AMD 簇（aiter 后端、block-fp8、DSA）同源；**性能数字随 `#46693` 合入后才可核验，本期记描述级/未独立验证，须以 #46693 实测为准**。
- **来源**：SGLang PR [#46690](https://github.com/sgl-project/sglang/pull/46690) / [#46691](https://github.com/sgl-project/sglang/pull/46691) / [#46692](https://github.com/sgl-project/sglang/pull/46692)（merge 2026-11-01T22:36:01Z / 23:16:09Z / 23:15:43Z）。
- **证据等级**：官方一手（kernel 系列，perf 数字待 `#46693` 合入，描述级/未独立验证）。

### 3.3 `#46395` 解耦：把 PD decode KV 写入 linker 并跳过 prefill 已存前缀（PD / KV 一致性）
- **主题**：PD 解耦下，decoder 侧 KV 不写外部存储，下一轮 prefill 只恢复上一轮 prompt 并重算上一轮输出；decode radix cache 虽可通过 linker 写这些页，但会**重复上传存储已委托给 prefill 的 prompt 页**。
- **内容详情**：prefill 在 bootstrap info 报 `stores_to_linker`；decode 每次预分配设 `req.linker_offload_skip_prefix_len`（peer 启用 linker 时为 prefill 填的前缀长度、否则 0）。linker 在保留本地 radix 插入的同时裁剪对应传输前缀；完全跳过的节点标记本地已存以抑制重复上传（不等待持久化 ACK）。`process_decode_queue` 在树 offload 挂起时轮询 linker 完成以释放节点锁。模型无关（Full/SWA/Mamba）。
- **应用场景**：PD 解耦 + 外部存储（Mooncake 等）的多轮对话，避免 decode KV 与 prefill 前缀重复上传。
- **研判**：【事实】PR 给出设计（skip prefix 转换为 page-aligned radix-key）与 e2e `test_disaggregation_decode_radix_linker.py`（Kimi Linear，MLA+KDA，1P1D，Mooncake store，4 turns）。【观点】属「PD/KV 一致性」（第 5、8 条）的存储责任委托优化——把前缀存储委托给 prefill、decoder 只补差，降低多轮重复上传；linker 为 best-effort，prefill 写失败仍可能需重算（已显式说明）。
- **来源**：SGLang PR [#46395](https://github.com/sgl-project/sglang/pull/46395)（merge 2026-11-01T12:52:16Z）。
- **证据等级**：官方一手（含 Kimi Linear 4-turn e2e 测试）。

### 3.4 `#47716` 调度器：在排程下一批前决定提前处理结果（调度正确性）
- **主题**：overlap 循环在**下一 batch 已排程后才决定**是否提前处理上一结果，然后对已完成请求仍随该 batch 启动。两个触发：`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`（连续 extend 步，含 DP 转换的 decode 步与混合步）与 spec+grammar（NGRAM，无 grammar overlap 时 EOS/stop finish 搭下一 decode 启动）。
- **内容详情**：`should_process_last_batch_early(last_batch)` 在 `get_next_batch_to_run` 之前、由唯一仍 pending 的结果决定；`event_loop_overlap` 与 `event_loop_overlap_disagg_decode` 复用；`_join_launched_reqs` 的旧 `not r.finished()` 过滤移除。NGRAM worker 按请求判断上一轮已接受 token 是否 splice（避免 overlap 下首个 decode 后旧 decode 行被双 splice）。
- **应用场景**：SGLang overlap 调度（连续 prefill overlap、NGRAM 投机 + grammar）的正确性。
- **研判**：【事实】PR 给出两触发路径与 `should_process_last_batch_early` 在排程前的再定位。【观点】属调度器隐式行为正确性（第 1、8 条）——已完成请求被错误带入下一批是隐蔽的调度正确性债，与 10 月「host sync 移出热路径」同源（性能/正确性耦合）。
- **来源**：SGLang PR [#47716](https://github.com/sgl-project/sglang/pull/47716)（merge 2026-11-01T21:54:37Z）。
- **证据等级**：官方一手（PR 正文，stacks on `#43900`）。

### 3.5 其余 SGLang 要点（官方一手，标题/正文级）
- `#47758` **disagg：给 decode queue 测试调度器补 `tree_cache`**（修 `#46395` 让 `process_decode_queue` 读 `self.tree_cache.linker` 后测试 stub 缺属性致 `base-a-test-cpu` 整分区 fail-fast；CI Base/Extra/AMD 均红，记为描述级/未全绿）。证据等级：官方一手（测试修复，CI 未全绿）。
- `#47180` **[Perf][AMD] DSA：在自有路由复制 DCP query projection**（去掉 query all-gather、a2a 载荷减半；MI355X GLM-5.2-MXFP4 TP4/EP4+DCP4 FP8 KV MTP，GSM8K 0.953→0.945/0.940，extend 2K **−13%**、4.6K 降）。证据等级：官方一手（含 MI355X 实测表）。
- `#47684` **[Perf] DSv4.1 dense indexer logits tiles 取整到整 SM 波**（B300，长上下文 400K keys 处 **0.770×**、500K 处 0.877×，top-k 逐 cell 一致）。证据等级：官方一手（含 B300 实测表）。
- `#39880` **[Feat] HiCache 新增 `fast_file` 本地存储后端**（便携本地 NVMe L3：`readv`/`writev` 直连 host pool buffer、bounded worker pool、原子发布（临时文件+rename）、双水位 LRU 淘汰、`min_free_space` 地板、按模型/布局/主机的命名空间目录避免误清；复用 `file` 后端磁盘格式）。证据等级：官方一手（feature）。
- `#47605` **[Feat] diffusion：LoRA merge cache LRU 淘汰**（低磁盘时先逐出最久未用 store，`get()` 在 store 被并发驱逐时返回 `None` 而非抛 `FileNotFoundError`；8 passed CPU）。证据等级：官方一手（CPU 测试）。
- 解耦/PD 修复簇（其余）：`#47727`（finalize_bootstrap 移出 assert）、`#47725`（bootstrap server loop/runner 在 server thread 启动前初始化）、`#47704`（驱逐死 prefill 连接时整匹配 bootstrap 地址）、`#47732`（kv_events 按 DP rank 偏移 ipc 端点、队列满时完成 shutdown）。diffusion 性能簇：`#46328`（Wan UMT5 padding trim）、`#47121`（Wan-Animate-2 FA3 1.7× denoising）、`#47586`（GQA/MOSS-Image 走 cuDNN SDPA）。CI 重分层：`#47619`（base/extra/nightly/weekly 按重要性重分层）、`#47789`（MiniMax-M3.1-Flash H200 夜间测试超时暂时禁用）、`#46685`（AMD monokernels 日更 pin）、`#47563`（GitHub API 瞬时失败重试）。AMD 路由：`#47236`（Qwen3-Next 512-expert Triton softmax top-k router）、`#40708`（Qwen3.5 512-expert 走 aiter topk_gating）。证据等级：官方一手（标题/正文级为主）。

---

## 4. TensorRT-LLM：2026-11-01 提交（7 条）——enc-dec 跨 KV dtype 修复 + OpenEngine 图像准备 + Qwen3.5 GB300 AgentX

### 4.1 `#19864` 让 encoder-decoder 跨注意力 KV 保持 model dtype（KV dtype 正确性，头号相关）
- **主题**：encoder-decoder 模型开 FP8 KV 缓存时，cross-attention pool 被 self-attention 的 KV 量化继承，导致 cross-attn 写未量化 K/V 但 pool element size 按量化算——**越界 cross-attn pool → CUDA 错误或错误输出**；早期「一刀切拒绝」还禁用了量化 decoder self-attention。
- **内容详情**：复用 external-drafter 的配置拷贝逻辑做 cross-cache 的 sizing/allocation，清掉两份缓存量化标志；cross-attn 保持 model dtype、decoder self-attn 保留 FP8/NVFP4 与既有硬件限制，原始 model/per-layer 配置不变。V1 各 pool token capacity 由自身 byte share 与真实 cache cost 推导（保留显式 `max_tokens` 上限），防止量化 self pool 被更大的 model-dtype cross-cache cost 卡容量；V2 仍直接消费拆分 byte budgets。新增 T5 LLM API 回归 `KvCacheConfig(dtype="fp8")`（FP8 self-cache / BF16 cross-cache / 三次重复生成），注册 L40S CI。修正后的 cross pool 比错误量化分配更费字节，固定显存容量可能下降。
- **应用场景**：encoder-decoder 模型（T5 类）开 FP8/NVFP4 KV 缓存；量化 KV 跨池 dtype 一致性。
- **研判**：【事实】PR 给出根因（cross-attn 写未量化 K/V 但 pool 继承 self-attn 量化 → element size 不一致）、修复与 T5 fp8 回归 + L40S CI。【观点】与 vLLM `#56155`（逐层 KV dtype 解析）同源，是「量化 KV 在 enc-dec 跨池」维度的静默正确性（强化第 4、6、8 条）；上线 enc-dec + 量化 KV 的方须验证 cross/self KV dtype 分离正确。
- **来源**：TRT-LLM PR [#19864](https://github.com/NVIDIA/TensorRT-LLM/pull/19864)（merge 2026-11-01T22:36:02Z）。
- **证据等级**：官方一手（含 T5 fp8 回归 + L40S CI）。

### 4.2 `#21037` OpenEngine：约束图像准备并复用 prefill 状态（性能 / 健壮性）
- **主题**：`#20446` 图像实现的跟进：限制每个 frontend 最多 4 个 media preparation（cancelled 请求保留准入至 executor 工作结束）；复用 placeholder-validation token ID；对 Qwen vision processor 把扩展 prompt ID、mRoPE metadata、原生多模态 cache identity/layout 从 prefill 带到 decode，避免 decode 侧重新取图与预处理。
- **内容详情**：handoff 为已有 `KvSessionRef` 属性的版本化 JSON（限 2 MiB，无协议/schema 变更）；缺失/未知版本保留普通准备；已知版本畸形 metadata / 不兼容 processor 设置 / 越界 layout 被拒绝；双方 worker 须同模型/tokenizer 修订。334 个 OpenEngine CPU 测试（native upb 与 Python protobuf 各 334）通过、无 skip。
- **应用场景**：TRT-LLM OpenEngine 多模态（Qwen vision）服务的图像准备节流与 prefill→decode 状态复用。
- **研判**：【事实】PR 给出 334 passed 与 handoff 设计。【观点】属多模态 KV/状态复用健壮性（第 8 条同源），无协议变更、向后兼容，风险低。
- **来源**：TRT-LLM PR [#21037](https://github.com/NVIDIA/TensorRT-LLM/pull/21037)（merge 2026-11-01T16:05:36Z）。
- **证据等级**：官方一手（含 334 CPU 测试）。

### 4.3 `#20815` Qwen3.5 GB300：上游 AgentX 侧分支优化（性能，描述级 / 未完全绿）
- **主题**：把 SM103 向的 attention / GEMM / MoE / all-reduce / 投机解码路径，以及 GDN state、executor 调度、CUDA-graph 准备、disaggregated transfer admission、prefix-tokenization 缓存、frontend affinity、公共配置等 AgentX 侧分支优化上游到 main。
- **内容详情**：新增 opt-in 项（FP8 all-reduce payload、embedding 复制、draft precision、frontend affinity、context 调度）；**供稿 CI 在 commit `f773e7d` 报告 pipeline 失败，PR 描述称 review 驱动验证进行中、数字来自 pre-review 状态；最终态测试结果与 review severity 不可用**。QA 判 coverage verdict：**needs follow-up**。
- **应用场景**：Qwen3.5 在 GB300 的 AgentX 优化（attention/GEMM/MoE/spec-decode 等）。
- **研判**：【事实】PR 含 SM103 路径变更与 opt-in 配置；CI 在指定 commit 失败、QA 判 needs follow-up。【观点】属「异构硬件（GB300）低比特/投机加速」主线（第 4 条），但**性能数字为 pre-review 状态、CI 未全绿，记描述级/未独立验证**——采用前须以 main 最终态 CI 与实测为准（与 10 月 TRT-LLM `#20694`/`#20755` 同类）。
- **来源**：TRT-LLM PR [#20815](https://github.com/NVIDIA/TensorRT-LLM/pull/20815)（merge 2026-11-01T06:20:13Z）。
- **证据等级**：官方一手（描述级，CI 在 f773e7d 失败、QA needs follow-up、性能数字 pre-review）。

### 4.4 其余 TRT-LLM 要点（官方一手，标题/正文级）
- `#20900` **[fix] 创建 micro-batch scheduler 时检查 `FORCE_CHUNK` context limit**（hybrid linear-attention + KV block reuse 选 `FORCE_CHUNK` 时，若 `max_num_tokens < tokens_per_block` 旧逻辑仅运行时报错；现构造函数即抛 `ValueError` 命名 `max_num_tokens`/chunk unit/来源；含 `test_forced_chunk_context_limit.py` 22 cases）。证据等级：官方一手（含单测）。
- `#21378` **[infra] 为 main 在 pre-merge 67464 豁免 1 个 failed case**；`#20494` **[infra] 构建产物从 On-prem 迁移到 AWS Artifactory**。证据等级：官方一手（chore/infra）。
- 提交计数仅 7 条（周日），无第二页。

---

## 5. llama.cpp：2026-11-01 提交（3 条）——OpenCL IQ3 量化支持 + 构建修复

### 5.1 `#31338` OpenCL：新增 IQ3 初始支持（量化特征）
- **主题**：为 OpenCL（Adreno）后端新增初始 IQ3 支持，具体为 **IQ3_S、IQ3_XXS**；以 `#30180` 为 Adreno 优化参考。
- **内容详情**：声明 AI 使用（Codex 基于既有步骤与框架实现）；属低比特量化格式的端侧后端扩展。
- **应用场景**：OpenCL / Adreno 设备上的 IQ3 低比特量化推理。
- **研判**：【事实】PR 新增 IQ3_S/IQ3_XXS 的 OpenCL 路径，参考 `#30180`。【观点】属「端侧低比特量化」（第 4 条端侧同源）的格式扩展；记为初始支持（Initial），精度/性能数字未在 PR 给出，采用前建议测 Adreno 实测。
- **来源**：llama.cpp PR [#31338](https://github.com/ggml-org/llama.cpp/pull/31338)（merge 2026-11-01T05:26:48Z）。
- **证据等级**：官方一手（初始支持，无独立性能/精度数字）。

### 5.2 `/ #31508` / `#31076` 构建/依赖修复（官方一手）
- `#31508` **requirements：修复 numpy 依赖**（修复 numpy 版本约束导致安装/构建失败）。证据等级：官方一手（build fix）。
- `#31076` **SYCL：尝试用 PkgConfig 定位 LevelZero 的 include / lib**（改善 Intel SYCL 后端在非常规安装路径下的构建探测）。证据等级：官方一手（build fix）。
- 提交计数仅 3 条（周日，为本期四家最低），无第二页；nightly：窗口内发布 **b12028–b12032**（2026-11-01，prerelease）。

---

## 6. 累计结论与研判（沿用 + 强化）

**沿用 2026-11-01 档案（覆盖 10-31）第 1–10 条：**
1. 调度器隐式阻塞的定量画像（host sync 移出热路径的 +4.5%~9.9% 画像）；
2. **发版窗口的双向性**：发版当天合入 main ≠ 在版内（compare API 祖先关系核验）；
3. **默认值成为最主要的风险/收益载体**（默认后端/量化/NUMA 静默改变行为）；
4. **异构后端静默错误高频**（AMD ROCm / Metal / 异构 kernel 不报错只掉精度）；
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**（草稿/索引 KV 所有权一致性）；
6. **量化路径静默错误**（scale 未应用/参数丢失/4D 量化漏算/舍入 1 ulp → 输出偏离且无报错）；
7. **专家张量/路由归属与分页漏接静默错误**（paged-experts 不被 paging、路由行覆盖 → 静默用错专家值）；
8. **静默正确性升维为跨引擎常态主线**（高频失效模式已从「崩溃」转为「不报错、精度静默掉/长上下文静默塌方」）；
9. **长上下文 / 大张量 int32 溢出为静默正确性新子项**（Triton 将 `<2^31` 整数参数按 int32 传递，Qwen3.5 #63761 在 ~209K token 回绕）；
10. **量化 KV 与 KV 卸载的正确性成为新集中暴露面**（MXFP8/FP8 KV 的 scale 初始化、host pool 钉页/卸载、L3 存储 round-trip）。

**本期强化（在周日低活跃下仍出现铁证）：**
- **第 9 条（int 溢出跨引擎）**：SGLang `#47029`（extend attention `kv_indices` int32 回绕，2.1M tokens 边界，aiter CUDA-graph verify 路径与 eagle/ngram kv_indices 亦 int32）证明「整型溢出」家族已从 vLLM（Qwen3.5 #63761 / Qwen4Exp #64820）、llama.cpp（Hexagon #31442）扩展到 **SGLang extend attention 与 aiter 投机 verify 路径**，横跨云/边/投机解码；边界从 ~209K（Qwen3.5）细化到 2,097,152 tokens（8 heads×128 dim）的具体公式。
- **第 4 / 6 / 8 条（异构/量化 KV 静默正确）**：vLLM `#56155`（attention runtime 逐层 KV dtype 解析，FP8/NVFP4 alias 归一与组内不兼容拒绝）+ TRT-LLM `#19864`（enc-dec cross-attn 在 FP8 KV 下 element size 不一致、越界 cross pool）从「逐层 dtype 解析」与「enc-dec 跨池 dtype」两个维度同时补强量化 KV 正确性。
- **第 5 / 8 条（PD / KV 一致性）**：SGLang `#46395`（decode KV 写入 linker + 跳过 prefill 已存前缀，Kimi Linear 4-turn e2e）+ `#47716`（overlap 调度提前处理结果，避免已完成请求误带入下一批）强化 PD 解耦的存储责任委托与调度正确性。

---

## 7.趋势判断与可执行清单

**趋势判断**
1. **静默正确性主线在周日低活跃下延续但仍有铁证**：本期虽为周日（vLLM 10 / TRT-LLM 7 / llama.cpp 3 均为低量、无第二页），仍出现 `#47029`（int64 KV offset）、`#56155`（逐层 KV dtype）、`#19864`（enc-dec cross KV dtype）三条量化/KV/整型静默正确性修补——印证「静默正确性已升维为跨引擎常态主线」（第 8 条），与「发布活跃度」解耦。
2. **int 溢出边界被量化到公式级**：`#47029` 给出 `slot >= 2^31 / stride`（8×128 下 2,097,152 tokens）的精确回绕点，建议作为可执行的回归门禁阈值（第 9 条）。
3. **PD 解耦走「存储责任委托 + 调度正确性」健壮性路线**：`#46395` 把前缀存储委托给 prefill、`#47716` 修 overlap 调度，与 10 月 PD 修复簇同源（第 5、8 条）。
4. **AMD 低比特栈持续扩张但有「描述级」风险**：SGLang `#46690`–`#46692`（gfx950 MiniMax 低比特）、TRT-LLM `#20815`（GB300 AgentX）均标注 perf 数字随后续 PR / pre-review 状态、CI 未全绿——异构硬件加速主线活跃，但采用方须以最终态 CI 与实测为准（第 4 条）。

**可执行清单（建议 serving / 研发方）**
- [ ] **int 溢出回归门禁**：把 `slot >= 2^31 / stride`（约 2.1M tokens @ 8 heads×128）纳入默认发布门禁，覆盖 SGLang extend attention / aiter 投机 verify / vLLM Qwen4Exp / llama.cpp Hexagon（#47029、#63761、#64820、#31442，第 9 条）。
- [ ] **量化 KV dtype 一致性**：启用 FP8/NVFP4 KV 前验证 vLLM `#56155` 逐层 dtype 解析与 TRT-LLM `#19864` enc-dec cross/self KV dtype 分离（第 4、6、8 条）。
- [ ] **PP 跨阶 indexer top-k**：GLM-5.x / DSv3.2 开 PP>1（含 DCP）时核对 `#59961`，PP 切分点避开 top-k 复用组边界（第 5 条）。
- [ ] **SGLang PD 评估**：`#46395`（decode KV→linker、跳过 prefill 前缀）、`#47716`（overlap 提前处理结果）多轮 + 调度正确性回归（第 5、8 条）。
- [ ] **vLLM KV Offload 上线**：`#63927`（GC 停顿 106→31ms、ITL p99 119→55ms）建议在大二级层规模下纳入验收；`#64955` MoE padding 融合（byte-identical）可采纳。
- [ ] **异构硬件加速验收**：SGLang `#46690`–`#46692`（gfx950 MiniMax，perf 待 `#46693`）、TRT-LLM `#20815`（GB300 AgentX，CI 在 f773e7d 失败 / needs follow-up）以最终态 CI + 实测为准，记描述级（第 4 条）。
- [ ] **llama.cpp OpenCL IQ3**：`#31338`（IQ3_S/IQ3_XXS 初始支持）Adreno 实测后再采纳；`#31508`/`#31076` 构建修复按需合入。

---

## 8. 来源索引与数据质量

**官方一手（提交 API + 发布说明 + PR 正文）**
- vLLM：https://github.com/vllm-project/vllm （**10 commits @2026-11-01**；v0.32.0 2026-10-18，窗口内无新正式版）；PR `#56155`/`#59961`/`#63927`/`#64955`/`#64999`/`#63695` 正文已抓。
- SGLang：https://github.com/sgl-project/sglang （**48 commits @2026-11-01**；v0.5.22 2026-10-19，窗口内无新 tag）；PR `#47029`/`#46690`/`#46691`/`#46692`/`#46395`/`#47716`/`#47758`/`#47180`/`#47684`/`#39880`/`#47605` 正文已抓。
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM （**7 commits @2026-11-01**；**v1.4.0rc2 2026-10-28T15:51:34Z** prerelease，窗口前一日发布、仍为最新）；PR `#19864`/`#21037`/`#20815`/`#20900` 正文已抓。
- llama.cpp：https://github.com/ggml-org/llama.cpp （**3 commits @2026-11-01**；nightly **b12028–b12032** 2026-11-01，prerelease）；PR `#31338`/`#31508`/`#31076` 正文已抓。

**社区 / 外部（二手线索，未核验）**
- HN（2026-11-01 窗口，Algolia HTTPS，story 过滤）：引擎相关**无高信号官方更新故事**。`vllm` 14 条均为「LLM」模糊匹配（如「So I guess everyone uses LLMs to think now」「QwenPU: An LLM running inside another LLM's weights」等，非 vLLM 引擎更新）；`sglang`/`llama.cpp`/`TensorRT-LLM`/`KV cache`/`speculative decoding`/`inference optimization`/`LLM inference` 故事数均为 **0**。均记为**二手线索/未核验**（非官方引擎更新），不计入结论。
- Reddit / YouTube / X：未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

**数据质量与限制**
- 计数口径：GitHub commit API 的 `since`/`until` UTC 日期窗口（vLLM 10 / SGLang 48 / TRT-LLM 7 / llama.cpp 3，均 HTTP 200）；按 commit date 过滤。**2026-11-01 为周日，vLLM / TRT-LLM / llama.cpp 计数显著低于工作日（无第二页）属正常低活跃**，SGLang 48 为当日实际量；如需交叉核验可改查 PR merge 口径。
- 证据等级：带验证表 / 逐字节一致 / 复现 fault 的 silent-error / 性能 PR 记为「官方一手（含验证）」；仅提交标题或正文未全抓的记为「标题级」；TRT-LLM `#20815`、SGLang `#46690`–`#46692`、`#47758` 记「官方一手（描述级，perf 数字待后续 PR / CI 未全绿 / 未独立验证）」；社区文章记「二手线索/未核验」。
- 未调用外部浏览器（agent-browser）：本期数据源（GitHub API / releases / HN Algolia）均可直接 HTTP 访问，沿用 2026-10-18 起确立的「API + PR 正文直抓」方法，未触发浏览器需求（与历次一致）。
- `find-skills`：经 SkillHub 语义检索（EN+ZH 各 1–2 组，含「LLM inference GitHub commit monitoring」「github trending hotspot daily digest」「git sync commit push」「大模型推理 GitHub 提交采集 热点归档 仓库同步」）最高相关性 0.050，返回编程专家.Skill / dev-expert / 育儿大师 / 技能发现·Find Skills 等通用编程/生产力类技能，**与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，未安装**（与历次一致）；本地 marketplace 目录为空。

> **归档空档说明**：本档案覆盖 **2026-11-01** UTC 窗口；**2026-10-29** UTC 窗口因 2026-10-30 当日自动化未执行而无档案（即 2026-10-29 档案覆盖 10-28、2026-10-31 档案覆盖 10-30、2026-11-01 档案覆盖 10-31、本期 2026-11-02 档案覆盖 11-01 之间，仅 2026-10-29 窗口一日空档）；**自 2026-10-30 窗口起已连续三期（2026-10-31、2026-11-01、2026-11-02 三期档案分别覆盖 10-30、10-31、11-01）无空档**。本目录不补写中间日期内容，各期仅按自身关注窗口采集。如需回溯 2026-10-29 窗口，请以各引擎官方提交 API 按日期区间重新采集。
> 本档案仅覆盖 2026-11-01 UTC 窗口；相邻日期请以官方提交 API 按日期区间重新采集。
