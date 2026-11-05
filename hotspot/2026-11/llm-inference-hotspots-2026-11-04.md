# 大模型推理优化每日热点档案

> 档案日期：2026-11-04（本自动化于该日执行）
> 关注窗口：2026-11-03（周二）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + PR 正文（issues API，markdown）+ 关键 PR 的 compare API 祖先核验 + HN Algolia（HTTPS，单参数 AND 组合 numericFilters）
> 数据质量：四家提交 API 均 HTTP 200；10 个 vLLM / 8 个 SGLang / 7 个 TRT-LLM / 8 个 llama.cpp PR 正文已抓取；releases API 确认窗口内无新正式版本；HN 窗口内引擎相关无高信号故事（vllm 模糊匹配多为「LLM」/招聘/无关帖）；compare 核验所有 11-03 PR 相对 11-02 标签均为 `diverged`。

## 1. 版本状态与本期头号事件：发布次日稳定化 + KV cache 容量大巡检 + 量化/路由静默错误延续

**提交计数（2026-11-03 UTC，均 HTTP 200；2026-11-03 为周二，vLLM 124 为追踪以来单日最高，高于 10-28 的 102、10-27 的 99，与 v0.33.0 发布次日补丁/收尾潮一致）**：vLLM **124** / SGLang **62** / TensorRT-LLM **53** / llama.cpp **36**。

**发布状态（窗口内无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-03 全部 124 条为发布后 `main` 补丁，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-03 为发布后稳定化与重构。
- **TensorRT-LLM**：仍为 **v1.4.0rc2（2026-10-28T15:51:34Z，prerelease）**，窗口内无更新。
- **llama.cpp**：窗口内 nightly 标签延续（约 `b12062–b12085`，prerelease）；releases API 显示 `b12086`/`b12087`/`b12088` 均于 **2026-11-04T00:xxZ** 发布（窗口刚过）。

**本期头号事件（发布次日主线：KV cache 容量与一致性大巡检 + 量化/路由静默错误延续 + 投机解码 CUDA graph 化 + MoE 路由内核加速；并出现 SGLang 单日三连 revert 信号）**：
1. **KV cache 容量/一致性成为本期最密集主线**：vLLM `#65413`（extensible KV 在 UMA/GB10 上因 `cudaMemGetInfo` 不计入可回收 page cache，KV 被低估**最高 84%**、GB10 仅 216 blocks/11.2 GiB → 用 `psutil` 校正后 1302 blocks/67.6 GiB、P/D 不再「0 KV blocks」）、`#65225`（extensible KV 段补齐替代取整块数）、`#65418`（UMA 边际地板 3 GiB）、`#65319`（batch 满时启动有上限的异步 KV 加载，P/D decode 吞吐）、`#65279`（KV Offload retry 时 SWA null-page 偏移错位致数据错）；TRT-LLM `#21432`（SINK_REQUIRED_ONLY 复用策略，解锁**混合精度 KV 前缀复用**：多轮 agentic GLM-5.2 8×GB300 命中率 80.4%→89.4%、TTFT p90 333→72 s）、`#21245`（Helix prefill owner-only MLA KV 写入）、`#20486`（Dflash Unified KVCache）；SGLang `#47476`（paged-experts 常驻专家 `auto` 自动定容）、`#48099`（Mega MoE localization 把 KV 容量 -1.47%→-0.10%）。
2. **量化 / 路由静默错误延续（强化累计结论第 4/6/8 条）**：vLLM `#61755`（NVFP4 `silu_and_mul` K-padded 输出静默垃圾，编译困惑度 1290959→29.45）、`#65496`（MiniMax-M3.1-Flash TP>1+EP+MegaMoE+SP 下 MXFP8 输入 norm 崩 `AttributeError`）、`#64580`（DSA 恢复非 DeepSeek 模型的 FP32 indexer logits——BF16 改动 0.4–0.75% 选中 token）；llama.cpp `#31427`（NVFP4 experts 张量名不匹配 → 回退到全局 algo，MIXED_PRECISION 模型在 Blackwell 误走原生 W4A4）、`#31079`（Vulkan NVFP4 UE4M3 scale 越界读）；TRT-LLM `#20763`（量化 `exclude_modules` 用 `set` 致顺序随 `PYTHONHASHSEED` 变化 → 两进程算不同 ModelExpress 指纹、**静默不共享权重**）。
3. **投机解码 CUDA graph 化 + 验收器演进**：vLLM `#60989`（MRV2 `AutoRegressiveSpeculator` 基类 + 给独立 draft-model 投机解码启用 CUDA graph，Qwen3-32B/1.7B 平均接受长度 2.80）、`#65175`（XPress + Acceptance Estimator，延续 v0.33.0 XPress）、`#64096`（KV connector 内 lookahead block hashes）；TRT-LLM `#20078`（DSpark decode 置信度调度，opt-in 默认关）；llama.cpp `#30141`（MTP 隐藏状态 save/restore，使 checkpoint 恢复正确）。
4. **MoE 路由精度 / 内核加速**：vLLM `#57167`（拒绝非连续 routing 张量——此前 `wide[:, :top_k]` 切片被内核静默读错专家，ZAYA1 `#65302` 命中）、SGLang `#48041`（hash-routed MoE 的 `topk_config` 在 breakable-prefill CUDA graph 捕获时 `AttributeError`）、`#47641`（单 warp/ token 的 JIT CUDA MoE router，B200 上 DeepSeek-V3 T=16384 路由 24.6→12.3 µs、Qwen3.5(512,top-10) 63.1→15.6 µs）。
5. **SGLang 单日三连 revert 信号（激进 perf 策略回退风险，呼应累计结论第 3 条）**：`#48055`（回退 FlashInfer prefill MoE tactic `#47653`——模型特定、钉死 FlashInfer 0.7.2、单一部署 TP1/PP8/EP1 16384 prefill chunk）、`#48054`（回退 BRIA Fibo 2.0 transformer `#46580`）、`#47994`（回退 JIT 重写 `moe_finalize_fuse_shared` `#47784`）。三者均是把「模型特定 autotune tactic / JIT fusion」撤回，改用通用 autotune 缓存——提示激进 perf 策略在 CI 覆盖外的稳定性风险。

---

## 2. vLLM：2026-11-03 提交（124 条）——发布次日 KV 容量巡检 + 量化静默修复 + 投机解码 CUDA graph

### 2.1 `#65413` Extensible KV cache：按 UMA 校正后的空闲显存定容（KV 容量正确性，UMA/GB10）
- **主题**：integrated（UMA）GPU 上，extensible KV cache（#56492）用原始 `cudaMemGetInfo` 定容，其不计入可回收 page cache，导致 DGX Spark（GB10）上 KV 被低估最高 84%、disaggregated serving 退化为 0 blocks。改用经典路径已有的 `psutil` 校正（#35920）读空闲显存。
- **内容详情**：修复后 GB10 上 Qwen3.8-27B NVFP4（`gpu_memory_utilization=0.8`，V2 runner，extensible KV 默认）得到 1302 blocks / 67.6 GiB（253,952 tokens），相较修复前 216 blocks / 11.2 GiB（42,130 tokens）；P/D 不再报「No available memory for the cache blocks」。离散 GPU 不变（`is_integrated_gpu()` 为 false 时直接返回 `torch.accelerator.get_memory_info`）。无需新配置，`--no-enable-extensible-kv-cache` 在 UMA 上不再需要。
- **应用场景**：GB10/DGX Spark 等 UMA 设备的 extensible KV cache 容量精度；disaggregated P/D 启动成功率。
- **研判**：【事实】PR 给出 GB10/RTX PRO 6000 双设备对照表（GB10 1302 vs 216 blocks）、page cache 差值（cudaMemGetInfo 59–81 GiB vs psutil 114–116 GiB）与 P/D 失败复现。【观点】属「KV 容量定容」主线（第 8 条同源）的 UMA 特例——`cudaMemGetInfo` 不计入可回收 page cache 是隐蔽的容量低估源；上线 UMA 设备须核对 extensible KV 实际块数。compare 对 v0.33.0rc3 `diverged`（合入 23:57Z，晚于发布），**未在版内**。
- **来源**：vLLM PR [#65413](https://github.com/vllm-project/vllm/pull/65413)（merge 2026-11-03T23:57:43Z）。
- **证据等级**：官方一手（含 GB10/离散双设备对照表与 P/D 复现）。

### 2.2 `#61755` 修复 NVFP4 `silu_and_mul` K-padded 输出写入（量化静默正确性，强化第 6 条）
- **主题**：`silu_and_mul_nvfp4_quant` 用未 padding 的行 pitch 写出 K-padded 缓冲，使每 batch >1 token 时在 K 为 16 mod 32 的 padded-K 模型（如 Qwen2.5-72B / Qwen2.5-VL-72B @ TP8，K=3696）上产出垃圾。#63382 修了该 kernel 的 padded scale，行 pitch 是余下问题。
- **内容详情**：按 `scaled_fp4_quant(..., padded_n=...)` 方式写出 K-padded 输出（行 pitch 与 scale tile 取自输出宽度、padding 补零）。修复后编译与 eager 一致：困惑度 1290959 → 29.45（eager 29.43）。24 个 shape 计时无变慢（−13% ~ +0.3%）。RTX PRO 6000 SM120 验证：kernel test 10 failed/20 passed → 30 passed；Qwen2.5-0.5B 收窄 MLP 到 K=4848，逐行相对误差 base 0.127/1.533/1.556/1.532 → 0.127/0.134/0.138/0.128。
- **应用场景**：NVFP4 模型在 padded-K 维度、batch>1 的 KV 前处理/MLP 正确性。
- **研判**：【事实】PR 含困惑度 1290959→29.45、24 shape 计时、kernel test 30 passed 验证。【观点】与累计结论第 6 条（量化路径静默错误）同源——K-padded 行 pitch 漏接致「不报错、输出垃圾」；属 NVFP4 融合 kernel 的静默正确性铁证。
- **来源**：vLLM PR [#61755](https://github.com/vllm-project/vllm/pull/61755)（merge 2026-11-03T03:48:08Z）。
- **证据等级**：官方一手（含困惑度/逐行误差/计时表）。

### 2.3 `#65496` MiniMax-M3.1-Flash：TP>1+EP+MegaMoE+SP 下不发射 MXFP8 输入 norm（量化/启动正确性）
- **主题**：MiniMax-M3 在 NVIDIA 上 TP>1+EP+`deep_gemm_mega_moe`（SP 自动开）组合时，input norm 输出被 all-gather 前若已是 `QuantizedActivation`（`#64852` 引入），all-gather 无法接收 → `AttributeError: 'QuantizedActivation' object has no attribute 'contiguous'`。
- **内容详情**：SP 下保持 norm 输出为 BF16，让 `qkv_proj` 在 gather 后再量化。4×B300 验证：TP4+EP+MegaMoE(SP on) 主线崩溃 → 本 PR 启动且 GSM8K 0.958（开 `VLLM_ENABLE_GEMM_RS=1` 为 0.964），与 #64659 报的 95.5–96.1% 一致。SP off 无变化。无单测（需 4×SM100 + M3.1 checkpoint，CI 无）。
- **应用场景**：MiniMax-M3.1-Flash 在 TP>1+EP+MegaMoE 下的启动与精度。
- **研判**：【事实】PR 给崩溃栈与 GSM8K 对照（崩溃→0.958）。【观点】属「量化/MoE 启动正确性」（第 4/6 条）——SP 与量化激活类型的交互漏接；典型「组合开关才触发」的隐性缺陷。
- **来源**：vLLM PR [#65496](https://github.com/vllm-project/vllm/pull/65496)（merge 2026-11-03T21:51:19Z）。
- **证据等级**：官方一手（含 GSM8K 对照；无单测，CI 不可复现该组合）。

### 2.4 `#64580` DSA：恢复非 DeepSeek 模型的 FP32 indexer MQA logits + `indexer_logits_dtype`（量化精度，SM100）
- **主题**：#61149 在 SM100 上强制 BF16 indexer logits（因当时 DeepGEMM 丢弃 `logits_dtype`）。本 PR bump DeepGEMM 到 `ae4fd25`（恢复 `logits_dtype` 参数），新增 `--attention-config.indexer_logits_dtype`（`auto`/`float32`/`bfloat16`）。
- **内容详情**：`auto` 在 SM100 上把 GLM-5/GLM-5.3/HY-V4 回到 FP32 logits，DeepSeek-V3.2/V4/V4.1 保持 BF16（main 行为）。对 GB300 上 FP64 参考，FP32 精确选中 top-2048；BF16 改动 0.4–0.75% 的选中 token。SM90/SM120 默认不变。
- **应用场景**：DSA 稀疏 indexer（DSv3.2/GLM-5.x/DSv4/V4.1/HY-V4）、GLM-5.3 kpool indexer、DSv4.1 专用 decode indexer 在 SM100 的 logits 精度。
- **研判**：【事实】PR 给 per-model 默认表与「BF16 改动 0.4–0.75% 选中 token」的 FP64 对照。【观点】属「量化精度」（第 4/6 条）——indexer logits dtype 是长上下文稀疏索引的精度债；`auto` 把非 DeepSeek 模型恢复 FP32 是静默精度修正。
- **来源**：vLLM PR [#64580](https://github.com/vllm-project/vllm/pull/64580)（merge 2026-11-03T23:13:24Z）。
- **证据等级**：官方一手（含 FP64 对照与 per-model 默认表）。

### 2.5 `#57167` MoE：拒绝非连续 routing 张量（路由静默错误，强化第 7 条）
- **主题**：MoE routing kernel 把 `topk_ids`/`topk_weights` 当稠密线性缓冲消费；切片结果 `wide[:, :top_k]` 形状 `(num_tokens, top_k)` 但 stride `(wide_top_k, 1)`，旧 `topk_weights.stride(1)==1` 检查接受该布局，而 kernel 忽略行 stride、静默读错路由数据。
- **内容详情**：在 `moe_align_block_size` C++ wrapper 校验 `topk_ids`、在共享 MoE dispatch 边界要求 `topk_weights` 完全连续、在 `FusedMoERouter.select_experts` 一次性拒绝非连续（覆盖所有 modular backend，含 FlashInfer CUTLASS）。ZAYA1 `#65302` 用返回 packed tensor 列切片的自定义 routing 命中：token 1 静默拿到 token 0 的 expert id 作权重。加三则回归测试。
- **应用场景**：自定义 routing 函数 / 切片路由结果的 MoE 正确性。
- **研判**：【事实】PR 给根因（stride 忽略）与 ZAYA1 命中案例 + 三则回归测试（2162 passed，1 CUDA 回归测试本地未跑由 CI 补）。【观点】与 SGLang `#48041`、llama.cpp 路由类同属「路由张量布局静默错误」（第 7 条）——改为 fail-fast 比静默错优；属路由精度债的显式收口。
- **来源**：vLLM PR [#57167](https://github.com/vllm-project/vllm/pull/57167)（merge 2026-11-03T23:03:53Z）。
- **证据等级**：官方一手（含根因与回归测试）。

### 2.6 `#60989` MRV2 投机解码：AR speculator 重构 + draft model 启用 CUDA graph（投机解码性能）
- **主题**：引入共享 `AutoRegressiveSpeculator` 基类，统一 CUDA graph 管理与自回归解码逻辑（target-dependent 与 standalone draft-model 两子类）；给 Model Runner V2 的独立 draft-model speculator 启用 CUDA graph 捕获/重放。
- **内容详情**：Qwen3-32B / Qwen3-1.7B、400 MT-Bench、temp 0、256 output、3 spec tokens、MRV2、FULL_AND_PIECEWISE：mean acceptance length **2.80**（tok0 0.76 / tok1 0.58 / tok2 0.46），num_accepted 65551 / num_draft 109029。性能对比同配置 eager FULL_AND_PIECEWISE（PR 未摘录具体中位加速百分比，仅给接受率与 5 次运行中位协议）。
- **应用场景**：MRV2 下 draft-model 投机解码的 CUDA graph 加速与代码归一。
- **研判**：【事实】PR 给 Qwen3-32B/1.7B 接受率分布与评估协议。【观点】属「投机解码」主线（第 5 条）——把 CUDA graph 扩到 draft model 是自 v0.33 投机体系重构后的性能收口；接受率 2.80 与往期 EAGLE/MTP 量级一致。具体端到端加速数字 PR 未给出，记为官方一手（接受率级）。
- **来源**：vLLM PR [#60989](https://github.com/vllm-project/vllm/pull/60989)（merge 2026-11-03T23:59:24Z）。
- **证据等级**：官方一手（含接受率与评估协议；端到端加速百分比未摘录）。

### 2.7 `#62565` ROCm：修复 Kimi-K2.6 int4 MoE prefill 在 Triton 3.8 的回归（异构性能，gfx950）
- **主题**：Kimi-K2.6（int4 W4A16, TP8）prefill 在 MI350/MI355 上 v0.29→v0.31 间 c32 TTFT 涨约 150%。根因是 Triton 3.7.1→3.8.0 对某个 tile 的 codegen 回退：`fused_moe_kernel_gptq_awq` 的 `"8192"` 项（BLOCK 256×256×32, 8 warps, 2 stages）在 Triton 3.8 下慢 3–8×。
- **内容详情**：把该 entry 改为 BLOCK_N 128 + num_stages=1。Kernel（Kimi-K2.6 TP8, M=8192/16384, Triton 3.8）：44180 µs → **6129 µs**。Serving（c32）：TTFT 1981→829 ms（**−58%**），tok/s +6.2%；c1 不命中该 entry（差异为运行间噪声）。
- **应用场景**：MI350/MI355 上 Kimi-K2.6 int4 MoE 长 prefill 性能。
- **研判**：【事实】PR 含 kernel 44180→6129 µs 与 serving c32 TTFT −58% 对照。【观点】属「异构硬件（AMD ROCm）性能回归」（第 4 条）——编译器升级引入的 tile codegen 回退，是「版本升级即性能债」的实例；适用方升级 Triton 后须重测 MoE prefill。
- **来源**：vLLM PR [#62565](https://github.com/vllm-project/vllm/pull/62565)（merge 2026-11-03T21:32:08Z）。
- **证据等级**：官方一手（含 kernel + serving 对照表；声明用 Claude 辅助分析）。

### 2.8 `#65319` KV Connector：batch 满时启动有上限的异步 KV 加载（PD 吞吐，第 5/8 条）
- **主题**：让异步 KV 加载（如 P/D decode 实例的 NIXL pull）在 running batch 满时即启动，受新 `--max-num-inflight-kv-loads` 上限约束，使释放的 slot 直接给「KV 已到达」的请求。替代 #64628/#61725。
- **内容详情**：默认 `None`（行为不变）；设上限后，可异步加载的请求进专属队列，加载在 batch 满或 token 预算用尽时也启动，full batch 只扫队列头。新增 connector hook `KVConnectorBase_V1.may_load_async`。单测：首提交 334 passed；两提交 + multi-connector/offloading 套件 1084 passed / 9 failed（TODO 确认 main 是否亦失败）。**TODO：端到端 P/D 基准未做**。
- **应用场景**：P/D decode 实例高并发、batch 满时的 KV 加载吞吐。
- **研判**：【事实】PR 给问题量化（#61725 测每步 90–140 请求 ×1.6 ms = 150–290 ms 全 DP 等待）与单测 334/1084 passed。【观点】属「PD/调度交叉债务」（第 5 条）——异步 KV 加载与 batch 满的耦合是 PD 吞吐隐性瓶颈；端到端基准待补，采用前建议自测。
- **来源**：vLLM PR [#65319](https://github.com/vllm-project/vllm/pull/65319)（merge 2026-11-03T23:47:05Z）。
- **证据等级**：官方一手（含单测与问题量化；端到端 P/D 基准待补）。

### 2.9 `#65279` KV Offload：store retry 时保留 SWA null-page 偏移（KV 卸载一致性，第 8 条）
- **主题**：CPU offload store 因容量耗尽重试时，SWA 可能释放并重新分配 GPU block 给另一逻辑位置；offloading scheduler 把旧位置标 `NULL_BLOCK_ID` 却从 source spec 删除，使后续页错位进目标 CPU chunk、加载回错误 KV。
- **内容详情**：在 GPU→CPU store spec 中保留每完整 SWA chunk 的逻辑页布局（含 null blocks 作占位）。回归测试（mock store 失败 + SWA block 复用）：修复前 8 源页（漏内部 null）→ 加载逻辑页 7/8 返回 `[9, -99]`；修复后 9 逻辑页（group_sizes=[9]）→ `[8, 9]`。真实 `CPUOffloadingManager` 复现一致。
- **应用场景**：KV Offload（CPU）在 store 重试 + SWA block 重用的数据一致性。
- **研判**：【事实】PR 给修复前后 store spec（8 vs 9 页）与真实复现 `[9,-99]→[8,9]`。【观点】属「KV 卸载/一致性」（第 8 条）——异常路径漏占位致静默错位；V1 KV Connectors GPU CI 当前禁用，测试手动跑，提示该路径回归覆盖薄弱。
- **来源**：vLLM PR [#65279](https://github.com/vllm-project/vllm/pull/65279)（merge 2026-11-03T12:15:33Z）。
- **证据等级**：官方一手（含回归测试与真实复现；GPU CI 禁用、手动验证）。

### 2.10 其余 vLLM 要点（官方一手，标题/正文级）
- `#65185` **[Compile] 修复 `topk_histogram_4096_overflow.cuh` 编译告警**（#802-D 重声明非标默认参数）——**仅编译告警清理，非运行时溢出修复**（文件名含 4096 overflow 但本 PR 不动运行时逻辑），记官方一手（编译告警级）。
- `#63419` **[ROCm] 迁移 Aiter sigmoid → topk_gating**；`#64317` **[ROCm] per-token-head KV 与 AITER cross-attn fallback 保 32 位 buffer 寻址**；`#60418` **[ROCm][RDNA] TRITON_ATTN 与 ROCM_ATTN 优先级互换**；`#65322` **[ROCm][Perf] W4A16 为 gfx115x 重调 int4 skinny GEMM decode 元组**；`#61435` **[ROCm] MI355X 专用 decode K1/K2 + MLA M1a/M1b（FlyDSL, TP8, DCP）**；`#62633` **[Perf][DSv4.1] Swap-AB CuTe DSL decode attention（小 TP2/TP4 步）**；`#64723` **[Perf][Qwen3.5] (NVIDIA) 融合 AR+norm+quant**；`#63801` **[Perf][HiSparse] 有界 copy worker 预取投机组**；`#63754`/`#63672` **[Fast Start] CUTLASS FP8 / NVFP4 权重缓存重启**；`#64916` **[Spec Decode][Perf] DFlash2 candidate edge scoring 融进 Triton kernel**；`#65175` **[Spec Decode] XPress + Acceptance Estimator**；`#64096` **[KV Connector][Spec Decode] KV connector 内 lookahead block hashes**；`#65230` **[Quant] 移除 QuTLASS/fp_quant，QuIP NVFP4 走 Humming**；`#64312` **[Quant] ModelOpt 支持 CSF 压缩 NVFP4 block scales**；`#62275` **[Bugfix] 在 sm_110 禁用 CUTLASS FP4 MoE 后端**；`#56102` **[Bugfix][Kernel] 修复 SM110 cooperative top-k dispatch**；`#63614` **[Bugfix][Kernel] hadacore 修复 odd row 部分块 warp 漏 wait_group（#63514）**；`#65050` **[Kernel] SM110 启用 CuteDSL GDN prefill**；`#61609` **[Attention][DCP] FlashMLA sparse bf16 KV 路径上 DCP**；`#65225` **[Bugfix] extensible KV 段补齐替代取整块数**；`#65418` **[Bugfix][ROCm] extensible KV 边际地板抬到 3 GiB**；`#65214` **[HiSparse] 支持 extensible KV cache**；`#65455` **[KV Offload] 适配新 KVCR key adapter API**；`#64299`/`#64330` **[KV Offload][P2P] disconnect 后结算缓冲终态 / ROCm MoRIIO 等 WRITE 完成**；`#64558`/`#64993` **[KV Offload] store_threshold 计数 / 陈旧 partial-tail 期望**；`#65271` **[Scheduler] prefix cache hit 落在多模态 item 内时不切分**；`#57167`（见 2.5）；`#62097`/`#61162`/`#62470`/`#57245` **[LoRA] CLIP CLS rows / KV-sharing fast prefill / embed_scale / DSV4.1 adapter**；`#63902` **[Rust Frontend] 无状态 Responses API**；`#65216` **granite_toolcall_parser**；`#64241` **pin 依赖哈希强制不可变引用**；`#63420` **[Mooncake] prefix lookup 选择性 lease**；`#65060` **[KV Connector] MultiConnector 嵌套 connector 角色检测**；`#65238` **[CI] batch-invariance 自动打标**。证据等级：官方一手（标题/正文级为主）。

---

## 3. SGLang：2026-11-03 提交（62 条）——采样/top-k 一致性 + MoE 路由内核加速 + 三连 revert 信号

### 3.1 `#41916` [Fix] sampling：跨采样路径统一 top-k/top-p 过滤顺序（采样正确性，静默）
- **主题**：SGLang 通常从原始概率分布算 top-p 截断并与 top-k 支持集相交；部分路径先 top-k、重归一化剩余概率、再 top-p，使 top-p 作用于 top-k 后残留质量，效果不可预测且与同参数的其他路径不一致。
- **内容详情**：受影响路径——FlashInfer min-p 路径、共享的 EAGLE 系列/standalone/n-gram verifier、DFlash/DSpark/UNO 共用的 target-distribution builder。改为先 top-p 后 top-k（sparse top-k 路径用 top-k 支持集的原始质量翻译 top-p 截断）。加聚焦 CUDA 回归测试。H200 验证：回归用例修复前失败、修复后通过（18 passed, 6 subtests passed）。示例 `[0.35,0.25,0.20,0.05,0.03,...]` top_k=4 top_p=0.82：原始 top-p 含 token3（其前质量 0.80），先 top-k 重归一化后该质量变 0.80/0.85，token3 被误删。
- **应用场景**：FlashInfer min-p、EAGLE/standalone/n-gram 验证、DFlash/DSpark/UNO target 分布的采样一致性。
- **研判**：【事实】PR 给根因（过滤顺序）、受影响的 4 类路径与 token3 误删的数值示例 + 回归测试。【观点】属「采样/验证静默正确性」——top-k/top-p 顺序错致分布静默偏移，是投机验证一致性的隐性债；与「验证分布须跨路径一致」主线同源。
- **来源**：SGLang PR [#41916](https://github.com/sgl-project/sglang/pull/41916)（merge 2026-11-03T20:06:33Z）。
- **证据等级**：官方一手（含数值示例与回归测试；compare 对 v0.5.23 `diverged`，未确认在版内）。

### 3.2 `#47641` [Perf][NVIDIA] moe：单 warp/token 的 JIT CUDA MoE router（MoE 路由性能）
- **主题**：新 JIT CUDA MoE router `moe_route_warp`：每 warp 一个 token，每 lane 持 E/32 专家为保序 u32 key，排序一次（编译期 Batcher 网络），每轮 top-k 只约减 32 个 lane 头。覆盖 128..512 专家（sigmoid/sqrtsoftplus/softmax，fp32/bf16，fused shared-expert，renorm）。
- **内容详情**：B200 CUDA graph 中位（router alone, µs）：DeepSeek-V3(256,8) T=16384 24.6→12.3；Kimi-K2(384) 39.7→13.1；GLM-4.5(160) 23.4→9.4；Qwen3-MoE(128) 13.9→8.2；Qwen3.5(512,top-10) 63.1→15.6；DSv4.1(384,top-6) 35.2→17.2；DSv4.1 VL(per-token bias) 35.6→17.6；MuseSpark 16.8→10.7。H200 增益更大（Kimi-K2 T=16384 45.6→15.8）。gate GEMM + PDL 后每 MoE 层省 0.5–2 µs。
- **应用场景**：DeepSeek/Kimi/GLM/Qwen3/DSv4.1/MuseSpark 等 MoE 模型的路由内核加速（SM100）。
- **研判**：【事实】PR 含 B200/H200 多模型路由耗时对照（多为 2–4×）。【观点】属「MoE 路由内核加速」（第 7 条同源）——JIT 单 warp/token router 是路由热路径的性能收口；auto-generated（Claude + MyGO agent）须以最终 CI 为准。
- **来源**：SGLang PR [#47641](https://github.com/sgl-project/sglang/pull/47641)（merge 2026-11-03T12:30:45Z）。
- **证据等级**：官方一手（含多模型对照表；auto-generated 内核）。

### 3.3 `#48041` [Fix] moe：prefill 路由路径在 topk_config 前跳过 hash 层（路由崩溃修复）
- **主题**：hash-routed MoE 用 `HashTopK`（无 `topk_config`）。两个路由路径在查 `topk_config` 前未查 `is_hash`：breakable-prefill 双流 graph 谓词、两个 vision top-k 分支，致 breakable prefill CUDA graph 捕获时 `AttributeError: 'HashTopK' object has no attribute 'topk_config'`。
- **内容详情**：在 `_can_dual_stream_graph` 先查 `is_hash`；vision topk 对 hash 层跳过（保留 `HashTopK` 配 `input_ids`）。eager 与 `--cuda-graph-backend-prefill disabled` 不受影响（前导检查短路）。无精度测试（仅改路由选择谓词）。
- **应用场景**：hash-routed MoE（如部分视觉/稀疏路由）在 breakable-prefill CUDA graph 下的捕获。
- **研判**：【事实】PR 给崩溃栈与修复点（先 `is_hash`）。【观点】属「MoE 路由谓词」（第 7 条）——`is_hash` 与 `topk_config` 的耦合漏判；属轻量但典型的图捕获期崩溃修复。
- **来源**：SGLang PR [#48041](https://github.com/sgl-project/sglang/pull/48041)（merge 2026-11-03T12:34:44Z）。
- **证据等级**：官方一手（含崩溃栈；无精度测试，CI Base 报红）。

### 3.4 `#47476` [Feat] paged-experts：常驻专家自动定容（MoE 内存容量，第 7/8 条）
- **主题**：Paged Experts（#42557）需手设 `--paged-experts-num-resident K`，K 与 KV cache 共享静态预算，正确 K 依赖 GPU/模型/量化/KV dtype/上下文。改为 `--paged-experts-num-resident` 接受 count 或 `auto`（默认 `auto`）。
- **内容详情**：**KV cache 优先**——在 KV 配置前后各一次受保护调用；默认保留一个模型自身上下文的请求（整页）。专家 slot 取剩余静态预算。Qwen3-30B-A3B-GPTQ-Int4 / 16 GB GPU：decode 208 tok/s @ K=63、233 tok/s @ K=92（说明手设 K 影响大）。
- **应用场景**：消费级/小显存 GPU 上跑超显存 MoE（Paged Experts）的免手调 K。
- **研判**：【事实】PR 给 `auto` 算法与 Qwen3-30B 208→233 tok/s 的 K 敏感度。【观点】属「MoE/内存容量默认体验」（第 3/7 条）——把易错的手设 K 改为 `auto` 是默认风险下降（第 3 条反向：更安全的默认）；落地前核对 KV 容量是否仍满足上下文。
- **来源**：SGLang PR [#47476](https://github.com/sgl-project/sglang/pull/47476)（merge 2026-11-03T01:05:24Z）。
- **证据等级**：官方一手（含定容算法与 K 敏感度）。

### 3.5 `#48099` [Perf][NVIDIA] moe：释放 localized Mega MoE 原权重使 KV 容量匹配非 localized（KV 容量）
- **主题**：开 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE=1`（#47976）后，DSv4.1-Flash 多载 1.5 GB 权重、KV tokens 少 1.47%（routed 专家仍单拷贝）。根因是 interleave 时临时 L1 scale 段被保留、fused shared expert 的 L2 被持两份。
- **内容详情**：`build_mega_moe_experts_weights` 先建 scales 再 interleave L1，localized 时每层 free 替换权重 + `empty_cache()`；`build_mega_moe_shared_weights` localized 时 free shared expert 原权重。DSv4.1-Flash（DP4/EP4, megamoe, DSPARK）：localized KV 容量从 −1.47% → **−0.10%**（98,056,960 → 98,187,776 tokens），GSM8K 0.995 不变。
- **应用场景**：DSv4.1-Flash 在 localization 开关下的显存/KV 容量权衡。
- **研判**：【事实】PR 给内存快照分析 + 多配置 KV tokens/GSM8K 表（localized −1.47%→−0.10%）。【观点】属「MoE/显存容量」（第 7/8 条）——localization 优化因内存碎片吞掉 KV 容量，是「优化引入隐性容量债」的实例。
- **来源**：SGLang PR [#48099](https://github.com/sgl-project/sglang/pull/48099)（merge 2026-11-03T17:48:23Z）。
- **证据等级**：官方一手（含内存快照与 KV tokens 表）。

### 3.6 `#47186` [Fix] diffusion：Anima RoPE 保留 FP32 乘积舍入（扩散精度，SM120）
- **主题**：RTX PRO 6000 Blackwell 上现有 Anima FP32 split-half RoPE kernel 首用精确性检查失败并回退 eager（Triton 3.8 在 FP16/BF16 输出特化中发射收缩算术，cubin 含 8 条 FFMA、捕获 Q/K 90 个差异值）。
- **内容详情**：两乘积在 FP32 加法与输出 cast 前均用 `mul_rn_f32` 辅助（替换的 cubin 无 FFMA），现有精确性门通过、真实 Anima serving 用融合 kernel。附 evidence index 与数值/codegen 审计 JSON、既有测试。
- **应用场景**：Anima 扩散模型（RTX PRO 6000 SM120）RoPE 融合 kernel 的精度与性能。
- **研判**：【事实】PR 给根因（Triton 3.8 收缩算术、90 差异值）与门通过/融合 kernel 复用证据（含审计 JSON）。【观点】属「扩散/异构精度」（第 4 条）——编译器收缩算术致 FP32 合约失效、回退 eager 丢融合，是「异构后端精度债」扩散到图像生成域的实例。
- **来源**：SGLang PR [#47186](https://github.com/sgl-project/sglang/pull/47186)（merge 2026-11-03T11:37:51Z）。
- **证据等级**：官方一手（含数值/codegen 审计与测试证据）。

### 3.7 `#43910` [Refactor] scheduler：waiting queue 与 chunked slot 纳入 request lists（调度器架构重构）
- **主题**：waiting queue 与 chunked slot 改为与其他队列一致的 `ReqList`（IndexedReqList/KeyedReqList），每列表自带 abort，统一 `abort_request`。dict 支持的列表每次 `join`/`leave` 整体复制（PD 长队列时平方复杂度），改为原地修改。
- **内容详情**：行为变化——多次 abort 同名 waiting 请求改为按队列顺序（原反向）。`SchedulePolicy.calc_priority` 返回准入序、调用方一次性 `replace` 写回。
- **应用场景**：调度器请求列表的统一与长 PD 队列的复杂度消除。
- **研判**：【事实】PR 给重构点（ReqList 两实现、原地修改、abort 顺序变化）。【观点】属「调度器架构」（第 5 条）——把平方复制改为原地是 PD 高并发下的隐性性能债收口；属架构级重构，非功能缺陷。
- **来源**：SGLang PR [#43910](https://github.com/sgl-project/sglang/pull/43910)（merge 2026-11-03T23:24:58Z）。
- **证据等级**：官方一手（含重构点与行为变化；Base CI 报红）。

### 3.8 SGLang 三连 revert 信号（perf 策略回退风险，第 3 条）
- `#48055` **Revert "[Perf] kimi-k3: Seed a validated FlashInfer prefill MoE tactic (#47653)"**——#47653 把模型特定、钉死 FlashInfer 0.7.2、仅 TP1/PP8/EP1 + 16384 prefill chunk 的 tactic 文件塞进共享 runner；回退改用 `flashinfer_autotune_context` 已有的 autotune 缓存。
- `#48054` **Revert "[Feat] diffusion/model: add the BRIA Fibo 2.0 transformer (#46580)"**——回退 BRIA Fibo 2.0 transformer 与文本条件。
- `#47994` **[Fix] moe: Revert the JIT rewrite of `moe_finalize_fuse_shared` (#47784)**——回退 #47784 的 JIT 重写。
- **研判**：【观点】单日三连 revert 指向同一信号——**模型特定 autotune tactic / JIT fusion 等激进 perf 策略在 CI 覆盖外的部署稳定性不足**，被撤回改用通用路径。这把累计结论第 3 条（默认值/能力移除 = 风险载体）反向延伸到「已合入的激进优化也可能被回退」，建议 serving 方锁版本并比对 revert 前后的 perf/精度。三则均记官方一手（revert 级，CI Base 报红）。

### 3.9 其余 SGLang 要点（官方一手，标题/正文级）
- `#48245` **[Perf] dsv4 mega MoE pre-dispatch prefill-sized batches 每线程四块**；`#48229` **[Perf] dsv4 一个 Triton kernel 展开低比 indexer page table**；`#48119` **[Refactor] tokenizer 折叠 TokenizerWorker 进 TokenizerManager**；`#47955` **[Perf] srt 记录每请求 extend row 是否完成 prompt**；`#48131` **[CI][AMD] nightly-disagg 保留 MI355X 设置 + 失败清理**；`#46930` **[Perf] hicache write_back eviction walk 每轮只等一次**；`#48192` **[Chore] mooncake bump 0.3.14**；`#48125` **[Feat] unified_cache MiniMax-M3.1 直连外部 linker**；`#47119` **[Feat] sgl-router 经 sglang-processor 渲染 MiniMax-M3/M3.1**；`#46921` **[Feat][NPU] attention 支持 modelslim MLA FP8 KVCache**；`#47928` **[Refactor] memory-pool 从 buffer info 组装 HiCache host pools**；`#47283` **[Fix][AMD] deepseek-v4 尊重 AITER Q-LoRA 量化选择**；`#45466` **[Fix] constrained 不把 grammar poll 超时缓存为无效 grammar**；`#47064` **[Perf][AMD] kda 窄化 Kimi-K3 TP8 KDA recurrent tile（小 batch, gfx950）**；`#47993` **[Perf][NVIDIA] qwen3 复用 table QK-norm/RoPE kernel 做 MRoPE decode**；`#44489` **[Perf][AMD] deterministic 允许 gfx950 AITER decode attention**；`#47292` **[Perf] diffusion SANA-Video depthwise conv + GLU 融于 SM120**；`#48070` **[Doc] cookbook MiniMax-M3.1-Flash GB300 MonoKernel / DEP16 AgentX**；`#47976` **[Perf][NVIDIA] moe 跨 locality domain 本地化 DeepGEMM Mega MoE 权重**；`#47990`/`#47992` **[Refactor] imports 从定义处导入、移除 re-export shim / lint 拒绝经仅 re-import 的模块导入**；`#47261` **[Fix] attention fused RoPE KV writes 避开 DCP layouts**；`#44148` **[Feat] moe Kimi-K3 W4A16 CuTe DSL NVFP4 MoE 允许 SiTU**；`#40825` **[Feat][NPU] kimi-k3 共享 compact communication 的 DCP**；`#44398` **[Test][NPU] kimi_k3 32p gpqa L2 host HiCache 命中率 bench**；`#43904`–`#43909` **[Refactor] scheduler 请求列表组（abort/ownership/interface/chunked）**；`#45785` **[Fix] scheduler idle 时仍发布 KV 指标**；`#48051` **[Doc] Qwen B300 Qwen3-VL-4B 用 TRTLLM MHA**；`#47929` **[Doc] SemiAnalysis AgentX launches 同步到 InferenceX**；`#47997` **[Doc] blogs 同步 LMSYS SGLang blog 卡**。证据等级：官方一手（标题/正文级为主，多数为 perf/refactor/CI/doc）。

---

## 4. TensorRT-LLM：2026-11-03 提交（53 条）——Dflash Unified KV + 混合精度 KV 复用 + 量化静默指纹

### 4.1 `#21432` SINK_REQUIRED_ONLY 注意力复用策略（KVCM v2，混合精度 KV 前缀复用）
- **主题**：KV cache manager v2 的滑动窗口生命周期新增 `ReusePolicy::SINK_REQUIRED_ONLY`：sink 块遵循 REQUIRED（约束前缀匹配、随匹配申领、驱逐 sink 仍削子树），window 块遵循 OPTIONAL（绝不缩短匹配）；仅带滑动窗口 + sink 的注意力层接受。
- **内容详情**：为**混合精度 KV** 前缀复用铺路——低比特 KV 保留首尾 N token 高精度恢复长上下文精度，但高精度末 N token 被驱逐后其余低精度前缀也无法复用；本策略让 follow-up 保留必需的高精度首 N、末 N 退回低精度副本，前缀仍复用。opt-in。多轮 agentic（GLM-5.2, 8×GB300, 并发 128）：**缓存命中率 80.4%→89.4%，TTFT p50 4.4→3.3 s、p90 333→72 s**。测试：KVCM2 C++ 161 passed；`test_optional_checkpoint.py` 595 passed/14 skipped。
- **应用场景**：混合精度 KV（首/末 N 高精度）下的前缀复用与多轮 agentic TTFT。
- **研判**：【事实】PR 给复用语义、GLM-5.2 8×GB300 命中率/TTFT 对照与单测 595 passed。【观点】属「KV 复用/混合精度」（第 8 条）——把混合精度 KV 的低精度副本纳入前缀复用，是长上下文 agentic 的首个量化收益实证（p90 TTFT 333→72 s）；opt-in，建议长上下文服务评估。
- **来源**：TRT-LLM PR [#21432](https://github.com/NVIDIA/TensorRT-LLM/pull/21432)（merge 2026-11-03T18:43:07Z）。
- **证据等级**：官方一手（含 GLM-5.2 8×GB300 实测对照与单测）。

### 4.2 `#20763` 量化 `exclude_modules` 按 checkpoint 顺序保留（量化静默正确性，第 6 条）
- **主题**：五处合并量化 `exclude_modules` 用 `set`，顺序随字符串哈希变化、且跨进程不同（`PYTHONHASHSEED` 未钉）：FP8 block-scale 与 MXFP4 的 `ModelConfig.load_hf_quant_config`、dtypes.json MXFP4 路径、两处 compressed-tensors 合并。
- **内容详情**：该列表喂给 ModelExpress(MX) `SourceIdentity` 量化指纹，故两进程加载同 checkpoint 可能算不同指纹、**永不共享权重**。改用 `dict.fromkeys`（与 NVFP4/MXFP8 分支一致）：同元素、checkpoint 顺序、去重。拆自 MX FP8 block-scale 资质 PR（TRTLLM-17209）。测试 `test_hf_quant_config.py`：5 项修复前全失败、修复后通过。
- **应用场景**：量化模型（FP8/MXFP4/compressed-tensors）`exclude_modules` 的跨进程权重指纹一致性（Tensor Parallel / 多进程）。
- **研判**：【事实】PR 给根因（set 顺序随哈希变 → 不同指纹 → 静默不共享权重）与 5 项测试修复前失败/修复后通过。【观点】与 vLLM `#61755`/`#65496`、llama.cpp `#31427` 同属「量化静默正确性」（第 6 条）——非确定性指纹致 TP 各 rank 权重错位且**不报错**，是最隐蔽的量化债之一；建议量化加载后核对 `SourceIdentity` 跨进程一致。
- **来源**：TRT-LLM PR [#20763](https://github.com/NVIDIA/TensorRT-LLM/pull/20763)（merge 2026-11-03T00:18:27Z）。
- **证据等级**：官方一手（含根因与 5 项测试对照）。

### 4.3 `#20486` Dflash Unified KVCache（PD/投机 KV 统一，描述级/未全绿）
- **主题**：`KvCacheCreator` 在兼容门控下把 DFlash target 与 draft 层放进同一个 V2 KV cache manager；DFlash 绑定 unified-manager 分配元数据与 block 偏移；peer 注册拒绝 mismatched unified draft-layer 计数。
- **内容详情**：Dev/QA 评审均指出 supplied CI 报告含成功与失败、未确认当前改动成功执行；作者给了 perf 结果链接但测量未纳入证据；QA 覆盖结论 **needs follow-up**，disaggregated DFlash 精度测试列于 `l0_dgx_b200.yml` 但无对应 manual-QA 条目。新增/修改测试覆盖 unified 资格、launcher 配置、混合格式 block-table 索引、peer 兼容等。
- **应用场景**：disaggregated DFlash 的 target/draft 统一 KV 管理。
- **研判**：【事实】PR 为 DFlash unified KV 架构改动；Dev/QA 均标注 CI 状态不可用、覆盖 needs follow-up。**性能/精度未独立验证，记描述级/未全绿**——采用前以最终 CI 与实测为准（与 10 月 TRT-LLM `#20694`/`#20815` 同类）。
- **来源**：TRT-LLM PR [#20486](https://github.com/NVIDIA/TensorRT-LLM/pull/20486)（merge 2026-11-03T22:44:42Z）。
- **证据等级**：官方一手（描述级，Dev/QA 均 needs follow-up、perf 未纳入证据）。

### 4.4 `#21245` Helix prefill 的 owner-only MLA KV 写入（PD/CP KV 一致性）
- **主题**：Helix MLA 的 CP rank 处理 prompt 注意力，但每 rank 仅拥有持久 KV cache 的一部分。只持久化分配给该 rank 缓存页的压缩 KV 记录，同时保留上下文注意力所需的临时 K/V。
- **内容详情**：用共享 Helix prefix-length helper 建全局 token 位置→rank-local 缓存 slot 的 batch-local 映射，经可选 MLA RoPE/append 输入传入；负 slot 抑制持久 KV 与 scale 写入而不跳过临时 Q/K 准备。稠密 FP4 KV-cache 注意力在 owner-only append 变更前被拒（native NVFP4 writer 测试不建立完整 FP4 CP 注意力支持）。移植自 #20921，不单独启用 aggregate serving。
- **应用场景**：Helix（aggregated）MLA 在 CP 下的 prefill KV 归属与持久化。
- **研判**：【事实】PR 给 owner-only 写入设计、guard（稠密 FP4 KV 拒）与移植来源。【观点】属「PD/CP KV 一致性」（第 5/8 条）——跨 CP rank 的 KV 归属是分布式正确性债；稠密 FP4 路径仍受限，采用前核对 CP 拓扑。
- **来源**：TRT-LLM PR [#21245](https://github.com/NVIDIA/TensorRT-LLM/pull/21245)（merge 2026-11-03T06:03:30Z）。
- **证据等级**：官方一手（含设计与 guard；FP4 CP 注意力未完全支持）。

### 4.5 `#20078` opt-in DSpark decode 置信度调度（投机解码，默认关）
- **主题**：为内嵌 DeepSeek-V4 DSpark decode batch 启用置信度引导验证：训练好的 confidence head 给 draft 位置打分，纯 decode 步可验证更少行（在测得成本的预算间选），draft 仍产出整块、target 验证与接受不变。实验性、**默认关**。
- **内容详情**：`speculative_config` 加 `enable_confidence_scheduling`/`confidence_cost_table_path`/`confidence_calibration_path`/`confidence_min_predicted_gain`(默认 0.04)；成本表含多 K、按 `max_draft_len` 与捕获的 CUDA-graph batch size 选行；缺表/缺 K/缺 batch size/异 GPU 均启动失败并给生成命令；`python -m tensorrt_llm.tools.dspark_profile_costs` 生成/扩展成本表。PR6 of DSpark confidence stack，依赖已合入 #20072。
- **应用场景**：DSv4 DSpark 投机解码在纯 decode 步的验证开销精简（opt-in）。
- **研判**：【事实】PR 给配置项、成本表语义与 profiler 用法。【观点】属「投机解码」（第 5 条）——置信度引导验证是 DSpark 的细粒度开销控制；默认关、需成本表，落地成本较高，记官方一手（配置/工具级）。
- **来源**：TRT-LLM PR [#20078](https://github.com/NVIDIA/TensorRT-LLM/pull/20078)（merge 2026-11-03T23:49:09Z）。
- **证据等级**：官方一手（含配置与 profiler；默认关）。

### 4.6 `#21239` CuteDSL MoE 启用 tiled Triton NVFP4 量化（量化性能，描述级/未全绿）
- **主题**：为连续 CUDA BF16 `[T,H]` 输入新增 tiled Triton NVFP4 量化器（要求 `T>0`、`H` 为 1024 正倍数、同设备单元素 float32 scale）；CuteDSL grouped-GEMM NVFP4 经 `TRTLLM_MOE_TRITON_FP4_QUANT=1`（默认）或 per-call `enable_triton` opt-in；CUDA 量化器为 fallback，prequantized 走旁路，AWQ scale 在其前应用。
- **内容详情**：Dev/QA 均指出无 review severity 发现或源测试执行结果；**报告性能数字为作者报、未独立验证**；QA 覆盖结论 **needs follow-up** 以补可复现 perf 测试。新增 `test_nvfp4_quant.py` 与 `test_triton_nvfp4_input_quant.py`（字节级 CUDA 奇偶、rounding、非有限值、eligibility、CUDA graph replay、PDL 顺序）。
- **应用场景**：CuteDSL MoE 的 Triton NVFP4 输入量化（opt-in 默认开）。
- **研判**：【事实】PR 为 tiled Triton NVFP4 量化器 + 测试文件。**性能为作者报、未独立验证，记描述级/未全绿**——采用前以实测为准（与 #20486 同类）。
- **来源**：TRT-LLM PR [#21239](https://github.com/NVIDIA/TensorRT-LLM/pull/21239)（merge 2026-11-03T22:45:13Z）。
- **证据等级**：官方一手（描述级，perf 未独立验证、QA needs follow-up）。

### 4.7 `#20524` modeling_v2：GLM-5.2 NVFP4 低延迟 decode target for GB300（异构，描述级/未全绿）
- **主题**：为 SM103 新增 GLM-5.2 NVFP4 `modeling_v2` targets（TP8 / TEP / DEP），含 CUDA/Triton/CuTe/Torch 算子、dispatch、attention、MoE、投机解码、KV-cache、runtime-hook。opt-in `RuntimeHooks`（默认关）。
- **内容详情**：Dev/QA 评审指出跨 rank top-k 交换、Lamport-buffer、producer-consumer 启动序、PDL 序、timeout、确定性归约为主要正确性风险；kernel 编译、动态共享内存、collective workspace、aux stream、fallback 为性能风险；**260 源文件被省略/截断，覆盖率与测试充分性不确定**；QA 覆盖结论 **needs follow-up**，报告测试超现有 30-min CI 超时。
- **应用场景**：GLM-5.2 NVFP4 在 GB300（SM103）的低延迟 decode（modeling_v2）。
- **研判**：【事实】PR 为 GLM-5.2 NVFP4 GB300 target 大型改动。**260 文件评审被截断、测试超时，记描述级/未全绿**——采用前以 GB300 实测与完整 CI 为准（与往期 TRT-LLM 大模型 target 同类）。
- **来源**：TRT-LLM PR [#20524](https://github.com/NVIDIA/TensorRT-LLM/pull/20524)（merge 2026-11-03T08:22:38Z）。
- **证据等级**：官方一手（描述级，评审截断、CI 超时、needs follow-up）。

### 4.8 其余 TRT-LLM 要点（官方一手，标题/正文级）
- `#18914` **[infra] CI 启用 GCP_IAD**；`#20078`（见 4.5）；`#21239`（见 4.6）；`#20486`（见 4.1/4.3）；`#21283` **[fix] Lamport all-reduce clear size 以字节计**；`#19786` **[perf] DSA 跳过 indexer schedule rebuild（index-sharing draft 步）/ 去重 kv-lens hook**；`#20639` **[feat] GLM-5.3-Flash 稀疏注意力经 FlashInfer 跑 SM120**；`#20704` **[refactor] 集中 PyTorch 分布式初始化**；`#21565`/`#21562`/`#21542`/`#21507`/`#21387` **[infra] 多批 waive failed cases（main pre/post-merge）**；`#21198` **[perf] opt-in tiled QKV preprocessing**；`#21550` **[fix] GB300 Nemotron-Ultra-V3 TEP4 MTP6 perf 设 host_cache_size 0**；`#20398` **[infra] 保留 SLURM job 输出并过期清理**；`#20708` **[feat] KV transfer timeout 归因到 deadline owner**；`#21199` **[fix] 拒超单 encoder 迭代的 encoder 输入于准入**；`#21437` **[docs] 链接当前 PyTorch KV cache managers**；`#19757` **[fix] MiniMax-M2 tool-call 参数按 delta 流式**；`#21426` **[test] Muse Spark Open NVFP4 核心性能覆盖**；`#21251` **[fix] MiniCPM-V 4.6 vision attention 传 max_seq_len**；`#21326` **[infra] 提升格式化/VisualGen 的 CBTS 命中**；`#21487` **[fix] Visual-Gen Rubin 三处 MiniMax-H3 E2E timeout 延到 60 min**；`#21521` **[chore] 细化 infra 所有权**；`#21391` **[fix] Kimi-K3 尊重 Anthropic/chat 的 thinking 与 effort**；`#20868` **[fix] Responses 终态快照保留流报 status**；`#20975` **[fix] Rows 在 backend contract 命名 slots/regions**；`#21272` **[feat] 保护 direct BF16 context-first KV transfers**；`#20709` **[feat] Per View KV head transfer**；`#21245`（见 4.4）；`#21421` **[test] 延长 Qwen3 NVFP4 CUTLASS RTX timeout**；`#21176` **[chore] 移 trtllm-serve 死代码**；`#21436` **[docs] 量化 dynamic-tree sliding-window 兼容性**；`#20913` **[infra] nSpect 注册与 launch 检查分离、共享 release 脚本**；`#21430` **[infra] Python 依赖安装 timeout 延到 60 min**；`#21357` **[test] MXFP8 refit logit-overlap 阈值放宽到 0.85**；`#21502` **[fix] revert 8f04d3a 的 git pin**；`#19783` **[fix] 启用 NIXL LIBFABRIC 并更 container 内 EFA**；`#21302` **[fix] 给五个 QA list 用例加覆盖冷 checkpoint 读的边界**；`#21355` **[fix] 用 TRTLLM 生成的 FastWan LPIPS golden**；`#20907` **[feat] agent-flow 统一 trtllm_modeling_v2 workflow**；`#20937` **[perf] Qwen3.8-Flash-Next NVFP4 modeling_v2 GB300 ISL 8K/OSL 1K Pareto 优化**；`#20501` **[fix] VisualGen perf 测试识别 Rubin GPU**；`#21065` **[feat] Responses 提供 server 默认不 honour 的 web_search 工具（不可用拒绝 400）**；`#21095` **[fix] 修 NCCL scope overhead、benchmark graph 默认、KV bounce streams**；`#21435` **[docs] 拆分 parallelism YAML 为独立示例**。证据等级：官方一手（标题/正文级为主，多数为 infra/test/perf/chore）。

---

## 5. llama.cpp：2026-11-03 提交（36 条）——gemma4 图优化 + 量化静默路径 + NCCL 多节点

### 5.1 `#31356` gemma4：图优化（prefill 巨大加速，端侧/通用）
- **主题**：gemma-4 E2B/E4B 仍为不贡献 KV$ 的层计算；在首个 KV-shared 层前丢弃未用 prompt token，精简 PLE 投影（把两个 PLE 缩放常数折进 RMSNorm eps / norm 权重 / tok_embd_scale，用 stride 直接切片 permuted 3D PLE tensor 避免 ggml_cont 拷贝）。仅 unified-KV 支持层修剪。
- **内容详情**：M4 Pro (Metal)，gemma4 Q4_0：E2B pp512 1613.78→**4603.18 t/s（≈2.85×）**、pp2048 1535→4451、pp8192 1323→3748；E4B pp512 805.91→1318.62；tg128 基本不变（~110）。26B（所有层喂 KV$）无收益。16K/32K Needle 测试输出正常。
- **应用场景**：gemma-4 E2B/E4B 在 Apple Silicon / 端侧的 prefill 吞吐。
- **研判**：【事实】PR 含 M4 Pro Metal 多 pp 长度对照（E2B ≈2.85×）与 Needle 测试。【观点】属「端侧/通用 KV 效率」（第 4 条端侧同源）——通过丢弃不写 KV 的层计算直接换 prefill 加速；26B 无收益说明仅对 KV-shared 架构有效，采用前核对模型结构。
- **来源**：llama.cpp PR [#31356](https://github.com/ggml-org/llama.cpp/pull/31356)（merge 2026-11-03T18:30:38Z）。
- **证据等级**：官方一手（含 M4 Pro 多长度对照与 Needle 测试）。

### 5.2 `#30375` ggml：避免 `ggml_roll` 边界检查中的有符号溢出（整数未定义行为，强化第 9 条）
- **主题**：`ggml_roll` 用 `abs(shift)` 校验每个位移；`abs(INT_MIN)` 是有符号整数溢出（未定义行为），在边界检查运行前求值，使 `shift = INT_MIN` 时 guard 被绕过。
- **内容详情**：改为有符号安全比较 `shift > -ne && shift < ne`。UBSan 构建在 `shift0 = INT_MIN` 报 `abs(-2147483648)` 否定；打补丁后同输入触发 `GGML_ASSERT`。
- **应用场景**：`ggml_roll` 在 shift 极端值下的行为正确性（UB 消除）。
- **研判**：【事实】PR 给根因（abs(INT_MIN) UB 绕过 guard）与 UBSan 验证。【观点】与 vLLM `#65185`（仅编译告警）、SGLang/TRT/llama.cpp 的 int 系列同属「整数溢出/UB」主线（第 9 条在 UB 层延展）——`shift=INT_MIN` 下 guard 失效是隐蔽的 UB 触发点；属轻量但底层的正确性收口。
- **来源**：llama.cpp PR [#30375](https://github.com/ggml-org/llama.cpp/pull/30375)（merge 2026-11-03T11:26:30Z）。
- **证据等级**：官方一手（含 UBSan 验证；AI 辅助定位、人工复核）。

### 5.3 `#31079` vulkan：修复 NVFP4 UE4M3 scale 越界查找（量化静默正确性，第 6 条）
- **主题**：非 OCP 路径 `ue4m3_fp32_lut` 仅 128 项、用原始 scale 字节索引；`ggml_validate_row_data` 接受每个字节、`ggml_ue4m3_to_fp32` 处理全字节范围；CPU 解码排除 bit 7，但检查特殊值 0/0x7F 时比较全字节。scale 字节 ≥0x80 时 Vulkan 路径索引越界共享数组（Vulkan robustness 不覆盖共享内存）。
- **内容详情**：把表、两初始化循环、两 host 共享内存预算都扩到 256 项（`ue4m3_to_fp32_build` 已 mask 指数、只认 0/0x7F 特殊，128–255 与 CPU 解码一致）。每非 OCP NVFP4 shader workgroup 多需 512 字节共享内存；`ggml_fp32_to_ue4m3` 不设 bit 7，llama.cpp 量化的模型不触发越界。
- **应用场景**：Vulkan 后端 NVFP4（非 OCP）的 UE4M3 scale 解码边界。
- **研判**：【事实】PR 给根因（scale 字节 ≥0x80 越界、Vulkan 不覆盖共享内存）与 256 项修复。【观点】与 vLLM `#61755`/`#65496`、TRT `#20763`、llama.cpp `#31427` 同属「量化静默正确性」（第 6 条）——越界查表是「不报错、读出错值」的典型；仅影响非 OCP 路径，llama.cpp 量化模型不触发，但第三方量化可能命中。
- **来源**：llama.cpp PR [#31079](https://github.com/ggml-org/llama.cpp/pull/31079)（merge 2026-11-03T06:58:56Z）。
- **证据等级**：官方一手（含根因与 256 项修复；AI 辅助补丁与复现）。

### 5.4 `#31427` conversion：NVFP4 experts 用原始模型张量名（量化静默正确性，强化第 6 条）
- **主题**：`_flush_nvfp4_experts` 查找的张量名非原始模型名（如 `model.layers.1.mlp.experts.up_proj.weight`），而 `_nvfp4_layer_algo` 用原始 per-expert 名（`backbone.layers.1.mixer.experts.0.up_proj`）。查找永不匹配专家、回退到 `_nvfp4_global_algo`；若全局 algo 是 W4A16_NVFP4 则不被注意，但对 `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`（MIXED_PRECISION）专家不会得到 `prec_a4=false`，**在 Blackwell 上误走原生 W4A4 而非模型声明的 W4A16_NVFP4**。
- **内容详情**：给 `_flush_nvfp4_experts` 传「key→原始模型张量名」映射。
- **应用场景**：NVFP4 专家层转换时的张量名解析（MIXED_PRECISION 模型）。
- **研判**：【事实】PR 给根因（张量名空间不一致 → 回退全局 algo → MIXED_PRECISION 专家误走 W4A4）与修复（传原始名映射）。【观点】与 vLLM `#61755`/`#65496`、TRT `#20763` 同属「量化静默正确性」（第 6 条）——张量名空间不匹配致静默错精度路径，是转换链路的隐性债；建议 NVFP4 转换后核对专家层实际精度。
- **来源**：llama.cpp PR [#31427](https://github.com/ggml-org/llama.cpp/pull/31427)（merge 2026-11-03T10:40:51Z）。
- **证据等级**：官方一手（含根因与 MIXED_PRECISION 案例）。

### 5.5 `#30141` speculative：保存/恢复 MTP 隐藏状态（投机解码 checkpoint）
- **主题**：MTP 未实现 `get_state`/`set_state`，恢复 target 与 draft KV checkpoint 时把后续 batch 的隐藏状态留在投机驱动里。
- **内容详情**：经现有 hook 保存/恢复 `pending_h` 与验证隐藏行；改状态前校验模型宽度/行数/载荷大小，成功恢复后清瞬态 batch 数据。加 `test-speculative-mtp-state`：无补丁 `08246a2` 失败（`MTP hidden state was not saved`），有补丁在 Qwen3.5-2B（CPU/Metal）与 Gemma4 E2B + assistant（Metal, shared KV）通过。
- **应用场景**：MTP 投机解码的 checkpoint 保存/恢复正确性。
- **研判**：【事实】PR 给根因与测试（无补丁失败/有补丁通过，含 Qwen3.5-2B/Gemma4 E2B）。【观点】属「投机解码」（第 5 条）——MTP 隐藏状态未随 KV checkpoint 恢复致后续 batch 错，是 checkpoint/resume 的静默缺陷；Full CI 未跑。
- **来源**：llama.cpp PR [#30141](https://github.com/ggml-org/llama.cpp/pull/30141)（merge 2026-11-03T15:53:58Z）。
- **证据等级**：官方一手（含测试与模型验证；Full CI 未跑，AI 辅助实现）。

### 5.6 `#31581` metal：修复 dst 在 dim 0 非连续的 cpy kernel（端侧/Apple 静默错误）
- **主题**：#31446 新增 `test-backend-ops` CPY 用例（`permute_dst = {1,0,2,3}`）在 Metal 失败。dst 是 permuted view，`nb0 != type size`，`kernel_cpy_t_t` 用隐式 element-size stride 而非 `args.nb0` 写最内维，打乱输出。
- **内容详情**：改用 `args.nb0` 写元素；连续 dst 寻址不变。在 Apple M5 Ultra 测 `-o CPY`/`-o DUP`/`-o CONT` 与完整 `test-backend-ops -b MTL0` 套件通过。
- **应用场景**：Metal 后端 permuted view 的 CPY 算子正确性（#31446 引入用例）。
- **研判**：【事实】PR 给根因（`nb0` 误用）与 M5 Ultra 测试通过。【观点】属「端侧/Apple 静默错误」（第 4 条端侧同源）——permuted view 下 element-size stride 误用致输出乱序；属 #31446 引入用例暴露的回归修复。
- **来源**：llama.cpp PR [#31581](https://github.com/ggml-org/llama.cpp/pull/31581)（merge 2026-11-03T08:41:24Z）。
- **证据等级**：官方一手（含 M5 Ultra 测试；AI 辅助）。

### 5.7 `#28967` 多节点 tensor parallelism 的 NCCL 支持（多卡 TP，端侧/通用）
- **主题**：为跨独立进程 rank 的集合通信加后端接口，CUDA 后端用 NCCL 实现，并在 RPC server 可用时启用（基于 #26610）。
- **内容详情**：DGX spark 实测（unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M）：单节点 pp512 846.54 t/s；**两节点 NCCL pp512 1348.29 t/s（+~59%）**、tg128 13.02→23.51 t/s（+~80%）；两节点 pairwise 类似。
- **应用场景**：llama.cpp 跨节点 tensor parallel（RPC + NCCL）。
- **研判**：【事实】PR 含 DGX spark 单节点 vs 两节点 NCCL 的 pp/tg 对照（pp512 +59%）。【观点】属「端侧/通用多卡」（第 4 条端侧同源）——NCCL 多节点 TP 是 llama.cpp 多机推理的能力补强；实测为 DGX spark（GB10 类），其他集群以实测为准。
- **来源**：llama.cpp PR [#28967](https://github.com/ggml-org/llama.cpp/pull/28967)（merge 2026-11-03T09:26:20Z）。
- **证据等级**：官方一手（含 DGX spark 对照表）。

### 5.8 其余 llama.cpp 要点（官方一手，标题/正文级）
- `#31619` **mtmd qwen3-vl 视频 prompt 格式精确支持**；`#30993` **model/mtmd 加 chatterbox TTS**；`#31552` **hexagon 加 TOP_K_MASK op**；`#31279` **hexagon 更快 backend sampler TOP_K**；`#31596` **ui 安装 Chrome sandbox 跑本地浏览器测试**；`#30953` **hexagon 加 DSV4_HC_PRE/POST ops**；`#31583` **ggml-zdnn 修 fattn 测试失败**；`#31134` **加 Metis-1.6 模型支持**；`#31608` **启用 NVFP4 LFM3 ggufs**；`#31471` **mtmd 加 Nemotron 3.5 Transcribe**；`#31443` **qwen4exp 无中间 cont 写 conv state tail**；`#31247` **cuda Q8_1 correction sum 潜在修复**；`#31391` **cuda 把 batched tensor 读聚集进 mapped host memory**；`#31579` **GGUF split merge 拒绝不匹配 shards**；`#31114` **vulkan 非 compute tensor 跳过 maxStorageBufferRange 预检**；`#30878` **delta_net_base 重构 conv state**；`#31558` **llama 从 llm_attn_mask 去冗余 cnv**；`#31292` **dsv4.1 部分 offload 的 context 测试修复**；`#30662` **cuda 防即将使用的 graph 过早驱逐**；`#31339` **opencl 加初始 iq4_xs 支持**；`#31556` **chat gpt-oss final channel 后允许 constraint type**；`#31550` **HIP VGPR spill ignore list 为 kv_rows FA kernel 参数更新**；`#31455` **webgpu 加 ARANGE 支持**；`#31570` **kv-cache 无 cell 的序列绝不返回 0 kv_rows**（#29510 unified kv 上下文，正文为空、记标题级）；`#31446` **hexagon tiled hvx permute**；`#31506` **dflash 经 device copies 拷隐藏特征**。证据等级：官方一手（标题/正文级为主，多为端侧/异构后端/UI/转换小修）。

---

## 6. 累计结论与研判（沿用 + 强化）

**沿用 2026-11-03 档案（覆盖 11-02）第 1–10 条：**
1. 调度器隐式阻塞的定量画像（host sync 移出热路径的 +4.5%~9.9% 画像）；
2. **发版窗口的双向性**：发版当天合入 main ≠ 在版内（compare API 祖先关系核验）；
3. **默认值/能力移除成为最主要的风险/收益载体**（默认后端/量化/移除 Medusa/MLP speculator/legacy Ray executor）；
4. **异构后端静默错误高频**（AMD ROCm / Metal / 异构 kernel / aiter 不报错只掉精度）；
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**（草稿/索引 KV 所有权一致性、索引器 TP/PP 分片）；
6. **量化路径静默错误**（scale 未应用/参数丢失/4D 量化漏算/舍入 1 ulp / 张量名不匹配 → 输出偏离且无报错）；
7. **专家张量/路由归属与分页漏接静默错误**（paged-experts 不被 paging、路由行覆盖、FP32 路由精度/放置 → 静默用错专家值）；
8. **静默正确性升维为跨引擎常态主线**（高频失效模式已从「崩溃」转为「不报错、精度静默掉/长上下文静默塌方」）；
9. **长上下文 / 大张量 int32 溢出为静默正确性新子项**（Triton 将 `<2^31` 整数参数按 int32 传递，Qwen3.5 #63761 在 ~209K token 回绕）；
10. **量化 KV 与 KV 卸载的正确性成为新集中暴露面**（MXFP8/FP8 KV 的 scale 初始化、host pool 钉页/卸载、L3 存储 round-trip）。

**本期强化（发布次日稳定化，铁证级延展）：**
- **第 9 条（int 溢出 / UB）延展到「有符号溢出 / 未定义行为」层**：llama.cpp `#30375`（`ggml_roll` 的 `abs(INT_MIN)` 有符号溢出 UB，guard 被绕过，UBSan 验证）把 int 主线从「运行时回绕」推到「编译期/UB 层」；vLLM `#65185` 仅修复 `topk_histogram_4096_overflow.cuh` 的**编译告警**（非运行时溢出，须准确降级，不可计入运行时回绕）。跨引擎整数正确性已从显式回绕（#63761/#64820/#47029/#21342/#64614）扩到 UB 层。
- **第 4 / 6 / 8 条（异构/量化静默正确）多条铁证**：vLLM `#61755`（NVFP4 K-padded 静默垃圾，困惑度 1290959→29.45）、`#65496`（MiniMax-M3.1 MXFP8+SP 启动崩）、`#64580`（DSA BF16 indexer logits 改动 0.4–0.75% token）；llama.cpp `#31079`（Vulkan NVFP4 UE4M3 scale 越界）、`#31427`（NVFP4 experts 张量名不匹配 → Blackwell 误走 W4A4）；TRT-LLM `#20763`（量化 `exclude_modules` 顺序随哈希变 → 跨进程不同指纹、**静默不共享权重**）。这是「量化链路漏接」最隐蔽的一例——非确定性指纹致 TP 各 rank 权重错位且**无报错**。
- **第 7 条（MoE 路由精度/一致性）**：vLLM `#57167`（拒绝非连续 routing 张量——此前静默读错专家、ZAYA1 `#65302` 命中）、SGLang `#48041`（hash-routed `topk_config` 在 breakable-prefill graph 捕获时崩）、`#47641`（单 warp/token JIT MoE router，B200 路由 2–4×）从「防错/加速」两线补强。
- **第 5 / 8 条（PD/调度/KV 一致性）大巡检**：vLLM `#65413`（UMA extensible KV 低估 84% → `psutil` 校正 216→1302 blocks）、`#65319`（batch 满时上限异步 KV 加载）、`#65279`（KV Offload SWA retry 偏移错位）；TRT-LLM `#21432`（SINK_REQUIRED_ONLY 解锁混合精度 KV 前缀复用，命中 80.4%→89.4%、p90 TTFT 333→72s）、`#21245`（Helix owner-only MLA KV）；SGLang `#47476`（paged-experts `auto` 定容）、`#48099`（Mega MoE localization KV −1.47%→−0.10%）。KV cache 容量与一致性成为本期最密集主线。
- **第 3 条（默认值/能力移除风险）反向延展——「已合入的激进优化也可能被回退」**：SGLang 单日三连 revert（`#48055`/`#48054`/`#47994`）把模型特定 autotune tactic / JIT fusion 撤回通用路径，提示「默认/已合入的 perf 策略」同样构成稳定性风险载体，serving 方须锁版本并比对 revert 前后 perf/精度。

---

## 7. 趋势判断与可执行清单

**趋势判断**
1. **发布次日进入「稳定化 + 容量大巡检」模式**：vLLM 124 条为追踪以来单日最高，且集中在 extensible KV 容量（UMA 84% 低估修复）、量化静默修复、投机解码 CUDA graph、MoE 路由防错——与 v0.33.0 发布后的补丁/收尾潮一致。
2. **KV cache 容量与一致性取代「单点静默错误」成为本期头号密度主线**：extensible KV（UMA/margin/段补齐）、P/D 异步加载上限、KV Offload retry 偏移、混合精度 KV 前缀复用（TRT `#21432` 实测 p90 TTFT 333→72s）、Helix owner-only MLA、paged-experts auto 定容——覆盖「容量定容 / 异步加载 / 卸载一致性 / 混合精度复用 / 专家常驻」五个子面。
3. **量化静默正确性持续高频，且出现「非确定性指纹」新形态**：TRT `#20763` 的 `exclude_modules` 顺序随 `PYTHONHASHSEED` 变化致跨进程权重指纹不同、静默不共享权重，是「量化链路静默错误」中最隐蔽的一类（无报错、TP 各 rank 权重错位）。
4. **激进 perf 策略回退信号（SGLang 三连 revert）**：模型特定 autotune tactic / JIT fusion 在 CI 覆盖外不稳定被撤回，建议把「锁版本 + revert 前后 perf/精度比对」纳入升级门禁。
5. **端侧/异构双线并进**：llama.cpp `#31356`（gemma4 E2B prefill ≈2.85× on M4 Pro）、`#28967`（NCCL 多节点 TP +59% pp512）强化端侧吞吐；Vulkan/hexagon/Metal 量化与路由修复延续。

**可执行清单（建议 serving / 研发方）**
- [ ] **UMA / 大显存设备 KV 容量核对**：vLLM `#65413`（extensible KV `psutil` 校正，GB10 216→1302 blocks）、`#65225`/`#65418`——UMA 设备上线前核对实际 KV 块数，勿信 `cudaMemGetInfo` 原始值（第 8 条）。
- [ ] **量化静默正确性门禁**：vLLM `#61755`/`#65496`/`#64580`、llama.cpp `#31079`/`#31427`、TRT `#20763`——NVFP4/MXFP8 模型启用前核对 (a) K-padded 输出、(b) SP+量化激活类型、(c) indexer logits dtype、(d) 跨进程 `SourceIdentity` 指纹一致（第 4/6 条）。
- [ ] **混合精度 KV 前缀复用评估**：TRT `#21432`（SINK_REQUIRED_ONLY，命中 80.4%→89.4%、p90 TTFT 333→72s，opt-in）——长上下文多轮 agentic 服务评估（第 8 条）。
- [ ] **MoE 路由防错**：vLLM `#57167`（连续 routing 张量）、SGLang `#48041`（hash-routed `topk_config`）——自定义 routing / hash MoE 上线前核对（第 7 条）。
- [ ] **投机解码 CUDA graph / 验收器**：vLLM `#60989`（MRV2 draft-model CUDA graph，接受率 2.80）、`#65175`（XPress + Acceptance Estimator）；TRT `#20078`（DSpark 置信度调度，默认关）按场景评估（第 5 条）。
- [ ] **SGLang 版本锁定 + revert 比对**：单日三连 revert（`#48055`/`#48054`/`#47994`）提示激进 perf 策略回退风险，升级 SGLang 时锁版本并比对 revert 前后 perf/精度（第 3 条）。
- [ ] **llama.cpp 采纳**：`#31356`（gemma4 E2B/E4B prefill ≈2.85×，仅 KV-shared 架构）、`#28967`（NCCL 多节点 TP +59% pp512）、`#30141`（MTP 隐藏状态 checkpoint）按场景评估（第 4 条端侧）。
- [ ] **未全绿项以最终 CI + 实测为准**：TRT `#20486`/`#21239`/`#20524`（Dev/QA 均 needs follow-up、perf 未独立验证）、`#31570`（正文为空、标题级）——采用前以发行分支与实测为准。

---

## 8. 来源索引与数据质量

**官方一手（提交 API + 发布说明 + PR 正文 + compare API）**
- **vLLM**：https://github.com/vllm-project/vllm （**124 commits @2026-11-03**，均 HTTP 200；**v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）为窗口前一日发布，11-03 全为 `main` 发布后补丁**）。PR `#65413`/`#61755`/`#65496`/`#64580`/`#57167`/`#60989`/`#62565`/`#65319`/`#65279`/`#65185` 正文已抓。compare 核验：`#65413`（合入 23:57Z，三档点 `diverged`，ahead_by=667）、`#65185`（合入 14:57Z，`diverged`，ahead_by=615）对 v0.33.0rc3 **均未在版内确认**（发布说明非包含证明）。
- **SGLang**：https://github.com/sgl-project/sglang （**62 commits @2026-11-03**，均 HTTP 200；**v0.5.23（2026-11-02T22:14:10Z）为窗口前一日发布**）。PR `#41916`/`#47641`/`#48041`/`#47476`/`#48099`/`#47186`/`#43910`/`#48055` 正文已抓。compare 核验：`#41916`（合入 20:06Z，`diverged`，ahead_by=586）、`#48041`（合入 12:34Z，`diverged`，ahead_by=578）对 v0.5.23 **均未在版内确认**。
- **TensorRT-LLM**：https://github.com/NVIDIA/TensorRT-LLM （**53 commits @2026-11-03**；**v1.4.0rc2 2026-10-28T15:51:34Z** prerelease，窗口内无更新）；PR `#21432`/`#20763`/`#20486`/`#21245`/`#20078`/`#21239`/`#20524` 正文已抓。`#20486`/`#21239`/`#20524` Dev/QA 评审均标注 needs follow-up、perf 未独立验证（描述级/未全绿）。
- **llama.cpp**：https://github.com/ggml-org/llama.cpp （**36 commits @2026-11-03**；nightly 标签窗口内延续约 `b12062–b12085`，releases API 显示 `b12086`/`b12087`/`b12088` 均于 **2026-11-04T00:xxZ** 发布，即窗口刚过）；PR `#31356`/`#30375`/`#31079`/`#31427`/`#30141`/`#31581`/`#28967`/`#31570` 正文已抓（`#31570` 正文为空、记标题级）。

**社区 / 外部（二手线索，未核验）**
- HN（2026-11-03 窗口，Algolia HTTPS，单参数 AND 组合 `created_at_i>1130380800,created_at_i<1130467199`）：引擎相关**无高信号官方更新故事**。`vllm` 515 命中实为模糊匹配（多为「LLM」/招聘帖/无关技术，如「V100 SXM2 pinout」「Ask HN: Who wants to be hired」等），**非 vLLM 引擎更新**；`sglang` 2、`llama.cpp` 9、`TensorRT-LLM` 0、`KV cache` 6、`speculative decoding` 2、`LLM inference` 32、`inference optimization` 6 亦多为招聘/无关帖。个别边缘项：`Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction`（KV cache 研究帖，1 分，二手线索/未核验）、「A 3KB LLM inference engine in C」（Show HN 玩具，二手）、「Choosing an LLM Inference Optimization Technique: A Decision-Tree Approach」（eigenBasis，2 分，二手）。均**记为二手线索/未核验**（非官方引擎更新），不计入结论。
- Reddit / YouTube / X：未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

**数据质量与限制**
- 计数口径：GitHub commit API 的 `since`/`until` UTC 日期窗口（vLLM 124 / SGLang 62 / TRT-LLM 53 / llama.cpp 36，均 HTTP 200）；按 commit date 过滤。**2026-11-03 为周二，vLLM 124 为追踪以来单日最高**（超 10-28 的 102、10-27 的 99），与 v0.33.0 发布次日补丁潮一致；SGLang/TRT/llama.cpp 量较 11-02（93/42/29）有升有降（62/53/36），无第二页。
- 证据等级：带验证表 / 对照 / UBSan / 跨进程指纹分析的记为「官方一手（含验证）」；仅提交标题或正文未全抓的记为「标题级」（如 llama.cpp `#31570` 正文空）；TRT-LLM `#20486`/`#21239`/`#20524` 记「官方一手（描述级，Dev/QA needs follow-up、perf 未独立验证）」；vLLM `#65185` 记「官方一手（编译告警级，非运行时溢出）」；社区文章记「二手线索/未核验」。
- **版本包含核验说明（本任务既定方法）**：对 11-03 合入的 `#65413`/`#65185`（vLLM）、`#41916`/`#48041`（SGLang）做 compare API 三档点核验，结果均为 `diverged`（ahead_by 数百、behind_by 2–12），故**按既定规则记为「未在版内确认」**——与「发布说明非包含证明」口径一致；四 PR 均于 2026-11-03（v0.33.0/v0.5.23 发布次日）合入 main，是否随后续 patch release 以发行分支为准。
- 未调用外部浏览器（agent-browser）：本期数据源（GitHub API / releases / HN Algolia / PR issues）均可直接 HTTP 访问，沿用 2026-10-18 起确立的「API + PR 正文直抓」方法，未触发浏览器需求（与历次一致）。
- `find-skills`：经 SkillHub 语义检索（含「LLM inference GitHub commit monitoring」「github trending hotspot daily digest」「git sync commit push」「大模型推理 GitHub 提交采集 热点归档 仓库同步」）最高相关性约 0.050，返回编程专家.Skill / dev-expert / 育儿大师 / 技能发现·Find Skills 等通用编程/生产力类技能，**与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，未安装**（与历次一致）；本地 marketplace 目录为空。

> **归档空档说明**：本档案覆盖 **2026-11-03** UTC 窗口；**2026-10-29** UTC 窗口因 2026-10-30 当日自动化未执行而无档案（即 2026-10-29 档案覆盖 10-28、2026-10-31 档案覆盖 10-30、2026-11-01 档案覆盖 10-31、2026-11-02 档案覆盖 11-01、2026-11-03 档案覆盖 11-02、本期 2026-11-04 档案覆盖 11-03 之间，仅 2026-10-29 窗口一日空档）；**自 2026-10-30 窗口起已连续五期（2026-10-31、2026-11-01、2026-11-02、2026-11-03、2026-11-04 五期档案分别覆盖 10-30、10-31、11-01、11-02、11-03）无空档**。本目录不补写中间日期内容，各期仅按自身关注窗口采集。如需回溯 2026-10-29 窗口，请以各引擎官方提交 API 按日期区间重新采集。
> 本档案仅覆盖 2026-11-03 UTC 窗口；相邻日期请以官方提交 API 按日期区间重新采集。
