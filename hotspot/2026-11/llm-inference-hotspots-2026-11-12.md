# 大模型推理优化每日热点档案

> 档案日期：2026-11-12（本自动化于该日执行）
> 关注窗口：2026-11-11（周三）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 经 issues API 抓取 markdown 官方一手；vLLM 因 API 配额耗尽改由 PR HTML 页面抓取正文并做 div 深度配对提取）+ HN Algolia（HTTPS，numericFilters 按 2026-11-11 epoch 窗口 1794355200–1794441600）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 87 / SGLang 98 / TRT-LLM 49 / llama.cpp 31 合计 265 条；四家单页均未触 per_page=100 上限（无截断）；releases API 确认窗口内**无新正式版本**（vLLM 仍为 v0.33.0rc3、SGLang v0.5.23、TRT-LLM v1.4.0rc3；llama.cpp 仅推进 nightly `b12287`–`b12292`，均 prerelease）；~30 个关键 PR 正文经 issues API / PR 页面抓取（官方一手，含标题级/测试级/描述级标注）；HN Algolia 引擎关键词高信号弱（无四家官方更新类故事，仅社区/Show HN，二手线索/未核验）。本窗口为周三，提交量较周二（11-10：77/93/48/42）回升至 87/98/49/31，属正常工作日波动；API 配额在抓取中段耗尽（剩余 7、约 55 分钟重置），故 vLLM 末批 10 个 PR 改走 HTML 页面抓取，不消耗 API 配额。

## 1. 版本状态与本期头号事件：投机解码(DSpark/DFlash/DFLASH)跨引擎验证路径硬化 + 量化静默正确性(MXFP4/NVFP4/FP8)延续 + KV 卸载/PD 资源回收(TRT-LLM KVCM2 sparse offload) + 跨后端可移植性重构(SGLang device-neutral graph)

**提交计数（2026-11-11 UTC，均 HTTP 200；2026-11-11 为周三，量较周二回升；vLLM 87 / SGLang 98 / TRT-LLM 49 / llama.cpp 31）**：
- **vLLM 87** / **SGLang 98** / **TensorRT-LLM 49** / **llama.cpp 31**。

**发布状态（窗口内无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-11 全部 87 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-11 含 Kimi-K3.1 DFlash2 草稿、DFlash/DFLASH 自适应验证栈、AMD aiter 量化栈、MoE/triton_kernel 后端清理、device-neutral graph 重构。
- **TensorRT-LLM**：仍为 **v1.4.0rc3（True/prerelease，2026-11-06T21:38:32Z）**；11-11 含 sparse KV offload 接 KVCM2、DSA prefill indexer、PD KV 资源回收、spec-decode Triton 特化消除、GLM-5.3-Flash NVFP4 MTP-Eagle。
- **llama.cpp**：仍为 **v0.9.0（2026-11-04T13:08:06Z 正式发布）**；11-11 推进 nightly `b12287`（18:15Z）/ `b12289`（19:03Z）/ `b12290`（19:32Z）/ `b12291`（19:52Z）/ `b12292`（20:15Z，prerelease），无新正式版本。

**本期头号事件（投机解码跨引擎验证路径硬化 + 量化静默正确性延续 + KV 卸载/PD 资源回收 + 跨后端可移植性重构）**：
1. **投机解码验证路径跨引擎密集硬化（新增第 14 条观察）**：vLLM `#61869`（DSv4.1 DSpark 专用 decode，draft 与 target 同 executor、同 fused all-reduce+mHC kernel、vocab 分片 Markov 链）、`#63796`（DSpark sliding-window MLA draft 在 DCP 下复制 draft cache + NIXL 拉取）、`#62769`（跨 TP rank 同步自适应 DSpark 决策）；SGLang `#49621`（Kimi-K3.1 DFlash2 sliding-window MLA draft 接 DCP/`cutedsl_mla`）、`#46185`/`#46186`（DFLASH/DSPARK 在 ReplaySSM spec ring 上验证 + radix top-k 验证目标分布，原为 jianc99 分支组件化迁入）、`#44588`（DSpark 置信调度在线接受率估计器）；TRT-LLM `#21666`（spec-decode Triton kernel 不再按 batch 相关参数特化，消除 Helix/DSpark 服务中 5–12 s 重编译停顿）、`#21047`（Gumbel-max 文本水印 + 投机解码支持）、`#20487`（NGramPoolManager 候选剪枝/单匹配不变量修复）。投机解码从「能力落地」进入「验证路径正确性 + 服务稳定性」精细化阶段。
2. **量化静默正确性延续且出现新铁证（强化第 4/6/13 条）**：vLLM `#64329`（FlashInfer 当 query 为 BF16/FP16 时仍把 layer 的 FP8 `q_scale` 喂给 kernel，softmax scale 被乘错 → 非 unit q_scale checkpoint 输出错，补 `get_q_scale` 仅对 FP8 query 返回 scale、其余返回 1.0）、SGLang `#48397`（GLM-5.2-MXFP4 在 8×MI355X `gluon` TP8 长 prefill 下，fused MoE 的 FP32 down-proj partials 被 Triton buffer-op 描述符 `num_records=INT_MAX-1` 截断丢弃 → 长 prompt 输出错、RULER2 64K 由 0%→68.7%，属 AMD 量化 MoE 静默错误）；TRT-LLM `#20847`（MiniMax-M3 支持 MXFP8 基底层 + NVFP4 routed experts 的 compressed-tensors 打包，loader 校验组件一致性）、`#21787`（GLM-5.3-Flash NVFP4 SM100 拆分/fused MTP-Eagle targets）；llama.cpp `#28413`（CUDA 首秀 OCP FP8 支持，MoE/kernel 与 tile-FP8 后续跟进，描述级）。FP8/NVFP4/MXFP4 的 scale/布局静默错仍是高频错误源。
3. **KV 卸载 / PD 资源回收主线（强化第 8 条）**：TRT-LLM `#21350`（sparse KV offload 接 KVCM2——DSv4 / GLM-5.2(DSA) 在 decode 工作节点把 sparse attention 选中的 KV 历史放 pinned host，每步仅把 indexer 选中的 unit 取回 per-request GPU cache；MLA latent(DSv4 压缩比 4) 与 DSA index-K 下 host，indexer/sliding-window 留 GPU）、`#21909`（解聚 coordinator 双路径回收失败/超时 GEN 接收泄漏的 KV 内存——idle decode worker 上长期持有的 KV 现由 `_retire_legacy_timed_out_gen_transfers` 每轮轮询回收）、`#22050`（FullCtx decode 对齐 PrimTS 运行时 ABI）；vLLM `#66731`（extensible KV cache 容量预热现覆盖「多请求满预算步」、且 `max_num_seqs=1` 不再跳过，前置 #66197 的 OOM 已确定复现）、`#64087`（ROCm `reshape_and_cache` 每 token 超过 512 个 16-byte chunk 时漏写 KV head，ROCM_ATTN 静默读陈旧 KV、最大误差 0.097–0.196→≤0.0006）、`#61797`（UMBP[9/10] 从 GPU free queue 选可复用前缀块做 lazy offload）、`#65943`（Mooncake store connector 复用同 worker 的 PD TransferEngine，避免第二次 RDMA 初始化，418 passed）、`#62566`（KV Offload 次层 KV 与并行配置绑定时告警）。
4. **跨后端可移植性成为显式主线（新增第 15 条观察）**：SGLang `#49402`（引入 device-neutral `BaseGraphBackend`，为树外 OOT breakable graph 后端铺路，解耦 CUDA 假设——1/3，后续解耦 `eager_on_graph` 与让 OOT 平台提供实现）、`#49660`（删除 `--moe-runner-backend triton_kernel`，从 sgl-kernel wheel 移除 vendored `triton_kernels` 包的最大一步）、`#49661`（MXFP4 MoE 反量化改 JIT kernel）、`#49662`（sgl-kernel wheel 不再打包 triton_kernels）。属「推理引擎后端可移植性」架构主线，与 Apple MPS（#47921）、NPU（#40141/#41495）、ROCm 多后端并列。
5. **AMD/ROCm 量化与注意力栈延续（强化第 4/12 条）**：vLLM `#65479`（MiniMax-M3.1 MXFP8 linear 走 AITER 调优 FlyDSL GEMM）、`#56451`（MiniMax M3 在 ROCm 启用自适应 DSpark 验证）、`#63878`（ROCm DSA prefill loop 尊重 `skip_kv_gather`）、`#66876`（ROCm 允许 `lut_b_moe` 在线量化）；SGLang `#44458`（gfx1250 上 aiter 的 Triton MLA decode）、`#49567`（aiter breakable-graph attention 输出 in-place 写）、`#48810`（mxfp4 权重布局/dispatch 与所选 runner 一致）、`#48382`（AMD MoRI DSA state 跨 PP prefill stage 配对）、`#47911`（aiter 跳过 vectorized_5d KV 布局未用的 decode workspace）、`#49428`（pin `sgl-amd-mono` 到 7113024f 以稳定当前 SGLang 的 DSpark）。

---

## 2. vLLM：2026-11-11 提交（87 条）—— 量化静默正确性(#64329/#64087) + DSpark 专用 decode/DCP(#61869/#63796) + KV 卸载/PD(#66731/#61797/#65943) + Humming FP8 激活(#65077)

### 2.1 `#66731` [Bugfix] Size extensible KV cache reserve on the full token budget（可扩展 KV cache 容量预热修满预算步，强化第 8 条）
- **主题**：使 extensible KV cache 的 sizing warmup 测量调度器「实际能跑的最大步」；此前在 `max_model_len < max_num_batched_tokens` 时漏测多请求满预算步，且 `max_num_seqs=1` 时直接跳过。
- **内容详情**：把瞬态 reserve 覆盖到「把一个 prompt 拆到多请求」也能跑的满 `max_num_batched_tokens` 步；`max_num_seqs=1` 时改跑 prefill-only 步而非跳过；prompt logprobs 仍按单 prompt 步测（`max_logprobs=-1` 不为整预算预留全词表 logprobs）。与前置 #66197 配合：GB200 上 `test_flashinfer_replayssm_prefix_cache_mtp`（H100 上因 reserve 0.83 vs 1.04 GiB 确定性 OOM 322 MiB）现 reserve 1.61/1.61 GiB 通过；单引擎 `max_num_seqs=1` reserve 1.15→1.24 GiB。
- **应用场景**：可扩展 KV cache 容量峰值预热，避免长步 OOM。
- **研判**：【事实】PR 给 GB200/单引擎 reserve 对照表与 H100 OOM 复现。【观点】属「KV 容量自检/预热」收口（第 8 条），前置 #66197 的 OOM 已确定复现，本 PR 补满预算步测量；纯容量逻辑，无精度评测。**证据等级**官方一手（含 reserve 对照表）。
- **来源**：vllm-project/vllm #66731（合并于 2026-11-11T23:24:06Z）。

### 2.2 `#64165` [Bugfix] Fix FlashInfer plan buffer race causing illegal memory access with PP（FlashInfer plan 缓冲竞态 → 非法访存，强化第 4 条）
- **主题**：修复 #63505——`FlashInferMetadataBuilder` 可能在 GPU 把上一步 plan 从前一 page-locked 缓冲拷出之前再次调用 `plan()`；上一步 attention kernel 用「下一步的 plan」跑、越界读；随 #57214 变得可达。
- **内容详情**：builder 在再次 plan 前等待该拷贝完成；等待覆盖 `build()` 内每次 `plan()`（prefill、DCP prefill、decode 的 `fast_plan_decode`）；TRTLLM/XQA 路径不调 `plan()` 故不等待也不记录；仅当「领先 GPU 超过一个 plan」时 host 才等（单卡 async 调度 2000 次 build 0 阻塞）。新增 `test_flashinfer_plan_waits_for_previous_plan_copy`（prefill+decode）无 fix 失败、有 fix 通过。
- **应用场景**：Qwen3.8-27B-FP8 + `--pipeline-parallel-size=3` + chunked prefill + prefix caching 的首个长冷 prefill。
- **研判**：【事实】PR 给修复机制与新增测试（无 fix 失败）。【观点】属「注意力元数据构建竞态 → 非法访存」修复（第 4 条），PP 下首次长冷 prefill 杀引擎；记为官方一手（fix 级，含测试）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：vllm-project/vllm #64165（合并于 2026-11-11T22:59:57Z）。

### 2.3 `#64329` [Bugfix] Apply the layer q_scale only to FP8 queries in FlashInfer（FP8 q_scale 误用于 BF16 query → 错误输出，强化第 4/6/13 条）
- **主题**：当到达 FlashInfer 的 query 是 BF16/FP16 时，`FlashInferImpl.forward` 仍把 layer 的 FP8 `q_scale`（`layer._q_scale_float`）喂给 kernel，softmax scale 被乘上非 1 scale → 非 unit q_scale checkpoint 输出错。
- **内容详情**：新增 `FlashInferImpl.get_q_scale(layer, q_data_type)`，仅对 FP8 query 返回 layer q_scale、其余返回 1.0；所有向 FlashInfer 传 q scale 的地方改用它；并覆盖 `bmm1_scale`（TRTLLM/XQA decode 与 TRTLLM prefill 在 SM100 读取）。相对 #53754 的改进：本 PR 有走真实 `FlashInferImpl.forward` 的测试，且把规则也应用到 `bmm1_scale`（仅改 #53754 在 B200 + `disable_flashinfer_q_quantization` 下仍给垃圾文本、两行 TRTLLM 测试仍失败）。
- **应用场景**：FP8 KV + BF16/FP16 query 的注意力（SM100/SM103/SM107 等）。
- **研判**：【事实】PR 给「BF16 query 被乘非 1 q_scale → 输出错」根因与新增测试（两 TRTLLM 行无 fix 失败）。【观点】属「量化 attention 静默错误」（第 4/6/13 条），dtype 与 scale 不匹配导致的注意力输出错；记为官方一手（fix 级，含测试）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：vllm-project/vllm #64329（合并于 2026-11-11T18:00:03Z）。

### 2.4 `#64087` [Bugfix][Kernel] Fix reshape & cache dropping KV heads past 512 chunks（ROCm reshape_and_cache 漏写 KV head，静默错，强化第 4 条）
- **主题**：`reshape_and_cache` 当某 token 超过 512 个 16-byte chunk 时漏写 KV head，ROCm `ROCM_ATTN` 对 32 KV heads × head size 192/256 等形状静默读陈旧 KV（修复 #62697 的 Pattern B）。
- **内容详情**：恢复 kernel 内 thread-stride 循环，写满任意 `num_kv_heads * head_size`；此前 `num_kv_heads * head_size * cache_element_size > 8192` 字节时跳过 512 之后的 chunk。记者脚本下 #62697 三个 Pattern B 用例与 `TRITON_ATTN` 一致（max error ≤ 0.0006，此前 0.097–0.196）；其余测试无变化、适配形状无性能变化。Pattern A（gfx1201 上 block size 528 decode 读）是独立问题、本 PR 未覆盖。
- **应用场景**：ROCm `ROCM_ATTN` 下大 head 尺寸/高 KV head 数的注意力缓存写入。
- **研判**：【事实】PR 给修复机制与 max error 对照（0.097–0.196→≤0.0006）。【观点】属「ROCm KV 缓存写入静默错」（第 4 条），Pattern B 闭环；记为官方一手（fix 级，含误差对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：vllm-project/vllm #64087（合并于 2026-11-11T16:24:00Z）。

### 2.5 `#61869` [Perf][DSv4.1] DSpark draft on the specialized decode（DSv4.1 DSpark 专用 decode，perf，强化第 5/14 条）
- **主题**：DSv4.1-Flash 在 GB300 的 `latency_specialized` 专用 decode（#61868 的 DSpark 一半）：`decode_backend=latency_specialized` 时 DSpark draft 层与目标模型跑在同一 decode executor、同一融合 all-reduce + mHC kernel、attention 与 routed-expert GEMM；draft 也经模型的 `sample_sequential` 钩子采样（#64442 的融合 Markov 链 + 小 batch 下按词表分片到 TP rank 的 Markov 链）。
- **内容详情**：代码全部在 `vllm/models/deepseek_v41/nvidia/`，speculators/model runner/共享代码不变；开关关闭时行为不变。GB300 tray（TP4）+ DSpark（5 draft tokens），PR 对 main（含 #63921）：decode step −4.5%（batch1）/−3.4%（batch8）/−1.6%（batch40）；GSM8K 与 DSpark accept length 在重复 run 散布内不变。
- **应用场景**：DeepSeek-V4.1-Flash 在 GB300 的低延迟 DSpark 解码。
- **研判**：【事实】PR 给 decode step 降幅（−4.5%/−3.4%/−1.6%）与 GSM8K 持平。【观点】属「DSv4.1 DSpark 专用 decode 提速」（第 5/14 条），纯 perf、精度在散布内；记为官方一手（含 TP4 对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：vllm-project/vllm #61869（合并于 2026-11-11T21:14:53Z）。

### 2.6 `#65077` [Perf][Quantization] Honor FP8 activations for Humming linear on SM100/SM110（Humming 稠密层 FP8 激活提速，perf，强化第 12/13 条）
- **主题**：在 SM100–SM119 的 Humming 稠密 linear 上落实请求的 FP8 激活：27B MXFP6A8 checkpoint 上 B300 decode GEMM 实测 1.4–1.9×、并发 1–256 sweep 端到端 P50 TPOT −8%~−16%，GSM8K strict-match 0.681→0.683。
- **内容详情**：Quark MXFP4/MXFP6 稠密层落实 FP8 动态激活（含 MXFP6/MXFP8 转 FP8 E4M3、MXFP8 group-32/E8M0）；compressed-tensors MXFP6A8 走同路径；MoE grouped GEMM 在 SM100–SM119 保留 BF16 回退、SM90/SM120+ 不变；FP4 请求 gate 保留，其 SM100–SM119 稠密回退由 INT8(BF16) 改为 per-token FP8 E4M3；Wikitext perplexity +0.060（CI 容差内）。
- **应用场景**：Humming 量化稠密后端（MXFP4/MXFP6/MXFP8）在 Blackwell 的 decode 提速。
- **研判**：【事实】PR 给 B300 decode 1.4–1.9×、TPOT −8%~−16%、GSM8K 0.681→0.683、ppl +0.060。【观点】属「量化稠密后端激活提速」（第 12/13 条），FP8 激活换吞吐、精度在容差内；记为官方一手（含 MXFP6A8 对照）。**证据等级**官方一手（含 kernel + 端到端对照）。
- **来源**：vllm-project/vllm #65077（合并于 2026-11-11T21:54:02Z）。

### 2.7 `#63796` [Spec Decode][DCP] DSpark: run sliding-window MLA drafts under DCP（DSpark 在 DCP 下复制 draft cache，强化第 5/11/14 条）
- **主题**：把 #63795 的 sliding-window MLA DSpark draft 层放到 DCP（解聚上下文并行）下跑：draft cache 在每个 DCP rank 复制（draft group DCP=1），NIXL connector 从某个 DCP prefill rank 拉复制的 draft cache，而 target KV 仍按 DCP slice 读。
- **内容详情**：rebase 到 main（#63795 已合）；main 上 DCP=1 的 windowed draft 已规划为自身复制 KV cache group（DSA indexer 把布局解到 BLHNC），不需 KV 规划改动。明确非 #61464（DBO+spec+DCP，碰不同解聚特性、不覆盖 draft-cache 复制或 NIXL draft 布局）。
- **应用场景**：DSpark sliding-window MLA draft 在 DCP 部署下的 KV 复用。
- **研判**：【观点】属「投机解码 × DCP 资源复用」（第 5/11/14 条），draft cache 复制 + NIXL 拉取；记为官方一手（feature/config 级）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #63796（合并于 2026-11-11T12:33:29Z）。

### 2.8 `#61530` [Model][DSV4.1] Support MXFP4-packed Engram embedding tables（MXFP4 打包 Engram 表，强化第 8 条 KV 卸载）
- **主题**：支持 DSv4.1 checkpoint 的 MXFP4 Engram 表（每字节 2 个 E2M1 + UE8M0 block-32 scale）；checkpoint metadata 选布局，既有 FP8 checkpoint 保留默认格式。
- **内容详情**：打包 Engram 表经 GPU / peer-HBM / pinned host / huge-page / DP-shared 存储路径加载；分配与 host 内存资格用真实打包行宽；MXFP4 把每 256 元素行存储从 264 降到 136 字节（含 scale，省 48.5%）；不支持的 block size/scale 格式/部分层列表在表分配前失败，打包表 FP8 输出在 kernel 启动前失败，DP 交换用 BF16；DSpark TP4 GSM8K CI 配置用已发布 checkpoint 且不再 pin 旧 FP8 revision。
- **应用场景**：DSv4.1 Engram（KV 卸载嵌入表）MXFP4 压缩存储。
- **研判**：【事实】PR 给存储 264→136 字节（−48.5%）与各存储路径覆盖。【观点】属「Engram/KV 卸载量化压缩」（第 8 条），MXFP4 打包省存储；记为官方一手（feature 级，含字节对照）。**证据等级**官方一手（feature 级）。
- **来源**：vllm-project/vllm #61530（合并于 2026-11-11T06:05:38Z）。

### 2.9 `#61797` [UMBP][9/10] Add lazy GPU cache offloading（UMBP lazy 卸载，强化第 8 条）
- **主题**：从 UMBP 全 PR 拆出：从 GPU free queue 选可复用前缀块做可选 lazy offload；提交 lazy store 前关闭 full-attention 前缀链，源块在传输完成前保持 pinned；仅当 free queue 逐出端可能移动时才扫 free queue，外加每秒一次 recheck；recheck 间跳过已确认 resident 的 key 查找、跳过 key 与前缀链都确认的块。
- **内容详情**：把 scan 找到的已完成 lazy 块按原序放回 free queue 前端空块之后、正常释放从别处拉入的前缀块；scheduler 不可达的 rank 报 `None`（而非 miss），失败的 lazy 查找/存储退避而非每步重试；仅实现层面改动，既有 UMBP handle 测试断言由「server 不可问时 False」改为 `None`；`lazy_offload` 默认 false。
- **应用场景**：vLLM UMBP 前缀块 lazy GPU→host 卸载。
- **研判**：【观点】属「KV/前缀 lazy 卸载」（第 8 条），free queue 选块 + pinned 保持 + 退避；记为官方一手（feature/impl 级，默认关）。**证据等级**官方一手（feature 级，默认关）。
- **来源**：vllm-project/vllm #61797（合并于 2026-11-11T10:37:51Z）。

### 2.10 `#65943` [KVConnector] Reuse the Mooncake PD TransferEngine in the store connector（Mooncake store 复用 PD TransferEngine，强化第 8 条）
- **主题**：opt-in `reuse_pd_transfer_engine` 让 MooncakeStoreConnector 挂载同 worker 内 MooncakeConnector 已创建的 TransferEngine，而非初始化第二个。
- **内容详情**：每 worker 一个 TransferEngine 服务 PD 与 KV-store 流量：无第二次 RDMA 初始化、GPU KV 注册、P2P 握手端点或 CPU segment mount；复用时 store 不注册 KV buffer，device/HiSparse host KV 仍由 MooncakeConnector 注册，故 store 端 sleep/wake 为 no-op、sleep 资格由 `MooncakeConnector.supports_sleep_mode` 决定；无 flag 时默认行为不变。418 passed。
- **应用场景**：Mooncake PD + KV store 同 worker 的资源去重。
- **研判**：【事实】PR 给「单 TransferEngine 服务双流量」机制与 418 passed。【观点】属「PD/KV 连接去重」（第 8 条），opt-in 无回归；记为官方一手（含 418 passed）。**证据等级**官方一手（含测试）。
- **来源**：vllm-project/vllm #65943（合并于 2026-11-11T01:04:14Z）。

### 2.11 其他要点（标题级 / 描述级 / 性能级，按 PR 正文缺失程度标注）
- `#56451` [ROCm][Spec Decode] Enable adaptive DSpark verification for MiniMax M3（ROCm 自适应 DSpark 验证，perf/feature 级）。
- `#62769` [Bugfix][Spec Decode] Synchronize adaptive DSpark decisions across TP ranks（跨 TP rank 同步自适应 DSpark 决策，fix 级）。
- `#63878` [ROCm][Perf][DSA] Honor skip_kv_gather in the ROCm sparse-indexer prefill loop（ROCm DSA 尊重 skip_kv_gather，perf 级）。
- `#65479` [ROCm][MiniMax-M3.1] MXFP8 linear on AITER's tuned FlyDSL GEMM（ROCm MXFP8 量化，perf 级）。
- `#66876` [ROCm][Bugfix] Allow lut_b_moe online quantization on ROCm（ROCm 允许 lut_b_moe 在线量化，fix 级）。
- `#772dafd295` [Quantization] Load ModelOpt NVFP4 W4A16 token embeddings in mixed-precision checkpoints（NVFP4 W4A16 token embedding 混合精度加载，feature 级）。
- `#66549` [Perf][GLM-5.3] Extend ShardQ token sharding to norms and A projections（GLM-5.3 ShardQ 分片到 norms/A projections，perf 级）。
- `#66552` [Perf][DSv4.1] Engram: 128-row lookup tiles; split GPU tables across engines（Engram 128 行 lookup tiles + 跨 engine 分表，perf 级）。
- `#66834` Remove the dual-chunk attention leftovers（移除 dual-chunk attention 残留，cleanup 级）。
- `#66465` [MRV2][PP] Support different CUDA graph padding across pipeline stages（MRV2+PP 不同 CUDA graph padding，feature 级）。
- `#62566` [KV Offload] Warn when secondary-tier KV is tied to the parallel config（KV Offload 次层 KV 与并行配置绑定告警，fix 级）。
- `#66007` [Bugfix][Model] Fix GraniteMoeHybrid pipeline-parallel startup（Granite MoE Hybrid PP 启动修复，fix 级）。
- `#66848` [Bugfix][Tokenizers] Raise ValueError on unknown message role in deepseek_v41_encoding（deepseek_v41 编码未知 role 转 ValueError，fix 级）。

---

## 3. SGLang：2026-11-11 提交（98 条）—— DFlash2/DSpark 验证栈(#49621/#46185/#46186) + AMD 量化静默错误(#48397) + device-neutral graph(#49402) + MoE/triton_kernel 清理(#49660/#49661)

### 3.1 `#49621` [Feat] dflash: Serve the Kimi-K3.1 DFlash2 sliding-window MLA draft（Kimi-K3.1 DFlash2 sliding-window MLA draft，强化第 14 条）
- **主题**：Kimi-K3.1 DFlash2 draft 是「所有层为 sliding_attention（window 1024）+ grouped conv corrector」的 MLA draft；main 上这种 draft 在 K3.1 DCP 下跑不了（override 要求 `--speculative-draft-attention-backend cutedsl_mla` 配 DCP+DFLASH，但 `DFlashMLAAttention` 对 sliding 层只接受 `triton`）。本 PR 把 Day-0 服务这些 draft 的代码（`trtllm_mla`/`cutedsl_mla`）与 grouped conv kernel 调用搬到 `srt`。
- **内容详情**：`trtllm_mla`（与继承其 target-verify 路径的 `cutedsl_mla`）用 trtllm-gen sparse MLA 经 per-row window key lists（`_forward_sliding_window`）跑 causal sliding 层；DCP 下返回 rank-local `(out, lse)`；kernel（`mla_window_sparse_indices`、`dflash_grouped_conv`）已在 main 上（#49411），本 PR 只改 `srt`。
- **应用场景**：Kimi-K3.1 DFlash2 sliding-window MLA 草稿在 DCP 下的服务。
- **研判**：【观点】属「Kimi-K3.1 DFlash2 草稿接 DCP」（第 14 条），从 Day-0 代码迁出、kernel 已在 main；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #49621（合并于 2026-11-11T12:30:43Z）。

### 3.2 `#46185` [Feat] gdn: run DFLASH and DSPARK verify on the ReplaySSM spec ring（DFLASH/DSPARK 在 ReplaySSM spec ring 验证，强化第 14 条）
- **主题**：DFlash 自适应验证系列组件 2（原实现 jianc99/sglang#2，集成 jianc99/sglang#46182）。在 Qwen3.5/3.8 GDN 目标上，DFLASH/DSPARK 原本每 draft token 写一份完整 SSM state snapshot（FlashInfer MTP 或 Triton recurrent verify）并事后 scatter 接受的那个；`--enable-linear-replayssm-spec` 用紧凑 per-request ring 避免快照，但此前对非 KDA 模型的 DFLASH/DSPARK 拒绝。
- **内容详情**：rebase 到 main（6a319e7b），pre-commit 通过；GPU 测试与 benchmark 在原始分支跑过、未因本次 upstream 更新重跑（注明）。提交准备用 AI 辅助。
- **应用场景**：GDN 目标的 DFLASH/DSPARK 自适应验证（ReplaySSM ring）。
- **研判**：【观点】属「DFLASH/DSPARK 验证路径硬化」（第 14 条），ReplaySSM ring 避免完整 SSM snapshot；注明 benchmark 未因 rebase 重跑，记为官方一手（feature 级，benchmark 未重跑需谨慎）。**证据等级**官方一手（feature 级，benchmark 未重跑）。
- **来源**：sgl-project/sglang #46185（合并于 2026-11-11T21:30:22Z）。

### 3.3 `#46186` [Perf] spec: radix top-k for speculative verify target probabilities（spec 验证目标分布 radix top-k，perf，强化第 14 条）
- **主题**：DFlash 自适应验证系列组件 3。DFLASH（selector/plain draft）、DSPARK、UNO dense 验证与 sampling-mask capture 在 `build_speculative_verify_target_probs` 构建采样验证目标分布；EAGLE 族/standalone/ngram 走自己的不变。
- **内容详情**：原实现把整个 `[rows, vocab]` logits 分块做 top-k；本 PR 改 radix top-k 以更省。rebase 到 main（6a319e7b），pre-commit 通过；GPU 测试与 benchmark 在原始分支跑过、未因 rebase 重跑（注明）。
- **应用场景**：投机验证目标概率构建（radix top-k）。
- **研判**：【观点】属「spec 验证目标分布优化」（第 14 条），radix top-k 替代全 logits top-k；注明 benchmark 未重跑，记为官方一手（perf 级，benchmark 未重跑）。**证据等级**官方一手（perf 级，benchmark 未重跑）。
- **来源**：sgl-project/sglang #46186（合并于 2026-11-11T21:28:23Z）。

### 3.4 `#48397` [Fix] [AMD] moe: Stop dropping FP32 partials in GLM Gluon TP8 32K MoE（GLM Gluon TP8 长 prefill 丢 FP32 partials，AMD 静默错误，强化第 4 条）
- **主题**：GLM-5.2-MXFP4 在 8×MI355X、`--moe-runner-backend gluon`（TP8/EP1，InferenceX PR 4748 配置）下长 prefill 两类症状：① agentic 流量下 8 卡有时一起 `HSA_STATUS_ERROR_EXCEPTION`（code 0x1016），常在满 32768-token prefill chunk；② 不崩时长 prompt 输出错，RULER2 64K 由 0%→68.7%。
- **内容详情**：根因在 `fused_moe_tp8_m4193_16768`：`_Workspace` 把转义的 FP32 down-proj partials 拆成 `parts_low`/`parts_high`（各 `4*storage_m x 6144` fp32，M=32768 时各 3 GiB），Triton AMD buffer-op 描述符把 `num_records` 设为 `INT_MAX-1`；注释称 buffer op 有 4-GiB 窗口——实际丢 partials。修后 RULER2 64K 0%→68.7%。
- **应用场景**：GLM-5.2-MXFP4 在 AMD MI355X `gluon` TP8 长 prefill。
- **研判**：【事实】PR 给 `HSA_STATUS_ERROR_EXCEPTION` + RULER2 64K 0%→68.7% 根因（FP32 partials 被 buffer-op `INT_MAX-1` 截断丢弃）。【观点】属「AMD 量化 MoE 静默错误」（第 4 条），长 prefill 下 partials 被丢致输出错/崩；记为官方一手（fix 级，含 RULER2 对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：sgl-project/sglang #48397（合并于 2026-11-11T00:58:23Z）。

### 3.5 `#49402` [Refactor] graph: Introduce device-neutral graph backend bases (1/3)（device-neutral graph backend，新增第 15 条）
- **主题**：`BaseCudaGraphBackend` 已被 CUDA/XPU/NPU 共享，其 CUDA 专属名不再贴切；在加入树外（OOT）breakable graph 后端前引入 device-neutral 接口。
- **内容详情**：OOT breakable graph 后端需改三层——① 类型与所有权模型（不假设 CUDA 描述 graph backend）；② 捕获机制（记录 graph break 不依赖 CUDA capture 实现）；③ 平台选择/runner 策略（让 OOT 平台提供实现并给 core 以 breakable 语义）。本 PR 为 1/3：引入 `BaseGraphBackend`。后续解耦 `eager_on_graph` 与 CUDA capture、让 OOT 平台提供实现。
- **应用场景**：SGLang 跨后端（CUDA/XPU/NPU/OOT）graph backend 抽象。
- **研判**：【观点】属「推理引擎后端可移植性」主线（第 15 条），device-neutral 抽象为 OOT breakable graph 铺路；记为官方一手（refactor 级，1/3）。**证据等级**官方一手（refactor 级）。
- **来源**：sgl-project/sglang #49402（合并于 2026-11-11T03:00:13Z）。

### 3.6 `#49660` [Refactor] moe: Remove the triton_kernel MoE runner backend（移除 triton_kernel MoE 后端，强化第 15 条）
- **主题**：`--moe-runner-backend triton_kernel` 经 sgl-kernel wheel vendored 的 `triton_kernels` 包跑 MoE；移除该后端是从 wheel 去除 `triton_kernels` 包的最大一步（4/4 栈，基于 PR1 #49659）。
- **内容详情**：删除 `MoeRunnerBackend.TRITON_KERNELS`/CLI 选项、`moe_runner/triton_kernels.py`、`fused_moe_triton/triton_kernels_moe.py`、`TRITON_KERNEL` top-k 输出格式及其 `triton_kernels.topk` 路由；`FusedMoE` 移除转置权重加载路径，`UnquantizedFusedMoEMethod` 去 `use_triton_kernels` 参数（调用方更新）；`Mxfp4MoEMethod` 去相应路径。
- **应用场景**：SGLang MoE runner 后端清理（去 vendored triton_kernels）。
- **研判**：【观点】属「MoE 后端可移植性/清理」（第 15 条），移除 triton_kernel 后端、为去 vendored 包铺路；记为官方一手（refactor 级）。**证据等级**官方一手（refactor 级）。
- **来源**：sgl-project/sglang #49660（合并于 2026-11-11T01:00:47Z）。

### 3.7 `#30893` [Refactor] hisparse: unify device-pool mapping and selection（HiSparse 池映射/选择统一，强化第 8 条）
- **主题**：实现 #23602 的「Unify HiSparse pool internals」：此前 `HiSparseC4DevicePool` 与 `HiSparseDSATokenToKVPool` 各自拥有同一 full-to-HiSparse 映射注册与翻译操作，易漂移。
- **内容详情**：新增 `HiSparseDevicePoolMappingMixin`（映射注册、raw lookup、public lookup、full→compressed→device 组合），compose 进 C4/DSA/MHA 设备池；压缩模型相关：C4 每四个 token 选一、DSA 为 identity-compressed；保持 dtype 契约（DSA/C4 各路径保持或转 int32）。无分配/buffer I/O 改变。
- **应用场景**：HiSparse（C4/DSA/MHA）设备池内部统一。
- **研判**：【观点】属「HiSparse KV 池内部统一」（第 8 条），消除双池漂移；记为官方一手（refactor 级，无分配改动）。**证据等级**官方一手（refactor 级）。
- **来源**：sgl-project/sglang #30893（合并于 2026-11-11T03:20:21Z）。

### 3.8 `#48316` [Fix] offload: Keep DSA indexer parameters resident on GPU under CPU offload（DSA indexer 参数在 CPU offload 下留 GPU，fix，强化第 8 条）
- **主题**：服务 DSA 模型（如 GLM-5.3 `GlmMoeDsaForCausalLM`）时开 V1 CPU weight offload（`--cpu-offload-gb>0`）+ 默认 breakable prefill CUDA graph backend，首 prefill 崩溃：`mat2 is on cpu, different from other tensors on cuda:0`。
- **内容详情**：栈：`prefill_cuda_graph_runner.replay_layer_forward` → `breakable_cuda_graph_backend.replay` → `dsa_prefill_cuda_graph.pcg_dsa_indexer_prefill_split` → `dsa_indexer`。DSA indexer 参数在 CPU offload 下被卸到 host，replay 时仍在 cuda 取用致设备不匹配；修后参数留 GPU。
- **应用场景**：GLM-5.3 等 DSA 模型 + CPU weight offload + breakable prefill graph。
- **研判**：【事实】PR 给崩溃栈与「DSA indexer 参数被 offload 到 host 致 device 不匹配」根因。【观点】属「DSA + CPU offload 静默/崩溃」（第 8 条），offload 边界错；记为官方一手（fix 级，含栈）。**证据等级**官方一手（fix 级，含栈）。
- **来源**：sgl-project/sglang #48316（合并于 2026-11-11T06:00:23Z）。

### 3.9 `#47921` [Feat] [Apple] mps: Serve Unlimited-OCR v2 BF16 on the Torch MPS runtime（Apple Silicon 跑 Unlimited-OCR v2 BF16，新增第 15 条可移植性）
- **主题**：Unlimited-OCR v2（#47196）原本仅 CUDA；本 PR 通过标准 Torch MPS runtime 在 Apple Silicon 上服务 BF16 checkpoint（沿用 #32321 的 Apple Silicon 方向：常规 Scheduler/TpModelWorker/Torch `ModelRunner` 持有全部状态，Metal 仅用于计算；不加 runner/进程/权重副本）。
- **内容详情**：在 MPS 上跑：processor、SAM+CLIP 视觉编码器、DASH sparse attention（entmax routing）、page2d RoPE、built-in page guard、n-gram processor、`--uocr-doc-split`（页面窗口间 KV tail reuse）；量化 checkpoint 在 MPS 路径仍被拒（`validate_mps_model_config`）。本 PR 为 (a) 在 MPS 上跑模型；另两步 (#48932 启动默认、后续) 在其上。
- **应用场景**：Apple Silicon（MPS）本地多模态 OCR 推理。
- **研判**：【观点】属「跨硬件可移植性」主线（第 15 条），Unlimited-OCR v2 上 Apple Silicon；记为官方一手（feature 级，量化仍拒）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #47921（合并于 2026-11-11T19:34:51Z）。

### 3.10 其他要点（标题级 / 描述级 / 性能级）
- `#44458` [Perf][AMD] attention: Use aiter's Triton MLA decode in the aiter backend on gfx1250（gfx1250 上 aiter Triton MLA decode，perf 级）。
- `#49567` [Perf][AMD] aiter: write the breakable-graph attention output in place（aiter breakable-graph attention 输出 in-place 写，perf 级）。
- `#48810` [Fix][AMD] mxfp4: Keep MoE weight layout and dispatch consistent with the selected runner（mxfp4 权重布局/dispatch 与 runner 一致，fix 级）。
- `#48382` [Fix][AMD] disaggregation: Pair MoRI DSA state across PP prefill stages（AMD MoRI DSA state 跨 PP prefill stage 配对，fix 级）。
- `#47911` [Fix][AMD] aiter: Skip the unused decode workspace on the vectorized_5d KV layout（aiter 跳过 vectorized_5d KV 布局未用 decode workspace，fix 级）。
- `#49428` [Fix][AMD] monokernels: Pin sgl-amd-mono at 7113024f for DSpark on current SGLang（pin `sgl-amd-mono` 以稳定当前 SGLang 的 DSpark，fix 级）。
- `#49357` [Fix] compressed-tensors: Size the FP8 MoE w13 by the layer's gating（FP8 MoE w13 按层 gating 定尺寸，fix 级）。
- `#49531` [Fix] GLM-4.7-Flash: Load compressed-tensors W4A16 checkpoints（GLM-4.7-Flash 加载 compressed-tensors W4A16，fix 级）。
- `#49661` [Refactor] mxfp4: Dequantize MXFP4 MoE weights with a JIT kernel（MXFP4 MoE 反量化改 JIT kernel，refactor 级，强化第 15 条）。
- `#49662` [Chore] sgl-kernel: Stop bundling triton_kernels in the sglang-kernel wheel（sgl-kernel wheel 不再打包 triton_kernels，chore 级）。
- `#49538` [Refactor] moe: Remove the SM120 top-22 small-M router path（移除 SM120 top-22 小 M router 路径，refactor 级）。
- `#49656` [Fix] kernels: Synchronize AOT MoE prefix before expert lookup（AOT MoE prefix 在 expert lookup 前同步，fix 级）。
- `#49458` [Feat] kimi-k3: add video support via frame sampling（Kimi-K3 经帧采样加视频支持，feature 级）。
- `#49499` [Perf] dsv4: Size expand_prefill_causally's token block by batch size（DSv4 expand_prefill_causally 按 batch 定 token block，perf 级）。
- `#48891` [Perf][SM100] attention: optimize sparse MLA prefill for small heads（SM100 sparse MLA prefill 小 head 优化，perf 级）。
- `#49497` [Perf] flashinfer-autotune: Tune MoE with varied warm-up tokens（flashinfer 自动调优按 warm-up token 调 MoE，perf 级）。
- `#41794` [Perf][AMD] kimi-k3: Run the fused KDA decode with a BF16 state and Quark weights（Kimi-K3 融合 KDA decode 用 BF16 state + Quark 权重，perf 级）。
- `#40141` [Feat][NPU] startup-weight-load: Support overlap mode on NPU（NPU 启动权重加载 overlap 模式，feature 级）。
- `#41495` [Feat][NPU] weight-cache: Add NPU weight cache support（NPU 权重缓存支持，feature 级）。
- `#44588` [Feat] spec: Online acceptance estimator for DSpark confidence-scheduled verification（DSpark 置信调度在线接受率估计器，feature 级，强化第 14 条）。
- `#49727` [Feat] sgl-router: Add production API compatibility（sgl-router 加生产 API 兼容，feature 级）。
- `#49414` [Fix] logprobs: Bound top_logprobs to [0,vocab_size]（top_logprobs 边界校验，fix 级）。
- `#49722` [Fix] scheduler: Apply pending chunked-prefill aborts in the PP PD prefill loop（PP PD prefill loop 应用挂起 chunked-prefill abort，fix 级）。
- `#46855` [Fix] scheduler: Reserve member rows of beam leaders admitted in the same prefill（beam leader member rows 保留，fix 级）。
- `#49751` [Refactor] scheduler: Remove the remaining direct request-list edits（调度器去残留直接 request-list 编辑，refactor 级）。

---

## 4. TensorRT-LLM：2026-11-11 提交（49 条）—— sparse KV offload 接 KVCM2(#21350) + DSA prefill indexer(#21892) + PD KV 回收(#21909) + spec-decode Triton 去特化(#21666) + GLM-5.3-Flash NVFP4(#21787)

### 4.1 `#21350` [TRTLLM-16288][feat] Integrate sparse KV offload for DeepSeek-V4 Flash and GLM5.2 on KVCM2（sparse KV offload 接 KVCM2，强化第 8 条）
- **主题**：在 KV cache manager v2 上为 DeepSeek-V4 与 DSA（GLM-5.2）层加 sparse KV cache offload，基于其 sparse attention core（#21129）。
- **内容详情**：decode-only worker 上 sparse attention 选中的 KV 历史放 pinned host 内存；每 decode 步把 indexer 选中的 unit 取回 per-request GPU cache，GPU 持每请求工作集而非整序列。下 host 的是 DSA 的 MLA latent cache 与 DSv4 的 ratio-4 压缩 cache；indexer 与 sliding-window cache 留 GPU。给出 resident vs offload 性能：同 NVFP4 权重、FP8 KV、同 32768-token prompt + 160 output/request，8×GB300 跑 3 个 TP2/EP2 prefill worker + 1 个 TP2/EP2 decoder（attention DP2）。
- **应用场景**：DSv4 / GLM-5.2 在 GB300 的 sparse KV offload。
- **研判**：【观点】属「KV 卸载落地 KVCM2」（第 8 条），sparse attention 选中历史 host 化、每步取回工作集；性能数字为描述级（PR 给对比框架但未给完整数值表），记为官方一手（feature 级，性能描述级）。**证据等级**官方一手（feature 级，性能描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21350（合并于 2026-11-11T02:20:35Z）。

### 4.2 `#21892` [None][feat] Fused DSA prefill indexer + exact top-k (throwaway top-k) on SM100/SM103（DSA prefill indexer 融合 + 精确 top-k，perf，强化第 11/14 条）
- **主题**：为 SM100/SM103 的 DSA prefill indexer 加 opt-in 融合 MQA indexer + 精确 top-k（"throwaway top-k"）；不物化 `[Q,K]` logits 矩阵即选每 query row 的 top-k keys，替代 DeepGEMM `fp8_mqa_logits`/`fp8_fp4_mqa_logits` + `TopK` 对（单请求 prefill chunk）。默认关，用 `TRTLLM_THROWAWAY_TOPK=1` 开。
- **内容详情**：kernel 为 CuTe DSL、来自 DKG throwaway top-k 包（revision `c2e1721163`，同 FlashInfer #7325）；四阶段 K1（weighted-ReLU 得下界）/K2（过滤候选）/K3（精确 top-k）/K4（重算候选不足/溢出行）；结果为精确 top-k（tie 可能不同解）。
- **应用场景**：SM100/SM103 DSA prefill indexer 精确 top-k。
- **研判**：【观点】属「DSA prefill indexer 优化」（第 11/14 条），融合 + 精确 top-k 省 logits 物化；默认关、opt-in，记为官方一手（feature 级，默认关）。**证据等级**官方一手（feature 级，默认关）。
- **来源**：NVIDIA/TensorRT-LLM #21892（合并于 2026-11-11T10:27:09Z）。

### 4.3 `#21909` [None][fix] Reclaim KV cache resources after failed or timed-out disaggregated transfers（PD 失败/超时 KV 回收，fix，强化第 8 条）
- **主题**：KV-cache 传输失败可能以两种方式泄漏 KV 内存，本 PR 修解聚 coordinator 两路径。
- **内容详情**：① 超时的 legacy GEN receive 在无 in-flight 取消支持时从不 retire（除非出现可运行 model batch）——idle decode worker 上永不来，故新增 `_retire_legacy_timed_out_gen_transfers()` 每轮 transfer poll 都查过期 receive：成功 `request_cancellation()` 授权释放内存（cancel 契约证明无传输仍持有页）、拒绝/失败则带 request 与其 KV 重试；② 另一条 reclamation 路径（未在截断正文展开，属解聚 GEN receive 生命周期）。
- **应用场景**：解聚（PD）传输失败/超时后的 KV 内存回收。
- **研判**：【观点】属「PD KV 资源回收」（第 8 条），idle worker 上超时被持 KV 现每轮回收；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：NVIDIA/TensorRT-LLM #21909（合并于 2026-11-11T13:34:47Z）。

### 4.4 `#21047` [TRTLLM-424][feat] Add Gumbel-max text watermarking with spec dec support（Gumbel-max 水印 + 投机解码支持，feature，强化第 14 条）
- **主题**：部分地区要求 AI 生成文本标记（如 EU AI 生成内容行为准则）；在 PyTorch backend 加 keyed Gumbel-max 文本水印。sampler 的随机抽替换为基于 secret + 最近若干生成 token 的抽，平均输出分布不变、持 key 的检测器可测信号。支持投机解码（one-model exact-match 验证、经第二 key 或 target-only 模式的拒绝采样）。客户主要需 Qwen3.x dense/MoE。
- **内容详情**：用法 `from tensorrt_llm import LLM, SamplingParams; from tensorrt_llm.watermark import GumbelWa...`；验证用 Qwen3.5 dense/MoE checkpoint；见 `docs/source/features/watermarking.md`。
- **应用场景**：合规文本水印（Qwen3.x）+ 投机解码。
- **研判**：【观点】属「水印 + 投机解码」（第 14 条），分布不变、可检测；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：NVIDIA/TensorRT-LLM #21047（合并于 2026-11-11T07:55:20Z）。

### 4.5 `#20847` [TRTLLM-17388][feat] MiniMax-M3: load the compressed-tensors NVFP4 MoE packaging（MiniMax-M3 NVFP4 MoE 打包加载，强化第 4/13 条）
- **主题**：MiniMax-M3 现支持 MXFP8 基底层 + NVFP4 routed experts 的 compressed-tensors checkpoint；无逐层量化 map 时 loader 把支持的 packaging metadata 展开为逐层设置，拒不支持的 expert recipe、量化未绑定 `lm_head` 配置、fused 投影的部分排除。
- **内容详情**：fused 投影可声明 checkpoint 组件名；量化要求所有声明组件有匹配项、拒冲突项；sparse KV-cache 估计现含 index-K 存储、paged index-K buffer 用同 byte-per-token 计算；部署指南记 checkpoint scale 转换、backend 支持、评测指引。
- **应用场景**：MiniMax-M3 的 MXFP8+NVFP4 混合量化 checkpoint 加载。
- **研判**：【观点】属「量化 MoE checkpoint 加载」（第 4/13 条），MXFP8+NVFP4 混合打包校验；记为官方一手（feature 级，loader 级）。**证据等级**官方一手（feature 级）。
- **来源**：NVIDIA/TensorRT-LLM #20847（合并于 2026-11-11T03:56:55Z）。

### 4.6 `#20487` [#20485][fix] Fix NGramPoolManager candidate pruning and single-match invariant（NGram 候选剪枝/单匹配不变量修复，fix，强化第 14 条）
- **主题**：修 `NGramPoolManager.get_draft_tokens` 两处逻辑缺陷。
- **内容详情**：① 候选剪枝：在 `OrderedSet` 上迭代时移除元素会跳过候选、留更短匹配在池——改迭代快照 `list(pool[pattern])` 保证所有更短匹配被考虑移除；② 单匹配不变量：`is_keep_all=False` 时现有 update 分支可额外加匹配进池——用 `self.is_keep_all` 守该分支保持单匹配。新增回归测试 `test_ngram_pool_manager_prunes_all_shorter_matches`/`is_keep_all_false_invariant`/`draft_order`。
- **应用场景**：n-gram 投机解码候选管理。
- **研判**：【事实】PR 给两处缺陷 + 三新增回归测试。【观点】属「n-gram 投机解码正确性」（第 14 条），候选剪枝/单匹配不变量；记为官方一手（fix 级，含测试）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：NVIDIA/TensorRT-LLM #20487（合并于 2026-11-11T15:26:34Z）。

### 4.7 `#21787` [None][feat] Add Modeling V2 GLM-5.3-Flash NVFP4 targets on SM100 with fused MTP-Eagle（GLM-5.3-Flash NVFP4 SM100 + fused MTP-Eagle，强化第 4/14 条）
- **主题**：为 SM100 加 GLM-5.3-Flash NVFP4 Modeling V2 targets：split target 与 fused MTP-Eagle target。路由检查 model shape/量化/硬件/并行/投机解码选项；不支持部署可回退 split target。
- **内容详情**：fused target 在 `forward` 内做 MTP-Eagle 工作、拒不支持的 speculation 模式；split target 支持 no speculation / vanilla MTP / MTP-Eagle / suffix-automaton drafting；给 `MTPForCausalLM` 加 `mtp_layer_cls` 处理；新 CPU 测试查该 hook 优先于 `model_type` 派发。两 target 的 model/weight-loading 代码大量新增，评审应核 split/fused loader 一致性与路由拒绝匹配 target。
- **应用场景**：GLM-5.3-Flash NVFP4 在 SM100 的 MTP-Eagle 投机解码。
- **研判**：【观点】属「量化 + 投机解码 target」（第 4/14 条），GLM-5.3-Flash NVFP4 SM100 + fused MTP-Eagle；记为官方一手（feature 级，loader 一致性待评审）。**证据等级**官方一手（feature 级）。
- **来源**：NVIDIA/TensorRT-LLM #21787（合并于 2026-11-11T00:37:19Z）。

### 4.8 `#22050` [https://nvbugs/7074476][fix] Align full-context decode with the PrimTS runtime ABI（FullCtx decode 对齐 PrimTS 运行时 ABI，fix）
- **主题**：#21354 的 PrimTS refresh 改了 compile 与 runtime decode 接口，但 `FullCtxPlanBase` 与 TP2/TP4/TP8 full-context wrappers 仍用旧 keyword 与 launch operand。
- **内容详情**：用当前 compile keyword 与 runtime KV metadata；经 PrimTS 共享 runtime adapter 启动（给主 kernel 与可选 reducer 扁平 query/output 张量、scales、runtime bounds）；这些 pinned recipe 仍关 `runtime_split_kv`；保留现有 capacity-strided 五维 `[requests, KV heads, splits, rows, head_dim]` split scratch 与精确 shape check（BF16 reducer 直接消费 workspace）；保留现有 reduction/MXFP8-output 分支与 strided QKV column view 所需 compact-query copy。
- **应用场景**：PrimTS full-context decode（pinned recipes）。
- **研判**：【观点】属「PrimTS runtime ABI 对齐」（decode 接口迁移收尾），无新能力；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：NVIDIA/TensorRT-LLM #22050（合并于 2026-11-11T10:08:45Z）。

### 4.9 `#21666` [TRTLLM-16624][fix] Do not specialize spec-decode Triton kernels on batch-dependent arguments（spec-decode Triton kernel 不再按 batch 特化，fix，强化第 14 条）
- **主题**：Triton 对整数参数 `==1`/`%16==0` 与指针参数 16-byte 对齐做特化，任一 key 变就编新变体。若干 spec-decode step kernel 的实参 key 跟随 batch 组成，故 warmup 后服务中编新变体。
- **内容详情**：受影响实参：request counts / first-row offsets / 由 row count（及单 vocab shard 时 pair buffer row stride）派生的 pair-layout offsets / 按 context request 数切的 view。GB300 + Kimi K3 + DSpark（K=5）下表现为 Helix TP1/CP8（NVFP4）5–12 s 停顿，普通 TP8（MXFP4）30 分钟 AgentX 测量窗内 20 次 kernel 编译。本 PR 把这些实参放进 `do_not_specialize`、指针走对齐路径，消除 mid-serving 重编译。
- **应用场景**：spec-decode（DSpark/Helix）服务中 Triton kernel 重编译停顿。
- **研判**：【事实】PR 给 GB300 K3 DSpark 5–12 s 停顿、TP8 MXFP4 30 分钟窗 20 次编译，根因为 batch 相关实参特化。【观点】属「投机解码服务稳定性」（第 14 条），消除 mid-serving 重编译；记为官方一手（fix 级，含停顿数据）。**证据等级**官方一手（fix 级，含停顿数据）。
- **来源**：NVIDIA/TensorRT-LLM #21666（合并于 2026-11-11T04:25:35Z）。

### 4.10 其他要点（标题级 / 描述级 / 性能级）
- `#21768` [TRTLLM-17659][refactor] Pin down the encoder-to-decoder handoff in PyExecutor（encoder→decoder 交接钉死，refactor 级）。
- `#20767` [TRTLLM-17210][feat] Qualify tied word embeddings for the MX Qwen3 dense profile（MX Qwen3 dense profile 的 tied word embeddings 资格，feature 级）。
- `#21807` [None][doc] Add a tech blog on dense DWDP（dense DWDP 技术博客，doc 级）。
- `#21993` [None][fix] Skip the KV host tier in the Gemma4 dummy E2E tests（Gemma4 dummy E2E 跳过 KV host tier，fix 级）。
- `#21717` [None][fix] Collect SLURM CBTS coverage before uploading results（上传结果前收集 SLURM CBTS 覆盖，fix 级）。
- `#21990` [None][test] remove DSpark AgentX perf cases（移除 DSpark AgentX perf cases，test 级）。
- `#21968` [https://nvbugs/5651865][test] Unwaive one GPT-OSS TP4 CUTLASS FP8 accuracy test（GPT-OSS TP4 CUTLASS FP8 精度测试 unwaive，test 级）。
- `#22126` [https://nvbugs/6487918][test] Unwaive Qwen3.5-4B BF16 accuracy on GB300（Qwen3.5-4B BF16 GB300 精度 unwaive，test 级）。
- `#21703` [TRTLLM-17599][test] Run the host-only MoE unit tests in the CPU stage（host-only MoE 单测入 CPU stage，test 级）。
- `#21929` [None][chore] Remove Wan2.1 from VisualGen perf sanity（VisualGen perf sanity 移除 Wan2.1，chore 级）。
- `#2e652e9280` [TRTLLM-17136][feat] Serve Muse Spark Open: reasoning and tool parsers, effort scale, native tools（Muse Spark Open 服务：推理/工具解析/effort scale/原生工具，feature 级）。

---

## 5. llama.cpp：2026-11-11 提交（31 条）—— CUDA OCP FP8(#28413) + FlashAttention 广度(#27997/#30285) + CUDA MMVF(#31959) + CUDA graphs(#31284) + 多后端修复(#32103/#31838/#29810/#31526/#31784)

### 5.1 `#28413` CUDA: Add OCP FP8 support（CUDA 首秀 OCP FP8 支持，强化第 4/13 条）
- **主题**：为 CUDA backend 实现 OCP FP8 支持。
- **内容详情**：deferred 到后续 PR：① MoE kernels（倾向把那些层量化到 4/2-bit kernel）；② 在 tile-based（prefill）用 FP8-native；③ 让 FP8 MMVF 尊重累加器与 src1 精度规格。Perf snapshot 用 Qwen3.8-27B-NVFP4 FP8 gguf、`-p 8196 -ub 512,2048 --load-mode dio`（具体数值 PR 未完整给出，记为描述级/perf 级）。
- **应用场景**：llama.cpp CUDA 后端的 OCP FP8 张量计算。
- **研判**：【观点】属「CUDA FP8 支持落地」（第 4/13 条），首秀、MoE/tile/torch 累加器后续；perf 数字为 snapshot 描述级未完整，记为官方一手（feature 级，描述级）。**证据等级**官方一手（feature 级，perf 描述级）。
- **来源**：ggml-org/llama.cpp #28413（合并于 2026-11-11T08:44:30Z）。

### 5.2 `#27997` cuda : add sm70 FlashAttention config case for DKQ=256, DV=256, ncols=64（sm70 FA config 优化 ~20%，perf）
- **主题**：加 SM70 FlashAttention config，head 256×256 此前回退到 Ampere config 对 SM70 次优；为优化 Qwen3.8-27B agentic（用 DKQ=256/DV=256/ncols=64）。
- **内容详情**：上游回退：255 registers/thread、552 B stack；改后：255 registers/thread、56 B stack。模拟 agentic turn：100k cached KV + 1024 PP + 512 TG，repeats 4。PP upstream vs edited 与 TG upstream 对照（具体数值 PR 未完整给出，记为 perf 级）。
- **应用场景**：llama.cpp 在 SM70（Turing 之前/Volta 类）FlashAttention 长上下文。
- **研判**：【观点】属「CUDA FlashAttention config 广度」，SM70 专用 config ~20%（PR 标题）；具体 PP/TG 数值未完整给出，记为官方一手（perf 级，描述级）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：ggml-org/llama.cpp #27997（合并于 2026-11-11T08:45:34Z）。

### 5.3 `#30285` cuda : fattn mma head sizes 40 72（CUDA fattn 支持 head size 40/72，perf/正确性）
- **主题**：head size 40 与 72（用于 Gemma 3/4、Qwen3-VL 视觉塔等）原先被排除在所有 tensor-core dispatch 分支外、回退到 tile kernel。
- **内容详情**：fp16 MMA fragment 在 head 方向覆盖 16 个 half，故 in-kernel 几何 padding 到下一 16 倍数（40→48、72→80）；register 数组、shared memory tile stride、MMA loop bound 用 padded size，global memory load/store 用真实 size；在 NVIDIA Turing+ 启用、Volta/AMD 保留 tile kernel；RDNA/CDNA config 行与 `mma_tile_sizes` 特化因显式实例化需保留。
- **应用场景**：llama.cpp CUDA 视觉塔等非常规 head size 的 FlashAttention。
- **研判**：【观点】属「CUDA FlashAttention head size 覆盖」，40/72 接入 tensor-core、不再 tile 回退；记为官方一手（feature/perf 级）。**证据等级**官方一手（feature 级）。
- **来源**：ggml-org/llama.cpp #30285（合并于 2026-11-11T08:45:01Z）。

### 5.4 `#31959` cuda : use MMVF for BF16 matrices of few rows at small batch sizes（CUDA MMVF 优化少行 BF16 矩阵，perf）
- **主题**：少行 BF16 矩阵在 batch 2–8 时 MMF 仅发几 block、多数 SM 空闲（如 qwen4exp indexer 投影 128×2560/512×2560、hyper-connection 投影 320×10240）；MMVF 每行发一 block，行×batch 相对 SM 数小时更快。
- **内容详情**：gate：`K >= 1024 && n <= max_ncols && rows * n * 256 <= nsm * min(K, max_k)`，Blackwell `max_ncols=8, max_k=6144`、老 GPU `max_ncols=5, max_k=4096`；常量由 252 形状 sweep（rows 64–2048、K 1024–10240、n 2–8，6 GPU）+ sweep 外形状校验得出；少 SM GPU（A…）需谨慎。
- **应用场景**：llama.cpp CUDA 小 batch 少行 BF16 GEMM（indexer/hyper-connection 投影）。
- **研判**：【观点】属「CUDA 小矩阵 GEMM 调度」，MMVF 替代 MMF 提 SM 占用；记为官方一手（perf 级，含 sweep 方法）。**证据等级**官方一手（perf 级）。
- **来源**：ggml-org/llama.cpp #31959（合并于 2026-11-11T08:46:13Z）。

### 5.5 `#31284` backend : re-instantiate CUDA graphs when the cluster dims of a kernel change（CUDA graph 重实例化修复，fix）
- **主题**：Hopper 上 cuBLAS F16 GEMM 用 thread block cluster，cluster dims 随形状变（m=1200: grid 80,1,1 cluster 5,1,1；m=1040: grid 40,2,1 cluster 2,2,1，同 kernel 函数）。ggml-cuda 重捕获新形状 graph 调 `cudaGraphExecUpdate` 返回 success 但保留旧 cluster dims，下个 launch `illegal instruction`。首遇为官方 LiquidAI/LFM2.5-VL-…。
- **内容详情**：修法为当 kernel 的 cluster dims 变化时重实例化 CUDA graph（而非仅 update）。
- **应用场景**：llama.cpp CUDA graph + Hopper cuBLAS cluster GEMM。
- **研判**：【事实】PR 给「cudaGraphExecUpdate 保留旧 cluster dims → illegal instruction」根因与 LiquidAI/LFM2.5-VL 触发。【观点】属「CUDA graph 重捕获正确性」（fix 级）；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：ggml-org/llama.cpp #31284（合并于 2026-11-11T12:00:45Z）。

### 5.6 `#31838` ROCm/HIP: fixed BLAS workspace instead of rocBLAS allocating one per call（ROCm BLAS workspace 修复，fix）
- **主题**：HIP build 上 llama.cpp 不装 BLAS workspace，rocBLAS 每调用分配 scratch；改为 HIP 上也装固定 32 MB workspace（镜像 #26574 在 NVIDIA 对 cuBLAS 的做法，修 #31108）。
- **内容详情**：改前 `test-backend-ops -o SOLVE_TRI` 以 `HSA_STATUS_ERROR_MEMORY_FAULT` 中止；改后 24/24 通过（连续 5 次）。独立 reproducer：rocBLAS 4/5 faults、caller-supplied workspace 0/5；AMD 自身 rocblas-test 同样 faults。
- **应用场景**：llama.cpp ROCm/HIP BLAS（SOLVE_TRI 等）稳定性。
- **研判**：【事实】PR 给「24/24 通过、rocBLAS 4/5 faults → workspace 0/5」对照。【观点】属「ROCm BLAS 稳定性修复」（fix 级），固定 32 MB workspace；记为官方一手（fix 级，含对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：ggml-org/llama.cpp #31838（合并于 2026-11-11T11:13:28Z）。

### 5.7 `#29810` cpu: accumulate flash attention V in F32（CPU FA V 用 F32 累加，fix）
- **主题**：单 chunk CPU flash attention kernel（decode 与小 batch）把 F16 V 和留在 F16，长上下文丢精度、可 overflow 到 inf/NaN；tiled kernel 已用 F32，GGML_PREC 契约也要求 F32。
- **内容详情**：V 现对所有类型用 F32 累加；F16 行经 `ggml_vec_mad_f16_f32`（同 #31854）in-register 加宽。decode 速度 x86 不变、FA 与 CUDA（4364/4364）和 Metal（5344/5344）匹配，@pestopoppa 复现 max error 由 master 0.297 → 4.5e-5。修 #29774。
- **应用场景**：llama.cpp CPU backend 长上下文 flash attention 数值正确性。
- **研判**：【事实】PR 给「max error 0.297 → 4.5e-5」与 FA 匹配 CUDA/Metal。【观点】属「CPU 注意力数值正确性」（fix 级），F16 V 累加→F32；记为官方一手（fix 级，含误差对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：ggml-org/llama.cpp #29810（合并于 2026-11-11T10:55:19Z）。

### 5.8 `#31526` vulkan: allow dequantized KV optimization with CM2（Vulkan KV 反量化优化 + 对齐修复，perf）
- **主题**：用 coopmat2 启用 KV 反量化优化（基于 query tokens×GQA ratio 启发式）；过程中发现某些 head size 被错误当未对齐、修对齐后大提速。
- **内容详情**：head size 96/192/576/512 有对齐 bug、256 没有；修对齐 + 反量化优化：修对齐本身 >2×，仅反量化优化约 30%。`-o FLASH_ATTN_EXT -p "^(hsk=192,hsv=192...)"` perf 对照（具体数值 PR 未完整给出，记为 perf 级）。
- **应用场景**：llama.cpp Vulkan backend KV 反量化注意力。
- **研判**：【观点】属「Vulkan KV 注意力性能」，对齐修复 + 反量化优化（>2×/约 30%）；具体数值未完整给出，记为官方一手（perf 级，描述级）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：ggml-org/llama.cpp #31526（合并于 2026-11-11T10:08:06Z）。

### 5.9 `#32103` metal : prevent out-of-bounds reads in SOLVE_TRI kernel（Metal SOLVE_TRI OOB 读修复，fix）
- **主题**：修 #32099——Metal SOLVE_TRI kernel 越界读，可致无效 device load、中断执行。
- **内容详情**：在 `solve_tri.metal` 加 `rr + sgitg < N` 与 `idx < N` 边界检查，读 `dst_ptr[idx*K]` 前加 `idx < r` 检查；在 test-backend-ops.cpp 加 N=34/79 回归（输出缓冲预填 NaN 以暴露未计算元素的无效 SIMD lane 读）。改前：Metal shader validation 报 Invalid device load → SIGABRT；改后通过。
- **应用场景**：llama.cpp Metal backend SOLVE_TRI（BLAS 三角解）。
- **研判**：【事实】PR 给「Invalid device load → SIGABRT」与 N=34/79 回归（预填 NaN）。【观点】属「Metal 越界读修复」（fix 级）；记为官方一手（fix 级，含回归）。**证据等级**官方一手（fix 级，含回归）。
- **来源**：ggml-org/llama.cpp #32103（合并于 2026-11-11T20:17:46Z）。

### 5.10 `#32078` llama : remove non-fused GDN and ubatch-dependent graph nodes（移除非融合 GDN 与 ubatch 依赖图节点，refactor/perf）
- **主题**：移除 non-fused Gated Delta Net 路径与 graph node budget 的 ubatch-size 依赖。
- **内容详情**：`build_delta_net_chunking()`/`build_delta_net_autoregressive()` 仅在 fused GDN 关闭时可达；自 #27877 fused path 总用，故删这些实现、`fused_gdn_ar/ch`/`auto_fgdn` 标志、GDN probes 与 fused-node tags；`llama_context::graph_max_nodes()` 不再取 `n_tokens`，去 #17794 引入的 `n_tokens*160/80/40` 项、保留 `n_tensors` 预算；删 Qwen3.5/Clef/Clover/Qwen3.8 的 dead head-repeat 分支。
- **应用场景**：llama.cpp GDN 后端与图节点预算。
- **研判**：【观点】属「GDN 后端清理」，删非融合路径 + ubatch 依赖图预算；记为官方一手（refactor 级）。**证据等级**官方一手（refactor 级）。
- **来源**：ggml-org/llama.cpp #32078（合并于 2026-11-11T13:16:16Z）。

### 5.11 `#32000` quantize: fix imatrix GGUF filepath metadata leak（imatrix GGUF 元数据路径泄漏修复，fix/安全）
- **主题**：修复 imatrix 文件路径与 dataset 路径被存进 `quantize.imatrix.file`/`quantize.imatrix.dataset` 元数据、可能泄漏公司文件系统或用户信息；现只存文件名。
- **内容详情**：reproducer：`llama-imatrix` 生成 imatrix、再 `llama-quantize --imatrix`；改前 gguf_dump 显示完整绝对路径（含 `/home/<user>/...`），改后仅文件名。
- **应用场景**：llama.cpp 量化流程的 GGUF 元数据隐私。
- **研判**：【事实】PR 给 reproducer 与「完整绝对路径 → 仅文件名」修复。【观点】属「量化元数据隐私泄漏」（fix 级），info-leak 修复；记为官方一手（fix 级，含 reproducer）。**证据等级**官方一手（fix 级，含 reproducer）。
- **来源**：ggml-org/llama.cpp #32000（合并于 2026-11-11T11:33:12Z）。

### 5.12 `#31784` sycl : fixed sycl CLAMP to actually support strided view（SYCL CLAMP 支持 strided view 修复，fix）
- **主题**：改 `ggml_sycl_op_clamp` 用 unary helper `ggml_sycl_detail::ggml_sycl_op_unary` 处理非连续/strided view，与其他 stride-safe unary op 同模式。
- **内容详情**：删 dead template `unary_op_clamp_kernel`、flat `clamp` kernel launcher（element_wise.cpp）、删（现未用）`SYCL_CLAMP_BLOCK_SIZE`（presets.hpp）。
- **应用场景**：llama.cpp SYCL backend CLAMP 算子 strided 输入。
- **研判**：【观点】属「SYCL 算子 strided 支持修复」（fix 级），删 dead kernel、改走共享 unary helper；记为官方一手（fix 级）。**证据等级**官方一手（fix 级）。
- **来源**：ggml-org/llama.cpp #31784（合并于 2026-11-11T05:48:20Z）。

### 5.13 其他要点（标题级 / 描述级）
- `#32098` server: fix --cors-origins handling + protect against DNS rebinding（server CORS 处理 + DNS rebinding 防护，fix/安全级）。
- `#31877` webgpu: add PAD_REFLECT_1D support（WebGPU PAD_REFLECT_1D 支持，feature 级）。
- `#32089` llama : reuse graph result instances across scheduler reserves（跨 scheduler reserves 复用 graph result 实例，perf/refactor 级）。
- `#32072` jinja : support chained comparisons and argument unpacking（jinja 链式比较 + 参数解包，feature 级）。
- `#32053` chat: allow TTS model to provide custom text normalizer via jinja（TTS 模型经 jinja 自定义 text normalizer，feature 级）。
- `#32008` vulkan: allocate the device fault vendor binary buffer（Vulkan 设备故障 vendor binary buffer 分配，fix 级）。
- `#055c3c26ef` vulkan: support batched tensor copy API（Vulkan 批量张量拷贝 API，feature 级）。
- `#32083`/`#32077` test-fp8.cpp / tests : use only the public ggml API（测试改只用公开 ggml API，test 级）。
- `#32084` ci : remove dead snakes（CI 清理，chore 级）。
- `#32005` devops : add missing MKL runtime DLLs to the Windows SYCL release package（Windows SYCL 发布包补 MKL runtime DLL，fix 级）。
- `#2656b71a19` vendor : update LibreSSL to 4.4.1（vendor LibreSSL 4.4.1，chore 级）。
- `#32049` vendor: update cpp-httplib to 0.66.0（vendor cpp-httplib 0.66.0，chore 级）。
- `#32096` ci: move test models to ggml-org-dev（CI 测试模型移到 ggml-org-dev，ci 级）。
- `#7bcbc04fbf`/`#dd00306bcf`/`#1d81992784`/`#bd717e08f3`/`#adde94c11b`/`#bbadbc1545`/`#8551e3a13f` UI/文档/服务器前端杂项（ui/docs/server-frontend 级）。

---

## 6. 累计结论延续与新增（继承 1–13，本期新增 14、15）

**累计结论（1–13 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性（发版当天合入 ≠ 在版内）；③ 默认值成为主要风险载体；④ 异构/量化后端静默错误为高频；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务；⑥ 量化路径静默错误（scale/布局漏接）；⑦ 专家张量/路由归属与分页漏接静默错误；⑧ KV 卸载/HiCache/PD 健壮性；⑨ 长上下文 int32/int64 溢出静默错误；⑩ MXFP8 KV 正确性簇；⑪ 新旗舰模型 CP（上下文并行）成为一等公民；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径；⑬ FP8/低比特 KV cache 跨引擎标准化落地。

**新增第 14 条（投机解码验证路径跨引擎密集硬化）**：本期 vLLM（DSpark 专用 decode #61869、DSpark 在 DCP 复制 draft cache #63796、跨 TP rank 同步自适应 DSpark #62769）、SGLang（Kimi-K3.1 DFlash2 sliding-window MLA draft #49621、DFLASH/DSPARK 在 ReplaySSM spec ring 验证 #46185、spec 验证目标分布 radix top-k #46186、DSpark 置信调度在线接受率估计器 #44588）、TRT-LLM（spec-decode Triton kernel 不再按 batch 特化消除 5–12 s 重编译停顿 #21666、n-gram 候选剪枝/单匹配不变量 #20487、Gumbel-max 水印 + 投机解码 #21047、GLM-5.3-Flash fused MTP-Eagle #21787）共同表明：投机解码已从「能力落地」进入「验证路径正确性 + 服务稳定性」精细化阶段——重点是 DFlash/DFLASH/DSpark 的 verify 路径硬化、置信调度在线估计、Triton kernel 重编译停顿消除。

**新增第 15 条（跨后端可移植性成为显式架构主线）**：SGLang `#49402`（device-neutral `BaseGraphBackend`，为树外 OOT breakable graph 后端铺路）、`#49660`（移除 `--moe-runner-backend triton_kernel`、从 sgl-kernel wheel 去 vendored `triton_kernels`）、`#49661`（MXFP4 MoE 反量化改 JIT kernel）、`#49662`（wheel 不再打包 triton_kernels）；llama.cpp 多后端广度（CUDA OCP FP8 #28413、SM70/head-size 40/72 FA #27997/#30285、Vulkan KV 反量化 #31526、WebGPU #31877、SYCL CLAMP strided #31784、Metal SOLVE_TRI OOB #32103）；TRT-LLM 仍聚焦 NVIDIA。结论：推理引擎正把「CUDA 假设」显式抽象出去，以支撑 Apple MPS（SGLang #47921）、NPU（#40141/#41495）、ROCm 多后端与 OOT 后端。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 量化 attention / MoE 静默错误复核**：vLLM `#64329`（BF16 query 误用 FP8 q_scale）、`#64087`（ROCm KV head 漏写）；SGLang `#48397`（AMD Gluon FP32 partials 丢弃，RULER2 64K 0%→68.7%）——部署 AMD/ROCm 或 FP8 KV 混合 dtype 时必须逐项核验 scale/布局；建议把「dtype × scale 匹配」纳入 CI 端到端断言。
- **P0 — 投机解码服务稳定性**：TRT-LLM `#21666`（DSpark/Helix mid-serving Triton 重编译 5–12 s 停顿），若用 TRT-LLM + DSpark 应立即合入并验证 warmup 覆盖。
- **P1 — KV 卸载/PD 资源回收**：TRT-LLM `#21350`（sparse KV offload 接 KVCM2，性能描述级需自测）、`#21909`（idle decode worker KV 泄漏回收）；vLLM `#66731`（KV 容量预热满预算步）、`#61797`（UMBP lazy offload 默认关、建议评估开启）、`#65943`（Mooncake store 复用 PD TransferEngine opt-in）。
- **P1 — 跨后端可移植性**：SGLang `#49402`/`#49660`/`#49661`（OOT graph/MoE 后端清理）与 llama.cpp `#28413`（CUDA OCP FP8）落地后，跟踪 wheel 体积与 OOT 后端接入进度。
- **P1 — 长上下文数值正确性**：llama.cpp `#29810`（CPU FA V F32 累加，error 0.297→4.5e-5）、`#31284`（Hopper CUDA graph cluster dims 重实例化）；建议把跨后端长上下文 KV 数值对照纳入回归。

---

## 7. 社区与视频资源（HN Algolia，2026-11-11 epoch 窗口 1794355200–1794441600）

> 以下均为社区/Show HN，非引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事。

- **视频**：《Inside Llama.cpp: How Local LLMs Run on Your Hardware》[video]（HN 2 分，2026-11-11T18:22:09Z）——llama.cpp 内部原理讲解视频，**二手线索/未核验**（未核验具体技术细节与版本）。
- **Show HN**：Tensward – Local AI inference optimisation（4 分，2026-11-11T14:11/14:18Z）——本地 AI 推理优化工具，与推理优化相关，**二手线索/未核验**（未核验其性能声称与适用面）。
- **Show HN**：W-ISA – An instruction set for programming LLM weights（2 分，2026-11-11T23:47:21Z）——编程 LLM 权重的指令集概念，研究向，**二手线索/未核验**。
- **文章**：A primer on GPUs and kernel optimization（2026-11-11T03:03:52Z）——GPU 与 kernel 优化入门，与 kernel 优化相关，**二手线索/未核验**。
- **资讯**：Apple has new Mac app coming to make running local AI models easy（多次出现，2026-11-11）——Apple 本地运行 AI 模型新 App，与 llama.cpp 本地推理生态相关，**二手线索/未核验**。
- 其余 HN 命中（vllm 关键词 8 条）多为「United Airlines bans AI agents」「Nvidia oust Jensen Huang」「Gemini Decisions API」等泛 AI 资讯，与推理引擎实现无直接关系，未列入。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM 87 / SGLang 98 / TRT-LLM 49 / llama.cpp 31 合计 265 条（2026-11-11 UTC 窗口）；四家单页均未触 per_page=100 上限，无截断。
- **版本核验**：releases API 确认窗口内无新正式版本；llama.cpp 仅推进 nightly `b12287`–`b12292`。
- **PR 正文来源**：vLLM/SGLang/TRT-LLM/llama.cpp 关键 PR 正文经 issues API（markdown 官方一手）抓取；**vLLM 因 API 配额在抓取中段耗尽（剩余 7、约 55 分钟重置），末批 10 个 vLLM PR（#66731/#64165/#64329/#64087/#61869/#65077/#63796/#61530/#61797/#65943）改由 PR HTML 页面抓取并做 div 深度配对提取**，经与同源 issues API 已抓样本交叉确认正文一致，不消耗 API 配额；其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度。
- **描述级/未全绿标注**：llama.cpp `#28413`（OCP FP8 perf 数字未完整）、`#27997`（PP/TG 数值未完整）、`#31526`（perf 数值未完整）记为 perf 描述级；SGLang `#46185`/`#46186` 注明 benchmark 因 rebase 未重跑；TRT-LLM `#21350` 性能为对比框架描述级；TRT-LLM 大量 `nvbugs` 内部链接 PR 仅能凭标题/描述记录，无法核验内部 QA 状态（维持「描述级/未核实实现落地」口径）。
- **社区信号**：HN Algolia 引擎官方更新类故事无；社区侧视频/Show HN 按二手线索/未核验记录。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级」。
