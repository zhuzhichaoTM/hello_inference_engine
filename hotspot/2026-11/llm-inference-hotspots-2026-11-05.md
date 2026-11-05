# 大模型推理优化每日热点档案

> 档案日期：2026-11-05（本自动化于该日执行）
> 关注窗口：2026-11-04（周三）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（issues API，markdown）+ HN Algolia（HTTPS，numericFilters 按 2026-11-04 epoch 窗口）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 11 / SGLang 13 / TRT-LLM 11 / llama.cpp 0 个 PR 正文已抓取（含 #65542/#54625/#60181/#63752/#64760/#61463、#48415/#48292/#48001/#48003、#21021/#21242/#21281，及 llama.cpp v0.9.0 发布说明全文）；releases API 确认 **llama.cpp v0.9.0 于 2026-11-04T13:08:06Z 正式发布**（窗口内唯一新正式版本），其余三家无新版本；HN 窗口内引擎相关无高信号发布故事（vllm 21 命中多为「LLM 网关/防火墙/工具」类，非引擎更新）。

## 1. 版本状态与本期头号事件：llama.cpp v0.9.0 正式发布 + vLLM Marlin→Humming 量化后端整合 + KV 正确性主线延续

**提交计数（2026-11-04 UTC，均 HTTP 200；2026-11-04 为周三工作日，vLLM 111 较上一窗口 11-03 的 124 略降，SGLang 62 持平，TRT-LLM 36 较 53 降，llama.cpp 22 较 36 降）**：vLLM **111** / SGLang **62** / TensorRT-LLM **36** / llama.cpp **22**。

**发布状态（窗口内唯一新正式版本：llama.cpp v0.9.0）**：
- **llama.cpp**：**v0.9.0 正式发布（2026-11-04T13:08:06Z）**——OCP FP8 E4M3 量化、新 `flash_attn_ext_rows` op（统一 KV 在多序列下高效）、ggml 升 v0.29.0（FP8 E4M3 / flash attention rows / NCCL 多节点 TP / 全后端性能）；新增 Clover / Limite / Metis-1.6 / Mistral Large 4 四架构；Chatterbox TTS / Nemotron 3.5 Transcribe / Pocket TTS；gemma4 大幅 prefill 优化。窗口内 nightly 标签延续 `b12114–b12119`（prerelease，21:33Z–22:34Z）。
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-04 全部 111 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-04 含 Kimi-K3.1 day-0 支持（#48415）等。
- **TensorRT-LLM**：仍为 **v1.4.0rc2（2026-10-28T15:51:34Z，prerelease）**，窗口内无更新。

**本期头号事件（llama.cpp 发版 + vLLM 量化后端整合 + 跨引擎 KV 正确性延续 + 新调度/KV 准入策略）**：
1. **llama.cpp v0.9.0 正式发布（#31646 bump，2026-11-04）**：年度级版本，亮点为 **OCP FP8 E4M3 张量类型**（#27512，saturating 模式，新增 `LLAMA_FTYPE_MOSTLY_F8_E4M3`）、**`flash_attn_ext_rows`**（#29510，使统一 KV 在多序列下恢复非统一 KV 的 prefill 性能）、**ggml batched tensor copy API**（#30881）。ggml 同步升 **v0.29.0**（含 NCCL 多节点张量并行、CPU/SYCL/Metal/Vulkan/OpenCL/Hexagon 全后端性能工作、FP8 E4M3、flash attention rows）。新增四架构 + 三类音频生成/转写能力 + Gemma4 E2B/E4B 大幅 prefill 优化（#31356，M4 Pro 上 E2B pp512 ≈2.85×）。连续第三年入选 GitHub Octoverse。
2. **vLLM 量化后端整合：Marlin 退守 16-bit 激活，8-bit 激活整体移交 Humming（#65542 / #65540，强化累计结论第 3 条）**：Marlin dense/MoE kernel 移除 INT8/FP8 激活路径（W4A8-INT8 与 W4A8-FP8），成为纯 16-bit 激活 kernel；`VLLM_MARLIN_INPUT_DTYPE` 环境变量移除并改报错，迁移路径为 `--quantization humming` + `VLLM_HUMMING_INPUT_QUANT_CONFIG`；SM89/SM12x 的 W4A8-FP8 compressed-tensors checkpoint 此前回退 Marlin，现改选 `HummingLinearKernel`；SM12x 专用 cubins 移除、改依赖 compute_80 PTX（#65540）。净删 2039 行。属「能力/默认迁移」风险载体——W4A8 用户须主动切 Humming，否则报错。
3. **KV cache 正确性延续为跨引擎最密集主线（强化累计结论第 4/6/8 条）**：vLLM `#60181`（NIXL `kv_buffer_device="cpu"` 时 h2d sync 用 NIXL 从未填充的 host 页覆盖已接收 KV → 多 KV 组/KV-sharing 模型静默错，GB10 Gemma-4-E2B main 0/4 与 device buffer 一致、PR 4/4 逐位相等）、`#59747`（拒绝不同 P/D block size 的 push 握手）、`#59383`/`#63696`（KV Offload 惰性块的回收与完整 store）、`#65331`（Mooncake store 登记 extensible KV 已提交块）、`#64677`（UMBP 逐 KV 布局拷贝每 head runs）；SGLang `#48292`（disaggregation 在 DSv4 PD 与 HiSparse decode 两配置下 KV checksum 无法工作、改启动期拒绝）、`#48001`（unified-memory 下 routed DFLASH/DSPARK draft 的 context-KV 误写 FULL id 空间的错行，silent）、`#48002`（拒绝 sharded-padded CP KV + unified memory）、`#39799`（非 DSA EAGLE draft KV pool 按 DCP 加宽尺寸预算）、`#43911`（HiSparse checksum 失配/chunked abort 释放行）；TRT-LLM `#21021`（mixed-precision KV 延伸到投机解码/sparse MLA）、`#21547`（恢复 DSA FMHA 的辅助 KV pool 选择）、`#21281`（按请求历史定 draft-KV mirror 尺寸）。
4. **vLLM 新增 cache-aware admission ordering（#54625，KV 准入策略，强化第 8 条）**：在 waiting queue 有界前瞻窗口内，把 prefix 已驻留 KV 的请求排在冷请求前，避免严格 FCFS 下冷请求分配挤掉即将被复用的 prefix 块；新增 `--cache-aware-admission-window N`（默认 0）与 `--cache-aware-admission-threshold F`（默认 0.0），仅 `fcfs` 策略 + prefix caching 下生效、优先级堆策略下禁用——属 KV 容量/复用智能的架构级新增。
5. **投机解码正确性 + 性能双线（强化第 5 条）**：vLLM `#62234`（对齐 GLM-4.5/4.6 MoE 与 GLM-OCR MTP 的 speculator hidden norm 以复用 post-final-hidden）、`#64059`（修复 DFlash draft acceptance=0 在 FULL CUDA graphs 下）、`#65100`（resample kernel 跳过无 target token 的 vocab 块）、`#65488`（DFlash draft 输入 padding 到同 launch 余 program）；SGLang `#48001`（见上，draft KV 错行）、`#48111`（默认 `SGLANG_SPEC_TP_SYNC=init`）、`#48298`（每步只抽一次 `SGLANG_SIMULATE_ACC_LEN`）；TRT `#21021`（mixed-precision KV 对 MTP/Eagle/DFlash drafter 的 dtype 规则）。

---

## 2. vLLM：2026-11-04 提交（111 条）——Marlin→Humming 整合 + cache-aware 准入 + NIXL/KV Offload 静默正确性

### 2.1 `#65542` Marlin：移除 8-bit（INT8/FP8）激活支持，改由 Humming 承担（量化后端迁移，强化第 3 条）
- **主题**：Marlin dense/MoE kernel 移除 W4A8-INT8（GPTQ/AWQ INT4）与 W4A8-FP8（GPTQ/AWQ INT4、MXFP4）路径，Marlin 退化为纯 16-bit 激活 kernel；Humming 覆盖 8-bit 激活。独立于此前的 #65540（Marlin 仅 PTX 构建）。
- **内容详情**：删除 sm89 FP8 激活 kernel 及 SM89/SM12x 构建目标、INT8 激活实例化、两模板的 8-bit 路径、`marlin_int4_fp8_preprocess` op、`marlin_gemm`/`moe_wna16_marlin_gemm` 的 `a_scales` 与 repack ops 的 `is_a_8bit`；净删 2039 行。`VLLM_MARLIN_INPUT_DTYPE` 移除并改报错，替换路径 `--quantization humming` + `VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8"}'`（或 `float8e4m3`）。SM89/SM12x 的 compressed-tensors W4A8-FP8 checkpoint 此前回退 Marlin，现选 `HummingLinearKernel`。FP8-weight Marlin（W8A16、MXFP4）与任何 16-bit 激活格式不变。
- **应用场景**：W4A8（INT8/FP8 激活）量化的 kernel 后端选型。
- **研判**：【事实】PR 给删除清单、迁移命令、DGX Spark(GB10,SM121) kernel 测试（test_marlin_gemm 254 passed、test_moe -k marlin 189 passed/6 skipped）与 GSM8K Qwen2.5-3B GPTQ-Int4 对照（default Marlin W4A16 0.62/3348 tok/s；W4A8-INT8 主分支 `VLLM_MARLIN_INPUT_DTYPE=int8` vs PR `--quantization humming`，精度/吞吐对位）。【观点】属「量化后端/默认迁移」（第 3 条）——Marlin 收窄为 16-bit、8-bit 激活整体移交 Humming，是「能力移除即风险载体」的典型；W4A8 用户须主动切 Humming，否则启动报错。证据等级：官方一手（含测试与 GSM8K 对照）。

### 2.2 `#54625` 新增 cache-aware admission ordering（KV 准入策略，强化第 8 条）
- **主题**：在 waiting queue 有界前瞻窗口内，把 prefix 已驻留 KV 的请求排在冷请求前 admitting，避免严格 FCFS 下冷请求分配挤掉队列后方请求仍需要的 prefix 块、把本可命中的缓存变重算（共享 prefix 越深损失越大）。
- **内容详情**：两个默认关的选项——`--cache-aware-admission-window N`（默认 0，前瞻深度；0=严格到达序）、`--cache-aware-admission-threshold F`（默认 0.0，KV 使用率低于此不启用窗口）。范围限定：仅在窗口内、且为持有 `WAITING` 且未算任何东西的普通请求间重排；`WAITING_FOR_REMOTE_KVS` 等阻塞态或已部分算过的请求保持原位；要求 `fcfs` 策略（优先级堆下 `prepend_request` 按优先级重插、忽略位置，且会与用户显式优先级冲突，故禁用）；要求 prefix caching、跳过重排 prefill 的步；阈值让运维把重排限制在确有驱逐可避免的时段。
- **应用场景**：高 prefix 复用、KV 压力下 FCFS 导致「即将复用的块被挤掉」的调度优化。
- **研判**：【事实】PR 给出两选项语义、作用边界（仅 fcfs + prefix caching、窗口内 peer 重排、阻塞态不动）。【观点】属「KV 容量/复用智能」（第 8 条）——把 prefix 驻留信息引入准入决策，是 KV 调度从「FCFS 公平」到「缓存感知」的架构级新增；默认关、需 prefix caching，采用前自测命中率与尾延迟。【证据等级】官方一手（设计说明级；未见端到端命中率数字，记为设计级一手）。

### 2.3 `#60181` [Bugfix][NIXL] 修复 host-buffer h2d sync 覆盖已接收 KV（KV 静默正确性，强化第 4/6/8 条）
- **主题**：`kv_buffer_device="cpu"` 时，`sync_recved_kv_to_device` 把 D 端已接收的 KV 用 NIXL 从未填充的 host 页覆盖，对含多个 KV cache group 或 KV-sharing 层的模型；多组情形在 #53780 回归。
- **内容详情**：每个 transfer group 现在只拷贝自身层的 host buffer；多组情形回归修复。`pytest test_nixl_connector_hma.py -k test_sync_recved_kv_copies_each_group_own_layers`：#54483 的参数化测试新增 KV-sharing 层，main 下 3 行全失败、PR 全通过。本地字节级检查（TP1/1、TP2/2 toy 模型）：main 在 16 个混合 sliding-window/full-attention 多组用例中 D 与 P 全不一致，4 个纯 full-attention 一致；PR 全部 20 个一致。GB10 Gemma-4-E2B-it（含 sliding-window+full-attention 组、20 个 KV-sharing 层，P→D TP1），`kv_buffer_device="cpu"` 两端：main 4 prompt 输出全异于 device buffer（max |Δlogprob| 0.81–3.74），PR 4/4 逐位相等。
- **应用场景**：NIXL KV 传输 + CPU host buffer 的 P/D 正确性。
- **研判**：【事实】PR 给 GB10 字节级对照（main 0/4 → PR 4/4 逐位相等）+ 本地 20 用例 + 单测。【观点】属「KV 传输静默正确性」（第 4/6/8 条）——多组/KV-sharing 下 host buffer 越界覆盖致 D 端 KV 静默错、无报错，是 P/D 正确性的隐蔽债；与往期 KV Offload/HiCache 静默损坏同源。证据等级：官方一手（含逐位相等验证）。

### 2.4 `#63752` [Bugfix][Model Loader] 拒绝只收到部分 shard 的 stacked 权重（模型加载静默正确性，强化第 6 条）
- **主题**：`track_weights_loading` 只 diff 参数名，故 `qkv_proj.weight` 任一 shard 写入即记为已加载；缺失全部 `v_proj` 的 checkpoint 能干净加载并送出**未初始化的行**。#55911 的 follow-up。
- **内容详情**：stacked 参数（MergedColumnParallelLinear / QKVParallelLinear 及其子类）只收到部分 shard 现于加载期失败，报参数名与缺失 shard id。新增基类 `StackedColumnParallelLinear`：每个 `weight_loader` 在 param 上记录所写 shard，`pop_missing_shards()` 报告未到位的预期 shard 后清记录。Qwen3-0.6B H100：删 safetensors 张量后 `LLM` 加载——`untouched` 两端 OK；`all v_proj removed` main 加载成功但生成 `!!!!!!!!`、PR 报 `ValueError: Following weights were not initialized... qkv_proj.weight.v`；`layers.3.mlp.up_proj removed` main 错误文本、PR 报 `gate_up_proj.weight.1` 缺失。有效 checkpoint 字节不变。
- **应用场景**：stacked 权重（QKV / gate_up）的分片完整性校验。
- **研判**：【事实】PR 给根因（按名计数而非按 shard）+ Qwen3-0.6B 三档对照（缺 v_proj/缺 up_proj 均从「静默生成乱码」变 `ValueError`）。【观点】属「模型加载静默正确性」（第 6 条）——部分 shard 缺失不报错、送未初始化权重，是最隐蔽的加载债之一；fail-fast 优于静默乱码。证据等级：官方一手（含对照表）。

### 2.5 `#64760` [GLM 5.3 Perf] H200 上 FP8 MoE down projection 的 Sorted-A TMA（6.0%~7.6% E2E 吞吐）
- **主题**：保持 MoE intermediate 张量按 expert 排序，使 down projection 对激活与权重都能用 TMA；把 SiLU、乘法、FP8 量化融进一个 Triton kernel。
- **内容详情**：`vllm bench serve` 同配置对照（GLM-5.3, tp8, fastsafetensors, MTP num_speculative_tokens=5）；aime25 exact_match 0.8333（±0.0692）。perf 区块给出 main vs PR 的 Serving Benchmark（吞吐数字 PR 正文在表格尾部截断，仅确认 ↑6.0%~7.6% E2E 声明；单测已有覆盖）。
- **应用场景**：GLM-5.3-Flash FP8 MoE 在 H200 的 down projection 吞吐。
- **研判**：【事实】PR 声明 Sorted-A TMA + 三算子融合，aime25 0.8333，声明 E2E +6.0%~7.6%（perf 表尾部截断，记为官方一手、端到端数字未完整呈现）。【观点】属「GLM/FP8 MoE 性能」（第 4 条端侧/异构同源的服务器端镜像）——排序使 TMA 可作用于权重是 MoE down proj 的常规加速手法。证据等级：官方一手（含 aime25；E2E 百分比声明级）。

### 2.6 `#61463` [Bugfix][Kernel] ll_bf16 split-K router GEMM 在 DSMEM store 前 sync cluster（kernel 崩溃正确性）
- **主题**：`ll_bf16` split-K router GEMM（#42562，M≥8）在任意 cluster barrier 前把 partial 写入 peer CTA 的 shared memory（`st.shared::cluster`）。若 peer CTA 尚未启动，GPU 抛 Xid 13 `CTA Not Present`，表现为 `CUDA error: unspecified launch failure`。
- **内容详情**：修复——所有线程在 pipeline init 后 `cluster_arrive_relaxed()`；DMA warp 在提交 stage 0 后 `cluster_wait()`，MMA warp 在分支开头 wait。B300 单 GPU harness（router GEMM 跑 CUDA graph，并发非 cluster copy kernel 写 mapped pinned host memory）：`main` 首 batch 即 `unspecified launch failure`（4/4 跑），PR 120 s 干净、4,912,896 次 router call 输出正确。serving 中同 Xid 曾在 GLM-5.3 HiSparse P/D decode（2×8×H200, TP4）数分钟内触发。无回归测试（依赖硬件 CTA 调度，不可确定化）。`test_ll_bf16_gemm.py` B300/H200 各 144 passed。
- **应用场景**：ll_bf16 split-K router GEMM（MoE 路由）在 cluster 协作下的启动期正确性。
- **研判**：【事实】PR 给根因（peer CTA 未驻留即 DSMEM store → Xid 13）+ B300 4/4 复现/修复对照 + 144 passed。【观点】属「kernel 协作/启动正确性」——cluster 间 DSMEM store 未等 peer 驻留致硬崩；属低概率但致命的并发 bug，与往期「静默/崩溃」主线同源（崩溃型）。证据等级：官方一手（含硬件复现与修复）。

### 2.7 `#59747` / `#59383` / `#63696` / `#65331` KV 传输与卸载正确性簇
- `#59747` **[Bugfix][NIXL] 拒绝不同 P/D block size 间的 push 握手**——防止块尺寸不匹配的 KV 错误接收。
- `#59383` **[Bugfix][KV Offload] 把惰性卸载的 GPU 块归还到驱逐端**——惰性卸载块的生命周期归位，避免与正常驱逐抢占。
- `#63696` **[KV Offload] 从请求上下文描述完整 store**——store 记录完整（含重试/部分），与 #59383 同簇。
- `#65331` **[Bugfix][KV Connector] Mooncake store：登记 extensible KV cache 已提交块**——extensible KV 下 Mooncake 仅登记了部分块，补登已提交块使远端可见。
- **研判**：【观点】四者共同构成「P/D KV 传输 / 卸载 / extensible KV 登记」的正确性收口（第 5/8 条），与 #60181 同源；#65331 为 extensible KV 在 UMA/GB10 容量大巡检（11-04 档案头号事件之一 #65413）后的连接器侧补强。证据等级：官方一手（标题/正文级为主）。

### 2.8 `#64677` / `#63752`（见 2.4）/ `#64782` UMBP / UMA KV 正确性
- `#64677` **[Bugfix][UMBP] 拷贝每个 KV cache 布局的 per-head runs**——UMBP（统一内存带外）下 KV 块元数据逐布局完整拷贝，防跨布局错行。
- `#64782` **[Bugfix][CUDA] 在 WSL 上对集成 GPU用 CUDA 显存记账**——与 11-04 档案 #65413（UMA 用 `psutil` 校正 `cudaMemGetInfo`）同源，把 UMA/集成 GPU 的显存记账补到 WSL 路径。
- **研判**：【观点】UMBP/WSL 两条均属「异构/集成设备 KV 与显存记账」（第 4/8 条）的边界补强。证据等级：官方一手（标题/正文级）。

### 2.9 GLM-5.3 性能簇 + 投机解码簇（官方一手，标题/正文级）
- GLM-5.3 性能：`#65198`（对齐 draft model kv cache dtype 与 target，每 draft 省 50% GPU 显存）、`#65553`（glm 5.3 flash draft dspark 启用 fp8 kv cache，每 rank 省 960MB）、`#64760`（见 2.5）。
- 投机解码：`#62234`（对齐 GLM-4.5/4.6 MoE 与 GLM-OCR MTP speculator hidden norm 以复用 post-final-hidden）、`#64059`（修 DFlash draft acceptance=0 在 FULL CUDA graphs）、`#65100`（resample kernel 跳过无 target token 的 vocab 块）、`#65488`（DFlash draft 输入 padding 到同 launch 余 program）、`#65528`（清理 AR speculator decode 路径）、`#65144`/`#65372`–`#65377`/`#64471`/`#64492`/`#64697`/`#65001`/`#64946`/`#65284`/`#64308`/`#59736`/`#64551`（Rust Frontend 与 Python frontend 的 API 对齐系列，Bugen Zhao 主导，多为空内容/工具调用/错误源/flag 形式兼容）。
- MoE / 模型：`#65498`（flashinfer_nvlink_one_sided 跳过 combine staging copy）、`#63921`（DSv4.1 latency_specialized decode 延伸 NVIDIA）、`#65619`（Qwen4Exp 跨层共享 QSA top-k buffer）、`#62172`（cooperative_topk 读 BF16 indexer logits）、`#65504`（MegaMoE 支持 PP）、`#65537`（MiniMax-M3 序列并行 + PP）、`#64077`（Muse Spark Open 专用解码 GB300 优化 1.5×）、`#59844`（DeepEP 升级 2.1、保留 per-context GIN 完成序）、`#65302`（ZAYA1 支持）、`#64016`（Unlimited-OCR-2 DASH attention）、`#62057`（ROCm Beam 1.0）、`#64583`（Kimi-K3 睡眠模式回退 latent-MoE tail fusion）、`#62729`（DeepSeek V4.1 tokenizer 尊重 add_generation_prompt）、`#65399`（调度器尊重多模态禁用 chunked prefill）、`#65466`（等待模型输出时处理新请求/中止）、`#63406`（prefill/decode CUDA graph 捕获尺寸分离）、`#57042`（2-stage custom all-reduce 按绝对序累加 rank）、`#55835`（跨节点 MNNVL 组不进 CUDA IPC all-reduce）、`#62265`（b12x 自选默认 KV block size）、`#63012`（b12x 不再广播 head_dim 64）、`#60919`/`#65087`/`#59802`/`#63367`/`#64836`（ROCm GLM-5.3/MiniMax-M3.1/DSv4.1/Step5 性能与预热）。

---

## 3. SGLang：2026-11-04 提交（62 条）——Kimi-K3.1 day-0 + 解聚 KV checksum 拒绝 + unified-memory draft KV 错行修复 + HiCache KV 存储重构

### 3.1 `#48415` [Feat] model：新增 Kimi-K3.1 基础支持（NV, AMD）（新模型 day-0）
- **主题**：新增 `KimiK31ForConditionalGeneration`——Kimi-K3 在 full-attention 层带 DeepSeek 式稀疏注意力（DSA）、`fp8_ds_mla` KV cache（其 tile scales 按参考 writer 上取整到 2 的幂）、CUDA 上 `flashmla_kv` 或 ROCm 上 Triton sparse MLA 后端。
- **内容详情**：其他模型行为不变；exact packed top-k kernel 现也在 CUDA 构建（仅本模型选它）。性能与后续特性（DCP、PD、投机解码）在 #48414 跟进。模型：`moonshotai/Kimi-K3.1-Pro`。**CI 状态：Base/Extra/AMD ROCm 10 三套均 :x: 红**。
- **应用场景**：Kimi-K3.1 推理（NV/AMD）。
- **研判**：【事实】PR 给模型结构（DSA + fp8_ds_mla + tile scale 上取整）+ 模型卡链接；CI 三套全红。【观点】属「新模型 day-0 支持」——SGLang 在 v0.5.23 发布后迅速补 Kimi-K3.1，与往期 vLLM ZAYA1/TRT Kimi 系列同源；CI 全红提示 day-0 尚不稳定，采用前以 CI 绿 + 实测为准。证据等级：官方一手（CI 全红，记 day-0/未全绿）。

### 3.2 `#48292` [Fix] disaggregation：在 DeepSeek-V4 与 HiSparse decode 上拒绝 KV checksum（PD KV 正确性，安全/正确性）
- **主题**：`--disaggregation-enable-kv-checksum` 在两配置无法工作，现改启动期拒绝。
- **内容详情**：① DSv4 在 PD 下 KV pool 注册了 state 条目（SWA、SWA ring、request state）但 `state_indices_for_request` 返回空 → prefill 在 `KvChecksumComputer.compute` 断言、decode 在校验时也会；② HiSparse decode 把 KV 收进 host pool，但 checksum 按请求 logical id 索引该 pool，读到错行——layout signature 不同时每请求带警告放行、相同时会 digest 错内存。plain MHA + `--kv-cache-dtype mxfp8` 也触发同断言（其 block-scale state 也无 indices），但不在此拒绝（dtype-only 检查会误伤 checksum 可用的 hybrid-linear 模型）。
- **应用场景**：P/D disaggregation 的 KV checksum 启用边界。
- **研判**：【事实】PR 给两配置的根因（DSv4 state indices 空 / HiSparse 按 logical id 读错行）+ 拒绝点。【观点】属「PD KV 正确性/安全」（第 5/8 条）——延续 2026-10-27 社区披露的跨请求 KV 泄露/暴露主题，把「checksum 在部分配置静默失效」显式化为启动期拒绝，避免带病运行。CI 红。证据等级：官方一手（含两配置根因；CI 红）。

### 3.3 `#48001` [Fix] spec：把 routed DFLASH/DSPARK context KV 写进 draft 自身行（投机解码 KV 静默正确性，强化第 5/6 条）
- **主题**：`--enable-unified-memory` 下 DFLASH/DSPARK draft 可融进 target 的统一 KV pool，其层被路由：full-attention draft 层写 pool 的 FULL id 空间、sliding-window draft 层写其 window id 空间。draft 的 context-KV 写（target hidden states 物化进 draft KV）仍经 pool 级 `set_kv_buffer_prefix_valid` / 单 location 向量，致 routed window 层收到 FULL id 并写错行。
- **内容详情**：`mem_cache/kv/router.py::KVLayer.store_prefix` 现经层自身 route 与物理 id 提交每请求 live prefix（同 `store` 对普通写）；`dflash_worker_v2.py` 的 context append（sequential/fused/prefix-valid）取 window cache locations 与 FULL 并列、经层 route 写、FULL/window 形状不匹配则拒绝；`dspark.py`/`dspark_kv_inject.py`/`dspark_verify.py` 传双 id 空间、仅当所有 draft 层属 FULL 才用 fused single-location kernel。26-PR 栈（#48003–#48026）之一，无 unified memory / 无 fused routed draft 时无变化。
- **应用场景**：unified-memory 下 DFLASH/DSPARK 投机解码的 draft context-KV 写入正确性。
- **研判**：【事实】PR 给根因（routed window 层收 FULL id 写错行）+ 修改点 + 测试（layer store 保留 FULL/window 双空间、DFLASH 普通/前缀写按层 route）。【观点】属「投机解码 KV 路由静默错误」（第 5/6 条）——unified-memory 下 draft/target KV 共享池的 id 空间错配致静默错行，与 vLLM 路由/分页静默错误同源。证据等级：官方一手（含测试；CI 状态未全绿，记正文级）。

### 3.4 `#48003` [Refactor] hicache：移动共享 host backing，从 layout view 构建统一 host pools（HiCache 架构重构）
- **主题**：unified pool 的 host（HiCache）侧原先在 `mem_cache/pool_host/unified.py` 内用 pool 专属知识构建 host pools；本栈后续 PR 让 unified pool 承载任意数量 window 子池与 fused draft 部分，host 侧须镜像而不按 pool 形状逐一 case。本 PR 把共享 host backing 移到设备内存代码旁，并从每个设备 layout view 自身描述构建 host pools。
- **内容详情**：`mem_cache/kv/memory/host_backing.py`（新）移入 `_SharedPageEnvelopeHostBacking`（行为不变）；`mem_cache/kv/layout/{base,mha,mla,state}.py` 各 layout view 用 `host_layout()` 描述 host 行并从 `HostPoolOptions` 构建 host pool；组装器/统一 pool/storage 改从 layout view 描述组装。测试 `test_host_layout.py`/`test_hybrid_pool_assembler.py`/`test_unified_kv_storage.py` 更新。无预期行为变化。26-PR 栈累计 serving 验证（非本 PR head）。
- **应用场景**：HiCache unified pool 的 host 侧构建与多子池/融合 draft 扩展。
- **研判**：【事实】PR 给重构点（host backing 机械移动 + layout view 自描述 host 行）。【观点】属「HiCache 架构重构」（第 5/8 条）——为 unified pool 支持多 window 子池/fused draft 铺路，是 KV 存储层的可扩展性重构；行为不变、风险低。证据等级：官方一手（含测试；stack 累计验证）。

### 3.5 其余 SGLang 要点（官方一手，标题/正文级）
- 解聚/PD：`#47255`（bind NIXL transfer workers 到 rank 设备）、`#48002`（拒绝 sharded-padded CP KV + unified memory）、`#39799`（非 DSA EAGLE draft KV pool 按 DCP 加宽尺寸预算）、`#43911`（HiSparse checksum 失配/chunked abort 释放行）、`#48283`/`#48327`（RequestRetirer 统一队列 abort/step 内完成）。
- 性能（DSv4/DSv4.1/Kimi/DSpark/MoE）：`#43786`（DSv4 inverse RoPE 融进 wo_a FP8 量化）、`#47935`（dsv41 MXFP4 Engram 表 packed + gather 内 E2M1 decode）、`#45828`（fp8 Hopper small-M GEMV + fused KDA output quant + CUDA grouped top-k + partial-block tails）、`#48243`（dsv4 V4.1 paged dequant 每 program 多 KV token）、`#47526`（Kimi-K3 fused KDA decode 6/3 heads per rank）、`#47552`/`#47553`（R3 routed-expert host cache 仅留 output rank / 无行 rank 跳 capture）、`#47643`（mega_moe EPLB load 计 in-kernel）、`#48246`（deep_select 3-CTA tier 用于 non-paged bf16 top-512）、`#48266`（deep_select 列 SM107 支持）、`#48109`（moe padded-row indices 匹配 token-count dtype，MoE 正确性）、`#47944`（FLA Triton kernel 迁离 `tl.make_block_ptr`）、`#47822`（AMD aiter unified extend 跳过 q.contiguous）、`#46443`（diffusion rollout conditioning 修复）、`#48244`（dspark DSpark draft forward 展开 [N,H] staged embeds）。
- 新后端/模型：`#47853`（HALOps 作为 Blackwell opt-in MXFP8 GEMM 后端 + mxfp8 megamoe）、`#45885`（Nemotron-3-Embed embedding）、`#38440`（NPU DSpark DeepSeek-V4 compact mode）、`#41504`/`#48352`/`#46519`（NPU/XPU 依赖与引用更新）。
- API/工具：`#48343`/`#48357`（openai tool schema ids / Kimi-K3 prompt fidelity）、`#48298`（spec 每步抽一次 `SGLANG_SIMULATE_ACC_LEN`）、`#48111`（默认 `SGLANG_SPEC_TP_SYNC=init`）、`#47120`/`#47235`（VLM cache 强身份/共享 typed 身份）、`#48301`（kv_canary 去 min_p）。文档：`#48413`/`#48417`/`#48429`/`#48435`/`#47861`/`#48432`（Kimi-K3.1-Pro 部署/模型卡/AMD HiCache 命令）。

---

## 4. TensorRT-LLM：2026-11-04 提交（36 条）——mixed-precision KV 延伸到投机解码 + NVFP4_PLUS + Kimi/ MiniMax Rubin 内核

### 4.1 `#21021` [feat] 把 mixed-precision KV cache 延伸到投机解码与 sparse MLA 后端（KV，强化第 8 条）
- **主题**：#20452 给标准 `KVCacheManagerV2` 加了 First-N/Last-N mixed-precision KV（每 full-attn 层历史存 NVFP4，首 `mixed_precision_first_n`/末 `mixed_precision_last_n` token 留注意力 dtype 于 retained-twin 层，注意力跑在 staging copy），但拒投机解码、hybrid Mamba manager 与所有 sparse-attention 后端；本 PR 解除这三限制。
- **内容详情**：**投机解码（MTP/NGram/Eagle 单模型）**——draft KV dtype 随 drafter 类型：target manager 内 MTP 层（如 attention DP）随 target 走 mixed（dsa/deepseek_v4 即 NVFP4 pool + fp8 first_n/last_n twin；DSv4 MTP 仅 sliding-window 故无 twin）；独立 draft manager 的 MTP 自建 twin；replacement MTP heads 随 target；自带架构的 drafter（Eagle3/DFlash/外部）保持 plain cache（draft kv_cache_config 丢 mixed dtype，服从 retained dtype：fp8 或未量化）；`draft_kv_cache_quant_algo_override`（如 MiniMax-M3）仍优先。**Hybrid Mamba（Qwen3.5/Kimi-K3）** 经 `_supports_mixed_precision_kv` 选入，recurrent-state/attention 层后追加 retained-token 层组。
- **应用场景**：mixed-precision KV（首/末 N 高精度）在投机解码 / sparse MLA / hybrid Mamba 下的前缀复用。
- **研判**：【事实】PR 给三类 drafter 的 dtype 规则与 hybrid Mamba 选入机制（延伸 #20452）。【观点】属「mixed-precision KV 复用」（第 8 条）——把 11-04 档案 #21432（SINK_REQUIRED_ONLY 混合精度前缀复用）的能力面从标准注意力扩到投机/sparse/hybrid，是长上下文 agentic 复用收益的通用化；opt-in。证据等级：官方一手（设计说明级）。

### 4.2 `#21242` [feat] 新增 NVFP4_PLUS（E2M1 + UE5M3 per 32）CuteDSL MoE on Rubin（新量化格式，SM107）
- **主题**：NVFP4_PLUS 保留 NVFP4 的 E2M1 值，但每 32 值存一个 UE5M3 block scale、无 FP32 per-tensor scale。SM107 block-scaled MMA（kind::mxf4nvf4）直接读 UE5M3 scale，UE5M3 跨度 2^-17~114688，激活块无需全局归一化。
- **内容详情**：MXFP4 checkpoint 原样加载（E2M1 payload 不变、每 E8M0 scale 变同幂次的 UE5M3 码，指数 -17..16 精确）。选 `OVERRIDE_QUANT_ALGO=NVFP4_PLUS` + `moe_config.backend=CUTEDSL`。含 QuantAlgo/QuantMode NVFP4_PLUS、UE5M3 参考与 CuTe DSL f32<->UE5M3 转换（round-up 前 saturate）、CuTe DSL 激活量化（amax/6 round-up 到 UE5M3，128x4 swizzled/row-major）、Rubin FC1 gather+activation kernel、新 grouped-GEMM ops、新 MoE impl（SM107, SwiGLU, EPLB）。NVFP4 EPLB 按 staged experts 定 host-staged alphas、默认 1（无 weight_scale_2 的 checkpoint 也能跑）。限制 MoE 路径到支持形状与 bias-free SwiGLU。
- **应用场景**：Rubin(SM107) CuteDSL MoE 的新低比特格式（UE5M3 block scale）。
- **研判**：【事实】PR 给格式定义（E2M1 + UE5M3/32，无 FP32 scale）、加载兼容性（MXFP4 checkpoint 原样）、选择方式与限制。【观点】属「量化格式创新」（第 4/6 条）——UE5M3 block scale 免全局归一化是 NVFP4 的 Rubin 特化，扩展低比特 MoE 工具箱；限形状/SwiGLU，采用前核对支持矩阵。Dev review 提示 checkpoint 含非常规字段时被拒。证据等级：官方一手（格式/加载级）。

### 4.3 `#21281` [fix] 按请求历史定 draft-KV mirror 尺寸（draft KV 正确性，描述级/未全绿）
- **主题**：cache-manager 按 target position 与 chunk history 定外部 drafter mirror 尺寸，并在解聚生成初始化与后续更新中保留该历史；DFlash 跟踪每 slot base pages、rebase 受管 paged-KV 读写；windowed draft pool 保留 trailing page。
- **内容详情**：`dflash_paged_latent_prepare` 现接 `base_pages` 返 `rebase_pages`；`decode_draft_block_table` 经最后有效项报每行 extent；K3 acceptance 在两 worker 配置均拒 sliding-window draft pool。**Dev/QA 均指出的硬件检查经 524k-token prompt，但无独立可验证测试结果；review 结论与完成状态未提供；覆盖 verdict：needs follow-up，测试执行结果未给出。**
- **应用场景**：外部 drafter（DFlash）的 draft-KV mirror 尺寸与解聚历史传播。
- **研判**：【事实】PR 为 draft-KV mirror sizing 改动 + API/block-table/eligibility 变更。**无独立可验证测试结果、QA 判 needs follow-up，记描述级/未全绿**——采用前以最终 CI 与实测为准（与往期 TRT `#20486`/`#21239`/`#20524` 同类）。证据等级：官方一手（描述级，Dev/QA needs follow-up）。

### 4.4 `#21547` / `#20623` / `#21399` KV 与内核性能
- `#21547` **[fix] 恢复 DSA FMHA 的辅助 KV pool 选择**——DSA FMHA 的 auxiliary KV pool 选择回归修复（KV 正确性）。
- `#20623` **[feat] MiniMax-M3 NVFP4 低延迟 decode megakernel for Rubin (sm_107) TP2/4/8 in modeling_v2**——MiniMax-M3 在 Rubin 的 NVFP4 低延迟 decode 大内核（modeling_v2）。
- `#21399` **[perf] Kimi-K3 decode attention-residual kernel：CuTe DSL PDL 与 MoE residual 融合，MXFP8 交棒 KDA，gated/sigmoid low-M GEMM 变体**——Kimi-K3 decode 注意力-残差 kernel 的 PDL 混合与 MoE residual 折叠、MXFP8→KDA 交接。
- **研判**：【观点】三者属「Rubin/Hybrid 内核性能与 KV 正确性」（第 4/8 条）——TRT-LLM 继续在 SM107 堆叠 MiniMax/Kimi 的低延迟 decode 内核；#21547 为 KV pool 选择回归修复。证据等级：官方一手（标题/正文级为主，#21399 含 perf 声明级）。

### 4.5 其余 TRT-LLM 要点（官方一手，标题/正文级）
- `#19285`（FP32/BF16/FP16 unified self-sampling GVR templates）、`#20068`（PrimsTS FMHA context 的 VC-Attention-QK(16,8) recipes）、`#18913`（opt-in 解聚传输诊断）、`#21489`/`#21503`/`#21108`（Kimi-K2/K3、GLM tool parser 参数释放/放弃修正）、`#21505`/`#21500`/`#21504`/`#21063`（KimiK3/KimiK25/Glm5Next/DeepseekV41 ForConditionalGeneration 覆盖测试）、`#21367`（logit transform + sliding window DFlash drafters 声明）、`#21584`（不支持的非生成请求/executor loop 干净失败）、`#21226`（video serving 合并 Omni references）、`#20973`（vision pixel norm CPU→GPU）、`#21465`（ADP connector fixtures 对齐 executor 契约）、`#20843`（release/1.3 批量合回 main batch 2，维护）、`#21464`/`#21608`/`#21627`/`#21617`/`#21633`/`#21626`/`#21623`/`#21661`（多批 main pre/post-merge waive failed cases，infra 噪声）。

---

## 5. llama.cpp：2026-11-04 提交（22 条）——v0.9.0 正式发布（OCP FP8 E4M3 + flash_attn_ext_rows + ggml v0.29.0 + 四新架构）

### 5.1 **llama.cpp v0.9.0 正式发布（#31646 bump，2026-11-04T13:08:06Z）**——本期头号事件
- **主题**：年度级正式版本，核心为 OCP FP8 E4M3 量化、新 `flash_attn_ext_rows` op、ggml 升 v0.29.0；新增四架构 + 音频生成/转写 + Gemma4 prefill 优化。
- **内容详情（发布说明要点）**：
  - **量化**：OCP FP8 E4M3 张量类型（saturating 模式）+ 转换辅助 + 新 ftype（#27512，新增 `LLAMA_FTYPE_MOSTLY_F8_E4M3`）。
  - **llama**：`flash_attn_ext_rows`（#29510）——每 query slice 注意力自己的 K/V 行集合，使统一 KV 在多序列下高效、恢复非统一 KV 的 prefill 性能。
  - **ggml**：batched tensor copy API（`ggml_backend_tensor_copy_n`/`set_n`/`get_n`，取代 `_2d`，Metal/CUDA 实现，#30881）；ggml v0.29.0 含 FP8 E4M3、flash attention rows、NCCL 多节点张量并行、CPU/CUDA/SYCL/Metal/Vulkan/OpenCL/Hexagon 全后端性能（SVE CPU kernel、SYCL DPAS flash attn for quantized KV、CUDA fused MoE MMQ、Metal 更快 MXFP4 mat-vec、Vulkan split-K、OpenCL 扩大量化 GEMM/GEMV）。
  - **新模型**：Clover-35B-A3B-Preview（diffusion, AR mode, #31306）、Limite 1B-Violetto（#29433）、Metis-1.6（Praxis~2.9B + Logos~10.2B，hybrid Mamba-2/GQA + fixed-depth recurrence + LatentMoE, #31134）、Mistral Large 4（#31376）。
  - **Core**：gemma4 graph 优化 E2B/E4B 大幅 prefill（#31356，M4 Pro Metal E2B pp512 ≈2.85×）；LFM3 NVFP4 GGUF（#31608）/ tensor split（#31431）/ audio codec 量化（#31386）；spec dec 保存恢复 MTP hidden state（#30141）/ 记 LFM3 层输入（#31598）；dflash device copies 拷隐藏特征（#31506）；kv-cache 旋转常量留 GPU（#31073）/ 共享 KV 层注意力旋转修正（#31406）/ 无 cell 序列绝不返 0 kv_rows（#31570）/ 拒回滚过最后 ubatch 快照（#31368）；Gemma4 MTP assistant 随 ctx_other/flash-attn（#31635）；DSV4 去简单 batch 拆分（#31300）。
  - **多模态**：Chatterbox TTS（#30993）、Nemotron 3.5 Transcribe（40 语言离线转写 + speaker 归属, #31471）、Pocket TTS 预定义 voices（#31429）、qwen3-vl 视频 prompt（#31619）。
  - **Server/UI**：`/props` 报预定义 TTS voices、`/v1/responses` 合并 extra_params（#30959）；PDF 附件文本/缩略图（#31468）；UI 右键菜单（#31545）等。
  - **连续第三年入选 GitHub Octoverse 报告**。
- **应用场景**：llama.cpp 全场景（量化、统一 KV 多序列、多节点 TP、端侧/异构后端、diffusion/音频多模态）。
- **研判**：【事实】发布说明列明上述全部要点与 PR 链接，含 gemma4 E2B pp512 ≈2.85×、NCCL 多节点 TP +59% pp512（#28967，见 11-04 档案）、Metis-1.6 hybrid Mamba-2/LatentMoE 等。【观点】属「端侧/通用推理里程碑」（第 4 条端侧主线）——OCP FP8 E4M3 + flash_attn_ext_rows + ggml v0.29.0 的 NCCL 多节点 TP 把 llama.cpp 从单卡端侧向多机协同推进；统一 KV 多序列效率修复是长上下文多轮对话的关键。证据等级：官方一手（发布说明全文 + 多 PR 链接）。

### 5.2 窗口内其余 llama.cpp 提交（官方一手，标题/正文级）
- `#31620`（common 修 `-hf` tag 匹配——`:F16` 误选 BF16 文件因 `BF16` 含 `F16`，已改 token 边界；`_` 作边界以匹配 `Q4_K_S` 等）。
- `#31584`（ggml 按 stride 实际抵达字节 bound views——视图原按同形状连续张量尺寸校验，行重叠失败/宽 stride 通过，现传 stride 校验实际抵达字节，与 `ggml_nbytes` 一致）。
- `#31609`（ggml 修 bf16）、`#31607`（ggml 用 `__riscv_v` gate RVV 代码于目标扩展，Clang 为 rv64gc 也定义 `__riscv_v_intrinsic` 致误入 RVV 分支）、`#31631`（gguf-py 校验每个 tokenizer merge pair 非仅首对，防静默错配）、`#31639`（sidecar 禁 `_`）、`#31444`（删 `common_get_path_from_env`）、`#31468`（UI PDF 预览）、`#31635`（Gemma4 assistant 随 ctx_other/flash-attn）、`#31605`（SYCL docker 加 oneDNN）、`#31348`（webgpu 返回 maxBufferSize 并避整缓冲绑定）、`#31613`（drama-3 tts）、`#31632`（server realtime 异常改发 error event 而非崩）、`#31545`（UI 右键动作）、`#31439`（CI 自托管 CPU 跑 hf/github）、`#31289`（hexagon LIGHTNING_INDEXER 支持：prefill hmx、decode q8_0 gather transpose、pipelined hmx、K(f32)/q4_0 支持、dma dbl buffering）。
- **研判**：【观点】`#31584`（视图 stride 边界校验）与 `#31631`（tokenizer merge pair 逐个校验）属「张量/分词静默正确性」补强；`#31289` Hexagon LIGHTNING_INDEXER 延续端侧 NPU 推理覆盖。均为标题/正文级一手。

---

## 6. 累计结论与研判（沿用 + 强化）

**沿用 2026-11-04 档案（覆盖 11-03）第 1–10 条：**
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

**本期强化（llama.cpp 发版 + vLLM 量化后端整合 + 跨引擎 KV 正确性延续）：**
- **第 3 条（默认/能力移除风险）延展到「量化后端整合」**：vLLM `#65542`/`#65540` 把 Marlin 退守 16-bit 激活、W4A8（INT8/FP8 激活）整体移交 Humming，移除 `VLLM_MARLIN_INPUT_DTYPE`——是「能力移除即风险载体」的教科书案例，W4A8 用户须主动迁移否则启动报错；与往期「默认后端/量化切换」同源。
- **第 4 / 6 / 8 条（异构/量化/静默正确性）多条铁证**：vLLM `#60181`（NIXL `kv_buffer_device=cpu` 下 h2d sync 用未填充 host 页覆盖已接收 KV → GB10 Gemma-4-E2B 4/4 逐位错、PR 修复）、`#63752`（缺 v_proj 的 checkpoint 静默送未初始化行、改 `ValueError`）、`#61463`（ll_bf16 split-K router GEMM 未等 peer CTA 驻留即 DSMEM store → Xid 13 硬崩）；SGLang `#48001`（unified-memory 下 routed DFLASH/DSPARK draft context-KV 误写 FULL id 空间错行，silent）、`#48292`（disagg KV checksum 在 DSv4 PD/HiSparse decode 无法工作改启动期拒绝）。
- **第 5 / 8 条（PD/KV 一致性）跨引擎大巡检续**：vLLM `#59747`（拒不同 P/D block size push 握手）、`#59383`/`#63696`（KV Offload 惰性块回收/完整 store）、`#65331`（Mooncake 登记 extensible KV 已提交块）、`#64677`（UMBP 逐布局拷贝 per-head runs）；SGLang `#47255`（bind NIXL workers 到设备）、`#48002`（拒 sharded-padded CP KV + unified memory）、`#39799`（非 DSA EAGLE draft KV pool 按 DCP 加宽预算）、`#43911`（HiSparse checksum 失配释放行）；TRT `#21021`（mixed-precision KV → 投机/sparse/hybrid）、`#21547`（恢复 DSA FMHA 辅助 KV pool）、`#21281`（draft-KV mirror 按历史定尺寸，needs follow-up）。
- **第 8 条新增「KV 准入智能」维度**：vLLM `#54625` cache-aware admission ordering——在 FCFS + prefix caching 下引入有界前瞻的缓存感知准入，避免冷请求挤掉即将复用的 prefix 块；是 KV 调度从公平到缓存感知的架构级新增。
- **第 4 条（端侧/通用）里程碑事件**：llama.cpp **v0.9.0** 正式发布——OCP FP8 E4M3 + `flash_attn_ext_rows`（统一 KV 多序列高效）+ ggml v0.29.0（NCCL 多节点 TP、全后端性能），把端侧/通用推理从单卡推向多机协同与更宽量化格式。

---

## 7. 趋势判断与可执行清单

**趋势判断**
1. **llama.cpp 进入「年度版本 + 多机协同」阶段**：v0.9.0 把 OCP FP8 E4M3、统一 KV 多序列、NCCL 多节点 TP、四新架构与音频多模态一并落地，端侧/通用推理的工具箱显著扩宽。
2. **vLLM 量化后端「收口到 Humming」**：Marlin 退守 16-bit、8-bit 激活整体移交 Humming（#65542/#65540），是 vLLM 量化 kernel 架构的明确整合方向——W4A8 用户须主动迁移。
3. **KV 正确性持续为跨引擎最密集主线**：本期 vLLM 5 项 + SGLang 5 项 + TRT 3 项 KV/PD 正确性修复，覆盖 NIXL host buffer 覆盖、unified-memory draft 错行、disagg checksum 拒绝、mixed-precision KV 延伸到投机解码——与「跨请求 KV 泄露/暴露」安全主题呼应。
4. **KV 调度从公平走向缓存感知**：vLLM #54625 的 cache-aware admission ordering 是新信号——prefix 驻留信息进入准入决策。
5. **新模型 day-0 节奏加快**：SGLang #48415（Kimi-K3.1）、vLLM #65302（ZAYA1）、llama.cpp v0.9.0（Clover/Limite/Metis-1.6/Mistral Large 4）同窗多新架构落地。

**可执行清单（建议 serving / 研发方）**
- [ ] **vLLM W4A8 用户迁移到 Humming**：#65542/#65540——`VLLM_MARLIN_INPUT_DTYPE` 已移除，改 `--quantization humming` + `VLLM_HUMMING_INPUT_QUANT_CONFIG`；SM89/SM12x 的 W4A8-FP8 compressed-tensors checkpoint 不再走 Marlin（第 3 条）。
- [ ] **P/D KV 传输正确性门禁**：vLLM #60181/#59747（NIXL host buffer 覆盖 / 不同 block size 拒绝）、SGLang #48292/#48001/#48002（disagg checksum 拒绝 / unified-memory draft 错行 / CP KV 拒 unified）、TRT #21021/#21281（mixed-precision KV→投机、draft-KV mirror 定尺寸）——上线 P/D 前核对（第 5/8 条）。
- [ ] **模型加载分片完整性**：vLLM #63752（stacked 权重缺 shard 改 `ValueError`）——加载后核对无 `Following weights were not initialized` 报错（第 6 条）。
- [ ] **KV 容量/准入策略评估**：vLLM #54625（cache-aware admission ordering，opt-in）——高 prefix 复用、KV 压力下评估命中率与尾延迟（第 8 条）。
- [ ] **llama.cpp v0.9.0 采纳评估**：OCP FP8 E4M3（#27512）、`flash_attn_ext_rows`（#29510）、ggml v0.29.0 NCCL 多节点 TP（#28967）、gemma4 prefill（#31356）、Metis-1.6（#31134）按场景评估（第 4 条端侧）。
- [ ] **未全绿项以最终 CI + 实测为准**：SGLang #48415（Kimi-K3.1 day-0，CI 三套全红）、#48001（unified-memory draft KV，CI 未全绿）、TRT #21281（draft-KV mirror，Dev/QA needs follow-up）——采用前以发行分支与实测为准。

---

## 8. 来源索引与数据质量

**官方一手（提交 API + 发布说明 + PR 正文 + HN Algolia）**
- **vLLM**：https://github.com/vllm-project/vllm （**111 commits @2026-11-04**，均 HTTP 200；**v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）为窗口前二日发布，11-04 全为 `main` 常规提交**）。PR `#65542`/`#54625`/`#60181`/`#63752`/`#64760`/`#61463` 正文已抓。Marlin→Humming 整合（#65542/#65540）为量化后端迁移信号。
- **SGLang**：https://github.com/sgl-project/sglang （**62 commits @2026-11-04**，均 HTTP 200；**v0.5.23（2026-11-02T22:14:10Z）为窗口前二日发布**）。PR `#48415`/`#48292`/`#48001`/`#48003` 正文已抓。`#48415`（Kimi-K3.1 day-0）CI Base/Extra/AMD 三套均红。
- **TensorRT-LLM**：https://github.com/NVIDIA/TensorRT-LLM （**36 commits @2026-11-04**，均 HTTP 200；**v1.4.0rc2 2026-10-28T15:51:34Z** prerelease，窗口内无更新）；PR `#21021`/`#21242`/`#21281` 正文已抓。`#21281` Dev/QA 均 needs follow-up、无独立可验证测试结果（描述级/未全绿）。
- **llama.cpp**：https://github.com/ggml-org/llama.cpp （**22 commits @2026-11-04**；**v0.9.0 正式发布 2026-11-04T13:08:06Z**（tag `v0.9.0`），nightly 标签 `b12114`–`b12119` 于 2026-11-04 21:33Z–22:34Z 发布）；**v0.9.0 发布说明全文已抓取**（含 OCP FP8 E4M3 / flash_attn_ext_rows / ggml v0.29.0 / 四新架构 / 音频多模态要点）。

**社区 / 外部（二手线索，未核验）**
- HN（2026-11-04 窗口，Algolia HTTPS，numericFilters `created_at_i>1793754178,created_at_i<1793840577`）：引擎相关**无高信号官方更新故事**。`vllm` 21 命中多为通用 LLM 生态帖（「Go vs. Python for an LLM Gateway」「Open-source LLM Firewall」「GPUWorks – A 40 MB GPU cluster manager for vLLM/SGLang」2 分 Show HN 工具、「LLMs replayed 50 years of EPA glyphosate decisions」等），**非 vLLM/SGLang 引擎提交更新**；`sglang` 1、`llama.cpp` 2、`TensorRT-LLM` 0、`KV cache`/`speculative decoding` 查询因空响应未计数。个别边缘项「GPUWorks」为社区集群管理工具（二手线索/未核验）。均不计入结论。
- Reddit / YouTube / X：未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

**数据质量与限制**
- 计数口径：GitHub commit API 的 `since`/`until` UTC 日期窗口（**vLLM 111 / SGLang 62 / TRT-LLM 36 / llama.cpp 22**，均 HTTP 200）；按 committer date 过滤。注：本期首轮抓取曾把 parent SHA 计入致计数虚高 3×，已用 raw JSON 重解析校正为真实提交数。**2026-11-04 为周三工作日**，四家量较 11-03 窗口（124/62/53/36）整体回落（vLLM 111、SGLang 62 持平、TRT 36、llama.cpp 22）。
- 证据等级：带验证表 / 对照 / 字节级 / 硬件复现的记为「官方一手（含验证）」；仅提交标题或 PR 描述级未全抓的记为「标题/正文级」；TRT `#21281` 记「官方一手（描述级，Dev/QA needs follow-up、测试未独立验证）」；SGLang `#48415`/`#48001` 记「官方一手（day-0/CI 未全绿）」；社区文章记「二手线索/未核验」。
- 未调用外部浏览器（agent-browser）：本期数据源（GitHub API / releases / HN Algolia / PR issues）均可直接 HTTP 访问，沿用 2026-10-18 起确立的「API + PR 正文直抓」方法，未触发浏览器需求（与历次一致）。
- `find-skills`：经 SkillHub 语义检索（含「LLM inference GitHub commit monitoring」「github trending hotspot daily digest」「git sync commit push」「大模型推理 GitHub 提交采集 热点归档 仓库同步」）最高相关性约 0.050，返回编程专家.Skill / dev-expert / 育儿大师 / 技能发现·Find Skills 等通用编程/生产力类技能，**与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，未安装**（与历次一致）；本地 marketplace 目录为空。

> **归档空档说明**：本档案覆盖 **2026-11-04** UTC 窗口；**2026-10-29** UTC 窗口因 2026-10-30 当日自动化未执行而无档案（即 2026-10-29 档案覆盖 10-28、2026-10-31 档案覆盖 10-30、2026-11-01 档案覆盖 10-31、2026-11-02 档案覆盖 11-01、2026-11-03 档案覆盖 11-02、2026-11-04 档案覆盖 11-03、本期 2026-11-05 档案覆盖 11-04 之间，仅 2026-10-29 窗口一日空档）；**自 2026-10-30 窗口起已连续六期（2026-10-31、2026-11-01、2026-11-02、2026-11-03、2026-11-04、2026-11-05 六期档案分别覆盖 10-30、10-31、11-01、11-02、11-03、11-04）无空档**。本目录不补写中间日期内容，各期仅按自身关注窗口采集。如需回溯 2026-10-29 窗口，请以各引擎官方提交 API 按日期区间重新采集。
> 本档案仅覆盖 2026-11-04 UTC 窗口；相邻日期请以官方提交 API 按日期区间重新采集。
