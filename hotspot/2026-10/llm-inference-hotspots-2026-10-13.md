# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-13（北京时间）
- **关注窗口：** 2026-10-12 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-12 08:00 – 2026-10-13 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（均返回 HTTP 200）与官方 release notes 原文，提交时间以 API 返回值为准。本期与上一期（2026-10-02 档案，覆盖 2026-10-01）之间存在归档空档，本期不补写中间日期内容，仅按关注窗口采集。HN Algolia 公开 API 本期可访问；Reddit 未纳入本期采集范围。SkillHub 检索本次可访问，但未找到与本任务（官方提交采集 + 热点归档 + 仓库同步）匹配的技能，故未安装任何外部技能。

## 1. vLLM：KV Connector 生命周期、HiSparse 与稀疏 MLA 工程化

**主题**  vLLM 在 10 月 12 日的官方提交集中在 KV Connector 请求状态机、HiSparse 容量可观测性与 TP 挂起修复、稀疏 MLA workspace 与 ROCm MLA 主机侧开销，以及投机解码元数据与 EPLB。

**内容详情**  官方提交显示：
- MoRI-IO KV Connector 修复 WRITE 模式请求卡在 `WAITING_FOR_REMOTE_KVS` 的问题，改动落在 `moriio_common.py`、`moriio_connector.py`、`moriio_engine.py`，并新增 `test_moriio_write_done_routing.py`；
- HiSparse 在 KV cache capacity 日志中上报 host pool 占用，补齐分层缓存的容量可观测性；
- HiSparse 修复在「本步无 scheduled tokens」时 TP group 挂起的问题；
- 稀疏 MLA prefill 的 workspace size 改为可配置；
- ROCm MLA metadata build 削减主机侧 dispatch，提交标题声明约 21× 降幅，改动位于 `mla_attention.py` 与 `rocm_aiter_mla.py`，该倍数为提交标题中的特定测试声明；
- ROCm 增加 batch invariance 支持；ROCm AITER Sparse MLA 增加 densemha 支持；
- 投机解码：dflash/dspark 在所有模型后端均支持时跳过 metadata build；GLM-5.3-Flash 恢复跨 draft 迭代的 MTP index sharing；修复 adaptive block verification 的越界 lookahead；修正 spec decode 未遵循 draft model config 的问题；
- EPLB 支持 dspark draft 模型（DeepSeekV4）；
- MoE：新增 MoNe experts pruning/substitution 支持（含设计文档 `docs/design/expert_substitution.md`）；CUTLASS FP4 MoE expert parallel 应用 expert_map 重映射；Humming 路径跳过 w13 GEMM 前的冗余 MoE input copy；
- 其他：Qwen3.8-Flash-Next 支持 Sequence Parallelism；静态 FP8 quant fusion 声明 `group_shape`；Triton kernel 迁移离开 `tl.make_block_ptr`；多模态 encoder cache 复用前校验 chunk 边界；EC Connector 按实测 encoder 输出宽度确定传输大小；RecoverSSM 在 block 边界对齐 state indices；XPU sleep mode 内存释放与 FA block size 约束；新增 admission rejection 指标与按 watermark 状态统计完成请求。

**应用场景**  分离式 KV cache 与远端写入完成语义、HiSparse 分层缓存容量规划、稀疏 MLA 长上下文 prefill、ROCm 上的 MLA/量化与 batch invariance 一致性、MTP/dflash/dspark 投机解码、MoE 专家剪枝与专家并行、多模态编码器缓存复用。

**研判**  **事实：** 本期 vLLM 同时出现「KV 传输请求状态机修复」「HiSparse 无 token 步骤的集合通信挂起」「投机解码元数据与 draft 配置生效」三类修复，且 MoE 方向新增专家替换/剪枝能力。**观点：** 分层缓存（HiSparse）与远端 KV 传输的故障模式正在从「传输本身」转移到「空闲步、边界步、配置未生效」这类非典型路径；上线前应把无 scheduled token 的步、encoder chunk 边界、以及 draft model config 覆盖纳入回归。提交标题中的 21× 等倍数只代表特定测试场景，不可外推。

**来源**
- vLLM 官方提交 API（2026-10-12 UTC）：https://api.github.com/repos/vllm-project/vllm/commits?since=2026-10-12T00:00:00Z&until=2026-10-13T00:00:00Z&per_page=100
- MoRI-IO WRITE 模式请求卡住修复：https://github.com/vllm-project/vllm/commit/802feb6770
- HiSparse 上报 host pool：https://github.com/vllm-project/vllm/commit/fe341a53dd
- HiSparse 无 token 步骤 TP 挂起：https://github.com/vllm-project/vllm/commit/5b0b8f251e
- 稀疏 MLA prefill workspace 可配置：https://github.com/vllm-project/vllm/commit/d6aca837c3
- ROCm MLA metadata dispatch 优化：https://github.com/vllm-project/vllm/commit/8055293353
- ROCm batch invariance：https://github.com/vllm-project/vllm/commit/e0317b28ce
- 移除 cascade attention：https://github.com/vllm-project/vllm/commit/e1a5086256
- dflash/dspark 跳过 metadata build：https://github.com/vllm-project/vllm/commit/36f3972812
- GLM-5.3-Flash MTP index sharing：https://github.com/vllm-project/vllm/commit/fe4df3df0e
- MoNe experts pruning/substitution：https://github.com/vllm-project/vllm/commit/189bbfb9f4
- dspark draft 模型 EPLB：https://github.com/vllm-project/vllm/commit/a281ad32f0
- 最新公开 release：v0.31.0（2026-10-05）https://github.com/vllm-project/vllm/releases

**证据等级：** 官方一手；性能倍数仅为官方提交标题中的特定声明。

## 2. SGLang：HiCache 状态完整性、PD 失败语义与 DSA 索引 KV 分片

**主题**  SGLang 官方提交在 10 月 12 日集中推进 HiCache 的 MTP/Mamba 状态保留、PD 解耦的失败与撤回语义、DSpark/DFlash/EAGLE 投机路径统一，以及 DSA 索引器的 KV 分片与 CP 并行。

**内容详情**  可核验提交包括：
- HiCache 保留全部 MTP KV 与 recurrent state，涉及 `mem_cache/pool_host/mamba.py`、`hybrid_cache/hybrid_cache_controller.py`、`hybrid_pool_assembler.py`、`kv_cache_builder.py` 等 16 个文件，并新增 `test_mtp_hicache.py`、`test_hicache_mtp_state.py`；
- HiCache 支持 Mamba 在 buffer-only host memory 模式下运行；
- 主机侧 KV 传输缓冲改为显式捕获，改动落在 `pool_host/mha.py` 与 `pool_host/mla.py`；
- ROCm 上消除 HiCache migration stalls，并改用精确大小主机分配；
- NPU 路径稳定 HiCache 与注册 CI 测试；
- PD 解耦：DSpark PD 允许 NIXL 传输后端；Mooncake send 失败时 poison deferred abort ack；retract pause 时撤回已传输的 decode 请求；统一缓存外部 linker 加载失败时中止请求而非使调度器崩溃；
- NIXL Mamba checkpoint key 按 TP rank 分离；
- 投机解码：为 PD decode 的首个 EAGLE draft step 给出 proposal distribution；DFlash 将 Domino candidate pool 决策收敛到单点并在 accept 内应用模拟接受；DSpark 在 graph 与 eager 之间共享 draft params、logits dtype、confidence 与 accept tail；
- 内核：暴露打包的 FlashMLA MegaAttention 内核与 API；在 TRTLLM-MLA 后端 init 时预构建 `fixup_zero_kv` JIT 模块；
- KV Cache：DSA indexer 支持 KV cache 分片（`page_interleave_pool.py`）；QSA decode indexer 的工作量按上下文长度设界；
- MemCache：tree core 接口增加共享组件大小与路径 API；SWA 映射迁移从 locked-full 恢复中拆分为 `RecoverEvictedSWA`；
- 并行：DSV4.1 prefill 节点支持 CP 并行策略；DeepSeek V4 prefill 支持 interleave CP 与可断 CUDA graph；XPU 支持 decode context parallelism（DCP）；
- AMD：新增 flydsl MXFP8 dense GEMM kernel 用于 dpsk v4.1 prefill；MORI EP V2 支持；修复 DP attention + TP MoE decode NaN 并支持 bounded replay；重校准 MXFP4 GSM8K 阈值；
- 性能与调度：BF16 mixed-batch decode 尾部改走 decode attention；MoE 跳过 TMA down activation/quant 中的 block-padding 行；调度器在无可运行请求时不再锁存 `batch_is_full`；
- 依赖与文档：FlashInfer 升级到 0.7.1；GLM-5.3-Flash playground 文档更新为 B300 默认、CP8、EP 与 PD。

**应用场景**  HiCache + MTP/Mamba 混合模型的缓存一致性、PD 解耦的传输失败与撤回、NIXL/Mooncake 传输后端选择、DSA/QSA 稀疏索引与 KV 分片、GLM-5.3-Flash 与 DeepSeek V4/V4.1 的 CP/EP 部署、AMD MI 系列 MXFP8 预填充。

**研判**  **事实：** 本期 SGLang 出现大量「状态完整性」类改动（MTP KV + recurrent state、Mamba buffer-only、KV host transfer buffer 捕获）与「失败语义」类改动（linker 加载失败中止、Mooncake send 失败 poison ack、retract 时撤回）。**观点：** 混合缓存架构（HiCache + 线性/循环状态模型）的正确性风险已经明确落在「状态是否被完整保留与恢复」上，而不是缓存命中率；建议把 MTP 迭代间的 KV、recurrent state、以及跨 TP rank 的 checkpoint key 一致性做成强制断言，而非仅靠端到端精度测试兜底。

**来源**
- SGLang 官方提交 API（2026-10-12 UTC）：https://api.github.com/repos/sgl-project/sglang/commits?since=2026-10-12T00:00:00Z&until=2026-10-13T00:00:00Z&per_page=100
- HiCache 保留 MTP KV 与 recurrent state：https://github.com/sgl-project/sglang/commit/5ca0695aa6
- HiCache Mamba buffer-only host memory：https://github.com/sgl-project/sglang/commit/f5f5003323
- 捕获 KV host transfer buffers：https://github.com/sgl-project/sglang/commit/b6ba148384
- ROCm 消除 HiCache migration stalls：https://github.com/sgl-project/sglang/commit/6966c5d027
- DSpark PD 允许 NIXL 后端：https://github.com/sgl-project/sglang/commit/6a7abcfc9b
- DSA indexer KV cache 分片：https://github.com/sgl-project/sglang/commit/20c49f12c9
- PD decode 首个 EAGLE draft proposal distribution：https://github.com/sgl-project/sglang/commit/7953751bfa
- DFlash Domino candidate pool：https://github.com/sgl-project/sglang/commit/0efbb3d2b9
- DSpark graph/eager 共享 draft 状态：https://github.com/sgl-project/sglang/commit/392039d752
- FlashMLA MegaAttention 内核与 API：https://github.com/sgl-project/sglang/commit/93809202a3
- AMD flydsl MXFP8 GEMM：https://github.com/sgl-project/sglang/commit/d6853f5066
- 最新公开 release：v0.5.21（2026-10-02）https://github.com/sgl-project/sglang/releases

**证据等级：** 官方一手。

## 3. TensorRT-LLM：KV Cache Manager V2 转为默认（破坏性变更）

**主题**  TensorRT-LLM 在 10 月 12 日的最高信号变更，是把 KV cache manager V2 与 Python transceiver 从可选路径改为默认路径；同时推进 Blackwell cuteDSL MLA 内核、分离式 KV 接收同步与投机解码稳定性。

**内容详情**  官方提交显示：
- **破坏性变更：** 默认启用 KV cache manager V2 与 Python transceiver（提交标记 `BREAKING`），改动覆盖 84 个文件，包含 `_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`、`_torch/disaggregation/kv_cache_transceiver.py`、`_torch/pyexecutor/connectors/kv_cache_connector.py`、`py_executor.py`、`model_engine.py` 及多个 disaggregated 示例与 Slurm 配置；
- 允许模型主动「要求」KV cache manager v2，而不是被静默降级到 v1；
- Blackwell cuteDSL MLA 内核升级，支持 packed q heads 与 compact input，改动位于 `mla_decode_fp16.py`、`mla_decode_fp8.py`、`mla_helpers.py`；
- 分离式场景下改为轮询 gen-init KV receive fence，而不是阻塞 overlap scheduler；
- 投机解码：expanded draft probabilities 按 batch capacity 分配大小；SID-GR 优化修复 speculative beam 的 d2h 竞争；
- NVLink one-sided all-to-all 重构；
- Rubin locality-domain execution 通过 PyTorch 模型接入；
- Modeling V2 支持基于 LLM config 与 args 的动态开关；
- 测试侧：KV pool rebalance 精度测试容忍近似并列的 greedy flips；disaggregated logprobs 在 KV reuse 路径上进行比较；MiniMax M3 EAGLE3 draft model 目录与 GB300/VR200 disaggregated 性能脚本更新（EAGLE3 + NIXL）；
- 官方 release notes 新增「TensorRT-LLM Release 1.4」章节，当前各子节为空占位，属 1.4 发布准备中的一手信号；最新已公开 release 仍为 v1.3.0rc29（2026-09-29）。

**应用场景**  KV cache 管理路径迁移与容量规划、分离式部署（ctx/gen 分离）、Blackwell MLA 解码、投机 beam search、MoE 微基准、多节点 NVLink 通信。

**研判**  **事实：** KV cache manager V2 与 Python transceiver 已成为默认路径，且提交明确标注为破坏性变更；同时官方提供了「模型可要求 v2、不再被降级到 v1」的显式开关。**观点：** 这是本期全量数据中影响面最大的单项变更——默认值切换意味着既有部署脚本、自定义 KV connector、以及依赖 v1 行为的显存/容量假设都可能失效。升级前应以「是否显式声明所需 KV manager 版本」为检查项，并把 disaggregated 示例配置与 Slurm 脚本一并纳入回归，而不是只跑精度测试。1.4 release notes 目前为空占位，不应据其推断任何 1.4 特性。

**来源**
- TensorRT-LLM 官方提交 API（2026-10-12 UTC）：https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits?since=2026-10-12T00:00:00Z&until=2026-10-13T00:00:00Z&per_page=100
- BREAKING：默认 KV cache manager V2 与 Python transceiver：https://github.com/NVIDIA/TensorRT-LLM/commit/afc1fa2744
- 模型可要求 KV cache manager v2：https://github.com/NVIDIA/TensorRT-LLM/commit/ca3a6c4c62
- Blackwell cuteDSL MLA 内核升级：https://github.com/NVIDIA/TensorRT-LLM/commit/9257aaefbb
- disagg gen-init KV receive fence 轮询：https://github.com/NVIDIA/TensorRT-LLM/commit/a8d7911bed
- expanded draft probabilities 按 batch capacity：https://github.com/NVIDIA/TensorRT-LLM/commit/7f011f1020
- NVLink one-sided all-to-all 重构：https://github.com/NVIDIA/TensorRT-LLM/commit/a4390ca755
- Rubin locality-domain execution：https://github.com/NVIDIA/TensorRT-LLM/commit/e67c26201a
- release notes 新增 1.4 章节：https://github.com/NVIDIA/TensorRT-LLM/commit/efeda44574
- 官方 release notes 原文：https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/release-notes.md
- 最新公开 release：v1.3.0rc29（2026-09-29）https://github.com/NVIDIA/TensorRT-LLM/releases

**证据等级：** 官方一手；release notes 1.4 章节为空占位，不构成特性事实。

## 4. llama.cpp：KV 维度对齐修复、图重分配可观测性与多后端收敛

**主题**  llama.cpp 在 10 月 12 日的官方提交覆盖 no-cache attention 的 KV 维度对齐、推理期 compute graph 重分配上报、投机示例的 draft/rejection 修复，以及 Metal/Hexagon/Vulkan/SYCL 多后端缺陷修复。

**内容详情**  官方提交显示：
- no-cache attention 路径将 KV 维度补齐到 256：该路径构造 `[n_tokens, n_tokens]` mask 并传未 padding 的 K/V，导致 head size 512 的 CUDA kernel 因不满足 `FATTN_KQ_STRIDE` 倍数而多数长度回退 CPU，并使 `-fa auto` 的单 token 探测误关 flash attention；修复后按 mask 的 KV 维度对 mask、K、V 补零；
- 新增 `ggml_backend_sched_realloc_callback` 与 `ggml_backend_sched_set_realloc_callback`，在推理期上报图缓冲不足以再次预留的情况，区分「后端分配变化」与「图大小变化」两类重分配，并计入 perf stats；
- 投机示例修复 draft sampling 与 rejection handling（对应上游 issue #29975），并新增 parallel > 1 时的 kv-unified 要求；
- 模型支持：新增 vllm d3 系列模型支持，并修复其多序列场景；新增 vllm decision 2 模型；
- Metal：修复 `mul_mat_id` 在 ids 重复时的问题——原 `kernel_mul_mm_id_map0` 假设每个 token 至多一个匹配且每专家固定容量 `ne21`，重复 id 时会产生错误或丢行，并可能溢出到下一专家的切片，现改为构建紧凑的 per-expert (token, slot) 列表；另修复大 tensor 的 `kernel_mul_mm` 错误；
- 后端：Vulkan 为 f16 累加打包 FMA、CM2 quant block size 改用三元表达式、修复 32 位平台 buffer 日志；SYCL 在 row tail 保护 DMMV 读取；Hexagon 优化 CPY/CONCAT、q2_k bpw 与 activation、新增 IM2COL_3D；
- ggml 符号默认可见性改为 hidden；
- releases：b11572–b11577 于 2026-10-12 发布，本期仅记录为版本可追踪线索。

**应用场景**  长 head size 模型的 flash attention 可用性、低内存与显存受限部署、推理期显存峰值诊断、本地投机解码示例、Metal/Hexagon/Vulkan/SYCL 异构后端。

**研判**  **事实：** no-cache attention 的 KV 维度未对齐会静默导致 CUDA flash attention 回退 CPU 并被自动探测误关闭；Metal `mul_mat_id` 在重复 id 下存在越界写入风险。**观点：** 本期 llama.cpp 的两个高价值项都属于「静默降级 / 静默错误」类问题——性能回退不报错、结果错误在特定 id 模式下才暴露。本地部署应把 `-fa auto` 的探测结果、实际启用的后端 kernel、以及 MoE 重复路由场景纳入 smoke test，而不是只看能否跑通。图重分配回调为显存峰值问题提供了官方可观测入口，建议直接接入。

**来源**
- llama.cpp 官方提交 API（2026-10-12 UTC）：https://api.github.com/repos/ggml-org/llama.cpp/commits?since=2026-10-12T00:00:00Z&until=2026-10-13T00:00:00Z&per_page=100
- no-cache attention KV 维度补齐到 256：https://github.com/ggml-org/llama.cpp/commit/358a420ec4
- compute graph 重分配上报：https://github.com/ggml-org/llama.cpp/commit/972b5fcc19
- 投机示例 draft sampling/rejection 修复：https://github.com/ggml-org/llama.cpp/commit/77757a3467
- Metal mul_mat_id 重复 ids：https://github.com/ggml-org/llama.cpp/commit/073b74034b
- vllm d3 模型支持：https://github.com/ggml-org/llama.cpp/commit/efe3626b11
- vllm d3 多序列修复：https://github.com/ggml-org/llama.cpp/commit/cbe28bb3cb
- 官方 releases：https://github.com/ggml-org/llama.cpp/releases

**证据等级：** 官方一手；nightly release 仅为版本线索。

## 5. 当前会话已形成的累计内容结论

1. **KV Cache 正在成为带生命周期证明的分布式资源**：前期已记录 vLLM KV Connector、SGLang HiCache/分层路径、跨节点迁移、TensorRT-LLM verified write coverage 与 ownership；本期新增 MoRI-IO WRITE 模式请求状态机修复、HiSparse host pool 容量上报、SGLang 主机侧 KV 传输缓冲显式捕获、DSA indexer KV 分片，以及 TensorRT-LLM 将 KV cache manager V2 转为默认。
2. **P/D 解耦需要协议级失败语义**：前期关注 P/D 路由、Radix Cache、version groups 与 state stride 校验；本期补充 Mooncake send 失败 poison deferred abort ack、retract pause 撤回已传输 decode 请求、linker 加载失败中止请求、以及 disagg gen-init KV receive fence 改为轮询以不阻塞 overlap scheduler。
3. **投机解码的收益依赖状态、图和后端共同稳定**：MTP、DSpark/DFlash、EAGLE3、KDA 与 llama.cpp MTP 持续演进；本期新增 dflash/dspark 跳过 metadata build、GLM-5.3-Flash MTP index sharing、PD decode 首个 EAGLE draft 的 proposal distribution、DFlash Domino candidate pool 单点决策、DSpark graph/eager 状态共享、以及 llama.cpp 示例的 draft/rejection 修复。
4. **量化结论必须绑定硬件与实现路径**：NVFP4/MXFP4/FP8 在各引擎对应不同 kernel、KV 格式与设备路径；本期新增 SGLang AMD flydsl MXFP8 dense GEMM（dpsk v4.1 prefill）、MXFP4 GSM8K 阈值重校准、vLLM 静态 FP8 quant fusion 声明 `group_shape`、以及 TensorRT-LLM Blackwell cuteDSL MLA 的 FP16/FP8 解码内核升级。
5. **混合缓存架构的正确性风险落在状态完整性上**：本期 SGLang 同时保留 MTP KV 与 recurrent state、支持 Mamba buffer-only host memory、并按 TP rank 分离 NIXL Mamba checkpoint key，说明「缓存命中」与「状态正确」是两个独立问题。
6. **默认值与静默降级是本期主要风险面**：TensorRT-LLM 的 KV cache manager 默认切换属显式破坏性变更；vLLM 的 spec decode draft config 未生效、llama.cpp 的 flash attention 静默回退 CPU，属静默降级，两类都难以通过端到端精度测试发现。

## 6. 今日可执行清单

1. 针对 TensorRT-LLM KV cache manager V2 转默认，逐项核对现有部署：是否依赖 v1 显存/容量假设、自定义 KV connector 是否兼容、disaggregated 示例与 Slurm 脚本是否需要同步更新，并确认是否需要显式声明 KV manager 版本。
2. 为分离式 KV cache 增加请求状态机埋点：覆盖 `WAITING_FOR_REMOTE_KVS` 超时、WRITE 模式完成通知、abort ack 中毒、retract 撤回、linker 加载失败与 disagg gen-init fence 轮询。
3. 为 HiCache + MTP/Mamba 混合模型增加状态完整性断言：MTP 迭代间 KV、recurrent state、buffer-only host memory 模式、以及跨 TP rank 的 checkpoint key 唯一性。
4. 对 llama.cpp 部署验证 flash attention 是否真正启用：检查 head size 512 等场景下的 `-fa auto` 探测结果与实际后端 kernel，并把 compute graph 重分配回调接入显存峰值监控；对 MoE 重复路由场景补充 Metal 回归。
5. 对投机解码路径（MTP / dflash / dspark / EAGLE3）建立跨 PP、draft 长度、graph 与 eager 两种执行模式、空/短 batch 的一致性回归，重点验证 draft model config 是否被真正遵循。
6. 对 AMD 路径单独记录设备型号、dtype、KV dtype、kernel 与评测集：本期 MXFP8/MXFP4 改动集中在 gfx95x 与 AITER/flydsl 路径，不能与 NVIDIA 路径的精度/性能结论互相外推。

## 7. 社区与视频资料

- HN Algolia 公开 API 本期可访问。在 2026-10-12 窗口内检索到少量相关条目，均为个人项目或 Show HN，**无一手性能数据，标注为二手线索、未核验**：
  - “Janas: Run LLMs larger than RAM by streaming experts from SSD”（https://github.com/prabanta-dev/janas），HN 2026-10-12 22:05 UTC，16 points / 5 comments；
  - “Show HN: Reflex – A GGUF-native Rust/CUDA engine for scale-to-zero LLM routing”（https://console.runpod.io/hub/listing/lateos-ai/reflex），HN 2026-10-12 16:05 UTC，1 point / 0 comments；
  - “Show HN: Boole-1.7B, a 1-bit LLM that outperforms Bonsai-1.7B”（https://precisionnumerics.ai/blog/boole-1p7b），HN 2026-10-12 15:49 UTC，属模型而非推理引擎。
- Reddit 未纳入本期采集范围（往期记录其 API 受限），本期不据 Reddit 内容形成结论。
- 视频资料：本期未核验到可确认发布日期与内容的官方视频，故不列入。
- 技能检索：SkillHub（https://lightmake.site）本期可访问，检索 “LLM inference optimization”“推理优化”“github trending hotspot tracking” 等关键词，返回结果的语义相关度均处于噪声水平（约 0.045–0.050），唯一相近的候选 `super-llm-inference-optimizer`（超级大模型推理优化师，下载量 179）面向「输出推理优化方案」，与本任务所需的「官方提交采集 + 热点归档 + 仓库同步」不同，故未安装任何外部技能。

---

**归档状态：** 已按 `hotspot/2026-10/` 月度目录归档。
**证据边界：** 事实与观点已分开；官方提交中的性能标题、nightly 版本、空占位 release notes 与测试/配置变更均按原文范围记录，不推导为普遍结论；社区条目一律标注为二手线索且未核验。
