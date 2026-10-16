# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-16（北京时间）
- **关注窗口：** 2026-10-15 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-15 08:00 – 2026-10-16 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（均返回 HTTP 200）、官方 releases API，以及 vLLM 官方博客与官方文档原文；提交时间以 API 返回值为准。本期与上一期（2026-10-13 档案，覆盖 2026-10-12）之间存在 2026-10-13、2026-10-14 两日归档空档，本期不补写该两日内容，仅按关注窗口采集；对窗口内检索到、但发布日早于窗口的官方博客（2026-10-14 发布）单独标注发布日期。HN Algolia 公开 API 本期可访问；Reddit 未纳入本期采集范围。技能检索本次可访问，返回候选技能均与「官方提交采集 + 热点归档 + 仓库同步」不匹配，故未安装任何外部技能（详见第 9 节）。

---

## 1. vLLM 官方博客：Push 模式 NIXL KV 传输（P/D 解耦）

**主题**  vLLM 官方博客发布《Push-Based KV Transfer to Improve TTFT and TPOT for Disaggregated Inference in vLLM》，提出 push-based NIXL connector（`NixlPushConnector`），把 prefill 后的 KV 传输协调与设置移出 TTFT 关键路径。

**内容详情**  官方博客原文要点：
- **发布信息：** 2026-10-14 发布，作者 Sunita Nadampalli（AWS）、Nicolò Lucchesi（Mistral），标签 `#disaggregation` `#kv_cache` `#nixl` `#performance`。
- **pull 模式的问题：** 默认 `NixlConnector` 为 pull 模式——prefill 算完 KV 后，router 把源元数据（含 prefill block IDs）转发给 decode 实例；decode 实例再分配目标 block、准备 NIXL 传输、发起 NIXL READ 拉取。这些协调与设置步骤**串行发生在 prefill 之后、decode 开始之前**，拉长了 TTFT 关键路径。
- **push 模式设计（三阶段协议）：**
  1. **Registration（decode 侧）**：decode 实例分配本地 block 时，**在同一个引擎 step 内**以 NIXL notification 把「自身身份 + 刚预留的 block 位置」发给 prefill，因此 connector 在 prefill 完成前就已知目标 block；
  2. **Rendezvous（prefill 侧）**：注册与 prefill 完成可能任意顺序到达，后到者触发传输；按标识符配对，可正确处理被抢占（preempted）或重试（retried）的请求；
  3. **Transfer（prefill 侧）**：建立/复用 NIXL 连接，发起 **NIXL WRITE** 直接写入 decoder 已注册内存，并逐个寻址目标 tensor-parallel rank。两侧可使用不同 block layout 与不同 block size。每个 WRITE 携带完成通知，decoder 统计**每个写入它的 producer rank 各一个**通知，全部到达后才视为 KV 已接收。
- **线程模型：** 每个 prefill worker 使用**专用后台传输线程**——prefill 主循环把 KV block 交给后台线程后立即返回模型执行；后台线程负责建连、把 block IDs 映射到 memory descriptors、**把连续 block 合并成更大的传输区域**、提交 NIXL WRITE。
- **数据路径不变：** pull 与 push **使用相同的 NIXL 与 EFA/RDMA 数据路径**，差异仅在协调时序。
- **基准结果（Qwen3-32B，1P1D，Amazon EC2 P5en + EFA，decode 侧 `--no-enable-prefix-caching`，128 output tokens，输入长度 1k–16k，请求率 1–8 QPS）：**
  - **TTFT：** 低负载短 prompt 下两者几乎相同；随 QPS 与输入长度增长 pull 急剧劣化，**8 QPS + 16k 时 pull 约 1,744 ms，push 约 540 ms，约 3× 降幅**；
  - **TPOT：** 轻负载下两者约 5.6–5.8 ms；**4–8 QPS + 16k 时 pull 升至约 10.4 ms，push 稳定在约 7 ms**；
  - 官方头条表述为 **TTFT 最高提升 3×、TPOT 最高提升 30%**，并明确「结果依赖工作负载与配置」。
- **启用方式：** 把 prefill 与 decode **两侧** server 的 `kv_connector` 都设为 `NixlPushConnector`（替代默认 `NixlConnector`）。
- **路由权衡（官方明示）：** pull 模式允许 router 在 prefill 完成后才选 decode 实例，对负载感知路由与 QoS 更灵活；push 更早选定 decode 实例，换来 **prefill 实例更早释放 KV block**。
- **局限性：** 结果依赖工作负载与配置；push **不适用于双向 KV 传输**场景；官方计划把后台 NIXL 线程模型引入 pull 模式 connector（PR #45211）。
- **复现环境（官方附录）：** `p5en.48xlarge`（8×NVIDIA H200）×2 节点，同一 AZ（ap-south-1b），Ubuntu 24.04.4 / kernel 6.14.0-1018-aws，NVIDIA driver 580.126.16 + CUDA 13.0，每实例 16 个 EFA 接口（各 200 Gb/s，合计 3.2 Tb/s），EFA installer 1.50.0 / libfabric 2.6.0amzn1.0；`vllm==0.29.0` + `nixl==1.4.1`；NIXL 只在该 GPU 共享 PCIe switch 的 2 条 rails 上注册该 GPU 的 KV cache，确保传输 PCIe-local。

**应用场景**  P/D 分离部署的 TTFT/TPOT 优化、高请求率长上下文场景、EFA/RDMA 互联的多节点集群、KV 传输与路由策略（负载感知 vs 低延迟）取舍。

**研判**  **事实：** 该博客为 vLLM 官方一手来源，给出 push/pull 两种 connector 的完整协议、线程模型与可复现基准；收益的核心机制是「把 decode 侧的 NIXL descriptor 准备与传输提交从延迟关键的主循环挪到 prefill 侧后台线程」，而非更换数据通路。**观点：** 这是本期最高价值的单项一手材料——它把 P/D 解耦的性能瓶颈从「带宽」明确改写为「**协调串行化 + 关键路径上的 CPU 工作**」，与本期 vLLM 提交中大量 KV connector/offload 的并发与背压修复方向一致。3× / 30% 是 Qwen3-32B + 1P1D + P5en + 关闭前缀缓存这一特定配置下的数字，**不可外推**为通用倍率；且 push 模式牺牲了「prefill 后才选 decode」的路由自由度，双向 KV 传输场景仍应留在 pull。建议在自建 P/D 集群上按同一脚本（`serve_pd.sh` / `run_sweep.sh`）复测后再决策。

**来源**
- vLLM 官方博客：https://vllm.ai/blog/2026-10-14-nixl-push-connector （发布日期 2026-10-14）
- push 模式 NIXL KV connector 设计文档：https://docs.vllm.ai/en/latest/design/nixl_kv_push_connector/
- P/D 分离服务指南：https://docs.vllm.ai/en/latest/features/disagg_prefill.html
- 复现脚本与绘图：https://vllm.ai/assets/repro/2026-10-14-nixl-push-connector/serve_pd.sh 、`run_sweep.sh`、`cleanup.sh`、`plot_push_vs_pull.py`
- 后续 PR（把后台线程模型引入 pull 模式）：https://github.com/vllm-project/vllm/pull/45211
- 基准脚本 PR：https://github.com/vllm-project/vllm/pull/57567

**证据等级：** 官方一手（官方博客 + 官方文档）；性能倍数为官方特定环境实测，不可外推。

---

## 2. vLLM 官方博客：无 confidence head 的自适应验证（在线拟合接受率估计器）

**主题**  vLLM 官方博客发布《Adaptive Verification Without a Confidence Head》，用**在线拟合的逻辑回归估计器**替代 confidence head，使 EAGLE / MTP / DFlash 以及无 head 的 DSpark checkpoint 也能用上 Adaptive Verification（AV）。

**内容详情**  官方博客原文要点：
- **发布信息：** 2026-10-14 发布，作者 Giancarlo Delfin（Inferact），建立在此前 Adaptive Verification 工作（Lucas Wilkinson / Red Hat、Benjamin Chislett / NVIDIA）之上；相关 PR #52228、#47808。
- **问题：** AV 需要逐 token 接受概率估计，此前**只支持自带 confidence head 的 DSpark drafter**；EAGLE、MTP、DFlash 及部分 DSpark checkpoint 完全无法使用 AV，而补 head 意味着**重新训练 drafter**，对已在线服务的 checkpoint 不可行。
- **方案：** 轻量逻辑回归，特征仅为 draft 分布的**单个标量**——最可能 token 的 log-odds，即 `logit(q_max)`；参数为**一个跨位置共享斜率 ω + 每个 draft 位置一个截距 βi**，共 **K + 1** 个参数；利用目标模型正常验证过程已产生的 accept/reject 决策**在线拟合**，因此**任何现有 draft checkpoint 免费获得 AV**。
- **特征四轴评估（约 4.5M labeled drafts，DeepSeek-V4-Flash + DSpark，temp=1、top-p=0.95、conc=64，未开启 AV，五个数据集）：**

| feature | lossless | AUC ↑ | log-loss ↓ | ECE ↓ |
|---|---|---|---|---|
| log(q_max) | yes | 0.8069 | 0.4813 | 0.0592 |
| log(q_max) − log(q_2nd) | yes | 0.8156 | 0.4437 | 0.0105 |
| H(q) | yes | 0.8321 | 0.4577 | 0.0638 |
| **logit(q_max)（最终采用）** | **yes** | **0.8336** | **0.4284** | **0.0100** |
| logit(q_sampled) | **no** | 0.8553 | 0.4040 | 0.0152 |
| q_max（EAGLE-2 式直接当概率用） | yes | 0.8259 | 0.4880 | 0.0736 |
| trained confidence head | yes | 0.8191 | 0.4449 | 0.0052 |

- **关键取舍：** `logit(q_sampled)` AUC 更高但**非无损**（基于 q_sampled 裁剪会让「是否在位置 i 验证」依赖 x_i 本身，幸存 draft 不再服从 q，输出分布偏向 drafter 高置信 token）；`q_max` 直接使用 AUC 甚至优于 confidence head，但**校准差约 7 倍 ECE**（0.0736 vs 0.0100），而 AV 的预算决策需要对预测接受概率求和，故必须校准。
- **与 confidence head 对比（全位置平均）：** estimator **AUC +0.0146、log-loss −0.0166**，ECE 略差（+0.0048）；**排除第一个 draft 后**：AUC +0.0278、log-loss −0.0323、ECE 仅 +0.0013。逐位置看，head 在第一个位置更好，estimator **从第二个位置起领先，最后一个位置约领先 0.08 AUC**——而 estimator 的优势正好落在**实际发生裁剪的位置**。
- **拟合细节：** IRLS（迭代重加权最小二乘）在 log-likelihood 的二阶 Taylor 近似上求解；每 N 个 decode step 触发一次 refit，期间只累积 Hessian（(K+1)×(K+1)）与梯度，内存上界 **O(K²)**，而非收集原始标签的 O(N×B×K)；仅**首个拒绝之前**的 draft token 参与训练。两项缓解深度标签稀疏的设计：**Shared Slope**（代价仅 0.0002 AUC / 0.0003 log-loss）与 **Step Damping**（按 n/(n+const) 缩放更新）。
- **斜率与截距的分工：** 固定模型对变换工作负载时，斜率几乎不动（**0.51–0.55**，与 mean draft entropy 的 r = −0.08），而第一个 draft 位置的截距与 draft entropy 相关 **r = −0.89**（从易预测负载的 ~0.5–0.6 降到最难负载的低于 0）；跨模型对斜率范围为 **0.45–0.59**。官方解释：斜率衡量 draft 模型「知道自己何时正确」的程度，截距吸收各位置基准接受率。
- **启用方式：** 当 draft 模型 (1) **无 confidence head** 且 (2) **所有 backend 均支持 variable-length attention** 时，online estimator 被**自动选择**；对无 head 的 draft model 设置 `enable_adaptive_verification: true` 即可（示例配置中另有 `draft_sample_method: "probabilistic"`）。
- **局限性（官方明示）：** 目标模型 attention backend 必须支持 variable-length verification；**MTP 与 EAGLE 从 AV 获益少于 DFlash/DSpark**，因为这类自回归 speculator 为可能被裁剪的 draft token 付出的 drafting 成本比例更大。

**应用场景**  投机解码在 compute-bound 大 batch 场景下的算力节省、无 confidence head 的存量 drafter 升级、DFlash/DSpark 部署、投机长度与并发的 Pareto 取舍。

**研判**  **事实：** 官方博客给出完整特征消融、与 trained confidence head 的逐项对比、拟合算法与内存上界，并明确 estimator 是「**缺失 head 时的 drop-in alternative**」而非 head 的替代品。**观点：** 这一改动的工程价值大于算法价值——它把「AV 需要重训 drafter」这一前置成本降为零，因此 **AV 的实际可用范围从单一 drafter 类型扩展到全部 draft-model-based speculator**。但官方同时指出 MTP/EAGLE 收益有限，说明 AV 的收益与 drafter 的「可裁剪比例」强相关：越接近并行 draft（DFlash/DSpark），裁剪越省算力；越接近自回归 draft（MTP/EAGLE），draft 成本本身已花掉。落地时应按 drafter 类型分别评估，而不是把 AV 当作全局开关。ECE 差异（0.0100 vs 0.0052）意味着在极低接受率负载下 estimator 的预算分配会比 head 略保守或略激进，建议监控实际接受率与预测值的漂移。

**来源**
- vLLM 官方博客：https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator （发布日期 2026-10-14）
- 本特性 PR：https://github.com/vllm-project/vllm/pull/52228
- Adaptive Verification 基础 PR：https://github.com/vllm-project/vllm/pull/47808
- 背景材料（官方博客，早于窗口）：Adaptive Verification in vLLM: DSpark confidence-scheduled verification，https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification

**证据等级：** 官方一手；AUC/log-loss/ECE 为官方在 DSV4-Flash + DSpark 特定数据集上的测量值，不可外推到其他模型对。

---

## 3. vLLM：10-15 提交（89 条）——KV 传输并发、MoE 统一与投机解码清理

**主题**  vLLM 在 10 月 15 日的 89 条官方提交集中在：KV connector / KV offload 的并发与背压正确性、UMBP 多级 KV 缓存抽象落地第一片、DeepGEMM MegaMoE 专家层统一并新增 NVFP4、投机解码的**能力移除**与无损性修复，以及 ROCm/gfx1250 路径扩展。

**内容详情**  官方提交 API 显示（节选可核验项）：
- **KV 传输与分层卸载（本期最大簇）：**
  - NIXL connector **拒绝**在异构 TP head split 下使用 NVFP4 KV cache（`#61865`）；
  - NIXL 把 **packed MLA KV rows 注册为单个区域**（`#61074`）；
  - Mooncake 每个 peer **只发送一次 KV layout**（`#59825`）；
  - CPU offload region **拆分到多个 shm 文件**（`#59546`）；
  - 暴露 `OffloadingManager.lock()/unlock()`，并把 **tiering 控制面移出调度线程**（`#58168`）；
  - 只 pin 各 worker **自己的** offload slots（`#59544`）；
  - 为未应答的 P2P lookup 等待**设上界**（`#56836`）；`mark_miss()` 不再翻转 pending/in-flight 的异步 lookup 探测（`#59096`）；
  - 新增 **`PROMOTION_LATENCY` 直方图指标**用于 tiering offload（`#53910`）；
  - peer teardown 时**总是尝试移除 NIXL agent**（`#60279`）。
- **UMBP 第一片：** `[UMBP][1/10]` 新增共享 connector、运行时契约与请求生命周期（lookup / load / store / block ownership），并引入 `UMBPStoreConnector` facade；本切片**不注册 MORI 后端**，实现位于 `tests/v1/kv_connector/unit/test_umbp_core.py` 所覆盖的契约层（`#61789`，源自 RFC `#57187`《Scheduler-Aware Multi-Tier KV Caching for vLLM with MORI-UMBP》，原始完整 PR `#58568`）。
- **投机解码：**
  - 为**共享 target head 的 drafter** 增加**量化 draft lm_head**（`#59973`）；
  - 修复**跨 block 验证轮次复用随机抽样**的问题（`#61161`，lossless 相关）；
  - **DFlash/DSpark 的 draft slots 不再占用调度器 token budget**（`#59468`）；
  - **移除 Medusa 与 MLP speculator 支持**（`#61701`，净 −858 行，`method="medusa"` / `"mlp_speculator"` 现被 `SpeculativeConfig` 校验拒绝；官方理由：MLP speculator 早已是死代码，Medusa 不被 Model Runner V2 支持，EAGLE/MTP/DFlash/DSpark 已覆盖其用例）；
  - GLM5.3 路径**原地更新 draft decode metadata**，提交标题声明 **1.7% E2E 吞吐提升**（`#61873`）。
- **MoE 与量化：** DeepGEMM MegaMoE 的三个实现（DeepSeek V4/V4.1、Kimi K3）**统一为单一 `DeepGemmMegaMoEExperts` 层**，由「routed precision × activation × fusable shared-expert precision」特征矩阵驱动，并**新增 NVFP4 routed experts**（DeepGEMM `nvfp4_mega_moe`）（`#61538`，PR 自述端到端评测尚未运行）；FlashInfer TRT-LLM MXFP8 experts 按形状 padding（`#61293`）；跳过 TRTLLM fp8-block activation-scale 的 transpose copy（`#61773`）。
- **内核与注意力：** FlashMLA 升级到 **2026.09.30 release** 并支持 **interleaved V4.1 KV cache**（`#61371`）；mamba align block-table gather 融合为**单个 Triton kernel（7 → 1）**，提交标题声明该 kernel 在所有形状上 **6×**（`#58737`）；跳过 sliding-window block table 的 **null head**（`#61724`）；XPU MLA 集成进既有 `FLASH_ATTN_MLA` backend（`#58297`）。
- **HiSparse：** 图 profiling 之后**复用 pinned host pool**（`#61668`）。
- **并行与调度：** DP metadata sync 使用**低延迟 CPU all-reduce**（`#61767`）；`prefill_schedule_interval` 应用到数据并行之外（`#54627`）；indexer TP row sharding 按**总行数**而非 per-rank 行数（`#61647`）。
- **ROCm / AMD：** AITER **Gluon KDA kernels** 用于 decode 与 spec decode（Kimi-K3，`#57392`）；**gfx1250** 优化与模型启用（`#58800`）；preshuffled FP4 GEMM config 按**逻辑 K** 探测（`#59406`）；fp16/bf16 GEMM weight row stride padding 以利 L2（`#55090`）；MoRI FP4 dispatch 用于 mxfp4 w4a4 MoE（`#59596`）；**DSpark adaptive verification**（`#58151`）；DeepSeek-V4 的 **DPA+ETP prefill compute/comm dual-batch-overlap**（`#57773`）；AITER MegaMoE fixed-slot dispatch（`#61262`）；DeepSeek-V4 hash routing 进 AITER Triton W4A16 MoE（`#60282`）；softmax MoE top-k（含 fused shared-expert sigmoid）走 AITER `topk_gating`（`#56226`）；MoRIIO HMA per-layer `block_len`（`#58585`）。
- **正确性与稳定性修复：** **拒绝 float16/bfloat16 KV cache dtype 不匹配**（`#60885`）；DeepSeek V4.1 decoder replay 期间保留 routing rows（`#61643`）；在 memory profiling **之前**创建 cuBLAS handle 以避免 OOM（`#61044`）；流式会话在 ABORT/ERROR 时终止并防止僵尸会话（`#58942`）。
- **模型与工程：** Decision 3.0 全尺寸支持（d3-flash/mini/nano/lite/edge，`#61778`）；ViT full CUDA graph 支持 DeepSeek-VL2（`#46005`）；EmbeddingGemma2 启用 torch.compile 与 CUDA graphs（`#61572`）；model-owned MTP shard selection 应用于 RunAI ModelStreamer（`#60759`）；Quark MTP `layer_quant_config` 重映射到 draft prefixes（`#56054`）；移除遗留 Ray compiled-graph executor（`#61712`）。

**应用场景**  分离式 KV cache 的并发与背压治理、多级/外部 KV 存储（UMBP、Mooncake、tiering offload）、MoE 专家并行与 FP4/NVFP4 推理、长上下文稀疏注意力、投机解码栈收敛、ROCm gfx1250 与 DeepSeek-V4/V4.1 部署。

**研判**  **事实：** 本期 vLLM 的 KV offload 相关提交在数量与细粒度上均为最高——涉及**等待上界、探测状态机、背压、agent 生命周期、控制面线程归属、pin 粒度**，且新增了 promotion 延迟指标；MoE 侧完成三个 MegaMoE 实现向单层的收敛并新增 NVFP4；投机解码侧**主动移除** Medusa 与 MLP speculator。**观点：** 三点判断——(1) KV offload 已经进入「**运维正确性**」阶段，问题不再是能不能卸载，而是卸载路径在超时、重试、peer 消失、控制面与数据面争线程时是否可预测，这类改动对线上稳定性价值高于任何单项性能优化，建议把 `PROMOTION_LATENCY` 与 lookup 等待上界直接接入监控；(2) UMBP 第一片刻意「不注册 MORI 后端」，说明官方在把 AMD 侧 KV 存储池按契约分层落地，短期内不应期待可用的 UMBP 端到端路径；(3) `#61538` 统一 MegaMoE 层并新增 NVFP4 属高影响重构，但 PR 自述端到端评测尚未运行，**升级前应按需回归**，不宜按「已完成验证」对待。

**来源**
- vLLM 官方提交 API（2026-10-15 UTC，89 条）：https://api.github.com/repos/vllm-project/vllm/commits?since=2026-10-15T00:00:00Z&until=2026-10-16T00:00:00Z&per_page=100
- NIXL 拒绝异构 TP head split 的 NVFP4 KV：https://github.com/vllm-project/vllm/commit/33212db2c2
- NIXL packed MLA KV rows 单区域注册：https://github.com/vllm-project/vllm/commit/7020600b1d
- Mooncake 每 peer 只发一次 KV layout：https://github.com/vllm-project/vllm/commit/6605802904
- CPU offload region 拆分到多个 shm：https://github.com/vllm-project/vllm/commit/21f4b6dc84
- OffloadingManager.lock()/unlock() 与控制面移出调度线程：https://github.com/vllm-project/vllm/commit/c5f8adc0b1
- 只 pin 本 worker 的 offload slots：https://github.com/vllm-project/vllm/commit/5341b5ba7d
- 未应答 P2P lookup 等待设界：https://github.com/vllm-project/vllm/commit/3f1bcfd82b
- PROMOTION_LATENCY 指标：https://github.com/vllm-project/vllm/commit/e10c10a644
- UMBP 第一片：https://github.com/vllm-project/vllm/commit/a97d19d792 （PR #61789）
- UMBP RFC：https://github.com/vllm-project/vllm/issues/57187
- 量化 draft lm_head：https://github.com/vllm-project/vllm/commit/784d1fff0c
- 避免跨 block 验证轮次复用随机抽样：https://github.com/vllm-project/vllm/commit/9edcc7fa7e
- DFlash/DSpark draft slots 不占 token budget：https://github.com/vllm-project/vllm/commit/7cd7982dfd
- 移除 Medusa 与 MLP speculator：https://github.com/vllm-project/vllm/commit/8530aed2b7
- GLM5.3 原地更新 draft decode metadata（1.7%）：https://github.com/vllm-project/vllm/commit/54d9b1cb48
- DeepGEMM MegaMoE 统一层 + NVFP4：https://github.com/vllm-project/vllm/commit/4d58fa254a
- FlashMLA 升级 + interleaved V4.1 KV cache：https://github.com/vllm-project/vllm/commit/f8460ef236
- mamba align block-table gather 融合（7→1）：https://github.com/vllm-project/vllm/commit/c825e1d235
- HiSparse 复用 pinned host pool：https://github.com/vllm-project/vllm/commit/49f944be3e
- DP metadata 低延迟 CPU all-reduce：https://github.com/vllm-project/vllm/commit/de7d70676d
- 拒绝 float16/bfloat16 KV cache dtype 不匹配：https://github.com/vllm-project/vllm/commit/48f17b77bb
- 最新公开 release 仍为 v0.31.0（2026-10-05）：https://github.com/vllm-project/vllm/releases

**证据等级：** 官方一手；提交标题中的 1.7%、6× 等倍数为特定场景声明，不可外推；`#61538` 的端到端评测状态以 PR 自述为准。

---

## 4. SGLang：HiCache 布局与 L3 key 统一、DSv4.1 请求窗口融合与调度器准入修正

**主题**  SGLang 在 10 月 15 日的 68 条官方提交集中在：HiCache 的 host 页布局与统一 L3 key、tail-only SWA 备份、DSv4.1 的 KV 拷贝与 SWA 池尺寸、DSA 索引器的 gfx950 融合与 CUDA graph、以及调度器准入循环的多次修正。

**内容详情**  官方提交 API 显示（节选可核验项）：
- **HiCache / MemCache：**
  - 新增 **`page_unified` host KV 布局与统一 L3 key**（`#39727`）：MHA 布局为 `(page, head_group, layer, 2, page_size, head_in_group, dim)`（K=0, V=1），MLA 布局为 `(page, layer, page_size, dim)`；目标是让一个 L3 chunk 成为**单段连续字节**，且 key 方案**不包含写入者身份**。PR 自述为**库层改动，尚无 server 参数与传输臂**，从 CLI 不可达；
  - **tail-only SWA 备份（opt-in）**（`#44173`）：`SGLANG_ENABLE_HICACHE_SWA_TAIL_ONLY_BACKUP=1`（**默认关闭**）把每个 radix node 的 SWA 备份限制在「以该 node 结尾的窗口」；动机是 `--hicache-host-memory-mode buffer_only` 下原有实现会把节点的**每一个** SWA page 写盘，而只在该 node 结束的 prefetch 只会恢复尾部窗口，长节点因此产生大量**无读者可用的写放大**；
  - 把 query beliefs **推迟到 rank 同步的 hit drain 之后**（`#43108`）。
- **P/D 解耦：** DSA indexer state **只在 `decode_prefix_len` 之后发送**（`#44140`）；`--dp-aware` 下把 `disagg_prefill_dp_rank` 转发给 decode（`#44072`）；sgl-router 新增 `--dp-rank-policy round_robin`（`#43511`）。
- **DSv4.1 / DSv4：**
  - **融合 request-window KV 拷贝**（`#44555`）；
  - encoder-replay SWA pool 按 **DSpark draft 的 stage 数**定尺寸并保留更多 prefix tails（`#43502`）；
  - **encoder-SWA replay 折叠进 extend forward**（`#43166`）；
  - **TRT-LLM decoder bounded replay with BCG**（`#44326`）；
  - **DeepSelect cluster top-k** 用于长行，索引 logits 使用 bf16（`#44051`）；
  - MoE routing **限制到 live SWA graph-tail 行**（`#44347`）；
  - 稀疏 prefill：**无 host sync** 构建 compressed lengths（`#44557`）。
- **DSA 索引器：** gfx950 **融合 DSA indexer** 用于 block-FP8 的 `wq_b`/`wk`（`#43643`）；DSA prefill 在 HIP 上以 **breakable CUDA graphs** 运行（`#43642`）；修复 legacy 与 v1 DSA top-k selector 的**溢出**（`#37625`）；fused top-k 在**溢出阈值 bin** 上精确（`#37941`）。
- **调度器（三条相邻修正）：** 在 **admission loop 头部**检查 prefill budget 并删除冗余 PD gate（`#44420`）；请求的 `req_to_token` row **只在 admission 计一次**（`#44401`）；**每次 admission 前**把已完成请求从 running batch 中丢弃（`#44351`）。
- **并行与通信：** PP 的 output send/recv 按 **rank 奇偶**排序以修复跨节点挂起（`#44453`）；CPU 侧新增 **group-aware CPU SHM collective kernels**（`#35435`）；修复 CP strategy 与 request-class override 单元测试中的过期 fake（`#44380`）。
- **内核与运行时：** 融合 MiMo verification 的 RoPE 与 KV cache 写入（`#43106`）；**sgl-kernel 升到 0.4.10**（`#44590`）；启动调优纳入 **FlashInfer 默认 tactic**（`#43927`）；AMD diffusion 启用 BF16 attention 与 MXFP4 编译（`#27702`）；**monokernels 单一运行时开关**统一 vendor monokernel 插件（`#44353`）；EPLB 支持 **Nemotron-H**（`#43685`）。
- **AMD：** **ROCm 10.1 for gfx1250 与 ROCm 10.0 并行发布**（`#42699`）；AITER pin 更新到 `2bb549102`（`#44379`）并修复其后 AITER build step（`#44534`）；AMD CI 跳过冗余 pip-cache chown 与 MORI 重建（`#42148`）。
- **服务与安全：** rust-server 应用 `x-override-*` 请求头（`#44312`）、在 `/health_generate` 健康时附带配置元数据头（`#44315`）；sgl-router 升级 rustls 到 0.23.46 修复 **RUSTSEC-2026-0346..0349**（`#44509`）。
- **NPU：** 修复 Qwen3.5 在 Pipeline Parallelism 下 rope sin/cos 预计算（`#29753`）；NPU 最佳实践与部署文档升级到 **CANN 9.1.0**（`#44432`）。
- **模型与文档：** FLUX.3 image（`#44418`）、MOSS-Image（`#44325`）、Nemotron 3.5 Super VL（`#44510`）、Jais 3 部署配方（`#42766`）、Step 5 Preview cookbook（`#44357`）、muse spark open cookbook（`#44527`）；`glm5_next` 的 `num_experts_per_tok` 默认值修正为 8（`#44093`）。
- **官方博客（2026-10-14，早于窗口）：** 《Step 5 Preview: Day-0 Inference and RL Support with SGLang and Miles》，见 sglang.io/blog。

**应用场景**  HiCache + SWA 混合模型的缓存写放大治理、跨进程/跨 rank 的 KV 键统一、DSv4.1 长上下文稀疏注意力与 encoder replay、DSA 索引器在 gfx950 上的融合与图捕获、调度器在高并发下的准入公平性、NPU/AMD 多后端部署。

**研判**  **事实：** 本期 SGLang 出现三类高信号改动——(1) HiCache 的**页布局与 key 语义统一**（库层，尚未接线）；(2) 多个针对**写放大与无读者写**的定向优化（tail-only SWA 备份、request-window KV 拷贝融合、compressed lengths 免 host sync）；(3) 调度器准入循环的三处相邻修正。**观点：** `page_unified` + 统一 L3 key 是本期最具结构性的一项——它把「L3 chunk 是单段连续字节」和「key 不包含写入者」作为设计前提，意味着后续跨引擎/跨进程复用 L3 KV 时不再需要按写入者做命名空间隔离；但官方明确当前**从 CLI 不可达**，因此不应据此规划上线。SWA 备份的 tail-only 优化则暴露了一个此前被忽视的事实：**在 buffer_only 模式下，SWA 组件的备份写量与「有读者可恢复的页数」不匹配**，属于典型的「缓存写放大」问题，默认关闭说明官方对正确性仍持保守态度。调度器三条修正同属一个主题——**准入时刻的状态视图必须自洽**（budget 在循环头部检查、token row 只计一次、已完成请求先出队），这类改动通常对应真实的高并发异常，建议在自建部署中重点回归长尾请求的准入行为。

**来源**
- SGLang 官方提交 API（2026-10-15 UTC，68 条）：https://api.github.com/repos/sgl-project/sglang/commits?since=2026-10-15T00:00:00Z&until=2026-10-16T00:00:00Z&per_page=100
- page_unified host KV 布局与统一 L3 key：https://github.com/sgl-project/sglang/commit/9ea5472833 （PR #39727）
- tail-only SWA 备份（buffer 模式）：https://github.com/sgl-project/sglang/commit/173e0f2e9b （PR #44173）
- query beliefs 推迟到 rank 同步 hit drain：https://github.com/sgl-project/sglang/commit/3c28614f12
- DSv4.1 融合 request-window KV 拷贝：https://github.com/sgl-project/sglang/commit/91f6ef2322
- encoder-replay SWA pool 按 DSpark draft stage 定尺寸：https://github.com/sgl-project/sglang/commit/1d809bbbbd
- 稀疏 prefill 无 host sync 构建 compressed lengths：https://github.com/sgl-project/sglang/commit/bc76a6b819
- PD 只在 decode_prefix_len 之后发 DSA indexer state：https://github.com/sgl-project/sglang/commit/2528cd2b91
- gfx950 融合 DSA indexer（block-FP8 wq_b/wk）：https://github.com/sgl-project/sglang/commit/a12814c158
- 调度器准入预算检查位置调整：https://github.com/sgl-project/sglang/commit/a776851c97
- req_to_token row 只计一次：https://github.com/sgl-project/sglang/commit/f9bdb74a5a
- 每次 admission 前丢弃已完成请求：https://github.com/sgl-project/sglang/commit/3b5ae4ea86
- PP 按 rank 奇偶排序修复跨节点挂起：https://github.com/sgl-project/sglang/commit/c5a5bf0d0b
- sgl-kernel 0.4.10：https://github.com/sgl-project/sglang/commit/4cca7fcd3e
- EPLB for Nemotron-H：https://github.com/sgl-project/sglang/commit/6fe5596ebb
- 最新公开 release 仍为 v0.5.21（2026-10-02）：https://github.com/sgl-project/sglang/releases
- 官方博客（2026-10-14）：https://www.sglang.io/blog/step-5-preview-day0-support

**证据等级：** 官方一手；`page_unified` 为库层改动，当前不可从 CLI 触达，不构成可用特性事实。

---

## 5. TensorRT-LLM：v1.4.0rc0 发布（含两项 BREAKING）与 10-15 提交

**主题**  TensorRT-LLM 于 **2026-10-14 发布 v1.4.0rc0**（prerelease），其中两项标注为 BREAKING 的 API 变更、Mooncake store pool 与 sparse KV cache 配置首次进入 release notes；10-15 的 35 条提交继续推进 KV cache V2 迁移、DeepSeek-V4 通信路径与 CuteDSL MoE 正确性。

**内容详情**  官方 releases API 与 release notes 原文显示：
- **v1.4.0rc0 发布信息：** tag `v1.4.0rc0`，published 2026-10-14T16:35:44Z，`prerelease=true`。上一公开 release 为 v1.3.0rc29（2026-09-29）。
- **v1.4.0rc0 Known Issues（官方原文）：**
  - 多 GPU（TP2）streaming 或 RPC executor 工作负载在跨 LLM 实例复用 MPI worker pool 时可能**在 worker 初始化阶段挂起**（H100 上观察到）；
  - **SM120/SM121** 上的 DeepSeek MLA 模型（如 RTX PRO 6000 Blackwell）在用 BF16 BMM 捕获 CUDA graphs 时可能**非法访存崩溃**；
  - SM120 上的 GPT-OSS 模型在 automatic communication 为不支持的 hidden size 选择 NCCL EP 时，attention data parallelism 可能失败；
  - 带 tied embeddings 的 Qwen3 模型（如 Qwen3-0.6B）在 TP>1 且启用 attention data parallelism 时**可能在模型构造阶段失败**；
  - 安装 FastAPI 0.142 或更新版本时，OpenTelemetry tracing 可能额外导出 FastAPI spans。
- **v1.4.0rc0 API 变更（两项 BREAKING）：**
  - **移除 KVCacheManagerV2 的 Python 后端**（`#19154`）；
  - **把 `LlmRequest` 字段迁移到 Python**（`#19484`）。
- **v1.4.0rc0 其他要点：** 新增 **GLM-5.3-Flash（glm5_next）支持**（`#19136`）；K-EXAONE-2（PyTorch backend）；Qwen-Image-Layered FP8 + Cache-DiT；静态 FP8 Cosmos3 Nano/Super 免重量化加载；Nemotron Super 3.5 VL 的 EVS 与 EPD 下 Nemotron-H 的 EVS handoff；量化 Nemotron-H MTP heads 加载；**MiniMax-M3 NVFP4 KV cache**（含 query-length buffer reuse 与 native draft KV view）；**Mooncake store pool + CLI + V2 scheduler preemption（part 1）**；**sparse KV cache 配置、tier-aware locking、offload 与 GPU metadata publication，并为 DeepSeek V4 attention 的 sparse KV offload 做准备**；C++ KVCacheManagerV2 支持 beam search；KV connector prefixes 纳入 V2 scheduler budgeting 并新增可配置的 SWA endpoint retention priority；speculative draft KV budget 按 **per-request spend** 加权；per-request speculative-decoding acceptance 在响应中上报；≤8 token batch 的 MoE routing 加速与 MegaMoE kernel descriptor 跨 forward 缓存；SM120 上 head dim 256 的 tiled FP8 context FMHA；**disaggregated KV transfer 加固**（late completion 后 retire ownership、要求 verified write coverage）；GC 关闭时释放异步 IPC payload 修复 host OOM。
- **10-15 提交（35 条，节选）：**
  - **DeepSeek-V4 TP decode 的 all-reduce 移出 NCCL**（`bf0e298f11`）；
  - 发布 **serving configuration schemas**（`7c03f0e8d1`）；
  - 修正 **CuteDSL MoE autotune 的 routing 输入**（`ed87ab0b01`）；
  - 验证 **Kimi 默认 KVCM V2 的精度与分离式传输**（`7adc880405`）；
  - 简化 **TorchSampler 策略分组**与 fused sampling kernel 快路径（`dba325dc1f`）；
  - 新增 **qwen4_exp modeling_v2 target for Qwen3.8-Flash-Next-NVFP4**（`7dc9012cad`）；
  - **indexer K-cache gather/scatter 用 64 位计算连续侧偏移**（`e30418f343`）；
  - **GLM-5.3-Flash indexer 只在 BF16 tensor cores 上对可见 pool 打分**（`172b96d051`）；
  - **Rubin merge-back 正确性修复**（`ff7ea92dfb`）；
  - **KV bounce recv-region 背压不再阻塞 executor**（`097d2adf4a`）；
  - attention fixtures 与 prims-ts 验证**迁移到 KV cache V2**（`69a455ae97`）；
  - **BREAKING：移除未使用的 `LlmRequest` 字段**（`0a594e29e4`）；
  - **CuTe MLA page-table batch bounds 修复**（`e87862cfc8`）；
  - PP rank 上 **MPI progress 由原生线程驱动**（`a77de9e3bf`）；
  - pre-Hopper 节点的 disagg 测试让 **NCCL 避开 PCIe P2P**（`306a8a7c30`）；
  - wide FP8 输出的 **CUDA-core GEMM 启动修复**（`eb745f8501`）；
  - perf agent 与 modeling agent 的 **agent flow 编排**与并行执行（`15f060c9f8`）。

**应用场景**  KV cache V2 迁移与容量规划、分离式部署（含 Mooncake 外部 KV 存储）、sparse KV offload、GLM-5.3-Flash / MiniMax-M3 / Qwen3.8-Flash-Next 部署、Blackwell（SM120/SM121）与 Rubin 平台适配、MoE 与 FP8 内核。

**研判**  **事实：** v1.4.0rc0 是本期唯一的新版本发布，且带两项 BREAKING（移除 KVCacheManagerV2 的 Python 后端、LlmRequest 字段迁到 Python）；Known Issues 明确列出 SM120/SM121 上 DeepSeek MLA + BF16 BMM CUDA graph 的非法访存、TP2 streaming/RPC 的初始化挂起等 5 项。**观点：** 与上一期（2026-10-12，KV cache manager V2 与 Python transceiver 转默认）连起来看，TensorRT-LLM 正在**连续两个阶段**推进 KV cache 路径的破坏性迁移——先切默认，再删 Python 后端。这意味着「默认切换」阶段的兼容窗口非常短：如果上一期尚未完成 v1 假设的清理，本期 rc0 会直接移除退路。建议把升级路径拆成两步（先确认无 Python backend 依赖，再升 v1.4），并把 5 项 Known Issues 逐条对照自有硬件与拓扑（尤其 SM120 + DeepSeek MLA 与 H100 + TP2 streaming 两类）。同时注意：Mooncake store pool 与 sparse KV offload 在本版为 **part 1 / 准备性改动**，不应视为已完备的生产能力。

**来源**
- TensorRT-LLM 官方提交 API（2026-10-15 UTC，35 条）：https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits?since=2026-10-15T00:00:00Z&until=2026-10-16T00:00:00Z&per_page=100
- v1.4.0rc0 release：https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc0 （published 2026-10-14T16:35:44Z）
- 官方 release notes 原文：https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/release-notes.md
- DeepSeek-V4 TP decode all-reduce 移出 NCCL：https://github.com/NVIDIA/TensorRT-LLM/commit/bf0e298f11
- 发布 serving configuration schemas：https://github.com/NVIDIA/TensorRT-LLM/commit/7c03f0e8d1
- CuteDSL MoE autotune routing 输入修复：https://github.com/NVIDIA/TensorRT-LLM/commit/ed87ab0b01
- Kimi 默认 KVCM V2 精度与分离传输验证：https://github.com/NVIDIA/TensorRT-LLM/commit/7adc880405
- GLM-5.3-Flash indexer 只在可见 pool 上打分：https://github.com/NVIDIA/TensorRT-LLM/commit/172b96d051
- KV bounce recv-region 背压不阻塞 executor：https://github.com/NVIDIA/TensorRT-LLM/commit/097d2adf4a
- BREAKING：移除未使用的 LlmRequest 字段：https://github.com/NVIDIA/TensorRT-LLM/commit/0a594e29e4
- CuTe MLA page-table batch bounds 修复：https://github.com/NVIDIA/TensorRT-LLM/commit/e87862cfc8
- 上一公开 release v1.3.0rc29（2026-09-29）：https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc29

**证据等级：** 官方一手；Known Issues 与 BREAKING 项均为官方 release notes 原文，未做推断。

---

## 6. llama.cpp：v0.7.0 发布——host 专家 GPU 缓存、LiliCorr 投机解码与默认端口变更

**主题**  llama.cpp 于 10 月 15 日发布 **v0.7.0**（自 0.x 系列以来的重要版本），核心是 host 内存 MoE 专家的 **GPU LRU 缓存**（`--moe-cache-mib`）、**LiliCorr 投机解码**、GLM5-Next 的 MTP 路径，以及 server 端口默认值从 8080 改为 9931。

**内容详情**  官方 releases API 与 release notes 原文显示：
- **v0.7.0 发布信息：** tag `v0.7.0`，published **2026-10-15T17:13:19Z**，`prerelease=false`；同日提交 `620d2225cc`「llama.cpp : bump version to 0.7.0 (#30603)」，`ggml` 同步升到 **0.27.0**。
- **v0.7.0 Highlights（官方原文）：**
  - **host 内存 MoE 专家的 GPU 缓存**（`--moe-cache-mib`）：专家从 LRU 缓存**在 GPU 上运行**，官方 release notes 声明在专家被卸载的场景下**最多约 2.2× decode 加速**（`#29887`）；专家缓存**可跨多 GPU 共享**（`#30112`）；
  - **LiliCorr 投机解码**：在 DFlash 之上叠加一个小型 draft transformer 做**候选选择**（`#30506`，对应论文 arXiv:2608.20530）——不再取 DFlash 在每个 slot 位置认定的最佳 token，而是为每个位置 top-k 候选计算 `in`/`out` 向量，选 `in` 向量与前一位已选候选 `out` 向量**余弦相似度最高**者；
  - **GLM5-Next：MTP 投机解码 + 优化的稀疏注意力路径**（`#29928`）；
  - 新模型：K2 Horizon（dense 与 MoVA 两种变体，含转换、tokenizer、chat template、`-sm tensor` 切分）、Prism Bonsai 2 27B（ternary）、EmbeddingGemma-2（text+vision+audio）、LiquidAI d1-3B / d1-omni-600M、vLLM decision 2 / d3、pplx-decider、MiniCPM-V 4.7、Nemotron Super 3.5 VL / Nemotron Bolt、maion-coder；
  - server 围绕 **`common_chat_session`** 重构 chat API（`feed()` / `finish()`），格式化输出改为 **parts 数组**而非单字符串；slot 与 prompt-cache 恢复行为大幅改善；web UI 新增完整 models manager；
  - **默认监听端口从 8080 改为 9931**（Docker 镜像容器内仍为 8080）（`#30159`）；
  - RPC server 支持 `-sm tensor`（张量并行 RPC，RPC 协议 v8）；
  - `include/llama.h`：新增 `moe_cache_size` context 参数；新增 `LLAMA_VOCAB_TYPE_PLAMO3`；`llama_perf_context_data` 上报**推理期图重分配**（与上一期 2026-10-13 档案记录的 `ggml_backend_sched_realloc_callback` 一脉相承）。
- **`--moe-cache-mib` 实测细节（PR `#29887` 原文）：** MUL_MAT_ID 在 host 专家上以 **LRU 缓存**在 GPU 执行，**只上传 miss**；**仅适用于小 batch（≤32 tokens）**，大 batch 走常规 offload 路径；专家布局不同的层使用**独立 bank**；实验建议取值**约为专家总大小的 10%**；测试模型为 Qwen3.8-Flash-Next Q4_0（93.7 GiB，其中专家 65.4 GiB），SPEED-Bench `qualitative`（11 类 × 3 样本，`--osl 512`，temp 0，无 MTP），`llama-server -fa on -c 32768 -b 2048 -ub 2048 -t 16`，CPU 位于 GPU 的 NUMA 节点（EPYC 7742，16 核），PCIe 4.0 x16；基线为 master 的默认 `--fit`，启用后 `--fit` 会把缓存计入并**保留更少的整层专家在 GPU 上**。
- **10-15 提交（36 条，节选）：**
  - **SWA reuse threshold 差二（off-by-two）修复**（`61ad80bf93`）；
  - server **把 draft state 存入 slot 文件**（`ea736c1d5c`）；**优先空 slot** 并保留被 child task 覆盖的状态（`b0d239164b`）；**只把可恢复的 slot 状态写入 prompt cache**（`2639a3b4e0`）；
  - **plain KV cache 遵守 `PARTIAL_ONLY`**（`e467e5b5e5`）；
  - `llama_prefetch()` 拆分为 **merge 与 prefetch 两步**（`68ac765726`）；
  - graph 在**无 embeddings 时保持非空**（`1ddc262558`）；
  - **Metal 新增 tensor API flash attention 的 BF16 变体**（`6e66cb5d4e`）；
  - **opencl** 修复 adreno noshuffle gemv kernel 的**越界尾读**（`9166ceaf16`）、flash attn softcap 在 mask **之前**应用（`77c1f39929`）；
  - **Vulkan** 在 AMD 上一次性拷贝 strided f16 KV，让读取**分散到内存通道**（`733f57a999`）；cm2 mulmat 使用 **L2 友好的 workgroup 调度**（`3e390af18d`）；修复并发首次设备初始化（`47f5a64ae1`）；
  - **SYCL** 用 DLE+oneDNN 包替换 oneAPI 包（`e36df60470`）；
  - CPU 新增 **F16 SWIGLU_OAI 参考实现与校验**（`5a638aedb7`）；`ggml-cpu` 优化 `pad_f32`（`23f83e056d`）；
  - **unicode 拒绝过长 UTF-8 与越界码点**（`2ce20d22fd`）；
  - jinja：上下文拷贝构造性能（`08698c81ed`）、`tojson` 按 json.dumps 写 nan/inf（`82e1e2fd84`）、loop body 变量作用域修复（`2130ca8476`）；
  - 模型：Muse Spark Open（text/vision/audio，`ba3c05f71d`）、Nemotron Super 3.5 VL 与 Nemotron Bolt 74B A9B（`cd0a5b6a18`）、maion-coder（`3e4dfa8f39`）、Mellum 从 GGUF 读取 MoE gating 参数并统一 RoPE（`a6b91d8c5e`）；
  - mtmd：新增 `mtmd_audio_stream` 并支持 voxtral-realtime（`2851b3bcbf`）；lightonocr 原生分辨率默认（`2f729cfa4c`）。

**应用场景**  本地/端侧 MoE 大模型在显存受限下的解码加速、专家卸载部署（含多 GPU 共享专家缓存）、本地投机解码（DFlash + LiliCorr）、GLM5-Next 与 Nemotron 系列本地推理、Metal/Vulkan/OpenCL/SYCL 异构后端、llama-server 的 slot 与 prompt cache 恢复。

**研判**  **事实：** v0.7.0 是本期唯一的**正式版本发布**（非 prerelease），含两项实质推理优化（`--moe-cache-mib`、LiliCorr）与一项**破坏性默认值变更**（监听端口 8080 → 9931）；`--moe-cache-mib` 的适用边界被 PR 明确限定为**小 batch（≤32 tokens）**，且 2.2× 是在专家已卸载的特定配置下测得。**观点：** 三点——(1) `--moe-cache-mib` 是本期对**端侧/单机 MoE 部署**最实用的改动：它把「专家放在 host 内存」从纯带宽瓶颈改写为「GPU 端 LRU 命中率」问题，代价是显存中要留出缓存空间并减少整层专家驻留，**在 `--fit` 自动分配下会改变既有显存布局**，升级后应重新确认 `--fit` 结果；(2) LiliCorr 的定位是「用额外计算换更高接受率」，属于**接受率-算力**权衡的又一次尝试，官方措辞为「hopes it pays off」，说明收益尚需自行验证，不宜预设；(3) 端口默认值变更属典型**静默破坏**——不报错、只是连不上，凡是脚本硬编码 8080 的部署都会受影响，应纳入升级检查项。此外 `2639a3b4e0`（只把可恢复状态写入 prompt cache）与 `e467e5b5e5`（plain KV cache 遵守 `PARTIAL_ONLY`）延续了上一期「静默降级/静默错误」的修复主线。

**来源**
- llama.cpp 官方 releases：https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0 （published 2026-10-15T17:13:19Z）
- llama.cpp 官方提交 API（2026-10-15 UTC，36 条）：https://api.github.com/repos/ggml-org/llama.cpp/commits?since=2026-10-15T00:00:00Z&until=2026-10-16T00:00:00Z&per_page=100
- 版本 bump 到 0.7.0：https://github.com/ggml-org/llama.cpp/commit/620d2225cc
- host 专家 GPU 缓存（`--moe-cache-mib`）：https://github.com/ggml-org/llama.cpp/pull/29887 （PR 标题：`llama : add a GPU cache for MoE experts kept in host memory`，merged 2026-10-07，随 v0.7.0 首次进入正式 release）
- 专家缓存跨多 GPU 共享：https://github.com/ggml-org/llama.cpp/pull/30112
- LiliCorr 投机解码：https://github.com/ggml-org/llama.cpp/commit/843e7dcc15 （PR #30506）
- GLM5-Next MTP：https://github.com/ggml-org/llama.cpp/pull/29928
- 默认端口改 9931：https://github.com/ggml-org/llama.cpp/pull/30159
- SWA reuse threshold off-by-two：https://github.com/ggml-org/llama.cpp/commit/61ad80bf93
- 只把可恢复 slot 状态写入 prompt cache：https://github.com/ggml-org/llama.cpp/commit/2639a3b4e0
- plain KV cache 遵守 PARTIAL_ONLY：https://github.com/ggml-org/llama.cpp/commit/e467e5b5e5
- Metal BF16 tensor API flash attention：https://github.com/ggml-org/llama.cpp/commit/6e66cb5d4e
- Vulkan 在 AMD 上一次性拷贝 strided f16 KV：https://github.com/ggml-org/llama.cpp/commit/733f57a999

**证据等级：** 官方一手；2.2× 为官方 release notes / PR 在特定配置下的声明，PR 同时限定适用 batch 上限，不可外推。

---

## 7. 当前会话已形成的累计内容结论

1. **KV Cache 已从「容量问题」推进到「分布式资源的生命周期与并发正确性」**：前期已记录 vLLM KV Connector、SGLang HiCache/分层路径、TensorRT-LLM verified write coverage 与 ownership；本期新增 vLLM 的 lookup 等待上界、`mark_miss()` 探测状态机、tiering 控制面移出调度线程、pin 粒度收敛、peer teardown 必做 NIXL agent 移除，以及 UMBP 契约第一片；SGLang 侧新增 `page_unified` 页布局与**不含写入者身份**的统一 L3 key。
2. **P/D 解耦的性能瓶颈被官方明确为「协调串行化 + 关键路径 CPU 工作」**：本期 vLLM 官方博客用 push-based NIXL connector 给出机制与实测（TTFT 最高 3×、TPOT 最高 30%，Qwen3-32B / 1P1D / P5en + EFA）；对应提交侧同步出现 Mooncake 每 peer 只发一次 KV layout、KV bounce 背压不阻塞 executor、disagg gen-init fence 轮询等改动。
3. **投机解码的竞争焦点转向「验证预算」与「能力收敛」**：本期出现三项方向性变化——**移除 Medusa 与 MLP speculator**（能力收敛到 EAGLE/MTP/DFlash/DSpark）、**AV 扩展到无 confidence head 的 drafter**（在线拟合估计器）、**DFlash/DSpark draft slots 不再占用调度器 token budget**；llama.cpp 侧同时引入 LiliCorr 与 GLM5-Next MTP。共识是：投机解码的收益由「draft 成本 × 可裁剪比例」决定，而非单看接受率。
4. **MoE 推理正沿「专家驻留策略」而非单纯 kernel 优化演进**：llama.cpp 用 host 专家的 GPU LRU 缓存（`--moe-cache-mib`，≤32 token batch，最多约 2.2× decode）把专家卸载从带宽问题改写为命中率问题；vLLM 把三个 DeepGEMM MegaMoE 实现统一为单一特征矩阵层并新增 NVFP4；SGLang 为 Nemotron-H 增加 EPLB 支持。三者共同点是**专家放置与路由的可配置性**成为一等公民。
5. **量化结论必须绑定硬件与实现路径**：NVFP4/MXFP4/FP8 在各引擎对应不同 kernel、KV 格式与设备路径；本期新增 vLLM NIXL **拒绝异构 TP head split 下的 NVFP4 KV cache**、MoRI FP4 dispatch for mxfp4 w4a4 MoE、preshuffled FP4 GEMM 按逻辑 K 探测、SGLang gfx950 融合 DSA indexer（block-FP8）、TensorRT-LLM MiniMax-M3 NVFP4 KV cache 与 SM120 head dim 256 tiled FP8 context FMHA。**「同一 dtype 在不同引擎/设备上不可互推结论」已被反复验证。**
6. **混合缓存架构的正确性风险落在状态完整性上**：本期 SGLang 的 tail-only SWA 备份揭示「**备份写量 > 有读者可恢复的页数**」这一写放大问题；llama.cpp 的「只把可恢复的 slot 状态写入 prompt cache」与「plain KV cache 遵守 PARTIAL_ONLY」同属一类。缓存命中率与状态正确性仍是两个独立问题。
7. **默认值与静默降级持续是主要风险面**：本期新增两项——llama.cpp **默认监听端口 8080 → 9931**（不报错、只连不上）、TensorRT-LLM v1.4.0rc0 **移除 KVCacheManagerV2 的 Python 后端**（上一期刚把 V2 转默认，本期即删除退路）。加上前期的 vLLM spec decode draft config 未生效、llama.cpp flash attention 静默回退 CPU，这类变更难以通过端到端精度测试发现。

---

## 8. 今日可执行清单

1. **P/D 集群按官方脚本复测 push vs pull**：使用官方 `serve_pd.sh` / `run_sweep.sh` / `plot_push_vs_pull.py`，在自有 EFA/RDMA 拓扑上以相同 `QPS_LIST`、`INPUT_LENS`、`OUTPUT_LENS` 对比 `NixlConnector` 与 `NixlPushConnector`；确认 decode 侧 `--no-enable-prefix-caching` 是否适用于自身工作负载（官方以此隔离 P→D 传输），并核对 `PREFILL_TP_SIZE` 与 prefill server 实际 TP 一致。
2. **把 KV offload 的运维正确性纳入监控**：接入 vLLM 新增的 `PROMOTION_LATENCY` 直方图；为 P2P lookup 等待上界、`mark_miss()` 探测状态、peer teardown 时的 NIXL agent 移除建立告警；确认 tiering 控制面已不在调度线程上执行。
3. **投机解码栈做一次能力清点**：确认是否仍在配置中使用 `method="medusa"` 或 `"mlp_speculator"`（v1.4 起被 vLLM 校验拒绝）；对无 confidence head 的 DFlash/MTP drafter 试用 `enable_adaptive_verification: true`，并**按 drafter 类型分别**记录接受率与吞吐（官方明确 MTP/EAGLE 收益低于 DFlash/DSpark）；确认 DFlash/DSpark 的 draft slots 已不再计入调度器 token budget。
4. **TensorRT-LLM 升级拆两步并逐条对照 Known Issues**：先确认无 KVCacheManagerV2 Python 后端依赖，再升 v1.4.0rc0；重点核对 SM120/SM121 + DeepSeek MLA + BF16 BMM CUDA graph、H100 + TP2 streaming/RPC 的 MPI worker pool 复用、SM120 GPT-OSS 的 ADP、Qwen3 tied embeddings + TP>1 四类问题；注意 Mooncake store pool 与 sparse KV offload 在本版为 part 1 / 准备性改动。
5. **llama.cpp 升级到 v0.7.0 前处理两件事**：把硬编码 8080 的脚本/探针改为 9931（Docker 镜像容器内仍为 8080）；重新确认 `--fit` 在启用 `--moe-cache-mib` 后的显存布局（缓存会被计入并减少整层专家驻留），并按官方建议从「约专家总大小 10%」起步，注意仅小 batch（≤32 tokens）走缓存路径。
6. **为 MoE 与稀疏路径建立形状/边界回归**：覆盖 ≤32 token 小 batch 与大批次的专家路径切换、重复路由 id、SWA 尾部窗口、`PARTIAL_ONLY` KV cache、sliding-window block table 的 null head，以及 DeepSeek V4.1 decoder replay 的 routing rows。
7. **AMD/异构路径单独记录设备型号与 dtype**：本期 gfx1250、gfx950、ROCm 10.0/10.1 并行发布、AITER 各 kernel（Gluon KDA、MegaMoE fixed-slot、topk_gating）与 NVIDIA 路径不可互相外推；同时注意 SGLang `page_unified` 当前**从 CLI 不可达**，不要据此规划上线。

---

## 9. 社区与视频资料

- **HN Algolia 公开 API 本期可访问。** 在 2026-10-15 窗口内按 “LLM inference”“vLLM”“SGLang”“llama.cpp”“KV cache”“speculative decoding”“TensorRT-LLM” 检索，命中条目多为个人项目、Show HN 或非推理主题内容，与本期主题直接相关的仅有下列几条，**除注明为官方来源者外均标注为二手线索/未核验**：
  - **vLLM 官方博客 push connector 的社区扩散**（可核验为官方来源）：HN 2026-10-15 18:41:41 UTC 提交，1 point / 0 comments，链接 https://vllm.ai/blog/2026-10-14-nixl-push-connector —— 与本档案第 1 节同一来源；
  - “Local and Distributed Large LLM Inference with MLX and RDMA on Mac Studio”（Apple Machine Learning Research 官方研究页）：HN 2026-10-15 17:12:48 UTC，4 points / 1 comment，https://machinelearning.apple.com/research/mlx-distributed-inference —— 来源为 Apple 官方研究页，**但本档案未核验其正文与性能数字**，故仅列为线索；
  - “Encrypted KV Cache Offloading for Confidential Computing”（confidential.ai 博客）：HN 2026-10-15 14:05:57 UTC，1 point / 0 comments，https://confidential.ai/blog/kv-offload-under-cc —— **二手线索，未核验**，主题与 KV offload 在机密计算下的加密传输相关；
  - “Same LLM, same input tokens, different answers on a MacBook and a Jetson”（kernwerk.org 博客）：HN 2026-10-15 20:09:27 UTC，1 point / 0 comments，https://kernwerk.org/blog/same-model-different-board/ —— **二手线索，未核验**，主题为跨设备数值一致性，与 batch invariance 类问题方向一致；
  - “If llama.cpp enables MoE to activate multiple experts”（openai-hub.com news）：HN 2026-10-15 21:08:53 UTC，1 point / 0 comments —— **二手转述，未核验**，未找到对应一手来源；
  - “A Graphics Engineer's Guide to AI Performance”（个人 Substack）：HN 2026-10-15 14:03:29 UTC，1 point / 0 comments —— 个人博客，非官方一手。
- **Reddit 未纳入本期采集范围**（往期记录其 API 受限），本期不据 Reddit 内容形成结论。
- **视频资料：** 本期未核验到可确认发布日期与内容的官方视频，故不列入。
- **技能检索：** 本期通过 WorkBuddy 技能检索（关键词：“LLM inference optimization”“推理引擎热点追踪”“github trending hotspot archive”“KV cache optimization”“AI news daily digest”）返回候选包括「AI 热点日报」（ai-hotspot-daily，面向科技媒体编辑规范的当日 AI 热点日报）、「GitHub AI趋势追踪」（github-ai-trends）、「GitHub热门项目」（github-trending-cn）、「开源雷达 · GitHub 热门榜」（github-open-source-radar）与「LLM推理加速方案制定与优化配置」（llm-inference-acceleration，面向输出推理加速方案与配置建议）。上述候选与本任务所需的「**官方提交/发布 API 按日期窗口采集 + 证据分级热点归档 + hotspot 目录与远端同步**」均不匹配（前者偏媒体式热点聚合，后者偏方案输出），故**未安装任何外部技能**；本期采集与归档由内置的 HTTP 抓取与本地文件/Git 能力完成。

---

**归档状态：** 已按 `hotspot/2026-10/` 月度目录归档。
**证据边界：** 事实与观点已分开；官方提交/发布中的性能标题、prerelease 版本、Known Issues、part 1 与「库层未接线」类改动均按原文范围记录，不推导为普遍结论；社区条目一律标注为二手线索且未核验；本期存在 2026-10-13、2026-10-14 两日归档空档，未补写。
