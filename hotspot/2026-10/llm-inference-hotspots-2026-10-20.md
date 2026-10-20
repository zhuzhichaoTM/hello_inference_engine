# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-20（北京时间）
- **关注窗口：** 2026-10-19 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-19 08:00 – 2026-10-20 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-19 UTC 窗口，分别返回 **83 / 80 / 27 / 27** 条，均 HTTP 200）、四家官方 releases API、**SGLang v0.5.22 发布说明全文**（110,527 字符）、**vLLM 官方博客窗口内新文《Achieving near-zero-downtime scaling in Elastic Expert Parallelism》全文**，以及 **37 个 PR 页面正文**（HTML 直抓，不消耗 API 配额）。**窗口特征：** 2026-10-19 为**周一**，四家提交量均明显高于上期（周日），属正常工作日节律。vLLM 的 83 条中 **Bugfix 30 条、ROCm 10 条、CI 7 条**，是 v0.32.0 发版后第一个工作日的「补丁 + CI 收敛」形态；SGLang 的 80 条中 **Fix 34 条、Perf 15 条**，且**绝大多数发生在 v0.5.22 发版（02:55Z）之后**（见第 1 节）。HN Algolia 公开 API 可访问（HTTP 200），窗口内 `sglang`、`llama.cpp`、`KV cache` **均 0 命中**，`speculative decoding` 2 条、`vllm` 32 条（其中绝大多数是 "LLM" 的模糊匹配）。Reddit、YouTube、X 未纳入采集范围；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** **SGLang v0.5.22 正式发布**（2026-10-19T02:55:19Z，950 PRs / 245 contributors），这是本期头号事件。vLLM 正式版仍为 **v0.32.0**（2026-10-18），窗口内无新版本；TensorRT-LLM 仍为 **v1.4.0rc0（prerelease，2026-10-14）**；llama.cpp 正式版仍为 **v0.7.0**，窗口内滚动 nightly 共 **15 个标签（b11722–b11747）**。
- **本期方法论沿用与新增：** 继续用 GitHub compare API 判断 commit 祖先关系来核验「某修复是否真的进版」。**本期据此得到一条与前一期互补的结论：v0.5.22 的 tag 打在 02:55Z，当日 80 条提交中的绝大部分（含两个正确性修复）都不在版内**——即「发版当天合入 main」不等于「在发版内」，与上期 vLLM `#61989` 的结论方向相反但机理相同。

---

## 1. SGLang v0.5.22 正式发布（窗口内唯一正式版本，本期头号事件）

**主题**  SGLang 发布 **v0.5.22** 稳定版（`published_at` 2026-10-19T02:55:19Z，非 prerelease），规模为 **950 个 PR、245 位贡献者**，距上一版 v0.5.21（2026-10-02）17 天。

**内容详情**  官方 "What's new" 原文条目（性能百分比均为官方口径）：

- **DeepSeek-V4.1：H200 上每 GPU 吞吐最多 +2.3×**；新增 AMD **MI350X / MI355X** 支持，**DSpark 带来最多 +1.9×**（`#41251`、`#41308`）。
- **Qwen3.8-Flash-Next 在 B200 解码快 12%**；**GLM-5.3-Flash 配 DFlash 在 B300 快 10%**（`#42175`、`#42489`）。**MiMo-V2.6-Pro 配 DFlash 在 B300 解码吞吐 1.85×**（`#41886`）。
- **模型加载快 10×**：新的 **InstantTensor** 加载器（`#40453`）。
- **PD 解耦在长 prompt 负载下吞吐最多 +27%、p95 TTFT 减半**（`#40703`）——**注意这条的代价，见下**。
- **HiSparse 支持 EAGLE 与 MTP**，用于「KV cache 放在 CPU 内存」的长上下文服务（`#37771`）。
- **GLM-5.2 / GLM-5.3 可以把 decode context parallelism 与 MTP 组合**，GLM-5.3 上**解码最多快 2.1×**（`#43735`）。
- **Apple Silicon Mac 可用标准 PyTorch 后端运行**（`#36780`）。
- **官方标注的 "Important fix"：** 混合线性模型在**同时使用投机解码与结构化输出**时，可能**从错误的 mamba 状态恢复**（`#43029`，由 @Wirick 与 Baseten 发现）。

与推理优化直接相关、但未进 "What's new" 的发布说明条目（按章节摘录）：

- **投机解码（12 项）：** HiSparse 多步 KV 交换式投机解码（`#37771`）、top-p mask capture 并为 DFlash/DSpark 落地（`#34201`）、PP × 投机解码下的混合递归状态提交与 micro-batch 配对（`#40001`）、K3 辅助输出跨 decode CUDA graph 尺寸复用（`#40159`）、PD decode 的**首个 EAGLE draft step 给出 proposal distribution**（`#41783`）、DFlash 分组卷积 tap 限制在请求自己的 block 内以防 NaN/Inf **跨请求泄漏**（`#41736`）、DFlash2 全 NaN 评分行下的贪心选择器越界（`#41737`）。
- **Attention 后端（31 项）：** **滑动窗口 extend attention 跳过完全被 mask 的 KV tile，8K token 时 kernel 快最多 39%**（`#39769`）；**MiMo 在 SM100 默认 FA4（8×B300 上 DFlash 解码 1.85×）**（`#41886`）；**QSA decode indexer 的工作量按上下文长度上界（bs64 decode step 28.75 → 26.54 ms，Qwen3.8-Flash-Next on 4×B200）**（`#43463`）；DSA 的 **k-pool indexer 取消每步 CPU seq_lens 同步**（`#41987`）；DSA indexer 支持 **KV cache 分片**（`#40925`）；DSv4.1 prefill indexer 改为 **raw selection、行对调度 id、无 host sync**（`#41659`）。
- **MoE 与 EP（20 项）：** **MegaMoE 的 DeepGEMM 里把 shared experts 与 routed experts 融合**（`#39313`）；**MoE prefill 用 FlashInfer A2A 替代 AG+RS**（`#39818`）；**EPLB 在 KV-cache sizing 之前预热默认组 NCCL P2P 传输**（`#42026`）；**GLM-5.3-Flash 在 prefill 尺寸上 autotune MoE kernel**（`#42693`）；MegaMoE 启动期 autotuning（`#43789`）；GLM-5.3 支持 DeepEP v2（`#43432`）。
- **量化（13 项）：** **把 FP8 KV-cache 量化融进 prefix-valid commit kernel（B200 上该 kernel 快 2.99×）**（`#40785`）；**GLM-5.3-Flash 的 Q8KV8 FP8 稀疏 prefill（BF16 KV cache），H200 上每 DSA 层比 tilelang 快 1.53–1.72×**（`#41790`）；MLA KV/Q 准备改用原生 SATFINITE FP8 转换（`#42659`）；CUDA 上 TileLang FP8 KV cache（`#42957`）。
- **并行与解耦（65 项）：** **`#40703`（opt-in prefill-complete decode KV 分配）**；**PD 异构 TP 的混合 SWA KV 传输**（`#40909`）；**DCP：DSA（GLM-5.2/5.3）的 decode context parallelism 配 MTP（GLM-5.3、8×B300 上，加 MTP 使 DCP2 快 1.8×、DCP8 快 2.1×）**（`#43735`）；**GLM-5.3-Flash 的 prefill CP 在层边界实现（4×B300 上 P50 TTFT 比 TP4 低 2%–9.5%）**（`#42044`）；`--attn-dp-size` 取代 `--enable-dp-attention`（`#41818`）；PD 的 **per-rank DP attention 不均衡指标**（`#42755`、`#43180`）。
- **HiCache 与 Radix Cache（约 90 项）：** **DSA index-K 省略在 HiCache 与 PD 下放开（GLM-5.2-NVFP4 PD：prefill KV 容量 +11.6%、decode +13.7%）**（`#41769`）；**新增 SeaweedFS L3 后端（零拷贝路径使单张 A10 上 Qwen3-4B 的 L3 TTFT 降到 1/3.5–1/4.7）**（`#42399`）；**staged write-back 支持 page-unified 布局**（`#39606`）；**HiCache 保留全部 MTP KV 与递归状态**（`#40223`）；`ChunkCache` 删除（`#43201`）。
- **Kernel 库（12 项）：** Mamba2 SSD kernel 的 Triton autotune（`#39130`）；SM90 用原生 block32 DeepGEMM（`#42604`）；Blackwell 上 JIT 激活开 fast-math（`#43236`）。
- **破坏性变更：** **要求 Python ≥ 3.11**（`#43021`；在 3.10 上 `pip install sglang` 不报错，而是**静默解析到最后一个支持 3.10 的版本**）；`--enable-dp-attention` 废弃、`--dp-size` 现在只计模型副本；**`--disaggregation-decode-host-receive-threshold` 编码翻转（`1` 为关闭 host receive 且是新默认，`0` 为总是 host receive）**；**`ChunkCache` 移除**；**`tc_piecewise` prefill CUDA graph 后端移除**（传 `--cuda-graph-backend-prefill tc_piecewise` 现在直接报错）；`SGLANG_AITER_MLA_DCP_DECODE_BACKEND` 改名 `SGLANG_AITER_MLA_DECODE_BACKEND` **且默认值由 `gluon` 改为 `asm`**；`SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW` → `SGLANG_OPT_RELEASE_PREFILL_SWA`；`SGLANG_DSPARK_FAST_SAMPLING` 等三个 DSpark 环境变量废弃且**不再有任何作用**；`--enable-layernorm-sp` 与流水并行同时使用会被拒绝；`ModelRunner.tp_group` 移除。

**关于 `#40703` 的代价（本期最值得读的一条细节，取自 PR 正文而非发布说明）：** 该 PR 是 **opt-in**（`--disaggregation-decode-allocation-policy prefill_complete`，默认仍是 `early`），官方给出的 GLM-5.2-NVFP4 对比（真 MTP3、12 张 Rubin、2 个 prefill 副本 + 1 个 decode 副本、各自 TP4/DP4/EP4、FP8 KV + Mooncake、**decode KV 被刻意压到 163,840 tokens/rank 以制造容量压力**，两次运行取均值）：

| 指标（混合负载） | early（两次） | prefill_complete（两次） | 均值变化 |
|---|---|---|---|
| 系统 output tokens/s | 458.88 / 435.98 | 616.39 / 518.39 | **+26.81%** |
| TTFT p95（s） | 71.22 / 79.97 | 36.03 / 41.27 | **−48.87%** |
| 平均 TPOT（ms） | 4.108 / 3.869 | 4.204 / 3.943 | **+2.13%** |
| **ITL p99（ms）** | 34.40 / 12.03 | 46.40 / 43.10 | **+92.77%** |

PR 正文自述：吞吐/TTFT 的收益**以更差的 ITL 尾部为代价**，因为延迟分配会**更久地保留源端 KV**，并**去掉 chunked prefill 与 KV 传输之间的重叠**。支持的配置也有限：目前只覆盖 Mooncake（含 TCP）、Python bootstrap registry、每 rank 的 attention TP1/CP1/PP1（含 attention DP）；prefill 侧 HiCache 只支持无存储的 L2 `write_back`；**decode radix/HiCache、staging、运行时角色切换、PD 多路复用、encoder 解耦、decoupled speculation、HiSparse、非 DSA 辅助状态池均被排除**。

**应用场景**  长 prompt 与短 prompt 混跑的 PD 部署（`#40703`）、DSA 稀疏注意力的长上下文服务（`#41769` 的 index-K 省略）、跨实例/跨节点的 KV 复用（SeaweedFS L3）、解码阶段的上下文并行（`#43735` 的 DCP + MTP）、显存不足的边缘设备（v0.5.22 之外的 paged-experts，见第 3 节）。

**研判**
- **事实：** v0.5.22 是一个「DSv4.1 + DSA 稀疏 + PD 解耦」的版本，性能条目几乎全部集中在 **DeepSeek-V4.x / GLM-5.x / Qwen3.8-Flash-Next 这三类模型 × NVIDIA SM100/SM103 与 AMD MI35x** 上，通用性收益（如 InstantTensor 10× 加载）反而更少。
- **事实（已用 commit 祖先关系核验）：** **v0.5.22 的 tag 打在 2026-10-19T02:55:19Z，当日 80 条提交中绝大部分不在版内。** 核验方式：取 PR 的 `merge_commit_sha` 与 tag `v0.5.22` 做 `compare/v0.5.22...{sha}`。在版内的：`#40703`（`5b5d7212`，`behind, ahead_by=0`）、`#42399`（`685d4d45`，`behind, ahead_by=0`）、`#43029`（`0acd93c3`，`behind, ahead_by=0`）、`#41769`（`e0c957e2`）、`#43735`（`8bf145d1`）。**不在版内的：`#42557`（paged-experts，`diverged, ahead_by=376`，merged 15:10Z）、`#44904`（MoE 共享专家路由修复，`ahead_by=402`，merged 23:56Z）、`#45075`（PD KV 布局校验，`ahead_by=346`，merged 04:16Z）、`#45223`（SWA host sync 消除，`ahead_by=388`，merged 20:34Z）**——四个的合并时间**都在 02:55Z 之后**，只能等 v0.5.23。
- **观点：** 三点。(1) **`#40703` 是「用 ITL 尾部换吞吐与 TTFT」的开关，不是无脑优化**：ITL p99 从 12.03 ms 涨到 43.10 ms（第二次运行）意味着交互式逐 token 输出的抖动明显变大，**对话式/流式产品开这个开关要单独看 p99 ITL，而批处理/离线摘要类负载才是它的主场**。(2) **`#41769` 揭示的模式值得记住**：一个 `and not memory_config.enable_hierarchical_cache and ... get_disagg().disaggregation_mode == "null"` 的**保守门控**，把 57/78 层的 index-K 显存一直占着——**「为了安全而禁用」的默认值，在真实部署里会变成持续的成本**，这类门控应定期重审。(3) **`--disaggregation-decode-host-receive-threshold` 的语义翻转是本期最危险的升级项**：此前用 `0` 来关闭 host receive 的配置，升级后 `0` 变成「总是 host receive」，**不会报错，只会变慢或吃满主机内存**。

**来源**
- Release：https://github.com/sgl-project/sglang/releases/tag/v0.5.22 （2026-10-19T02:55:19Z，发布说明 110,527 字符）
- `#40703` https://github.com/sgl-project/sglang/pull/40703 （merged **2026-10-03T13:03:09Z**，merge commit `5b5d72123935dc2a3fb341790796ec1a987d31ce`——**注意：该 PR 合并在 10-03，不在本期提交窗口内，本期记录的是它作为 v0.5.22 的 "What's new" 条目**）
- `#41769` https://github.com/sgl-project/sglang/pull/41769 ｜ `#42399` https://github.com/sgl-project/sglang/pull/42399 ｜ `#43735` https://github.com/sgl-project/sglang/pull/43735 ｜ `#43029` https://github.com/sgl-project/sglang/pull/43029
- 未入版核验：`GET /repos/sgl-project/sglang/compare/v0.5.22...8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043`（`#42557`）等四条

**证据等级：** 官方一手（release 页面 + GitHub compare API + PR 正文）。所有吞吐/TTFT/ITL 百分比均为**各 PR 自测口径**，非统一基准；「不在版内」为本次基于 commit 祖先关系的**独立核验**，非官方声明。

---

## 2. vLLM：官方博客《Elastic EP 近零停机扩缩容》+ 10-19 提交（83 条）

### 2.1 官方博客：把「扩缩容停机」从 197 秒压到 0.82 秒（窗口内唯一新文）

**主题**  vLLM 官方博客 2026-10-19 发布 **《Achieving near-zero-downtime scaling in Elastic Expert Parallelism》**（作者 Itay Alroy，NVIDIA，标注 12 min read，标签 `#elastic-ep` `#expert-parallelism` `#moe` `#nixl` `#cuda-graphs`）。

**内容详情**  核心事实链（全部取自博客正文）：

- **基线之痛：** 传统上改变 vLLM 服务的 GPU 数必须增删整个模型副本或重启引擎。把 **DeepSeek V4 Pro 从 16 张 GB200 扩到 20 张**若靠重启，**停服 197 秒**。
- **上一篇文章的 Elastic EP：** 已能不停机地增删 GPU（DEP：data-parallel attention + expert-parallel MoE），但**重配置期间服务仍然停摆**（初始化新 worker、准备扩大的通信组、加载权重、重复 kernel warmup 与 CUDA graph capture）。实测**扩容停 100 秒、缩容停 56 秒**。
- **本篇的三段式改造：**
  1. **把重配置拆成「后台准备 + 短提交」两段。** 已有 worker 用独立后台线程准备目标通信、传权重、与加入者同步 autotuning 结果，**同时继续用当前通信组和 CUDA graph 服务请求**；加入者并行完成初始化、收权重、准备通信与 GPU 执行。提交阶段只**短暂暂停调度器**、激活已准备好的拓扑、恢复服务。
  2. **NIXL EP：就地扩缩的 EP 通信库。** 此前 vLLM 的 EP 后端**成员固定**，换成员就要换实例；新旧实例并存会**双份通信缓冲**（内存注册、IPC handle、RDMA QP），且切换通信器会使已捕获的 CUDA graph 失效、必须全部重捕。NIXL EP **就地增删 rank，复用同一个 NIXL agent、缓冲与连接**，新 peer 在已有 worker 继续服务时被连接，且**保留已捕获的 EP dispatch/combine 条目**，使保留 worker 可复用原 CUDA graph。
  3. **加入者独立完成 graph capture**，因此**已有 worker 的 warmup 与 graph capture 变成「不需要」**。
- **关键中间结论（很值得记）：** 即使做了异步准备，**如果在提交阶段仍要 warmup 与重捕 graph，停服仍达 58 秒（扩容）/ 44 秒（缩容）**——也就是说，**收益的绝大部分来自「保住已有 worker 的 CUDA graph」，而不是来自「准备放到后台」**。
- **最终数字（DeepSeek V4 Pro on GB200，TP=1，DP/EP 16↔20，权重预置于 Lustre，kernel 编译缓存已预热）：**

| 指标 | A：阻塞式 | B：异步准备 | C：异步准备 + graph 复用 |
|---|---|---|---|
| 16→20 停机 | 99.62 s | 58.04 s | **0.82 s** |
| 20→16 停机 | 56.25 s | 44.07 s | **1.96 s** |
| 16→20 总时长 | 99.72 s | 97.27 s | 92.34 s |
| 20→16 总时长 | 56.21 s | 51.46 s | **9.32 s** |

  配置 C 仅 NIXL EP 支持；三配置均通过数值正确性检查（含对每个加入 worker 的检查）。
- **SLO 场景：** 16 卡起服，请求速率从约 8 升到 16 req/s，回放同一份 **11,391 请求**的场景，对比「静态 16 卡」「重启到 20 卡」「Elastic EP 扩到 20 卡」三者，SLO 为**滚动 p95 TTFT < 5 s**。Elastic EP **全程低于 SLO**，准备期性能与静态 16 卡持平，扩容后与「重启后恢复的 20 卡」持平，**11,391 个请求全部完成**。
- **可用性：** 后台准备、CUDA graph 复用、加入者独立 graph capture **均已在最新 vLLM 与 NIXL 中可用**；示例为 vLLM Elastic EP 测试里的 `cuda_graphs_heavy_nixl_ep`（2 卡扩到 4 卡再缩回，边扩边发请求）。
- **后续路线：** Dynamo Snapshot 恢复热 worker、NIXL EP 的 SOTA EP 通信、接入 Dynamo 与 llm-d 编排。

**应用场景**  MoE 大模型（DeepSeek V4 Pro 级）在弹性云上的按需扩缩、按流量曲线调 GPU 数量、把「扩缩容」从运维事件降级为常态操作。

**研判**
- **事实：** 这篇博客最有价值的信息不是 0.82 秒，而是**归因**——把 197 s → 99.62 s 是「不重启」，99.62 s → 58.04 s 是「准备放后台」，58.04 s → 0.82 s 是「**保住已有 worker 的 CUDA graph**」。三者贡献量级完全不同。
- **观点：** (1) **对自建服务的直接启示是「CUDA graph 的可复用性应当被当作通信后端的选型标准之一」**，而不只是带宽/延迟；一个「换通信器就要重捕图」的后端，在弹性场景下的真实成本是几十秒停服。(2) **NIXL EP 的「连接但不激活（masked peers）」是个可迁移的设计**：它把「成员变更」从「替换对象」变成「状态位翻转」，这正是能保住图的前提。(3) **注意口径**：0.82 s 是「停机」而非「总时长」，总时长仍是 92.34 s；**缩容方向的总时长只有 9.32 s 而扩容是 92.34 s**，说明扩容的主要时间花在权重传输与加入者准备上——**扩容的「慢」被藏在了后台，但并没有消失**。

**来源**  https://vllm.ai/blog/2026-10-19-elastic-ep-near-zero-downtime （2026-10-19，12 min read）

**证据等级：** 官方一手（官方博客）。所有数字为官方自测，环境为 GB200 + DeepSeek V4 Pro + 预热编译缓存；**博客未给出未预热编译缓存下的表现**。

### 2.2 `#61938` Triton MLA 解码的 int32 溢出：默认上下文长度直接崩，且「修完溢出才暴露第二个 bug」

**主题**  vLLM `#61938`（`[Bugfix][Attention]`）修 Triton MLA 在 MI355 上**默认上下文长度即崩溃**的两个叠加 bug。

**内容详情**
- **现象：** 在 MI355 上用默认设置 `vllm serve` **Kimi-Linear-48B-A3B-Instruct（默认 1M 上下文）、LongCat-Flash-Lite-FP8（320K）、DeepSeek-V2-Lite-Chat（160K）**，均在 **CUDA graph capture 期间**以 `Memory access fault by GPU ... Write access to a read-only page` 死掉。
- **Bug 1（int32 溢出）：** split-KV 的 logits 偏移 `cur_batch * stride_mid_ob` 用 int32 计算。**≥128K 上下文时 `_compute_num_kv_splits` 在 MI355（2×256 CU）上选 512 个 split**，32 个 head 时单请求 logits 占 32 × 512 × 513 = **8.4M 个 float**；**从 batch 第 256 行起偏移越过 2^31**，而 **FULL graph capture 会跑到 batch 512**。
- **Bug 2（显存记账缺失，修完 Bug 1 才暴露）：** logits workspace **没有被显存 profiling 计入**。`#46819` 在 metadata builder 创建时预留最坏情况 logits workspace（**LongCat 默认配置下 32 GiB：1024 seqs × 32 heads × 512 splits × 513 × fp32**），而 builder 创建发生在 profiling **之后**，于是 KV cache 按「这块内存是空闲的」来定尺寸，分配 OOM（`Tried to allocate 164.67 GiB`，当时空闲 157.84 GiB）。
- **修法：** 六个 logits 偏移改 int64；新增 `MLAAttentionImpl.reserve_profiling_workspace()` 空钩子，`TritonMLAImpl` 在 profiling 期按上界预留。
- **验证：** MI355 TP1，三个模型在 main 上均 memory access fault，改后均可服务（GSM8K 87.34 / 86.88 / 可服务）。**「只修溢出」→ KV 分配 OOM；「只修记账」→ 仍然 fault**，两个都必须修。并发 384（decode 行 ≥256 走 512 split）0 失败。
- **意外收益：解码变快了。** rocprofv3 显示差异**全部**在 `_fwd_grouped_kernel_stage1`：**6062 → 5230 ms（−13.7%）**，grid 与寄存器数完全相同，其他 kernel 均在 ±0.5% 内。原因是 **int32 偏移把 logits 存储降级为标量 `buffer_store_dword`，而 int64 基址产生向量化的 `global_store_dwordx3`**。kernel 微基准在长序列上中性（30 组形状，中位 1.000×），但**在 512 split 且 batch ≥128 时快 15–17%**。
- **32K 上下文下（main 可用的区间）TPOT 变化：** Kimi C1 +0.4/+0.3%、C128 +2.2/+2.1%；LongCat C1 +1.0/+0.6%、C128 **+6.0/+5.3%**。GSM8K：Kimi 86.88 → 86.58（McNemar p = 0.75）、LongCat 86.88 → 87.34（p = 0.57），**均不构成精度回归**。

**应用场景**  ROCm（MI300/MI355）上的 MLA 模型长上下文服务、CUDA graph 全图捕获、任何 `batch × heads × splits` 会越过 2^31 的 logits 布局。

**研判**
- **事实：** 这是一个「**只在默认配置下、只在 ≥128K 上下文、只在 batch ≥256 时**」才触发的崩溃，而这三个条件在 MI355 上恰好是**默认值**。
- **观点：** (1) **「int32 偏移」是本领域复发率最高的一类 bug**，本期 SGLang 也有一条同族的（`#43438`，见第 3 节），说明**地址/偏移算术的位宽应当在代码审查清单里单独列项**。(2) **「修完 A 才暴露 B」是显存记账类 bug 的典型形态**：溢出让程序早死，反而掩盖了「workspace 没计入 profiling」；**这类 bug 的验收不能只看「不崩了」，必须看 KV cache 的实际尺寸是否符合预期**。(3) int64 反而更快这件事提醒：**「位宽换正确性」不一定有性能代价，有时因为向量化而变成净收益**——不要把 int32→int64 默认当作性能妥协。

**来源**  https://github.com/vllm-project/vllm/pull/61938 （merged 2026-10-19T00:39:14Z，merge commit `689effee`）

**证据等级：** 官方一手（PR 正文 + 实测表）。性能数字为 PR 作者自测，MI355 TP1 单机。

### 2.3 `#62206` EAGLE + 稀疏保留下，共享前缀的请求「一个 token 都命中不了」

**主题**  vLLM `#62206`（`[Bugfix][Core]`）修 **EAGLE/MTP + 稀疏 prefix-cache retention** 下，**只共享前缀的兄弟请求 prefix cache 命中为 0** 的问题。

**内容详情**
- **根因：** `#47782` 引入的「共享前缀接点（junction）」pin **把滑动窗口尾部多留了一个 block**，导致只共享前缀的请求**完全拿不到 prefix-cache 命中**。修法是把尾部 pin **下移一个 block**——那才是 EAGLE 兄弟请求的命中真正落点。
- **端到端（Step-5-Preview，MTP 层是滑动窗口、target 还有 full-attention DSA 层；4×GB300、TP4 + EP、MTP k=3、`--prefix-cache-retention-interval 1024`）：** 12 条 GSM8K 5-shot prompt 逐条发送，共享 565 token 前缀，每条都短于 1024 token：

| 配置 | 请求 1 | 请求 2 | 请求 3–12 |
|---|---|---|---|
| 无 MTP | 0 | 0 | 560 each |
| MTP k=3，**无本 PR** | 0 | 0 | **0** |
| MTP k=3，**本 PR** | 0 | 0 | **544 each** |

  544 = 560 接点下移一个 block，正是 EAGLE 查询能验证的最深命中。
- **全量 GSM8K（1319 题、并发 64、MTP k=3、开 Mooncake Store 连接器）：** 本地 prefix-cache 命中 token **0 / 839,793 → 737,520 / 839,793（87.8%）**；准确率 **0.933 → 0.935**；Mooncake Store 回放与 GPU cache 回放 **64/64 逐字节一致**。
- **代价：** 每个注册 junction 的请求，**每个滑动窗口组多缓存一个 block**。

**应用场景**  多轮对话 / few-shot 共享前缀 + MTP 投机解码 + 稀疏 retention（`--prefix-cache-retention-interval`）同时开启的在线服务。

**研判**
- **事实：** 命中率从 **0 直接到 87.8%**，且这是**纯逻辑修复**（下移一个 block），不涉及任何新 kernel。
- **观点：** (1) **「命中率 0」比「命中率低」更难被发现**：监控面板上 prefix cache 命中率长期为 0，很容易被解释成「负载本来就没有共享前缀」。**建议把「同前缀请求的命中率」作为独立指标，而不是只看总体命中率。** (2) 这条与上期 `#61455`（EAGLE group 改由 runner 上报的显式 draft-layer 标签，修 hybrid 模型 prefix cache 被静默清零）构成**同一族**：**投机解码的「层归属」信息一旦表达错，prefix cache 会静默失效而不是报错**。(3) 只有 MTP + 稀疏 retention + SWA 组三者同时成立才触发，**这解释了为什么它长期没被发现**。

**来源**  https://github.com/vllm-project/vllm/pull/62206 （merged 2026-10-19T12:38:18Z，merge commit `a608b6e6`）

**证据等级：** 官方一手（PR 正文 + 端到端表 + 单元测试 `test_swa_shared_prefix_reuse_under_zero_retention` 的 `use_eagle` 用例，在 main 上返回 0 而非 48 个命中 token）。

### 2.4 `#61465` cuDNN 从 CUDA graph 里重放了一个「已被卸载的 kernel」——最坏结果是静默错误输出

**主题**  vLLM `#61465` 把 `nvidia-cudnn-frontend` 的版本下限从 `1.19.1` 提到 `1.31.0`，以纳入上游 `NVIDIA/cudnn-frontend#1195`（*Keep execution plans alive for the CUDA graphs recorded from them*）。

**内容详情**
- **故障机理：** 修复前，cuDNN 可能在**最后一个 cuDNN 所有者消失时卸载一个 JIT 编译 kernel 的库**，而**一个 CUDA graph 里仍含有启动该 kernel 的节点**；重放这张图时，就会**派发到已经占据该代码地址的别的东西**。
- **受影响栈（`vllm/vllm-openai:v0.29.0-ubuntu2404` 同时具备两个条件）：**
  1. `FlashInferFP8ScaledMMLinearKernel` 在 CUDA 的 `_POSSIBLE_FP8_KERNELS` 里排第一，调用 `bmm_fp8(..., backend="auto")`；因为 `requirements/cuda.txt` 声明了 `nvidia-cudnn-frontend`，`cudnn` 模块可导入，`auto` 就会接纳 cuDNN runner。
  2. FlashInfer 用 `functools.lru_cache(maxsize=2048)` **按 shape 缓存这些 cuDNN graph**，会淘汰冷条目；而这些 GEMM **本身被捕获进 decode CUDA graph**。
- **可观测结果三档：** 非法地址错误、同步时报 CUDA 错误、**或者——对服务栈最要紧的——静默输出错误且不报任何错**。
- **归因不是统计性的：** GPU coredump 与 CUDA-GDB 独立确认了出错 kernel 为 `cudnn_generated_fort_native_sm80_matMul_pointwise_pointwise_knob_0_256x128x128_3_0`，**以 block (384,1,1) 启动，而 cubin 声明的 launch bound 是 256 线程、246 个寄存器**。
- **双重缓解：** 被测基线所钉的 **FlashInfer 0.7.2 已带 capture-retention 辅助**（让被捕获的 cuDNN GEMM graph 活到进程结束），单独即可缓解 GEMM 淘汰路径；本 PR 再确保 cudnn-frontend 自己的 CUDA-graph 生命周期修复被安装。

**应用场景**  FP8 scaled MM（`bmm_fp8`）+ CUDA graph 全图解码 + 长跑服务的任意组合；尤其 `v0.29.0` 时代的官方镜像。

**研判**
- **事实：** 这是一个**版本依赖层面的**修复——不改 vLLM 任何代码，只提下限。
- **观点：** (1) **「依赖库在后台卸载资源，而 CUDA graph 仍持有指向它的节点」是一个极难复现的静默错误源**，它的触发条件是**缓存淘汰**（时间/负载相关），因此**基准测试很难撞到，长跑服务才会撞到**。(2) **工程上的可迁移结论：任何「把 JIT 编译产物捕获进 CUDA graph」的路径，都必须确认该产物的生命周期被图本身持有**，而不是被某个 LRU 缓存持有。(3) 这条与上期 `#60735`（DBO × DeepEP 跨流缓冲复用）、本期 SGLang `#44564`、llama.cpp `#30795` 共同说明：**「同一块内存/地址被两个语义主体同时认为归自己」是本领域静默错误的第一大根因**。

**来源**  https://github.com/vllm-project/vllm/pull/61465 （merged 2026-10-19T15:48:13Z，merge commit `a3cee21d`）；上游 `NVIDIA/cudnn-frontend#1195`

**证据等级：** 官方一手（PR 正文 + GPU coredump/CUDA-GDB 归因）。「静默错误」为 PR 自述的可能结果之一，**未给出复现概率**。

### 2.5 `#53987` FLy：一个明确「有损」的投机验证策略被合入

**主题**  vLLM `#53987`（`[Spec Decode][ROCm]`）新增 **FLy：熵门控的延迟验证**，支持 MRV1 与 MRV2。

**内容详情**
- **机制：** 标准验证在**第一个被拒的 draft token 处停止**。FLy 在**目标熵高**且**后续 `W` 个 draft token 全部通过原生验证**时，**接受一个孤立的拒绝**。不需要额外 checkpoint 或 target 前向。
- **明确的有损声明：** PR 正文自述「**FLy 不保持目标分布，可能降低输出质量；标准验证仍是默认**」。
- **MRV2 结果：** 四个近期模型（DSpark/MTP），AMD MI355X 与 NVIDIA B300，batch 1/4/8/16/32/64，**48/48 个模型×平台×batch 组合全部优于标准 SD：AMD 1.055×–1.310×，NVIDIA 1.041×–1.273×**。
- **质量：** 在 GSM8K / MATH-500 / AIME25 / GPQA-Diamond 上，相对 Target 的**最低准确率保留为 AMD 96.43%、NVIDIA 96.30%**，两者都是 **AIME25 上差一道题**；**为单次测量，不是质量保证**。
- **V1 结果：** 现有四组 target/draft 对、十个基准，相对标准 SD **1.067×–1.938×**（AMD MI355X）。
- **实现细节（值得注意的口径偏差）：** 熵按 `-sum(p·log p)` 在**最大的 `fly_entropy_top_k` 个概率**上计算（默认 3），**且不对该子集重新归一化**——PR 自述这是「**实现的 top-k 门控，而非论文的全词表归一化熵**」。
- **配置：** `rejection_sample_method="fly"`、`fly_window_size`（默认 `min(6, num_speculative_tokens - 1)`）、`fly_entropy_threshold`（示例 0.3）、`fly_entropy_top_k`。要求至少两个投机 token 且窗口小于该数量。同时拒绝 `use_heterogeneous_vocab=True` 与 `use_local_argmax_reduction=True` 并用。

**应用场景**  吞吐优先、对输出分布保真度不敏感的负载（批量生成、草稿、数据合成）；**不适合**需要严格可复现或严格采样式保证的场景。

**研判**
- **事实：** 这是一个**被官方合入的、明确标注有损的**投机解码策略，且默认关闭。
- **观点：** (1) **「48/48 全胜」这类表述需要配合「最低保留 96.30% 且只差一道题」一起读**：AIME25 只有 30 题，**一道题就是 3.3 个百分点**，因此「96.3% 保留」在统计上等价于「可能没差别，也可能差几道」。(2) **`fly_entropy_top_k=3` 的熵不是论文的熵**，这意味着**照论文阈值调参会得到不同的接受率**——这是一个「实现与论文口径不一致」的典型，迁移时必须重调。(3) **对生产环境的建议是「不要用它替代标准验证，而是用它做 A/B 的一个臂」**，并且**必须把「与 Target 的分布距离」而不只是「准确率」纳入验收**。

**来源**  https://github.com/vllm-project/vllm/pull/53987 （merged 2026-10-19T07:30:09Z，merge commit `c1dc2e9c`）

**证据等级：** 官方一手（PR 正文）。**性能与质量数字为作者自测，作者自己声明「不是质量保证」；FLy 为有损方法，这一点是官方明确表述。**

### 2.6 其余 vLLM 提交（按主题归类）

窗口内 83 条，Bugfix 30、ROCm 10、CI 7、Frontend 5、Doc/Docs 4、Perf 3、Model 3、CPU 3、Core 2。**与推理优化直接相关的其余条目：**

- **KV Cache 与调度（正确性）：**
  - `#60767`（`[Bugfix][V1]`）把**滑动窗口以下的 block-table 条目置空**——SWA 下窗口外的 block 表项若不置空，会被当作有效引用。
  - `#61002`（`[Bugfix][KV Connector]`）**NixlConnector 支持「互不整除」的混合 MLA block 大小**：Kimi-K3 在 ROCm 上 P TP2（2944 token/block）配 D TP1（5760）原本**握手直接断言失败**。原因：混合 block 大小跟随 TP 分片后的 SSM page，TP 不同则大小不同；CUDA 上 MLA 后端强制 64-token kernel block 所以总能整除，**ROCm AITER MLA 接受任意 kernel block 大小，于是 `block_size_ratio` 断言**。修法：握手时把远端 block 视作 **gcd 大小的 kernel block**（2944 → 23 × 128），再用既有 `block_size_ratio > 1` 路径配对。实测（MI355 + AITER MLA）：`test_prefix_cache.py -k pd-tp2-dep2` 原本失败、现在 4/4 通过（10/10 轮）；新增 `test_hetero_non_divisible_block_sizes` 28 个用例由失败转通过；P TP2→D TP1、1024→1920 blocks 由 `handshake assert` 变为 **3.432 ± 0.027 req/s**，GSM8K 0.8748 ± 0.0020。**注意**：开 prefix caching 时 decode 仍会经既有 hybrid 检查拒绝这类配对（与 CUDA 行为一致）。
  - `#60374`（`[Bugfix][KV Offload][P2P]`）**NIXL 提交失败不再静默**：fetch 期间 `write_blocks` 抛出的 NIXL 错误此前**逃逸到 session dispatcher**，导致对端**等满 30 秒 load 超时**才重算；持续性 NIXL 故障此前会让「控制面健康但 NIXL 注册过期」的 peer **在此后整个进程生命周期内每次 WRITE 都失败**，现在升级为**结束 session**，并加入重连退避。关联 `#61500` 在 NIXL 1.5.0 上追到同一根因：**一次 `REMOTE_DISCONNECT` 之后 UCX endpoint 会一直是 `FAILED`，直到远端 agent 被重新加入**。
  - `#59635`（`[KV Connector]`）NIXL **pull** 连接器支持 sleep 模式。
  - `#51488`（`[Core]`）**generation 暂停期间拒绝新请求**。
  - `#59030`、`#60553`（`[Bugfix]`）分别修「encoder 停滞的等待请求重新入队」与「drafter look-ahead 导致的 encoder cache 死锁」。
- **投机解码与采样：** `#58231`（`[Bugfix][Sampler]`）**巨大 logits 下 split top-p 每 token 都做掩码**，症状是 **thinking_token_budget + 投机解码时会输出 token 0**；`#62377`（`[PCP]`）**DCP=1 下为 DSv3.2/GLM-5.2/5.3 稀疏 MLA 开 decode-only 全 CUDA graph**——GLM-5.3-NVFP4、4×B300、`-tp 1 -pcp 4 --enable-expert-parallel --kv-cache-dtype fp8`、MRV2：`cudagraph_mode=NONE` 时 GSM8K 0.9075 / 379.5 s / 330 tok/s，`FULL_AND_PIECEWISE` 时 **0.9083 / 145.9 s / 859 tok/s（2.6×）**，每 rank 捕获 11 张 FULL + 19 张 PIECEWISE；**纯 FULL 仍被拒（PCP 下 prefill 继续 piecewise），且尚未用 MTP 验证**。
- **MoE 与量化：** `#60118`（`[Perf][DSv4.1]`）**MegaAttn 的 `wo_a` einsum 直接把 `z` 写成 MXFP8**（连同 `wo_b` 的 FlashInfer MXFP8 GEMM 所读的 F8_128x4 UE8M0 scale），依赖 `vllm-project/DeepGEMM#40` 把 128x4 scale 布局加进 FP8 einsum epilogue。效果：`_o_proj` **去掉独立的 `mxfp8_quantize` kernel**，GB200 上 decode（1–256 token）每层快 0.4–2.0 µs、prefill（512–32768）快 2.8–141 µs（32768 token：2127.5 → 1986.9 µs），**唯一回归是 384 token 时 +0.6 µs**；20 个尺寸上输出**逐比特一致**。`#62392`（`[Perf][AuxOutput]`）routed-experts store 不再重扫已引用键。`#59737`（`[Perf]`）**fused MoE top-k gating 开启 PDL**。`#60805`（`[Bugfix]`）**HiSparse 热缓冲按「取整后的稀疏 top-k 宽度」定尺寸**：`index_topk=2048`、`index_kpool=4` 时 GLM-5.3-Flash 预留不完整池尾并把宽度取整到 **2176**，配 7 个投机 token 时每请求需要 `9 × 2176 = 19584` 默认热行，而原来只按 `9 × 2048 = 18432` 分配；同时放开 `device_buffer_size` 的启动期相等断言（用户选 129 逻辑行会占三个 64 行 block）。
- **ROCm：** `#61943`（`[ROCm][Perf]`）DSv3.2/GLM-5 稀疏 indexer 改用 **AITER top-k**；`#62371`（`[ROCm][Compile]`）把 **AITER 静态 FP8 量化融进 MLA attention 输出**；`#62131`（`[ROCm][Perf][DSv4.1]`）**DSpark draft 层的 mono decode**；`#61345`（`[ROCm][DSv4]`）**MegaMoE 共享专家融合改为必须显式 opt-in MXFP4**（从默认开启改为默认关闭）；`#60816`（`[ROCm][Qwen4Exp]`）**移除 Qwen3.8-Flash-Next TEP8 MXFP4 仿真守卫**；`#61726`（`[Bugfix][ROCm][MoRIIO]`）WRITE 模式支持**异构 TP**（覆盖所有 attention 与 mamba 层）；`#62241`、`#62286`、`#62280` 为 ROCm 侧通信器释放与 CI 稳定性。
- **CPU / XPU：** `#60740` 优化 routed INT8 W8A8 MoE GEMM 分发；`#61329` 修 **FP8 AMX attention 下溢**；`#61965` 展平 AMX int8 W8A8 线性路径中 **>2 维激活**；`#56835`（`[XPU]`）新增 triton/sycl GDN 开关；`#62375`（`[Benchmark][XPU]`）前向 bench 的 graph capture/profiling 支持 XPU。
- **前端与模型：** `#61727` 新增 **Step-5 Preview** 支持（本期 `#62206` 的验证环境）；`#62074` 新增 Reflection Beam；`#62237` 新增 ColUltraVBERT late-interaction 检索模型；`#61192`（`[Frontend]`）`/v1/systemone` 在混合模型上**对共享前缀只 prefill 一次**；`#60994`、`#60179`（`[Bugfix][Weight Cache]`）分别把 **RoPE 参数纳入 IPC cache key**、把**非 MoE DP daemon 按 DP=1 建键**；`#62206` 见 2.3。
- **CI/构建（占比最高的一类，7 条）：** `#61983` 升 Transformers 到 5.20.0；`#62564`、`#62567` 把最重的 4×L4 步骤改为仅 nightly 跑；`#61442` DeepEP 构建加 SM107a；`#62090` 为 Blackwell 与 Rubin 打补丁 MoE 单测；`#62547`（`[Security]`）**为硬编码 `trust_remote_code=True` 加 pre-commit 检查**。

---

## 3. SGLang：10-19 提交（80 条）——把 host sync 从调度器热路径里拿掉

窗口内 80 条（Fix 34、Perf 15、CI 10、Feat 10、Doc 7、Refactor 4）。**当日主线异常清晰：四条由同一小组提交的 PR，全部在消除「调度器主线程被 GPU 拖住」的隐式同步，合计带来 4.5%–9.9% 的吞吐提升**——而且**四条都不在 v0.5.22 内**（见第 1 节核验）。

### 3.1 四条同源 PR：`index_put` / 布尔掩码 / pageable H2D 三种隐式同步

| PR | 隐式同步的位置 | py-spy 定位 | 修复后 |
|---|---|---|---|
| `#45247` | `RequestWindow.reset` 用标量 `index_put`（`tags[:, loc] = -1`） | **调度器主线程 44%** 的采样 | `index_fill_`，同一行**371 ms → 0.14 ms** |
| `#45223` | 每 SWA 层每 extend step 用布尔掩码 + `torch.equal` 查历史，**每层 3 次 host sync** | **主线程 75%** 在 `RequestWindow.buffer` | 改走 `invariants.py` 的无同步 API，**+4.5%–+6.7% tok/s/GPU**（三节点），step 间隔 **~405 → ~375 ms** |
| `#45234` | DSpark `_forward_prefill` 用 Python list 索引 CUDA tensor + `torch.tensor(..., device=...)`，都是 pageable H2D 拷贝 | **主线程 50%** | 改 `async_h2d`（pinned、非阻塞）+ `index_select`，step 间隔 **~375 → ~312 ms**，吞吐 **1,459,199 → 1,581,184 tok/s/GPU（+8.4%）**、另一节点 **1,453,741 → 1,598,164（+9.9%）** |
| `#45062` | `enter_late_layer_dp` 用 `gpu.new_tensor(counts)` 装各 DP rank 的 late-layer 行数 | torch-profiler：**host 每 step 241 ms** | 改 `async_h2d`（pinned staging），**241 ms → 0.16 ms**；kernel 时间/step 占比 **~75% → ~97%**；step span 500–680 → 425–500 ms |

`#45062` 的端到端（GB300、DeepSeek-V4.1-Flash、PD 6 个 DEP2 prefill worker + 1 个 DEP16 decode worker、agentic 多轮、并发 2884、60 分钟）：**prefill GPU 利用率 75–79% → 91–93%**；**每 GPU 输入吞吐 475.2k → 535.8k tok/s**。`#45223` 的端到端（同机型、DP2 attention + MegaMoE EP2、encoder 与 decoder replay 均开、并发 512、900 s）：**+4.5% 到 +6.7% tok/s/GPU（三节点）**。

**`#45223` 的隐藏代价（必须与收益一起读）：** PR 正文明确写了行为变更——**`SGLANG_INVARIANT_CHECK` 关闭（默认）时，服务端不再检查 SWA 历史**；CI 在 `STRICT` 下仍会因缺失历史而失败。也就是说，**这条 +4.5%–6.7% 的收益里，有一部分是「把检查从生产路径里拿掉」换来的**，而不是纯粹把同一件事做得更快。

**应用场景**  任何「调度器需要读 GPU 侧结果才能继续」的设计；DSv4.1 的 encoder SWA bounded replay、DP attention、DSpark draft、多轮 agentic 长会话。

**研判**
- **事实：** 四条 PR 的收益量级（4.5%–9.9%）**大于**当日绝大多数 kernel 级优化，而它们改的都是**调度器里的几行 Python/PyTorch**。
- **观点：** (1) **「调度器不可 run-ahead」是本领域当前性价比最高的一类问题**：只要调度器主线程被 `index_put`/`.item()`/`torch.equal`/`new_tensor` 卡住，GPU 就会在两次 kernel 之间出现空洞，**profiler 上表现为「某阶段变长」，但根因不在那个阶段**——这与上期 `#43170`（布尔掩码释放致每 prefill 2.6 次 host sync）、TRT-LLM `#20250`（退役请求占座）是同一条主线。(2) **`#45223` 的做法值得单独记：把「必须检查」降级为「可配置的不变量断言」**，生产默认关、CI 用 STRICT 兜——这是「**性能与正确性的检查频率解耦**」的一个标准范式，但**代价是生产环境失去该检测**，必须在文档里写清楚。(3) 四条 PR 全部由 py-spy 采样定位，**说明这类问题用「采样调度器主线程」比用 GPU profiler 更有效**。

**来源**
- `#45247` https://github.com/sgl-project/sglang/pull/45247 （merged 2026-10-19T22:45:17Z）
- `#45223` https://github.com/sgl-project/sglang/pull/45223 （merged 2026-10-19T20:34:52Z）
- `#45234` https://github.com/sgl-project/sglang/pull/45234 （merged 2026-10-19T21:44:01Z）
- `#45062` https://github.com/sgl-project/sglang/pull/45062 （merged 2026-10-19T03:19:26Z）

**证据等级：** 官方一手（PR 正文 + py-spy/torch-profiler 定位 + 同机配对实测）。百分比为 PR 自测；`#45223` 的「移除检查」为官方自述的行为变更。

### 3.2 `#44904` 共享专家融合把路由权重的和从 1.0 压到 0.85–0.97（本期最严重的静默精度问题）

**主题**  SGLang `#44904`（`[Fix] moe`）修 **per-rank shared-slot 路径下，每个 MoE 层都把 routed 权重多归一化了一个专家**。

**内容详情**
- **触发条件：** `--enforce-shared-experts-fusion` + **per-rank slot** 的 a2a 后端（`deepep` / `mooncake` / `mori` / `pplx` / `megamoe`）+ **CUDA（或任何没有 aiter 的平台）**。DeepSeek 系模型开启该 flag 即命中。
- **机理：** 在 per-rank slot 路径上，`select_experts` 不给 router 共享槽（`num_fused_shared_experts=0`），因为 `_post_process_topk_ids` 事后追加共享列。**aiter 上 router 拿到的是 routed 数量 `top_k - num_fused_shared_experts`；而非 aiter 的每次 router 调用保留 `top_k`（`num_routed_topk if _use_aiter else top_k`），把共享槽也算进去了。** 于是 router 选了 `top_k` 个 routed 专家（比模型 `num_experts_per_tok` 多一个）并把这 K+1 个权重归一化到 1；随后 `remap_topk_for_per_rank_shared_slots` 用 home-rank 共享专家覆盖最后一列。**结果：每层每 token 保留下来的 routed 专家权重和变成 `sum(top K) / sum(top K+1)`。** CUDA router 按分数返回 top-k，所以**保留下来的专家是对的，只是权重变小了**。
- **实测（一张 H200、64 个随机 logits token、EP8、rank 3、一个共享专家）：**

| router 路径 | 修复前 | 修复后 |
|---|---|---|
| grouped sigmoid + bias，256 专家，top-8（DeepSeek-V3 形状） | **0.873–0.913** | 1.0000 |
| grouped sigmoid + bias，384 专家 8 组，top-8 | 0.873–0.914 | 1.0000 |
| grouped 单组 sigmoid + bias，384 专家，top-8（Kimi-K2 形状） | 0.873–0.919 | 1.0000 |
| grouped softmax 无 bias，64 专家，top-6 | 0.886–0.967 | 1.0000 |
| ungrouped sqrtsoftplus + bias，256 专家，top-6（**DeepSeek-V4**） | **0.850–0.885** | 1.0000 |
| ungrouped sigmoid + bias，256 专家，top-8 | 0.877–0.912 | 1.0000 |
| ungrouped softmax，128 专家，top-8 | 0.909–0.949 | 1.0000 |
| `HashTopK`（DeepSeek-V4 hash 层）、torch 与融合 kernel | 1.0000 | 1.0000 |
| DeepSeek-V4.1 vision routing on MegaMoE | 1.0000 | 1.0000 |

- **溯源：** `#36515`（非分组 router）与 `#40943`（分组 router）在**该路径上给 router 0 个共享槽**——这修好了 aiter（aiter 的 router 本来就只拿 routed 数量），**却把 CUDA 留在了上述状态**。也就是说，**这两个「修复」在 CUDA 上引入了本次的回退**。
- **同 PR 一并修的两个相关问题：** (1) **FlashInfer 的 DeepSeek router 忽略 `renormalize`**——`fused_topk_deepseek` 总是归一化，**它的第 9 个参数其实是 `launch_with_pdl`，而 sglang 的包装把它叫作 `renormalize`**；(2) per-rank 路径上的 **padding 行**：append 会把 padding 行的 `-1` 移位。
- **同栈的另外三条：** `#44912`（拆分 `num_fused_shared_experts` 的两种含义）、`#44913`（把共享专家配置收敛到各模型）、`#44901`（top-k 折叠 `routed_scaling_factor` 时只计一次）。

**应用场景**  DeepSeek-V3/V4/V4.1、Kimi-K2/K3 等「共享专家 + 分组/非分组 router」模型在 **CUDA 上开启共享专家融合**的部署。

**研判**
- **事实：** 这是一个**不崩溃、不报错、不影响专家选择、只影响权重幅度**的问题，**所有「看起来正常」的验证都会通过**；受影响的正是**DeepSeek 系在 CUDA 上的共享专家融合**这条主流路径。
- **观点：** (1) **「0.85–0.97」这个量级很危险**：它不会让模型胡说，只会让它「稍微变差」，因此**在评测噪声内**——这类 bug 通常只能靠「把 router 输出与不融合路径逐 token 比对」发现，正如本 PR 所做（64/64 行 routed 专家完全一致）。(2) **`#36515`/`#40943` 的教训是可迁移的**：**修一个平台的 bug 时，把条件写成 `if _use_aiter else <旧表达式>`，等于把另一个平台钉在错误语义上**；正确做法是**先统一语义，再按平台实现**。(3) **FlashInfer 的第 9 个参数被误命名**这件事说明：**跨库调用时「参数位置对」不等于「参数语义对」**，一旦上游改变参数含义，包装层会静默传错。

**来源**  https://github.com/sgl-project/sglang/pull/44904 （merged 2026-10-19T23:56:21Z，merge commit `48ea4869`）；同栈 `#44901` `#44912` `#44913`。**核验：不在 v0.5.22 内**（`compare/v0.5.22...48ea4869` → `diverged, ahead_by=402, behind_by=2`）。

**证据等级：** 官方一手（PR 正文 + 逐 router 路径实测表）。表内数值为单卡 64 token 的统计，**未给出端到端精度（如 GSM8K）对比**。

### 3.3 `#44564` DSA 索引缓存迁移：小 token id 会「整页复制」并覆盖无关的键

**主题**  SGLang `#44564`（`[Fix] dsa-eagle-topk`）修 **`speculative_eagle_topk=2` 下索引缓存迁移用错索引语义**。

**内容详情**
- **机理：** `DSATokenToKVPool.move_kv_cache` 把 **token-slot ID** 传给 `IndexKeyCache.move`，而后者**用这些 ID 去索引 page buffer**。**64-token 页下，slot 319 实际是第 4 页第 63 个偏移；把它当作第 319 页就会越界。** 更糟的是：**「小到能落进 page 维度」的 ID 会静默地整页复制，并覆盖掉无关的键。**
- **触发面：** 只在**投机解码重定位已接受的树 token** 时发生；普通单链（`speculative_eagle_topk=1`）路径**跳过该压缩步骤**，所以不触发。
- **修法：** 在 cache 读写方使用的**物理页视图**内移动每个被选 token 的 FP8 index key 与 scale；赋值前先做**高级索引收集**以保住重叠交换；复用别的层 indexer 的层继续跳过。**`index_kpool > 1` 的非空移动现在显式失败**（迁移单个 token 需要重建池化键）。
- **验证：** 在 main（`87211088`）上，新回归用例**交换 slot 1 与 2 会污染相邻键且不报 CUDA 错误**；打补丁后三个 GPU 用例在单张 GB300 上全过（小 ID 曾整页选中、跨页重叠交换含最后一个已分配槽、空移动），校验了**精确的键字节、scale、未变的邻居、latent KV、以及复用 indexer 的层**。**未做全模型分支投机解码的精度跑测。**

**应用场景**  GLM-5.x / DeepSeek-V3.2 等 DSA 稀疏模型 + EAGLE top-k>1 的树状投机解码 + FP8 索引缓存。

**研判**
- **事实：** 这是一个**只在「投机解码 + top-k>1 + 小 token id」组合下出现**的静默键污染，**普通单链投机解码完全不触发**。
- **观点：** (1) **「token 下标」与「页下标」混用**是本领域最经典的语义错配之一，与上期 `#41907`（MLA Mamba 408/544 取错 rank）、`#43333`（unified + `page_first_direct` 写穿 8 倍页）同族。(2) **最危险的是「小 ID 恰好能落进页维度」这种情况**——它不越界、不报错，只是**整页覆盖**，因此**只有在「相邻键是否被改动」这种逐字节校验下才可见**，GSM8K 之类的端到端精度很可能看不出来。(3) 建议把「**索引/页/槽三种下标的类型区分**」做成强制约束（例如用不同的 newtype 而不是裸 int），这比事后加断言更可靠。

**来源**  https://github.com/sgl-project/sglang/pull/44564 （merged 2026-10-19T14:54:08Z，merge commit `96ba1080`）

**证据等级：** 官方一手（PR 正文 + GPU 回归用例）。**无端到端精度数据**，PR 自述「no full-model branching-speculation accuracy run was performed」。

### 3.4 `#45075` PD 握手只比了 dtype 没比布局：MegaAttention 与 FlashMLA 的 KV 布局不同，decode 会读错

**主题**  SGLang `#45075`（`[Perf] deepseek-v4.1`）把 **PD 默认后端改为 MegaAttention**，并**把 KV 布局写进 PD 握手**。

**内容详情**
- **默认变更：** `_default_dsv4_attention_backend` 在 PD 下**不再回退到 FlashMLA**，两个角色在 **SM100/SM103** 上**都选 `flashmla_mega`**，与单机服务一致。
- **隐藏的坑：** **MegaAttention 与 FlashMLA 的 KV 布局不同**（默认分别是 **packed V4.1** 与 **V4**），而 **PD 握手此前只比较 `kv_cache_dtype`**——**布局不匹配的一对会照常传输 KV，然后由 decode 端读错**。修法：prefill 注册自己的 KV 布局（`DeepSeekV4TokenToKVPool.pd_kv_layout`，主布局加各压缩比的布局），Python 与 Rust bootstrap server 转发，decode 端在 dtype 检查旁**抛 `KV layout mismatch`**。
- **实测（两个 GB300 节点、DEP2 prefill + DEP2 decode、DSpark、Mooncake、不传 `--dsv4-attn-backend`）：** 两端都解析为 `flashmla_mega`，GSM8K（1319 题、贪心）**96.66%**，与两端都用 `flashmla` 时相同。**只把 decode 设为 `flashmla` 时，第一个请求就失败**并给出完整布局差异：`prefill server has kv_layout=v41,c1:v41_fp4,c2:v41_fp4, but decode server has kv_layout=v4,c1:v4,c2:v4`。
- **升级注意：** 行为变更——DSv4.1 PD 在 SM100/SM103 上默认两角色都跑 MegaAttention；**SM100/SM103 的 prefill 配其它 GPU 的 decode 需要在 prefill 上加 `--dsv4-attn-backend flashmla`**；**旧 decode 无法解析新的 `kv_layout` 字段，两个角色必须一起升级**。

**应用场景**  DeepSeek-V4.1 的 PD 解耦部署（尤其混用不同代 GPU 的场景）。

**研判**
- **事实：** 这是一条**「默认值变更 + 握手协议扩展」**的组合改动，风险不在性能而在**混版本部署**。
- **观点：** (1) **「握手只比 dtype」是一个普遍的隐患**：KV 的可传输性取决于 **dtype + 布局 + 每页形状 + 压缩比**，只比其一是**必然会被踩到的**。本期 vLLM `#61002`（block 大小整除性）、上期 `#43594`（NIXL Mamba checkpoint key 按 TP rank 分开）都是同一类「**元数据校验不全**」。(2) **`KV layout mismatch` 这类「显式失败」应当被当作正面改进**——它把一个「跑起来但输出错」的问题变成「起不来」，**在 PD 场景里这是净收益**。(3) 默认值变更的方向是对的（与单机一致、少一个特例），但**升级顺序必须写进 runbook**。

**来源**  https://github.com/sgl-project/sglang/pull/45075 （merged 2026-10-19T04:16:45Z，merge commit `82c9a8b0`）。**核验：不在 v0.5.22 内**（`diverged, ahead_by=346`）。

**证据等级：** 官方一手（PR 正文 + 两节点实测）。

### 3.5 其余 SGLang 提交（按主题归类）

- **新特性（10 条）：**
  - **`#42557` paged-experts：服务「权重装不进显存」的 MoE 模型**（本期最值得读的新特性，见下）。
  - **`#44945` `--enable-nvidia-megakernel` + NVIDIA megakernel 插件层**：插件以**树外包** `sgl-project/sgl-nvidia-mega` 形式发布，覆盖 **GB300 上 DeepSeek-V4.1-Flash 的 decode**；SGLang 侧只加**一个启动 flag + 一层框架**（`spec.py` 契约、`guards.py` 参数解析、`runtime.py` 的 `install_megakernel`），把 KV 池、attention metadata、Engram 表以**数据**形式交给插件。**守卫会在任何权重加载之前**一次性列出所有拒绝理由（插件未安装、模型无该 GPU 的 spec、SGLang 特性不被 spec 接受、TP/dtype/decode graph 后端/服务参数/HF config 字段/环境变量不被接受、watchdog 等待期限、spec 自身配置检查、同时选了另一个整步 kernel 后端）。`install_megakernel` 在**所有 post-load 模型改动之后、KV cache sizing 之前**执行。**开启后每 rank 每安装模型打一行日志**（含 spec、宽度、数值、构建耗时与 MiB、等待期限）。
  - `#44689` / `#44402` `sgl-router --decode-dispatch-limit`：**按 KV 到达情况挑 PD decode worker**；`#45120` 配套**拒绝「在 bootstrap 之前无法计算的 prefill worker」**；`#45102` `--pd-stream-after-prefill`；`#45100` 修 **kv-indexer 连接在取消前缀查询突发下被断**。
  - `#44518` Nemotron 3.5 Super 服务改动移植；`#36632` / `#44624` CPU 上支持 LTX/MOVA 与 Muse 模型；`#42118` Rust gRPC 增加 ListModels/Detokenize/Abort。
- **正确性修复（34 条中与推理相关者）：**
  - `#43438` `[Fix] attn_residual`：score/combine kernel 的 program id 改 int64——**37,451 token 处 int32 溢出**（与 vLLM `#61938` 同族）。
  - `#33007` `[Fix] attention`：**尊重用户选定的 FlashAttention 实现**（此前被覆盖）。
  - `#39592` `[Fix] dLLM`：**前缀复用时保住 prompt 尾部与字面 mask token**。
  - `#34535` `[Fix] hybrid-SSM`：**低并发下 DP attention 失败**。
  - `#42702` `[Fix] PD`：**retraction 恢复与在途 decode forward 竞争**。
  - `#44150` `[Fix] Qwen3.8next`：**breakable prefill CUDA graph 里不再保留按尺寸的 PLE 输出**。
  - `#44138` `[Fix] moe`：gated NVFP4 MoE 的中间维度对 FlashInfer TRTLLM 做对齐。
  - `#44500` `[Fix] nvfp4`：**等待配对权重时持有流式张量的所有权**。
  - `#45013` `[CI][Intel]`：**关闭 L0 copy offload 以修首次 H2D 拷贝挂死**。
  - `#44306` `[Fix] Z-Image`：warmup 请求后同步设备。
  - `#40066` `[Fix][CPU]`：**CPU 内存按 cgroup `memory.max` 定量**。
  - `#44975` `[CI]`：**恢复 muse spark B300 PD 覆盖**（上期 P2 观察项结清）。
- **性能（15 条中其余）：** `#44618` masked silu-mul-quant 的 grid 按 GPU 而非 padded 容量定尺寸；`#44906` mhc 小批量 post 按 hidden tile 并行；`#43775` dsa 的 k-pool indexer top-k 走 **top-k v2 kernel**；`#44927`（AMD）对**已证明为正零并列**跳过 top-k 精化；`#41229`（AMD）把不均匀 Triton decode batch **跨 XCD 铺开**；`#43390` `fp8_fa_sm120` 用 block-scaled MMA 编码；`#44580` FlashMLA scheduler 启用动态共享内存；`#43345`（XPU）**默认开启 fused QK-norm + RoPE**；`#43227` gated RMSNorm 避免 gate 拷贝。
- **重构（4 条）：** `#44912`、`#44913`（见 3.2）、`#45047`（encoder replay 行数计入 `_update_prefill_budget`）、`#44690`（清理 diffusion kernel 死代码并修「从未真正跑过」的测试）。
- **文档（7 条）：** 其中 `#45245` 把 **Qwen3.8-Flash-Next 数据中心 recipe 默认改为 FP8 KV cache + QSA indexer**，`#45246` 引入 **SGLang-TorchTPU**（TPU 指南与 README），`#45039` / `#45111` 刷新 SemiAnalysis AgentX 启动配置。

**`#42557` paged-experts 的关键数字（新特性，本期最有工程价值的一条）：** 每个 token 只路由到每层 `E` 个专家中的 `top_k` 个，且相邻 token 复用大部分专家。做法：**GPU 上保留每层 `E` 个专家中的 `K` 个（K 槽表），全部 `E` 个放在 pinned host 内存，只把本步需要且未驻留的专家拷进来**，然后**照常在 GPU 上跑该层原有的 fused-MoE kernel**——PR 自述这是「**把 HiCache 的思路用在专家权重上**」。`K` 与 KV cache 直接换：**K 小 → 更多上下文/并发（更多换页），K 大 → 更快**。实测（对比各方案的最佳设置）：
- **对 SGLang 现状（`--cpu-offload-gb`）：** 单流解码在 16 GB 卡上 **12×（int4）/ 30×（FP8）**，在 8 GB 笔记本上 **27×**。因为 paging 只搬「本步路由到且不驻留」的专家，而 `--cpu-offload-gb` **每次前向都把所有 offload 权重流一遍**。
- **对调优后的 llama.cpp（`--n-cpu-moe` 取最佳值，字节对齐的 `IQ4_XS`/`Q8_0`）：** 大部分模型能装下时（16 GB int4 模型放 16 GB 卡）或主机链路慢时（8 GB 笔记本 int4 走 PCIe 4.0 x8）**两者接近**：单流解码 **205.7 vs 194.4 tok/s**，4 请求 **379 vs 312 tok/s**；笔记本上 **30.6–32.9 vs 23.0–32.4**。**当大比例专家必须留在主机内存且主机链路快时（16 GB 卡 FP8 走 PCIe 5.0 x16）paging 明显更快：87.7 vs 45.2 tok/s。**
- **TTFT 是 paging 的弱项：** prefill 几乎碰到所有专家，因此会**把所有非驻留专家搬一遍**（Qwen3-30B int4、K=92 时约 4 GB，PCIe 5 上约 80 ms），而 llama.cpp 就地用 CPU 算那几层。**短 prompt llama.cpp 更快**（89–90 vs 68 ms、284–285 vs 226 ms、872–953 vs 620 ms）；**长 prompt paging 反超**（2048-token：267 vs 344 ms int4、443 vs 984 ms FP8、1441–1538 vs 2396 ms 笔记本）。
- **并发时可能落后：** 多请求 + 小 K/E 时，每步路由到的专家多于可驻留数，paging 会跟不上——**笔记本上 4 请求 11–12 vs 41 tok/s（K=26 of 128）**。

**研判（`#42557`）**
- **事实：** 这是**目前公开材料里第一份把「专家权重分页」与「llama.cpp 调优后的 `--n-cpu-moe`」做正面对比**的工作，而且**官方明确列出了自己输的场景**（短 prompt TTFT、高并发小 K）。
- **观点：** (1) **它的定位是「内存旋钮」而非「加速特性」**——PR 原文即如此表述，`K` 与 KV cache 直接互换，**这比「快 12 倍」的标题更值得记住**。(2) **它赢的地方是「大比例专家在主机 + 快主机链路」**，输的地方是 **prefill（要搬全部专家）与高并发**；因此**它适合「单流/低并发 + 长 prompt + 显存严重不足」的边缘场景**，不适合作为通用默认。(3) **与 llama.cpp 的对比里，paging 的 205.7 vs 194.4（+5.8%）在单流下并不算大**，真正的差距来自「SGLang 原来的 `--cpu-offload-gb` 每次都全量流」，**这是把「基线选错」造成的 12×/30×**——读这类数字时要先确认基线。

---

## 4. TensorRT-LLM：10-19 提交（27 条）——一个「288 专家」被静默回退到 cuBLAS，以及一次用填充带宽换执行器可用的取舍

窗口内 27 条，其中 **17 条带 `[None]` 前缀**（无组件标签），另有 **3 条 nightly lock file 自动提交**，实际有效改动约 24 条。**本期没有新的性能栈或新模型支持**，两条最有价值的提交都属于「**发现默认路径其实没生效**」与「**启动期资源竞争的取舍**」。

### 4.1 `#20390` GLM-5.3-Flash：FP8 1x128 量化向量化（1.80×）与 288 专家 router GEMM（原本静默走 cuBLAS）

**主题**  TensorRT-LLM `#20390`（`[perf]`）两个 C++ kernel 提速，**一个逐比特一致，一个改变 FP32 累加顺序**。

**内容详情**
- **第一部分：BF16→FP8 1x128 激活量化的向量化。** 原 `scale_1x128_kernel` 与 `scale_1x128_reshape_kernel` **每 128 宽的一组用一个 warp**，每 lane 两次 4 字节读、四次 1 字节写。新版：**16 个 lane 负责一组，每 lane 用一次 16 字节读装 8 个输入，组内 amax 用 4 次 warp shuffle 归约，再用一次 8 字节写存 8 个输出**；末尾不完整的组（如 K=576）在同一 kernel 内处理。host 启动器在「每行都是完整 16 字节 lane 块（`dim_x` 与行 stride 能被 8 整除）且输入输出 16/8 字节对齐」时选向量化版本，否则**回退到原逐元素 kernel**。**88 个基准用例全部与 main 逐比特一致（FP8 字节与 scale）**。kernel 时间（B200，profiler 计 kernel duration，5×200 次中位）：dense 4096×4096 **26.15 → 15.34 µs（1.70×）**、16384×4096 **107.27 → 59.75（1.80×）**、16384×7168 **184.34 → 102.62（1.80×）**、16384×4096 UE8M0 **116.51 → 63.27（1.84×）**、reshape 8192×32×512（行 stride 576）**305.54 → 161.73（1.89×）**；**小 M 是 launch-bound，基本不变（约 1.9 µs）**；两个刻意错位的用例走逐元素路径，**1.00×**。**没有用例变慢。**
- **端到端（GLM-5.3-Flash、`trtllm-serve`、TP4/EP4、MTP3、`max_num_tokens` 16384、`max_seq_len` 131072、4×B200、aiperf sonnet prompt、ISL 13312/OSL 33；两构建 Python 代码相同，只有原生库不同；每构建 2 个服务实例、4 轮中位）：** C1 TTFT p50 **372.9 → 366.6 ms（+1.7%，4 轮全部更低）**；C1 请求延迟 p50 **481.2 → 477.6 ms（+0.7%）**；**C8 TTFT p50 在运行间波动内**（main 上 1088–1553 ms）；C8 输出吞吐在噪声内。**显存不变**（warmup 峰值 98.88 GiB、每 GPU KV cache 61.20 GiB，两构建相同）。
- **第二部分：288 专家 dsv3 router GEMM。** `dsv3_router_gemm_op` 的低延迟 kernel **只覆盖 256 专家**（hidden 7168/6144/4096），而**专家数是 `shape_ok` 的一部分**——于是**任何其它专家数（包括 GLM-5.3-Flash 的 288 × hidden 4096）都会静默回退到 cuBLAS**。本 PR 为 N=1..16 实例化 `invokeRouterGemm<__nv_bfloat16, N, 288, 4096>`，把专家数检查移进各 dispatch 分支。
- **顺带修的内存安全问题：** 低延迟 kernel 与 cuBLAS 回退都**按 16 字节边界读 `hidden_dim` 个元素的行**，但该 op **既不检查行 pitch 也不检查对齐**。**在 main 上，从更宽的行切出来的操作数会让两条路径都算出错误 logits**，而从 16 字节边界后 8 字节开始的操作数会**报 CUDA 错误**（256 专家 kernel 报 misaligned address，256×7168、1–8 token 的 cuBLASLt 回退报错）。现在这类操作数会**先拷进 packed、对齐的布局**；模型本来传的就是 packed 对齐操作数，因此**不产生拷贝**。BF16 kernel 现在还要求权重 dtype 与输入一致。
- **精度影响：** 新 kernel 的累加顺序与 cuBLAS 不同，**同样的输入下，288×4096 的 logits 大多数在最后几个 FP32 位上不同（随机数据上最大绝对差约 3.5e-4）**；模型内精度由 `#20389` 的 GSM8K 覆盖（两个 PR 都合入后）。

**应用场景**  GLM-5.3-Flash 在 B200/GB300 上的 FP8 服务、MLA BMM 操作数的量化与重排、专家数非 256 的 MoE router。

**研判**
- **事实：** 第一部分**逐比特一致且无回归**；第二部分**改变了 FP32 累加顺序（最大差 3.5e-4）**。
- **观点：** (1) **「专家数写死在 `shape_ok` 里导致静默回退到 cuBLAS」是本期最好的「默认路径其实没生效」案例**：它不报错、不告警，只是**慢**。**这类「shape 特化 kernel + 保守 fallback」的组合，必须配一条「回退率」指标**，否则新增模型时会一直跑在慢路径上而不自知。(2) **行 pitch 与对齐未校验 → 两条路径都算错**这一点更值得警惕：**同一个 op 的两个实现共享同一个隐含假设，而该假设没有被断言**——这是「**多真相源**」的另一种形态。(3) **端到端收益（C1 TTFT +1.7%）远小于 kernel 收益（1.80×）**，说明该 kernel 在端到端里占比不大；**读这类 PR 时要先看端到端，再看 kernel**。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20390 （merged 2026-10-19T13:42:24Z，merge commit `47de5e4c`）

**证据等级：** 官方一手（PR 正文 + kernel 与端到端实测表）。精度差异为 PR 自述的 FP32 舍入级差异，**模型级 GSM8K 覆盖被推迟到 `#20389`**。

### 4.2 `#19650` KVCM2 的 host tier 改为渐进填充：用 5.2–5.6 GiB/s 换「执行器不被冻结 34 秒」

**主题**  TensorRT-LLM `#19650`（`[perf]`）让 **KVCM2 的 host tier 渐进分配**：**地址范围一次性预留、后台线程逐块 commit、分配只阻塞它真正需要的那一块**，**第一块驻留后就开始服务**，而不是等整个 tier 就绪。

**内容详情**
- **动机：** host tier 按主机内存定尺寸，**动辄几百 GiB**；让全部驻留在服务第一个请求前完成，**这段时间进程什么也不做**，且**随 tier 增大而变差**。
- **实现要点：** 在 `max_quota` 处一次性预留（`max_quota` 默认取 OS 报告的主机内存），**基址在对象生命周期内固定**；commit 在 worker 线程跑并发布一个 watermark；**等待很紧**（新槽 id 从 0 递增、回收槽排在最前，因此第 N 次新分配只阻塞包含它的那一块；填充完成后每次等待都是 relaxed load）；watermark 是普通 `std::atomic` + `atomic::wait`，**阻塞的分配是一次 futex 等待而不是条件变量加锁**。**commit 单位（512 MiB，控制进度）与内存分配单位（决定后续 resize 能多细地归还，且是槽大小的整数倍，使任何拷贝都不跨两次注册）刻意分开。**
- **关键取舍：** tier 由**内核 fault page in + `cuMemHostRegister`** 填充，而**不是**用 `cuMemCreate` 在驱动内分配。原因是**执行器饥饿**：`cuMemCreate` 在驱动内分配并清零页，**其 ioctl 路径持有 nvidia 内核模块里跨进程共享的锁**——一个 rank 填 tier 时，**该 rank 执行器线程上无关的 CUDA 调用会排在所有其它 rank 的分配工作后面**。实测（8×B200、8 个 rank 各填 64 GiB、一个探针线程持续计时 `cudaPointerGetAttributes`，即评审者观察到的卡住的调用）：

| 填充方式 | 填充速率 | 探针被阻塞比例 | 最差探针停顿 |
|---|---|---|---|
| `cuMemCreate` | 6.4–10.4 GiB/s | 96–100% | **4.6–34 s** |
| registered | **5.2–5.6 GiB/s** | 30–46% | **0.3–2.6 s** |

  **PR 自述得很直白：它牺牲填充吞吐，来避免一次性的启动填充把进程冻结几十秒。**
- **两个更便宜的修法都不行（同样 8 rank × 64 GiB）：** 在驱动调用间插 `nanosleep(1us)`——**单进程**下探针迭代从 12 增到 253、最差从 366 降到 1.3 ms，但**8 rank 下仍有 98% 被阻塞**（降到 96%）；把 `cuMemSetAccess` 按 granule 拆分——单进程最差从约 200 降到 26 ms，**8 rank 下 98% → 96.7%**。**两者单进程都「漂亮」，多 rank 都无效。**

**应用场景**  大 host KV tier（KVCM2）的启动、多 rank 同机并发启动、PD 解耦的 host 侧缓存。

**研判**
- **事实：** 这是一个**明确的「用带宽换可用性」的取舍**，且**官方把失败的两个替代方案与它们的实测数字一起写了出来**。
- **观点：** (1) **「单进程有效、8 rank 无效」这个对比是本篇最有价值的信息**：它说明**这类启动期资源竞争只在多 rank 同机时出现**，**单卡开发环境永远测不出来**。(2) **「填充单位」与「分配单位」分离**是个可迁移的设计细节：它同时满足「进度平滑」与「resize 可细粒度归还」两个相互冲突的目标。(3) **代价要算清楚**：填充从 6.4–10.4 降到 5.2–5.6 GiB/s，**64 GiB tier 的填充时间从约 6–10 s 变成约 11–13 s**——**总时间是变长的**，收益在「**第一块可用即可服务**」与「**执行器不被冻结**」。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/19650 （merged 2026-10-19T01:41:55Z，merge commit `0cd9d0b5`）

**证据等级：** 官方一手（PR 正文 + 8 rank 实测表）。探针方法为 `cudaPointerGetAttributes` 计时，**是间接指标**。

### 4.3 其余 TensorRT-LLM 提交

- **`#20443`（`[fix]`）融合 MXFP8 权重的 scale 现在跟随权重行加载。** 融合 MXFP8 投影（QKV / gate-up 分片装进一个 `Linear`）**用普通 TP 切分去切每个分片的 `weight_scale`，而权重本身走 `Linear.load_shard`（尊重 `override_tp_sharding` 的不均匀/复制行范围）**——**一旦用了 override，加载进来的 scale 与该 rank 的权重行不对应**。现在每个分片的 scale 也走 `Linear.load_shard` 并按 out-features 轴拼接；缺失或非 `uint8` 的 scale 会抛 `ValueError`。**这是又一条「scale 与权重不同源」的静默错误**（与 SGLang `#44904` 的 router 权重归一化同族）。
- **`#20223`（`[fix]`）Blackwell 投机解码树掩码按 SwapsMmaAb kernel 布局定尺寸。** 把 Python 的固定 tile 估算换成 C++ 容量计算（`trtllm::spec_dec_tree_custom_mask_capacity`），覆盖枚举到的 KeepsMmaAb 与 SwapsMmaAb 生成形状，并把容量经 attention 与 XQA 参数结构传到 FMHA runner；`runPrepareCustomMask` 在派发前检查容量为正且足够，Python 侧还检查掩码连续且非空。**针对的是「SwapsMmaAb 写到分配掩码之外」的风险。** **QA 备注值得记录：单元与集成测试结果为正，但后续 L0 pipeline 在未报告原因的情况下失败，覆盖结论为「需跟进直到这些 pipeline 失败被理解」。**
- **`#20405`（`[perf]`）降低 Gemma 多模态 attention 掩码构建的峰值显存**（`TRTLLM-17109`）。
- **`#20432`（`[feat]`）把 `spec_dec_variant` 加进 telemetry**——投机解码变体首次可观测。
- **`#20048`（`[fix]`）V2 改用字符串 cache salt 并迁移 LLM API 覆盖**。
- **`#20471`、`#20522`** 在 KVCM2 teardown 修复后**取消 attention 单测的 waive**，并让 attention 测试的 KV cache manager 在失败路径上也关闭——**说明 KVCM2 的拆除路径近期是问题集中区**（上期 `#20250` 亦为 KVCM2/overlap scheduler 相关）。
- **测试与基建（约 10 条）：** `#20593` 增加 Qwen3ForCausalLM 覆盖；`#20134` 把 decoder 测试与私有方法解耦并按阶段拆 DecoderRunner；`#20080` 从镜像取 UCXX 下载并重试 `build_wheel.py`；`#20140` 隔离 disaggregated client 输出；`#20477` 限制 LoRA param-fill kernel 中 `module_out_sizes` 的读取；`#19405` 加 SPHINX 生成页的精确路径 allowlist；`#20449` 把源码仓库与 commit 传给 PLC 容器扫描；`#20442` 保持 MiniMax-M3 流式与非流式推理解析同步；`#20475` 修 CI overview 描述；另有 3 条 nightly lock file 自动提交与 2 条 waive 记录。

---

## 5. llama.cpp：10-19 提交（27 条）——Metal 上「融合 kernel 的输出别名输入」的竞态，与 SWA checkpoint 的失效判断

窗口内 27 条，**15 个 nightly 标签（b11722–b11747）**，正式版仍为 v0.7.0。当日提交以**后端正确性**为主，性能条目集中在 CUDA/OpenCL/Hexagon。

### 5.1 `#30795` Metal `MOE_REDUCE` 融合在 `dst` 与输入重叠时产生罕见非确定性错误结果

**主题**  llama.cpp `#30795`（ggerganov）修 **Metal 融合 `MOE_REDUCE` kernel 的写后读/读后写竞态**。

**内容详情**
- **机理：** 融合的 `kernel_moe_reduce_f32` **在一次 dispatch 内**计算 `dst = Σ experts[e]·weights[e]`，**边读 `experts` 与 `weights` 边写 `dst`**。而**图分配器按「未融合的图」计算张量生命周期**——在未融合图里，这些输入**已被融合组中更早的节点完全消费**，因此分配器**可以合法地把输入内存回收给 `dst`**。在融合 kernel 里，这就变成**同一次 dispatch 内 threadgroup 与 threadgroup 之间的竞态**，产生**罕见的非确定性错误结果**。
- **修法：** 在 encode 阶段**用既有的 `ggml_metal_fusion_overlap` 辅助函数拒绝融合**（当 `dst` 与任一输入重叠时），与 `MUL_MAT_ADD` 融合用的是**同一套守卫模式**；未融合的 `MUL + ADD` 路径永远正确。
- **测试环境（PR 自述）：** 2× DGX Spark（RDMA、TP=2）、模型 `ggml-org/GLM-5.3-Flash-GGUF`、会话统计 8.4 小时。

**应用场景**  Apple Silicon 上跑 MoE 模型（GLM-5.3-Flash、gpt-oss、Qwen3-MoE 等）的融合 kernel 路径。

**研判**
- **事实：** 这是一个**只在「分配器恰好复用输入内存」时出现**的竞态，**错误结果是非确定性的**。
- **观点：** (1) **根因是「生命周期在两个不同的图上计算」**：**融合改变了数据依赖，但分配器仍按未融合图判断存活**——这与本期 vLLM `#61465`（cuDNN 卸载 kernel 而 CUDA graph 仍持有节点）是**完全同构**的问题：**优化把「消费完成」的时刻提前了，但资源回收方不知道。** (2) **修法是「拒绝融合」而不是「加屏障」**，这在正确性优先的场景下是对的；**代价是失去该融合**。(3) 值得注意**这条 PR 的测试环境是 2× DGX Spark + GLM-5.3-Flash-GGUF 跑了 8.4 小时**——**非确定性竞态只能靠长跑撞**。

**来源**  https://github.com/ggml-org/llama.cpp/pull/30795 （merged 2026-10-19T08:55:02Z，merge commit `098ce526`）

**证据等级：** 官方一手（PR 正文）。**未给出复现频率或修复前后的错误率对比。**

### 5.2 `#30666` llama-server 会从「上一个不相关 prompt」的 SWA checkpoint 恢复

**主题**  llama.cpp `#30666` 修 **`llama-server` 恢复由先前不相关 prompt 保存的 SWA checkpoint** 的问题。

**内容详情**  PR 正文极简：**修 `llama-server` 恢复 SWA checkpoint 时未正确判断失效**，并**新增一个修复前失败、修复后通过的 A/B 测试用例**。AI 使用声明：用于调试与起草测试用例。

**应用场景**  带 SWA（滑动窗口注意力）模型的 `llama-server` 多轮/多请求服务、prompt cache / checkpoint 复用。

**研判**
- **事实：** 正文未给出受影响的模型范围、错误表现（是精度下降还是直接错答）或性能数据。
- **观点：** (1) **「从上一个不相关 prompt 的 checkpoint 恢复」是本领域最典型的静默错误之一**：它**不崩溃、不报错**，只是**把不该复用的状态当成了前缀**。这与本期 SGLang `#45075`（KV 布局不匹配仍传输）、vLLM `#62206`（EAGLE 下 prefix cache 静默清零）构成同一条主线：**前缀/状态复用的正确性判断，是这个领域最容易出错的地方。** (2) **信息量不足**：由于 PR 正文未描述根因细节，本条**只能作为「已修复」的事实记录，不足以据此判断影响面**——需要看 diff 才能确认失效判断的具体条件。**本条证据等级因此降为「官方一手但信息有限」。**

**来源**  https://github.com/ggml-org/llama.cpp/pull/30666 （merged 2026-10-19T13:58:14Z，merge commit `f174553e`）

**证据等级：** 官方一手（PR 存在与修复事实），**但根因与影响面描述不足**。

### 5.3 `#30553` CUDA MMQ Stream-K 的网格上限：RTX 5090 上 TG +4.2%–4.5%

**主题**  llama.cpp `#30553` 给 **MMQ 的 Stream-K 网格加两道上限**。

**内容详情**
- **问题：** Stream-K **可能启动比「有用的 K 工作单元」更多的 CUDA block**；空 block **增加启动开销**，并**拉长 fixup kernel 的反向扫描**。
- **改动：** 仅 `ggml/src/ggml-cuda/mmq.cuh` 两处——(1) 把 Stream-K 网格**上限设为输出 tile × K-block 的工作总数**；(2) **上限设为总计算迭代数**，并对每 tile 不完整的迭代向上取整。**既有的 NVIDIA ≥90% tile 效率启发式保留。**
- **实测（12 组全新的 baseline/caps 配对，99.75% CI）：** Dell RTX 5090 P1 prefill **2030.61 → 2033.75 tok/s（+0.15%，[+0.06, +0.25]）**；P6 TG **215.21 → 224.79（+4.47%，[+2.10, +6.85]）**；P8 TG **268.39 → 279.57（+4.17%，[+2.20, +6.15]）**。**DGX Spark 无显著改善。**
- **精度注意（PR 自述）：** **不完整的迭代可能改变 FP32 求和分组**，即**可能有末位差异**。

**应用场景**  消费级 NVIDIA（RTX 5090 级）上 llama.cpp 的解码（TG）为主场景，prefill 收益极小。

**研判**
- **事实：** 收益几乎全部在**解码（TG）**，prefill 只有 +0.15%。
- **观点：** (1) **「启动比工作量更多的 block」是一个通用反模式**，它的成本不在 kernel 主体而在**启动开销 + fixup 扫描**，**profiler 上表现为「kernel 变长」但看不出空转**。(2) **DGX Spark 无改善**说明**该问题只在「SM 数量相对工作量偏多」的消费级卡上显著**——**这解释了为什么它在数据中心卡上长期没被优化**。(3) **FP32 求和分组会变**这一点必须在「要求逐比特可复现」的场景里注意。

**来源**  https://github.com/ggml-org/llama.cpp/pull/30553 （merged 2026-10-19T07:41:14Z，merge commit `4c1004c4`）

**证据等级：** 官方一手（PR 正文 + 带 CI 的实测表）。

### 5.4 `#30637` MoE 专家缓存从 LRU 改为 50/50 SLRU（1%–5%）

**主题**  llama.cpp `#30637` 把 MoE 的简单 LRU 缓存换成 **50/50 分段 LRU（SLRU）**——类似分代 GC，**一个较老的稳定集 + 一个较年轻的易变集**。

**内容详情**  作者自述在其各类测试中**比简单 LRU 好约 1%–5%**。测试覆盖**两张 GPU（3060 走慢 PCI 总线、4500 走快总线）、两个模型（Gemma4-26B-A4B、Qwen3.8-Flash-Next）**，并比较了 LFU、「把稀有专家留给 CPU 执行」以及多种混合调优；**该方法在无回归的前提下提升最好，且是其中最简单的之一**。作者同时说明：**依系统约束不同，有些方法还能再挤出约 5%，但在其它版本的设置上会比纯 LRU 更差**。AI 使用声明：**Astra 跑实验，作者审阅后由 Astra 写实现**。

**应用场景**  llama.cpp 的 `--n-cpu-moe` / MoE 权重缓存场景（显存不足以容纳全部专家时）。

**研判**
- **事实：** 收益 **1%–5%**，且作者明确说明**这是「无回归」前提下的最佳选择**，而非绝对最优。
- **观点：** (1) **「在有回归风险的方案里选无回归的那个」是这类缓存放量优化的正确姿势**，因为**专家访问模式高度依赖模型与硬件**。(2) **SLRU 的直觉与 `#42557` paged-experts 一致**：**近期用过的专家会被近期再用**（局部性），因此**分代结构比单一 LRU 更贴合**。(3) **注意这条 PR 是纯软件缓存策略**，**不改变数值结果**——这类改动是「安全收益」，值得优先采纳。

**来源**  https://github.com/ggml-org/llama.cpp/pull/30637 （merged 2026-10-19T11:28:50Z，merge commit `33d6135c`）

**证据等级：** 官方一手（PR 正文）。**1%–5% 为作者自测，未给出逐项数据表。**

### 5.5 其余 llama.cpp 提交

- **`#30573`（`[CUDA]`）修 `flash_attn_ext_vec` 的共享内存 WAR 冒险。** KV 循环在迭代间**复用共享的 KQ 缓冲**：写 scores 与读它们做 V 累加之间**有 warp 屏障**，但**下一轮迭代覆盖之前没有**。修法是在循环末尾加 `ggml_cuda_syncwarp()`（读都发生在写它的同一 warp 内，所以 warp 屏障足够）。**作者明确说「我并未观察到该竞态导致的错误输出」，是 compute-sanitizer 报出的写后读冒险**：Racecheck 在 `hsk=256, nb=1` 下**修复前 8,007,424 条 WAR 警告、修复后 0 条**；38 个正确性测试前后均通过；完整 FLASH_ATTN_EXT 正确性套件 **3993/3993 通过**（RTX 3090、sm_86、驱动 610.43.03，对 master `e467e5b`）。
- **`#30741`（`[opencl]`）为 q4_0 / q4_k / q6_k 增加 cooperative-k 的 dp4a GEMM。**
- **`#30756`（`[opencl]`）增加 i32 的 `get_rows`、i32 与 f32 的 `cpy`。**
- **`#30430`（`[sycl]`）修「重排后的量化权重上 `get_rows`」的问题。**
- **`#30731`、`#30218`（`[hexagon]`）**：前者改进 **FLASH_ATTN_EXT 以支持更长上下文与 SWA 模型**，后者**允许多流 FA**。
- **`#30607`（`qwen3`）双向注意力与 `dense_2` 输出投影**（多模态/非因果场景）。
- **`#30791`（`[ggml-cpu]`）向量化 `ggml_v_silu` 与 `ggml_v_expf`。**
- **`#30736`（`[mtmd]`）对齐 transformers 的 Qwen-VL resize**；**`#30825` 混合 DNA tokenizer 修复**；**`#30819`（`[jinja]`）实现 `unique` 过滤器**。
- **`#28277`（`[server]`）新增 `--connect`，允许 P2P 访问局域网外的服务端**；**`#30785` 把 `cpp-httplib` 升到 0.62.0**。
- **`#30497`（`[tools]`）split 与 export 工具拒绝被截断的 GGUF 张量数据**——**这是又一条「静默错误数据」的入口防护**。
- **`#30705`（`[tests]`）增加大端转换测试**；**`#30754`（`[ci]`）新增 WASI CI 与文档**；**`#30713` 重新启用 skills 的 type/lint 检查**；**`#30762` 把 `.pi/gg` 变成系统提示附录**；**`#30764` 更新 Metal 融合的 beam 基线**；**`#30774` 修 CUDA 多余分号的告警**；`#30806`、`#30687` 为 UI 修复。

---

## 6. 当前会话已形成的累计内容结论

按主题回溯本目录自 2026-09-25 起的连续档案，截至本期已沉淀的判断如下（**仅列已被多期独立证据支持者**；标注「本期新增」者为本期首次形成）：

1. **KV Cache 的低精度化已成默认路径，且进入「按硬件分档」阶段。** 09-25 的探索 → 10-13 的 KV 维度对齐 → 10-19 档案的 vLLM v0.32.0（NVFP4 覆盖 SM8x/SM90/SM100/SM12x、FP8 成为 GLM-5.x 在 SM100 的默认，代价是 TTFT +7–11%）。本期 v0.5.22 继续（`#40785` FP8 KV 量化融合 2.99×、`#41790` Q8KV8 稀疏 prefill 1.53–1.72×、`#42957` TileLang FP8 KV）。**主线是「同一件事在四种硬件上分别落地」，且官方开始公开代价。**
2. **KV 分层与迁移的瓶颈从「带宽」转向「每次命中的固定开销」。** 上期 arXiv 2610.21500 给出 `B* = sR`（2.53 GB/s）与 `T0 ≈ 0.33 s` 的判据，使**短于约 17k token 的 prompt 在任何带宽下都无法胜过重算**。本期新增两个同向证据：SGLang `#42399` 的 SeaweedFS 零拷贝路径把 **L3 TTFT 降到 1/3.5–1/4.7**（说明固定开销可被压），vLLM `#60374` 揭示**一次 NIXL 提交失败会让对端等满 30 秒**（说明固定开销的**尾部**远比均值重要）。
3. **PD 解耦的验收指标必须包含接受率，不能只看精度与吞吐。** 10-13（PD 失败语义）、10-17（PD 延迟释放导致 KV 页重复释放）、10-19（`#45037` accept length 3.16→3.85 而 GSM8K 97.12→96.74）。**本期强化为「PD 的验收指标清单还要再加两项」**：SGLang `#40703` 显示**吞吐 +26.81% 的同时 ITL p99 恶化 92.77%**；`#45075` 显示**布局不匹配会静默读错 KV**。→ **PD 的验收至少要有：接受率、ITL p99、KV 布局/dtype 一致性、前缀命中率。**
4. **「静默错误」是本领域最稳定的故障类别，且本期出现了它的三个新变种。** 累计已记录：SGLang `#44220`（prefill CUDA graph 重放与逐层异步加载冲突）、`#44847`（EPLB 漏搬 expert scale，GSM8K 97.5%→72.5%）、`#40917`/`#41907`/`#43333`（HiCache 三处损坏）。**本期新增：** (a) **数值幅度类**——SGLang `#44904`，共享专家融合把 routed 权重和从 1.0 压到 **0.85–0.97**，**不崩溃、不改变专家选择**；(b) **下标语义类**——SGLang `#44564`，token 下标当页下标用，**小 ID 静默整页覆盖**；(c) **资源生命周期类**——vLLM `#61465`，cuDNN 卸载了 CUDA graph 仍在启动的 kernel，**最坏静默输出错**；llama.cpp `#30795`，融合 kernel 的输出别名输入，**罕见非确定性错**。**共性未变：都不崩溃、都不报错，且现有测试都不覆盖。**
5. **「同一份布局/状态信息存在多个真相源」仍是第一大根因，且本期扩展到「同一 op 的两个实现共享同一隐含假设」。** 上期四条同源（`#40917`、`#43333`、`#41907`、`#44968`）。**本期新增两条更隐蔽的形态：** TRT-LLM `#20443`（**scale 走普通 TP 切分、权重走 `load_shard`**，两者不同源）与 `#20390`（router GEMM 的**低延迟 kernel 与 cuBLAS 回退都不检查行 pitch 与对齐**，**两条路径共享同一个未断言的假设**）。→ **工程解法仍收敛为「导出 typed buffer info，消费方只吃这一份」**，但对 kernel 特化路径还需加一条：**回退路径与快路径必须共用同一份前置校验**。
6. **调度器的「隐式阻塞」比 kernel 慢更常见，且本期拿到了它的定量画像（本期新增，最重要）。** 上期 `#43170`（布尔掩码释放 → 每 prefill 2.6 次 host sync）、TRT-LLM `#20250`（退役请求占座 → 批大小在 1/3 间振荡）。**本期 SGLang 四条同源 PR 把它做成了可复用的方法论**：`index_put`（**371 ms → 0.14 ms**）、每层 3 次 host sync（**+4.5%–6.7%**）、pageable H2D（**+8.4%/9.9%**）、`new_tensor` 装行数（**host 241 ms → 0.16 ms，GPU 利用率 75–79% → 91–93%**）。**结论：当前「调度器可 run-ahead」的收益量级大于多数 kernel 级优化，且定位手段应当是「采样调度器主线程」而非 GPU profiler。**
7. **发版节奏与「已修但未发」之间存在可观测窗口，且方向是双向的（本期新增）。** 上期用 commit 祖先关系证实 **vLLM v0.32.0 不含 `#61989`**（发版前 21 小时合入 main 但未进发布分支）。**本期反向验证：SGLang v0.5.22 的 tag 打在 02:55Z，当日 80 条提交中 `#42557`、`#44904`、`#45075`、`#45223` 均不在版内**（`diverged, ahead_by` 分别为 376/402/346/388）。→ **「发版当天合入 main」与「在发版内」是两个独立事实；release notes 的有无也不能作为任一方向的证据。**
8. **agent 生成推理内核正在改变维护方式，且本期出现「官方把 AI 实验记录写进 PR」的形态。** 10-16 的 vLLM megakernel 博客提出 fast-track RFC 四原则；v0.32.0 与多个 PR 出现 `[CC]`/`codex/claude generated` 标注。**本期新增三例：** SGLang `#44945` 把 **NVIDIA megakernel 做成树外插件**（SGLang 侧只留 flag + 契约 + 守卫）；llama.cpp `#30637` **由 Astra 跑实验、作者审阅后由 Astra 写实现**；llama.cpp `#30795` **由 ggerganov 本人提交且标注 AI 辅助**。→ **趋势是：性能改动更多落在「可插拔的树外插件」与「可复现的实验记录」上，core 的稳定性反而更有保障。**
9. **「默认值」正在成为主要的风险载体（本期新增）。** 本期三例：SGLang `--disaggregation-decode-host-receive-threshold` **语义翻转**（`0` 从「关闭」变成「总是开启」，**不报错只变慢**）、SGLang `#45075` **PD 默认后端变更**（旧 decode 无法解析新字段）、SGLang **要求 Python ≥3.11 但 3.10 上 `pip install` 会静默装旧版**。→ **升级检查清单里应当单独有一栏「默认值/语义是否变化」，而不是只看 breaking changes 列表。**

---

## 7. 今日可执行清单

**P0（今天就能做，风险低）**

1. **升级 SGLang v0.5.22 前，先逐条核对「默认值与语义」这一类变更：** `--disaggregation-decode-host-receive-threshold` **`1` 现在是「关闭 host receive」且是新默认**（此前用 `0` 来关闭的必须改成 `1`）；`SGLANG_AITER_MLA_DCP_DECODE_BACKEND` **默认值由 `gluon` 改为 `asm`**；`ChunkCache` 与 `tc_piecewise` prefill 图后端**已移除**（传参直接报错）；`SGLANG_DSPARK_FAST_SAMPLING` 等三个变量**已无任何作用**。**在 Python 3.10 上 `pip install sglang` 不会报错，而是静默装回最后一个支持 3.10 的版本**——务必显式检查 `python -V`。
2. **若用 vLLM 的 Triton MLA 后端跑 ROCm + 长上下文（≥128K）：立刻确认已含 `#61938`。** 否则在 MI355 上**默认配置就会 memory access fault**（Kimi-Linear-48B-A3B 1M、LongCat-Flash-Lite-FP8 320K、DeepSeek-V2-Lite 160K 均中招）。**注意必须两个修复都在**：只修 int32 溢出会转为 KV 分配 OOM（LongCat 上 logits workspace 最坏 32 GiB）。
3. **若在 vLLM 里同时用 MTP/EAGLE + 稀疏 prefix retention：确认已含 `#62206`。** 判据很简单——**看「同前缀请求的命中 token 数」是否为 0**。修复后 12 条共享 565 token 前缀的 GSM8K prompt 从「每条 0 命中」变成「每条 544 命中」，全量 GSM8K 本地命中率 **0 → 87.8%**。
4. **检查 `nvidia-cudnn-frontend` 版本是否 ≥ 1.31.0（vLLM）。** 低于此版本 + FP8 scaled MM（`bmm_fp8`）+ CUDA graph 解码的组合，**cuDNN 可能从图里重放一个已卸载的 kernel，最坏静默输出错**。已在 `vllm/vllm-openai:v0.29.0-ubuntu2404` 上确认条件成立。
5. **若在 CUDA 上开 `--enforce-shared-experts-fusion` 跑 DeepSeek/Kimi 系：核对是否已含 SGLang `#44904`。** 症状是**routed 专家权重和只有 0.85–0.97**（不报错、不改变专家选择）。**自查方式：拿一层 MoE，把 router 输出与「不融合」路径逐 token 对比**（本 PR 的验证即 64/64 行完全一致）。

**P1（本周内，需排期）**

6. **用 py-spy 采样一次调度器主线程（SGLang / vLLM 均适用），找 `index_put` / `.item()` / `torch.equal` / `new_tensor` / `torch.tensor(..., device=...)`。** 本期的四条 PR 收益是 **4.5%–9.9%**，且全部是几行改动。**定位口径：看调度器主线程的采样分布，而不是看 GPU kernel 时间。** 改法优先级：`index_put` → `index_fill_`（**371 ms → 0.14 ms**）、pageable H2D → `async_h2d`（pinned 非阻塞）、CUDA 张量索引 → `index_select`。
7. **PD 部署的验收指标补齐四项：接受率、ITL p99、KV 布局/dtype 一致性、前缀命中率。** 参考基线：SGLang `#40703` 在**吞吐 +26.81%、TTFT p95 −48.87%** 的同时 **ITL p99 +92.77%**——**如果只监控吞吐与 TTFT，这个开关会被误判为纯收益**。若要用 `prefill_complete` 策略，先在**非流式/批处理**负载上试，流式对话场景必须单看 p99 ITL。
8. **若用 TensorRT-LLM 跑非 256 专家的 MoE（如 GLM-5.3-Flash 的 288 专家）：确认已含 `#20390`。** 否则 router GEMM **静默回退到 cuBLAS**。同时注意该 PR 修了「行 pitch/对齐未校验导致两条路径都算错 logits」的问题——**如果你的模型传的是从更宽行切出来的操作数，升级前先验证 logits**。
9. **若用 TensorRT-LLM 的 KVCM2 host tier：评估 `#19650`。** 判据是「**多 rank 同机启动时，是否有无关 CUDA 调用被卡住几十秒**」。收益是**执行器不被冻结**（最差探针停顿 **4.6–34 s → 0.3–2.6 s**），代价是**填充带宽从 6.4–10.4 降到 5.2–5.6 GiB/s**（64 GiB tier 的填充时间约从 6–10 s 变成 11–13 s）。**单卡环境测不出这个问题**（两个更便宜的修法在单进程下都有效、8 rank 下都无效）。
10. **若在 Apple Silicon 上跑 MoE（GLM-5.3-Flash-GGUF 等）：确认 llama.cpp 已含 `#30795`。** 否则 Metal 的 `MOE_REDUCE` 融合在 `dst` 别名输入时**会产出罕见非确定性错误结果**。同时可评估 `#30637`（MoE 缓存改 50/50 SLRU，**1%–5%，无回归**）。
11. **若用 SGLang 的 PD 且 prefill/decode 不在同一代 GPU：升级前确认 `#45075` 的布局握手已生效。** SM100/SM103 的 prefill 配其它 GPU 的 decode **需要在 prefill 上加 `--dsv4-attn-backend flashmla`**；**旧 decode 无法解析新的 `kv_layout` 字段，两个角色必须一起升级**。

**P2（观察项，不急于行动）**

12. **跟踪 SGLang `#42557`（paged-experts）的后续。** 它的价值在「**K 与 KV cache 直接互换**」这个内存旋钮，但**已知输的场景**是：短 prompt TTFT（prefill 要搬全部专家）、高并发小 K（笔记本 4 请求 11–12 vs 41 tok/s）。**判断是否适用只需回答一个问题：你的负载是「单流/低并发 + 长 prompt + 显存严重不足」吗？** 若是，再评估；若否，`--cpu-offload-gb` 的 12×/30× 差距只是「基线选错」造成的。
13. **跟踪 vLLM Elastic EP 的后续路线**（Dynamo Snapshot 恢复热 worker、接入 Dynamo 与 llm-d 编排），以及 **NIXL EP 的「连接但不激活」模式是否被其它 EP 后端采纳**。**关键观察指标：换通信后端时 CUDA graph 是否仍可复用**——本期博客证明这一项的收益量级（58 s → 0.82 s）远大于「把准备放到后台」（99.62 s → 58.04 s）。
14. **跟踪 TRT-LLM `#20223` 的 CI 状态。** 该 PR 修「SwapsMmaAb 写到分配掩码之外」，**但 QA 明确记录「后续 L0 pipeline 在未报告原因的情况下失败，覆盖结论为需跟进」**——**在 pipeline 失败被理解之前，不应视为已验证**。
15. **跟踪 SGLang `#44945`（`--enable-nvidia-megakernel`）的插件生态。** 它把 NVIDIA megakernel 做成**树外包** `sgl-project/sgl-nvidia-mega`，SGLang 侧只留 flag + 契约 + 守卫。**值得观察的是「守卫一次性列出所有拒绝理由」这个设计是否成为树外 kernel 插件的标准形态**——它把「插件不兼容」从运行期崩溃变成启动期明确拒绝。
16. **跟踪 `vllm-project/vllm-torchtpu` 仓库。** HN 窗口内有一条 1 point、0 评论的条目指向它（`PyTorch-native TPU back end for vLLM`）。**已核验该仓库存在**（公开、创建于 2026-06-10T18:21:28Z、最近推送 2026-10-20T02:01:46Z、67 stars、无描述），**但未找到官方公告，内容未核验**——仅作线索记录。

---

## 8. 社区与视频资料

**本期可确认的事实：**

- **vLLM 官方博客：窗口内有 1 篇新文。** 《Achieving near-zero-downtime scaling in Elastic Expert Parallelism》（2026-10-19，Itay Alroy，NVIDIA，12 min read），**已全文核验并收录于第 2.1 节**。此前 10-14/10-15/10-16 的六篇（`train-inference-consistency`、`muse-spark-open`、`nemotron-3-5-super-vl-vllm`、`kimi-k3-megakernel`、`kimi-k3-mi355x`、`reflection-beam`）均已在此前档案中收录或说明，**本期不重复**。
- **SGLang 官方发布说明：v0.5.22，110,527 字符，已全文核验**（第 1 节）。
- **Hacker News（Algolia 公开 API，HTTP 200，按 2026-10-19 UTC 精确时间窗过滤）：**
  - `vllm` 32 条命中，**但与推理引擎实现直接相关的只有一条**（见下），其余多为 "LLM" 的模糊匹配。
  - **`sglang` 0 命中、`llama.cpp` 0 命中、`KV cache` 0 命中**（本期与前两期一致：这几个关键词在 HN 上的讨论密度极低）。
  - `speculative decoding` 2 条：一条是 Show HN《Acyclic; Rapid agent swarms》（无关）；另一条是 **《Why Is Speculative Decoding Fast?》**（2026-10-19T13:53:41Z 提交，2 points，0 评论，URL 路径显示为 `jbarrow.ai/2026-10-08-...`，**即文章本身日期为 2026-10-08，HN 提交在窗口内**）。**内容未核验**，仅记录线索。
  - `inference engine` 3 条，均为无关或产品页（《Yzma now over 2.6X faster at embeddings than Ollama or Docker Model Runner》为厂商自述基准，**未核验**）。
- **官方视频：未核验到可确认发布日期与内容的条目。** 本期不以推测替代。

**二手线索（标注为未核验）**

- **`PyTorch-native TPU back end for vLLM`**（HN 2026-10-19T22:34:38Z，1 point，0 评论，指向 `https://github.com/vllm-project/vllm-torchtpu`）。**已直接核验该仓库存在**：公开、创建 2026-06-10T18:21:28Z、最近推送 2026-10-20T02:01:46Z、67 stars、**无仓库描述**。**未找到 vLLM 官方公告或文档提及，仓库内容未核验**——**不能据此认为「vLLM 官方已支持 TPU」**，只能作为「存在一个 org 下的 TPU 后端仓库」这一事实记录。**证据等级：二手线索/未核验。**
- **《Why Is Speculative Decoding Fast?》**（`jbarrow.ai`，文章日期 2026-10-08，HN 提交 2026-10-19）。**内容未核验**，未记录任何数字。
- **HN 上「Run LLM Locally – which open model fits your GPU or Mac, and how fast」**（2026-10-19T14:57:49Z，4 points，2 评论，`runllmlocally.com`）。**为聚合/工具站点，内容未核验。**

**采集限制说明：** Reddit 未采集；YouTube 等视频平台未采集（无法在不引入推测的前提下确认发布日期与内容）；X/Twitter 未采集。GitHub 未认证 API 剩余配额 42/60（本期消耗 18），**PR 正文全部通过 HTML 直抓获取，未消耗 API 配额**。以上缺口不影响第 1–5 节的官方一手结论，仅影响「社区声量」这一维度。
