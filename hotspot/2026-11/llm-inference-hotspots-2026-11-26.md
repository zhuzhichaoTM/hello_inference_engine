# 大模型推理优化每日热点档案

> 档案日期：2026-11-26（本自动化于该日执行）
> 关注窗口：2026-11-25（周三）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，四家单页取全无截断）+ releases API + tag/release 精确查询 + **48 个关键 PR 正文全部经官方 issues API 抓取 markdown（无 HTML 降级）** + vLLM 官方博客全文（窗口前一天发布，回溯补录）+ HN Algolia（HTTPS，`numericFilters` 按 2026-11-25 epoch 窗口 1795564800–1795651200）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 83 / SGLang 55 / TensorRT-LLM 39 / llama.cpp 44 合计 **221** 条（2026-11-25 为周三，工作日正常量，较上一期 253 回落）；**窗口内无新正式版本、无新 prerelease**（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、TRT-LLM v1.3.0 11-20 与滚动 RC v1.4.0rc4 11-18 均落窗口前；llama.cpp 窗口内有 `#33043` 版本 bump 至 1.1.0 的提交，但 `v1.1.0` 的 tag 与 release 精确查询均为 **404**，releases API 窗口内仅 nightly，如 b12720–b12751，故本期**不记为已发布**）。本期头号事件为 **请求长度边界 × 每请求表/行宽不一致导致投机解码越界写污染同批其他请求**（SGLang `#52926` + `#53185`），并据此新增累计结论第 26 条；次主线为 vLLM MoRIIO/HiSparse 跨节点 KV 正确性与 Kimi-K3 启动编译优化、SGLang PD 状态竞态三连、TRT-LLM `#22561` 预嵌图像 embedding 的 KV 复用误图。HN 引擎官方更新类故事无（引擎相关帖均二手线索/未核验）。

---

## 1. 版本状态与本期头号事件：请求长度边界 × 每请求表宽静默越界 + 跨节点 KV 正确性 + vLLM ERNIE 5.5 官方博客补录

### 1.1 头号事件：投机解码在请求长度边界越界写，污染同批无关请求（官方一手，fix，新增累计结论第 26 条）

- **主题**：`req_to_token` 行宽与注意力每请求表宽各自按错误口径计算，投机 verify 阶段写/读越过本请求行尾。
- **内容详情（事实，PR 正文实测/代码路径）**：
  - SGLang `#52926` [Fix] mem_cache: Size req_to_token rows from the per-decode allocation reach：一次 decode 写入可到 `kv_committed_len + get_decode_reserve()` 按页对齐后的位置；行宽原为 `4 + num_draft_tokens`（仅 page_size>1/UNO 时才抬到 reserve），page_size=1 + EAGLE 草稿>4 + topk>1/DFlash 下**行比写入短且无越界检查**，尾部写进下一请求的行。修法把 decode 分配 reach 收敛到 `allocation_sizing.py` 单一链路（`get_decode_reserve()` / `decode_alloc_lens()` / `get_req_to_token_headroom()`），所有调用点统一读取。
  - SGLang `#53185` [Fix] attention: Size per-request tables from the req_to_token row width：verify 读 `seq_len + num_draft_tokens`，重叠调度再多一步；到达长度上限时可读到约 `context_len - 4 + 2*num_draft_tokens`。注意力后端却按 `model_config.context_len` 分配 page table / CUDA graph table / kv-index buffer：`build_trtllm_mha_page_table` 按 `cdiv(seq_len,page_size)` 掩膜、按表宽取整网格，尾部落到**同批下一请求的行**；aiter `scatter_ragged_to_page_table_kernel`、`create_flashmla_kv_indices_triton` 同类；Ascend 镜像图 replay 拷贝源切片宽于 `block_tables` 目标，`copy_` 抛错。修法为 trtllm_mha、FlashAttention/local、aiter、FlashInfer MLA、Ascend、XPU、hpc_ops、MiniMax sparse、Triton、wave、hybrid wrapper、Qwen sparse 等统一取 `req_to_token_pool.max_context_len`（行宽）。
- **应用场景**：开启投机解码（EAGLE / DFLASH / DSpark / MTP）+ 长上下文（如 context_len=128000、page_size=1）+ 批内混合长度 + PD 解耦的在线服务；`--max-new-tokens` 接近上下文上限时风险最高。
- **研判（观点）**：这是第 19 条「非量化常规路径静默错误」与第 9 条「边界计算」的交叉新子项——不崩溃、不报错，受害者是**同批的另一个请求**，比单请求输出错更难定位。`#53185` 的 Ascend 分支会直接抛错，属同根因的显式失败；`context_len` 为 kernel block 整数倍时不触发，等于把安全边界绑在配置巧合上，须立为第 26 条。
- **来源**：`https://github.com/sgl-project/sglang/pull/52926`（2026-11-25T23:46:30Z 提交）、`https://github.com/sgl-project/sglang/pull/53185`（2026-11-25T23:45:32Z 提交），PR 正文经官方 issues API 抓取。
- **证据等级**：官方一手（上游 main 已合入；修复含代码路径与单测，未见端到端精度 A/B 数字）。

### 1.2 次主线：跨节点 KV 与状态正确性簇（第 4/5/8/22 条延续）

- **vLLM `#67074` [Bugfix][ROCm][MoRIIO] Skip zeroing READ destinations and wait for their pending copies（官方一手，fix，强静默铁证）**：混合精度 KV + 异步调度下，model runner 在计算流上把新块清零、且没有任何顺序约束保证 MoRI 写入发生在其后，READ 已收到的 prompt KV 被抹掉。MI355X、MoRIIO READ over xGMI、DSv4-Flash P TP4→D TP4：GSM8K strict/flexible **0.600/0.644 → 0.954/0.954**（无 connector 0.955/0.955）；Qwen3-32B fp8 KV（layer0 bf16）P1→D1：xGMI **0.056/0.064 → 0.923/0.905**、RDMA **0.036/0.036 → 0.934/0.918**（无 connector 0.927/0.918）。仅当计算流上确有块拷贝写入 READ 目的块时才等待，8k/512 与 16k/16 吞吐/TTFT/TPOT 与 main 差 ≤0.3%。**研判（观点）**：第 4/8 条「跨层/异步 KV 生命周期」的定量铁证，「省一次 host sync」的优化必须配对写入依赖图。
- **vLLM `#67216` [Bugfix][ROCm][MoRIIO] READ mode restore the cached recurrent state for hybrid prefixes that end on a block（官方一手，fix）**：hybrid 模型 prompt 的 N−1 个 token 恰在块边界时，decode 缓存的 Mamba 状态块未被写入，后续同 prompt 请求从陈旧状态续写。Qwen3.6-35B-A3B on MI355X，32 prompt 发两遍 greedy：main 第二次仅 **8/32** 与首次一致，修后 **32/32**（与无 connector 相同）；GSM8K 0.955/0.955 不变。
- **vLLM `#69901` [Bugfix][HiSparse][NIXL] Import shared host KV on every node's host leader（官方一手，fix）**：`#68747` 后多节点 TP 每节点各有 HiSparse 共享 host 池（由节点最低 TP rank 写），但 NIXL 只在 global TP rank 0 注册/READ DRAM 部分，导致后续节点 decode rank 收不到 prefill 主 KV。单测在 base 上 nnodes=2/4 的 DRAM readers 为 `[0]`（期望 `[0,4]`/`[0,2,4,6]`），修后 3 passed；相关 50 + 55 单测通过。**注意**：作者仅有 CPU(macOS) 验证，无多节点 Hopper/Blackwell 硬件。
- **vLLM `#68742` [Bugfix][NIXL] Tile whole kernel rows of groups to persist logical block for drafter blocks（官方一手，fix）**：Kimi K3 + DSpark draft 在 GB300 上加载 NixlConnector 时 `register_kv_caches` **启动即失败**（`#59586` 只接受 dense token-major 行分割，草稿行 917504B token 落在 1032192B 行内带 padding）。修后 8 轮×15 分钟、并发 32、30,032 次传输 0 失败，接受率 0.151–0.174（接近无 KV 移动的聚合服务 0.158–0.160）；相对既有 workaround 0.975× geomean（t=−0.94，无显著差异）。作者明示可能 K3 专属、DP→TP 未充分测。
- **vLLM `#69774` [Bugfix][Multimodal] Make vision FA3 batch invariant（官方一手，fix，第 19 条）**：仅在 `VLLM_BATCH_INVARIANT=1` 时把 CUDA vision FA3 钉到 `num_splits=1`；否则打包第二条视觉序列会改变 split 数与 KV 归约顺序。H100 + Qwen3-VL-2B：修前 BF16 kernel 1,048,576 元素中 **277,130（26.4%）不匹配**、logprob max_diff=1.139e-1，修后 2 passed。
- **TRT-LLM `#22561` [fix] Key attached multimodal embeddings for KV cache reuse（官方一手，fix，第 4/5 条）**：请求附带预计算图像 embedding 时无多模态 hash，KV 复用仅按 placeholder id 命中——**同文本不同图可复用上一请求的图像 KV 并描述错图**（#22117）。修法按 embedding 行内容（+ `multi_modal_uuids`）做键，覆盖 Qwen2.5-VL/Qwen3-VL、LLaVA-NeXT、Llama 4；32MiB Qwen3-VL-8B embedding 哈希 3–6 ms，API 吞吐在 main 的 2.1% 内；H100 回归：不同 embedding 在 main 上误复用 269 个缓存 token。Llama 4 同 embedding 仍无法复用（token 数不匹配，`fuse_input_embeds`），未解决。
- **TRT-LLM `#22742` [fix] Account for encoder replay rows when gathering context logits（官方一手，fix）**：一请求命中缓存前缀 replay 物理行、另一请求请求 context logits 时，只数新 prompt token 会错位采样行与 logits 行；修法在 forward 消费 replay plan 前捕获各请求 replay 长度，按打包后真实顺序定位。B200 + 随机初始化 DSv4.1（FP8 KV）：A 命中 256-token 前缀 replay 140 行 + 1 新 token、B 请求 logits 的复现用例。**注意**：与在途 `#22038` 需合并对账，合并顺序未定。
- **SGLang `#52627` [Fix] kimi-k3: wait on the PDL primary before the first attention-residual bank load（官方一手，fix，第 22 条三期再延长）**：`attn_res_fused_tma` 以 PDL 发射，producer 在 `griddepcontrol.wait` 前发首 token 的 bulk TMA load；PP stage 入口 bank 由 `new_empty` + copy 填充，首层聚合可读到陈旧字节（通常有限，若陈旧字节为 NaN 则污染后续 linear-attention 与复用该 recurrent state 的后续轮次）。6 次 1 小时 PP8 prefill（共 50,340 forward）中 4 个 forward 出现非有限首 token（0/2/0/2/0/0），有限错误未计入；修法在 token 循环前等 primary 一次。

### 1.3 官方学习资料补录（窗口前一天发布，非 11-25 窗口事件）

- **vLLM 官方博客《vLLM Day-0 Support for ERNIE 5.5 Flash Preview: Plugin Deployment and E2E Inference Acceleration》**：`https://vllm.ai/blog/2026-11-24-ernie-5-5-flash-preview`，HTTP 200、`datePublished=2026-11-24`、标题页显示 November 24, 2026、11 min read（约 151K 字符 HTML 全文核验）。**明确标注**：发布日期落在本期窗口前一天，上期档案（覆盖 11-24）未收录，按前几期惯例回溯补录，不计入 11-25 窗口事件。
- **事实（官方）**：ERNIE 5.5 Flash Preview 为 280B 总参 / 14B 激活、43 层（37 层 HCA head_dim=512 + 6 层 MLA-DSA head_dim=256）、1M 上下文；默认权重 routed experts 用 MXFP4(W4A8 QAT)、其余 FP8，KV cache FP8、DSA indexer cache 可 MXFP4。FlashIndexer（动态 head 路由 + 阈值预筛）在 1M 上下文索引提速 **1.4×**（Highlights 记 40% speedup），下游 sparse attention kernel 不变；router 冻结主干与 indexer、约 1B token 微调；专用融合 kernel 比朴素实现快 **1.55–6.53×**（VHA pre-fold 1.56–1.73×）。3 层 MoE DSpark draft 每步最多 7 token，真实 coding/agent 负载平均接受长度 **>4.0**，训练链含 mid-training / post-training / on-policy distillation（draft 训练 >3× 于常规联合训练）。部署为独立 `vLLM-ERNIE` 插件 + `vllm==0.32.0` wheel，无 fork/专用镜像；示例命令 DP4+EP、FP8 KV、`indexer_kv_dtype=mxfp4`、DSpark 7 token。
- **应用场景/研判（观点）**：新旗舰稀疏 MoE 的 day-0 路径从「等 nightly/维护 fork」转为插件化（与第 20 条发版/破坏性变更主线互补）；接受长度 >4.0 与 indexer 加速为官方口径，未给端到端 tok/s 对照，按「官方学习资料/描述级」处理。

### 1.4 版本状态与破坏性默认值

- **无新正式版/无新 prerelease**（四家 releases API 均 HTTP 200）：vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、TRT-LLM v1.3.0 11-20 与 RC v1.4.0rc4 11-18 均落窗口前不变；llama.cpp releases 窗口内仅 prerelease nightly（如 b12720–b12751）。
- **llama.cpp `#33043` [Bump version to 1.1.0](https://github.com/ggml-org/llama.cpp/pull/33043)**（官方一手，feat/API，第 3/20 条）：提交在窗口内 15:11Z，但 `releases/tags/v1.1.0` 与 `git/ref/tags/v1.1.0` 均 **HTTP 404**，故本档案只记「版本 bump 已入 main、尚未打 tag/发 release」。发布说明要点：Inkling 2 家族（base/Mini/DSpark）、Domyn Edge 4B、Zaya1 8B；DFlash 拆为 `draft-dflash` / `draft-dflash2` / `draft-dspark` / `draft-lilicorr`；**弃用 `llama_batch`，改 `llama_batch_ext` + `llama_process`**；`llama_model_quantize_params` 增 `target_bpw`/`target_size`/`state_file`/abort callback（自动按 bpw 或尺寸目标的逐张量量化）；ggml 同步 v1.1.0；llama-server slot 调度重构；多模态新增大批 TTS/音频模型。
- **TRT-LLM `#22334` [BREAKING] Decode images as PIL and video as uint8 by default（官方一手，perf/破坏性，第 20 条）**：默认把图像解为 RGB PIL、视频帧解为 RGB uint8 `(frames,h,w,3)`，8 帧 1920×1080 存储 **199,065,600 → 49,766,400 字节（−75%）**，Kimi K2.5 token 计数中位 **13.26 ms → 0.024 ms**；移除 `media_io_kwargs` 的 `format`/`device`（再用则 LLM/VisualGen 启动报错、chat HTTP 400、OpenEngine invalid-argument）。**BREAKING**：同一 modality 混用 float tensor 与 PIL/uint8 的离线请求报错；Qwen 需按字节尺度输入在 `images_kwargs`/`videos_kwargs` 设 `do_rescale=True`；共享 encoder cache 的离线请求须统一格式或用不同 `multi_modal_uuids`。491 单表示用例 284 结果不变、207 改善，138 混合用例全按预期报错。

---

## 2. vLLM：2026-11-25 提交（83 条）—— Kimi-K3 启动编译 583→260s(#69669) + Qwen skinny GEMM TPOT −16.7%@c1(#69736) + GDN 融合 prefill 3.2×(#69135) + 跨节点 KV 正确性四连 + 安全边界

### 2.1 `#69669` [Perf][Kimi K3] Specialized decode: compile each kernel once, only for captured sizes（官方一手，perf，启动成本）

`decode_backend=latency_specialized` 此前把 o-projection 的 `@cache` 挂在 kernel 从不读取的参数上（producer 不读 `nb`/`has_delta`/`front`，consumer 不读 `k`/`mla`/精确 batch），约 93 层 × 6 种 ≈ 570 个 kernel/-rank；且除 CUDA graph 捕获尺寸外还预准备固定 batch 1–8/16/32/64。改为按实际读取项缓存 + 只准备捕获尺寸。8×B300、Kimi-K3.1-Pro、`--max-num-seqs 64`：**TP8 编译 1435→479/-rank、权重加载后启动 583→260 s**；TP8+DCP8 1553→549、675→316 s。捕获图与 kernel key 集合为原集合的严格子集，c1–c64 decode 中位 step/吞吐无回归；相对 stock decode 仍多约 2.6/3.6 分钟。与跨重启编译缓存 `#65026`/`#66623` 可叠加。

### 2.2 `#69736` [Perf] SM100/SM103 skinny GEMM plans for Qwen3 / Qwen3.5/3.6 MoE / Qwen3.8 decode（官方一手，perf，纯表）

B200(SM100)/B300(SM103) 上 1–8 token 的 BF16 decode projection 走 cuBLAS split-K（小算子约 5 µs，skinny kernel 2–4 µs）；本 PR 合入此前 `#69743`/`#69773`/`#69750` 五部分（共 14+49 / 41+91 / 15+49 个表项 + Qwen3.8 router/LM head）。入选规则：sweep 中胜 cuBLAS ≥10% 且独立复核（4 seeds×3 交错轮、CUPTI、CUDA graph replay、冷 L2），误差与 cuBLAS 相同（最大 3.8e-3）。E2E 中位 TPOT（c1/c2）：Qwen3-8B **−16.7%/−14.9%**、Qwen3-30B-A3B **−12.1%/−10.5%**、Qwen3.6-35B-A3B-FP8 −1.4%/−2.3%（c4/c8 为 **−6.2%/−4.6%**）、Qwen3-32B TP8 −14.0%/−10.4%；Qwen3.8-Flash-Next-FP8 TEP4 1k/1k c2 仅 −3.0%。GSM8K 基本不变（如 0.907→0.908）。SM103 专用表经 sweep 中位差 0，故 B200/B300 共用一张表。**研判（观点）**：收益集中在 c1–c2，c4–c8 多数形状 cuBLAS 已具竞争力；升级须按并发分层验收。

### 2.3 `#69135` [Perf][Kernel][GDN] Fuse prefill causal conv1d + post-conv prep（官方一手，perf）

纯 prefill 批（Qwen3-Next/Qwen3.5/Qwen4Exp）把 `causal_conv1d_fn` + `fused_post_conv_prep` 合一，conv 输出不再回 HBM。GB300：16–32K token **3.0–3.2×**（5.0–5.3 TB/s，约同机 device copy 6.6 TB/s 的 80%），更小形状 1.2–2.2×，stock 链约 1.65 TB/s；Qwen3.8-Flash-Next-FP8 TP1 端到端 prefill 吞吐 **+2.0%**。conv state 逐位一致，q/k/v 在约 2 bf16 ULP 内；GSM8K 不变。仅 3 个 chunk-size 变体覆盖任意 head 数/GQA/TP；`f32x2` 打包路径限 sm_103a（B200 用标量 FMAs）；含 decode 行/conv bias/非 SiLU/head_dim≠128/ROCm 走原路径。

### 2.4 `#69738` [Perf][Model] Nemotron Shortcut SDSA：NVFP4 K/V 按 key position 每请求转换一次（官方一手，perf/依赖）

FlashInfer `#9321`（逐行 `gather_nvfp4_kv_cache`）未合并即关闭，导致 SDSA 后端拒绝 NVFP4 KV；改用 0.7.6rc1+ 的 `gather_nvfp4_kv_cache_by_position` + `compact_nvfp4_kv_cache`，decode 一次 launch 转完 16 请求（含 MTP 草稿行）所选 key，prefill 每请求每步每层转一次。GB300 4 卡：GSM8K 各路（BF16 ckpt + `--kv-cache-dtype nvfp4` TP4、+MTP3、NVFP4 ckpt TP4/DP4+EP）均在 baseline 标准误内，接受长度不变；DP4+EP 5/5 启动干净、192 题@c64 无 CUDA 错误。**注意**：vLLM 仍 pin FlashInfer 0.7.4，CI 路径跳过新测试，**未给吞吐数字**，属功能打通+数值等价。

### 2.5 其他要点（官方一手）

- **`#69461` [Security] Bound synchronous media body fetches**：text-only pooling 模型在发出媒体请求前拒绝图像/音频/视频输入；同步 body 读取与重试共用「该模态 fetch timeout×2」的 deadline，慢速滴流不再长期占用 renderer 唯一 worker；超时返回 HTTP 422 并带源字段 `audio_url`/`video_url`。回归测试在 base 上失败、修后 15 passed；pooling server 集成测试本地无硬件、待 CI。延续第 24 条安全边界。
- **`#69677` [KV Offload] Share replicated MLA KV per host**：多云主机 TP 下 CPU offload 每 rank 存一份完全复制的 MLA KV；每 host 首 rank 写共享副本、本地 rank 均可恢复。TP8 跨两台 4 卡主机 **8 份→2 份**，同 16 MiB 容纳 512 chunk（原 256）；两 H100×2 主机上 4 rank 恢复字节一致、DeepSeekV2 两 prompt 各重载 512 token 输出 48 token 全一致；179 单测通过。
- **`#68357` [Bugfix][KV Connector][Mooncake] Keep extensible KV with RDMA-capable peermem allocations**：复用 allocator 的逐设备 VMM GPUDirect RDMA 查询，在支持设备上放开 `WITH_NVIDIA_PEERMEM=1` + extensible KV（不支持设备启动即报错，未设时保持 DMA-BUF 默认）。两节点各 8×H20 + RoCE + 4 HCA，DSv4.1-Flash TP8 PD，Mooncake CUDA 13 0.3.14。
- **`#69814` [Core] Simplify KV cache planning**：清理 `kv_cache_planning.py` 死代码（未使用的 ungrouped 内存检查、`may_override_num_blocks`/`saved_override` 间接层、`available_memory` 包装），并修 `--num-gpu-blocks-override` 泄漏进 `_init_minimal_kv_cache_for_profiling`；`#68248` 修 AFMoE BF16 router 精度（router 权重/投影保持 BF16、选专家时提升 FP32，修正 bias 仍 FP32）与 output-head 量化排除前缀——HellaSwag 512 题 NVFP4 归一化 333→335/512、原始 267→262/512，作者明示该混合结果**不能证明质量等价**。
- **`#69604` [Frontend] Allow only the Omni shared span among overlapping multimodal placeholders**：统一 Python/Rust 前端重叠规则（仅不同模态共享单一 span 且 `is_embed` 掩码互斥才允许；其余 400 / `INVALID_ARGUMENT`），修 Qwen2.5/3-Omni `use_audio_in_video=True` 从 `/render` 到 `/generate` 自相矛盾；Rust 2904 测试通过、Python 43 passed。`#69642` 修 Nemotron Shortcut 分桶 refit 误拒、视频帧按训练规格 448×448（原 1036×588 → 视觉 embedding 777→256）。`#69816` 修重复 kv quant mode int enum（正文为空，标题级）。
- **结构性/标题级**：`#249e6529` `[Core][PP]` 空闲空步后接纳新请求并按 backlog 定 prefill microbatch；`#dd083855` KV 规划简化；`#3332daf8` FlashKDA 升级（variant 132→9）；`#67340`/`#67306`/`#67382` ROCm Muse Spark DSpark FP8 draft/native routes/Markov 采样融合；`#64881` XPU MLA decode tensor-descriptor Q load；`#69864` Rust 前端默认限制 Qwen3-VL 每 prompt 32 视频；多条 Rust frontend/XPU/CI 变更。

---

## 3. SGLang：2026-11-25 提交（55 条）—— 请求表宽静默越界(#52926/#53185) + Kimi-K3 PDL 竞态(#52627) + Mooncake 恢复带宽 +43~58%(#52906) + PD 状态竞态三连

### 3.1 头号事件 `#52926` / `#53185` —— 见第 1.1 节

### 3.2 `#52906` [Perf] linker: Bound the ranges per Mooncake ranged get（官方一手，perf，第 17 条 KV 分层）

`MooncakeDirectLinker._range_get_reads` 原把每层 group 的 key 精确切成 `linker_load_threads` 个 ranged get，单次 get 随批量增长；Mooncake ranged get 对 range 数超线性变慢，长前缀大批量（每 rank 数十 GB）带宽骤降。修法 cap 每次 4096 ranges、拆更多 read 而非更大 read（线程数、读取内容与顺序不变）。MiniMax-M3.1-Flash on B300、PD + Mooncake direct linker、TP2/角色、DSpark、多轮闭环：每 rank 恢复带宽 set1 **29.2→41.9 GB/s（+43%）**、set2 **26.7→42.2（+58%）**；>20GB 批量 **16.5/15.1 → 42.8/43.2 GB/s**；6 分钟轮次 forward 等层加载 98/124/123→82/95/91 s（set1）、144/145/128→73/85/71 s（set2）；req/s 六轮全胜、均值 +5.6%；137 单测通过。

### 3.3 `#53024` [Perf][NVIDIA] DSA: Enable DCP top-k merge for DCP groups of 2 and 4（官方一手，perf）

`DcpTopKMerge` 原仅 DCP=8（megakernel 16）启用，DCP=2/4 走 `dcp_global_topk`（all-gather + 稠密 score + 确定性 top-k）。修法放宽到 2/4 ranks，poll mask 扩 64 位（每 rank 最多 40 candidate words/thread，32 位只覆盖 8-rank），2 ranks 每 rank 1280 candidate、4 ranks 640。GB300 16 卡 K3 decode-only（TP16/DP4/DCP4/EP16、MegaMoE、75K 上下文、5 draft）：step 128/256/384/512 总批 40.2→38.7 / 53.4→50.8 / 66.3→62.0 / 80.0→74.6 ms（**快 4–7%**）；2/4 ranks 各 7 测试通过，8 ranks 本次未重跑。

### 3.4 PD 状态竞态三连（官方一手，fix，第 5/8 条）

- **`#52976`**：`CommonKVManager.update_status` 读-改-写无锁；prefill 的 bootstrap 线程写 `WaitingForInput`、scheduler 线程写 `Bootstrapping`，decode 元数据在 sender 创建期间到达时可被后者覆盖，sender 永久轮询 `Bootstrapping` 直到 5 分钟 bootstrap timeout。加 `_status_lock`（失败通知仍在锁外）；复现测试 main 3/3 失败、修后 3/3 通过。
- **`#53037`**（`#52976` 后续）：sender/receiver 的 `clear()` pop room 未持锁，已读到非 None 状态的线程可在 pop 后写回「复活」房间；后续 decode retraction/recompute 复用同一 `bootstrap_room` 时命中陈旧 opening status、`max()` 保留旧值、跳过 deferred-ACK/staging/send-record 重置；PP 下 staging sync 还会在检查后写回 held key 造成泄漏。补锁后两类竞态测试 main 3/3 失败→修后 3/3 通过；`clear()` 开销 689→790 ns（无 PP）/1095→1290 ns（PP）。
- **`#52654`**：transfer 已 drain 但 scheduler 尚未 poll 时到达的 abort 把 `Success` 覆盖成 `Failed`（`#49349` 后续）；修法在 `poll()` 将报 Success 时提前返回，失败仍是唯一吸收态。`abort_all` 每 2s 的复现中 455 次 abort 命中 2 次；CPU 单测 32+23 passed，无 router 可复现触发，作者明示「若所有取消都应为 Failed 请关闭本 PR」。

### 3.5 正确性/健壮性与 AMD 后端（官方一手）

- **`#52948`**：DeepGEMM MegaMoE 编译期上限 2048 routed experts / 512 per rank（sgl-deep-gemm 0.2.9），Kimi-K3/K3.1 896 experts 在 `--tp-size 1 --moe-a2a-backend megamoe` 下每 rank 896，原来首次 MoE forward 断言或 OOM；改为建模型时抛 `ValueError`。VR200 真实权重：main 41s OOM、修后 42s 快速失败（读权重前）。
- **`#53150`**：`#48414` 把 hybrid MLA+KDA 绑定 view 换成 decode-only，致 Kimi-K3 megakernel 的 DSpark target verify 契约检查失败（缺 `tokens_per_request`/`verify_scratch_indices`/`aux_hidden_states`/`intermediate_conv_window`/`intermediate_ssm`）；补回默认字段。CPU 70 passed；装此 PR 后 sgl-nvidia-mega 44 passed、否则 14 fail；**尚无 GPU serving 验证**。
- **`#53107`**：`#50804` 删除 `act_quant` 参数后 `#46784` 的 K-cache fill 仍传 `act_quant=None`，DSA MTP draft prefill 直接 `TypeError`（base-b-test-1-gpu-large 每 PR 失败）；CI 状态：Base/AMD 进行中、Extra 跳过。
- **`#50299`**：DFLASH/DSpark 显式 `--speculative-num-steps != 1` 从 warning+静默改 1 改为启动 `ValueError`（macOS CPU 验证 2/0/−1 均报错，CI Base 跳过、AMD 告警）。
- **AMD Inkling 2 on MI35x 栈**：`#52726`（attention kernels）/`#52727`（MoE routing + expert GEMM）/`#52728`（norm/sconv/MTP kernels），均 [Feat][AMD]；`#52641` 固定 ROCm 10.0 Triton 以修 masked loads into LDS；`#47278` gfx950 64-head fp8 sparse MLA split-K decode plan；`#51506` HiSparse swap-in copy kernel 在 B300 64–256 请求时快 4–5%（其余区间噪声内）。
- **其他**：`#50804` srt 模型/层静默失败清理与死代码裁剪（2/4，正文 33K）；`#52969` rust-server launch core 分区留在 server 实例；`#53019` CI 分分区预取 HF 模型且 token 仅由可信 step 持有；`#53050` Kimi-K3 online EPLB 专家迁移；`#52897` diffusion Anima LayerNorm + timestep modulation 融合进 graph replay。

---

## 4. TensorRT-LLM：2026-11-25 提交（39 条）—— BREAKING 媒体解码默认值(#22334) + 预嵌视觉 KV 误图(#22561) + encoder replay 行(#22742) + CPython 栈帧预留(#22396)

### 4.1 `#22334` —— 见第 1.4 节

### 4.2 `#22561` / `#22742` —— 见第 1.2 节

### 4.3 `#22396` [perf] Run the executor loop under a CPython frame-stack reserve（官方一手，perf，非典型瓶颈）

CPython ≥3.11 把 frame 放在每线程 data-stack chunk：热调用恰在 chunk 边界时每次调用映射新块、返回时 unmap（gh-142183）；GB200 上 NCCL peer-map 后每次 `munmap` 还触发 Grace SMMU invalidation，**MiniMax-M3 piecewise CUDA graph prefill 约一半 host 时间耗在 munmap**。新增 `run_with_frame_stack_reserve()`（把 `co_stacksize` 改为 65536 的副本）包住 `PyExecutor.event_loop`，仅对受影响版本（3.11 ≤3.13.13、3.14 ≤3.14.4；3.12 官方不修）启用，`TLLM_DISABLE_FRAME_STACK_RESERVE=1` 可关闭。**注意**：运行时改 code object 依赖 CPython 内部实现，backport 版本号若不同则门控失准（评审意见，已在正文记录）。

### 4.4 其他要点（官方一手/描述级）

- **`#22890` [fix]**：wide-tactic FP8 MoE 检查误用 router 量化的输入（值达 2^11）使 SiLU 饱和、随机三行失败率 5–9%；改用独立 seeded BF16 输入（rows×2560，seed=19*rows+offset），router logits 仍取原 x；作者报两块 GR100 + B300 通过、失败率 3/2048，但 coderabbit 判 **needs follow-up**（宽度 2560 硬编码、test-list 未核验、head CI 未确立）。
- **`#22878` [fix]**：executor 构造失败时释放 host-resident 表（Qwen4Exp FP8/NVFP4 各加 4 行 `release_host_resident_weights()`），CUDA 同步失败则跳过释放；CPU 单测覆盖，`_close_ngram_table` 幂等性未验证、真实 hook 的 head CI 为 ABORTED，覆盖判 needs follow-up。
- **`#20841` [feat]**：n-sample 请求共享前缀 FMHA（opt-in `sampling_shared_prefix`，Triton shared-prefix/private-tail attention、KV 写、RoPE、state merge，含 planner/cache-key/CUDA graph 适配）；性能链接指向 `claude.ai/artifact`，评审明示变更超出该特性范围、无测试结果、**correctness/API/性能均未验证**（needs follow-up）。
- **`#22787` [fix]**：GLM-5.2 TEP4 MTP1 GSM8K 差异阈值 1.2→2.0（放松判定）；`#22883` [doc] Qwen3.5 GB300 AgentX + remote KV cache；`#22765` [doc] 限定 Rubin NVFP4 KV 硬件支持；`#19098` [feat] 轻量逐池 KV 利用率日志；`#22322` 保持 Kimi SM107 FP32 split-K 累加；`#22479` MiniMax M3 fullgraph context warmup；`#22570`/`#22619` recurrent-state 检查下沉进 cache manager；多条 waive/infra/CI 变更。

---

## 5. llama.cpp：2026-11-25 提交（44 条）—— 1.1.0 版本 bump(未发版) + Metal 图分块提交(#32947) + DSv4.1 indexer K 原地读 +15%(#32795) + ON_DEVICE 状态恢复崩溃(#30318)

### 5.1 `#33043` 版本 bump —— 见第 1.4 节（**tag/release 均 404，不记为发布**）

### 5.2 `#32947` metal: commit the graph in chunks with one command buffer thread（官方一手，perf/内存重叠）

默认 `n_cb=1` 时旧路径先编码前 10% 节点并提交、其余 90% 编码完才提交，GPU 在后 90% 编码期间空等数 ms/token（Inkling 2 Mini decode 图约 7k 节点）。改为每 256 节点一个 chunk、边编码边提交（在可融合组边界切分，融合行为不变；abort callback / capture / `n_cb!=1` 走旧路径）。M3 Ultra、Inkling 2 Mini IQ4_NL tg128 31.1/31.0→31.4/33.8 t/s，Qwen3.8-27B Q4_0 tg128 34.84–34.96→34.92–35.03，Qwen3.5-0.8B Q4_K_M tg128 263.0/263.5→278.8/275.6；pp512 基本不变；4×512 token KLD mean 0。**研判（观点）**：收益小而稳定，主要价值是给 `#33050` 回退 iPhone 默认 command buffer 数铺路——默认值治理的正面样本（第 16 条）。

### 5.3 `#33034` metal: spread the short mat-vec lanes over src1 rows（官方一手，perf）

K<32 的短 mat-vec 核在 src0 行少时 32 lane 多数空闲；把 lane 摊到更多 src1 行（对应 Inkling 2 depth mix）。Inkling-2-Mini UD-Q4_K_XL on M5 Max：pp512 **888→1052 t/s**、pp2048 771→894、tg128 49.7→50.0；PPL 与贪心输出逐位一致，MUL_MAT 新增小 K 用例通过。`#33050` 随即回退此前 iPhone 默认 command buffer 数变更（`#31815`），理由正是 `#32947` 已解决等待。

### 5.4 `#32795` deepseek41: read quantized indexer K in place when fused_idx is set（官方一手，perf，量化路径）

indexer K 与 KV latent 同池行 `[latent | indexer tail]`，量化 KV 时图原先无谓 cast 到 F32、每 ubatch 复制整段 tail；自 `#32650` fused indexer 已成稀疏常态且 CUDA 核原生支持 q8_0 跨步读取，改为 `fused_idx && quantized` 时不 cast。DSv4.1-Flash、q8_0 KV、ctx 65k、4×RTX 4000 Ada：PP/TG 在 ~7k/16k/28k/48k prompt 分别 +2%/+2%、+4%/+4%、+9%/+8%、**+15%/+15%**；decode 从 7k→48k 的跌幅由 −15% 收窄到 −5%；显存不变（scratch 流量）。**注意**：小样本冷启动、特定硬件/模型形状。

### 5.5 其他要点（官方一手）

- **`#30318` llama: fix seq state restore with ON_DEVICE（fix，崩溃）**：ON_DEVICE 恢复路径在 try 外读 magic/source seq_id，无效或截断 header 的异常可逃出 C API 触发 `std::terminate`；缺失 stored device state 触发 `GGML_ASSERT` 直接 abort。修法把 header 校验移入 try、断言改成可恢复异常。5 类坏输入（含缺 device state）修复前 **146/150 模型 `rf failed: unhandled exception`**，修后 CPU `ctest` 72/72、`rf` 150/150，CUDA RTX 4070 三项各 150/150。
- **`#31898` ggml: allow reserving scratch memory in graph_optimize**：`ggml_backend_graph_optimize_params` 增 `add_scratch`，允许后端为多节点共享数据预留 compute buffer（在节点 `first` 前分配、`last` 后释放），用于 src1 激活共享量化、`mul_mat_id` 共享专家路由等，并使其计入 `--fit`。`#30685` ggml 允许输入异步拷贝；`#33002` ggml-openvino 修 CI op 测试与 4D KV cache、更细 KV 长度取整改善 GPU decode（无数字）。
- **标题/描述级**：`#32901` CUDA FA 数据加载重构与去重（FP16 路径为主，逐卡微基准多数 ±1%）；`#32971` OpenCL clc coopmat MoE kernel（q4_0/q4_1/q4_k/q5_0/q5_1/q5_k/q6_k/mxfp4）；`#31871` SYCL F8 e4m3 初支持 SET_ROWS/CPY/ADD/MUL_MAT/MUL_MAT_ID；`#32960` hy_v4_flash DSA iSWA cache；`#33053` Inkling 2 indexed bias；`#33055` Inkling-2 LoRA 张量名映射；`#32948` Inkling2 graph 简化 mix/control；`#5917b39f`/`#2d5b3ddde` Inkling-2 LoRA/indexed bias 相关提交；多条 UI/构建/文档变更。

---

## 6. 趋势研判与累计结论

累计结论继承第 1–25 条（见上一期档案第 6 节），**本期新增第 26 条**；本期铁证按既有条目归类并强化如下：

1. **【新增第 26 条】请求长度边界 × 每请求表/行宽不一致，使投机解码越界写污染同批无关请求**：SGLang `#52926`（`req_to_token` 行宽小于 decode 写入 reach，尾部落入下一请求行）与 `#53185`（注意力 page table / CUDA graph table / kv-index buffer 按 `model_config.context_len` 而非行宽分配，多后端尾部覆盖或 Ascend `copy_` 抛错）。成立条件：spec decode + 接近长度上限 + context_len 非 kernel block 整数倍；观察指标：同批无关请求 logprob/输出漂移、Ascend 启动/运行异常；停止条件：后端统一取 `req_to_token_pool.max_context_len`、行宽由 `get_req_to_token_headroom()` 单一链路给出；验证动作：在 c≈长度上限、page_size=1、EAGLE topk>1/DFlash 下做 batch-invariance 与跨请求污染回归。
2. **静默正确性（第 4/5/8/19/22 条）继续最高频，跨节点/异步 KV 与 PDL 时序是两大来源**：vLLM `#67074`（GSM8K 0.600→0.954）、`#67216`（重复 prompt 8/32→32/32）、`#69901`（多节点 DRAM reader 仅 rank0）、`#69774`（视觉 FA3 26.4% 元素不匹配）、SGLang `#52627`（PP8 50,340 forward 中 4 次非有限首 token）、TRT `#22561`（不同图像误复用 269 token KV）、`#22742`（replay 行错位）。
3. **第 13 条 KV 容量/共享治理扩到「跨主机去重」**：vLLM `#69677` TP8 两主机 8 份 MLA KV→2 份、同等预算 chunk 容量翻倍；`#69814` KV 规划死代码清理；与上期第 25 条「KV 下沉 NVMe」同属容量层治理。
4. **默认性能化与破坏性默认值（第 16/20 条）**：TRT `#22334` 图像/视频解码默认改变（显存 −75%、token 计数 −99.8%，但混合表示报错与 `media_io_kwargs` 移除为 BREAKING）；llama.cpp `#33043` 弃用 `llama_batch`、DFlash draft 拆分（版本 bump 尚未发版）。升级检查表必须同时覆盖输入表示与 API 兼容。
5. **host-sync / host 时间移出热路径延续**：SGLang `#52906` Mooncake ranged get 限 4096 ranges（恢复带宽 +43~58%、req/s +5.6%）；vLLM `#69669` 启动编译 1435→479、583→260 s；TRT `#22396` CPython 栈帧 chunk churn（MiniMax-M3 约半 host 时间在 munmap）；llama.cpp `#32947` Metal 图分块提交 + `#33050` 回退 iPhone 默认值。
6. **量化/低比特路径的「能跑通 + 数值等价」成为主流验收口径**：vLLM `#69738` NVFP4 KV 按 key position 转换（GSM8K 在标准误内、无吞吐数字、CI pin 旧 FlashInfer）、`#69736` skinny GEMM 逐表 ≥10% 门槛且误差等于 cuBLAS、llama.cpp `#32795` q8_0 indexer K 原地读 +2~15%。**观点**：绝不能把「功能打通」当性能收益，须分列功能/数值/吞吐三类结论。
7. **服务安全（第 24 条）延伸到媒体获取与配置可预测性**：vLLM `#69461`（媒体 body deadline、422 错误字段）、SGLang `#50299`（不支持的 spec steps 从静默改 1 变为启动报错）、`#52948`（MegaMoE 布局超限建模型时拒绝）、`#53019`（HF token 仅可信 step 持有）。

---

## 7. 可执行清单（面向工程落地）

1. **投机解码边界回归（新增，P0）**：所有启用 EAGLE/DFlash/DSpark/MTP 的服务，升级 SGLang `#52926` + `#53185`；升级前在 `context_len` 接近上限、`page_size=1`、批内混长下做「同批无关请求输出是否漂移」的 A/B，并显式检查 Ascend 的 `copy_` 异常。
2. **跨节点/异步 KV 正确性**：ROCm + MoRIIO READ + 混合精度 KV 升级 vLLM `#67074`；hybrid/Mamba 模型 + PD 升级 `#67216`（重复 prompt 一致性 8/32→32/32）；HiSparse+NIXL 多节点 TP 升级 `#69901`（CPU-only 验证，须在自有硬件复验）；Kimi K3 + DSpark + NixlConnector 升级 `#68742`。
3. **视觉/多模态确定性与缓存键**：batch-invariance 模式升级 vLLM `#69774`（视觉 FA3 num_splits）；Kimi-K3 图像 chat 升级 `#69584`（提示 token 与 checkpoint 处理器一致、KVV 13 视觉用例）；TRT-LLM 使用预嵌视觉 embedding 的服务升级 `#22561`（否则同文本不同图复用错 KV）。
4. **PD 状态机竞态**：SGLang PD 服务升级 `#52976`/`#53037`（5 分钟 bootstrap 超时与房间复活）/`#52654`（drained transfer 被迟到 abort 置 Failed，若业务要求一律 Failed 则先评估再升级）；Kimi-K3 PP8 升级 `#52627`（PDL 首 token 竞态）。
5. **吞吐/启动优化**：Qwen3 家族低并发解码评估 vLLM `#69736`（c1 TPOT −10~17%，c4+ 收益有限）；Kimi-K3.1-Pro 冷启动评估 `#69669`（583→260s）；GDN prefill 评估 `#69135`（16–32K 3.0–3.2×、E2E +2%）；Mooncake linker 大批量恢复评估 SGLang `#52906`；K3 DCP=2/4 评估 `#53024`（step −4~7%）。
6. **显存/容量**：vLLM 多主机 TP 的 MLA KV offload 升级 `#69677`（8→2 份）；llama.cpp Metal 大图升级 `#32947` 并评估回退 iPhone 默认值 `#33050`；DSv4.1 q8_0 + fused indexer 评估 `#32795`（长 prompt +15%）。
7. **破坏性变更与版本**：TRT-LLM `#22334` 上线前审计所有 media 请求的表示类型与 `media_io_kwargs`（格式混用会报错）；llama.cpp 1.1.0 **尚未打 tag/release**，不要在生产按已发布版本升级，先评估 `llama_batch` 弃用与 DFlash draft 方法拆分。
8. **谨慎采用（验证未闭合）**：TRT `#20841`（共享前缀 FMHA，评审明示无测试结果/needs follow-up）、`#22890`/`#22878`（head CI ABORTED/needs follow-up）、`#22396`（依赖 CPython 内部结构）、SGLang `#53150`（无 GPU serving）、`#53107`/`#50299`（CI 跳过/告警）、vLLM `#68357`/`#69901`（特定硬件/CPU-only 验证）、vLLM 官方博客 ERNIE 5.5 的接受长度与 indexer 加速（描述级、无端到端 tok/s 对照）。

---

## 8. 来源、数据质量与限制

- **一手来源（官方仓库/API/文档）**：vLLM 83 / SGLang 55 / TRT-LLM 39 / llama.cpp 44 条提交均经 `api.github.com/repos/<repo>/commits?since=2026-11-25T00:00:00Z&until=2026-11-26T00:00:00Z` 取得（全部 HTTP 200、单页取全无截断，最高 83 < per_page=100）；releases API 确认窗口内无新正式版/无新 prerelease；`releases/tags/v1.1.0` 与 `git/ref/tags/v1.1.0` 对 llama.cpp 均返回 404。**48 个关键 PR 正文全部经官方 issues API 抓取 markdown（vLLM 18 / SGLang 14 / TRT-LLM 7 / llama.cpp 9；GitHub core 配额 60→4，无 HTML 降级；仅 `#69816` 正文为模板占位，按标题级处理）**；`#69816` 正文为模板占位，按标题级处理。vLLM 官方博客全文 `https://vllm.ai/blog/2026-11-24-ernie-5-5-flash-preview` HTTP 200、`datePublished=2026-11-24` 核验（窗口前一天，已显式标注补录）。性能数字如无特别说明均来自 PR 正文/官方博客（作者实测）。
- **二手线索/未核验**：HN Algolia 窗口内 `vllm` 34 命中（多为「LLM」模糊匹配）、`sglang` 1（无关）、`llama.cpp` 2、`inference engine` 3、`TensorRT-LLM`/`KV cache`/`speculative decoding`/`paged attention` 均 0。引擎相关线索——2 分《Production Grammar-Guided Decoding in vLLM with Xgrammar Benchmarks》(`github.com/grfdsasdvef/vllm-constrained-decoding-cookbook`)、5 分《Run Inkling 2 Mini on Mac Faster Than Anywhere Else》(`subconscious.dev/local`)、9 分《Rust agent with an Inkling 2 Mini inference engine for Apple Silicon》(`github.com/onatm/rustling`)、2 分 Ask HN「I'm building an inference engine. Ask me anything」——均二手线索/未核验，未作为确定事实采纳。
- **未采集/无可用数据**：Reddit、YouTube、X 本期未采集；窗口内未核验到可确认的官方视频发布。
- **证据等级口径**：官方一手=官方提交/releases/issues API 正文、官方 PR 页面正文、官方博客；官方学习资料=vLLM 官方博客（ERNIE 性能为官方口径、未给端到端对照，标「描述级」）；二手线索=HN 故事与社区转述；未核验=社区 API/视频页不可达或无一手确认的内容。
- **描述级/未闭合项清单（本期）**：SGLang `#52926`/`#53185`（无端到端精度 A/B）、`#53150`（CPU-only + 无 GPU serving）、`#53107`/`#50299`（CI 跳过/告警）、`#52948`（单卡启动级验证）、`#52654`（无 router 可复现触发）、`#51506`（多数区间噪声内）；TRT `#22890`/`#22878`/`#20841`（needs follow-up / CI ABORTED / 无测试结果）、`#22787`（放宽差异阈值）、`#22396`（运行时改 code object、版本门控脆弱）、`#22742`（与 `#22038` 合并顺序未定）；vLLM `#69901`（CPU-only）、`#68742`（可能 K3 专属、DP→TP 未测）、`#69738`（无吞吐数字、CI pin FlashInfer 0.7.4）、`#68357`（特定 H20/RoCE 环境）、`#69461`（pooling server 集成测试本地未跑）、`#68248`（HellaSwag 结果混合、作者明示不能证明质量等价）、`#69816`（正文空）；llama.cpp `#33043`（版本 bump 但 tag/release 404）、`#32947`/`#33034`/`#32795`（形状/硬件特定微基准）、`#32901`/`#33002`（无端到端数字）、`#31898`（架构能力、无实测）。以上均按「官方一手但验证未闭合/描述级」处理，不作为确定结论。

---

> **复核说明（本期）**：GitHub commits API 与 releases API 均 HTTP 200、四家提交单页取全（最高 83 < per_page=100）；llama.cpp 1.1.0 仅见 main 的版本 bump 提交，tag 与 release 精确查询均 404，已据此避免误记为已发布。vLLM 官方博客发布日期为 2026-11-24（窗口前一天），系回溯补录并显式标注，以免与 2026-11-25 关注窗口混淆。累计结论继承 1–25、新增第 26 条。
