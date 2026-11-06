# 大模型推理优化每日热点档案

> 档案日期：2026-11-06（本自动化于该日执行）
> 关注窗口：2026-11-05（周四）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（issues API，markdown）+ HN Algolia（HTTPS，numericFilters 按 2026-11-05 epoch 窗口）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 9 / SGLang 8 / TRT-LLM 9 / llama.cpp 8 个 PR 正文已抓取（含 #63231/#64095(标题级)/#65812/#65598/#65237/#65631/#65124/#64880/#51808/#62759、#48734/#46928/#48421/#48452/#48588/#48622/#47815/#48082/#48501/#47960、#21597/#21684/#21525/#20498/#21319/#21527、#31683/#31615/#31296/#31657）；releases API 确认窗口内**无新正式版本**（llama.cpp 仅推进 nightly `b12135–b12138`，prerelease），vLLM 仍为 v0.33.0(rc3)、SGLang v0.5.23、TRT-LLM v1.4.0rc2；HN 窗口内引擎相关唯一高信号为 **KVGuard**（SNU compsec 团队发布的 KV cache 安全检测器，关联 Cloudflare GLM 泄露，arXiv 2611.06138 + 博客，二手线索/未核验）。vLLM 提交计数命中 per_page=100 上限，page2 返回 0，按 API 窗口实计数 100 记录（未做 GitHub 网页交叉核验，记为 API 窗口计数）。

## 1. 版本状态与本期头号事件：vLLM 默认 `--block-size` 16→64 + 静默正确性/默认值风险双主线延续

**提交计数（2026-11-05 UTC，均 HTTP 200；2026-11-05 为周四工作日，vLLM 100 较上一窗口 11-04 的 111 略降、SGLang 90 较 62 显著回升、TRT-LLM 30 较 36 降、llama.cpp 43 较 22 回升）**：vLLM **100** / SGLang **90** / TensorRT-LLM **30** / llama.cpp **43**。

**发布状态（窗口内无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-11-02T10:10:05Z）**；11-05 全部 100 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-11-02T22:14:10Z）**；11-05 含 DSA DCP、HiCache、disaggregation、paged-experts 等密集修复/性能。
- **TensorRT-LLM**：仍为 **v1.4.0rc2（2026-10-28T15:51:34Z，prerelease）**，窗口内无更新；继续向 v1.4 收口（废弃 KVCM V1 / C++ transceiver、BREAKING 配置枚举化）。
- **llama.cpp**：仍为 **v0.9.0（2026-11-04T13:08:06Z 正式发布）**；11-05 仅推进 nightly `b12135–b12138`（21:13Z–22:07Z，prerelease），无新正式版本。

**本期头号事件（vLLM 默认 block-size 变更 + 跨引擎静默正确性/默认值风险双主线延续 + 量化新能力 + KV 卸载/PD 健壮性）**：
1. **vLLM 默认 `--block-size` 从 16 改为 64（`#63231`，强化累计结论第 3 条「默认值风险」）**：`CacheConfig.DEFAULT_BLOCK_SIZE` 改 64；ROCm 无显式 `--block-size` 时仍走原生 C++ paged-attention 内核（16/32 token 块），避免回退更慢的 Triton 路径。属「默认值迁移」——所有未显式指定 `--block-size` 的部署会一次性改变 KV 块粒度，首跑在 KV connector offload chunk、shared-storage connector hash、autotune cache-key、MI250 ROCm 25 分钟 hang 等处暴露了大量「16-token 假设」回归（#96922），修复后 Buildkite 171 jobs 全绿、MI250 从 timeout 到 8m51s。**这是本期最需运维感知的默认值变更**。
2. **静默正确性主线延续且 AMD 再添铁证（强化第 4/6/9 条）**：SGLang `#46928`（AMD `--use-aiter-unified-attn` 在 dense 模型 + 默认 page size 1 + CUDA graph 下 decode 产出垃圾、GSM8K→0、**无任何拒绝**、首 token 正常后续全错）是「异构后端静默错误」的教科书级案例；vLLM `#65812`（Qwen4Exp QSA 未加载 checkpoint FP8 KV scales，host K/V 恒为 1.0/1.0 致 scale assertion 失败）、llama.cpp `#31657`（Qwen2 NVFP4 FFN 漏接 scales → 加载但从未应用、`llama-sampler` 断言崩溃）；TRT-LLM `#21684`（causal_conv1d 卷积状态槽偏移改 64 位、拒绝超过 32 位的 batch stride，防截断溢出，强化第 9 条 int 溢出）。
3. **投机解码 / KV 写入正确性（强化第 5 条）**：SGLang `#48734`（per-step draft `out_cache_loc` 返回 strided 视图，Triton MLA KV write 按 flat offset 写，CUDA graph replay 下把 request 1 的 KV 写进 request 0 的下一步槽、自身从未写；`test_flashinfer.py` 自 #48005 起 max abs diff 43，已修为 contiguous 并在 `KVWriteLoc`/`unwrap_write_loc` 对非连续 loc 直接报错）。
4. **KV 卸载 / PD 解耦健壮性簇（强化第 5/8 条）**：vLLM `#65237`（lazy SimpleCPUOffload 的 store 回到 GPU 淘汰端 + 每步从 free-queue 头部走，DeepSeek-V4.1 sliding-window 组重算块 67.5%→59.7%）、`#65631`（sliding-window 组的 whole-chunk `max_load_tokens`/`max_num_new_tokens` 整块丢失修复）、`#65124`（P2P 消费端 `reset_cache` 排空时中止等待 prefiller 的入站 load，~35s 阻塞→有界 5s）；SGLang `#48452`（write-through HiCache 下 optimistic prefill 的 KV  checkpoint 被跳过、retry 重算整 prompt，改无条件 checkpoint）、`#48421`（HiCache 对 hybrid 模型拒绝 partial L3 prefetch 分配，防 recovery 点缺对应 hybrid state）、`#48606`/`#48491`（PD 退出经 RequestRetirer 统一释放）。
5. **量化新能力（强化第 6 条 + 量化后端迁移）**：vLLM `#62759`（新增 `--quantization lut_b_moe`：MoE 路由专家 LUT-B 在线量化 + MXFP8 激活，Rubin SM10.7 原生 grouped GEMM、其余 GPU 数值仿真，~3.14 bits/weight 逻辑）、`#51808`（online MXFP4 激活量化 key 覆盖；后端不支持的激活格式**显式报错**而非静默跑错格式）；llama.cpp `#31296`（CUDA 融合 NVFP4 gate/up + GLU MMQ，W4A4/W4A8，GB10 +12.9%~+28.1%、RTX 5090 +9.6%~+15.6% tok/s）、`#31615`（Metal 融合 DeepSeek-V4/V4.1 与 GLM MoE 路由）。
6. **TRT-LLM v1.4 收口：废弃与 BREAKING（强化第 3 条）**：`#21525`（声明 KV cache manager V1 与 C++ cache transceiver 自 v1.4 废弃，3 个月迁移期、行为不变仅加一次性告警）、`#20498`（BREAKING：checkpoint I/O policy 由 `str` 改为自有枚举 `CheckpointIOPolicy`，serve schema 同步）。

---

## 2. vLLM：2026-11-05 提交（100 条）——默认 block-size 迁移 + KV 静默正确性 + 跨层 KV 共享 + 量化新能力

### 2.1 `#63231` [Core] 默认 `--block-size` 从 16 改为 64（默认值迁移，强化第 3 条）
- **主题**：`CacheConfig.DEFAULT_BLOCK_SIZE` 16→64；显式 `--block-size` 仍优先（`user_specified_block_size=True`）；声明 preferred/restricted block size 的后端仍按原逻辑覆盖默认。
- **内容详情**：ROCm 无显式 `--block-size` 时保留原生 C++ paged-attention 内核（16/32 token 块），避免任何 16 的倍数都误路由到更慢的 Triton 路径。首跑 #96922 在 16-token 假设处全红：offloading-connector chunk 尺寸（48/16 vs 64 GPU 块）、shared-storage connector hash-count 期望、autotune cache-key 把 64 当「非默认」对照、MI250 25 分钟 ROCmAttn hang（engine 不返回 step）；全部修复，MI250 从 timeout → 8m51s，确认 hang 与块尺寸相关而非 flake。
- **应用场景**：所有 `vllm serve` 未显式 `--block-size` 的部署的 KV 块粒度默认值。
- **研判**：【事实】PR 给默认变更语义、ROCm 保留逻辑、Buildkite 171 jobs 通过、#96922 各回归项与 MI250 对照。**【观点】**属「默认值迁移风险」（第 3 条）——影响面极广（KV 容量、connector、autotune、ROCm 内核选择），所有未显式指定的部署一次性改变行为；运维升级 v0.33.x 后应主动核验 KV 容量与 connector 行为。【证据等级】官方一手（含 CI 与回归对照）。

### 2.2 `#64095` [Core][MLA] 跨层 KV cache 共享（注意力层，设计级）
- **主题**：为注意力层引入跨层（cross-layer）KV cache 共享。正文未抓取（issues API 返回空），仅标题级。
- **研判**：【观点】若落地，属 KV 容量/复用智能（第 8 条）的架构级新增，预期降低层间重复 KV 存储；但本期仅标题级、无正文与验证数字，记为**标题级/设计级**，不将其节省量当作已验证事实。证据等级：官方（标题级，正文未获取）。

### 2.3 `#65812` [Bugfix] Qwen4Exp QSA 加载 checkpoint FP8 KV scales（KV 静默正确性，强化第 4/6 条）
- **主题**：把 checkpoint 的 K/V scales 载入 Qwen4Exp QSA 的 FP8 KV cache 所用 host floats（从 #59209 拆出）；CUDA 融合 cache writer 与稀疏注意力 reader 用上 scales，ROCm 稀疏 reader 要求两 scales 均为 1.0 并在加载后正确拒绝非单位值。
- **内容详情**：验证用 `k_proj.output_scale=0.5`/`v_proj.output_scale=2.0` 的 safetensors。CUDA（SM100, fp8_e4m3）：修复前 device/host K/V=0.5/2.0 vs 1.0/1.0、scale assertion 失败；修复后均为 0.5/2.0、通过。ROCm QSA 在 CPU（PyTorch 2.13.0）：修复前 host K/V=1.0/1.0 reader 通过；修复后 0.5/2.0，reader 在 GPU dispatch 前抛出既有 `NotImplementedError`。
- **应用场景**：Qwen4Exp QSA + FP8 KV cache 的 scale 正确性。
- **研判**：【事实】PR 给 CUDA/ROCm 双对照（scale 加载前后）。【观点】属「FP8 KV scale 漏接」静默正确性（第 4/6 条）——host 端 scale 恒为 1.0 时数值错但可能仅在高精度场景暴露，属量化路径静默债。【证据等级】官方一手（含逐 scale 对照）。

### 2.4 `#65598` [Bugfix][Attention] FlashInfer TRTLLM prefill query offsets 存入持久缓冲（kernel 崩溃正确性）
- **主题**：修复 SM100（Blackwell）上 standalone draft-model（`method:"draft_model"`）prefill 在 CUDA graph replay 时的非法访存/segfault（源自 #60989 对 FULL CUDA graph 的捕获）。
- **内容详情**：修正后每个请求被归类为 prefill（token 数比 `reorder_batch_threshold` 多 1），SM100 走 TRTLLM prefill 路径；该路径 `build()` 把 `cum_seq_lens_q` 分配为新 tensor，但 attention metadata 在图外构建、FULL graph 保留了 capture 时地址，tensor 在 capture 后被释放并被复用，replay 时读到复用内存 → 错误 draft 或大规模 `CUDA error: an illegal memory access`。修复把该 buffer 改为持久。
- **应用场景**：SM100 + 独立 draft model + FULL CUDA graph 的投机 prefill。
- **研判**：【事实】PR 给根因（capture 后 tensor 释放复用 → replay 读脏地址）+ 报错来源 issue 链接。【观点】属「CUDA graph capture/replay 生命周期」崩溃型 bug（与往期静默/崩溃主线同源）。证据等级：官方一手（根因级）。

### 2.5 `#65237` / `#65631` / `#65124` KV Offload 正确性/健壮性簇
- `#65237` **[Bugfix][KV Offload] lazy SimpleCPUOffload 的 store 回到 GPU 淘汰端**——store 的块回到 GPU free-queue 头部、每步从头部走而非保存游标；含 #59383 两 commit 重基；DeepSeek-V4.1 sliding-window 组 + 争用 CPU tier 下重算块 67.5%→59.7%（仿真，32 此类块/缓存块）；Net -3 行（非测试）。
- `#65631` **[Bugfix][KV Offload] sliding-window 组保留 whole-chunk `max_load_tokens` 上限**——follow-up #55885；此前 prompt 末 `-1` 在整块上限之后施加，整块上限丢一块（1 块上限加载 0）；新测试（16-token chunk、65-token prompt）：上限 16/32/none 修复前 0/16/64 → 修复后 16/32/64。
- `#65124` **[Bugfix][KV Offload][P2P] 排空缓存重置时中止入站 load**——消费端（decoder）`reset_cache` 在 load 等永不写入的 prefiller 时阻塞 engine 线程至 load 超时 + abort-ack 窗口（PD ~35s）；现 `drain_jobs()` 在轮询前中止每个未中止的入站 load（同 #63948/#63988 生产者侧路径），排空仅等 acks（有界 5s `_ABORT_ACK_TIMEOUT_S`）。
- **研判**：【观点】三者共构「KV Offload 生命周期/上限/排空」的健壮性收口（第 5/8 条），与 #60181（NIXL host buffer 覆盖）同源；#65631 为对齐整块上限的边界修复，#65124 把 P2P 消费端阻塞从 ~35s 降到有界 5s。【证据等级】官方一手（#65237/#65631 含仿真/单测对照；#65124 设计级）。

### 2.6 `#62759` [Quant] LUT-B MoE + MXFP8 激活：Rubin 原生内核 + 仿真（量化新能力，强化第 6 条）
- **主题**：新增 `--quantization lut_b_moe`：MoE 路由专家的在线 LUT-B 量化 + MXFP8 激活；Rubin(SM10.7) 跑新的 native grouped GEMM（`tcgen05.mma ... lut::b`），其余 GPU 数值仿真以在无 Rubin 硬件下验精度。
- **内容详情**：MoE-only（dense 不量化）：3-bit 索引进 E4M3 codebook + 每 8×64 权重块的 UE8M0 指数（逻辑 ~3.14 bits/weight），MXFP8 激活（E4M3，每 1×32 一个 UE8M0）。原生 Rubin 路径权重保持 packed、由 tensor core 解码（`UTCQMMA.LUTB`）；重做 #51880（其 BF16 激活仿真 Rubin 跑不了）。
- **应用场景**：Rubin 上 LUT-B MoE 在线量化（3.14 bits/weight 量级）。
- **研判**：【事实】PR 给方案（3-bit 索引 + UE8M0、MXFP8 激活、Rubin 原生 vs 仿真）。【观点】属「MoE 低比特在线量化新格式」（第 6 条）——在 NVFP4/ MXFP4 之外新增 LUT-B 路径，需关注 Rubin 实测吞吐与精度；本期无端到端数字，记为官方一手（设计/方案级）。证据等级：官方一手（方案级）。

### 2.7 `#51808` [Online quantization] MXFP4 激活量化 key 覆盖（量化后端迁移，强化第 3 条）
- **主题**：online MXFP4 量化的激活 quant key 覆盖，例如 `*experts*` 设 `activation:null`（ROCm 走 AITER_MXFP4_BF16）或 `activation:"mxfp8"`；**所选后端无法计算的激活格式显式报错**而非静默跑另一格式（如 ROCm `mxfp8` 抛 `NotImplementedError`）。
- **内容详情**：同样，`fp8_per_tensor_static` 对 online 量化（无校准激活 scale）被拒绝。其余 MXFP4 以外的在线量化 weight key 仍抛 `NotImplementedError`。
- **应用场景**：online MXFP4 量化的逐层激活格式覆盖与 fail-fast。
- **研判**：【观点】属「量化配置 fail-fast」（第 3/6 条）——把「静默跑错格式」转化为启动期报错，降低量化静默错误风险；与 #65542/#65540 Marlin→Humming 迁移同属「量化后端/配置显式化」主线。证据等级：官方一手（报错路径级）。

### 2.8 `#64880` [Bugfix][LoRA] DeepSeek-V4 compressor / wo_a 适配器（LoRA 静默丢弃，强化第 6 条）
- **主题**：自 #53361/#55897 起 DeepSeek-V4 声明 `SupportsLoRA`，但 CUDA 路径上 `compressor.wkv`/`wgate` 与 `wo_a` 的适配器被静默丢弃（GEMM 直接跑 base weight，绕过 LoRA wrapper）；#57245 已修 V4.1，本 PR 把两层从 V4.1 移入 V4 共享一份。
- **内容详情**：三套 V4 CUDA 注意力后端（FlashMLA、FlashInfer SM100、FlashInfer SM120）上这些投影的适配器现均应用；无 LoRA 时逐投影 bit-identical（测试断言）；LoRA 启用时关闭 FlashInfer RopeQuant(SM100)。
- **研判**：【事实】PR 给根因（绕过 LoRA wrapper）+ bit-identical 断言。【观点】属「LoRA 适配器静默丢弃」（第 6 条）——量化/适配器路径漏接的同源静默债，fail-fast 优于静默无 LoRA 生效。证据等级：官方一手（含 bit-identical 断言）。

### 2.9 本期其余 vLLM 要点（标题/正文级，未逐条深采）
- **默认/配置**：`#63231`（见 2.1）。
- **新特性**：`#64095`（跨层 KV 共享，设计级）、`#64689`（MoE 可选共享 routed-expert 行恢复器，vLLM-Ascend 等后端可恢复 token 布局而不替换 `capture()`，生产改动仅 `routed_experts_capturer.py`）、`#65088`（Ling3 SwiGLU clamping 修复，标题级）。
- **性能**：`#60895`（GLM-5.3-Flash DeepGEMM FP8 MegaMoE 融合共享专家）、`#64898`（Qwen4Exp 融合 HyperConnection up proj + gate mix，长 prefill）、`#63956`（DSA token-sharded sparse-MLA prefill 去 indexer row sharding）、`#64510`（SM90 sparse FlashMLA 原生小 head 内核）、`#64187`（EP 权重过滤 EPLB-aware）、`#65285`（缓存 parked async KV load 准入预留）、`#63891`/`#63888`（ROCm DSv4.1 DSpark fused Markov / 跳过 decode top-k -1 填充）、`#63730`（ROCm Kimi-K3 prefill GEMM AITER 调优）。
- **ROCm/异构**：`#63117`（AITER MLA KV cache dtype 取自组 spec）、`#64886`（Kimi-K3 ROCm 原地权重写后刷新派生权重拷贝，opt-in hook）、`#62725`（gfx942 MLA verify 过 4 token 用 Triton）、`#60487`（AITER FA chunked-context extend 注意力 sinks 计一次）、`#59124`（opt-in FlyDSL FP8 prefill attention for AITER MLA）。
- **调度/前端**：大批 Rust Frontend PR（`#65863`/`#65853`/`#65852`/`#65851`/`#65850`/`#65636`/`#65635`/`#65625`/`#65624`/`#65623`/`#65622`/`#65621`/`#65294`/`#65293`/`#64695` 等，多为 Responses API / 模板 / 指标兼容，非推理热点核心，记为前端工程潮）；`#65773`（等模型输出自旋廉价化）、`#63488`（IPC fallback 后恢复引擎内存预算）。
- **量化/模型**：`#65806`（MiniMax-M3.1 indexer QKV 层初始化 shard record，标题级 silent init）、`#65825`（GPT-OSS MXFP4 LoRA TP shard 对齐 base weights）、`#65552`（DSv4.1 MXFP8 linear test stub params_dtype）、`#65745`（Ricursive Ripple-65B 模型支持）、`#65412`（Granite4Vision strip padding tiles）。

---

## 3. SGLang：2026-11-05 提交（90 条）——投机 KV 写入正确性 + DSA DCP 精确 top-k + HiCache/PD 健壮性 + 性能

### 3.1 `#48734` [Fix] 投机 per-step draft `out_cache_loc` 连续化（投机/KV 写入正确性，强化第 5 条）
- **主题**：`per_step_draft_out_cache_loc` 返回 strided 行（`view(...).permute(2,0,1).reshape(steps,-1)` 仍是 view），EagleDraftWorker 把 step `i` 写到 `out_cache_loc[i]`；Triton MLA KV write 按 flat offset 读 `loc`，故 `bs*topk>1` 时某 draft step 把 request 1 的 KV 写进 request 0 的下一步槽、自身从未写；CUDA graph replay 下 padding 行的 KV 落在那里。
- **内容详情**：#48005 让 FlashInfer MLA draft step attend 自身更早 draft token，此前 eager/graph 都只读 prefix、错位写从未被读；`test_flashinfer.py`（`test_runner_mode_eagle_draft_cuda_graph_runner_cases`）自 #48005 起 main 失败（max abs diff 43）。修复返回 contiguous tensor，并删两处 workaround。新增 `KVWriteLoc`/`unwrap_write_loc` 对非连续 loc 直接报错。
- **应用场景**：EAGLE/MTP draft（`bs*topk>1`）+ Triton MLA KV write + CUDA graph 的 KV 写入正确性。
- **研判**：【事实】PR 给根因（strided view → flat offset 错位写）+ 自 #48005 起的单测 max abs diff 43 + 非连续报错护栏。【观点】属「投机解码 × KV 写入布局」正确性（第 5 条）——CUDA graph replay 放大错位写，是 prefix cache × 投机解码 交叉债的又一例。证据等级：官方一手（含单测复现）。

### 3.2 `#46928` [Fix][AMD] aiter unified-attn 在 page size 1 + CUDA graph 下 decode 垃圾输出（静默错误，强化第 4 条）
- **主题**：`SGLANG_USE_AITER_UNIFIED_ATTN=1` + dense 模型 + 默认 page size 1 + CUDA graph on 时 decode 产出垃圾；首 token（prefill）正确、其后每个 decode token 错误（贪婪下 `"The capital of France is"` 续成 `" a ( ( ( ( ( ..."` 而非正确句，GSM8K→0.00），**无任何拒绝、静默失败**。
- **内容详情**：`_apply_cuda_graph_metadata` 仅在 sliding-window 池或 `page_size>1` 时把 unified-attention page table 拷入 captured `cuda_graph_page_table`；page size 1 时未拷，`unified_attention` 读到全零 page table。sliding-window 模型与 Qwen3-VL（page size 16 override）不受影响；非 CUDA graph 路径直接构建 page table 不受影响；SHUFFLE 5D KV layout 要求 page size 可被 8/16 整除、永不跑 page size 1。AMD CI 与 cookbook recipe 此前未覆盖该组合。
- **应用场景**：AMD aiter unified attention + dense 模型 + page size 1 + CUDA graph。
- **研判**：【事实】PR 给根因（page table 未拷入 captured buffer → 全零）+ 复现（贪婪续写 garbage、GSM8K→0）+ 不受影响路径枚举。【观点】属「异构后端静默错误」教科书案例（第 4 条）——无任何报错、GSM8K 直接归零，是 AMD aiter 路径上继 #46528/#45486/#46833 之后的又一条静默正确性铁证；用户开启 aiter unified-attn 必须显式避开 page size 1 + CUDA graph 组合，或升级含本修复的版本。证据等级：官方一手（含复现与根因）。

### 3.3 `#48421` / `#48452` HiCache / disaggregation 健壮性（强化第 5/8 条）
- `#48421` **[Fix] HiCache 对 hybrid 模型拒绝 partial L3 prefetch 分配**——cache-mode L3 prefetch 一直 `allow_partial=True`；host 容量不足时 allocator 返回更短 KV 前缀、调用方缩短 `storage_hit_count` 与 hash 列表；对 hybrid 模型，SWA window 与 Mamba checkpoint 相对 recovery boundary 存储，截断后的 KV 前缀可能选到无对应 hybrid state 的 recovery 点。修复：当 `info.comp_xfers` 含 hybrid component 时要求分配整个 storage hit（用完整 hit 长度作 `min_tokens`），容量不足走既有 `host_capacity` revoke + `poll_miss` 重试；新增 Unified Radix Cache storage 回归测试（8-token hit、容量仅 4 token，验证无缩短 transfer 入队）。
- `#48452` **[Fix] disaggregation 在 write-through cache 下发布 yielded optimistic prefill 的 KV**——optimistic prefill 在 decode peer 未 bootstrap 完时 yield，`optimistic_release_and_requeue` 在释放前把 KV checkpoint 进 radix tree；write-through HiCache 下该 checkpoint 被跳过（pending-bootstrap KV 在 write-through 下保持 request-owned），yielded 请求无 bootstrap 可延迟，跳过 checkpoint 只丢其超出已发布 prefix 的计算——device 副本被驱逐后 retry 重算整 prompt。修复：`optimistic_release_and_requeue` 现在无条件 checkpoint KV 进 tree（write-through 下 insert 备份、释放槽变可驱逐 tree 页、retry 的 storage baseline 覆盖整 prefix）。
- **研判**：【观点】两者共构「Hybrid 状态 / write-through KV 与 prefix 完整性」的 PD-KV 正确性收口（第 5/8 条），与往期 HiCache 静默损坏、PD KV 泄露安全主线同源；#48421 防 hybrid recovery 点缺 state，#48452 防 write-through 下 KV 丢失重算。【证据等级】官方一手（#48421 含回归测试；#48452 设计级）。

### 3.4 `#48588` / `#48501` DSA DCP 精确 top-k merge（DSA 索引器分片，强化第 7 条）
- `#48588` **[Feat] dsa: DCP decode 的精确 multicast top-k merge**——port #48414：DCP 下 Kimi-K3.1 把 DSA indexer 的 index-K cache 按 token 分片，每 rank 选 local top-k 再合并成 global top-k；本 PR 把 decode 行的合并 port 过来（一个 kernel 经 NVLink multicast 交换候选并精确选 global top-k），带测试、无 caller。相对 raw port：丢弃 `split` verify 模式、用 `DcpTopKMerge.create(group, max_rows, topk, dcp_rank)`（不能跑时返回 None、断言 group rank 即候选编码的 rank）、`-0==+0` 与 #48501 一致。
- `#48501` **[Feat] dsa: DCP-sharded indexer 的内核前置条件**——port #48414 的 DSA indexer index-K 跨 DCP rank 分片所需内核（存本 rank 的 index-K slot、选可跨 rank 精确合并的 local top-k、为短 per-rank 长度廉价构建 FlashMLA split-KV schedule），带测试、无 caller；tie 顺序与 cluster path 复用 #47960（其 cluster path 漏掉 top coarse bin 的 tie）；`schedule`/`dcp_size`/`dcp_rank` 替换分支自有 flag。
- **研判**：【观点】两者为 Kimi-K3.1 `--dcp-size>1` 在 main 落地的前置（当前 main 仍拒绝），属「MoE/索引器路由分片精度」（第 7 条）——精确合并保证 DCP 行选到与单 GPU 一致的集合；#47960 提供 `SGLANG_TOPK_V2_MODE`（fast/exact/deterministic，deterministic 按最小 index 与 `-0==+0` 保证每跑一致）作 tie 语义基础。【证据等级】官方一手（port/内核前置级，无 caller 故记为设计/内核级）。

### 3.5 `#48622` [Perf] DSv4.1 调 DeepGEMM paged MQA API（性能回退修复）
- **主题**：#48424 把 sgl-deep-gemm 升 0.2.6，0.2.6 的 SM100 paged MQA API 仅 varlen；DSv4.1 c1/c2 indexer 仍用 0.2.3 签名（位置参数 `clean_logits`+`logits_dtype`）且未传 `indices`，DeepGEMM shim 把每个 decode 调用路由到冻结 legacy kernel。在 DSv4.1 decode benchmark（VR）上使 paged MQA logits 每调用 +20%（bs64）/+30%（bs256），相对 10-30 栈。
- **内容详情**：`PagedIndexerMetadata` 在 SM100 + bf16 index weights + ≥96 query rows 时构建 varlen schedule（每 token 一行、per-row 长度、request ids 作 `indices`）；低于此 legacy kernel 仍更快（bs8-16）。`paged_logits_args` 传 per-row 布局与 `indices` 给 `fp8_fp4_paged_mqa_logits`。选中的 kernel 即 0.2.5 栈跑的 `sm100_paged_mqa_logits<...>`。精度：new vs legacy logits 逐位相同。
- **应用场景**：DSv4.1（SM100）decode 的 paged MQA logits 性能。
- **研判**：【事实】PR 给根因（API 版本错位致路由到 legacy kernel）+ VR benchmark（+20%~+30% logits/调用）+ 逐位相同精度。【观点】属「依赖版本错位致性能回退」（第 3 条默认值/配置风险的旁支）——一次依赖 bump 未同步调用签名即吃掉 20~30% decode 性能；升级 sgl-deep-gemm 须同步调用层。【证据等级】官方一手（含 benchmark 与逐位精度）。

### 3.6 本期其余 SGLang 要点（标题/正文级）
- **性能**：`#48082`（paged-experts 去 staging copy——专家不再经 pageable 暂存再 K 个/次填入 store，省一份与 store 等大的 RAM；例 Qwen3-Next-80B-A3B-NVFP4 在 RTX 5070 Ti + 62GB RAM：staging 44GB、19-26GB swap、load 177-352s，修复后显著下降）、`#47815`（SM12x FA4 graph replay 用 live KV lengths 分区，避免大 KV 容量 bound 致多数 split 空闲；新 GPU 回归未在 SM12x 实跑，记为设计级）、`#47960`（见 3.4）、`#44491`（Qwen2MLP 把 SiLU*mul 与 NVFP4 量化融合 down_proj）、`#44459`（AMD gfx1250 MLA decode 融合 RoPE/concat/KV-cache write）、`#45202`（Kimi-K3 为 mono fusion 准备 attention 接口）、`#48166`（L4 Triton fused MoE for Qwen3-30B-A3B）、`#47898`（CuTe DSL fused-a-gemm K 对齐 1024→256）、`#47683`/`#47678`/`#47676`（tiny GEMM SIMT、dflash 批量填 decode-prep、dsv4 padded decode replay 走 metadata glue graph）、`#42198`（AMD fused AR+RMSNorm 1-stage kernel）、`#41294`（DSv4.1 Hopper fused low-ratio compressor）。
- **DSA/DeepSeek**：`#46890`（DSA interleave-CP breakable prefill graph 扩到 GlmMoeDs…）、`#42718`（DSv4.1 DSpark compact verify 把 packed verify tokens 映射回请求）、`#40421`（DSv4.1 PD CP prefill 复用 dequantized compressed KV，7/N）。
- **调度/PD**：`#48606`（scheduler 经 RequestRetirer 释放 retracted/paused/chunked-aborted 请求）、`#48491`（PD disaggregation 退出经 RequestRetirer 统一释放）、`#48037`（disaggregation 跨 transfer pass 缓存 host-staged decode 准入记账）、`#47728`（dspark 在 batch size 1 为每个请求 stage 自己的 SWA ring slot）、`#48328`（mamba lazy boundary snapshot 独立 slot）。
- **AMD/异构**：`#48573`（aiter 升 gfx1250 pin）、`#48501`/`#47960`（见 3.4）、`#45601`（gfx950 允许 AITER prefill attention）、`#46928`（见 3.2）、`#34212`（AMD DSP 跳过未用 QuickAllReduce 分配）、`#48134`（AMD CI 修两处从未运行的注册）。
- **模型/其他**：`#39793`（xing4_0 模型支持）、`#37597`（glm_ocr 模型支持）、`#46964`（diffusion Hy-Nomas INT8 DiT + FP8 VAE）、`#47187`（MiniMax-M3 GB300 decode 走 trtllm_mha，prefill 留 fa4）、`#48066`（sgl-router opt-in fips，aws-lc-rs）、`#47442`（Cosmos 3 RoPE 表建在 GPU 上以对齐 CPU offload 输出）。

---

## 4. TensorRT-LLM：2026-11-05 提交（30 条）—— v1.4 收口（废弃/ BREAKING）+ KV 正确性 + int 溢出 + 确定性

### 4.1 `#21525` [chore] 废弃 KV cache manager V1 与 C++ cache transceiver（废弃/默认值风险，强化第 3 条）
- **主题**：声明 KV cache manager V1（`kv_cache_config.use_kv_cache_manager_v2=False`）与 C++ cache transceiver（`transceiver_runtime 'CPP'/None`）自 TensorRT-LLM 1.4 废弃，遵循 README 废弃策略；三个月迁移期、行为不变，仅在唯一构造根加一次性告警。
- **内容详情**：告警加在 `KVCacheManager.__init__`（KVCacheManagerCpp 唯一构建处）与 `BindKvCacheTransceiver.__init__`（CacheTransceiverCpp 唯一构建处）；显式值、'auto' 解析、fallback 全部经这些构造根；字段描述、1.4 release notes、根文档页同步声明废弃。
- **应用场景**：TRT-LLM v1.4 起 KVCM V1 / CPP transceiver 用户。
- **研判**：【事实】PR 给废弃范围、告警注入点、迁移期（3 个月）、文档同步。【观点】属「废弃/默认值迁移」（第 3 条）——v1.4 起 KVCM V2 成唯一路径、CPP transceiver 进入淘汰；沿用 v1.4.0rc1 已默认 KVCM V2 的路线，用户须规划迁移。【证据等级】官方一手（废弃宣告级）。

### 4.2 `#21684` [fix] causal_conv1d 卷积状态槽偏移改 64 位（int 溢出，强化第 9 条）
- **主题**：forward / general update / single-token update 三内核把 conv-state slot offsets 改 64 位算术；forward/update 拒绝负数或超过 32 位的 batch stride 而非截断。
- **内容详情**：单测新增跨 2³² 元素的 slot offset（三内核路径）与拒绝 2³² batch stride 的测试，结果与状态同 compact copy 比较、检查 32 位回绕区域未被触碰、跳过 <16GiB 设备；QA 判覆盖充分；CI/测试结果当前 head 不可用（记为描述级/未独立验证）。
- **应用场景**：causal_conv1d 在大 batch stride / 大 slot offset 下的正确性。
- **研判**：【事实】PR 给修复（64 位 offset + 拒绝超 32 位 stride）+ 单测覆盖。【观点】属「整型溢出」（第 9 条）——此前 32 位截断静默错（或越界），现显式拒绝；延续 int 溢出跨引擎主线。证据等级：官方一手（单测级；CI 当前 head 结果未提供，记为描述级）。

### 4.3 `#21597` [fix] 解聚传输中 mixed-precision 高层的 KV-head 数（KV 正确性，强化第 5/8 条）
- **主题**：#20709（per-view KV head transfer）让解聚 page-table builder 为每个内部 cache 层解析 KV-head 数；#21021（mixed-precision KV 用于投机解码 / sparse MLA 后端）在其后合入，把 mixed-precision 高层记在 manager 的 `mixed_precision_high_layers` 记录而非 virtual layer mapping，导致构建 mixed-precision KV cache manager 的 page table 抛 `Cannot resolve KV-head count for internal layer ...`。两 PR 各自过 CI，#21021 合入后 main 失败 `test_disagg_exports_both_pools_and_pairs_sink_tail` 与 `test_deepseek_v4_mixed_disagg_names_every_layer`。
- **内容详情**：builder 现在从 `mixed_precision_high_layers` 记录取该层所属层的 head 数；其余层按原方式解析；无 manager 改动。测试覆盖两个失败用例。
- **应用场景**：DeepSeek-V4 mixed-precision KV + 解聚传输的 page table 构建。
- **研判**：【事实】PR 给根因（#21021 记录位置与 #20709 解析假设错配）+ 两失败单测 + 修复 + 测试覆盖。【观点】属「mixed-precision KV × 解聚传输」正确性（第 5/8 条）——KV 格式/层级元数据错配致构建崩溃，是 KV 正确性主线的 TRT-LLM 侧延续。【证据等级】官方一手（含回归测试）。

### 4.4 `#20498` [refactor] BREAKING：checkpoint I/O policy 改字符串枚举（BREAKING/配置，强化第 3 条）
- **主题**：checkpoint_io_policy 由 `str` 改为自有枚举 `CheckpointIOPolicy`；serve schemas 同步。
- **内容详情**：收益为类型化输入/输出、合法 policy 名可追踪；CPU-only 测试覆盖默认值/字符串与枚举输入/公开导出/JSON 序列化/归一化/serve 与解聚 schema 的接受与拒绝。
- **应用场景**：TRT-LLM checkpoint 加载策略配置。
- **研判**：【观点】属「配置类型化 BREAKING」（第 3 条）——旧 `str` 配置在升级后需改枚举；属 v1.4 配置面清理。证据等级：官方一手（CPU-only 测试级）。

### 4.5 `#21319` [fix] 移除 DeepSeek-V4 metadata recompile 与不确定 mHC split-K（确定性/性能，强化第 9 条旁支）
- **主题**：修复 nvbugs/6993575（与 #21121 合入）：8×B300 上 DeepSeek-V4-Pro + DSpark 投机解码 + DeepGEMM MegaMoE 的 run-to-run 吞吐波动（512 请求、ISL1000/OSL2000；单次 commit 内 33,930–39,598 tok/s，限 5%）。两来源：(1) 服务期内编译的内核（空编译缓存机上，混 context/generation 的首批编译了 DeepGEMM MegaMoE 240-row token tile kernel——#21121 在 autotuning 预填，及本 PR 移除的 DeepSeek-V4 metadata helper 的 `torch.compile` 重编译）；(2) 不确定 mHC split-K（greedy 输出逐跑变化、末请求与 decode tail 逐跑变）。
- **内容详情**：commit 1 移除 DeepSeek-V4 position ids 的 `torch.compile` recompile；commit 2/3 修非确定 mHC split-K。
- **应用场景**：DeepSeek-V4-Pro + DSpark 的 run-to-run 确定性与吞吐稳定。
- **研判**：【事实】PR 给波动数字（33,930–39,598 tok/s，限 5%）、两来源、commit 分解（含 #21121 联动）。【观点】属「服务期内核编译 + 非确定 split-K」导致的 run-to-run 不确定（第 9 条旁支——非 int 溢出但属确定性债）；与 vLLM #65518（压缩 fatbins）同属「启动期/服务期编译稳定」议题。【证据等级】官方一手（含波动数字与 commit 分解）。

### 4.6 本期其余 TRT-LLM 要点（标题/正文级）
- **特性/修复**：`#21527`（Helix chunked prefill 收集跨 CP rank 的 cached MLA KV——把 bounded global history chunk 映射到 rank-local compressed-KV 区间后与本有变长 CP all-gather 组装，CUDA reorder 还原顺序、分离 compressed KV 与共享 positional-key 段；persistent KV 仍分片、FP8 记录本地 dequant 再通信；需 SM100+ chunked prefill，不支持的 dense Helix FP4-KV 在 dispatch 前拒绝）、`#21543`（移除 TensorRT 后端移除后的无用内核）、`#21680`（Qwen MoE native FP8 LoRA 参考测试配置修复）、`#21332`（GDN decode 移出 bounded prefill state workspace）、`#21192`（单 CP rank 时忽略 cp_type、多 rank 要求 HELIX）、`#21187`（存 metadata server URL 原样用、支持 etcd 凭据与签名 server）、`#21035`（beam-history copy 按生成 token 数而非 prompt 长度 spread 定尺寸）、`#21409`（QSA rows 与 GDN replay work items 的一启动 decode metadata）、`#21410`（EncoderExecutor 由共享 setup 构建而非 ModelEngine）、`#21417`（外部启动关闭时等 rank-0 VisualGen worker teardown）、`#21341`（从最终流文本丢弃 streamed stop-word ids）、`#21149`（chat streaming 在独立 chunk 发 finish_reason）、`#20483`（Visual Gen token-sharded TP 收集 FP8 block-scale 输入）。
- **基础设施**：`#21715`/`#21714`/`#21713`/`#21697`/`#21694`/`#21609`/`#21686`（多 nvbugs waive / ModelEngine 测试修复）、`#21488`（平台中立 LTX2 perf recipe 文件名）、`#21424`（strict_accept 每 batch 一个编译变体）。

---

## 5. llama.cpp：2026-11-05 提交（43 条）—— NVFP4 量化融合/修复 + KV shifting + 多后端 + 新模型

### 5.1 `#31296` [CUDA] 融合 NVFP4 gate/up + GLU 于 MMQ（量化性能，强化第 6 条）
- **主题**：扩展 #29948/#31188 到 NVFP4：融合 gate/up + GLU 的 MMQ kernel 现也跑 NVFP4 FFN（dense 与 MoE，`MUL_MAT_ID`）；两条 NVFP4 MMQ 路径 W4A4（原生 FP4 MMA，Blackwell 默认）/ W4A8（Q8_1 激活）。NVFP4 权重 scales（`.scale`，per tensor 或 per expert）在融合 write-back 中应用，gate/up matmul 后的 `MUL` 节点随之融合。FP4 激活的融合 kernel 仅 Blackwell 构建。
- **内容详情**：`llama-bench -ngl 99 -fa on -p 16384 -n 0 -b 2048 -ub 2048 -r 3`，W4A4 tok/s——DGX Spark(GB10) Llama-3.1-8B +28.1%、Qwen3.6-27B +21.7%、unsloth/Qwen3.8-27B-NVFP4 +18.1%、Qwen3.6-35B-A3B MoE +12.9%；RTX 5090 Llama-3.1-8B +15.6%、Qwen3.6-27B +11.7%、unsloth/Qwen3.8-27B-NVFP4 +9.8%、Qwen3.6-35B-A3B MoE +9.6%。更小 ubatch 提升更大。
- **应用场景**：NVFP4 FFN（dense/MoE）在 Blackwell/GB10 的 MMQ 吞吐。
- **研判**：【事实】PR 给方案（W4A4/W4A8 双路径、scale 融合）+ GB10/RTX5090 逐模型 tok/s 对照（+9.6%~+28.1%）。【观点】属「NVFP4 FFN 算子融合」（第 6 条量化性能）——把 gate/up matmul 与 GLU 融进 MMQ 显著提吞吐，是 llama.cpp 在 Blackwell NVFP4 上的关键优化。【证据等级】官方一手（含逐卡对照表）。

### 5.2 `#31657` [qwen2] 给 build_ffn 加 NVFP4 scales（量化静默崩溃，强化第 6 条）
- **主题**：qwen2 的 `build_ffn` 当前把 NVFP4 scales 以 NULL 传入；转为 NVFP4 的模型会因 scales 已加载却从未在 FFN 应用而崩溃（`llama-sampler` 断言 `found` failed）。attention 已应用 scales，FFN 漏了。
- **内容详情**：修复把 scales 传入 FFN；AI 使用披露 No。
- **应用场景**：Qwen2 系模型 NVFP4 量化加载。
- **研判**：【事实】PR 给根因（FFN 漏接 NVFP4 scales → 断言崩溃）+ attention 已应用对比。【观点】属「量化 scale 漏接致崩溃」（第 6 条）——与 vLLM #65812、TRT 量化静默债同源；fail-fast 崩溃优于静默数值错，但提示 llama.cpp NVFP4 路径仍需逐层核对 scale 应用。【证据等级】官方一手（根因级）。

### 5.3 `#31683` [llama] 修 KV shifting + 加测试 + 逐模型 build_rope（正确性，强化第 9 条旁支）
- **主题**：当前 KV shifting 跨模型处理基本雷同，但复杂 rope 配置的模型会失败；解法把逐模型 `build_rope` 移入模型类（类似 mtmd 已有 `build_rope_2d`），模型可扩展定制。明确为 stop-gap，作为下一步重构参考（见 #31693）。
- **内容详情**：AI 使用披露 yes；要求用户知悉 AGENTS.md/CONTRIBUTING.md（llama.cpp 限制 AI 生成内容）。
- **应用场景**：复杂 rope 配置模型的 KV shifting 正确性。
- **研判**：【观点】属「rope/KV shifting 逐模型正确性」——复杂 rope 配置下通用 KV shifting 失败，移入模型类修复；记为设计/修复级（stop-gap），非性能项。【证据等级】官方一手（修复/设计级）。

### 5.4 `#31615` [metal] 融合 DeepSeek-V4/V4.1 与 GLM MoE 路由（Metal 性能，强化第 4 条旁支）
- **主题**：Metal `TOPK_MOE` 融合此前只覆盖 plain-softmax + 下钳位 norm 路由，DeepSeek-V4/V4.1 层从未融合；本 PR 加三项：(1) 路由权 norm 可为 scale+bias（sum+1e-20）如 DSv4.1 构建，两变体共用一 kernel（bias 形式 clamp=-INFINITY）；(2) 锚定 softplus 节点的融合链（UNARY+SQRT+ADD+ARGSORT+GET_ROWS + 可选 norm/scale）使 DSv4/V4.1 路由在 Metal 融合，kernel 以 sqrt(softplus) 打分、对 selection-biased keys 排序，gathered 权重与 norm 保持无偏（bias 只影响选择，匹配参考图与 CUDA kernel）；(3) 锚定 sigmoid 节点的融合链使 sigmoid-gated 路由（GLM、DSv2/V3.2…）在 Metal 融合。
- **内容详情**：含 CPU/GPU 对照表（路由融合后延迟下降，具体数字截断未全呈现，记为方案/性能级一手）。
- **应用场景**：Metal 后端 DeepSeek-V4/V4.1、GLM 等 MoE 路由融合。
- **研判**：【事实】PR 给融合方案（softplus/sigmoid 双锚点、无偏权重保持）。【观点】属「Metal MoE 路由融合」（第 4 条异构/Apple 后端性能）——把 DSv4/V4.1/GLM 路由纳入 Metal 融合，提升 Apple GPU 上 MoE 效率；性能表尾部截断，记为官方一手（方案/部分性能级）。【证据等级】官方一手（方案级；性能表截断）。

### 5.5 本期其余 llama.cpp 要点（标题/正文级）
- **NVFP4/量化**：`#31296`（见 5.1）、`#31657`（见 5.2）、`#31623`（OpenCL 加 DSV4_HC_PRE/POST）、`#31587`（Vulkan 在 AMD 专有驱动 26.10.1+ 重启用大 coopmat matmul 配置）。
- **Metal/Apple**：`#31615`（见 5.4）、`#31688`（Metal 把可融合节点组留在同一 command buffer）、`#31679`（CUDA `get_mmvq_mmid_max_batch` 用最高编译 arch）。
- **KV/rope**：`#31683`（见 5.3）、`#31660`（graph 把每 recurrent 模型的 MTP conv 快照存于一个 set_rows）、`#31733`（llama 重构：把 model 移为 llm_graph_context 的 class member）。
- **模型支持**：`#31692`（加 Ripple ricursiveai 模型）、`#31589`（加 xing4_0 模型）、`#31611`（conversion 从源 tensor names 解析 quantized_layers）、`#31749`（models recursive fix）、`#31698`（mtmd/chatterbox 加 use_classifier_free_guidance 作 gguf metadata）。
- **后端/CI**：`#31591`（WebGPU 加 MEAN 支持）、`#31569`（fit：设 n_gpu_layers 时计入跳过 NextN 层）、`#31705`（ci 修 build-cmake-pkg ccache）、`#31717`（gguf-py 外部 chat templates 优先于 tokenizer_config）、`#31684`（rpc 与 common.cpp 对齐）、`#31712`（ui 防隐藏 sidebar 拦截点击）、`#31728`/`#31714`（server windows MCP 解析 / 拒绝非对象 session 更新）、`#31731`（vendor 升 cpp-httplib 0.65.0）。

---

## 6. 累计结论（继承 1–10，本期更新）

- **第 1 条（调度器隐式阻塞的定量画像）**：本期 vLLM `#65773`（等模型输出自旋廉价化）、SGLang `#48606`/`#48491`（RequestRetirer 统一释放）继续补强「释放/阻塞路径」；沿用，无新增铁证。
- **第 2 条（发版窗口的双向性：发版当天合入 ≠ 在版内）**：本期无新版本，沿用；llama.cpp 仍 v0.9.0（11-04），TRT-LLM 仍 v1.4.0rc2（10-28）。
- **第 3 条（默认值成为主要风险载体）**：**本期重大更新**——vLLM `#63231` 默认 `--block-size` 16→64（影响面极广的 KV 块粒度默认值）、TRT-LLM `#21525` 废弃 KVCM V1 / CPP transceiver（v1.4）、`#20498` BREAKING checkpoint I/O policy 枚举化、vLLM `#51808` 量化激活 key 覆盖 fail-fast；SGLang `#48622` 依赖版本错位致 +20~30% decode 回退也属「配置/依赖风险」旁支。**默认值/废弃/配置迁移仍是最高频风险载体**。
- **第 4 条（异构后端静默错误高频）**：**本期 AMD 再添铁证**——SGLang `#46928`（aiter unified-attn page size 1 + CUDA graph → GSM8K→0、无任何拒绝）；沿用往期 #46528/#45486/#46833（AMD aiter/CTA fp8/softmax scale 静默错误）；llama.cpp `#31615` 为 Metal 路由融合（Apple 后端性能）。**异构/AMD 静默错误持续为最密集主线之一**。
- **第 5 条（前缀缓存 × MoE 路由 × 投机解码交叉债务）**：SGLang `#48734`（投机 draft KV 错位写入，CUDA graph replay 放大）、vLLM `#65237`/`#65631`/`#65124` 与 SGLang `#48421`/`#48452`/`#48606`/`#48491` 的 KV 卸载/HiCache/write-through/PD 释放；TRT `#21597`（mixed-precision KV × 解聚）。沿用并强化。
- **第 6 条（量化路径静默错误，与异构后端并列的第二类高频静默错误）**：vLLM `#65812`（Qwen4Exp QSA FP8 KV scale 漏载）、`#64880`（DeepSeek-V4 LoRA 适配器静默丢弃）、llama.cpp `#31657`（Qwen2 NVFP4 FFN scale 漏接致崩溃）、vLLM `#62759`/`#51808`（LUT-B / online MXFP4 量化新能力与 fail-fast）；llama.cpp `#31296`（NVFP4 FFN 融合性能）。**量化静默债跨引擎延续**。
- **第 7 条（专家张量/路由归属与分页漏接静默错误）**：SGLang `#48588`/`#48501`（DSA DCP 精确 top-k merge 与 indexer 分片前置）、vLLM `#64689`（共享 routed-expert 行恢复器）相关；沿用，无新增静默错误铁证。
- **第 8 条（KV 容量/复用智能：容量自检、准入、传输/卸载正确性）**：vLLM `#65237`/`#65631`/`#65124`、SGLang `#48421`/`#48452`、TRT `#21597` 共构 KV 卸载/HiCache/write-through/解聚 正确性收口；vLLM `#64095`（跨层 KV 共享，设计级）为容量/复用架构新增。**KV 正确性仍是跨引擎最密集主线**。
- **第 9 条（长上下文 int 溢出为静默正确性新子项）**：TRT-LLM `#21684`（causal_conv1d 64 位 slot offset、拒绝超 32 位 stride）；llama.cpp `#31683`（复杂 rope 模型 KV shifting 失败，非 int 但属正确性债）。**int 溢出主线扩展到卷积状态偏移**。
- **第 10 条（量化/MXFP8 KV 正确性簇）**：vLLM `#65812`（FP8 KV scale）、SGLang `#46928`（AMD aiter 静默）等延续；沿用。

---

## 7. 社区侧（HN Algolia，2026-11-05 窗口）—— KVGuard：KV cache 安全检测器

HN Algolia（2026-11-05 窗口，epoch 1793836800–1793923199）引擎相关唯一高信号为 **KVGuard**（二手线索/未核验）：
- **「KVGuard: Catching the KV cache bug behind Cloudflare's GLM leak」**（2026-11-05T10:48:54Z，8 points，来源：SNU compsec 博客 `https://compsec.snu.ac.kr/blog/kvguard`）。
- **「KVGuard: A KV Cache Safety Checker for LLM Serving Systems」**（2026-11-05T10:22:21Z，3 points，arXiv `https://arxiv.org/abs/2611.06138`）。

**研判**：【事实】两条均为 2026-11-05 当日 HN 发布的 Show HN / 论文公告，指向一个 KV cache 安全检测器 KVGuard（SNU 计算机安全实验室），直接关联 2026-10-26 档案记录的 **Cloudflare GLM-5.3 Workers AI 跨请求 KV cache 暴露**安全披露。【观点】这与本期及往期「PD-KV 正确性升维至安全」主线呼应——社区开始工具化检测 KV cache 跨请求泄露/污染；但本档案**未核验 KVGuard 的实现、覆盖范围与对 vLLM/SGLang/TRT-LLM/llama.cpp 的可用性**，记为**二手线索/未核验**。【证据等级】二手线索/未核验（HN 公告 + 外部博客/arXiv，未独立验证工具效果）。

其余 HN 命中多为评论级或无关（vllm 20/llama.cpp 10/sglang 5 命中大量为评论 `None` 标题；`LLM inference`/`GPU inference` 各 20 命中以「LLM 网关/防火墙/工具/自托管前端」类为主，如「Inference business simulator」「Cerea self-hosted chat front end」「LLMbrake dollar caps」，均非引擎更新），TensorRT-LLM 0 命中。

---

## 8. 趋势判断与可执行清单

**趋势判断**：
1. **默认值/废弃/配置迁移仍是最高频风险载体**（第 3 条）：vLLM 默认 block-size 16→64 是本期最需运维感知的默认值变更；TRT-LLM v1.4 收口（废弃 KVCM V1 / CPP transceiver、BREAKING 枚举化）要求用户规划迁移；SGLang 依赖版本错位致 +20~30% decode 回退提示「依赖 bump 须同步调用层」。
2. **静默正确性跨引擎延续且 AMD 再添铁证**（第 4 条）：SGLang `#46928`（aiter unified-attn page size 1 + CUDA graph → GSM8K→0、无任何拒绝）是异构后端静默错误的教科书级案例；vLLM `#65812`、llama.cpp `#31657` 的量化 scale 漏接同源。
3. **KV 正确性/卸载/PD 健壮性仍是跨引擎最密集主线**（第 5/8 条）：vLLM KV Offload 三连、SGLang HiCache/write-through/RequestRetirer、TRT mixed-precision KV 解聚，共同把「KV 容量/复用/传输正确性」推向架构级。
4. **量化新能力与性能**：vLLM LUT-B MoE（Rubin 原生 + 仿真）、online MXFP4 激活 fail-fast；llama.cpp NVFP4 FFN 融合（+9.6%~+28.1%）；SGLang paged-experts 去 staging copy（省一倍 RAM）。
5. **确定性/稳定性**：TRT `#21319`（DeepSeek-V4-Pro + DSpark run-to-run 波动 33,930–39,598 tok/s，限 5%，来源为服务期编译 + 非确定 mHC split-K）；llama.cpp `#31683`（复杂 rope 模型 KV shifting）。

**可执行清单（建议）**：
- **P0（升级门禁）**：vLLM 升级 v0.33.x 前，对所有未显式 `--block-size` 的部署核验 KV 容量、KV connector offload chunk、autotune cache-key 与 ROCm 内核选择（对应 #63231）；TRT-LLM 规划 KVCM V1 / CPP transceiver → V2 / Python transceiver 迁移（对应 #21525，3 个月窗口）。
- **P0（静默错误自查）**：开启 AMD aiter unified-attn 时显式避开「page size 1 + CUDA graph」组合或升级含 #46928 的版本；对 vLLM Qwen4Exp QSA / llama.cpp Qwen2 NVFP4，逐层核对 FP8/NVFP4 KV scale 是否真正应用（对应 #65812/#31657）。
- **P1（投机/PD 验收）**：EAGLE/MTP + Triton MLA KV write + CUDA graph 组合加 `out_cache_loc` 连续性断言（对应 #48734）；PD/HiCache 验收含 write-through 与 hybrid 模型的 prefix 完整性（对应 #48421/#48452）。
- **P1（稳定性）**：DeepSeek-V4-Pro + DSpark 部署监控 run-to-run 吞吐波动，预填编译缓存（对应 TRT #21319 + #21121）。
- **P2（跟踪）**：KVGuard（arXiv 2611.06138）是否提供对主流引擎的 KV 跨请求泄露检测，纳入 PD-KV 安全验收工具链（二手线索/未核验）。
- **P2（新能力评估）**：vLLM LUT-B MoE 在 Rubin 实测吞吐与精度；llama.cpp NVFP4 FFN 融合在目标 Blackwell 机型落地。

> 证据等级说明：本档案「官方一手」指来自四家官方仓库的提交/PR/发布元数据与 PR 正文；「二手线索/未核验」指来自社区公告/HN/arXiv 且未经一手来源确认的内容（如 KVGuard）。所有性能/精度数字均标注来源 PR 或注明「截断/未完整呈现/设计级」，未外推未呈现数字。
