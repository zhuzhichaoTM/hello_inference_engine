# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-27（北京时间）
- **关注窗口：** 2026-10-26 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-26 08:00 – 2026-10-27 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性、社区安全披露
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-26 UTC 窗口，分别返回 **99 / 73 / 42 / 24** 条，均 HTTP 200）、四家官方 releases API、**20 个 PR 正文**（经 GitHub issue API 抓取，返回 markdown 文本，含 vLLM `#59774`/`#63101`/`#63669`/`#63698`/`#63554`/`#61994`、SGLang `#45925`/`#39622`/`#46340`/`#46263`/`#33250`/`#45710`、TensorRT-LLM `#20209`/`#20447`/`#20790`/`#20830`、llama.cpp `#31147`/`#30072`/`#31137`/`#29882`）。**窗口特征：** 2026-10-26 为**周一（工作日）**，四家提交量显著高于上周日（10-25 为 34/44/7/20）——vLLM 99 条为本任务追踪以来单日最高，主因量化/后端性能、Kimi-K3、投机解码与 KV-Offload 健壮性簇同日合入；TRT-LLM 42 条为近期高位；SGLang 73 条仍高（MiniMax-M3.1-Flash day0 + KDA HIP + diffusion 潮）。HN Algolia 公开 API 可访问（HTTPS 200）；窗口内出现**两条高信号安全披露**（LMSYS「SGLang PD disaggregation 跨请求 KV cache 泄露」、Cloudflare「GLM-5.3 Workers AI 跨请求 KV cache 暴露」），与本期 PD-KV 正确性主线直接呼应。Reddit、YouTube、X 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** 窗口内 vLLM / SGLang / TensorRT-LLM **均无新正式版本**（vLLM 仍 **v0.32.0** 2026-10-18；SGLang 仍 **v0.5.22** 2026-10-19；TensorRT-LLM 仍 **v1.4.0rc1** 2026-10-22，prerelease）。llama.cpp 滚动 nightly 共 **6 个标签**（b11908–b11913；其中 b11913 发布于 2026-10-27T00:19:01Z，紧贴窗口边界，窗口内可见 b11908–b11912）。无新版本事件，故本期头号事件为「新 KV 后端落地 + 跨后端默认切换（SM100/RoCm）延续『默认值风险』主线」+ 社区 KV cache 泄露安全披露（见第 1、6 节）。
- **本期方法论沿用：** 继续用 GitHub commit API 的 UTC 日期窗口 + PR 正文经 issue API 抓取（干净 markdown）；延续「版本核验/描述级核验」口径——发布说明/描述 ≠「修复已进版」，PR 描述 ≠ diff 落地，CI 状态须纳入证据等级（如 SGLang `#45925`/`#46340` AMD ROCm 10 红、llama.cpp `#31137` 描述为空）。

---

## 1. 版本状态与本期头号事件：周一高吞吐 + 新 KV 后端/跨后端默认切换延续「默认值风险」主线

**主题** 本期无新版本，但出现一组**新的 KV-cache 后端与跨后端默认切换**：vLLM **Octave 原生 3–4 bit KV 后端**（opt-in env）、TRT-LLM **PrimTS 稀疏 MLA 在 SM100 默认替换 TRTLLM-Gen**、**KVCMv2 默认 True 经文档确认**、vLLM **DeepSeek-V4 在 ROCm gfx950 默认切到 MXFP4 indexer KV**。其中后三项为「默认即行为变更」，是累计结论第 3 条（默认值成为主要风险载体）的最新证据；同时社区侧同日曝出跨请求 KV cache 泄露安全披露（第 6 节），使「KV cache 的正确性/隔离」成为本期贯穿引擎与社区的主线。

### 1.1 vLLM `#59774` Octave KV（`ROCM_OCTAVE`）：原生 3–4 bit KV cache 后端（AMD RDNA3/MI300X）

**主题**  `[ROCm] Octave KV (ROCM_OCTAVE): native 3-bit KV cache backend for AMD GPUs`（`#59774`，RFC `#59773`）。

**内容详情（取自 PR 正文）**  新增 `ROCM_OCTAVE` attention 后端，将 full-attention 层 KV 以 3–4 bit/value 存放，带 RDNA3（gfx1100）与 MI300X（gfx942）原生 HIP kernel；用 `--kv-cache-dtype octave_k3v3` / `octave_k3v4` / `octave_k3v3_compact` 选用（head size 256 下每 slot 216 / 248 / 196 字节，fp16 为 1024）。
- **容量：** 4× RX 7900 XTX + Qwen3.8-27B 达 fp16 的 **4.6× KV tokens**；单 MI300X 达 AITER fp8 的 **1.95×**（7.0M vs 3.6M，即 53 vs 27 个并发 128k 请求）。
- **质量：** GSM8K 与 126k needle-in-a-haystack 在两种架构上均与 fp16/fp8 对齐；380k 处 next-token NLL 较 fp16 **+0.0009 ~ +0.0015 nats**。
- **速度：** MI300X 单请求 decode 与 AITER fp8 持平，每层自 32k 起更快（128k 处 65 vs 103 µs）；RDNA3 380k 每层 decode 较 TurboQuant **4.6×**、MTP verification **11×**。
- **约束：** 与 MTP 和 CUDA graphs 兼容、metadata builder 无 host sync；自 2026-10-02 起已在 RDNA3 生产使用；`supports_compute_capability` 仅放行 gfx1100/gfx942，其余架构测试跳过。

**应用场景**  AMD RDNA3/MI300X 上的长上下文 KV 容量扩展（Qwen3.8-27B 等）。

**研判**
- **事实：** 有双架构测试（gfx1100 105 passed、gfx942 87 passed/18 skipped）、容量/质量/速度三类数字与「10-02 起生产」自述。属 opt-in 新后端，**非默认**，风险可控。
- **观点：** 是继 NVFP4/block-FP8 KV 后的另一条「低比特 KV 后端」主线，对 AMD 长上下文服务方价值高；但因仅 gfx1100/gfx942 且为 env 开关，建议 AMD 部署方在精度对拍（NLL/GSM8K）后再启用，不要盲开。

**来源**  https://github.com/vllm-project/vllm/pull/59774 （merge 2026-10-26T04:35:00Z；正文已核验，RFC #59773）

### 1.2 TRT-LLM `#20209` PrimTS 原生稀疏 MLA FMHA：DeepSeek-V4 生成阶段在 SM100 默认替换 TRTLLM-Gen

**主题**  `[None][feat] Add the PrimTS native sparse MLA FMHA for DeepSeek-V4 generation and enable it by default on SM100`（`#20209`）。

**内容详情（取自 PR 正文）**  新增 FMHA 库 `prims_ts_sparse_mla`（`PrimsTSSparseMlaFmha`，基于 vendored PrimTS CuTe DSL kernel，flashinfer#5434）：
- **默认路径变更，仅 SM100（B200/GB200）：** DeepSeek-V4 Flash/Pro 生成注意力改走 PrimTS 而非 TRTLLM-Gen，`DEFAULT_FMHA_LIBS` 顺序 `prims_ts_sparse_mla > flashinfer_trtllm_gen > fallback`；切换在 info 级日志可见（`FMHA layer N: PrimsTSSparseMlaFmha serves causal requests`）。
- **不变：** context 阶段（仍 TRTLLM-Gen）、SM103/Rubin、其它架构、其它模型、`fp8_ds_mla` footer-scale 缓存与 NVFP4 compressed pools。
- **退出：** `TLLM_FMHA_LIBS=-prims_ts_sparse_mla` 可恢复 TRTLLM-Gen；vendored `flashinfer-prims-ts` 重钉到 `trtllm-prims-ts-dev-20261023 @ aa709061`，并引入持久 CuTe DSL 编译缓存。

**应用场景**  TRT-LLM 在 B200/GB200 上跑 DeepSeek-V4 生成阶段的注意力后端选择。

**研判**
- **事实：** 明确的默认顺序变更 + 仅 SM100 生效 + 退出开关 + vendored 重钉；属「默认即行为变更」。
- **观点：** 与第 3 条（默认值风险）同源——SM100 用户升级到含本 PR 的版本后，DSv4 生成注意力后端**静默切换**到 PrimTS；若遇精度/性能回退，须知道用 `TLLM_FMHA_LIBS` 退出。本期将其作为「新默认风险」的最新铁证之一。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20209 （merge 2026-10-26T07:34:40Z；正文已核验）

### 1.3 TRT-LLM `#20447` + vLLM `#63669`：默认值的「文档与行为」两处确认

**主题**  TRT-LLM `#20447`（文档修正：`use_kv_cache_manager_v2` 默认 True）/ vLLM `#63669`（DeepSeek-V4 在 ROCm gfx950 默认切 MXFP4 indexer K cache dtype）。

**内容详情（取自 PR 正文）**
- TRT-LLM `#20447`：`#17495` 已将 `KvCacheConfig.use_kv_cache_manager_v2` 默认改为 `True`，但 KV Cache 指南仍写「defaults to `auto`」并误标表头；本 PR 把文档更正为**默认 True（即选 V2）**，`auto` 改为「显式 opt-in 到 model preference（V1 fallback、PD NIXL/PYTHON 条件）」。经 coderabbit 生成、无源码改动。
- vLLM `#63669`：`#60834` 把 paged MXFP4 sparse indexer 带到 DeepSeek-V4 的 ROCm gfx950，原为 opt-in（`--attention-config '{"indexer_kv_dtype":"mxfp4"}'`）；本 PR 因 `#60834` 实测 MXFP4 在几乎各轴优于 FP8（TTFT 最多 −30.7%、KV 容量 +9.2%、精度近无损：GSM8K −0.16pt、GPQA Diamond 不变 0.7121），**将 MXFP4 设为 DeepSeek-V4（`model_type=="deepseek_v4"`，即 V4-Pro/V4-Flash）在 ROCm CDNA4 的默认 indexer K cache dtype**（DSv4.1 仍 opt-in，CUDA/XPU/CPU 与其它模型不变）。DSv4-Pro TP8 MTP=2：conc1 吞吐 +7.09%、TTFT −30.70%、TPOT −4.49%；conc16 吞吐 +0.92%、TTFT −8.46%。

**应用场景**  TRT-LLM 的 KV cache 管理器默认选择；vLLM 在 AMD CDNA4 上跑 DeepSeek-V4 的 indexer KV 精度。

**研判**
- **事实：** 两 PR 均给出「默认变更 + 实测/文档依据」；TRT-LLM 为纯文档修正（无代码改动），vLLM 为实测驱动的默认翻转。
- **观点：** 二者共同坐实累计结论第 3 条——**「默认值」是推理引擎最主要的风险载体之一**：KVCMv2 默认 True（已多次出现，本次仅文档补全）、SM100 DSv4 生成走 PrimTS、ROCm DSv4 走 MXFP4 indexer，三处默认切换均不改变显式配置却改变数值/性能。建议 Serving 方把「升级后 diff 默认项 + 对应精度/回归测试」列为发布门禁。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20447 （merge 2026-10-26T11:43:29Z；正文已核验，纯文档）、`https://github.com/vllm-project/vllm/pull/63669` （merge 2026-10-26T13:14:56Z；正文已核验）

**证据等级（第 1 节）：** 官方一手（提交 API + PR 正文）。Octave（`#59774`）为 opt-in 新后端、有双架构测试；PrimTS（`#20209`）为 SM100 默认变更、有退出开关；`#20447`/`#63669` 为默认/文档变更，均按官方一手记录，但「默认切换的端到端回归面」建议用户自行在目标 checkpoint 上复跑精度（已在研判中标注）。

---

## 2. vLLM：10-26 提交（99 条，单日最高）——量化/后端性能、Kimi-K3、投机解码、KV-Offload 健壮性簇

**主题**  vLLM 在 2026-10-26 UTC 窗口合并 **99 条**提交（本任务追踪以来单日最高），主线为 **MXFP4/MXFP8 量化性能**、**Kimi-K3 agentic 路径**、**投机解码内存/正确性**、以及一组 **KV-Offload 同日密集健壮性修复（12+ 项）**。

### 2.1 `#63101` MXFP4 激活量化：传入 sorted_expert_ids，B300 上 kernel 最多 3.39×、E2E +5.7%

**主题**  `[Perf][MXFP4] Optimize activation quant for mxfp4. 3x kernel speedup, 5% throughput improvement`（`#63101`）。

**内容详情（取自 PR 正文）**  把可选 `sorted_expert_ids` 透传给 Cutlass MXFP4 MoE 激活量化，使每个 sorted token 行直接用专家索引而非查 offset 表。微基准（B300）：`mxfp4` kernel 在 256 专家从 7.33→2.16 µs（**3.39×**），`silu+mul` 7.44→2.43 µs（3.07×）；E2E（RedHatAI/GLM-5.3-MXFP4，TP=4，无投机）输出 **+5.7% tok/s**。kernel 路径与 offset_search **逐位一致**；API 在 `mxfp4_experts_quant`/`silu_and_mul_mxfp4_experts_quant`/`get_cutlass_moe_mm_data` 加可选参数。

**应用场景**  MXFP4 MoE（GLM-5.3-MXFP4 等）激活量化的 kernel 与端到端吞吐。

**研判**
- **事实：** 有 B300 微基准全表（E=16~256）+ E2E GLM-5.3-MXFP4 +5.7%，且 kernel 逐位一致。
- **观点：** 良性量化性能（路由已知时跳过 offset 查找）；属「白拿」收益，建议 MXFP4 MoE 部署方确认已含。

**来源**  https://github.com/vllm-project/vllm/pull/63101 （merge 2026-10-26T20:51:56Z；正文已核验）

### 2.2 `#63698` 量化正确性：初始化 ModelOpt NVFP4 MoE 有效正 dummy scales

**主题**  `[Bugfix] Initialize valid ModelOpt NVFP4 MoE dummy scales`（`#63698`）。

**内容详情（取自 PR 正文）**  `--load-format dummy` 基准下，随机初始化可能产生非正的 block/input scale 及不一致的 gate/up global scale。修复使 dummy NVFP4 MoE 权重走既有 `dummy_weight_value` 契约给出正 scale。回归测试在 1× GB200、BF16 激活、NVFP4 dummy experts、FlashInfer TRTLLM、eager：修复前 `assert (values>0).all()` 失败（w13_weight_scale），修复后 1 passed 且生成两 token、log 概率有限。

**应用场景**  NVFP4 MoE（Blackwell）dummy 基准的正确性。

**研判**
- **事实：** 有 GB200 单测翻转（失败→通过）与前后输出对照。
- **观点：** 属量化正确性（dummy 路径），影响面窄但「无报错、仅基准数值坏」；做 NVFP4 dummy 压测者须确认已含。

**来源**  https://github.com/vllm-project/vllm/pull/63698 （merge 2026-10-26T16:31:42Z；正文已核验）

### 2.3 `#63554` 投机解码：batch-sharded 下 draft logits 按 rank 分片省显存

**主题**  `[Spec Decode][Sampling] Batch-sharded draft logits`（`#63554`）。

**内容详情（取自 PR 正文）**  batch-sharded sampling 下 target logits 按 token 维分片（请求按 `req_state_idx % tp_size` 分配到 rank）；原 draft speculator 为所有请求分配 `max_num_seqs` 行 draft logits，TP 下仅用 `max_num_seqs // TP` 行，浪费显存。本 PR 使每 rank 只分配自己校验的请求：`cdiv(max_num_seqs, tp_size)` 行，释放的显存回流 KV cache。两个 Triton helper（`shard_owner`/`shard_row`）统一读写，单 shard 时退化为原行为。

**应用场景**  TP>1 + 投机解码（DFlash2/DSpark）下的 draft logits 显存。

**研判**
- **事实：** 明确的布局/分片改动 + TP 下显存回收机制。
- **观点：** 良性内存优化（TP 越大收益越大），不影响数值；属「默认配置下白捡」。

**来源**  https://github.com/vllm-project/vllm/pull/63554 （merge 2026-10-26T23:59:05Z；正文已核验）

### 2.4 `#61994` 投机解码：DeepSeek-V4.1-Flash PCP 支持固定长度 DSpark

**主题**  `[Spec Decode] Support PCP with DSpark for DeepSeek-V4.1`（`#61994`，扩展 `#59857` 独立 PCP 路径）。

**内容详情（取自 PR 正文）**  在 NVIDIA CUDA 上为 DeepSeek-V4.1-Flash 的 Prefill Context Parallelism（PCP）开启固定长度 DSpark 投机解码；保留 compressor ring tails 跨 PCP rank 传播（使被拒后缀可在更早位置续解、不丢 ratio-2 压缩所需前驱），并从 PCP rank 0 broadcast draft tokens 及（概率采样时）对应 logits；覆盖 PCP2/4、0/5/9 投机 token、ring 容量 8/16、partial rejection、odd/even accepted prefixes、空 local rank、decode-owner 变更等回归。

**应用场景**  DeepSeek-V4.1-Flash 的 PCP + DSpark 组合路径。

**研判**
- **事实：** 明确的 compressor 状态跨 rank 保留 + broadcast 语义 + 回归覆盖。
- **观点：** 属「前缀缓存 × MoE 路由 × 投机解码交叉债务」（累计结论第 5 条）在 DSv4.1 PCP 上的延伸；PCP+DSpark 部署方应确认已含本修复以保压缩状态一致。

**来源**  https://github.com/vllm-project/vllm/pull/61994 （merge 2026-10-26T19:34:57Z；正文已核验；GSM8K 命令已列，具体分值正文截断未给出，记为**官方一手但 GSM8K 数值未完整呈现**）

### 2.5 KV-Offload 同日密集健壮性修复簇（12+ 项，正确性/可观测性）

**主题**  vLLM KV-Offload（Mooncake/原生 `OffloadingConnector`/P2P NIXL/OBJ）在 2026-10-26 出现一组集中修复，均属「默认配置下易踩、无显眼报错」的健壮性收敛：

- `#63781` `[KV Offload][P2P] Fail faster`：每模式 load 超时、更短 abort-ack 与 lookup 超时（避免长尾挂起）。
- `#63708` `[Bugfix][KV Offload][P2P] Never block in ZmqConnection.send()`：发送路径不再阻塞。
- `#62929` `[Bugfix][KV Offload] Hold promotions after a cache reset until jobs report`：缓存重置后保留提升直到作业上报（避免重置期间丢提升）。
- `#63628` `[Bugfix][KV Offload] Roll back failed P2P NIXL registration`：P2P NIXL 注册失败回滚。
- `#63655` `[Bugfix][KV Offload][OBJ] Namespace object keys by blocks_per_chunk`：OBJ 对象键按 `blocks_per_chunk` 命名空间化（避免跨配置键冲突）。
- `#63664` `[Bugfix][KV Offload][P2P] Mark the session dead when its ConnectAck send fails`：ConnectAck 发送失败时标记会话死亡。
- `#63652` `[Bugfix][KV Offload] Do not store sliding-window chunks with null blocks`：滑动窗口 null 块不存（避免脏存）。
- `#61961` `[KV Offload] Load small CPU-tier hits synchronously within the step`：小 CPU-tier 命中在步内同步加载（降尾延迟）。
- `#61050` `[Bugfix][KV Offload] Preserve running request prefixes across LRU store batches`：LRU 存储批次间保留 running 请求前缀（避免前缀丢块）。
- `#60574` `[Bugfix][KV Offload] Preserve new loads during same-step resume`：同步恢复期间保留新加载。
- `#53902` `[Metrics][KV Offload] Fill vllm:kv_offload_config_info with the CPU cache capacity`：指标补全 CPU cache 容量。
- `#57646` `[KV Offload] Expose the processed prefix at request finalization`：终化时暴露已处理前缀。
- `#63224` `[Bugfix][KV Events] Prevent replay truncation when the send queue fills`：发送队列满时防止 replay 截断。

**应用场景**  vLLM 的 CPU/分层/PD KV offload（Mooncake、原生 `OffloadingConnector`、NIXL P2P、OBJ）。

**研判**
- **事实：** 13 项同日合入，覆盖超时/阻塞/回滚/键冲突/前缀保留/指标等多类边界；与往期 KV-Offload 静默正确性簇（如 10-25 的 `#63266`/`#61019`）同源。
- **观点：** 说明 KV-Offload 路径仍在「正确性 + 健壮性双修」，是生产可用前的必经收敛；建议重度使用 KV offload 的 Serving 方把上述 PR 纳入升级清单。

**证据等级：** 官方一手（提交 API + 标题）；上述 13 项仅标题级，未逐一抓取正文，按规范记为**官方一手（标题级）**。

### 2.6 其余 vLLM 窗口提交（标题级，正文未逐一抓取）

- **量化/MoE 性能：** `#61875` `[Perf][DeepSeek V4] Declare MXFP8 MoE activations and run them as block FP8 on Hopper`、`#63541` `[Perf][MoE] Size the DeepGEMM MegaMoE symmetric buffer for one SP shard`、`#61685` `[MoE] Add locality-domain allocation for FlashInfer CuTeDSL NVFP4`、`#62763` `[Perf][MoE] Add H200 FP8 fused MoE config for Beam-1.0 TP8`、`#62219` `[Kernel][Perf] Add H100 tuned configs...`、`#59255` `[ROCm] Enable expert parallelism for AITER Triton W4A16 MoE`。
- **Kimi-K3 / ROCm：** `#63638` `[Perf][Kimi K3] Opt-in KV-balanced FP8 MLA decode for agentic`、`#61527` `[Kimi-K3] Route the fp8 chunked-context K/V pack through FlashInfer`、`#63576` `[Bugfix][ROCm][DSv4.1] Await krdy before K2's combine reads klen`、`#63558` `[Bugfix][ROCm] Keep the AITER GDN decode fast path off for the DS conv-state layout`、`#63099` `[Bugfix][ROCm] Match the DeepSeek-V4.1 FP8 config in the shared-expert FSE compatibility check`、`#59478` `[ROCm][Perf] Use AITER FlyDSL INT4 QuickReduce`、`#59774` 见第 1.1 节（Octave）。
- **投机解码其它：** `#63421` `[Spec Decode] Replace torch.compile'd AV draft-budget selection with a Triton kernel`、`#63436` `[Bugfix][Spec Decode] DFlash: run the aux hidden state FC in the draft dtype`、`#61589` `[Core][Spec Decode] Lookahead block hashes for EAGLE prefix caching`、`#62301` `[Bugfix][Spec Decode] Preserve DP replica index for standalone draft models`、`#63498`/`#63708`/`#63664` KV Offload P2P 边界。
- **内核/正确性：** `#63475` `[Bugfix][Kernel] persistent_topk: sync the CTA around radix barrier arrivals`、`#62936` `[2/N][Bugfix] Make persistent TopK exact on candidate overflow`、`#62525` `[MoE][Model] Gemma 4: skip CUDA-graph padding rows in the routing kernel`、`#61038` `[Bugfix] Keep legacy WNA16 MoE dispatch on Triton under batch invariance`、`#59931` `[Bugfix] Preserve fp16 in legacy fused MoE LoRA`、`#61636` `[Bugfix][LoRA] Fix double pinning of 3D MoE LoRA`、`#62330` `[Bugfix][DCP] Finalize the NIXL DCP interleave size before profiling`、`#63431` `[Bugfix][MRV2] Skip adaptive verification cost profiling in the CUDA graph memory-profiling pass`、`#57926`/`#58243` MRV2 采样器/编码器图内存。
- **模型/前端/其它：** `#62201`/`#61879`/`#63651`/`#63629` Muse Spark Open 模型与文档、`#63569` Tool Calling 解析器迁移、`#61128`/`#58885`/`#62456`/`#62450`/`#63661` Rust Frontend、`#63817` NIXL 升 1.5.1、`#63852`/`#63665`/`#63657` Kimi-K3/DSv4.1-Flash ROCm 调度文档、`#62091` `[Perf][DSv4.1] Read GPU Engram tables from peers' HBM`。

**证据等级：** 官方一手（提交 API + 标题）；除 2.1–2.4、1.1 已抓正文者外，其余为**官方一手（标题级）**。

---

## 3. SGLang：10-26 提交（73 条）——MiniMax-M3.1-Flash day0、KDA HIP、DSv4.1、DSPARK fp8、MiniMax 默认修正

**主题**  SGLang 在 2026-10-26 UTC 窗口合并 **73 条**提交（周一量偏高），主线为 **MiniMax-M3.1-Flash day0 支持**（#45925，即上期在途的 `#46139`/`#46140`/`#46141` 伞 PR）、**KDA 在 gfx950 的 HIP kernel 默认**（#45710/#45713）、**DSv4.1 注意力后端默认与 TBO 正确性**（#46263/#33250）、**DSPARK fp8 verify-head**（#39622），以及一条 **MiniMax 在 ROCm 的默认数值分歧修正**（#46340）。

### 3.1 `#45925` MiniMax-M3.1-Flash day0 支持（新模型，CI AMD 红）

**主题**  `[Feat] minimax: Support MiniMax-M3.1-Flash`（`#45925`）。

**内容详情（取自 PR 正文 + CI 状态）**  新增 MiniMax-M3.1-Flash 模型支持（HF: `MiniMaxAI/MiniMax-M3.1-Flash`），附 day0 博客（sglang.io/blog/minimax-m3.1-flash-day0-support）、cookbook 与 DFlash2 draft。CI 状态：**Base / Extra 绿，AMD ROCm 10 红（:x:）**。

**应用场景**  SGLang 对 MiniMax-M3.1-Flash 的模型接入与 DFlash2 投机。

**研判**
- **事实：** 新模型接入 + 官方 day0 博客 + cookbook；CI Base/Extra 绿、AMD ROCm 10 红。
- **观点：** 属新模型 day0 支持（呼应 vLLM 侧同类「新模型快速跟进」节奏）；因 **AMD ROCm 10 CI 红**，按规范记为**官方一手（描述级/未完全绿）**——AMD 部署方勿在 CI 转绿前采信其 ROCm 路径数值。

**来源**  https://github.com/sgl-project/sglang/pull/45925 （merge 2026-10-26T08:42:30Z；正文已核验；CI 状态取自 PR 页）

### 3.2 `#45710` / `#45713` KDA 在 gfx950 的 HIP kernel 默认（Rubin/Kimi-K3 后端）

**主题**  `[Perf][AMD] kda: Use a HIP chunked-prefill kernel by default for KDA (Kimi-K3) on gfx950`（`#45710`）与 `[Perf][AMD] kda: Use a HIP fused decode kernel by default for KDA on gfx950`（`#45713`）。

**内容详情（取自 PR 正文）**  Kimi-K3 有 69 个 KDA（Kimi Delta Attention）层；原 gfx950（MI350X/MI355X）KDA prefill 走多 kernel Triton `chunk_kda` 流水线。`#45710` 新增 gfx950 专属 HIP chunked-prefill kernel（由 KDAgent 生成，作者称 humanize2 流程约耗 5k 美元），在 `TritonKDAKernel.extend` 内优先尝试、无 flag 即默认更快；kernel 仅 gfx950（需 bf16 16x16x32 MFMA 与至多 157 KiB LDS）。`#45713` 类似地把 HIP fused decode kernel 设为 gfx950 KDA 默认。

**应用场景**  Kimi-K3 NVFP4 在 AMD MI350X/MI355X（gfx950）的 KDA prefill/decode 性能。

**研判**
- **事实：** 明确「gfx950 默认更快、无新 flag」+ kernel 生成来源与约束（gfx950-only、LDS 上限）。
- **观点：** 良性 AMD 后端性能（默认切换），但属「跨后端默认变更」——gfx950 用户升级后 KDA 走新 HIP kernel；建议 AMD Kimi-K3 部署方在精度对拍后启用，并留意这是 agent 生成 kernel（可维护性/回归依赖上游 KDAgent）。

**来源**  https://github.com/sgl-project/sglang/pull/45710 （merge 2026-10-26T08:39:55Z；正文已核验）、`https://github.com/sgl-project/sglang/pull/45713` （merge 时间窗内，标题级）

### 3.3 `#46340` MiniMax-M3 在 ROCm 的默认数值分歧修正（默认值风险）

**主题**  `[Fix][AMD] MiniMax-M3: default SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ to 1`（`#46340`）。

**内容详情（取自 PR 正文）**  `#36527` 为 MiniMax-M3 在 gfx95 加了 sparse index top-k sharing 并**默认开启**（`SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=2`）；开启后每两个稀疏层复用上一层所选 KV block，使 ROCm 上跑出一个与其它平台**数值不同的模型**（其它平台后端固定为 1），除非用户知道 opt out。本 PR 把默认改回 **1（关闭 sharing）**，sharing 路径仍可作 `=2` 显式 opt-in。CI：**Base / Extra / AMD ROCm 10 均红（:x:）**。

**应用场景**  MiniMax-M3 在 AMD ROCm 上的稀疏 index 共享默认行为。

**研判**
- **事实：** 明确「默认开启 sharing → ROCm 与其它平台数值不一致」的根因 + 改默认为 1；CI 红。
- **观点：** 是累计结论第 3 条（默认值风险）的**教科书级案例**——一个默认开关让同一模型在不同硬件上跑出不同数值，且无报错。本期记为**官方一手（描述级/CI 红）**；MiniMax-M3 ROCm 部署方应确认已含本修复（或显式设 `=1`）以保证跨平台数值一致。

**来源**  https://github.com/sgl-project/sglang/pull/46340 （merge 2026-10-26T23:09:59Z；正文已核验；CI 状态取自 PR 页）

### 3.4 `#46263` DSv4.1 聚合服务：仅在每 rank 持 64 query heads 时默认 MegaAttention

**主题**  `[Perf][NVIDIA] deepseek-v4.1: Default aggregated servers to MegaAttention only when each rank holds 64 query heads`（`#46263`）。

**内容详情（取自 PR 正文）**  自 `#43922`，`--dsv4-attn-backend auto` 在 SM100/SM103 的 DSv4.1 上（fp8 KV）不论 attn TP 宽度都解析为 `flashmla_mega`（64-head tile，不足零填）。DSv4.1-Flash 有 64 query heads，TP2（每 rank 32）/TP4（每 rank 16）会跑 2×/4× 不需要的注意力工作；`flashmla` 路径有专用于 16/32 heads 的小批 SM100 kernel（更优）。本 PR：聚合服务（`--disaggregation-mode null`）下 `attn_tp_size>1` 时返回 `flashmla`；PD 角色仍保持 `flashmla_mega`（两后端 KV 布局不同，按 head 数 gate 会让 TP prefill 与 DP-attention decode 解析不一致、KV layout handshake 失败）。TP4 单步 target-verify 图因此获得 MegaAttention kernel（~1.45 ms/40 层）与其喂的 fp8 `wo_a` GEMM（~0.29 ms），丢掉 FlashMLA partial/combine 等（~0.85 ms），每步约慢 1.1 ms（即不盲目用 mega，反而更快）。

**应用场景**  DeepSeek-V4.1 在 SM100/SM103 的注意力后端自动选择（TP>1 聚合服务）。

**研判**
- **事实：** 明确的「head 数→后端选择」逻辑 + TP2/TP4 工作倍率分析 + KV 布局握手风险。
- **观点：** 属注意力后端默认选择的精细化（避免 2×/4× 浪费）；与第 3 条默认风险同源但为**正向优化**——提示 DSv4.1 SM100 部署方：TP>1 聚合服务路径的默认后端已改变，应复跑 TPOT/吞吐确认收益。

**来源**  https://github.com/sgl-project/sglang/pull/46263 （merge 2026-10-26T07:14:17Z；正文已核验）

### 3.5 `#33250` DeepSeek-V4 非 EP TBO 在 attention TP>1 的正确性修复

**主题**  `[Fix] deepseek-v4: Fix DeepSeek-V4 non-EP TBO for attention TP > 1`（`#33250`，依赖 `#31700`）。

**内容详情（取自 PR 正文）**  DSv4 两批重叠（TBO）+ DP attention、`moe_a2a_backend=none`、attention TP>1 时，对 attention-TP 组内复制的张量套用了全 TP 变长集合通信，破坏 gather/combine 布局、产出**无效输出**（H100 报告）；当前 main 在同配置下 TBO gather 直接失败。修复：给每个全 TP rank 分配其复制 DP-local 张量的一个分片，combine 后 reduce-scatter 到每 rank 一分片再 all-gather 重建；review 后补充：① 空 micro-batch（短 extend 拆出零 token 微批 → 融合 SiLU-and-mul 因 SwiGLU clamp 拒绝零块 launch、单条短请求崩服务）跳过专家；② attention TP>1 的 eager decode 补齐。

**应用场景**  DeepSeek-V4 TBO + DP attention + attention TP>1（非 EP）路径。

**研判**
- **事实：** 明确「全 TP 集合通信误用于复制张量 → 无效输出/崩溃」根因 + 空微批与 eager decode 两处连带修复 + AMD/H200 测试。
- **观点：** 属「调度/通信隐式阻塞与跨 TP 正确性」（累计结论第 1、5 条）在 DSv4 TBO 上的体现；非 EP TBO + attn TP>1 的部署方**必须**确认已含本修复，否则静默产出错结果或崩服务。

**来源**  https://github.com/sgl-project/sglang/pull/33250 （merge 2026-10-26T11:34:03Z；正文已核验）

### 3.6 `#39622` DSPARK 在 AMD 上 opt-in fp8 verify-head GEMM（性能）

**主题**  `[Perf][AMD] dsv4: fp8 target lm_head GEMM for DSPARK verify logits`（`#39622`）。

**内容详情（取自 PR 正文）**  DSv4 DSPARK 投机解码路径上，产生 verify logits 的 target `lm_head` 投影原为 bf16 `[M,hidden]×[hidden,vocab]`（vocab 维巨大、weight-read 受限），随 verify batch 增长。本 PR 加 opt-in fp8 路径（`torch._scaled_mm`，weight 量化一次并缓存），**默认关闭**；端到端为 decode-bound 小收益（高并发 **+1–1.5% tput / −3% TPOT**），**无 GSM8K 精度回归**。

**应用场景**  DSv4 DSPARK 在 AMD 上的 verify-head 精度/吞吐权衡。

**研判**
- **事实：** 有 kernel 减半论证 + 端到端小收益 + GSM8K 无回归；**默认关闭**（opt-in）。
- **观点：** 良性、opt-in 的性能选项；AMD DSv4 DSPARK 部署方可按需开启，注意为显式 flag 而非默认。

**来源**  https://github.com/sgl-project/sglang/pull/39622 （merge 2026-10-26T22:39:42Z；正文已核验）

### 3.7 其余 SGLang 窗口提交（标题级，正文未逐一抓取）

- **DSv4.1 / 重构：** `#46530`/`#46516`/`#46399`/`#46505` 把 engram hash、replay tail、candidate table 写入 caller-held prefill graph 缓冲（性能/重构）、`#46543`/`#46535`/`#46551`/`#46553` QSA/attention/nemotron_h eager break 输出缓冲重构、`#45449`/`#45659`/`#46668`/`#46432` rust-server/router 与 chat 层对齐、`#46547`/`#46546`/`#46506` MiniMax-M3.1-Flash AgentX cookbook。
- **HiCache / PD：** `#45673` `[Fix] hicache: Skip empty auxiliary transfers`、`#46501` `[Fix] hicache: Delegate on_host_pools_registered in the failure injector`、`#46570` `[Test] hicache: Accept session_ids in the L3 object-layout test`、`#40467` `[Feat] hicache: Expose the Mooncake L3 object-key layout`、`#44611` `[Feat] mem_cache: tag Mooncake offloads with the writing session`。
- **diffusion 性能潮（仍约占当日 1/4+）：** `#46227`/`#46308`/`#46174`/`#46240`/`#46247`/`#46249`/`#46254`/`#46263`/`#46393`/`#46449`/`#46396` CuTe-DSL norm/kernel 优化、Ulysses 交换隐藏、VAE 接缝、row chunk、realtime 清理、XPU/ROCm 跳过、Wan q/k norm RoPE 导入保护。
- **其它修复/功能：** `#45923` scheduler Req 自引用弱引用、`#44160` multimodal caller hashes/offsets、`#46060`/`#46193`/`#46057` diffusion 图像原子发布与 GLMs 像素保留、`#46049` grammar accept 失败给错误状态、`#43341` `[Feat][CPU] Grok-model: enable grok-2 on xeon cpu`、`#33250` 见 3.5、`#45791` `[Perf] QSA: Optimize prefill indexer in long cached prefixes`、`#46417`/`#45531` Intel/AMD 导入与 softcap 修正、`#46236` dsv4.1 k-norm-rope store on ROCm 10。

**证据等级：** 官方一手（提交 API + 标题）；除 3.1–3.6 已抓正文者外，其余为**官方一手（标题级）**；`#45925`/`#46340` 因 CI 红记为描述级/未完全绿。

---

## 4. TensorRT-LLM：10-26 提交（42 条）——PrimTS 默认、KVCMv2 并行 prefault、SID-GR、Muse Spark、serving

**主题**  TensorRT-LLM 在 2026-10-26 UTC 窗口合并 **42 条**提交（近期高位），主线为 **PrimTS 稀疏 MLA 默认 SM100**（#20209，见第 1.2 节）、**KVCMv2 初始 host tier 并行 prefault**（#20790）、**SID-GR beam search 每步 −1ms**（#20830）、以及 serving/模型支持（#20643 Muse Spark Open、多处以 Anthropic/Chat Completions 修正）。

### 4.1 `#20790` KVCMv2 初始 host tier 并行 prefault（性能/正确性）

**主题**  `[https://nvbugs/6900756][perf] KVCMv2: fill the initial host tier in parallel, populate in 4 MiB calls`（`#20790`）。

**内容详情（取自 PR 正文）**  KVCMv2 host tier 由一个 worker 线程按 `HostMem` 填充，每次 512 MiB：`mprotect` 开步 → 一次 `MADV_POPULATE_WRITE` 预缺页 → 注册。本 PR 改两处：① **每次填充用 4 MiB 调用**（原 512 MiB）：`MADV_POPULATE_WRITE` 持 `mmap_lock` 读锁直到返回，期间 executor 分配、其它池的 `mprotect`/注册都需写锁而等待；② **初始填充并行**：构造期填充跨多 512 MiB chunk 时一次性开整个范围，7 个辅助线程（共 8 线程）各绑 GPU 的 NUMA 节点，在逐步 commit 前预缺页；`resize()` 增长不用辅助线程（会盖服务、并行预缺页拖慢 executor）。附带修复一个 race：无 `MADV_POPULATE_WRITE`（内核 <5.14 或未编宏）时回退为零化内存，若辅助线程落后 commit 循环会零化已发布步内的 KV 数据（CodeRabbit 发现）。

**应用场景**  TRT-LLM KVCMv2 host tier 的启动预缺页与并发服务。

**研判**
- **事实：** 明确的 `mmap_lock` 争用根因 + 4 MiB 拆分 + NUMA 绑定并行预缺页 + race 修复。
- **观点：** 属 KVCMv2（默认 V2，见第 1.3 节）的启动/扩容健壮性；默认启用 V2 的用户在长上下文/大 host tier 场景会直接受益，建议确认已含。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20790 （merge 2026-10-26T22:38:09Z；正文已核验）

### 4.2 `#20830` SID-GR 宽 beam search 每步 host 开销 −1ms

**主题**  `[TRTLLM-16810][perf] SID-GR 3/N: Reduce per-step host overhead for wide beam search`（`#20830`）。

**内容详情（取自 PR 正文）**  三处改动：① 用 inplace `index_select` 替换 `index_select`+copy；② 不需要时不物化 `finish_reasons` 列表；③ 用单次 `add_new_tokens` 把该步 token 追加到每个 beam。在测试的 Qwen3 负载上**每 iter 省约 1 ms**。

**应用场景**  TRT-LLM SID-GR（宽 beam search）的逐步 host 开销。

**研判**
- **事实：** 有明确的三处 micro-opt + Qwen3 负载 −1ms/iter。
- **观点：** 良性 host 侧微优化，宽 beam 场景收益；无行为变更。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20830 （merge 2026-10-26T22:21:18Z；正文已核验）

### 4.3 其余 TRT-LLM 窗口提交（标题级，正文未逐一抓取）

- **KV / 注意力：** `#20209` 见 1.2（PrimTS 默认）、`#20790` 见 4.1、`#20178` `[None][feat] Unified kv cache manager foundation`、`#18843` `[TRTLLM-16303][refactor] modularize Mamba cache managers`、`#20928` `[None][perf] Speed up the post-warmup KV cache NaN/Inf scan`、`#20447` 见 1.3（KVCMv2 默认文档）。
- **模型支持：** `#20643` `[TRTLLM-17144][feat] Load the Muse Spark Open (muse_spark) config`（跨引擎 day0，呼应 vLLM `#61879`/`#63651`）、`#20837` `[None][fix] MiniMax-M3: resolve an unprefilled prompt as outside reasoning`、`#20279` `[https://nvbugs/6926284][fix] Align qwen3_coder parameter typing with vLLM`（与 vLLM 互操作对齐）、`#20391` `[TRTLLM-14477][refactor] unify multimodal request data`。
- **serving / gRPC：** `#20437` `[None][feat] Expose standard gRPC health in OpenEngine`、`#19760` `[None][perf] use native protobuf and streamline gRPC result delivery`、`#20874`/`#20867`/`#20864` Anthropic Messages / Chat Completions 修正、`#19883` `[https://nvbugs/6781271][fix] Responses API: agent-client fixes and disaggregated serving`、`#20736` `[None][feat] TTFT-percentile GEN admission for disaggregated serving, and a fix for a context-chunk replay after a skipped forward`、`#20842` `[TRTLLM-17241][fix] Start trtllm-render on a host with no GPU driver`。
- **beam / 采样正确性：** `#21027` `[None][fix] Use decoder mapping for beam-history rank checks`、`#20915` `[#19902][fix] Give logits processors each beam's hypothesis under beam search`、`#20830` 见 4.2、`#20947` EagerMoeWorkspace/LoRA 测试。
- **infra / 文档：** `#19557`/`#20666`/`#20584`/`#20786`/`#20892`/`#20898`/`#20748`/`#20839`/`#20789`/`#21006`/`#20985` PLC/scratch/CI/文档/版本 schema 同步。

**证据等级：** 官方一手（提交 API + 标题）；除 1.2、4.1、4.2、1.3 已抓正文者外，其余为**官方一手（标题级）**。

---

## 5. llama.cpp：10-26 提交（24 条）——Laguna MoE amax、Metal 64-bit 偏移、DSV4.1 follow-up、Vulkan RMSNorm、CPU concat

**主题**  llama.cpp 在 2026-10-26 UTC 窗口合并 **24 条**提交（周一量），主线为 **两条 Metal 静默数值错误修复**（#31147 Laguna MoE amax、`#30072` 64-bit 偏移）、**DeepSeek-V4.1 follow-up**（#31137）、**Vulkan/CPU 性能**（#29882 RMSNorm、`#31179` concat 重写），及多后端（Hexagon/SYCL/OpenCL）与 UI/CI 修缮。

### 5.1 `#31147` Laguna MoE amax 重缩放：修复 Metal 在 ubatch≥32 的溢出（静默错误）

**主题**  `llama : enable moe amax rescaling for laguna`（`#31147`）。

**内容详情（取自 PR 正文）**  把 `LLM_ARCH_LAGUNA` 加入 `build_moe_ffn` 既有 `build_amax_src1_f16` 条件，使 Laguna 复用 Mistral4/Mellum/Flint 已有的 f16 激活重缩放。原因：缺它时 Laguna MoE 在 Metal 上 prefill microbatch ≥32 token 溢出（#30896）——`kernel_mul_mm_id` 经 `half` 在 shared memory 暂存 f32 激活，gate/up 输出 >65504 时 down projection 变 inf；更小 microbatch 走另一 kernel 路径，故故障只在 ubatch≥32 出现。本地 Laguna-XS-2.1 验证：`ffn_moe_down` 的 src1 amax 从 **103671.5 → 1018.9**；未打标的 Metal `MUL_MAT_ID` 路径在 src1~1e5、n≥32 时产出 NaN 而 CPU 不。

**应用场景**  llama.cpp Metal 后端 Laguna MoE 的 prefill 数值正确性。

**研判**
- **事实：** 明确「溢出阈值 65504 + ubatch≥32 触发 + 只影响 Metal 路径」根因 + 本地 amax 对照（103671.5→1018.9）。
- **观点：** 与累计结论第 4、6 条（异构后端/量化静默错误）同源——**Metal 路径在特定 microbatch 下静默产出 NaN/inf、无报错**；Laguna Metal 部署方必须确认已含本修复，否则长 prefill 静默坏结果。属 AI agent（Devin）定位与改写。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31147 （merge 2026-10-26T22:04:20Z；正文已核验）

### 5.2 `#30072` Metal `kernel_mul_mm` 用 64 位目的偏移：修复超大矩阵溢出（静默错误）

**主题**  `metal : stop overflow from happening in large matrix output offsets`（`#30072`）。

**内容详情（取自 PR 正文）**  在两个 store 路径对目的偏移改用 64 位算术，避免偏移超过有符号 32 位范围时出错。附回归测试（默认关闭，需约 45 GB 内存）：在 M4 Pro Mac mini 上修复前失败、修复后通过；启用后 1,860 个 Metal `MUL_MAT` 测试全过。修复前 `MUL_MAT(type_a=f32,type_b=f32,m=1048576,n=40,k=64,...)` 在 M4 Pro 上 `ERR=0.0203 > 0.0005` FAIL。

**应用场景**  llama.cpp Metal 后端超大矩阵（`m` 很大）的 `MUL_MAT` 数值正确性。

**研判**
- **事实：** 明确「有符号 32 位偏移溢出 → 大 m 矩阵错结果」+ M4 Pro 回归（1,860 测试全过）。
- **观点：** 另一条 **Metal 静默数值错误**（大矩阵偏移溢出、无报错），与 5.1 共同说明 Metal 后端仍需逐路径数值对拍；跑超大 `m` 的 Metal 用户须确认已含本修复。

**来源**  https://github.com/ggml-org/llama.cpp/pull/30072 （merge 2026-10-26T08:13:40Z；正文已核验）

### 5.3 `#31137` DeepSeek-V4.1 follow-up review fixes（新模型，描述级）

**主题**  `deepseekv4.1: follow-up review fixes`（`#31137`，续 `#31086`）。

**内容详情**  PR 描述极简（仅「cont #31086」），未给实现细节。仅能从标题确认对 DeepSeek-V4.1 模型支持的 review 补丁。

**应用场景**  llama.cpp 对 DeepSeek-V4.1 的模型支持补丁。

**研判**
- **事实：** 仅标题级信息；描述为空，无法核验具体能力/精度/性能。
- **观点：** 新模型支持类 follow-up；因**描述为空**，按规范记为**官方一手（标题级/描述级）**，不对其内容做任何推断；待后续 PR 或模型卡补充。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31137 （merge 2026-10-26T07:52:03Z；正文为空）

### 5.4 `#29882` Vulkan RMS Norm 子组归约（性能）

**主题**  `vulkan: rms norm optimization using subgroup reductions`（`#29882`）。

**内容详情（取自 PR 正文）**  把 RMS Norm shader 的 workgroup 归约替换为两级 subgroup 归约。实测（pp8192）：Intel B70 Arc Pro（Win）gemma4 26B A4B Q4_K_M **1525.36→1544.14 t/s（+1.2%，fa0）**、880.44 vs 874.89（fa1，+0.6%）；另测 Nvidia RTX 4060 Ti（Win）、AMD 7900 XT（Win）。

**应用场景**  llama.cpp Vulkan 后端 RMS Norm 的 prompt-processing 吞吐。

**研判**
- **事实：** 有 B70/RTX4060Ti/7900XT 三卡 llama-bench 全表，+0.6~1.2%。
- **观点：** 良性融合性能，跨 Vulkan 设备通用；属「白拿」收益。

**来源**  https://github.com/ggml-org/llama.cpp/pull/29882 （merge 2026-10-26T14:14:32Z；正文已核验）

### 5.5 其余 llama.cpp 窗口提交（标题级，正文未逐一抓取）

- `#31179` `ggml-cpu : rewrite concat as one row kernel for all types`（CPU 性能）、`#31141` `llama : move no_alloc out of llama_hparams`、`#31142` `models : replace ggml_tensor** out-params with output structs`、`#30471` `opencl: write KV-cache views in set_tensor so a state restore is not dropped`、`#31180` `ggml-cpu : fix unused variables in arm arch`、`#31183` `mtmd: add mtmd_helper_get_capabilities`、`#31221` `hexagon: add GGML_HEXAGON_VTCM_SIZE`、`#31169` `hexagon: MUL_MAT_ID row skipping optimizations`、`#31013`/`#31008`/`#30358`/`#31062` `sycl: template/set_tensor_2d/bound reorder temp memory`、`#30548` `common: pin model resolution on real HF repo layouts`、`#31030`/`#31119` `ui: UX/sidebar improvements`、`#31191`/`#31162`/`#31174`/`#31175` `ci : windows behavior / webgpu memset race / server timeout / drop macos-latest-x64`、`#30880` `tests : refactor test-llama-archs`、`#31147` 见 5.1、`#30072` 见 5.2、`#31137` 见 5.3、`#29882` 见 5.4。

**滚动 nightly：** 10-26 共 **6 个 nightly 标签** b11908–b11913（b11913 发布于 2026-10-27T00:19:01Z，紧贴 UTC 窗口边界；窗口内可见 b11908–b11912），较 10-25 的 b11870–b11889 向前推进。

**证据等级：** 官方一手（提交 API + 标题）；`#31147`/`#30072`/`#29882` 已抓正文，其余为**官方一手（标题级）**；`#31137` 描述为空 → 标题级/描述级。

---

## 6. 社区与视频信号（HN / 安全披露）

**主题**  2026-10-26 窗口 HN Algolia 出现**两条高信号安全披露**，均指向**跨请求 KV cache 泄露/暴露**，与本期第 1 节「KV cache 后端与正确性」及往期 PD-KV 校验（SGLang `#46423`、HiCache）主线直接呼应；此外有 vLLM tool-call 精度第三方博客与一例推理调度器新项目。

- **「Cross-request KV cache leakage in SGLang PD disaggregation」**（LMSYS 官方博客，作者 moelove，2026-10-26T23:02:11Z，HN 1 分）：链接 `https://www.lmsys.org/blog/2026-10-26-cf-security/`。即 SGLang **PD（prefill/decode 分离）场景下的跨请求 KV cache 泄露**披露。
  - **研判：** 这是**与本期引擎侧 KV 正确性追踪直接对应的安全维度**——往期档案已记录 SGLang PD KV checksum（`#46423`）与 HiCache 多类静默损坏（#40917/#41907/#43333），本次披露把「KV 跨请求隔离」从正确性上升为**安全漏洞**级别。来源为 LMSYS 官方博客（权威一手），但**引擎侧 remediation/补丁状态在本窗口（10-26）内未捕获到对应 PR**，建议核实 SGLang 最新 main 与 releases 是否已合入修复（记为**官方一手（安全公告）/ 引擎补丁状态未核验**）。
  - **来源：** LMSYS `https://www.lmsys.org/blog/2026-10-26-cf-security/`（已核验链接可访问）；HN Algolia 命中（标题/作者/时间已核验）。**证据等级：官方一手（安全公告）。**
- **「Cross-request KV cache exposure in the GLM-5.3 Workers AI model」**（Cloudflare 博客，155 条评论、50 分，2026-10-26T22:41:48Z）：链接 `https://blog.cloudflare.com/kv-cache-exposure-remediation/`。即 GLM-5.3 在某 Workers AI 服务上的跨请求 KV cache 暴露与 remediation。与上一条同源（CF 安全团队协调披露），说明**跨请求 KV 隔离是当下推理服务端的普遍隐患**，不止单一引擎。
  - **研判：** 属权威厂商（Cloudflare）安全公告，证据等级**官方一手（安全公告）**；其 remediation 涉及服务端 KV 隔离改造，非单一引擎 PR，故引擎侧普适修复状态未核验。对推理 Serving 方的启示：上线前必须验证「KV cache 按请求/租户隔离 + 前缀缓存命中不跨请求泄露」。
  - **来源：** Cloudflare `https://blog.cloudflare.com/kv-cache-exposure-remediation/`（已核验链接可访问）。**证据等级：官方一手（安全公告）。**
- **「Fixing Tool-Call Accuracy in vLLM」**（Ubicloud 技术博客，2026-10-26，多条评论级命中）：链接 `https://www.ubicloud.com/blog/fixing-tool-call-accuracy-in-vllm`。第三方（非官方）关于 vLLM 工具调用精度的复盘与修复建议。
  - **研判：** 属第三方工程博客（与本期 vLLM `#63569` Tool Calling 解析器迁移为不同来源），对 vLLM 工具调用部署有参考价值，但为**二手线索/未核验**（非官方、单厂商口径）。
- **「Show HN: Resin, fast native scheduler for LLM inference engines」**（作者 ekzhang，2026-10-26T17:06:14Z，1 分）：链接 `https://github.com/ekzhang/resin`。一个新的原生 LLM 推理引擎调度器项目。
  - **研判：** 社区新项目（Show HN，低分），与调度器/编排相关，可作技术选型参考，但属**二手线索/未核验**（个人项目、单 star 起步、未跨硬件验证）。
- `vllm`/`sglang`/`TensorRT-LLM`/`KV cache`/`LLM inference`/`speculative decoding` 的其余命中**多为评论级或无关前台发布**（Anthropic S-1、Microsoft 买断、IBM CEO 谈开放权重、OpenAI 数学预印本等），无一条指向引擎本身的官方发布，与既往各期一致。
- **视频 / Reddit / X：** 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

---

## 7. 累计趋势判断（继承 + 本期新增）

**继承（截至 2026-10-26 的各期累计结论）：**
1. **调度器隐式阻塞的定量画像**——优先级/调度路径的隐式退化（本期 vLLM 仍有 KV-Offload 同质边界）。
2. **发版窗口的双向性**——「发版当天合入 main」≠「在发版内」（compare API 已多次证实）。
3. **默认值成为主要风险载体**——`--max-model-len auto`、默认开启的 flag、跨后端默认后端切换等改变行为且不报错（**本期强化**：TRT-LLM KVCMv2 默认 True 文档确认 `#20447`、DSv4 生成在 SM100 默认走 PrimTS `#20209`、DeepSeek-V4 在 ROCm 默认 MXFP4 indexer `#63669`、MiniMax-M3 在 ROCm 默认 sharing 导致跨平台数值分歧 `#46340`）。
4. **异构后端静默错误高频**——ROCm/Blackwell/Rubin 的 miscompile / 精度 / 同步陷阱（本期 llama.cpp `#30072` Metal 64-bit 偏移溢出、Metal 路径大矩阵静默错结果）。
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**——三者任一组合易踩、无显眼报错（本期 `#61994` DSv4.1 PCP+DSpark 压缩状态跨 rank）。
6. **量化路径静默错误** 为与异构后端并列的第二类高频静默错误（TRT-LLM `#20823` NVFP4_AWQ scale 漏应用、llama.cpp `#31097` MiniMax-M3 MoE scale 未前向）。
7. **专家张量/路由归属与分页漏接静默错误**（2026-10-25 立）：SGLang `#45787`、vLLM `#63372`、llama.cpp `#30820` 同源。

**本期新增 / 强化：**
- **第 3 条（默认值风险）获四项硬证据**（见上），且本次出现「同一模型在不同硬件因默认开关跑出不同数值」（SGLang `#46340`）这一最危险形态——建议把「升级后 diff 默认项 + 对应精度/回归测试」列为推理 Serving 的发布门禁。
- **静默错误主线从「异构后端/量化/MoE 分页」延伸到「Metal 后端大矩阵与 MoE 溢出」**（llama.cpp `#30072`/`#31147`）：本期两条 Metal 静默数值错误再次表明，**Apple Silicon 路径仍需逐 kernel 数值对拍**，与第 4、6 条共构「低比特/后端静默正确性」主线。
- **KV cache 的正确性议题升维至「安全」**（第 6 节）：LMSYS + Cloudflare 同日披露跨请求 KV cache 泄露/暴露，把往期 PD-KV 校验/HiCache 静默损坏的「正确性」问题推到「多租户隔离漏洞」层面——建议推理 Serving 方把「KV 按请求/租户隔离 + 前缀命中不跨请求泄露」纳入上线安全门禁。

**趋势判断：** 2026-10-26（周一）四家提交量 vLLM 99 / SGLang 73 / TRT-LLM 42 / llama.cpp 24；vLLM 单日 99 条为追踪以来最高，主因量化/后端性能 + Kimi-K3 + 投机解码 + KV-Offload 同日密集健壮性修复。本期主线可概括为「**新 KV 后端/跨后端默认切换（SM100/RoCm）+ 静默正确性（Metal/MXFP4/MoE）+ KV 安全升维**」三条并行；建议在量化、MoE、跨后端默认、Metal 后端、KV 隔离五处均固化为上线验收。

---

## 8. 可执行清单 + 数据质量与采集说明

**可执行清单（建议动作）**
- **默认/跨后端风险（第 3 条）：** 升级后**强制 diff 默认项**——TRT-LLM 含 `#20209` 时 SM100 的 DSv4 生成注意力静默切 PrimTS（回退 `TLLM_FMHA_LIBS=-prims_ts_sparse_mla`）、KVCMv2 默认 True（`#20447`）；vLLM `#63669` 使 ROCm gfx950 的 DSv4 默认 MXFP4 indexer；SGLang `#46340` 使 MiniMax-M3 在 ROCm 默认 sharing 关闭（或显式设 `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1` 保跨平台数值一致）。
- **vLLM 用户：** 量化性能 `#63101`（MXFP4 +5.7%）、`#61875`（MXFP8→block FP8 Hopper）；Kimi-K3 `#63638`/`#61527`；投机 `#63554`（TP 下 draft logits 省显存）/`#61994`（DSv4.1 PCP+DSpark）/`#61589`/`#62301`；KV-Offload 健壮性簇（2.5 节 13 项，尤其 `#63781`/`#63708`/`#63628`/`#61050`）；量化正确性 `#63698`；AMD `#59774` Octave 3-bit KV（opt-in，精度对拍后启用）。
- **SGLang 用户：** `#45925` MiniMax-M3.1-Flash（**AMD ROCm 10 CI 红，待转绿再采信数值**）；KDA `#45710`/`#45713`（gfx950 默认 HIP kernel，精度对拍）；`#46263` DSv4.1 TP>1 聚合服务后端选择变更；`#33250` DSv4 非 EP TBO attn TP>1 正确性（必须含）；`#39622` DSPARK fp8 verify-head（opt-in）；扩散/HiCache 按标题评估。
- **TRT-LLM 用户：** `#20209` PrimTS 默认 SM100、`#20790` KVCMv2 并行 prefault、`#20830` SID-GR −1ms、`#20643` Muse Spark Open、`#20279` qwen3_coder 与 vLLM 对齐。
- **llama.cpp 用户：** `#31147`（Laguna MoE amax，Metal ubatch≥32 溢出）、`#30072`（Metal 64-bit 偏移，大矩阵静默错）、`#29882`（Vulkan RMSNorm +0.6~1.2%）、`#31179`（CPU concat）；关注 nightly **b11908–b11913**；`#31137` DSV4.1 follow-up 仅标题级，待补细节。
- **KV 安全（第 6 节）：** 推理 Serving 方上线前验证「KV cache 按请求/租户隔离 + 前缀缓存命中不跨请求泄露」，并跟进 SGLang/服务端对应 remediation 是否已合入（LMSYS/Cloudflare 同日披露）。
- **社区参考：** vLLM tool-call 精度（Ubicloud 博客，二手）、Resin 调度器（Show HN，个人项目，二手）——均须自行验证，不当普适结论。

**数据质量与采集说明**
- 官方提交 API（UTC 窗口 2026-10-26）：vLLM **99** / SGLang **73** / TensorRT-LLM **42** / llama.cpp **24**，均 HTTP 200。
- 20 个 PR 正文经 GitHub issue API 抓取（markdown 干净文本）；四家 releases API（均 HTTP 200）；HN Algolia 公开 API（HTTPS 200）。
- 版本核验口径延续：发布说明/描述 ≠ 进版；PR 描述 ≠ diff 落地。本期 **SGLang `#45925`/`#46340` CI 红（AMD ROCm 10）** 按描述级/未完全绿；**llama.cpp `#31137` 描述为空**按标题级；**社区 KV 泄露为安全公告（LMSYS/Cloudflare 官方一手），引擎侧补丁状态未核验**。
- 未采集 Reddit / YouTube / X；未核验到官方视频。所有性能数字均来自 PR 自述，未做独立第三方对拍（已在各条目标注）。
- 本档案与上一期（2026-10-26 档案，覆盖 2026-10-25）连续无空档；本期为连续第四期。
