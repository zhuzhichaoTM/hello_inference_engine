# 大模型推理优化每日热点档案

> 档案日期：2026-11-19（本自动化于该日执行）
> 关注窗口：2026-11-18（周三）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，vLLM 分两页 100+10）+ releases API + 40 个关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 均经官方 issues API 抓取 markdown，官方一手）+ TRT-LLM v1.4.0rc4 发布说明全文（62,024 字符）+ HN Algolia（HTTPS，numericFilters 按 2026-11-18 epoch 窗口 1794960000–1795046399）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 110 / SGLang 93 / TRT-LLM 66 / llama.cpp 31 合计 **300** 条（2026-11-18 为周三，vLLM 分两页取全、无截断；TRT-LLM 66 较上一窗口 73 略回落但仍处高位）；**窗口内有新 prerelease 版本**——TRT-LLM **v1.4.0rc4 发布**（2026-11-18T19:16:49Z，prerelease），且当日 main 已 `Bump version to 1.4.0rc5`（#98c7b46f5d）；其余无新正式版：vLLM v0.34.0（11-16）、SGLang v0.5.23、llama.cpp 正式版 v1.0.0（11-13），llama.cpp 推进 nightly b12503–b12528（b12529 已落 11-19 窗口外）。40 个关键 PR 正文全部抓取成功（issues API 配额充足、无降级）。HN 引擎相关高信号弱：Show HN「Production Grammar-Guided Decoding in vLLM with Xgrammar Benchmarks」（1 分）、Show HN「QuadGate」（1 分），均二手线索/未核验。

---

## 1. 版本状态与本期头号事件：TRT-LLM v1.4.0rc4 发布（窗口内，prerelease）+ 静默正确性多引擎密集延续 + PDL 依赖同步顺序成跨引擎新子项（新增第 22 条）

**提交计数（2026-11-18 UTC，均 HTTP 200；周三）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 110 | 分两页 100+10；Kimi-K3.1-Pro/DSv4.1 recipe 簇 + ROCm GLM-5.3-Flash 性能簇 + 静默正确性 + Rust 前端 |
| SGLang | 93 | rust-server API 大栈（约 20 条）+ Kimi-K3.1 AMD gfx950 day-0 栈 + DSv4.1 性能 + kernel 清理 |
| TensorRT-LLM | 66 | **v1.4.0rc4 窗口内发布**，仍以 infra/waive/test 为主，feature/fix/perf 约 25 条 |
| llama.cpp | 31 | 多后端（CUDA/HIP/OpenCL/SYCL/Vulkan/WebGPU）+ 状态静默错误 + UI |
| **合计** | **300** | 工作日正常量；与上一窗口（11-17 的 298）连续，无空档 |

### 1.1 本期头号事件：TRT-LLM v1.4.0rc4 发布（窗口内 prerelease）

- **v1.4.0rc4**（官方一手，2026-11-18T19:16:49Z，prerelease，62,024 字符发布说明全文核验，`compare/v1.4.0rc3...v1.4.0rc4`）：本期唯一窗口内版本事件。**BREAKING**：checkpoint I/O policy 与 checkpoint formats 改用 string enum（#20498/#20703）。**弃用**：KV cache manager V1 与 C++ cache transceiver（#21525）。**默认变更**：Modeling V2 LLM 参数默认 `auto`（#21712）；Qwen3.5/Qwen3.8 dense NVFP4 checkpoint 在 Ada/Hopper 默认走 Marlin（#21434，延续第 20 条「发版固化默认值风险」）。**新特性/修复主线**：sparse KV offload（DS-V4-Flash/GLM-5.2，KVCM V2，#21350）、mixed-precision KV 扩到投机解码/sparse MLA/冷页压缩（#21021/#20197）、共享 drafter KV pool + FP8 MLA + 统一 DSpark/DFlash KV（#21972/#20664/#20486）、KVCM v2 的 SINK_REQUIRED_ONLY 与 GEN slot 抢占（#21432/#21334/#21793）、SSD offload 三项优化（#21822）、DSA 原生 prefill indexing + 精确 top-k（SM100/103，#21892）、Qwen3.8 QSA PrimTS（#20714）、非-MLA PrimTS FP8 KV（#19396）、NVFP4_PLUS CuTe DSL MoE on Rubin（#21242）、Rubin MLA/DSpark/DWDP/Helix 合并回灌（#21067）。**Known Issues 官方明示 5 项**：DeepSeek-R1 FP4 解聚在 B300 + CUDA graph 下可能产出错误输出；GLM-5.2 NVFP4 SM103 TP8 在 Modeling V2 + FlashInfer 0.7.3 下可能初始化失败；Kimi K3 NVFP4 KV 在 B300 因 FP4 MLA eligibility 可能初始化失败；Kimi K3 DSpark + MegaMoE CuTe DSL TEP + CUDA graph 在 GB300 warmup 可能 hang；Qwen3.8-Flash-Next FP8 + PLE offload + attention DP + MTP 可能初始化失败。研判（观点）：rc4 把前几期追踪到的「异构/量化 KV 静默正确性 + 默认值风险」集中固化为**发布页可查的 Known Issues**，是本追踪第 20 条（正式发版固化风险）在 RC 序列上的再次印证；按本追踪方法论，发布说明为精选 highlights，**不作为「某修复已进版」的依据**。窗口内 main 已 bump rc5（#98c7b46f5d），说明 rc4 为滚动 RC 中间点。

### 1.2 静默正确性多引擎密集延续（强化第 4/6/19 条，本期无结构新增）

本窗口一手静默/崩溃级案例延续上期密度，覆盖四引擎：

- **SGLang #50008**（官方一手，fix，强化第 6 条）：SM100 上 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`（hidden 2688，是 128 倍数但非 256 倍数）经 TRT-LLM NVFP4 MoE 内核**静默产出错误 MoE 输出、无报错**——cookbook 命令 GSM8K 仅 **17%**（BF16 为 90%）。根因：内核要求 hidden 为 256 倍数，原 `trtllm_nvfp4_hidden_alignment` 只 pad per-token non-gated 变体到 512，漏掉默认选中的 static-scale ModelOpt NVFP4 路径。修为返回 256（2688→2816）并对 padded 路径关 FP4 dispatch。与 vLLM 同名函数（pad 到 256）行为对齐；FlashInfer 0.7.4 已文档化该要求。属量化布局漏接静默错误铁证。
- **vLLM #64737**（官方一手，fix，强化第 4/19 条）：ROCm 上 `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` 用于 softmax 路由且共享专家无 gate 的模型（DeepSeek-V2-Lite、Mistral-Large-4-1T-A52B）时，`AiterSharedRoutedFusedMoERouter`（#39280 为 Qwen3-Next 引入）把末列 router logit 一律当共享专家 gate，**静默错路由**。DeepSeek-V2-Lite GSM8K strict **0.095→0.661**（fusion off 0.656）；Mistral-Large-4 5-shot 0.718→0.959、20-shot 0.770→0.965（fusion off 0.957/0.962）。修后 fusion 可用，MI325X TP8 c1 TPOT 较 fusion off 低 21%。属异构后端默认融合 × 模型结构假设失配（延续第 4 条）。
- **vLLM #66679**（官方一手，fix，强化第 19 条）：Kimi-VL 开 encoder CUDA graph（`cudagraph_mm_encoder=True`）时视觉编码器对多图**静默产出错误输出**（读图文字从正确串变成 `'0123456789 ABCDEFGHIj abcdefghij'`）。根因：图以 token budget 设 `max_seqlen`，而 MoonViT attention 跑在合并前的 patch 上（数量 ×kh·kw=4）。修为按 patch 计（Kimi-K2.5 已有做法），一行修复+单测；33/35（FLASH_ATTN）、34/35（TRITON_ATTN）批次接近 eager，其余为单小图误差 <2.9%。与上一期 llama.cpp #32433 / vLLM #68021 的 Muse Glimmer 视觉坐标错同属**新模型视觉移植静默错误高频区**。
- **vLLM #68359**（官方一手，fix，强化第 8 条）：KV Offload 的 `OnComputeStoreHandler.first_live_chunk_idx` 上报了高于最终 SWA tail 的 chunk，导致调度器陈旧块扫描不清该 tail，finish 时 store **把已分配给其他请求的 GPU 块复制走**（abort、正常结束、DCP、跨窗前一块均受影响）。新测试在 upstream/main（b5b9f7982c）4 failed、本 PR 4 passed。
- **vLLM #66011**（官方一手，fix，强化第 8 条）：NIXL P/D 解聚中，consumer 可能在 producer 的 KV lease 过期并回收块后仍发 NIXL read，**读取成功但静默加载其他请求的 KV cache 与 SSM state**（修复前无报错）。修为对所有 remote-prefill read 复用 deadline gate（安全边际 `min(5s, kv_lease_duration/6)`），新增 `vllm:nixl_num_declined_expired_reads` 计数，按 `kv_load_failure_policy`（默认 `fail`）处理；NIXL_CONNECTOR_VERSION bump 到 17。
- **vLLM #65353**（官方一手，fix，强化第 6/21 条）：direct weight reload（#64336）下 gpt-oss MXFP4 专家以 TP>1 流式加载时，按 padded size 收窄每 rank 切片，导致除 rank0 外**每个 rank 取错 checkpoint 行**——gpt-oss-20b TP2 每次同步后 240 个专家张量与冷启动不一致（修后 0）；同时三个加载器在 TP rank>0 对入参 `w2_bias` 调 `zero_()`，IPC 传递下即训练器缓冲区，被静默清零。修为按整 MXFP4 block 切分、用 `zeros_like`。
- **llama.cpp #31913**（官方一手，fix，强化第 19 条）：循环层（recurrent/hybrid）状态中出现 NaN 后，因 reset 逻辑用「乘 0」而非直接写 0，**NaN 永久粘滞**在状态中。在 Qwen3.8-flash-next over RPC + q8 K/V cache 发现，适用所有 recurrent/hybrid 模型。修为 `ggml_scale_inplace`→`ggml_fill_inplace`；NaN 投毒测试修前失败修后通过；Fixes #29830（曾以 completed 关闭但未修）。
- **TRT-LLM #22342**（官方一手，fix）：phase-interleaved FC12 mapping 的 prologue probe 传入未解析的 stream extent，四个 exhaustion 回归各报 96/192、160/192 元素不匹配（修前 4 failed/4 passed→修后 8 passed）。属 MoE prologue 探测流状态码漏解析。

### 1.3 新增累计结论第 22 条：PDL（Programmatic Dependent Launch）依赖同步次序成为跨引擎静默/崩溃新子项

- **TRT-LLM #22089**（官方一手，fix，新增第 22 条）：trtllm-gen context FMHA 在 PDL 下、`griddepcontrol.wait` **之前**读取累积序列长度，可能看到前序 kernel 写入的陈旧长度，**导致 persistent context kernel hang 或输出损坏**，最常见于 BF16-KV MLA context 路径（长度写入 kernel 紧邻 FMHA）。修为更新 FMHA artifacts，使所有导出 cubin 在读 seq length 或任何前序产物前先 `griddepcontrol.wait`（SASS 逐个 cubin 核验通过）；同时**回滚 #19873**（`TRTLLM_GEN_FMHA_CONTEXT_STATIC_TILE_SCHEDULER`，原为 hang 的 workaround 但未生效，默认关）。
- **跨期/跨引擎串联**：上一期（11-17 档案）TRT-LLM #22287 为 SM103 NVFP4 GEMM PDL 读陈旧 scale（GSM8K 94.05→0.53）、#22258 把 split-KV softmax scale 读取移到 `cudaGridDependencySynchronize()` 之后；再上一期 vLLM #65288（本期亦有 CI follow-up）与 SGLang #51080（本期，把 qk-norm/KV-prepare/gate 三个 kernel 改 PDL launch 以消依赖）分别从正确性/性能两侧触及 PDL。**研判（观点）**：PDL 用「前序 kernel 可提前释放依赖、后继 kernel 内显式 wait」换掉全依赖串行，收益明确（SGLang #51080 消三个 full dependency、vLLM #64737 fused 后 TPOT −21%），但把**「读任何前序产物前是否 wait」变成正确性不变式**；一旦漏 wait，表现为 hang/静默错值而非报错，且只在特定 kernel 相邻与 graph 捕获形状下复现。本追踪此前静默主线聚焦量化/KV/索引/权重时序，第 22 条把它扩展到 **PDL 依赖同步次序**——凡启用 PDL 的引擎/自研 kernel，须把 wait 位置纳入 code review 强制项与 SASS 级核验。

### 1.4 其他头号线索

- **SGLang kernel 包静默失败清理（#50805，3/4）**：四部分代码清理的第 3 部分（`python/sglang/kernels/**`，77 文件 +390/−986），修一批「检查了但什么都没查」的问题——条件写成 size 的 `TORCH_CHECK`、拿张量与自身比较的测试 helper、永远打印 mismatch 的 verify flag、可能返回 module 而非函数的包导出（MiniCPM-SALA `get_block_table` 在子模块先导入后变 `TypeError: 'module' object is not callable`）、按 capability major 硬写的架构门、不可达分支。属「静默失败防御性清理」主线，按官方一手 fix 级记录。
- **SGLang #45261（fix）**：DeepSeek-V4.1-Flash 在 SM120（Blackwell 消费级）TP8/EP8 默认参数下**起不来**——两个连续失败：① c1/c2 低比率 decode 总走 DeepGEMM FP4 paged MQA logits，但默认（未开 `--enable-deepseek-v4-fp4-indexer`）时 `deep_gemm_metadata=None`，直接触发 `t.is_cuda()` 断言；② 修好后 decode CUDA graph capture OOM（`Tried to allocate 994.00 MiB`，8 rank 中 4/3 报错），根因 dense-slot gather 按 graph capacity 收集 int64 slot map + fp32 score。同 PR 承接 #42371 的 decode-scoring 改动。
- **安全**：vLLM #52602（Rust 前端对未配置 modality 默认 item 上限从「无限」改为 999，与 Python 对齐，封堵 32 MiB body 内触发无界 media-task fan-out→OOM）；SGLang #50805 部分；llama.cpp #32562（畸形 limite GGUF 在 load 期越界读，`n_layer*n_mudd_taps>512` 拒绝）。延续第 18 条。

---

## 2. vLLM：2026-11-18 提交（110 条，两页 100+10）—— 静默正确性(#64737/#66679/#68359/#66011/#65353) + GLM-5.3-Flash ROCm 性能簇(#63910, TPOT −1.7~4.4%) + Kimi-K3.1-Pro/DSv4.1 recipe 簇 + Rust 前端 + 安全(#52602)

### 2.1 `#64737` [ROCm][Bugfix] AITER 共享专家融合对无 gate MoE 的路由修复（fix，强化第 4/19 条）——见第 1.2 节

### 2.2 `#66679` [Bugfix] Kimi-VL encoder CUDA graph 按 patch 设 max_seqlen（fix，强化第 19 条）——见第 1.2 节

### 2.3 `#68359` / `#66011` KV 生命周期静默错误两连（fix，强化第 8 条）——见第 1.2 节

### 2.4 `#65353` [Bugfix][gpt-oss] 流式 MXFP4 专家按整块跨 TP 切分 + 不清零调用方 w2_bias（fix，强化第 6/21 条）——见第 1.2 节

### 2.5 `#63910` [ROCm][Perf][GLM-5.3-Flash] kpool 稀疏索引器去 int32/int64 拷贝（perf，强化第 15/16 条）

GLM-5.3-Flash 在 ROCm 上，kpool sparse indexer 每个稀疏注意力层都发起 int32↔int64 转换、`+1`、`-1` fill 与 buffer copy。本 PR 让 expand kernel 直接读 int32 pool id 与 int64 position（`+1` 移入 kernel）、直写 `topk_indices_buffer`，每层省 6 次 launch、每 decode step 省 66 次。**MI355X TP4（amd/GLM-5.3-Flash-Quark-MXFP4）**：1–32 并发 median TPOT **−1.7%~−4.4%**（1k/8k/64k/128k 输入），64 并发 −0.9%~−1.8%，64k c16 持平（−0.2%）；MTP3/MTP5 median TPOT −1.6%/−1.5% geomean（acceptance 两侧相同）；decode step kernel 数 1295→1229、kernel time −308 µs。gfx942 MI325X FP8 TP4 总吞吐 +1.9% geomean（15 点全快）。精度不变：GSM8K 0.920 vs 0.921、GPQA 0.904/0.909 vs 0.914/0.894、RULER NIAH 64k 1.0。双轮换 GPU half 跑，128k c8（+2.1%，run 间差 5.7%）在噪声内；属硬件特定、launch 级优化。

### 2.6 `#67053` [Core][Mamba] 稀疏 retention 下保留 decode-crossed 边界状态（fix，强化第 5 条）

经设计重写（新设计 @ZJY0516）：请求只缓存**一个** decode-crossed 边界状态（最新），并在请求释放（finish/preempt）时 hash 一次。align 模式在跨越边界时把精确状态写进独立块，sparse retention 下 `MambaManager.remove_skipped_blocks` 保留该块而非一步后释放，下一边界跨越时再无 hash 释放；`KVCacheCoordinator.free`/`pop_blocks_for_free` 接收 Request 并对每个 manager 调 hook。`cache_blocks` 恢复与 main 一致（decode 区掩码不变），避免旧版每 preempt 泄漏一条缓存。目标：回放输出的后续轮次在稀疏 retention 下从末尾附近恢复。属 Mamba/混合模型的 KV 状态生命周期修复。

### 2.7 其他要点（标题级 / 描述级 / 性能级）

- **ROCm/DSv4.1 性能簇（标题/描述级）**：`#67906` 融合 `fused_wqa_wkv` 的 MXFP8 输入量化与共享专家进 producer；`#63254` Muse Spark token-sharded MoE reduction 收尾进一次 reduce-scatter；`#68086` OCP MX 仿真只反量化不同 routed experts；`#66541` 按 tensor 缓存 AiterFlashAttention K/V cache views；`#68310` ROCm 导出 Triton/AITER FlyDSL GDN prefill checkpoints；`#67302` ROCm 上 ModelOpt INT8_PB_WO 用 8-bit Triton WNA16 kernel；`#66031` AITER CK DASH attention for Unlimited-OCR-2；`#65595` Engram host tables 跨 DP rank 用 HIP VMM 共享。
- **KV/连接器**：`#50118` DCP full KV report 用 raw page size 而非 DCP-effective block size，在客户端 `kv_cache_report_mode=full` + 前缀命中时 `resolve_block_hashes()` 断言失败、EngineCore 终止（修为用 `manager.block_size`）；`#67497` KV Offload P2P 在 decoder 抢占/CPU 主层满导致部分 promotion 后仍报 PD cache hit，触发无法满足的二次 fetch 与 30 s 超时——GB10 实测 PD 完成 45.99→13.52 s（FCFS）、49.06→27.71 s（priority），failed promotions 3→0/10→0；`#59547` 每 rank 在 replicated layout 外独立 CPU region；`#66278` FS tier 短写按磁盘满而非 EIO 上报；`#66011`/`#68359` 见 2.3。
- **安全/RL 权重**：`#52602` 多模态 item 默认上限（见 1.4）；`#65799` direct reload 在 `finish_weight_update` 检查更新是否加载了冷启动加载的每个参数，漏载即报错并点名（漏载参数保留运行时字节并被 post-load 处理重跑，转置/重量化/padding 下是静默损坏，padded MoE 更会直接服务零值；豁免 `--sleep-preserve-parameter-names`）；`#67563` padded 参数构造期标 `fill_on_reload`；`#f55dbdfb9d` direct reload 不再重绑 word embeddings。
- **量化正确性（描述级）**：`#65288` MXFP8 selector 对 weight-only checkpoint 仍可能选内部量化激活的后端——修为按请求的 activation scheme 过滤 kernel，选 Marlin/emulation（Fixes #65193；11-18 CI follow-up 修了两个测试 setup 问题：DeepSeek MXFP8 用 `SimpleNamespace` 缺 `quantization_config`、Eagle 测试全局默认 device 未恢复导致 Qwen Omno audio 同步，`git diff --check --cached` 通过、Buildkite #103760 过但 GPU 未在 merge commit 重跑）。
- **多模态/Kimi-VL（描述级）**：`#66679` 见 2.2；`#68135` `/inference/v1/generate` 返回 `mm_hashes`；`#63369` TorchCodec 图像解码后端；`#66067` 共享 encoder CUDA graph 代码（Qwen2-VL 风格 mixin）。
- **Rust 前端簇（约 15 条，标题级）**：`#67902`（1/4）rust shm obj storage；`#61856` streaming/cancelled 请求的 HTTP metrics；`#63843` auto tool parser 不可用时返 400；`#63739` streaming chat choices 含 null `finish_reason`；`#66998` 省略 parser flags 时仍服务 gpt-oss；`#68253` gRPC 省略温度用模型默认；`#65915` `--override-generation-config`；`#67889` scale-out render 路由；`#52602` 安全（见 1.4）。
- **新模型/recipe（标题/描述级）**：`#67325` Inkling 2 Small for Blackwell；`#68487`/`#68486` MiniMax-M3.1-Flash H100/H200 AgentX TP8 DSpark preset；`#68295` HY4 Flash；`#68344` pooling entrypoint `add_special_tokens` 用 tri-state；`#52602` 见上。
- **Kimi-K3.1-Pro recipe 簇（约 6 条，标题/描述级）**：B300/GB300 AgentX 分离式 preset、DSpark draft 上线、前端 session affinity/权重加载格式，与 InferenceX 对齐，属部署配方非引擎核心。

---

## 3. SGLang：2026-11-18 提交（93 条）—— rust-server API 大栈 + Kimi-K3.1 AMD gfx950 day-0(#51000/#50999/#50997/#50998) + DSv4.1 FP8 DeepGEMM prefill(#49066) + PDL kernel(#51080) + 静默错误(#50008) + kernel 清理(#50805) + DWDP(#50842)

### 3.1 `#50008` [Fix] NVFP4 trtllm hidden size pad 到 256（fix，强化第 6 条）——见第 1.2 节

### 3.2 `#50805` [Refactor] kernels 包修静默失败与死代码清理（3/4，fix）——见第 1.4 节

### 3.3 `#51080` [Perf] 用 PDL launch qk-norm / KV-prepare / gate kernel（perf，关联第 22 条）

QSA 模型（Qwen3.8-Flash-Next NVFP4 B200）decode CUDA graph 中大部分 kernel-to-kernel 边是 PDL，但 q/k RMSNorm+RoPE+gate、QSA KV prepare、sigmoid output gate 三个 Triton kernel 无 PDL，且 `fast_topk` 每次 call 为 `row_starts` 分配 `torch.zeros` fill——每一处都是 full dependency，后继需等前序排空。本 PR 在三个 kernel 内按 `ENABLE_PDL`/`USE_PDL` 在开头 wait、用 `launch_pdl` 发射；`fast_topk` 把 `row_starts` 改 `Optional<TensorView>`，缺省时用行起点 0（空指针）不再每 call 分配 fill。两个共享 kernel 在 sm90+ 上给 Gemma QKV（Qwen3.5/InternS2/MiniCPM-V 全注意力层）、`fused_sigmoid_mul`（Qwen3.5/Qwen3-Next/ERNIE 5.5/Muse Glimmer/Hunyuan V4 输出门）同样走 PDL，其他后端 launch 不变。属第 22 条 PDL 主线的性能侧样本（正确性侧见 TRT #22089）。

### 3.4 `#51000` + `#50999` [Feat][AMD] Kimi-K3.1 在 MI350X/MI355X（gfx950）day-0（feature，强化第 11/15 条）

AMD Kimi-K3.1 系列最后一 PR（承接 #48427）：kernel PR #50997（DSA gfx950 sparse MLA + DCP top-k + FP4 index-K）、#50998（Kimi-K3 gfx950 KDA verify + A8W4 MoE + latent-tail + LL 集合 kernel）、#50999（投机解码 kernel：滑窗 grouped-head MLA draft verify、DFLASH MLA context 一次 GEMM+norm+RoPE、CUDA target-only tree sampler 的 Triton 移植、extend attention `bf16_dot` 选项）已合入，**#51000 加调用点**。全部行为变更门到 gfx950（`_is_gfx95`），CUDA/NPU/其他 ROCm 保持 main 代码；共享文件仅在 gfx950 加载 `*_hip`/`kimi_k3_amd` 模块（38 共享文件 +769/−211）。Kimi-K3.1-Pro 在 MI350X/MI355X 具与 CUDA 同 day-0 特性：DSA + decode CP + index-K 按 token shard、PD over Mooncake/MoRI、DFlash（MLA5 SWA1024 draft）、HiCache。延续 DeepSeek-V4.1（#41018-21）与 MiniMax-M3.1（#46690-93）的 AMD 布局。

### 3.5 `#49066` [Perf] DSv4.1-Flash 在 Hopper 上 FP8 DeepGEMM prefill scoring（perf，opt-in）

把 #50719 的 Hopper FP8 scoring 共享索引器从 full top-k 层/候选源扩到 prefill candidate consumers，用现有 ragged top-k 替换 source block selection；紧凑 K 每 batch 上限 65,536 向量（超出回退 Torch scorer）。`SGLANG_OPT_DSV41_SM90_DEEPGEMM_PREFILL=1` 默认关、构造 backend 时解析。历史实测（rebase 前，8×H20-3e，4 prefill/4 decode，DSpark block 5，8192-token prefill chunk，并发 24，256 请求，61,440 共享前缀+4096 请求特定+500 输出）：Baseline CP prefill + DP decode + Torch BF16 prefill scoring vs Opt FP8 DeepGEMM prefill scoring，具体数字见 PR comment #6953701024（当期未在正文列出，按描述级/历史口径记录）。

### 3.6 `#50842` [Feat] Kimi-K3 PD prefill 上 KDA 与共享专家权重的 DWDP（feature，opt-in）

attention DP 下每个 K3 PD prefill rank 持有全部 KDA 层的 `[q,k,v,g]` in-proj + o_proj（attention TP1 时 60.8 GB BF16/rank）+ EP all-to-all 下 24 GB 复制共享专家，ADP8 在 8×GB300 上勉强装下。`--kda-dwdp-size N`（仅 `--disaggregation-mode prefill`、opt-in）把这些权重跨 N 个 attention-DP rank 分片，在 copy stream 上把下一层 gather 进两个旋转 slot 之一。head-aligned layout 使 N==TP 时分片与 KDA TP rank r 权重逐字节相同；copy-engine peer pull 走 cuMem 映射，不用 SM/集合通信/副本，prefill 输出与复制权重位相同。截断 40 层 GB300 代理：DWDP4 权重 171.6 vs 192.9 GB/rank（−21.3 GB）、KV 6.35M→10.32M tokens（+62%）；TTFT/吞吐 ADP4 vs ADP4+DWDP4 接近（1k 0.117/17.4K vs 0.131/15.9K，32k 1.449/59.1K vs 1.471/58.3K），而 plain ADP8 因权重后处理 OOM 装不下、DWDP8 使其可加载。拒绝 PP/attention CP/full CUDA graph/量化分片权重/在线权重更新/非 K3；属 opt-in 部署能力。

### 3.7 `#45261` [Fix] DSv4.1 在 SM12x 以 TP8/EP8 默认参数启动（fix）——见第 1.4 节

### 3.8 rust-server API 大栈（约 20 条，标题级）

本窗口 rust-server 集中补齐 API 面：`#49748` 无状态 `/v1/responses`（走 chat 路由）；`#49747` Vertex/SageMaker generation wrapper + PUT /generate + 斜杠 model id；`#49746` `/v1/score` 及 embeddings/classify/rerank/tokenize schema；`#49745` OpenAI completions/chat schema + gRPC；`#49744` Anthropic Messages API over HTTP/gRPC；`#49743` Ollama routes；`#49742` configure_logging/dumper 路由/HiCache L3 layout/elastic EP 状态；`#49741` `/open_session`/`/close_session`；`#49740` NGRAM 外部语料路由；`#49739`/`#49738`/`#49737`/`#49736`/`#49735`/`#49734` function-call 解析、权重读取/更新组/LoRA 拒绝、pd_role_switch/update_weight_version、内部状态/profiling/专家分布/内存占用、暂停继续/freeze_gc/ping/ready、tagged payload rules。配套 `#50831`/`#50832`/`#50851`/`#50854` spec_verify 指标、请求交给调度器时刻、dp_rank=null 对齐 Python、HTTP 请求起始计时。延续第 17 条「推理服务前端 Rust 化」，本窗口为该主线单日最大规模合入。

### 3.9 其他要点（标题/描述级）

- **调度/负载**：`#51038` DP attention 下 idle rank 走 `process_batch_result` 而非 `on_idle`，导致 idle batch 被按 decode 间隔节流、DP 均衡读到的负载快照滞后——修为对 idle batch result 也强制发布（两 sink）；`#51147` sgl-router `--dp-rank-policy least_tokens` 按引擎上报 KV 均衡。
- **投机解码/图健壮性**：`#47644` DFlash2 draft sampler 折进 draft CUDA graph 后静态 buffer 按最大捕获 bucket 定尺寸，batch 超过 graph max 时 stage 越界崩（`size 4 must match 8`）——修为仅当 `bs<=max_bs` 才 stage；`#47703` trtllm_mha 在 unified memory 下读到 CUDA graph 就地构建的 strided page table view（#42930 断言），attention DP 下 draft replay + verify overflow 跑 eager 触发，修为在 reader 端 pack；两 PR 为同一 workload 的两个 crash（stack 51142）。
- **跨硬件**：`#51000`/`#50999` 见 3.4；`#9e5c4a78e8` gfx950 KDA/A8W4 kernel；`#8d72d2ea11` gfx950 DSA for Kimi-K3.1；`#48313` H20 GLM-5.3-Flash fused_moe down-projection TMA 配置；`#50023` NPU HiCache modelslim MLA fp8 KV/GQA mxfp8 KV；`#41329` CPU FP8/INT8 quant ops；`#44769` intel_xpu LoRA 后端。
- **kernel/量化**：`#49457` graph eager_on_graph 与 CUDA capture 解耦（2/3，device-neutral recorder，铺路 OOT 平台，延续第 15 条）；`#0fcd253a87` moe NVFP4 trtllm hidden pad 到 256（见 3.1）；`#50008` 见 3.1；`#51081` HiCache page-unified L3 对象名 dtype 后缀测试修复（test-only，nightly 自 11-15 失败）。
- **安全/其他**：`#51114`/`#51123` encoder 拒绝 scalar hash、超时 tick 保留结果、gRPC URL IPv6-safe；`#7946e589c5` server_args 移除废弃 flags；`#cd3045fa6e` Mooncake room transfer infos 快照 + aux 写失败拉黑 session。

---

## 4. TensorRT-LLM：2026-11-18 提交（66 条）—— v1.4.0rc4 发布(#1.1) + PDL FMHA(#22089, 新增第22条) + FC1 prologue(#22342) + BF16 PLE/NVFP4 MTP(#22473) + DSpark ancestral(#21478) + INT8 KV XQA(#22216, 2×) + 混合 KV 重构(#22121) + 大量 infra/waive

### 4.1 `#22089` [fix] trtllm-gen FMHA 在 PDL 下 wait 后再读 seq length（fix，新增第 22 条）——见第 1.3 节

### 4.2 `#22342` [fix] 解析 FC1 prologue probe 的 stream extent（fix）——见第 1.2 节

### 4.3 `#22473` [fix] 加载 BF16 PLE 表与原生 NVFP4 MTP 权重（fix，强化第 6 条）

原 base 假设 scaled FP8 PLE n-gram shard 与 FP8 block-scaled MTP 专家，未缩放的 BF16 表触发 dtype 断言、原生打包 NVFP4 draft 专家触发 FP8 shape 声明失败。修为：保留 BF16 表存储并对未缩放 BF16 初始化 unit scale；按字节定私有映射 buffer/共享 node chunk/VMM view 尺寸再恢复逻辑 `[rows, cols]`；用 `moe_experts_quant_config` 解析 draft 精度（逐层覆盖、`exclude_modules` 优先）；经现有 relayout 加载原生打包 NVFP4 操作数。BF16 保留避免有损表量化；rank mode 下 `[320001536,160]` 表约 **95.4 GiB/BF16 rank**（4 复制 rank 381.5 GiB，FP8 为 47.7 GiB/rank，为字节计算非实测峰值）。

### 4.4 `#21478` [feat] DSpark 祖先式 draft 采样 + block verification + 前缀命中时的 drafter context（feature，opt-in）

为 standalone DSpark drafter（Markov head）加 opt-in `use_ancestral_draft_sampling`：要求 rejection sampling（否则 `ValueError`）、拒绝 confidence scheduling；`rejection_sample_method`（`standard`/`block`）中 block verification 不得与 dynamic trees/relaxed thinking acceptance/DSpark confidence scheduling 同用；embedded DSpark drafter（含 deepseek_v4 embedded draft）启用时报 `ValueError`。评审指出选项名 `ancestral_draft_sampling` 为 stale、golden manifest 描述不可靠、`draft_temperature_only` 在 HEAD 不存在、源码检查未在一处覆盖所有禁用组合（needs follow-up）。属投机解码验证路径硬化（第 14 条），opt-in。

### 4.5 `#21250` [perf] Hopper 上 decode-sized batch 的 Triton FP8-activation MoE（perf，opt-in）

从 #20878（GLM-5.3-Flash FP8 on Hopper）拆出。SM90 上 FP8 block-scale MoE 走 `fp8_block_scales` CUTLASS leaf（DeepGEMM grouped kernel 单 persistent CTA/SM 遍历所有专家组），decode batch 下多数专家为空、远未达带宽上限。本 PR 加 opt-in SM90 leaf `TrtllmTritonFp8ActGroupedGemmFp8BlockScalesImpl`，仅当 `TLLM_MOE_ENABLE_SM90_TRITON_FP8_DECODE=1`、rows≤32、无 `moe_output` buffer（非 NVLinkOneSided）、BF16 无 scale 输入、所有本地专家 clamp 相同时走 Triton 路径（per-1x128 FP8 激活量化 + FP8×FP8 GEMM + 128x128 权重 scale，grid 只覆盖 batch 可能占据的专家块），否则回退 DeepGEMM leaf。默认关，环境指纹记录该 flag。

### 4.6 `#22216` [perf] INT8 KV cache 启用 XQA multi-block（perf）

INT8 KV cache 的 XQA multi-block 模式在 `attentionOp.cpp` 两处被无条件禁用（源于 2024 年 bug 4565783 workaround，#1427/#1492 后未复核；multi-block races 已在 #17976/#18141 修复）。移除 `DATA_TYPE_INT8` 子句后，长上下文 decode 从每 (sequence, KV head) 单 CTA 变多 CTA：**A10G 46k TPOT 48.4→22.0 ms（约 2×）**；op 级 kernel time（head 256，spec-dec 4 token）64k 4766→646 µs、32k 2411→348 µs，plain decode 64k 969→386 µs（1k 略慢 36.8→40.6 µs）。360-case 精度矩阵：INT8 multi-block 匹配 BF16 multi-block（max relL2 差 1.05e-8），无 NaN/Inf、重跑位相同。6 个新单测 A10G(SM86)/H200(SM90) 修前 6 failed、修后 6 passed。FP8(E4M3, SM<90) 条件不变。

### 4.7 `#22121` [refactor] 把模型族特有状态移出共享 hybrid KV cache manager（refactor，行为不变）

#21324 把混合模型投机解码 replay state 放进 KV cache region 后，KDA 已自有 `KDAReplayRegion`，而 Mamba2/GDN replay 仍由 `MambaHybridCacheManagerV2` 集中管理。本 PR 把这部分移到各模型族各自的 provider（`Mamba2ReplayRegion` for Nemotron、`GDNReplayRegion` for Qwen3.5/3-Next/4Exp），generic runtime 不再知道 replay state；region/slot 数/kernel 不变，无行为变更。延续 KV cache 管理接口重构（第 8 条）。

### 4.8 rc4 发布说明要点与 v1.4.0rc5 滚动

- **v1.4.0rc4**（见 1.1）：发布说明 What's Changed 列出约 315 条 PR，New Contributors 12 人；Mooncake 升 0.3.14（#22191）、FlashInfer Python 升 0.7.3（#21824，注意 Known Issues 中 GLM-5.2 SM103 TP8 与该版本相关）。
- **#98c7b46f5d [chore] Bump version to 1.4.0rc5**：窗口内 main 已切到 rc5 开发，rc4 为滚动 RC 中间点。其余 fix/perf 约 15 条按标题/描述级记录（`#22364` DSv4.1 Engram decode hashes + history scatter 一 launch；`#22359` DSv4.1 small-M FP8/MXFP8 GEMM + BF16 权重转换 + SwiGLU 输出；`#22366` CSA2 attention-TP Full-layer selection chain + in-kernel page mask + cluster GVR；`#22356` 并行 inverse-CDF 融合采样；`#21209` Qwen3-Next residual+RMSNorm+static FP8 量化融合——FP8 copy 较 FlashInfer 差 1 ulp、作者称更接近 fp32 参考；`#22032` GLM-5.2 sm_103 tp8 精度引擎环境隔离；`#97124490e5` 恢复 FP8 前缀缓存精度覆盖）。
- **infra/waive/test 约 30 条**：大量 waive/unwaive、post-merge waive、Modeling V2 CI 选择、artifactory 迁移（#22195）、Pyrefly 类型检查扩展（#22006）——按 infra 级记录，不逐条展开。

---

## 5. llama.cpp：2026-11-18 提交（31 条）—— 循环状态 NaN 粘滞(#31913) + GGUF/模型加载越界(#32562/#32495) + SKU 感知精度(#30672) + NORM+MUL 融合(#32454, +4.6~9.5%) + 多后端/UI

### 5.1 `#31913` 修循环状态中的粘滞 NaN（fix，强化第 19 条）——见第 1.2 节

### 5.2 `#32562` [limite] 约束 n_layer*n_mudd_taps 防加载期越界读（fix，强化第 18 条）——见第 1.4 节

### 5.3 `#32495` [gguf-py] 修 GGUFReader 张量偏移 uint64 溢出（fix，强化第 9 条）

`GGUFReader._build_tensors` 以 `data_offs = int(start_offs + offset_tensor[0])` 计算张量数据偏移，而 `offset_tensor[0]` 为 `np.uint64`，加法在 uint64 域回绕后再转 Python int。修为先把两值转 Python int 再相加，并校验张量数据在文件内。Related #25401（同函数类似 uint64 溢出）；新增 `test_tensor_offset_no_uint64_wraparound` 修前失败修后通过。

### 5.4 `#30672` [CUDA] NVIDIA GPU 的 SKU 感知 FP16/BF16 累加精度门控（perf，硬件特定）

按 GPU SKU 自动选择 cuBLAS 精度：GeForce 上合格量化 op 保留 FP16 输入+累加，专业卡/Spark 优先 BF16 输入+FP32 累加；用 NVML brand 信息分类 GPU、用 compute capability 识别 Spark（GB10，cc 12.1），保留已有 op 级精度要求与手动覆盖，避免对 GeForce 施加 FP32 累加吞吐惩罚。承接 #29173（NVFP4 on cuBLAS path）。**DGX Spark GB10，Qwen3.8-27B-UD-Q4_K_M，全 GPU offload + 强制 cuBLAS + auto precision，batch 2048/microbatch 512**：pp512 596.10→639.38 t/s（**+7.26%**）、tg128 13.44→13.43 t/s（−0.06%）。两次调用×10 重复、before/after/after/before 顺序，无 profiling。属硬件特定、微基准口径。

### 5.5 `#32454` [CUDA] 融合 NORM + MUL (+ ADD)（perf）

对连续 F32 1D 权重/偏置把 `GGML_OP_NORM + GGML_OP_MUL (+ GGML_OP_ADD)` 融进 `norm_f32`，适用于 NORMAL_TYPE 模型（embedding 编码器 bert/nomic-bert、视觉编码器 qwen3vl/siglip）。RTX 5070 Ti Laptop（SM120，CUDA 13.4）`test-backend-ops` 全过、Qwen3VL-2B greedy 前后文本一致：bge-large-zh-v1.5 F16 pp128 24117→25218 t/s（+4.6%）、pp512 37620→41212（+9.5%）；nomic-embed-v1.5 pp512 +5.4%；Qwen3-VL-2B 视觉 640×480 31.09→28.56 ms（+8.8%）、1024×1024 +6.2%、3840×1620 +3.0%。分辨率越大增益越小，输出位相同。

### 5.6 其他要点（标题级 / 描述级）

- **模型/后端**：`#32071` Zaya1 8B（Zyphra）；`#29539` `LLAMA_MOE_RANDOM_ROUTING=1`（仅基准用随机专家选择）；`#32553` d1-2B/d1-Omni-2B packing；`#32172` ggml-openvino 扩模型支持 + NPU/GPU/内存优化；`#32202` Vulkan 多 op bfloat16；`#30824` Vulkan Qualcomm 上 rms_norm 用 runtime loop count；`#31352` Pascal DP4A MMQ I tile（J=64）；`#32504` CUDA MMQ for MoE on V100；`#32492` conv2d patch index 用 fastdiv；`#30672` 见 5.4。
- **量化/精度**：`#30672` SKU 感知累加精度门控（见 5.4）；`#2cbfc3cf34` CUDA 按 SKU 对 FP16/BF16 累加精度门控（与 #30672 同簇，标题级）；`#03ec5b4326` NORM+MUL 融合（见 5.5）；本窗口量化静默错误见第 1.2 节与 5.1–5.3。
- **server/UI/工具**：`#32567` server sleep state for v1/realtime；`#32507` realtime transcription prompt；`#32485` Llama UI/Chat Storybook；`#32523` DESIGN.md tokens；`#32524`/`#32543` Llama Chat 走 `npx @ggml-org/llama-chat`；`#32527` 默认隐藏 Models/MCP；`#32535` WebGPU FA mask 跨 Q batch 广播（修 CI）；`#32548` llvm 22.1.8；`#32563` cpp-httplib 0.67.0。
- **KV/调度**：`#31913` 见 5.1；`#32562`/`#32495` 见 5.2/5.3；nightly b12503/b12521/b12527/b12528（窗口内，b12529 落 11-19 窗口外，均 prerelease）。

---

## 6. 累计结论延续与新增（继承 1–21，本期新增 22）

**累计结论（1–21 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性；③ 默认值成为主要风险载体；④ 异构/量化后端静默错误为高频（本期 vLLM #64737 AITER 无 gate MoE 错路由 GSM8K 0.095→0.661、SGLang #50008 NVFP4 hidden 未 pad 让 GSM8K 17% vs BF16 90%）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（本期 vLLM #67053 Mamba 稀疏 retention 边界状态、TRT #21478 DSpark ancestral/block verification）；⑥ 量化路径静默错误（scale/布局漏接，本期 vLLM #65353 gpt-oss MXFP4 专家跨 TP 错切片、TRT #22473 BF16 PLE 表 + NVFP4 MTP 权重）；⑦ 专家张量/路由归属与分页漏接静默错误（本期 vLLM #64737）；⑧ KV 卸载/HiCache/PD 健壮性 + 内存/缓存管理接口重构（本期 vLLM #68359 SWA tail 复制到他请求块、#66011 NIXL 过期 lease 静默读他请求 KV、#67497 P2P 抢占后重复 fetch、TRT #22121 混合 KV 状态分族）；⑨ 长上下文 int32/int64 溢出静默错误（本期 llama.cpp #32495 GGUFReader uint64 回绕、#32562 limite n_layer*taps 越界）；⑩ MXFP8 KV 正确性簇；⑪ 新旗舰模型 CP/KV/性能成为一等公民（本期 SGLang #51000/#50999 Kimi-K3.1 AMD day-0、#50842 K3 DWDP、vLLM Kimi-K3.1-Pro recipe 簇）；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径（本期 TRT #21250 Hopper Triton FP8 MoE decode leaf、SGLang #49066 FP8 DeepGEMM prefill scoring）；⑬ FP8/低比特 KV cache 跨引擎标准化落地（本期 rc4 mixed-precision KV 扩到投机/sparse MLA/冷页、TRT #22216 INT8 KV XQA multi-block）；⑭ 投机解码验证路径跨引擎密集硬化（本期 TRT #21478、SGLang #47644/#47703 DFlash under DP 两连崩修复）；⑮ 跨后端可移植性成为显式架构主线（本期 SGLang #49457 device-neutral graph 2/3、#51000 AMD gfx950、vLLM #63910/#63254 ROCm、llama.cpp 多后端簇）；⑯ 新旗舰模型进入「默认性能化」阶段（本期 vLLM #63910 去 launch、SGLang #51080 PDL launch）；⑰ 推理服务前端 Rust 化（本期 SGLang rust-server API 大栈约 20 条）；⑱ 推理服务默认安全加固（本期 vLLM #52602 多模态 item 上限、llama.cpp #32562 加载期越界拒绝）；⑲ 非量化的常规优化路径静默错误密度不亚于量化路径（本期 vLLM #66679 Kimi-VL 视觉图、#68359 KV tail、SGLang #50805 kernel 静默失败清理、llama.cpp #31913 NaN 粘滞）；⑳ 正式发版集中固化「默认值风险 × 破坏性变更」（本期 TRT v1.4.0rc4 含 2 项 BREAKING + Modeling V2 默认 auto + Marlin 默认 + 5 项 Known Issues）；㉑ 在线权重热更新的时序竞态成为静默正确性新子项（本期 vLLM #65799/#65353 direct reload 漏载/错位/清零入射缓冲区）。

**新增第 22 条（PDL 依赖同步次序成为跨引擎静默/崩溃新子项）**：TRT-LLM #22089 表明，启用 PDL 的 kernel 若在 `griddepcontrol.wait` 前读取前序产物（seq length），会读到陈旧值并导致 persistent context kernel hang 或输出损坏，且只在特定 kernel 相邻与 graph 形状下复现（SASS 级核验才能确认所有导出 cubin 合规）；与此前 #22287（读陈旧 scale）、#22258（scale 读取移到 wait 后）构成三期连续的同一主题，SGLang #51080 则从性能侧把三个 kernel 改 PDL launch。**研判（观点）**：PDL 以「显式 wait 替代全依赖串行」换取 kernel 重叠收益，但把「读任何前序产物前必须 wait」变成写 kernel 的强制不变式；漏 wait 的故障形态是 hang/静默错值而非显式报错，常规精度测试难以覆盖。凡启用 PDL 的引擎或自研 fused kernel，须把 wait 位置列入 code review 强制项，并用 SASS/cubin 级检查确认「首个 load 在首个 wait 之后」，同时审计为 PDL hang 引入的 workaround（如 #19873 已被本 PR 回滚）。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 静默正确性复核（KV 布局/lease/路由/视觉）**：SGLang `#50008`（NVFP4 hidden 2688 未 pad 到 256 → GSM8K 17% vs BF16 90%；升级 FlashInfer ≥0.7.4 并核对 `trtllm_nvfp4_hidden_alignment`）、vLLM `#64737`（AITER 共享专家融合在 softmax/无 gate MoE 上错路由，DeepSeek-V2-Lite/Mistral-Large-4）、`#66679`（Kimi-VL encoder CUDA graph 按 patch 设 max_seqlen）、`#68359`/`#66011`（KV Offload final SWA tail 复制他请求块；NIXL P/D 过期 lease 静默读他请求 KV，升级后 `NIXL_CONNECTOR_VERSION=17`）、TRT `#22342`（FC1 prologue probe stream extent 未解析）、llama.cpp `#31913`（循环状态 NaN 粘滞）。
- **P0 — PDL 依赖同步（第 22 条）**：TRT `#22089`——凡用 trtllm-gen context FMHA 或自研 PDL kernel，确认所用版本含「读 seq length 前 `griddepcontrol.wait`」修复；自研 PDL kernel 须把 wait 位置列入 review 并做 SASS 级核验；移除 PDL hang workaround 前确认根因已修（#19873 已回滚）。
- **P0 — 发版/RC 升级检查（第 20 条）**：TRT-LLM `v1.4.0rc4`——2 项 BREAKING（#20498/#20703 checkpoint I/O string enum）、Modeling V2 默认 `auto`（#21712）、dense NVFP4 默认 Marlin（#21434）、弃用 KVCM V1/C++ transceiver（#21525）；对照官方 5 项 Known Issues（DS-R1 FP4 解聚 B300 CUDA graph 错输出、GLM-5.2 NVFP4 SM103 TP8 init、Kimi K3 NVFP4 KV B300 init、K3 DSpark MegaMoE CuTe TEP GB300 warmup hang、Qwen3.8-Flash-Next FP8+PLE+ADP+MTP init）逐项验证；窗口内 main 已 bump rc5，避免把 rc4 当终点。
- **P1 — KV 容量/前缀/性能权衡**：SGLang `#50842`（K3 DWDP KV +62% 但单 TTFT 略升、仅 prefill/opt-in）、`#47644`/`#47703`（自定义 `--cuda-graph-max-bs-decode` 低于 running-request cap 或单 rank >512 请求时 DFlash/attention-DP 两连崩，需升级）、`#51038`（DP attention 负载快照滞后）；vLLM `#67497`（抢占后 PD 完成 45.99→13.52 s）、`#67053`（Mamba 稀疏 retention 边界状态）。
- **P1 — 跨硬件性能（口径特定）**：vLLM `#63910`（MI355X GLM-5.3-Flash TPOT −1.7~4.4%、kernel −308 µs；MI325X +1.9%）、SGLang `#51080`（PDL launch，sm90+）、TRT `#22216`（INT8 KV XQA multi-block，A10G 46k TPOT 48.4→22.0 ms，1k 略慢）、`#21250`（Hopper Triton FP8 MoE decode opt-in）、llama.cpp `#30672`（DGX Spark pp512 +7.26%、微基准）、`#32454`（NORM+MUL 融合 pp512 +9.5%、大图增益收敛）。关注硬件/后端特定性与吞吐/延迟/内存权衡。

---

## 7. 社区与视频资源（HN Algolia，2026-11-18 epoch 窗口 1794960000–1795046399）

> 以下均为社区/Show HN，非四家引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词检索中 `sglang` / `llama.cpp` / `TensorRT-LLM` / `speculative decoding` / `inference engine` **均 0 命中**；`vLLM` 34 条多为泛 AI 资讯与模糊「LLM」匹配；`KV cache` 1 条为下述 QuadGate。

- **vLLM Xgrammar 基准（Show HN）**：「Show HN: Production Grammar-Guided Decoding in vLLM with Xgrammar Benchmarks」（**1 分**，2026-11-18T08:16），非 vLLM 官方一手、基准数字未经核验。**二手线索/未核验**。
- **QuadGate（Show HN）**：「QuadGate – 4-gate control plane that keeps agent KV prefix caches warm」（**1 分**，2026-11-18T01:48），非四家引擎官方、无一手性能证据。**二手线索/未核验**。
- 其余 HN 命中（如「LLM assisted reverse engineering」38 分、「LLMs can read and write executables byte by byte」5 分等）为泛 AI 资讯，与推理引擎实现无直接关系，未列入。本期**无引擎相关的可确认官方视频**；Reddit / YouTube / X 未采集。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **110**（分两页 100+10，page2 返回 10 条，合并去重后取全）/ SGLang **93** / TRT-LLM **66** / llama.cpp **31** 合计 **300** 条（2026-11-18 UTC 窗口；周三，工作日正常量）。与上一期（2026-11-18 档案覆盖 11-17）连续，无空档。
- **版本核验**：releases API 确认——**窗口内有新 prerelease**：TRT-LLM **v1.4.0rc4**（2026-11-18T19:16:49Z，prerelease，62,024 字符发布说明全文核验，含 5 项 Known Issues、2 项 BREAKING、Modeling V2 默认 auto、dense NVFP4 默认 Marlin）；窗口内 main 另有 `Bump version to 1.4.0rc5`（#98c7b46f5d），rc4 为滚动 RC 中间点。**无新正式版**：vLLM v0.34.0（11-16）、SGLang v0.5.23（11-02）、llama.cpp v1.0.0（11-13）；llama.cpp 推进 nightly b12503/b12521/b12527/b12528（prerelease，b12529 落窗口外）。发布说明为精选 highlights，按本追踪方法论不作为「修复已进版」依据。
- **PR 正文来源**：**40 个关键 PR 正文经官方 issues API 抓取**（markdown 官方一手；vLLM 12 + SGLang 13 + TRT-LLM 10 + llama.cpp 5），配额充足（core 51/60 起）、无 IncompleteRead、无降级为标题级；所有 PR 号均在窗口 commits API 中确认为已合入 main 的提交。其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度。
- **描述级/未核验标注**：vLLM `#63910` 性能为作者在 PR 正文报告的双轮换卡微基准（128k c8 在 run 间噪声内）、`#64737` GSM8K 提升为 fix 前后对照（Mistral-Large-4 为本地实现非 main 模型）、`#65288` GPU 测试未在 merge commit 重跑、Kimi-K3.1-Pro recipe 簇按标题/描述级；SGLang `#49066` FP8 DeepGEMM prefill scoring 为 rebase 前历史口径且数字在 PR comment（正文未列）、`#50842` 为截断 40 层 proxy 且 opt-in、`#51081` 为 test-only、`#50805`/`#45261`/`#47644`/`#47703` 按官方 fix 级记录（部分 CI 未全绿）；TRT `#22089` SASS 核验为作者报告、`#21209` FP8 copy 与 FlashInfer 差 1 ulp（作者称更接近 fp32 参考）、`#21478` 评审指出 stale 选项名/manifest 不可靠/禁用组合未一处覆盖（needs follow-up）、`#22473` 表大小为字节计算非实测峰值、`#21478`/`#21250` 均 opt-in，大量 infra/waive；llama.cpp `#30672`/`#32454` 为硬件特定微基准（无 PPL/精度对照，`#30672` 仅 pp512 提升、tg 持平）、`#31913`/`#32562`/`#32495` 附复现与测试但为作者自述。
- **社区信号**：HN Algolia 引擎官方更新类故事无；Xgrammar 基准（1 分）与 QuadGate（1 分）均按二手线索/未核验记录；无引擎相关的可确认官方视频。
- **技能检索（find-skills / agent-browser）**：find-skills 经 SkillHub 语义检索（EN+ZH 各 1 组，含「LLM inference GitHub commit monitoring daily hotspot digest」「大模型推理 官方提交采集 热点归档 git 同步」）返回 `编程专家Pro.Skill`/`育儿大师.Skill`/`多引擎搜索`/`Agent Browser` 等通用编程/育儿/搜索类技能，最高相关性 **0.050**，与「官方一手提交 API + 证据分级归档 + 指定目录归档 + git 提交推送」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/issues API + PR 页面）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次一致。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/CI 未全绿/作者报告/needs follow-up/微基准/prerelease」。
