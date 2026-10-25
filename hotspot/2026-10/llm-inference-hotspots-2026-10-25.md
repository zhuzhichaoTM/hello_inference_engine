# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-25（北京时间）
- **关注窗口：** 2026-10-24 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-24 08:00 – 2026-10-25 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-24 UTC 窗口，分别返回 **26 / 79 / 14 / 19** 条，均 HTTP 200）、四家官方 releases API、**22 个 PR 正文**（经 GitHub issue API 抓取，返回 markdown 文本，含 vLLM `#62102`/`#63001`/`#61791`/`#62713`/`#61482`/`#63319`、SGLang `#44029`/`#42929`/`#46176`/`#41169`/`#45758`/`#46242`、TensorRT-LLM `#20823`/`#20517`/`#19961`/`#20270`/`#20774`、llama.cpp `#31097`/`#30924`/`#31055`/`#30915`/`#30656`）。**窗口特征：** 2026-10-24 为**周六**，vLLM（26）/ TRT-LLM（14）提交量显著低于工作日中位，SGLang 仍高达 **79**（diffusion + DSv4.1 性能潮持续，多为在途性能 PR 的合入，详见第 3 节），llama.cpp（19）为常态周六量。HN Algolia 公开 API 可访问（HTTPS 200），窗口内 `vllm` 6 条命中**全部为「LLM」模糊匹配，无一条指向 vLLM 引擎本身**；`llama.cpp` 1 条为无关的 JVM 本地推理替代方案；`sglang`、`TensorRT-LLM`、`speculative decoding`、`KV cache` **均 0 命中**。Reddit、YouTube、X 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** 四家窗口内**均无新正式版本**。vLLM 仍为 **v0.32.0**（2026-10-18），SGLang 仍为 **v0.5.22**（2026-10-19），TensorRT-LLM 仍为 **v1.4.0rc1**（2026-10-22，prerelease）；llama.cpp 滚动 nightly 共 **15 个标签**（b11850–b11869，均发布于 2026-10-24，已纳入 UTC 窗口）。无新版本事件，故本期头号事件为跨引擎「静默正确性」聚类（见第 1 节）。
- **本期方法论沿用：** 继续用 GitHub commit API 的 UTC 日期窗口 + PR 正文经 issue API 抓取（返回干净 markdown，比 HTML div 配对更省去截断处理）；延续「版本核验」口径：发布说明/描述不等于「修复已进版」，且对 PR 实际 diff 与描述不一致者（如 TRT-LLM `#19961`）按「描述级/未核实实现落地」记录。

---

## 1. 版本状态与本期头号事件：周六窗口「静默正确性」聚类——量化链路的漏接

**主题**  本期无新版本发布，但跨引擎出现**两条「无报错却产出错误结果」的量化链路缺陷修复**，是上期累计结论第 4 条（异构后端静默错误）的同源延伸，只是落点从「后端位/精度/同步陷阱」转移到「量化（scale/反量化）链路的漏接/漏应用」。

**内容详情（取自 PR 正文）**

- **TensorRT-LLM `#20823` —— NVFP4_AWQ 预量化 scale 加载但未应用（静默数值偏移）：** MegaMoE CuteDSL（`TrtllmCutedslMegaMoeNvfp4Impl`）对 NVFP4_AWQ / NVFP4_ARC checkpoint 的 raw activation 做量化时，**逐通道的 `pre_quant_scale` 被加载却从未应用**：FC1 因此跑在「未平滑（unsmoothed）的输入」上，而权重与 input scale 都是按平滑后校准的，于是输出**与 CUTLASS、TRTLLM-Gen 在同一 checkpoint 上偏离，且不抛出任何错误**。根因链路：解析阶段把 `NVFP4_AWQ`/`NVFP4_ARC` 折叠成 `NVFP4`，MegaMoE 的 gate（仅 `has_nvfp4()`）放行该层；`process_weights_after_loading` 把 `fc31_act_scale` 写进了状态，但 `quantize_input` 直接把 raw chunk 交给 `fp4_quantize`，kernel 侧从不读 `fc31_act_scale`。该修复把 scale 真正接入 `quantize_input`。
- **llama.cpp `#31097` —— MiniMax-M3 MoE 量化 scale 从未前向到图（静默「token soup」）：** `MiniMaxM3ForCausalLM` checkpoint 携带 NVFP4 routed experts 与 MXFP8 dense 权重，通用 scale 解析正确装载到 `ffn_*_exps_s` / `ffn_*_s` / `ffn_*_shexp_s`，但 `src/models/minimax-m3.cpp` **从未把这些 scale 前向给图**，`build_moe_ffn()` 默认 `nullptr`，导致 `build_lora_mm_id()` 不进入 `w_s` 分支，routed-expert 权重被当成 `q` 而非 `q * block_scale * G`（M3.1 Flash 的 `G ≈ 15123`）。**MoE 分支压过残差，输出是「token soup」，同样无报错。** 修复在三个 FFN call site 前向全部 9 个 scale 张量（仿 `flint`/`deepseek32`/`qwen3moe` 既有模式），对无 scale 的 checkpoint 为 no-op。

**应用场景**  NVFP4 / NVFP4_AWQ / NVFP4_ARC（TRT-LLM MegaMoE CuteDSL）与 GGUF NVFP4 / MXFP8 MoE（llama.cpp MiniMax-M3·M3.1 Flash）量化部署；两者都是「量化 scale 链路漏接 → 无断言的错结果」，对生产服务的危害在于**不会触发任何异常、只能靠对拍或人工抽查发现**。

**研判**
- **事实：** 两条 PR 均为「加载了量化 scale 但未接入计算图/量化入口，输出偏离基准且无报错」的静默错误，分别由 xxi-nv（TRT-LLM, #20823）与 origin-labs-ai（llama.cpp, #31097）提交，正文均给出明确根因链路与复现条件。
- **观点：** 这是上期累计结论第 4 条（异构后端静默错误）的**第二子类**——量化链路静默错误。差异在于：后端 miscompile 多与 ROCm/Blackwell/Rubin 强相关，而量化静默错误与前后端无关、纯属「scale 前向/应用」的代码疏漏，因此**任何量化部署（不限于异构硬件）都应把「与黄金参考逐层对拍」列为上线验收**，因为这类错误不触发断言。本期将其立为累计结论第 6 条。

**来源**
- TRT-LLM：`https://github.com/NVIDIA/TensorRT-LLM/pull/20823` （merge 在 2026-10-24 窗口内；正文已核验）
- llama.cpp：`https://github.com/ggml-org/llama.cpp/pull/31097` （创建 2026-10-24T13:08:50Z，Fixes #31063；正文已核验；作者注明端到端需在（尚未公开）M3.1 Flash GGUF 上复现）

**证据等级：** 官方一手（GitHub 提交 API + PR 正文）。两条均为「修复类 PR 自述 + 根因链路」，未做独立数值对拍，但根因与复现条件描述具体、可被后续版本核验；其中 llama.cpp `#31097` 作者明确「runtime 验证需未公开 GGUF，未能本地端到端复现」，按规范记为**官方一手但端到端未在公开 checkpoint 复现**。

---

## 2. vLLM：10-24 提交（26 条）——ROCm/AMD 性能潮、投机/量化正确性与 KV 传输连接器

**主题**  vLLM 在 2026-10-24 UTC 窗口合并 **26 条**提交（周六量，低于工作日中位），主线为 **ROCm/AMD 性能**（MI325X/MI355X mono decode、MXFP4 GEMV、MiniMax-M3 Indexer CP）、**投机/量化正确性**（ModelOpt 混合精度 DFlash/DSpark drafts、batch-sharded adaptive verification）、以及 **KV 传输连接器 UMBP 第 3/10 阶段**。本期无「崩溃/静默」级修复，但量化与 LoRA-MoE 重载路径各有一处正确性修复。

### 2.1 `#62102` GLM-5.3：`concat_and_cache_mla` 跳过 scale 计算，B300 上 E2E TPOT +1.2%~3.0%

**主题**  `[GLM 5.3 Perf] Skip scale calculation for concat_and_cache_mla`（`#62102`）。

**内容详情（取自 PR 正文）**  在 Blackwell B300、TP4、`--enable-prefix-caching`、MTP（`num_speculative_tokens=5`）下跑 `vllm bench serve` 与 `aime25`：吞吐侧 E2E TPOT 提升 **1.2%~3.0%**（bench 输出 tok/s 由基线上升）；精度侧 `aime25 exact_match` 基线 0.8333 与改后 0.8333 一致（±0.0692）。作者 yewentao256，PR 标注「AI assistance was used to draft this description」且由提交人复核。

**应用场景**  GLM-5.3 在 B300/Blackwell 上的 decode 路径（MLA concat_and_cache 的 scale 项可省）。

**研判**
- **事实：** 这是一项有逐卡 bench + 精度对拍佐证的性能改动，TPOT 小个位数提升、精度不变。
- **观点：** 与上期 ROCm 侧「静默错误」形成对照——本条是**良性性能优化**，说明 vLLM 对 GLM-5.3 MLA 的热路径仍在微雕。对 GLM-5.3 部署方属「白拿」收益，但幅度小，优先级低。

**来源**  https://github.com/vllm-project/vllm/pull/62102 （merge 在 2026-10-24 窗口内；正文已核验）

### 2.2 `#63001` 修复 ModelOpt 混合精度 DFlash/DSpark draft 加载（量化 × 投机解码）

**主题**  `[Bugfix][Quantization][Spec Decode] Load ModelOpt mixed-precision DFlash and DSpark drafts`（`#63001`，合并了早期未合入的 #36312 解析修复）。

**内容详情（取自 PR 正文）**  修复嵌套 `MIXED_PRECISION` checkpoint 的解析，使 DFlash/DSpark draft 的量化设置匹配权重别名、runtime module prefix 与 decoder layer offset；并对 FNUZ GPU（如 AMD MI300）做 per-tensor FP8 权重与静态 input scale 的 FN→FNUZ 归一化。AMD CI 暴露了既有 per-tensor FP8 loader 的遗漏：两个 checkpoint 回归在引擎初始化时因 `input.dtype == weight.dtype` 失败（MI300 量化 CI buildkite #96958）——修复在 requant 前应用既有 FN→FNUZ 归一化（含权重与静态 input scale），并新增 24 个 CPU 检查 + 回归测试。

**应用场景**  用 ModelOpt 混合精度 checkpoint 跑 DFlash/DSpark 投机解码；特别影响 AMD MI300（FNUZ）部署。

**研判**
- **事实：** 这是「量化 × 投机解码」交叉路径的正确性修复，AMD FNUZ 路径此前会初始化失败；修复后走既有归一化 helper，未改 FP8 kernel。
- **观点：** 与上期累计结论第 5 条（前缀缓存 × MoE 路由 × 投机解码交叉债务）呼应——投机解码的量化 draft 是另一条「默认配置下易踩、无显眼报错」的链路，AMD 用户应确认已含本修复。

**来源**  https://github.com/vllm-project/vllm/pull/63001 （创建 2026-10-22T00:40:22Z，merge 在窗口内；正文已核验）

### 2.3 `#61791` UMBP [3/10]：共享 lookup / transfer / hybrid / observability（KV 传输连接器）

**主题**  `[UMBP][3/10] Add shared lookup, transfer, hybrid, and observability`（`#61791`，从 #58568 切分，配合 RFC #57187）。

**内容详情（取自 PR 正文）**  在共享 KV connector 之上新增：紧凑整块 load/store 计划、transfer cancellation、异步 lookup 与 hybrid/Mamba 组协调；full-attention 命中保持 prefix-closed，Mamba/sparse 组自行恢复自身状态；新增 partial hash 命中、KV events、驱逐上报与 transfer metrics。纯实现层，行为留在共享 connector 之上、不触及任何 runtime；测试在后续 PR（`test_umbp_async_lookup.py` / `test_umbp_events.py` / `test_umbp_hybrid.py`）。

**应用场景**  vLLM 的 unified memory-backed prefix (UMBP) KV 传输/分层连接器——前缀缓存的跨实例 lookup、传输取消与可观测性。

**研判**
- **事实：** 这是 UMBP 系列第 3/10 阶段，聚焦「共享连接器」的能力补全，不含运行时耦合。
- **观点：** 与前几期 vLLM 的 KV 分层/前缀缓存主线一致——KV 传输正从「能命中」走向「可取消、可观测、可混合 hybrid/Mamba 状态」，是 PD/分层 KV 落地的工程成熟度信号。

**来源**  https://github.com/vllm-project/vllm/pull/61791 （创建 2026-10-15，merge 在窗口内；正文已核验）

### 2.4 `#62713` / `#61482` ROCm：MI325X 开 `--monokernel`、MI355X 独立 mono decode 层（DSv4.1-Flash / V4-Pro）

**主题**  `[DeepSeek][ROCm] V4.1-Flash: enable --monokernel on MI325X`（`#62713`）与 `[ROCm][Perf][DSv4] Standalone mono decode layer for DeepSeek-V4-Pro on MI355X`（`#61482`）。

**内容详情（取自 PR 正文）**
- `#62713`：把 `--monokernel` 在 MI325X 的 TP recipe 打开（gfx942 mono decode 来自 #60731，仅 TP4、decode 步最多 42 行 = 7 请求 × 5 DSpark draft）。MI325X 在 `--gpu-memory-utilization 0.9` 下 KV 池仍留 11.18M tokens；MI300X 因 192GiB 下仅余 ~2.7GiB 仍关闭该 flag。`0.31.0` 拒绝 `--monokernel`，故 MI325X pip 标签隐藏。
- `#61482`：DeepSeek-V4-Pro 在 MI355X（gfx950）的 mono decode 层——一个 decode 步作为两个持久 FlyDSL launch（TP8），opt-in `--monokernel`，独立 `vllm/models/deepseek_v4/amd/mono/` 包（基于 ROCm/ATOM 参考实现）。8×MI355X、TP8、DSv4-Pro-0813、DSpark K6 下：tok/s/user **c1 +28.7%（221.45→285.03）、c4 +26.5%（205.30→259.79）**、c8 亦正向，错误 0；复用了 InferenceX 历史基线（未重跑基线）。

**应用场景**  DeepSeek-V4.1-Flash（MI325X）与 DeepSeek-V4-Pro（MI355X）在 ROCm 上的 mono-kernel decode 加速。

**研判**
- **事实：** 两条均为有实测吞吐表的 ROCm 性能改动；`#61482` 在 c1/c4 给出 +26.5%~28.7% tok/s/user，且声明复用历史基线（非全新基准），精度维度未在表中给出。
- **观点：** 与上期 `#63268`（gfx942 Triton decode 静默错）对照——ROCm 侧一边在修静默错误、一边在推 mono-kernel decode 性能，说明 **MI300/MI355 是 vLLM ROCm 当前的双线战场**。MI300X 不开 `--monokernel` 的显存余量约束值得记：部署方升级 vLLM 版本时不要假定该 flag 在所有 gfx942 卡上默认可用。

**来源**  https://github.com/vllm-project/vllm/pull/62713 （创建 2026-10-20，merge 窗口内）；https://github.com/vllm-project/vllm/pull/61482 （创建 2026-10-13，merge 窗口内；正文已核验）

### 2.5 其余值得记的 vLLM 提交

- **`#63319`** `[Bugfix][LoRA][MoE] Process reloaded weights of LoRA-wrapped MoE layers`——开启 LoRA 时，layerwise 权重重载（`reload_weights`，用于 RL 权重同步）会跳过量化 MoE experts 的 post-load 处理；修复让 `FusedMoEModularMethod.process_weights_after_loading` 委托给被包装的量化方法。main 上重载会因形状不匹配失败；测试 `test_reload_weights_with_lora_moe` 在 GB300 + Qwen3-30B-A3B W4A16 上验证 base/adapter 输出一致。
- **`#58466`** `[Sampling] Add adaptive verification support to batch-sharded sampling`——batch-sharded 采样加入自适应验证支持（投机解码验证侧）。
- **`#63466`** `[Kernel] Bump up FlashInfer to 0.7.2.post1`——依赖 bump。
- **`#62850`** `[Perf][Multimodal] Make encoder cache free linear in items per request`——多模态 encoder 缓存复杂度优化。
- **`#63428`** `[Doc][Recipe] MiniMax-M3.1-Flash: DEP2 + DEP4 PD on GB200/GB300, 7-token FLASHINFER_TS DSpark`——官方 recipe 文档（PD 解耦部署）。
- **`#63045`** `[Kimi K3] KDA RecoverSSM: CuTe verify and commit kernels, records outside the KV cache`——Kimi K3 的 KDA/SSM 恢复。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文）。所有吞吐/TPOT/命中率数字均为各 PR 自测口径，非统一基准；`#62102`/`#62713`/`#61482`/`#63001`/`#61791`/`#63319` 正文细节为本次独立抓取核验。

---

## 3. SGLang：10-24 提交（79 条）——diffusion 性能潮、DSv4.1 prefill 图、KV 分层 0SM Allgather

**主题**  SGLang 在 2026-10-24 UTC 窗口合并 **79 条**提交（周六仍高量），构成两大块：**① diffusion/DiT serving 性能潮**（约占 1/3 以上，集中在 FA4 over cuDNN、VAE decode 切分、breakable CUDA graph 默认开、host sync 去除）；**② DSv4.1 快路径与 MoE/投机解码性能**（KV 分层 Allgather、paged KV 反量化向量化、DSpark decode 开销缩减、MegaGate 路由在 prefill 的回退）。另含 AMD GPT-OSS W4A8 MoE 后端开关。

### 3.1 `#44029` KVShard：0SM KV Cache Allgather（KV 分层/PD 通信优化，默认开）

**主题**  `[Perf] KVShard: 0SM KV Cache Allgather for KV Cache Sharding`（`#44029`）。

**内容详情（取自 PR 正文）**  分片池的逐层 prefix allgather 原先用 NCCL ring/LL kernel，与 attention/GEMM 重叠时会争抢 SM 与 HBM 带宽。新开关 `SGLANG_ENABLE_KV_SHARD_ZERO_SM_GATHER` 启用**零-SM** gather 且**默认开启**：communicator 以 `NCCL_CTA_POLICY_ZERO` 创建，让 NCCL 在 copy engine 上跑 in-place allgather 而非 SM kernel。两个 scratch slot 从单一 `ncclMemAlloc` 对称窗口切出（4KiB 对齐、跨 rank 一致）；引擎在单节点部署时默认 `NCCL_CUMEM_ENABLE=1`。回退条件：开关关闭 / 启用 memory saver 或自定义内存池 / NCCL < 2.28 / 跨多节点（跨机走未验证的 hierarchical RMA）。window 注册返回 NULL 时记一条 warning 并回退 Torch 分配。

**应用场景**  SGLang 的 KV Cache Sharding（KV 分片 / PD 解耦场景下的跨 rank KV 聚合）。

**研判**
- **事实：** 一项「把 KV allgather 从 SM kernel 挪到 copy engine」的通信优化，默认开启、带明确回退边界（多节点未验证是已知限制）。
- **观点：** 这是 SGLang 在 KV 分层/PD 通信上的持续投入（与 vLLM UMBP、TRT-LLM KVCM V2 同源方向）。对单节点多卡 KV 分片部署属「默认即收益」；**多节点跨机场景须确认未误入未验证的 RMA 路径**，应保留 Torch 分配回退。

**来源**  https://github.com/sgl-project/sglang/pull/44029 （创建 2026-10-13，merge 在窗口内；正文已核验）

### 3.2 `#42929` dsv4：paged KV cache 反量化向量化（4 token/program，~4.3–4.4× kernel）

**主题**  `[Perf] dsv4: Vectorize paged KV cache dequantization`（`#42929`）。

**内容详情（取自 PR 正文）**  V4 paged KV-cache dequant kernel 原每 program 处理 1 token（长压缩历史 gather 时发小 load）。改后每 program 处理 **4 token**，保持逐元素反量化算术与 BF16 RoPE copy：`[4,8,64]` FP8 load + `[4,8]` scale load + `[4,64]` BF16 RoPE load，mask 掉 padded scale tile 与不完整 token 组。B200 空闲机、page size 256、V4 layout 的**历史**测量（rebase 前）：gathered 262K 行 0.381→0.089 ms、524K 行 0.754→0.171 ms（约 **4.3–4.4×**），16K 行仍 launch-bound。端到端 TTFT 未对 rebase 后基线重测；V4.1 layout 仍走各自既有 dequant kernel。

**应用场景**  DeepSeek-V4 压缩 KV 缓存的反量化（长上下文 prefill 的 KV gather 路径）。

**研判**
- **事实：** kernel 级向量化带来 ~4.3–4.4× 反量化加速（历史测量，作者未重测 rebase 后端到端 TTFT）。
- **观点：** 与本期 `#44029`（KV 传输）同属「KV 链路微优化」主线；SGLang 对 DSv4/DSv4.1 的 KV 处理在做系统性降本（反量化、Allgather、prefill 图融合）。

**来源**  https://github.com/sgl-project/sglang/pull/42929 （创建 2026-10-07，merge 在窗口内；正文已核验）

### 3.3 `#46176` / `#41169` 投机解码：DSpark decode 开销缩减 + 模拟验收张量融合

**主题**  `[Perf] dsv41: Trim per-step DSpark decode overhead`（`#46176`）与 `[Perf] spec-decoding: fuse simulated speculative acceptance tensor preparation`（`#41169`）。

**内容详情（取自 PR 正文）**
- `#46176`：在 DeepSeek-V4.1 上削减若干每步 DSpark decode 小开销——`mhc_stream_mean`（Triton kernel 对 4 条 hc stream 做 in-order fp32 均值，未验证 shape/dtype 回退 `x.mean(dim=1)`）、aux packing（V4 在 `AuxHiddenStatePacker` 直接写槽、logits processor 不再 `torch.cat`）、markov draft step（单 rank 持有整词表时跳过 identity vocab gather）、draft `wo_a`（Blackwell 上 TP4 fast path 外的 verify 用 strided bmm 替代 `einsum`+转置拷贝）。**全部 bitwise 相等、无输出位变化**；新 `test_mhc_stream_mean.py` 把 kernel 钉到 `x.mean(dim=1)`（含 inf/NaN/溢出/全 -0 行）。不开 DSpark 时无任何变化。
- `#41169`：把验收长度模拟的若干 device 张量填充/拷贝融合（保留既有 CPU 验收采样与随机序列）；B200 上 24 项分支测试通过，官方 `sgl-eval` 0.1.0（temp1/top-p1、max 131072 tokens、128 并发、thinking on、无验收模拟）下 Normal decode 229/240=95.42%、NEXTN 3步/top-k1/4 draft 228/240=95.00%，真实 MTP 验收长度约 2.146。

**应用场景**  DeepSeek-V4.1 的 DSpark 投机解码每步开销缩减；模拟验收（speculative acceptance）张量准备融合。

**研判**
- **事实：** 两条均为「bitwise 相等」的性能改动，#46176 明确「未改变任何输出位」，#41169 有整合精度表佐证。
- **观点：** SGLang 对 DSv4.1 投机解码在做「零精度代价」的微优化（与 vLLM `#62102` GLM TPOT 微雕同一思路），说明头部引擎在「已正确」之后进入「逐位抠开销」阶段。

**来源**  https://github.com/sgl-project/sglang/pull/46176 （创建 2026-10-23T22:31:23Z，merge 在窗口内）；https://github.com/sgl-project/sglang/pull/41169 （创建 2026-09-24，merge 窗口内；正文已核验）

### 3.4 `#45758` [AMD] GPT-OSS：gfx1250 W4A8 MoE 后端开关 + `#46242` 回退 MegaGate prefill 路由

**主题**  `[Perf] [AMD] GPT-OSS: Pick the gfx1250 W4A8 MoE backend`（`#45758`）与 `[Fix] dsv41: Revert MegaGate routing in prefill`（`#46242`）。

**内容详情（取自 PR 正文）**
- `#45758`：给 gfx1250 W4A8 MoE 路径（aiter `moe_gemm_a8w4`）加两个开关——`SGLANG_GFX1250_W4A8_BACKEND=triton|gluon`（传 `backend=` 到两个 expert GEMM）、`SGLANG_GFX1250_W4A8_KERNEL_GATHER=1`（少 downcast 行、GEMM1 已知 `gather_idx` 避免拷贝与量化开销）。配新 recipe 的实测：Output Throughput **+30%**、TOPT **−20%**、TTFT **−40%**。
- `#46242`：**回退 `#46232`**——`SGLANG_OPT_DEEPGEMM_MEGA_GATE=1` 让 MegaGate 在 extend batch（含 breakable prefill CUDA graph）上重跑，但带回了 #45969 的 prefill capture 失败：GB300（DSv4.1-Flash、DP attention 4 + MegaMoE、prefill graph 4096/8192 行）上第二个 shape 的 capture 以 `CUDA error: unspecified launch failure` 在非 0 DP rank 失败、服务起不来（用 `sgl-deep-gemm` 0.2.3.post2 可复现）。回退恢复 decode/target-verify gate 与 2048 行上限；prefill MegaGate 待 capture 失败理清后再回。

**应用场景**  AMD gfx1250（如 MI3xx）上的 GPT-OSS W4A8 MoE；DSv4.1 在 GB300 的 MegaGate/DeepGEMM prefill 图。

**研判**
- **事实：** `#45758` 是有实测三项的 AMD 性能开关（作者未给独立基准 A/B，仅「配新 recipe」前后对比）；`#46242` 是一次明确的「性能优化引入 prefill capture 崩溃 → 回退」事件，且回退说明中复现条件具体（GB300/DP4/MegaMoE/特定 shape）。
- **观点：** 这两条合起来是本期「**性能优化 ↔ 图捕获稳定性**」张力的缩影：AMD 侧在推 W4A8 收益，DeepGEMM MegaGate 侧则因 prefill CUDA graph capture 失败被迫回退。**开启 `SGLANG_OPT_DEEPGEMM_MEGA_GATE` 的 DSv4.1 用户须确认已停在 decode/target-verify gate + 2048 行上限**，否则 GB300 服务可能起不来。

**来源**  https://github.com/sgl-project/sglang/pull/45758 （创建 2026-10-22，merge 窗口内）；https://github.com/sgl-project/sglang/pull/46242 （创建 2026-10-24T04:14:48Z，merge 窗口内；正文已核验）

### 3.5 SGLang 其余值得记的提交（diffusion 性能潮 + DSv4.1 图融合）

- **diffusion/DiT serving 性能潮（约 1/3+）：** `#46046`（去 per-step host sync/uploads）、`#46171`（host 上 gather 空间并行 decode 高度）、`#46221`（idle loop 不再饿死 async-save 回复）、`#46146`/`#46107`/`#46102`（长自注意力在 B300 上优先 FA4 over cuDNN SDPA、子 tile 侧路由到 FA4、避免回落 sm80 SDPA）、`#46149`/`#46148`（Wan spatial-parallel VAE decode 提速、latents 直 patchify 成 token）、`#45977`/`#45965`（LTX-2 两阶段 breakable CUDA graph 默认开、按 CFG rank 做 Ulysses 切分）、`#46101`（qwen-image-2.1 把图像前缀 K/V 留在 packed buffer 跨步复用）。这些共同指向 SGLang 在** diffusion/视频生成 serving 上的系统性性能投入**。
- **DSv4.1 prefill 图融合：** `#46230`（prefill graph 用 fused ratio-2 compressor）、`#46229`（每 prefill 步只翻译一次 SWA 位置）、`#46231`（把 fused MXFP8 attention 输出喂给 `wo_b`）、`#46163`（Mega-mHC 在 MegaGate 下写 MegaMoE 输入）、`#46180`（从 peer device shard 读 Engram 行）、`#46176` 见 3.3。
- **AMD 其他：** `#34543`（custom-allreduce 不再注册 oversized inputs）、`#46161`（rotary 给 HIP fallback 传 query-dtype `cos_sin_cache`）、`#46151`（修复量化低 ratio 激活下的 decode graph capture）、`#45758` 见 3.4。
- **PD 与可观测：** `#41140`（pd：合并 sender/receiver 实现）、`#46074`（rust-server 在 prefill 起自己的 transfer bootstrap）、`#45970`（导出 `prealloc_full/swa_token_usage` 指标，面向待定 KV 传输的 decode 请求）。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文）。性能数字（#45758 +30%/−20%/−40%、#42929 ~4.3–4.4×）为各 PR 自测口径；#44029/#46176/#41169 正文细节为本次独立抓取核验。

---

## 4. TensorRT-LLM：10-24 提交（14 条）——Rubin 快路径、量化静默修复与运行时重构

**主题**  TensorRT-LLM 在 2026-10-24 UTC 窗口合并 **14 条**提交（周六低量，rc1 后稳定期），主线为 **Rubin(SM107) 快路径与建模**（DSv4/DSpark、GLM-5.2 NVFP4 低延迟 decode target）、**量化静默修复**（#20823 NVFP4_AWQ pre-quant scale）、**LoRA 格式一致性**与**运行时接口显式化重构**。

### 4.1 `#20823` NVFP4_AWQ pre-quant scale 未应用（静默数值偏移）——见第 1 节头号事件

已在第 1 节详述：MegaMoE CuteDSL 加载 `fc31_act_scale` 但 `quantize_input` 不消费，FC1 跑在 unsmoothed 输入、输出与 CUTLASS/TRTLLM-Gen 偏离且无报错。

### 4.2 `#20517` Staircase：GLM-5.2 NVFP4 低延迟 decode target for Rubin（TP16, MTP-Eagle）

**主题**  `[None][feat] Staircase: GLM-5.2 NVFP4 low-latency decode target for Rubin (modeling_v2, TP16)`（`#20517`，取代 #20502）。

**内容详情（取自 PR 正文）**  为 **GLM-5.2-NVFP4 on Rubin (sm_107) 在 TP16** 新增 modeling_v2（Staircase）target：16 rank 上 tensor-parallel attention + experts，无 attention DP、无 expert parallelism，配 one-model MTP-Eagle（任意 draft 长度）或无投机；是**解耦低延迟部署的生成服务器**（一个 DEP4 context 服务器用内置实现、一个 TP16 生成服务器用本 target，`modeling_v2: require`）。`routing.py` 仅认 compute capability (10,7)、GLM-5.2 shape（78 层、hidden 6144、256 routed experts、q_lora_rank 2048）、上述 TP16 映射与 MTP-Eagle/无投机；不匹配的部署得 `NoTarget`（`auto` 回落内置、`require` 报错）。target 自带 decoder/checkpoint loader/embedded MTP，缓存布局与 context 服务器兼容；DecodeTarget 跑单请求生成步（verify rows ≤ 8），attention 含 fused-A GEMM、q_b 在 epilogue 用 FP8 RoPE、KV 半量直写缓存、sparse-MLA flash decode 自映射 selection 到 cache 行。

**应用场景**  GLM-5.2 NVFP4 在 Rubin(SM107) 的**解耦低延迟生成部署**（TP16 + MTP-Eagle）。

**研判**
- **事实：** 一个面向 Rubin 的、显式建模_v2 生成 target，路由门控严格（不匹配即 `NoTarget`），缓存布局与 context 服务器兼容，支持 MTP-Eagle 投机。
- **观点：** 与 vLLM/SGLang 的 PD 解耦主线同频——TRT-LLM 也在把「context 服务器 + 专用生成 target」做成一等公民（DEP4 + TP16 + MTP-Eagle）。对 Rubin 上的 GLM-5.2 低延迟服务是直接可用的新能力。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20517 （创建 2026-10-17，merge 在窗口内；正文已核验）

### 4.3 `#19961` DeepSeek V4 / DSpark optimizations on Rubin——描述与 diff 不一致（记描述级）

**主题**  `[None][feat] DeepSeek V4 / DSpark optimizations on Rubin`（`#19961`）。

**内容详情（取自 PR 正文 + 自动评审）**  PR 描述声称「完成 DSv4 在 Rubin(SM107) 上已合入 main 但未接线的快路径，并新增用于验证的 DSpark 精度覆盖」，给出 `q_b GEMM + RMSNorm + RoPE` 融合（`cute_dsl_dsv4_qb_gemm_fused_rmsnorm_rope_quant`，仅 decode、context/mixed 走未融合路径）等细节。**但 coderabbit 自动评审指出：实际 diff 仅重排 `cute_dsl_custom_ops.py` / `accuracy_core.py` / `test_fp8_block_scale_gemm.py` 的 import，不改变运行时行为、API、测试逻辑或测试列表**，QA 结论「coverage sufficient for formatting-only diff / test execution not established」。**故按本档案规范，本条记为「描述级/未核实实现已随本 PR 落地」**——描述中的快路径接线可能已在更早的 commit 合入，本 PR 自身只做 import 格式化。

**应用场景**  DeepSeek-V4 在 Rubin 的快路径（需另行核验实际接线落点）。

**研判**
- **事实：** diff 内容（import 重排）与 PR 描述（DSv4 Rubin 快路径完成）**不一致**；自动评审明确「no observable behavior change」。
- **观点：** 这是「PR 描述 ≠ 实际变更」的典型样本，再次印证累计结论的「发布说明/描述 ≠ 已进版」原则——**对 TRT-LLM，PR 描述中的能力声称必须落到具体 commit/diff 才能采信**。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/19961 （创建 2026-10-08，merge 在窗口内；正文已核验；diff 与描述不一致，记为描述级）

### 4.4 `#20270` / `#20774` 运行时重构与 LoRA 格式一致性

**主题**  `[TRTLLM-15763][refactor] Narrow the runner surface, slim engine state, and make the engine's executor interface explicit`（`#20270`）与 `[fix] Reload a request's LoRA adapter in the format it was admitted with`（`#20774`）。

**内容详情（取自 PR 正文）**
- `#20270`：`PyTorchModelEngine` 现组装 runners 并直接传运行时特性；去除重复的 engine state、收窄 runner API、使 executor 面对的方法显式化；`PyExecutor` 现持有每迭代投机解码状态并传给 `forward()`；嵌套 `warmup_scope()` 管理 warmup 副作用，`release_executor_resources()` 提供资源释放入口。调用方须用修订后的构造器/私有方法名/forward 参数/draft-length 处理；KV-cache manager lookup 改用 `ResourceManagerType.KV_CACHE_MANAGER`。coderabbit 注明「needs follow-up」回归验证（无测试 ID 增删证据）。
- `#20774`：PyTorch 后端某请求的 LoRA adapter 在 `BaseWorker._enqueue_request` 用请求的 `lora_ckpt_source` 加载；当 adapter 不在 PEFT cache、`PeftCacheManager.add_request_peft` 在每个 rank 重加载时却用了**服务器级** `LoraConfig.lora_ckpt_source` 而非请求的。修复把请求的 checkpoint 格式随 `py_lora_path` 一并带到 executor（`py_lora_ckpt_source`，在 `BaseWorker` 设置、广播、存于 `LlmRequest`），`add_request_peft` 现按请求格式重载。新增单测 `test_add_request_peft_reloads_with_request_ckpt_source` / `..._defaults_to_config_ckpt_source`，并用 `trtllm-serve`（TinyLlama-1.1B 单卡）前后对比 NeMo/HF 格式 LoRA 均按各自格式重载。

**应用场景**  TRT-LLM PyTorch 运行时的 executor/runner 接口与生命周期；多格式（NeMo/HF）LoRA 适配器重载一致性。

**研判**
- **事实：** `#20270` 是运行时架构重构（接口显式化、状态收窄），coderabbit 标记需回归验证；`#20774` 是 LoRA 适配器按请求格式重载的正确性修复，带单测与端到端验证。
- **观点：** `#20270` 这类「收窄接口/显式化 executor」重构是 PD/解耦与多推理路径（context/generate target）成熟化的底层铺垫；`#20774` 则避免了「同一请求的不同 LoRA 被错误格式加载」这类隐蔽问题。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20270 （创建 2026-10-14，merge 窗口内）；https://github.com/NVIDIA/TensorRT-LLM/pull/20774 （创建 2026-10-21，merge 窗口内；正文已核验）

### 4.5 TRT-LLM 其余值得记的提交

- **`#20808`** `[fix] Fork CuteDSL MoE input quantization only under multi-stream`——多流下才 fork CuteDSL MoE 输入量化。
- **`#20663`** `[fix] Schedule Gemma4 multimodal encoder items under the encoder budget`——Gemma4 多模态 encoder 项纳入 encoder 预算调度。
- **`#20774`** 见 4.4。
- **`#19080`** `[test] Restore LTX-2 worker CUDA graph regression`（两次提交，TRTLLM-15840）——LTX-2 worker CUDA graph 回归测试恢复。
- **`#20899` / `#20935` / `#20927`** `[infra]` B200 多 GPU stage 恢复、pre-merge 豁免个别失败用例——CI/infra 维护。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文）。`#20823`/`#20517`/`#20774` 正文细节本次独立抓取核验；`#19961` 因 diff 与描述不一致记为描述级；`#20270` 为重构、coderabbit 注需回归验证。

---

## 5. llama.cpp：10-24 提交（19 条）——MoE 量化正确性、异构后端与图级 rescale

**主题**  llama.cpp 在 2026-10-24 UTC 窗口合并 **19 条**提交（周六常态量），主线为 **MoE 量化正确性**（#31097 MiniMax-M3 scale 前向）、**Metal MoE rescale 清理**（#30924）、**Hexagon HMX flash attention KV-head team split**（#31055）、**HIP/Vulkan 后端稳健性**（#30915/#30656）与多后端 MoE 边界（#31081/#26631）。

### 5.1 `#31097` MiniMax-M3：量化 scale 前向到图（修复 MoE「token soup」）——见第 1 节头号事件

已在第 1 节详述：9 个 scale 张量未前向到 `build_moe_ffn()`，routed-expert 权重被当 `q` 而非 `q*block_scale*G`（G≈15123），输出「token soup」，无报错；修复仿既有 NVFP4 MoE 模型模式前向全部 scale。

### 5.2 `#30924` Metal：移除 `mul_mm_id` 的 amax rescaling（图级 rescale 已精确）

**主题**  `metal : remove mul_mm_id amax rescaling`（`#30924`，ggerganov）。

**内容详情（取自 PR 正文）**  移除 Metal `mul_mm_id` kernel 中对 `src1` 的 amax rescaling（#26223 为规避 MoE 激活超出 f16 范围导致的 inf/NaN 而加）。自 #30861 起 MoE down projection 已在计算图按 expert-token 做 rescale（`build_amax_src1_f16` + `ggml_pool_1d` max，输出撤销），到达 `MUL_MAT_ID` 的激活已能放进 f16，故 in-kernel rescale 冗余；且该路径还被 `GGML_PREC_F32` on `src1` 门控，而当前图中没有任何东西设置它，故从未对真实模型 dispatch。删除相关 kernel/管线/scratch/测试参数。`test-backend-ops -b MTL0` 在 Apple M5 Max 上 18297/18297 OK。

**应用场景**  Apple Metal 后端的 MoE `MUL_MAT_ID` 路径（MoE 激活 f16 溢出规避）。

**研判**
- **事实：** 删除的是「已被图级精确 rescale 取代、且从未对真实模型 dispatch」的 Metal-only workaround；图级 rescale（#30861）是精确（2 的幂）且覆盖所有后端。
- **观点：** 与上期各引擎「默认/历史 workaround 成为技术债」的主题一致——llama.cpp 在清理 MoE 量化的临时规避。对 Metal 用户是减负债、无精度影响。

**来源**  https://github.com/ggml-org/llama.cpp/pull/30924 （创建 2026-10-21，merge 在窗口内；正文已核验）

### 5.3 `#31055` Hexagon：KV-head team split for multicore HMX flash attention（+15.1% PP）

**主题**  `hexagon: KV-head team split for multicore HMX flash attention`（`#31055`）。

**内容详情（取自 PR 正文）**  当 `n_kv_heads` 不整除核心数时，head-split 不可用、FA 回落到 token partition，此时每个核心遍历每个 KV head。该 PR 把核心分成 `gcd(n_kv_heads, N)` 个 team：每 team 拥有部分 KV head、team 成员切分该 team 的 Q-block、Q-block 大小按每核 Q-block 数选取。每核现只流式处理本 team 的 KV head，**每核 KV 流量下降**。`GGML_HEXAGON_FA_SPLIT` 现为单 flag：`1`=head-split+team-split（默认）、`0`=关。llama-bench 4 核、`-sm row`、ubatch 1024、1024-token prompt、PP depth 0 / TG depth 1024 / -n 128 下（team-split 关 vs 开）：

| model | KV heads | PP before | PP after | PP Δ | TG before | TG after |
|---|---|---|---|---|---|---|
| gemma-4-E4B | 2 | 2832 | 3260 | **+15.1%** | 34.1 | 36.1 |
| Qwen3.5-0.8B | 2 | 7300 | 7646 | **+4.7%** | 102.9 | 105.3 |

`n_kv_heads % N == 0` 或 `n_kv_heads == 1` 的模型不形成多成员 team、不受影响。

**应用场景**  Qualcomm Hexagon（HMX）上的 flash attention——尤其 KV head 数不整除核心数的小 KV-head 模型。

**研判**
- **事实：** 通过「team-split」降低每核 KV 流量，gemma-4-E4B PP +15.1%、Qwen3.5-0.8B PP +4.7%，TG 小幅正向；有逐模型实测表。
- **观点：** 这是端侧/异构 NPU 上「flash attention 的 KV 分片策略」的精细优化——与 vLLM/SGLang 在 GPU 上抠 KV 传输同源，只是落在 Hexagon 的核间 KV 流量。对端侧 llama.cpp 部署是直接收益。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31055 （创建 2026-10-23T17:48:38Z，merge 在窗口内；正文已核验）

### 5.4 llama.cpp 其余值得记的提交

- **`#30915`** `HIP: add byte permute fallback when __builtin_amdgcn_perm is unavailable`——`amdgcnspirv` 拒绝 target-specific builtin 除非用 `__builtin_amdgcn_is_invocable` 守护，导致 66/145 个 HIP 文件用 SPIR-V 编译失败；用 helper 包一层，有 v_perm_b32 时保留、否则回退可移植 byte select。ROCm 头已采用同法。
- **`#30656`** `vulkan: ggml src1 precision support`——Vulkan 后端 MUL_MAT/MUL_MAT_ID 处理 `src1` 精度（缩进 f16 范围，仿 Metal #29029），取代 #30105/#30544。
- **`#31081`** `llama : deny -sm tensor for arches with moe amax rescaling`——对仍走 MoE amax rescale 的架构拒绝 `-sm tensor`（与 #30924 图级 rescale 清理呼应）。
- **`#26631`** `ggml : allow MUL_MAT_ID -1 index to skip the computation`——`MUL_MAT_ID` 的 -1 索引跳过计算。
- **`#30399`** `rpc : add bounds check in rdma_recv`——RPC 接收边界检查（稳健性）。
- **`#31059`** `jinja : fix select and selectattr argument handling`——模板引擎修复。
- **`#31063`** 由 `#31097` 修复（MiniMax-M3 scale 前向）。

**证据等级：** 官方一手（GitHub 提交 API + PR 正文）。`#31097`/`#30924`/`#31055` 正文细节本次独立抓取核验；#31055 性能表为作者 llama-bench 实测口径。

---

## 6. 当前会话已形成的累计内容结论

> 以下为「逐日热点档案」跨期累积的判断（非单日结论）。第 1–5 条继承自 2026-10-24 档案（其中第 1–3 条源自 2026-10-20，第 4–5 条为近期强化），第 6 条为本期的新增。

1. **调度器隐式阻塞的定量画像（继承自 10-20/10-24）：** 多引擎反复出现「host sync / 逐层加载 / 页面 H2D」落在调度器热路径上，导致 GPU 利用率与吞吐被隐性压低（SGLang 多期把 host sync 移出热路径、合计 +4.5–9.9%；vLLM ROCm 侧持续 micro-perf）。**判读：调度器的「隐性阻塞」已可用提交计数与 GPU 利用率变化定量画像，是评估一个引擎成熟度的高频信号。**
2. **发版窗口的双向性（继承自 10-20/10-24）：** 「发版当天合入 main」与「在发版内」是两个独立事实；且「PR 描述/发布说明含某能力」≠「该能力随该 PR/diff 真正落地」（本期 TRT-LLM `#19961` 描述声称 DSv4 Rubin 快路径完成、实际 diff 仅 import 格式化）。**判读：对 TRT-LLM 尤甚，PR 描述中的能力声称必须落到具体 commit/diff 才能采信。**
3. **默认值成为主要风险载体（继承自 10-20/10-24）：** 大量破坏性/静默行为来自默认值翻转（SGLang `--disaggregation-decode-host-receive-threshold` 语义翻转、TRT-LLM rc1「默认 KVCM V2 + Python transceiver」BREAKING）。**判读：升级验收应把「默认值变更」列为一级检查项，而非只看新增 flag。**
4. **「静默错误 / 进程崩溃」在异构后端高密度聚集（继承自 10-24，本期再次印证）：** 上期四家引擎均有「异构后端位/精度/同步陷阱 → 静默错或崩溃」修复（vLLM gfx942、TRT-LLM CuteDSL 量化 prefill 等）；本期虽无新的异构后端 miscompile，但量化链路静默错误（见第 6 条）接棒成为同类主题。**判读：ROCm/Blackwell/Rubin 等后端的正确性验收应保留数值抽检与 warmup/graph-capture 边界测试。**
5. **「前缀缓存 × MoE 路由 × 投机解码」三者交叉的正确性债务持续累积（继承自 10-24）：** vLLM `#62591`/`#54163`、SGLang `#45700`、llama.cpp `#30663` 等持续暴露「KV 分层/前缀缓存」与「MoE 专家状态 / 递归状态 / 投机 draft 状态」分开管理时的冲突。**判读：任何把这几者分开管理的引擎都需要一条显式的「保留/裁剪/对齐」协议，否则表现为硬失败（BlockObjectStoreError）或静默命中率损失。**
6. **「量化路径的静默错误」成为与异构后端并列的第二类高频静默错误（本期新增）：** 上期第 4 条聚焦异构后端的位/精度/同步陷阱；本期 TRT-LLM `#20823`（NVFP4_AWQ 预量化 scale 加载但未应用、输出与 CUTLASS/TRTLLM-Gen 偏离、无报错）与 llama.cpp `#31097`（MiniMax-M3 MoE 量化 scale 未前向到图、输出「token soup」、无报错）表明**量化（scale/反量化）链路的「漏接/漏应用」同样会产出无报错的错结果**，且与前后端无关。llama.cpp `#30924` 同期清理了 Metal `mul_mm_id` 的冗余 amax rescale（已被图级精确 rescale 取代），说明 MoE 量化链路正在被系统梳理。**判读：除异构后端数值抽检外，量化部署（NVFP4 / NVFP4_AWQ / MXFP4 / GGUF 量化 scale 前向）应把「与黄金参考逐层对拍」列为上线验收，因为这类错误不触发断言。**

---

## 7. 今日可执行清单

针对本期发现，给正在跑或准备升级这些引擎的团队：

1. **TRT-LLM NVFP4_AWQ / NVFP4_ARC 量化部署（高优先）：** 确认 `#20823` 已在所用版本内（修复把 `fc31_act_scale` 实际接入 `quantize_input`），并对 MegaMoE CuteDSL 路径做与 CUTLASS / TRTLLM-Gen 的**逐层数值对拍**，避免「无报错的输出偏移」。
2. **llama.cpp MiniMax-M3 / M3.1 Flash（NVFP4 / MXFP8 MoE）部署（高优先）：** 确认 `#31097` 已含量化 scale 前向；用真实 M3.1 Flash GGUF 跑端到端，核对 MoE 输出非「token soup」（`G ≈ 15123` 的 block_scale 必须进入 `w_s` 分支）。
3. **SGLang KV 分层 / PD 部署检查 `#44029` 0SM Allgather：** 单节点 + NCCL ≥ 2.28 + 无自定义内存池时默认开启；**多节点跨机场景该路径未验证**，应保留 Torch 分配回退作安全网。
4. **SGLang DSv4.1 开启 `SGLANG_OPT_DEEPGEMM_MEGA_GATE` 的用户（高优先）：** 确认已停在 decode / target-verify gate + 2048 行上限下（因 `#46242` 回退了 `#46232` 的 prefill MegaGate 路由——GB300 / DP4 / MegaMoE / 特定 shape 下第二个 shape 的 prefill capture 会 `CUDA error: unspecified launch failure`、服务起不来）。
5. **vLLM ROCm/AMD 部署（DSv4.1-Flash / V4-Pro）：** `#62713` MI325X 开 `--monokernel`（仅 TP4）、`#61482` MI355X 独立 mono decode 层（DSv4-Pro +28.7% tok/s/user @ c1）；**MI300X 仍不开该 flag**（显存余量仅 ~2.7GiB）。GLM-5.3 用户可评估 `#62102`（B300 TP4 下 E2E TPOT +1.2%~3.0%、aime25 精度不变）。
6. **TRT-LLM Rubin(SM107) GLM-5.2 NVFP4 低延迟服务：** `#20517` 提供 TP16 + MTP-Eagle 的 modeling_v2 生成 target（解耦部署的生成服务器），路由严格按 shape/算力门控，可据此搭 DEP4 context + TP16 generate。
7. **量化（NVFP4 / NVFP4_AWQ / MXFP4 / GGUF scale）上线验收新增「黄金参考对拍」：** 见第 6 节第 6 条；这类静默错误不触发断言，只能靠对拍或人工抽查发现。
8. **SGLang diffusion/DiT serving（如涉及）：** 本期 79 条中 diffusion 性能 PR 约占 1/3+（FA4 over cuDNN、VAE decode 切分、breakable CUDA graph 默认开、常驻显存策略），可关注其长序列自注意力与显存常驻对贵司视频/图像生成服务的可借鉴点。

---

## 8. 社区与视频资料

- **Hacker News（Algolia，2026-10-24 UTC 窗口）：** `vllm` 6 条命中**全部为「LLM」模糊匹配，无一条指向 vLLM 引擎本身**（如「Kevlar = MCP Firewall for LLM's」「Show HN: A client-side adaptive cognitive test」）；`llama.cpp` 1 条为 Show HN《Local inferencing in one JVM JAR, dependency free, faster than llama.cpp》（1 point，**自称快于 llama.cpp、未提供基准或复现，按规范标记为二手线索/未核验**）；`sglang`、`TensorRT-LLM`、`speculative decoding`、`KV cache` **均 0 命中**。
- **官方视频：** 本期（窗口内）四家引擎**均未核验到可确认发布日期与内容的官方视频**（vLLM 官方博客窗口内无新文；SGLang/TRT-LLM/llama.cpp 无新官方视频公告）。
- **未采集渠道：** Reddit、YouTube、X（Twitter）按既定范围未纳入；如后续需要，应单独说明并标注来源类型。

**证据等级说明：** 本期所有引擎事实均来自官方 GitHub 提交 API / releases / PR 正文（官方一手）；HN 单条 Show HN 为二手线索；社区转述如无官方源印证一律标记为「未核验」。TRT-LLM `#19961` 因实际 diff 与 PR 描述不一致，记为描述级/未核实实现落地。

---

> 归档说明：本档案为自动化每日追踪产出，覆盖 2026-10-24 UTC 窗口。上一期 2026-10-24 档案（覆盖 2026-10-23）已恢复每日追踪，本期（2026-10-25，覆盖 2026-10-24）与之连续、无空档；2026-10-21/22/23 为历史空档（自动化未执行），本目录不补写中间日期内容，各期仅按自身关注窗口采集。
