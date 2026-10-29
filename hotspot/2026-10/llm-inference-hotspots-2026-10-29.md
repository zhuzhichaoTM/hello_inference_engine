# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-29（北京时间）
- **关注窗口：** 2026-10-28 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-28 08:00 – 2026-10-29 08:00；2026-10-28 为**周三（工作日）**）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性、社区 KV cache 分发
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-28 UTC 窗口，分别返回 **77 / 90 / 45 / 27** 条，均 HTTP 200；四家均为工作日活跃量，vLLM 77 / SGLang 90 延续高位、TRT-LLM 45 / llama.cpp 27 略高于周二）、四家官方 releases API、**22 个 PR 正文**（经 GitHub issue API 抓取，返回 markdown 文本，含 vLLM `#63827`/`#62407`/`#63791`/`#64250`/`#61464`/`#62240`/`#63018`/`#61699`、SGLang `#46987`/`#46983`/`#47148`/`#46805`/`#34031`、TensorRT-LLM `#20694`/`#20755`/`#20893`/`#20597`、llama.cpp `#31201`/`#31300`/`#31188`/`#31288`/`#28141`）。**窗口特征：** 2026-10-28 为**周三（工作日）**，四家提交量维持工作日高位；**当日发生两次版本发布事件**（llama.cpp v0.8.0 正式版、TensorRT-LLM v1.4.0rc2 prerelease），故本期头号事件为「**双版本发布 + SGLang KV cache 架构重构主线**」（见第 1 节）。HN Algolia 公开 API 经 HTTPS 可访问（此前 HTTP 301 重定向，已改 https 复采成功）；窗口内出现**一条引擎相关社区文章**（「DeepSeek-v4.1-Flash: Pushing the Limits of KV Cache Compression」，5 分）及两条弱相关（Rust 版 LLM 上下文窗口 GC、ThreadShelf Show HN），其余引擎关键词命中多为空标题/评论级。Reddit、YouTube、X 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** 窗口内 **llama.cpp 发布 v0.8.0 正式版**（2026-10-28T12:47:51Z，stable）+ 滚动 nightly **b11968–b11972**；**TensorRT-LLM 发布 v1.4.0rc2 prerelease**（2026-10-28T15:51:34Z；Python wheel 为 **1.4.0rc2.post1**，post1 仅移除阻止 PyPI 发布的 Git 依赖 #21128 + 更新 Boost 镜像 #21133）。vLLM / SGLang **均无新版本**（vLLM 仍 **v0.32.0** 2026-10-18；SGLang 仍 **v0.5.22** 2026-10-19）。
- **本期方法论沿用：** 继续用 GitHub commit API 的 UTC 日期窗口 + PR 正文经 issue API 抓取（干净 markdown）；延续「版本核验/描述级核验」口径——发布说明/描述 ≠「修复已进版」，PR 描述 ≠ diff 落地，CI 状态须纳入证据等级。本期新增：TensorRT-LLM `#19961`（DeepSeek V4/DSpark on Rubin）在 rc2 发布说明中列为 perf 项，但依 2026-10-18 档案记为**描述级**（diff 仅 import 格式化、未显式落地快路径）；SGLang `#46983`（AMD 翻译窗口误写修复）CI 中 **AMD ROCm 10 与 Extra 均 :x:、Base 未运行（:no_entry_sign:）**，且正文自述「AMD hardware and ROCm kernels were not tested」，记为**描述级/未全绿**。

---

## 1. 版本状态与本期头号事件：双版本发布（llama.cpp v0.8.0 正式版 + TRT-LLM v1.4.0rc2）+ SGLang KV cache 架构重构

**主题** 本期是近期少见的「同日双版本发布」窗口：llama.cpp 推出 **v0.8.0 正式版**（含 LFM3 家族、实时音频/语音 API、ggml v0.28.0 后端大改），TensorRT-LLM 推出 **v1.4.0rc2 prerelease**（含 4 项 BREAKING + 弃用 n-gram/user 投机解码）；与此同时 SGLang 完成一组大体量的 **KV cache 架构重构**（#46973–#46991，分离字节存储与分配身份）。三者共同把「KV cache 正确性与可维护性」「量化/默认切换的破坏性」「跨后端版本门槛」推到本期最前。

### 1.1 llama.cpp v0.8.0 正式版（2026-10-28T12:47:51Z，stable）

**主题**  `ggml-org/llama.cpp` 发布 **v0.8.0** 正式版（nightly 基准 b11960），距 v0.7.0 累积 864+ 提交。

**内容详情（取自发布说明）**
- **新增模型家族：** LiquidAI **LFM3**（LFM3-1.7B 文本、LFM3-VL-2B 视觉、LFM3-Audio-2B / LFM3-Omni-2B 音频+语音聊天，单模型实现 ASR/TTS/voice chat，流式音频输入/输出 #31252/#31257）；LiquidAI **d1-2B / d1-omni-2B** 决策模型 #31275；**DeepSeek V4.1 Flash** #31086、**SenseNova U1** #28919、**Beam** (Reflection) #30670、**Kolibri-1** (Aleph-Alpha) #30673、**Flint** #30928、**Muse Spark Open** #30616、**Granite Switch Decision**（256-way 决策头）#30676/#30693、Nemotron-3(.5)-Embed #30877/#23343、H2O-Lightning-4B #30361、MiniCPM-OCR 1.0/Jev #30979/#30980、Soprano TTS #28607。
- **实时/语音 API：** 服务端新增 **`/v1/realtime`**（OpenAI realtime 协议，对话+转录）#31269/#30636/#30994 与 **`POST /tts`** 语音合成 #26603；`mtmd` 新增实验性 `mtmd_audio_stream` 流式音频输入 #30629；推理力度（reasoning effort）经 API 可选 #31271。
- **核心推理：** 稀疏注意力 **QSA/DSA 改造到融合 mask 算子**（`GGML_OP_TOP_K_MASK`、无 `kq_mask` 的 QSA 选择、k-pool 跨层共享视图）#30640/#31009/#30965/#31243/#30613；host 常驻 MoE 专家改用 **50/50 SLRU 缓存**（替代纯 LRU）#30637；`no_alloc` 在所有内存缓存中受尊重 #31087；MoE `-sm tensor` 在启用 amax rescaling 的架构上被拒 #31081；MiniMax-M3 量化 scale 前向到图（修 #31063）#31097。
- **ggml 升至 v0.28.0：** 新增 `GGML_OP_TOP_K_MASK` 融合稀疏注意力 mask；`MUL_MAT_ID`/`ADD_ID` 支持 **`-1` 专家索引**（跳过该 slot、输出行置零）；后端：CUDA **chunked graph launches** + 融合 MoE top-k/gate-up+GLU、SYCL graph record/replay、OpenCL Q2_K/Q3_K + lightning indexer、Hexagon HTP 重构、WebGPU Q2_0、**KleidiAI SVE GEMV/GEMM 内核**。
- **量化/工具：** `llama-quant` 修 4D 张量量化（#30820，跨引擎同类静默错误的 llama.cpp 侧修复）；GGUF 工具拒收 `split`/`export-lora`/`merge` 中的截断张量数据 #30497/#31044。
- **CLI 破坏性变更：** `--model` 或 `-hf` **现为必填** #31242（升级注意）；`/v1/messages` 转发 llama.cpp 采样参数 #30856；server 支持 `logprobs` + 投机解码 #27196。

**应用场景**  llama.cpp 全平台（CPU/CUDA/Metal/ROCm/Vulkan/SYCL/Hexagon/OpenCL/WebGPU）的端侧/服务端推理，尤其实时语音、多模态、MoE 与稀疏注意力模型部署。

**研判**
- **事实：** 发布说明结构完整（Overview/Highlights/API changes/New models/Core/Multi-modality/Server/UI/ggml/Changelog），含逐 PR 链接与 864-commit changelog；属官方一手。
- **观点：** v0.8.0 是 llama.cpp 进入「LFM3 多模态 + 实时语音 + 稀疏注意力 + ggml v0.28 后端大改」的里程碑，但 **`--model`/`-hf 必填**与 ggml v0.28 的内核语义变更（如 `-1` 专家索引、TOP_K_MASK）意味着旧脚本/封装需回归；列为本期头号事件之一（稳定版发布）。

**来源**  https://github.com/ggml-org/llama.cpp/releases/tag/v0.8.0 （published 2026-10-28T12:47:51Z；正文已核验，含逐 PR 链接）

### 1.2 TensorRT-LLM v1.4.0rc2 prerelease（2026-10-28T15:51:34Z；wheel 1.4.0rc2.post1）

**主题**  `NVIDIA/TensorRT-LLM` 发布 **v1.4.0rc2**（prerelease）；Python wheel 标为 **1.4.0rc2.post1**（仅修 #21128 Git 依赖、#21133 Boost 镜像，便于 PyPI 发布；源码仍 1.4.0rc2）。

**内容详情（取自发布说明，重点 BREAKING 与版本风险）**
- **4 项 BREAKING：** ① `#20388` 将 `LlmRequest` 字段从 C++ 迁移到 Python；② `#20129` 从 transceiver V2 **移除 KV cache manager V1 支持**（与 2026-10-24 档案记的 rc1「默认 KVCM V2」一脉相承，V1 路径彻底清理）；③ `#20043` 移除废弃的 executor 后端兼容路径；④ `#20108` Mooncake Store **启用零拷贝模式**（BREAKING）。
- **弃用：** `#20438` 弃用 **n-gram 与用户提供的投机解码模式**（与 10-25/10-26 档案「单模型/通用投机解码」主线一致，旧式 spec dec 退出）。
- **KV cache V2 主线：** `#20086`/`#19710` KVCM V2 **原生跨轮生成块复用（retention）**；`#19877`/`#20180` 在 SWA **前缀 fork 点提交滑动窗口前缀**并计入 connector 提供的复用块；`#20452` First-N/Last-N **混合精度 KV**（native SWA + NVFP4 历史 + 高精度首尾）；`#20544` **FP4 MLA 在 disagg 部署**；`#20048` V2 用 string cache salts。
- **模型/硬件：** `#20517` GLM-5.2 NVFP4 低延迟 decode target（Rubin，modeling_v2，TP16）；`#20167` MiniMax-M3 在 Rubin(SM107) NVFP4 KV 豁免 + 128-token paged FMHA；`#20276` Qwen3.8-Flash-Next modeling_v2（CUTEDSL、MTP 1..7、video、FP8/NVFP4 KV、GB300 快路径）；`#19961` DeepSeek V4 / DSpark on Rubin（**描述级**：依 10-18 档案，diff 仅 import 格式化，快路径未显式落地）；`#20576` Kimi K3 fabric gather + TCPStore rendezvous + KDA replay buffers。
- **量化正确性（跨引擎静默错误主线）：** `#20823` **应用 NVFP4_AWQ 预量化 scale 到 MegaMoE CuteDSL**（即 2026-10-25 档案头号事件「TRT-LLM #20823 scale 未应用→输出偏离且无报错」的修复，本期进入 rc2 的 Fix 列表——记为静默正确性主线在 rc 的闭环）；`#20547` 每个专家的 input_scale 传入 glm5_next NVFP4 MoE loader；`#20443` MXFP8 融合权重 scale 按行对齐。
- **已知问题：** Tensor-parallel serving 复用 MPI worker pool 跨线程时，模型初始化可能 hang。

**应用场景**  NVIDIA GPU（Hopper/Blackwell/Rubin/SM103）上的 TRT-LLM  serving，尤其 DeepSeek-V4/V4.1、GLM-5.x、MiniMax-M3、Kimi-K3、Qwen3.8-Flash-Next 的量化与 PD 部署。

**研判**
- **事实：** 发布说明含 Highlights/API/Feature/Fix/Documentation/Test&Infra 全节与逐 PR 链接；BREAKING 与弃用项明确标注；`#20823` 出现在 Fix 列表（与 10-25 档案呼应）。属官方一手。
- **观点：** rc2 把「KVCM V1 移除 + 4 项 BREAKING + n-gram/user spec dec 弃用」成体系落地，**升级门槛显著**——沿用 V1/旧 executor/n-gram spec dec 的用户必须改造成本。建议把「rc2 发布说明 diff + 破坏性变更回滚预案」纳入发布门禁（强化累计结论第 2、3 条）。`#19961` 仍维持描述级，不夸大其已落地。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc2 （published 2026-10-28T15:51:34Z；wheel 1.4.0rc2.post1 commit `43f8f70…`；正文已核验）

### 1.3 SGLang KV cache 架构重构集群 #46973–#46991（分离字节存储与分配身份）

**主题**  SGLang 在 2026-10-28 合并一组大体积 **KV cache 重构 stack**（#46973 → #46991，约 19 个 PR），统一 KV 的「字节存储 / 分配身份 / 地址翻译 / 生命周期」职责。

**内容详情（取自 stack 代表 PR `#46987` 正文）**  `#46987`「[Refactor] kv-cache: Separate byte storage from allocation identities」：引入共享字节 arena 用于 layout-aware page 移动与清零；把**分配 band 与虚拟→物理 ID 空间**从 layout 视图分离；KV 存储构建与运行时消费者改用这些边界。同栈还含 `#46990` typed KV views + 共享层路由、`#46989`/`#46988` 共享存储接口、`#46984` 用 routed cache 替换 unified pool 变体、`#46983` AMD 翻译窗口避免 fused writes（见第 3 节）、`#46982` 从 storage plan 构建 cache transfers、`#46981` hybrid recurrent host pools 一致性等。

**验证（#46987 正文）：** 历史验证 `a2070bc811`/`17935efc49`（8× H200，PyTorch 2.14.1）：四套分配链各 **153 runs / 43,368 operations：0 divergences**；准入 **4,608 个不变决策**；unified HiCache 下 GSM8K-200：gpt-oss-120b **97.0%**、Qwen3.8-27B **96.5%**（在记录基线 spread 内）。但正文注明「**AMD hardware and ROCm kernels were not tested**」，且 Base/Extra/AMD ROCm 10 三组 CI 状态分别为 :no_entry_sign:/:x:/:x:。

**应用场景**  SGLang 的 unified/分层的 KV cache（HiCache、KVShard、PD 分离、recurrent 状态、跨后端翻译窗口）的可维护性与正确性底座。

**研判**
- **事实：** 明确的架构动机（KV 分配与 buffer layout 共享 page 移动/ID 翻译所有权）+ 重建验证 0 divergences + GSM8K 基线持平；属官方一手（功能/重构 PR）。
- **观点：** 这是 SGLang 把「KV cache 正确性」从补丁式修复推向**架构级职责分离**的信号，与 vLLM 的 KVCacheConfigBuilder 可插拔化（10-17）、TRT-LLM 的 KVCM V2 同构。但 stack 中 **`#46983` 等 AMD 路径仅 CPU 回归、ROCm 未实测**，升级到含此 stack 的 main 时 AMD ROCm 部署方需自测（记为描述级/未全绿）。

**来源**  https://github.com/sgl-project/sglang/pull/46987 （merge 2026-10-28T 窗口内；正文已核验，stack 含 #46973–#46991）

**证据等级（第 1 节）：** 官方一手（提交 API + 发布说明 + PR 正文）。llama.cpp v0.8.0 / TRT-LLM v1.4.0rc2 为发布说明一手；SGLang `#46987` 重构为功能 PR 一手（含 0-divergence 验证，但 AMD hw 未测）；`#19961` 维持描述级。

---

## 2. vLLM：2026-10-28 提交（77 条）——HiSparse/KV-Offload 静默正确性簇 + DSA DeepSelect + NVFP4 反量化

**主题**  vLLM 在 2026-10-28 UTC 窗口合并 **77 条**提交，主线为 **「静默正确性」修复簇**（HiSparse 异步 KV 覆盖、量化参数丢失、FlashInfer 稀疏 MLA 65535 静默跳过、KV-Offload P2P 未确认 WRITE）、**DSA/稀疏注意力性能**（DeepSelect BF16、NVFP4 反量化 gather）、以及 **MRV2/DBO/投机解码/DCP** 的协同。

### 2.1 `#63827` HiSparse 解码器异步 KV 加载静默覆盖（输出坍缩成重复 token）

**主题**  `[Bugfix][Core] Defer async KV loads on HiSparse decoders until in-flight steps finish`（`#63827`，fixes #63811）。

**内容详情（取自 PR 正文）**  HiSparse P/D 解码器上，NIXL 异步 load 可在「上一步仍在 GPU 上读该页」时把新请求的 KV 写入同一页，使原请求后续 token 拿到错误 KV，**输出从头到尾不可恢复**（GLM-5.3 上表现为中段坍缩成重复 token `locklocklock…`、MTP 接受率掉到 0）。四个条件同时成立才触发：① HiSparse 开启（早归还能被复用的 resident 页）；② 异步调度（`max_concurrent_batches>1`）；③ KV 经异步 load（NIXL consumer）到达；④ 下一步池把归还页分给一个经该 load 到达 KV 的新请求。修复：调度器把 HiSparse 解码器上的每个异步 load **推迟一个 step**（等 in-flight step 处理完再注册），无 device sync、无复用页检查。

**验证（GLM-5.3 H100 P/D，TP8×DP2，MTP3，MRV2，NIXL+HiSparse，20K-block 池）：**
| 空闲队列顺序 | 引擎 | 答案数 | 读中被 NIXL load 覆盖的页 | 坍缩 |
|---|---|---|---|---|
| default | base | 133 | 0 | 0 |
| released pages first | base | 162 | 90 | 81 |
| released pages first | this PR | 173 | 106（load 被 hold） | **0** |

后轮 TTFT：p50 1.98s(base) → 2.07s(PR)。CI 不覆盖 GPU 路径（需 NIXL P/D on GPUs）。

**应用场景**  vLLM HiSparse + 异步 KV 加载 + P/D 分离的 serving（如 GLM-5.3 长多轮）。

**研判**
- **事实：** 完整根因 + 复现步骤 + 量化表（坍缩 81→0）+ TTFT 代价；属一手细节（silent-error）。
- **观点：** 这是「HiSparse 把页提前归还 + 异步 KV 加载无栅栏」耦合出的**静默正确性**案例，与累计结论第 4/6/7 条同源；建议 HiSparse+异步 KV 部署方把「长多轮 GLM 级回归」列为默认（强化第 4、6 条）。

**来源**  https://github.com/vllm-project/vllm/pull/63827 （merge 2026-10-28T 窗口内；正文已核验，含复现与验证表）

### 2.2 `#62407` 量化：process_weights_after_loading 后保留被删/改名的参数（静默丢失）

**主题**  `[Quantization] Keep deleted and renamed params registered after process_weights_after_loading`（`#62407`，PWAL 保参三步之三）。

**内容详情（取自 PR 正文）**  PWAL 删除/重命名参数会导致 loader 可见参数「消失」：本 PR 让重命名参数**双名别名**（如 compressed-tensors `*_weight_packed`↔`weight`、INC MXFP4、`kv_cache.py` 的 KV scale 占位）、不再被 kernel 读取的已加载参数**标记 retired 保留数据**、大体积释放源**空存储保留注册**。覆盖：NVFP4/MXFP4/W4A8 MoE、INC/Quark/Humming NVFP4、block-FP8、int8 scaled-mm 等。整树枚举 PWAL 可达站点 1786 行，本 PR 修 73 个 delete/rename 站点 + 36 个联动。

**验证（GB200，GSM8K 5-shot greedy，loader 可见参数注册率）：**
| 配置 | GSM8K main/stack | 注册率 main→stack |
|---|---|---|
| ModelOpt NVFP4 (Qwen3-30B-A3B) | 0.8878/0.8886 | 69%→100% |
| compressed-tensors NVFP4 (Humming) | 0.8931/0.8946 | 36%→100% |
| compressed-tensors MXFP4 MoE | 0.8582/0.8582（1314/1319 文本一致） | 46%→100% |

Post-PWAL 张量字节级一致；保留数据每模型 ≤0.15 MiB（<0.001%）。

**研判**
- **事实：** 参数注册率 36–69%→100%、GSM8K 精度不变、张量字节一致；属一手。
- **观点：** 这是量化链路上「参数在 PWAL 后被悄悄注销 → 重载/直载失败（`has no parameter 'w2_input_scale'`）」的静默错误修复，与累计结论第 6 条（量化路径静默错误）同源；vLLM 量化重载路径用户应关注（强化第 6 条）。

**来源**  https://github.com/vllm-project/vllm/pull/62407 （merge 2026-10-28T 窗口内；正文已核验，含 GB200 验证）

### 2.3 `#63791` FlashInfer 稀疏 MLA：>65535 token 静默跳过注意力（NaN/未初始化）

**主题**  `[Bugfix][Attention] FlashInfer sparse MLA: launch at most 65535 tokens per trtllm-gen call`（`#63791`）。

**内容详情（取自 PR 正文）**  FlashInfer 稀疏 MLA kernel 对**超过 65535 token 的 forward（如 `--max-num-batched-tokens 65536` 的完整 prefill）静默跳过注意力**，返回未初始化内存（根因在 FlashInfer 侧 `flashinfer-ai/flashinfer#3125`：`trtllm-gen` 把每个 query token 当独立 request 映射到 `gridDim.z`，CUDA 上限 65535，更大 launch 使 `cuLaunchKernelEx` 返回 `CUDA_ERROR_INVALID_VALUE` 且 FlashInfer 仅打印到 stderr）。影响 **DeepSeek-V3.2 / GLM-5 / GLM-5.3 的 DSA 层**与 `FLASHATTN_MLA_SPARSE`(FA4) 的 prefill。修复：按 ≤65535 token 分块写入同一预分配输出（无额外显存，DCP softmax-stats 按单 launch 大小）。

**验证（SM107）：**
| 检查 | base | this PR |
|---|---|---|
| `test_flashinfer_sparse_mla_attends_tokens_past_grid_limit`（69635 token，SM10x） | 2/2 fail（token 65535 后 NaN） | 2/2 pass |
| 40-shot GSM8K，GLM-5.3-Flash NVFP4，DEP2，`--max-num-batched-tokens 65536` | 比基线低 ~3 点、满屏 `CUDA_ERROR_INVALID_VALUE` | 在基线内 |
| RULER-64K（13 task×100），`--max-model-len 131072` | 整体低 ~35 点、NIAH/变长追踪每任务低 11–82 点、~7.7K 行 launch 错误 | NIAH 8 task + 变长追踪近 100%、无错误 |

**研判**
- **事实：** 根因（CUDA `gridDim.z` 65535 上限）+ 分块修复 + RULER-64K 从低 ~35 点到近满分；属一手（silent-error，长上下文致命）。
- **观点：** 这是「kernel launch 参数静默失效、仅 stderr 报错、长上下文精度塌方」的典型案例，与第 4/6 条同源；所有用 FlashInfer 稀疏 MLA + 长 `--max-num-batched-tokens` 的部署须升级（强化第 4、6 条）。

**来源**  https://github.com/vllm-project/vllm/pull/63791 （merge 2026-10-28T 窗口内；正文已核验，含 RULER-64K 表）

### 2.4 其余 vLLM 要点（标题/发布说明级，部分正文未抓取）

- **`#64250`** `[Bugfix][KV Offload][P2P] Retain unconfirmed source WRITEs across session teardown`（#63602 gap3）：P2P 会话拆除时 source 不再立即释放可能仍在跑的 NIXL WRITE（RDMA WRITE 不被 `cancel` 停止），保留 handle/引脚/对端旧注册直到 transport 报告停止，**上限 120s**（超时强制释放并错误日志）。验证：issue CPU 复现 `main FAILED / this PR PASSED`；`tests/v1/kv_offload/tiering/p2p` 338 passed（CPU-only fake transport，未上 GPU）。**静默正确性**（KV-Offload 簇延续）。
- **`#61464`** `[MRV2][DBO] Support DBO with speculative decoding and DCP`：MRV2 支持 DBO + 投机解码 + DCP 协同（正文 1102 字符，标题级细节）。
- **`#62240`** `[Perf][DSA] Rank the prefill indexer logits with DeepSelect in BF16`：用 DeepSelect 在 BF16 下对 prefill indexer logits 排序（5479 字符正文）。
- **`#63018`** `[Perf][DSv4.1] Tile the NVFP4 compressed-KV dequant-gather`：DSv4.1 的 NVFP4 压缩 KV 反量化-gather 分块（2725 字符正文）。
- **`#61699`** `[Bugfix][Spec Decode] Honor standalone draft load config`：尊重独立 draft 加载配置（投机解码）。
- **`#63871`** `[Bugfix][Muse] Finalize dummy-loaded MegaMoE weights`：终结 dummy-loaded 的 MegaMoE 权重（避免未初始化）。
- **`#61294`** `[Bugfix] Merge supported KV cache layouts across pipeline stages instead of asserting they match`：跨 pipeline stage 合并支持的 KV cache 布局（不再断言一致）。
- **`#64230`** `[KV Connector][NIXL] Select push mode for NixlConnector via transfer_mode`：NIXL connector 经 `transfer_mode` 选 push 模式。
- **`#58305`** `[Perf][SM120][MLA] Use compact FP8 KV rows for NoPE models`：SM120 上 NoPE 模型用紧凑 FP8 KV 行。
- **`#64186`** `[Feature][Core] Add opt-in multi-worker CUDA checkpoint sleep`：可选多 worker CUDA checkpoint sleep。
- **`#63637`** `[ROCm][MiniMax-M3.1] Fuse MXFP8 activation quant into the Gemma input_layernorm for qkv_proj`：ROCm 上把 MXFP8 激活量化融合进 Gemma `input_layernorm`（省 qkv_proj 前量化拷贝）。
- 文档：DeepSeek-V4-Pro B200 AgentX 预设 #64316、MiniMax-M3.1-Flash GB200/GB300 AgentX 预设 #64283/#64284；CI 维护多条（ROCm lockfile、pre-commit、GitHub Actions 版本）。

**证据等级（第 2 节）：** `#63827`/`#62407`/`#63791`/`#64250` 为带验证表的 silent-error/正确性一手；`#61464`/`#62240`/`#63018`/`#61699` 为正文一手（标题+正文）；`#63871`/`#61294`/`#64230`/`#58305`/`#64186`/`#63637` 为提交标题级（依 commit message，未抓正文）。

---

## 3. SGLang：2026-10-28 提交（90 条）——KV cache 重构 stack + MegaGate prefill + NPU/AMD 性能

**主题**  SGLang 在 2026-10-28 UTC 窗口合并 **90 条**提交，主线为 **KV cache 架构重构 stack（#46973–#46991）**、**DSv4.1 性能**（MegaGate routing 进 prefill、Engram all-to-all、NPU 去 host-sync）、以及 **AMD ROCm 正确性**（翻译窗口 fused writes 修复）。

### 3.1 `#46987` 等 KV cache 重构（见第 1.3 节）

已详述：分离字节存储与分配身份，历史验证 0 divergences、GSM8K-200 基线持平，但 AMD hw 未测。

### 3.2 `#46983` AMD ROCm：避免对翻译窗口存储做 fused writes（静默错误修复，未全绿）

**主题**  `[Fix] [AMD] attention: Avoid fused writes to translated window storage`（`#46983`，属 #46973–#46991 stack）。

**内容详情（取自 PR 正文）**  ROCm fused KV store 使用模型提供的 location ID；对需要物理翻译的 unified sliding-window 视图，这些 ID 会寻址到错误存储，除非 pool 执行写。修复：当窗口视图需要翻译物理写位置时**禁用 fused store**，保留 full-only / static-window 存储的资格（含既有 DCP 排除）。

**验证（正文）：** 扩展回归套件「旧实现 3 失败、修复后 7 case 全过」；后续 CPU 探针也过 translated-window/static-window/DCP-mask。但正文明确：**「AMD hardware and ROCm kernels were not tested」**；CI：Base :no_entry_sign:、Extra :x:、AMD ROCm 10 :x:。

**研判**
- **事实：** 明确的「翻译窗口 + fused store 寻址错误」根因 + CPU 回归 3→7；但 AMD 硬件/ROCm 内核未实测、三组 CI 未全绿。
- **观点：** 与累计结论第 4 条（异构后端静默错误）同源——AMD ROCm 上 unified sliding-window 的 fused KV store 可能静默写错位置；**修复仅 CPU 回归，AMD 部署方须自测**（记为描述级/未全绿，强化第 4 条）。

**来源**  https://github.com/sgl-project/sglang/pull/46983 （merge 2026-10-28T 窗口内；正文已核验，注明 AMD hw 未测、CI 未全绿）

### 3.3 `#47148` DSv4.1：MegaGate routing 进 prefill + `#46805` NPU 去 host-sync + `#46743` LoRA csgmv 网格

- **`#47148`** `[Perf] dsv41: Run MegaGate routing in prefill`：DSv4.1 在 prefill 阶段跑 MegaGate routing（1891 字符正文，性能向）。
- **`#46805`** `[Perf] [NPU] dsv4: drop host-syncing ops from the decode scheduler chain (aclnnNonzeroV2 + D2H)`（8236 字符正文）：NPU 上从 decode 调度链移除 host-sync 操作（aclnnNonzeroV2 + D2H），降延迟。
- **`#46743`** `[Perf] LoRA: bound csgmv CUDA-graph segment grids by token count`：LoRA 的 csgmv CUDA-graph segment 网格按 token 数封顶（避免大 token 数下越界/低效）。
- **`#47033`** `[Perf] dsv41: Read Engram rows in place by default for GPU tables over NVLink`：DSv4.1 在 NVLink 的 GPU 表上默认就地读 Engram 行。
- **`#46955`** `[Perf] dsv41: Use the Engram all-to-all lookup for decode and CUDA graphs`：DSv4.1 用 Engram all-to-all lookup 做 decode 与 CUDA graphs。
- **`#47146`** `[Fix] dsv4: Capture the replay tail thread-locally while serving`：DSv4 serving 时线程本地捕获 replay tail。
- **`#47145`** `[Fix] docker: update NVIDIA megakernel pin to tested commit`：docker 把 NVIDIA megakernel pin 更新到已测 commit。
- **`#34031`** `[Refactor] spec: Make ForwardBatch the single channel for the ragged-verify layout`：投机解码把 ForwardBatch 作为 ragged-verify 布局的单一通道（1307 字符正文，重构）。
- **`#47078`** `[Fix] ci: Repair main after the #39728/#46990 merge race and RUSTSEC-2026-0414`：修复 #39728/#46990 merge race 与 RUSTSEC-2026-0414 后的 main。
- **`#46789`** `[Test] KVShard: Extend HiCache E2E with GLM accuracy coverage`：KVShard 用 GLM 精度覆盖扩展 HiCache E2E（与 10-28 档案 #42973 GLM DSA/MTP 接入 KVShard 同主线）。

**证据等级（第 3 节）：** `#46983` 为描述级/未全绿（AMD hw 未测）；`#47148`/`#46805`/`#46743`/`#34031` 为正文一手（标题+正文）；`#47033`/`#46955`/`#47146`/`#47145`/`#47078`/`#46789` 为提交标题级。

---

## 4. TensorRT-LLM：2026-10-28 提交（45 条）——rc2 稳定化 + GLM-5.3 Blackwell 性能 + Kimi K3 helix

**主题**  TRT-LLM 在 2026-10-28 UTC 窗口合并 **45 条**提交（含 **v1.4.0rc2 于 15:51:34Z 发布**），主线为 **rc2 稳定化/修复**、**GLM-5.3-Flash 在 Blackwell 的性能**、**Kimi K3 helix_a2a**、以及 **NVFP4 MiniMax-M3 权重加载修复**。

### 4.1 `#20755` GLM-5.3-Flash 在 Blackwell 提速（NVFP4 +19.8%、FP8 +2.3% C1）

**主题**  `[None][perf] Speed up GLM-5.3-Flash serving on Blackwell`（`#20755`）。

**内容详情（取自 PR 正文，4×B200 逐 commit 测量 + 4×B300 端到端）**
- **TRTLLM-Gen MoE 后端优先（SM100 族）：** `moe_config.backend: AUTO` 原把 NVFP4 GLM-5.3-Flash 解析到 CUTLASS，现对 `Glm5Next*` 解析到 TRTLLM；NVFP4 EP4：decode 7.52→6.46 ms/step、C1 355→412 tok/s/user。
- **context indexer 池用 FP8 打分：** SM100 族上把 packed context 行用 DeepGEMM `fp8_mqa_logits` 打分（替代 BF16 Triton），indexer 打分 0.93→0.21 ms/sparse layer（>10K-token prompt）；LongBench-v2/ GSM8K 不变。
- **稀疏 decode 省 kernel + 共享生成行长度跨稀疏层 + o_proj 布局写吸收 V 输出** 等（bit-identical）；**bound context pool-scoring 内存** 修长缓存前缀潜在 OOM。

**端到端（4×B300，>10K-token prompt，MTP3，FP8 KV；C1 tok/s/user / C8 tok/s）：**
| Checkpoint | Config | C1 main→PR | C8 main→PR |
|---|---|---|---|
| NVFP4 | TP4 EP4 | 364 → **436 (+19.8%)** | 137.4 → **153.8 (+11.9%)** |
| FP8 | TP4 EP1 | 391 → **400 (+2.3%)** | 121.3 → **124.9 (+3.0%)** |

TTFT：NVFP4 C1 260→230ms(-30)、FP8 C1 292→278ms(-14)。GSM8K：PR 93.21(NVFP4)/92.42(FP8) vs main 93.52/92.99（在 ~1 点 run-to-run spread 内）。FP8 context indexer vs BF16：LongBench-v2 56.11 vs 57.78（sign test p=0.68）。

**研判**
- **事实：** 逐 commit 测量 + 4×B300 端到端表（NVFP4 +19.8%/+11.9%、FP8 +2.3%/+3.0%）+ GSM8K 不变；属一手（性能 PR）。
- **观点：** GLM-5.3-Flash 在 Blackwell 的系统性优化（MoE 后端选择 + FP8 indexer + 内存 bounded），NVFP4 收益显著；同时文档建议长 prompt 低并发用 EP1（同节点 B300 FP8 TP4 EP1 vs EP4 C1 +6%）。建议 Blackwell GLM-5.3-Flash 部署方采用 TRTLLM-Gen 后端 + FP8 indexer 默认。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20755 （merge 2026-10-28T 窗口内；正文已核验，含 4×B300 表）

### 4.2 `#20694` NVFP4 MiniMax-M3 权重加载修复 + `#20597` Kimi K3 helix_a2a + `#20938` serving runtime KVCMv2

- **`#20694`** `[None][fix] Load compressed-tensors NVFP4 weights in MiniMax-M3`（10788 字符正文）：把 llm-compressor compressed-tensors NVFP4 的命名/scale 约定（`weight_packed`→`weight`、`weight_global_scale`→`weight_scale_2` 取倒数、`input_global_scale`→`input_scale` 取倒数）抽成共享 helper `convert_compressed_tensors_nvfp4_weights()`，MiniMax-M3 与 Laguna 共用；**全局 scale 为 0（未校准专家）保持 0**（取倒数会变 Inf 并赢过 MoE 逐层 max）；负数/非有限 scale 置 0 并汇总告警（坏 checkpoint 信号）。验证：3 个 helper 单测本地通过；更广的 MiniMax-M3/Laguna pytest、GPU、端到端 serving 未本地跑（CUDA build 缺失），**QA 判「needs follow-up」**。属量化加载正确性（与第 6 条同源）。
- **`#20597`** `[None][feat] Kimi K3: helix_a2a token-sliced MoE under helix + attention DP`（13386 字符正文）：Kimi K3 在 helix 下的 token-sliced MoE + attention DP（helix_a2a）。
- **`#20893`** `[None][feat] Consolidate Kimi K3 native modeling work into main (#19990, #19991)`（16324 字符正文）：把 Kimi K3 native modeling 工作收敛进 main。
- **`#20938`**（提交标题级，未抓正文）：`[None][perf] Serving runtime: disagg/KV-transfer/ADP fixes, KV cache manager V2 prefix wait, ConversationRouter prefix placement, KV-balanced ADP admission`——rc2 serving runtime 的 KVCMv2 prefix wait 等修复（标题级）。
- **`#21196`** `[https://nvbugs/6986044][fix] Re-enable GLM modeling_v2 routing tests`：重新启用 GLM modeling_v2 routing 测试。
- **`#20912`** `[https://nvbugs/6973693][fix] Disable KV block reuse in test_llmapi_backend`：在 test_llmapi_backend 禁用 KV block reuse。
- **`#19961`**（描述级，依 10-18 档案）：DeepSeek V4 / DSpark on Rubin 优化，diff 仅 import 格式化、快路径未显式落地，不夸大已进版。

**证据等级（第 4 节）：** `#20755` 为带 4×B300 表的性能一手；`#20694` 为量化加载修复一手（QA 判 needs follow-up）；`#20597`/`#20893` 为正文一手（功能 PR）；`#20938`/`#21196`/`#20912` 为提交标题级；`#19961` 维持描述级。

---

## 5. llama.cpp：2026-10-28 提交（27 条）——v0.8.0 正式版 + CUDA/量化/后端性能

**主题**  llama.cpp 在 2026-10-28 UTC 窗口合并 **27 条**提交，并**发布 v0.8.0 正式版**（见第 1.1 节）；窗口内另见 nightly **b11968–b11972**（22:50Z 前的 b11968–b11970 等）。主线为 **量化加速（IQ4 2×）**、**CUDA MoE 融合（gate/up+GLU、top-k 384 专家）**、**DSV4 batch 拆分移除**、**Metal TOP_K_MASK**、**SVE mxfp4 gemv**。

### 5.1 `#31201` IQ4 量化创建 2× 加速（bit-identical）

**主题**  `Speed up IQ4 quant creation with pre-computed halfway points`（`#31201`）。

**内容详情（取自 PR 正文）**  对 `IQ4_NL`/`IQ4_XS` 的量化创建，预计算相邻 16 值间的 15 个 halfway point，用「多少个 halfway point 高于该数」替代二分搜索选值，结果**位级一致（bit-identical）**，约 **2× 加速**。

**验证（llama-quantize --pure --imatrix，sha256 一致）：**
| 量化 | upstream | PR |
|---|---|---|
| 27B IQ4_NL | 79.5 s | **42.9 s** |
| 27B IQ4_XS | 76.4 s | **38.3 s** |
| 35B IQ4_NL | 99.5 s | **48.4 s** |
| 35B IQ4_XS | 92.3 s | **43.7 s** |

**研判** 一手（bit-identical + sha256 一致 + 2×）；量化工具链用户直接受益。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31201 （merge 2026-10-28T 窗口内；正文已核验，含 sha256）

### 5.2 `#31188` CUDA 融合 MoE gate/up+GLU（DGX Spark +17.2%）+ `#31288` top-k MoE 384 专家（router 子图 5×）

- **`#31188`** `CUDA: fuse MoE gate/up matmuls and GLU in MMQ`（1111 字符正文，#29948 扩展到 MoE）：专家 FFN 的 gate/up matmul 与 GLU 合成一次 `mul_mat_q` launch。**`llama-bench`（PP 16384, -ngl 999）：**
| GPU | model | master→PR |
|---|---|---|
| DGX Spark (GB10) | Qwen3-30B-A3B Q4_K_M | 2911→**3411 (+17.2%)** |
| RTX 5090 | Qwen3-30B-A3B Q4_K_M | 12612→**13330 (+5.7%)** |
| RTX 5090 | Qwen3.6-35B-A3B Q4_K_M | 15000→**15852 (+5.7%)** |

Perplexity（wiki.test）bit-identical。
- **`#31288`** `cuda : add top-k MoE fusion for 384 experts`（1514 字符正文）：启用既有 CUDA top-k MoE kernel 支持 DeepSeek-V4.1 的 384 专家；RTX A6000 上 router 子图（sqrt-softplus+bias+top-6+归一化+缩放）：1 token 23.91→**4.78 µs**、22 token 28.03→**5.74 µs**、512 token 76.82→**6.60 µs**（672/672 测例过 CPU 参考）。
- **`#31300`** `llama : remove simple batch splitting for DSV4`（2097 字符正文）：DSV4 移除简单 batch 拆分。
- **`#31230`** `Metal : add GGML_OP_TOP_K_MASK`（发布说明级，配合 ggml v0.28 稀疏注意力）。
- **`#28141`** `SVE 256-bit Implementation of mxfp4 gemv kernel`（1005 字符正文）：Arm SVE 256-bit 的 mxfp4 GEMV 内核（KleidiAI 系）。
- **`#30557`** `chat : stop qwen3-coder reasoning and content before template separators`：qwen3-coder 在模板分隔符前停止 reasoning/content。
- 发布说明另列：DeepSeek V4.1 Flash #31086、LiquidAI d1/LFM3 #31275/#31252/#31257、Flint #30928、50/50 SLRU MoE #30637、QSA/DSA 融合 mask #30640/#31009、CUDA chunked graph #30804、SYCL graph replay #28725、OpenCL Q2_K/Q3_K #30999/#31172、Hexagon HTP 重构、KleidiAI SVE #30808 等（均见第 1.1 节）。

**证据等级（第 5 节）：** `#31201`/`#31188`/`#31288` 为带逐卡表的性能一手（bit-identical 验证）；`#31300`/`#28141`/`#30557` 为正文一手（标题+正文）；`#31230` 及发布说明所列其余为发布说明级/标题级。

---

## 6. 累计结论与研判（沿用 + 新增）

**沿用 2026-10-28 档案（覆盖 10-27）第 6 节累计结论 1–7 条：**
1. 调度器隐式阻塞的定量画像（host sync 移出热路径的 +4.5%~9.9% 画像）；
2. **发版窗口的双向性**：发版当天合入 main ≠ 在版内（compare API 祖先关系核验）；
3. **默认值成为最主要的风险/收益载体**（默认后端/量化/NUMA 静默改变行为）；
4. **异构后端静默错误高频**（AMD ROCm / Metal / 异构 kernel 不报错只掉精度）；
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**（草稿/索引 KV 所有权一致性）；
6. **量化路径静默错误**（scale 未应用/参数丢失/4D 量化漏算 → 输出偏离且无报错）；
7. **专家张量/路由归属与分页漏接静默错误**（paged-experts 不被 paging、路由行覆盖 → 静默用错专家值）。

**本期新增第 8 条（静默正确性升维为跨引擎常态主线）：**
- vLLM 四条（`#63827` HiSparse 异步 KV 覆盖、`#62407` 量化参数丢失、`#63791` FlashInfer 稀疏 MLA 65535 静默跳过、`#64250` KV-Offload P2P 未确认 WRITE）+ SGLang 一条（`#46983` AMD 翻译窗口 fused writes 误写）+ TRT-LLM rc2 把 `#20823` NVFP4_AWQ scale 修复纳入 Fix 列表 + llama.cpp 历史 `#31097`（MiniMax-M3 scale 未前向）共同证明：**推理引擎当前的高频失效模式已不是「崩溃」，而是「不报错、精度静默掉/长上下文静默塌方」**。建议 serving 方把「带量化的端到端 GSM8K/长上下文（RULER-64K 级）回归」列为默认发布门禁（强化第 4、6、7 条）。

**版本门槛研判（强化第 2、3 条）：** 本期 llama.cpp v0.8.0 与 TRT-LLM v1.4.0rc2 同日发布，且 rc2 含 4 项 BREAKING + 弃用 n-gram/user 投机解码；升级方须把「发布说明 diff + 破坏性变更回滚预案」纳入门禁。`#19961`（DeepSeek V4/DSpark on Rubin）在 rc2 仍维持描述级，不夸大为已落地。

---

## 7. 趋势判断与可执行清单

**趋势判断**
1. **KV cache 正确性走向架构级**：vLLM KVCacheConfigBuilder 可插拔、SGLang #46973–#46991 字节存储/分配身份分离、TRT-LLM KVCM V2 原生 retention/SWA fork，三家共同把 KV cache 从「补丁修复」推向「职责分离架构」。
2. **静默正确性成为头号主题**：本期跨四家的 silent-error 案例数量与严重性均居首，长上下文（FlashInfer 65535）与量化（scale/参数）是重灾区。
3. **同日双版本 + 多 BREAKING**：rc2 的 4 项 BREAKING 与 llama.cpp `--model` 必填显示「版本门槛」在升高，升级回归成本上升。
4. **MoE/稀疏注意力硬件适配深化**：TRT-LLM GLM-5.3 Blackwell +19.8%、llama.cpp CUDA MoE 融合 +17.2%、SGLang DSv4.1 MegaGate/Engram、ggml v0.28 SVE/OpenCL/Hexagon 后端——MoE 与稀疏注意力的后端专项优化是性能主战场。

**可执行清单（建议 serving/研发方）**
- [ ] **llama.cpp 升级 v0.8.0**：回归 `--model`/`-hf` 必填改动；验证 ggml v0.28 内核语义（`-1` 专家索引、TOP_K_MASK、CUDA chunked graph）；量化方用 `#31201` 新 IQ4 路径（bit-identical，2×）。
- [ ] **TRT-LLM 升级 v1.4.0rc2**：处理 4 项 BREAKING（#20388 LlmRequest Python 化、#20129 KVCM V1 移除、#20043 旧 executor 兼容移除、#20108 Mooncake 零拷贝）；n-gram/user 投机解码（#20438）迁移到单模型/通用投机解码；确认 `#20823` NVFP4_AWQ scale 修复随 rc2 生效。
- [ ] **vLLM HiSparse + 异步 KV + 量化重载路径**：跑 GLM-5.3 长多轮（#63827）、量化重载 GSM8K（#62407）、FlashInfer 稀疏 MLA 长 `--max-num-batched-tokens`（#63791）回归。
- [ ] **SGLang AMD ROCm 部署**：对含 #46973–#46991 stack 的 main，自测 unified sliding-window + fp8 KV 的 fused store（#46983 仅 CPU 回归、ROCm 未实测）。
- [ ] **Blackwell GLM-5.3-Flash**：采用 TRTLLM-Gen MoE 后端 + FP8 context indexer 默认（#20755，NVFP4 +19.8%）；长 prompt 低并发试 EP1。
- [ ] **跨引擎回归门禁**：把「带量化端到端 GSM8K + RULER-64K 长上下文」列为默认发布门禁（第 8 条）。

---

## 8. 来源索引与数据质量

**官方一手（提交 API + 发布说明 + PR 正文）**
- vLLM：https://github.com/vllm-project/vllm （77 commits @2026-10-28；v0.32.0 2026-10-18）；PR `#63827`/`#62407`/`#63791`/`#64250`/`#61464`/`#62240`/`#63018`/`#61699` 正文已抓。
- SGLang：https://github.com/sgl-project/sglang （90 commits @2026-10-28；v0.5.22 2026-10-19）；PR `#46987`/`#46983`/`#47148`/`#46805`/`#46743`/`#34031` 正文已抓。
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM （45 commits @2026-10-28；**v1.4.0rc2 2026-10-28T15:51:34Z** prerelease，wheel 1.4.0rc2.post1）；发布说明全文已抓；PR `#20694`/`#20755`/`#20893`/`#20597` 正文已抓。
- llama.cpp：https://github.com/ggml-org/llama.cpp （27 commits @2026-10-28；**v0.8.0 2026-10-28T12:47:51Z** stable；nightly b11968–b11972）；发布说明全文已抓；PR `#31201`/`#31300`/`#31188`/`#31288`/`#28141` 正文已抓。

**社区/外部（二手线索，未核验）**
- HN（2026-10-28 窗口，Algolia HTTPS）：「DeepSeek-v4.1-Flash: Pushing the Limits of KV Cache Compression」（5 分，引擎相关文章，未核验其内容主张）；「A garbage collector for LLM context windows, written in Rust」（1 分）；「Show HN: ThreadShelf – Obsidian and AI chats … llama.cpp」（1 分，弱相关）。其余 vllm/sglang/TensorRT-LLM/「LLM inference」等关键词命中多为空标题/评论级，未计入。
- Reddit / YouTube / X：未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

**数据质量与限制**
- 计数口径：GitHub commit API 的 `since`/`until` UTC 日期窗口（vLLM 77 / SGLang 90 / TRT-LLM 45 / llama.cpp 27，均 HTTP 200）；该窗口按 commit date 过滤，偶有相邻日 commit 边界误差，属历史一致口径。
- 证据等级：带验证表的 silent-error/性能 PR 记为「官方一手（含验证）」；仅提交标题、未抓正文的记为「标题级」；TRT-LLM `#19961` 维持「描述级」（diff 未显式落地快路径）；SGLang `#46983` 记为「描述级/未全绿」（AMD hw 未测、CI 未全绿）；社区文章记为「二手线索/未核验」。
- 未调用外部浏览器（agent-browser）：本期数据源（GitHub API / releases / HN Algolia）均可直接 HTTP 访问，沿用 2026-10-18 起确立的「API + PR 正文直抓」方法，未触发浏览器需求（与历次一致）。
- `find-skills`：经 SkillHub 语义检索（EN+ZH 各 1 组）最高相关性 0.05（防骗大师等通用技能），与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，**未安装**（与历次一致）。

> 本档案仅覆盖 2026-10-28 UTC 窗口；如需回溯相邻日期，请以各引擎官方提交 API 按日期区间重新采集。
