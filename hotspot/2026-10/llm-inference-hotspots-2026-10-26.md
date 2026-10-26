# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-26（北京时间）
- **关注窗口：** 2026-10-25 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-25 08:00 – 2026-10-26 08:00）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-25 UTC 窗口，分别返回 **34 / 44 / 7 / 20** 条，均 HTTP 200）、四家官方 releases API、**27 个 PR 正文**（经 GitHub issue API 抓取，返回 markdown 文本，含 vLLM `#61976`/`#60513`/`#60084`/`#61019`/`#63476`/`#63372`/`#63266`/`#62698`、SGLang `#46464`/`#46465`/`#46423`/`#46359`/`#45787`/`#46141`/`#46140`/`#46139`/`#46365`/`#46137`、TensorRT-LLM `#20679`/`#20213`/`#20948`、llama.cpp `#31086`/`#29948`/`#31035`/`#30820`/`#31044`/`#30999`）。**窗口特征：** 2026-10-25 为**周日**，TRT-LLM（7）提交量显著低于工作日中位、符合既往周末规律；SGLang（44）仍高，主因 MiniMax-M3.1 / Kimi-K3 性能栈与 diffusion 性能潮在途合入；vLLM（34）/ llama.cpp（20）为常态周日量。HN Algolia 公开 API 可访问（HTTPS 200），窗口内仅 1 条指向推理引擎的前台故事（「Benchmarking DSpark speculative decoding on a consumer GPU」，2 分）；`vllm`/`sglang`/`TensorRT-LLM`/`KV cache`/`LLM inference` 的其余命中**均为评论级（无标题、无分值）**，无一条指向引擎本身的前台发布。Reddit、YouTube、X 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** 窗口内 vLLM / SGLang / TensorRT-LLM **均无新正式版本**（vLLM 仍 **v0.32.0** 2026-10-18；SGLang 仍 **v0.5.22** 2026-10-19；TensorRT-LLM 仍 **v1.4.0rc1** 2026-10-22，prerelease）。llama.cpp 滚动 nightly 共 **18 个标签**（b11870–b11889，均发布于 2026-10-25，已纳入 UTC 窗口；b11872 未落在当日窗口）。无新版本事件，故本期头号事件为跨引擎「量化 × MoE 专家分页/路由静默错误」+ vLLM KV-Offload 静默正确性簇（见第 1 节）。
- **本期方法论沿用：** 继续用 GitHub commit API 的 UTC 日期窗口 + PR 正文经 issue API 抓取（返回干净 markdown，比 HTML div 配对更省截断）；延续「版本核验」口径——发布说明/描述 ≠「修复已进版」，且对 PR 描述与 diff/CI 不一致者按「描述级/未完全绿」记录（如 SGLang `#46139`/`#46140`/`#46141` CI 未全绿、llama.cpp `#31086` 描述为空）。

---

## 1. 版本状态与本期头号事件：周日窗口「量化 × MoE 专家分页/路由静默错误」跨引擎共振 + vLLM KV-Offload 静默正确性簇

**主题**  本期无新版本发布，但跨引擎集中出现**「无报错却产出错误结果」的量化 / MoE 专家归属与分页漏接**修复，是上期累计结论第 6 条（量化路径静默错误）的同源延伸，并把落点从「量化 scale 漏接」扩到「MoE 专家张量 / 路由的归属或分页漏接」。同窗口 vLLM 还出现一组 KV-Offload 静默正确性修复。

**内容详情（取自 PR 正文）**

- **SGLang `#45787` —— paged-experts 多量化格式「未声明张量不被 paging → 静默用错专家值」：** 将 paged-experts（`#42557`）扩展到 AWQ / compressed-tensors / MXFP4 / NVFP4 / W8A8 / block-INT8 专家。PR 自述按手工程式声明格式时**会静默失败**：① 未列入 paging 的张量不被分页，于是 paged-in 专家用**另一专家的值**跑（AWQ 对照组 zero points 未分页：**median 1.67 nats，而正确分页为 0.003**）；② 声明的 runner 可能与层实际构建的 runner 不一致（ModelOpt NVFP4 默认走 FlashInfer CUTLASS、compressed-tensors int4 默认走 Triton WNA16）；③ 每个 repacking 格式都需要重置「已 repack」钩子。这是与 TRT-LLM `#20823`、llama.cpp `#31097` 同构的**量化/专家路径静默错误**。
- **vLLM `#63372` —— MRV2 + `--enable-return-routed-experts`：每个 DBO microbatch 从 row 0 写路由，最后一个覆盖其余（静默错路由）：** DBO 开启时每个 microbatch 都从行 0 写路由结果，末批覆盖前批。PR 给出对拍：DBO on / main 与 DBO-off 同路由占比 **0%**，本 PR **89.4%**（噪声 88–91%）。无报错，纯静默错路由。
- **llama.cpp `#30820` —— 4D 张量量化漏算 `ne[3]` + 专家张量误判：** 修 `#29916`：`a.conv1d.0.weight [3,3,1,128]` 因行数只取 `ne[1]`/`ne[2]` 漏算 `ne[3]` 触发 assert；专家张量改用名字（`_exps.`/`_chexps.`）而非形状判定，**额外捕获一处边界：deepseek-v3.1 的注意力权重曾被误判为专家张量（本应按 `Q8_0` 量化）**。属量化路径的「张量归类漏接」。
- **vLLM KV-Offload 静默正确性簇（同日合入，均属「默认配置下易踩、无显眼报错」）：** `#63266` 当 GPU 前缀落在 chunked-local attention 块内时，`SimpleCPUOffloadConnector` 误命中 CPU 后缀、产出**错误输出**（修复前 16-token 局部前缀两用例返回 `(64, True)` 而非 `(0, False)`，2 failed/4 passed → 修复后 112 passed）；`#61019` 在 EAGLE/MTP draft + `mamba_cache_mode="align"` 下，原生 CPU `OffloadingConnector` 从不服务 partial tail，每次 CPU 层命中停在调度块边界、后续 token 重算（修复后 8×A100 MLPerf agentic SLO 92.42% vs 91.70%；存储策略避免 P95 TTFT +13%）；另有 `#61390`/`#63010`/`#63126`/`#61780`/`#60810`/`#55576` 一组 KV offload 健壮性修复（已发布块当已存、重启游标重置、异步迭代器错误捕获、次级层失败转背压、续流会话块表重置、NIXL OBJ 层遥测）。

**应用场景**  NVFP4 / MXFP4 / block-FP8 / AWQ / compressed-tensors 的 MoE 与 paged-experts 部署（SGLang、TRT-LLM MegaMoE、llama.cpp GGUF）；vLLM 的 CPU/分层 KV offload（Mooncake/原生 `OffloadingConnector` 等）与 MRV2/DBO 多微批路由。共性危害：**不触发任何异常，只能靠与黄金参考对拍或人工抽查发现**。

**研判**
- **事实：** 上述均为「张量/路由归属或分页漏接 → 输出偏离基准且无报错」的静默错误，分别由 N3mezis（SGLang `#45787`）、algebra-MCX（vLLM `#63372`）、anthony-wss（llama.cpp `#30820`）、charxwu（vLLM `#63266`）、kimseunghyun-kr（vLLM `#61019`）提交，正文均给根因链路与对拍/复现数据。
- **观点：** 这是上期累计结论第 4、6 条的**更宽子类**——「量化/MoE 静默正确性」债务：① 异构后端 miscompile（#4）；② 量化 scale 漏接（#6）；③ 本期新增**专家张量/路由归属与分页漏接**（#7）。三者都不依赖特定硬件、纯属代码归属/前向疏漏，因此**任何量化或 MoE 部署（含纯 CPU/同构 GPU）都应把「与黄金参考逐层对拍 + 专家输出抽样核对」列为上线验收**。本期将其立为累计结论第 7 条。

**来源**
- SGLang：`https://github.com/sgl-project/sglang/pull/45787` （merge 2026-10-25T15:26:25Z；正文已核验；median 1.67 nats 为 PR 自述控制组数字）
- vLLM：`https://github.com/vllm-project/vllm/pull/63372` （merge 2026-10-25T04:45:26Z；89.4% 对拍为 PR 自述）、`https://github.com/vllm-project/vllm/pull/63266` （merge 2026-10-25T04:18:50Z）、`https://github.com/vllm-project/vllm/pull/61019` （merge 2026-10-25T06:03:53Z）
- llama.cpp：`https://github.com/ggml-org/llama.cpp/pull/30820` （merge 2026-10-25T11:28:39Z；正文已核验）

**证据等级：** 官方一手（GitHub 提交 API + PR 正文，均含根因与对拍/复现数据）。SGLang `#45787` 的 1.67 nats 为 PR 自身控制组数字、未做独立第三方对拍，但复现条件具体；vLLM `#63372`/`#63266`/`#61019` 为修复类 PR 自述 + 测试增量，未做独立数值对拍，按规范记为**官方一手但端到端未在公开 checkpoint 外复现**。

---

## 2. vLLM：10-25 提交（34 条）——KV Offload 静默正确性、MRV2/DBO 路由与分级写、ROCm 性能

**主题**  vLLM 在 2026-10-25 UTC 窗口合并 **34 条**提交（周日量），主线为 **KV Offload 静默正确性簇**（见第 1 节）、**MRV2/DBO 路由与分级写**、**ROCm/AMD 性能**、以及调度优先级与投机 auto-fit 正确性。

### 2.1 `#62698` MRV2 分级写并行化：长 prompt 写 31×–120× 更快（GB200）

**主题**  `[Perf][MRV2] Parallelize staged writes across 1024-element chunks`（`#62698`）。

**内容详情（取自 PR 正文）**  `_apply_write_kernel`（MRV2 `StagedWriteTensor`/`FusedStagedWriter`）原每 staged write 起一个 program 并循环 1024 元素块，长 prompt 写落到单 SM。改为每块起一个 program；并在无前缀缓存/指标收集时去掉 `BlockPool.get_new_blocks` 的逐块检查。微基准（GB200，FlashInfer CUPTI）：1×128K token 写 **76.32 → 2.46 us（31×）**、1×855K **443.42 → 3.71 us（119.5×）**、1×128K+63×1 token **75.20 → 5.94 us（12.7×）**；decode 步/块表小写入不回归（~1.2×）。服务侧（DeepSeek V4.1 Flash，P/D decode，DEP，128K prompt，GB200）：原 admitting rank 在 90/98 步成为最慢、落后中位 252 us；改后 DEP8 仅 ~10 us 偏移。

**应用场景**  MRV2 长上下文 prefill 落盘/块表写入热路径；DEP 下 admitting rank 的步内尾延迟。

**研判**
- **事实：** 有 GB200 微基准 + 服务侧步内偏移实测的性能改动，长写数量级加速、小写不回归。
- **观点：** 这是 MRV2 长上下文路径的「白拿」性能，对长 prompt 服务方收益显著；与第 1 节静默正确性修复同属 MRV2 主线，说明该路径仍在正确性 + 性能双修。

**来源**  https://github.com/vllm-project/vllm/pull/62698 （merge 2026-10-25T02:57:08Z；正文已核验）

### 2.2 `#60084` ROCm：恢复 TRITON_ATTN KV 视图 32 位缓冲寻址，RDNA3 吞吐 +18%

**主题**  `[ROCm][Perf] Restore 32-bit buffer addressing for TRITON_ATTN KV cache views`（`#60084`）。

**内容详情（取自 PR 正文）**  ROCm 上给 `TRITON_ATTN` 用的 K/V cache 视图加 `ptr_range()` 限定到本层区域，使 Triton AMD 后端保留 32 位缓冲寻址，修掉 `#51718`（把 KV 移入单一 backing allocation）引入的 **~16% 吞吐回归**。Radeon Pro W7900（gfx1100）：Whisper-large-v3 **5.34 → 6.31 req/s（+18.2%）**、Qwen2.5-7B-Instruct **2.70 → 3.19 req/s（+18.4%）**；精度不变（2,703 条 LibriSpeech 转写 token 一致，GSM8K 在噪声内）。Python-only 改动，不影响非 ROCm 与其它 KV 布局。

**应用场景**  ROCm RDNA3 上 `TRITON_ATTN` 后端的吞吐回归回收。

**研判**
- **事实：** 有同 GPU 回退实测（base/PR 背靠背）、吞吐 +18%、精度不变。
- **观点：** 与往期 ROCm 侧「后端回归/静默错误」形成对照——本条是良性回归回收；提醒 ROCm 部署方若跑 `TRITON_ATTN` 且内核版本含 `#51718`，应确认已含本修复。

**来源**  https://github.com/vllm-project/vllm/pull/60084 （merge 2026-10-25T06:59:13Z；正文已核验）

### 2.3 `#61976` 调度优先级自抢占后继续调度后续 running 请求（正确性）

**主题**  `[Bugfix][Core] Keep scheduling running requests after a priority self-preemption`（`#61976`）。

**内容详情（取自 PR 正文）**  `--scheduling-policy priority` 下，当 running 请求拿不到 KV 块时调度器抢占最低优先级 running 请求；但 `self.running` 按准入序而非优先级序，若被抢者是当前正在调度的中低位请求，会触发 `break` 使之后所有更高优先级 running 请求当步拿不到 token（优先级反转）。自抢占后该请求已移出 `self.running`，`req_index` 已指向下一个，修复即继续循环。新增 `test_priority_self_preemption_keeps_scheduling_later_running_requests`。

**应用场景**  优先级调度策略下的 decode 步公平性/正确性。

**研判**
- **事实：** 明确的优先级反转根因 + 单测；FCFS 路径仅当最后一条 self-preempt 才影响，行为不变。
- **观点：** 属调度器隐式阻塞类（累计结论第 1 条）的边界修正，影响面窄但属「默认配置下静默退化」，优先级部署方应确认已含。

**来源**  https://github.com/vllm-project/vllm/pull/61976 （merge 2026-10-25T16:31:09Z；正文已核验）

### 2.4 `#60513` 投机解码：`--max-model-len auto` 缩减后同步 speculator 上限（正确性）

**主题**  `[Bugfix][Spec Decode] Keep speculator limits in sync after auto-fit`（`#60513`）。

**内容详情（取自 PR 正文）**  `--max-model-len auto` 缩减上下文后，MRV2 speculator 仍保留原上限，draft attention 元数据越界。从 `ModelConfig` 读已解析上限并据此截断 draft 序列边界。单测 `test_build_draft_attn_metadata_caps_max_seq_len_at_max_model_len`：main `assert 1024 == 512` 失败，本 PR 通过；RTX PRO 4000 Blackwell + Qwen3-8B-eagle3 实测 auto-fit 后 draft 元数据被正确 cap。

**应用场景**  投机解码 + 上下文自适应缩减（auto-fit）组合路径。

**研判**
- **事实：** 有单测断言翻转 + GPU 实测；属「投机解码 × auto-fit」交叉路径（呼应累计结论第 5 条）。
- **观点：** 默认开启 auto-fit 且用投机解码的服务方应确认已含本修复，否则 draft 元数据越界。

**来源**  https://github.com/vllm-project/vllm/pull/60513 （merge 2026-10-25T16:14:30Z；正文已核验）

### 2.5 其余 vLLM 窗口提交（标题级，正文未逐一抓取）

- `#63476` `[Bugfix][MRV2] Report the first failed DBO microbatch`：两个 DBO microbatch 失败时原报最低 index 而非先失败者，掩盖根因（`min(errors)`→`next(iter(errors))`，单测参数化）。
- `#63451` `[Bugfix] Release cached cuBLAS workspaces after daemon warmup`：释放 daemon warmup 后缓存的 cuBLAS workspace（内存）。
- `#63088` `[Fast Start] daemon-to-daemon warmup cache seeding for Preload`：Preload 的 daemon 间 warmup cache 播种（冷启加速）。
- `#59223` `[Bugfix] Preserve FP8 indexer WK pairs across load_weights`：跨 `load_weights` 保留 FP8 indexer 的 W/K 配对（量化正确性）。
- `#63495` `[Pooling] Reduce Clef metadata overhead`：Pooling 路径 Clef 元数据开销。
- `#60149` `[Model] Add Dory support`：新增 Dory 模型支持。
- `#63616`/`#63613` `[Benchmark]`：bench forward 增加 steady-state PP 与 PCP 支持。

**证据等级：** 官方一手（提交 API + 标题）；`#63476`/`#63451`/`#63088`/`#59223`/`#63495`/`#60149` 仅标题级，未抓取正文，按规范记为**官方一手（标题级）**。

---

## 3. SGLang：10-25 提交（44 条）——MiniMax-M3.1 MSA-v1 NVFP4 稀疏、paged-experts 多量化、Kimi-K3/KDA SM107 性能、PD KV 校验

**主题**  SGLang 在 2026-10-25 UTC 窗口合并 **44 条**提交（周日量偏高），主线为 **MiniMax-M3.1 MSA-v1 NVFP4 稀疏注意力栈**（#46139/#46140/#46141，均从 `#45925` 按字节切出）、**paged-experts 多量化格式**（#45787，含静默失败，见第 1 节）、**Kimi-K3 / KDA 在 SM107(Rubin) 的性能微雕**（#46464/#46465）、**PD KV checksum 校验补全**（#46423）、以及 DCP block-FP8 q_b_proj 复制（#46359）。

### 3.1 `#46464` / `#46465` Kimi-K3 与 KDA 在 SM107 的性能微雕（Rubin）

**主题**  `[Perf][NVIDIA] kimi-k3: FP32 warp split-K GEMM for the MoE front projection on SM107`（`#46464`）与 `[Perf][NVIDIA] kda: Narrower V tiles for the KDA spec-verify recurrence on SM107`（`#46465`）。

**内容详情（取自 PR 正文）**
- `#46464`：Kimi-K3 合并 MoE 前投影 `(6016, 7168)` 需 FP32 输出以保 router logits 精度，在 spec-verify 批大小时走 TGV；SM107 上 warp split-K（cp.async ring loads）在 `M=8/16/32` 更快。临时拷出 FlashInfer 同核（后者仅收 bf16 out），差异 71 行即 `out.dtype in (bf16, fp32)` 并按输出 dtype 键化编译缓存。精度：vs PyTorch FP32 最大 abs 误差 0.001221（容差 0.005），router top-16 每轮一致。
- `#46465`：`fused_sigmoid_gating_delta_rule_update_kernel` 是 KDA 每层 recurrence、位于 spec-verify 关键路径；原 `grid=(V/32,1,heads)`、每 CTA 一 warp，TP8（每 rank 12 heads）即 48 个单 warp CTA 在 212-SM 卡上大部分空闲。SM107 改 `BV=4` 一 warp（TP8 下 384 CTA）。输出与 BV=32 逐列相同；state 在 bf16 舍入内（max abs diff 1.2e-4）。微基准：BV=32 **13.66 us → BV=4 6.99 us**；整模 Kimi-K3 NVFP4 TP8/DCP8、128k 输入 95% 前缀命中、1k 输出、DSpark block 7 + ReplaySSM：KDA `delta_rule_update` 每 verify 步 **1072.5 → 789.3 us（−283 us，约 −26%）**。

**应用场景**  Kimi-K3 NVFP4 在 Rubin(SM107) 的 spec-verify 关键路径（MoE 前投影精度保持 + KDA recurrence 占用）。

**研判**
- **事实：** 两项均有 SM107 实测微基准 + 精度对拍（误差在容差/舍入内）。
- **观点：** 属 Rubin 后端 hot-path 微雕，逐核小幅但叠加在 spec-verify 关键路径；非静默错误，是性能收益。

**来源**  https://github.com/sgl-project/sglang/pull/46464 （merge 2026-10-25T22:25:33Z）、`https://github.com/sgl-project/sglang/pull/46465` （merge 2026-10-25T20:38:45Z；正文已核验）

### 3.2 `#46423` PD KV checksum 在 decode 侧 radix cache 下验证（正确性）

**主题**  `[Feat] disaggregation: Verify the KV checksum under a decode-side radix cache`（`#46423`，#39729 后续）。

**内容详情（取自 PR 正文）**  #39729 留两缺口：① 开 `--disaggregation-decode-enable-radix-cache` 时 decode 只缓存前缀、prefill 只发后缀，digest 却覆盖整个 prompt，于是每次命中都像损坏、decode 带警告关掉 checksum（恰是常见生产形态，无法在关键处开）；② layout 签名只哈希页大小 + 每 buffer 字节长，fp16 vs bf16 / 两个 DeepSeek-V4 页布局 / 不同 state 类型集会碰撞，导致每次请求都报不匹配。修复：prefill 与 decode 各 digest `[prefix_len, end_idx)` 同一范围（同来自 `prefix_match.decode_prefix_len`，不新增线上字段）；签名纳入 `kv_cache_dtype_str`/`kv_layout`/`swa_kv_cache_layout`/state 类型（attention backend 故意排除，因角色间可能不同）。滚动说明：签名变更后，本 PR 的 decode 配 #39729 的 prefill 会打一条「layout differs」警告并放行未校验请求，直至 prefill 升级；**不 abort、不 hang**。

**应用场景**  SGLang PD（prefill/decode 分离）部署的 KV 完整性校验（防脏 KV 命中）。

**研判**
- **事实：** 明确两个缺口根因 + 修复 + 滚动兼容语义；属 PD KV 完整性正确性。
- **观点：** 与累计结论第 5 条（前缀缓存 × MoE 路由 × PD 边界债务）同源；PD 部署方升级 `#46423` 时须把 prefill 一并升级到含 #39729 之后版本，否则该窗口内请求走未校验放行。

**来源**  https://github.com/sgl-project/sglang/pull/46423 （merge 2026-10-25T17:51:23Z；正文已核验）

### 3.3 `#46359` DCP 复制 block-FP8 q_b_proj：Kimi-K3 NVFP4 移出 decode 图内的逐层 Q all-gather

**主题**  `[Feat] dcp: Replicate a block-FP8 q_b_proj under --dcp-replicate-q-proj`（`#46359`）。

**内容详情（取自 PR 正文）**  `--dcp-replicate-q-proj` 原本只收 bf16/fp16 `q_b_proj`，Kimi-K3 NVFP4 checkpoint 默认开此 flag 于 a2a/fi_a2a 后端却跳过每层 MLA、把逐层 Q all-gather 留在 decode 图内。现接受 128×128 block-FP8 `Fp8LinearMethod`，`_gather_block_fp8_linear` 还原逻辑 fp32 block scales（撤销 UE8M0 打包）、沿输出维 all-gather 权重与 scales 并重建副本的 `process_weights_after_loading`。精度：GB300(SM103) 上 block-FP8 线性拆 4 个 DCP 风格行分片、经 `_gather_block_fp8_linear` 重建与未分片层**逐字节一致**（权重字节 + GEMM 输出在 M=1/8/48 于 triton/deep_gemm/flashinfer_trtllm/flashinfer_cutlass 均一致）；作者指出 deep_gemm 下对 gathered 权重重跑 UE8M0 requant 非幂等，故在副本内重打包。GSM8K 200 题 5-shot：baseline（`--no-dcp-replicate-q-proj`，即 main 在该 checkpoint 的实际行为）与本 PR 对拍（表头在正文截断，未见具体数字）。

**应用场景**  Kimi-K3 NVFP4 在 DCP 下的 MLA q-proj 复制（decode 图瘦身）。

**研判**
- **事实：** 有 SM103 逐字节一致性对拍（4 后端）；属 DCP decode 图优化 + 正确性。
- **观点：** 良性优化（移除 decode 图内逐层 Q all-gather），且 bit-identical；Kimi-K3 NVFP4 + DCP 部署方应确认已含本修复以回收 decode 图开销。

**来源**  https://github.com/sgl-project/sglang/pull/46359 （merge 2026-10-25T20:06:34Z；正文已核验；GSM8K 具体数值正文截断未给出，记为**官方一手但 GSM8K 数值未完整呈现**）

### 3.4 MiniMax-M3.1 MSA-v1 NVFP4 稀疏栈（#46139/#46140/#46141，CI 未全绿）

**主题**  MiniMax-M3.1 的三连 PR：`#46139`（MSA-v1 decode + NVFP4 KV cache kernels）、`#46140`（MiniMax-M3.1 NVFP4 MoE method）、`#46141`（MSA-v1 sparse backend + NVFP4 KV pool），均从 `#45925`（MiniMax-M3.1-Flash 支持）按字节切出。

**内容详情（取自 PR 正文 + CI 状态）**  `#46141` 在 NVFP4 KV pool 上做 MiniMax-M3.1 的 MSA-v1 稀疏注意力：主 KV 与 index K 存为 NVFP4 page 行、Q8KV4 decode indexer（FP8 回退）、MSA-v1 prefill、池容量按每 token 字节数定；并给 grouped QK-norm/RoPE kernel 加 FP8 Q epilogue。`#46140` 将 `minimax_m3_nvfp4` 量化接到 TRT-LLM routed runner / MegaMoE / Marlin，并支持 deferred-finalize 交接。`#46139` 为独立 kernel（decode 规划、NVFP4↔FP8 反量化、NVFP4 K/V/index 存、HiCache 用的页行 host/device 传输），main 内尚无调用方、由下一 PR 接入。

**应用场景**  MiniMax-M3.1-Flash 的 NVFP4 KV + 稀疏注意力（新后端/量化栈）。

**研判**
- **事实：** 三个 PR 均为「从 `#45925` 按字节切出」的功能切片；CI 状态显示 **Base / Extra 未运行（:no_entry_sign:）、AMD ROCm 10 失败（:x:）**。
- **观点：** 属新功能栈接入、非修复；因 **CI 未全绿**，按规范记为**官方一手（描述级/未完全绿）**——采信任一性能/精度声称前须等 CI 转绿与后续接入 PR。本期不把其数值/精度当作已验证事实。

**来源**  https://github.com/sgl-project/sglang/pull/46139 （merge 2026-10-25T09:41:02Z）、`https://github.com/sgl-project/sglang/pull/46140` （merge 2026-10-25T12:41:58Z）、`https://github.com/sgl-project/sglang/pull/46141` （merge 2026-10-25T14:34:04Z；CI 状态取自 PR 页）

**证据等级：** 官方一手（提交 API + PR 正文 + CI 状态），但 **CI 未全绿** → 描述级/未完全绿。

### 3.5 其余 SGLang 窗口提交（标题级，正文未逐一抓取）

- `#46365` `[Fix] deepseek-v4: Complete the MoE output locally when the transport declares no sum group`：修 `#46316`，a2a MoE（`flashinfer_megamoe`）下 DeepSeek-V4 捕获 decode CUDA graph 报「transport finalize owes an undeclared sum group」；当 transport 声明无 sum group 且专家输出已求和时本地 `complete(reduce=False)` 收尾（避免双加）。
- `#46137` `[Feat] layer-boundary: Fuse TP all-reduce, residual add and RMSNorm on the push plane`：`CustomAllReduceFusion` 把 TP all-reduce + 残差加 + RMSNorm 融为单 kernel，并折叠 MoE finalize 与 shared-expert 加；CI Base/Extra 绿、AMD ROCm 10 红。
- `#46463` `[Fix][NVIDIA] dist: Detect MNNVL fabric devices by NVML fabric state`：按 NVML fabric 状态而非仅 GB200/GB300 名字识别 MNNVL fabric 设备（正确性）。
- `#46138` `[Fix] dp-attention: Align prefill graph buckets and decode padding across DP ranks`。
- `#46330`/`#46290` `[Feat/Refactor] spec: CUDA attention DP for DFlash/DSpark` + `Resolve draft parallel placement from model declarations`（投机解码）。
- `#45076` `[Fix] hicache: Count the EAGLE bigram boundary token in PD decode restores`；`#43727` `[Feat] hicache: Export host Mamba pool occupancy next to host KV`；`#41701` `[Fix][AMD] mamba: chunked prefill cut on mamba chunk grid so DCP can donate checkpoints`；`#45104` `[Fix] hisparse: Copy DeepSeek-V4 eager backups token by token`。
- diffusion 性能潮（约占当日 1/4+）：`#46154`/`#46173`/`#46111`/`#46047`/`#46044`/`#46025`/`#46156`/`#46264`（shared-memory 回收、VAE decode 快路径、RMSNorm/RoPE 向量化、video 模型去 host sync、zimage RoPE 表隔离、失败隔离、cursor 分页、分段行并行权重加载）、`#46124` `[Feat] diffusion/model: support Hy-Nomas`、`#40595` `[Feat] unified-cache: Support out-of-tree external linkers`。

**证据等级：** 官方一手（提交 API + 标题）；除 3.1–3.4 已抓正文者外，其余为**官方一手（标题级）**。

---

## 4. TensorRT-LLM：10-25 提交（7 条）——采样器内存、Qwen3-VL 编码器 CUDA graph、惰性导入错误归因

**主题**  TensorRT-LLM 在 2026-10-25 UTC 窗口仅合并 **7 条**提交（周日量，符合周末规律），无性能潮，以 PyTorch 采样器内存修复、多模态编码器 CUDA graph、serve 惰性导入错误归因为主，余为 pre-merge 测试豁免（infra）。

### 4.1 `#20679` 避免 PyTorch 采样器全局 stop-word 缓冲增长（内存/正确性，CI 失败须谨慎）

**主题**  `[TRTLLM-17184][fix] Avoid global stop-word buffer growth in the PyTorch sampler`（`#20679`）。

**内容详情（取自 PR 正文）**  采样器把全局 padded 的 stop-word / history 缓冲替换为每请求紧凑分配；定长 slot 元数据支撑 Triton 匹配与 beam-history 重排；单 token 规则不需 history 分配；无保留请求需要时不跑 history 工作。`PyExecutor._free_request_resources` 在 executor 释放前释放采样器拥有的请求数据。Beam-search 元数据改用紧凑 stop-word 元数据（移除 `CBAStepResult.reordered_window`）。**作者自陈限制：** 报告的 B200 内存测量是隔离的 handler 测量、**非整模 GPU 内存**；报告的采样器 benchmark **不构成通用加速或整模吞吐增益**；全源码构建 executor 集成与全模型精度**未重跑**；**报告 CI 运行失败、上下文未含失败细节**。

**应用场景**  TRT-LLM PyTorch 采样器的 per-request 内存（beam search / stop-word）。

**研判**
- **事实：** 有 Dev/QA 双评审摘要；但作者明确标注 CI 失败、全模型精度未重跑、内存数字非整模。
- **观点：** 属内存/API 清理，价值偏内部；因 **CI 失败且全模型精度未复跑**，建议谨慎评估，勿据「内存下降」外推到生产收益。本期记为**官方一手但 CI 未通过、全模型精度未验证**。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20679 （merge 2026-10-25T21:29:16Z；正文已核验；CI 状态取自 PR 自述）

**证据等级：** 官方一手（PR 正文）但 **CI 未通过、全模型精度未重跑** → 描述级/未完全验证。

### 4.2 `#20213` Qwen3-VL 视觉编码器 CUDA graph（性能）

**主题**  `[None][perf] enable Qwen3.*-VL encoder CUDA graphs via shared setup`（`#20213`）。

**内容详情（取自 PR 正文）**  `multimodal_config.encoder_cuda_graph` 已存在，但 Qwen3-VL 视觉编码器未捕获/重放 graph；RADIO/GLM 经独立权重加载钩子初始化。本 PR 加 Qwen3-VL 编码器 graph dispatch，并在权重与 attention 元数据就绪后、KV-cache 定容前给受支持编码器共享生命周期；复用 `MultimodalEncoderMixin`/`MultimodalEncoderGraphRunner`，跨 Qwen3-VL/RADIO/GLM 共享 graph teardown，保持 NCCL window 所有权策略；共享 packed no-KV attention 元数据刷新。新增可选 `max_batch_size`/`max_num_tokens` 自动 shape。正文**未给具体数值性能**。

**应用场景**  TRT-LLM 多模态（Qwen3-VL / RADIO / GLM）视觉编码器图捕获。

**研判**
- **事实：** 明确把 Qwen3-VL 编码器纳入既有 CUDA graph 机制；配置示例给出。
- **观点：** 良性多模态性能路径（减少编码器重放开销），但**无量化收益数字**，按事实记录、不夸大。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20213 （merge 2026-10-25T15:02:06Z；正文已核验）

### 4.3 `#20948` serve 惰性导入错误归因（可观测性）

**主题**  `[TRTLLM-15863][fix] Raise ImportError with the cause when a lazily imported serve module raises AttributeError`（`#20948`，#20660 后续）。

**内容详情（取自 PR 正文）**  #19964 把 `tensorrt_llm.serve` 改成 PEP 562 惰性包，其 `__getattr__` 让 server 模块导入时的 `AttributeError` 原样逃逸。`trtllm-serve` 在模块级 `from tensorrt_llm.serve import OpenAIServer` 会**把真正的因由吞掉**，仅报「cannot import name 'OpenAIServer'」；`hasattr` 返回 False、`getattr(..., default)` 返回默认，掩盖 bug。本 PR 改为 `ImportError(..., name=__name__) from e`。单测：A10 上 base 8 个新 `tensorrt_llm.serve` 用例失败、本 PR 全 44 通过；`test_serve_lazy_import.py` 与 `--help` 两树输出一致。

**应用场景**  TRT-LLM serve 启动错误可诊断性。

**研判**
- **事实：** 有 A10 单测翻转（8 失败→44 全过）；属错误归因修正，无行为变更当模块干净时。
- **观点：** 纯可观测性/健壮性，影响面小但能避免「serve 起不来却看不到真因」。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20948 （merge 2026-10-25T11:49:53Z；正文已核验）

### 4.4 其余 TRT-LLM 窗口提交（标题级）

- `#20889` `[None][fix] retry wrapped connection resets with transient budget`：连接重置带瞬时预算重试（健壮性）。
- `#20959`/`#20960`/`#20953` `[None][infra] Waive N failed cases for main in pre-merge`：pre-merge 测试豁免（infra，低信号）。

**证据等级：** 官方一手（提交 API + 标题）；`#20889`/`#20959`/`#20960`/`#20953` 仅标题级。

---

## 5. llama.cpp：10-25 提交（20 条）——DSV4.1 flash、CUDA FFN 融合、量化/GGUF 静默损坏修复

**主题**  llama.cpp 在 2026-10-25 UTC 窗口合并 **20 条**提交（周日量），主线为 **DSV4.1 flash 模型支持**（#31086）、**CUDA FFN 融合性能**（#29948）、以及一组 **量化 / GGUF 静默损坏与资源耗尽修复**（#30820/#31044/#31035）。

### 5.1 `#29948` CUDA：FFN gate/up 融合 GLU 扩展到 MMQ 的 prompt-processing（性能）

**主题**  `CUDA: fuse FFN gate/up matmuls and GLU in MMQ with interleaved gate/up rows`（`#29948`）。

**内容详情（取自 PR 正文）**  把现有仅覆盖小批（MMVQ/MMVF）的 gate/up + GLU 融合扩展到 MMQ 路径的 prompt-processing：保持单累加器、在 write-back 应用 GLU（替代为 gate 另起累加 tile）。loader 以 8 为块交错 up/gate 行，使每线程持同一输出元素的 up/gate 值；MMA 循环/瓦片/`sum[]` 不变，write-back 从寄存器应用 GLU 并每 tile 写 64 输出行。支持 Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS 于 NVIDIA Turing+；SWIGLU/GEGLU/SWIGLU_OAI/SWIGLU_CLAMP。

**性能（DGX Spark GB10，`llama-bench -ngl 999 -fa on -p 16384 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -r 5`，tok/s）：**

| 模型 | 类型 | master | PR | 提升 |
|---|---|---:|---:|---:|
| Qwen3.8-27B | Q4_K_M | 840.1 | 899.9 | +7.1% |
| | Q5_K_M | 840.1 | 880.6 | +4.8% |
| | Q6_K | 751.2 | 785.8 | +4.6% |
| | Q8_0 | 876.2 | 940.1 | +7.3% |
| | IQ4_XS | 882.4 | 948.0 | +7.4% |
| Qwen3.6-27B | Q4_K_M | 816.6 | 865.6 | +6.0% |
| | Q5_K_M | 800.0 | 851.1 | +6.4% |
| | Q6_K | 720.8 | 770.4 | +6.9% |
| | Q8_0 | 876.9 | 940.2 | +7.2% |
| | IQ4_XS | 882.2 | 948.5 | +7.5% |

**应用场景**  llama.cpp CUDA 后端 Qwen3.8/3.6-27B 等稠密 FFN 的 prompt-processing 吞吐。

**研判**
- **事实：** 有 GB10 逐卡 bench 全表（Q4_K~IQ4_XS 多档 + 两模型），+4.6~7.5%。
- **观点：** 良性融合性能，NVIDIA Turing+ 多量化格式通用；属「白拿」收益。作者注明计划验证并启用其它 GGML 后端。

**来源**  https://github.com/ggml-org/llama.cpp/pull/29948 （merge 2026-10-25T09:24:39Z；正文已核验）

### 5.2 `#31044` gguf_merge 拒绝截断的 GGUF 张量数据（静默损坏修复）

**主题**  `tools : reject truncated GGUF tensor data in merge`（`#31044`）。

**内容详情（取自 PR 正文）**  `gguf_merge` 未检查 `f_input.read()` 返回值，截断的分片文件产出**静默损坏的合并结果**。复现：分片截到 50%、merge 退出码 0、**45% 输出字节损坏**。修复 4 行，镜像 split 路径既有检查（#30497 修了 split 路径、漏了 merge 路径）。

**应用场景**  GGUF 分片合并工具链（多文件模型合并）。

**研判**
- **事实：** 有明确复现（截断 50% → 45% 字节损坏、退出码却 0）与最小修复。
- **观点：** 与第 1 节「静默错误」主线同源——工具链静默损坏，用户无报错却得到坏文件；凡做 GGUF 合并应确认已含本修复。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31044 （merge 2026-10-25T09:25:11Z；正文已核验）

### 5.3 `#31035` dry sequence breaker 资源耗尽修复（健壮性）

**主题**  `fix exhausting CPU and MEM caused by dry sequence breaker`（`#31035`，修 `#30833`）。

**内容详情（取自 PR 正文）**  启用 `dry_sequence_breakers` 且 breaker 数量很大（如 10000）时，初始化耗时数分钟、CPU/内存暴涨。根因：重复 detokenize 整个 vocab 数组（`get_overlapping_token_sequences` 每次调用都按 `i` 重复 `vocab.tokenize(str.substr(i))`），且去重检查泄漏/不全。修复：`seen_breakers` 集合去重 + 超 `LLAMA_DRY_MAX_SEQ_BREAKERS` 截断。

**应用场景**  llama.cpp sampler 的 DRY 防重复解码（高 breaker 数配置）。

**研判**
- **事实：** 明确根因（重复 detokenize 全 vocab）+ 修复（去重 + 上限）。
- **观点：** 属资源耗尽/潜在 DoS 式退化，非数值错误；高 DRY 配置用户应确认已含本修复。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31035 （merge 2026-10-25T11:31:09Z；正文已核验）

### 5.4 `#31086` 模型：DSV4.1 flash（新模型，描述级）

**主题**  `model: DSV4.1 flash`（`#31086`）。

**内容详情**  PR 描述**为空**（仅贡献指南与 AI 使用披露模板），未给出实现细节。仅能从标题确认新增 DeepSeek-V4.1 flash 模型支持。

**应用场景**  llama.cpp 对 DeepSeek-V4.1 flash 的模型接入。

**研判**
- **事实：** 仅标题级信息；正文无技术细节，无法核验具体能力/精度/性能。
- **观点：** 新模型接入类；因**描述为空**，按规范记为**官方一手（标题级/描述级）**，不对其能力做任何推断；待后续 PR 或模型卡补充。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31086 （merge 2026-10-25T01:43:51Z；正文为空）

**证据等级：** 官方一手（提交 API + 标题）；描述为空 → 标题级/描述级。

### 5.5 其余 llama.cpp 窗口提交（标题级，正文未逐一抓取）

- `#30820` 见第 1 节（量化修复 + deepseek-v3.1 注意力误判为专家张量）。
- `#31087` `[llama] respect no_alloc in all memory caches`：所有内存缓存尊重 `no_alloc`（bugfix）。
- `#31025` `[ggml-cuda] fuse qwen MRoPE + set-rows`（性能）；`#29920` `[vulkan] larger tile size for conv3d (coopmat2)`；`#30106` `[vulkan] one warp tile per subgroup in small matmul tiles`；`#31071` `[hexagon] generic dims collapse/reshape`；`#31026` `[opencl] add topk and lightning indexer`；`#30999` `[opencl] add initial q2_k support`（初始 q2_k 量化后端，Codex 初始移植后人工重写）；`#31028` `[qwen] merge tensors on convert`；`#31153`/`#29495`/`#29755` `[grammar] recursive depth limit to prevent stack overflow`；`#30242` `[chat] reject JSON nested more than 128 levels`；`#31157`/`#30977`/`#31136` CI/测试作业调整。

**滚动 nightly：** 10-25 共 **18 个 nightly 标签** b11870–b11889（b11872 未落当日窗口），较 10-24 的 b11850–b11869 向前推进。

**证据等级：** 官方一手（提交 API + 标题）；`#30820` 已抓正文，其余为**官方一手（标题级）**。

---

## 6. 社区与视频信号（HN / 社区仓库）

**主题**  HN Algolia 窗口内仅 1 条指向推理引擎的前台故事；其余推理引擎相关查询均为评论级命中。

- **「Show HN: Benchmarking DSpark speculative decoding on a consumer GPU」**（作者 Termich，2026-10-25T17:07:06Z，2 分）：链接 `https://github.com/juraj-jumic/dspark-local-bench`。已核验仓库存在（HTTP 200，1 star，创建 2026-10-24T20:46:34Z），描述为「Does DSpark speculative decoding help on your GPU, for one user? Reproducible batch-size-1 benchmarks for llama.cpp on c[onsumer GPU]」。即对 **SGLang DSpark 投机解码在消费级 GPU（batch-size-1）** 的可复现基准。
  - **研判：** 属社区个人基准仓库（1 star、新创建），对 DSpark 在消费级 GPU 的实际收益提供可复现入口，但**具体性能数字为二手线索/未核验**（非官方、样本小、未跨硬件交叉）。可作自测起点，不应当作普适结论。
  - **来源：** HN `https://news.ycombinator.com/item?id=`（经 Algolia `search_by_date` 命中，标题/URL/作者/时间已核验）；仓库 `https://github.com/juraj-jumic/dspark-local-bench`（HTTP 200 已核验）。**证据等级：二手线索（仓库已存在但内容为个人基准，数字未核验）。**
- `vllm` / `sglang` / `TensorRT-LLM` / `KV cache` / `LLM inference` 在窗口内的其它命中**均为评论级（title=null、points=null）**，无一条是引擎相关前台发布，与既往各期一致——推理引擎在 HN 前台曝光极低。
- **视频 / Reddit / X：** 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

---

## 7. 累计趋势判断（继承 + 本期新增）

**继承（截至 2026-10-25 的各期累计结论）：**
1. **调度器隐式阻塞的定量画像**——优先级/调度路径的隐式退化（本期 vLLM `#61976` 再次印证）。
2. **发版窗口的双向性**——「发版当天合入 main」≠「在发版内」（compare API 已多次证实）。
3. **默认值成为主要风险载体**——`--max-model-len auto`、默认开启的 flag 等改变行为且不报错（本期 `#60513` 自动缩减后 speculator 上限未同步）。
4. **异构后端静默错误高频**——ROCm/Blackwell/Rubin 的 miscompile / 精度 / 同步陷阱（本期 `#60084` 为良性回归回收，非错误）。
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**——三者任一组合易踩、无显眼报错（本期 `#60513` 投机×auto-fit、`#46423` PD KV 校验）。
6. **量化路径静默错误** 为与异构后端并列的第二类高频静默错误（TRT-LLM `#20823` NVFP4_AWQ scale 漏应用、llama.cpp `#31097` MiniMax-M3 MoE scale 未前向）。

**本期新增（第 7 条）：**
7. **专家张量/路由归属与分页漏接静默错误** 是量化静默错误（#6）的近亲且高频：本期 **SGLang `#45787` paged-experts「未声明张量不被 paging → 静默用错专家值（median 1.67 nats）」**、**vLLM `#63372`「DBO 每个 microbatch 从 row 0 写路由、末批覆盖其余 → 路由 0% 正确」**、**llama.cpp `#30820`「deepseek-v3.1 注意力权重被误判为专家张量（应 Q8_0）」** 三处同源——落点是「MoE 专家张量/路由的归属或分页漏接 → 无断言错结果」，与 #4/#6 共构「量化/MoE 静默正确性」主线，且**不依赖特定硬件**。

**趋势判断：** 2026-10-25（周日）四家提交量 vLLM 34 / SGLang 44 / TRT-LLM 7 / llama.cpp 20；TRT-LLM 周末量低符合规律，SGLang 仍高因 MiniMax-M3.1 / Kimi-K3 性能栈 + diffusion 潮在途。跨引擎「静默正确性」议题从「异构后端」（#4）经「量化 scale」（#6）演进到「专家分页/路由」（#7），建议把「与黄金参考逐层对拍 + 专家输出抽样核对」固化为量化/MoE 部署的上线门禁。

---

## 8. 可执行清单 + 数据质量与采集说明

**可执行清单（建议动作）**
- **量化/MoE 部署验收：** 把「与黄金参考逐层对拍 + 专家输出抽样核对」列为上线门禁，覆盖 NVFP4 / MXFP4 / block-FP8 / AWQ / compressed-tensors MoE 与 paged-experts；本期 SGLang `#45787` 提示须确认目标格式张量**均被 paging**（否则静默用错专家值）。
- **vLLM 用户：** 确认含 `#63266`（chunked-local CPU 后缀误命中）、`#61019`（EAGLE/MTP partial tail）、`#63372`（MRV2 路由）、`#62698`（分级写性能）、`#61976`（优先级自抢占）、`#60513`（投机×auto-fit）；ROCm `TRITON_ATTN` 用户确认含 `#60084`（+18% 回归回收）。
- **SGLang 用户：** MiniMax-M3.1 / Kimi-K3 NVFP4 路径关注 `#46139`/`#46140`/`#46141`（**CI 未全绿、AMD 红**，待转绿与接入 PR 再采信数值）；`#46359` DCP block-FP8 q_b_proj（bit-identical，可启用）；PD 部署关注 `#46423` KV checksum（升级须 prefill+decode 同版本，否则窗口内走未校验放行）。
- **TRT-LLM 用户：** `#20679` 内存修复 **CI 失败、全模型精度未重跑**，谨慎评估；`#20213` Qwen3-VL 编码器 CUDA graph 可启用（无量化数字）；`#20948` serve 错误归因可启用。
- **llama.cpp 用户：** `#29948` FFN 融合（GB10 +4.6–7.5%）、`#30820` 量化修复、`#31044` GGUF 静默损坏修复、`#31035` dry breaker 资源修复；关注 nightly **b11870–b11889**；`#31086` DSV4.1 flash 仅标题级，待补细节。
- **社区参考：** DSpark 消费级 GPU 基准（juraj-jumic/dspark-local-bench）可作自测起点，但属二手、须自行复现。

**数据质量与采集说明**
- 官方提交 API（UTC 窗口 2026-10-25）：vLLM **34** / SGLang **44** / TensorRT-LLM **7** / llama.cpp **20**，均 HTTP 200。
- 27 个 PR 正文经 GitHub issue API 抓取（markdown 干净文本）；四家 releases API；HN Algolia 公开 API（HTTPS 200）。
- 版本核验口径延续：发布说明/描述 ≠ 进版；PR 描述 ≠ diff 落地。本期 **SGLang `#46139`/`#46140`/`#46141` CI 未全绿**（Base/Extra 未运行、AMD :x:）按描述级/未完全绿；**llama.cpp `#31086` 描述为空**按标题级；**TRT-LLM `#20679` CI 失败、全模型精度未重跑**按描述级。
- 未采集 Reddit / YouTube / X；未核验到官方视频。所有性能数字均来自 PR 自述，未做独立第三方对拍（已在各条目标注）。
- 本档案与上一期（2026-10-25 档案，覆盖 2026-10-24）连续无空档。
