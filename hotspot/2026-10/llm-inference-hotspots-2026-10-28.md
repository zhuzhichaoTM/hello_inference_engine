# 大模型推理优化每日热点档案

- **档案日期：** 2026-10-28（北京时间）
- **关注窗口：** 2026-10-27 00:00–23:59（UTC 日期过滤，对应北京时间 2026-10-27 08:00 – 2026-10-28 08:00；2026-10-27 为**周二**）
- **范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、MoE、异构硬件与运行时稳定性、社区 KV cache 分发
- **数据质量说明：** 本期确定事实来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API（2026-10-27 UTC 窗口，分别返回 **102 / 84 / 48 / 32** 条，均 HTTP 200；vLLM 102 条为本任务追踪以来单日最高，超过周一的 99）、四家官方 releases API、**21 个 PR 正文**（经 GitHub issue API 抓取，返回 markdown 文本，含 vLLM `#63820`/`#63841`/`#61258`/`#62011`/`#60960`/`#57420`/`#52970`、SGLang `#42973`/`#46528`/`#43815`/`#46497`/`#45474`/`#45624`、TensorRT-LLM `#20940`/`#20942`/`#19369`/`#20759`、llama.cpp `#31230`/`#31009`/`#31172`/`#31243`）。**窗口特征：** 2026-10-27 为**周二（工作日）**，四家提交量延续周一高位（vLLM 102、SGLang 84）；TRT-LLM 48 与 llama.cpp 32 持平近期。HN Algolia 公开 API 可访问（HTTPS 200）；窗口内出现**一条引擎相关社区故事**（「Two RTX 5090 hosts sharing a KV cache over Ethernet」，3 分），其余引擎关键词命中多为「LLM」模糊匹配或 0。Reddit、YouTube、X 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。
- **版本状态：** 窗口内 vLLM / SGLang / TensorRT-LLM **均无新正式版本**（vLLM 仍 **v0.32.0** 2026-10-18；SGLang 仍 **v0.5.22** 2026-10-19；TensorRT-LLM 仍 **v1.4.0rc1** 2026-10-22，prerelease）。llama.cpp 滚动 nightly 共 **b11914–b11938**（窗口内可见 b11934–b11938；b11940 发布于 2026-10-28T00:32:11Z，紧贴 UTC 窗口边界，窗口内不可见）。无新版本事件，故本期头号事件为「**默认值 × 静默正确性双主线延续 + GLM DSA/MTP 接入 KVShard/PD/CPU HiCache**」（见第 1、6、7 节）。
- **本期方法论沿用：** 继续用 GitHub commit API 的 UTC 日期窗口 + PR 正文经 issue API 抓取（干净 markdown）；延续「版本核验/描述级核验」口径——发布说明/描述 ≠「修复已进版」，PR 描述 ≠ diff 落地，CI 状态须纳入证据等级。本期新增：SGLang `#45624`（Flint 新模型）描述为官方空模板、无实现细节，按描述级/标题级记录；vLLM `#61258` 为从社区 fork（ivanium/vllm#53）cherry-pick 并保留作者署名，已注明来源。

---

## 1. 版本状态与本期头号事件：默认值 × 静默正确性双主线延续 + GLM DSA/MTP 接入 KVShard

**主题** 本期无新版本，但出现一组**「默认值触发 / 默认切换」叠加「静默正确性」**的高信号事件：SGLang **#46528**（AMD 上 `--attention-backend aiter` 默认后端 + fp8 KV cache + 校准 scale → **静默产出垃圾、GSM8K 直接掉到 0、无需任何 flag**）是本期最危险的案例——它把累计结论第 3 条（默认值风险）与第 4/6 条（异构后端/量化静默错误）直接耦合；同时 vLLM **#61258**（best-effort NUMA 绑定**默认开**）、SGLang **#43815**（gfx942 AITER CK blockscale GEMM **默认开**）是两条明确的「默认即行为变更」；SGLang **#42973** 把 GLM **DSA/MTP 草稿/索引 KV** 接入 KVShard + PD + CPU HiCache，使 PD-KV 主线从「正确性/安全」进一步走向「草稿与索引 KV 的所有权与一致性」架构级问题。

### 1.1 SGLang `#46528`：AMD fp8 KV cache 默认 aiter 后端静默产出垃圾（GSM8K→0，无需 flag）

**主题**  `[Fix] [AMD] aiter: Fix garbage output with an fp8 KV cache and calibrated KV scales`（`#46528`）。

**内容详情（取自 PR 正文）**  在 `--attention-backend aiter`（ROCm 上 MHA 模型的**默认**后端）且 fp8 KV cache 下，携带校准 KV scale 的 checkpoint 会**静默产出垃圾输出：GSM8K 掉到 0**；**无需任何 flag 即可触发**。根因：KV cache 本身写入正确（`K / k_scale`），但三处 attention 调用点**误用了 scale**：
- **默认 decode（`paged_attention_ragged`，无滑动窗口 MHA 的 decode kernel）：** 传入的是 backend-wide 的 unit scale 而非该层的 k/v scale，导致 cache 从未被反量化；
- **unified-attention decode（`unified_attention`，滑动窗口模型如 gpt-oss、`SGLANG_USE_AITER_UNIFIED_ATTN=1`）：** q 用 `k_scale` 量化（该 scale 是为 K 校准的，不是 q）；
- **NHD prefill（`mha_batch_prefill_func`，gfx942，gfx950 上 HD128 ASM 路径）：** 同样误用 scale。

触发路径：默认 `--kv-cache-dtype auto` 会在 checkpoint 声明量化（compressed-tensors `kv_cache_scheme` / ModelOpt `kv_cache_quant_algo: FP8`）时自动开启 fp8 KV 并载入 scale；未声明的 checkpoint（如 `amd/Llama-3.1-8B-Instruct-FP8-KV`、`amd/gpt-oss-20b-...-FP8-KV-FP8`）在显式 `--kv-cache-dtype fp8_e4m3` 时同样命中。

**应用场景**  SGLang 在 AMD ROCm 上对带 fp8 KV cache 的模型（Llama-3.2-1B-FP8-KV 等）的 serving。

**研判**
- **事实：** 有完整的根因分解（三处误用 scale）+ 触发条件（默认 aiter + fp8 KV auto-on）+ 量化后果（GSM8K→0）+ 复现 checkpoint 清单；属一手细节。
- **观点：** 这是本任务追踪以来**最典型的「默认值 × 静默正确性」耦合案例**——默认后端 + 默认 KV 量化自动开启，**既不报错也不告警，直接把精度打到 0**。它把累计结论第 3 条（默认值风险）与第 4/6 条（异构后端/量化静默错误）合流，建议 AMD 部署方在启用 fp8 KV + aiter 后端时**强制跑 GSM8K 级回归**，并在升级后 diff 默认项；列为本期头号事件。

**来源**  https://github.com/sgl-project/sglang/pull/46528 （merge 2026-10-27T 窗口内；正文已核验，含复现 checkpoint 与三处根因）

### 1.2 默认值风险两条明确「默认即行为变更」

**主题**  vLLM `#61258`（best-effort NUMA 绑定默认开）/ SGLang `#43815`（gfx942 AITER CK blockscale GEMM 默认开）。

**内容详情（取自 PR 正文）**
- **vLLM `#61258`** `[Feature][Core] Enable best-effort NUMA binding by default`：对符合条件的 GPU 多进程启动（含 PCP）**默认开启 best-effort NUMA 绑定**，自动模式优先 GPU-local 内存；显式 `--numa-bind`/绑定列表保留严格策略；失败的自动探测回退到 CPU-only 再 unbound。该 PR 从社区 fork `ivanium/vllm#53` cherry-pick（保留作者署名与 sign-off），并合入共享 GPU 索引修复 `#61237`。影响：升级后 eligible 启动的 NUMA 内存策略**静默改变**（通常有益，但会改变 NUMA 拓扑假设）。
- **SGLang `#43815`** `[Perf][AMD] qwen3_5: Use the AITER CK blockscale GEMM by default on gfx942`（延续 `#41071` 的 opt-in→默认）：`SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK` 现**默认 `true`**（`=0` 保留 Triton GEMM）。依据：`#43526` 追踪到 Qwen3-32B-FP8 在 5-shot/512-token 下的 GSM8K 掉点主要来自评测设置（答案被截断、小数值变化翻转），8-shot 下与 Triton 持平、MMLU 持平；CK 在 gfx942 上提供最高 **4× decode 吞吐**。

**应用场景**  vLLM 多 GPU/多进程（含 PCP）启动的 NUMA 内存策略；SGLang 在 gfx942（MI325X 等）上 Qwen3-32B 级 block-FP8 稠密线性的默认 GEMM 后端。

**研判**
- **事实：** 两 PR 均给出「默认变更 + 依据（NUMA 探测/CK 与 Triton 精度持平）」；`#43815` 有 8×MI325X 的 GSM8K/MMLU 对照。
- **观点：** 二者共同坐实累计结论第 3 条——**「默认值」仍是推理引擎最主要的风险/收益载体之一**，且「默认开」多为良性优化，但升级后必须 diff 默认项。建议 serving 方把「升级后 diff 默认项 + 对应回归」列为发布门禁（已在 10-26 档案第 1、7 节提出，本期再添两条铁证）。

**来源**  https://github.com/vllm-project/vllm/pull/61258 （merge 2026-10-27T 窗口内；正文已核验，含 ivanium/vllm#53 来源声明）、`https://github.com/sgl-project/sglang/pull/43815` （merge 2026-10-27T 窗口内；正文已核验）

### 1.3 SGLang `#42973`：GLM DSA/MTP 接入 KVShard + PD + CPU HiCache（草稿/索引 KV 所有权一致性）

**主题**  `[Feat] KVShard: Integrate GLM DSA/MTP with PD and CPU HiCache`（`#42973`）。

**内容详情（取自 PR 正文）**  把 GLM 的 **DSA/MTP serving** 接入 page-sharded KV cache，使 prefill、P/D 传输与 CPU HiCache 对 **target/draft 的 latent KV 与 DSA indexer 的 keys/scales** 保持**同一所有权**。改动：选择 page-sharded 的 DSA target 与 single-layer GLM NextN draft pools；校验匹配的 cache geometry 与 budget latent/indexer scratch；保留共享 sender 生命周期；在 owner 配对前配对 source/destination DSA pages。依赖 `#43778`（已 rebase 到本 PR 的 main base）、DSA attention/indexer 支持 `#44280`、MTP pool/scratch `#40929` 等已合入项；Rust 支持在独立 `#46445`。

**应用场景**  SGLang 中 GLM DSA（稀疏注意力）+ MTP 投机 + PD 分离 + CPU HiCache 的联合部署，尤其草稿 latent KV 与 DSA 索引 KV 的跨 prefill/decode/分层缓存一致性。

**研判**
- **事实：** 明确的「目标/draft latent KV + indexer keys/scales 同一所有权」设计 + 依赖清单 + Rust 拆分。
- **观点：** 与累计结论第 5 条（前缀缓存 × MoE 路由 × 投机解码交叉债务）及往期 PD-KV 主线同源——投机解码的**草稿 KV 与索引 KV 现在是 PD/分层缓存的一等公民**，其所有权/一致性若处理不当会静默错配。本期记为「官方一手（功能 PR，依赖未全绿）」；部署方应关注 `#43778` 是否随本 PR 一同合入。

**来源**  https://github.com/sgl-project/sglang/pull/42973 （merge 2026-10-27T 窗口内；正文已核验）

**证据等级（第 1 节）：** 官方一手（提交 API + PR 正文）。`#46528` 为带完整根因与复现的 silent-error 一手；`#61258`/`#43815` 为默认变更一手（含依据）；`#42973` 为功能 PR 一手（含依赖 `#43778` 未全绿提示）。

---

## 2. vLLM：10-27 提交（102 条，单日最高）——TMA MoE 性能、NUMA 默认、DSv4.1 稀疏 logits、前缀缓存评分、MiniMax-M3 稀疏 prefill、RDNA3 aiter

**主题**  vLLM 在 2026-10-27 UTC 窗口合并 **102 条**提交（本任务追踪以来单日最高，超过周一 99），主线为 **MoE/GLM-5.3 性能**（#63820 TMA up/down proj、#63841 DSpark fp8 KV 省显存）、**默认/运行时**（#61258 NUMA 默认、#63778 MRV2 runner 决定 graph 内存 profiling）、**DSv4.1 稀疏 logits**（#62011）、**前缀缓存正确性**（#60960）、以及 **AMD/ROCm**（#52970 RDNA3 aiter 官方支持、#61863/#63409/#63735/#60110 KDA/MLA 修复与性能）。

### 2.1 `#63820` TMA 用于 MoE up/down proj：GLM-5.3 H200 E2E +3.1~3.4%

**主题**  `[Perf] Use TMA for MOE up/down proj, 3.1~3.4% E2E Throughput Improvement for GLM5.3`（`#63820`）。

**内容详情（取自 PR 正文）**  把 MoE 的 up/down projection 改用 TMA。测试在 H200、GLM-5.3、TP8、MTP=5、max-num-seqs 48、prefix-caching 开。guidellm 并发 1/4/16 的 P50 吞吐对照：并发 4 当前 P50 吞吐未给出完整表（正文截断于「297.7 tok/s |」），但标题与概览明确 E2E **+3.1~3.4%**。

**应用场景**  GLM-5.3（及同类 MoE）在 Hopper 上的 up/down proj 吞吐。

**研判**
- **事实：** 有 H200/TP8/MTP5 的 guidellm 测试设置与「+3.1~3.4% E2E」声明；正文在吞吐明细处截断，记为**官方一手但并发 4/16 具体分值未完整呈现**。
- **观点：** 良性 MoE 性能（TMA 搬运），属「白拿」收益；建议 GLM-5.3 Hopper 部署方确认已含。

**来源**  https://github.com/vllm-project/vllm/pull/63820 （merge 2026-10-27T 窗口内；正文已核验，吞吐明细截断）

### 2.2 `#63841` DSpark fp8 KV cache 用于 draft 模型：每 rank 省 5.38 GB

**主题**  `[GLM5.3 Perf] Enable dspark fp8 kv cache for draft model, 5.38 GB GPU Memory saved per rank`（`#63841`）。

**内容详情（取自 PR 正文）**  为 DSpark 草稿模型（RedHatAI/GLM-5.3-speculator.dspark，TP4）的 3 层 draft attention 启用 fp8 KV cache：每层每 token 每 GPU 存 6,144 K/V 值，BF16 为 12 KiB、FP8 为 6 KiB。按 8,192 缓存 token 计，每 GPU 省 48 MiB；按既有 FP8 DSpark 实测（940,736 KV 槽、47.18 GiB/GPU）反推，BF16 draft 需约 52.56 GiB/GPU，即**每 rank 省 5.38 GiB、TP4 共省 21.53 GiB**，固定 cache 预算下约 **+11.4% token 槽**；性能几乎不变，精度（aime25）不受影响。

**应用场景**  GLM-5.3 DSpark 投机解码的草稿模型 KV 显存。

**研判**
- **事实：** 有 BF16 vs FP8 字节级测算 + TP4 槽数反推 + aime25 精度声明。
- **观点：** 良性显存优化（草稿 KV 量化，精度近无损）；属「默认配置下白捡」，建议 DSpark 部署方确认已含。

**来源**  https://github.com/vllm-project/vllm/pull/63841 （merge 2026-10-27T 窗口内；正文已核验）

### 2.3 `#62011` DeepGEMM 稀疏 MQA logits 接入 FP8 indexer cache（DSv4.1，不再需 MXFP4）

**主题**  `[Perf][DSv4.1] Enable DeepGEMM sparse MQA logits on the FP8 indexer cache, DSv4.1`（`#62011`，扩展 `#56254`）。

**内容详情（取自 PR 正文）**  `indexer_sparse_logits` 让 DSv4.1 的 candidate-consuming indexer 层用 DeepGEMM 稀疏 MQA-logits kernel 只给候选 block 打分，而非对整段 context 算 dense logits。此前该 flag 要求 MXFP4 indexer cache（默认 FP8 时启动报 `ValueError`）；自 `#61149` 起 SM100 FP8 indexer cache 已按 UE8M0 四字节对齐存储 power-of-two scale，正是 DeepGEMM 期望的 MXFP8 布局。本 PR 把 FP8 cache 接入既有稀疏路径，**decode 加速不再需要把 K cache 降到 4 bit**。结果：DeepSeek-V4.1-Flash、TP4、4×B300、fp8 KV、greedy，decode/prefill 吞吐由 wall-time 对输出长度的线性拟合得到（正文截断于「Decode and prefill throughput from a linear fit…」）。

**应用场景**  DSv4.1-Flash 在 SM100 的 indexer 稀疏 logits（FP8 KV 下）。

**研判**
- **事实：** 明确的「MXFP4 限制来源（#61149 前 FP8 存 fp32 scale）+ FP8 现已满足 DeepGEMM 布局」论证 + TP4/B300 测试设置；吞吐数值正文截断，记为**官方一手但 decode/prefill 具体 % 未完整呈现**。
- **观点：** 良性（FP8 下即享稀疏 logits 加速、不牺牲 K cache 精度）；DSv4.1 SM100 部署方应确认已含。

**来源**  https://github.com/vllm-project/vllm/pull/62011 （merge 2026-10-27T 窗口内；正文已核验，吞吐数值截断）

### 2.4 `#60960` 前缀缓存复用按首个被请求 prompt score 封顶（评分正确性）

**主题**  `[Bugfix] Bound prefix cache reuse by the first requested prompt score`（`#60960`）。

**内容详情（取自 PR 正文）**  在 `prompt_logprob_token_ids` 且 `skip_reading_prefix_cache=False` 时，cache hit 会跳过被请求的 score；本 PR 把前缀复用封顶到 `prompt_logprob_start`，使这些行被重算。验证：10 个定向用例 + Qwen3-0.6B 测试 + pre-commit 通过；batch-invariant score 精确匹配（含单 host NIXL 检查）；**默认 BF16 返回完整 score 但差异可达 0.29844，根因未解**（仅验证 score 完整性与一致性，非 MMLU 精度）。RTX 5090、PyTorch 2.13 复现：`skip_cache=False, cached=32, first_scored_row=27` 缺失 score 的断言在 main 触发、候选修复后跨 8 组 cache-on/off 精确匹配。

**应用场景**  vLLM 开启 prompt logprob 评分时的前缀缓存复用正确性。

**研判**
- **事实：** 有明确的断言复现（main 失败/候选修复后通过）+ 10 用例 + batch-invariant 精确匹配；但默认 BF16 0.29844 差异根因未解，记为**官方一手（正确性修复，残留数值差异未闭环）**。
- **观点：** 属前缀缓存 × 评分交叉的正确性（累计结论第 5 条外延）；做 prompt logprob 服务的部署方应关注本修复与残留差异。

**来源**  https://github.com/vllm-project/vllm/pull/60960 （merge 2026-10-27T 窗口内；正文已核验）

### 2.5 `#57420` MiniMax-M3 查询分块稀疏 prefill（Hopper 自动选）

**主题**  `[Kernel][MiniMax-M3] Add Triton query-tiled sparse prefill tuned on Hopper`（`#57420`）。

**内容详情（取自 PR 正文）**  MiniMax-M3 稀疏 prefill 原每 query token 起一个 Triton program；相邻 query 常选重叠稀疏 KV block。本 PR 加 opt-in query-tiled kernel：相邻 query 归一组、枚举去重后的并集 block、每块只加载一次、按每 query 成员资格与因果性门控。Hopper 自动选 tiled kernel；其它架构仍用标量 kernel。decode kernel 现在解引用 block table 前校验 persistent top-k block id，使混合 chunked-prefill/prefix-reuse batch 安全。与 `#52963`（ROCm/gfx950）、`#55372`（query-tiled verify-decode）不重复。AI 辅助实现、人工审 diff 后提交。

**应用场景**  MiniMax-M3 在 Hopper 上的稀疏 prefill 性能（重叠 block 复用）。

**研判**
- **事实：** 明确的 tiled 算法 + Hopper 自动选择 + decode 安全性校验 + H100 Buildkite 正确性测试。
- **观点：** 良性稀疏 prefill 性能（opt-in/自动）；MiniMax-M3 Hopper 部署方可获益，注意为 opt-in（非默认）。

**来源**  https://github.com/vllm-project/vllm/pull/57420 （merge 2026-10-27T 窗口内；正文已核验）

### 2.6 `#52970` 官方支持 RDNA3 的 AITER Triton kernel + dsv4（AMD 平价）

**主题**  `[ROCm] DeepSeek-V4-Pro PD Disaggregation through MORI IO KV Connector on AMD GPUs` 同窗口另有 `#52970` `[ROCm] officially supporting aiter Triton kernels+dsv4 on RDNA3`（`#52970`）。

**内容详情（取自 PR 正文）**  在 **RDNA3** 启用 AITER Triton FP8 路径（与 gfx12/RDNA4 平价），并完成 **dsv4** 模型的完整 GPU kernel 支持。改动：`VLLM_USE_AITER=1` 时启用 AITER unified attention（优先级在 `ROCM_ATTN` 之后、`TRITON_ATTN` 之前）；启用 AITER GDN kernel；把多数 gate 从仅 RDNA4 放宽到全部 RDNA。关系：`#43615` 的 RDNA3 镜像/延续；补充 `#47017`（让 dsv4 在 gfx11 能跑但依赖 CPU/torch fallback）；撤销 `#51555`。

**应用场景**  vLLM 在 AMD RDNA3（如 RX 7900 系）上的 AITER Triton 与 DSV4 模型支持。

**研判**
- **事实：** 明确的 gate 放宽（gfx11+gfx12 用 `on_rdna()`）+ 与 `#43615`/`#47017` 的依赖/补充关系。
- **观点：** 良性 AMD 后端覆盖（RDNA3 与 RDNA4 平价 + DSV4 完整 kernel）；AMD RDNA3 部署方受益，仍受 `VLLM_USE_AITER=1` 开关控制。

**来源**  https://github.com/vllm-project/vllm/pull/52970 （merge 2026-10-27T 窗口内；正文已核验）

### 2.7 其余 vLLM 窗口提交（标题级，正文未逐一抓取）

- **量化/MoE 性能：** `#62011` 见 2.3、`#63820` 见 2.1、`#63841` 见 2.2、`#63889` `[Perf][KV Offload] Reduce KVCR adapter overhead`、`#61863` `[ROCm][Bugfix][Kimi-K3] Pass zero positions to the fused MLA decode q-prep kernel`、`#63409` `[ROCm][Bugfix][Kimi-K3] Pass zero positions…`（与 `#61863` 同源修复）、`#63735` `[Bugfix][Perf][ROCm] RDNA launch config for kernel_paged_attention_2d`、`#60110` `[ROCm][Perf][Kimi-K3] Fold q_b_proj and W_UK into one GEMM`、`#53166` `[Perf][ROCm][Attention] Fuse Kimi-K3 MLA chunked-context gather on AITER`、`#59668` `[ROCm][DSV4.1][Perf] Single-launch DSA decode candidate mask`、`#63342` `[ROCm][Perf][Kimi-K3] Use block_m=128 for the fused KDA prefill conv1d`、`#60154` `[Perf][ROCm][DSV4/4.1] Fuse inverse RoPE and MXFP8 quant into the aiter sparse MLA`、`#61903` `[Perf][Kernel] Add a CuteDSL BF16x3 router GEMM for fp32 router weights on Hopper`、`#63894` `[Bugfix][LoRA] Avoid double rounding in fused MoE expand`、`#63864` `[Perf] Tune the Triton packed GDN decode tile on SM90`、`#62133` `[Kernel][MiniMax-M3] Add Triton query-tiled sparse prefill tuned on Hopper`（即 `#57420`，见 2.5）、`#57420` 见 2.5。
- **默认/运行时/调度：** `#61258` 见 1.2、`#63778` `[MRV2] Let the runner decide whether to profile CUDA graph memory`、`#63548` `[Bugfix] Keep MRV2 model state in the runtime pool under sleep mode`、`#63785` `[Perf] Drop BasevLLMParameter's Python __torch_function__`、`#60514` `[Bugfix] Fall back from DeepGEMM and FlashInfer sampling when nvcc is missing`、`#55112` `[Bugfix][Core] Copy every overlaid KV view in copy_kv_cache_blocks_inplace`、`#63427` `[Bugfix][CUDA Graph] Weak-ref nested outputs of captured graphs`、`#63853` `[Bugfix][Spec Decode] Apply the draft max_model_len limit before model config hooks`、`#61557` `[Bugfix][Spec Decode] Preserve grammar-invalid drafts for rejection sampling`、`#53104` `[Bugfix][Spec Decode] Fix eagle draft sharing with a quantized lm_head`、`#63996` `[Model][Spec Decode] Refactor #41834`。
- **KV Connector/Offload/PD：** `#60469` `[KV Offload] Align KVCR cache recency with request prefixes`、`#62317` `[KV Connector][NIXL] Read only a group's own regions in hybrid SSM transfers`、`#62478` `[Perf][KV Connector][Mooncake] Vectorize transfer descriptor build`、`#48989` `[ROCm] DeepSeek-V4-Pro PD Disaggregation through MORI IO KV Connector on AMD GPUs`、`#63064` `[Docs][KV Connector] Document standalone Mooncake EGM deployment`、`#63873` `[Bugfix][Bench] Handle packed hybrid KV and speculative recurrent state`、`#58744` `[Bugfix][MLA] Gather nvfp4_ds_mla chunked-context KV with the NVFP4 kernel`、`#62592` `[Feature][HiSparse] Add a platform backend for device operations`、`#61543` `[Bugfix][HiSparse][KV Connector] Fix DecodeBenchConnector crash on bounded KV cache groups`。
- **模型/前端/文档：** `#64114`/`#64108` MiniMax-M3/M3.1-Flash GB200/B300 AgentX presets、`#62318` `Qwen3.8-Flash-Next: load checkpoints with quantized q/k/v and a BF16 QSA indexer`、`#63332` `[ROCm][Recipe] Qwen3.8-Flash-Next: add MI355X Quark MXFP4 variant`、`#63920` `[Model] Add JIT warmup for Muse Spark NVFP4 MegaMoE`、`#62318` 见上、`#64032`/`#63433` 多媒体 400 错误处理、`#63274` 移除 Qwen3-ASR 的 `--hf-overrides` 必要性、`#63709`/`#63707`/`#62312`/`#61877`/`#63292`/`#63789`/`#63762`/`#63768`/`#63763`/`#63764`/`#63765`/`#63766`/`#63767`/`#63771`/`#63772`/`#60856`/`#51109` Rust Frontend 系列、`#63812` Mistral response_format、`#62570` checkpoint 生命周期 HTTP 准备、`#64085` revert NIXL 1.5.1 bump、`#63817` NIXL 升 1.5.1（被 revert）。
- **正确性/CI：** `#63989` `[Bugfix][Kimi-K3] Reserve KDA prefill workspace before profiling`、`#60960` 见 2.4、`#61557` 见上、`#64081` `[CI][Test] Adjust TRT-LLM and GDN numerical tolerances for Rubin`、`#64107` `[CI] Fix stale NVFP4 decode test selector on DGX Spark`、`#63973` `[CI][CPU] Initialize DFlash test FC after loading weights`、`#61832`/`#62120`/`#62537`/`#61904`/`#63907`/`#63850`/`#62141` CI 自动分片/AMD 镜像、`#60991` `Fuse GLM-5.3 KDA q|k|v convs into one conv1d`。

**证据等级：** 官方一手（提交 API + 标题）；除 2.1–2.6、1.1–1.3 已抓正文者外，其余为**官方一手（标题级）**；`#63820`/`#62011` 正文在吞吐明细处截断，记为官方一手但部分数字未完整呈现。

---

## 3. SGLang：10-27 提交（84 条）——#46528 AMD fp8 KV 静默错误、#42973 GLM DSA/MTP KVShard、#43815 AITER 默认、#46497 EPLB 融合、#45474 Qwen3.5 MTP、#45624 Flint

**主题**  SGLang 在 2026-10-27 UTC 窗口合并 **84 条**提交（周二量偏高），主线为 **AMD fp8 KV 静默错误修复**（#46528，见第 1.1 节，本期头号）、**GLM DSA/MTP 接入 KVShard+PD+CPU HiCache**（#42973，见第 1.3 节）、**gfx942 AITER CK blockscale 默认开**（#43815，见第 1.2 节）、**EPLB expert-count 融合**（#46497）、**Qwen3.5 MTP host work 入图**（#45474）、**新模型 Flint**（#45624）。

### 3.1 `#46497` 融合 expert-count 记录 kernel（EPLB 在 MegaMoE decode 的固定成本消除）

**主题**  `[Perf] eplb: Fuse expert-count recording into one kernel on CUDA decode graphs`（`#46497`）。

**内容详情（取自 PR 正文）**  `--enable-eplb` 下每个 decode step 都要付固定的 expert-distribution 记录 + logical→physical remap 成本，足以抵消 MegaMoE decode 的均衡收益。`_SelectExpertsSinglePassGatherer.on_select_experts` 用约 8 个小 elementwise op 建每层 count；`_on_hook` 在流捕获时记录，使这些全部进 decode CUDA graph 并在每次重放运行（即使记录已停）。GB300 实测 40 MoE 层单 CUDA graph、每 rank 20–120 token：仅 recorder hook（MegaGate 内核内 map）额外 **+570~+720 us**，加 torch static remap 后 **+763~+1069 us**。DSv4.1-Flash DEP32 decode（6×DEP2 prefill）动态 EPLB（32 冗余专家）在早前 nightly 无 decode-step 收益（最忙 rank 1.03–1.07×均值，trivial 为 4.2×）；匹配 batch 比较：4 running req/rank 时 +5.6% step time、24 时 −5.5%。本 PR 把记录融合进一个 kernel。

**应用场景**  SGLang 在 GPU 上 `--enable-eplb` 的 MegaMoE decode step 成本。

**研判**
- **事实：** 有明确 GB300 微基准全表（+570~1069 us）+ 动态 EPLB 无收益证据 + 匹配 batch 增益/损失对照。
- **观点：** 良性 EPLB 性能（消除固定开销、保住均衡收益）；EPLB MegaMoE decode 部署方应确认已含。

**来源**  https://github.com/sgl-project/sglang/pull/46497 （merge 2026-10-27T 窗口内；正文已核验）

### 3.2 `#45474` Qwen3.5 MTP host work 移入 CUDA 图（GB300 低延迟 decode）

**主题**  `[Perf][NVIDIA] speculative: Move the host work between Qwen3.5 MTP decode graphs into the graphs (1/2)`（`#45474`）。

**内容详情（取自 PR 正文）**  Qwen3.5-397B-A17B（NVFP4）在 GB300、TP8、MTP（top-k 1，7 draft token）低延迟 decode，配合 NVIDIA megakernel 插件后 GPU 部分足够短，host work 成瓶颈：图间输入 staging、verify 的 eager sampling 与 commit、scheduler 每 pass 的 TP sync。本 PR 把这些工作移入 CUDA 图、合并 launch 或移出关键路径（Qwen3.5 对应 `#45450`；插件 `sgl-nvidia-mega#20`）。**每个新路径都默认关**（env switch 默认 off）；main 的 Qwen3.5 AgentX recipes 已设 6 个 switch 为 1。修一处（Post-verify CUDA graphs，`SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH`，≈78 TPS/User）即把 EAGLE verify 的 sampling/acceptance/commit 重放为 CUDA 图、mamba state commit 入图。无 megakernel 代码，独立于接线 PR `#45266`。

**应用场景**  Qwen3.5 NVFP4 在 GB300 的低延迟 MTP decode（host work 消除）。

**研判**
- **事实：** 明确的「host work 成瓶颈」根因 + 每路径默认关 + 与 `#45266`/插件关系 + 单路径 ≈78 TPS/User 量化。
- **观点：** 良性 MTP 性能（host 工作入图，默认关、可控）；Qwen3.5 GB300 低延迟部署方按需开启，注意为显式 flag。

**来源**  https://github.com/sgl-project/sglang/pull/45474 （merge 2026-10-27T 窗口内；正文已核验）

### 3.3 `#44797` AMD MoE topk 跨 aiter 路由路径保留 fp32 bias（正确性）

**主题**  `[Fix][AMD] moe/topk: Preserve fp32 bias across aiter routing paths`（`#44797`）。

**内容详情（取自 PR 正文）**  在 AMD aiter 路由路径上，MoE topk 的 fp32 bias 未被保留（跨 aiter routing 路径丢失）；本 PR 修正以保留 fp32 bias。

**应用场景**  SGLang 在 AMD aiter 路径下的 MoE 路由 bias 正确性。

**研判**
- **事实：** 明确的「aiter 路由路径丢失 fp32 bias」根因 + 修正。
- **观点：** 属 AMD MoE 路由正确性（与第 4/6 条异构/量化静默错误同源）；AMD MoE 部署方应确认已含。记为**官方一手（标题级+根因，正文较简）**。

**来源**  https://github.com/sgl-project/sglang/pull/44797 （merge 2026-10-27T 窗口内；正文已核验，较简）

### 3.4 `#45624` 新增 Flint（RecursiveSI/flint）模型支持（描述级）

**主题**  `[Feat] model: Add Flint (RecursiveSI/flint) support`（`#45624`）。

**内容详情**  PR 描述为官方空模板（仅 PR 类型/范围说明与 Checklist，Motivation/Modifications/Accuracy/Speed 全空），未给实现细节或精度/性能数据。仅能从标题确认新增 Flint（RecursiveSI/flint）模型支持。

**应用场景**  SGLang 对 Flint（RecursiveSI/flint）模型的接入。

**研判**
- **事实：** 仅标题级信息；描述为官方空模板，无法核验具体能力/精度/性能。
- **观点：** 新模型支持类；因**描述为官方模板、无实现细节**，按规范记为**官方一手（描述级/标题级）**，不对其内容做任何推断；待后续 PR 或模型卡补充。

**来源**  https://github.com/sgl-project/sglang/pull/45624 （merge 2026-10-27T 窗口内；描述为官方空模板）

### 3.5 其余 SGLang 窗口提交（标题级，正文未逐一抓取）

- **DSv4.1/Engram/PD：** `#46896`/`#46893` dsv41 Engram docs 与 `--engram-config` 选择 Engram 表位置、`#46891` dsv41 DSpark draft block metadata 不依赖 host seq lens、`#46718` dsv4 tail source KV layer id in capture stub、`#41275` `deepseek-v4.1: Match KV capacity to physical layouts`、`#46700` `mem_cache: Clamp SWA tombstone recovery to protected prefix`、`#46706` dsv4 渲染 null/trailing tool-call/numeric string efforts、`#43271` `DSA/KDA: Handle MoE padding in hybrid MTP verification`、`#46792`/`#46719` bcg capture refactor、`#42973` 见 1.3。
- **AMD/量化/性能：** `#46528` 见 1.1、`#43815` 见 1.2、`#46497` 见 3.1、`#44797` 见 3.4、`#45555` `[Perf][AMD] vattn-asm: Split small-batch MTP verify attention over all CUs on gfx950`、`#43839` `[Perf][AMD] qwen3_5: Quantize the BF16 GDN projections to MXFP4 online`、`#45753` `[Perf] QSA: Remove the D2D copy in prefill generated kv gather`、`#45739` `[Perf][AMD] Qwen3-Next: fused QK Gemma RMSNorm + partial RoPE on gfx950`、`#46366` `[Perf][AMD] diffusion: Route large BF16 linears to FlyDSL a16w16 on gfx1250`、`#45576` `[Perf][AMD] rccl: upgrade rccl to 2.32.3`、`#46528` 见 1.1、`#44778` `kda: Use raw beta in prefill CUDA graphs`、`#46093` `[Perf] moe: Always pass the global token count to the Mega MoE tile heuristic`、`#46457` `[Perf] paged-experts: fuse slot lookup into decide`、`#46567` `[Perf] paged-experts: fewer host syncs`、`#45363` `[Fix] eplb: Load all local replicas of each expert`、`#43978` `[Perf] hicache: Parallel THP-backed host pool allocation (opt-in)`。
- **HiCache/PD/调度：** `#42973` 见 1.3、`#46689` `[Perf] disaggregation: Restore host-received decode KV without blocking the scheduler`、`#45678` `[Fix] disaggregation: Avoid extra GPU contexts with PD`、`#44081` `[Fix] PD: Skip the cached-prefix early send in batches that start a HiCache load-back`、`#46674` `[Perf] radix-tree: Keep the GIL for tree-core reads unless busy`、`#31286` `[Refactor] HiCache: Remove unused cache_ttl_second`、`#46620` startup remove TorchScript deprecations、`#43450` `[Feat] scheduler: Add forward observer and deferred responses for plugins`。
- **diffusion：** `#45638`/`#46600`/`#45833`/`#44888`/`#46536`/`#46659`/`#46660`/`#46144`/`#46252`/`#46155`/`#46679`/`#46638` diffusion 各模型/管线/CLI 修复与重构（仍占当日相当比例）。
- **模型/文档/CI/其它：** `#45624` 见 3.4、`#46879` rust-server info 端点、`#46853` custom-all-reduce remove multi-node VMM allocator gate、`#46807`/`#46859`/`#46886`/`#46895` cookbook/doc、`#45858` docker torch 2.15rc1、`#43989` Intel oneAPI 升级、`#46844`/`#46774`/`#46686`/`#46785`/`#46708`/`#46510`/`#45735` NPU/AMD/CI、`#42760` model_loader avoid unsafe pickle、`#44124` qwen3_coder structured values、`#44487` nemotron nested Puzzle、`#46424` dsa preserve token-first sparse attention、`#46528` 见 1.1、`#32272` `[Perf] fmha_v2: non-interleaved paged KV for trtllm sm90/120 GQA prefill`、`#33743` `[Fix] moe: Fix flashinfer TRT-LLM BF16 expert weight reload on refit`、`#46309` `[Perf] deepep_v2: Reduce masked GEMM buffer memory during CUDA Graph capture`。

**证据等级：** 官方一手（提交 API + 标题）；除 1.1–1.3、3.1–3.4 已抓正文者外，其余为**官方一手（标题级）**；`#45624` 描述为官方空模板 → 描述级/标题级；`#46528` 为带完整根因与复现的一手 silent-error。

---

## 4. TensorRT-LLM：10-27 提交（48 条）——单模型投机解码 DSpark、FP8 block-scale low-M GEMM、V2 KV 连接器 attention DP、Agg Helix KV、KVCMv2 页表、V2 调度

**主题**  TensorRT-LLM 在 2026-10-27 UTC 窗口合并 **48 条**提交（近期高位），主线为 **单模型投机解码 DSpark Markov 链**（#20940）、**FP8 block-scale low-M GEMM 解码投影**（#20942）、**V2 KV 连接器支持 attention DP**（#19369）、**Agg Helix KV cache 第一步**（#20759，默认关）、**KVCMv2 页表在 CC 下物化**（#17658），以及 serving/模型支持与大量 infra/CI。

### 4.1 `#20940` 单模型投机解码：vocab-parallel token selection + 融合 DSpark Markov 链 + spec tail

**主题**  `[None][perf] One-model spec decode: vocab-parallel token selection, fused DSpark Markov chain and spec tail, drafter TP fusion`（`#20940`）。

**内容详情（取自 PR 正文）**  - vocab-parallel selection 为支持的均匀分片 LM head 加 token-selection 路径；不支持的（含自定义 logits processors）保留 gathered-logit 路径。- DFlash/DSpark 加 TP projection 与 reduction 路径（split projection 条件与 unsplit fallback）；DSpark 改 Markov token selection 并把 residual add + norm 与 TP reductions 融合。- DFlash 加 paged-latent 准备与 draft-block placement metadata；speculative metadata 在建 CUDA-graph view 前准备 rejection/penalty buffer。- 公开方法签名改变，`dspark_markov_chain_logits` 被移除（改用共享 Markov-chain 实现）。- 评审关注 drafter 模型与 speculative worker 的耦合（含运行时依赖与重复逻辑），作者认可并留 follow-up contract refactor。- 报告测试适用于整个 PR 系列而非单 PR；speculative 回归测试已有 HF-download 失败。

**应用场景**  TRT-LLM 单模型投机解码（DFlash/DSpark）的 vocab-parallel 选择、TP 融合与 Markov 链加速。

**研判**
- **事实：** 明确的架构改造（vocab-parallel 选择 + DSpark Markov 融合 + TP 融合）+ 签名变更 + 评审遗留耦合 concern（留 follow-up）。
- **观点：** 属投机解码性能/架构主线（与第 5 条交叉债务同源）；签名变更影响调用方，升级方需注意 API 迁移；记为**官方一手（架构 PR，部分测试受 HF 下载失败影响）**。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20940 （merge 2026-10-27T 窗口内；正文已核验，coderabbit 自动生成）

### 4.2 `#20942` FP8 block-scale low-M GEMM 解码投影（SM100/103、SM107 tactic 表）

**主题**  `[None][perf] FP8 block-scale low-M GEMM for decode projections (SM100/103, SM107 tactic tables)`（`#20942`）。

**内容详情（取自 PR 正文）**  `fp8_block_scale_low_m_gemm` 为符合条件的低行 FP8 block-scale GEMM 加 CuTe DSL custom-op 路径，校验输入、返回 BF16、提供 fake 实现。Linear dispatch 用每行权重上限并在范围外回退 `fp8_swap_ab_gemm`；Kimi weight-read linear 准备 scale 并选独立 SM107 路径；fused-projection row views 禁用 low-M dispatch。QA 加了 GPU-gated 单元测试（精度、split-K、mid-M、partial N、long-K、dispatch 上限、不支持输入、CUDA graph capture，含 SM107 有无 Rubin-capable CuTe DSL）。B200 `l0_b200.yml` 加 `TestDeepSeekV3Lite::test_fp8_block_scales`（MTP/attention DP 关、CUDA graph 开、overlap 与 torch.compile 关）。覆盖率 verdict：对所述 kernel 与选择行为**充分**；测试执行结果 unavailable。

**应用场景**  TRT-LLM 在 SM100/103/SM107 上低行（decode 投影）FP8 block-scale GEMM。

**研判**
- **事实：** 明确的 custom-op 路径 + 行上限 dispatch/回退 + SM107 独立路径 + 单元测试覆盖 verdict 充分。
- **观点：** 良性 FP8 解码投影性能（low-M GEMM 专用路径）；SM100/103/SM107 部署方受益，记为**官方一手（perf PR，测试执行结果未提供）**。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20942 （merge 2026-10-27T 窗口内；正文已核验）

### 4.3 `#19369` V2 KV 连接器支持 attention DP（PD/KV 传输一致性）

**主题**  `[TRTLLM-12891][feat] Support attention DP in V2 KV cache connectors`（`#19369`）。

**内容详情（取自 PR 正文）**  在 V2 KV cache connector 支持 attention data parallelism（ADP）：每个 ADP owner 跑自己的 scheduler adapter 与 worker，带本地请求元数据、分配反馈、传输完成与前缀保留身份；adapter 可共享兼容的外部 content pool；无 ADP 的 TP 保留 leader/broadcast。已合入当前 main（含 `#17974`）。接受的 prefix loads 参与 pending-transfer、取消、恢复与 shutdown 记账；completion 在所属 rank 应用；取消-load 响应在对齐 gate 前投递。recovery 路径可 pad 一个 owner 在其最终本地 prefix load 取消且 slot 释放后。draining control 请求统一跳过跨 ADP owner 广播。

**应用场景**  TRT-LLM V2 KV 连接器在 attention-DP 下的 PD/KV 传输、前缀保留与恢复一致性。

**研判**
- **事实：** 明确的 ADP owner 模型 + 共享 content pool + 前缀保留/取消/恢复记账 + 与 `#17974` 集成。
- **观点：** 属 PD-KV 一致性主线（与第 5 条及往期 HiCache/PD 同源）；attention-DP 下的 KV 传输部署方应确认已含。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/19369 （merge 2026-10-27T 窗口内；正文已核验）

### 4.4 `#20759` Agg Helix KV cache 第一步：prefill 分配与 sizing（默认关）

**主题**  `[TRTLLM-17086][feat] Agg Helix KV cache, part 1: prefill allocation and sizing`（`#20759`）。

**内容详情（取自 PR 正文）**  Helix context parallelism 在 aggregated serving 的 KV cache 侧第一步。aggregated serving 中请求现在每个 CP rank 保留完整 prompt（而非按 rank 切分），故 KVCacheManagerV2 可用全局 ledger block 分配 prefill KV。**Agg Helix 默认关**，直到端到端验证。`#17124`：aggregated Helix 请求用 `merge_requests_to_llm_requests` 合并，每 CP rank 持完整 prompt（`prompt_len == total_input_len_cp`）；disaggregated Helix 仍用 `merge_helix_requests`（round-robin 切 block）。`PyExecutor` 传 `is_disagg`。

**应用场景**  TRT-LLM 在 aggregated serving 下运行 Helix context parallelism 的 KV cache 分配。

**研判**
- **事实：** 明确的「每 CP rank 持完整 prompt → 全局 ledger block 分配」设计 + 默认关 + 与 disaggregated Helix 的区分。
- **观点：** 属 Helix/CP 的 KV 架构演进（功能 PR，默认关、未端到端验证）；记为**官方一手（功能 PR，默认关）**，部署方暂不需启用。

**来源**  https://github.com/NVIDIA/TensorRT-LLM/pull/20759 （merge 2026-10-27T 窗口内；正文已核验）

### 4.5 其余 TRT-LLM 窗口提交（标题级，正文未逐一抓取）

- **KV/注意力/调度：** `#20940` 见 4.1、`#20942` 见 4.2、`#19369` 见 4.3、`#20759` 见 4.4、`#17658` `[https://nvbugs/6248648][fix] Materialize KVCM V2 page tables under CC`、`#20772` `[None][fix] Count retained snapshots in hybrid cache sizing`、`#20600` `[None][perf] V2 scheduler: demote long uncached prefills behind short reqs`、`#20803` `[None][fix] Reduce sampler logprobs temporary memory`、`#20008` `[TRTLLM-16320][feat] Add FlashInfer ring-buffer replay for GDN MTP`、`#19575` `[None][fix] Python transceiver: run gen-side KV transfer consensus across helix CP ranks under attention-DP`。
- **模型/serving/多模态：** `#20446` `[None][feat] Support image inputs in OpenEngine gRPC`、`#20932` `[None][perf] Qwen: one-launch vision RoPE and batched mRoPE prep`、`#20904` `[None][fix] test Qwen3.8-Flash-Next QSA layouts and FP8/NVFP4 KV`、`#21055`/`#21051`/`#20873`/`#21025` DeepSeek/DSML tool parsers、`#20866`/`#20881` Anthropic Messages thinking blocks`、`#21061`/`#21056` Responses keepalive/tool calls`、`#20474` disaggregated beam-search docs。
- **正确性/infra/CI：** `#20905` `[None][refactor] Improve child request creation and preserve request inputs`、`#20434`/`#20467`/`#20814` MPI shutdown/heavy import/rank import 修复、`#20996` `[TRTLLM-15765][refactor] Remove the ModelEngine ABC`、`#20894` Component-local C++ Tests Part 4、`#21090` CuTe DSL offline loader CPU test、`#21099` restore default model storage for GH200、`#20884` blob backend with batch registration、`#20979` cherry-pick #20393 save stress logs、`#21117`/`#21110`/`#20696`/`#20891`/`#20439`/`#20933`/`#20827`/`#20845`/`#20799`/`#20829`/`#20288`/`#20227` 各类 waive/pre-merge/修复/精度日期钉死、`#19645` `[TRTLLM-16599][perf] Fuse MiniMax-H3 BF16 QK-norm and rotary embeddings`。

**证据等级：** 官方一手（提交 API + 标题）；除 4.1–4.4 已抓正文者外，其余为**官方一手（标题级）**；`#20759` 默认关、`#20940` 部分测试受 HF 下载失败影响，均已注明。

---

## 5. llama.cpp：10-27 提交（32 条）——glm5-next QSA 去 kq_mask 省显存、Metal TOP_K_MASK、OpenCL q3_k、k-pool 稀疏选择、LiquidAI d1/LFM3 新模型、UI

**主题**  llama.cpp 在 2026-10-27 UTC 窗口合并 **32 条**提交（周二量），主线为 **QSA/DSA 内存优化**（#31009 glm5-next 去 kq_mask 省 2–4 GB VRAM、#31243 k-pool 稀疏选择解耦 kq_mask）、**后端移植**（#31230 Metal `GGML_OP_TOP_K_MASK`、#31172 OpenCL 初版 q3_k、#31239 Vulkan CI llvmpipe 修复）、**新模型**（#31275/#31257/#31252 LiquidAI d1/LFM3 系列）、以及大量 **UI/CI**（#27401 对话压缩、#31212/#31215/#31246/#31276/#31187 等）。

### 5.1 `#31009` glm5-next QSA 去 kq_mask（+DSA）：ub4096 省 2 GB、decode +10~25%

**主题**  `glm5-next : QSA without kq_mask (+ DSA)`（`#31009`）。

**内容详情（取自 PR 正文）**  与 qwen4exp `#30965` 同理，GLM-5 与 DSA 的注意力层是 QSA，无需分配 `kq_mask`；在 ub 4096 省 **2 GB GPU/host 显存**、ub 8192 省 **4 GB**，可让给专家缓存或更大上下文。实测：512k 上下文 + ub 8192 下 GLM-5.3-Flash UD-IQ1_M 在 5090 原本无法运行，现 **651 t/s prefill、12.6 t/s decode**；DSA 同样 VRAM 节省（含 `#30815` 的类似 DSA 改动，仅用切片全模测试、logits 一致）。Compute buffers（`--moe-cache-mib 4000`）：ub4096 CUDA0 6938→4896 MiB、CUDA_Host 2819→771 MiB；ub8192 13874→9784、5636→1540。注释中还提到（被折叠）：GLM-5.3-Flash UD-IQ1_M 去 `kq_mask` 使 decode 更快（ub4096 +10%、ub8192 +25%），小 KV 量化下 prefill +5–7%；ub8192/512k 在 5090 原本“根本不可能”。

**应用场景**  llama.cpp 中 GLM-5/DSA 架构（QSA）的 `kq_mask` 精简与显存/吞吐收益。

**研判**
- **事实：** 有 ub4096/ub8192 的 CUDA0/Host MiB 全表（显存节省量化）+ 5090 651pp/12.6tg 实测 + PPL 一致性声明。
- **观点：** 良性 QSA 内存优化（去冗余 mask，省 2–4 GB、decode 提速）；GLM-5/DSA 部署方应确认已含；本期记为「官方一手（含折叠的 +10%/+25% decode 数据，已在研判引用）」。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31009 （merge 2026-10-27T 窗口内；正文已核验，含折叠性能数据）

### 5.2 `#31230` Metal 移植 `GGML_OP_TOP_K_MASK`（QSA 稀疏注意力掩码）

**主题**  `metal : add GGML_OP_TOP_K_MASK`（`#31230`）。

**内容详情（取自 PR 正文）**  把 `GGML_OP_TOP_K_MASK` 从 CUDA 后端（`#30640`）移植到 Metal。这是融合的 QSA 稀疏注意力掩码：每行中，属 live pick（范围内 `top_k` 索引、`score>-inf` 时按 `blocks` 展开）或 `extra` 单元的单元格保留 `mask` 值，其余变 `-inf`。该 op 为单图节点、2 个 dispatch（`kernel_top_k_mask_fill` 网格步长填 `-inf`；`kernel_top_k_mask` 每 row 一个 threadgroup 拷贝 live picks/extra）。graph builder 已发融合形式，本 PR 使其在 MTL0 解析而非回退 CPU。Harness：MacBook M5 Max 128GB、Qwen3.8-Flash-Next-GGUF；Stats：⇈735.3 tok/s 等。

**应用场景**  llama.cpp Metal 后端的 QSA 稀疏注意力掩码（Qwen3.8-Flash-Next 等）。

**研判**
- **事实：** 明确的 op 语义 + 2 dispatch 实现 + graph builder 已发融合形式 + M5 Max 测试 harness。
- **观点：** 良性 Metal 后端移植（避免 QSA 回退 CPU）；Apple Silicon 上 QSA 模型部署方受益。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31230 （merge 2026-10-27T 窗口内；正文已核验）

### 5.3 `#31243` k-pool 稀疏选择解耦 kq_mask 布局

**主题**  `llama : build the k-pool sparse selection independently of the kq_mask shape`（`#31243`）。

**内容详情（取自 PR 正文）**  qwen4exp（QSA）与 glm5-next 通过 k-pool indexer 按 index 选 cache cell，自 `#31009` 起都不读 KQ mask 值；但选择仍按 dense KQ mask 的形状（`build_fused_top_k_mask` 取行布局、两模型都 assert 选择有 mask 形状），使稀疏层仍绑定 dense mask 布局。本 PR 把选择建成 `[n_kv, n_tokens]` 的 cache cell，并在注意力前按 K view 的 stream 拆分，使 dense mask 可改布局而不影响这些模型。为 `#29510`（unified cache 按每序列 K/V row list 注意力）所需；该处 qwen4exp 现需 opt-out，且会漏掉其 MTP 图。

**应用场景**  llama.cpp 中 qwen4exp/glm5-next 的 k-pool 稀疏选择（为 unified cache 改造铺路）。

**研判**
- **事实：** 明确的「选择按 mask 形状 → 按 `[n_kv,n_tokens]` cache cell」重构 + RTX PRO 6000/GB10 测试（`test-llama-archs` 过、PPL 一致、MTP/greedy 输出一致）。
- **观点：** 属 QSA/k-pool 架构重构（为 unified cache 铺路），正确性有 PPL/greedy 对照；记为**官方一手（重构 PR，为 `#29510` 前置）**。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31243 （merge 2026-10-27T 窗口内；正文已核验）

### 5.4 `#31275`/`#31257`/`#31252` LiquidAI d1 / LFM3 新模型支持（描述级/标题级）

**主题**  `model : add LiquidAI/d1-2B and LiquidAI/d1-omni-2B`（`#31275`）、`model : add LiquidAI/LFM3-Audio-2B and LiquidAI/LFM3-Omni-2B`（`#31257`）、`model : add LiquidAI/LFM3-1.7B and LiquidAI/LFM3-VL-2B`（`#31252`）。

**内容详情**  三个 PR 均为新增 LiquidAI 系列模型支持：d1-2B、d1-omni-2B、LFM3-Audio-2B、LFM3-Omni-2B、LFM3-1.7B、LFM3-VL-2B。描述较简，主要为模型注册/转换支持，未见逐模型精度/性能数据。

**应用场景**  llama.cpp 对 LiquidAI d1 / LFM3（文本/音频/omni/VL）系列模型的接入。

**研判**
- **事实：** 仅标题级信息；描述为简洁的模型新增，无逐模型精度/性能数字。
- **观点：** 新模型支持类（呼应跨引擎「新模型快速跟进」节奏）；因描述简、无性能数字，按规范记为**官方一手（标题级）**，待模型卡补充精度。

**来源**  https://github.com/ggml-org/llama.cpp/pull/31275 （merge 2026-10-27T 窗口内）、`https://github.com/ggml-org/llama.cpp/pull/31257`、`https://github.com/ggml-org/llama.cpp/pull/31252`（均窗口内；描述简）

### 5.5 其余 llama.cpp 窗口提交（标题级，正文未逐一抓取）

- **后端/内核：** `#31009` 见 5.1、`#31230` 见 5.2、`#31243` 见 5.3、`#31172` `opencl: initial support for q3_k`（Adreno 优化参考 `#30180`，Codex 实现）、`#31170` `opencl: refine mul mat id -1 index`、`#31239` `vulkan: fix CI llvmpipe by allowing dim 3 broadcast in MUL_MAT`、`#31255` `HIP/CUDA: remove pragma unroll from conv2d loops`、`#31240` snapdragon QDC artifact v2 API、`#31146` `ggml : relax scale to non-contiguous views`。
- **模型/推理/工具：** `#31275`/`#31257`/`#31252` 见 5.4、`#30980` `model: Add MiniCPM-OCR-Jev`、`#31271` `server: expose reasoning effort levels`、`#31256` `common : support deferred tools and tool_reference content parts`、`#31242` `--model/-hf required`、`#31204` `rpc : catch std::length_error in recv_msg`、`#31145` tests speed up tensor gen/saving、`#31216` tests fix multi-seq split replay、`#31200` remove obsolete n_pad、`#31187`/`#27401`/`#31212`/`#31215`/`#31246`/`#31276`/`#31187` UI（reasoning toggle、org-less 模型优先、conversation compaction、drawer、confirmation dialog 等）。
- **CI/构建：** `#31265` riscv CI timeout、`#31231` ROCm release ccache depend mode、`#31222` ubuntu 26.04 python 3.12、`#30751` WASI no spawn、`#31165` mtmd-cli batch pointer scope。

**滚动 nightly：** 10-27 共 **b11914–b11938** 区间 nightly 标签（窗口内可见 b11934–b11938；b11940 发布于 2026-10-28T00:32:11Z，紧贴 UTC 窗口边界，窗口内不可见），较 10-26 的 b11908–b11913 向前推进。

**证据等级：** 官方一手（提交 API + 标题）；`#31009`/`#31230`/`#31243` 已抓正文，其余为**官方一手（标题级）**；`#31275`/`#31257`/`#31252`（LiquidAI 新模型）描述简 → 标题级；`#31172`（OpenCL q3_k）描述简 → 标题级。

---

## 6. 社区与视频信号（HN）

**主题**  2026-10-27 窗口 HN Algolia 出现**一条引擎相关社区故事**（「Two RTX 5090 hosts sharing a KV cache over Ethernet」），与本期 KV cache 主线呼应；其余引擎关键词（`sglang`/`llama.cpp`/`TensorRT-LLM`/`speculative decoding`/`inference optimization`/`quantization inference`）命中均为 0，`vllm` 的 35 条命中全为「LLM」模糊匹配（无一条指向引擎本身），与既往各期一致。

- **「Two RTX 5090 hosts sharing a KV cache over Ethernet」**（作者 velmu，2026-10-27T16:09:50Z，3 分，0 评论）：链接 `https://viiwork.pulu.app/blog/kv-share-two-5090-hosts/`。即两台 RTX 5090 主机通过以太网共享 KV cache 的个人/小项目分享。
  - **研判：** 与本期引擎侧 KV cache 架构（SGLang `#42973` GLM DSA/MTP + KVShard + PD + CPU HiCache、TRT-LLM `#19369` V2 KV 连接器 attention DP）同源——都指向「跨主机/跨层 KV 复用与一致性」是当下热点；但本条目为个人博客（非官方、单项目、低分），属**二手线索/未核验**，对引擎普适架构无证据力。
  - **来源：** Hacker News `https://news.ycombinator.com/item?id=50261373`（已核验链接/时间）；博客 `https://viiwork.pulu.app/blog/kv-share-two-5090-hosts/`。**证据等级：二手线索/未核验。**
- `vllm`（35 命中）全部为「LLM」模糊匹配（如 MathMixup、Survival LLM、各类 Show HN/Tell HN），**无一条指向 vLLM 引擎**；`sglang`/`llama.cpp`/`TensorRT-LLM`/`speculative decoding`/`inference optimization`/`quantization inference` 均为 **0 命中**。
- **视频 / Reddit / X：** 未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

---

## 7. 累计趋势判断（继承 + 本期新增）

**继承（截至 2026-10-27 的各期累计结论）：**
1. **调度器隐式阻塞的定量画像**——优先级/调度路径的隐式退化（本期 vLLM 仍有 MRV2/sleep-mode、SGLang radix-tree GIL 等同类边界）。
2. **发版窗口的双向性**——「发版当天合入 main」≠「在发版内」（compare API 已多次证实）。
3. **默认值成为主要风险载体**——`--max-model-len auto`、默认开启的 flag、跨后端默认后端切换等改变行为且不报错（**本期强化**：vLLM `#61258` NUMA 默认绑定、SGLang `#43815` gfx942 AITER CK 默认开、`#46528` 默认 aiter 后端+fp8 KV 静默产出垃圾）。
4. **异构后端静默错误高频**——ROCm/Blackwell/Rubin 的 miscompile / 精度 / 同步陷阱（本期 SGLang `#46528` AMD fp8 KV 静默垃圾、llama.cpp 持续 Metal/OpenCL 后端修缮）。
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**——三者任一组合易踩、无显眼报错（本期 SGLang `#42973` GLM DSA/MTP 草稿/索引 KV 接入 KVShard+PD+CPU HiCache、vLLM `#60960` 前缀缓存评分封顶）。
6. **量化路径静默错误** 为与异构后端并列的第二类高频静默错误（TRT-LLM `#20823` NVFP4_AWQ、llama.cpp `#31097` MiniMax-M3 MoE scale）。
7. **专家张量/路由归属与分页漏接静默错误**（2026-10-25 立）：SGLang `#45787`、vLLM `#63372`、llama.cpp `#30820` 同源。

**本期新增 / 强化：**
- **第 3 条（默认值风险）获两条硬证据 + 一个最危险形态**：vLLM `#61258` NUMA 默认绑定、SGLang `#43815` gfx942 AITER CK 默认开均为「默认即行为变更」；而 **SGLang `#46528` 是「默认值 × 静默正确性」的耦合案例**——默认 aiter 后端 + 默认 fp8 KV auto-on，**既不报错也不告警，GSM8K 直接掉到 0**。建议把「升级后 diff 默认项 + 对应精度/回归（尤其 GSM8K 级）测试」列为推理 Serving 的发布门禁（本期再强化）。
- **静默错误主线从「异构后端/量化/MoE 分页」延伸到「AMD fp8 KV cache 误用 scale」**（SGLang `#46528`）：本期该条与第 4、6 条共构「低比特/后端静默正确性」主线，且是迄今**唯一导致精度归零（GSM8K→0）且不报错的案例**——提示 AMD fp8 KV + aiter 后端组合须强制精度回归。
- **KV cache 架构主线从「正确性/安全」走向「草稿与索引 KV 的所有权一致性」**（SGLang `#42973` GLM DSA/MTP 接入 KVShard+PD+CPU HiCache）：投机解码的 draft latent KV 与 DSA indexer keys/scales 现在与 PD/分层缓存共享同一所有权，其一致性与往期 PD-KV 校验/HiCache 静默损坏同源，建议 PD/分层缓存部署方把草稿/索引 KV 一致性纳入验收。

**趋势判断：** 2026-10-27（周二）四家提交量 vLLM 102 / SGLang 84 / TRT-LLM 48 / llama.cpp 32；vLLM 单日 102 条为追踪以来最高（超过周一 99），主因 MoE/GLM-5.3 性能 + DSv4.1 稀疏 logits + MiniMax-M3 稀疏 prefill + AMD/RDNA3 aiter + 默认值变更。本期主线可概括为「**默认值 × 静默正确性双主线延续（SGLang #46528 为教科书级耦合）+ GLM DSA/MTP KV 架构一致性**」三条并行；建议在量化（尤其 AMD fp8 KV + aiter）、MoE、跨后端默认、Metal/OpenCL 后端、KV 所有权五处均固化为上线验收。

---

## 8. 可执行清单 + 数据质量与采集说明

**可执行清单（建议动作）**
- **默认/跨后端风险（第 3 条）：** 升级后**强制 diff 默认项**——vLLM `#61258` 默认开 best-effort NUMA 绑定（尤其多 GPU/PCP 启动的 NUMA 拓扑假设）；SGLang `#43815` gfx942 AITER CK blockscale 默认开（回退 `SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK=0`）；AMD 部署方**必须**对 fp8 KV + aiter 后端组合跑 GSM8K 级回归以规避 `#46528` 的静默垃圾（GSM8K→0）。
- **vLLM 用户：** `#63820` TMA MoE up/down proj（GLM-5.3 H200 +3.1~3.4%）、`#63841` DSpark fp8 KV draft 省 5.38 GiB/rank、`#62011` DSv4.1 FP8 indexer 稀疏 logits（不再需 MXFP4）、`#60960` 前缀缓存评分封顶、`#57420` MiniMax-M3 Hopper tiled sparse prefill（opt-in）、`#52970` RDNA3 aiter 官方支持+dsv4、`#61557` 保留 grammar-invalid drafts、`#63889` KV-Offload KVCR adapter 开销；关注 `#61258` NUMA 默认。
- **SGLang 用户：** `#46528` AMD fp8 KV 静默垃圾修复（**必含**，否则 GSM8K→0）、`#42973` GLM DSA/MTP + KVShard + PD + CPU HiCache（关注依赖 `#43778` 是否同合）、`#43815` gfx942 AITER CK 默认开、`#46497` EPLB expert-count 融合（MegaMoE decode）、`#45474` Qwen3.5 MTP host work 入图（默认关）、`#44797` AMD aiter 路由 fp32 bias、`#45624` Flint 新模型（描述级、待补）；注意 `#46528`/AMD 路径 CI 状态。
- **TRT-LLM 用户：** `#20940` 单模型投机解码 DSpark Markov（注意签名变更 `dspark_markov_chain_logits` 移除）、`#20942` FP8 block-scale low-M GEMM（SM100/103/SM107）、`#19369` V2 KV 连接器 attention DP、`#20759` Agg Helix KV（默认关）、`#17658` KVCMv2 页表在 CC 下物化、`#20600` V2 调度降级长 uncached prefill。
- **llama.cpp 用户：** `#31009` glm5-next QSA 去 kq_mask（省 2–4 GB、5090 651pp/12.6tg）、`#31230` Metal TOP_K_MASK（QSA 不回退 CPU）、`#31243` k-pool 稀疏选择解耦、关注 nightly **b11914–b11938**；`#31275`/`#31257`/`#31252` LiquidAI d1/LFM3 新模型（标题级）；`#31172` OpenCL q3_k、`#31239` Vulkan CI llvmpipe 修复。
- **社区参考：** 「Two RTX 5090 hosts sharing a KV cache over Ethernet」（HN/个人博客，二手）——与 KV 复用主线呼应，但属个人项目、单 star 起步、未跨硬件验证，须自行验证，不当普适结论。
- **KV 架构主线（第 6/7 节）：** PD/分层缓存部署方把「草稿 latent KV 与 DSA indexer keys/scales 的所有权一致性」纳入验收（呼应 `#42973` 与往期 PD-KV 校验）。

**数据质量与采集说明**
- 官方提交 API（UTC 窗口 2026-10-27）：vLLM **102** / SGLang **84** / TensorRT-LLM **48** / llama.cpp **32**，均 HTTP 200（vLLM 分两页：100+2）。
- 21 个 PR 正文经 GitHub issue API 抓取（markdown 干净文本，均 HTTP 200）；四家 releases API（均 HTTP 200）；HN Algolia 公开 API（HTTPS 200）。
- 版本核验口径延续：发布说明/描述 ≠ 进版；PR 描述 ≠ diff 落地。本期 **SGLang `#45624`（Flint）描述为官方空模板**、`#31275`/`#31257`/`#31252`（LiquidAI 新模型）描述简，均按描述级/标题级；**vLLM `#63820`/`#62011` 正文在吞吐明细处截断**，记为官方一手但部分数字未完整呈现；**SGLang `#46528` 为带完整根因与复现的 silent-error 一手**；**vLLM `#61258` 注明来自社区 fork ivanium/vllm#53 的 cherry-pick**。
- 未采集 Reddit / YouTube / X；未核验到官方视频。所有性能数字均来自 PR 自述，未做独立第三方对拍（已在各条目标注）。
- 本档案与上一期（2026-10-27 档案，覆盖 2026-10-26）连续无空档；本期为连续第五期。
