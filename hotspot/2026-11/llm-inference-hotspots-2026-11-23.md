# 大模型推理优化每日热点档案

> 档案日期：2026-11-23（本自动化于该日执行）
> 关注窗口：2026-11-22（周日）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，四家单页取全无截断）+ releases API + 33 个关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 均经官方 issues API 抓取 markdown，官方一手；配额 60 起、抓取后剩余 13，无降级）+ HN Algolia（HTTPS，`numericFilters` 按 2026-11-22 epoch 窗口 1795305600–1795392000）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 18 / SGLang 54 / TRT-LLM 7 / llama.cpp 19 合计 **98** 条（2026-11-22 为周日，低活跃，较上一窗口 146 进一步回落，符合周末规律）；**窗口内无新正式版本、无新 prerelease**（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、llama.cpp 正式版 v1.0.0 11-13、TRT-LLM v1.3.0 11-20 与滚动 RC v1.4.0rc4 11-18 均落窗口前不变；llama.cpp 仅推进 nightly b12624–b12635，prerelease）。本期头号事件为 **静默正确性的两类新载体 + 采样安全维度**——TRT-LLM `#22328` 揭示 Qwen-VL 以图像 embedding 提交时 mRoPE 位置靠行数「猜」grid，对高>宽图像必然猜错（1920×1080 得到 `[1,80,102]` 而非 `[1,68,120]`，2052 位置中 1997 错，且**不报错、只改变输出**，标 BREAKING）；SGLang `#52277` 把多租户下可预测的 filler seed 换为 `secrets.randbits(63)`（此前 `random.Random(0)` 每次启动同序列，同批未设 seed 请求可被他人复现/预判），把第 24 条「服务安全隔离」从认证/输入边界延伸到**采样随机源**。HN 引擎官方更新类故事无；两条引擎相关故事（A100 调参工具、Gall/M2）均二手线索/未核验。

---

## 1. 版本状态与本期头号事件：无新版本 + 静默正确性新载体（mRoPE grid 猜测 / LoRA int32 溢出 / NVFP4 双重除半）+ 采样随机源安全加固

**提交计数（2026-11-22 UTC，均 HTTP 200；周日）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 18 | Kimi-K3.1-Pro 三条 recipe + Nemotron Shortcut 新模型 + KV Offload 四连 + Inkling2-Mini 5090 perf |
| SGLang | 54 | 采样/投机大规模重构栈（约 14 条）+ diffusion 簇（约 20 条）+ int64 溢出 + AMD aiter KV layout + hicache 栅栏 |
| TensorRT-LLM | 7 | BREAKING Qwen-VL mRoPE + Gemma4 MTP XQA + PageTableSpec + EPLB waiver 解除 |
| llama.cpp | 19 | NVFP4 scale 双重除半 + 多节点专家复制 + GGUF enum UB + audio FFT 2–4.5× + 多后端 |
| **合计** | **98** | 周末正常低量；与上一窗口（11-21 的 146）连续，无空档 |

### 1.1 头号事件一：TRT-LLM `#22328` Qwen-VL mRoPE 图像位置静默错误（BREAKING API，强化第 19 条「非量化常规路径静默错误」）

- **TRT-LLM `#22328` [BREAKING][fix] 用 image grid 修正解聚 serving 中 Qwen-VL 图像位置（官方一手但 CI 未闭合，fix + 破坏性 API 变更）**：Qwen2.5-VL / Qwen3-VL 用 mRoPE 放置图像 token，位置依赖每张图的 grid `(t,h,w)`。调用方一旦改用 `multi_modal_embeddings`（只传 embedding 行），TRT-LLM 只能**从行数反推 grid**；该猜测「从不比宽更高」，因此对所有高>宽图像必然错，部分宽图也错。**不报错、模型只是拿到错误位置、输出随之改变**。官方实例：Qwen3-VL-8B 处理 1920×1080 图像得到 grid `[1,80,102]` 而非正确的 `[1,68,120]`，**2052 个位置中 1997 个错误**。
  - 修法：每张图可携带自身 grid `{"image":[{"mm_embeddings":tensor,"image_grid_thw":[t,h,w]}]}`（复用 Llama4 的 per-item dict 形式），公共 helper `split_multimodal_embedding_items` 解析；不带 grid 的 Qwen-VL embedding 直接报错（BREAKING）。LLaVA-NeXT 保持 tensor-list 输入、忽略 grid。
  - **局限（coderabbit 评审明示）**：交换 `h`/`w` 不改变行数，代码无法从 embedding 形状识别，调用方必须用 processor 给出的 grid；解聚 serving 依赖先合入 `#20949`，且 `#22561`/`#22119` 中较晚合入者须把 `image_grid_thw` 加进 KV-cache key；L0 在 commit `933ae4f` 报 FAILURE、后续 `c2e2fa5` 无结果；QA 判 needs follow-up（多个新测试未列入 CI test-db）。
  - **研判（观点）**：这是「非量化常规路径静默错误」（第 19 条）在多模态解聚 serving 上的典型样本——错误来自**元数据缺失后的启发式猜测**而非算力路径。实践含义：① 以 embedding 形式接入 VLM 时，grid/位置元数据必须随张量显式传递，不能依赖形状反推；② 破坏性 API 变更本身就是修复的一部分（宁可拒绝不带 grid 的请求，也不要静默产出错误位置）；③ 该 PR 尚未闭合 CI 与 KV-cache key 依赖，跟进者需确认配套 PR 落地后再升级。

### 1.2 头号事件二：SGLang `#52277` filler seed 改为安全随机——采样随机源升维为多租户隔离问题（第 24 条延伸）

- **SGLang `#52277` [Fix] 用安全 RNG 为每个请求抽取 filler seed（官方一手，fix，安全加固，CI 全部跳过）**：确定性模式之外，当一个 batch 内含至少一个自带 `sampling_seed` 的请求，整批都走 seeded 采样路径，未设 seed 的行获得 filler seed。旧实现的 filler 来自模块级 `random.Random(0)`、**每次启动都是同一序列**，仅按调度顺序推进——因此与 seeded 请求同批的未设 seed 请求，其采样可被他人复现或预判。常量最初存在只是为了让独立采样的 TP rank 保持一致。
  - 修法：每个 generate 请求在进入调度器时（`attn_tp_rank==0` 的接收 rank、广播给其他 rank 之前）用 `secrets.randbits(63)` 取 filler seed，值随请求经 TP/DP-attention 广播与 PP relay 传递，所有采样 rank 共用同一 seed；`filler_seed` 为默认尾字段，Rust server 位置化 wire schema 不变，调度器接收时覆盖该字段（调用方无法设置），`sampling_seed` 仍是唯一可选 seed 方式。
  - **研判（观点）**：把上一期立下的第 24 条（多租户服务整型/输入边界从鲁棒性升维为安全隔离/认证）延伸到**采样随机源**——可预测的随机数与可绕过的认证、越界的指针算术同属「跨请求隔离面」。实践含义：① 共享 serving 实例中「未显式设 seed」不等于「不可预测」，此前实现下与任意 seeded 请求同批即被牵连；② 修法保留了 TP/PP 多 rank 一致性（seed 随请求传播而非各自抽取），是多租户安全与分布式一致性兼顾的范本；③ 该 PR 三套 CI 均标注 Skipped（需 `run-ci` 标签），合并后应关注实际 CI 覆盖。

### 1.3 静默正确性铁证（跨 TRT-LLM/SGLang/llama.cpp，强化第 6/9/19 条）

- **SGLang `#51199` [Fix] csgmv LoRA kernel 用 int64 token 行偏移（官方一手，fix，强化第 9 条整型溢出）**：chunked-SGMV LoRA kernel 从 int32 `permutation` 读物理行后乘 row stride（shrink 的 `s_offset_physical * x_stride_0`/`* output_stride_0`、expand 的 `* output_stride_0`）。当 `row*stride >= 2^31` 时 int32 乘积回绕，kernel 在张量之前读（shrink）/读写（expand）；一次约 `2^31/K` token 的未分块长 prefill 即触发（如复现用 `x` 为 65538×32768 fp16、末物理行偏移 2,147,516,416）。修法为 `tl.load` 后立即把 `s_offset_physical` 拓宽为 int64（shrink/split-K shrink/expand 三个 kernel），与 Triton LoRA 后端既有做法一致；L4 上复现 main 越界 fault、修后末行 shrink=32.0/expand=1.0，3 个 CUDA 回归测试在 <10 GB 空闲显存时跳过；`2^31` 元素以下输出不变。
- **llama.cpp `#32790` HIP 修 CDNA3 上 NVFP4 scale 双重除半（官方一手，fix，强化第 6 条量化 scale 静默错误）**：`__hip_fp8_e4m3_fnuz` 之后的 `/2` 是冗余操作，导致 gfx942（CDNA3）上 NVFP4 scale 被除两次、输出错误；移除该 `/2`。正文明示由 AI 发现问题，附最小复现 issue `#32789`；正文无性能数字。
- **SGLang `#51164` [Fix] mamba Triton verify 用中间状态 buffer stride（官方一手，fix，强化第 5/19 条投机×Mamba 状态正确性）**：自适应投机验证时 Mamba2 中间 SSM buffer 的容量可能多于当前 batch 使用步数，Triton kernel 却用活跃步数定位下一个请求。从第二个请求起会写入前一请求未用 slot、**接受步的状态留空未写**。修为把 buffer 的 batch stride（元素数）传入 kernel，存每步与树验证读父步都用该 stride；A10 上原 kernel 容量-8 的两个用例失败（4 活跃步、chain/tree）、修后全过（整模块 328 passed），独立 eager 检查中 1536/3072 活跃状态元素原本不一致且 1536 未用元素被覆写，修后逐位一致。CI Base ✓ / AMD ✗。
- **llama.cpp `#32641` GGUF 防止不可信文件数据生成非法 enum 值（官方一手，fix，安全/鲁棒性）**：`gguf_type`/`ggml_type` 是无固定底层类型的 C++ enum，仅 0–15 / 0–63 为合法值，把其他整数转成 enum 在 C++17 起是未定义行为（UBSan 报「load of value … not a valid value」）。旧代码先把文件里的原始 `int32_t` 转 enum 再检查；修为先保持整数、范围检查后再转换，移除两个 enum `read()` 重载，截断文件在 type 字段即中止。clang18+UBSan/gcc+ASan 多 seed 验证、合法文件行为不变；修 `ggml-org/ggml#1660`。

### 1.4 本期其他结构性观察

- **默认/后端切换的静默加载失败（第 4/19 条）**：SGLang `#51221` [Fix][AMD] 在 `SGLANG_USE_AITER=1 SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d` 下，混合线性注意力模型（Qwen3.5/Qwen3-Next）的 `HybridLinearKVPool` 是复合池，`_kv_cache_is_vectorized_5d` 只在 router 有 sliding-window 子池时才解析包装视图，对只有单个 FULL 子池的复合池读不到 layout、回退 `"nhd"`，后端在 5D 池上跑 NHD 路径→首次 prefill 即 `RuntimeError: if page_block_size > 1, must pass kv_last_page_lens`。修为有 FULL 子池就读 FULL 视图 layout；MI300X/MI355X GSM8K 从「启动即 RuntimeError」变为 0.335/0.220（与 NHD 基线 0.320/0.195 同量级）。**CI Base ✓ / AMD ROCm ✗**。
- **采样/投机验证正确性栈（第 14 条）**：SGLang `#52184` 揭示投机 verify 前调整 logits 的三份实现彼此漂移、并与非投机采样器漂移——EAGLE/NGRAM 路径**静默跳过 custom logit processor**、所有 verify 路径**忽略 min-p**（仅出现在 fast-path gate）。修订为统一 `adjust_verify_logits`（按 penalties→grammar→bias→custom processor 顺序，watermark 由各调用点后置），并新增 `min_p_renorm_prob`；`#52159` 把各处「是否调整 logits/是否截断采样」的判断收敛为 `adjusts_logits(info,*,has_grammar)` 与 `need_truncated_sampling(info)` 两个单一定义。**属采样正确性集中收口；`#52184` 行为变更明示 EAGLE/NGRAM 现在会运行 custom processor**。这一栈还包含 `#52174`/`#52164`/`#52170`/`#52185`/`#52192`/`#52270` 等重构，为同一批「逐行 seed / mask / penalty / verify ring 单一化」改动，非多次独立功能活动。
- **HiCache 与调度的异步栅栏（第 8/14 条）**：SGLang `#50469` [Perf] `write_back` 下设备逐出walks把 victim 备份到 host 后，原本阻塞调度器 `Event.synchronize()` 等 D2H 拷贝完成；在 DSv4.1-Flash 解聚 prefill 上该等待占调度时间 50–56%。修为提交拷贝后让 forward stream `wait_event` 在各拷贝完成事件上、不再阻塞 host，配合 overlap 调度；GB300（14 prefill + DEP16 decode）A/B：端到端 **958.8k→998.9k tok/s/GPU（+4.2%）**、prefill 计算 92.8k→96.4k（+3.9%）、**TTFT p50/p90 4.26s/12.0s→2.73s/9.6s**、TPOT p50 6.75→6.92ms（+1%），GSM8K 0.9735→0.9750（严格匹配）。保留阻塞等待的例外：L3 storage backend、buffer pipeline、cache linker、统一内存池、PD decode（拷贝完成前可能被改写 slot 的场景）。
- **KV Offload 生命周期健壮性（第 8 条）**：vLLM `#69110` 让 ARC 在 running-request 逐出时保留被复用的 T1 前缀（`#67098` 后续），GB10 上 SmolLM2-135M 4 请求：重放缓存 token 0→1,664、单 token 重放中位延迟 268.3→133.9ms，生成 token/logprob 三次配对全一致，输出差异依赖「请求部分驻留」。`#66719` 修请求未存块数超过 CPU tier 容量时整批 store 被拒、每步反复重试——改为按最低 token 位置存得下的块并上报无空间 key，RTX PRO 6000 上 7,663-token prompt：`cannot store chunks` 401→0、分配失败 401→0、store transfers 0→10、fs tier 0→119 块（873 MB）；存储字节由 sha256 验证与整块写入一致（重放续写约 90 token 后与 fresh 分歧，作者归因于加载-vs-fresh 数值差异而非本改动）。`#68451` 把 FS job 完成广播与 idle worker 等待条件分离（同一锁上第二个 condition），16+16 FS worker 上 store batch 延迟 −2.95%~−3.83%、FS worker CPU 39.6→19.9ms，4,608 请求输出一致，reload 端到端变化未确立。

---

## 2. vLLM：2026-11-22 提交（18 条）—— Kimi-K3.1-Pro 三 recipe(#69163/#69264/#69182) + Nemotron Shortcut 新模型(#68198) + KV Offload 四连(#69110/#66719/#68451/#67961) + Inkling2-Mini 5090(#69168, decode +6.5–7.5%) + DFlash anchor-first(#68827)

### 2.1 `#69163` / `#69264` / `#69182` [Recipe] Kimi-K3.1-Pro 三条硬件 recipe 对齐 InferenceX（官方一手，配置对齐/文档，性能来自第三方 InferenceX run）

周末 vLLM 18 条中 3 条为 Kimi-K3.1-Pro recipe，把 MI355X / B300 / GB300 三条 AgentX 预设对齐到 SemiAnalysisAI InferenceX 的实测配置：

- `#69163`（MI355X）：切到 `kimi-k31-712d9eb` 镜像与已发布 DSpark draft（`Inferact/Kimi-K3.1-Pro-DSpark` revision `e4e6e626`），prefix-match unit 从 64 改为 KV block span（DCP8 下 1152、无 DCP 1280），并发 32/48 跑 4 draft token。引用 InferenceX run 44640765021：并发 1→48 交互 p50/p90 从 337.5/328.9 降到 47.6/30.3 tok/s/user、每 GPU 总吞吐 2,539→22,240 tok/s，GSM8K strict EM 0.979–0.980。**注意**：吞吐点用新 draft golden AL（4/5/7 token 分别 3.46/3.81/4.16）的合成接受，与上一版 preview draft 的较低 AL 数字**非同口径对比**；本 PR 未新增 GPU 实测，全部证据来自 InferenceX run，且镜像内 `K31_*`/`KIMI_K3_FLYDSL_SWA_GQA` 等开关不在 vLLM main。
- `#69264`（B300 PD）：切到 `kimi-k31-mc-f4c9618` 镜像 + Dynamo `ec06c09e72`，点集 1(1P1D,7 draft)/20、48(1P3D,5)/96(1P3D,4)/660(4P5D,5)，Mooncake store 尾清理/lookup 预取/BF16 indexer logits/2% 逐出等设置全点生效，两角色加 `--numa-bind`；1→96 客户端 decode 开镜像小批 decode 核（`latency_specialized` + `K31_*`），660 客户端 decode 改 FlashInfer autotune；InferenceX#5175 run 44931328829 五点全过、48/96/660 GSM8K strict 0.978–0.981。**实测数字**：c96 1P3D 137.8 tok/s/user、18,834 tok/s/GPU；c48 1P3D 257.7 tok/s/user、11,322 tok/s/GPU。
- `#69182`（GB300）：四个预设全部换用已发布 DSpark draft `e4e6e626`，新增 104 client 点、移除 56 client 点；DEP32(960) 点改为单个 DEP32 prefill worker（1P1D）并更新 `--prefix-match-unit`、EPLB、Mooncake 5% 逐出等；逐 flag 对齐 InferenceX#5109。
- **研判（观点）**：这类「对齐 InferenceX」recipe 的价值在于把服务栈参数（镜像 digest、draft revision、block/prefix-match unit、捕获尺寸、NUMA 绑定）与权威跑分逐点锁死，可复现性强；但其性能数字属**第三方基准/描述级**（vLLM 本身未重跑），比较时须注意 draft AL 口径与「镜像专属开关不在 main」的前提。

### 2.2 `#68198` [Model] 新增 Nemotron Shortcut（文本/MTP/视觉语言，官方一手，新模型支持）

以自包含包 `vllm/models/nemotron_shortcut/`（沿用 per-model 布局）加入：文本/图像/视频输入、TP/PP/DP+EP、MTP；BF16/MXFP8/NVFP4 权重，BF16/FP8/NVFP4（待 FlashInfer bump）KV cache。模型部件含 SDSA 稀疏注意力（indexer 选每 query 的 Top-K keys + MQA core）、Gated DeltaProduct（GDP）mixer、shortcut MoE（routed experts 读 block 输入以便 mixer 未跑完即开始 dispatch）。依赖 `#67941`（混合页大小的 layer-compact KV layout）才能运行、`#68353`（N 未对齐的 MXFP8 linear）才能用 MXFP8 权重；模型目录 7,610 行、测试 1,913 行、目录外仅 68 行注册。4×GB300：单元 1,496 passed/1 failed（该失败 main 上也存在），评审修正后 165 passed/1 skipped。**研判（观点）**：延续 RFC #42770 的 per-model 自包含趋势——新架构尽量把算子/布局放进模型目录、引擎改动收归 KV layout 与注册表，降低对通用 runner/scheduler 的侵入。

### 2.3 `#68827` [Spec Decode] DFlash/DFlash2 支持 anchor-first（sample_from_anchor）draft（官方一手，feat，强化第 14 条）

此前 MTP draft checkpoint 若设 `sample_from_anchor: true`（anchor-first：K query 对 K draft token，如公开 `DaoCloud/Qwen3.8-27B-DFlash2-Exp`）会在启动即 `ValueError`，因配置/分配侧硬编码 `1+K` fill-in 布局（输入 kernel 已有 `SAMPLE_FROM_ANCHOR` 路径）。修法从 checkpoint 的 `dflash_config` 读该字段并统一 `num_query_tokens`（K 或 1+K），speculator/KV lookahead/DSFlash2 grouped-conv block size 共用；DSpark 本就 anchor-first，继承逻辑并仅覆盖默认值；无新 flag。B200 验证：此前启动报错的 anchor-first checkpoint 现 GSM8K 0.920/接受长度 6.03；fill-in 布局 128/128 输出逐字一致；anchor-first（私密 draft、Qwen3-Omni ASR）与 out-of-tree patch 同 driver 下 4,090/4,096 逐字一致、接受率 0.5148→0.5139、RPS/P99 基本持平（驱动变更本身会改变约 900 个输出，故只比同 driver 配对）。supersedes `#54154`。

### 2.4 `#67961` / `#68451` / `#69110` / `#66719` KV Offload 四连 —— 见第 1.4 节

### 2.5 `#69168` [Perf][Inkling2-Mini] 5090(sm120) 小 M GEMM 三项改进（官方一手，perf，强化第 12/14 条）

三项 decode 期低 M（M≤16）BF16 GEMM 优化（`#69028` 引入的 `sm12x_bf16_ll_gemm`）：① 列 tile 不足填不满 GPU 时确定性 split-K，short-conv 投影（N=32）改用后**最高 7.1×**；② loop blend 的 3D 权重走 batched matmul 替代两次 `torch.bmm`，**最高 5.1×**；③ q/k/v 三次 launch 合一，**1.2×**。RTX 5090 E2E（Inkling-2-Mini，DSpark 7 token）decode step **并发 1 −7.5%、并发 2 −6.5%**；并发 1 接受长度相同（3.53 vs 3.52），吞吐增益来自 kernel；并发 2 接受长度各 run 不同，故以 ITL 比较。不 split 的形状与 main `_gemv` 逐位一致；**未在 DGX Spark(SM121) 调优/验证**（dispatch 覆盖）。

### 2.6 其他要点（标题级 / 描述级）

- **KV Offload 新能力**：`#9ae9c15c`（#67961）给多层 KV offload connector 加 `access_mode`（read/write/readwrite），支持「CPU→FS 写」与「GDS 从 FS 直读进 GPU」经 multi-connector 组合（正文无 Validation，描述级）。
- **多模态 bugfix**：`#68802` 加载视频时套用模型 HF video processor 的帧数上限（此前 chat 路径经 Qwen3-VL backend 硬编码 `max_frames=768`，LFM3-Omni-2B 声明 384 却拿到更多帧）；`#68317` OpenCUA processor 透传 image kwargs；`#69052`/`#69099` 接受 torch.Tensor 视频与 NumPy 图像/PIL 帧视频。
- **新模型/recipe**：`#0fa66689`（#68198）Nemotron Shortcut；三条 Kimi recipe 见 2.1；`#118d12f7` 重组 docs/training。
- **服务/前端**：`#68408` Rust frontend 支持 `allowed_tools` tool_choice；`#68667` offline LLM API 错误分类。
- **CI/构建**：`#68867` LoRA layer 测试与 GLM 模型导入解耦。

---

## 3. SGLang：2026-11-22 提交（54 条）—— 采样/投机验证重构栈(约14条, custom processor/min-p 静默漏用) + int64 LoRA 溢出(#51199) + AMD aiter KV layout(#51221) + hicache write_back 栅栏(#50469, +4.2%) + DSA MTP prefill(#46784, +2.8–3.7%) + diffusion 簇(约20条) + filler seed 安全(#52277)

### 3.1 `#51199` / `#51221` / `#51164` / `#52277` 静默正确性与安全 —— 见第 1.2/1.3/1.4 节

### 3.2 `#50469` [Perf] hicache write_back eviction 拷贝在 forward stream 上栅栏（perf，见第 1.4 节）

### 3.3 `#46784` [Perf] DSA MTP draft prefill KV 免 attention 填充（官方一手，perf，强化第 14 条）

GPU profile 发现 GLM-5.3-Flash 8k prompt 的 MTP draft extend 占 prefill step 4.4% GPU 时间、draft 的 DSA attention+indexer 占 2.7%，但 draft 输出**只用于每个请求最后一个 prompt token**。修法：prefill 后仅对末 token 跑 draft attention/DSA indexer/MoE，其他 token 只写 draft KV cache（latent KV 与 indexer keys 来自 attention 之前的输入）。H200：8k/32k prompt 中位 TTFT −2.8~−3.9%、输入吞吐 +1.8~+2.2%；4 GPU FP8 KV 下输入吞吐 +3.3~+5.5%；8×B300（合作者复现）TTFT −3.5~−5.3%。**精度基本无损**：非末 token 的 draft latent KV 在 BF16/FP8 下与 main 逐字节一致，GSM8K/RULER2(16k)/接受长度在误差内。代价：每个 split prefill 约 3.5ms CPU，故仅对足够长的 prefill 启用（默认 GLM-5 系列与 DeepSeek-V3.2 on CUDA；≤2048 token 的请求与列出的 FP8 KV 配置走 main 路径）；无新 flag；8×B300 有一行非 CUDA graph 的 32k prefill 慢 2%（target prefill graph 覆盖该 chunk 后恢复 +2%）。

### 3.4 `#52184` / `#52159` 投机 verify logits 调整统一 + gate 单一定义 —— 见第 1.4 节

### 3.5 `#50906` [Feat] glm5_next 支持 LayerNorm sequence parallelism（官方一手，feat，opt-in）

TP attention 下每 rank 都在全部 token 行上跑 GLM-5.3-Flash 的 mHC（`mhc_pre_big_fuse_with_norm`/`mhc_post`/`hc_prenorm` GEMM）——8×H200 TP8 上 128K 冷 prefill 中 mHC 占 rank0 651ms GPU 时间的 81ms，8 个 rank 重复同样工作。修法：SP 下 attention（KDA 与 MLA/DSA）与 FFN 经 `run_replicated_token_rows` 在聚集行上跑、再交还本 rank 行数；FFN 输出在欠 TP sum 时 reduce-scatter；把 `Glm5NextForConditionalGeneration` 加入 SP allowlist。无新 flag/环境变量，仅 `--enable-layernorm-sp` 开启时生效；GLM 的 SP 校验拒绝 attention CP、MoE a2a backend 与 `--enable-attn-tp-input-scattered`。精度段被截断未读全（描述级性能/精度数字），作者另试 `--enable-attn-tp-input-scattered` 对 mHC 无改善（32K 冷 prefill mHC 20.14 vs 20.09ms）。

### 3.6 diffusion 簇（约 20 条，周日主力，标题级/描述级）

- `#52023` 一个 server 同时服务 MiniMax-H3 fl2va 与 ref2va（`--model-variant fl2va+ref2va`）；`#52027` ROCm(gfx950) 上 MiniMax-H3 视频 VAE decode 加速；`#52051` MiniMax-H3 音频与视频 tile 并行解码；`#51999` LTX-2.3 STG pass 复用 cond pass 前缀块；`#52001` Ulysses pipeline pool 保留首次检查并按 headroom 增长。
- `#52224` VSA-H3 四个 Triton kernel 与 VAE scaled-residual-add 不再在请求路径重编译（把随请求形状变化的整数改为 `do_not_specialize`）：2×B300 上 ready 行之后编译的 kernel 变体从 5/2/1/5 降到 2/0/0/0，MP4 md5 与 main 一致（见 3.7 描述级性能）。
- `#52012` [Feat] diffusion 请求 watchdog：请求超 `--request-time-limit` 则 `/health` 503 + SIGUSR2 打印各线程 Python 栈（`faulthandler`，无需 ptrace/GIL，可在主线程卡在 CUDA call 时工作）；默认关闭、挂住请求不中止（需重启实例）、节点本地 socket 不支持解聚角色。2×H200 MiniMax-H3 注入卡死测试：`t2va=60` 时 `/health` 在 +61.19s 由 200→503。
- 其余：`#52225` warm-up 构建 MiniMax-H3 vision-tower cuDNN attention kernel；`#51187` Qwen2.5-VL vision RoPE 表在 GPU 上重建（encoder offload 下 Qwen-Image-Edit 输出一致）；`#52226` offloaded weight 提交合并 LoRA 时先 snapshot；`#51425` Wan2.1 I2V 图像分支 LoRA key 映射；`#36727` 低显存 offload 保留 mapped 组件权重；`#51753` `/v1/videos` 每视频独立 sample；`#52074` grouped_gemma_rmsnorm 零 token launch 保护；`#51451` diffusers 格式 LoRA 加载。

### 3.7 其他要点（标题级 / 描述级）

- **PD 解耦**：`#46460` 把 MooncakeTransport 抽到通用 Transport 接口后；`#52283` 给非末级 PP stage 边界测试接 Mooncake transport；`#46460` 是接口重构、非性能变更。
- **采样栈（`#52170`/`#52185`/`#52192`/`#52270`/`#7df7c097` 等）**：逐步把 seed、sampling-mask 行、penalty history、grammar mask 改为 per-step 构建/流水线阶段，删除 `SamplingBatchInfo` filter/merge；`#49982` 用 fast tokenizer 声明的 decoder 判定 byte-level BPE。属同批重构栈（见 1.4）。
- **AMD/性能**：`#51170` 去 mxfp8-moe 冗余输出清零；`#50779` AITER 不再拒绝 `apply_routed_scaling_factor_on_output`；`#51632` 按真实形状 tune NemotronH 非 gated relu2 MoE；`#52173` custom-all-reduce 注册时一次映射每个 peer graph-input 分配；`#48746` 外部模型可用 `use_mla: true` opt-in MLA caching（CPU 52 测试通过、无 GPU 端到端精度评测，见 1.4）。
- **模型/调度**：`#50906` GLM LayerNorm SP（见 3.5）；`#48746` 见上；`#448764da`（#47047）GLM-5-Next 为每种 scattered MoE all-to-all backend 收集 DFlash aux hidden states。

---

## 4. TensorRT-LLM：2026-11-22 提交（7 条）—— BREAKING Qwen-VL mRoPE(#22328) + Gemma4 MTP XQA on SM120(#22515, +22.6%) + PageTableSpec(#21769) + EPLB waiver 解除(#22748)

### 4.1 `#22328` [BREAKING] Qwen-VL mRoPE 图像位置 —— 见第 1.1 节（本期头号）

### 4.2 `#22515` [feat] Gemma4-E4B MTP using XQA on SM120（官方一手，feat，perf，强化第 12/14 条）

无 trtllm-gen decode 的 FlashInfer 后端（如 SM120）上，Gemma4 MTP 的 shared-KV assistant 启动即失败（`FlashInfer draft metadata views require the trtllm-gen decode backend`），且 overlap scheduler 用不了（FA2 需 plan() 时 host 侧 KV 长度、与设备侧读取冲突）。修法：这些调用走 FlashInfer XQA（`xqa_batch_decode_with_kv_cache`，设备侧读 block table 与逐行 KV 长度、无 plan 步骤），在 overlap 调度与 CUDA graph replay 下仍有效。**RTX PRO 6000(SM120)、Gemma-4-E4B FP8 + MTP assistant（3 draft、batch 8）、ISL 4104/OSL 554**：总 decode 吞吐较非 overlap 基线 **+22.6%**——XQA target verify 启用 overlap（1054.9→1233.0 tok/s，+16.9%）+ breakable prefill CUDA graph（1201.4→1230.1，+2.4%）+ 跳过逐层 VSWA page-index 拷贝（~180 µs/iter，~1.6%）+ FP8 attention 输出直入 FP8 o_proj（ITL −1%，~1%）；GSM8K 在噪声内不变。含配套修复（decode-only XQA 跳过 VSWA 拷贝、breakable CUDA graph 只路由已注册元数据、FP8 KV 下 XQA 直写 FP8 o_proj 输入）。coderabbit 评审提示若干契约需确认（XQA 输出 scale 缓存刷新时机、per-layer KV pool 索引、paged append 跨 launch 切分）。

### 4.3 `#21769` [feat] KV cache manager 用 PageTableSpec 声明页布局（官方一手，架构重构）

RI-10 系列第一弹：manager 一次声明 cache 布局、lender 与各传输后端读取同一描述，新增 cache 结构只需覆写一个 manager 方法。`KVCacheManagerV2.page_table_spec()` 返回新模块 `pyexecutor/kv_cache/page_table.py` 的 `PageTableSpec`，含 `role_layouts`/`default_layout`、`untransferred_roles`、`derived_roles`、`request_rebuilt_roles`、`layer_ids`（混合精度 KV 与 FP4 MLA 显式给名）、`quant_mode`、`mixed_precision`（预留）、`retention_margins`；`layout_of(role)` 返回角色布局。`RoleLayoutV2` 五类：`ByHead`（head/token-major）、`Sectioned`、`Replicated`、`RingSlots`、`Own`，各自序列化为 plain dict 供 peer 读取，新布局=新增一个子类；模块仅用标准库便于任意传输后端导入。base manager 声明按 KV head 拆 K/V 与 block scale、K-only(MLA) 一个 latent head 每 rank 持有完整、index-K 复制；各 V2 manager（FP4 MLA、CSA2 等）声明各自 spec 与现有 `get_disagg_*` hook 一致。**研判（观点）**：KV 传输正确性长期受「各后端各自理解布局」困扰（历史多次 role/layout/scale 静默错），统一声明是把正确性约束前移到类型的架构动作；本期仅加声明、传输层仍读旧 hook，属渐进迁移第一步。

### 4.4 `#22748` / `#22139` 其他（官方一手，fix）

- `#22748` 移除 GB200 上 `TestDeepSeekV3Lite::test_bfloat16_4gpus_online_eplb[mtp_nextn=2-moe_tactic=trtllm_gen]` 的最后一条 waiver；运行时修复已在 `#19949`（连续高 rank 专家 slot 可被负载均衡描述、BF16 TRTLLM-Gen host 专家权重与 device slot 同用 BlockMajorK 布局）。4×GB200 本地跑 3 个精度用例全过、无 skip；coderabbit 判 needs follow-up（无当前 head 的 CI pass）。
- `#22139` rank-striped 内存检查把可回收 page cache（`active_file`+`inactive_file`，扣 `memory.min`/`memory.low` 保护部分）计为可用内存，cgroup v1/v2 分别处理；此前把完整 `current` 计入 in_use 使有大 page cache 的 host 上加载被推迟。KV cache manager V2 host-tier 自动定容经此路径变化。GB300 A/B、150 campaign runs、46 测试为作者报告（评审未独立跑）；风险：准入放宽后更多加载会启动、目录扫描/多次 `memory.stat` 读的 I/O 成本未测；QA 判 needs follow-up。

### 4.5 其他要点（标题级）

- `#b86c541b` [perf] 用 NumPy 构建 DSA request indices；`#93861e58` [infra] 提交 nightly 最新 lock file。

---

## 5. llama.cpp：2026-11-22 提交（19 条）—— NVFP4 scale 双重除半(#32790) + 多节点专家复制(#32807) + GGUF enum UB(#32641) + audio FFT 2–4.5×(#32815) + Adreno coopmat(#32686) + 多后端

### 5.1 `#32790` / `#32807` / `#32641` 静默正确性与安全 —— 见第 1.3 节

### 5.2 `#32815` mtmd 加速共享 audio FFT（官方一手，perf，非推理核心路径）

把递归 radix-2 拆分改为迭代 stage、把奇数长度 DFT 跨子序列批处理，保留原浮点运算顺序；`mtmd_audio_fft/ifft` 改为 `const float *` 输入、不再拿输入当 scratch。Xeon 2.1GHz 4vCPU 上 960 组正反变换（24 尺寸×20 输入）与 master 默认 x86-64 flags 逐位一致；中位耗时降到 0.22–0.45×（**约 2.2–4.5× 加速**），如 zonos3 ifft N=2208 322212→71304ns（0.22×）、s3tok fft N=400 20149→5274ns（0.26×）。属多模态音频预处理优化，非 LLM 解码核心。Claude Code 编写 FFT 与基准、作者审阅。

### 5.3 `#32686` / `#32727` 多后端要点（官方一手，feat/fix）

- `#32686` OpenCL 为 Adreno 850 加 coopmat GEMM（`cl_qcom_kernel_cooperative_matrix`），覆盖 Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K；正文无性能数字。
- `#32727` 修 WebGPU 与非 UMA 设备的非零 clear：`vkCmdFillBuffer` 与 WebGPU memset shader 写的是 32-bit word 而非 byte，给 `0xAB` 时实际写 `AB 00 00 00` 到每个 word；作者评「影响低」（树内测试外无非零 clear 用法），本地验证 WebGPU 修复。

### 5.4 其他要点（标题级 / 描述级）

- `#5bacb7c8`（#32807）多节点共享 host 权重的专家全量复制（见 1.3）；`#94f3af3d`（#32815）见 5.2；`#f505c30b` 修 qwen3a 音频尾部；`#d2bceda3` inkling2 启用 `ignore_merges`；`#29307` chat 实现 token-aware parsing；`#32497` 裸 JSON 工具调用触发 lazy grammar。
- `#8a97fa98` Vulkan 支持 SCALE 的非连续 src；`#b0520492` Vulkan 修非 UMA 设备未对齐 range 的 memset；`#1fb8a269` OpenCL Adreno 850 coopmat（见 5.3）；`#5d28d64b` ggml-webgpu 更新 Dawn；`#e300919c` WebGPU 更新 GDN supports_op check。
- `#225a5345` server 把非因果 batch 塞进单个 ubatch；`#0b51bb2f` 单模型模式支持 `/models/sse`；`#56eea9e4` grep_search 工具加 offset 分页；`#8d60873d` GGUF 防止不可信 enum 值（见 1.3）；`#9c10c94e` 修 WebGPU/非 UMA 非零 clear（见 5.3）；`#ad50351f` 加 SIGMOID 反向传播。

---

## 6. 趋势研判与累计结论

累计结论继承第 1–24 条（见上一期档案第 6 节），**本期无新增条目**；本期铁证按既有条目归类并强化如下：

1. **静默正确性（第 4/6/9/19 条）持续为最高频主线，且载体继续多样化**：本期三类新载体——① **元数据缺失后的启发式猜测**（TRT-LLM `#22328` Qwen-VL mRoPE grid 靠行数反推，高>宽图必错、1997/2052 位置错误）；② **int32 行偏移溢出**（SGLang `#51199` csgmv LoRA，2^31 元素边界，第 9 条再添铁证）；③ **量化 scale 操作冗余/双重**（llama.cpp `#32790` HIP CDNA3 NVFP4 scale 双重除半，第 6 条）。加上投机×Mamba 状态 stride（`#51164`）与 AMD 复合 KV pool layout 误读（`#51221`），静默错误已覆盖「元数据/整型/量化/状态/布局」五类，且多数「不报错、只改变输出」。
2. **服务安全隔离（第 24 条）延伸到采样随机源**：SGLang `#52277` 表明可预测的 filler seed 与可绕过认证、越界指针同属跨请求隔离面；与 `#52212`（diffusion request watchdog，`/health` 挂起检测）、GGUF enum UB（`#32641`）共同构成「安全/鲁棒性」维度。第 24 条口径更新为：多租户推理服务的整型边界、输入边界、采样随机源与文件解析边界均已从鲁棒性升维为安全隔离问题。
3. **采样/投机验证正确性集中收口（第 14 条）**：SGLang `#52184`/`#52159` 揭示并修复 custom logit processor 与 min-p 在投机 verify 中被静默跳过，把分散的 logits 调整与 gate 判断收敛为单一定义；vLLM `#68827` 扩展 DFlash anchor-first 布局支持。这类「多条重构同一批改动」须按栈理解（非多次独立功能活动）。
4. **默认值/后端切换的风险面（第 3/16 条）**：SGLang `#51221` 的 AMD aiter + 复合 KV pool + vectorized_5d 组合在首次 prefill 即 RuntimeError（非静默但启动失败）；TRT-LLM `#22328` 的 BREAKING 变更「宁可拒绝不带 grid 的请求」是反向的默认值安全实践。
5. **KV 卸载与调度的异步栅栏优化（第 8 条）**：SGLang `#50469` 把 D2H 备份拷贝从阻塞 host 改为 forward stream 栅栏，端到端 +4.2%、TTFT p50 减半；vLLM `#69110`/`#66719`/`#68451` 完善 KV Offload 前缀保留、超容量增量存储与 idle worker 唤醒。调优重心从「算得快」转向「别让调度器空等 / 别丢可复用前缀」。
6. **旗舰模型 recipe 与多硬件 day-0 成常态（第 11/17 条延续）**：vLLM 三条 Kimi-K3.1-Pro recipe（MI355X/B300/GB300）逐 flag 对齐 InferenceX 跑分、SGLang GLM-5-Next LayerNorm SP、TRT-LLM Gemma4 MTP XQA on SM120（+22.6%）；性能数字多为第三方基准或描述级，须注意 draft AL 口径与镜像专属开关前提。
7. **架构性 KV 抽象收敛**：TRT-LLM `#21769` PageTableSpec 让 manager 单点声明页布局、传输后端统一读取，是对长期 layout/role/scale 静默错的类型级治理；SGLang `#46460` MooncakeTransport 抽通用 Transport 接口同属此类。

---

## 7. 可执行清单（面向工程落地）

1. **VLM 多模态接入审查**：若以 `multi_modal_embeddings` 形式提交图像，务必随附 `image_grid_thw`（用 processor 输出），不要依赖行数反推 grid；TRT-LLM 侧确认 `#20949` 与 `#22561`/`#22119` 中较晚者已把 grid 加入 KV-cache key，再升级含 `#22328` 的版本。
2. **共享 serving 的多租户隔离**：升级含 SGLang `#52277` 的版本，避免同批 seeded 请求牵连未设 seed 请求的采样可预测性；未显式设 seed 不等于不可预测。
3. **长上下文/LoRA/量化路径排查**：长未分块 prefill 触发的 LoRA csgmv 越界（`#51199`）与 gfx942 NVFP4 scale 双重除半（`#32790`）属静默错误，升级对应修复；对 2^31 元素边界、CDNA3/NVFP4 路径做专项回归。
4. **投机/Mamba/AMD 组合**：启用 EAGLE+NGRAM 自定义 logit processor 或 min-p 的场景升级 `#52184`/`#52159`；Mamba2+投机 verify 升级 `#51164`；AMD `vectorized_5d`+混合线性注意力升级 `#51221`（注意其 AMD CI 仍红）。
5. **解码服务性能**：GSML-5.3-Flash/DSv4.1-Flash 解聚 prefill 用 HiCache write_back 时评估 `#50469`（+4.2% tok/s/GPU、TTFT p50 减半）；Inkling-2-Mini 5090 部署评估 vLLM `#69168`（decode −6.5~7.5%，DGX Spark 未验证）；GLM-5.3-Flash 长 prompt 评估 SGLang `#46784`（+2.8~3.7%，基本无损）。
6. **KV Offload 运维**：vLLM KV Offload 用户升级 `#69110`（ARC 复用前缀保留）、`#66719`（超 CPU tier 增量存储，消除每步 `cannot store chunks`）、`#68451`（idle worker 唤醒）；三者均有 A/B 与字节一致性验证。
7. **Recipe/跑分复现**：引用 Kimi-K3.1-Pro recipe 数字时标注 draft AL 口径（4/5/7 token=3.46/3.81/4.16）与「镜像专属开关不在 vLLM main」，避免与 preview draft 数字直接对比。

---

## 8. 来源、数据质量与限制

- **一手来源（官方仓库/API/文档）**：vLLM 18 / SGLang 54 / TRT-LLM 7 / llama.cpp 19 条提交均经 `api.github.com/repos/<repo>/commits?since=2026-11-22T00:00:00Z&until=2026-11-23T00:00:00Z` 取得（全部 HTTP 200、单页取全无截断）；releases API 确认窗口内无新正式版/无新 prerelease；33 个关键 PR 正文经官方 issues API 抓取 markdown（配额 60→13，无降级），PR 号均已在窗口内提交标题中二次确认。性能数字如无特别说明均来自 PR 正文（作者/合作者实测）。
- **二手线索/未核验**：HN Algolia 窗口内 `vllm` 22 命中（多为「LLM」模糊匹配）、`sglang` 1、`llama.cpp` 1、`TensorRT-LLM`/`speculative decoding`/`paged attention` 均 0。两条引擎相关故事——3 分「Automated K8s hill-climbing harness and tuning vLLM and SGLang on a single A100」（仓库 `johnsonshi/a100-inference-hillclimb`，已核验存在、1 star）与 5 分「Show HN: Gall – Inkling 2 Mini on 24GB M2」（`ltrgoddard/gall`，4 stars）——均为二手线索/未核验，未作为确定事实采纳。
- **未采集/无可用数据**：Reddit、YouTube、X 本期未采集；窗口内未核验到可确认的官方视频发布。
- **证据等级口径**：官方一手=官方提交/releases/issues API 正文；官方学习资料=vLLM/SGLang 官方文档与 recipe（性能多为第三方 InferenceX run，标「描述级/第三方基准」）；二手线索=HN 故事与社区转述（已核验仓库存在仅说明真实性，不代表内容准确性）；未核验=社区 API/视频页不可达或无一手确认的内容。
- **描述级/未闭合项清单（本期）**：TRT-LLM `#22328` L0 FAILURE + QA needs follow-up + grid h/w 交换不可检测；`#22748`/`#22139` QA needs follow-up（无当前 head CI pass / 作者报告未独立验证）；SGLang `#51221` AMD ROCm CI ✗、`#52277` CI 全 Skipped、`#48746` 无 GPU 端到端精度、`#50906` 性能/精度正文截断；vLLM `#67961` 正文无 Validation、三条 Kimi recipe 性能来自 InferenceX 第三方 run 且 draft AL 口径不同；llama.cpp `#32790`/`#32686` 正文无性能数字、`#32727` 作者自评影响低。以上均按「官方一手但验证未闭合/描述级」处理，不作为确定结论。
