# 大模型推理优化每日热点档案

> 档案日期：2026-11-22（本自动化于该日执行）
> 关注窗口：2026-11-21（周六）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，四家单页取全无截断）+ releases API + 41 个关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 均经官方 issues API 抓取 markdown，官方一手；配额 60 起、抓取后剩余 11，无降级）+ HN Algolia（HTTPS，`numericFilters` 按 2026-11-21 epoch 窗口 1795219200–1795305600）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 28 / SGLang 74 / TRT-LLM 19 / llama.cpp 25 合计 **146** 条（2026-11-21 为周六，低活跃，较上一窗口 338 大幅回落，符合周末规律）；**窗口内无新正式版本、无新 prerelease**（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、llama.cpp 正式版 v1.0.0 11-13、TRT-LLM v1.3.0 11-20 [落窗口前] 与滚动 RC v1.4.0rc4 11-18 [落窗口前] 均保持不变；llama.cpp 仅推进 nightly b12613–b12619，prerelease）。本期头号事件为 **vLLM 密集服务安全加固簇**——周六 vLLM 仅 28 条提交中 6 条为 `[Security]`/认证边界修复，其中 `#53131` 把第 9 条「整型溢出」从静默正确性**升维为已确认可利用的跨用户数据泄露（confirmed exploitable with PoC）**，`#52794` 修复 `--root-path` 非段前缀导致的 `--api-key` 认证完全绕过。HN 引擎官方更新类故事无；引擎相关命中多为泛 LLM 帖与配置工具（均二手线索/未核验）。

---

## 1. 版本状态与本期头号事件：无新版本 + vLLM 服务安全加固簇（整型溢出升维为跨用户数据泄露）+ 静默正确性跨引擎延续

**提交计数（2026-11-21 UTC，均 HTTP 200；周六）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 28 | 单页取全；其中 6 条 `[Security]`/认证边界 + 静默正确性 + GDN MTP 核 + DiffusionGemma 默认切换 |
| SGLang | 74 | diffusion/采样/调度大簇 + int64 溢出 + KV storage swap + HiCache NIXL + MXFP8 MoE |
| TensorRT-LLM | 19 | DSv4.1 kernel 四合一修复 + PrimTS MLA prefill + MoE tactics key + 若干 waive/infra |
| llama.cpp | 25 | `.scale` 统一自动应用 + OOB 三则 + SYCL/Vulkan 多后端 + zdnn 竞态 |
| **合计** | **146** | 周末正常低量；与上一窗口（11-20 的 338）连续，无空档 |

### 1.1 头号事件：vLLM 服务安全加固簇——整型溢出升维为跨用户数据泄露 + 认证绕过

周六 vLLM 提交量骤降至 28 条，但其中 **6 条为 `[Security]` 或认证边界修复**，构成本期最集中的一手信号：

- **vLLM `#53131` [Security] 把 int32 指针算术拓宽到 int64（官方一手，fix，安全漏洞修复，把第 9 条升维为第 24 条）**：CUDA 内建 `blockIdx.x` 为 32 位无符号，乘以 32 位 `hidden_size`/stride 后在 2^32 回绕，使 kernel block 读到/写到**同一 batch 内其他用户的 hidden state**。`hidden_size=8192` 时约 52.4 万 token 触发。覆盖 `rms_norm_kernel`、`rms_norm_static_fp8_quant_kernel`、`fused_add_rms_norm_static_fp8_quant_kernel`（Root cause 表标 `High — confirmed exploitable with PoC`）、MoE routing（`moeSoftmax`/`moeSigmoid`/`moeTopK`，Moderate）、AllSpark W8A16 GEMM（Low，纵深防御）。修法沿用 `451227cb3f` 既有模式：把 `blockIdx.x` 先转 `int64_t` 再乘。
- **vLLM `#52794` [Security] 修复认证中间件的 root path 匹配（官方一手，fix，认证绕过）**：`AuthenticationMiddleware` 用 `str.removeprefix(root_path)` 纯字符串剥离，当 `--root-path` 是被保护路由的**非段前缀**（如 `/v` 对 `/v1/models`）时，剥离得 `1/models` 不以 `/v1` 开头→跳过鉴权，而 Starlette 路由仍正常派发→`--api-key` bearer token 鉴权在 `/v1`、`/v2`、`/inference`、`/cohere` 全部路由族上**完全绕过**。修为改用 Starlette `get_route_path(scope)`（仅段边界剥离），加 9 项回归测试。
- **vLLM `#54721` / `#53473` / `#68249` 输入边界纵深加固（官方一手，fix，DoS 边界）**：`#54721` 补全 `#51447` 遗留的四处边界缺口——`stop_token_ids` 加 `VLLM_MAX_STOP_TOKEN_IDS`（默认 128）+ `frozenset` O(1) 查找、`stop` 字符串去重与上限、DeepSeek-V3.2 chat 编码器把 `role=="tool"` 的 O(k²) 反向扫描降为 O(1)、`bad_words` 加 Pydantic 类型与总数上限（159 测试通过）；`#53473` 把 `VLLM_MAX_COMPLETION_PROMPTS` 从 `/v1/completions` 扩到 `/score`/`/rerank`/`/pooling`/`/classify`，防单请求扇出任意数量 engine 子请求；`#68249` 让整个批处理 HTTP 请求共享一对媒体字节预算（pooling/scoring/embedding/batched chat，105 项聚焦测试通过）。三者均标注未认证客户端（默认 `vllm serve` 无 `--api-key`）可借以强制不成比例的 CPU/调度工作，**影响是降低服务可用性，不泄露数据、不执行代码**。
- **研判（观点）**：立本期**新增累计结论第 24 条**——**多租户推理服务的整型边界与输入边界已同时从「鲁棒性/静默正确性」升维为「安全隔离/认证」问题**。这与第 18 条（服务安全加固）同源但性质升级：第 9 条此前记录的长上下文 int 溢出致崩溃/静默错，在 `#53131` 中**首次带 confirmed-exploitable PoC 指向跨用户数据泄露**；`#52794` 则说明认证中间件与路由器的路径语义不一致本身就是可绕过面。实践含义：① 面向多租户的 serving 部署，int32 指针算术与 `str.removeprefix` 类字符串操作应按安全缺陷审查，而非仅按鲁棒性；② 这些修复多为社区/AI 辅助贡献（`Made with Cursor`、Codex 协助 rebase），`#53131` 的 test plan 中"已有 LayerNorm/MoE kernel 测试通过"三项仍未勾选（仅 pre-commit 勾选），合并后须关注上游 CI 对大 token 边界的实际覆盖；③ `#68249`/`#53473` 明示端到端验证 pending、GPU/RDMA 未跑，属"官方一手但验证未闭合"。

### 1.2 静默正确性铁证（跨 vLLM/SGLang/TRT-LLM/llama.cpp，强化第 9/19/22/23 条）

- **vLLM `#68600` [Bugfix] batch invariance 下手工 FlashInfer norm fusion（官方一手，fix，强化第 1/19 条）**：手工 all-reduce + RMSNorm 融合助手在 `VLLM_BATCH_INVARIANT=1` 时未按批大小无关语义切换，被 `VLLM_BATCH_INVARIANT=0` 的 fusion eligibility 误命中。2×B200、TP2、hidden 6144、BF16：无防御时单行输出对显式 all-reduce+norm 参考 **19.4% 元素不匹配（1193/6144，最大绝对差 0.03125）**；加 guard 后 2 passed，且批首/批尾目标行逐位一致。取自 `#66088` 独立评审，FlashInfer 0.7.2.post1（与仓库 pin 环境不同、未测 pin 环境）。
- **vLLM `#68739` [Bugfix][Multimodal] rope_2d DP vision 输出按空间网格定尺寸（官方一手，fix，强化第 19 条）**：`--mm-encoder-tp-mode data` 下 Kimi-K2.5/K2.6 视频 chunk 得到错误数量的视觉 embedding——DP 路径假设每帧各产 embedding，但 `rope_2d` 编码器把 chunk 内所有帧平均。真权重 1×B300：期望每 item `[448,252,112,448]`，main 给 `[448,1008,224,1792]`（4 帧 336×588 chunk 从 252 变 1008），修后与非 DP 路径逐位相等；同批图像也被错位（最大差 1.73）。修为仅对 `rope_2d` 按 h×w/factor 定尺寸，纯 host 侧、无新增同步。
- **vLLM `#69023` [Bugfix][DSv4.1] 从 layer 读 DSpark draft 的 native mega MoE flag（官方一手，fix，强化第 23 条语义合并冲突）**：`decode_backend=latency_specialized` + `speculative_config.method=dspark` 在 main 启动即 `AttributeError: 'DSparkDeepseekV4Model' has no attribute 'use_native_mega_moe'` 崩溃。根因是 `#61869`（DSpark draft 接 specialized decode）加的 guard 读 `backbone.use_native_mega_moe`，而约三小时后合并的 `#64061`（DeepGEMM MegaMoE）把该 flag 移到了 draft 的 MoE layer——**两个各自正确、相隔约 3 小时合并的 PR 产生语义接口错配**（与第 23 条同族）。4×GB300 上 main 每 TP rank 启动失败、修后 test passed 且端到端 1366 请求 30 分钟 0 错误；该测试因需 4×SM103 在 CI 被 skip。
- **SGLang `#51148` [Fix] block-scaled FP8/INT8 matmul 用 int64 row offset（官方一手，fix，强化第 9 条）**：`_w8a8_block_fp8_matmul` 与 INT8 变体用 int32 row offset 计算 `offs_am*stride_am`、`stride_cm*offs_cm`，当 A 的 `M*K` 或 C 的 `M*N` 达 2^31 元素时回绕、越界读写。实例：Qwen3-32B-FP8 TP1 gate_up 投影（N=51,200）约 4.19 万 token 即触发。修为加 `USE_INT64_OFFSETS` constexpr，仅当 A/C 达 2^31 元素时启用（其余形状仍编原 int32 核）。L4 上 FP8 A/C 越界、INT8 三项 main error→pass；Hopper/gfx1250 变体同一行改动但本环境未跑。
- **SGLang `#51256` [Fix] mem_cache 跨 static KV storage swap 携带 pending request-row reset（官方一手，fix，强化第 5/19 条）**：static KV 池上复用 request row 的 spec ReplaySSM cursor 是惰性重置的；当 draft 模型用私有 request table 时，`kv_storage_of` 新建 `StaticKVStorage`，`adopt_bindings` 未把 pending reset 标记带过去——若 draft forward 先跑，该行保留上一请求的 cursor。修为让 `adopt_bindings` 共享 `_row_resets`。新增"draft table forwards first"用例 main 失败（cursor 停在 5）、修后通过；相关 static-pool 套件 127 passed（H200）。
- **TRT-LLM `#22354` [fix] DSv4.1 kernel 四合一修复（官方一手但 CI 未闭合，强化第 9/22 条）**：`fusedCatFp4.cu`/`fusedCatFp8.cu` 的 `base_out` 改 `int64_t`（32 位大输出溢出）；`csa2/kernel.py` 用 `Int64(1)` 单位 stride 保持 >2 GiB 偏移并给 unmapped-row zero buffer；`blackwell/utils.py` 把 `griddepcontrol.wait`/`launch_dependents` 约束串改 `~{memory}`（影响**所有** PDL 调用点，与第 22 条同族）；`mhc_lowm.py` 对 16 字节未对齐操作数克隆。**局限（coderabbit + QA 明示）**：CI 早前 commit 为 UNSTABLE、后续 FAILURE（失败项未识别、当前 head 无状态）；64-bit offset 修复、PDL memory clobber、CSA2 stride/mapping 改动**均无直接测试覆盖**；精度/性能为作者报告、此处未独立验证；MoE 分支把 `x` 别名到 `shared` 未校验数据相同（评审提问）。
- **llama.cpp `#32669` 在 `build_moe_ffn` 自动应用 `.scale` 张量（官方一手，fix，强化第 6 条量化 scale 漏接）**：把 NVFP4 量化 scale 从"逐模型手工补线"改为在 `build_moe_ffn` 自动应用——正文列出 15+ 个历史 PR（#20506 Qwen3.5、#21971 Gemma4、#28000/#30945 DFlash、#31097 MiniMax-M3、#31449 Granite、#31657 Qwen2、#32241 Qwen2-VL、#32393 Qwen3-Next、#32657 Inkling2 等）均在补同类 scale。造 dummy GGUF dump logits 测试：scale 全为 1 时 1,722 次比较逐位一致；"main 错、本 PR 正确"558（无 bias）/589（有 bias）例；"main 对、本 PR 错"**0 例**。**遗留**：`kimi-k3/k3` 等模型直接调 `ggml_mul_mat`（不经 `build_lora_mm`），scale/LoRA 仍不应用，标为 follow-up。此为量化 scale 静默错误主线（第 6 条）从"逐个补丁"走向"框架统一收口"的重要节点。
- **llama.cpp `#32720` / `#32722` 两则 OOB（官方一手，fix，强化第 19 条）**：`#32720`（cont #32601）给 `hrm_text` 增加快捷检查，防最终 `n_slot` 回绕（回绕型 OOB）；`#32722` 修 Parakeet relative-pos shift 的越界张量 view——中间 `(n_frame,pos_window,n_head)` view 超出源张量边界触发 `GGML_ASSERT`，致 Nemotron-3-Nano-Omni 30B 音频投影在 DGX Spark 初始化失败，合并两个 view 为单个 `ggml_view_3d`、保留对角抽取 stride 语义（Grok Bot 辅助实现）。

### 1.3 本期其他结构性观察

- **KV/HiCache 健壮性与静默损坏（第 8 条）**：SGLang `#46540` 修 NIXL + O_DIRECT（默认开）下 GLM-5.x 非 4 KiB 整块的 host 页（KV 页 3,316,736 B、indexer 页 667,392 B）无法满足 4 KiB 对齐→fallback copy mode→kernel 每次 EINVAL、NIXL 留空文件、L3 要么无命中要么（旧 NIXL）恢复全零 KV；修为 staging slot 补齐到整块 + `page_first_direct` 用 4 KiB 倍数 stride 保住零拷贝（每页 +1,024 B KV/+256 B indexer）。`#51254` 新 backend 挂载时清除 storage existence beliefs（避免误信已存）；`#51252` 把 HiSparse 所有 allocator free 收归一条 `_free_rows()` 路径，修 `free_full()` 等漏释放 C4 slot 与 mapping 致 slot 泄漏（main `192 != 256`，35+123 passed，含与 #50633 同修复说明）。
- **PD 解耦 + 投机 + PP>1 的边界（第 5/14 条）**：SGLang `#47840` 修 HiSparse + 投机 + PD 解聚在 `--pp-size>1` prefill 首请求即崩——非末级 PP stage 无 draft runner（`num_draft_entries=0`），`send_kvcache` 据此推 host-backed/device-only 边界并抛 `ValueError`，杀掉该 stage 传输线程，**后续请求全部挂在 0 完成而 `/health` 仍 200**（GLM-5.3-FP8 TP8 88.5 GB/GPU>80 GB HBM 的生产拓扑必触发）；修为把接收端注册的 draft-entry 数随 mooncake wire 下发，旧 peer 行为不变。
- **默认值/后端切换的性能-精度权衡（第 3/16 条）**：vLLM `#62516` 让 DiffusionGemma 混合 causal/bidirectional batch 在 SM100 族走 FlashInfer TRTLLM-Gen 并成为 text-only（`--language-model-only`）默认（此前 B200 用户必须手动 `--attention-backend TRITON_ATTN`，FA4 在 SM90-only causal mask 下启动即失败）。1×B200：output tok/s/GPU **1.10–1.45×**（BF16 KV）/1.15–1.54×（FP8 KV），长输入 TTFT 16K 1.61×/64K 4.60×（FP8 KV），但 GPQA-Diamond 69.95（TRITON_ATTN）→68.69（BF16 KV）/69.67（FP8 KV）——**性能默认化伴随小幅精度回退**，作者报告。

---

## 2. vLLM：2026-11-21 提交（28 条，单页）—— 安全加固簇(#53131/#52794/#54721/#53473/#68249) + 静默正确性(#68600/#68739/#69023) + GDN MTP 核(#69065, 1.3–1.6×) + DiffusionGemma 默认(#62516, 1.10–1.54×) + Inkling2 SM12x(#69021)

### 2.1 `#53131` [Security] int32 指针算术拓宽到 int64 —— 见第 1.1 节（本期头号）

### 2.2 `#52794` / `#54721` / `#53473` / `#68249` 认证绕过与输入边界加固 —— 见第 1.1 节

### 2.3 `#68600` [Bugfix] batch invariance 下手工 FlashInfer norm fusion —— 见第 1.2 节

### 2.4 `#68739` [Bugfix][Multimodal] rope_2d DP vision 输出尺寸 —— 见第 1.2 节

### 2.5 `#69023` [Bugfix][DSv4.1] 从 layer 读 DSpark native mega MoE flag —— 见第 1.2 节

### 2.6 `#62516` [Attention][FlashInfer] DiffusionGemma 混合注意力上 TRTLLM-Gen 并为 SM100/SM103 text-only 默认 —— 见第 1.3 节

### 2.7 `#69065` [Kernel][GDN] 16-bit recurrent state 的融合 CUDA MTP decode 核（perf，bitwise 不变，强化第 12/14 条）

为 BF16/FP16 recurrent state 新增独立融合 GDN MTP decode 核，并让融合 CUDA 路径接受 FP16 state（main 上 `--mamba-ssm-cache-dtype float16` 会回退 Triton）。16 位下原核受 warp-shuffle reduction 限制而非 state 流量：**kernel 1.3–1.6×**、Qwen3.8-Flash-Next-FP8 **decode step 最高 +5.3%**。GB300、H=16/HV=48、MTP3、FP16 state：4 请求 16.9→12.7 µs（1.33×）、32 请求 69.2→46.5 µs（1.49×）、128 请求 241.1→163.4 µs（1.48×）；16 位达 5.8–6.3 TB/s（原 ~4.0），与 FP32 state 相当。端到端 Qwen3.8-Flash-Next-FP8 DEP4+DeepGEMM MegaMoE：每 engine 32 请求 step 16.92–16.98→16.07–16.11 ms（1.05×），1 请求走旧核（0.99×）。**输出与 state 对 240 例逐位一致、GSM8K 88.02%→87.79%**（描述为 unchanged 口径内微降）；核设计改编自 FlashInfer `cake_gdn`（flashinfer#4581，Apache-2.0，v0.7.3）。130 passed。

### 2.8 `#69021` [Perf][Model] Inkling2 在 SM12x 保留 TMA k_group_partials 路径（perf，bitwise 不变）

`#69028`（修 #69012 的 SM12x 启动崩溃）曾让 SM12x 绕开 TMA 核；本 PR 证明 64 宽 N tile + 2 stage（2×48 KiB，SM12x 共 101,376 B）下 TMA 路径仍可容纳且更快，故保留。GB10(sm_121)、hidden 4608、µs/call：N=256/2048 120→90、N=512/4096 407→290、N=512/16384 1547→1146。输出与 main 逐位相等（partial 顺序累加）；Hopper/SM100 不变。`test_moe.py` 103 passed/1 failed（该失败 main 上同样存在，为 `test_routing_quantizes_tokens_into_their_rows[200-nvfp4]`）。

### 2.9 其他要点（标题级 / 描述级）

- **模型/recipe**：`#69028` 新增 Inkling 2 Mini；`#1bdc437` Hy4-flash Cleanup；`#68921` 导出 ZAYA1 CCA prefill checkpoint；`#bf4a317` HiSparse 支持 HY4 Flash；`#93c590` 新增 `masked_token_ids` 禁止采样指定 token。
- **多模态**：`#68314` 修 encoder-only/`ec_producer` 实例在 Qwen3-VL + `--video-pruning-rate` 下启动崩溃（`torch.nonzero` meta 未实现），保留 marker token embedding 表（RTX 6000 Ada，1 failed→1 passed，6 项权重映射测试）；`#68831` Qwen3-Omni/ASR 音频塔 encoder CUDA graph；`#bf151b7` PIL 视频帧转 RGB；`#a25064c` rope_2d DP 视觉输出（与 #68739 同族函数、修不同 bug）；`#c2fb5c5` 不可读 embedding payload 返回 400。
- **构建/CI/文档**：`#68036` Rubin 用 PyTorch cu135 nightly 并记录 ptxas 13.5.37 SM107 workaround；`#25d7765` 同主题；`#69101` 新增 vLLM security review skill（文档）；`#68971` XPU 跳过不适用测试；`#68760` bench serve 加 decisions backends。

---

## 3. SGLang：2026-11-21 提交（74 条）—— int64 溢出(#51148) + mem_cache storage swap(#51256) + HiCache NIXL O_DIRECT(#46540)/hisparse free(#51252) + PD+投机 PP>1(#47840) + 采样 penalty 精确化(#52054) + MXFP8 Marlin MoE(#51936) + diffusion/调度大簇

### 3.1 `#51148` [Fix] block-scaled FP8/INT8 matmul 用 int64 row offset —— 见第 1.2 节

### 3.2 `#51256` / `#46540` / `#51254` / `#51252` mem_cache/HiCache/hisparse 健壮性 —— 见第 1.3 节

### 3.3 `#47840` [Fix] HiSparse spec PD transfer 在 PP>1 prefill —— 见第 1.3 节

### 3.4 `#52054` [Feat] verify 中逐行精确施加采样 penalty（feat，强化第 14 条投机正确性）

`#52035` 的后续：投机 verify 一次 forward 给同一请求的多个 draft 位置打分，旧实现所有 verify 行用同一 penalty（block 起始时的历史），深度 d 的行漏掉其路径上前 d 个 draft token，`min_new_tokens` 也差 d。新增 JIT verify kernel `sampling.verify_penalties`（CUDA sm_80+）：每行计 `counts[penalty_index_i]` + 该行路径上 in-vocab draft token（排除根、含自身），`min_new_tokens` 由 `lens+depth(r)` 门控；数值上 fp32 表计算、加性步后一次舍入到 logits dtype、与编译版 `apply_scaling_penalties` 同一 `div.full.f32` 除法，**逐行与非投机解码逐位相等**。替换 `#51903`，覆盖 EAGLE trees/NGRAM/DFlash/DSpark。`MEGAKERNEL_API_VERSION` 升 3；DFlash/DSpark verify 现按非投机顺序施加 custom logit processor→grammar mask→logit bias（bias 在 fp32 加）。规模较大的投机正确性收口。

### 3.5 `#51995` [Fix] 权重加载时预分配 trtllm-gen MXFP4 MoE workspace（fix，缓解第 3 条启动 OOM）

`test_kimi_k31_low_latency_b300.py` 在首个 ~16K-token prefill batch 间歇性 CUDA OOM：FlashInfer trtllm-gen MoE launcher 每次调用分配 scratch（其中 2.83 GiB 张量），发生在 KV sizing 已把内存交给 pool 之后，请求差 12–33 MB。修为 SM100 的 trtllm-gen MXFP4 MoE 层在权重加载时预留一份共享 scratch buffer 并作为 `workspace_buffer=` 传给每次 MXFP4 launch，先于 KV sizing 存在。**局限**：正文明示"尚未在 GPU 上运行、buffer 大小与 KV 预算影响未测、draft 状态"；CI Base ✓/AMD ✓、Extra ⊘（跳过）；属官方一手但验证未闭合。

### 3.6 `#51936` [Feat] Marlin 跑 MXFP8（E4M3 权重 + E8M0 group-32 scale）MoE experts（feat，强化第 12/13 条）

Marlin MoE 此前只跑 MXFP4（E2M1+E8M0）experts 配 BF16 激活；MXFP8 checkpoint（E4M3+E8M0 group-32）在无原生 MXFP8 MoE 核的 GPU 上无 W8A16 路径、无法服务。本 PR 让 Marlin MoE 核用 E8M0 scale 反量化 E4M3 权重（保留 dequant flop，因 E8M0 不像普通 FP8 Marlin scale 那样吸收 FP8 指数偏置）；把 scale dtype 纳入 dispatch key（8-bit→E8M0，仅 FP8 group-32 + BF16 激活接受；否则须 16-bit），无 flag。

### 3.7 `#44265` [Feat][AMD] dsv4 把 Mega-mHC prefill dispatch 到 gfx950 的 aiter（perf，opt-in）

`SGLANG_OPT_DSV41_MEGA_MHC_PREFILL` 此前只在 SM100（DeepGEMM）生效，gfx950 上该 flag 无效、`is_sm100` 为 false，MI355X 上的 DSv4.1-Flash prefill 从不调用 aiter 的 Mega-mHC（`flydsl_mega_mhc`，ROCm/aiter#6179）。修为 opt-in 默认关：gfx950 在 flag 置位、`hc_mult==4`、`hidden==5120` 且非 CP prefill 时启用，decode/verify 保持 fused boundary（不双重施加 post）。MI355X TP4 GSM8K（1,314 题）Mega-mHC off/on：0.972/0.969、0.970/0.971、AMD monokernel 0.973/0.975、0.973/0.974——**精度基本持平**，正文未给加速数字（性能声称未量化）。

### 3.8 `#51279` [Perf] paged-experts 在计算旁预拷贝下一层预测专家（perf，强化第 7/12 条）

捕获的 Paged Experts decode step 中，MoE 层路由后才把缺失专家 page-in，每次 page-in 都在关键路径。Qwen3-Next-80B-A3B-NVFP4 on RTX 5070 Ti（每层 ~100/512 专家驻留）：单请求 step page-in ~62 专家、占 10.2 ms 中的 2.4 ms。本 PR 用当前层输入跑下一层 router（一次 GEMV）预测其 top-k，在当前层计算时用侧流预拷贝（窄 grid：半 SM、≤32 blocks，约 -1.6% 带宽）。单请求 decode：Qwen3-Next **+12%**、Qwen3-30B-A3B-GPTQ-Int4 +2.8%（RTX 5070 Ti）/+7~13%（RTX 4070 Laptop，PCIe 4.0×8），GSM8K 不变。仅对连续 paged MoE 层且第二层用 plain softmax top-k 路由时启用（sigmoid/grouped/bias/多专家数跳过），无 flag。

### 3.9 其他要点（标题级 / 描述级）

- **调度/采样**：`#51957` 新增 load-scaled class boundary 与 token budget 的 shortest-prefill-first；`#50088` 把分块请求未 prefill 部分计入 total_tokens 负载；`#50955` 把 capped chunk 未完部分留在 prefill 预算；`#51398` 处理重复/重试的 memory release 与 resume 请求；`#52035`/`#d643847` per-request pool 在采样后立即计 penalty history；`#1222adc` 把 penaltylib 合并入单一 penalties 模块；`#b83d28a` 用 host 内存 LRU 约束编译 grammar cache（止 RSS 增长）；`#50859` 等待队列中止请求返回采样掩码。
- **性能/核**：`#af80e8c` H20 小 M block-FP8 GEMM 走 swapAB kernel；`#77c393a7` AMD qknorm 走融合 in-place kernel；`#b563de5` DSA metadata kernel batch size 改 runtime arg 避免逐 bs Triton 重编译；`#51348` deep_select opt-in 多波 cluster top-k（长尾行）；`#8396d60` Kimi-K3 共享专家 side-stream free 排在当前 stream；`#5b03592` reject_sampling 词表分块 block-verification accept；`#a8ac8659` beam 对 NVFP4 checkpoint 用 tiny FP32 router GEMM。
- **模型/PD/路由**：`#51937` Hunyuan-4-Flash 视觉语言支持；`#40afa8e6` K3.1 verify megakernel 服务自己的请求 bucket；`#f9847764` NPU DCP DSA extend 在搬运字节更少时 gather 前缀 KV；`#50243` diffusion 集成 FlashInfer Ulysses for Wan/MiniMax-H3；`#51946` rust-server 的 `/internal/peer` 仅服务本 launch 自己的 listener；`#46a931cc` sgl-router 在 API worker 持有 URL 时延后 discovery restore；`#aaa14413` sgl-router 新增运行时 worker 管理 API。
- **diffusion 大簇（约 25 条，周六主力，标题级）**：`#44651` 替换 FLUX.1/2 nightly 并加 FLUX.3 TP2 e2e；`#51563` cosmos3 仅主 CFG rank 解码；`#51954`/`#51962`/`#51971` media URL/内容不进错误信息与文件名（安全）；`#52030` Ulysses peer wait 超时则失败请求并限制 head-group pipeline 等待；`#51260` warmup 中 worker 死亡不报 ready；`#51783` 拒绝不匹配任何 LoRA 层的 adapter；`#51494` fp16 LoRA delta 在激活 dtype 计算；`#50602` Nomas VAE 融合有序均方归约；`#51825` MiniMax-H3 图像输出 GPU 上量化+LANCZOS 缩放；`#51965` 按时步集缓存 MiniMax-H3 block AdaLN 投影；`#51939` 并发取 MiniMax-H3 远端 condition 输入；`#37917` 按校准 warmup 记录规划组件驻留（2/4）。

---

## 4. TensorRT-LLM：2026-11-21 提交（19 条）—— DSv4.1 kernel 四合一(#22354) + sparse Top-K PDL 时序(#22588, ~27→23ms) + PrimTS MLA prefill/Rubin(#19395) + MoE tactics key(#22668) + FP32 logits cast(#21748, 1.7–6.4×) + FlashMLA SM107(#22695)

### 4.1 `#22354` [fix] DSv4.1 kernel：64-bit 寻址 / PDL asm ordering / mHC 边界对齐 / constant zeros —— 见第 1.2 节

### 4.2 `#22588` [fix] sparse decode Top-K 排在 Q projection 之后（fix，强化第 22 条 PDL/事件时序）

把 DSv4 decode Top-K 排到 query 分支的完成 event 之后（indexer 投影与 score 生成仍走辅助流，投影 dispatch 与 selection kernel 不变）：先入队 query 分支以便 consumer wait 前记录 event，decode 在 logits 后等待。GB300、DEP32/con180/MTP3 八节点 32 GPU、3 次交错重复：main mean 27.04/26.74/26.62 ms → 排序改动 23.34/23.28/23.26 ms，数值参考 21.83 ms；在 10% mean/median 阈值下三次 main 全被拒、三次改动全被接受。新增 event 边界测试（indexer/decode/MLA.forward_impl），旧序与缺 handoff 对照均按预期失败。移除对应 GB300 perf waiver。属第 22 条"依赖/事件同步次序"在 DSv4 sparse decode 的定量兑现（**性能收益来自时序修正而非数值变化**）。

### 4.3 `#19395` [feat] PrimTS MLA prefill + MTP/Helix 集成 + Rubin 支持（feat，描述级，51.9K 字符正文）

扩展 `PrimsTSFmha`：新增 MLA prefill、linear MTP、Helix 集成，并在 SM100/SM103/SM107 上自动选择 MLA generation。Kimi K3 全注意力层可在 context/decode 选 PrimTS，支持 BF16 或全 FP8 target KV cache。Dense PrimTS 在默认注册表中排在 `CuteDslMlaFmha` **之前**，由 `_is_perf_favorable()` 仅在 SM100/103/107 启用 generation-only MLA；context/混合 batch/标准注意力保留既有默认限制；显式 `TLLM_FMHA_LIBS=prims_ts` 只绕过性能启发式、不绕过可用性/正确性检查。支持 QK/V 维度 (128,128)/(192,128)/(256,256)，直接传对齐 strided FP16/BF16 V（去掉 packing copy），紧凑 base-2 LSE 供 partition/Helix 合并。**局限**：51.9K 字符正文为 auto-generated release-notes 风格，35 个 feature commit / 58 个改动文件的聚合栈；本 PR 为在 `fb0aee5` 上追加三个可独立评审的 follow-up、未 rebase 到新 main；性能/精度断言未在窗口内独立核验，按"官方一手（大型特性栈，描述级）"记录。

### 4.4 `#22668` [fix] TRTLLM-Gen MoE tactics 按 EP 布局/attention DP/dummy routing 设 key（fix，QA needs follow-up）

7 个 TRTLLM-Gen MoE runner 此前只按 kernel shape（`top_k`、intermediate size、`local_num_experts`、激活等）设 autotuner key，但 profiled bucket 的含义还取决于路由：attention DP 下 dummy 每个 top-k slot 落到本地专家 shard，否则散布到全部 `num_experts`（EP size 决定本地负载）；routing method 与 group（`routing_method_type`/`n_group`/`topk_group`）也影响路由。修为给 autotuner 加共享 routing key；`Bf16MxE2m1BlockScaleMoERunner` 用 tuple 替代 `hash()`。**局限（coderabbit）**：key 不含 `local_expert_offset`（专家 shard 复用 offset-0 条目）；QA 判 **needs follow-up**（routing method 与 group 参数未见覆盖）；dummy-distribution 环境变量改为只读一次并缓存（运行时改动不再生效）。

### 4.5 `#22363` [perf] DSv4.1 MoE decode at attention TP：FC1-only TRTLLM-Gen + CuTe DSL FC2 + one-cluster routing + dynamic EP（perf，非 bitwise + silent fallback 风险，强化第 16 条）

MoE decode 在 attention TP 下用 FC1-only TRTLLM-Gen runs + CuTe DSL FC2 + one-cluster routing + dynamic EP。**关键研判**：dynamic EP 改变 BF16 partial 分组与求和顺序，作者报告最大差为调用最大值的 **8.7e-3**（非 bitwise，下游须接受）；适用条件极窄（`moe_dynamic_ep` 置位、batch=1、attention TP 且 EP=2/MoE TP≥4、无 EPLB/DWDP、sqrt-softplus 路由、TRTLLM-Gen NVFP4、SM100/103）。`dsv4_route_one` 核首次 eager 调用用 NVRTC 编译，**NVRTC/import 失败仅记 warning 并标记不可用、静默回退旧核**（评审要求确认 warning 是否足够显眼以防静默性能回退）；`MoeLowMFc2Nvfp4` 构造签名新增无默认值 `zero_tail`（out-of-tree 调用方须改），`cute_fc2_tile_k` 返回类型改 `Optional[int]`（真值判断语义变化）。属"默认性能化 × 非 bitwise × 静默回退"复合风险样本。

### 4.6 `#21748` [perf] 向量化 Triton 核做 FP32 LM-head logits cast（perf，bit 相同）

`LogitsProcessor` 用 `Tensor.float()` 把 BF16 LM-head 输出拓宽到 FP32，在 [rows,vocab] logits 上仅约 2 TB/s（H200/B200/VR200 皆然）。本 PR 用 Triton 核替代，跑在接近显存速度，比 `Tensor.float()` **快 1.7–6.4×、结果逐位相同**。H200/B200/VR200 各 48 形状、256×163840 BF16：119→62 µs / 111→35 µs / 131→21 µs。核按元素数 `do_not_specialize` 只编一版，一次 warmup 即可、无 CUDA graph 捕获期编译风险。非连续/CPU/空张量/非 BF16-FP16/需梯度/torch.compile trace 仍走 `Tensor.float()`；无 env 开关、条件允许即常开。无 CUDA graph 时每调用多 7–22 µs CPU 开销（图重放不付）。端到端服务未测。

### 4.7 `#22695` [fix] 为 SM107-only 目标构建 FlashMLA（fix，部分覆盖恢复）

main 基线 `49f4b27` 上 `flash_mla/CMakeLists.txt` 的 supported/requested 架构交集对 SM107-only 请求为空，空架构文件阻止 `build_wheel.py` 打包 FlashMLA，而 `glm_kpool` sparse core 在该架构需要它。修为把 107 加入 CUDA 12.9+ admission list，并仅给 `adp-mtp3-fp8-kv` 参数在 SM107 加 7200 s 超时（冷启动 autotune 撞默认 3600 s）。修前三个精度 selector 全因"requires FlashMLA in this build"失败、修后全过（诊断运行、7200 s 上限）；**独立冷进程 3600 s 验证仍缺，属部分覆盖恢复**。

### 4.8 其他要点（标题级 / 描述级）

`#22547` 把独立多模态编码器移到 `ScheduledEncoderExecutor`；`#21825` 加速 QSA PrimTS indexer decode scorer；`#21113` 新增全局内存监控；`#22418` 首 chunk 前报告解聚流失败；`#22677` 关闭时释放 suffix automaton mask/staging buffer；`#22698` 从 response channel 读 early responses；`#22707` 恢复 B200/GB200 TEP4 精度覆盖；`#22427` 按 scope 整理测试指南并合并 waive 规则；`#21439` 刷新 B200 VisualGen perf sanity recipes；`#22724` 把 LTX-2 VisualGen worker NVFP4 GEMM 钉到 CUTLASS 并刷 golden。

---

## 5. llama.cpp：2026-11-21 提交（25 条）—— `.scale` 统一自动应用(#32669) + OOB 三则(#32720/#32722/#32754) + grammar int 溢出(#32551) + fastdiv UB(#32698) + zdnn 竞态(#32279) + SYCL/Vulkan 多后端

### 5.1 `#32669` 在 `build_moe_ffn` 自动应用 `.scale`（fix，量化 scale 主线框架化）—— 见第 1.2 节

### 5.2 `#32720` / `#32722` hrm_text n_slot 回绕与 Parakeet rel-pos OOB —— 见第 1.2 节

### 5.3 `#32279` ggml-zdnn 修 zTensor 加载竞态（fix，强化第 19 条时序/竞态）

`#31481` 为惰性加载权重张量引入竞态：多线程互相争抢并覆写已加载数据，在 `-lm none|dio` 下明显（默认 `-lm mmap` 走串行 `.set_tensor` 路径未暴露）。修为给 buffer 加 mutex，一次只让一个线程加载 zTensor，加载完成后其他线程见其已 stickify 而绕过。`ggml_zdnn_load_tensor_q` 调用峰值 886→224（Q8_0）。属资源生命周期/并发竞态类静默数据损坏（第 19 条）。

### 5.4 `#32551` llama-grammar 修整型溢出（fix，DoH；强化第 9 条）

修 `#32550`：解析畸形 grammar 时整型溢出可致 llama-server 挂起并耗尽内存（拒绝服务面）。正文极简（"Fixes #32550"），无 PoC/复现细节，按标题+简述级记录。

### 5.5 `#32698` CUDA fastdiv 遇 UB 即中止（fix，强化第 9 条边界）

`fastdiv` 目前允许 32 位移位（UB），仅在除以 >2^31 的整数时发生（实践中极罕见）。修为在 host 代码捕获并中止，避免潜在错误结果。作者自述由语言模型判断该处可在 host 检测（CSV 第 32575 那类数值问题则不可）。属防御性边界加固，影响面低。

### 5.6 `#32754` 修 clef 与 vllm-decision2 的 no_realloc（fix）

让二者兼容 no_realloc（关联 `#32708`）：clef 改为始终按最大尺寸构建 head、忽略未用行（多耗计算 buffer）。clef-flash-q4_K_M：Metal compute buffer 474→538 MiB（+64）、CPU 136→172 MiB（+36）；短 prompt 中位 162→163 ms、长 prompt 544→559 ms（约 +3%）。以少量内存/时延换 no_realloc 正确性。

### 5.7 其他要点（多后端 / 标题级）

- **SYCL**：`#32456` column-blocked gated_delta_net 核；`#31998` IQ4_NL/IQ4_XS 权重重排；`#31753` 融合 ADD/RMS_NORM/MUL 并向量化 rmsnorm；`#31428` Grouped XMX GEMM 避免 SLM staging；`#31109` 限制 CONV_TRANSPOSE_1D 输入迭代。
- **Vulkan/CPU**：`#31804` 改进 RDNA4 MMVQ（Q3_K/Q5_K/IQ4_XS）；`#32466` 因 AMD Windows 驱动 bug 展开 FA coopmat1 数组；`#32659` 修 tinyBLAS_PPC_Q0 的 nc 对齐；`#30846` 改进 `ggml_compute_forward_mul_mat_one_chunk` 访存模式。
- **图/其他**：`#32650` 移除 fused lightning indexer gating；`#32519` QSA 只 score ubatch 序列的 k-pool；`#690b87d8` inkling2_loop 在 scale pass 后共享 loop slots；`#32763` chat 在支持时传 `reasoning_effort=none`；`#32491` mtmd 支持 kyutai/hibiki 实时全双工；`#30967` 补 OpenVINO 后端文档。

---

## 6. 累计结论延续与新增（继承 1–23，新增第 24 条）

**累计结论（1–23 主线回顾，本期证据）**：① 调度器隐式阻塞的定量画像（本期 SGLang #51279 侧流预拷贝移除 page-in 关键路径 +12%、#52054 penalty H2D 收口）；② 发版窗口的双向性（本期无新版本，延续）；③ 默认值成为主要风险载体（本期 vLLM #62516 DiffusionGemma SM100 text-only 默认切换伴随 GPQA 69.95→68.69、TRT #22363 dynamic EP 极窄条件默认 opt-in）；④ 异构/量化后端静默错误高频（本期 TRT #22354 mHC 16B 对齐、SGLang #44265 AMD Mega-mHC、llama.cpp #32698 fastdiv UB）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（本期 SGLang #51256 draft 私有表 cursor 残留、#47840 HiSparse 投机 PD PP>1）；⑥ 量化路径静默错误（本期 llama.cpp #32669 `.scale` 框架化收口、SGLang #51936 MXFP8 Marlin）；⑦ 专家张量/路由归属与分页漏接（本期 SGLang #51279 paged-experts 预取、TRT #22668 MoE tactics key 按路由）；⑧ KV 卸载/HiCache/PD 健壮性（本期 SGLang #46540 NIXL O_DIRECT 页对齐、#51254/#51252 existence belief 与 hisparse free、#47840 PD 挂起但 /health 200）；⑨ 长上下文整型溢出（本期 **vLLM #53131 升维为安全漏洞、SGLang #51148 2^31 matmul 越界、llama.cpp #32551 grammar 溢出**，见第 24 条）；⑩ MXFP8 KV 正确性簇（本期 SGLang #51936 MXFP8 MoE 延续）；⑪ 新旗舰模型 CP/KV/性能一等公民（本期 vLLM #69026 HY4 Flash on HiSparse、SGLang #51937 Hunyuan-4-Flash VL、TRT #19395 PrimTS Rubin）；⑫ DeepGEMM MegaMoE 量化后端主路径（本期 vLLM #69065 GDN MTP 配 DeepGEMM MegaMoE E2E +5.3%）；⑬ FP8/低比特 KV 标准化（本期 SGLang #51936 MXFP8 Marlin W8A16 路径）；⑭ 投机解码验证路径硬化（本期 SGLang #52054 逐行精确 penalty、vLLM #69023 DSpark flag、#69065 GDN MTP 核）；⑮ 跨后端可移植性（本期 llama.cpp SYCL/Vulkan 大簇、TRT #22695 SM107 FlashMLA、vLLM #68036 Rubin cu135）；⑯ 新旗舰模型默认性能化（本期 vLLM #62516 DiffusionGemma 默认切换、TRT #22363 dynamic EP 非 bitwise）；⑰ 前端 Rust 化（本期 SGLang rust-server #51946/router #50242/#aaa14413，vLLM 本期无 Rust 簇）；⑱ 服务安全加固（本期 **vLLM 安全簇 6 条**，见第 24 条）；⑲ 非量化常规路径静默错误密度不亚于量化（本期 vLLM #68600 norm fusion 19.4% mismatch、#68739 vision 尺寸、SGLang #51256 cursor、TRT #22354 PDL、llama.cpp #32279 竞态）；⑳ 正式发版固化默认值风险 × 破坏性变更（本期无新版本，延续）；㉑ 在线权重热更新时序竞态（本期 SGLang #105aa2688 CUDA-IPC tensor 重建、延续）；㉒ PDL/异步依赖同步次序成静默/崩溃新子项（本期 TRT #22354 `~{memory}` 约束、#22588 sparse Top-K 后移 ~27→23 ms）；㉓ 语义合并冲突成静默正确性新子项（本期 **vLLM #69023**——`#61869` 与 `#64061` 相隔约 3 小时合并致 DSpark flag 接口错配、启动崩溃，再添铁证）。

**本期新增第 24 条**：**多租户推理服务的整型边界与输入边界已同时从「鲁棒性/静默正确性」升维为「安全隔离/认证」问题**。证据三支：① **跨用户数据泄露**——vLLM #53131 的 int32 指针算术溢出在 `hidden_size=8192`、~52.4 万 token 时回绕，使 kernel 读/写同 batch 内其他用户 hidden state，Root cause 表标 `High — confirmed exploitable with PoC`（这是本追踪首次出现带 PoC 的跨用户数据泄露一手）；② **认证绕过**——vLLM #52794 的 `str.removeprefix(root_path)` 与 Starlette 段边界语义不一致，使 `--api-key` 在所有受保护路由族上完全绕过；③ **资源扇出 DoS**——vLLM #54721/#53473/#68249 闭环未认证客户端借输入边界强制不成比例的 CPU/调度工作。判读：第 9 条此前记录的 int 溢出多致崩溃/静默错，本条把它与第 18 条（服务安全）交叉——**同一类整型边界缺陷在 serving 多租户场景即成为隔离漏洞**；部署暴露面（是否 `--api-key`、是否多租户共享、是否长上下文）决定其严重级别。验证限制：#53131 test plan 三项未勾选、#52794 为社区/AI 辅助贡献，合并后的上游 CI 覆盖须持续观察。

**本期结论**：新增累计条目 1 条（第 24 条）。周六低活跃但信号密度高——vLLM 28 条中 6 条安全修复，把"整型/输入边界"主题从正确性推进到安全；静默正确性铁证（第 1.2 节 7 条）以既有第 6/9/19/22/23 条归类，其中 TRT #22354（CI FAILURE/覆盖缺口）、SGLang #51995（未跑 GPU/draft）按"官方一手但验证未闭合"处理，不将作者报告当已验证事实。

---

## 7. 社区与视频资源（HN Algolia，2026-11-21 epoch 窗口 1795219200–1795305600）

- **检索口径**：`search_by_date` + `tags=story` + `numericFilters=created_at_i>1795219200,created_at_i<1795305600`（epoch 由 `date -u -j -f` 生成，避免硬编码落错年份）。关键词命中：vllm 16 / sglang 1 / paged attention 1 / llama.cpp 0 / TensorRT-LLM 0 / KV cache 0 / speculative decoding 0。
- **引擎相关（均二手线索/未核验）**：
  - `Hill-climbing vLLM and SGLang configs on a single A100`（2 分，twitter.com）——配置自动调优讨论，非官方更新。
  - `Kvdiff – Local proxy showing the byte breaking your LLM prompt cache`（1 分，github.com/flipcyde101/kvdiff）——本地 prompt-cache 命中诊断代理，仓库未在窗口内核验其引擎关联性。
  - `Show HN: Find and serve fast LLM config for your GPU`（2 分，github.com/Aagam-Bothara/rigtune）——GPU 配置查找工具。
  - `Tell HN: OpenRouter prompts to Minimax may have been read`（43 分）——第三方服务隐私披露，与本追踪的引擎/serving 无一手关联，仅记录社区高热度。
- **官方一手视频/文档**：窗口内未核验到可确认的官方发布视频；vLLM `#69101` 新增 "vLLM security review skill" 为本窗口唯一官方新增文档类资源（见 2.9 节）。
- **限制**：Reddit / YouTube / X 时间线未直接采集；HN 命中多为泛 LLM 话题（"Brain Doesn't Think in Sentences" 4 分、"I got banned from Lobsters for writing like an LLM" 9 分等），与推理引擎优化无一手关联，未纳入正文结论。

---

## 8. 数据质量与采集说明

- **一手来源**：vLLM / SGLang / TensorRT-LLM / llama.cpp 官方 commits API 与 releases API（均 HTTP 200，2026-11-21 UTC 窗口）；41 个关键 PR 正文经官方 issues API 抓取（markdown，官方一手，抓取时 API 配额 60→11，无降级）；无 PR 正文缺失。所有性能/精度数字均出自 PR 正文或发布说明原文，未推算或杜撰。
- **证据分级应用**：`官方一手` = 官方仓库/API/PR 正文；`官方一手但验证未闭合` = TRT #22354（CI FAILURE/覆盖缺口、作者报告）、SGLang #51995（未跑 GPU/draft）、TRT #22668（QA needs follow-up）、TRT #22695（部分覆盖恢复）；`描述级` = TRT #19395（51.9K 字符 auto-generated 特性栈、未独立核验）、llama.cpp #32551（正文极简）；`二手线索/未核验` = 第 7 节 HN 条目。凡"确认可利用 PoC""认证绕过"等均如实转述 PR 原文口径（vLLM #53131/#52794），未越界断言已在生产复现。
- **版本核验**：沿用 compare API 祖先关系方法（PR 的 `merge_commit_sha` 与 tag 做 `compare/{tag}...{sha}`，`behind, ahead_by=0` 为在版内、`diverged` 为不在）。本窗口无新版本，故未触发新的版本核验；按方法论，发布说明/PR 描述一律不作为"某修复已进版"依据。
- **连续性**：本期覆盖 2026-11-21（周六）窗口，与上一期 2026-11-21 档案（覆盖 11-20 周五）连续，无空档；自 2026-11-10 窗口起持续每日追踪。
- **工作区规范**：本期仅在 `hotspot/2026-11/` 新增本档案并刷新 `hotspot/README.md`；仓库既有未提交改动（`analysis/L5/ggml-gpu/` 下 5 个已修改 + `06-cuda-capabilities.md` 未跟踪）非本次产生，按归档规则未纳入提交。
