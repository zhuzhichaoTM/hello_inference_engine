# 大模型推理优化每日热点档案

> 档案日期：2026-11-03（本自动化于该日执行）
> 关注窗口：2026-11-02（周一）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + PR 正文（issues API，markdown）+ HN Algolia（HTTPS）+ 关键 PR 的 compare API 祖先核验
> 数据质量：四家提交 API 均 HTTP 200；PR 正文 19 篇已抓取；两个新版本（vLLM v0.33.0、SGLang v0.5.23）发布说明已抓取；HN 窗口内引擎相关无高信号故事（vllm 20 条为「LLM」模糊匹配、非引擎更新）。

## 1. 版本状态与本期头号事件：双版本发布日 + 静默正确性/整型溢出主线延续

**提交计数（2026-11-02 UTC，均 HTTP 200；2026-11-02 为周一，较上一期周日显著回升，无第二页）**：vLLM **90** / SGLang **93** / TensorRT-LLM **42** / llama.cpp **29**。

**发布状态（窗口内两个正式版本同日落地，为本期头号事件）**：
- **vLLM：v0.33.0 正式发布**（2026-11-02T10:10:05Z，以 tag `v0.33.0rc3` 发布，GitHub 标记 `prerelease=false`；**927 commits / 389 贡献者（122 新）**）。要点：升级到 **PyTorch 2.14.0 / torchvision 0.29.0 / Triton 3.8.0**；**CUDA 12.9 不再发布**，发布件仅 CUDA 13.0（`-cu129` 镜像停更，#62575）；新增 AWS EFA 镜像、短名 `vllm/vllm` 镜像别名、libtorch ABI 稳定 CUDA wheel；投机解码新增 XPress 精修器（block-diffusion 草稿 DFlash/DSpark，#54448）、FLy 有损验证策略（MI355X/B300 1.04–1.31× decode，#53987）、DFlash/DSpark 不再预留草稿槽位（DeepSeek-V4.1-Flash + DSpark GB300 1373→3390 output tok/s，#59468）、量化草稿 `lm_head`（并发 1 +18%，#59973）、lookahead block hashes 使 EAGLE/MTP 前缀命中保留末块（默认开，#61589）、Transformers 后端 MTP 草稿（#61270）；NVIDIA 解码优化（Kimi-K3 GB300 延迟专用解码 1.5–2× #61867、DeepSeek-V4.1-Flash batch1 解码步 −18.3% #61868、DSv4/V4.1 MXFP4 专家经 Humming 走 block FP8 在 Hopper +9–21% #61875、FA4 稀疏 MLA 解码 Blackwell #55866、FlashInfer DCP prefill trtllm-gen 100k 输入 TTFT 均值 −39~48% #62029、MRV2 prefill CP 下纯解码全 CUDA graph 2.6× #62377）；AMD ROCm 默认 10.0（7.2 仍发 `-rocm72`）、MonoKernel 解码 `--monokernel`（DSv4.1-Flash MI300X/MI325X P50 交互 2.32× #60731、MI355X 每层 #61082、MiniMax-M3-MXFP4 MI355X TPOT −16.6% #62712/#62994）、ROCM_OCTAVE 3–4 bit KV 缓存（#59774）、gfx1250 启用（#58800）；API 新增 `/v1/decisions`（默认开）、`/v1/embeddings` late chunking token embeddings、Rust 前端大对齐（长对话不再 500、不支持 flag 启动即报错、TP>1 并发 worker 启动、卡死 watchdog 栈 dump）；**破坏性变更**：移除 Medusa 与 MLP speculator（#61701）、移除 legacy Ray compiled-graph executor（#61712/#62228）、`--mamba-cache-mode` 废弃（#60043）、暂停时请求发 HTTP 503（#51488）、NIXL 需 ≥1.5（#61032）、LM Format Enforcer 与 `--ir-op-priority` 废弃（#63423/#63371）。
- **SGLang：v0.5.23 正式发布**（2026-11-02T22:14:10Z，**1007 PRs / 257 贡献者**）。要点：新增模型 Muse Spark Open、Beam-1.0、MiniMax-M3.1-Flash、MiniCPM-V 4.7、Kev、Cosmos3-Nano-Sim、FastH3 Omni、FLUX.3、HY-Nomas、MOSS-Image；DeepSeek-V4.1 在 GB300 负载下 prefill 吞吐最多 +1.5×、32K–64K 首 token 延迟 −28%（#44403/#45059/#45107）；低延迟模式（NVIDIA megakernel + AMD monokernel 插件）单用户 decode GB300 最多 +44%、MI350X +75%（#44945/#45450/#44353/#44621）；MoE 模型（DSv4.1-Flash、GLM-5.3-Flash、gpt-oss）8K prompt 首 token 早 30–58%（#43363）；Paged Experts（opt-in）单消费级 GPU 跑超过显存的 MoE，比 CPU offload 最多快 30×（#42557/#45787）；新增 Google TPU7x 支持（SGLang-TorchTPU）；量化 MoE（FP8/NVFP4）上的 LoRA 引擎（opt-in，#45892）；DeepSeek 式 MLA decode CP 的 prefix cache 可落 HiCache 文件型 L3（#41187）。
- TensorRT-LLM：仍为 **v1.4.0rc2**（2026-10-28T15:51:34Z，prerelease），窗口内无更新。
- llama.cpp：窗口内发布 nightly 标签 **b12054–b12061**（2026-11-02，均为 prerelease 自动构建）。

**本期头号事件（双版本发布 + 静默正确性/整型溢出主线在周一高活跃下延续）**：
1. **双版本同日发布，vLLM v0.33.0 含多项与往期热点直接呼应的落地**：本期发布说明把往期档案追踪的 FLy 有损验证（#53987）、ROCM_OCTAVE KV 缓存（#59774）、Kimi-K3 GB300 延迟专用解码（#61867）、FlashInfer DCP prefill（#62029）、DFlash/DSpark 去草稿槽位（#59468）等一并收口进正式版；同时以**移除 Medusa/MLP speculator、移除 legacy Ray compiled-graph executor** 等破坏性变更，印证「默认值/能力移除成为主要风险载体」（累计结论第 3 条）。
2. **整型溢出家族再落 TRT-LLM KDA prefill 与 vLLM RDNA W4A16（强化累计结论第 9 条）**：TRT-LLM `#21342` 的 Small-BH KDA prefill 把 state-row stride 编成 32 位常量，Kimi-K3 循环状态池 row 起点越过 2^31 元素（4 GiB BF16）即回绕、触发 `chunk_kda_fwd_k2` CUDA 非法访存（Xid 13/31）；vLLM `#64614` 的 RDNA 混合 W4A16 GEMM 用 int32 row offsets，M×K 或 M×N 越过 2^31 即 `HSA_STATUS_ERROR_MEMORY_FAULT` 挂死、load 返回映射垃圾使 127 行变 NaN（与 `#63049`/`#63054` 的 CDNA 同族，本 PR 为 gfx11/gfx12 兄弟）。至此「整型溢出静默/崩溃」已从 vLLM（Qwen3.5 #63761、Qwen4Exp #64820）、SGLang（extend attention #47029、aiter verify）、llama.cpp（Hexagon #31442）**扩展到 TRT-LLM KDA prefill 与 vLLM RDNA W4A16**。
3. **SGLang `#47030` 补全 aiter fp8 KV 的静默错误（闭环累计结论第 4/6/8 条）**：该 PR 修 `#46528`（2026-10-28 头号静默错误）遗留的 legacy target-verify 路径——aiter + fp8 KV 缓存 + 校准 scale 时，非 MLA 的 target verify（NGRAM/EAGLE）在 `extend_attention_fwd` 把 `k_scale`/`v_scale` 硬编为 1.0（#19736 为修 TypeError 引入），而缓存里存的是 `K/k_scale`，导致读回未反量化、投机解码产出垃圾；triton 后端在同处传 `layer.k_scale_float`，aiter 漏传。`#46528` 只修了 decode 与 NHD prefill 读取，未修此调用——`#47030` 用 `layer.k_scale_float`/`layer.v_scale_float` 补上。

---

## 2. vLLM：2026-11-02 提交（90 条）——v0.33.0 发布 + Extensible KV cache + SM100 索引器静默修复

### 2.1 `#56492` Extensible KV cache：按真实空闲显存定容，预留虚拟地址、warmup 后映射物理页（架构级，v0.33 特征）
- **主题**：把 KV cache 容量从「CUDA graph 捕获前的 profiling 估值」改为「捕获 + warmup 后按 GPU 真实空闲显存定容」。先用驱动虚拟内存 API 把整段地址区间预留为虚拟地址（无物理页），仅捕获与 warmup 实际触碰的 block 才背物理页；warmup 后各 worker 量真实空闲与瞬态峰值，再一步把最终 block 数提交（往同一预留区间映射更多物理页）。
- **内容详情**：地址不变，layer 视图与已捕获 CUDA graph 保持有效（cache 由测量而非估计定容）；不再依赖 `gpu_memory_utilization=0.92` 默认留 8% 吸收估计误差（约 15 GiB/192GiB 设备），估计过乐观时运行时仍 OOM；CUDA graph 内存不再被忽略或用一次性捕获估算（省启动时间）；覆盖 attention workspaces、sampler/logprobs buffer、spec-decode draft 等 profiling 不练的瞬态。基于 @zhuohan123 #47363 演示产品化，取代 #50779；**V2 model runner 上 CUDA/ROCm 默认开**。
- **应用场景**：大显存设备（如 192 GiB）的 KV cache 容量精度；减少启动开销与运行时 OOM。
- **研判**：【事实】PR 给出设计（VA 预留 + 物理页后映射、warmup 后定容、地址不变使 CUDA graph 有效）与产品化来源。【观点】属「KV cache 定容」主线（第 8 条同源）的基础设施级重构——把容量从估计改为测量，直接削弱「估计误差→运行时 OOM」的隐性风险；上线前建议核对 V2 runner 默认开启后的实际容量与稳定性。
- **来源**：vLLM PR [#56492](https://github.com/vllm-project/vllm/pull/56492)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含设计说明；v0.33.0 发布说明未单独列此条，按既定口径发布说明非包含证明）。

### 2.2 `#64806` SM100 解码索引器 logits 按 query-row 分块（静默正确性，SM100 长上下文）
- **主题**：在 SM100 把稀疏索引器的解码 logits 按 query-row 分块（每块至多 `VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB`，默认 2 GiB），而非一步为一个 step 的全部 decode row 生成一张张量；覆盖所有用该索引器的模型（DeepSeek-V3.2/V4/V4.1、GLM-5.x `GlmMoeDsaForCausalLM`、HY-V4），FP8（默认）或 MXFP4 索引器缓存均可。
- **内容详情**：稀疏索引器为每个 query row 给每个缓存 KV 位打分为取 top-k（DSv4/V4.1 为 512、V3.2/GLM-5.x 为 2048）；SM100 解码打分来自一次 DeepGEMM `fp8_fp4_paged_mqa_logits`，写出 `[decode rows, W]` 的稠密 BF16——decode rows = 请求数 ×（1+投机 token），长上下文下占用巨量显存。分块删除大部分 logits 显存、并解锁 `#63406` 的宽解码 CUDA graph 重构（否则宽图会持有完整 logits）。
- **应用场景**：SM100（Blackwell）上 DSv4/V4.1/GLM-5.x 的长上下文解码；解锁更宽解码 CUDA graph。
- **研判**：【事实】PR 给出根因（一步整张 logits 张量在长上下文显存爆 + 阻塞宽解码 graph 重构）与分块默认阈值。【观点】属「长上下文显存/静默正确性」（第 8、9 条同源）——未分块时显存随上下文线性膨胀直至失败；因 `#64806` 合入时间 23:48Z 晚于 v0.33.0rc3 发布 10:10Z 且 compare 三档点 `diverged`，**未确认随 v0.33.0 发布**，须以发行分支为准。
- **来源**：vLLM PR [#64806](https://github.com/vllm-project/vllm/pull/64806)（merge 2026-11-02T23:48:40Z）。
- **证据等级**：官方一手（含根因与默认阈值；compare 对 v0.33.0rc3 `diverged`，未确认在版内）。

### 2.3 `#59999` FlashInfer：只声明其能寻址的 KV 布局（静默正确性，启动期）
- **主题**：修 #59998：非 SM100 上让 FlashInfer 声明其能寻址的 5 种 KV 缓存布局，使布局解析不再选中 LHBNC。
- **内容详情**：复现路径——FlashInfer 在 SM80/SM90/SM120 配 `VLLM_KV_CACHE_LAYOUT=LHBNC` 时，`supported_kv_cache_layouts()` 返回 `None`，`resolve_kv_cache_layout` 接受 LHBNC；但 FlashInfer 只接受 NHD/HND 页、`get_flashinfer_layout_string` 没有 LHBNC 项，引擎在启动期 `AssertionError: KV cache layout LHBNC has no FlashInfer equivalent`（A100/Llama-3-8B）而非在解析期拒绝。修后返回不含 LHBNC 的默认偏好序（LBNHC/LBHNC/BLNHC/BLHNC/BHLNC）。默认布局（仍 LBNHC）、SM100 集合、connector 偏好均不变。
- **应用场景**：FlashInfer 后端在 SM80/SM90/SM120 的 KV 布局解析；防止非法布局拖到启动期才崩。
- **研判**：【事实】PR 给出复现步骤与修复（返回不含 LHBNC 的偏好序）。【观点】属「配置/布局静默错误」（第 3、4 条同源）——非法布局应在解析期拒而非启动期崩；轻量但典型的「错误时机」修复。
- **来源**：vLLM PR [#59999](https://github.com/vllm-project/vllm/pull/59999)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含复现与修复；与 #46329/#56550 同函数 SM120 分支不冲突）。

### 2.4 `#64472` 保留 mixed-precision `UniformTypeKVCacheSpecs` 组的 KV 清零（静默正确性，跨 PP）
- **主题**：混合精度 `UniformTypeKVCacheSpecs` 组在调度器归约时保留 KV 缓存清零决策；跨 worker 解析清零，使流水线各 stage 一致（即使仅一个 stage 为混合精度）。从 #63232（MiniMax-M3.1-Flash）拆出，独立于此 PR 组其余变更。
- **内容详情**：调度器投影保留「清零新分配 attention block」的决策；任一 worker 需要清零时所有 PP worker 均初始化清零，防止 MRV2 在「无清零者的 stage」收到清零工作；纯精度 attention-only 配置仍跳过清零。验证表：原 PR head 配置矩阵（uniform / mixed first stage / mixed second stage）为 1 passed/2 failed（`False is True` on a uniform worker），本修订 3 passed。
- **应用场景**：混合精度 KV 规格（如 NVFP4/FP8 混合）跨 PP 部署的 KV 清零一致性。
- **研判**：【事实】PR 给出根因（归约丢清零→uniform worker 上 `False is True`）与 3 passed 验证表。【观点】属「量化 KV 静默正确性」（第 4、6、8 条同源）——混合精度组跨 PP 的清零一致性是隐性精度债。
- **来源**：vLLM PR [#64472](https://github.com/vllm-project/vllm/pull/64472)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含配置矩阵验证表；split from #63232）。

### 2.5 `#64614` ROCm RDNA 混合 W4A16 GEMM 改用 64 位 row offsets（整型溢出，强化第 9 条）
- **主题**：`_triton_w4a16_skinny_fmt_kernel` 用 int32 row offsets 索引激活与输出，M×K 或 M×N 越过 2^31 即回绕——store 以 `HSA_STATUS_ERROR_MEMORY_FAULT` 失败并挂死、load 返回映射垃圾使 127 行变 NaN；与 `#63049`（#63054 修 CDNA 内核）同族，本 PR 为 gfx11/gfx12 兄弟。
- **内容详情**：把 `pid_m` cast 到 int64，加宽 `offs_m*stride_am` 与 `offs_m*N`（`offs_m` 是唯一随 M 缩放的偏移）。验证（R9700 gfx1201）：`(256,34816) M=61808` main 为 store 后 `Fatal Python error: Aborted` 然后挂死 → 本 PR 2 passed、最大 abs dev 0.023；`(17408,256) M=123489` main 出现 127 行 NaN（首行 123362）→ 本 PR 2 passed、全 M 无 NaN 行。
- **应用场景**：AMD RDNA（gfx11/gfx12）混合 W4A16 量化 GEMM 的大 M 场景。
- **研判**：【事实】PR 含 R9700 验证表（int32 在 M≈61.8K/123.5K 处 fault/127 NaN 行，int64 修复）。【观点】与 TRT-LLM `#21342`、SGLang `#47029`、vLLM `#64820`/`#63761`、llama.cpp `#31442` 同族，把「整型溢出静默/崩溃」主线（第 9 条）扩展到 **vLLM RDNA W4A16**；但 `#64614` 合入 04:06Z 早于发布 10:10Z 而 compare 三档点仍 `diverged`，按既定规则**未确认随 v0.33.0 发布**。
- **来源**：vLLM PR [#64614](https://github.com/vllm-project/vllm/pull/64614)（merge 2026-11-02T04:06:06Z）。
- **证据等级**：官方一手（含 R9700 验证表；compare 对 v0.33.0rc3 `diverged`，未确认在版内）。

### 2.6 `#64759` Kimi-K3：钉 FlashKDA/FlashMLA 到带 NaN 修复的 commit（静默正确性，跨请求 NaN 污染）
- **主题**：修 #64758：负载下部分 Kimi-K3 请求变成一连串 token 0（`!!!!`）直到 `max_tokens`。把 FlashKDA 与 FlashMLA 的 pin 移到带两个 kernel 修复的 commit（FlashKDA #16、FlashMLA #41，均已合入上游）。
- **内容详情**：FlashKDA prefill 不再把一个请求的 NaN 拷到同 batch 其他请求——注入一个 NaN 请求时，其他 decode 的 bad 从 96/240 降到 0/240；FlashMLA 的 SM90 dense decode kernel 在读 query 前等上一 kernel（PDL），强制竞争下 stock 50 次 49 次 NaN、修复后 0/50；自然突发复现器上 bad 请求从 14（stock）降到 0。无 measurable 端到端性能代价。
- **应用场景**：Kimi-K3 在负载下的解码正确性（跨请求 NaN 污染 + PDL 竞争）。
- **研判**：【事实】PR 含 96/240→0/240、49/50→0/50、14→0 三组对照。【观点】属「异构/外部 kernel 静默正确性」（第 4、8 条）——vendored kernel（FlashKDA/FlashMLA）的 NaN 跨请求污染与 PDL 竞争，是「不报错、精度静默塌方」高发区；建议 vendored kernel 钉版本纳入发布门禁。
- **来源**：vLLM PR [#64759](https://github.com/vllm-project/vllm/pull/64759)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含三组对照验证；pin 上游已合入 commit）。

### 2.7 `#58618` KV Offload：把 partial-tail 查找锚定在 full-attention 前缀（静默正确性，DCP）
- **主题**：从 #57216 拆出（part 1/2，lookup 半；DCP store 半为 #58619）。`_lookup` 把 partial-tail 查找锚定在 `_lookup_complete_chunks`，要求**每个** KV 缓存组（含 recurrent 组）在锚点边界都有完整 chunk。recurrent `align` 组只在调度步结束于某 block 边界处持有状态（实践中在 producer prompt 的最后完整 recurrent block 与 tail）。
- **内容详情**：uniform block size（dcp=1，当前 `supports_partial_tail` 唯一的形状）下，最后完整 block 边界也是 full-attention chunk 整数倍，对齐行落在 chunk 网格、`complete_hit` 够到它、现有单窗探针找到 tail；锚点仅在缺该行时失败（如被单独驱逐）。故在 `main` 今日属健壮性修复。mixed block size 下（DCP 的 full-attention offload block 为 `dcp ×` attention block）路径不同，本 PR 主要修 uniform 形状。
- **应用场景**：KV Offload 的 DCP partial-tail 查找健壮性（recurrent 组边界）。
- **研判**：【事实】PR 给出锚定逻辑与 uniform-block 下为何成功的推导、mixed-block 为后续。【观点】属「KV 卸载/一致性」（第 8 条同源）——partial-tail 查找漏 recurrent 组会静默失配；属健壮性补强。
- **来源**：vLLM PR [#58618](https://github.com/vllm-project/vllm/pull/58618)（2026-11-02 合入 main）。
- **证据等级**：官方一手（split from #57216，lookup 半）。

### 2.8 其余 vLLM 要点（官方一手，标题/正文级）
- `#65000` **Kernel：batch invariance 合并 NVFP4 linear kernel 覆盖**（#62192 的 follow-up，钉死 NVFP4 linear 的 batch 不变性）。证据等级：官方一手（follow-up）。
- `#62560` **[Core][MRV2] CUDA graph 捕获后 warm up sampler 以停内存增长**（负载下内存增长修复）。证据等级：官方一手（perf/robustness）。
- `#64815` **[Perf][MXFP8] DSv4.1 fusions 留在 torch._scaled_mm kernel**（保住融合）。证据等级：官方一手（perf）。
- `#64475` **[MiniMax-M3.1][P/D] NIXL KV-head re-sharding**（per-head-contiguous NVFP4 页）。证据等级：官方一手（PD）。
- `#63971` **[Bugfix][Mooncake] decode 复用 divergent local hybrid prefix hits**。证据等级：官方一手（prefix cache）。
- `#61950` **[Bugfix][ROCm] Kimi-K3 online quantization padded projections 修复**；`#65059` **[Bugfix][Kernel] 仅 SM90+ 给 MoE top-k gating 设 PDL launch attr**；`#64984` **[Bugfix][MM] Qwen3-VL encoder CUDA graph replay 对齐 FlashInfer cu_seqlens halves**；`#61219` **[Bugfix][Spec Decode] V2 rejection sampler 返回 logprob_token_ids 分数**。证据等级：官方一手（标题/正文级）。
- `#64616`/`#61267` **LoRA 修复**（reload 时 KeyError 崩溃、`attention_k_eq_v=True` 的 QKK LoRA）；`#64816` DSv4.1 首层 HC 扩展拷贝向量化；`#60543` DSv4.1 稀疏索引器 wq_b GEMM autotune；`#63232` FA4 单位 per-tensor scale 不传 FP8 descale；`#65118` Mistral skip_special_tokens 保留；`#64971` DeepSeek V4.1 `/v1/messages` tool_reference 修复；`#63302`/`#63103`/`#63137`/`#64265`/`#63974`/`#63901`/`#63870` **Rust 前端对齐**（abort_requests 校验、generate 流式 derender、重复 reasoning alias、取消时跑完 utility call、Python 文本拼接对齐、developer→system 映射、gRPC liveness/readiness 分离）；`#64585` **NIXL 升 1.5.2 并放弃 Rubin 源码构建**；`#65162` triton_kernels 改源码 tarball 避免 +400MB clone；`#62716` ROCm paged attention 实例化代码生成提速；`#60102` gfx942 sparse MLA prefill 路由 AITER pa_prefill_sparse；`#62270` gfx942 fp8 indexer cache；`#64833` CDNA3+ 才要 sparse attention 层；`#64444`/`#63988`/`#64300`/`#64819` **KV Offload/P2P 健壮性**（成功 store 发布到 lookup、reset 前 drain 未取 P2P store、finished 请求不漏失败 ID、注册失败断连 peer）；`#65211`/`#64788`/`#64272`/`#64174`/`#64597`–`#64599`/`#64355`/`#64181`/`#65184` **自动分片 CI 大规模入册**（约 60+ 测试步骤）；`#64838` `--mamba-cache-mode` 移除推迟到 v0.34.0；`#57678`/`#64692` model loader 钩子统一与权重派生张量原地 refill；`#64076` 清理死代码。证据等级：官方一手（标题/正文级为主）。

---

## 3. SGLang：2026-11-02 提交（93 条）——v0.5.23 发布 + aiter fp8 KV 静默错误闭环 + FP32 路由精度

### 3.1 `#47030` [Fix][AMD] aiter：legacy target-verify 用层的 KV scales（静默正确性，闭环 #46528，头号相关）
- **主题**：aiter 注意力后端 + fp8 KV 缓存 + 校准 KV scales 时，spec decode 在 target verify 走 legacy 路径（每个非 MLA 的 target verify——NGRAM/EAGLE——都走，除非 unified verify 适用：topk 1 + unified attention 开，即 SWA 池或 `SGLANG_USE_AITER_UNIFIED_ATTN=1`）产出垃圾；根因是 `AiterAttnBackend.forward_extend` 调 `extend_attention_fwd` 时把 `k_scale`/`v_scale` 硬编为 1.0（#19736 为修 missing-argument TypeError 引入），而缓存里存的是 `K/k_scale`，读回未反量化。`#46528` 修了 decode 与 NHD prefill 读取，未修此调用；triton 后端在同处传 `layer.k_scale_float`。
- **内容详情**：`aiter_backend.py` 在缓存为 fp8 且层有 scales 时传 `layer.k_scale_float`/`layer.v_scale_float`，否则 1.0；triton kernel 同处已传。`#46528` 修的是 decode 路径与 NHD prefill 读取，本 PR 补 legacy target-verify 的 extend 调用，闭合 aiter + fp8 KV + 投机解码的静默错误全路径。
- **应用场景**：AMD aiter 后端 + fp8 KV 缓存 + 校准 scale 的投机解码（NGRAM/EAGLE）正确性。
- **研判**：【事实】PR 给出根因（#19736 硬编 1.0）与修复（传 `layer.k_scale_float`），并指明 `#46528` 已修 decode/NHD prefill 而遗漏此路径。【观点】与 vLLM `#64614`、TRT-LLM `#21342` 同属发布日的静默正确性铁证，且**直接闭环 2026-10-28 头号静默错误 #46528（aiter fp8 KV + 默认后端 → GSM8K→0）**；强化累计结论第 4/6/8 条。但 `#47030` 合入 04:50Z 早于 v0.5.23 发布 22:14Z 而 compare 三档点 `diverged`，按既定规则**未确认随 v0.5.23 发布**，以发行分支为准。
- **来源**：SGLang PR [#47030](https://github.com/sgl-project/sglang/pull/47030)（merge 2026-11-02T04:50:48Z）。
- **证据等级**：官方一手（含根因与 #46528 对照；compare 对 v0.5.23 `diverged`，未确认在版内）。

### 3.2 `#38719` [Fix] MoE：跨模型 gate 保留 FP32 路由精度（静默正确性，路由，强化第 7 条）
- **主题**：多个 MoE 模型仍在模型特定边界丢弃路由精度：MiMo-V2/Ernie4/Kimi Linear 用 bf16 构造 checkpoint correction bias（而专家选择以 fp32 敏感的分数算术消费它）；Bailing MoE / Bailing MoE Linear 用 fp32 路由权重计算后，top-k 前把 logits cast 回 hidden dtype。LLaDA2 同路径已由 #31768 独立修，本 PR 不动。
- **内容详情**：让 MiMo-V2/Ernie4/Kimi Linear 的 `e_score_correction_bias` 保持 fp32；Bailing MoE / Bailing MoE Linear 的 gate GEMM 输出以路由权重 dtype 返回。改动限于 5 个模型文件。
- **应用场景**：MiMo-V2/Ernie4/Kimi Linear/Bailing MoE 系列的 MoE 路由精度。
- **研判**：【事实】PR 列出五模型文件改动与精度保持点，呼应 #38695 的独立模型修复。【观点】属「专家路由精度静默错误」（第 7 条）——bf16 构造 correction bias / fp32 logits 回 cast 是隐蔽的路由精度债；与 #47742（Kimi-K3 融合路由初值放置）同源。
- **来源**：SGLang PR [#38719](https://github.com/sgl-project/sglang/pull/38719)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含五模型文件改动；呼应 #38695）。

### 3.3 `#47742` [Fix] Kimi-K3：融合路由中尊重初始专家放置（路由正确性，强化第 7 条）
- **主题**：Kimi-K3 不暴露 `--init-expert-location` 所需元数据，其合并 gate/latent-projection 前端还把逻辑 top-k ID 包了一层、没有普通放置重映射或活跃 batch padding mask。配置的放置须与权重加载和运行时 dispatch 都一致。
- **内容详情**：为文本模型与多模态 wrapper 都暴露专家位置元数据，使现有 loader 能按初始排列放置 checkpoint 专家；合并前端之后把当前路由输入传给现有 top-k 后处理 helper；保留现有 capture/recorder（映射与记录融合时避免重复计数）；fake dispatch 仍在普通路由路径（top-k 前变换 logits）；拒绝在线 EPLB 与冗余专家（其权重布局 Kimi-K3 未实现）；复用现有 `--init-expert-location`/`--ep-dispatch` 路径。
- **应用场景**：Kimi-K3 的 EP/专家放置一致性与融合路由正确性。
- **研判**：【事实】PR 给出元数据暴露 + top-k 后处理接线的修复与拒绝项。【观点】属「专家张量/路由归属」（第 7 条）——融合路由漏放置重映射是隐蔽的路由错位；与 #38719 共构路由精度/一致性主线。
- **来源**：SGLang PR [#47742](https://github.com/sgl-project/sglang/pull/47742)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含接线与拒绝项）。

### 3.4 `#46606` [Fix] cache：分配抛错时回滚 host slots（HiCache 健壮性，第 8 条）
- **主题**：HiCache 可在另一池分配或回收回调抛错前，预留 primary KV 与一个 auxiliary 池；当前仅 `None` 结果被回滚，异常会留下已预留 host slots 却未排队备份。
- **内容详情**：回滚当前解析调用做出的 auxiliary 分配，再在 controller 边界释放 primary KV 预留；调用方提供的分配仍归调用方；成功排队的操作用其 slots。若释放回调也抛错，先 drain 其他预留再 re-raise 原分配错误（把清理错误作为原因）。`TestHostPoolGroup`：7 passed、28 deselected。
- **应用场景**：HiCache host 池组在部分分配失败时的 slot 一致性。
- **研判**：【事实】PR 给出回滚顺序与 7 passed 验证。【观点】属「KV 卸载/cache 健壮性」（第 8 条同源）——异常路径漏回滚会静默泄漏 host slots。
- **来源**：SGLang PR [#46606](https://github.com/sgl-project/sglang/pull/46606)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含 7 passed 验证）。

### 3.5 `#42968` [Perf] DeepSeek-V4.1：索引器 TP shard（性能，强化第 5 条）
- **主题**：DSv4.1-Flash 的 ratio-1/2 稀疏索引器在 attention-TP 组全复制：`wq_b`/`weights_proj` 为 `ReplicatedLinear`、`n_local_heads==n_heads`，每 TP rank 跑相同 dense `fp8_fp4_mqa_logits` 扫描与相同 top-k，3/4 工作冗余。该冗余是长 prefill 最大的 O(context²) 项——1M-token prefill 在 4×B200（TP4+EP4、DSPARK、cookbook 低延迟配方）下，dense index logits 每 rank **6.8 s / 2746 launches**，ragged top-k 再 **1.9 s**，且随长度上升。
- **内容详情**：在 `SGLANG_OPT_DSV4_INDEXER_TP_SHARD`（默认 **off**）后，让索引器在 TP 组分片。属 opt-in 性能路径。
- **应用场景**：DSv4.1-Flash 长 prefill 的索引器冗余消除（TP4）。
- **研判**：【事实】PR 给出 6.8s/1.9s 的 O(context²) 量化与默认 off 开关。【观点】属「索引器/PP-TP 交叉债务」（第 5 条）——与 vLLM `#59961`（DSv3.2 跨 PP indexer top-k）同源，是长上下文索引器算力的集中优化；opt-in 默认关，采用前测实际收益。
- **来源**：SGLang PR [#42968](https://github.com/sgl-project/sglang/pull/42968)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含 6.8s/1.9s 量化；opt-in 默认 off）。

### 3.6 其余 SGLang 要点（官方一手，标题/正文级）
- `#47731` **[Perf] imports：轻量模块脱离 HF patches/模型配置/调度器**（启动提速）；`#47949` **[Perf] dsv41：小 batch decode mHC post-combine-norm 行按 CTA cluster 拆分**；`#46088` **dsv41 融 decode mHC post+combine+RMSNorm 为一 kernel**；`#46082` **[Perf] trtllm_mha：大 batch 下 full-attention decode KV 跨 CTA 拆分（opt-in）**；`#47306` **[Perf] allreduce：Kimi-K3 自动启用 FlashInfer AllReduce Fusion**；`#47512` **[Perf] sampling：为 sampler/spec verify JIT softmax+argmax**；`#47784` **[Perf][NVIDIA] moe：重写 moe_finalize_fuse_shared 为全 top-k gathers in flight 的 JIT kernel**；`#47743` **[Perf] mla：padded head 计数用 masked K 拼接**；`#46478` **[Fix] DSpark FA drafts 用 ModelRunner query width**；`#47680` **[Fix] mem_cache：slots 耗尽时跳过 Mamba checkpoint donation**；`#47675`/`#47714` **memory-pool：MiniMax K-only / QSA indexer host pools 由 buffer info 构建（sidecar/hybrid）**；`#47234` **[Feat] attention：OOT linear-attention 后端注册**；`#47902`/`#44967`/`#47611`/`#47023` **[Test][NPU] DeepSeek_v4_pro / kimi_k3 / deepseek_v4_flash 性能与 gpqa 验收**；`#47574`/`#47373`/`#47211` **[NPU] EPLB 动态专家均衡 / mlaprolog for kimik3 / 量化工具与 DSpark 权重映射重构**；`#47429`/`#47844`/`#47951`/`#47791` **[Doc][AMD/NPU] cookbook 同步（MI355X monokernel、MiniMax-M3.1-Flash 高吞吐、Qwen3.8-Flash-Next、GB300 AgentX）**；`#47875` owner 更新；`#47804` 运行开关 `SGLANG_Q35MF_*` → `SGLANG_OPT_*` 重命名；`#46580`/`#39682`/`#43194` diffusion 模型（BRIA Fibo 2.0、HiDream-O1、cosmos3 bugs）与 `#47852`/`#46905`/`#46956` Hunyuan3D 加速。`#47737` PD 用共享 ZMQ context 做 staging prefetch socket；`#46334` 引擎暂停时仍发布 KV 事件；`#43901`/`#43903`/`#43902`/`#46654`/`#41771` 调度器重构（请求持有列表、PD prefill inflight 队列改名、无解码时跳过 `--prefill-decode-interval`、单 input_embeds 不拖垮 DP server）；`#44974` ep_scatter 避免 stale prefix reads；`#41768` n>1 时保留每图/视频/音频；`#43865` tokenizer_manager 防音频解码崩溃；`#44494` CPU 监控时钟重置处理。证据等级：官方一手（标题/正文级为主）。

---

## 4. TensorRT-LLM：2026-11-02 提交（42 条）——KDA prefill 整型溢出修复 + SM103 AdaLN + Helix KV 抢占

### 4.1 `#21342` KDA Small-BH prefill：64 位 state-pool row stride（整型溢出，强化第 9 条）
- **主题**：Small-BH KDA prefill（`kda_chunked_bt16_small_bh.py`，vendored FlashInfer kernel + TRT-LLM 来自 #20653 的兼容补丁）把索引的 state-row stride 编成常量，K2 kernel 以 32 位元素偏移寻址 state row。Kimi-K3 循环状态池是 KV-cache 内存的 strided view；row 起点越过 2^31 元素（4 GiB BF16）即回绕，kernel 在 `chunk_kda_fwd_k2` 读/写池外（CUDA 非法访存，Xid 13/31）。
- **内容详情**：把 row stride 声明为 64 位运行时值（`cute.sym_int64`，BT16 已如此），并刷新 `3rdparty/vendor_patches/flashinfer-kda-bt16.patch` 与其 lock digests（`scripts/vendor/manage.py patch flashinfer-kda-bt16 refresh`；stride 代码只在 TRT-LLM 补丁里，不在上游）。可达性：自 #20888 起，`trtllm::kda_prefill` 在 `auto`/`bt16` 模式不再选 Small-BH，故默认 Kimi-K3 服务今日不命中此 kernel；Small-BH 仍可显式选。
- **应用场景**：TRT-LLM KDA Small-BH prefill 的大状态池（Kimi-K3 循环状态）。
- **研判**：【事实】PR 给出根因（32 位 stride 回绕→Xid 13/31）、修复（`cute.sym_int64`）与可达性说明（默认不命中）。【观点】与 vLLM `#64614`/`#64820`/`#63761`、SGLang `#47029`、llama.cpp `#31442` 同族，把「整型溢出静默/崩溃」主线（第 9 条）**扩展到 TRT-LLM KDA prefill**——vendored kernel 的 int32 偏移是跨引擎共病；默认路径不命中，但显式选 Small-BH 时须纳入回归。
- **来源**：TRT-LLM PR [#21342](https://github.com/NVIDIA/TensorRT-LLM/pull/21342)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含根因与可达性；stride 仅在 TRT-LLM 补丁内）。

### 4.2 `#21396` 在 SM103（B300/GB300）启用融合 per-token AdaLN（异构硬件，GB300）
- **主题**：WAN VisualGen 的融合 per-token AdaLN（#17695）在两处被精确限制为 compute capability 10.0；B300/GB300（10.3）因此总走 unfused eager 路径。kernel 本身无 SM100 专属构造（只用 `thread_idx`/`block_idx`、shuffle、sync、向量拷贝，无 tcgen05/TMA/TMEM，`cute.compile` 按设备 JIT）。
- **内容详情**：`pertoken_adaln.py` 启动检查接受 (10,0) 与 (10,3)；`_pertoken_adaln_arch_ok` 在 `utils_wan.py` 同样放宽；仍排除 10.7，匹配 `is_blackwell_datacenter()`。`sol_attn/interface.py` 已用同样推理支持 SM103。
- **应用场景**：WAN（视频/图像生成）在 B300/GB300 的融合 per-token AdaLN 加速。
- **研判**：【事实】PR 给出门控放宽点与「无 SM100 专属构造」依据。【观点】属「异构硬件（GB300）低比特/融合加速」（第 4 条）——SM103 被误关融合路径是「能力默认关闭」类风险（第 3 条同源）；放开后 B300/GB300 可走融合路径。
- **来源**：TRT-LLM PR [#21396](https://github.com/NVIDIA/TensorRT-LLM/pull/21396)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含门控点与依据）。

### 4.3 `#21122` aggregated Helix 下 KV 压力抢占（调度正确性）
- **主题**：为 aggregated Helix 加 KV 压力处理：prefill 或 decode 拿不到足够 KV 页时，调度器在各 CP rank 上一致地 recompute-pause（抢占）该请求、稍后重跑其 prefill。各 rank 独立调度，每个决策只读 rank-invariant 输入；disaggregated Helix 与非 Helix 行为不变。
- **内容详情**：调度——aggregated Helix 跳过 suspend（单热 GPU 池不释放东西）直接抢占；context 路径只抢占请求者之后的请求，避免两请求互抢循环。Re-prefill——`reset_for_recompute` 重置 Helix decode counter 与 inactive-rank flag，使 owner rank 与 per-rank 长度重跑后正确。守卫——aggregated Helix 拒绝 host/disk tier 与 KV connector、跳过 auto host tier；Helix 下拒绝 KV pool rebalance。调试——`TLLM_HELIX_AGG_CHECK_SCHEDULE=1`（每 rank 设）比对各 rank 决策。
- **应用场景**：aggregated Helix 在 KV 压力下的抢占/重 prefill 一致性。
- **研判**：【事实】PR 给出抢占语义、re-prefill 重置与守卫项。【观点】属「调度/PD 一致性」（第 5、8 条同源）——KV 压力下的跨 CP rank 一致抢占是分布式正确性债。
- **来源**：TRT-LLM PR [#21122](https://github.com/NVIDIA/TensorRT-LLM/pull/21122)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含守卫与调试开关）。

### 4.4 `#21210` FlashInfer GDN chunked prefill 在 SM107 启用（描述级/未全绿）
- **主题**：prefill 架构 allow-list 加 SM107；decode 架构谓词不变，故 SM107 仍不支持 FlashInfer decode。prefill selector 仍需 `TLLM_USE_FLASHINFER_GDN_PREFILL=1` 与受支持架构；置 0 仍禁用。改动限于 prefill gating、相关测试期望与 CI 测试选择。
- **内容详情**：作者报 CI 通过但覆盖降低（L40S stage 暂停）；所报 H200 测试运行**未验证 SM107 执行**，SM107 运行仍为 follow-up。QA 指出架构谓词测试现期望 SM107 prefill 支持 + decode 排除，路由测试架构列表不变。
- **应用场景**：SM107（Blackwell）上 FlashInfer GDN chunked prefill（opt-in）。
- **研判**：【事实】PR 限 prefill gating，decode 仍排除；作者报 CI 通过但 L40S 暂停、SM107 运行待 follow-up。【观点】属「异构硬件（SM107）路径扩展」（第 4 条），但**性能/正确性数字为 pre-review、SM107 执行未验证，记描述级/未全绿**——采用前以最终 CI 与实测为准（与 10 月 TRT-LLM `#20694`/`#20755`/`#20815` 同类）。
- **来源**：TRT-LLM PR [#21210](https://github.com/NVIDIA/TensorRT-LLM/pull/21210)（2026-11-02 合入 main）。
- **证据等级**：官方一手（描述级，L40S 暂停、SM107 运行为 follow-up、QA 覆盖受限）。

### 4.5 其余 TRT-LLM 要点（官方一手，标题/正文级）
- `#21301` **[feat] MiniMax-H3 CPU offloading 支持**；`#21347` **[fix] 在 Ray 下从 torch EP group bootstrap NCCL EP**（nvbugs/6908161）；`#21121` **[fix] autotuning 时 prime MegaMoE tile variants**（nvbugs/6993575）；`#21339` **[chore] 通用 HF 权重映射器拒绝 per-layer KV-head 配置**；`#20049` **[chore] 移除 Python KV bounce buffer（`kv_cache_bounce_size_mb` 选 C++ agent buffer）**；`#20901` **[fix] 无 KV-cache runner 用 FlashInfer attention**（nvbugs/6858845）；`#20999` **[test] GLM-5.3 core 性能覆盖**；`#21002` **[fix] qwen3_32b_fp8 disagg 输出检查与启动预算**（nvbugs/6770977）；`#21390` **[docs] 重写 Modeling V2 开发者指南**；`#21314` **[test] fixed pools/shutdown 的 store driver conformance**；`#21271` **[perf] GVR V2 reg_clus hint-free 路径 prefix sample/suffix guard/bounded exact rescue**；`#21266` **[fix] 限 ZeroMqQueue close linger 防 teardown 挂死**；`#21371` **[test] 移除 MLA mixed-precision KV numeric test**；`#19907`/`#21106`/`#21384`/`#21411`/`#21413`/`#21412`/`#21231`/`#21225` 测试/infra；`#21415`/`#21477`/`#21476`/`#21446`/`#21398`/`#21402` **waive failed cases**（main pre/post-merge）。证据等级：官方一手（标题/正文级为主，多数为 chore/infra/test）。

---

## 5. llama.cpp：2026-11-02 提交（29 条）——unified kv 多序列优化 + Apple 内存回收 + Vulkan TP

### 5.1 `#29510` CUDA：flash_attn_ext_rows 更高效的 unified kv 多序列（KV 正确性/性能）
- **主题**：多序列推理下，unified kv-cache 因要从 mask 过滤掉其他序列 token 而付出大代价。本 PR 给 flash attention 加扩展（`flash_attn_ext_rows`）：后端支持时让 llama.cpp 发送 query 的优化表示——query 共享 kv rows 与其 mask + 新数据结构 `kv_rows` 做紧凑 token 的间接查找。
- **内容详情**：性能约等于非 unified kv（`-nkvu`）同时保留 unified kv 的好处。在 Qwen3.8-27B 与 Gemma-26BA4B 上测（RTX Pro 6000）：`llama-batched-bench` 多配置对比 master 与 patch 的 PP/TG 吞吐表（patch 在统一 kv 下接近非统一）。
- **应用场景**：llama.cpp 多序列（batch）场景的 unified kv-cache 效率。
- **研判**：【事实】PR 给出 `kv_rows` 设计与 Qwen3.8-27B/Gemma-26BA4B 的 bench 对比（patch ≈ 非 unified）。【观点】属「端侧/通用 KV 效率」（第 4 条端侧同源）——unified kv 的多序列代价是实用瓶颈，本 PR 用间接查找抵消；已含逐模型 bench 表，可采纳。
- **来源**：llama.cpp PR [#29510](https://github.com/ggml-org/llama.cpp/pull/29510)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含 Qwen3.8-27B/Gemma-26BA4B bench 表）。

### 5.2 `#31437` 所有 Apple 平台用 `vm_allocate` 替代 `posix_memalign`（Metal/Apple 内存回收）
- **主题**：iOS 27 上模型释放后 `phys_footprint` 仍含 ggml 刚释放的内存；macOS 27.0.1 的 Mac Catalyst 同态。`os_proc_available_memory()` 跟随 `phys_footprint`，故「加载下一模型前查可用内存」的 app 看到的比实际少。根因是 `ggml_aligned_malloc` 与 Metal host buffer 在所有 Apple 平台（macOS 除外）用 `posix_memalign`+`free`；iOS 27/macOS 27.0.1 的 libmalloc 把大释放块留在进程 footprint、稍后归还（非泄漏，内核在触限前回收）。macOS 上这些 buffer 已用 `vm_allocate`/`vm_deallocate`（#7078/#9875），`vm_deallocate` 立即归还页。
- **内容详情**：本 PR 在所有 Apple 平台用 `vm_allocate`/`vm_deallocate`；`ggml.c` 两处 `TARGET_OS_OSX` 检查改 `defined(__APPLE__)`；`ggml-metal-device.m` 仅 Apple 平台构建，其检查相应调整。
- **应用场景**：iOS 27 / macOS 27.0.1 上 llama.cpp 释放模型后的内存可用量准确性（Metal 后端）。
- **研判**：【事实】PR 给出 iOS 27/macOS 27.0.1 的 libmalloc 行为与 macOS 既有 `vm_allocate` 方案，并把其扩展到所有 Apple 平台。【观点】属「端侧/Apple 内存管理」（第 4 条端侧同源，非精度类）——影响「下一模型加载前可用内存」判断；属健壮性修复。
- **来源**：llama.cpp PR [#31437](https://github.com/ggml-org/llama.cpp/pull/31437)（2026-11-02 合入 main）。
- **证据等级**：官方一手（含 iOS 27/macOS 27.0.1 行为说明）。

### 5.3 `#25051` Vulkan：新增 tensor-parallel allreduce（多卡 TP 支持）
- **主题**：新增基础 Vulkan tensor parallel 支持：专用 allreduce kernel；用 spin-wait（NVIDIA，带最大时长）或 SFD semaphores（AMD）同步。目前仅优化 2 设备，ring-buffer 支持等后续 PR。
- **内容详情**：属 Vulkan 后端的 TP 基础能力起步（allreduce + 同步原语）。
- **应用场景**：Vulkan 后端在多 GPU 上的 tensor parallel 推理起步。
- **研判**：【事实】PR 给出 allreduce kernel + 两种同步原语与「仅 2 设备优化」范围。【观点】属「异构后端（Vulkan）多卡」（第 4 条端侧/异构同源）——TP 基础能力补强，目前仅 2 设备，属早期特征。
- **来源**：llama.cpp PR [#25051](https://github.com/ggml-org/llama.cpp/pull/25051)（2026-11-02 合入 main）。
- **证据等级**：官方一手（初始特征，仅 2 设备优化）。

### 5.4 其余 llama.cpp 要点（官方一手，标题/正文级）
- `#31546` s390x 构建文档；`#31429` mtmd Pocket tts 支持无 voice cloning 的 voices；`#31306` 新增 Clover 模型支持；`#31522` imatrix `--nextn` 崩溃修复；`#29730` context setters 简化；`#31487`/`#31530`/`#31463`/`#31534` 文档与 python 3.14 wheels；`#31249` CANN 后端 910B 支持 BF16 + `GGML_CANN_NO_PINNED`；`#31481` ggml-zdnn 与 host 共享 buffer；`#31441` opencl 加 `topk_mask`；`#30146` Vulkan RDNA3/4 小 n split-K 选择改进；`#31266` Vulkan 零化跳过行时避免冗余扫描；`#30956` SYCL Intel Arc Pro B70s 快速 `ggml_sycl_comm_exchange()`；`#31325` SYCL dmmv 清理；`#31276` clang-cl AVX512 BF16 构建修复；`#31414` cmake 显式 thread 链接；`#31422` WebGPU CI scratch 分配随 batch 单调；`#24033` jinja 实现 `map('filter')`；`#31304`/`#31305`/`#31312`/`#31314`/`#31326` UI 修复（steering 消息归属、拒绝删除、alert 样式、tool call 中断态、加载模型不选重载）。证据等级：官方一手（标题/正文级为主，多为 docs/build/ui/后端小修）。

---

## 6. 累计结论与研判（沿用 + 强化）

**沿用 2026-11-02 档案（覆盖 11-01）第 1–10 条：**
1. 调度器隐式阻塞的定量画像（host sync 移出热路径的 +4.5%~9.9% 画像）；
2. **发版窗口的双向性**：发版当天合入 main ≠ 在版内（compare API 祖先关系核验）；
3. **默认值/能力移除成为最主要的风险/收益载体**（默认后端/量化/移除 Medusa/MLP speculator/legacy Ray executor）；
4. **异构后端静默错误高频**（AMD ROCm / Metal / 异构 kernel / aiter 不报错只掉精度）；
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**（草稿/索引 KV 所有权一致性、索引器 TP/PP 分片）；
6. **量化路径静默错误**（scale 未应用/参数丢失/4D 量化漏算/舍入 1 ulp → 输出偏离且无报错）；
7. **专家张量/路由归属与分页漏接静默错误**（paged-experts 不被 paging、路由行覆盖、FP32 路由精度/放置 → 静默用错专家值）；
8. **静默正确性升维为跨引擎常态主线**（高频失效模式已从「崩溃」转为「不报错、精度静默掉/长上下文静默塌方」）；
9. **长上下文 / 大张量 int32 溢出为静默正确性新子项**（Triton 将 `<2^31` 整数参数按 int32 传递，Qwen3.5 #63761 在 ~209K token 回绕）；
10. **量化 KV 与 KV 卸载的正确性成为新集中暴露面**（MXFP8/FP8 KV 的 scale 初始化、host pool 钉页/卸载、L3 存储 round-trip）。

**本期强化（周一高活跃下出现铁证级延展）：**
- **第 9 条（int 溢出跨引擎）**：TRT-LLM `#21342`（KDA Small-BH prefill 32 位 state-row stride 回绕 → Xid 13/31）与 vLLM `#64614`（RDNA 混合 W4A16 int32 row offsets，`HSA_STATUS_ERROR_MEMORY_FAULT`、127 行 NaN）两处新实证，与 SGLang `#47029`（extend attention）、vLLM `#64820`/`#63761`（Qwen4Exp/Qwen3.5）、llama.cpp `#31442`（Hexagon）同族——**「整型溢出静默/崩溃」现已横跨 vLLM / SGLang / TRT-LLM / llama.cpp 四家引擎**，且覆盖云（Blackwell/SM100）、边（RDNA/ Hexagon）与 vendored kernel。边界从 ~209K token（Qwen3.5）细化到 2^31 元素（4 GiB BF16 / 8 heads×128 dim 的 2,097,152 token）。
- **第 4 / 6 / 8 条（异构/量化 KV 静默正确）**：SGLang `#47030` **闭环 2026-10-28 头号静默错误 #46528**（aiter + fp8 KV + 校准 scale 的 legacy target-verify 路径硬编 `k_scale=1.0` → 读回未反量化、投机产出垃圾），补全 decode/NHD prefill 之外的遗留调用；vLLM `#59999`（FlashInfer LHBNC 布局拖到启动期才崩）、`#64472`（mixed-precision KV 组归约丢清零 → `False is True`）、`#64759`（Kimi-K3 FlashKDA/FlashMLA 跨请求 NaN 污染 + PDL 竞争）从布局/清零/外部 kernel 三个维度补强。
- **第 3 条（默认值/能力移除风险）**：vLLM v0.33.0 **移除 Medusa 与 MLP speculator（#61701）、移除 legacy Ray compiled-graph executor（#61712/#62228）** 两项破坏性变更，把「能力移除 = 风险载体」从默认切换扩展到显式删减；SGLang `#47804` 运行开关 `SGLANG_Q35MF_*` → `SGLANG_OPT_*` 重命名亦属默认/开关面变更。
- **第 7 条（专家路由精度/放置）**：SGLang `#38719`（MiMo-V2/Ernie4/Kimi Linear 的 `e_score_correction_bias` 保持 fp32、Bailing MoE 路由 logits 不再回 cast）+ `#47742`（Kimi-K3 融合路由暴露专家位置元数据、尊重初始放置）从精度与放置两条线补强 MoE 路由静默正确性。
- **第 5 条（索引器/TP-PP 交叉债务）**：SGLang `#42968`（DSv4.1 索引器 TP shard，默认 off，长 prefill 6.8s+1.9s 的 O(context²) 冗余）与往期 vLLM `#59961`（DSv3.2 跨 PP indexer top-k）同源。
- **第 8 条（KV 卸载/cache 健壮性）**：SGLang `#46606`（HiCache 分配抛错回滚 host slots）+ vLLM `#58618`（DCP partial-tail 查找锚定 full-attention 前缀）补强 KV 一致性与异常路径。

---

## 7. 趋势判断与可执行清单

**趋势判断**
1. **双版本同日发布，vLLM v0.33.0 与 SGLang v0.5.23 把往期追踪的多条热点收口进正式版**：FLy 有损验证（#53987）、ROCM_OCTAVE KV（#59774）、Kimi-K3 GB300 延迟解码（#61867）、FlashInfer DCP prefill（#62029）、Paged Experts（#42557/#45787）、megakernel/monokernel 低延迟等——印证本雷达站「持续追踪 → 版本落地」的闭环价值；同时两版均含破坏性变更（vLLM 移 Medusa/MLP speculator、legacy Ray executor），须纳入升级门禁。
2. **整型溢出成为跨四引擎的共病**：本期 TRT-LLM `#21342` 与 vLLM `#64614` 把「int32 回绕」从云/边/投机解码扩展到 vendored kernel（FlashInfer-KDA、RDNA Triton），建议作为可执行的回归门禁阈值（第 9 条）。
3. **静默正确性主线从「崩溃」彻底转向「不报错的精度塌方」**：本期头号证据（#47030 aiter fp8 KV 闭环 #46528、#64759 跨请求 NaN 污染、#64472 混合精度丢清零、#59999 布局拖到启动期）均为「无报错、精度/行为静默错」，建议 serving 方把 vendored kernel 版本钉死、量化 KV 默认路径交叉验证纳入验收。
4. **异构硬件加速主线活跃但「描述级/未全绿」风险仍存**：TRT-LLM `#21396`（SM103 AdaLN）、`#21210`（SM107 GDN prefill，L40S 暂停、SM107 未验证）与往期 #46690–#46692（gfx950 MiniMax）、#20815（GB300 AgentX）同源——采用方须以最终态 CI 与实测为准。

**可执行清单（建议 serving / 研发方）**
- [ ] **int 溢出回归门禁**：把 `slot/stride/row offset >= 2^31`（约 2.1M tokens @ 8 heads×128、4 GiB BF16）纳入默认发布门禁，覆盖 vLLM `#64614`/Qwen4Exp `#64820`/#63761、SGLang `#47029`、`#21342`（TRT-LLM KDA）、llama.cpp `#31442`（第 9 条）。
- [ ] **aiter + fp8 KV 静默错误闭环验收**：SGLang `#47030`（补 #46528 的 legacy target-verify 路径）+ vLLM `#59999`/`#64472`/`#64759` 在启用 fp8 KV / 混合精度 / Kimi-K3 前交叉验证（第 4、6、8 条）。
- [ ] **v0.33.0 升级门禁**：vLLM v0.33.0 移除 Medusa/MLP speculator 与 legacy Ray executor、CUDA 12.9 停更、NIXL ≥1.5、mamba-cache-mode 废弃——升级前核对依赖与启动 flag（第 3 条）。
- [ ] **SGLang v0.5.23 评估**：GB300/MI350X 低延迟（megakernel/monokernel）、Paged Experts（#42557/#45787）、量化 MoE 上的 LoRA（#45892）、HiCache 文件型 L3（#41187）；`#46606` HiCache 异常路径回滚（第 5、8 条）。
- [ ] **MoE 路由精度/放置**：SGLang `#38719`（FP32 路由 bias）+ `#47742`（Kimi-K3 融合路由放置）上线前核对（第 7 条）。
- [ ] **llama.cpp 采纳**：`#29510`（unified kv 多序列 ≈ 非统一）、`#31437`（Apple 内存回收）、`#25051`（Vulkan TP 起步，仅 2 设备）按场景评估（第 4 条端侧）。
- [ ] **异构硬件加速验收**：TRT-LLM `#21396`（SM103 AdaLN）、`#21210`（SM107 GDN prefill，描述级/未全绿）以最终态 CI + 实测为准（第 4 条）。

---

## 8. 来源索引与数据质量

**官方一手（提交 API + 发布说明 + PR 正文 + compare API）**
- **vLLM v0.33.0**（tag `v0.33.0rc3`，2026-11-02T10:10:05Z，prerelease=False，927 commits/389 contributors）：https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3 ；**90 commits @2026-11-02**（均 HTTP 200）。PR `#56492`/`#64806`/`#59999`/`#64472`/`#64614`/`#64759`/`#58618`/`#65000`/`#62560`/`#64815`/`#64475`/`#63971`/`#61950`/`#65059`/`#64984`/`#61219`/`#64616`/`#61267` 正文已抓。compare 核验：`#64806`（合入 23:48Z，晚于发布，diverged）、`#64614`（合入 04:06Z，三档点 diverged）对 v0.33.0rc3 均**未在版内确认**（发布说明未列此二条，按既定口径发布说明非包含证明）。
- **SGLang v0.5.23**（2026-11-02T22:14:10Z，1007 PRs/257 contributors）：https://github.com/sgl-project/sglang/releases/tag/v0.5.23 ；**93 commits @2026-11-02**（均 HTTP 200）。PR `#47030`/`#38719`/`#47742`/`#46606`/`#42968` 正文已抓。compare 核验：`#47030`（合入 04:50Z，三档点 diverged）对 v0.5.23 **未在版内确认**（发布说明未单列此条）。
- **TensorRT-LLM**：https://github.com/NVIDIA/TensorRT-LLM （**42 commits @2026-11-02**；**v1.4.0rc2 2026-10-28T15:51:34Z** prerelease，窗口内无更新）；PR `#21342`/`#21396`/`#21122`/`#21210` 正文已抓。
- **llama.cpp**：https://github.com/ggml-org/llama.cpp （**29 commits @2026-11-02**；nightly **b12054–b12061** 2026-11-02，prerelease）；PR `#29510`/`#31437`/`#25051` 正文已抓。

**社区 / 外部（二手线索，未核验）**
- HN（2026-11-02 窗口，Algolia HTTPS，story 过滤）：引擎相关**无高信号官方更新故事**。`vllm` 20 条均为「LLM」模糊匹配（如「Teaching a Small LLM to Plan for Robots」「LLMs are not superhuman at reading and writing neural network weights」等，非 vLLM 引擎更新）；`sglang`/`llama.cpp`/`TensorRT-LLM`/`speculative decoding`/`LLM inference`/`inference optimization` 故事数均为 **0**；`KV cache` 1 条为「Garrul – Comments system」（非 KV cache 引擎技术）。均记为**二手线索/未核验**（非官方引擎更新），不计入结论。
- Reddit / YouTube / X：未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

**数据质量与限制**
- 计数口径：GitHub commit API 的 `since`/`until` UTC 日期窗口（vLLM 90 / SGLang 93 / TRT-LLM 42 / llama.cpp 29，均 HTTP 200）；按 commit date 过滤。**2026-11-02 为周一，较上一期周日（10/48/7/3）显著回升，无第二页**。
- 证据等级：带验证表 / 复现对照 / 跨请求 NaN 对照的 silent-error / 性能 PR 记为「官方一手（含验证）」；仅提交标题或正文未全抓的记为「标题级」；SGLang `#47030`、`#38719` 等记「官方一手（含根因/对照）」；TRT-LLM `#21210` 记「官方一手（描述级，L40S 暂停、SM107 运行为 follow-up、QA 覆盖受限）」；社区文章记「二手线索/未核验」。
- **版本包含核验说明（本任务既定方法）**：对发布当日合入的 `#64806`/`#64614`（vLLM）、`#47030`（SGLang）做 compare API 三档点核验，结果为 `diverged`（未见 `status=behind, ahead_by=0`），故**按既定规则记为「未在版内确认」**——与「发布说明非包含证明」口径一致；三者均于 2026-11-02（发布当日）合入 main，是否随版发布以发行分支与最终发布说明为准。
- 未调用外部浏览器（agent-browser）：本期数据源（GitHub API / releases / HN Algolia / PR issues）均可直接 HTTP 访问，沿用 2026-10-18 起确立的「API + PR 正文直抓」方法，未触发浏览器需求（与历次一致）。
- `find-skills`：经 SkillHub 语义检索（含「LLM inference GitHub commit monitoring」「github trending hotspot daily digest」「git sync commit push」「大模型推理 GitHub 提交采集 热点归档 仓库同步」）最高相关性约 0.050，返回编程专家.Skill / dev-expert / 育儿大师 / 技能发现·Find Skills 等通用编程/生产力类技能，**与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，未安装**（与历次一致）；本地 marketplace 目录为空。

> **归档空档说明**：本档案覆盖 **2026-11-02** UTC 窗口；**2026-10-29** UTC 窗口因 2026-10-30 当日自动化未执行而无档案（即 2026-10-29 档案覆盖 10-28、2026-10-31 档案覆盖 10-30、2026-11-01 档案覆盖 10-31、2026-11-02 档案覆盖 11-01、本期 2026-11-03 档案覆盖 11-02 之间，仅 2026-10-29 窗口一日空档）；**自 2026-10-30 窗口起已连续四期（2026-10-31、2026-11-01、2026-11-02、2026-11-03 四期档案分别覆盖 10-30、10-31、11-01、11-02）无空档**。本目录不补写中间日期内容，各期仅按自身关注窗口采集。如需回溯 2026-10-29 窗口，请以各引擎官方提交 API 按日期区间重新采集。
> 本档案仅覆盖 2026-11-02 UTC 窗口；相邻日期请以官方提交 API 按日期区间重新采集。
