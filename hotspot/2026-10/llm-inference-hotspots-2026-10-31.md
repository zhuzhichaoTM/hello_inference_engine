# 大模型推理优化每日热点档案

> 档案日期：2026-10-31（本自动化于该日执行）
> 关注窗口：2026-10-30（周四）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + PR 正文（issues API，markdown）+ HN Algolia（HTTPS）
> 数据质量：四家提交 API 均 HTTP 200；PR 正文 30 篇已抓取；HN 窗口内无引擎相关高信号故事。

## 1. 版本状态与本期头号事件：双主线延续（静默正确性 × 默认值风险）+ 一项能力解锁

**提交计数（2026-10-30 UTC，均 HTTP 200）**：vLLM **97** / SGLang **93** / TensorRT-LLM **48** / llama.cpp **14**。

**发布状态**：
- vLLM：最新正式版仍为 **v0.32.0**（2026-10-18T06:50:48Z），窗口内无新正式版本；nightly 持续。
- SGLang：最新正式版 **v0.5.22**（2026-10-19T02:55:19Z）；窗口内 `#47419` 将文档安装版本号 bump 至 **0.5.23**（说明 0.5.23 已在路上，但窗口内无新 tag）。
- TensorRT-LLM：最新仍为 **v1.4.0rc2**（2026-10-28T15:51:34Z，prerelease，wheel 1.4.0rc2.post1），发布于窗口前一日，窗口内无更新。
- llama.cpp：窗口内发布 nightly 标签 **b12004–b12017**（2026-10-30，均为 prerelease 自动构建）。

**本期头号事件（双主线延续 + 能力解锁）**：
1. **「静默正确性」新增两处教科书级证据**：
   - **长上下文 int32 溢出**（vLLM `#63761`）：Qwen3.5 的多个 Triton 注意力/GDN prefill 核在 `token*stride >= 2^31` 时 int32 回绕，触发错误输出或非法访存；最早在 ~209,717-token 的启动 profile 即命中。这是「不报错、长上下文静默塌方」的又一实例。
   - **AMD aiter 默认后端静默错误**（SGLang `#46833` / `#45486`）：aiter 后端（ROCm 上 MHA 模型的默认）在 `softmax_scale != head_dim^-0.5` 时整段输出垃圾（Granite 3.3-2B 把「法国首都是」接成乱码），CT FP8 的 Triton 布局还会静默丢掉 128 个 reduction 元素导致错误输出。与 10-28 `#46528`、10-29 `#46983` 同族——**AMD 默认 aiter 后端仍是静默错误高发区**。
2. **「默认值风险」继续兑现**：TRT-LLM `#19379` 把 `num_serve_frontends` 默认 `1→8`（行为变更默认）；`#21041` 移除 TensorRT 后端兼容面（BREAKING API）；`#20888` 修复 **Kimi K3 的 NVFP4 KV 在 SM107 被静默降级为 FP8**——「静默降级」本身也是默认值风险的一种。
3. **能力解锁（非错误）**：TRT-LLM `#20888` 让 Kimi K3（混合 KDA+MLA）真正可用 FP4 MLA KV；SGLang `#46456` 把 paged-experts 的锁定内存浪费砍掉（402 MB 张量不再被向上取整到 512 MiB）。

---

## 2. vLLM：2026-10-30 提交（97 条）——长上下文静默溢出 + DSA/投机/MoE 性能簇

### 2.1 `#63761` Qwen3.5 注意力与 GDN prefill 核的 int32 token offset 溢出（静默正确性，头号）
- **主题**：多个 Triton 核将 int32 token 索引乘以行 stride 后加指针；Triton 把 `<2^31` 的整数参数按 int32 传递，一旦 `token*stride >= 2^31` 即回绕，产生错误输出或非法访存。
- **内容详情**：逐核修复，每核带回归测试。触发阈值（Qwen3.5-9B 除非注明）：
  - `_fused_qk_rmsnorm_rope_gate_kernel`：token ≥ 209,716（qkv row stride 10,240）→ 209,717-token step，启动 profile 已命中；
  - `store_cache_checkpoints_kernel`（Mamba）：token ≥ 174,763（qkvz stride 12,288）；
  - `_fused_post_conv_kernel`（GDN）：token ≥ 262,144（conv row stride 8,192）；
  - `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`：H=64 时后段序列起点 ≥ 131,072；
  - `solve_tril` merge 核：H=64 时行超过 524,288。
- **应用场景**：长上下文（≥~200K token）Qwen3.5 prefill/recurrent 路径；serving 方长文档/agent 多轮易踩。
- **研判**：【事实】PR 给出逐核触发阈值与每核回归测试，属已定位并已修。【观点】长上下文回归门禁应把 int32 边界（~200K–524K token）纳入，否则静默错误只在高负载长序列暴露。
- **来源**：vLLM PR [#63761](https://github.com/vllm-project/vllm/pull/63761)（fixes #63759，merge 2026-10-30）。
- **证据等级**：官方一手（含逐核回归测试）。

### 2.2 `#60867` MoRIIO 混合 READ（ROCm，GLM-5.3-Flash）——静默跳过循环态传输的修复
- **主题**：GLM-5.3-Flash 在 `main` 无法跑 MoRIIO 混合 READ P/D；其有两个可传输注意力组（MLA+indexer、kpool tail ring）+ KDA 组 + MTP，`main` 只允许一个注意力组且仅 DSpark，并用 packed layout 把 KDA 组误判为注意力、静默跳过循环态传输。
- **内容详情**：connector-only 改动（`moriio_connector.py`+测试+1 行文档）。Hybrid READ 载荷为 `[*attn_groups,*mamba_groups]`，与单组时字节一致；`_align_read_blocks` 逐注意力组运行；非前缀可缓存组即便本地全命中也读取；每个混合层路由到自有载荷组；把「仅 DSpark」检查改为「drafter 共享 target 的注意力组」（MTP/DSpark）。依赖 `#58585`（已合）。
- **应用场景**：ROCm 上 GLM-5.3-Flash 的混合 P/D 传输；多注意力组 + 循环态模型。
- **研判**：【事实】PR 声明非混合模型不触达改动代码、单组 READ 有字节一致单测。【观点】属「连接器层静默跳过循环态」类修复，与跨引擎量化/MoE 静默错误同源，应在混合模型 P/D 上加回归。
- **来源**：vLLM PR [#60867](https://github.com/vllm-project/vllm/pull/60867)。
- **证据等级**：官方一手（connector-only，含字节一致单测）。

### 2.3 `#54307` Mooncake Store 混合 KV 的异构 TP 共享
- **主题**：把 `#53129` 引入的异构 TP Store 布局从单一 Full-Attention 缓存组扩展到统一的混合 KV 缓存分配。
- **内容详情**：本地 TP 能整除公共 Store TP 的 vLLM 实例可通过 Mooncake Store 共享同一 Attention KV 与 Mamba/GDN 状态；混合 KV 的三类挑战（Attention 按 token 增长 vs Mamba/GDN 整块 checkpoint 不同 TP 复制规则、本地页大小随 TP 变、prefix 复用需各缓存组边界求交）。依赖 `#53129`，合并顺序固定。
- **应用场景**：跨 TP 尺寸的 PD/共享前缀部署；多实例 KV 复用。
- **研判**：【事实】PR 描述混合 Store 布局与命名空间设计。【观点】异构 TP 共享是 PD 解耦落地的关键基建，但依赖链（`#53129`）未合前不在版内，需跟踪。
- **来源**：vLLM PR [#54307](https://github.com/vllm-project/vllm/pull/54307)（depends #53129）。
- **证据等级**：官方一手（PR 正文，依赖未合）。

### 2.4 性能 / 量化 / 投机解码簇（官方一手，含验证或声明）
- `#64612` **GLM5.3 dlfash fp8 kv cache draft model，每 rank 省 2.85 GB 显存**（标题/PR 声明；含 `vllm bench serve` 脚本与 Serving Benchmark 结果，完整数字在截断处，2.85 GB/rank 为 PR 明确声明）。证据等级：官方一手（含 bench 脚本）。
- `#64467` **MegaMoE 从 1024 token 起用 32-row tiles**：`prepare_megamoe_inputs` 在 GB200 上 1.3–1.6× 更快且 bit-identical；DSv4.1-Flash DEP2 prefill 每 forward 省 0.27–0.57 ms（8K–16K 新 token，约 0.3%）；含逐 token 表（512→32768 加速 1.27–1.64×）与 `vllm bench forward` GPU us 表。证据等级：官方一手（GB200 CUPTI，bit-identical）。
- `#64509` **Qwen3.5（NVIDIA-only）融合 norm+quant 与 gate+quant**：声明 **1.7–2.0× 更快**（截断处可见）；覆盖 `nvidia/Qwen3.8-27B-NVFP4`（4 个融合核，192 quant kernels/step，运行时验证）、`Qwen3.8-2.4T-A95B-NVFP4` 等；`Qwen3.5-*-FP8` 官方权重量化不匹配则不启用。证据等级：官方一手（first-party census 表 + 声明）。
- `#64138` **DSA 在 persistent top-k 直接读 BF16 logits**（SM100 decode，无 DCP）：移除整块 FP32 logits 分配；与 `#62172`（cooperative BF16）互补。证据等级：官方一手（WoosukKwon merge，含性能拆解）。
- `#63205` **MRv2 投机解码：声明式 speculator 配置类 + 自定义 speculator 插件 API**（`register_speculative_method`，调度进程不再 import 实现）。证据等级：官方一手（PR 正文）。
- `#63886` DSpark batch-sharded sequential block；`#57770` DSpark greedy draft 的 split-vocabulary Triton argmax；`#63219` 结构化输出下 PP 的 per-request draft retention。证据等级：官方一手（标题+正文，投机解码簇）。
- `#63054` Muse Spark Open 的 CUDA routing kernel；`#64004` ROCm Kimi-K3 削减 gfx942 MXFP4→int4 requant（纯 TP）；`#61971` DSv4.1 FlashMLA 传 int64 positions 给 MegaAttn；`#62007` ROCm DeepSeek-V3 NIXL 精度步骤；`#64494` `silu_and_mul_nvfp4_quant` 加 size-gated PDL；`#59958` Qwen4Exp PLE 磁盘卸载；`#64110` Qwen4Exp 移除 FlashAttention 依赖并统一 QSA 命名；`#62979` HiSparse 准入时为在途导入预留 host 页；`#63249` KV Offload 保持存储事件前缀顺序；`#62613` KV Offload 为循环态组发自描述事件；`#63005` 前端 run-batch TypeAdapters 仅构建一次；`#62111` 修复 Qwen3.5/Next 的 MTP 多层模块选择；`#63774`/`#64610`/`#64727` AITER 版本与 CI 更新。证据等级：官方一手（标题级为主，部分含正文）。

---

## 3. SGLang：2026-10-30 提交（93 条）——AMD/aiter 静默错误 + KV/PD/HiCache 调度优化

### 3.1 `#46833` AMD aiter 后端 softmax scale 用错 → 整段输出垃圾（静默正确性，头号）
- **主题**：aiter 注意力后端（ROCm 上 MHA 模型的默认）在 `softmax_scale != head_dim^-0.5` 时产出垃圾，即便 bf16 KV 也如此。
- **内容详情**：decode 把后端级 `self.scale = 1/sqrt(head_dim)` 传给 `paged_attention_ragged`；NHD prefill 三处调用不传 `softmax_scale`，aiter 回退到 `head_dim^-0.5`。受影响模型：Granite 3.x（`attention_multiplier` 1/64 或 1/128，低 8–11×）、Granite 4.0、GraniteMoE、Phi-3-small、Transformers 自定义 scaling。复现：main 上 `ibm-granite/granite-3.3-2b-instruct` 把「The capital of France is」续成「a fascinating number of times…」，加 `--attention-backend triton` 才答「Paris.」。
- **应用场景**：ROCm 上所有非标准attention缩放的 MHA 模型；默认 aiter 后端下的静默精度事故。
- **研判**：【事实】PR 给出明确复现（Granite 3.3-2B 主/备后端输出对比）与四处调用点修复。【观点】这是「默认 aiter 后端 + 非标准 scale」组合下的静默错误，与 10-28 `#46528`、10-29 `#46983` 同族，AMD ROCm 部署方应把非标准 scale 模型列入默认后端回归。
- **来源**：SGLang PR [#46833](https://github.com/sgl-project/sglang/pull/46833)。
- **证据等级**：官方一手（含复现对比）。

### 3.2 `#45486` AMD MoE：保留 CT FP8 Triton 布局与 reduction 维度（静默错误修复）
- **主题**：`SGLANG_USE_AITER=1` + 显式 Triton MoE runner 时，compressed-tensors CHANNEL FP8 层拿到 AITER-shuffled 权重，并在非 block FP8 核中丢失 1024→896 个 reduction 元素，两者都导致错误输出。
- **内容详情**：仅对真正 AITER runner 做 CT CHANNEL FP8 权重 shuffle；让非 block FP8 核的 padding 计算匹配其既有调用方；新增 AMD CI 量化测试（MI355X/gfx950，1 与 33 token 的 Triton/AITER MoE 数值对比）。
- **应用场景**：AMD ROCm 上的 compressed-tensors FP8 MoE 模型。
- **研判**：【事实】回归测试表显示 Base/AIter 在 1/33 token 均 pass，修复前丢 128 reduction 元素。【观点】又一处「AMD 默认路径 + 量化布局」静默错误，印证异构后端静默错误主线。
- **来源**：SGLang PR [#45486](https://github.com/sgl-project/sglang/pull/45486)。
- **证据等级**：官方一手（MI355X 验证表）。

### 3.3 `#46456` paged-experts：精确尺寸 pinning（内存效率，非错误但同源）
- **主题**：Paged Experts（`#42557`）把每个专家都锁页在 host 内存，torch pinned allocator 向上取整到 2 的幂，导致 402 MB 的 `w13` 实际 pin 512 MiB。
- **内容详情**：改用 HiCache 的 host allocator（`alloc_with_host_register` + 大页 + 预 fault + CUDA 注册），按张量精确尺寸分配；gather 仍经 `cudaHostGetDevicePointer` 读取。pinned 内存不可 swap，且常决定是否放得下模型（WSL2 上限约半 RAM）。
- **应用场景**：paged-experts 大模型（Qwen3-30B-A3B 多 pin ~1/3 冗余；gpt-oss-20b 多 85%）。
- **研判**：【事实】4-layer 测试（RTX 5070 Ti）显示 experts 字节不变、decode max logprobs 基本不变。【观点】非正确性错误，但属「专家张量分页」主线（累计结论第 7 条）的内存侧延伸，降低部署门槛。
- **来源**：SGLang PR [#46456](https://github.com/sgl-project/sglang/pull/46456)。
- **证据等级**：官方一手（RTX 5070 Ti 测试）。

### 3.4 其余 SGLang 要点（官方一手，标题/正文级）
- `#44457` MLA `k_scale==1.0` 时跳过 Triton K rescale（Quark fp8 KV 每层的 k_scale 恒为 1，避免无谓除 1；Kimi-K2-Thinking-MXFP4 TP8 MI455X，含 GSM8K）。证据等级：官方一手。
- `#47149` DSv4.1 `amax8_varlen` 与 mega MoE pre-dispatch 网格在 1024-thread SM（sm_107）上浪费空 CTA，按 occupancy 上限两波并 grid-stride；bitwise-identical。证据等级：官方一手（sm_107）。
- `#46800` AMD quark：把 MXFP4 共享专家排除出 dense-first 模型的 MXFP4 fused slot；`#46677` AMD dsv4.1 接入 aiter 的 32×32 block fp8 `wo_a` GEMM（gfx950，避免 fp8 权重被 dequant 成 bf16 多读一倍字节）；`#46676` dsv4 把协调 verify rank 移出 decoder-tail idle replay。证据等级：官方一手。
- `#45328` PD：把 decode KV 目标地址的发布推迟 1 个 decode 迭代，避免 D2H 同步卡住调度器（overlap 下）。`#46884` HiCache：当 offload 拷贝已在 forward stream 上 fence 时，提前发布 prefill read-done（overlap scheduling 下减少 GPU 空闲）。证据等级：官方一手（动机+改动）。
- `#47419` 文档安装版本 bump 至 **0.5.23**（CI：Base ✓ / Extra ✗ / AMD ROCm 10 N/A）——印证 0.5.23 在路上。证据等级：官方一手。
- `#46857` GLM-5.3：恢复 TP decode 融合与 CP prefill 的 graph sizing（标题级；正文 JSON 截断，未抓到细节）。证据等级：标题级。
- `#47070` NVIDIA H20 的 Qwen3-30B-A3B block-FP8 Triton 配置；`#47540` spec：在 verify-rows 测试中给 DSpark MLA draft 权重播种。证据等级：官方一手。
- `#43898` scheduler：移除 PD multiplexing（重构）；`#39947` DSpark 支持 dsv4 在 ascend 950；`#46278` NPU dspark 集成原生采样/验证核；`#47415`/`#47260` detokenizer 优化；`#46961` rust-server 按 cache tier 报告 cached tokens；`#47113`/`#47319`/`#46963` rust-server metrics 修复；`#46627` minimax-m3.1 在 H200/B300 加 nightly e2e；`#46907`/`#44725` minimax-h3 修复；`#47300` TensorMatcher。证据等级：官方一手（标题/正文）。

---

## 4. TensorRT-LLM：2026-10-30 提交（48 条）——Kimi K3 FP4 KV 解锁 + 两项 BREAKING + 新模型/MoE 能力

### 4.1 `#20888` Kimi K3：混合 KV 管理器下的 NVFP4 KV（FP4 MLA）能力解锁 + 静默降级修复
- **主题**：Kimi K3（混合 KDA+MLA）此前无法用 NVFP4 KV——模型加载在 SM107 上静默降级为 FP8，且 KV cache manager 对混合线性注意力模型拒绝 FP4 MLA。
- **内容详情**：把 `Fp4MlaV2CacheLayoutPolicy` 与 `MambaHybridCacheManagerV2` 组合（布局/尺寸/池角色/诊断），保留 KDA replay buffer，解除 `supports_fp4_mla_attention` 的混合排除；默认把最后两层 MLA 留在 FP8 KV（由 `TLLM_KIMI_K3_FP8_KV_LAYERS` 控制，取代旧的 BF16 变量），CuTe DSL MLA decode 在混合 batch 中服务这两层；新增 FP4 MLA chunked prefill、混合 FP4 MLA 的 disagg 服务（含 collision-free transfer layer ID）、chunked prefill 下的 KV block reuse。
- **应用场景**：Kimi K3 在 Blackwell/Rubin(SM107) 上的低比特 KV 部署；PD 解耦与长上下文前缀复用。
- **研判**：【事实】PR 描述组合策略、默认最后两层 FP8 及两套 fallback。【观点】把「用户以为用了 FP4 KV 实则被静默降级 FP8」这一默认值风险闭环，是本期最值得 serving 方核验的变更——升级后须确认 `TLLM_KIMI_K3_FP8_KV_LAYERS` 与实测 KV 精度。
- **来源**：TRT-LLM PR [#20888](https://github.com/NVIDIA/TensorRT-LLM/pull/20888)。
- **证据等级**：官方一手（PR 正文）。

### 4.2 两项 BREAKING 默认/API 变更（默认值风险主线）
- `#19379` **BREAKING：serve 的 `num_serve_frontends` 默认 `1→8`**（单 frontend fallback）。动机：高并发下单个 asyncio 事件循环做 json.loads+pydantic 校验+SSE 写出成为 host 瓶颈（c512→c1024→c2048 时 `gen_preprocessing` p50 从 54ms→109ms→1520ms，而 decode step 不变）；8 个 frontend 走 `SO_REUSEPORT`。`LlmArgs.num_serve_frontends` 字段默认仍 1（`LLM()`/bench/eval 不付 8 路 IPC 代价）。**【观点】行为变更默认，纯 `trtllm-serve` 用户会无声获得 8 进程；需注意多进程资源与日志拓扑。** 证据等级：官方一手（含动机数据）。
- `#21041` **BREAKING：移除过时 TensorRT-backend 兼容面**：删除 C++ executor API 的 `BatchingType`、`ExtendedRuntimePerfKnobConfig` 及其绑定/序列化，删除 `EagleDecodingConfig.num_eagle_layers`（保留 `max_draft_len`）。下游须移除引用并针对新头文件重编。coderabbit 评审显示 CI 有一成功一失败关联流水线，整体测试结论未完全建立。证据等级：官方一手（BREAKING API 移除，coderabbit 评审）。

### 4.3 能力 / 性能 / 新模型（官方一手）
- `#18555` 原生 FP8 E4M3 MoE LoRA 权重（PyTorch backend，routed-expert MoE）；H100 eager 236.55s、CUDA graph 75.98s 与 BF16 参考一致；转换 workspace 当前每次 MoE LoRA 运行都预留（性能 follow-up）。证据等级：官方一手（H100 验证）。
- `#16309` 新增 DSA Vanilla sparse attention（独立 PyTorch golden 实现，与 TRTLLM DSA backend 同契约，避免生产 fused kernel 缺陷被 golden 共享掩盖）；B200 测试通过。证据等级：官方一手（PyTorch golden）。
- `#19646` 新增 **DynamicEPLB kernels**（per-layer dynamic 专家并行负载均衡，MEGAMOE_CUTEDSL 后端，`moe_config.load_balancer.mode: per_layer_dynamic`）；每层 forward 据该层路由即时重规划并搬权重，无 host 统计/collective/权重暂存。证据等级：官方一手（含设计）。
- `#21042` 新增 **Muse Spark Open** 模型（PyTorch decoder + NaViT 视觉塔 + 音频编码器 + 多模态输入处理 + checkpoint 加载；强制 KV cache manager V2）。coderabbit 评审：范围不含 fused kernel/serving/文档/多卡 CI；作者报的测试与 serving 结果未独立验证。证据等级：官方一手（作者结果未独立验证）。
- `#20769` DeepSeek-V4.1-Flash 跑在 SM120；`#20764` Qwen3.5 dense VLM root 适配 MX；`#21256`/`#21257` GLM MoE（Glm4MoeForCausalLM / GlmMoeDsaForCausalLM）覆盖率；`#21105` Mamba2 MTP 避免整池 SSM 拷贝；`#20871` 给 TriAttention eviction workspace 加常驻 fallback 上界；`#21254` Qwen3-235B TP1→TEP4 的 KV bounce buffer；`#21131` 保留 per-rank attention-DP KV 指标；`#21088` 注意力/SSM 的层与 buffer 复用策略；`#21104` DeepSeek-V4.1 命名空间 Responses tools/calls。证据等级：官方一手（标题/正文级）。

---

## 5. llama.cpp：2026-10-30 提交（14 条）——KV 旋转常量上 GPU + OpenCL 低比特 + gemma 内存修复

### 5.1 `#31073` kv-cache：把旋转常量留在 GPU（长尾 PCIe 带宽优化）
- **主题**：把 Haramard 静态旋转矩阵常驻 GPU，而非每次 host↔GPU 搬移。
- **内容详情**：作者环境为 RTX 3090 + GTX 1080 接 PCIe 2.0 老笔记本，故寻找减少 host↔GPU 传输的机会。对比 master（`edc353842c1b`，Linux，RTX 3090，i7-2720QM 4 线程，Qwen3.5 9B Q4_K_M / Q8_0 KV / flash attention / batch 512 / ubatch 128，`--load-mode none` 全 GPU 卸载）：`pp512` 2228.14→2216.04（**-0.54%**），`tg128` 71.23→77.43（**+8.70%**）；WikiText-2 ppl 两侧一致（7.5811±0.61829），两段 128-token 贪婪输出完全相同；CUDA trace 显示旋转矩阵上传从 22 次降到 4 次。
- **应用场景**：PCIe 带宽受限 / 多卡老平台的长上下文生成（tg 阶段收益明显）。
- **研判**：【事实】单模型单机本地测试，tg128 +8.70%、pp512 微降，ppl/输出一致。【观点】属边缘平台优化，对现代 PCIe4/5 平台收益可能更小；但思路（常驻旋转常量）对多上下文并发有意义。
- **来源**：llama.cpp PR [#31073](https://github.com/ggml-org/llama.cpp/pull/31073)。
- **证据等级**：官方一手（本地 RTX3090+PCIe2.0 测试）。

### 5.2 其余 llama.cpp 要点（官方一手）
- `#31340` OpenCL 初始支持 **q2_0**（参考 #30180 的 adreno 优化）；`#31341` OpenCL 初始支持 **IQ1**（IQ1_S / IQ1_M）。属后端低比特矩阵乘扩展，暂无性能数字。证据等级：官方一手（初始支持）。
- `#31394` gemma3n / gemma4：在 tied output **之前**创建 `token_embd`，修复非 mmap 加载（`-lm none`/`-lm mlock`、CPU-only、`-ngl 0`）时重复分配一份 `token_embd` 的问题。实测 `gemma-4-E2B-it` Q4_K_M CPU-only 无 mmap：CPU 模型缓冲 3163.73→2947.73 MiB（省 216 MiB，恰为 `token_embd` 大小），贪婪输出不变。证据等级：官方一手（gemma-4-E2B-it 对比）。
- `#31368` memory：拒绝回滚超过最后一个 ubatch 的快照；`#31386` quant：允许对 LFM3-Audio codec decoder 量化；`#31374` common：在 prompt 的结束标签处结束 reasoning budget 强制；`#31408`/`#31413`/`#31417`/`#31424` CI/Windows 清理（apple jobs 抽出 daily、Windows ccache、删除 `fs_path_to_utf8`）；`#31400` vendor cpp-httplib 0.64.0；`#31329` Windows exe 内嵌 app manifest。证据等级：官方一手（标题/正文）。
- nightly：窗口内发布 **b12004–b12017**（2026-10-30，prerelease）。

---

## 6. 累计结论与研判（沿用 + 新增）

**沿用 2026-10-29 档案（覆盖 10-28）第 6 节累计结论 1–8 条：**
1. 调度器隐式阻塞的定量画像（host sync 移出热路径的 +4.5%~9.9% 画像）；
2. **发版窗口的双向性**：发版当天合入 main ≠ 在版内（compare API 祖先关系核验）；
3. **默认值成为最主要的风险/收益载体**（默认后端/量化/NUMA 静默改变行为）；
4. **异构后端静默错误高频**（AMD ROCm / Metal / 异构 kernel 不报错只掉精度）；
5. **前缀缓存 × MoE 路由 × 投机解码交叉债务**（草稿/索引 KV 所有权一致性）；
6. **量化路径静默错误**（scale 未应用/参数丢失/4D 量化漏算 → 输出偏离且无报错）；
7. **专家张量/路由归属与分页漏接静默错误**（paged-experts 不被 paging、路由行覆盖 → 静默用错专家值）；
8. **静默正确性升维为跨引擎常态主线**（高频失效模式已从「崩溃」转为「不报错、精度静默掉/长上下文静默塌方」）。

**本期新增第 9 条（长上下文 int32 溢出为静默正确性新子项）：**
- vLLM `#63761` 证实：Triton 将 `<2^31` 整数参数按 int32 传递，Qwen3.5 多个注意力/GDN 核在 `token*stride >= 2^31` 回绕，长上下文（最早 ~209,717 token）触发错误输出/非法访存。**建议长上下文回归把 int32 边界（~200K–524K token）与「带量化端到端 GSM8K + RULER-64K 级长上下文」一并纳入默认发布门禁**（强化第 4、6、8 条）。

**主线强化（本期的进一步铁证）：**
- **第 3 条（默认值风险）**：TRT-LLM `#19379`（`num_serve_frontends` 默认 1→8）、`#21041`（BREAKING API 移除）、TRT-LLM `#20888`（修复 Kimi K3 NVFP4 KV 在 SM107 **被静默降级 FP8**）——「静默降级」本身也是默认值风险的一种，serving 方升级后须实测确认所用精度。
- **第 4 条（异构后端静默错误）**：SGLang `#46833`（aiter softmax scale → 整段垃圾）、`#45486`（AMD CT FP8 布局丢 128 reduction 元素 → 错误输出）与 10-28 `#46528`、10-29 `#46983` 同族；**AMD 默认 aiter 后端仍是静默错误高发区**，ROCm 部署方应把非标准 scale / 量化布局模型列入默认后端回归。
- **第 7 条（专家张量/paging）**：SGLang `#46456` paged-experts 精确 pinning（砍掉锁定内存浪费）属「专家张量分页」主线（累计结论第 7 条）的内存侧延伸，非正确性错误但同源。

---

## 7. 趋势判断与可执行清单

**趋势判断**
1. **静默正确性 + 默认值风险双主线继续主导**：本期跨 vLLM/TRT-LLM/SGLang 的 silent-error 与行为变更默认，数量与严重性仍居首；长上下文 int32 溢出与 AMD aiter 静默错误是新增高频点。
2. **AMD/ROCm 成为静默错误的集中暴露面**：10-28 `#46528`、10-29 `#46983`、本期 `#46833`/`#45486` 连续命中，且多与「默认 aiter 后端 + 量化/非标准 scale」组合。建议把 ROCm 默认后端回归视为独立门禁。
3. **KV/PD/HiCache 调度优化走向「去 host sync」架构**：SGLang `#45328`（推迟 1 decode 迭代发布 KV 目标）、`#46884`（forward-stream fence 后提前 read-done）、vLLM `#60867`（混合 READ 不静默跳循环态）共同指向「让调度流跑在设备事件之前、避免 D2H 同步卡顿」的方向。
4. **投机解码与 MoE 走声明式/插件化**：vLLM `#63205`（MRv2 声明式 speculator 配置 + 插件 API）、TRT-LLM `#19646`（DynamicEPLB 每层动态负载均衡）显示两大引擎把「可插拔 speculator / 动态专家放置」作为性能主战场。

**可执行清单（建议 serving/研发方）**
- [ ] **长上下文回归门禁**：把 int32 边界（~200K–524K token）与「带量化端到端 GSM8K + RULER-64K 长上下文」纳入默认发布门禁（第 9 条、第 8 条）。
- [ ] **ROCm/aiter 部署回归**：对 AMD 默认 aiter 后端的非标准 `softmax_scale` 模型（Granite 等）与 compressed-tensors FP8 MoE，加入默认后端冒烟测试（#46833、#45486）。
- [ ] **TRT-LLM 升级关注**：`#19379` 纯 `trtllm-serve` 默认变 8 frontend（确认资源/日志拓扑）；`#21041` 移除 `BatchingType`/`ExtendedRuntimePerfKnobConfig`/`num_eagle_layers`（下游重编）；`#20888` Kimi K3 升级后实测 `TLLM_KIMI_K3_FP8_KV_LAYERS` 与 FP4 KV 精度（确认非静默降级）。
- [ ] **vLLM 长上下文 Qwen3.5**：合入 `#63761` 后验证 ≥200K token prefill/recurrent 不再溢出；关注 `#64467`（MegaMoE 32-row tiles，bit-identical 加速）、`#64509`（Qwen3.5 NVIDIA 融合 norm+quant 1.7–2.0×）。
- [ ] **SGLang 内存/调度**：评估 `#46456` paged-experts 精确 pinning 对 WSL2 / 大 paged-experts 模型的放得下收益；PD/HiCache 评估 `#45328`/`#46884` 的 overlap 收益。
- [ ] **llama.cpp**：`#31073` 旋转常量上 GPU 对 PCIe 受限平台 tg 阶段有 +8.7% 收益，可据平台采纳；`#31394` gemma3n/gemma4 CPU-only 省 216 MiB，建议 -ngl 0 场景合入。

---

## 8. 来源索引与数据质量

**官方一手（提交 API + 发布说明 + PR 正文）**
- vLLM：https://github.com/vllm-project/vllm （**97 commits @2026-10-30**；v0.32.0 2026-10-18，窗口内无新正式版）；PR `#63761`/`#60867`/`#54307`/`#64612`/`#64467`/`#64509`/`#64138`/`#63205`/`#63886`/`#57770` 正文已抓。
- SGLang：https://github.com/sgl-project/sglang （**93 commits @2026-10-30**；v0.5.22 2026-10-19；`#47419` bump 文档至 0.5.23 但无新 tag）；PR `#46833`/`#45486`/`#46456`/`#44457`/`#47149`/`#46800`/`#46677`/`#46676`/`#45328`/`#46884`/`#47419` 正文已抓（`#46857` 正文 JSON 截断，记标题级）。
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM （**48 commits @2026-10-30**；**v1.4.0rc2 2026-10-28T15:51:34Z** prerelease，窗口前一日发布、仍为最新）；PR `#20888`/`#19379`/`#21041`/`#18555`/`#16309`/`#19646`/`#21042` 正文已抓。
- llama.cpp：https://github.com/ggml-org/llama.cpp （**14 commits @2026-10-30**；nightly **b12004–b12017** 2026-10-30，prerelease）；PR `#31073`/`#31340`/`#31341`/`#31394` 正文已抓。

**社区/外部（二手线索，未核验）**
- HN（2026-10-30 窗口，Algolia HTTPS，story 过滤）：引擎相关**无高信号故事**。命中多为模糊匹配——`vllm` 22 命中里仅「Ask HN: What inference engine and KV offload engine do you use and why」(1 分)、「Learn LLM inference system on Apple Silicon」(2 分，学习资源非引擎更新)、「PostgreSQL 19… AI/LLM Burden」等；`sglang`/`TensorRT-LLM`/`llama.cpp`/`KV cache`/`speculative decoding` 故事数均为 0；`GLM`/`DeepSeek` 命中均为无关（Intel 股价、CrowdStrike 等）。均记为低信号/未核验，不计入结论。
- Reddit / YouTube / X：未纳入采集；本期未核验到可确认发布日期与内容的官方视频。

**数据质量与限制**
- 计数口径：GitHub commit API 的 `since`/`until` UTC 日期窗口（vLLM 97 / SGLang 93 / TRT-LLM 48 / llama.cpp 14，均 HTTP 200）；按 commit date 过滤，偶有相邻日 commit 边界误差，属历史一致口径。
- 证据等级：带验证表的 silent-error/性能 PR 记为「官方一手（含验证）」；仅提交标题或正文未全抓的记为「标题级」（如 SGLang `#46857`）；TRT-LLM `#21042` 记「官方一手（作者结果未独立验证）」；社区文章记「二手线索/未核验」。
- 截断说明：vLLM `#64612` 完整 benchmark 数字与 `#64509` 的精确加速比在抓取时被截断，仅保留 PR 明确可见的声明（2.85 GB/rank、1.7–2.0×），未外推未呈现数字。
- 未调用外部浏览器（agent-browser）：本期数据源（GitHub API / releases / HN Algolia）均可直接 HTTP 访问，沿用 2026-10-18 起确立的「API + PR 正文直抓」方法，未触发浏览器需求（与历次一致）。
- `find-skills`：经 SkillHub 语义检索（EN+ZH 各 1 组）最高相关性 0.050（防骗大师等通用技能），与「官方提交采集 + 证据分级归档 + 仓库同步」不匹配，**未安装**（与历次一致）。

> **归档空档说明**：本档案覆盖 **2026-10-30** UTC 窗口；**2026-10-29** UTC 窗口因 2026-10-30 当日自动化未执行而无档案（即上一期 2026-10-29 档案覆盖 2026-10-28 之后、本期之前存在一日空档）。本目录不补写中间日期内容，各期仅按自身关注窗口采集。如需回溯 2026-10-29 窗口，请以各引擎官方提交 API 按日期区间重新采集。
> 本档案仅覆盖 2026-10-30 UTC 窗口；相邻日期请以官方提交 API 按日期区间重新采集。
