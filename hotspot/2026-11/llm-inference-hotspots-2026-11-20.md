# 大模型推理优化每日热点档案

> 档案日期：2026-11-20（本自动化于该日执行）
> 关注窗口：2026-11-19（周四）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，vLLM 分两页 100+24）+ releases API + 48 个关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 均经官方 issues API 抓取 markdown，官方一手，配额 57/60 起、无降级）+ HN Algolia（HTTPS，numericFilters 按 2026-11-19 epoch 窗口 1795046400–1795132799）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 124 / SGLang 82 / TRT-LLM 51 / llama.cpp 28 合计 **285** 条（2026-11-19 为周四，vLLM 分两页取全、无截断；较上一窗口 300 略回落）；**窗口内无新正式版本**——vLLM v0.34.0（11-16）、SGLang v0.5.23（11-02）、llama.cpp 正式版 v1.0.0（11-13）、TRT-LLM 最新仍为 v1.4.0rc4（11-18 prerelease，非本窗口），llama.cpp 推进 nightly b12552–b12556（prerelease）。48 个关键 PR 正文全部抓取成功。HN 引擎相关高信号弱：「Taking vLLM Apart: A Practical Guide to Disaggregated Serving」（2 分）、「High-Interactivity Decode with FlyDSL MonoKernels in vLLM」（1 分），均二手线索/未核验。

---

## 1. 版本状态与本期头号事件：无新版本 + 「语义合并冲突」成为静默正确性新子项（新增第 23 条）+ 量化/MoE/KV 路由静默错误密集延续

**提交计数（2026-11-19 UTC，均 HTTP 200；周四）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 124 | 分两页 100+24；Kimi K3.1/DSv4.1/Muse Spark recipe 簇 + ROCm gfx1250/gfx1151 性能簇 + 静默正确性 + Rust 前端 |
| SGLang | 82 | diffusion/flux3 大簇 + DSA/FP4 indexer + Inkling2 拆分栈 + linker/量化 |
| TensorRT-LLM | 51 | Modeling V2 / GMS / PrimTS 主线，仍以 infra/waive/test 为主，feature/fix/perf 约 20 条 |
| llama.cpp | 28 | 多后端（CUDA/Metal/OpenCL/Vulkan/WASM/CPU）+ GGUF 加载越界 + MoE 重复专家 ID |
| **合计** | **285** | 工作日正常量；与上一窗口（11-18 的 300）连续，无空档 |

### 1.1 本期头号事件：静默正确性密集爆发 + 新增第 23 条「语义合并冲突」

本窗口静默/崩溃级案例延续上期密度，覆盖四引擎，且出现一类此前未单列的根因——**两个各自正确的 PR 因语义合并冲突叠加，产出静默错误**：

- **SGLang #51289**（官方一手，fix，**新增第 23 条**）：Kimi-K3.1 在 decode context parallelism（DCP）下用 `--dsa-decode-backend prims_ts` 或 `trtllm` 时**产出 garbage——GSM8K 从 ~0.97 掉到 ~0.03，答案混入其他请求内容**；默认 `flashmla_kv` 不受影响（故 `test_kimi_k31_b300.py` 保持绿）。根因是**两个各自正确的 PR 的语义合并冲突**：#49106 给 `_forward_trtllm` 加 `dcp_partial` 并把全局 slot 用 `localize_dcp_page_table` 局部化（其 base 上 indexer 返回 global slot）；#48895 先合入，把局部化移进 indexer（`dcp_local_topk` 直接返回本机紧致 local slot），并只更新当时的唯一消费者 `flashmla_kv`。两者改不同行、干净合并，但 `_forward_trtllm` 对**已经是 local 的 slot 又局部化一次**，只保留约 1/dcp_size 的选中 key 并指向无关 KV 条目。InferenceX Kimi-K3.1 GB300 recipe 用 `prims_ts`，从 Day-0 镜像切到 main nightly 后所有解聚点崩溃。修为对 `dcp_partial` 分支不再二次局部化。属跨 PR 语义合并冲突导致的静默正确性铁证（第 23 条）。
- **vLLM #68084**（官方一手，fix，强化第 6/19 条）：DeepSeek-V4-Flash-Base 设 `swiglu_limit=10`，但 batched DeepGEMM 在融合 SwiGLU + FP8 量化步骤漏掉该 limit。8×H200 开 `deepep_low_latency` 时，16,384 token 中出现 **9 个严重 logprob 不匹配**——单个 token logprob −0.31 vs 参考 −31.68。修为在 SiLU/乘法前应用模型 limit（gate 封顶、up 限幅 ±limit）；关 clamp 时 6 个 native CUDA 变体在 SM90/SM100 发射字节相同、25 配置中位 +0.01%，开 clamp 时中位 +2.91%/最大 +5.46%。属量化路径参数漏接静默错误。
- **vLLM #68386**（官方一手，fix，强化第 9/19 条）：`fused_qk_norm_rope` 一个 PR 修三处 bug：① 32 位 QKV offset 在大 `num_tokens` 溢出；② 多头每 warp kernel 的 NeoX 部分 rotary 损坏（Fixes #51049）；③ NeoX partner lane 在 `rotary_dim/2` 非 2 的幂时取错；另加一条 warp barrier。GLM-5.3-Flash 在 H200 DP4+EP 开视觉编码器时，ViT 跑 960,000 patches，main 在 encoder profiling 触发 illegal memory access；多头每 warp kernel 在 partial rotary 下 main 与参考在 33–70% 元素上不一致、8 头时返回 NaN。修后所有 head 数与参考逐位一致。属长上下文 int32 溢出 + 部分 rotary 静默错误（第 9 条）。
- **llama.cpp #32374**（官方一手，fix，强化第 9 条）：`mul_mat_q` 的 dst tile offset（`jt*J*stride_col_dst`）用 32 位 int 计算，dst 超过 2^31 元素时列 tile 从 `INT_MAX/nrows` 之后开始即回绕越界写。248,320 词表 q8_0 输出头 + 大 ubatch（`Qwen3.6-35B-A3B`，`-b 32768 -ub 16384`）可达——失败节点 `result_output` q8_0 [2048, 248320]×f32，dst 4.07e9 元素，触发 illegal memory access。修为把三处 dense path 与 stream-k fixup 的 `offset_dst` 加宽到 `int64_t`。与 #32601（下）同属长上下文/大张量整型溢出族。

### 1.2 其他静默/崩溃/安全级案例（强化第 4/5/6/8/9/19 条）

- **vLLM #68353**（官方一手，fix，强化第 6 条）：FlashInfer MXFP8 linear kernel 在 layer 的 N 不是 8（CuTe-DSL）/32（CUTLASS）的倍数时**算 garbage 或直接抛错**——CuTe-DSL 在 M=64 对未对齐 N 写 `inf`/垃圾，CUTLASS 抛 `Check failed: n % alignment == 0`。4×GB300（SM103）上新测试 main **11 failed/13 passed**、修后 24 passed。修为把 N 用零行 pad 到 8/32 倍数并切回原 N。属量化布局未对齐静默错误。
- **vLLM #68072**（官方一手，fix，强化第 4/8 条）：MoRIIO heterogeneous-TP WRITE（prefill TP≠decode TP）无法服务 MiniMax-M3，两个独立原因：① ROCm 上 `--kv-cache-dtype nvfp4` 的 NVFP4 KV page 按「每 KV head 字节连续」切片，但 ROCm page 实际先存所有 head 的 FP4 数据再存所有 head 的 block scale——TP4→TP1 的 decode 收到错乱 KV、**生成 garbage**；② MiniMax-M3 的 lightning-indexer key 住在 MLA 型 cache（GQA 模型里的每 token 单向量、各 TP rank 相同），按 KV head 切片会每请求报 `MoRIIOError`。修为按真实布局切 NVFP4 page、拒绝无法安全切的 packed page（改抛明确 `ValueError`），并把 MLA 型 indexer cache 整块复制。
- **vLLM #68164**（官方一手，fix，强化第 14 条）：投机解码在线接受率估计器 refit（`_refit_kernel`）的 NaN-only guard 放过 `+inf`——一次无穷 Newton step 会让估计器在该服务生命周期内永久 non-finite，直接毁掉自适应验证。修为丢弃非有限 slope/intercept step；H100 上新增 3 个针对性测试（构造 Schur complement 恰为 0 使 slope step = 4/0 = +inf）。
- **vLLM #67947**（官方一手，fix，强化第 14/19 条）：`deduplicate_contexts="all"` + 投机解码下，早现于 prompt、终止于 output 的重复上下文未被识别，导致该 token 被加水印（非投机路径会跳过）。真实 Triton kernel 上错误 skip 决策从 34/112,521 verification rows + 45/145,813 draft steps 降到 0；323→327 passed。此修复是 #65961 token-identity 声称的前置依赖。
- **vLLM #68586**（官方一手，fix，强化第 8 条）：投机层与 host-resident target（`--enable-extensible-kv-cache` + HiSparse）共享 KV 时，KV cache extend 因重建的 committed views 遗漏 host-resident 张量而在 `bind_kv_caches` 抛 `KeyError` 崩溃。修为把修复移到 KV cache extend 步骤（不改 `bind_kv_caches`，缺共享 target 仍在启动期失败）；HiSparse+DSpark 精简 Mistral Large 4 上 main 崩溃、修后可服务。
- **llama.cpp #32601**（官方一手，fix，强化第 9/18 条·安全）：HRM 架构 `load_arch_hparams` 把 `n_hrm_layers_per_stack × n_hrm_h_cycles × (n_hrm_l_cycles+1)` 算进 `uint32_t` 再断言等于 `block_count`——真实乘积为 `block_count + k·2^32` 时断言仍通过（如 `1073741825×2×2 = 2^32+4` 对比 `block_count=4`），`load_arch_tensors` 随后用 int 迭代并**越界写 `model.layers` vector**。手工构造的恶意 HRM GGUF 可绕过架构检查并越界写（ASan 报 heap overflow，`test-llama-archs -a hrm_text` 退出码 134→0）。因仓库 `SECURITY.md` 禁止公开 security issue 且禁用了私有漏洞上报，作者未开 issue。属 GGUF 加载期整型回绕 + 安全加固。
- **TRT-LLM #22563**（官方一手，fix，强化第 19 条）：`GMSBackend.move_untracked_params()` 对共享 sliced storage 的未跟踪 CUDA 参数只复制第一个 view 并在 offset 0 重建所有 view——后续 slice 读到第一个 slice 的值，第一个 view 之外的字节未被复制。修为对每个完整 `UntypedStorage` 复制一次，再按原 dtype/shape/strides/storage offset 在同一映射存储上重建参数；拒绝零字节 CUDA 参数存储与 lazy conjugate/negative view。作者指出 upstream Dynamo 集成有同样缺陷，另行跟进。

### 1.3 本期其他结构性观察

- **encoder cache 也进入分层卸载**（vLLM #63294，官方一手，perf/feature）：新增 `ECTieringConnector`，CPU 内存作主层、encoder 输出持久化到文件系统供工作集超 CPU 容量时复用（异步 CPU→FS write-through、FS→CPU promotion、排队期预取、跨重启复用）。Qwen3-VL-8B BF16 TP1 / H800 / 8 GiB CPU + 128 GiB FS / 22.4 GiB 工作集 / 48 并发下，重放 mean TTFT **1102.80→723.30 ms（−34.41%）**、p95 TTFT −31.61%、吞吐 **+25.84%**。属 KV 分层/卸载思路向 encoder cache 的延伸（第 8 条）。注：prefix caching 关闭、fsync 关闭、FS backpressure 关闭，作者自报口径。
- **调度策略学术落地**（SGLang #35637，官方一手，feature/opt-in）：新增 **UniBoost** 尾部感知调度策略（ICML 2026《Beyond Prediction: Tail-Aware Scheduling for LLM Inference》），每次 prefill admission 前按软优先级重排等待队列 `Phi(i)=arrival_i−b_γ(w_i)`，`b_γ` 有界/递减/凸→短请求与 radix-cache-hot 请求前移、长请求退回 FCFS 且**无饥饿**，γ 插值 FCFS↔SJF 而无需输出长度预测；可选 gamma-Ada 由 t99/t95 在线自适应。TP/PP>1 用 rank-invariant 输入保证各 rank 接纳同一批。头部结果为作者在其移植到 v0.5.13 的早期版本上所测（非本 PR 版本），按作者报告/描述级记录。
- **KV dtype 与统一内存持续标准化**（强化第 13/15 条）：SGLang #51296 把 MXFP8 KV 的 UE8M0 scale 行存进 unified memory page（Inkling-2-Small `--kv-cache-dtype mxfp8` 前置，E4M3 K/V + head_dim 128）；vLLM #68533 让 draft-only 滑窗桶不再决定 KV cache group 尺寸（Muse Spark Open 全注意力层去掉 20%/token 的 padding）；vLLM #67941 允许 layer-compact KV layout 混合 page size（Nemotron Shortcut）。

---

## 2. vLLM：2026-11-19 提交（124 条，两页 100+24）—— 静默正确性(#68084/#68386/#68353/#68072/#68164/#67947/#68586) + GLM-5.3-Flash SM90 MegaMoE(#66812, +12~24%) + encoder cache 分层(#63294, TTFT −34.41%) + gfx1151 W4A16(#66133, +37~44%) + 安全(#67348) + Rust 前端

### 2.1 `#68084` [Bugfix][MoE] batched DeepGEMM 补 SwiGLU clamp —— 见第 1.1 节

### 2.2 `#68386` [Bugfix][Kernel] fused_qk_norm_rope 三合一（int32 溢出 + NeoX rotary）—— 见第 1.1 节

### 2.3 `#68353` / `#68072` 量化布局与 KV 切片静默错误两连 —— 见第 1.2 节

### 2.4 `#68164` / `#67947` 投机解码正确性两连（接受估计器 +inf / 水印跨 prompt 边界）—— 见第 1.2 节

### 2.5 `#68586` [Bugfix][HiSparse] host-resident 共享 KV 层 extend 崩溃 —— 见第 1.2 节

### 2.6 `#63294` [EC Connector] encoder cache 文件系统分层（perf/feature，强化第 8 条）—— 见第 1.3 节

### 2.7 `#66812` [Kernel][Perf] GLM-5.3-Flash FP8 routed experts 跑 SM90 DeepGEMM MegaMoE（perf，强化第 12 条）

新增权重格式 `MEGA_MOE_FP8_BLOCK_SM90`，把 checkpoint 的 E4M3 权重 + FP32 128×128 block scale 不经 MXFP8 requantize 直接送 kernel（`--moe-backend deep_gemm_mega_moe` on H200）；SM100 仍用 `MEGA_MOE_MXFP8`。DeepGEMM 钉到 vllm-project/DeepGEMM `dev` 44cba7d。**8×H200 默认显存**：各测量点均领先 main 默认与 FI MoE-EP，KV 池最大 66.44 GiB/GPU（vs 60.12 / 54.05）。**AgentX c=1..256 扫描**：4×/8×H200 每级 output tok/s/GPU 领先 main 默认（c=1..32 +12%~+24%），per-user 除 4×H200 c=128（−12%，噪声内）外全领先（c=1..32 +26%~+34%）。同等显存下多留 3.1–4.4 GiB/GPU KV（DP4，+11–16.5% KV tokens）或 6.3–12.4 GiB（DP8，+10.5–22.9%）。GSM8K 逐题配对与 main 默认无差异、logprobs 接近。属量化 MoE 后端在 SM90 的新落地（第 12 条），KV 容量增益是主要领先来源（作者明示）。

### 2.8 `#66133` [ROCm][Perf] gfx1151 W4A16 bf16 decode-sized-M tile（perf，硬件特定）

gfx1151 上 bf16 W4A16 Triton 层在 decode 尺寸 M 存在 tile cliff：宽 N 层 M=33..64、g32 tall-K 层 M=65..128 时 prefill 调优的 tile 极慢。本 PR 为 M=9..128 加 gfx1151 bf16 decode tile 选择，扫得 tile 未持续更优时保留原阶梯。**Qwen3.8-27B**：原始 run C=64 **+43.8%**、MTP C=12 +41.9%、C=16 +33.6%；独立复跑 +36.9%/+37.8%/+33.2%（三次 PR run 相差 ≤1.3%）。kernel 基准 35 个旋转权重 case 几何均值 1.38×（最差 0.90×），相对 fp32 参考最大相对误差 0.0031。属硬件/形状特定的 launch 级优化（避免 32 MB MALL 复用同一权重张量，256 MB flush）。

### 2.9 `#66631` [HiSparse] unpinning 页的批量 owner 记账（perf，强化第 8 条）

P/D 导入长多轮前缀（#65888）会采纳约 1,250 页 × 20 resident group 的 GPU 副本并在下一步 unpin，原实现每 block 建一个 owner `set` 与 `(request,page)` tuple，约 5 万长生命周期对象/次导入。8×B200 上这占 17.7 ms `schedule()` 步中的约 12 ms，并触发最高 76 ms 的 full-GC 尖峰。本 PR 用 block→owner tuple 映射（单 owner block 零分配）、每列查一次 null page、绑定一次 block-reuse watcher。CPU 微基准 + P/D A/B（GLM-5.3-NVFP4，1P TP4 + 1D TP4 HiSparse，NIXL over RDMA，C=8）验证，行为不变。

### 2.10 `#67348` [Frontend] max_tokenize_tokens 上限防 DoS（安全，强化第 18 条）

`/tokenize`、`/detokenize`、`/v1/messages/count_tokens` 此前接受任意长输入，存在 CPU 耗尽/OOM/DoS 风险。新增 `--max-tokenize-tokens`（默认 `max(max_model_len*2, 131072)`，≤0 关闭）在标准模板与 GPT-OSS Harmony 渲染路径拒超大请求（HTTP 400）；Rust 前端标记为不支持该 flag，传入即启动失败而非静默忽略。Fixes #67197。

### 2.11 其他要点（标题级 / 描述级 / 性能级）

- **ROCm 性能簇（标题/描述级）**：`#63396` gfx1250 DSv4 W4A4 MoE 融合 silu+quant+preshuffled 权重；`#63243` gfx1250 Router GEMM 用 AITER tuned GEMM；`#67365` DSv4.1 decode sparse-MLA ragged top-k 从 MXFP4 indexer row bounds 一次 launch 打包；`#67364` 压缩 KV cache 插入 compression launch；`#67363` MXFP4 indexer decode metadata 一次 launch；`#66184` MiniMax-M3 整层 mono decode（`--monokernel`）；`#63397` decode W_UV 用 AITER Triton bf16 BMM（tuned 时）；`#67133` AITER FA 专用 split-KV kernel（GQA8/head64/FP8 KV）。
- **KV/连接器**：`#68710` SimpleCPUOffloadConnector 补 `shutdown()`——copy thread 未停、CPU KV 映射 `cudaHostRegister` 释放后仍注册，同进程后续注册落到同地址使 ROCm runtime abort（MI355 DPX V1 KV Offload 自 #68296 起 CI abort）；`#67939` Mooncake batch reply 校验，畸形回复不再当成功、失败 prefix lookup 返 IPC error（`kv_load_failure_policy=recompute` 下重算），512 passed，无 live GPU/RDMA/模型输出评测；`#66795` NIXL Push 三项修复（默认 num_threads=0 避免 UCX 线程池 WRITE 停顿、`NIXL_THREAD_SYNC_RW` 锁 agent、WRITE in-flight 期间不释放 block），pull 模式 lost-connection 修复拆为 #68047；`#68529` 按请求遵守 Mooncake offload 限额；`#63294`(EC Connector) 见 2.6。
- **量化/权重**：`#68171` online requantization 用 meta device + 逐层加载把峰值显存从「序列化模型+量化模型」降为单份——Mistral-Large-4-1T-A52B（FP8 block→MXFP4 experts，TP4）峰值 287.9→147.9 GiB/GPU；`#68043` RL packed transfer 单 buffer chunk 布局供 NCCL/IPC 共用；`#68048` layerwise reload 保留 CPU-offloaded Engram storage；`#68329` 移除 torchao reload decorator。
- **MoE 路由**：`#62912` full-warp top-k 行用 redux.sync 并把选择留在寄存器（SM80+）；`#68574` Qwen4Exp PLE short-conv prefill checkpoint 导出 ROCm；`#66791` Qwen4Exp HyperConnection up projection + gate mix 在 SM90 融合；`#66812` 见 2.7。
- **新模型/recipe（标题/描述级）**：`#67880` Inkling 2 Hopper SM90 支持、`#68761` Inkling 2 Humming expert GEMMs on Hopper；`#68296` HY4 Flash + 多模态；`#68556` DeepSeek-V4.1-Flash GB300 Dynamo AgentX presets from InferenceX #5015；Kimi K3.1 specialized decode（`#68701` GQA DSpark drafts on specialized drafter）；`#66133` 见 2.8。
- **多模态/其他**：`#67976` LFM3-VL encoder CUDA graph；`#68655` 接受 NumPy 图像；`#68367` 归一化负 stride 图像；`#68254` Unlimited-OCR-2 关 extensible KV cache；`#68563` Unlimited-OCR-2 DASH routing batch invariant；`#67954` HF renderer 保留 assistant refusal history。
- **Rust 前端簇（约 10 条，标题级）**：`#68390` /v1/completions 批量 prompts；`#68612` /generative_scoring；`#68584` 解 Mooncake lookup_lease stats；`#68186` server 级 mm processor kwargs；`#68565` EPD metadata-only input capability handshake；延续第 17 条。
- **CI/构建/安全**：`#68457` Python 3.14 成镜像与 CI 默认；`#68348` [Do Not Merge] vllm-flash-attn 指向 flash-attention#222 CI；`#68526` RayExecutorV2 actor 先 kill 再 join；`#68288` liveness monitor peek EngineCore 退出状态；`#63113` 丢弃会溢出 max_model_len 的 streaming 更新。

---

## 3. SGLang：2026-11-19 提交（82 条）—— 语义合并冲突静默错误(#51289, GSM8K 0.97→0.03, 新增第23条) + KDA int32 溢出(#50775) + FP4 indexer Kimi-K3.1 DCP(#51450) + DFLASH ring 默认关闭(#51386) + UniBoost 调度(#35637) + inkling2/MXFP8 KV 拆分栈 + diffusion 大簇

### 3.1 `#51289` [Fix] DSA trtllm/prims_ts DCP decode 局部 slot 二次局部化 —— 见第 1.1 节（新增第 23 条）

### 3.2 `#50775` [Fix] KDA prefill 超过 2**31 token offset 溢出（fix，强化第 9 条）—— 见第 1.1 节

补充：三个 KDA prefill kernel 全用 int32 构 per-sequence row offset，`bos*H*K` 在 packed prefill 起点越过 `2**31/(H*K)` 时回绕（H=32,K=128 每 rank 约 524k tokens，H=64 约 262k，`chunked_prefill_size` 调大或关闭可达）。`chunk_local_cumsum_vector_kernel` 与 `kda_gate_chunk_cumsum_vector_kernel` 在 MI300X 抛 `HSA_STATUS_ERROR_MEMORY_FAULT`（KDA prefill 第一个 kernel，整个崩）；`chunk_kda_fwd_kernel_intra_token_parallel` 不 fault 但**读写 q/k/g/Aqk/Akk 错行**。修为把三个 kernel 的 `bos`/`eos` 与 program id 加宽到 int64（与 upstream flash-linear-attention 一致）。该 PR 更正了 #49803 关于其他 FLA kernel 溢出阈值的说明（那些因 `i_h` 已是 int64 实际不溢出）。

### 3.3 `#51450` [Feat][NVIDIA] Kimi-K3.1 FP4 indexer + DCP 支持（feature，强化第 6/11 条）

`--enable-fp4-indexer`（CUDA SM100）此前对 Kimi-K3.1 与 DCP 均不工作：① FP4 index-K store 忽略 DCP rank ownership，某 rank 写了不属于它的 widened slot；② 合并的 no-RoPE key/head-gate prologue 把 query 量化到 FP8 传给期望 BF16 query 的 FP4 scorer，**静默拉低 top-k 选择质量**。修为 store 只写 `slot//dcp_size` 自有 slot、`_fused_q_quant` 保留 BF16 query 并缩放 BF16 head gate。**2×4 GB300、TP8/DCP8**：KV pool +8.8% tokens（index key 132→68 字节/token/层）；GSM8K(200) FP8 99.0% / FP4 98.5%（FP8 二次 99.5%）；102,912-token 缓存前缀后 prompt logprobs FP4 vs FP8 top-1 一致 99.72–100%、top-5 KL 0.0001–0.015；no-RoPE prologue 修前 top-k overlap 掉到 81–88%、修后 logits 与未合并路径逐位相同。未覆盖 FP4 LiteTopK prefill、HiCache/PD with FP4 keys、吞吐。注：FP4 98.5% 为作者单次评测口径。

### 3.4 `#51386` [Fix] DFLASH draft KV ring 默认关闭（fix，强化第 3/14 条·默认值风险）

#50740 把所有滑窗 MHA draft 的 DFLASH draft KV ring 默认打开（叠加 #49664 的 MLA ring），但每个 PD role 仅凭自身启动 flag 决定是否用 ring——当一侧能用、另一侧不能时，两 server 注册不同的 state entry，decode scheduler 在 bootstrap 崩溃（`PD state entry mismatch: only the decode server registers a DFLASH draft KV ring`），heterogeneous-TP PD 同样失败。这是上一期 #50740「+19.7% KV 容量但命中后接受长度 −7.5~17.7%」那条默认化的**回退闭环**。修为 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING` 默认 `False`（改 opt-in），且开关开启但本 launch 无法服务 ring 时在启动期抛 `ValueError`（原逻辑未设开关时静默 fallback，正是 PD 不匹配根源）。

### 3.5 `#35637` [Feat] UniBoost 尾部感知调度策略（feature，opt-in）—— 见第 1.3 节

### 3.6 `#51296` / `#51295` / `#51300` / `#51302` Inkling-2-Small 拆分栈（feature，强化第 13/15 条）

`#50186`（Inkling-2-Small 支持）拆成多 PR 独立合入：`#51296` 把 MXFP8 KV 的 UE8M0 scale 行存进 unified memory page（见 1.3）；`#51295` decode slot 分配与 `req_to_token` 写入融合进一次 launch（从 #50186 原样迁出，本分支未单独跑精度/速度，CI 跑单测）；`#51300` 把 Inkling2 MTP draft 的 hybrid-SWA sidecar pool 按 target 的 KV/SWA 索引放到对应 page（PD 转移与 host restore 时 draft 随 target 走）；`#51302` 加 SM90 Marlin/expert/MTP kernel ops；`#c97d47b87` 加 inkling2 renderer spec。均标注 CI 状态，多个 Base ✓ / Extra ⊘ skipped / AMD ROCm 10 进行中。

### 3.7 `#51162` [Fix] SM100 保留 AWQ/GPTQ checkpoint 的 Marlin 转换（fix，强化第 3 条）

1×GB200 上用默认设置启动 `Qwen3.5-35B-A3B-GPTQ-Int4` 在加载权重前报 `ValueError: Unknown quantization method: gptq`（同 checkpoint 在 H200 可启动）。根因：SM100 上 Qwen3 MoE 家族（含 Qwen3.5/Qwen3-VL MoE）、Glm4MoeForCausalLM、DeepSeek 家族及 MiniMax-M3 override 把 checkpoint 的 `quant_method` 拷进 `server_args.quantization`，`ModelConfig._verify_quantization` 视其为显式选择而不切到 `gptq_marlin`/`awq_marlin`。修为对 loader 可解析的 `{awq, gptq}` 不再写 `quantization`，交回 loader 解析。

### 3.8 `#50767` / `#51455` PD 解耦健壮性（feature/perf，强化第 5 条）

`#50767` 让 `--enable-unified-memory` 支持 NIXL PD 解聚，复用与 Mooncake 相同的共享 arena 与物理索引传输契约（无需改 NIXL 库；需两端 attention TP/page size/model 与版本匹配；Base/AMD CI 进行中）；`#51455` DCP-1 prefill→DCP-N decode 转移时把等宽 collapsed draft 并入 target 行的 pack，16K token chunk @ DCP8 ×7 draft 层的描述符数量从 114,688 降到「每目标 page run 一个」（**计算值、未实测**，无 GPU PD run）。

### 3.9 其他要点（标题/描述级）

- **DSA/indexer**：`#51515` 融合 FP4 indexer key store 与 head gate；`#51297` 每层 FA override 与 block-anchored windows；`#45274` TileLang sparse decode split 取 tile count 任一因子（标题级）；`#44217` CUDA 上 raw FP8 KV cache + Q8 prefill。
- **diffusion/flux3 大簇（约 15 条，标题级）**：`#51411` conditioning cache 编码 prompt；`#51380` CFG parallel 下静态 stream；`#51406` 仅 `--vae-sp` 时分片 VAE decode；`#51387` 原生 SwiGLU 走共享 SiluAndMul；`#46603` GPU BLAKE3 加速 conditioning hash；`#51554` Ulysses exchange 按 payload 选 transport；`#49532` Cosmos3 Bimanual VAE fast path；多条 Doc/Test。
- **量化/MoE**：`#46401` SM103 单 token BF16 MoE kernel；`#49275` RTX PRO 5000 tuned channelwise FP8 GEMM tile；`#f29cc397a` qwen4 offloaded PLE table 按精确尺寸 pin。
- **调度/router**：`#51521` scheduler 把 `Engine.collective_rpc` 走 control request 路径；`#42314` sgl-router 在 /v1/responses 提供 OpenAI Responses API；`#51415` `--decode-dp-rank-policy` 与按期望 KV 的 least_tokens；`#51607` draft 收窄到 plain TP 时清 `moe_dense_tp_size`。
- **linker/内存**：`#49127` external cache linker 暴露 lookup/load/offload/wait 指标（修 linker 命中被误记为 host L2 命中、Rust 前端完全不计的问题）；`#46499` UnifiedL3 支持 mooncake storage backend；`#46953` weight_cache vmm_fd transport。
- **安全/多后端**：`#47977` `--disable-multimodal` 以 text-only 服务多模态模型；`#50518` tokenizer worker 全部 drain 后才停 scheduler；`#51270` Rust-server 下 `--tokenizer-worker-num>1` 与 `--grpc-port` 兼容；NPU/Intel XPU 多条（`#40580` MindIE-SD sparse attention、`#47961` Intel XPU EPD 多模态、`#49790` Qwen3.5 MTP fc 量化适配）。

---

## 4. TensorRT-LLM：2026-11-19 提交（51 条）—— GMS 共享存储视图(#22563) + mHC NCCL 窗口所有权(#22321) + PrimTS FMHA/MLA API 兼容(#21151) + DSv4.1 SM107 MXFP8 prefill(#22358) + GLM-5.3-Flash sm_103(#22621) + NVFP4-FP8RoPE layout(#22080) + 大量 infra/waive

### 4.1 `#22563` [fix] GMS 迁移保留共享存储参数视图 —— 见第 1.2 节

### 4.2 `#22321` [fix] 恢复 mHC 激活在 NCCL window scope 的所有权（fix，强化第 8 条）

DeepSeek-V4/V4.1、TP>1 且 attention DP 关闭时，MoE all-reduce 可能产生 NCCL-window 张量到 `HCState.x_prev`。eager decoder-layer 输入收集排除普通 dataclass 作共享元数据，故遗漏这个被消费的激活，其租约一直留在外层 model scope 直到 forward 结束，**跨层累积 window-pool 内存、压缩可用 KV cache 容量**。修为把 `HCState` 注册进 `nccl_window_tensor_container()` 并用共享 marker/谓词收集其激活张量，由既有 `beginTensorLeaseScope` 完成所有权转移。8×B200 DSv4-Pro FP4 min-latency（TP8/EP8，batch 128，8192/1024）三次交错验证（作者报告）。

### 4.3 `#21151` [fix] 恢复 PrimTS FMHA/MLA task API 兼容（fix，强化第 15 条）

vendor FlashInfer patch 移到 commit 31fe2dbd，适配 DSL 4.9/4.8 两条路径的 Task API（增加 older/tree-walking Task API 支持，`task_compat.py` +226 行）；新增 `precompile_only()`/`probe_cached()` 与 `KernelBusy`/`KernelNotCached` 异常、sparse MLA decode launch caching。覆盖 FMHA context/decode、MLA decode（nvfp4_kv、throughput_2cta、throughput_latency_1cta）。作者报告 block-sparse attention 上约 >1.5% 差异并归因于生成的 FMHA kernel；**无 CI stage 安装 DSL 4.9，结果未独立验证**（coderabbit 评审），按作者报告/描述级记录。

### 4.4 `#22358` [perf] DSv4.1 SM107 MXFP8 prefill GEMM + CSA2 投影 + self-sampling GVR decode（perf，描述级）

为 DSv4.1 启用 SM107（Rubin）MXFP8 投影与 CSA2 self-sampling decode：prefill 尺寸 MXFP8 Linear 用固定 `CuteDSLMXFP8RubinLinear` tactic（M≥1024，不支持则保留 native GEMM）；Q-B 在 M=1024–2048/N=32768/K=1280 用 K64 MMA + 256×256 tile；O-B 对 CSA2 TP1 投影（N=5120,K=8192）显式允许 `cutedsl_sm107`；CSA2 单独开 `sm107_decode=True`，候选 consumer 保留精确选择、>256 行 candidate-source 保留精确 radix top-k；GVR phase policy 保留 148-SM wave 预算。无端到端性能数字，按描述级记录。

### 4.5 `#22621` / `#22080` GLM5.3 目标扩面与 NVFP4 KV layout（feature，强化第 11/13 条）

`#22621` 把 GLM-5.3-Flash block-FP8 Modeling V2 target（`glm5_next/glm_5_3_flash_fp8__sm_10x__tp4`）从仅 sm_100 扩到 sm_103（B300/GB300），native kernel 以 `ARCHS 100f` 覆盖 sm_103 无需改构建。**4×B300 / 单 GB300 tray**：routing 116 passed、target operation 315 passed、4-rank all-reduce+mHC 边界 passed；`test_gsm8k_mtp3`（参考 92.91）为 92.80/93.82、接受长度 3.390（与 B200 的 3.38 一致）；`test_gsm8k`（参考 92.57）为 91.36/90.45（**低于参考，作者报告口径**）。`#22080` 加 NVFP4 DSA KV-cache layout selector `TRTLLM_NVFP4_MLA_KV_CACHE_LAYOUT`（`auto`/`residual`/`fp8rope`，默认 `auto`），`fp8rope` 每条 latent 记录 352 字节（NoPE 用 E2M1 + inline E4M3 block scale、RoPE 用 E4M3），需 DSACacheManagerV2 且限 generation-only/无 aux cache 等条件，注册 `prims_ts_dsa_sparse_mla` adapter 直接读 fp8rope 记录。coderabbit 指出三处"是否跳 gather"决策路径的一致性 diff 未验证。

### 4.6 其他要点（标题/描述级）

- **采样正确性**：`#22303` RAW logprobs 在 penalty/token-ban 等 in-place 改写 logits **之前**按行 `index_select` 保存原始行、采样后 `index_copy_` 恢复（只影响需 raw logprobs 且请求改写 logits 的行，内存随合规行数而非 batch 增长）；`#22194` chat 请求接受 `top_logprobs=0` 无 logprobs。
- **内存/GMS**：`#22563` 见 4.1；`#22321` 见 4.2；`#21835` 按需用更多 torch CPU 线程加载权重。
- **config/部署**：`#22517` qwen4_exp PLE host table 默认从 `rank` 改为 `node`（每 host 存一次；coderabbit 判 fallback 路径与 server/rank 模式未被 diff 覆盖，needs follow-up，属默认值风险第 3 条）；`#22498` Muse Spark Open GB300 AgentX serving recipes；`#22247` Kimi K3 Modeling V2 target 不再依赖 Modeling V1 model code。
- **perf/waive/infra（多数标题级）**：`#22513` 开 prefill CUDA graph backend 时跳过无引导的 eager guided-decoding；`#22462` MegaMoE scheduling 支持 dummy all-reduce；`#22080`/`#22358`/`#22621` 见上；多条 `[nvbugs] Waive N failed cases`（#22601/#22607/#22567/#22591/#22589/#22593）、restore B300 attention timeout/准确性覆盖（#22409/#22488）、UCX timer-wheel backport（#21381）、C++ license validation（#22447）、Staircase/Beam 测试套件。

---

## 5. llama.cpp：2026-11-19 提交（28 条）—— GGUF 加载越界(#32601, 安全) + mul_mat_q int32 溢出(#32374) + 重复专家 ID(#32505) + IQ2 量化加速(#31910, 3.04×) + Metal CONV_2D M5(#32542, 至 19.79×) + 多后端

### 5.1 `#32601` [Security] HRM GGUF slot count uint32 回绕越界写 —— 见第 1.2 节

### 5.2 `#32374` [CUDA] mul_mat_q 用 64 位 dst offset —— 见第 1.1 节

### 5.3 `#32505` [CUDA] 支持重复专家 ID（feature/perf，强化第 7 条）

为 CUDA 实现重复专家 ID 支持：重写 mmid helper kernel 用 `__popc`/`__ballot`（比原 `__shfl` 更快，加 `ggml_cuda_popc` 兼容 warp=64 的 AMD）；处理 batch≤16 且不用 helper kernel 的 MMF kernel 在 ID 不唯一时从根本上失效，故移除该不含 mmid 的 MMF 变体（作者称与现 mmid 路径性能差仅几个百分点）。General MoE（MI60/MI50 granitemoe 3B Q4_0）pp4096 t/s 基本持平（1.0–1.01×）。属 MoE 路由/专家归属正确性扩展（第 7 条）。

### 5.4 `#31910` IQ2 量化创建提速 2–3×（perf，非推理路径）

通过缓存重复 pattern 加速 IQ2 量化**创建**（非推理）：Qwen3.6-35B-A3B 32 线程下 full quant speedup IQ2_XS 3.04× / IQ2_S 2.99× / IQ2_XXS 2.08× / IQ2_M 1.46×。作者称字节级相同，跨 8 模型 5 架构（qwen35/qwen35moe/granite/gemma4/bailingmoe）测试。注意这是**离线量化速度**，与 decode/prefill 推理性能无关。

### 5.5 `#32542` [Metal] CONV_2D M5 tensor API + 全芯片 split-K（perf，硬件特定微基准）

#28069 的后续：为 M5 加 tensor API 实现（CONV_2D 走 tensor core matmul，M1–M4 保留 simdgroup kernel），两个 kernel 都支持长 reduction 的 split-K（部分和按固定顺序相加，结果确定）。`test-backend-ops perf -o CONV_2D` F16 M5 Max：多个形状 4.77×/2.36×/3.34×/5.80×/19.79×（最高 19.79× 为 stride-2 音频 shape），22 case perf suite 结果正文截断。**微基准、形状特定**，非通用推理加速。

### 5.6 其他要点（标题/描述级）

- **模型/注意力**：`#32429` Inkling 2 attention mask 修复（加 `llm_graph_input_attn_rel_pos` 处理相对位置输入）；`#32387` ggml 给 ROPE/SET_ROWS 加 BF16 支持（CPU/CUDA）；`#32589` qwen3vl 为 EAGLE-3 设 `t_layer_inp`。
- **多后端**：`#32577` OpenCL decomposed FA 的 KQ mm 支持 dp4a；`#32522` OpenCL 允许调 decomposed FA 的 KV padding；`#31045` Vulkan MMQ 支持混合符号整数 dot；`#28528` Vulkan stream-k MUL_MAT；`#32379` Vulkan DIAG/TRI 对 storage-only FP16 用浮点零；`#19590` WASM Relaxed SIMD 增强；`#32590` ggml-openvino CI 修复。
- **加载/健壮性**：`#32545` EIO 时回退 buffered I/O；`#32443` mtmd 支持 voxcpm2；`#32606`修复 YuE2 projector 加载；`#32557` 移除 PATH_MAX 依赖。
- **其他**：`#32555` UI 首载 JS 减 63%（按需加载图表/数学/PDF）；`#32387`/`#27258` backend scheduler tests；`#32602` Metal NSMakeRange 第二参数修复；`#32565` CI 用 LLVM 22；`#32597` release badge 修复。nightly 推进 b12552–b12556（prerelease）。

---

## 6. 累计结论延续与新增（继承 1–22，本期新增 23）

**累计结论（1–22 主线回顾）**：① 调度器隐式阻塞的定量画像（本期 SGLang #35637 UniBoost 从调度策略层直面长尾）；② 发版窗口的双向性；③ 默认值成为主要风险载体（本期 SGLang #51386 DFLASH ring 默认化后被回退、TRT #22517 PLE host table 默认 rank→node）；④ 异构/量化后端静默错误为高频（本期 vLLM #68072 ROCm NVFP4 KV 错切 garbage、SGLang #51450 FP4 indexer query 误量化拉低 top-k、#51162 SM100 AWQ/GPTQ 启动失败）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（本期 vLLM #67947 水印跨 prompt 边界、SGLang #51300 hybrid-SWA sidecar draft pool）；⑥ 量化路径静默错误（scale/布局漏接，本期 vLLM #68084 SwiGLU clamp 漏接、#68353 MXFP8 N 未对齐）；⑦ 专家张量/路由归属与分页漏接静默错误（本期 llama.cpp #32505 重复专家 ID、vLLM #68084/#68072）；⑧ KV 卸载/HiCache/PD 健壮性 + 内存/缓存管理接口重构（本期 vLLM #63294 encoder cache 分层、#68586 shared KV extend 崩溃、#68710 CPU offload shutdown、SGLang #51289 DCP 二次局部化、#50767 NIXL unified PD、TRT #22321 mHC window 内存）；⑨ 长上下文 int32/int64 溢出静默错误（本期 vLLM #68386 fused_qk_norm_rope int32、SGLang #50775 KDA 2**31、llama.cpp #32374 mul_mat_q 32 位 dst、#32601 HRM uint32 回绕）；⑩ MXFP8 KV 正确性簇（本期 SGLang #51296 MXFP8 scale 入 unified page、vLLM #68353）；⑪ 新旗舰模型 CP/KV/性能成为一等公民（本期 SGLang #51450 Kimi-K3.1 FP4 indexer + DCP、#51289 Kimi-K3.1 DCP、TRT #22621 GLM-5.3-Flash sm_103）；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径（本期 vLLM #66812 GLM-5.3-Flash SM90 block-FP8 MegaMoE +12~24%）；⑬ FP8/低比特 KV cache 跨引擎标准化落地（本期 TRT #22080 NVFP4-FP8RoPE layout、SGLang #51296）；⑭ 投机解码验证路径跨引擎密集硬化（本期 vLLM #68164/#67947、SGLang #51386/#51458）；⑮ 跨后端可移植性成为显式架构主线（本期 vLLM #66133 gfx1151、#63396/#63243 gfx1250、TRT #21151 DSL4.9/4.8 兼容、llama.cpp 多后端簇）；⑯ 新旗舰模型进入「默认性能化」阶段（本期 vLLM #66812/#66631 去 launch、SGLang #51382 shared expert overlap）；⑰ 推理服务前端 Rust 化（本期 vLLM Rust 前端约 10 条、SGLang sgl-router Responses API）；⑱ 推理服务默认安全加固（本期 vLLM #67348 max_tokenize_tokens 防 DoS、llama.cpp #32601 GGUF 越界拒绝）；⑲ 非量化的常规优化路径静默错误密度不亚于量化路径（本期 vLLM #68386 rotary、SGLang #51289 DCP、TRT #22563 GMS 视图、#22321 window 所有权）；⑳ 正式发版集中固化「默认值风险 × 破坏性变更」（本期无新正式版，TRT v1.4.0rc4 仍为最新 RC，延续上期结论）；㉑ 在线权重热更新的时序竞态成为静默正确性新子项（本期 vLLM #68171 requantization 峰值显存、#68048 layerwise reload 保留 Engram）；㉒ PDL 依赖同步次序成为跨引擎静默/崩溃新子项（本期无新增强证据，延续）。

**新增第 23 条（语义合并冲突成为静默正确性新子项）**：SGLang #51289 表明，当两个 PR 各自在其 base 上正确、修改不同行、且 CI 全绿而**干净合并**时，仍可能因**数据流同一环节被重复处理**（此例：indexer 与 `_forward_trtllm` 都做 DCP slot 局部化，后者对已局部化的 slot 再局部化一次）产出静默错误——Kimi-K3.1 DCP decode 用 prims_ts/trtllm 时 GSM8K 0.97→0.03、答案混入其他请求内容，而默认后端 flashmla_kv 因只被其中一个 PR 更新反而不受影响，导致常规 CI 无法发现。**研判（观点）**：这是与量化漏接（第 6 条）、PDL 次序（第 22 条）、权重时序（第 21 条）不同的新故障模式——根因在**跨 PR 的语义交互**而非单 PR 内部逻辑；风险放大器有三：① 修改点分散在不同文件、diff 无冲突；② 各 PR base 不同导致各自测试无法覆盖组合态；③ 默认路径恰好避开、使回归只在特定后端/配置下暴露。可执行做法：对同一数据结构（如 DCP slot/页表、KV layout、投机草稿池）的多处 transform 建立「单一所有权」约定与断言（如幂等性/来源标签），新增后端时以「默认+非默认」双路径跑精度门；对 cherry-pick 到 release/nightly 的组合态做针对性 smoke（本例正是 Day-0 镜像→main nightly 切换触发）。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 静默正确性复核（DCP 布局/量化/KV 切片/整型溢出）**：SGLang `#51289`（Kimi-K3.1 DCP + prims_ts/trtllm GSM8K 0.97→0.03，升级含本修复的 main/nightly；默认 flashmla_kv 不受影响但 recipe 用 prims_ts）、`#50775`（KDA prefill 2**31 溢出，大/禁 chunked_prefill 的高上下文场景）、`#51450`（FP4 indexer 未对齐 query 致 top-k 静默拉低，作者报告 GSM8K 98.5%）；vLLM `#68084`（DSv4-Flash batched DeepGEMM 漏 SwiGLU clamp，9/16384 token 错）、`#68386`（fused_qk_norm_rope int32 + NeoX rotary，GLM-5.3-Flash 视觉 960K patches IMA）、`#68353`（MXFP8 N 非 8/32 倍数 garbage）、`#68072`（ROCm NVFP4 KV page 错切 + MLA 型 indexer cache，MiniMax-M3 heterogeneous TP）、`#68164`/`#67947`（投机接受估计器 +inf / 水印跨 prompt 边界）、`#68586`（HiSparse host-resident 共享 KV extend KeyError）；llama.cpp `#32374`（mul_mat_q dst >2^31 越界，大词表+q8_0 输出头）、`#32601`（HRM GGUF uint32 回绕越界写，加载不可信 GGUF 场景）；TRT `#22563`（GMS 共享存储视图漏拷）。
- **P0 — 语义合并冲突审计（第 23 条）**：对 KV slot/页表、KV layout、投机草稿池等被多处 transform 的数据结构建立单一所有权与幂等断言；凡新增 attention/量化后端，"默认 + 非默认"双路径均跑精度门；对 Day-0 recipe 镜像与 main nightly 的组合态做针对性 smoke；cherry-pick 组合态单独回归。
- **P1 — 默认值/回退核查（第 3 条）**：SGLang `#51386`（DFlash draft KV ring 改回 opt-in，此前 #50740 默认化导致 PD bootstrap 崩溃；若依赖 ring 容量需显式 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1` 且两端一致）、TRT `#22517`（PLE host table 默认 rank→node，需确认 host 支持 node 及 fallback 路径）。
- **P1 — KV 容量/分层/性能权衡（口径特定）**：vLLM `#63294`（encoder cache FS 分层 TTFT −34.41%/吞吐 +25.84%，Qwen3-VL-8B H800、prefix cache off、fsync off 的作者口径）、`#66631`（HiSparse unpin 省 schedule 步内 12 ms/GC 尖峰）、`#68171`（online requantization 峰值 287.9→147.9 GiB）、`#68533`（draft-only 桶不再撑大 group，全注意力 KV −20%/token）、`#66812`（SM90 MegaMoE +12~24% tok/s/GPU，KV 池增益是主因）、`#66133`（gfx1151 W4A16 +37~44%，硬件/形状特定）；SGLang `#35637`（UniBoost 学术策略，头部结果为旧版本作者报告）、`#51382`（MiniMax-M3 shared expert overlap +3.8%@32×200k）、`#51455`（DCP collapsed draft pack 描述符减少为计算值未实测）；llama.cpp `#31910`（IQ2 量化**创建** 2–3×，非推理）、`#32542`（Metal CONV_2D 微基准至 19.79×，形状特定）；TRT `#22621`（GLM-5.3-Flash sm_103：`test_gsm8k` 91.36/90.45 低于参考 92.57，需关注精度回退）。

---

## 7. 社区与视频资源（HN Algolia，2026-11-19 epoch 窗口 1795046400–1795132799）

> 以下均为社区/Show HN，非四家引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词检索中 `SGLang` / `TensorRT-LLM` / `llama.cpp` / `KV cache` / `speculative decoding` **均 0 命中**；`vLLM` 23 条多为泛 AI 资讯与模糊「LLM」匹配；`inference engine` 2 条均为无关创业帖。

- **「Taking vLLM Apart: A Practical Guide to Disaggregated Serving」（2 分）**：2026-11-19T03:19，社区对 vLLM PD 解聚的实操拆解文章，非 vLLM 官方一手、内容未核验。**二手线索/未核验**。
- **「High-Interactivity Decode with FlyDSL MonoKernels in vLLM」（1 分）**：2026-11-19T03:21，社区讨论 vLLM FlyDSL monokernel 高交互 decode（呼应本期 vLLM #66184 mono decode 主线），非官方一手、未核验。**二手线索/未核验**。
- 其余 HN 命中（如「Previous-Token Prediction Based LLM Inversion」47 分、「22 local LLMs benchmark」3 分等）为泛 AI/安全/资讯，与推理引擎实现无直接关系，未列入。本期**无引擎相关的可确认官方视频**；Reddit / YouTube / X 未采集。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **124**（分两页 100+24，page2 返回 24 条，合并去重后取全）/ SGLang **82** / TRT-LLM **51** / llama.cpp **28** 合计 **285** 条（2026-11-19 UTC 窗口；周四，工作日正常量）。与上一期（2026-11-19 档案覆盖 11-18）连续，无空档。
- **版本核验**：releases API 确认——**窗口内无新正式版、无新 prerelease**：vLLM v0.34.0（11-16）、SGLang v0.5.23（11-02）、llama.cpp 正式版 v1.0.0（11-13）、TRT-LLM 最新 **v1.4.0rc4**（2026-11-18T19:16:49Z，prerelease，**落在本窗口之前**，属上一期已记录的版本事件，本期不重复作为头号事件）；llama.cpp 推进 nightly b12552/b12553/b12554/b12555/b12556（prerelease）。发布说明为精选 highlights，按本追踪方法论不作为「修复已进版」依据。
- **PR 正文来源**：**48 个关键 PR 正文经官方 issues API 抓取**（markdown 官方一手；vLLM 18 + SGLang 16 + TRT-LLM 8 + llama.cpp 6），配额充足（core 57/60 起、用至 9 remaining）、无 IncompleteRead、无降级为标题级；所有 PR 号均在窗口 commits API 中确认为已合入 main 的提交。其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度。
- **描述级/未核验标注**：vLLM `#63294` 性能为作者在固定配置（Qwen3-VL-8B/H800/prefix cache off/fsync off）下自报、`#66812` KV 池增益作者明示是领先主因、`#66133` 为 gfx1151 形状特定微基准（独立复跑 +36.9%）、`#66631`/`#68171` 为作者报告、SGLang `#35637` 头部结果为作者移植到 v0.5.13 的早期版本所测（非本 PR 版本）、`#51450` FP4 98.5% 为作者单次评测、`#51455` 描述符减少为计算值未实测且无 GPU PD run、`#51295`/`#51300`/`#51296` 精度/速度沿用 #50186 未在本分支单独跑、多个 inkling2 PR CI Extra ⊘ skipped/AMD 进行中；TRT `#22621` `test_gsm8k` 低于参考（91.36/90.45 vs 92.57，作者报告）、`#21151` DSL4.9 无 CI stage 安装故未独立验证、`#22080` 三处"跳 gather"决策一致性 diff 未验证（coderabbit）、`#22517` fallback/server/rank 模式未被 diff 覆盖（needs follow-up）、`#22321` 性能为作者报告；llama.cpp `#31910` 为离线量化创建速度（非推理）、`#32542` 为形状特定微基准（22 case 表截断）、`#32505` verify 表为作者脚本生成、`#32601` 因 SECURITY.md 未开公开 issue（安全上下文自述）。均已按实际缺失程度标注。
- **社区信号**：HN Algolia 引擎官方更新类故事无；「Taking vLLM Apart」（2 分）与「FlyDSL MonoKernels in vLLM」（1 分）按二手线索/未核验记录；无引擎相关的可确认官方视频。
- **技能检索（find-skills / agent-browser）**：find-skills 经 SkillHub 语义检索（EN+ZH 各 1 组）返回通用编程/生产力/搜索类技能，最高相关性仍约 **0.050**，与「官方一手提交 API + 证据分级归档 + 指定目录归档 + git 提交推送」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/issues API）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次一致。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/作者报告/计算值未实测/CI 未全绿/needs follow-up/微基准/形状特定/prerelease」。
