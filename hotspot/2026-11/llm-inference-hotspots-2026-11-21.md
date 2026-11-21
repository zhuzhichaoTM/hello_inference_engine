# 大模型推理优化每日热点档案

> 档案日期：2026-11-21（本自动化于该日执行）
> 关注窗口：2026-11-20（周五）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口，vLLM 与 SGLang 各分两页 100+22 / 100+24）+ releases API + 约 40 个关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 均经官方 issues API 抓取 markdown，官方一手；配额充足无降级）+ HN Algolia（HTTPS，`numericFilters` 按 2026-11-20 epoch 窗口 1795132800–1795219199）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 122 / SGLang 124 / TRT-LLM 49 / llama.cpp 43 合计 **338** 条（2026-11-20 为周五，工作日正常量，较上一窗口 285 回升）；**窗口内有新正式版本——TensorRT-LLM v1.3.0 于 2026-11-20T02:45:00Z 正式发布（非 prerelease，18,233 字符发布说明全文核验），为独立稳定线（tag commit c66df86d，日期 11-16；与 main `diverged` ahead 1604/behind 210）**；其余无新正式版（vLLM v0.34.0 11-16、SGLang v0.5.23 11-02、llama.cpp v1.0.0 11-13），TRT-LLM 滚动 RC 最新仍为 v1.4.0rc4（11-18，落窗口前），llama.cpp 推进 nightly b12588–b12600（prerelease）。HN 引擎官方更新类故事无；引擎关键词高信号弱：仅 1 分 KV cache SQLite 审计帖（二手线索/未核验），`sglang`/`TensorRT-LLM`/`llama.cpp`/`speculative decoding`/`paged attention` 均 0 命中。

---

## 1. 版本状态与本期头号事件：TensorRT-LLM v1.3.0 正式发布（独立稳定线）+ 静默正确性跨引擎延续

**提交计数（2026-11-20 UTC，均 HTTP 200；周五）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 122 | 分两页 100+22；ROCm/AITER 大簇 + Rust 前端 + 安全簇 + 静默正确性 + KV/连接器 |
| SGLang | 124 | 分两页 100+24；rust-server/diffusion 大簇 + PD 死锁 + hiCache/KDA + AMD/QSA |
| TensorRT-LLM | 49 | Modeling V2 / GMS / RocketKV / KVCacheManagerV2 主线，含 20+ infra/waive/test |
| llama.cpp | 43 | 多后端（CUDA/Metal/OpenCL/Vulkan/WebGPU/CPU/OpenVINO）+ UI 大簇 + MoE cache |
| **合计** | **338** | 工作日正常量；与上一窗口（11-19 的 285）连续，无空档 |

### 1.1 头号事件：TensorRT-LLM v1.3.0 正式发布（独立稳定线，非 main 滚动 RC 的延续）

- **事实（官方 releases API + 发布说明全文，官方一手）**：v1.3.0 于 **2026-11-20T02:45:00Z** 发布，`prerelease=false`，发布说明 18,233 字符全文核验。关键点：
  - **发布线性质**：tag commit `c66df86d8aab`（日期 **2026-11-16T20:54:50Z**，msg 为 `#22398 use Apache license for LTX-2`），与当前 main `compare/v1.3.0...main` 返回 `diverged, ahead_by=1604, behind_by=210`。即 **v1.3.0 是独立稳定线正式版，并非 v1.4.0rc1–rc4 滚动 RC 序列的直接延续**；窗口内 main 仍在推进 v1.4.0 开发（`#22627` Inkling 2 Modeling V2、`#22651` GLM-5.3-NVFP4 sm103 等）。
  - **大量 API BREAKING**（与历次 RC 中记录的变化高度呼应，正式版集中固化）：TensorRT backend 整体移除（PyTorch 为唯一后端，`trtllm-build`/`TrtLlmArgs`/`BuildConfig` 移除）；Legacy C++ Executor 与 runtime/batch-manager API 移除（`cpp_only` 移除）；TensorRT-engine Triton 后端模型移除（仅留 `llmapi`）；`TRTLLMSampler` 移除（`TorchSampler` 唯一、需 FlashInfer）；two-model 投机解码移除（`eagle3_one_model`/`mtp_eagle_one_model`）；AutoDeploy/Star Attention/WIDEEP 移除；CLI flags 覆盖 YAML config；KV cache transfer 超时默认 60 s；per-request chat template 改 opt-in；logprobs 封顶 100；`/perf_metrics` 端点移除；动态投机字段重命名；过期选项移除。
  - **核心能力（发布说明级）**：Rubin(SM107) preview（Vera Rubin 上验证，DSv4/Kimi K3）、KV Cache Manager V2 默认（多数模型族）、NVFP4 cold-page compression（默认关）、TriAttention/disk cache、per-conversation block reuse 与精确多模态 KV block hashing、FlexKV/LMCache 连接器；投机解码 DSpark/DFlash/Suffix Automaton/PARD、EAGLE3 dynamic trees、one-model acceptance-rate auto-disable；MoE 新增 `DENSEGEMM`/`MEGAMOE_DEEPGEMM`/`MEGAMOE_CUTEDSL`/`CUTEDSL_FC12`(Rubin only)/`MARLIN`，NCCL-EP 与最多 2048 专家；量化新增 MXFP8(W8A8) linear/MoE、W4A16 NVFP4 via Marlin（Hopper/Ada 可跑 NVFP4 checkpoint）；稀疏 DSA FP4/FP8 decode、GVR Top-K、Skip Softmax Attention、MiniMax-M3 MSA；VisualGen 视频/图像生成（FLUX.1/2、Wan 2.2、LTX-2、HunyuanVideo 1.5、Cosmos3、MiniMax-H3）。基础设施：CUDA 13.4.1 / PyTorch 2.14.0 / transformers 5.5.4 / FlashInfer 0.6.18 / Triton 3.8.0 / NCCL 2.31.2 / NIXL 1.4.0 / CUTLASS 4.4.2 / xgrammar 0.1.32 / C++20，Python 3.10/3.12。
  - **Known Issues（发布说明明示，官方一手）**：Kimi-K3 NVFP4 随 bundled transformers 可能无法加载 tokenizer；**解聚 + DSpark 投机时 draft KV 不转移到 generation worker，接受率降低（聚合不受影响）**；GLM-5.2 NVFP4 GB200 TP8/attention DP off 且 `max_num_tokens>4096` 时 warmup 可能 segfault；Nemotron-3-Super FP8 B300 解聚并发超 batch size 可能失败；Qwen3-32B-FP8 B300 解聚可能产出错误输出；encoder-decoder + FP8 KV 可能越界 cross-attention KV pool 致非法访存/错误输出；Qwen3-32B TP4 A100(p4d) 开 `aws-ofi-nccl` 时 autotuner warmup 可能 segfault。
- **研判（观点）**：这是累计结论第 20 条（正式发版集中固化默认值风险 × 破坏性变更）的又一实证，但性质与 v0.34.0 不同——**v1.3.0 属"稳定线迟到的正式版"，其 BREAKING 与 RC 序列内容大量重叠**。实践含义：① 由 RC 直接升级到 v1.3.0 **不等于**退步，二者是不同分支；② 按本追踪方法论，发布说明为精选 highlights，**不作为"某修复已进版"依据**，需按 PR 号做 compare 祖先核验；③ 上述 Known Issues 中"解聚 DSpark draft KV 不转移降接受率"与近期 PD+投机主线（第 5/14 条）直接相关，是部署前必读项。

### 1.2 静默正确性铁证（跨 vLLM/SGLang/TRT-LLM/llama.cpp，强化第 4/6/19/21/22 条）

- **vLLM #66586**（官方一手，fix，强化第 4/6 条）：gfx94x（MI300）FP8 为 `e4m3fnuz`（最大值 240 vs `e4m3fn` 448），标定给 `e4m3fn` 的 scale 需 ×2。`BaseKVCacheMethod` 会做，但 `CompressedTensorsKVCacheMethod.process_weights_after_loading` 覆写该 hook 且未调 `super()`，使 compressed-tensors checkpoint 的 FP8 KV cache scale 只有一半、写入时截断 key。MI300X 上饱和 key 占比 **0.9–1.5%→<0.001%**、GSM8K-text PPL 相对 bf16 KV 偏差最高 **20.5%→0.7%**；以 gfx950(MI350, stock 行为)为 ground truth，修后 gfx942 两后端结果贴近 MI350（如 TinyLlama/TRITON_ATTN PPL 4.6967→3.9247，MI350 3.9265）。以 gfx950 为交叉核验基准，属量化/异构 KV scale 静默错误。
- **vLLM #68749**（官方一手，fix，强化第 22 条）：AMD CI 自 #63910 起 `kv_cache K differs` 失败——batched kpool decode-update kernel 读取同一次 launch 中较早写入的 tail-ring slot 却无 barrier，ROCm 上读 wave 不一定是写 wave。4 waves × 64 lanes 覆盖 128 宽 head 两次，waves 0/1 写、四 wave 都读，陈旧值经 Hadamard 跨 wave 交换进入结果；int64 构建 57/6000 次失败、int32 有同样竞态但调度未暴露。修为每 token 加 `tl.debug_barrier()`（workgroup fence），57/6000→0/6000，12 形状开销 <0.05 µs。把第 22 条（依赖同步次序）扩展到 ROCm wave 级。
- **TRT-LLM #21220**（官方一手，fix，强化第 6 条）：FlashInfer 激活量化 tactic 把 SF-layout 请求传成 `do_shuffle`，而 pinned FlashInfer 0.6.18 中该选项置换矩阵行、与 `sfLayout` 无关，导致 tuning 选项改变 GEMM 前激活顺序。修为显式选 `SfLayout.layout_128x4`/`layout_linear` 且 `do_shuffle=False`。B200：新增 13 项回归修前失败、修后量化套件 31/31；LTX2 CUDA-graph TRTLLM-backend 测试 golden LPIPS **0.193212→0.0**（阈值 0.15）。QA 判 **needs follow-up**（新单测未在 CI list、所引 CI UNSTABLE/FAILURE），按作者报告的 B200 记录。
- **SGLang #50635**（官方一手，fix，CI 未全绿，强化第 19/21 条）：breakable prefill CUDA graph 捕获 KDA extend 时，replay 须显式等在途 layer load；#45839 的 `HiCacheGraphEvents` 仅当 `--enable-hierarchical-cache` 开启，而统一缓存 external linker 与之互斥，故 linker+KDA 模型（Kimi-Linear 类）机制从不启用——replay 在 linker load 落地前读 conv/recurrent state，**静默按 state slot 旧内容计算**。H200 注入 0.3 s 延迟：main 四请求 logprob 最大绝对差 4.9e-4~9.8e-4、修后全 0。CI：Base ✓ / Extra ⊘ / AMD ✗。属 positional/时序静默错误（第 19/21 条）。
- **SGLang #50841**（官方一手，fix，CI Base ✗，强化第 5/8 条）：`--hicache-write-policy write_back` 下，radix split 后 Mamba checkpoint 只在后缀、重算前缀在 Full KV 已入 host 的节点装了 device-only checkpoint，内部 Mamba 淘汰把"Full KV 已备份"当作充分（backuped 只覆盖 Full KV）而丢弃 state——前缀保留 KV 却丢 recurrent state；另 aux-only backup 成功返回 0 被当作分配失败。GB300 Python core split-checkpoint write-back 往返修前 8/8 失败、修后通过；Rust core 未在 GPU 重跑。属 KV cache/混合缓存管理正确性。
- **llama.cpp #31345**（官方一手，fix，强化第 4/6 条）：CUDA 上无 MMQ kernel 的量化类型（IQ1_M）batch>MMVQ 上限(8) 时回退 dequant+cuBLAS，默认 fp16 操作数/结果使 >65504 变 inf；`ggml_cuda_mul_mat_id` 建 `dst_slice` 时未设 `op`，致 `ggml_prec_set_acc` 请求的精度被忽略。A10(sm_86)：IQ1_M MUL_MAT_ID 配 BF16/F32 精度 main 出 inf、修后有限且匹配 CPU；默认精度不变。属量化回退路径精度漏接。
- **llama.cpp #32657**（官方一手，fix，强化第 6 条）：通用 scale pass 为 `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` 加载 per-expert scales，却漏掉合并的 `ffn_gate_up_exps`；本 PR 在该处加载 `ffn_gate_up_exps_s` 并在无独立 up 张量时暴露为 `ffn_up_exps_s`（因 `build_moe_ffn` 对合并 gate_up 输出施加 up_exps_s）。Inkling 2 据此像其他 MoE 一样传 expert scales。属量化 scale 漏接。

### 1.3 本期其他结构性观察

- **PD 解耦死锁两连（强化第 5 条）**：SGLang `#51882`（optimistic prefill 下，已占 decode reservation 的 bootstrapped 请求被 parked 请求持有的 KV 拒绝，双方互等到 PD 超时；原释放规则只在 row 用尽时触发，长 prompt 下 KV 池先满故不触发）与 `#51892`（stacked on #51882：storage L3 tier 下等待队列/无 reservation 的 optimistic 请求占住 anchor pin/host staging，已 bootstrap 请求 park 在 `anchor-lock cap reached` 并互等；修为按 scheduler 顺序提交 parked hit、让无 reservation 请求按 pass 让出 prefetch）。均调度/资源所有权修复，无数值变化；#51892 CI Base/Extra 红。
- **host work 削减持续见效（第 1 条定量画像延续）**：vLLM `#68351` 把 safetensors prefetch 从按全局 rank 改为按每节点 rank 分片，使每节点预热整个 checkpoint——Qwen3.5-397B-FP8 在 Lustre 上 TP8/2 节点模型加载 **244.7→78.3 s（−68%）**、DEP16/4 节点 105.5→43.2 s（−59%）、TEP8 63.9→47.5 s（−26%），time-to-/health −43%/−21%/−6%，贪心输出逐位一致；SGLang `#48208` 投机解码在每请求 top_logprobs k 相同时批量化 top-k 的 D2H 拷贝，GB300(Grace CPU) output tok/s **1627→2034（+25%）**、TTFT 583→318 ms（−45%）、TPOT −13.5%，而 H100(x86 host) 无收益（−1.3% 噪声内）——再次印证"收益来自 host 侧瓶颈而非通用加速"。
- **KV 分层/共享新形态（第 8 条）**：vLLM `#67539` 新增 Engram pool，把只读 Engram 表按行切到引擎组各 GPU，NVLink 或 RDMA（含 host 内存）取行且可在 CUDA graph 中捕获；GB200 NVL72 2×4 GPU、DSv4.1-Flash 两 DEP4 引擎，取 1 token NVLink 0.017 ms / RDMA 0.184–0.190 ms、8K 0.257/3.58–6.9 ms，model step 开销在噪声内（+0.1~0.8%），GSM8K 0.928/0.927；`peer_group_rdma_inflight=8` 默认限制 P/D 共享 NIC 时的 prefill gather 干扰（不设时 decode step +9.9%，设 8 后 +0.2%）。属 KV/表跨引擎共享的新数据结构。

---

## 2. vLLM：2026-11-20 提交（122 条，两页 100+22）—— 静默正确性(#66586/#68749/#68782/#65256) + 多节点加载预取(#68351, −68%) + EAGLE 块丢弃移除(#66234) + SM90 MXFP4 MegaMoE(#64289, +8~18%) + Engram pool(#67539) + 安全簇 + Rust 前端

### 2.1 `#66586` [Bugfix][ROCm] compressed-tensors FP8 KV scale 缺 ×2 —— 见第 1.2 节

### 2.2 `#68749` [Bugfix][ROCm] batched kpool decode 跨 wave 竞态 —— 见第 1.2 节

### 2.3 `#68782` [Bugfix][Attention] FlashInfer XQA ragged draft mask 去 GPU↔CPU 同步（fix，强化第 1/14 条）

`_make_xqa_ragged_draft_block_mask`（#49718 引入）用阻塞 H2D copy 和两次无 `output_size` 的 `repeat_interleave`，每个 ragged step 同步 host；CI 镜像设 `VLLM_GPU_SYNC_CHECK=error` 时直接杀死 engine core。DGX Spark(GB10, sm_121) 上 MTP + `--attention-backend FLASHINFER`：main eager GSM8K **0.074**、CUDA graph 0.187（均 engine died），修后 **0.9113 / 0.9174**、接受长度 3.54、Invalid 0。修为在 host 侧用 `qo_indptr_cpu` 构行索引，一次 pinned 非阻塞 `async_tensor_h2d` 送到 GPU（位相同），替代 GPU 端 cumsum/repeat_interleave/arange。CI 无 XQA+ragged spec decode 步骤覆盖，新单测加入 H100/B200 V1 Attention 步骤。属"调度器隐式同步"与投机正确性交叉。

### 2.4 `#65256` [Quantization] E8M0 scale 字节按字节加载进 uint8 参数（fix，强化第 6 条）

MX scale 参数（compressed-tensors MXFP4/MXFP8、ModelOpt MXFP8、Quark OCP-MX、INC MXFP4/MXFP8）是 uint8，源为 `float8_e8m0fnu` 时 `copy_` 按值转换（2^(b-127) 转整数）存错 scale。修为 `copy_weight_` 在字节容器（`uint8`/`int8`/`float8_e8m0fnu`/`float4_e2m1fn_x2`）间按 `src.view(dst.dtype)` 字节复制；真实数值格式（e4m3/e5m2）间仍按值转换以免静默错值。覆盖所有以 `copy_weight_` 结尾的 loader；DFlash/DSpark(MiniMax-M3.1 DSPark 等) MXFP8 context K/V loader 改用 `copy_weight_`（否则 QKV 投影保住 scale 而 context K/V 丢）。CPU 127 passed、main 上 14 fail。**局限**：load plan 首次更新固定每名字转换类型，后续以不同 dtype 更新会按值转换存错字节，发送方须保持 dtype 固定。

### 2.5 `#66234` [Refactor][KV Cache] 移除 EAGLE 块丢弃、lookahead hashes 成唯一行为（refactor，强化第 5/14 条）

Lookahead block hashes 已被核心前缀缓存与所有 hash-keyed in-tree KV connector 使用，前缀命中即证明 draft KV 在块边界可复用，legacy"丢弃最后匹配块并在各处补偿"不再需要。七个 commit 逐模块移除 `disable_eagle_block_drop` 等开关、proof margin、Mamba checkpoint backoff、eagle group 标记链、各 connector 复制的规则，统一为 `get_reusable_prefix_length(N,a,L)=floor((N-max(1,L))/a)*a`（L=1 EAGLE/EAGLE3/单模块 MTP、k 模块 MTP 为 k、否则 0）。默认配置下四项行为可见：重放边界多留一状态、Gemma4 MTP 用 plain hash(L=0)、多模块 MTP 的 `scheduler_block_size>=k` 检查移除、`supports_lookahead_block_hashes()==False` 不再禁用而仅告警。同代码两臂（baseline 恢复旧行为）对比：每组每题恰好多命中一块（Qwen3.6-35B-A3B +1088 tokens 等），LongBench-v2/gsm8k/gpqa 与接受长度基本持平（部分小幅波动）。属投机解码 × 前缀缓存所有权重构。

### 2.6 `#68351` [Bugfix][Model Loader] 每节点预取整个 checkpoint —— 见第 1.3 节

### 2.7 `#64289` [Perf][MoE] SM90 MXFP4 MegaMoE via FlashInfer MoE-EP（perf，opt-in，强化第 12 条）

`--moe-backend flashinfer_moe_ep_cutedsl` 现可在 Hopper 上跑 MXFP4 MoE checkpoint（DSv4.1-Flash），打包 MXFP4 权重不经 bf16 round-trip 直接进 kernel（FlashInfer#4843，0.7.3 起）。历史结果（**2026-10-28** 采集、8×H200、DSv4.1-Flash）：decode DP8/EP8 128K/512 + DSpark3 合成接受，c32–192 output tok/s **+8~18%**、TPOT p50 −6~13%（c8 无定论）；prefill DP4/EP4 满 1M 上下文 +3.5% 饱和吞吐；对照 #64061 早期版 `deep_gemm_mega_moe` decode +12~21% 但 prefill −8.7%；GSM8K 三后端无定论。rebase 到 main `2440ffc3870c` 后 CPU/API 48 passed、H200 Hopper 矩阵 16 passed（13:52），但**服务/GSM8K 数字沿用 10-28 旧版本、未在本 base 重跑**，按作者报告的历史口径记录；默认后端不变。

### 2.8 `#67539` [Feature][DSv4.1] Engram pool 跨引擎 NVLink/RDMA 分片 —— 见第 1.3 节

### 2.9 `#67687` [Bugfix][Kernel] SM89 int8 CUTLASS scaled_mm 在 batch invariance 下失效（fix，强化第 1 条）

SM89(L4/L40S/RTX40xx) 上 `VLLM_BATCH_INVARIANT=1` 破坏所有 int8 W8A8 模型：`cutlass_scaled_mm_sm89()` 在检查操作数 dtype 之前进入 batch-invariant 分支，而该分支只接受 fp8，报 `cutlass_scaled_mm_sm89_fp8_batch_invariant_epilogue` 运行时错误。修为只对 fp8 走固定配置分支（SM90/100/120 入口已如此）。L4：int8→bf16/fp16 带/不带 bias main 4/4 失败→4/4 通过且与常规 int8 dispatch 逐位一致；`RedHatAI/Qwen2-0.5B-Instruct-quantized.w8a8` main 无法启动→修后启动、单跑/批跑 token 与 logprob 一致。int8 用 int32 精确累加无需固定 tile（固定配置反在 9/14 尺寸慢 13–66%）。无 CI 在 SM89 跑该测试（batch-invariance 作业在 H100/B200/MI300）。

### 2.10 `#61620` / `#68120` 量化加载两则（feature/fix）

- `#61620` [Feat][Quantization] Online linear NVFP4→MXFP4：保留 ModelOpt checkpoint loader 的同时在线反量化 NVFP4 线性权重再选量化方法，支持 MXFP4 shorthand 与 target-based 转换；67 passed，XPU 上 Ripple-65B-Preview-NVFP4 TP4 冒烟生成 "4."（描述/冒烟级，无精度对照）。
- `#68120` [Bugfix][Quantization] 从 INC/AutoRound checkpoint 加载量化 Qwen3-Next MoE gate（Fixes #59248）：`Qwen3NextSparseMoeBlock` 此前不传 quant config 给 routed gate 致 `gate.qweight` 缺失无法加载；修为让 INC 从 safetensors header 判断 gate 是否真量化。Intel Arc Pro B70×4 TP4：`Intel/Qwen3-Coder-Next-int4-AutoRound` main 加载失败→GSM8K(200) 0.950；Qwen3.5-35B 保持 16-bit gate。默认 ARK backend + CUDA graph 时 ARK MoE kernel 图捕获失败（kernel 库限制，与本 PR 无关）。

### 2.11 安全加固簇（强化第 18 条）

窗口内 vLLM 出现密集输入边界/DoS 修复：`#68904` 限制 batch chat 会话数（默认 1024，`VLLM_MAX_COMPLETION_PROMPTS`，超限 HTTP 400，见方法说明）；`#68245` 解码前拒绝 ICNS 图像输入；`#56842` 限制 torchaudio resample kernel 大小；`#55182` 校验 `kv_transfer_params` role 子字典为 dict；`#68907` Responses input 校验在首个无效项后停止；`#52605` PyNvVideoCodec pool lease 计入 GPU 峰值占用；`#6fe28b1ed`/`#14da26e3e` 修复 Qwen3-VL 整数 fps 列表与负 stride 视频输入；`#23b5918d9` 加载 E8M0 scale 字节入 uint8 参数（见 2.4）。

### 2.12 其他要点（标题级 / 描述级）

- **KV/连接器**：`#898a7c68` NIXL compatibility hash 排除 indexer backend；`#82c6b325` 不断开 agents 已消失的 engine；`#59058` MoRIIO consumer abort 后保留 WRITE receive mapping 避免 KV block 泄漏（MI308X Qwen3-32B P/D，非流式 abort 后 D 停在 11.25% KV→释放）；`#12924016e` schedule 时释放已消费 encoder 输入；`#1c0b13929` Transformers backend 后续 pipeline stage 保留收到的 hidden states；`#7fb48d156`/`#66234` EAGLE block（见 2.5）。
- **ROCm/性能**：`#7bce843a0` gfx942 AITER Triton MXFP8 MoE backend；`#cc23ce153` gfx1151 W4A16 6–8 decode 行走 HIP skinny GEMM；`#34c653886` GDN decode(conv1d+delta rule+gated norm) 融进 AITER gluon(Qwen3.8-27B)；`#4d92bb632` Qwen4Exp HC combine_and_mix 走 AITER FlyDSL 两阶段 kernel；`#0aa2bb194` QK-norm+RoPE+KV-cache 融合里折入 k_out 反量化与 fp8 query 量化；`#61214` AITER FA extend kernel 预热。
- **调度/Model Runner**：`#1934f186e` MR V2 全图捕获用 `FULL` 而非 `NONE`；`#9a4930a79`/`#63830` 移除 `mamba_cache_mode`；`#feca14903` 无 EC/KV transfer 覆盖的 metadata-only item 直接失败；`#aefea5c8e` Mamba block size 从所有 pipeline stage 的 KV cache specs 解析。
- **Rust 前端簇（约 15 条，标题级，延续第 17 条）**：`#69019`/`#68781` Responses API `additional_tools`；`#68917` `/v1/decisions` 谓词读取；`#68766` Muse tool-call 参数流式；`#55399` logprobs 完成流上报 engine 错误；`#68911` tokenizer.json 优先于 tekken.json；`#68409`/`#68407`/`#68913`/`#68910`/`#68143`/`#68406`/`#48013`/`#63967`/`#68073`/`#68672` 等 API 对齐；`#61544` 跨机快照（portable attachments）。
- **新模型/recipe**：`#69039`/`#69027` Inkling 2 Mini（单 GPU 服务满 1M 上下文）；`#68864` Inkling 2 Small on H200；`#700af4612`(HY4)/`#68844` HY4 recipe；`#68948` 上游 DeepseekV32Config + DSv3.2 NVFP4 加载；`#68622` Transformers GTE rerankers + LoRA；`#8f8f6c8fc`/`#88c12de0f` Transformers backend Omni / 上游 config。

---

## 3. SGLang：2026-11-20 提交（124 条，两页 100+24）—— PD 死锁两连(#51882/#51892) + hiCache/KDA checkpoint(#50841/#50635) + EAGLE DP 死锁(#41269) + top-logprobs 批量(#48208, +25%) + Kimi-K3 multicast SP(#51656, TTFT −5%) + rust-server/diffusion 大簇

### 3.1 `#51882` / `#51892` [Fix] PD 解耦资源所有权死锁两连 —— 见第 1.3 节

### 3.2 `#50841` [Fix] hiCache write_back 丢 device-only Mamba checkpoint —— 见第 1.2 节

### 3.3 `#50635` [Fix] linker 在捕获 KDA state 读的 breakable replay 前等 external load —— 见第 1.2 节

### 3.4 `#41269` + `#51779` [Fix] EAGLE/NEXTN + DP attention 下 DSA seed 缺失死锁 + 字段重命名（fix，强化第 5/14 条）

`#41269`：DP attention + EAGLE/NEXTN(topk1) 的 DSA 模型，若草稿从 draft extend 播种稀疏注意力（`seed_dsa_topk_from_draft_extend`），decode 首轮 batch 无 seed 即死锁——`EAGLEWorkerV2.draft()` 在 `dsa_topk_indices is None` 时按 rank 单独丢草稿 CUDA graph，DP 下只有持有新请求的 rank 走 eager，其 MoE collective 跨所有 attention-DP rank，不再配对（DP0 卡在 eager `moe_fused_gate`、DP1-3 卡在 target-verify metadata、host 线程卡在 `cuModuleLoadData` 等 NCCL）。修为 DP attention 下跳过该 fallback 让所有 rank replay 草稿图（图 runner 已对缺失 seed 零填充，该轮草稿被 verify 拒绝）。GLM-5.3-Flash NVFP4 4×H20 decode：修前 PD warm-up 挂起(/health 503)，修后 warm-up 通过、20/20 短问答正确、接受长度 3.24–3.32（与无 DP attention 一致）。`#51779` 跟进：#41818 把 `enable_dp_attention` 字段重命名为 `attn_dp_enabled`，#41269 读了旧字段致缺失 seed 分支触发 `AttributeError`；178 tests + 156 subtests 通过。

### 3.5 `#48208` [Perf] speculative top-logprobs 批量化（perf，强化第 1 条）—— 见第 1.3 节

### 3.6 `#51656` [Perf] Kimi-K3 LayerNorm sequence-parallel prefill 用 multicast collective（perf，描述级）

#51520 的后续：K3.1 TP8/DCP8 两 GB300 tray 上，SP prefill 用 NCCL 仅比 replicated 快 ~0.5%，而 K3 的 multicast kernel + shared-expert side stream 历史测 ~5%。本 PR 在 `k3_sp_collective` 的 all-gather/reduce-scatter/partial_rows 上，当 `k3_ar_fusion` TP plane 就绪且 shard 为整 16B 向量时走直接 multicast，否则回落 layernorm_sp 的 NCCL；latent MoE 的 latent/shared sum 改 reduce-scatter 且 shared-expert 仍与 routed 在 side stream 重叠，新增两个持久 symmetric buffer（16K 行约 350 MB/GPU）。GB300 16K chunked prefill：cold 16K TTFT 中位 0.635→0.601 s（−5.4%）、cold 49K 1.943→1.842 s（−5.2%）；约 5% 中约 2 点来自 multicast、其余来自 shared-expert side stream。GSM8K(1319) B300 CI 0.9742、两节点 GB300 97.65%（flag off 97.35%，分次运行）；CI 状态多为 Skipped，按作者报告记录。

### 3.7 `#51463` / `#51933` AMD 两则（fix/perf，强化第 4 条）

- `#51463` [Fix][AMD] fp8-moe 标记 AITER-shuffled per-tensor FP8 权重 `is_shuffled`：AITER bump 到 `581579ab5`(#50759) 后 Grok-2 在 MI35x(gfx950) 无法启动——ROCm/aiter#6736 新增 gfx950 检查，要求 per-tensor/per-token FP8 MoE 权重的 w1/w2 都带 `is_shuffled=True`，而 shuffle 结果被重包进新 `torch.nn.Parameter` 丢了标记。修为在 `_use_aiter` 分支 shuffle 后设该标记；副作用是 aiter 不再丢弃 tuned Opus 配置，MI300 上 kernel 选择也可能变。**精度测试未跑**（计划 `/rerun-test`），CI 全 Skipped。
- `#51933` [Perf][AMD] diffusion MiniMax-H3 packed attention 用 aiter batch/MHA v4 kernel（gfx950-only）：单活跃序列的 `forward_varlen` 走 batch kernel（fmha v3，9–14% 快、免 `.contiguous()` 拷贝）、非因果 BF16 headdim128 走 MHA v4（8–9% 快）、尾波 ≤96 tiles 时 key split 8 段（Pika-H3-Flex denoise 3.32→3.125 s）。**风险**：旧 aiter（无 ROCm/aiter#6841）的 MHA v4 BF16 有 RAW race（~2400 calls 中 3 次不一致、最大偏 0.25），故按 aiter 探针仅在修复版启用；benchmark 在开发分支测得、未在本 PR base 重跑。

### 3.8 其他要点（标题/描述级）

- **新模型/Hy4 栈**：`#2195fbd3c` Hunyuan-4-Flash 文本模型；`#f389121c0` Hy4-Flash IndexPool/Engram/fused iHC post-pre kernels；`#d2d70c6dd` HY4-Flash 用 `get_parallel()` 读 TP size/rank 并登记 fusion gate（修 #51668 遗留的两个单测）；`#a3b756844`/`#51897` Hy4 cookbook。
- **量化/MoE**：`#d96ae9588` [CPU] per-token FP8 量化 AVX10.2 runtime dispatch（CI Base/AMD ✗）；`#f69412389` [AMD] 标记 AITER-shuffle per-tensor FP8 MoE 权重；`#e7c382c12` 为 deferred-finalize FP4 TRT-LLM MoE 保留 tuned autotune key；`#68c262b30` H20-3e configs for GLM-5.3-Flash TP8/TP4；`#923a8b5d1` [XPU] W8A8 int8 + Whisper MLP 量化。
- **cuda-graph/调度**：`#0f3159a68` FlashInfer CuTe DSL MoE/MegaMoE prefill CUDA graphs；`#bb8699efe` zigzag CP 允许 batch 含短于 2×cp_size 的请求；`#b5554189a` 上报 ignore_eos 请求首个 stop token 本应结束的位置；`#f7ff1e0a9` chunked ignore-eos streaming turn 在前缀后接纳；`#d8eb51846` `SGLANG_DISABLE_THP` 关透明大页；`#a0bd0c56a` `/token_ids_logprob_rows`；`#237cbfd96`/`#bb06e8163` prompt logprob candidates。
- **hiCache/mem_cache/linker**：`#448efb131` 转发 request-scoped KV hints 给存储后端；`#23b2e671c` native hash 扩展编译失败时存储 attach 直接 fail；`#35321ed07` page-sharded DSA 测试 mock 设 FP4 默认；`#0c6d6eb04` 修显式 HND 初始化。
- **rust-server 大簇（约 15 条，延续第 17 条）**：`#51897` processor 在自有 transport 发布多模态特征；`#51945` byte-level BPE 逐 chunk 解码；`#51904` 削减 request-log/crash-dump 拷贝与 recorder 锁持有；`#50846` 由特征 BLAKE3 digest 派生多模态 item 身份；`#51790` PD decode 在 /health 答心跳不做 generation probe；`#51637` 对齐 Responses summary events/参数强转/错误类型；`#48937` 按前缀缓存路由 DeepSeek 图像对话；`#51895`/`#51891`/`#51257` 等 API/流对齐。
- **diffusion/flux3 大簇（约 20 条，标题级）**：`#51935` Pika-H3-Flex 关键帧拟合/音频长度/参考 resize；`#51908` Qwen3 text encoder 图在 TP 下重放；`#51562` flux3 text encoder forward 走 CUDA graph；`#47889` cosmos3 Sim-Bimanual KV cache 管理优化；`#084c9e7677` multimodal_gen 修静默失败并删死代码(4/4)。
- **NPU/Intel XPU**：`#48420` Ascend A5 上 DSv4.1；`#28e576357` Ascend 上 Kimi-K3.1 + DSA 路径；#NPU PD KV checksum `#6ccb37eb5`；`#2510a8b7b` XPU enablements 合并；`#9cca7c9fd` Wan2.2-S2V-14B(XPU)；`#d215bb3b8` [Intel] gemma4 CPU GPTQ INT4。

---

## 4. TensorRT-LLM：2026-11-20 提交（49 条）—— v1.3.0 正式发布(见第 1.1 节) + TMEM fence(#21849) + NVFP4 scale/layout(#21220/#22340) + low-M row bucket(#22571, 1106→1238 tok/s) + RocketKV 接 KVCM V2(#21555) + BF16 MoE in-place refit(#22238, 省 60GiB/rank) + GLM-5.3 sm103(#22651, −25.5% step)

### 4.1 `#21849` [fix] MLA decode 在释放 correction stage 前 fence TMEM load（fix，强化第 22 条，latent）

Blackwell MLA decode（fp16/fp8）`get_correction_factor` 用 `tcgen05.ld` 异步读 4-word correction 记录，但下一条语句就 `consumer_release` 把 stage 交还 softmax warps，而首个真实使用在 caller 中（release 之后）——PTX 规定 `tcgen05.ld` 只在发起线程的 `tcgen05.wait::ld` 完成，`mbarrier.arrive` 与其无寄存器依赖，故不保证顺序；若触发会静默把 tile t+1 的 correction_factor 用于 tile t 的 O rescale。修为 release 前加 `cute.arch.fence_view_async_tmem_load()`（与同文件 S-stage release 及 DSpark 副本一致）。B200 实张量 100 launch/变体、每 kernel 3,276,800 次 correction-factor load，sentinel 探针修前/修后均 0 命中、参考检查通过、速度在噪声内——**latent bug（当前硬件不触发），author 报告，coderabbit 指出正确性声明未独立验证**。属第 22 条 PDL/异步依赖同步次序。

### 4.2 `#21220` / `#22340` 量化布局两则（fix，强化第 6 条）—— #21220 见第 1.2 节

`#22340` [fix] MXFP4 FP8 MoE 的 TMA descriptor 有效维度对齐：TRTLLM-Gen MoE `W4A8_MXFP4_FP8` 在 TP shard 的逻辑 intermediate/hidden 非 128 倍数时（如 `intermediate_size=1408`@TP4→逻辑宽 352）TMA 初始化报 `CUDA_ERROR_INVALID_VALUE`。修为集中 `alignMoeValidDimensions` 按 kernel config 的 `dtypeNeedsPadding` 动态对齐 valid K，移除外层 runner 的重复舍入，移除对非 128 对齐 TP shard 的 `W4A8_MXFP4_FP8` 排除并扩展 4-GPU B200 selector 到 `e60_k4_h2048_i1408`，新增三个 direct-op 回归 `(h=512,i=352)`/`(h=352,i=512)`/`(h=2880,i=2880)`。**final commit 的 GPU/build 验证、CI 结果均 pending（needs follow-up）**；历史 commit `d17c89dc23` 上 4×GB200 多 GPU selector 3/3、三个 direct case 通过，但不验证 final commit。

### 4.3 `#22571` [fix] eager low-M MMA GEMM 行计数分桶（perf/fix，削峰）

eager forward 把 low-M warp-MMA GEMM（`mxfp8_lowm_linear`+SwiGLU、`mxfp8_lowm_index_query`、`bf16_lowm_linear`、`fp8_grouped_lowm`）按精确行数编译 CuTe kernel，新行数在服务端步骤内联编译；DSv4.1 混合步骤把 prompt chunk 末行与生成行一起重放，warmup 从不跑这些行数，GB300 `v41_flash_tep4_8k1k`(con8) 首批步骤停 5–6 s。修为在 decode CUDA graph 外用行桶 kernel（≤8 精确行、之上 16/32/48/64）并把行 pad 到桶（每行只依赖自身输入故逐位不变），decode graph warmup/capture 保留精确行数，新增 warmup 阶段 `lowm_mma_row_buckets` 在图捕获前预编译各 GEMM 的桶。CI 同配置 2×GB300 交替：output tok/s **1106.1→1237.6（+11.9%）**，6 次 before 全部在测量中编译 kernel、after 全部不编译；启动 +15 s（prewarm 代价）。CSA2 grouped MXFP8 O-A GEMM 首步仍编译（~0.8 s），未修。

### 4.4 `#21555` [feat] RocketKV sparse attention 接 KVCacheManager V2（feat，强化第 3/8 条）

RocketKV（`sparse_attention_config.algorithm="rocket"`）此前无视 `use_kv_cache_manager_v2`（默认 True）始终走 V1 manager——**默认 RocketKV 被静默路由到 V1，而 model engine 按 V2 定 overlap headroom/sequence slots、V2 校验器根本没看到该 manager**。本 PR 新增 `RocketKVCacheManagerV2`：KT(key landmark) cache 作为每层 `_extra_buffers_per_layer` 放进请求 slots 并纳入 KV 预算（V1 是独立 torch pool + 自己的 BlockManager）；prompt pruning 用 `_resize_context_history` 收缩 cache + 两个新 hook 让 manager 计 retained 条目；KT Triton kernel 用 int64 加载 block index（B200 Llama-3.1-8B KT page-index scale 32、element offset 达 ~1.25e10）；Vanilla backend 也改按 KT block table 寻址。B200：71 passed（NIAH）、KV cache 套件 2799 passed；NIAH TRTLLM/Vanilla V1/V2 输出 20/20、10/10 一致。属默认值/后端归属修复。

### 4.5 `#22238` [fix] BF16 TRTLLM-Gen MoE experts 原地 refit（fix，强化第 21 条）

RL 权重更新此前每 rank 保留整层 routed experts 的第二份拷贝（Nemotron-3 Ultra BF16 DEP16 约 **+60 GiB/rank**）。修为 `_use_inplace_refit` 允许时保留 kernel-layout 张量、逐 expert 直接转进目标 slot（未送达 expert 保留权重），gated 层保留 recreate 路径；并修复共享 helper 的 permute cache key 只按 shape——同进程 gated 与非 gated 同形状层共享条目致第二层拿到第一层行序，key 加入 `is_gated_act_gemm`。48×GB200 NeMo-RL GRPO：一次权重同步 ~7.5 s、`gen_kl_error` 稳定 ~5e-4。属在线权重热更新（第 21 条）。

### 4.6 `#22651` [feat] modeling_v2 GLM-5.3-NVFP4 路由到 sm103 TP8 低延迟 target（feat，描述/作者报告）

GLM-5.3-NVFP4 声明 `MIXED_PRECISION` 而 `_nvfp4_with_bf16_mtp_layer` 只接受 `NVFP4`，TP8+MTP5+FP8 KV 部署在 `modeling_v2=auto` 下回落内置模型、`require` 下抛 `NoTarget`。修为按逐层 `quantized_layers` map 判定（每层 MoE 有 NVFP4 routed experts、除 dense MLP 外无其他量化模块、MTP 层不量化）。GB300 LHR 2 节点 TP8 前 400 题 GSM8K：`off` 0.970(388/400)、step p50 16.21 ms、409 tok/s/user；`auto` 0.975(390/400)、12.08 ms、501 tok/s/user（**step p50 −25.5%、tok/s/user +22.5%**，精度噪声内持平）。**两臂为分开作业、部分在不同节点、非配对**；`require` 在该镜像不可达（draft_front 路径拒绝 FlashInfer 0.7.3），按作者报告记录。

### 4.7 其他要点（标题/描述级）

- **KVCacheManagerV2/块复用**：`#21829` per-conversation drop-extended-turns（opt-in、需 `policy=per_conversation`，BLAKE2b prefix digest，263K token 约 12 ms/turn，作者报告 GB300 13/13 + 251/252；新集成测试未入 test-db/qa list，needs follow-up）；`#15419029d` 同主题提交。
- **Inkling 2 / 新模型**：`#22627` Inkling 2 Modeling V2 family（operators/kernels/Inkling-2-Small target）；`#22261` MiniMax-M3.1-Flash on Rubin(SM107) 回灌 Modeling V2（mass merge）；`#21019` GLM-5.3-Flash-NVFP4 on sm_10x(B200/GB300) tp4+MTP；`#752e6b375` GLM-5.3-NVFP4 sm103 TP8 低延迟 target。
- **perf/host 开销**：`#22355` DSv4.1 更省的 executor/scheduler/sampler host work per step；`#369b5b99b` DSv4.1 CSA2 attention-TP decode 预取 Reindex 候选块进 L2；`#22497` Gemma4 decode 融合（norm/RoPE、FP8 输入量化、PLE tail、assistant head）；`#22058` 一模型投机 worker 的 BF16 LM head + argmax draft head 融合；`#22610` GVR Top-K CuTe DSL kernel 跨进程持久化。
- **其他 fix/infra**：`#9d3921216` fence TMEM load（见 4.1）；`#22151` 发送请求前注册 RPC response futures；`#a4e0134a9` qwen4_exp 收集节点 PLE host table 时释放；`#21220`/`#22340` 见 4.2；多条 waive/unwaive（`#22684`/`#22667`/`#22647` 等）、test 修复（`#22597`/`#22631`/`#22637`）、infra（NVSkills CI `#22671`、OpenSearchDB retry `#16418`、lock file、CI node placement）。

---

## 5. llama.cpp：2026-11-20 提交（43 条）—— Inkling 2 Mini(#32694) + MoE cache graph nodes(#32636) + MUL_MAT_ID 精度(#31345) + ffn_gate_up_exps scales(#32657) + gemma4 多序列 PP ~3×(#32342) + Orin MMVQ(#31268) + 多后端/UI

### 5.1 `#32657` / `#31345` 量化 scale 与精度两则 —— 见第 1.2 节

### 5.2 `#32636` [CUDA/GGML] 为 MoE cache 预留 graph nodes（fix）

`graph_max_nodes()` 未计 MoE cache 给每层加的节点（slot lookup + `supports_op` probe 张量，`--moe-cache-cpu-share` 再加一条 expert path）。32x/64x 预算的架构有余量，但 cohere2moe 用默认 `8*n_tensors` 且每层张量很少（443 张量、49 层→3544）；叠加量化 KV（K/V rotation 也加节点）后 graph context 耗尽，`GGML_ASSERT(obj_new) failed`。GB10 实测：f16 KV ~2757、q8_0 KV ~3345、f16 KV+MoE cache ~3189、q8_0 KV+MoE cache ~3777(>3544 崩)、再加 cpu-share 0.5 ~4113。修为 MoE cache 开启时加 `32*n_layer_all` 节点（覆盖最坏 ~16 张量/层、2× 余量）。

### 5.3 `#32342` gemma4 多序列 query 的 prompt processing 优化（perf，硬件特定）

#31356 的后续：Gemma 4 E4B/E2B 处理多序列时跳过不需要算 logits 的 prompt 行。MacBook Air M3（gemma-4-E2B-it-Q4_0，`-c 20000 -b 2048 -ub 512 -fa on`）：PP512/B1 的 S_PP **673.93→1700.60 t/s**、PP2048/B8 从 256.99→1928.93 t/s（prompt processing 约 3×），TG 基本不变；作者自测、单一硬件/模型，按硬件特定口径记录。

### 5.4 `#31268` [CUDA] Orin single-column MMVQ 优化（perf，描述级）

针对 Orin 单列 MMVQ 的 CUDA 优化（正文较长，性能表未在本次抓取窗口完整呈现，按标题/描述级记录，不做通用加速结论）。

### 5.5 其他要点（标题/描述级）

- **新模型**：`#32694` Inkling 2 Mini；`#32559` Domyn Edge 4B。
- **多后端**：`#32389` HIP RDNA1 MMVQ 参数表（ncols=1 单 warp/block）；`#31268` CUDA Orin MMVQ（见 5.4）；`#7e960f40f` OpenCL f16 FA 折 GQA head；`#16f80bd0c` Vulkan chunked GATED_DELTA_NET；`#6f5c9ad21` Vulkan COL2IM_1D misalign/split workgroups；`#d7d1b632d` WebGPU copy kernel 2D 索引修复；`#2d8d00393` WebGPU COL2IM_1D；`#4b69c79cc` CUDA 更新 GDN `supports_op`；`#bed8ec2f8` llamafile(s390x) QK 数据类型；`#52e8db655` kleidiai 允许 OHOS(HarmonyOS) AArch64。
- **图/杂项**：`#a83d68b56` Metal 允许 norm chain 出现在融合 MUL/ADD 任一操作数；`#4783b1145` 把 indexer key rotation 选择移入 model class；`#652d559ed` 把 graph node budget 移入 model class；`#7716ec8cb` ggml 主版本 bump 提醒 + 弃用 `ggml_ftype`；`#36cd3aff4` server 新增 `--ui-mcp-proxy-allow-private-urls`。
- **UI 大簇（约 15 条，标题级）**：`#79e81b0bd` 语音消息与实时波形/转录听写；`#059cac284` 搜索消息内容/Discover 模型；`#55b8d8d89` 全屏图表预览；`#b81aa847c`/`#9d6f04fa9` design tokens/图标；`#c6a25fc62` highlight.js 按需加载；`#32615`/`#32617` 等。

---

## 6. 累计结论延续与新增（继承 1–23，本期无新增）

**累计结论（1–23 主线回顾，本期证据）**：① 调度器隐式阻塞的定量画像（本期 SGLang #48208 GB300 +25%/H100 无收益、vLLM #68351 多节点加载 −68% 再证 host 侧瓶颈）；② 发版窗口的双向性（本期 TRT v1.3.0 迟到正式版与 main diverged 1604/210，再证"发版≠最新代码"）；③ 默认值成为主要风险载体（本期 TRT #21555 RocketKV 默认静默走 V1、v1.3.0 一次性固化 14 项 BREAKING）；④ 异构/量化后端静默错误高频（本期 vLLM #66586 gfx942 FP8 KV scale 缺 ×2、TRT #21220 do_shuffle 误用、SGLang #51463 AITER 标记丢失、llama.cpp #31345 回退精度）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（本期 vLLM #66234 EAGLE block 所有权重构、SGLang #41269 EAGLE DP seed 死锁、#51882/#51892 PD 资源所有权）；⑥ 量化路径静默错误（本期 vLLM #65256 E8M0 scale 按值错转、llama.cpp #32657 gate_up scale 漏载）；⑦ 专家张量/路由归属与分页漏接（本期 llama.cpp #32657、TRT #22238 gated/non-gated 共享 permute cache）；⑧ KV 卸载/HiCache/PD 健壮性（本期 SGLang #50841 丢 Mamba checkpoint、vLLM #67539 Engram pool、#59058 MoRIIO KV 泄漏）；⑨ 长上下文整型溢出（本期无新增强证据，延续）；⑩ MXFP8 KV 正确性簇（本期 vLLM #65256 MX scale 字节加载）；⑪ 新旗舰模型 CP/KV/性能一等公民（本期 TRT #22651 GLM-5.3 sm103、#22627 Inkling 2、SGLang #51656 K3.1 SP）；⑫ DeepGEMM MegaMoE 量化后端主路径（本期 vLLM #64289 SM90 MXFP4 MegaMoE +8~18%）；⑬ FP8/低比特 KV 标准化（本期 TRT v1.3.0 NVFP4 cold-page compression/disk cache）；⑭ 投机解码验证路径硬化（本期 vLLM #66234/#68782、SGLang #41269/#51779/#48208）；⑮ 跨后端可移植性（本期 llama.cpp 多后端簇、SGLang XPU/NPU、vLLM ROCm AITER 簇）；⑯ 新旗舰模型默认性能化（本期 TRT #22651 `modeling_v2=auto` 路由、#22571 prewarm）；⑰ 前端 Rust 化（本期 vLLM ~15 条、SGLang rust-server ~15 条）；⑱ 服务安全加固（本期 vLLM 安全簇 #68904/#68245/#56842 等）；⑲ 非量化常规路径静默错误密度不亚于量化（本期 vLLM #68749 wave 竞态、#68782 同步崩溃、SGLang #50635 KDA state 时序、TRT #21849 TMEM fence）；⑳ 正式发版固化默认值风险 × 破坏性变更（本期 TRT v1.3.0 集中 14 项 BREAKING + 8 项 Known Issues）；㉑ 在线权重热更新时序竞态（本期 TRT #22238 BF16 MoE 原地 refit 省 60 GiB/rank）；㉒ PDL/异步依赖同步次序成静默/崩溃新子项（本期 vLLM #68749 ROCm wave、TRT #21849 TMEM load fence，跨 CUDA/ROCm 两后端）；㉓ 语义合并冲突成静默正确性新子项（本期无新增强证据，延续）。

**本期结论**：无新增累计条目。头号事件 TRT v1.3.0 是第 20 条的新样本且补充了"迟到稳定线正式版 ≠ 滚动 RC 延续"的版本判读要点；静默正确性铁证（第 1.2 节 7 条）以既有第 4/5/6/8/14/19/21/22 条归类即可覆盖，未出现需要单列的新故障模式。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 版本升级决策（新增关注）**：TRT-LLM **v1.3.0**（2026-11-20 正式版，独立稳定线）含 14 项 BREAKING（TensorRT backend/C++ Executor/TRTLLMSampler/two-model spec 移除、CLI 覆盖 YAML、KV transfer 超时默认 60 s、logprobs 封顶 100 等）与 8 项 Known Issues（尤其**解聚+DSpark draft KV 不转移致接受率降低**、GLM-5.2 NVFP4 GB200 `max_num_tokens>4096` segfault、Qwen3-32B-FP8 B300 解聚错误输出）；升级前逐项核对，勿将 v1.3.0 与 v1.4.0rcX 视为同一序列；按 PR 号做 compare 祖先核验而非依赖发布说明。
- **P0 — 静默正确性复核**：vLLM `#66586`（gfx942 compressed-tensors FP8 KV scale 缺 ×2，PPL 偏差最高 20.5%）、`#68749`（ROCm batched kpool wave 竞态）、`#68782`（XQA ragged draft mask 同步崩溃，DGX Spark MTP）、`#65256`（MX E8M0 scale 按值错转）；SGLang `#50635`（linker+KDA state 时序）、`#50841`（hiCache write_back 丢 Mamba checkpoint）、`#41269`/`#51779`（EAGLE DP seed 死锁/字段重命名）；TRT `#21220`（NVFP4 activation shuffle）、`#22340`（MXFP4 MoE TMA 对齐，final commit 待验证）、`#21849`（TMEM fence，latent）；llama.cpp `#31345`（MUL_MAT_ID 回退精度）、`#32657`（gate_up expert scale 漏载）。
- **P1 — 性能/默认值核查（口径特定）**：vLLM `#68351`（多节点预取 −68%，Lustre/GB300 作者口径）、`#64289`（SM90 MXFP4 MegaMoE +8~18% 沿用 10-28 旧版本数字）、`#66234`（EAGLE block 丢弃移除，多命中一块但部分精度小幅波动）、`#67539`（Engram pool NVLink/RDMA，模型步骤开销在噪声内）；SGLang `#48208`（top-logprobs 批量 GB300 +25%、H100 无收益）、`#51656`（K3.1 multicast SP TTFT −5%，作者报告）、`#51933`（AMD MHA v4 需 ROCm/aiter#6841，否则 RAW race）；TRT `#22571`（low-M row bucket +11.9% 但启动 +15 s）、`#22651`（GLM-5.3 sm103 −25.5% step，非配对 A/B）；llama.cpp `#32342`（gemma4 多序列 PP ~3×，M3 单硬件）。
- **P1 — CI/验证状态跟踪**：SGLang `#51463`/`#51459`/`#51892`/`#50841`（Base/AMD 红或 Skipped、精度未跑）、TRT `#21220`/`#22340`/`#21829`（needs follow-up、未入 CI list）、vLLM `#64289`（服务数字未在本 base 重跑）、llama.cpp `#31268`（性能表未完整呈现）——均按描述级/作者报告记录，不作为已全绿验证事实。

---

## 7. 社区与视频资源（HN Algolia，2026-11-20 epoch 窗口 1795132800–1795219199）

> 以下均为社区/Show HN，非四家引擎官方，按「二手线索/未核验」记录；无四家引擎官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词检索中 `sglang` / `TensorRT-LLM` / `llama.cpp` / `speculative decoding` / `paged attention` **均 0 命中**；`vllm` 18 条均为泛 AI 资讯与「LLM」模糊匹配（如 11 分「The Hungry Butler Effect: Concealed Commercial Incentives in LLMs」、29 分「Llms.txt and MCP didn't move the needle」等），与推理引擎实现无直接关系。

- **「Auditing 23.5k Google AI Ultra ($200/mo) SQLite logs: 0% KV cache and tiered IDs」（1 分）**：2026-11-20T17:05，社区对 Google AI Ultra 客户端 SQLite 日志的审计帖，提及 KV cache 命中为 0，非四家引擎官方一手、内容未核验。**二手线索/未核验**。
- **「AI Inference for Beginners: 6 hours of video and 85 pages of notes [pdf]」（1 分）**：2026-11-20T14:39，社区 AI 推理入门视频/笔记，非引擎官方发布、未核验。**二手线索/未核验**。
- 其余命中（如「Show HN: GhostFleet」2 分「inference engine」模糊匹配、`inference engine` 关键词 1 条无关游戏帖）与推理引擎实现无直接关系，未列入。本期**无引擎相关的可确认官方视频**；Reddit / YouTube / X 未采集。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **122**（分两页 100+22，page2 返回 22 条，合并去重取全）/ SGLang **124**（分两页 100+24）/ TRT-LLM **49** / llama.cpp **43** 合计 **338** 条（2026-11-20 UTC 窗口；周五工作日正常量）。与上一期（2026-11-20 档案覆盖 11-19）连续，无空档。
- **版本核验**：releases API 确认——**窗口内有新正式版**：TensorRT-LLM **v1.3.0**（2026-11-20T02:45:00Z，`prerelease=false`，18,233 字符发布说明全文核验；tag commit `c66df86d8aab` 日期 11-16，`compare/v1.3.0...main` 返回 `diverged, ahead_by=1604, behind_by=210`，属独立稳定线；**无新 prerelease**）。无新正式版：vLLM v0.34.0（11-16）、SGLang v0.5.23（11-02）、llama.cpp v1.0.0（11-13），TRT-LLM 滚动 RC 最新仍为 v1.4.0rc4（11-18，落窗口前、上一期已记录），llama.cpp 推进 nightly b12588–b12600（prerelease）。按本追踪方法论，发布说明为精选 highlights，**不作为「修复已进版」依据**。
- **PR 正文来源**：约 **40 个关键 PR 正文经官方 issues API 抓取**（markdown 官方一手），配额充足（core 60 起、抓取后剩余 4）、无 IncompleteRead 重试失败、无降级为标题级；所有 PR 号均在窗口 commits API 中确认为已合入 main 的提交。其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度。
- **描述级/未核验标注**：TRT-LLM `v1.3.0` 发布说明为官方一手但属精选 highlights，Known Issues/能力声称按发布说明级记录、`#21849` 为 latent bug（B200 sentinel 探针 0 命中、coderabbit 指正确性未独立验证）、`#21220` QA 判 needs follow-up（新单测未入 CI、所引 CI UNSTABLE/FAILURE）、`#22340` final commit GPU/build 与 CI 验证 pending、`#21829` 新集成测试未入 test-db/qa list（needs follow-up）、`#22651` A/B 非配对且部分在不同节点且 `require` 在该镜像不可达；vLLM `#64289` 服务/GSM8K 数字沿用 2026-10-28 旧版本、未在本 base 重跑、`#61620` 仅 XPU 冒烟无精度对照、`#67539` 模型步骤开销在噪声内（作者报告）；SGLang `#51656` CI 多 Skipped（作者报告）、`#51933` benchmark 在开发分支测得未在本 PR base 重跑且旧 aiter 的 MHA v4 有 RAW race、`#51463` 精度测试未跑、多个 `#50635`/`#50841`/`#51892` CI Base/Extra/AMD 红；llama.cpp `#32342` 为 M3 单硬件作者自测、`#31268` 性能表未在抓取窗口完整呈现。均已按实际缺失程度标注。
- **社区信号**：HN Algolia 引擎官方更新类故事无；「Auditing 23.5k Google AI Ultra」（1 分）与「AI Inference for Beginners」视频笔记（1 分）按二手线索/未核验记录；无引擎相关的可确认官方视频。
- **技能检索（find-skills / agent-browser）**：find-skills 经 SkillHub 语义检索（EN+ZH 各 1 组：LLM inference GitHub commit monitoring / 大模型推理热点归档 git 同步 / github trending daily digest）返回 `github-trending-cn` / `github-ai-trends` / `github-trend-radar` / `full-auto-info-collector` / `free-trending-hub` 等趋势榜/信息聚合类技能，与「官方一手提交 API + 证据分级归档 + 指定目录归档 + git 提交推送」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/issues API）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次一致。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/作者报告/计算值未实测/CI 未全绿/needs follow-up/微基准/硬件特定/prerelease/latent/非配对 A/B」。
