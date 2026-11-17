# 大模型推理优化每日热点档案

> 档案日期：2026-11-17（本自动化于该日执行）
> 关注窗口：2026-11-16（周一）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（vLLM/SGLang/TRT-LLM 经 issues API 抓取 markdown 官方一手，llama.cpp 6 项因 API 配额用尽改由 GitHub PR 页面 HTML 提取）+ HN Algolia（HTTPS，numericFilters 按 2026-11-16 epoch 窗口 1794787200–1794873599）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 113 / SGLang 74 / TRT-LLM 41 / llama.cpp 34 合计 **262** 条（2026-11-16 为周一，较上一窗口周日 116 大幅回升，属工作日正常量；vLLM 分两页取全 100+13，其余单页取全）；**窗口内 vLLM v0.34.0 正式发布（2026-11-16T23:22:00Z，1061 commits / 412 贡献者，非 prerelease）**，为本追踪以来继 v0.32.0/v0.33.0 之后第三个正式版本且首次落在采集窗口末尾；其余三家无新正式版本（llama.cpp 推进 nightly b12441–b12445，prerelease）；**50 个关键 PR 正文（44 个经 issues API + 6 个经 HTML 页面）抓取**，issues API 未认证 60/h 配额于 llama.cpp 段用尽；HN Algolia 引擎相关高信号弱（Intel D270 vs RTX PRO 6000 基准 2 分、Goethe 本地推理引擎 4 分，均二手线索/未核验）。本窗口 vLLM 单日 113 为追踪以来单日新高（超过此前 2026-10-27 的 102）。

---

## 1. 版本状态与本期头号事件：vLLM v0.34.0 正式发布（Extensible KV 默认开启 + Model Runner V1 移除 + 默认 block-size 64）+ 静默正确性（TRT-LLM #22287 PDL 读陈旧 scale GSM8K 94→0.5 + SGLang #50019 漏用 swiglu_limit cos 0.64 + vLLM #62474 MoRIIO 异构 TP + llama.cpp #32433 视觉嵌入颠倒）+ DSpine 投机解码新成员（#67094）

**提交计数（2026-11-16 UTC，均 HTTP 200；2026-11-16 为周一，量较上一窗口周日显著回升）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 113 | 分两页（100+13）取全；Rust frontend / perf / ROCm / bugfix 密集 |
| SGLang | 74 | diffusion / AMD / NPU / 模型支持活跃，含 #50156 合入后被 revert |
| TensorRT-LLM | 41 | 大量 infra/waive/test，feature/fix 约 12 条 |
| llama.cpp | 34 | 多后端（Vulkan/OpenCL/CUDA/HIP/WebGPU）+ 模型/转换 |
| **合计** | **262** | 较上一窗口（周日的 116）大幅回升，工作日正常 |

### 1.1 vLLM v0.34.0 正式发布（窗口内，2026-11-16T23:22:00Z；1061 commits / 412 贡献者 / 120 新贡献者，非 prerelease）——本期头号事件

官方一手（[GitHub Release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)，47,575 字符发布说明全文核验）。要点（**区分发布说明原文事实与研判**）：

- **Extensible KV cache 默认开启**（#56492）：KV cache 按 CUDA graph capture / warmup **之后**实际空闲内存定容，而非估算值。**发布说明原文**：MiniMax-M3 on H200 从 433k → 672k tokens；**明确 Trade-off**：部分 H200 模型在并发 128 下 TPOT **+6–18%**，可用 `--gpu-memory-utilization 0.9` 或 `--no-enable-extensible-kv-cache` 退出。配套 8 项 allocator 碎片/启动 OOM/KV connector 加固（#67296/#67485/#66731/#66523/#66197/#65137/#66605/#66674）。
- **Model Runner V1 移除**（#63373）：`VLLM_USE_V2_MODEL_RUNNER=0` 现在启动即失败；MRV2 成为唯一 GPU runner，dual-batch overlap 在 MRV2 上支持投机解码与 DCP（#61464）。
- **默认 `--block-size` 64**（#63231，此前已在本追踪多期记录其合入，现随 v0.34.0 正式落地为默认）：ROCm paged attention 仍用 16。
- **默认值/破坏性变更集中（升级检查表，官方原文）**：`gpu_memory_utilization` 默认 1.0；ROCm 7.2 镜像取消（仅 ROCm 10.0）；要求 `huggingface_hub >= 2.0.0` / `tokenizers >= 0.23.3` / `timm >= 1.0.30`；QuTLASS / `fp_quant` 移除（#65230）；Marlin W4A8 移除、改用 `--quantization humming`（#65542）；Keye-VL / Keye-VL-1.5 移除（#64287）。
- **性能（发布说明报告值）**：Step5 NVFP4 MoE on TRTLLM-Gen 4×GB200 解码最高 **1.95×**（#65152）；GLM-5.3-Flash DeepGEMM FP8 MegaMoE 吞吐 **1.6×**（#60895）；token-sharded sparse-MLA prefill under TP 使 GLM-5.3 在 32K–128K TTFT 快 **1.24–1.40×**（#63956，默认）；Kimi-K3 KV-balanced MLA decode 1.34–1.59×（#63638，opt-in）；Qwen3.5 breakable CUDA graph 降 ITL **3.0–8.8%**（#51012）；FlyDSL decode monokernel on 8×MI355X interactivity +35–57%（#63440）。
- **投机解码**（发布说明原文）：Suffix decoding 上 MRV2 不再依赖 Arctic Inference（#60339）；声明式 config + `register_speculative_method` 插件 API（#63205）；MRV2 draft-model CUDA graph（#60989）；自适应验证接 XPress（#65175）；**draft KV cache dtype 与 target 对齐**（#65198，本期 PR 亦在窗口内，FP8 target 得 FP8 draft cache，每个 draft 模型省约 50% GPU 内存）。
- **10 个新模型**：Mistral Large 4（1T/52B，FP8+NVFP4）、MiniMax-M3.1-Flash、Nemotron-3.5-Transcribe、Ripple-65B、ZAYA1、LFM3 家族等。

**研判（观点）**：v0.34.0 把本追踪长期跟踪的三条主线一次性固化为正式默认——Extensible KV（第 8 条 KV 管理重构）、Model Runner V1 移除（第 3 条默认/破坏性变更）、block-size 64（第 3 条）。**最值得警惕的新默认值为 `gpu_memory_utilization=1.0`**：它把 KV 容量与系统/框架内存的缓冲压到极限，官方已明示 TPOT 代价，部署需在吞吐与尾延迟间重新标定。发布说明为精选 highlights，**未逐条列出本窗口 PR**（#67315/#65575/#62474/#67316 在发布说明中 0 次出现）——按本追踪一贯方法论，**发布说明不作为「某修复已进版」的依据**。

### 1.2 静默正确性铁证（延续累计结论第 19 条：非量化常规路径同样高频）

本窗口 4 条一手静默/崩溃级案例中，仅 2 条源于量化 scale，另 2 条是索引/坐标/竞态错误，继续印证第 19 条：

- **TRT-LLM #22287**（官方一手，fix）：SM103 block-scaled CUTLASS GEMM 的 scale-factor 加载 warp 未调用 `wait_on_dependent_grids()`，programmatic dependent launch（PDL）下读到 producer 尚未写入的激活 scale。由 #21273 的 SwiGLU+NVFP4 融合暴露（此前非 PDL SwiGLU 恰起到串行作用）。**DeepSeek-R1 FP4 + cache_mgr_v2 + NIXL，B300 GSM8K 从 94.05 掉到 0.53，请求正常完成但输出 garbage**；SM100 不受影响。强化第 4/6 条。
- **SGLang #50019**（官方一手，fix）：`--moe-runner-backend flashinfer_cutedsl` 构建 `CuteDslMoEWrapper` 时漏传 `swiglu_limit`（DeepSeek-V4 / GLM-5.3-Flash 用 10），NVFP4 routed experts 跑了未裁剪的 SwiGLU 且**无报错、输出不同**；新测试 cosine **0.6355 → 0.9990**。强化第 6 条。
- **vLLM #62474**（官方一手，fix）：异构 TP 的 MoRIIO WRITE 用 model-wide KV-head 数（各层最大值）给每层切片，Gemma4 滑动层 4 head / full 层 8 head 时写不存在的 head（prefill TP1→decode TP2 报 "Head slice outside 4/2 heads"；EAGLE3 drafter IndexError）。44/48 case 原本失败，修复后全过。属坐标/归属错误，强化第 19 条。
- **llama.cpp #32433**（官方一手，fix）：Muse Glimmer 视觉编码器位置嵌入网格按 (h,w) resize，但模型训练用 h/w 互换（应 xy 而非 ij），port 时遗漏；方形图不变，Muse Spark Open 同图受益；HF/vLLM/SGLang 实现本就正确。属坐标语义错误，强化第 19 条。

### 1.3 其他头号线索

- **投机解码新成员 DSpine**：vLLM #67094 基于 arxiv 2609.36173 与腾讯 DSpine 参考实现，为 Qwen3.8-27B 增加 DSpine（Draft in Parallel, Condition Through Depth）支持，单 H100 上 15 draft token、GSM8K T0 接受长度 6.9271 / pooled 5.9024。强化第 14 条。
- **SGLang #50156 合入后回滚**：5084 行插入的「Blackwell/Rubin hybrid GDN+QSA+MoE 服务优化栈」（全 opt-in）于 01:35Z 合入，04:17Z 被 #50703 以 "This PR is extremely invasive" 为由 Revert。研判（观点）：大规模、多开关、跨 MTP verify/CUDA graph staging/HiCache/Rust TreeCore 的侵入式优化即便全 opt-in，也因可维护性与回归面被拒，印证「发版窗口双向性」与复杂度风险。
- **安全**：vLLM #67316（embedding 压缩包在 `torch.load` 前未查未压缩大小，4.2 MB payload 触发 3072 MiB 分配，压缩全零存储可绕过 2 GiB 限制）、SGLang #49761（HiCache 存储凭据 S3 key 明文出现在 /server_info、日志、crash dump、gRPC，本期新增脱敏）。延续第 18 条。

---

## 2. vLLM：2026-11-16 提交（113 条）—— v0.34.0 发布(#release) + DSpine(#67094) + monokernel/latency_specialized 合一(#67315) + KV 健壮性(#65575/#66583/#62474/#67486) + 量化融合(#65578/#63272/#64406/#67267) + 安全(#67316) + Rust frontend 簇

### 2.1 `v0.34.0` 正式发布（窗口内，官方发布说明）——见第 1.1 节专题

1061 commits / 412 贡献者；Extensible KV 默认开、V1 runner 移除、block-size 64、gpu_memory_utilization 默认 1.0、Marlin W4A8 / QuTLASS / Keye-VL 移除。发布说明为官方一手。

### 2.2 `#67094` [Spec Decode] Add DSpine support（feature，强化第 14 条）

腾讯 DSpine（arxiv 2609.36173）投机解码接入，基于 DFlash/DFlash2 路径做 checkpoint 适配 + TP-aware transition scoring，支持 Tencent / Speculators 两种 checkpoint 布局，复用现有采样与验证。单 H100 80GB BF16 TP1 FA3 c8、15 draft tokens、思考开启 xhigh：GSM8K T0/T1 平均接受长度 6.9271/6.4762、pooled 5.9024/5.4559；MATH-500 6.3780/5.8710。新增 Qwen3.8-27B GSM8K 回归入 CI。**官方一手（含完整数据表）。**

### 2.3 `#67315` [Config] --monokernel 与 decode_backend=latency_specialized 合一（config/feature，强化第 3 条）

两个开关分别被不同模型读取，其他模型静默忽略、`additional_config` 不校验，导致「传错开关 → 静默走标准 decode、无报错无日志」。本 PR 在 `VllmConfig.__post_init__` 令二者互为别名、`--monokernel` 配其他 decode_backend 值报错、两者共享编译缓存。属「默认值/开关静默失效」治理，v0.34.0 发布说明未列出该 PR（0 次）。

### 2.4 `#62474` [Bugfix][ROCm][MoRIIO] 按每层 head 数切片 WRITE（fix，强化第 19 条）——见第 1.2 节

### 2.5 `#67316` [Bugfix][Security] torch.load 前查压缩包未压缩大小（fix，强化第 18 条）——见第 1.3 节

### 2.6 `#65575` [KV Connector][Mooncake] prefill worker 失联即失败（fix，强化第 8 条）

prefill worker 死亡时 decoder 的 pull 会持 KV block 等 540 s（ZMQ 静默重连），且缓存死地址致后续请求再等 540 s，KV 被永不能完成的请求占满。修复后 P 进程死亡时等待请求 **545 s/1084 s → 2–4 s**，D 的 KV 占用降到 50% 以下从 556 s → 5.5 s（2×A800、DeepSeek-MoE-16B、128 请求 2.4× KV 容量压测，unit 750 passed）。decoder 侧对应 #64193。

### 2.7 `#66583` [KV Offload] async_init 后台建 CPU tier（feature/perf，强化第 8 条）

`async_init: true` 下后台线程预 fault + pin CPU 区域，引擎先服务、就绪后再开卸载；setup 线程切 relaxed capture mode 避免使 #45850 被 revert 的 CUDA graph capture 竞态。启动到 /health：GLM-5.3 16×H100 439→334 s、Qwen3-8B TP4 1250 GiB 241.5→57–61 s；pinning 加 pacing 后最坏延迟 0.4–1.4 s → 0.18 s。默认关、仅 CUDA。

### 2.8 量化/性能簇（官方一手，含微基准）

- `#65578` [Perf] GELU + NVFP4 quant 融合（BERT 类 FFN）：BGE reranker v2-m3 on RTX PRO 6000 cutlass 后端 GPU 时间 **−18–21%**、分数逐位相同；默认 FlashInfer 后端仅约 −1%。
- `#63272` [ROCm][Perf] block Hadamard 输入旋转融进 AITER MXFP4 激活量化：8×MI355X Qwen3.8-2.4T TP8 恢复 #62750 旋转损失的 6.0–12.4% 吞吐，与纯 RTN 相差 0.6% 内；每层旋转开销 70–170% → 0–5%。
- `#64406` [Perf][MoE] OCP MX emulation 只反量化被路由的专家：MI250X 256 专家 MXFP4 MoE 单层 6.6 ms → 1.5 ms（8 routed）/2.3 ms（31 routed）；注意不同 tile config 下输出可能因舍入不同（MI250X 无 tuned config 故 bit-identical）。
- `#67267` [Quant] LUT-B per-channel block32 weight scales：对齐 Rubin 指令粒度 `[N,K/32]`，重建误差 1.09–1.73× 更低，运行时镜像/kernel 不变（无 speed/memory 变化）。
- `#59564` [Perf] FP8 group quant 用 value-only reduction（`amax()` 替 `torch.max()[0]`）：H200 编译路径从比 CUDA 慢约 10% → 最高快 20%（如 8192×7168：82.9→59.1 µs）。
- `#65931` H100 BF16 batch-invariant matmul configs（gpt-oss）；`#67393` Qwen3.5 动态 per-token FP8 输入量化融进 add-RMSNorm；`#67725` Qwen4Exp MTP drafter 两个 pre-FC Gemma RMSNorm 合一 Triton launch（B300 微基准 12–77×，ITL −0.2–0.45%）。

### 2.9 其他要点（标题级 / 描述级 / 性能级）

- **性能/新模型**：`#66324` MiniMax-M3.1-Flash GB300 专用 decode（CuTe DSL，TP4，opt-in，decode step c1 −32–33%、AgentX p90 interactivity 1.45×，精度一标准误差内）；`#65198` draft KV dtype 对齐 target（进入 v0.34.0 投机解码亮点）；`#66311` LFM3 EAGLE-3/DSpark aux hidden states（issue 型，本地 patch 验证）；`#62550` Step5 Blackwell opt-in PrimTS attention；`#66497` DSv4.1 on A100/H100；`#65947`/`#66866` 序列并行×流水并行。
- **默认/配置**：`#54268` 仅 Blackwell/Rubin 默认启用 all2all fi_one_sided；`#66653` 全执行模式禁用 expanded DeepEP v2 dispatch（1548 passed，duplicate of #59337）。
- **KV/内存**：`#67486` 共享 Full Attention 前缀不再误触发 pending offload store flush（537 passed）；`#66583`（见 2.7）；`#62058` 解释满 /dev/shm 的 EFAULT；`#67258` mmap_path to SharedOffloadRegion；`#65197` 重命名 filtered group index。
- **Bugfix**：`#67806` VLLM_BATCH_INVARIANT 下只选 batch-invariant MXFP8 线性 kernel（main 32/32 batch invariance 失败 → 修复，强化 batch invariance 主线）；`#68005` 懒加载 SM80/SM90 metadata builder；`#59208` 跨节点 symmetric-memory rendezvous 加保护；`#67882` pynccl 禁用时回退 torch.distributed；`#64965` 拒绝 decoder attention 用 TORCH_SDPA；`#67316`（见 2.5）；`#65672` Mooncake early prefill abort 幂等。
- **Rust frontend 簇（约 15 条，标题级）**：`#67708`（--tokens-only）、`#67893`（/inference/v1/abort_requests）、`#67399`（X-Vllm-Priority/拒不可解析头）、`#67400`、`#63773`（保留 SentencePiece 前导空格）、`#67872`（--kv-sharing-fast-prefill）、`#67871`（--hf-token）、`#67873`（--default-mm-loras）、`#67870`（多模态缓存键域标签）、`#63912`（透明图复合）、`#67869`/`#63749`（错误码）、`#61788`、`#63902`/`#65625`（v0.34.0 Responses API / n>1）等。
- **CI/ROCm/Doc 约 35 条**：镜像/队列/缓存/测试等，详见第 8 节证据等级说明。

---

## 3. SGLang：2026-11-16 提交（74 条）—— #50156 合入即回滚 + swiglu_limit 静默(#50019) + GLM-4.7-Flash DCP 前缀(#50849) + Kimi-K3 MegaMoE DP(#50838) + HiCache 凭据脱敏(#49761) + HiSparse HiCache 数据面(#49473) + 跨硬件(AMD/NPU/NVIDIA)

### 3.1 `#50019` [Fix] CuTe DSL MoE runner 应用 swiglu_limit（fix，强化第 6 条）——见第 1.2 节

NVFP4 routed experts 未裁剪 SwiGLU，无报错、输出不同；cos 0.6355 → 0.9990（B200）。DeepEP 低延迟路径改为显式拒绝 swiglu_limit。

### 3.2 `#50849` [Fix] GLM-4.7-Flash DCP 前缀命中规划 attention metadata（fix，强化第 8 条）

`Glm4MoeLiteForCausalLM` 缺 `prepare_context_parallel_metadata_for_dcp` 钩子，`--dcp-size>1` 下 radix 前缀命中走 MLA 时 `bcg_dcp_extend_kv_gather` 解引用 None（flashinfer 后端首次前缀命中即 AttributeError）。加钩子后 GB300 2 卡 GSM8K cutedsl_mla 96.0→97.5%、trtllm_mla 94.0→95.5%，breakable graph 前缀命中 5/5 重放。**注意 CI Base/Extra/AMD 均 ✗**（记录为 fix 级、CI 未全绿）。

### 3.3 `#50838` [Fix] Kimi-K3 MegaMoE attention DP 传 EP-wide token 数（fix，强化第 5/7/12 条）

attention DP + megamoe 下各 DP rank prefill 长度不同则各挑不同 BLOCK_M（EP4 K3 形状 16/64/128/192/240），combine 永不完成、服务器超时中止（2/2 launch 挂起）。传 `global_num_tokens` 后 30/30 轮 0.3–0.7 s 通过；并扩展提示覆盖 DP 内 attention TP（DP4×TP2 on EP8）。

### 3.4 `#49761` [Fix] HiCache 存储凭据脱敏（security，强化第 18 条）——见第 1.3 节

S3 access/secret key 明文出现在 /server_info、启动命令、repr、crash dump、gRPC；本 PR 统一输出 `<redacted>`。CI 三套均 ✗（记录为 fix 级、CI 未全绿）。

### 3.5 `#50156` 合入后回滚（事件，强化第 3/16 条）——见第 1.3 节

5084 行插入的 Blackwell/Rubin hybrid GDN+QSA+MoE opt-in 服务优化栈，01:35Z 合入、04:17Z 被 #50703 以 "extremely invasive" Revert。

### 3.6 `#49473` [Feat] HiSparse 的 HiCache 后端数据面（feature，强化第 8 条）

新增 `HiSparseHiCacheBuffer`（Device/Host 索引寻址、驻留 KV 读、late-bound host buffer、per-request LRU），DSA indexer 可用独立于 attention KV 页的页表，使共享前缀 KV 页离开 GPU 后 DSA 索引仍可读。#32314 拆分 PR，Accuracy/Speed 段为空（描述级）。

### 3.7 性能与跨硬件（官方一手，含微基准）

- `#50660` [Perf][AMD] gfx1250 新增 `hier_ring_simple`（分区对上的 ring all-reduce）：8 卡 bf16 256MB 双 TP2 ring 跨加速器 634–777 µs → 每加速器发送量 1.75×→1.5× 张量，partition-pair 链路单 kernel 内完成。
- `#44612` [Perf][NVIDIA] CuTe DSL KDA MTP verify kernel 扩展到 SM90（Hopper）：GLM-5.3-Flash H200 attn DP8 + ReplaySSM 稳态解码吞吐 +2.7–6.5%、TPOT −3.0–6.0%，最重负载 +11.7%；逐层 verify 1.7–2.3×（GLM）/2.0–3.0×（K3）；GSM8K 97.08→97.27%（一标准误差内）。
- `#42224` [Perf][NPU] DSA INT8 LightningIndexer（A2/A3，opt-in `--indexer-kv-cache-dtype int8`）：indexer 2× 加速、降端到端 TTFT；与 hierarchical cache 组合显式拒绝（fail-fast）。
- `#50608` [Perf][NVIDIA] MegaMoE DeepGEMM 权重本地化扩展到 SM10.0/10.3（原仅 SM10.7，B300 实测可用，B200 未自测）。
- `#50840` [Fix] mamba extra_buffer_lazy 允许用于 PD prefill worker（production stack 实测 24/24 逐位一致，K3 GB300 699K→449K KV tokens）；`#49294` [NPU] DSpark 草稿层多流（GPQA 0.90/0.89，描述级无 speed 表）。
- `#48409` [Feat] 固定批次 bounded LPM 调度（最多 64 请求、限制跨批超越，仅在威胁重用 ≥1.5× KV 容量时启用）；`#44576` [Fix] PD prefill 完成释放多模态特征（PP8 图片流量第一阶段 scheduler 30 分钟 44.6→141.3 GiB → 38.9→38.8 GiB）；`#50572` [Fix] custom-ar-v2 按注册顺序 finalize 对称 workspace（修 PP2×CP2 启动挂起）。
- `#50157` [Feat] megakernel-binding 为 hybrid 模型提供 state/pool/prefill-graph/draft-step 视图（无插件时不变）。

### 3.8 其他要点（标题级 / 描述级）

新模型支持 `#46858` LFM3 文本/视觉/音频/Omni、`#49511` gemma4 --language-model-only；`#40333` DSv4.1 prefill CP with BCG in PD（2/N）；`#49294`/`#42224` NPU；`#48118` AMD gdn aiter 后端；diffusion 主题约 20 条（H3 Image FP8 attention、LTX2、Anima/MOSS-Image 16-pixel 对齐、end-to-end profiling 等，多数 Doc/Feat）；gateway 重构 `#50732`/`#50671`（删 sgl-model-gateway、统一 sgl-router）；`#50751` router NPU 产物名 + 多节点 KV events 警告。

---

## 4. TensorRT-LLM：2026-11-16 提交（41 条）—— SM103 NVFP4 静默(#22287, GSM8K 94→0.5) + GEN worker 挂起(#22220) + Kimi K3 MXFP8 共享专家(#21959) + FP4 MLA(#21313) + Qwen3.5 GB300 round2(#21997) + KVCM V2 覆盖(#21553) + 大量 infra/waive

### 4.1 `#22287` [fix] SM103 NVFP4 GEMM scale-factor 加载前等依赖 grid（fix，强化第 4/6 条）——见第 1.2 节

PDL 竞态读陈旧 scale，B300 DeepSeek-R1 FP4 GSM8K **94.05 → 0.53**（输出 garbage，请求照常完成）；根因 CUTLASS v4.4.2 scale-load warp 缺 `wait_on_dependent_grids()`，由 #21273 SwiGLU+NVFP4 融合暴露；SM100 不受影响。

### 4.2 `#22220` [fix] GEN worker 挂起：跨 TP rank 对齐 replicated GEN 取消与准入（fix，强化第 8 条）

异步非 ADP 解聚服务下 TP ranks 在不同时间退役被取消的 GEN 请求，一个 rank 准入新请求进 transfer consensus、另一个跳到 request broadcast → 挂起。方案：跨 TP/CP 对齐、共享 loop-head 投票取消、传播 cancel IDs、保留请求直到所有持有副本的 rank 都能退役。**PR 仍为 draft**（额外 per-iteration collective 开销未测）；692 CPU 测试通过；reported decode 栈与复现一致但生产侧触发请求/恢复未确立（描述级/draft）。

### 4.3 其他 feature/fix（官方一手）

- `#21959` [feat] Kimi K3 共享专家走 SM107 MXFP8 FC12 kernel（opt-in `KIMI_K3_MXFP8_SHARED_EXPERT=1`，仅 attention DP；TP 分片共享专家因会返回部分和而不转换，且 TP 场景下 MXFP8 解码最高慢 34%）。
- `#21313` [feat] FP4 MLA dense block 全块前缀复用 + 4× decode kernel + dispatcher（auto-generated，CI mixed——一次 abort/一次带 failed pipeline 成功/一次 failed；本地测试未独立验证，描述级）。
- `#21997` [feat] Qwen3.5-397B NVFP4 GB300 round 2（round1 #20815 之后；解码/预填 kernel、GQA KV-replica decode、MTP draft、stage-KV FP8 FMHA、KVCM2 remote tier over fabric/NIXL、输出长度预测 GEN 准入；184 个环境开关收敛为默认/派生/声明式 opt-in）。
- `#21553` [feat] Skip Softmax Attention 支持 KVCM V2（此前 `use_kv_cache_manager_v2=True` 默认下静默跑 V1 且 seat pool 按 V2 定容，无警告；强化第 3 条默认值风险）。
- `#22245` [fix] conversation KV retention 用现有 block key 守卫（重叠 context 请求在注册请求移除状态后完成，`save_drop_plan()` 未校验查找抛 KeyError 拖垮 worker；修后仅比较最后完整 committed block key，尾块变化仍允许推进）。
- `#21185` [feat] 用 nvMatmulHeuristics 裁剪 CuTe DSL GEMM autotuner tactics（默认关，opt-in `max_tactics`，~2000 tactics/shape 冷启动 JIT  profiling 成为启动瓶颈）。
- `#21181` [doc] NVFP4 MLA 技术博客（新增 466 行 + 5 SVG，README +3 行，描述级/文档）。
- `#20739` [feat] 输入处理与多模态编码器耗时纳入 request perf metrics（Prometheus，54 CPU 测试通过）。
- `#22287` 之外 infra/waive/test 约 28 条（post-merge waive、test-db 清理、CI 迁移等），记为 infra 级。

---

## 5. llama.cpp：2026-11-16 提交（34 条）—— Muse Glimmer 视觉嵌入静默(#32433) + OpenCL FA 三修(#32110) + RDNA3_5 MMVQ(#32308) + CDNA GDN mma(#31327) + Vulkan top-k 确定性(#31931) + KV 回收 O(n)→O(1)(#32380) + 多后端修复

### 5.1 `#32433` mtmd: 修 Muse Glimmer 视觉嵌入顺序（fix，强化第 19 条）——见第 1.2 节

### 5.2 `#32110` opencl: 修 bin flash-attention prefill 路径三缺陷（fix，强化第 8 条）

Adreno kernel library 的 `flash_attn_f32_f16_wmm` 路径（>1 query row）：① mask fallback image 每次调用创建于 K cache 子缓冲且不释放，投机验证每步每层泄漏、server 逐请求变慢（九请求序列 decode t/s 见正文表）；② q8_0 K/V 在 head size 512 且 V≠K 时结果错；③ 长上下文超过设备 3D image 宽度（X2-90 约 16k token）assert 失败。三处均回退到后端自带 kernel/释放资源。

### 5.3 `#31931` [HTML 一手] vulkan radix top-k tie 选择确定性（fix，强化第 9 条同类）

radix top-k（k≥1024 及融合 TOPK_QSA）用共享原子计数器选阈值以上与并列元素，tie 多于剩余槽位时保留列每次运行不同；改为按 Metal 实现（subgroup 拥有连续块、ballot 按列序排 tie）。RTX PRO 6000 Vulkan 上 master 每轮 1–2 个 tie case 失败 → 修后 TOP_K 534/534（3 轮）、TOPK_QSA 4/4；且去掉原子后 k=2048 geomean 0.77×、131072 列 0.68×（更快）。

### 5.4 性能/跨后端（官方一手，含逐卡表）

- `#32308` MMVQ RDNA3_5 ne11 Gate 调优：Strix Halo (8060S) gemma4 31B Q8_0 microbatch 8 pp128 从 39.07→48.51 t/s（1.24×），Q8_0 batch 7–8 改用 int8 mma（MMQ）。
- `#31327` HIP/CUDA 在 CDNA 启用 GDN mma kernel：MI100 qwen35 27B Q5_K_M pp1024@d32768 在 batch 512 从 422.29→529.34 t/s（1.25×）、batch 1024 1.19×；RDNA4/RDNA3.0 更慢故不启用。
- `#32380` [HTML 一手] kv-cells 用 #28040 的位置索引做 seq_rm，避免每次 O(n_cells) 全扫：Ryzen 7 5800X 上回滚末尾 4 位置，262144 cells/8192 used 从 233.08 µs → 0.60 µs（数百倍）。
- `#32344` [HTML 一手] 修 #29948 引入的 HIP MoE 回退：GFX906 gpt-oss-20b MXFP4 pp512 +6.09%、Qwen3.5-35B-A3B Q4_K_XL 相对 #29948 前 +22–26%（dense 对照无回归）。
- `#32385` [HTML 一手] CUDA MUL_MAT_ID 在 AMD 上 src0 奇数列（rank1 LoRA）触发 MMVF 断言 abort，改为仅偶数列走 MMVF（R9700 gfx1201 1040 通过 + 2 新 case abort → 1042/1042）。

### 5.5 其他要点（标题级 / 描述级）

`#32356` [HTML 一手] 全项目迁移 llama_batch_ext（AI 生成大部分迁移代码，人工审 2.5 h；realtime audio server 未测）；`#32436` granite hybrid 尊重 max_position_embeddings；`#32430` Inkling 2 mmproj 转换；`#32369`/`#32330` YuE2-3B / fish audio s2；`#30948` OpenCL A7 Q4_K non-MoE binary kernel；`#32244` [HTML 一手] Vulkan Nvidia fp32 FA shmem 溢出（shmem 超限时禁 shmem_staging）；`#32394` WebGPU per-head FA mask；`#29929` Vulkan conv3d split_k；`#32275` Vulkan 32 位编译；`#32236` hip gfx1013；`#31435` --tensor-split 允许最多 llama_max_devices 配置；`#32306`/`#32356` batch API 弃用/迁移；`#32032` Web UI 移至 ui/apps/chat（Llama Chat）；nightly b12441–b12445（窗口内，prerelease）。

---

## 6. 累计结论延续与新增（继承 1–19，本期新增 20）

**累计结论（1–19 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性（发版当天合入 ≠ 在版内；本期 v0.34.0 发布说明未逐条列本窗口 PR）；③ 默认值成为主要风险载体（本期 v0.34.0 `gpu_memory_utilization=1.0` + Extensible KV + block-size 64、vLLM #67315 双开关静默失效、TRT #21553 V1/V2 静默错配）；④ 异构/量化后端静默错误为高频（本期 TRT #22287 SM103 PDL、SGLang #50019 CuTe DSL swiglu_limit、vLLM #62474 MoRIIO）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（本期 SGLang #50838 MegaMoE DP token 数、#50849 DCP 前缀元数据、#50840 mamba PD prefill）；⑥ 量化路径静默错误（scale/布局漏接，本期 TRT #22287 陈旧 scale、SGLang #50019 漏 SwiGLU 裁剪）；⑦ 专家张量/路由归属与分页漏接静默错误（本期 SGLang #50838 EP-wide 行数、#50019 routed expert 裁剪）；⑧ KV 卸载/HiCache/PD 健壮性 + 内存/缓存管理接口重构（本期 vLLM #65575 Mooncake 失联、#66583 async CPU tier、#67486 假 flush、SGLang #49473 HiSparse HiCache 数据面、TRT #22220 GEN 挂起）；⑨ 长上下文 int32/int64 溢出静默错误（本期 llama.cpp #31931 top-k tie 非确定性，归属索引/确定性同类）；⑩ MXFP8 KV 正确性簇；⑪ 新旗舰模型 CP/KV/性能成为一等公民（本期 vLLM #67094 DSpine、#66324 MiniMax-M3.1-Flash GB300、SGLang #46858 LFM3、TRT #21959 K3 SM107）；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径（本期 SGLang #50608 SM10.0/10.3、#50838 MegaMoE DP）；⑬ FP8/低比特 KV cache 跨引擎标准化落地（本期 vLLM #65198 draft KV dtype 对齐进 v0.34.0）；⑭ 投机解码验证路径跨引擎密集硬化（本期 vLLM #67094 DSpine 五数据集、#66311 LFM3 EAGLE3、SGLang #49294 NPU DSpark）；⑮ 跨后端可移植性成为显式架构主线（本期 llama.cpp #31327 CDNA/#30948 OpenCL/#32394 WebGPU/#32385 AMD CUDA、SGLang #50660 gfx1250/#42224 NPU、vLLM ROCm 簇）；⑯ 新旗舰模型进入「默认性能化」阶段 + 量化/MegaMoE 通用化（本期 v0.34.0 多项默认性能化落地）；⑰ 推理服务前端 Rust 化成为跨引擎并行主线（本期 vLLM Rust frontend 约 15 条、SGLang #50572/router 簇）；⑱ 推理服务默认安全加固成为跨引擎主线（本期 vLLM #67316 压缩包解压炸弹、SGLang #49761 HiCache 凭据脱敏）；⑲ 非量化的常规优化路径静默错误密度不亚于量化路径（本期 vLLM #62474 + llama.cpp #32433 再添坐标/归属类铁证）。

**新增第 20 条（正式版本发布本身集中固化「默认值风险 × 破坏性变更」，且升级检查表成为必读项）**：v0.34.0 一次发布同时把 Extensible KV 默认开、`gpu_memory_utilization=1.0`、block-size 64、Model Runner V1 移除、Marlin W4A8/QuTLASS/Keye-VL 移除、ROCm 7.2 停止发布、依赖版本下限抬高等 8 类破坏性/默认变更写入 "Before you upgrade" 表。研判（观点）：本追踪此前把「默认值风险」归结为单个 PR 的隐式行为改变（如 #63231 block-size、#61258 NUMA），本期表明**正式发版是该风险的集中兑现点**——尤其 `gpu_memory_utilization=1.0` 把 KV 容量推向物理极限（官方明示 TPOT +6–18%），要求升级时重做吞吐/尾延迟标定与依赖审计。配套方法论不变：发布说明是精选 highlights，**不能据此判断某修复是否进版**，须以 compare API 祖先关系为准。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 升级 vLLM v0.34.0 的破坏性变更核验**：`VLLM_USE_V2_MODEL_RUNNER=0` 将启动失败、默认 block-size 64（ROCm paged 仍 16）、`gpu_memory_utilization` 默认 1.0（吞吐↑但部分模型 TPOT +6–18%，建议显式设 0.9 对比）、Marlin W4A8→Humming、QuTLASS/`fp_quant` 与 Keye-VL 不再加载、ROCm 仅 10.0、huggingface_hub≥2.0.0/tokenizers≥0.23.3/timm≥1.0.30；**升级前用 "Before you upgrade" 表逐项对照**。
- **P0 — 量化/PDL 静默正确性复核**：TRT-LLM `#22287`（SM103 NVFP4 GEMM PDL 读陈旧 scale，B300 GSM8K 94.05→0.53，#21273 融合暴露）、SGLang `#50019`（CuTe DSL MoE 漏 swiglu_limit，cos 0.6355→0.9990）；NVFP4/FP8 MoE 部署须核验 scale 加载的依赖栅栅与激活裁剪参数是否在所用 backend 生效（SM100 不受 #22287 影响）。
- **P0 — 坐标/归属类静默错误（第 19 条）**：vLLM `#62474`（MoRIIO 异构 TP 用 model-wide KV head 切片写错 head，Gemma4/EAGLE3 drafter）、llama.cpp `#32433`（Muse Glimmer/Muse Spark Open 视觉位置嵌入 h/w 颠倒）；异构 TP、视觉编码器移植须核验每层 head 数与 (h,w)/(x,y) 坐标序。
- **P1 — KV 连接器/卸载健壮性（第 8 条）**：vLLM `#65575`（Mooncake P 失联 545 s→2–4 s）、`#66583`（async CPU tier 启动 439→334 s，默认关）、`#67486`（共享前缀假 flush）；SGLang `#49473`（HiSparse HiCache 数据面，描述级）、TRT `#22220`（GEN 取消跨 TP 对齐，draft）。
- **P1 — 投机解码新成员/验证（第 14 条）**：vLLM `#67094`（DSpine on Qwen3.8-27B，五数据集接受长度）、`#66324`（MiniMax-M3.1-Flash GB300 opt-in，interactivity 1.45×）、`#65198`（draft KV dtype 对齐，进 v0.34.0）；SGLang `#44612`（KDA verify kernel 上 SM90，+2.7–6.5%）。
- **P1 — 微基准 vs 服务级 / 单指标权衡**：llama.cpp `#32308`（RDNA3_5 Q8_0 1.24×）、`#31327`（CDNA 1.25×）、`#32380`（KV 回收数百倍，CPU 侧）、vLLM `#65578`（GELU+NVFP4 −18–21%，cutlass 后端）、`#63272`（Hadamard 融合恢复 6–12%）；关注硬件/后端特定性与口径。
- **P1 — 服务安全（第 18 条）**：vLLM `#67316`（解压炸弹）、SGLang `#49761`（HiCache 凭据脱敏，CI 未全绿）；暴露服务端点前核验输入体积上限与凭据外泄面。

---

## 7. 社区与视频资源（HN Algolia，2026-11-16 epoch 窗口 1794787200–1794873599）

> 以下均为社区/Show HN，非引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词检索中 `sglang`/`tensorrt-llm`/`llama.cpp`/`speculative decoding`/`KV cache` 均 **0 命中**；`inference engine` 1 条；`vLLM` 17 条多为泛 AI 资讯与模糊「LLM」匹配。

- **本地推理引擎**：「Show HN: Goethe – turn your computer into an inference engine」（**4 分**，2026-11-16T20:52）——社区本地推理引擎项目，与本期跨硬件/端侧主线（第 15 条）主题相关，但无一手来源确认内容与性能。**二手线索/未核验**。
- **硬件基准**：「Intel benchmarks D270 vs. RTX PRO 6000: 1.12x throughput on Qwen3-30B-A3B (vLLM)」（**2 分**，2026-11-16T18:51）——社区转述的 Intel vs NVIDIA 基准，提及 vLLM；数字未经官方一手核验。**二手线索/未核验**。
- 其余 HN 命中（`vLLM` 17 条如「A new implementation for LLMs through Nvidia GPUs natively on Apple Silicon」1 分、「Near-Realtime LLM Steering」8 分、`inference engine` 外）多为泛 AI/LLM 资讯、工具或舆情，与推理引擎实现无直接关系，未列入。本期**无引擎相关的可确认官方视频**；Reddit / YouTube / X 未采集。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **113**（分两页 100+13，为追踪以来单日新高，超过此前 2026-10-27 的 102）/ SGLang **74** / TRT-LLM **41** / llama.cpp **34** 合计 **262** 条（2026-11-16 UTC 窗口；周一，较上一窗口周日 116 大幅回升，属工作日正常量）。与上一期（2026-11-17 档案覆盖 11-15）连续，无空档。
- **版本核验**：releases API 确认——**vLLM v0.34.0 于 2026-11-16T23:22:00Z 正式发布（非 prerelease，1061 commits/412 贡献者）**，发布说明全文 47,575 字符核验；SGLang 仍为 v0.5.23（2026-11-02）、TRT-LLM dev 线 v1.4.0rc3（prerelease，2026-11-06）、llama.cpp 窗口内推进 nightly b12441–b12445（prerelease，v1.0.0 正式版仍为 2026-11-13）。发布说明为精选 highlights，未逐条列本窗口 PR（#67315/#65575/#62474/#67316 出现 0 次）——按本追踪方法论，不以发布说明作为「修复已进版」依据。
- **PR 正文来源**：**50 个关键 PR 正文抓取**——vLLM 20 + SGLang 16 + TRT-LLM 10 经官方 issues API（markdown 官方一手），llama.cpp 前 4 项经 issues API、后 6 项因未认证 60/h 配额用尽（起始 51→0）改由 GitHub PR 页面 HTML 按 div 深度配对提取 `#32244`/`#31931`/`#32385`/`#32380`/`#32344`/`#32356`（均取得有效正文 229–2931 字符）。其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度。
- **描述级/未核验标注**：vLLM 发布说明性能数字为官方报告值（非独立复现）；`#66324`/`#63272`/`#65578`/`#64406` 含微基准/特定后端口径（cutlass vs 默认后端差异已并列）；TRT-LLM `#22287` 修复本身有完整根因与 GSM8K 对照但为 fix 级、`#22220` 仍为 draft（开销未测、生产触发未确立）、`#21313` auto-generated 且 CI mixed、本地测试未独立验证、`#21181` 为文档；SGLang `#50849`（CI Base/Extra/AMD ✗）、`#49761`（CI 三套 ✗）按 fix 级/CI 未全绿记录，`#49473` Accuracy/Speed 段为空（描述级）、`#49294` speed 表缺失（描述级）、`#50156` 已被 #50703 回滚（"extremely invasive"）；llama.cpp `#32356` 迁移代码 AI 生成、realtime audio server 未测，`#32433`/`#32110` 为 fix 一手。vLLM `#66311` 为 issue 型（本地 patch 验证）。
- **社区信号**：HN Algolia 引擎官方更新类故事无；社区 Goethe（4 分）、Intel D270 vs RTX PRO 6000 基准（2 分）均按二手线索/未核验记录；无引擎相关的可确认官方视频。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/CI 未全绿/draft/微基准」。
- **技能检索（find-skills / agent-browser）**：find-skills 经 SkillHub 语义检索（EN+ZH 各 1 组，含「LLM inference GitHub commit monitoring daily hotspot digest」「大模型推理 GitHub 提交采集 热点归档 仓库同步」）最高相关性仍为 0.050，返回编程大神.Skill / 多引擎搜索 / 育儿大师 / 电子印章等通用编程/办公/育儿类技能，与「官方提交采集 + 证据分级归档 + 仓库同步」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/issues API + PR 页面 HTML）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次一致。
