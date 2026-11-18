# 大模型推理优化每日热点档案

> 档案日期：2026-11-18（本自动化于该日执行）
> 关注窗口：2026-11-17（周二）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 34 个关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 均经官方 issues API 抓取 markdown，官方一手）+ HN Algolia（HTTPS，numericFilters 按 2026-11-17 epoch 窗口 1794873600–1794959999）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 91 / SGLang 99 / TRT-LLM 73 / llama.cpp 35 合计 **298** 条（2026-11-17 为周二，SGLang 99 与 TRT-LLM 73 均处近期高位；vLLM 91 较上一窗口的 113 回落、单页取全无截断）；**窗口内无新正式版本**——vLLM v0.34.0（2026-11-16T23:22:00Z）仍为最新正式版，SGLang v0.5.23、TRT-LLM v1.4.0rc3（prerelease）、llama.cpp 正式版 v1.0.0 均未变，llama.cpp 仅推进 nightly b12480–b12485（prerelease）；34 个关键 PR 正文全部抓取成功（issues API 配额充足、无降级）。HN 引擎相关高信号为 Show HN「TrnRT」（43 分，仓库已核验存在，二手线索/未核验）与 tensward 博客「一个环境变量让 llama.cpp 快 21%」（1 分，未核验）。

---

## 1. 版本状态与本期头号事件：静默正确性多引擎密集爆发（KV 布局/scale 漏接/整型溢出/权重时序）+ 在线权重热更新时序竞态成新类（SGLang #50717）+ 跨期 PDL 读顺序再现（TRT-LLM #22258）

**提交计数（2026-11-17 UTC，均 HTTP 200；周二）**：

| 引擎 | 提交数 | 备注 |
|---|---:|---|
| vLLM | 91 | 较上一窗口 113 回落；Kimi-K3.1-Pro recipe 簇 + 静默正确性 + 跨硬件性能 |
| SGLang | 99 | 近期高位；Kimi-K3.1 Day-0 合入、权重更新时序、KV ring、GC 冻结簇 |
| TensorRT-LLM | 73 | 高位以 infra/waive/test 为主，feature/fix/perf 约 20 条 |
| llama.cpp | 35 | 多后端（CUDA/HIP/OpenCL/SYCL/Vulkan/WebGPU）+ 量化转换 + UI |
| **合计** | **298** | 工作日正常量；与上一窗口（11-16 的 262）连续，无空档 |

### 1.1 本期头号事件：静默正确性跨四引擎密集爆发，且「无报错错误输出」占比高

本窗口一手静默/崩溃级案例覆盖四引擎，既含量化 scale 漏接（延续第 6 条），也含非量化的 KV 布局、索引位宽、权重时序错误（延续第 19 条）：

- **vLLM #68014**（官方一手，fix）：`FLASHMLA_SPARSE` + HiSparse 在 FP8 稀疏解码里崩溃并**在崩溃前读到错误 KV 行**。HiSparse 把 top-k 解析为热缓存 flat-row 下标（`block*stride+offset`），`fp8_ds_mla` 下热块与其他层行交错、stride 为 271 而非 64；旧代码把该缓存当 64-token 页传给 FlashMLA（`page*64+offset`），多数下标指向错行、高块越界。修复改为按 flat rows 读取。GSM8K（GLM-5.3-NVFP4，TP8 8×B200）：resident 配置 main 约 180–590 请求后 **illegal memory access → 0.9174**，forced offload main 崩溃 → **0.9280**，kernel 位精确（max diff 0.0）。属 KV 分层/页布局错配，强化第 8/19 条。
- **vLLM #67489**（官方一手，fix；修 #67307）：1×H100 上 DeepSeek-V2-Lite prefetch offload 在第一个大 GSM8K step 于 FA3 MLA 内 OOM、引擎崩溃、nightly **accuracy 0.000**。根因：decode 路径 batch 超过最大已捕获 CUDA graph 时 `FlashAttnMLAMetadataBuilder` 传 `num_splits=0`，FA3 分配未使用的 fp32 split-KV buffer（该 step 1.2 GiB、其他形状最高 6.5 GiB）。改传 `num_splits=1` 后 KV 块数不变（26766，位精确），3/3 通过、accuracy 0.335–0.345。延续第 9 条邻域与 KV 容量自检维度。
- **SGLang #50870**（官方一手，fix）：AMD Triton sparse MLA 四个 kernel 的 token 下标来自 32 位 `tl.program_id(0)`，超过 2^31 元素回绕，**wrong output 无报错**（single-pass）或显存访问异常（split-K/fused）。GLM-5.x / DeepSeek-V3.2 在 `--dsa-prefill-backend triton`（ROCm FP8 KV 自动选）触发；MI355X TP2 实测 131,072 token 单次 prefill 中 **14,563 token 从 token 116,509 起读错内存、无报错**，split-K 直接 fault；配套修复 ROCm/aiter#7952（RoPE+KV-cache kernel 同类溢出）。CI 三套均 ✗（记录为 fix 级、CI 未全绿）。强化第 9 条。
- **llama.cpp #32393**（官方一手，fix）：Qwen3-Next GDN 输出投影漏传 `ssm_out.scale` 给 `build_lora_mm()`，原生 NVFP4 / per-tensor FP8 权重下**推理无异常但 GDN 输出量级错误**（加到残差上即污染）。属量化 scale 漏接，强化第 6 条。
- **llama.cpp #32476 + #32483**（官方一手，fix）：FP8 `qwen4exp` checkpoint 转换中 n-gram（PLE）表只存 F8_E4M3 shard + 共享 scale，旧转换丢 scale 导致表值**约 5000× 过大且无报错**（`--outtype f16/bf16/f32`）；ModelOpt 路径还丢 `weight_scale_inv`（128×128 块，MTP routed experts）。#32476 修转换、#32483 修其引入的 CUDA/ROCm CI graph reallocation abort（scale 是 host weight，`ggml_mul` 按 batch 切后端 → graph 不匹配；改 `ggml_scale` 读值）。两份均以 BF16 原模型对照（相对误差 0.027 vs 未缩放 5017/约4700）。强化第 6 条。
- **llama.cpp #31536**（官方一手，fix）：SYCL MKL flash-attention 在 mask 跨 head/batch 广播时越界读，f16 K/V 触发 `UR_RESULT_ERROR_DEVICE_LOST`、q8_0 K/V 返回 NaN；按 CPU 后端取模语义修。属索引/边界静默错误，强化第 19 条。

### 1.2 新增累计结论第 21 条：在线权重热更新的时序竞态成为静默正确性新子项

- **SGLang #50717**（官方一手，fix）：`pause_generation(mode="in_place")` 后，暂停前发出的 step 可能仍在 `forward_stream` 运行，而权重更新在 `schedule_stream` 写参数；WAR barrier 只等在途 step 的 shared-read event（标记输入读完，早于模型层），**在途 step 用新权重计算，其 token/logprob/KV 被保留且无报错**。修复在 7 个权重写入入口前加 `wait_for_launched_forward()`。GB200 Qwen3-0.6B 实测：未修复 8/8 次更新后首 token 为新权重 token，修复后 0/8。研判（观点）：此前静默正确性主线聚焦 KV/量化/索引/整型溢出的数据面，本例是**控制面（权重生命周期）与在途计算的时序竞态**，是一类新的静默错误来源，热更新/RLHF/权重回滚场景须显式同步。

### 1.3 跨期呼应与其他头号线索

- **跨期 PDL 读顺序再现**：上一期（11-17 档案）TRT-LLM #22287 为 SM103 NVFP4 GEMM PDL 读陈旧 scale；本期 **TRT-LLM #22258**（perf，声称 bitwise identical）把 split-KV reduction 中设备 softmax scale 的读取移到 `cudaGridDependencySynchronize()` 之后，并把 sink/stats 的 FP64 _math 放进真实分支（SM103 FP64 为 1/64 FP32 速率）。两期同属「PDL 依赖同步次序」主题，本期偏性能（作者报告 GB300 数字、评审未验证 bit-identical 声称）。
- **TS/路由收敛（行为不变）**：vLLM #67338 把稀疏 MLA prefill 的三种路线（dense MHA / masked MHA / sparse MQA）的决策从三处收敛到 metadata builder 一次决策并存 `route` 字段，消除「三处各算一部分、七处重推导」导致的不一致隐患，官方声明 serving behavior 不变（属防御性重构，非性能）。
- **GC 冻结堆防悬垂簇（SGLang）**：#50883/#50980/#50981/#50982/#988b0439/#aa3ccefd 同窗口落地「idle freeze + collect before freezing + keep heap frozen after CUDA graph capture + frozen garbage 告警」，把堆冻结纳入 CUDA graph capture 生命周期以防引用失效；属新的内存/图捕获健壮性主线，本文按标题/描述级记录。
- **安全**：vLLM #67765（错误消息中非法模型名长度设界）、#67314（采样视频帧在解码前计入请求预算）、SGLang #49290（DiffusionServer 入站请求改用 `safe_pickle_loads`）。延续第 18 条。

---

## 2. vLLM：2026-11-17 提交（91 条）—— 静默正确性(#68014/#67489/#68021/#66839) + Kimi-K3.1-Pro recipe 簇 + DSv4.1 gfx942 TP8 monokernel(#67540, 2.16×) + QSA gvr_2(#67741, +2.4% E2E) + SWA split-K(#61646, 8.7×) + W4A16 carrier(#64375)

### 2.1 `#68014` [Bugfix][HiSparse] 热缓存按 flat rows 读（fix，强化第 8/19 条）——见第 1.1 节

### 2.2 `#67489` [Bugfix] 超 CUDA graph 尺寸的 FA3 MLA decode batch 免 OOM（fix，延续第 9 条邻域）——见第 1.1 节

### 2.3 `#68021` [Bugfix] Muse Glimmer 视觉嵌入顺序（fix，强化第 19 条）

与上一窗口 llama.cpp #32433 为**同一模型同一 bug 的跨引擎修复**：vLLM 侧视觉编码器按 (h,w) 采样 learned position-embedding grid，模型训练用转置 `xy` 顺序（Muse Spark Open 在 vLLM 已用 xy），port 时遗漏。方框图不变；新增 `pos_emb_layout` 测试（非方框无修复则失败）；Muse-Glimmer-30B 640×480 图前后均正确描述。研判（观点）：跨引擎独立出现同一移植错误，说明视觉位置嵌入坐标序是新模型 day-0 支持的高频陷阱。

### 2.4 `#67741` [Perf][Qwen4Exp] QSA top-k 默认走 FlashInfer gvr_2 on SM100（feature/perf，强化第 16 条「默认性能化」）

对 SM100 族（GB300/B200）QSA indexer 的 top-k 用 `gvr_2`（k∈{512,1024,2048}、logits 宽度 4 的倍数），其余保持 cooperative/persistent；`auto` 默认 prefer `gvr_2`，可用 `sparse_indexer_topk_backend` 覆盖。**kernel（GB300 CUPTI，冷 L2）：长上下文 decode 1.30–2.25×（1→1024 行），prefill 长上下文 1.28–2.54×；短上下文小 batch 回退 0.70–0.77×。** E2E（Qwen3.8-Flash-Next-FP8，4×GB300 TEP4，128k 输入 c16）输出吞吐 **+2.4%**，8k 输入 −0.2~−0.7%。精度 GSM8K 96.89→97.01%、LongBench v2 37.2→38.2%。限制：gvr_2 延迟依赖 score 分布（测试用随机 q/K）、warmup 最多 +24 s、仅 GB300 实测（B200 仅 CI 正确性）。

### 2.5 `#67540` [ROCm][Perf][DSv4.1] gfx942 TP8 mono decode（perf，强化第 15/16 条）

为 gfx942（MI300X/MI325X）开启 `--monokernel` 的 TP8（此前仅 TP4），TP8 下每 rank 8 个 attention head pad 到 16 宽 tile、routed expert 中间宽度 2304/TP=288。8×MI300X：**decode step 12.50→5.80 ms（2.16×, c1）、16.73→9.81 ms（1.71×, c4）**，GSM8K 0.9234 vs 0.9280（McNemar p=0.50，无差异）；AgentX interactivity P90 2.09×/1.97×/1.54×（c2/4/8）。**代价**：权重 89.09→124.76 GiB、KV 69.52→33.82 GiB、CUDA graph capture 127→795 s、就绪 8.5→24.5 min（官方自述 kernel 自带权重副本）。属 opt-in、硬件特定。

### 2.6 `#61646` [Perf][Attention] 仅切滑动窗口的 Triton split-K decode（perf，强化第 15 条）

TRITON_ATTN 3D split-K 旧路径按整段序列切分，窗口远小于上下文时仅 1–2 段命中窗口且串行扫描；改为只把窗口 tile 切段，滑动层 decode 开销不再随上下文增长。R9700（gfx1201）窗口 2048：**batch1 32K 2.7× / 64K 4.8× / 128K 8.7×**，batch4 64K 及以上约 10×；默认 16 段路径 128K batch1 约 11×。E2E Muse Glimmer 30B（4×R9700）TPOT −4.9%@12K、GSM8K 0.9363→0.9348（±0.0067、99.2% 答案一致）。

### 2.7 `#64375` [ROCm][RDNA][Perf] W4A16 融合 scale/zero-point carrier（perf，硬件特定）

gfx1151（Strix Halo）非对称 W4A16 把每 group 的 scale/zero-point 打包成 int32 carrier，配 #60644 的 fp16 Triton prefill 反量化。**batch1 prefill 端到端最高 −18.4%**（Qwen3-8B @224 token）、batch128/256 快 1–11%；kernel g128 geomean 在 M193–256 快 15–18%。代价：权重内存 +8/G（g128 为 6.25%、g64 为 12.5%）；GSM8K 一题内、对称层位精确。

### 2.8 `#67136` [Quantization] online NVFP4 MoE 支持 CPU offload（fix/feature）

两处修复：① `--cpu-offload-gb` 下任何在线量化模型因 `UVAOffloader.wrap_modules` 在 meta tensor 上操作而加载失败（`Cannot copy out of meta tensor`）；② `nvfp4_per_token` MoE 用 `stage_on_cpu` 逐 expert 上 GPU 量化，全 BF16 层不上 GPU。GB300 实测 Muse Spark Open 权重加载峰值 225 GB→**201 GB**（接近无在线量化的 203 GB），加载 +18 s（+4.6%）；scales 位精确。

### 2.9 其他要点（标题级 / 描述级 / 性能级）

- **KV/连接器**：`#67960` CPU KV offload buffer 排除 core dump；`#64574` MoRIIO 在 TP>1 且 DP>1 时握手/通知端口冲突致 TP2×DP2 prefill 启动挂起（939 unit passed，MI350X GPT-OSS-120B 端到端 170 s ready、4/4 成功）；`#8c9e6f91` 仅回填 DFlash/DSpark 滑窗草稿 KV 尾部（RFC #59117 池崩溃的一个成因，草稿上下文只从末 tail 读）；`#cb55df89` D2H 拷贝改用 device-generic stream/event（#67724，使投机解码可在 XPU/非 CUDA 启动，Ascend NPU 20 passed/7 skipped）。
- **Kimi-K3.1-Pro recipe 簇（约 6 条，标题/描述级）**：`#68093`/`#68220`/`#68213`/`#67719`/`#67722`/`#67807`——B300/GB300 AgentX 分离式 preset、DSpark draft 上线、前端 session affinity/权重加载格式，与 InferenceX#5068/#5069/#5108 对齐；属部署配方、非引擎核心改动。
- **性能/新模型（标题/描述级）**：`#65706` GLM-5.3 FlashInfer warp split-K GEMM 进 SM107 低延迟表并修 q_b M=1 fallback；`#c2d218ee` QSA top-k gvr_2（见 2.4）；`#d92a7652` copy-engine PP transport（TP>1/PCP>1/多节点 NVLink）；`#db8e4658` DSV4.1 candidate prefill/decode CuTe；`#3ea7fcdc` Muse Spark FP8 KV 专用 decode；新模型 `#58387` NemotronLabsDiffusion（tri-mode）、`#63114` mammoth-1/elephant-1（block-bidirectional mask）、`#68078` MiniCPM-OCR 1.0、`#444a9f5b` Nemotron VL CV-CUDA GPU resize。
- **安全/其他**：`#66839` top-k kernel 将 NaN score 归到 +inf 之上（排序一致性）；`#67765` 非法模型名长度设界；`#67314` 采样视频帧计入请求预算；`#048e1b88` LoRA 拒绝同 id 不同 adapter 复用；CI/Doc/ROCm/XPU 约 30 条，详见第 8 节。

---

## 3. SGLang：2026-11-17 提交（99 条）—— 权重更新时序静默(#50717) + int32 溢出(#50870) + DFlash MHA KV ring(#50740) + PD retraction 挂起(#50712) + SWA 多轮链式 miss(#50059) + Kimi-K3 prefill CP(#38344) + GC 冻结簇

### 3.1 `#50717` [Fix] 权重写入前等在途 forward（fix，新增第 21 条）——见第 1.2 节

### 3.2 `#50870` [Fix][AMD] Triton sparse MLA 用 64 位 token 下标（fix，强化第 9 条）——见第 1.1 节

### 3.3 `#50740` [Feat] DFlash MHA 草稿按请求 KV ring（feature，强化第 5/14 条）

把 #49664 的逐请求 ring 从全滑窗 MLA 草稿扩到全滑窗 **MHA** 草稿（`trtllm_mha`/`fa4` 后端）。Muse-Spark-Open DFlash2 TP4：草稿 KV 恒为 2176 行/请求（0.17 GB/rank@32 请求），释放内存给目标 KV，**max_total_num_tokens 7.83M→9.37M（+19.7%）**；共享前缀基准 +5.3% 吞吐、TTFT −7%。**代价（官方明示行为变更）**：前缀命中后草稿看不到命中部分直到新 token 填满窗口，bs=1 命中后 2400 token 接受长度低 7.5–17.7%。GSM8K 97.65→97.57%（一标准误差内）/ PD 0.9773→0.9765。默认对符合条件草稿开启。

### 3.4 `#50712` [Fix] PD retraction 期间继续轮询 decode transfer queue（fix，强化第 8 条）

`process_decode_queue` 在 retracted 队列非空时提前返回、不轮询 transfer queue：KV 已到达的请求占着页和 `num_reserved_decode_tokens`，被 retract 的请求再也放不下，服务器停摆而 `/health` 仍 200。改为仅跳过 `_preallocate_new_requests`、继续 `move_transferred`。GB200 Qwen3-0.6B + Mooncake TCP 实测：未修复 decode 服务冻结整 360 s（363/363 次探活 200），修复后全部完成。CI Extra/AMD ✗（记录为 fix 级、CI 未全绿）。

### 3.5 `#50059` [Fix] SWA 分支点同 checkpoint 插入 prompt end（fix，强化第 5/8 条）

多轮 SWA 对话中，单 chunk prefill 且 decode 跨过一个窗口时，prompt end 被存成 SWA tombstone，下一轮又只匹配到分支点，miss 逐轮链式传播。修复后在覆盖整 prompt 的 checkpoint 内同批插入后续分支。MI355X TP2 + Rust TreeCore、AgentX 1 h replay：**miss 98→25（4.3%→1.1%）、P(miss|上一轮 miss) 0.796→0**、最长链 6→0；新增 prefill token −2.0%，TTFT p90 −10.5%、吞吐 +0.1%。GSM8K 0.9727→0.9704（3 题、一标准误差内）。CI Extra/AMD ✗。

### 3.6 `#38344` [Feat] Kimi K3 prefill 上下文并行（CP，feature，强化第 11 条）

K3（93 层：69 KDA + 24 MLA，896 NVFP4 专家 top-16）全序列 CP：仅 MLA 层按 zigzag 分 token shard，KDA/MoE 仍全序列 + TP8 集合通信；CP8 时 A=1、MLA o_proj all-reduce 消失。8×B200 真权重 8 层 proxy：CP8 分支默认 32K/64K prefill **+45%/+87%**（比 TP8 慢），Phase 0（默认）降到 +0.3%/+2.4%，Phase 0+Phase 1（env opt-in）**−16.6%/−15.4%**。CI Base ✗、AMD 进行中（记录为 feature 级、CI 未全绿，proxy 非全模型）。

### 3.7 `#50719` [Perf][NVIDIA] Hopper 上用 DeepGEMM fp8 MQA logits 打分 DSv4.1 prefill indexer（perf）

旧 `torch` fallback 每请求每层做 bf16 score chunk + head sum + `torch.topk` + 主机同步；本 PR 用 `widen_fp4_to_fp8`（每向量 1 个 fp32 2 次幂 scale，14 binade 内位精确）+ `fp8_mqa_logits`（SM90 已有）替换打分 kernel，消费者路径不变。8×H200 DSv4.1-Flash：GSM8K 98.00%→98.00%、RULER2 128K 74.45→74.63%；kernel 对 fp64 参考误差 ≤9.4e-8（torch 路 5.3e-3~8.4e-3），top-512 全行一致。DSPARK 接受长度未测。

### 3.8 其他要点（标题级 / 描述级 / 性能级）

- **跨硬件/新模型**：`#6321f714` Kimi-K3.1 Day-0 剩余分支合入；`#48065` gfx1250 DSv4 mHC 权重跨层转置读（无此修复 TileLang `mhc_fused_post_pre` 编译失败、服务起不来，修后 GSM8K 0.965、decode VERIFY 17.77 ms）；`#9e14a219` NPU DSA decode CP 的 FP8 KV；`#44447` Qwen3.8-Flash-Next KVShard（CP/TP prefill 共享组、Mooncake/NIXL gather，CI 三套 ✗）；`#50747` AMD MI35x 上 Muse Spark Open MXFP4 DFlash2 AgentX 曲线；`#9c1b3ece` beam-search trie head + shared-context attention；`#eed42710` Qwen3.8-Flash-Next NPU graph replay/MTP/Triton kernel。
- **GC 冻结簇（标题/描述级）**：`#50883`/`#988b0439`/`#aa3ccefd`/`#50980`/`#50981`/`#50982`——idle freeze、记录 GC pass 与长 GC 告警、冻结前先 collect、graph capture 后保持堆冻结、frozen garbage 改为告警而非报泄漏。
- **inkling2 簇（标题/描述级）**：`#471ae1b3`/`#41419ca9`/`#70b34f32`/`#98c45e11`——相对偏置 decode kernel、MoE routing/量化/grouped GEMM、dense path、all-reduce/exchange kernels；`#0596e4d7` AMD Inkling ROCm 安全初始化。
- **AMD/NPU 性能**：`#5830ea82` gfx942 AITER mHC kernel；`#5cefe8de` gfx1250 AITER MegaMoE + mori dispatch；`#87f393c2` DSv4 aiter sparse prefill + fp8 wo_a + mxfp8_128 GEMM；`#04af670a` DSv4.1 low-ratio extend slot gather 去 host sync；`#435ee288` Intel DP shm allgather/reduce-scatter；`#3400b40b` NPU MiniMax-H3 RMSNorm/SwiGLU。
- **diffusion（约 20 条，标题/描述级）**：`#49290` safe_pickle_loads；`#9d9b45d4` MiniMax-H3 rank-local TP checkpoint；`#8241f8e6`/`#2123631e` Ideogram4/FLUX.3 对齐 16-pixel grid；`#ba932654` Minimax-Music3。
- **安全**：`#49290` DiffusionServer 入站请求 `safe_pickle_loads`（延续第 18 条）。

---

## 4. TensorRT-LLM：2026-11-17 提交（73 条）—— PDL 读顺序(#22258, 跨期呼应) + DSv4.1 CSA2 indexer(#22361) + 混合精度前缀复用(#21619) + KVCM v2 agentic(#21935) + FP4 MLA SM100/103(#22078) + BREAKING ModelLoader(#20711) + 大量 infra/waive

### 4.1 `#22258` [perf] trtllm-gen split-KV reduction：PDL wait 后读 scale + FP64 分支（perf，跨期呼应）——见第 1.3 节

`fmhaReductionKernel` 两处改动均声称 bitwise identical：① attention sink 项与导出 softmax stats 的 FP64 数学（`M_LOG2E` 为 double）被 nvcc if-convert 成谓词区、SM103 FP64 为 1/64 速率，改写成显式 FP64 FMA 使 sink/stats 落入真实分支；② 设备 softmax scale 原在 `cudaGridDependencySynchronize()` 前读（PDL 下非Ensure顺序），移到 wait 后。**评审指出 bit-identical 声称未验证**、作者 GB300 数字未独立核验、CI 当前 head 无结果、test-list 未确认，记为 perf 级/needs follow-up。

### 4.2 `#22361` [perf] DSv4.1 CSA2 decode indexer block-hierarchical Top-K + group consumer Top-K（perf）

SM100/SM103 新增两条 selection 路径（block-hierarchical top-k、group top-k）并降低候选发布开销，13 文件 +2590/−285 行。**作者报告（未经评审验证）**：GB300 DEP4 decode step 32.39→32.04 ms、consumer selection 40.7→30.1 µs（1280 行/rank）；tie 行为可能与既有 radix/GVR 链不同、`split_row_logits_heads` 签名变更（不向后兼容）、需要 graph capture 前 warmup、CI 需 SM100/103 runner。记为 perf 级/作者报告/needs follow-up。

### 4.3 `#21619` [feat] 高精度末 N token 被逐出后复用混合精度前缀（feature，opt-in）

`KvCacheConfig.mixed_precision_reuse` 三模式（`default` 复用至末块边界、`recompute` 只复用到缺失 token 之前并重算、`strict` 仅在高精度副本完整处复用），两个新字段默认 False；覆盖 dense/DSA/DSv4，与 cold-page first/last-N 混合精度窗口配合。**评审发现**：PR 描述的「跨 draft 层恢复 + 移除 restore kernel」方向与最终 diff 不一致（diff 不含这些移除），记为描述级/设计待确认；QA 部分测试未列入 test-db（needs follow-up）。

### 4.4 `#21935` [perf] KVCM v2 面向 agentic 流量（perf，强化第 8 条）

三项：① 降 claim/release/prefix match 主机开销（`cuEventQuery` 5.41M→166K；kvbench 每请求 7.13→1.89 ms；正式 3600 s c224 吞吐 +7.6%、TTFT p90 8.4→3.7 s，单次运行）；② in-forward snapshot 批量 restore（ctx_bench step −6.0%；32 行 staging c240 吞吐 +1.9%、TTFT p90 −34%，单次运行）；③ 冷层（host/disk）按 cold typical step 定尺寸（block hit ratio 0.9327→0.9444）。多数为单次运行、捆绑多改动，记为作者报告/perf 级。

### 4.5 其他 feature/fix/perf（官方一手）

- `#22078` [fix] 在 SM100/SM103 启用 FP4 MLA（此前 SM107-only 门拒绝 Triton 生成路径），并保持 Kimi KV 精度策略不被按层权重排除覆盖；拆分 context workspace 预算（拦截固定 KV 开销）；Triton 生成保留 KV 长度/页有效掩码。
- `#20711` [refactor] **BREAKING** 简化 `ModelLoader.load()`（load() 行数 −61%、类 −36%），helper 拆到 `checkpoints/` 新文件；移除 `BaseCheckpointLoader.open_weight_session()`（非官方 API）。属破坏性重构，升级受影响方须核对自定义 CheckpointLoader。
- `#22175` [feat] Kimi K3 SM107 modeling_v2 运行时支持（RuntimeHooks、KVCM V2/hybrid manager 扩展、ConversationRouter opt-in 等；默认不变，含 3 项修复：disagg role 透传、FallbackFmha 缓存前缀 K/V、DFlash/DSpark slot 按 `num_seq_slots` 定尺寸）。
- `#21738` [feat] KVCM V2 上宽 beam search 的 opt-in cascade attention；`#21887` `cuda_graph_config.sampling_variants` 选择捕获的采样变体；`#20711`/`#384ad7fa` modeling_v2 移出 `_torch/_experimental`；`#21179` GLM-5.3-Flash FP8 on Hopper 性能修复；`#22337` 降 MLA prefill 中间 KV 存储；`#6529f081` 跨 head 共享 RoPE cos/sin 缩 QK-norm+RoPE kernel；`#6cc9dd04` 见 4.1。
- **infra/waive/test 约 45 条**：大量 QA CI waive/unwaive（`#22432`/`#22445` 等）、post-merge waive、test-db 清理、自动更新 test durations、release/1.3 合回（`#21944`）——按 infra 级记录，不逐条展开。

---

## 5. llama.cpp：2026-11-17 提交（35 条）—— Qwen3-Next GDN scale 漏接(#32393) + FP8 qwen4exp 转换 5000×(#32476/#32483) + SYCL mask OOB(#31536) + AMD stream_k(#30022, NVFP4 +85%) + K/V 读序 L2(#32407, 1.41×) + 多后端/UI

### 5.1 `#32393` / `#32476` / `#32483` 量化 scale 静默错误三连（fix，强化第 6 条）——见第 1.1 节

### 5.2 `#31536` [SYCL] 修 MKL FA mask broadcast 越界（fix，强化第 19 条）——见第 1.1 节

### 5.3 `#30022` ggml-cuda/hip AMD GCN stream_k + MMQ 调优（perf，硬件特定）

为 AMD GCN 架构调 stream_k 算法并在特定配置让 MMQ 走 stream_k。逐量化 pp512 微基准（3 次均值）：**NVFP4 最高 +84.97%（ub64）**、IQ2_XS +52.61%（ub16）、Q4_K +62.98%（ub64）、Q8_0 +44.87%（ub128）等；ub512/pp2048 增益收敛到约 +2%（Q8_0 仍 +18–20%）。属 GCN/HIP 特定、微基准口径，未给 PPL/精度对照。

### 5.4 `#32407` 旋转 K/V 读顺序提升 L2 命中（perf）

stream-K 下不同 CUDA block 同时读同一 K/V cache 的不同 tile，降低 L2 复用。FP16 MMA flash attention 旋转 K/V tile 读序（当前仅限 multi-stage kernel，single-stage 有回退）。RTX 5090 Qwen3.8-27B Q4_K_M prefill：131,072 token **1.41×**（1290→1822 t/s）、65,536 1.25×、32,768 1.07×，≤16K 基本持平。

### 5.5 其他要点（标题级 / 描述级）

- `#8777c0c8` CUDA few-row MMQ for MUL_MAT_ID；`#10152bc4` 所有 Pascal GPU 禁量化 matvec 融合；`#8b8ca4c9` F8_E4M3 不用 ldexpf 解到 fp32；`#a49db739` 仅对已启动 kernel 抬 shared mem 上限；`#ebc9fb93` 整行 pool_1d 用 block reduction；`#9048bad8` F32 kernel 用 implicit GEMM conv2d。
- 多后端：`#31002`/`#31003` OpenCL A7 Q5_K/Q8_0 non-MoE binary kernel；`#dde9de82` SYCL FA mkl mask（见 5.2）；`#8f647f1f` AMD Windows 驱动 bug 下禁 Vulkan flip FA shader；`#140a2798`/`#c145db27` WebGPU timestep embedding / dequant 用 f32 避 FTZ；`#e0a9118f` Metal implicit GEMM CONV_2D；`#9214bdb5` SYCL bin-bcast 快速除法；`#eada89c3` Nix SYCL 包。
- 模型/TTS/UI：`#32427` tiny-voca；`#307e3f15` kokoro TTS；`#8c16babb` Qwen3-TTS ICL/reference mode；`#d92ea9bb` Minimax-Music3；`#8d1599fc` kokoro G2P tts_dict；`#744c68cd` 从 CI 发布 `@ggml-org/llama-ui` 0.4.0；`#67cf8967` Llama Chat 迁 SvelteKit 3；`#7f106b54` UI 设计系统；`#32469`/`#513ad7ae` 默认关实时 markdown 预览/conversation tabs。
- KV/调度：`#08a9b358` 序列编辑保留 k-pool layout prefix；`#52975342` slot_sched 修 lora/recurrent；`#955c4e8e` grammar + ignore_eos 返回 400；nightly b12480–b12485（窗口内，prerelease）。

---

## 6. 累计结论延续与新增（继承 1–20，本期新增 21）

**累计结论（1–20 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性；③ 默认值成为主要风险载体；④ 异构/量化后端静默错误为高频（本期 vLLM #68014 HiSparse FP8 读错 KV 行、llama.cpp #31536 SYCL mask OOB）；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务（本期 SGLang #50740 草稿 ring × 前缀命中接受率、#50059 SWA 链式 miss、vLLM #64109 草稿尾 KV）；⑥ 量化路径静默错误（scale/布局漏接，本期 llama.cpp #32393 GDN scale、#32476/#32483 FP8 PLE 表 5000×）；⑦ 专家张量/路由归属与分页漏接静默错误；⑧ KV 卸载/HiCache/PD 健壮性 + 内存/缓存管理接口重构（本期 vLLM #68014/#67489、SGLang #50712/#50059、TRT #21935）；⑨ 长上下文 int32/int64 溢出静默错误（本期 SGLang #50870 14,563 token 读错、vLLM #67489 split-KV buffer OOM 邻接）；⑩ MXFP8 KV 正确性簇；⑪ 新旗舰模型 CP/KV/性能成为一等公民（本期 SGLang #38344 K3 CP、#6321f714 K3.1 Day-0、vLLM Kimi-K3.1-Pro recipe 簇）；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径（本期 SGLang #50719 Hopper fp8 MQA logits）；⑬ FP8/低比特 KV cache 跨引擎标准化落地；⑭ 投机解码验证路径跨引擎密集硬化（本期 vLLM #64109/#67724、SGLang #50740）；⑮ 跨后端可移植性成为显式架构主线（本期 vLLM #67540 gfx942/#64375 gfx1151/#61646 RDNA、SGLang #48065/#5cefe8de gfx1250/#9e14a219 NPU、llama.cpp 多后端簇）；⑯ 新旗舰模型进入「默认性能化」阶段（本期 vLLM #67741 默认 gvr_2、SGLang #50740 ring 默认开）；⑰ 推理服务前端 Rust 化；⑱ 推理服务默认安全加固（本期 vLLM #67765/#67314、SGLang #49290）；⑲ 非量化的常规优化路径静默错误密度不亚于量化路径（本期 vLLM #68014 KV 布局、SGLang #50717 权重时序、llama.cpp #31536 索引越界）。

**新增第 21 条（在线权重热更新的时序竞态成为静默正确性新子项）**：SGLang #50717 表明，控制面（权重写入）与在途计算流的同步缺失可产出「用错权重但 token/logprob/KV 完整保留、无报错」的静默错误，GB200 实测 8/8 次更新后首 token 用了新权重。研判（观点）：本追踪此前静默正确性主线集中在数据面（KV/量化/索引/整型溢出），第 21 条把它扩展到权重生命周期与调度时序——**凡支持 in-place 权重热更、RLHF 在线更新、权重回滚的服务，都须把「写入前等在途 forward」作为强制不变式，并审计所有权重写入入口**，否则精度事故会表现为「结果看似正常」而非显式崩溃。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 静默正确性复核（量化 scale 漏接 + KV 布局）**：llama.cpp `#32393`（Qwen3-Next GDN 输出投影缺 ssm_out.scale）、`#32476`/`#32483`（FP8 qwen4exp PLE 表 scale 丢失约 5000×）、vLLM `#68014`（HiSparse FP8 stride 271 当 64 页读、GSM8K 崩溃→0.9174）、SGLang `#50870`（Triton sparse MLA 32 位 token 溢出、14,563 token 读错；须同步升级 ROCm/aiter#7952）；部署 NVFP4/FP8 + HiSparse/DSA 前核验 scale 是否前向到算子、KV 页 stride/布局是否与后端一致、token 偏移是否 64 位。
- **P0 — 在线权重热更新时序（第 21 条）**：SGLang `#50717`——凡用 `pause_generation(in_place)` 后更新权重，确认所用版本含「7 个写入入口前等在途 forward」修复；自研服务须审计所有权重写入路径的 stream 同步。
- **P0 — OOM 导致的「静默 accuracy 0.000」**：vLLM `#67489`（FA3 MLA 超 CUDA graph 尺寸 batch 传 num_splits=0 → 无用 1.2–6.5 GiB buffer → OOM），nightly 评测因此记 0 分；升级后复核可扩展 KV 默认开启下的 decode 大 batch。
- **P1 — KV 容量/前缀/草稿 KV 权衡**：SGLang `#50740`（草稿 ring +19.7% KV 容量但命中后接受长度 −7.5~17.7%，需按命中率场景评估）、`#50059`（SWA 多轮链式 miss 98→25）、vLLM `#64109`（草稿尾 KV）；PD 部署关注 `#50712`（retraction 挂起但 /health 200）。
- **P1 — 跨硬件性能（口径特定）**：vLLM `#67540`（gfx942 TP8 monokernel 2.16× 但权重 +35 GiB、启动 8.5→24.5 min）、`#67741`（SM100 gvr_2 +2.4% E2E@128k 但短上下文回退、warmup +24 s）、`#61646`（RDNA SWA 128K 8.7×）、`#64375`（gfx1151 W4A16 −18.4% 但权重 +6.25%）；llama.cpp `#30022`（GCN NVFP4 +85%）、`#32407`（RTX 5090 131K prefill 1.41×）。关注硬件/后端特定性与吞吐/延迟/内存权衡。
- **P1 — 破坏性/ opt-in 变更**：TRT-LLM `#20711`（BREAKING ModelLoader.load() 重构、移除 `open_weight_session()`）、`#21619`（混合精度前缀复用 opt-in、描述与设计 diff 有出入）、`#22361`（CSA2 indexer 签名变更 + tie 行为差异）、`#22078`（FP4 MLA 准入扩展）。

---

## 7. 社区与视频资源（HN Algolia，2026-11-17 epoch 窗口 1794873600–1794959999）

> 以下均为社区/Show HN，非四家引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。引擎关键词检索中 `TensorRT-LLM`/`speculative decoding` **0 命中**；`KV cache` 1 条为无关摄影暗房目录；`SGLang`/`inference engine` 各 1 条即下述 TrnRT；`vLLM` 38 条多为泛 AI 资讯与模糊「LLM」匹配；`llama.cpp` 1 条为下述 tensward 博客。

- **TrnRT（AWS Trainium 开源推理引擎）**：「Show HN: TrnRT – An open-source LLM inference engine for AWS Trainium」（**43 分**，2026-11-17T16:01），URL `github.com/RunanywhereAI/trnrt`。已核验仓库存在、公开、**1612 stars**、描述「The open LLM inference engine for AWS Trainium: NKI megakernels, 1M-token context」（GitHub API，HTTP 200）。属本追踪第 15 条「跨后端可移植性」主题的社区新成员，但**非 vLLM/SGLang/TRT-LLM/llama.cpp 官方**，性能/能力数字未经一手核验。**二手线索/未核验**。
- **llama.cpp 大页环境变量提速**：「One environment variable made our llama.cpp server up to 21% faster」（**1 分**，2026-11-17T16:54），URL `tensward.com/blog/huge-pages-llama-cpp`（厂商博客，称大页配置使 llama.cpp server 提速至多 21%）。非官方一手、无独立复现。**二手线索/未核验**。
- 其余 HN 命中（如「Not Everything Needs an LLM」83 分、`Spaceship IDE / Cerebras` 5 分等）为泛 AI 资讯，与推理引擎实现无直接关系，未列入。本期**无引擎相关的可确认官方视频**；Reddit / YouTube / X 未采集。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM **91** / SGLang **99** / TRT-LLM **73** / llama.cpp **35** 合计 **298** 条（2026-11-17 UTC 窗口；周二，工作日正常量；SGLang 99/TRT-LLM 73 处近期高位，vLLM 91 单页取全无截断）。与上一期（2026-11-17 档案覆盖 11-16）连续，无空档。
- **版本核验**：releases API 确认——**窗口内无新正式版本**：vLLM 仍为 v0.34.0（2026-11-16T23:22:00Z，非 prerelease，为上一期头号事件）、SGLang v0.5.23（11-02）、TRT-LLM v1.4.0rc3（11-06，prerelease）、llama.cpp 正式版 v1.0.0（11-13）；llama.cpp 窗口内推进 nightly b12480–b12485（prerelease）。发布说明为精选 highlights，按本追踪方法论不作为「修复已进版」依据。
- **PR 正文来源**：**34 个关键 PR 正文经官方 issues API 抓取**（markdown 官方一手；vLLM 12 + SGLang 9 + TRT-LLM 7 + llama.cpp 6），配额充足、无 IncompleteRead、无降级为标题级；所有 PR 号均在窗口 commits API 中确认为已合入 main 的提交。其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度。
- **描述级/未核验标注**：vLLM `#67741`/`#67540`/`#61646`/`#64375` 性能数字为作者在 PR 正文报告的微基准/E2E（含硬件特定、E2E 口径与权衡，均并列说明）；`#67338` 官方声明 serving 行为不变（防御性重构）；SGLang `#50740` 明示前缀命中后接受长度下降的代价、`#50712`/`#50059`/`#50870`/`#44447` CI 未全绿、`#38344` 为 proxy 测量且 CI Base ✗、`#44447`/`#6321f714` 正文 Accuracy/Speed 段为空（描述级）；TRT-LLM `#22258` bit-identical 声称未经评审验证、`#22361` 作者数字未独立核验且签名不兼容、`#21619` 描述与设计方向 diff 有出入（needs follow-up）、`#21935` 多为单次运行捆绑多项、大量 infra/waive；llama.cpp `#30022` 为逐量化微基准无 PPL 对照、`#32476` 完整转换未测（dry-run）、`#32483` 未在真机模型验证（仅生成的 qwen4exp gguf）。GC 冻结簇与 Kimi-K3.1-Pro recipe 簇按标题/描述级记录。
- **社区信号**：HN Algolia 引擎官方更新类故事无；TrnRT（43 分，仓库已核验存在、1612 stars）与 tensward 大页博客（1 分）均按二手线索/未核验记录；无引擎相关的可确认官方视频。
- **技能检索（find-skills / agent-browser）**：find-skills 经 SkillHub 语义检索（EN+ZH 各 1 组，含「LLM inference GitHub commit monitoring daily hotspot digest」「大模型推理 官方提交采集 热点归档 仓库同步」）返回 `github-ai-trends`（GitHub AI 热门项目趋势榜）、`arxiv-daily-digest`（论文精选）、`full-auto-info-collector`（RSS/GitHub 热门/关键词聚合到飞书日报）三个候选，均为趋势榜/论文/泛聚合类，与「官方一手提交 API + 证据分级归档 + 指定目录归档 + git 提交推送」工作流不匹配，**未安装**；agent-browser 因四家数据源（commits/releases/issues API + PR 页面）与 HN Algolia 均可直接 HTTP 访问而未调用。结论与历次一致。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级/CI 未全绿/作者报告/needs follow-up/微基准」。
