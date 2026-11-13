# 大模型推理优化每日热点档案

> 档案日期：2026-11-13（本自动化于该日执行）
> 关注窗口：2026-11-12（周四）UTC 全天的官方提交 / 发布 / PR
> 采集方式：GitHub 官方 commits API（`since/until` UTC 窗口）+ releases API + 关键 PR 正文（vLLM/SGLang/TRT-LLM/llama.cpp 经 pulls API 抓取 markdown 官方一手；全部 42 个详述 PR 均经官方 pulls API 一手获取，无 API 配额耗尽、未走 HTML 页面抓取）+ HN Algolia（HTTPS，numericFilters 按 2026-11-12 epoch 窗口 1794441600–1794528000）
> 数据质量：四家提交 API 均 HTTP 200；vLLM 94 / SGLang 89 / TRT-LLM 39 / llama.cpp 39 合计 261 条；四家单页均未触 per_page=100 上限（无截断）；releases API 确认窗口内**无新正式版本**（vLLM 仍为 v0.33.0rc3、SGLang v0.5.23、TRT-LLM v1.4.0rc3；llama.cpp 仍为 v0.9.0 正式版，nightly 推进 `b12323`–`b12336`，均 prerelease）；42 个关键 PR 正文经 pulls API（markdown 官方一手）抓取（API 配额充足、未触发 60/h 限制）；HN Algolia 引擎关键词高信号弱（无四家官方更新类故事，仅社区/Show HN，二手线索/未核验）。本窗口为周四，提交量较上一窗口（11-11：87/98/49/31）回升至 94/89/39/39，属正常工作日波动。

## 1. 版本状态与本期头号事件：vLLM GLM-5.3「默认性能化」(latency_specialized/MTP row-pruning 默认开) + MegaMoE 跨 SM90 落地 + SGLang DFlash 草稿 KV ring 跨 PD(Mooncake) 与 DSA 性能(bf16 logits/Prims-TS/native-head FlashMLA) + TRT-LLM 量化静默正确性铁证(#22106/#21346) + llama.cpp 服务端安全加固(SSRF/端口扫描)

**提交计数（2026-11-12 UTC，均 HTTP 200；2026-11-12 为周四，量较周三回升；vLLM 94 / SGLang 89 / TRT-LLM 39 / llama.cpp 39）**：
- **vLLM 94** / **SGLang 89** / **TensorRT-LLM 39** / **llama.cpp 39**。

**发布状态（窗口内无新正式版本）**：
- **vLLM**：仍为 **v0.33.0（tag `v0.33.0rc3`，2026-10-31T07:15:45Z）**；11-12 全部 94 条为 `main` 常规提交，无新 tag。
- **SGLang**：仍为 **v0.5.23（2026-10-30T00:13:59Z）**；11-12 含 DFlash 草稿 KV ring 跨 PD、DSA 性能栈、MoE 量化融合、NPU/跨硬件后端。
- **TensorRT-LLM**：仍为 **v1.4.0rc3（prerelease，2026-11-03T11:21:00Z）**；11-12 含 MXFP4 FP8 MoE 量化修复、GLM-5.2 NVFP4 性能、资源释放修复、混合投机 replay state 入 KV region。
- **llama.cpp**：仍为 **v0.9.0（2026-11-04T13:08:06Z 正式发布）**；11-12 推进 nightly `b12323`（07:44Z）–`b12336`（11:15Z，prerelease），无新正式版本。

**本期头号事件（GLM-5.3 默认性能化 + MegaMoE 跨 SM90 + DFlash 草稿 KV ring 跨 PD + DSA 性能 + 量化静默正确性 + llama.cpp 安全加固）**：
1. **vLLM 把 GLM-5.3 的「已有 specialized 路径」默认开启（正向默认，强化/补充第 3、11 条）**：`#67099`（batch<64 时 `latency_specialized` 默认开，E2E 吞吐 +27.0%、aime25 0.6667→0.7333）、`#67104`（把已存在的 specialized MTP row-pruning 路径在 CUDA Graph capture 前绑定 `last_token_indices` 默认生效，E2E 吞吐 +4.2%、aime25 0.7333→0.7667）。此前 `latency_specialized` 是 opt-in，本期反向——把验证过的性能路径默认化。这是「默认值成为主要风险载体」主线（第 3 条）的反向印证：默认值既可以是风险（漏开的量化/后端），也可以是被低估的免费收益。
2. **量化 MoE / MegaMoE 跨硬件落地（强化第 12、13 条）**：vLLM `#64061`（DeepGEMM MegaMoE 上 SM90/H200 + MXFP4 routed experts，DSv4.1-Flash 单节点 DP8+EP8 输出吞吐较默认高 17–26%、较 deepep_v2 高 12–19%；2 节点 AgentX ITL p50 −10~11%、TTFT p50 −12~17%）、`#62603`（SM90 block-FP8 MegaMoE via FlashInfer MoE-EP，GLM-5.3-Flash 8×H200 解码 +11~18% 输出吞吐、TTFT p50 −14~54%）。此前 MegaMoE 仅 SM100，本期把 NVFP4/FP8 量化 MoE 的融合 mega-kernel 推到 Hopper，量化后端成为新旗舰 MoE 主路径收口。
3. **SGLang DFlash/DFLASH 草稿 KV ring 跨 PD 落地（强化第 14 条投机解码验证路径硬化）**：`#49664`（Kimi-K3.1 DFlash2 sliding-window MLA draft 的 KV 存到每请求 ring，K3.1 同址部署 +15.0% target KV）、`#49665`（该 ring 经 Mooncake 在 PD 间从 prefill 传到 decode，PD 拿到同等的 +15.0% target KV 收益）、`#50005`（Kimi-K3.1 DSA 层在 PD decode 实例上开 HiSparse + DFLASH 至 block size 8）。草稿 KV 的复用从「同实例 per-token 池」升级为「跨 PD 的 per-request ring」，是 DFlash/DFLASH 服务的资源正确性主线。
4. **SGLang DSA 性能栈（强化第 11 条新旗舰模型一等公民）**：`#49102`（DSA prefill indexer 用 bf16 MQA logits 替代 fp32，GSM8K 97.42→97.80%、AIME26 99.58%）、`#48602`（opt-in FlashInfer Prims-TS sparse MLA backend，decode 快于 trtllm-gen/FlashMLA 且 cache 更省字节，需 FlashInfer 0.7.4）、`#49980`（DSA 在 SM100 跑真实 head count 的 native-head FlashMLA + 预计算 DCP decode schedule，Kimi-K3.1 的 96 heads 不再 pad 到 64/128）。
5. **量化静默正确性铁证延续（强化第 4、6、13 条）**：TRT-LLM `#22106`（W4A8_MXFP4_FP8 用 raw FC1 scale 而非 FC1/FC2 input-scale ratio，否则 4 个默认 SwiGLU DEP/NVLINK 用例约 97% mismatch，5 个 DEP 用例 GB200 全过）、`#21346`（Qwen3.8-Flash-Next 的 FP8 PLE 表在 attention DP + tp>1 下因 PyTorch 无 FP8 `masked_fill_` 在首个 forward 崩，改用 uint8 view 位精确）；SGLang `#49851`（Kimi-K3.1-Pro MXFP8 KDA `b_proj`/`f_a_proj` 加载在 TP>1 失败、会拿 raw fp8 跑融合 GEMM，修后 GSM8K 0.9742）；vLLM `#66274`（Triton block-FP8 fused MoE 在 Triton 3.9 因 scalar `b_scale` 索引被拒，广播修复）。scale/布局漏接仍是最高频静默错误源。
6. **llama.cpp 服务端安全加固（新增部署安全主线）**：`#32115`（CORS proxy 与输入 media URLs 默认禁止达本地/内网地址，防 SSRF/内网逃逸，需 `--media-allow-private-urls` 才放开）、`#32120`（禁止浏览器端口扫描发现 child instance）、`#32113`（非回环地址显式 opt-in tools）。此前 llama.cpp 把本地服务默认信任，本期把「可达内网」显式关掉，反映本地推理直接跑在个人机的部署现实风险。

---

## 2. vLLM：2026-11-12 提交（94 条）—— GLM-5.3 默认性能化(#67099/#67104) + MegaMoE 跨 SM90(#64061/#62603) + Mamba2 ReplaySSM Triton(#66318) + KV Offload 健壮性(#66391/#67152/#67080/#65448) + ROCm mxfp4_mla(#60321)

### 2.1 `#67099` [GLM5.3 Perf] Enable `latency_specialized` by default, 27.0% E2E Throughput improvement（GLM-5.3 latency_specialized 默认开，perf，强化第 3、11 条）
- **主题**：继 #63102 之后，把 GLM-5.3 的 `latency_specialized` 解码设为默认（batch size < 64 时启用）。
- **内容详情**：TP4、MTP5、prefix caching 配置下，E2E 基准输出吞吐 +27.0%；aime25 exact_match 0.6667（main）→ 0.7333（now）。bench：32 prompts、output 32768 token、concurrency 1，输出 620.58 tok/s。精度波动在散布内。
- **应用场景**：GLM-5.3-Flash 在 GB300 的低延迟默认解码。
- **研判**：【事实】PR 给 +27.0% E2E、aime25 0.6667→0.7333 与 bench 对照（620.58 tok/s）。【观点】属「把已验证 specialized 路径默认化」（第 3 条反向、第 11 条），正向默认；记为官方一手（含 bench + aime25 对照）。**证据等级**官方一手（含对照）。
- **来源**：vllm-project/vllm #67099（合并于 2026-11-12T16:44:29Z）。

### 2.2 `#67104` [GLM 5.3 Perf] Skip unused MTP post-attention rows, 4.2% E2E throughput improvement（MTP post-attention 行裁剪默认绑定，perf，强化第 3、11 条）
- **主题**：在 CUDA Graph capture 前绑定 speculator 的 `last_token_indices`，使已有的 specialized MTP row-pruning 路径默认生效（先对 4 行做 select 再做 norm/MoE/final norm，而非对 24 行全跑）。
- **内容详情**：TP4、MTP5 配置；E2E 吞吐 +4.2%；aime25 0.7333（now）→ 0.7667（after）。bench 同 #67099 配置，输出吞吐对照在散布内。
- **应用场景**：GLM-5.3-Flash MTP 解码的计算量裁剪。
- **研判**：【事实】PR 给 +4.2% E2E、aime25 0.7333→0.7667。【观点】属「MTP 专用路径默认化」（第 3、11 条），纯 perf、精度在散布内；记为官方一手（含对照）。**证据等级**官方一手（含对照）。
- **来源**：vllm-project/vllm #67104（合并于 2026-11-12T16:45:41Z）。

### 2.3 `#64061` [Kernel][Perf] Run DeepGEMM Mega MoE on SM90 with MXFP4 experts（DeepGEMM MegaMoE 上 SM90 + MXFP4，perf，强化第 12、13 条）
- **主题**：在 SM90（H200）上启用 `deep_gemm_mega_moe` MoE 后端（MXFP4 routed experts），用 DeepGEMM 新 SM90 FP8×FP4 Mega MoE kernel，把 dispatch、L1 GEMM、带 FP8 量化的 SwiGLU、L2 GEMM、combine 融成一个 persistent kernel over NVLink 对称内存。
- **内容详情**：依赖 vllm-project/DeepGEMM#44（SM90 用例在该 PR 落地前 skip）。DSv4.1-Flash 单节点 DP8+EP8：输出吞吐较默认高 17–26%、较 deepep_v2 高 12–19%（concurrency 1–1024）；2 节点两副本：较默认高 16–27%、较 deepep_v2 高 10–16%；2 节点 P/D AgentX：ITL p50 −10~11%、TTFT p50 −12~17%；贪心生成与默认后端内容一致。SM90 激活按每 128 通道 1 个 FP32 scale 暂存。测试 120 passed / 12 skipped（skips 为 SM100-only NVFP4）。
- **应用场景**：DeepSeek-V4.1-Flash 在 Hopper 的量化 MegaMoE 融合解码。
- **研判**：【事实】PR 给 DSv4.1-Flash 17–27% 吞吐对照、120 passed、bit-exact 验证。【观点】属「量化 MegaMoE 跨 SM90 落地」（第 12、13 条），融合 mega-kernel 推到 Hopper；记为官方一手（含多节点对照）。**证据等级**官方一手（含对照）。
- **来源**：vllm-project/vllm #64061（合并于 2026-11-12T00:28:06Z）。

### 2.4 `#62603` [Perf][MoE] Enable SM90 block-FP8 MegaMoE via FlashInfer MoE-EP（SM90 block-FP8 MegaMoE via FlashInfer，perf，强化第 12、13 条）
- **主题**：用 FlashInfer 0.7.2 的 SM90 FP8 MegaMoE（经现有 `--moe-backend flashinfer_moe_ep_cutedsl`），把 block-FP8 MoE checkpoint 在 Hopper 上跑起来（融合 dispatch + 两个 expert GEMM + combine）。
- **内容详情**：GLM-5.3-Flash 8×H200（DP8/EP8，128K/512，MTP3）解码 +11~18% 输出吞吐、TPOT p50 −6~14%（concurrency 8–192）；4×H200（TP2/DP2/EP4，MTP3，ISL 2048/OSL 1）prefill +23% 饱和吞吐、TTFT p50 −14~54%（15–21 req/s）；单 MoE 层 EP8：tokens/rank 8 时 −7%、8192 时 −48%。GSM8K 两配置可比。后端 opt-in，默认不变。FlashInfer 0.7.3：CPU 32 passed、H200 4 passed。
- **应用场景**：GLM-5.3-Flash 在 Hopper 的 block-FP8 MegaMoE。
- **研判**：【事实】PR 给 GLM-5.3-Flash H200 +11~18% 解码、+23% prefill 与 GSM8K 可比。【观点】属「量化 MegaMoE 跨 SM90」（第 12、13 条），与 #64061 并列把 MegaMoE 推到 Hopper；记为官方一手（含 H200 对照）。**证据等级**官方一手（含对照）。
- **来源**：vllm-project/vllm #62603（合并于 2026-11-12T09:07:35Z）。

### 2.5 `#66318` [Kernel][Mamba] Support Mamba2 ReplaySSM speculative decoding on the Triton backend（Mamba2 ReplaySSM 投机解码上 Triton，强化第 14 条）
- **主题**：Mamba2 ReplaySSM 投机解码此前仅 FlashInfer（CUDA）。新增 Triton verify 与 materialize kernel（同 ring 契约），使现有 runner 拥有的 ReplaySSM tracker 同时驱动两种后端；并加 MI355 镜像测试（带 MTP）。
- **内容详情**：新增 `selective_state_update_replayssm_spec`（变长 verify 窗口、paged slots、pad rows）与 `replayssm_materialize`（一次 launch 覆盖所有层对齐 prefix snapshots）；`VllmConfig.use_replayssm_trackers` 替代 `use_flashinfer_replayssm` 驱动 V1/V2 runner、metadata builder、block copies、warmup；校验接受同 buffer/PP/microbatching 限制、拒绝随机舍入。MI350X（gfx950）验证：ReplaySSM+MTP5 连续 10 run，GSM8K 0.9318–0.9393（期望 0.93）、mean accept length 达标。
- **应用场景**：Nemotron-3-Super-120B 等 Mamba2 ReplaySSM 在 Triton/AMD 后端的投机解码。
- **研判**：【事实】PR 给 MI355 GSM8K 0.93+ 与 accept length 验证、新增 AMD 镜像。【观点】属「混合模型投机解码跨后端正确性」（第 14 条），Triton 后端补齐；记为官方一手（含正确性对照）。**证据等级**官方一手（含测试）。
- **来源**：vllm-project/vllm #66318（合并于 2026-11-12T15:47:46Z）。

### 2.6 `#60321` [ROCm][Attention] Add an mxfp4_mla KV cache for rope-free sparse MLA（ROCm mxfp4_mla KV cache，强化第 8、13 条）
- **主题**：为 ROCm AITER sparse MLA 后端（rope-free/NoPE latent，如 GLM-5.3-Flash）加 `mxfp4_mla` KV cache dtype：每 512 宽 latent 存 256 字节 E2M1 + 16 字节 E8M0 scale，272 字节/token vs bf16 1024 字节，即 3.76× token/GiB。
- **内容详情**：从 #59739 拆出的首个小 PR，含 cache 格式、write path、ragged 与 split-K decode 的 Triton read path（#58584）；纯 Triton/Python，无 HIP/C++。声称：KV 容量 3.76×、pack 后注意力 bit-identical、其他 dtype 不变（仅按 element type 编译期分支）、同 bf16 速度不宣称提速（读到 bf16 再 dot）。端到端 GLM-5.3-Flash TP4：bf16 322.5 tok/s vs mxfp4_mla 323.9 tok/s（中位 TTFT 1596 vs 1608ms、TPOT 9.8ms）。MI355X 单元测试 16 passed。
- **应用场景**：ROCm 上 GLM-5.3-Flash 等 rope-free sparse MLA 的 KV 容量压缩。
- **研判**：【事实】PR 给 3.76× token/GiB、bit-identical 声明与 MI355X 16 passed。【观点】属「低比特 KV cache 跨引擎标准化」（第 13 条）+ KV 容量（第 8 条），容量收益、无算术提速需谨慎；记为官方一手（含声明 + 单测）。**证据等级**官方一手（声明级 + 单测）。
- **来源**：vllm-project/vllm #60321（合并于 2026-11-12T04:46:23Z）。

### 2.7 `#66391` [Perf][KV Offload] Split the shared Tiering region and skip the scheduler pre-fault（KV Offload 共享 Tiering region 拆分 + 去 scheduler 预错，perf，强化第 8 条）
- **主题**：Tiering 每引擎一个 `/dev/shm` 共享 CPU region，大尺寸下 setup 慢（worker 预错序列化 page-cache 插入、scheduler 单线程再映射预错全部页）。拆分到 `REGION_SEGMENTS` 文件（如 #59546 的 CPUOffloadingSpec），scheduler 侧映射但不预错。
- **内容详情**：GLM-5.3 尺寸行（8×3.8MB slot），单 H100 8 worker 共享一 GPU、P2P tier：512 GiB 时 CPU tier setup 167–175s→59s；1250 GiB 时 397–460s→135–138s（worker 预错 255–317s→55s，scheduler 预错 59–140s→0）。
- **应用场景**：vLLM KV Offload 大容量 CPU tier 的启动 setup 提速。
- **研判**：【事实】PR 给 512/1250 GiB 的 setup 59s/135–138s 对照（前 167–175s/397–460s）。【观点】属「KV 卸载大 tier 启动提速」（第 8 条），纯 setup 性能；记为官方一手（含计时对照）。**证据等级**官方一手（含计时）。
- **来源**：vllm-project/vllm #66391（合并于 2026-11-12T08:27:59Z）。

### 2.8 `#67152` [Bugfix][KV Offload] Clear partial-tail pending state on a required group miss（KV Offload partial-tail 缺块清 pending，fix，强化第 8 条）
- **主题**：partial-tail 边界需要多 group 的 KV，任一 required group 返回 MISS 使该边界不可用；lookup 当前保留早先 group 看到的 pending，若无其他边界可命中且无其他 pending 需等，则把请求拖延到下一调度轮而非重算缺前缀。
- **内容详情**：MISS 时清除被拒边界的 pending（其他可行边界与各 in-flight offload 不变）。回归覆盖 RETRY / HIT_PENDING 后 HIT 或 MISS：基线 2 failed/31 passed → 修复 33 passed；offloading scheduler 全 CPU 标记测试 37 passed。GB10 Qwen3.5-0.8B+FS：受影响请求 TTFT 60.3→23.7ms、E2E 90.2→53.1ms、8 请求 replay 总耗时 159.5→125（ms，中位数）。
- **应用场景**：vLLM KV Offload 多 group partial-tail 前缀重算正确性。
- **研判**：【事实】PR 给 TTFT 60.3→23.7ms、E2E 90.2→53.1ms 与 33 passed。【观点】属「KV 卸载前缀重算正确性」（第 8 条），MISS 不再拖延请求；记为官方一手（fix 级，含计时 + 测试）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：vllm-project/vllm #67152（合并于 2026-11-12T11:56:18Z）。

### 2.9 `#67080` [Bugfix][KV Connector] Skip Mooncake saves after failed existence checks（Mooncake 失败存在检查跳过 PUT，fix，强化第 8 条）
- **主题**：Mooncake cache save 的存在检查失败时，此前把失败 key 当 miss 发 PUT；现跳过可选 save 并报告准确失败计数。
- **内容详情**：`batch_is_exist` 命中 1/未命中 0/失败负码；两条 save 路径共享结果分类，任一 key 检查失败即跳过可选 save。修复前 focused regression 6 failed/3 passed（每个失败存在用例误发 PUT）→ 修复后 worker+scheduler 套件 362 passed。无吞吐/等价声明，测试用受控外部 SDK、无真实 RDMA。
- **应用场景**：Mooncake KV store connector 的失败容错。
- **研判**：【事实】PR 给修复前 6 failed → 修复后 362 passed。【观点】属「KV 连接失败容错」（第 8 条），避免失败 key 误 PUT；记为官方一手（fix 级，含测试）。**证据等级**官方一手（fix 级，含测试）。
- **来源**：vllm-project/vllm #67080（合并于 2026-11-12T10:54:38Z）。

### 2.10 `#66197` [Bugfix] Exclude FlashInfer autotune peak from KV cache sizing（KV 容量排除 FlashInfer autotune 峰值，fix，强化第 3、8 条）
- **主题**：修复 #66193 的启动回退（由 #56492 引入）：extensible KV cache 的瞬时 reserve 把一次性的 FlashInfer autotune 峰值长期保留，挤占 KV cache；现排除该峰值但保留前后 tuning 的激活 reserve 与瞬时。
- **内容详情**：单 RTX 5090、Qwen3.8-27B-NVFP4、MTP3、gpu-memory-utilization 0.85：基线 reserve 3.83/ KV 1.51 GiB → 修复 1.79/3.56 GiB，ValueError→通过；输出 tok/s 66.1/33.7（2k/128）与 68.5/36.2（2k/1000+32k 共享前缀）。用 #66193 记录的既有测量，非本分支新跑。
- **应用场景**：extensible KV cache 在 autotune 开启下的容量峰值自愈。
- **研判**：【事实】PR 给 RTX 5090 启动 ValueError→通过、reserve 3.83→1.79 GiB。【观点】属「KV 容量默认值/峰值错误」（第 3、8 条），autotune 峰值不应计入常驻；记为官方一手（fix 级，含对照）。**证据等级**官方一手（fix 级，含对照）。
- **来源**：vllm-project/vllm #66197（合并于 2026-11-12T02:19:28Z）。

### 2.11 其他要点（标题级 / 描述级 / 性能级）
- `#67040` [Perf][GLM-5.3] DeepGEMM BF16 MegaMoE for unquantized checkpoints（GLM-5.3 非量化 DeepGEMM BF16 MegaMoE，perf 级，强化第 12 条）。
- `#67269` [Recipe] Enable Muse Spark MegaMoE and B300 specialized decode（Muse Spark MegaMoE + B300 specialized decode，recipe 级）。
- `#65521` [Perf][mHC] Use FlashInfer mega mHC for MHCFusedPostPreOp（GLM-5.3-Flash mHC 用 FlashInfer mega mHC，perf 级）。
- `#66036` [ROCm][DSv4/4.1] Enable Tensor scale FP8 KV Cache for Sparse MLA（ROCm sparse MLA 开 Tensor scale FP8 KV，perf/feature 级，强化第 13 条）。
- `#64560` [ROCm][Perf][MLA] Fuse decode W_UK absorb, q concat + fp8 quant + head pad via AITER（ROCm AITER 融合 MLA decode，perf 级）。
- `#54899` [ROCm][MLA][DCP] Pad the AITER MLA decode head count to a native tile（ROCm AITER MLA head pad，perf 级）。
- `#67226` [ROCm][RDNA] ROCM_RDNA_ATTN 绕开 LLVM VGPR 回退（RDNA decode kernel spill，fix 级）。
- `#67263` [CI][Quantization] Skip ModelOpt INT8_PB_WO routing test on ROCm（ROCm 跳 INT8_PB_WO 路由测试，ci 级）。
- `#66859` [MoE] Replace routed-scale bool flags with a RoutedScaleMode enum（MoE routed-scale 改枚举，refactor 级）。
- `#67225` [Bugfix][MoE] Skip DeepGEMM localizer cleanup on SM90（SM90 跳过 DeepGEMM localizer cleanup，fix 级）。
- `#65373` [Bugfix] Share DeepSeek V4.1 CPU KV across TP ranks（DSv4.1 CPU KV 跨 TP rank 共享，fix 级）。
- `#61872` [Bugfix][ROCm] Restore MoE weight padding dropped by #46009（ROCm 恢复被 #46009 丢的 MoE 权重 padding，fix 级）。
- `#66674` [KVConnector] Allocate the extensible KV cache exportable for EXPORTS_FD (LMCache MP)（extensible KV cache 对 LMCache MP 可导出，feature 级，强化第 8 条）。
- `#66523` [Bugfix] Extensible KV cache: release warmup's blocks before CUDA graph capture（KV 容量预热块在 CUDA graph capture 前释放，fix 级）。
- `#66704` [Misc] Warn when gpu_memory_utilization withholds memory from the extensible KV cache（gpu_memory_utilization 扣留内存时警告，fix 级，强化第 3 条）。
- `#66561` [Bugfix] Use max_split_size_mb:21 so freed 1-10 MiB tensors are reused（1-10 MiB 张量复用，fix 级）。
- `#65448` [KV Connector][Spec Decode] Support lookahead block hashes in remaining KV connectors（lookahead block hashes 覆盖 UMBP/HiSparse/MoRIIO 全部 connector，强化第 5、14 条）。
- `#62964` [Bugfix][Spec Decode] Keep FlashInfer CUDA graphs for drafters with their own head count（保留 drafter 自有 head count 的 FlashInfer CUDA graph，fix 级，强化第 14 条）。
- `#66063` [Spec Decode][PP] Run draft models as a single pipeline stage（draft 模型作单一 pipeline stage，perf/feature 级，强化第 14 条）。
- `#66910` [CI] Run the NIXL PD tests as pytest（NIXL PD 测试 pytest 化，ci 级）。
- `#67035` [Model] Remove TeleChat2 and TeleChat3（移除 TeleChat2/3，cleanup 级）。
- `#67205` [Model Loader] Load tensorized vLLM models into the other model implementation（tensorized vLLM 模型加载到另一实现，feature 级）。
- `#67042` [Dependency] Upgrade FlashInfer to 0.7.4（FlashInfer 升 0.7.4，dep 级，与 SGLang #49771 同步）。
- `#65685` [Model] Kimi K3.1 Pro (split 5/6: ROCm DCP, P/D, indexer top-k)（Kimi K3.1 Pro ROCm DCP/P-D/indexer，feature 级，强化第 11 条）。
- `#67212` [Bugfix] build with param.dtype when one is not specified（未指定时按 param.dtype 构建，fix 级）。
- `#67085` [Refactor] Remove dead code models（移除死代码模型，refactor 级）。
- `#65149` [Quantization][Fusion] Extend manual activation fusion to more NVFP4 linear backends（手动激活融合扩展到更多 NVFP4 linear 后端，perf 级，强化第 13 条）。

---

## 3. SGLang：2026-11-12 提交（89 条）—— DFlash 草稿 KV ring 跨 PD(Mooncake)(#49664/#49665) + DSA 性能(#49102/#48602/#49980) + MoE 量化融合(#48263/#48295) + NPU 跨硬件(#47934/#48031) + HiCache 健壮性(#50056)

### 3.1 `#49664` [Perf] dflash: Keep a sliding-window MLA draft's KV in a per-request ring（DFlash 草稿 KV 存每请求 ring，强化第 14 条）
- **主题**：Kimi-K3.1 DFlash2（全 sliding_attention、window 1024）的 draft KV 原本每 target slot 占一行、计入 target 的 per-token KV 预算；本 PR（#48414 的 Day-0 port）改用 per-request ring，K3.1 同址部署 +15.0% target KV。
- **内容详情**：基于 #49621/#49664 首提交的 Day-0 字节级移植，逐 commit 改动：去掉 Day-0 的 megakernel tail check、PD 下 ring 关、unified-memory 下 ring 关、ring 预算在 token sizing 前计费、ring 决策改为声明的 spec 字段、去掉 `SGLANG_OPT_DFLASH_RING_LOCS_FUSED`（两路径 bit-exact 故只留 fused launch）、DCP collapsed-rows 检查从 worker init 移到 resolution。
- **应用场景**：Kimi-K3.1 DFlash2 草稿 KV 的 per-request ring 复用。
- **研判**：【观点】属「DFlash 草稿 KV ring 复用」（第 14 条），+15.0% target KV 为 Day-0 实测；记为官方一手（feature/perf 级，移植自 Day-0 分支）。**证据等级**官方一手（feature 级，移植）。
- **来源**：sgl-project/sglang #49664（合并于 2026-11-12T03:09:15Z）。

### 3.2 `#49665` [Feat] dflash: Send the draft KV ring over PD disaggregation (Mooncake)（DFlash 草稿 KV ring 经 Mooncake 跨 PD，强化第 14 条）
- **主题**：在 #49664 基础上把 per-request ring 从 prefill 传到 decode（PD 拿到同 +15.0% target KV）。
- **内容详情**：scheduler 把 ring 交给 `PrefillBootstrapQueue`/`DecodePreallocQueue` → `setup_state_kv_args`（读入 payload），去掉 Day-0 的进程级全局 `set/get_draft_kv_ring`；Mooncake（`mooncake_tcp` 重写为 `mooncake`）下 ring 保持开，其他后端（NIXL/Mori/Ascend/fake）无 DRAFT_RING 路径、显式 log 关闭（这些后端显式 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1` 为错误）；`--disaggregation-decode-host-receive-threshold < 1` 时也关（decode host buffering 只收 SWA state）。KV checksum 路径同步。
- **应用场景**：Kimi-K3.1 DFlash2 草稿 KV 在 Mooncake PD 部署下的跨实例复用。
- **研判**：【观点】属「DFlash 草稿 KV ring 跨 PD」（第 14 条），PD 拿到同 +15.0% target KV；记为官方一手（feature 级，移植 + 接口清理）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #49665（合并于 2026-11-12T10:10:19Z）。

### 3.3 `#49102` [Perf] dsa: bf16 MQA logits for the prefill indexer（DSA prefill indexer 用 bf16 MQA logits，perf，强化第 11、13 条）
- **主题**：DSA indexer 的 ragged（prefill）路径原本用 DeepGEMM `fp8_mqa_logits`（fp32 logits）；改用其 legacy 入口的 bf16 logits（相同 FP8 query/keys），top-k 字节减半、chunk scratch 减半。
- **内容详情**：opt-in（CUDA，`SGLANG_DSA_INDEXER_BF16_LOGITS=1`），v2 ragged top-k 加 `score_t=bf16` 构建（JIT 模块 `mode{N}_bf16`）。精度：冷 16384-token prefill 的 top-2048 overlap 与 fp32 参考 0.9792 vs 0.9791；bf16 top-k 与 fp32 选出值相同。端到端 Kimi-K3.1 TP8/DCP8、2×4 GB300：GSM8K 97.80%（bf16）vs 97.42%（fp32）、AIME26 pass@1 99.58% vs 100.00%（1 个 AIME miss 为 100K token 截断）、100K 检索 24/24。
- **应用场景**：Kimi-K3.1 DSA prefill indexer 的 top-k 性能/带宽优化。
- **研判**：【事实】PR 给 bf16 vs fp32 的 top-2048 overlap 0.9791/0.9792、GSM8K 97.80%/97.42%、AIME26 99.58%。【观点】属「DSA indexer 低精度 top-k」（第 11、13 条），bf16 不损精度、省带宽；记为官方一手（perf 级，含精度对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：sgl-project/sglang #49102（合并于 2026-11-12T02:44:16Z）。

### 3.4 `#48602` [Perf] dsa: Add an opt-in FlashInfer Prims-TS sparse MLA backend（DSA opt-in FlashInfer Prims-TS sparse MLA backend，perf，强化第 11、13 条）
- **主题**：port #48414：Kimi-K3.1 分支的 opt-in DSA 后端 `--dsa-prefill-backend prims_ts`/`--dsa-decode-backend prims_ts`，用 FlashInfer 的 Prims-TS sparse MLA（native E4M3 KV、FP8 MMA、natural-log LSE）替代 trtllm-gen，在既有 `trtllm` DSA route 上仅换 attention kernel 调用。
- **内容详情**：需 FlashInfer 0.7.4（#49771，ship Prims-TS D576 native-FP8 sparse MLA）；decode 快于 trtllm-gen 与 FlashMLA、cache 每 token 比 FlashMLA 更省字节；默认不变。含 Kimi-K3.1 sharded index-K DCP、SM107（Rubin）支持；测试改为新文件 `test_dsa_prims_ts.py`（B200 注册，因需 SM100/103/107）、DCP8 24/1025 行 partial/packed plan 用例；文档列两 DSA backend 表。
- **应用场景**：Kimi-K3.1 DSA 的 Prims-TS sparse MLA 后端（decode 提速 + KV 省字节）。
- **研判**：【观点】属「DSA sparse MLA 后端可移植/性能」（第 11、13 条），Prims-TS 替代 trtllm-gen 更快更省、需 FlashInfer 0.7.4；记为官方一手（feature/perf 级，移植自分支）。**证据等级**官方一手（feature 级，移植）。
- **来源**：sgl-project/sglang #48602（合并于 2026-11-12T08:18:01Z）。

### 3.5 `#49980` [Perf] dsa: Native-head FlashMLA and a precomputed schedule for DCP decode（DSA native-head FlashMLA + 预计算 DCP decode schedule，perf，强化第 11 条）
- **主题**：`sgl-flash-mla` 1.0.4 在 SM100 支持任意 head count [1,128]，但 DSA backend 仍把 q pad 到 64/128 heads（Kimi-K3.1 的 96 heads 与 TP8 rank 留的 12 跑 padded）。port Day-0 两 FlashMLA 件：native head 与预计算 DCP decode schedule。
- **内容详情**：wheel 支持时 `flashmla_kv` 跑真实 head count，否则按旧 pad；DCP rank 的 split-KV schedule 改从 scheduler kernel 的 `windowed` schedule 取（#48501 已在 main），不再每调用重建。native heads 适用于 `flashmla_kv` DSA 路径上所有 SM100 模型（不只 K3.1）。
- **应用场景**：Kimi-K3.1 等 DSA 模型在 SM100 的非 pad FlashMLA decode。
- **研判**：【观点】属「DSA FlashMLA native head 去 pad」（第 11 条），真实 head count 省 padding 浪费；记为官方一手（perf 级，移植自 Day-0）。**证据等级**官方一手（perf 级，移植）。
- **来源**：sgl-project/sglang #49980（合并于 2026-11-12T03:24:50Z）。

### 3.6 `#48263` [Perf][NVIDIA] nvfp4: Fuse SiLU-and-mul with the next linear layer's NVFP4 activation quant（SiLU-and-mul 与下一层 NVFP4 激活量化融合，perf，强化第 13 条）
- **主题**：扩展 #44491 的 SiLU-and-mul 与 down projection FP4 量化融合，从 Qwen3/Qwen3.5 的 `Qwen2MLP` 扩展到 `LlamaMLP`（Llama、Ministral3，含 `Nemotron-3-Embed-1B-NVFP4`）与 compressed-tensors W4A4 NVFP4 checkpoint（如 `Llama-3.1-8B-Instruct-NVFP4`、`Qwen3-8B-NVFP4`）。
- **内容详情**：`maybe_enable_silu_fp4_quant_fusion` 也接受 compressed-tensors W4A4 NVFP4；scale 用格式自身输入量化（compressed-tensors 的 `input_global_scale`、ModelOpt 的 `input_scale_inv`）；`can_run_silu_fp4_quant_fusion` 只融 2-D 激活且 dtype 与 `down_proj.params_dtype` 相同（FP16 激活喂 BF16 linear 走非融合）；kill switch `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION` 不变。
- **应用场景**：NVFP4（W4A4）MLP 的 SiLU+量化融合（Llama/compressed-tensors）。
- **研判**：【观点】属「NVFP4 激活量化融合扩展」（第 13 条），从 Qwen 扩展到 Llama/compressed-tensors；记为官方一手（perf/feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #48263（合并于 2026-11-12T01:35:23Z）。

### 3.7 `#48295` [Feat] modelopt: Quantize block-FP8 MTP draft experts to NVFP4 for MIXED_PRECISION checkpoints（block-FP8 MTP 草稿专家转 NVFP4，feature，强化第 13、14 条）
- **主题**：`--speculative-draft-model-quantization modelopt_fp4` 在加载时把 MTP draft 的 MoE experts 转 NVFP4；但 ModelOpt MIXED_PRECISION 导出（如 `Qwen3.8-Flash-Next-NVFP4`）的 MTP experts 是 block FP8，draft 解析成 `modelopt_mixed`、显式请求被忽略、512 个 draft experts 留 block FP8（单 96GB GPU 上挤占 KV 池）。
- **内容详情**：`convert_block_fp8_experts_to_nvfp4()` 把 block-FP8 FusedMoE 层路由到既有在线 FP8→NVFP4 loader（与 `--quantization modelopt_fp4` 同路径），按 checkpoint block size；`_resolve_explicit_draft_quant_config` 在用户显式传 `modelopt_fp4` 时调用。验证 RTX PRO 6000（SM120）与 GB200（SM100）；无原生 NVFP4 MoE kernel 的 GPU（如 H100）此前忽略、现会构建（未测）。
- **应用场景**：MIXED_PRECISION checkpoint 的 MTP 草稿专家量化以省 KV 池字节。
- **研判**：【观点】属「MTP 草稿专家量化」（第 13、14 条），block-FP8→NVFP4 省显存；记为官方一手（feature 级，RTX PRO 6000/GB200 验证）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #48295（合并于 2026-11-12T01:41:39Z）。

### 3.8 `#50005` [Feat] hisparse: Support Kimi-K3.1 DSA layers on PD decode with DFLASH（Kimi-K3.1 DSA 层 PD decode 上 HiSparse + DFLASH，强化第 8、14 条）
- **主题**：port #48414 的 HiSparse 部分：让 Kimi-K3.1 在 CUDA 上接受 `--enable-hisparse`，在 PD decode 实例上启用 DSA 层的 HiSparse + DFLASH 投机解码（至 block size 8）。
- **内容详情**：DSA backend 须为 `flashmla_kv`（`trtllm` 因不在 HiSparse FP8 allow-list被拒，ROCm 仍拒）；DCP 与 decode radix cache 仍按既有检查拒 HiSparse。KV 池在 HiSparse 开时把 DSA 子池建为 `HiSparseDSATokenToKVPool` 并转发 mapping API；coordinator 把 hybrid linear 池的 dense DSA 子池返回给 `get_hisparse_kvcache()`；DFLASH verify 在 verify locs 分配后调 `prepare_spec_verify`、accept 后 `commit_spec_accept_tokens`（EAGLE topk-1 布局）；`DFLASH` 加入 `HISPARSE_VALIDATED_SPEC_ALGORITHMS`；retract 的 HiSparse decode 请求回 prefill（rebootstrap），如 DeepSeek-V4。
- **应用场景**：Kimi-K3.1 在 PD decode 的 HiSparse + DFLASH（block size 8）。
- **研判**：【观点】属「HiSparse + DFLASH 跨 PD decode」（第 8、14 条），K3.1 DSA 层开 HiSparse；记为官方一手（feature 级，移植自分支）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #50005（合并于 2026-11-12T08:57:56Z）。

### 3.9 `#47934` [Fix][NPU] dsv4: transfer A3 C4 state using peer-local rows（Ascend A3 DSV4 C4 state 按行传输，fix，强化第 15 条跨硬件）
- **主题**：DeepSeek-V4-Flash 在 A3（Ascend 910C）的 PD 解聚：prefill 无 speculation、decode 开 DSpark 时返回 HTTP 500。根因：target C4 compressor 在 P 是 8 行 ring、D 是 16 行 ring，旧 A3 注册把整 ring 当一个 `StateType.SWA` transfer item，stride 不等被 Mooncake 布局校验拒（P 65,536/16,384 字节 vs D 131,072/32,768）。
- **内容详情**：把 C4 compressor state 注册为 `AscendStateType.DSV4_C4_STATE` 下的**独立行**，P/D 各自把同一逻辑历史位置映射到自身物理 state 行（item stride 两 peer 都变 8,192/2,048 字节）；payload callback 按 `state_row = (swa_location // swa_page_size) * local_ring_size + swa_location % local_ring_size` 取行，保持逻辑 token 顺序。
- **应用场景**：Ascend 910C 上 DSV4 PD 解聚的 C4 compressor state 传输。
- **研判**：【事实】PR 给 P/D 8/16 行 ring stride 不等导致 Mooncake 校验拒、HTTP 500 的根因与按行注册修复。【观点】属「NPU PD KV 传输正确性」（第 15 条），stride 不等改按行传输；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：sgl-project/sglang #47934（合并于 2026-11-12T03:01:18Z）。

### 3.10 `#48031` [Feat][NPU] hicache: Support AIV transfers for MLA and DeepSeek V4（Ascend HiCache AIV transfers for MLA/DSV4，强化第 8、15 条）
- **主题**：此前 Ascend HiCache sparse-copy 路径需 MemFabric-backed host 内存；本 PR 把 MLA 与 DeepSeek V4 的 L2 传输经 in-tree AIV `kv_exchange` 算子，默认用 ACL-registered host 内存、需要时选 MemFabric。
- **内容详情**：MLA 建池时准备 per-layer device/host 视图，load-back 复用并选匹配 Indexer slot/scale，backup 把 target + 附加 draft 池送到各自 host 层范围；DeepSeek V4 把 paged KV 与 state 池适配到同一 split-KV API，host 池按布局经既有 backup/load-back hook 派发。`kernel_ascend` 路径的 `--hicache-mem-layout`：MLA 用 `page_first_kv_split`/`layer_first_kv_split`，DSV4 用 `page_first`/`layer_first`；ACL 在池分配时给每块向 CANN 注册。
- **应用场景**：Ascend（华为）上 HiCache MLA/DSV4 的 KV 卸载与回灌。
- **研判**：【观点】属「NPU HiCache KV 传输」（第 8、15 条），AIV 算子替代 MemFabric 依赖；记为官方一手（feature 级）。**证据等级**官方一手（feature 级）。
- **来源**：sgl-project/sglang #48031（合并于 2026-11-12T11:50:31Z）。

### 3.11 `#50056` [Fix] hicache: resolve prefetch trailing-window keys before the storage read（HiCache prefetch trailing-window keys 在 storage read 前解析，fix，强化第 8 条）
- **主题**：trailing-window sidecar 池（SWA window、Mamba state）在独立 host side 池上，prefetch 中每 rank 的 storage read 失败使 TP ranks 对 storage 信念不同、后续入队的 storage writes 数量不同；backup ack drain 是跨 rank MIN 计数 + 位置对齐，差异永不愈合。
- **内容详情**：根因：PREFETCH 阶段 trailing-window 池带 placeholder keys（shared page-envelope 布局在分配时填真实 key、独立 side 池只在 prefetch IO 线程 `_page_transfer_sidecar` 且本 rank KV read 完成时填）；discard 是 rank 约定但按 `transfer.keys` 愈合、各 rank 此时不同。发现于 2-node PD stress（gpt-oss-120b、Full+SWA window 128/page 64、5% 注入后端失败）。修法为在 storage read 前解析 trailing-window keys。
- **应用场景**：SGLang HiCache 在注入故障下的 PD prefetch 健壮性。
- **研判**：【事实】PR 给「rank 信念分歧 → ack drain MIN 0 永不愈合」根因与 2-node PD stress 复现。【观点】属「HiCache prefetch 健壮性」（第 8 条），trailing-window keys 前置解析；记为官方一手（fix 级，含根因 + stress 复现）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：sgl-project/sglang #50056（合并于 2026-11-12T21:10:44Z）。

### 3.12 `#49851` [Feat] kimi_k3: Support ModelOpt MXFP8 KDA for Kimi-K3.1-Pro（Kimi-K3.1-Pro MXFP8 KDA 加载，fix/feature，强化第 4、13 条）
- **主题**：Kimi-K3.1-Pro NVFP4（MIXED_PRECISION）的 KDA `b_proj`/`f_a_proj` 存 MXFP8；两条 K3 加载路径假设 FP8_PB_WO，导致 TP>1 加载失败、且会用 raw fp8 字节跑融合 `[f_a|b]` GEMM。
- **内容详情**：`b_proj.weight_scale_inv` 仅当 scale 匹配参数 shape 时整份拷贝（MXFP8 per-row scale 现经 weight loader 分片）；`_merge_bfa_weights` 把 MXFP8 `f_a`/`b` 解量化到 BF16 tiny-GEMM buffer（同 FP8_PB_WO）；`_get_k3_dense_weight` 用 `2 ** (byte-127)` 转 UE8M0。GB300 TP8、MTP off、FP8 KV/indexer、FP32 SSM：GSM8K 5-shot 0.9742。新增 `test_kimi_k3_bfa_overlap.py` 含 MXFP8 dequant 回归（无 fix 失败）。
- **应用场景**：Kimi-K3.1-Pro NVFP4 MIXED_PRECISION 的 KDA 量化加载。
- **研判**：【事实】PR 给 TP>1 加载失败根因与 GB300 GSM8K 0.9742。【观点】属「量化 checkpoint 加载正确性」（第 4、13 条），MXFP8 KDA scale 分片修复；记为官方一手（fix/feature 级，含 GSM8K）。**证据等级**官方一手（fix 级，含 GSM8K）。
- **来源**：sgl-project/sglang #49851（合并于 2026-11-12T00:54:46Z）。

### 3.13 其他要点（标题级 / 描述级 / 性能级）
- `#47765` [Perf] dflash: Fuse the simulated-acceptance overwrite into one kernel（DFlash 模拟接受覆盖融成一个 kernel，perf 级，强化第 14 条）。
- `#47267` [Fix] dflash: Gather candidate top-k over the target lm_head's own shard group（DFlash 在 target lm_head 自身 shard group 上 gather candidate top-k，fix 级）。
- `#48367` [Perf] speculative: Fuse the draft proposal's Gumbel-max sample into one pass（draft 提案的 Gumbel-max sample 融成一次，perf 级，强化第 14 条）。
- `#44017` [Feat][XPU] GDN: support speculative decoding（XPU GDN 支持投机解码，feature 级，强化第 15 条）。
- `#43517` [Feat][NPU] sampling: Add sampling-mask support for Ascend（Ascend sampling-mask，feature 级，强化第 15 条）。
- `#47934` [Fix][NPU] dsv4: transfer A3 C4 state using peer-local rows（已详述，#47934）。
- `#49704` [Perf][AMD] muse-spark: Prefill and decode optimization（Muse Spark prefill/decode 优化，perf 级）。
- `#48996` [Perf][AMD] dsa: Default the DSA KV cache to fp8_e4m3 on gfx950（gfx950 上 DSA KV cache 默认 fp8_e4m3，perf/feature 级，强化第 13 条）。
- `#49929` [Fix][AMD] kimi-k3: Skip the CuTe DSL TGV tactic registration on ROCm（ROCm 跳 CuTe DSL TGV tactic 注册，fix 级）。
- `#47870` [Fix][AMD] ct-fp8-moe: Mark the pre-shuffled CT FP8 MoE weights is_shuffled（CT FP8 MoE 标记 is_shuffled，fix 级，强化第 4 条）。
- `#48219` [Fix][AMD] moe: Keep is_shuffled on AITER-shuffled unquantized MoE weights（AITER-shuffled 非量化 MoE 保留 is_shuffled，fix 级）。
- `#49268`/`#47468` [Fix] diffusion: Align Cosmos 3 / Pika H3 Flex inference numerics（扩散模型数值对齐，fix 级）。
- `#49195` [Feat] paged-experts: a disk tier for experts that do not fit host memory（paged-experts 磁盘层，feature 级，强化第 8 条）。
- `#45947` [Feat] Comm: Add gated allgather in kv cache sharding（KV cache sharding 加 gated allgather，feature 级）。
- `#49430` [Feat] mem_cache: Evict Mamba states by GreedyDual-Size（Mamba states 按 GreedyDual-Size 驱逐，feature 级）；`#50118` Revert 该 PR（撤回）。
- `#49922` [Perf] dsa: Stage the DCP top-k merge push in smem（DSA DCP top-k merge push 在 smem 分级，perf 级）。
- `#49943` [Perf] dcp: Replay prefix hits under breakable prefill graphs（DCP 在 breakable prefill graph 下重放前缀命中，perf 级）。
- `#48271` [Perf] qsa: drop the per-layer RoPE cache length check in the indexer（QSA indexer 去 per-layer RoPE cache 长度检查，perf 级）。
- `#49931` [Fix][NVIDIA] unquant: restore eager low-M BF16 performance（恢复非量化 eager low-M BF16 性能，fix 级）。
- `#49601` [Fix][NVIDIA] moe: Stop DeepGEMM's MLOPart helper once MegaMoE creates its buffer（MegaMoE 建 buffer 后停 DeepGEMM MLOPart helper，fix 级，强化第 12 条）。
- `#48949` [Perf] reject_sampling: Vocabulary-tiled final-token pass at small batch（小 batch 词汇分片最终 token pass，perf 级）。
- `#48840`/`#48846` [Fix/Feat] hisparse: disambiguate __ffsll for HIP dual overloads / reject unvalidated speculative algorithms by name（HiSparse HIP 重载消歧 / 按名拒未验证投机算法，fix/feature 级）。
- `#49762`/`#49308`/`#49391`/`#49808`/`#49456`/`#49354` [Perf] diffusion: 各类扩散模型 kernel/融合/缓存优化（perf 级）。
- `#48031`（已详述）、`#50018` [Perf] model_overrides: Default GLM-5.3-Flash to an agentic Mamba pool split（GLM-5.3-Flash 默认 agentic Mamba pool split，perf 级）。
- `#48412` [Feat] pp: Relay routed experts through pipeline-parallel stages（routed experts 经 PP stage 中继，feature 级）。
- `#50009` [Fix] mamba: Clear a stale COW source when an unadmitted request is matched again（未准入请求再匹配时清 stale COW source，fix 级）。
- `#49669` [Fix] disaggregation: Bound unsharded PD prefill prompts at bootstrap（bootstrap 时限制未分片 PD prefill prompt，fix 级）。
- `#49798` [Fix] scheduler: Answer release_memory_occupation only after every pipeline stage released（每 pipeline stage 释放后才答 release_memory_occupation，fix 级）。
- `#50052` [Fix] sgl-router: Close the PD gaps that block replacing sgl-model-gateway in CI（sgl-router 关 PD gaps，fix 级）。
- `#49771` [Chore] deps: Bump FlashInfer to 0.7.4（FlashInfer 升 0.7.4，dep 级，与 vLLM #67042 同步）。
- `#50144` [Feat] JIT: Add SGLANG_JIT_ENABLE_LINEINFO for profiler source views（JIT lineinfo，feature 级）。

---

## 4. TensorRT-LLM：2026-11-12 提交（39 条）—— 量化静默正确性(#22106/#21346) + GLM-5.2 NVFP4 sm_103 性能(#21695) + KV/资源释放修复(#21784/#20840/#21612) + 混合投机 replay state 入 KV region(#21324) + 冷页 KV 混合精度(#20197)

### 4.1 `#22106` [None][fix] Use raw FC1 scale for MXFP4 FP8 MoE input quantization（MXFP4 FP8 MoE 用 raw FC1 scale，fix，强化第 4、6、13 条）
- **主题**：修复 NVBug 6178915：TRTLLM-Gen `W4A8_MXFP4_FP8` 在 dispatch 跟随量化时用 FC1/FC2 input-scale 比来量化 raw FC1 输入，导致 4 个默认 SwiGLU DEP/NVLINK 用例约 97% mismatch。
- **内容详情**：两种 dispatch 顺序都用 `fc31_input_gate_dequant`（raw FC1 input scale）作 FP8 量化，保留 `fc31_input_dequant` 作 fused kernel 的 GEMM1 ScaleC 比；受影响路径含 NVLINK 单侧/双侧 DEP、attention-DP + MoE-TP (DTP)、DEP 的 `AllGatherReduceScatter` 回退；dispatch/combine 行为与精度容差不变。移除 NVBug 6178915 测试 skip、为两 scale 字段在创建/加载处文档化、在查 FP8 输入值前跑真实量化方法 loader；聚焦回归 4 个用例（两 dispatch flag × padded/unpadded 宽度），GB200 全过。
- **应用场景**：TRT-LLM W4A8_MXFP4_FP8 MoE 的输入量化正确性。
- **研判**：【事实】PR 给「4 个默认 DEP/NVLINK 用例约 97% mismatch」与 GB200 5 用例全过。【观点】属「量化 MoE 静默错误」（第 4、6、13 条），FC1 scale 误用致 97% mismatch；记为官方一手（fix 级，含 mismatch 数字）。**证据等级**官方一手（fix 级，含 mismatch 数字）。
- **来源**：NVIDIA/TensorRT-LLM #22106（合并于 2026-11-12T16:48:00Z）。

### 4.2 `#21346` [None][fix] Qwen3.8-Flash-Next: mask FP8 PLE rows by bytes under attention DP（Qwen3.8-Flash-Next FP8 PLE 行按字节 mask，fix，强化第 4、13 条）
- **主题**：Qwen3.8-Flash-Next NVFP4/FP8 checkpoint 的 PLE n-gram 表存 FP8 e4m3 + scalar scale、查表后才解量化；attention DP + tp>1 时每 rank 拥有行切片、all-gather 各 rank n-gram ID、lookup 自身行、清零其余、解量化、reduce-scatter。清零在仍 FP8 的行上做，而 PyTorch 无 FP8 `masked_fill_` kernel，每 worker 首个 forward（warmup）崩。
- **内容详情**：对 FP8 表（`float8_e4m3fn`/`float8_e5m2`）经 `uint8` view 做 mask——全零字节在两格式都是 +0.0，结果与解量化后 mask 位精确相同；reduce-scatter 仍在解量化 BF16 行上、通信量不变。CI 原 `_run_evals` 对所有用例设 `TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1` 故未触发；现 `_run_evals` 取 `ple_host_offload`（默认 True），新增 `test_nvfp4_adp4_mtp3_trtllm` 把表放 GPU 内存（原 offload 用例保留 pre-merge 覆盖）。影响：NVFP4/FP8 + attention DP + tp>1（dep2–dep16）+ PLE 驻 GPU；tp=1、无 attention DP、`TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1`、BF16 checkpoint 不受影响。
- **应用场景**：Qwen3.8-Flash-Next 在 attention DP 下的 FP8 PLE 表 mask。
- **研判**：【事实】PR 给「PyTorch 无 FP8 masked_fill_ → NotImplementedError」与 uint8 view 位精确修复、新增 GPU 内存用例。【观点】属「量化 attention DP 静默崩」（第 4、13 条），FP8 表 mask 缺失 kernel；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：NVIDIA/TensorRT-LLM #21346（合并于 2026-11-12T12:30:33Z）。

### 4.3 `#21695` [None][perf] modeling_v2 GLM-5.2 NVFP4 sm_103 tep4: faster prefill and decode, MTP3 / MTP5, long-context decode（GLM-5.2 NVFP4 sm_103 tep4 性能，perf，强化第 11 条）
- **主题**：在 #20662 的 `glm_5_2_nvfp4__sm_103__tep4`（4×B300/GB300、attention TP4、experts EP4、MTP1）基础上：让 context（prefill）与 MTP1 decode 更快（多数用 target-owned CuTe DSL kernel）、放开 MTP3/MTP5（宽 65–256 行 decode 路径）、加速长上下文 decode（64k–160k KV、batch 1–2：draft step 间 index share、padded-width decode cell、把长行 decode top-k 跨 CTA cluster 切、非 2 幂的 unpadded decode MoE、每 group 一次 launch 的 MTP draft-loop 簿记），并修沿途正确性（FC1 shared-memory staging race、producer tails/TMEM-dealloc fences、chunked prefill 下 context indexer 的 q-split all-gather、decode top-k ties 过 32768 列——自 #20662 起存在、MTP draft-chain 归一化）。
- **内容详情**：长上下文 decode、解聚 GB300（CTX DEP4 + GEN TEP4、MTP5、batch 2、ISL 160k、forced acceptance 2.61）：本 PR（tep4 target）decode step **12.30 ms** vs #20524 `__tep` target 12.69 ms；聚合 4×B300 等其余对照（标题级，具体数值 PR 未完整给出，记为 perf 描述级）。其余模型默认行为不变。
- **应用场景**：GLM-5.2 NVFP4 在 GB300/B300 的 MTP3/MTP5 长上下文解码。
- **研判**：【观点】属「GLM-5.2 NVFP4 长上下文解码性能」（第 11 条），tep4 decode step 12.30 vs 12.69ms、MTP3/5 放开；其余对照数值未完整，记为官方一手（perf 级，部分描述级）。**证据等级**官方一手（perf 级，部分描述级）。
- **来源**：NVIDIA/TensorRT-LLM #21695（合并于 2026-11-12T01:44:13Z）。

### 4.4 `#21784` [https://nvbugs/7023112][fix] Release profiling KV pools before NIXL deregistration（profiling KV pools 在 NIXL 注销前释放，fix，强化第 8 条）
- **主题**：`PyExecutor._release_execution_resources()` 现先准备 transceiver、关 resource managers、再完成 transceiver shutdown，改变 profiling 与 terminal release 的 teardown 顺序，使 pools 在 NIXL deregistration 前 unmapped。
- **内容详情**：`KvCacheTransceiver.prepare_shutdown()` 默认全关；用默认的后端须在其 resource managers 释放 pools 前停止 cache 访问；`KvCacheTransceiverV2` 在准备阶段保留注册、session closure/worker shutdown 失败时使 shutdown 可重试。改动两单测文件（teardown 顺序/drain-failure 与 profiling/terminal release 排序参数化、registration retention/drain/session refusal/partial shutdown failure/retry）。评审标注 GB200/GB300 资格未跑、current-head 结果不可用（coverage needs follow-up）。
- **应用场景**：TRT-LLM 解聚 KV transceiver 的 teardown 顺序正确性。
- **研判**：【观点】属「profiling KV pool 与 NIXL 注销顺序」（第 8 条），teardown 顺序修正；评审标注硬件资格未跑，记为官方一手（fix 级，needs follow-up 标注）。**证据等级**官方一手（fix 级，needs follow-up）。
- **来源**：NVIDIA/TensorRT-LLM #21784（合并于 2026-11-12T18:22:25Z）。

### 4.5 `#20840` [https://nvbugs/6939213][fix] Release cached MoE runner workspaces at executor shutdown（executor shutdown 释放缓存 MoE runner workspaces，fix，强化第 4、8 条）
- **主题**：`TestDeepSeekV4Pro::test_gsm8k_full_accuracy` 在 DGX B200 x8、KV-cache 大小估算创建临时资源时因「Executor creation failed due to insufficient GPU memory」失败，且仅在 `TestDeepSeekR1::test_nvfp4_multi_gpus` 之后同一复用 MPI worker 池中跑才失败；单独跑通过。
- **内容详情**：根因：`MoERunner` 在类级 dict 缓存 C++ `FusedMoeRunner` 实例（随进程存活），每 runner 在调用间保留 GPU workspace；`PyExecutor.shutdown()` 从未释放这些 workspace，下一个 executor 起时带着上一模型的 MoE workspace。唯一释放 `MoERunner.clear_all_workspaces()` 发生在下一 executor 的 general warmup——对 V4-Pro 太晚（在 KV-cache 估算处就 OOM）。B200 x8 复用 worker 中 traced 28.27 GiB/rank 仍分配（DeepSeek-R1 NVFP4 CUTLASS MoE workspace）；V4-Pro torch 内存加载权重后从 126.53 GiB（fresh pool）到 156.79 GiB，仅约 7.5 GiB 留给 6.56 GiB 估算分配。修法：executor 永久 shutdown 时、CUDA graph 释放前释放缓存 MoE runner workspaces；KV-cache 估算 shutdown 保留（其模型与模型拥有的 graph 被最终 executor 复用、warmup 已释放）；下次使用再分配。
- **应用场景**：TRT-LLM 复用 MPI worker 池下的 MoE workspace 内存泄漏/OOM。
- **研判**：【事实】PR 给「B200 x8 28.27 GiB/rank 泄漏 → V4-Pro OOM」根因与 126.53→156.79 GiB 内存轨迹。【观点】属「MoE runner workspace 释放」（第 4、8 条），shutdown 漏释放致复用 OOM；记为官方一手（fix 级，含根因 + 内存轨迹）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：NVIDIA/TensorRT-LLM #20840（合并于 2026-11-12T18:45:45Z）。

### 4.6 `#21612` [None][fix] NIXL: release transfer status handles without holding the GIL（NIXL 释放 transfer status 不持 GIL，fix，强化第 8 条）
- **主题**：`TransferStatus.release()` 现放 GIL 运行；成功时 `NixlTransferStatus.release()` 在 C++ 析构前清 handle；失败时返回 false 保留 handle，Python 析构忽略该结果、然后 C++ 析构重试 `release()`——该重试可能在 GIL 持有时等 NIXL 锁。
- **内容详情**：dev review 提示验证失败-release 清理不重引 reported freeze；tests 不覆盖失败 release、C++ 析构重试、backend handle 清理、锁竞争或 GIL 可用性。QA 标注 coverage needs follow-up。`tests/unittest/disaggregated/test_agent.py` 改不可用-status mock、加 4 个 CPU-only binding-gated 测试（binding 暴露、forwarding、delete 时 release、在 `_cpp_status` 初始化前 delete）；文件经 `l0_cpu.yml` 的 `unittest/disaggregated` selector 覆盖。
- **应用场景**：TRT-LLM NIXL 传输状态句柄释放（去 GIL）。
- **研判**：【观点】属「NIXL 句柄释放去 GIL」（第 8 条），避免 GIL 持有下重试死锁；评审标注覆盖需 follow-up，记为官方一手（fix 级，needs follow-up）。**证据等级**官方一手（fix 级，needs follow-up）。
- **来源**：NVIDIA/TensorRT-LLM #21612（合并于 2026-11-12T09:30:32Z）。

### 4.7 `#21324` [TRTLLM-16303][refact] Keep hybrid speculative replay state in KV cache regions（混合投机 replay state 存 KV cache region，refactor，强化第 14 条）
- **主题**：带投机解码的混合模型在 recurrent（SSM）state 旁保留 per-request replay state（自上次 checkpoint 以来被接受的 draft token 输入，verify kernel 重放而非每步写 checkpoint）。在 KVCacheManagerV2 下有三类问题：① Mamba2/GDN 在 suspend/resume 丢失（V2 suspend 可把 SSM page 移到另一 slot，history 留下）；② KDA 把 replay buffer 放进 snapshotted SSM slot，每前缀 snapshot 都带 replay 字节、suspend 把 pending draft 折进 state、resume 重播；③ Mamba2/GDN tensor 按 SSM pool slot（含 snapshot slot）每行、不在内存预算内。
- **内容详情**：把 replay state 在 KVCacheManagerV2 中作为 **request region** 存：生命周期独立、每 living cache 一 slot、永不发布给前缀复用、随请求过 suspend/resume（来自 @eopXD 的 `user/yuehtingc/kda-owned-replay-region`）；本 PR 加 Mamba2/GDN 的共享 replay region 与 kernel 的 replay-slot 接线。region 模型与 native 支持为 Yueh-Ting Chen 原提交、本 PR 合入其 authorship。
- **应用场景**：TRT-LLM 混合模型（Mamba2/GDN/KDA）投机解码 replay state 的 KV 管理。
- **研判**：【观点】属「混合模型投机解码 replay state 入 KV region」（第 14 条），修 suspend/resume 丢失与 snapshot 冗余；记为官方一手（refactor 级，移植自 upstream 作者分支）。**证据等级**官方一手（refactor 级）。
- **来源**：NVIDIA/TensorRT-LLM #21324（合并于 2026-11-12T13:40:07Z）。

### 4.8 `#20197` [TRTLLM-17230][feat] Support First N/Last N Mix precision for cold-page KV Cache Compression（冷页 KV 混合精度 First N/Last N，feature，强化第 8、13 条）
- **主题**：冷页 KV 压缩（`quantization_for_cold_page`）离开 GPU 时每页存 NVFP4；本 PR 加 `mixed_precision_first_n`/`mixed_precision_last_n`：序列的前 N 与最近 N token 在 host/disk tier 保持运行时 KV 类型（FP8/BF16/FP16），只有中间 token 才 compact 存储。
- **内容详情**：KVCM V2 给每层一个 mixed-precision high layer（即 `mixed_nvfp4_*` 的机制），把前 N 作 sink token、最近 N 作 sliding window；high page 是同层 page 的同池同槽副本，codec 看不见、其页在 tier 间迁移不变。attention 从 high page 就地读 held token（dense backend 把 held block 的 block offsets 指向 high layer 的 block、再把新行写回自身层；sparse MLA backend 把选中行经既有 index remap 指向 high layer 持有的行，无 staging copy）；Drafter 层（MTP/Eagle3）也拿 high layer、单独 drafter 留 plain cache。`TorchLlmArgs` 按 mixed-precision KV 规则校验、收窄到 TRTLLM attention backend（FlashInfer 整页索引）、拒 `fp8_ds_mla` 布局。默认关（两字段 0），无它则行为不变；Mixed-precision KV（`mixed_nvfp4_*`）保留其 staging path。文档 `docs/source/features/kv-cache-compression.md`。
- **应用场景**：TRT-LLM KVCM V2 冷页压缩的 sink/sliding-window 混合精度（保近因/远因精度）。
- **研判**：【观点】属「冷页 KV 混合精度」（第 8、13 条），前 N/后 N 保高精度、中间 compact；记为官方一手（feature 级，默认关）。**证据等级**官方一手（feature 级，默认关）。
- **来源**：NVIDIA/TensorRT-LLM #20197（合并于 2026-11-12T08:34:28Z）。

### 4.9 其他要点（标题级 / 描述级 / 性能级）
- `#21863` [fix] Size the GQA segmented-decode FMHA workspace for FlashInfer's LSE softmax stats（GQA segmented-decode FMHA workspace 按 FlashInfer LSE 定尺寸，fix 级）。
- `#21788` [fix] Runtime fixes found while bringing up Mistral Large 4（Mistral Large 4 启动期运行时修复，fix 级）。
- `#20110` [fix] Correct HF reference precision in BF16 all-reduce accuracy test（BF16 all-reduce 精度测试 HF 参考精度修正，fix 级）。
- `#21308` [perf] Cut per-turn prompt handling cost for long multi-turn prompts（长多轮 prompt 每轮处理成本削减，perf 级）。
- `#22028` [perf] Build FlashMLA without relocatable device code（FlashMLA 不编 relocatable device code，perf 级）。
- `#21791` [perf] Build each DeepGEMM and CuTe DSL warmup kernel once across the ranks of a launch（DeepGEMM/CuTe DSL warmup kernel 跨 rank 只编一次，perf 级）。
- `#22174` [fix] Align the serve constructor ratchet with its landing baseline（serve 构造 ratchet 与落地基线对齐，fix 级）。
- `#22103` [fix] Reject encoder-decoder models in recompute_active_requests（recompute_active_requests 拒 encoder-decoder 模型，fix 级）。
- `#20943` [fix] Cherry-pick #20459: Unblock Python disagg transfers stalled by a context-side timeout（cherry-pick 解聚 Python 传输因 context 侧超时卡住，fix 级，强化第 8 条）。
- `#21716` [fix] Support etcd credentials, TLS and signed records for the disagg cluster storage（解聚 cluster storage 支持 etcd 凭据/TLS/签名记录，fix 级）。
- `#21895` [doc] Document runtime speculation controls（运行时投机解码控制文档，doc 级）。
- `#22087` [test] Unwaive Qwen3.5 VL and modeling unit tests fixed by #21876（Qwen3.5 VL 与 modeling 单测 unwaive，test 级）。
- `#19996` [feat] agent-flow: opt-in MX qualification for modeling-bringup（agent-flow 建模启动 opt-in MX 资格，feature 级）。
- `#21471` [feat] Add Beam StairCase operators and model components（Beam StairCase 算子与模型组件，feature 级）。
- `#21942`/`#21970`/`#22154`/`#22156`/`#22116`/`#22129`/`#22165`/`#21948`/`#21976`/`#21993`/`#21703`/`#21990`/`#19212`/`#20038`/`#20089` infra/waive/test 杂项（infra/ci 级）。

---

## 5. llama.cpp：2026-11-12 提交（39 条）—— 服务端安全加固(#32115/#32120/#32113) + CUDA NVFP4 prefill 融合(#26048) + few-row MMQ(#31533) + 多后端广度(Adreno #32046 / Intel Vulkan #29357 / Hexagon #32094 / SpacemiT #31897 / ARM NEON #32128) + SYCL/量化修复

### 5.1 `#32115` [server] disallow reaching local addresses from CORS proxy and media URLs（CORS proxy/media URLs 禁达本地地址，fix/安全，新增部署安全主线）
- **主题**：CORS proxy 与输入 media URLs（经 chat completion 端点）原本允许达任意本地地址（按设计，期望用户自行 sandbox `llama-server`）；但许多部署直接跑在本地机，成为真实问题。
- **内容详情**：改动后 CORS proxy 只允许达公网地址（不再能经它达 `localhost`/同 LAN 的 MCP server）；给 chat completion 输入 media 若存于同 LAN 的 NAS，现需 `--media-allow-private-urls` 才放开。这两个原本允许的有效用例现不再默认可达内网。
- **应用场景**：llama.cpp `llama-server` 的 SSRF / 内网逃逸防护。
- **研判**：【事实】PR 给「默认禁止 CORS proxy/media URLs 达本地地址、需 `--media-allow-private-urls` 放开」的语义变更。【观点】属「服务端部署安全」（新增部署安全主线），把「可达内网」显式关掉；记为官方一手（fix/安全 级）。**证据等级**官方一手（fix/安全 级）。
- **来源**：ggml-org/llama.cpp #32115（合并于 2026-11-12T11:33:49Z）。

### 5.2 `#32120` [server] disallow direct access child instance from browser（浏览器禁端口扫描发现 child instance，fix/安全）
- **主题**：禁止浏览器端口扫描发现 child instance（child instance 在 master 已仅 loopback 监听）。
- **内容详情**：仍允许本地 app/本地脚本在无 api key 下直连 child instance（视为可信）；仅禁止浏览器侧发现。
- **应用场景**：llama.cpp server 多实例部署的浏览器侧端口扫描防护。
- **研判**：【观点】属「服务端部署安全」（新增主线），child instance 防浏览器发现；记为官方一手（fix/安全 级）。**证据等级**官方一手（fix/安全 级）。
- **来源**：ggml-org/llama.cpp #32120（合并于 2026-11-12T11:16:30Z）。

### 5.3 `#26048` CUDA: Optimize prefill via fuse of w_s scale in epilogue MMQ for nvfp4 checkpoints（CUDA prefill 融合 w_s scale 到 MMQ epilogue，perf，强化第 4、13 条）
- **主题**：对 FP4 量化权重（`*_s` tensor 的 per-tensor / per-expert scale），master 在 dispatch 到 MMQ 的 prefill 尺寸下把它们跑成独立图操作：`C=W_hat X` 后 `Y=s_W ⊙ C`（MMQ 写 C 到全局内存、再独立 elementwise kernel 读 C 应用 scale/bias 写 Y）；MoE 更糟（per-expert scale 先 RESHAPE→REPEAT→GET_ROWS 再 MUL）。本 PR 把 `Y` 直接在 MMQ writeback 阶段（含 stream-k fixup kernel）算，去掉中间张量写读与 kernel launches。
- **内容详情**：epilogue 支持 scalar scale（dense projection，`GGML_OP_MUL_MAT`）与 ID-indexed scale（MoE，`GGML_OP_MUL_MAT_ID`，按 token 路由 expert）；融合仅启用 NVFP4，其余 MMQ 类型保持非融合；scale 在共享 `write_back` 中按 `type==NVFP4 && x_scale ? *x_scale : 1` 应用、stream-k fixup 不变。新增 `ggml_cuda_mul_mat_uses_mmq()` 供融合 matcher 判断节点是否真去 MMQ。去除 `MUL_MAT→MUL(scale)` 的 1 个 `MUL`；`MUL_MAT_ID→RESHAPE→REPEAT→GET_ROWS→MUL(scale)` 的 3 个（REPEAT/GET_ROWS/MUL）+ gathered scale 张量。
- **应用场景**：llama.cpp CUDA 后端的 NVFP4（W4A4）dense/MoE prefill 融合。
- **研判**：【观点】属「NVFP4 激活/权重 scale 融合到 MMQ」（第 4、13 条），去中间张量写读与 kernel launch；记为官方一手（perf 级，含 kernel 去除对照）。**证据等级**官方一手（perf 级）。
- **来源**：ggml-org/llama.cpp #26048（合并于 2026-11-12T09:28:37Z）。

### 5.4 `#31533` CUDA: few-row MMQ for BS 2-24（CUDA few-row MMQ 2-24 列，perf）
- **主题**：加一个量化权重的 CUDA mat-mul kernel（2–24 个 src1 列），主要动机是投机解码（draft 7 token → 8 列）与小 batch 性能。
- **内容详情**：这些尺寸下既有 kernel 都不高效——MMVQ 每列重复乘加工作、MMQ 为大 batch 设计（2–24 列时 128 行 tile 太少、stream-k 需 fixup kernel 且每 K step 有 block-wide barrier）。新 kernel 每权重读一次、解一次到 int8 MMA fragment、与所有列在 tensor cores 上乘。支持 Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ3_S/IQ4_NL/IQ4_XS（NVIDIA Ampere 至 DGX Spark）；DGX Spark 需每 block 先预取权重到 L2。无 stream-k fixup kernel、结果确定性。
- **应用场景**：llama.cpp CUDA 投机解码 draft / 小 batch 的量化权重 GEMM。
- **研判**：【观点】属「CUDA 小矩阵 GEMM 调度」，2–24 列专用 kernel 去 MMVQ/MMQ 低效；记为官方一手（perf 级）。**证据等级**官方一手（perf 级）。
- **来源**：ggml-org/llama.cpp #31533（合并于 2026-11-12T13:12:46Z）。

### 5.5 `#32046` opencl: add decomposed FA for Adreno（OpenCL Adreno decomposed FA，perf，强化第 15 条跨硬件）
- **主题**：Adreno 上 fused FA 因资源限制对部分配置慢；本 PR 做 decomposed FA（非独立 op 的 FA，而是按 fused FA 流程、用多个独立 kernel 减资源压力），基于 #30182。
- **内容详情**：gemma-4-E4B-it-Q4_K_M：master decode 24.98 t/s → PR 35.29 t/s（约 +41%）、accept 56.8%→61.1%；Qwen3.6-35B-A3B-Q4_K_M：decode 20.52→23.79 t/s（+15.9%）、accept 65.8%→64.8%；gemma-4-12b-it-Q4_K_M：decode 15.08→（master 数值截断，PR 提速）等（具体数值 PR 未完整给出，记为 perf 描述级）。属移动 GPU（Adreno）注意力性能。
- **应用场景**：llama.cpp OpenCL 后端在 Adreno 的移动端 FA。
- **研判**：【观点】属「移动 GPU 注意力性能」（第 15 条），decomposed FA 减资源压力；gemma-4-E4B decode +41% 但部分数值截断，记为官方一手（perf 级，部分描述级）。**证据等级**官方一手（perf 级，部分描述级）。
- **来源**：ggml-org/llama.cpp #32046（合并于 2026-11-12T00:18:25Z）。

### 5.6 `#29357` vulkan : Intel FA kernel for prefill.（Vulkan Intel FA prefill kernel，perf，强化第 15 条）
- **主题**：为 Intel 平台改进 FA prefill 性能，加 Intel prefill FA shader 并统一 prefill/decode shader 的 push constants 定义、按 FA query 激活尺寸更新 shader 选择逻辑；为修老 GLSLC 编译问题加 `GGML_VULKAN_COOPMAT_GLSLC_SUPPORT` 宏检查。
- **内容详情**：定制 shader 现启用在 XE1 ARL-H、XE2/XE3 平台；A770（XE1 dGPU）因性能数据不一致仍默认关（shader 需 coopmat）。prefill perf（llama-bench，pp8192，Qwen3 8B/27B/35B-MoE/gpt-oss-20B Q4_K_M）：ARL-H Windows 各模型 pp8192 t/s 提升（如 qwen3 8B 126.58→[PR 数值截断]、qwen35moe 35B.A3B 247.82→[截断] 等，具体数值 PR 未完整给出，记为 perf 描述级）。
- **应用场景**：llama.cpp Vulkan 后端在 Intel（XE1/XE2/XE3）的 FA prefill。
- **研判**：【观点】属「Intel Vulkan 注意力性能」（第 15 条），定制 FA shader 提速；具体数值截断，记为官方一手（perf 级，描述级）。**证据等级**官方一手（perf 级，描述级）。
- **来源**：ggml-org/llama.cpp #29357（合并于 2026-11-12T10:19:18Z）。

### 5.7 `#32094` hexagon: add Q5_0 and Q5_1 support（Hexagon 加 Q5_0/Q5_1，feature，强化第 15 条）
- **主题**：为 Hexagon HTP 加 Q5_0 与 Q5_1 数据类型支持：Q5_0 有独立 tile（704 字节）与 HVX/HMX kernel；Q5_1 共享 Q5_K tile 与 kernel。
- **内容详情**：验证：v73 与 v81 WoS 上测试；Q5_0 59/59 通过、Q5_1 52/52 通过。
- **应用场景**：llama.cpp Hexagon（骁龙 HTP）后端的 Q5_0/Q5_1 量化。
- **研判**：【事实】PR 给 Q5_0 59/59、Q5_1 52/52 通过。【观点】属「Hexagon 量化覆盖」（第 15 条）；记为官方一手（feature 级，含通过数）。**证据等级**官方一手（feature 级，含通过数）。
- **来源**：ggml-org/llama.cpp #32094（合并于 2026-11-12T16:33:17Z）。

### 5.8 `#31897` ggml : route GATED_DELTA_NET to the SpacemiT RVV kernel（SpacemiT RVV GATED_DELTA_NET kernel，perf，强化第 15 条）
- **主题**：OSPP 2026 / #22863：在 SpacemiT 后端用 RVV kernel 启用 `GGML_OP_GATED_DELTA_NET`（此前 compute case 被注释、走标量 CPU 路径）。
- **内容详情**：加 `spacemit_kernels::rvv::forward_gated_delta_net_f32`、经 `compute_forward`/`get_tensor_traits` 路由；RVV 做 dot/outer/decay、`expf` 仍标量、短向量 tail 避免把垃圾折进 reduction。`test-backend-ops -b CPU -o GATED_DELTA_NET` A100 与 X100 均 43/43；A100 micro（8 线程、3 interleaved runs）geomean ~1.61×；X100 重测 `32,128,64,1` 1.28×、其余 ~1.07–1.73×；E2E Qwen3.5-0.8B A100 pp ~1.06×、tg ~1.02–1.03×。
- **应用场景**：llama.cpp SpacemiT（RISC-V）后端的 GATED_DELTA_NET（GDN）算子。
- **研判**：【事实】PR 给 A100/X100 43/43、A100 geomean 1.61×。【观点】属「SpacemiT RVV 算子性能」（第 15 条）；记为官方一手（perf 级，含对照）。**证据等级**官方一手（perf 级，含对照）。
- **来源**：ggml-org/llama.cpp #31897（合并于 2026-11-12T10:20:26Z）。

### 5.9 `#31873` [SYCL] SYCL AOT produces garbage output for models > 4 GB（SYCL AOT >4GB 垃圾输出修复，fix，强化第 15 条）
- **主题**：修 issue #31685/#31033：SYCL AOT（`GGML_SYCL_DEVICE_ARCH=arl-h`）对 >4 GB 模型产生垃圾输出（GGGG...），由 buffer size flag 引起。
- **内容详情**：更新 Cmakefile 支持 AOT >4G；build scripts 加 `--aot` 与 `--no-xmx`（设 `GGML_SYCL_SUPPORT_XMX=OFF`）；代码跳过 `GGML_SYCL_SUPPORT_XMX=OFF` 的 DPAS 代码。由 @Dosperado74 报告并 patch。
- **应用场景**：llama.cpp SYCL（Intel GPU）AOT 编译下 >4GB 模型的正确性。
- **研判**：【事实】PR 给「SYCL AOT >4GB 产生 GGGG... 垃圾输出」根因（buffer size flag）与修复。【观点】属「SYCL AOT 正确性」（第 15 条）；记为官方一手（fix 级，含根因）。**证据等级**官方一手（fix 级，含根因）。
- **来源**：ggml-org/llama.cpp #31873（合并于 2026-11-12T06:54:59Z）。

### 5.10 `#32128` ggml-cpu: use the 4x4 simd_gemm tile on ARM NEON（ARM NEON 4x4 simd_gemm tile，perf）
- **主题**：修宏以在 Linux AArch64 CPU 的 flash attention 上启用 4x4 tile path。
- **内容详情**：在 Arduino UNOQ（A53）上测得 7%/8% 提升。
- **应用场景**：llama.cpp ARM NEON（AArch64）CPU 的 flash attention 性能。
- **研判**：【事实】PR 给 Arduino UNOQ（A53）7%/8% 提升。【观点】属「ARM NEON 注意力性能」；记为官方一手（perf 级，含 A53 对照）。**证据等级**官方一手（perf 级）。
- **来源**：ggml-org/llama.cpp #32128（合并于 2026-11-12T10:14:00Z）。

### 5.11 `#32067` CUDA: skip empty MoE tiles in MMQ stream-k fixup（CUDA MMQ stream-k fixup 跳过空 MoE tiles，fix）
- **主题**：CUDA MMQ stream-k fixup 中跳过空 MoE tiles。
- **内容详情**：标题级（PR 正文未完整抓取，记为 fix 级）。
- **应用场景**：llama.cpp CUDA MoE MMQ 的 stream-k fixup 正确性/效率。
- **研判**：【观点】属「CUDA MoE MMQ 正确性/效率」修复；记为官方一手（fix 级，描述级）。**证据等级**官方一手（fix 级，描述级）。
- **来源**：ggml-org/llama.cpp #32067（合并于 2026-11-12T03:45:45Z）。

### 5.12 其他要点（标题级 / 描述级）
- `#32113` [server] explicit opt-in tools via non-loopback addr, check cors host port（非回环地址显式 opt-in tools + CORS host 端口检查，fix/安全级，新增部署安全主线）。
- `#32133` [server] keep the server awake while a request uses its state（请求用 state 时保持 server awake，fix 级）。
- `#32119`/`#32051`/`#32112` [server/ui] server/test/UI 杂项（test/UI 级）。
- `#32130`/`#32129`/`#32132`/`#32131` [mtmd] projectors 张量/空间合并/position embedding 校验（fix 级）。
- `#30947` [opencl] bin kernel `kernel_gemm_noshuffle_q4_0_f32_32b_trans_ila_a7_bin`（OpenCL bin kernel，feature 级）。
- `#31919` [CUDA] add cuda debug buffer pool（CUDA 调试 buffer pool，feature 级）。
- `#32013` [sycl] use cached device-id in set_main_device（SYCL 缓存 device-id，fix 级）。
- `#31138` [sycl] make oneDNN FA cache thread-safe and memoize runtime failures（SYCL oneDNN FA cache 线程安全，fix 级）。
- `#32043` [llama] avoid touching unused graph metadata pages（llama 不碰未用 graph metadata 页，perf 级）。
- `#31915` [chat] refactor gemma4 to build loose schema rules（chat gemma4 重构，refactor 级）。
- `#32100` [ggml-backend] prefetch only inputs computed by earlier splits（ggml-backend 只预取早 split 算的输入，perf/refactor 级）。
- `#31929` [ggml] dispatch SpacemiT Q4_1 through the HP kernel（SpacemiT Q4_1 经 HP kernel，feature 级，强化第 15 条）。
- `#32102` [qwen3.5] merge tensors on convert also（qwen3.5 convert 也合并张量，feature 级）。
- `#32025` [server] cancel MCP tool calls on client disconnect or timeout（server 客户端断开/超时取消 MCP 工具调用，fix 级）。
- `#29801`/`#30565` [llama] prune（llama 清理，cleanup 级）。
- `#32138`/`#32166`/`#32167`/`#32168` [ci/server] UI 构建/局域网测试/server 测试（ci/测试级）。
- `#32011` [fix(hexagon)] make host build work with clang-cl on Windows ARM64（Hexagon 主机构建 clang-cl Windows ARM64 修复，fix 级，强化第 15 条）。
- `#30335` [chat] accept closer at medium/low effort（chat 中/低 effort 接受 closer，fix 级）。

---

## 6. 累计结论延续与新增（继承 1–15，本期新增 16）

**累计结论（1–15 主线回顾）**：① 调度器隐式阻塞的定量画像；② 发版窗口的双向性（发版当天合入 ≠ 在版内）；③ 默认值成为主要风险载体（本期出现「正向默认化」反向印证）；④ 异构/量化后端静默错误为高频；⑤ 前缀缓存 × MoE 路由 × 投机解码交叉债务；⑥ 量化路径静默错误（scale/布局漏接）；⑦ 专家张量/路由归属与分页漏接静默错误；⑧ KV 卸载/HiCache/PD 健壮性；⑨ 长上下文 int32/int64 溢出静默错误；⑩ MXFP8 KV 正确性簇；⑪ 新旗舰模型 CP/KV/性能成为一等公民；⑫ DeepGEMM MegaMoE 量化后端成为新旗舰 MoE 落地主路径；⑬ FP8/低比特 KV cache 跨引擎标准化落地；⑭ 投机解码验证路径跨引擎密集硬化（DFlash/DSpark/DFLASH/ReplaySSM/lookahead）；⑮ 跨后端可移植性成为显式架构主线（Apple MPS/NPU/Hexagon/SpacemiT/ROCm/OOT）。

**新增第 16 条（新旗舰模型进入「默认性能化」阶段 + 量化/MegaMoE 通用化）**：本期 vLLM 把 GLM-5.3 已验证的 specialized 路径**默认开启**——`#67099`（batch<64 `latency_specialized` 默认，+27.0% E2E）、`#67104`（MTP row-pruning 默认绑定，+4.2% E2E）；同时把量化 MegaMoE 从 SM100 推到 SM90（vLLM `#64061`/`#62603`、SGLang `#49601`）、把 MoE 量化融合从 Qwen 扩展到 Llama/compressed-tensors（SGLang `#48263`）、把 NVFP4 MTP draft experts 量化通用化（SGLang `#48295`）。结论：新旗舰模型（GLM-5.3 / DeepSeek-V4.1 / Kimi-K3.1）的推理优化正从「加能力」转向「把已有能力默认化 + 把量化/MegaMoE 路径通用化到更多模型与硬件」，默认值的杠杆（第 3 条）继续放大——只是本期多为低风险的「免费性能默认」。

**可执行清单（P0/P1，本窗口观测）**：
- **P0 — 量化静默错误复核**：TRT-LLM `#22106`（W4A8_MXFP4_FP8 用 raw FC1 scale，否则 ~97% mismatch）、`#21346`（Qwen3.8 FP8 PLE 表 masked_fill_ 崩）；SGLang `#49851`（Kimi-K3.1-Pro MXFP8 KDA TP>1 失败）；vLLM `#66274`（Triton block-FP8 scalar scale 编译）。部署任何 MXFP4/FP8/NVFP4 MoE 或 attention-DP 前，必须逐项核验 scale/布局/量化 kernel 兼容性。
- **P0 — 量化 MoE 默认路径**：SGLang `#48263`（SiLU+NVFP4 量化融合扩展到 Llama/compressed-tensors）、`#48295`（MTP draft experts 转 NVFP4 省 KV）；vLLM `#64061`/`#62603`（MegaMoE 跨 SM90）。建议在目标模型/硬件上复测精度与吞吐，确认默认后端无回归。
- **P1 — KV 卸载/PD 健壮性**：vLLM `#66391`（大 tier setup 397–460s→135–138s）、`#67152`（partial-tail 缺块 TTFT 60.3→23.7ms）、`#67080`（Mooncake 失败 skip PUT，362 passed）、`#65448`（lookahead block hashes 全 connector）；SGLang `#50056`（HiCache prefetch trailing-window 信念分歧）；TRT-LLM `#21784`/`#20840`/`#21612`（KV pool/MoE workspace/NIXL 释放顺序）。
- **P1 — 跨后端/跨硬件**：SGLang `#47934`/`#48031`（Ascend A3 C4 state 按行传输、HiCache AIV）、`#44017`（XPU GDN 投机解码）；llama.cpp `#31873`（SYCL AOT >4GB 垃圾输出）、`#32046`（Adreno decomposed FA +41% decode）、`#32094`（Hexagon Q5_0/Q5_1 59/52 通过）、`#31897`（SpacemiT GDN RVV 1.61×）。
- **P1 — 部署安全（新增）**：llama.cpp `#32115`（CORS proxy/media 禁达内网防 SSRF）、`#32120`（禁浏览器发现 child instance）、`#32113`（非回环 opt-in tools）。若直接在本机/内网跑 `llama-server`，应确认已升级到含这些修复的版本。

---

## 7. 社区与视频资源（HN Algolia，2026-11-12 epoch 窗口 1794441600–1794528000）

> 以下均为社区/Show HN，非引擎官方，按「二手线索/未核验」记录；无四家官方更新类故事，也无引擎相关的可确认官方视频。

- **视频/教程**：本窗口 HN 未检索到与 vLLM/SGLang/llama.cpp/TensorRT-LLM 实现直接相关的官方教学视频；引擎关键词（vllm/llama.cpp/sglang/tensorrt-llm）的 relevance 检索在窗口内均无命中。
- **Show HN**：SCache and MOS – 32.9k O(1) KV cache (22.7 MiB) and 72B INT4/ASVD engine（1 分，2026-11-12T21:34:10Z）——KV cache + 72B INT4 引擎，KV 缓存/量化推理相关，**二手线索/未核验**（未核验其架构与性能声称）。
- **Show HN**：DeepSeek-Coder-V2-Lite Under 5GB: 65 TPS 100% In-VRAM on 6GB GPU（8 分，2026-11-12T05:07:43Z）——本地小显存推理，llama.cpp/GGUF 生态相关，**二手线索/未核验**。
- **Show HN/Tell HN**：Keeping names, emails and phone numbers out of LLM prompts with a local proxy（2026-11-12T00:09:39Z）——本地 LLM 隐私代理，与本地推理部署相关，**二手线索/未核验**。
- **资讯（泛 AI 推理）**：Axelera AI Thinks Titania Is a Better Way to Scale AI Inference（1 分，2026-11-12T23:31:10Z）、Why Is Nvidia Valuable?（2 分，2026-11-12T22:25:14Z）——AI 推理扩展/算力，与推理硬件生态相关，**二手线索/未核验**。
- 其余 HN 命中（共 1106 条）多为泛 AI/科技资讯，与推理引擎实现无直接关系，未列入。

---

## 8. 数据质量与采集说明

- **采集完整性**：四家 commits API 均 HTTP 200；vLLM 94 / SGLang 89 / TRT-LLM 39 / llama.cpp 39 合计 261 条（2026-11-12 UTC 窗口）；四家单页均未触 per_page=100 上限，无截断。
- **版本核验**：releases API 确认窗口内无新正式版本；llama.cpp 仅推进 nightly `b12323`–`b12336`。
- **PR 正文来源**：42 个关键 PR 正文经官方 pulls API（markdown 官方一手）抓取，API 配额充足、未触发 60/h 限制、未走 HTML 页面抓取；其余 PR 仅标题级/描述级者已在各节「其他要点」标注缺失程度（如 TRT-LLM 大量 `nvbugs` 内部链接 PR 仅能凭标题/描述记录、无法核验内部 QA 状态，维持「描述级/未核实实现落地」口径；llama.cpp `#32067` 正文未完整抓取记为描述级）。
- **描述级/未全绿标注**：TRT-LLM `#21695`（长上下文 decode 其余对照数值未完整）、`#21784`/`#21612`（评审标注 coverage needs follow-up、硬件资格未跑）；SGLang `#48602`/`#49980`（移植自 Day-0 分支、benchmark 未独立重跑需谨慎）；llama.cpp `#32046`/`#29357`（perf 数值部分截断）；vLLM `#60321`（mxfp4_mla 容量声明 + 单测、无端到端提速声称）等均已按实际缺失程度标注。
- **社区信号**：HN Algolia 引擎官方更新类故事无；社区侧 Show HN（SCache/MOS、DeepSeek-Coder-V2-Lite 6GB）与本地推理隐私代理按二手线索/未核验记录。
- **交叉核验优先级**：官方仓库/API/文档 > 官方博客/发布页 > 可信二手报道 > 社区转述；本档案对无法由一手来源确认的内容已显式标注「二手线索/未核验/描述级」。
