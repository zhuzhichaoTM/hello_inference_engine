# 大模型推理优化每日热点档案

**档案日期：** 2026-10-02（北京时间）  
**关注窗口：** 2026-10-01 00:00–23:59（前一日；官方 API 按 UTC 日期过滤）  
**范围：** 推理引擎、KV Cache、Prefill/Decode 解耦、投机解码、量化、异构硬件与运行时稳定性  
**数据质量说明：** 本期确定事实主要来自 vLLM、SGLang、TensorRT-LLM、llama.cpp 官方 GitHub 提交 API。提交时间以官方 API 返回值为准，档案日期按北京时间记录。未将无法独立复核的社区帖子、视频摘要或二手报道当作确定事实。SkillHub 查询因 TLS 协议错误未完成，未据此引入外部技能。

## 1. vLLM：NIXL KV 生命周期、HiSparse/NVFP4 与 MTP/MLA 推进

**主题**  vLLM 在 10 月 1 日的官方提交集中在 KV Connector 生命周期、HiSparse、MiniMax-M3 NVFP4 KV cache、Model Runner V2 的多层 MTP，以及 ROCm/MLA 路径。

**内容详情**  官方提交显示：
- NIXL KV Connector 增加 host-buffer KV copies 的 coalesce，并统计 KV expiry 之后到达的 completion notifications；
- HiSparse 修复 chunked-prefill preemption livelock，并优化 prefix scans/residency updates；
- MiniMax-M3 在 MSA sparse attention 路径加入 NVFP4 KV cache；
- Model Runner V2 为多层 MTP 支持 per-module LM heads，并加入 sampling mask replay 与 randomized dummy inputs；
- Kimi-K3 增加 FlashInfer speculative KDA backend，ROCm 路径继续修复 MLA prefill scheduling metadata race；
- GLM 5.3 Perf 提交记录跳过 MHA 的 qlnorm calculation，提交标题声称 E2E TTFT 改善 4.4–7.7%，该数字仅作为官方提交标题中的特定测试声明，不能外推为普遍性能。

**应用场景**  分离式 KV cache、主机缓冲区搬运、HiSparse 长上下文、MiniMax-M3/NVFP4、MTP 投机解码、Kimi-K3/ROCm 与 GLM 5.3 性能调优。

**研判**  **事实：** 本期同时涉及 KV 传输完成语义、缓存驻留、MTP 输出头和后端 kernel 调度。**观点：** KV cache 系统的关键边界正在从“是否传输”扩展到“过期后完成如何记账、搬运如何合并、驻留如何更新”；性能标题应配套记录模型、硬件、batch、精度和测试脚本，避免把单次 benchmark 当作通用收益。

**来源**
- vLLM 官方提交 API（2026-10-01 UTC）：https://api.github.com/repos/vllm-project/vllm/commits?since=2026-10-01T00:00:00Z&until=2026-10-02T00:00:00Z&per_page=100
- NIXL completion after KV expiry：https://github.com/vllm-project/vllm/commit/ea84051819
- NIXL host-buffer copy coalescing：https://github.com/vllm-project/vllm/commit/47de9d4d04
- HiSparse prefix/residency 优化：https://github.com/vllm-project/vllm/commit/ca65eb67d9
- MiniMax-M3 NVFP4 KV cache：https://github.com/vllm-project/vllm/commit/7566d83bd3
- MTP per-module LM heads：https://github.com/vllm-project/vllm/commit/c4fc3c7d32
- GLM 5.3 qlnorm 优化提交：https://github.com/vllm-project/vllm/commit/4d6874d137

**证据等级：** 官方一手；性能数字仅为官方提交标题中的特定声明。

## 2. SGLang：PD/HiCache 约束、DSpark/投机解码与 AMD 异构路径加速

**主题**  SGLang 官方提交强化了分离式 Prefill/Decode 的路由与传输校验，同时推进 HiCache、DSpark、FlashMLA、AMD gfx950 和 GLM-5.3-Flash serving。

**内容详情**  可核验提交包括：
- sgl-router 强制兼容的 PD version groups，prefill 失败时 fail fast 并取消 decode；
- disaggregated 路径在 Mooncake transfer 前校验 state strides；HiCache/PD 修复 decode load-back polling 避免进入 rank-local collectives；
- gfx950 上启用 GLM-5.3-Flash FP8/MXFP4 serving，并支持 DeepSeek V4.1 的 FP8 unified KV 与 PD-disagg；
- 在 ROCm CUDA graph 内构建 DSpark draft metadata，增加 greedy draft/accept 路径；
- DFlash 修复 grouped-conv taps 跨请求 NaN/Inf 泄漏，DCP/FlashMLA 继续推进 Kimi-K3 与 DeepSeek 路径；
- 为 Mamba2 SSD kernel 增加 Triton autotune，并为 DSA indexer 声明 HiCache host pools。

**应用场景**  Prefill/Decode 解耦、Mooncake/NIXL 类 KV 传输、HiCache 分层缓存、GLM-5.3-Flash 与 DeepSeek V4.1 的 AMD 部署、DSpark/DFlash 投机解码。

**研判**  **事实：** 本期变更把 PD 路由版本、传输前 state stride、decode load-back collective 边界和 draft metadata capture 都纳入工程实现。**观点：** P/D 解耦的可靠性不仅取决于网络吞吐，还取决于版本、状态布局和失败取消语义；上线前应把 prefill failure、跨版本 worker、partial transfer 和 decode abort 作为一等回归场景。

**来源**
- SGLang 官方提交 API（2026-10-01 UTC）：https://api.github.com/repos/sgl-project/sglang/commits?since=2026-10-01T00:00:00Z&until=2026-10-02T00:00:00Z&per_page=100
- GLM-5.3-Flash FP8/MXFP4 on gfx950：https://github.com/sgl-project/sglang/commit/f17f7705a5
- PD version groups：https://github.com/sgl-project/sglang/commit/41cbe65de0
- prefill failure cancels decode：https://github.com/sgl-project/sglang/commit/3b2ad1c6ae
- state stride validation：https://github.com/sgl-project/sglang/commit/1a0250c5b6
- DSpark CUDA-graph metadata：https://github.com/sgl-project/sglang/commit/f45c004e18
- HiCache host pools：https://github.com/sgl-project/sglang/commit/573b3caad3

**证据等级：** 官方一手。

## 3. TensorRT-LLM：稀疏 KV offload 与 KV retirement 安全边界

**主题**  TensorRT-LLM 在 10 月 1 日的公开提交较少，但与推理基础设施直接相关的变化集中在 DeepSeek V4 sparse KV offload、KV retirement deadline 和运行时配置。

**内容详情**  官方提交显示：
- 为 DeepSeek V4 attention 准备 sparse KV offload；
- 对未被证明安全的 KV retirement 增加不可重置 deadline；
- 允许通过 CLI 配置覆盖项，改善部署时的参数注入；
- 对 OpenEngine Python bindings 进行 vendor；
- GenerationRequest 接受 flat int32 array，避免先物化为 list。

**应用场景**  稀疏注意力与 KV offload、长上下文缓存回收、服务启动参数治理和高频请求对象构造。

**研判**  **事实：** 官方提交把 KV offload 能力和 KV retirement 的安全截止时间同时推进。**观点：** KV 回收策略应优先保证不会因状态不确定而无限期占用资源；deadline 设计需要结合 late completion、失败重试、服务重启和可观测性验证，不能只看显存占用下降。

**来源**
- TensorRT-LLM 官方提交 API（2026-10-01 UTC）：https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits?since=2026-10-01T00:00:00Z&until=2026-10-02T00:00:00Z&per_page=100
- DeepSeek V4 sparse KV offload：https://github.com/NVIDIA/TensorRT-LLM/commit/cc71759303
- KV retirement deadline：https://github.com/NVIDIA/TensorRT-LLM/commit/6f43d7edab
- CLI config overrides：https://github.com/NVIDIA/TensorRT-LLM/commit/80509acfc0
- flat int32 prompt：https://github.com/NVIDIA/TensorRT-LLM/commit/95cb50c358

**证据等级：** 官方一手；未据提交标题推导尚未公布的性能收益。

## 4. llama.cpp：MTP、NVFP4、递归内存与多后端稳定性

**主题**  llama.cpp 在 10 月 1 日的官方提交覆盖 Qwen4Exp MTP、CUDA NVFP4 计算类型、递归内存断言、Metal/Hexagon/WebGPU 与直接 I/O 路径。

**内容详情**  官方提交显示：
- 为 Qwen4Exp 增加 MTP，并修复相关模型路径；
- CUDA cuBLAS 路径处理 NVFP4 compute type；Metal 为 MXFP4 mul-mat 使用 BF16 math；
- 修复 recurrent memory 的 invalid assert，以及 WebGPU SSM_SCAN binding aliasing；
- Hexagon 增加 q2_k/q3_k 量化类型与共享 DMA copy，WebGPU 增加 BF16 支持；
- llama-mmap direct-io 避免每个 tensor 的第二次全量拷贝，CUDA 修复 Volta FA cases；
- b11325–b11331 等 nightly releases 在 10 月 1–2 日发布，但本期仅记录为版本可追踪线索，未将其当作独立性能结论。

**应用场景**  本地 MTP、NVFP4/MXFP4 量化、递归状态模型、Metal/CUDA/WebGPU/Hexagon 异构部署与低内存加载。

**研判**  **事实：** llama.cpp 同日覆盖 MTP、低精度计算、递归状态和多后端边界修复。**观点：** 本地运行时的优化通常伴随后端差异；接入 MTP 或新量化格式时，应按后端分别验证状态模型、短 batch、模型加载峰值和输出一致性。

**来源**
- llama.cpp 官方提交 API（2026-10-01 UTC）：https://api.github.com/repos/ggml-org/llama.cpp/commits?since=2026-10-01T00:00:00Z&until=2026-10-02T00:00:00Z&per_page=100
- Qwen4Exp MTP：https://github.com/ggml-org/llama.cpp/commit/c061df1983
- NVFP4 cuBLAS compute type：https://github.com/ggml-org/llama.cpp/commit/b56f34ab13
- recurrent memory assert：https://github.com/ggml-org/llama.cpp/commit/869034b4bb
- direct-io tensor copy：https://github.com/ggml-org/llama.cpp/commit/32dd62ee6d
- 官方 releases：https://github.com/ggml-org/llama.cpp/releases

**证据等级：** 官方一手；nightly release 仅为版本线索。

## 5. 当前会话已形成的累计内容结论

1. **KV Cache 正在成为带生命周期证明的分布式资源**：前期已记录 vLLM KV Connector、SGLang HiCache/分层路径、跨节点迁移、TensorRT-LLM verified write coverage 与 ownership；本期新增 completion-after-expiry 统计、host-buffer copy coalescing、sparse KV offload 和 retirement deadline。
2. **P/D 解耦需要协议级失败语义**：前期关注 P/D 路由、Radix Cache 和 Dspark coordination；本期补充 PD version groups、state stride 校验、prefill failure cancel decode 与 load-back collective 边界。
3. **投机解码的收益依赖状态、图和后端共同稳定**：MTP、DSpark/DFlash、EAGLE3、KDA 与 llama.cpp MTP 的连续推进说明，draft/accept 逻辑、batch 形状、状态提交和异构 kernel 需要联合回归。
4. **量化结论必须绑定硬件与实现路径**：NVFP4/MXFP4/FP8 在 vLLM、SGLang、TensorRT-LLM 和 llama.cpp 中分别对应不同 kernel、KV 格式和设备路径，不能仅按精度名称比较。

## 6. 今日可执行清单

1. 为分离式 KV cache 增加 expiry-after-completion、copy coalescing、retirement deadline、partial transfer 与重启恢复指标。
2. 为 PD 部署增加 version group、state stride、prefill failure、decode cancel 和 load-back collective 的故障注入测试。
3. 为 MTP/DSpark/DFlash 建立跨 PP、draft 长度、stream overlap、后端和空/短 batch 的一致性回归。
4. 对 GLM-5.3-Flash、DeepSeek V4.1、MiniMax-M3 等模型分别记录 GPU、dtype、KV dtype、kernel、并发度与评测集，不把提交标题中的性能数字外推。
5. 对 llama.cpp 的 MTP、NVFP4/MXFP4 和多后端路径执行内存峰值、递归状态、短 batch、模型加载及输出一致性 smoke test。

## 7. 社区与视频资料

本期未将无法由官方一手来源确认发布日期和内容的社区帖子、视频或二手报道纳入确定热点。官方 GitHub API 可访问；SkillHub 查询因 TLS 协议错误未完成，未据此推断技能可用性。

---

**归档状态：** 已按 `hotspot/2026-10/` 月度目录归档。  
**证据边界：** 事实与观点已分开；官方提交中的性能标题、nightly 版本和测试/配置变更均按原文范围记录，不推导为普遍结论。
