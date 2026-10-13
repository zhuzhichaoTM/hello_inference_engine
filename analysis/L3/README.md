# L3 · 端侧推理运行时、内存、KV Cache 与生成

本层承载模型加载、图执行、内存规划、KV Cache、Prefill/Decode、采样和流式生成资料，重点是在端侧内存约束下不 OOM 地稳定运行。

**在业务地图中的位置**：对应[端侧大模型推理业务地图](../端侧大模型推理业务地图.md) §1.6；支撑业务流程 06 在线请求处理、08 内存与缓存治理、09 性能分析与优化，以及工程闭环的「2 性能基线」「3 瓶颈定位」「4 优化实施」。

## 现有入口

- [llama.cpp 量化执行流程分析](../L2/quant/llama_cpp/llama-quantize-impl-flow.md)
- [llama.cpp 量化工具分析](../L2/quant/llama_cpp/llama-quantize-tool-analysis.md)
- [端侧大模型量化系统设计](../L2/quant/端侧大模型量化系统设计文档.md)

## 建议沉淀

- 单请求 Prefill/Decode 与 KV Cache 实践（含 KV Cache 量化/淘汰）
- 端侧内存峰值、统一内存、逐层加载与降级策略
- 采样与生成控制、投机解码与真机时延/功耗报告
- 运行时错误、取消和降级路径分析
