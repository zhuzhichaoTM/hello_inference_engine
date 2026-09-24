# L3 · 推理运行时、内存、KV Cache 与生成

本层承载模型加载、图执行、内存规划、KV Cache、Prefill/Decode、采样和流式生成资料。

## 现有入口

- [llama.cpp 量化执行流程分析](../quant/llama_cpp/llama-quantize-impl-flow.md)
- [llama.cpp 量化工具分析](../quant/llama_cpp/llama-quantize-tool-analysis.md)
- [端侧大模型量化系统设计](../quant/端侧大模型量化系统设计文档.md)

## 建议沉淀

- 单请求 Prefill/Decode 与 KV Cache 实践
- 内存峰值、Cache 复用/淘汰和采样器报告
- 运行时错误、取消和降级路径分析
