# L2 · 端侧 Tensor、算子、计算图、精度与量化

本层承载 Tensor 布局、核心算子、计算图、精度、量化算法和量化误差资料，并覆盖 ggml 框架与 CUDA 后端的源码级剖析。

**在业务地图中的位置**：对应[端侧大模型推理业务地图](../端侧大模型推理业务地图.md) §1.5；支撑业务流程 03 硬件与推理后端适配、04 质量评测与验收、09 性能分析与优化，以及工程闭环的「3 瓶颈定位」「4 优化实施」。

## 现有入口

### 量化

- [ggml 量化源码详细分析](quant/ggml/ggml-quants源码详细分析.md)
- [Q4_K_M 量化分析与 C 参考实现](quant/ggml/ggml-quants_q4_k_m_analysis.md)
- [Q4_K_M 可视化与调试](quant/ggml/q4km_debug/)
- [llama.cpp 量化工具链分析](quant/llama_cpp/)
- [端侧大模型量化系统设计](quant/端侧大模型量化系统设计文档.md)

### 框架与后端

- [ggml 系统架构与源码解析](framework/ggml/)
- [ggml CUDA 后端剖析与题库](framework/ggml/backend_cuda/)

## 建议沉淀

- 算子契约、reference/kernel 一致性测试
- F16/BF16/INT8/INT4/Q4_K_M 精度与性能对比
- 计算图优化、内存复用和量化方案分析
- 算子融合、后端支持矩阵与 CPU fallback
