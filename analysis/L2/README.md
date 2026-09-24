# L2 · Tensor、算子、计算图、精度与量化

本层承载 Tensor 布局、核心算子、计算图、精度、量化算法和量化误差资料。

## 现有入口

- [ggml 量化源码详细分析](../quant/ggml/ggml-quants源码详细分析.md)
- [Q4_K_M 量化分析与 C 参考实现](../quant/ggml/ggml-quants_q4_k_m_analysis.md)
- [Q4_K_M 可视化与调试](../quant/ggml/q4km_debug/)
- [端侧大模型量化系统设计](../quant/端侧大模型量化系统设计文档.md)

## 建议沉淀

- 算子契约、reference/kernel 一致性测试
- F16/BF16/INT8/INT4 精度与性能对比
- 计算图优化、内存复用和量化方案分析
