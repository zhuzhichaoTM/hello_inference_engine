# L4 · 端侧 CPU、GPU、NPU、DSP 与 Kernel 优化

本层承载 ARM/x86/RISC-V CPU、移动/集成 GPU、NPU/DSP 执行模型、Kernel、编译器与端侧性能剖析资料。

**在业务地图中的位置**：对应[端侧大模型推理业务地图](../端侧大模型推理业务地图.md) §1.7；支撑业务流程 03 硬件与推理后端适配、09 性能分析与优化，以及工程闭环的「3 瓶颈定位」「4 优化实施」。

## 关联入口

- [CUDA 后端剖析（ggml）](../L2/framework/ggml/backend_cuda/)
- [端侧大模型量化系统设计](../L2/quant/端侧大模型量化系统设计文档.md)

## 建议沉淀

- ARM NEON/i8mm、x86 AVX/AMX、RISC-V RVV 等 SIMD 与 CUDA/NPU Kernel
- 移动 GPU（Adreno/Mali/Apple GPU）与 NPU/DSP 算子覆盖矩阵
- Roofline、带宽、占用率、能效（tokens/Joule）与 Profile 报告
- 不同硬件、布局、精度和后端的性能与功耗对比
