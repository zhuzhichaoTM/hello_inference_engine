"""
perfkit · 大模型推理性能分析与优化框架
==============================================================================
基于「端侧大模型推理业务地图」第三部分 · 工程闭环地图（7 步曲）固化的通用框架。

七步闭环 ↔ 子命令映射：
    0 环境准备   （由 configs/ 的 device/model 画像承载，不单独设子命令）
    1 问题定义   perfkit define      画像 / 显存预算 / roofline / KPI
    2 性能基线   perfkit baseline    replay 读台账 或 live 真机采集
    3 瓶颈定位   perfkit profiling   阶段 / 逐层 / 算子 / 硬件 四维剖面
    4 优化实施   perfkit optimize    方案筛选 / bundle 组合 / 启动脚本
    5 收益量化   perfkit verify      前后对比 / 收益归因 / 副作用清单
    6 回归验收   perfkit report      七步闭环报告导出
    7 可视化     perfkit visualize   自包含 charts.html

通用性设计：换模型 / 换硬件 / 换任务目标时，
只改 configs/ 下的 YAML，七步流程代码零改动。
"""

__version__ = "1.0.0"
__all__ = ["config"]
