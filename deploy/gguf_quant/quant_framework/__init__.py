"""
量化转换框架入口 (__init__.py)

模块划分：
  quant_config       - 超参数、枚举、配置校验、外部配置导入导出
  quant_utils        - 跨模块共享工具（子进程执行、日志、目录）
  imatrix_generator  - 专责 imatrix.gguf 的生成/确认（四种模式：none, dataset, self_gen, existing）
  gguf_quantizer     - 专责 BaseGGUFConverter 与 GGUFQuantizer（支持 1bit 到 16bit）
  quant_pipeline     - 编排器：串联两引擎，提供一键自动化全流程
  quant_cli          - 命令行 CLI 入口
"""

from .quant_config import QuantConfig, ImatrixMode, QuantType
from .imatrix_generator import ImatrixGenerator
from .gguf_quantizer import GGUFQuantizer, BaseGGUFConverter
from .quant_pipeline import QuantPipeline

__all__ = [
    "QuantConfig",
    "ImatrixMode",
    "QuantType",
    "ImatrixGenerator",
    "GGUFQuantizer",
    "BaseGGUFConverter",
    "QuantPipeline",
]
