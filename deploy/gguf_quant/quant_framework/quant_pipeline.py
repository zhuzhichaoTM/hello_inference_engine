"""
量化流水线编排器 (quant_pipeline.py)
职责：按配置顺序串联 BaseGGUFConverter, ImatrixGenerator 与 GGUFQuantizer，
严格对齐交付文件清单与全流程自动化执行。
"""

import os
import logging
from typing import Dict, Optional

from .quant_config import QuantConfig, ImatrixMode
from .quant_utils import ensure_dirs
from .imatrix_generator import ImatrixGenerator
from .gguf_quantizer import GGUFQuantizer, BaseGGUFConverter

logger = logging.getLogger("QuantPipeline")


class QuantPipeline:
    """
    量化流水线编排器。
    """

    def __init__(self, config: QuantConfig):
        config.validate()
        self.cfg = config
        ensure_dirs(config.output_dir)

        self.converter = BaseGGUFConverter(config, config.output_dir)
        self.imatrix_gen = ImatrixGenerator(config, base_gguf_path="", work_dir=config.output_dir)
        self.quantizer = GGUFQuantizer(config, config.output_dir)

        # 暴露子目录路径
        self.dir_base = self.converter.dir_base
        self.dir_calib = self.imatrix_gen.dir_calib
        self.dir_quant = self.quantizer.dir_quant
        self.dir_logs = self.quantizer.dir_logs
        self.base_gguf_path = self.converter.base_gguf_path

        self.imatrix_path: Optional[str] = None
        self.generated_quants: Dict[str, str] = {}

    def check_prerequisites(self) -> bool:
        logger.info("=== 阶段 0: 原生模型与工具链环境校验 ===")
        return self.converter.check_prerequisites()

    def run_base_conversion(self, outtype: str = "bf16") -> str:
        logger.info(f"=== 阶段 1: 基础转换 (输入 -> {outtype.upper()} GGUF) ===")
        self.base_gguf_path = self.converter.convert(outtype=outtype)
        return self.base_gguf_path

    def run_imatrix_generation(self, base_gguf_path: str) -> Optional[str]:
        logger.info(f"=== 阶段 2: imatrix 重要性矩阵生成 (模式: {self.cfg.imatrix_mode.value}) ===")
        self.imatrix_gen.base_gguf_path = base_gguf_path
        self.imatrix_path = self.imatrix_gen.generate()
        return self.imatrix_path

    def run_quantization(
        self,
        base_gguf_path: str,
        imatrix_path: Optional[str] = None,
        is_ud_mode: bool = False
    ) -> Dict[str, str]:
        logger.info(f"=== 阶段 3: 批量量化转换 (目标: {[q.value for q in self.cfg.target_quants]}) ===")
        quants = self.quantizer.quantize_all(base_gguf_path, imatrix_path, is_ud_mode=is_ud_mode)
        self.generated_quants.update(quants)
        return self.generated_quants

    def run_cleanup(self) -> None:
        if self.cfg.clean_intermediate_base and os.path.exists(self.base_gguf_path) and not self.cfg.is_input_gguf():
            logger.info(f"清理基准中间文件: {self.base_gguf_path}")
            os.remove(self.base_gguf_path)

    def run_all(self) -> Dict[str, str]:
        """一键全流程执行"""
        logger.info(f">>> 启动 {self.cfg.model_name} 自动化量化流水线 <<<")
        self.check_prerequisites()
        base_gguf = self.run_base_conversion()
        imx = self.run_imatrix_generation(base_gguf)
        is_ud = (self.cfg.imatrix_mode == ImatrixMode.SELF_GENERATED)
        results = self.run_quantization(base_gguf, imatrix_path=imx, is_ud_mode=is_ud)
        self.run_cleanup()
        logger.info(">>> 流水线执行完毕 <<<")
        return results
