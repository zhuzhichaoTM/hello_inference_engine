"""
GGUF 量化与转换引擎 (gguf_quantizer.py)
专责：
  1. BaseGGUFConverter: 将 safetensors 转换为 BF16/FP16 GGUF，或校验已输入的 GGUF。
  2. GGUFQuantizer: 以基准 GGUF 为输入，批量产出 1-bit 到 16-bit 目标档位 GGUF 模型。
"""

import os
import glob
import logging
from typing import Dict, List, Optional

from .quant_config import QuantConfig, QuantType
from .quant_utils import ensure_dirs, run_command

logger = logging.getLogger("GGUFQuantizer")

# 必须依赖 imatrix 的低比特档位
_IMATRIX_REQUIRED_TYPES = {
    QuantType.IQ1_S, QuantType.IQ1_M,
    QuantType.IQ2_XXS, QuantType.IQ2_XS, QuantType.IQ2_S, QuantType.IQ2_M,
    QuantType.IQ3_XXS, QuantType.IQ3_S, QuantType.IQ3_M,
    QuantType.IQ4_XS, QuantType.IQ4_NL,
}


class GGUFQuantizer:
    """
    基准 GGUF (BF16/F16) → 多档位低比特量化批量转换器。
    """

    def __init__(self, config: QuantConfig, work_dir: str):
        self.cfg = config
        self.dir_quant = os.path.join(work_dir, config.dir_name_quant)
        self.dir_logs = os.path.join(work_dir, config.dir_name_logs)
        ensure_dirs(self.dir_quant, self.dir_logs)

        self.generated: Dict[str, str] = {}

    def quantize_all(
        self,
        base_gguf_path: str,
        imatrix_path: Optional[str] = None,
        is_ud_mode: bool = False,
    ) -> Dict[str, str]:
        """
        批量量化生成指定的所有低比特模型。
        :param base_gguf_path: 输入的 BF16/F16 基准 GGUF
        :param imatrix_path: 可选重要性矩阵文件
        :param is_ud_mode: 是否输出 UD 优化标记命名
        """
        if not os.path.exists(base_gguf_path):
            raise FileNotFoundError(f"基准 GGUF 输入不存在: {base_gguf_path}")

        effective_imatrix = self._resolve_imatrix(imatrix_path)
        self._warn_iq_without_imatrix(effective_imatrix)

        quant_bin = os.path.join(self.cfg.llama_cpp_bin_dir, "llama-quantize")
        targets = [q.value for q in self.cfg.target_quants]
        logger.info(
            f"=== GGUFQuantizer: 目标档位 {targets} | "
            f"imatrix={'启用 (' + str(effective_imatrix) + ')' if effective_imatrix else '未启用(标准量化)'} | "
            f"UD模式={'是' if is_ud_mode else '否'} ==="
        )

        for q_type in self.cfg.target_quants:
            # 命名对齐最终交付清单
            if is_ud_mode:
                filename = f"{self.cfg.model_name}-UD-{q_type.value}.gguf"
            elif effective_imatrix:
                filename = f"{self.cfg.model_name}-{q_type.value}-imatrix.gguf"
            else:
                filename = f"{self.cfg.model_name}-{q_type.value}.gguf"

            out_file = os.path.join(self.dir_quant, filename)
            cmd = self._build_cmd(quant_bin, base_gguf_path, out_file, q_type, effective_imatrix)

            logger.info(f"正在量化: {filename} ...")
            run_command(cmd, log_path=os.path.join(self.dir_logs, f"quant_{q_type.value}.log"))

            if not os.path.exists(out_file):
                raise FileNotFoundError(f"量化文件未生成: {out_file}")

            size_gb = os.path.getsize(out_file) / (1024 ** 3)
            logger.info(f"✅ 生成成功: {filename} ({size_gb:.2f} GB)")

            # 使用唯一 key 区分
            res_key = f"UD_{q_type.value}" if is_ud_mode else (f"{q_type.value}_imatrix" if effective_imatrix else q_type.value)
            self.generated[res_key] = out_file

        return self.generated

    def _resolve_imatrix(self, imatrix_path: Optional[str]) -> Optional[str]:
        if imatrix_path is None:
            return None
        if not os.path.exists(imatrix_path):
            logger.warning(f"imatrix 路径无效 ({imatrix_path})，降级为标准量化。")
            return None
        return imatrix_path

    def _warn_iq_without_imatrix(self, imatrix_path: Optional[str]) -> None:
        if imatrix_path is not None:
            return
        risky = [q.value for q in self.cfg.target_quants if q in _IMATRIX_REQUIRED_TYPES]
        if risky:
            logger.warning(
                f"⚠️ 档位 {risky} 强依赖 imatrix 张量重要性感知。"
                "当前未激活 imatrix，该档位量化质量将受损，建议提供校准数据后使用。"
            )

    def _build_cmd(
        self,
        quant_bin: str,
        base_path: str,
        out_file: str,
        q_type: QuantType,
        imatrix_path: Optional[str],
    ) -> List[str]:
        cmd = [quant_bin]

        if imatrix_path:
            cmd.extend(["--imatrix", imatrix_path])

        if self.cfg.enable_layered_overrides and self.cfg.tensor_overrides:
            for pattern, override_type in self.cfg.tensor_overrides.items():
                cmd.extend(["--override-tensor", f"{pattern}={override_type}"])

        cmd.extend([base_path, out_file, q_type.value])
        return cmd


class BaseGGUFConverter:
    """
    基准 GGUF 转换与前置校验器。
    支持：
      1. 直接输入已有的 BF16/F16 GGUF（直接校验放行）
      2. 输入 safetensors 权重目录（调用 convert_hf_to_gguf.py 转换输出 BF16/F16 GGUF）
    """

    def __init__(self, config: QuantConfig, work_dir: str):
        self.cfg = config
        self.dir_base = os.path.join(work_dir, config.dir_name_base_gguf)
        self.dir_logs = os.path.join(work_dir, config.dir_name_logs)
        ensure_dirs(self.dir_base, self.dir_logs)

        # 默认产出基准文件路径
        self.base_gguf_path = os.path.join(self.dir_base, f"{self.cfg.model_name}-BF16.gguf")

    def check_prerequisites(self) -> bool:
        """校验输入路径的有效性"""
        p = self.cfg.model_input_path
        if not os.path.exists(p):
            raise FileNotFoundError(f"输入路径不存在: {p}")

        if self.cfg.is_input_gguf():
            logger.info(f"输入为已有 GGUF 文件: {p} ({os.path.getsize(p)/(1024**3):.2f} GB)，校验通过。")
            self.base_gguf_path = p
            return True

        # safetensors 分片校验
        weights = glob.glob(os.path.join(p, "*.safetensors"))
        if not weights:
            weights = glob.glob(os.path.join(p, "*.bin"))
        if not weights:
            raise FileNotFoundError(f"未在 {p} 下找到权重文件 (*.safetensors 或 *.bin)")

        logger.info(f"扫描到 {len(weights)} 个权重分片，safetensors 输入校验通过。")
        return True

    def convert(self, outtype: str = "bf16") -> str:
        """
        若输入是 GGUF 则直接复用；若输入是 safetensors 则转为 GGUF。
        """
        if self.cfg.is_input_gguf():
            logger.info(f"输入已是 GGUF 模型，无需执行转换: {self.cfg.model_input_path}")
            return self.cfg.model_input_path

        # 检查是否已转换过
        if os.path.exists(self.base_gguf_path) and os.path.getsize(self.base_gguf_path) > 100 * 1024 * 1024:
            logger.info(f"发现已有基准 GGUF ({os.path.getsize(self.base_gguf_path)/1e9:.2f} GB)，跳过重复转换。")
            return self.base_gguf_path

        import sys
        script = os.path.join(self.cfg.llama_cpp_src_dir, "convert_hf_to_gguf.py")
        cmd = [
            sys.executable, script,
            self.cfg.model_input_path,
            "--outfile", self.base_gguf_path,
            "--outtype", outtype,
            "--split-max-size", "0"
        ]
        run_command(cmd, log_path=os.path.join(self.dir_logs, "convert_base_gguf.log"))
        logger.info(f"基准 GGUF 转换完成: {self.base_gguf_path}")
        return self.base_gguf_path
