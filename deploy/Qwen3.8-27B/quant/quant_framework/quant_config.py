"""
通用大模型低比特量化框架配置模块 (quant_config.py)
定义量化转换超参数、数据类与枚举类型，支持 1-bit 到 16-bit 全范围量化。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import os
import json


class ImatrixMode(Enum):
    """重要性矩阵（imatrix）生成模式"""
    NONE = "none"                  # 不使用 imatrix，执行标准量化
    DATASET = "dataset"            # 外部语料库（如 WikiText-2, Alpaca 或自定义文本）
    SELF_GENERATED = "self_gen"    # unsloth 风格：由原始模型自身自回归采样生成
    EXISTING = "existing"          # 使用已存在的 imatrix 文件


class QuantType(Enum):
    """
    支持 1-bit 到 16-bit 的全系列量化档位枚举 (llama.cpp 原生支持)
    """
    # ── 1-bit 极限压缩 ──
    IQ1_S = "IQ1_S"          # 1.56 bpw 极低比特
    IQ1_M = "IQ1_M"          # 1.75 bpw

    # ── 2-bit 超低比特 ──
    IQ2_XXS = "IQ2_XXS"      # 2.06 bpw
    IQ2_XS = "IQ2_XS"        # 2.31 bpw
    IQ2_S = "IQ2_S"          # 2.50 bpw
    IQ2_M = "IQ2_M"          # 2.70 bpw
    Q2_K = "Q2_K"            # 2-bit K-quant
    Q2_K_S = "Q2_K_S"

    # ── 3-bit 档位 ──
    IQ3_XXS = "IQ3_XXS"      # 3.06 bpw
    IQ3_S = "IQ3_S"          # 3.44 bpw
    IQ3_M = "IQ3_M"          # 3.66 bpw
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"

    # ── 4-bit 常用档位（甜点档） ──
    IQ4_NL = "IQ4_NL"        # 4.50 bpw 非线性
    IQ4_XS = "IQ4_XS"        # 4.25 bpw
    Q4_0 = "Q4_0"            # 经典 4-bit 对称
    Q4_1 = "Q4_1"            # 经典 4-bit 非对称
    Q4_K_S = "Q4_K_S"        # 紧凑型 4-bit K-quant
    Q4_K_M = "Q4_K_M"        # ⭐ 推荐甜点档（精度与体积最佳平衡）

    # ── 5-bit 档位 ──
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"        # 5-bit 高精度档

    # ── 6-bit 档位 ──
    Q6_K = "Q6_K"            # 6-bit 近无损档

    # ── 8-bit 档位 ──
    Q8_0 = "Q8_0"            # 8-bit 工业基准档

    # ── 16-bit 浮点基准 ──
    F16 = "F16"              # 半精度浮点
    BF16 = "BF16"            # 脑浮点


@dataclass
class QuantConfig:
    """量化管道总配置超参数，支持灵活配置与外部序列化导入"""
    # 路径配置
    model_input_path: str                          # 原始模型路径 (safetensors 目录 或 已有的 BF16/F16 GGUF 文件)
    output_dir: str                                # 量化结果与交付物根目录
    model_name: str = "Qwen3.8-27B"                # 模型名称前缀
    llama_cpp_bin_dir: str = "/root/autodl-tmp/llama.cpp/build/bin" # llama.cpp 编译产物目录
    llama_cpp_src_dir: str = "/root/autodl-tmp/llama.cpp"           # llama.cpp 源码根目录

    # 交付目录子路径自定义（严格对齐 CLAUDE.md 交付清单）
    dir_name_base_gguf: str = "bf16_gguf"          # BF16/F16 基准 GGUF 目录
    dir_name_quant: str = "gguf"                   # 最终量化模型输出目录
    dir_name_calib: str = "calibration"            # 校准数据集目录
    dir_name_logs: str = "logs"                    # 日志目录

    # 目标量化档位列表（支持 1bit 到 16bit 任意档位）
    target_quants: List[QuantType] = field(default_factory=lambda: [
        QuantType.Q4_K_M,
        QuantType.Q5_K_M,
        QuantType.Q8_0
    ])

    # imatrix 模式与超参数配置
    imatrix_mode: ImatrixMode = ImatrixMode.DATASET
    calibration_dataset_path: Optional[str] = None # 外部校准语料文本路径 (wikitext / alpaca / custom / final)
    existing_imatrix_path: Optional[str] = None    # 已有 imatrix 文件路径
    imatrix_chunks: int = 100                      # imatrix 计算分块数
    imatrix_chunk_size: int = 512                  # 每块 Token 长度
    imatrix_gpu_layers: int = 99                   # imatrix 运行时 GPU 卸载层数

    # UD 自校准数据生成参数（unsloth 风格）
    self_gen_prompts_count: int = 20               # 自生成主题数量
    self_gen_samples_per_prompt: int = 3           # 每个提示词生成样本数
    self_gen_max_tokens: int = 256                 # 生成最大 Token 数

    # UD 动态层级量化开关与自定义规则
    enable_layered_overrides: bool = True          # 是否开启层级动态保护
    tensor_overrides: Dict[str, str] = field(default_factory=lambda: {
        "blk.0.*": "Q6_K",                         # 输入浅层保留高精度
        "blk.1.*": "Q6_K",
        "blk.62.*": "Q6_K",                        # 语义深层保留高精度
        "blk.63.*": "Q6_K",
        "blk.64.nextn.*": "Q8_0",                  # MTP 投机头最高保真
        ".*attn_q.*": "Q5_K_M",                    # 注意力 Q 键保真
        ".*attn_k.*": "Q5_K_M",                    # 注意力 K 键保真
        ".*ffn_gate.*": "Q5_K_M",                  # FFN 门控保真
        ".*ffn_up.*": "Q5_K_M",
    })

    # 特殊模块提取
    extract_mtp: bool = True                       # 是否尝试提取 MTP Draft 头
    extract_mmproj: bool = False                   # 是否尝试提取视觉投影层

    # 资源与并发控制
    num_threads: int = 8                           # 并发线程数
    clean_intermediate_base: bool = False          # 是否在量化后清理基准 GGUF

    def is_input_gguf(self) -> bool:
        """判断输入是否已是 GGUF 文件"""
        if not self.model_input_path:
            return False
        return os.path.isfile(self.model_input_path) and self.model_input_path.lower().endswith(".gguf")

    def validate(self) -> None:
        """配置完整性验证"""
        if not self.model_input_path:
            raise ValueError("model_input_path 不能为空")
        if not self.output_dir:
            raise ValueError("output_dir 不能为空")
        if self.imatrix_mode == ImatrixMode.DATASET and not self.calibration_dataset_path:
            raise ValueError("模式为 DATASET 时必须提供 calibration_dataset_path")
        if self.imatrix_mode == ImatrixMode.EXISTING and not self.existing_imatrix_path:
            raise ValueError("模式为 EXISTING 时必须提供 existing_imatrix_path")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantConfig":
        """从字典构建 QuantConfig，支持枚举与格式转换"""
        d = dict(data)
        if "imatrix_mode" in d and isinstance(d["imatrix_mode"], str):
            d["imatrix_mode"] = ImatrixMode(d["imatrix_mode"])
        if "target_quants" in d and isinstance(d["target_quants"], list):
            quants = []
            for item in d["target_quants"]:
                if isinstance(item, str):
                    quants.append(QuantType[item])
                elif isinstance(item, QuantType):
                    quants.append(item)
            d["target_quants"] = quants
        # 兼容旧字段 model_input_dir
        if "model_input_dir" in d and "model_input_path" not in d:
            d["model_input_path"] = d.pop("model_input_dir")
        return cls(**d)

    @classmethod
    def from_file(cls, config_path: str) -> "QuantConfig":
        """从 JSON 或 YAML 文件读取外部超参配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        ext = os.path.splitext(config_path)[1].lower()
        if ext in [".yaml", ".yml"]:
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                # 若无 PyYAML，回退简单解析或提示
                raise ImportError("读取 YAML 配置文件需安装 PyYAML: pip install pyyaml")
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """导出为纯字典结构"""
        return {
            "model_input_path": self.model_input_path,
            "output_dir": self.output_dir,
            "model_name": self.model_name,
            "llama_cpp_bin_dir": self.llama_cpp_bin_dir,
            "llama_cpp_src_dir": self.llama_cpp_src_dir,
            "dir_name_base_gguf": self.dir_name_base_gguf,
            "dir_name_quant": self.dir_name_quant,
            "dir_name_calib": self.dir_name_calib,
            "dir_name_logs": self.dir_name_logs,
            "target_quants": [q.value for q in self.target_quants],
            "imatrix_mode": self.imatrix_mode.value,
            "calibration_dataset_path": self.calibration_dataset_path,
            "existing_imatrix_path": self.existing_imatrix_path,
            "imatrix_chunks": self.imatrix_chunks,
            "imatrix_chunk_size": self.imatrix_chunk_size,
            "imatrix_gpu_layers": self.imatrix_gpu_layers,
            "self_gen_prompts_count": self.self_gen_prompts_count,
            "self_gen_samples_per_prompt": self.self_gen_samples_per_prompt,
            "self_gen_max_tokens": self.self_gen_max_tokens,
            "enable_layered_overrides": self.enable_layered_overrides,
            "tensor_overrides": self.tensor_overrides,
            "extract_mtp": self.extract_mtp,
            "extract_mmproj": self.extract_mmproj,
            "num_threads": self.num_threads,
            "clean_intermediate_base": self.clean_intermediate_base,
        }
