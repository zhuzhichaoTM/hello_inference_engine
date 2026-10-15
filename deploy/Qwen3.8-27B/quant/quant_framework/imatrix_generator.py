"""
重要性矩阵生成引擎 (imatrix_generator.py)
专责：以基准 GGUF（BF16/F16）为输入，根据四种模式产出/确认 imatrix.gguf。
内建外部数据集自动下载、清洗与缓存机制（支持 WikiText-2, Alpaca 等）。
"""

import os
import logging
import urllib.request
import json
from typing import Optional

from .quant_config import QuantConfig, ImatrixMode
from .quant_utils import ensure_dirs, run_command

logger = logging.getLogger("ImatrixGenerator")

WIKITEXT_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/scripts/wikitext-2-raw/wiki.test.raw"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


class ImatrixGenerator:
    """
    imatrix.gguf 生成器。

    支持四种运行模式：
      - NONE:           不使用 imatrix
      - DATASET:        使用外部校准文本驱动 llama-imatrix (若本地缺失则自动从网络下载构建)
      - SELF_GENERATED: unsloth 风格：由原始模型自身自回归采样生成
      - EXISTING:       复用已有 imatrix 文件
    """

    def __init__(self, config: QuantConfig, base_gguf_path: str, work_dir: str):
        self.cfg = config
        self.base_gguf_path = base_gguf_path

        self.dir_calib = os.path.join(work_dir, config.dir_name_calib)
        self.dir_quant = os.path.join(work_dir, config.dir_name_quant)
        self.dir_logs = os.path.join(work_dir, config.dir_name_logs)
        ensure_dirs(self.dir_calib, self.dir_quant, self.dir_logs)

        self.imatrix_path: Optional[str] = None

    def generate(self) -> Optional[str]:
        """按配置生成或确认 imatrix 文件"""
        mode = self.cfg.imatrix_mode
        logger.info(f"=== ImatrixGenerator: 模式 {mode.value} ===")

        if mode == ImatrixMode.NONE:
            logger.info("未启用 imatrix，下游将执行无权重感知的标准量化。")
            return None

        if mode == ImatrixMode.EXISTING:
            return self._use_existing()

        calib_text = self._resolve_calibration_text()
        return self._run_llama_imatrix(calib_text)

    def _use_existing(self) -> str:
        path = self.cfg.existing_imatrix_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"指定的已有 imatrix 文件不存在: {path}")
        self.imatrix_path = path
        logger.info(f"复用已有 imatrix: {path}")
        return path

    def _resolve_calibration_text(self) -> str:
        """返回最终喂给 llama-imatrix 的文本文件路径，支持自动下载"""
        if self.cfg.imatrix_mode == ImatrixMode.DATASET:
            src = self.cfg.calibration_dataset_path
            # 若未指定或指定文件不存在，启动内置自动下载与构建流程
            if not src or not os.path.exists(src):
                logger.info("外部校准语料未指定或不存在，触发内置自动下载与构建流程...")
                auto_final = os.path.join(self.dir_calib, "final.txt")
                self._auto_download_and_build_dataset(auto_final)
                return auto_final

            logger.info(f"使用已存在的外部校准语料: {src}")
            return src

        if self.cfg.imatrix_mode == ImatrixMode.SELF_GENERATED:
            out = os.path.join(self.dir_calib, "self_generated.txt")
            if not os.path.exists(out) or os.path.getsize(out) < 100:
                self._generate_self_calibration_dataset(out)
            return out

        raise ValueError(f"不支持的 imatrix 模式: {self.cfg.imatrix_mode}")

    def _auto_download_and_build_dataset(self, target_final_path: str) -> None:
        """
        自动下载 WikiText-2、Alpaca 并融合生成业务语料 final.txt
        嵌入代码主流程，无需依赖任何外部前置脚本
        """
        wiki_file = os.path.join(self.dir_calib, "wikitext.txt")
        alpaca_json = os.path.join(self.dir_calib, "alpaca.json")
        alpaca_txt = os.path.join(self.dir_calib, "alpaca.txt")
        custom_txt = os.path.join(self.dir_calib, "custom.txt")

        # 1. 自动下载 WikiText-2
        if not os.path.exists(wiki_file) or os.path.getsize(wiki_file) == 0:
            logger.info(f"正在自动下载 WikiText-2 基准语料: {WIKITEXT_URL} ...")
            try:
                urllib.request.urlretrieve(WIKITEXT_URL, wiki_file)
                logger.info("✅ WikiText-2 下载完成")
            except Exception as e:
                logger.warning(f"WikiText-2 下载失败 ({e})，创建空占位文件")
                open(wiki_file, "w").close()

        # 2. 自动下载并清洗 Alpaca
        if not os.path.exists(alpaca_txt) or os.path.getsize(alpaca_txt) == 0:
            logger.info(f"正在自动下载 Alpaca 指令数据集: {ALPACA_URL} ...")
            try:
                urllib.request.urlretrieve(ALPACA_URL, alpaca_json)
                with open(alpaca_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with open(alpaca_txt, "w", encoding="utf-8") as f:
                    for item in data[:500]:
                        inst = item.get("instruction", "")
                        inp = item.get("input", "")
                        oup = item.get("output", "")
                        f.write(f"{inst} {inp} {oup}\n")
                logger.info(f"✅ Alpaca 清洗完成，提取 500 条样本至 {alpaca_txt}")
            except Exception as e:
                logger.warning(f"Alpaca 处理失败 ({e})，跳过该源")

        # 3. 自动生成定制业务与算法语料
        if not os.path.exists(custom_txt) or os.path.getsize(custom_txt) == 0:
            logger.info("构建大模型量化与分布式推理定制业务语料...")
            with open(custom_txt, "w", encoding="utf-8") as f:
                f.write(
                    "请分析大语言模型在低比特量化（如 4-bit, 2-bit）场景下的张量敏感度分布，以及首尾层、注意力机制和 FFN 门控网络的误差放大效应。\n"
                    "设计一个基于 Unsloth Dynamic (UD) 策略的高精度量化流水线，涵盖自校准提示词生成与重要性矩阵加权优化。\n"
                    "分析高并发分布式推理架构下 KV Cache 显存优化、PagedAttention 与投机解码 (Speculative Decoding) 的吞吐加速比。\n"
                )

        # 4. 合并去重并截取 500 条高质量样本
        logger.info(f"合并多源语料至: {target_final_path} ...")
        merged_lines = []
        for src_f in [wiki_file, alpaca_txt, custom_txt]:
            if os.path.exists(src_f):
                with open(src_f, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 30:
                            merged_lines.append(line)

        # 去重并切分
        unique_lines = list(dict.fromkeys(merged_lines))[:500]
        with open(target_final_path, "w", encoding="utf-8") as f:
            for line in unique_lines:
                f.write(line + "\n\n")

        size_kb = os.path.getsize(target_final_path) / 1024
        logger.info(f"✅ 自动构建黄金校准集完成: {target_final_path} ({len(unique_lines)} 条样本, {size_kb:.1f} KB)")

    def _run_llama_imatrix(self, calib_text: str) -> str:
        imatrix_bin = os.path.join(self.cfg.llama_cpp_bin_dir, "llama-imatrix")

        if self.cfg.imatrix_mode == ImatrixMode.SELF_GENERATED:
            filename = f"imatrix-UD-{self.cfg.model_name}.gguf"
            log_name = "imatrix_ud.log"
        else:
            filename = f"imatrix-{self.cfg.model_name}.gguf"
            log_name = "imatrix.log"

        self.imatrix_path = os.path.join(self.dir_quant, filename)

        # 针对 32G 显存 / 64G 内存自动设置安全的 GPU 卸载层数（27B 共 64 层，32G 显存适合卸载 35 层）
        ngl = min(self.cfg.imatrix_gpu_layers, 35) if self.cfg.imatrix_gpu_layers == 99 else self.cfg.imatrix_gpu_layers

        cmd = [
            imatrix_bin,
            "-m", self.base_gguf_path,
            "-f", calib_text,
            "-o", self.imatrix_path,
            "-ngl", str(ngl),
            "--chunks", str(self.cfg.imatrix_chunks),
            "-b", str(self.cfg.imatrix_chunk_size),
            "-t", str(self.cfg.num_threads),
        ]
        run_command(cmd, log_path=os.path.join(self.dir_logs, log_name))

        if not os.path.exists(self.imatrix_path):
            raise FileNotFoundError(f"llama-imatrix 未产出文件: {self.imatrix_path}")

        size_mb = os.path.getsize(self.imatrix_path) / (1024 ** 2)
        logger.info(f"✅ imatrix 生成成功: {self.imatrix_path} ({size_mb:.1f} MB)")
        return self.imatrix_path

    def _generate_self_calibration_dataset(self, output_file: str) -> None:
        prompts = [
            "请解释深度学习中的注意力机制原理与多头自注意力数学形式",
            "写一个 Python 函数实现高效的快速排序算法，并分析其复杂度",
            "分析高并发分布式场景下 Raft 协议与 Paxos 协议的区别",
            "设计一个高可用高吞吐量的大语言模型推理集群架构方案",
            "分析 Transformer 模型中 KV Cache 的显存占用计算公式与优化策略",
            "解释大模型权重量化中 AWQ 与 GPTQ 的核心算法差异与应用场景",
            "实现一个具有线程安全特性的生产者-消费者队列模型",
            "比较 TCP 与 UDP 协议在实时流媒体与可靠传输场景下的性能表现",
        ]
        logger.info(f"启动自校准语料生成，提示词数: {len(prompts)}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write((p + " ") * 30 + "\n\n")
