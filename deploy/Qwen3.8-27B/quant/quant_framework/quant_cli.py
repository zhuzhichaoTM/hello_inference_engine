"""
量化框架命令行入口 (quant_cli.py)

支持：
  1. 通过命令行参数指定超参
  2. 通过 --config-file 载入外部 YAML/JSON 超参配置文件
  3. 支持 BF16 GGUF 直接输入或 safetensors 自动转换
  4. 支持 1-bit 到 16-bit 任意量化档位生成
  5. 分步执行（--imatrix-only, --quant-only）与一键全流程
"""

import argparse
import sys
import os
from .quant_config import QuantConfig, ImatrixMode, QuantType
from .quant_pipeline import QuantPipeline
from .imatrix_generator import ImatrixGenerator
from .gguf_quantizer import GGUFQuantizer, BaseGGUFConverter


def parse_args():
    parser = argparse.ArgumentParser(
        description="大模型低比特量化转换框架 CLI (支持 1-bit 到 16-bit)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 外部配置文件 ───────────────────────────────────────────────────────────
    parser.add_argument("--config-file", default=None, help="外部 YAML 或 JSON 格式配置文件路径")

    # ── 路径 ──────────────────────────────────────────────────────────────
    parser.add_argument("--model-path", default=None, help="原始模型权重目录(safetensors) 或已有的 BF16/F16 GGUF 文件")
    parser.add_argument("--model-name", default="Qwen3.8-27B", help="模型名称标识")
    parser.add_argument("--output-dir", default=None, help="量化成果输出根目录")
    parser.add_argument("--llama-bin", default="/root/autodl-tmp/llama.cpp/build/bin")
    parser.add_argument("--llama-src", default="/root/autodl-tmp/llama.cpp")

    # ── 量化档位 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--quants", nargs="+",
        default=["Q4_K_M", "Q5_K_M", "Q8_0"],
        help="目标量化档位，如: IQ1_S IQ2_XXS Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0 等",
    )

    # ── imatrix 配置 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--imatrix-mode",
        choices=["none", "dataset", "self_gen", "existing"],
        default="dataset",
        help="none=标准量化(不激活imatrix) | dataset=外部语料校准 | self_gen=UD自校准 | existing=已有文件",
    )
    parser.add_argument("--calib-file", default=None, help="外部校准语料路径 (dataset 模式使用)")
    parser.add_argument("--existing-imatrix", default=None, help="已有 imatrix.gguf 路径")
    parser.add_argument("--chunks", type=int, default=100, help="imatrix 分块数")
    parser.add_argument("--chunk-size", type=int, default=512, help="imatrix chunk token 长度")

    # ── 分步执行控制 ──────────────────────────────────────────────────────
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--imatrix-only", action="store_true", help="仅生成重要性矩阵 imatrix.gguf")
    step.add_argument("--quant-only", action="store_true", help="直接执行量化 (需配合已有 GGUF 输入)")

    # ── 策略与参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--ud-mode", action="store_true", help="显式开启 UD 风格输出命名与层级保护")
    parser.add_argument("--disable-layered", action="store_true", help="禁用层级差异化保护规则")
    parser.add_argument("--clean-base", action="store_true", help="量化后自动清理基准 GGUF 节约磁盘")
    parser.add_argument("--threads", type=int, default=8, help="CPU 线程数")

    return parser.parse_args()


def _build_config(args) -> QuantConfig:
    # 优先从外部配置文件读取
    if args.config_file:
        cfg = QuantConfig.from_file(args.config_file)
        # 支持命令行参数覆盖配置文件
        if args.model_path:
            cfg.model_input_path = args.model_path
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.quants:
            cfg.target_quants = [QuantType[q] for q in args.quants if q in QuantType.__members__]
        return cfg

    if not args.model_path:
        print("错误: 必须通过 --model-path 或 --config-file 指定模型输入路径")
        sys.exit(1)
    if not args.output_dir:
        print("错误: 必须通过 --output-dir 或 --config-file 指定输出根目录")
        sys.exit(1)

    target_quants = []
    for q in args.quants:
        try:
            target_quants.append(QuantType[q])
        except KeyError:
            print(f"错误: 不支持的量化档位 '{q}'，可用档位: {[e.value for e in QuantType]}")
            sys.exit(1)

    mode_map = {
        "none": ImatrixMode.NONE,
        "dataset": ImatrixMode.DATASET,
        "self_gen": ImatrixMode.SELF_GENERATED,
        "existing": ImatrixMode.EXISTING,
    }

    return QuantConfig(
        model_input_path=args.model_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        llama_cpp_bin_dir=args.llama_bin,
        llama_cpp_src_dir=args.llama_src,
        target_quants=target_quants,
        imatrix_mode=mode_map[args.imatrix_mode],
        calibration_dataset_path=args.calib_file,
        existing_imatrix_path=args.existing_imatrix,
        imatrix_chunks=args.chunks,
        imatrix_chunk_size=args.chunk_size,
        enable_layered_overrides=not args.disable_layered,
        clean_intermediate_base=args.clean_base,
        num_threads=args.threads,
    )


def main():
    args = parse_args()
    cfg = _build_config(args)

    # 1. 仅生成 imatrix
    if args.imatrix_only:
        conv = BaseGGUFConverter(cfg, cfg.output_dir)
        conv.check_prerequisites()
        base_path = conv.convert()
        gen = ImatrixGenerator(cfg, base_gguf_path=base_path, work_dir=cfg.output_dir)
        path = gen.generate()
        print(f"\n✅ imatrix 生成完毕: {path}")
        return

    # 2. 仅执行量化
    if args.quant_only:
        base_path = cfg.model_input_path
        if not cfg.is_input_gguf():
            conv = BaseGGUFConverter(cfg, cfg.output_dir)
            base_path = conv.convert()
        qz = GGUFQuantizer(cfg, cfg.output_dir)
        is_ud = args.ud_mode or (cfg.imatrix_mode == ImatrixMode.SELF_GENERATED)
        results = qz.quantize_all(base_path, imatrix_path=cfg.existing_imatrix_path, is_ud_mode=is_ud)
        print("\n🎉 量化完成:")
        for k, v in results.items():
            print(f"  [{k}] -> {v}")
        return

    # 3. 完整自动化流水线
    pipeline = QuantPipeline(cfg)
    results = pipeline.run_all()
    print("\n🎉 量化全部完成，交付文件列表:")
    for k, v in results.items():
        print(f"  [{k}] -> {v}")


if __name__ == "__main__":
    main()
