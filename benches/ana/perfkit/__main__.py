"""
perfkit CLI 入口
==============================================================================
    python3 -m perfkit <subcommand> [options]

全局选项：
    --config DIR   配置目录（默认 configs）
    --run ID       运行标识（默认取 task.id），决定 output/<ID>/ 目录

示例：
    python3 -m perfkit all      --run qwen3.8-27b-decode-opt
    python3 -m perfkit define   --run qwen3.8-27b-decode-opt
    python3 -m perfkit baseline --run qwen3.8-27b-decode-opt --mode live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .commands import MODULES
from .config import ConfigError, load_yaml


BANNER = r"""
                    _    _ _
  _ __   ___ _ __  | | _(_) |_
 | '_ \ / _ \ '__| | |/ / | __|  大模型推理性能分析与优化框架
 | |_) |  __/ |    |   <| | |_   七步业务闭环 · 可复用 · 多语言
 | .__/ \___|_|    |_|\_\_|\__|  v{ver}
 |_|
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="perfkit",
        description="大模型推理性能分析与优化框架（基于端侧推理业务地图·工程闭环 7 步）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令一览（对应七步闭环）：
  define      1 问题定义：画像汇总 / 显存预算 / roofline / KPI
  baseline    2 性能基线：replay 读台账 或 live 真机采集
  profiling   3 性能剖面：阶段 / 逐层 / 算子 / 硬件 四维
  diagnose    4 瓶颈定位：决策树 + 证据链 + 反证
  optimize    5 优化方案：筛选排序 / bundle 对比 / 启动脚本
  verify      6 效果验证：前后对比 / 收益归因 / 副作用
  report      7 报告导出：七步闭环 Markdown
  visualize   可视化：自包含 charts.html
  probe       硬件带宽探针：实测内存/显存带宽
  query       跨运行查询：运行清单 / 单指标 / 横向对比 / 后端清单
  all         一键跑完全部环节

典型用法：
  perfkit all --run my-run                       # 全流程
  perfkit baseline --run my-run --mode live      # 真机采集
  perfkit define --run my-run                    # 单环节续跑
  perfkit query --list                           # 看有哪些运行
  perfkit query --backends                       # 看支持哪些硬件后端
  perfkit probe                                  # 测本机内存带宽
""")
    p.add_argument("--version", action="version", version=f"perfkit {__version__}")
    p.add_argument("--config", default="configs", help="配置目录（默认 configs）")
    p.add_argument("--run", default=None,
                   help="运行标识，决定 output/<ID>/；默认取 task.id")
    sub = p.add_subparsers(dest="cmd", metavar="<subcommand>")
    # 全局选项同时挂到每个子命令上，使得 `perfkit define --run X` 与
    # `perfkit --run X define` 两种写法都成立。
    # default=SUPPRESS 保证未显式传入时不会覆盖顶层解析出的值；
    # 子命令若自己定义了同名选项（如 query 的历史遗留），跳过以免冲突。
    for mod in MODULES:
        sp = mod.add_parser(sub)
        existing = set()
        for action in sp._actions:
            existing.update(action.option_strings)
        if "--config" not in existing:
            sp.add_argument("--config", default=argparse.SUPPRESS,
                            help="配置目录（默认 configs）")
        if "--run" not in existing:
            sp.add_argument("--run", default=argparse.SUPPRESS,
                            help="运行标识，决定 output/<ID>/")
    return p


def _resolve_run(args) -> None:
    """--run 缺省时从 task 配置里取 task.id，避免用户必须手写。"""
    if args.run:
        return
    cfgdir = Path(args.config)
    try:
        # 与 PerfConfig.from_dir 同一套「按角色前缀 glob」的解析，
        # 否则 CPU 配置目录（task_cpu_optimize.yaml）会取不到 id。
        hits = sorted(cfgdir.glob("task_*.yaml"))
        task = load_yaml(hits[0]) if hits else load_yaml(cfgdir / "task_decode_optimize.yaml")
        args.run = task.get("task", {}).get("id", "default")
    except ConfigError:
        args.run = "default"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "cmd", None):
        print(BANNER.format(ver=__version__))
        parser.print_help()
        return 0

    _resolve_run(args)

    try:
        return args.func(args)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[中断] 用户取消", file=sys.stderr)
        return 130
    except FileNotFoundError as e:
        print(f"[缺少文件] {e}\n提示：请先执行上游子命令（如 perfkit define）", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
