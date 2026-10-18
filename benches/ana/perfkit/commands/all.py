"""
perfkit all —— 一键跑完七步闭环
==============================================================================
等价于依次执行：
    define → baseline → profiling → diagnose → optimize → verify
    → report → visualize

每一步的产物都落盘到 output/<run_id>/，任一步失败即中断并保留已完成产物，
便于定位问题后从中间环节续跑（各子命令可独立执行）。
"""

from __future__ import annotations

import argparse
import time

from ..config import PerfConfig


STEPS = [
    ("define",    "第1步 问题定义"),
    ("baseline",  "第2步 性能基线"),
    ("profiling", "第3步 性能剖面"),
    ("diagnose",  "第4步 瓶颈定位"),
    ("optimize",  "第5步 优化方案"),
    ("verify",    "第6步 效果验证"),
    ("report",    "第7步 报告导出"),
    ("visualize", "可视化图表"),
]

# 各子命令私有选项的默认值。全流程串联时，子命令的 main() 会读取这些字段，
# 但它们的 argparse 定义只在单独调用时生效，因此这里统一补齐。
# 注意 ledger 必须是 None —— 这样 baseline 才会走 cfg.ledger() 的
# 「先查配置目录、再回落到全局台账」解析顺序；写死路径会覆盖该逻辑。
STEP_DEFAULTS = {
    "mode": "replay",
    "ledger": None,
    "bench_bin": None,
    "baseline": None,
    "from_gguf": None,      # 透传给 define：核验架构超参
    "only": None,           # 透传给 baseline：只跑指定场景
    "write_ledger": None,   # 透传给 baseline：导出台账
    "load_seconds": 20,     # 透传给 profiling：live 采样秒数
}


def main(args) -> int:
    from . import (baseline, define, diagnose, optimize, profiling, report,
                   verify, visualize)
    registry = {
        "define": define, "baseline": baseline, "profiling": profiling,
        "diagnose": diagnose, "optimize": optimize, "verify": verify,
        "report": report, "visualize": visualize,
    }

    # 合并默认值：显式传入的字段优先，缺省字段用 STEP_DEFAULTS 补齐
    merged = {**STEP_DEFAULTS, **vars(args)}
    step_args = argparse.Namespace(**merged)

    cfg = PerfConfig.from_dir(step_args.config, step_args.run)
    print("#" * 74)
    print(f"# perfkit 七步闭环 · {cfg.model['name']} @ {cfg.device['name']}")
    print(f"# run_id={cfg.run_id}  产物目录={cfg.outdir}")
    print("#" * 74)

    t_all = time.time()
    for i, (name, desc) in enumerate(STEPS, 1):
        print(f"\n{'─' * 74}")
        print(f"  [{i}/{len(STEPS)}] {desc}  ({name})")
        print(f"{'─' * 74}")
        t0 = time.time()
        rc = registry[name].main(step_args)
        if rc != 0:
            print(f"\n[FAIL] {name} 返回 {rc}，闭环中断。"
                  f"修复后可从 `perfkit {name}` 单独续跑。")
            return rc
        print(f"  ✓ {desc} 完成（{time.time()-t0:.2f}s）")

    print(f"\n{'#' * 74}")
    print(f"# 七步闭环全部完成，耗时 {time.time()-t_all:.2f}s")
    print(f"# 报告：{cfg.outdir/'report.md'}")
    print(f"# 图表：{cfg.outdir/'charts.html'}")
    print(f"# 启动脚本：{cfg.outdir/'start_optimized.sh'}")
    print("#" * 74)
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("all", help="一键跑完七步闭环（define→…→report→visualize）")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--from-gguf", default=None, metavar="MODEL.GGUF",
                   help="透传给 define：从 GGUF 元数据核验架构超参")
    # 真机跑时这些参数必须能透传，否则 live 模式在「llama-bench 不在 PATH」
    # 或「只想先跑一个场景冒烟」的情况下无法使用 —— 而那恰恰是换硬件后
    # 第一次跑的场景。
    p.add_argument("--bench-bin", default=None,
                   help="透传给 baseline：llama-bench 路径")
    p.add_argument("--only", default=None, metavar="ID[,ID...]",
                   help="透传给 baseline：只跑指定场景（冒烟验证用）")
    p.add_argument("--write-ledger", default=None, metavar="PATH",
                   help="透传给 baseline：把 live 结果导出为台账")
    p.add_argument("--load-seconds", type=int, default=20,
                   help="透传给 profiling：live 模式下采样硬件时序的秒数")
    p.set_defaults(func=main)
    return p
