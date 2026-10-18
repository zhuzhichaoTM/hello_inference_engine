#!/usr/bin/env python3
"""
pack_skills.py —— 把每个技能打成可分发的 .skill 包（zip）
==============================================================================
打包规则与 skill-creator 的约定一致：排除缓存与评测产物，
保留 SKILL.md 与 references/。

用法：
    python3 skills/pack_skills.py              # 输出到 skills/dist/
    python3 skills/pack_skills.py --out DIR    # 指定输出目录
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {"__pycache__", "node_modules", "evals", "dist"}
EXCLUDE_NAMES = {".DS_Store"}


def pack(skill_dir: Path, out_dir: Path) -> tuple[Path, int, int]:
    out = out_dir / f"{skill_dir.name}.skill"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(skill_dir.rglob("*")):
            if f.is_dir():
                continue
            rel = f.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if f.name in EXCLUDE_NAMES or f.suffix == ".pyc":
                continue
            z.write(f, rel)
            n += 1
    return out, n, out.stat().st_size


def main() -> int:
    out_dir = ROOT / "dist"
    if "--out" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    skills = sorted(d for d in ROOT.iterdir()
                    if d.is_dir() and (d / "SKILL.md").exists())
    if not skills:
        print(f"（{ROOT} 下没有找到任何技能）")
        return 1

    print(f"{'技能':<26} {'文件数':>6} {'大小':>9}")
    print("-" * 46)
    for sd in skills:
        out, n, size = pack(sd, out_dir)
        print(f"{sd.name:<26} {n:>6} {size / 1024:>8.1f}K")
    print(f"\n共 {len(skills)} 个包 → {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
