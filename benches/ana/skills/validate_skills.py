#!/usr/bin/env python3
"""
validate_skills.py —— 技能包自检
==============================================================================
不依赖任何第三方库（本机没装 PyYAML 也能跑），校验每个技能的结构完整性
与「引用一致性」。

为什么需要这个：
    技能最容易出的两类问题都不会报错，只会静默失效：
      1) frontmatter 里 name 与目录名不一致 → 触发不了
      2) SKILL.md 里引用了不存在的参考文件 → 加载时才发现缺失
    这两类问题在写的时候看不出来，跑一遍校验就能拦住。

用法：
    python3 skills/validate_skills.py            # 全部检查
    python3 skills/validate_skills.py -v         # 打印每项细节
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERBOSE = "-v" in sys.argv
PASS = FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    PASS, FAIL = PASS + (1 if ok else 0), FAIL + (0 if ok else 1)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if (VERBOSE or not ok) else ""))


def parse_frontmatter(text: str) -> dict:
    """
    解析 YAML frontmatter。

    只处理技能需要的两种形态：单行标量、以及以 `>-`/`>` 起始的折叠块
    （description 常常很长，会折成多行）。刻意不引入 YAML 依赖 ——
    技能校验工具本身不该有安装门槛。
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    body = text[3:end]
    out: dict = {}
    key = None
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^[A-Za-z_][\w-]*\s*:", line):
            k, _, v = line.partition(":")
            key = k.strip()
            v = v.strip()
            out[key] = "" if v in (">-", ">", "|", "|-") else v
        elif key and (line.startswith(" ") or line.startswith("\t")):
            out[key] = (out[key] + " " + line.strip()).strip()
    return out


def main() -> int:
    skill_dirs = sorted(
        d for d in ROOT.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists())

    if not skill_dirs:
        print(f"（{ROOT} 下没有找到任何技能）")
        return 1

    print(f"校验 {len(skill_dirs)} 个技能：{ROOT}\n")

    for sd in skill_dirs:
        print(f"── {sd.name} ──")
        text = (sd / "SKILL.md").read_text(encoding="utf-8")
        fm = parse_frontmatter(text)

        # ---- frontmatter ----
        check("有 frontmatter", bool(fm), "缺少 --- 包裹的 YAML 头")
        name = fm.get("name", "")
        check("name 与目录名一致", name == sd.name,
              f"name={name!r} dir={sd.name!r}")
        check("name 符合 kebab-case", bool(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name)),
              name)
        desc = fm.get("description", "")
        check("有 description", len(desc) > 40, f"{len(desc)} 字符")
        # 触发机制靠 description：既要说明「做什么」，也要说明「何时用」
        check("description 含触发场景", ("当用户" in desc or "Use when" in desc
                                        or "whenever" in desc.lower()),
              "缺少「何时使用」的描述，会影响触发率")
        check("description 长度合理", 40 < len(desc) < 1200, f"{len(desc)} 字符")

        # ---- 正文结构 ----
        lines = text.splitlines()
        check("正文行数 < 500", len(lines) < 500, f"{len(lines)} 行")
        check("有 H1 标题", any(l.startswith("# ") for l in lines))

        # ---- 引用一致性：SKILL.md 里提到的参考文件必须存在 ----
        # 两种写法都要支持：
        #   references/foo.md                    → 本技能内
        #   other-skill/references/foo.md        → 跨技能（合法，但要能解析）
        local_refs = re.findall(r"(?<![\w/-])references/([\w.-]+\.md)", text)
        cross_refs = re.findall(r"([\w-]+)/references/([\w.-]+\.md)", text)

        missing = [r for r in set(local_refs) if not (sd / "references" / r).exists()]
        check("本技能 references 引用都存在", not missing,
              f"缺失: {missing}" if missing else f"{len(set(local_refs))} 个引用")

        bad_cross = [f"{s}/{r}" for s, r in set(cross_refs)
                     if not (ROOT / s / "references" / r).exists()]
        check("跨技能 references 引用都存在", not bad_cross,
              f"缺失: {bad_cross}" if bad_cross else f"{len(set(cross_refs))} 个引用")

        # ---- 反向：存在但没被引用的参考文件（不报错，只提示）----
        ref_dir = sd / "references"
        if ref_dir.exists():
            on_disk = {p.name for p in ref_dir.glob("*.md")}
            unreferenced = on_disk - set(local_refs)
            if unreferenced:
                print(f"  [note] 未被 SKILL.md 引用: {sorted(unreferenced)}")
            # 大文件应有目录，便于按需跳读
            for p in ref_dir.glob("*.md"):
                rl = p.read_text(encoding="utf-8").splitlines()
                if len(rl) > 300 and not any("目录" in l or "Contents" in l
                                             for l in rl[:20]):
                    check(f"{p.name} 有目录", False,
                          f"{len(rl)} 行但没有目录段")

        # ---- 技能间互引：提到的其他技能必须存在 ----
        siblings = {d.name for d in skill_dirs}
        mentioned = set(re.findall(r"`(llm-[\w-]+|seven-step-closure)`", text))
        unknown = {m for m in mentioned if m not in siblings}
        check("技能间互引有效", not unknown,
              f"引用了不存在的技能: {sorted(unknown)}" if unknown else
              f"{len(mentioned)} 个互引")
        print()

    print("=" * 60)
    print(f"校验结果：{PASS} 通过，{FAIL} 失败")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
