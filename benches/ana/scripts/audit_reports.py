#!/usr/bin/env python3
"""
报告一致性审计：把「报告/图表/台账三处数字对不上」这类问题自动查出来。

为什么需要它：perfkit 的报告由多个子命令分别生成，同一份数据会被
台账 → baseline.json → optimize.json → verify.json → report.md → charts.html
搬运五次。任何一处口径不同（跨会话、扁平值 vs 场景值、不同 bundle），
都会让报告自相矛盾，而这种矛盾**读起来像结论、查起来要半小时**。
本项目实际踩过的两个坑（都已修）正好是它的前两条检查：

  ① 图表基线取自台账 native，候选取自 bundle measured —— 跨会话差 10.7%；
  ② CPU 配置的扁平 measured.decode_tps（depth=8192）与 KPI 绑定的
     short_chat 差 2.4×。

检查项（任一 FAIL 即退出码 1）：
  C1  图表「优化基线」柱 == 报告 §5.1 的 decode_tps before（差 <2%）
  C2  报告 §2.2 场景表的 decode 值 == 台账 scenarios[*].decode_tps（差 <2%）
  C3  verify 的 before == 台账对应场景值（差 <2%）
  C4  报告里不出现 None / nan / {placeholder} 这类未替换占位
  C5  投机配置的报告必须出现「口径失效」（band_util / 带宽反推）
  C6  结论摘要的章节序号连续（一、二、三…）
  C7  报告 §6.1 验收表 == verify.json 的逐项判定（invalid 项记「— 待复测」）

用法：
    python3 scripts/audit_reports.py                     # 审计 output/ 下全部运行
    python3 scripts/audit_reports.py --run <run_id>      # 只审计一个
    python3 scripts/audit_reports.py --strict            # 任一 WARN 也失败
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# 运行 id → 配置目录（用于读台账）
CONFIG_OF = {
    "qwen3.8-27b-decode-opt": "configs",
    "qwen3.8-27b-decode-multi-mode-opt": "configs/qwen3.8-27b-multimodal",
    "qwen3.8-27b-decode-multi-mode-uplimit-opt": "configs/qwen3.8-27b-multimodal-uplimit",
    "qwen3.8-27b-decode-new-arch-opt": "configs/qwen3.8-27b-multimodal-newarch",
    "qwen2.5-1.5b-cpu": "configs/qwen2.5-1.5b-cpu",
}

TOL = 0.02          # 2% 容差：低于此视为同一口径


class Audit:
    def __init__(self, strict: bool = False):
        self.rows: list[tuple[str, str, str, str]] = []   # (run, check, status, detail)
        self.strict = strict

    def add(self, run: str, check: str, ok: bool | None, detail: str):
        status = "PASS" if ok else ("WARN" if ok is None else "FAIL")
        self.rows.append((run, check, status, detail))

    def report(self) -> int:
        w = max((len(r[0]) for r in self.rows), default=10)
        fails = 0
        cur = None
        for run, check, status, detail in self.rows:
            if run != cur:
                print(f"\n=== {run} ===")
                cur = run
            mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
            print(f"  [{mark}] {check:<26s} {detail}")
            if status == "FAIL":
                fails += 1
            elif status == "WARN" and self.strict:
                fails += 1
        print()
        total = len(self.rows)
        warn = sum(1 for r in self.rows if r[2] == "WARN")
        print(f"合计 {total} 项：PASS {total - fails - (0 if self.strict else warn)}"
              f" / WARN {warn} / FAIL {sum(1 for r in self.rows if r[2] == 'FAIL')}")
        return 0 if fails == 0 else 1


def _close(a, b, tol=TOL) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tol


def _scenario_table(rep: str) -> dict[str, float]:
    """
    从报告 §2.2 的「场景数据」表里抽出 {场景: decode_tps}。

    表结构固定为：| 场景 | 说明 | Prefill | Decode | 系统吞吐 | TTFT | E2E |
    因此取第 4 列（index 3）。表格以空行结束，遇到非 `|` 开头的行即停。
    """
    out: dict[str, float] = {}
    lines = rep.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "**场景数据**："), None)
    if start is None:
        return out
    for line in lines[start + 1:]:
        if not line.startswith("|"):
            if out:            # 表格已结束
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] in ("场景",) or set(cells[0]) <= set(":- "):
            continue
        try:
            out[cells[0]] = float(cells[3])          # Decode (t/s)
        except ValueError:
            continue
    return out


def audit_run(a: Audit, run: str, cfg_rel: str | None) -> None:
    d = OUT / run
    rep_p = d / "report.md"
    if not rep_p.exists():
        a.add(run, "report 存在", False, "缺 report.md")
        return
    rep = rep_p.read_text(encoding="utf-8")

    # ---- C4 未替换占位 ----
    # 只查「正文里裸出现的 None/nan」与「未知占位符」：
    #   · `None` 作为字段值（如 sliding_window | `None`）是合法的，反引号内的不算；
    #   · {draft}/{mmproj}/{model} 是启动命令里的合法占位符，白名单放行。
    _LEGIT_PH = {"{draft}", "{mmproj}", "{model}", "{mmproj_f16}", "{mmproj_q8}",
                 "{mmproj_q5}", "{mmproj_q4}", "{draft_mtp}"}
    body = re.sub(r"```.*?```", "", rep, flags=re.S)          # 去掉代码块
    body = re.sub(r"`[^`]*`", "", body)                        # 去掉行内代码
    bad = []
    for pat, label in [(r"\bNone\b", "None"), (r"\bnan\b", "nan")]:
        hits = re.findall(pat, body)
        if hits:
            bad.append(f"{label}×{len(hits)}")
    ph = set(re.findall(r"\{[a-z_]+\}", rep)) - _LEGIT_PH
    if ph:
        bad.append("未知占位符 " + "、".join(sorted(ph)))
    a.add(run, "C4 无未替换占位", not bad, "、".join(bad) if bad else "干净")

    # ---- C1 图表基线 == §5.1 before ----
    ver_p, chart_p = d / "verify.json", d / "charts.html"
    if ver_p.exists() and chart_p.exists():
        ver = json.loads(ver_p.read_text(encoding="utf-8"))
        row = next((r for r in ver.get("comparison", [])
                    if r.get("key") == "decode_tps"), None)
        h = chart_p.read_text(encoding="utf-8")
        i = h.find("Decode 吞吐：理论上界")
        seg = h[i:h.find("</svg>", i)] if i >= 0 else ""
        bars = dict(re.findall(
            r'text-anchor="end">([^<]+)</text>\s*<rect[^>]*/>\s*'
            r'<text[^>]*>([\d.]+) t/s', seg))
        if row and "优化基线" in bars:
            ok = _close(bars["优化基线"], row.get("before"))
            a.add(run, "C1 图基线 == §5.1",
                  ok, f'图 {bars["优化基线"]} vs 报告 {row.get("before")}')
        else:
            a.add(run, "C1 图基线 == §5.1", None,
                  "图上无「优化基线」柱或 §5.1 无 decode_tps")
    else:
        a.add(run, "C1 图基线 == §5.1", None, "缺 verify.json 或 charts.html")

    # ---- C2/C3 台账 vs 报告 vs verify ----
    if not cfg_rel:
        a.add(run, "C2 报告 §2.2 == 台账", None, "未登记该运行的配置目录")
        return
    led_p = ROOT / cfg_rel / "ledger.json"
    if not led_p.exists():
        a.add(run, "C2 报告 §2.2 == 台账", None, f"缺 {led_p}")
        return
    led = json.loads(led_p.read_text(encoding="utf-8"))
    tbl = _scenario_table(rep)
    led_scn = led.get("scenarios") or {}
    mism = []
    for sid, v in tbl.items():
        lv = (led_scn.get(sid) or {}).get("decode_tps")
        if lv is None:
            continue
        if not _close(v, lv):
            mism.append(f"{sid}: 报告 {v} vs 台账 {lv}")
    a.add(run, "C2 报告 §2.2 == 台账", not mism,
          "；".join(mism) if mism else f"比对 {len(tbl)} 个场景")

    if ver_p.exists():
        ver = json.loads(ver_p.read_text(encoding="utf-8"))
        # KPI 名 → 台账 scenario 里的字段名（只列确实同义的；
        # 其余如 image_tokens_per_image / vision_encode_ms 台账不存，
        # 无法逐值核对，跳过而不是误报）。
        ALIAS = {"vis_ttft_ms": "ttft_ms", "vis_decode_tps": "decode_tps",
                 "vis_decode_tps_2048": "decode_tps",
                 "vis_decode_tps_repetitive": "decode_tps"}
        bad3, checked = [], 0
        for r in ver.get("comparison", []):
            sid = r.get("scenario")
            if not sid or r.get("before") is None:
                continue
            fld = ALIAS.get(r["key"], r["key"])
            lv = (led_scn.get(sid) or {}).get(fld)
            if lv is None:
                continue
            checked += 1
            if not _close(r["before"], lv):
                bad3.append(f'{r["key"]}: verify {r["before"]} vs 台账 {lv}')
        a.add(run, "C3 verify before == 台账", not bad3,
              "；".join(bad3) if bad3 else f"比对 {checked} 项")

    # ---- C5 投机配置必须标注口径失效 ----
    plan_p = next((ROOT / cfg_rel).glob("plan_*.yaml"), None)
    spec = False
    if plan_p:
        txt = plan_p.read_text(encoding="utf-8")
        m = re.search(r"^selected_bundle:\s*(\S+)", txt, re.M)
        if m:
            blk = txt.split(f"- id: {m.group(1)}")[-1].split("\n  - id:")[0]
            spec = "--spec-type" in blk
    if spec:
        ok = "口径失效" in rep
        a.add(run, "C5 投机口径标注", ok,
              "已标注" if ok else "配置开了投机，但报告未说明带宽口径失效")
    else:
        a.add(run, "C5 投机口径标注", None, "非投机配置，不适用")

    # ---- C6 摘要序号连续 ----
    nums = re.findall(r"^### ([一二三四五六七八九十]+)、", rep, re.M)
    if nums:
        order = "一二三四五六七八九十"
        seq = [order.index(n[0]) for n in nums if n and n[0] in order]
        ok = seq == list(range(len(seq)))
        a.add(run, "C6 摘要序号连续", ok,
              "→".join(nums) if ok else f"不连续：{'→'.join(nums)}")

    # ---- C7 §6.1 验收表 == verify.json 判定 ----
    # 只比对「可判定项」：`invalid` 口径在报告里写成「— 待复测」，
    # 那是**有意为之**（说明「测了但口径不成立」），不应算作缺行。
    if ver_p.exists():
        k = rep.find("### 6.1 验收判定")
        if k >= 0:
            body = rep[k:]
            i1 = body.find("① must_pass")
            i2 = body.find("② should_pass")
            i3 = body.find("③ guardrail")
            seg_m = body[i1:i2] if i1 >= 0 and i2 > i1 else ""
            # 「③ guardrail」不一定是 §6.1 里的标题（有的模板把它渲染成 §6.3 的
            # 「质量与稳定性红线」表），所以结束位置要依次回退到
            # ③ / ### 6.2 / ### 6.3 / 正文末尾。
            _ends = [x for x in (i3, body.find("### 6.2"), body.find("### 6.3"))
                     if x > i2]
            seg_s = body[i2:min(_ends)] if i2 >= 0 and _ends else \
                (body[i2:] if i2 >= 0 else "")
            def _verdicts(seg: str) -> list[bool | None]:
                out = []
                for line in seg.splitlines():
                    if not line.startswith("| ") or line.startswith("|:") \
                            or "验收条件" in line:
                        continue
                    if "✅ PASS" in line:
                        out.append(True)
                    elif "❌ FAIL" in line:
                        out.append(False)
                    elif "待复测" in line or "不判定" in line:
                        out.append(None)
                return out
            vm, vs = _verdicts(seg_m), _verdicts(seg_s)
            em = [c["pass"] for c in ver["acceptance_checks"]]
            es = [c["pass"] for c in ver["should_pass_checks"]]
            ok = vm == em and vs == es
            detail = (f"must {len(vm)}/{len(em)}、should {len(vs)}/{len(es)}"
                      + ("" if ok else " —— 行数或判定与 verify.json 不一致"))
            a.add(run, "C7 验收表 == verify", ok, detail)
        else:
            a.add(run, "C7 验收表 == verify", None, "报告缺 §6.1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    runs = [a.run] if a.run else list(CONFIG_OF)
    audit = Audit(strict=a.strict)
    for r in runs:
        audit_run(audit, r, CONFIG_OF.get(r))
    return audit.report()


if __name__ == "__main__":
    sys.exit(main())
