"""
perfkit optimize —— 工程闭环第 5 步「优化实施」（方案生成）
==============================================================================
输入：diagnose.json 的优化路线 + plan 库
输出：optimize.json + 可直接执行的启动脚本

做的事：
  1. 按 applicable_when 条件从 plan 库筛选**适用**方案；
  2. 按瓶颈路线排序，给出每条的期望收益 / 代价 / 风险 / 回滚；
  3. 组装候选 bundle（组合方案），估算每个 bundle 的预期指标；
  4. 生成 bundle 的 llama-server 启动命令与回滚命令。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from ..config import (GIB, ConfigError, PerfConfig, flag_present, flag_value,
                      load_json, mem_budget, parse_quant_bits, save_json,
                      weight_bytes)


def applicable(plan: dict, route: list[str], ctx: dict) -> tuple[bool, str]:
    """
    判断方案是否适用于当前瓶颈。

    两种条件语义（可同时使用）：
      require_all     —— **全部**命中才适用（硬门槛，如 NPU 方案必须设备有 NPU）
      applicable_when —— **任一**命中即适用（软条件，如「访存受限 或 长上下文」）

    分开的理由：把「设备根本不具备该能力」和「当前瓶颈不是它」混在一个
    列表里，会让方案库在换设备时静默给出不可执行的建议。
    """
    tags = set(ctx.get("tags", [])) | set(route)
    need_all = plan.get("require_all") or []
    miss = [r for r in need_all if r not in tags]
    if miss:
        return False, f"硬门槛未满足：缺少 {miss}（需 {need_all}）"
    when = plan.get("applicable_when", [])
    if "always" in when:
        return True, "通用方案"
    if not when:
        return True, "无适用条件限制（仅受硬门槛约束）"
    hit = [w for w in when if w in tags]
    if hit:
        return True, f"命中条件 {hit}"
    return False, f"未命中（需要 {when} 之一，当前路线 {route}）"


def estimate_bundle(cfg: PerfConfig, bundle: dict, plans_by_id: dict,
                    base_tps: float) -> dict:
    """
    估算某个 bundle 的预期显存与吞吐。

    显存走**几何模型**（第一性原理，两侧同口径）；
    吞吐以**台账实测的原生基线**为锚，按 plan 的 gain 连乘 ——
    gain 只对「类别互不重叠」的方案连乘，避免重复计入同一份收益。

    base_tps 必须来自台账而非硬编码：不同设备/模型的原生基线相差
    可达一个数量级（RTX 4090 上 50.47 t/s vs i7-4770HQ 上 17.85 t/s），
    写死一个值会让所有估算在别的设备上完全失真。
    """
    flags = bundle.get("startup_flags", [])
    ctx = flag_value(flags, "--ctx-size", int) or 8192
    par = flag_value(flags, "--parallel", int) or 1
    k_bits = parse_quant_bits(flag_value(flags, "--cache-type-k") or "f16")
    v_bits = parse_quant_bits(flag_value(flags, "--cache-type-v") or "f16")
    has_mmproj = flag_present(flags, "--mmproj")
    b = mem_budget(cfg.device, cfg.model, ctx, par, k_bits, v_bits,
                   include_mmproj=has_mmproj)

    # 吞吐：同一类别只取一次 gain。多个同类别方案时取「偏离 1.0 最远」的那个
    # —— 注意不能对 1.0 取 max，否则 <1 的摊薄系数（如并发的 0.91）会被抹掉。
    by_cat: dict[str, float] = {}
    sys_tps = None
    for pid in bundle.get("plans", []):
        pl = plans_by_id.get(pid)
        if not pl:
            continue
        g = pl.get("gain", {})
        cat = pl["category"]
        v = g.get("decode_tps")
        if v and cat in ("speculative", "kernel", "scheduling"):
            v = float(v)
            cur = by_cat.get(cat)
            if cur is None or abs(math.log(v)) > abs(math.log(cur)):
                by_cat[cat] = v
        if g.get("decode_tps_system"):
            sys_tps = base_tps * float(g["decode_tps_system"])
    tps = base_tps
    for v in by_cat.values():
        tps *= v
    return {
        "decode_tps_est": round(tps, 2),
        "decode_tps_system_est": round(sys_tps, 2) if sys_tps else None,
        "base_tps": round(base_tps, 2),
        "peak_vram_gib": b["total_gib"],
        "safety_margin_gib": b["safety_margin_gib"],
        "ctx_total": ctx,
        "parallel": par,
        "kv_bits": {"k": k_bits, "v": v_bits},
        "fits": b["fits"],
    }


def build_command(cfg: PerfConfig, flags: list[str]) -> str:
    """
    生成可直接执行的 llama-server 命令。

    两个约定：
      · 占位符 {draft} / {mmproj} 从 `model.artifacts` 的 role 解析，
        **不硬编码文件名** —— 换模型时文件名几乎必然不同。
      · 二进制用 `"$SERVER_BIN"` 而不是 `./llama-server`：
        前者由脚本里的环境变量决定（默认走 PATH），后者只在
        「可执行文件恰好与脚本同目录」时成立，实际几乎从不成立。
    """
    base = cfg.model.get("file", "<model.gguf>")
    d = Path(base).parent
    roles = {a.get("role"): str(d / a.get("file", ""))
             for a in (cfg.model.get("artifacts") or []) if a.get("role")}
    mapping = {"{draft}": roles.get("draft_mtp", "<draft.gguf>"),
               "{mmproj}": roles.get("mmproj", "<mmproj.gguf>"),
               "{model}": base}
    resolved = []
    for f in flags:
        for k, v in mapping.items():
            f = f.replace(k, v)
        resolved.append(f)
    lines = ['"$SERVER_BIN"', f"  -m {base}"] + [f"  {f}" for f in resolved]
    if not any("--host" in x for x in lines):
        lines += ["  --host 0.0.0.0", "  --port 8000",
                  f'  --alias {cfg.model["name"].lower()}']
    return " \\\n".join(lines)


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    diag = load_json(cfg.out("diagnose.json"))
    baseline = load_json(cfg.out("baseline.json"))
    route = diag["optimization_route"]

    # 原生基线吞吐：优先取台账实测值，缺失时退回单流场景的实测 decode。
    # 绝不硬编码 —— 这是所有收益估算的锚点。
    base_tps = baseline["extra"].get("baseline_native_decode_tps")
    if not base_tps:
        for _s in (baseline.get("scenarios") or {}).values():
            if _s.get("decode_tps") and _s.get("parallel", 1) == 1:
                base_tps = _s["decode_tps"]
                break
    if not base_tps:
        print("[optimize] 警告：基线中无 decode 实测值，收益估算将不可用", file=sys.stderr)
        base_tps = 1.0

    # 从瓶颈结论补充 route 标签
    tags = []
    for f in diag["findings"]:
        tags.append(f["type"])
        if f.get("subtype"):
            tags.append(f["subtype"])
    # 后端与设备类型也是方案适用性的判据（例如 CPU 专属内核只在 cpu 后端适用）
    backend = cfg.task["task"].get("backend")
    kind = cfg.device.get("kind")
    if backend:
        tags.append(f"{backend}_backend")
    if kind:
        tags.append(kind)
    if bool(cfg.device.get("memory", {}).get("unified_memory")):
        tags.append("unified_memory")
    pt = cfg.device.get("power_thermal", {})
    if pt.get("thermal_throttle"):
        tags += ["thermal_bound", "power_bound"]
    if pt.get("battery"):
        tags.append("battery")
    ctx = {"tags": tags}

    plans = cfg.plan.get("plans", [])
    plans_by_id = {p["id"]: p for p in plans}

    # 「适用」与「已采用」是两件事：
    #   适用   —— 该方案对症（可作候选）
    #   已采用 —— 该方案真的进了 selected_bundle
    # 只看 applicable_when 会把「已在选定配置里用了的方案」误判为不适用
    # （例如 sched_parallel2 的触发条件是 throughput_bound，而本任务的
    #   主瓶颈是 memory_bound —— 但它明明就在选定 bundle 里）。
    # 因此把选定 bundle 的 plans 也作为标签注入。
    _sel_bundle = cfg.selected_bundle()
    _used_ids = set(_sel_bundle.get("plans") or [])
    ctx["tags"] = sorted(set(ctx["tags"]) | {f"plan:{pid}" for pid in _used_ids})

    selected, rejected = [], []
    for pl in plans:
        ok, why = applicable(pl, route, ctx)
        _used = pl["id"] in _used_ids
        if _used and not ok:
            ok, why = True, "已在选定 bundle 中采用"
        _rec = {**pl, "_why": why, "_used_in_selected": _used}
        (selected if ok else rejected).append(_rec)

    # 按 route 顺序排序，保证「最对症」的方案排在最前
    def rank(pl):
        for i, cat in enumerate(route):
            if pl.get("category") == cat:
                return i
        return len(route) + 1
    selected.sort(key=rank)

    # ---- 组装候选 bundle ---------------------------------------------------
    # 回滚目标 = 优化基线（而不是「清空所有 flag」的空配置）
    rollback_flags = (cfg.plan.get("optimization_baseline", {}).get("startup_flags") or [])
    bundles = []
    for b in cfg.plan.get("bundles", []):
        est = estimate_bundle(cfg, b, plans_by_id, base_tps)
        bundles.append({
            "id": b["id"], "name": b["name"],
            "plans": b["plans"],
            # 方案明细内联进 bundle，让报告/图表无需回头查 plan 库即可
            # 完整归因「这条收益来自哪个方案、依据是什么」。
            "plan_details": [
                {"id": pid,
                 "name": plans_by_id[pid]["name"],
                 "category": plans_by_id[pid]["category"],
                 "gain": plans_by_id[pid].get("gain", {}),
                 "source": plans_by_id[pid].get("source", ""),
                 "cost": plans_by_id[pid].get("cost", ""),
                 "risk": plans_by_id[pid].get("risk", ""),
                 "rollback": plans_by_id[pid].get("rollback", "")}
                for pid in b["plans"] if pid in plans_by_id
            ],
            "startup_flags": b["startup_flags"],
            "estimated": est,
            "expected_from_config": b.get("expected", {}),
            "measured": b.get("measured", {}),
            "measured_source": b.get("measured_source"),
            "command": build_command(cfg, b["startup_flags"]),
            "rollback_command": build_command(cfg, rollback_flags),
            "is_selected": b["id"] == cfg.plan.get("selected_bundle"),
            "is_baseline": b["id"] == cfg.plan.get("optimization_baseline", {}).get("id"),
        })

    sel_bundle = next((b for b in bundles if b["is_selected"]), bundles[0] if bundles else None)
    base_bundle = next((b for b in bundles if b["is_baseline"]), None)

    out = {
        "route": route,
        "primary_bottleneck": diag["primary_bottleneck"],
        "selected_plans": selected,
        "rejected_plans": rejected,
        "bundles": bundles,
        "selected_bundle": sel_bundle,
        "startup_command": sel_bundle["command"] if sel_bundle else "",
        "rollback_command": sel_bundle["rollback_command"] if sel_bundle else "",
        "guardrails": cfg.task["acceptance"]["guardrail"],
    }
    path = save_json(out, cfg.out("optimize.json"))

    # 生成可执行脚本
    script = cfg.out("start_optimized.sh")
    _bin_hint = {
        "gpu": "llama-server（CUDA 构建）",
        "cpu": "llama-server（CPU 构建）",
    }.get(cfg.backend.kind, "llama-server")
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# 由 perfkit optimize 自动生成 —— 优化后配置启动脚本\n"
        "# 回滚：见 optimize.json 的 rollback_command\n"
        "set -euo pipefail\n"
        "# 二进制路径：默认走 PATH；不在 PATH 里时显式覆盖，例如\n"
        f"#   SERVER_BIN=/path/to/build/bin/llama-server bash {script.name}\n"
        f'# 需要的是 {_bin_hint}（与本次测量所用的构建一致）\n'
        'SERVER_BIN="${SERVER_BIN:-llama-server}"\n'
        'if ! command -v "$SERVER_BIN" >/dev/null 2>&1; then\n'
        '  echo "[perfkit] 找不到 $SERVER_BIN —— 请用 SERVER_BIN=... 指定" >&2\n'
        '  exit 127\n'
        'fi\n'
        "echo '[perfkit] 启动优化配置...'\n"
        f"{out['startup_command']}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    print("=" * 74)
    print(f"优化方案选型 · {cfg.model['name']}  [run={cfg.run_id}]")
    print("=" * 74)
    print(f"\n瓶颈路线  {' → '.join(route)}")
    print(f"\n适用方案（{len(selected)} 条，按对症程度排序）")
    for pl in selected:
        g = pl.get("gain", {})
        gain_str = ", ".join(f"{k}={v}" for k, v in g.items())
        print(f"  [{pl['category']:<12}] {pl['id']:<24} {pl['name']}")
        print(f"     收益 {gain_str}   来源 {pl.get('source', '')}")
        print(f"     风险 {pl.get('risk', '').strip()[:70]}")

    print(f"\n候选 bundle 对比（吞吐估算以台账原生基线 "
          f"{base_tps:.2f} t/s 为锚）")
    print(f"  {'ID':<20} {'方案':<24} {'ctx':>7} {'并':>3} "
          f"{'实测t/s':>8} {'估算t/s':>8} {'峰值GiB':>9} {'余量':>7}")
    for b in bundles:
        e = b["estimated"]
        m = b.get("measured") or {}
        mark = " ←选定" if b["is_selected"] else ""
        _meas = m.get("decode_tps")
        print(f"  {b['id']:<20} {b['name'][:22]:<24} {e['ctx_total']:>7} "
              f"{e['parallel']:>3} "
              f"{(f'{_meas:.2f}' if _meas else '—'):>8} "
              f"{e['decode_tps_est']:>8.2f} "
              f"{e['peak_vram_gib']:>9.2f} {e['safety_margin_gib']:>7.2f}{mark}")
    # 估算只按「权重字节 × 方案增益」推，不建模上下文深度。
    # 若各 bundle 的实测深度不同，估算值会系统性偏离，必须显式提示。
    _depths = {(b.get("measured") or {}).get("n_depth", 0) for b in bundles}
    print(f"\n  注：估算列只按方案增益与权重字节推算，**不建模上下文深度**；"
          f"实测列才是该配置在该深度下的真值，两者不可互换。")

    print(f"\n选定配置启动命令：\n")
    print("\n".join("    " + ln for ln in out["startup_command"].splitlines()))
    print(f"\n回滚命令（恢复基线）：\n")
    print("\n".join("    " + ln for ln in out["rollback_command"].splitlines()))
    print(f"\n产出 → {path}")
    print(f"     → {script}")
    print("=" * 74)
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("optimize", help="【第5步】优化方案生成：筛选/排序/组 bundle/出启动脚本")
    p.set_defaults(func=main)
    return p
