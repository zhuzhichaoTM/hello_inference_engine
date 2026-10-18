"""
perfkit query —— 跨运行、跨硬件的统一查询接口
==============================================================================
闭环跑完之后，产物是一堆 JSON。本命令提供**一组固定的查询指令**，
让人不必记住目录结构就能回答四类问题：

  1. 有哪些运行？           perfkit query --list
  2. 某次运行的某个指标是多少、口径是什么？
                            perfkit query --run X --kpi decode_tps
  3. 多次运行横向对比？      perfkit query --runs a,b,c
  4. 框架支持哪些硬件后端？  perfkit query --backends
     某次运行用了什么配置？  perfkit query --run X --config

设计约束
------------------------------------------------------------------------------
· **只读**：query 不修改任何产物，可以随时跑。
· **不猜**：查不到就明确说「无数据」并指出缺哪一步产物，
  而不是回退到估算值 —— 那会让人误以为拿到的是实测。
· **带口径**：每个数值都连同它的来源标记（measured/model/derived/config）一起返回，
  这是本框架「数字可追溯」原则在查询接口上的延续。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..backends import backend_summary
from ..config import PerfConfig, load_json


# ============================================================================
# 运行发现
# ============================================================================

def _list_runs(root: Path) -> list[dict]:
    """扫描 output/ 下所有含 baseline.json 的目录，汇总成运行清单。"""
    outdir = root / "output"
    runs = []
    if not outdir.exists():
        return runs
    for d in sorted(outdir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if not (d / "baseline.json").exists():
            continue
        runs.append(_run_summary(d))
    return runs


def _run_summary(run_dir: Path) -> dict:
    """读一份运行目录的关键信息。缺哪份产物就少哪些字段。"""
    s: dict = {"run_id": run_dir.name, "dir": str(run_dir)}
    files = {p.name for p in run_dir.glob("*")}
    s["artifacts"] = sorted(files)
    s["complete"] = {"definition.json", "baseline.json", "profiling.json",
                     "diagnose.json", "optimize.json", "verify.json",
                     "report.md"} <= files
    try:
        b = load_json(run_dir / "baseline.json")
        s["mode"] = b.get("mode")
        s["backend"] = b.get("backend") or (b.get("meta") or {}).get("backend")
        s["model"] = (b.get("meta") or {}).get("model")
        s["quant"] = (b.get("meta") or {}).get("quant")
        s["device"] = (b.get("meta") or {}).get("device")
        s["native_decode_tps"] = (b.get("extra") or {}).get("baseline_native_decode_tps")
        s["decode_upper_bound"] = (b.get("_derived") or {}).get("decode_tps_upper_bound")
    except Exception as e:
        s["error"] = f"baseline.json 读取失败: {e}"
    try:
        v = load_json(run_dir / "verify.json")
        s["verdict"] = v.get("verdict")
        s["after_bundle"] = (v.get("after_bundle") or {}).get("id")
        rows = {r["key"]: r for r in v.get("comparison", [])}
        s["decode_tps_after"] = (rows.get("decode_tps") or {}).get("after")
        s["ctx_after"] = (rows.get("ctx_total") or {}).get("after")
        # ctx_total 不是每个任务的 KPI（如 CPU 任务关心的是线程缩放），
        # 此时从 bundle 的启动参数里取 --ctx-size，保证这一列总有意义。
        if s["ctx_after"] is None:
            from ..config import flag_value
            s["ctx_after"] = flag_value(
                (v.get("after_bundle") or {}).get("flags") or [], "--ctx-size", int)
    except Exception:
        pass
    try:
        d = load_json(run_dir / "diagnose.json")
        s["primary_bottleneck"] = (d.get("primary_bottleneck") or {}).get("type")
    except Exception:
        pass
    return s


def _resolve_run(cfg: PerfConfig, run_id: str | None) -> Path:
    rid = run_id or cfg.run_id
    d = cfg.root / "output" / rid
    if not d.exists():
        raise FileNotFoundError(
            f"运行目录不存在: {d}\n"
            f"提示：先跑 `perfkit all --run {rid}`，或用 `perfkit query --list` 看现有运行。")
    return d


# ============================================================================
# 查询动作
# ============================================================================

def q_list(cfg: PerfConfig, args) -> int:
    runs = _list_runs(cfg.root)
    if not runs:
        print(f"（{cfg.root / 'output'} 下没有找到任何运行）")
        print("先跑一次：python3 -m perfkit all --run my-first-run")
        return 0
    # 列宽按实际内容自适应：模型名与量化名长度差异很大
    # （Qwen2.5-1.5B vs Qwen3.8-27B-UD-Q4_K_M），写死宽度会串列。
    w_run = max(8, max(len(r["run_id"]) for r in runs)) + 2
    w_model = max(6, min(20, max(len(str(r.get("model") or "—")) for r in runs))) + 2
    w_quant = max(4, min(14, max(len(str(r.get("quant") or "—")) for r in runs))) + 2
    hdr = (f'{"run_id":<{w_run}} {"后端":<8} {"模型":<{w_model}} '
           f'{"量化":<{w_quant}} {"原生t/s":>9} {"优化后t/s":>10} '
           f'{"ctx":>7} {"主瓶颈":<18} {"验收":<7}')
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        model = str(r.get("model") or "—")
        quant = str(r.get("quant") or "—")
        if len(model) > w_model - 2:
            model = model[:w_model - 3] + "…"
        if len(quant) > w_quant - 2:
            quant = quant[:w_quant - 3] + "…"
        print(f'{r["run_id"]:<{w_run}} {str(r.get("backend") or "—"):<8} '
              f'{model:<{w_model}} {quant:<{w_quant}} '
              f'{_n(r.get("native_decode_tps"), 9)} {_n(r.get("decode_tps_after"), 10)} '
              f'{_n(r.get("ctx_after"), 7)} '
              f'{str(r.get("primary_bottleneck") or "—"):<18} '
              f'{str(r.get("verdict") or "—"):<7}')
    print(f"\n共 {len(runs)} 次运行。"
          f"用 --run <id> --kpi <key> 查具体指标，--runs a,b,c 横向对比。")
    return 0


def _n(v, width: int) -> str:
    if v is None:
        return f'{"—":>{width}}'
    if isinstance(v, float):
        return f"{v:>{width}.2f}"
    return f"{v:>{width}}"


def q_kpi(cfg: PerfConfig, args) -> int:
    run_dir = _resolve_run(cfg, args.run)
    ver_p, base_p = run_dir / "verify.json", run_dir / "baseline.json"
    if not ver_p.exists():
        print(f"该运行还没有 verify.json（第6步未跑）。")
        print(f"先执行：python3 -m perfkit verify --run {run_dir.name}")
        return 3
    ver = load_json(ver_p)
    rows = {r["key"]: r for r in ver.get("comparison", [])}

    keys = [args.kpi] if args.kpi else list(rows)
    missing = [k for k in keys if k not in rows]
    for k in missing:
        print(f"[{k}] 无数据：该运行未定义此 KPI"
              f"（可用 --kpis 查看已定义的指标）")
    print(f"运行 {run_dir.name} · {len(keys) - len(missing)} 项指标")
    print()
    for k in keys:
        r = rows.get(k)
        if not r:
            continue
        print(f"  {r['key']}  （{r['name']}）")
        print(f"    优化前    {_n(r.get('before'), 12)} {r['unit']:<10} "
              f"[{r.get('before_source', '—')}]")
        print(f"    优化后    {_n(r.get('after'), 12)} {r['unit']:<10} "
              f"[{r.get('after_source', '—')}]")
        print(f"    变化      {_pct(r.get('delta_pct'))}")
        print(f"    目标      {_n(r.get('target'), 12)} {r['unit']:<10} "
              f"达标={'是' if r.get('meets_target') else ('否' if r.get('meets_target') is False else '—')}")
        if r.get("after_note"):
            print(f"    口径备注  {r['after_note']}")
        print(f"    定义      {r['definition'].strip().splitlines()[0][:70]}")
        print()

    # 顺带把该运行的基础事实也列出来，避免为了几个数字来回切命令
    if base_p.exists():
        b = load_json(base_p)
        ub = (b.get("_derived") or {}).get("decode_tps_upper_bounds") or {}
        print(f"  参考事实")
        print(f"    后端          {b.get('backend') or (b.get('meta') or {}).get('backend') or '—'}")
        print(f"    数据模式      {b.get('mode')}")
        print(f"    理论上界      {ub.get('no_kv_tps')} t/s（不计 KV）"
              f" / {ub.get('at_ctx_tps')} t/s（含 {ub.get('ctx')} ctx 的 KV 读）")
        ro = (b.get("_derived") or {}).get("roofline") or {}
        print(f"    roofline      {ro.get('verdict')}（强度 {ro.get('intensity')}"
              f" vs 拐点 {ro.get('ridge_point')}）")
    return 0


def _pct(v) -> str:
    return f"{v:+.2f}%" if isinstance(v, (int, float)) else "—"


def q_compare(cfg: PerfConfig, args) -> int:
    ids = [x.strip() for x in (args.runs or "").split(",") if x.strip()]
    if len(ids) < 2:
        print("--runs 需要至少两个 run_id，用逗号分隔，例如："
              "--runs qwen3.8-27b-decode-opt,qwen2.5-1.5b-cpu")
        return 2

    data = {}
    for rid in ids:
        d = cfg.root / "output" / rid
        if not (d / "verify.json").exists():
            print(f"[{rid}] 缺少 verify.json，跳过")
            continue
        ver = load_json(d / "verify.json")
        base = load_json(d / "baseline.json") if (d / "baseline.json").exists() else {}
        rows = {r["key"]: r for r in ver.get("comparison", [])}
        data[rid] = {
            "rows": rows,
            "backend": base.get("backend") or (base.get("meta") or {}).get("backend"),
            "model": (base.get("meta") or {}).get("model"),
            "verdict": ver.get("verdict"),
        }
    if not data:
        print("没有可对比的运行。")
        return 3

    all_keys = []
    for v in data.values():
        for k in v["rows"]:
            if k not in all_keys:
                all_keys.append(k)

    print("运行对比（单元格为「优化后」值）")
    print()
    hdr = f"{'指标':<20} " + " ".join(f"{rid[:18]:>20}" for rid in data)
    print(hdr)
    print("-" * len(hdr))
    for k in all_keys:
        cells = []
        for rid in data:
            r = data[rid]["rows"].get(k)
            if not r:
                cells.append(f"{'—':>20}")
            else:
                v = r.get("after")
                mark = {"measured": "M", "model": "m", "derived": "d",
                        "config": "c", "todo": "?"}.get(r.get("after_source"), "?")
                cells.append(f"{(f'{v:.2f}' if isinstance(v, float) else str(v)) + ' [' + mark + ']':>20}")
        print(f"{k:<20} " + " ".join(cells))
    print()
    print("口径标记：M=实测  m=几何模型  d=实测推导  c=配置声明  ?=待采集")
    print()
    for rid, v in data.items():
        print(f"  {rid:<28} 后端={v['backend'] or '—':<8} "
              f"模型={str(v['model'] or '—')[:20]:<22} 验收={v['verdict']}")
    return 0


def q_backends(cfg: PerfConfig, args) -> int:
    print("已注册的硬件后端（见 perfkit/backends.py）")
    print()
    print(f"{'kind':<10} {'实现类':<24} {'说明':<34} {'内存称呼':<14}")
    print("-" * 88)
    for b in backend_summary():
        print(f"{b['kind']:<10} {b['class']:<24} {b['label']:<34} {b['memory_label']:<14}")
    print()
    print("当前配置使用的后端：")
    print(f"  kind       {cfg.backend.kind}")
    print(f"  实现类     {type(cfg.backend).__name__}")
    print(f"  说明       {cfg.backend.label}")
    print(f"  内存称呼   {cfg.backend.memory_label}")
    print(f"  加速标志   {' '.join(cfg.backend.accel_flags()) or '（无）'}")
    print(f"  回退代价   {'廉价（同物理内存）' if cfg.backend.fallback_is_cheap() else '昂贵（需跨总线搬运）'}")
    bwc = cfg.backend.bandwidth_probe_command("build/ggml_probe")
    print(f"  带宽探针   {' '.join(bwc) if bwc else '（该后端无可用探针）'}")
    notes = cfg.backend.notes()
    if notes:
        print()
        print("该后端注意事项：")
        for n in notes:
            print(f"  · {n}")
    return 0


def q_config(cfg: PerfConfig, args) -> int:
    run_dir = _resolve_run(cfg, args.run)
    defn_p = run_dir / "definition.json"
    if not defn_p.exists():
        print(f"[{run_dir.name}] 缺少 definition.json（第1步未跑）")
        return 3
    defn = load_json(defn_p)
    print(f"运行 {run_dir.name} 的配置事实")
    print()
    m, d = defn.get("model", {}), defn.get("device", {})
    print(f"  模型      {m.get('name')} {m.get('quant')} "
          f"（{m.get('params_total', 0)/1e9:.2f}B，{m.get('weight_gib')} GiB）")
    print(f"  设备      {d.get('name')}（kind={d.get('kind')}，"
          f"{d.get('mem_total_gib') or d.get('vram_gib')} GiB，"
          f"{d.get('bw_gbps')} GB/s）")
    print(f"  统一内存  {d.get('unified_memory')}")
    print(f"  任务      scope={defn.get('task', {}).get('scope')}  "
          f"priority={defn.get('task', {}).get('priority')}")
    print(f"  选定方案  {defn.get('selected_bundle')}")
    if defn.get("reconcile"):
        print(f"  待对账项  {list(defn['reconcile'].keys())}")
    print()
    print(f"  {'KPI':<20} {'目标':>12} {'单位':<10} 方向")
    print("  " + "-" * 56)
    for k in defn.get("kpis", []):
        print(f"  {k['key']:<20} {str(k.get('target')):>12} "
              f"{str(k.get('unit')):<10} {k.get('direction')}")
    print()
    print("  产物清单：")
    for a in sorted(p.name for p in run_dir.glob("*")):
        print(f"    {a}")
    return 0


# ============================================================================

def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)

    if getattr(args, "backends", False):
        return q_backends(cfg, args)
    if getattr(args, "list", False):
        return q_list(cfg, args)
    if getattr(args, "runs", None):
        return q_compare(cfg, args)
    if getattr(args, "kpi", None) is not None or getattr(args, "kpis", False):
        if getattr(args, "kpis", False) and not getattr(args, "kpi", None):
            args.kpi = None
        return q_kpi(cfg, args)
    if getattr(args, "facts", False):
        return q_config(cfg, args)

    # 无参数：给一份概览，比报错更有用
    return q_list(cfg, args)


def add_parser(sub) -> None:
    p = sub.add_parser("query", help="跨运行查询：运行清单 / 单指标 / 横向对比 / 后端清单")
    p.add_argument("--list", action="store_true", help="列出所有运行及其摘要")
    p.add_argument("--kpi", default=None, metavar="KEY",
                   help="查某个 KPI（如 decode_tps）；不带则列出全部")
    p.add_argument("--kpis", action="store_true", help="列出该运行的全部 KPI")
    p.add_argument("--runs", default=None, metavar="A,B,C",
                   help="横向对比多次运行（逗号分隔的 run_id）")
    p.add_argument("--backends", action="store_true", help="列出已注册的硬件后端")
    p.add_argument("--facts", action="store_true",
                   help="显示该运行的配置事实（模型/设备/KPI/产物清单）")
    p.set_defaults(func=main)
    return p
