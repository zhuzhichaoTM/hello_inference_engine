"""
perfkit visualize —— 性能剖面可视化（自包含 HTML + 内联 SVG）
==============================================================================
产出 output/<run>/charts.html：零外部依赖、零 JS 框架，浏览器直接打开。
六张图分别回答一个具体问题，不做装饰性图表。

配色遵循可访问性原则：分类色互相可区分、深浅对比满足 WCAG AA，
浅底与深底均可用（颜色不承载唯一信息，均带直接标注）。
"""

from __future__ import annotations

from ..config import PerfConfig, flag_value, load_json

# ---------------------------------------------------------------------------
# 调色板（语义化：同一含义在所有图中使用同一颜色）
# ---------------------------------------------------------------------------
C = {
    "weight":    "#4263EB",   # 权重
    "kv":        "#0CA678",   # KV Cache
    "mmproj":    "#F08C00",   # 视觉投影
    "runtime":   "#868E96",   # 运行时
    "margin":    "#DEE2E6",   # 安全冗余
    "baseline":  "#ADB5BD",   # 基线
    "optimized": "#4263EB",   # 优化后
    "upper":     "#E8590C",   # 理论上界
    "prefill":   "#7048E8",
    "decode":    "#0CA678",
    "warn":      "#E03131",
    "grid":      "#E9ECEF",
    "ink":       "#212529",
    "ink_dim":   "#868E96",
}

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', sans-serif")
MONO = "'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _mem_label(device: dict) -> str:
    """该后端下「加速器内存」的称呼（显存 / 内存 / 统一内存 / App 配额）。"""
    from ..backends import get_backend
    return get_backend(device).memory_label


def _peak_label(device: dict) -> str:
    """「峰值内存」指标的显示名（由后端决定完整措辞）。"""
    from ..backends import get_backend
    return get_backend(device).peak_metric_label


def _mem_total_gib(device: dict) -> float | None:
    from ..backends import get_backend
    v = get_backend(device).mem_total_bytes()
    return round(v / 1024 ** 3, 2) if v else None


def _runtime_label(device: dict) -> str:
    """页眉里的运行时标签：GPU 显示 CUDA 版本，其余显示后端名。"""
    sw = device.get("software", {})
    cuda = sw.get("cuda")
    if isinstance(cuda, (list, tuple)) and cuda:
        return f"CUDA {cuda[0]}"
    return sw.get("engine_backend") or sw.get("os") or "—"


# ---------------------------------------------------------------------------
# 图元
# ---------------------------------------------------------------------------

def _svg_open(w: int, h: int) -> list[str]:
    return [f'<svg viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px" role="img">']


def _title(w: int, title: str, sub: str = "") -> list[str]:
    out = [f'<text x="0" y="18" font-family="{FONT}" font-size="15" '
           f'font-weight="600" fill="{C["ink"]}">{_esc(title)}</text>']
    if sub:
        out.append(f'<text x="0" y="36" font-family="{FONT}" font-size="11.5" '
                   f'fill="{C["ink_dim"]}">{_esc(sub)}</text>')
    return out


def _axis_label(x: int, y: int, text: str, anchor: str = "middle") -> str:
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="10.5" '
            f'fill="{C["ink_dim"]}" text-anchor="{anchor}">{_esc(text)}</text>')


def _empty_chart(title: str, reason: str, hint: str = "") -> str:
    """
    数据缺失时的占位图。

    为什么要有它，而不是「直接不画这张图」：
      · 图表清单是固定的七张，少一张会让读者以为「这个维度不重要」；
      · 但画一张空坐标轴更糟 —— 会被误读成「指标为 0」。
    因此画一个明确的「无数据」框，把**缺什么、怎么补**写在图上。

    live 模式下这尤其重要：只跑部分场景时 KV 剖面、长上下文曲线都可能为空，
    此时正确的行为是如实标注，而不是让 `max([])` 抛 ValueError 把整份
    charts.html 一起废掉（这是真实发生过的崩溃）。
    """
    W, H = 720, 130
    s = _svg_open(W, H)
    s += _title(W, title, "本次运行缺少该维度的数据")
    s.append(f'<rect x="0" y="52" width="{W-2}" height="62" rx="6" '
             f'fill="{C["grid"]}" opacity="0.45"/>')
    s.append(f'<text x="16" y="78" font-family="{FONT}" font-size="12.5" '
             f'fill="{C["ink"]}">缺什么：{_esc(reason)}</text>')
    if hint:
        s.append(f'<text x="16" y="100" font-family="{FONT}" font-size="11.5" '
                 f'fill="{C["ink_dim"]}">怎么补：{_esc(hint)}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 图 1：内存瀑布（横向堆叠条，语义随后端：显存/内存/配额）
# ---------------------------------------------------------------------------

def chart_memory(defn: dict, base: dict, cfg: PerfConfig | None = None) -> str:
    b = defn.get("memory_budget")
    if not b or not b.get("vram_total_gib"):
        return _empty_chart(
            "内存占用分解与安全冗余",
            "definition.json 里没有 memory_budget（或内存总量为 0）",
            "检查 device_*.yaml 的 memory 段是否有 vram_total_mib / ram_total_gib")
    segs = [("主模型权重", b["weight_gib"], C["weight"]),
            ("mmproj", b["mmproj_gib"], C["mmproj"]),
            ("KV Cache", b["kv_total_gib"], C["kv"]),
            ("运行时", b["runtime_gib"], C["runtime"])]
    total = b["vram_total_gib"]
    W, H = 720, 200
    x0, y0, bw, bh = 12, 70, 696, 46
    s = _svg_open(W, H)
    s += _title(W, "内存占用分解与安全冗余",
                f'{b["total_gib"]} / {total} GiB 已用'
                f'（{b["total_gib"]/total*100:.1f}%），冗余 {b["safety_margin_gib"]} GiB')
    # 已用部分
    cx = x0
    for name, v, col in segs:
        if v <= 0:
            continue
        w = bw * v / total
        s.append(f'<rect x="{cx:.1f}" y="{y0}" width="{w:.1f}" height="{bh}" '
                 f'fill="{col}"/>')
        if w > 46:
            s.append(f'<text x="{cx + w/2:.1f}" y="{y0 + bh/2 + 4:.1f}" '
                     f'font-family="{MONO}" font-size="11" fill="#fff" '
                     f'text-anchor="middle">{v:.2f}</text>')
        cx += w
    # 冗余部分
    mw = bw * b["safety_margin_gib"] / total
    s.append(f'<rect x="{cx:.1f}" y="{y0}" width="{mw:.1f}" height="{bh}" '
             f'fill="{C["margin"]}" stroke="{C["grid"]}"/>')
    s.append(f'<text x="{cx + mw/2:.1f}" y="{y0 + bh/2 + 4:.1f}" '
             f'font-family="{MONO}" font-size="11" fill="{C["ink_dim"]}" '
             f'text-anchor="middle">{b["safety_margin_gib"]:.2f}</text>')
    # 图例
    lx = x0
    for name, v, col in segs + [("安全冗余", b["safety_margin_gib"], C["margin"])]:
        s.append(f'<rect x="{lx}" y="{y0 + bh + 22}" width="10" height="10" fill="{col}"/>')
        s.append(f'<text x="{lx + 15}" y="{y0 + bh + 31}" font-family="{FONT}" '
                 f'font-size="11" fill="{C["ink"]}">{_esc(name)}</text>')
        lx += 22 + len(name) * 12
    s.append(_axis_label(x0, y0 - 10, f"0", "start"))
    _label = (_mem_label(cfg.device) if cfg is not None else "内存")
    s.append(_axis_label(x0 + bw, y0 - 10,
                         f"{total:.1f} GiB（{_label}总量）", "end"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 图 2：KV Cache 随上下文增长（折线 + 实测点）
# ---------------------------------------------------------------------------

def chart_kv(defn: dict, base: dict, prof: dict) -> str:
    rows = (prof.get("kv_cache") or {}).get("ladder") or []
    if not rows:
        # 没有深度扫描就没有这条曲线。live 模式只跑不带 depth 的场景时
        # 就是这种情况 —— 直接 max([]) 会 ValueError，把整份 charts.html 废掉。
        return _empty_chart(
            "KV Cache 随上下文增长",
            "长上下文曲线为空（场景矩阵里没有带 depth 的点）",
            "在 task_*.yaml 的场景里加 depth: <tokens>，或在台账 "
            "curves.long_context 里补点")
    # 实测点：台账曲线里的 kv_gib 可能是 None（live 模式没有逐点实测），
    # 直接进 max() 会 TypeError —— 这是 live 路径上真实崩过的地方。
    # 过滤掉 None 后如果为空，就不画实测点，而不是画一条 0 线。
    meas = {r["ctx"]: r["kv_gib"]
            for r in (base.get("extra") or {}).get("long_context_curve") or []
            if isinstance(r.get("kv_gib"), (int, float))}
    W, H = 720, 260
    L, R, T, B = 62, 20, 56, 44
    pw, ph = W - L - R, H - T - B
    xs = [r["ctx"] for r in rows]
    xmax = max(xs) or 1
    ymax = max(max(r["kv_gib_model"] for r in rows),
               max(meas.values()) if meas else 0) * 1.18 or 1.0

    def px(v): return L + pw * v / xmax
    def py(v): return T + ph - ph * v / ymax

    s = _svg_open(W, H)
    s += _title(W, "KV Cache 随上下文线性增长（本次优化的第一性瓶颈）",
                "权重是常数，KV 随 ctx 线性增长 —— 上下文越长，KV 量化收益越大")
    # 网格
    for i in range(5):
        v = ymax * i / 4
        s.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{L+pw}" y2="{py(v):.1f}" '
                 f'stroke="{C["grid"]}" stroke-width="1"/>')
        s.append(_axis_label(L - 8, py(v) + 4, f"{v:.1f}", "end"))
    for xv in (0, 16384, 32768, 49152, 65536):
        if xv > xmax:
            continue
        s.append(_axis_label(px(xv), T + ph + 18, f"{xv//1024}K"))
    s.append(_axis_label(L + pw/2, H - 8, "上下文长度 (tokens)"))
    s.append(f'<text x="14" y="{T + ph/2}" font-family="{FONT}" font-size="10.5" '
             f'fill="{C["ink_dim"]}" transform="rotate(-90 14 {T+ph/2})" '
             f'text-anchor="middle">KV Cache (GiB)</text>')

    # 模型预测曲线
    pts = " ".join(f"{px(r['ctx']):.1f},{py(r['kv_gib_model']):.1f}" for r in rows)
    s.append(f'<polyline points="{pts}" fill="none" stroke="{C["kv"]}" stroke-width="2.5"/>')
    for r in rows:
        s.append(f'<circle cx="{px(r["ctx"]):.1f}" cy="{py(r["kv_gib_model"]):.1f}" '
                 f'r="3.5" fill="{C["kv"]}"/>')
        s.append(f'<text x="{px(r["ctx"]):.1f}" y="{py(r["kv_gib_model"])-9:.1f}" '
                 f'font-family="{MONO}" font-size="10" fill="{C["kv"]}" '
                 f'text-anchor="middle">{r["kv_gib_model"]:.2f}</text>')
    # 实测点
    for ctx, kv in meas.items():
        s.append(f'<circle cx="{px(ctx):.1f}" cy="{py(kv):.1f}" r="4.5" '
                 f'fill="none" stroke="{C["ink"]}" stroke-width="2"/>')
    if meas:
        s.append(f'<text x="{px(max(meas))+6:.1f}" y="{py(meas[max(meas)])+4:.1f}" '
                 f'font-family="{FONT}" font-size="10.5" fill="{C["ink"]}">'
                 f'○ 实测值</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 图 3：算子耗时归因（横向条形）
# ---------------------------------------------------------------------------

def chart_ops(prof: dict) -> str:
    ops = (prof.get("ops") or [])
    if not ops:
        return _empty_chart(
            "Decode 单步耗时归因",
            "算子剖面为空（权重分解与几何公式都没产出条目）",
            "检查 model_*.yaml 的 architecture 字段是否完整，"
            "或 weight_breakdown.ops 是否为空")
    n = len(ops) + 1
    W = 720
    row_h = 26
    H = 60 + n * row_h + 20
    L, pw = 104, 470
    s = _svg_open(W, H)
    _hybrid = bool((prof.get("per_layer") or {}).get("n_ssm_layer"))
    s += _title(W, "Decode 单步耗时归因（按权重读取字节占比）",
                "Memory-bound 下「谁读的字节多，谁就占时间」—— "
                + ("混合架构：FFN 是最大头，其后是 SSM 层的输入/输出投影"
                   if _hybrid else "FFN 三矩阵是最大头"))
    y = 56
    for r in ops:
        w = pw * r["pct_of_step"] / 100
        s.append(f'<text x="{L-10}" y="{y+13}" font-family="{MONO}" font-size="11.5" '
                 f'fill="{C["ink"]}" text-anchor="end">{_esc(r["op"])}</text>')
        s.append(f'<rect x="{L}" y="{y}" width="{w:.1f}" height="18" '
                 f'fill="{C["optimized"]}" rx="2"/>')
        s.append(f'<text x="{L+w+8:.1f}" y="{y+13}" font-family="{MONO}" '
                 f'font-size="11" fill="{C["ink_dim"]}">'
                 f'{r["pct_of_step"]:.2f}%　{r["mib_per_layer"]:.1f} MiB/层</text>')
        y += row_h
    e = prof["per_layer"]
    w = pw * e["embedding_share_pct"] / 100
    s.append(f'<text x="{L-10}" y="{y+13}" font-family="{MONO}" font-size="11.5" '
             f'fill="{C["ink"]}" text-anchor="end">embedding</text>')
    s.append(f'<rect x="{L}" y="{y}" width="{w:.1f}" height="18" '
             f'fill="{C["mmproj"]}" rx="2"/>')
    s.append(f'<text x="{L+w+8:.1f}" y="{y+13}" font-family="{MONO}" font-size="11" '
             f'fill="{C["ink_dim"]}">{e["embedding_share_pct"]:.2f}%　'
             f'{e["embedding_gib"]:.2f} GiB 合计</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 图 4：吞吐对比（理论上界 / 基线 / 各 bundle）
# ---------------------------------------------------------------------------

def chart_throughput(base: dict, opt: dict, ver: dict) -> str:
    ub = base["_derived"]["decode_tps_upper_bound"]
    native = (base.get("extra") or {}).get("baseline_native_decode_tps")
    items = [("理论上界", ub, C["upper"], "带宽 ÷ 权重体积（不含 KV 读）")]
    if native:
        # live 模式若只跑了部分场景，可能推不出「原生基线」——
        # 缺就少画一根柱子，不要用 None 去参与 max()。
        items.append(("优化基线", native, C["baseline"], "本次实测"))
    for b in opt["bundles"]:
        if b.get("is_baseline"):
            continue
        m = b.get("measured") or {}
        e = b["estimated"]
        v = m.get("decode_tps") or e["decode_tps_est"]
        src = "实测" if m.get("decode_tps") else "模型估算"
        items.append((b["id"], v, C["optimized"], f'{b["name"][:14]}（{src}）'))
    W, H = 720, 60 + len(items) * 34 + 20
    L, pw = 132, 470
    vmax = max(v for _, v, _, _ in items) * 1.15
    s = _svg_open(W, H)
    s += _title(W, "Decode 吞吐：理论上界 vs 基线 vs 候选方案",
                f"理论上界 {ub} t/s（{base['theoretical']['vram_bandwidth_gbps']} GB/s "
                f"÷ {base['theoretical']['weight_gib']} GiB）")
    y = 56
    for name, v, col, note in items:
        w = pw * v / vmax
        s.append(f'<text x="{L-10}" y="{y+14}" font-family="{FONT}" font-size="11.5" '
                 f'fill="{C["ink"]}" text-anchor="end">{_esc(name)}</text>')
        s.append(f'<rect x="{L}" y="{y}" width="{w:.1f}" height="22" fill="{col}" rx="2"/>')
        s.append(f'<text x="{L+w+8:.1f}" y="{y+15}" font-family="{MONO}" '
                 f'font-size="11.5" font-weight="600" fill="{C["ink"]}">{v:.1f} t/s</text>')
        s.append(f'<text x="{L+6}" y="{y+15}" font-family="{FONT}" font-size="10" '
                 f'fill="#fff">{_esc(note)}</text>')
        y += 34
    s.append(f'<line x1="{L+pw*ub/vmax:.1f}" y1="52" x2="{L+pw*ub/vmax:.1f}" '
             f'y2="{y}" stroke="{C["upper"]}" stroke-dasharray="4 3" stroke-width="1.5"/>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 图 5：长上下文速率曲线（Prefill / Decode 双轴）
# ---------------------------------------------------------------------------

def chart_longctx(base: dict) -> str:
    rows = (base.get("extra") or {}).get("long_context_curve") or []
    if not rows:
        # 没有深度扫描数据时不画空图，明确说明缺什么、怎么补。
        W, H = 720, 120
        s = _svg_open(W, H)
        s += _title(W, "长上下文速率曲线 —— 无数据",
                    "场景矩阵里没有带 depth 的点；live 模式请给场景加 depth，"
                    "或在台账 curves.long_context 里补点")
        s.append("</svg>")
        return "".join(s)
    W, H = 720, 250
    L, R, T, B = 60, 60, 56, 44
    pw, ph = W - L - R, H - T - B
    xs = [r["ctx"] for r in rows]
    xmax = max(xs)
    ymax = max(r["prefill_tps"] for r in rows) * 1.2
    y2max = max(r["decode_tps"] for r in rows) * 1.5

    def px(v): return L + pw * v / xmax
    def py(v): return T + ph - ph * v / ymax
    def py2(v): return T + ph - ph * v / y2max

    s = _svg_open(W, H)
    s += _title(W, "长上下文下的 Prefill / Decode 速率衰减",
                "Prefill 随 ctx 明显下降（算力被 O(N) 摊薄）；"
                "Decode 几乎恒定（KV 量化抵消了 KV 增长）")
    for i in range(5):
        v = ymax * i / 4
        s.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{L+pw}" y2="{py(v):.1f}" '
                 f'stroke="{C["grid"]}"/>')
        s.append(_axis_label(L - 8, py(v) + 4, f"{v:.0f}", "end"))
    for xv in xs:
        s.append(_axis_label(px(xv), T + ph + 18, f"{xv//1024}K"))

    pf = " ".join(f"{px(r['ctx']):.1f},{py(r['prefill_tps']):.1f}" for r in rows)
    dc = " ".join(f"{px(r['ctx']):.1f},{py2(r['decode_tps']):.1f}" for r in rows)
    s.append(f'<polyline points="{pf}" fill="none" stroke="{C["prefill"]}" stroke-width="2.5"/>')
    s.append(f'<polyline points="{dc}" fill="none" stroke="{C["decode"]}" stroke-width="2.5" '
             f'stroke-dasharray="6 3"/>')
    for r in rows:
        s.append(f'<circle cx="{px(r["ctx"]):.1f}" cy="{py(r["prefill_tps"]):.1f}" r="3.5" '
                 f'fill="{C["prefill"]}"/>')
        s.append(f'<circle cx="{px(r["ctx"]):.1f}" cy="{py2(r["decode_tps"]):.1f}" r="3.5" '
                 f'fill="{C["decode"]}"/>')
    s.append(f'<text x="{L+pw}" y="{py(rows[0]["prefill_tps"])-10:.1f}" '
             f'font-family="{FONT}" font-size="11" fill="{C["prefill"]}" '
             f'text-anchor="end">Prefill（左轴）</text>')
    s.append(f'<text x="{L+pw}" y="{py2(rows[-1]["decode_tps"])+18:.1f}" '
             f'font-family="{FONT}" font-size="11" fill="{C["decode"]}" '
             f'text-anchor="end">Decode（右轴）</text>')
    s.append(_axis_label(L + pw/2, H - 8, "上下文长度 (tokens)"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 图 6：前后对比（分组条形）
# ---------------------------------------------------------------------------

def chart_compare(ver: dict) -> str:
    keys = ["decode_tps", "decode_tps_system", "prefill_tps", "ttft_ms",
            "peak_vram_mib", "safety_margin_gib"]
    rows = {r["key"]: r for r in (ver.get("comparison") or [])}
    if not rows:
        return _empty_chart(
            "优化前后关键指标对比",
            "verify.json 里没有 comparison（KPI 表为空）",
            "检查 task_*.yaml 的 kpis 段，并确认第 6 步 verify 已成功执行")
    W = 720
    row_h = 52
    H = 60 + len(keys) * row_h + 10
    L, pw = 140, 300
    s = _svg_open(W, H)
    s += _title(W, "优化前后关键指标对比", "左条=优化前，右条=优化后；数值直接标注")
    y = 56
    for k in keys:
        r = rows.get(k)
        if not r or r["before"] is None or r["after"] is None:
            continue
        mx = max(abs(r["before"]), abs(r["after"])) * 1.25 or 1
        wb = pw * abs(r["before"]) / mx
        wa = pw * abs(r["after"]) / mx
        s.append(f'<text x="{L-10}" y="{y+22}" font-family="{MONO}" font-size="11.5" '
                 f'fill="{C["ink"]}" text-anchor="end">{k}</text>')
        s.append(f'<rect x="{L}" y="{y}" width="{wb:.1f}" height="15" '
                 f'fill="{C["baseline"]}" rx="2"/>')
        s.append(f'<text x="{L+wb+7:.1f}" y="{y+12}" font-family="{MONO}" '
                 f'font-size="10.5" fill="{C["ink_dim"]}">{r["before"]:g}</text>')
        s.append(f'<rect x="{L}" y="{y+20}" width="{wa:.1f}" height="15" '
                 f'fill="{C["optimized"]}" rx="2"/>')
        s.append(f'<text x="{L+wa+7:.1f}" y="{y+32}" font-family="{MONO}" '
                 f'font-size="10.5" font-weight="600" fill="{C["optimized"]}">'
                 f'{r["after"]:g}</text>')
        if r["delta_pct"] is not None:
            col = C["decode"] if r["improved"] else C["warn"]
            s.append(f'<text x="{L+pw+64}" y="{y+22}" font-family="{MONO}" '
                     f'font-size="12" font-weight="600" fill="{col}" '
                     f'text-anchor="end">{r["delta_pct"]:+.1f}%</text>')
        y += row_h
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 图 7：roofline 定位
# ---------------------------------------------------------------------------

def chart_roofline(defn: dict) -> str:
    roof = defn.get("roofline")
    if not roof:
        return _empty_chart(
            "Roofline 定位",
            "definition.json 里没有 roofline（缺峰值算力或带宽）",
            "在 device_*.yaml 补 fp16_tflops_peak（GPU）或 "
            "cpu_cores+cpu_ghz+simd_flops_per_core_cycle（CPU）")
    W, H = 720, 250
    L, R, T, B = 66, 24, 56, 46
    pw, ph = W - L - R, H - T - B
    ridge = roof.get("ridge_point")
    if not ridge:
        # 缺峰值算力时无法画 roofline 拐点，改为直接说明
        return _svg_open(720, 120)[0] + "".join(
            _title(720, "Roofline 定位：不可用",
                   "设备画像缺少峰值算力字段，无法计算拐点 —— "
                   "请在 device_*.yaml 的 compute 段补充 fp16_tflops_peak")) + "</svg>"
    xmax = ridge * 3.2
    ymax = ridge * 1.15   # 以拐点高度为 1.0 归一（fp16 峰值）

    def px(v): return L + pw * min(v, xmax) / xmax
    def py(v): return T + ph - ph * min(v, ymax) / ymax

    s = _svg_open(W, H)
    s += _title(W, "Roofline 定位：本模型 Decode 落在访存受限区",
                f"算术强度 {roof['intensity']} FLOPs/Byte 远低于拐点 "
                f"{ridge} —— 算力无从发挥")
    # 拐点线
    s.append(f'<line x1="{px(ridge):.1f}" y1="{T}" x2="{px(ridge):.1f}" '
             f'y2="{T+ph}" stroke="{C["grid"]}" stroke-dasharray="4 3"/>')
    s.append(_axis_label(px(ridge), T - 6, f"拐点 {ridge:.1f}", "middle"))
    # roofline 折线
    s.append(f'<polyline points="{px(0):.1f},{py(0):.1f} '
             f'{px(ridge):.1f},{py(ridge*1.0):.1f} {px(xmax):.1f},{py(ymax):.1f}" '
             f'fill="none" stroke="{C["upper"]}" stroke-width="2.5"/>')
    # 当前点
    cx, cy = px(roof["intensity"]), py(roof["intensity"] * 1.0 / ridge)
    s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{C["optimized"]}"/>')
    s.append(f'<text x="{cx+12:.1f}" y="{cy+4:.1f}" font-family="{FONT}" '
             f'font-size="11.5" fill="{C["optimized"]}" font-weight="600">'
             f'本模型 Decode（强度 {roof["intensity"]}）</text>')
    s.append(_axis_label(L + pw/2, H - 8, "算术强度 (FLOPs / Byte)"))
    s.append(f'<text x="16" y="{T+ph/2}" font-family="{FONT}" font-size="10.5" '
             f'fill="{C["ink_dim"]}" transform="rotate(-90 16 {T+ph/2})" '
             f'text-anchor="middle">达成算力（归一化）</text>')
    s.append(f'<text x="{L+8}" y="{T+18}" font-family="{FONT}" font-size="11" '
             f'fill="{C["ink_dim"]}">← 访存受限区（Memory-bound）</text>')
    s.append(f'<text x="{px(xmax)-8:.1f}" y="{T+18}" font-family="{FONT}" font-size="11" '
             f'fill="{C["ink_dim"]}" text-anchor="end">算力受限区 →</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
# 组装 HTML
# ---------------------------------------------------------------------------

def build_html(cfg: PerfConfig, defn: dict, base: dict, prof: dict, diag: dict,
               opt: dict, ver: dict) -> str:
    m, d = cfg.model, cfg.device
    hw = prof["hardware"]
    cmp_rows = {r["key"]: r for r in ver["comparison"]}
    # ctx_total 不是每个任务的 KPI；缺失时从 bundle 的启动参数取，
    # 再退到配置里声明的部署上下文 —— 不要写死一个数字。
    ctx_v = (cmp_rows.get("ctx_total", {}).get("after")
             or flag_value(ver["after_bundle"].get("flags") or [], "--ctx-size", int)
             or cfg.arch.get("max_ctx_deployed") or 8192)
    # 内存指标名随后端变化：GPU 是 peak_vram_mib，CPU/端侧是 peak_ram_mib
    _mem_key = ("peak_vram_mib" if "peak_vram_mib" in cmp_rows else "peak_ram_mib")
    _mem_row = cmp_rows.get(_mem_key, {})
    peak_v = (_mem_row.get("after") or 0) / 1024
    margin_v = cmp_rows.get("safety_margin_gib", {}).get("after")
    if margin_v is None:
        _tot = _mem_total_gib(d)
        if _tot and _mem_row.get("after") is not None:
            margin_v = round(_tot - _mem_row["after"] / 1024, 2)
    dec_v = cmp_rows.get("decode_tps", {}).get("after")
    sys_v = cmp_rows.get("decode_tps_system", {}).get("after")
    dec_b = cmp_rows.get("decode_tps", {}).get("before")
    flags_after = ver["after_bundle"]["flags"] or []
    kv_k = flag_value(flags_after, "--cache-type-k")
    kv_v = flag_value(flags_after, "--cache-type-v")
    par_v = flag_value(flags_after, "--parallel", int) or 1
    cards = [
        ("Decode 吞吐（人均）", f'{dec_v:.1f}' if dec_v else "—", "tokens/s",
         (f'基线 {dec_b:.1f} → 系统总吞吐 {sys_v:.1f} t/s'
          if sys_v and dec_b else ""), C["optimized"]),
        ("带宽利用率", f'{hw["bw_utilization"]*100:.1f}', "%",
         f'{hw["achieved_bw_gbps"]} / {hw["peak_bw_gbps"]} GB/s', C["kv"]),
        (_peak_label(d), f'{peak_v:.2f}', "GiB",
         (f'冗余 {margin_v:.2f} GiB / '
          f'{(_mem_total_gib(d) or 0):.1f} GiB'
          if margin_v is not None else ""), C["weight"]),
        ("总上下文", f'{ctx_v//1024}', "K tokens",
         f'{par_v} 槽 × {ctx_v//par_v//1024}K', C["prefill"]),
        ("KV 量化", f'{kv_k or "f16"}/{kv_v or "f16"}', "",
         f'KV 合计 {defn["memory_budget"]["kv_total_gib"]} GiB', C["mmproj"]),
        ("主瓶颈", diag["primary_bottleneck"]["type"].replace("_", " "), "",
         f'置信度 {diag["primary_bottleneck"]["confidence"]}', C["warn"]),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v" style="color:{col}">{_esc(v)}'
        f'<span class="u">{_esc(u)}</span></div>'
        f'<div class="s">{_esc(sub)}</div></div>'
        for k, v, u, sub, col in cards)

    findings = "".join(
        f'<li><b>{_esc(f["type"])}</b>'
        f'{(" / " + _esc(f["subtype"])) if f.get("subtype") else ""}'
        f' <span class="pill">{_esc(f["severity"])}</span>'
        f' <span class="conf">{_esc(f["confidence"])}</span>'
        f'<br><span class="ev">{_esc(f["evidence"][0] if f.get("evidence") else "")}</span></li>'
        for f in diag["findings"])

    # 每张图独立兜底：一张图的数据缺失不该让整份 charts.html 生成失败。
    # 但也不能静默吞掉 —— 占位图里写明异常，stderr 也打一条，否则
    # 「图没画出来」会被当成「这个维度不重要」。
    def _safe(title: str, fn, *fn_args) -> str:
        try:
            return fn(*fn_args)
        except Exception as e:      # noqa: BLE001 —— 这里就是要兜住一切
            print(f"[visualize] ⚠ 图表「{title}」生成失败：{type(e).__name__}: {e}",
                  file=sys.stderr)
            return _empty_chart(
                title, f"生成时抛出 {type(e).__name__}: {e}",
                "这是框架缺陷而非数据问题，请把该异常连同本次 output/<run>/ 的 "
                "JSON 一起反馈")

    charts = [
        ("内存预算", _safe("内存预算", chart_memory, defn, base, cfg)),
        ("KV Cache 增长", _safe("KV Cache 增长", chart_kv, defn, base, prof)),
        ("算子耗时归因", _safe("算子耗时归因", chart_ops, prof)),
        ("吞吐对比", _safe("吞吐对比", chart_throughput, base, opt, ver)),
        ("长上下文衰减", _safe("长上下文衰减", chart_longctx, base)),
        ("前后对比", _safe("前后对比", chart_compare, ver)),
        ("Roofline 定位", _safe("Roofline 定位", chart_roofline, defn)),
    ]
    chart_html = "".join(f'<section class="chart">{svg}</section>' for _, svg in charts)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(m['name'])} 性能分析 · {_esc(cfg.run_id)}</title>
<style>
  :root {{
    --ink: {C['ink']}; --dim: {C['ink_dim']}; --line: {C['grid']};
    --bg: #FFFFFF; --card: #F8F9FA;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink: #E9ECEF; --dim: #ADB5BD; --line: #343A40;
             --bg: #14161A; --card: #1C1F24; }}
    svg text {{ fill: var(--ink) !important; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 32px 20px 64px; background: var(--bg);
         color: var(--ink); font-family: {FONT}; line-height: 1.6; }}
  .wrap {{ max-width: 780px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; letter-spacing: -0.2px; }}
  h2 {{ font-size: 15px; margin: 40px 0 14px; padding-bottom: 8px;
        border-bottom: 1px solid var(--line); font-weight: 600; }}
  .meta {{ color: var(--dim); font-size: 12.5px; margin-bottom: 26px; }}
  .meta code {{ background: var(--card); padding: 2px 6px; border-radius: 4px;
                font-family: {MONO}; font-size: 11.5px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
  @media (max-width: 640px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  .card {{ background: var(--card); border: 1px solid var(--line);
           border-radius: 10px; padding: 12px 14px; }}
  .card .k {{ font-size: 11.5px; color: var(--dim); }}
  .card .v {{ font-size: 24px; font-weight: 650; font-family: {MONO};
              letter-spacing: -0.5px; margin: 2px 0; }}
  .card .v .u {{ font-size: 11px; font-weight: 400; color: var(--dim);
                 margin-left: 4px; font-family: {FONT}; }}
  .card .s {{ font-size: 10.5px; color: var(--dim); }}
  .chart {{ margin: 22px 0; padding: 4px 0; }}
  ul.findings {{ list-style: none; padding: 0; margin: 0; }}
  ul.findings li {{ padding: 10px 0; border-bottom: 1px solid var(--line);
                    font-size: 13px; }}
  .pill {{ font-size: 10px; padding: 1px 7px; border-radius: 20px;
           background: var(--card); border: 1px solid var(--line);
           color: var(--dim); margin-left: 6px; }}
  .conf {{ font-size: 10px; color: var(--dim); }}
  .ev {{ color: var(--dim); font-size: 11.5px; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line);
            color: var(--dim); font-size: 11.5px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(m['name'])} 推理性能分析</h1>
  <div class="meta">
    {_esc(d['name'])} · {_esc(m['quantization'])} · {_esc(d['software']['engine'])}
    <code>{_esc(d['software'].get('engine_commit', 'n/a'))}</code> ·
    {_esc(_runtime_label(d))} · run <code>{_esc(cfg.run_id)}</code>
  </div>

  <div class="grid">{card_html}</div>

  <h2>瓶颈定位结论</h2>
  <ul class="findings">{findings}</ul>

  <h2>性能剖面可视化</h2>
  {chart_html}

  <footer>
    由 perfkit 自动生成 · 数据来源见 output/{_esc(cfg.run_id)}/report.md 附录 B<br>
    图中实心点/实心条为实测值，空心点为实测对照，标注 [D]/[I] 的为推导值。
  </footer>
</div>
</body>
</html>
"""


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    defn = load_json(cfg.out("definition.json"))
    base = load_json(cfg.out("baseline.json"))
    prof = load_json(cfg.out("profiling.json"))
    diag = load_json(cfg.out("diagnose.json"))
    opt = load_json(cfg.out("optimize.json"))
    ver = load_json(cfg.out("verify.json"))

    html = build_html(cfg, defn, base, prof, diag, opt, ver)
    path = cfg.out("charts.html")
    path.write_text(html, encoding="utf-8")
    print(f"[visualize] 已生成 {path}（自包含 HTML，含 7 张内联 SVG 图）")
    print(f"  浏览器打开：open {path}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("visualize", help="剖面可视化：生成自包含 charts.html（7 张 SVG 图）")
    p.set_defaults(func=main)
    return p
