"""
perfkit define —— 工程闭环第 1 步「问题定义」
==============================================================================
把「模型画像 + 设备画像 + 任务 KPI」三张表汇总成一份可验收的问题定义书，
并做三件在动手前必须完成的事：

  1. 校验架构自洽性（n_head×head_dim == hidden_size 等）；
  2. 打印显存预算表 —— 先确认「装不装得下」，再谈优化；
  3. 打印 roofline 判定 —— 先确认「瓶颈在哪一侧」，再谈方案。

产出：output/<run_id>/definition.json
"""

from __future__ import annotations

from ..config import (GIB, PerfConfig, arithmetic_intensity, attn_layers,
                      embedding_bytes, layer_weight_bytes, mem_budget,
                      reconcile, roofline_verdict, save_json, ssm_layers,
                      weight_bytes)


def _fmt_row(name: str, gib: float, note: str = "") -> str:
    # 注意：中文字符在等宽终端里占 2 列，直接 ljust 会错位，
    # 这里按「显示宽度」补齐（CJK 计 2，其余计 1）。
    width = sum(2 if _is_wide(c) else 1 for c in name)
    pad = " " * max(0, 22 - width)
    return f"  {name}{pad}{gib:>7.2f} GiB  {note}"


def _is_wide(ch: str) -> bool:
    o = ord(ch)
    return (0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or
            0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF or
            0xFE30 <= o <= 0xFE6F or 0xFF00 <= o <= 0xFF60 or
            0xFFE0 <= o <= 0xFFE6 or 0x20000 <= o <= 0x3FFFD)


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    a, dev = cfg.arch, cfg.device

    # ---- 0. 可选：从 GGUF 元数据核验架构超参 --------------------------------
    # 这是消除「[I] 反推值」不确定性的唯一手段，也是本次分析中
    # KV Cache 模型值/实测值差额的根治办法。
    gguf_check = None
    gguf_path = getattr(args, "from_gguf", None)
    if gguf_path:
        gguf_check = _verify_from_gguf(cfg, gguf_path)
    elif cfg.model.get("gguf_verification"):
        # 配置里固化的核验记录（在目标机上跑过 --from-gguf 后写回）。
        # 这样本地 replay 时报告也能显示「已核验」，而不是每次都重新读 16GB 文件。
        gguf_check = {**cfg.model["gguf_verification"], "recorded": True}
        gguf_check.setdefault("diff", [])

    # ---- 1. 架构自洽性 ------------------------------------------------------
    # 注意力输出维度：标准 Transformer 等于 hidden；Qwen3-Next 类混合架构
    # 由 attn_o 投影回 hidden，需显式声明 attn_out_dim。
    _attn_out = a.get("attn_out_dim") or a["hidden_size"]
    _out_label = "attn_out_dim" if a.get("attn_out_dim") else "hidden_size"
    checks = [
        (f"n_head × head_dim == {_out_label}",
         a["n_head"] * a["head_dim"] == _attn_out,
         f'{a["n_head"]}×{a["head_dim"]}={a["n_head"]*a["head_dim"]} vs {_attn_out}'),
        ("GQA 分组比为整数",
         a["n_head"] % a["n_head_kv"] == 0,
         f'{a["n_head"]}/{a["n_head_kv"]}={a["n_head"]//a["n_head_kv"]}:1'),
        ("部署上下文 ≤ 原生上下文",
         a["max_ctx_deployed"] <= a["max_ctx_native"],
         f'{a["max_ctx_deployed"]} ≤ {a["max_ctx_native"]}'),
    ]
    if a.get("hybrid"):
        # 混合架构的额外校验：全注意力层数必须与实际层数吻合
        checks.append(
            ("混合架构层数拆分自洽",
             attn_layers(cfg.model) + ssm_layers(cfg.model) == a["n_layer"],
             f'全注意力 {attn_layers(cfg.model)} + SSM {ssm_layers(cfg.model)} '
             f'= {a["n_layer"]}'))

    # ---- 2. 显存预算 --------------------------------------------------------
    ctx = a["max_ctx_deployed"]
    budget = cfg.mem_budget()
    k_bits, v_bits = cfg.kv_bits()

    # 同一模型在不同 KV 精度下的对比（说明为什么必须量化 KV）
    # 可行性判据有两条，缺一不可：
    #   ① 装得下（total < 显存总量）
    #   ② 安全冗余 ≥ device.constraints.min_safety_margin_gib
    # 只判 ① 会把「刚好装下、峰值一来就 OOM」的档位标成可行 ——
    # 本次 64K + F16 KV 就是这种情况（余量 3.79 GiB < 硬约束 4.0 GiB）。
    _min_margin = float((dev.get("constraints") or {}).get("min_safety_margin_gib") or 0)
    kv_ladder = []
    for name, kb, vb in [("F16/F16", 16.0, 16.0), ("Q8_0/Q8_0", 8.5, 8.5),
                         ("Q8_0/Q4_0", 8.5, 4.5), ("Q4_0/Q4_0", 4.5, 4.5)]:
        b = mem_budget(dev, cfg.model, ctx, 2, kb, vb, include_mmproj=True)
        _fits = b["fits"] and b["safety_margin_gib"] >= _min_margin
        kv_ladder.append({"kv_type": name, "kv_total_gib": b["kv_total_gib"],
                          "total_gib": b["total_gib"],
                          "margin_gib": b["safety_margin_gib"], "fits": _fits,
                          "margin_ok": b["safety_margin_gib"] >= _min_margin,
                          "min_margin_gib": _min_margin})

    # ---- 3. roofline 判定 ---------------------------------------------------
    wb = weight_bytes(cfg.model)
    ai = arithmetic_intensity(cfg.model, ctx, wb)
    roof = roofline_verdict(dev, ai)

    # ---- 4. 逐算子字节占比（Decode 阶段 = 权重读取占比）----------------------
    # 优先用 GGUF 实测分解（混合架构下几何公式不成立）；缺失时退回几何公式。
    lw = layer_weight_bytes(cfg.model)
    op_bytes: dict = {}
    _br = (cfg.model.get("weight_breakdown") or {}).get("ops") or []
    if _br:
        for o in _br:
            gib = float(o.get("gib") or 0)
            if gib <= 0:
                continue
            n_layer = int(o.get("n_layer") or 0)
            op_bytes[o["op"]] = {
                "share": round(gib * GIB / wb, 4),
                "gib_total": round(gib, 3),
                "mib_per_layer": round(gib * 1024 / n_layer, 2) if n_layer else 0.0,
                "kind": o.get("kind", "global"),
                "n_layer": n_layer,
            }
    else:
        h = a["hidden_size"]
        inter = a["intermediate_size"]
        n_kv = a["n_head_kv"]
        hd = a["head_dim"]
        lw = layer_weight_bytes(cfg.model)
        q = h * h
        k = h * (n_kv * hd)
        v = h * (n_kv * hd)
        o_ = h * h
        attn = q + k + v + o_
        ffn = 3 * h * inter
        per_layer = attn + ffn
        op_share = {
            "attn_q": q / per_layer, "attn_k": k / per_layer, "attn_v": v / per_layer,
            "attn_o": o_ / per_layer, "ffn_gate": inter * h / per_layer,
            "ffn_up": inter * h / per_layer, "ffn_down": h * inter / per_layer,
        }
        op_bytes = {kk: {"share": round(vv, 4),
                         "gib_total": round(lw * a["n_layer"] / GIB * vv, 3),
                         "mib_per_layer": round(lw * vv / 1024 / 1024, 2),
                         "kind": "attn" if kk.startswith("attn") else "dense",
                         "n_layer": a["n_layer"]}
                    for kk, vv in op_share.items()}
    emb = embedding_bytes(cfg.model)
    op_bytes["embedding_io"] = {
        "share": round(emb / wb, 4),
        "gib_total": round(emb / GIB, 3),
        "mib_per_layer": 0.0,
        "kind": "global",
        "n_layer": 0,
        "note": "词表嵌入/输出头（实测字节），不参与逐层摊分",
    }

    # ---- 5. KPI 表 ----------------------------------------------------------
    # KPI 的 scenario 与 target_rationale 必须一起落盘：
    # 前者决定「按哪个场景的口径判定」，后者回答「为什么目标定这个数」。
    # 少了任何一个，§1.7 的 KPI 表都无法被读者检验。
    kpis = [{"key": k["key"], "name": k["name"], "unit": k["unit"],
             "direction": k["direction"], "target": k["target"],
             "scenario": k.get("scenario"),
             "target_rationale": k.get("target_rationale"),
             "definition": k.get("definition", "")}
            for k in cfg.task["kpis"]]

    out = {
        "task": cfg.task["task"],
        "model": {"name": cfg.model["name"], "quant": cfg.model["quantization"],
                  "weight_format": cfg.model["weight_format"],
                  "params_total": cfg.model["params_total"],
                  "weight_gib": round(wb / GIB, 2)},
        "device": {"name": dev["name"], "kind": dev["kind"], "arch": dev["arch"],
                   "vram_gib": round(dev["memory"]["vram_total_mib"] * 1024 * 1024 / GIB, 2),
                   "bw_gbps": dev["memory"]["mem_bandwidth_gbps_peak"],
                   "fp16_tflops": dev["compute"].get("fp16_tflops_peak"),
                   "tdp_w": dev["power_thermal"]["tdp_w"],
                   "unified_memory": bool(dev["memory"].get("unified_memory")),
                   "power_thermal": dev["power_thermal"],
                   "constraints": dev["constraints"]},
        "consistency_checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
        "memory_budget": budget,
        "kv_quant_ladder": kv_ladder,
        "roofline": roof,
        "op_weight_share": op_bytes,
        "kpis": kpis,
        "acceptance": cfg.task["acceptance"],
        "selected_bundle": cfg.selected_bundle()["id"],
        "reconcile": reconcile(cfg.model, cfg.device),
        "source_conflicts": cfg.task.get("source_conflicts", []),
        "gguf_verification": gguf_check,
        "measured_memory": None,
    }
    path = save_json(out, cfg.out("definition.json"))

    # ---- 打印给人看的摘要 ---------------------------------------------------
    print("=" * 74)
    print(f"问题定义 · {cfg.task['task']['title']}  [run={cfg.run_id}]")
    print("=" * 74)
    print(f"\n模型  {cfg.model['name']} / {cfg.model['quantization']}"
          f"  ({cfg.model['params_total']/1e9:.2f}B, {wb/GIB:.2f} GiB)")
    print(f"设备  {dev['name']}  {dev['memory']['vram_total_mib']/1024:.1f}GiB"
          f" @ {dev['memory']['mem_bandwidth_gbps_peak']} GB/s")
    print(f"任务  scope={cfg.task['task']['scope']}  priority={cfg.task['task']['priority']}")

    print("\n[0] GGUF 元数据核验")
    if gguf_check is None:
        print("  未执行（加 --from-gguf <model.gguf> 可消除架构反推的不确定性）")
    elif gguf_check.get("error"):
        print(f"  失败：{gguf_check['error']}")
    else:
        _gib = (gguf_check["file_gib"] if gguf_check.get("file_gib") is not None
                else gguf_check.get("file_bytes", 0) / GIB)
        _src = "配置内固化的核验记录" if gguf_check.get("recorded") else "本次现场核验"
        print(f"  文件  {gguf_check['path']}  （GGUF v{gguf_check['version']}，"
              f"{_gib:.2f} GiB）  [{_src}]")
        print(f"  架构  {gguf_check['arch']}，张量 {gguf_check['tensor_count']} 个，"
              f"层数（按张量名）{gguf_check['layers_by_tensor']}")
        print(f"  量化类型分布  {gguf_check['type_histogram']}")
        _diff = gguf_check.get("diff") or []
        if _diff:
            print(f"  {'字段':<20} {'配置值':>12} {'文件真值':>12}  {'一致':>4}")
            for r in _diff:
                mark = "OK" if r["match"] else "**不符**"
                print(f"  {r['field']:<20} {str(r['config']):>12} "
                      f"{str(r['gguf']):>12}  {mark:>4}")
        elif gguf_check.get("all_match"):
            _ex = gguf_check.get("extracted") or {}
            print(f"  架构字段与配置**逐项一致**（{len(_ex)} 项："
                  f"{', '.join(f'{k}={v}' for k, v in list(_ex.items())[:6])} …）")
        if gguf_check.get("kv_gap_explained"):
            print(f"  → KV Cache 差额解释：{gguf_check['kv_gap_explained']}")

    print("\n[1] 架构自洽性")
    for n, ok, d in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}  ({d})")

    print(f"\n[2] 显存预算（ctx={ctx}, parallel=2, KV={k_bits}/{v_bits} bit）")
    print(_fmt_row("主模型权重", budget["weight_gib"], "[必选]"))
    _emb_src = ("实测字节（token_embd + output_head）"
                if a.get("embedding_bytes_measured") else "按 F16 估算")
    print(f"    ├─ 词表嵌入/输出头      {emb/GIB:>7.2f} GiB  [{_emb_src}]")
    _plm = (cfg.model.get("weight_breakdown") or {}).get("per_layer_mib") or []
    _layer_note = (f"{len(_plm)} 层 × {min(_plm):.1f}~{max(_plm):.1f} MiB，实测逐层"
                   if _plm else f"{a['n_layer']} 层 × {lw/1024/1024:.1f} MiB（均值）")
    print(f"    └─ Transformer 层      {(wb-emb)/GIB:>7.2f} GiB  [{_layer_note}]")
    if budget.get("draft_gib"):
        print(_fmt_row("MTP Draft 权重", budget["draft_gib"],
                       "[-md 挂载时生效；参与显存预算]"))
    if budget.get("ssm_state_gib"):
        print(_fmt_row("SSM 固定状态", budget["ssm_state_gib"],
                       f"[混合架构，不随 ctx 增长；{ssm_layers(cfg.model)} 层]"))
    if budget["mmproj_gib"]:
        print(_fmt_row("视觉投影 mmproj", budget["mmproj_gib"], "[可选]"))
    print(_fmt_row("KV Cache 合计", budget["kv_total_gib"],
                   f'[{2} 槽 × {budget["kv_per_slot_gib"]:.2f} GiB]'))
    print(_fmt_row("运行时缓冲", budget["runtime_gib"], "[FA/激活/CUDA ctx]"))
    print("  " + "-" * 52)
    print(_fmt_row("合计", budget["total_gib"]))
    print(_fmt_row("显存总量", budget["vram_total_gib"]))
    print(_fmt_row("安全冗余", budget["safety_margin_gib"],
                   "OK" if budget["fits"] else "!! OOM 风险"))

    print("\n[3] KV Cache 精度阶梯（同 ctx，说明量化的必要性）")
    for row in kv_ladder:
        if row["fits"]:
            flag = "OK "
        elif not row.get("margin_ok", True):
            flag = "!余量"          # 装得下但冗余不足 —— 峰值一来就 OOM
        else:
            flag = "OOM"
        print(f"  {flag:<4} {row['kv_type']:<12} KV={row['kv_total_gib']:>5.2f}GiB"
              f"  合计={row['total_gib']:>5.2f}GiB  余量={row['margin_gib']:>5.2f}GiB"
              f"  (硬约束 ≥{row.get('min_margin_gib', 0):.1f})")

    print("\n[4] Roofline 判定")
    print(f"  算术强度 {roof['intensity']} FLOPs/Byte  vs  拐点 {roof['ridge_point']}"
          f"  ⇒  {roof['verdict'].upper()}")
    print(f"  可达算力上界 {roof['attainable_tflops']} TFLOPS(fp16)")

    print("\n[5] Decode 单步权重读取占比（= 耗时占比，Memory-bound 下）")
    for kk, vv in sorted(op_bytes.items(), key=lambda x: -x[1]["share"]):
        bar = "█" * int(vv["share"] * 60)
        print(f"  {kk:<10} {vv['share']*100:>5.1f}%  {vv['mib_per_layer']:>6.1f} MiB/层  {bar}")

    print("\n[6] 参考源内部矛盾（并列展示，不选择性引用）")
    for c in out.get("source_conflicts", []):
        print(f"  · {c['topic']}")
        print(f"      A: {c['claim_a']['value']} {c['claim_a']['unit']}"
              f"  ← {c['claim_a']['source']}")
        print(f"      B: {c['claim_b']['value']} {c['claim_b']['unit']}"
              f"  ← {c['claim_b']['source']}")
        _ad = str(c.get("adopted", "")).lower()
        _which = "A" if _ad in ("claim_a", "a") else ("B" if _ad in ("claim_b", "b") else "未声明")
        _tail = (f"消除方式：{c['action']}" if not c.get("resolved")
                 else f"**已消除**：{c['action']}")
        print(f"      比值 {c['ratio']}×，采信 {_which}。{_tail}")
    if not out.get("source_conflicts"):
        print("  无")

    print("\n[7] 模型预测 vs 实测对账（不一致必须显式暴露，不可静默校准）")
    rec = out["reconcile"]
    if rec.get("kv_cache"):
        r = rec["kv_cache"]
        print(f"  KV Cache @ ctx={r['ctx']} ({r['kv_type']}):")
        print(f"    模型几何公式  {r['model_gib']:>6.2f} GiB")
        print(f"    台账实测值    {r['measured_gib']:>6.2f} GiB")
        print(f"    差额          {r['gap_gib']:>6.2f} GiB  (×{r['ratio']})")
        print(f"    可能原因      {r['likely_cause']}")
        print(f"    核验方式      {r['verify_by']}")
        print(f"    影响          {r['impact']}")
    else:
        print("  无待对账项")

    print(f"\n[8] KPI 与验收（共 {len(kpis)} 项）")
    for k in kpis:
        arrow = "↓" if k["direction"] == "lower" else "↑"
        print(f"  {arrow} {k['key']:<18} target={k['target']}  {k['unit']}")
    print(f"\n  选定方案 bundle: {out['selected_bundle']}")
    print(f"\n产出 → {path}")
    print("=" * 74)
    return 0


def _verify_from_gguf(cfg: PerfConfig, gguf_path: str) -> dict:
    """
    读取 GGUF 元数据并与配置中的架构反推值对比。

    如果真实架构与配置不符，本函数会**顺便解释** KV Cache 的模型/实测差额 ——
    因为差额几乎必然来自 n_head_kv 或 head_dim 的反推错误。
    """
    from ..gguf import (GgufError, count_layers_by_tensor_names,
                        diff_against_config, extract_architecture, read_metadata,
                        tensor_type_histogram)
    from ..config import GIB, kv_cache_bytes
    try:
        info = read_metadata(gguf_path)
    except GgufError as e:
        return {"error": str(e), "path": str(gguf_path)}

    ext = extract_architecture(info)
    diff = diff_against_config(ext, cfg.arch)

    # 用真值重算 KV，看能否解释差额
    explained = None
    cal = (cfg.model.get("calibration") or {}).get("kv_cache")
    if cal and {"n_layer", "n_head_kv", "head_dim"} <= set(ext):
        real_model = {"architecture": {**cfg.arch, **ext}}
        real_gib = kv_cache_bytes(real_model, cal["ctx"], cal["k_bits"],
                                  cal["v_bits"]) / GIB
        cfg_gib = cal["geometric_gib"]
        if abs(real_gib - cfg_gib) > 0.05:
            explained = (
                f"按文件真值（n_layer={ext['n_layer']}, n_head_kv={ext['n_head_kv']}, "
                f"head_dim={ext['head_dim']}）重算为 {real_gib:.2f} GiB，"
                f"配置反推值为 {cfg_gib:.2f} GiB，实测为 {cal['gib']:.2f} GiB。"
                + ("→ 真值解释了大部分差额，应更新配置并删除 calibration 锚点。"
                   if abs(real_gib - cal["gib"]) < abs(cfg_gib - cal["gib"]) else
                   "→ 真值仍未解释差额，说明台账的实测增量本身需要复核。")
            )
        else:
            explained = "真值与配置一致，差额应归因于台账实测值，需复核台账。"

    return {
        "path": str(gguf_path),
        "version": info["version"],
        "file_gib": info["file_bytes"] / GIB,
        "arch": ext.get("arch"),
        "tensor_count": info["tensor_count"],
        "layers_by_tensor": count_layers_by_tensor_names(info),
        "type_histogram": tensor_type_histogram(info),
        "extracted": {k: v for k, v in ext.items()},
        "diff": diff,
        "kv_gap_explained": explained,
    }


def add_parser(sub) -> None:
    p = sub.add_parser("define", help="【第1步】问题定义：画像汇总 / 显存预算 / roofline 判定")
    p.add_argument("--from-gguf", default=None, metavar="MODEL.GGUF",
                   help="从 GGUF 文件核验架构超参（消除 [I] 反推的不确定性）")
    p.set_defaults(func=main)
    return p
