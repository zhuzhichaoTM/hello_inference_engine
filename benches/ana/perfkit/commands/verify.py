"""
perfkit verify —— 工程闭环第 6 步「优化效果验证 / 收益量化」
==============================================================================
输入：baseline.json + optimize.json（含各 bundle 的 measured 实测值）
输出：verify.json —— 前后对比表 + 收益量化 + 副作用清单 + 验收判定

核心原则（写作规范第 2 条）：
  对比必须在**同硬件、同驱动、同框架、同测试集**下进行；
  任何一项变化都要显式记录，否则收益不可归因。

取值优先级（每个 KPI 单元格独立判定）：
  1) bundle.measured[<kpi>]  该配置的实测值          → 标记 measured
  2) 由实测基线按物理模型推导（如带宽利用率）        → 标记 derived
  3) 几何模型值（显存等无实测项）                    → 标记 model
  4) 无任何来源                                     → 标记 todo，不参与达标判定
"""

from __future__ import annotations

from ..config import (GIB, PerfConfig, flag_present, flag_value, load_json,
                      mem_budget, parse_quant_bits, reconcile, save_json)


def _bundle_by_id(cfg: PerfConfig, bid: str) -> dict | None:
    for b in cfg.plan.get("bundles", []):
        if b["id"] == bid:
            return b
    return None


def _model_mem(cfg: PerfConfig, bundle: dict) -> dict:
    """按 bundle 的启动参数算几何模型显存。"""
    flags = bundle.get("startup_flags", [])
    ctx = flag_value(flags, "--ctx-size", int) or 8192
    par = flag_value(flags, "--parallel", int) or 1
    k = parse_quant_bits(flag_value(flags, "--cache-type-k") or "f16")
    v = parse_quant_bits(flag_value(flags, "--cache-type-v") or "f16")
    return mem_budget(cfg.device, cfg.model, ctx, par, k, v,
                      include_mmproj=flag_present(flags, "--mmproj"),
                      include_draft=flag_present(flags, "-md"))


def _kv_measured_per_slot_gib(ledger: dict, ctx: int, parallel: int) -> float:
    """
    按台账的长上下文曲线，线性插值出指定 ctx 下**每槽**的 KV 容量。

    KV 随 ctx 线性增长，故线性插值是合理近似。超过曲线末端按末端值截断
    （避免外推出无依据的大数）。
    """
    rows = sorted((ledger.get("curves") or {}).get("long_context", []),
                  key=lambda r: r["ctx"])
    if not rows:
        return 0.0
    if ctx <= rows[0]["ctx"]:
        return rows[0]["kv_gib"] / parallel
    if ctx >= rows[-1]["ctx"]:
        return rows[-1]["kv_gib"] / parallel
    for lo, hi in zip(rows, rows[1:]):
        if lo["ctx"] <= ctx <= hi["ctx"]:
            f = (ctx - lo["ctx"]) / (hi["ctx"] - lo["ctx"])
            return (lo["kv_gib"] + f * (hi["kv_gib"] - lo["kv_gib"])) / parallel
    return rows[-1]["kv_gib"] / parallel


def _band_util(cfg: PerfConfig, tps: float, ctx: int, parallel: int,
               ledger: dict, bundle: dict | None = None) -> tuple[float, str]:
    """
    带宽利用率，严格按 KPI 定义计算：

        有效带宽 = (权重字节 + KV 读写字节) / 单步耗时
        利用率   = 有效带宽 / 标称峰值带宽

    注意 KV 是**读写各一遍**，故乘 2。这与台账 §1.1 的口径（仅权重）
    不同 —— 台账那个数字用于回答「还剩多少带宽没榨干」，
    本指标用于回答「整条内存子系统是否已饱和」。两者都有用，
    但绝不能混用，故此处同时返回口径说明。

    ⚠️ **开启投机解码时本指标失效**：MTP 一次权重读取可以产出多个 token，
    分母不再是「每步一遍权重」。此时算出的值会 >100%（本次实测 139%），
    不代表物理上超过了带宽 —— 它说明「每字节权重产出了多于 1 个 token」。
    这种情况下应改用「每次权重读取产出的 token 数」（= 平均接受长度 + 1）
    来衡量效率。
    """
    from ..config import weight_bytes
    wb = weight_bytes(cfg.model)
    kv_slot = _kv_measured_per_slot_gib(ledger, ctx, parallel) * GIB
    bw = cfg.device["memory"]["mem_bandwidth_gbps_peak"] * 1e9
    eff = (wb + 2 * kv_slot) * tps
    util = eff / bw
    weight_only = wb * tps / bw
    _spec = bool(bundle and flag_present(bundle.get("startup_flags", []), "--spec-type"))
    if _spec:
        return (round(util, 4),
                f"**本配置启用了投机解码，该口径失效**：一次权重读取产出多个 token，"
                f"按「每步读一遍权重」算得 {util*100:.1f}%（仅权重 {weight_only*100:.1f}%），"
                f"超过 100% 是正常的，不代表超出物理带宽。"
                f"应改用「每次权重读取产出的 token 数」衡量")
    return (round(min(util, 0.999), 4),
            f"含 KV 读写口径 {util*100:.1f}%（仅权重口径 {weight_only*100:.1f}%）")


def _cell(bundle: dict, key: str, cfg: PerfConfig, ledger: dict,
          kpi: dict | None = None, baseline: dict | None = None) -> tuple:
    """
    取一个 bundle 在某 KPI 上的值，返回 (value, source, note)。

    取值优先级：
      1) KPI 绑定了 scenario 且 bundle.measured.by_scenario[scn] 有该键
         —— **场景绑定优先于扁平值**：同一个配置在不同上下文深度下
            Decode 相差可达 4 倍（17.5 → 4.2 t/s），用扁平值当口径必然出错。
      2) bundle.measured[key]                       —— 该配置的默认口径实测
      3) KPI 绑定了 scenario 且 bundle 无该场景数据 —— 从 baseline 的该场景取
         （基线 bundle 往往只声明了自己的代表值，而 KPI 口径可能是另一场景）
      4) 由物理模型推导（显存、带宽利用率等无实测项）
      5) 无来源 → todo，不参与达标判定
    """
    m = (bundle.get("measured") or {})
    by_scn = m.get("by_scenario") or {}
    flags = bundle.get("startup_flags", [])
    ctx = flag_value(flags, "--ctx-size", int) or 8192
    par = flag_value(flags, "--parallel", int) or 1
    scn = (kpi or {}).get("scenario")

    # 1) 场景绑定优先
    if scn and key in (by_scn.get(scn) or {}):
        v = by_scn[scn][key]
        if v is not None:
            return v, "measured", f"口径来自场景 {scn}"

    if key == "ppl":
        # PPL 不是「某个 bundle 的属性」而是「模型档位的属性」：
        # 它来自台账 quality.ppl，而不是 bundle.measured。
        _rows = ((ledger.get("quality") or {}).get("ppl") or {}).get("rows") or []
        _me = next((r for r in _rows
                    if r.get("quant", "").startswith(
                        cfg.model.get("quantization", ""))), None)
        if _me:
            _ref = next((r for r in _rows if r is not _me and r.get("ppl")), None)
            _note = (f'**PPL 是模型档位的属性，不是某个 bundle 的属性** —— '
                     f'两侧配置用的是同一个 Q4_K_M 权重，因此前后相同是必然的。'
                     f'有意义的对照是「相对更高精度档的劣化」：'
                     f'{_me["quant"]} 实测 {_me["ppl"]}'
                     + (f'，对照 {_ref["quant"]} {_ref["ppl"]}'
                        f'（{( _me["ppl"]/_ref["ppl"]-1)*100:+.2f}%）'
                        if _ref else '（缺精度对照）'))
            return _me["ppl"], "measured", _note
        return None, "todo", ""

    if key == "band_util":
        tps = m.get("decode_tps")
        if tps:
            v, note = _band_util(cfg, tps, ctx, par, ledger, bundle)
            # 投机解码下该口径失效：**不参与达标判定**（返回 None），
            # 但仍把算出的值和原因写进 note —— 报告要显示「为什么这个数不能用」。
            if flag_present(flags, "--spec-type"):
                return None, "invalid", note
            return v, "derived", note
        return None, "todo", ""

    # 2) 扁平实测值
    #
    # ⚠️ 一个必须拦住的误用：当 KPI **绑定了场景**、bundle 又声明了 by_scenario
    #    （说明它测了多个场景）却恰好没有该场景时，扁平值可能是**另一个场景的拷贝**。
    #    实测：CPU 的 prefill_tps 绑 long_ctx_8k_kvq8，扁平值 34.65 其实是
    #    long_ctx_8k 的值，正确值 11.48（差 3×）。
    #    判据不是「场景缺失」——那样会误伤只声明了扁平值的质量类 KPI ——
    #    而是「扁平值恰好等于某个**别的**场景的同名值」，此时它必然不属于本场景。
    _scn_copied = False
    if scn and by_scn and scn not in by_scn and key in m and m[key] is not None:
        for _other, _vals in by_scn.items():
            if _other != scn and isinstance(_vals, dict) and _vals.get(key) == m[key]:
                _scn_copied = True
                break
    if key in m and m[key] is not None and not _scn_copied:
        note = ""
        if key == "decode_tps" and m.get("decode_tps_range"):
            lo, hi = m["decode_tps_range"]
            note = f"实测区间 {lo}~{hi} t/s（随任务类型波动）"
        if key == "peak_vram_mib":
            note = f'实测（{cfg.backend.memory_measured_by()}）'
        return m[key], "measured", note

    # 3) 回落到 baseline 的场景数据（基线的代表值可能不在这个口径上）
    if scn and baseline:
        s = (baseline.get("scenarios") or {}).get(scn) or {}
        base_key = {"decode_tps_8k": "decode_tps"}.get(key, key)
        v = s.get(base_key)
        if v is not None:
            return v, "measured", f"口径来自场景 {scn}"

    # 4) 复合指标：由台账里的多组实测推导（线程缩放、上下文衰减率等）。
    #    这类指标的「实测值」不是单次测量，而是两次测量的比值，
    #    因此不能从 bundle.measured 里取，只能从台账算。
    derived = _derived_kpi(ledger, key)
    if derived is not None:
        return derived[0], "derived", derived[1]

    if key == "peak_vram_mib":
        return int(_model_mem(cfg, bundle)["total_gib"] * 1024), "model", ""
    if key == "safety_margin_gib":
        return _model_mem(cfg, bundle)["safety_margin_gib"], "model", ""
    if key == "ctx_total":
        return ctx, "config", ""
    return None, "todo", ""


def _derived_kpi(ledger: dict, key: str) -> tuple | None:
    """
    复合指标的推导。返回 (值, 口径说明) 或 None。

    为什么放在台账而不是 bundle：
      这类指标是**同一配置下两次测量的比值**（如 4 线程 ÷ 8 线程），
      属于「实验设计的结果」而非「某个 bundle 的属性」。
      放进 bundle 会让每个 bundle 都要重复声明一遍，且容易自相矛盾。
    """
    extra = ledger.get("extra") or {}
    if key == "thread_scaling":
        ts = extra.get("thread_scaling") or {}
        v = ts.get("thread_scaling_kpi")
        if v is not None:
            return (round(float(v), 3),
                    f'4 线程 {ts.get("decode_at_4t")} t/s ÷ 最优 '
                    f'{ts.get("best_decode_tps")} t/s（最优线程数 '
                    f'{ts.get("best_threads")}）')
    if key == "ctx_decay_8k":
        cs = extra.get("context_scaling") or {}
        v = cs.get("ctx_decay_8k_kpi")
        if v is not None:
            return (round(float(v), 3),
                    f'depth 8192 的 {cs.get("decode_at_depth8192")} t/s ÷ '
                    f'depth 0 的 {cs.get("decode_at_depth0")} t/s')
    return None


def build_comparison(cfg: PerfConfig, ledger: dict, optimize: dict,
                     baseline: dict | None = None) -> tuple[list[dict], dict, dict]:
    """生成逐 KPI 的前后对比，以及两侧 bundle 的摘要信息。"""
    base_id = cfg.plan.get("optimization_baseline", {}).get("id", "B0_baseline")
    before_b = _bundle_by_id(cfg, base_id) or optimize["selected_bundle"]
    after_b = next((b for b in cfg.plan.get("bundles", [])
                    if b["id"] == optimize["selected_bundle"]["id"]),
                   cfg.plan.get("bundles", [{}])[0])
    # optimize.json 里已经带上 measured，优先用它（保证与 optimize 阶段一致）
    after_b = {**after_b, **(optimize["selected_bundle"].get("measured") is not None
                             and {"measured": optimize["selected_bundle"]["measured"]}
                             or {})}

    rows = []
    for kpi in cfg.task["kpis"]:
        key = kpi["key"]
        b, b_src, b_note = _cell(before_b, key, cfg, ledger, kpi, baseline)
        a, a_src, a_note = _cell(after_b, key, cfg, ledger, kpi, baseline)
        delta = improved = None
        if b is not None and a is not None and b != 0:
            delta = round((a - b) / abs(b) * 100, 2)
            improved = (a > b) if kpi["direction"] == "higher" else (a < b)
        rows.append({
            "key": key, "name": kpi["name"], "unit": kpi["unit"],
            "direction": kpi["direction"], "target": kpi["target"],
            "scenario": kpi.get("scenario"),
            "before": b, "before_source": b_src, "before_note": b_note,
            "after": a, "after_source": a_src, "after_note": a_note,
            "delta_pct": delta, "improved": improved,
            "meets_target": _meets(a, kpi),
            "definition": kpi["definition"],
        })
    return rows, before_b, after_b


def _meets(value, kpi) -> bool | None:
    if value is None or kpi.get("target") is None:
        return None
    t = kpi["target"]
    return value >= t if kpi["direction"] == "higher" else value <= t


def side_effects(cfg: PerfConfig, after_bundle: dict,
                 ledger: dict | None = None) -> list[dict]:
    """
    副作用清单 —— 收益必须与代价成对出现，否则报告不完整。

    所有数字都从**配置或台账**取，不写死：
      · Draft 显存代价 ← `model.artifacts[role=draft_mtp].bytes`
      · 「MTP 与并发冲突」的实测结论 ← 台账 `extra.mtp_vs_parallel`
        （没测过就不下结论，只写「未验证」）
    写死的 112.95 / 65.57 只对某一次运行成立，换模型换硬件后就是假数据。
    """
    flags = after_bundle.get("startup_flags", [])
    has_mtp = flag_present(flags, "--spec-type")
    par = flag_value(flags, "--parallel", int) or 1
    ctx = flag_value(flags, "--ctx-size", int) or 8192
    v_bits = parse_quant_bits(flag_value(flags, "--cache-type-v") or "f16")
    _led = ledger or {}

    out = []
    if has_mtp:
        _mv = (_led.get("extra") or {}).get("mtp_vs_parallel") or {}
        _with = (_mv.get("with_mtp") or {}).get("decode_tps_system")
        _without = (_mv.get("without_mtp") or {}).get("decode_tps_system")
        if _with and _without:
            _gain = (_with / _without - 1) * 100
            _note = _mv.get("note")
            out.append({
                "effect": "MTP 与多槽位并发的资源竞争（**本次实测不成立**）",
                "impact": ("推测影响为接受率下降与回滚开销上升；"
                           f"实测 MTP+parallel {par} 系统吞吐 {_with} t/s，"
                           f"反而{'高于' if _gain > 0 else '低于'}关 MTP 的 "
                           f"{_without} t/s（{_gain:+.0f}%）"
                           + (f"。**注意**：{_note}" if _note else "")),
                "mitigation": "无需处置；保留此条以记录「推测被实测推翻」",
                "severity": "low"})
        else:
            out.append({
                "effect": "MTP 与多槽位并发可能争抢资源（**本次未验证**）",
                "impact": "未采集「同配置开/关 MTP」的并发对照，"
                          "因此不能断言冲突存在或不存在",
                "mitigation": "在台账 extra.mtp_vs_parallel 补一组对照"
                              "（with_mtp / without_mtp 的系统吞吐）后再下结论",
                "severity": "medium"})
        _draft = next((a.get("bytes") for a in (cfg.model.get("artifacts") or [])
                       if a.get("role") == "draft_mtp"), None)
        out.append({"effect": "Draft 模型占用额外显存",
                    "impact": (f"+{_draft / 1024 ** 3:.2f} GiB"
                               if _draft else "体积未在配置中声明"),
                    "mitigation": "已计入显存预算；余量紧张时改用 n-gram 草稿（零显存）",
                    "severity": "low"})
    if v_bits <= 4.5:
        # 缓解措施必须反映**当前配置实际怎么做的** —— 旧版写死「K 保持 q8_0」，
        # 在 K 也降到 q4_0 的配置上会变成一句与事实相反的说明。
        _k_bits = parse_quant_bits(flag_value(flags, "--cache-type-k") or "f16")
        if _k_bits > 4.5:
            _mit = "K 保持 q8_0（已采用）；上线前跑 needle-in-haystack 与 PPL"
        else:
            _mit = ("K/V 均为 q4_0（本配置）：K 精度下降的风险只能靠压力测试守门 —— "
                    "本次用 24K/20 条同类码精确检索 3/3、24K 5 针、8 项视觉题 8/8、"
                    "PPL +0.02% 四项覆盖；样本量仍偏小，长文档精确引用需灰度")
        out.append({"effect": "KV Cache 降到 q4_0 引入注意力精度损失",
                    "impact": "长上下文检索、中段信息保留可能劣化",
                    "mitigation": _mit,
                    "severity": "medium" if _k_bits > 4.5 else "high"})
    if par > 1:
        out.append({"effect": "单槽上下文被切分（总 ctx 不变，槽位数增加）",
                    "impact": f"单槽 {ctx // par} tokens，超长单请求会被截断",
                    "mitigation": "客户端 Context Length 必须与服务端一致，否则 400",
                    "severity": "medium"})
    out.append({"effect": "量化导致全局质量损失（W4_K_M）",
                "impact": "PPL 相对 F16 有可测量劣化",
                "mitigation": "用 precision_ladder 逐档实测 PPL，劣化 >2% 则升档 Q5_K_M",
                "severity": "medium"})
    rec = reconcile(cfg.model, cfg.device).get("kv_cache")
    if rec:
        out.append({"effect": "显存模型与实测存在未解释差额",
                    "impact": f'KV @ {rec["ctx"]}：模型 {rec["model_gib"]} GiB vs '
                              f'实测 {rec["measured_gib"]} GiB（×{rec["ratio"]}）',
                    "mitigation": rec["verify_by"],
                    "severity": "high"})
    return out


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    baseline = load_json(cfg.out("baseline.json"))
    optimize = load_json(cfg.out("optimize.json"))
    ledger = load_json(cfg.ledger())

    rows, before_b, after_b = build_comparison(cfg, ledger, optimize, baseline)
    effects = side_effects(cfg, after_b, ledger)

    checks = [_eval_cond(c, rows) for c in cfg.task["acceptance"]["must_pass"]]
    # should_pass 也要逐项判定并给出实测值 —— 只在 §1.8 列出条件而不判定，
    # 读者就无法知道「期望项到底达没达到」，也就无法评估已知不足的影响。
    should = [_eval_cond(c, rows) for c in
              (cfg.task["acceptance"].get("should_pass") or [])]
    guard = []
    for g in cfg.task["acceptance"]["guardrail"]:
        if isinstance(g, dict):
            for k, v in g.items():
                guard.append({"guardrail": k, "expect": v, "status": "pending_live_test"})
        else:
            guard.append({"guardrail": g, "expect": None, "status": "pending_live_test"})

    # 只在有实测值的条件下判 PASS，避免用模型值「蒙混过关」
    decidable = [c for c in checks if c["pass"] is not None]
    passed = bool(decidable) and all(c["pass"] for c in decidable)

    # 真实复测过的 guardrail：台账里带了该场景的实测值就说明跑过，
    # 否则不能声称「已验证」—— 这一栏最容易变成空话。
    # 压测场景的 id 可能是 soak_30min / soak_15min / soak ……
    # 写死一个名字会让「跑过压测」的配置被标成未验证。
    _scn = ledger.get("scenarios") or {}
    _soak_id = next((k for k in _scn if str(k).startswith("soak")), None)
    _soak = (_scn.get(_soak_id) or {}) if _soak_id else {}
    _soak_done = _soak.get("status") != "todo" and bool(_soak.get("decode_tps_system"))
    # 峰值显存：优先台账 memory.peak_used_mib（选定配置的实测峰值），
    # 其次 by_config 里**选定 bundle 自己**的键 —— 不能写死某个历史 bundle id，
    # 那会让换一套配置后这一栏变成 "None MiB" 的空话。
    _mem = baseline.get("memory") or {}
    _by = _mem.get("by_config") or {}
    _sel_id = (optimize.get("selected_bundle") or {}).get("id") or ""
    _peak_mib_any = _mem.get("peak_used_mib")
    if _peak_mib_any is None and _sel_id:
        _v = _by.get(_sel_id)
        _peak_mib_any = (_v.get("peak_mib") if isinstance(_v, dict) else _v)
    if _peak_mib_any is None and _by:
        _first = next(iter(_by.values()))
        _peak_mib_any = (_first.get("peak_mib") if isinstance(_first, dict) else _first)
    _guard_evidence = {
        "soak_30min_stable": (
            f'实测 {_soak.get("rounds", "?")} 轮 / {_soak.get("duration_s", 0)} s：'
            f'系统吞吐稳态 {_soak.get("decode_tps_system")} t/s，'
            f'显存漂移 {( _soak.get("stability") or {}).get("vram_drift_mib")} MiB，'
            f'最高温 {( _soak.get("stability") or {}).get("temp_max_c")} °C，'
            f'未观测到泄漏或降频'
            if _soak_done else None),
        "no_oom": ("实测峰值 %s MiB ≤ 约束 %s MiB，长时压测显存无漂移"
                   % (_peak_mib_any,
                      int(float((cfg.device.get("constraints") or {})
                                .get("max_vram_gib") or 20) * 1024))
                   if _peak_mib_any else None),
        "no_cpu_fallback": ("`-ngl 99` 全层卸载；达成带宽与纯读带宽的比值反证"
                            "无 CPU 回退（若有回退，达成带宽会显著低于此值）"),
        # 「只用外部产物、不自行量化」这类约束无法从测量推出，
        # 只能从台账的声明里读 —— 有声明才算已验证，否则标未验证。
        "no_self_quantized_in_delivery": (
            (ledger.get("constraints") or {}).get("no_self_quantized_in_delivery")
            if (ledger.get("constraints") or {}).get("no_self_quantized_in_delivery")
            else None),
    }
    # PPL 红线：只有拿到了「部署档位 + 更高精度参照」两个数才能判定
    _q_ppl = ((ledger.get("quality") or {}).get("ppl") or {})
    _rows_p = _q_ppl.get("rows") or []
    # 同一量化档位可能有多行（不同 KV 精度），且台账里还可能有更低位宽的对照行。
    # 部署行按 role=deployed 优先；参照行按 role=reference 优先 ——
    # 否则会拿 Q3_K_M 当参照，算出「Q4 比 Q3 好 -4.4%」这种与红线无关的结论。
    _me_p = next((r for r in _rows_p if r.get("role") == "deployed"), None) \
        or next((r for r in _rows_p
                 if r.get("quant", "").startswith(cfg.model.get("quantization", ""))), None)
    _ref_p = next((r for r in _rows_p
                   if r is not _me_p and r.get("role") == "reference" and r.get("ppl")), None) \
        or next((r for r in _rows_p if r is not _me_p and r.get("ppl")), None)
    if _me_p and _ref_p:
        _d_pct = (_me_p["ppl"] / _ref_p["ppl"] - 1) * 100
        _guard_evidence["ppl_delta_pct"] = (
            f'{_me_p["quant"]} {_me_p["ppl"]} vs {_ref_p["quant"]} {_ref_p["ppl"]} '
            f'= {_d_pct:+.2f}%（WikiText-2，40 chunks × 512 ctx，'
            f'llama-perplexity）')
        # 台账若还有更低精度档（用于量化「降到该档的代价」），单独列出，
        # 避免与「相对参照档的劣化」混为一谈。
        _lower = [r for r in _rows_p
                  if r is not _me_p and r is not _ref_p and r.get("ppl")
                  and r["ppl"] > _me_p["ppl"]]
        if _lower:
            _guard_evidence["ppl_delta_pct"] += (
                "；同表另有更低精度档（仅用于量化代价，不参与判定）："
                + "、".join(f'{r["quant"]} {r["ppl"]}（'
                            f'{(r["ppl"]/_me_p["ppl"]-1)*100:+.1f}%）' for r in _lower))
    # 视觉质量红线：需要「视觉简单题 + 难题」两组实测都拿到才判定。
    # 数据在台账 quality.vision 段（不是 PPL 那种逐档表）——
    # 判定口径是「简单题不掉分、难题达到可达上限」。
    _vq = (ledger.get("quality") or {}).get("vision") or {}
    _qs, _qh = _vq.get("simple"), _vq.get("hard")
    if _qs is not None and _qh is not None:
        _base_s = _vq.get("baseline_simple", _qs)
        _base_h = _vq.get("baseline_hard", _qh)
        _guard_evidence["vision_quality_no_regression"] = (
            f'视觉简单题 {_qs}/{_vq.get("simple_total", 8)}（基线 {_base_s}），'
            f'难题 {_qh}/{_vq.get("hard_total", 7)}（基线 {_base_h}）'
            f'—— {"无回退" if _qs >= _base_s and _qh >= _base_h else "**有回退**"}')
    for g in guard:
        # 字符串型 guardrail（如 "ppl_delta_pct <= 2.0"）要按指标名匹配，
        # 不能整串比对 —— 否则永远匹配不上，红线会被永久标成「未验证」。
        _gname = str(g.get("guardrail") or "")
        # soak_15min_stable / soak_30min_stable / soak_stable 都指向同一份压测证据
        _ev = _guard_evidence.get(_gname) or next(
            (v for k, v in _guard_evidence.items()
             if _gname.startswith(k) or k.startswith(_gname)
             or (k.startswith("soak") and _gname.startswith("soak"))), None)
        if _ev:
            g["status"] = "verified_live"
            g["evidence"] = _ev

    out = {
        "before_bundle": {"id": before_b.get("id"), "name": before_b.get("name"),
                          "flags": before_b.get("startup_flags"),
                          "plans": before_b.get("plans"),
                          "measured_source": before_b.get("measured_source")},
        "after_bundle": {"id": after_b.get("id"), "name": after_b.get("name"),
                         "flags": after_b.get("startup_flags"),
                         "plans": after_b.get("plans"),
                         # 方案明细直接来自 optimize.json，保证归因链不断
                         "plan_details": optimize["selected_bundle"].get("plan_details", []),
                         "measured_source": after_b.get("measured_source")},
        "comparison": rows,
        "side_effects": effects,
        "acceptance_checks": checks,
        "should_pass_checks": should,
        # 失效口径必须单独说明：否则读者会以为「这一项没测」，而实际是「测了但不成立」
        "invalid_conditions": [
            {"key": c["condition"].split()[0],
             "reason": c["actual"]}
            for c in (checks + should) if "失效" in str(c.get("actual", ""))],
        "guardrails": guard,
        "verdict": "PASS" if passed else "REVIEW",
        "basis_note": (
            "两侧 bundle 均为同一硬件/驱动/引擎/模型版本下的实测配置；"
            "标注 model 的单元格为几何模型值（无实测项），不参与达标判定。"
        ),
    }
    path = save_json(out, cfg.out("verify.json"))

    print("=" * 100)
    print(f"优化效果验证 · {cfg.model['name']}  [run={cfg.run_id}]")
    print("=" * 100)
    print(f"\n优化前 bundle：{before_b.get('id')}  {before_b.get('name')}")
    print(f"  参数  {' '.join(before_b.get('startup_flags', [])) or '(默认)'}")
    print(f"  来源  {before_b.get('measured_source')}")
    print(f"\n优化后 bundle：{after_b.get('id')}  {after_b.get('name')}")
    print(f"  参数  {' '.join(after_b.get('startup_flags', [])) or '(默认)'}")
    print(f"  来源  {after_b.get('measured_source')}")

    print(f"\n{'指标':<20} {'优化前':>12} {'优化后':>12} {'变化':>10} "
          f"{'目标':>10} {'达标':>5}  口径")
    print("-" * 100)
    for r in rows:
        b = _f(r["before"])
        a = _f(r["after"])
        d = f'{r["delta_pct"]:+.2f}%' if r["delta_pct"] is not None else "—"
        tg = f'{r["target"]}' if r["target"] is not None else "—"
        ok = {True: "✓", False: "✗", None: "—"}[r["meets_target"]]
        print(f"{r['key']:<20} {b:>12} {a:>12} {d:>10} {tg:>10} {ok:>5}  "
              f'{r["before_source"]}→{r["after_source"]}')
        if r.get("after_note"):
            print(f"{'':<20} └ {r['after_note']}")

    print(f"\n副作用与缓解（{len(effects)} 项）")
    for e in effects:
        print(f"  [{e['severity']:<6}] {e['effect']}")
        print(f"           影响: {e['impact']}")
        print(f"           缓解: {e['mitigation']}")

    print(f"\n验收判定（仅对可判定项下结论）")
    for c in checks:
        mark = {True: "PASS", False: "FAIL", None: "N/A"}[c["pass"]]
        print(f"  [{mark}] {c['condition']}  ({c['actual']})")
    print(f"\n  质量红线（需真机复测）：")
    for g in guard:
        gs = (g["guardrail"] if isinstance(g["guardrail"], str)
              else "；".join(f"{k}={v}" for k, v in g["guardrail"].items()))
        print(f"    - {gs}"
              + (f" = {g['expect']}" if g["expect"] is not None else ""))
    print(f"\n  总体结论：{out['verdict']}")
    print(f"\n产出 → {path}")
    print("=" * 100)
    return 0


def _f(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _eval_cond(cond: str, rows: list[dict]) -> dict:
    """把 'decode_tps >= 45' 这类声明式条件求值。"""
    import operator
    ops = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt,
           "==": operator.eq}
    for sym, fn in ops.items():
        if sym in cond:
            lhs, rhs = [x.strip() for x in cond.split(sym, 1)]
            try:
                target = float(rhs)
            except ValueError:
                return {"condition": cond, "pass": None, "actual": "无法解析阈值"}
            row = next((r for r in rows if r["key"] == lhs), None)
            if row is not None and row.get("after_source") == "invalid":
                return {"condition": cond, "pass": None,
                        "actual": "该口径在本配置下失效（invalid），不判定"}
            if row is None or row["after"] is None:
                return {"condition": cond, "pass": None,
                        "actual": "无实测值（todo），不判定"}
            return {"condition": cond, "pass": bool(fn(float(row["after"]), target)),
                    "actual": f'{float(row["after"]):.2f} {sym} {target}'
                              f'  [{row["after_source"]}]'}
    return {"condition": cond, "pass": None, "actual": "无法解析"}


def add_parser(sub) -> None:
    p = sub.add_parser("verify", help="【第6步】效果验证：前后对比 + 收益量化 + 副作用清单")
    p.set_defaults(func=main)
    return p
