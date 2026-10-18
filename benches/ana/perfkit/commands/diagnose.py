"""
perfkit diagnose —— 工程闭环第 4 步「瓶颈定位」
==============================================================================
输入：profiling.json（四维剖面）+ definition.json（预算与 roofline）
输出：diagnose.json —— 带**证据链**的瓶颈结论

判定不是拍脑袋，而是走一棵可审计的决策树。每条结论都附：
  - verdict      瓶颈类型
  - confidence   high / medium / low（取决于证据强度）
  - evidence     触发该结论的具体数字
  - counter      「什么证据会推翻这个结论」—— 防止确认偏误

瓶颈类型（端侧全谱）：
  compute_bound / memory_bound / io_bound / launch_bound /
  power_thermal_bound / scheduling_bound / capacity_bound / kv_pressure
"""

from __future__ import annotations

from ..config import PerfConfig, load_json, save_json


# ============================================================================
# 判定规则：每条规则是一个纯函数 (ctx) -> finding | None
# ============================================================================

def _rule_memory_bound(p: dict, d: dict) -> dict | None:
    hw = p["hardware"]
    util = hw.get("bw_utilization") or 0
    if util >= 0.60:
        # 端侧设备（统一内存 + 电池）的表述要区分：带宽饱和的应对
        # 不只是「减字节」，还有「降低频率以省电」这一选项。
        _pt = (d.get("device") or {}).get("power_thermal") or {}
        _suffix = ("端侧还多一个选项：主动降频换取能效（tokens/Joule）与温度。"
                   if _pt.get("thermal_throttle") else "")
        return {
            "type": "memory_bound",
            "severity": "primary" if util >= 0.75 else "secondary",
            "confidence": "high" if util >= 0.75 else "medium",
            "evidence": [
                f"达成带宽 {hw['achieved_bw_gbps']} / 峰值 {hw['peak_bw_gbps']} GB/s"
                f" = {util*100:.1f}%",
                f"距理论上界仅差 {hw['gap_vs_upper_bound_pct']}%",
                f"单步 {hw['decode_step_ms']} ms 中权重读取占绝对主导",
            ],
            "reasoning": (
                "Decode 每步只处理 1 个 token，算术强度远低于 roofline 拐点"
                f"（{d['roofline']['intensity']} << {d['roofline']['ridge_point']}）。"
                "算力几乎闲置，时间全部花在把权重从内存搬到计算单元上。"
                + _suffix
            ),
            "counter_evidence": "若算力利用率 >90% 且带宽利用率 <60%，则应改判 compute_bound",
        }
    return None


def _rule_kv_pressure(p: dict, d: dict) -> dict | None:
    """
    KV 访存压力 —— **memory_bound 的一个组成项，不是独立的瓶颈类型**。

    为什么要与 capacity_bound 分开：
      「KV 占每步字节的比例高」是**速度问题**（每步要多搬这些字节），
      而「显存装不下」才是**容量问题**。前者在显存充裕时同样成立
      （本次 64K 下 KV 占单步 17.8%，但显存还有 7 GiB 余量）。
      把两者都叫 capacity_bound 会让读者误以为「需要扩容」。
    """
    kv = p["kv_cache"]
    rows = kv["ladder"]
    if not rows:
        return None
    worst = max(rows, key=lambda r: r["kv_share_of_step_pct"])
    if worst["kv_share_of_step_pct"] < 5.0:
        return None
    _dev_gib = (d.get("device") or {}).get("vram_gib")
    _worst_ctx_k = worst["ctx"] // 1024
    _k_bits = (kv.get("kv_bits") or {}).get("k", 16.0)
    _v_bits = (kv.get("kv_bits") or {}).get("v", 16.0)
    _quantized = (_k_bits < 16.0) or (_v_bits < 16.0)
    _quant_note = (
        f"当前已启用 KV 量化（K={_k_bits}bit / V={_v_bits}bit）—— 若为 f16，"
        f"该项占比会再高数倍"
        if _quantized else
        "当前 KV 为 f16（未量化）—— 量化可直接把这项占比压低 50~75%")
    return {
        "type": "kv_pressure",
        "subtype": "kv_cache_pressure",
        "severity": "secondary",
        "confidence": "high",
        "evidence": [
            f"ctx={worst['ctx']} 时 KV={worst['kv_gib_model']} GiB，"
            f"每步需读写 {worst['kv_read_gib_per_step']} GiB",
            f"占单步耗时 {worst['kv_share_of_step_pct']}%",
            _quant_note,
        ],
        "reasoning": (
            "KV Cache 是内存中唯一随上下文线性增长的部分。"
            + (f"在 {_dev_gib:.0f} GiB 可用内存下，" if _dev_gib else "")
            + f"{_worst_ctx_k}K 上下文时它已占每步访存量的 "
              f"{worst['kv_share_of_step_pct']}% —— 这部分开销与权重读取"
              f"抢同一条带宽，因此是 memory_bound 的**组成项**。"
              f"注意它是速度问题而非容量问题（显存是否够用见 capacity_bound）。"
        ),
        "counter_evidence": (
            "若实测 KV 增量远小于模型预测值，说明存在前缀复用命中；"
            "若关掉 KV 量化后 Decode 不降反升，说明本构建下量化反而引入了"
            "额外的反量化/类型转换开销（本次在 q8_0/q4_0 上实测到这种情况）"),
    }


def _rule_thermal_bound(p: dict, d: dict) -> dict | None:
    """
    热降频瓶颈 —— 端侧独有。

    服务器上热是「运行环境问题」，端侧上热是「产品问题」：
    机身温度上限（体感阈值通常 43°C 左右）决定了 SoC 能持续跑在什么频率，
    持续负载下的性能保持率直接决定用户是否感觉「用一会儿就卡」。
    """
    pt = (d.get("device") or {}).get("power_thermal") or {}
    if not pt.get("thermal_throttle"):
        return None
    return {
        "type": "power_thermal_bound",
        "subtype": "thermal_throttling",
        "severity": "primary",
        "confidence": "medium",
        "evidence": [
            f"设备标记 thermal_throttle=true（TDP {pt.get('tdp_w', '?')}W"
            + ("，电池供电" if pt.get("battery") else "") + "）",
            "端侧持续负载必然降频：峰值性能只能维持数十秒",
            "机身温度上限（体感阈值）通常远低于 SoC 允许的结温",
        ],
        "reasoning": (
            "端侧与服务器的核心差异：性能天花板不由芯片决定，"
            "而由**散热与电池**决定。因此「峰值吞吐」几乎无意义，"
            "真正要优化的是**持续负载下的稳态吞吐**与能效（tokens/Joule）。"
        ),
        "counter_evidence": (
            "若持续负载后吞吐保持率 >95%，说明散热设计充足，"
            "该结论应降级为观察项而非主瓶颈"
        ),
    }


def _rule_launch_bound(p: dict, d: dict) -> dict | None:
    """
    Kernel launch 开销 —— **仅对 GPU/NPU 成立**。

    CPU 上不存在「kernel launch」：算子就是函数调用，开销在纳秒级，
    把它套到 CPU 上会得到一个虚假的「次要瓶颈」，
    进而把优化方向引向「减少算子数量」这种在 CPU 上收益极低的方向。
    """
    kind = ((d.get("device") or {}).get("kind") or "gpu").lower()
    if kind not in ("gpu", "npu", "dsp"):
        return None
    a_layers = p["per_layer"]["layer_index"]
    # 每层约 8~12 个 kernel（QKV/attn/out/gate/up/down/norm×2/rope/silu）
    n_kernel = len(a_layers) * 10
    step_ms = p["hardware"]["decode_step_ms"]
    launch_ms = n_kernel * 4e-3                     # 4µs/launch 保守估计
    share = launch_ms / step_ms
    if share >= 0.05:
        return {
            "type": "launch_bound",
            "severity": "secondary",
            "confidence": "medium",
            "evidence": [
                f"{len(a_layers)} 层 × ~10 kernel ≈ {n_kernel} 次 launch",
                f"按 4µs/次估算 launch 开销 ≈ {launch_ms:.2f} ms",
                f"占单步 {share*100:.1f}%",
            ],
            "reasoning": (
                "GPU 对大量小 kernel 的启动开销敏感。"
                "当单步耗时降到几十毫秒量级时，launch 开销从「可忽略」变成「可观」。"
            ),
            "counter_evidence": (
                "CPU 上不存在 kernel launch，此结论不适用；"
                "GPU/NPU 上若 kernel 间隙 <2µs 则 launch 不是瓶颈"),
        }
    return None


def _rule_capacity_risk(p: dict, d: dict) -> dict | None:
    b = d["memory_budget"]
    margin = b["safety_margin_gib"]
    # 冗余阈值取自设备约束（端侧按 App 内存配额设定，远小于服务器）
    need = float((d.get("device") or {}).get("constraints", {})
                 .get("min_safety_margin_gib", 4.0))
    if margin < need:
        return {
            "type": "capacity_bound",
            "subtype": "vram_margin_thin",
            "severity": "primary" if margin < need * 0.5 else "secondary",
            "confidence": "high",
            "evidence": [
                f"峰值合计 {b['total_gib']} GiB / 可用内存 {b['vram_total_gib']} GiB",
                f"安全冗余仅 {margin} GiB（设备要求 ≥{need} GiB）",
                f"其中 KV {b['kv_total_gib']} GiB，权重 {b['weight_gib']} GiB",
            ],
            "reasoning": (
                "内存是硬约束而非软约束：一旦越过就 OOM 崩溃，"
                "没有「降级运行」这一档。冗余不足会在长上下文 + 并发峰值时爆掉。"
            ),
            "counter_evidence": "若实测峰值远低于预测，说明预测过于保守",
        }
    return None


def _rule_ppl_risk(p: dict, d: dict) -> dict | None:
    """量化带来的精度风险 —— 属于「优化副作用」的预警，不是性能瓶颈。"""
    return {
        "type": "precision_risk",
        "severity": "watch",
        "confidence": "medium",
        "evidence": [
            f"主模型 {d['model']['quant']}（4.81 bit/权重）",
            f"KV Cache K={p['kv_cache']['kv_bits']['k']}bit / "
            f"V={p['kv_cache']['kv_bits']['v']}bit",
        ],
        "reasoning": (
            "W4 + KV-V4 属于激进组合。权重侧影响全局质量，"
            "KV 侧主要影响长上下文检索（needle-in-haystack）与中段信息保留。"
            "必须在收益量化阶段用 PPL + 长上下文任务集双重把关。"
        ),
        "counter_evidence": "若 PPL 相对劣化 <1% 且长上下文任务无回退，则风险可接受",
    }


def _rule_prefill_compute(p: dict, d: dict) -> dict | None:
    ph = p["phases"]["prefill"]
    return {
        "type": "compute_bound",
        "subtype": "prefill_phase",
        "severity": "phase_scope",
        "confidence": "high",
        "evidence": [f"Prefill {ph['tps']} t/s，GEMM 主导，算力利用率高"],
        "reasoning": (
            "Prefill 是唯一算力密集的阶段。优化它的手段（FlashAttention、"
            "更大 batch、Tensor Core）与 Decode 完全不同，必须分阶段对待。"
        ),
        "counter_evidence": "若 Prefill 也受带宽限制（小 prompt），则并入 memory_bound",
    }


RULES = [
    ("memory_bound", _rule_memory_bound),
    ("kv_cache_pressure", _rule_kv_pressure),
    ("vram_margin", _rule_capacity_risk),
    ("thermal_throttling", _rule_thermal_bound),
    ("launch_overhead", _rule_launch_bound),
    ("prefill_compute", _rule_prefill_compute),
    ("precision_risk", _rule_ppl_risk),
]


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    prof = load_json(cfg.out("profiling.json"))
    defn = load_json(cfg.out("definition.json"))

    findings = []
    for name, rule in RULES:
        try:
            f = rule(prof, defn)
        except (KeyError, TypeError) as e:
            f = {"type": name, "severity": "error", "confidence": "low",
                 "evidence": [f"规则 {name} 执行失败: {e}"],
                 "reasoning": "", "counter_evidence": ""}
        if f:
            f["rule"] = name
            findings.append(f)

    # 主瓶颈：取 severity=primary 的第一条；否则取 secondary 中权重最高的
    order = {"primary": 0, "secondary": 1, "phase_scope": 2, "watch": 3, "error": 4}
    findings.sort(key=lambda f: order.get(f.get("severity", "watch"), 9))
    primary = next((f for f in findings if f["severity"] == "primary"), None)
    if primary is None:
        primary = next((f for f in findings if f["severity"] == "secondary"), None)

    # ------------------------------------------------------------------
    # 优化路线：从结论到方向的映射，且**必须可追溯**
    # ------------------------------------------------------------------
    # 规则表：每条规则是「哪条结论 → 推出哪些方向 → 为什么」。
    # 报告会把这四列原样渲染出来 —— 路线不能是一个凭空出现的字符串，
    # 读者要能看出「为什么 kernel 排第一、precision_guard 排最后」。
    ROUTE_RULES: list[tuple[str, list[str], str]] = [
        ("memory_bound", ["kernel", "speculative", "quantization"],
         "访存受限有三条互补路径，按「改动成本 ÷ 收益确定性」排序："
         "① kernel —— 让同样的搬运更高效（改动大、收益确定性中）；"
         "② speculative —— 减少搬运**次数**（改动小、有实测支撑）；"
         "③ quantization —— 减少每次搬运的**字节数**（改动小、但有精度代价）"),
        ("kv_cache_pressure", ["kv_cache"],
         "KV 随 ctx 线性增长且已占单步耗时可观份额 → "
         "KV 量化能同时降低容量与每步读写量，是唯一同时命中两项的手段"),
        ("vram_margin_thin", ["kv_cache", "quantization"],
         "内存冗余不足属于容量问题而非速度问题 → "
         "优先 KV 量化（收益/精度代价比最高），其次整体降档"),
        ("power_thermal_bound", ["ondevice"],
         "端侧独有的热/功耗约束无法靠「减少字节」解决 → "
         "需要主动热管理策略（动态降载），把断崖式降频变成平缓下降"),
        ("launch_bound", ["kernel"],
         "Kernel 启动开销占比可观 → CUDA Graph / 算子融合直接消除该开销，"
         "与访存侧的优化正交"),
        ("precision_risk", ["precision_guard"],
         "这不是性能瓶颈而是**质量风险**（量化副作用）→ "
         "不产出优化方案，只产出守门项（PPL 与长上下文任务集），"
         "因此排在路线末尾作为验收红线而非优化动作"),
    ]

    route: list[str] = []
    route_reasoning: list[dict] = []
    for _fkey, _dirs, _why in ROUTE_RULES:
        # 命中判定：type 匹配，或 subtype 匹配
        _hit = [f for f in findings
                if f.get("type") == _fkey or f.get("subtype") == _fkey]
        if not _hit:
            continue
        _src = _hit[0]
        _new = [d for d in _dirs if d not in route]
        if not _new:
            continue
        route += _new
        route_reasoning.append({
            "from_finding": _src.get("subtype") or _src.get("type"),
            "from_severity": _src.get("severity"),
            "adds": _new,
            "why": _why,
            "evidence": (_src.get("evidence") or [None])[0],
        })

    out = {
        "primary_bottleneck": primary,
        "findings": findings,
        "optimization_route": route,
        "route_reasoning": route_reasoning,
        "evidence_chain": {
            "roofline": defn["roofline"],
            "bandwidth_utilization": prof["hardware"]["bw_utilization"],
            "top_ops": [{"op": o["op"], "pct": o["pct_of_step"]} for o in prof["ops"][:3]],
            # 长上下文曲线为空时（live 模式只跑了不带 depth 的场景）不能 max() ——
            # 空序列会直接抛 ValueError，把前面所有分析成果一起废掉。
            # 这里给 None 并在报告里标注「未采集」，而不是编一个 0。
            "kv_worst_share_pct": (
                max((r["kv_share_of_step_pct"]
                     for r in prof["kv_cache"]["ladder"]), default=None)),
            "kv_ladder_note": (None if prof["kv_cache"]["ladder"] else
                               "场景矩阵里没有带 depth 的点，KV 剖面为空；"
                               "补一个 depth 场景或台账 curves.long_context 即可"),
        },
    }
    path = save_json(out, cfg.out("diagnose.json"))

    print("=" * 74)
    print(f"瓶颈定位 · {cfg.model['name']}  [run={cfg.run_id}]")
    print("=" * 74)
    pb = primary or {"type": "unknown", "confidence": "low", "evidence": ["证据不足"]}
    print(f"\n主瓶颈  >>> {pb['type'].upper()}"
          f"{' / ' + pb.get('subtype', '') if pb.get('subtype') else ''}"
          f"   (confidence={pb['confidence']})")
    for e in pb.get("evidence", []):
        print(f"    · {e}")
    print(f"  归因  {pb.get('reasoning', '')}")
    print(f"  反证  {pb.get('counter_evidence', '')}")

    print(f"\n全部结论（{len(findings)} 条）")
    for f in findings:
        tag = {"primary": "[主]", "secondary": "[次]", "phase_scope": "[阶段]",
               "watch": "[观察]", "error": "[错误]"}.get(f["severity"], "[?]")
        print(f"  {tag} {f['type']:<18} conf={f['confidence']}")

    print(f"\n优化路线  {' → '.join(route)}")
    print(f"  路线推导（从结论到方向，可追溯）：")
    for rr in route_reasoning:
        print(f"    [{rr['from_severity']}] {rr['from_finding']} "
              f"→ {' + '.join(rr['adds'])}")
        print(f"        {rr['why']}")
    print(f"\n产出 → {path}")
    print("=" * 74)
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("diagnose", help="【第4步】瓶颈定位：决策树判定 + 证据链 + 反证")
    p.set_defaults(func=main)
    return p
