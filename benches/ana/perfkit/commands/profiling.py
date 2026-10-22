"""
perfkit profiling —— 工程闭环第 3 步「性能剖面采集」
==============================================================================
采集四类剖面，回答「时间与带宽到底花在哪」：

  A. 阶段剖面   Prefill / Decode / E2E 的耗时与吞吐拆解
  B. 逐层剖面   每层耗时、每层权重读取量、达成带宽、效率
  C. 算子剖面   各算子（Q/K/V/O/Gate/Up/Down）的字节占比与耗时占比
  D. 硬件剖面   带宽利用率、SM 利用率、KV 读写开销、PCIe 回退开销

replay 模式（默认）：以实测的单步耗时为总预算，按权重字节做带宽归因。
    —— 这是 Memory-bound 场景下的标准归因法：谁读的字节多，谁就占时间。
live 模式：调用 llama.cpp 计时输出 / nsys / nvidia-smi dmon 采集原始数据。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import (GIB, PerfConfig, attn_layers, compute_side_analysis,
                      embedding_bytes, kv_cache_bytes, layer_weight_bytes,
                      load_json, parse_quant_bits, save_json, ssm_layers,
                      weight_bytes)


# ============================================================================
# A/B/C/D 四类剖面的 replay 归因
# ============================================================================

def _baseline_is_speculative(cfg: PerfConfig, baseline: dict | None = None) -> bool:
    """
    本次 profiling 用的「基线 decode 值」是否来自含投机解码的配置。

    判据优先级：
      1) 台账 `baseline_native.speculative`（显式声明，推荐）
      2) 回落：选定 bundle 是否带 `--spec-type`
    """
    try:
        _led = load_json(cfg.ledger())
    except Exception:
        _led = {}
    decl = (_led.get("baseline_native") or {}).get("speculative")
    if decl is not None:
        return bool(decl)
    b = next((x for x in cfg.plan.get("bundles", [])
              if x.get("id") == cfg.plan.get("selected_bundle")), None)
    return bool(b and flag_present(b.get("startup_flags", []), "--spec-type"))


def _pick_single_stream_scenario(baseline: dict) -> tuple[str, dict] | tuple[None, None]:
    """
    从基线里挑一个「单流、有 decode 实测值」的场景做单步耗时归因。

    选择规则（按优先级）：
      1) parallel == 1 且有 decode_tps
      2) 其中 prompt_tokens 最小的那个（最接近纯 Decode 稳态）
    这样换任务、换场景命名后仍能自动选到合适的基准点，不依赖具体 id。
    """
    cands = [(sid, s) for sid, s in (baseline.get("scenarios") or {}).items()
             if s.get("decode_tps") and s.get("parallel", 1) == 1
             and s.get("status") != "todo"]
    if not cands:
        # 退一步：任何有 decode_tps 的场景
        cands = [(sid, s) for sid, s in (baseline.get("scenarios") or {}).items()
                 if s.get("decode_tps") and s.get("status") != "todo"]
    if not cands:
        return None, None
    cands.sort(key=lambda kv: kv[1].get("prompt_tokens", 0))
    return cands[0]


def _ops_from_breakdown(cfg: PerfConfig, step_ms: float) -> list[dict] | None:
    """
    用模型配置里的**实测权重分解表**（GGUF 张量表逐张量累加）做算子归因。

    为什么需要这条路：几何公式只认得「同构的 7 个稠密算子」。一旦模型是
    混合架构（如 SSM/线性注意力 + 全注意力交替），几何假设会同时错两件事：
      · 层内构成（SSM 层的 in-proj / z-gate / out-proj 与稠密层完全不同）
      · 层间分布（全注意力层与 SSM 层的权重体积差 20%+）
    此时唯一可靠的口径是**按 GGUF 里每个张量的真实字节数**归类累加。

    配置里有 `weight_breakdown.ops` 时走这条路；没有则退回几何公式
    （纯 Transformer 模型上两者等价，几何公式无需改动）。

    口径约定（与 report §3.5 一致）：
      pct_of_step  占整个 Decode 单步搬运量（分母 = 全部权重）
      pct_of_layer 占**同类单层**权重（attn 层 / ssm 层 / 全局）
      mib_per_layer 该算子每层的平均字节（global 类为 0）
    """
    br = cfg.model.get("weight_breakdown") or {}
    ops = br.get("ops") or []
    if not ops:
        return None
    wb = weight_bytes(cfg.model)
    # 同类单层的平均字节（分母）：dense 类覆盖全部层，attn/ssm 只覆盖各自的层，
    # global 类是整表一份。用它算「占同类单层」才有可比性 ——
    # 混合架构里 SSM 层与全注意力层的层内构成完全不同，混在一起算没有意义。
    _kind_bytes: dict[str, int] = {}
    _kind_layers: dict[str, int] = {}
    for o in ops:
        k = o.get("kind", "global")
        n = int(o.get("n_layer") or 0)
        _kind_bytes[k] = _kind_bytes.get(k, 0) + int(float(o.get("gib") or 0) * GIB)
        if n and k != "global":
            _kind_layers[k] = max(_kind_layers.get(k, 0), n)
    rows = []
    for o in ops:
        gib = float(o.get("gib") or 0)
        if gib <= 0:
            continue
        b = int(gib * GIB)
        n_layer = int(o.get("n_layer") or 0)
        kind = o.get("kind", "global")
        step_share = b / wb if wb else 0.0
        ms = step_ms * step_share
        _denom = (_kind_bytes.get(kind, 0) / max(_kind_layers.get(kind, 1), 1)) * n_layer \
            if kind != "global" and n_layer else _kind_bytes.get(kind, b)
        rows.append({
            "op": o["op"],
            "kind": kind,
            "n_layer": n_layer,
            "gib_total": round(gib, 3),
            "params_m": round(float(o["params_m"]), 1) if o.get("params_m") else None,
            "bytes_share": round(step_share, 4),
            # 同类单层内部的构成比（attn 层 / ssm 层 / dense 层分别归一）
            "pct_of_layer": round(b / _denom * 100, 2) if _denom else 0.0,
            "pct_of_step": round(step_share * 100, 2),
            "mib_per_layer": round(gib * 1024 / n_layer, 2) if n_layer else 0.0,
            "mib_per_step_all_layers": round(gib * 1024, 1),
            "ms_per_step_all_layers": round(ms, 3),
            "achieved_bw_gbps": (round(b / (ms / 1000) / 1e9, 1) if ms > 0 else None),
            "note": o.get("note", ""),
        })
    rows.sort(key=lambda r: -r["pct_of_step"])
    return rows


def _baseline_extra(baseline: dict) -> dict:
    """
    取基线的 `extra` 段；**live 模式没有这一段，需要现场推导**。

    为什么不能直接 `baseline["extra"]`：
      replay 的 extra 是台账里人工整理过的（原生基线吞吐、长上下文曲线…）；
      live 模式跑完 llama-bench 只有逐场景结果，没有这些派生字段。
      直接下标会 KeyError —— 而 live 恰恰是「换硬件后第一次跑」的必经路径。

    推导规则（每条都只用 live 真有的数据）：
      · baseline_native_decode_tps ← 最短 prompt 的单流场景（最接近纯 Decode 稳态）
      · long_context_curve        ← 场景里带 depth 的那些点，按 depth 排序
    """
    extra = dict(baseline.get("extra") or {})
    scs = baseline.get("scenarios") or {}

    if extra.get("baseline_native_decode_tps") is None:
        cands = [(sid, s) for sid, s in scs.items()
                 if s.get("decode_tps") and s.get("parallel", 1) == 1
                 and s.get("status") != "todo"]
        if cands:
            cands.sort(key=lambda kv: kv[1].get("prompt_tokens", 0))
            extra["baseline_native_decode_tps"] = cands[0][1]["decode_tps"]
            extra["baseline_native_source"] = (
                f'live 推导：场景 {cands[0][0]}（prompt '
                f'{cands[0][1].get("prompt_tokens")} token，单流）')

    if not extra.get("long_context_curve"):
        rows = []
        for sid, s in scs.items():
            if s.get("status") == "todo" or not s.get("depth"):
                continue
            rows.append({
                "ctx": s["depth"],
                "prefill_tps": s.get("prefill_tps"),
                "decode_tps": s.get("decode_tps"),
                # live 模式下没有独立的 KV 实测，用几何公式补，
                # 并如实标注来源 —— 不能让读者以为这是实测值。
                "kv_gib": None,
                "kv_gib_source": "derived",
                "prompt_tokens": s.get("prompt_tokens"),
                "gen_tokens": s.get("gen_tokens"),
                "command": s.get("command"),
                "command_source": s.get("command_source"),
            })
        if rows:
            rows.sort(key=lambda r: r["ctx"])
            extra["long_context_curve"] = rows
            extra["long_context_source"] = "live 推导：场景矩阵里带 depth 的点"
    return extra


def profile_replay(cfg: PerfConfig, baseline: dict) -> dict:
    a, dev = cfg.arch, cfg.device
    wb = weight_bytes(cfg.model)
    lw = layer_weight_bytes(cfg.model)
    emb = embedding_bytes(cfg.model)
    bw_peak = dev["memory"]["mem_bandwidth_gbps_peak"] * 1e9

    # 单步解码耗时：优先用「原生基线」的实测 decode 速率（最能代表纯 Decode 稳态）；
    # 缺失时退回场景矩阵里挑到的单流场景。
    _extra = _baseline_extra(baseline)
    native = _extra.get("baseline_native_decode_tps")
    ref_id, ref_sc = _pick_single_stream_scenario(baseline)
    if native is None:
        native = (ref_sc or {}).get("decode_tps")
    if native is None:
        raise RuntimeError(
            "基线中找不到任何 decode 实测值，无法做单步耗时归因。"
            "请先运行 `perfkit baseline` 并确认台账含 scenarios 条目。")
    step_ms = 1000.0 / native
    step_s = step_ms / 1000.0

    # ---- D. 硬件剖面：达成带宽与利用率 --------------------------------------
    #
    # ⚠️ **投机解码下「权重字节 ÷ 单步耗时」不是带宽**：一次权重读取产出
    #    mean_len 个 token，算出的值会超过物理带宽（本项目实测 1347 GB/s =
    #    标称的 134%，纯读的 153%）。这不是「kernel 突破极限」，而是分母口径
    #    失效。此时必须显式标注，并改用「每次权重读取产出的 token 数」衡量。
    _spec = _baseline_is_speculative(cfg, baseline)
    achieved_bw = wb / step_s                        # B/s
    bw_util = achieved_bw / bw_peak
    kv_peak = kv_cache_bytes(cfg.model, a["max_ctx_native"], *cfg.kv_bits())

    hw = {
        "achieved_bw_gbps": round(achieved_bw / 1e9, 1),
        "peak_bw_gbps": round(bw_peak / 1e9, 1),
        "bw_utilization": round(bw_util, 4),
        "sm_utilization": baseline.get("memory", {}).get("gpu_util"),
        "decode_step_ms": round(step_ms, 3),
        "theoretical_upper_bound_tps": round(bw_peak / wb, 2),
        "gap_vs_upper_bound_pct": round((1 - native / (bw_peak / wb)) * 100, 2),
        "verdict": ("speculative_metric_invalid" if _spec else
                    ("bandwidth_saturated" if bw_util > 0.75
                     else "bandwidth_underused")),
        "speculative": _spec,
    }
    if _spec:
        hw["metric_note"] = (
            "**本配置的基线含投机解码，达成带宽口径失效**："
            f"权重 {wb/1e9:.1f} GB ÷ 单步 {step_ms:.2f} ms = "
            f"{achieved_bw/1e9:.0f} GB/s（标称的 {bw_util*100:.0f}%），"
            "超过物理带宽是正常的 —— 一次权重读取产出了 mean_len 个 token。"
            "正确的效率指标是「每次权重读取产出的 token 数」（= mean_len），"
            "不是带宽达成率。无投机时的对照见报告的基线章节。")
    # 算力侧：只报带宽无法回答「算力是否也饱和了」。
    # 用 roofline 的语义给出「该强度下可达算力」，才能说明
    # 「算力利用率低」是缺陷还是带宽限死的必然结果。
    _ref_ctx = ((ref_sc or {}).get("depth")
                or (ref_sc or {}).get("prompt_tokens") or 0)
    hw["compute"] = compute_side_analysis(
        cfg.device, cfg.model, _ref_ctx, wb, step_ms)

    # ---- C. 算子剖面：字节占比 → 耗时占比 -----------------------------------
    # 优先用 GGUF 实测分解（混合架构唯一正确口径）；缺失时退回几何公式。
    op_rows = _ops_from_breakdown(cfg, step_ms)
    if op_rows is None:
        h, inter = a["hidden_size"], a["intermediate_size"]
        n_kv, hd = a["n_head_kv"], a["head_dim"]
        ops = {
            "attn_q": h * h,
            "attn_k": h * n_kv * hd,
            "attn_v": h * n_kv * hd,
            "attn_o": h * h,
            "ffn_gate": h * inter,
            "ffn_up": h * inter,
            "ffn_down": h * inter,
        }
        total = sum(ops.values())
        op_rows = []
        for name, n_param in ops.items():
            share = n_param / total              # 占单层权重
            mib = lw * share / 1024 / 1024       # 单层字节
            mib_all = mib * a["n_layer"]         # 全模型该算子字节
            # 关键口径：Decode 单步耗时的分母是**全部权重（含词表嵌入）**。
            # 若按单层占比直接乘 step_ms，会把嵌入层的时间重复摊给层内算子，
            # 使「各算子 + 嵌入层」之和超过 100%。这里统一按全模型字节占比摊分。
            step_share = (lw * share * a["n_layer"]) / wb
            ms_all = step_ms * step_share
            op_rows.append({
                "op": name,
                "kind": "attn" if name.startswith("attn") else "dense",
                "n_layer": a["n_layer"],
                "gib_total": round(mib_all / 1024, 3),
                "params_m": round(n_param / 1e6, 1),
                "bytes_share": round(step_share, 4),           # 占单层权重
                "pct_of_layer": round(share * 100, 2),    # 单层内部构成
                "pct_of_step": round(step_share * 100, 2),  # 占整个 Decode 单步
                "mib_per_layer": round(mib, 2),
                "mib_per_step_all_layers": round(mib_all, 1),
                "ms_per_step_all_layers": round(ms_all, 3),
                "achieved_bw_gbps": (round(mib_all * 1024 * 1024 / (ms_all / 1000) / 1e9, 1)
                                     if ms_all > 0 else None),
                "note": "",
            })
    op_rows.sort(key=lambda r: -r["pct_of_step"])

    # ---- B. 逐层剖面 --------------------------------------------------------
    # 混合架构里各层权重差 20%+（全注意力层有 Q/K/V/O，SSM 层有 in-proj /
    # z-gate / out-proj），用平均值会掩盖这个结构。配置里有逐层字节表时直接使用。
    _per_layer_mib = (cfg.model.get("weight_breakdown") or {}).get("per_layer_mib") or []
    per_layer = {
        "layer_index": list(range(a["n_layer"])),
        "bytes_per_layer_mib": round(lw / 1024 / 1024, 2),
        "ms_per_layer_avg": round(step_ms / a["n_layer"], 4),
        "ms_per_layer_uniform": not _per_layer_mib,
        "note": ("replay 模式下按权重均匀摊分；真实逐层耗时需 live 模式，"
                 "UD 动态量化会让首/末层偏慢（Q6_K/Q8_0 层字节更多）"),
        "embedding_gib": round(emb / GIB, 3),
        "embedding_ms_per_step": round(step_ms * emb / wb, 3),
        "embedding_share_pct": round(emb / wb * 100, 2),
        "n_attn_layer": attn_layers(cfg.model),
        "n_ssm_layer": ssm_layers(cfg.model),
    }
    if _per_layer_mib:
        per_layer["bytes_per_layer_list_mib"] = [round(x, 2) for x in _per_layer_mib]
        per_layer["bytes_per_layer_min_mib"] = round(min(_per_layer_mib), 2)
        per_layer["bytes_per_layer_max_mib"] = round(max(_per_layer_mib), 2)
        # 逐层耗时按各层真实字节占比摊分（比均值更接近真机）
        _pl_total = sum(_per_layer_mib)
        per_layer["ms_per_layer_list"] = [
            round(step_ms * x / _pl_total, 4) for x in _per_layer_mib]

    # ---- KV Cache 剖面 ------------------------------------------------------
    # 模型值（几何公式）与实测增量（台账）并列，差额不在本模块消化，
    # 交由 define 的对账与报告的 §1.4 显式呈现。
    kv_rows = []
    for row in (_extra.get("long_context_curve") or []):
        ctx = row["ctx"]
        k_bits, v_bits = cfg.kv_bits()
        kb = kv_cache_bytes(cfg.model, ctx, k_bits, v_bits)
        kv_rows.append({
            "ctx": ctx,
            "kv_gib_model": round(kb / GIB, 3),
            "kv_gib_measured": row["kv_gib"],
            # 来源标记：台账里的 kv_gib 可能是实测，也可能是几何推算 ——
            # 报告必须按来源如实标注列名，不能一律写「实测」。
            "kv_gib_source": row.get("kv_gib_source", "measured"),
            "kv_read_gib_per_step": round(2 * kb / GIB, 4),
            "kv_ms_per_step": round(2 * kb / bw_peak * 1000, 3),
            "kv_share_of_step_pct": round(2 * kb / (wb + 2 * kb) * 100, 2),
            "decode_tps_measured": row["decode_tps"],
        })
    kv_profile = {
        "ladder": kv_rows,
        "kv_bits": {"k": cfg.kv_bits()[0], "v": cfg.kv_bits()[1]},
        "kv_at_native_ctx_gib": round(kv_peak / GIB, 2),
        "model_vs_measured_note": (
            "模型值按几何公式（反推架构）；实测值取自台账的 KV 增量曲线。"
            "若两者差异显著，须先核验 GGUF 元数据再采信任一侧（见 define 对账）"
        ),
        "verdict": (
            f"KV 随 ctx 线性增长：{kv_rows[0]['ctx']//1024}K → "
            f"{kv_rows[-1]['ctx']//1024}K 期间 KV 容量增长 "
            f"{kv_rows[-1]['kv_gib_model']/max(kv_rows[0]['kv_gib_model'], 1e-9):.1f}×，"
            f"是长上下文内存墙的主因"
            if kv_rows else "无长上下文曲线数据"
        ),
    }

    # ---- A. 阶段剖面 --------------------------------------------------------
    # 阶段数据优先取「单流参考场景」（上面挑到的 ref_sc），它最接近
    # 一次独立请求的真实体验；缺失字段按物理关系推导而非编造。
    #
    # Prefill 的口径要特别小心：**短 prompt 的端到端 Prefill 被固定开销主导**
    # （实测 84 token 用 258 ms，其中约 230 ms 是 HTTP/Tokenize/调度），
    # 直接拿来当「Prefill 吞吐」会把 3200 t/s 的机器说成 325 t/s。
    # 因此这里优先用场景矩阵里 prompt 最长、或长上下文曲线里 prompt 明确的那一点。
    ref = ref_sc or {}
    _pf_row = max((s for s in (baseline.get("scenarios") or {}).values()
                   if s.get("prefill_tps") and s.get("status") != "todo"),
                  key=lambda s: s.get("prompt_tokens") or 0, default=None)
    _pf_curve = [r for r in (baseline.get("extra", {}).get("long_context_curve") or [])
                 if r.get("prefill_tps") and (r.get("prompt_tokens") or 0) >= 512]
    _pt = (_pf_row or {}).get("prefill_tps")
    _pt_src = (f'场景 {( _pf_row or {}).get("desc") or ""}'
               f'（prompt {(_pf_row or {}).get("prompt_tokens")} token）'
               if _pf_row else None)
    if _pf_curve:
        _c0 = max(_pf_curve, key=lambda r: r["prefill_tps"])
        if not _pt or _c0["prefill_tps"] > _pt * 1.5:
            _pt = _c0["prefill_tps"]
            _pt_src = f'深度曲线 @ ctx={_c0["ctx"]}（prompt {_c0.get("prompt_tokens")} token）'
    # 耗时按「同一口径」算：若用了曲线点，用曲线点的 prompt 长度；否则用场景的。
    _pf_ptok = ((_pf_row or {}).get("prompt_tokens") if _pt_src and _pt_src.startswith("场景")
                else (max((r for r in _pf_curve if r["prefill_tps"] == _pt),
                          key=lambda r: r.get("prompt_tokens") or 0, default={})
                      .get("prompt_tokens") if _pf_curve else None))
    _pf_s = (_pf_ptok / _pt) if (_pt and _pf_ptok) else None
    _dc_s = ref.get("decode_seconds")
    if _dc_s is None and ref.get("gen_tokens") and ref.get("decode_tps"):
        _dc_s = ref["gen_tokens"] / ref["decode_tps"]
    phases = {
        "prefill": {
            "tps": _pt,
            "ms": round(_pf_s * 1000, 1) if _pf_s else None,
            "source": _pt_src,
            "compute_bound": True,
            "note": ("Prefill 是 GEMM 主导的计算密集阶段，"
                     "算力利用率可接近峰值；FlashAttention 决定其上限。"
                     "口径取 prompt ≥512 token 的点 —— 更短的 prompt 会被"
                     "HTTP/Tokenize 等固定开销主导，不代表 Prefill 能力"),
        },
        "decode": {
            "tps": ref.get("decode_tps"),
            "ms": round(_dc_s * 1000, 1) if _dc_s else None,
            "ms_per_token": (round(1000 / ref["decode_tps"], 2)
                             if ref.get("decode_tps") else None),
            "compute_bound": False,
            "note": "Decode 每步只算 1 个 token，算力闲置、带宽吃满 —— 本次优化主战场",
        },
        "ttft": {"ms": ref.get("ttft_ms"),
                 "breakdown": "排队 + Tokenize + Prefill + 采样"},
        "reference_scenario": ref_id,
    }

    # ---- 跨设备回退代价（独立显存 / 多卡场景的关键风险量化）-----------------
    # 统一内存设备上 Host↔Device 无显式拷贝，此处的数值只是「等价带宽代价」，
    # 真实约束是算子覆盖与同步开销 —— 报告里会按设备类型分别表述。
    transport = dev["constraints"]["host_device_transport_gbps"]
    unified = bool(dev.get("memory", {}).get("unified_memory"))
    fallback = {
        "transport_gbps": transport,
        "unified_memory": unified,
        "transport_label": "统一内存（无 PCIe，等价带宽代价）" if unified else "PCIe",
        "ms_per_layer_if_on_cpu": round(lw / (transport * 1e9) * 1000, 3),
        "ms_if_one_layer_falls_back": round(lw / (transport * 1e9) * 1000 * 2, 3),
        "pct_of_decode_budget_per_layer": round(
            lw / (transport * 1e9) * 1000 * 2 / step_ms * 100, 2),
        "verdict": (
            "统一内存下 CPU/NPU 分工在原理上可行，真正约束是算子覆盖与同步开销"
            if unified else
            "任何 CPU 回退都是不可接受的（单层往返即吃掉显著预算）"
        ),
    }

    return {
        "mode": "replay",
        "meta": baseline["meta"],
        "hardware": hw,
        "phases": phases,
        "per_layer": per_layer,
        "ops": op_rows,
        "kv_cache": kv_profile,
        "cpu_fallback_cost": fallback,
        "memory_waterfall": _memory_waterfall(cfg, baseline),
    }


def _memory_waterfall(cfg: PerfConfig, baseline: dict) -> dict:
    """
    内存分解瀑布：优先取台账里人工整理的 `breakdown_gib`，
    缺失时（live 模式）用**几何预算**补，并标注来源。

    为什么不能直接下标：live 模式刚跑完 llama-bench 时只有峰值/总量，
    没有分项；缺了它就崩在最后一行，前面的剖面全都白算了。
    """
    bd = (baseline.get("memory") or {}).get("breakdown_gib")
    if bd:
        return bd
    b = cfg.mem_budget()
    return {
        "main_weight": b["weight_gib"],
        "mmproj": b["mmproj_gib"],
        "kv_cache": b["kv_total_gib"],
        "ssm_state": b.get("ssm_state_gib", 0.0),
        "runtime_buffer": b["runtime_gib"],
        "_source": "derived（live 模式无台账分项，按几何预算补）",
    }


# ============================================================================
# live 模式：真机采集
# ============================================================================

def profile_live(cfg: PerfConfig, outdir: Path, baseline: dict,
                 load_seconds: int = 20) -> dict:
    """
    live 模式：在**真实推理负载下**采集硬件时序，其余剖面沿用 replay 归因。

    为什么不能只用「跑一次 dmon 就完事」：
      · 空闲时的利用率/功耗全是 0，对判断瓶颈毫无价值 —— 必须同时有负载；
      · 单步耗时与算子归因需要的是**稳态 Decode 的计时**，那由 llama-bench 给；
        自己造负载容易造出「和真实推理不一样」的负载。
    因此做法是：**用 llama-bench 制造稳态负载，同时采样硬件时序**。

    产出的 schema 与 replay 完全一致（phases/ops/kv_cache/per_layer 都在），
    只是把 `hardware` 里能用真机测的字段换成实测值，并加 `live` 块说明来源。
    这样 diagnose 与 report 不需要知道数据来自哪种模式。
    """
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # ① 结构基座：归因口径与 replay 相同（字节占比 → 时间），
    #    因为 live 模式下我们也拿不到逐 kernel 时间线（那需要 nsys/ncu）。
    base = profile_replay(cfg, baseline)
    base["mode"] = "live"

    # ② 制造稳态负载 + 同时采硬件时序
    backend = cfg.backend
    bench = shutil.which("llama-bench") or os.environ.get("LLAMA_BENCH")
    cmd_metrics = backend.live_metrics_command(load_seconds)
    live: dict = {"sources": [], "load_seconds": load_seconds}

    if cmd_metrics:
        load_proc = None
        if bench:
            # Decode 稳态负载：足够长的 -n，让采样窗口落在生成阶段而不是加载阶段
            load_cmd = [bench, "-m", cfg.model["file"],
                        "-p", "0", "-n", str(max(load_seconds * 4, 64)),
                        "-r", "1", "-o", "json"] + backend.accel_flags() + backend.bench_flags()
            load_proc = subprocess.Popen(load_cmd, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
            # 给模型加载留出时间，否则采样窗口全落在 load 阶段
            time.sleep(min(12, max(4, load_seconds // 3)))
        try:
            p = subprocess.run(cmd_metrics, capture_output=True, text=True,
                               timeout=load_seconds + 180)
            (raw / f"live_metrics_{backend.kind}.log").write_text(
                p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
            parsed = backend.parse_live_metrics(p.stdout)
        except subprocess.TimeoutExpired:
            parsed = {"error": "采集超时"}
        finally:
            if load_proc:
                load_proc.terminate()
                try:
                    load_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    load_proc.kill()
        live["source"] = backend.live_metrics_label()
        live["command"] = " ".join(cmd_metrics)
        live["load_generated_by"] = ("llama-bench -p 0 -n <N>" if bench else
                                     "未找到 llama-bench，采样可能落在空闲状态")
        live["metrics"] = parsed
        live["sources"].append(backend.live_metrics_label())
    else:
        live["source"] = None
        live["note"] = (f"{backend.label} 无标准时序采集手段"
                        f"（{sys.platform}），硬件利用率标注为未采集")

    # ③ 用实测值覆盖能覆盖的字段
    hw = base["hardware"]
    hw["mode"] = "live"
    m = live.get("metrics") or {}
    sm = (m.get("sm_util_ratio") or {}).get("mean")
    if sm is not None:
        hw["sm_utilization"] = sm
        hw["sm_utilization_source"] = backend.live_metrics_label()
    elif backend.kind == "cpu":
        # CPU 没有「SM 利用率」，但负载与频率能解释吞吐漂移
        hw["sm_utilization"] = None
        hw["sm_utilization_source"] = "CPU 无 SM 概念；见 live.metrics 的负载与频率"
    base["live"] = live
    base["sources"] = live.get("sources", [])

    # ④ nsys 只给提示，不自动执行（需要 sudo 且会产生大文件）
    nsys = shutil.which("nsys")
    if nsys:
        base["nsys_hint"] = (
            f"{nsys} profile --trace=cuda,nvtx -o {raw}/nsys_llama "
            f"<启动命令>；随后 nsys stats 导出 kernel 级耗时。"
            f"**本报告的算子归因仍是按字节摊分**，kernel 级时间线需手动跑这一步。")
    return base


def add_parser(sub) -> None:
    p = sub.add_parser("profiling", help="【第3步】性能剖面：阶段/逐层/算子/硬件四维采集")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--baseline", default=None, help="baseline.json 路径，默认取 output/<run>/")
    p.add_argument("--load-seconds", type=int, default=20, metavar="N",
                   help="live 模式下制造负载并采样硬件时序的秒数（默认 20）")
    p.set_defaults(func=main)
    return p


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    bpath = Path(args.baseline) if args.baseline else cfg.out("baseline.json")
    baseline = load_json(bpath)

    if args.mode == "live":
        data = profile_live(cfg, cfg.outdir, baseline,
                            load_seconds=getattr(args, "load_seconds", 20))
        data["baseline_ref"] = str(bpath)
    else:
        data = profile_replay(cfg, baseline)

    path = save_json(data, cfg.out("profiling.json"))

    hw = data.get("hardware", {})
    print("=" * 74)
    print(f"性能剖面 · {cfg.model['name']} @ {cfg.device['name']}  "
          f"[run={cfg.run_id}, mode={data.get('mode')}]")
    print("=" * 74)

    print(f"\n[D] 硬件剖面")
    print(f"  单步解码耗时      {hw.get('decode_step_ms', 'n/a')} ms")
    print(f"  达成带宽          {hw.get('achieved_bw_gbps', 'n/a')} / "
          f"{hw.get('peak_bw_gbps', 'n/a')} GB/s  "
          f"= {hw.get('bw_utilization', 0)*100:.1f}%")
    print(f"  理论上界          {hw.get('theoretical_upper_bound_tps', 'n/a')} t/s"
          f"  差距 {hw.get('gap_vs_upper_bound_pct', 'n/a')}%")
    print(f"  判定              {hw.get('verdict', 'n/a')}")
    _live = data.get("live") or {}
    if _live.get("metrics"):
        _m = _live["metrics"]
        _parts = []
        for _k, _lbl in (("sm_util_ratio", "SM 利用率"), ("mem_util_ratio", "显存带宽利用率"),
                         ("power_w", "功耗"), ("temp_c", "温度"),
                         ("load_avg", "系统负载"), ("freq_ghz", "频率")):
            if _k in _m and isinstance(_m[_k], dict):
                _parts.append(f"{_lbl} 均值 {_m[_k].get('mean')}"
                              f"（峰值 {_m[_k].get('max')}）")
        if _parts:
            print(f"  负载中实测        {'；'.join(_parts)}")
            print(f"                    来源 {_live.get('source')}"
                  f"（{_live.get('samples', _m.get('samples', '?'))} 个样本）")
    elif _live:
        print(f"  负载中实测        未采集 —— {_live.get('note', _live.get('source'))}")

    print(f"\n[A] 阶段剖面")
    for name in ("prefill", "decode"):
        ph = data["phases"][name]
        print(f"  {name:<8} {ph['tps']:>8.2f} t/s   "
              f"{'compute-bound' if ph['compute_bound'] else 'memory-bound'}")

    print(f"\n[C] 算子剖面（按单步耗时占比排序）")
    print(f"  {'算子':<10} {'参数量':>9} {'占单步':>8} {'占单层':>8} {'耗时/步':>9} {'达成带宽':>10}")
    for r in data["ops"]:
        print(f"  {r['op']:<10} {r['params_m']:>7.1f}M {r['pct_of_step']:>7.2f}% "
              f"{r['pct_of_layer']:>7.2f}% {r['ms_per_step_all_layers']:>8.3f}ms "
              f"{r['achieved_bw_gbps']:>8.1f}GB/s")
    emb = data["per_layer"]
    print(f"  {'embedding':<10} {'—':>9} {emb['embedding_share_pct']:>8.2f}% "
          f"{emb['embedding_ms_per_step']:>8.3f}ms")

    print(f"\n[B] 逐层剖面")
    print(f"  层数 {cfg.arch['n_layer']}  单层权重 {emb['bytes_per_layer_mib']} MiB"
          f"  单层耗时 {emb['ms_per_layer_avg']} ms")

    print(f"\n[KV] KV Cache 剖面（K={data['kv_cache']['kv_bits']['k']}bit"
          f" / V={data['kv_cache']['kv_bits']['v']}bit）")
    print(f"  {'ctx':>7} {'KV(GiB)':>9} {'每步读(GiB)':>12} {'每步KV耗时':>11} {'占比':>7}")
    for r in data["kv_cache"]["ladder"]:
        print(f"  {r['ctx']:>7} {r['kv_gib_model']:>9.3f} {r['kv_read_gib_per_step']:>12.4f} "
              f"{r['kv_ms_per_step']:>9.3f}ms {r['kv_share_of_step_pct']:>6.2f}%")

    fb = data.get("cpu_fallback_cost", {})
    if fb:
        print(f"\n[风险] CPU 回退代价：单层往返 {fb['ms_if_one_layer_falls_back']} ms "
              f"= 解码预算的 {fb['pct_of_decode_budget_per_layer']}%")

    print(f"\n产出 → {path}")
    print("=" * 74)
    return 0
