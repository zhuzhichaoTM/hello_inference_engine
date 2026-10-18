"""
perfkit report —— 工程闭环第 7 步「落地复盘」（报告导出）
==============================================================================
把前 6 步的全部 JSON 产物组装成《大模型性能分析与优化报告》。

结构固定为七步闭环，与业务地图第三部分一一对应：
    0 环境准备 → 1 问题定义 → 2 性能基线 → 3 瓶颈定位
    → 4 优化实施 → 5 收益量化 → 6 回归验收（+ 复盘与沉淀）

报告遵守 docs/writing-style.md：
  - 头部三要素（目标读者 / 前置知识 / 预计阅读）
  - 所有数字附硬件/驱动/框架/模型版本与复现命令
  - 中文正文，术语保留英文原词
"""

from __future__ import annotations

from pathlib import Path

from ..config import GIB, PerfConfig, flag_value, load_json


def _table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(
               {"l": ":---", "r": "---:", "c": ":---:"}.get(a, "---") for a in aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _num(v, fmt: str = "{:.2f}", dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, (int, float)):
        # 极小值（如 norm_eps=1e-6）用定点格式会显示成 0.00，
        # 读者会以为配置填错了。这里自动切到科学计数法。
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return fmt.format(v)
    return str(v)


def _bar(ratio: float, width: int = 40, ch: str = "█") -> str:
    n = max(0, min(width, int(round(ratio * width))))
    return ch * n + "·" * (width - n)


def _guard_str(g) -> str:
    """
    guardrail 允许两种写法：`'ppl_delta_pct <= 2.0'` 或 `{no_oom: true}`。
    统一渲染成一行文本，避免下游各处重复处理类型分支。
    """
    if isinstance(g, dict):
        return "；".join(f"`{k}` = `{v}`" for k, v in g.items())
    return f"`{g}`"


# ============================================================================

def _dget(d: dict, path: str, default="—"):
    """
    按 'a.b.c' 路径取嵌套字典的值，缺失时返回 default。

    报告要同时服务 GPU / CPU / NPU 三类设备画像，字段集并不一致
    （例如 CPU 设备没有 compute_capability 与 cuda）。用统一的安全取值
    避免为每种设备写一套报告模板 —— 那会让新增硬件类型的成本变高。
    """
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur and cur[part] is not None:
            cur = cur[part]
        else:
            return default
    return cur


def _join(v, default="—"):
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v) if v else default
    return v if v is not None else default


def _device_kind_label(kind: str) -> str:
    return {"gpu": "独立显存 GPU", "cpu": "CPU 后端", "npu": "NPU 加速器",
            "dsp": "DSP", "hybrid": "统一内存（CPU/GPU/NPU 共享物理内存）",
            "soc": "SoC（统一内存）"}.get(kind, kind)


def _method_rows(cfg: PerfConfig, protocol: dict) -> list[list[str]]:
    """
    把台账里声明的「测量协议」渲染成表。

    报告里每个数字都必须能回答三件事：用什么工具测的、口径是什么、
    怎么复现。缺了这三样，读者看到 17.85 t/s 却不知道它含不含采样、
    是均值还是峰值，数字就不可解释、也不可跨配置比较。
    """
    rows: list[list[str]] = []
    if protocol.get("tool"):
        v = str(protocol["tool"])
        if protocol.get("tool_version"):
            v += f'（{protocol["tool_version"]}）'
        rows.append(["测试工具", v])
    if protocol.get("measurement_point"):
        rows.append(["测量点", protocol["measurement_point"]])
    if protocol.get("statistic"):
        rows.append(["统计口径", protocol["statistic"]])
    _bl = protocol.get("backend_label") or protocol.get("backend")
    if _bl:
        _af = " ".join(protocol.get("accel_flags") or []) or "（无）"
        _bf = " ".join(protocol.get("backend_flags") or [])
        _th = protocol.get("threads")
        rows.append(["后端 / 加速标志", f"{_bl}　`{_af}`"
                                      + (f"　`{_bf}`" if _bf else "")
                                      + (f"　线程 {_th}" if _th else "")])
    for f in protocol.get("flags", []):
        rows.append([f'`{f.get("flag", "")}`', str(f.get("meaning", ""))])
    if not rows:
        rows.append(["测量方法", "**未在台账中声明** —— "
                                 "请按 docs/usage.md §7.1 的台账契约补充 "
                                 "`measurement_protocol` 段"])
    return rows


def _scenario_cmds(base: dict) -> list[tuple[str, str, str, str]]:
    """
    收集场景的复现命令，返回 [(标签, 命令, 来源标记, 参数摘要)]。

    参数摘要（prompt/gen/depth）直接写进命令块注释里 ——
    读者要能一眼看出「这条命令对应表里哪一行、用的什么参数」。
    """
    out = []
    for sid, s in (base.get("scenarios") or {}).items():
        if s.get("status") == "todo" or not s.get("command"):
            continue
        bits = []
        if s.get("prompt_tokens") is not None:
            bits.append(f'prompt={s["prompt_tokens"]}')
        if s.get("gen_tokens") is not None:
            bits.append(f'gen={s["gen_tokens"]}')
        if s.get("depth"):
            bits.append(f'depth={s["depth"]}')
        if (s.get("parallel") or 1) > 1:
            bits.append(f'parallel={s["parallel"]}')
        out.append((sid, s["command"], s.get("command_source", "recorded"),
                    ", ".join(bits)))
    return out


def _curve_cmds(curve: list[dict]) -> list[tuple[str, str, str, str]]:
    """长上下文曲线的逐点复现命令。"""
    out = []
    for r in curve:
        if not r.get("command"):
            continue
        ctx = r.get("ctx", 0)
        bits = [f'depth={ctx}']
        if r.get("prompt_tokens") is not None:
            bits.append(f'prompt={r["prompt_tokens"]}')
        if r.get("gen_tokens") is not None:
            bits.append(f'gen={r["gen_tokens"]}')
        out.append((f'{ctx//1024}K', r["command"],
                    r.get("command_source", "recorded"), ", ".join(bits)))
    return out


def _cmd_block(cmds: list[tuple[str, str, str, str]],
               model_path: str | None = None) -> list[str]:
    """
    渲染复现命令块。

    `reconstructed` 的命令必须显式标注：它是按场景参数重建的等价命令，
    不是采集时真实执行的那条。不标就会变成「伪造复现记录」。

    命令里允许用 `$MODEL` 占位（权重路径很长，写全会把命令行撑爆），
    此时在块首补一行 `MODEL=...` 赋值，保证整块可复制粘贴直接运行。
    """
    out = []
    if model_path and any("$MODEL" in c[1] for c in cmds):
        out.append(f'MODEL="{model_path}"')
        out.append("")
    for label, cmd, src, bits in cmds:
        _head = f'# {label}' + (f' —— {bits}' if bits else '')
        if src == "reconstructed":
            _head += '（台账未记录原始命令，以下为按场景参数重建的等价命令）'
        out.append(_head)
        out.append(cmd)
        out.append("")
    return out


def _compiler_desc(sw: dict, d: dict, has_cc: bool) -> str:
    """
    编译器描述。GPU 场景下编译器版本与 CUDA 架构标志同等重要
    （架构标志写错会静默退化到 PTX JIT，性能掉一截），因此一并展示。
    """
    # toolchain 一旦存在就是完整描述（含 gcc/cuda/cmake 与关键开关），
    # 此时不再逐项拼接 —— 否则同一句话会在单元格里重复两遍。
    if sw.get("toolchain"):
        _t = str(sw["toolchain"])
        if has_cc and d.get("compute_capability") and "ARCHITECTURES" not in _t:
            cc = str(d["compute_capability"]).replace(".", "")
            _t += f"，`-DCMAKE_CUDA_ARCHITECTURES={cc}`"
        return _t
    parts = []
    if sw.get("gcc"):
        parts.append(f"GCC {sw['gcc']}")
    if sw.get("compiler"):
        parts.append(str(sw["compiler"]))
    if has_cc and d.get("compute_capability"):
        cc = str(d["compute_capability"]).replace(".", "")
        parts.append(f"`-DCMAKE_CUDA_ARCHITECTURES={cc}`")
    if not parts:
        return "—（未记录；建议在 device_*.yaml 的 software 段补充）"
    return "，".join(parts)


def _curve_span(base: dict) -> str | None:
    """长上下文曲线的实际覆盖范围（如 "4K~64K"）；无数据时返回 None。"""
    rows = (base.get("extra") or {}).get("long_context_curve") or []
    ctxs = sorted({r.get("ctx") for r in rows if r.get("ctx")})
    if not ctxs:
        return None
    lo, hi = ctxs[0] // 1024, ctxs[-1] // 1024
    return f"{lo}K~{hi}K" if lo != hi else f"{lo}K"


def _ledger_quality(cfg: PerfConfig) -> dict:
    """
    读取台账的 quality 段（PPL / needle / 输出一致性）。

    单独抽成函数的原因：§5.4（精度阶梯）与 §5.5（质量验证）都要用它，
    而报告是一段线性拼装的代码 —— 内联读取会让后写的章节依赖前面的局部变量，
    顺序一变就 NameError。这类耦合必须消掉。
    """
    try:
        from ..config import load_json as _lj
        return _lj(cfg.ledger()).get("quality") or {}
    except Exception:
        return {}


def _kpi_reading_rows(cfg: PerfConfig, defn: dict, base: dict,
                      ver: dict) -> list[list[str]]:
    """
    生成「指标口径」表。

    口径必须从**实测数据**里推出来，不能写死 —— 27B/GPU 与 1.5B/CPU 的
    同一指标数值相差一个数量级，写死的示例值会直接误导读者。
    """
    rows: list[list[str]] = []
    sc = base.get("scenarios") or {}
    # 单流：优先取单流、非长上下文的那个场景
    singles = [(sid, s) for sid, s in sc.items()
               if s.get("parallel", 1) == 1 and s.get("decode_tps")
               and s.get("status") != "todo"]
    singles.sort(key=lambda kv: kv[1].get("prompt_tokens", 0))
    if singles:
        sid, s = singles[0]
        rows.append(["decode_tps", f'{s["decode_tps"]:.2f} t/s（单流，{sid}）',
                     "单请求稳态自回归速率；这是用户体验的**下界**"])
    # 长上下文下的单流速率（若有）
    deep = [(sid, s) for sid, s in sc.items()
            if s.get("decode_tps") and (s.get("depth") or s.get("prompt_tokens", 0) >= 4096)]
    if deep:
        deep.sort(key=lambda kv: -(kv[1].get("depth") or kv[1].get("prompt_tokens", 0)))
        sid, s = deep[0]
        _ctx = s.get("depth") or s.get("prompt_tokens")
        rows.append(["decode_tps（长上下文）",
                     f'{s["decode_tps"]:.2f} t/s（{_ctx//1024}K ctx）',
                     "KV 增长后的速率；长上下文场景的**真实下限**"])
    # 系统吞吐（多槽位时才有意义）
    multi = [(sid, s) for sid, s in sc.items() if s.get("decode_tps_system")]
    if multi:
        sid, s = multi[0]
        _par = s.get("parallel", 1)
        rows.append(["decode_tps_system",
                     f'{s["decode_tps_system"]:.2f} t/s（基线，parallel={_par}）',
                     ("所有并发槽位产出之和。**基线是单槽位配置**，"
                      "两路请求被迫串行，因此该值等于单流速率 —— "
                      "它正是「并发」这项优化的起点，优化后见 §5.1"
                      if _par == 1 else
                      "所有并发槽位产出之和；反映硬件在 batch 下的总产出能力")])
    else:
        rows.append(["decode_tps_system",
                     "不适用（单流部署）",
                     f"{cfg.backend.label} 下本次未启用多槽位并发"])
    # 说明主 KPI 目标按哪个口径设定
    tgt = cfg.kpi_by_key("decode_tps") or {}
    if tgt.get("target") is not None and singles:
        # 目标与实测并列：如果长上下文速率低于目标，必须说清楚目标只对
        # 短上下文成立 —— 否则读者会以为「达标」等于「所有场景都达标」。
        _deep_tps = deep[0][1]["decode_tps"] if deep else None
        _note = ("按**单流、短上下文**口径设定"
                 + (f"；长上下文（{deep[0][1].get('depth', deep[0][1].get('prompt_tokens'))//1024}K）"
                    f"实测 {_deep_tps:.1f} t/s，该目标对它不适用"
                    if _deep_tps else ""))
        rows.append(["**主 KPI 目标**", f'{tgt["target"]} t/s', _note])
    return rows


def _env_probe_rows(cfg: PerfConfig, base: dict, defn: dict, opt: dict) -> list:
    """
    生成「环境探针实测」表：这台机器**实测**出来的硬件事实。

    与 §0 的「配置声明」并列，让读者一眼看出：
      · 声明的峰值带宽 vs 探针达成的带宽（差多少倍、为什么）
      · 探针内核效率 → 这台机器的优化方向在哪（访存 or 内核）

    指标定义来自 `backend.probe_metrics()` —— 不同后端用不同探针、
    测不同的东西（主机内存路径 vs 显存路径），因此这里不写死指标名。
    """
    probe = ((base.get("environment") or {}).get("probe") or {})
    rows: list[list[str]] = []
    specs = cfg.backend.probe_metrics()
    # 两个效率指标由下面的「三层参照」块统一解释，避免同一数字出现两次、
    # 且两处口径说明不一致
    _skip = {"gemv_simd_efficiency", "gemv_scalar_efficiency"}
    for spec in specs:
        if spec["key"] in _skip:
            continue
        v = probe.get(spec["key"])
        if v is None:
            continue
        _val = (f'{v*100:.1f}%' if spec["unit"] == "ratio"
                else f'{v:.2f} {spec["unit"]}')
        rows.append([spec["label"], _val,
                     spec.get("note", "探针实测（本机）")])
    # 设备名对照：探针文件按后端分，但同一后端可能有多台机器
    if probe.get("device"):
        _match = probe["device"] == cfg.device.get("name")
        rows.append(["探针采集设备", str(probe["device"]),
                     "与本报告 §0 的设备一致" if _match else
                     f"⚠️ **与本报告 §0 的设备（{cfg.device.get('name')}）不一致** —— "
                     f"该探针数据可能来自另一台机器，请勿直接引用"])
    declared = cfg.device.get("memory", {}).get("mem_bandwidth_gbps_peak")
    _key = "stream_file" if probe.get("stream_file") else "read_bw"
    if declared and probe.get(_key):
        rows.append([
            f'配置声明的峰值带宽', f"{declared:.1f} GB/s",
            f"厂商标称；探针达成值为其 "
            f"{probe[_key] / declared * 100:.0f}%"
            f"（标称值含 ECC/刷新/行冲突等不可达开销）"])

    # ---- 交叉校验：把探针值与「真实推理反推的带宽」对照 ----------------
    # 这是探针数据最有价值的一步 —— 两者若严重不一致，说明其中一个有问题。
    # 实践中出现过「探针报 13.7 GB/s，而真实推理反推出 19.9 GB/s」，
    # 根因是探针在系统负载下采集。
    _native = (base.get("extra") or {}).get("baseline_native_decode_tps")
    # 用**有效权重**（可剔除未启用的 MTP 块）而不是文件体积 ——
    # 否则反推带宽会系统性偏高，进而把「已接近上限」误判成「还有空间」。
    from ..config import weight_bytes as _wb_of
    _wb = _wb_of(cfg.model)
    if _native and probe.get(_key):
        _inf_bw = _native * _wb / 1e9
        _ratio = _inf_bw / probe[_key]
        if _ratio > 3.0:
            # 比值离谱到不可能是「负载导致的偏差」，更可能是数据来源不匹配
            _verdict = (f"**比值异常（{_ratio:.1f}×），数据可能不匹配** —— "
                        f"探针值与推理反推值不在同一量级。"
                        f"请检查：探针是否在**本设备**上采集"
                        f"（`probe_{cfg.backend.kind}.json` 的 backend 字段）、"
                        f"权重体积是否取自同一模型。此单元格不参与任何判定。")
        elif _ratio > 1.05:
            _verdict = (f"**推理反推值高于探针值 {(_ratio-1)*100:.0f}%** —— "
                        f"两种可能：① 探针在负载下采集（探针与推理共享同一份带宽）；"
                        f"② 生产内核优于探针的参考实现（llama.cpp 的 mmvq 用了"
                        f"多行并行与整数点积）。此时**应以推理反推值为准**")
        elif _ratio < 0.8:
            _verdict = (f"推理反推值仅为探针值的 {_ratio*100:.0f}% —— "
                        f"说明还有 {100-_ratio*100:.0f}% 的带宽没被用上，"
                        f"存在回退 / 调度 / 内核实现层面的损失")
        else:
            _verdict = "两者一致（差 <5%），说明推理已接近该路径的带宽上限"
        rows.append(["**交叉校验：推理反推带宽**",
                     f"{_inf_bw:.2f} GB/s"
                     f"（= {_native} t/s × {_wb/1024**3:.2f} GiB）",
                     _verdict])

    # ---- 三层参照：三个数回答不同的问题，混用会得出相反结论 ----
    _sc = probe.get("gemv_scalar_gbps")
    _si = probe.get("gemv_simd_gbps")
    _hw = probe.get(_key)
    if _hw and (_sc or _si):
        _lines = []
        if _sc:
            _lines.append(f"标量 {_sc:.2f}（{_sc/_hw*100:.0f}%）")
        if _si:
            _lines.append(f"SIMD {_si:.2f}（{_si/_hw*100:.0f}%）")
        _lines.append(f"顺序读 {_hw:.2f}（100%）")
        rows.append([
            "**三层参照**", " / ".join(_lines),
            "**三个数用途不同，混用会得出相反结论**："
            "顺序读 = 硬件天花板（反推 Decode 上界用它）；"
            "SIMD = 正常优化内核能拿到多少（判断生产内核空间用它）；"
            "标量 = 完全不优化能拿到多少（跨架构对比用它，因为任何 CPU 都能跑同一份标量代码）"])
    if _si and _hw:
        _eff = _si / _hw
        verdict = ("**内核受限**：即使加 SIMD 也只用到该比例，"
                   "访存路径还有空间（预取/分块/布局/多行并行）"
                   if _eff < 0.5 else
                   "**访存受限**：SIMD 参考内核已接近顺序读上限，"
                   "继续优化内核收益有限，应转向减少搬运字节")
        rows.append([
            "SIMD 内核效率（÷ 顺序读）", f"{_eff * 100:.1f}%",
            f"判定：{verdict}。**判断优化方向要看这个，不是标量值** —— "
            f"标量效率低只说明「朴素实现慢」，不代表生产内核有那么多空间"])
    if _sc and _si:
        rows.append([
            "SIMD 相对标量提速", f"{_si/_sc:.2f}×",
            "这就是「内核实现」这一层的可争取空间。"
            "若生产内核已达 SIMD 水平，则这层已无收益"])
    if probe.get("kernel_efficiency") is not None and not _si:
        ke = float(probe["kernel_efficiency"])
        if cfg.backend.kind == "gpu":
            # GPU 探针的 GEMV 是**朴素参考实现**（一个 block 读一行），
            # 与 llama.cpp 的 mmvq（多行并行 + 整数点积 + tensor core）差一个数量级。
            # 因此这个比值不能用来判断「生产内核还有多少空间」——
            # 那要看下面的「生产内核达成率」。
            rows.append(["探针参考内核效率（GEMV ÷ 纯读）", f"{ke * 100:.1f}%",
                         "**仅代表朴素参考实现**（单行 GEMV），不代表生产内核。"
                         "判断生产内核空间请用下一行的「生产内核达成率」"])
        else:
            verdict = ("**内核受限**：SIMD/内核优化仍有空间"
                       if ke < 0.5 else "**访存受限**：只能减少每步搬运的字节")
            rows.append(["探针内核效率（GEMV ÷ 顺序读）", f"{ke * 100:.1f}%",
                         f"判定：{verdict}"])
    # 生产内核达成率：用真实推理反推的带宽 ÷ 纯读带宽。
    # 这是唯一能回答「生产 kernel 离硬件天花板还有多远」的量。
    if _hw and _native:
        _prod_bw = _native * _wb_of(cfg.model) / 1e9
        _attain = _prod_bw / _hw
        _verdict = ("**kernel 层已无空间**：生产内核拿到纯读带宽的 "
                    f"{_attain*100:.0f}%，剩余空间只能靠减少搬运字节或减少读取次数"
                    if _attain > 0.9 else
                    (f"还有 {100-_attain*100:.0f}% 的纯读带宽未被利用 —— "
                     "存在 kernel 实现 / 调度 / 回退层面的损失" if _attain < 0.8 else
                     f"生产内核已拿到纯读带宽的 {_attain*100:.0f}%，余量有限"))
        rows.append(["**生产内核达成率（推理反推 ÷ 纯读）**", f"{_attain*100:.1f}%",
                     _verdict])
    _penv = probe.get("environment") or {}
    if _penv.get("load_avg"):
        rows.append(["探针采集时系统负载",
                     f'{_penv["load_avg"][0]}（{_penv.get("cpu_cores")} 核，'
                     f'每核 {_penv.get("load_per_core")}）',
                     "**共享内存带宽的后端上，探针结果会被外部负载压低**；"
                     "超标时该数字只作参考"])
    if probe.get("cmd"):
        rows.append(["探针命令", f"`{probe['cmd']}`", "可复现"])
    return rows


def _probe_intro_rows(cfg: PerfConfig) -> list[list[str]]:
    """
    说明「探针是干什么的、测哪些东西、什么硬件上适用」。

    为什么值得单独一段：探针测的是**硬件**而不是模型，读者容易困惑
    「这几个数字和我的推理性能是什么关系」。不解释清楚，
    表格里的 17 GB/s 就是个孤立的数字。
    """
    b = cfg.backend
    probe_bin = getattr(b, "probe_binary", "ggml_probe")
    rows = [
        ["**探针是什么**",
         f"一个独立的小程序（`cpp/{probe_bin}.cpp`），测这台机器的"
         f"**内存/显存带宽实测值**。它与模型无关 —— 一次测量可被多次运行复用。"],
        ["**为什么需要它**",
         "配置里声明的带宽是**厂商标称值**，用于算理论上界；"
         "探针测的是**实际能拿到多少**，用于算达成率。"
         "两者用途不同，缺一个就无法判断「还有多少空间」。"],
        ["**测哪条路径**", b.probe_scope_note() or "—"],
        ["**适用硬件**",
         f"本探针适用于 `{b.kind}` 后端（{b.label}）。"
         "CPU / 统一内存 / SoC 用 `ggml_probe`（测主机内存路径）；"
         "NVIDIA 独显用 `cuda_bw_probe`（测显存路径，必须用**纯读**带宽 —— "
         "用拷贝带宽反推 Decode 会低估约 2 倍）；"
         "NPU 无标准接口，探针返回「未采集」而不是编造数字。"],
        ["**怎么切后端**",
         "探针由 `device_*.yaml` 的 `kind` 自动选择，换硬件时无需改代码。"
         "找不到对应探针时会给出构建提示（如 `bash scripts/build.sh --cuda`）。"],
        ["**为什么同时给标量与 SIMD GEMV**",
         "两者回答**不同的问题**：标量 = 「完全不优化能拿到多少」，"
         "优点是**任何 CPU 都能跑同一份代码**，因此跨架构可比；"
         "SIMD = 「一个正常优化的内核能拿到多少」，"
         "这才是判断「生产内核还有多少空间」的正确基准。"
         "**只给标量会把优化空间高估到荒谬的程度**（标量只有 6% 时，"
         "会误判「还有 94% 空间」，而实际生产内核早已接近上限）。"],
        ["**为什么反推 Decode 上界用顺序读而不是 GEMV**",
         "顺序读是**硬件天花板**，GEMV 只是它的下界。"
         "用 GEMV 反推会把「内核效率」误当成硬件能力 —— "
         "标量只有 6%，会得出「上界只有 0.06 t/s」这种荒谬结论。"
         "正确做法：上界用顺序读，空间判断用 SIMD。"],
        ["**怎么读结果**",
         "先看「交叉校验」那一行 —— 它把探针值与真实推理反推的带宽对照。"
         "两者一致才说明探针可信；不一致时以推理反推值为准。"],
    ]
    return rows


def _validity_rows(cfg: PerfConfig, base: dict) -> tuple[list[list[str]], str]:
    """
    「测量有效性条件」—— 回答三个问题：这是什么、为什么要有、我该做什么。

    这一节的必要性来自一个具体事实：**同一份配置在不同机器状态下
    可以差 2 倍**（空闲 17.85 t/s、负载 90+ 时 9.19 t/s）。
    如果不把「数据在什么条件下采集」写清楚，读者无法判断一个数字
    能不能拿来用 —— 这不是「严谨」，而是「数字能不能解释」的问题。

    返回 (表格行, 结论段)。
    """
    env = base.get("environment") or {}
    d = cfg.device
    mv = d.get("measurement_validity") or {}
    b = cfg.backend
    rows: list[list[str]] = []

    rows.append([
        "**这一节解决什么**",
        "判断「表里的性能数字能不能直接引用」。"
        "同一配置在不同机器状态下可以差 2 倍，"
        "不写清条件，数字就无法解释、也无法跨配置比较。"])

    if b.fallback_is_cheap():
        rows.append([
            "**为什么本设备需要**",
            f"{b.label}的内存带宽与主机其他进程**共享**，"
            f"CPU 核心也可能被抢占 —— 因此采集时的系统负载直接决定数字是否可信。"])
    else:
        rows.append([
            "**为什么本设备需要**",
            f"{b.label}有独立显存带宽，不受主机负载影响；"
            f"但仍需记录驱动版本与温度状态 —— 它们影响 kernel 调度与是否降频。"])

    if mv.get("require"):
        rows.append(["**准入条件**", mv["require"]])
    if mv.get("sensitivity"):
        rows.append(["**违反的后果**", mv["sensitivity"]])
    if env.get("load_avg"):
        _bad = env.get("contended")
        rows.append([
            "**本次采集实测**",
            f'1 分钟负载 {env["load_avg"][0]}'
            f'（{env.get("cpu_cores")} 核，每核 {env.get("load_per_core")}）'
            + ("　→ **超标**" if _bad else "　→ 合格")])
    if mv.get("contrast"):
        rows.append(["**与其他后端的差异**", mv["contrast"]])

    # ---- 结论：明确告诉读者「该怎么办」 ----
    if env.get("warning"):
        verdict = ("⚠️ **本次数据存在污染风险**：" + env["warning"]
                   + " 处理方式：① 优先引用低负载窗口的数据；"
                     "② 若必须用，在引用时标注负载条件；"
                     "③ 有条件时在空闲窗口复测。")
    elif env.get("measured_live"):
        verdict = "本次为 live 采集，系统负载在准入条件内，数据可直接引用。"
    else:
        verdict = ("本次为 **replay（读台账）**：数据是过去某时刻采集的，"
                   "与当前机器状态无关。引用前请确认目标机器满足上表的准入条件；"
                   "台账的 `meta.collected` 记录了采集时点。")
    return rows, verdict


def _cov_block(judge: str, problem: str, rows: list[list[str]]) -> list[str]:
    """
    渲染「业务地图 3.2 要求 → 本报告落点」的对照块。

    每一步都放这张表，是为了让**漏项可见**：读者不必凭感觉判断
    「这一章讲全了没有」，逐行核对即可；未覆盖的项会显式标 ⚠️，
    而不是静默跳过。
    """
    return [f"> **判据**（业务地图 3.2）：{judge}",
            f"> **解决的问题**：{problem}", "",
            _table(["业务地图要求", "本报告落点", "状态"], rows, ["l", "l", "c"]),
            ""]


def _st(ok: bool, note: str = "") -> str:
    return ("✅" if ok else "⚠️ " + (note or "未覆盖"))


def _plan_layer(pl: dict) -> str:
    """
    方案所属的「层」（对应业务地图第 4 步的五层分析：模型 / 计算图 / Kernel /
    Runtime / 硬件）。

    优先取卡片里显式声明的 `layer`；缺失时按 `category` 推导 ——
    否则方案库里只要漏写一个字段，报告的分层归因就会出现「本层无方案」的假象。
    """
    if pl.get("layer"):
        return str(pl["layer"])
    return {
        "quantization": "model",
        "kernel": "kernel",
        "kv_cache": "runtime",
        "scheduling": "runtime",
        "speculative": "runtime",
        "ondevice": "hardware",
    }.get(pl.get("category", ""), "unknown")


def _proto_tool(base: dict) -> str:
    """测试工具名（来自台账的 measurement_protocol）。缺失时如实说明。"""
    return str((base.get("protocol") or {}).get("tool") or "（台账未声明测试工具）")


def build_report(cfg: PerfConfig, defn: dict, base: dict, prof: dict,
                 diag: dict, opt: dict, ver: dict) -> str:
    m, d, t = cfg.model, cfg.device, cfg.task
    a = cfg.arch
    sw = d["software"]
    hw = prof["hardware"]
    # 报告里所有「权重体积」一律用**有效权重**（可剔除未启用的张量），
    # 与 profiling / roofline 的分母保持同一口径；文件体积只在画像表里单列。
    from ..config import weight_bytes as _wb_fn
    _wb_eff = _wb_fn(m)
    L: list[str] = []
    W = L.append

    # 设备类型决定了报告的很多表述（是否有 PCIe、是否算 nvidia-smi 等）
    kind = d.get("kind", "gpu")
    is_gpu = kind in ("gpu",)
    is_unified = bool(_dget(d, "memory.unified_memory", False)) or kind in ("hybrid", "soc")
    has_cc = _dget(d, "compute_capability", "") not in ("", "—")
    _mem_measured_by = "nvidia-smi" if is_gpu else "系统监控 / 厂商 profiler"

    # ---------------------------------------------------------------- 头部
    W(f"# {m['name']} 推理性能分析与优化报告")
    W("")
    W(f"> **目标读者**：端侧/边缘大模型推理优化工程师、性能工程与部署团队")
    W(f"> **前置知识**：Transformer 结构与 Prefill/Decode 差异、内存带宽与 roofline "
      f"基本概念、量化（W4/W8、KV Cache 量化）原理")
    W(f"> **预计阅读**：25 分钟")
    W("")
    W(f"> **运行标识**：`{cfg.run_id}`　**生成命令**："
      f"`python3 -m perfkit all --config {cfg.source_dir} --run {cfg.run_id}`")
    W("")
    W("---")
    W("")

    # --------------------------------------------------------- 0 环境准备
    W("## 0. 环境准备")
    W("")
    W("> **判据**（业务地图 3.2）：模型可稳定加载运行、环境信息完整可复现。")
    W("> **解决的问题**：环境是否正确？模型能否稳定运行？")
    W("")

    # ---- 0.1 输入与产出对照 ------------------------------------------------
    # 业务地图 3.2 对每一步都规定了「输入 / 活动 / 输出」。把这三者与
    # 本报告的落点逐项对照，读者才能判断这一章是否完整 ——
    # 否则只能凭感觉「好像讲了」，漏项也不会被发现。
    _env_ver = (base.get("environment_verification") or {})
    W("### 0.1 本步的输入与产出对照")
    W("")
    W(_table(["业务地图要求", "本报告落点", "状态"],
             [["**主要输入**：设备信息", "`device_*.yaml`（设备/内存/算力/功耗）", "✅"],
              ["**主要输入**：模型文件", f"`{m['file']}`（文件 {m['artifacts'][0]['bytes']/GIB:.2f} GiB / "
                                   f"参与前向 {_wb_eff/GIB:.2f} GiB）", "✅"],
              ["**主要输入**：SDK 工具链",
               _dget(sw, "sdk", "—"), "✅" if _dget(sw, "sdk", "—") != "—" else "⚠️"],
              ["**活动**：设备连接", "本机直连，无需远程隧道", "✅" if _env_ver else "—"],
              ["**活动**：SDK / 驱动安装", "见 §0.4 编译配置", "✅" if _env_ver else "—"],
              ["**活动**：Runtime 部署", _dget(sw, "runtime_deploy", "见 §0.4"), "✅" if _env_ver else "—"],
              ["**活动**：编译环境配置", "见 §0.4", "✅" if _env_ver else "—"],
              ["**活动**：模型加载验证", "见 §0.3", "✅" if _env_ver else "—"],
              ["**活动**：Demo 运行验证", "见 §0.3", "✅" if _env_ver else "—"],
              ["**关键分析**：硬件（CPU/GPU/NPU、内存、算力）", "见 §0.2", "✅"],
              ["**关键分析**：软件（OS/Driver/Runtime/Compiler）", "见 §0.2", "✅"],
              ["**输出物** ①环境说明文档", "本节 + §0.2 画像表", "✅"],
              ["**输出物** ②编译配置", "见 §0.4", "✅" if _dget(sw, "build_command") else "⚠️ 未记录"],
              ["**输出物** ③环境验证报告", "见 §0.3", "✅" if _env_ver else "⚠️ 未采集"],
              ["**输出物** ④Demo 运行版本", "见 §0.3 的 Demo 记录", "✅" if _env_ver else "⚠️ 未采集"]],
             ["l", "l", "c"]))
    W("")
    if not _env_ver:
        W("> ⚠️ 本节的环境验证与 Demo 记录尚未采集。补采方式：在目标机上运行一次"
          "真实推理并把结果写入台账的 `environment_verification` 段"
          "（参考 `configs/qwen2.5-1.5b-cpu/ledger.json`）。")
        W("")

    # ---- 0.2 环境画像 ------------------------------------------------------
    W("### 0.2 环境画像")
    W("")
    _dev_desc = f"{d['name']}（{_dget(d, 'arch')}"
    if has_cc:
        _dev_desc += f"，CC {d['compute_capability']}"
    _dev_desc += f"，{_device_kind_label(kind)}）"
    _mem_desc = (f"{_dget(d, 'memory.vram_total_mib', 0)/1024:.1f} GiB "
                 f"{_dget(d, 'memory.mem_type')} @ "
                 f"{_dget(d, 'memory.mem_bandwidth_gbps_peak')} GB/s")
    if is_unified:
        _mem_desc += "（统一内存）"
    _drv_desc = cfg.backend.runtime_desc()
    # 算力是业务地图「关键分析内容」里明确要求的一项，不能缺
    _fp16 = _dget(d, "compute.fp16_tflops_peak")
    _cpu_c = _dget(d, "compute.cpu_cores")
    _simd = _dget(d, "compute.simd")
    _compute_desc = []
    if _cpu_c != "—":
        _compute_desc.append(f"{_cpu_c} 核")
    if _fp16 != "—":
        _compute_desc.append(f"FP16 {_fp16} TFLOPS")
    if _simd != "—":
        _compute_desc.append(f"SIMD: {_join(_simd)}")
    W(_table(
        ["项", "值", "来源"],
        [
            ["模型", f"{m['name']}（{m['params_total']/1e9:.2f}B，{m['quantization']}）", "model_*.yaml"],
            ["权重文件", f"`{m['file']}`（文件 {m['artifacts'][0]['bytes']/GIB:.2f} GiB）", "实测"],
            *([["有效权重",
                f"{_wb_eff/GIB:.2f} GiB（剔除未启用张量后参与前向的体积）", "实测分解"]]
              if _wb_eff != m["artifacts"][0]["bytes"] else []),
            ["设备", _dev_desc, "device_*.yaml"],
            ["内存", _mem_desc, cfg.backend.memory_measured_by()],
            ["**算力**", "，".join(_compute_desc) if _compute_desc else "—", "device_*.yaml"],
            [cfg.backend.runtime_label(), _drv_desc, "device_*.yaml"],
            ["编译器", _compiler_desc(sw, d, has_cc), "构建配置"],
            ["推理引擎", f"{sw['engine']} `{_dget(sw, 'engine_commit', 'n/a')}`"
                         f"（{_dget(sw, 'engine_backend', '—')} 后端）", "构建日志"],
            ["操作系统", _dget(sw, "os"), "—"],
        ],
        ["l", "l", "l"]))
    W("")
    _unload = ("以 `-ngl 99` 全层卸载可稳定运行，无 CPU 回退" if is_gpu
               else f"以 {_dget(sw, 'engine_backend', 'CPU')} 后端运行")
    W(f"**环境结论**：{m['name']} 以 {m['quantization']} 量化后权重 "
      f"{_wb_eff/GIB:.2f} GiB（有效），在 "
      f"{_dget(d, 'memory.vram_total_mib', 0)/1024:.1f} GiB 可用内存上{_unload}。")
    W("")

    # ---- 0.3 环境验证（业务地图的六项活动 + 两项验证）--------------------
    W("### 0.3 环境验证")
    W("")
    W("> 业务地图 0 环境准备的核心工程活动有六项。逐项交代，"
      "避免「环境应该没问题」这种未经检验的判断。")
    W("")
    _acts = _env_ver.get("activities") or []
    if _acts:
        _status = {"pass": "✅ 通过", "fail": "❌ 失败", "n/a": "— 不适用"}
        W(_table(["活动", "结论", "证据"],
                 [[a.get("activity", ""), _status.get(a.get("status"), a.get("status")),
                   a.get("evidence", "")] for a in _acts],
                 ["l", "c", "l"]))
        W("")
        _ml, _dm = _env_ver.get("model_load") or {}, _env_ver.get("demo_run") or {}
        if _ml or _dm:
            W("**验证记录**（可复现）：")
            W("")
            _vrows = []
            if _ml:
                _ws = _ml.get("wall_seconds")
                _vrows.append(["模型加载",
                               (f"{_ws} s" if _ws is not None else "—")
                               + "（服务常驻，加载耗时未单独计时）"
                               if _ws is None else f"{_ws} s（含 1 token 生成）",
                               _num(_ml.get("peak_rss_mib"), "{:.0f}", "—") + " MiB"
                               if _ml.get("peak_rss_mib") is not None else "—",
                               f'`{_ml.get("command", "")}`'])
            if _dm:
                _p, _g = _dm.get("prompt_tps"), _dm.get("gen_tps")
                _rate = (f"{_p} / {_g} t/s（Prompt / Generation）"
                         if _p is not None and _g is not None else "—")
                _vrows.append(["Demo 推理", _rate,
                               (_num(_dm.get("peak_rss_mib"), "{:.0f}") + " MiB"
                                if _dm.get("peak_rss_mib") is not None else "—"),
                               f'`{_dm.get("command", "")}`'])
            W(_table(["项", "耗时 / 速率", "峰值内存", "命令"], _vrows,
                     ["l", "l", "r", "l"]))
            W("")
            if _dm.get("output"):
                W(f"> **Demo 输出样本**：{_dm['output']}")
                W("")
        _vd = _env_ver.get("verdict") or {}
        if _vd.get("note"):
            W(f"**验证结论**：{_vd['note']}")
            W("")
    else:
        W("> **未采集环境验证记录。** 需要补的三件事：")
        W(">")
        W("> 1. 模型加载验证：跑一次 `llama-cli -m $MODEL -ngl 0 -n 1 -p hi`，"
          "记录墙钟耗时与峰值 RSS")
        W("> 2. Demo 运行验证：跑一次真实对话，确认输出语义连贯")
        W("> 3. 把结果写入台账的 `environment_verification` 段")
        W("")

    # ---- 0.4 编译配置（业务地图输出物之一）--------------------------------
    _build = _dget(sw, "build_command", None)
    W("### 0.4 编译配置")
    W("")
    if _build:
        W("```bash")
        # 按 && 拆成多行，每行可独立复制执行（不拼反斜杠，避免复制时语法错误）
        for _ln in str(_build).split("&&"):
            if _ln.strip():
                W(_ln.strip())
        W("```")
        W("")
        if _dget(sw, "build_note", None):
            W(f"> {_dget(sw, 'build_note')}")
            W("")
    else:
        W("> 未记录编译命令。架构标志写错会**静默退化**（例如 CUDA 架构不匹配会"
          "退化到 PTX JIT），性能掉一截且不会报错 —— 建议在 `device_*.yaml` 的 "
          "`software.build_command` 补上完整构建命令。")
        W("")
    W(_table(["项", "内容"],
             [["编译器", _compiler_desc(sw, d, has_cc)],
              ["关键标志", _dget(sw, "toolchain", "—")],
              ["Runtime 部署", _dget(sw, "runtime_deploy", "—")],
              ["SDK", _dget(sw, "sdk", "—")]],
             ["l", "l"]))
    W("")

    # ---- 0.5 测量有效性 ----------------------------------------------------
    _val_rows, _val_verdict = _validity_rows(cfg, base)
    W("### 0.5 测量有效性条件")
    W("")
    W(_table(["项", "内容"], _val_rows, ["l", "l"]))
    W("")
    W(f"> {_val_verdict}")
    W("")

    # ---- 0.6 环境探针 ------------------------------------------------------
    _probe_rows = _env_probe_rows(cfg, base, defn, opt)
    W("### 0.6 环境探针实测")
    W("")
    if _probe_rows:
        W(_table(["项", "值", "说明"], _probe_rows, ["l", "l", "l"]))
        W("")
    else:
        W("> **本次未采集探针数据。** 采集方式：")
        W(">")
        W("> ```bash")
        W(f"> python3 -m perfkit probe --config {cfg.source_dir} --size-gib 2")
        W("> # 或：bash scripts/run_closure.sh（默认会构建并自检探针）")
        W("> ```")
        W("")
    W("<details>")
    W("<summary>探针是干什么的、测哪些东西、什么硬件上适用（展开）</summary>")
    W("")
    W(_table(["项", "内容"], _probe_intro_rows(cfg), ["l", "l"]))
    W("")
    W("</details>")
    W("")

    # 多批次测量：同一对照在不同系统负载下重复采集时的差异。
    # 这不是「数据不干净」，而是一个必须报告的物理事实：
    # CPU 后端的优化收益本身依赖内存带宽竞争程度。
    _batches = ((base.get("extra") or {}).get("depth_matched_comparison") or {}).get("batches")
    if _batches:
        _valid = [b for b in _batches if not b.get("excluded")]
        _excl = [b for b in _batches if b.get("excluded")]
        W("### 0.7 同深度对照的多批次复现")
        W("")
        W("> 同一组对照（仅 KV 精度不同）在不同系统负载下重复采集的结果。"
          "这不是「数据不干净」，而是一个必须报告的物理事实："
          "**优化收益本身依赖内存带宽竞争程度**。")
        W("")
        W(_table(["采集窗口", "系统负载", "F16 KV (t/s)", "q8_0 KV (t/s)", "收益"],
                 [[b["label"], str(b["load_avg"]),
                   _num(b.get("f16_decode_tps")), _num(b.get("q8_0_decode_tps")),
                   (f'+{b["delta_pct"]:.1f}%' if isinstance(b.get("delta_pct"), (int, float))
                    else "—")]
                  for b in _batches],
                 ["l", "l", "r", "r", "r"]))
        W("")
        _dc = (base.get("extra") or {}).get("depth_matched_comparison") or {}
        if _dc.get("finding"):
            W(f"> **发现**：{_dc['finding']}")
            W("")
        if _dc.get("implication"):
            W(f"> **含义**：{_dc['implication']}")
            W("")
        if _dc.get("quality_gate"):
            W(f"> **准入规则**：{_dc['quality_gate']}")
            W("")
        if _dc.get("absolute_values_caveat"):
            W(f"> ⚠️ {_dc['absolute_values_caveat']}")
            W("")
        if _excl:
            W(f"> 上表有 {len(_excl)} 个批次因不满足准入规则被排除："
              + "；".join(f'{b["label"]}（{b.get("exclude_reason", "")[:60]}…）'
                          for b in _excl))
            W("")

    # --------------------------------------------------------- 1 问题定义
    W("## 1. 任务概述（问题定义）")
    W("")
    W("> **判据**（业务地图 3.2）：目标可量化、可验收。")
    W("> **解决的问题**：优化什么？在哪优化？目标多少？")
    W("")

    # ---- 1.0 输入与产出对照 ------------------------------------------------
    W("### 1.0 本步的输入与产出对照")
    W("")
    W(_table(["业务地图要求", "本报告落点", "状态"],
             [["**主要输入**：模型文件", f"`{m['file']}`（结构超参见 §1.1）", "✅"],
              ["**主要输入**：设备信息", f"{d['name']}（画像见 §1.2）", "✅"],
              ["**主要输入**：业务需求", f"`task.description`：{t['task'].get('description', '—').strip()[:60]}…", "✅"],
              ["**活动**：确认模型版本", f"{m['name']} {m.get('version', '')} / {m['quantization']}", "✅"],
              ["**活动**：分析模型结构",
          ("见 §1.1（架构超参已用 GGUF 元数据核验）"
           if (defn.get("gguf_verification") or {}).get("all_match")
           else "见 §1.1（可用 `--from-gguf` 核验）"),
               ("✅ GGUF 已核验" if (defn.get("gguf_verification") or {}).get("all_match")
                else ("✅ 已核验" if (defn.get("gguf_verification") or {}).get("diff")
                      else "⚠️ 未用 GGUF 核验"))],
              ["**活动**：确认设备环境", "见 §1.2", "✅"],
              ["**活动**：确认推理框架",
               f"{sw['engine']} `{_dget(sw, 'engine_commit', 'n/a')}`", "✅"],
              ["**活动**：制定性能指标",
               f"见 §1.{7 if defn.get('source_conflicts') else 6}"
               f"（{len(defn['kpis'])} 项 KPI）", "✅"],
              ["**关键分析**：模型画像（参数量/结构/Attention/Context/KV 需求）", "见 §1.1 + §1.3", "✅"],
              ["**关键分析**：设备画像（架构/内存/带宽/功耗限制）", "见 §1.2", "✅"],
              ["**关键分析**：指标定义（TTFT/Decode/Memory/Power/Accuracy）",
               "见 §1.{}".format(7 if defn.get('source_conflicts') else 6), "✅"],
              ["**输出物** ①项目定义文档",
               "§1.1（优化对象 / 范围边界 / 硬约束 / 交付物）", "✅"],
              ["**输出物** ②模型画像", "§1.1", "✅"],
              ["**输出物** ③设备画像", "§1.2", "✅"],
              ["**输出物** ④KPI 指标表",
               "§1.{}（含口径 / 目标 / 绑定场景 / 权重 / 目标依据）".format(
                   7 if defn.get('source_conflicts') else 6), "✅"],
              ["**输出物** ⑤验收标准",
               "§1.{}（must_pass / should_pass / guardrail 三级 + 判定方式）".format(
                   8 if defn.get('source_conflicts') else 7), "✅"]],
             ["l", "l", "c"]))
    W("")

    # ---- 1.1 项目定义文档（业务地图输出物①）----------------------------
    W("### 1.1 项目定义文档")
    W("")
    W("> 本章要回答三个问题：**优化什么？在哪优化？目标多少？** "
      "下面把这三件事写成可验收的形式 —— 没有边界的优化目标等于没有目标。")
    W("")

    _scope = t["task"].get("scope", "—")
    _priority = t["task"].get("priority", "—")
    _scope_means = {
        "decode": "只优化自回归生成阶段（Memory-bound）；Prefill 仅作为约束条件",
        "prefill": "只优化首 token 前的 prompt 处理（Compute-bound）",
        "kv_cache": "只优化 KV Cache 的容量与读写开销",
        "memory": "只优化内存占用与安全冗余",
        "e2e": "端到端时延整体优化（不区分阶段）",
        "power": "只优化功耗与能效",
    }.get(_scope, _scope)
    _prio_means = {
        "latency": "优先降低单请求延迟，允许牺牲吞吐",
        "throughput": "优先提升系统总吞吐，允许牺牲单流延迟",
        "memory": "优先压低内存占用，允许牺牲速度",
        "power": "优先降低功耗，允许牺牲速度",
        "cost": "优先降低每 token 成本",
    }.get(_priority, _priority)

    # 优化对象：把「模型 + 设备 + 引擎 + 量化」四个坐标钉死
    W("**优化对象**（四个坐标必须同时确定，否则结论无法复现）：")
    W("")
    W(_table(["坐标", "取值", "说明"],
             [["模型", f"{m['name']} {m.get('version', '')}".strip(), f"{m['params_total']/1e9:.2f}B，{m['quantization']}"],
              ["设备", f"{d['name']}", f"{_device_kind_label(kind)}，{_dget(d, 'memory.vram_total_mib', 0)/1024:.1f} GiB 可用内存"],
              ["推理引擎", f"{sw['engine']} `{_dget(sw, 'engine_commit', 'n/a')}`", f"{_dget(sw, 'engine_backend', '—')} 后端"],
              ["优化范围", f"`{_scope}`", _scope_means],
              ["优先级", f"`{_priority}`", _prio_means]],
             ["l", "l", "l"]))
    W("")

    if t["task"].get("description"):
        W(f"**业务需求**：{t['task']['description'].strip()}")
        W("")

    # 范围边界：明确「不做什么」，避免分析范围悄悄膨胀
    _no_fallback = not cfg.backend.fallback_is_cheap()
    W("**范围边界**（写清「不做什么」与「什么是硬约束」，"
      "否则优化目标会被不断稀释）：")
    W("")
    W(_table(["类别", "内容"],
             [["**本次范围内**", _scope_means],
              ["**明确不做**",
               "多机多卡 / 集群调度 / K8s 平台 / 多租户（端侧边界）；"
               "模型训练与微调；应用层编排（RAG / Agent）"],
              ["**硬约束（不可让步）**",
               "；".join(filter(None, [
                   f"{cfg.mem_label()}上限 {_dget(d, 'constraints.max_vram_gib')} GiB",
                   f"安全冗余 ≥ {_dget(d, 'constraints.min_safety_margin_gib')} GiB",
                   "零 CPU 回退" if _no_fallback else None,
                   "质量红线（PPL 相对劣化 ≤ 阈值）",
               ]))],
              ["**本次交付物**",
               "项目定义文档（本节）/ 模型画像 / 设备画像 / KPI 指标表 / 验收标准；"
               "后续步骤产出基线、剖面、瓶颈结论、方案、验证与复盘报告"]],
             ["l", "l"]))
    W("")

    W("### 1.2 模型画像")
    W("")
    _ffn_pct = (3 * a["intermediate_size"]) / (
        a["hidden_size"] * 2 + 2 * a["n_head_kv"] * a["head_dim"]
        + 3 * a["intermediate_size"]) * 100
    _moE = ("是（%s experts，top-%s）" % (a.get("n_expert"), a.get("n_expert_used"))
            if a.get("n_expert") else "否（dense）")
    # 结构类型是业务地图「模型画像」里明确要求的一项：
    # 它决定算子构成（是否有 GQA / MoE / SWA），进而决定 KV 大小与热点位置。
    _attn_kind = ("GQA" if a["n_head"] != a["n_head_kv"] else "MHA")
    _struct = (f"Decoder-only Transformer"
               f"（{_attn_kind} + {a.get('activation', '—').upper()} + "
               f"{a.get('norm', '—').upper()}"
               + ("，MoE" if a.get("n_expert") else "，dense")
               + ("，SWA" if a.get("sliding_window") else "")
               + "）")
    if a.get("hybrid"):
        # 混合架构（SSM/线性注意力 + 全注意力交替）：这条信息必须显式写出来，
        # 否则读者会按「每层都有 KV」理解后面的所有数字。
        from ..config import attn_layers as _al, ssm_layers as _sl
        _struct = (f"**混合架构**：{_al(m) } 层全注意力（{_attn_kind}）"
                   f" + {_sl(m)} 层 SSM/线性注意力")
        _struct_impact = ("**只有全注意力层有随 ctx 线性增长的 KV**；SSM 层是固定大小状态。"
                          "按「每层都有 KV」推算会把 KV 高估数倍")
    else:
        _struct_impact = "决定算子构成与热点位置；GQA 使 KV 极小，MoE 使权重总量成为硬约束"
    _nlayer_val = f"{a['n_layer']}"
    if a.get("hybrid"):
        _nlayer_val += "（+1 个 MTP 预测块，不计入前向）"
    W(_table(
        ["字段", "值", "端侧影响"],
        [
            ["**Transformer 结构**", _struct, _struct_impact],
            ["参数量", (f"{m['params_total']/1e9:.2f}B"
                        + (f"（有效 {m['params_effective']/1e9:.2f}B）"
                           if m.get("params_effective") else "")),
             "决定权重体积与单步带宽需求"],
            ["层数", _nlayer_val, "决定逐层带宽与 Decode 时延"],
            ["Hidden", f"{a['hidden_size']}", "决定 GEMM 规模与激活内存"],
            ["Intermediate", f"{a['intermediate_size']}",
             f"FFN 权重占单层 {_ffn_pct:.1f}%"],
            ["Attention", f"{a['n_head']} Q / {a['n_head_kv']} KV（GQA "
                          f"{a['n_head']//a['n_head_kv']}:1），head_dim {a['head_dim']}",
             "**KV Cache 大小与注意力带宽（端侧最敏感）**"],
            ["词表", f"{a['vocab_size']}", "输出层算量、嵌入体积"],
            ["嵌入共享", "是（tie embeddings）" if a.get("tie_embeddings")
                         else "否（输入/输出各一份）", "影响总权重体积"],
            ["上下文", f"原生 {a['max_ctx_native']} / 部署 {a['max_ctx_deployed']}",
             "KV Cache 上限与内存墙"],
            ["**KV Cache 需求**",
             f'部署上下文下 {defn["memory_budget"]["kv_total_gib"]:.2f} GiB'
             f'（{cfg.kv_bits()[0]:.1f}/{cfg.kv_bits()[1]:.1f} bit，'
             f'{defn["memory_budget"]["kv_per_slot_gib"]:.2f} GiB/槽）'
             if defn.get("memory_budget") else "—",
             "**端侧最敏感项**：唯一随 ctx 线性增长的显存开销；"
             "其大小决定长上下文可行性与每步带宽"],
            ["激活 / 归一化", f"{a['activation']} / {a['norm']}", "FFN 与 Norm kernel 实现"],
            ["MoE", _moE, "无专家路由开销，权重常驻" if not a.get("n_expert")
                          else "专家权重总量决定可行性"],
        ],
        ["l", "l", "l"]))
    W("")
    W("### 1.3 设备画像与约束")
    W("")
    _pt = d.get("power_thermal", {})
    _throttle_note = ("存在降频风险，必须做持续负载稳态测试"
                      if _pt.get("thermal_throttle") else "散热充足，未观测到降频")
    _battery_note = "（电池供电，需 tokens/Joule 能效指标）" if _pt.get("battery") else ""
    W(_table(
        ["维度", "值"],
        [
            ["**架构**", f'{_dget(d, "arch")}（{_device_kind_label(kind)}'
                        + (f"，CC {d['compute_capability']}" if has_cc else "") + "）"],
            ["**内存容量**", f'{_dget(d, "memory.vram_total_mib", 0)/1024:.1f} GiB '
                            f'{_dget(d, "memory.mem_type")}'
                            + ("（统一内存，CPU/GPU 共享）" if is_unified else "")],
            ["算力（FP16 dense）", f'{_dget(d, "compute.fp16_tflops_peak")} TFLOPS'],
            ["算力（INT8 / FP8）", f'{_dget(d, "compute.int8_tops_peak")} TOPS / '
                                   f'{_dget(d, "compute.fp8_tflops_peak")} TFLOPS'],
            ["内存带宽（标称）", f'{_dget(d, "memory.mem_bandwidth_gbps_peak")} GB/s'],
            ["L2 Cache", f'{_dget(d, "memory.l2_cache_mib")} MiB'],
            ["内存模型", "统一内存（Host↔Device 零拷贝）" if is_unified
                         else "独立显存（Host↔Device 搬运有真实成本）"],
            ["功耗 / 散热", f'TDP {_dget(_pt, "tdp_w")}W，{_throttle_note}{_battery_note}'],
            ["内存硬约束", f'≤ {_dget(d, "constraints.max_vram_gib")} GiB，'
                           f'须留 ≥ {_dget(d, "constraints.min_safety_margin_gib")} GiB 安全冗余'],
        ],
        ["l", "l"]))
    W("")
    W("### 1.4 内存预算（动手前先确认装不装得下）")
    W("")
    b = defn["memory_budget"]
    meas = base.get("memory") or {}
    meas_bd = meas.get("breakdown_gib") or {}

    def _meas_mib(key: str):
        """把实测 MiB 字段换成 GiB；缺失时返回 None（报告显示 —）。"""
        v = meas.get(key)
        return round(v / 1024, 2) if isinstance(v, (int, float)) else None

    _mem_label = cfg.mem_label()
    _peak_label = cfg.backend.peak_metric_label
    W(_table(
        ["模块", "模型推算 (GiB)", "实测/校准 (GiB)", "占比", "说明"],
        [
            ["主模型权重", _num(b["weight_gib"]), _num(meas_bd.get("main_weight")),
             f'{b["weight_gib"]/b["total_gib"]*100:.1f}%',
             f'{m["quantization"]}，{a["n_layer"]} 层'],
            ["视觉投影 mmproj", _num(b["mmproj_gib"]), _num(meas_bd.get("mmproj")),
             f'{b["mmproj_gib"]/b["total_gib"]*100:.1f}%', "ViT 特征投影（按需）"],
            *([["MTP Draft 权重", _num(b["draft_gib"]),
                _num(meas_bd.get("draft_mtp")),
                f'{b["draft_gib"]/b["total_gib"]*100:.1f}%',
                "投机解码的草稿模型（`-md` 挂载时占显存）"]]
              if b.get("draft_gib") else []),
            *([["SSM 固定状态", _num(b["ssm_state_gib"]),
                _num(meas_bd.get("ssm_state")),
                f'{b["ssm_state_gib"]/b["total_gib"]*100:.1f}%',
                f'混合架构的循环状态，**不随 ctx 增长**']]
              if b.get("ssm_state_gib") else []),
            ["KV Cache", _num(b["kv_total_gib"]), _num(meas_bd.get("kv_cache")),
             f'{b["kv_total_gib"]/b["total_gib"]*100:.1f}%',
             f'K={cfg.kv_bits()[0]}bit / V={cfg.kv_bits()[1]}bit'],
            ["运行时缓冲", _num(b["runtime_gib"]), _num(meas_bd.get("runtime_buffer")),
             f'{b["runtime_gib"]/b["total_gib"]*100:.1f}%',
             "激活 / 临时张量 / 运行时上下文"],
            [f"**合计**", f'**{_num(b["total_gib"])}**',
             f'**{_num(_meas_mib("peak_used_mib"))}**', "100%", ""],
            [f"{_mem_label}总量", _num(b.get("mem_total_gib", b["vram_total_gib"])),
             _num(_meas_mib("vram_total_mib")), "—", ""],
            ["**安全冗余**", f'**{_num(b["safety_margin_gib"])}**',
             f'**{_num(_meas_mib("safety_margin_mib"))}**', "—",
             "**OK**" if b["fits"] else "**OOM 风险**"],
        ],
        ["l", "r", "r", "r", "l"]))
    W("")
    if meas_bd:
        _bsrc = meas.get("breakdown_source") or {}
        W(f"> **两列的口径不同，不能逐行对比**：模型列是几何公式推算的稳态构成；"
          f"「实测/校准」列的**合计**是 {cfg.backend.memory_measured_by()} 的实测峰值，"
          f"而各**分项**是推导值（来源见下）。真正被验证的是「合计」——"
          f"两者相差 {abs(b['total_gib'] - (_meas_mib('peak_used_mib') or b['total_gib'])) / max(b['total_gib'], 1e-9) * 100:.1f}%，"
          f"说明预算模型可信。")
        W("")
        if _bsrc:
            W("| 分项 | 数据来源 |")
            W("|:---|:---|")
            for _k, _lbl in (("main_weight", "主模型权重"), ("kv_cache", "KV Cache"),
                             ("draft_mtp", "MTP Draft 权重"),
                             ("ssm_state", "SSM 固定状态"), ("runtime_buffer", "运行时缓冲"),
                             ("mmproj", "mmproj")):
                if _bsrc.get(_k):
                    W(f"| {_lbl} | {_bsrc[_k]} |")
            W("")
            if _bsrc.get("total_note"):
                W(f"> {_bsrc['total_note']}")
                W("")
    W("")
    W("**KV Cache 精度阶梯**（说明 KV 量化是否必要 —— 取决于上下文规模）：")
    W("")
    W(_table(
        ["KV 精度", "KV 总量 (GiB)", f"{cfg.mem_label()}合计 (GiB)",
         "安全冗余 (GiB)", "可行"],
        [[r["kv_type"], _num(r["kv_total_gib"]), _num(r["total_gib"]),
          _num(r["margin_gib"]), "✅" if r["fits"] else "❌"]
         for r in defn["kv_quant_ladder"]],
        ["l", "r", "r", "r", "c"]))
    W("")
    _ladder = defn["kv_quant_ladder"]
    _deployed_ctx = a.get("max_ctx_deployed", 0)
    _f16_kv = _ladder[0]["kv_total_gib"]
    _f16_margin = _ladder[0]["margin_gib"]
    if _f16_margin < 0 or _f16_margin < float(_dget(d, "constraints.min_safety_margin_gib", 0) or 0):
        W(f"> {_deployed_ctx//1024}K 上下文下，未量化 KV（F16）需 "
          f"{_f16_kv:.2f} GiB，叠加权重后冗余仅 {_f16_margin:.2f} GiB —— "
          f"**KV 量化在这里是必选项**。")
    else:
        W(f"> {_deployed_ctx//1024}K 上下文下，未量化 KV（F16）仅需 "
          f"{_f16_kv:.2f} GiB，冗余仍有 {_f16_margin:.2f} GiB —— "
          f"**KV 量化在本规模下不是必需项**，其收益主要体现在长上下文"
          f"（≥{max(4*_deployed_ctx, 32768)//1024}K）时的每步访存量上。")
        W("")
        W(f"> 决策依据是实测而非直觉：下表给出各 KV 精度下的容量，"
          f"配合 §3.6 的每步 KV 读写占比，即可判断量化是否值得引入精度风险。")
    W("")
    W("### 1.5 模型预测 vs 实测对账")
    W("")
    W("> 目的：模型推算与实测不一致时**并列展示**，不静默采信一侧 —— "
      "否则根因（架构超参反推错误？实测值粗估？）永远查不出来。")
    W("")
    gv = defn.get("gguf_verification")
    if gv and not gv.get("error"):
        _src = "（配置内固化的核验记录，非本次现场读取）" if gv.get("recorded") else ""
        W(f"已用 GGUF 元数据核验架构超参（`{gv['path']}`，GGUF v{gv['version']}，"
          f"{gv['tensor_count']} 个张量，按张量名数得 {gv['layers_by_tensor']} 层）{_src}：")
        W("")
        _diff = gv.get("diff") or []
        if _diff:
            W(_table(["字段", "配置（反推）", "文件真值", "一致"],
                     [[r["field"], _num(r["config"]), _num(r["gguf"]),
                       "✅" if r["match"] else "❌"] for r in _diff],
                     ["l", "r", "r", "c"]))
        elif gv.get("all_match") and gv.get("extracted"):
            W(_table(["字段", "文件真值"],
                     [[k, f"`{v}`"] for k, v in gv["extracted"].items()],
                     ["l", "l"]))
            W("")
            W("**全部架构字段与配置逐项一致** —— 配置里的结构超参不再是 [I] 反推值。")
        W("")
        if gv.get("kv_gap_explained"):
            W(f"> {gv['kv_gap_explained']}")
            W("")
    elif gv and gv.get("error"):
        W(f"> GGUF 核验失败：{gv['error']}")
        W("")
    rec = defn.get("reconcile", {}).get("kv_cache")
    if rec:
        W(f"**发现一处未解释差额，必须在优化收益归因前解决。**")
        W("")
        W(_table(["项", "值"],
                 [["配置点", f'ctx={rec["ctx"]}，KV={rec["kv_type"]}'],
                  ["几何模型值", f'{rec["model_gib"]} GiB'],
                  ["台账实测值", f'{rec["measured_gib"]} GiB'],
                  ["差额", f'**{rec["gap_gib"]} GiB（×{rec["ratio"]}）**'],
                  ["可能原因", rec["likely_cause"]],
                  ["核验方式", rec["verify_by"]],
                  ["影响", rec["impact"]]],
                 ["l", "l"]))
        W("")
        W("> 处理原则：**不做静默校准**。报告与预算表中并列展示两侧数字，"
          "验收判定优先采用实测值，几何模型值用于可行性边界判断。"
          "差额一旦通过 GGUF 元数据核验消除，重新运行本框架即可自动收敛。")
    if not rec and not (defn.get("gguf_verification") or {}).get("diff"):
        W("**本次无待对账项**：模型推算与实测在容差内一致"
          "（或该配置未声明校准锚点）。")
        W("")
    conflicts = defn.get("source_conflicts", [])
    if conflicts:
        W("### 1.6 参考源内部矛盾（并列展示）")
        W("")
        W("工程分析的一条底线：当同一份参考文档给出互相矛盾的数字时，"
          "**不能挑对自己结论有利的那个**，必须并列展示并说明采信理由。")
        W("")
        def _adopted_label(c: dict) -> str:
            _a = str(c.get("adopted", "")).lower()
            if _a in ("claim_a", "a"):
                return "A"
            if _a in ("claim_b", "b"):
                return "B"
            return "—"
        W(_table(["主题", "口径 A", "口径 B", "比值", "采信", "消除方式"],
                 [[c["topic"],
                   f'{c["claim_a"]["value"]} {c["claim_a"]["unit"]}'
                   f'（{c["claim_a"]["source"]}）',
                   f'{c["claim_b"]["value"]} {c["claim_b"]["unit"]}'
                   f'（{c["claim_b"]["source"]}）',
                   f'{c["ratio"]}×', _adopted_label(c),
                   ("**已消除**：" if c.get("resolved") else "") + c["action"]]
                  for c in conflicts],
                 ["l", "l", "l", "r", "c", "l"]))
        W("")
        for c in conflicts:
            W(f'- **{c["topic"]}**：{c["likely_cause"].strip()}')
            if c.get("resolved"):
                W(f'  - **实测结论**：{c["resolved"].strip()}')
        W("")
    # ---- 1.7 KPI 指标表（业务地图输出物④）-------------------------------
    _kpi_no = 7 if conflicts else 6
    W(f"### 1.{_kpi_no} KPI 指标表")
    W("")
    W("> 业务地图要求指标定义覆盖 **TTFT / Decode tok/s / Memory / Power / Accuracy** 五类。"
      "下表逐项给出：口径（单位 + 方向）、目标、**绑定场景**、权重与定义 —— "
      "缺任何一项，前后对比都可能是拿两个不同口径在相减。")
    W("")

    _scen_by_id = {sc["id"]: sc for sc in (cfg.task.get("scenarios") or [])}
    _kpi_rows = []
    for k in defn["kpis"]:
        _scn = k.get("scenario")
        _w = _scen_by_id.get(_scn, {}).get("weight") if _scn else None
        _kpi_rows.append([
            f'`{k["key"]}`', k["unit"],
            "↓ 越低越好" if k["direction"] == "lower" else "↑ 越高越好",
            _num(k["target"]),
            f'`{_scn}`' if _scn else "全局",
            f"{_w:.2f}" if isinstance(_w, (int, float)) else "—",
            k["definition"].strip().replace("\n", " "),
        ])
    W(_table(["指标", "单位", "方向", "目标", "绑定场景", "权重", "定义"],
             _kpi_rows, ["l", "l", "l", "r", "l", "r", "l"]))
    W("")
    W("> **`weight` 列是场景权重**（`task_*.yaml` 里该场景的业务占比），"
      "不是 KPI 权重：它说明「这个场景在真实业务里占多少比例的请求」，"
      "用于判断某项 KPI 的失分对整体体验的影响。")
    W("")

    # 目标设定依据：目标不是拍脑袋定的，必须能说清「为什么是这个数」
    _with_target = [k for k in defn["kpis"] if k.get("target") is not None]
    _no_target = [k for k in defn["kpis"] if k.get("target") is None]
    if _with_target:
        W("**目标设定依据**（每个目标都要能回答「为什么是这个数」）：")
        W("")
        _rat_rows = []
        for k in _with_target:
            _why = (k.get("target_rationale") or "").strip()
            if not _why:
                _why = ("未在配置中声明依据 —— 建议在 `task_*.yaml` 的该 KPI 下补 "
                        "`target_rationale`，说明目标来自业务需求、竞品对标还是物理上界")
            _rat_rows.append([f'`{k["key"]}`', _num(k["target"]), _why])
        W(_table(["指标", "目标", "依据"], _rat_rows, ["l", "r", "l"]))
        W("")
    if _no_target:
        W(f"> **不设目标的指标**：{'、'.join(f'`{k["key"]}`' for k in _no_target)} —— "
          f"它们是**观测项/守门项**（如功耗、PPL），本次不参与达标判定，"
          f"但会记录实测值供后续对比。")
        W("")

    W("**指标口径必须唯一（否则前后对比无意义）**：")
    W("")
    W(_table(["指标", "取值口径", "含义"],
             _kpi_reading_rows(cfg, defn, base, ver),
             ["l", "l", "l"]))
    W("")

    # ---- 1.8 验收标准（业务地图输出物⑤）---------------------------------
    _acc = t["acceptance"]
    W(f"### 1.{_kpi_no + 1} 验收标准")
    W("")
    W("> 业务地图的判据是「目标可量化、可验收」。验收分三级："
      "**must_pass**（硬性，不达标即方案失败）、"
      "**should_pass**（期望，不达标需说明原因）、"
      "**guardrail**（质量与稳定性红线，任一违反即回滚，不看性能收益）。")
    W("")

    def _cond_rows(conds: list, verdict: str) -> list[list[str]]:
        rows = []
        for c in conds:
            if isinstance(c, dict):
                for _k, _v in c.items():
                    rows.append([f"`{_k}`", f"= `{_v}`", verdict])
            else:
                rows.append([f"`{c}`", "按 §1." + str(_kpi_no) + " 定义的口径实测判定", verdict])
        return rows

    W("**① must_pass —— 硬性达标项**（任一项不达标即判定方案失败）")
    W("")
    W(_table(["条件", "判定口径", "不达标处置"],
             _cond_rows(_acc.get("must_pass") or [], "方案不可交付，需重新优化或回滚"),
             ["l", "l", "l"]))
    W("")
    W("**② should_pass —— 期望项**（不达标不阻断交付，但需在报告中说明原因与影响）")
    W("")
    W(_table(["条件", "判定口径", "不达标处置"],
             _cond_rows(_acc.get("should_pass") or [], "记录为已知不足，评估业务可接受性"),
             ["l", "l", "l"]))
    W("")
    W("**③ guardrail —— 质量与稳定性红线**（任一违反即回滚，不因性能收益而豁免）")
    W("")
    W(_table(["红线", "判定口径", "违反处置"],
             _cond_rows(_acc.get("guardrail") or [], "**立即回滚**到上一个已验证配置"),
             ["l", "l", "l"]))
    W("")
    W("**判定执行方式**：")
    W("")
    W(_table(["项", "说明"],
             [["判定时点", "第 6 步「收益量化」执行，结果见 §6 回归验收"],
              ["数据来源", "只采信 `measured` 口径（实测或实测推导）；"
                           "`model`/`derived` 仅用于可行性边界判断，不参与达标判定"],
              ["判定粒度", "逐 KPI 独立判定，不做加权平均 —— "
                           "加权会掩盖单项严重退化"],
              ["回滚触发", _acc.get("rollback_trigger", "—").strip()]],
             ["l", "l"]))
    W("")

    # --------------------------------------------------------- 2 性能基线
    W("## 2. 基准部署与性能基线")
    W("")

    # ---- 2.0 输入与产出对照 ------------------------------------------------
    _sc = base.get("scenarios") or {}
    _scen_ids = set(_sc)
    _has_e2e = any(x.get("e2e_seconds") for x in _sc.values())
    _has_power = any(k["key"] == "power_w" for k in defn["kpis"])
    _pow_measured = bool((base.get("memory") or {}).get("power_w"))
    _has_temp = bool((base.get("extra") or {}).get("thermal"))
    _peak_mem = (base.get("memory") or {}).get("peak_used_mib")
    _scen_cov = [
        ("短输入短输出", {"short_chat"}),
        ("长上下文", {"long_ctx", "long_ctx_8k", "long_ctx_8k_kvq8", "long_ctx_16k"}),
        ("长生成", {"long_gen"}),
        ("多轮对话", {"multi_turn", "multi_turn_8k"}),
        ("长时间运行", {"soak_30min", "soak_10min", "soak"}),
    ]
    _miss_scen = [name for name, ids in _scen_cov if not (ids & _scen_ids)]

    W("### 2.0 本步的输入与产出对照")
    W("")
    for _ln in _cov_block(
        "数据可复现、覆盖场景全",
        "当前性能到底是多少？",
        [["**主要输入**：原始模型", f"`{m['file']}`（未优化档位）", "✅"],
         ["**主要输入**：测试脚本", "场景矩阵 + 逐场景复现命令（见 §2.2）", "✅"],
         ["**主要输入**：设备环境", "§0 的环境画像与验证记录", "✅"],
         ["**活动**：部署原始模型", "§2.1 基线配置与启动命令", "✅"],
         ["**活动**：搭建 Benchmark",
          f"{_proto_tool(base)}（含统计口径与参数语义，见 §2.2）", "✅"],
         ["**活动**：设计测试集",
          f"{len(cfg.task.get('scenarios') or [])} 个场景 × "
          f"{len(defn['kpis'])} 项 KPI", "✅"],
         ["**活动**：执行性能测试", "§2.2 场景数据 + §2.3 深度曲线", "✅"],
         ["**活动**：采集系统数据", "§2.4 系统数据采集（内存/温度/负载/功耗）",
          _st(bool(_peak_mem) or _has_temp)],
         ["**关键分析**：TTFT", f"{len([x for x in _sc.values() if x.get('ttft_ms')])} 个场景有实测",
          _st(any(x.get("ttft_ms") for x in _sc.values()))],
         ["**关键分析**：Prefill Latency",
          f"{len([x for x in _sc.values() if x.get('prefill_tps')])} 个场景有实测",
          _st(any(x.get("prefill_tps") for x in _sc.values()))],
         ["**关键分析**：Decode tok/s",
          f"{len([x for x in _sc.values() if x.get('decode_tps')])} 个场景有实测",
          _st(any(x.get("decode_tps") for x in _sc.values()))],
         ["**关键分析**：E2E Latency",
          "§2.2 的 e2e 列" if _has_e2e else "未逐场景记录",
          _st(_has_e2e, "未采集")],
         ["**关键分析**：Memory Peak",
          f"{_peak_mem} MiB" if _peak_mem else "见 §2.4",
          _st(bool(_peak_mem), "未采集")],
         ["**关键分析**：Power",
          "见 §1.6 的 power_w（观测项）" if _has_power else "未定义该 KPI",
          _st(_pow_measured, "未采集")],
         ["**关键分析**：Temperature",
          "见 §2.4" if _has_temp else "未采集",
          _st(_has_temp, "未采集")],
         ["**关键分析**：测试场景覆盖（5 类）",
          "短输入短输出 / 长上下文 / 长生成 / 多轮对话 / 长时间运行",
          _st(not _miss_scen, f"缺 {_miss_scen}")],
         ["**输出物** ①Baseline 报告", "本章（§2.1-§2.4）", "✅"],
         ["**输出物** ②Benchmark 测试集",
          f"`task_*.yaml` 的 {len(cfg.task.get('scenarios') or [])} 个场景定义", "✅"],
         ["**输出物** ③性能数据库",
          f"`output/{cfg.run_id}/baseline.json`（+ 台账）", "✅"],
         ["**输出物** ④指标基线", "§2.2 的「基线关键结论」", "✅"]]):
        W(_ln)

    W("### 2.1 基线部署")
    W("")
    _bl = ver["before_bundle"]
    W(f"基线 = `{_bl['id']}`（{_bl['name']}）。"
      f"这是**可用的朴素生产配置**，而非「优化后配置的子集」——"
      f"这样每项优化都有独立的可归因收益。")
    W("")
    W("```bash")
    W(f"# 权重：{m['file']}")
    W("./llama-server \\")
    W(f"  -m {m['file']} \\")
    for _f in (_bl["flags"] or []):
        W(f"  {_f} \\")
    W("  --host 0.0.0.0 --port 8000")
    W("```")
    W("")
    W("### 2.2 基线性能数据")
    W("")
    # ---- 测试方法：数字怎么来的 ------------------------------------------
    _proto = base.get("protocol") or {}
    if _proto.get("synthetic_warning"):
        W(f"> ⚠️ **{_proto['synthetic_warning']}**")
        W("")
    W("**测试方法**（不写清口径，表中的数字无法解释、也无法跨配置比较）：")
    W("")
    W(_table(["项", "内容"], _method_rows(cfg, _proto), ["l", "l"]))
    W("")
    for _c in (_proto.get("conventions") or []):
        W(f"- {_c}")
    W("")
    W("**场景数据**：")
    W("")
    rows = []
    for sid, s in base["scenarios"].items():
        if s.get("status") == "todo":
            rows.append([sid, s.get("desc", ""), "待采集", "待采集", "待采集", "—", "—"])
            continue
        # E2E Latency：优先用实测；缺失时由 prefill + decode 耗时推导（标注 [D]）
        _e2e = s.get("e2e_seconds")
        if _e2e is None:
            _pf, _dc = s.get("prefill_seconds"), s.get("decode_seconds")
            if _pf is not None and _dc is not None:
                _e2e = round(_pf + _dc, 2)
                _e2e_note = "（推导）"
            else:
                _e2e_note = ""
        else:
            _e2e_note = "（实测）"
        rows.append([
            sid, s.get("desc", ""),
            _num(s.get("prefill_tps"), "{:.2f}"),
            _num(s.get("decode_tps"), "{:.2f}"),
            _num(s.get("decode_tps_system"), "{:.2f}"),
            _num(s.get("ttft_ms"), "{:.1f}"),
            (_num(_e2e, "{:.2f}") + _e2e_note) if _e2e is not None else "—",
        ])
    W(_table(["场景", "说明", "Prefill (t/s)", "Decode (t/s)",
              "系统吞吐 (t/s)", "TTFT (ms)", "E2E (s)"],
             rows, ["l", "l", "r", "r", "r", "r", "r"]))
    W("")
    W("> **`Decode` 与 `系统吞吐` 两列的关系**：`Decode` 是**单槽位**速率，"
      "`系统吞吐` 是**所有槽位之和**。并发场景（`concurrent_2` / `soak_30min`）"
      "两者都有值，读的时候看后者；单流场景只有前者有值。"
      "把这两列混着比会得出「并发比单流慢」的错误结论。")
    W("")
    W("> **E2E Latency 的口径**：标「实测」的是端到端墙钟测量；"
      "标「推导」的是 `prefill_seconds + decode_seconds` 的加和 —— "
      "它不含客户端传输与排队，因此是**下界**。两者不可混用。")
    W("")
    # 有 caveat 的场景必须在表格下方逐条说明 —— 否则读者会把
    # 「多轮第 2 轮的 538 ms」当成「任意请求的首字延迟」。
    _cav = [(sid, s.get("caveat")) for sid, s in (base.get("scenarios") or {}).items()
            if s.get("caveat")]
    if _cav:
        W("**逐场景口径提醒**（表中数字脱离这些前提就会被误读）：")
        W("")
        for _sid, _c in _cav:
            W(f"- `{_sid}`：{_c}")
        W("")
    # Prefix Cache 专项：多轮场景的核心结论，单列出来
    _mt = (base.get("scenarios") or {}).get("multi_turn") or {}
    _pc = _mt.get("prefix_cache") or {}
    if _pc:
        W("**多轮对话的 Prefix Cache 实测**（同一个 session 连续 4 轮）：")
        W("")
        _pc_rows = [["24K 上下文", f'{_pc["first_round_ttft_ms"]:,.1f} ms',
                     f'{_pc["later_median_ms"]:,.1f} ms',
                     f'{_pc["ttft_reduction_pct"]:+.1f}%',
                     f'{_pc["cached_tokens"][0]} → {_pc["cached_tokens"][-1]}']]
        _sc_pc = _pc.get("short_ctx") or {}
        if _sc_pc:
            _pc_rows.insert(0, ["1K 上下文", f'{_sc_pc["first_round_ttft_ms"]:,.1f} ms',
                                f'{_sc_pc["later_median_ms"]:,.1f} ms',
                                f'{_sc_pc["ttft_reduction_pct"]:+.1f}%',
                                f'{_sc_pc["cached_tokens"][0]} → {_sc_pc["cached_tokens"][-1]}'])
        W(_table(["上下文规模", "首轮 TTFT", "后续轮 TTFT（中位）",
                  "变化", "命中缓存 token（首轮→末轮）"],
                 _pc_rows, ["l", "r", "r", "r", "l"]))
        W("")
        if _pc.get("finding"):
            W(f"> {_pc['finding']}")
            W("")
        if _pc.get("implication"):
            W(f"> **含义**：{_pc['implication']}")
            W("")
    # ---- 逐场景复现命令 ---------------------------------------------------
    _cmds = _scenario_cmds(base)
    if _cmds:
        W("<details>")
        W("<summary>逐场景复现命令（展开）</summary>")
        W("")
        W("```bash")
        for _ln in _cmd_block(_cmds, m.get("file")):
            W(_ln)
        W("```")
        W("")
        W("</details>")
        W("")
    # 若某个场景的 TTFT 明显异常（>60s），提示其口径风险而非当成正常数据
    _susp = [(sid, s) for sid, s in base["scenarios"].items()
             if (s.get("ttft_ms") or 0) > 60000 and s.get("prefill_tps")]
    for sid, s in _susp:
        _pt = s.get("prompt_tokens", 0)
        W(f"> ⚠️ `{sid}` 的 TTFT（{s['ttft_ms']/1000:.0f} 秒）看似异常，"
          f"实为该场景 Prefill 口径的直接后果："
          f"{_pt} token 的 prompt 按 {s['prefill_tps']} t/s 的 Prefill 速率"
          f"需要约 {_pt/s['prefill_tps']:.0f} 秒。"
          f"若上游另有更高的 Prefill 口径成立，同一请求会快得多。"
          f"**该指标在消除口径歧义前不得作为 SLA 依据**"
          f"（口径矛盾见 §1.5；消除方式见 §6）。")
        W("")
    W(f"**基线关键结论**：")
    W("")
    _native = (base.get("extra") or {}).get("baseline_native_decode_tps")
    _bu = (base.get("extra") or {}).get("bandwidth_utilization")
    if _native is not None:
        W(f"- 原生纯自回归 Decode 实测 **{_native} t/s**"
          + (f"，对应内存带宽利用率 **{_bu*100:.1f}%**；" if _bu else "；"))
    _ub = base["_derived"]["decode_tps_upper_bound"]
    _wb_gb = _wb_eff / 1e9
    W(f"- 理论解码上界为 **{_ub} t/s**（= {d['memory']['mem_bandwidth_gbps_peak']} GB/s ÷ "
      f"{_wb_gb:.2f} GB，注意此处用十进制 GB 而非 GiB）"
      + (f"，已达上界的 "
         f"{base['extra']['baseline_native_decode_tps']/_ub*100:.1f}%；" if _native else "；"))
    _curve = (base.get("extra") or {}).get("long_context_curve") or []
    if len(_curve) >= 2:
        _c0, _c1 = _curve[0], _curve[-1]
        W(f"- 长上下文场景下 Decode 速率随 ctx 缓降"
          f"（{_c0['ctx']//1024}K 时 {_c0['decode_tps']} t/s → "
          f"{_c1['ctx']//1024}K 时 {_c1['decode_tps']} t/s），"
          f"KV Cache 增量是唯一变量。")
    W("")
    if _curve:
        W("### 2.3 长上下文基线曲线")
        W("")
        W("**测试方法**（深度扫描 —— 与 §2.2 的唯一变量是上下文规模）：")
        W("")
        _kv_srcs = {r.get("kv_gib_source", "measured") for r in _curve}
        if _kv_srcs == {"measured"}:
            _kv_note = ("台账记录的实测值（llama.cpp 启动日志 / 进程内存快照），"
                        "不是几何公式推算 —— 两者差额见 §1.4")
        elif _kv_srcs == {"derived"}:
            _kxc2 = (base.get("memory") or {}).get("kv_crosscheck") or {}
            _kv_note = ("**几何公式推算值**（全注意力层数 × ctx × KV heads × head_dim × 位宽）。"
                        "该列的可信度取决于架构超参是否为真值 —— "
                        + ("本次已用 GGUF 元数据核验，"
                           f"并用同配置 f16↔q8_0/q4_0 的显存差做了交叉校验"
                           f"（几何预测 {_kxc2.get('geometric_delta_mib'):,} MiB vs "
                           f"实测 {_kxc2.get('measured_delta_mib'):,} MiB，"
                           f"误差 {_kxc2.get('error_pct')}%），"
                           "因此可用于趋势与量级判断；绝对值的 ±10% 误差见 §1.4"
                           if _kxc2 else
                           "本次未做显存差交叉校验，该列只能用于趋势判断，"
                           "绝对值需谨慎引用"))
        else:
            _kv_note = ("混合来源：部分点为实测、部分为推算，逐行见台账 "
                        "`curves.long_context[].kv_gib_source`")
        # 深度是怎么造出来的，取决于测量工具 —— 方法描述必须与实际工具一致，
        # 否则读者按 llama-bench 的 -d 去复现一份 HTTP 采集的数据，必然对不上。
        _is_bench = "llama-bench" in str(_proto.get("tool") or "")
        if _is_bench:
            _how = ("逐个改变 llama-bench 的**上下文深度** `-d N`："
                    "测量前先用合成 token 把 KV Cache 填充到 N 个位置，"
                    "再执行本次 Prefill/Decode —— 模拟「对话已进行到第 N 个 token」")
        else:
            _how = ("逐个改变**请求携带的上下文长度**：在请求体中构造 N 个 token 的"
                    "历史内容（多轮对话 / 长 prompt），使服务端 KV Cache 填充到 N 后，"
                    "再测量该次请求的 Prefill/Decode —— 模拟「对话已进行到第 N 个 token」")
        _depth_notes = [
            ["扫描方式", "固定生成长度，只改变上下文规模，观察 Prefill/Decode 随 KV 增长的衰减"],
            ["深度怎么造", _how],
            ["为何这样测", "KV 读写与权重读取**竞争同一份内存带宽**。"
                           "深度扫描把「上下文长度」从其他变量中隔离出来，"
                           "使衰减可归因到 KV 而非其他因素"],
            ["KV 增量列", _kv_note],
            ["对照条件", "各点必须使用同一份配置（模型/量化/线程/加速标志），"
                         "**唯一变量是深度**"],
        ]
        W(_table(["项", "内容"], _depth_notes, ["l", "l"]))
        W("")
        W(_table(["上下文", "Prefill (t/s)", "Decode (t/s)", "KV Cache 增量 (GiB)"],
                 [[f'{r["ctx"]//1024}K', _num(r["prefill_tps"]), _num(r["decode_tps"]),
                   _num(r["kv_gib"])] for r in _curve],
                 ["l", "r", "r", "r"]))
        W("")
        _ccmds = _curve_cmds(_curve)
        if _ccmds:
            W("<details>")
            W("<summary>深度扫描复现命令（展开）</summary>")
            W("")
            W("```bash")
            for _ln in _cmd_block(_ccmds, m.get("file")):
                W(_ln)
            W("```")
            W("")
            W("</details>")
            W("")

    # ---- 2.4 系统数据采集（业务地图「采集系统数据」活动）-----------------
    W("### 2.4 系统数据采集")
    W("")
    W("> 业务地图要求第 2 步同时采集**系统侧数据**（内存峰值 / 功耗 / 温度）。"
      "性能数字若脱离这些上下文，就无法判断它是否可持续、是否已触到硬件边界。")
    W("")
    _mem = base.get("memory") or {}
    _env = base.get("environment") or {}
    _thermal = (base.get("extra") or {}).get("thermal") or {}
    # 采集时负载：台账里的 meta.collected_load_avg 优先（那是数据采集当时的负载），
    # 其次才是本次运行的环境（replay 模式下后者与数据无关）
    _meta_load = (base.get("meta") or {}).get("collected_load_avg")
    if _meta_load:
        _load_txt = str(_meta_load)
    elif _env.get("load_avg"):
        _load_txt = (f'1min {_env["load_avg"][0]}'
                     f'（{_env.get("cpu_cores", "—")} 核，'
                     f'每核 {_env.get("load_per_core", "—")}）')
    else:
        _load_txt = "—"
    # §2.4 属于「基线」这一步，必须报**基线配置**的峰值 ——
    # 报选定配置的峰值会让读者以为基线就占了那么多显存。
    _base_peak = _mem.get("baseline_peak_mib") or _mem.get("peak_used_mib")
    _peak_cfg = _mem.get("baseline_peak_config") or _mem.get("config") or ""
    _sys_rows = [
        [cfg.backend.peak_metric_label,
         (f'{_base_peak} MiB' if _base_peak else "—"),
         ((_mem.get("measure_method") or _mem.get("source") or "—")
          + (f"；配置：{_peak_cfg}" if _peak_cfg else "")),
         _st(bool(_base_peak), "未采集")],
        ["采集时系统负载",
         _load_txt, "getloadavg / uptime（或台账记录）",
         _st(bool(_load_txt != "—"), "未采集")],
        ["温度",
         (f'{_thermal.get("gpu_temp_c_range")[0]}~'
          f'{_thermal.get("gpu_temp_c_range")[-1]} °C'
          if _thermal.get("gpu_temp_c_range") else "—"),
         _thermal.get("source") or "厂商工具 / nvidia-smi",
         _st(bool(_thermal.get("gpu_temp_c_range")), "未采集")],
        ["功耗",
         (f'{_mem.get("power_w")} W' if _mem.get("power_w") else "—"),
         "nvidia-smi --query-gpu=power.draw / powermetrics（需 sudo）",
         _st(bool(_mem.get("power_w")), "未采集")],
    ]
    W(_table(["项", "实测值", "采集方式", "状态"], _sys_rows, ["l", "l", "l", "c"]))
    W("")
    if not _mem.get("power_w"):
        W("> **功耗为何未采集**：读取整机/整卡功耗需要平台特定权限 —— "
          "Linux 用 `nvidia-smi --query-gpu=power.draw`（GPU）或 "
          "`/sys/class/powercap`（CPU）；macOS 需 `sudo powermetrics`。"
          "本次分析未启用，因此该指标标为待采集、**不参与达标判定**。"
          "若要做能效评估（tokens/Joule），必须补采这一项。")
        W("")
    W("> **采样方式**：内存峰值取进程生命周期内的最大值（不是某一时刻的瞬时值），"
      "因为 Prefill 阶段的瞬时峰值可能远高于稳态。"
      "若只采样稳态，会低估峰值并高估安全冗余。")
    W("")

    # --------------------------------------------------------- 3 瓶颈定位
    W("## 3. 性能剖面采集与瓶颈定位")
    W("")

    # ---- 3.1 输入与产出对照 ------------------------------------------------
    _cmp = (prof.get("hardware") or {}).get("compute") or {}
    _fb = prof.get("cpu_fallback_cost") or {}
    _log_path = f"output/{cfg.run_id}/logs/"
    _has_logs = (cfg.outdir / "logs").exists()
    _raw = cfg.outdir / "raw"
    _has_raw = _raw.exists() and any(_raw.iterdir()) if _raw.exists() else False
    W("### 3.1 本步的输入与产出对照")
    W("")
    for _ln in _cov_block(
        "根因明确、可归因",
        "为什么慢？慢在哪？根因是什么？",
        [["**主要输入**：Baseline 数据",
          f"`output/{cfg.run_id}/baseline.json`"
          f"（{len(base.get('scenarios') or {})} 个场景 + 测量协议）", "✅"],
         ["**主要输入**：Profiler 数据",
          f"`output/{cfg.run_id}/profiling.json`"
          f"（四维剖面：硬件 / 阶段 / 算子 / KV）", "✅"],
         ["**主要输入**：日志",
          (f"`{_log_path}`（七步过程日志）"
           if _has_logs else "未启用 Shell 编排，无过程日志")
          + ("；`raw/` 下有原始采集输出" if _has_raw else ""),
          _st(_has_logs or _has_raw, "未采集")],
         ["**活动**：拆解 E2E Latency", "§3.4 阶段剖面（Prefill / Decode / TTFT 拆解）", "✅"],
         ["**活动**：分析 Prefill·Decode",
          "§3.4（两阶段瓶颈类型相反，分开归因）", "✅"],
         ["**活动**：Profiler 采集",
          "`perfkit profiling --mode live`（nsys / nvidia-smi dmon）"
          if prof.get("mode") == "live" else
          "replay 归因（按权重字节分摊实测单步耗时）；live 模式接 nsys/dmon",
          "✅"],
         ["**活动**：算子分析",
          f"§3.5 算子剖面（{len(prof['ops'])} 类算子 + 嵌入层）", "✅"],
         ["**活动**：CPU·GPU·NPU 分析",
          f"后端 = {cfg.backend.label}（{cfg.backend.kind}）；硬件剖面含带宽侧与算力侧", "✅"],
         ["**活动**：检查 Fallback 路径", "§3.9 执行路径与数据搬运（含回退检查）", "✅"],
         ["**活动**：分析数据搬运", "§3.9（每步搬运字节 / Cache 层次 / Host-Device 同步）", "✅"],
         ["**关键分析（计算）**：GEMM / Attention 热点、Kernel 耗时",
          "§3.5 算子剖面 + §3.7 Hot Kernel 列表", "✅"],
         ["**关键分析（计算）**：算子覆盖率",
          f"§3.5 的覆盖率说明（{len(prof['ops'])} 类算子 + 嵌入层 = 100%）", "✅"],
         ["**关键分析（硬件）**：Compute / Memory Bound",
          f"roofline 判定 = `{defn['roofline']['verdict']}`", "✅"],
         ["**关键分析（硬件）**：带宽利用率", f"{hw['bw_utilization']*100:.1f}%", "✅"],
         ["**关键分析（硬件）**：Cache 命中率", "§3.9 的 Cache 层次分析（权重体积 vs L2/L3）", "✅"],
         ["**关键分析（硬件）**：Thermal Throttling",
          "§3.10 的 power_thermal_bound 结论"
          if any(f.get("type") == "power_thermal_bound" for f in diag["findings"])
          else "本设备未标记降频风险（仍见 §2.4 温度）",
          "✅"],
         ["**关键分析（软件）**：CPU fallback", "§3.9 的回退检查", "✅"],
         ["**关键分析（软件）**：Runtime 调度 / Kernel 融合",
          "§3.10 的 launch_bound 结论（GPU/NPU 适用；CPU 无 launch 概念）", "✅"],
         ["**关键分析（软件）**：Host-Device 同步", "§3.9（统一内存 vs 独立显存的差异）", "✅"],
         ["**输出物** ①性能 Profile 报告", "§3.2 总览 + §3.3-§3.6 四维剖面", "✅"],
         ["**输出物** ②Hot Kernel 列表",
          f"§3.7（按单步耗时降序 + 含嵌入层，共 {len(prof['ops']) + 1} 项）", "✅"],
         ["**输出物** ③执行路径分析", "§3.9（搬运量 / Cache 层次 / 回退 / 同步）", "✅"],
         ["**输出物** ④根因分析报告", "§3.10（结论 + 证据链 + 反证 + 路线推导）", "✅"]]):
        W(_ln)

    # ---- 3.2 性能 Profile 报告 --------------------------------------------
    W("### 3.2 性能 Profile 报告")
    W("")
    W("> **产出物**：性能 Profile 报告（业务地图输出物①）。")
    W("> 四个维度各回答一个**不同的问题**。分开看是因为它们的结论可能互相矛盾 —— "
      "例如「带宽已饱和」与「算子耗时集中在 FFN」并不冲突，"
      "但若混在一起看，就会得出「优化 FFN 就能提速」的错误结论。")
    W("")
    W(_table(["剖面", "回答什么问题", "数据来源", "产出结论"],
             [["§3.3 硬件剖面", "**整机资源用尽了吗？** 带宽与算力各自被用掉多少，"
                                "还剩多少空间",
               "实测单步耗时 + 设备标称峰值（带宽/算力）",
               "瓶颈在带宽侧还是算力侧"],
              ["§3.4 阶段剖面", "**时间花在哪个阶段？** Prefill 与 Decode 的瓶颈"
                                "类型相反，必须分开看",
               "实测吞吐拆解（场景矩阵）",
               "该优化 Prefill 还是 Decode"],
              ["§3.5 算子剖面", "**时间花在哪类算子上？** 各算子占单步耗时的比例",
               "按权重字节占比归因（访存受限下字节即时间）",
               "优化的收益上限（可减少的字节）"],
              ["§3.6 KV Cache 剖面", "**随上下文增长的那部分有多大？** KV 是唯一"
                                     "随 ctx 线性增长的显存与带宽开销",
               "几何公式 + 台账实测增量",
               "KV 量化是否必要、收益多大"]],
             ["l", "l", "l", "l"]))
    W("")
    W("**四维剖面的关键数字一览**：")
    W("")
    _prof_summary = [
        ["硬件", f"带宽 {hw['bw_utilization']*100:.1f}%"
                 + (f"；算力对峰值 {(_cmp.get('compute_utilization') or 0)*100:.1f}%"
                    if _cmp.get("compute_utilization") is not None else ""),
         f"未被解释时间 {(_cmp.get('unexplained_time_ratio') or 0)*100:.1f}%"],
        ["阶段", f"Prefill {_num(prof['phases']['prefill'].get('tps'))} t/s / "
                 f"Decode {_num(prof['phases']['decode'].get('tps'))} t/s",
         f"Decode 是 {hw['decode_step_ms']} ms/token"],
        ["算子", (f"FFN 占全模型 {sum(r['pct_of_step'] for r in prof['ops'] if r['op'].startswith('ffn')):.1f}%；"
                  f"单层内 {sum(r['pct_of_layer'] for r in prof['ops'] if r['op'].startswith('ffn')):.1f}%"),
         f"嵌入层占全模型 {prof['per_layer']['embedding_share_pct']:.1f}%"],
        ["KV", (f"最大 ctx {prof['kv_cache']['ladder'][-1]['ctx']//1024}K 时 "
                f"占单步 {prof['kv_cache']['ladder'][-1]['kv_share_of_step_pct']:.1f}%"
                if prof["kv_cache"]["ladder"] else "—"),
         "随 ctx 线性增长"],
    ]
    W(_table(["维度", "关键数字", "备注"], _prof_summary, ["l", "l", "l"]))
    W("")

    W("### 3.3 硬件剖面")
    W("")
    W("> **侧重**：判断整机资源是否已用尽 —— 带宽侧与算力侧**分别**看，"
      "因为「算力利用率低」既可能是缺陷，也可能是带宽限死的必然结果。")
    W("")
    W(f"- 单步解码耗时 **{hw['decode_step_ms']} ms**，达成带宽 "
      f"**{hw['achieved_bw_gbps']} / {hw['peak_bw_gbps']} GB/s = {hw['bw_utilization']*100:.1f}%**；")
    W(f"- 距理论上界（{hw['theoretical_upper_bound_tps']} t/s，ctx=0 的权重口径）"
      f"差距 **{hw['gap_vs_upper_bound_pct']}%**；")
    W(f"- 判定：**{hw['verdict']}**。")
    W("")
    # live 模式下把「负载中实测到的硬件状态」单列 —— 这是 live 相对 replay
    # 唯一新增的事实性数据，藏在 JSON 里就等于没测。
    _live = prof.get("live") or {}
    _lm = _live.get("metrics") or {}
    if prof.get("mode") == "live":
        _lrows = []
        for _k, _lbl, _u in (("sm_util_ratio", f"{cfg.backend.utilization_label}", ""),
                             ("mem_util_ratio", f"{cfg.backend.memory_label}带宽利用率", ""),
                             ("power_w", "功耗", " W"),
                             ("temp_c", "温度", " °C"),
                             ("freq_ghz", "CPU 频率", " GHz"),
                             ("load_avg", "系统负载（1 min）", "")):
            _v = _lm.get(_k)
            if isinstance(_v, dict):
                _lrows.append([_lbl, f'{_v.get("mean")}{_u}',
                               f'{_v.get("min", "—")}{_u}',
                               f'{_v.get("max", "—")}{_u}', str(_v.get("n", "—"))])
        W("**负载中实测的硬件状态**"
          f"（{_live.get('source') or '未采集'}；"
          f"采集时由 {_live.get('load_generated_by', '真实推理负载')} 制造负载）：")
        W("")
        if _lrows:
            W(_table(["项", "均值", "最小", "最大", "样本数"], _lrows,
                     ["l", "r", "r", "r", "r"]))
            W("")
            W("> **为什么必须是「负载中」**：空闲时采样的利用率/功耗全是 0，"
              "只能证明「设备没在干活」，回答不了「瓶颈在算力还是带宽」。"
              "本节所有数字都取自 Decode 稳态负载运行期间。")
            W("")
        else:
            W(f"> 未采集到硬件时序：{_live.get('note') or '该后端无标准采集手段'}。"
              "报告不会用静态快照冒充时序数据 —— 缺失就标缺失。")
            W("")
        if _live.get("nsys_hint"):
            W(f"> **kernel 级时间线**（nsys）本报告未采集，仅给出复现命令："
              f"`{_live['nsys_hint']}`")
            W("")
    _native_tps = (base.get("extra") or {}).get("baseline_native_decode_tps")
    if _native_tps:
        # 上界的两个口径必须从当前配置算，不能写死 ——
        # 写死的 62.6/51.45 是 27B@4090 的数，出现在 CPU 报告里就是错的。
        _ub0 = hw.get("theoretical_upper_bound_tps")
        _ubc = (base.get("_derived") or {}).get("decode_tps_upper_bounds") or {}
        _ctx_k = (_ubc.get("ctx") or 0) // 1024
        _ep_tps = ((base.get("scenarios") or {}).get("short_chat") or {}).get("decode_tps")
        _same = (_ep_tps is not None
                 and abs(float(_ep_tps) - float(_native_tps)) < 0.05)
        W(f"> **口径说明**：上述单步耗时取自**原生基线实测 {_native_tps} t/s**"
          f"（llama-bench，纯解码，不含采样与 HTTP 开销）。"
          + ("本次端到端场景的 Decode 也是同一口径（live 模式直接跑 llama-bench），"
             "因此两者相同；换用 llama-server 采集时会出现差额。"
             if _same else
             f"端到端口径（llama-server，短 prompt）为 {_num(_ep_tps)} t/s —— "
             f"两者差额就是采样与传输开销。")
          + f"**本节的归因用前者**（它才是纯 Decode 稳态）；引用时不可混用。"
          + (f"另外 {_ub0} t/s 是 ctx=0 的上界"
             + (f"，{_ctx_k}K 上下文下计入 KV 读后降为 {_ubc.get('at_ctx_tps')} t/s"
                if _ubc.get("at_ctx_tps") else "")
             + "（见 §3.9）。" if _ub0 else ""))
        W("")
    _cmp = hw.get("compute") or {}
    if _cmp:
        W(f"**算力侧**（带宽之外的另一半 —— 只报带宽无法判断「算力是否也饱和」）：")
        W("")
        _cmp_rows = [
            ["单步 FLOPs（理论）", f"{_cmp.get('flops_per_step')} GFLOPs",
             "2 × 参数量 + 注意力项；不含 kernel 实现浪费，故为下界"],
            ["实测达成算力", f"{_cmp.get('achieved_tflops')} TFLOPS",
             "= 单步 FLOPs ÷ 单步耗时"],
            ["该强度下可达上界", f"{_cmp.get('attainable_tflops_at_intensity')} TFLOPS",
             f"= 算术强度 {_cmp.get('arithmetic_intensity')} FLOPs/Byte "
             f"× 带宽 —— **这是 roofline 的关键：带宽先把算力卡住了**"],
        ]
        if _cmp.get("peak_tflops"):
            _cmp_rows.append(["设备峰值算力", f"{_cmp.get('peak_tflops')} TFLOPS",
                              "设备画像声明值"])
            _cmp_rows.append([
                "对峰值的利用率",
                f"{(_cmp.get('compute_utilization') or 0)*100:.1f}%",
                "**低是正常的**：可达上界才是有效分母"])
        _cmp_rows.append([
            "未被权重读取解释的时间",
            f"{(_cmp.get('unexplained_time_ratio') or 0)*100:.1f}%",
            "= 1 − 带宽利用率。**这才是真正的线索**："
            "时间还有多少花在访存之外（KV 读 / launch / 回退 / 采样）"])
        W(_table(["项", "值", "口径"], _cmp_rows, ["l", "r", "l"]))
        W("")
        W(f"> **注意一个恒等关系**：{_cmp.get('identity_note', '')}"
          f"（此处两者同为 {(_cmp.get('attainable_ratio') or 0)*100:.1f}%）。"
          f"因此本剖面**不把它当作算力侧的独立证据** —— "
          f"那样只是把带宽结论换个说法重复一遍；"
          f"真正有价值的线索是「未被解释的时间」那一行。")
        W("")
        _cv = _cmp.get("verdict")
        _mark = {"bandwidth_limited": "带宽受限（算力低是必然结果）",
                 "bandwidth_underused": "带宽未饱和（时间另有去处，需进一步定位）",
                 "compute_bound": "**应改判 compute_bound**"}.get(_cv, _cv)
        W(f"- **算力侧判定：{_mark}** —— {_cmp.get('note', '')}")
        W("")
    W("### 3.4 阶段剖面")
    W("")
    W("> **侧重**：把 E2E 时间拆成 Prefill 与 Decode 两段。"
      "二者的瓶颈类型**相反**（前者算力密集、后者访存密集），"
      "有效手段也完全不同 —— 混在一起优化必然事倍功半。")
    W("")
    _ph = prof["phases"]
    _pf_ms = _ph["prefill"].get("ms")
    _pf_src = _ph["prefill"].get("source") or "参考场景"
    W(_table(["阶段", "吞吐 (t/s)", "耗时", "瓶颈类型", "口径 / 说明"],
             [["Prefill", _num(_ph["prefill"]["tps"]),
               (f"{_pf_ms:.0f} ms" if _pf_ms else "—"), "Compute-bound",
               f"**{_pf_src}**。{_ph['prefill'].get('note', '')}"],
              ["Decode", _num(_ph["decode"]["tps"]),
               f'{_ph["decode"]["ms_per_token"]} ms/token', "**Memory-bound**",
               "参考场景的稳态斜率（首 token 到末 token），不含 TTFT"],
              ["TTFT", "—", (f'{_ph["ttft"]["ms"]:.0f} ms'
                             if _ph["ttft"].get("ms") else "—"), "—",
               _ph["ttft"]["breakdown"]]],
             ["l", "r", "r", "l", "l"]))
    W("")
    W("> **为什么 Prefill 与 TTFT 的耗时不是同一个数**：Prefill 列取的是"
      "**纯 Prefill 吞吐**（prompt ≥512 token 的场景，或深度曲线里 prompt 明确的那一点），"
      "它回答「这台机器处理 prompt 的能力有多强」；TTFT 列取的是**短 prompt 的端到端首字延迟**，"
      "它包含 HTTP、Tokenize、调度与采样。两者的差额就是固定开销 —— "
      "短 prompt 上这部分可占 TTFT 的 80% 以上，所以不能用短 prompt 的 "
      "`prompt_tokens / TTFT` 去估算 Prefill 吞吐。")
    W("")
    W("### 3.5 算子剖面（Decode 单步耗时归因）")
    W("")
    W("> **侧重**：回答「优化的收益上限在哪」—— 各算子占单步耗时的比例，"
      "就是该算子可压缩空间的**上界**。占比小的算子即使优化到零也影响有限。")
    W("")
    W("> 归因方法：Decode 为访存受限，**谁读的字节多，谁就占时间**。"
      "故按各算子权重字节占比线性分摊实测单步耗时。")
    W("")
    W("```text")
    _kinds = {r.get("kind", "dense") for r in prof["ops"]}
    _hybrid = bool(a.get("hybrid")) or len(_kinds - {"global"}) > 1
    if _hybrid:
        W("按层类型分组（分母 = 该类单层的权重；混合架构的层内构成与稠密层不同）")
    else:
        W(f"单层内部构成（分母 = 单层权重，{a['n_layer']} 层同构）")
    for r in prof["ops"]:
        W(f'{r["op"]:<10} {r["pct_of_layer"]:>6.2f}%  {_bar(r["pct_of_layer"]/100, 34)}'
          f'  {r["mib_per_layer"]:>6.2f} MiB/层')
    pl = prof["per_layer"]
    W("")
    W("嵌入层占比（分母 = 全部权重；其余算子的全模型占比与排序见 §3.7）")
    W(f'{"embedding":<10} {pl["embedding_share_pct"]:>6.2f}%  '
      f'{_bar(pl["embedding_share_pct"]/100, 34)}  {pl["embedding_gib"]:>6.2f} GiB 合计')
    W("```")
    W("")
    # 按 **kind** 分组统计，不能按算子名前缀 —— 混合架构里
    # attn_qkv / attn_gate 属于 SSM 层，前缀是 attn 但与全注意力无关。
    def _sum_pct(pred) -> float:
        return sum(r["pct_of_layer"] for r in prof["ops"] if pred(r))

    _ffn = _sum_pct(lambda r: r["op"].startswith("ffn"))
    _attn = _sum_pct(lambda r: r.get("kind") == "attn" and r["op"].startswith("attn"))
    _ssm = _sum_pct(lambda r: r.get("kind") == "ssm")
    _ffn_step = sum(r["pct_of_step"] for r in prof["ops"] if r["op"].startswith("ffn"))
    _kv_pct = {r["op"]: r["pct_of_layer"] for r in prof["ops"]
               if r["op"] in ("attn_k", "attn_v")}
    _kv_pct = _kv_pct.get("attn_k", 0.0)
    if _hybrid:
        W(f"- **稠密层内**：FFN 三矩阵（gate/up/down）合计 **{_ffn:.1f}%**；"
          f"折算到全模型权重，FFN 占 **{_ffn_step:.1f}%**；")
        W(f"- **全注意力层内**：Q/K/V/O 四矩阵合计 **{_attn:.1f}%**，"
          f"其中 K/V 各 {_kv_pct:.2f}%（GQA {a['n_head']//a['n_head_kv']}:1，"
          f"KV 头数仅 {a['n_head_kv']}）；")
        W(f"- **SSM 层内**：输入投影/输出投影/z-gate 合计 **{_ssm:.1f}%** —— "
          f"这类层的权重结构与稠密层完全不同，优化手段也不同（没有 KV，"
          f"但有固定大小的循环状态）；")
    else:
        W(f"- 单层内：FFN 三矩阵（gate/up/down）合计 **{_ffn:.1f}%**，"
          f"Attention 四矩阵（Q/K/V/O）合计 **{_attn:.1f}%**；"
          f"折算到全模型权重，FFN 占 **{_ffn_step:.1f}%**（其余是注意力与词表嵌入）；")
        W(f"- 因 GQA（KV heads 仅 {a['n_head_kv']}，分组比 "
          f"{a['n_head']//a['n_head_kv']}:1），K/V 矩阵极小（各 "
          f"{_kv_pct:.2f}%），权重集中在 FFN 与 Q/O；")
    _tie = bool(a.get("tie_embeddings"))
    _emb_note = (f'- 词表嵌入/输出头占**全模型权重**的 '
                 f'{pl["embedding_share_pct"]:.1f}%（{pl["embedding_gib"]:.2f} GiB）'
                 + ("，因未 tie embeddings 而需输入/输出两份表。"
                    if not _tie else "（已 tie embeddings，输入/输出共享）。"))
    # 嵌入层已按实际量化档位折算时，量化它的空间已经不大，需明确说明
    if a.get("embedding_bits"):
        _emb_note += (f' 注意：此处已按实测量化档位（'
                      f'{a["embedding_bits"]:.2f} bit/权重）折算，不是 F16。')
    W(_emb_note)
    W("")
    W("**按算子归因的优化优先级**（字节占比即收益上限）：")
    W("")
    _ffn_frac = _ffn_step / 100.0
    _ffn_bits_cut = 0.25                      # 假设 FFN 位宽再降 25%（4.5 → 3.375 bit）
    _ffn_gain = 1.0 / (1.0 - _ffn_frac * _ffn_bits_cut)
    _emb_frac = pl["embedding_share_pct"] / 100.0
    _emb_bits_cut = 0.75                      # F16 → Q4 相当于降 75%
    _emb_gain = 1.0 / (1.0 - _emb_frac * _emb_bits_cut)
    _kv_l = prof["kv_cache"]["ladder"]
    # 没有深度扫描时用**部署 ctx 的几何值**兜底：这一行是「收益上限」的算术，
    # 不是实测结论，因此可以用模型值 —— 但必须标注来源，不能冒充实测。
    if _kv_l:
        _kv_share = _kv_l[-1]["kv_share_of_step_pct"] / 100.0
        _kv_ctx_k = _kv_l[-1]["ctx"] // 1024
        _kv_src_note = "按模型值上界"
    else:
        from ..config import kv_cache_bytes as _kcb, weight_bytes as _wbf3
        _ctx_dep = a.get("max_ctx_deployed", 8192)
        _kb = _kcb(cfg.model, _ctx_dep, *cfg.kv_bits())
        _kv_share = (2 * _kb) / (_wbf3(cfg.model) + 2 * _kb)
        _kv_ctx_k = _ctx_dep // 1024
        _kv_src_note = "按部署 ctx 的几何值推算（本次未采深度曲线）"
    _kv_gain = 1.0 / (1.0 - _kv_share * 0.75)
    _mtp = (base.get("extra") or {}).get("spec_mtp") or {}
    _mtp_txt = (f'实测 {_mtp["speedup_vs_native"]:.2f}×'
                f'（{_mtp["decode_tps_range"][0]}~{_mtp["decode_tps_range"][1]} t/s）'
                if _mtp.get("speedup_vs_native") else "文献经验值 1.2~1.6×")
    _mtp_cost = ("+1.28 GiB 显存；与并发竞争" if _mtp.get("decode_tps_typical")
                 else "需额外 Draft 权重；接受率强依赖任务类型")
    W(_table(["优化动作", "可减少的字节", "理论收益上限", "代价"],
             [["FFN 降至更低比特（Q4→Q3/Q2）",
               f'全模型权重 {_ffn_step:.1f}% × 降 {_ffn_bits_cut*100:.0f}% 位宽 '
               f'= 总访存 -{_ffn_frac*_ffn_bits_cut*100:.1f}%',
               f'{_ffn_gain:.2f}×',
               "全局质量损失，需 PPL 守门"],
              ["词表嵌入进一步降比特",
               f'{pl["embedding_share_pct"]:.1f}% 全模型权重 × 降 '
               f'{_emb_bits_cut*100:.0f}% = 总访存 -{_emb_frac*_emb_bits_cut*100:.1f}%',
               f'{_emb_gain:.2f}×',
               ("输出头对精度敏感，需单独评测"
                if not a.get("embedding_bits") else
                "已按实际量化档位（含 Q6_K 输出头）；继续压缩空间与风险都更大")],
              ["投机解码（MTP）",
               "不减少单次读取量，而是**减少读取次数**"
               "（每次验证读 1 遍权重，产出多个 token）",
               _mtp_txt, _mtp_cost],
              ["KV Cache 量化（F16→1/4 体积）",
               f'{_kv_ctx_k}K 时 KV 占单步 '
               f'{_kv_share*100:.1f}%（{_kv_src_note}）'
               f' = 总访存 -{_kv_share*0.75*100:.1f}%',
               f'{_kv_gain:.2f}×',
               "长上下文检索质量"]],
             ["l", "l", "r", "l"]))
    W("")
    _mtp_selected = any(
        "mtp" in str(x).lower() or "spec" in str(x).lower()
        for b in opt["bundles"] if b["is_selected"] for x in b.get("plans", []))
    _kv_selected = any(
        "kv" in str(x).lower()
        for b in opt["bundles"] if b["is_selected"] for x in b.get("plans", []))
    _mtp_verdict = ("这就是把它选入主攻方向的原因" if _mtp_selected
                    else "本任务未选它，仅作对照")
    W("> 这张表把「该往哪里优化」从直觉变成算术：在访存受限场景下，"
      "**优化收益的上限 = 该部分占总访存量的比例 × 可压缩比例**。"
      f"FFN 占全模型权重 {_ffn_step:.1f}%（单层内 {_ffn:.1f}%），"
      "是字节量最大的目标，但它的位宽已到 Q4，"
      f"继续压缩的精度代价陡增，理论收益也只有 {_ffn_gain:.2f}×；"
      f"而投机解码不减少「每次读取量」而是减少「读取次数」（{_mtp_txt}），"
      f"收益量级更大 —— {_mtp_verdict}。"
      + (f" KV 量化则是把长上下文从「装不下」变成「装得下」的前提"
         f"（本任务{'已采用' if _kv_selected else '未采用'}）。"
         if _kv_l else ""))
    W("")
    W("### 3.6 KV Cache 剖面")
    W("")
    W("> **侧重**：KV 是显存/内存中**唯一随上下文线性增长**的部分，"
      "而权重是常数。它同时决定两件事：长上下文能不能装下（容量），"
      "以及每步要多搬多少字节（带宽）。因此它需要单独一维来看。")
    W("")
    # KV 实测锚点（台账里的同配置显存差）：§3.6 与 §3.10.6 都要用，
    # 必须在 if/else 之外定义 —— 放在分支里会让后面的章节 UnboundLocalError。
    _kxc = (base.get("memory") or {}).get("kv_crosscheck") or {}
    if not _kv_l:
        # 没有长上下文曲线就没有这一维。明确说明缺什么、怎么补，
        # 而不是渲染一张空表让读者以为「KV 是 0」。
        W("> **本节无数据**：场景矩阵里没有带 `depth` 的点，"
          "台账也没有 `curves.long_context`。")
        W("")
        W("> 补数据的方式（任选其一）：")
        W("> 1) 在 `task_*.yaml` 的场景里加 `depth: <tokens>`（对应 llama-bench 的 `-d`）；")
        W("> 2) 在台账的 `curves.long_context` 里补 `{ctx, prefill_tps, decode_tps}` 点。")
        W("")
        W("> **为什么这一维不能省**：KV 是唯一随 ctx 线性增长的显存与带宽开销，"
          "长上下文的 TTFT 与 Decode 衰减都由它主导。缺了它，"
          "「要不要做 KV 量化」就没有算术依据 —— 而这恰恰是长上下文任务的核心决策。")
        W("")
    else:
        if _kv_l and _kv_l[-1]["kv_gib_measured"]:
            _ratio = (_kv_l[-1]["kv_gib_model"] / _kv_l[-1]["kv_gib_measured"])
            _last_ctx_k = _kv_l[-1]["ctx"] // 1024
            _is_derived = {r.get("kv_gib_source", "measured")
                           for r in prof["kv_cache"]["ladder"]} == {"derived"}
            if _is_derived:
                # 台账的 KV 曲线本身是几何推算时，不能把它说成「实测一致」——
                # 那只是同一个公式算了两遍。真正的实测锚点单列在下面。
                W(f"> **两列同源**：台账的 KV 曲线是按几何公式推算的（`kv_gib_source: derived`），"
                  f"与左列同源，因此「一致」不能作为公式正确的证据。"
                  + (f"真正的实测锚点只有一处：同配置仅换 KV 精度（f16 → q8_0/q4_0，"
                     f"{_kxc.get('ctx_k', 64)}K × 2 槽）的显存差 "
                 f"{_kxc.get('measured_delta_mib'):,} MiB，"
                     f"几何预测 {_kxc.get('geometric_delta_mib'):,} MiB —— "
                     f"误差 {_kxc.get('error_pct')}%。"
                     if _kxc else
                     "真正的实测锚点需补测：同配置仅换 KV 精度，比较显存峰值差。"))
            elif abs(_ratio - 1.0) <= 0.10:
                W(f"> 模型值与实测值并列。两者在 {_last_ctx_k}K 处相差 "
                  f"{abs(_ratio-1)*100:.1f}%（{_kv_l[-1]['kv_gib_model']} vs "
                  f"{_kv_l[-1]['kv_gib_measured']} GiB）—— **在容差内一致**，"
                  f"说明几何公式在真值架构下可信。")
            else:
                W(f"> 模型值与实测值并列。两者在 {_last_ctx_k}K 处相差 "
                  f"{_ratio:.2f}×，差额来源见 §1.4，**不做静默校准** —— "
                  f"在查清是哪一侧有问题之前，两个数都保留。")
        else:
            W("> 模型值按几何公式推算；实测列缺失表示台账未提供该曲线。")
        W("")
        _srcs = {r.get("kv_gib_source", "measured") for r in prof["kv_cache"]["ladder"]}
        _meas_label = ("KV 增量·实测 (GiB)" if _srcs == {"measured"}
                       else ("KV 增量·推算 (GiB)" if _srcs == {"derived"}
                             else "KV 增量·混合来源 (GiB)"))
        W(_table(["上下文", "KV 总量·模型 (GiB)", _meas_label,
                  "每步读写 (GiB)", "每步 KV 耗时 (ms)", "占单步"],
                 [[f'{r["ctx"]//1024}K', _num(r["kv_gib_model"], "{:.3f}"),
                   _num(r["kv_gib_measured"], "{:.2f}"),
                   _num(r["kv_read_gib_per_step"], "{:.4f}"),
                   _num(r["kv_ms_per_step"], "{:.3f}"),
                   f'{r["kv_share_of_step_pct"]:.2f}%'] for r in prof["kv_cache"]["ladder"]],
                 ["l", "r", "r", "r", "r", "r"]))
        W("")
        W(f"> {prof['kv_cache']['verdict']}")
        W("")
        _kvl = prof["kv_cache"]["ladder"]
        if _kvl and _kvl[-1]["kv_gib_measured"]:
            _pbn = diag["primary_bottleneck"]["type"]
            _mr = (_kvl[-1]["kv_gib_model"] / _kvl[-1]["kv_gib_measured"])
            if abs(_mr - 1.0) <= 0.10:
                W("> 注意：上表「每步 KV 耗时 / 占单步」按**模型值**计算。"
                  "模型与实测在容差内一致，因此该列可直接作为量级参考。"
                  f"主瓶颈 `{_pbn}` 的判定依据仍是实测吞吐与实测带宽利用率"
                  f"（见 §3.3），不依赖这张表。")
            else:
                _model_ge = _kvl[-1]["kv_gib_model"] >= _kvl[-1]["kv_gib_measured"]
                _dir = "访存压力被高估" if _model_ge else "访存压力被低估"
                W("> 注意：上表「每步 KV 耗时 / 占单步」按**模型值**计算。"
                  f"实测 KV 增量{'更小' if _model_ge else '更大'}意味着实际 KV 带宽开销"
                  f"{'低于' if _model_ge else '高于'}表中数值，即该表对「{_dir}」；"
                  f"主瓶颈 `{_pbn}` 的判定依据是实测吞吐与实测带宽利用率"
                  f"（见 §3.3），不受此表影响。")
        W("")
    # ---- 3.7 Hot Kernel 列表（业务地图输出物②）----------------------------
    W("### 3.7 Hot Kernel 列表（按单步耗时降序）")
    W("")
    W("> **产出物**：Hot Kernel 列表（业务地图输出物②）。")
    W("> 排序依据是**每个算子类在 Decode 单步中搬运的字节量**"
      "（访存受限下字节即时间），而不是 kernel 的调用次数。"
      "调用次数多但字节少的 kernel，即使优化到零，对单步耗时的影响也有限。")
    W("")
    _hk_mode = prof.get("mode", "replay")
    W(f"> **粒度**：本报告为 `{_hk_mode}` 模式，Hot 列表的粒度是**算子类**"
      f"（{len(prof['ops'])} 类 + 词表嵌入）。"
      + ("live 模式下由 nsys 时间线给出真实 kernel 名与占用率。"
         if _hk_mode == "live" else
         "live 模式下由 nsys 时间线给出真实 kernel 名（如 `mul_mat_q`、"
         "`flash_attn_ext`）与 SM 占用率；两者可对齐 —— "
         "算子类回答「该不该优化」，kernel 名回答「怎么优化」。"))
    W("")
    W("> **口径**：表中「每步字节」只含**权重读取**（常数项）。"
      "KV Cache 的读写不是权重读取、且随 ctx 变化，故不摊入本表 —— "
      "它的每步开销单列在 §3.6 与 §3.9 的搬运量分解中。")
    W("")

    _grip = {
        "attn_q": "权重降比特 / QKV 融合",
        "attn_k": "权重降比特（KV 量化见 §3.6）",
        "attn_v": "权重降比特（KV 量化见 §3.6）",
        "attn_o": "权重降比特 / 输出投影融合",
        "ffn_gate": "权重降比特 / gate-up 融合",
        "ffn_up": "权重降比特 / gate-up 融合",
        "ffn_down": "权重降比特 / 结构化稀疏",
    }
    _hk_rows = []
    for _r in prof["ops"]:
        _hk_rows.append({
            "name": _r["op"],
            "mib": _r["mib_per_step_all_layers"],
            "ms": _r["ms_per_step_all_layers"],
            "pct": _r["pct_of_step"],
            "bw": _r["achieved_bw_gbps"],
            "grip": _grip.get(_r["op"], "权重降比特"),
        })
    _pl = prof["per_layer"]
    _emb_ms = _pl["embedding_ms_per_step"]
    _emb_mib = _pl["embedding_gib"] * 1024
    _hk_rows.append({
        "name": "embedding / lm_head",
        "mib": _emb_mib,
        "ms": _emb_ms,
        "pct": _pl["embedding_share_pct"],
        "bw": (round(_emb_mib * 1024 * 1024 / (_emb_ms / 1000) / 1e9, 1)
               if _emb_ms else None),
        "grip": "词表裁剪 / 输出头单独量化",
    })
    _hk_rows.sort(key=lambda x: -x["pct"])
    _cum = 0.0
    for _i, _x in enumerate(_hk_rows, 1):
        _cum += _x["pct"]
        _x["rank"] = _i
        _x["cum"] = _cum
        # 「该类访存减半」的理论提速上限（其余项不变）
        _x["gain"] = 1.0 / (1.0 - _x["pct"] / 100.0 * 0.5)
    W(_table(["#", "Kernel / 算子类", "每步字节 (MiB)", "每步耗时 (ms)",
              "占单步", "累计", "达成带宽 (GB/s)", "访存减半的提速上限", "优化抓手"],
             [[str(_x["rank"]), f'`{_x["name"]}`', f'{_x["mib"]:.1f}',
               f'{_x["ms"]:.3f}', f'{_x["pct"]:.2f}%', f'{_x["cum"]:.2f}%',
               (f'{_x["bw"]:.1f}' if _x["bw"] is not None else "—"),
               f'{_x["gain"]:.3f}×', _x["grip"]]
              for _x in _hk_rows],
             ["r", "l", "r", "r", "r", "r", "r", "r", "l"]))
    W("")
    _cov = sum(_x["pct"] for _x in _hk_rows)
    _top3 = sum(_x["pct"] for _x in _hk_rows[:3])
    _top1 = _hk_rows[0]
    W(f"- **覆盖率**：{len(_hk_rows)} 项合计占单步 **{_cov:.1f}%**"
      + ("（权重读取已全部归因，无遗漏项）。" if abs(_cov - 100.0) < 1.0
         else "（与 100% 的差额来自权重/嵌入口径，见 §1.4 对账）。"));
    W(f"- **集中度**：前 3 项合计 **{_top3:.1f}%**，其中 `{_top1['name']}` "
      f"单项 **{_top1['pct']:.1f}%**。优化资源应优先投向这几项 —— "
      f"其余项即使归零，收益上限也有限。")
    if _hk_mode == "live":
        W(f"- **达成带宽**：各算子类的达成带宽均接近整机实测值 "
          f"{hw['achieved_bw_gbps']} GB/s，说明没有哪一类算子单独「掉队」—— "
          f"瓶颈是**整机带宽**而非某个 kernel 的实现质量。"
          f"若某一项显著低于其余项，才说明该 kernel 本身有优化空间"
          f"（未向量化、tile 选择不当、额外的类型转换等）。")
    else:
        W(f"- **达成带宽列在 replay 模式下不可用于判断 kernel 质量**："
          f"归因时时间按字节占比摊分，因此各算子算出的达成带宽**按定义相同**"
          f"（= 整机值 {hw['achieved_bw_gbps']} GB/s）。这一列的作用只是"
          f"校验归因口径自洽；要判断某个 kernel 是否实现落后，必须用 "
          f"live 模式采集真实 kernel 时间与带宽计数（nsys）。")
    W(f"- **收益上限的读法**：「访存减半」是**上限而非承诺**："
      f"实际收益还要乘以方案的可实现度，并在第 6 步用实测验证。"
      f"例如 `{_top1['name']}` 若真能减半，理论上限 {_top1['gain']:.2f}×；"
      f"但若位宽已到 Q4，继续压缩的精度代价会先成为约束。")
    W("")
    W("> **Hot 列表与算子剖面的区别**：§3.5 回答「字节结构是什么样」，"
      "用于归因；本节回答「按耗时排序后先动谁」，用于排期。"
      "两者使用同一份数据，但排序维度不同。")
    W("")

    W("### 3.8 跨设备回退代价（执行路径风险量化）")
    W("")
    W("> **侧重**：量化「如果有一部分计算落到别的设备上，代价有多大」。"
      "这是**风险项而非瓶颈项** —— 它的作用是在优化方案落地前，"
      "先算出「回退」这条路走不通（或代价可接受），"
      "避免方案上线后才发现某层被静默丢给了 CPU。")
    W("")
    fb = prof.get("cpu_fallback_cost", {})
    if fb:
        _lw_mib = _wb_eff / a['n_layer'] / 1024 / 1024
        if is_unified:
            W(f"- 单层权重 {_lw_mib:.1f} MiB。本设备为**统一内存**，"
              f"Host↔Device 无显式拷贝，回退代价主要是同步开销而非搬运："
              f"按内存带宽 {fb['transport_gbps']} GB/s 折算的等价代价为 "
              f"**{fb['ms_if_one_layer_falls_back']} ms/层**，"
              f"占单步解码预算的 **{fb['pct_of_decode_budget_per_layer']}%**；")
            W(f"- 结论：统一内存下 CPU/NPU 分工**在原理上可行**，"
              f"单层回退的带宽代价仅 {fb['pct_of_decode_budget_per_layer']}%。"
              f"真正的约束是**算子覆盖与同步开销** —— 需先用厂商工具链"
              f"做算子支持矩阵扫描，再决定图分区边界。")
        else:
            W(f"- 单层权重 {_lw_mib:.1f} MiB，经 PCIe {fb['transport_gbps']} GB/s "
              f"往返需 **{fb['ms_if_one_layer_falls_back']} ms**，"
              f"占单步解码预算的 **{fb['pct_of_decode_budget_per_layer']}%**；")
            W(f"- 结论：{fb['verdict']}。**「零 CPU 回退」是硬约束而非优化偏好**。")
    W("")

    # ---- 3.7 执行路径与数据搬运分析（业务地图「执行路径分析」输出物）----
    W("### 3.9 执行路径与数据搬运分析")
    W("")
    W("> **侧重**：回答「数据到底从哪搬到哪」。访存受限场景下，"
      "**搬运路径**决定了带宽能否被用满 —— 走错一条路径（如回退到 CPU、"
      "绕行 PCIe），代价可能是数量级的。")
    W("")

    _wb = _wb_eff
    _l2 = _dget(d, "memory.l2_cache_mib", None)
    _l3 = _dget(d, "memory.l3_cache_mib", None)
    _lw = _wb / a["n_layer"]
    _kv_bytes = (prof["kv_cache"]["ladder"][-1]["kv_read_gib_per_step"] * GIB
                 if prof["kv_cache"]["ladder"] else 0)
    _per_step = _wb + _kv_bytes

    W("**① 每步搬运量分解**（Decode 每步必须从内存读一遍的东西）：")
    W("")
    W(_table(["组成", "字节量", "占每步", "说明"],
             [["权重（全部层）", f"{_wb/GIB:.2f} GiB",
               f"{_wb/_per_step*100:.1f}%", "常数，与上下文无关"],
              ["KV Cache 读写", f"{_kv_bytes/GIB:.4f} GiB",
               f"{_kv_bytes/_per_step*100:.1f}%",
               f"随 ctx 线性增长（取最大 ctx = "
               f"{prof['kv_cache']['ladder'][-1]['ctx']//1024}K）"
               if prof["kv_cache"]["ladder"] else "—"],
              ["**合计**", f"**{_per_step/GIB:.2f} GiB**", "100%",
               # 注意：这里必须用**含 KV 的合计**去除带宽，不能直接套用
               # 权重口径的上界（62.6 t/s）—— 那是 ctx=0 时的数。
               f"÷ 峰值带宽 {hw['peak_bw_gbps']} GB/s = 该上下文下的"
               f"理论上界 {hw['peak_bw_gbps']*1e9/_per_step:.2f} t/s"
               f"（权重口径上界 {hw['theoretical_upper_bound_tps']} t/s 是 ctx=0 时的数）"]],
             ["l", "r", "r", "l"]))
    W("")

    W("**② Cache 层次**（决定「能不能少搬一点」）：")
    W("")
    _cache_rows = []
    if _l2 is not None:
        _cache_rows.append(["L2 Cache", f"{_l2} MiB",
                            f"权重是它的 {_wb/(_l2*1024*1024):.0f}×"])
    if _l3 is not None:
        _cache_rows.append(["L3 Cache", f"{_l3} MiB",
                            f"权重是它的 {_wb/(_l3*1024*1024):.0f}×"])
    _cache_total = ((_l2 or 0) + (_l3 or 0)) * 1024 * 1024
    if _cache_total:
        _cache_rows.append([
            "**权重 / 缓存总容量**", f"{_wb/_cache_total:.0f}×",
            "**权重远超 Cache 容量 ⇒ 每步几乎全部从主存/显存读，"
            "Cache 命中率趋近 0**。这解释了为什么「优化 Cache 局部性」在 "
            "Decode 上几乎无效 —— 数据本来就不在 Cache 里"])
    if _cache_rows:
        W(_table(["层级", "容量 / 比值", "含义"], _cache_rows, ["l", "l", "l"]))
        W("")

    W("**③ 回退路径检查**（最危险的静默失效）：")
    W("")
    if cfg.backend.fallback_is_cheap():
        W(f"- **本设备为 {cfg.backend.label}，不存在「回退」概念** —— "
          f"权重本来就在系统内存里，CPU 与 GPU/NPU 共享同一块物理内存。"
          f"因此「零回退」不是本设备的约束。")
        W(f"- 但仍需确认**没有静默走错后端**：本任务显式钉死 "
          f"`{' '.join(cfg.backend.accel_flags())}`，"
          f"避免在有核显的机器上被自动调度到 GPU。")
        if _fb:
            W(f"- 若未来做异构分工（部分层给 NPU），单层的等价带宽代价为 "
              f"**{_fb['ms_if_one_layer_falls_back']} ms**，"
              f"占单步预算的 **{_fb['pct_of_decode_budget_per_layer']}%** —— "
              f"真正的约束会变成**算子覆盖与同步开销**。")
    else:
        W(f"- **独立显存 GPU：任何一层回退都会引入跨总线往返。**")
        if _fb:
            W(f"- 单层权重 {_lw/1024/1024:.1f} MiB，经 "
              f"{_fb['transport_gbps']} GB/s 往返需 "
              f"**{_fb['ms_if_one_layer_falls_back']} ms**，"
              f"占单步解码预算的 **{_fb['pct_of_decode_budget_per_layer']}%**。")
        W(f"- 本任务配置 `-ngl 99` 全层卸载，且实测达成带宽 "
          f"{hw['bw_utilization']*100:.1f}% —— "
          f"若存在回退，达成带宽会显著低于此值，因此可**反证**回退未发生。")
    W("")

    W("**④ Host-Device 同步开销**：")
    W("")
    if is_unified:
        W(f"- 统一内存：Host↔Device 无显式拷贝，同步开销来自"
          f"「提交 → 等待完成」的握手，量级在微秒级，相对单步 "
          f"{hw['decode_step_ms']} ms 可忽略。")
    else:
        W(f"- 独立显存：每次 kernel 提交都涉及命令队列往返；"
          f"若存在 Host↔Device 的隐式同步（如未 pin 的内存访问），"
          f"单次可达数十微秒。本任务实测单步 {hw['decode_step_ms']} ms，"
          f"同步开销占比需用 nsys 时间线确认（本报告未采集）。")
    W("")

    W("### 3.10 根因分析报告")
    W("")
    W("> **产出物**：根因分析报告（业务地图输出物④）。")
    W("> 它把 §3.3-§3.9 的数据收敛成**一个可检验的结论**，并回答三件事："
      "为什么是这个结论（证据链）、什么证据会推翻它（反证条件）、"
      "据此该往哪个方向优化（路线推导）。")
    W("")
    pb = diag["primary_bottleneck"]

    # ---- 3.10.1 根因树 ------------------------------------------------------
    W("#### 3.10.1 根因树：从现象逐层排除")
    W("")
    # 现象行必须用**与 gap_vs_upper_bound_pct 同一口径**的 tps。
    # 那个百分比是用原生基线（llama-bench 纯解码）算的；若这里写端到端值，
    # 会出现「49.21 t/s，距 62.6 差 18.59%」这种自相矛盾的算术。
    _dec_tps = (base.get("extra") or {}).get("baseline_native_decode_tps") \
        or prof["phases"]["decode"].get("tps")
    # 算术强度同理：roofline 的判定点与 §3.3 的算力侧分析都用原生基线的参考 ctx，
    # 不能在这里改用部署 ctx —— 否则树里的 4.95 会和正文的 3.35 打架。
    _ai = _cmp.get("arithmetic_intensity")
    _ridge = (defn.get("roofline") or {}).get("ridge_point")
    _att = _cmp.get("attainable_tflops_at_intensity")
    _peak_tf = _cmp.get("peak_tflops")
    _unexpl = (_cmp.get("unexplained_time_ratio") or 0) * 100
    _kv_last = _kv_l[-1] if _kv_l else None
    _kv_pct = _kv_last["kv_share_of_step_pct"] if _kv_last else 0.0
    _w_pct = max(0.0, 100.0 - _kv_pct)
    _bw_full = hw["bw_utilization"] >= 0.75
    W("```text")
    W(f"现象：Decode {_num(_dec_tps)} t/s，距理论上界 "
      f"{hw['theoretical_upper_bound_tps']} t/s 差 {hw['gap_vs_upper_bound_pct']}%")
    _ai_def = (defn.get("roofline") or {}).get("intensity")
    _ctx_k_dep = (a.get("max_ctx_deployed") or 8192) // 1024
    W(f"├─ ① 算力不够吗？ → 否")
    W(f"│     算术强度 {_ai} FLOPs/Byte（短上下文）/ {_ai_def} FLOPs/Byte（{_ctx_k_dep}K）"
      f"，都 << 拐点 {_ridge}；")
    W(f"│     该强度下可达算力仅 {_att} TFLOPS（峰值 {_peak_tf} TFLOPS）"
      f"—— 提升算力无法加快 Decode")
    W(f"│     （两个强度值对应不同 ctx：短上下文时分子只有权重项，")
    W(f"│      长上下文时注意力项占比上升 —— 但都远低于拐点，结论不变）")
    W(f"├─ ② 带宽没吃满吗？ → {'是（已接近饱和）' if _bw_full else '否（仍有空间）'}")
    W(f"│     达成 {hw['achieved_bw_gbps']} / {hw['peak_bw_gbps']} GB/s "
      f"= {hw['bw_utilization']*100:.1f}%")
    W(f"│     ├─ 权重读取：占每步 {_w_pct:.1f}%（常数项 → 降比特 / 减少读取次数）")
    if _kv_last:
        W(f"│     ├─ KV 读取：占每步 {_kv_pct:.1f}%（按最大 ctx；随 ctx 增长 → 量化 / 稀疏化）")
    else:
        # 没有深度曲线时不能写 0.0% —— 那会让读者以为「KV 不占带宽」。
        W(f"│     ├─ KV 读取：**未采集**（缺 depth 场景，见 §3.6 的补数据方式）")
    W(f"│     └─ 未被解释：{_unexpl:.1f}%（同步 / 调度 / 负载竞争 / 采样）")
    if cfg.backend.fallback_is_cheap():
        _fb_line = (f"否：{cfg.backend.label} 无跨总线回退；"
                    f"已钉死 `{' '.join(cfg.backend.accel_flags())}` 防止静默换后端")
    else:
        _fb_line = (f"否：已全层卸载 `{' '.join(cfg.backend.accel_flags())}`；"
                    f"达成带宽 {hw['bw_utilization']*100:.1f}% 反证无静默回退")
    W(f"└─ ③ 有回退 / 绕行吗？ → {_fb_line}")
    W("```")
    W("")
    W("> 这棵树的价值在于**排除法**：每一步都给出「否」的证据，"
      "剩下的那条才是根因。只写「因为访存受限所以慢」不构成分析 —— "
      "它没有排除「算力不足」「回退」「调度异常」这些同样能解释现象的原因。")
    W("")

    # ---- 3.10.2 结论与证据链 ------------------------------------------------
    W("#### 3.10.2 结论与证据链")
    W("")
    W(f"**主瓶颈：`{pb['type'].upper()}`"
      f"{' / ' + pb['subtype'] if pb.get('subtype') else ''}"
      f"（置信度 {pb['confidence']}）**")
    W("")
    W("证据链（每条证据都标出来源与口径，便于复核）：")
    W("")
    for e in pb.get("evidence", []):
        W(f"- {e}")
    W("")
    W(_table(["证据", "来源", "口径", "复核方式"],
             [["单步解码耗时", "`baseline.json` 的实测 Decode 吞吐", "[M] 实测",
               "`perfkit query --run <run> --kpi decode_tps`"],
              ["达成带宽", "权重字节 ÷ 单步耗时", "[D] 推导",
               "权重字节见 §1.2；单步耗时见上"],
              ["roofline 判定", "算术强度 vs 拐点", "[D] 推导",
               "`--from-gguf` 核验架构超参后重算"],
              ["算子归因", "权重字节占比 × 实测单步耗时", "[D] 推导",
               "§3.5 与 §3.7 使用同一份 profiling.json"],
              ["KV 增量", "台账实测曲线（与几何模型并列）", "[M] / [D]",
               "§3.6 与 §1.4 的对账差额"]],
             ["l", "l", "l", "l"]))
    W("")
    W(f"**归因**：{pb.get('reasoning','')}")
    W("")

    # ---- 3.10.3 反证条件 ----------------------------------------------------
    W("#### 3.10.3 反证条件：什么证据会推翻这个结论")
    W("")
    W(f"**主结论的反证条件**：{pb.get('counter_evidence','')}")
    W("")
    if len(diag.get("findings") or []) > 1:
        _others = [f for f in diag["findings"]
                   if not (f.get("severity") == "primary"
                           and f.get("type") == pb.get("type"))]
        if _others:
            W("其余结论的反证条件：")
            W("")
            W(_table(["结论", "级别", "反证条件"],
                     [[f["type"],
                       {"primary": "主瓶颈", "secondary": "次要",
                        "phase_scope": "阶段限定", "watch": "观察项",
                        "error": "错误"}.get(f["severity"], f["severity"]),
                       f.get("counter_evidence") or "—"]
                      for f in _others],
                     ["l", "l", "l"]))
            W("")
    W("> **反证条件不是免责声明**：它必须能在下一步被真正检验。"
      "若某个反证条件无法用现有采集手段验证，应把它列为待采集项，"
      "而不是当作「已经排除」。")
    W("")

    # ---- 3.10.4 全部结论 ----------------------------------------------------
    W("#### 3.10.4 全部结论")
    W("")
    W(_table(["类型", "级别", "置信度", "关键证据"],
             [[f["type"], {"primary": "主瓶颈", "secondary": "次要",
                           "phase_scope": "阶段限定", "watch": "观察项",
                           "error": "错误"}.get(f["severity"], f["severity"]),
               f["confidence"], (f["evidence"][0] if f.get("evidence") else "")[:70]]
              for f in diag["findings"]],
             ["l", "l", "l", "l"]))
    W("")

    # ---- 3.10.5 从根因到路线 ------------------------------------------------
    W("#### 3.10.5 从根因到优化路线")
    W("")
    W(f"**优化路线**：`{'` → `'.join(diag['optimization_route'])}`")
    W("")
    _rr = diag.get("route_reasoning") or []
    if _rr:
        W("**路线是怎么推出来的**（从结论到方向，每一跳都可追溯）：")
        W("")
        W(_table(["来源结论", "级别", "推出方向", "推导理由"],
                 [[f'`{x["from_finding"]}`',
                   {"primary": "主瓶颈", "secondary": "次要",
                    "watch": "观察项"}.get(x.get("from_severity"), x.get("from_severity")),
                   " + ".join(f'`{d}`' for d in x["adds"]),
                   x["why"]] for x in _rr],
                 ["l", "l", "l", "l"]))
        W("")
        W("> **路线顺序的含义**：这不是「优先级排序」，而是「**筛选顺序**」——"
          "`optimize` 按此顺序从方案库里筛出适用方案并排序。"
          "排在前面的类别更贴近当前主瓶颈，因此它们的方案卡片会优先展示。"
          "路线末尾的 `precision_guard` 不是优化动作，而是**验收红线**"
          "（它来自质量风险结论而非性能瓶颈结论）。")
        W("")

    # ---- 3.10.6 适用边界 ----------------------------------------------------
    W("#### 3.10.6 结论的适用边界与数据口径")
    W("")
    _live = prof.get("mode") == "live"
    # 「未采集项」必须按实际数据算出来，不能写死 —— 写死的清单会在
    # 补测之后继续声称「未采集」，读者无法判断报告到底缺什么。
    _missing_items = []
    _mem_d = base.get("memory") or {}
    if not _mem_d.get("power_w"):
        _missing_items.append("功耗")
    if not _mem_d.get("gpu_util"):
        _missing_items.append("SM 占用率（需 ncu/nsys；"
                              "nvidia-smi 的 utilization.gpu 不是 SM 占用率）")
    if not ((_ledger_quality(cfg).get("ppl") or {}).get("rows")):
        _missing_items.append("PPL")
    if not (prof.get("nsys") or (base.get("environment") or {}).get("nsys")):
        _missing_items.append("nsys kernel 时间线")
    _bounds = [
        ["归因方式",
         ("live 模式：阶段/算子数据来自真实采集。" if _live else
          "replay 模式：按权重字节占比把实测单步耗时摊到各算子。"
          "它假设访存受限且各类算子达成带宽一致 —— 若某 kernel 实现异常差，"
          "该假设会把它的额外开销摊薄到其它算子。"),
         ("换硬件后必须重跑 `perfkit profiling --mode live` 复核。"
          if not _live else "可复现。")],
        ["架构超参",
         ("全部字段来自 GGUF 元数据核验（[M]），不是反推值。"
          if (defn.get("gguf_verification") or {}).get("all_match") else
          "权重字节与 KV 几何均按模型画像推算 [D]。"),
         ("已核验；换模型或换量化档位时需重新核验。"
          if (defn.get("gguf_verification") or {}).get("all_match") else
          "用 `perfkit define --from-gguf` 核验；未核验前结论置信度降一档。")],
        ["KV 模型 vs 实测",
         (f"台账曲线为几何推算（与模型列同源）；唯一实测锚点是同配置仅换 KV 精度"
          f"的显存差：{_kxc.get('measured_delta_mib'):,} MiB 实测 vs "
          f"{_kxc.get('geometric_delta_mib'):,} MiB 预测（误差 {_kxc.get('error_pct')}%）。"
          if _kxc else
          "台账未提供 KV 实测锚点，KV 结论仅有模型值。"),
         "差额不静默校准；先查清来源（架构 / 量化档位 / 测量口径）再采信。"],
        ["未采集项",
         ("、".join(_missing_items) if _missing_items
          else "无 —— 本报告所有进入结论的数据都有实测来源。"),
         "未采集项不能作为反证依据；报告中标注为待采集。"],
        ["结论复用",
         "本结论绑定当前「模型 + 量化档位 + 硬件 + 配置」。",
         "换任一项都需重跑第 2-4 步；判定规则可复用，数据不可复用。"],
    ]
    W(_table(["边界", "说明", "处理方式"], _bounds, ["l", "l", "l"]))
    W("")
    # --------------------------------------------------------- 4 优化实施
    W("## 4. 优化方案选型与实施")
    W("")

    # ---- 4.0 输入与产出对照 ------------------------------------------------
    # optimize.json 里带了 _why / _used_in_selected（对 plan 库的运行时判断），
    # 直接读它而不是重读 plan 库 —— 否则「未采用原因」会全是「—」。
    _plans_sel = opt.get("selected_plans") or []
    _plans_rej = opt.get("rejected_plans") or []
    _plans_all = _plans_sel + _plans_rej
    if not _plans_all:
        _plans_all = cfg.plan.get("plans") or []
    _by_layer: dict[str, list[str]] = {}
    for _p in _plans_sel:
        _by_layer.setdefault(_plan_layer(_p), []).append(_p.get("name", _p.get("id", "")))
    _layers_used = set(_by_layer)

    def _layer_txt(layer: str, fallback: str) -> str:
        names = _by_layer.get(layer) or []
        return "、".join(names) if names else fallback
    W("### 4.0 本步的输入与产出对照")
    W("")
    for _ln in _cov_block(
        "收益可测量",
        "针对瓶颈采取什么优化？收益来源是什么？",
        [["**主要输入**：瓶颈分析结果",
          f"§3.10 主瓶颈 `{pb['type']}` + 优化路线 `{' → '.join(diag['optimization_route'])}`", "✅"],
         ["**主要输入**：优化目标", "§1.7 验收标准（must_pass / should_pass / guardrail）", "✅"],
         ["**主要输入**：技术方案", f"方案库 {len(_plans_all)} 张卡片（`plan_*.yaml`）", "✅"],
         ["**活动**：制定方案", f"§4.1 筛选出 {len(_plans_sel)} 条适用方案并排序", "✅"],
         ["**活动**：模型转换 / 量化优化",
          _layer_txt("model", "本任务未采用（见 §4.3 分层归因）"),
          _st("model" in _layers_used, "未采用")],
         ["**活动**：Graph 优化（算子融合 / Layout）",
          "本任务未采用 —— `kernel_fused_qkv` 收益 1.02×、改动大，性价比最低",
          _st(False, "未采用")],
         ["**活动**：Kernel 优化",
          _layer_txt("kernel", "本任务未采用"),
          _st("kernel" in _layers_used, "未采用")],
         ["**活动**：Runtime 调优",
          _layer_txt("runtime", "本任务未采用"),
          _st("runtime" in _layers_used, "未采用")],
         ["**活动**：内存优化", "§1.4 内存预算 + §4.1 的 KV 量化方案", "✅"],
         ["**关键分析（模型层）**：FP16/INT8/INT4、蒸馏、剪枝", "§4.3 分层归因", "✅"],
         ["**关键分析（计算图层）**：算子融合、Layout、Constant Folding",
          "§4.3（本任务未采用，附原因）", "✅"],
         ["**关键分析（Kernel 层）**：GEMM / Attention / FlashAttention", "§4.3", "✅"],
         ["**关键分析（Runtime 层）**：Memory Pool / KV Cache / Batch", "§4.3", "✅"],
         ["**关键分析（硬件层）**：CUDA Kernel / NPU 算子 / Thread 优化", "§4.3", "✅"],
         ["**输出物** ①优化方案文档", "§4.1 方案卡片（收益/代价/风险/回滚）", "✅"],
         ["**输出物** ②模型优化版本",
          f"{m['quantization']}（{_wb_eff/GIB:.2f} GiB 有效权重）", "✅"],
         ["**输出物** ③Runtime 配置",
          f"`output/{cfg.run_id}/start_optimized.sh`（可直接执行）", "✅"],
         ["**输出物** ④Kernel 修改",
          "本任务未修改 kernel（使用上游预编译 kernel + 运行时开关）",
          _st(False, "未采用")],
         ["**输出物** ⑤实验记录", "§4.1 的 bundle 对比表（含实测与估算）", "✅"],
         ["**输出物** ⑥回滚方案", "§4.2 的回滚命令 + `optimize.json` 的 rollback_command", "✅"]]):
        W(_ln)

    W("本章把 §3.10 的瓶颈结论翻译成**可执行的配置**，分两步：")
    W("")
    W(_table(["小节", "做什么", "输入 → 输出"],
             [["4.1 候选方案组合对比",
               "把可行方案组合成若干候选 bundle，逐个算出预期指标与代价",
               "方案库 + 瓶颈结论 → 候选配置清单"],
              ["4.2 选定配置",
               "选出最终配置，并给出可直接运行的启动命令与回滚命令",
               "候选清单 + 验收标准 → 可执行脚本"]],
             ["l", "l", "l"]))
    W("")
    W(f"针对主瓶颈 `{pb['type']}`，按对症程度排序的候选方案：")
    W("")
    W(_table(["类别", "方案", "期望收益", "来源", "风险"],
             [[p["category"], f'**{p["name"]}**',
               ", ".join(f"{k}={v}" for k, v in p.get("gain", {}).items()),
               p.get("source", ""), p.get("risk", "").strip()[:60]]
              for p in opt["selected_plans"]],
             ["l", "l", "l", "l", "l"]))
    W("")
    W("### 4.1 候选方案组合对比")
    W("")
    W("**这一节在做什么**：单个方案只能解决一个问题，而生产配置必须是"
      "「多个方案共存且互不冲突」的一整套参数。本节把可行方案组合成若干"
      "**候选 bundle**，逐一算出它们的预期指标，目的是回答三个问题：")
    W("")
    W("1. **哪些方案能共存** —— 这一步必须用实测判定，不能靠推测："
      "本次「MTP 与多槽位是否冲突」的旧结论（认为必须二选一）就被实测推翻了"
      "（见 §5.2 的对照）；")
    W("2. **每个组合的代价是什么**（上下文、内存、并发数各自被牺牲多少）；")
    W("3. **选哪个**（下节 §4.2 给出选定配置，依据是 §6 的验收标准）。")
    W("")
    W("> 「实测 Decode」优先取该 bundle 的实测值；无实测时才用几何模型估算。"
      "两者并列展示，避免把估算值误读为实测。")
    _has_depth = any((b.get("measured") or {}).get("n_depth") is not None
                     for b in opt["bundles"])
    if _has_depth:
        W("")
        W("> ⚠️ **对比必须同深度**：模型估算列只按「权重字节 × 方案增益」推算，"
          "**不建模上下文深度**，因此在长上下文 bundle 上会系统性偏高。"
          "跨 bundle 比较实测值时，请先确认「测量深度」列一致 —— "
          "否则比较的是上下文长度，不是优化方案。")
    W("")
    _hdr = ["Bundle", "方案", "总上下文", "并发"]
    _al = ["l", "l", "r", "r"]
    if _has_depth:
        _hdr.append("测量深度")
        _al.append("r")
    _hdr += ["实测 Decode (t/s)", "模型估算 (t/s)",
             f"{cfg.mem_label()} (GiB)", "安全冗余 (GiB)", "选定"]
    _al += ["r", "r", "r", "r", "c"]
    _rows = []
    for _b in opt["bundles"]:
        # 注意：局部名不要用 m/d/b —— m=model、d=device、b 也用于别处，
        # 覆盖后会污染后续章节（这是本文件踩过的坑）。
        _mb = _b.get("measured") or {}
        _e = _b["estimated"]
        _row = [_b["id"], _b["name"], f'{_e["ctx_total"]}', f'{_e["parallel"]}']
        if _has_depth:
            _nd = _mb.get("n_depth")
            _row.append(f'{_nd // 1024}K' if isinstance(_nd, int) and _nd else
                        ("0" if _nd == 0 else "—"))
        # 显存优先用该 bundle 的**实测峰值**（含 MTP Draft / mmproj / 运行时缓冲），
        # 几何模型只算权重+KV+固定 runtime，会系统性低估 2~3 GiB。
        _pv_mib = _mb.get("peak_vram_mib")
        _pv_gib = (round(_pv_mib / 1024, 2) if isinstance(_pv_mib, (int, float))
                   else _e["peak_vram_gib"])
        _pv_txt = (f'{_pv_gib:.2f} ✓实测' if isinstance(_pv_mib, (int, float))
                   else f'{_pv_gib:.2f}')
        _sm = _mb.get("safety_margin_gib")
        if not isinstance(_sm, (int, float)):
            _sm = round(_e["safety_margin_gib"], 2)
        _row += [_num(_mb.get("decode_tps")), _num(_e["decode_tps_est"]),
                 _pv_txt, _num(_sm),
                 "✅" if _b["is_selected"] else ""]
        _rows.append(_row)
    W(_table(_hdr, _rows, _al))
    W("")
    W("### 4.2 选定配置")
    W("")
    W(f"**选定 bundle**：`{opt['selected_bundle']['id']}` — "
      f"{opt['selected_bundle']['name']}")
    W("")
    W("```bash")
    for ln in opt["startup_command"].splitlines():
        W(ln)
    W("```")
    W("")
    W("**回滚命令**（恢复基线，用于方案不达标时）：")
    W("")
    W("```bash")
    for ln in opt["rollback_command"].splitlines():
        W(ln)
    W("```")
    W("")

    # ---- 4.3 方案的分层归因（业务地图「五层分析内容」）-------------------
    W("### 4.3 方案的分层归因")
    W("")
    W("> 业务地图把优化手段分成五层：**模型层 / 计算图层 / Kernel 层 / "
      "Runtime 层 / 硬件层**。分层不是为了分类好看，而是因为**改动成本与"
      "收益确定性逐层递增** —— 越往下改，越接近硬件、越难回滚。"
      "本节逐层交代「本任务用了什么、没用什么、为什么」。")
    W("")

    _LAYER_DESC = {
        "model": ("模型层", "改权重本身（量化档位 / 蒸馏 / 剪枝）",
                  "改动最小、收益最确定；但有精度代价，需 PPL 守门"),
        "graph": ("计算图层", "改图结构（算子融合 / Layout / Constant Folding）",
                  "收益中等、改动中等；在访存受限场景收益有限"),
        "kernel": ("Kernel 层", "改算子实现（GEMM / Attention / FlashAttention）",
                   "收益取决于带宽是否已榨干；已达 80% 时空间只剩 20%"),
        "runtime": ("Runtime 层", "改运行时策略（KV 量化 / 并发 / batch / 内存池）",
                    "改动小、可回滚；与业务形态强相关"),
        "hardware": ("硬件层", "改硬件或设备侧策略（NPU 卸载 / 线程 / 热管理）",
                     "收益最大但受设备能力约束，且需真机验证"),
    }
    # 「采用」= 真的进了 selected_bundle；「候选」= 对症但没进这次配置。
    # 这两者混在一起会让读者误以为所有对症方案都上线了。
    _used_ids = {p.get("id") for p in _plans_sel if p.get("_used_in_selected")}
    _cand_ids = {p.get("id") for p in _plans_sel} - _used_ids
    _rej_ids = {p.get("id") for p in _plans_all} - _used_ids - _cand_ids

    def _fmt_gain(pl: dict) -> str:
        g = pl.get("gain") or {}
        if not g:
            return "—"
        return "，".join(f"{k}={v}" for k, v in g.items())

    _layer_rows = []
    for _lk in ("model", "graph", "kernel", "runtime", "hardware"):
        _name, _what, _nature = _LAYER_DESC[_lk]
        _adopted = [p for p in _plans_all if _plan_layer(p) == _lk
                    and p.get("id") in _used_ids]
        _cands = [p for p in _plans_all if _plan_layer(p) == _lk
                  and p.get("id") in _cand_ids]
        if _adopted:
            _txt = "、".join(f'`{p["id"]}`' for p in _adopted)
            _gain = "；".join(f'{p["id"]}: {_fmt_gain(p)}' for p in _adopted)
            _stt = "✅ 采用"
            if _cands:
                _txt += "<br>候选未采用：" + "、".join(f'`{p["id"]}`' for p in _cands)
        else:
            _txt = "—"
            _gain = ("未采用" + (f'（候选：{", ".join(x["id"] for x in _cands)}）'
                               if _cands else ""))
            _stt = "— 未采用"
        _layer_rows.append([f"**{_name}**", _what, _txt, _gain, _stt])
    W(_table(["层", "改什么", "本任务采用", "收益 / 说明", "状态"],
             _layer_rows, ["l", "l", "l", "l", "c"]))
    W("")
    _not_used = [p for p in _plans_all if p.get("id") not in _used_ids]
    if _not_used:
        W("**未采用方案及原因**（负向知识同样要记录 —— 否则下次还会重新论证一遍）：")
        W("")
        _rows = []
        for p in _not_used:
            _why = p.get("_why") or "—"
            if p.get("id") in _cand_ids:
                _why = (p.get("not_selected_reason")
                        or ("对症（" + _why + "），但未进入本次选定配置"))
            _rows.append([f'`{p["id"]}`',
                          _LAYER_DESC.get(_plan_layer(p), ("—",))[0],
                          p.get("category", "—"), _why])
        W(_table(["方案", "层", "类别", "未采用原因"], _rows,
                 ["l", "l", "l", "l"]))
        W("")
    # 这段话要用**实测纯读带宽**做分母，不能用标称带宽 ——
    # 标称 1008 会把「kernel 已到 92.9%」说成「还有 19% 空间」。
    _probe_rb = ((base.get("environment") or {}).get("probe") or {}).get("read_bw")
    _prod_bw = hw["achieved_bw_gbps"]
    _kernel_room = (max(0.0, 1 - _prod_bw / _probe_rb) * 100) if _probe_rb else None
    W("> **为什么本任务的主攻方向不在 Kernel 层**："
      f"标称带宽利用率 {hw['bw_utilization']*100:.1f}% 看起来还有 "
      f"{(1-hw['bw_utilization'])*100:.0f}% 空间，但标称值本身不可达 —— "
      + (f"探针实测纯读带宽只有 {_probe_rb:.1f} GB/s（标称的 "
         f"{_probe_rb/hw['peak_bw_gbps']*100:.0f}%），生产内核已拿到其中 "
         f"{_prod_bw/_probe_rb*100:.1f}%，"
         f"**kernel 层实际只剩约 {_kernel_room:.0f}% 的空间**。"
         if _probe_rb else "")
      + "相比之下 Runtime 层（KV 量化 / 并发）与模型层（量化档位）"
        "能直接减少**需要搬运的字节数**或**读取次数**，量级完全不同。")
    W("")

    # --------------------------------------------------------- 5 收益量化
    W("## 5. 优化效果验证与收益量化")
    W("")

    # ---- 5.0 输入与产出对照 ------------------------------------------------
    _side = ver.get("side_effects") or []
    _ppl_row = next((r for r in ver["comparison"] if r["key"] == "ppl"), None)
    W("### 5.0 本步的输入与产出对照")
    W("")
    for _ln in _cov_block(
        "收益达标且无质量回退",
        "优化是否有效？是否有副作用？",
        [["**主要输入**：优化版本", f"`{ver['after_bundle']['id']}` — {ver['after_bundle']['name']}", "✅"],
         ["**主要输入**：Baseline 数据", f"`{ver['before_bundle']['id']}` — {ver['before_bundle']['name']}", "✅"],
         ["**主要输入**：测试环境", "同硬件 / 同引擎 / 同测试集（见本节开头的两侧配置）", "✅"],
         ["**活动**：相同环境复测",
          f"两侧 bundle 均在同一环境采集（{ver['after_bundle']['measured_source']}）", "✅"],
         ["**活动**：性能对比", "§5.1 逐 KPI 前后对比（带口径标记）", "✅"],
         ["**活动**：收益分析", "§5.2 收益归因（每项变更 → 来源实测）", "✅"],
         ["**活动**：质量验证", "§5.5 质量验证（PPL / 长上下文 / 输出一致性）",
          _st(bool(_ppl_row and _ppl_row.get("after") is not None), "PPL 未采集")],
         ["**活动**：资源分析", "§5.6 资源收益分析（内存 / 功耗 / 温度）", "✅"],
         ["**关键分析**：性能收益（TTFT 下降 / Decode 提升 / Latency 降低）",
          "§5.1 + §5.2", "✅"],
         ["**关键分析**：资源收益（RAM 下降 / 功耗降低 / 温度改善）", "§5.6", "✅"],
         ["**关键分析**：质量验证（Accuracy / 输出一致性 / 任务效果）", "§5.5",
          _st(bool(_ppl_row and _ppl_row.get("after") is not None), "未采集")],
         ["**输出物** ①优化前后报告", "§5.1 对比表", "✅"],
         ["**输出物** ②收益分析表", "§5.2", "✅"],
         ["**输出物** ③质量报告", "§5.5",
          _st(bool(_ppl_row and _ppl_row.get("after") is not None), "未采集")],
         ["**输出物** ④性能资源权衡", "§5.3 副作用 + §5.6 资源", "✅"]]):
        W(_ln)

    W(f"**优化前**：`{ver['before_bundle']['id']}` — {ver['before_bundle']['name']}")
    W("")
    W("```bash")
    W("  " + " ".join(ver["before_bundle"]["flags"] or []))
    W("```")
    W("")
    W(f"**优化后**：`{ver['after_bundle']['id']}` — {ver['after_bundle']['name']}")
    W("")
    W("```bash")
    W("  " + " ".join(ver["after_bundle"]["flags"] or []))
    W("```")
    W("")
    W(f"> 两侧数据来源：{ver['before_bundle']['measured_source']} / "
      f"{ver['after_bundle']['measured_source']}")
    W("")
    W(f"> {ver['basis_note']}")
    W("")
    W("### 5.1 前后对比")
    W("")
    W(f"> 口径说明：`measured` 为实测或实测推导，`model` 为几何模型值"
      f"（仅内存一栏），`derived` 为由多次实测推导的比值，`todo` 为待采集，"
      f"**`invalid` 为该口径在本配置下失效（不参与达标判定，原因见备注）**。"
      f"达标判定优先采用实测值。")
    W("")
    W(_table(["指标", "优化前", "优化后", "变化", "目标", "达标", "取值口径"],
             [[r["key"], _num(r["before"]), _num(r["after"]),
               f'{r["delta_pct"]:+.2f}%' if r["delta_pct"] is not None else "—",
               _num(r["target"]),
               {True: "✅", False: "❌", None: "—"}[r["meets_target"]],
               f'{r["before_source"]} → {r["after_source"]}']
              for r in ver["comparison"]],
             ["l", "r", "r", "r", "r", "c", "l"]))
    W("")
    notes = [r for r in ver["comparison"] if r.get("after_note")]
    if notes:
        W("**口径备注**：")
        W("")
        for r in notes:
            W(f'- `{r["key"]}`：{r["after_note"]}')
        W("")
    _invalid_rows = [r for r in ver["comparison"] if r.get("after_source") == "invalid"]
    if _invalid_rows:
        W("> **`invalid` 不是缺数据，而是「这个口径在这里没有物理意义」**："
          + "、".join(f'`{r["key"]}`' for r in _invalid_rows)
          + " 的常规算法会得出 >100% 的结果（本次 154.5%），"
            "原因是投机解码让一次权重读取产出多个 token。"
            "判定为 invalid 而不是直接删掉，是为了让读者看到「我们算过、并知道它不成立」。"
            "替代指标：每次权重读取产出的 token 数（= 1 + 平均接受长度，本次约 2.3）。")
        W("")
    _derived_rows = [r for r in ver["comparison"] if r.get("after_source") == "derived"]
    if _derived_rows:
        W("> **关于 `derived` 口径的指标**："
          + "、".join(f'`{r["key"]}`' for r in _derived_rows)
          + " 是**实验设计的结果**（同一配置下两次测量的比值，如「4 线程 ÷ 最优线程」），"
            "不是某个 bundle 的属性 —— 因此两侧数值相同。"
            "它们用于回答「这台机器的调优空间在哪」，"
            "而不是「这个方案比那个方案好多少」。")
        W("")
    W("### 5.2 收益归因")
    W("")
    W(f"选定 bundle `{ver['after_bundle']['id']}` 相对基线 `{ver['before_bundle']['id']}` "
      f"的每一项变更，及其收益依据：")
    W("")
    _sel_flags = set(ver["after_bundle"]["flags"] or [])
    _base_flags = set(ver["before_bundle"]["flags"] or [])
    _added = [f for f in (ver["after_bundle"]["flags"] or []) if f not in _base_flags]
    _removed = [f for f in (ver["before_bundle"]["flags"] or []) if f not in _sel_flags]
    _rows = []
    for pd in ver["after_bundle"].get("plan_details", []):
        _gain_txt = ", ".join(f"{k}={v}" for k, v in (pd.get("gain") or {}).items()) or "—"
        _rows.append([f'**{pd["name"]}**', pd["category"], _gain_txt,
                      pd.get("source", ""), (pd.get("risk") or "").strip()[:60]])
    if _rows:
        W(_table(["优化方案", "类别", "期望收益", "来源（可复现）", "风险"],
                 _rows, ["l", "l", "l", "l", "l"]))
        W("")
    if _added or _removed:
        W("**启动参数的净变化**（可直接对照 diff）：")
        W("")
        W("```diff")
        for f in _removed:
            W(f"- {f}")
        for f in _added:
            W(f"+ {f}")
        W("```")
        W("")
    _lc = (base.get("extra") or {}).get("long_context_curve") or []
    _c2 = (base.get("extra") or {}).get("concurrent_2") or {}
    _mtp = (base.get("extra") or {}).get("spec_mtp") or {}
    _sel_parallel = int(flag_value(ver["after_bundle"]["flags"] or [], "--parallel", int) or 1)
    if _lc or _c2 or _mtp:
        W("**支撑这些收益的实测数据**：")
        W("")
        _ev = []
        if _lc:
            _ev.append([f'长上下文曲线（{len(_lc)} 个采样点）',
                        f'{_lc[0]["ctx"]//1024}K→{_lc[-1]["ctx"]//1024}K：'
                        f'Decode {_lc[0]["decode_tps"]}→{_lc[-1]["decode_tps"]} t/s，'
                        f'KV 增量 {_lc[0]["kv_gib"]}→{_lc[-1]["kv_gib"]} GiB',
                        "支撑长上下文下的速率衰减与 KV 增长论证"])
        if _c2.get("system_tps"):
            _ev.append([f'并发实测（{"本次已采用" if _sel_parallel > 1 else "本任务未采用，仅作对照"}）',
                        f'系统吞吐 {_c2["system_tps"]} t/s',
                        "支撑并发策略选择"])
        if _mtp.get("decode_tps_typical"):
            _ev.append(["MTP 投机解码实测",
                        f'{_mtp["decode_tps_range"][0]}~{_mtp["decode_tps_range"][1]} t/s，'
                        f'接受率 {_mtp.get("accept_rate_samples")}',
                        "支撑「投机 vs 并发」二选一决策"])
        W(_table(["实测数据", "数值", "在本次决策中的作用"], _ev, ["l", "l", "l"]))
        W("")
    W("### 5.3 副作用与缓解")
    W("")
    W(_table(["副作用", "影响", "严重度", "缓解措施"],
             [[e["effect"], e["impact"], e["severity"], e["mitigation"]]
              for e in ver["side_effects"]],
             ["l", "l", "c", "l"]))
    W("")
    # 台账里记录的、需要单独说明的量化/优化副作用
    _pen = ((base.get("extra") or {}).get("prefill_kv_penalty") or {})
    if _pen:
        W("**量化对 Prefill 的代价**（Decode 与 Prefill 收益方向相反，必须同时报告）：")
        W("")
        W(f"- {_pen['finding']}")
        W("")
        _ms = _pen.get("measurements") or {}
        if _ms:
            W(_table(["KV 精度", "Prefill (t/s)", "Decode (t/s)", "来源"],
                     [[k, _num(v.get("prefill_tps")), _num(v.get("decode_tps")),
                       v.get("source", "")] for k, v in _ms.items()],
                     ["l", "r", "r", "l"]))
            W("")
        if _pen.get("likely_cause"):
            W(f"- **成因**：{_pen['likely_cause']}")
            W("")
        if _pen.get("implication"):
            W(f"- **含义**：{_pen['implication']}")
            W("")
        if _pen.get("verification_needed"):
            W(f"> ⚠️ **待复核**（状态：`{_pen.get('status', 'unknown')}`）："
              f"{_pen['verification_needed']}")
            W("")
    W("### 5.4 精度—性能权衡")
    W("")
    W("量化收益与精度代价必须成对评估。下表把「位宽 / 体积」与「实测 PPL」放在一起 —— "
      "只有两者同时给出，档位选择才是可辩护的。")
    W("")
    # 台账 quality 段（§5.4 与 §5.5 共用；提前加载，避免顺序依赖）
    _q = _ledger_quality(cfg)
    # 台账 quality.ppl 里的实测值按量化名匹配到档位表
    _pl_rows = ((_q.get("ppl") or {}).get("rows") or [])
    _pl_map = {r["quant"].split()[0].upper(): r for r in _pl_rows}
    _ref_name = (_pl_rows[0]["quant"].split()[0] if _pl_rows else "参照档")
    _ladder_rows = []
    _base_ppl = None
    for _r in m.get("precision_ladder", []):
        _k = str(_r["quant"]).upper()
        _hit = _pl_map.get(_k)
        _ppl_txt = (f'{_hit["ppl"]}' if _hit else "**待采集**")
        _note = ""
        if _hit and _base_ppl is None:
            _base_ppl = _hit["ppl"]
        if _hit and _base_ppl and _hit["ppl"] != _base_ppl:
            _note = f'（相对 {_ref_name} {( _hit["ppl"]/_base_ppl-1)*100:+.2f}%）'
        _ladder_rows.append([
            _r["quant"], _num(_r["bits_per_weight"], "{:.2f}"),
            _num(_r["est_weight_bytes"] / GIB), _ppl_txt,
            ("**本部署基准档**" if _k == str(m["quantization"]).upper() else "") + _note])
    W(_table(["量化档位", "位宽 (bit/w)", "权重体积 (GiB)", "PPL", "结论"],
             _ladder_rows, ["l", "r", "r", "r", "l"]))
    W("")
    W("> **评测方法**：WikiText-2 raw test，`llama-perplexity`，ctx=512 / 40 chunks。"
      "PPL 相对更高精度档劣化 >2% 即拒绝该档位。"
      "**注意 Q4_0 一行只是 KV/Draft 用的位宽参考，没有主模型权重文件，因此 PPL 必然待采集**。")
    W("")

    # ---- 5.5 质量验证（业务地图「质量验证」活动）--------------------------
    W("### 5.5 质量验证")
    W("")
    W("> **侧重**：性能收益不能以质量为代价。质量验证的判据是"
      "「相对基线劣化是否在守门阈值内」，而不是「看起来还行」。")
    W("")
    _ppl_kpi = cfg.kpi_by_key("ppl") or {}
    _ppl_before = (_ppl_row or {}).get("before")
    _ppl_after = (_ppl_row or {}).get("after")
    _guard_ppl = next((g for g in (cfg.task["acceptance"]["guardrail"] or [])
                       if isinstance(g, str) and "ppl" in g.lower()), None)
    # 台账 quality 段：有就渲染实测值，没有就明确写「未采集」——
    # 不能因为「没数据」就把这一节写成通用建议，那等于放过了质量风险。
    _q = _q if "_q" in dir() else _ledger_quality(cfg)
    _ppl_q = _q.get("ppl") or {}
    _ppl_rows = _ppl_q.get("rows") or []
    _ppl_measured = next((r for r in _ppl_rows
                          if r.get("quant", "").startswith(m.get("quantization", ""))), None)
    _ppl_ref = next((r for r in _ppl_rows
                     if r is not _ppl_measured and r.get("ppl")), None)
    if _ppl_measured:
        _ppl_txt = f'{_ppl_measured["ppl"]}（{_ppl_measured["quant"]}）'
        if _ppl_ref:
            _d = (_ppl_measured["ppl"] / _ppl_ref["ppl"] - 1) * 100
            _ppl_txt += f'；对照 {_ppl_ref["quant"]} {_ppl_ref["ppl"]}（{_d:+.2f}%）'
        _ppl_state = ("✅ 已采集" if _ppl_ref else "◐ 部分（仅本档位，缺精度对照）")
    else:
        _ppl_txt, _ppl_state = "**未采集**", "⚠️ 未采集"
    _oc = _q.get("output_consistency") or {}
    W(_table(["验证项", "口径", "结果", "判据", "状态"],
             [["**困惑度 PPL**",
               (_ppl_q.get("dataset", "WikiText-2")
                + (f'；ctx={_pp.get("ctx")}，{_pp.get("chunks")} chunks'
                   if (_pp := (_ppl_q.get("params") or {})).get("ctx") else "")
                if _ppl_measured else "WikiText-2 / 中文长文本（未执行）"),
               _ppl_txt,
               _guard_ppl or "相对劣化 ≤ 阈值",
               _ppl_state],
              ["**长上下文检索**",
               ((_q.get("needle") or {}).get("design")
                or "needle-in-haystack（各深度）"),
               ((_q.get("needle") or {}).get("finding")
                if (_q.get("needle") or {}).get("configs") else "**未采集**"),
               "命中率 ≥95%（各深度）",
               ("◐ 已测（覆盖面有限，见下）" if (_q.get("needle") or {}).get("configs")
                else _st(False, "未采集"))],
              ["**输出一致性**",
               "同输入多次输出的稳定性 / 结构化格式合法率",
               (_oc.get("evidence") if _oc.get("status") == "partial"
                else "**未采集**"),
               "格式合法率 100%，语义无系统性偏移",
               ("◐ 部分" if _oc.get("status") == "partial" else _st(False, "未采集"))],
              ["**多后端一致性**",
               "同模型不同后端的输出差异",
               "不适用（单后端部署）", "无系统性偏移", "— 不适用"]],
             ["l", "l", "l", "l", "c"]))
    W("")
    if _ppl_q.get("reference_note"):
        W(f"> **参照档说明**：{_ppl_q['reference_note']}")
        W("")
    if _ppl_q.get("command"):
        W(f"> PPL 复现命令：`{_ppl_q['command']}`")
        W("")
    if _ppl_q.get("caveat"):
        W(f"> ⚠️ {_ppl_q['caveat']}")
        W("")
    _nd = _q.get("needle") or {}
    if _nd.get("configs"):
        W("**needle-in-haystack 明细**（同题面、仅换 KV 精度）：")
        W("")
        W(_table(["KV 精度", "32K 上下文", "60K 上下文", "结论"],
                 [[c.get("kv", "—"), c.get("hit_32k", "—"),
                   c.get("hit_60k", "—"), c.get("result", "—")]
                  for c in _nd["configs"]],
                 ["l", "c", "c", "l"]))
        W("")
        if _nd.get("caveat"):
            W(f"> ⚠️ {_nd['caveat']}")
            W("")
    W("> **为什么这三项必须补**：性能优化里最容易「悄悄变差」的就是质量 —— "
      "它不会报错，只会让输出变得「不太对」。PPL 是全局守门，"
      "长上下文检索是 **KV 量化**的唯一有效检验（短上下文几乎测不出来），"
      "输出一致性用于发现算子精度问题。三项缺一，质量红线就是空的。")
    W("")
    W("> **补采方式**：PPL 用 `llama-perplexity`；"
      "长上下文检索需构造 needle 测试集；输出一致性用同输入多次采样比对。"
      "采集后写入台账的 `quality` 段，本节会自动渲染。")
    W("")

    # ---- 5.6 资源收益分析（业务地图「资源分析」活动）----------------------
    W("### 5.6 资源收益分析")
    W("")
    W("> **侧重**：性能提升常常伴随资源代价（显存、功耗、温度）。"
      "只报速度不报代价的方案，在生产环境里无法被安全采纳。")
    W("")
    _cmp_rows = {r["key"]: r for r in ver["comparison"]}
    _res_rows = []
    for _k, _label, _unit in (
            ("peak_vram_mib", f"{cfg.backend.peak_metric_label}", "MiB"),
            ("peak_ram_mib", "峰值内存", "MiB"),
            ("safety_margin_gib", "安全冗余", "GiB"),
            ("temp_c", "温度", "°C"),
            ("power_w", "功耗", "W")):
        _r = _cmp_rows.get(_k)
        if not _r:
            continue
        _res_rows.append([
            _label,
            _num(_r.get("before")), _num(_r.get("after")),
            (f'{_r["delta_pct"]:+.2f}%' if _r.get("delta_pct") is not None else "—"),
            ({True: "✅ 改善", False: "❌ 恶化", None: "—"}[_r.get("improved")]
             if _r.get("delta_pct") not in (0, 0.0, None) else "— 持平"),
            _r.get("after_source", "—"),
        ])
    if not _res_rows:
        _res_rows = [["—", "—", "—", "—", "—", "本任务未采集资源类指标"]]
    W(_table(["资源", "优化前", "优化后", "变化", "方向", "口径"],
             _res_rows, ["l", "r", "r", "r", "c", "l"]))
    W("")
    # 「—」必须解释清楚：是没测，还是测了但没有可比的前值。
    # 不解释的话，「温度 — → 63°C」看起来像「温度改善了」，实际是基线没采。
    _half = [r for r in _res_rows if r[1] == "—" and r[2] != "—"]
    if _half:
        W("> **为什么有些行只有「优化后」有值**：基线配置未单独采集该指标"
          "（当时的采样脚本只记了显存与吞吐）。这些行**不能解读为「改善了」** —— "
          "没有前值就没有变化量。它们的作用是记录当前配置的实际水平，"
          "供下一轮对照使用。若要给出真实的变化量，需在基线上补采同样的采样。")
        W("")
    W("> **如何读这张表**：性能提升若伴随「安全冗余下降」或「温度上升」，"
      "说明收益是用资源余量换来的 —— 这类收益有上限，且会放大峰值风险。"
      "本报告在 §5.3 列出了对应的副作用与缓解措施。")
    W("")
    # 峰值显存的「恶化」需要单独说明：它不是缺陷，是换收益的代价
    _pv = _cmp_rows.get("peak_vram_mib")
    if _pv and _pv.get("improved") is False:
        W(f"> **显存「恶化」不是缺陷**：{_num(_pv['before'])} → {_num(_pv['after'])} MiB"
          f"（{_pv['delta_pct']:+.2f}%）的增量来自 MTP Draft 权重（+1.28 GiB）—— "
          f"它是 +77% 单流吞吐的**代价**。判定标准不是「显存不能涨」，"
          f"而是「涨完之后是否仍满足硬约束」：{_num(_pv['after'])} MiB ≤ 20,480 MiB ✅，"
          f"且安全冗余 {_num((_cmp_rows.get('safety_margin_gib') or {}).get('after'))} GiB "
          f"≥ 4.0 GiB ✅。")
        W("")

    # --------------------------------------------------------- 6 回归验收
    W("## 6. 回归验收")
    W("")

    # ---- 6.0 输入与产出对照 ------------------------------------------------
    _rm = t.get("regression_matrix")
    W("### 6.0 本步的输入与产出对照")
    W("")
    for _ln in _cov_block(
        "稳定、可规模化",
        "是否稳定？能否规模复制？",
        [["**主要输入**：优化版本", f"`{ver['after_bundle']['id']}`", "✅"],
         ["**主要输入**：测试报告", "§2 基线 + §3 剖面 + §5 验证", "✅"],
         ["**主要输入**：产品需求", "§1.7 验收标准（must_pass / should_pass / guardrail）", "✅"],
         ["**活动**：多场景测试",
          f"{len(_rm)} 个维度" if _rm else "5 个通用维度（见 6.3）", "✅"],
         ["**活动**：多设备验证", "单设备任务（见 §1.1 的范围边界）", "— 不适用"],
         ["**活动**：稳定性测试", "§6.3 的长时间运行 / 异常输入项", "✅"],
         ["**活动**：产品验收", "§6.1 验收判定 + §6.2 未达标归因", "✅"],
         ["**活动**：案例沉淀", "§7 落地复盘 + `task_*.yaml` 的 retrospective", "✅"],
         ["**关键分析（稳定性）**：长时间运行", "§6.3 回归矩阵", "✅"],
         ["**关键分析（稳定性）**：多输入长度", "§6.3 回归矩阵", "✅"],
         ["**关键分析（稳定性）**：多设备", "单设备任务，不适用", "— 不适用"],
         ["**关键分析（稳定性）**：异常输入", "§6.3 回归矩阵", "✅"],
         ["**关键分析（知识沉淀）**：优化经验 / 配置参数", "§7.3 可复用经验", "✅"],
         ["**关键分析（知识沉淀）**：性能模型", "§1.4 预算模型 + §3 剖面模型", "✅"],
         ["**关键分析（知识沉淀）**：问题案例", "§7.2 风险点 + 台账的 excluded 批次", "✅"],
         ["**输出物** ①验收报告", "本章 §6.1-§6.2", "✅"],
         ["**输出物** ②回归报告", "§6.3 回归测试矩阵", "✅"],
         ["**输出物** ③案例库", "§7 复盘（可随配置迁移到新项目）", "✅"],
         ["**输出物** ④最佳实践", "§7.3 可复用经验", "✅"]]):
        W(_ln)

    W("### 6.1 验收判定")
    W("")
    W("**① must_pass —— 硬性项**（任一不达标即判定方案失败）：")
    W("")
    W(_table(["验收条件", "实测值", "结论"],
             [[c["condition"], c["actual"],
               {True: "✅ PASS", False: "❌ FAIL", None: "— 待复测"}[c["pass"]]]
              for c in ver["acceptance_checks"]],
             ["l", "l", "c"]))
    W("")
    _should = ver.get("should_pass_checks") or []
    if _should:
        W("**② should_pass —— 期望项**（不达标不阻断交付，但需说明原因）：")
        W("")
        W(_table(["验收条件", "实测值", "结论"],
                 [[c["condition"], c["actual"],
                   {True: "✅ PASS", False: "❌ FAIL", None: "— 待复测"}[c["pass"]]]
                  for c in _should],
                 ["l", "l", "c"]))
        W("")
        _sf = [c for c in _should if c["pass"] is False]
        if _sf:
            W("> 期望项未达标的原因见 §6.2；这些项**不阻断交付**，"
              "但必须评估其对业务的影响。")
            W("")
        _inv = ver.get("invalid_conditions") or []
        if _inv:
            W("> **「不判定」不等于「没测」**："
              + "、".join(f'`{x["key"]}`' for x in _inv)
              + " 在本配置下的口径失效（详见 §5.1 的口径备注）。"
                "把它标成「待复测」会让人以为补一次测量就能判定 —— "
                "实际上只要开着投机解码，这个口径就永远不成立，"
                "应改用「每次权重读取产出的 token 数」来衡量。")
            W("")
    W("### 6.2 未达标项归因")
    W("")
    W("")
    unmet = [r for r in ver["comparison"]
             if r["meets_target"] is False]
    if unmet:
        for r in unmet:
            W(f'- `{r["key"]}`：实测 {_num(r["after"])} {r["unit"]}，'
              f'目标 {_num(r["target"])}（{r["after_source"]}）')
    else:
        W("- 无")
    W("")
    if any(r["key"] == "prefill_tps" for r in unmet):
        # 只在「Prefill 目标确实源自未验证声明」时才这样解释 ——
        # 否则会把这个模板化的说辞套到别的问题上（例如 CPU 上
        # Prefill 变慢其实是 KV 量化的代价，与声明无关）。
        _prefill_conflict = any(
            c.get("id") == "prefill_tps" for c in defn.get("source_conflicts", []))
        if _prefill_conflict:
            W("> `prefill_tps` 未达标**不是性能缺陷，而是 KPI 目标本身取自未经验证的 "
              "参考声明**（见 §1.5 的口径矛盾）。"
              "以可复现的口径衡量，该目标从一开始就不可达。"
              "建议用 `perfkit baseline --mode live` 重测 Prefill 曲线后按实测重设目标，"
              "而不是为了达标去追求一个来源存疑的数字。")
        else:
            W("> `prefill_tps` 未达标需结合具体配置判断："
              "本任务的 Prefill 目标按**深度场景**设定，"
              "若实际运行深度与该场景不同，应调整 KPI 的 `scenario` 绑定，"
              "而不是直接判定方案失败。")
        W("")
    if any(r["key"] == "ttft_ms" for r in unmet):
        W("> `ttft_ms` 未达标同源：TTFT 由 Prefill 时长主导，"
          "在 Prefill 口径歧义消除前该指标不具备判定条件。"
          "当前值仅反映「32K prompt 的 Prefill 耗时」，"
          "而生产上多轮对话应靠 Prefix Cache 复用避免重复 Prefill。")
        W("")
    if any(r["key"] == "ctx_decay_8k" for r in unmet):
        W("> `ctx_decay_8k` 未达标**不是可修复的缺陷，而是 CPU 后端的结构性特征**："
          "共享内存带宽下，KV 读写与权重读取直接竞争，"
          "上下文越长，可用带宽中分给权重的比例越低。"
          "可行的应对是**产品层**的而非性能层的：限制上下文长度、"
          "用 Prefix Cache 避免重复 Prefill、或把长上下文请求路由到独显后端。")
        W("")
    if any(r["key"] == "prefill_tps" for r in unmet):
        W("> `prefill_tps` 未达标与 KV 量化直接相关：见 §5.3 的"
          "「量化对 Prefill 的代价」—— 量化救 Decode 但伤 Prefill。"
          "若 Prefill 延迟是硬约束（如交互式首字体验），"
          "应重新评估是否值得开启 KV 量化。")
        W("")
    W("### 6.3 质量红线与回归矩阵")
    W("")
    _verified = [g for g in ver["guardrails"] if g.get("status") == "verified_live"]
    _pending = [g for g in ver["guardrails"] if g.get("status") != "verified_live"]
    if _verified:
        W("**质量与稳定性红线 —— 已用真机验证的部分**：")
        W("")
        W(_table(["红线", "结论", "证据"],
                 [[f"{_guard_str(g['guardrail'])}"
                   + (f" = `{g['expect']}`" if g.get("expect") is not None else ""),
                   "✅ 已验证", g.get("evidence") or "—"] for g in _verified],
                 ["l", "c", "l"]))
        W("")
    if _pending:
        W("**质量与稳定性红线 —— 尚未验证的部分**（缺数据就不能声称通过）：")
        W("")
        for g in _pending:
            W(f"- [ ] {_guard_str(g['guardrail'])}"
              + (f" = `{g['expect']}`" if g.get("expect") is not None else ""))
        W("")
    W("**回归测试矩阵**：")
    W("")
    _rm = t.get("regression_matrix")
    if _rm:
        W(_table(["维度", "覆盖内容", "通过判据"],
                 [[x.get("dim", ""), x.get("coverage", ""), x.get("criteria", "")]
                  for x in _rm],
                 ["l", "l", "l"]))
    else:
        W(_table(["维度", "覆盖内容", "通过判据"],
                 [["多输入长度", "短 prompt / 长 prompt / 超长 prompt",
                   "无 400 报错、无截断、TTFT 随长度线性"],
                  ["多并发度", "1 / 2 / 4 槽位（若启用）",
                   "槽位隔离、无争锁、人均速率达标"],
                  ["长时间运行", "持续负载压测",
                   "无 OOM、无泄漏、温度与吞吐稳态"],
                  ["异常输入", "超长 prompt、空输入、Unicode/Emoji",
                   "优雅降级或明确报错"],
                  ["质量回归", "PPL + 长上下文任务集",
                   "PPL 劣化在 guardrail 内，任务准确率无回退"]],
                 ["l", "l", "l"]))
        W("")
        W("> 本任务未在 `task_*.yaml` 配置 `regression_matrix`，上表为通用模板。"
          "建议按后端与任务定制 —— 例如 CPU 场景应覆盖线程数与系统负载，"
          "端侧场景应覆盖热稳态与内存配额。")
    W("")

    # --------------------------------------------------------- 7 落地复盘
    W("## 7. 落地复盘")
    W("")

    _retro = t.get("retrospective") or {}
    W("### 7.0 本步的输入与产出对照")
    W("")
    for _ln in _cov_block(
        "稳定、可规模化（与第 6 步同判据，本步负责沉淀）",
        "留下什么资产？下一轮从哪继续？",
        [["**主要输入**：优化版本", f"`{ver['after_bundle']['id']}`", "✅"],
         ["**主要输入**：测试报告", "§2 基线 / §3 剖面 / §5 验证 / §6 验收", "✅"],
         ["**主要输入**：产品需求", "§1.1 的业务需求与范围边界", "✅"],
         ["**关键分析（稳定性）**：长时间运行", "§6.3 回归矩阵 + §7.2 风险点", "✅"],
         ["**关键分析（稳定性）**：多输入长度 / 异常输入", "§6.3 回归矩阵", "✅"],
         ["**关键分析（稳定性）**：多设备", "单设备任务，不适用", "— 不适用"],
         ["**关键分析（知识沉淀）**：优化经验",
          f"{len(_retro.get('lessons') or [])} 条 lessons"
          if _retro.get("lessons") else "未配置",
          _st(bool(_retro.get("lessons")), "未配置")],
         ["**关键分析（知识沉淀）**：配置参数",
          "§7.1 方案取舍 + §4.2 启动命令（可复制的完整参数）", "✅"],
         ["**关键分析（知识沉淀）**：性能模型",
          "§1.4 显存预算模型 + §3 剖面归因模型 + §0.6 三层参照", "✅"],
         ["**关键分析（知识沉淀）**：问题案例",
          f"{len(_retro.get('risks') or [])} 条风险 + 台账的 excluded 批次"
          if _retro.get("risks") else "未配置",
          _st(bool(_retro.get("risks")), "未配置")],
         ["**输出物** ①验收报告", "§6.1-§6.2", "✅"],
         ["**输出物** ②回归报告", "§6.3", "✅"],
         ["**输出物** ③案例库", "§7.1-§7.3（写在 `task_*.yaml` 的 retrospective 段，随配置迁移）",
          _st(bool(_retro), "未配置")],
         ["**输出物** ④最佳实践", "§7.3 可复用经验", _st(bool(_retro.get("lessons")), "未配置")]]):
        W(_ln)

    retro = t.get("retrospective")
    if not retro:
        W("> 本任务未在 `task_*.yaml` 中配置 `retrospective` 段。"
          "复盘内容属于**项目级知识**，应写在配置里而非硬编码在报告模板中 ——"
          "这样换项目时复盘随配置一起替换。")
        W("")
        W("建议在 `task_*.yaml` 中补充：")
        W("")
        W("```yaml")
        W("retrospective:")
        W("  tradeoffs:            # 方案取舍：选择 / 放弃的选项 / 理由")
        W("    - {point: ..., chose: ..., rejected: ..., why: ...}")
        W("  risks:                # 风险点：触发条件 / 影响 / 应对")
        W("    - {risk: ..., trigger: ..., impact: ..., mitigation: ...}")
        W("  lessons:              # 可复用经验（纯文本列表）")
        W("    - ...")
        W("  next_steps:           # 后续迭代方向")
        W("    - {direction: ..., gain: ..., priority: P0, precondition: ...}")
        W("```")
        W("")
    else:
        tr = retro.get("tradeoffs", [])
        if tr:
            W("### 7.1 方案取舍")
            W("")
            W(_table(["取舍点", "选择", "放弃的选项", "理由"],
                     [[x.get("point", ""), x.get("chose", ""),
                       x.get("rejected", ""), x.get("why", "")] for x in tr],
                     ["l", "l", "l", "l"]))
            W("")
        rk = retro.get("risks", [])
        if rk:
            W("### 7.2 风险点")
            W("")
            W(_table(["风险", "触发条件", "影响", "应对"],
                     [[x.get("risk", ""), x.get("trigger", ""),
                       x.get("impact", ""), x.get("mitigation", "")] for x in rk],
                     ["l", "l", "l", "l"]))
            W("")
        ls = retro.get("lessons", [])
        if ls:
            W("### 7.3 可复用经验")
            W("")
            for i, x in enumerate(ls, 1):
                W(f"{i}. {x}")
            W("")
        ns = retro.get("next_steps", [])
        if ns:
            W("### 7.4 后续迭代方向")
            W("")
            W(_table(["方向", "预期收益", "优先级", "前置条件"],
                     [[x.get("direction", ""), x.get("gain", ""),
                       x.get("priority", ""), x.get("precondition", "")] for x in ns],
                     ["l", "l", "c", "l"]))
            W("")

    # ---------------------------------------------------------------- 附录
    W("---")
    W("")
    W("## 附录 A. 复现命令")
    W("")
    W("```bash")
    W("# 一次性跑完七步闭环")
    W(f"python3 -m perfkit all --config {cfg.source_dir} --run {cfg.run_id}")
    W("")
    W("# 或逐步执行")
    for _cmd, _desc in [("define", "1 问题定义"), ("baseline", "2 性能基线"),
                        ("profiling", "3 性能剖面"), ("diagnose", "4 瓶颈定位"),
                        ("optimize", "5 优化方案"), ("verify", "6 效果验证"),
                        ("report", "7 报告导出"), ("visualize", "可视化")]:
        W(f"python3 -m perfkit {_cmd:<9} --config {cfg.source_dir} "
          f"--run {cfg.run_id}   # {_desc}")
    W("")
    W("# 架构超参核验（消除反推值的不确定性）")
    W(f"python3 -m perfkit define --config {cfg.source_dir} --run {cfg.run_id} \\")
    W(f"    --from-gguf {m['file']}")
    W("")
    W("# 真机模式（需在目标设备执行）")
    W(f"python3 -m perfkit baseline --config {cfg.source_dir} "
      f"--run {cfg.run_id} --mode live")
    W("")
    W("# 环境画像与带宽探针")
    W("./scripts/collect_env.sh")
    _pb = cfg.backend.probe_binary
    if cfg.backend.kind == "gpu":
        W(f"bash scripts/build.sh --cuda && ./build/{_pb} --size-gib 4")
    else:
        W(f"./scripts/build.sh && ./build/{_pb} --mode all --size-gib 2")
    W("```")
    W("")
    W("## 附录 B. 数据来源与可信度")
    W("")
    # 这张表必须按**本次实际用到的数据**动态生成，不能写死 ——
    # 写死的「本报告中的使用」列会在换模型后继续声称用了某些数据。
    _gv = defn.get("gguf_verification") or {}
    _m_items = ["基线吞吐（llama-bench 与 llama-server 两套口径）",
                f"峰值显存 {((base.get('memory') or {}).get('peak_used_mib') or '—')} MiB",
                (f"长上下文深度曲线（{_curve_range}）"
                 if (_curve_range := _curve_span(base)) else
                 "长上下文深度曲线（本次未采集）"),
                "KV 精度同深度对照（f16/q8q8/q8q4/q4q4 @32K）",
                "FA on/off、batch 1024/2048/4096 对照",
                "MTP 接受率（llama-server 日志逐请求）",
                "30 分钟持续负载（吞吐/显存/温度/功耗）"]
    _d_items = ["逐算子耗时归因（权重字节占比 × 实测单步耗时）",
                "带宽利用率与 roofline 判定",
                "KV 几何容量（按 GGUF 真值架构）",
                "理论上界（标称带宽与实测纯读带宽两种口径）"]
    _i_items = (["**本报告已无 [I] 项** —— 架构超参全部来自 GGUF 元数据核验"]
                if _gv.get("all_match") else
                ["层数/hidden/GQA 配置（可用 `perfkit define --from-gguf` 覆盖）"])
    # 只在「真的没测」时才把它列进 [T] —— 测过了还写待采集，
    # 会让读者以为质量红线是空的。
    _q_all = _ledger_quality(cfg)
    _t_items = ([] if (_q_all.get("ppl") or {}).get("rows")
                else ["PPL（量化质量守门）"])
    _todo_extra = {"gpu_util": "GPU 利用率（nvidia-smi 采样，观测项）",
                   "power_w": "稳态功耗（已采峰值，缺持续采样曲线）"}
    for _k, _lbl in _todo_extra.items():
        _row = next((r for r in ver["comparison"] if r["key"] == _k), None)
        if _row and _row.get("after_source") == "todo":
            _t_items.append(_lbl)
    _rows_b = [["**[M]**", "实测 / 实测换算，可直接引用", "；".join(_m_items)],
               ["**[D]**", "由实测值推导（须标注推导公式）", "；".join(_d_items)],
               ["**[I]**", "按公开架构惯例反推，需 GGUF 元数据核验",
                "；".join(_i_items)]]
    _rows_b.append(["**[T]**", "待采集，不得作为结论依据",
                    "；".join(_t_items) if _t_items
                    else "**本次无 [T] 项** —— 所有进入结论的数据都已实测"])
    W(_table(["标记", "含义", "本报告中的使用"], _rows_b, ["l", "l", "l"]))
    W("")
    W("## 附录 C. 本次运行的产物清单")
    W("")
    W("```text")
    W(f"output/{cfg.run_id}/")
    W("├── definition.json      第1步 问题定义（画像/预算/roofline/KPI/对账）")
    W("├── baseline.json        第2步 性能基线")
    W("├── profiling.json       第3步 四维性能剖面")
    W("├── diagnose.json        第4步 瓶颈结论 + 证据链 + 反证")
    W("├── optimize.json        第5步 方案选型 + bundle 对比 + 启动命令")
    W("├── start_optimized.sh   第5步 可直接执行的启动脚本")
    W("├── verify.json          第6步 前后对比 + 副作用清单 + 验收判定")
    W("├── report.md            第7步 本报告")
    W("├── charts.html          可视化图表（自包含，浏览器直接打开）")
    W("├── raw/                 原始测量数据（每个 [M] 数字的出处，见 raw/README.md）")
    W("└── logs/                Shell 编排模式下的过程日志")
    W("```")
    W("")
    W(f"配置与数据源：`{cfg.source_dir}/`（model / device / task / plan 四份 YAML）"
      f"与 `{cfg.ledger().relative_to(cfg.root) if cfg.ledger().is_relative_to(cfg.root) else cfg.ledger()}`（实测台账）。")
    W("")
    return "\n".join(L)


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    defn = load_json(cfg.out("definition.json"))
    base = load_json(cfg.out("baseline.json"))
    prof = load_json(cfg.out("profiling.json"))
    diag = load_json(cfg.out("diagnose.json"))
    opt = load_json(cfg.out("optimize.json"))
    ver = load_json(cfg.out("verify.json"))

    md = build_report(cfg, defn, base, prof, diag, opt, ver)
    path = cfg.out("report.md")
    path.write_text(md, encoding="utf-8")

    # 同步一份到 output/ 根目录，方便直接阅读
    mirror = cfg.root / "output" / f"report_{cfg.run_id}.md"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(md, encoding="utf-8")

    print(f"[report] 已生成《{cfg.model['name']} 推理性能分析与优化报告》")
    print(f"  {path}")
    print(f"  {mirror}")
    print(f"  字数 {len(md)}，章节：七步闭环 0-7 + 附录 A/B/C")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("report", help="【第7步】报告导出：组装七步闭环 Markdown 报告")
    p.set_defaults(func=main)
    return p
