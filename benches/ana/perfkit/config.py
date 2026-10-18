"""
perfkit · 配置加载与画像计算
==============================================================================
职责：把 configs/ 下的 model / device / task / plan 四份 YAML 读进来，
     做占位符解析与交叉校验，并提供全部「第一性原理估算器」。

设计原则：估算器是纯函数，不依赖任何外部状态，方便单测与复用。
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

# ============================================================================
# YAML 加载：优先 PyYAML；缺失时退化为内置的「受限 YAML」解析器。
# ----------------------------------------------------------------------------
# 受限解析器覆盖本仓库配置用到的全部语法：
#   - 缩进嵌套映射 / 序列
#   - 行内流式映射 {a: 1, b: 2} 与序列 [a, b, c]
#   - 字符串（含引号、含冒号与中文）、整数、浮点、bool、null
#   - `>` 折叠块标量（用于多行 description）
#   - `#` 行注释与空行
# 不支持锚点/别名/多文档/复杂块标量，够用即可。
# ============================================================================


class ConfigError(Exception):
    """配置层错误：路径缺失、字段类型不符、交叉校验失败。"""


def _strip_comment(line: str) -> str:
    """剥离行尾注释，但保留引号内的 # 与 URL 中的 #。"""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (not out or out[-1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _scalar(tok: str) -> Any:
    """把标量字符串转成 Python 原生类型。"""
    tok = tok.strip()
    if not tok:
        return None
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    low = tok.lower()
    if low in ("null", "~", "none"):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    # 去掉下划线分隔的数字字面量（如 24_564）
    num = tok.replace("_", "")
    try:
        return int(num)
    except ValueError:
        pass
    try:
        return float(num)
    except ValueError:
        pass
    return tok


def _split_top(text: str, sep: str = ",") -> list[str]:
    """按 sep 切分，忽略引号与括号内部的 sep。"""
    parts, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _inline(text: str) -> Any:
    """解析行内流式集合 {..} / [..]。"""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_inline(p) for p in _split_top(inner)] if inner else []
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        out: dict[str, Any] = {}
        if not inner:
            return out
        for pair in _split_top(inner):
            if ":" not in pair:
                raise ConfigError(f"行内映射缺少冒号: {pair!r}")
            k, _, v = pair.partition(":")
            out[str(_scalar(k))] = _inline(v)
        return out
    return _scalar(text)


def _depth(text: str) -> int:
    """统计行内流式集合的未闭合深度（忽略引号内部）。"""
    depth, quote = 0, None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
    return depth


def _mini_yaml(text: str) -> Any:
    """受限 YAML 解析器（PyYAML 不可用时的兜底）。"""
    raw_lines: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        body = _strip_comment(ln)
        if not body.strip():
            continue
        if pending is not None:
            # 续接上一行的未闭合流式集合，缩进沿用起始行
            indent, acc = pending
            acc = f"{acc} {body.strip()}"
            if _depth(acc) > 0:
                pending = (indent, acc)
                continue
            raw_lines.append((indent, acc))
            pending = None
            continue
        indent = len(body) - len(body.lstrip(" "))
        content = body.strip()
        if _depth(content) > 0:
            pending = (indent, content)
            continue
        raw_lines.append((indent, content))
    if pending is not None:
        raw_lines.append(pending)

    def parse_block(idx: int, indent: int) -> tuple[Any, int]:
        # 判断当前块是映射还是序列
        is_seq = raw_lines[idx][1].startswith("- ")
        container: Any = [] if is_seq else {}
        while idx < len(raw_lines):
            cur_indent, content = raw_lines[idx]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ConfigError(f"意外的缩进: {content!r}")
            if is_seq:
                if not content.startswith("- "):
                    break
                item = content[2:].strip()
                idx += 1
                if not item:                                  # 嵌套块
                    if idx < len(raw_lines) and raw_lines[idx][0] > indent:
                        val, idx = parse_block(idx, raw_lines[idx][0])
                        container.append(val)
                    else:
                        container.append(None)
                elif ":" in item and not item.startswith(("{", "[")) \
                        and re.match(r"^[^:\s][^:]*:", item):
                    # 序列项本身是个映射，如 "- id: foo"
                    key, _, rest = item.partition(":")
                    entry: dict[str, Any] = {}
                    rest = rest.strip()
                    if rest:
                        entry[str(_scalar(key))] = _inline(rest)
                    elif idx < len(raw_lines) and raw_lines[idx][0] > indent:
                        val, idx = parse_block(idx, raw_lines[idx][0])
                        entry[str(_scalar(key))] = val
                    else:
                        entry[str(_scalar(key))] = None
                    # 继续吸收同级缩进的兄弟键
                    while idx < len(raw_lines) and raw_lines[idx][0] > indent:
                        sub_indent, sub = raw_lines[idx]
                        if sub.startswith("- "):
                            break
                        k2, _, v2 = sub.partition(":")
                        idx += 1
                        v2 = v2.strip()
                        if v2 in (">", "|"):                      # 折叠块标量
                            buf = []
                            while idx < len(raw_lines) and raw_lines[idx][0] > sub_indent:
                                buf.append(raw_lines[idx][1])
                                idx += 1
                            entry[str(_scalar(k2))] = " ".join(buf)
                        elif v2:
                            val2 = _inline(v2)
                            while idx < len(raw_lines) and raw_lines[idx][0] > sub_indent \
                                    and isinstance(val2, str):
                                val2 = f"{val2} {raw_lines[idx][1]}"
                                idx += 1
                            entry[str(_scalar(k2))] = val2
                        elif idx < len(raw_lines) and raw_lines[idx][0] > sub_indent:
                            val, idx = parse_block(idx, raw_lines[idx][0])
                            entry[str(_scalar(k2))] = val
                        else:
                            entry[str(_scalar(k2))] = None
                    container.append(entry)
                else:
                    # 纯标量项：更深缩进的后续行是它的续行（YAML 多行 plain scalar）
                    val = _inline(item)
                    while idx < len(raw_lines) and raw_lines[idx][0] > indent \
                            and isinstance(val, str):
                        val = f"{val} {raw_lines[idx][1]}"
                        idx += 1
                    container.append(val)
            else:
                if content.startswith("- "):
                    break
                if ":" not in content:
                    raise ConfigError(f"映射行缺少冒号: {content!r}")
                key, _, rest = content.partition(":")
                idx += 1
                rest = rest.strip()
                if rest in (">", "|"):                            # 折叠块标量
                    buf = []
                    while idx < len(raw_lines) and raw_lines[idx][0] > cur_indent:
                        buf.append(raw_lines[idx][1])
                        idx += 1
                    container[str(_scalar(key))] = " ".join(buf)
                elif rest:
                    val = _inline(rest)
                    while idx < len(raw_lines) and raw_lines[idx][0] > cur_indent \
                            and isinstance(val, str):
                        val = f"{val} {raw_lines[idx][1]}"
                        idx += 1
                    container[str(_scalar(key))] = val
                elif idx < len(raw_lines) and raw_lines[idx][0] > cur_indent:
                    val, idx = parse_block(idx, raw_lines[idx][0])
                    container[str(_scalar(key))] = val
                else:
                    container[str(_scalar(key))] = None
        return container, idx

    if not raw_lines:
        return {}
    value, _ = parse_block(0, raw_lines[0][0])
    return value


def load_yaml(path: str | Path) -> dict:
    """加载 YAML。优先 PyYAML，不可用时走内置受限解析器。"""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except ImportError:
        data = _mini_yaml(text)
    if not isinstance(data, dict):
        raise ConfigError(f"配置根节点必须是映射: {path}")
    return data


def save_json(obj: Any, path: str | Path) -> Path:
    """写 JSON，自动建目录，UTF-8 + 缩进 + 中文不转义。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"数据文件不存在: {path}（请先执行上游子命令）")
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize_flags(flags: list[str]) -> list[str]:
    """
    把 bundle 里的 startup_flags 展平成 token 序列。

    配置里为了可读性写成 `"--parallel 2"`、`"--cache-type-k q8_0"`，
    但下游需要 `["--parallel", "2"]` 这样的形式才能做键值查找。
    这里统一做一次展平，避免每个消费方各写一遍字符串匹配。
    """
    out: list[str] = []
    for f in flags:
        out.extend(shlex.split(str(f)))
    return out


def flag_value(flags: list[str], name: str, cast=str):
    """在展平后的 token 序列里取 `--name value` 的 value。"""
    toks = flags if isinstance(flags, list) and all(
        " " not in t for t in flags) else tokenize_flags(flags)
    for i, t in enumerate(toks):
        if t == name and i + 1 < len(toks):
            try:
                return cast(toks[i + 1])
            except (TypeError, ValueError):
                return None
    return None


def flag_present(flags: list[str], name: str) -> bool:
    toks = tokenize_flags(flags)
    return any(t == name for t in toks)


# ============================================================================
# 第一性原理估算器（纯函数）
# ============================================================================

GIB = 1024 ** 3


def attn_layers(model: dict) -> int:
    """
    参与 KV Cache 的**全注意力层**数。

    为什么不能直接用 n_layer：
      混合架构（如 Qwen3.5 的 SSM/线性注意力 + 全注意力交替）里，
      只有全注意力层拥有随 ctx 线性增长的 KV Cache；SSM 层维护的是
      **固定大小**的循环状态（见 `ssm_state_bytes`），与上下文长度无关。
      用 n_layer 代入几何公式会把 KV 高估「总层数 ÷ 注意力层数」倍。

    配置里没有 `n_attn_layer` 时退回 n_layer（纯 Transformer 模型）。
    """
    a = model["architecture"]
    return int(a.get("n_attn_layer") or a["n_layer"])


def ssm_layers(model: dict) -> int:
    """线性注意力（SSM）层数；纯 Transformer 模型为 0。"""
    a = model["architecture"]
    return int(a.get("n_ssm_layer") or max(a["n_layer"] - attn_layers(model), 0))


def kv_cache_bytes(model: dict, ctx_tokens: int, k_bits: float, v_bits: float) -> int:
    """
    KV Cache 精确容量（字节）—— 几何公式（第一性原理）。

        n_attn_layer × ctx × n_head_kv × head_dim × 2(K,V) × bytes_per_elem

    分母是**全注意力层数**而不是总层数：混合架构里 SSM 层不产生
    随 ctx 增长的 KV（见 attn_layers 的说明）。

    ggml 的 q8_0 为 32 元素/块、每块 34 字节（2 字节 fp16 scale）
        ⇒ 有效 8.5 bit/权重；
    q4_0 为 32 元素/块、每块 18 字节 ⇒ 有效 4.5 bit/权重；
    f16 即 16 bit。

    这是端侧最敏感的公式：KV 随 ctx 线性增长，是长上下文显存墙的主因。
    """
    a = model["architecture"]
    elems = attn_layers(model) * ctx_tokens * a["n_head_kv"] * a["head_dim"] * 2
    return int(elems * (k_bits + v_bits) / 2.0 / 8.0)


def ssm_state_bytes(model: dict, parallel: int = 1) -> int:
    """
    混合架构中 SSM（线性注意力）层的**固定状态**字节数。

    与 KV Cache 的本质区别：它**不随 ctx 增长**，只随并发槽位数线性增长。
    Mamba-2 风格的状态由两部分构成（按 f16 计）：

        循环状态  n_ssm_layer × inner_size × state_size
        卷积状态  n_ssm_layer × inner_size × (conv_kernel - 1)

    为什么必须单列：把它和 KV 混在一起会让「长上下文显存墙」的判断失真 ——
    64K 时 KV 会涨到 GiB 级，而 SSM 状态始终是几十 MiB 的常数。
    纯 Transformer 模型没有这些字段，返回 0。
    """
    a = model["architecture"]
    n_ssm = ssm_layers(model)
    if not n_ssm:
        return 0
    inner = int(a.get("ssm_inner_size") or 0)
    state = int(a.get("ssm_state_size") or 0)
    if not inner or not state:
        return 0
    conv_k = int(a.get("ssm_conv_kernel") or 0)
    bytes_per_elem = float(a.get("ssm_state_dtype_bytes", 2.0))
    per_layer = inner * state + inner * max(conv_k - 1, 0)
    return int(n_ssm * per_layer * bytes_per_elem * max(parallel, 1))


def kv_cache_bytes_calibrated(model: dict, ctx_tokens: int, k_bits: float,
                              v_bits: float) -> tuple[int, bool]:
    """
    KV 容量：默认返回**几何公式**结果（第一性原理、内部自洽）。

    台账中给出的 KV 实测增量（64K/q8_0+q4_0 ≈ 2.2 GiB）与几何公式（6.50 GiB）
    相差约 3×。在拿到 GGUF 元数据核验 n_head_kv / head_dim 之前，
    无法判定哪一侧更接近真值，因此这里**不做静默校准**，
    而是在 reconcile() 中显式暴露差额，交由报告与核验流程处理。

    返回 (字节数, 是否与实测锚点存在未解释差额)。
    """
    geo = kv_cache_bytes(model, ctx_tokens, k_bits, v_bits)
    cal = (model.get("calibration") or {}).get("kv_cache")
    if cal and abs(cal["k_bits"] - k_bits) < 0.01 and abs(cal["v_bits"] - v_bits) < 0.01:
        ratio = cal["gib"] / cal["geometric_gib"]
        if abs(ratio - 1.0) > 0.25:
            return geo, True
    return geo, False


def reconcile(model: dict, device: dict) -> dict:
    """
    模型预测 vs 实测的对账表。

    工程闭环第 5 步「收益量化」的前提是模型可信；模型与实测不一致时，
    必须先把差额讲清楚，否则所有优化收益都无法归因。
    """
    cal = (model.get("calibration") or {}).get("kv_cache")
    if not cal:
        return {}
    a = model["architecture"]
    geo_at_anchor = kv_cache_bytes(model, cal["ctx"], cal["k_bits"], cal["v_bits"]) / GIB
    return {
        "kv_cache": {
            "ctx": cal["ctx"],
            "kv_type": f"K{cal['k_bits']}/V{cal['v_bits']}bit",
            "model_gib": round(geo_at_anchor, 2),
            "measured_gib": cal["gib"],
            "gap_gib": round(geo_at_anchor - cal["gib"], 2),
            "ratio": round(geo_at_anchor / cal["gib"], 2),
            "likely_cause": (
                f"几何公式用反推的 n_head_kv={a['n_head_kv']} / head_dim={a['head_dim']}；"
                "若真实架构的 KV 头数或 head_dim 更小，模型值会被高估。"
                "另一可能是台账 §4.4 的 KV 增量为工程粗估值。"
            ),
            "verify_by": (
                "在目标机上执行 `perfkit define --from-gguf <model.gguf>` "
                "读取真实元数据，并对比 llama.cpp 启动日志中的 KV buffer size"
            ),
            "impact": (
                "差额直接影响显存预算的安全冗余判定（本部署 5.4 GiB vs 1.1 GiB），"
                "进而影响并发槽位数与上下文上限的决策"
            ),
        },
        "source": cal.get("source", ""),
    }


def weight_bytes(model: dict) -> int:
    """
    单步 Decode 需要搬运的权重字节数。

    优先级：
      1) weight_breakdown.total_gib —— 实测分解的**有效权重**（可剔除未启用
         的 MTP 预测块；llama.cpp 启动时会忽略 blk.64，文件体积因此高估 2%）
      2) artifacts[role=main].bytes —— 文件实际体积
      3) 按量化位宽估算
    """
    br = model.get("weight_breakdown") or {}
    if br.get("total_gib"):
        return int(float(br["total_gib"]) * GIB)
    for art in model.get("artifacts", []):
        if art.get("role") == "main" and art.get("bytes"):
            return int(art["bytes"])
    bits = _bits_for_quant(model, model["quantization"])
    return int(model["params_total"] * bits / 8.0)


def _bits_for_quant(model: dict, quant: str) -> float:
    for row in model.get("precision_ladder", []):
        if row["quant"].upper() == quant.upper():
            return float(row["bits_per_weight"])
    raise ConfigError(f"量化档位 {quant} 未在 precision_ladder 中定义")


def theoretical_decode_tps(device: dict, weight_b: int, ctx_tokens: int,
                           model: dict, k_bits: float, v_bits: float) -> float:
    """
    Decode 理论速度上界（tokens/s）。

        tps_max = 带宽 ÷ 每步需搬运的字节数

    每步字节 = 权重（必读）+ KV Cache（读写各一遍）。
    自回归 Decode 是 **Memory-bound**：算力几乎不参与，带宽是唯一分母。
    这是 roofline 分析在端侧的最简形式，也是判断「是否已触及物理天花板」的基准。
    """
    bw = device["memory"]["mem_bandwidth_gbps_peak"] * 1e9
    kv, _ = kv_cache_bytes_calibrated(model, ctx_tokens, k_bits, v_bits)
    per_step = weight_b + 2 * kv
    return bw / per_step


def decode_flops_per_step(model: dict, ctx_tokens: int,
                          weight_b: int = 0) -> float:
    """
    单步 Decode 的 FLOPs 估算（乘加各计 1 次）。

        线性层       2 × 参数量
        全注意力层   4 × n_attn_layer × ctx × n_head × head_dim
                     （Q·Kᵀ 与 A·V 各 2 FLOPs/元素）
        SSM 层       2 × n_ssm_layer × inner_size × state_size
                     （递推状态更新，量级远小于前两项）

    两个容易错的地方：
      · 注意力项用 **n_head 而不是 n_head_kv** —— KV 头数决定*读多少字节*，
        但计算时每个 query 头都要与 K/V 相乘，GQA 只是省了访存不是省了算力。
      · 层数用 **n_attn_layer 而不是 n_layer** —— 混合架构里 SSM 层没有
        全注意力的 O(ctx) 项。

    注意：这里按**理论 FLOPs** 计，不含 kernel 实现浪费（padding、
    非融合算子的中间读写）。因此算出的「达成算力」是下界，
    利用率是保守估计。
    """
    a = model["architecture"]
    n_params = model.get("params_active") or model.get("params_total")
    if not n_params:
        n_params = weight_b * 8.0 / _bits_for_quant(model, model.get("quantization", ""))
    flops = 2.0 * float(n_params)
    flops += 4.0 * attn_layers(model) * ctx_tokens * a["n_head"] * a["head_dim"]
    if ssm_layers(model):
        inner = int(a.get("ssm_inner_size") or 0)
        state = int(a.get("ssm_state_size") or 0)
        flops += 2.0 * ssm_layers(model) * inner * state
    return flops


def arithmetic_intensity(model: dict, ctx_tokens: int, weight_b: int) -> float:
    """
    算术强度（FLOPs/Byte），用于 roofline 判定 compute-bound 还是 memory-bound。

        强度 = 每 token 的 FLOPs ÷ 每 token 需搬运的字节数

    这是 roofline 的横轴。强度低于拐点 ⇒ 算力无法被喂饱，
    无论算力多强都只能发挥出「强度 × 带宽」那一部分。

    分母取**权重字节**：Decode 每步都要把全部权重读一遍，这是常数项；
    KV Cache 的读写随 ctx 增长，属于可变的附加项。长上下文时把它计入
    会让强度进一步下降（更偏向 memory-bound），但不改变判定方向，
    因此这里保持「权重字节」这一保守口径，并在报告中单独给出 KV 占比。
    """
    return decode_flops_per_step(model, ctx_tokens, weight_b) / max(weight_b, 1)


def compute_side_analysis(device: dict, model: dict, ctx_tokens: int,
                          weight_b: int, step_ms: float) -> dict:
    """
    硬件剖面的**算力侧**分析（与带宽侧并列）。

    只报带宽回答不了「算力是不是也饱和了」。本函数给出四个量：

      1. 单步 FLOPs（理论）—— 分子
      2. 实测达成算力 = FLOPs ÷ 单步耗时
      3. 设备峰值算力 → 对峰值利用率（**低是正常的**）
      4. 该算术强度下的可达算力上界 = 强度 × 带宽

    ⚠️ 一个必须讲清的恒等关系
    ------------------------------------------------------------------
        达成算力 ÷ 可达上界 = (FLOPs/step_s) ÷ (强度 × 带宽)
                            = (FLOPs/step_s) ÷ (FLOPs/wb × 带宽)
                            = wb / (step_s × 带宽)
                            = **带宽利用率**

    也就是说：在 roofline 框架下，「算力达成率」与「带宽利用率」
    是同一个数的两种表达，**不是两个独立证据**。
    这不是巧合而是访存受限的定义 —— 该区间内算力完全由带宽决定。

    因此本函数**不**用这个比值去判定「算力侧另有损失」（那会变成
    把带宽结论换个说法重复一遍）。它只做两件有意义的事：

      · 给出「对峰值算力的利用率」—— 回答「硬件算力被用到了几成」
        （低是必然的，但要让读者看到这个数，而不是以为没测）
      · 给出「未被带宽解释的时间占比」= 1 − 带宽利用率
        —— 这才是真正的线索：时间还有多少花在访存之外。
        它指向什么需要看别的剖面（KV 读、launch、回退、采样），
        因此这里只标注「需进一步定位」，不擅自下结论。
    """
    flops = decode_flops_per_step(model, ctx_tokens, weight_b)
    step_s = max(step_ms, 1e-9) / 1000.0
    achieved_tflops = flops / step_s / 1e12

    peak = peak_compute_tflops(device)
    bw = device["memory"]["mem_bandwidth_gbps_peak"] * 1e9
    intensity = flops / max(weight_b, 1)
    attainable = intensity * bw / 1e12          # 该强度下受带宽限制的可达算力
    bw_util = (weight_b / step_s) / bw
    unexplained = max(0.0, 1.0 - bw_util)

    out: dict = {
        "flops_per_step": round(flops / 1e9, 2),          # GFLOPs
        "achieved_tflops": round(achieved_tflops, 3),
        "attainable_tflops_at_intensity": round(attainable, 3),
        "arithmetic_intensity": round(intensity, 2),
        # 保留该比值以便读者核对恒等关系，但显式标注它与带宽利用率同源
        "attainable_ratio": round(bw_util, 4),
        "unexplained_time_ratio": round(unexplained, 4),
        "identity_note": ("「算力达成率」在数学上恒等于「带宽利用率」"
                          "（roofline 框架下的恒等关系，非独立证据）"),
    }
    if peak:
        out["peak_tflops"] = round(peak, 3)
        out["compute_utilization"] = round(achieved_tflops / peak, 4)
    else:
        out["peak_tflops"] = None
        out["compute_utilization"] = None

    # ---- 判定：只对「时间去哪了」下结论，不对算力重复下结论 ---------------
    if out["compute_utilization"] is not None \
            and out["compute_utilization"] > 0.90 and bw_util < 0.60:
        out["verdict"] = "compute_bound"
        out["note"] = ("算力对峰值利用率 >90% 且带宽利用率 <60% —— "
                       "反证条件成立，应改判 compute_bound")
    elif bw_util >= 0.60:
        _cu = out["compute_utilization"]
        out["verdict"] = "bandwidth_limited"
        out["note"] = (
            f"带宽已用到 {bw_util*100:.1f}%"
            + (f"，而算力对峰值仅 {_cu*100:.1f}%" if _cu is not None else "")
            + " —— 算力低**不是缺陷**：该强度下带宽先把算力卡住了"
              f"（可达上界仅 {attainable:.2f} TFLOPS），"
              "提升算力无法加快 Decode。")
    else:
        out["verdict"] = "bandwidth_underused"
        out["note"] = (f"带宽仅用到 {bw_util*100:.1f}%，"
                       f"约 {unexplained*100:.1f}% 的单步时间**未被权重读取解释**。"
                       "这不能直接归因到算力（两者在 roofline 下同源），"
                       "需要看 §3.2 阶段剖面、§3.3 算子剖面、§3.6 回退代价定位："
                       "常见原因是 KV 读占比高、kernel 启动开销、"
                       "存在回退、或单步耗时被固定开销（采样/调度）主导。")
    return out


def peak_compute_tflops(device: dict, dtype: str = "fp16") -> float | None:
    """
    峰值算力（TFLOPS）。不同后端能拿到的字段不同，按优先级找：

      GPU  → compute.fp16_tflops_peak（CUDA/Tensor Core）
      SoC  → compute.fp16_tflops_peak 或 compute.npu_tops（INT8，量级参考）
      CPU  → compute.fp32_tflops_peak 或由核数×SIMD 宽度×频率推导

    返回 None 表示无可用数据 —— 此时 roofline 只能做定性判断，
    报告会显式标注，而不是拿一个编造的分母去算拐点。
    """
    c = device.get("compute", {}) or {}
    for key in (f"{dtype}_tflops_peak", "fp16_tflops_peak", "bf16_tflops_peak",
                "fp32_tflops_peak"):
        if c.get(key):
            return float(c[key])
    if c.get("npu_tops"):
        return float(c["npu_tops"])
    if c.get("int8_tops_peak"):
        return float(c["int8_tops_peak"]) / 2.0     # INT8 TOPS 折半近似 FP16 TFLOPS
    # CPU：按「物理核 × SIMD 每周期 FLOPs × 频率」估
    cores = c.get("cpu_cores") or c.get("physical_cores")
    ghz = c.get("cpu_ghz")
    fma_per_core_per_cycle = c.get("simd_flops_per_core_cycle")
    if cores and ghz and fma_per_core_per_cycle:
        return float(cores) * float(ghz) * float(fma_per_core_per_cycle) / 1000.0
    return None


def roofline_verdict(device: dict, intensity: float) -> dict:
    """
    roofline 判定：给出峰值算力/带宽下的拐点与结论。

    算力字段缺失时（常见于 CPU 画像未填 FP16 峰值）退化为**仅带宽判定**：
    仍能回答「每字节搬运算几次乘加」这个关键问题，但拐点标为未知。
    """
    bw = device["memory"]["mem_bandwidth_gbps_peak"] * 1e9
    tflops = peak_compute_tflops(device)
    out = {
        "intensity": round(intensity, 2),
        "peak_tflops_used": round(tflops, 2) if tflops else None,
    }
    if not tflops:
        out.update({
            "ridge_point": None,
            "verdict": "unknown_compute_peak",
            "attainable_tflops": None,
            "note": "设备画像缺少峰值算力字段，无法计算拐点；"
                    "请在 device_*.yaml 的 compute 段补充 fp16_tflops_peak 或 "
                    "cpu_cores + cpu_ghz + simd_flops_per_core_cycle",
        })
        return out
    fp16 = tflops * 1e12
    ridge = fp16 / bw
    out.update({
        "ridge_point": round(ridge, 2),
        "verdict": "memory_bound" if intensity < ridge else "compute_bound",
        "attainable_tflops": round(min(intensity * bw, fp16) / 1e12, 2),
    })
    return out


def _mem_total_bytes(device: dict) -> int:
    """
    该设备可承载模型的「加速器内存」总量（字节）。

    GPU 取显存容量，CPU/SoC 取可用内存（App 配额或系统内存）—— 语义不同
    但预算模型一致，故统一成一个入口。字段优先级由 backend 层定义，
    这里保留一份不依赖 backend 实例的解析，供纯函数路径使用。
    """
    mem = device.get("memory", {}) or {}
    for key in ("vram_total_mib", "ram_total_mib", "mem_total_mib"):
        if mem.get(key):
            return int(mem[key]) * 1024 * 1024
    for key in ("ram_total_gb", "vram_total_gb"):
        if mem.get(key):
            return int(float(mem[key]) * 1024 ** 3)
    raise ConfigError("device.memory 缺少内存容量字段"
                      "（vram_total_mib / ram_total_mib / ram_total_gb 之一）")


def mem_budget(device: dict, model: dict, ctx_tokens: int, parallel: int,
               k_bits: float, v_bits: float, include_mmproj: bool = False,
               include_draft: bool = False) -> dict:
    """
    显存预算表：把 24GB 拆成权重 / mmproj / KV / runtime 四块。
    这是工程闭环第 1 步「问题定义」里「显存预算」产出的核心。
    """
    w = weight_bytes(model)
    kv_total, calibrated = kv_cache_bytes_calibrated(model, ctx_tokens, k_bits, v_bits)
    kv_per_slot = kv_total // max(parallel, 1)
    ssm_state = ssm_state_bytes(model, parallel)
    mm = 0
    if include_mmproj:
        for art in model.get("artifacts", []):
            if art.get("role") == "mmproj":
                mm = int(art["bytes"])
    # MTP Draft 权重：只有挂 `-md` 时才占显存。它不是一个可忽略的小数
    # （1.28 GiB ≈ 显存预算的 7%），漏算会让「安全冗余」被高估。
    draft = 0
    if include_draft:
        for art in model.get("artifacts", []):
            if art.get("role") == "draft_mtp":
                draft = int(art["bytes"])
    runtime = 200 * 1024 * 1024                   # FlashAttention/激活/CUDA context ~0.2GiB
    total = w + mm + draft + kv_total + ssm_state + runtime
    vram = _mem_total_bytes(device)
    return {
        "weight_gib": round(w / GIB, 2),
        "mmproj_gib": round(mm / GIB, 2),
        "draft_gib": round(draft / GIB, 2),
        "kv_total_gib": round(kv_total / GIB, 2),
        "kv_per_slot_gib": round(kv_per_slot / GIB, 2),
        # 混合架构（SSM/线性注意力）的固定状态：不随 ctx 增长，只随槽位数增长
        "ssm_state_gib": round(ssm_state / GIB, 3),
        "runtime_gib": round(runtime / GIB, 2),
        "total_gib": round(total / GIB, 2),
        "mem_total_gib": round(vram / GIB, 2),
        # 兼容旧字段名（报告/图表历史引用）；语义同 mem_total_gib
        "vram_total_gib": round(vram / GIB, 2),
        "safety_margin_gib": round((vram - total) / GIB, 2),
        "fits": total < vram,
        "kv_calibrated": calibrated,
        "kv_geometric_gib": round(kv_cache_bytes(model, ctx_tokens, k_bits, v_bits) / GIB, 2),
    }


def embedding_bytes(model: dict) -> int:
    """
    词表嵌入层字节数（输入嵌入 + 输出头，若未 tie）。

    默认按 F16 计，但这在**小模型 + 大词表**上会严重高估：
    例如 Qwen2.5-1.5B（vocab 151936 × hidden 1536），嵌入层占全部权重的
    绝大部分，而它的实际量化档位通常是 Q4_K/Q6_K。
    因此支持在 model 配置里声明 `architecture.embedding_bits`，
    未声明时才退回 F16。

    做逐层耗时拆解时必须先把它从总权重里剔出去，
    否则会高估 Transformer 层的单层权重、进而高估单层耗时。
    """
    a = model["architecture"]
    # 实测覆盖：混合量化下输入嵌入与输出头往往用不同档位
    # （如 token_embd=Q4_K、output=Q6_K），单一 bits 值算不准，直接读实测字节。
    if a.get("embedding_bytes_measured"):
        return int(a["embedding_bytes_measured"])
    bits = float(a.get("embedding_bits", 16.0))
    per_table = a["vocab_size"] * a["hidden_size"] * bits / 8.0
    n_tables = 1 if a.get("tie_embeddings") else 2
    return int(per_table * n_tables)


def layer_weight_bytes(model: dict) -> int:
    """单个 Transformer 层的平均权重字节数 = (总权重 - 嵌入层) / 层数。"""
    a = model["architecture"]
    return max(weight_bytes(model) - embedding_bytes(model), 0) // a["n_layer"]


def parse_quant_bits(quant: str) -> float:
    """把 ggml 量化名映射到有效位宽。未知档位按前缀推断。"""
    table = {
        "F32": 32.0, "F16": 16.0, "BF16": 16.0,
        "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K_M": 5.5, "Q5_K_S": 5.5,
        "Q5_0": 5.5, "Q5_1": 6.0, "Q4_K_M": 4.8125, "Q4_K_S": 4.5,
        "Q4_0": 4.5, "Q4_1": 5.0, "Q3_K_M": 3.4375, "Q3_K_S": 3.4375,
        "Q2_K": 2.5625, "IQ4_XS": 4.25, "IQ3_XXS": 3.0625, "IQ2_XXS": 2.0625,
    }
    key = quant.upper()
    if key in table:
        return table[key]
    for k, v in table.items():
        if key.startswith(k.split("_")[0]):
            return v
    return 4.5


# ============================================================================
# PerfConfig：一次运行的完整上下文
# ============================================================================

def _section(raw: dict, key: str) -> dict:
    """
    把 YAML 里的 `<key>: {...}` 段落提升到根层，使 `device.name` 与
    `device.memory`、`device.compute` 处于同一层，便于代码统一访问。

    配置里保留段落写法（可读性好、可加注释），代码侧统一扁平化。
    """
    out = {k: v for k, v in raw.items() if k != key}
    sec = raw.get(key, {})
    if isinstance(sec, dict):
        out.update(sec)
    return out


class PerfConfig:
    """
    加载并持有 model / device / task / plan 四份配置，提供运行目录与校验。

    典型用法：
        cfg = PerfConfig.from_dir("configs", run_id="qwen3.8-27b-decode-opt")
        cfg.mem_budget()          # 显存预算
        cfg.out("baseline.json")  # output/<run_id>/baseline.json
    """

    def __init__(self, model: dict, device: dict, task: dict, plan: dict,
                 run_id: str, root: Path, source_dir: Path):
        self.model = model
        self.device = device
        self.task = task
        self.plan = plan
        self.run_id = run_id
        self.root = root
        self.source_dir = source_dir
        self.outdir = root / "output" / run_id
        #: 非致命配置缺口（如 CPU 画像缺峰值算力），报告会据此标注
        self.warnings: list[str] = []

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_dir(cls, config_dir: str | Path, run_id: str,
                 root: str | Path | None = None) -> "PerfConfig":
        """
        从配置目录加载四份配置。

        文件名按**角色前缀 glob** 解析（`model_*.yaml` / `device_*.yaml` /
        `task_*.yaml` / `plan_*.yaml`），而不是写死具体文件名。这样：
          · 换硬件时 `device_rtx4090.yaml` → `device_cpu_x86.yaml` 无需改代码；
          · 同一目录里放多份同角色配置会**报错**而非静默取第一个，
            避免「改了 A 文件、跑的是 B 文件」这类难查的问题。
        """
        config_dir = Path(config_dir)
        # 项目根 = perfkit 包的父目录（本文件在 <root>/perfkit/config.py）。
        # 刻意不从 config_dir 推 root：配置目录可以嵌套（如
        # configs/qwen2.5-1.5b-cpu/），而 output/ 与 data/ 始终在项目根，
        # 否则产物会散落到配置目录里，且台账路径会算错。
        if root is None:
            root = Path(__file__).resolve().parent.parent
        root = Path(root)

        def _pick(prefix: str) -> Path:
            hits = sorted(p for p in config_dir.glob(f"{prefix}_*.yaml")
                          if not p.name.startswith("."))
            if not hits:
                raise ConfigError(
                    f"配置目录 {config_dir} 下找不到 {prefix}_*.yaml。\n"
                    f"一套完整配置需要四份文件："
                    f"model_*.yaml / device_*.yaml / task_*.yaml / plan_*.yaml")
            if len(hits) > 1:
                raise ConfigError(
                    f"配置目录 {config_dir} 下有 {len(hits)} 个 {prefix}_*.yaml：\n"
                    + "\n".join(f"  - {h.name}" for h in hits)
                    + f"\n每类配置只能有一份（框架按角色前缀 glob 解析）。")
            return hits[0]

        model = _section(load_yaml(_pick("model")), "model")
        device = _section(load_yaml(_pick("device")), "device")
        task = load_yaml(_pick("task"))
        plan = load_yaml(_pick("plan"))
        cfg = cls(model, device, task, plan, run_id, root, config_dir)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """
        交叉校验：配置错误必须在跑测试之前就暴露，而不是跑完才发现口径不对。

        校验分两类：
          硬错误（errs）    → 抛异常，流程无法继续（如架构超参自相矛盾）
          软警告（warns）   → 打印提示，流程继续但报告会标注该缺口
        分开的理由：CPU/NPU 画像天然缺少 GPU 才有的字段（如 CUDA 版本），
        把它当硬错误会让框架无法在非 GPU 后端上运行 —— 这正是要避免的。
        """
        errs: list[str] = []
        warns: list[str] = []
        a = self.model.get("architecture", {})
        for f in ("n_layer", "hidden_size", "n_head", "n_head_kv", "head_dim", "vocab_size"):
            if not a.get(f):
                errs.append(f"model.architecture.{f} 缺失")
        if a.get("hidden_size") and a.get("n_head") and a.get("head_dim"):
            # 标准 Transformer：n_head × head_dim == hidden_size。
            # 少数新架构（如 Qwen3-Next 系）的注意力输出维度不等于 hidden，
            # 由 attn_o 投影回 hidden —— 这类模型必须显式声明 attn_out_dim，
            # 否则本校验会误报。声明后按声明值校验，严格性不变。
            expect = a["n_head"] * a["head_dim"]
            target = a.get("attn_out_dim") or a["hidden_size"]
            if expect != target:
                field = "attn_out_dim" if a.get("attn_out_dim") else "hidden_size"
                errs.append(
                    f"n_head×head_dim={expect} 与 {field}={target} 不一致")
        if a.get("n_head") and a.get("n_head_kv") and a["n_head"] % a["n_head_kv"]:
            errs.append("n_head 必须是 n_head_kv 的整数倍（GQA 分组比）")
        ctx = a.get("max_ctx_deployed", 0)
        native = a.get("max_ctx_native", 0)
        if ctx and native and ctx > native:
            errs.append(f"max_ctx_deployed({ctx}) 超过 max_ctx_native({native})")

        if not self.device.get("memory", {}).get("mem_bandwidth_gbps_peak"):
            errs.append("device.memory.mem_bandwidth_gbps_peak 缺失（roofline 必需）")
        try:
            _mem_total_bytes(self.device)
        except ConfigError as e:
            errs.append(str(e))

        if not self.task.get("kpis"):
            errs.append("task.kpis 为空，无法定义验收口径")

        # ---- 软警告：不阻塞，但会在报告里显式标注缺口 ----
        if peak_compute_tflops(self.device) is None:
            warns.append("设备画像缺少峰值算力，roofline 拐点无法计算"
                         "（请在 compute 段补 fp16_tflops_peak，"
                         "或 CPU 的 cpu_cores + cpu_ghz + simd_flops_per_core_cycle）")
        if not self.device.get("power_thermal", {}).get("tdp_w"):
            warns.append("device.power_thermal.tdp_w 缺失，无法评估能效与热约束")
        if self.device.get("kind") == "gpu" and not self.device.get("compute_capability"):
            warns.append("device.compute_capability 缺失，编译架构标志无法自动生成")

        if warns:
            self.warnings = warns
        if errs:
            raise ConfigError("配置校验失败:\n  - " + "\n  - ".join(errs))

    # ------------------------------------------------------------ 便捷访问
    @property
    def arch(self) -> dict:
        return self.model["architecture"]

    @property
    def backend(self):
        """当前设备对应的硬件后端实现（见 perfkit/backends.py）。"""
        from .backends import get_backend
        if not hasattr(self, "_backend"):
            self._backend = get_backend(self.device)
        return self._backend

    @property
    def mem_total_gib(self) -> float | None:
        v = self.backend.mem_total_bytes()
        return round(v / GIB, 2) if v else None

    def mem_label(self) -> str:
        return self.backend.memory_label

    def out(self, *parts: str) -> Path:
        """output/<run_id>/<parts...> 的路径，自动建父目录。"""
        p = self.outdir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def ledger(self) -> Path:
        """
        实测台账路径。解析顺序：
          1) <config_dir>/ledger.json        —— 该套配置自带的台账（推荐，便于整体搬迁）
          2) <root>/data/reference_measurements.json —— 全局默认台账
        这样多套配置并存时，各自的实测数据互不干扰。
        """
        local = self.source_dir / "ledger.json"
        if local.exists():
            return local
        return self.root / "data/reference_measurements.json"

    def kv_bits(self) -> tuple[float, float]:
        """从 plan 的 selected_bundle 反查 KV 量化位宽，默认 f16/f16（未量化）。"""
        flags = self.selected_bundle().get("startup_flags", [])
        k = flag_value(flags, "--cache-type-k")
        v = flag_value(flags, "--cache-type-v")
        return (parse_quant_bits(k) if k else 16.0,
                parse_quant_bits(v) if v else 16.0)

    def selected_bundle(self) -> dict:
        want = self.plan.get("selected_bundle")
        for b in self.plan.get("bundles", []):
            if b["id"] == want:
                return b
        raise ConfigError(f"plan.selected_bundle={want} 未在 bundles 中找到")

    def mem_budget(self, ctx: int | None = None, parallel: int | None = None) -> dict:
        flags = self.selected_bundle().get("startup_flags", [])
        if ctx is None:
            ctx = flag_value(flags, "--ctx-size", int) or \
                self.arch.get("max_ctx_deployed", 8192)
        if parallel is None:
            parallel = flag_value(flags, "--parallel", int) or 1
        k, v = self.kv_bits()
        return mem_budget(self.device, self.model, ctx, parallel, k, v,
                          include_mmproj=flag_present(flags, "--mmproj"),
                          include_draft=flag_present(flags, "-md"))

    def decode_upper_bound(self, ctx: int = 0) -> float:
        """
        Decode 理论上界。ctx=0 表示「不计 KV 读」，与台账 §1.1 的
        1008/16.4 = 61.4 t/s 同口径；给出 ctx 则计入每步 KV 读写。
        """
        k, v = self.kv_bits()
        return theoretical_decode_tps(self.device, weight_bytes(self.model),
                                      ctx, self.model, k, v)

    def decode_upper_bounds(self) -> dict:
        """上界的两种口径并列，避免报告里出现口径不一致的数字。"""
        ctx = self.arch.get("max_ctx_deployed", 8192)
        return {
            "no_kv_tps": round(self.decode_upper_bound(0), 2),
            "at_ctx_tps": round(self.decode_upper_bound(ctx), 2),
            "ctx": ctx,
        }

    def resolve_flags(self) -> list[str]:
        """把 bundle 的 startup_flags 里的 {draft}/{mmproj} 占位符换成真实路径。"""
        paths = {}
        for art in self.model.get("artifacts", []):
            base = os.path.dirname(self.model.get("file", ""))
            paths[art["role"]] = os.path.join(base, art["file"])
        mapping = {
            "{draft}": paths.get("draft_mtp", ""),
            "{mmproj}": paths.get("mmproj", ""),
            "{model}": self.model.get("file", ""),
        }
        out = []
        for f in self.selected_bundle().get("startup_flags", []):
            for k, v in mapping.items():
                f = f.replace(k, v)
            out.append(f)
        return out

    def startup_command(self) -> str:
        """生成可直接执行的 llama-server 启动命令（多行，便于粘贴）。"""
        b = self.selected_bundle()
        lines = [
            "./llama-server",
            f'  -m {self.model.get("file")}',
        ]
        for f in self.resolve_flags():
            lines.append(f"  {f}")
        if not any("--host" in x for x in lines):
            lines += ["  --host 0.0.0.0", "  --port 8000",
                      f'  --alias {self.model["name"].lower()}']
        return " \\\n".join(lines)

    def kpi_by_key(self, key: str) -> dict | None:
        for k in self.task.get("kpis", []):
            if k["key"] == key:
                return k
        return None
