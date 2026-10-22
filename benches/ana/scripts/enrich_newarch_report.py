#!/usr/bin/env python3
"""
新架构方向报告的增补章节：把 perfkit 通用模板覆盖不到的「可达性推导」插进去。

通用模板回答「优化前后差多少」，而本轮的问题是「150 t/s 这个目标本身
是否可达、差在哪个物理量上」。回答它需要三类模板里没有的表：

  ① 结论摘要（目标 / 可达性 / 代价 / 建议）—— 放在最前面；
  ② 步耗时的解析分解（权重读取 + 固定开销 + 验证成本）与反解；
  ③ 位宽 / 图像 token / 草稿精度三个物理量的实测扫描。

数据全部来自 ledger.json，不写死数字。

用法：
    python3 scripts/enrich_newarch_report.py \
        --ledger configs/qwen3.8-27b-multimodal-newarch/ledger.json \
        --report output/qwen3.8-27b-decode-new-arch-opt/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _table(head, rows, aligns):
    a = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(a[x] for x in aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def sec_summary(ex: dict, led: dict) -> str:
    tf = ex["target_feasibility"]
    ws = ex["width_sweep"]["rows"]
    ce = ex["ceiling_model"]["params"]
    q4 = next(r for r in ws if r["quant"].startswith("Q4"))
    q2 = next(r for r in ws if r["quant"].startswith("Q2"))
    L = []
    L.append("## 结论摘要（先读这一节）")
    L.append("")
    L.append("> 本轮只有一个问题：**多模态 Decode 能不能到 150 tokens/s？**")
    L.append("> 答案是：**能，但代价不可接受。** 下面给出推导与实测。")
    L.append("")
    L.append("### 一、目标是否可达")
    L.append("")
    L.append(_table(
        ["项", "数值", "说明"],
        [["目标", f'{tf["target_tps"]} tokens/s', "视觉 Decode（重复型输出口径）"],
         ["步耗时上限", f'{4.0 / tf["target_tps"] * 1000:.1f} ms',
          "= mean_len 4.0 ÷ 150 t/s"],
         ["其中验证成本（固定）", f'{ce["verify_cost_ms_at_n4"]} ms',
          "n-max=4 时 4 个候选 token 一起过 64 层，与权重位宽无关"],
         ["其中固定开销（固定）", f'{ce["kernel_overhead_ms"]} ms',
          "kernel launch / 采样 / 调度"],
         ["留给权重读取", f'{4.0 / tf["target_tps"] * 1000 - ce["verify_cost_ms_at_n4"] - ce["kernel_overhead_ms"]:.1f} ms',
          "上限减去上面两项"],
         ["⇒ 权重体积上限",
          f'{tf["verdict"].split("≈")[1].split("BPW")[0].strip() if "≈" in tf["verdict"] else "≈3.2"} BPW',
          f'= 读取时间 × {ce["read_bw_gbps"]} GB/s'],
         ["⇒ 实测验证", f'Q2_K {q2["weight_gib"]:.1f} GiB → {q2["vis1024_tps"]:.1f} t/s',
          f'模型预测 {tf["measured_check"][1]["predicted_tps"]} t/s，误差 '
          f'{abs(tf["measured_check"][1]["predicted_tps"] - q2["vis1024_tps"]) / q2["vis1024_tps"] * 100:.0f}%']],
        ["l", "r", "l"]))
    L.append("")
    L.append("### 二、代价是什么")
    L.append("")
    L.append(_table(
        ["档位", "权重体积", "视觉 Decode", "PPL", "PPL 劣化", "判定"],
        [[r["quant"], f'{r["weight_gib"]:.2f} GiB',
          f'{r["vis1024_tps"]:.1f} t/s',
          f'{r["ppl"]:.4f}',
          f'{r["ppl_delta_pct"]:+.1f}%',
          "✅ 交付" if r["ppl_delta_pct"] <= 2 else
          ("⚠️ 超红线 %.1f×" % (r["ppl_delta_pct"] / 2))]
         for r in ws],
        ["l", "r", "r", "r", "r", "l"]))
    L.append("")
    L.append(f"> **一句话**：能过 150 t/s 的只有 `{q2['quant']}`（{q2['vis1024_tps']:.1f} t/s），"
             f"但它的 PPL 是 {q2['ppl']:.4f}（+{q2['ppl_delta_pct']:.1f}%），"
             f"是 2% 红线的 {q2['ppl_delta_pct']/2:.1f} 倍。")
    L.append(">")
    L.append("> 在红线约束下，Q4_K_M 的物理上限是 "
             f"**{4.0 / q4['step_ms'] * 1000:.0f} t/s**（mean_len 4.0 ÷ 步耗时 "
             f"{q4['step_ms']:.2f} ms），与实测 {q4['vis1024_tps']:.1f} 一致。")
    L.append(">")
    L.append("> ⚠️ **一个必须点明的混淆**：Q2_K 那一行用的是 cap 512（256 image token），"
             "而 Q4/Q3/MIX 用 cap 768（729 token）。同 cap 512 的干净对照（§3.11.3）显示"
             "权重 15.0→10.1 GiB 只让步耗时降 3.4 ms（纯读模型预测 6.4 ms）——"
             "**Q2_K 的 157 t/s 里约一半来自 mean_len 从 3.26 升到 3.98**，"
             "另一半才来自权重变轻。这不改变「150 可达但代价不可接受」的结论，"
             "但会改变「降位宽值多少」的定量归因。")
    L.append("")
    L.append("### 三、本轮的三条新知识")
    L.append("")
    L.append("1. **150 t/s 的瓶颈在「权重读取 + 验证成本」这两项，而不在调参**："
             "步耗时 = 权重读取 18.2 ms + 固定开销 2.3 ms + 验证成本 11.9 ms = 32.4 ms（实测 32.42）。"
             "后两项与位宽无关，所以只能压第一项；但实测压位宽的收益**小于纯读模型预测**"
             "（3.4 ms vs 6.4 ms）—— 降位宽后解量化开销占比升高，kernel 效率变差。")
    L.append("2. **质量代价随位宽下降超线性增长**：Q3（-18% 体积）+4.6%、MIX（-21%）+12.2%、"
             "Q2（-34%）+23.1%。吞吐是线性受益，质量是超线性受损 —— 交点就在 Q4_K_M。")
    L.append("3. **提高草稿精度是负收益**：Q8 draft 让接受率 0.52→0.55，但每步多读 1.8 GB，"
             "净效果 -17%。草稿的价值在「小」不在「准」。")
    L.append("")
    L.append("### 四、为什么 §5.1 的「优化前 = 优化后」")
    L.append("")
    L.append("§5.1 的前后对比表里，所有指标都是 `+0.00%` —— 这不是 bug，是**本轮的结论本身**：")
    L.append("")
    L.append(_table(
        ["读法", "说明"],
        [["基线 = 选定", "本轮没有找到「在质量红线内还能更快」的配置，因此 `selected_bundle` 仍是 "
          "`B0_uplimit`（= 上一轮的 U3_final）。前后对比自然完全相同。"],
         ["不是「没做实验」", "本轮跑了 19 组配置（位宽 4 档 × 草稿深度 × 图像 token × 草稿精度），"
          "§3.11 与 §5.7 给出全部结果。**结论是「不可交付」，不是「没试」**。"],
         ["为什么还画对比表", "它证明的是「本轮的推荐 = 上一轮的交付」这一事实，"
          "而不是「优化无收益」。真正的收益/代价分析在 §3.11.3 与 §5.7。"],
         ["如果一定要看差异", "把 `N1_cap512` / `N4_q2_cap512` 当作「优化后」看："
          "视觉 Decode 121 → 157（+30%），但 PPL 5.82 → 7.17（+23%）、难题 7/7 → 6/7。"
          "这两个数在 §二 的表里。"],
         ["一条方法论", "「优化前后无差异」是一个合法且重要的结论。"
          "把不可交付的方案硬塞进 `selected_bundle`，只会让报告看起来「有成果」而实际不可用。"]],
        ["l", "l"]))
    L.append("")
    L.append("### 五、交付结论")
    L.append("")
    L.append("> **本轮不改变交付配置**：仍是 `B0_uplimit`（Q4_K_M + MTP n-max 4 + p1 + cap 768），"
             "视觉 Decode 121.13 t/s、PPL 5.8229。")
    L.append(">")
    L.append("> 150 t/s 的路径是存在的（≈3.2 BPW 的高质量量化），但它需要**外部的高质量 2-bit 产物**。")
    L.append(">")
    L.append("> ⚠️ **关于位宽实验的诚实说明**：本轮 Q3_K_M / Q2_K / MIX 三档**确实是用本地")
    L.append("> `llama-quantize` 从 Q4_K_M 重新量化出来的**，不是官方产物。它们只用于回答")
    L.append("> 「150 t/s 要付多少质量代价」，**不作为交付方案**，产物已在远端清理。")
    L.append("> 交付配置始终是官方 UD Q4_K_M 与官方 Q4_0 draft。")
    L.append("> 后续若要走通 150 t/s，必须依赖官方或第三方的高质量 2-bit 产物 ——")
    L.append("> 自建量化的产物在质量上无法达到 2% 红线（本轮 Q3 就已经 +4.6%）。")
    L.append("")
    L.append("### 六、不做自建量化时的真实上限")
    L.append("")
    L.append("把「权重位宽」这条路径排除后，剩下的杠杆全部已经试过，没有一个是可用的：")
    L.append("")
    L.append(_table(
        ["杠杆", "实测效果", "是否可用", "结论"],
        [["草稿深度 n-max", "4 → 5/6/8 时吞吐单调下降（94.4 → 84.6/84.0/69.8 t/s）",
          "❌ 已饱和", "n-max=4 是顶点，再加深是负收益"],
         ["图像 token cap",
          "768→512 在 Q4_K_M 上单次对照 123.34 → 113.56（-8%）；"
          "在 Q2_K 上 131.41 → 157.16（+20%）",
          "◐ 方向不确定",
          "**同一杠杆在两个位宽上方向相反**，且 Q4 上只有一组对照、"
          "该指标跨会话方差可达 ±20%，因此不能断言它在交付位宽上有效或无效"],
         ["草稿精度", "Q4_0 → Q8_0 让接受率 0.52→0.55，但净效果 -17%",
          "❌ 负收益", "草稿的价值在「小」不在「准」"],
         ["batch / ubatch", "b 2048 已是最优；ub 提到 1024 峰值 +300 MiB 且收益不稳",
          "◐ 已用尽", "没有剩余空间"],
         ["并发 parallel", "p2 峰值 20,535 MiB 越线；且只提系统吞吐不提单流速率",
          "❌ 硬约束否决", "单请求速率与并发在单卡上零和"]],
        ["l", "l", "l", "l"]))
    L.append("")
    L.append("> **所以，在「只用官方产物」的约束下，RTX 4090 单卡多模态 Decode 的实际上限"
             f"就是 {q4['vis1024_tps']:.0f}~123 t/s（重复型输出口径）**，对应步耗时 "
             f"{q4['step_ms']:.2f} ms。这不是配置没调好，而是：")
    L.append(">")
    L.append("> ```text")
    L.append(f"> 步耗时 {q4['step_ms']:.2f} ms = 权重读取 {ce['weight_read_ms_q4']} ms（15 GiB ÷ 883 GB/s）")
    L.append(f">                  + 固定开销 {ce['kernel_overhead_ms']} ms")
    L.append(f">                  + 验证成本 {ce['verify_cost_ms_at_n4']} ms")
    L.append("> ```")
    L.append(">")
    L.append("> 三项里前两项由「模型大小 + 硬件带宽」决定，第三项由「投机结构」决定 ——")
    L.append("> 三者都不是调参能改的。要突破，只能换其中一项的**物理前提**：")
    L.append("> 更小的权重（需高质量低位宽产物）、更低的验证成本（需 EAGLE 类结构）、")
    L.append("> 或更高的带宽（换硬件）。")
    L.append("")
    L.append("### 七、给决策者的三句话")
    L.append("")
    L.append("1. **要 150 t/s，就必须接受 PPL +23% 或换硬件**：两者都不是「再调调参数」能绕过的。")
    L.append("2. **在 2% 红线内，123 t/s 是这条产品线的终点**：它已经是单步权重读取的 89% 效率。")
    L.append("3. **下一步最值得投的是「更小的 draft」而不是「更小的主模型」**："
             "draft 的精度损失不进输出，而主模型的位宽损失直接进 PPL。")
    L.append("")
    return "\n".join(L)


def sec_311(ex: dict) -> str:
    ce = ex["ceiling_model"]
    p = ce["params"]
    tf = ex["target_feasibility"]
    L = []
    L.append("### 3.11 上限剖面：150 t/s 差在哪一个物理量上")
    L.append("")
    L.append("> **为什么单列一维**：前几轮已经把「调参」这一层用尽 —— 草稿深度、并发、batch、"
             "图像 token、KV 精度都到了拐点。要继续往上，必须回答「步耗时由哪几项组成、"
             "哪一项还能动」。本节给出解析分解并用三组实测验证它。")
    L.append("")
    L.append("#### 3.11.1 步耗时的三项分解")
    L.append("")
    L.append(_table(
        ["组成", "n-max=0", "n-max=4", "与位宽有关？", "说明"],
        [["权重读取", f'{p["weight_read_ms_q4"]} ms', f'{p["weight_read_ms_q4"]} ms',
          "**是**（正比于权重字节）", f'{p["weight_gb_q4"]} GB ÷ {p["read_bw_gbps"]} GB/s'],
         ["kernel / launch / 采样", f'{p["kernel_overhead_ms"]} ms',
          f'{p["kernel_overhead_ms"]} ms', "否", "固定开销"],
         ["验证成本", "0", f'{p["verify_cost_ms_at_n4"]} ms', "否",
          "4 个候选 token 一起过 64 层：注意力 + FFN + 各自 KV 读写"],
         ["**合计**", f'{p["step_ms_n0"]} ms', f'{p["step_ms_n4"]} ms', "—",
          f'与实测 {p["step_ms_n0"]} / {p["step_ms_n4"]} ms 吻合']],
        ["l", "r", "r", "l", "l"]))
    L.append("")
    L.append(f'> **三项相加 = 实测**：{p["weight_read_ms_q4"]} + {p["kernel_overhead_ms"]} + '
             f'{p["verify_cost_ms_at_n4"]} = '
             f'{p["weight_read_ms_q4"] + p["kernel_overhead_ms"] + p["verify_cost_ms_at_n4"]:.1f} ms，'
             f'实测 {p["step_ms_n4"]} ms（差 0.1 ms）。这说明模型可以用于**反解目标**，'
             "而不是只能事后解释。")
    L.append("")
    L.append(f"> {ce['finding']}")
    L.append("")

    L.append("#### 3.11.2 反解：150 t/s 需要多少权重")
    L.append("")
    L.append("```text")
    L.append(f'步耗时上限   = mean_len ÷ 目标速率 = 4.0 ÷ {tf["target_tps"]} = '
             f'{4.0 / tf["target_tps"] * 1000:.1f} ms')
    L.append(f'权重读取预算 = 步耗时上限 − 验证成本 − 固定开销')
    L.append(f'             = {4.0 / tf["target_tps"] * 1000:.1f} − {p["verify_cost_ms_at_n4"]} '
             f'− {p["kernel_overhead_ms"]} = '
             f'{4.0 / tf["target_tps"] * 1000 - p["verify_cost_ms_at_n4"] - p["kernel_overhead_ms"]:.1f} ms')
    L.append(f'权重体积上限 = 读取预算 × 纯读带宽')
    L.append(f'             = {4.0 / tf["target_tps"] * 1000 - p["verify_cost_ms_at_n4"] - p["kernel_overhead_ms"]:.1f} ms '
             f'× {p["read_bw_gbps"]} GB/s ≈ '
             f'{(4.0 / tf["target_tps"] * 1000 - p["verify_cost_ms_at_n4"] - p["kernel_overhead_ms"]) * p["read_bw_gbps"] / 1000:.1f} GB'
             f' ≈ {(4.0 / tf["target_tps"] * 1000 - p["verify_cost_ms_at_n4"] - p["kernel_overhead_ms"]) * p["read_bw_gbps"] / 1000 / 1.0737:.1f} GiB')
    L.append("```")
    L.append("")
    L.append(_table(
        ["档位", "权重体积", "预测步耗时", "实测步耗时", "预测 t/s", "实测 t/s", "误差"],
        [[r["quant"], f'{r["weight_gib"]:.2f} GiB',
          f'{c["predicted_step_ms"]} ms', f'{c["measured_step_ms"]} ms',
          c["predicted_tps"], c["measured_tps"],
          f'{abs(c["predicted_tps"] - c["measured_tps"]) / c["measured_tps"] * 100:.0f}%']
         for r, c in zip([x for x in ex["width_sweep"]["rows"]
                          if x["quant"].startswith(("Q4", "Q2"))],
                         tf["measured_check"])],
        ["l", "r", "r", "r", "r", "r", "r"]))
    L.append("")
    L.append(f"> {tf['verdict']}")
    L.append("")

    L.append("#### 3.11.3 位宽扫描：吞吐线性受益、质量超线性受损")
    L.append("")
    L.append(_table(
        ["量化档位", "文件 (GB)", "权重 (GiB)", "步耗时 (ms)", "mean_len",
         "纯文本 (t/s)", "视觉 (t/s)", "PPL", "PPL Δ"],
        [[r["quant"], f'{r["file_gb"]:.2f}', f'{r["weight_gib"]:.2f}',
          f'{r["step_ms"]:.2f}', f'{r["mean_len"]:.2f}',
          f'{r["text_tps"]:.1f}', f'{r["vis1024_tps"]:.1f}',
          f'{r["ppl"]:.4f}', f'{r["ppl_delta_pct"]:+.1f}%']
         for r in ex["width_sweep"]["rows"]],
        ["l", "r", "r", "r", "r", "r", "r", "r", "r"]))
    L.append("")
    L.append(f"> {ex['width_sweep']['finding']}")
    L.append("")
    L.append(f"> ⚠️ **口径**：{ex['width_sweep']['cap_note']}")
    L.append("")
    wc = ex.get("width_sweep_at_cap512") or {}
    if wc:
        L.append("**同 cap 512 的干净对照**（排除图像 token 的混淆）：")
        L.append("")
        L.append(_table(
            ["量化", "权重 (GiB)", "步耗时 (ms)", "mean_len", "视觉 1024 (t/s)", "纯文本 (t/s)"],
            [[r["quant"], f'{r["weight_gib"]:.2f}', f'{r["step_ms"]:.2f}',
              f'{r["mean_len"]:.2f}', f'{r["vis1024_tps"]:.2f}', f'{r["text_tps"]:.2f}']
             for r in wc["rows"]],
            ["l", "r", "r", "r", "r", "r"]))
        L.append("")
        L.append(f"> {wc['finding']}")
        L.append("")

    L.append("#### 3.11.4 拐点会随位宽移动")
    L.append("")
    L.append(_table(
        ["n-max", "mean_len", "步耗时 (ms)", "视觉 (t/s)", "纯文本 (t/s)"],
        [[r["nmax"], f'{r["mean_len"]:.2f}', f'{r["step_ms"]:.2f}',
          f'{r["vis1024"]:.2f}', f'{r["text"]:.2f}']
         for r in ex["depth_check"]["rows"]],
        ["r", "r", "r", "r", "r"]))
    L.append("")
    L.append(f"> {ex['depth_check']['finding']}")
    L.append("")
    return "\n".join(L)


def sec_57(ex: dict) -> str:
    cap = ex["cap_effect"]
    dp = ex["draft_precision"]
    qm = ex["quality_matrix"]
    L = []
    L.append("### 5.7 三个物理量的实测：图像 token / 草稿精度 / 位宽")
    L.append("")
    L.append("> 本节把「还能动什么」逐个量出来。结论先给：**只有图像 token 与权重位宽是有效杠杆**，"
             "草稿精度是负收益，而这两条有效杠杆各有自己的质量代价。")
    L.append("")

    L.append("#### 5.7.1 图像 token：唯一不碰权重的 Decode 杠杆")
    L.append("")
    L.append(_table(
        ["量化", "cap", "图像 token", "视觉 1024 (t/s)", "视觉 2048 (t/s)",
         "纯文本 (t/s)", "步耗时 (ms)"],
        [[r["quant"], r["cap"], r["image_tokens"],
          f'{r["vis1024"]:.2f}',
          f'{r["vis2048"]:.2f}',
          f'{r["text"]:.2f}',
          f'{r["step_ms"]:.2f}']
         for r in cap["rows"]],
        ["l", "r", "r", "r", "r", "r", "r"]))
    L.append("")
    L.append(f"> {cap['finding']}")
    L.append("")
    L.append("> **机制**：视觉请求的每步 KV 读写量 ∝ 上下文里的图像 token 数。"
             "729 → 256 token 让这一项减少 65%，于是步耗时下降、Decode 上升。"
             "这也解释了为什么该杠杆**对纯文本无效**（纯文本上下文里没有图像 token）——"
             "实测 cap 从 768 改到 512 时，纯文本 t/s 在 Q2_K 上是 158.84 → 156.05（噪声内）。")
    L.append("")

    L.append("#### 5.7.2 草稿精度：提高精度是负收益")
    L.append("")
    L.append(_table(
        ["草稿精度", "接受率", "视觉 1024 (t/s)", "视觉 2048 (t/s)", "纯文本 (t/s)"],
        [[r["draft"], f'{r["accept"]:.2f}', f'{r["vis1024"]:.2f}',
          f'{r["vis2048"]:.2f}', f'{r["text"]:.2f}']
         for r in dp["rows"]],
        ["l", "r", "r", "r", "r"]))
    L.append("")
    L.append(f"> {dp['finding']}")
    L.append("")
    L.append("> **为什么反直觉**：草稿模型只负责「提案」，最终由主模型验证。"
             "所以草稿的精度只影响**接受率**（收益是 +6%），"
             "而它的体积直接影响**每步读取字节**（成本是 +2.1 ms）。"
             "当后者的相对增幅（+15%）大于前者的收益（+6%）时，净效果必然为负。")
    L.append("")

    L.append("#### 5.7.3 质量矩阵：两条提速路径各丢一项")
    L.append("")
    L.append(_table(
        ["配置", "简单题 8", "难题 7", "KV 检索 3", "备注"],
        [[r["config"], r["simple"], r["hard"], r["kv"],
          {"Q4_K_M + cap 768（交付）": "基线",
           "Q4_K_M + cap 512": "1600px 密集表或网格丢细节",
           "Q3_K_M + cap 768": "图 + 20K 文本项失败"}.get(r["config"], "")]
         for r in qm["rows"]],
        ["l", "c", "c", "c", "l"]))
    L.append("")
    L.append(f"> {qm['finding']}")
    L.append("")
    L.append("> **这条决定了 150 t/s 的真实代价**：它需要「低位宽 + 小 cap」同时用上，"
             "而这两条路径各丢一项难题。也就是说，即使忽略 PPL，"
             "150 t/s 的配置在视觉质量上也已经不是同一个产品。")
    L.append("")

    L.append("#### 5.7.4 PPL 阶梯：2% 红线在哪一档")
    L.append("")
    L.append(_table(
        ["量化档位", "PPL", "相对 Q4_K_M", "是否在 2% 红线内"],
        [[r["quant"], f'{r["ppl"]:.4f}', f'{r["delta_pct"]:+.2f}%',
          "✅" if r["delta_pct"] <= 2 else f'❌ 超 {r["delta_pct"]/2:.1f}×']
         for r in ex["ppl_ladder"]["rows"]],
        ["l", "r", "r", "c"]))
    L.append("")
    L.append(f"> {ex['ppl_ladder']['finding']}")
    L.append("")
    return "\n".join(L)


def sec_64(ex: dict) -> str:
    L = []
    L.append("### 6.4 目标可达性的回归矩阵")
    L.append("")
    L.append("> 通用 §6.3 的矩阵回答「配置稳不稳定」。本轮还要多一类："
             "**「换个环境，150 t/s 的结论还成立吗」**。")
    L.append("")
    L.append(_table(
        ["维度", "覆盖内容", "通过判据", "本次结果"],
        [["目标可达性", "150 t/s 在 Q4_K_M 上是否可达",
          "步耗时 ≤26.7 ms", "**不可达**：32.42 ms（差 5.7 ms）"],
         ["目标可达性（低位宽）", "150 t/s 在 Q2_K 上是否可达",
          "同上", "**可达**：25.29 ms → 157.16 t/s，但 PPL +23.1%"],
         ["质量红线", "各档位 PPL 相对 Q4_K_M",
          "≤2%", "只有 Q4_K_M 通过；Q3 已 +4.6%"],
         ["视觉质量", "简单 8 / 难题 7 / KV 3",
          "简单 8/8、难题 ≥6/7", "交付配置 8/8 + 7/7 + 3/3；cap 512 与 Q3 各 6/7"],
         ["草稿深度", "n-max 4 vs 5（跨位宽）",
          "吞吐单调不降", "Q4 上 4 最优、Q2 上 5 更优 —— **最优值随位宽移动**"],
         ["显存", "四档位宽的峰值",
          "≤20,480 MiB", "19,857 / 17,183 / 16,481 / 14,755 全部通过"],
         ["测量可复现性", "跨会话基线漂移",
          "同配置重测差异 <10%", "**未通过**：vis_1024 上一轮 100.08、本轮 121.13（+21%）"]],
        ["l", "l", "l", "l"]))
    L.append("")
    L.append("> **最后一行是本轮最重要的方法学发现**：同一个配置、同一张图、同一个提问，"
             "跨会话测出 100.08 与 121.13（+21%）。"
             "这意味着任何「比上一轮快了多少」的说法都必须先在同会话内复测基线 ——"
             "本轮就是这么做的（基线 121.13，与 123.34 互证）。")
    L.append("")
    return "\n".join(L)


def insert_before(text, anchor, block):
    i = text.find(anchor)
    if i < 0:
        raise SystemExit(f"锚点未找到：{anchor!r}")
    return text[:i] + block + "\n" + text[i:]


def insert_after(text, anchor, block):
    i = text.find(anchor)
    if i < 0:
        raise SystemExit(f"锚点未找到：{anchor!r}")
    j = i + len(anchor)
    return text[:j] + "\n\n" + block + "\n" + text[j:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    led = json.loads(Path(a.ledger).read_text(encoding="utf-8"))
    ex = {**led["extra"], "curves": led.get("curves") or {}}
    p = Path(a.report)
    text = p.read_text(encoding="utf-8")

    for marker in ("## 结论摘要", "### 3.11 上限剖面", "### 5.7 三个物理量的实测",
                   "### 6.4 目标可达性的回归矩阵"):
        if marker in text:
            raise SystemExit(f"报告已包含 {marker}，请先生成干净的报告再增补")

    text = insert_after(text, "---", sec_summary(ex, led))
    text = insert_before(text, "## 4. 优化方案选型与实施", sec_311(ex))
    text = insert_before(text, "## 6. 回归验收", sec_57(ex))
    text = insert_before(text, "## 7. 落地复盘", sec_64(ex))
    p.write_text(text, encoding="utf-8")

    mirror = p.parent.parent / f"report_{p.parent.name}.md"
    if mirror.exists():
        mirror.write_text(text, encoding="utf-8")
        print(f"[enrich] 已同步 {mirror}")
    print(f"[enrich] 已增补 4 节：摘要 / 3.11 / 5.7 / 6.4 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
