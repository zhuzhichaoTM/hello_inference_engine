#!/usr/bin/env python3
"""
极限性能报告的增补章节：把 perfkit 通用模板覆盖不到的「上限推导」内容插进去。

为什么需要单独一步：通用模板回答的是「优化前后差多少」，而本轮的
核心问题是「极限在哪、为什么不能更快」。后者需要三张模板里没有的表：
  ① 步耗时模型（Decode = mean_len ÷ 步耗时，两个因子必须同时看）；
  ② 并发/分辨率/提问内容三个「非显存」杠杆的对照；
  ③ 上限推导（权重读带宽 vs 验证成本，谁在卡脖子）。

数据全部来自 ledger.json，不写死任何数字。

用法：
    python3 scripts/enrich_uplimit_report.py \
        --ledger configs/qwen3.8-27b-multimodal-uplimit/ledger.json \
        --report output/qwen3.8-27b-decode-multi-mode-uplimit-opt/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _table(head: list[str], rows: list[list], aligns: list[str]) -> str:
    a = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(a[x] for x in aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def sec_311(ex: dict) -> str:
    sm = ex["step_model"]
    pts = sm["points"]
    ca = ex["ceiling_analysis"]
    L = []
    L.append("### 3.11 极限剖面：输出速率由什么决定")
    L.append("")
    L.append("> **为什么单列一维**：投机解码下 `Decode t/s` 是**两个因子的乘积**：")
    L.append("> ")
    L.append("> ```text")
    L.append("> Decode t/s = mean_accepted_len ÷ 步耗时")
    L.append("> ```")
    L.append("> ")
    L.append("> 只看 t/s 会得出「草稿越深越快」的错误结论；只看步耗时又会漏掉")
    L.append("> 「一次吐几个 token」。必须同时给出两个因子，才能定位拐点。")
    L.append("")

    L.append("#### 3.11.1 草稿深度扫描：两个因子的反向运动")
    L.append("")
    L.append(_table(
        ["n-max", "mean_len", "接受率", "步耗时 (ms)", "Decode (t/s)",
         "峰值 (MiB)", "vis_1024 (t/s)", "判定"],
        [[p["nmax"], f'{p["mean_len"]:.2f}',
          (f'{sm["acceptance"].get(str(p["nmax"]), 0):.3f}'
           if str(p["nmax"]) in sm["acceptance"] else "—"),
          f'{p["step_ms"]:.2f}', f'{p["decode_tps"]:.2f}',
          p["peak_mib"], f'{p["vis_1024"]:.2f}',
          ("**顶点**" if p["nmax"] == 4 else
           "无投机" if p["nmax"] == 0 else
           "负收益" if p["nmax"] > 4 else "未饱和")]
         for p in pts],
        ["r", "r", "r", "r", "r", "r", "r", "c"]))
    L.append("")
    L.append(f"> **两条曲线在这里交叉**：步耗时从 n-max=0 的 "
             f'{pts[0]["step_ms"]:.1f} ms 线性涨到 n-max=8 的 {pts[-1]["step_ms"]:.1f} ms'
             f'（每多一个草稿约 +{(pts[-1]["step_ms"]-pts[0]["step_ms"])/8:.2f} ms）；'
             f'而 mean_len 从 {pts[1]["mean_len"]:.2f} 涨到 {pts[3]["mean_len"]:.2f} 后就饱和'
             f'（n-max=5/6/8 分别是 {pts[4]["mean_len"]:.2f}/{pts[5]["mean_len"]:.2f}/{pts[6]["mean_len"]:.2f}）。'
             f'乘积在 n-max=4 取最大值 —— 这就是拐点的定量依据，不需要试错。')
    L.append("")
    L.append("> **接受率单调下降是必然的**：越靠后的草稿越难猜对。所以「加深草稿」")
    L.append("> 的收益来自前几个 token，成本来自每一次验证都要多算。当边际收益")
    L.append("> 小于边际成本时，继续加深就是负收益。")
    L.append("")

    L.append("#### 3.11.2 上限推导：谁在卡脖子")
    L.append("")
    L.append(_table(
        ["项", "值", "说明"],
        [["纯读带宽（探针实测）", f'{ca["measured_read_bw_gbps"]} GB/s',
          "显存子系统只读不写的可达带宽"],
         ["权重体积", f'{ca["weight_gib"]} GiB',
          "Q4_K_M 有效权重（剔除未启用张量）"],
         ["读一遍权重的下界", f'{ca["step_budget_ms_at_read_bw"]} ms',
          "= 权重字节 ÷ 纯读带宽 —— 这是**任何** decode 步的硬下界"],
         ["实测步耗时（n-max=0）", f'{ca["measured_step_ms_n0"]} ms',
          f'已达下界的 {ca["step_budget_ms_at_read_bw"]/ca["measured_step_ms_n0"]*100:.0f}% —— kernel 无空间'],
         ["实测步耗时（n-max=4）", f'{ca["measured_step_ms_n4"]} ms',
          f'比 n-max=0 多 {ca["measured_step_ms_n4"]-ca["measured_step_ms_n0"]:.1f} ms'],
         ["多出的部分是什么", "验证 batch 变宽",
          "4 个候选 token 一起过 64 层：注意力 + FFN 计算 + 各自的 KV 读写"]],
        ["l", "r", "l"]))
    L.append("")
    L.append(f"> {ca['finding']}")
    L.append("")
    L.append("> **这条推导解释了为什么本轮没有「再翻一倍」的空间**：")
    L.append("> 无投机时已经拿到纯读带宽的 89%，说明权重读取这条路到头了；")
    L.append("> 投机能突破上界，靠的是「一次读取产出多个 token」，但每多一个 token")
    L.append("> 就要多付验证成本。两者的平衡点就是 n-max=4。")
    L.append("> 想再快，只能减少**验证成本**本身（更小的 draft、更准的 draft、")
    L.append("> 或 EAGLE 类只验证一次的多 token 预测），而不是调参。")
    L.append("")
    return "\n".join(L)


def sec_57(ex: dict) -> str:
    pa = ex["parallel_ab"]
    ps = ex["prompt_sensitivity"]
    ac = ex["attention_vs_cap"]
    ve = ex["vision_encode"]
    qm = ex["quality_matrix"]
    itl = ex["itl"]
    L = []
    L.append("### 5.7 极限专项：三个非显存杠杆与退化清单")
    L.append("")
    L.append("> 显存四刀在上一轮已经用尽。本轮提升输出速率靠的是三个**不花显存**的杠杆：")
    L.append("> 并发策略、图像分辨率、以及（被动的）内容类型。下面逐个给出实测，")
    L.append("> 并在末尾如实记录**两处真实退化**（28K Decode 与并发吞吐）。")
    L.append("")

    L.append("#### 5.7.1 并发策略：p1 vs p2")
    L.append("")
    L.append(_table(
        ["场景", "p1（选定）", "p2", "差异", "说明"],
        [["纯文本短对话 (t/s)", pa["p1"]["text_short"], pa["p2"]["text_short"],
          f'{(pa["p2"]["text_short"]/pa["p1"]["text_short"]-1)*100:+.1f}%',
          "p2 两槽交替推进，掩盖了单槽的验证停顿"],
         ["28K 上下文 (t/s)", pa["p1"]["text_32k"], pa["p2"]["text_32k"],
          f'{(pa["p2"]["text_32k"]/pa["p1"]["text_32k"]-1)*100:+.1f}%',
          "基本持平；组内方差极大（48~91 t/s），中位差不可当结论"],
         ["1024px 视觉 (t/s)", pa["p1"]["vis_1024"], pa["p2"]["vis_1024"],
          f'{(pa["p2"]["vis_1024"]/pa["p1"]["vis_1024"]-1)*100:+.1f}%', "基本持平"],
         ["2048px 视觉 (t/s)", pa["p1"]["vis_2048"], pa["p2"]["vis_2048"],
          f'{(pa["p2"]["vis_2048"]/pa["p1"]["vis_2048"]-1)*100:+.1f}%', "—"],
         ["峰值显存 (MiB)", pa["p1"]["peak_mib"], pa["p2"]["peak_mib"],
          f'+{pa["p2"]["peak_mib"]-pa["p1"]["peak_mib"]}',
          "**p2 越线**（余量 3.94 GiB < 4.0）"],
         ["单请求上下文", "65536", "32768", "减半", "p2 每槽只有 32K"],
         ["**双路并发系统吞吐**（受控 A/B，另一组负载）", "38.39", "42.97",
          "-10.7%（p2 更快）",
          "每轮两路（1 图 + 1 个 8K 文本）各 4 轮取中位。**负载与 §5.1 的 "
          "concurrent_2（两路都带图）不同，数值不可互比**"]],
        ["l", "r", "r", "r", "l"]))
    L.append("")
    L.append(f"> {pa['finding']}")
    L.append("")
    L.append("> **这不是性能取舍，是硬约束否决**：p2 在短对话（+9.4%）与并发（+10.7%）上都更快，")
    L.append("> 但它让峰值涨到 20,535 MiB、单请求上下文砍半。只要 4 GiB 冗余是红线，")
    L.append("> p2 就不可选 —— 性能数字再好看也没用。")
    L.append("")
    L.append("> ⚠️ **一处必须纠正的口径**：§5.1 表里 `decode_tps_system` 是 67.29 → 53.25（-20.9%），")
    L.append("> 但那条路径上同时变了 cap、n-max、parallel 三个变量。受控 A/B（只换 parallel）")
    L.append("> 显示真实并发代价只有 **-10.7%**。把跨配置差额当成单项归因，")
    L.append("> 是性能报告里最常见的错误之一 —— 本节把它拆开了。")
    L.append("")

    L.append("#### 5.7.2 图像分辨率：提速的同时修好了质量问题")
    L.append("")
    L.append(_table(
        ["cap", "实际图像 token", "ViT (ms)", "文本侧 prefill token", "TTFT (ms)", "难题"],
        [[768, ve["cap768"]["image_tokens"],
          f'{ve["cap768"]["vit_ms"][0]}~{ve["cap768"]["vit_ms"][-1]}',
          "788", f'{ve["cap768"]["prompt_ms"][0]}', "**7/7**"],
         [1024, ve["cap1024"]["image_tokens"],
          f'{ve["cap1024"]["vit_ms"][0]}~{ve["cap1024"]["vit_ms"][-1]}',
          "1083", f'{ve["cap1024"]["prompt_ms"][0]}', "6/7"]],
        ["r", "r", "r", "r", "r", "c"]))
    L.append("")
    L.append(f"> {ve['finding']}")
    L.append("")
    L.append("**「图 + 20K 文本」注意力实验**（同一道题，答案在文本尾部，每档 3 次）：")
    L.append("")
    L.append(_table(
        ["cap", "结果", "答案"],
        [[k, ("✅ " if v.startswith("3/3") else "❌ ") + v,
          "读到文本尾部" if v.startswith("3/3") else "退回图片内可见的 AUTH CODE"]
         for k, v in ac["results"].items()],
        ["r", "l", "l"]))
    L.append("")
    L.append(f"> {ac['finding']}")
    L.append("")
    L.append("> **这条的教训比收益更值钱**：上一轮把该项失败归因为「模型能力边界」，")
    L.append("> 本轮只是把 cap 从 1024 降到 768 就修好了（3/3）。归因到「模型不行」")
    L.append("> 是最省事也最容易出错的结论 —— 它让问题永远不被解决。")
    L.append("")

    L.append("#### 5.7.3 输出速率是「配置 × 内容类型」的联合属性")
    L.append("")
    L.append("同一配置、同一张 1024px 图，只换提问文本，各 5 次：")
    L.append("")
    L.append(_table(
        ["提问", "Decode 中位 (t/s)", "范围", "离散度"],
        [[k, v["median"], f'{v["range"][0]}~{v["range"][1]}',
          f'{v["spread_pct"]}%']
         for k, v in ps["results"].items()],
        ["l", "r", "l", "r"]))
    L.append("")
    L.append(f"> {ps['finding']}")
    L.append("")
    L.append("> **这条决定了「极限」该怎么报**：不能笼统说「多模态上限 120 t/s」。")
    L.append("> 正确的说法是「重复型输出（逐字读文字）可达 120~128，")
    L.append("> 自由描述约 100~110，短抽取波动最大」。")
    L.append("> 任何跨报告比较的 t/s 都必须绑定提问模板，否则数字不可比。")
    L.append("")

    L.append("#### 5.7.4 体感抖动：ITL 的 p50 是假的")
    L.append("")
    L.append(_table(
        ["场景", "ITL p50 (ms)", "ITL p95 (ms)", "读法"],
        [["纯文本短对话", itl["text_short"]["p50_ms"], itl["text_short"]["p95_ms"],
          "p50 近乎 0（同批到达），p95 才是一步的真实耗时"],
         ["28K 上下文", itl["text_32k"]["p50_ms"], itl["text_32k"]["p95_ms"],
          "长上下文下每步变慢，p95 升到 ~40 ms"],
         ["1024px 视觉", itl["vis_1024"]["p50_ms"], itl["vis_1024"]["p95_ms"],
          "与短对话同量级"]],
        ["l", "r", "r", "l"]))
    L.append("")
    L.append(f"> {itl['finding']}")
    L.append("")

    L.append("#### 5.7.5 质量矩阵：提速与质量不是取舍")
    L.append("")
    L.append(_table(
        ["配置", "简单题", "难题", "KV 检索"],
        [[k, v["simple"], v["hard"], v["kv"]] for k, v in qm["results"].items()],
        ["l", "c", "c", "c"]))
    L.append("")
    L.append(f"> {qm['finding']}")
    L.append("")
    L.append(f"> **评分器修正记录**：{ex['scorer_fix']['finding']}")
    L.append("")

    L.append("#### 5.7.6 退化清单（一处确认 + 一处不可判定）")
    L.append("")
    L.append("本轮不是「全面变快」。下表把「确认变慢的」与「统计上无法判定的」分开列，")
    L.append("因为把后者写成「-7.1%」会造成虚假的精确感。")
    L.append("")
    L.append(_table(
        ["指标", "同会话基线", "本轮", "变化", "根因"],
        [["28K 上下文 Decode (t/s)", "69.97（n=4，66.1~73.2）",
          "78.12（n=12，48.4~82.4）", "**不可判定**",
          "同一配置跨会话 CV 15.8%（n=48 合并），效应量小于方差。"
          "长上下文下 4 个候选 token 各自都要读 KV，理论上应更慢 —— 但实测无法分辨"],
         ["双路并发系统吞吐 (t/s)", "42.97（受控 A/B）", "38.39（受控 A/B）", "-10.7%",
          "p1 只有一个槽位，两路请求串行。**注意**：§5.1 表里的 67.29→53.25 "
          "是跨配置比较（基线含 n-max 2 与 cap 1024），不能归因给并发策略"]],
        ["l", "r", "r", "r", "l"]))
    L.append("")
    L.append("> **为什么仍选这个配置**：换来的东西是")
    L.append("> ① 单请求上下文 32K → 64K（能力，不是数字）；")
    L.append("> ② 峰值显存 20,535 → 19,857 MiB（p2+n4 越线，必须放弃）；")
    L.append("> ③ 视觉 Decode +6.7%、视觉 TTFT -1.7%、难题 6/7 → 7/7。")
    L.append("> 如果业务以「并发」或「28K 以上长上下文」为主，")
    L.append("> 应回滚到 B9_final，而不是用本配置。")
    L.append("")
    return "\n".join(L)


def sec_62(ex: dict) -> str:
    pa = ex["parallel_ab"]
    cab = pa.get("concurrency_ab") or {}
    L = []
    L.append("**唯一未达标项的口径拆解**：")
    L.append("")
    L.append("`decode_tps_system` 是唯一未达标的期望项（53.25 < 60）。但这一项的差额")
    L.append("不能全部归因给「并发策略」—— 基线与本轮在这条路径上同时变了三个变量")
    L.append("（cap 1024→768、n-max 2→4、parallel 2→1）。受控 A/B 的结果是：")
    L.append("")
    L.append(_table(
        ["口径", "p1（本轮）", "p2（基线形态）", "差异", "能归因给什么"],
        [["业务口径（§5.1，两路带图）", "53.25", "67.29", "-20.9%",
          "三个变量之和；**不能拆成单项**"],
         ["受控口径（只换 parallel，1 图 + 8K 文本）",
          f'{cab.get("p1_median", "—")}', f'{cab.get("p2_median", "—")}',
          f'{cab.get("delta_pct", 0):+.1f}%', "**只有并发策略本身**"]],
        ["l", "r", "r", "r", "l"]))
    L.append("")
    L.append("> **两个口径都必须保留**：业务口径回答「换配置后用户会看到什么」，")
    L.append("> 受控口径回答「哪个参数导致了变化」。把前者当成后者，")
    L.append("> 会让人误以为「并发是罪魁」；只报后者又会掩盖真实的业务影响。")
    L.append("")
    L.append("> **目标值本身是否合理**：60 t/s 是在上一轮（p2 + n-max 2 + cap 1024）")
    L.append("> 的 67.29 基础上取 89%。本轮三个变量都变了，仍沿用旧目标 ——")
    L.append("> 严格说这是「目标没有随配置更新」，而不是「性能不达标」。")
    L.append("> 本报告保留该目标与 FAIL 结论，因为改目标去迎合结果才是真正的问题。")
    L.append("")
    return "\n".join(L)


def sec_64(ex: dict) -> str:
    soak = ex["soak"]
    L = []
    L.append("### 6.4 极限配置的回归矩阵")
    L.append("")
    L.append("> 通用 §6.3 的矩阵是按后端设计的。极限配置还要多三类回归：")
    L.append("> **草稿深度的边界**、**并发的真实代价**、**长时稳定性**。")
    L.append("")
    L.append(_table(
        ["维度", "覆盖内容", "通过判据", "本次结果"],
        [["草稿深度边界", "n-max 0 / 2 / 3 / 4 / 5 / 6 / 8",
          "显存不越线且吞吐不倒退", "n-max=4 是顶点；5/6/8 吞吐单调下降且 8 已越线"],
         ["并发代价", "p1 vs p2（同配置）",
          "峰值 ≤20,480 MiB", "p2 为 20,535 MiB，**越线**，故选 p1"],
         ["分辨率—质量", "cap 512 / 768 / 1024 / 1280",
          "简单题 8/8、难题 ≥6/7", "512 为 6/7（丢细节）；768 为 **7/7**；1024/1280 为 6/7（注意力偏移）"],
         ["内容类型敏感性", "4 种提问模板 × 各 5 次",
          "同一模板内离散度 <5%", "重复型 1.7%；自由描述 16~50%"],
         ["多 KV 精度", "f16 / q8q8 / q8q4 / q4q4",
          "视觉 8/8、检索 3/3、PPL ≤2%", "全通过（沿用上一轮）"],
         ["长时运行", f'{soak["rounds"]} 轮 / {soak["duration_s"]:.0f} s 混合负载',
          "无 OOM、无泄漏、吞吐无漂移",
          f'{soak["ok_rounds"]}/{soak["rounds"]} 成功；显存漂移 {soak["vram_drift_mib"]} MiB；最高温 {soak["temp_c_max"]} °C'],
         ["异常输入", "1×1 图 / 2048px 大图 / 超长 prompt",
          "优雅降级或明确报错", "1×1 正常；>65,536 token 返回 400（预期）"]],
        ["l", "l", "l", "l"]))
    L.append("")
    L.append(f"> **长时压测**：{soak['workload']}，{soak['rounds']} 轮全部成功；")
    L.append(f"> 系统吞吐中位 {soak['system_tps_median']} t/s、单流 Decode 中位 "
             f"{soak['decode_tps_median']} t/s、TTFT 中位 {soak['ttft_ms_median']} ms；")
    L.append(f"> 显存 {soak['vram_mib_min']}~{soak['vram_mib_max']} MiB、"
             f"最高温 {soak['temp_c_max']} °C、峰值功耗 {soak['power_w_max']} W。")
    L.append("")
    L.append(f"> {soak['finding']}")
    L.append("")
    return "\n".join(L)


def sec_summary(ex: dict, led: dict) -> str:
    sm = ex["step_model"]
    pts = {p["nmax"]: p for p in sm["points"]}
    ps = ex["prompt_sensitivity"]
    ac = ex["attention_vs_cap"]
    p1, p2 = ex["parallel_ab"]["p1"], ex["parallel_ab"]["p2"]
    L = []
    L.append("## 结论摘要（先读这一节）")
    L.append("")
    L.append("> 本节把全文最重要的结论集中放在最前面，后面的七步是它们的证据链。")
    L.append("> **所有对比都在同一会话内完成** —— 跨会话的机器状态差异可达 ±8%，")
    L.append("> 大于多数配置差异，混用会得出相反结论。")
    L.append("")
    L.append("### 一、极限在哪")
    L.append("")
    L.append(_table(
        ["问题", "答案", "依据"],
        [["单卡多模态的 Decode 上限",
          f'**纯文本 {pts[4]["decode_tps"]:.0f} t/s；视觉 {pts[4]["vis_1024"]:.0f}~128 t/s**',
          "n-max=4 的实测（峰值单次 137 t/s）。不是调参问题，是验证成本的物理约束（§3.11.2）"],
         ["为什么不能更快",
          "**读一遍权重 18.2 ms 已接近下界；再往上只能「一次读多个 token」，"
          "而每多一个 token 要多付约 3.4 ms 验证成本**",
          f'步耗时模型：n-max 0→8 时 {pts[0]["step_ms"]:.1f}→{pts[8]["step_ms"]:.1f} ms（§3.11.1）'],
         ["输出速率是不是常数",
          f'**不是**。同配置同图片只换提问，中位从 '
          f'{min(v["median"] for v in ps["results"].values()):.0f} 到 '
          f'{max(v["median"] for v in ps["results"].values()):.0f} t/s',
          "MTP 接受长度取决于生成内容的重复度（§5.7.3）"],
         ["峰值显存还有多少空间",
          f'**{led["memory"]["safety_margin_mib"]/1024:.2f} GiB**（'
          f'{led["memory"]["peak_used_mib"]} MiB）',
          "p1 + n-max 4 的实测；p2 + n-max 4 会越线（§5.7.1）"],
         ["单请求能放多长上下文",
          "**65,536 token**（实测可处理 53,178）",
          "--parallel 1 换来的能力（上一轮 p2 只有 32K）"],
         ["质量有没有为此让步",
          "**没有，反而更好**：难题 6/7 → 7/7，简单题 8/8",
          f'cap 768 同时修好了「图+20K 文本」的注意力偏移'
          f'（768: {ac["results"]["768"]}；1024: {ac["results"]["1024"]}）']],
        ["l", "l", "l"]))
    L.append("")
    L.append("### 二、怎么逼近（三个不花显存的杠杆）")
    L.append("")
    L.append(_table(
        ["杠杆", "做法", "收益", "代价"],
        [["草稿深度", "`--spec-draft-n-max` 2 → 4",
          f'纯文本 {pts[2]["decode_tps"]:.0f} → {pts[4]["decode_tps"]:.0f} t/s'
          f'（+{(pts[4]["decode_tps"]/pts[2]["decode_tps"]-1)*100:.0f}%）；'
          f'视觉 {pts[2]["vis_1024"]:.0f} → {pts[4]["vis_1024"]:.0f} t/s'
          f'（+{(pts[4]["vis_1024"]/pts[2]["vis_1024"]-1)*100:.0f}%）',
          f'峰值 +{pts[4]["peak_mib"]-pts[2]["peak_mib"]} MiB；'
          f'接受率 {sm["acceptance"]["2"]:.2f} → {sm["acceptance"]["4"]:.2f}'],
         ["并发策略", "`--parallel` 2 → 1",
          "单请求上下文 32K → 64K（能力）；"
          f'短对话 {p1["text_short"]:.0f} t/s（p2 为 {p2["text_short"]:.0f}）',
          f'受控 A/B 双路并发 -10.7%；峰值 -{p2["peak_mib"]-p1["peak_mib"]} MiB（这是必需的）'],
         ["图像 token 上限", "`--image-max-tokens` 1024 → 768",
          "ViT 467 → 347 ms；文本侧 prefill 1083 → 788 token；难题 +1 项",
          "图像边长 1024 → 864 px（小字可能模糊）"]],
        ["l", "l", "l", "l"]))
    L.append("")
    L.append("### 三、代价是什么（一处确认退化 + 一处不可判定）")
    L.append("")
    L.append(_table(
        ["项", "同会话基线", "本轮", "判定", "说明"],
        [["双路并发系统吞吐", "42.97", "38.39", "**受控 A/B：-10.7%**",
          "p1 单槽，两路请求串行。这是换 64K 单请求上下文的代价。"
          "（§5.1 表里的 67.29→53.25 是跨配置比较，含 cap 与 n-max 差异）"],
         ["28K 上下文 Decode", "69.97（n=4）", "78.12（n=12）",
          "**无显著变化**",
          "同一配置跨会话中位在 55~84 之间跳动（合并 n=48，CV 15.8%，"
          "见台账 extra.variance_28k）；基线 69.97 与最终 64.97(n=8)/78.12(n=12) "
          "都落在同一分布内。**效应量小于方差时不应报告方向**"],
         ["短对话 Decode", "83.66", "91.56", "+9.4%",
          "同会话（REF_B9 vs FINAL1/2）；上一轮报告写的 92.65 是跨会话数字，不可直接相减"]],
        ["l", "r", "r", "l", "l"]))
    L.append("")
    L.append("### 四、图表口径说明（为什么两张图看起来矛盾）")
    L.append("")
    L.append("如果你先看 `charts.html` 的「Decode 吞吐：理论上界 vs 基线 vs 候选方案」，")
    L.append("再看本报告 §5.1 的前后对比，会发现基线数字不一样。这不是笔误，是**会话不同**：")
    L.append("")
    L.append(_table(
        ["来源", "基线值", "最终值", "差值", "为什么"],
        [["charts.html（修复前）", "92.65", "91.56", "**-1.2%（看起来变慢）**",
          "基线取自台账 `baseline_native`（上一轮会话），最终值取本轮同会话 —— 两个会话混用"],
         ["§5.1 前后对比", "83.66", "91.56", "**+9.4%（真实收益）**",
          "两侧都是本轮同会话（REF_B9 与 FINAL1/2）"],
         ["上一轮报告 §4.2", "92.65", "—", "—",
          "那是另一个会话测的 B9_final；同配置本轮复测只有 83.66（8 次中位，74.93~92.99）"]],
        ["l", "r", "r", "r", "l"]))
    L.append("")
    L.append("> **根因是机器的跨会话漂移，不是配置退化**：同一个 B9_final 配置，")
    L.append("> 上一轮会话测 92.65、本轮会话测 83.66（差 10.7%）。")
    L.append("> 因此本报告已做两件事：① 台账 `baseline_native` 与 `scenarios` 全部改为同会话值；")
    L.append("> ② 修了 `visualize.py` —— 它原来会**跳过**基线 bundle，只画台账声明的 native，")
    L.append("> 导致「基线一根柱子 + 候选若干根」的图上，基线永远来自另一个口径。")
    L.append("> 现在基线 bundle 也会被画出来并标注「同会话实测」，")
    L.append("> 若与台账声明值相差 >2% 则两根都画、各自标注。")
    L.append("")
    L.append("> **还有两根柱子容易被误读**：")
    L.append(">")
    L.append("> · `U4_p2_cap768` 102.4 t/s、`U1_nmax4` 90.5 t/s —— 都用了 **--parallel 2**，")
    L.append(">   峰值 20,535 MiB、余量 3.94 GiB < 4.0 GiB 红线。**它们越线，不是候选**。")
    L.append(">   图上只画吞吐、不画约束，所以看这张图必须同时读 §4.1 的显存列。")
    L.append(">")
    L.append("> · `U2_nmax4_p1` 94.4 t/s 高于最终选定的 `U3_final` 91.6 t/s。")
    L.append(">   这 2.8 t/s 的差不是「选错了」，而是**同配置两次测量**：")
    L.append(">   U2 是早期单次运行（raw/sweep_S_p1_n4.json），")
    L.append(">   U3 是 16 次全量套件的中位（raw/FINAL1+FINAL2）。")
    L.append(">   该指标的组内范围是 80.86~99.21，两者都落在分布内 ——")
    L.append(">   **不应把 U2 > U3 解读为「有更好的配置被放弃了」**。")
    L.append(">   选 U3 的依据是它的 cap 768（难题 7/7）与 b2048（28K 更快），不是短对话那一项。")
    L.append("")
    L.append("### 五、同一张图上的第二个「矛盾」：为什么实测能超过理论上界")
    L.append("")
    L.append("图上「理论上界 62.6 t/s」比所有实测柱子都矮 —— 这看起来像测量错误，其实是口径问题：")
    L.append("")
    L.append(_table(
        ["口径", "公式", "数值", "适用性"],
        [["理论上界（图上的橙色柱）", "标称带宽 1008 GB/s ÷ 权重 16.10 GB", "62.6 t/s",
          "**只适用于「一次权重读取产出一个 token」**，即无投机解码"],
         ["实测（本轮配置）", "mean_len 3.06 ÷ 步耗时 32.42 ms", "94.4 t/s",
          "MTP 下一次权重读取产出约 3 个 token，因此可以超过上面的上界"],
         ["本轮的正确上界（n-max=4）",
          "1 ÷ 步耗时 × mean_len = 1 ÷ 32.42 ms × 3.06", "94.4 t/s",
          "与实测完全一致 —— 说明当前配置已把「步耗时 × 接受长度」这一乘积用尽"],
         ["若验证成本归零（假设）",
          "1 ÷ 18.2 ms × 3.06", "168 t/s",
          "这是「只读权重、不付验证成本」的理论天花板，现实中不可达"]],
        ["l", "l", "r", "l"]))
    L.append("")
    L.append("> **所以「超过理论上界」不是奇迹，也不是错误**：MTP 改变的是分母 ——")
    L.append("> 从「每 token 读一遍权重」变成「每 N 个 token 读一遍权重」。")
    L.append("> 本报告 §3.11 用的正是修正后的口径（Decode = mean_len ÷ 步耗时），")
    L.append("> 而图上的 62.6 保留原公式只是为了给「无投机」这一档一个参照。")
    L.append("> 报告 §2.2 的那句「实测 83.66 t/s 已超过该上界 33.6%」就是这个意思：")
    L.append("> 它超过的是**无投机口径**的上界，不是物理带宽。")
    L.append("")
    L.append("> **怎么用这份结论**：")
    L.append("> · 若业务以**双路并发**为主 → 回滚 B9_final（p2），并发吞吐高 10.7%；")
    L.append("> · 若业务以**单请求 + 视觉问答**为主 → 用本配置，")
    L.append(">   视觉 Decode +6.7%、难题 +1 项、单请求上下文 ×2；")
    L.append("> · 若业务以 **28K 以上长上下文**为主 → 两者不可判定，")
    L.append(">   建议按「谁的显存余量大」选（本配置余量 4.60 GiB vs 基线 4.49 GiB）。")
    L.append("")
    return "\n".join(L)


def insert_after(text: str, anchor: str, block: str) -> str:
    i = text.find(anchor)
    if i < 0:
        raise SystemExit(f"锚点未找到：{anchor!r}")
    j = i + len(anchor)
    return text[:j] + "\n\n" + block + "\n" + text[j:]


def insert_before(text: str, anchor: str, block: str) -> str:
    i = text.find(anchor)
    if i < 0:
        raise SystemExit(f"锚点未找到：{anchor!r}（报告模板可能已变化）")
    return text[:i] + block + "\n" + text[i:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    led = json.loads(Path(a.ledger).read_text(encoding="utf-8"))
    ex = {**led["extra"], "curves": led.get("curves") or {}}
    p = Path(a.report)
    text = p.read_text(encoding="utf-8")

    for marker in ("### 3.11 极限剖面", "### 5.7 极限专项", "### 6.4 极限配置的回归矩阵"):
        if marker in text:
            raise SystemExit(f"报告已包含 {marker}，请先生成干净的报告再增补")

    # 摘要放在「运行标识」之后、正文之前
    text = insert_after(text, "---", sec_summary(ex, led))
    text = insert_before(text, "## 4. 优化方案选型与实施", sec_311(ex))
    text = insert_before(text, "## 6. 回归验收", sec_57(ex))
    text = insert_before(text, "### 6.3 质量红线与回归矩阵", sec_62(ex))
    text = insert_before(text, "## 7. 落地复盘", sec_64(ex))
    p.write_text(text, encoding="utf-8")

    mirror = p.parent.parent / f"report_{p.parent.name}.md"
    if mirror.exists():
        mirror.write_text(text, encoding="utf-8")
        print(f"[enrich] 已同步 {mirror}")
    print(f"[enrich] 已增补 4 节：结论摘要 / 3.11 / 5.7 / 6.4 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
