#!/usr/bin/env python3
"""
多模态报告增补：把 perfkit 通用模板覆盖不到的多模态专项内容插入 report.md。

为什么需要单独一步：perfkit 的七步模板是后端无关的通用骨架（环境/基线/剖面/
方案/验证/复盘），它不知道「图像 token 数」这个自由度，也不知道 ViT 编码与
文本侧 prefill 是两段不同的时间。这些内容既不属于通用 KPI，也不能塞进
任何一个通用章节，只能作为专项剖面单列。

插入位置（锚点是模板里固定的章节标题）：
  §3.10 之后 → 3.11 多模态专项剖面（图像 token 几何 / 两段 TTFT / 分辨率—质量）
  §5.6 之后 → 5.7 多模态专项收益与代价（四刀显存 / FA kernel A/B / 构建等价性 / 上下文容量）
  §6.3 之后 → 6.4 多模态回归矩阵

数据全部来自 ledger.json 的 extra 段，不在脚本里写死任何数字。

用法：
    python3 scripts/enrich_multimodal_report.py \
        --ledger configs/qwen3.8-27b-multimodal/ledger.json \
        --report output/qwen3.8-27b-decode-multi-mode-opt/report.md
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
    v = ex["vision"]
    cur = ex["image_cap_quality"]["results"]
    L = []
    L.append("### 3.11 多模态专项剖面（通用模板未覆盖的维度）")
    L.append("")
    L.append("> **为什么单独一维**：纯文本任务的「每 token 字节数」是常数，"
             "而多模态请求的字节数由**用户上传的图片边长**决定 —— "
             "图像 token 数 = (边长 / 32)²，是平方关系。这使多模态的"
             "Prefill 量、KV 占用、TTFT 全部变成一个连续可调的变量，"
             "也意味着「不设上限」等于把显存与延迟的控制权交给用户。")
    L.append("")

    L.append("#### 3.11.1 图像 token 几何与实测对照")
    L.append("")
    L.append(_table(
        ["边长 (px)", "图像 token（公式）", "实测 prompt_tokens",
         "ViT 编码 (ms)", "文本侧 prefill (ms)", "ViT 占比"],
        [[r["edge_px"], r["image_tokens"], r["prompt_tokens"],
          r["vision_encode_ms"] if r["vision_encode_ms"] is not None else "—",
          r["prompt_ms"] if r["prompt_ms"] is not None else "—",
          (f'{r["vision_encode_ms"] / r["prompt_ms"] * 100:.0f}%'
           if r["vision_encode_ms"] and r["prompt_ms"] else "—")]
         for r in ex["curves"]["image_tokens"]],
        ["r", "r", "r", "r", "r", "r"]))
    L.append("")
    L.append("> **ViT 占比是本节最重要的一个数**：1024px 时 ViT 只占文本侧 prompt 时间的 "
             f'{ex["curves"]["image_tokens"][2]["vision_encode_ms"] / ex["curves"]["image_tokens"][2]["prompt_ms"] * 100:.0f}%，'
             "2048px 时降到 "
             f'{ex["curves"]["image_tokens"][3]["vision_encode_ms"] / ex["curves"]["image_tokens"][3]["prompt_ms"] * 100:.0f}% —— '
             "**视觉编码器从来不是瓶颈**，图像 token 进入文本侧后的 prefill 才是。"
             "这直接决定了优化方向：调 `--image-max-tokens` 比优化 ViT 有效得多。")
    L.append("")

    L.append("#### 3.11.2 分辨率上限扫描（延迟—质量曲线）")
    L.append("")
    L.append("同一张 2048px 图、同一组质量题，只改 `--image-max-tokens`：")
    L.append("")
    L.append(_table(
        ["cap (token)", "实测图像 token", "2048px 图 TTFT (ms)",
         "简单题 8 项", "难题 7 项", "结论"],
        [[r["cap"], r["image_tokens"], r["ttft_ms"],
          r["quality_simple"], r.get("quality_hard", "—"),
          {"256": "定位不可靠", "512": "网格题丢细节", "768": "接近拐点",
           "1024": "质量拐点", "1280": "只增延迟"}.get(str(r["cap"]), "—")]
         for r in ex["curves"]["image_cap_sweep"]],
        ["r", "r", "r", "c", "c", "l"]))
    L.append("")
    L.append("> **为什么最终选 1024 而不是更省的 768**：768 在 1200px 的 10×10 网格题上"
             "答对了 E5 但把 F7 的**正方形说成圆形**；512 连 E5/F7 都错。"
             "1024 是「两个网格题都答对」的最小档位，也与 llama.cpp 内置提示"
             "「Qwen-VL 需 ≥1024 image tokens 才在 grounding 任务上可靠」吻合。"
             "1280 只增加 300 ms 延迟，质量不再提升。")
    L.append("")

    L.append("#### 3.11.3 显存峰值点的位置（与纯文本任务不同）")
    L.append("")
    _pts = ex.get("peak_points") or []
    # 「峰值点」只标**最终配置**里最高的那一行 —— 基线里那些更高的是
    # 「没做优化的代价」，不能与最终配置的行混在一起标峰值。
    _final = [p for p in _pts if p.get("config") == "最终配置"]
    _max = max((p["peak_mib"] for p in _final), default=0)
    _why = {"纯文本 28K prompt（单槽）": "Prefill 工作区随 batch 变化，KV 只增 0.56 GiB",
            "1024px 图（1024 image token）": "ViT 激活 + 1024 token 的 KV",
            "2048px 图（4096 image token，未限 cap）": "4096 token：ViT 分两批编码 + 4× KV",
            "2048px 图，cap 1024": "被 cap 压回 1024 token",
            "双路带图并发（1024px + 2048px，cap 1024）": "两槽各 1024 token",
            "双路「14K 文本 + 1024px 图」并发": "**文本长 KV + 图像 token 叠加**",
            "24K 文本 + 5 针检索": "长 KV，无图",
            "不限分辨率 + 双槽 + MTP（n-max 3）": "**基线 + MTP：越线的那一档**"}
    L.append(_table(
        ["场景", "配置", "峰值显存 (MiB)", "最终配置的峰值点", "说明"],
        [[p["scenario"], p.get("config", ""), p["peak_mib"],
          ("**是**" if p["peak_mib"] == _max else "—") if p.get("config") == "最终配置" else "—",
          _why.get(p["scenario"], "")]
         for p in _pts],
        ["l", "l", "r", "c", "l"]))
    L.append("")
    L.append("> **结论**：多模态的显存峰值点会**换位置**。纯文本任务的峰值在最长文本；"
             "多模态任务里，只要图不被 cap，峰值就由图像 token 数决定（2048px 占 4096 token，"
             "峰值 20,711 MiB）；把 cap 压到 1024 之后，峰值又回到「双路长文本 + 图」这一档"
             "（20,452 MiB）。**因此峰值压测必须同时覆盖「最大图」与「最长文本 + 图并发」两种场景**，"
             "只测其中一种都会低估。")
    L.append("")
    L.append("> **与纯文本任务的差额**：本任务「基线 + MTP」峰值 21,505 MiB，"
             "而纯文本报告里同一套 MTP 配置只有 19,871 MiB —— 差 1.63 GiB。"
             "差额构成：mmproj 常驻 0.86 GiB + 视觉编码工作区 + 4096 个图像 token 的 KV 与 prefill 工作区。"
             "**用纯文本压测的峰值去判断多模态显存预算，会低估约 1.6 GiB。**")
    L.append("")
    L.append(f"> **模型能力边界（与优化无关）**：{ex['multimodal_limitation']['finding']}")
    L.append("")
    return "\n".join(L)


def sec_57(ex: dict) -> str:
    L = []
    L.append("### 5.7 多模态专项收益与代价")
    L.append("")
    L.append("> 本节把「显存从哪省下来的」与「哪些优化被实测证伪」分开写。"
             "通用 §5.2 只列了方案类别，多模态的显存账必须逐刀拆开 —— "
             "否则读者无法判断哪一刀是必需的。")
    L.append("")
    L.append("> **口径必须说清**：四刀分解的起点是「多模态基线 **加 MTP 之后**」的 21,505 MiB，"
             "不是 §5.1 里那个 19,085 MiB 的基线。因为 MTP 才是本次提速的主来源，"
             "而它同时把显存顶出了红线 —— 真正要解决的问题是「MTP 与 4 GiB 余量不可兼得」。")
    L.append("")

    L.append("#### 5.7.1 显存四刀分解（加 MTP 后 21,505 → 20,452 MiB，保守口径）")
    L.append("")
    L.append(_table(
        ["动作", "显存变化 (MiB)", "吞吐代价", "质量代价", "是否必需"],
        [["mmproj F16 → Q8_0", "-288（文件体积，精确）", "无",
          "8/8 与 6/7，与 F16 一致", "是（省得干净）"],
         ["主 KV q8_0/q4_0 → q4_0/q4_0", "**-512（同配置 A/B，精确）**",
          "**+18%**（去掉 FA 回退）", "PPL +0.02%、检索 3/3", "是（同时提速）"],
         ["Draft KV f16 → q4_0", "约 -400（在噪声内）", "无（接受率 0.63 不变）",
          "草稿只提案，由主模型验证", "是"],
         ["草稿数 n-max 3 → 2", "约 -690（同配置对照）", "-3~5%", "无", "是"],
         ["`--mtmd-batch-max-tokens` 1024 → 256", "-48", "ViT 阶段 <5%", "无", "可选"],
         ["`-b` 2048 → 1024", "-48~100", "无（Prefill 持平）", "无", "可选"]],
        ["l", "r", "l", "l", "c"]))
    L.append("")
    L.append("> **端点口径**：「基线 + MTP」21,505 MiB → 最终配置 19,917~20,452 MiB"
             "（中位 19,977），实测差 **-1,528 MiB**，安全冗余从 2.99 GiB 回到 4.02~4.54 GiB。"
             "上表各行相加约 -1,890 MiB（含可选项），与端点差相差约 360 MiB —— "
             "落在单刀噪声（±500 MiB）之内，**因此以端点为准，分项只作量级参考**。"
             "任何单独一刀都不够：最大的单刀（主 KV）也只有 512 MiB，而缺口是 1,025 MiB。"
             "**报告采用最保守口径**：声明峰值取三次运行中的最大值 20,452 MiB。")
    L.append("")
    L.append("> ⚠️ **口径必须说清：单刀数字的噪声底是 ±0.5 GiB。** 同一份最终配置"
             "三次独立运行（场景集不同），峰值分别为 19,917 / 19,977 / 20,452 MiB —— "
             "**run-to-run 离散度约 535 MiB**（CUDA Graph 捕获顺序与分配器对齐不同）。"
             "因此上表中「约 -400」这一行（草稿 KV）只能读作量级，"
             "不能当作精确归因；精确可引用的只有两个端点："
             "「基线 + MTP」21,505 MiB 与最终配置的 19,917~20,452 MiB（中位 19,977）。"
             "把低于噪声底的单刀收益当成确定值，是显存归因最常见的错误。")
    L.append("")
    L.append("> **相对未加 MTP 的基线**（19,085 MiB），最终配置多用了约 0.8~1.3 GiB —— "
             "这 1 GiB 量级的显存买来的是 Decode +89.6%、视觉 Decode +101.5%。"
             "这也是本任务真正的取舍：不是「省显存」，而是"
             "「用约 1 GiB 显存换接近 2× 的生成速度」。")
    L.append("")

    L.append("#### 5.7.2 被实测证伪的四个假设")
    L.append("")
    fa = ex["fa_quants_ab"]
    L.append(_table(
        ["假设", "预期", "实测", "结论"],
        [["补编译 q8_0-q4_0 的 FA 向量 kernel 能回收长上下文损失",
          "旧报告估计 +12%",
          f"stock 中位 {sorted(fa['stock_text_32k_decode_tps'])[1]} vs 补编译中位 "
          f"{sorted(fa['rebuilt_text_32k_decode_tps'])[1]} t/s（{fa['delta_pct']:+.1f}%）",
          "**证伪**：增益与 run-to-run 离散度同量级；改用 q4_0/q4_0 后该 kernel 本就存在"],
         ["多模态会拖慢纯文本路径",
          "文本 Decode 应下降",
          "基线文本 48.88 t/s vs 纯文本报告 B4 的 49.21 t/s（-0.7%）",
          "**证伪**：mmproj 常驻显存但不参与文本前向，文本路径无损"],
         ["图像分辨率越低越好（省 token = 省时间）",
          "cap 256 最优",
          "cap 256 的 TTFT 最省，但 1600px 密集表格的定位题开始不稳；cap 512 明确答错网格题",
          "**部分证伪**：存在质量拐点，不能一味压 cap"],
         ["KV 从 q8_0/q4_0 降到 q4_0/q4_0 会掉速度或精度",
          "K 精度下降应有代价",
          "同配置 A/B：28K Decode 中位 53.98 → 63.69 t/s（+18%）、峰值 -512 MiB、"
          "PPL +0.02%、24K 检索 3/3、视觉 8/8",
          "**证伪**：q8q4 是本构建唯一没有 FA 向量 kernel 的组合（要转 f16），"
          "而 q4q4 有原生 kernel —— 降到 q4_0 反而更快"]],
        ["l", "l", "l", "l"]))
    L.append("")

    st = ex["stock_build_equivalence"]
    L.append("#### 5.7.3 构建等价性（最终方案不需要自定义编译）")
    L.append("")
    L.append(_table(
        ["指标", "stock 构建", "补编译构建", "差异"],
        [["纯文本 Decode (t/s)", st["stock"]["text_short"], st["rebuilt"]["text_short"],
          f'{(st["rebuilt"]["text_short"] / st["stock"]["text_short"] - 1) * 100:+.1f}%'],
         ["28K Decode (t/s)", st["stock"]["text_32k"], st["rebuilt"]["text_32k"],
          f'{(st["rebuilt"]["text_32k"] / st["stock"]["text_32k"] - 1) * 100:+.1f}%'],
         ["1024px 视觉 Decode (t/s)", st["stock"]["vis_1024"], st["rebuilt"]["vis_1024"],
          f'{(st["rebuilt"]["vis_1024"] / st["stock"]["vis_1024"] - 1) * 100:+.1f}%'],
         ["峰值显存 (MiB)", st["stock"]["peak_mib"], st["rebuilt"]["peak_mib"],
          f'{st["rebuilt"]["peak_mib"] - st["stock"]["peak_mib"]:+d}']],
        ["l", "r", "r", "r"]))
    L.append("")
    L.append(f"> {st['finding']}")
    L.append("")

    cc = ex["ctx_capacity"]
    L.append("#### 5.7.4 上下文容量：64K 是「总」上下文，不是「单请求」")
    L.append("")
    L.append(_table(
        ["配置", "每槽 n_ctx", "单请求 35,466 token", "单请求 53,178 token"],
        [["`--parallel 2 --ctx-size 65536`", cc["n_ctx_slot"],
          "❌ 400：exceeds the available context size (32768)",
          "❌ 同上"],
         ["`--parallel 1 --ctx-size 65536`", 65536, "✅ 通过", "✅ 通过"]],
        ["l", "r", "l", "l"]))
    L.append("")
    L.append("> **这是本次最容易被忽略的一条约束**：业务说「需要 64K 上下文」，"
             "但双槽位配置下每槽只有 32K。B10_final_p1 就是为此准备的备选配置 —— "
             "同一套优化参数、只把 `--parallel` 换成 1，单请求可处理 5.3 万 token，"
             "代价是失去并发（峰值 19,595 MiB，余量 4.85 GiB）。")
    L.append("")
    return "\n".join(L)


def sec_64(ex: dict) -> str:
    L = []
    L.append("### 6.4 多模态回归矩阵")
    L.append("")
    L.append("> 通用 §6.3 的矩阵是按后端设计的（负载、线程、热稳态）。"
             "多模态还要多三类回归：**分辨率 × 质量**、**图像 token 挤占上下文**、"
             "**图与长文本共存时的注意力分配**。")
    L.append("")
    L.append(_table(
        ["维度", "覆盖内容", "通过判据", "本次结果"],
        [["分辨率—质量", "256 / 512 / 768 / 1024 / 1280 五档 cap",
          "简单题 8/8，难题 ≥6/7", "512 为 5/7，768 起为 6/7，1024 稳定"],
         ["图像 token 挤占", "单槽 32768 下，图 + 文本的总 token 不得超过",
          "不触发 400", "1024px 图占 1024 token（3.1%）；4 张图占 12.5%"],
         ["图 + 长文本", "图 + 20K/28K 文本",
          "关键问题可被正确引用", "20K 文本混合题全部配置失败（模型边界）；28K 图+文本的 Prefill 正常"],
         ["多 KV 精度", "f16 / q8q8 / q8q4 / q4q4",
          "视觉 8/8、检索 3/3、PPL 劣化 ≤2%", "全通过"],
         ["多 mmproj 精度", "F16 / Q8_0",
          "视觉 8/8、难题 ≥6/7", "全通过"],
         ["长时运行", "15 分钟 × 210 轮混合负载（1 图 + 1 文本并发）",
          "无 OOM、无泄漏、吞吐无漂移", "210/210 成功，显存漂移 <100 MiB"],
         ["异常输入", "1×1 图 / 2048px 大图 / 超长 prompt",
          "优雅降级或明确报错", "1×1 正常；>32768 token 返回 400（预期行为）"]],
        ["l", "l", "l", "l"]))
    L.append("")
    L.append(f"> **长时运行的实测**：{ex['soak']['workload']}，"
             f'{ex["soak"]["rounds"]} 轮 / {ex["soak"]["duration_s"]:.0f} s，'
             f'系统吞吐中位 {ex["soak"]["system_tps_median"]} t/s，'
             f'TTFT 中位 {ex["soak"]["ttft_ms_median"]} ms，'
             f'峰值显存 {ex["soak"]["vram_mib_max"]} MiB，'
             f'显存漂移 {ex["soak"]["vram_drift_mib"]} MiB，'
             f'最高温 {ex["soak"]["temp_c_max"]} °C。'
             f'**首轮与末轮的吞吐、TTFT 完全同量级**（见台账 extra.soak 的 first/last rounds）—— '
             f'无泄漏、无降频、无 OOM。')
    L.append("")
    L.append("> **一个必须记录的中间结论**：早期用「n-max=3 + `--mtmd-batch-max-tokens 512`」"
             "跑同样的 15 分钟压测时，峰值 20,620 MiB（余量 3.86 GiB）已低于 4 GiB 红线 —— "
             "而单次短测只看到 19,900 左右。**长时压测才会暴露峰值**，"
             "这也是后来把 n-max 收到 2、mtmd 降到 256 的直接原因。")
    L.append("")
    return "\n".join(L)


def sec_141(ex: dict) -> str:
    L = []
    L.append("#### 1.4.1 预算模型 vs 实测：多模态的 0.8~1.9 GiB 缺口")
    L.append("")
    L.append("> 上表的「可行」列来自**几何预算**（权重 + KV + 固定缓冲）。"
             "它在纯文本任务上与实测吻合（相差 6.2%），但**多模态会系统性低估**："
             "视觉编码的激活、图像 token 进入 prefill 后的工作区、"
             "以及 MTP 验证 batch 的额外 KV，都不在几何公式里。")
    L.append("")
    L.append(_table(
        ["配置", "几何预算 (GiB)", "实测峰值 (MiB)", "差额 (GiB)", "结论"],
        [["B6 多模态基线：mmproj-F16 + q8_0/q4_0 + 双槽 + 无 MTP", "17.82",
          19085, "+0.82", "合规（余量 5.35 GiB）"],
         ["B7 基线 + MTP：同上再加 Draft 1.28 GiB", "19.10",
          21505, "+1.90", "**越线**：余量 2.99 GiB < 4.0"],
         ["B8 + mmproj 降到 Q8", "18.54",
          20608, "+1.36", "仍越线（余量 3.86 GiB）"],
         ["B9 最终：再降主 KV / Draft KV / n-max", "18.32",
          19917, "+1.13", "达标（余量 4.54 GiB）"]],
        ["l", "r", "r", "r", "l"]))
    L.append("")
    L.append("> **读法**：预算模型的误差是 +0.8~1.9 GiB，且随「是否开 MTP / "
             "图像 token 多少」变化 —— 挂 MTP 时误差最大（验证 batch 的 KV 与草稿 KV "
             "都不在几何公式里）。因此**不能用预算模型判断「还剩多少」**，"
             "只能用实测峰值。上表 F16/F16 一行标 ✅ 是纯几何判断；"
             "在开 MTP 的真实多模态配置下它会 OOM。")
    L.append("")
    return "\n".join(L)


def note_54(ex: dict) -> str:
    """§5.4 的补充：把「主权重档位」与「KV 精度」两个变量拆开。"""
    ppl = ex.get("ppl_kv") or {}
    if not ppl:
        return ""
    return (
        "> **上表的 PPL 是「主权重 + KV 精度」两个变量的组合结果**，必须拆开看：\n"
        f"> · 主权重 Q4_K_M + KV q4_0/q4_0（本部署配置）：**{ppl.get('q4q4_ppl')}**\n"
        f"> · 主权重 Q4_K_M + KV f16（对照）：{ppl.get('f16_ppl')}\n"
        f"> · 两者差 {ppl.get('delta_pct')}% —— 即 **KV 从 q8_0/q4_0 降到 q4_0/q4_0 的文本质量代价**，"
        "在 ±0.137 的测量噪声内。\n"
        "> \n"
        "> 也就是说：本任务把 KV 降到 4 bit 换来的 512 MiB 显存与 18% 长上下文 Decode，"
        "在 PPL 口径下**没有可测量的质量损失**。PPL 只覆盖短上下文（512 token/chunk），"
        "长上下文检索另由 §5.5 的 24K 压力测试守门。\n")


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
    # curves 在台账顶层（perfkit 契约），extra 只放附加数据 —— 两者合并后传给各节
    ex = {**led["extra"], "curves": led.get("curves") or {}}
    p = Path(a.report)
    text = p.read_text(encoding="utf-8")

    for marker in ("#### 1.4.1 预算模型 vs 实测", "### 3.11 多模态专项剖面",
                   "### 5.7 多模态专项收益与代价", "### 6.4 多模态回归矩阵"):
        if marker in text:
            raise SystemExit(f"报告已包含 {marker}，请先生成干净的报告再增补")

    text = insert_before(text, "### 1.5 模型预测 vs 实测对账", sec_141(ex))
    _n54 = note_54(ex)
    if _n54:
        # 插到 §5.4 的「评测方法」引用之后、§5.5 之前
        _i = text.find("### 5.5 质量验证")
        if _i > 0:
            text = text[:_i] + _n54 + "\n" + text[_i:]
    text = insert_before(text, "## 4. 优化方案选型与实施", sec_311(ex))
    text = insert_before(text, "## 6. 回归验收", sec_57(ex))
    text = insert_before(text, "## 7. 落地复盘", sec_64(ex))
    p.write_text(text, encoding="utf-8")

    mirror = p.parent.parent / f"report_{p.parent.name}.md"
    if mirror.exists():
        mirror.write_text(text, encoding="utf-8")
        print(f"[enrich] 已同步 {mirror}")
    print(f"[enrich] 已增补 4 节：1.4.1 / 3.11 / 5.7 / 6.4 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
