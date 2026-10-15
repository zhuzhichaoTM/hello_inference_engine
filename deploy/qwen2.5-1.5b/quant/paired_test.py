#!/usr/bin/env python3
"""
配对显著性检验：利用 per-chunk NLL 做配对 t 检验
背景：各模型在同一批 32 个 chunk 上评测，误差是配对的，
      配对差分能消除 chunk 难度带来的方差，灵敏度远高于各自独立的标准误。
"""
import re, math, os
from statistics import mean, stdev

LOGDIR = '/Users/zhuzhichao/Documents/agent/deploy/llama/llama_test/quant_verification/logs'

MODELS = ['A0-f16-baseline', 'B7-q4km-noimat', 'B1-q4km-imat',
          'B2-iq4xs-noimat', 'B3-iq4xs-imat', 'B4-iq4xs-imat-embdQ6',
          'B8-iq4xs-imat-embdQ8', 'B5-iq3xxs-imat', 'B6-iq3xxs-imat-embdQ6',
          'B9-iq3xxs-imat-embdQ8', 'Z-delivered-q4km']

def load_nll(name):
    """从日志提取 per-chunk PPL，转成 NLL 序列"""
    path = os.path.join(LOGDIR, f'{name}.ppl.log')
    txt = open(path, encoding='utf-8', errors='ignore').read()
    # 取最后一次出现的 per-chunk 列表（最后一行最全）
    matches = re.findall(r'\[(\d+)\]([\d.]+)', txt)
    # 按 chunk 编号归位，取最后一个完整序列
    d = {}
    for idx, val in matches:
        d[int(idx)] = float(val)
    n = max(d.keys())
    chunks = [d[i] for i in range(1, n + 1)]
    return [math.log(p) for p in chunks]  # NLL

def paired_ttest(a, b):
    """配对 t 检验: a - b 的均值是否显著非零"""
    assert len(a) == len(b)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    m = mean(diffs)
    sd = stdev(diffs)
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float('inf')
    # 双尾 p 值（t 分布，df=n-1），用正态近似够用
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return m, se, t, p, n

data = {}
print("加载 per-chunk NLL ...")
for m in MODELS:
    try:
        data[m] = load_nll(m)
        print(f"  {m:28s} {len(data[m])} chunks")
    except Exception as e:
        print(f"  {m:28s} FAILED: {e}")

print()
print("=" * 88)
print("配对 t 检验（H0: 两模型逐块 NLL 无差异）")
print("=" * 88)

COMPARISONS = [
    ('B7-q4km-noimat', 'B1-q4km-imat',        'imatrix 对 Q4_K_M 的效果'),
    ('B2-iq4xs-noimat', 'B3-iq4xs-imat',      'imatrix 对 IQ4_XS 的效果'),
    ('B3-iq4xs-imat',   'B4-iq4xs-imat-embdQ6', 'embd Q6_K 显式指定（应无差异）'),
    ('B3-iq4xs-imat',   'B8-iq4xs-imat-embdQ8', 'embd Q6_K -> Q8_0'),
    ('B5-iq3xxs-imat',  'B6-iq3xxs-imat-embdQ6', 'embd Q5_K -> Q6_K (IQ3_XXS)'),
    ('B7-q4km-noimat',  'Z-delivered-q4km',   '同一配置重复（应为恒等）'),
    ('A0-f16-baseline', 'B1-q4km-imat',       'F16 -> Q4_K_M+imat 的总代价'),
    ('B1-q4km-imat',    'B3-iq4xs-imat',      'Q4_K_M vs IQ4_XS（均带 imatrix）'),
]

print(f"{'对比':44s} {'ΔNLL均值':>11s} {'配对SE':>9s} {'t':>8s} {'p值':>10s} {'显著':>6s}")
print("-" * 88)
for a, b, label in COMPARISONS:
    if a not in data or b not in data:
        continue
    m, se, t, p, n = paired_ttest(data[a], data[b])
    dppl = mean(data[a]) - mean(data[b])
    sig = 'YES' if p < 0.05 else 'no'
    # PPL 变化比
    ratio = math.exp(-m)
    print(f"{label:44s} {m:+11.5f} {se:9.5f} {t:8.2f} {p:10.2e} {sig:>6s}")
    print(f"{'':44s}  -> PPL 比值 {a.split('-')[0]}/{b.split('-')[0]} = {ratio:.5f}  ({'更差' if ratio>1 else '更好' if ratio<1 else '相同'})")
