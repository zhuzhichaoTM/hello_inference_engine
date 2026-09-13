#!/usr/bin/env python3
"""解析冒烟测试输出，提取每个 prompt 的生成文本"""
import re, glob, os, json

OUT = '/Users/zhuzhichao/Documents/agent/deploy/llama/llama_test/quant_verification/smoke'
result = {}

for f in sorted(glob.glob(os.path.join(OUT, '*.txt'))):
    name = os.path.basename(f).replace('.txt', '')
    text = open(f, encoding='utf-8', errors='ignore').read()
    # 按 PROMPT 分块
    blocks = re.split(r'--- PROMPT \d+: (.+?) ---', text)
    gens = []
    for i in range(1, len(blocks), 2):
        prompt = blocks[i].strip()
        body = blocks[i+1] if i+1 < len(blocks) else ''
        # 生成内容在 "> prompt" 之后，到 "[ Prompt:" 之前
        m = re.search(r'>\s*' + re.escape(prompt) + r'\s*\n(.*?)\n\[ Prompt:', body, re.S)
        gen = m.group(1).strip() if m else body.strip()
        gens.append({'prompt': prompt, 'gen': gen})
    result[name] = gens

json.dump(result, open(os.path.join(OUT, 'parsed.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

# 打印每个模型对 prompt 3 (算术题) 的输出，便于快速对比
for name, gens in result.items():
    print(f"\n{'='*70}\n### {name}")
    for g in gens:
        short = g['prompt'][:20]
        gen = g['gen'].replace('\n', ' ')
        print(f"  [{short}...] -> {gen[:150]}")
