#!/bin/bash
# 冒烟测试：所有模型用同一组 prompt 生成，对比输出一致性
set -u
cd /Users/zhuzhichao/Documents/agent/deploy/llama/llama_test

CLI=./llama.cpp/build/bin/llama-cli
OUT=./quant_verification/smoke
mkdir -p $OUT

PROMPTS=(
  "用一句话解释什么是机器学习。"
  "写一个 Python 函数，判断一个整数是否为素数。"
  "如果 3 个苹果 5 元，那么 12 个苹果多少钱？请给出计算过程。"
)

run_smoke() {
  local name="$1"; local model="$2"
  local outfile="$OUT/${name}.txt"
  : > "$outfile"
  echo "===== MODEL: $name =====" >> "$outfile"
  local i=0
  for p in "${PROMPTS[@]}"; do
    i=$((i+1))
    echo "" >> "$outfile"
    echo "--- PROMPT $i: $p ---" >> "$outfile"
    $CLI -m "$model" -p "$p" -n 160 -t 4 -no-cnv -st -ngl 0 --seed 42 --temp 0 </dev/null \
      >> "$outfile" 2>/dev/null
    echo "" >> "$outfile"
  done
  echo "[done] $name"
}

run_smoke "A0-f16-baseline" "./qwen2.5-1.5b-instruct-f16.gguf"
for f in ./quant_verification/models/*.gguf; do
  n=$(basename "$f" .gguf)
  run_smoke "$n" "$f"
done

echo "SMOKE TESTS DONE"
