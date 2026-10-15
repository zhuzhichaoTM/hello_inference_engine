#!/bin/bash
# PPL 评测：基线 F16 + 所有量化模型，统一参数保证可比性
# 参数: c=512 (与 imatrix 校准一致), chunks=32, CPU(-ngl 0), 4线程
set -u
cd /Users/zhuzhichao/Documents/agent/deploy/llama/llama_test

PP=./llama.cpp/build/bin/llama-perplexity
DATA=./wiki.test.raw
OUT=./quant_verification/ppl
LOG=./quant_verification/logs
CHUNKS=32

: > $OUT/summary.txt
echo "PPL evaluation: ctx=512 chunks=$CHUNKS backend=CPU threads=4"
echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

run_ppl() {
  local name="$1"; local model="$2"
  echo "[$(date '+%H:%M:%S')] evaluating $name ..."
  local t0=$(date +%s)
  $PP -m "$model" -f $DATA -c 512 --chunks $CHUNKS -ngl 0 -t 4 > "$LOG/${name}.ppl.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local ppl=$(grep -E "Final estimate" "$LOG/${name}.ppl.log" | tail -1 | sed 's/.*Final estimate: //')
  printf "  exit=%s time=%ss  PPL = %s\n" "$rc" "$((t1-t0))" "$ppl"
  echo "${name}|${ppl}|$((t1-t0))s" >> $OUT/summary.txt
}

run_ppl "A0-f16-baseline" "./qwen2.5-1.5b-instruct-f16.gguf"
run_ppl "B7-q4km-noimat" "./quant_verification/models/B7-q4km-noimat.gguf"
run_ppl "B1-q4km-imat" "./quant_verification/models/B1-q4km-imat.gguf"
run_ppl "B2-iq4xs-noimat" "./quant_verification/models/B2-iq4xs-noimat.gguf"
run_ppl "B3-iq4xs-imat" "./quant_verification/models/B3-iq4xs-imat.gguf"
run_ppl "B4-iq4xs-imat-embdQ6" "./quant_verification/models/B4-iq4xs-imat-embdQ6.gguf"
run_ppl "B8-iq4xs-imat-embdQ8" "./quant_verification/models/B8-iq4xs-imat-embdQ8.gguf"
run_ppl "B5-iq3xxs-imat" "./quant_verification/models/B5-iq3xxs-imat.gguf"
run_ppl "B6-iq3xxs-imat-embdQ6" "./quant_verification/models/B6-iq3xxs-imat-embdQ6.gguf"
run_ppl "B9-iq3xxs-imat-embdQ8" "./quant_verification/models/B9-iq3xxs-imat-embdQ8.gguf"
run_ppl "Z-delivered-q4km" "./qwen2.5-1.5b-instruct—new—q4_k_m.gguf"

echo "=============================================="
echo "PPL SUMMARY (chunks=$CHUNKS)"
cat $OUT/summary.txt
echo "end: $(date '+%Y-%m-%d %H:%M:%S')"
