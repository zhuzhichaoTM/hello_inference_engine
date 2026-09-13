#!/bin/bash
# 批量量化对比实验
set -u
cd /Users/zhuzhichao/Documents/agent/deploy/llama/llama_test

QZ=./llama.cpp/build/bin/llama-quantize
F16=./qwen2.5-1.5b-instruct-f16.gguf
IMAT=./quant_verification/imatrix-qwen25-1.5b.gguf
OUT=./quant_verification/models
LOG=./quant_verification/logs

run() {
  local name="$1"; shift
  echo "=============================================="
  echo "[$(date '+%H:%M:%S')] QUANTIZING: $name"
  echo "  args: $*"
  local t0=$(date +%s)
  "$@" > "$LOG/${name}.quantize.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local sz=$(stat -f%z "$OUT/${name}.gguf" 2>/dev/null || echo 0)
  echo "  exit=$rc  size=$sz  time=$((t1-t0))s"
  grep -E "quant size|model size" "$LOG/${name}.quantize.log" | tail -2
}

# B1: Q4_K_M + imatrix
run "B1-q4km-imat" $QZ --imatrix $IMAT $F16 $OUT/B1-q4km-imat.gguf Q4_K_M

# B2: IQ4_XS 无 imatrix
run "B2-iq4xs-noimat" $QZ $F16 $OUT/B2-iq4xs-noimat.gguf IQ4_XS

# B3: IQ4_XS + imatrix
run "B3-iq4xs-imat" $QZ --imatrix $IMAT $F16 $OUT/B3-iq4xs-imat.gguf IQ4_XS

# B4: IQ4_XS + imatrix + token_embd 保护为 Q6_K
run "B4-iq4xs-imat-embdQ6" $QZ --imatrix $IMAT --token-embedding-type Q6_K $F16 $OUT/B4-iq4xs-imat-embdQ6.gguf IQ4_XS

# B5: IQ3_XXS + imatrix
run "B5-iq3xxs-imat" $QZ --imatrix $IMAT $F16 $OUT/B5-iq3xxs-imat.gguf IQ3_XXS

# B6: IQ3_XXS + imatrix + token_embd Q6_K
run "B6-iq3xxs-imat-embdQ6" $QZ --imatrix $IMAT --token-embedding-type Q6_K $F16 $OUT/B6-iq3xxs-imat-embdQ6.gguf IQ3_XXS

# B7: Q4_K_M 无 imatrix (复现当前交付模型, 作为对照)
run "B7-q4km-noimat" $QZ $F16 $OUT/B7-q4km-noimat.gguf Q4_K_M

echo "=============================================="
echo "[$(date '+%H:%M:%S')] ALL QUANTIZATION DONE"
ls -la $OUT/
