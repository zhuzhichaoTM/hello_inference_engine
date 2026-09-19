#!/usr/bin/env bash
# ==============================================================================
# Qwen3.8-27B 高性能生产启动脚本 (llama.cpp on RTX 4090 24GB)
# 特性：Q4_K_M 主模型 + MTP 投机解码 + mmproj 多模态 + 64K 上下文 + 混合 KV Cache
# ==============================================================================

MODEL_DIR="/root/autodl-tmp/models"
LLAMA_BIN="/root/autodl-tmp/llama.cpp/build/bin"

MAIN_MODEL="${MODEL_DIR}/Qwen3.8-27B-UD-Q4_K_M.gguf"
DRAFT_MODEL="${MODEL_DIR}/mtp-Qwen3.8-27B-Q4_0.gguf"
MMPROJ_MODEL="${MODEL_DIR}/mmproj-F16.gguf"
LOG_FILE="/root/server.log"

export CUDA_VISIBLE_DEVICES=0

# 优雅停止旧实例
pkill -9 llama-server 2>/dev/null
sleep 1

echo "正在启动 Qwen3.8-27B 生产服务..."
echo "  主模型: ${MAIN_MODEL}"
echo "  MTP Draft: ${DRAFT_MODEL}"
echo "  视觉投影: ${MMPROJ_MODEL}"
echo "  上下文长度: 65536 (64K)"

nohup ${LLAMA_BIN}/llama-server \
  -m "${MAIN_MODEL}" \
  --mmproj "${MMPROJ_MODEL}" \
  -md "${DRAFT_MODEL}" \
  --spec-type draft-mtp \
  -ngl 99 \
  -ngld 99 \
  --ctx-size 65536 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --batch-size 1024 \
  --ubatch-size 512 \
  --parallel 1 \
  --host 0.0.0.0 \
  --port 8000 \
  --alias qwen3.8-27b \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "服务已在后台启动 (PID: ${PID})，日志输出至 ${LOG_FILE}"
echo "检查就绪: curl http://127.0.0.1:8000/v1/models"
