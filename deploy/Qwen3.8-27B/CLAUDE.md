# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 目的与项目概况
- **目标模型**：Qwen3.8-27B (27.32B 参数)
- **部署环境**：云端单卡 NVIDIA GeForce RTX 4090 (24GB GDDR6X)
- **推理引擎**：llama.cpp (CUDA 后端，sm_89 架构编译)
- **量化策略**：`Q4_K_M` 甜点档位主模型 + `Q4_0` MTP Draft 投机解码加速

## 云端算力登录信息
- **SSH 指令**：`ssh -p 24171 root@connect.bjb1.seetacloud.com`
- **密码**：`CZ7wtR+cq8Bj`
- **主要工作目录**：数据盘 `/root/autodl-tmp/`（模型在 `/root/autodl-tmp/models/`，源码在 `/root/autodl-tmp/llama.cpp/`）

## 常用服务管理与开发命令

### 1. 启动推理服务 (llama-server)
包含全层显卡卸载、MTP 投机解码、mmproj 多模态、64K 上下文、混合 KV-Cache 量化、Flash Attention：
```bash
cd /root/autodl-tmp/llama.cpp/build/bin

./llama-server \
  -m /root/autodl-tmp/models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --mmproj /root/autodl-tmp/models/mmproj-F16.gguf \
  -md /root/autodl-tmp/models/mtp-Qwen3.8-27B-Q4_0.gguf \
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
  --alias qwen3.8-27b
```

### 2. 性能压测与基准评估
```bash
# 执行原生 Prefill/Decode 基准压测
/root/autodl-tmp/llama.cpp/build/bin/llama-bench \
  -m /root/autodl-tmp/models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  -ngl 99 -p 512 -n 128 -fa 1
```

### 3. API 调用与验证
```bash
# 检查模型挂载
curl http://127.0.0.1:8000/v1/models

# 发送对话请求 (OpenAI 兼容)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.6,
    "max_tokens": 128
  }'
```
