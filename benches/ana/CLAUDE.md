GG# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 目的与项目概况
- **目标模型**：Qwen3.8-27B (27.32B 参数)
- **部署环境**：云端单卡 NVIDIA GeForce RTX 4090 
- **推理引擎**：llama.cpp (CUDA 后端，sm_89 架构编译)

## 云端算力登录信息
- **SSH 指令**：`ssh -p 37607 root@connect.cqa1.seetacloud.com`
- **密码**：`4g3+Kjz0CNHl`
- **主要工作目录**：数据盘 `/root/autodl-tmp/`（模型在 `/root/autodl-tmp/models/`，源码在 `/root/autodl-tmp/llama.cpp/`）
