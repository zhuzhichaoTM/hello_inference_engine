# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| main    | ✅         |

本仓库以文档与实战脚本为主，不发布二进制制品。`main` 分支为唯一维护分支。

## Reporting a Vulnerability

**请勿直接提公开 Issue 披露安全问题。**

1. 通过 GitHub [Discussions](https://github.com/zhuzhichaoTM/hello_inference_engine/discussions) 私信 maintainer，或发邮件给仓库所有者（见 GitHub 主页）。
2. 请包含：影响范围（文件/脚本/命令）、复现步骤、潜在影响、已知的修复建议。
3. 我们将在 7 天内确认收悉，30 天内给出修复或缓解方案。

## Scope

- 适用：`deploy/*/` 下可执行脚本（含 `start_*.sh`、`run_*.sh`、`quant_*.py`）的命令注入、路径遍历、敏感信息泄露风险。
- 不适用：第三方模型权重、数据集本身的安全性（请遵循其原始发布方的许可与安全说明）。

## Safe Usage

- 运行 `deploy/` 脚本前请先阅读头部注释，确认模型路径、端口与数据目录。
- 不要在公开日志中粘贴 API Key、内网地址与私有数据集链接；`logs/` 样例需脱敏后提交。
