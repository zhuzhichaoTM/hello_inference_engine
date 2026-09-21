# Contributing to hello_inference_engine

感谢你对本仓库的关注！本仓库是 **大模型推理系统工程化实践知识库**，目标是提供可复现、可验证、可引用的推理部署与评测指南。所有贡献——纠错、补充实测数据、新增硬件/框架实战——都欢迎。

> 风格对标：`ai-dynamo/dynamo` 的 OSS-first 协作模式 —— Issue 先行、PR 小步、CI 必过、Review 公开。

## 1. 参与方式

| 方式 | 适用场景 | 入口 |
|------|----------|------|
| Report an issue | 文档错误、链接失效、实测不可复现 | [Bug / Content Fix 模板](.github/ISSUE_TEMPLATE/bug_report.yml) |
| Propose content | 新增篇章、新增模型部署实战、新增硬件洞察 | [Content Proposal 模板](.github/ISSUE_TEMPLATE/content_proposal.yml) |
| Submit a PR | 直接修文档、补代码、补评测报告 | [PR 模板](.github/PULL_REQUEST_TEMPLATE.md) |
| Ask a question | 部署报错、量化选型、硬件选型咨询 | [Q&A 讨论](https://github.com/zhuzhichaoTM/hello_inference_engine/discussions) |

**首次贡献推荐路径**：`good first issue` → 小幅纠错 PR → 独立篇章 PR。

## 2. 分支与提交规范

```bash
# 1. Fork + 克隆
git clone https://github.com/<you>/hello_inference_engine.git

# 2. 新建分支：一律从 main 切出
git checkout -b docs/<short-topic>   # 文档类
git checkout -b feat/<short-topic>   # 新增实战/代码类
git checkout -b fix/<short-topic>    # 纠错类

# 3. 提交信息：Conventional Commits
# docs(basics): 修正第3章推理技术栈分类
# feat(deploy): 新增 Qwen3-8B vLLM 部署实战
# fix(hw): 修正 NPU 算力数据引用
# feat(benches): 补充 Q4_K_M 压测报告
```

规则：

- 一个 PR 只做一件事，保持可审查（建议 < 500 行变更）。
- 中文提交信息可接受，但推荐 `type(scope): subject` 前缀以便生成 Changelog。
- 实测类内容必须注明环境（GPU/NPU 型号、驱动、框架版本、模型版本）。

## 3. 文档写作规范

详见 [docs/writing-style.md](docs/writing-style.md)，核心要求：

1. 每篇文档头部包含：目标读者、前置知识、预计阅读时间。
2. 所有性能数字必须给出复现方式（命令 + 配置 + 数据集），禁止孤立数字。
3. 中文正文，技术术语保留英文原词，首次出现加中文解释，如 KV Cache（键值缓存）。
4. 图片放同级 `images/` 目录，相对路径引用；超大附件（>10MB）放 Release 不进 git。
5. 新增篇章必须同步更新：本目录 `README.md` + [docs/index.md](docs/index.md)。

## 4. 代码与脚本规范

- `deploy/*/` 下的可执行脚本需 `chmod +x` 且头部含 `set -euo pipefail`。
- Python 代码遵循 `ruff` / `black` 默认风格，提供 `requirements.txt` 或 `uv` 说明。
- 量化/评测脚本必须可一键复现，输出日志样例放 `logs/`。

## 5. CI 检查

PR 会触发 [docs-lint](.github/workflows/docs-lint.yml)：

- Markdown 格式（prettier）、死链检查、敏感大文件拦截（>5MB）。
- 本地预检：`npx prettier --check "**/*.md"`。

## 6. Review 与合并

- 至少 1 位 maintainer approve 后可合并，实测类内容建议 2 位复核。
- 合并策略：Squash and merge，保持 main 历史线性清晰。

## 7. 行为准则

参与即表示同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。如遇不当行为，请联系 maintainer 或按 [SECURITY.md](SECURITY.md) 私下报告。
