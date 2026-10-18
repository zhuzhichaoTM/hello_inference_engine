# perfkit 常用命令（make 是可选糖，所有命令都能直接跑）
# ---------------------------------------------------------------------------
.PHONY: help selftest closure closure-probe closure-live env build probe query backends skills skills-install clean report charts

RUN ?= qwen3.8-27b-decode-opt
CONFIG ?= configs
PY ?= python3

help:  ## 显示本帮助
	@echo "perfkit 常用命令："
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "变量：RUN=$(RUN)  CONFIG=$(CONFIG)  PY=$(PY)"
	@echo "示例：make closure RUN=qwen2.5-1.5b-cpu CONFIG=configs/qwen2.5-1.5b-cpu"
	@echo "      make backends CONFIG=configs/qwen2.5-1.5b-cpu"

selftest:  ## 框架自检（不依赖 GPU，校验公式与两套参考配置可跑通）
	$(PY) scripts/selftest.py

env:  ## 采集环境画像 → output/_env/
	bash scripts/collect_env.sh

build:  ## 构建 CPU 带宽探针
	bash scripts/build.sh

probe: build  ## 构建并运行带宽探针（结果落盘，baseline 会自动引用）
	$(PY) -m perfkit probe --config $(CONFIG) --run $(RUN) --size-gib 2

query:  ## 查看所有运行
	$(PY) -m perfkit query --list

backends:  ## 查看已注册的硬件后端与当前配置用的是哪个
	$(PY) -m perfkit query --config $(CONFIG) --backends

closure:  ## 七步闭环（Shell 编排：环境快照 + 探针自检 + 七步）
	bash scripts/run_closure.sh --run $(RUN) --config $(CONFIG) --skip-probe

closure-probe:  ## 七步闭环（含带宽探针构建与自检）
	bash scripts/run_closure.sh --run $(RUN) --config $(CONFIG)

closure-live:  ## 七步闭环（真机模式，需推理服务已启动）
	bash scripts/run_closure.sh --run $(RUN) --config $(CONFIG) --mode live

report:  ## 只看报告
	@cat output/$(RUN)/report.md

charts:  ## 打开可视化图表
	@open output/$(RUN)/charts.html 2>/dev/null || \
	  xdg-open output/$(RUN)/charts.html 2>/dev/null || \
	  echo "请手动打开 output/$(RUN)/charts.html"

skills:  ## 校验并打包技能包 → skills/dist/*.skill
	$(PY) skills/validate_skills.py
	$(PY) skills/pack_skills.py

skills-install:  ## 把技能包软链到 ~/.claude/skills/
	@for s in seven-step-closure llm-perf-optimization llm-backend-adaptation \
	          llm-quality-eval llm-memory-cache llm-incident-triage; do \
	  t="$$HOME/.claude/skills/$$s"; \
	  if [ -e "$$t" ] || [ -L "$$t" ]; then echo "  skip $$s (已存在)"; \
	  else ln -s "$(CURDIR)/skills/$$s" "$$t" && echo "  linked $$s"; fi; \
	done

clean:  ## 清理构建产物与运行产物
	rm -rf build output skills/dist
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "已清理 build/ output/ 与 __pycache__/"
