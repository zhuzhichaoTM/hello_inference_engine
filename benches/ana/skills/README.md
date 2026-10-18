# skills/ —— 七步闭环技能包

把「端侧大模型推理业务地图」第三部分的**七步工程闭环**固化成一组
可复用的 Agent 技能。装进 `~/.claude/skills/` 后，Claude 会在相关任务上
自动加载对应技能。

## 设计：两层结构

```
skills/
├── seven-step-closure/          ★ 第一层：方法论底座（与领域无关）
│   ├── SKILL.md
│   └── references/
│       ├── step-details.md        七步执行手册
│       ├── evidence-chain.md      证据链与反证条件规范
│       ├── ledger-contract.md     台账契约（实测数据的唯一载体）
│       └── process-catalog.md     13 个业务流程 → 七步的映射表
│
├── llm-perf-optimization/       ★ 第二层：流程 09 性能分析与优化
│   ├── SKILL.md
│   └── references/
│       ├── cli-reference.md       perfkit 命令速查
│       ├── bottleneck-rules.md    瓶颈判定规则 + 反证条件
│       └── hardware-backends.md   多后端差异
│
├── llm-backend-adaptation/      流程 01/03 选型规划与硬件后端适配
├── llm-quality-eval/            流程 04/10 质量评测验收与版本回归
├── llm-memory-cache/            流程 08/11 内存缓存治理与成本能效
└── llm-incident-triage/         流程 12/13 故障定位与安全可靠性
```

**为什么分两层**：七步骨架与领域无关 —— 它回答的是「环境是什么、
要改善什么、现在多少、为什么、改什么、真的改善了吗、留下什么」。
换业务流程时只有三处会变（KPI 是什么、剖面采什么、判定规则是什么），
骨架与三条规则不变。

所以第一层是**方法论**，第二层是**方法论在具体流程下的实例化**。
新增一个业务流程时，写一份第二层的技能即可，不必重写方法论。

## 覆盖的业务流程

来自业务流程地图的四类共 13 个流程：

| 类别 | 流程 | 对应技能 |
|:---|:---|:---|
| 交付类 | 01 模型选型与部署规划 | `llm-backend-adaptation` |
| | 02 模型上线与交付 | `llm-quality-eval` |
| | 03 硬件与推理后端适配 | `llm-backend-adaptation` |
| | 04 模型质量评测与验收 | `llm-quality-eval` |
| | 05 AI 应用链路交付 | `llm-perf-optimization` |
| 运行类 | 06 在线请求处理 | `llm-perf-optimization` |
| | 07 线上监控与容量治理 | `llm-memory-cache` |
| | 08 内存与缓存治理 | `llm-memory-cache` |
| 优化类 | **09 性能分析与优化** | **`llm-perf-optimization`** |
| | 10 版本迭代与回归 | `llm-quality-eval` |
| | 11 成本、能效与业务价值优化 | `llm-memory-cache` |
| 保障类 | 12 故障定位与恢复 | `llm-incident-triage` |
| | 13 安全、隐私与可靠性治理 | `llm-incident-triage` |

映射的完整说明（每个流程的 KPI 建议、剖面维度、判定要点）见
`seven-step-closure/references/process-catalog.md`。

## 安装

**方式一：软链到用户技能目录**（推荐，改动即时生效）

```bash
ln -s "$(pwd)/skills/seven-step-closure"        ~/.claude/skills/
ln -s "$(pwd)/skills/llm-perf-optimization"     ~/.claude/skills/
ln -s "$(pwd)/skills/llm-backend-adaptation"    ~/.claude/skills/
ln -s "$(pwd)/skills/llm-quality-eval"          ~/.claude/skills/
ln -s "$(pwd)/skills/llm-memory-cache"          ~/.claude/skills/
ln -s "$(pwd)/skills/llm-incident-triage"       ~/.claude/skills/
```

**方式二：复制**

```bash
cp -r skills/* ~/.claude/skills/
```

**方式三：项目级**（只在本项目生效）

```bash
mkdir -p .claude/skills && cp -r skills/* .claude/skills/
```

安装后验证：让 Claude 描述一下可用的技能，或直接提一个相关需求，
例如「帮我分析这个模型的推理性能瓶颈」。

## 触发示例

| 用户说法 | 触发的技能 |
|:---|:---|
| 「分析一下 Qwen2.5-1.5B 在 CPU 上的推理性能瓶颈」 | `llm-perf-optimization` |
| 「Decode 只有 7 t/s，怎么优化」 | `llm-perf-optimization` |
| 「这个 27B 模型能跑在 24G 卡上吗」 | `llm-backend-adaptation` |
| 「换到手机 SoC 上要改什么」 | `llm-backend-adaptation` |
| 「量化后质量掉没掉，怎么测」 | `llm-quality-eval` |
| 「升级引擎后要做什么回归」 | `llm-quality-eval` |
| 「显存占用一直涨，是不是泄漏」 | `llm-memory-cache` |
| 「线上 OOM 了，帮我排查」 | `llm-incident-triage` |
| 「做一次完整的工程闭环分析」 | `seven-step-closure` + 具体流程技能 |

## 与框架的关系

`llm-perf-optimization` 配套本项目根目录的 `perfkit` 框架 ——
技能讲**怎么做与为什么**，框架负责**实际执行**。两者配合可以真正跑完七步：

```bash
python3 -m perfkit all --config configs/qwen2.5-1.5b-cpu --run my-run
```

其余流程技能目前是**方法论 + 判据**形式（不含可执行框架），
因为它们的度量方式更依赖具体业务（评测集、监控系统、日志平台）。
`perfkit` 的扩展机制（加判定规则、加硬件后端、加方案卡片）
同样适用于把这些流程也做成可执行的。

## 技能写作的三条原则

新增或修改技能时遵循：

1. **解释 why 而不是堆 MUST** —— 说明「为什么这步重要」比「必须这么做」有效。
   Claude 有很强的推断能力，给它理由它能适配新情况；给它规则它只能照做
2. **给判据而不是给步骤** —— 判据能适配新情况，步骤不能
3. **写清边界** —— 什么情况不适用，比写它适用什么更有价值

另外：**技能里不写「算错了但看起来合理」的内容**。
本技能包中所有数字都来自真实测量或明确标注的推导，
所有教训都来自实践中真实踩过的坑（例如「算力达成率恒等于带宽利用率」
这个错误，就完整记录在 `evidence-chain.md` 里）。
