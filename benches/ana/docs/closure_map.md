# 七步闭环与业务地图的对应关系

> **目标读者**：熟悉「端侧大模型推理业务地图」、想知道这套框架如何落地其中
> 第三部分的人
> **前置知识**：读过业务地图第一/二/三部分
> **预计阅读**：10 分钟

本文说明：业务地图第三部分「工程闭环地图」的 7 步曲，
如何在 `perfkit` 里被固化成可执行的代码与命令，
以及为什么框架在 7 步之外增加了「环境准备」与「可视化」两个环节。

---

## 1. 一一对应表

| 业务地图（第三部分 3.1） | 框架实现 | 关键交付 |
|:---|:---|:---|
| 0 环境准备 | `scripts/collect_env.sh` + `scripts/build.sh` | 环境画像快照、带宽探针 |
| 1 问题定义 | `perfkit define` | `definition.json`：模型画像、设备画像、KPI 表、验收标准 |
| 2 性能基线 | `perfkit baseline` | `baseline.json`：Baseline 报告、测试场景矩阵、指标基线 |
| 3 瓶颈定位 | `perfkit profiling` + `perfkit diagnose` | `profiling.json`（Profile 报告、Hot Kernel 列表）+ `diagnose.json`（根因报告）+ 报告 §3.9 执行路径分析 |
| 4 优化实施 | `perfkit optimize` | `optimize.json`：优化方案文档、Runtime 配置、回滚方案 |
| 5 收益量化 | `perfkit verify` | `verify.json`：前后对比、收益分析表、质量与资源权衡 |
| 6 回归验收 | `perfkit report`（§6）+ 回归矩阵 | `report.md`：验收报告、回归报告 |
| 7 落地复盘 | `perfkit report`（§7）+ `task_*.yaml` 的 `retrospective` | 案例库、最佳实践、下一轮计划 |
| （可视化） | `perfkit visualize` | `charts.html` |
| （探针） | `perfkit probe` | `output/_env/probe.json`：本机实测带宽 |
| （查询） | `perfkit query` | 跨运行的只读视图（不产出新数据） |

### 1.1 为什么把「瓶颈定位」拆成两个子命令

业务地图的第 3 步同时包含「采集」与「定位」两件事：

> **核心工程活动**：拆解 E2E Latency / 分析 Prefill·Decode / Profiler 采集 /
> 算子分析 / CPU·GPU·NPU 分析 / 检查 Fallback 路径 / 分析数据搬运
>
> **关键分析内容**：计算（GEMM 热点…）、硬件（Compute/Memory Bound…）、
> 软件（CPU fallback…）
>
> **主要输出物**：①性能 Profile 报告 ②Hot Kernel 列表 ③执行路径分析
> ④根因分析报告

前两项输出（Profile 报告、Hot Kernel 列表）是**数据**，
后两项（执行路径分析、根因分析报告）是**结论**。

四件产出物的分工与报告落点：

| 产出物 | 由谁产出 | 报告落点 |
|:---|:---|:---|
| ① 性能 Profile 报告 | `profiling`（四维剖面数据） | §3.2-§3.6 |
| ② Hot Kernel 列表 | `profiling`（按单步耗时降序） | §3.7 |
| ③ 执行路径分析 | `profiling` 的搬运量/Cache/回退数据 + `report` 串联 | §3.8-§3.9 |
| ④ 根因分析报告 | `diagnose`（决策树 + 证据链 + 反证） | §3.10 |

因此框架拆成：

- `profiling` —— 只采集与整理数据，不做判断（产出 ①②，并为 ③ 提供原始数据）
- `diagnose` —— 只做判断，用决策树 + 证据链 + 反证（产出 ④）
- `report` —— 把 ③ 串成可读的路径分析，并统一呈现四件产出物

**拆分的好处**：换硬件重测时，只需重跑 `profiling`，
`diagnose` 的规则（对瓶颈类型的判断逻辑）可复用；
反之调整判定规则时，不必重新采集数据。

### 1.2 为什么把「环境准备」单列

业务地图 3.2 的阶段明细表把「0 环境准备」列为独立阶段，
判据是「模型可稳定加载运行」。框架用一个脚本对应：

```bash
./scripts/collect_env.sh   # 采集 CPU/GPU/内存/工具链，产出设备画像快照
./scripts/build.sh         # 构建带宽探针，自检编译环境
```

**为什么这一步不能省**：设备画像里的 `mem_bandwidth_gbps_peak`
是整个性能模型的**分母**。它填错，后面所有推导都错。
而人工填这个值很容易填成实测值（两者用途不同：标称值用于算上界，
实测值用于算达成率）。

### 1.3 为什么加了「可视化」

业务地图没有单列可视化，但 3.2 表的「关键分析内容」里反复要求
「带宽利用率」「Cache 命中率」「Compute/Memory Bound」这类判断 ——
这些判断在文字表格里很难一眼看出趋势，尤其：

- KV Cache 随 ctx 的**线性增长**（折线图一眼可见，表格要逐行算）
- 显存分解的**占比结构**（堆叠条一眼可见）
- 各 bundle 的**吞吐对比**（条形图一眼可见）
- 算子耗时的**排序**（横条一眼可见）

因此 `perfkit visualize` 生成一张自包含 HTML（内联 SVG，
无外部依赖、无 JS 框架），把六类剖面各画一张图。

### 1.4 为什么把「探针」与「查询」放在七步之外

业务地图的 7 步是**产出新数据 / 新结论**的流程。探针与查询都不产出结论：

- `perfkit probe` 测的是**硬件**（内存/显存带宽、内核效率），与模型无关。
  它服务第 0 步「环境准备」的判据（硬件事实是否明确、环境是否可复现），
  但一次测量可被多次运行复用，因此单独设命令、结果落盘到
  `output/_env/probe.json` 由 baseline 读取。
  把它塞进闭环会让每次跑模型都重测一遍硬件 —— 既慢又无意义。

- `perfkit query` 是**只读视图**：把散落在 `output/<run>/` 的 JSON
  整理成能回答问题的形式（有哪些运行？某指标多少、口径是什么？
  跨运行怎么比？支持哪些后端？）。它不产生新数据，因此不进闭环。

判断标准很简单：**闭环的每一步都应该让分析向前推进一步**，
只提供视图或只测量环境事实的动作，应该独立成命令。

---

## 2. 与业务地图第二部分的接口

业务地图第二部分（业务流程地图）的 13 个流程中，
本框架直接支撑的是 **09 性能分析与优化**（P0 重点流程）：

| 09 流程的要素 | 框架对应 |
|:---|:---|
| 何时进入：时延、吞吐、功耗或资源指标不达标 | `task.scope` + `kpis[].target` |
| 主要产出：瓶颈结论、优化收益对比 | `diagnose.json` + `verify.json` |

同时框架间接支撑另外两个 P0 流程：

| 流程 | 框架如何支撑 |
|:---|:---|
| 03 硬件与推理后端适配 | `device_*.yaml` 换画像 + `cpp/*_probe` 实测该硬件带宽，再跑闭环对比 |
| 10 版本迭代与回归 | 同一 `run_id` 换模型/引擎版本重跑，`verify` 自动出前后对比 |

**与 08 内存与缓存治理的接口**：把 `task.scope` 设为 `memory`，
`diagnose` 会优先命中 `capacity_bound` 规则，`optimize` 优先筛出
KV 量化与显存相关方案。

---

## 3. 与业务地图第一部分的接口

业务地图第一部分（知识地图 L0–L8）为框架提供了判断依据的来源：

| 层 | 框架里用到该层知识的地方 |
|:---|:---|
| L0 数学与 Transformer 基础 | `config.py` 的所有估算公式（KV 公式、上界公式、roofline） |
| L1 模型资产与格式 | `gguf.py`（GGUF 元数据解析）、`model_*.yaml` 的画像字段 |
| L2 算子/图/量化 | `plan_*.yaml` 的量化方案卡片、`profiling` 的算子归因 |
| L3 运行时/内存/KV | `profiling` 的 KV 剖面、`define` 的显存预算 |
| L4 CPU/GPU/NPU 与 Kernel | `cpp/ggml_probe.cpp`、`cpp/cuda_bw_probe.cu`、`device_*.yaml` |
| L5 推理引擎 | `plan_*.yaml` 的启动参数（llama.cpp 的 flag 语义） |
| L6 部署形态与工程化 | `start_optimized.sh`、回归测试矩阵 |
| L7 多模态/Agent | `bundle B3_multimodal`（mmproj 挂载） |
| L8 评测/功耗/安全/成本 | `task.kpis`（含功耗温度）、`acceptance.guardrail`（质量红线） |

**框架不替代知识**：它把知识固化成默认值与检查项。
例如「KV Cache 的 K 不能降到 q4_0」这条 L2/L3 的经验，
写在 `plan_*.yaml` 的 `kv_mixed_q8q4` 卡片的 `risk` 字段里，
每次生成方案时都会被打印出来。

---

## 4. 闭环的「回流」如何实现

业务地图 3.1 的闭环图末尾有一条虚线：`回归验收、复盘与沉淀 → 下一轮优化`。

框架里这条回流通过三个机制实现：

### 4.1 经验回流到方案库

复盘阶段发现的新优化手段，加成 `plan_*.yaml` 的一条卡片。
下次跑闭环时 `optimize` 自动纳入筛选 —— **不需要改代码**。

### 4.2 证据回流到台账

真机采集到的新实测值，写进 `data/reference_measurements.json`。
下次跑 `baseline` 时自动使用 —— **不需要改代码**。

### 4.3 缺口回流到规则

如果某类瓶颈反复出现但 `diagnose` 没有对应规则（`severity` 为
`error` 或结论为空），说明判定规则有缺口，按
[usage.md §5.2](usage.md) 加一条规则。

```text
流程中暴露的问题
    ↓
是数据缺口？ → 补台账 / 跑 live 模式
    ↓ 否
是方案缺口？ → 加 plan 卡片
    ↓ 否
是规则缺口？ → 加 diagnose 规则
    ↓ 否
是能力缺口？ → 回到业务地图 L0–L8 补知识
```

这与业务地图「E → 暴露缺口 → K」的箭头完全对应。

### 4.4 本次运行的回流实例

这一轮分析里，四条回流通道**全部被触发过**，可以作为范例：

| 缺口类型 | 本次的具体表现 | 回流动作 |
| :--- | :--- | :--- |
| **数据缺口** | 部署报告只有零散的单点数字，没有 Prefill 规模扫描、KV 精度同深度对照、并发 A/B | 补跑 13 组 llama-bench 矩阵 + 6 个 server 配置 + 30 分钟压测，写入 `ledger.json` |
| **方案缺口** | 方案库里没有「重编译 FA 量化 kernel」这张卡片，导致 q8q4 掉速 13% 无解释 | 加 `kernel_fa_quants_rebuild` 卡片，并在 `kv_mixed_q8q4` 的 `risk` 里写明 kernel 覆盖缺口 |
| **规则缺口** | `capacity_bound` 把「KV 占每步比例高」（速度问题）与「显存装不下」（容量问题）混为一谈 | 拆出 `kv_pressure` 类型，改 `diagnose.py` 的 `_rule_kv_pressure` |
| **能力缺口** | 几何公式按「64 层全注意力」推 KV，与实测差 2.95× | 给 `kv_cache_bytes` 加 `n_attn_layer` 概念、给 `mem_budget` 加 SSM 固定状态、给 `weight_bytes` 加实测分解优先 —— 并把这四条写进 `docs/methodology.md` §3 |

> **值得注意**：前两条只需改配置，第三条只改一个规则函数，
> 第四条才动到核心公式。**缺口越深，改动越接近 L0–L8 的知识层** ——
> 这正是业务地图「E → K」箭头要表达的意思。

---

## 5. 端侧边界

业务地图明确限定「只覆盖端侧」。框架在这一点上的处理：

| 端侧特征 | 框架如何体现 |
|:---|:---|
| 统一内存（CPU/GPU/NPU 共享物理内存） | `device.memory.unified_memory` 字段；影响 CPU 回退代价的计算 |
| 功耗与热降频 | `device.power_thermal.*`；`task.scenarios` 含 `soak_30min` 稳态场景 |
| App 内存配额（Jetsam/LMK） | `device.constraints.min_safety_margin_gib` 按平台配额设定 |
| NPU 算子覆盖与定点限制 | `device.kind: npu`；`plan_*.yaml` 的 `ondevice_npu_offload` 卡片 |
| 权重 streaming（大模型跑小内存） | `plan_*.yaml` 的 `ondevice_weight_streaming` 卡片，明确其速率受存储 IO 限制 |
| 端云协同 | 框架不做，仅通过 `min_safety_margin_gib` 与「降级策略」间接体现 |

**不在范围内**（与业务地图一致）：数据中心多卡多机并行、集群调度、
K8s 平台、多租户云服务、按 token 计费。

---

## 6. 与其他 P0 流程的组合用法

业务地图 2.4 列出 5 个 P0 重点流程。本框架是 09 性能分析与优化的
完整实现，同时可与其他 P0 流程组合：

```text
03 硬件与后端适配
    └→ 换 device_*.yaml → 跑 define + baseline → 得到该硬件的上界与实测
       └→ 09 性能分析与优化（本框架的完整闭环）

10 版本迭代与回归
    └→ 同一 run_id 换 model 版本 / engine commit 重跑
       └→ verify 自动出前后对比 → 04 质量评测与验收（PPL 守门）

12 故障定位与恢复
    └→ 用 profiling 的「执行路径分析」与「CPU 回退代价」定位异常
       └→ 根因确认后回到 09 做优化
```
