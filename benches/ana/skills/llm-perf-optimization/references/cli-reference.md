# perfkit 命令速查

> 本文件是 `llm-perf-optimization` 的参考文件。**要跑框架时读它。**
> 目录：
> 1. 配置目录结构
> 2. 七步子命令
> 3. 辅助命令（probe / query）
> 4. 常见操作路径
> 5. 产物说明
> 6. 扩展方式

---

## 1. 配置目录结构

一套配置 = 一个目录，内含四份 YAML + 一份台账：

```text
configs/<project>/
├── model_*.yaml      模型画像：结构超参、量化档位、权重文件
├── device_*.yaml     设备画像：带宽/算力/内存/功耗/约束
├── task_*.yaml       任务定义：KPI、验收、场景矩阵、复盘
├── plan_*.yaml       方案库：方案卡片 + bundle 模板
└── ledger.json       实测台账（该套配置自带，优先于全局台账）
```

**文件名按角色前缀解析**（`model_*.yaml` 等），所以换硬件时把
`device_rtx4090.yaml` 改成 `device_cpu_intel.yaml` 即可，无需改代码。
同一目录下同类配置只能有一份 —— 多份会**报错**而不是静默取第一个，
避免「改了 A 文件、跑的是 B 文件」这类难查的问题。

多套配置靠**目录隔离**：

```bash
python3 -m perfkit all --config configs              --run run-a
python3 -m perfkit all --config configs/qwen2.5-1.5b-cpu --run run-b
```

---

## 2. 七步子命令

### `perfkit define` —— 第 1 步 问题定义

```bash
python3 -m perfkit define --config configs --run my-run
python3 -m perfkit define --config configs --run my-run --from-gguf model.gguf
```

产出：画像汇总、内存预算、KV 精度阶梯、roofline 判定、逐算子字节占比、
KPI 表、模型/实测对账、参考源矛盾。

`--from-gguf` 会读模型文件的元数据，把「反推的架构超参」换成真值 ——
这是消除模型/实测差额的根治手段。

### `perfkit baseline` —— 第 2 步 性能基线

```bash
# replay：从台账读（默认，无需真机）
python3 -m perfkit baseline --config configs --run my-run

# live：真机采集
python3 -m perfkit baseline --config configs --run my-run --mode live \
    --bench-bin /path/to/llama-bench

# 只跑指定场景（冒烟验证，避免一上来就等 10 分钟持续负载）
python3 -m perfkit baseline --run my-run --mode live --only short_chat

# 采集后导出成台账，供后续 replay 复用
python3 -m perfkit baseline --run my-run --mode live \
    --write-ledger configs/<project>/ledger.json
```

其他参数：`--cache-type-k/v` 覆盖 KV 量化档位、`--extra` 追加引擎参数、
`--ledger` 指定台账路径。

### `perfkit profiling` —— 第 3 步 性能剖面

```bash
python3 -m perfkit profiling --config configs --run my-run
python3 -m perfkit profiling --config configs --run my-run --mode live
```

产出四维剖面：阶段 / 逐层 / 算子 / 硬件（含**带宽侧与算力侧**）。

算子剖面同时给出两套占比，口径不同、不可混用：

| 字段 | 分母 | 用途 |
|:---|:---|:---|
| `pct_of_layer` | 单层权重 | 回答「一层内部谁占大头」 |
| `pct_of_step` | 全部权重（含词表嵌入） | 回答「整个 Decode 单步谁占大头」，与 `ms_per_step_all_layers` 同口径 |

两者之和都应为 100%（后者需加上 `per_layer.embedding_share_pct`）。
报告 §3.7 的 Hot Kernel 列表按 `pct_of_step` 降序排列。

### `perfkit diagnose` —— 第 3 步 瓶颈定位

```bash
python3 -m perfkit diagnose --config configs --run my-run
```

产出带证据链的结论：主瓶颈 + 全部 findings（含 confidence / evidence /
counter_evidence）+ **优化路线的推导过程**（从结论到方向，每一跳可追溯）。

### `perfkit optimize` —— 第 4 步 方案

```bash
python3 -m perfkit optimize --config configs --run my-run
```

产出：适用方案（按对症程度排序）+ 候选 bundle 对比 + 选定配置的启动命令
+ 回滚命令 + `start_optimized.sh`。

> `start_optimized.sh` 用 `SERVER_BIN="${SERVER_BIN:-llama-server}"` 定位二进制，
> 不写死 `./llama-server`（那个路径几乎从不成立）。执行时若二进制不在 PATH：
> `SERVER_BIN=/path/to/build/bin/llama-server bash start_optimized.sh`。
>
> **检查清单**：脚本里必须能看到后端对应的卸载标志（GPU 是 `-ngl 99`，
> CPU 是 `-ngl 0`）。漏写不会报错 —— llama-server 默认 `n_gpu_layers=0`，
> 会静默把整个模型跑在 CPU 上。`scripts/selftest.py` 第 22 组专门拦这个。

### `perfkit verify` —— 第 5 步 验证

```bash
python3 -m perfkit verify --config configs --run my-run
```

产出：逐 KPI 前后对比（带口径标记）+ 收益归因 + 副作用清单 + 验收判定。

### `perfkit report` / `perfkit visualize` —— 第 6 步 复盘

```bash
python3 -m perfkit report    --config configs --run my-run   # report.md
python3 -m perfkit visualize --config configs --run my-run   # charts.html
```

报告结构固定为七步闭环 0-7 + 附录 A（复现命令）/ B（数据可信度）/ C（产物清单）。

### `perfkit all` —— 一键串联

```bash
python3 -m perfkit all --config configs --run my-run
python3 -m perfkit all --config configs --run my-run --mode live
```

---

## 3. 辅助命令

### `perfkit probe` —— 硬件带宽探针

```bash
python3 -m perfkit probe --config configs --run my-run --size-gib 2
```

结果落盘 `output/_env/probe_<kind>.json`（**按后端分文件**），
`baseline` 会自动读取并写入报告 §0.2。

| 后端 | 探针 | 关键指标 | 测什么 |
|:---|:---|:---|:---|
| `cpu` / `soc` | `ggml_probe` | `stream-file` | mmap 顺序读真实权重文件（**硬件上限**，反推上界用它） |
| | | `stream` | 匿名缓冲区顺序读（对照；差值 = mmap 路径开销） |
| | | `gemv-scalar` | 标量 GEMV（**下界**，跨架构可比） |
| | | `gemv-simd` | SIMD GEMV（**可达参照**，判断内核空间用它） |
| `gpu` | `cuda_bw_probe` | `read-bw` | 显存**纯读**带宽（必须用这个，拷贝带宽会低估 2 倍） |
| `npu` | — | — | 无标准接口，返回「未采集」 |

**为什么按后端分文件**：CPU 探针测主机内存路径、GPU 探针测显存路径，
两者不可互换。混存会导致「拿 CPU 的数字算 GPU 的达成率」，
得出 4963% 这类荒谬比值。读取时框架还会校验文件里的 `backend` 字段。

**怎么读**：先看「交叉校验」行 —— 探针值 vs 真实推理反推带宽
（`decode_tps × 权重体积`）。比值 ≈1 说明探针可信；>1.05 说明探针在
负载下采集（以推理反推值为准）；>3 说明数据来源不匹配。

**三层参照（不能混用）**：

| 参照 | 用途 |
|:---|:---|
| 顺序读 | 硬件天花板 → 反推 Decode 上界 |
| SIMD GEMV | 正常优化内核能拿到多少 → 判断「还该不该优化内核」 |
| 标量 GEMV | 完全不优化能拿到多少 → 跨架构横向对比 |

**只给标量会高估优化空间**：实测标量仅占顺序读 6%，拿它当基准会误判
「还有 94% 空间」；而 SIMD 已达 25%，生产内核更高。判断优化方向看
**SIMD 内核效率**，不是标量。

### `perfkit query` —— 跨运行查询

```bash
python3 -m perfkit query --list                    # 所有运行及摘要
python3 -m perfkit query --run X --kpi decode_tps  # 某指标的前后值 + 口径
python3 -m perfkit query --runs a,b,c              # 横向对比（带口径标记）
python3 -m perfkit query --backends                # 硬件后端清单
python3 -m perfkit query --run X --facts           # 配置事实 + 产物清单
```

设计约束：**只读**、**不猜**（查不到就明说无数据，不回退到估算值）、
**带口径**（每个数字带来源标记）。

---

## 4. 常见操作路径

### 换模型

```bash
cp -r configs configs/my-project
# 改 model_*.yaml 的 architecture / artifacts / precision_ladder
# 改 ledger.json 的 meta 与 scenarios
python3 -m perfkit define --config configs/my-project --run new --from-gguf m.gguf
python3 -m perfkit all --config configs/my-project --run new
```

### 换硬件

```bash
cp -r configs configs/my-hw
mv configs/my-hw/device_rtx4090.yaml configs/my-hw/device_myhw.yaml
# 改 device_*.yaml：kind / 带宽 / 算力 / 内存 / 功耗 / 约束
python3 -m perfkit query --config configs/my-hw --backends   # 确认后端解析正确
python3 -m perfkit all --config configs/my-hw --run new
```

`kind` 决定后端行为：`gpu` / `cpu` / `hybrid`（Apple 统一内存）/
`soc`（移动 SoC）/ `npu`。后端差异见 `hardware-backends.md`。

### 换任务目标

```bash
# 改 task_*.yaml 的 scope: decode | prefill | kv_cache | memory | e2e | power
# 改 KPI 表与验收标准
```

`scope` 是 `diagnose` 的分支开关，也是 `optimize` 的筛选依据。

### 加一个优化方案

在 `plan_*.yaml` 的 `plans` 里加一张卡片即可，**无需改代码**：

```yaml
- id: my_plan
  name: 方案名
  category: kernel | speculative | quantization | kv_cache | scheduling | ondevice
  applicable_when: [memory_bound, long_context]   # 触发条件
  gain: {decode_tps: 1.2}
  cost: ...
  risk: ...
  rollback: ...
  source: "[M] 实测..."
```

`applicable_when` 的标签来自 `diagnose` 的 `optimization_route`
与 findings 的 `type`/`subtype`。

---

## 5. 产物说明

```text
output/<run_id>/
├── definition.json      画像 / 预算 / roofline / KPI / 对账
├── baseline.json        基线数据 + 测量协议 + 环境上下文
├── profiling.json       四维剖面（含算力侧分析）
├── diagnose.json        结论 + 证据链 + 反证 + 路线推导
├── optimize.json        方案选型 + bundle 对比 + 启动命令
├── start_optimized.sh   可直接执行的启动脚本
├── verify.json          前后对比 + 副作用 + 验收判定
├── report.md            七步闭环报告
├── charts.html          自包含可视化（7 张内联 SVG）
├── raw/                 live 采集的原始输出
└── logs/                Shell 编排模式下的过程日志
```

`output/_env/probe.json` 是**跨运行共享**的硬件事实（探针测的是硬件，
与模型无关，一次测量可被多次运行复用）。

---

## 6. 扩展方式

### 加一条瓶颈判定规则

```python
# perfkit/commands/diagnose.py
def _rule_my_bottleneck(p: dict, d: dict) -> dict | None:
    if 某个条件:
        return {
            "type": "my_bottleneck",
            "severity": "primary",        # primary|secondary|phase_scope|watch
            "confidence": "high",
            "evidence": ["具体数字1", "具体数字2"],   # 必须带数字
            "reasoning": "为什么这构成瓶颈",
            "counter_evidence": "什么证据会推翻它",   # 必填
        }
    return None

RULES = [("my_bottleneck", _rule_my_bottleneck), ...]   # 顺序 = 优先级
```

同时要在 `ROUTE_RULES` 里加一条「结论 → 方向 → 理由」的映射，
否则优化路线里不会出现对应类别。

### 加一类硬件后端

```python
# perfkit/backends.py
class MyBackend(Backend):
    kind = "myaccel"
    label = "自研加速器"
    memory_label = "加速器内存"

    def accel_flags(self) -> list[str]: return ["-ngl", "0"]
    def mem_snapshot(self): return None          # 无标准接口时返回 None
    def fallback_is_cheap(self) -> bool: return True

BACKENDS["myaccel"] = MyBackend
```

然后改 `device_*.yaml` 的 `kind`。**流程代码零改动。**

### 自检

```bash
python3 scripts/selftest.py      # 94 项断言，不依赖真机
```

自检覆盖：量化位宽映射、KV 几何公式、理论上界、roofline 拐点、
内存预算、参数展平、多后端抽象、算力侧恒等关系、测量协议与复现命令、
配置目录解析。**改动估算公式后务必跑一遍** —— 性能框架最危险的失败模式
是「算错了但看起来合理」。
