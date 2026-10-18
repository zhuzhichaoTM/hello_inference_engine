# 台账契约（Ledger Contract）

> 本文件是 `seven-step-closure` 的参考文件。**要落盘实测数据时读它。**
> 目录：
> 1. 台账解决什么问题
> 2. 完整结构
> 3. measurement_protocol：数字怎么来的
> 4. 逐场景复现命令
> 5. 口径标记
> 6. 按业务流程调整台账

---

## 1. 台账解决什么问题

分析里会出现大量数字：来自文档的、来自实测的、来自公式推导的。
如果不区分来源，几周后没人能回答「这个 47.94 t/s 是哪次测的、什么条件」。

台账是**所有实测数据的唯一载体**，每条都带来源与复现方式。
它带来的三个好处：

1. **可追溯**：任何数字都能追到一次具体测量
2. **可复现**：每次测量都有可执行的命令
3. **可替换**：换项目时只换台账文件，流程代码零改动

---

## 2. 完整结构

```json
{
  "meta": {
    "model": "Qwen2.5-1.5B-Instruct",
    "quant": "Q4_K_M",
    "engine": "llama.cpp b9962-07d937828",
    "device": "Intel Core i7-4770HQ",
    "backend": "cpu",
    "collected": "2026-10-14"
  },

  "theoretical": {
    "vram_bandwidth_gbps": 25.6,
    "weight_gib": 1.039,
    "decode_tps_upper_bound": 23.8
  },

  "baseline_native": {
    "config": "纯自回归，无任何优化",
    "decode_tps": 17.85,
    "bandwidth_utilization": 0.78,
    "confidence": "measured"
  },

  "scenarios": {
    "short_chat": {
      "prompt_tokens": 64, "gen_tokens": 128, "parallel": 1,
      "prefill_tps": 69.13, "decode_tps": 17.54, "ttft_ms": 930.0,
      "cache_type_k": "f16", "cache_type_v": "f16",
      "command": "llama-bench -m $MODEL -ngl 0 -o json -t 4 -p 64 -n 128 -r 5",
      "command_source": "recorded",
      "source": "本机实测 2026-10-14",
      "confidence": "measured"
    },
    "another_scenario": {
      "status": "todo",
      "note": "尚未采集，需在目标机型上补测"
    }
  },

  "curves": {
    "long_context": [
      {
        "ctx": 8192,
        "prefill_tps": 34.65, "decode_tps": 7.22,
        "kv_gib": 0.016,
        "kv_gib_source": "derived",
        "command": "llama-bench -m $MODEL -ngl 0 -o json -t 4 -p 512 -n 64 -d 8192 -r 3",
        "command_source": "recorded",
        "prompt_tokens": 512, "gen_tokens": 64
      }
    ]
  },

  "memory": {
    "breakdown_gib": {
      "main_weight": 1.039, "mmproj": 0.0,
      "kv_cache": 0.031, "runtime_buffer": 0.15
    },
    "peak_used_mib": 1974,
    "vram_total_mib": 16384,
    "safety_margin_mib": 14410,
    "measure_method": "/usr/bin/time -l (maximum resident set size)"
  },

  "measurement_protocol": { "...见 §3..." },

  "extra": {
    "...": "任意附加数据，原样透传给报告"
  }
}
```

**关键约定**

| 约定 | 理由 |
|:---|:---|
| `scenarios` 按任务的 scenario id 组织 | 换任务时只改两边配置，读取代码零改动 |
| 缺数据的场景标 `status: "todo"` | **不编造数字**。下游一律按「无数据」处理 |
| 每个场景记 `command` | 报告里每个数字都要能回答「怎么复现」 |
| 记 `cache_type_k/v` | 否则「同深度对照」的两组命令长得一样，看不出唯一变量 |
| 曲线点记 `kv_gib_source` | 实测增量与公式推算的可信度不同，必须区分 |

---

## 3. measurement_protocol：数字怎么来的

报告里的每个数字都必须能回答三件事：**用什么工具测的、口径是什么、
怎么复现**。缺了这三样，读者看到 17.85 t/s 却不知道它含不含采样、
是均值还是峰值，数字就不可解释、也不可跨配置比较。

```json
"measurement_protocol": {
  "tool": "llama-bench",
  "tool_version": "llama.cpp b9962-07d937828（CPU 后端）",
  "statistic": "每个配置重复 3 次，取 avg_ts（算术均值）",
  "measurement_point": "进程内计时：不含 HTTP 传输、排队与采样开销",
  "backend_label": "CPU 后端",
  "accel_flags": ["-ngl", "0"],
  "threads": 4,

  "flags": [
    {"flag": "-p N", "meaning": "Prefill 口径：纯 prompt 吞吐，不含采样"},
    {"flag": "-n N", "meaning": "Decode 口径：稳态速率，排除首 token"},
    {"flag": "-d N", "meaning": "上下文深度：先填 KV 到 N 再测"},
    {"flag": "-t N", "meaning": "线程数（共享内存后端的关键参数）"},
    {"flag": "-ngl 0", "meaning": "显式钉死纯 CPU 后端"}
  ],

  "conventions": [
    "本工具报的是纯解码吞吐，不含采样与首 token，因此高于端到端墙钟速率。同机实测两者相差 35%，引用必须标明来源。",
    "表中 TTFT 为推导值（prompt_tokens ÷ prefill_tps），不是端到端实测。",
    "采集窗口的 1 分钟系统负载 < 2。负载超标的数据不纳入结论。"
  ]
}
```

**`conventions` 是最容易被跳过、但最有价值的一段**。它记录的是
「这个口径容易在哪儿被误用」。例如：

- 纯解码口径 vs 端到端口径（可差 35%）
- 推导的 TTFT vs 实测的 TTFT
- 测量有效性前提（负载、温度、缓存状态）

---

## 4. 逐场景复现命令

**优先记录真实执行的原始命令**（`command_source: "recorded"`）。

台账没记命令时，框架按场景参数**重建**一条等价命令，
并标注 `command_source: "reconstructed"`，报告里写：
「台账未记录原始命令，以下为按场景参数重建的等价命令」。

**重建命令 ≠ 原始命令。** 不标注就变成了伪造复现记录。

重建时的三个要点：

1. 用**裸命令名**而不是探测到的绝对路径 —— 探测到的是「本机现在装的那个」，
   未必是当初采集用的那个，写绝对路径会误导
2. 还原**唯一变量**（如 KV 量化档位、上下文深度），否则对照组的两条命令
   会长得一模一样
3. 权重路径很长时用 `$MODEL` 占位，在命令块首补一行 `MODEL="..."` 赋值，
   保证整块可复制粘贴直接运行

---

## 5. 口径标记

分析中会同时出现四类数字。**混用它们是最隐蔽的错误来源。**

| 标记 | 含义 | 使用规则 |
|:---:|:---|:---|
| `measured` | 实测 / 实测换算 | 可直接作为结论依据 |
| `derived` | 由实测值推导（如两次测量的比值） | 必须标注推导公式 |
| `model` | 几何公式/物理模型推算 | 仅用于可行性边界判断 |
| `config` | 配置声明（如 `--ctx-size`） | 陈述事实，不是性能数据 |
| `todo` | 待采集 | **不得作为结论依据**，不参与达标判定 |

**达标判定只采信 `measured`。** 这条规则看似保守，但它防止了
「用模型估算值凑达标」这类最难发现的错误。

**复合指标要单独处理**：像「线程缩放 = 4线程 ÷ 最优线程」这类指标，
它的值不是单次测量而是**两次测量的比值**，属于「实验设计的结果」
而非「某个配置的属性」。这类指标应该由台账的多组数据推导，
而不是塞进每个配置的 `measured` 里（那会导致每个配置都要重复声明一遍，
且容易自相矛盾）。

---

## 6. 按业务流程调整台账

台账结构对任何业务流程都适用，只是 `scenarios` / `curves` 的内容不同：

| 流程 | scenarios 里放什么 | curves 里放什么 |
|:---|:---|:---|
| 性能优化（09） | 短/长上下文、并发、持续负载 | 随 ctx 的吞吐与缓存增长 |
| 质量评测（04） | 各评测集 × 各量化档位 | 随量化位宽的质量曲线 |
| 内存治理（08） | 各上下文/并发下的峰值内存 | 随 ctx 的内存占用曲线 |
| 故障定位（12） | 各故障场景（OOM/超时/回退） | 故障前后资源时间线 |
| 成本能效（11） | 各负载下的功耗与吞吐 | 随负载的能效曲线 |

**共同点**：每个场景给 `command`，整批给 `measurement_protocol`，
每个数字带口径标记。这三点与业务流程无关。
