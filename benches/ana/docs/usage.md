# 使用文档

> **目标读者**：要跑通框架并做二次开发的工程师  
> **前置知识**：读过 [README.md](../README.md)；了解 Prefill/Decode 与量化基本概念  
> **预计阅读**：20 分钟

---

## 1. 命令行接口

### 1.1 全局选项

两种写法等价（全局选项同时挂在顶层与各子命令上）：

```bash
python3 -m perfkit --run my-run define
python3 -m perfkit define --run my-run
```

| 选项             | 默认          | 说明                     |
| :------------- | :---------- | :--------------------- |
| `--config DIR` | `configs`   | 配置目录                   |
| `--run ID`     | 取 `task.id` | 运行标识，决定 `output/<ID>/` |

### 1.2 子命令

```bash
python3 -m perfkit --help          # 子命令一览
python3 -m perfkit <cmd> --help    # 某子命令的参数
```

| 子命令         | 关键参数                                                                                                                  | 说明                                   |
| :---------- | :-------------------------------------------------------------------------------------------------------------------- | :----------------------------------- |
| `define`    | `--from-gguf PATH`                                                                                                    | 问题定义；给 GGUF 路径则核验架构超参                |
| `baseline`  | `--mode replay\|live`、`--ledger PATH`、`--bench-bin PATH`、`--only ID[,ID...]`、`--cache-type-k/v`、`--write-ledger PATH` | 基线；live 模式调 `llama-bench`            |
| `profiling` | `--mode replay\|live`、`--baseline PATH`                                                                               | 四维剖面；live 模式采样 `nvidia-smi dmon`     |
| `diagnose`  | —                                                                                                                     | 瓶颈判定                                 |
| `optimize`  | —                                                                                                                     | 方案生成（读 `plan_*.yaml`）                |
| `verify`    | —                                                                                                                     | 收益量化                                 |
| `report`    | —                                                                                                                     | 报告导出                                 |
| `visualize` | —                                                                                                                     | 生成 `charts.html`                     |
| `probe`     | `--size-gib`、`--timeout`                                                                                              | 硬件带宽探针；结果落盘 `output/_env/probe_<kind>.json` |
| `query`     | `--list`、`--kpi`、`--runs`、`--backends`、`--facts`                                                                      | 跨运行只读查询                              |
| `all`       | 同 baseline/profiling 的 `--mode`                                                                                       | 一键串联                                 |

---

## 2. 配置文件详解

四份配置分工明确，改哪份取决于你要换什么。

当前仓库里有**五套参考配置**（同一套流程代码，只换 YAML）：

| 目录 | 后端 | 演示的重点 |
| :--- | :--- | :--- |
| `configs/` | NVIDIA GPU | 纯文本：显存墙、KV 量化、MTP、并发 |
| `configs/qwen3.8-27b-multimodal/` | NVIDIA GPU | 多模态：图像 token 杠杆、两段 TTFT、分辨率—质量 |
| `configs/qwen3.8-27b-multimodal-uplimit/` | NVIDIA GPU | 极限：草稿深度拐点、步耗时模型 |
| `configs/qwen3.8-27b-multimodal-newarch/` | NVIDIA GPU | 可达性：位宽—质量曲线、目标反解 |
| `configs/qwen2.5-1.5b-cpu/` | x86 CPU | CPU：线程缩放、负载敏感性、深度衰减 |

每套目录自带 `ledger.json`（实测台账），互不干扰。

### 2.1 `model_*.yaml` —— 模型画像

```yaml
model:
  name: <模型名>
  params_total: <总参数量>          # 含未启用张量（如 MTP 预测块）
  params_effective: <有效参数量>     # 可选：剔除未启用张量后的前向参数量
  quantization: Q4_K_M
  file: /path/to/model.gguf
architecture:            # 结构超参 —— 整个性能模型的基础
  arch: <GGUF 里的 general.architecture>
  n_layer: <层数>
  hidden_size: <hidden>
  intermediate_size: <FFN 中间维>
  n_head: <Q 头数>
  n_head_kv: <KV 头数>             # GQA：决定 KV Cache 大小（端侧最敏感）
  head_dim: <每头维度>
  vocab_size: <词表>
  tie_embeddings: false
  max_ctx_native: <原生上下文>
  max_ctx_deployed: <部署上下文>
  # ---- 以下是「混合架构」才需要的字段（纯 Transformer 可全部省略）----
  hybrid: true
  n_attn_layer: <全注意力层数>       # 只有这些层有随 ctx 增长的 KV
  n_ssm_layer: <SSM/线性注意力层数>
  attn_out_dim: <注意力输出维度>      # 若 n_head×head_dim ≠ hidden 则必填
  ssm_inner_size: <SSM 内部维>
  ssm_state_size: <SSM 状态维>
  ssm_conv_kernel: <卷积核>
  embedding_bytes_measured: <嵌入层实测字节>  # 输入/输出表量化档位不同时用
artifacts:               # 权重文件清单
  - {role: main, file: model.gguf, bytes: <文件字节>}
precision_ladder:        # 量化档位对照（精度-性能权衡用）
  - {quant: Q4_K_M, bits_per_weight: 4.8125, est_weight_bytes: <字节>}
weight_breakdown:        # 可选：GGUF 张量实测分解（混合架构强烈建议）
  total_gib: <有效权重 GiB>
  ops:
    - {op: ffn_down, kind: dense, gib: 3.132, n_layer: 64, params_m: 5793.4}
    - {op: attn_qkv, kind: ssm,   gib: 1.382, n_layer: 48, params_m: 2516.6}
  per_layer_mib: [<blk.0 字节>, <blk.1 字节>, ...]
gguf_verification:       # 可选：`--from-gguf` 的核验结果，固化后本地 replay 也能显示「已核验」
  all_match: true
  extracted: {n_layer: 64, hidden_size: 5120, ...}
calibration:             # 校准锚点（模型与实测不符时显式记录，见 §7）
  kv_cache: {ctx: <ctx>, k_bits: 8.5, v_bits: 4.5, gib: <实测>, geometric_gib: <模型>}
```

**必须填对的字段**（填错会直接导致结论错误）：

| 字段 | 影响 |
| :--- | :--- |
| `n_layer` / `hidden_size` | 逐层权重体积 → 逐层耗时归因 |
| `n_head_kv` / `head_dim` | **KV Cache 大小 → 显存预算与长上下文可行性** |
| `intermediate_size` | FFN 权重占比 → 算子归因 |
| `vocab_size` / `tie_embeddings` | 嵌入层体积（常被忽略的 20%+ 权重） |
| `max_ctx_native` | 上下文上限的硬约束 |

**混合架构的三个坑**（本仓库在 Qwen3.8-27B 上踩过，KV 因此被高估 4×）：

| 坑 | 症状 | 正确做法 |
| :--- | :--- | :--- |
| 把 `n_layer` 当全注意力层数 | KV 容量算出来比实测大数倍 | 填 `n_attn_layer`（只有全注意力层有 KV；SSM 层是固定大小状态） |
| 忽略 `head_dim ≠ hidden/n_head` | 校验报「n_head×head_dim ≠ hidden_size」 | 填 `attn_out_dim`（注意力输出经 attn_o 投影回 hidden） |
| 用单一 `embedding_bits` 估嵌入层 | 输入 Q4_K + 输出 Q6_K 时误差可达 30% | 填 `embedding_bytes_measured`（GGUF 张量表直接累加） |

> **建议**：这些值若为反推值，务必用
> `python3 -m perfkit define --from-gguf <model.gguf>` 核验。
> 核验结果可以固化到 `gguf_verification` 段 —— 这样换机器 replay 时报告仍显示「已核验」，
> 而不必每次重读 16GB 的模型文件。

### 2.2 `device_*.yaml` —— 设备画像

```yaml
device:
  name: NVIDIA GeForce RTX 4090
  kind: gpu              # cpu | gpu | npu | dsp | hybrid
memory:
  vram_total_mib: 24564
  mem_bandwidth_gbps_peak: 1008   # 标称峰值 —— roofline 的分母
  unified_memory: false
compute:
  fp16_tflops_peak: 165.2
power_thermal:
  tdp_w: 450
  thermal_throttle: false         # 端侧设 true
constraints:
  max_vram_gib: 20.0
  min_safety_margin_gib: 4.0
  host_device_transport_gbps: 26  # 用于量化 CPU 回退代价
```

**`mem_bandwidth_gbps_peak` 是最关键的字段**：它是 `tps = 带宽 ÷ 权重体积`  
这个上界的唯一分母。填标称值（厂商标称），不要填实测值 ——  
实测值用于计算「达成率」，两者用途不同。

设备切换模板（**字段示意，不是已验证配置**）：

```yaml
# 端侧 SoC / 统一内存设备：按厂商规格填写，勿直接抄数值
memory:
  ram_total_gb: <设备内存>
  unified_memory: true
  mem_bandwidth_gbps_peak: <标称内存带宽>
compute:
  npu_tops: <标称 NPU 算力>
power_thermal:
  tdp_w: <整机功耗>
  battery: true
  thermal_throttle: true
constraints:
  max_vram_gib: <内存上限>          # 统一内存设备填可用内存配额
  min_safety_margin_gib: <安全冗余>
  host_device_transport_gbps: <等价搬运带宽>  # 统一内存可填内存带宽
```

> ⚠️ 本仓库只验证过 **GPU** 与 **x86 CPU** 两套配置。SoC/NPU 的数值
> 必须来自厂商规格或本机实测，不能沿用示例值；未经 `perfkit probe`
> 与真机闭环验证的配置，不应作为结论依据。

### 2.3 `task_*.yaml` —— 任务定义

```yaml
task:
  id: qwen3.8-27b-decode-opt
  scope: decode          # prefill | decode | kv_cache | memory | e2e | power
  backend: gpu
  priority: latency      # latency | throughput | memory | power | cost
scenarios:               # 测试场景矩阵（baseline/verify 共用，保证可比）
  - {id: short_chat, prompt_tokens: 90, gen_tokens: 256, parallel: 1, weight: 0.30}
kpis:                    # KPI 定义（口径必须唯一）
  - key: decode_tps
    unit: tokens/s
    direction: higher
    target: 45
    definition: 稳态自回归生成速率，排除首 token
acceptance:
  must_pass: [decode_tps >= 45, peak_vram_mib <= 20480]
  guardrail: [ppl_delta_pct <= 2.0, {no_oom: true}]
```

**`scope` 是分支开关**：改成 `kv_cache` 后，`diagnose` 会优先命中  
KV 相关规则，`optimize` 会优先筛出 `kv_cache` 类方案。

**`retrospective` 是复盘内容**（项目级知识）：

```yaml
retrospective:
  tradeoffs:                  # 方案取舍：选择 / 放弃的选项 / 理由
    - {point: Decode 加速手段, chose: MTP 投机解码, rejected: 纯调 kernel, why: ...}
  risks:                      # 风险点：触发条件 / 影响 / 应对
    - {risk: 长上下文质量回退, trigger: ctx > 16K, impact: 中段信息遗漏, mitigation: ...}
  lessons:                    # 可复用经验（纯文本列表）
    - "先算再测：动手前用带宽÷权重体积算出理论上界"
  next_steps:                 # 后续迭代方向
    - {direction: Prefix Cache, gain: TTFT 下降, priority: P0, precondition: ...}
```

**为什么写在配置里而不是硬编码进报告模板**：复盘是项目级知识，  
换项目时应该随配置一起替换。报告 §7 会按这个结构自动渲染；  
未配置时报告会提示如何补充，而不是塞一段与当前项目无关的通用经验。

### 2.4 `plan_*.yaml` —— 优化方案库

```yaml
plans:                   # 方案卡片
  - id: spec_mtp
    name: MTP 投机解码
    category: speculative        # quantization|speculative|kv_cache|kernel|scheduling|ondevice
    applicable_when: [memory_bound, decode_latency]   # 触发条件（对应 diagnose 的路线）
    gain: {decode_tps: 1.60, vram_gib: 1.28}
    source: "[M] 50.47 → 80.8~104.5 t/s"
    cost: 显存 +1.28GiB；Draft 必须全层卸载
    risk: 与多槽位并发存在资源竞争
    rollback: 去掉 -md / --spec-type 参数
optimization_baseline:   # 「优化前」的参照配置
  id: B0_baseline
  startup_flags: [-ngl 99, --parallel 1, --ctx-size 8192, "--flash-attn on"]
  measured: {decode_tps: 50.47}
bundles:                 # 候选组合（optimize 会逐一出对比表）
  - id: B2_concurrent64k
    plans: [quant_q4km_ud, kernel_flash_attn, kv_mixed_q8q4, sched_parallel2]
    startup_flags: [-ngl 99, --parallel 2, --ctx-size 65536,
                    --cache-type-k q8_0, --cache-type-v q4_0, "--flash-attn on"]
    measured: {decode_tps: 47.94, decode_tps_system: 64.74, peak_vram_mib: 18625}
selected_bundle: B2_concurrent64k
```

**关键约定**：

- `applicable_when` 里的标签来自 `diagnose` 输出的 `optimization_route`  
  与 findings 的 `type`/`subtype`。加新标签时同步改 `diagnose.py`。
- `measured` 与 `expected` 分列：`expected` 是几何模型估算，  
  `measured` 是实测。`verify` 优先用 `measured`。
- `startup_flags` 支持 `{draft}` / `{mmproj}` / `{model}` 占位符，  
  由 `model.artifacts` 的路径自动替换。
- `optimization_baseline` 定义了「优化前」是什么 ——  
  它是**可用的朴素配置**，不是「优化后配置的子集」。  
  这样每项优化才有独立的可归因收益。

---

## 3. 多硬件后端（`perfkit/backends.py`）

同一套七步流程要跑在 GPU / CPU / NPU / SoC 上，差异不能散落在各处 `if` 里。  
框架把这些差异收敛成**后端对象**，由 `device_*.yaml` 的 `kind` 决定用哪个：

| kind          | 实现类                   | 内存称呼     | 加速标志      | 带宽探针                       | 回退代价     |
| :------------ | :-------------------- | :------- | :-------- | :------------------------- | :------- |
| `gpu`         | `NvidiaGpuBackend`    | 显存       | `-ngl 99` | `cuda_bw_probe`（纯读带宽）      | 昂贵（PCIe） |
| `cpu`         | `CpuBackend`          | 内存       | `-ngl 0`  | `ggml_probe`（mmap 真实文件）    | 廉价       |
| `hybrid`      | `AppleUnifiedBackend` | 统一内存     | `-ngl 99` | `ggml_probe --mode stream` | 廉价       |
| `soc`         | `SocBackend`          | App 内存配额 | `-ngl 0`  | `ggml_probe`               | 廉价       |
| `npu` / `dsp` | `NpuBackend`          | 加速器内存    | —（无标准接口）  | 无                          | —        |

后端负责回答的问题：**在这类硬件上，什么测量手段有效、什么结论成立、  
什么话能说**。例如：

- CPU 后端显式返回 `-ngl 0`。这不是「默认值」，而是**必须钉死**：  
  本机装了 Metal（Intel Iris），不写 `-ngl 0` 会静默走 GPU，  
  与「验证 CPU 可用性」的目标直接相悖。
- CPU/统一内存后端的 `fallback_is_cheap()` 返回 `True`，  
  报告 §3.8 的表述随之从「PCIe 往返代价」改为「同步开销」，  
  「零回退」也不再被列为硬约束 —— 因为 CPU 后端根本不存在回退。
- `npu` 后端没有标准内存/利用率接口，`mem_snapshot()` 返回 `None`，  
  报告显示「—」而不是编造一个数字。

**新增一类硬件**：继承 `Backend`，实现需要的钩子（都有保守默认值），  
在 `BACKENDS` 注册，然后改 `device_*.yaml` 的 `kind`。流程代码零改动。

查看当前支持与当前配置用的是哪个：

```bash
python3 -m perfkit query --backends
```

---

## 4. 跨运行查询（`perfkit query`）

闭环跑完后产物是一堆 JSON，`query` 提供固定查询指令，不必记目录结构：

```bash
python3 -m perfkit query --list                    # 所有运行及其摘要
python3 -m perfkit query --run X --kpi decode_tps  # 某指标的前后值 + 口径
python3 -m perfkit query --run X --kpis            # 列出该运行全部 KPI
python3 -m perfkit query --runs a,b,c              # 横向对比（带口径标记）
python3 -m perfkit query --backends                # 硬件后端清单
python3 -m perfkit query --run X --facts           # 该运行的配置事实 + 产物清单
```

设计约束：

- **只读**：不修改任何产物。
- **不猜**：查不到就明说「无数据」并指出缺哪一步产物，  
  不回退到估算值 —— 那会让人误以为拿到的是实测。
- **带口径**：每个数值都连同来源标记一起返回  
  （`M`=实测 / `m`=几何模型 / `d`=实测推导 / `c`=配置声明 / `?`=待采集）。

横向对比示例（两次运行、两种后端）：

```bash
python3 -m perfkit query --runs qwen2.5-1.5b-cpu,qwen3.8-27b-decode-opt
```

---

## 5. 硬件带宽探针（`perfkit probe`）

探针回答一个框架其它部分都依赖的问题：**这台机器的内存/显存带宽实测是多少？**

```bash
python3 -m perfkit probe --config configs/qwen2.5-1.5b-cpu --size-gib 2
```

结果落盘到 `output/_env/probe_<kind>.json`，`baseline` 会自动读取并写入报告
§0.2 的「环境探针实测」表。之所以用文件解耦而不是让 baseline 每次重跑探针：
**探针测的是硬件，与模型无关**，一次测量可被多次运行复用。

**按后端分开存**（`probe_cpu.json` / `probe_gpu.json`）：CPU 探针测主机内存路径、
GPU 探针测显存路径，两者不可互换。读取时还会校验文件里的 `backend` 字段，
来源不匹配的一律不采信 —— 否则会得出「推理反推带宽是探针值的 49 倍」
这类荒谬结论。

### 探针指标与适用硬件

| 后端 | 探针 | 指标 | 测什么 |
| :--- | :--- | :--- | :--- |
| `cpu` / `soc` | `ggml_probe` | `stream-file` | mmap 顺序读**真实权重文件**（硬件上限，反推 Decode 上界用它） |
| | | `stream` | 匿名缓冲区顺序读（对照组；差值反映 mmap 路径开销） |
| | | `gemv-scalar` | 标量 GEMV（**下界参照**，跨架构可比） |
| | | `gemv-simd` | SIMD GEMV（**可达参照**，判断生产内核空间用它） |
| `gpu` | `cuda_bw_probe` | `read-bw` | 显存**纯读**带宽（推荐口径；用拷贝带宽会低估约 2 倍） |
| | | `copy-bw` | 显存拷贝带宽（对照项） |
| | | `gemv` | q4_0 GEMV 达成带宽 |
| `npu` | — | — | 无标准接口，返回「未采集」而不是编造数字 |

### 三层参照：三个数回答不同的问题

探针给出三个量级不同的数字。它们**不能混用** —— 混用会得出方向相反的结论：

| 参照 | 是什么 | 用来回答 |
| :--- | :--- | :--- |
| **顺序读** | 硬件天花板 | 「这台机器的访存上限是多少」→ **反推 Decode 上界** |
| **SIMD GEMV** | 正常优化的内核能拿到多少 | 「生产内核还有多少空间」→ **判断该不该继续优化内核** |
| **标量 GEMV** | 完全不优化能拿到多少 | 「朴素实现差多远」→ **跨架构横向对比**（任何 CPU 都能跑同一份标量代码） |

**为什么必须同时给标量和 SIMD**：只给标量会把优化空间高估到荒谬的程度。
实测这台机器标量只有顺序读的 6%，若拿它当基准会得出「还有 94% 空间」；
而 SIMD 参考实现已达 25%，生产内核（llama.cpp 的 mmvq）更高。
**判断「该不该继续优化内核」必须看 SIMD 效率，不是标量效率。**

**为什么反推 Decode 上界必须用顺序读**：顺序读是硬件天花板，GEMV 只是它的下界。
用 GEMV 反推会把「内核效率」误当成硬件能力 —— 标量 6% 会得出
「上界只有 0.06 t/s」这种荒谬结论。

探针由 `device_*.yaml` 的 `kind` 自动选择。找不到对应探针时会给出构建提示
（CPU 用 `bash scripts/build.sh`，GPU 用 `bash scripts/build.sh --cuda`）。

### 怎么读探针结果

**先看「交叉校验」那一行** —— 它把探针值与「真实推理反推的带宽」
（`decode_tps × 权重体积`）对照。这是探针数据最有价值的一步：

| 比值 | 含义 | 处理 |
| :--- | :--- | :--- |
| ≈1（差 <5%） | 推理已接近该路径的带宽上限 | 探针可信 |
| >1.05 | 推理反推值更高 | 探针可能在负载下采集，或生产内核优于标量实现；**以推理反推值为准** |
| >3 | 数据不匹配 | 检查探针是否在本设备采集、权重体积是否取自同一模型 |

**内核效率**（SIMD GEMV 达成带宽 ÷ 顺序读上界）用于判断优化方向：

- 接近 1 → 已访存受限，加 SIMD 也救不回来，只能减少搬运字节
- 明显 < 1 → 访存路径还有空间（预取 / 分块 / 布局 / 多行并行）

> 探针的 SIMD 版仍是**参考实现**（未做多行并行、未用整数点积），
> 因此它给的是「可达」的**下界**：生产内核通常还能再高一些。

---

## 6. 常见任务的操作路径

### 6.1 换一个模型

```bash
cp configs/model_qwen3.8-27b.yaml configs/example-llama3-8b/model_qwen3.8-27b.yaml
cp configs/device_rtx4090.yaml    configs/example-llama3-8b/device_rtx4090.yaml
cp configs/task_decode_optimize.yaml configs/example-llama3-8b/task_decode_optimize.yaml
cp configs/plan_q4km_mtp_kvq.yaml  configs/example-llama3-8b/plan_q4km_mtp_kvq.yaml
cp configs/qwen2.5-1.5b-cpu/ledger.json configs/example-llama3-8b/ledger.json

# 改 model_*.yaml 的 architecture / artifacts / precision_ladder
# 改 ledger.json 的 meta 与 scenarios

# 核验架构超参（强烈建议）
python3 -m perfkit define --config configs/example-llama3-8b --run llama3-8b \
    --from-gguf /path/llama3-8b-q4_k_m.gguf

python3 -m perfkit all --config configs/example-llama3-8b --run llama3-8b
```

> **配置文件名在每套目录内是固定的**（`model_qwen3.8-27b.yaml` 等），  
> 多套配置靠**目录隔离**：`configs/<project>/`。这样 `--config` 一个参数  
> 就能切换整套环境，且各套的台账互不干扰。

### 6.2 换任务目标为「内存占用瓶颈分析」

```bash
cp -r configs configs/example-memory-analysis
# 改 task.scope: memory，调整 KPI（去掉吞吐类，加内存类）
python3 -m perfkit all --config configs/example-memory-analysis --run mem-analysis
```

`scope` 是 `diagnose` 的分支开关：设为 `memory` 后，  
`capacity_bound` 类规则优先命中，`optimize` 优先筛出显存/KV 相关方案。

### 6.3 真机采集

加速器标志由后端自动注入（GPU 为 `-ngl 99`，CPU 为 `-ngl 0`），  
不需要在命令里手写：

```bash
# GPU 场景
python3 -m perfkit baseline  --run my-run --mode live \
    --bench-bin /path/to/llama-bench

# CPU 场景（同上，后端会自动加 -ngl 0）
python3 -m perfkit baseline --config configs/qwen2.5-1.5b-cpu \
    --run cpu-run --mode live --bench-bin /path/to/llama-bench

# 先做冒烟验证：只跑短场景，避免一上来就等 10 分钟持续负载
python3 -m perfkit baseline --run my-run --mode live --only short_chat

# 采集完导出成台账，供后续 replay 复用
python3 -m perfkit baseline --run my-run --mode live \
    --write-ledger configs/my-project/ledger.json

# kernel 级时间线（仅 GPU，可选）
nsys profile --trace=cuda,nvtx -o output/my-run/raw/nsys ./llama-server ...
```


**CPU 后端采集的注意事项**（与 GPU 的本质差异）：

| 项     | CPU                      | 独显 GPU          |
| :---- | :----------------------- | :-------------- |
| 带宽归属  | 与主机其他进程**共享**内存带宽        | 独立显存带宽，不受主机负载影响 |
| 负载敏感度 | 负载 90+ 时吞吐掉约 50%         | 基本无感            |
| 采集前提  | `uptime` 的 1 分钟负载 < 物理核数 | 无特殊要求           |
| 线程数   | 存在最优值（通常是物理核数）           | 不适用             |

框架会把采集时的系统负载写入 `baseline.json` 的 `environment`，  
报告 §0 的「测量有效性条件」表据此给出可否直接引用的判断。

### 6.4 加一个优化方案

在 `plan_*.yaml` 的 `plans` 里加一条卡片即可，**无需改代码**：

```yaml
  - id: kv_prefix_cache
    name: KV Cache 前缀复用
    category: kv_cache
    applicable_when: [long_context, multi_turn]
    gain: {ttft_ms: -0.60}
    source: "[L] 多轮对话中重复前缀可完全跳过 Prefill"
    cost: 需客户端保持稳定 session
    risk: 缓存命中率依赖业务形态
    rollback: 关闭前缀缓存开关
```

若新增了 `applicable_when` 里未出现过的标签，需要在  
`diagnose.py` 的某条规则里产出该标签（见 §5.2）。

---

## 7. 数字可信度机制

框架用三个机制保证报告里的数字可追溯。

### 7.1 台账（Ledger）

所有 **[M]** 实测值集中在台账 JSON 里，每条都带 `source`（文档章节）。  
这是唯一的实测数据来源 —— **代码里不硬编码任何实测数字**。

**解析顺序**（`PerfConfig.ledger()`）：

1. `<config_dir>/ledger.json` —— 该套配置自带的台账（推荐）
2. `<root>/data/reference_measurements.json` —— 全局默认台账

推荐做法是把台账放在配置目录内，这样一套「模型 + 设备 + 任务 + 台账」  
可以整体拷贝到另一个项目，不会互相污染。

**台账结构**：

```json
{
  "meta":        { "model": "...", "engine": "...", "device": "...", "collected": "..." },
  "theoretical": { "vram_bandwidth_gbps": 1008, "weight_gib": 15.33 },
  "baseline_native": { "decode_tps": 50.47, "bandwidth_utilization": 0.822 },

  "scenarios": {
    "<scenario_id>": {
      "prompt_tokens": 90, "gen_tokens": 256, "parallel": 1,
      "prefill_tps": 85.49, "decode_tps": 47.94, "decode_tps_system": null,
      "ttft_ms": 1050.0, "source": "报告 §4.1"
    },
    "<另一个 id>": { "status": "todo", "note": "尚未采集" }
  },

  "curves": {
    "long_context": [ { "ctx": 4096, "prefill_tps": 105.3,
                        "decode_tps": 46.8, "kv_gib": 0.15,
                        "kv_gib_source": "measured",   // measured | derived
                        "command": "llama-bench -m $MODEL ... -d 4096 -r 3",
                        "command_source": "recorded" } ]
  },

  "memory": { "breakdown_gib": {...}, "peak_used_mib": 18625,
              "vram_total_mib": 24564, "safety_margin_mib": 5939 },

  // ★ 报告 §2.2 / §2.3 的「测试方法」表直接渲染这一段。
  //   没有它，读者看到 17.85 t/s 不知道含不含采样、是均值还是峰值。
  "measurement_protocol": {
    "tool": "llama-bench",              // 或 "llama-server + HTTP 客户端"
    "tool_version": "llama.cpp 07d937828（CPU 后端）",
    "statistic": "每个配置重复 3 次，取 avg_ts（算术均值）",
    "measurement_point": "进程内计时：不含 HTTP 传输、排队与采样开销",
    "backend_label": "CPU 后端",
    "accel_flags": ["-ngl", "0"],
    "threads": 4,
    "flags": [
      { "flag": "-p N", "meaning": "Prefill 口径：纯 prompt 吞吐，不含采样" },
      { "flag": "-n N", "meaning": "Decode 口径：稳态速率，排除首 token" },
      { "flag": "-d N", "meaning": "上下文深度：先填 KV 到 N 再测" }
    ],
    "conventions": [
      "llama-bench 的 Decode 是纯解码吞吐，高于端到端墙钟速率",
      "TTFT 列为推导值（prompt_tokens ÷ prefill_tps），非端到端实测"
    ]
  },

  "extra":  { "spec_mtp": {...}, "tunnel": {...}, "mode_switch": {...} }
}
```

**每个场景还应记 `command`**（原始复现命令）与 `command_source`：

```json
"scenarios": {
  "short_chat": {
    "prompt_tokens": 64, "gen_tokens": 128, "parallel": 1,
    "prefill_tps": 69.13, "decode_tps": 17.54, "ttft_ms": 930.0,
    "cache_type_k": "f16", "cache_type_v": "f16",
    "command": "llama-bench -m $MODEL -ngl 0 -o json -t 4 -p 64 -n 128 -r 5",
    "command_source": "recorded",
    "source": "/tmp/qwen15b_bench/06_short.json",
    "confidence": "measured"
  }
}
```

- `command` 缺失时框架会**按场景参数重建**一条等价命令，  
  并在报告里显式标注 `reconstructed` —— 重建命令 ≠ 原始命令，  
  正式报告建议如实记录。
- `cache_type_k` / `cache_type_v` 用于重建时还原 KV 量化档位，  
  否则「同深度对照」的两条命令会长得一模一样，读者看不出唯一变量在哪。
- 命令里可用 `$MODEL` 占位（权重路径很长），报告会在命令块首补 `MODEL=...`。

**关键约定**：

- `scenarios` **按 `task_*.yaml` 的 scenario id 组织**。换任务时只改两边配置，  
  `baseline` 代码无需改动。台账里没有的场景会被标记为 `todo`，  
  下游一律按「无数据」处理，**不会编造数字**。
- `curves.long_context` 用于 KV 随上下文增长的插值  
  （`verify` 计算带宽利用率时按 ctx 查表）。
- `extra` 是自由区，原样透传给报告与图表，用于放各类专项数据。
- `baseline_native` 是**纯原生基线**（无任何优化）的实测值，  
  用于计算「理论上界达成率」，与 `optimization_baseline` 是两回事。

真机采集时用 `--mode live`，框架会调用 `llama-bench` 并把原始输出  
落到 `output/<run>/raw/`，同时把结果写回统一结构（不自动改写台账 ——  
写台账是人工确认的动作，避免脏数据污染参考值）。

### 7.2 对账（Reconcile）

模型推算与实测不一致时，`define` 会输出 `reconcile` 段，  
报告 §1.4 并列展示两侧数字：

```text
几何模型值  6.50 GiB
台账实测值  2.20 GiB
差额        4.30 GiB（×2.95）
可能原因    ...
核验方式    perfkit define --from-gguf ...
影响        ...
```

**不做静默校准**是刻意设计：如果偷偷把模型改成实测值，  
差额就永远查不出根因。差额消除后重新运行，框架自动收敛。

### 7.3 参考源矛盾（Source Conflicts）

同一份文档内部矛盾时（例如部署报告 §3.1 说 Prefill 2900 t/s、  
§4.4 说 85.5 t/s，差 34×），配置里显式列出：

```yaml
source_conflicts:
  - id: prefill_tps
    claim_a: {value: 2900, source: "§3.1"}
    claim_b: {value: 85.5, source: "§4.4"}
    ratio: 33.9
    likely_cause: "..."
    adopted: claim_b
    action: "用 live 模式重测"
```

报告 §1.5 并列展示，并说明为何采信其一。

---

## 8. 二次开发

### 8.1 加一个子命令

```python
# perfkit/commands/mycmd.py
from ..config import PerfConfig, load_json, save_json

def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    data = load_json(cfg.out("profiling.json"))
    # ... 你的处理 ...
    save_json(result, cfg.out("mycmd.json"))
    return 0

def add_parser(sub) -> None:
    p = sub.add_parser("mycmd", help="我的新环节")
    p.add_argument("--threshold", type=float, default=0.8)
    p.set_defaults(func=main)
    return p          # 必须返回，__main__ 会往它上面挂全局 --config/--run
```

然后在 `perfkit/commands/__init__.py` 的 `MODULES` 里加一项，  
并在 `all.py` 的 `STEPS` 里决定是否纳入一键流程。

> 若新子命令需要读取子命令私有选项，记得同时在 `all.py` 的  
> `STEP_DEFAULTS` 里给出默认值 —— 一键串联时子命令的 argparse  
> 定义不会生效。

### 8.2 加一条瓶颈判定规则

```python
# perfkit/commands/diagnose.py
def _rule_my_bottleneck(p: dict, d: dict) -> dict | None:
    if 某个条件:
        return {
            "type": "my_bottleneck",
            "severity": "primary",          # primary|secondary|phase_scope|watch
            "confidence": "high",
            "evidence": ["具体数字1", "具体数字2"],   # 必须带数字
            "reasoning": "为什么这构成瓶颈",
            "counter_evidence": "什么证据会推翻它",   # 必填，防确认偏误
        }
    return None

RULES = [
    ("my_bottleneck", _rule_my_bottleneck),   # 注意顺序 = 优先级
    ...
]
```

**`evidence` 必须带具体数字**，`counter_evidence` 必须填 ——  
没有反证的结论等于没有结论。

### 8.3 加一类硬件后端

在 `perfkit/backends.py` 里继承 `Backend`，实现需要的钩子  
（基类都有保守默认值，只覆盖必须改的），然后注册：

```python
class MyAccelBackend(Backend):
    kind = "myaccel"
    label = "自研加速器"
    memory_label = "加速器内存"
    utilization_label = "NPU 利用率"

    def accel_flags(self) -> list[str]:
        return ["-ngl", "0", "--device", "myaccel"]

    def mem_snapshot(self) -> MemoryReading | None:
        # 没有标准接口时返回 None，报告会显示「—」而不是编造数字
        return None

    def bandwidth_probe_command(self, probe_bin, size_gib=2.0):
        return [probe_bin, "--mode", "stream"]

    def fallback_is_cheap(self) -> bool:
        return True          # 决定报告 §3.8 的表述与「零回退」是否列为硬约束

BACKENDS["myaccel"] = MyAccelBackend
```

再把 `device_*.yaml` 的 `kind` 设成 `myaccel` 即可，**流程代码零改动**。  
用 `perfkit query --backends` 验证注册结果。

> 基类的每个钩子都有保守默认值（返回 `None` / 空列表），  
> 宁可让上层标注「待采集」，也不要返回一个看起来合理的编造值 ——  
> 后者会污染整条证据链。

### 8.4 加一张图表

在 `visualize.py` 里加一个 `chart_xxx(...)` 函数返回 SVG 字符串，  
然后加进 `build_html` 的 `charts` 列表。配色统一用模块顶部的 `C` 字典  
（语义化命名，同一含义在所有图里用同一颜色）。

---

## 9. 自检与验证

框架自带一组**不依赖 GPU** 的自检，用手算可验证的值校验核心公式：

```bash
python3 scripts/selftest.py      # 或 make selftest
```

覆盖 22 组检查（185 项断言）：

| 组  | 校验内容                                                                        |
| :- | :-------------------------------------------------------------------------- |
| 1  | 量化位宽映射（q8_0=8.5、q4_0=4.5、q4_K_M=4.8125 bit，按 ggml 块布局手算）                    |
| 2  | KV Cache 几何公式（8K/f16 = 正好 2 GiB = 2³¹ 字节；ctx 翻倍 KV 翻倍）                      |
| 3  | Decode 理论上界（1008 GB/s ÷ 16.462 GB = 61.23 t/s）                              |
| 4  | 嵌入层与逐层权重拆分（未 tie 时两份表；tie 时减半）                                              |
| 5  | Roofline 拐点（165.2 TFLOPS ÷ 1008 GB/s = 163.9）与判定                            |
| 6  | 显存预算分解与冗余                                                                   |
| 7  | 启动参数展平（含 `"--flash-attn on"` 这类带空格的多词 flag）                                 |
| 8  | 两个参考配置的端到端闭环可跑通 + 产物齐全                                                      |
| 9  | **硬件后端抽象**：5 类后端注册、CPU 钉死 `-ngl 0`、GPU 用 `-ngl 99`、回退代价判定、内存字段解析、未知 kind 回落 |
| 10 | **缺峰值算力时的 roofline 退化**（CPU 画像常见；退化为 `unknown_compute_peak` 而非崩溃）           |
| 11 | **嵌入层按实际量化档位折算**（小模型大词表场景，F16 会高估 3 倍）                                      |
| 12 | 配置校验能拦住错误输入（hidden_size 不一致、GQA 分组比不整除）                                     |
| 13 | **测量协议与复现命令**：无原始命令时重建并标注 `reconstructed`、KV 档位/深度还原、协议表渲染、两套台账都必须声明协议      |
| 14 | **硬件剖面的算力侧分析**：含 roofline 恒等关系（算力达成率 ≡ 带宽利用率）、未被解释时间的推导、缺峰值算力时不崩            |
| 15 | **探针的后端适配**：CPU/GPU 各自选对探针二进制与解析字段，缺顺序读时拒绝反推 Decode 上界                            |
| 16 | **探针的 GEMV 双实现**：标量下界与 SIMD 可达参照，校验 checksum 等价、nibble 布局正确                            |
| 17 | 报告覆盖业务地图 0/1 步的输入输出（3.2 阶段明细表）                                               |
| 18 | 报告覆盖业务地图 2-6 步的输入输出 + 每步都有「输入产出对照表」                                          |
| 19 | **配置目录按角色前缀解析**（换硬件不改代码；同角色多份配置必须报错）                                        |
| 20 | **第 3 步四件产出物**：算子占比合计 100%、各算子耗时合计 = 单步耗时、Hot Kernel/执行路径/根因树/反证条件齐备       |
| 21 | **混合架构口径**：KV 只按 `n_attn_layer` 计（错用 n_layer 会高估 4×）、SSM 状态不随 ctx 增长、权重优先取实测分解、注意力 FLOPs 用 `n_head` 而非 `n_head_kv` |
| 22 | **启动脚本必须带加速器卸载标志**：所有 bundle（尤其选定 bundle）都要含 `-ngl 99` / `-ngl 0` —— 漏写不会报错，只会静默跑 CPU |

**为什么需要这个**：性能框架最危险的失败模式是「算错了但看起来合理」。  
例如 KV 公式少乘一个 2，结果只是数字偏小一半，报告照样生成，  
但结论已经错了。因此每个公式都有独立的手算对照。

### 9.1 报告一致性审计（`scripts/audit_reports.py`）

自检校验的是**公式**；审计校验的是**口径**。同一个指标会在五处出现：

```text
台账 → baseline.json → optimize.json → verify.json → report.md → charts.html
```

任何一处口径不同（跨会话、扁平值 vs 场景值、不同 bundle），
都会让报告自相矛盾，而这种矛盾**读起来像结论、查起来要半小时**。

```bash
python3 scripts/audit_reports.py            # 全部运行
python3 scripts/audit_reports.py --run <id> # 单个
make audit                                  # 同上
```

| 检查 | 内容 | 本项目实际抓到的问题 |
| :--- | :--- | :--- |
| C1 | 图表「优化基线」柱 == 报告 §5.1 的 before（差 <2%） | 图表取台账 `baseline_native`（上一会话 92.65）、候选取本轮 measured（91.56），得出「优化后变慢」的假象；同配置跨会话实测差 **10.7%** |
| C2 | 报告 §2.2 场景表 == 台账 `scenarios[*].decode_tps` | — |
| C3 | verify 的 before == 台账对应场景值 | CPU 的 `prefill_tps` 绑 `long_ctx_8k_kvq8`，却取到 `long_ctx_8k` 的扁平值 34.65（正确 11.48，**差 3×**） |
| C4 | 无 `None` / `nan` / 未知占位符（反引号内与合法占位符除外） | — |
| C5 | 投机配置必须标注「口径失效」 | 投机解码下「t/s × 权重字节」算出的带宽 >100%（本项目 153%），不能读作突破物理带宽 |
| C6 | 结论摘要的章节序号连续 | newarch 曾出现两个「五、」 |
| C7 | 报告 §6.1 验收表 == verify.json 的逐项判定 | — |

**三条硬规则**（任一违反即 FAIL）：

1. **同会话**：基线与候选必须在同一会话内测。跨会话漂移可达 10~21%，
   大于多数配置差异。
2. **场景绑定**：KPI 绑了场景就取 `by_scenario[该场景]`；
   扁平值可能是别的场景的拷贝。
3. **口径失效要标注**：投机解码让「每步读一遍权重」不再成立，
   带宽类指标必须显式标 `invalid`，而不是算一个 >100% 的数。

---

## 10. 故障排查

| 现象                                          | 原因              | 处理                                                                                    |
| :------------------------------------------ | :-------------- | :------------------------------------------------------------------------------------ |
| `配置错误: 意外的缩进`                               | YAML 缩进不一致      | 用 `python3 -c "from perfkit.config import load_yaml; load_yaml('configs/x.yaml')"` 定位 |
| `数据文件不存在: .../baseline.json`                | 上游子命令未跑         | 先跑 `perfkit define` / `baseline`                                                      |
| `找不到 llama-bench`                           | live 模式缺可执行文件   | 用 `--mode replay` 或 `--bench-bin` 指定路径                                                |
| `配置校验失败: n_head×head_dim 与 hidden_size 不一致` | 架构超参填错          | 用 `--from-gguf` 核验                                                                    |
| 报告里出现大量 `—`                                 | 对应 KPI 无实测数据    | 属正常：标 `todo` 的项不参与达标判定                                                                |
| `verify` 结论为 REVIEW                         | must_pass 有未达标项 | 看报告 §6 的「未达标项归因」，可能是 KPI 目标本身有问题                                                      |

---

## 11. 端侧场景的额外注意点

框架默认按 GPU 独立显存建模。移植到端侧设备时需注意：

1. **统一内存**：`unified_memory: true` 时，Host↔Device 搬运无代价，  
   CPU 回退的惩罚远小于独立显存场景 —— `cpu_fallback_cost` 的计算需相应调整。
2. **功耗与热**：`power_thermal.thermal_throttle: true` 时，  
   必须补做持续负载稳态测试（框架的 `soak_30min` 场景），  
   并引入 `tokens/Joule` 能效指标。
3. **NPU 算子覆盖**：`kind: npu` 时，先用厂商工具链做算子支持矩阵扫描，  
   再决定图分区边界。多数移动 NPU 只吃 INT8/INT16 定点，  
   对 Q4_K_M 权重无原生支持，需离线重编译。
4. **权重 streaming**：模型远超设备内存时（如 27B 跑端侧设备），  
   逐层 streaming 的速度受存储 IO 限制，理论值 ≈ 存储带宽 ÷ 权重体积，  
   往往不可接受 —— 此时应改用更激进的量化 + 模型裁剪，或走端云协同。
5. **App 内存配额**：iOS Jetsam / Android LMK 会强杀超配额进程，  
   `min_safety_margin_gib` 需按平台配额设定，通常远小于服务器场景。
