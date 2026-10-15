# Qwen3.8-27B 算子节点结构解析报告

**解析依据**：`quant_Q4_K_M_imatrix.log`（llama.cpp `llama-quantize`，Q4_K_M + imatrix）
**权重格式**：GGUF V3（输入 BF16 → 输出 Q4_K_M）
**日志完整性**：完整（1–866 号张量全部覆盖，无截断）

---

## 一、日志可信度与关键元信息

| 项 | 值 | 说明 |
|---|---|---|
| 引擎 | llama.cpp `0.4.1-dev` (build 1, commit `60081bb`), GNU 11.4.0, x86_64 | 定制分支，arch 为 `qwen35` |
| 硬件 | Tesla V100S-PCIE-32GB, CC 7.0, VRAM 32494 MiB | CC 7.0 = **无 BF16/FP8 Tensor Core**，无 FP8 |
| 输入模型 | `Qwen3.8-27B-BF16.gguf`，43 个 KV，**866 张量** | `general.file_type = 32` → BF16 |
| 张量类型分布 | **f32: 360 / bf16: 506** | 506 个 bf16 全部被量化 |
| 量化类型分布 | **q4_K: 439 / q6_K: 67**（f32 保持 360） | 合计 506 ✔ |
| 体积 | 52115.19 MiB (16.00 BPW) → 16021.46 MiB (4.92 BPW) | 压缩比 **3.25×** |
| 量化耗时 | 859029.33 ms（≈14.3 min），10 线程 | — |
| imatrix | **496 条 entry**，7 chunks，PPL = 24.4637 ± 3.02382 | `imatrix-UD-Qwen3.8-27B.gguf` |
| 校准集 | `calibration/self_generated.txt` | 自生成（self_gen） |
| 上下文 | 262144 | RoPE freq_base = 1e7 |

> ⚠️ 与 `quant_config.yaml` 的 `tensor_overrides`（blk.0/1/62/63→Q6_K、attn_q/k/ffn_gate/up→Q5_K_M）**不一致**：本日志是**标准 Q4_K_M recipe**，未走 UD 覆盖路径。对比文件应为 `quant_UD_Q4_K_M.log`。做方案选型时需注意这一点。

---

## 二、模型网络结构概述

### 2.1 全局超参

| 参数 | 值 | 推断 |
|---|---|---|
| `block_count` | **65** | blk.0–63 主层(64) + blk.64 MTP 块(1) |
| `embedding_length` | 5120 | hidden |
| `feed_forward_length` | 17408 | FFN 中间维 |
| `head_count / head_count_kv` | 24 / 4 | GQA = 6:1 |
| `key_length / value_length` | 256 / 256 | head_dim = 256 |
| `rope.dimension_count` | 64 | **部分旋转**：256 维中仅 64 维加 RoPE |
| `rope.dimension_sections` | [11,11,10,0] | **MRoPE**（多模态 4 段式），与 `tags=["image-text-to-text"]` 吻合 |
| `full_attention_interval` | 4 | 每 4 层 1 层全注意力 |
| `attention.recurrent_layers` | `arr[bool,65]` | 层型掩码 |
| `nextn_predict_layers` | 1 | MTP / 投机解码草稿头 |
| `ssm.conv_kernel / state_size / group_count / time_step_rank / inner_size` | 4 / 128 / 16 / 48 / 6144 | 线性注意力分支参数 |
| vocab / eos | 248320 / 248046 | gpt2 BPE，pre=`qwen35`，`add_bos=false` |
| norm eps | 1e-6 | RMSNorm |

### 2.2 拓扑：混合注意力（Hybrid Attention）

```
Input Tokens
   │
   ▼
[token_embd.weight  Gather]  [5120 × 248320]
   │
   ├─ blk.0  ── 循环层 (Gated DeltaNet 线性注意力 + SSM)
   ├─ blk.1  ── 循环层
   ├─ blk.2  ── 循环层
   ├─ blk.3  ── 全注意力层 (Full Attention, GQA)   ◄── 每 4 层
   ├─ ...      (blk.0–63 交替，共 64 层)
   ├─ blk.63 ── 全注意力层
   │
   ├─ blk.64 ── MTP / nextn 块 (全注意力 + eh_proj + 共享头)
   │
   ▼
[output_norm.weight  RMSNorm]
   │
   ▼
[output.weight  LM Head GEMM]  [5120 × 248320]
   │
   ▼
Sampler (temp 1.0 / top_p 0.95 / top_k 20)
```

**层型统计（与日志逐块核对一致）**

| 层型 | 层号 | 数量 | 判定依据 |
|---|---|---|---|
| **循环层**（线性注意力/SSM） | 0,1,2,4,5,6,8,…,62 | **48** | 含 `attn_qkv` + `ssm_*` |
| **全注意力层** | 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 | **16** | 含 `attn_q`/`attn_k`/`attn_v` |
| **MTP 块** | 64 | **1** | 含 `nextn.*`，全注意力结构 |

### 2.3 三种层的张量构成

**① 循环层（48 个，以 blk.0 为例）**

| 张量 | 形状 | 作用 |
|---|---|---|
| `attn_norm.weight` | [5120] f32 | 前置 RMSNorm |
| `attn_qkv.weight` | [5120, 10240] | 融合 QKV 投影（Q=2048, K=2048, V=6144） |
| `attn_gate.weight` | [5120, 6144] | 线性注意力输出门 |
| `ssm_conv1d.weight` | [4, 10240] f32 | 深度可分离因果卷积（k=4） |
| `ssm_a` | [48] f32 | 每头衰减系数 A_log |
| `ssm_dt.bias` | [48] f32 | 时间步偏置 |
| `ssm_alpha.weight` | [5120, 48] | 输入相关 α（衰减门）投影 |
| `ssm_beta.weight` | [5120, 48] | 输入相关 β（写入门）投影 |
| `ssm_norm.weight` | [128] f32 | 每头 gated RMSNorm（head_v_dim=128） |
| `ssm_out.weight` | [6144, 5120] | SSM 分支输出投影 |
| `post_attention_norm.weight` | [5120] f32 | 残差前 RMSNorm |
| `ffn_gate / ffn_up / ffn_down` | [5120,17408]×2 / [17408,5120] | SwiGLU FFN |

**② 全注意力层（16 个，以 blk.3 为例）**

| 张量 | 形状 | 作用 |
|---|---|---|
| `attn_q.weight` | [5120, 12288] | Q 投影（6144）+ **Q-Gate**（6144）融合 |
| `attn_k / attn_v` | [5120, 1024] | K/V 投影（4 头 × 256） |
| `attn_q_norm / attn_k_norm` | [256] f32 | QK-Norm（per-head RMSNorm） |
| `attn_output.weight` | [6144, 5120] | 输出投影 |
| + 与循环层相同的 norm / FFN | | |

**③ MTP 块（blk.64）**

在全注意力层基础上追加：`nextn.eh_proj.weight [10240,5120]`、`nextn.enorm.weight [5120]`、`nextn.hnorm.weight [5120]`、`nextn.shared_head_norm.weight [5120]`。

---

## 三、模型节点统计

### 3.1 权重块（张量）统计 — 直接来自日志，确定性

**总量：866 张量** = 506 线性权重 + 360 非量化参数

| 张量基名 | 数量 | 量化类型 | 归属 |
|---|---:|---|---|
| `attn_norm.weight` | 65 | f32 | 全部层 |
| `post_attention_norm.weight` | 65 | f32 | 全部层 |
| `ffn_gate.weight` | 65 | q4_K ×65 | 全部层 |
| `ffn_up.weight` | 65 | q4_K ×65 | 全部层 |
| `ffn_down.weight` | 65 | **q6_K ×33**, q4_K ×32 | 全部层 |
| `attn_qkv.weight` | 48 | **q6_K ×24**, q4_K ×24 | 循环层 |
| `attn_gate.weight` | 48 | q4_K ×48 | 循环层 |
| `ssm_alpha / ssm_beta.weight` | 48 / 48 | q4_K ×48 / ×48 | 循环层 |
| `ssm_out.weight` | 48 | q4_K ×48 | 循环层 |
| `ssm_a` | 48 | f32 | 循环层 |
| `ssm_dt.bias` | 48 | f32 | 循环层 |
| `ssm_conv1d.weight` | 48 | f32 | 循环层 |
| `ssm_norm.weight` | 48 | f32 | 循环层 |
| `attn_q / attn_k / attn_output.weight` | 17 / 17 / 17 | q4_K ×17 各 | 全注意力(16) + MTP(1) |
| `attn_v.weight` | 17 | **q6_K ×9**, q4_K ×8 | 全注意力 + MTP |
| `attn_q_norm / attn_k_norm.weight` | 17 / 17 | f32 | 全注意力 + MTP |
| `nextn.eh_proj.weight` | 1 | q4_K | MTP |
| `nextn.enorm / hnorm / shared_head_norm` | 3 | f32 | MTP |
| `token_embd.weight` | 1 | q4_K | 全局 |
| `output.weight` | 1 | **q6_K** | 全局 |
| `output_norm.weight` | 1 | f32 | 全局 |
| **合计** | **866** | | |

**恒等式校验（三重一致，可确认解析无误）**
- `506 量化张量` = 195(FFN) + 240(循环分支) + 68(全注意力 Q/K/V/O) + 1(eh_proj) + 1(token_embd) + 1(output) ✔
- `360 f32 张量` = 216(RMSNorm) + 48(ssm_a) + 48(ssm_dt.bias) + 48(ssm_conv1d) ✔
- `67 q6_K` = 33(ffn_down) + 24(attn_qkv) + 9(attn_v) + 1(output) ✔

### 3.2 算子节点统计 — 按层型推算（引擎相关）

> 说明：日志给出的是**权重块**，算子节点数依赖引擎的融合策略。下表按 **llama.cpp 图节点粒度**（含常见融合）推算，并给出公式以便复用。

| 作用域 | 单块节点数 | 块数 | 小计 |
|---|---:|---:|---:|
| 公共部分（2×RMSNorm + FFN 3×GEMM + SiLU + Mul + 2×Add） | 9 | 65 | **585** |
| 循环层专属（QKV/Gate/α/β/Out 5×GEMM + ssm_norm + conv1d + SiLU + 2×L2Norm + DeltaRule + 2×Mul + Sigmoid + decay-softplus + Split） | 16 | 48 | **768** |
| 全注意力层专属（Q/K/V/O 4×GEMM + 2×RMSNorm + RoPE + SDPA + Sigmoid + Mul + KV-Write + Reshape） | 12 | 16 | **192** |
| MTP 块专属（同全注意力 12 + eh_proj GEMM + 3×RMSNorm + Concat） | 17 | 1 | **17** |
| 全局（Gather + token_embd + output_norm + LM-Head GEMM + Sampler） | 5 | 1 | **5** |
| **合计（统一口径，含 Split/Reshape/Transpose 胶水算子）** | | | **1567** |
| **合计（融合粒度，胶水算子并入相邻 kernel）** | | | **≈ 1471** |

> ⚠️ 本表已按**统一口径（含胶水算子 Split/Reshape/Transpose）**修正为 **1567**，与 §四 逐项加总一致。早期口径曾记为 ≈1471（未计入胶水算子）。口径差异与自校验详见 **§九**。

---

## 四、基础算子节点清单

### 4.1 算子类型 × 出现次数 × 作用

| # | 算子 | 次数 | 作用说明 | 对应权重/依据 |
|---|---|---:|---|---|
| 1 | **MatMul / GEMM(GEMV)** | **506** | 全部线性投影。prefill 为 GEMM（计算受限），decode 为 GEMV（**访存受限**） | 506 个量化张量 |
| 2 | **RMSNorm** | **216** | 归一化。含 attn_norm 65、post_attention_norm 65、ssm_norm(gated,per-head) 48、attn_q_norm 17、attn_k_norm 17、output_norm 1、nextn 3 | 216 个 f32 norm 张量 |
| 3 | **Residual Add** | **130** | 每块 2 次（注意力后、FFN 后） | 拓扑推算 |
| 4 | **SiLU / Swish** | **113** | FFN 激活 65 + 循环层 conv1d 后激活 48 | — |
| 5 | **Elementwise Mul** | **178** | FFN 门控 65 + 线性注意力输出门 48 + Q-Gate(sigmoid 后) 17 + DeltaNet 内 β 门控 48 | — |
| 6 | **Causal Conv1D (depthwise, k=4)** | **48** | 线性注意力分支的短卷积，通道数 10240 | `ssm_conv1d.weight` |
| 7 | **Gated Delta Rule / 分块线性递推扫描** | **48** | 核心 SSM 算子，逐头递推 S ← S·α + β·(k⊗v)，**非 O(n²)** | `ssm_a`, `ssm_dt.bias` |
| 8 | **L2 Normalize** | **96** | 线性注意力内对 Q/K 做 L2 归一（每层 2 次） | DeltaNet 结构约定 |
| 9 | **Softplus / Exp** | **48** | 衰减门 `g = -exp(A_log)·softplus(a + dt_bias)` | `ssm_a`, `ssm_dt.bias` |
| 10 | **Sigmoid** | **65** | Q-Gate 17 + DeltaNet β 门控 48 | — |
| 11 | **RoPE（部分旋转，64/256 维，MRoPE 4 段）** | **17** | 位置编码，仅作用于全注意力层 | `rope.dimension_sections=[11,11,10,0]` |
| 12 | **Scaled Dot-Product Attention (GQA 6:1, causal)** | **17** | Flash-Attention / 分块注意力 | — |
| 13 | **KV Cache Write/Read** | **17** | 仅全注意力层有 KV Cache | `FA_QUANTS=q4_0/q8_0/f16/bf16` |
| 14 | **Embedding Gather** | **1** | token → hidden | `token_embd.weight` |
| 15 | **Concat** | **1** | MTP 块拼接 `norm(embd) ‖ norm(hidden)` | `nextn.enorm/hnorm/eh_proj` |
| 16 | **Sampler（Softmax + Top-K + Top-P + Temp）** | **1** | 采样，非权重算子 | `sampling.top_k=20, top_p=0.95` |
| 17 | **Split / Reshape / Transpose** | **65** | QKV 拆分、多头重排（胶水算子） | — |
| | **合计** | **1567** | 与 §3.2 统一口径一致 ✔ | |

### 4.2 算子族归类（工程视角）

| 族 | 算子 | 节点数 | 优化优先级 |
|---|---|---:|---|
| **线性层（权重）** | MatMul/GEMV | 506 | ★★★ 访存瓶颈（4.92 BPW，16 GB 权重） |
| **归一化** | RMSNorm / L2Norm | 312 | ★★ 融合进 GEMM 前处理 |
| **逐元素** | Add / Mul / SiLU / Sigmoid / Softplus | 472 | ★★ 全部可 kernel-fuse |
| **序列算子** | Conv1D / DeltaRule / SDPA / RoPE | 130 | ★★★ 延迟关键路径 |
| **状态算子** | KV-Cache | 17 | ★★★ 长上下文显存瓶颈 |
| **胶水算子** | Split / Reshape / Transpose | 65 | ★ 融合进相邻 kernel |
| **前后处理** | Gather / Concat / Sampler | 3 | ★ 可忽略 |
| **合计** | | **1567** | |

---

## 五、日志关键信息提取

### 5.1 imatrix 信息

| 项 | 值 |
|---|---|
| entry 数 | **496** |
| 计算 chunks | 7（配置期望 100，实际校准集仅够 7 chunk） |
| chunk size / ctx | 512 |
| PPL | 24.4637 ± 3.02382 |
| 路径 | `gguf/imatrix-UD-Qwen3.8-27B.gguf` |

### 5.2 ⚠️ 10 个张量缺失 imatrix（重要质量风险）

日志中 10 处 `did not find weights for X`，即**未使用 imatrix 加权，回退为标准量化**：

| 张量 | 原因 |
|---|---|
| `token_embd.weight`、`output.weight` | llama.cpp imatrix 采集约定不覆盖 |
| `blk.64.attn_q/k/v/output`、`blk.64.ffn_gate/up/down`、`blk.64.nextn.eh_proj` | **imatrix 计算时 MTP 块未启用** |

`imatrix_ud.log` 佐证：明确列出 `model has unused tensor blk.64.*`（15 条）。**MTP 草稿头（对投机解码加速比影响最大）恰好是量化质量最没保障的部分**，建议后续重跑 imatrix 时开启 nextn 层，或对 blk.64 单独走 Q5_K_M/Q6_K。

### 5.3 Q6_K 提升带（Q4_K_M recipe 的混合精度分布）

```
ffn_down  q6_K: 0-7, 10,13,16,...,55, 56-64   (33个)  ← 首尾 1/8 + 周期中段
attn_qkv  q6_K: 0,1,2,4,5,6, 10,13,...,52, 56,57,58,60,61,62 (24个)
attn_v    q6_K: 3,7,19,31,43,55,59,63,64      (9个)
output    q6_K: 1个
```
特征：**`ffn_down`、`attn_v`、`attn_qkv` 被识别为量化敏感张量提升至 Q6_K**，其余 439 个走 Q4_K。这是标准 Q4_K_M 混合精度策略，与日志中"没有使用 yaml 覆盖规则"互相印证。

### 5.4 量化过程引入的计算逻辑（**不属于模型推理算子**）

| 量化期逻辑 | 说明 | 为何不是推理算子 |
|---|---|---|
| 张量类型分派（recipe） | 按张量名匹配 Q4_K/Q6_K | 编译期/离线决策 |
| imatrix 加权误差最小化 | `Σ w_i·(x_i - x̂_i)²` 逐通道 importance 加权 | 离线校准 |
| 块量化编码 | Q4_K 每 256 元素一块、含 `d`/`dmin` 两级 scale + 6-bit 子量化；Q6_K 每 256 元素 8-bit 子块 | 仅生成权重格式 |
| 反量化 scale 存储 | 每块 scale/min 作为元数据落盘 | 推理期仅做 dequant，非独立算子 |
| `did not find weights` 回退 | 无 imatrix 时的降级路径 | 量化期控制流 |
| GGUF 重写 / 张量重排 | 输出文件 16021 MiB | I/O |

> **推理期唯一由量化引入的算子**：权重 **Dequant（反量化）**，通常已融合进 MatMul kernel（如 `ggml` 的 `dequantize_row_q4_K` 内联），不单独成节点。

---

## 六、性能关注点（端侧/V100S 视角）

| 瓶颈 | 量化分析 | 建议 |
|---|---|---|
| **权重访存** | 16.02 GB @ 4.92 BPW，decode 阶段每 token 需全量读一遍 | 4-bit 已到位；可上 Q4_K_S/IQ4_XS 进一步压到 ~14 GB |
| **KV Cache** | 17 层 × 2 × 4 头 × 256 dim × 262144 ctx = **9.1 G 元素 ≈ 18.3 GB (f16)** | 必须开 `--cache-type-k q8_0`（V100S 的 FA 支持 `q8_0-q8_0`）；或用 `-ctk q4_0` |
| **SSM 状态** | 48 层 × 48 头 × 128 × 128 = 151 MB (f32) | 可降 f16/bf16 |
| **循环层 conv 状态** | 48 层 × 3 × 10240 ≈ 5.9 MB | 忽略不计 |
| **DeltaNet 递推** | 48 个串行 scan，decode 延迟关键路径 | 需专用 kernel（llama.cpp `GGML_OP_GATED_DELTA_NET`），不要拆成小算子 |
| **MTP 块** | 缺 imatrix → 量化误差偏大，直接影响投机解码接受率 | 重跑 imatrix 或单独提精度 |
| **V100S 限制** | CC 7.0 无 BF16 TC，反量化 + FP16 GEMM 混合路径；无 FP8 | 避免 BF16 累加；用 FP16 激活 |

---

## 七、缺失信息 & 补充采集

**日志已覆盖**：张量清单、形状、类型、量化分布、imatrix 覆盖度、体积。
**日志未覆盖**（如需进一步分析需补采）：

1. **算子级图结构**（节点连接、融合边界）→ 日志只有权重，无计算图
2. **实际 kernel 实现与融合策略**
3. **端到端性能基线**（prefill/decode 吞吐、TTFT、显存峰值）

**补充采集命令**

```bash
# 1) 导出完整计算图（GGUF → Graphviz/JSON），获取真实算子节点与连接
python3 llama.cpp/gguf-py/scripts/gguf_dump.py \
    --no-tensors Qwen3.8-27B-Q4_K_M-imatrix.gguf

# 2) 打印推理期实际执行的 ggml 图（含融合结果）
./llama-cli -m Qwen3.8-27B-Q4_K_M-imatrix.gguf \
    -p "test" -n 1 --verbose-prompt -ngl 99 2>&1 | tee infer_graph.log

# 3) 逐层/逐算子耗时分解（perf）
./llama-bench -m Qwen3.8-27B-Q4_K_M-imatrix.gguf -ngl 99 -p 512 -n 128 -r 3

# 4) 补齐 blk.64(MTP) 的 imatrix 后重新量化
./llama-imatrix -m Qwen3.8-27B-BF16.gguf -f calibration/self_generated.txt \
    -o imatrix-full.gguf -c 512 --chunks 100 -ngl 99
./llama-quantize --imatrix imatrix-full.gguf \
    Qwen3.8-27B-BF16.gguf Qwen3.8-27B-Q4_K_M-full.gguf Q4_K_M 10

# 5) 查看每层 imatrix 覆盖情况
./llama-imatrix -m Qwen3.8-27B-BF16.gguf --show-statistics -f calibration/self_generated.txt
```

---

## 八、可复用解析流程（换模型/换日志时套用）

```
Step 1  抓元信息        grep 'kv ' 日志 → arch/block_count/hidden/heads/ssm/rope
Step 2  列张量清单      grep '^\[ *N/ *M\]' → 名称 + 形状 + 类型
Step 3  基名聚合        sed 去掉 blk.\d+. → 统计每种张量出现次数
Step 4  层型判定        含 attn_qkv/ssm_* → 循环层；含 attn_q/attn_k → 全注意力；含 nextn.* → MTP
Step 5  恒等式校验      量化张量数 == 所有 2D 权重矩阵数；f32 数 == norm+scale+bias+conv
Step 6  算子映射        张量 → 算子（见 §4.1），按层型累加
Step 7  量化分布交叉表  基名 × {q4_K,q6_K,f32} → 找 Q6_K 提升带
Step 8  imatrix 覆盖度  grep 'did not find weights' → 定位未加权张量（质量风险）
Step 9  瓶颈建模        BPW × 参数量 → 权重流量；KV/SSM 状态 → 显存；层型 → 延迟路径
```

---

## 九、§三 与 §四 的关系：同一节点集合的两个正交投影

> 本节回答一个常见疑问：**「三、模型节点统计」和「四、基础算子节点清单」看起来都在数数，区别是什么？**
> 结论：它们是同一批算子节点的**两个正交投影**，回答**两个完全不同的问题**。

### 9.1 一句话区别

| | §三 模型节点统计 | §四 基础算子节点清单 |
|---|---|---|
| **分组维度** | 按**位置**分组（这节点长在哪一层） | 按**类型**分组（这节点是干什么的） |
| **回答的问题** | 模型**有多大**？压力**集中在哪**？ | 模型**由哪些算子构成**？每种算子**什么语义**？ |
| **产物形态** | 数量分布表 | 算子字典（类型 → 次数 → 作用） |
| **类比** | 人口普查「按地区分布」 | 职业分类表「按工种分布」 |

### 9.2 各自的目的

**§三 → 服务于「规模评估与资源规划」**

说明计算和访存的重量压在哪里：

- **§3.1 权重块统计**：866 个张量、16.02 GB、4.92 BPW → 回答「显存装得下吗」
- **§3.2 算子节点统计**：585 / 768 / 192 / 17 / 5 → 回答「哪一层型是主体」

得出的判断是：**48 个循环层贡献 768/1567 ≈ 49% 的节点**，优化重心在线性注意力分支；65 个 FFN 贡献 195 个 GEMM，是访存主体。这些是**排优先级**的依据。

**§四 → 服务于「kernel 实现与后端适配」**

说明到底需要实现/支持哪些算子，每个算子长什么样：

- 17 类基础算子，每类的语义、数学形式、对应权重张量
- 例如第 7 行 `Gated Delta Rule`：作用是「S ← S·α + β·(k⊗v) 逐头递推，非 O(n²)」→ 这是**要写一个专用 kernel** 的信号
- §4.2 的算子族归类 → 这是**kernel 融合方案的输入**：哪些能 fuse、哪些必须独立

得出的判断是：**需要实现的算子集合 + 每类的融合可行性**。这是**定工作量、选算子库、查后端支持度**的依据。

### 9.3 两者关系：二维交叉表

把它们交叉起来看，就是一个二维表（行 = 算子类型，列 = 作用域）。**§三 = 最后一行的列合计，§四 = 最后一列的列合计**——同一个集合，换个方向加而已。

| 算子 | 公共(×65) | 循环专属(×48) | 全注意力专属(×16) | MTP专属(×1) | 全局(×1) | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| MatMul/GEMV | 195 | 240 | 64 | 5 | 2 | **506** |
| RMSNorm | 130 | 48 | 32 | 5 | 1 | **216** |
| Elementwise Mul | 65 | 96 | 17 | 0 | 0 | **178** |
| Add | 130 | 0 | 0 | 0 | 0 | **130** |
| SiLU | 65 | 48 | 0 | 0 | 0 | **113** |
| L2Norm | 0 | 96 | 0 | 0 | 0 | **96** |
| Split/Reshape/Transpose | 0 | 48 | 16 | 1 | 0 | **65** |
| Sigmoid | 0 | 48 | 17 | 0 | 0 | **65** |
| Conv1D | 0 | 48 | 0 | 0 | 0 | **48** |
| DeltaRule | 0 | 48 | 0 | 0 | 0 | **48** |
| Softplus/Exp | 0 | 48 | 0 | 0 | 0 | **48** |
| RoPE | 0 | 0 | 16 | 1 | 0 | **17** |
| SDPA | 0 | 0 | 16 | 1 | 0 | **17** |
| KV-Cache | 0 | 0 | 16 | 1 | 0 | **17** |
| Gather/Concat/Sampler | 0 | 0 | 0 | 0 | 3 | **3** |
| **列合计** | **585** | **768** | **192** | **17** | **5** | **1567** |

### 9.4 ⚠️ 一致性自校验与口径修正

正因为是同一集合的两个投影，**§三的总计和 §四的总计理论上必须相等**。本报告已按统一口径修正：

| 口径 | §三 总计 | §四 总计 | 是否一致 |
|---|---:|---:|---|
| 早期版本（口径不一致） | ≈1471 | ≈1567 | ✗ 差 ~6% |
| **当前版本（统一口径，含胶水算子）** | **1567** | **1567** | ✔ |

差异根因：**`Split / Reshape / Transpose` 这类「胶水算子」的计入口径不一致**——早期 §3.2 的「循环层 14 节点/块」把 `Split` 列进了清单但算式未计入，而 §4.1 单独给了 `Split/Reshape/Transpose ≈65` 一行。现已统一为**含胶水算子口径**，循环层 16 节点/块。

**精度声明**：

- **§3.1 权重块统计是日志直出的确定值**（866 = 506 + 360，三重恒等式校验通过）→ 做方案选型以此为准
- **§3.2 / §4.1 的算子节点数是按 llama.cpp 图节点粒度的估计值**，精度约 ±5–10% → 用于看**量级与比例**，不要当精确值引用

### 9.5 落到端侧优化上的实际分工

| 你手里的任务 | 看哪一节 |
|---|---|
| 显存够不够 / 要不要再压量化 | §3.1 + §5 |
| 先优化哪个分支（先做 SSM 还是先做 FFN） | §3.2 |
| 要写哪些 kernel、哪些能 fuse | §4 |
| 选算子库 / 查后端支持度（NPU 是否支持 DeltaNet） | §4.1 |
| 性能瓶颈定位（访存 vs 延迟路径） | §4.2 的族归类 |

---

### 结论摘要

- **Qwen3.8-27B** 是 **65 块混合注意力**模型：**48 层 Gated DeltaNet 线性注意力 + 16 层 GQA 全注意力 + 1 个 MTP 草稿块**，hidden 5120、FFN 17408、24/4 头、head_dim 256、部分 RoPE(64维)/MRoPE、262144 上下文。
- **866 个权重块** → **506 个线性算子 + 360 个归一化/缩放/卷积参数**，推算 **约 1471 个融合后算子节点**。
- **17 类基础算子**，其中 MatMul(506)、RMSNorm(216)、逐元素(≈470) 是数量主体；Conv1D/DeltaRule/SDPA(130) 是延迟关键路径。
- **两个需立刻处理的风险**：① **blk.64(MTP) 全部无 imatrix**，投机解码质量无保障；② **262k 上下文下 KV Cache ≈18 GB(f16)**，必须开 q8_0/q4_0 KV 量化。
