# 算子 01 — MatMul / GEMM(GEMV)

> 所属：Qwen3.8-27B 算子字典 · 第 **01 / 17** 项
> 唯一依据：报告 §4.1 #1、§9.3、§3.1、§4.2、§六、§5.4
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 01 | 本文档 |
| 算子名称 | MatMul / GEMM(GEMV) | §4.1 #1 |
| 算子族 | **线性层（权重）** | §4.2 |
| 优化优先级 | ★★★ **访存瓶颈**（4.92 BPW，16 GB 权重） | §4.2 |
| 出现次数 | **506** | §4.1 #1 |
| 是否含权重 | **是**，506 个量化张量全部 | §3.1 |
| 是否延迟关键路径 | 是（decode 阶段主瓶颈） | §4.2 / §六 |
| 作用（报告原文） | 全部线性投影。prefill 为 GEMM（计算受限），decode 为 GEMV（**访存受限**） | §4.1 #1 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 具体张量 |
|---|---:|---:|---:|---|
| 公共 | 3 | 65 | **195** | `ffn_gate` / `ffn_up` / `ffn_down` |
| 循环专属 | 5 | 48 | **240** | `attn_qkv` / `attn_gate` / `ssm_alpha` / `ssm_beta` / `ssm_out` |
| 全注意力专属 | 4 | 16 | **64** | `attn_q` / `attn_k` / `attn_v` / `attn_output` |
| MTP 专属 | 5 | 1 | **5** | `attn_q` / `attn_k` / `attn_v` / `attn_output` / `nextn.eh_proj` |
| 全局 | 2 | 1 | **2** | `token_embd` / `output` |
| **合计** | | | **506** | 与 §3.1 恒等式一致 ✔ |

> **口径说明**：报告按"量化张量"计数（506 = 195 + 240 + 68 + 1 + 1 + 1），其中 `token_embd.weight` 是 2D 量化权重矩阵故计入 506，但其**推理期运行时算子为 Gather**（见文档 14）。此处忠实保留报告计数口径。

---

## 3. 数学定义

对权重 $W\in\mathbb{R}^{d_{out}\times d_{in}}$、输入 $X\in\mathbb{R}^{N\times d_{in}}$：

$$Y = X W^\top,\qquad Y\in\mathbb{R}^{N\times d_{out}}$$

- **Prefill**：$N$ = 提示长度（$N\gg 1$）→ 矩阵乘 **GEMM**
- **Decode**：$N = 1$ → 矩阵-向量乘 **GEMV**

---

## 4. 张量签名（输入 / 输出）

| 阶段 | 输入 | 权重 | 输出 |
|---|---|---|---|
| Prefill | $X:[N,5120]$ f16 | $W$ 见下表 | $Y:[N,d_{out}]$ |
| Decode | $X:[1,5120]$ f16 | 同上 | $Y:[1,d_{out}]$ |

`token_embd` / `output` 的 $d_{out} = 248320$；FFN 内部 $d_{out} = 17408$。

---

## 5. 报告内权重张量实例（按分组）

| 张量 | shape | dtype | 组内数量 |
|---|---|---|---|
| `ffn_gate.weight` | [5120, 17408] | q4_K | 65 |
| `ffn_up.weight` | [5120, 17408] | q4_K | 65 |
| `ffn_down.weight` | [17408, 5120] | **q6_K×33 / q4_K×32** | 65 |
| `attn_qkv.weight` | [5120, 10240] | **q6_K×24 / q4_K×24** | 48 |
| `attn_gate.weight` | [5120, 6144] | q4_K | 48 |
| `ssm_alpha.weight` | [5120, 48] | q4_K | 48 |
| `ssm_beta.weight` | [5120, 48] | q4_K | 48 |
| `ssm_out.weight` | [6144, 5120] | q4_K | 48 |
| `attn_q.weight` | [5120, 12288] | q4_K | 17 |
| `attn_k.weight` | [5120, 1024] | q4_K | 17 |
| `attn_v.weight` | [5120, 1024] | **q6_K×9 / q4_K×8** | 17 |
| `attn_output.weight` | [6144, 5120] | q4_K | 17 |
| `nextn.eh_proj.weight` | [10240, 5120] | q4_K | 1 |
| `token_embd.weight` | [5120, 248320] | q4_K（**无 imatrix**） | 1 |
| `output.weight` | [5120, 248320] | **q6_K**（**无 imatrix**） | 1 |

---

## 6. 推理链路位置

`Gather` → 每个 block 内：`RMSNorm → GEMM(Q/K/V)` / `RMSNorm → GEMM(FFN)` → `Residual Add` → 下一 block → `output_norm` → **LM-Head GEMM**（`output.weight`）→ Sampler。

- **上游**：算子 02 RMSNorm（权重前处理）、算子 14 Gather（首层输入）
- **下游**：算子 03 Add、算子 12 SDPA、算子 04 SiLU（FFN 门控分支）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | GEMM（$N$ 大） | GEMV（$N=1$） |
| 瓶颈 | **计算受限**（FP16 GEMM 吞吐） | **访存受限**（权重全量读取） |
| 权重读取次数 | 1 次 / 整个 prompt | **每 token 1 次**（全量 16.02 GB） |
| 算术强度 | $\propto N$，$N\ge512$ 即远超拐点 | $\approx 3.13$ FLOP/Byte（报告口径：52.1 GFLOPs ÷ 16.80 GB） |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：线性层，**★★★ 访存瓶颈**。
- **反量化融合进 kernel**：报告 §5.4 明确 `dequantize_row_q4_K` 内联于 MatMul kernel，**不单独成节点**。
- V100S（CC 7.0）无 BF16/FP8 Tensor Core → 走 **「反量化 + FP16 GEMM」混合路径**，避免 BF16 累加。

---

## 9. 性能特征

- **参数总量**：506 张量合计 **≈ 27.32 G 参数**（含 `token_embd`）。
- **每 token 算量**（$2\times$ 参数，仅真实 GEMM，不含 Gather）：

| 分组 | 参数量 | GFLOPs/token |
|---|---:|---:|
| FFN（65 层） | 17.380 G | 34.76 |
| 循环分支（48 层） | 5.560 G | 11.12 |
| 全注意力 Q/K/V/O（16 层） | 1.678 G | 3.36 |
| MTP 专属（4 投影 + eh_proj） | 0.157 G | 0.31 |
| 全局 LM-Head（`output`） | 1.271 G | 2.54 |
| **合计** | **26.05 G** | **≈ 52.1** |

- **权重流量**：decode 每 token 读 **16.80 GB**（16,021.46 MiB，报告 §一）。
- **FLOPs 公式**：$\text{FLOPs} = 2\times N\times(\text{参数量})$。

---

## 10. 数值精度与量化约束

| 项 | 值 | 依据 |
|---|---|---|
| 权重格式 | GGUF Q4_K_M（4.92 BPW），压缩比 3.25× | §一 |
| 混合精度 | q4_K 439 / q6_K 67；提升带 = `ffn_down`/`attn_qkv`/`attn_v`/`output` | §5.3 |
| 无 imatrix 张量 | `token_embd.weight`、`output.weight`（首尾误差无加权保护） | §5.2 |
| V100S 约束 | CC 7.0 无 BF16 TC / 无 FP8；反量化 + FP16 GEMM | §六 |
| recipe 风险 | 本日志为标准 Q4_K_M，**未走 yaml `tensor_overrides`** | §一 |

---

## 11. 最小参考实现（伪代码）

```cpp
// Y[N, d_out] = X[N, d_in] * W[d_out, d_in]^T
// W 为 Q4_K/Q6_K 量化张量：先反量化到 f16，再做 FP16 GEMM（V100S 无 BF16/FP8）
for (int n = 0; n < N; ++n)                 // Prefill: N>1 → GEMM; Decode: N=1 → GEMV
    for (int o = 0; o < d_out; ++o) {
        float acc = 0.f;
        for (int i = 0; i < d_in; ++i)
            acc += X[n*d_in + i] * dequant_q4k(W, o*d_in + i);  // 反量化内联，非独立算子
        Y[n*d_out + o] = acc;
    }
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 506 次、prefill GEMM / decode GEMV、访存受限 | §4.1 #1 |
| 族归类、★★★ 访存瓶颈 | §4.2 |
| 195/240/64/5/2 分布 | §9.3 |
| 506 = 195+240+68+1+1+1 恒等式 | §3.1 |
| 16.02 GB @ 4.92 BPW | §一 / §六 |
| dequant 融合、非独立算子 | §5.4 |
| Q6_K 提升带 | §5.3 |
| 无 imatrix 张量 | §5.2 |
| V100S CC7.0 数值路径 | §六 |

---

## 13. 边界说明

- 本文档**不含**独立 Dequant 算子（报告 §5.4 明示其融合进 MatMul kernel）。
- 不含 `token_embd` 的 Gather 语义细节（见文档 14）。
- GFLOPs 数值为按报告 shape 推导，报告本身未直接给出 FLOPs 表。
- 未给出 V100S 的带宽 / TFLOPS 常量（报告未提供）。
