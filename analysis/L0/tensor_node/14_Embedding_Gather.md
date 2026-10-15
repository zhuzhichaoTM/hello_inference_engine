# 算子 14 — Embedding Gather

> 所属：Qwen3.8-27B 算子字典 · 第 **14 / 17** 项
> 唯一依据：报告 §4.1 #14、§9.3、§2.2、§2.3、§5.2、§4.2
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 14 | 本文档 |
| 算子名称 | Embedding Gather | §4.1 #14 |
| 算子族 | **前后处理** | §4.2 |
| 优化优先级 | ★ 可忽略 | §4.2 |
| 出现次数 | **1** | §4.1 #14 |
| 是否含权重 | 是，`token_embd.weight` | §4.1 #14 |
| 是否延迟关键路径 | 否 | §4.2 |
| 作用（报告原文） | token → hidden | §4.1 #14 |
| 对应权重/依据 | `token_embd.weight` | §4.1 #14 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 判定 |
|---|---:|---:|---:|---|
| 全局 | 1 | 1 | **1** | 输入首步，仅一次 |
| 其余 | 0 | — | 0 | — |
| **合计** | | | **1** | ✔（§9.3 中 Gather 归入"Gather/Concat/Sampler = 3"的全局行） |

---

## 3. 数学定义

$$h_t = W_{emb}[\text{token}_t],\qquad W_{emb}\in\mathbb{R}^{5120\times248320}$$

即从嵌入矩阵按 token id **取列**（查表），**非矩阵乘**。

---

## 4. 张量签名（输入 / 输出）

| 项 | shape | 说明 |
|---|---|---|
| 输入 | $[T]$（token id，整数） | $T$ = 序列长度 |
| 权重 | $[5120, 248320]$ | `token_embd.weight` |
| 输出 | $[T, 5120]$ | hidden 维 = `embedding_length` |

> vocab = 248320（§2.1），hidden = 5120（§2.1）。

---

## 5. 报告内张量实例

| 张量 | shape | dtype | 数量 | 作用 |
|---|---|---|---|---|
| `token_embd.weight` | [**5120, 248320**] | **q4_K**（**无 imatrix**） | 1 | token → hidden |

- 参数量：$5120\times248320 = 1{,}271{,}398{,}400$（≈ 1.27 G）。
- **无 imatrix 风险**：报告 §5.2 明确 `token_embd.weight` 属「llama.cpp imatrix 采集约定不覆盖」的 10 个张量之一 → 回退标准量化，误差无加权保护。

---

## 6. 推理链路位置

```
Input Tokens [T]
   │
   ▼
[Embedding Gather: token_embd.weight [5120,248320]]  →  x0 [T,5120]
   │
   ▼
blk.0 → blk.1 → … → blk.63 → blk.64 (MTP)
   │
   ▼
output_norm → LM-Head GEMM → Sampler
```

- **下游**：算子 03（首 block 的残差）、算子 02（首 block 前置 RMSNorm）
- 特殊：MTP 块的 `nextn.enorm` 也消费**同一份** embedding 输出（报告 §2.3 / §4.1 #15 Concat）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | $T$ 次查表 | 1 次查表 |
| 成本 | 读 $T\times5120$ 元素 | 读 $5120$ 元素 |
| 瓶颈 | 几乎为零 | 几乎为零 |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：前后处理（Gather/Concat/Sampler = 3），**★ 可忽略**。
- 要点：查表 = 只读被选中的行（或列，取决于布局），**不需要读整个 1.27 G 参数**。
- 注意：权重是 q4_K 量化 → 查表时需对**被选中行**做 dequant（融合进 Gather kernel）。

---

## 9. 性能特征

- FLOPs：0（纯访存）。
- 访存：每 token 读 $5120$ 元素（约 10 KB @ f16）→ 相对 16.80 GB 权重流量可忽略。
- 权重常驻：`token_embd.weight` ≈ **682 MiB**（q4_K，见文档 01 体积推导），是显存常驻项。

---

## 10. 数值精度与量化约束

| 项 | 说明 | 依据 |
|---|---|---|
| 量化 | q4_K | §3.1 |
| **无 imatrix** | 输入嵌入误差无加权保护 | §5.2 |
| 影响面 | 影响所有下游 token 表征 | 推导 |
| V100S | 反量化 + FP16 | §六 |

---

## 11. 最小参考实现（伪代码）

```cpp
// 报告 §4.1 算子#14：token → hidden；token_embd.weight [5120,248320] q4_K（无 imatrix）
// 注：布局 [hidden, vocab] → 按 token id 取"列"，逐元素 dequant
void embedding_gather(const uint8_t* W_q4k, const int* tokens, float* x, int T, int hidden=5120) {
    for (int t = 0; t < T; ++t) {
        int id = tokens[t];                              // 0 ≤ id < 248320
        for (int i = 0; i < hidden; ++i)
            x[t*hidden + i] = dequant_q4k_element(W_q4k, /*row=i, col=id*/ i, id);
    }
}
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 1 次、token → hidden | §4.1 #14 |
| 全局 1 次分布 | §9.3 |
| `token_embd.weight [5120,248320]` q4_K | §2.3 / §3.1 |
| vocab 248320、hidden 5120 | §2.1 |
| 无 imatrix | §5.2 |
| 族归类、可忽略 | §4.2 |

---

## 13. 边界说明

- 报告 §9.3 中 `token_embd` 亦被计入 MatMul 的"全局 2"（因它是 2D 量化权重矩阵，参与 §3.1 的 506 计数），但**运行时算子是 Gather 而非 GEMM** → 本文档按 §4.1 #14 记为 Gather，并保留该口径差异说明。
- 报告未给出嵌入矩阵的**物理布局**（行主序 [vocab,hidden] 还是 [hidden,vocab]）；本文档按报告 shape 记法 `[5120, 248320]` 处理。
- 未涉及 tokenizer（报告未覆盖，§7 列为缺失信息）。
