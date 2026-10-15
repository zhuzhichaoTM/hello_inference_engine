# 算子 05 — Elementwise Mul

> 所属：Qwen3.8-27B 算子字典 · 第 **05 / 17** 项
> 唯一依据：报告 §4.1 #5、§9.3、§2.3、§4.2
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 05 | 本文档 |
| 算子名称 | Elementwise Mul | §4.1 #5 |
| 算子族 | **逐元素** | §4.2 |
| 优化优先级 | ★★ 全部可 kernel-fuse | §4.2 |
| 出现次数 | **178** | §4.1 #5 |
| 是否含权重 | **否** | §4.1 #5 |
| 是否延迟关键路径 | 否 | §4.2 |
| 作用（报告原文） | FFN 门控 65 + 线性注意力输出门 48 + Q-Gate(sigmoid 后) 17 + DeltaNet 内 β 门控 48 | §4.1 #5 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 具体用途 |
|---|---:|---:|---:|---|
| 公共 | 1 | 65 | **65** | FFN 门控（SiLU(gate) ⊙ up） |
| 循环专属 | 2 | 48 | **96** | ① 线性注意力输出门 48 ② DeltaNet β 门控 48 |
| 全注意力专属 | 1 | 16 | **16** | Q-Gate（Sigmoid 后）逐元素乘 |
| MTP 专属 | 1 | 1 | **1** | MTP 全注意力层的 Q-Gate |
| 全局 | 0 | 1 | 0 | — |
| **合计** | | | **178** | 65 + 96 + 16 + 1 = 178 ✔ |

> 校验：报告 §9.3 记 Elementwise Mul = 公共 65 / 循环专属 96 / 全注意力专属 17 / 其它 0 = **178** ✔（全注意力专属 17 = 16 全注意力 + 1 MTP）

---

## 3. 数学定义

$$y_i = a_i \odot b_i \quad(\text{逐元素})$$

四个具体场景：

| 场景 | 公式 | 门控信号来源 |
|---|---|---|
| **FFN 门控** | $\text{SiLU}(W_{gate}x)\odot(W_{up}x)$ | 算子 04 SiLU |
| **线性注意力输出门** | $g_{\text{out}} \odot \text{SSM分支输出}$ | `attn_gate.weight` 投影 → Sigmoid |
| **Q-Gate** | $\sigma(q_{gate})\odot \text{SDPA}(q,k,v)$ | `attn_q.weight` 后 6144 维 → Sigmoid（算子 10） |
| **DeltaNet β 门控** | $\beta \odot (\cdot)$ | `ssm_beta.weight` 投影 → Sigmoid（算子 10） |

---

## 4. 张量签名（输入 / 输出）

| 场景 | 输入 A | 输入 B | 输出 |
|---|---|---|---|
| FFN 门控 | $[T,17408]$ | $[T,17408]$ | $[T,17408]$ |
| 输出门（循环层） | $[T,6144]$（`attn_gate` 投影） | $[T,6144]$（SSM 分支） | $[T,6144]$ |
| Q-Gate（全注意力） | $[T,24,256]$（Sigmoid 后） | $[T,24,256]$（SDPA 输出） | $[T,24,256]$ |
| β 门控 | $[T,48]$ | $[T,48,128]$（广播） | $[T,48,128]$ |

---

## 5. 报告内张量实例

**本算子无权重张量**，但其门控信号由以下权重张量产生：

| 门控 | 关联权重 | shape | 量化 |
|---|---|---|---|
| FFN 门控 | `ffn_gate.weight` / `ffn_up.weight` | [5120,17408]×2 | q4_K |
| 输出门（循环层） | `attn_gate.weight` | [5120, 6144] | q4_K ×48 |
| Q-Gate | `attn_q.weight`（后 6144 维） | [5120, 12288] | q4_K ×17 |
| β 门控 | `ssm_beta.weight` | [5120, 48] | q4_K ×48 |

---

## 6. 推理链路位置

```
循环层:   attn_qkv → Split → SSM分支 ────────────────┐
                                                     ├─ Mul → ssm_out GEMM
          attn_gate GEMM → Sigmoid ─────────────────┘   （输出门）
          ssm_beta GEMM → Sigmoid → Mul（β 门控，作用于 DeltaRule 内）

全注意力: attn_q GEMM → Split → [SDPA] ─────────────┐
                                                    ├─ Mul → attn_output GEMM
          attn_q GEMM → Split → Sigmoid ───────────┘   （Q-Gate）

公共:     ffn_gate → SiLU ─┐
                            ├─ Mul → ffn_down GEMM
          ffn_up   ────────┘   （FFN 门控）
```

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | $[N,\cdot]$ 批量逐元素乘 | 单行 |
| 成本 | $\propto N$ | 固定 |
| 瓶颈 | 与相邻 GEMM/激活 fuse，开销可忽略 | 同上；β 门控因作用于 $[48,128]$ 状态，需与 DeltaRule kernel 融合 |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：逐元素，**★★ 全部可 kernel-fuse**。
- 融合优先级：FFN 门控与 SwiGLU 融合；Q-Gate 与 SDPA epilogue 融合；输出门与 `ssm_out` GEMM 融合。

---

## 9. 性能特征

- 每次 1 次乘法/元素；总元素量级与对应 GEMM 输出同阶。
- 178 个节点在 1567 中占 **11.4%**，属数量大但算量小的节点，**是融合收益最高的类别之一**。

---

## 10. 数值精度与量化约束

| 项 | 说明 | 依据 |
|---|---|---|
| 无权重 | 不涉及量化与 imatrix | — |
| 门控输入精度 | Sigmoid 输出 ∈ (0,1)，与 FP16 激活相乘安全 | §六 |
| 风险 | Q-Gate / 输出门若在 FP16 下相乘，需注意下游 GEMM 的累加精度 | 推导（报告未明示） |

---

## 11. 最小参考实现（伪代码）

```cpp
// 报告 §4.1 算子#5：FFN门控65 + 输出门48 + Q-Gate17 + β门控48 = 178；无权重
void mul_elementwise(const float* a, const float* b, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = a[i] * b[i];
}
// FFN:      y = SiLU(ffn_gate·x) * (ffn_up·x)
// Q-Gate:   y = sigmoid(q_gate)  * sdpa_out
// 输出门:   y = sigmoid(attn_gate·x) * ssm_branch_out
// β 门控:   y = sigmoid(ssm_beta·x) * delta_rule_state_update
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 178 次及四类构成 | §4.1 #5 |
| 65/96/17/0/0 分布 | §9.3 |
| Q-Gate 融合于 `attn_q` | §2.3 |
| 输出门 `attn_gate` | §2.3 |
| 族归类、可 fuse | §4.2 |

---

## 13. 边界说明

- 报告未给出 β 门控的精确广播规则（$[T,48]$ 门控作用于 $[T,48,128]$ 状态），本文档按 shape 推导标注。
- 未展开 DeltaNet 内部 β 门控与递推的耦合细节（见文档 07）。
