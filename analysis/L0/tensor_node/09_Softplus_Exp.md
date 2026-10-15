# 算子 09 — Softplus / Exp

> 所属：Qwen3.8-27B 算子字典 · 第 **09 / 17** 项
> 唯一依据：报告 §4.1 #9、§9.3、§2.1、§2.3、§4.2
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 09 | 本文档 |
| 算子名称 | Softplus / Exp | §4.1 #9 |
| 算子族 | **逐元素** | §4.2 |
| 优化优先级 | ★★ 全部可 kernel-fuse | §4.2 |
| 出现次数 | **48** | §4.1 #9 |
| 是否含权重 | **间接**，`ssm_a` / `ssm_dt.bias`（f32） | §4.1 #9 |
| 是否延迟关键路径 | 是（衰减门在递推关键路径） | §4.2 |
| 作用（报告原文） | 衰减门 $g = -\exp(A_{log})\cdot\text{softplus}(a + dt\_bias)$ | §4.1 #9 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 判定 |
|---|---:|---:|---:|---|
| 循环专属 | 1 | 48 | **48** | 仅循环层有衰减门 |
| 其余 | 0 | — | 0 | — |
| **合计** | | | **48** | ✔ 与 §9.3 一致 |

---

## 3. 数学定义

$$\text{softplus}(z)=\ln(1+e^{z}),\qquad g = -\exp(A_{log})\cdot\text{softplus}(a + dt\_bias)$$

- $A_{log}$：`ssm_a`（每头 [48]，f32），**取 exp 得到每头衰减基**
- $dt\_bias$：`ssm_dt.bias`（每头 [48]，f32）
- $a$：输入相关的时间步（由 `ssm_alpha` 投影得到，见文档 01）
- $g$：衰减门 → 进入 DeltaRule 递推（文档 07）作为 $\alpha$ 的驱动量

**数值语义**：$g \le 0$（因 $-\exp(\cdot)<0$，softplus $\ge 0$），是**衰减**（decay）而非增长。

---

## 4. 张量签名（输入 / 输出）

| 项 | shape | 说明 |
|---|---|---|
| `ssm_a`（$A_{log}$） | [48] | 每头，广播到 $[T,48]$ |
| `ssm_dt.bias` | [48] | 每头 |
| $a$（输入相关） | $[T,48]$ | `ssm_alpha.weight` 投影（文档 01） |
| 输出 $g$ | $[T,48]$ | 衰减门，逐头标量 |

---

## 5. 报告内张量实例

| 张量 | shape | dtype | 数量 | 作用 |
|---|---|---|---|---|
| `ssm_a` | [**48**] | **f32** | 48 | 每头衰减系数 A_log（取 exp） |
| `ssm_dt.bias` | [**48**] | **f32** | 48 | 时间步偏置 |
| `ssm_alpha.weight` | [5120, 48] | q4_K | 48 | 输入相关 α 投影（提供 $a$） |

> 长度 48 = `ssm.time_step_rank`（§2.1）✔。`ssm_a` 与 `ssm_dt.bias` 均为 f32（§3.1：360 f32 中 48 + 48 属此）。

---

## 6. 推理链路位置

```
循环层:
  ssm_alpha·x → a [T,48] ─┐
  ssm_dt.bias ────────────┼─► (a + dt_bias) → softplus → ×(-exp(ssm_a)) → g [T,48]
  ssm_a ──────► exp ──────┘                          │
                                                     ▼
                                     α 门 → DeltaRule(文档07) S ← S·α + β(k⊗v)
```

- **上游**：算子 01（`ssm_alpha` GEMM）
- **下游**：算子 07（DeltaRule，作为衰减门驱动）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | $[T,48]$ 批量 | $[1,48]$ |
| 成本 | $\propto T$，每元素 1 softplus + 1 exp | 固定 48 元素 |
| 瓶颈 | 与 DeltaRule 融合 | **在递推关键路径上**（每 token 必须算 α） |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：逐元素，**★★ 全部可 kernel-fuse**。
- 建议：α 门计算（softplus + exp + mul）**必须与 DeltaRule 专用 kernel 融合**，避免关键路径上多出独立节点 launch（报告 §六 强调 DeltaNet 不要拆成小算子）。

---

## 9. 性能特征

- FLOPs：softplus ≈ exp + log ≈ 3 ops；exp ≈ 1；mul ≈ 1 → 约 6 ops/元素。
- 每层 48 元素 → 288 ops；48 层 → 13.8 K ops/token，**可忽略**。
- 延迟敏感性：高（在 48 层串行 scan 的输入侧）。

---

## 10. 数值精度与量化约束

| 项 | 说明 | 依据 |
|---|---|---|
| `ssm_a`/`ssm_dt.bias` | **f32**，不量化 | §3.1 |
| `ssm_alpha` | q4_K | §3.1 |
| 数值范围 | $g\le0$，exp/softplus 均需 f32 计算防溢出 | 推导（报告未明示） |
| V100S | FP16 激活，避免 BF16 累加 | §六 |

---

## 11. 最小参考实现（伪代码）

```cpp
// 报告 §4.1 算子#9：g = -exp(A_log) * softplus(a + dt_bias)；ssm_a[48]/ssm_dt.bias[48] f32
void decay_gate(const float* a /*[T,H]*/, const float* A_log /*[H]*/, const float* dt_bias /*[H]*/,
                float* g /*[T,H]*/, int T, int H=48) {
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            float z = a[t*H + h] + dt_bias[h];
            float sp = log1pf(expf(z));            // softplus
            g[t*H + h] = -expf(A_log[h]) * sp;     // 衰减门（≤ 0）
        }
}
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 48 次、衰减门公式 | §4.1 #9 |
| 0/48/0/0/0 分布 | §9.3 |
| `ssm_a[48]`/`ssm_dt.bias[48]` f32 | §2.3 / §3.1 |
| `time_step_rank=48` | §2.1 |
| 族归类、可 fuse | §4.2 |

---

## 13. 边界说明

- 报告只给 $g$ 的公式，**未**明确 $g$ 与 DeltaRule 中 $\alpha$ 的换算关系（是否 $\alpha=\exp(g)$）→ 本文档不展开，仅标注 $g$ 驱动 α。
- 未给出 $a$（输入相关时间步）是否经过额外激活或缩放。
- 不引入报告外的 SSM 变体（如 Mamba 的 $\Delta$ 参数化细节）。
