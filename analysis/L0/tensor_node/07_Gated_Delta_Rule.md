# 算子 07 — Gated Delta Rule / 分块线性递推扫描

> 所属：Qwen3.8-27B 算子字典 · 第 **07 / 17** 项
> 唯一依据：报告 §4.1 #7、§9.3、§2.1、§2.3、§六、§9.2
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 07 | 本文档 |
| 算子名称 | Gated Delta Rule / 分块线性递推扫描 | §4.1 #7 |
| 算子族 | **序列算子** | §4.2 |
| 优化优先级 | ★★★ 延迟关键路径 | §4.2 |
| 出现次数 | **48** | §4.1 #7 |
| 是否含权重 | **间接**，`ssm_a` / `ssm_dt.bias`（f32） | §4.1 #7 |
| 是否延迟关键路径 | **是（核心 SSM 算子）** | §4.1 #7 |
| 作用（报告原文） | 核心 SSM 算子，逐头递推 $S \leftarrow S\cdot\alpha + \beta\,(k\otimes v)$，**非 O(n²)** | §4.1 #7 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 判定 |
|---|---:|---:|---:|---|
| 循环专属 | 1 | 48 | **48** | 仅循环层（Gated DeltaNet） |
| 公共 / 全注意力 / MTP / 全局 | 0 | — | 0 | — |
| **合计** | | | **48** | ✔ 与 §9.3 一致 |

> 报告 §9.2：「48 个循环层贡献 768/1567 ≈ 49% 的节点，优化重心在线性注意力分支」——DeltaRule 是该分支的**计算核心**。

---

## 3. 数学定义（报告 §4.1 #7 原文递推式）

**逐头线性递推**（每头持有一个 $[d_k,d_v]=[128,128]$ 状态矩阵 $S$）：

$$S \leftarrow S\cdot\alpha \;+\; \beta\,(k\otimes v)$$

- $\alpha$：**衰减门**（输入相关），由 `ssm_alpha.weight` 投影 + Softplus/Exp 衰减（算子 09）产生
- $\beta$：**写入门**（输入相关），由 `ssm_beta.weight` 投影 + Sigmoid（算子 10）产生，作用于 Mul（算子 05）
- $k\otimes v$：外积，$k\in\mathbb{R}^{128}, v\in\mathbb{R}^{128}$ → 秩 1 更新 $[128,128]$

**复杂度**：每头每 token $O(d_k d_v)$，与序列长度 **无关** → **非 O(n²)**（报告强调点）。

---

## 4. 张量签名（输入 / 输出）

| 项 | shape | 说明 |
|---|---|---|
| 状态 $S$ | $[48\ \text{头},\,128,\,128]$ | 持久递推状态（= SSM 状态，§六） |
| $k$（输入） | $[T,\,48,\,128]$ | conv1d+L2Norm 后 |
| $v$（输入） | $[T,\,48,\,128]$ | `attn_qkv` 的 V 分支（6144=48×128） |
| $\alpha$ | $[T,\,48]$ | 每头标量衰减 |
| $\beta$ | $[T,\,48]$ | 每头标量写入门 |
| 输出 | $[T,\,48,\,128]$ | $S\cdot q$ 或头读出，送 `ssm_norm` |

**SSM 超参（报告 §2.1）**：`state_size=128`、`group_count=16`、`time_step_rank=48`、`inner_size=6144`。

> V 头数 = 6144 / 128 = **48** = `time_step_rank` ✔（与 `ssm_a`/`ssm_dt.bias` 长度 48 吻合）。Q/K 头数 = 2048/128 = **16** = `group_count` ✔。

---

## 5. 报告内张量实例

| 张量 | shape | dtype | 数量 | 作用 |
|---|---|---|---|---|
| `ssm_a` | [48] | **f32** | 48 | 每头衰减系数 A_log |
| `ssm_dt.bias` | [48] | **f32** | 48 | 时间步偏置 |
| `ssm_alpha.weight` | [5120, 48] | q4_K | 48 | 输入相关 α（衰减门）投影 |
| `ssm_beta.weight` | [5120, 48] | q4_K | 48 | 输入相关 β（写入门）投影 |
| `ssm_norm.weight` | [128] | f32 | 48 | 每头 gated RMSNorm（head_v_dim=128，见文档 02） |
| `ssm_out.weight` | [6144, 5120] | q4_K | 48 | SSM 分支输出投影（见文档 01） |

---

## 6. 推理链路位置

```
循环层(blk∈{0,1,2,4,…,62}):
  attn_qkv GEMM → Split:
      ├ Q(2048) → L2Norm(算子08) ───────────┐
      ├ K(2048) → Conv1D(算子06) → SiLU(04) → L2Norm(08) ─┤
      └ V(6144) ──────────────────────────────────────────┤
                                                          ├─ [本算子 DeltaRule]
  ssm_alpha·x → Softplus/Exp(算子09) → α 门 ──────────────┤   S ← S·α + β·(k⊗v)
  ssm_beta·x  → Sigmoid(算子10) → β 门 → Mul(算子05) ─────┘
  → ssm_norm(文档02) → attn_gate 输出门(文档05) → ssm_out GEMM
```

- **上游**：算子 01/06/08/09/10/05
- **下游**：算子 02（`ssm_norm`）、算子 05（输出门）、算子 01（`ssm_out`）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | 分块线性递推扫描（chunked scan，报告称「分块」） | 单步状态更新 |
| 状态读 | 初始化后顺序扫描 | 每 token 读 $[48,128,128]$ |
| 计算 | $T\times48\times128\times128\times2$ | $48\times128\times128\times2 \approx 1.57\,\text{M FLOPs/层}$ |
| 瓶颈 | 串行依赖（扫描结构） | **48 个串行 scan = decode 延迟关键路径**（§六） |

---

## 8. 实现要点与融合建议（§六）

报告 §六 明确：

> **DeltaNet 递推**：48 个串行 scan，decode 延迟关键路径 → **需专用 kernel（llama.cpp `GGML_OP_GATED_DELTA_NET`），不要拆成小算子**。

- 必须用**单一专用 kernel**承载整段递推，避免逐 token launch。
- 衰减门 $\alpha$、写入门 $\beta$ 在 kernel 内消费，**不应外溢**到独立 Mul/Sigmoid 节点（报告把 178/65/48 的 Mul/Sigmoid/Softplus 计入节点，工程上应 fuse）。

---

## 9. 性能特征

- 每 token 每层 FLOPs：$48\times128\times128\times2 = 1{,}572{,}864 \approx 1.57\,\text{MFLOPs}$。
- 48 层 → **≈ 75 MFLOPs/token**，相对 52.1 GFLOPs 线性层 **占 0.14%**（可忽略）。
- **但延迟关键**：FLOPs 小 ≠ 延迟小。串行 scan 是 decode 主路径瓶颈之一（§六）。
- 优势：相比全注意力 $O(n^2)$，DeltaRule 状态尺寸与 $T$ **无关** → 长上下文线性扩展。

---

## 10. 数值精度与量化约束

| 项 | 说明 | 依据 |
|---|---|---|
| `ssm_a`/`ssm_dt.bias` | **f32**，不参与 Q4_K/Q6_K | §3.1 |
| `ssm_alpha`/`ssm_beta` | q4_K | §3.1 |
| 状态 dtype | SSM 状态 48×48×128×128 = 151 MB (f32)，**可降 f16/bf16** | §六 |
| 衰减门公式 | $g = -\exp(A_{log})\cdot\text{softplus}(a + dt\_bias)$ | §4.1 #9 |
| V100S | FP16 激活，避免 BF16 累加 | §六 |

---

## 11. 最小参考实现（伪代码，逐头递推，非 chunked 版）

```cpp
// 报告 §4.1 算子#7：S ← S·α + β·(k⊗v)，逐头递推，非 O(n²)
// S: [n_head=48, dk=128, dv=128] f32（持久状态）
// k, v: [T, 48, 128];  alpha: [T, 48];  beta: [T, 48]
void gated_delta_rule(const float* k, const float* v, const float* alpha, const float* beta,
                      float* S, float* out, int T, int H=48, int D=128) {
    for (int h = 0; h < H; ++h) {
        float* Sh = S + (size_t)h*D*D;
        for (int t = 0; t < T; ++t) {                       // 串行递推：t 依赖 t-1
            const float a = alpha[t*H + h];                 // 衰减门（来自 Softplus/Exp，文档09）
            const float b = beta [t*H + h];                 // 写入门（来自 Sigmoid，文档10）
            const float* kt = k + (size_t)t*H*D + (size_t)h*D;
            const float* vt = v + (size_t)t*H*D + (size_t)h*D;
            float* ot = out + (size_t)t*H*D + (size_t)h*D;
            for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) {
                Sh[i*D+j] = Sh[i*D+j] * a + b * kt[i] * vt[j];   // S·α + β(k⊗v)
                ot[i]    += Sh[i*D+j] /* × q_i（读出，简化省略） */;
            }
        }
    }
}
```

> 生产环境**必须**用专用 kernel（报告 §六 `GGML_OP_GATED_DELTA_NET`），上式仅为逻辑演示。

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 48 次、递推式、非 O(n²)、核心 SSM | §4.1 #7 |
| 0/48/0/0/0 分布 | §9.3 |
| `ssm_a[48]`/`ssm_dt.bias[48]`/`ssm_alpha`/`ssm_beta`/`ssm_norm`/`ssm_out` | §2.3 |
| `state_size=128`/`time_step_rank=48`/`group_count=16`/`inner_size=6144` | §2.1 |
| SSM 状态 151 MB、可降 f16/bf16 | §六 |
| 专用 kernel、不拆小算子 | §六 |
| 循环层占节点 ≈49% | §9.2 |

---

## 13. 边界说明

- 报告未给出 DeltaRule 的**读出**（query 侧）完整公式，只给递推式与 `ssm_a`/`ssm_dt.bias`；本文档状态更新忠实于 §4.1 #7，读出以"简化省略"标注。
- 分块（chunked）扫描的具体块大小报告未给出。
- 衰减门 $g$ 与 α 的关系（文档 09）报告给公式，但未给出 α 是否等于 $\exp(g)$ 的确证，本文档不展开。
