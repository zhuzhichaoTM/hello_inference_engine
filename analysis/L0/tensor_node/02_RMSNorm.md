# 算子 02 — RMSNorm

> 所属：Qwen3.8-27B 算子字典 · 第 **02 / 17** 项
> 唯一依据：报告 §4.1 #2、§9.3、§2.3、§4.2、§5.2、§六
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 02 | 本文档 |
| 算子名称 | RMSNorm | §4.1 #2 |
| 算子族 | **归一化** | §4.2 |
| 优化优先级 | ★★ 融合进 GEMM 前处理 | §4.2 |
| 出现次数 | **216** | §4.1 #2 |
| 是否含权重 | 是，216 个 **f32** norm 张量 | §4.1 #2 / §3.1 |
| 是否延迟关键路径 | 否（但为 GEMM 前处理必经） | §4.2 |
| 作用（报告原文） | 归一化。含 attn_norm 65、post_attention_norm 65、ssm_norm(gated,per-head) 48、attn_q_norm 17、attn_k_norm 17、output_norm 1、nextn 3 | §4.1 #2 |
| eps | **1e-6** | §2.1 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 具体张量 |
|---|---:|---:|---:|---|
| 公共 | 2 | 65 | **130** | `attn_norm` / `post_attention_norm` |
| 循环专属 | 1 | 48 | **48** | `ssm_norm`（gated, per-head） |
| 全注意力专属 | 2 | 16 | **32** | `attn_q_norm` / `attn_k_norm` |
| MTP 专属 | 5 | 1 | **5** | `attn_norm` / `post_attention_norm` / `nextn.enorm` / `nextn.hnorm` / `nextn.shared_head_norm` |
| 全局 | 1 | 1 | **1** | `output_norm` |
| **合计** | | | **216** | 与 §3.1 恒等式（216 f32 norm）一致 ✔ |

---

## 3. 数学定义

$$\text{RMSNorm}(x)=\frac{x}{\sqrt{\dfrac{1}{n}\sum_{i=1}^{n}x_i^2+\epsilon}}\odot w,\qquad \epsilon=10^{-6}$$

### 三种变体（报告 §2.3）

| 变体 | 归一化轴 | 权重长度 | 张量 |
|---|---|---|---|
| **普通 RMSNorm** | 整条 hidden（5120） | 5120 | `attn_norm` / `post_attention_norm` / `output_norm` / `nextn.enorm` / `nextn.hnorm` / `nextn.shared_head_norm` |
| **gated per-head RMSNorm** | 每头沿 head_v_dim = **128** | 128 | `ssm_norm` |
| **per-head RMSNorm（QK-Norm）** | 每头沿 head_dim = **256** | 256 | `attn_q_norm` / `attn_k_norm` |

> 关键区分：`attn_q_norm`/`attn_k_norm` 是对 **每个头的 256 维**独立归一（per-head），**不是**对 24×256=6144 整段归一。

---

## 4. 张量签名（输入 / 输出）

| 变体 | 输入 | 权重 | 输出 |
|---|---|---|---|
| 普通 | $[T,5120]$ f16 | $[5120]$ f32 | $[T,5120]$ |
| per-head (QK-Norm) | $[T,24,256]$ | $[256]$ f32 | 同输入 |
| gated per-head (ssm) | $[T,48,128]$ | $[128]$ f32 | 同输入 |

---

## 5. 报告内张量实例

| 张量 | shape | dtype | 数量 | 作用 |
|---|---|---|---|---|
| `attn_norm.weight` | [5120] | f32 | 65 | 注意力分支前置 RMSNorm |
| `post_attention_norm.weight` | [5120] | f32 | 65 | 残差前 RMSNorm |
| `ssm_norm.weight` | [**128**] | f32 | 48 | 每头 gated RMSNorm（head_v_dim=128） |
| `attn_q_norm.weight` | [**256**] | f32 | 17 | QK-Norm，per-head |
| `attn_k_norm.weight` | [**256**] | f32 | 17 | QK-Norm，per-head |
| `output_norm.weight` | [5120] | f32 | 1 | 最终归一化 |
| `nextn.enorm / hnorm / shared_head_norm.weight` | [5120] | f32 | 3 | MTP 块归一化 |

---

## 6. 推理链路位置

- `attn_norm` → 各分支投影 GEMM 之前（算子 01 的上游）
- `attn_q_norm` / `attn_k_norm` → 在 Q/K 投影后、**MRoPE 之前**（算子 11 的上游）
- `ssm_norm` → DeltaNet 递推（算子 07）之后、`ssm_out` GEMM 之前
- `post_attention_norm` → 注意力输出与残差相加之前
- `output_norm` → LM-Head GEMM 之前（链路末端）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | 行批量归一化，$[N,5120]$ | 单行，$[1,5120]$ |
| 成本 | 与 N 线性 | 固定小开销 |
| 瓶颈 | 计算受限侧，占比极低 | 访存受限侧，需与相邻 GEMM 融合以省一次读写 |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：归一化（与 L2Norm 合并计 312 个），**★★ 融合进 GEMM 前处理**。
- 数值：全程 **f32** 计算（报告 §3.1：360 个 f32 张量中 216 个是 norm），最终以 f16 送入 GEMM。

---

## 9. 性能特征

- 计算量：$\approx 2n$ FLOPs/元素（平方和 + 缩放），$n$ = 归一化轴长度。
- 访存：读 $[T,n]$ 激活 + 读 $[n]$ 权重，写 $[T,n]$；**典型 memory-bound 小算子**。
- 无独立权重流量压力（216 个 f32 张量合计仅 **≈ 10.19 MiB**，见文档 01 §2 推导）。

---

## 10. 数值精度与量化约束

| 项 | 值 | 依据 |
|---|---|---|
| 权重 dtype | **f32**（不参与 Q4_K/Q6_K 量化） | §3.1 |
| eps | 1e-6 | §2.1 |
| 无 imatrix 影响 | norm 权重是 f32，**不受 imatrix 缺失影响** | §5.2（缺失清单中无 norm） |
| V100S | 用 FP16 激活，避免 BF16 累加 | §六 |

---

## 11. 最小参考实现（C++，与模块二一致）

```cpp
// 报告 §4.1 算子#2，eps=1e-6；张量实例 attn_norm[5120]/post_attention_norm[5120]/output_norm[5120]
void rms_norm(const float* x, const float* w, float* out, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float r = 1.0f / sqrtf(ss / (float)n + 1e-6f);   // eps = 1e-6
    for (int i = 0; i < n; ++i) out[i] = x[i] * r * w[i];
}

// per-head 变体：ssm_norm[128](head_v_dim=128) / attn_q_norm,attn_k_norm[256](head_dim=256)
void rms_norm_per_head(const float* x, const float* w, float* out, int n_head, int head_dim) {
    for (int h = 0; h < n_head; ++h)
        rms_norm(x + (size_t)h*head_dim, w, out + (size_t)h*head_dim, head_dim);
}
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 216 次及 7 类构成 | §4.1 #2 |
| 130/48/32/5/1 分布 | §9.3 |
| 三种变体及 shape | §2.3 |
| eps = 1e-6 | §2.1 |
| f32、不量化 | §3.1 |
| 融合进 GEMM 前处理 | §4.2 |

---

## 13. 边界说明

- 未实现 LayerNorm（报告未记录该算子）。
- `ssm_norm` 的 **gated** 语义（门控信号来源）报告未展开，本文档只标注其为 per-head gated RMSNorm。
- 未讨论 MTP 块 `nextn.*` 三个 norm 在投机解码中的具体接法（报告 §2.3 只给张量，无连接图）。
