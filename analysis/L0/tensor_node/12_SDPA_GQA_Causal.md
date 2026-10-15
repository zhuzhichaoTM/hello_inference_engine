# 算子 12 — Scaled Dot-Product Attention (GQA 6:1, causal)

> 所属：Qwen3.8-27B 算子字典 · 第 **12 / 17** 项
> 唯一依据：报告 §4.1 #12、§9.3、§2.1、§2.3、§4.2、§六
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。
> **重要：仅全注意力层使用本算子；48 个循环层是 Gated DeltaNet，不是标准 SDPA。**

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 12 | 本文档 |
| 算子名称 | Scaled Dot-Product Attention (GQA 6:1, causal) | §4.1 #12 |
| 算子族 | **序列算子** | §4.2 |
| 优化优先级 | ★★★ 延迟关键路径 | §4.2 |
| 出现次数 | **17** | §4.1 #12 |
| 是否含权重 | **否**（权重在 Q/K/V 投影，见文档 01） | §4.1 #12 |
| 是否延迟关键路径 | 是 | §4.2 |
| 作用（报告原文） | Flash-Attention / 分块注意力 | §4.1 #12 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 判定 |
|---|---:|---:|---:|---|
| 全注意力专属 | 1 | 16 | **16** | blk=3,7,11,…,63 |
| MTP 专属 | 1 | 1 | **1** | blk.64 |
| 循环专属 | 0 | 48 | 0 | **DeltaNet 无 SDPA（无 KV Cache）** |
| 公共 / 全局 | 0 | — | 0 | — |
| **合计** | | | **17** | ✔ 与 §9.3 一致 |

---

## 3. 数学定义

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V,\qquad d=\texttt{key\_length}=256$$

**GQA 6:1 分组**：24 个 Q 头，4 个 KV 头，每 **6** 个 Q 头共享 1 个 KV 头：

$$\text{kv}(h) = \left\lfloor \frac{h}{6} \right\rfloor,\qquad h\in[0,24)$$

**因果掩码**：$A_{ij}=0$（softmax 前设 $-\infty$）当 $j > i + \text{offset}$，$\text{offset}=T_k - T_q$。

---

## 4. 张量签名（输入 / 输出）

| 项 | shape | 说明 |
|---|---|---|
| Q | $[T_q, 24, 256]$ | 24 = `head_count`，256 = `key_length` |
| K | $[T_k, 4, 256]$ | 4 = `head_count_kv` |
| V | $[T_k, 4, 256]$ | 256 = `value_length` |
| 输出 | $[T_q, 24, 256]$ | → 重排 [T,6144] → `attn_output` GEMM |
| 缩放因子 | $1/\sqrt{256}$ | 固定 |

- $T_k = T_q + \text{offset}$；decode 时 $T_q=1$，$T_k$ = 已缓存长度。

---

## 5. 报告内张量实例

**本算子无权重**；其 Q/K/V 来自（文档 01）：

| 张量 | shape | 量化 |
|---|---|---|
| `attn_q.weight` | [5120,12288] | q4_K ×17 |
| `attn_k.weight` | [5120,1024] | q4_K ×17 |
| `attn_v.weight` | [5120,1024] | q6_K×9 / q4_K×8 |
| `attn_output.weight` | [6144,5120] | q4_K ×17 |

> 形状自洽：$24\times256=6144$（Q/O 维），$4\times256=1024$（K/V 维）✔

---

## 6. 推理链路位置

```
全注意力层 / MTP:
  attn_q GEMM [5120,12288] → Split(文档17) → 前6144 → QK-Norm(02) → MRoPE(11) → Q
  attn_k GEMM [5120,1024]  →             → QK-Norm(02) → MRoPE(11) → K → KV-Cache写(13)
  attn_v GEMM [5120,1024]  →                                          V → KV-Cache写(13)
  → [SDPA(GQA 6:1, causal)] → Mul(Q-Gate, 文档05) → attn_output GEMM
```

- **上游**：算子 02、11、13
- **下游**：算子 05（Q-Gate Mul）、算子 01（`attn_output` GEMM）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | $T_q = T$ 全序列，$O(T^2)$ | $T_q=1$，$O(T_k)$ |
| 计算 | $2\cdot\frac{T(T+1)}{2}\cdot(24\times256)\cdot2$/层 | $2\cdot2\cdot T_k\cdot(24\times256)$/层 |
| 瓶颈 | 长上下文时二次项与线性项同阶（$T=262144$ 时约相等） | **KV 读取访存受限**（文档 13） |
| 实现 | Flash-Attention 分块（报告 §4.1 #12） | 单 query 流式 |

**二次项量级**（推导，基于报告 head 数）：16 层合计 $\approx 1.97\times10^5\,T^2$ FLOPs；$T=4096$ 时 3.3 TFLOPs（占 1.5%），$T=262144$ 时 13.5 PFLOPs（与线性项 13.8 PFLOPs **同阶**）。

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：序列算子，**★★★ 延迟关键路径**。
- 要点：
  - **GQA**：KV 头仅 4 个 → KV 读取量是 MHA 的 1/6，是长上下文省显存的关键。
  - **causal**：只算下三角。
  - 报告 §六：V100S 的 FA 支持 **`q8_0-q8_0`** KV（与文档 13 联动）。
  - **不得**把循环层当作 SDPA（报告 §9.3：SDPA 仅 17 次，循环层 0 次）。

---

## 9. 性能特征

- 每 token decode FLOPs（$T_k=N_{ctx}$）：$16\ \text{层}\times 2\times2\times N_{ctx}\times6144 = 393{,}216\,N_{ctx}$。
  - $N_{ctx}=262144$ → ≈ **103 GFLOPs**，是线性层 52.1 GFLOPs 的 **约 2 倍** → 长上下文下注意力本身成为算量主体。
- KV 读取（f16）：18.25 GB/token（文档 13），**超过权重流量 16.80 GB** → 必须 KV 量化。

---

## 10. 数值精度与量化约束

| 项 | 说明 | 依据 |
|---|---|---|
| 无权重 | 不涉及 Q4_K/Q6_K 本身 | — |
| Q/K 精度 | 经 QK-Norm（f32 权重）与 RoPE | §2.3 |
| KV 精度 | 可 f16/bf16/q8_0/q4_0（`FA_QUANTS`） | §4.1 #13 |
| V100S | CC 7.0 无 BF16 TC/FP8 → FP16 路径 | §六 |
| 长上下文 | 262144 下 softmax 数值需 f32 中间量 | 推导 |

---

## 11. 最小参考实现（C++，与模块二一致）

```cpp
// 报告 §4.1 算子#12：GQA 6:1 causal，head=24, kv_head=4, head_dim=256
static const int N_HEAD = 24, N_HEAD_KV = 4, HEAD_DIM = 256, GQA_GROUP = N_HEAD / N_HEAD_KV; // =6
void gqa_sdpa_causal(const float* q, const float* k, const float* v, float* out, int T_q, int T_k) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);       // 1/sqrt(256)
    const int offset = T_k - T_q;
    std::vector<float> score(T_k);
    for (int i = 0; i < T_q; ++i)
      for (int h = 0; h < N_HEAD; ++h) {
        const int kvh = h / GQA_GROUP;                        // 6 个 Q 头共享 1 个 KV 头
        const float* qh = q + ((size_t)i*N_HEAD + h)*HEAD_DIM;
        const int nk = offset + i + 1;                        // 因果可见长度
        float mx = -1e30f;
        for (int j = 0; j < nk; ++j) {                        // QK^T
            const float* kj = k + ((size_t)j*N_HEAD_KV + kvh)*HEAD_DIM;
            float d = 0.f; for (int t = 0; t < HEAD_DIM; ++t) d += qh[t]*kj[t];
            score[j] = d*scale; if (score[j] > mx) mx = score[j];
        }
        float sum = 0.f;
        for (int j = 0; j < nk; ++j) { score[j] = expf(score[j]-mx); sum += score[j]; }
        float* oh = out + ((size_t)i*N_HEAD + h)*HEAD_DIM;
        for (int t = 0; t < HEAD_DIM; ++t) oh[t] = 0.f;
        for (int j = 0; j < nk; ++j) {                        // ×V
            const float p = score[j]/sum;
            const float* vj = v + ((size_t)j*N_HEAD_KV + kvh)*HEAD_DIM;
            for (int t = 0; t < HEAD_DIM; ++t) oh[t] += p*vj[t];
        }
      }
}
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 17 次、Flash-Attention/分块 | §4.1 #12 |
| 0/0/16/1/0 分布 | §9.3 |
| GQA 6:1、24/4、head_dim 256 | §2.1 |
| Q/K/V/O 投影张量 | §2.3 |
| 族归类、延迟关键路径 | §4.2 |
| V100S FA 支持 q8_0-q8_0 | §六 |
| 循环层不是 SDPA | §9.3 / §4.1 #7 |

---

## 13. 边界说明

- 报告未给出 SDPA 的**分块大小**、是否 sliding-window（仅记 causal + FA）。
- 未给出 attention logit 的额外 softcap/scale 参数（推导，不外推）。
- 未实现 DeltaNet 的注意力（属文档 07）；本算子在循环层出现次数为 **0**。
- GQA 头映射（$h/6$）为按报告 6:1 分组的自然推导。
