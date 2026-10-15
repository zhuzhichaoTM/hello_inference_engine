# 算子 11 — RoPE（部分旋转，64/256 维，MRoPE 4 段）

> 所属：Qwen3.8-27B 算子字典 · 第 **11 / 17** 项
> 唯一依据：报告 §4.1 #11、§9.3、§2.1、§4.2
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。
> **核心约束：本模型是 MRoPE 4 段式「部分旋转」，禁止实现完整全维度 RoPE。**

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 11 | 本文档 |
| 算子名称 | RoPE（部分旋转，64/256 维，MRoPE 4 段） | §4.1 #11 |
| 算子族 | **序列算子** | §4.2 |
| 优化优先级 | ★★★ 延迟关键路径 | §4.2 |
| 出现次数 | **17** | §4.1 #11 |
| 是否含权重 | **否**（频率表由超参生成） | §4.1 #11 |
| 是否延迟关键路径 | 是 | §4.2 |
| 作用（报告原文） | 位置编码，仅作用于全注意力层 | §4.1 #11 |
| 对应依据 | `rope.dimension_sections=[11,11,10,0]` | §4.1 #11 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 判定 |
|---|---:|---:|---:|---|
| 全注意力专属 | 1 | 16 | **16** | 含 `attn_q`/`attn_k` |
| MTP 专属 | 1 | 1 | **1** | blk.64 全注意力结构 |
| 循环专属 | 0 | 48 | 0 | **DeltaNet 层无 RoPE** |
| 公共 / 全局 | 0 | — | 0 | — |
| **合计** | | | **17** | ✔ 与 §9.3 一致 |

---

## 3. 数学定义（部分旋转 + MRoPE 分段）

对 head 向量的**前 64 维**按频率对做二维旋转，**后 192 维原样透传**：

$$\begin{pmatrix} x'_{2p}\\ x'_{2p+1}\end{pmatrix}=\begin{pmatrix}\cos(\varphi_p) & -\sin(\varphi_p)\\ \sin(\varphi_p) & \cos(\varphi_p)\end{pmatrix}\begin{pmatrix} x_{2p}\\ x_{2p+1}\end{pmatrix},\qquad p\in[0,32)$$

$$\varphi_p = \text{pos}_{s(p)}\cdot \theta_p,\qquad \theta_p = \text{base}^{-2p/64},\qquad \text{base}=10^{7}$$

$$x'_i = x_i,\qquad i\in[64,256)$$

### MRoPE 4 段映射（sections 单位 = 频率对，每对 2 维）

| 段 $s$ | `sections[s]` | 频率对数 | 占用维度 | 位置源 |
|---|---:|---:|---|---|
| 0 | 11 | 11 | [0, 22) | 位置分量 0 |
| 1 | 11 | 11 | [22, 44) | 位置分量 1 |
| 2 | 10 | 10 | [44, 64) | 位置分量 2 |
| 3 | **0** | 0 | — | **空段，无贡献** |
| 合计 | **32** | 32 对 | **64 维** | 与 `dimension_count=64` 吻合 ✔ |

> $11+11+10+0=32$ 对 $=64$ 维，**严格等于** `rope.dimension_count=64`。报告 §2.1 记录 `tags=["image-text-to-text"]`，与 MRoPE 多模态定位吻合。

---

## 4. 张量签名（输入 / 输出）

| 项 | shape | 说明 |
|---|---|---|
| Q 输入/输出 | $[T,24,256]$ | 旋转前 64 维 |
| K 输入/输出 | $[T,4,256]$ | 旋转前 64 维 |
| 透传维 | $[T,\cdot,192]$ | 256 − 64，**不参与三角函数** |

- 频率表：32 个 $\theta_p$，由 `base=1e7`、`rope_dim=64` 生成（无持久权重）。

---

## 5. 报告内张量实例

**无权重张量**。RoPE 作用对象：

| 对象 | 来源 | shape |
|---|---|---|
| Q | `attn_q.weight` 前 6144 维 → 重排为 24 头 | [T,24,256] |
| K | `attn_k.weight` | [T,4,256] |

上游紧邻 QK-Norm（`attn_q_norm`/`attn_k_norm`，文档 02）。

---

## 6. 推理链路位置

```
全注意力层(blk=3,7,…,63) / MTP(blk.64):
  attn_q GEMM → Split → QK-Norm(文档02) → [MRoPE 部分旋转] ─┐
  attn_k GEMM →         QK-Norm(文档02) → [MRoPE 部分旋转] ─┤→ SDPA(文档12) → KV-Cache(文档13)
  attn_v GEMM →                                            ─┘
```

- **上游**：算子 02（QK-Norm）
- **下游**：算子 12（SDPA）、算子 13（KV-Cache 写 K）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | $[T,24,256]$ / $[T,4,256]$ 批量旋转 | 单 token，pos = 当前长度 |
| 成本 | $\propto T$（仅 64 维） | 固定（64 维 × 28 头） |
| 瓶颈 | 相对 SDPA 小 | 相对权重访存小 |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：序列算子（Conv1D/DeltaRule/SDPA/RoPE 共 130），**★★★ 延迟关键路径**。
- **关键实现约束**：
  1. **只旋转前 64 维**，后 192 维必须逐字节透传。
  2. **MRoPE 4 段**：不同频率对用**不同的位置索引**（sections 决定），不能全用同一 pos。
  3. `sections[3]=0` → 第 4 段为空，**无维度**。
  4. 通常与 SDPA 的 Q/K 准备融合。

---

## 9. 性能特征

- 旋转元素数：$64\times(24+4)=1792$ 元素/层（Q+K）。
- FLOPs：每对约 6（2 mul + 2 mul + 2 add）→ $32\times6\times28 = 5376$/层；16+1 层 → ≈ 91 KFLOPs/token，**可忽略**。
- 延迟敏感性：中（在注意力路径上，但计算极轻）。

---

## 10. 数值精度与量化约束

| 项 | 说明 |
|---|---|
| 无权重 | 不涉及量化、imatrix |
| 角度精度 | $\theta_p$ 与 $\cos/\sin$ 建议 f32 计算（长上下文 pos 大，FP16 精度不足） |
| V100S | FP16 激活；RoPE 前建议 f32 中间量（推导） |
| 上下文 | 262144，`base=1e7`（§2.1）→ 长上下文频率衰减需注意 |

---

## 11. 最小参考实现（C++，与模块二一致）

```cpp
// 报告 §4.1 算子#11：部分旋转 64/256，MRoPE sections=[11,11,10,0]，base=1e7
static const int ROPE_DIM = 64, HEAD_DIM = 256;
static const int SECTIONS[4] = {11,11,10,0};   // 单位：频率对
static const float ROPE_BASE = 1e7f;

void mrope_partial(float* x, int n_head, const int pos[4]) {
    static int section_of[ROPE_DIM/2];
    static bool init = false;
    if (!init) { int p = 0;
        for (int s = 0; s < 4; ++s) for (int k = 0; k < SECTIONS[s]; ++k) section_of[p++] = s; // 32
        init = true; }
    for (int h = 0; h < n_head; ++h) {
        float* head = x + (size_t)h * HEAD_DIM;
        for (int p = 0; p < ROPE_DIM/2; ++p) {                 // 仅前 32 对 = 64 维
            const float theta = powf(ROPE_BASE, -2.0f*p/(float)ROPE_DIM);
            const float ang = (float)pos[section_of[p]] * theta;   // MRoPE：按段取 pos
            const float c = cosf(ang), s_ = sinf(ang);
            const float a = head[2*p], b = head[2*p+1];
            head[2*p]     = a*c - b*s_;
            head[2*p+1]   = a*s_ + b*c;
        }
        // head[64..256) —— 192 维原样透传（不写代码即正确）
    }
}
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 17 次、仅全注意力层 | §4.1 #11 |
| 0/0/16/1/0 分布 | §9.3 |
| `rope.dimension_count=64`、`sections=[11,11,10,0]` | §2.1 |
| `freq_base=1e7`、上下文 262144 | §2.1 |
| 部分旋转（256 中仅 64） | §2.1 |
| 族归类、延迟关键路径 | §4.2 |

---

## 13. 边界说明

- 报告未规定 RoPE 的**配对布局**（交错 NEOX 式 `(2p,2p+1)` 还是半分裂式）；本文档取 NEOX 交错配对并标注。
- 报告未给出各段位置索引的具体生成方式（多模态下 temporal/height/width 如何映射），文本场景取 4 段同位。
- **禁止**实现全维度 RoPE：本模型仅 64 维旋转，192 维透传。
- 循环层（48 个）无 RoPE（§4.1 #11「仅作用于全注意力层」）。
