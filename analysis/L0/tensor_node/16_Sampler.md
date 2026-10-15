# 算子 16 — Sampler（Softmax + Top-K + Top-P + Temp）

> 所属：Qwen3.8-27B 算子字典 · 第 **16 / 17** 项
> 唯一依据：报告 §4.1 #16、§9.3、§2.2、§4.2
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。
> **定位：非权重算子**，仅作用于 LM-Head 输出的 logits。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 16 | 本文档 |
| 算子名称 | Sampler（Softmax + Top-K + Top-P + Temp） | §4.1 #16 |
| 算子族 | **前后处理** | §4.2 |
| 优化优先级 | ★ 可忽略 | §4.2 |
| 出现次数 | **1** | §4.1 #16 |
| 是否含权重 | **否**（报告原文：非权重算子） | §4.1 #16 |
| 是否延迟关键路径 | 否 | §4.2 |
| 作用（报告原文） | 采样，非权重算子 | §4.1 #16 |
| 对应依据 | `sampling.top_k=20, top_p=0.95` | §4.1 #16 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 判定 |
|---|---:|---:|---:|---|
| 全局 | 1 | 1 | **1** | 每步生成末端一次 |
| 其余 | 0 | — | 0 | — |
| **合计** | | | **1** | ✔（§9.3 归入"Gather/Concat/Sampler = 3"全局行） |

---

## 3. 数学定义

对 logits $z\in\mathbb{R}^{248320}$：

1. **温度**：$z \leftarrow z/\tau$，$\tau=1.0$（§2.2）
2. **Top-K**：保留概率最大的 $K=20$ 个（`top_k=20`）
3. **Top-P（核采样）**：在剩余候选中按累积概率截断，保留累积和 $\le p=0.95$ 的最小集合
4. **Softmax**：$p_i = \dfrac{\exp(z_i)}{\sum_j \exp(z_j)}$
5. **采样**：按 $p$ 抽样

---

## 4. 张量签名（输入 / 输出）

| 项 | shape | 说明 |
|---|---|---|
| 输入 | $[1, 248320]$（logits） | 来自 `output.weight` GEMM |
| 输出 | 标量（token id） | 反馈到下一轮 Embedding Gather（文档 14） |

- vocab = 248320（§2.1）。

---

## 5. 报告内张量实例

**无权重张量**。相关采样配置（报告 §2.2 / §4.1 #16）：

| 参数 | 值 | 依据 |
|---|---|---|
| temp | 1.0 | §2.2 |
| top_p | 0.95 | §2.2 |
| top_k | 20 | §2.2 |
| eos | 248046 | §2.1 |
| add_bos | false | §2.1 |

---

## 6. 推理链路位置

```
主干 blk.0..63 (→ blk.64 MTP) → output_norm(文档02) → output.weight LM-Head GEMM(文档01)
   → logits [1,248320] → [Sampler] → token id → 下一轮 Embedding Gather(文档14)
```

- **上游**：算子 01（LM-Head GEMM）、算子 02（`output_norm`）
- **下游**：算子 14（下一轮输入）
- 与 MTP 联动：MTP 块（blk.64）产出草稿 token，接受/拒绝由引擎控制（报告未覆盖细节）。

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | 通常只对**最后一个**位置采样 | 每 token 采样 |
| 成本 | 对 248320 个 logits 排序/筛选 | 同 |
| 瓶颈 | Top-K/Top-P 的排序 $O(V\log V)$，可优化为部分选择 | 相对 GEMM 可忽略 |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：前后处理（3 个），**★ 可忽略**。
- 要点：
  - 报告明确其为**非权重算子**，不参与 506/216 等权重统计。
  - Top-K(20) 极小 → 可用部分选择（partial sort）而非全排序。
  - Softmax 可与 LM-Head GEMM 的 epilogue 融合。

---

## 9. 性能特征

- FLOPs：Softmax $\approx 3V$（$V=248320$）≈ 0.75 MFLOPs；Top-K/P 为排序开销。
- 相对 52.1 GFLOPs/token 线性层，占比 **< 0.01%**。
- 访存：读 $[1,248320]$ logits（约 0.5 MB @ f16）→ 可忽略。

---

## 10. 数值精度与量化约束

| 项 | 说明 | 依据 |
|---|---|---|
| 无权重 | 不涉及量化、imatrix | §4.1 #16 |
| logits 来源 | `output.weight` 为 **q6_K**（且**无 imatrix**）→ logits 精度受其影响 | §3.1 / §5.2 |
| 数值 | Softmax 需 max-subtract 防溢出；建议 f32 | 推导 |
| V100S | FP16 前向，采样建议 f32 | §六 |

---

## 11. 最小参考实现（伪代码）

```cpp
// 报告 §4.1 算子#16：temp1.0 / top_p0.95 / top_k20；非权重算子
int sample(const float* logits /*[V=248320]*/, int V, float temp=1.0f, int top_k=20, float top_p=0.95f) {
    std::vector<int> idx(V); for (int i=0;i<V;++i) idx[i]=i;
    // Top-K：按 logits/temp 取前 top_k（部分选择）
    std::partial_sort(idx.begin(), idx.begin()+top_k, idx.end(),
        [&](int a,int b){ return logits[a]/temp > logits[b]/temp; });
    // Top-P：在 top_k 内按概率累积截断到 top_p
    float mx = logits[idx[0]]/temp, sum = 0.f;
    for (int i=0;i<top_k;++i) sum += expf(logits[idx[i]]/temp - mx);
    float cum = 0.f; int keep = 0;
    for (int i=0;i<top_k;++i) { cum += expf(logits[idx[i]]/temp - mx)/sum; ++keep; if (cum >= top_p) break; }
    // 采样
    float r = (float)rand()/RAND_MAX, acc = 0.f;
    for (int i=0;i<keep;++i) { acc += expf(logits[idx[i]]/temp - mx)/sum; if (r <= acc) return idx[i]; }
    return idx[keep-1];
}
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 1 次、非权重算子 | §4.1 #16 |
| 全局行分布 | §9.3 |
| `top_k=20`/`top_p=0.95`/temp 1.0 | §2.2 / §4.1 #16 |
| vocab 248320、eos 248046、add_bos=false | §2.1 |
| 族归类、可忽略 | §4.2 |
| `output.weight` q6_K 且无 imatrix | §3.1 / §5.2 |

---

## 13. 边界说明

- 报告只给出采样参数，**未**规定重复惩罚 / presence penalty / 随机种子等（不外推）。
- 未展开 MTP 投机解码的**草稿接受/拒绝算法**（报告 §六 仅指出其受 imatrix 缺失影响）。
- 不讨论 logits 后处理（如 logit_bias），报告未记录。
