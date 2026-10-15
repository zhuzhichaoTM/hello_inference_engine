# 算子 15 — Concat

> 所属：Qwen3.8-27B 算子字典 · 第 **15 / 17** 项
> 唯一依据：报告 §4.1 #15、§9.3、§2.3、§4.2
> 本文档只描述报告中已记录的该算子，不引入报告外的模型设定、超参或张量。

---

## 1. 规范化摘要

| 项 | 值 | 依据 |
|---|---|---|
| 算子编号 | 15 | 本文档 |
| 算子名称 | Concat | §4.1 #15 |
| 算子族 | **前后处理** | §4.2 |
| 优化优先级 | ★ 可忽略 | §4.2 |
| 出现次数 | **1** | §4.1 #15 |
| 是否含权重 | **否**（权重在 `nextn.enorm/hnorm/eh_proj`） | §4.1 #15 |
| 是否延迟关键路径 | 否 | §4.2 |
| 作用（报告原文） | MTP 块拼接 `norm(embd) ‖ norm(hidden)` | §4.1 #15 |
| 对应权重/依据 | `nextn.enorm/hnorm/eh_proj` | §4.1 #15 |

---

## 2. 出现次数与层型分布

| 作用域 | 每层数 | 层数 | 小计 | 判定 |
|---|---:|---:|---:|---|
| MTP 专属 | 1 | 1 | **1** | 仅 blk.64 |
| 其余 | 0 | — | 0 | — |
| **合计** | | | **1** | ✔（§9.3 归入"Gather/Concat/Sampler = 3"全局行） |

---

## 3. 数学定义

沿特征维拼接两路归一化后的向量：

$$c = \big[\,\text{RMSNorm}_{enorm}(e)\;\big\|\;\text{RMSNorm}_{hnorm}(h)\,\big]\in\mathbb{R}^{10240}$$

$$e,h\in\mathbb{R}^{5120}\ \Rightarrow\ c\in\mathbb{R}^{10240}$$

随后由 `nextn.eh_proj` 投影回 5120：

$$c' = W_{eh}\,c,\qquad W_{eh}\in\mathbb{R}^{10240\times5120}\ \Rightarrow\ c'\in\mathbb{R}^{5120}$$

---

## 4. 张量签名（输入 / 输出）

| 项 | shape | 说明 |
|---|---|---|
| 输入 A（embedding） | $[T,5120]$ | `nextn.enorm` 归一化后的 token 嵌入 |
| 输入 B（hidden） | $[T,5120]$ | `nextn.hnorm` 归一化后的主干 hidden |
| 输出 | $[T,10240]$ | 拼接结果 → `nextn.eh_proj` |

> $10240 = 5120 + 5120$ ✔ 与 `nextn.eh_proj.weight [10240, 5120]` 的第一维吻合（§2.3）。

---

## 5. 报告内张量实例

| 张量 | shape | dtype | 数量 | 作用 |
|---|---|---|---|---|
| `nextn.enorm.weight` | [5120] | f32 | 1 | 输入 embedding 归一化 |
| `nextn.hnorm.weight` | [5120] | f32 | 1 | 主干 hidden 归一化 |
| `nextn.eh_proj.weight` | [**10240**, 5120] | q4_K | 1 | 拼接后投影 → 5120 |
| `nextn.shared_head_norm.weight` | [5120] | f32 | 1 | 共享头前归一化 |

> `nextn.eh_proj` 的 10240 是**拼接维度的直接证据**（§4.1 #15 记为"拼接 `norm(embd) ‖ norm(hidden)`"）。

---

## 6. 推理链路位置

```
主干 blk.0..63 → hidden h [T,5120]
token_embd     → embd   e [T,5120]

MTP 块 (blk.64):
  RMSNorm_enorm(e) ─┐
                    ├─► [Concat] → c [T,10240] → eh_proj GEMM → c' [T,5120]
  RMSNorm_hnorm(h) ─┘
        → 全注意力 11 张量（同 blk.3 结构）→ shared_head_norm → 共享 LM-Head
```

- **上游**：算子 02（`nextn.enorm`/`nextn.hnorm`）、算子 14（embedding）
- **下游**：算子 01（`nextn.eh_proj` GEMM）

---

## 7. Prefill / Decode 差异

| | Prefill | Decode |
|---|---|---|
| 形态 | $[T,10240]$ 拼接 | $[1,10240]$ |
| 成本 | 一次内存拷贝 | 一次拷贝 |
| 瓶颈 | 可忽略 | 可忽略 |

---

## 8. 实现要点与融合建议（§4.2）

- 报告族归类：前后处理（Gather/Concat/Sampler = 3），**★ 可忽略**。
- 要点：Concat 后立即接 `eh_proj` GEMM → **建议把 Concat 融合进 `eh_proj` GEMM 的 A 矩阵准备**（避免多一次 $[T,10240]$ 写读）。
- MTP 块的意义：报告 §2.1 `nextn_predict_layers=1`，是**投机解码草稿头**（§六）。

---

## 9. 性能特征

- FLOPs：0（拷贝）。
- 访存：读 $2\times[T,5120]$ + 写 $[T,10240]$，纯 bandwidth-bound。
- 节点占 1/1567，**可忽略**。

---

## 10. 数值精度与量化约束

| 项 | 说明 | 依据 |
|---|---|---|
| 无权重 | 不涉及 Q4_K/Q6_K | — |
| **MTP 块无 imatrix** | `blk.64.nextn.eh_proj` 及 blk.64 全部张量缺 imatrix | §5.2 |
| 质量影响 | MTP 草稿头量化误差偏大 → **直接影响投机解码接受率** | §六 |
| V100S | FP16 激活 | §六 |

---

## 11. 最小参考实现（伪代码）

```cpp
// 报告 §4.1 算子#15：MTP 块拼接 norm(embd) ‖ norm(hidden)，1 次；无权重
// enorm/hnorm 为 f32 RMSNorm(文档02)；输出 [T,10240] → eh_proj GEMM
void concat_embd_hidden(const float* e /*[T,5120]*/, const float* h /*[T,5120]*/,
                        float* c /*[T,10240]*/, int T, int H=5120) {
    for (int t = 0; t < T; ++t) {
        memcpy(c + (size_t)t*2*H,     e + (size_t)t*H, H*sizeof(float));  // 前半 = norm(embd)
        memcpy(c + (size_t)t*2*H + H, h + (size_t)t*H, H*sizeof(float));  // 后半 = norm(hidden)
    }
}
```

---

## 12. 依据引用

| 结论 | 报告定位 |
|---|---|
| 1 次、拼接 norm(embd)‖norm(hidden) | §4.1 #15 |
| 全局行分布 | §9.3 |
| `nextn.eh_proj [10240,5120]`、`enorm/hnorm/shared_head_norm [5120]` | §2.3 |
| `nextn_predict_layers=1`（MTP/投机解码草稿头） | §2.1 |
| blk.64 无 imatrix | §5.2 |
| MTP 影响投机解码接受率 | §六 |
| 族归类、可忽略 | §4.2 |

---

## 13. 边界说明

- 报告未给出 Concat 的**顺序**（embd 在前还是 hidden 在前）；本文档按 §4.1 #15 记法 `norm(embd) ‖ norm(hidden)` 排列。
- 未展开 MTP 块投机解码的**草稿验证/接受逻辑**（报告未覆盖，属引擎行为）。
- blk.64 的具体张量列表见文档 01/02/11/12/13 的 MTP 行。
