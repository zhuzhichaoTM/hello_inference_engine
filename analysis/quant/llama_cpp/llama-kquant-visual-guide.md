# K-Quant 算法图形化解说

> 基于 `ggml/src/ggml-quants.c` 的 `quantize_row_q4_K_impl`（Q4_K 为代表）逐段图形化解说。
> 覆盖：K 系列 block 结构、加权量化、二级 scale/min、打包布局，以及与 Q4_0 的对比。
>
> 阅读前建议先了解 "以 Q4_K 为代表的 k-quant 完整链路"。

---

## 目录

1. [K-Quant 家族一览](#1-k-quant-家族一览)
2. [Q4_K 的 block 内存结构](#2-q4_k-的-block-内存结构)
3. [总体流程：一行 256 元素如何变成 block](#3-总体流程一行-256-元素如何变成-block)
4. [Step 1：整体统计](#4-step-1整体统计)
5. [Step 2：逐 32 子块加权量化 make_qkx3_quants](#5-step-2逐-32-子块加权量化-make_qkx3_quants)
6. [Step 3：scale/min 二级量化 make_qp_quants](#6-step-3scalemin-二级量化-make_qp_quants)
7. [Step 4：打包 scales 到 12 字节](#7-step-4打包-scales-到-12-字节)
8. [Step 5：写入 4bit 权重 qs](#8-step-5写入-4bit-权重-qs)
9. [数字示例：一个 256 元素行完整走一遍](#9-数字示例一个-256-元素行完整走一遍)
10. [K-Quant vs Q4_0 对比图](#10-k-quant-vs-q4_0-对比图)
11. [反量化（dequantize）图形化](#11-反量化dequantize图形化)

---

## 1. K-Quant 家族一览

K 系列包括 Q2_K / Q3_K / Q4_K / Q5_K / Q6_K。它们的共同点：

```text
① 大 block：QK_K = 256 个元素（不是 32）
② 内部再分 8 个子块，每个 32 元素
③ scale 和 min 都做二级量化
④ 支持 imatrix 加权
差异只是"每个元素多少 bit + 是否分 tanh/对称"

Q2_K：2 bit/element + 16 个高精单元
Q3_K：3 bit/element + 高精单元
Q4_K：4 bit/element
Q5_K：5 bit/element（4bit + 1bit 高位）
Q6_K：6 bit/element（对称）
```

---

## 2. Q4_K 的 block 内存结构

```text
一个 block_q4_K 存储 256 个 float 权重         （原本 256×4 = 1024 字节）
压缩后固定大小 = 144 字节

┌─────────────────────────────────────────────────────────────────────┐
│  block_q4_K (144 bytes)                                              │
│                                                                     │
│  ┌────────┬──────────┬───────────────┬─────────────────────────────┐ │
│  │ d (2B) │ dmin(2B) │  scales[12]   │   qs[128]                  │ │
│  │ 主scale │ 主min   │ 二级量化值     │  256 个 4bit 权重          │ │
│  │ fp16   │ fp16    │ (见§7 布局)    │  每 32 元素 = 16 字节       │ │
│  └────────┴──────────┴───────────────┴─────────────────────────────┘ │
│      4B          12B                   128B = 144B                 │
└─────────────────────────────────────────────────────────────────────┘

压缩比 = 1024/144 ≈ 7.1x  → 对应 4.58G（8B 模型）
```

### 内部 8 个子块

```text
256 个元素（索引 0..255）
  ├── 子块 0: 元素 0..31
  ├── 子块 1: 元素 32..63
  ├── 子块 2: 元素 64..95
  ├── ...
  └── 子块 7: 元素 224..255

每个子块 32 个元素 → 共享一个子 scale[j] + 一个子 min[j]
```

---

## 3. 总体流程：一行 256 元素如何变成 block

```text
输入：一行 256 个 float（n_per_row = 256）
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1  整体统计                                           │
│         sum_x2 = Σx²  →  sigma2 = 2*sum_x2/256             │
│         av_x = sqrt(sigma2)（整体幅值参考）                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2  逐子块加权量化（8 次，j=0..7）                     │
│     ┌──────────────────────────────────────────────────┐  │
│     │ 取子块 x[32j..32j+31] + 权重 weights              │  │
│     │  （有 imatrix → 用重要性矩阵×幅值                  │  │
│     │    无 → av_x + |x| 启发式）                       │  │
│     │                                                  │  │
│     │ make_qkx3_quants(32, 15, x, weights, L, mins)    │  │
│     │   → 迭代找最优 scale[j] 和 min[j]                │  │
│     │   → 每个元素量化成 4bit 存入 L[32j..32j+31]      │  │
│     │   → 记录子块权重和 sw[j]                         │  │
│     └──────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3  scale/min 二级量化（8 个子块 → 6bit 缩放）          │
│         d_block = make_qp_quants(8, 63, scales, Ls, sw)    │
│         m_block = make_qp_quants(8, 63, mins,   Lm, sw)    │
│           → 压缩成 Ls[8] / Lm[8]（各 6bit）                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4  打包：scales[12] 字节布局（见 §7）                  │
│         d = fp16(d_block); dmin = fp16(m_block)             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5  写入 qs[128]：L[0..255] 每 2 个元素 1 字节           │
│         （4bit 低位 + 4bit 高位）                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Step 1：整体统计

```c
float sum_x2 = 0;
for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];   // Σx²
float sigma2 = 2*sum_x2/QK_K;   // 2×均值方差
float av_x = sqrtf(sigma2);     // RMS 幅值
```

```text
示例数值（简化）：
  x = [0.1, 0.4, -0.2, 0.8, ..., 1.5]  256 个值
  sum_x2 ≈ 300
  sigma2 = 2*300/256 ≈ 2.34
  av_x ≈ 1.53

用途：
  这个 av_x 只是"无 imatrix 时的权重参考"，
  让大值获得更大权重，量化误差按重要性分配。
```

---

## 5. Step 2：逐 32 子块加权量化 make_qkx3_quants

### 5.1 权重计算

```c
for (int j = 0; j < QK_K/32; ++j) {          // 8 次
    if (quant_weights) {                     // 有 imatrix
        const float * qw = quant_weights + QK_K*i + 32*j;
        for (int l = 0; l < 32; ++l)
            weights[l] = qw[l] * sqrtf(sigma2 + x[32*j+l]*x[32*j+l]);
    } else {                                 // 无 imatrix 启发式
        for (int l = 0; l < 32; ++l)
            weights[l] = av_x + fabsf(x[32*j+l]);
    }
    float sumw = 0;
    for (int l = 0; l < 32; ++l) sumw += weights[l];
    sw[j] = sumw;                            // 记录权重和
    scales[j] = make_qkx3_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
}
```

```text
图形化（子块 j=0，32 元素）：

  子块元素:  x0  x1  x2 ... x31
  权重(有im): qw0×A0 qw1×A1 ...   （A = sqrt(sigma2+x²)）
  权重(无im): av+|x0| av+|x1| ...

  目标：找一个 scale d 和 min m，使：
        Σ weights[l] × (x[l] - (m + d×q[l]))²  最小
        其中 q[l] ∈ {0..15}（4bit）

  大权重（重要元素）→ 误差惩罚大 → 反而量化得更准
```

### 5.2 make_qkx3_quants 迭代求解

```c
static float make_qkx3_quants(int n, int nmax, const float * x,
        const float * weights, uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
```

`Q4_K` 调用它时使用 `n=32`、`nmax=15`，也就是对一个 32 元素子块产生 16 个离散量化级别 `L[i] ∈ [0,15]`。

#### 5.2.1 先明确优化目标

量化不是简单地把最大值和最小值等分，而是在寻找一个**非对称线性模型**：

```text
x[i] ≈ min + scale × L[i]

其中：
  x[i]       = 原始浮点权重
  L[i]       = 4bit 整数，0..15
  scale      = 子块步长
  min        = 子块偏移（通常 ≤ 0）
```

算法要最小化加权误差：

```text
E(scale, min, L)
  = Σᵢ w[i] × (min + scale×L[i] - x[i])²
```

- `w[i]` 越大，该元素越重要，误差惩罚越大；
- `w[i]` 越小，该元素允许有更大量化误差；
- `L[i]` 是离散变量，`scale/min` 是连续变量，因此采用交替优化。

#### 5.2.2 第一段：扫描范围与加权统计（函数前半段）

```c
float min = x[0];
float max = x[0];
float sum_w = weights ? weights[0] : x[0]*x[0];
float sum_x = sum_w * x[0];

for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    float w = weights ? weights[i] : x[i]*x[i];
    sum_w += w;
    sum_x += w * x[i];
}
```

逐步含义：

```text
min/max  → 确定初始量化覆盖范围
sum_w    = Σ w[i]
sum_x    = Σ w[i]x[i]
```

若没有 imatrix，调用方已经传入启发式权重；这里的 fallback `x[i]^2` 是该通用 helper 在无显式权重时的默认策略。

```c
if (min > 0) {
    min = 0;
}
```

如果整个子块都是正数，将 min 限制为 0，使反量化表达式的起点不向正方向移动。

```c
if (max <= min) {
    memset(L, 0, n);
    *the_min = -min;
    return 0.f;
}
```

处理全零/退化范围：所有量化码写 0，scale 返回 0，避免除零。

#### 5.2.3 第二段：生成初始量化解

```c
float iscale = nmax/(max - min);
float scale = 1/iscale;
float best_mad = 0;
```

这里：

```text
iscale = 15 / (max-min)   // 原值 → 量化码的反向比例
scale  = (max-min) / 15   // 量化码 → 原值的步长
```

然后把每个浮点数映射到 0..15：

```c
for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale*(x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff*diff;
    float w = weights ? weights[i] : x[i]*x[i];
    best_mad += w * diff;
}
```

每行作用：

```text
l = round((x[i]-min)/(max-min)×15)
  → 把 x 映射到最近的量化级别

clamp 0..15
  → 防止浮点边界导致超出 4bit 范围

diff = min + scale×L[i] - x[i]
  → 计算反量化值与原值的误差

best_mad += w×error
  → 计算当前初始解的加权误差
```

变量名 `best_mad` 是历史命名；在 `use_mad=false` 时实际累计的是**加权平方误差**，不是绝对误差。

#### 5.2.4 第三段：搜索候选 scale

```c
for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta*is + nmax)/(max - min);
```

Q4_K 调用参数为：

```text
rmin   = -0.9
rdelta = 0.05
nstep  = 36
```

因此算法不会只使用一个初始 scale，而会在一系列相邻范围上尝试候选量化网格。`rmin/rdelta/nstep` 控制搜索范围和精度。

候选量化码生成：

```c
int l = nearest_int(iscale*(x[i] - min));
l = MAX(0, MIN(nmax, l));
Laux[i] = l;
```

这一步只产生临时 `Laux`，不立即覆盖当前最优 `L`。

#### 5.2.5 第四段：固定 Laux，解析求最优 scale/min

```c
float sum_l = 0, sum_l2 = 0, sum_xl = 0;
for (int i = 0; i < n; ++i) {
    float w = weights ? weights[i] : x[i]*x[i];
    sum_l  += w*l;
    sum_l2 += w*l*l;
    sum_xl += w*l*x[i];
}
```

得到：

```text
sum_l  = Σ w[i]L[i]
sum_l2 = Σ w[i]L[i]²
sum_xl = Σ w[i]L[i]x[i]
```

```c
float D = sum_w * sum_l2 - sum_l * sum_l;
if (D > 0) {
    float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
    float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
```

这是加权最小二乘的闭式解。固定离散量化码 `Laux` 后，求解：

```text
min Σ w[i] × (this_min + this_scale×Laux[i] - x[i])²
```

`D` 是 2×2 正规方程的行列式：

```text
D = (Σw)(ΣwL²) - (ΣwL)²
```

`D <= 0` 表示当前量化码没有足够的变化（例如全都映射到同一值），无法稳定求解 scale/min，因此跳过该候选。

#### 5.2.6 为什么要限制 this_min ≤ 0？

```c
if (this_min > 0) {
    this_min = 0;
    this_scale = sum_xl / sum_l2;
}
```

量化表达式希望用非正 min 表示以 0 为参考的范围。若无约束最小二乘解出了正偏移，就固定 `min=0`，重新求只有 scale 的最小二乘解：

```text
this_scale = Σ w[i]L[i]x[i] / Σ w[i]L[i]²
```

这样既避免不符合 block 解码约定的正 min，也保持误差尽量小。

#### 5.2.7 第五段：候选误差比较与提交

```c
float mad = 0;
for (int i = 0; i < n; ++i) {
    float diff = this_scale * Laux[i] + this_min - x[i];
    diff = use_mad ? fabsf(diff) : diff*diff;
    float w = weights ? weights[i] : x[i]*x[i];
    mad += w * diff;
}
```

用候选 `this_scale/this_min/Laux` 重新计算加权误差。

```c
if (mad < best_mad) {
    for (int i = 0; i < n; ++i) {
        L[i] = Laux[i];
    }
    best_mad = mad;
    scale = this_scale;
    min = this_min;
}
```

只有候选误差严格小于当前最优值时才提交：

```text
Laux       → L
this_scale → scale
this_min   → min
best_mad   → mad
```

这保证搜索过程不会让结果变差。

#### 5.2.8 返回结果

```c
*the_min = -min;
return scale;
```

注意这里的接口约定：函数内部使用 `min`，返回时输出的是 `-min`。后续 Q4_K block 用它填充 `mins[j]`，再参与 dmin 二级量化。

### 5.3 用一个极简例子理解“加权”

假设只有 4 个数、2bit（示意，不是 Q4_K 的真实参数）：

```text
原始 x = [0.10, 0.40, 0.80, 1.00]
量化级别 q = [0, 1, 2, 3]
```

#### 不同重要性

```text
权重 w = [1, 1, 1, 10]
                              ↑ 最后一个值最重要
```

优化目标：

```text
E = 1×误差0² + 1×误差1² + 1×误差2² + 10×误差3²
```

算法会倾向于选择更合适的 `scale/min`，让最后一个重要值接近原值，即使牺牲前面某些不重要值一点误差。

```text
没有权重：误差平均分配
有权重：   误差优先分配给低重要性元素
```

这不是让某个元素使用更多 bit，而是通过调整整个子块的量化网格，让重要元素落在更准确的位置。

### 5.4 imatrix 权重如何进入 Q4_K

在 `quantize_row_q4_K_impl` 中：

```c
if (quant_weights) {
    const float * qw = quant_weights + QK_K*i + 32*j;
    for (int l = 0; l < 32; ++l) {
        weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
    }
}
```

数据流：

```text
imatrix importance
  × sqrt(sigma2 + x²)
  → weights[l]
  → make_qkx3_quants
  → 加权误差最小化
```

因此：

```text
重要性高的通道 → w 高 → 量化误差惩罚大 → 网格优先保护
重要性低的通道 → w 低 → 允许更大误差
```

### 5.5 算法本质：交替优化，而不是“先算 scale 再结束”

```text
给定 scale/min → 求离散 L（round + clamp）
给定 L         → 求连续 scale/min（加权最小二乘闭式解）
给定新 scale/min→ 再求新 L
重复搜索并保留误差最小方案
```

因此它同时处理了：

- 4bit 离散码的取整误差；
- 非对称范围的 min 偏移；
- 不同通道的相对重要性；
- scale/min 本身的最佳拟合。

### 5.6 子块量化完整流程图

```text
32 个 float x[0..31]
  │
  ├─ 扫描 min/max
  ├─ 构造 weights（imatrix 或启发式）
  │
  ▼
初始网格：min + scale×[0..15]
  │
  ▼
候选循环 is=0..36
  │  ├─ 用候选 iscale 生成 Laux[0..31]
  │  ├─ 计算 Σw、ΣwL、ΣwL²、ΣwLx
  │  ├─ 闭式求 this_scale/this_min
  │  ├─ 计算加权误差
  │  └─ 误差更小 → 提交 L/scale/min
  │
  ▼
输出：
  L[32]        每个 0..15（4bit）
  scale        子块 scale
  the_min      子块 min（以 -min 输出）
```

### 5.7 数学结论

`make_qkx3_quants` 求解的目标是：

```text
min_{L, scale, min}
    Σᵢ wᵢ (min + scale·Lᵢ − xᵢ)²

约束：
    Lᵢ ∈ {0, 1, ..., 15}
    min ≤ 0
    scale > 0（正常非退化块）
```

它采用“离散 `L` + 连续 `scale/min`”的交替搜索近似求解。完成后，Q4_K 上层再将 8 个子块的 scale/min 做二级量化，最终打包进 `block_q4_K`。

---

## 6. Step 3：scale/min 二级量化 make_qp_quants

```c
float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);
```

```text
输入：8 个子块的 scale[0..7]（或 mins[0..7]）
     8 个子块的权重和 sw[0..7]
输出：二级缩放因子 + Ls[0..7]（或 Lm[0..7]）

原理：8 个 scale 再次共享一个"超 scale" d_p，
      每个 scale 量化成 6bit (0..63)
      Ls[j] = round((scale[j]) / d_p)
```

```text
图形化：

  scale[0] scale[1] ... scale[7]        （8 个原始子块 scale）
     │       │            │
     └───────┴───┬────────┘
                 ▼
      ┌──────────────────────┐
      │  make_qp_quants      │
      │  找公共 d_p          │
      │  每个 scale → 6bit   │
      └──────────┬───────────┘
                 ▼
      d_p（超 scale）  +  Ls[0..7]（各 6bit）
```

**为什么二级量化**：直接存 8 个 fp16 scale = 16 字节，太贵；二级量化后：

```text
d_p(约若干bit) + 8×(6bit) ≈ 4-5 字节  → 大幅压缩
```

这是 k-quant 能在"尽量小空间"内保持 scale 精度的关键。

---

## 7. Step 4：打包 scales 到 12 字节

```c
for (int j = 0; j < QK_K/32; ++j) {          // j = 0..7
    uint8_t ls = Ls[j];                       // 二级 scale（≤63）
    uint8_t lm = Lm[j];                       // 二级 min（≤63）
    if (j < 4) {
        y[i].scales[j]   = ls;                // 子块 0-3 scale → 字节 0-3
        y[i].scales[j+4] = lm;                // 子块 0-3 min   → 字节 4-7
    } else {
        // 子块 4-7：把 ls/lm 的低 4bit 合并，高 2bit 进位到前面
        y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);   // 字节 8-11 的低/高半
        y[i].scales[j-4] |= ((ls >> 4) << 6);                // 高位进位到字节 0-3
        y[i].scales[j-0] |= ((lm >> 4) << 6);                // 高位进位到字节 4-7
    }
}
```

```text
scales[12] 布局：

 字节0:  scale0(6bit) | scale4高位(2bit)
 字节1:  scale1(6bit) | scale5高位(2bit)
 字节2:  scale2(6bit) | scale6高位(2bit)
 字节3:  scale3(6bit) | scale7高位(2bit)
 字节4:  min0(6bit)  | min4高位(2bit)
 字节5:  min1(6bit)  | min5高位(2bit)
 字节6:  min2(6bit)  | min6高位(2bit)
 字节7:  min3(6bit)  | min7高位(2bit)
 字节8:  scale4低4bit | min4低4bit
 字节9:  scale5低4bit | min5低4bit
 字节10: scale6低4bit | min6低4bit
 字节11: scale7低4bit | min7低4bit

即：6bit 的 scale/min 拆成 [低4bit 存字节8-11] + [高2bit 进位字节0-7]
```

```text
为什么这么设计：
  ① 只用 12 字节存下 8×scale + 8×min（各 6bit）
  ② 高 2bit 放低位区，方便反量化时统一还原
  ③ 这是 Q4_K 独有的"紧凑打包"
```

---

## 8. Step 5：写入 4bit 权重 qs

```c
y[i].d    = GGML_FP32_TO_FP16(d_block);
y[i].dmin = GGML_FP32_TO_FP16(m_block);

// L[0..255] 每个元素 4bit
// 每 2 个元素打包 1 字节：低位元素在前
for (int j = 0; j < 128; ++j) {
    y[i].qs[j] = L[2*j] | (L[2*j+1] << 4);
}
```

```text
qs[128] 布局：

 元素 0 元素 1  → 字节0 = [q1(4bit高)] [q0(4bit低)]
 元素 2 元素 3  → 字节1
 ...
 元素 254 元素 255 → 字节127

q ∈ {0..15}
```

---

## 9. 数字示例：一个 256 元素行完整走一遍

以"一行 256 元素 = 一个 Q4_K block"为例，简化数字。

### 输入（示意）

```text
x[0..255] ：
  子块0: [0.10, 0.35, -0.20, 0.80, ...]（32 个）
  子块1: [1.20, -0.50, ...]
  ...
  子块7: [...]
```

### Step 1 整体统计

```text
sum_x2 ≈ 300 → sigma2 ≈ 2.34 → av_x ≈ 1.53
```

### Step 2 子块 0 加权量化

```text
假设无 imatrix：weights[0..31] = 1.53 + |x|

x = [0.10, 0.35, -0.20, 0.80, ...]
w = [1.63, 1.88,  1.73, 2.33, ...]

make_qkx3_quants 迭代得：
  scale[0] ≈ 0.12，min[0] ≈ 0.05
  量化:
    0.10 → q=1  (0.05+0.12×1=0.17)
    0.35 → q=3  (0.05+0.12×3=0.41)
   -0.20 → q=0  (0.05+0.12×0=0.05)  误差-0.25（min 兜底到 0.05）
    0.80 → q=6  (0.05+0.12×6=0.77)
  ...
  L[0..31] = [1,3,0,6,...]
```

**注意**：`min` 的存在使量化支持非对称（负值也能表示），与 Q4_0 的对称量化不同。

### Step 3 二级量化

```text
scales = [0.12, 0.15, 0.09, 0.13, ...]（8 个）
mins   = [0.05, -0.02, 0.08, 0.01, ...]

make_qp_quants：
  d_block(super scale) ≈ 0.015
  Ls = round(0.12/0.015)=8, round(0.15/0.015)=10, ...
  m_block(super min)   ≈ 0.010
  Lm = round(0.05/0.010)=5, ...
```

### Step 4 打包

```text
scales[12] 按 §7 布局填充
d = fp16(0.015...)
dmin = fp16(0.010...)
```

### Step 5 写 qs

```text
L = [1,3,0,6, ...(256个)]
qs = [(1)|(3<<4)=0x31, (0)|(6<<4)=0x60, ...]  (128 字节)
```

---

## 10. K-Quant vs Q4_0 对比图

```text
┌─────────────────────────────────────────────────────────────────┐
│  Q4_0（block=32，对称）                                         │
│                                                                 │
│  ┌──────────────────────┐ ┌──────┐                              │
│  │ 32×4bit 权重 (16B)   │ │scale │ (2B)                        │
│  └──────────────────────┘ └──────┘                              │
│  每 32 元素 1 个 fp16 scale，无 min（对称）                      │
│  简单、kernel 快、精度略低                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Q4_K（block=256，非对称 + 二级）                               │
│                                                                 │
│  ┌────┬────┬───────┬──────────────────────────────┐             │
│  │d   │dmin│scales│qs (256×4bit=128B)             │             │
│  │2B  │2B  │12B   │                               │             │
│  └────┴────┴───────┴──────────────────────────────┘             │
│                                                                 │
│  8 个 32 元素子块，各自 scale/min                               │
│  scale/min 二次量化（6bit） → 12 字节                           │
│  支持 imatrix 加权 → 精度更高                                  │
└─────────────────────────────────────────────────────────────────┘

主要差异总结：
┌──────────┬──────────────┬──────────────────────────┐
│ 项目      │ Q4_0         │ Q4_K                       │
├──────────┼──────────────┼──────────────────────────┤
│ block 大小│ 32           │ 256（8×32 子块）            │
│ scale    │ 每块 1 fp16  │ 二级量化 + 超 scale         │
│ min      │ 无（对称）    │ 有 dmin（非对称）           │
│ 加权     │ 无           │ 有（含 imatrix）           │
│ 精度      │ 较低         │ 同 bit 更高               │
│ kernel    │ 简单快       │ 复杂                      │
│ 端侧推荐  │ 内存紧张用   │ 主力（Q4_K_M）            │
└──────────┴──────────────┴──────────────────────────┘
```

---

## 11. 反量化（dequantize）图形化

运行时读取 block_q4_K 还原 float：

```c
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k) {
    ...
    for (int i = 0; i < nb; i++) {
        // 还原 8 个子块的 scale 和 min（从 12 字节逆向解）
        const float d  = GGML_FP16_TO_FP32(x[i].d);
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin);
        for (int j = 0; j < QK_K/32; ++j) {
            float dl = d * ...;   // 子块 scale = d × 二级值
            float ml = dmin * ...;// 子块 min = dmin × 二级值

            for (int l = 0; l < 32; ++l) {
                // 从 qs 读出 4bit，12bit 有符号扩展
                y[...] = dl * ((int8_t)(q << 4) >> 4) + ml;
            }
        }
    }
}
```

```text
还原公式：
  value ≈ ml + dl × (有符号扩展的 4bit)

例如：
  4bit q=6, dl=0.12, ml=0.10
  → value ≈ 0.10 + 0.12×6 = 0.82

（实际有符号扩展，范围 -8..7 或类似）
```

```text
图形化（一子块反量化）：

  qs[ ]:  [4bit][4bit][4bit]...
             │     │     │
             ▼     ▼     ▼
        有符号扩展 +12bit
             │     │     │
             ▼     ▼     ▼
         × dl   × dl  × dl      （子块 scale）
             │     │     │
             ▼     ▼     ▼
          + ml   + ml  + ml      （子块 min）
             │     │     │
             ▼     ▼     ▼
          近似 float 值（用于矩阵乘）
```

---

## 一句话总结

> K-Quant 算法（以 Q4_K 为代表）把一个 256 元素的大 block 拆成 8 个 32 元素子块，先用**加权最小二乘迭代**（`make_qkx3_quants`，支持 imatrix）为每个子块求最优非对称 scale/min，再用**二级量化**（`make_qp_quants`）把 8 组 scale/min 压缩到 6bit 并紧凑打包进 12 字节，最后把 4bit 权重写入 qs。相比 Q4_0 的"32 元素单 scale"，它用"大 block + 二级 scale/min + 加权"换取了同 bit 下明显更高的精度，这也是端侧默认选 Q4_K_M 只因要多付出很小容量的根本原因。