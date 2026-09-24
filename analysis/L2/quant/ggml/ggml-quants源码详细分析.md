# ggml-quants 源码详细分析文档

> 逐函数剖析 `ggml/src/ggml-quants.c`（5667 行）与 `ggml/src/ggml-cpu/quants.c`（1339 行），
> 并解析 `ggml/src/ggml-common.h` 中的 block 结构与查找表。
> 系统性设计视角见 [03-量化完整功能设计文档.md](03-量化完整功能设计文档.md)。
> 本文聚焦**代码细节**：每个函数的算法、位运算技巧、边界处理与设计取舍。

---

## 目录

1. [代码地图与函数索引](#一代码地图与函数索引)
2. [基础工具函数逐函数分析](#二基础工具函数逐函数分析)
3. [查找表与常量数据结构](#三查找表与常量数据结构)
4. [经典量化族](#四经典量化族q1_0--q8_1--mxfp4--nvfp4)
5. [K-quant 超块族](#五k-quant-超块族)
6. [三值量化族](#六三值量化族tq1_0--tq2_0)
7. [非线性量化族](#七非线性量化族iq4_nl--iq4_xs)
8. [IQ 格点量化族](#八iq-格点量化族)
9. [q8_K 与点积伙伴](#九q8_k-与点积伙伴)
10. [数据验证](#十数据验证)
11. [CPU 侧实现与 vec_dot](#十一cpu-侧实现quantsc-与-vec_dot)
12. [源码技巧与陷阱总结](#十二源码技巧与陷阱总结)

---

## 一、代码地图与函数索引

### 1.1 文件顶部宏定义（`ggml-quants.c:1-26`）

```c
1   #define GGML_COMMON_IMPL_C          // 触发 ggml-common.h 的查找表实体定义
2   #include "ggml-common.h"
4   #include "ggml-quants.h"
5   #include "ggml-impl.h"
6   #include "ggml-cpu/ggml-cpu-impl.h"
7   #include "ggml-cpu.h"
20  #define GROUP_MAX_EPS 1e-15f        // 通用"全零块"判定门限
21  #define GROUP_MAX_EPS_IQ3_XXS 1e-8f // IQ 系列门限各不相同
22  #define GROUP_MAX_EPS_IQ2_S 1e-8f
23  #define GROUP_MAX_EPS_IQ1_M 1e-7f
24  #define GROUP_MAX_EPS_IQ1_S 1e-12f  // IQ1_S 反而最严格
26  #define UNUSED GGML_UNUSED
```

**为什么每个 IQ 类型有不同的 `GROUP_MAX_EPS`？** 该门限用于判断"子块是否全零"
（典型用法见 `ggml-quants.c:3364-3368`：`if (max < GROUP_MAX_EPS) { scales[ib] = 0; ... }`）。
门限与量化值幅值尺度必须匹配：IQ3_XXS 的格点坐标是奇数 `{1,3,5,7}`，归一化后 `max` 天然偏大，
用 `1e-15` 会过早把有效块判为零；IQ1_S 的门限极严（`1e-12`）是因为其量化值仅 ±1，
任何微小非零值都需正确编码，否则整块缩放因子失真。

`GGML_COMMON_IMPL_C` 的作用是让 `ggml-common.h` 展开查找表的**实体定义**
（`GGML_TABLE_BEGIN/END` 在 `ggml-common.h:472-473` 定义为 `static const type name[size] = {`），
而其他翻译单元只包含声明版本（`GGML_COMMON_DECL_C`），避免重复定义。

### 1.3 三段式函数命名约定

| 后缀 | 语义 | 是否使用 imatrix | 确定性 | 出现位置 |
|------|------|-----------------|--------|---------|
| `_ref` | 参考实现 | ❌ | 强确定性（文件生成权威路径） | `ggml-quants.c` |
| `_impl` | 高精度实现 | ✅ `quant_weights` | 依赖 imatrix | `ggml-quants.c` |
| 无后缀 | SIMD/分发版 | 视类型 | 需与 `_ref` 数值一致 | `ggml-cpu/quants.c`、`arch/*` |
| `dequantize_row_*` | 反量化 | — | — | `ggml-quants.c` |
| `quantize_<type>` | 行/多行批量入口 | 透传 | — | `ggml-quants.c` |

---

## 二、基础工具函数逐函数分析

### 2.1 `best_index_int8` —— 非线性电平的二分查找（`ggml-quants.c:28-37`）

```c
static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}
```

**算法**：在单调递增的电平表 `val[0..n-1]` 上找与 `x` 最接近的电平下标。

**关键细节**：

1. **两端的提前返回**（29-30 行）不止是优化，更是**正确性保证**：当 `x` 超出 `val` 范围时，
   二分循环会把 `mu-ml` 收敛到 1，得到 `ml=0, mu=1`，最后一行比较的是 `val[0]` 与 `val[1]`——
   这在逻辑上正确，但提前返回避免了对越界语义的依赖。
2. **浮点误差不影响结果**：比较用 `x - val[mu-1] < val[mu] - x` 而非 `fabsf`，
   两侧都是非负量（因 `val[mu-1] <= x <= val[mu]`），省去一次 `fabsf`。
3. **调用频率**：在 `quantize_row_iq4_nl_impl` 中最坏情况被调用 `(2*ntry+2) × block_size` 次
   （`ntry=7` 时为 16 次 × 32 = 512 次/子块）。二分（4 次比较）相比线性扫描（16 次）收益约 4 倍。
4. **要求 `val` 单调**：`kvalues_iq4nl`（`ggml-common.h:1120-1124`）确实是单调递增的：
   ```c
   -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
   ```
   这些值近似高斯分布的最优量化电平（非等间距，尾部稀疏、中心密集）。

### 2.2 `nearest_int` —— 位技巧的浮点取整（`ggml-quants.c:621-626`）

```c
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
```

这是整个文件中**最精巧也最需要小心**的函数，是全文件调用最频繁的原语。

**原理拆解**：

- `12582912.0f = 1.5 × 2^23`。IEEE-754 单精度有 23 位尾数，因此对 `|fval| ≤ 2^22` 的范围，
  `fval + 1.5×2^23` 的舍入（默认 round-to-nearest-even）恰好等价于对 `fval` 取整。
- 结果的整数部分编码为 `1.5×2^23 + round(fval)`，其位表示为 `0x4B400000 + round(fval)`。
- `i & 0x007fffff` 取出低 23 位尾数，`- 0x00400000`（即减去 2^22）还原出有符号整数。

**为什么断言上界是 4194303 = 2^22 - 1？** 超过该值时 `fval + 1.5×2^23` 的尾数溢出到指数位，
结果不再正确。下界同理受 `-2^22` 限制。断言是 debug 期的护栏。

**为什么用这个技巧而不用 `roundf`？** 在 x86 上，`roundf` 需 `SSE4.1 roundss` 指令且遵循当前舍入模式；
该位技巧只用一次浮点加法 + 一次 `memcpy`（编译器优化后为整数再解释），
在早期硬件上更快，且**不依赖 FENV 舍入模式**——这对保证跨平台量化结果的逐位一致性至关重要
（文件头注释明确要求 "deterministic creation of model files"）。

> **注意**：`memcpy` 而非指针类型双关（`*(int*)&val`）是刻意的：后者违反严格别名规则（strict aliasing），
> GCC/Clang 在 `-O2` 下可能产生错误优化。`memcpy` 是标准保证的、可被优化为零开销的写法。

### 2.3 `make_qx_quants` —— 对称型 RMSE 优化量化（`ggml-quants.c:628-695`）

用于 `q3_K`、`q6_K` 的**对称**（无 min）子块量化。签名：

```c
static float make_qx_quants(int n, int nmax, const float * x, int8_t * L, int rmse_type, const float * qw)
```

**第 1 步：找绝对最大并记录符号（632-641）**

```c
for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) { amax = ax; max = x[i]; }   // 注意：保留带符号的 max
}
if (amax < GROUP_MAX_EPS) { memset L=0; return 0.f; }  // 全零块短路
```

保留 `max` 的**符号**是关键：`iscale = -nmax / max`（第 642 行），
符号被吸收进 `iscale`，因此量化值 `L[i] = l + nmax` 为非负（`int8_t` 存 0..2*nmax），
反量化时再乘回带符号的 `scale`。这使 `L` 可以用无符号语义打包进位域。

**第 2 步：`rmse_type == 0` 的快速路径（643-649）**

```c
float iscale = -nmax / max;
if (rmse_type == 0) {
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
    }
    return 1/iscale;
}
```

注意 clamp 的**不对称性**：`MIN(nmax-1, l)` 而非 `MIN(nmax, l)`。
原因：索引是 `nmax + l` 且 `L` 用有符号 8 位存储，`l = nmax` 会得到 `L = 2*nmax`，
后续打包到 k-bit 域时溢出。实践中 `l = nmax` 只在 `x[i] == -max` 且恰好舍入到边界时出现，属边际情况。

**第 3 步：负 `rmse_type` 的"提前返回"语义（650-654）**

```c
bool return_early = false;
if (rmse_type < 0) {
    rmse_type = -rmse_type;
    return_early = true;
}
```

负数是一个**标志位**编码：`-1` 表示"用权重 `x²`，但跳过第 4 步的搜索，走加权平均返回"。
`return_early` 的返回公式（671 行）`0.5f*(scale + 1/iscale)` 是**初始尺度与最小二乘尺度的中点**，
是一种廉价折衷。

**第 4 步：RMSE 权重族的定义（666 行）**

```c
float w = qw ? qw[i]
        : rmse_type == 1 ? x[i] * x[i]
        : rmse_type == 2 ? 1
        : rmse_type == 3 ? fabsf(x[i])
        : sqrtf(fabsf(x[i]));
```

四种内置权重族：`x²`（RMSE，对大权重敏感）、`1`（MAE，等权）、`|x|`、`√|x|`。
`q6_K` 用 `rmse_type = 1`（`ggml-quants.c:1883`），`q3_K` 用 `rmse_type = 1` 且带 `qw`（1382 行）。

**第 5 步：最小二乘尺度求解（656-670）**

```c
sumlx += w*x[i]*l;   suml2 += w*l*l;
...
float scale = suml2 ? sumlx/suml2 : 0.0f;
```

给定整数分配 `l`，最优实值尺度 `scale = Σ(w·x·l) / Σ(w·l²)`（加权最小二乘闭式解）。

**第 6 步：尺度扰动搜索（673-693）**

```c
float best = scale * sumlx;                       // 当前解的目标值
for (int is = -9; is <= 9; ++is) {
    if (is == 0) continue;
    iscale = -(nmax + 0.1f*is) / max;             // 扰动 -0.9 ~ +0.9
    ... 重新量化并对所有 i 重算 sumlx/suml2 ...
    if (suml2 > 0 && sumlx*sumlx > best*suml2) {  // 目标函数更优？
        for (int i = 0; i < n; ++i) { L[i] = nmax + MAX(-nmax, MIN(nmax-1, nearest_int(iscale*x[i]))); }
        scale = sumlx/suml2; best = scale*sumlx;
    }
}
```

**为什么比较 `sumlx² > best·suml2` 而不是直接比 `scale`？**
因为 `scale = sumlx/suml2`，且目标函数是 `scale·sumlx = sumlx²/suml2`，
比较 `sumlx² / suml2 > best` 等价于 `sumlx² > best·suml2`。这样避免了除法，且与已有 `best` 同量纲。

**为什么扰动步长是 0.1？** `nmax` 是搜索的"名义量化级数"，`nmax + 0.1·is` 让 `iscale` 在
`(nmax-0.9)/|max|` 到 `(nmax+0.9)/|max|` 间变化。因为 `l = nearest_int(iscale·x)` 的整数解
只在 `iscale` 跨越若干阈值时改变，0.1 步长足以覆盖所有不同的整数分配组合（对 `nmax≤32` 而言）。

> **注意 `HAVE_BUGGY_APPLE_LINKER` 分支（657-662）**：在两个循环上用 `for (volatile int i = ...)`
> 强制禁止展开，以绕过 Apple `ld64 1015.7` 的链接器 bug。同样的 workaround 出现在
> `make_qkx2_quants`（806-811）与 `make_qkx3_quants`（1000-1005）。

### 2.4 `make_q3_quants` —— 带迭代重分配的 3-bit 专用量化（`ggml-quants.c:697-754`）

仅在 `quantize_row_q3_K_ref`（1241 行）使用。与 `make_qx_quants` 的差异：

**`do_rmse` 分支的坐标下降（720-742）**

```c
for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
        float w = x[i]*x[i];
        float slx = sumlx - w*x[i]*L[i];       // 暂时剔除第 i 项
        if (slx > 0) {
            float sl2 = suml2 - w*L[i]*L[i];
            int new_l = nearest_int(x[i] * sl2 / slx);   // 给定其他项最优的新 l
            new_l = MAX(-nmax, MIN(nmax-1, new_l));
            if (new_l != L[i]) {
                slx += w*x[i]*new_l;  sl2 += w*new_l*new_l;
                if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {  // 交叉相乘比较，无除法
                    L[i] = new_l; sumlx = slx; suml2 = sl2;
                    ++n_changed;
                }
            }
        }
    }
    if (!n_changed) break;    // 收敛提前退出
}
```

**`new_l = nearest_int(x[i] * sl2 / slx)` 的推导**：固定其他元素的 `L`，目标是最小化
`Σ w(x - scale·L)²`。对 `L[i]` 求导得最优 `L[i] = x[i]·Σ(w·L²) / Σ(w·x·L) = x[i]·sl2/slx`。
这是**逐元素的坐标下降（coordinate descent）**。

**交叉相乘的优势**：`slx²·suml2 > sumlx²·sl2` 等价于 `slx²/sl2 > sumlx²/suml2`，
即新目标优于旧目标，但完全避开浮点除法——在 5 轮 × n 次迭代的内循环中，这显著减少了延迟。

**返回值差异**：`do_rmse` 路径返回 `sumlx/suml2`（真实最小二乘尺度），
非 rmse 路径返回 `1/iscale`（`nmax/max` 的倒数）。

### 2.5 `make_qkx1_quants` —— 非对称量化的最简版本（`ggml-quants.c:756-797`）

```c
static float make_qkx1_quants(int n, int nmax, const float * x, uint8_t * L,
                              float * the_min, int ntry, float alpha)
```

**当前代码中已被注释掉调用**（`ggml-quants.c:1471` 与 `1658` 保留了
`//scales[j] = make_qkx1_quants(32, 15, ...)`），但仍保留实现与文档价值。

**算法**：交替优化 `min` 与 `scale`：

```c
if (min > 0) min = 0;                 // 769: 强制 min ≤ 0（数据之上必有 0 才能用无符号 L）
float iscale = nmax/(max - min);
for (int itry = 0; itry < ntry; ++itry) {
    ... l = clamp(nearest_int(iscale*(x[i] - min)), 0, nmax); ...
    scale = sumlx/suml2;              // 给定 min，最优 scale
    float sum = 0;
    for (int i = 0; i < n; ++i) sum += x[i] - scale*L[i];
    min = alpha*min + (1 - alpha)*sum/n;   // 790: 带阻尼的最小值更新
    if (min > 0) min = 0;
    iscale = 1/scale;
    if (!did_change) break;
}
*the_min = -min;
return scale;
```

**第 790 行的阻尼更新是关键**：`sum/n` 是"平均残差"，即让 `min` 恰好使残差均值为零的值。
但直接赋值会震荡，故用 `alpha` 做指数平滑。`alpha = 0.5` 表示新旧各半。

**`*the_min = -min` 的符号翻转**：量化值为 `L = round((x - min)/scale)`，
反量化是 `w = scale*L + min`。若约定输出 `m = -min`（`d` 与 `m` 均为正），
则反量化写作 `w = scale*L - m`——与 `q4_1`/`q5_1` 反量化代码中的 `x0*d + m` 形式符号相反。
这正是 `qkx` 系列与 `q4_1_ref` 的 `m` 语义**不同**之处，阅读时须留意具体 caller。

### 2.6 `make_qkx2_quants` —— 双参数联合最小二乘（`ggml-quants.c:799-878`）

用于 `q2_K`（908 行）与 `q4_K`/`q5_K` 的 `_ref`（1476/1663 行）。

**第 1 步：加权统计量（802-817）**

```c
float sum_w = weights[0], sum_x = sum_w * x[0];
for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    float w = weights[i];  sum_w += w;  sum_x += w * x[i];
}
if (min > 0) min = 0;
```

**第 2 步：初始误差基准（826-834）**

```c
float best_error = 0;
for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale*(x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    best_error += weights[i] * diff;
}
```

`use_mad` 标志切换 MAE（`fabsf`）与 MSE（`diff*diff`）度量。
`q2_K` 用 `use_mad = true`（908 行），`q4_K`/`q5_K` 用 `false`。

**第 3 步：闭式双参数求解（851-858）**

```c
float D = sum_w * sum_l2 - sum_l * sum_l;
if (D > 0) {
    float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
    float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
    if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
    ...
}
```

这是**加权线性回归** `x_i ≈ scale·L_i + min` 的正规方程解：

$$\text{scale} = \frac{S_w S_{xl} - S_x S_l}{S_w S_{l^2} - S_l^2}, \qquad \text{min} = \frac{S_{l^2} S_x - S_l S_{xl}}{S_w S_{l^2} - S_l^2}$$

**`this_min > 0` 的降级处理（855-858）**：约束 `min ≤ 0`（因 `L` 是无符号存储）。
若闭式解违反约束，则退化为"固定 `min = 0` 的单参数解 `scale = sum_xl/sum_l2`"。
这是标准的**带约束最小二乘投影**。

**第 4 步：尺度扫描（839-875）**

```c
for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta*is + nmax)/(max - min);
    ... 量化 → 算 sum_l, sum_l2, sum_xl → 闭式解 → 算 cur_error ...
    if (cur_error < best_error) { 接受并更新 L/scale/min }
}
```

`rmin`/`rdelta`/`nstep` 由 caller 给出，构成扫描参数：
`q2_K` 用 `(-0.5, 0.1, 15)`，`q4_K` 用 `(-1.0, 0.1, 20)`，`q5_K` 用 `(-0.5, 0.1, 15)`。
即扫描 `iscale` 从 `(nmax + rmin)` 到 `(nmax + rmin + nstep·rdelta)`。

### 2.7 `make_qkx3_quants` —— 权重可选的进化版（`ggml-quants.c:993-1074`）

与 `make_qkx2_quants` 的三点差异：

1. **`weights` 可为 NULL**（998-999、1008 行）：`float w = weights ? weights[i] : x[i]*x[i];`
   默认权重退化为 `x²`（即 RMSE 语义）。
2. **全零块判定改为 `max <= min`**（1015 行）而非 `max == min`，更稳健。
3. **全零时用 `memset(L, 0, n)`**（1016 行）而非逐元素循环。

被 `q2_K`/`q4_K`/`q5_K` 的 `_impl` 版本调用（1172、1583、1789 行），
参数统一为 `(-0.9f, 0.05f, 36, false)`——比 `_ref` 版本扫描更细（36 步 vs 15/20 步，步长 0.05 vs 0.1）。

### 2.8 `make_qp_quants` —— 极值搜索 + 坐标下降（`ggml-quants.c:1076-1147`）

用于量化**已经量化的 scales**（如 `q2_K` 的 15 级 scales、`q4_K` 的 63 级 scales）。
名字中的 `p` 指"positive"——它处理的是**非负**输入。

```c
float max = 0;
for (int i = 0; i < n; ++i) max = MAX(max, x[i]);      // 1078-1080: 注意只看 max，不看 |x|
if (max < GROUP_MAX_EPS) { memset L; return 0.f; }
float iscale = nmax / max;
for (int i = 0; i < n; ++i) L[i] = nearest_int(iscale * x[i]);   // 无 clamp（1087 行）
```

**第 1 阶段：只在负数方向搜索（1096-1112）**

```c
for (int is = -4; is <= 4; ++is) {
    if (is == 0) continue;
    float iscale_is = (0.1f*is + nmax)/max;   // 注意：无 clamp，允许 iscale 小于 nmax/max
    ...
    int l = nearest_int(iscale_is*x[i]);
    l = MIN(nmax, l);                          // 只 clamp 上界
    ...
    if (mse < best_mse) { best_mse = mse; iscale = iscale_is; }
}
```

搜索范围是 `(nmax - 0.4)/max` 到 `(nmax + 0.4)/max`，且**只接受更小的 MSE**，
所以尺度会单调向小调整。

**第 2 阶段：坐标下降（1123-1145）**，与 `make_q3_quants` 同构但**无 `slx > 0` 约束**：

```c
float slx = sumlx - w*x[i]*L[i];
float sl2 = suml2 - w*L[i]*L[i];
if (slx > 0 && sl2 > 0) {                      // 两个都需为正
    int new_l = nearest_int(x[i] * sl2 / slx);
    new_l = MIN(nmax, new_l);
    ...
}
```

### 2.9 `get_scale_min_k4` —— 12 字节承载 8×(scale+min)（`ggml-quants.c:880-887`）

```c
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

**位布局推导**（`j ∈ [0,8)`，`q` 为 `y[i].scales[12]`）：

- **`j ∈ [0,4)`**：`q[j]` 低 6 位是第 j 个子块的 scale；`q[j+4]` 低 6 位是其 min。
- **`j ∈ [4,8)`**：`q[j+4]` 的两个 nibble 分别是 scale/min 的低 4 位；
  高位来自 `q[j-4]`（scale）与 `q[j]`（min）的**高 2 位**（`>> 6`），左移 4 位拼接。

**字节分配表**（共 12 字节 = 96 bit = 8×6 + 8×6，零冗余）：

| 字节 | 承载内容 |
|------|---------|
| `q[0]`-`q[3]` | 子块 0-3 的 scale（低 6 位）+ 子块 4-7 的 scale 高 2 位 |
| `q[4]`-`q[7]` | 子块 0-3 的 min（低 6 位）+ 子块 4-7 的 min 高 2 位 |
| `q[8]`-`q[11]` | 子块 4-7 的 scale 低 4 位（`q[8+j-4]` 低 nibble）+ min 低 4 位（高 nibble） |

**这是本文件中最值得学习的位打包设计**：96 bit 的信息量被精确填满，
且编码/解码都是纯移位与掩码（无循环、无查表），可完全内联展开。
`quantize_row_q4_K_ref` 第 1494-1501 行的编码侧与之严格对偶：

```c
if (j < 4) {
    y[i].scales[j] = ls;
    y[i].scales[j+4] = lm;
} else {
    y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
    y[i].scales[j-4] |= ((ls >> 4) << 6);
    y[i].scales[j-0] |= ((lm >> 4) << 6);
}
```

**注意写入顺序依赖**：`y[i].scales[j-4] |= ...` 是**或等**操作，要求 `q[0..3]` 已被前 4 次迭代
（`j<4`）赋过初值。因此循环必须从 `j=0` 递增——这是一个隐式的循环顺序约束，
调换顺序会静默产生错误结果。

---

## 三、查找表与常量数据结构

### 3.1 语言前端宏机制（`ggml-common.h:1-72`）

`ggml-common.h` 被设计成**一个文件服务六种语言**，靠的是文件顶部的 `#if/#elif` 链：

| 宏 | 语言 | `ggml_half` 类型 | `GGML_COMMON_AGGR_S/U` |
|----|------|-----------------|----------------------|
| `GGML_COMMON_DECL_C` | C (`stdint.h`) | `uint16_t` | 空 |
| `GGML_COMMON_DECL_CPP` | C++ | `uint16_t` | `data`（匿名 union 的成员名） |
| `GGML_COMMON_DECL_METAL` | Metal | `half` | 空 |
| `GGML_COMMON_DECL_CUDA` | CUDA/MUSA | `half` | 空 / `data` |
| `GGML_COMMON_DECL_HIP` | HIP | `half` | 空 / `data` |
| `GGML_COMMON_DECL_SYCL` | SYCL | `sycl::half` | 空 / `data` |

**`GGML_COMMON_AGGR_U` / `GGML_COMMON_AGGR_S` 的设计意图**：

```c
// ggml-common.h:202-211
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qs[QK4_1 / 2];
} block_q4_1;
```

- **C 语言下**（`GGML_COMMON_AGGR_U` 为空）：得到匿名 union，可直接写 `y[i].d`、`y[i].dm`。
- **C++ 下**（`= data`）：`union { struct {...} data; ggml_half2 dm; } data;`，
  需写 `y[i].data.d`。这是因为 C++ 标准不允许匿名 struct 成员访问（虽然编译器普遍支持）。
- **Metal**：`GGML_COMMON_AGGR_U` 为空 —— Metal 支持匿名 union。

`GGML_EXTENSION`（`ggml-common.h:174-178`）在 GCC/Clang 下展开为 `__extension__`，
用于抑制 `-pedantic` 对匿名 union 的警告；MSVC 下展开为空。

**为什么这些结构体要跨语言共享？** 因为同一个 GGUF 文件必须被 CPU、CUDA、Metal 后端
以**完全相同的内存布局**解释。把结构体定义放在一个共享头文件中，是保证这一点的最简单方法——
比在每个后端各写一份、再靠人工同步可靠得多。

### 3.2 核心查找表清单

| 表名 | 尺寸 | 行号 | 结构与用途 |
|------|------|------|-----------|
| `kmask_iq2xs` | 8 | 509-511 | `{1,2,4,8,16,32,64,128}`：逐位掩码，用于 `signs & kmask_iq2xs[j]` 判断第 j 个元素的符号 |
| `ksigns_iq2xs` | 128 | 513-522 | 7-bit 符号组合 → 8 位符号掩码（**偶校验**编码） |
| `ksigns64` | 128 | 524-557 | 上述的 64-bit 展开版，供 SIMD 一次载入 |
| `iq2xxs_grid` | 256 | 560 | IQ2_XXS 格点，每项 `uint64` = 8 个 `int8` |
| `iq2xs_grid` | 512 | 627 | IQ2_XS 格点 |
| `iq2s_grid` | 1024 | 758 | IQ2_S 格点 |
| `iq3xxs_grid` | 256 | 1017 | IQ3_XXS 格点，每项 `uint32` = 4 个 `int8` |
| `iq3s_grid` | 512 | 1052 | IQ3_S 格点 |
| `kvalues_iq4nl` | 16 | 1120-1124 | IQ4_NL 非线性电平（单调递增，供二分查找） |
| `kvalues_fp4` | 16 | 1126 | FP4 电平表（`kvalues_mxfp4` 的来源） |
| `iq1s_grid` | `NGRID_IQ1S`(2048) | 1135 | IQ1_S/IQ1_M 共用格点，`uint64` 打包 |
| `iq1s_grid_gpu` | 2048 | 1650 | 上述的 GPU 膨胀版，每分量一字节（4 倍空间换无位运算） |
| `IQ1S_DELTA` / `IQ1M_DELTA` | — | 1132-1133 | `0.125f`，IQ1 系列的"非零电平偏移" |

### 3.3 `ksigns_iq2xs` 的偶校验设计（`ggml-common.h:513-522`）

```c
GGML_TABLE_BEGIN(uint8_t, ksigns_iq2xs, 128)
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147, ...
GGML_TABLE_END()
```

第 0 项是 `0`（全正），第 1 项是 `129 = 0b10000001`（第 0 位与第 7 位置 1）。

**为什么只存 128 项（7 bit）而非 256（8 bit）？** 因为第 8 个元素的符号由前 7 个决定：
格点坐标都是**奇数**，而 IQ2 系列的量化值为奇数 `{1,3,5,7}`，8 个奇数的乘积绝对值仍是奇数，
其符号与"8 个符号位中 1 的个数的奇偶性"相关。代码通过**强制偶校验**省下 1 bit：
`quantize_row_iq2_xxs_impl` 第 3349-3359 行在 `nflip % 2` 为奇数时**翻转权重最小的那个元素的符号**：

```c
if (nflip%2) {
    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
    for (int i = 1; i < 8; ++i) {
        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
        if (ax < min) { min = ax; imin = i; }
    }
    xval[8*k+imin] = -xval[8*k+imin];
    s ^= (1 << imin);
}
```

**为什么选权重最小的元素翻转？** 翻转会引入量化误差，选权重最小者使误差代价最小。
（权重的定义见下文 8.1 节的 imatrix 融合公式。）

### 3.4 `kvalues_iq4nl` 的电平推导

```c
GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
GGML_TABLE_END()
```

16 个电平，非等间距。观察归一化后（除以约 127）的间距：
`0.18, 0.17, 0.14, 0.13, 0.11, 0.10, 0.09, 0.09, 0.09, 0.09, 0.10, 0.12, 0.13, 0.16, 0.19`
——**中间密、两端疏**，这是标准正态分布的最优量化电平特征（Lloyd-Max 条件：
`(y_{i+1} - y_{i-1})/2 = y_i` 的离散近似）。

**注意 `-127` 而非 `-128`**：因为反量化后要与 `d`（fp16）相乘，
用 `-127` 使电平近似对称于 `[-1, 1]`，且避免 `-128` 在某些 SIMD 指令（如 `int8` 取绝对值）上的边界问题。

### 3.5 格点表的编码规律：`pg[i] = 2*l + 1`

所有格点表（`iq2xxs_grid` 等）都有一个统一规律，在初始化代码中被显式揭示：

```c
// ggml-quants.c:3770-3776  (iq3xs_init_impl)
for (int k = 0; k < grid_size; ++k) {
    int8_t * pos = (int8_t *)(the_grid + k);
    for (int i = 0; i < 4; ++i) {
        int l = (kgrid[k] >> 3*i) & 0x7;   // 取 3 bit
        pos[i] = 2*l + 1;                   // 映射到奇数 {-7..7}
    }
}
```

原始表 `kgrid_256` 等存的是**紧凑的 3-bit 值 `l`**（`ggml-quants.c:3708-3760`），
初始化时展开为奇数 `2l+1 ∈ {-7,-5,...,7}`。这样做的收益：

1. **表本身紧凑**：`uint16_t` 存 4 个 3-bit 值，而非 4 字节。
2. **展开后无零值**：奇数集合不含 0，使"格点是否有效"的判断可以与零值区分。
3. **反量化快速**：`dequantize_row_iq3_xxs`（2592-2593 行）直接读展开后的字节：
   ```c
   const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
   ```
   `uint8_t` 与 `int8_t` 位模式相同，直接当有符号用。

**`2*L+1` 的逆运算**出现在量化侧（`ggml-quants.c:3290`）：

```c
for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
```

`pg[i]` 是 `int8_t`（可能是负数），`(pg[i]-1)/2` 在 C 的向零截断语义下对负奇数正确：
`(-7-1)/2 = -4`，`(-5-1)/2 = -3`，与 `2*(-4)+1 = -7` 一致。

---

## 四、经典量化族（q1_0 ~ q8_1 / mxfp4 / nvfp4）

### 4.1 `quantize_row_q1_0_ref`（`ggml-quants.c:40-72`）

```c
const float d = sum_abs / qk;          // 48-52: L1 均值，而非 amax
y[i].d = GGML_FP32_TO_FP16(d);
for (int j = 0; j < qk / 8; ++j) y[i].qs[j] = 0;    // 57-59: 必须先清零
for (int j = 0; j < qk; ++j) {
    if (x[i*qk + j] >= 0.0f) y[i].qs[byte_index] |= (1 << bit_offset);
}
```

**设计要点**：

1. **缩放因子是 L1 均值 `Σ|x|/QK`**（48-52 行），而非其他类型的 `amax`。
   原因：1-bit 只有符号信息，若用 `amax` 作为 `d`，则所有反量化值都是 `±amax`，
   与真实幅值的偏差由 `amax - E|x|` 决定；用 `E|x|` 使期望误差最小化。
   注释明确说明："Just store sign of each weight directly (no normalization)"。
2. **反量化 `dequantize_row_q1_0`（419-437）**：`bit ? d : -d`（`434` 行）。
注意 `0` 被编码为**正号**（`x >= 0.0f` → 置位），因此重构值**永远不可能是 0**，
恒为 `±d`。这是 1-bit 量化的固有特性：只有符号可表达，
而 `d = E|x|` 是整块幅度的最优统一代表。

`LLAMA_FTYPE_MOSTLY_Q1_0` 在 `include/llama.h:157` 定义为枚举值 40，
与 `LLAMA_FTYPE_MOSTLY_Q2_0`（值 41，`llama.h:158`）相邻，
是该文件中最后加入的两个类型。
3. **`bit_offset = j % 8`，低位在前**：与其他类型的位序约定一致。

> **不可用作输出类型的推测**：由于反量化值恒定非零（±d），
> `q1_0` 的残差在零附近权重上表现差；它被设计用于**极端压缩实验**，
> 而非生产模型（`LLAMA_FTYPE_MOSTLY_Q1_0` 在 `llama.h:157` 标注）。

### 4.2 `quantize_row_q2_0_ref`（`ggml-quants.c:74-110`）

```c
const float d = amax;                            // 88: scale 就是 amax
int q = (int)roundf(w * id) + 1;                 // 102: round 后偏移 +1
if (q < 0) q = 0;  if (q > 3) q = 3;             // 103-104: clamp
y[i].qs[byte_index] |= ((uint8_t)q << bit_offset);   // 107
```

**编码方案**（注释在 98-99 行）：`00 = -1, 01 = 0, 10 = +1, 11 = +2`，
即量化值域是 $\{-1, 0, +1, +2\}$ —— **非对称三值 + 1**。

**为什么是 `{-1,0,1,2}` 而非 `{-1.5,-0.5,0.5,1.5}`？** 后者需要额外的偏移量存储；
前者让 `q - 1` 直接就是重构系数（反量化 `454` 行：`((int)q - 1) * d`），
且 `q` 天然非负可直接位打包。代价是引入了**量化值的直流偏置**（均值 0.5），
但因为 `d = amax` 且权重近似零均值，偏置被 `d` 的保守取值部分抵消。

### 4.3 `quantize_row_q4_0_ref`（`ggml-quants.c:113-148`）

```c
float amax = 0.0f, max = 0.0f;
for (int j = 0; j < qk; j++) {
    const float v = x[i*qk + j];
    if (amax < fabsf(v)) { amax = fabsf(v); max = v; }   // 126-129: 记录带符号值
}
const float d  = max / -8;              // 132: 负号！
const float id = d ? 1.0f/d : 0.0f;
...
const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));   // 141-142
y[i].qs[j]  = xi0;  y[i].qs[j] |= xi1 << 4;
```

**四层设计细节**：

1. **`d = max / -8` 的负号**：使 `d` 的符号与 `max` 相反，于是 `x*id = x*(-8/max)`，
   对 `x = max` 得 `-8`，加 8.5 后为 `0.5`，截断为 `0`。
   即**量化值 0 对应最大幅值，量化值 15 对应最大负值**——量化值域是 `[0,15]` 映射到 `[max, -8·|d|]`。
2. **`+8.5f` 而非 `+8.0f` 的舍入技巧**：`(int8_t)(x0 + 8.5f)` 对正数是四舍五入，
   对负数是"向零方向 + 0.5 后截断"≈ 四舍五入。因为 `x0` 可能为负（`id` 带负号），
   直接 `roundf` 也可行，但 `+8.5` 一次加法即可覆盖舍入与偏移两个操作。
3. **`MIN(15, ...)` 缺少下界 clamp**：理论上 `x0 + 8.5f` 可能为负（当 `x` 与 `max` 反号且幅值更大时？
   ——不可能，因 `max` 是 `amax` 对应的值）。但浮点边界下可能出现 `-0.0001` 截断为 `0`，
   恰好是合法的下界。这是**依赖不变量而非防御性检查**的写法。
4. **反量化是纯减法**（`469-475`）：`(qs & 0x0F) - 8` 与 `(qs >> 4) - 8`，
   **注意输出顺序**：`y[i*qk + j] = x0`（元素 `j`），`y[i*qk + j + qk/2] = x1`（元素 `j+16`）。
   即低 nibble 存前半块、高 nibble 存后半块。

### 4.4 `quantize_row_q4_1_ref` / `q5_1_ref` —— 非对称版本

```c
// q4_1, ggml-quants.c:157-183
float min = FLT_MAX, max = -FLT_MAX;
for (int j = 0; j < qk; j++) { if (v < min) min = v; if (v > max) max = v; }
const float d  = (max - min) / ((1 << 4) - 1);   // 168: 除以 15
const float id = d ? 1.0f/d : 0.0f;
y[i].d = GGML_FP32_TO_FP16(d);
y[i].m = GGML_FP32_TO_FP16(min);                  // 172: m = min（非 -min！）
const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f)); // 178: 只需 +0.5 舍入
```

**与 `q4_0` 的三个关键差异**：

| 维度 | `q4_0`（对称） | `q4_1`（非对称） |
|------|---------------|-----------------|
| scale 计算 | `max / -8` | `(max - min) / 15` |
| 额外存储 | 无 | `m = min`（fp16） |
| 量化值域 | `[0,15]` 映射到 `[+8,-7]` | `[0,15]` 映射到 `[min, max]` |
| 舍入偏移 | `+8.5f`（含偏移+舍入） | `+0.5f`（仅舍入） |
| type_size | 18 | 20 |

`q4_1` 用 `+0.5f` 因为量化值不需要偏移（`x0 = (x-min)*id` 已经非负）。

**`y[i].m` 存的是 `min`**，反量化写作 `x0*d + m`（`494-495` 行）。
这与 `make_qkx*_quants` 返回的 `*the_min = -min` 语义**相反**，是阅读时必须区分的陷阱。

### 4.5 `quantize_row_q5_0_ref` / `q5_1_ref` —— 第 5 位的分离存储

```c
// ggml-quants.c:211-228
uint32_t qh = 0;
for (int j = 0; j < qk/2; ++j) {
    const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
    const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));
    y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);      // 低 4 位照旧
    qh |= ((xi0 & 0x10u) >> 4) << (j + 0);                // 第 5 位 → bit j
    qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);             // 第 5 位 → bit j+16
}
memcpy(&y[i].qh, &qh, sizeof(qh));   // 227: memcpy 而非直接赋值
```

**位布局**：`qs` 存低 4 位（与 `q4_0` 完全相同的布局），`qh`（4 字节 = 32 bit）存第 5 位，
按 `j` 顺序排列（元素 0..15 在 bit 0..15，元素 16..31 在 bit 16..31）。

**为什么用 `uint32_t qh` 累加再 `memcpy`？** `y[i].qh` 声明为 `uint8_t qh[4]`（`ggml-common.h:232`），
若逐字节写入需要 4 次字节操作且顺序依赖端序；用 `uint32_t` 在寄存器中累加后一次 `memcpy`，
让编译器生成单条 32 位存储。这是一个**把端序问题交给 `memcpy` 解决**的典型写法。

**反量化的位提取**（`514-515`）：

```c
const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;   // 取 bit j → 移到 bit 4
const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;   // 取 bit j+16 → 留在 bit 4
```

注意两行的**不对称**：第一个先右移再左移 4 位；第二个先右移 12（= j+16-4）再直接掩码。
两者等价，但第二个少一次移位——这是刻意的手工优化（编译器通常也能完成此变换，
但显式写出可以保证）。

**`q5_1` 的差异**（`231-273`）：`d = (max-min)/31`，`m = min`，
量化值 `(uint8_t)(x0 + 0.5f)` **无 `MIN(31, ...)` clamp**（261-262 行）——
因为 `x0 = (x-min)/d` 已在 `[0,31]` 内，clamp 冗余。

### 4.6 `quantize_row_q8_0_ref` / `q8_1_ref`

```c
// q8_0, ggml-quants.c:276-299
const float d = amax / ((1 << 7) - 1);   // 288: 除以 127
const float id = d ? 1.0f/d : 0.0f;
y[i].qs[j] = roundf(x0);                  // 296: int8 输出
```

**`/127` 而非 `/128`**：`int8_t` 范围是 `[-128, 127]`，除以 127 保证 `±amax` 都落在合法域内
（`amax/127 * 127 = amax`）。若除以 128，`+amax` 会溢出到 `-128`。

**`q8_1` 的额外 `s` 字段**（`302-335`）：

```c
int sum = 0;
for (int j = 0; j < QK8_1/2; ++j) {
    y[i].qs[          j] = roundf(v0);
    y[i].qs[QK8_1/2 + j] = roundf(v1);
    sum += y[i].qs[          j];
    sum += y[i].qs[QK8_1/2 + j];
}
y[i].s = GGML_FP32_TO_FP16(sum*d);        // 333: s = d * Σq
```

`s = d·Σq` 预存了"反量化值之和"，用于非对称点积的第二项展开（见第十一章）。
注意累加的是 **量化后**的整数和（而非浮点和），保证精确整数一致。

### 4.7 `quantize_row_mxfp4_ref`（`ggml-quants.c:350-382`）

```c
const uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;   // 368
const float d = GGML_E8M0_TO_FP32_HALF(e);
y[i].e = e;
for (int j = 0; j < qk/2; ++j) {
    const uint8_t x0 = best_index_mxfp4(x[i*qk + 0    + j], d);
    const uint8_t x1 = best_index_mxfp4(x[i*qk + qk/2 + j], d);
    y[i].qs[j]  = x0;
    y[i].qs[j] |= x1 << 4;
}
```

**E8M0 编码推导**：E8M0 是"8 位全指数、无尾数"的格式，值 = `2^(e-127)`。
代码要找一个 `e` 使 `2^(e-127)` 约为 `amax/4`：

```c
floorf(log2f(amax)) - 2 + 127     // ← -2 即除以 4
```

**为什么除以 4？** MXFP4 的 E2M1 分量最大值是 `6.0`（`{0.5,1,1.5,2,3,4,6}` 配符号）。
缩放因子取 `amax/4` 时，`amax` 对应 E2M1 值为 `4.0`（合法），
留出一点余量避免边界舍入问题。`GGML_E8M0_TO_FP32_HALF` 宏名中的 `HALF` 提示
它返回的是 `0.5 * 2^(e-127)` 或类似半值，具体定义在 `ggml-impl.h`。

**`best_index_mxfp4`（337-348）** 是**线性扫描**而非二分：

```c
int best_index = 0;
float best_err = fabsf(kvalues_mxfp4[0]*e - x);
for (int i = 1; i < 16; i++) {
    float err = fabsf(kvalues_mxfp4[i]*e - x);
    if (err < best_err) { best_index = i; best_err = err; }
}
```

**为什么不用 `best_index_int8` 的二分？** 因为 `kvalues_mxfp4` 的**符号与幅度交错**，
不单调递增（FP4 电平是 `{0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, ...}` 之类的排列）。
况且只有 16 项，线性扫描的代价可接受。

### 4.8 `quantize_row_nvfp4_ref`（`ggml-quants.c:384-417`）

```c
for (int s = 0; s < n_sub; s++) {          // n_sub = 64/16 = 4
    const float * xb = x + i*qk + s*qk_sub;
    float amax = 0.0f;
    for (int j = 0; j < qk_sub; j++) if (amax < fabsf(xb[j])) amax = fabsf(xb[j]);
    const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);   // 405: 除以 6.0
    y[i].d[s] = ue;
    const float d = ggml_ue4m3_to_fp32(ue);
    for (int j = 0; j < qk_sub/2; ++j) {
        const uint8_t x0 = best_index_mxfp4(xb[0        + j], d);
        const uint8_t x1 = best_index_mxfp4(xb[qk_sub/2 + j], d);
        y[i].qs[s*(qk_sub/2) + j] = x0 | (x1 << 4);
    }
}
```

**与 MXFP4 的两点差异**：

1. **缩放因子是 UE4M3（4 位指数 + 3 位尾数）而非 E8M0**，精度更高，但需软件编解码
   （`ggml_fp32_to_ue4m3`/`ggml_ue4m3_to_fp32` 在 `ggml-impl.h`）。
2. **`amax / 6.0` 而非 `/4`**：注释说明 "amax / 6.0 maps the max E2M1 value (6.0) to amax"，
   即让 `amax` 精确对应 E2M1 的最大值 6.0，充分利用 FP4 的动态范围。
   （MXFP4 之所以用 `/4` 是因为 E8M0 只有 2 的幂次，无法精确表示 1/6。）

**`nvfp4` 复用 `best_index_mxfp4` 与 `kvalues_mxfp4`**（410-411 行）：
NVFP4 与 MXFP4 的**分量编码完全相同**（都是 E2M1），只有缩放因子格式与分组大小不同。
这是代码复用的良好示例——两个类型的 `_ref` 实现共享了电平查找逻辑。

---

## 五、K-quant 超块族

K-quant 的核心结构是 **256 元素超块 → 子块 → 两级缩放**。以下逐类型分析。

### 5.1 `quantize_row_q2_K_ref`（`ggml-quants.c:891-959`）

**第 1 步：16 个子块各自量化（906-917）**

```c
uint8_t L[QK_K];
uint8_t Laux[16];
float   weights[16];
float mins[QK_K/16];
float scales[QK_K/16];
const float q4scale = 15.f;

for (int j = 0; j < QK_K/16; ++j) {
    for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);   // 907: 权重 = |x|
    scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
    if (scale > max_scale) max_scale = scale;
    if (min   > max_min  ) max_min   = min;
}
```

- **子块大小 16，量化级数 4**（`nmax = 3`，即 2-bit → 4 个电平）。
- `weights[l] = fabsf(x[...])`：用**绝对幅度**做权重，让大权重的量化误差被优先优化。
- 注释 904 行说明 "as we are deducting the min, scales are always positive"，
  故可用 `max_scale` 而非 `max_abs_scale` 来归一。

**第 2 步：对 scales 本身再量化（919-939）**

```c
if (max_scale > 0) {
    float iscale = q4scale/max_scale;              // q4scale = 15
    for (int j = 0; j < QK_K/16; ++j) {
        int l = nearest_int(iscale*scales[j]);
        y[i].scales[j] = l;                        // 低 4 位存 scale
    }
    y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
}
...
if (max_min > 0) {
    float iscale = q4scale/max_min;
    for (int j = 0; j < QK_K/16; ++j) {
        int l = nearest_int(iscale*mins[j]);
        y[i].scales[j] |= (l << 4);                // 高 4 位存 min
    }
    y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
}
```

**`q2_K` 的 scales 打包是最简单的**：一个字节 = 低 4 位 scale + 高 4 位 min，
16 个子块用 16 字节（`scales[QK_K/16] = scales[16]`）。
两级缩放：`d`/`dmin` 是 fp16 的"超级缩放"，`scales[j]` 的 nibble 是 4-bit 的子块缩放。
反量化体现为（`979` 行）：

```c
dl = d * (sc & 0xF);  ml = min * (sc >> 4);
... *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
```

**第 3 步：用最终量化参数重新量化（940-949）**

```c
for (int j = 0; j < QK_K/16; ++j) {
    const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
    if (!d) continue;
    const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
    for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int((x[16*j + ii] + dm)/d);
        l = MAX(0, MIN(3, l));
        L[16*j + ii] = l;
    }
}
```

**这一步是关键的"重量化"（requantize）**：`make_qkx2_quants` 输出的 `L` 是基于**浮点** scale/min 的，
但最终存储的是**量化后**的 scale/min（经过 fp16 截断 + 4-bit 截断）。
直接用原始 `L` 会引入二次误差。这里用**实际存储值**（`GGML_FP16_TO_FP32(y[i].d) * nibble`）
重新计算 `L`，使编码与解码严格一致。`if (!d) continue;` 跳过零 scale 子块（保持 `L` 为 0）。

**第 4 步：位打包（951-955）** —— 4 路交织（见 4.4 节范式二）。

**反量化 `dequantize_row_q2_K`（961-991）的结构**：

```c
for (int n = 0; n < QK_K; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t sc = x[i].scales[is++];
        dl = d * (sc & 0xF); ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
        sc = x[i].scales[is++];
        dl = d * (sc & 0xF); ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
        shift += 2;
    }
    q += 32;
}
```

每个 128 元素半块由 4 次 `shift += 2` 迭代覆盖，每次产出 32 个元素（两个 16 元素子块）。
`is` 连续递增索引 `scales[0..15]`，与 `.qs` 的 4 路交织布局精确对应。

### 5.2 `quantize_row_q3_K_ref`（`ggml-quants.c:1229-1303`）

`q3_K` 是**对称**类型（无 min），用 6-bit 有符号 scales。

**第 1 步：`make_q3_quants` + 幅度追踪（1240-1246）**

```c
float max_scale = 0;
float amax = 0;
for (int j = 0; j < QK_K/16; ++j) {
    scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);   // nmax=4, do_rmse
    float scale = fabsf(scales[j]);
    if (scale > amax) { amax = scale; max_scale = scales[j]; }     // 保留带符号的 max_scale
}
```

**注意 `max_scale` 保留符号**（1244 行），与 `q2_K` 的 `max_scale` 恒正不同——
因为 `q3_K` 无 min，scale 的符号承载了数据的方向信息。

**第 2 步：6-bit 有符号 scale 的位打包（1248-1265）**

```c
memset(y[i].scales, 0, 12);
if (max_scale) {
    float iscale = -32.f/max_scale;
    for (int j = 0; j < QK_K/16; ++j) {
        int8_t l = nearest_int(iscale*scales[j]);
        l = MAX(-32, MIN(31, l)) + 32;         // 映射到 [0, 63]
        if (j < 8) y[i].scales[j] = l & 0xF;
        else       y[i].scales[j-8] |= ((l & 0xF) << 4);
        l >>= 4;
        y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
    }
    y[i].d = GGML_FP32_TO_FP16(1/iscale);
}
```

**位布局推导**（16 个子块的 6-bit scale → 12 字节 = 96 bit = 16 × 6）：

| 字节 | 承载 |
|------|------|
| `scales[0..7]` 的低 nibble | 子块 0..7 的 scale 低 4 位 |
| `scales[0..7]` 的高 nibble | 子块 8..15 的 scale 低 4 位 |
| `scales[8..11]` | 子块 0..15 的 scale **高 2 位**，每字节装 4 个子块（`2*(j/4)` 定位） |

`l += 32` 把 `[-32,31]` 映射到 `[0,63]`，使 6-bit 可作为无符号存储；
解码侧（`1270` 行）再 `- 32` 还原：`sc = (...) - 32`。

**第 3 步：重量化（1267-1280）** 与 `q2_K` 同构，但用**除法**而非减 min：

```c
for (int ii = 0; ii < 16; ++ii) {
    int l = nearest_int(x[16*j + ii]/d);      // 对称：直接除以 d
    l = MAX(-4, MIN(3, l));                    // 量化值域 [-4, 3]
    L[16*j + ii] = l + 4;                      // 存为 [0,7]
}
```

**第 4 步：高位分离（1282-1294）**

```c
memset(y[i].hmask, 0, QK_K/8);
// We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
int m = 0;
uint8_t hm = 1;
for (int j = 0; j < QK_K; ++j) {
    if (L[j] > 3) {
        y[i].hmask[m] |= hm;
        L[j] -= 4;
    }
    if (++m == QK_K/8) { m = 0; hm <<= 1; }
}
```

`L` 的范围是 `[0,7]`（3 bit），但只存 2 bit 到 `qs`，第 3 位存进 `hmask`。
`hmask[m]` 的第 `hm` 位代表元素 `m + 32*k` 的高位。注释准确描述了布局。

**注意条件 `L[j] > 3` 与 `L[j] -= 4`**：等价于取 `L[j] >> 2` 与 `L[j] & 3`
（因为 `L ∈ [0,7]`）。但写成减法的形式让意图更清晰。

**反量化的位重组（1305-1353）** 是 `q3_K` 中最复杂的部分，用了 `uint32_t aux[4]` 做**批量位重排**：

```c
const uint32_t kmask1 = 0x03030303;
const uint32_t kmask2 = 0x0f0f0f0f;
uint32_t aux[4];
const int8_t * scales = (const int8_t*)aux;

memcpy(aux, x[i].scales, 12);          // 12 字节 → aux[0..2]（aux[3] 未初始化！）
uint32_t tmp = aux[2];
aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
```

**这段代码在做什么？** 把"低 nibble + 高 2 位分散在别处"的 12 字节布局，
**重排成 16 个连续的 6-bit 值**（每个占 1 字节，值域 `[0,63]`），便于后续 `*y++` 逐个取用。

拆解：

- `aux[0]`/`aux[1]` 的低 nibble 各含 4 个子块的 scale 低 4 位（`& kmask2` 保留）。
- 高 2 位来自 `tmp = aux[2]`（即 `scales[8..11]`）中按 2 bit 分组的值。
- `((tmp >> shift) & kmask1) << 4` 把每字节中 2 bit 的高位搬到 bit 4-5，
  与低 nibble 拼成完整 6 bit。

**`aux[3]` 未初始化的问题**：`memcpy(aux, x[i].scales, 12)` 只写了 12 字节，
`aux[3]` 是**未初始化**的。但 `aux[3]` 在第 3 行被整体赋值（`aux[3] = ...`），
不依赖旧值——所以正确。这是一个**刻意的、依赖人工核对的优化**：
用 `uint32_t` 载入刚好凑齐 12 字节的对齐读取，代价是引入一个未初始化变量
（Valgrind 类工具会报警，但逻辑正确）。

**为什么 `scales` 声明为 `int8_t*` 别名？** 因为解码后按 `int8_t` 读取并减去 32：
`dl = d_all * (scales[is++] - 32)`（1336 行）。用 `int8_t` 视图让减法在 8 位域内自然完成。

### 5.3 `quantize_row_q4_K_ref`（`ggml-quants.c:1457-1527`）

**第 1 步：8 个 32 元素子块（1470-1485）**

```c
for (int j = 0; j < QK_K/32; ++j) {
    //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);  // 已弃用
    float sum_x2 = 0;
    for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
    float av_x = sqrtf(sum_x2/32);
    for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);   // 1475: 关键权重公式
    scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
    ...
}
```

**权重公式 `av_x + |x|` 的设计**：`av_x` 是该子块的 RMS 幅度。
权重 = RMS + 绝对幅度，即"基础底噪 + 自身重要性"。
- 加 `av_x` 保证**所有权重均为正**（避免 `|x|` 近零时权重过小导致该元素被忽略）。
- 用 RMS 而非均值是量纲匹配（与 `|x|` 同量纲）。
- 注释 1471 行保留了被弃用的 `make_qkx1_quants` 调用，反映算法演进历史。

**第 2 步：直接用 max 归一（1487-1504）**

```c
float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
for (int j = 0; j < QK_K/32; ++j) {
    uint8_t ls = nearest_int(inv_scale*scales[j]);
    uint8_t lm = nearest_int(inv_min*mins[j]);
    ls = MIN(63, ls);  lm = MIN(63, lm);
    ... 与 get_scale_min_k4 对偶的打包（1494-1501）...
}
y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);
```

**注意 `_ref` 版本没有用 `make_qp_quants`**，而是直接用 `63/max` 归一 + 舍入。
对比 `_impl` 版本（1586-1587 行）：

```c
float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);
```

`make_qp_quants` 会做极值搜索 + 坐标下降来最小化加权 MSE，
这就是 `_impl` 精度更高（但不确定）的原因之一。

**第 3 步：位打包（1519-1523）**

```c
for (int j = 0; j < QK_K; j += 64) {
    for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
    q += 32;
}
```

64 元素一组、2 路交织（低 nibble 是元素 `j+l`，高 nibble 是 `j+l+32`）。
反量化（1541-1548 行）与此对偶，每 64 元素组消耗 2 个 scales（`is` 递增 2）。

### 5.4 `quantize_row_q5_K_ref`（`ggml-quants.c:1644-1729`）

与 `q4_K` 的同与异：

| 维度 | `q4_K` | `q5_K` |
|------|--------|--------|
| `nmax` | 15（4 bit） | 31（5 bit） |
| `make_qkx2_quants` 参数 | `(-1.f, 0.1f, 20, false)` | `(-0.5f, 0.1f, 15, false)` |
| 重量化 clamp | `MIN(15, l)` | `MIN(31, l)` |
| 第 5 位存储 | 无 | `qh[j]` 按 `m1`/`m2` 逐位（`1710-1724`） |

**第 5 位的独特交织**（1710-1724）：

```c
uint8_t m1 = 1, m2 = 2;
for (int n = 0; n < QK_K; n += 64) {
    for (int j = 0; j < 32; ++j) {
        int l1 = L[n + j];
        if (l1 > 15) { l1 -= 16; qh[j] |= m1; }
        int l2 = L[n + j + 32];
        if (l2 > 15) { l2 -= 16; qh[j] |= m2; }
        ql[j] = l1 | (l2 << 4);
    }
    m1 <<= 2; m2 <<= 2;
    ql += 32;
}
```

`m1` 从 1 开始每次左移 2 位（1, 4, 16, 64），`m2` 从 2 开始（2, 8, 32, 128）。
于是 `qh[j]` 的**偶数位**（0,2,4,6）承载低 nibble 元素的第 5 位，
**奇数位**（1,3,5,7）承载高 nibble 元素的第 5 位。
这使一个字节的 8 位被两个 4 路交织组均匀占用。反量化（1750-1751 行）用 `u1`/`u2` 顺序读取。

### 5.5 `quantize_row_q6_K_ref`（`ggml-quants.c:1869-1937`）

`q6_K` 是**高精度对称**类型（6-bit），实现结构最清爽。

**第 1 步：16 个 16 元素子块（1881-1892）**

```c
for (int ib = 0; ib < QK_K/16; ++ib) {
    const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);  // nmax=32, rmse_type=1
    scales[ib] = scale;
    const float abs_scale = fabsf(scale);
    if (abs_scale > max_abs_scale) { max_abs_scale = abs_scale; max_scale = scale; }
}
```

注意这里用 **`max_abs_scale`** 做比较、`max_scale` 存带符号值——
因为 `make_qx_quants` 的返回值符号取决于 `max` 的符号，可能出现负 scale。

**第 2 步：全零块特判（1894-1899）**

```c
if (max_abs_scale < GROUP_MAX_EPS) {
    memset(&y[i], 0, sizeof(block_q6_K));
    y[i].d = GGML_FP32_TO_FP16(0.f);
    x += QK_K;
    continue;
}
```

这是所有 K-quant 中**唯一用 `memset` 整块清零**的地方。为什么不直接返回？
因为 `L` 数组在循环中复用，`qs`/`qh`/`scales` 需要显式置零。

**第 3 步：8-bit scale 的截断陷阱（1901-1905）**

```c
float iscale = -128.f/max_scale;
y[i].d = GGML_FP32_TO_FP16(1/iscale);
for (int ib = 0; ib < QK_K/16; ++ib) {
    y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
}
```

**只有上界 `MIN(127, ...)`，没有下界 clamp**——因为 `y[i].scales` 是 `int8_t`，
负值合法（`-128` 到 `127`）。但 `nearest_int` 的结果若小于 `-128` 会**实现定义地截断**。
实践中不会发生：`max_scale` 是 `abs_scale` 最大的那个，故 `|iscale*scales[ib]| ≤ 128`。
这又是一个**依赖不变量**的写法。

**第 4 步：4 路交织 + 高位分离（1919-1933）**

```c
for (int j = 0; j < QK_K; j += 128) {
    for (int l = 0; l < 32; ++l) {
        const uint8_t q1 = L[j + l +  0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l+ 0] = q1 | (q3 << 4);      // 低 nibble 存 q1, 高 nibble 存 q3
        ql[l+32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2)
              | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
    }
    ql += 64;  qh += 32;
}
```

`ql` 的布局与 `q2_K`/`q3_K` 不同：这里 `ql[l]` 装元素 `l` 和 `l+64`（跨 64），
`ql[l+32]` 装元素 `l+32` 和 `l+96`。高位 `qh[l]` 每 2 bit 一个元素（共 4 个）。

**反量化（1939-1968）** 用 `is = l/16` 计算 scale 索引（`1952` 行），
即每 16 个元素共享一个 scale，与 `sc[is+0]`、`sc[is+2]`、`sc[is+4]`、`sc[is+6]` 的步进对应。

**一个值得注意的实现差异**：`q6_K` 的 `_impl` 版（1970-2052）与 `_ref` 版几乎逐行相同，
唯一区别是 `make_qx_quants` 的最后一个参数——`_ref` 传 `NULL`（1996 行对应 1883 行），
`_impl` 传 `qw`（1994 行）：

```c
// _impl
if (quant_weights) {
    const float * qw = quant_weights + QK_K*i + 16*ib;
    scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, qw);   // 直接传，不做 sigma2 融合
} else {
    scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
}
```

**注意 `q6_K` 不融合 `sigma2`**（对比 `q4_K` 的 `qw[l] * sqrtf(sigma2 + x²)`），
1992-1993 行的注释保留了被弃用的融合写法：

```c
//for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
//scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
```

即 `q6_K` 的 imatrix 使用更"生硬"（直接用 imatrix 值当权重），是历史遗留。

### 5.6 `quantize_row_q5_K_impl` 与 `q4_K_impl` 的 `sigma2` 因子差异

对比两个 `_impl` 版本的 `sigma2` 计算：

```c
// q4_K_impl, ggml-quants.c:1570
float sigma2 = 2*sum_x2/QK_K;
// q5_K_impl, ggml-quants.c:1775
float sigma2 = 2*sum_x2/QK_K;
// q2_K_impl, ggml-quants.c:1167
float sigma2 = sumx2/QK_K;          // 注意：无 2 倍
// q3_K_impl, ggml-quants.c:1369
float sigma2 = 2*sumx2/QK_K;
// iq1_s_impl, ggml-quants.c:4548
float sigma2 = 2*sumx2/QK_K;
```

`q2_K` 缺少 `2*` 因子是一个**历史不一致**（该行在 git 历史中未与其余类型同步调整）。
它只影响 `weight[l] = qw[l] * sqrtf(sigma2 + x²)` 的权重分配尺度，
从而轻微影响 imatrix 引导下的量化结果——对无 imatrix 路径无影响。

---

## 六、三值量化族（tq1_0 / tq2_0）

三值量化面向 BitNet b1.58 与 TriLMs 这类**训练时即约束权重为 {-1,0,+1}** 的模型。
与其余类型的本质差异：**缩放因子 `d = amax` 是唯一的元数据**，且不需要子块两级缩放
（因为三值权重的分布本身具有良好的动态范围）。

### 6.1 `quantize_row_tq1_0_ref`（`ggml-quants.c:2316-2380`）

**核心思路**：用**5 个三进制数字打包进 1 字节**（$3^5 = 243 < 256$）。

```c
const float d = amax;
const float id = d ? 1.0f/d : 0.0f;
y[i].d = GGML_FP32_TO_FP16(d);

// 5 elements per byte, along 32 bytes
for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
    for (size_t m = 0; m < 32; ++m) {
        uint8_t q = 0;
        for (size_t n = 0; n < 5; ++n) {
            int xi = lroundf(x[m + n*32] * id) + 1;   // -1,0,1 → 0,1,2
            q *= 3;
            q += xi;
        }
        q = ((uint16_t)q * 256 + (243 - 1)) / 243;    // 243 → 256 的缩放
        y[i].qs[j + m] = q;
    }
    x += 5*32;
}
```

**三个关键设计**：

1. **`q *= 3; q += xi;` 的进制转换**：等价于 `q = q*3 + xi`，
   5 次迭代后得到 $\sum xi_n 3^{4-n}$，即 5 位三进制数的十进制表示（范围 `[0, 242]`）。
   注意这是**大端**编码（第一个元素是最高位）——反量化时用 `pow3[n]` 顺序读取（2439 行）。

2. **`q * 256 / 243` 的"重新标定"**：把 `[0,242]` 线性映射到 `[0,255]`。
   这一步是 `tq1_0` 最独特的地方——**它牺牲了一点精度换取解码的极简性**。
   解码时（`2441-2442` 行）：
   ```c
   uint8_t q = x[i].qs[j + m] * pow3[n];
   int16_t xi = ((uint16_t) q * 3) >> 8;
   ```
   先乘 `3^n`，然后 `(q*3) >> 8` 即 `q * 3 / 256 ≈ q / 85.33`，
   相当于除以 `243/256 * 85.33 ≈ 81`。**只用整数乘法和移位就完成了三进制解码**——
   这正是"映射到 [0,255]"的目的：让 `>> 8` 这个免费的移位成为除法。

3. **边界处理**：`q = ((uint16_t)q * 256 + (243 - 1)) / 243` 中的 `+242` 是**向上取整除法** `⌈q*256/243⌉`，
   避免向下取整把最大码字 `242` 映射到 `255` 以下（那会浪费一个码位）。
   用 `uint16_t` 是因为 `242*256 = 61952` 超过 `uint8_t`。

**两段循环的原因**：`sizeof(y->qs)` 是 48 字节（`QK_K - 4*QK_K/64)/5 = (256-16)/5 = 48`），
不是 32 的倍数。故前 32 字节用"每字节 5 元素、跨 32 的步长"处理，剩余 16 字节换步长。
`sizeof(y->qs) - sizeof(y->qs) % 32 = 48 - 16 = 32`，所以第二个循环处理 `j = 32` 一档。

**`qh` 的 4 元素/字节**（2364-2377）：

```c
for (size_t j = 0; j < sizeof(y->qh); ++j) {
    uint8_t q = 0;
    for (size_t m = 0; m < 4; ++m) {
        int xi = lroundf(x[j + m*sizeof(y->qh)] * id) + 1;
        q *= 3;
        q += xi;
    }
    q *= 3;                       // 2373: 额外乘 3 —— 放到最高位
    q = ((uint16_t)q * 256 + (243 - 1)) / 243;
    y[i].qh[j] = q;
}
```

**第 2373 行的 `q *= 3` 是刻意的**：`qh` 用 4 元素/字节（$3^4 = 81 < 256$），
但为了与 `qs` 共享同一个解码路径，冗余乘一次 3 把它"提升"为 5 位三进制形式
（最低位为 0）。注释说明："shift the first value to the most significant trit"。
这样 `dequantize_row_tq1_0` 的 `qh` 段（2457-2463）可以复用与 `qs` 完全相同的解码代码。
**用一点存储冗余换取代码路径的统一**——这是 `tq1_0` 中值得记录的设计取舍。

### 6.2 `quantize_row_tq2_0_ref`（`ggml-quants.c:2382-2412`）

```c
for (size_t j = 0; j < sizeof(y->qs); j += 32) {
    for (size_t m = 0; m < 32; ++m) {
        uint8_t q = 0;
        for (size_t n = 0; n < 4; ++n) {
            int xi = lroundf(x[m + n*32] * id) + 1;   // -1,0,1 → 0,1,2
            q += (xi & 3) << (2*n);
        }
        y[i].qs[j + m] = q;
    }
    x += 4*32;
}
```

比 `tq1_0` 简单得多：**2 bit 一个元素，4 元素/字节**，`xi ∈ {0,1,2}` 直接位打包。
反量化（`2478` 行）`(qs >> (l*2)) & 3` 再 `- 1`。

**`xi & 3` 的必要性**：`xi` 理论上 ∈ `{0,1,2}`，但 `lroundf` 在极端浮点情况下
可能返回 `-1`（当 `x*id < -0.5` 时加 1 后仍为负）。`& 3` 把 `-1` 变成 `3`，
即量化值 `+2` 而非溢出。这是**用位掩码代替 clamp** 的廉价防御。

### 6.3 反量化的解码数学

```c
// ggml-quants.c:2438-2445
for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
    for (size_t n = 0; n < 5; ++n) {
        for (size_t m = 0; m < 32; ++m) {
            uint8_t q = x[i].qs[j + m] * pow3[n];
            int16_t xi = ((uint16_t) q * 3) >> 8;
            *y++ = (float) (xi - 1) * d;
        }
    }
}
```

**循环顺序 `n` 在外、`m` 在内**，与编码时的 `m` 在外、`n` 在内相反——
这是编码-解码对偶性的直接体现：编码把 5 个元素压进一个字节，
解码从每个字节中取出各个 trit，因此外层遍历 trit 序号。

**`pow3[6] = {1, 3, 9, 27, 81, 243}`**（2432 行）中的 `243` 从未被用到（`n < 5` 只到 `pow3[4]=81`），
是预留的数组哨兵。

**`xi - 1` 的偏移还原**：编码时 `+1` 把 `{-1,0,1}` 搬到 `{0,1,2}`，解码减回。
`(float) (xi - 1) * d` 中 `xi` 是 `int16_t`，减法在整数域完成后再转浮点——
避免了浮点累减的精度问题。

---

## 七、非线性量化族（iq4_nl / iq4_xs）

### 7.1 `quantize_row_iq4_nl_impl` 的统一设计（`ggml-quants.c:4966-5075`）

这个函数**同时服务 `iq4_nl`（块 32）与 `iq4_xs`（超块 256）**，通过参数区分：

```c
static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size,
        const float * x, ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values, const float * quant_weights, const int ntry) {
```

| 调用方 | `super_block_size` | `block_size` | 是否有两级 scale | `ntry` |
|--------|-------------------|-------------|----------------|--------|
| `quantize_iq4_nl`（5077） | `QK4_NL` = 32 | 32 | 否（`super/block == 1`） | 7 |
| `quantize_iq4_xs`（5115） | `QK_K` = 256 | 32 | 是（`super/block == 8`） | 7 |
| `quantize_row_iq4_nl_ref`（5100） | `QK4_NL` = 32 | 32 | 否 | **-1** |

**`ntry = -1` 与 `ntry = 7` 的语义差异**（源码 5001、5015、5062 行）：

```c
float d = ntry > 0 ? -max/values[0] : max/values[0];    // 5001: 符号不同
...
for (int itry = -ntry; itry <= ntry; ++itry) { ... }     // 5015: ntry=-1 时循环体不执行
...
if (ntry > 0) { ... 重新量化 ... }                       // 5062: ntry=-1 时跳过
```

`ntry = -1` 表示**不做尺度搜索、不做重量化**的"快速参考路径"，
用于 `quantize_row_iq4_nl_ref` —— 即供 `type_traits.from_float_ref` 调用的确定性路径。
`ntry = 7` 表示做 15 次尺度扫描（`itry ∈ [-7, 7]`）的最优路径。

**第 1 步：加权搜索（4981-5035）**

```c
for (int ib = 0; ib < super_block_size/block_size; ++ib) {
    const float * xb = x + ib*block_size;
    uint8_t * Lb = L + ib*block_size;
    if (quant_weights) {
        const float * qw = quant_weights + ib*block_size;
        for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
    } else {
        for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];   // 退化权重
    }
    ...
    float d = ntry > 0 ? -max/values[0] : max/values[0];
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < block_size; ++j) {
        float al = id*xb[j];
        int l = best_index_int8(16, values, al);
        Lb[j] = l;
        float q = values[l];
        float w = weight[j];
        sumqx += w*q*xb[j];
        sumq2 += w*q*q;
    }
    d = sumq2 > 0 ? sumqx/sumq2 : 0.f;      // 最小二乘尺度
    float best = d*sumqx;
    for (int itry = -ntry; itry <= ntry; ++itry) { ... 扫描 ... }
    scales[ib] = d;
    ...
}
```

**`-max/values[0]` 的负号**：`values[0]` 是 `kvalues_iq4nl[0] = -127`（负），
故 `max/values[0]` 为负；加负号后 `d` 为正。这是为了对齐"最大幅值对应最负电平"的约定，
使 `Lb[j]` 的索引与幅度单调相关（便于调试）。

**第 2 步：两级 scale 的打包（5037-5059）** —— `iq4_xs` 专属

```c
if (super_block_size/block_size > 1) {
    int nb = super_block_size/block_size;                  // 8
    memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
    float d = -max_scale/32;
    dh[0] = GGML_FP32_TO_FP16(d);
    float id = d ? 1/d : 0.f;
    for (int ib = 0; ib < nb; ++ib) {
        int l = nearest_int(id*scales[ib]);
        l = MAX(-32, MIN(31, l));                          // 6-bit 有符号
        float dl = d * l;
        float idl = dl ? 1/dl : 0.f;
        uint8_t * Lb = L + ib*block_size;
        const float * xb = x + ib*block_size;
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = best_index_int8(16, values, idl*xb[j]);  // 用实际 dl 重量化
        }
        l += 32;                                            // → [0,63]
        uint8_t l_l = l & 0xf;
        uint8_t l_h = l >>  4;
        if (ib%2 == 0) scales_l[ib/2] = l_l;
        else          scales_l[ib/2] |= (l_l << 4);
        scales_h[ib/8] |= (l_h << 2*(ib%8));
    }
}
```

对应 `block_iq4_xs` 的 `d`(fp16) + `scales_h`(uint16) + `scales_l[QK_K/64 = 4]`：
8 个子块的 6-bit scale 拆为低 4 位（`scales_l`，2 个/字节）与高 2 位（`scales_h`，4 个/uint16）。
**与 `q3_K` 的 12 字节方案思路一致**，但因为只有 8 个 6-bit 值（48 bit），需要 6 字节 = 4 + 2。

**第 3 步：单个 scale 的情形（5060-5068）** —— `iq4_nl` 专属

```c
} else {
    dh[0] = GGML_FP32_TO_FP16(scales[0]);
    if (ntry > 0) {
        float id = scales[0] ? 1/scales[0] : 0;
        for (int j = 0; j < super_block_size; ++j) {
            L[j] = best_index_int8(16, values, id*x[j]);
        }
    }
}
```

`iq4_nl` 只有 1 个块（`super/block == 1`），故 `dh[0]` 直接就是块 scale。

**第 4 步：最终打包（5070-5074）**

```c
for (int i = 0; i < super_block_size/32; ++i) {
    for (int j = 0; j < 16; ++j) {
        q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
    }
}
```

**与所有其他类型的布局都不同**：这里 `q4[j]` 的低 nibble 是元素 `j`、
高 nibble 是元素 `j+16`（同一 32 元素块内），而非 `q4_0` 的 `j`/`j+qk/2`。
反量化（`2734-2737`）与此对偶。

### 7.2 `quantize_iq4_nl` 与 `quantize_iq4_xs` 的入口（5077-5138）

```c
size_t quantize_iq4_nl(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK4_NL == 0);
    int64_t nblock = n_per_row/QK4_NL;
    char * qrow = (char *)dst;
    uint8_t L[QK4_NL];  float weight[QK4_NL];
    uint16_t unused_h;  uint8_t * unused_l = NULL;
    float scale;
    for (int64_t row = 0; row < nrow; ++row) {
        block_iq4_nl * iq4 = (block_iq4_nl *)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = quant_weights ? quant_weights + QK4_NL*ibl : NULL;
            quantize_row_iq4_nl_impl(QK4_NL, 32, src + QK4_NL*ibl, &iq4[ibl].d, iq4[ibl].qs,
                    &unused_h, unused_l, &scale, weight, L, kvalues_iq4nl, qw, 7);
        }
        src += n_per_row;  qrow += nblock*sizeof(block_iq4_nl);
    }
    return nrow * nblock * sizeof(block_iq4_nl);
}
```

**注意 `unused_h` / `unused_l` 的哑参数技巧**：`iq4_nl` 没有两级 scale，
但复用的 `_impl` 函数签名要求这两个指针。传入**栈上哑变量**而非 `NULL`，
避免了在 `_impl` 内部加 `if (scales_h)` 分支——用一次无用赋值换取热路径无分支。
`unused_l = NULL` 则因为单 scale 分支（5060-5068）确实不碰 `scales_l`。

**`quantize_iq4_xs` 的 `ntry = 7`**（5127 行）与 `iq4_nl` 相同，
但 `quantize_row_iq4_nl_ref` 传 `ntry = -1`（5111 行）——参考路径不做搜索。

`quantize_row_iq4_xs_ref` 的实现非常简洁（5135-5138）：

```c
void quantize_row_iq4_xs_ref(const float * GGML_RESTRICT x, block_iq4_xs * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);     // 直接复用 imatrix 路径（传 NULL）！
}
```

即 `iq4_xs` 的"参考实现"就是 `quantize_iq4_xs` 传 `quant_weights = NULL`。
对比其他类型普遍为 `_ref` 写独立实现，这里是代码复用的做法——
`ntry = 7` 的搜索在无 imatrix 时是纯精度收益（退化为 `weight[j] = xb[j]*xb[j]`），
故可以接受其非"最简"的实现。

---

## 八、IQ 格点量化族

IQ 族是本文件中最复杂的部分（行 2818-5327，约 2500 行），其核心是**格点预计算 + 最近邻搜索**。

### 8.1 IQ 数据结构的运行时管理（`ggml-quants.c:2820-2851`）

```c
typedef struct {
    uint64_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq2_entry_t;

static iq2_entry_t iq2_data[4] = { {NULL,NULL,NULL}, ... };

static inline int iq2_data_index(enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S
             || type == GGML_TYPE_IQ1_M  || type == GGML_TYPE_IQ2_S);
    return type == GGML_TYPE_IQ2_XXS ? 0 :
           type == GGML_TYPE_IQ2_XS  ? 1 :
           type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? 2 : 3;
}

static inline int iq2_grid_size(enum ggml_type type) {
    GGML_ASSERT(...);
    return type == GGML_TYPE_IQ2_XXS ? 256 :
           type == GGML_TYPE_IQ2_XS  ? 512 :
           type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? NGRID_IQ1S : 1024;
}
```

**`iq2_data[4]` 的槽位分配是理解全族的关键**：

| 槽位 | 类型 | `grid_size` | 共享理由 |
|------|------|------------|---------|
| 0 | `IQ2_XXS` | 256 | 独立（`kgrid_2bit_256`） |
| 1 | `IQ2_XS` | 512 | 独立（`kgrid_2bit_512`） |
| 2 | `IQ1_S`, `IQ1_M` | `NGRID_IQ1S` = 2048 | **两者共享 `iq1s_grid`** |
| 3 | `IQ2_S` | 1024 | 独立 |

**`IQ1_S` 与 `IQ1_M` 共享槽位 2** 的原因：两者的格点表都是 `iq1s_grid`
（`ggml-common.h:1135`），差异仅在**打包方式**（`IQ1_S` 每 32 元素一组、
`IQ1_M` 每 16 元素一组且有内嵌 scale）。因此格点/映射/邻居表可完全复用，
只是 `iq2xs_init_impl` 的 `nwant` 参数需按类型区分（见下）。

**`nwant`（邻居数量）的差异**（`ggml-quants.c:3108`）：

```c
const int nwant = type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? 3
                : type == GGML_TYPE_IQ2_S ? 1 : 2;
```

- `IQ1_S`/`IQ1_M` 取 3：格点仅 8 元素 1-bit（值域极小），需更多候选才能找到好的匹配。
- `IQ2_S` 取 1：格点 1024 项已很密集，1 个最近邻足够。
- `IQ2_XXS`/`IQ2_XS` 取 2：居中。

### 8.2 `iq2xs_init_impl` 的三阶段构建（`ggml-quants.c:2853-3258`）

**阶段 1：展开格点坐标（3105-3115）**

```c
uint64_t * the_grid = (uint64_t *)malloc(grid_size*sizeof(uint64_t));
for (int k = 0; k < grid_size; ++k) {
    int8_t * pos = (int8_t *)(the_grid + k);
    for (int i = 0; i < 8; ++i) {
        int l = (kgrid[k] >> 2*i) & 0x3;    // 取 2 bit
        pos[i] = (int8_t)(2*l + 1);          // → 奇数 {1,3,5,7}
    }
}
```

把 `kgrid_2bit_256`（紧凑 2-bit/元素）展开为 8 个 `int8_t`（每项 8 字节）。
`2*l+1` 使值域为 `{1,3,5,7}`（全正），符号由 `block_signs` 单独承载。

**阶段 2：正查表 `map`（3116-3127）**

```c
kmap_q2xs = (int *)malloc(kmap_size*sizeof(int));       // kmap_size = 43692
for (int i = 0; i < kmap_size; ++i) kmap_q2xs[i] = -1;
uint32_t aux32; uint8_t * aux8 = (uint8_t *)&aux32;
for (int i = 0; i < grid_size; ++i) {
    aux32 = kgrid_q2xs[i];
    uint16_t index = 0;
    for (int k = 0; k < 8; ++k) {
        uint16_t q = (aux8[k] - 1)/2;        // 奇数 → [0,3]
        index |= (q << 2*k);
    }
    kmap_q2xs[index] = i;                    // 逆映射：干净码字 → 格点索引
}
```

`kmap_size = 43692` 是**8 个 2-bit 值的所有可能组合数**？不——$4^8 = 65536$，
而 43692 < 65536。这是**实测的"有效码字"数量**：只有这些码字才会被
`quantize_row_iq2_xxs_impl` 的初始量化产生（因为量化值是 `2*L+1` 且 `L ∈ [0,3]`，
但受符号/尺度约束，实际可达的码字少于 $4^8$）。用更小的 `kmap_size` 节省内存。

`map[i] >= 0` 表示码字 `i` **恰好在格点上**；`map[i] < 0` 表示不在，
其负值编码了邻居列表的偏移（见阶段 3）。

**阶段 3：邻居表构建（3129-3257）** —— 分两趟

*第一趟（并行计数）*：

```c
int * n_per_i = (int *)malloc(kmap_size*sizeof(int));
int num_neighbors = 0, num_not_in_map = 0;
#ifdef GGML_USE_OPENMP
    #pragma omp parallel reduction(+:num_neighbors,num_not_in_map)
#endif
{
    int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
    int8_t pos[8];
    int i;
#ifdef GGML_USE_OPENMP
    #pragma omp for schedule(dynamic, 64)
#endif
    for (i = 0; i < kmap_size; ++i) {
        if (kmap_q2xs[i] >= 0) { n_per_i[i] = 0; continue; }
        ++num_not_in_map;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2*k) & 0x3;
            pos[k] = 2*l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
            dist2[2*j+0] = d2;   dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
        ... 统计前 nwant 个距离层级的元素个数 ...
        n_per_i[i] = n;  num_neighbors += n;
    }
    free(dist2);
}
```

**这一段是初始化最耗时的部分**：对每个未命中码字（最多 43692 个），
计算到所有格点（256 个）的距离并排序。复杂度 $O(43692 \times 256 \log 256) \approx 9 \times 10^7$ 次操作。
OpenMP 的 `schedule(dynamic, 64)` 用于负载均衡（不同码字的邻居数差异较大）。

*`iq2_compare_func`（2847-2851）*按 `(距离, 索引)` 字典序排序：

```c
return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
```

加索引作为次序键是为了**结果确定性**——相同距离的格点顺序固定，
否则不同平台的 `qsort` 可能产生不同邻居表，破坏量化确定性。

*"取前 n 个距离层级"的微妙逻辑（3242-3252）*：

```c
int local_counter = offsets[i];
kmap_q2xs[i] = -(local_counter + 1);
int d2 = dist2[0];
uint16_t * start = &kneighbors_q2xs[local_counter++];
int n = 0, nhave = 1;
for (int j = 0; j < grid_size; ++j) {
    if (dist2[2*j] > d2) {
        if (nhave == nwant) break;
        d2 = dist2[2*j];
        ++nhave;
    }
    kneighbors_q2xs[local_counter++] = dist2[2*j+1];
    ++n;
}
*start = n;
```

逻辑是：**取距离最小的 `nwant` 个"距离层级"的全部格点**（而非固定个数）。
例如 `nwant = 2` 时，若最近的 3 个格点距离都是 5，则 3 个全取。
这样保证选出的候选**完整覆盖**了最优距离层级，不受格点编号影响。

`kmap_q2xs[i] = -(local_counter + 1)` 用**负值减一**编码偏移：
解码侧 `kneighbors_q2xs - kmap_q2xs[u] - 1`（如 `3389` 行）还原指针。

*第二趟（串行填充）*：与第一趟逻辑相同，但把结果写入 `kneighbors_q2xs`。
偏移表 `offsets[i]` 在第一趟后即可确定，故两趟方案可行。
代码注释（3190 行附近）说明了为何分两趟：`neighbours` 数组的大小需先统计才能分配。

### 8.3 `iq2_find_best_neighbour`（`ggml-quants.c:3270-3292`）

```c
static int iq2_find_best_neighbour(const uint16_t * neighbours, const uint64_t * grid,
        const float * xval, const float * weight, float scale, int8_t * L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) { best_d2 = d2; grid_index = neighbours[j]; }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}
```

**关键点**：距离度量是**加权的量化域误差**（`weight[i]*(scale*q - x)²`），
而非初始化阶段的欧氏距离。因为此时已有 `scale` 估计，可以评估**真实的量化误差**。
`weight` 参数是 `waux[i] = sqrtf(weight[i])`（3340 行），
即预开方的权重——因为这里需要 `weight[i]*diff² = (sqrt(w)*diff)²`，
传入开方值可避免在循环内开方。

**返回值**：函数返回格点索引，同时**写出** `L[i] = (pg[i]-1)/2`（逆运算，见 3.5 节）。
这个"输入是浮点、输出是格点索引 + 整数量化值"的双重职责使调用点代码更紧凑。

### 8.4 `quantize_row_iq2_xxs_impl` 的完整流程（`ggml-quants.c:3294-3471`）

**步骤 1：权重与辅助权重（3336-3339）**

```c
for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
```

**步骤 2：符号分离 + 偶校验修正（3340-3361）** —— 见 3.3 节
`block_signs[k] = s & 127` 只保留 7 bit。

**步骤 3：全零块特判（3362-3368）**

```c
float max = xval[0];
for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
if (max < GROUP_MAX_EPS) { scales[ib] = 0; memset(L, 0, 32); continue; }
```

注意 `xval` 已取绝对值，故用 `MAX` 而非 `fabsf` 比较。

**步骤 4：初始尺度估计（3369-3375）**

```c
float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
float eff_max = scale*kMaxQ;
if (eff_max <= 0) { scales[ib] = 0; memset(L, 0, 32); continue; }
```

`kMaxQ = 3`（3308 行），`nmax = kMaxQ+1 = 4`，量化值域 `[0,4]`。
`eff_max = scale*kMaxQ` 是该尺度下能表示的最大值（用 `kMaxQ=3` 而非 4，
因为量化值域是 `2L+1 ∈ {1,3,5,7}`，最大值对应 `L=3`）。

**步骤 5：尺度扫描找最优（3376-3404）**

```c
float best = 0;
for (int is = -6; is <= 6; ++is) {
    float id = (2*kMaxQ-1+is*0.1f)/eff_max;      // 5.9 ~ 7.1 之间
    float this_scale = 1/id;
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 8; ++i) {
            int l = nearest_int(0.5f*(id*xval[8*k+i]-1));    // 反解 L = (x/scale - 1)/2
            Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
        }
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0) {
            const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
        }
    }
    float sumqx = 0, sumq2 = 0;
    for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2*Laux[i] + 1;                 // 格点坐标是奇数
        sumqx += w*xval[i]*q;
        sumq2 += w*q*q;
    }
    if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
        scale = sumqx/sumq2; best = scale*sumqx;
        memcpy(L, Laux, 32);                     // 保存最优解
    }
}
```

**`neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1` 的指针算术**：
`kmap_q2xs[u]` 是负数 `-(offset+1)`，故 `-kmap_q2xs[u] - 1 = offset`，
把基指针前移 `offset` 个元素即可指向该码字的邻居列表。这是**用负数编码偏移**的技巧。
（注意 `-` 作用在负值上是加，故实际是 `kneighbors_q2xs + offset`。）

**`id = (2*kMaxQ-1 + is*0.1)/eff_max`**：`2*3-1 = 5`，故 `id ≈ 5.9/eff_max` 到 `7.1/eff_max`。
`id*xval - 1` 的 `-1` 来自奇数格点的偏移（`q = 2L+1`，
故 `L = (q-1)/2 = (id*x - 1)/2`，代码里 `0.5f*(...)` 即此）。

**`sumqx² > best*suml2` 的用法与前文一致**，避免除法。

**步骤 6：用最终尺度重新解码（3405-3430）**

```c
if (scale > 0) {
    float id = 1/scale;
    for (int k = 0; k < 4; ++k) {
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) {
            int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
            l = MAX(0, MIN(kMaxQ-1, l));
            u |= (l << 2*i);
        }
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0) {
            const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
        }
        const int8_t * pg = (const int8_t *)(kgrid_q2xs + grid_index);
        for (int i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
    }
    float sumqx = 0, sumq2 = 0;
    for (int i = 0; i < 32; ++i) { sumqx += weight[i]*xval[i]*(2*L[i]+1); sumq2 += weight[i]*(2*L[i]+1)*(2*L[i]+1); }
    if (sumq2 > 0) scale = sumqx/sumq2;
}
```

**为什么扫描后就够了，还要再解码一次？** 因为扫描循环里 `iq2_find_best_neighbour`
在命中 `map` 时**不会更新 `Laux`**（只有 `kmap_q2xs[u] >= 0` 时直接用 `u` 的格点，
但没有把格点值写回 `Laux`）。步骤 6 无条件从格点表重新解码 `L`，
保证 `L` 与最终 `grid_index` 严格一致。这是一个**易错点的显式修复**。

**步骤 7：负尺度防御（3431-3436）**

```c
if (scale < 0) {
    // This should never happen, but just in case, flip scale so that it is positive
    // (we use uint's to encode the scale) and correspondingly flip quant signs.
    scale = -scale;
    for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
}
```

注释清楚地解释了必要性：scale 用 **无符号** `uint16` 存储（`y[ibl].d` 是 `ggml_half`，
但量化值域由 `block_signs` 与格点共同决定），故必须为正。翻转符号时同步翻转 `block_signs`。

**步骤 8：打包（3437-3460）**

```c
for (int k = 0; k < 4; ++k) {
    uint16_t u = 0;
    for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
    int grid_index = kmap_q2xs[u];
    if (grid_index < 0) {
        printf("Oops: found point %u not on grid:", u);
        for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
        printf("\n");
        GGML_ABORT("fatal error");
    }
    q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
    q2[2*ib+1] |= (block_signs[k] << 7*k);
}
```

**这里的 `kmap_q2xs[u] < 0` 是致命错误**（而非走邻居搜索），因为 `L` 已由
`iq2_find_best_neighbour` 从格点解码得到，其对应的码字**必须**在格点上。
若不在，说明算法逻辑有 bug。这是全文件唯一的 `GGML_ABORT` 用法之一，
也是一个**自检式断言**。

`q2[2*ib+0]` 存 4 个格点索引（每个 8 bit），`q2[2*ib+1]` 存 4 组符号（每个 7 bit）。
最后一次性写入 `y[ibl].qs`：

```c
memcpy(y[ibl].qs, q2, QK_K/4);      // 3460 附近
```

### 8.5 `quantize_row_iq3_xxs_impl` 与 D4 格点（`ggml-quants.c:3938-4152`）

`iq3_xxs` 用 **4 元素/格点**而非 8 元素，这是本质差异：

```c
// 量化值域更小：{1,3,5,7} 中的 4 个 → 用 3 bit 索引
for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 4; ++i) {
        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
    }
    uint16_t u = 0;
    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);   // 3 bit/元素
    int grid_index = kmap_q3xs[u];
    ...
}
```

**`kmap_size = 4096 = 8^4`**（`ggml-quants.c:3761`）—— 4 个 3-bit 值的所有组合，
**完整覆盖**（不像 IQ2 的 43692 < 65536）。因此理论上所有码字都可达。

**D4 格点的几何意义**：`iq3xxs_grid` 的 256 个点是 **D4 格（4 维根格）** 的
最近邻壳层。D4 格是 4 维空间中"最密"的格，其 Voronoi 胞体积最大——
这意味着同样数量的格点能覆盖 $[-1,1]^4$ 空间的最大比例，量化误差最小。

### 8.6 `quantize_row_iq1_s_impl` 的结构（`ggml-quants.c:4508-4671`）

`iq1_s` 是最低比特（1.5625 bpw）的 IQ 类型，机制上有几处独特设计。

**电平表：带偏移的三值**（4540-4541）

```c
const float x_p[3] = {-1 + IQ1S_DELTA,  IQ1S_DELTA, 1 + IQ1S_DELTA};
const float x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};
```

`IQ1S_DELTA = 0.125f`（`ggml-common.h:1132`）。这六个数是**格点的连续基**——
格点坐标为整数 `{-1, 0, 1}`，量化时对其加/减 `0.125` 使候选值不再重合，
从而让搜索能区分"应当取 0 还是 ±1"的边界情况。

**对称搜索**（`x_p` vs `x_m`）用于处理 `delta` 符号的不确定性：
`delta` 本身也作为一个参数被编码（`qh[ib] & 0x8000`，见 `2662` 行），
故搜索需考虑两种符号。

**`pairs` 参数与 `idx` 别名**（4553 行）：

```c
int * idx = (int *)(pairs + 1);
```

`pairs` 是 caller 传入的工作区，前 4 字节（1 个 float）存 `sumw`，其余作为 `int` 数组使用。
这种**用同一块内存承载不同类型数据**的做法节省了栈空间，
但要求 caller 与 callee 严格约定布局（见 caller `quantize_iq1_s`，4672 行）。

**反量化的 delta 处理**（2662 行）：

```c
const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
for (int l = 0; l < 4; ++l) {
    const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
    for (int j = 0; j < 8; ++j) {
        y[j] = dl * (grid[j] + delta);
    }
}
```

格点索引 = 低 8 位（`qs[l]`）+ 高 3 位（`qh[ib]` 的 `3*l` 位段），共 11 bit → 2048 项。
`delta` 作为**加性偏移**统一施加（而非逐元素），因为它编码的是该 8 元素组的整体偏置。

`dl = d * (2*((qh[ib] >> 12) & 7) + 1)`（2661 行）——3-bit scale 映射为**奇数** `{1,3,...,15}`，
这是一种"隐式指数"编码，扩大动态范围而无需额外位数。

### 8.7 `quantize_row_iq1_m_impl` 的内嵌 scale（`ggml-quants.c:4692-4945`）

`iq1_m` 的 `block_iq1_m` **没有独立的 `d` 字段**（`ggml-common.h:433-437`），
scale 被内嵌在 `scales[QK_K/32]` 中。反量化侧的解包（`2687` 行）：

```c
const uint16_t * sc = (const uint16_t *)x[i].scales;
scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
const float d = GGML_FP16_TO_FP32(scale.f16);
```

**这个 fp16 是从 4 个 uint16 的**高位 nibble** 中"拼"出来的**：
`sc[0]` 的 bit 12-15、`sc[1]` 的 bit 12-15、`sc[2]` 的 bit 12-15、`sc[3]` 的 bit 12-15
恰好组成 16 bit 的 fp16。

**为什么这样设计？** `iq1_m` 的 `scales` 数组同时承载三类信息：
- 每个 32 元素组有**两个** 3-bit scale（`sc[ib/2]` 的 `6*(ib%2)` 与 `6*(ib%2)+3` 位段，见 2694-2695 行）；
- 累积起来，4 个 `sc` 值用掉 12 位 × 4 = 48 位承载 8 个组的 3-bit scale（共 24 位）；
- **剩余的高 4 位**（每个 `sc` 的 bit 12-15）刚好 16 位，用于存全局 fp16 缩放因子。

**这是全文件信息密度最高的位打包**：一个 8 字节数组同时容纳
"8 组 × 3-bit scale" 与 "1 个 fp16 全局 scale"，零浪费。
`iq1m_scale_t` union（`ggml-common.h:441-444`）就是为此定义的：

```c
typedef union {
    ggml_half f16;
    uint16_t  u16;
} iq1m_scale_t;
```

`validate` 侧也复用了同样的解码逻辑（`5615-5618` 行），
且 `ggml_validate_row_data` 对 `IQ1_M` 的校验**必须自己拼出 fp16 再校验**——
这是唯一需要"解码后校验"的类型（其余类型的 `d` 都是直接字段）。

**打包侧的 `1.1125f` 修正因子**（`ggml-quants.c:4937`）：

```c
s.f16 = GGML_FP32_TO_FP16(d*1.1125f); // 1.1125f is another fudge factor. Don't ask me why it is needed.
sc[0] |= ((s.u16 & 0x000f) << 12);
sc[1] |= ((s.u16 & 0x00f0) <<  8);
sc[2] |= ((s.u16 & 0x0f00) <<  4);
sc[3] |= ((s.u16 & 0xf000) <<  0);
```

**注意四个移位方向**：`<<12`、`<<8`、`<<4`、`<<0`，把 fp16 的 4 个 nibble
从低到高**依次放到 4 个 `sc` 值的 bit 12-15**。解码侧（`2687` 行）用
`(sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)`
精确逆运算：注意掩码的**不对称**（第一个不需掩码因为 `>>12` 已只剩 4 位，
最后一个不需移位因为 `sc[3]` 的 bit 12-15 就是目标位置）。

`1.1125f` 这个"魔法系数"作者自己都在注释里承认无法解释（"Don't ask me why it is needed"）
——它是通过实验校准的经验修正，用于补偿 IQ1_M 特有的多层级量化误差累积。
**这类注释在本文件里非常罕见**，值得记录：它诚实地标记了一处未被完全理解的实现细节。

**`iq1_m` 的 3-bit scale 与 `masks[]`**（`ggml-quants.c:4916`）：

```c
y[ibl].qh[ib] |= masks[shifts[ib]];
```

`shifts[ib] = best_k ∈ [0,3]` 编码了该 32 元素组使用了哪一组电平偏移
（`x_p` / `x_m` 的 4 种组合，见 4871-4872 行）。`masks[]` 把 `best_k` 转成
写入 `qh` 的位模式——这 2 bit 与 `qh` 的格点高位共享同一数组，通过位掩码冲突避免。

---

## 九、q8_K 与点积伙伴

### 9.1 `quantize_row_q8_K_ref`（`ggml-quants.c:2768-2805`）

```c
float max = 0, amax = 0;
for (int j = 0; j < QK_K; ++j) {
    float ax = fabsf(x[j]);
    if (ax > amax) { amax = ax; max = x[j]; }
}
if (!amax) { y[i].d = 0; memset(y[i].qs, 0, QK_K); x += QK_K; continue; }
//const float iscale = -128.f/max;
// We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
const float iscale = -127.f/max;
for (int j = 0; j < QK_K; ++j) {
    int v = nearest_int(iscale*x[j]);
    y[i].qs[j] = MIN(127, v);
}
for (int j = 0; j < QK_K/16; ++j) {
    int sum = 0;
    for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j*16 + ii];
    y[i].bsums[j] = sum;
}
y[i].d = 1/iscale;
```

**三个关键点**：

1. **`-127` 而非 `-128`**，且注释明确说明了原因："We need this change for IQ2_XXS,
   else the AVX implementation becomes very awkward"。
   `-127` 使量化值域对称（`[-127, 127]`），AVX2 的 `_mm256_sign_epi8` 等指令
   在 `-128` 上会有符号边界问题（`abs(-128)` 溢出）。这是**为一个特定后端的
   SIMD 实现而调整通用量化参数**的典型案例。

2. **`for (int ii = 0; ii < 16; ++ii)` 使用了内层循环变量名 `ii`**，
   而外层循环用 `j`。但注意：**内外层循环变量名重复了外层的 `j`**——
   外层是 `for (int j = 0; j < QK_K/16; ++j)`，内层 `ii` 从 0 到 15，
   累加 `y[i].qs[j*16 + ii]`。这个写法正确（`ii` 是新变量），
   但**遮蔽风险**在于：若写成 `j` 就会无限循环。这是一处需要注意的命名细节。

3. **`y[i].d = 1/iscale` 而非 `-max/127`**：`iscale = -127/max`，
   故 `1/iscale = -max/127`。写成 `1/iscale` 是**数学等价但精度略异**的表达式
   （两次除法 vs 一次）。保持与其他类型一致的写法便于对拍验证。

**`bsums[j] = Σq` 的用途**：第 11.3 节展示，它用于非对称点积的 min 项。

### 9.2 `dequantize_row_q8_K`（`ggml-quants.c:2807-2816`）

```c
for (int i = 0; i < nb; i++) {
    for (int j = 0; j < QK_K; ++j) {
        *y++ = x[i].d * x[i].qs[j];
    }
}
```

**注意 `d` 是 `float` 而非 `ggml_half`**（`block_q8_K` 定义见 `ggml-common.h:371-375`）：
```c
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
```

`q8_K` 是**唯一使用 fp32 scale 的量化类型**。原因：它是纯中间量（不写文件、
不关心体积），用 fp32 可避免 fp16 截断引入的额外误差——在两级量化的乘积链中，
`d` 的精度损失会被放大。`bsums` 用 `int16_t` 而非 `int8_t`：
16 个 int8 量化值的和最大可达 `16*127 = 2032`，超过 int8 范围。

---

## 十、数据验证

### 10.1 验证体系结构（`ggml-quants.c:5330-5667`）

验证的目标是**防御量化过程产生的 NaN/Inf 污染 GGUF 文件**。
设计上分为三层：

**第 1 层：标量验证函数（5330-5373）**

```c
static bool validate_float(float f, size_t i) {
    if (isinf(f)) { fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i); return false; }
    if (isnan(f)) { fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i); return false; }
    return true;
}
static bool isinf_fp16(ggml_fp16_t f) { return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) == 0; }
static bool isnan_fp16(ggml_fp16_t f) { return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) != 0; }
static bool validate_e_e8m0(uint8_t e, size_t i) {
    if (e == 0xff) { fprintf(stderr, "... found invalid e value %d at block %zu\n", e, i); return false; }
    return true;
}
```

**`isinf_fp16`/`isnan_fp16` 的手写位判断**：fp16 布局是 1 符号 + 5 指数 + 10 尾数，
`0x7c00` 是"指数全 1"掩码，`0x03ff` 是尾数掩码。
指数全 1 且尾数为 0 → Inf；指数全 1 且尾数非 0 → NaN。
**为什么不用 `isinf()` 强转 float？** 因为转换本身在 NaN 输入上是实现定义行为，
位判断是唯一可靠的方式。

**`validate_e_e8m0` 的特殊规则**：E8M0 格式中 `0xff` 是**保留值**（表示 NaN），
故 MXFP4 的 `e` 字段不允许取 `0xff`。这是 OCP MX 规范的要求，
而 `quantize_row_mxfp4_ref` 的编码 `(uint8_t)(floorf(log2f(amax)) - 2 + 127)`
在 `amax` 极大时理论上可能产生 `0xff`——验证捕获这种情况。

**第 2 层：宏封装（5375-5407）**

```c
#define VALIDATE_ROW_DATA_D_F16_IMPL(type, data, nb) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_fp16(q[i].d, i)) { return false; } \
    }

#define VALIDATE_ROW_DATA_DM_F16_IMPL(type, data, nb, d, m) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_fp16(q[i].d, i) || !validate_fp16(q[i].m, i)) { return false; } \
    }

#define VALIDATE_ROW_DATA_E_E8M0_IMPL(type, data, nb) ... 
#define VALIDATE_ROW_DATA_DVEC_F16_IMPL(type, data, nb, nr) \  // 用于 nvfp4 的 d[4]
    for (size_t i = 0; i < (nb); ++i) \
        for (size_t j = 0; j < (nr); ++j) \
            if (!validate_fp16(q[i].d[j], i)) return false;
```

四个宏对应四类 block 布局：

| 宏 | 校验字段 | 使用类型 |
|----|---------|---------|
| `VALIDATE_ROW_DATA_D_F16_IMPL` | `d` | `q1_0`,`q2_0`,`q4_0`,`q5_0`,`q8_0`,`q3_K`,`q6_K`,`tq1_0`,`tq2_0`, 全部 IQ（除 `iq1_m`） |
| `VALIDATE_ROW_DATA_DM_F16_IMPL` | `d` + `m` | `q4_1`,`q5_1`,`q2_K`,`q4_K`,`q5_K` |
| `VALIDATE_ROW_DATA_E_E8M0_IMPL` | `e` | `mxfp4` |
| `VALIDATE_ROW_DATA_DVEC_F16_IMPL` | `d[0..nr-1]` | `nvfp4`（`nr = QK_NVFP4/QK_NVFP4_SUB = 4`） |

**注意宏的 `d`/`m` 参数是"提示性"的**（`VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_1, data, nb, d, m)`），
实际代码中硬编码了 `q[i].d`、`q[i].m`。参数存在是为了**让调用点自文档化**
（读者能看出校验的是哪两个字段），但改变了字段名也不会自动生效——
这是宏设计中的一个小缺陷，值得注意。

**第 3 层：主分发函数（5409-5667）**

```c
bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes) {
    if (type < 0 || type >= GGML_TYPE_COUNT) { fprintf(stderr, "%s: invalid type %d\n", __func__, type); return false; }
    if (nbytes % ggml_type_size(type) != 0) {
        fprintf(stderr, "%s: invalid size %zu for type %s (type size = %zu)\n", __func__, nbytes, ggml_type_name(type), ggml_type_size(type));
        return false;
    }
    const size_t nb = nbytes/ggml_type_size(type);
    switch (type) { ... }
    return true;
}
```

**`nbytes % type_size != 0` 的检查**是先决条件——否则 `nb` 计算错误会导致越界读取。

**F16/F32 的 SIMD 快速路径（5445-5521）** 值得单独分析：

```c
case GGML_TYPE_F16:
    {
        const ggml_fp16_t * f = (const ggml_fp16_t *) data;
        size_t i = 0;
#if defined(__AVX2__)
        for (; i + 15 < nb; i += 16) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(f + i));
            __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi16(0x7c00));
            __m256i cmp = _mm256_cmpeq_epi16(vexp, _mm256_set1_epi16(0x7c00));
            int mask = _mm256_movemask_epi8(cmp);
            if (mask) {
                for (size_t j = 0; j < 16; ++j) {
                    if (!validate_fp16(f[i + j], i + j)) return false;
                }
                GGML_UNREACHABLE();
            }
        }
#elif defined(__ARM_NEON)
        ... vld1q_u16 / vandq_u16 / vceqq_u16 / vget_lane_u64 ...
#endif
        for (; i < nb; ++i) {
            if (!validate_fp16(f[i], i)) return false;
        }
    } break;
```

**设计要点**：

1. **向量比较掩码后只在命中时才逐个检查**：`mask` 非零表示这 16 个元素中
   **至少有一个**可能是 Inf/NaN（因为"指数全 1"也包括合法的大数）。
   此时逐个调用 `validate_fp16` 做精确判断。**这是"快速过滤 + 精确确认"的两级策略**。
2. **`GGML_UNREACHABLE()` 的用法很巧**：在逐个检查循环之后加这个宏，
   语义是"如果走到这里说明 mask 误报了"——但实际上 `validate_fp16` 返回 false 会直接 return，
   所以走到 `GGML_UNREACHABLE()` 意味着 mask 命中了但校验都通过。
   这是一种**用于告知编译器/静态分析"此处不会继续"的标注**，
   避免它认为循环可能正常结束从而影响优化。
3. **`vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(cmp, 4)), 0)`（NEON 版）**：
   把 8 个 16-bit 比较结果压缩成 8 个 4-bit 值再取低位，
   等效于 `movemask`（NEON 无直接等价指令，需手工实现）。

**`IQ1_M` 的特殊校验分支（5611-5622）**：

```c
case GGML_TYPE_IQ1_M:
    {
        const block_iq1_m * q = (const block_iq1_m *) data;
        for (size_t i = 0; i < nb; ++i) {
            iq1m_scale_t scale;
            const uint16_t * sc = (const uint16_t *)q[i].scales;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            if (!validate_fp16(scale.f16, i)) return false;
        }
    } break;
```

**唯一需要"先解码再校验"的类型**，因为 `iq1_m` 没有独立的 `d` 字段
（如 8.7 节所分析）。这段代码与 `dequantize_row_iq1_m` 的 `2687` 行**完全重复**——
是一处可以抽取为 helper 的重复，但保留在验证函数中让 `ggml-quants.c` 的验证部分
成为"完全自包含"的模块（不依赖反量化函数的正确性）。

**`NVFP4` 的"无需校验"**（5564-5569）：

```c
case GGML_TYPE_NVFP4:
    {
        // UE4M3 scales are uint8_t — all byte values are valid
        GGML_UNUSED(data);
        GGML_UNUSED(nb);
    } break;
```

注释解释了为什么不需要校验：UE4M3 编码了 0..255 的所有字节值
（`ggml_ue4m3_to_fp32` 对所有输入都有定义），无无效编码。
这与 `E8M0` 不同（`0xff` 是保留的 NaN）。**这是一处正确的"不校验"决策**，
且有注释说明理由——比默默省略好得多。

### 10.2 验证在流水线中的位置

验证在两处被调用（均由 `llama-quant.cpp` 发起）：

```c
// llama-quant.cpp:714-720（单线程）
size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
if (!ggml_validate_row_data(new_type, new_data, new_size)) {
    throw std::runtime_error("quantized data validation failed");
}

// llama-quant.cpp:745-751（多线程，每个 chunk 校验）
const size_t row_size = ggml_row_size(new_type, n_per_row);
void * this_data = (char *) new_data + first_row * row_size;
if (!ggml_validate_row_data(new_type, this_data, this_size)) {
    std::unique_lock<std::mutex> lock(mutex);
    valid = false;
    break;
}
```

**设计含义**：验证是**量化的内建契约**而非可选诊断。
若某次量化产生 NaN（通常源于输入模型的权重本身含 NaN，或 `--allow-requantize` 
对已损坏张量二次量化），llama-quantize 会**立即中止而非写出损坏文件**——
这比在推理阶段出现乱码输出更容易定位问题。

---

## 十一、CPU 侧实现（`quants.c`）与 `vec_dot`

### 11.1 `ggml-cpu/quants.c` 的三层结构

`ggml-cpu/quants.c`（1339 行）的组织非常规范，分三段：

**第 1 段：SIMD 量化入口（44-125 行）**

```c
void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, y, k);
}
void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}
...
void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}
void quantize_row_q8_1_generic(const float * ...) { quantize_row_q8_1_ref(x, y, k); }
```

**注意这里多数函数直接转发到 `_ref`**。看似冗余，但意义在于：

1. `type_traits_cpu.from_float` 需要**签名一致的可取址符号**（`ggml_from_float_t`），
   而 `_ref` 的签名带具体 block 类型（如 `block_q4_0*`），不能直接赋给 `void*` 版本。
2. 这层转发为架构覆盖提供**稳定的挂钩点**：`arch/x86/quants.c` 等可以定义
   `quantize_row_q8_0` 覆盖它（通过链接顺序/LTO 内联）。

`q8_0`/`q8_1` 有 `_generic` 后缀，说明**它们才是真正被 SIMD 替换的目标**
（其他类型在推理中只做反量化，不做量化）。这与 7.3 节的结论一致：
推理期的量化只发生在激活值 → `q8_0`/`q8_1`/`q8_K`。

**第 2 段：`vec_dot` 实现（127-1339 行）**，每个类型一个 `_generic` 函数。

### 11.2 `ggml_vec_dot_q4_0_q8_0_generic`（`ggml-cpu/quants.c:262` 起）

```c
// ggml/src/ggml-cpu/quants.c:225-256
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;          // = 32，注意这里是 QK8_0（伙伴类型）而非 QK4_0
    const int nb = n / qk;
    ...
    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }
    *s = sumf;
}
```

**核心是"先整数累加，最后一次性乘缩放因子"**：

$$s = \sum_i d_{w,i} \cdot d_{a,i} \cdot \sum_j q_{w,j} q_{a,j}$$

浮点乘法从每元素一次降低到每块一次（`sumi*GGML_CPU_FP16_TO_FP32(...)*GGML_CPU_FP16_TO_FP32(...)`），
整数累加在 SIMD 下可 16~32 路并行。

**注意 `qk = QK8_0` 而非 `QK4_0`**：虽然两者数值都是 32，但语义上以**伙伴类型**的块大小为准
（`n` 是量化元素总数，块数由伙伴类型决定）。这个细节在 `q4_0`/`q4_1` 上恰好无差别，
但对 `q1_0`（`QK1_0=128`）与 `q2_0`（`QK2_0=64`）这种块大小与 `q8_0` 不同的类型就关键了：

```c
// ggml_vec_dot_q1_0_q8_0_generic, quants.c:127-176
const int qk = QK1_0;                        // = 128
const int nb = n / qk;
...
for (int i = 0; i < nb; i++) {
    const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);
    float sumi = 0.0f;
    for (int k = 0; k < 4; k++) {            // 1 个 q1_0 块（128 元素）= 4 个 q8_0 块
        const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
        const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
        int sumi_block = 0;
        const uint8_t * GGML_RESTRICT bits = &x[i].qs[k * 4];   // 每个 q8_0 块占 4 字节
        const int8_t  * GGML_RESTRICT qy   = yb->qs;
        for (int b = 0; b < 4; ++b, qy += 8) {
            const unsigned mask = bits[b];
            sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                       +  ((mask & 0x02) ? qy[1] : -qy[1])
                       +  ... ;
        }
        sumi += d1 * sumi_block;
    }
    sumf += d0 * sumi;
}
```

**`q1_0` 的 1 个量化块跨 4 个 `q8_0` 块**，故内层需要 `k` 循环与 `&y[i*4 + k]` 索引。
这里每个元素都是**条件取负**（`mask & 0x01 ? qy[0] : -qy[0]`）——因为 1-bit 量化
只有符号信息，量化值为 `±1`，点积退化为"按位选择符号后求和"。
这正是 `q1_0` 能极快的原因：无乘法，只有符号选择与加法。

`q2_0` 同理（1 个 `q2_0` 块 = 64 元素 = 2 个 `q8_0` 块，`quants.c:177-226`），
但其量化值域是 `{-1,0,1,2}`（需 `-1` 偏移），故有实际乘法（`quants.c:203`）：
```c
sumi_block += ((int)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
```

**`GGML_CPU_FP16_TO_FP32` 而非 `GGML_FP16_TO_FP32`**：前者是 CPU 后端的
F16C 硬件加速版本（`vcvtph2ps`），后者是纯软件位操作版本。

### 11.3 `ggml_vec_dot_q4_K_q8_K_generic` 的 min 项处理（`ggml-cpu/quants.c:696-770`）

这是展示 `q8_K.bsums` 用途的最佳例子：

```c
int sumi = 0;
for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
...
// 主项：整数点积
const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
// min 项：减去 dmin * Σ(q_a) 的加权和
const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
sumf -= dmin * sumi;
```

**数学推导**。`q4_K` 的反量化是 `w = d·sc·q - dmin·m`，`q8_K` 是 `a = d_a·q_a`。
点积展开：

$$s = \sum_j (d \cdot sc_j \cdot q_j - d_{min} \cdot m_j)(d_a \cdot q_{a,j})$$

$$= d \cdot d_a \sum_j sc_j q_j q_{a,j} \;-\; d_{min} \cdot d_a \sum_j m_j q_{a,j}$$

- **第一项**（主项）：`aux32[l]` 按 8 路并行累加 `scale * q8[l] * a[l]`（见 748-768 行的展开循环）。
- **第二项**（min 项）：`m_j` 在 32 元素子块内恒定，且 `q8_K.bsums` 已预存
  **每 16 元素的量化值之和**，故 `Σ_j m_j q_{a,j} = Σ_{groups} m_{group} · bsums[group]`。
  `mins[j/2]` 的下标 `j/2` 是因为 `bsums` 按 16 元素分组、
  而 `q4_K` 的子块是 32 元素（含 2 个 `bsums` 组），故每两个 bsums 共享一个 min。

**`bsums` 的设计价值**：若不预存，第二项需遍历全部 256 个元素做乘法。
预存后只需 16 次乘法（`sumi` 循环）。这就是 `q8_K` 必须携带 `bsums` 的全部理由。

### 11.4 `ggml_vec_dot_iq2_xxs_q8_K_generic` 的格点解码（`ggml-cpu/quants.c:906-946`）

```c
for (int i = 0; i < nb; ++i) {
    const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
    const uint16_t * GGML_RESTRICT q2 = x[i].qs;
    const int8_t   * GGML_RESTRICT q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        memcpy(aux32, q2, 2*sizeof(uint32_t));
        q2 += 4;
        const uint32_t ls = 2*(aux32[1] >> 28) + 1;
        int32_t sumi = 0;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
            const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
            for (int j = 0; j < 8; ++j) {
                sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
            }
            q8 += 8;
        }
        bsum += sumi * ls;
    }
    sumf += d * bsum;
}
*s = 0.125f * sumf;
```

**与反量化版本的对应关系**（对比 `dequantize_row_iq2_xxs`，`2499-2510` 行）：

| 反量化 | 点积 |
|--------|------|
| `db = d * (0.5f + (aux32[1] >> 28)) * 0.25f` | `ls = 2*(aux32[1] >> 28) + 1` |
| `y[j] = db * grid[j] * (signs & kmask ? -1.f : 1.f)` | `sumi += grid[j] * q8[j] * (signs & kmask ? -1 : 1)` |
| 标量乘 fp16 | 整数乘 8-bit |

**`0.125f * sumf` 的来源**：反量化中的 `d * (0.5 + s) * 0.25`，
其中 `(0.5 + s)` 对应点积里的 `ls/2 = (2s+1)/2`，而 `0.25 = 1/4`。
合起来 `d * ls/2 * 1/4 = d * ls * 0.125`。故点积在整数域累加后统一乘 `0.125f`。
**注意 `(0.5f + s)` 是"格点索引加偏移"**——`s = aux32[1] >> 28 ∈ [0,7]`，
故 `0.5 + s ∈ {0.5, 1.5, ..., 7.5}`，即 IQ2_XXS 的隐式 scale 是**半整数**。

**符号乘法用三元表达式而非查表**：`(signs & kmask_iq2xs[j] ? -1 : 1)`。
在 SIMD 版本（`arch/x86/quants.c:3655` 等）中，这会变成
`_mm256_sign_epi8` / `vqnegq_s8` 之类的单指令操作。标量版的写法是为了让
编译器能识别成"条件取负"模式。

---

## 十二、源码技巧与陷阱总结

### 12.1 值得学习的技巧

| 技巧 | 位置 | 价值 |
|------|------|------|
| 位技巧浮点取整（`+1.5×2²³`） | `nearest_int` (621) | 不依赖 FENV 舍入模式，保证跨平台确定性 |
| `memcpy` 类型双关 | `nearest_int` (624)、`q5_0` (227) | 规避严格别名规则；一次整数存储替代多次字节写 |
| 二分查找替代线性扫描 | `best_index_int8` (28) | 非线性电平量化中 4 倍于线性扫描 |
| 交叉相乘比较避免除法 | `make_qx_quants` (686) | `sumlx² > best·suml2` 等价 `scale·sumlx > best` |
| 坐标下降 + 提前退出 | `make_q3_quants` (720-742) | 加权最小二乘的整数分配优化 |
| 闭式双参数最小二乘 | `make_qkx2/3_quants` (851) | 一次求解 scale+min，无需迭代 |
| 12 字节装 8×(6+6) bit | `get_scale_min_k4` (880) | 信息密度零浪费，编解码纯位运算 |
| 用负值编码邻居表偏移 | `kmap_q2xs[i] = -(offset+1)` (3239) | 省一个偏移数组 |
| 5 个三进制数字打包 + `/243*256` | `tq1_0` (2343) | 解码只用乘法和 `>>8`，无需除法 |
| SIMD 掩码快速过滤 + 精确确认 | `ggml_validate_row_data` (5445) | 校验的批量加速 |
| 快速过滤后 `GGML_UNREACHABLE()` | 同上 (5457) | 向编译器表达"此处不会继续" |
| 按距离层级取全格点 | `iq2xs_init_impl` (3243-3251) | 保证候选完整性 |
| qsort 加索引作次序键 | `iq2_compare_func` (2847) | 保证邻居表跨平台确定性 |
| 一次整数累加后统一乘 scale | 全部 `vec_dot` (11.2 节) | 浮点乘法从每元素降至每块 |
| 预存 `bsums` 复用 min 项 | `q8_K` (2795-2801) | 第二项从 256 次乘法降至 16 次 |

### 12.2 阅读时必须注意的陷阱

| 陷阱 | 具体表现 | 位置 |
|------|---------|------|
| **`m` 的符号约定不统一** | `q4_1_ref` 的 `y[i].m = min`，但 `make_qkx*_quants` 返回 `*the_min = -min` | 172 vs 795/876/1072 |
| **`scale` 的符号约定不统一** | `q4_0` 用 `d = max/-8`（负）；`q2_K` 的 `max_scale` 恒正；`q3_K`/`q6_K` 带符号 | 132、904、1244 |
| **`aux[3]` 未初始化** | `memcpy(aux, x[i].scales, 12)` 后 `aux[3]` 未定义，但第 3 行整体赋值 | `dequantize_row_q3_K` (1323) |
| **循环顺序隐式约束** | `y[i].scales[j-4] \|= ...` 要求 `j<4` 的迭代先执行 | `q4_K_ref` (1494-1501) |
| **依赖不变量而非防御检查** | `MIN(15, ...)` 无下界；`MIN(127, ...)` 无下界 | `q4_0_ref` (141)、`q6_K_ref` (1904) |
| **`GROUP_MAX_EPS` 各类型不同** | `q2_K` 的 `sigma2` 缺 `2*` 因子 | 1167（对比 1570/1775） |
| **`nwant` 语义是"距离层级数"而非"元素个数"** | 取最近的 nwant 个距离层级的全部格点 | `iq2xs_init_impl` (3243) |
| **宏参数 `d`/`m` 是自文档而非功能** | 改名不影响实际校验字段 | 5383-5389 |
| **`q8_K` 是唯一 fp32 scale** | 其余类型均为 fp16/E8M0/UE4M3 | `ggml-common.h:372` |
| **`IQ1_M` 无独立 `d` 字段** | 校验时必须自行拼 fp16 | 4937 vs 5617 |
| **Apple 链接器 workaround 影响可读性** | `for (volatile int i = ...)` 禁用展开 | 659、808、1002 |
| **`GGML_ABORT` 自检点** | 打包时若 `kmap[u] < 0` 说明算法有 bug | 3445 |

### 12.3 与设计文档的对应关系

| 本文节 | 对应 [03-量化完整功能设计文档.md](03-量化完整功能设计文档.md) |
|--------|----------------------------------------------------------------|
| 二（工具函数） | 5.1-5.3 节（算法族分类） |
| 三（查找表） | 第八章（查找表与格点） |
| 四（经典族） | 4.2-4.4 节（block 结构、位打包范式） |
| 五（K-quant） | 5.1 节（两级缩放） |
| 六（三值） | 4.2 节（`tq1_0`/`tq2_0` 条目） |
| 七（非线性） | 5.3 节（非线性电平设计原理） |
| 八（IQ 格点） | 5.2 节（IQ 格点机制）、8 章（格点数据底座） |
| 九（q8_K） | 6.4 节（点积联盟表） |
| 十（验证） | 1.1 节 R5 需求、6.2 节（`type_traits` 中的验证） |
| 十一（CPU 侧） | 7.3 节（激活值量化调度）、10.3 节（repack） |

---

## 参考

- `ggml/src/ggml-quants.c`（5667 行）—— 本文主要分析对象
- `ggml/src/ggml-quants.h`（115 行）—— 函数声明
- `ggml/src/ggml-common.h`（1911 行）—— block 结构与查找表
- `ggml/src/ggml-cpu/quants.c`（1339 行）、`ggml-cpu/quants.h`（106 行）
- `ggml/src/ggml-cpu/arch/{x86,arm,riscv,loongarch,powerpc,s390,wasm}/quants.c` —— 各架构 SIMD 覆盖
- `ggml/src/ggml.c:631-945`（`type_traits`）、`7722-7805`（`ggml_quantize_*`）
- 配套设计文档：[03-量化完整功能设计文档.md](03-量化完整功能设计文档.md)
- 容器格式配套：[01-gguf.cpp-源码逐行解析.md](01-gguf.cpp-源码逐行解析.md)、[02-gguf.cpp-完整设计文档.md](02-gguf.cpp-完整设计文档.md)
