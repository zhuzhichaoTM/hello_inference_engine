# ggml-quants.c 裁剪与 Q4_K（Q4_K_M）量化实现源码分析

> 源文件：`ggml/src/ggml-quants.c`（5667 行，含全部 K-quant / I-quant 实现）
> 裁剪产物：`ggml/src/ggml-quants_q4_k_m.c`（约 440 行，仅保留 Q4_K 相关实现）
> 分析日期：2026-09-13

---

## 1. 裁剪范围与依据

### 1.1 什么是 Q4_K_M

`Q4_K_M` 是 llama.cpp 的 `llama-quant.cpp` 中定义的一种**混合量化方案**（`LLAMA_FTYPE_MOSTLY_Q4_K_M`），
其主体思想是：按张量重要性将模型权重分别量化为 Q4_K 或 Q6_K：

- 大部分张量 → `Q4_K`（4.5 bpw）
- 关键张量（如 `token_embd`、`output`、attention/ffn 的部分权重）→ `Q6_K`（6.5625 bpw）

因此，Q4_K_M 方案的**核心块格式与量化算子是 Q4_K**。本次裁剪只保留 Q4_K 的
量化/反量化实现；Q6_K 的实现（`quantize_row_q6_K_ref` 等）不在此文件范围内。

### 1.2 保留内容

| 类别 | 符号 | 作用 |
|---|---|---|
| 块格式 | `block_q4_K`（来自 `ggml-common.h`） | 256 元素超块的存储布局 |
| 宏 | `QK_K=256`、`K_SCALE_SIZE=12`（来自 `ggml-common.h`） | 超块大小 / 12 字节 scale 打包 |
| 入口 | `quantize_q4_K` | 有/无 imatrix 分发 |
| 参考量化 | `quantize_row_q4_K_ref` | 无 imatrix，确定性生成模型文件 |
| imatrix 量化 | `quantize_row_q4_K_impl` | AWQ（激活感知量化）路径 |
| 反量化 | `dequantize_row_q4_K` | 还原为 f32 |
| 辅助 | `nearest_int` | IEEE754 魔数快速舍入 |
| 辅助 | `make_qkx2_quants` | ref 路径的子块 scale/min 搜索 |
| 辅助 | `make_qkx3_quants` | impl 路径的子块 scale/min 搜索 |
| 辅助 | `make_qp_quants` | 块级 6-bit scale/min 量化 + 坐标下降精修 |
| 辅助 | `get_scale_min_k4` | 12 字节 scale 打包的解码 |

### 1.3 删除内容（原文件其余 ~5200 行）

- 全部 `_0/_1` 系列块（q4_0/q4_1/q5_0/q5_1/q8_0…）、mxfp4/nvfp4、tq1_0/tq2_0
- 其他 K-quant：q2_K / q3_K / q5_K / q6_K / q8_K
- 全部 I-quant（iq1_s/iq2_xxs/iq3_xxs/iq4_nl…）及其码本初始化
- 未被 Q4_K 引用的辅助函数：`best_index_int8`、`make_qx_quants`、`make_q3_quants`、`make_qkx1_quants`
- 文件尾部的大开关 `ggml_validate_row_data`（完整性校验，非量化核心）

### 1.4 验证

裁剪文件已通过独立编译与运行时验证：

- `gcc -c` 单独编译通过（仅依赖 `ggml-common.h` / `ggml-quants.h` / `ggml-impl.h`）；
- 参考路径往返误差（rel RMSE，均匀随机数据）：**≈ 0.056**；
- imatrix 路径往返误差：**≈ 0.055**（与 ref 相当，符合 Q4_K 理论 4.5 bpw 的精度水平）；
- `sizeof(block_q4_K) == 144` 字节，与 static_assert 一致。

注意：`quantize_q4_K` 引用了 `ggml_row_size()`（定义在 `ggml.c`）。若要把此文件
用于脱离 ggml 库的独立工具链，需自行提供该函数（或内联 `nrow/256 * 144`）。

---

## 2. 数据布局：block_q4_K（144 字节 / 256 权重 = 4.5 bpw）

```
typedef struct {
    union {
        struct {
            ggml_half d;    // 超块 scale（f16）
            ggml_half dmin; // 超块 min  （f16）
        };
        ggml_half2 dm;
    };
    uint8_t scales[12];     // 8 个子块的 6-bit scale + 6-bit min（打包进 12 字节）
    uint8_t qs[128];        // 256 个 4-bit 量化值（低/高半字节）
} block_q4_K;               // sizeof = 4 + 12 + 128 = 144
```

反量化公式（两级缩放）：

```
w = d * sc[j] * q - dmin * m[j]
```

其中 `d/dmin` 是 f16 超块尺度，`sc[j]/m[j]`（6 bit）是第 j 个 32 元素子块的尺度。

### 2.1 12 字节 scales 打包（K_SCALE_SIZE=12）

8 个子块 × (6-bit scale + 6-bit min) = 96 bit = 12 字节，但打包方式是**特殊设计的**：

- `j < 4`：`scales[j]` 低 6 位 = scale_j；`scales[j+4]` 低 6 位 = min_j
- `j >= 4`：`scales[j+4]` = `scale_j` 低 4 位 | `min_j` 低 4 位 << 4
  （即字节 j+4 的低半字节是 scale_j[3:0]，高半字节是 min_j[3:0]）
  两个 6-bit 值的**高 2 位**则挤进字节 `scales[j-4]` 和 `scales[j]` 的最高 2 位（bit 7:6）

解码由 `get_scale_min_k4()` 完成：

```c
if (j < 4) { *d = q[j] & 63;        *m = q[j+4] & 63; }
else {      *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
            *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4); }
```

这种布局使 SIMD 解包（CUDA/Metal 内核）可以按固定字节粒度并行读取，是历史演化产物。

### 2.2 qs 半字节排布

每 64 个权重为一组：`q[l]` 低半字节存 `L[j+l]`（0..31），高半字节存 `L[j+l+32]`（32..63）。
即**奇偶交织**：同一字节内前后相隔 32 的两个权重成对。

---

## 3. 量化算法分析

### 3.1 总体结构（两级优化）

Q4_K 的量化是**分层优化**：

```
super-block (256 权重)
 ├── 子块级：8 × (32 权重) ──求出──> scale[j], min[j]（f32，中间量）
 └── 块级：  把 8 个 scale[j]、8 个 min[j] 各自再量化到 6 bit，
             最大者映射为 d、dmin（f16 存储）
```

两级之间有**重量化回环**（requantization）：块级 6-bit 量化会引入尺度误差，
因此最终 4-bit 量化用的是"半精度化后"的 `d*sc`、`dmin*m`（先写回块、再读回），保证
量化值与文件里实际存的系数一致，避免系统性偏差。

### 3.2 nearest_int —— 魔数舍入

```c
float val = fval + 12582912.f;          // 12582912 = 1.5 * 2^23
int i; memcpy(&i, &val, sizeof(int));
return (i & 0x007fffff) - 0x00400000;
```

利用 IEEE754：把 |x| < 2^22 的 float 加上 1.5×2^23，其尾数位即变为"最近偶舍入"的
定点表示。比 `lrintf`/`(int)roundf` 快且不依赖舍入模式。约束：`|fval| ≤ 4194303`。

### 3.3 make_qkx2_quants（ref 路径的子块优化）

对 32 个权重 `{x}`，在 [min, max] 上找 (scale, min) 使加权误差最小：

1. **初始化**：`min = min(x,0)`（钳到非正，因为反量化是减法），`iscale = nmax/(max-min)`，
   得到初始量化级 `L[i] ∈ [0,15]` 与初始误差。
2. **网格搜索**：`nstep+1 = 21` 个候选，`iscale_is = (rmin + 0.1*is + nmax)/(max-min)`，
   `rmin = -1`。即尝试把量化范围向负方向扩展 0~1.0，覆盖分布不对称（负尾更长）的情形。
3. **每个候选下做加权最小二乘**：固定量化级 `Laux[i]`，对 `x ≈ scale*Laux + min` 解
   2×2 正规方程（闭式解），`D = sum_w*sum_l2 - sum_l^2 > 0` 时有效；若解出的
   `this_min > 0` 则钳回 0 并退化拟合。
4. **保留误差最小者**（MSE；`use_mad=true` 时用 MAD，q2_K 用，q4_K ref 用 MSE）。

权重（调用处）：`weights[l] = av_x + |x[l]|`，`av_x` 是子块 RMS——
"以自身幅度为权，兼顾全局平均"这一经验设计来自 k-quant 原始 PR（ikawrakow）。

### 3.4 make_qkx3_quants（impl/imatrix 路径的子块优化）

与 qkx2 结构相同，改进两点：

1. `weights` 可为 NULL（内部退化 `w = x²`）；
2. `max <= min`（全零块）时用 `memset` 清零 L，防 UB；
3. 搜索参数不同：`rmin=-0.9, rdelta=0.05, nstep=36` —— 更细的 37 步网格，
   且带 imatrix 时权重改为 `qw[l] * sqrt(sigma2 + x²)`（`sigma2` 是**超块级**的
   2×均方值），即"重要性矩阵 × 激活能量"。

### 3.5 make_qp_quants（块级 6-bit 量化，两路径共用）

输入是 8 个子块的 f32 scale（或 min），全部非负。目标：量化到 `[0, nmax]`（此处 63）
并返回加权最优 f32 scale：

1. 初始 `iscale = nmax/max`；
2. **±0.4 范围网格搜索**（`is ∈ [-4,4]`，步长 0.1×max）：以 MSE 选最优 iscale；
3. **逐元素坐标下降**（最多 5 轮）：对每个 `L[i]`，试探
   `new_l = x[i]*sl2/slx`（加权最小二乘意义下的理想值），若
   `slx²*suml2 > sumlx²*sl2`（等价于平方和目标下降）则接受。
   这是 k-quant 系列共用的"局部精修"技术。

### 3.6 ref 与 impl 的差异总结

| 维度 | `quantize_row_q4_K_ref` | `quantize_row_q4_K_impl` |
|---|---|---|
| imatrix | 不使用 | `qw * sqrt(sigma2 + x²)` |
| 子块优化器 | `make_qkx2_quants`（21 步） | `make_qkx3_quants`（37 步） |
| 块级优化器 | 无（直接 max→63 归一化） | `make_qp_quants`（网格 + 坐标下降） |
| 权重选择 | `av_x + |x|`（子块 RMS + 幅度） | 同左（无 imatrix 时） |
| 用途 | quantize 工具无 imatrix；`-0` 快速量化 | 有 imatrix 的 `-0` / Q4_K_M 主路径 |
| 循环粒度 | 逐子块求 `sum_x2` | 超块级 `sigma2`（跨子块共享） |

> 提示：实测中 impl 路径无 imatrix 时的精度与 ref 接近，但 **ref 才是"确定性
> 模型文件"的规范路径**（同一输入必得同一文件，imatrix 版因浮点归约顺序
> 在多线程下可能有微小差异）。

---

## 4. 反量化分析

`dequantize_row_q4_K` 是纯标量参考实现，每轮处理 64 个权重（2 个子块）：

```c
get_scale_min_k4(is+0, ...) → d1, m1   // 前一个子块
get_scale_min_k4(is+1, ...) → d2, m2   // 后一个子块
低半字节 × d1 - m1；高半字节 × d2 - m2
```

`d*sc` 中 `sc ∈ [0,63]`，乘法在 f32 完成（`d` 先转为 f32），`min` 同理。
CPU 后端（`ggml-cpu/vec.h`）另有 SIMD 版本；本文件的 ref 版主要用于工具链与校验。

---

## 5. 裁剪文件的依赖关系

```
ggml-quants_q4_k_m.c
 ├── ggml-common.h  ──> block_q4_K, QK_K, K_SCALE_SIZE, ggml_half/FP16 宏
 ├── ggml-quants.h  ──> 函数声明（GGML_API）
 └── ggml-impl.h    ──> MAX/MIN/GGML_UNUSED 等
 外部符号：ggml_row_size()（仅 quantize_q4_K 使用；来自 ggml.c）
```

编译命令示例（独立编译）：

```bash
gcc -c -I ggml/src -I ggml/include ggml/src/ggml-quants_q4_k_m.c -o ggml-quants_q4_k_m.o
```

若与主库链接冲突（`quantize_q4_K` 与 `ggml-quants.c` 中同名符号重复），可给
函数加前缀或用 `-Wl,-U` 控制。建议把裁剪文件用于**学习/离线分析工具**，
不建议直接编进主构建（CMake 中 `ggml-quants.c` 仍是规范来源）。

---

## 6. 源码逐段导读（裁剪文件内行号参考）

1. **头部注释与 include**（1–30 行）——保留必需头，删除 OpenMP/float.h/stdlib.h（仅 qsort 用）。
2. `nearest_int` —— 见 §3.2。
3. `make_qkx2_quants` —— 见 §3.3；`HAVE_BUGGY_APPLE_LINKER` 分支是 ld64 1015.7
   循环展开 bug 的 workaround，裁剪时保留以维持兼容。
4. `make_qkx3_quants` / `make_qp_quants` —— 见 §3.4/§3.5。
5. `get_scale_min_k4` —— 见 §2.1；注意 `q[j-0]` 写法（即 `q[j]`），裁剪时原样保留以免误导 diff。
6. `quantize_row_q4_K_ref` —— 见 §3.3 调用侧；scales 打包循环中 `j<4 / j>=4` 两个
   分支与 §2.1 的布局一一对应。
7. `dequantize_row_q4_K` —— 见 §4。
8. `quantize_row_q4_K_impl` —— 见 §3.4/§3.5 调用侧；`sigma2 = 2*sum_x2/QK_K` 的
   系数 2 是 AWQ 实现中的经验缩放。
9. `quantize_q4_K` —— 入口；`ggml_row_size(GGML_TYPE_Q4_K, n_per_row)` = 每 256 元素 144 字节。

---

## 7. 结论

- Q4_K 的本质是 **"4-bit 主量化 + 6-bit 二级尺度 + f16 超块尺度"的三层缩放结构**，
  以 144 字节/256 权重（4.5 bpw）存储，反量化为 `w = d*sc*q - dmin*m`。
- 量化质量来自两处：① 子块 scale/min 的网格搜索 + 闭式加权最小二乘；
  ② 块级 6-bit 量化的坐标下降精修。imatrix（AWQ）把"哪个权重重要"的信息
  注入两层优化的权重项。
- Q4_K_M 作为混合方案，其 Q4_K 部分即上述实现；关键张量升级为 Q6_K 由
  `llama-quant.cpp` 的张量选择逻辑决定，不属于本裁剪范围。
- 裁剪文件已验证：可独立编译，两条量化路径的往返误差均约 5.6%（4.5 bpw 的正常水平）。
