# Q4_K_M 裁剪实现 — 测试与断点调试指南

本目录包含针对 `ggml/src/ggml-quants_q4_k_m.c`（从 `ggml-quants.c` 裁剪的 Q4_K 实现）
的独立测试程序和调试方法。

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `q4km_test.c` | 测试程序：5 个单测 + `--debug` 中间量转储 |
| `q4km_stubs.c` | `ggml_row_size`/`ggml_type_size` 桩（脱离 ggml 主库编译用） |
| `q4km_orig_frag.c` | 原实现的 q4_K 片段（符号重命名为 `*_orig`），用于逐位对比 |
| `Makefile` | 构建、运行、对比、启动 lldb |

## 2. 快速开始

```bash
cd <llama.cpp 根目录>

# 构建并运行（-O0 -g，断点友好）
make -f doc/q4km_debug/Makefile run

# 与原实现逐位对比（验证裁剪未改变行为）
make -f doc/q4km_debug/Makefile compare

# 清理
make -f doc/q4km_debug/Makefile clean
```

预期输出（实测值）：

```
sizeof(block_q4_K) = 144 (expect 144)
[test_roundtrip]  rel RMSE = 0.07130  (expect < 0.08)
[test_imatrix_path] rel RMSE = 0.05523  (expect < 0.08)
[test_parity] outputs identical to original implementation
RESULT OK,: 0 failures
```

## 3. 测试覆盖说明

| 测试 | 验证目标 | 断言 |
|---|---|---|
| `test_roundtrip` | 量化→反量化往返误差 | rel RMSE < 0.08 |
| `test_known_values` | 斜坡数据的结构性不变量 | 单调性 + 误差带 |
| `test_imatrix_path` | `quantize_q4_K` imatrix 分发路径 | rel RMSE < 0.08 |
| `test_edge_cases` | 全零/恒定/全正/极小值 | 不崩溃、输出有限 |
| `test_parity_with_orig` | 与原实现**逐位一致** | `memcmp == 0` |

逐位对比的原理：`q4km_orig_frag.c` 是从原 `ggml-quants.c` 自动提取的 q4_K 片段，
所有符号带 `_orig` 后缀（`static` 辅助函数用 `#define` 重命名，公共函数直接改名），
因此两个实现可以链接进同一个程序，用同一份输入数据做 `memcmp`。
若原文件更新，重新生成片段即可（见 `q4km_orig_frag.c` 头部注释）。

## 4. 断点调试

### 4.1 推荐断点位置（对应分析文档 §6）

在 `ggml/src/ggml-quants_q4_k_m.c` 中：

| 断点位置 | 观察什么 |
|---|---|
| `quantize_row_q4_K_ref` 内 `make_qkx2_quants(...)` 调用处 | 每个子块的输入 `x + 32*j`、权重 `weights` |
| `make_qkx2_quants` 中 `if (cur_error < best_error)` | 网格搜索收敛过程，`is`、`this_scale`、`this_min` |
| `quantize_row_q4_K_ref` 内 `get_scale_min_k4(j, ...)` | 12 字节 scales 打包结果 |
| `nearest_int` 入口 | 魔数舍入的行为（观察 `fval` 与返回值） |
| `quantize_row_q4_K_impl` 内 `make_qp_quants(...)` 调用处 | 块级 6-bit 量化的输入 `scales[]`/`sw[]` |
| `make_qp_quants` 中坐标下降循环 `if (slx*slx*suml2 > ...)` | 精修是否接受新量化级 |

### 4.2 lldb（macOS）

```bash
make -f doc/q4km_debug/Makefile lldb
# 等价于手动：
lldb build-q4km-debug/q4km_test --debug
(lldb) b make_qkx2_quants          # 函数断点（static 函数也可断）
(lldb) b ggml-quants_q4_k_m.c:150  # 文件:行号 断点
(lldb) run
(lldb) p j                         # 当前子块索引
(lldb) p *x                        # 打印子块数据
(lldb) memory read -f f -c 32 x    # 以 float 格式查看 32 个权重
(lldb) watchpoint set variable best_error   # 数据断点：误差被改进时停下
```

常用技巧：

- **条件断点**只在第 j 个子块停下：`b make_qkx2_quants -c j == 3`
  （或 `break set --file ggml-quants_q4_k_m.c --line 150 --condition 'j==3'`）
- **观察打包位运算**：在 `get_scale_min_k4` 处断点后 `p/x *q@12` 查看 12 字节 scales。
- **对照参考输出**：先跑 `./build-q4km-debug/q4km_test --debug`，记住 `scales[12]`
  与 `qs[0..15]` 的期望值（斜坡数据是确定性的），断点单步时逐一比对。

### 4.3 gdb（Linux）

```bash
gdb --args build-q4km-debug/q4km_test --debug
(gdb) b make_qkx2_quants
(gdb) run
(gdb) p j
(gdb) x/32fw x          # float 格式查看子块
(gdb) watch best_error  # 数据断点
```

### 4.4 VS Code / Xcode

VS Code（launch.json 片段）：

```json
{
  "type": "cppdbg", "request": "launch", "name": "q4km_test",
  "program": "${workspaceFolder}/build-q4km-debug/q4km_test",
  "args": ["--debug"], "cwd": "${workspaceFolder}",
  "MIMode": "lldb"
}
```

Xcode：直接把 `q4km_test.c`、`ggml-quants_q4_k_m.c`、`q4km_stubs.c` 拖入一个
Command Line Tool 项目，include 路径加 `ggml/src` 与 `ggml/include`。

### 4.5 调试路线建议

1. **先跑 `--debug` 模式**，读一遍转储（d/dmin 的 f16 位、12 字节 scales、
   解码出的 8 对 scale6/min6、qs 半字节），建立"输入→中间量→输出"的地图。
2. **斜坡数据是最佳调试输入**（`debug_dump_block` 用的就是它）：值域 [-1,1]
   线性递增，人工可预测量化级分布，任何异常一眼可见。
3. **验证 12 字节打包**时，在 `get_scale_min_k4` 断点对照 §2.1 的布局公式；
   `j<4` 与 `j>=4` 两个分支各断一次。
4. **验证两级缩放**时，检查 `(x + dm)/d` 用的 `d`/`dm` 是**半精度化后**的值
   （即先 `GGML_FP32_TO_FP16` 再 `GGML_FP16_TO_FP32`），这是最容易被误解的点。
5. 改动算法后跑 `make compare`，逐位对比能捕获任何行为差异（包括浮点
   归约顺序变化）。

## 5. 常见问题

- **链接错误 `ggml_row_size` 未定义**：测试用 `q4km_stubs.c` 提供桩；若接入真实
  ggml 库，改为链接 `ggml.c` 并去掉桩。
- **`static` 函数断点断不上**：确认 Makefile 用了 `-O0 -g`（默认即是）；内联后
  的 `nearest_int` 可改断调用处。
- **为什么 imatrix 路径误差和 ref 差不多**：随机 imatrix 对精度帮助有限；
  真实模型中 imatrix 来自激活统计，才会显著降低误差。
