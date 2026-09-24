# 本地测试验证结果（保留存档）

> 验证日期：2026-09-13
> 环境：macOS (Darwin 23.6.0, x86_64)，clang + LLDB，`-O0 -g`
> 结论：**全部测试通过**（单测 5/5、逐位对比通过、LLDB 断点命中验证通过）

## 1. 单元测试（make run）

命令：`make -f doc/q4km_debug/Makefile run`，退出码 0。

```
sizeof(block_q4_K) = 144 (expect 144)

=== debug: single super-block dump ===
d(f16 bits)=0x143e  dmin(f16 bits)=0x240c
scales[12]: 0f 4f cf cf 3f 2f 1f 0f 0f 0f 00 0f
sub-block (scale, min) pairs decoded:
  j=0  scale6=15  min6=63
  j=1  scale6=15  min6=47
  j=2  scale6=15  min6=31
  j=3  scale6=15  min6=15
  j=4  scale6=15  min6= 0
  j=5  scale6=31  min6= 0
  j=6  scale6=48  min6= 0
  j=7  scale6=63  min6= 0
qs[0..15]: 00 00 11 11 22 22 33 33 44 44 55 55 66 66 77 77
  (低半字节=L[i], 高半字节=L[i+32])
rel RMSE = 0.01647

[test_roundtrip]  rel RMSE = 0.07130  (expect < 0.08)     ✅
[test_known_values] ramp rel RMSE = 0.01647               ✅
  - 斜坡反量化单调性检查通过
  - d = 0.001036（f16 解码），dmin_raw = 0x240c
[test_imatrix_path] rel RMSE = 0.05523  (expect < 0.08)   ✅
[test_edge_cases]                                         ✅
  - 全零块：反量化全 0（无 NaN/Inf）
  - 恒定块 rel RMSE = 0.00036
  - 全正块 rel RMSE = 0.02948
  - 极小值块（1e-20 量级）：输出全部有限
RESULT OK,: 0 failures
```

## 2. 与原实现逐位对比（make compare）

命令：`make -f doc/q4km_debug/Makefile compare`，退出码 0。

```
[test_parity] outputs identical to original implementation
RESULT OK,: 0 failures
```

裁剪版 `ggml-quants_q4_k_m.c` 的 `quantize_row_q4_K_ref` 输出与原
`ggml-quants.c` 中同一函数在相同输入（确定性 LLM 分布数据，8 个超块）下
**逐位一致**（memcmp 通过）；反量化输出亦逐位一致。

## 3. 断点调试验证（LLDB）

命令（README §4.2 的批处理等价形式）：

```
lldb -s cmds.txt -- build-q4km-debug/q4km_test --debug
  b make_qkx2_quants
  run
```

实测断点命中输出：

```
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: q4km_test`make_qkx2_quants(n=32, nmax=15,
        x=0x00007ff7bfefea10, weights=0x00007ff7bfefe390,
        rmin=-1, rdelta=0.100000001, nstep=20, use_mad=false)
        at ggml-quants_q4_k_m.c:49:17
```

断点解析到正确源码行（`-O0 -g` 有效），函数签名参数与源码分析文档 §3.3
的描述完全一致（n=32、nmax=15、rmin=-1、rdelta=0.1、nstep=20、MSE 模式）。

注：交互式 `lldb` 通过管道喂命令时若进程未正常退出会挂起，建议用
`lldb -s <命令文件> </dev/null` 批处理模式，或按 README 用交互终端。

## 4. 验证过程中发现并修复的问题

| 问题 | 现象 | 修复 |
|---|---|---|
| 裁剪文件引用 `ggml_row_size()`（定义在 ggml.c） | 独立链接失败 | 新增 `q4km_stubs.c` 提供 Q4_K 正确行大小桩 |
| 对比片段缺少 `ggml-impl.h`/`GROUP_MAX_EPS`/qkx3 与 qp 辅助函数 | orig 片段编译失败 | 重新提取片段（621–887、993–1147、1455–1640 行）并补 include 与宏 |

## 5. 复现方式

```bash
cd <llama.cpp 根目录>
make -f doc/q4km_debug/Makefile run       # 单测 + debug 转储
make -f doc/q4km_debug/Makefile compare   # 与原实现逐位对比
```

构建产物位于 `build-q4km-debug/`（`make clean` 可清除）。
