# variants：三组追加实验的可复现源码

> 对应 [04-gdb-trace.md](../04-gdb-trace.md) §11–§13。
>
> 这些文件是 `examples/simple/simple-backend.cpp` 的最小改动版本，用于验证 CUDA 路径切换、跨 backend copy 和内存复用。

---

## 0 文件说明

| 文件 | 改动 | 验证目标 |
|---|---|---|
| `exp1_rowsB4.cpp` | `rows_B` 3 → 4 | `MUL_MAT` 从 MMVF 切到 cuBLAS；`cublasCreate`、4 MiB workspace、`cublasSgemm` 参数 |
| `exp2_cross_backend.cpp` | `a` 预分配到 CPU buffer，`b` 预分配到 CUDA buffer | `n_inputs=1`、图副本中的 dep/copy 节点、异步拷贝降级、gallocr 只为 copy+result 分配 |
| `exp3_mlp3.cpp` | 三层 `MUL_MAT`，`H1=H3=128` | free-list 复用、尾部块保留策略、峰值显存 |
| `exp3b_mlp3_silu.cpp` | 三层 `MUL_MAT` + 两个 `silu` | inplace 复用、`ggml_gallocr_free_extra_space` |

---

## 1 编译

在云端 `/home/ggml` 下（与 `build-debug` 同源）：

```bash
cd /home/ggml/variants

CXX=/usr/bin/c++
INC="-I/home/ggml/include"
FLAGS="-DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -g -O0 -std=gnu++17"
LIBS="/home/ggml/build-debug/src/libggml.so.0.25.3 \
      /home/ggml/build-debug/src/libggml-cpu.so.0.25.3 \
      /home/ggml/build-debug/src/ggml-cuda/libggml-cuda.so.0.25.3 \
      /home/ggml/build-debug/src/libggml-base.so.0.25.3"
RPATH="-Wl,-rpath,/home/ggml/build-debug/src:/home/ggml/build-debug/src/ggml-cuda"

for f in exp1_rowsB4 exp2_cross_backend exp3_mlp3 exp3b_mlp3_silu; do
    $CXX $FLAGS $INC -o $f $f.cpp $LIBS $RPATH
done
```

也可以用 `build-debug/examples/simple/CMakeFiles/simple-backend.dir/link.txt` 里的链接命令作为模板。

---

## 2 运行

```bash
./exp1_rowsB4
./exp2_cross_backend
./exp3_mlp3
./exp3b_mlp3_silu

# 调度视图
GGML_SCHED_DEBUG=2 ./exp2_cross_backend
GGML_SCHED_DEBUG=2 ./exp3_mlp3
```

预期输出：

```text
exp1: result ne=[4,4]，第 4 列 [18 7 8 20]
exp2: a buffer = CPU, b buffer = CUDA0，结果与原始 3x4 矩阵一致
exp3: t2 ne=[128,3] first=524288 expected=524288
exp3b: t2 ne=[128,3] first=524288
```

---

## 3 gdb 脚本

三组实验使用的脚本已内联在 [04-gdb-trace.md](../04-gdb-trace.md) 对应章节。核心断点：

```gdb
# 实验一
break ggml_cuda_mul_mat
break ggml_cuda_mul_mat_vec_f
break ggml_cuda_mul_mat_cublas
break cublasCreate_v2
break cublasSetWorkspace_v2
break cublasSgemm_v2

# 实验二
break ggml_backend_sched_alloc_splits
break ggml_backend_sched_compute_splits
break ggml_backend_cuda_cpy_tensor_async
break ggml_backend_tensor_copy
break ggml_backend_cuda_buffer_set_tensor

# 实验三
break ggml_dyn_tallocr_alloc
break ggml_dyn_tallocr_free_bytes
break ggml_gallocr_free_node
break ggml_gallocr_free_extra_space
break ggml_vbuffer_tensor_alloc
```

---

## 4 注意事项

1. **cuBLAS 符号带 `_v2` 后缀**：`cublasCreate_v2`、`cublasSgemm_v2`、`cublasSetWorkspace_v2`。直接 `break cublasCreate` 会 pending 不命中。
2. **先 `start` 再下断点**：共享库在 `main` 前已加载，`start` 之后设置断点可以立即解析到正确位置（避免源码行断点 pending 不命中）。
3. **`ggml_dyn_tallocr_alloc` 首次调用时 `n_chunks = 0`**：打印 `chunks[0]` 前要判断 `n_chunks > 0`。
4. **地址每次不同**：只比较相对偏移，不比较绝对地址。
5. **`exp2` 的 `0x200` 间隔**：来自 CUDA 分配器对小块的 512 B 取整，不是 gallocr 的对齐。
