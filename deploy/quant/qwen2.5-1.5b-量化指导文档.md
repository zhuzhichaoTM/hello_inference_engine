# Qwen2.5-1.5B-Instruct 量化为 Q4_K_M GGUF 指导文档

> 目标产物：`qwen2.5-1.5b-instruct—new—q4_k_m.gguf`（约 986 MB，5.08 BPW）
> 执行环境：macOS (Darwin 23.6.0) + llama.cpp 本地源码 + 本地 venv（经 uv 管理）
> 实测路径：本仓库 `~/Documents/agent/deploy/llama/llama_test/`

---

## 1. 量化原理简述

Q4_K_M 是 llama.cpp 的 **K-quant 混合精度 4-bit** 格式：

| 张量类型 | 量化方式 | 典型用途 |
|---|---|---|
| `attn_q / attn_k / attn_output / ffn_gate / ffn_up` | Q4_K | 大部分 4bit 权重 |
| `attn_v / ffn_down` | **Q6_K**（自动提升精度） | 对误差敏感的降维/下投影层 |
| `attn_norm / ffn_norm` | F32 | 保留原始精度 |
| bias | F32 | 保留原始精度 |

Q4_K_M 相比纯 Q4_K 将关键层提升到 6bit， perplexity 损失显著更小，是 1.5B 量级模型的**性价比最优默认选择**。

量化不是"逐层直接压"，而是 llama.cpp 内部有一套 `tensor_type → GGML 类型` 的映射规则（见 `tools/quantize/` 源码），对 `ffn_down`、`attn_v` 这类误差敏感层自动用更高精度。

## 2. 环境要求

| 组件 | 本机实际版本 | 说明 |
|---|---|---|
| llama.cpp 源码 | 本地 `./llama.cpp` | 含 `convert_hf_to_gguf.py` 与 CMake 构建 |
| `llama-quantize` | `./llama.cpp/build/bin/llama-quantize` | C++ 量化二进制 |
| Python | 3.11.12（本地 `./venv`） | 由 `uv` 管理 |
| transformers | 5.13.0 | 仅用于 tokenizer/config 读取 |
| torch | 2.2.2 | 仅转换时使用 |
| 源模型 | HF safetensors（BF16，~3.0 GB） | `./qwen2.5-1.5b-instruct/` |

### 2.1 用 uv 准备 Python 环境（借用本地 venv）

```bash
# 若 venv 不存在，用 uv 创建（指定 python 3.11）
uv venv venv --python 3.11

# 用 uv 向该 venv 安装 llama.cpp 转换依赖
uv pip install --python ./venv/bin/python \
    -r llama.cpp/requirements.txt

# 验证
./venv/bin/python -c "import torch, transformers, gguf; print('ok')"
```

> 关键点：`uv pip install --python ./venv/bin/python` 明确指定目标 venv，避免 uv 误装到全局或其他环境。`requirements.txt` 位于 llama.cpp 根目录，内含 `gguf`、`sentencepiece`、`safetensors` 等转换必备依赖。

### 2.2 编译 llama.cpp（若尚未编译）

```bash
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
```

## 3. 量化完整流程（三步）

### 第一步：HuggingFace safetensors → GGUF (F16 中间格式)

Q4_K_M 量化**必须**从 16bit/32bit 精度出发，不允许从已有低比特模型 requantize（会严重损失精度），所以先产出 F16 中间文件：

```bash
./venv/bin/python llama.cpp/convert_hf_to_gguf.py \
    ./qwen2.5-1.5b-instruct \
    --outfile ./qwen2.5-1.5b-instruct-f16.gguf \
    --outtype f16
```

实测输出：

```
INFO:gguf.gguf_writer:qwen2.5-1.5b-instruct-f16.gguf: n_tensors = 338, total_size = 3.1G
INFO:hf-to-gguf:Model successfully exported to qwen2.5-1.5b-instruct-f16.gguf
```

生成 `qwen2.5-1.5b-instruct-f16.gguf`（3.09 GB，338 个张量，转换约 2 分钟）。

> 注意：本仓库实际是通过 `uv run` 方式借 venv 执行的，本质等价于直接调 `./venv/bin/python`。若用 uv 一体化方式：
> ```bash
> uv run --python ./venv/bin/python llama.cpp/convert_hf_to_gguf.py ...
> ```

### 第二步：F16 → Q4_K_M 量化

```bash
./llama.cpp/build/bin/llama-quantize \
    qwen2.5-1.5b-instruct-f16.gguf \
    qwen2.5-1.5b-instruct—new—q4_k_m.gguf \
    Q4_K_M
```

实测输出（节选）：

```
[ 338/ 338] blk.27.ffn_up.weight    - [1536, 8960, 1, 1], type = f16,
    converting to q4_K .. size = 26.25 MiB -> 7.38 MiB
llama_model_quantize_impl: model size  = 2944.68 MiB (16.00 BPW)
llama_model_quantize_impl: quant size  =  934.69 MiB ( 5.08 BPW)
llama_quantize: quantize time = 80037.48 ms
```

最终产物：**`qwen2.5-1.5b-instruct—new—q4_k_m.gguf`（986,048,096 字节 ≈ 940 MB）**，压缩率约 3.1×。

### 第三步：验证

```bash
# 强制 CPU 推理（Intel Mac 上 Metal 常见报错，见 §4.2）
./llama.cpp/build/bin/llama-cli \
    -m qwen2.5-1.5b-instruct—new—q4_k_m.gguf \
    -p "你好，请用一句话介绍你自己。" \
    -n 64 -t 4 -no-cnv -st -ngl 0 </dev/null
```

实测结果：

```
> 你好，请用一句话介绍你自己。
你好，我是阿里云开发的一款语言模型，可以回答问题、创作文字，也能表达观点、撰写故事。

[ Prompt: 56.4 t/s | Generation: 14.3 t/s ]
```

模型输出语义正确、中文流畅，量化质量验证通过。

## 4. 常见问题（踩坑实录）

### 4.1 `--no-cnv` vs `-no-cnv`
新版 llama-cli 用**单横线** `-no-cnv`，`--no-cnv` 会报 `invalid argument`。

### 4.2 Metal 报错 `e00002bd: Internal Error`
Intel 集成显卡（本机 Intel Iris/UHD）跑 llama.cpp Metal 后端时，Q4_K_M 反量化 kernel 触发 GPU 崩溃，出现：

```
E ggml_metal_synchronize: error: command buffer 0 failed with status 5
E error: Internal Error (e00002bd:Internal Error)
```

**解决办法**：加 `-ngl 0` 强制纯 CPU 推理。实测 1.5B Q4_K_M 在 4 线程 CPU 上可达 14 t/s 生成速度，完全可用。

### 4.3 quantize 命名里的特殊字符
文件名中的 `—`（em dash）是全角字符，注意在 shell 中**务必加引号**或整体作为一个参数传入，避免被 shell 拆分。

### 4.4 量化必须从 F16/F32 出发
`llama-quantize` 默认拒绝 requantize 已量化模型（除非加 `--allow-requantize`，但会明显掉精度）。正确流程永远是 `safetensors → F16 GGUF → Q4_K_M`。

## 5. 后续可选优化

| 方向 | 命令/方法 | 说明 |
|---|---|---|
| 更高质量 | `--imatrix imatrix.gguf` | 先用 `llama-imatrix` 在校准数据上生成重要性矩阵，再量化，可显著降低 Q4 精度损失 |
| 保持 output 层精度 | `--output-tensor-type Q6_K` | 手动指定 output.weight 用 6bit |
| 保留 embedding 精度 | `--token-embedding-type Q8_0` | 手动指定 token emb 用 8bit |
| 更小体积 | `Q4_K_S` / `IQ4_XS` | 再省 ~10% 体积，精度略降 |
| 更高精度 | `Q5_K_M` / `Q6_K` | 体积增大 15–30%，精度接近 F16 |

## 6. 产物清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `qwen2.5-1.5b-instruct/model.safetensors` | 3.09 GB | 原始 HF 权重（BF16） |
| `qwen2.5-1.5b-instruct-f16.gguf` | 3.09 GB | 中间 F16 GGUF（338 tensors） |
| **`qwen2.5-1.5b-instruct—new—q4_k_m.gguf`** | **986 MB** | **最终量化产物（5.08 BPW）** |
