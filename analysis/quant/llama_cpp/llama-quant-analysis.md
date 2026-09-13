# llama-quant.cpp / .h 源码逐行分析报告

> 本报告**只分析** `src/llama-quant.cpp`（1409 行）与 `src/llama-quant.h`（1 行）。
> 内容分三部分：
> 1. 源码业务逻辑、原理、每个函数、关键代码段；
> 2. 从容量、带宽、速度、精度、功耗五个维度做量化分析；
> 3. llama.cpp 的量化解决方案与工程措施。

---

## 目录

1. [文件定位与整体架构](#1-文件定位与整体架构)
2. [头文件 llama-quant.h](#2-头文件-llama-quanth)
3. [类型定义与张量分类](#3-类型定义与张量分类)
4. [名称重映射辅助](#4-名称重映射辅助)
5. [张量量化许可判断](#5-张量量化许可判断)
6. [张量类型选择与回退](#6-张量类型选择与回退)
7. [反量化实现](#7-反量化实现)
8. [量化实现](#8-量化实现)
9. [imatrix 重要性矩阵](#9-imatrix-重要性矩阵)
10. [文件类型与默认类型映射](#10-文件类型与默认类型映射)
11. [主量化驱动流程](#11-主量化驱动流程)
12. [对外接口与扩展](#12-对外接口与扩展)
13. [五维量化分析：容量/带宽/速度/精度/功耗](#13-五维量化分析)
14. [llama.cpp 量化解决方案与措施](#14-llamacpp-量化解决方案与措施)
15. [函数清单速查](#15-函数清单速查)

---

## 1. 文件定位与整体架构

### 1.1 文件职责

`llama-quant.cpp` 实现 llama.cpp 的**模型量化**功能：把权重从高精度（F32/F16/BF16）或已有低精度，转换为更低 bit 的量化格式，并写出新的 GGUF 模型文件。

### 1.2 量化发生在哪个阶段

```text
模型加载（运行时推理）  → 读取已有量化 GGUF
模型量化（离线工具）    → 生成量化 GGUF（本文件负责）
```

本文件属于**离线量化工具链**，对应可执行程序为 `llama-quantize`。它不参与运行时推理的反量化计算，只负责“把权重压缩成更小 bit 的格式”。

### 1.3 核心职责分解

| 子任务 | 实现位置 |
|--------|----------|
| 决定每个张量用什么类型量化 | `tensor_allows_quantization` + `llama_tensor_get_type` |
| 把 F32 权重转成量化格式 | `llama_tensor_quantize_impl` |
| 把已有量化/低精度转回 F32 | `llama_tensor_dequantize_impl` |
| 处理重要性矩阵（imatrix） | `tensor_requires_imatrix` + imatrix 查找 |
| 写新的 GGUF 文件 | `llama_model_quantize_impl` |
| 对外封装 | `llama_model_quantize` + `llama-ext.h` 工具 |

### 1.4 两个阶段的设计

量化过程分两个遍历：

```text
① 预遍历：决定每个张量的目标类型、是否需要 imatrix、统计计数
② 主遍历：真正反量化→量化→写文件
```

这样做的原因是：目标类型的选择依赖“前面张量/层”的计数（如 `i_attention_wv`、`i_layer`），需要先算好再写，否则无法保持一致。

---

## 2. 头文件 llama-quant.h

### 2.1 第 1 行

```cpp
#pragma once
```

这是 `llama-quant.h` 的全部内容：只有一个头文件保护，没有声明任何公共类型或函数。

实际量化的公开接口定义在：

```cpp
// llama-ext.h（外部工具扩展接口）
llama_model_quantize()
llama_quant_init/free()
llama_quant_model_from_metadata()
llama_quant_tensor_allows_quantization()
llama_quant_compute_types()
```

以及：

```cpp
// llama-model.h
struct llama_model_quantize_params;
```

因此 `llama-quant.h` 只是一个占位头，真正的量化 API 分散在 `llama-ext.h` 和 `llama-model.h`。

---

## 3. 类型定义与张量分类

### 3.1 第 17-21 行：`tensor_type_option`

```cpp
// result of parsing --tensor-type option
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};
```

用于解析 `--tensor-type` 命令行选项：把用户指定的张量名和对应量化类型配对。该结构的变更必须同步到 `tools/quantize/quantize.cpp`。

### 3.2 第 25-38 行：`tensor_category` 枚举

```cpp
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};
```

这是**粗粒度张量分类**，与 `LLM_TN`（特定架构张量名）不同。分类的目的是：

- 不需要为每种架构写特定逻辑；
- 只按张量的“功能角色”区分；
- 因为不同角色对量化精度的敏感度不同。

分类依据张量名中的子串（见 `tensor_get_category`）。

### 3.3 第 40-45 行：`zeros()`

```cpp
static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}
```

向文件写 `n` 个零字节。

用途：写 GGUF metadata 的占位区，或写入张量数据的对齐 padding。逐字节写虽然慢，但只用于元数据和 padding，可接受。

---

## 4. 名称重映射辅助

### 4.1 第 47-73 行：`remap_layer()`

```cpp
static std::string remap_layer(const std::string & orig_name,
        const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }
    static const std::regex pattern(R"(blk\.(\d+)\.)");
    ...
}
```

**功能**：当用户请求裁剪某些层（`prune_layers`）时，把 `blk.N.` 的层号重新编号为连续序号。

业务逻辑：

```text
用正则 blk.(\d+). 提取层号 blk
mapped.count(blk)      → 已映射，跳过
blk 在 prune 列表      → 该层被裁剪，映射为 ""（空）
blk < prune.front()    → 保留原层号
否则                   → 映射到 next_id 连续编号
```

返回空字符串表示该张量应被裁剪（从模型移除）。

### 4.2 第 75-94 行：`remap_imatrix()`

```cpp
static std::string remap_imatrix(const std::string & orig_name,
        const std::map<int, std::string> & mapped) {
```

**功能**：反向把 remap 后的层号映射回原始层号，用于匹配 imatrix 名称。

因为 imatrix 是按**原始层号**记录的，而量化时张量可能已被 remap，所以需要反查：

```text
mapped: 原层号 → 新层号
remap_imatrix: 新层号 → 原层号（找到 imatrix 条目）
```

找不到时 `GGML_ABORT`，避免静默用错 imatrix。

---

## 5. 张量量化许可判断

### 5.1 名称匹配辅助（第 100-107 行）

```cpp
static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}
```

用于识别特殊张量：token embedding 和 output 权重。

### 5.2 第 115-150 行：`tensor_get_category()`

```cpp
static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(...))   return tensor_category::OUTPUT;
    if (tensor_name_match_token_embd(...))      return tensor_category::TOKEN_EMBD;
    if (tensor_name.find("attn_qkv.weight")...) return tensor_category::ATTENTION_QKV;
    if (tensor_name.find("attn_kv_b.weight")...) return tensor_category::ATTENTION_KV_B;
    if (tensor_name.find("attn_v.weight")...)   return tensor_category::ATTENTION_V;
    if (tensor_name.find("attn_k.weight")...)   return tensor_category::ATTENTION_K;
    if (tensor_name.find("attn_q.weight")...)   return tensor_category::ATTENTION_Q;
    if (tensor_name.find("attn_output.weight")...) return tensor_category::ATTENTION_OUTPUT;
    if (tensor_name.find("ffn_up")...)          return tensor_category::FFN_UP;
    if (tensor_name.find("ffn_gate")...)        return tensor_category::FFN_GATE;
    if (tensor_name.find("ffn_down")...)        return tensor_category::FFN_DOWN;
    return tensor_category::OTHER;
}
```

**注意判断顺序**：`attn_qkv` 和 `attn_kv_b` 必须在 `attn_v`/`attn_k`/`attn_q` 之前检查，因为 `qkv` 含 `q`、`kv_b` 含 `v`。子串匹配顺序决定了正确分类。

### 5.3 第 153-157 行：`category_is_attn_v()`

```cpp
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V ||
           cat == tensor_category::ATTENTION_QKV ||
           cat == tensor_category::ATTENTION_KV_B;
}
```

定义“attention-V 类”张量。这类张量对量化精度**更敏感**，在低 bit 量化中需要更高精度。

---

## 6. 张量类型选择与回退

### 6.1 第 163-207 行：`quantize_state_impl`

```cpp
struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;   // attention-v 类张量总数
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;   // 当前已处理的 attention-v 序号
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_fallback = 0;        // 形状不兼容而回退的计数
    bool has_imatrix = false;
    bool has_tied_embeddings = true;
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;
    ...
};
```

这是量化过程的**运行状态**，保存：

- 各类张量的总数和当前计数；
- 是否有 imatrix；
- 是否 tied embeddings；
- 用户类型覆盖的编译后正则。

计数的作用：决定“第几个张量/第几层”是否给更高精度（见 `use_more_bits` 和分层策略）。

### 6.2 第 199-206 行：`tensor_metadata`

```cpp
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
};
```

预遍历阶段缓存每个张量的决策结果，主遍历直接复用，避免重复计算。

### 6.3 第 288-355 行：`tensor_allows_quantization()`

判断张量是否允许量化。大量使用字符串匹配排除“不宜量化”的张量：

```cpp
if (params->only_copy) return false;              // 纯复制不量化
if (ggml_n_dims(tensor) < 2) return false;        // 只量化 2D/3D
bool quantize = name.rfind("weight") == name.size() - 6;  // 以 weight 结尾
quantize &= name.find("_norm.weight") == std::string::npos;   // 不量化 norm
quantize &= params->quantize_output_tensor || name != "output.weight";
quantize &= name.find("ffn_gate_inp.weight") == std::string::npos; // 不量化 expert gate
...
```

排除列表（每项一个原因）：

| 排除张量 | 原因 |
|---|---|
| norm 权重 | 对精度极其敏感 |
| output.weight（除非显式允许） | 输出层对精度敏感 |
| ffn_gate_inp | expert 门控，极敏感 |
| altup/laurel | 非常小（如 4x4） |
| pos_embd/token_types | 位置/类型嵌入 |
| ssm_conv1d/shortconv | Mamba/Kimi 小 conv1d |
| time_mix_* | RWKV 小而敏感 |
| attn_rel_b | T5 相对位置偏置 |
| 各种多模态张量 | SAM/位置嵌入/补丁嵌入 |

**设计意图**：不是所有张量都适合量化。量化精度损失主要集中在敏感层，对它们保留较高精度或排除量化，能显著提升整体质量而代价很小。

### 6.4 第 362-409 行：`tensor_type_fallback()`

```cpp
static ggml_type tensor_type_fallback(quantize_state_impl & qs,
        const ggml_tensor * t, const ggml_type target_type) {
    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);
    if (ncols % qk_k != 0) {  // 形状不兼容
        ++qs.n_fallback;
        switch (target_type) {
            // block size 256 → 32
            case GGML_TYPE_IQ1_S: ... return GGML_TYPE_IQ4_NL;
            case GGML_TYPE_Q2_0: ... return GGML_TYPE_Q4_0;
            case GGML_TYPE_Q4_K: return GGML_TYPE_Q5_0;
            ...
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            return_type = GGML_TYPE_F16;   // 仍不兼容→F16
        }
    }
    return return_type;
}
```

**原因**：量化格式要求张量第一维（ncols）是 block size 的倍数。若张量形状特殊，必须回退到兼容类型；最终兜底为 F16。

### 6.5 第 412-704 行：`llama_tensor_get_type_impl()` 和 `llama_tensor_get_type()`

这是**量化类型决策的核心**，包含大量基于 ftype、张量类别、层号、模型类型的规则。

关键辅助：

```cpp
auto use_more_bits = [](int i_layer, int n_layers) -> bool {
    return i_layer < n_layers/8 ||          // 前 1/8 层
           i_layer >= 7*n_layers/8 ||       // 后 1/8 层
           (i_layer - n_layers/8)%3 == 2;   // 每隔 3 层
};
```

**分层策略**：模型的**前几层和后几层**对输出质量影响更大，给予更多 bit；中间层可更激进量化。

几个关键分支：

```text
OUTPUT / tied TOKEN_EMBD：
  默认 Q8_0 或 Q6_K（输出层敏感）

MXFP4_MOE：
  MoE 张量 → MXFP4
  其他 → Q8_0

TOKEN_EMBD：
  低 bit ftype → 较高精度的 Q2_K/IQ3_S

ATTENTION_V 类：
  通常给比 K/Q 更高精度（V 更敏感）
  70B 模型因共享 head → 升到 Q5_K
  8-expert → Q8_0

FFN_DOWN：
  分层策略，前后层更高 bit
```

**核心思想**：

```text
量化不是“统一给每个张量同样 bit”，而是按张量敏感度分配 bit：
  V > Q/K 敏感
  output/norm 敏感
  前后层敏感
  expert gate 敏感
```

这种“非均匀量化”是 llama.cpp 精度损失小的关键。

`llama_tensor_get_type()`（外层包装）顺序：

```text
① 不允许量化 → 返回原类型
② 用户指定 token_embd/output 类型 → 用之
③ 默认类型 = llama_ftype_get_default_type(ftype)
④ 非 pure 且默认类型是量化 → 决策
     └ 用户正则覆盖 → 用之
     └ 否则 llama_tensor_get_type_impl 决策
     └ 形状回退
```

---

## 7. 反量化实现

### 7.1 第 212-282 行：`llama_tensor_dequantize_impl()`

```cpp
static void llama_tensor_dequantize_impl(ggml_tensor * tensor,
        std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
        const size_t nelements, const int nthread) {
    ...
    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) { throw ... }
    } else if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        throw ...;  // 不能转换的类型
    }
```

**功能**：把量化/低精度张量转成 F32 数组，供后续量化使用。

路径：

```text
F16   → ggml_fp16_to_fp32_row
BF16  → ggml_bf16_to_fp32_row
量化  → qtype->to_float
```

多线程版本按 block 切分：

```cpp
size_t nblocks = nelements / block_size;
size_t blocks_per_thread = nblocks / nthread;
size_t spare_blocks = nblocks - (blocks_per_thread * nthread);
```

每个线程处理一部分 block，最后 join。F16/BF16 的 block size 为 1，量化类型的 block size 为 `ggml_blck_size(type)`。

**用途**：重新量化（requantize）或把非 F32 转成目标量化类型时，需要先统一到 F32。

---

## 8. 量化实现

### 8.1 第 710-762 行：`llama_tensor_quantize_impl()`

```cpp
static size_t llama_tensor_quantize_impl(enum ggml_type new_type,
        const float * f32_data, void * new_data, const int64_t chunk_size,
        int64_t nrows, int64_t n_per_row, const float * imatrix,
        std::vector<std::thread> & workers, const int nthread) {
```

**功能**：把 F32 数据量化为目标类型。

单线程路径：

```cpp
size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
if (!ggml_validate_row_data(new_type, new_data, new_size)) { throw ... }
```

多线程路径：用互斥锁切分 row chunk，每个线程处理一批 `nrows_per_chunk` 行，并校验量化后的数据。

**关键点**：

- 调用 `ggml_quantize_chunk` 完成实际量化；
- 量化后调用 `ggml_validate_row_data` 校验，失败即抛异常；
- 支持 imatrix 逐行加权。

---

## 9. imatrix 重要性矩阵

### 9.1 第 768-787 行：`tensor_requires_imatrix()`

```cpp
static bool tensor_requires_imatrix(const char * tensor_name,
        const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(...) || tensor_name_match_output_weight(...)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;   // 极低 bit 必须 imatrix
        case GGML_TYPE_Q2_K:
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;  // 仅 Q2_K_S 需要
        default:
            return false;
    }
}
```

**imatrix（importance matrix）是什么**：

```text
每行/每列激活幅度的统计信息
  → 记录哪些权重通道“更重要”
  → 量化时给重要通道更合适的 scale
```

**为什么极低 bit 必须 imatrix**：IQ 系列（IQ1/IQ2/IQ3）量化非常激进，误差大；没有 imatrix 会产出“垃圾”。因此明确要求必须提供 imatrix，否则报错退出。

### 9.2 imatrix 在量化中的使用

量化时，imatrix 被传给 `ggml_quantize_chunk`，使每个通道的量化 step 依据重要性调整。

```cpp
const float * imatrix = nullptr;
if (imatrix_data) {
    auto it = imatrix_data->find(tm.remapped_imatrix_name);
    if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
        imatrix = it->second.data();   // 匹配成功
    } else {
        ... // 大小不匹配 → 报错
    }
}
```

**校验**：imatrix 大小必须等于 `ne[0]*ne[2]`（每行重要性 × 通道数）。不匹配且不是 token_embd 时直接报错，避免“大部分模型没用 imatrix 量化”。

---

## 10. 文件类型与默认类型映射

### 10.1 第 793-836 行：`llama_ftype_get_default_type()`

```cpp
ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        ...
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return GGML_TYPE_Q4_K;
        ...
        default: return GGML_TYPE_COUNT;
    }
}
```

把“文件类型（ftype）”映射到“张量默认量化类型”。ftype 是用户选择的目标格式（如 Q4_K_M），默认类型决定非特殊张量用什么 ggml_type。

---

## 11. 主量化驱动流程

### 11.1 第 859-872 行：入口与线程

```cpp
static void llama_model_quantize_impl(const std::string & fname_inp,
        const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;
    int nthread = params->nthread;
    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();  // 默认用所有核
    }
    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw ...;  // 非法 ftype
    }
```

### 11.2 第 873-897 行：mmap 与模型加载

```cpp
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;   // Linux/Windows 用 mmap 更快
#else
    constexpr bool use_mmap = false;  // macOS 可能因空闲内存变慢
#endif

    llama_model_loader ml(..., fname_inp, splits, ..., use_mmap, ...);
    ml.init_mappings(false);
    auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
    model->load_hparams(ml);
    model->load_stats(ml);
```

**平台差异**：量化时的 mmap 使用因平台而异——Linux/Windows 加快，macOS 可能变慢。

### 11.3 第 900-936 行：imatrix 与输出上下文

```cpp
if (params->only_copy) { ftype = ml.ftype; }   // 纯复制保持原 ftype

if (params->imatrix) {
    for (...) { i_data.emplace(p->name, ...); }   // 收集 imatrix
    qs.has_imatrix = true;
    // 校验 imatrix 无 NaN/Inf
    for (float f : kv.second) {
        if (!std::isfinite(f)) { throw ...; }
    }
}

gguf_context_ptr ctx_out { gguf_init_empty() };
gguf_set_kv(ctx_out.get(), ml.metadata);       // 复制 KV 元数据
gguf_set_val_u32(..., GGML_QNT_VERSION);       // 量化版本
gguf_set_val_u32(..., ftype);                   // 文件类型
```

### 11.4 第 935-1015 行：层裁剪与张量列表

```cpp
if (params->prune_layers) { ... 收集裁剪层号 ... }
...
for (const auto & it : ml.weights_map) {
    const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
    if (remapped_name.empty()) { continue; }   // 被裁剪
    tensors.push_back(&it.second);
}
if (params->keep_split) { ... 按 split 排序 ... }
```

### 11.5 第 1017-1058 行：预遍历

```cpp
for (size_t i = 0; i < tensors.size(); ++i) {
    ...
    metadata[i].allows_quantization = tensor_allows_quantization(...);
    if (metadata[i].allows_quantization) {
        metadata[i].target_type = llama_tensor_get_type(...);
    } else {
        metadata[i].target_type = tensor->type;
    }
    metadata[i].requires_imatrix = tensor_requires_imatrix(...);
    if (params->imatrix) {
        metadata[i].remapped_imatrix_name = remap_imatrix(...);
    } else if (metadata[i].allows_quantization && metadata[i].requires_imatrix) {
        if (params->dry_run) { will_require_imatrix = true; }
        else { throw ...; }   // 必须 imatrix 而未提供 → 报错
    }
}
```

### 11.6 第 1117-1265 行：主遍历（逐张量量化）

对每个张量：

```cpp
const ggml_type cur_type = tensor->type;
const ggml_type new_type = tm.target_type;
bool quantize = cur_type != new_type;   // 类型不同才量化
```

**dry-run**：只估算大小，不写文件。

```cpp
if (params->dry_run) {
    if (quantize) {
        new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
    } else {
        new_size = tensor_size;
    }
    total_size_org += tensor_size;
    total_size_new += new_size;
    continue;
}
```

**实际量化**：

```cpp
if (!quantize) {
    new_data = tensor->data;  // 保持原类型，直接写
} else {
    const float * imatrix = ...;   // 查找该张量的 imatrix
    if (!imatrix && tm.requires_imatrix) { throw ...; }  // 缺少→报错
    // 转 F32
    if (tensor->type == GGML_TYPE_F32) f32_data = tensor->data;
    else if (ggml_is_quantized(...) && !params->allow_requantize) throw ...;
    else llama_tensor_dequantize_impl(...);  // 反量化到 F32
    // 分 chunk 量化
    for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {  // 每个 expert
        new_size += llama_tensor_quantize_impl(...);
    }
}
// 更新 GGUF 元数据 + 写文件 + padding
gguf_set_tensor_type(...);
gguf_set_tensor_data(...);
fout.write(new_data, new_size);
zeros(fout, GGML_PAD(new_size, align) - new_size);
```

**多专家处理**：`tensor->ne[2]` 对应 expert 维度，每个 expert 独立量化，因为它们有不同的 imatrix。

### 11.7 第 1267-1283 行：收尾统计

```cpp
LLAMA_LOG_INFO("%s: model size  = %.2f MiB (%.2f BPW)", ..., total_size_org*8.0/ml.n_elements);
LLAMA_LOG_INFO("%s: quant size  = %.2f MiB (%.2f BPW)", ..., total_size_new*8.0/ml.n_elements);
```

输出：

```text
原始模型大小（MiB）和 bit-per-weight（BPW）
量化后模型大小（MiB）和 BPW
```

`BPW = total_size_bytes * 8 / 总元素数`，是衡量量化密度的核心指标。

---

## 12. 对外接口与扩展

### 12.1 第 1290-1323 行：默认参数与入口

```cpp
llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        .nthread = 0,
        .ftype = LLAMA_FTYPE_MOSTLY_Q8_0,   // 默认 Q8_0
        .output_tensor_type = GGML_TYPE_COUNT,
        ...
    };
    return result;
}

uint32_t llama_model_quantize(const char * fname_inp, const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s", __func__, err.what());
        return 1;
    }
    return 0;
}
```

`llama_model_quantize` 是量化入口，捕获异常返回错误码（0 成功，1 失败）。

### 12.2 第 1329-1409 行：外部工具辅助

```cpp
quantize_state_impl * llama_quant_init(model, params);   // 创建量化状态
void llama_quant_free(qs);                                // 释放

llama_model * llama_quant_model_from_metadata(desc);      // 从元数据构建模型
// 推断 LLM_TYPE_70B：80 层 + head != head_kv
if (model->arch == LLM_ARCH_LLAMA && desc->n_layer == 80 && desc->n_head != desc->n_head_kv) {
    model->type = LLM_TYPE_70B;
}

bool llama_quant_tensor_allows_quantization(qs, tensor); // 张量是否可量化

void llama_quant_compute_types(qs, ftype, tensors, result_types, n_tensors);
// 计算一组张量的目标类型（供外部工具复用决策逻辑）
```

这些接口让 `llama-ext.h` 的外部工具（如在线量化/评估工具）能复用 llama.cpp 的类型决策逻辑。

---

## 13. 五维量化分析：容量/带宽/速度/精度/功耗

### 13.1 容量（Capacity）

**权重内存**：

```text
权重内存 = 参数量 × 每参数字节数
BPW 越低，每参数占用 bit 越少
```

常见量化的 BPW 与压缩比：

| 类型 | 约 BPW | 相对 F16 压缩比 |
|---|---|---|
| F32 | 32 | 1x（基准） |
| F16 | 16 | 2x |
| BF16 | 16 | 2x |
| Q8_0 | 9.0 | ~3.5x |
| Q5_K | ~5.5 | ~5.8x |
| Q4_K_M | ~4.8 | ~6.7x |
| Q4_0 | ~4.5 | ~7.1x |
| IQ2_XS | ~2.3 | ~13.9x |
| IQ1_S | ~1.6 | ~20x |

**公式**：

```text
模型大小(MiB) ≈ 参数量 × BPW / 8 / 1024²
```

示例：

```text
7B 模型：
  F16 → 14GB
  Q8  → ~7.9GB
  Q4_K_M → ~4.2GB
  Q2_K → ~2.4GB
```

**KV Cache 容量**：权重量化不直接减小 KV，但量化 KV 缓存（如 Q8 的 K/V）可进一步减：

```text
KV 内存 = 2 × n_layers × n_ctx × n_embd_k/v_gqa × dtype_bytes
F16 KV → Q8 KV 减半
```

### 13.2 带宽（Bandwidth）

端侧推理（尤其 decode）通常**带宽受限**。权重量化直接降低每次读取权重的字节数：

```text
decode 每 token 需读取所有权重一次
带宽需求 ∝ 权重字节数
→ 权重减半 → 带宽压力减半
```

```text
单 token 读取量 = 参数量 × bytes_per_param
7B F16 → 14GB/次
7B Q4  → ~4GB/次
```

量化对带宽的收益：

```text
decode（权重读密集）收益最大
prefill（计算密集）收益相对较小，但大 batch 下仍有利
```

**K/V 量化**还降低 attention 阶段读取历史 KV 的带宽：

```text
attention 读 KV 量 ∝ n_kv × n_embd × dtype_bytes
KV 量化 → 降低长上下文 attention 带宽
```

### 13.3 速度（Speed）

量化对速度的影响是双面的：

**有利面**：

- 权重更小 → 内存带宽压力小 → decode 快；
- cache 命中率更高；
- 更多权重可放进有限缓存。

**不利面**：

- 反量化/dequantize 计算开销；
- 低 bit 可能需要特殊 kernel；
- 部分类型（如 IQ）需要额外反量化算法。

**实际结论**：

```text
端侧/带宽受限 → 量化通常更快（尤其 decode）
算力受限 → 量化收益可能被反量化开销抵消

通常：
  Q8_0 精度接近 F16，速度略快或相当
  Q4 比 Q8 更省带宽，但若反量化开销大，未必更快
  必须实测，不能只看 BPW
```

### 13.4 精度（Precision）

**量化误差来源**：

```text
① 权重截断：低 bit 表示不了原值
② scale/zero-point 量化：block 内共享统计量的近似
③ 反量化往返：量化→反量化有信息损失
```

**误差敏感度**（llama.cpp 设计依据）：

```text
输出层 > 归一化层 > V > Q/K > FFN
前几层/后几层 > 中间层
```

**llama.cpp 缓解精度的措施**：

```text
① 非均匀量化：敏感张量给更多 bit
② 分层策略：前后层更高 bit
③ 排除敏感张量：norm/output/expert gate 不量化
④ imatrix：极低 bit 用重要性矩阵加权
⑤ 更高 bit 的 V 类张量
```

**典型精度影响**（经验值，非绝对）：

```text
Q8_0 → 质量几乎无损
Q5_K/Q4_K_M → 轻微损失，可用
Q4_0 → 可见损失
IQ2/IQ1 → 明显损失，需 imatrix
```

### 13.5 功耗（Power）

量化对功耗的影响：

```text
① 数据搬运减少 → DRAM 访问功耗下降（带宽受限场景主因）
② 计算量可能变化 → 反量化开销可能增加计算功耗
③ 更多权重驻留缓存 → 减少外部内存访问
```

```text
功耗主要来自：
  DRAM 带宽（权重读取）
  计算单元
  缓存

权重量化 → 降低 DRAM 带宽 → 降低功耗
但低 bit 反量化 kernel 可能增加计算功耗

净效果取决于：带宽受限 vs 算力受限
```

**端侧综合**：

```text
端侧通常带宽/内存受限 + 功耗受限
→ 量化是降功耗的关键手段
→ 配合稀疏注意力（如 DSV4）进一步降功耗
```

### 13.6 五维权衡总结

```text
量化越激进（BPW 越低）：
  容量 ↓（好）
  带宽 ↓（好）
  速度：端侧通常 ↑，但受反量化开销影响
  精度 ↓（坏）
  功耗 ↓（通常好）

目标是在容量/带宽/功耗与精度之间找平衡
```

| 场景 | 推荐 |
|---|---|
| 精度优先、资源充足 | F16 / Q8_0 |
| 均衡 | Q4_K_M / Q5_K_M |
| 端侧内存紧张 | Q4_0 / Q2_K |
| 极致压缩、有 imatrix | IQ2/IQ3 |

---

## 14. llama.cpp 量化解决方案与措施

### 14.1 方案一：丰富的量化类型

llama.cpp 提供多档位：

```text
传统型：Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
K型：Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
IQ型：IQ1/IQ2/IQ3/IQ4（需 imatrix）
MXFP4：MXFP4_MOE（MoE 专用）
```

覆盖从 F32 到 ~1.6 BPW 的全谱系，满足不同精度/容量需求。

### 14.2 方案二：非均匀/分层量化

不是所有张量同 bit，而是按敏感度分配：

```text
V 类 > Q/K 类（V 更敏感）
output/norm 不量化或高 bit
前后层高 bit
expert gate 不量化
```

这是 llama.cpp 在低 bit 下保持质量的关键。

### 14.3 方案三：重要性矩阵（imatrix）

```text
imatrix 记录每行/通道的重要性
量化时据此调整 scale
极低 bit（IQ）强制要求 imatrix
```

### 14.4 方案四：张量白名单/黑名单

`tensor_allows_quantization` 维护一组排除规则：

```text
不量化：norm、output（可选）、expert gate、位置嵌入、
        conv1d、RWKV time_mix、多模态位置/补丁嵌入
```

避免敏感/小张量被错误量化。

### 14.5 方案五：形状回退

量化类型要求张量第一维是 block size 倍数，`tensor_type_fallback` 自动回退：

```text
不兼容 → 更高兼容类型 → 仍不兼容 → F16
```

保证量化过程不会因形状崩溃。

### 14.6 方案六：dry-run 与预遍历

```text
dry-run：先估算最终大小，不写文件
预遍历：先计算所有张量类型决策，再执行
```

### 14.7 方案七：多线程与 mmap

```text
多线程分 chunk 量化（按 row 切分）
Linux/Windows 用 mmap 加速读取
```

### 14.8 方案八：与 KV Cache 量化联动

```text
权重量化 → 减少权重内存
KV 量化（Q8 K/V）→ 减少缓存内存
Hadamard 旋转 → 降低量化 KV 的 outlier 误差
```

这三者配合，是端侧长上下文部署的组合拳。

### 14.9 方案九：speculative / 混合精度

```text
MoE 用 MXFP4（experts 大量、低敏感）
关键层用更高 bit
输出层独立配置
```

### 14.10 方案十：验证与回滚

```text
量化后校验 ggml_validate_row_data
imatrix 校验非有限值
类型/大小一致性断言
失败即报错，避免产出损坏模型
```

---

## 15. 函数清单速查

| 函数 | 行号 | 作用 |
|------|------|------|
| `zeros` | 40 | 写零字节（padding/占位） |
| `remap_layer` | 47 | 层裁剪后的重编号 |
| `remap_imatrix` | 75 | imatrix 层号反查 |
| `tensor_name_match_token_embd` | 100 | 识别 token 嵌入 |
| `tensor_name_match_output_weight` | 105 | 识别输出权重 |
| `tensor_get_category` | 115 | 张量粗分类 |
| `category_is_attn_v` | 153 | 判断 V 类张量 |
| `quantize_state_impl` 构造 | 186 | 编译用户正则 |
| `tensor_allows_quantization` | 288 | 张量可否量化 |
| `tensor_type_fallback` | 362 | 形状不兼容回退 |
| `llama_tensor_get_type_impl` | 412 | 类型决策核心 |
| `llama_tensor_get_type` | 662 | 类型决策包装 |
| `llama_tensor_dequantize_impl` | 212 | 反量化到 F32 |
| `llama_tensor_quantize_impl` | 710 | F32 量化到目标类型 |
| `tensor_requires_imatrix` | 768 | 是否必须 imatrix |
| `llama_ftype_get_default_type` | 793 | ftype→默认类型 |
| `init_quantize_state_counters` | 839 | 统计各分类计数 |
| `llama_model_quantize_impl` | 859 | 主量化驱动 |
| `llama_model_quantize_default_params` | 1290 | 默认量化参数 |
| `llama_model_quantize` | 1311 | 量化入口 |
| `llama_quant_init` | 1329 | 创建量化状态 |
| `llama_quant_free` | 1335 | 释放量化状态 |
| `llama_quant_model_from_metadata` | 1339 | 从元数据建模型 |
| `llama_quant_tensor_allows_quantization` | 1365 | 张量可量化查询 |
| `llama_quant_compute_types` | 1371 | 批量计算类型 |

---

## 一句话总结

> `llama-quant.cpp` 通过“张量分类 + 非均匀分层量化 + imatrix 重要性矩阵 + 敏感张量白名单 + 形状回退 + 多线程”实现了从 F32/F16 到极低 bit 的完整量化工具链。量化在**容量、带宽、功耗**上带来显著收益（端侧尤其明显），在**速度**上通常有利但需实测，在**精度**上有可控损失——llama.cpp 用多种措施把精度损失压到可接受范围，并为端侧/长上下文部署提供“权重量化 + KV 量化 + 稀疏注意力”的组合方案。