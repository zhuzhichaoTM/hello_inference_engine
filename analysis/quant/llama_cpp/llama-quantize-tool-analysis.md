# tools/quantize/quantize.cpp 源码逐行分析报告

> 本报告只分析 `tools/quantize/quantize.cpp`（656 行）。
> 这是命令行工具 `llama-quantize` 的入口实现，负责**参数解析、imatrix 加载、量化调用和耗时统计**。
> 它本身**不实现量化算法**，而是把用户意图转换为 `llama_model_quantize_params`，再调用 `src/llama-quant.cpp` 的 `llama_model_quantize()` 完成实际量化。

---

## 目录

1. [文件定位与整体架构](#1-文件定位与整体架构)
2. [头文件依赖（第 1-19 行）](#2-头文件依赖第-1-19-行)
3. [数据结构：tensor_type_option 与 quant_option（第 21-32 行）](#3-数据结构tensor_type_option-与-quant_option第-21-32-行)
4. [量化类型表 QUANT_OPTIONS（第 34-75 行）](#4-量化类型表-quant_options第-34-75-行)
5. [imatrix KV 常量（第 77-80 行）](#5-imatrix-kv-常量第-77-80-行)
6. [辅助函数：striequals / try_parse_ftype（第 82-119 行）](#6-辅助函数striequals--try_parse_ftype第-82-119-行)
7. [usage 帮助（第 121-178 行）](#7-usage-帮助第-121-178-行)
8. [imatrix 加载：load_imatrix（第 180-258 行）](#8-imatrix-加载load_imatrix第-180-258-行)
9. [imatrix 准备：prepare_imatrix（第 260-300 行）](#9-imatrix-准备prepare_imatrix第-260-300-行)
10. [类型与张量解析（第 302-361 行）](#10-类型与张量解析第-302-361-行)
11. [层裁剪解析 parse_layer_prune（第 363-387 行）](#11-层裁剪解析-parse_layer_prune第-363-387-行)
12. [main 入口与参数解析（第 389-473 行）](#12-main-入口与参数解析第-389-473-行)
13. [imatrix/tensor_override/kv_override 数据组装（第 483-544 行）](#13-imatrixtensor_overridekv_override-数据组装第-483-544-行)
14. [文件路径与 ftype 解析（第 548-594 行）](#14-文件路径与-ftype-解析第-548-594-行)
15. [执行量化与耗时统计（第 627-655 行）](#15-执行量化与耗时统计第-627-655-行)
16. [函数清单速查](#16-函数清单速查)

---

## 1. 文件定位与整体架构

### 1.1 文件职责

`tools/quantize/quantize.cpp` 是命令行工具 `llama-quantize` 的唯一入口。

```text
用户命令行
  → 参数解析（本文件）
  → imatrix 加载/过滤（本文件）
  → 组装 llama_model_quantize_params（本文件）
  → 调用 llama_model_quantize()（src/llama-quant.cpp 实现）
  → 耗时统计与输出（本文件）
```

### 1.2 与 src/llama-quant.cpp 的分工

| 职责 | 本文件（tools/quantize） | 实现文件（src/llama-quant.cpp） |
|------|--------------------------|-------------------------------|
| 命令行解析 | ✅ | ❌ |
| imatrix 文件加载 | ✅ | ❌ |
| 量化类型表 | ✅（展示用） | ❌ |
| 张量类型决策 | ❌ | ✅ |
| 反量化/量化执行 | ❌ | ✅ |
| GGUF 写出 | ❌ | ✅ |
| 耗时统计 | ✅ | ❌ |

本文件是"前端"，`src/llama-quant.cpp` 是"后端"。

---

## 2. 头文件依赖（第 1-19 行）

```cpp
#include "llama.h"           // 公共 API（llama_model_quantize 等）
#include "build-info.h"      // 构建信息（版本号）
#include "common.h"          // 通用工具（string_split/string_parse_kv_override）
#include "imatrix-loader.h"  // imatrix 文件加载器
#include "gguf.h"            // GGUF 格式常量

#include <algorithm>
#include <cctype>            // tolower/toupper
#include <clocale>           // setlocale
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>            // 读取 tensor-type-file
#include <filesystem>         // 检查输入输出是否同文件
```

每个头文件都有明确用途，没有冗余依赖。

---

## 3. 数据结构：tensor_type_option 与 quant_option（第 21-32 行）

```cpp
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;   // 默认无效
};

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};
```

- `tensor_type_option`：解析 `--tensor-type` 选项的结果（张量名→目标类型）。
- `quant_option`：量化档位选项（名称、ftype、描述）。

注意：`tensor_type_option` 与 `src/llama-quant.cpp` 中的同名结构**必须保持一致**（注释第 22 行明确说明）。

---

## 4. 量化类型表 QUANT_OPTIONS（第 34-75 行）

```cpp
static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q1_0",     LLAMA_FTYPE_MOSTLY_Q1_0,     " 1.125 bpw quantization",           },
    { "Q2_0",     LLAMA_FTYPE_MOSTLY_Q2_0,     " 2.25 bpw quantization (group 64)",  },
    ...
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    ...
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};
```

这是所有可用量化档位的**静态注册表**，包含：
- 名称（如 `Q4_K_M`）；
- 对应的 `llama_ftype` 枚举；
- 描述（含 BPW、模型大小、PPL 退化参考值）。

**用途**：
1. `try_parse_ftype` 解析用户输入；
2. `usage` 打印帮助；
3. `COPY` 特殊项表示"只复制不量化"。

注释第 73 行：
```cpp
// Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
```
因为 `LLAMA_FTYPE_ALL_F32` 和 `LLAMA_FTYPE_GGUF` 都是 0，`COPY` 必须在后面，否则会先匹配到 COPY。

---

## 5. imatrix KV 常量（第 77-80 行）

```cpp
static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";
```

这些是写入量化后 GGUF 的元数据 key，记录"量化时用了哪个 imatrix、多少条目、多少 chunk"。用于**可复现性追踪**。

---

## 6. 辅助函数：striequals / try_parse_ftype（第 82-119 行）

### 6.1 `striequals`（第 82-90 行）

```cpp
static bool striequals(const char * a, const char * b) {
    while (*a && *b) {
        if (std::tolower(*a) != std::tolower(*b)) return false;
        a++; b++;
    }
    return *a == *b;
}
```

大小写不敏感的字符串比较。用于让用户输入 `q4_k_m` 或 `Q4_K_M` 都能匹配。

### 6.2 `try_parse_ftype`（第 92-119 行）

```cpp
static bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    // ① 先转大写
    std::string ftype_str;
    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    // ② 按名称匹配 QUANT_OPTIONS
    for (const auto & it : QUANT_OPTIONS) {
        if (striequals(it.name.c_str(), ftype_str.c_str())) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    // ③ 按数字匹配（允许直接传 ftype 整数）
    try {
        int ftype_int = std::stoi(ftype_str);
        for (const auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) { ... return true; }
        }
    } catch (...) {}
    return false;
}
```

支持两种输入：
- 名称：`Q4_K_M`；
- 数字：`ftype` 枚举值。

---

## 7. usage 帮助（第 121-178 行）

`[[noreturn]]` 标注表示调用后一定 `exit(1)`，不返回。

打印所有命令行选项的说明和所有可用量化类型。这是用户的第一手文档。

---

## 8. imatrix 加载：load_imatrix（第 180-258 行）

```cpp
static int load_imatrix(const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
```

### 8.1 加载文件

```cpp
common_imatrix loaded;
if (!common_imatrix_load(imatrix_file, loaded)) {
    fprintf(stderr, "%s: failed to load imatrix from '%s'\n", ...);
    exit(1);
}
if (!loaded.is_legacy && !loaded.has_metadata) {
    fprintf(stderr, "%s: missing imatrix metadata ...\n", ...);
    exit(1);
}
```

调用 `common_imatrix_load` 读取 imatrix 文件，失败即退出。非 legacy 格式必须有 metadata。

### 8.2 归一化

```cpp
for (const auto & [name, entry] : loaded.entries) {
    auto & e = imatrix_data[name];
    e.resize(entry.sums.size());

    if (!loaded.is_legacy) {
        // GGUF 格式：按 per-expert counts 归一化
        const int64_t ncounts = entry.counts.size();
        const int64_t ne0     = (int64_t) entry.sums.size() / ncounts;
        for (int64_t j = 0; j < ncounts; ++j) {
            const float count = (float) entry.counts[j];
            if (count > 0.0f) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j*ne0 + i] = entry.sums[j*ne0 + i] / count;
                }
            } else {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j*ne0 + i] = 1;   // 无数据时默认 1
                }
            }
        }
    } else {
        // Legacy 格式：sums 含 (raw/count)*ncall，除以 ncall
        const int64_t ncall = entry.counts.empty() ? 0 : entry.counts[0];
        if (ncall > 0) {
            for (size_t i = 0; i < entry.sums.size(); ++i) {
                e[i] = entry.sums[i] / ncall;
            }
        } else {
            for (size_t i = 0; i < entry.sums.size(); ++i) {
                e[i] = entry.sums[i];
            }
        }
    }
}
```

**核心逻辑**：把累积的 `sums` 除以 `counts`，得到**每行平均重要性**。两种格式：
- GGUF 格式：按 expert 分组归一化；
- Legacy 格式：统一除以 `ncall`。

### 8.3 返回 chunk 数

```cpp
return loaded.chunk_count;
```

返回 imatrix 的 chunk 数，用于写入 KV 元数据。

---

## 9. imatrix 准备：prepare_imatrix（第 260-300 行）

```cpp
static int prepare_imatrix(const std::string & imatrix_file,
        std::vector<std::string> & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
```

### 9.1 加载 + 过滤

```cpp
int m_last_call = -1;
if (!imatrix_file.empty()) {
    m_last_call = load_imatrix(imatrix_file, imatrix_dataset, imatrix_data);
}
if (imatrix_data.empty()) {
    return m_last_call;
}
// 排除指定张量
if (!excluded_weights.empty()) {
    for (const auto & name : excluded_weights) {
        for (auto it = imatrix_data.begin(); it != imatrix_data.end();) {
            auto pos = it->first.find(name);
            if (pos != std::string::npos) {
                it = imatrix_data.erase(it);   // 删除匹配的
            } else {
                ++it;
            }
        }
    }
}
// 只包含指定张量
if (!included_weights.empty()) {
    std::unordered_map<std::string, std::vector<float>> tmp;
    for (const auto & name : included_weights) {
        for (auto & e : imatrix_data) {
            auto pos = e.first.find(name);
            if (pos != std::string::npos) {
                tmp.emplace(std::move(e));
            }
        }
    }
    imatrix_data = std::move(tmp);
}
```

**作用**：加载 imatrix 后，按 `--include-weights` / `--exclude-weights` 过滤条目。

---

## 10. 类型与张量解析（第 302-361 行）

### 10.1 `parse_ggml_type`（第 302-312 行）

```cpp
static ggml_type parse_ggml_type(const char * arg) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        auto type = (ggml_type)i;
        const auto * name = ggml_type_name(type);
        if (name && striequals(name, arg)) {
            return type;
        }
    }
    fprintf(stderr, "\n%s: invalid ggml_type '%s'\n\n", __func__, arg);
    return GGML_TYPE_COUNT;
}
```

遍历所有 `ggml_type`，按名称匹配（大小写不敏感）。

### 10.2 `parse_tensor_type`（第 314-344 行）

```cpp
static bool parse_tensor_type(const char * data, std::vector<tensor_type_option> & tensor_type) {
    const char * sep = strchr(data, '=');   // 找 '='
    if (sep == nullptr) { return false; }    // 必须有 '='
    const size_t tn_len = sep - data;
    if (tn_len == 0) { return false; }       // 张量名不能为空
    if (const size_t qt_len = strlen(sep); qt_len == 1) { return false; } // 类型不能为空

    std::string tn(data, tn_len);
    std::transform(tn.begin(), tn.end(), tn.begin(), tolower);  // 张量名转小写
    sep++;
    tensor_type_option tensor_type_opt;
    tensor_type_opt.name = tn;
    tensor_type_opt.type = parse_ggml_type(sep);
    tensor_type.emplace_back(std::move(tensor_type_opt));
    ...
}
```

解析 `tensor_name=ggml_type` 格式（如 `attn_q=q8_0`）。

### 10.3 `parse_tensor_type_file`（第 346-361 行）

从文件批量读取 `tensor_name=ggml_type`，每行或空格分隔一条。

---

## 11. 层裁剪解析 parse_layer_prune（第 363-387 行）

```cpp
static bool parse_layer_prune(const char * data, std::vector<int> & prune_layers) {
    const auto block_ids = string_split<std::string>(data, ',');
    for (const auto & block_id : block_ids) {
        int id = std::stoi(block_id);
        if (id < 0) { return false; }
        prune_layers.emplace_back(id);
    }
    sort(prune_layers.begin(), prune_layers.end());
    prune_layers.erase(std::unique(...), prune_layers.end());  // 去重
    return true;
}
```

解析 `--prune-layers 0,1,5` 格式，结果排序去重。

---

## 12. main 入口与参数解析（第 389-473 行）

### 12.1 声明

```cpp
// satisfies -Wmissing-declarations
int llama_quantize(int argc, char ** argv);
```

显式声明满足编译器警告。

### 12.2 参数解析循环（第 407-473 行）

```cpp
for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
    if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
        params.quantize_output_tensor = false;
    } else if (strcmp(argv[arg_idx], "--output-tensor-type") == 0) {
        if (arg_idx < argc-1) {
            params.output_tensor_type = parse_ggml_type(argv[++arg_idx]);
            ...
        }
    } else if (strcmp(argv[arg_idx], "--token-embedding-type") == 0) {
        ...
    } else if (strcmp(argv[arg_idx], "--tensor-type") == 0) {
        ...
    } else if (strcmp(argv[arg_idx], "--tensor-type-file") == 0) {
        ...
    } else if (strcmp(argv[arg_idx], "--prune-layers") == 0) {
        ...
    } else if (strcmp(argv[arg_idx], "--override-kv") == 0) {
        ...
    } else if (strcmp(argv[arg_idx], "--dry-run") == 0) {
        params.dry_run = true;
    } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
        params.allow_requantize = true;
    } else if (strcmp(argv[arg_idx], "--pure") == 0) {
        params.pure = true;
    } else if (strcmp(argv[arg_idx], "--imatrix") == 0) {
        imatrix_file = argv[++arg_idx];
    } else if (strcmp(argv[arg_idx], "--include-weights") == 0) {
        included_weights.emplace_back(argv[++arg_idx]);
    } else if (strcmp(argv[arg_idx], "--exclude-weights") == 0) {
        excluded_weights.emplace_back(argv[++arg_idx]);
    } else if (strcmp(argv[arg_idx], "--keep-split") == 0) {
        params.keep_split = true;
    } else {
        usage(argv[0]);   // 未知选项 → 打印帮助退出
    }
}
```

每个 `--xxx` 选项映射到 `llama_model_quantize_params` 的一个字段。这是"命令行→参数结构"的直接翻译。

### 12.3 互斥检查

```cpp
if (!included_weights.empty() && !excluded_weights.empty()) {
    usage(argv[0]);   // 不能同时 include 和 exclude
}
```

---

## 13. imatrix/tensor_override/kv_override 数据组装（第 483-544 行）

### 13.1 imatrix 数据组装

```cpp
if (!imatrix_data.empty()) {
    i_data.reserve(imatrix_data.size() + 1);
    for (const auto & kv : imatrix_data) {
        i_data.push_back({kv.first.c_str(), kv.second.data(), kv.second.size()});
    }
    i_data.push_back({nullptr, nullptr, 0});  // 数组终结符
    params.imatrix = i_data.data();

    // 写入 imatrix 元数据到 KV overrides
    {
        llama_model_kv_override kvo;
        std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_FILE);
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        strncpy(kvo.val_str, imatrix_file.c_str(), 127);
        kvo.val_str[127] = '\0';
        kv_overrides.emplace_back(std::move(kvo));
    }
    // 同理写入 dataset / n_entries / n_chunks
}
```

**关键点**：
1. imatrix 数据转成 C 风格数组（以 `{nullptr, nullptr, 0}` 结尾）；
2. imatrix 的来源信息写入 KV overrides，让量化后的 GGUF 可追溯。

### 13.2 tensor_override 组装

```cpp
if (!tensor_type_opts.empty()) {
    t_override.reserve(tensor_type_opts.size() + 1);
    for (const auto & tt : tensor_type_opts) {
        t_override.push_back({tt.name.c_str(), tt.type});
    }
    t_override.push_back({nullptr, GGML_TYPE_COUNT});  // 终结符
    params.tt_overrides = t_override.data();
}
```

### 13.3 prune_layers 组装

```cpp
if (!prune_layers.empty()) {
    prune_layers.push_back(-1);  // 终结符
    params.prune_layers = prune_layers.data();
}
```

### 13.4 kv_overrides 组装

```cpp
if (!kv_overrides.empty()) {
    kv_overrides.emplace_back();
    kv_overrides.back().key[0] = 0;   // 终结符
    params.kv_overrides = kv_overrides.data();
}
```

---

## 14. 文件路径与 ftype 解析（第 548-594 行）

### 14.1 两种命令行格式

```cpp
// 格式 1：<input> <ftype>           → 自动生成输出名
// 格式 2：<input> <output> <ftype>  → 显式指定输出名
```

### 14.2 自动输出名（格式 1）

```cpp
if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
    if (!params.dry_run) {
        std::string fpath;
        const size_t pos = fname_inp.find_last_of("/\\");
        if (pos != std::string::npos) {
            fpath = fname_inp.substr(0, pos + 1);
        }
        fname_out = fpath + "ggml-model-" + ftype_str;
        if (!params.keep_split) {
            fname_out += suffix;   // 加 .gguf
        }
    }
    arg_idx++;
    if (ftype_str == "COPY") {
        params.only_copy = true;
    }
}
```

自动生成 `ggml-model-Q4_K_M.gguf` 风格的输出名。

### 14.3 显式输出名（格式 2）

```cpp
else {
    fname_out = argv[arg_idx];
    if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
        fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
    }
    arg_idx++;
    if (argc <= arg_idx) { return 1; }   // 缺少 ftype
    if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) { return 1; }
    if (ftype_str == "COPY") { params.only_copy = true; }
    arg_idx++;
}
```

### 14.4 线程数解析

```cpp
if (argc > arg_idx) {
    try {
        params.nthread = std::stoi(argv[arg_idx]);
    } catch (...) { return 1; }
}
```

### 14.5 输入输出同名检查

```cpp
if (!params.dry_run) {
    if (std::error_code ec; std::filesystem::equivalent(fname_inp, fname_out, ec)) {
        fprintf(stderr, "%s: error: input and output files are the same\n", ...);
        return 1;
    }
}
```

防止覆盖原文件。

---

## 15. 执行量化与耗时统计（第 627-655 行）

### 15.1 打印开始信息

```cpp
if (params.dry_run) {
    fprintf(stderr, "%s: calculating quantization size for '%s' as %s", ...);
} else {
    fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", ...);
}
if (params.nthread > 0) {
    fprintf(stderr, " using %d threads", params.nthread);
}
fprintf(stderr, "\n");
```

### 15.2 调用核心量化

```cpp
const int64_t t_main_start_us = llama_time_us();
int64_t t_quantize_us = 0;

{
    const int64_t t_start_us = llama_time_us();

    if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
        fprintf(stderr, "%s: failed to quantize model from '%s'\n", ...);
        return 1;
    }

    t_quantize_us = llama_time_us() - t_start_us;
}
```

**这是整个文件唯一调用量化核心的地方**。`llama_model_quantize` 在 `src/llama-quant.cpp` 实现。

### 15.3 耗时报告

```cpp
{
    const int64_t t_main_end_us = llama_time_us();
    printf("\n");
    printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
    printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
}

llama_backend_free();

return 0;
```

### 15.4 后端初始化与释放

```cpp
llama_backendinit();   // 第 546 行
...
llama_backend_free();   // 第 652 行
```

初始化和释放 ggml 后端（CPU/GPU），确保量化过程有正确的运行环境。

---

## 16. 函数清单速查

| 函数 | 行号 | 作用 |
|------|------|------|
| `striequals` | 82 | 大小写不敏感字符串比较 |
| `try_parse_ftype` | 92 | 解析 ftype（名称或数字） |
| `usage` | 121 | 打印帮助并退出 |
| `load_imatrix` | 180 | 加载 imatrix 文件并归一化 |
| `prepare_imatrix` | 260 | 加载 + include/exclude 过滤 |
| `parse_ggml_type` | 302 | 解析 ggml_type |
| `parse_tensor_type` | 314 | 解析 `name=type` |
| `parse_tensor_type_file` | 346 | 从文件批量解析 `name=type` |
| `parse_layer_prune` | 363 | 解析 `L0,L1,L2` 裁剪列表 |
| `llama_quantize`（main） | 392 | 入口：参数解析 + 组装 + 调用 + 统计 |

---

## 一句话总结

> `tools/quantize/quantize.cpp` 是 `llama-quantize` 命令行工具的入口，它负责：**解析命令行选项 → 加载和过滤 imatrix → 组装 `llama_model_quantize_params`（含 imatrix/KV override/tensor override/prune） → 解析输入输出路径和 ftype → 调用 `llama_model_quantize()` → 统计耗时**。它本身不实现任何量化算法，是纯粹的"参数前端 + 编排层"。