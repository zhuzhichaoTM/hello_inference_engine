# GGUF

GGUF 是一种文件格式，用于存储可供 GGML 以及基于 GGML 的执行器（executor）进行推理的模型。GGUF 是一种二进制格式，旨在快速加载和保存模型，并便于读取。模型通常使用 PyTorch 或其他框架开发，然后转换为 GGUF 供 GGML 使用。

它是 GGML、GGMF 和 GGJT 的后继文件格式，通过包含加载模型所需的全部信息来做到无歧义。它还被设计为可扩展的，因此可以向模型添加新信息而不破坏兼容性。

有关 GGUF 背后设计动机的更多信息，请参阅[历史背景](#historical-state-of-affairs)。

## 规范

GGUF 是一种基于现有 GGJT 的格式，但对格式做了若干改动，使其更具可扩展性且更易使用。期望具备以下特性：

- 单文件部署：模型可以轻松分发和加载，不需要任何外部文件来提供额外信息。
- 可扩展：可以向基于 GGML 的执行器添加新特性，也可以向 GGUF 模型添加新信息，而不破坏与现有模型的兼容性。
- `mmap` 兼容：可以使用 `mmap` 加载模型，以实现快速加载和保存。
- 易于使用：无论使用哪种语言，只需少量代码即可轻松加载和保存模型，无需外部库。
- 信息完整：加载模型所需的全部信息都包含在模型文件内，用户无需提供任何额外信息。

GGJT 与 GGUF 的关键区别在于，超参数（现在称为元数据）采用了键值结构，而不是无类型值的列表。这样可以添加新元数据而不破坏与现有模型的兼容性，并可以为模型标注可能对推理或识别模型有用的额外信息。

### GGUF 命名约定

GGUF 遵循 `[<Sidecar>]<BaseName><SizeLabel><FineTune><Version><Encoding><Type><Shard>.gguf` 的命名约定，其中每个组成部分在存在时以 `-` 分隔。这样做最终是为了让人们能一眼就获得模型最重要的细节。由于现有 gguf 文件名多种多样，这一约定并不保证在实际使用中可以被完美解析。

各组成部分如下：
1. **Sidecar**：（可选）前缀，用于标明该文件是随基础模型一同加载的辅助模块，而不是独立模型。存在时位于文件名最前面，其后跟 `-`。按惯例使用小写。
    - `mmproj`：多模态投影器（与基础 LLM 配合使用的视觉/音频编码器和投影层）
    - `mtp`：多 Token 预测（Multi-Token Prediction）头（推测解码草稿模块，用于与架构和版本匹配的基础模型一同加载）。注意，MTP 权重常常可以分发在基础模型内部，这种情况下就没有单独的 `mtp-` 伴随文件。
1. **BaseName**：模型基础类型或架构的描述性名称。
    - 这可以从 gguf 元数据 `general.basename` 推导而来，将空格替换为短横线。
1. **SizeLabel**：参数权重类别（对排行榜很有用），表示为 `<expertCount>x<count><scale-prefix>`
    - 如果可用，这可以从 gguf 元数据 `general.size_label` 推导而来；如果缺失，则可以计算得出。
    - 计数支持带单个字母量级前缀的圆整小数，用于辅助表示如下所示的浮点指数
      - `Q`：千万亿（Quadrillion）参数。
      - `T`：万亿（Trillion）参数。
      - `B`：十亿（Billion）参数。
      - `M`：百万（Million）参数。
      - `K`：千（Thousand）参数。
    - 可按需追加 `-<attributes><count><scale-prefix>`，以表示其他关注的属性
1. **FineTune**：模型微调目标的描述性名称（例如 Chat、Instruct 等……）
    - 这可以从 gguf 元数据 `general.finetune` 推导而来，将空格替换为短横线。
1. **Version**：（可选）表示模型版本号，格式为 `v<Major>.<Minor>`
    - 如果模型缺少版本号，则假定为 `v1.0`（首次公开发布）
    - 这可以从 gguf 元数据 `general.version` 推导而来
1. **Encoding**：表示应用于该模型的权重编码方案。但内容、类型混合方式和排列方式由用户代码决定，可能因项目需求而异。
1. **Type**：表示 gguf 文件的种类及其预期用途
    - 如果缺失，则文件默认为典型的 gguf 张量模型文件
    - `LoRA`：GGUF 文件是一个 LoRA 适配器
    - `vocab`：仅含词表数据和元数据的 GGUF 文件
1. **Shard**：（可选）表示该模型已被拆分为多个分片，格式为 `<ShardNum>-of-<ShardTotal>`。
    - *ShardNum*：分片在该模型中的位置。必须为 5 位数字，用零填充。
      - 分片编号始终从 `00001` 开始（例如第一个分片始终是 `00001-of-XXXXX`，而不是 `00000-of-XXXXX`）。
    - *ShardTotal*：该模型中分片的总数。必须为 5 位数字，用零填充。


#### 验证上述命名约定

至少，所有模型文件都应包含 BaseName、SizeLabel、Version，以便能轻松验证其是否符合 GGUF 命名约定。这种问题的一个例子是：如果省略 Version，Encoding 很容易被误认为是 FineTune。

可使用以下正则表达式进行验证 `^(?:(?<Sidecar>mmproj|mtp)-)?(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$` 它会检查是否以正确的顺序至少包含 BaseName、SizeLabel 和 Version。

例如：

  * `Mixtral-8x7B-v0.1-KQ2.gguf`:
    - 模型名称：Mixtral
    - 专家数量：8
    - 参数数量：7B
    - 版本号：v0.1
    - 权重编码方案：KQ2

  * `Hermes-2-Pro-Llama-3-8B-F16.gguf`:
    - 模型名称：Hermes 2 Pro Llama 3
    - 专家数量：0
    - 参数数量：8B
    - 版本号：v1.0
    - 权重编码方案：F16
    - 分片：N/A

  * `Grok-100B-v1.0-Q4_0-00003-of-00009.gguf`
    - 模型名称：Grok
    - 专家数量：0
    - 参数数量：100B
    - 版本号：v1.0
    - 权重编码方案：Q4_0
    - 分片：共 9 个分片中的第 3 个

  * `mtp-Qwen3-27B-v1.0-Q4_K_M.gguf`
    - 伴随模块：mtp（多 Token 预测草稿模块）
    - 模型名称：Qwen3
    - 专家数量：0
    - 参数数量：27B（主模型的——伴随模块的张量更小）
    - 版本号：v1.0
    - 权重编码方案：Q4_K_M

  * `mmproj-Qwen2-VL-7B-v1.0-F16.gguf`
    - 伴随模块：mmproj（多模态投影器）
    - 模型名称：Qwen2-VL
    - 专家数量：0
    - 参数数量：7B（主模型的——伴随模块的张量更小）
    - 版本号：v1.0
    - 权重编码方案：F16


<details><summary>Node.js 正则表达式函数示例</summary>

```js
#!/usr/bin/env node
const ggufRegex = /^(?:(?<Sidecar>mmproj|mtp)-)?(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$/;

function parseGGUFFilename(filename) {
  const match = ggufRegex.exec(filename);
  if (!match)
    return null;
  const {Sidecar = null, BaseName = null, SizeLabel = null, FineTune = null, Version = "v1.0", Encoding = null, Type = null, Shard = null} = match.groups;
  return {Sidecar: Sidecar, BaseName: BaseName, SizeLabel: SizeLabel, FineTune: FineTune, Version: Version, Encoding: Encoding, Type: Type, Shard: Shard};
}

const testCases = [
  {filename: 'Mixtral-8x7B-v0.1-KQ2.gguf',                         expected: { Sidecar: null,   BaseName: 'Mixtral',              SizeLabel: '8x7B',     FineTune: null, Version: 'v0.1',   Encoding: 'KQ2',    Type: null, Shard: null}},
  {filename: 'Grok-100B-v1.0-Q4_0-00003-of-00009.gguf',            expected: { Sidecar: null,   BaseName: 'Grok',                 SizeLabel: '100B',     FineTune: null, Version: 'v1.0',   Encoding: 'Q4_0',   Type: null, Shard: "00003-of-00009"}},
  {filename: 'Hermes-2-Pro-Llama-3-8B-v1.0-F16.gguf',              expected: { Sidecar: null,   BaseName: 'Hermes-2-Pro-Llama-3', SizeLabel: '8B',       FineTune: null, Version: 'v1.0',   Encoding: 'F16',    Type: null, Shard: null}},
  {filename: 'Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf', expected: { Sidecar: null,   BaseName: 'Phi-3-mini',           SizeLabel: '3.8B-ContextLength4k', FineTune: 'instruct', Version: 'v1.0', Encoding: null,    Type: null, Shard: null}},
  {filename: 'mtp-Qwen3-27B-v1.0-Q4_K_M.gguf',                     expected: { Sidecar: 'mtp',  BaseName: 'Qwen3',                SizeLabel: '27B',      FineTune: null, Version: 'v1.0',   Encoding: 'Q4_K_M', Type: null, Shard: null}},
  {filename: 'mmproj-Qwen2-VL-7B-v1.0-F16.gguf',                   expected: { Sidecar: 'mmproj', BaseName: 'Qwen2-VL',           SizeLabel: '7B',       FineTune: null, Version: 'v1.0',   Encoding: 'F16',    Type: null, Shard: null}},
  {filename: 'not-a-known-arrangement.gguf',                       expected: null},
];

testCases.forEach(({ filename, expected }) => {
  const result = parseGGUFFilename(filename);
  const passed = JSON.stringify(result) === JSON.stringify(expected);
  console.log(`${filename}: ${passed ? "PASS" : "FAIL"}`);
  if (!passed) {
      console.log(result);
      console.log(expected);
  }
});
```

</details>


### 文件结构

![image](https://github.com/ggerganov/ggml/assets/1991296/c3623641-3a1d-408e-bfaf-1b7c4e16aa63)
*示意图由 [@mishig25](https://github.com/mishig25) 绘制（GGUF v3）*

GGUF 文件的结构如下。它们使用 `general.alignment` 元数据字段指定的全局对齐，下文称为 `ALIGNMENT`。在需要的地方，文件会用 `0x00` 字节填充到 `general.alignment` 的下一个倍数。

除非另有说明，字段（包括数组）都是按顺序写入的，不进行对齐。

模型默认采用小端序。它们也可以采用大端序，以供大端序计算机使用；在这种情况下，所有值（包括元数据值和张量）也将采用大端序。在撰写本文时，还无法判断模型是否为大端序；这可能会在未来的版本中得到解决。如果未提供其他信息，请假定模型为小端序。

```c
enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT   = 40,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```

## 标准化键值对

以下键值对是标准化的。随着发现更多用例，这个列表未来可能会扩展。在可能的情况下，名称与原始模型定义保持一致，以便于两者之间的映射。

并非所有键值对都是必需的，但都建议使用。必需的键以粗体显示。对于省略的键值对，读取方应假定该值未知，并视情况采用默认值或报错。

社区可以开发自己的键值对来携带额外数据。不过，这些键值对应使用相关社区名称作为命名空间，以避免冲突。例如，`rustformers` 社区可能会使用 `rustformers.` 作为其所有键的前缀。

如果某个社区键被广泛使用，它可能会被提升为标准化键。

按照惯例，除非另有说明，大多数计数/长度等均为 `uint64`。这是为了将来能够支持更大的模型。有些模型的值可能使用 `uint32`；建议读取方同时支持两者。

### 通用

#### 必需

- **`general.architecture: string`**：描述该模型实现了哪种架构。全部为小写 ASCII，只允许 `[a-z0-9]+` 字符。已知取值包括：
  - `llama`
  - `mpt`
  - `gptneox`
  - `gptj`
  - `gpt2`
  - `bloom`
  - `falcon`
  - `mamba`
  - `rwkv`
- **`general.quantization_version: uint32`**：量化格式的版本。如果模型未量化（即没有张量被量化），则不是必需的。如果有任何张量被量化，则此键*必须*存在。这与张量本身的量化方案是分开的；量化版本可能会在不改变方案名称的情况下发生变化（例如量化方案是 Q5_K，而量化版本是 4）。
- **`general.alignment: uint32`**：如上文所述要使用的全局对齐。它可以变化以支持不同的对齐方案，但必须是 8 的倍数。有些写入方可能不写入对齐值。如果**未**指定对齐，则假定为 `32`。

#### 通用元数据

- `general.name: string`：模型名称。这应该是一个人类可读的名称，可用于标识模型。它应在定义该模型的社区内唯一。
- `general.author: string`：模型的作者。
- `general.version: string`：模型的版本。
- `general.organization: string`：模型所属的组织。
- `general.basename: string`：模型的基础模型名称/架构
- `general.finetune: string`：基础模型针对什么进行了优化。
- `general.description: string`：模型的自由格式描述，可包含其他字段未涵盖的任何内容
- `general.quantized_by: string`：对该模型进行量化的个人名称
- `general.size_label: string`：模型的规模类别，例如权重数量和专家数量。（对排行榜很有用）
- `general.license: string`：模型的许可证，以 [SPDX 许可证表达式](https://spdx.github.io/spdx-spec/v2-draft/SPDX-license-expressions/) 表示（例如 `"MIT OR Apache-2.0`）。不要包含任何其他信息，例如许可证正文或许可证的 URL。
- `general.license.name: string`：对人友好的许可证名称
- `general.license.link: string`：许可证的 URL。
- `general.url: string`：模型主页的 URL。可以是 GitHub 仓库、论文等。
- `general.doi: string`：数字对象标识符（DOI）https://www.doi.org/
- `general.uuid: string`：[通用唯一标识符](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.repo_url: string`：模型仓库的 URL，例如 GitHub 仓库或 HuggingFace 仓库
- `general.tags: string[]`：标签列表，可用作搜索引擎或社交媒体的搜索词
- `general.languages: string[]`：模型会说哪些语言。编码为 [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) 两字母代码
- `general.datasets: string[]`：模型训练所用数据集的链接或引用
- `general.file_type: uint32`：一个枚举值，描述文件中大多数张量的类型。可选；可以从张量类型推断。
  - `ALL_F32 = 0`
  - `MOSTLY_F16 = 1`
  - `MOSTLY_Q4_0 = 2`
  - `MOSTLY_Q4_1 = 3`
  - `MOSTLY_Q4_1_SOME_F16 = 4`
  - `MOSTLY_Q4_2 = 5` (support removed)
  - `MOSTLY_Q4_3 = 6` (support removed)
  - `MOSTLY_Q8_0 = 7`
  - `MOSTLY_Q5_0 = 8`
  - `MOSTLY_Q5_1 = 9`
  - `MOSTLY_Q2_K = 10`
  - `MOSTLY_Q3_K_S = 11`
  - `MOSTLY_Q3_K_M = 12`
  - `MOSTLY_Q3_K_L = 13`
  - `MOSTLY_Q4_K_S = 14`
  - `MOSTLY_Q4_K_M = 15`
  - `MOSTLY_Q5_K_S = 16`
  - `MOSTLY_Q5_K_M = 17`
  - `MOSTLY_Q6_K = 18`

#### 来源元数据

关于该模型来源的信息。这对于追踪模型的出处，以及在模型被修改时找到原始来源很有用。例如，对于从 GGML 转换而来的模型，这些键将指向被转换的源模型。

- `general.source.url: string`：模型主页来源的 URL。可以是 GitHub 仓库、论文等。
- `general.source.doi: string`：来源数字对象标识符（DOI）https://www.doi.org/
- `general.source.uuid: string`：来源[通用唯一标识符](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.source.repo_url: string`：模型仓库来源的 URL，例如 GitHub 仓库或 HuggingFace 仓库

- `general.base_model.count: uint32`：父模型的数量
- `general.base_model.{id}.name: string`：父模型的名称。
- `general.base_model.{id}.author: string`：父模型的作者。
- `general.base_model.{id}.version: string`：父模型的版本。
- `general.base_model.{id}.organization: string`：父模型所属的组织。
- `general.base_model.{id}.url: string`：父模型主页来源的 URL。可以是 GitHub 仓库、论文等。
- `general.base_model.{id}.doi: string`：父模型数字对象标识符（DOI）https://www.doi.org/
- `general.base_model.{id}.uuid: string`：父模型[通用唯一标识符](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.base_model.{id}.repo_url: string`：父模型仓库来源的 URL，例如 GitHub 仓库或 HuggingFace 仓库

### LLM

在下文中，`[llm]` 用于代指特定 LLM 架构的名称。例如，LLaMA 使用 `llama`，MPT 使用 `mpt`，等等。如果在某个架构的章节中提到某个键，则该键对该架构是必需的，但并非所有键对所有架构都是必需的。更多信息请查阅相关章节。

- `[llm].context_length: uint64`：也称为 `n_ctx`。模型训练时所用上下文（以 token 计）的长度。对大多数架构来说，这是输入长度的硬性上限。像 RWKV 这类不依赖 Transformer 式注意力的架构或许能够处理更长的输入，但这并不保证。
- `[llm].embedding_length: uint64`：也称为 `n_embd`。嵌入层大小。
- `[llm].block_count: uint64`：注意力+前馈层的块数（即 LLM 的主体部分）。不包括输入层或嵌入层。
- `[llm].feed_forward_length: uint64`：也称为 `n_ff`。前馈层的长度。
- `[llm].use_parallel_residual: bool`：是否应使用并行残差逻辑。
- `[llm].tensor_data_layout: string`：当模型转换为 GGUF 时，可能会重新排列张量以提升性能。该键描述张量数据的布局。它不是必需的；如果不存在，则假定为 `reference`。
  - `reference`：张量按与原始模型相同的顺序排列
  - 每个架构的更多选项可以在其各自的章节中找到
- `[llm].expert_count: uint32`：MoE 模型中的专家数量（对非 MoE 架构可选）。
- `[llm].expert_used_count: uint32`：每个 token 评估期间使用的专家数量（对非 MoE 架构可选）。

#### 注意力

- `[llm].attention.head_count: uint64`：也称为 `n_head`。注意力头数量。
- `[llm].attention.head_count_kv: uint64`：分组查询注意力（Grouped-Query-Attention）中每组使用的头数。如果不存在，或者存在且等于 `[llm].attention.head_count`，则该模型不使用 GQA。
- `[llm].attention.max_alibi_bias: float32`：ALiBI 使用的最大偏置。
- `[llm].attention.clamp_kqv: float32`：值（`C`），用于把 `Q`、`K` 和 `V` 张量的值限制在（`[-C, C]`）区间内。
- `[llm].attention.layer_norm_epsilon: float32`：层归一化 epsilon。
- `[llm].attention.layer_norm_rms_epsilon: float32`：层 RMS 归一化 epsilon。
- `[llm].attention.key_length: uint32`：键头（key head）的可选大小 $d_k$。如果未指定，则为 `n_embd / n_head`。
- `[llm].attention.value_length: uint32`：值头（value head）的可选大小 $d_v$。如果未指定，则为 `n_embd / n_head`。

#### RoPE

- `[llm].rope.dimension_count: uint64`：RoPE 的旋转维度数量。
- `[llm].rope.freq_base: float32`：RoPE 的基频。

##### 缩放

以下键描述 RoPE 缩放参数：

- `[llm].rope.scaling.type: string`：可以是 `none`、`linear` 或 `yarn`。
- `[llm].rope.scaling.factor: float32`：用于调整上下文长度的 RoPE 缩放因子。
- `[llm].rope.scaling.original_context_length: uint32_t`：基础模型的原始上下文长度。
- `[llm].rope.scaling.finetuned: bool`：如果模型使用 RoPE 缩放进行过微调，则为 true。

注意，较旧的模型可能没有这些键，而可能使用以下键：

- `[llm].rope.scale_linear: float32`：用于调整上下文长度的 RoPE 线性缩放因子。

建议模型尽可能使用较新的键，因为它们更灵活，并支持更复杂的缩放方案。执行器将需要无限期地同时支持两者。

#### SSM

- `[llm].ssm.conv_kernel: uint32`：滚动/移位状态的大小。
- `[llm].ssm.inner_size: uint32`：状态的嵌入大小。
- `[llm].ssm.state_size: uint32`：循环状态的大小。
- `[llm].ssm.time_step_rank: uint32`：时间步的秩。

#### 模型

以下各节描述每种模型架构的元数据。其中指定的每个键都*必须*存在。

##### LLaMA

- `llama.context_length`
- `llama.embedding_length`
- `llama.block_count`
- `llama.feed_forward_length`
- `llama.rope.dimension_count`
- `llama.attention.head_count`
- `llama.attention.layer_norm_rms_epsilon`

###### 可选

- `llama.rope.scale`
- `llama.attention.head_count_kv`
- `llama.tensor_data_layout`:
  - `Meta AI original pth`:
    ```python
    def permute(weights: NDArray, n_head: int) -> NDArray:
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))
    ```
- `llama.expert_count`
- `llama.expert_used_count`

##### MPT

- `mpt.context_length`
- `mpt.embedding_length`
- `mpt.block_count`
- `mpt.attention.head_count`
- `mpt.attention.alibi_bias_max`
- `mpt.attention.clip_kqv`
- `mpt.attention.layer_norm_epsilon`

##### GPT-NeoX

- `gptneox.context_length`
- `gptneox.embedding_length`
- `gptneox.block_count`
- `gptneox.use_parallel_residual`
- `gptneox.rope.dimension_count`
- `gptneox.attention.head_count`
- `gptneox.attention.layer_norm_epsilon`

###### 可选

- `gptneox.rope.scale`

##### GPT-J

- `gptj.context_length`
- `gptj.embedding_length`
- `gptj.block_count`
- `gptj.rope.dimension_count`
- `gptj.attention.head_count`
- `gptj.attention.layer_norm_epsilon`

###### 可选

- `gptj.rope.scale`

##### GPT-2

- `gpt2.context_length`
- `gpt2.embedding_length`
- `gpt2.block_count`
- `gpt2.attention.head_count`
- `gpt2.attention.layer_norm_epsilon`

##### BLOOM

- `bloom.context_length`
- `bloom.embedding_length`
- `bloom.block_count`
- `bloom.feed_forward_length`
- `bloom.attention.head_count`
- `bloom.attention.layer_norm_epsilon`

##### Falcon

- `falcon.context_length`
- `falcon.embedding_length`
- `falcon.block_count`
- `falcon.attention.head_count`
- `falcon.attention.head_count_kv`
- `falcon.attention.use_norm`
- `falcon.attention.layer_norm_epsilon`

###### 可选

- `falcon.tensor_data_layout`:

  - `jploski`（Falcon 原始 GGML 实现的作者）：

    ```python
    # The original query_key_value tensor contains n_head_kv "kv groups",
    # each consisting of n_head/n_head_kv query weights followed by one key
    # and one value weight (shared by all query heads in the kv group).
    # This layout makes it a big pain to work with in GGML.
    # So we rearrange them here,, so that we have n_head query weights
    # followed by n_head_kv key weights followed by n_head_kv value weights,
    # in contiguous fashion.

    if "query_key_value" in src:
        qkv = model[src].view(
            n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)

        q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
        k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
        v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)

        model[src] = torch.cat((q,k,v)).reshape_as(model[src])
    ```

##### Mamba

- `mamba.context_length`
- `mamba.embedding_length`
- `mamba.block_count`
- `mamba.ssm.conv_kernel`
- `mamba.ssm.inner_size`
- `mamba.ssm.state_size`
- `mamba.ssm.time_step_rank`
- `mamba.attention.layer_norm_rms_epsilon`

##### RWKV

词汇表大小与 `head` 矩阵的行数相同。

- `rwkv.architecture_version: uint32`：目前唯一允许的值是 4。版本 5 预计会在未来某个时候出现。
- `rwkv.context_length: uint64`：训练或微调时使用的上下文长度。RWKV 能够处理比此上限更长的上下文，但输出质量可能会下降。
- `rwkv.block_count: uint64`
- `rwkv.embedding_length: uint64`
- `rwkv.feed_forward_length: uint64`

##### Whisper

未定义类型的键应假定与 `llm.` 键共享定义。
（例如，`whisper.context_length` 等同于 `llm.context_length`。）
这是因为它们都是 Transformer 模型。

- `whisper.encoder.context_length`
- `whisper.encoder.embedding_length`
- `whisper.encoder.block_count`
- `whisper.encoder.mels_count: uint64`
- `whisper.encoder.attention.head_count`

- `whisper.decoder.context_length`
- `whisper.decoder.embedding_length`
- `whisper.decoder.block_count`
- `whisper.decoder.attention.head_count`

#### 提示

**TODO**：包含提示格式，和/或关于其使用方式（指令、对话、自动补全等）的元数据。

### LoRA

**TODO**：弄清楚 LoRA 需要哪些元数据。可能期望的特性：
- 与现有模型精确匹配，以免被误用
- 标记为 LoRA，这样执行器就不会尝试单独运行它
- be marked as a LoRA so executors won't try to run it by itself

这应该作为一种架构，还是应该共享原始模型的详细信息，并附加字段将其标记为 LoRA？

### 分词器

以下键用于描述模型的分词器。建议模型作者尽可能支持其中的更多键，因为这可以在受支持的执行器上获得更好的分词质量。

#### GGML

GGML 支持一种嵌入式词汇表，可用于模型的推理，但使用该词汇表的分词实现（即 `llama.cpp` 的分词器）的准确度可能低于该模型使用的原始分词器。当有更准确且受支持的分词器可用时，应改用该分词器。

它不保证在各模型之间是标准化的，未来也可能变化。建议模型作者尽可能使用更标准化的分词器。

- `tokenizer.ggml.model: string`：分词器模型的名称。
  - `llama`：Llama 风格的 SentencePiece（token 和分数从 HF 的 `tokenizer.model` 中提取）
  - `replit`：Replit 风格的 SentencePiece（token 和分数从 HF 的 `spiece.model` 中提取）
  - `gpt2`：GPT-2 / GPT-NeoX 风格的 BPE（token 从 HF 的 `tokenizer.json` 中提取）
  - `rwkv`：RWKV 分词器
- `tokenizer.ggml.tokens: array[string]`：按模型使用的 token ID 索引的 token 列表。
- `tokenizer.ggml.scores: array[float32]`：如果存在，表示每个 token 的分数/概率。如果不存在，则假定所有 token 具有相同概率。如果存在，其长度和索引必须与 `tokens` 相同。
- `tokenizer.ggml.token_type: array[int32]`：token 类型（1=normal，2=unknown，3=control，4=user defined，5=unused，6=byte）。如果存在，其长度和索引必须与 `tokens` 相同。
- `tokenizer.ggml.merges: array[string]`：如果存在，表示分词器的合并规则。如果不存在，则假定 token 是原子的。
- `tokenizer.ggml.added_tokens: array[string]`：如果存在，表示训练之后添加的 token。

##### 特殊 token

- `tokenizer.ggml.bos_token_id: uint32`：序列起始标记
- `tokenizer.ggml.eos_token_id: uint32`：序列结束标记
- `tokenizer.ggml.unknown_token_id: uint32`：未知 token
- `tokenizer.ggml.separator_token_id: uint32`：分隔符 token
- `tokenizer.ggml.padding_token_id: uint32`：填充 token

#### Hugging Face

Hugging Face 维护着自己的 `tokenizers` 库，支持多种多样的分词器。如果你的执行器使用这个库，可能可以直接使用模型自带的分词器。

- `tokenizer.huggingface.json: string`：给定模型的 HF `tokenizer.json` 的全部内容（例如 <https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/tokenizer.json>）。包含此项是为了与直接支持 HF 分词器的执行器兼容。

#### 其他

也可以使用其他分词器，但它们不一定已标准化。它们可能是特定于执行器的。随着它们被发现/进一步开发，将在此处记录。

- `tokenizer.rwkv.world: string`：一个 RWKV World 分词器，类似[这个](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt)。该文本文件应原样包含。
- `tokenizer.chat_template : string`：一个 Jinja 模板，用于指定模型期望的输入格式。更多细节请参见：<https://huggingface.co/docs/transformers/main/en/chat_templating>

### 计算图

这是未来的扩展，仍需讨论，并且可能需要新的 GGUF 版本。在撰写本文时，主要的障碍是计算图格式的稳定化。

GGML 节点的示例计算图可以包含在模型自身中，使执行器无需自己提供该架构的实现即可运行模型。这将使不同执行器之间获得更一致的体验，并允许在无需执行器实现这些架构的情况下支持更复杂的架构。

## 标准化张量名称

为了最小化复杂性并最大化兼容性，建议使用 Transformer 架构的模型为其张量采用以下命名约定：

### 基础层

`AA.weight` `AA.bias`

其中 `AA` 可以是：

- `token_embd`：Token 嵌入层
- `pos_embd`：位置嵌入层
- `output_norm`：输出归一化层
- `output`：输出层

### 注意力与前馈层块

`blk.N.BB.weight` `blk.N.BB.bias`

其中 N 表示该层所属的块编号，`BB` 可以是：

- `attn_norm`：注意力归一化层
- `attn_norm_2`：注意力归一化层
- `attn_qkv`：注意力查询-键-值层
- `attn_q`：注意力查询层
- `attn_k`：注意力键层
- `attn_v`：注意力值层
- `attn_output`：注意力输出层

- `ffn_norm`：前馈网络归一化层
- `ffn_up`：前馈网络 "up" 层
- `ffn_gate`：前馈网络 "gate" 层
- `ffn_down`：前馈网络 "down" 层
- `ffn_gate_inp`：MoE 模型中前馈网络的专家路由层
- `ffn_gate_exp`：MoE 模型中每个专家的前馈网络 "gate" 层
- `ffn_down_exp`：MoE 模型中每个专家的前馈网络 "down" 层
- `ffn_up_exp`：MoE 模型中每个专家的前馈网络 "up" 层

- `ssm_in`：状态空间模型的输入投影层
- `ssm_conv1d`：状态空间模型的滚动/移位层
- `ssm_x`：状态空间模型的选择性参数化层
- `ssm_a`：状态空间模型的状态压缩层
- `ssm_d`：状态空间模型的跳跃连接层
- `ssm_dt`：状态空间模型的时间步层
- `ssm_out`：状态空间模型的输出投影层

## 版本历史

本文档会持续更新以描述元数据的当前状态，这些变更不会在提交之外单独记录。

不过，格式*本身*已经发生了变化。以下各节描述格式本身的变化。

### v3

增加大端序支持。

### v2

大多数可计数的值（长度等）从 `uint32` 改为 `uint64`，以便将来支持更大的模型。

### v1

初始版本。

## 历史背景

以下信息仅为提供背景，并非理解本文档其余部分所必需。

### 概述

目前，LLM 领域存在三种 GGML 文件格式：

- **GGML**（无版本号）：基线格式，没有版本号或对齐。
- **GGMF**（有版本号）：与 GGML 相同，但带有版本号。只有一个版本存在。
- **GGJT**：将张量对齐，以便与 `mmap` 配合使用，因为这要求对齐。v1、v2 和 v3 完全相同，但较新的版本使用了与先前版本不兼容的不同量化方案。

GGML 主要被 `ggml` 中的示例使用，而 GGJT 被 `llama.cpp` 模型使用。其他执行器可以使用这三种格式中的任意一种，但这并不被“官方”支持。

这些格式共享相同的基本结构：

- 一个魔数，可选带版本号
- 模型特定的超参数，包括
  - 关于模型的元数据，例如层数、头数等。
  - 一个 `ftype`，描述大多数张量的类型，
    - 对于 GGML 文件，量化版本编码在 `ftype` 中，即除以 1000
- 一个嵌入式词汇表，是一个带长度前缀的字符串列表。GGMF/GGJT 格式在字符串旁边嵌入一个 float32 分数。
- 最后是一个张量列表，包含带长度前缀的名称、类型以及（在 GGJT 的情况下是对齐的）张量数据

值得注意的是，这种结构既不能标识模型属于哪种架构，也没有为更改超参数结构提供任何灵活性。这意味着添加新超参数的唯一方法是将它们追加到列表末尾，而这对于现有模型是一种破坏性变更。

### 缺点

遗憾的是，在过去几个月中，现有模型暴露出了一些明显的问题：

- 无法确定给定模型对应哪种架构，因为该信息并不存在
  - 同样，现有程序在遇到新架构时也无法智能地失败
- 添加或删除任何新的超参数都是破坏性变更，读取方若不使用启发式方法就无法检测到
- 每种模型架构都需要自己的转换脚本，以转换为其架构对应的 GGML 变体
- 在不破坏格式结构的情况下保持向后兼容，需要一些巧妙的技巧，例如将量化版本打包进 ftype，而这些技巧不保证能被读取方/写入方识别，而且在两种格式之间也不一致

### 为什么不用其他格式？

还有一些其他格式可以使用，但问题包括：

- 需要额外的依赖才能加载或保存模型，这在 C 环境中很复杂
- 对 4 位量化的支持有限或完全没有
- 既有的文化预期（例如模型是目录还是文件）
- 缺乏对嵌入式词汇表的支持
- 缺乏对未来发展方向的控制

最终，在可预见的未来，GGUF 很可能仍是必要的，而拥有一个文档完善且被所有执行器支持的单一格式，比为了满足 GGML 的需求而扭曲现有格式要好。
