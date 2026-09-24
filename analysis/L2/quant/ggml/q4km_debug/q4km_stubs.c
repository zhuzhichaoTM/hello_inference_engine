#include <stddef.h>
#include <stdint.h>

// 独立环境的 ggml_row_size 桩：仅供 q4km_test 使用。
// 真实实现在 ggml.c；此处按 ggml_type 枚举提供 Q4_K（=12）的正确行大小。
#define GGML_TYPE_Q4_K 12

size_t ggml_row_size(int type, int64_t ne) {
    if (type == GGML_TYPE_Q4_K) {
        return (size_t)(ne / 256) * 144; // 8 sub-blocks: 4B d/dmin + 12B scales + 128B qs
    }
    return 0; // 其他类型本测试不使用
}

// ggml_validate_row_data 的依赖桩（原 ggml-quants.c 尾部会引用）
size_t ggml_type_size(int type) { (void)type; return 0; }
