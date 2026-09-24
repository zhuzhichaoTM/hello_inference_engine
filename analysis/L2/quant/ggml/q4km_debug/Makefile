# 构建/对比测试 — Q4_K 裁剪实现
# 用法：
#   make -f doc/q4km_debug/Makefile            # 构建并运行单测
#   make -f doc/q4km_debug/Makefile compare    # 构建并与原实现逐位对比
#   make -f doc/q4km_debug/Makefile lldb       # 启动 LLDB 调试会话

CC       ?= clang
ROOT     := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST))))/../..)
GGML_SRC := $(ROOT)/ggml/src
INC      := -I$(GGML_SRC) -I$(ROOT)/ggml/include
CFLAGS   ?= -O0 -g -Wall -Wextra -fno-omit-frame-pointer
BUILD    := $(ROOT)/build-q4km-debug

.PHONY: all run compare clean lldb

all: $(BUILD)/q4km_test
run: all
	$(BUILD)/q4km_test --debug

$(BUILD)/q4km_test: doc/q4km_debug/q4km_test.c $(GGML_SRC)/ggml-quants_q4_k_m.c doc/q4km_debug/q4km_stubs.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ -lm

# ---- 与原实现的逐位对比 -------------------------------------------
# 原实现的 q4_K 片段已生成到 q4km_orig_frag.c（符号重命名 *_orig），
# 由 tools/gen_orig_frag.sh 自动再生成（若原文件更新）
$(BUILD)/q4km_orig_frag.o: doc/q4km_debug/q4km_orig_frag.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

compare: $(BUILD)/q4km_orig_frag.o
	$(CC) $(CFLAGS) $(INC) \
	  -DCOMPARE_WITH_ORIG \
	  doc/q4km_debug/q4km_test.c $(GGML_SRC)/ggml-quants_q4_k_m.c doc/q4km_debug/q4km_stubs.c \
	  $(BUILD)/q4km_orig_frag.o \
	  -o $(BUILD)/q4km_compare -lm
	$(BUILD)/q4km_compare

lldb: $(BUILD)/q4km_test
	lldb -- \
	  -o 'b make_qkx2_quants' \
	  -o 'b quantize_row_q4_K_ref' \
	  -o run \
	  -- $(BUILD)/q4km_test --debug

clean:
	rm -rf $(BUILD)
