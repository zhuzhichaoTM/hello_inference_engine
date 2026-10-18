#!/usr/bin/env bash
# ============================================================================
# build.sh —— 构建 perfkit 的本地探针（C++ / CUDA）
# ----------------------------------------------------------------------------
# 对应工程闭环第 0 步「环境准备」的编译环境验证环节。
#
# 用法：
#   ./scripts/build.sh              # 构建 CPU 探针（必选）
#   ./scripts/build.sh --cuda       # 额外构建 CUDA 探针（需 nvcc）
#   ./scripts/build.sh --clean      # 清理构建产物
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
SRC_DIR="${ROOT}/cpp"

WITH_CUDA=0
CLEAN=0
for arg in "$@"; do
    case "${arg}" in
        --cuda)  WITH_CUDA=1 ;;
        --clean) CLEAN=1 ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "未知参数: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ "${CLEAN}" == "1" ]]; then
    rm -rf "${BUILD_DIR}"
    echo "[build] 已清理 ${BUILD_DIR}"
    exit 0
fi

mkdir -p "${BUILD_DIR}"

# ---- 探测可用的 C++ 编译器 ------------------------------------------------
CXX_BIN="${CXX:-}"
if [[ -z "${CXX_BIN}" ]]; then
    for c in clang++ g++ c++; do
        if command -v "${c}" >/dev/null 2>&1; then CXX_BIN="${c}"; break; fi
    done
fi
if [[ -z "${CXX_BIN}" ]]; then
    echo "[build] 错误：未找到 C++ 编译器（clang++ / g++ / c++）" >&2
    exit 1
fi

# ---- 按目标架构选择 SIMD 开关 --------------------------------------------
# 探针默认走标量参考实现，加 SIMD 只影响编译器对循环的自动向量化程度，
# 因此这里只在明确支持的架构上开启，避免在旧机器上生成非法指令。
ARCH="$(uname -m)"
ARCH_FLAGS=()
case "${ARCH}" in
    arm64|aarch64)
        # dotprod 能显著加速 q4_0 的 nibble 点积；旧核不支持，故只开到 v8.2
        ARCH_FLAGS=(-march=armv8.2-a+dotprod)
        ;;
    x86_64|amd64)
        ARCH_FLAGS=(-mavx2 -mfma)
        ;;
    *)
        echo "[build] 提示：未识别的架构 ${ARCH}，使用通用编译选项"
        ;;
esac

echo "[build] 编译器   ${CXX_BIN}"
echo "[build] 架构     ${ARCH} ${ARCH_FLAGS[*]:-（通用）}"
echo "[build] 产物目录 ${BUILD_DIR}"

# ---- CPU 探针（必选）-----------------------------------------------------
"${CXX_BIN}" -O3 -std=c++17 -pthread -Wall -Wextra \
    "${ARCH_FLAGS[@]}" \
    "${SRC_DIR}/ggml_probe.cpp" -o "${BUILD_DIR}/ggml_probe"
echo "[build] ✓ ggml_probe"

# ---- CUDA 探针（可选）----------------------------------------------------
if [[ "${WITH_CUDA}" == "1" ]]; then
    NVCC="${NVCC:-nvcc}"
    if ! command -v "${NVCC}" >/dev/null 2>&1; then
        echo "[build] 警告：未找到 nvcc，跳过 CUDA 探针" >&2
        echo "        提示：export PATH=/usr/local/cuda/bin:\$PATH" >&2
        exit 0
    fi
    # 用 GPU 的实际计算能力编译，避免在目标卡上出现 no kernel image 错误
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
    fi
    CC="${CC:-89}"
    echo "[build] nvcc 编译 CUDA 探针，-arch=sm_${CC}"
    "${NVCC}" -O3 -std=c++14 -arch="sm_${CC}" \
        "${SRC_DIR}/cuda_bw_probe.cu" -o "${BUILD_DIR}/cuda_bw_probe"
    echo "[build] ✓ cuda_bw_probe"
fi

echo
echo "[build] 完成。快速自检："
echo "  ${BUILD_DIR}/ggml_probe --mode all --size-gib 1 --rows 4096 --cols 4096 --iters 5"
