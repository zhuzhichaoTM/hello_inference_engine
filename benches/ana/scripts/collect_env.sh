#!/usr/bin/env bash
# ============================================================================
# collect_env.sh —— 环境画像采集（工程闭环第 0 步）
# ----------------------------------------------------------------------------
# 把「设备画像」从人工填写变成自动采集：输出可直接粘贴进
# configs/device_*.yaml 的字段，并落盘一份完整证据到 output/_env/。
#
# 覆盖三类平台：
#   · NVIDIA GPU（Linux/Windows）—— nvidia-smi
#   · Apple Silicon（macOS）—— system_profiler
#   · 通用 CPU —— lscpu / sysctl / /proc/cpuinfo
#
# 用法：
#   ./scripts/collect_env.sh                 # 打印 + 落盘
#   ./scripts/collect_env.sh --out DIR       # 指定输出目录
#   ./scripts/collect_env.sh --yaml          # 只输出 YAML 片段（便于粘贴）
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/output/_env"
YAML_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT_DIR="$2"; shift 2 ;;
        --yaml) YAML_ONLY=1; shift ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${OUT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# ---------------------------------------------------------------------------
# 采集原语：把命令输出同时写进证据文件与临时变量
# ---------------------------------------------------------------------------
EVIDENCE="${OUT_DIR}/env_${STAMP}.txt"
: > "${EVIDENCE}"

have() { command -v "$1" >/dev/null 2>&1; }

capture() {
    # capture <标签> <命令...>
    local label="$1"; shift
    {
        echo "### ${label}"
        echo "\$ $*"
        if have "$1"; then
            "$@" 2>&1 || echo "(命令返回非零: $*)"
        else
            echo "(未安装: $1)"
        fi
        echo
    } >> "${EVIDENCE}"
}

say() { [[ "${YAML_ONLY}" == "1" ]] || printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# 通用信息
# ---------------------------------------------------------------------------
capture "操作系统"   uname -a
capture "发行版"     sh -c 'cat /etc/os-release 2>/dev/null || sw_vers 2>/dev/null || true'
capture "主机型号"   sh -c 'sysctl -n machdep.cpu.brand_string 2>/dev/null || hostname'

say "==============================================================="
say "环境画像采集 · $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
say "证据文件：${EVIDENCE}"
say "==============================================================="
say

# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
if have lscpu; then
    capture "CPU (lscpu)" lscpu
    CPU_MODEL="$(lscpu | awk -F': +' '/Model name/{print $2; exit}')"
    CPU_CORES="$(lscpu | awk -F': +' '/^CPU\(s\)/{print $2; exit}')"
    CPU_MHZ="$(lscpu | awk -F': +' '/CPU max MHz/{print $2; exit}')"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    capture "CPU (sysctl)" sysctl -n machdep.cpu.brand_string
    CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 0)"
    CPU_MHZ="$(sysctl -n hw.cpufrequency_max 2>/dev/null | awk '{print $1/1000000}' || echo '')"
else
    capture "CPU (proc)" sh -c 'grep -m1 "model name" /proc/cpuinfo'
    CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo unknown)"
    CPU_CORES="$(nproc 2>/dev/null || echo 0)"
    CPU_MHZ=""
fi

# 检查 ARM 的量化加速指令集 —— 直接决定 CPU 后端能跑多快
ARM_FEATURES=""
if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        ARM_FEATURES="$(sysctl -n hw.optional.arm.FEAT_DotProd 2>/dev/null | \
            awk '{print ($1==1)?"dotprod":""}')"
        [[ "$(sysctl -n hw.optional.arm.FEAT_I8MM 2>/dev/null)" == "1" ]] && ARM_FEATURES="${ARM_FEATURES:+${ARM_FEATURES},}i8mm"
        [[ "$(sysctl -n hw.optional.arm.FEAT_BF16 2>/dev/null)" == "1" ]] && ARM_FEATURES="${ARM_FEATURES:+${ARM_FEATURES},}bf16"
        [[ "$(sysctl -n hw.optional.arm.FEAT_SME 2>/dev/null)" == "1" ]] && ARM_FEATURES="${ARM_FEATURES:+${ARM_FEATURES},}sme"
    else
        ARM_FEATURES="$(grep -m1 'Features' /proc/cpuinfo 2>/dev/null | \
            tr ' ' '\n' | grep -E '^(asimd|dotprod|i8mm|bf16|sve|sve2|sme)$' | paste -sd, - || echo '')"
    fi
fi

say "[CPU]"
say "  型号    ${CPU_MODEL}"
say "  逻辑核  ${CPU_CORES}"
[[ -n "${CPU_MHZ}" ]] && say "  主频    ${CPU_MHZ} MHz"
[[ -n "${ARM_FEATURES}" ]] && say "  SIMD    ${ARM_FEATURES}"
say

# ---------------------------------------------------------------------------
# 内存
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" == "Darwin" ]]; then
    RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
else
    RAM_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    RAM_BYTES=$((RAM_KB * 1024))
fi
RAM_GIB="$(awk -v b="${RAM_BYTES}" 'BEGIN{printf "%.1f", b/1073741824}')"
capture "内存" sh -c 'free -h 2>/dev/null || vm_stat 2>/dev/null || true'
say "[内存]"
say "  物理内存  ${RAM_GIB} GiB"
say

# ---------------------------------------------------------------------------
# GPU（NVIDIA / Apple）
# ---------------------------------------------------------------------------
GPU_KIND="none"
VRAM_MIB=""
BW_GBPS=""
CC=""
DRIVER=""

if have nvidia-smi; then
    GPU_KIND="nvidia"
    capture "NVIDIA GPU" nvidia-smi
    capture "NVIDIA GPU 详细" nvidia-smi -q
    read -r GPU_NAME VRAM_MIB DRIVER CC < <(
        nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
                   --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1" "$2" "$3" "$4}'
    )
    # 显存带宽 = 2 × 等效频率(GHz) × 位宽(bit) / 8  —— 注意要乘 2（DDR）
    BW_GBPS="$(nvidia-smi --query-gpu=clocks.max.memory,memory.bus_width \
                --format=csv,noheader,nounits 2>/dev/null | head -1 | \
                awk -F', ' '{printf "%.0f", 2*$1*1e6*$2/8/1e9}' || echo '')"
    CUDA_VER="$(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | awk '{print $3}' || echo '')"

    say "[GPU]"
    say "  型号      ${GPU_NAME}"
    say "  显存      $(awk -v m="${VRAM_MIB}" 'BEGIN{printf "%.1f", m/1024}') GiB"
    say "  驱动      ${DRIVER}"
    say "  CUDA      ${CUDA_VER:-未报告}"
    say "  计算能力  ${CC}"
    say "  标称带宽  ${BW_GBPS:-?} GB/s"
    say

elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    GPU_KIND="apple"
    capture "Apple GPU" system_profiler SPDisplaysDataType
    say "[GPU] Apple Silicon 统一内存（CPU/GPU 共享同一物理内存）"
    say "  内存带宽需用探针实测：build/ggml_probe --mode stream"
    say
fi

# ---------------------------------------------------------------------------
# 构建工具链
# ---------------------------------------------------------------------------
capture "编译器" sh -c 'clang++ --version 2>/dev/null | head -2; g++ --version 2>/dev/null | head -1'
capture "CMake" cmake --version
capture "nvcc" nvcc --version

say "[工具链]"
for t in cmake nvcc clang++ g++ python3; do
    if have "${t}"; then
        say "  ✓ ${t}"
    else
        say "  ✗ ${t}（缺失）"
    fi
done
say

# ---------------------------------------------------------------------------
# 输出 YAML 片段
# ---------------------------------------------------------------------------
YAML_FILE="${OUT_DIR}/device_snapshot_${STAMP}.yaml"
# kind 映射：nvidia→gpu（独立显存）；apple→hybrid（统一内存）；
# 其余情况（无独显 / 纯 CPU 机器）→ cpu，便于后续按 kind 分支处理
case "${GPU_KIND}" in
    nvidia) KIND="gpu" ;;
    apple)  KIND="hybrid" ;;
    *)      KIND="cpu" ;;
esac
cat > "${YAML_FILE}" <<YAML
# 由 scripts/collect_env.sh 自动采集于 ${STAMP}
# 用法：核对后合并进 configs/device_<name>.yaml
device:
  name: ${GPU_NAME:-${CPU_MODEL}}
  kind: ${KIND}
memory:
  vram_total_mib: ${VRAM_MIB:-null}
  mem_bandwidth_gbps_peak: ${BW_GBPS:-null}   # 标称值；实测请用 ggml_probe
  unified_memory: $([[ "${KIND}" == "hybrid" ]] && echo true || echo false)
compute:
  compute_capability: "${CC:-}"
host:
  cpu_model: "${CPU_MODEL}"
  cpu_cores: ${CPU_CORES}
  ram_gib: ${RAM_GIB}
  arm_features: "${ARM_FEATURES}"
software:
  driver: "${DRIVER:-}"
collected_at: "${STAMP}"
collected_by: "scripts/collect_env.sh"
YAML

say "==============================================================="
say "已生成设备画像快照："
say "  ${YAML_FILE}"
say "完整证据（含 nvidia-smi -q / lscpu 全量输出）："
say "  ${EVIDENCE}"
say "==============================================================="
say
say "下一步："
say "  1) 用 build/ggml_probe --mode stream 实测内存带宽，替换 mem_bandwidth_gbps_peak"
say "  2) 把快照字段合并进 configs/device_*.yaml"
say "  3) python3 -m perfkit define --run <你的 run id>"

if [[ "${YAML_ONLY}" == "1" ]]; then
    cat "${YAML_FILE}"
fi
