#!/usr/bin/env bash
# ============================================================================
# run_closure.sh —— 七步闭环一键执行（Shell 编排层）
# ----------------------------------------------------------------------------
# 与 `python3 -m perfkit all` 的区别：
#   · 本脚本负责「真机侧」的动作：环境快照、探针自检、服务启停、监控采样
#   · perfkit 负责「分析侧」的动作：画像、剖面、诊断、方案、验证、报告
#   二者组合起来才是完整的闭环；只有分析没有真机采集，收益无法归因。
#
# 用法：
#   ./scripts/run_closure.sh --run qwen3.8-27b-decode-opt
#   ./scripts/run_closure.sh --run my-run --mode live          # 真机模式
#   ./scripts/run_closure.sh --run my-run --skip-probe         # 跳过探针
#   ./scripts/run_closure.sh --run my-run --service-url http://127.0.0.1:8000
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ID=""
CONFIG="configs"
MODE="replay"
SKIP_PROBE=0
SERVICE_URL="http://127.0.0.1:8000"
PY="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)         RUN_ID="$2"; shift 2 ;;
        --config)      CONFIG="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --skip-probe)  SKIP_PROBE=1; shift ;;
        --service-url) SERVICE_URL="$2"; shift 2 ;;
        -h|--help)     sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${RUN_ID}" ]]; then
    # 未指定则从 task 配置里取，与 perfkit 的行为保持一致
    RUN_ID="$("${PY}" - "${CONFIG}" <<'PY'
import sys
sys.path.insert(0, '.')
try:
    from perfkit.config import load_yaml
    print(load_yaml(f'{sys.argv[1]}/task_decode_optimize.yaml')['task']['id'])
except Exception:
    print('default-run')
PY
)"
fi

LOGDIR="${ROOT}/output/${RUN_ID}/logs"
mkdir -p "${LOGDIR}"

# ---------------------------------------------------------------------------
# 日志与阶段管理：每一步都落盘，失败时明确停在哪一步
# ---------------------------------------------------------------------------
STEP_NO=0
TOTAL_STEPS=7
step() {
    STEP_NO=$((STEP_NO + 1))
    echo
    echo "───────────────────────────────────────────────────────────────"
    echo "  [${STEP_NO}/${TOTAL_STEPS}] $*"
    echo "───────────────────────────────────────────────────────────────"
}
ok()   { echo "  ✓ $*"; }
warn() { echo "  ! $*" >&2; }
die()  { echo "  ✗ $*" >&2; echo; echo "闭环在步骤 ${STEP_NO} 中断。修复后可从对应子命令单独续跑。" >&2; exit 1; }

echo "###############################################################"
echo "# perfkit 七步闭环 · Shell 编排"
echo "# run_id   ${RUN_ID}
# config   ${CONFIG}"
echo "# mode     ${MODE}"
echo "# 日志目录 ${LOGDIR}"
echo "###############################################################"

# ===========================================================================
step "第0步 环境准备：采集环境画像"
# ===========================================================================
if bash "${ROOT}/scripts/collect_env.sh" --out "${ROOT}/output/_env" \
        > "${LOGDIR}/00_env.log" 2>&1; then
    ok "环境画像已落盘 output/_env/"
else
    warn "环境采集失败（非致命），详见 ${LOGDIR}/00_env.log"
fi

# ===========================================================================
step "第0步 环境准备：构建并运行带宽探针"
# ===========================================================================
if [[ "${SKIP_PROBE}" == "1" ]]; then
    ok "按要求跳过探针"
else
    if bash "${ROOT}/scripts/build.sh" > "${LOGDIR}/00_build.log" 2>&1; then
        ok "探针构建成功"
        # 用一个小矩阵快速自检，确认探针本身可用（全尺寸测试在 live 模式做）
        # 通过 perfkit probe 统一驱动：后端决定用哪个探针与什么参数，
        # 并把结果落盘到 output/_env/probe.json（baseline 会自动读取）。
        if "${PY}" -m perfkit probe --config "${CONFIG}" --run "${RUN_ID}" \
                --size-gib 2 > "${LOGDIR}/00_probe.log" 2>&1; then
            ok "探针完成 → ${LOGDIR}/00_probe.log"
            grep -E "标称|实测达成|内核效率" "${LOGDIR}/00_probe.log" \
                | sed 's/^/    /' || true
        else
            warn "探针运行失败（非致命），详见 ${LOGDIR}/00_probe.log"
            warn "提示：报告 §0 的「环境探针实测」将显示为未采集"
        fi
    else
        warn "探针构建失败，详见 ${LOGDIR}/00_build.log"
    fi
fi

# ===========================================================================
step "第1步 问题定义"
# ===========================================================================
"${PY}" -m perfkit define --config "${CONFIG}" --run "${RUN_ID}" \
    2>&1 | tee "${LOGDIR}/01_define.log" || die "define 失败"

# ===========================================================================
step "第2步 性能基线"
# ===========================================================================
if [[ "${MODE}" == "live" ]]; then
    # 真机模式：先确认服务在线，再采集
    if curl -sf --max-time 5 "${SERVICE_URL}/v1/models" > /dev/null 2>&1; then
        ok "推理服务在线：${SERVICE_URL}"
        curl -s "${SERVICE_URL}/v1/models" > "${LOGDIR}/02_models.json" || true
    else
        warn "推理服务未响应 ${SERVICE_URL}；请先启动 llama-server"
        warn "启动命令见 output/${RUN_ID}/start_optimized.sh 或 configs/plan_*.yaml"
    fi
    # 采样显存/利用率/温度（60 秒），这是显存峰值的唯一可信来源
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  · 采样 nvidia-smi dmon 60s ..."
        nvidia-smi dmon -s u,m,p,t -c 60 -d 1 > "${LOGDIR}/02_dmon.log" 2>&1 || true
        ok "监控采样 → ${LOGDIR}/02_dmon.log"
    fi
fi
"${PY}" -m perfkit baseline --config "${CONFIG}" --run "${RUN_ID}" --mode "${MODE}" \
    2>&1 | tee "${LOGDIR}/02_baseline.log" || die "baseline 失败"

# ===========================================================================
step "第3步 性能剖面采集"
# ===========================================================================
"${PY}" -m perfkit profiling --config "${CONFIG}" --run "${RUN_ID}" --mode "${MODE}" \
    2>&1 | tee "${LOGDIR}/03_profiling.log" || die "profiling 失败"

if [[ "${MODE}" == "live" ]] && command -v nsys >/dev/null 2>&1; then
    warn "检测到 nsys。kernel 级时间线请手动执行："
    warn "  nsys profile --trace=cuda,nvtx -o ${LOGDIR}/nsys_llama <llama-server 启动命令>"
    warn "  nsys stats ${LOGDIR}/nsys_llama.nsys-rep"
fi

# ===========================================================================
step "第4步 瓶颈定位"
# ===========================================================================
"${PY}" -m perfkit diagnose --config "${CONFIG}" --run "${RUN_ID}" \
    2>&1 | tee "${LOGDIR}/04_diagnose.log" || die "diagnose 失败"

# ===========================================================================
step "第5步 优化方案生成"
# ===========================================================================
"${PY}" -m perfkit optimize --config "${CONFIG}" --run "${RUN_ID}" \
    2>&1 | tee "${LOGDIR}/05_optimize.log" || die "optimize 失败"
if [[ -x "${ROOT}/output/${RUN_ID}/start_optimized.sh" ]]; then
    ok "启动脚本已生成：output/${RUN_ID}/start_optimized.sh"
fi

# ===========================================================================
step "第6步 效果验证与收益量化"
# ===========================================================================
"${PY}" -m perfkit verify --config "${CONFIG}" --run "${RUN_ID}" \
    2>&1 | tee "${LOGDIR}/06_verify.log" || die "verify 失败"

# ===========================================================================
step "第7步 报告导出与可视化"
# ===========================================================================
"${PY}" -m perfkit report --config "${CONFIG}" --run "${RUN_ID}" \
    2>&1 | tee "${LOGDIR}/07_report.log" || die "report 失败"
"${PY}" -m perfkit visualize --config "${CONFIG}" --run "${RUN_ID}" \
    2>&1 | tee -a "${LOGDIR}/07_report.log" || die "visualize 失败"

# ---------------------------------------------------------------------------
echo
echo "###############################################################"
echo "# 七步闭环完成"
echo "###############################################################"
echo "  报告      output/${RUN_ID}/report.md"
echo "  图表      output/${RUN_ID}/charts.html"
echo "  启动脚本  output/${RUN_ID}/start_optimized.sh"
echo "  过程日志  ${LOGDIR}/"
echo
echo "  真机验证（在目标设备上）："
echo "    bash output/${RUN_ID}/start_optimized.sh"
echo "    ./scripts/run_closure.sh --run ${RUN_ID} --mode live"
echo "###############################################################"
