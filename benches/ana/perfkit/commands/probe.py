"""
perfkit probe —— 硬件带宽探针（工程闭环第 0 步的「探针自检」环节）
==============================================================================
探针回答一个框架其它部分都依赖的问题：**这台机器的内存/显存带宽实测是多少？**

为什么单独设一个子命令：
  · 探针测的是**硬件**，与模型无关 —— 一次测量可被多次运行复用，
    不该每次都跟着 baseline 重跑；
  · 探针结果要落盘（output/_env/probe.json），供 baseline 读取并写进报告 §0；
  · 编译/运行探针在 CPU 上要几十秒到几分钟，单独执行才能控制节奏。

不同后端用不同探针（由 backends.py 决定）：
  · CPU / 统一内存 → cpp/ggml_probe（按 q4_0 真实布局测，不是 fp32）
  · NVIDIA GPU     → cpp/cuda_bw_probe（必须用纯读带宽反推 Decode 上界）
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ..config import PerfConfig, save_json


def _probe_env(cfg: PerfConfig) -> dict:
    """
    采集探针运行时的有效性上下文。

    与 baseline 的 environment 段同构 —— 两处用同一套判据，
    报告里才能用同一句话解释「这个数字能不能引用」。
    """
    out: dict = {"backend": cfg.backend.kind}
    mv = cfg.device.get("measurement_validity")
    if mv:
        out["declared_validity"] = mv
    try:
        la = os.getloadavg()
        cores = cfg.device.get("compute", {}).get("cpu_cores") or os.cpu_count() or 1
        out["load_avg"] = [round(x, 2) for x in la]
        out["cpu_cores"] = cores
        out["load_per_core"] = round(la[0] / cores, 2)
        if la[0] > cores:
            out["warning"] = (
                f"1 分钟负载 {la[0]:.1f} 超过物理核数 {cores}，"
                f"本次探针结果可能偏低（共享内存带宽的后端上，"
                f"探针与真实推理一样会被外部负载压低）。"
                f"建议在机器空闲时重测。")
    except (OSError, AttributeError):
        pass
    return out


def _find_probe(cfg: PerfConfig) -> Path | None:
    """
    定位该后端对应的探针可执行文件。

    不同后端用不同探针（CPU → ggml_probe，GPU → cuda_bw_probe），
    文件名由后端声明。找不到时给出构建提示，而不是静默用一个错的探针 ——
    用 CPU 探针测 GPU 会得到一个「显存带宽」的假数字。
    """
    name = getattr(cfg.backend, "probe_binary", "ggml_probe")
    p = cfg.root / "build" / name
    if p.exists():
        return p
    return None


def _parse_probe_output(text: str, backend) -> dict:
    """
    按后端声明的指标定义从探针 stdout 提取结果。

    解析规则收在 backend 里（`probe_metrics()`），而不是在这里写 if：
    CPU 探针与 GPU 探针测的是不同的东西（主机内存路径 vs 显存路径），
    指标名与正则都不同。这样新增后端时只改后端类。
    """
    out: dict = {}
    for spec in backend.probe_metrics():
        m = re.search(spec["pattern"], text)
        if m:
            out[spec["key"]] = float(m.group(1))
    # 内核效率（GEMV ÷ 顺序读）：CPU 探针直接打印，GPU 探针同理
    m = re.search(r"([\d.]+) / [\d.]+ = ([\d.]+)%（最优线程数", text)
    if m:
        out["kernel_efficiency"] = round(float(m.group(2)) / 100.0, 4)
    else:
        # GPU 探针的写法：内核效率 = xx.x%（相对纯读上界）
        m = re.search(r"内核效率 = ([\d.]+)%", text)
        if m:
            out["kernel_efficiency"] = round(float(m.group(1)) / 100.0, 4)
    m = re.search(r"反推取用\s+([\d.]+) GB/s", text)
    if m:
        out["best_gbps"] = float(m.group(1))
    elif out.get("stream_file") or out.get("read_bw"):
        # 结论段文案可能调整，这里按语义回退：反推取用 = 硬件上限
        out["best_gbps"] = out.get("stream_file") or out.get("read_bw")
    # 机器可解析的汇总行：三层参照（标量下界 / SIMD 可达 / 效率）
    for _k in ("gemv_scalar_gbps", "gemv_scalar_efficiency",
               "gemv_simd_gbps", "gemv_simd_efficiency"):
        m = re.search(rf"{_k}=([\d.]+)", text)
        if m:
            out[_k] = float(m.group(1))
    # 内核效率：优先取 SIMD（判断优化空间的正确基准），
    # 其次取探针直接打印的比值（旧格式），最后退回标量效率
    if "gemv_simd_efficiency" in out:
        out["kernel_efficiency"] = out["gemv_simd_efficiency"]
        out["kernel_efficiency_basis"] = "simd"
    elif "kernel_efficiency" not in out and "gemv_scalar_efficiency" in out:
        out["kernel_efficiency"] = out["gemv_scalar_efficiency"]
        out["kernel_efficiency_basis"] = "scalar"
    # 兼容：某些版本只有匿名缓冲区（无 mmap 模式）
    if "stream_file" not in out and "stream" in out:
        out["stream_file"] = out["stream"]
        out["stream_file_source"] = "匿名缓冲区（该探针版本无 mmap 模式）"
    return out


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    backend = cfg.backend

    probe = _find_probe(cfg)
    if not probe:
        name = getattr(backend, "probe_binary", "ggml_probe")
        print(f"[probe] 未找到 build/{name}（后端 {backend.label} 使用的探针）",
              file=sys.stderr)
        print("        构建方式：", file=sys.stderr)
        if name == "cuda_bw_probe":
            print("          bash scripts/build.sh --cuda   # 需 nvcc", file=sys.stderr)
        else:
            print("          bash scripts/build.sh          # 或 make build",
                  file=sys.stderr)
        return 3

    # 后端决定用哪个探针与什么参数
    argv = backend.bandwidth_probe_command(str(probe), size_gib=args.size_gib)
    if not argv:
        print(f"[probe] 后端 {backend.label} 无可用探针"
              f"（需厂商工具链），跳过。", file=sys.stderr)
        return 0

    # 模型相关参数由后端决定：
    #   CPU → `--file` mmap 真实权重（最贴近 llama.cpp 的实际访存形态）
    #   GPU → 只用 `--weight-gib`（显存探针不接受主机文件路径）
    model_file = cfg.model.get("file")
    _mf = model_file if (model_file and Path(model_file).exists()) else None
    if model_file and not _mf:
        print(f"[probe] 提示：模型文件不存在（{model_file}），"
              f"跳过 mmap 口径，只用匿名缓冲区", file=sys.stderr)
    from ..config import weight_bytes as _wbf
    _wg = _wbf(cfg.model) / 1024 ** 3
    argv += backend.probe_extra_args(_mf, _wg)

    cmd = " ".join(argv)
    print(f"[probe] 后端 {backend.label}")
    print(f"[probe] 命令 {cmd}")
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=args.timeout)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        return proc.returncode

    result = _parse_probe_output(proc.stdout, backend)
    result["cmd"] = cmd
    result["backend"] = backend.kind
    # 记录采集设备：探针文件按后端分，但同一后端可能有不同机器。
    # 报告会展示这一行，让「在 A 机采集、用在 B 机配置里」这种错配可见。
    result["device"] = cfg.device.get("name")
    result["returncode"] = proc.returncode
    # 探针测的是硬件，但**共享内存带宽的后端**（CPU / 统一内存）上，
    # 结果强依赖采集时的系统负载。不记录负载，这个数字事后无法判断是否可信 ——
    # 实践中出现过「探针报 13.7 GB/s，而真实推理反推出 19.9 GB/s」的情况，
    # 根因就是探针在负载下采集。与 baseline 用同一套判据。
    result["environment"] = _probe_env(cfg)
    if not result:
        print("[probe] 警告：未能从输出中解析出任何指标，请检查探针版本",
              file=sys.stderr)

    outdir = cfg.root / "output" / "_env"
    # 按后端分开存：CPU 探针与 GPU 探针测的是**不同的东西**，
    # 混在一个文件里会导致「拿 CPU 的带宽去校验 GPU 的达成率」这类错误
    # （实践中确实发生过：GPU 报告显示了 CPU 探针的数字，得出 4963% 的荒谬比值）。
    path = save_json(result, outdir / f"probe_{backend.kind}.json")
    (outdir / f"probe_{cfg.run_id}.log").write_text(
        f"$ {cmd}\n\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
        encoding="utf-8")

    print(f"[probe] 已落盘 → {path}")
    env = result.get("environment") or {}
    if env.get("load_avg"):
        print(f"[probe] 采集时系统负载 {env['load_avg'][0]}"
              f"（{env['cpu_cores']} 核，每核 {env['load_per_core']}）")
    if env.get("warning"):
        print(f"[probe] ⚠ {env['warning']}")
    declared = cfg.device.get("memory", {}).get("mem_bandwidth_gbps_peak")
    got = result.get("stream_file") or result.get("stream")
    if declared and got:
        print(f"[probe] 标称 {declared} GB/s，实测达成 {got:.2f} GB/s "
              f"（{got / declared * 100:.0f}%）")
    if result.get("kernel_efficiency") is not None:
        ke = result["kernel_efficiency"]
        print(f"[probe] 内核效率 {ke * 100:.1f}% → "
              + ("内核受限，SIMD/内核优化有空间" if ke < 0.5
                 else "访存受限，只能减少每步搬运字节"))
    print(f"[probe] 提示：baseline 会自动读取该文件并写入报告 §0")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("probe",
                       help="硬件带宽探针：实测内存/显存带宽，落盘供 baseline 引用")
    p.add_argument("--size-gib", type=float, default=2.0,
                   help="缓冲区大小（GPU 建议 ≥4，需超过 L2 才测得到真带宽）")
    p.add_argument("--timeout", type=int, default=1800,
                   help="探针超时（秒）")
    p.set_defaults(func=main)
    return p
