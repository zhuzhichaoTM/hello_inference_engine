"""
perfkit · 硬件后端抽象（Hardware Backend Abstraction）
==============================================================================
把「这套框架跑在什么硬件上」从散落各处的 if 判断，收敛成一组后端类。

为什么需要这一层
------------------------------------------------------------------------------
最初框架是按 GPU 写的：内存读 nvidia-smi、bench 传 -ngl 99、
带宽用 cuda_bw_probe、报告里写「显存」。搬到 CPU 上时这些假设全部失效 ——
不是跑不起来，而是**会跑出错误的结论**：
  · 把 host RAM 当显存算预算（CPU 后端权重就在系统内存里，没有「卸载」概念）
  · 用 PCIe 往返代价判定「CPU 回退」（CPU 后端根本不存在回退）
  · 报告里说「SM 利用率」（CPU 没有 SM）

因此后端层要回答的问题是：**在这类硬件上，什么测量手段是有效的、
什么结论是成立的、什么话是能说的。**

扩展方式
------------------------------------------------------------------------------
新增一类硬件（如某厂商 NPU）时：
  1) 继承 `Backend`，实现必要的钩子（都有合理默认值，按需覆盖）；
  2) 在 `BACKENDS` 里注册 `kind` → 类；
  3) 在 `configs/device_*.yaml` 里把 `kind` 设成对应值。
框架其余部分零改动。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# 数据契约
# ============================================================================

@dataclass
class MemoryReading:
    """一次内存/显存快照。"""
    used_bytes: int | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    source: str = ""
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"source": self.source}
        for k in ("used_bytes", "total_bytes", "free_bytes"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
                d[k.replace("_bytes", "_gib")] = round(v / 1024 ** 3, 3)
        d.update(self.raw)
        return d


@dataclass
class ThermalReading:
    """温度/频率快照。CPU 与 GPU 的可得字段差别很大，故用自由字典。"""
    temp_c: float | None = None
    source: str = ""
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"temp_c": self.temp_c, "source": self.source, **self.raw}


# ============================================================================
# 基类
# ============================================================================

class Backend:
    """
    一类硬件后端的统一接口。

    所有钩子都有**保守默认值**：宁可返回 None 让上层标注「待采集」，
    也不要返回一个看起来合理的编造值 —— 后者会污染整条证据链。
    """

    kind: str = "unknown"
    label: str = "未知后端"
    #: 该后端下「加速器内存」的称呼（报告与图表直接使用）
    memory_label: str = "内存"
    #: 该后端下「加速器利用率」的称呼
    utilization_label: str = "利用率"

    @property
    def peak_metric_label(self) -> str:
        """
        报告/图表里「峰值内存」这一指标的显示名。

        各后端的内存语义不同，直接拼「峰值」+ memory_label 会出现
        「峰值App 内存配额」这类别扭的说法，因此在后端层定好完整名称。
        """
        return {
            "显存": "峰值显存",
            "内存": "峰值内存",
            "统一内存": "峰值内存（统一内存）",
            "App 内存配额": "峰值内存（配额内）",
            "加速器内存": "峰值加速器内存",
        }.get(self.memory_label, f"峰值{self.memory_label}")

    def __init__(self, device: dict):
        self.device = device
        self.mem = device.get("memory", {}) or {}
        self.pt = device.get("power_thermal", {}) or {}
        self.sw = device.get("software", {}) or {}
        self.constraints = device.get("constraints", {}) or {}

    # ---------------------------------------------------------------- 内存
    def mem_total_bytes(self) -> int | None:
        """
        该后端下「可用于承载模型的内存」总量（字节）。

        这是显存/内存预算的分母。GPU 取显存容量，CPU 取系统内存 ——
        两者语义不同但预算模型一致，所以统一成这一个方法。
        """
        for key in ("vram_total_mib", "ram_total_mib", "mem_total_mib"):
            v = self.mem.get(key)
            if v:
                return int(v) * 1024 * 1024
        for key in ("ram_total_gb", "vram_total_gb"):
            v = self.mem.get(key)
            if v:
                return int(float(v) * 1024 ** 3)
        return None

    def mem_snapshot(self) -> MemoryReading | None:
        """运行时内存快照。默认返回 None（表示该后端无标准采集手段）。"""
        return None

    #: 峰值内存是否靠「运行期间轮询采样」获得。
    #: GPU 是（nvidia-smi 读的是整卡已用显存，能反映被测进程）；
    #: CPU 不是（采样只能看到 perfkit 自己，必须用 RUSAGE_CHILDREN）。
    peak_via_sampling: bool = True

    def peak_after_children(self) -> int | None:
        """
        已退出子进程的**峰值内存**（字节）。默认 None。

        CPU 后端必须用这个而不是采样：CPU 上的「峰值内存」是推理进程的 RSS，
        而采样线程只能看到 perfkit 自己；子进程的峰值由内核在
        `RUSAGE_CHILDREN.ru_maxrss` 里保留，退出后仍可读。
        """
        return None

    # ------------------------------------------------------------ 基准测试
    def accel_flags(self) -> list[str]:
        """
        llama.cpp 的加速器卸载参数。

        GPU：把全部层放到显卡上（`-ngl 99`）；
        CPU：显式 `-ngl 0`，避免在装了 GPU 的机器上静默走加速器 ——
             这正是本框架在 CPU 验证场景下必须显式钉死的一点。
        """
        return []

    def bench_command(self, bench_bin: str, model_file: str, scenario: dict,
                      kv_k: str | None = None, kv_v: str | None = None,
                      extra: list[str] | None = None) -> list[str]:
        """
        构造一条 llama-bench 命令。

        参数分三层，顺序即优先级：
          1) 场景参数（-p/-n/-d/-r）—— 来自 task 的场景矩阵
          2) 后端参数（卸载标志 + `bench_flags()`）—— 来自 backends.py / 设备配置
          3) 显式覆盖（`extra`）—— 命令行 `--extra`，放在最后，llama-bench
             对重复参数取最后出现的那个，因此能覆盖前两层
        """
        cmd = [
            bench_bin, "-m", model_file,
            "-p", str(scenario.get("prompt_tokens", 512)),
            "-n", str(scenario.get("gen_tokens", 128)),
            "-r", str(scenario.get("repetitions", 3)),
            "-o", "json",
        ]
        cmd += self.accel_flags()
        cmd += self.bench_flags()
        threads = self.default_threads()
        if threads:
            cmd += ["-t", str(threads)]
        if kv_k:
            cmd += ["-ctk", kv_k]
        if kv_v:
            cmd += ["-ctv", kv_v]
        depth = scenario.get("depth")
        if depth:
            cmd += ["-d", str(depth)]
        cmd += (extra or [])
        return cmd

    def default_threads(self) -> int | None:
        """该后端推荐的线程数。CPU 取物理核数（超线程对访存密集负载无益）。"""
        return None

    def bench_extra_flags(self) -> list[str]:
        """
        `llama-bench` 的**后端相关**额外参数（不含卸载标志，那个由
        `accel_flags()` 负责）。

        为什么必须有这一层：同一个 flag 在不同后端上的收益方向可能相反。
        FlashAttention 在 GPU 上给 Prefill +10.8%（实测），在 CPU 上基本持平；
        把 `-fa on` 写死在公共代码里，会让 CPU 结果也背上一个未经其验证的假设。
        因此「这个后端测基准时该带什么」由后端自己回答。

        设备配置里的 `software.bench_flags` 可覆盖本默认值 —— 换构建/换硬件时
        不需要改代码。
        """
        return []

    def bench_flags(self) -> list[str]:
        """实际使用的基准参数：设备配置覆盖优先，其次后端默认。"""
        override = self.sw.get("bench_flags")
        if isinstance(override, list):
            return [str(x) for x in override]
        return self.bench_extra_flags()

    def bandwidth_is_shared(self) -> bool:
        """
        该后端的访存带宽是否与**主机上的其他进程共享**。

        这决定了「测量时要不要记录系统负载」：
          · 独显 GPU：显存带宽独立，主机负载基本不影响（只有 PCIe/喂数据受影响）
          · CPU / 统一内存 / SoC：带宽是整机共享的，外部负载能把吞吐压低 50%

        不做这个区分的后果是：GPU 报告上挂一条「负载 90，数据可能偏低」的
        假告警，而真正的风险（CPU 上没记负载）被淹没在噪声里。
        """
        return bool(self.mem.get("unified_memory")) or self.kind in ("cpu", "soc", "hybrid")

    # ------------------------------------------------------------ 带宽探针
    #: 该后端使用的探针可执行文件名（build/ 目录下）
    probe_binary: str = "ggml_probe"

    def probe_accepts_model_file(self) -> bool:
        """
        探针是否支持 `--file <模型文件>`（mmap 真实权重测带宽）。

        CPU 探针支持：mmap 权重路径与匿名缓冲区在页表/大页/page cache 上
        行为不同，能测到更贴近生产的口径。
        GPU 探针不支持：它测的是显存子系统，与主机文件 mmap 无关；
        传了会直接报参数错误。
        """
        return True

    def probe_extra_args(self, model_file: str | None, weight_gib: float | None
                         ) -> list[str]:
        """
        构造探针的「模型相关」参数。

        统一在这里回答「这个后端能拿模型文件做什么」：
          · CPU → 用 `--file` mmap 真实权重 + `--weight-gib` 算反推上界
          · GPU → 只用 `--weight-gib` 打印「该体积对应的 Decode 上界」表
        """
        out: list[str] = []
        if self.probe_accepts_model_file() and model_file:
            out += ["--file", model_file]
        if weight_gib is not None:
            out += ["--weight-gib", f"{weight_gib:.3f}"]
        return out

    def bandwidth_probe_command(self, probe_bin: str, size_gib: float = 2.0
                                ) -> list[str] | None:
        """该后端应使用哪个带宽探针。None 表示无可用探针。"""
        return None

    def live_metrics_command(self, seconds: int = 20) -> list[str] | None:
        """
        live 模式下采集**负载中的**硬件时序数据（利用率 / 功耗 / 温度）。

        注意这是「负载中」而不是「空闲时」：空闲采样得到的是 0% 利用率，
        对判断瓶颈毫无价值。调用方需要同时制造负载（见 profiling）。

        返回 None 表示该后端没有标准时序采集手段 —— 此时报告标注「未采集」，
        而不是拿一个静态快照冒充时序数据。
        """
        return None

    def live_metrics_label(self) -> str:
        """时序采集手段的显示名。"""
        return "—"

    def parse_live_metrics(self, text: str) -> dict:
        """
        把时序采集的原始输出解析成统计量（均值 / 峰值 / 样本数）。

        解析规则收在后端里，与 `probe_metrics()` 同理：不同后端的时序
        字段数、列序、单位都不一样，写在公共代码里就会变成一堆 if。
        """
        return {}

    def probe_metrics(self) -> list[dict]:
        """
        该后端探针输出的指标定义。

        每项：key / label（给人看的名字）/ unit / pattern（从 stdout 提取的正则）
              / note（这个数测什么、怎么用）

        为什么要定义在 backend 层：CPU 探针与 GPU 探针测的是**不同的东西**
        （前者测主机内存与 mmap 权重路径，后者测显存纯读与拷贝），
        指标名与解析方式都不同。把定义收在后端里，报告与 probe 命令
        就能用同一套代码渲染两类结果，而不是各写一遍 if。
        """
        return []

    def probe_scope_note(self) -> str:
        """该后端探针的适用范围说明（写进报告，避免读者误用）。"""
        return ""

    # ------------------------------------------------------------ 环境采集
    def env_commands(self) -> list[tuple[str, list[str]]]:
        """环境画像采集命令列表（标签, argv）。"""
        return []

    def runtime_desc(self) -> str:
        """
        「驱动 / 运行时」一栏的内容。

        各后端的这一栏含义完全不同：
          · GPU  → 驱动版本 + CUDA 版本（**性能强相关**：驱动影响 kernel 调度）
          · CPU  → 数学库后端（Accelerate / OpenBLAS / oneDNN），
                   它决定 GEMM/GEMV 走哪条实现，同样是性能相关的
          · NPU  → 厂商运行时与固件版本

        返回「—」表示该后端确实没有这一栏（此时报告会省略该行，
        而不是显示一个空值让读者困惑）。
        """
        return "—"

    def runtime_label(self) -> str:
        """该后端下这一栏的显示名（GPU 叫「驱动 / CUDA」，CPU 叫「数学库」）。"""
        return "运行时"

    def memory_measured_by(self) -> str:
        return "系统监控"

    def thermal_snapshot(self) -> ThermalReading | None:
        """温度/频率快照。默认 None —— 无标准采集手段时不编造数字。"""
        return None

    def fallback_is_cheap(self) -> bool:
        """
        「回退到 CPU 执行」在这类设备上是否廉价。

        · 独立显存 GPU：昂贵（PCIe 往返，单层即可吃掉大半解码预算）
        · 统一内存 / CPU / SoC：廉价或不存在（同一块物理内存）
        报告与诊断据此决定是否把「零回退」列为硬约束。
        """
        return False

    def notes(self) -> list[str]:
        """该后端特有的注意事项，写进报告的设备画像。"""
        return []


# ============================================================================
# CPU 后端
# ============================================================================

class CpuBackend(Backend):
    kind = "cpu"
    label = "CPU 后端"
    memory_label = "内存"
    utilization_label = "CPU 利用率"

    def __init__(self, device: dict):
        super().__init__(device)
        self._ncpu = self._detect_cpu_count()

    @staticmethod
    def _detect_cpu_count() -> int:
        try:
            import os
            n = os.cpu_count() or 0
        except Exception:
            n = 0
        # 超线程对访存密集的 GEMV 基本无益，取物理核数更接近最优
        if sys.platform == "darwin":
            try:
                p = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                   capture_output=True, text=True, timeout=5)
                if p.returncode == 0 and p.stdout.strip().isdigit():
                    return int(p.stdout.strip())
            except Exception:
                pass
        return max(1, n // 2) if n > 2 else max(1, n)

    def mem_total_bytes(self) -> int | None:
        # 优先取实测总量，其次读系统
        v = super().mem_total_bytes()
        if v:
            return v
        try:
            if sys.platform == "darwin":
                p = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True, timeout=5)
                if p.returncode == 0:
                    return int(p.stdout.strip())
            else:
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return None

    def mem_snapshot(self) -> MemoryReading | None:
        """
        当前进程的 RSS + 系统总量。

        ⚠️ 注意这里量的是**本进程**（perfkit），不是被测的 llama-bench。
        要拿「推理进程的峰值内存」必须用 `peak_after_children()` ——
        子进程的峰值只能在其退出后由 RUSAGE_CHILDREN 读到。
        这个区分很重要：拿 perfkit 自己的 17 MB RSS 当「模型峰值内存」，
        会得出「1.5B 模型只占 17 MB」这种荒谬结论。
        """
        rss = None
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux 单位是 KB，macOS 是字节
            rss = rss if sys.platform == "darwin" else rss * 1024
        except Exception:
            pass
        free = total = None
        try:
            if sys.platform == "darwin":
                p = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
                if p.returncode == 0:
                    pages = {}
                    for line in p.stdout.splitlines():
                        if ":" in line:
                            k, _, v = line.partition(":")
                            v = v.strip().rstrip(".")
                            if v.isdigit():
                                pages[k.strip()] = int(v)
                    page_size = 4096
                    free_pages = (pages.get("Pages free", 0)
                                  + pages.get("Pages inactive", 0))
                    free = free_pages * page_size
        except Exception:
            pass
        total = self.mem_total_bytes()
        if free is None and total is not None:
            free = total
        return MemoryReading(used_bytes=rss, total_bytes=total, free_bytes=free,
                             source="resource.getrusage + vm_stat"
                                    if sys.platform == "darwin" else "resource + /proc/meminfo")

    def accel_flags(self) -> list[str]:
        # 显式钉死纯 CPU：这台机器装了 Metal（Intel Iris），
        # 不写 -ngl 0 会静默走 GPU，与「验证 CPU 可用性」的目标相悖。
        return ["-ngl", "0"]

    peak_via_sampling = False

    def peak_after_children(self) -> int | None:
        """
        llama-bench 子进程的峰值 RSS（`RUSAGE_CHILDREN.ru_maxrss`）。

        这是 CPU 后端唯一可靠的「峰值内存」口径：
          · 采样线程看到的是 perfkit 自己（十几 MB），不是被测进程；
          · 内核会在子进程退出后保留其 ru_maxrss，因此跑完再读是正确的，
            而且是**整个运行期间的峰值**，不是某一瞬间的采样值。
        """
        try:
            import resource
            v = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            return int(v if sys.platform == "darwin" else v * 1024)
        except Exception:
            return None

    def bench_extra_flags(self) -> list[str]:
        """
        CPU 上**不**加 `-fa on`。

        实测（i7-4770HQ，Qwen2.5-1.5B）：`-fa on` 77.88 / `off` 71.86 /
        auto 78.42 t/s（Prefill）—— auto 与 on 持平，说明本构建的 CPU
        后端会自动选择，显式指定既无收益又多一个可能失效的假设。
        真要在 CPU 上对比 FA，用 `--extra "-fa on"` 显式覆盖。
        """
        return []

    def bandwidth_is_shared(self) -> bool:
        # CPU 的内存带宽与整机所有进程共享，这是它最本质的测量约束。
        return True

    def live_metrics_command(self, seconds: int = 20) -> list[str] | None:
        """
        CPU 没有「利用率」这种统一指标，但**频率**与**负载**能解释吞吐波动：
        降频（热/功耗墙）与外部负载是 CPU 上两个最常见的性能漂移原因。

        Linux 读 `/proc/loadavg` + `scaling_cur_freq`；macOS 无免权限频率接口，
        返回 None 让报告标注「未采集」，而不是拿 `top` 的输出冒充。
        """
        if sys.platform == "darwin":
            return None
        return ["bash", "-c",
                f"for i in $(seq 1 {max(seconds, 1)}); do "
                "cat /proc/loadavg | cut -d' ' -f1; "
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq "
                "2>/dev/null || echo 0; sleep 1; done"]

    def live_metrics_label(self) -> str:
        return "/proc/loadavg + cpufreq（负载 / 频率，1 s 粒度）"

    def parse_live_metrics(self, text: str) -> dict:
        """按「负载行 / 频率行」交替的固定格式解析。"""
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        loads, freqs = [], []
        for i in range(0, len(lines) - 1, 2):
            try:
                loads.append(float(lines[i]))
                f = float(lines[i + 1])
                if f > 0:
                    freqs.append(f / 1e6)      # kHz → GHz
            except ValueError:
                continue
        out: dict = {}
        if loads:
            out["load_avg"] = {"mean": round(sum(loads) / len(loads), 2),
                               "max": round(max(loads), 2), "n": len(loads)}
        if freqs:
            out["freq_ghz"] = {"mean": round(sum(freqs) / len(freqs), 2),
                               "min": round(min(freqs), 2),
                               "max": round(max(freqs), 2), "n": len(freqs)}
        return out

    def default_threads(self) -> int | None:
        return self._ncpu

    def bandwidth_probe_command(self, probe_bin: str, size_gib: float = 2.0
                                ) -> list[str] | None:
        # CPU 侧用 ggml_probe：它按 q4_0 的真实打包布局测 GEMV，
        # 而不是用 fp32 数组测「内存带宽」—— 后者会低估约 7 倍。
        return [probe_bin, "--mode", "all",
                "--size-gib", f"{size_gib}", "--threads", str(self._ncpu)]

    def probe_metrics(self) -> list[dict]:
        return [
            {"key": "stream_file", "label": "mmap 真实文件顺序读", "unit": "GB/s",
             "pattern": r"stream_file_gbps=([\d.]+)",
             "note": "**推荐口径**：直接 mmap 模型权重文件顺序读一遍。"
                     "这是最贴近 llama.cpp 实际访存形态的测量 —— "
                     "它反映「把权重从内存喂给计算单元」的带宽上限"},
            {"key": "stream", "label": "匿名缓冲区顺序读", "unit": "GB/s",
             "pattern": r"stream_gbps=([\d.]+)",
             "note": "**对照组**：测同一台机器在理想缓冲上的纯顺序读带宽。"
                     "与上一行的差值反映 mmap 路径的开销（页表、大页、"
                     "page cache 命中路径）"},
            {"key": "gemv_scalar_gbps", "label": "标量 GEMV（下界参照）",
             "unit": "GB/s",
             "pattern": r"gemv_scalar_gbps=([\d.]+)",
             "note": "**下界**：完全不优化（逐权重反量化 + 标量乘加）时能拿到的带宽。"
                     "它的价值是**跨架构可比** —— 任何 CPU 都能跑同一份标量代码，"
                     "因此可以用它横向对比「这台机器的朴素实现水平」"},
            {"key": "gemv_simd_gbps", "label": "SIMD GEMV（可达参照）",
             "unit": "GB/s",
             "pattern": r"gemv_simd_gbps=([\d.]+)",
             "note": "**可达参照**：一个正常优化的内核（SIMD + FMA）能拿到的带宽。"
                     "判断「生产内核还有多少空间」应该看这个，而不是标量值 —— "
                     "拿标量值当基准会把空间高估到荒谬的程度"},
            {"key": "gemv_simd_efficiency", "label": "SIMD 内核效率（÷ 顺序读）",
             "unit": "ratio",
             "pattern": r"gemv_simd_efficiency=([\d.]+)",
             "note": "**这才是判断优化方向的正确指标**：接近 1 ⇒ 已访存受限，"
                     "只能减字节；明显 <1 ⇒ 内核实现还有空间"},
            {"key": "gemv_scalar_efficiency", "label": "标量内核效率（÷ 顺序读）",
             "unit": "ratio",
             "pattern": r"gemv_scalar_efficiency=([\d.]+)",
             "note": "仅用于说明「朴素实现差多远」，不用于判断优化空间"},
        ]

    def probe_scope_note(self) -> str:
        return ("本探针测的是**主机内存子系统**（CPU 核心 → 内存控制器 → DRAM "
                "这条路径）的带宽，以及量化矩阵-向量乘在该路径上的达成率。"
                "它不测任何加速器 —— 独立显存的 GPU 需要用另一个探针"
                "（显存带宽路径与主机内存完全不同）。")

    def runtime_desc(self) -> str:
        """
        CPU 后端没有「驱动」这个概念，但有**数学库后端** —— 它决定
        GEMM/GEMV 走哪条实现（Accelerate / OpenBLAS / oneDNN / MKL），
        对性能的影响与 GPU 驱动相当。这一栏不该留空。
        """
        parts = []
        backend = self.sw.get("engine_backend") or ""
        m = re.search(r"\(([^)]*)\)", backend)
        if m:
            parts.append(m.group(1))
        if self.sw.get("math_lib"):
            parts.append(str(self.sw["math_lib"]))
        return " / ".join(parts) if parts else "—"

    def runtime_label(self) -> str:
        return "数学库 / 线程后端"

    def env_commands(self) -> list[tuple[str, list[str]]]:
        if sys.platform == "darwin":
            return [("CPU", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                    ("核数", ["sysctl", "-n", "hw.ncpu"]),
                    ("物理核", ["sysctl", "-n", "hw.physicalcpu"]),
                    ("内存", ["sysctl", "-n", "hw.memsize"]),
                    ("SIMD", ["sysctl", "-n", "machdep.cpu.features"])]
        return [("CPU", ["lscpu"]), ("内存", ["free", "-h"])]

    def thermal_snapshot(self) -> ThermalReading | None:
        # macOS 无免权限的 CPU 温度读取；Linux 可读 thermal zone。
        if sys.platform == "darwin":
            return ThermalReading(temp_c=None, source="macOS 无免权限温度接口",
                                  raw={"note": "需 powermetrics（sudo）或外部温度计"})
        try:
            zones = list(Path("/sys/class/thermal").glob("thermal_zone*"))
            temps = []
            for z in zones:
                t = z / "temp"
                if t.exists():
                    temps.append(int(t.read_text().strip()) / 1000.0)
            if temps:
                return ThermalReading(temp_c=max(temps), source="/sys/class/thermal")
        except Exception:
            pass
        return None

    def memory_measured_by(self) -> str:
        return ("sysctl hw.memsize（总量）/ getrusage（进程 RSS，峰值）"
                if sys.platform == "darwin" else
                "/proc/meminfo（总量）/ getrusage（进程 RSS，峰值）")

    def fallback_is_cheap(self) -> bool:
        # CPU 后端下「回退」这个概念不成立：权重本来就在系统内存里。
        return True

    def notes(self) -> list[str]:
        n = ["CPU 后端不存在「层卸载」概念：权重常驻系统内存，"
             "因此没有 PCIe 往返代价，也就没有「禁止回退」这条约束。"]
        n.append("访存带宽由内存控制器与 cache 层级共同决定，"
                 "通常显著低于同价位独显；瓶颈几乎必然在访存侧。")
        n.append("线程数存在最优值：超过物理核数后，访存竞争会让吞吐不升反降。")
        return n


# ============================================================================
# NVIDIA 独显后端
# ============================================================================

class NvidiaGpuBackend(Backend):
    kind = "gpu"
    label = "NVIDIA 独显"
    memory_label = "显存"
    utilization_label = "SM 利用率"

    def mem_total_bytes(self) -> int | None:
        v = super().mem_total_bytes()
        if v:
            return v
        r = self.mem_snapshot()
        return r.total_bytes if r else None

    def mem_snapshot(self) -> MemoryReading | None:
        """
        nvidia-smi 快照：显存 + 温度 + 利用率 + 功耗。

        四个字段一起查的原因：它们回答的是**同一个问题**（这次测量处于什么
        硬件状态）。分四次调用 nvidia-smi 会拿到四个不同时刻的值，
        在负载波动时可能自相矛盾（比如「利用率 90% 但功耗 15 W」）。
        """
        if not shutil.which("nvidia-smi"):
            return None
        try:
            p = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total,temperature.gpu,"
                 "utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if p.returncode != 0 or not p.stdout.strip():
                return None
            cells = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
            used, total, temp, util = cells[:4]
            raw = {"temp_c": int(float(temp)), "gpu_util": int(float(util)) / 100.0}
            if len(cells) > 4:
                try:
                    raw["power_w"] = round(float(cells[4]), 1)
                except ValueError:
                    pass    # 部分虚拟化环境返回 [N/A]
            return MemoryReading(
                used_bytes=int(float(used)) * 1024 * 1024,
                total_bytes=int(float(total)) * 1024 * 1024,
                source="nvidia-smi", raw=raw)
        except Exception:
            return None

    def accel_flags(self) -> list[str]:
        return ["-ngl", "99"]

    def bench_extra_flags(self) -> list[str]:
        """
        GPU 上基准必须显式开 FlashAttention。

        实测（RTX 4090 D，Qwen3.8-27B）：`-fa on` vs `off` 的 Prefill 是
        3064.7 vs 2765.5 t/s（+10.8%），Decode 持平。不开会系统性低估 Prefill，
        而 Prefill 又直接决定 TTFT —— 不写这一项，报告里的首字延迟就是错的。

        新版 llama.cpp 要求带值（`-fa on`），裸 `--flash-attn` 会报错，
        因此这里写成两个 token。
        """
        return ["-fa", "on"]

    def probe_accepts_model_file(self) -> bool:
        # cuda_bw_probe 测的是显存子系统，与主机文件 mmap 无关；
        # 它只接受 --weight-gib（用于打印「该体积对应的上界」表）。
        return False

    probe_binary = "cuda_bw_probe"

    def bandwidth_probe_command(self, probe_bin: str, size_gib: float = 4.0
                                ) -> list[str] | None:
        # GPU 侧用 cuda_bw_probe：必须用**纯读**带宽反推 Decode 上界，
        # 用拷贝带宽会低估约 2 倍。
        return [probe_bin, "--size-gib", f"{size_gib}"]

    def live_metrics_command(self, seconds: int = 20) -> list[str] | None:
        """
        `nvidia-smi dmon` 的时序采样。

        字段（`-s ump`）：u=利用率  m=显存  p=功耗。
        必须与**真实推理负载**同时进行 —— 空闲时的 dmon 全是 0，
        只能证明「GPU 没在干活」，不能回答「瓶颈在算力还是带宽」。
        """
        if not shutil.which("nvidia-smi"):
            return None
        return ["nvidia-smi", "dmon", "-s", "ump", "-c", str(max(seconds, 1)), "-d", "1"]

    def live_metrics_label(self) -> str:
        return "nvidia-smi dmon（利用率 / 显存 / 功耗，1 s 粒度）"

    def parse_live_metrics(self, text: str) -> dict:
        """
        解析 dmon 输出。

        列序：`# gpu  pwr gtemp mtemp sm mem enc dec mclk pclk`（随版本略有差异），
        因此这里按**表头**定位列，而不是写死下标 —— 写死会在换驱动版本后
        静默读错列（把温度当成功率，数字看着合理但完全错）。
        """
        cols: dict[str, int] = {}
        rows: list[list[str]] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                parts = s.lstrip("#").split()
                # 表头里 gpu 在最前，其余按名字记下标
                for i, p in enumerate(parts):
                    cols[p] = i
                continue
            if s.startswith("gpu") or "Id" in s:
                continue
            parts = s.split()
            if parts and parts[0].isdigit():
                rows.append(parts)
        if not rows:
            return {}

        def _col(name: str) -> int | None:
            return cols.get(name)

        def _vals(name: str) -> list[float]:
            i = _col(name)
            if i is None:
                return []
            out = []
            for r in rows:
                if i < len(r):
                    try:
                        out.append(float(r[i]))
                    except ValueError:
                        continue
            return out

        def _stat(v: list[float]) -> dict:
            if not v:
                return {}
            return {"mean": round(sum(v) / len(v), 2),
                    "min": round(min(v), 2), "max": round(max(v), 2),
                    "n": len(v)}

        res: dict = {"samples": len(rows)}
        for key, name in (("sm_util_pct", "sm"), ("mem_util_pct", "mem"),
                          ("power_w", "pwr"), ("temp_c", "gtemp")):
            st = _stat(_vals(name))
            if st:
                res[key] = st
        # dmon 的 sm/mem 是「百分比」，直接给 0~1 便于与其它剖面统一
        for k in ("sm_util_pct", "mem_util_pct"):
            if k in res:
                res[k.replace("_pct", "_ratio")] = {
                    "mean": round(res[k]["mean"] / 100, 4),
                    "max": round(res[k]["max"] / 100, 4)}
        return res

    def probe_metrics(self) -> list[dict]:
        return [
            {"key": "read_bw", "label": "显存纯读带宽", "unit": "GB/s",
             "pattern": r"\[1\] 纯读带宽\s+([\d.]+) GB/s",
             "note": "**推荐口径**：只读不写。Decode 每步只读权重，"
                     "因此这是反推上界的正确分母"},
            {"key": "copy_bw", "label": "显存拷贝带宽（读+写各一遍）", "unit": "GB/s",
             "pattern": r"\[2\] 拷贝带宽\s+([\d.]+) GB/s",
             "note": "**对照项**：读写各占一半带宽。用它反推 Decode 会"
                     "**低估约 2 倍** —— 这是常见的估算错误来源"},
            {"key": "gemv", "label": "q4_0 GEMV 达成带宽", "unit": "GB/s",
             "pattern": r"\[3\] q4_0 GEMV\s+([\d.]+) GB/s",
             "note": "**参考下界**：按量化布局做矩阵-向量乘时能拿到的带宽。"
                     "生产内核（llama.cpp 的 mmvq）通常显著高于此值"},
            {"key": "gemv_gflops", "label": "q4_0 GEMV 算力", "unit": "GFLOPS",
             "pattern": r"\[3\] q4_0 GEMV\s+[\d.]+ GB/s，([\d.]+) GFLOPS",
             "note": "与上一行同一次测量，用于对照 roofline 拐点"},
        ]

    def probe_scope_note(self) -> str:
        return ("本探针测的是**显存子系统**（SM → L2 → 显存控制器 → GDDR/HBM）"
                "的带宽。Decode 只读权重，因此**必须用纯读带宽**反推上界；"
                "拷贝带宽含写入，会低估约 2 倍。")

    def runtime_desc(self) -> str:
        d, c = self.sw.get("driver"), self.sw.get("cuda")
        parts = []
        if d:
            parts.append(f"驱动 {d}")
        if c:
            parts.append(f"CUDA {'/'.join(str(x) for x in c)}"
                         if isinstance(c, (list, tuple)) else f"CUDA {c}")
        return " / ".join(parts) if parts else "—"

    def runtime_label(self) -> str:
        return "驱动 / CUDA"

    def env_commands(self) -> list[tuple[str, list[str]]]:
        return [("NVIDIA GPU", ["nvidia-smi"]), ("GPU 详细", ["nvidia-smi", "-q"])]

    def thermal_snapshot(self) -> ThermalReading | None:
        r = self.mem_snapshot()
        if r and "temp_c" in r.raw:
            return ThermalReading(temp_c=r.raw["temp_c"], source="nvidia-smi")
        return None

    def memory_measured_by(self) -> str:
        return "nvidia-smi"


# ============================================================================
# Apple 统一内存后端
# ============================================================================

class AppleUnifiedBackend(Backend):
    kind = "hybrid"
    label = "Apple Silicon（统一内存）"
    memory_label = "统一内存"
    utilization_label = "GPU 利用率"

    def mem_total_bytes(self) -> int | None:
        v = super().mem_total_bytes()
        if v:
            return v
        try:
            p = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                return int(p.stdout.strip())
        except Exception:
            pass
        return None

    def mem_snapshot(self) -> MemoryReading | None:
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return MemoryReading(used_bytes=rss, total_bytes=self.mem_total_bytes(),
                                 source="resource.getrusage (RSS)")
        except Exception:
            return None

    def accel_flags(self) -> list[str]:
        return ["-ngl", "99"]

    def bandwidth_probe_command(self, probe_bin: str, size_gib: float = 2.0
                                ) -> list[str] | None:
        # 统一内存：CPU 侧的 stream 带宽即 GPU 可用的上界，用 ggml_probe 即可。
        return [probe_bin, "--mode", "stream", "--size-gib", f"{size_gib}"]

    def fallback_is_cheap(self) -> bool:
        return True


# ============================================================================
# 移动 SoC / NPU 后端
# ============================================================================

class SocBackend(Backend):
    """移动 SoC（统一内存 + 电池 + 热约束）。"""
    kind = "soc"
    label = "移动 SoC"
    memory_label = "App 内存配额"
    utilization_label = "SoC 利用率"

    def accel_flags(self) -> list[str]:
        # 端侧 CPU 后端
        return ["-ngl", "0"]

    def bandwidth_probe_command(self, probe_bin: str, size_gib: float = 0.5
                                ) -> list[str] | None:
        return [probe_bin, "--mode", "all", "--size-gib", f"{size_gib}"]

    # 与 CPU 后端共用同一套探针与指标（端侧 CPU 路径即主机内存路径）
    probe_metrics = CpuBackend.probe_metrics
    probe_scope_note = CpuBackend.probe_scope_note

    def runtime_desc(self) -> str:
        return CpuBackend.runtime_desc(self)

    def runtime_label(self) -> str:
        return CpuBackend.runtime_label(self)

    def fallback_is_cheap(self) -> bool:
        return True

    def notes(self) -> list[str]:
        return ["统一内存下 CPU/NPU 分工无显式拷贝，约束是算子覆盖与同步开销。",
                "移动 NPU 多为 INT8/INT16 定点，Q4_K_M 权重需离线重编译。",
                "热降频是端侧独有的瓶颈类型，必须做持续负载稳态测试。"]


class NpuBackend(Backend):
    """独立 NPU 加速器（无统一工具链，多数探针不可用）。"""
    kind = "npu"
    label = "NPU 加速器"
    memory_label = "加速器内存"
    utilization_label = "NPU 利用率"

    def mem_snapshot(self) -> MemoryReading | None:
        return None

    def bandwidth_probe_command(self, probe_bin: str, size_gib: float = 2.0
                                ) -> list[str] | None:
        return None

    def notes(self) -> list[str]:
        return ["NPU 无标准内存/利用率采集接口，需厂商工具链（QNN/RKNN/Core ML）。",
                "先用算子支持矩阵扫描确定图分区边界，再谈性能优化。"]


# ============================================================================
# 注册表
# ============================================================================

BACKENDS: dict[str, type[Backend]] = {
    "cpu": CpuBackend,
    "gpu": NvidiaGpuBackend,
    "hybrid": AppleUnifiedBackend,
    "soc": SocBackend,
    "npu": NpuBackend,
    "dsp": NpuBackend,
}


def get_backend(device: dict) -> Backend:
    """
    按 device.kind 取后端实现。未知 kind 回落到 CPU 后端并保留原 kind 标签，
    这样新增硬件类型时框架不会崩，只会少一些专用能力。
    """
    kind = str(device.get("kind", "cpu")).lower()
    cls = BACKENDS.get(kind)
    if cls is None:
        b = CpuBackend(device)
        b.kind = kind
        b.label = f"{kind}（未注册的后端类型，按 CPU 语义处理）"
        return b
    return cls(device)


def backend_summary() -> list[dict]:
    """列出已注册的后端，供 `perfkit query --backends` 展示。"""
    return [{"kind": k, "class": c.__name__, "label": c.label,
             "memory_label": c.memory_label} for k, c in sorted(BACKENDS.items())]
