"""
perfkit baseline —— 工程闭环第 2 步「性能基线」
==============================================================================
目标：建立优化前的真实性能数据，且**可复现**。

两种模式：
  --mode replay  从实测台账读入已采集的数据（默认，无需 GPU）
  --mode live    真机执行 llama-bench / 压测脚本，采集数据写回台账

无论哪种模式，产出结构完全一致，下游命令无需感知差异：
    output/<run_id>/baseline.json

台账契约（ledger.json）
--------------------------------------------------------------------------
    {
      "meta":           {模型/量化/引擎/设备/驱动/采集日期},
      "theoretical":    {带宽、权重体积、理论上界},
      "memory":         {显存分解与实测峰值},
      "baseline_native":{原生单流基线的 decode_tps 与带宽利用率},
      "scenarios": {                     # ← 按 task.scenarios 的 id 组织
        "<scenario_id>": {prompt_tokens, gen_tokens, parallel,
                          prefill_tps, decode_tps, decode_tps_system,
                          ttft_ms, ...}
      },
      "curves": {                        # 曲线类数据，按需消费
        "long_context": [{ctx, prefill_tps, decode_tps, kv_gib,
                          kv_gib_source, command}, ...],
        ...
      },
      "measurement_protocol": {          # ← 报告 §2.2/§2.3 的「测试方法」表
        "tool": "llama-bench",           # 或 "llama-server + HTTP 客户端"
        "tool_version": "llama.cpp 07d937828",
        "statistic": "每配置重复 3 次取均值",
        "measurement_point": "进程内计时 / 端到端墙钟",
        "backend_label": "CPU 后端", "accel_flags": ["-ngl", "0"],
        "flags":  [{flag, meaning}, ...],   # 参数语义（-p/-n/-d/-t/...）
        "conventions": ["口径约定与陷阱", ...],
      },
      "extra": { ... }                   # 任意附加数据，原样透传给报告
    }

**按 scenario id 组织**是关键：这样换任务时只改 task 配置里的场景矩阵，
台账跟着补对应条目即可，baseline 代码无需改动。
缺条目的场景会被标记为 todo，而不是编造数字。

**为什么每个场景要记 `command`、整批要记 `measurement_protocol`**：
报告里每个数字都必须能回答「用什么工具测的、口径是什么、怎么复现」。
只给数字不给方法，读者无法判断它含不含采样、是均值还是峰值，
也就无法跨配置比较。台账里没记 `command` 时框架会**按场景参数重建**
一条等价命令并显式标注 reconstructed —— 但重建命令不等于原始命令，
正式报告建议如实记录原始命令。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from ..config import ConfigError, PerfConfig, load_json, save_json


def _tps(gen_tokens: int, seconds: float) -> float:
    return round(gen_tokens / seconds, 2) if seconds > 0 else 0.0


class _PeakSampler:
    """
    在基准运行期间持续采样内存/功耗/温度，取峰值。

    为什么需要它：跑完 llama-bench 再读一次 nvidia-smi，读到的必然是
    「进程已退出、显存已释放」的状态。那个 0 MiB 会被当成峰值写进报告，
    而真正的峰值（Prefill 阶段）从来没被观测到。

    CPU 后端没有「峰值」可采（进程 RSS 由 getrusage 自己记录），
    因此后端返回 None 时本采样器直接空转，不产生任何假数据。
    """

    def __init__(self, backend, interval: float = 0.5):
        self.backend = backend
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._used: list[float] = []
        self._power: list[float] = []
        self._temp: list[float] = []
        self._util: list[float] = []

    def start(self) -> None:
        # 该后端不用采样口径（CPU 走 RUSAGE_CHILDREN），或根本没有采集手段
        if not getattr(self.backend, "peak_via_sampling", True):
            return
        if self.backend.mem_snapshot() is None:
            return      # 该后端无可用采集手段，不采样
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r = self.backend.mem_snapshot()
            except Exception:
                r = None
            if r:
                if r.used_bytes is not None:
                    self._used.append(r.used_bytes / 1024 / 1024)
                for key, bucket in (("power_w", self._power),
                                    ("temp_c", self._temp),
                                    ("gpu_util", self._util)):
                    v = (r.raw or {}).get(key)
                    if isinstance(v, (int, float)):
                        bucket.append(float(v))
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        if self._thread is None:
            return {}
        self._stop.set()
        self._thread.join(timeout=10)

        def _stat(v: list[float], nd: int = 1) -> dict:
            if not v:
                return {}
            return {"min": round(min(v), nd), "max": round(max(v), nd),
                    "mean": round(sum(v) / len(v), nd), "n": len(v)}

        out: dict = {"interval_s": self.interval,
                     "samples": max(len(self._used), len(self._power))}
        if self._used:
            out["peak_used_mib"] = round(max(self._used), 0)
            out["used_mib"] = _stat(self._used, 0)
        if self._power:
            out["power_w"] = _stat(self._power)
        if self._temp:
            out["temp_c"] = _stat(self._temp)
        if self._util:
            out["gpu_util"] = _stat(self._util, 3)
        return out


def _bench_protocol(cfg: PerfConfig, bench_bin: str | None,
                    repetitions: int = 3) -> dict:
    """
    描述「这批数据是怎么测出来的」。

    报告里的每个数字都必须能回答「用什么工具、什么口径、怎么复现」——
    否则读者看到 17.85 t/s 却不知道它含不含采样、是均值还是峰值、
    上下文深度是多少，数字就不可解释、不可比较。
    因此协议与数据一起落盘，而不是散落在文档里。
    """
    backend = cfg.backend
    return {
        "tool": "llama-bench",
        "tool_bin": bench_bin,
        "statistic": f"每个配置重复 {repetitions} 次，取 llama-bench 的 avg_ts（算术均值）；"
                     f"离散度见 samples_ts / stddev_ts",
        "measurement_point": "进程内计时：不含 HTTP 传输、排队与客户端开销",
        "backend": backend.kind,
        "backend_label": backend.label,
        "accel_flags": backend.accel_flags(),
        # 后端专属参数（GPU 的 `-fa on` 等）也要落盘：
        # 否则读者按「accel_flags + 场景参数」重建命令，会漏掉这些
        # 影响 Prefill 的开关，复现出的数字必然对不上。
        "backend_flags": backend.bench_flags(),
        "threads": backend.default_threads(),
        "flags": [
            {"flag": "-p N", "meaning": "Prefill 口径：一次性处理 N 个 token 的"
                                        "**纯 prompt 吞吐**，不含采样与首 token 排队"},
            {"flag": "-n N", "meaning": "Decode 口径：在已有上下文中自回归生成 N 个"
                                        " token 的**稳态速率**，排除首 token"},
            {"flag": "-d N", "meaning": "上下文深度：先用合成 token 把 KV Cache 填到 N，"
                                        "再测 Prefill/Decode —— 用于模拟长对话"},
            {"flag": "-t N", "meaning": "线程数（CPU 后端的关键调优参数）"},
            {"flag": "-ngl N", "meaning": f"卸载层数；本后端固定为 "
                                          f"{' '.join(backend.accel_flags()) or '默认'}"},
            {"flag": "-r N", "meaning": "重复次数（本批为 "
                                        f"{repetitions}）"},
            {"flag": "-o json", "meaning": "结构化输出，供框架自动解析；原始 stdout 落盘 "
                                           "output/<run>/raw/"},
        ],
        "conventions": [
            "llama-bench 的 Decode（tg）是**纯解码**吞吐：不含采样、不含首 token "
            "排队，因此**高于**端到端墙钟速率。",
            "端到端口径（HTTP 客户端计时）会额外包含采样、排队与传输开销，"
            "两者相差可达 35%，**引用时必须标明来源**。",
        ],
    }


def _scenario_command(cfg: PerfConfig, sc: dict, recorded: str | None,
                      kv_k: str | None = None, kv_v: str | None = None,
                      extra: list[str] | None = None) -> tuple[str, str]:
    """
    给出一个场景的复现命令，返回 (命令, 来源标记)。

    台账里若已记录原始命令，直接用（来源 recorded）；
    否则按场景参数 + 后端重建一条等价命令（来源 reconstructed）——
    重建命令让读者能自己跑一遍，但必须标注它不是原始那条，
    否则就成了「伪造复现记录」。

    重建时用**裸命令名**而不是探测到的绝对路径：探测到的是「本机现在
    装的那个」，未必是当初采集用的那个，写出绝对路径会误导。
    """
    if recorded:
        return recorded, "recorded"
    # 场景自带的 KV 量化档位也要还原，否则「同深度对照」的两条命令
    # 会看起来一模一样，读者无法分辨唯一变量在哪。
    k = kv_k if kv_k is not None else sc.get("cache_type_k")
    v = kv_v if kv_v is not None else sc.get("cache_type_v")
    cmd = cfg.backend.bench_command("llama-bench",
                                    cfg.model.get("file", "<model.gguf>"),
                                    sc, kv_k=k, kv_v=v, extra=extra)
    return " ".join(cmd), "reconstructed"


def run_replay(cfg: PerfConfig, ledger_path: Path, scenarios: list[dict]) -> dict:
    """把台账里的实测数据映射到任务定义的场景矩阵上。"""
    led = load_json(ledger_path)
    led_scen = led.get("scenarios", {})

    by_scenario: dict[str, dict] = {}
    missing: list[str] = []
    for sc in scenarios:
        sid = sc["id"]
        entry = led_scen.get(sid)
        if not entry:
            missing.append(sid)
            by_scenario[sid] = {
                "status": "todo",
                "desc": sc.get("desc", ""),
                "note": "台账中无该场景的实测数据；请用 --mode live 采集或补充台账",
            }
            continue
        merged = dict(entry)
        # 场景矩阵里的定义优先（保证口径与任务一致），台账只提供测量值
        for k in ("prompt_tokens", "gen_tokens", "parallel", "depth"):
            if k in sc:
                merged.setdefault(k, sc[k])
        merged.setdefault("desc", sc.get("desc", ""))
        # 推导缺失的派生指标，避免下游各处重复算
        if merged.get("decode_tps") and not merged.get("decode_seconds") \
                and merged.get("gen_tokens"):
            merged["decode_seconds"] = round(
                merged["gen_tokens"] / merged["decode_tps"], 2)
        if merged.get("prefill_tps") and not merged.get("prefill_seconds") \
                and merged.get("prompt_tokens"):
            merged["prefill_seconds"] = round(
                merged["prompt_tokens"] / merged["prefill_tps"], 2)
        if merged.get("prefill_seconds") and not merged.get("ttft_ms"):
            merged["ttft_ms"] = round(merged["prefill_seconds"] * 1000, 1)
        # 复现命令：优先用台账记录的原始命令，缺失则按场景参数重建
        _cmd, _src = _scenario_command(cfg, merged, merged.get("command"))
        merged["command"] = _cmd
        merged["command_source"] = _src
        by_scenario[sid] = merged

    extra = dict(led.get("extra", {}))
    # 常用字段提升到 extra 顶层，报告与图表直接消费
    extra.setdefault("baseline_native_decode_tps",
                     led.get("baseline_native", {}).get("decode_tps"))
    extra.setdefault("bandwidth_utilization",
                     led.get("baseline_native", {}).get("bandwidth_utilization"))
    extra.setdefault("long_context_curve",
                     (led.get("curves") or {}).get("long_context", []))
    extra.setdefault("thermal", led.get("thermal", {}))
    # 并发场景的摘要：优先取 extra 里的显式声明，否则从多槽位场景推导
    if "concurrent_2" not in extra:
        for _sid, _s in by_scenario.items():
            if _s.get("parallel", 1) > 1 and _s.get("decode_tps_system"):
                extra["concurrent_2"] = {
                    "config": f'{_sid}（parallel={_s["parallel"]}）',
                    "system_tps": _s["decode_tps_system"],
                    "per_slot_tps": _s.get("per_slot_tps", []),
                    "source": _s.get("source", ""),
                }
                break

    # 长上下文曲线：逐点补复现命令（同样标注来源）
    _curve = []
    for row in (extra.get("long_context_curve") or []):
        row = dict(row)
        sc = {"prompt_tokens": row.get("prompt_tokens", 512),
              "gen_tokens": row.get("gen_tokens", 64),
              "depth": row["ctx"], "repetitions": row.get("repetitions", 3)}
        _cmd, _src = _scenario_command(cfg, sc, row.get("command"))
        row["command"] = _cmd
        row["command_source"] = _src
        _curve.append(row)
    if _curve:
        extra["long_context_curve"] = _curve

    # 测量协议：优先取台账顶层，其次取 meta 里的
    protocol = (led.get("measurement_protocol")
                or (led.get("meta") or {}).get("measurement_protocol") or {})

    return {
        "mode": "replay",
        # 后端标记必须落盘：query --list 与报告都要用它区分「这套数据来自哪类硬件」。
        # 优先取台账 meta 里记录的（那是实测时真实使用的后端），
        # 缺失时才回落到当前配置声明的后端。
        "backend": (led.get("meta") or {}).get("backend") or cfg.backend.kind,
        "backend_label": (led.get("meta") or {}).get("backend_label")
                         or cfg.backend.label,
        "ledger": str(ledger_path),
        "protocol": protocol,
        # 环境验证记录（业务地图 0 环境准备的输出物③④）：
        # 模型加载 + Demo 运行的可复现记录。原样透传给报告。
        "environment_verification": led.get("environment_verification") or {},
        "meta": led.get("meta", {}),
        "theoretical": led.get("theoretical", {}),
        "memory": led.get("memory", {}),
        "scenarios": by_scenario,
        "scenarios_missing": missing,
        "extra": extra,
    }


def _find_bench_bin(explicit: str | None = None) -> str | None:
    """
    定位 llama-bench。解析顺序：
      1) --bench-bin 显式指定
      2) PATH 里的 llama-bench
      3) 环境变量 LLAMA_BENCH
      4) 常见安装位置（含本机 llama.cpp 源码树）
    """
    cands: list[str] = []
    if explicit:
        cands.append(explicit)
    w = shutil.which("llama-bench")
    if w:
        cands.append(w)
    if os.environ.get("LLAMA_BENCH"):
        cands.append(os.environ["LLAMA_BENCH"])
    cands += [
        "/root/autodl-tmp/llama.cpp/build/bin/llama-bench",
        str(Path.home() / "llama.cpp/build/bin/llama-bench"),
        "/usr/local/bin/llama-bench",
    ]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def _parse_bench_json(stdout: str) -> list[dict]:
    """
    解析 llama-bench 的 `-o json` 输出。

    llama-bench 会在 stdout 里混入后端初始化日志（如 Metal/OpenCL），
    因此不能直接 json.loads 整个输出 —— 需要先定位 JSON 数组的起止。
    """
    start = stdout.find("[")
    if start < 0:
        return []
    depth, end = 0, -1
    for i in range(start, len(stdout)):
        if stdout[i] == "[":
            depth += 1
        elif stdout[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return []
    try:
        data = json.loads(stdout[start:end])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def run_live(cfg: PerfConfig, outdir: Path, scenarios: list[dict],
             bench_bin: str | None = None,
             kv_k: str | None = None, kv_v: str | None = None,
             extra_flags: list[str] | None = None) -> dict:
    """
    真机采集。

    硬件差异全部由 `cfg.backend` 承担：命令怎么拼、内存怎么读、
    加速器标志怎么传，本函数不出现任何 `-ngl` / `nvidia-smi` 字样。

    原始输出（命令、stdout、stderr、墙钟时间）都落盘到
    output/<run>/raw/，保证每个数字都能事后复核。
    """
    backend = cfg.backend
    bench = _find_bench_bin(bench_bin)
    if not bench:
        raise RuntimeError(
            "找不到 llama-bench。三种解法：\n"
            "  1) 把它加进 PATH；\n"
            "  2) 用 --bench-bin /path/to/llama-bench 指定；\n"
            "  3) 设环境变量 LLAMA_BENCH=/path/to/llama-bench；\n"
            "  4) 改用 --mode replay 从台账读数据。")

    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    # 峰值采样：**必须在基准运行的同一时刻采**。
    # 之前是跑完再读一次 nvidia-smi，此时 llama-bench 进程已退出、
    # 显存已释放，读到的 0 MiB 会被当成「峰值」写进报告。
    # 因此在每个场景运行期间开一个采样线程。
    peaks = _PeakSampler(backend, interval=0.5)
    peaks.start()
    try:
        for sc in scenarios:
            if sc.get("duration_s"):
                # 持续负载场景（soak）：llama-bench 不支持按时长跑，
                # 用等价的 token 预算近似，并在结果里标注换算方式。
                sc = {**sc, "gen_tokens": max(int(sc.get("gen_tokens", 128)),
                                              int(sc["duration_s"] * 2)),
                      "_duration_approx": True}
            cmd = backend.bench_command(
                bench, cfg.model["file"], sc,
                kv_k=kv_k, kv_v=kv_v, extra=extra_flags)
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            wall = round(time.time() - t0, 2)
            (raw_dir / f"llama-bench_{sc['id']}.log").write_text(
                f"$ {' '.join(cmd)}\n\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
                encoding="utf-8")

            entry: dict = {
                "command": " ".join(cmd),
                "command_source": "recorded",
                "wall_seconds": wall,
                "returncode": proc.returncode,
                "prompt_tokens": sc.get("prompt_tokens"),
                "gen_tokens": sc.get("gen_tokens"),
                "parallel": sc.get("parallel", 1),
                "backend": backend.kind,
            }
            if sc.get("depth"):
                entry["depth"] = sc["depth"]
            if sc.get("_duration_approx"):
                entry["duration_approx_note"] = (
                    f'原场景按时长 {sc["duration_s"]}s 定义，'
                    f"llama-bench 不支持按时长跑，改用 {entry['gen_tokens']} token 预算近似")

            rows = _parse_bench_json(proc.stdout)
            if not rows:
                # 退化路径：解析人类可读表格（老版本 llama-bench 无 -o json）
                for line in proc.stdout.splitlines():
                    if "|" not in line:
                        continue
                    cells = [c.strip() for c in line.split("|")]
                    if len(cells) < 2:
                        continue
                    tag, val = cells[0], cells[1].split("±")[0].strip()
                    try:
                        v = float(val)
                    except ValueError:
                        continue
                    if tag.startswith("pp"):
                        entry["prefill_tps"] = v
                    elif tag.startswith("tg"):
                        entry["decode_tps"] = v
            else:
                for row in rows:
                    # llama-bench 用 n_prompt / n_gen 区分两类测试
                    if row.get("n_prompt"):
                        entry["prefill_tps"] = round(float(row["avg_ts"]), 2)
                        entry["prefill_stddev"] = round(float(row.get("stddev_ts", 0)), 2)
                        entry["prefill_samples"] = row.get("samples_ts")
                    if row.get("n_gen"):
                        entry["decode_tps"] = round(float(row["avg_ts"]), 2)
                        entry["decode_stddev"] = round(float(row.get("stddev_ts", 0)), 2)
                        entry["decode_samples"] = row.get("samples_ts")
                    for k in ("model_size", "model_n_params", "build_commit",
                              "n_threads", "type_k", "type_v", "n_depth",
                              "n_gpu_layers", "flash_attn", "cpu_info", "gpu_info"):
                        if row.get(k) is not None:
                            entry.setdefault(k, row[k])
                if rows:
                    entry["test_time"] = rows[0].get("test_time")

            if entry.get("prefill_tps"):
                entry["prefill_seconds"] = round(
                    sc["prompt_tokens"] / entry["prefill_tps"], 2)
                entry["ttft_ms"] = round(entry["prefill_seconds"] * 1000, 1)
            if entry.get("decode_tps"):
                entry["decode_seconds"] = round(
                    sc["gen_tokens"] / entry["decode_tps"], 2)
            results[sc["id"]] = entry

        # 深度扫描（可选）：受控实验，只改 -d，产出 KV 随 ctx 的衰减曲线。
        # 放在场景循环之后、采样线程停止之前 —— 曲线点也要计入显存峰值。
        _sweep_cfg = (cfg.task.get("curves") or {}).get("long_context") or {}
        if _sweep_cfg.get("depths"):
            print("  · 深度扫描（只改 -d，其余与场景一致）")
            depth_rows = _run_depth_sweep(cfg, backend, bench, outdir,
                                          _sweep_cfg, kv_k, kv_v, extra_flags)
            if depth_rows:
                results["_curves"] = {"long_context": depth_rows}
    finally:
        peak_stats = peaks.stop()

    # 内存快照：峰值来自运行期间的采样；静态快照只用于记录「总量」
    mem_reading = backend.mem_snapshot()
    mem = mem_reading.as_dict() if mem_reading else {}
    # CPU 后端的峰值走 RUSAGE_CHILDREN（子进程退出后仍保留最大值），
    # 采样线程在 CPU 上只会看到 perfkit 自己，拿不到被测进程。
    _child_peak = backend.peak_after_children()
    if _child_peak:
        mem["peak_used_mib"] = round(_child_peak / 1024 / 1024, 0)
        mem["measure_method"] = (
            "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss —— "
            "子进程退出后仍保留整个生命周期的峰值 RSS")
        if mem.get("total_gib"):
            mem["safety_margin_gib"] = round(
                mem["total_gib"] - mem["peak_used_mib"] / 1024, 2)
        # CPU 上「用量」这一栏若来自 RUSAGE_SELF 会显示 perfkit 自己的十几 MB，
        # 必须用子进程峰值覆盖，否则报告里会出现「1.5B 模型只占 17 MB」。
        mem["used_gib"] = round(mem["peak_used_mib"] / 1024, 3)
        mem["used_bytes"] = int(_child_peak)
        mem["source"] = "resource.getrusage(RUSAGE_CHILDREN)"
    if peak_stats and not _child_peak:
        mem["peak_used_mib"] = peak_stats.get("peak_used_mib")
        mem["peak_sampling"] = peak_stats
        mem["measure_method"] = (
            f"{backend.memory_measured_by()}，每 {peak_stats.get('interval_s')} s 采样，"
            f"取**运行期间**的最大值（{peak_stats.get('samples')} 个样本）")
        if peak_stats.get("peak_used_mib") and mem.get("total_gib"):
            mem["safety_margin_gib"] = round(
                mem["total_gib"] - peak_stats["peak_used_mib"] / 1024, 2)
        # 把时序统计提升到顶层，与 replay 台账的字段名对齐 ——
        # 报告 / verify 只认这几个键，藏在 peak_sampling 里就取不到。
        #
        # 口径必须与 KPI 定义一致，不能图省事全用均值：
        #   · power_w  → **峰值**：KPI 定义就是「采样峰值」，用均值会被
        #                rep 之间的空闲拉低（实测均值 245 W vs 峰值 353 W）
        #   · temp_c   → **峰值**：热风险看的是最高点，均值会掩盖瞬时过热
        #   · gpu_util → **均值**：它是「有 kernel 在跑的时间占比」，
        #                峰值恒为 1.0，没有信息量
        _ps = peak_stats
        if (_ps.get("power_w") or {}).get("max") is not None:
            mem["power_w"] = _ps["power_w"]["max"]
            mem["power_w_mean"] = _ps["power_w"]["mean"]
        if (_ps.get("temp_c") or {}).get("max") is not None:
            mem["temp_c"] = _ps["temp_c"]["max"]
            mem["temp_c_mean"] = _ps["temp_c"]["mean"]
        if (_ps.get("gpu_util") or {}).get("mean") is not None:
            mem["gpu_util"] = _ps["gpu_util"]["mean"]
            mem["gpu_util_peak"] = _ps["gpu_util"]["max"]
        if getattr(backend, "kind", "") == "gpu":
            mem["sampling_semantics"] = {
                "power_w": "峰值（采样窗口内最大 power.draw）",
                "temp_c": "峰值（采样窗口内最高温度）",
                "gpu_util": "均值（utilization.gpu 的时间占比，峰值恒为 1 无信息量）",
            }
    if mem_reading:
        (raw_dir / "mem_snapshot.json").write_text(
            json.dumps(mem_reading.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    thermal = backend.thermal_snapshot()
    if thermal:
        mem.setdefault("temp_c", thermal.temp_c)
        mem["thermal_source"] = thermal.source
        mem.update({k: v for k, v in thermal.raw.items() if k not in mem})

    protocol = _bench_protocol(
        cfg, bench,
        repetitions=scenarios[0].get("repetitions", 3) if scenarios else 3)
    # 工具版本从 llama-bench 自己报告的 build_commit 取（比配置里写的更可信）
    _first = next(iter(results.values()), {})
    if _first.get("build_commit"):
        protocol["tool_version"] = (f'llama.cpp {_first["build_commit"]}'
                                    + (f'（{_first["cpu_info"]}）'
                                       if _first.get("cpu_info") else ''))
    if results and all(r.get("command") for r in results.values()):
        protocol["commands_recorded"] = True

    curves = results.pop("_curves", {})
    return {
        "mode": "live",
        "backend": backend.kind,
        "backend_label": backend.label,
        "protocol": protocol,
        "curves": curves,
        "meta": {**cfg.model, "device": cfg.device.get("name"),
                 "engine_commit": cfg.device.get("software", {}).get("engine_commit"),
                 "bench_bin": bench,
                 "accel_flags": backend.accel_flags()},
        "scenarios": results,
        "memory": mem,
    }


def add_parser(sub) -> None:
    p = sub.add_parser("baseline",
                       help="【第2步】建立性能基线（replay 读台账 / live 真机采集）")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--ledger", default=None,
                   help="台账路径；默认取 <config_dir>/ledger.json，"
                        "否则取 data/reference_measurements.json")
    p.add_argument("--bench-bin", default=None,
                   help="live 模式下 llama-bench 路径"
                        "（也可用环境变量 LLAMA_BENCH）")
    p.add_argument("--cache-type-k", default=None,
                   help="live 模式下覆盖 K 量化类型（如 q8_0）")
    p.add_argument("--cache-type-v", default=None,
                   help="live 模式下覆盖 V 量化类型（如 q4_0）")
    p.add_argument("--extra", nargs="*", default=None,
                   help="live 模式下追加给 llama-bench 的额外参数")
    p.add_argument("--only", default=None, metavar="ID[,ID...]",
                   help="只跑指定场景（逗号分隔）。用于先做冒烟验证，"
                        "避免一次性把 10 分钟持续负载等长场景全跑完")
    p.add_argument("--write-ledger", default=None, metavar="PATH",
                   help="live 模式采集后把结果导出为台账 JSON（供 replay 复用）")
    p.set_defaults(func=main)
    return p


def _derive_extra(cfg: PerfConfig, data: dict) -> dict:
    """
    从场景矩阵推导 `extra` 段 —— **两种模式都必须有这一段**。

    为什么 replay 有、live 没有会出问题：
      `extra` 原本只是台账里人工整理的派生字段（原生基线吞吐、长上下文曲线…）。
      下游（profiling / optimize / report / visualize）统一按
      `baseline["extra"][...]` 取值，于是 live 模式一跑到第 3 步就 KeyError ——
      而 live 恰恰是「换硬件后第一次跑」的必经路径，等于最需要它的时候崩了。

    修法不是在每个下游加 `.get`，而是**让两种模式产出同构的数据**：
      · baseline_native_decode_tps ← 最短 prompt 的单流场景（最接近纯 Decode 稳态）
      · bandwidth_utilization      ← 由该场景的 tps 反推（仅权重口径）
      · long_context_curve         ← 场景里带 depth 的点，按 depth 排序
    """
    extra = dict(data.get("extra") or {})
    scs = data.get("scenarios") or {}
    from ..config import weight_bytes as _wbf
    _wb = _wbf(cfg.model)
    _bw = cfg.device["memory"]["mem_bandwidth_gbps_peak"] * 1e9

    if extra.get("baseline_native_decode_tps") is None:
        cands = [(sid, s) for sid, s in scs.items()
                 if s.get("decode_tps") and s.get("parallel", 1) == 1
                 and s.get("status") != "todo"]
        if cands:
            cands.sort(key=lambda kv: kv[1].get("prompt_tokens", 0))
            sid, s = cands[0]
            extra["baseline_native_decode_tps"] = s["decode_tps"]
            extra["baseline_native_source"] = (
                f'{data.get("mode")} 推导：场景 {sid}'
                f'（prompt {s.get("prompt_tokens")} token，单流）')
            # 带宽利用率：只按权重口径反推（与台账 §1.1 同口径），
            # KV 读的那部分单列在 profiling 的 KV 剖面里。
            extra["bandwidth_utilization"] = round(_wb * s["decode_tps"] / _bw, 4)
            extra["bandwidth_utilization_note"] = (
                f"仅权重口径：{_wb/1e9:.2f} GB × {s['decode_tps']} t/s ÷ "
                f"{_bw/1e9:.0f} GB/s")

    if not extra.get("long_context_curve"):
        # live 模式若跑了深度扫描，曲线就在 data["curves"] 里 —— 优先用它，
        # 因为它才是「只改 -d」的受控实验；场景矩阵里的 depth 点没有。
        _cv = (data.get("curves") or {}).get("long_context") or []
        if _cv:
            extra["long_context_curve"] = _cv
            extra["long_context_source"] = "live 深度扫描（只改 -d，其余固定）"
            return extra
        rows = []
        for sid, s in scs.items():
            if s.get("status") == "todo" or not s.get("depth"):
                continue
            rows.append({
                "ctx": s["depth"],
                "prefill_tps": s.get("prefill_tps"),
                "decode_tps": s.get("decode_tps"),
                # live 模式没有独立的 KV 实测，用几何公式补，
                # 并如实标注来源 —— 不能让读者以为这是实测值。
                "kv_gib": None,
                "kv_gib_source": "derived",
                "prompt_tokens": s.get("prompt_tokens"),
                "gen_tokens": s.get("gen_tokens"),
                "command": s.get("command"),
                "command_source": s.get("command_source"),
            })
        if rows:
            rows.sort(key=lambda r: r["ctx"])
            extra["long_context_curve"] = rows
            extra["long_context_source"] = "live 推导：场景矩阵里带 depth 的点"
    return extra


def _run_depth_sweep(cfg: PerfConfig, backend, bench: str, outdir: Path,
                     sweep: dict, kv_k: str | None, kv_v: str | None,
                     extra_flags: list[str] | None) -> list[dict]:
    """
    深度扫描：同一配置只改 `-d`（上下文深度），产出 KV 随 ctx 的衰减曲线。

    为什么单独跑而不是混进场景矩阵：
      场景矩阵回答「真实业务下有多快」，每个场景有自己的 prompt/gen 组合；
      深度扫描是**受控实验** —— 除深度外一切固定，专门回答
      「KV 增长让速率掉多少」。两者的 prompt/gen 口径不同，混在一起会让
      「场景表」和「曲线」互相污染。

    没有这条曲线时，报告 §3.6（KV 剖面）与 §3.10 的根因树都只能留空 ——
    而 KV 恰恰是长上下文任务的第一性瓶颈。因此 live 模式必须有它。
    """
    depths = sweep.get("depths") or []
    if not depths:
        return []
    rows: list[dict] = []
    for d in depths:
        sc = {
            "id": f"depth_{d}",
            "prompt_tokens": sweep.get("prompt_tokens", 512),
            "gen_tokens": sweep.get("gen_tokens", 64),
            "depth": d,
            "repetitions": sweep.get("repetitions", 3),
        }
        cmd = backend.bench_command(bench, cfg.model["file"], sc,
                                    kv_k=kv_k, kv_v=kv_v, extra=extra_flags)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        (outdir / "raw" / f"depth_{d}.log").write_text(
            f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n", encoding="utf-8")
        entry: dict = {"ctx": d, "command": " ".join(cmd),
                       "command_source": "recorded",
                       "wall_seconds": round(time.time() - t0, 2),
                       "prompt_tokens": sc["prompt_tokens"],
                       "gen_tokens": sc["gen_tokens"]}
        rows_ = _parse_bench_json(proc.stdout)
        for r in rows_:
            if r.get("n_prompt"):
                entry["prefill_tps"] = round(float(r["avg_ts"]), 2)
            if r.get("n_gen"):
                entry["decode_tps"] = round(float(r["avg_ts"]), 2)
        if entry.get("prefill_tps") is None and entry.get("decode_tps") is None:
            entry["status"] = "failed"
            entry["note"] = (proc.stderr or proc.stdout)[-300:]
        # KV 容量：live 模式没有逐点内存快照，用几何公式补并标注来源
        from ..config import kv_cache_bytes
        entry["kv_gib"] = round(
            kv_cache_bytes(cfg.model, d, *cfg.kv_bits()) / 1024 ** 3, 3)
        entry["kv_gib_source"] = "derived"
        rows.append(entry)
        print(f"    depth {d:>6}  prefill {entry.get('prefill_tps', '—')}"
              f"  decode {entry.get('decode_tps', '—')}")
    return rows


def _filter_scenarios(scenarios: list[dict], only: str | None) -> list[dict]:
    """按 --only 过滤场景。未匹配到任何场景时报错而非静默跑全量。"""
    if not only:
        return scenarios
    want = [x.strip() for x in only.split(",") if x.strip()]
    hit = [s for s in scenarios if s["id"] in want]
    miss = [w for w in want if w not in {s["id"] for s in scenarios}]
    if miss:
        avail = ", ".join(s["id"] for s in scenarios)
        raise ConfigError(f"--only 里的场景不存在：{miss}\n可用场景：{avail}")
    return hit


def main(args) -> int:
    cfg = PerfConfig.from_dir(args.config, args.run)
    scenarios = _filter_scenarios(cfg.task["scenarios"],
                                 getattr(args, "only", None))

    if args.mode == "live":
        data = run_live(cfg, cfg.outdir, scenarios,
                        bench_bin=getattr(args, "bench_bin", None),
                        kv_k=getattr(args, "cache_type_k", None),
                        kv_v=getattr(args, "cache_type_v", None),
                        extra_flags=getattr(args, "extra", None))
    else:
        # 显式传入的 --ledger 按 cwd 解析（CLI 常规语义）；
        # 未传入时用 cfg.ledger() 的自动解析顺序。
        ledger = Path(args.ledger) if getattr(args, "ledger", None) else cfg.ledger()
        data = run_replay(cfg, ledger, scenarios)

    # 补齐每个场景的权重与描述，方便 verify 阶段加权打分
    for sc in scenarios:
        if sc["id"] in data["scenarios"]:
            data["scenarios"][sc["id"]].setdefault("weight", sc.get("weight", 0.0))
            data["scenarios"][sc["id"]].setdefault("desc", sc.get("desc", ""))

    # 两种模式都补齐 extra / theoretical：下游统一按这些键取值，
    # 只让 replay 有这两段会让 live 模式在第 3 步、第 7 步各崩一次。
    data["extra"] = _derive_extra(cfg, data)
    if not data.get("theoretical"):
        from ..config import weight_bytes as _wbf2
        _w2 = _wbf2(cfg.model)
        data["theoretical"] = {
            "vram_bandwidth_gbps": cfg.device["memory"]["mem_bandwidth_gbps_peak"],
            "weight_gib": round(_w2 / 1024 ** 3, 3),
            "file_gib": round(cfg.model["artifacts"][0]["bytes"] / 1024 ** 3, 3),
            "decode_tps_upper_bound": round(cfg.decode_upper_bound(0), 2),
            "confidence": "derived",
            "_source": "live 模式：由设备标称带宽 ÷ 有效权重推出",
        }
    if not data.get("meta"):
        data["meta"] = {}
    data["_derived"] = {
        "decode_tps_upper_bound": round(cfg.decode_upper_bound(0), 2),
        "decode_tps_upper_bounds": cfg.decode_upper_bounds(),
        "mem_budget": cfg.mem_budget(),
        "roofline": _roofline(cfg),
    }
    # 环境上下文：CPU 后端的测量结果对机器负载高度敏感，
    # 记录当时的负载，否则事后无法判断一个数字是否可信。
    # 注意：replay 模式下的数据来自台账（可能是几天前采集的），
    # 当前负载与数据本身无关，因此只在 live 模式下把负载当作有效性判据。
    data["environment"] = _env_context(cfg, live=(args.mode == "live"))
    path = save_json(data, cfg.out("baseline.json"))

    # 可选：把 live 采集结果导出为台账，供后续 replay 复用。
    # 这是「测一次、反复用」的关键 —— 台账是人工确认过的参考值，
    # 因此导出是显式动作（--write-ledger），不自动覆盖。
    ledger_out = getattr(args, "write_ledger", None)
    if ledger_out:
        _export_ledger(cfg, data, Path(ledger_out))

    backend = cfg.backend
    print(f"[baseline] mode={data['mode']}  后端={backend.label}（{backend.kind}）")
    if data.get("ledger"):
        print(f"           台账 {data['ledger']}")
    print(f"           产出 → {path}")
    for sid, s in data["scenarios"].items():
        if s.get("status") == "todo":
            print(f"  - {sid:<14} [待采集] {s.get('note', '')}")
        elif s.get("decode_tps"):
            print(f"  - {sid:<14} decode {s['decode_tps']:>6.2f} t/s"
                  f"  prefill {(s.get('prefill_tps') or 0):>7.2f} t/s"
                  f"  ttft {(s.get('ttft_ms') or 0):>8.1f} ms")
    ub = data["_derived"]["decode_tps_upper_bounds"]
    print(f"  - 理论上界: 不计 KV {ub['no_kv_tps']} t/s"
          f" | 含 {ub['ctx']//1024}K KV 读 {ub['at_ctx_tps']} t/s")
    # live 模式没有 extra 段（那是 replay 从台账提升上来的），故用 .get 取
    bw = (data.get("extra") or {}).get("bandwidth_utilization")
    if bw:
        print(f"  - 带宽利用率: {bw}")
    elif data.get("memory"):
        print(f"  - 内存快照: {data['memory'].get('used_gib', '—')} GiB used"
              f" / {data['memory'].get('total_gib', '—')} GiB total")
    if data.get("scenarios_missing"):
        print(f"  - ⚠ 台账缺失场景: {', '.join(data['scenarios_missing'])}")
    env = data.get("environment") or {}
    if env.get("load_avg"):
        print(f"  - 采集时系统负载: {env['load_avg'][0]}"
              f"（{env['cpu_cores']} 核，每核 {env['load_per_core']}）")
    if env.get("warning"):
        print(f"  - ⚠ {env['warning']}")
    for w in (cfg.warnings or []):
        print(f"  - ⚠ 配置缺口: {w}")
    return 0


def _export_ledger(cfg: PerfConfig, data: dict, out: Path) -> Path:
    """
    把 baseline 的实测结果导出成台账结构。

    只写入真实测量到的字段 —— 没测到的保持缺失，由下游标注为 todo，
    而不是填 0 或沿用旧值。台账是「参考真值」，掺水就失去意义。
    """
    meta_in = data.get("meta") or {}
    led = {
        "$comment": "由 perfkit baseline --mode live --write-ledger 导出；"
                    "请人工核对后再作为参考台账使用。",
        "meta": {
            **{k: v for k, v in meta_in.items()
               if k in ("model", "quant", "engine", "device", "engine_commit",
                        "build_commit", "n_threads", "type_k", "type_v",
                        "cpu_info", "gpu_info", "accel_flags")},
            "backend": data.get("backend"),
            "collected_by": "perfkit baseline --mode live",
        },
        # 测量协议与数据一起落盘：报告 §2.2 直接渲染它，
        # 保证「数字」与「数字怎么来的」永远成对出现。
        "measurement_protocol": data.get("protocol") or {},
        "theoretical": {
            "vram_bandwidth_gbps": cfg.device["memory"]["mem_bandwidth_gbps_peak"],
            "weight_gib": round(cfg.model["artifacts"][0]["bytes"] / 1024 ** 3, 3),
            "decode_tps_upper_bound": data["_derived"]["decode_tps_upper_bound"],
        },
        "scenarios": {},
        "memory": data.get("memory", {}),
        "extra": {},
    }
    keep = ("prompt_tokens", "gen_tokens", "parallel", "depth",
            "prefill_tps", "decode_tps", "decode_tps_system", "ttft_ms",
            "prefill_seconds", "decode_seconds", "decode_stddev", "prefill_stddev",
            "command", "command_source", "test_time")
    for sid, s in data.get("scenarios", {}).items():
        if s.get("status") == "todo":
            led["scenarios"][sid] = {"status": "todo", "note": s.get("note", "")}
            continue
        entry = {k: s[k] for k in keep if s.get(k) is not None}
        entry["source"] = s.get("source") or "perfkit baseline --mode live"
        entry["confidence"] = "measured"
        led["scenarios"][sid] = entry
    # 原生基线吞吐：取单流场景的实测 decode（不含任何优化）
    for s in led["scenarios"].values():
        if s.get("parallel", 1) == 1 and s.get("decode_tps"):
            led["baseline_native"] = {
                "config": "纯自回归，无投机解码，无 KV 量化",
                "decode_tps": s["decode_tps"],
                "source": s.get("source"),
                "confidence": "measured",
            }
            break
    path = save_json(led, out)
    print(f"  - 台账已导出 → {path}（请人工核对）")
    return path


def _load_probe(cfg: PerfConfig) -> dict | None:
    """
    读取带宽探针结果（output/_env/probe_<kind>.json）。

    由 `perfkit probe` 或 scripts/run_closure.sh 生成。用文件解耦，
    避免每次 baseline 都重新编译/运行探针 —— 探针测的是**硬件**，
    与模型无关，一次测量可被多次运行复用。

    按后端分文件存放，并且**校验来源**：CPU 探针测的是主机内存路径，
    GPU 探针测的是显存路径，两者不可互换。拿 CPU 的数字去算 GPU 的
    达成率会得出荒谬的结论（实践中出现过 4963% 的比值）。
    """
    env_dir = cfg.root / "output" / "_env"
    kind = cfg.backend.kind
    cands = [env_dir / f"probe_{kind}.json", env_dir / "probe.json"]
    for p in cands:
        if not p.exists():
            continue
        try:
            d = load_json(p)
        except Exception:
            continue
        # 来源校验：backend 字段不匹配的一律不采信（含历史遗留的 probe.json）
        src = d.get("backend")
        if src and src != kind:
            continue
        return d
    return None


def _env_context(cfg: PerfConfig, live: bool = True) -> dict:
    """
    采集「测量有效性」上下文。

    CPU 后端共享内存带宽与物理核：同一份配置在空闲机器上跑 17.85 t/s、
    在负载 90+ 的机器上只跑 9.19 t/s。不记录当时负载，
    这个数字事后就无法判断是否可信 —— 这是 CPU 场景必须补的一环。

    live=False（replay）时只记录「台账声明的有效性条件」，
    不把**当前**负载当成数据有效性的判据 —— 台账数据是过去采集的，
    与现在这台机器忙不忙无关。混在一起会产生误导性告警。
    """
    out: dict = {"backend": cfg.backend.kind, "measured_live": live}
    mv = cfg.device.get("measurement_validity")
    if mv:
        out["declared_validity"] = mv
    # 带宽探针结果：由 scripts/run_closure.sh（或人工）写到 output/_env/probe.json。
    # 用文件解耦，避免 baseline 每次都重新编译/运行探针。
    probe = _load_probe(cfg)
    if probe:
        out["probe"] = probe
    if not live:
        out["note"] = ("replay 模式：数据来自台账（采集时点见 ledger.meta.collected），"
                       "本次运行未做真机测量，故不评估当前系统负载。")
        return out
    try:
        la = os.getloadavg()
        out["load_avg"] = [round(x, 2) for x in la]
        cores = cfg.device.get("compute", {}).get("cpu_cores") or os.cpu_count() or 1
        out["cpu_cores"] = cores
        out["load_per_core"] = round(la[0] / cores, 2)
        # 只有**共享内存带宽**的后端才把主机负载当作数据有效性判据。
        # 独显 GPU 的显存带宽是独立的，主机负载压不低它 —— 在这里挂告警
        # 会让真正的风险（CPU 上没记负载）被淹没在噪声里。
        out["bandwidth_shared"] = cfg.backend.bandwidth_is_shared()
        if cfg.backend.bandwidth_is_shared():
            out["contended"] = la[0] > cores
            if la[0] > cores:
                out["warning"] = (
                    f"1 分钟负载 {la[0]:.1f} 超过物理核数 {cores}，"
                    f"本次 live 测量可能被外部负载压低"
                    f"（{cfg.backend.label} 的内存带宽与主机共享）。"
                    f"建议在机器空闲时重测，或用 `uptime` 复核。")
        else:
            out["note"] = (
                f"{cfg.backend.label} 的{cfg.backend.memory_label}带宽独立于主机负载，"
                f"当前负载 {la[0]} 不影响本次测量；仍记录以备排查异常。")
    except (OSError, AttributeError):
        pass
    return out


def _roofline(cfg: PerfConfig) -> dict:
    from ..config import arithmetic_intensity, roofline_verdict, weight_bytes
    ctx = cfg.arch.get("max_ctx_deployed", 8192)
    ai = arithmetic_intensity(cfg.model, ctx, weight_bytes(cfg.model))
    return roofline_verdict(cfg.device, ai)
