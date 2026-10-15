"""
量化框架共享工具模块 (quant_utils.py)
提供跨模块复用的子进程执行、日志落盘与目录初始化能力，
供 imatrix_generator 与 gguf_quantizer 两个独立引擎共同调用。
"""

import os
import subprocess
import logging
from typing import List, Optional

logger = logging.getLogger("QuantUtils")


def ensure_dirs(*dirs: str) -> None:
    """批量确保目录存在"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def run_command(
    cmd: List[str],
    log_path: Optional[str] = None,
    quiet_keywords: Optional[List[str]] = None,
) -> None:
    """
    安全运行外部子进程，流式输出并可选落盘日志。

    :param cmd: 完整命令行参数列表
    :param log_path: 日志文件绝对路径；为 None 时仅走终端
    :param quiet_keywords: 命中这些关键字的行仅写 debug 级别，避免刷屏
    :raises RuntimeError: 子进程返回码非 0
    """
    cmd_str = " ".join(cmd)
    logger.info(f"执行命令: {cmd_str}")

    if quiet_keywords is None:
        quiet_keywords = ["writing", "layer", "chunks", "imatrix", "error", "Error", "100%"]

    log_fp = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            if log_fp:
                log_fp.write(line)
            if any(k in line for k in quiet_keywords):
                logger.debug(line.strip())

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"命令执行失败，退出码 {proc.returncode}，详情查看: {log_path or '终端'}"
            )
    finally:
        if log_fp:
            log_fp.close()
