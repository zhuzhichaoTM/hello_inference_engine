"""perfkit 子命令注册表。新增环节只需在此追加一个模块。"""

from . import (all, baseline, define, diagnose, optimize, probe, profiling,
               query, report, verify, visualize)

# 顺序即 `perfkit --help` 里的展示顺序，与七步闭环的先后一致；
# probe/query 是辅助命令，排在流程命令之后。
MODULES = [define, baseline, profiling, diagnose, optimize, verify, report,
           visualize, probe, query, all]

__all__ = ["MODULES"]
