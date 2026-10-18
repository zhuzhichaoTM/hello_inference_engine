"""
perfkit · GGUF 元数据解析
==============================================================================
用途：把 model_*.yaml 里「[I] 反推」的架构超参，换成从模型文件里读出来的真值。

为什么这件事重要？
    整个性能模型建立在架构超参之上：n_head_kv 与 head_dim 决定 KV Cache 大小，
    n_layer 与 hidden 决定逐层带宽。这些值如果是猜的，那么显存预算、
    瓶颈归因、方案收益全都是在猜的基础上推出来的 —— 误差会被逐级放大。
    本次分析中 KV Cache 的「模型 6.5 GiB vs 实测 2.2 GiB」差额，
    根源就在这里。

本模块只做一件事：把 GGUF 文件的头部读出来，不加载权重，因此
对 15 GiB 的文件也是毫秒级返回（只读前若干 MB）。

GGUF 文件布局（v2/v3）：
    magic        char[4]   "GGUF"
    version      uint32
    tensor_count uint64
    kv_count     uint64
    kv_pairs     [key:string, type:uint32, value:...] × kv_count
    tensors      [name:string, n_dims:uint32, dims:uint64[], type:uint32,
                  offset:uint64] × tensor_count

    string = uint64 长度 + UTF-8 字节（v1 用 uint32 长度）
"""

from __future__ import annotations

import struct
from pathlib import Path

# GGUF 元数据值类型枚举
GGUF_TYPES = {
    0: ("B", 1, "uint8"), 1: ("b", 1, "int8"),
    2: ("H", 2, "uint16"), 3: ("h", 2, "int16"),
    4: ("I", 4, "uint32"), 5: ("i", 4, "int32"),
    6: ("f", 4, "float32"), 7: ("?", 1, "bool"),
    8: ("str", 0, "string"), 9: ("arr", 0, "array"),
    10: ("Q", 8, "uint64"), 11: ("q", 8, "int64"),
    12: ("d", 8, "float64"),
}


class GgufError(Exception):
    """GGUF 解析错误：文件损坏、版本不支持、格式不符。"""


class _Reader:
    """按需读取的小端字节流读取器，带边界检查。"""

    def __init__(self, fh, size: int):
        self.fh = fh
        self.size = size
        self.pos = 0

    def _need(self, n: int) -> None:
        if self.pos + n > self.size:
            raise GgufError(f"读取越界：需要 {n} 字节，剩余 {self.size - self.pos}")

    def read(self, n: int) -> bytes:
        self._need(n)
        b = self.fh.read(n)
        if len(b) != n:
            raise GgufError("文件被截断")
        self.pos += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self, str_len_is_u64: bool) -> str:
        n = self.u64() if str_len_is_u64 else self.u32()
        if n > 1 << 24:                       # 单字符串 >16MB 视为异常
            raise GgufError(f"字符串长度异常：{n}")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, str_len_is_u64: bool):
        if vtype == 8:
            return self.string(str_len_is_u64)
        if vtype == 9:                        # array
            elem_type = self.u32()
            count = self.u64()
            if count > 1 << 24:
                raise GgufError(f"数组长度异常：{count}")
            # 只保留前若干项，避免 vocab 之类的大数组吃光内存
            items = []
            for i in range(count):
                v = self.value(elem_type, str_len_is_u64)
                if i < 64:
                    items.append(v)
            return {"_array_len": count, "_elem_type": elem_type, "items": items}
        fmt = GGUF_TYPES.get(vtype)
        if fmt is None:
            raise GgufError(f"未知的元数据类型 {vtype}")
        code, sz, _ = fmt
        return struct.unpack("<" + code, self.read(sz))[0]


def read_metadata(path: str | Path, max_bytes: int = 64 << 20) -> dict:
    """
    读取 GGUF 头部：元数据 KV + 张量名列表。

    max_bytes 限制只读文件前 N 字节（默认 64MB），避免大文件全量读。
    元数据通常在文件开头几百 KB 内，64MB 有充足余量。
    """
    path = Path(path)
    if not path.exists():
        raise GgufError(f"模型文件不存在: {path}")
    size = min(path.stat().st_size, max_bytes)

    with path.open("rb") as fh:
        r = _Reader(fh, size)
        magic = r.read(4)
        if magic != b"GGUF":
            raise GgufError(f"不是 GGUF 文件（magic={magic!r}）")
        version = r.u32()
        if version not in (1, 2, 3):
            raise GgufError(f"不支持的 GGUF 版本 {version}")
        str_len_is_u64 = version >= 2
        tensor_count = r.u64()
        kv_count = r.u64()

        meta: dict = {}
        for _ in range(kv_count):
            key = r.string(str_len_is_u64)
            vtype = r.u32()
            meta[key] = r.value(vtype, str_len_is_u64)

        # 张量名：用于核对层数与张量类型分布
        tensors = []
        for _ in range(min(tensor_count, 4096)):
            name = r.string(str_len_is_u64)
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            _offset = r.u64()
            tensors.append({"name": name, "dims": dims, "type": ttype})

    return {"version": version, "tensor_count": tensor_count,
            "kv_count": kv_count, "metadata": meta, "tensors": tensors,
            "file_bytes": path.stat().st_size, "path": str(path)}


# ---------------------------------------------------------------------------
# 把 GGUF 元数据映射成 perfkit 的 architecture 字段
# ---------------------------------------------------------------------------
# 不同架构的键名不同，这里按 llama.cpp 的 gguf-py 约定做映射。
# 键名统一带 `<arch>.` 前缀（如 qwen3.block_count）。
ARCH_KEYS = {
    "n_layer":          ["block_count"],
    "hidden_size":      ["embedding_length"],
    "intermediate_size": ["feed_forward_length"],
    "n_head":           ["attention.head_count"],
    "n_head_kv":        ["attention.head_count_kv"],
    "vocab_size":       ["vocab_size"],
    "rope_theta":       ["rope.freq_base"],
    "norm_eps":         ["attention.layer_norm_rms_epsilon",
                         "attention.layer_norm_epsilon"],
    "max_ctx_native":   ["context_length"],
    "sliding_window":   ["attention.sliding_window"],
    "n_expert":         ["expert_count"],
    "n_expert_used":    ["expert_used_count"],
}

QUANT_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 28: "BF16", 30: "TQ1_0", 31: "TQ2_0",
}


def extract_architecture(info: dict) -> dict:
    """从元数据里提取架构超参，返回可直接覆盖 configs 的字段。"""
    meta = info["metadata"]
    arch = meta.get("general.architecture", "")
    out: dict = {"arch": arch}
    for field, suffixes in ARCH_KEYS.items():
        for suf in suffixes:
            key = f"{arch}.{suf}" if arch else suf
            if key in meta and not isinstance(meta[key], dict):
                out[field] = meta[key]
                break
    # head_dim：GGUF 里通常不直接给，由 hidden/n_head 推得（这是唯一可靠来源）
    if "hidden_size" in out and "n_head" in out and out["n_head"]:
        out["head_dim"] = out["hidden_size"] // out["n_head"]
    # tie_embeddings：看是否存在 output.weight
    names = {t["name"] for t in info["tensors"]}
    out["tie_embeddings"] = "output.weight" not in names
    return out


def tensor_type_histogram(info: dict) -> dict:
    """统计各量化类型占用的张量数 —— 用于核对 UD 动态量化的混合策略。"""
    hist: dict[str, int] = {}
    for t in info["tensors"]:
        name = QUANT_TYPE_NAMES.get(t["type"], f"type{t['type']}")
        hist[name] = hist.get(name, 0) + 1
    return hist


def count_layers_by_tensor_names(info: dict) -> int:
    """按张量名里的 blk.N 前缀数层数 —— 独立于元数据的交叉校验。"""
    idx = set()
    for t in info["tensors"]:
        n = t["name"]
        if n.startswith("blk."):
            try:
                idx.add(int(n.split(".")[1]))
            except (IndexError, ValueError):
                continue
    return len(idx)


def diff_against_config(extracted: dict, config_arch: dict) -> list[dict]:
    """逐字段对比「文件真值」与「配置里的反推值」。"""
    rows = []
    for k, v in extracted.items():
        if k in ("arch", "tie_embeddings"):
            continue
        cur = config_arch.get(k)
        if cur is None:
            continue
        same = (abs(float(v) - float(cur)) < 1e-6
                if isinstance(v, (int, float)) and isinstance(cur, (int, float))
                else v == cur)
        rows.append({"field": k, "config": cur, "gguf": v, "match": same})
    return rows
