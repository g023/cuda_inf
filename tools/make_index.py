"""Emit a compact binary index for the C loader from manifest.json.

index.bin format (little-endian):
  u32 num_tensors
  per tensor:
    u8  name_len, name bytes (no null)
    u8  kind        (0 = fp16, 1 = int4g)
    u64 offset      (byte offset into weights.bin)
    u64 packed_nbytes
    u64 scales_nbytes   (0 for fp16; for fp16 packed_nbytes holds the data size)
    i64 dim0, i64 dim1  (shape; dim1=0 if 1-D)
    

Author: g023 (https://github.com/g023/)
"""
import json, struct, os
ROOT = os.environ.get("SPINT4_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/scratch/engine_weights"
m = json.load(open(f"{OUT}/manifest.json"))
ts = m["tensors"]
with open(f"{OUT}/index.bin", "wb") as o:
    o.write(struct.pack("<I", len(ts)))
    for t in ts:
        name = t["name"].encode()
        kind = 0 if t["kind"] == "fp16" else 1
        offset = t["offset"]
        if kind == 0:
            packed = t["nbytes"]; scales = 0
        else:
            packed = t["packed_nbytes"]; scales = t["scales_nbytes"]
        sh = t["shape"]
        d0 = sh[0]; d1 = sh[1] if len(sh) > 1 else 0
        o.write(struct.pack("<B", len(name))); o.write(name)
        o.write(struct.pack("<B", kind))
        o.write(struct.pack("<QQQ", offset, packed, scales))
        o.write(struct.pack("<qq", d0, d1))
print("wrote index.bin for", len(ts), "tensors")
