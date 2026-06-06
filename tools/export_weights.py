"""Offline weight export: safetensors (bf16) -> custom packed file for the C/CUDA engine.

Output:
  scratch/engine_weights/weights.bin   (concatenated tensor data)
  scratch/engine_weights/manifest.json (list of tensors: name, kind, dtype, shape, offset, nbytes, + quant meta)

Tensor kinds:
  - "fp16": raw little-endian fp16 (embed, norms, conv kernel, gate, expert_bias->fp32)
  - "int4g": group-wise symmetric INT4. Layout per matrix W[N,K] (row-major over N, K inner):
        packed nibbles: N*(K/2) bytes, two K-adjacent nibbles per byte (k even in low nibble).
        scales: N*(K/G) fp16.
    dequant: w[n,k] = (nibble - 8) * scale[n, k//G]
This is DENSE INT4 (no 2:4 sparsity) -- see kb/02_decisions.md.

Author: g023 (https://github.com/g023/)
"""
import os, json, struct, time
import numpy as np
import torch
from safetensors import safe_open

ROOT = os.environ.get("SPINT4_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = f"{ROOT}/scratch/LFM2.5-8B-A1B/model.safetensors"
OUT = f"{ROOT}/scratch/engine_weights"
os.makedirs(OUT, exist_ok=True)
G = 128  # group size along K

NUM_LAYERS = 24
ATTN_LAYERS = {2, 6, 10, 14, 18, 21}
NUM_DENSE = 2
NUM_EXPERTS = 32

def log(*a): print("[export]", *a, flush=True)

f = safe_open(MODEL, framework="pt")
keys = set(f.keys())

binf = open(f"{OUT}/weights.bin", "wb")
manifest = []
offset = 0

def bf16_to_fp16_bytes(arr_u16):
    # arr_u16: uint16 bf16 bits -> float32 -> float16 bytes
    u32 = arr_u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.astype(np.float16)

def write_fp16(name, np_fp16):
    global offset
    b = np_fp16.tobytes()
    binf.write(b)
    manifest.append({"name": name, "kind": "fp16", "shape": list(np_fp16.shape),
                     "offset": offset, "nbytes": len(b)})
    offset += len(b)

def to_f32(name):
    t = f.get_tensor(name)  # torch tensor (bf16/f32)
    return t.float().numpy()

def write_fp16_from(name, outname=None):
    arr = to_f32(name).astype(np.float16)
    write_fp16(outname or name, arr)

def quant_int4_group(w_f32):
    # w_f32: [N, K]
    N, K = w_f32.shape
    assert K % G == 0, (name, K, G)
    wg = w_f32.reshape(N, K // G, G)
    amax = np.abs(wg).max(axis=2, keepdims=True)  # [N, K/G, 1]
    scale = (amax / 7.0).astype(np.float32)
    scale_safe = np.where(scale == 0, 1.0, scale)
    q = np.round(wg / scale_safe).clip(-7, 7).astype(np.int8)  # [N,K/G,G]
    nib = (q + 8).astype(np.uint8).reshape(N, K)  # 0..15
    # pack two along K: byte = low(k even) | high(k odd)<<4
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)  # [N, K/2]
    scales_fp16 = scale.reshape(N, K // G).astype(np.float16)
    return packed, scales_fp16

def write_int4(name, outname=None):
    global offset
    w = to_f32(name)
    assert w.ndim == 2
    N, K = w.shape
    packed, scales = quant_int4_group(w)
    pb = packed.tobytes(); sb = scales.tobytes()
    binf.write(pb); binf.write(sb)
    manifest.append({"name": outname or name, "kind": "int4g", "shape": [N, K], "G": G,
                     "offset": offset, "packed_nbytes": len(pb), "scales_nbytes": len(sb),
                     "nbytes": len(pb) + len(sb)})
    offset += len(pb) + len(sb)

t0 = time.time()
# --- non-layer ---
write_fp16_from("model.embed_tokens.weight")          # tied lm_head
write_fp16_from("model.embedding_norm.weight")

for L in range(NUM_LAYERS):
    p = f"model.layers.{L}."
    write_fp16_from(p + "operator_norm.weight")
    write_fp16_from(p + "ffn_norm.weight")
    if L in ATTN_LAYERS:
        write_int4(p + "self_attn.q_proj.weight")
        write_int4(p + "self_attn.k_proj.weight")
        write_int4(p + "self_attn.v_proj.weight")
        write_int4(p + "self_attn.out_proj.weight")
        write_fp16_from(p + "self_attn.q_layernorm.weight")
        write_fp16_from(p + "self_attn.k_layernorm.weight")
    else:
        write_int4(p + "conv.in_proj.weight")
        write_int4(p + "conv.out_proj.weight")
        # conv depthwise kernel [H,1,3] -> store fp16 as [H,3]
        ck = to_f32(p + "conv.conv.weight")  # [2048,1,3]
        write_fp16(p + "conv.conv.weight", ck.reshape(ck.shape[0], ck.shape[2]).astype(np.float16))
    # feed forward
    if L < NUM_DENSE:
        write_int4(p + "feed_forward.w1.weight")
        write_int4(p + "feed_forward.w3.weight")
        write_int4(p + "feed_forward.w2.weight")
    else:
        write_fp16_from(p + "feed_forward.gate.weight")  # router, small, keep fp16
        eb = to_f32(p + "feed_forward.expert_bias").astype(np.float16)
        write_fp16(p + "feed_forward.expert_bias", eb)
        for e in range(NUM_EXPERTS):
            ep = p + f"feed_forward.experts.{e}."
            write_int4(ep + "w1.weight")
            write_int4(ep + "w3.weight")
            write_int4(ep + "w2.weight")
    if L % 4 == 0:
        log(f"layer {L}/{NUM_LAYERS} done, {offset/1e9:.2f} GB, {time.time()-t0:.0f}s")

binf.close()
meta = {"G": G, "hidden": 2048, "num_layers": NUM_LAYERS, "attn_layers": sorted(ATTN_LAYERS),
        "num_dense": NUM_DENSE, "num_experts": NUM_EXPERTS, "head_dim": 64,
        "n_heads": 32, "n_kv_heads": 8, "vocab": 128000, "rope_theta": 5000000.0,
        "norm_eps": 1e-5, "moe_top_k": 4, "moe_inter": 1792, "dense_inter": 7168,
        "tensors": manifest}
json.dump(meta, open(f"{OUT}/manifest.json", "w"))
log(f"DONE {offset/1e9:.2f} GB, {len(manifest)} tensors, {time.time()-t0:.0f}s")
