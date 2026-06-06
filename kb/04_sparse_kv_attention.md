# Sparse Tensor-Core GEMM, FP8 KV cache, fused FlashAttention, H2O (task 7 internals)

Crucial low-level details so these can be re-derived without re-discovery. Code: `src/engine.cu`
(FP8 KV + fused attention + H2O) and `src/mma_sp_test.cu` (sparse Tensor-Core GEMM validation).

## 1. mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32  (Ampere sm_86)
Tile: D[16x8] += A[16x32, 2:4 sparse] * B[32x8], fp32 accumulate. A's 32 logical K-cols are 2:4
sparse so only 16 nonzeros/row are stored; B is dense 32x8.

Per-thread fragments (lane = threadIdx.x in the warp; gr = lane>>2 in 0..7; tc = (lane&3)*2):
- **A (4x .b32 = 8 halfs)**: the 16 stored nonzeros laid out exactly like a dense m16n8k16 A:
  rows {gr, gr+8}, stored-cols {tc, tc+1} and {tc+8, tc+1+8}.
    a0=(gr,tc),(gr,tc+1)  a1=(gr+8,tc),(gr+8,tc+1)
    a2=(gr,tc+8),(gr,tc+9) a3=(gr+8,tc+8),(gr+8,tc+9)
- **B (4x .b32 = 8 halfs)**, B is K x N, column n=gr:
    b_r = (k=tc+16*r{,+1}, n=gr) for r=0..3  ->  k in {tc,tc+8,tc+16,tc+24} (+1)
- **C/D (4x .f32)**: rows {gr, gr+8}, cols {tc, tc+1}:  d0=(gr,tc) d1=(gr,tc+1) d2=(gr+8,tc) d3=(gr+8,tc+1)
- **metadata (1x .b32) + sparsity_selector immediate (use 0x0)**. The b32 packs 8 nibbles, one per
  group-of-4 along K; each nibble = (idx1<<2)|idx0 with idx0<idx1 the two NONZERO positions in that
  quad. e.g. nonzeros at {0,1} -> nibble 0x4 -> word 0x44444444; {1,3} -> 0xd -> 0xdddddddd.

Validation trick used: a UNIFORM per-row sparsity pattern makes every row's metadata word identical,
so correctness no longer depends on the (fiddlier) metadata->thread distribution; this isolates and
proves the instruction's numeric contract + A/B/C/D mapping + metadata nibble encoding. Verified
exact for patterns {0,1},{1,3},{0,3},{2,3}. A tiled INT4 GEMM (nibble decode + per-128 scale +
K/N tiling) on top matches CPU within ~1% (fp16 rounding). NOTE: arbitrary PER-ROW-varying patterns
additionally need the metadata->thread distribution table (mechanical extension, not needed here).

ptxas advisory: prefer `.sp::ordered_metadata` on newer archs; plain `.sp` is correct on sm_86.

## 2. Why mma.sp is OFF the generation path
2:4 = zero 50% of weights. Magnitude-pruning a PRETRAINED model to 2:4 without sparsity-aware
finetuning wrecks perplexity -> incoherent text. Restoring coherence needs RETRAINING, which the
goal excludes. So the kernel is validated standalone; inference stays dense INT4 (kb/02_decisions.md).

## 3. FP8 E4M3 KV cache (default path)
- `#include <cuda_fp8.h>`; convert with `__nv_cvt_float_to_fp8(x,__NV_SATFINITE,__NV_E4M3)` and
  `__nv_cvt_fp8_to_halfraw(b,__NV_E4M3)`. E4M3 max normal = 448; E4M3 is FLOAT (relative precision
  ~3 mantissa bits) so a per-group scale only needs to center magnitudes in range, not per-channel.
- Group-wise: KVGROUP=64 tokens along sequence share one fp16 scale per kv-head (scale =
  group_absmax/448). The in-progress partial group is held fp16 in a 64-wide "tail"; when it fills
  it is committed to FP8 (`kv_commit_group`, grid=NKV). `committed[l]` (multiple of 64) splits the
  cache: keys [0,committed) are FP8, [committed,T) are the fp16 tail at index t-committed.
- Append (`kv_write_tail` + commit loop) handles batches that span a 64-boundary by committing each
  completed group before the next reuses tail slots; tail capacity 64 suffices, no shifting.

## 4. Fused FlashAttention (`attention_fused`)
One thread per (query i, head h); online softmax (running max m, denom l). Two BRANCH-FREE segments:
committed keys (FP8, dequant = e4m3*scale) then tail keys (fp16). Splitting avoids a per-key
fp8/fp16 ternary inside the unrolled HEAD_DIM loop -> recovered ~105 tok/s (a single branchy loop
dropped to ~86). Optional 2nd pass (only when attn_sum != null) re-reads K to atomicAdd each key's
normalized prob exp(s-m)/l into attn_sum (for H2O); skipped on the default path so it's free.

## 5. H2O + local-window eviction (`--kv-budget N`, `attention_h2o`)
Tension: group-64 quant assumes 64 contiguous tokens with a shared scale, but circular-buffer
eviction reuses individual slots. Resolution: when eviction is on, use a SEPARATE per-token-scaled
FP8 circular buffer (each slot has its own per-head fp16 scale via `kv_quant_write`) so a slot can
be evicted/reused wholesale. Per attn layer: N slots, `slot_pos[]` (logical position, for causal
mask + policy), `hattn[]` (cumulative attention). Decode: if full, fetch hattn to host, evict the
min-hattn slot with pos <= cur-W (W = local window, default budget/2), reset its hattn, write the
new token there. `attention_h2o` attends all live slots with slot_pos<=qpos. Budget is clamped >=
prompt length (eviction is for decode). Default path (no flag) is untouched and lossless.
