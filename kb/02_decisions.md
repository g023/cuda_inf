# Engineering decisions and constraints

## Memory budget (the binding constraint)
GPU = RTX 3060, 12 GB total, but ~2.4 GB used by Xorg/desktop -> ~9.5 GB free.
Model = 8.5B params total (MoE; experts dominate at ~7.75B over 22 MoE layers).
- fp16 weights = 17 GB  -> does NOT fit. (Even the HF bf16 load OOMs on this GPU.)
- INT8 = 8.5 GB -> fits but too tight with activations + KV cache.
- INT4 = ~4.25 GB + scales -> comfortable. CHOSEN.

Per-tensor param counts: embed/lm_head (tied) 262M; 6 attn layers ~63M; 18 conv layers ~302M;
2 dense MLP ~88M; 22 MoE layers ~7.75B (32 experts x (w1+w3+w2), each ~11M/layer).

## Precision plan (accuracy-first path)
- embed_tokens / lm_head (tied): keep fp16 (0.52 GB). It feeds logits; INT4 here risks argmax errors.
- All large matmul weights (attn q/k/v/o, conv in/out_proj, dense w1/w2/w3, MoE gate, expert w1/w2/w3):
  INT4 symmetric group-wise, group size G=128 along K (in-dim). All K dims divisible by 128.
  scale = max(|w_group|)/7, q=clamp(round(w/scale),-7,7) stored as uint nibble (q+8) in [1,15].
  dequant w = (nibble-8)*scale. Two nibbles per byte. scales fp16 per (row, K/128) group.
- Norms (RMSNorm weights), conv depthwise kernel [H,1,3], router gate is INT4 too (small),
  expert_bias (fp32), all kept full/near-full precision: norms+conv+expert_bias stored fp32/fp16.
  -> conv kernel kept fp16 (tiny, precision-sensitive). gate kept fp16 too (routing-sensitive, tiny: 32x2048).

## CRITICAL tradeoff: 2:4 sparsity is OMITTED on the coherence path
The blueprint specifies 2:4 structured sparsity + INT4. Naive 2:4 pruning (zeroing 50% of weights)
of a pretrained model WITHOUT sparsity-aware finetuning massively raises perplexity -> incoherent text.
The session GOAL is "coherent and readable text". These conflict. Resolution:
  - Primary engine uses DENSE INT4 (group-wise) -> preserves coherence, fits memory. This is what
    produces the required coherent text.
  - The mma.sp sparse-INT4 tensor-core GEMM (blueprint centerpiece) is built + unit-validated
    separately (task 7) as a perf kernel, but is NOT on the default generation path because it
    degrades quality. Documented, not hidden.

## Compute path
- Activations fp16. GEMM dequantizes INT4->fp16 on the fly, fp32 accumulate. Correctness first
  (simple tiled GEMM); tensor-core (mma) optimization is task 7.
- Single CUDA stream, token-by-token decode + batched prefill. Bump allocator over one big slab.

## Oracle
- transformers ref runs on CPU fp32 (GPU too small). fp32 ref is the gold standard; engine is fp16/INT4.
  Validation = greedy argmax token match + low logit error, not bit-exactness.
- Prompt: "The capital of France is". See scratch/oracle/manifest.json.
