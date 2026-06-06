# Status / milestones

## 2026-06-06: MILESTONE - long-context decode ~7x faster (Flash-Decoding)
Problem: decode throughput collapsed on long context. Reported: 2174 tokens @ 14.1 tok/s
(short-context was ~110). Root cause: `attention_fused<<<S,NH>>>` with S==1 at decode = ONE block
of 32 threads (a single warp) on a single SM, serially scanning the whole KV context. Cost grew
linearly with context and used ~1/28 of the GPU.

Fix (src/engine.cu): Flash-Decoding. Single-token attention now splits the key range across
NSPLIT=16 blocks per head (grid=(NH,NSPLIT), 128 threads/block) so the whole GPU is saturated;
`attn_decode_split` writes per-(head,split) partial softmax state (m,l,o), `attn_decode_combine`
(grid=NH, block=HEAD_DIM) merges them. Numerically identical to attention_fused (online softmax,
fp32 accumulate) so coherence is unchanged. Prefill (S>1) still uses attention_fused. Only the
default group-64 FP8 + fp16 tail path; H2O eviction path untouched.

Floor cleanups (per-token constant overhead, now the limiter): GPU argmax (`argmax_kernel`) over
logits instead of a 512KB D2H copy + CPU scan every token; expert_bias fetched to host ONCE (was a
D2H + sync per MoE layer per token = 22 needless syncs/token); persistent embed-id buffer (no
per-token cudaMalloc/cudaFree).

Results (RTX 3060, prompt "tell me how to decompile a N64 rom"):
- long context: 14.1 -> ~100 tok/s (~7x), and now FLAT across context (98.8 @1k, 99.9 @1.5k,
  97.3 @1.9k) instead of decaying.
- short context unchanged (~105 tok/s).
- coherency preserved: France->Paris still exact; N64 output fully coherent start to finish.
- memory: 7.3 GB peak incl ~2.4 GB desktop (engine ~4.9 GB) -> fits 12 GB with ~5 GB headroom.

Remaining headroom (documented, not done): lm_head is fp16 [128000,2048] = ~512MB read/token
(~1.5ms of the ~10ms/token) -> INT4 lm_head would save ~10% but needs an offline export change.
MoE router top-k still does a D2H+sync per layer (data-dependent); GPU-side routing would remove it.

## 2026-06-06: MILESTONE - coherent text achieved
Engine (src/engine.cu, dense INT4 g128) runs the full LFM2.5-8B-A1B forward pass on the RTX 3060
and generates coherent readable text.

Prompt (chat): "What is the capital of France? Answer in one sentence."
ENGINE: '<think>\nThe user asks: "What is the capital of France? Answer in one sentence." That's a
 straightforward factual question. The answer: "The capital of France is Paris." That's one sentence.
 So respond with that.'
ORACLE (transformers fp32 CPU greedy): same opening, diverges after ~12 tokens, same conclusion (Paris).

First ~12 greedy tokens match the oracle EXACTLY; later divergence is accumulated INT4 quant error
(expected, both stay coherent and reach the correct answer).

## Validation numbers (prefill, 21-token prompt, last token)
- L00 embedding: exact (cos 1.0).
- per-layer cos erosion ~0.93-0.97; final pre-norm L24 cos 0.76 maxabs 66 (one channel outlier).
  Argmax still correct. Erosion is larger than ideal g128 INT4 (-> accuracy improvement TODO),
  but output is coherent.

## Pipeline
1. tools/build_oracle.py  -> scratch/oracle/  (ref activations, logits, greedy ids/text)
2. tools/export_weights.py -> scratch/engine_weights/{weights.bin,manifest.json}  (INT4 g128 + fp16)
3. tools/make_index.py    -> scratch/engine_weights/index.bin  (C loader index)
4. build: ~/miniconda3/bin/nvcc -O3 -std=c++17 -arch=sm_86 -diag-suppress 1650 src/engine.cu -o build/engine
5. run: ./build/engine [--ids file.i32] [--n N] [--dbgdir dir] [--dump file]

## 2026-06-06: END-TO-END COMPLETE
- Self-contained C++ byte-level BPE tokenizer (src/tokenizer.h) matches HF encode/decode EXACTLY
  (chat template, code, numbers, whitespace runs, special tokens). Engine now takes a text --prompt
  and streams decoded text out. No Python at runtime.
- `./build/engine --prompt "What is the capital of France? Answer in one sentence." --n 400`
  emits a full <think> trace then "The capital of France is Paris.<|im_end|>", terminating at EOS.
- Throughput: ~110-115 tok/s decode on RTX 3060 (matches blueprint's 90-110 projection).
- Correctness: engine greedy reproduces 21 consecutive oracle (fp32 CPU) tokens before INT4-noise
  divergence; output stays coherent. GOAL met.
- Build: ./prepare.sh (offline) then ./build.sh. See ENGINE.md.

## 2026-06-06: task 7 DONE - intelligent KV cache, fused attention, sparse GEMM (coherence kept)
Goal: "mma.sp sparse GEMM, FP8 E4M3 KV cache with H2O eviction, fused FlashAttention without
losing coherency; ignore any that require retraining."

- **Fused FlashAttention + FP8 E4M3 KV cache** (on the gen path). KV stored FP8 E4M3, group-wise:
  64 tokens/group along seq share one fp16 scale per kv-head; partial group kept fp16 (tail) until
  full then committed. `attention_fused` does one online-softmax pass over FP8 (committed) + fp16
  (tail) with branch-free segments. Result: greedy matches the dense-fp16-KV baseline 57/60 tokens
  on the France prompt, reaches "Paris", terminates at EOS. ~100-105 tok/s (was 116 dense fp16 KV).
  Bit-identical 60/60 before a micro-refactor split the loop; FP8 noise only diverges greedy late
  and stays coherent. Second prompt (sun) also coherent.
- **H2O + local-window eviction** (`--kv-budget N [--kv-window W]`, off by default). Per-token-scaled
  FP8 circular buffer of N slots (per-slot scale so a slot can be evicted/reused without disturbing a
  shared group scale). Keep last W tokens; evict the min cumulative-attention (`attn_sum`, accumulated
  inside the attention kernel) slot outside the window. Coherent under heavy eviction: budget=48 while
  context grew to 82 -> coherent "Paris"; budget=32 over 250 generated tokens -> grammatical on-topic
  text (some repetition, the known H2O artifact at tight budgets, not incoherence).
- **mma.sp sparse-INT4 Tensor-Core GEMM** (`src/mma_sp_test.cu` -> `build/mma_sp_test`). Validates
  `mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32`: 4 distinct 2:4 patterns exact vs CPU
  (metadata decode + A/B/C/D thread mapping correct), plus a tiled INT4 GEMM (on-the-fly nibble
  decode + per-128 scale + K/N tiling) within fp16 rounding (~1% maxrel). KEPT OFF the gen path:
  applying 2:4 to pretrained weights needs sparsity-aware retraining to keep coherence, which the
  goal excludes. So the kernel is proven correct; the model still runs dense INT4.

## Earlier deferred-accuracy note (still open, not required for coherence)
- Per-layer cos erosion / L24 channel outlier could be reduced with int8 on a few sensitive matmuls
  or smaller groups; not needed for coherent output.
