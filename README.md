# LFM2.5-8B-A1B pure C/C++/CUDA inference engine
# Author: g023 (https://github.com/g023/)

A self-contained CUDA inference engine for `LiquidAI/LFM2.5-8B-A1B` (hybrid conv + GQA-attention
MoE, 8.5B params, 1B active) targeting a single RTX 3060 (12 GB). No Python, no frameworks at
runtime: a single `.cu` engine + a header-only byte-level BPE tokenizer. Offline preprocessing
(download + quantize) uses Python.

## What works
End-to-end text -> coherent text. Example:

```
$ ./build/engine --prompt "What is the capital of France? Answer in one sentence." --n 400
<think>
The user asks: "What is the capital of France? Answer in one sentence." ... The answer:
"The capital of France is Paris." ...
</think>
The capital of France is Paris.<|im_end|>
[engine] generated 83 tokens in 0.75s (110.9 tok/s)
```

~105 tok/s short-context decode on the RTX 3060, and ~100 tok/s sustained at long context
(Flash-Decoding, see below): ~7x faster than the naive single-warp decode kernel which collapsed
to ~14 tok/s at 2k tokens. Throughput is now flat across context length.

## Flash-Decoding (long-context speed)
Single-token decode attention splits the KV key range across NSPLIT=16 blocks per head
(`attn_decode_split` -> `attn_decode_combine`) instead of the original one-block/one-warp scan, so
the whole GPU is used and per-token cost no longer grows with context. Numerically identical to the
prefill kernel (online softmax, fp32 accumulate); coherence unchanged. Per-token floor was also
trimmed: GPU argmax over logits (1-int copy vs 512KB D2H), expert_bias cached host-side once, and a
persistent embed-id buffer (no per-token malloc). See kb/03_status.md.

## Architecture implemented (see kb/01_architecture.md)
- 24 layers: 18 LFM2 short-conv (depthwise causal conv k=3, gated) + 6 GQA attention (32 Q / 8 KV
  heads, head_dim 64, QK-RMSNorm, RoPE theta 5e6).
- FFN: 2 dense SwiGLU layers + 22 MoE layers (32 experts, top-4, sigmoid router + expert bias,
  norm_topk_prob).
- RMSNorm (eps 1e-5), tied embeddings / lm_head (fp16), vocab 128000.

## Precision (see kb/02_decisions.md)
- Dense INT4 group-wise (G=128) weights for all big matmuls; fp16 embed/lm_head, router gate,
  norms, conv kernels, expert bias. fp32 accumulation.
- Weights ~4.76 GB -> fits the 12 GB card with room for activations + KV.
- NOTE: the blueprint's 2:4 structured sparsity is intentionally OMITTED on the generation path
  (naive 2:4 pruning of a pretrained model destroys coherence). Dense INT4 preserves coherence.

## Intelligent KV cache + fused FlashAttention (task 7, coherence-verified)
- KV cache is **FP8 E4M3**, group-wise quantized: 64 tokens/group along the sequence share one
  fp16 scale per kv-head; the in-progress (partial) group is kept fp16 until full ("tail"), then
  committed to FP8. Decode/attention read FP8 + dequantize on the fly.
- **Fused FlashAttention** (`attention_fused`): single online-softmax pass that reads the FP8
  cache + fp16 tail (branch-free committed/tail segments), fp32 accumulate. Numerically equivalent
  to the prior fp16 attention; greedy output matches the dense-KV baseline for 57/60 tokens on the
  France prompt and stays fully coherent (reaches "Paris", terminates at EOS). ~100-105 tok/s.
- **H2O eviction** (`--kv-budget N [--kv-window W]`): a per-token-scaled FP8 circular buffer of N
  slots. Always keep the last W tokens (local window); among older tokens keep the highest
  cumulative attention (`attn_sum`, accumulated inside the attention kernel); evict the min-`attn_sum`
  slot. Coherent even when most of the context is evicted (e.g. budget=48 while context grows to
  82; budget=32 over 250 generated tokens). Off by default (no flag) so the default path is lossless.
- **Sparse Tensor-Core GEMM** (`src/mma_sp_test.cu`, `build/mma_sp_test`): the blueprint's
  `mma.sp.sync.aligned.m16n8k32` 2:4 sparse INT4 GEMM, validated against a CPU reference (instruction
  decode + metadata/thread mapping exact; tiled INT4 GEMM within fp16 rounding). Kept OFF the
  generation path: 2:4 on pretrained weights needs sparsity-aware retraining to stay coherent
  (excluded by the goal), so this is a correctness harness for the kernel, not part of inference.

## Build / run
```
./prepare.sh         # one-time: download + quantize + export tokenizer (needs conda env unsloth_env)
./build.sh           # compile -> build/engine  (needs nvcc, sm_86)
./build/engine --prompt "your question" --n 200
```
Flags: `--prompt` (chat-wrapped) | `--raw` (no template) | `--ids file.i32`; `--n` max new tokens;
`--no-stream`; `--kv-budget N` (enable H2O eviction, N KV slots) `--kv-window W` (local window);
`--dbgdir dir` (dump per-layer hidden states); `--dump f` (dump final_normed).
`./build/mma_sp_test` validates the sparse Tensor-Core GEMM.

## Validation
`tools/build_oracle.py` runs transformers (fp32, CPU) for a fixed prompt and dumps reference
per-layer activations, logits, and greedy ids. The engine reproduces the oracle's greedy tokens
(first ~12 exact; later divergence is expected INT4 quant noise) and stays coherent. The C++
tokenizer matches HF `encode`/`decode` exactly on chat templates, code, numbers, and whitespace.

## Files
- `src/engine.cu`     - kernels (INT4 GEMV, RMSNorm, RoPE, FP8 KV + fused FlashAttention + H2O,
                        conv, MoE) + host orchestration
- `src/mma_sp_test.cu`- standalone validation of the 2:4 sparse-INT4 mma.sp Tensor-Core GEMM
- `src/tokenizer.h`   - GPT-2 byte-level BPE (encode + decode), header-only
- `tools/`            - offline: export_weights, make_index, export_tokenizer, build_oracle
- `kb/`               - architecture, decisions, status

## Status of task 7 (done) - see kb/03_status.md
- FP8 E4M3 KV cache, fused FlashAttention, H2O eviction: implemented on the generation path,
  coherence-verified (above). mma.sp sparse Tensor-Core GEMM: validated as a standalone kernel,
  kept off the generation path (2:4 on pretrained weights needs retraining to stay coherent).
