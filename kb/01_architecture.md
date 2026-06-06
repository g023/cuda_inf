# LFM2.5-8B-A1B (Lfm2MoeForCausalLM) architecture

Source: transformers/models/lfm2_moe/modeling_lfm2_moe.py (transformers 5.5).
Hybrid conv + attention MoE. All math verified against modeling source.

## Config (config.json)
- hidden_size H = 2048
- num_hidden_layers = 24
- head_dim = 64, num_attention_heads = 32 (Q dim 2048), num_key_value_heads = 8 (KV dim 512), GQA groups = 4
- layer_types (per index): conv conv ATTN conv conv conv ATTN conv conv conv ATTN conv conv conv ATTN conv conv conv ATTN conv conv ATTN conv conv
  - full_attention layers = indices {2,6,10,14,18,21}  (6 total). rest are conv (18).
- conv_L_cache = 3 (depthwise causal conv kernel), conv_bias = false
- num_dense_layers = 2  -> layers 0,1 use dense MLP (intermediate 7168). layers 2..23 use MoE.
- MoE: num_experts = 32, num_experts_per_tok = 4, moe_intermediate_size = 1792, use_expert_bias = true, norm_topk_prob = true, routed_scaling_factor = 1.0
- intermediate_size (dense) = 7168
- norm_eps = 1e-5 (RMSNorm), rope_theta = 5e6, rope_type default, attention_scaling = 1.0
- vocab_size = 128000, tie_word_embeddings = true (lm_head.weight == model.embed_tokens.weight)
- dtype bfloat16. bos=124894 eos=124900 pad=124893

## RMSNorm
fp32: var = mean(x^2) over last dim; x = x * rsqrt(var+eps); out = weight * x (cast back to input dtype AFTER scaling by rsqrt, then * weight). eps=1e-5.

## RoPE (default, NEOX/rotate_half style)
dim=head_dim=64. inv_freq[i] = 1/theta^(2i/64), i=0..31, theta=5e6.
freqs = pos * inv_freq (len 32); emb = cat(freqs,freqs) (len 64). cos=cos(emb), sin=sin(emb).
rotate_half(x): x1=x[:32], x2=x[32:]; return cat(-x2, x1).
q' = q*cos + rotate_half(q)*sin. Same for k. Applied AFTER q/k layernorm, in [head,seq,dim] layout (per head).

## Decoder layer (forward)
residual = h
hn = operator_norm(h)                 # RMSNorm
mix = self_attn(hn) if attn else conv(hn)
h = mix + residual
h = h + feed_forward(ffn_norm(h))     # ffn_norm RMSNorm; feed_forward = dense MLP (L<2) or MoE

## Attention (full_attention layers)
q = q_proj(hn) -> view [.,32,64]; q = q_layernorm(q)  (RMSNorm over head_dim=64, per head)
k = k_proj(hn) -> view [.,8,64];  k = k_layernorm(k)
v = v_proj(hn) -> view [.,8,64]
apply rope to q,k. KV cache append. GQA repeat kv x4.
scores = q.k^T * (1/sqrt(64)=0.125); causal mask; softmax fp32; out = scores.v
out = out_proj(concat heads -> 2048)
Projections all bias=false. q_proj 2048->2048, k_proj/v_proj 2048->512, out_proj 2048->2048.

## ShortConv (conv layers) — depthwise causal conv, gated
in_proj: 2048 -> 3*2048 (bias false). BCx = in_proj(hn).transpose -> [3H, seq]; chunk -> B,C,x each [H,seq].
Bx = B * x  (elementwise per channel/time)
conv: depthwise Conv1d(H,H,kernel=3,groups=H,padding=2,bias=false). weight shape [H,1,3]. w[c,j].
  prefill: conv_out[c,t] = sum_{j=0..2} Bx[c, t-2+j] * w[c,j]  (causal, left pad 2, take [:seq])
  decode (cache of last 3 Bx incl current): conv_out[c] = sum_{j=0..2} state[c,j]*w[c,j]
y = C * conv_out
out = out_proj(y.transpose -> [seq,H])   # out_proj 2048->2048 bias false
NOTE: conv "KV cache" is just last L_cache-1=2 Bx vectors per channel.

## Dense MLP (layers 0,1) — SwiGLU
w1:2048->7168, w3:2048->7168, w2:7168->2048 (all bias false). out = w2(silu(w1 x) * w3 x).

## MoE block (layers 2..23)
gate: Linear 2048->32 (bias false) -> router_logits.
routing_weights = sigmoid(router_logits)            # per expert, [.,32]
scores = routing_weights + expert_bias               # expert_bias fp32 buffer [32]
selected = topk(scores, k=4).indices                 # selection uses biased scores
w = gather(routing_weights, selected)                # WEIGHTS are unbiased sigmoid values
if norm_topk_prob: w = w / (w.sum(-1,keepdim)+1e-6)
w = w * routed_scaling_factor(1.0)
experts: gate_up_proj [32, 3584, 2048], down_proj [32, 2048, 1792].
  for token, for each of 4 experts e:
    gu = gate_up_proj[e] @ x  (3584); gate,up = gu[:1792], gu[1792:]
    h = silu(gate) * up  (1792)
    o = down_proj[e] @ h  (2048)
    out += w_e * o
SwiGLU per expert, intermediate 1792.

## Model wrap
emb = embed_tokens[input_ids]
... 24 layers ...
h = embedding_norm(h)        # final RMSNorm
logits = lm_head(h) = h @ embed_tokens.weight^T   # tied
