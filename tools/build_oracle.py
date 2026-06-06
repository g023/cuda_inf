"""Build a reference oracle from transformers for validating the C/CUDA engine.

Dumps, for a fixed prompt:
  - input_ids
  - per-layer hidden states (embeddings + each of 24 decoder layer outputs + final normed)
  - final logits for all positions
  - greedy continuation token ids + decoded text
Everything saved as little-endian raw float32 / int32 plus a JSON manifest.

Author: g023 (https://github.com/g023/)
"""
import os, json, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("SPINT4_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = f"{ROOT}/scratch/LFM2.5-8B-A1B"
OUT = f"{ROOT}/scratch/oracle"
os.makedirs(OUT, exist_ok=True)

USER_MSG = "What is the capital of France? Answer in one sentence."
GEN_STEPS = 48

def log(*a):
    print("[oracle]", *a, flush=True)

torch.set_num_threads(16)
t0 = time.time()
log("loading tokenizer")
tok = AutoTokenizer.from_pretrained(MODEL)
log("loading model (fp32, cpu, eager attn) -- GPU too small for 16GB bf16")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").eval()
log("loaded in %.1fs" % (time.time() - t0))

msgs = [{"role": "user", "content": USER_MSG}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(ids):
    ids = ids["input_ids"]
PROMPT = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
log("prompt:", repr(PROMPT), "ids:", ids.tolist())

with torch.no_grad():
    out = model(ids, output_hidden_states=True, use_cache=False)

hs = out.hidden_states  # tuple len 25: embeddings + 24 layer outputs (pre final norm)
logits = out.logits[0].float().cpu().numpy()  # [seq, vocab]
log("num hidden_states:", len(hs), "logits shape:", logits.shape)

# final normed hidden (last_hidden_state) = embedding_norm(hs[-1])
with torch.no_grad():
    final_normed = model.model.embedding_norm(hs[-1]).float().cpu().numpy()[0]

manifest = {"prompt": PROMPT, "input_ids": ids[0].tolist(), "seq_len": int(ids.shape[1]),
            "vocab": int(logits.shape[1]), "hidden": int(hs[0].shape[-1]),
            "num_hidden_states": len(hs)}

ids[0].cpu().numpy().astype(np.int32).tofile(f"{OUT}/input_ids.i32")
for i, h in enumerate(hs):
    h[0].float().cpu().numpy().astype(np.float32).tofile(f"{OUT}/hidden_{i:02d}.f32")
final_normed.astype(np.float32).tofile(f"{OUT}/final_normed.f32")
logits.astype(np.float32).tofile(f"{OUT}/logits.f32")

# argmax of last position
last_logits = logits[-1]
log("last-pos top5 token ids:", np.argsort(-last_logits)[:5].tolist())
log("last-pos argmax:", int(last_logits.argmax()), repr(tok.decode([int(last_logits.argmax())])))

# greedy continuation
log("greedy generating %d steps" % GEN_STEPS)
with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=GEN_STEPS, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
gen_ids = gen[0].cpu().tolist()
new_ids = gen_ids[ids.shape[1]:]
text = tok.decode(gen_ids, skip_special_tokens=True)
cont = tok.decode(new_ids, skip_special_tokens=True)
manifest["greedy_new_ids"] = new_ids
manifest["greedy_text"] = text
manifest["greedy_continuation"] = cont
log("greedy continuation:", repr(cont))

json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
log("done in %.1fs -> %s" % (time.time() - t0, OUT))
