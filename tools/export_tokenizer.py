"""Export tokenizer.json into compact files for the C++ tokenizer.

Outputs in scratch/engine_weights/tok/:
  vocab.tsv   : <id>\t<token_string>   (token_string in byte-level unicode form, e.g. "ĠParis")
  merges.tsv  : <rank>\t<A>\t<B>        (rank = merge priority, lower = earlier)
  special.tsv : <id>\t<content>         (added/special tokens matched literally)
All strings are UTF-8. The C++ side maps byte-level unicode chars <-> raw bytes with the GPT-2 table.

Author: g023 (https://github.com/g023/)
"""
import json, os
ROOT = os.environ.get("SPINT4_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = json.load(open(f"{ROOT}/scratch/LFM2.5-8B-A1B/tokenizer.json"))
OUT = f"{ROOT}/scratch/engine_weights/tok"
os.makedirs(OUT, exist_ok=True)

vocab = T["model"]["vocab"]          # str -> id
merges = T["model"]["merges"]        # list of [A,B] (or "A B")
added = T["added_tokens"]            # list of dicts

with open(f"{OUT}/vocab.tsv", "w", encoding="utf-8") as f:
    for s, i in sorted(vocab.items(), key=lambda kv: kv[1]):
        f.write(f"{i}\t{s}\n")

with open(f"{OUT}/merges.tsv", "w", encoding="utf-8") as f:
    for rank, m in enumerate(merges):
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m[0], m[1]
        f.write(f"{rank}\t{a}\t{b}\n")

with open(f"{OUT}/special.tsv", "w", encoding="utf-8") as f:
    for a in added:
        f.write(f"{a['id']}\t{a['content']}\n")

print("vocab", len(vocab), "merges", len(merges), "special", len(added), "->", OUT)
