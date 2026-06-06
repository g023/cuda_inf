#!/usr/bin/env bash
# One-time offline preprocessing: download model, export INT4 weights + tokenizer + index.
# Uses (torch + transformers + safetensors). Non-destructive.
# Author: g023 (https://github.com/g023/)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}" # override with PY=/path/to/python ./prepare.sh (needs torch+transformers+safetensors+huggingface_hub)
echo "[prepare] python: $PY"
# 1. download model (skips if present)
[ -f "$ROOT/scratch/LFM2.5-8B-A1B/model.safetensors" ] || \
  "$PY" -c "from huggingface_hub import snapshot_download; \
snapshot_download('LiquidAI/LFM2.5-8B-A1B', local_dir='$ROOT/scratch/LFM2.5-8B-A1B', \
allow_patterns=['*.safetensors','*.json','*.jinja','tokenizer*'])"
# 2. export INT4 weights
"$PY" "$ROOT/tools/export_weights.py"
# 3. binary index for the C loader
"$PY" "$ROOT/tools/make_index.py"
# 4. tokenizer tables
"$PY" "$ROOT/tools/export_tokenizer.py"
echo "[prepare] done -> $ROOT/scratch/engine_weights/"
