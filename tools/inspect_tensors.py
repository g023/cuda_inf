"""
Author: g023 (https://github.com/g023/)
"""

import json, struct, sys, os
ROOT = os.environ.get("SPINT4_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f = f'{ROOT}/scratch/LFM2.5-8B-A1B/model.safetensors'
fh = open(f, 'rb')
n = struct.unpack('<Q', fh.read(8))[0]
hdr = json.loads(fh.read(n))
keys = [k for k in hdr if k != '__metadata__']
lines = ["num tensors %d" % len(keys)]
for k in sorted(keys):
    if not k.startswith('model.layers.'):
        lines.append("NONLAYER %s %s %s" % (k, hdr[k]['dtype'], hdr[k]['shape']))
for L in ['0', '2']:
    for k in sorted(keys):
        if k.startswith('model.layers.%s.' % L):
            lines.append("L%s %s %s %s" % (L, k, hdr[k]['dtype'], hdr[k]['shape']))
open(f'{ROOT}/scratch/tensors.txt', 'w').write("\n".join(lines) + "\n")
print("wrote", len(lines), "lines")
