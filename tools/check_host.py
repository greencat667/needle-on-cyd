#!/usr/bin/env python3
"""Compare the C runtime (host build) against the NumPy reference, which is
itself pinned to JAX: tokenizer ids, every position's logits, confidence logit.

    python tools/check_host.py models/needle3-L2.cact "I'm hungry"
"""
import json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prompt import render_prompt
from ref_forward import Model

path, query = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "I'm hungry")
exe = os.path.join(HERE, "..", "host", "needle_host")
tools = json.load(open(os.path.join(HERE, "..", "bench", "tools.json")))
m = Model(path)
text = render_prompt(tools, query)
ref_ids = m.tok.encode(text)
out = subprocess.run([exe, "--model", path, "--tokenize", text], capture_output=True, text=True).stdout
c_ids = [int(x) for x in out.split()]
print(f"tokenizer: {len(ref_ids)} ids, identical: {c_ids == ref_ids}")
ids = [2] + ref_ids + [6]
with tempfile.TemporaryDirectory() as d:
    np.asarray(ids, np.uint16).tofile(f"{d}/ids.bin")
    r = subprocess.run([exe, "--model", path, "--logits", f"{d}/ids.bin", f"{d}/out.bin"],
                       capture_output=True, text=True)
    print(r.stderr.strip().splitlines()[-1])
    raw = np.fromfile(f"{d}/out.bin", np.float32)
V = m.emb.shape[0]
c = raw[:-1].reshape(len(ids), V)
conf_c = float(raw[-1])
ref = np.stack([m.logits(m.step(t)) for t in ids])
cos = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(ref, c)]
top = float(np.mean(ref.argmax(-1) == c.argmax(-1)))
print(f"logits cosine min {min(cos):.6f} mean {np.mean(cos):.6f}  top-1 agreement {top:.3f}  "
      f"max|d| {np.abs(ref - c).max():.4f}")
print(f"confidence logit  numpy {m.confidence_logit():.5f}  C {conf_c:.5f}")
