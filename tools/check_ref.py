#!/usr/bin/env python3
"""Pin the token-at-a-time NumPy forward (ref_forward.py) to the JAX reference.

Runs the same prompt through both on weights read from the same .cact and
reports the cosine of the last-position logits and of every position's top-1.

    python tools/check_ref.py models/needle3-L2.cact
"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cact_params import load
from prompt import render_prompt
from ref_forward import Model
from needle.model.architecture import SimpleAttentionNetwork
from needle.model import quantize

path = sys.argv[1]
query = sys.argv[2] if len(sys.argv) > 2 else "I'm hungry"
tools = json.load(open(os.path.join(os.path.dirname(__file__), "..", "bench", "tools.json")))
params, cfg, tok, _ = load(path)
quantize.configure_deploy(act_bits=8, kv_bits=cfg.kv_bits)
ids = [2] + tok.encode(render_prompt(tools, query)) + [6]
model = SimpleAttentionNetwork(cfg)
ref = np.asarray(model.apply({"params": params}, jnp.asarray([ids]), quant=True))[0]
conf_ref = float(np.asarray(model.apply({"params": params}, jnp.asarray([ids]), quant=True,
                 method=SimpleAttentionNetwork.forward_confidence))[0]) if "confidence_head" in params else None
m = Model(path)
t0 = time.time()
mine = np.stack([m.logits(m.step(t)) for t in ids])
dt = time.time() - t0
cos = [float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(ref, mine)]
top = np.mean(ref.argmax(-1) == mine.argmax(-1))
print(f"{len(ids)} tokens, numpy {dt:.1f}s")
print(f"cosine min {min(cos):.6f}  mean {np.mean(cos):.6f}  last {cos[-1]:.6f}  top1 agreement {top:.3f}")
print(f"max |dlogit| last {np.max(np.abs(ref[-1]-mine[-1])):.4f}")
if conf_ref is not None:
    print(f"confidence logit jax {conf_ref:.5f}  numpy {m.confidence_logit():.5f}")
