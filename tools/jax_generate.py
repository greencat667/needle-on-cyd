#!/usr/bin/env python3
"""Greedy generation with the reference JAX model on weights read from a .cact.

Used to pin the prompt format: when this reproduces the shipped engine's
reasoning text token for token, the wire format and the numerics agree.

    python tools/jax_generate.py models/needle3.cact bench/tools.json "I'm hungry"
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cact_params import load  # noqa: E402
from prompt import render_prompt  # noqa: E402
from needle.model.architecture import SimpleAttentionNetwork  # noqa: E402
from needle.model import quantize  # noqa: E402

EOS, IM_END, THINK = 1, 5, 6


def main():
    path, tools_path, query = sys.argv[1:4]
    quant = "--noquant" not in sys.argv
    max_new = 96
    params, cfg, tok, _ = load(path)
    quantize.configure_deploy(act_bits=8, kv_bits=cfg.kv_bits)
    tools = json.load(open(tools_path))
    ids = [2] + tok.encode(render_prompt(tools, query)) + [THINK]
    model = SimpleAttentionNetwork(cfg)
    n = len(ids) + max_new
    fn = jax.jit(lambda p, t: model.apply({"params": p}, t, quant=quant))
    buf = np.zeros((1, n), np.int32)
    buf[0, :len(ids)] = ids
    out = []
    for pos in range(len(ids) - 1, n - 1):
        logits = np.asarray(fn(params, jnp.asarray(buf)))[0, pos]
        nxt = int(np.argmax(logits))
        out.append(nxt)
        buf[0, pos + 1] = nxt
        if nxt in (EOS, IM_END):
            break
    print("prompt tokens:", len(ids))
    print("generated ids:", out)
    print(repr(tok.decode([THINK] + out)))


if __name__ == "__main__":
    main()
