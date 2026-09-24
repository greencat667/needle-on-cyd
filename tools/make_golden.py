#!/usr/bin/env python3
"""Golden files from the NumPy oracle (pinned to the JAX reference).

For each case: prompt token ids, the embedding and every layer's residual
lanes (first values and checksums) at the last prompt position, the top-10
logits there, the greedy generation under the tool grammar, and the
confidence-head logit. tests/test_golden.py checks the C runtime against them;
the ESP32 prints the same checksums over serial for the same prompt.

    python tools/make_golden.py models/needle3-L2.cact
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ref_forward import Model
from prompt import render_prompt

CASES = {
    "rest_case": "I need a rest",
    "eat_case": "I'm starving, rested and bored.",
    "sleep_case": "I'm not hungry, exhausted and a bit curious.",
    "explore_case": "I'm not hungry, full of energy and very curious.",
}


def summary(v):
    v = np.asarray(v, np.float64).reshape(-1)
    return {"first8": [round(float(x), 6) for x in v[:8]], "sum": float(v.sum()),
            "abs_sum": float(np.abs(v).sum()), "n": int(v.size)}


def main():
    path = sys.argv[1]
    out_dir = os.path.join(HERE, "..", "tests", "golden")
    os.makedirs(out_dir, exist_ok=True)
    tools = json.load(open(os.path.join(HERE, "..", "bench", "tools.json")))
    m = Model(path)
    for name, query in CASES.items():
        m.reset()
        ids = [2] + m.tok.encode(render_prompt(tools, query)) + [6]
        trace = {}
        for i, t in enumerate(ids):
            h = m.step(t, trace if i == len(ids) - 1 else None)
        lg = m.logits(h)
        top = np.argsort(-lg)[:10]
        g = {"model": os.path.basename(path), "query": query, "prompt_ids": ids,
             "embedding_last": summary(trace["x0"]),
             "layers_last": [summary(trace[f"layer{l}"]) for l in range(m.L)],
             "top10_ids": [int(i) for i in top], "top10_logits": [round(float(lg[i]), 4) for i in top],
             "confidence_logit_prompt": m.confidence_logit() if m.heads else None}
        # greedy continuation, unconstrained, 12 tokens (the device's reasoning budget)
        gen = []
        for _ in range(12):
            t = int(np.argmax(lg))
            gen.append(t)
            if t in (1, 5, 7):
                break
            lg = m.logits(m.step(t))
        g["greedy_reasoning_ids"] = gen
        g["greedy_reasoning_text"] = m.tok.decode(gen)
        json.dump(g, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
        print(f"{name}: {len(ids)} ids, top1 {top[0]}, reasoning {g['greedy_reasoning_text']!r}")


if __name__ == "__main__":
    main()
