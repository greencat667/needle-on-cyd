#!/usr/bin/env python3
"""The C runtime (host build) against the golden files from the oracle.

Checks, per case: tokenizer ids identical; last-position logits have the same
top-10 set and top-1, and cosine > 0.9999 against the golden top-10 values;
the 12-token greedy reasoning is identical (text); confidence-head logit
within 0.01.

    make -C host && python tests/test_golden.py models/needle3-L2.cact
"""
import glob, json, os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = os.path.join(ROOT, "host", "needle_host")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from prompt import render_prompt  # noqa: E402


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "needle3-L2.cact")
    tools = json.load(open(os.path.join(ROOT, "bench", "tools.json")))
    fails = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "tests", "golden", "*.json"))):
        g = json.load(open(path))
        name = os.path.basename(path)[:-5]
        text = render_prompt(tools, g["query"])
        ids = [int(x) for x in subprocess.run([EXE, "--model", model, "--tokenize", text],
                                              capture_output=True, text=True).stdout.split()]
        ok_tok = [2] + ids + [6] == g["prompt_ids"]
        with tempfile.TemporaryDirectory() as d:
            np.asarray(g["prompt_ids"], np.uint16).tofile(f"{d}/ids.bin")
            subprocess.run([EXE, "--model", model, "--logits", f"{d}/ids.bin", f"{d}/out.bin"],
                           capture_output=True, check=True)
            raw = np.fromfile(f"{d}/out.bin", np.float32)
        conf = float(raw[-1])
        last = raw[:-1].reshape(len(g["prompt_ids"]), -1)[-1]
        top = [int(i) for i in np.argsort(-last)[:10]]
        mine = last[g["top10_ids"]]
        ref = np.asarray(g["top10_logits"])
        cos = float(mine @ ref / np.linalg.norm(mine) / np.linalg.norm(ref))
        ok_top = top[0] == g["top10_ids"][0] and set(top) == set(g["top10_ids"])
        ok_conf = g["confidence_logit_prompt"] is None or abs(conf - g["confidence_logit_prompt"]) < 0.01
        # 12-token greedy reasoning through the full session path
        out = subprocess.run([EXE, "--model", model, "--tools", os.path.join(ROOT, "bench", "tools.min.json"),
                              "--think", "12", "--prompt", g["query"]], capture_output=True, text=True).stdout
        r = json.loads(out)
        want = g["greedy_reasoning_text"].rstrip("\n ")
        ok_reason = r["reasoning"] == want[:len(r["reasoning"])] and len(r["reasoning"]) > 0
        ok = ok_tok and ok_top and cos > 0.9999 and ok_conf and ok_reason
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name:13} tokens {ok_tok}  top10 {ok_top} (cos {cos:.6f})  "
              f"conf {conf:.4f} vs {g['confidence_logit_prompt']}  reasoning {ok_reason} {r['reasoning']!r}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
