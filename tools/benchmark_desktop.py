#!/usr/bin/env python3
"""Four-tool creature benchmark on the desktop.

Runs every case in bench/cases.json through either the shipped Needle engine
(`--engine`, the reference) or this project's C runtime built for the host
(`--port`), for each model file and state->text template, and reports
tool-selection accuracy, confidence, latency and peak memory.

    python tools/benchmark_desktop.py --engine models/needle3-L2.cact models/needle3.cact
    python tools/benchmark_desktop.py --port models/needle3-L2.cact --think 12
"""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from state_text import render, TEMPLATES

ENGINE = os.path.join(ROOT, "engines", "macos-arm64", "needle")
PORT = os.path.join(ROOT, "host", "needle_host")


def run_engine(model, text, max_tokens):
    env = dict(os.environ, NEEDLE_TELEMETRY="0", DO_NOT_TRACK="1")
    t0 = time.time()
    out = subprocess.run([ENGINE, "--model", model, "--tools", os.path.join(ROOT, "bench", "tools.json"),
                          "--max", str(max_tokens), "--threads", "1", "--prompt", text],
                         capture_output=True, text=True, env=env).stdout
    r = json.loads(out)
    return ([c["name"] for c in r["function_calls"]] or [c["name"] for c in r.get("suppressed_calls", [])],
            r["confidence"], time.time() - t0, r.get("peak_ram_mb"))


def run_port(model, text, think):
    t0 = time.time()
    out = subprocess.run([PORT, "--model", model, "--tools", os.path.join(ROOT, "bench", "tools.min.json"),
                          "--think", str(think), "--allowed-prob", "--prompt", text],
                         capture_output=True, text=True).stdout
    r = json.loads(out)
    return [c["name"] for c in r["function_calls"]], r["confidence"], time.time() - t0, r["peak_ram"] / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", action="store_true")
    g.add_argument("--port", action="store_true")
    ap.add_argument("--max", type=int, default=512, help="engine token budget")
    ap.add_argument("--think", type=int, default=12, help="port reasoning budget")
    ap.add_argument("--templates", default=",".join(TEMPLATES))
    ap.add_argument("--json", default=None, help="write results here")
    a = ap.parse_args()
    cases = json.load(open(os.path.join(ROOT, "bench", "cases.json")))
    results = []
    for model in a.models:
        for tpl in a.templates.split(","):
            ok = 0; confs = []; lat = []; mem = 0; rows = []
            for c in cases:
                text = render(c["state"], tpl)
                calls, conf, dt, m = (run_engine(model, text, a.max) if a.engine
                                      else run_port(model, text, a.think))
                hit = len(calls) >= 1 and calls[0] in c["expected"]
                ok += hit; confs.append(conf or 0); lat.append(dt); mem = max(mem, m or 0)
                rows.append({"text": text, "calls": calls, "confidence": conf, "correct": hit})
            n = len(cases)
            res = {"model": os.path.basename(model), "template": tpl, "runner": "engine" if a.engine else "port",
                   "accuracy": ok / n, "correct": ok, "n": n, "mean_conf": sum(confs) / n,
                   "mean_latency_s": sum(lat) / n, "peak_mem_mb": mem, "rows": rows}
            results.append(res)
            print(f"{res['runner']:6} {res['model']:28} {tpl:8} acc {ok:2}/{n} ({ok/n:.0%})  "
                  f"conf {res['mean_conf']:.2f}  {res['mean_latency_s']*1000:6.0f} ms  {mem:.1f} MB", flush=True)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
