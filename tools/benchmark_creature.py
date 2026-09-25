#!/usr/bin/env python3
"""The creature's benchmark, four tools eat / sleep / explore / play, on the
shipped engine or this runtime's host build. Version 5 (the default): 30 cases,
10 with a world event, in the device's own wording (bench/creature5_cases.json).
--v4: the older 24 four-stat cases.

    python tools/benchmark_creature.py models/needle3-L2-creature5-2bit.cact [--port] [--v4]
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import make_creature4_data, make_creature5_data  # noqa: E402

model = sys.argv[1]
port = "--port" in sys.argv
v4 = "--v4" in sys.argv
cases = json.load(open(os.path.join(ROOT, "bench", "creature_cases.json" if v4 else "creature5_cases.json")))
ok, rows = 0, []
for c in cases:
    text = (make_creature4_data.render_device(c["state"]) if v4
            else make_creature5_data.render_device(c["state"], c["event"]))
    if port:
        out = subprocess.run([os.path.join(ROOT, "host", "needle_host"), "--model", model, "--tools",
                              os.path.join(ROOT, "bench", "creature_tools.min.json"), "--ctx", "180",
                              "--think", "12", "--prompt", text], capture_output=True, text=True).stdout
    else:
        out = subprocess.run([os.path.join(ROOT, "engines", "macos-arm64", "needle"), "--model", model,
                              "--tools", os.path.join(ROOT, "bench", "creature_tools.json"), "--prompt", text],
                             capture_output=True, text=True,
                             env=dict(os.environ, NEEDLE_TELEMETRY="0", DO_NOT_TRACK="1")).stdout
    r = json.loads(out)
    calls = [x["name"] for x in r["function_calls"]]
    hit = bool(calls) and calls[0] in c["expected"]
    ok += hit
    rows.append({"text": text, "calls": calls, "reasoning": r.get("reasoning"), "correct": hit})
    if not hit:
        print("MISS", text, calls)
print(f"{'port' if port else 'engine'} {os.path.basename(model)}: {ok}/{len(cases)}")
json.dump(rows, open(os.path.join(ROOT, "bench", f"creature{'' if v4 else '5'}_{'port' if port else 'engine'}.json"), "w"), indent=1)
