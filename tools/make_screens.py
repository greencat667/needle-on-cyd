#!/usr/bin/env python3
"""Screenshots of the creature screen, drawn by the firmware's own UI code
(host/ui_preview), at 2x: docs/screens/creature_*.png and creature_sheet.png.

    .venv/bin/python tools/make_screens.py
"""
import os, subprocess, tempfile
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "screens")
SHEET = ["creature_explore", "creature_eat", "creature_sleep", "creature_play", "creature_speaking",
         "creature_thinking"]

exe = os.path.join(ROOT, "host", "ui_preview")
subprocess.check_call(["c++", "-std=c++17", "-O2", "-Ihost/sim", "-Ifirmware", "-DCONFIG_NEEDLE_HEADLESS=0",
                       "-o", exe, "host/ui_preview.cpp", "firmware/ui/ui.cpp", "firmware/app/creature.cpp",
                       "firmware/app/state_text.cpp"], cwd=ROOT)
with tempfile.TemporaryDirectory() as d:
    subprocess.check_call([exe, d])
    for f in sorted(os.listdir(d)):
        if f.endswith(".ppm"):
            im = Image.open(os.path.join(d, f))
            im.resize((im.width * 2, im.height * 2), Image.NEAREST).save(os.path.join(OUT, f[:-4] + ".png"))
ims = [Image.open(os.path.join(OUT, f"{n}.png")) for n in SHEET]
W, H = ims[0].size
g = Image.new("RGB", (W * 3 + 40, H * 2 + 30), (30, 30, 30))
for i, im in enumerate(ims):
    g.paste(im, (10 + (i % 3) * (W + 10), 10 + (i // 3) * (H + 10)))
g.save(os.path.join(OUT, "creature_sheet.png"))
print("wrote", len(os.listdir(OUT)), "files in docs/screens")
