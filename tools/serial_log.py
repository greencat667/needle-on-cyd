#!/usr/bin/env python3
"""Reset the board and log its serial output; optionally send commands.

    python tools/serial_log.py /dev/cu.usbserial-130 --seconds 60 [--send "bench"] [--after 20]
"""
import argparse, sys, time
import serial

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--send", action="append", default=[])
ap.add_argument("--after", type=float, default=0, help="send once this much time has passed")
ap.add_argument("--until", default=None, help="stop early when this text appears")
ap.add_argument("--no-reset", action="store_true")
a = ap.parse_args()
s = serial.Serial(a.port, 115200, timeout=0.2)
if not a.no_reset:
    s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False
t0 = time.time(); sent = False; buf = b""
while time.time() - t0 < a.seconds:
    d = s.read(4096)
    if d:
        sys.stdout.write(d.decode("utf-8", "replace")); sys.stdout.flush(); buf += d
        if a.until and a.until.encode() in buf[-8192:]:
            break
    if not sent and a.send and time.time() - t0 >= a.after:
        for c in a.send:
            s.write((c + "\n").encode()); time.sleep(0.3)
        sent = True
