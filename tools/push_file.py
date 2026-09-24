#!/usr/bin/env python3
"""Send a file to the CYD's SD card over the serial console (no card swap).

    python tools/push_file.py /dev/cu.usbserial-1130 models/x.cact /sdcard/needle/needle3.cact
The firmware must be at its main screen (it answers `recv`). Every 512-byte
chunk is acknowledged with a running CRC32, checked here.
"""
import sys, time, zlib
import serial

port, src, dst = sys.argv[1:4]
baud = int(sys.argv[4]) if len(sys.argv) > 4 else 460800   # transfer rate (console stays 115200)
data = open(src, "rb").read()
s = serial.Serial(port, 115200, timeout=5)
s.reset_input_buffer()
s.write(f"recv {dst} {len(data)} {baud}\n".encode())


def wait(prefix):
    while True:
        line = s.readline().decode("utf-8", "replace").strip()
        if not line:
            raise SystemExit(f"no reply (waiting for {prefix})")
        if line.startswith(prefix) or line.startswith("RECV-ERR"):
            return line


line = wait("RECV-READY")
if line.startswith("RECV-ERR"):
    raise SystemExit(line)
time.sleep(0.2)
s.baudrate = baud
crc, t0 = 0, time.time()
for off in range(0, len(data), 512):
    chunk = data[off:off + 512]
    s.write(chunk)
    crc = zlib.crc32(chunk, crc)
    ack = wait("ACK")
    if ack.startswith("RECV-ERR"):
        raise SystemExit(ack)
    n, c = ack.split()[1:3]
    if int(n) != off + len(chunk) or int(c, 16) != crc:
        raise SystemExit(f"mismatch at {off}: {ack} (expected crc {crc:08x})")
    if off % (512 * 256) == 0:
        el = time.time() - t0
        print(f"\r{off / 1e6:6.2f} / {len(data) / 1e6:.2f} MB  {off / max(el, 1e-3) / 1e3:5.1f} KB/s", end="", flush=True)
print()
s.baudrate = 115200
print(wait("RECV-DONE"))
