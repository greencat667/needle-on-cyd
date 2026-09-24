#!/bin/bash
# Build the firmware, flash the app only, reset and run serial commands.
#   tools/cyd_flash_run.sh <log> <seconds> <until-text> [command...]
set -e
cd "$(dirname "$0")/.."
source ~/esp/esp-idf/export.sh >/dev/null 2>&1
PORT=${PORT:-/dev/cu.usbserial-130}
idf.py build > logs/idf_build.log 2>&1 || { grep -E "error" logs/idf_build.log | head; exit 1; }
python -m esptool --chip esp32 --port $PORT -b 460800 write_flash 0x10000 build/needle_cyd.bin 2>&1 | grep -E "Hash|rror"
LOG=$1; SECS=$2; UNTIL=$3; shift 3
ARGS=()
for c in "$@"; do ARGS+=(--send "$c"); done
.venv/bin/python tools/serial_log.py $PORT --seconds $SECS --until "$UNTIL" --after 20 "${ARGS[@]}" > "$LOG" 2>&1 || true
