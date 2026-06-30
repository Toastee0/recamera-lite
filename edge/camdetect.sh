#!/bin/sh
# camdetect — probe the MIPI sensor on i2c bus 2 and write /userdata/sensor.conf,
# the shared sensor descriptor the camera tools (vision, S99 autostart) read so they
# don't need a hard-coded per-board flag. Run at boot before vision, or once to seed.
# Confirms by chip-ID (robust), falls back to address-ACK. Leaves an existing conf
# untouched if nothing answers (e.g. sensor not yet powered), and exits non-zero.
BUS=2
CONF=/userdata/sensor.conf
SENSOR= ; FLAG= ; ADDR=

# GC2053 @ 0x3f — chip-id reg 0xf0 = 0x20 (id 0x2053), 8-bit reg
if [ "$(i2cget -y $BUS 0x3f 0xf0 b 2>/dev/null)" = "0x20" ]; then
  SENSOR=GC2053; FLAG=-gc2053; ADDR=0x3f
# OV5647 @ 0x36 — chip-id reg 0x300a = 0x56, 16-bit reg (via i2ctransfer)
elif [ "$(i2ctransfer -y $BUS w2@0x36 0x30 0x0a r1 2>/dev/null)" = "0x56" ]; then
  SENSOR=OV5647; FLAG=-ov5647; ADDR=0x36
# fallback: bare address ACK at the OV5647 address
elif i2cget -y $BUS 0x36 0x00 b >/dev/null 2>&1; then
  SENSOR=OV5647; FLAG=-ov5647; ADDR=0x36
fi

if [ -n "$SENSOR" ]; then
  printf '# written by camdetect\nSENSOR=%s\nI2CADDR=%s\nFLAG=%s\n' "$SENSOR" "$ADDR" "$FLAG" > "$CONF"
  echo "camdetect: $SENSOR ($ADDR) -> $CONF"
else
  echo "camdetect: no sensor on i2c-$BUS (powered?); keeping existing $CONF" >&2
  [ -f "$CONF" ] && { echo "existing:"; cat "$CONF"; }
  exit 1
fi
