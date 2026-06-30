#!/bin/sh
# install.sh — deploy the recamera-lite edge daemon to a reCamera over SSH.
#
# Prereqs: you've built ./src/vision (see src/Makefile) and have an INT8 .cvimodel
# (see ./model-build). The device runs stock-ish reCamera-OS with SSH enabled.
#
# Usage:
#   ./install.sh <device-ip> <path/to/vision> <path/to/model.cvimodel> [ssh-user]
# Example:
#   ./install.sh 192.168.1.50 src/vision model-build/yolo11n_ours.cvimodel recamera
#
# Then edit /userdata/recamera-lite.conf on the device (at least SERVER=) and reboot,
# or run: ssh <user>@<ip> /etc/init.d/S99recamera-lite restart
set -e
IP="$1"; BIN="$2"; MODEL="$3"; USER="${4:-recamera}"
HERE=$(dirname "$0")
[ -n "$IP" ] && [ -f "$BIN" ] && [ -f "$MODEL" ] || {
  echo "usage: $0 <device-ip> <vision-binary> <model.cvimodel> [ssh-user]"; exit 1; }

SSH="ssh ${USER}@${IP}"
SCP="scp"

echo "[1/5] /userdata dirs"
$SSH 'mkdir -p /userdata/Models'

echo "[2/5] copy binary + scripts + model"
$SCP "$BIN"                       "${USER}@${IP}:/userdata/vision"
$SCP "$HERE/camdetect.sh"         "${USER}@${IP}:/userdata/camdetect.sh"
$SCP "$MODEL"                     "${USER}@${IP}:/userdata/Models/$(basename "$MODEL")"

echo "[3/5] config (kept if it already exists)"
$SSH 'test -f /userdata/recamera-lite.conf' 2>/dev/null \
  && echo "    /userdata/recamera-lite.conf exists — leaving it" \
  || $SCP "$HERE/recamera-lite.conf" "${USER}@${IP}:/userdata/recamera-lite.conf"

echo "[4/5] init script"
$SCP "$HERE/S99recamera-lite" "${USER}@${IP}:/tmp/S99recamera-lite"
$SSH 'chmod +x /userdata/vision /userdata/camdetect.sh /tmp/S99recamera-lite && \
      sudo mv /tmp/S99recamera-lite /etc/init.d/S99recamera-lite 2>/dev/null || \
      mv /tmp/S99recamera-lite /etc/init.d/S99recamera-lite'

echo "[5/5] done."
echo "    Edit the model path + SERVER= in /userdata/recamera-lite.conf, then:"
echo "    ssh ${USER}@${IP} /etc/init.d/S99recamera-lite restart"
