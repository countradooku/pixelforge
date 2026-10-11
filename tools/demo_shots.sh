#!/bin/bash
# Runs a PF_DEMO scenario and captures the window at every "SHOT name" line.
#   tools/demo_shots.sh tools out_dir [phases]
set -u
scen=$1; out=$2; mkdir -p "$out"
cd "$(dirname "$0")/.."
PF_DEMO=$scen PF_DEMO_PHASE=${3:-0,1,2,3,4,5} ./target/release/pixelforge 2>&1 | while IFS= read -r line; do
  echo "$line"
  case "$line" in
    "SHOT "*) name=${line#SHOT }; tools/shoot.sh "$out/$name.png" && echo "  -> $out/$name.png";;
  esac
done
