#!/bin/bash
# tools/photoshop_gaps.sh out_dir [phases]   e.g. phases "A,B"
# Runs tools/photoshop_gaps.jsx inside Photoshop 2026: prints its timings and
# writes PNG outputs into out_dir for the Rust-side quality comparisons.
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$1"; out=$(cd "$1" && pwd)
phases=${2:-A}
# Photoshop may not read ~/Desktop (macOS folder permissions): stage inputs in /tmp.
photos=/tmp/pf_gaps_photos; mkdir -p "$photos"; cp "$root"/testdata/photos/*.jpg "$photos"/
# The GPU phase times PixelForge's exported scene at several sizes.
case ",$phases," in *,GPU,*)
  for side in 2048 4096 8192 16384; do
    [ -f "$photos/scene_$side.tif" ] || "$root/target/release/pixelforge" --export-scene $side "$photos/scene_$side.tif"
  done
  [ -f "$photos/scene_2048.png" ] || sips -s format png "$photos/scene_2048.tif" --out "$photos/scene_2048.png" >/dev/null;;
esac
jsx=/tmp/pf_gaps.jsx; cp "$root/tools/photoshop_gaps.jsx" "$jsx"
osascript -e "with timeout of 3600 seconds" -e "tell application \"Adobe Photoshop 2026\" to do javascript \"var PHOTOS='$photos'; var OUT='$out'; var PHASES='$phases'; ${PF_SKIP_PROBES:+var PF_SKIP_PROBES=1; }\$.evalFile(new File('$jsx'));\"" -e "end timeout" | tee "$out/timings_$phases.txt"
