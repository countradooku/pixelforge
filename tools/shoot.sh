#!/bin/bash
# Captures the PixelForge window (by window ID, so overlapping windows don't
# matter): tools/shoot.sh out.png
# macOS stops redrawing occluded windows: bring PixelForge forward first.
osascript -e 'tell application "System Events" to set frontmost of (first process whose name contains "pixelforge") to true' >/dev/null 2>&1
sleep 0.6
id=$(osascript -l JavaScript "$(dirname "$0")/winid.js" pixelforge)
[ -z "$id" ] && { echo "no PixelForge window" >&2; exit 1; }
screencapture -x -o -l"$id" "$1"
