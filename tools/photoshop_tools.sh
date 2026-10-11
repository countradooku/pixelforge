#!/bin/bash
# tools/photoshop_tools.sh scene.tif [phases]   e.g. phases "1,2"
# Runs tools/photoshop_tools.jsx inside Photoshop 2026 and prints its timings.
scene=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
phases=${2:-1,2,3,4,5}
jsx=$(cd "$(dirname "$0")" && pwd)/photoshop_tools.jsx
osascript -e "tell application \"Adobe Photoshop 2026\" to do javascript \"var SCENE='$scene'; var PHASES='$phases'; \$.evalFile(new File('$jsx'));\""
