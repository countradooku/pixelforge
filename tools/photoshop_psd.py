#!/usr/bin/env python3
"""Local Photoshop PSD corpus and independent dialog watchdog (stdlib only).

Examples:
  python3 tools/photoshop_psd.py generate /tmp/pf-psd --list-cases
  python3 tools/photoshop_psd.py generate /tmp/pf-psd --case blend-normal-o100-f100
  python3 tools/photoshop_psd.py generate /tmp/pf-psd --large 24,61,150,268
  python3 tools/photoshop_psd.py reopen /tmp/pf-psd /tmp/pf-psd/roundtrip

Roundtrip files use each manifest case's filename. PixelForge's --psd-corpus
performs codec/structural checks; this runner independently opens the output in
Photoshop, captures rendering and layer metadata, and compares reference PNGs.
No result from a missing file, unsupported case, dialog, timeout, or failed
export is counted as a pass. Run serially with builds and other benchmarks.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
import zlib


BLENDS = (
    "normal dissolve darken multiply colorburn linearburn darkercolor lighten "
    "screen colordodge lineardodge lightercolor overlay softlight hardlight "
    "vividlight linearlight pinlight hardmix difference exclusion subtract "
    "divide hue saturation color luminosity"
).split()
ADJUSTMENTS = (
    "brightness levels curves exposure vibrance huesaturation colorbalance "
    "blackwhite photofilter channelmixer selectivecolor gradientmap invert "
    "posterize threshold colorlookup"
).split()
EFFECTS = (
    "dropShadow innerShadow outerGlow innerGlow bevelEmboss chromeFX "
    "solidFill gradientFill patternFill frameFX"
).split()
TIERS = {"24": (6000, 4000), "61": (9504, 6336),
         "150": (14204, 10652), "268": (16384, 16384)}


def cases(large: str = "") -> list[dict]:
    result = []

    def add(family, feature, variant="default", **params):
        ident = f"{family}-{feature}-{variant}".lower()
        result.append({"id": ident, "family": family, "feature": feature,
                       "variant": variant, "params": params,
                       "filename": ident + (".psb" if params.get("psb") else ".psd"),
                       "fidelity": "exact" if family in {
                           "pixels", "blend", "mask", "group", "layer"} else "perceptual"})

    for mode in BLENDS:
        for opacity, fill in [(100, 100), (37, 100), (100, 37), (37, 37)]:
            add("blend", mode, f"o{opacity}-f{fill}", opacity=opacity, fill=fill)
    for feature in ["opaque", "transparent", "partial-alpha", "negative-offset",
                    "offcanvas", "unicode-name", "psb", "empty-layer"]:
        add("pixels", feature, psb=feature == "psb")
    for feather in [0, 1, 5]:
        for inverse in [False, True]:
            add("mask", "user", f"feather{feather}-inverse{int(inverse)}",
                feather=feather, inverse=inverse)
    add("mask", "disabled")
    add("mask", "vector")
    add("mask", "user-and-vector")
    for kind in ["passthrough", "isolated", "nested"]:
        for opacity in [100, 43]:
            add("group", kind, f"o{opacity}", opacity=opacity)
    for feature in ["clipping", "hidden", "lock-all", "lock-position",
                    "lock-transparency", "color-tag", "blend-if", "channels",
                    "knockout-shallow", "knockout-deep"]:
        add("layer", feature)
    for adjustment in ADJUSTMENTS:
        for level in ([1] if adjustment == "invert" else [0, 1, 2]):
            add("adjustment", adjustment, f"setting{level}", level=level)
    for kind in ["solid", "gradient", "pattern"]:
        for level in range(3):
            add("fill", kind, f"setting{level}", level=level)
    for effect in EFFECTS:
        # Overlay/stroke effects have no contour parameter; exercise a real
        # blend setting instead of silently creating the same small effect.
        variants = ["small", "large", "contour" if effect in {
            'dropShadow', 'innerShadow', 'outerGlow', 'innerGlow', 'bevelEmboss', 'chromeFX'} else 'blend']
        if effect in {"dropShadow", "innerShadow", "solidFill", "gradientFill", "frameFX"}:
            variants.append("multiple")
        for variant in variants:
            add("style", effect, variant)
    for feature in ["point", "paragraph", "unicode", "vertical", "runs",
                    "paragraph-styles", "path", "area"]:
        add("text", feature)
    for warp in ["ARC", "ARCLOWER", "ARCUPPER", "ARCH", "BULGE", "SHELLLOWER",
                 "SHELLUPPER", "FLAG", "WAVE", "FISH", "RISE", "FISHEYE",
                 "INFLATE", "SQUEEZE", "TWIST"]:
        for bend in [-35, 45]:
            add("text", "warp", f"{warp}-{bend}", warp=warp, bend=bend)
    for geom in ["rectangle", "ellipse", "rounded", "polygon"]:
        for op in ["add", "subtract", "intersect", "xor"]:
            add("shape", geom, op, operation=op)
    for feature in ["stroke-solid", "stroke-dashed", "gradient-fill", "pattern-fill",
                    "paths-panel", "clipping-path"]:
        add("shape", feature)
    for feature in ["embedded-raster", "embedded-psd", "linked", "transform",
                    "warp", "smart-filter"]:
        add("smart", feature)
    for mode in ["rgb", "grayscale", "bitmap", "indexed", "duotone", "lab", "cmyk", "multichannel"]:
        add("mode", mode)
    for feature in ["alpha", "alpha-multiple", "spot", "spot-multiple"]:
        add("channel", feature)
    for feature in ["guides", "xmp-iptc", "icc-resolution", "layer-comps", "slices",
                    "notes", "count", "print", "artboard", "timeline"]:
        add("metadata", feature)
    for tier in filter(None, large.split(",")):
        if tier not in TIERS:
            raise ValueError(f"Unknown large tier: {tier}")
        width, height = TIERS[tier]
        add("large", "layered", tier, width=width, height=height, psb=tier in {"150", "268"})
    assert len({c["id"] for c in result}) == len(result)
    return result


WATCH = r'''ObjC.import('CoreGraphics'); ObjC.import('Foundation');
var ws=ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly|$.kCGWindowListExcludeDesktopElements,0)));
JSON.stringify(ws.filter(function(w){return (w.kCGWindowOwnerName||'').indexOf('Photoshop')>=0;}).map(function(w){return {id:w.kCGWindowNumber,name:w.kCGWindowName||'',layer:w.kCGWindowLayer,bounds:w.kCGWindowBounds};}));'''


def windows():
    proc = subprocess.run(["osascript", "-l", "JavaScript", "-e", WATCH],
                          text=True, capture_output=True, timeout=10, check=True)
    return json.loads(proc.stdout)


def dialog_windows(current, baseline):
    candidates = []
    for window in current:
        title = window["name"].lower()
        # Photoshop's named progress surface is noninteractive
        # save/export status, observed on the large corpus. It is not an error
        # or compatibility choice. Its observed layers are 0 and 8. Other
        # geometry/titles, including generic Photoshop errors, still fail.
        if (title == 'progress' and window['layer'] in (0, 8)
                and 466 <= window['bounds'].get('Width', 0) <= 474
                and 127 <= window['bounds'].get('Height', 0) <= 129):
            continue
        bounds = window["bounds"]
        # Verified with the generated two-pixel raw-composite PSD: the
        # document and its editable layers remain accessible and close normally
        # while this compact contextual control exists. It is not a modal.
        # Keep this exemption narrow; generic Photoshop alerts still fail.
        has_document = any(w['layer'] == 0 and any(ext in w['name'].lower()
                           for ext in ('.psd', '.psb')) for w in current)
        if (title == 'pixel layer bar' and window['layer'] == 8 and has_document
                and bounds.get('Height') == 32 and bounds.get('Width') in (509, 539)):
            continue
        known_dialog = any(word in title for word in (
            "camera raw", "warning", "error", "profile mismatch", "missing profile",
            "color profile", "could not", "format options", "compatibility"))
        new = window["id"] not in baseline
        compact = 220 < bounds["Width"] < 950 and 100 < bounds["Height"] < 650
        document = any(ext in title for ext in (".psd", ".psb", ".png", "pf-corpus", "pf-export"))
        if known_dialog or (new and (window["layer"] > 0 or (compact and not document))):
            candidates.append(window)
    return candidates



def wait_native_document(path, baseline, observations, started, timeout):
    """Wait without changing dialog preferences while a native Open is pending.

    A queued JSX inspector must never set DialogModes.NO before native Open has
    finished: that can suppress the very warning we are trying to measure.
    This probe reads document paths only. The independent window monitor remains
    active even when Photoshop blocks the probe behind a modal.
    """
    probe = ("(function(){var requested=new File(" + json.dumps(str(path)) +
             ").fsName;for(var i=0;i<app.documents.length;i++){try{if(app.documents[i].fullName.fsName===requested)return 'ready';}catch(e){}}return 'pending';}())")
    apple = ('tell application "Adobe Photoshop 2026" to do javascript ' + json.dumps(probe))
    while True:
        process = subprocess.Popen(['osascript', '-e', apple], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while True:
                current = windows()
                dialogs = dialog_windows(current, baseline)
                now = time.monotonic()
                observations.append({'seconds': round(now-started, 3), 'stage': 'native-open-readiness',
                                     'windows': current, 'dialogs': dialogs})
                if dialogs:
                    raise RuntimeError('Native Photoshop open displayed a dialog before inspection: ' + json.dumps(dialogs))
                if now-started > timeout:
                    raise TimeoutError('Native Photoshop open did not finish within watchdog limit')
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=10)
                    if process.returncode:
                        raise RuntimeError('Read-only native readiness probe failed: ' + stderr.strip())
                    if stdout.strip() == 'ready':
                        return
                    if stdout.strip() != 'pending':
                        raise RuntimeError('Unexpected native readiness response: ' + stdout.strip())
                    break
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()  # Only the read-only osascript probe.
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        time.sleep(0.5)


def run_photoshop(config, output, timeout):
    """Watch through a separate process even when Photoshop's bridge is blocked."""
    observations, seen = [], {}
    started = time.monotonic()
    baseline_windows = windows()
    baseline = {w["id"] for w in baseline_windows}
    # A pre-existing recognized modal is a failure too, never a baseline exemption.
    dialogs = dialog_windows(baseline_windows, baseline)
    if dialogs:
        (output / "watchdog.json").write_text(json.dumps({"status": "failed",
            "script_started": False, "restoration_confirmed": True,
            "error": "Pre-existing Photoshop dialog", "observations": [{"seconds": 0, "windows": baseline_windows, "dialogs": dialogs}]}, indent=2))
        raise RuntimeError("Photoshop already has a dialog; no script started: " + json.dumps(dialogs))
    outcome = {"status": "failed", "restoration_confirmed": False}
    with tempfile.TemporaryDirectory(prefix="pixelforge-psd-script-") as staging:
        script = pathlib.Path(staging) / "corpus.jsx"
        source_bytes = pathlib.Path(__file__).with_suffix('.jsx').read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if config.get('expected_jsx_sha256') and config['expected_jsx_sha256'] != source_sha256:
            raise RuntimeError('Photoshop JSX changed after the invocation was planned; no script started')
        script.write_bytes(source_bytes)
        (output / 'staged-script.jsx').write_bytes(source_bytes)
        (output / 'staged-invocation.json').write_text(json.dumps(config, indent=2))
        (output / 'script-provenance.json').write_text(json.dumps({'jsx_sha256': source_sha256,
            'jsx_copy': str(output / 'staged-script.jsx'),
            'python_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            'invocation_copy': str(output / 'staged-invocation.json'),
            'watchdog': 'Independent CoreGraphics query through a separate osascript process'}, indent=2))
        js = "var PF_CONFIG=" + json.dumps(config) + "; $.evalFile(new File(" + json.dumps(str(script)) + "));"
        apple = (f"with timeout of {timeout + 30} seconds\n"
                 'tell application "Adobe Photoshop 2026" to do javascript ' + json.dumps(js) + "\nend timeout")
        if config.get('native_file'):
            # LaunchServices sends an ordinary Open Documents event, outside
            # JSX/ActionManager dialog modes. The watchdog baseline is already
            # captured, so any native compatibility warning remains observable.
            launched = subprocess.run(['open', '-a', 'Adobe Photoshop 2026', config['native_file']],
                                      text=True, capture_output=True, timeout=15)
            if launched.returncode:
                raise RuntimeError('Native Photoshop open failed: ' + launched.stderr)
            try:
                wait_native_document(config['native_file'], baseline, observations, started, timeout)
            except BaseException as error:
                # Inspection has not started; no preferences were changed and
                # no document is closed here. Leave any warning available for
                # explicit inspection. Never queue the mutating JSX behind it.
                outcome.update(error=str(error), inspection_started=False,
                               restoration_confirmed=True,
                               seconds=round(time.monotonic()-started, 3), observations=observations)
                (output / 'watchdog.json').write_text(json.dumps(outcome, indent=2))
                raise
        process = subprocess.Popen(["osascript", "-e", apple], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while process.poll() is None:
                current = windows()
                dialogs = dialog_windows(current, baseline)
                now = time.monotonic()
                observations.append({"seconds": round(now - started, 3),
                                     "windows": current, "dialogs": dialogs})
                seen = {w["id"]: seen.get(w["id"], now) for w in dialogs}
                persistent = [w for w in dialogs if now - seen[w["id"]] >= 2]
                if persistent:
                    raise RuntimeError("Photoshop dialog detected; only the script bridge is stopped: " + json.dumps(persistent))
                if now - started > timeout:
                    raise TimeoutError(f"Photoshop script exceeded {timeout}s watchdog limit")
                time.sleep(0.5)
            stdout, stderr = process.communicate(timeout=10)
            (output / "photoshop-stdout.log").write_text(stdout)
            (output / "photoshop-stderr.log").write_text(stderr)
            outcome["bridge_exit_code"] = process.returncode
            if process.returncode:
                raise RuntimeError("Photoshop scripting failed: " + stderr.strip())
            result_file = output / "photoshop-results.json"
            if not result_file.is_file():
                raise RuntimeError("Photoshop produced no completion marker")
            result = json.loads(result_file.read_text())
            outcome["restoration_confirmed"] = result.get("restoration_confirmed", False)
            if not result.get("restoration_confirmed"):
                raise RuntimeError("Photoshop did not confirm restoration of preferences and active document")
            if result.get("fatal_error"):
                raise RuntimeError(result["fatal_error"])
            # Even transient detected dialogs invalidate warning-free certification.
            if any(x["dialogs"] for x in observations):
                raise RuntimeError("A transient Photoshop dialog was observed; warning-free gate failed")
            outcome["status"] = "passed"
            return result
        except BaseException as exc:
            outcome["error"] = str(exc)
            # A blocked script can continue after the user dismisses its dialog.
            # Ask it to stop at the next safe boundary instead of starting more
            # cases. This never dismisses a dialog or controls Photoshop itself.
            pathlib.Path(config["abort_file"]).write_text(str(exc))
            raise
        finally:
            # Never kill Photoshop, dismiss a dialog, or close a user's document.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()  # osascript only; Photoshop remains untouched.
                    process.wait(timeout=3)
            outcome["seconds"] = round(time.monotonic() - started, 3)
            outcome["observations"] = observations
            (output / "watchdog.json").write_text(json.dumps(outcome, indent=2))


def native_verification(config, output, timeout, negative=False):
    """Native open plus JSX inspection, preserving the pre-open active document."""
    snapshot_dir = output / 'snapshot'
    snapshot_dir.mkdir()
    snapshot_config = {**config, 'operation': 'snapshot', 'cases': [],
                       'output': str(snapshot_dir), 'abort_file': str(snapshot_dir / 'abort-requested.txt')}
    snapshot = run_photoshop(snapshot_config, snapshot_dir, min(timeout, 60))['snapshot']
    previous = snapshot['active_document_id']
    already_open = {d['path'] for d in snapshot['documents'] if d.get('path')}
    combined = {'cases': [], 'restoration_confirmed': True, 'warning_gate_verified': True,
                'open_dialog_mode': 'native LaunchServices', 'fatal_error': None}
    for case in config['cases']:
        case_dir = output / case['id']
        case_dir.mkdir()
        path = pathlib.Path(config['input']) / case['filename']
        record = {'id': case['id'], 'family': case['family'], 'status': 'failed'}
        if not path.is_file():
            record['error'] = 'Missing roundtrip file'
            combined['warning_gate_verified'] = False
            combined['cases'].append(record)
            continue
        if str(path.resolve()) in already_open:
            record['error'] = 'Requested file was already open before verification; existing document left untouched'
            combined['warning_gate_verified'] = False
            combined['cases'].append(record)
            continue
        native_config = {**config, 'operation': 'inspect_open', 'cases': [case],
                         'output': str(case_dir), 'native_file': str(path.resolve()),
                         'previous_document_id': previous, 'abort_file': str(case_dir / 'abort-requested.txt')}
        try:
            result = run_photoshop(native_config, case_dir, timeout)
            record = result['cases'][0]
            record['restoration_confirmed'] = result.get('restoration_confirmed') is True
            record['warning_gate_verified'] = result.get('warning_gate_verified') is True and record.get('native_open') is True
            record['watchdog_passed'] = json.loads((case_dir / 'watchdog.json').read_text()).get('status') == 'passed'
            combined['restoration_confirmed'] &= result.get('restoration_confirmed', False)
            combined['warning_gate_verified'] &= record['warning_gate_verified'] and record['watchdog_passed']
            if negative:
                record['rejection_confirmed'] = record.get('status') != 'generated'
                record['watchdog_dialog_confirmed'] = False
                record['status'] = 'control-failed'
                record['error'] = 'Malformed input was rejected without an observed native dialog; watchdog control is not proven'
        except Exception as error:
            record['error'] = str(error)
            watchdog_path = case_dir / 'watchdog.json'
            watchdog = json.loads(watchdog_path.read_text()) if watchdog_path.is_file() else {}
            observed = any(row.get('dialogs') for row in watchdog.get('observations', []))
            record['watchdog_dialog_confirmed'] = observed
            record['watchdog_passed'] = False
            record['warning_gate_verified'] = False
            record['restoration_confirmed'] = watchdog.get('restoration_confirmed') is True
            if negative and observed:
                record['status'] = 'expected-dialog-detected'
                record['rejection_confirmed'] = True
                record['cleanup_required'] = 'Only the task-owned malformed-file dialog remains; no automatic dismissal was performed'
            combined['warning_gate_verified'] = False
            combined['restoration_confirmed'] = watchdog.get('restoration_confirmed', False)
            combined['cases'].append(record)
            # A modal may still be present. Never queue another file behind it.
            break
        combined['cases'].append(record)
        (output / 'partial-results.json').write_text(json.dumps(combined, indent=2))
    if len(combined['cases']) != len(config['cases']):
        combined['fatal_error'] = 'Native verification stopped before all planned cases'
    (output / 'photoshop-results.json').write_text(json.dumps(combined, indent=2))
    (output / 'watchdog.json').write_text(json.dumps({
        'status': 'passed' if combined['warning_gate_verified'] and combined['restoration_confirmed'] else 'failed',
        'mode': 'native LaunchServices', 'per_case_evidence': [c['id'] + '/watchdog.json' for c in combined['cases']],
        'restoration_confirmed': combined['restoration_confirmed']}, indent=2))
    return combined


def png_rgba(path):
    """Decode noninterlaced 8-bit reference PNGs with no image dependency."""
    raw = pathlib.Path(path).read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not PNG")
    pos, compressed, palette, transparency = 8, bytearray(), None, b""
    width = height = depth = color = interlace = None
    while pos < len(raw):
        if pos + 12 > len(raw):
            raise ValueError("Truncated PNG chunk")
        size = struct.unpack_from(">I", raw, pos)[0]
        tag, data = raw[pos + 4:pos + 8], raw[pos + 8:pos + 8 + size]
        if len(data) != size or pos + size + 12 > len(raw):
            raise ValueError("Truncated PNG payload")
        if zlib.crc32(tag + data) & 0xffffffff != struct.unpack_from(">I", raw, pos + 8 + size)[0]:
            raise ValueError("PNG CRC mismatch")
        pos += size + 12
        if tag == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            if compression or filtering:
                raise ValueError("Unsupported PNG method")
        elif tag == b"IDAT":
            compressed.extend(data)
        elif tag == b"PLTE":
            palette = [data[i:i + 3] for i in range(0, len(data), 3)]
        elif tag == b"tRNS":
            transparency = data
        elif tag == b"IEND":
            break
    if depth != 8 or interlace or not width or not height:
        raise ValueError("Reference PNG must be noninterlaced 8-bit")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    if not channels:
        raise ValueError("Unsupported PNG colour type")
    stride = width * channels
    expected = (stride + 1) * height
    unpacker = zlib.decompressobj()
    decoded = unpacker.decompress(compressed, expected + 1)
    if len(decoded) != expected or not unpacker.eof or unpacker.unused_data:
        raise ValueError("Invalid PNG decompressed size")
    previous = bytearray(stride)
    rgba = bytearray(width * height * 4)
    out = 0
    for y in range(height):
        offset = y * (stride + 1)
        filt, row = decoded[offset], bytearray(decoded[offset + 1:offset + stride + 1])
        if filt > 4:
            raise ValueError("Invalid PNG filter")
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = previous[x]
            upper_left = previous[x - channels] if x >= channels else 0
            prediction = 0
            if filt == 1:
                prediction = left
            elif filt == 2:
                prediction = up
            elif filt == 3:
                prediction = (left + up) // 2
            elif filt == 4:
                estimate = left + up - upper_left
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - upper_left))
                prediction = (left, up, upper_left)[distances.index(min(distances))]
            row[x] = (row[x] + prediction) & 255
        for x in range(0, stride, channels):
            if color == 6:
                pixel = row[x:x + 4]
            elif color == 2:
                pixel = row[x:x + 3] + b"\xff"
            elif color == 0:
                pixel = bytes([row[x], row[x], row[x], 255])
            elif color == 4:
                pixel = bytes([row[x], row[x], row[x], row[x + 1]])
            else:
                index = row[x]
                if palette is None or index >= len(palette):
                    raise ValueError("Invalid PNG palette index")
                pixel = palette[index] + bytes([transparency[index] if index < len(transparency) else 255])
            rgba[out:out + 4] = pixel
            out += 4
        previous = row
    return width, height, rgba


def pixel_diff(first, second):
    # Most valid lossless roundtrips export identical PNG streams. Avoid
    # allocating full image buffers just to reconfirm a large fixture's hash.
    with pathlib.Path(first).open("rb") as one, pathlib.Path(second).open("rb") as two:
        identical = hashlib.file_digest(one, "sha256").digest() == hashlib.file_digest(two, "sha256").digest()
        if identical:
            one.seek(0)
            header = one.read(33)
            if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise ValueError("Invalid reference PNG header")
            width, height = struct.unpack_from(">II", header, 16)
            return {"exact": True, "different_pixels": 0, "total_pixels": width * height,
                    "max_channel_error": 0, "identical_png_sha256": True}
    a, b = png_rgba(first), png_rgba(second)
    if a[:2] != b[:2]:
        return {"exact": False, "error": "Dimensions differ", "a": a[:2], "b": b[:2]}
    different = sum(a[2][i:i + 4] != b[2][i:i + 4] for i in range(0, len(a[2]), 4))
    return {"exact": different == 0, "different_pixels": different,
            "total_pixels": a[0] * a[1],
            "max_channel_error": max((abs(x - y) for x, y in zip(a[2], b[2])), default=0)}


def compare_evidence(corpus, reference, selected, results):
    by_id = {r["id"]: r for r in results.get("cases", [])}
    generation = json.loads((corpus / 'photoshop-results.json').read_text(encoding='utf-8-sig'))
    originals = {r['id']: r for r in generation.get('cases', [])}
    checks = []
    for case in selected:
        record = {"id": case["id"], "family": case["family"], "status": "failed"}
        try:
            result = by_id.get(case["id"], {})
            if result.get("status") != "generated":
                raise ValueError(result.get("error", "Photoshop did not reopen this case"))
            original = pathlib.Path(originals.get(case['id'], {}).get('reference_path', corpus / "reference" / case["id"]))
            reopened = reference / case["id"]
            record["composite"] = pixel_diff(original / "composite.png", reopened / "composite.png")
            first = json.loads((original / "tree.json").read_text())
            second = json.loads((reopened / "tree.json").read_text())
            # Layer IDs are intentionally excluded by the JSX semantic tree.
            record["tree_equal"] = first == second
            record["layers"] = []
            for png in sorted((original / "layers").glob("*.png")):
                target = reopened / "layers" / png.name
                check = {"file": png.name, **pixel_diff(png, target)}
                record["layers"].append(check)
            if not record["layers"]:
                raise ValueError("No per-layer pixels were exported")
            if record["composite"]["exact"] and record["tree_equal"] and all(x["exact"] for x in record["layers"]):
                record["status"] = "passed"
            else:
                record["error"] = "Photoshop round-trip pixels or semantic layer tree differ"
        except (OSError, ValueError, KeyError, zlib.error) as exc:
            record["error"] = str(exc)
        checks.append(record)
    return checks


def coverage_failures(corpus, generation):
    """Apply immutable evidence-specific audits, never flags to a repaired retry."""
    path = corpus / 'coverage-audit.json'
    if not path.is_file():
        return []
    audit = json.loads(path.read_text(encoding='utf-8-sig'))
    records = {record['id']: record for record in generation.get('cases', [])}
    failures = []
    for finding in audit['failures']:
        record = records.get(finding['id'], {})
        source = pathlib.Path(record.get('source_path', corpus / 'originals' / finding['filename']))
        if record.get('status') == 'generated' and source.is_file():
            with source.open('rb') as stream:
                fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
            if fingerprint == finding['source_sha256']:
                failures.append(finding)
    return failures


def merge_retry(corpus, output):
    """Reconcile only completed task evidence; never rerun Photoshop or rewrite logs."""
    manifest = json.loads((corpus / 'corpus-manifest.json').read_text(encoding='utf-8-sig'))
    retry_manifest = json.loads((output / 'corpus-manifest.json').read_text(encoding='utf-8-sig'))
    invocation = json.loads((output / 'invocation.json').read_text(encoding='utf-8-sig'))
    results = json.loads((output / 'photoshop-results.json').read_text(encoding='utf-8-sig'))
    if invocation.get('operation') != 'generate' or output == corpus:
        raise ValueError('Only a separate generation retry can be reconciled')
    if results.get('restoration_confirmed') is not True or results.get('cleanup_errors'):
        raise ValueError('Retry completion must confirm Photoshop restoration with no cleanup errors')
    planned = manifest['cases']
    definitions = {case['id']: case for case in planned}
    selected = {case['id']: case for case in retry_manifest['cases']}
    if any(definitions.get(ident) != case for ident, case in selected.items()):
        raise ValueError('Retry case definitions differ from the full corpus manifest')
    prior = json.loads((corpus / 'photoshop-results.json').read_text(encoding='utf-8-sig'))
    if str(output) in prior.get('retry_runs', []):
        return prior
    if not (output / 'prior-results.json').exists():
        (output / 'prior-results.json').write_text(json.dumps(prior, indent=2))
    if (corpus / 'report.json').is_file() and not (output / 'prior-report.json').exists():
        (output / 'prior-report.json').write_text((corpus / 'report.json').read_text())
    merged = {record['id']: record for record in prior.get('cases', [])}
    for record in results['cases']:
        record = dict(record)
        ident = record['id']
        if ident not in selected:
            raise ValueError('Retry result has an unplanned case: ' + ident)
        record['retry_evidence'] = str(output)
        if record.get('status') == 'generated':
            source = output / 'originals' / selected[ident]['filename']
            reference = output / 'reference' / ident
            if not source.is_file() or not (reference / 'tree.json').is_file() or not (reference / 'composite.png').is_file():
                raise ValueError('Generated retry is missing evidence: ' + ident)
            record.update(source_path=str(source), reference_path=str(reference),
                          generator_sha256=retry_manifest['generator_sha256'])
        merged[ident] = record
    combined = dict(prior)
    combined['cases'] = [merged.get(case['id'], {'id': case['id'], 'status': 'missing'}) for case in planned]
    combined.setdefault('retry_runs', []).append(str(output))
    combined['restoration_confirmed'] = prior.get('restoration_confirmed') is True and results.get('restoration_confirmed') is True
    combined['fatal_error'] = results.get('fatal_error')
    combined.setdefault('prior_run_failures', []).append({
        'fatal_error': prior.get('fatal_error'), 'evidence': str(output / 'prior-results.json')})
    temporary = corpus / 'photoshop-results.pending.json'
    temporary.write_text(json.dumps(combined, indent=2))
    temporary.replace(corpus / 'photoshop-results.json')
    failed = sum(record.get('status') != 'generated' for record in combined['cases'])
    (corpus / 'report.json').write_text(json.dumps({'operation': 'generate', 'count': len(planned),
        'failed': failed, 'status': 'failed' if failed or combined['fatal_error'] else 'passed',
        'coverage_complete': manifest['coverage_complete'], 'warning_gate_verified': False,
        'retry_runs': combined['retry_runs'], 'cases': combined['cases']}, indent=2))
    (output / 'reconciliation.json').write_text(json.dumps({'aggregate': str(corpus / 'photoshop-results.json'),
        'restoration_confirmed': True, 'historical_watchdog_unchanged': True,
        'generated': sum(r.get('status') == 'generated' for r in results['cases']),
        'remaining_generation_failures': failed}, indent=2))
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("operation", choices=["generate", "reopen", "native-verify", "native-control"])
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("roundtrip", type=pathlib.Path, nargs="?")
    parser.add_argument("--families", default="all", help="Comma-separated family names")
    parser.add_argument("--case", action="append", default=[], help="Exact case ID; repeat for multiple")
    parser.add_argument("--large", default="", help="Optional large tiers: 24,61,150,268")
    parser.add_argument("--limit", type=int, help="Smoke-test limit, explicitly recorded as incomplete coverage")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--list-cases", action="store_true", help="Print planned cases without Photoshop or file writes")
    parser.add_argument("--output", type=pathlib.Path, help="Evidence directory for this invocation")
    parser.add_argument("--reference-output", type=pathlib.Path, help="Isolated reopened images/metadata directory")
    parser.add_argument('--input-file', type=pathlib.Path, help='Native-verify one standalone PSD/PSB in isolation; never counts as full corpus coverage')
    parser.add_argument('--pause-before-export', type=float, metavar='SECONDS', help='Diagnostic native-verify --input-file only: pause 0..30 seconds after refreshing the opened document; the strict watchdog remains active')
    parser.add_argument('--resave-output', type=pathlib.Path, help='Native-verify: save a fresh Photoshop copy here for independent private-resource retention checks')
    parser.add_argument("--defer-fidelity", action="store_true", help="Export reopening evidence; the invoking corpus runner applies its documented exact/perceptual gates")
    parser.add_argument("--control", choices=['truncated', 'corrupt'], default='truncated')
    parser.add_argument('--retry-failed', action='store_true', help='Generate only failed cases in a fresh retry directory; retain the original full manifest and all prior evidence')
    parser.add_argument('--reconcile-retry', type=pathlib.Path, help='Offline: aggregate a completed interrupted retry, preserving its failed watchdog and every prior log')
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    if args.input_file and args.operation != 'native-verify':
        parser.error('--input-file requires native-verify')
    if args.resave_output and args.operation != 'native-verify':
        parser.error('--resave-output requires native-verify')
    if args.pause_before_export is not None:
        if args.operation != 'native-verify' or not args.input_file:
            parser.error('--pause-before-export requires native-verify with one --input-file')
        if not 0 <= args.pause_before_export <= 30:
            parser.error('--pause-before-export must be between 0 and 30 seconds')
    if args.reconcile_retry:
        if args.operation != 'generate' or args.retry_failed:
            parser.error('--reconcile-retry requires generate and cannot be combined with --retry-failed')
        try:
            combined = merge_retry(corpus, args.reconcile_retry.resolve())
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        print(json.dumps({'reconciled': str(args.reconcile_retry.resolve()),
            'count': len(combined['cases']), 'generated': sum(r.get('status') == 'generated' for r in combined['cases']),
            'restoration_confirmed': combined['restoration_confirmed'], 'fatal_error': combined['fatal_error']}, indent=2))
        return 0
    planned = cases(args.large)
    prior_generation = None
    if args.retry_failed:
        if args.operation != 'generate':
            parser.error('--retry-failed applies only to generate')
        planned = json.loads((corpus / 'corpus-manifest.json').read_text(encoding='utf-8-sig'))['cases']
        if not (corpus / 'photoshop-results.json').is_file():
            parser.error('The original generation must finish before failed cases can be retried')
        prior_generation = json.loads((corpus / 'photoshop-results.json').read_text(encoding='utf-8-sig'))
        failures = {r['id'] for r in prior_generation.get('cases', []) if r.get('status') != 'generated'}
        failures.update(row['id'] for row in coverage_failures(corpus, prior_generation))
        failures.update(c['id'] for c in planned if c['id'] not in {r['id'] for r in prior_generation.get('cases', [])})
    if args.input_file:
        args.input_file = args.input_file.resolve()
        if not args.input_file.is_file() or args.input_file.suffix.lower() not in {'.psd', '.psb'}:
            parser.error('--input-file must name an existing PSD or PSB')
        if not args.output or not args.reference_output or not args.defer_fidelity:
            parser.error('Standalone input requires --output, --reference-output, and --defer-fidelity; no unrelated corpus reference can certify its fidelity')
        args.roundtrip = args.input_file.parent
        planned = [{'id': 'standalone-' + args.input_file.stem, 'filename': args.input_file.name,
                    'family': 'standalone-writer', 'feature': 'native-open-and-resave',
                    'params': {'psb': args.input_file.suffix.lower() == '.psb'}, 'fidelity': 'exact'}]
    elif args.operation in {"reopen", "native-verify"}:
        if not args.roundtrip:
            parser.error("reopen requires ROUNDTRIP_DIR")
        planned = json.loads((corpus / "corpus-manifest.json").read_text())["cases"]
    elif args.operation == 'native-control':
        planned = [{'id': 'negative-' + args.control, 'filename': 'negative-' + args.control + '.psd',
                    'family': 'negative-control', 'feature': args.control, 'params': {}, 'fidelity': 'exact'}]
    selected = [c for c in planned if (args.families == "all" or c["family"] in args.families.split(","))
                and (not args.case or c["id"] in args.case) and (not args.retry_failed or c['id'] in failures)]
    if args.case and set(args.case) - {c["id"] for c in selected}:
        parser.error("Unknown or filtered case ID")
    if args.limit is not None:
        selected = selected[:max(args.limit, 0)]
    if args.list_cases:
        print(json.dumps({"count": len(selected), "families": dict(collections.Counter(c["family"] for c in selected)), "cases": selected}, indent=2))
        return 0
    if not selected:
        parser.error("Empty corpus is not a compatibility proof")
    if args.timeout < 1:
        parser.error("timeout must be positive")
    if args.retry_failed and not args.output:
        retries = corpus / 'retries'
        retries.mkdir(exist_ok=True)
        output = pathlib.Path(tempfile.mkdtemp(prefix='retry-', dir=retries))
    else:
        output = (args.output or (corpus if args.operation == "generate" else corpus / ('photoshop-' + args.operation))).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; choose a fresh --output to retain previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    reference_path = (args.reference_output.resolve() if args.reference_output else
                      corpus / 'reopened' if args.operation in {'reopen', 'native-verify'} else output / 'reference')
    if reference_path.exists() and any(reference_path.iterdir()):
        parser.error("Reference output is not empty; use a fresh --reference-output to retain previous evidence")
    if args.resave_output:
        args.resave_output = args.resave_output.resolve()
        if args.resave_output.exists() and any(args.resave_output.iterdir()):
            parser.error('--resave-output must be new or empty; existing copies are never overwritten')
        args.resave_output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "operation": args.operation, "cases": selected,
                "planned_count": len(planned), "selected_count": len(selected),
                "coverage_complete": not args.input_file and len(selected) == len(planned),
                "standalone_validation": bool(args.input_file),
                "note": "Generated feature coverage is measured per distinct feature and parameter; failures remain failures.",
                "generator_sha256": hashlib.sha256(pathlib.Path(__file__).with_suffix(".jsx").read_bytes()).hexdigest()}
    (output / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2))
    config = {"operation": args.operation, "output": str(output), "cases": selected,
              "input": str(args.roundtrip.resolve()) if args.roundtrip else "",
              "reference": str(reference_path),
              "abort_file": str(output / "abort-requested.txt"),
              'expected_jsx_sha256': manifest['generator_sha256']}
    if args.resave_output:
        config['resave_output'] = str(args.resave_output)
    if args.pause_before_export is not None:
        config['pause_before_export_seconds'] = args.pause_before_export
    (output / "invocation.json").write_text(json.dumps(config, indent=2))
    try:
        if args.operation == 'native-control':
            control_dir = output / 'control-input'
            control_dir.mkdir()
            header = b'8BPS' + struct.pack('>H', 1) + bytes(6) + struct.pack('>HIIHH', 3, 1, 1, 8, 3)
            (control_dir / selected[0]['filename']).write_bytes(header if args.control == 'truncated' else b'BROKEN' + header[6:])
            config['input'] = str(control_dir)
        results = native_verification(config, output, args.timeout, args.operation == 'native-control') if args.operation.startswith('native-') else run_photoshop(config, output, args.timeout)
        compare = args.operation in {"reopen", "native-verify"} and not args.defer_fidelity
        checks = compare_evidence(corpus, pathlib.Path(config["reference"]), selected, results) if compare else results["cases"]
        good = "passed" if compare else "generated"
        failed = [c for c in checks if c.get("status") != good]
        report = {"operation": args.operation, "count": len(checks), "failed": len(failed),
                  "status": "failed" if failed else "passed", "coverage_complete": manifest["coverage_complete"],
                  "fidelity_deferred": args.defer_fidelity,
                  "warning_gate_verified": results.get('warning_gate_verified', False),
                  "cases": checks}
        if args.operation == 'native-control':
            report['expected_negative_control'] = True
            report['control'] = args.control
            report['watchdog_proven'] = any(row.get('watchdog_dialog_confirmed') for row in checks)
            report['status'] = 'expected-dialog-detected' if report['watchdog_proven'] else 'failed'
            report['expected_rejections'] = sum(row.get('status') == 'expected-dialog-detected' for row in checks)
            report['failed'] = 0 if report['watchdog_proven'] else len(checks)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        if args.retry_failed:
            merge_retry(corpus, output)
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
        return 0 if args.operation == 'native-control' and report.get('watchdog_proven') else 1 if failed else 0
    except Exception as exc:
        (output / "report.json").write_text(json.dumps({"status": "failed", "error": str(exc),
            "restoration": "Check watchdog.json. A blocked bridge cannot confirm restoration; Photoshop was not killed and dialogs were not dismissed."}, indent=2))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
