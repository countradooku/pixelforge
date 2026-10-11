#!/usr/bin/env python3
"""Serial PSD acceptance runner. Missing evidence is a failure, never a pass.

Invoked by pixelforge --psd-corpus. Photoshop is accessed only through the
independently watched photoshop_psd.py runner. All fixtures are locally made.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from photoshop_psd import coverage_failures


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(argv, log, env, timeout=600):
    started = time.monotonic()
    timed = sys.platform == 'darwin' and Path('/usr/bin/time').is_file()
    command = ['/usr/bin/time', '-l', *argv] if timed else argv
    with log.open('w') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = 'timeout'
            # Stop only this invocation's process group. Photoshop itself is
            # launched by macOS, outside it, and is never terminated here.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    result = {'exit_code': code, 'seconds': time.monotonic() - started,
              'log': str(log), 'peak_rss_bytes': None,
              'rss_source': 'macOS /usr/bin/time -l' if timed else 'unavailable',
              'rss_scope': 'Timed process and children; the separately running Photoshop application is excluded'}
    if timed:
        statistics = log.read_text(errors='replace')
        for label, key in [('maximum resident set size', 'peak_rss_bytes'),
                           ('peak memory footprint', 'peak_memory_footprint_bytes'),
                           ('page reclaims', 'page_reclaims'), ('page faults', 'page_faults'),
                           ('swaps', 'process_swaps')]:
            values = re.findall(r'^\s*(\d+)\s+' + re.escape(label) + r'\s*$', statistics, re.MULTILINE)
            if values:
                result[key] = int(values[-1])
        result['paging_note'] = 'Page reclaims/faults are reported separately; neither is treated as swap. process_swaps is the time utility counter, not system swap bytes.'
    return result


def source_fingerprints(project):
    paths = sorted((project / 'src').rglob('*.rs'))
    paths += [project / name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs',
              'tools/photoshop_psd.py', 'tools/photoshop_psd.jsx', 'tools/psd_corpus.py') if (project / name).is_file()]
    return {str(path.relative_to(project)): digest(path) for path in paths}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def watchdog_accepted(watchdog):
    if watchdog.get('status') == 'passed':
        return True, 'passed'
    dialogs = [window for row in watchdog.get('observations', []) for window in row.get('dialogs', [])]
    progress_only = bool(dialogs) and all(w.get('name', '').lower() == 'progress' and w.get('layer') in (0, 8)
        and 466 <= w.get('bounds', {}).get('Width', 0) <= 474
        and 127 <= w.get('bounds', {}).get('Height', 0) <= 129 for w in dialogs)
    if progress_only and str(watchdog.get('error', '')).startswith(('Photoshop dialog detected', 'A transient Photoshop dialog was observed')):
        return True, 'Retained historical false positive: only the independently observed Writing PNG Format Progress animation (exact title Progress, layer 0 or 8, width 466..474 and height 127..129) was classified as a dialog; native opening remains independently required'
    return False, 'failed'


def fidelity_pass(metrics, kind):
    if kind == 'exact':
        return metrics.get('exact') is True
    if kind != 'perceptual':
        return False
    ssim, delta = metrics.get('ssim'), metrics.get('mean_delta_e00')
    return (metrics.get('alpha_exact') is True and isinstance(ssim, (int, float))
            and isinstance(delta, (int, float)) and math.isfinite(ssim)
            and math.isfinite(delta) and ssim >= 0.995 and delta <= 0.5)


def native_case_pass(record):
    return (record.get('status') == 'generated' and record.get('native_open') is True
            and record.get('warning_gate_verified') is True
            and record.get('watchdog_passed') is True
            and record.get('restoration_confirmed') is True)


def file_passed(item):
    text_measured = item['family'] != 'text' or item.get('text_line_breaks', {}).get('measured') is True
    return bool(item.get('structural_equal') and item.get('mapping_supported')
                and item.get('codec_run', {}).get('exit_code') == 0
                and item['generation'].get('status') == 'generated'
                and item.get('descriptor_inspection_passed')
                and item['pixel_fidelity'].get('passed')
                and item['photoshop_fidelity'].get('passed')
                and item['layer_tree_equal'] and item['layer_pixels_passed'] and text_measured
                and native_case_pass(item['photoshop_reopen']))


def measure(binary, reference, candidate, destination, env, kind):
    if not reference.is_file() or not candidate.is_file():
        return {'passed': False, 'error': 'Missing composite or layer pixels'}
    result = run([str(binary), '--psd-metrics', str(reference), str(candidate), str(destination)],
                 destination.with_suffix('.log'), env)
    try:
        metrics = read_json(destination)
    except (OSError, ValueError) as error:
        return {'passed': False, 'error': str(error), 'run': result}
    metrics['threshold'] = kind
    metrics['passed'] = result['exit_code'] == 0 and fidelity_pass(metrics, kind)
    metrics['run'] = result
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New empty output directory; default creates a unique corpus subdirectory')
    parser.add_argument('--warning-control', type=Path, action='append', default=[], help='Native negative-control report; repeat for corrupt and truncated controls')
    args = parser.parse_args()
    root = args.corpus.resolve()
    if not root.is_dir() or not args.binary.is_file():
        parser.error('Corpus and executable must exist')
    if args.output:
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            parser.error('Output must be empty; retained evidence is never overwritten')
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(tempfile.mkdtemp(prefix='pixelforge-results-', dir=root))
    roundtrips = output / 'roundtrip'
    roundtrips.mkdir(exist_ok=True)
    config = output / 'config'
    reopened_root = output / 'reopened'
    photoshop_output = output / 'photoshop'
    env = os.environ | {'PF_CONFIG_DIR': str(config)}
    manifest_path = root / 'corpus-manifest.json'
    report = {'binary': str(args.binary.resolve()), 'binary_sha256': digest(args.binary),
              'configuration': {'PF_CONFIG_DIR': str(config), 'PF_GPU': env.get('PF_GPU', 'auto')},
              'acceptance': {'integer_exact': True, 'ssim_min': 0.995,
                             'mean_delta_e00_max': 0.5},
              'files': [], 'failures': [], 'generation_failures': [], 'complete': False,
              'output': str(output)}
    project = Path(__file__).resolve().parents[1]
    report['source_fingerprints_start'] = source_fingerprints(project)
    report['tool_copies'] = {}
    provenance = output / 'provenance'
    provenance.mkdir()
    for name in ('photoshop_psd.py', 'photoshop_psd.jsx', 'psd_corpus.py'):
        source = Path(__file__).with_name(name)
        destination = provenance / name
        destination.write_bytes(source.read_bytes())
        report['tool_copies'][name] = {'path': str(destination), 'sha256': digest(destination)}
    (output / 'run-provenance.json').write_text(json.dumps({key: report[key] for key in (
        'binary', 'binary_sha256', 'configuration', 'source_fingerprints_start', 'tool_copies')}, indent=2))
    cases, generated = [], {}
    control_paths = args.warning_control or sorted((root / 'warning-controls').glob('*/report.json'))
    report['warning_controls'] = []
    for path in control_paths:
        try:
            control = read_json(path)
            report['warning_controls'].append({'path': str(path), **control})
        except (OSError, ValueError) as error:
            report['warning_controls'].append({'path': str(path), 'error': str(error)})
    proven_controls = {r.get('control') for r in report['warning_controls'] if r.get('watchdog_proven') is True}
    if not {'corrupt', 'truncated'}.issubset(proven_controls):
        report['failures'].append('Native warning watchdog needs both corrupt and truncated negative controls')
    if not manifest_path.is_file():
        report['failures'].append('No Photoshop corpus-manifest.json; provenance/coverage unverified')
    else:
        report['manifest_sha256'] = digest(manifest_path)
        report['manifest'] = read_json(manifest_path)
        cases = report['manifest'].get('cases', [])
        if (not cases or not report['manifest'].get('coverage_complete')
                or report['manifest'].get('planned_count') != len(cases)):
            report['failures'].append('Corpus coverage is partial or empty; full phase coverage is not verified')
        identities, filenames = set(), set()
        for case in cases:
            ident, filename = case.get('id'), case.get('filename')
            if (not isinstance(ident, str) or not isinstance(filename, str) or not ident
                    or Path(ident).name != ident or Path(filename).name != filename
                    or ident in identities or filename in filenames
                    or case.get('fidelity') not in {'exact', 'perceptual'}):
                parser.error('Invalid, unsafe, or duplicate corpus manifest identity')
            identities.add(ident)
            filenames.add(filename)
        try:
            generation = read_json(root / 'photoshop-results.json')
            generated = {r['id']: r for r in generation.get('cases', [])}
            report['generation_coverage_audit'] = coverage_failures(root, generation)
            for finding in report['generation_coverage_audit']:
                original = generated[finding['id']]
                generated[finding['id']] = {**original, 'status': 'invalid-authoring-coverage',
                    'original_status': original['status'], 'error': finding['reason'],
                    'failure_kind': 'authoring_coverage_invalid', 'coverage_audit': finding}
            report['generation'] = generation
            report['generation_retry_watchdogs'] = []
            for retry_dir in generation.get('retry_runs', []):
                retry_watchdog = read_json(Path(retry_dir) / 'watchdog.json')
                retry_accepted, retry_disposition = watchdog_accepted(retry_watchdog)
                report['generation_retry_watchdogs'].append({'path': retry_dir, 'disposition': retry_disposition, **{k: v for k, v in retry_watchdog.items() if k != 'observations'}})
                if not retry_accepted:
                    report['failures'].append('Generation retry failed watchdog: ' + retry_dir)
            watchdog = read_json(root / 'watchdog.json')
            report['generation_watchdog'] = {k: v for k, v in watchdog.items() if k != 'observations'}
            accepted, disposition = watchdog_accepted(watchdog)
            report['generation_watchdog']['disposition'] = disposition
            if (not generation.get('restoration_confirmed') or generation.get('fatal_error')
                    or not accepted):
                report['failures'].append('Photoshop generation failed completion, restoration, or watchdog gate')
        except (OSError, ValueError, KeyError) as error:
            report['failures'].append('Missing generation evidence: ' + str(error))
        for case in cases:
            result = generated.get(case['id'], {'status': 'missing', 'error': 'Missing generation result'})
            if result.get('status') != 'generated':
                report['generation_failures'].append({'id': case['id'], 'family': case.get('family'), **result})
        if report['generation_failures']:
            report['failures'].append(f'{len(report["generation_failures"])} planned feature cases failed generation')
    sources = sorted((root / 'originals').glob('*.psd')) + sorted((root / 'originals').glob('*.psb'))
    if not sources:
        report['failures'].append('Empty corpus')
    by_filename = {c['filename']: c for c in cases}
    unplanned = [p.name for p in sources if p.name not in by_filename]
    if unplanned:
        report['failures'].append('Unplanned files cannot establish provenance: ' + ', '.join(unplanned))
    for case in cases:
        generated_case = generated.get(case['id'], {})
        source = Path(generated_case.get('source_path', root / 'originals' / case['filename']))
        name = source.stem
        print('Checking ' + source.name, flush=True)
        item = {'id': case['id'], 'family': case.get('family', 'unknown'), 'fidelity': case['fidelity'],
                'file': source.name, 'generation': generated.get(case['id'], {'status': 'missing'})}
        report['files'].append(item)
        if not source.is_file():
            item['error'] = 'Planned Photoshop source was not generated'
            continue
        item['source_sha256'] = digest(source)
        destination = roundtrips / source.name
        result_path = output / (name + '.json')
        result_path.unlink(missing_ok=True)
        item['codec_run'] = run([str(args.binary.resolve()), '--psd-check', str(source),
                                 str(destination), str(result_path)], output / (name + '.log'), env)
        if result_path.exists():
            item.update(read_json(result_path))
        else:
            item['error'] = 'No codec report was produced'
        if destination.exists():
            item['output_sha256'] = digest(destination)
        inspections = item.get('inspections', [])
        item['inspection_counts'] = dict(collections.Counter(row.get('status', 'unknown') for row in inspections))
        item['descriptor_inspection_passed'] = 'inspections' in item and all(row.get('status') in {'typed', 'opaque'} for row in inspections)
    photoshop_cases = {}
    if any(roundtrips.iterdir()):
        harness = Path(__file__).with_name('photoshop_psd.py')
        report['photoshop_run'] = run(['python3', str(harness), 'native-verify', str(root), str(roundtrips),
                                      '--output', str(photoshop_output), '--reference-output', str(reopened_root), '--defer-fidelity'],
                                      output / 'photoshop-reopen.log', env, timeout=7200)
        try:
            photoshop_result = read_json(photoshop_output / 'photoshop-results.json')
            photoshop_cases = {r['id']: r for r in photoshop_result.get('cases', [])}
            photoshop_watchdog = read_json(photoshop_output / 'watchdog.json')
            report['photoshop_warning_free'] = (photoshop_watchdog.get('status') == 'passed'
                and photoshop_result.get('restoration_confirmed') is True
                and photoshop_result.get('warning_gate_verified') is True)
        except (OSError, ValueError, KeyError) as error:
            report['photoshop_warning_free'] = False
            report['failures'].append('Missing Photoshop evidence: ' + str(error))
        if not report.get('photoshop_warning_free'):
            report['failures'].append('Photoshop reopening warning/dialog/restoration gate failed')
    # Each row must prove all four gates. Codec preservation alone deliberately
    # cannot certify mapping, independent compositing, or Photoshop acceptance.
    for item in report['files']:
        name = Path(item['file']).stem
        reference = Path(item['generation'].get('reference_path', root / 'reference' / item['id']))
        reopened = reopened_root / item['id']
        item['photoshop_reopen'] = photoshop_cases.get(item['id'], {'status': 'missing'})
        item['photoshop_warning_free'] = native_case_pass(item['photoshop_reopen'])
        for key, candidate in [('pixel_fidelity', output / (name + '.png')),
                               ('photoshop_fidelity', reopened / 'composite.png')]:
            metrics_path = output / (name + '-' + key + '.json')
            item[key] = measure(args.binary.resolve(), reference / 'composite.png', candidate, metrics_path, env, item['fidelity'])
        try:
            before = read_json(reference / 'tree.json')
            after = read_json(reopened / 'tree.json')
            # IDs are only stable inside one PSD. Harness stores semantic tree
            # fields and opaque descriptor streams separately from runtime IDs.
            item['layer_tree_equal'] = before == after
        except (OSError, ValueError) as error:
            item['layer_tree_equal'] = False
            item['tree_error'] = str(error)
        item['layer_fidelity'] = []
        for layer in sorted((reference / 'layers').glob('*.png')):
            metrics = measure(args.binary.resolve(), layer, reopened / 'layers' / layer.name,
                              output / (name + '-layer-' + layer.stem + '.json'), env, item['fidelity'])
            item['layer_fidelity'].append({'layer': layer.stem, **metrics})
        item['layer_pixels_passed'] = bool(item['layer_fidelity']) and all(row['passed'] for row in item['layer_fidelity'])
        if item['family'] == 'text':
            item.setdefault('text_line_breaks', {'measured': False, 'error': 'No independent PixelForge versus Photoshop line-break measurement'})
        item['passed'] = file_passed(item)
        if not item['passed']:
            report['failures'].append(item['file'])
    report['families'] = {}
    for item in report['files']:
        family = report['families'].setdefault(item['family'], {'planned': 0, 'generated': 0, 'preserved': 0, 'mapped': 0, 'reopened': 0, 'passed': 0})
        family['planned'] += 1
        family['generated'] += item['generation'].get('status') == 'generated'
        family['preserved'] += item.get('structural_equal') is True
        family['mapped'] += item.get('mapping_supported') is True
        family['reopened'] += item['photoshop_reopen'].get('status') == 'generated'
        family['passed'] += item['passed']
    report['source_fingerprints_end'] = source_fingerprints(project)
    report['binary_sha256_end'] = digest(args.binary)
    report['source_unchanged_during_run'] = report['source_fingerprints_start'] == report['source_fingerprints_end']
    report['binary_unchanged_during_run'] = report['binary_sha256'] == report['binary_sha256_end']
    if not report['source_unchanged_during_run'] or not report['binary_unchanged_during_run']:
        report['failures'].append('Source, tooling, or executable changed during the corpus run; evidence does not describe one verified build')
    report['complete'] = bool(report['files']) and not report['failures']
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    passed = sum(bool(item.get('passed')) for item in report['files'])
    table = '\n'.join('| ' + name + ' | ' + ' | '.join(str(row[key]) for key in ['planned', 'generated', 'preserved', 'mapped', 'reopened', 'passed']) + ' |'
                      for name, row in sorted(report['families'].items()))
    (output / 'REPORT.md').write_text(
        '# PSD corpus verification\n\n'
        f'{passed}/{len(report["files"])} planned files passed all measured gates. '
        f'Phase acceptance: {"PASS" if report["complete"] else "FAIL / INCOMPLETE"}.\n\n'
        '| Family | Planned | Generated | Preserved | Mapped | Reopened | Passed |\n'
        '|---|---:|---:|---:|---:|---:|---:|\n' + table + '\n\n'
        'See report.json and per-file logs for separate authoring, structural, descriptor, '
        'mapping, rendering, and Photoshop evidence. Opaque retention is not interpreted/editable support. '
        'Missing evidence fails; text line breaks remain separately measured.\n')
    print(f'{passed}/{len(report["files"])} passed; report: {output / "report.json"}', flush=True)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
