#!/usr/bin/env python3
"""Serial alternating RGB8 tier comparisons against the preserved executable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

NUMBER = r'\d+(?:\.\d+)?'
TIMING = re.compile(rf'^(.+?)\s+({NUMBER}) ms\b')
COMPOSITING_ROWS = {
    'CPU, all tiles': 'compositing / CPU all tiles',
    'GPU cold (uploads), all tiles': 'compositing / GPU cold uploads all tiles',
    'GPU warm, all tiles': 'compositing / GPU warm all tiles',
    'GPU after one-tile edit (1 tile)': 'compositing / GPU one-tile edit',
}
GPU_TABLE = re.compile(
    rf'^(.+?)\s{{2,}}({NUMBER})\s+({NUMBER}|—)\s+({NUMBER}|—)\s+({NUMBER}|—)'
    r'(?:\s+(CPU|GPU)\s+(ok|OVER))?\s*$'
)


def parse_log(text):
    """Return operation timings and the original resource/budget evidence."""
    result = {'timings_ms': {}, 'resources': [], 'resource_lines': [],
              'budget_misses': [], 'budget_summaries': [], 'warnings': []}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(('memory before:', 'memory after:', 'budget before:', 'budget after:')):
            result['resource_lines'].append(raw)
            kind, phase = line.split(':', 1)[0].split()
            counters = {'kind': kind, 'phase': phase}
            if kind == 'memory':
                for label, key in [('RSS', 'rss_mib'), ('peak RSS', 'peak_rss_mib')]:
                    match = re.search(rf'\b{label} ({NUMBER}|unavailable) MiB', line)
                    if match:
                        counters[key] = None if match[1] == 'unavailable' else float(match[1])
                match = re.search(r'page-ins (\d+|unavailable)', line)
                if match:
                    counters['pageins'] = None if match[1] == 'unavailable' else int(match[1])
            else:
                match = re.search(rf': ({NUMBER}) / ({NUMBER}) MB', line)
                if match:
                    counters.update(used_mb=float(match[1]), budget_mb=float(match[2]))
                for label in ('compressed', 'scratch', 'GPU'):
                    match = re.search(rf'({NUMBER}) MB {label}', line)
                    if match:
                        counters[f'{label.lower()}_mb'] = float(match[1])
                match = re.search(rf'peak tracked ({NUMBER}) MiB', line)
                if match:
                    counters['peak_tracked_mib'] = float(match[1])
            result['resources'].append(counters)
            continue
        summary = re.fullmatch(r'(?:(\d+) checks, )?(\d+) over budget', line)
        if summary:
            result['budget_summaries'].append({'checks': int(summary[1]) if summary[1] else None,
                                               'over_budget': int(summary[2]), 'line': raw})
            continue
        if re.search(r'\b(?:OVER|FAIL|FAILED)\b', line):
            result['budget_misses'].append(raw)
        if 'scratch warning:' in line or 'GPU:' in line or 'no GPU' in line:
            result['warnings'].append(raw)
        # Calibration, target thresholds, and summary lines are not workload
        # timings. In particular a no-op filter must not pass on calibration.
        if line.startswith(('cost model:', 'latency target:', 'OVER:', 'resident probe',
                            'scratch warning:', 'GPU benchmark:', 'generate test scene')):
            continue
        match = TIMING.match(line)
        if match:
            label = COMPOSITING_ROWS.get(match[1].strip(), match[1].strip())
            result['timings_ms'].setdefault(label, []).append(float(match[2]))
            continue
        table = GPU_TABLE.match(line)
        if table:
            label = table[1].strip()
            for column, value in zip(('CPU', 'GPU end-to-end', 'GPU resident', 'GPU kernel'), table.group(2, 3, 4, 5)):
                if value != '—':
                    result['timings_ms'].setdefault(f'{label} / {column}', []).append(float(value))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('--after', type=Path, default=Path('target/release/pixelforge'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tiers', default='24,61,150,268')
    parser.add_argument('--suite', choices=['engine', 'tools', 'filters', 'gpu'], default='engine')
    parser.add_argument('--only', help='Optional suite filter, forwarded after --tier TIER as a positional argument')
    parser.add_argument('--reps', type=int, default=9)
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error('--reps must be positive')
    if args.only is not None and (not args.only.strip() or args.suite == 'engine'):
        parser.error('--only requires a nonempty filter and a tools, filters, or gpu suite')
    tiers = args.tiers.split(',')
    if len(set(tiers)) != len(tiers) or any(tier not in {'24', '61', '150', '268'} for tier in tiers):
        parser.error('--tiers requires distinct values from 24,61,150,268')
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error('Use a new output directory to retain earlier evidence')
    args.out.mkdir(parents=True, exist_ok=True)
    flag = {'engine': '--bench', 'tools': '--bench-tools', 'filters': '--bench-filters', 'gpu': '--bench-gpu'}[args.suite]
    mode = 'force' if args.suite == 'gpu' else 'off'
    binaries = {'before': args.before.resolve(), 'after': args.after.resolve()}
    report = {'suite': args.suite, 'only': args.only, 'repetitions': args.reps, 'PF_GPU': mode,
              'status': 'running', 'binary_sha256': {}, 'runs': [], 'tiers': {},
              'failures': [], 'regressions_over_5_percent': [], 'unquantified_comparisons': []}

    def write_report():
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def fail(message):
        report['status'] = 'failed'
        report['failures'].append(message)
        write_report()
        raise SystemExit(message)

    for name, binary in binaries.items():
        try:
            with binary.open('rb') as stream:
                report['binary_sha256'][name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError as error:
            fail(f'Cannot read {name} binary: {error}')
    for tier in tiers:
        measurements = {}
        expected_shape = None
        report['tiers'][tier] = measurements
        for repeat in range(args.reps):
            for name in (['before', 'after'] if repeat % 2 == 0 else ['after', 'before']):
                config = (args.out / 'config' / tier / name / str(repeat)).resolve()
                log = args.out / f'{args.suite}-{tier}-{name}-{repeat}.log'
                command = [str(binaries[name]), flag, '--tier', tier]
                if args.only is not None:
                    command.append(args.only)
                run = {'tier': tier, 'name': name, 'repeat': repeat, 'command': command,
                       'config': str(config), 'log': str(log)}
                report['runs'].append(run)
                started = time.monotonic()
                print(f'{args.suite} {tier} MP: {name} {repeat + 1}/{args.reps}', flush=True)
                try:
                    with log.open('w') as stream:
                        result = subprocess.run(command,
                            env=os.environ | {'PF_GPU': mode, 'PF_CONFIG_DIR': str(config)},
                            stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                    run['exit_code'] = result.returncode
                except (OSError, subprocess.TimeoutExpired) as error:
                    run['seconds'] = time.monotonic() - started
                    run.update(parse_log(log.read_text(errors='replace') if log.exists() else ''))
                    fail(f'Benchmark execution failed: {log}: {error}')
                run['seconds'] = time.monotonic() - started
                run.update(parse_log(log.read_text(errors='replace')))
                if result.returncode:
                    fail(f'Benchmark exited {result.returncode}: {log}')
                timings = run['timings_ms']
                if not timings:
                    fail(f'No workload timing rows found: {log}')
                if args.suite == 'gpu' and args.only and 'compositing'.find(args.only.lower()) >= 0:
                    missing = set(COMPOSITING_ROWS.values()) - timings.keys()
                    if missing:
                        fail(f'Missing GPU compositing timings {sorted(missing)}: {log}')
                shape = {label: len(values) for label, values in timings.items()}
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    fail(f'Timing rows differ between repetitions/binaries: {log}; expected {expected_shape}, got {shape}')
                for label, values in timings.items():
                    measurements.setdefault(label, {}).setdefault(name, []).extend(values)
                write_report()
        for label, row in measurements.items():
            before, after = [statistics.median(row[name]) for name in binaries]
            row.update(before_median_ms=before, after_median_ms=after,
                       change_percent=(after / before - 1) * 100 if before else None)
            if before and after > before * 1.05:
                report['regressions_over_5_percent'].append({'tier': tier, 'operation': label, **row})
            elif not before:
                report['unquantified_comparisons'].append({'tier': tier, 'operation': label,
                    'reason': 'Baseline timing rounded to zero; percentage change cannot be established', **row})
        write_report()
    report['status'] = 'completed'
    write_report()
    print(f'{len(report["regressions_over_5_percent"])} measurements over 5%; investigate, do not discard.', flush=True)


if __name__ == '__main__':
    main()
