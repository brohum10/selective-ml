"""Verify SelectiveML -> EvalDelta with real CLI pass/block/invalid outcomes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from walkthrough import ROOT, export


def run_gate(script, baseline, candidate, expected_code):
    result = subprocess.run([sys.executable, str(script), str(baseline), str(candidate),
                             '--margin', '0.02', '--min-cases', '20', '--samples', '2000',
                             '--seed', '42', '--max-p95-ms', '1000'],
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != expected_code:
        raise RuntimeError(f'expected exit {expected_code}, received {result.returncode}: {result.stderr}')
    if expected_code == 2:
        if not result.stderr.startswith('Invalid comparison:'):
            raise RuntimeError('invalid-data check failed for an unexpected reason')
        return None
    report = json.loads(result.stdout)
    expected_decision = 'pass' if expected_code == 0 else 'block'
    if report.get('decision') != expected_decision:
        raise RuntimeError('gate decision does not match the CLI exit code')
    return report


def verify(script, output):
    script, output = Path(script).resolve(), Path(output)
    if not script.is_file():
        raise ValueError('provide the path to an existing trusted EvalDelta checkout')
    trace = export(output)
    baseline, candidate = output/'baseline.jsonl', output/'candidate.jsonl'
    gate = run_gate(script, baseline, candidate, 1)
    if gate['slices'][0]['quality_status'] != 'regression':
        raise RuntimeError('stress test did not detect the expected overall regression')
    control = run_gate(script, baseline, baseline, 0)
    with tempfile.TemporaryDirectory() as folder:
        rows = [json.loads(line) for line in baseline.read_text().splitlines()]
        rows[0]['case_hash'] = '0'*64
        invalid = Path(folder)/'mismatched.jsonl'
        invalid.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        run_gate(script, baseline, invalid, 2)
    for name, data in (('gate.json', gate), ('control.json', control)):
        (output/name).write_text(json.dumps(data, indent=2, allow_nan=False))
    verification = {'version': 1, 'cases': trace['n'], 'checks': {
        'noise_stress_blocks': True, 'identical_run_passes': True,
        'mismatched_case_hash_is_invalid': True},
        'scope': 'Local CLI integration checks, not a deployment approval or production safety certificate.'}
    (output/'integration.json').write_text(json.dumps(verification, indent=2))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaldelta', type=Path, required=True,
                        help='path to evaldelta.py from a trusted checkout')
    parser.add_argument('--output', type=Path, default=ROOT/'results/walkthrough')
    args = parser.parse_args()
    try:
        result = verify(args.evaldelta, args.output)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f'integration check failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
