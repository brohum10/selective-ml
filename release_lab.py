"""Real candidate comparisons on identical inputs, with auditable release checks.

Classifier training is separate from the comparator. Noise stress is reported
separately and is never used as the model-release comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

import selective_ml as ml
from walkthrough import ROOT, case_hash, make_noise

COMPARATOR_HASH = '82870b5ca827ff1a57640c79832d916f5daac31cce47cd9bcc162c0e2b7fb97e'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_protocol():
    protocol = json.loads((ROOT/'lab_protocol.json').read_text())
    if (protocol['seeds'] != [42, 43, 44] or protocol['epochs'] != 120
            or protocol['baseline'] != 'linear-softmax' or protocol['candidate'] != 'relu-mlp-64'
            or protocol['evaldelta_sha256'] != COMPARATOR_HASH):
        raise ValueError('unexpected protocol; create a separately versioned experiment instead')
    return protocol


def infer(params, temperature, threshold, pixels):
    """Run each example; latency includes NumPy conversion and prediction only."""
    results = []
    for image in pixels:
        started = time.perf_counter_ns()
        probabilities = ml.softmax(ml.logits(params, np.asarray(image)[None, :])/temperature)[0]
        if not np.isfinite(probabilities).all():
            raise ValueError('nonfinite model prediction')
        prediction, confidence = int(probabilities.argmax()), float(probabilities.max())
        elapsed_ms = (time.perf_counter_ns()-started)/1_000_000
        results.append({'prediction': prediction, 'confidence': confidence,
                        'decision': 'accept' if confidence >= threshold else 'review',
                        'probabilities': probabilities.tolist(), 'latency_ms': elapsed_ms})
    return results


def policy_summary(predictions, labels):
    accepted = [i for i, row in enumerate(predictions) if row['decision'] == 'accept']
    reviewed = [i for i, row in enumerate(predictions) if row['decision'] == 'review']
    accepted_errors = sum(predictions[i]['prediction'] != int(labels[i]) for i in accepted)
    reviewed_errors = sum(predictions[i]['prediction'] != int(labels[i]) for i in reviewed)
    return {'total': len(labels), 'accepted': len(accepted), 'deferred': len(reviewed),
            'coverage': len(accepted)/len(labels),
            'accepted_accuracy': 1-accepted_errors/len(accepted) if accepted else None,
            'accepted_errors': int(accepted_errors), 'deferred_errors': int(reviewed_errors),
            'deferred_correct': len(reviewed)-int(reviewed_errors)}


def make_rows(rows, images, labels, predictions, seed, run):
    return [{'id': f'digits-row-{int(row):04}',
             'case_hash': case_hash(row, images[i]*16, labels[i]),
             'suite': f'digits-seed{seed}-test-v2', 'scorer': 'exact-match-v1',
             'run': run, 'tags': [f'digit-{int(labels[i])}'],
             'score': int(predictions[i]['prediction'] == int(labels[i])),
             'latency_ms': predictions[i]['latency_ms']} for i, row in enumerate(rows)]


def write_rows(path, rows):
    path.write_text(''.join(canonical(row)+'\n' for row in rows), encoding='utf-8')


def gate(script, baseline, candidate, config):
    args = [sys.executable, str(script), str(baseline), str(candidate)]
    for key, value in config.items():
        args += ['--'+key.replace('_', '-'), str(value)]
    process = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    if process.returncode not in (0, 1):
        raise ValueError('comparator rejected comparison: '+process.stderr[:500])
    report = json.loads(process.stdout)
    if report['decision'] != ('pass' if process.returncode == 0 else 'block'):
        raise ValueError('comparator JSON and exit code disagree')
    return report


def run(output, comparator, progress=None):
    """Write a fresh run directory; never replace an earlier evidence bundle."""
    output, comparator = Path(output).resolve(), Path(comparator).resolve()
    if not comparator.is_file() or hashlib.sha256(comparator.read_bytes()).hexdigest() != COMPARATOR_HASH:
        raise ValueError('EvalDelta source does not match the pinned, reviewed comparator')
    output.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol()
    save(output/'protocol.json', protocol)
    x, y = ml.load_data(ROOT/'data/digits.csv.gz')
    events = []

    def emit(message):
        event = {'sequence': len(events)+1, 'message': message}
        events.append(event)
        print(message, flush=True)
        if progress:
            progress(event)

    emit('Verified dataset checksum and fixed protocol: seeds 42, 43, 44.')
    runs = []
    for seed in protocol['seeds']:
        folder = output/f'seed-{seed}'
        folder.mkdir()
        split = ml.split_indices(y, seed)
        a, c, v, t = (split[name] for name in ('train', 'calibration', 'policy', 'test'))
        save(folder/'splits.json', {name: ids.tolist() for name, ids in split.items()})
        emit(f'Seed {seed}: training linear V1 and neural V2 on the same {len(a)} training rows.')
        models, predictions, model_meta = {}, {}, {}
        for name, hidden in (('baseline', 0), ('candidate', 64)):
            params = ml.train(x[a], y[a], hidden=hidden, seed=seed, epochs=protocol['epochs'])
            temperature = ml.calibrate(ml.logits(params, x[c]), y[c])
            threshold = ml.choose_threshold(ml.softmax(ml.logits(params, x[v])/temperature), y[v],
                                            protocol['policy_target'], protocol['policy_min_coverage'])
            artifact = {'version': 1, 'architecture': 'linear-softmax' if hidden == 0 else 'relu-mlp-64',
                        'seed': seed, 'input_scale': 16, 'temperature': temperature,
                        'threshold': threshold, 'weights': {k: value.tolist() for k, value in params.items()}}
            save(folder/f'{name}-model.json', artifact)
            models[name] = params
            predictions[name] = infer(params, temperature, threshold, x[t])
            model_meta[name] = {'temperature': temperature, 'threshold': threshold,
                'architecture': artifact['architecture'],
                'sha256': hashlib.sha256((folder/f'{name}-model.json').read_bytes()).hexdigest(),
                'nll_before': ml.nll(ml.logits(params, x[t]), y[t]),
                'nll_after': ml.nll(ml.logits(params, x[t]), y[t], temperature)}
        images, labels = x[t], y[t]
        rows_by_name = {name: make_rows(t, images, labels, values, seed, 'linear-v1' if name == 'baseline' else 'neural-v2')
                        for name, values in predictions.items()}
        for name, values in rows_by_name.items():
            write_rows(folder/f'{name}.jsonl', values)
        # These are genuinely the exact same clean task inputs and labels.
        for baseline_row, candidate_row in zip(rows_by_name['baseline'], rows_by_name['candidate'], strict=True):
            if any(baseline_row[k] != candidate_row[k] for k in ('id', 'case_hash', 'suite', 'scorer', 'tags')):
                raise ValueError('paired evaluation alignment failed')
        emit(f'Seed {seed}: comparing {len(t)} identical clean cases with the actual EvalDelta CLI.')
        report = gate(comparator, folder/'baseline.jsonl', folder/'candidate.jsonl', protocol['gate'])
        control = gate(comparator, folder/'baseline.jsonl', folder/'baseline.jsonl', protocol['gate'])
        if control['decision'] != 'pass':
            raise ValueError('identical-run control did not pass')
        save(folder/'gate.json', report)
        save(folder/'control.json', control)
        noisy = make_noise(images, protocol['noise_sigma'], seed=seed+1000)
        stress = infer(models['candidate'], model_meta['candidate']['temperature'], model_meta['candidate']['threshold'], noisy)
        examples = [{'row': int(row), 'label': int(labels[i]), 'pixels': (images[i]*16).tolist(),
                     'baseline': predictions['baseline'][i], 'candidate': predictions['candidate'][i]}
                    for i, row in enumerate(t)]
        save(folder/'predictions.json', examples)
        save(folder/'noise-stress.json', {
            'scope': 'Separate simulated corruption; original labels retained, label preservation NOT established. Not used in the model-release comparison.',
            'sigma': protocol['noise_sigma'], 'seed': seed+1000,
            'agreement_with_original_labels': float(np.mean([r['prediction'] == int(label) for r, label in zip(stress, labels, strict=True)])),
            'rows': [{'row': int(t[i]), 'original_label': int(labels[i]), 'pixels': (noisy[i]*16).tolist(), **row} for i, row in enumerate(stress)]})
        summary = {'seed': seed, 'n': len(t), 'split_counts': {name: len(ids) for name, ids in split.items()},
            'baseline_accuracy': report['slices'][0]['baseline_mean'],
            'candidate_accuracy': report['slices'][0]['candidate_mean'],
            'gate': report, 'control_decision': control['decision'], 'models': model_meta,
            'policy': {name: policy_summary(values, labels) for name, values in predictions.items()},
            'noise_stress': {'sigma': protocol['noise_sigma'], 'scope': 'Label agreement only; not a release comparison or verified label-preserving corruption.',
                            'agreement_with_original_labels': float(np.mean([r['prediction'] == int(label) for r, label in zip(stress, labels, strict=True)]))},
            'examples': examples}
        save(folder/'summary.json', summary)
        runs.append(summary)
        emit(f"Seed {seed}: {summary['baseline_accuracy']:.2%} -> {summary['candidate_accuracy']:.2%}; gate {report['decision'].upper()}.")
    # Selection is by predetermined seed/row order, never by maximum confidence or score.
    deferred = next(({'seed': run['seed'], **example} for run in runs for example in run['examples']
                     if example['candidate']['decision'] == 'review'), None)
    changed = next((example for example in runs[0]['examples']
                    if example['baseline']['prediction'] != example['candidate']['prediction']), None)
    emit('Complete. Saved models, exact predictions, aligned cases, gate reports, and file hashes.')
    result = {'version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
              'protocol_sha256': hashlib.sha256(canonical(protocol).encode()).hexdigest(),
              'dataset_sha256': ml.DATA_SHA256, 'comparator_sha256': COMPARATOR_HASH,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ('selective_ml.py', 'release_lab.py', 'walkthrough.py', 'lab_protocol.json')},
              'runs': runs, 'primary_seed': 42, 'first_changed_prediction': changed,
              'first_deferred_prediction': deferred,
              'selection_rule': 'Primary seed fixed at 42. First changed row in that seed; first deferred row in seed order 42/43/44. No best-seed selection.',
              'limitations': ['Retrospective educational dataset; not a blind holdout claim.',
                 'Test sets overlap across seeds; do not pool observations.',
                 'Random row splits do not establish unseen-writer generalization.',
                 'Gate intervals are approximate and unadjusted per slice.',
                 'BLOCK can mean insufficient evidence, not a statistically established regression.',
                 'Selective deferral is a signal, not an implemented human review service or correctness guarantee.'],
              'events': events}
    save(output/'summary.json', result)
    files = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(output.rglob('*')) if path.is_file()}
    save(output/'manifest.json', {'version': 1, 'sha256': files})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaldelta', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='a new directory; existing outputs are never overwritten')
    parser.add_argument('--enforce', action='store_true', help='exit 1 if any planned seed is blocked; without this flag, exit 0 means analysis completed')
    args = parser.parse_args()
    try:
        result = run(args.output, args.evaldelta)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f'release-lab: {error}\n')
    print('Primary comparison:', result['runs'][0]['gate']['decision'].upper())
    print('Evidence:', args.output.resolve())
    return 1 if args.enforce and any(run['gate']['decision'] == 'block' for run in result['runs']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
