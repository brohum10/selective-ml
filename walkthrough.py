"""Export real inference traces and paired EvalDelta inputs, without retraining.

The candidate is the SAME frozen model behind a deterministic noise transform.
This is a robustness stress test, not a comparison of two newly trained models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import selective_ml as ml

ROOT = Path(__file__).resolve().parent


def case_hash(row, pixels, label):
    payload = {"dataset_sha256": ml.DATA_SHA256, "row": int(row),
               "pixels": [int(v) for v in pixels], "label": int(label)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def make_noise(pixels, sigma, seed=1042):
    if type(sigma) not in (int, float) or not np.isfinite(sigma) or not 0 <= sigma <= .5:
        raise ValueError("sigma must be a finite number between 0 and 0.5")
    return np.clip(pixels + np.random.default_rng(seed).normal(0, sigma, pixels.shape), 0, 1)


def infer(artifact, pixels):
    """Time each actual single-example prediction; no simulated latency values."""
    predictions = []
    for image in pixels:
        start = time.perf_counter_ns()
        result = ml.predict(artifact, [(image * 16).tolist()])[0]
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        predictions.append({**result, "latency_ms": elapsed_ms})
    return predictions


def export(output, *, sigma=.2, reference=None):
    reference = Path(reference) if reference is not None else ROOT / 'results/reference'
    x, y = ml.load_data(ROOT / 'data/digits.csv.gz')
    artifact_path = reference / 'model.json'
    artifact = json.loads(artifact_path.read_text())
    splits = json.loads((reference / 'splits.json').read_text())
    expected = {k: v.tolist() for k, v in ml.split_indices(y, 42).items()}
    if splits != expected:
        raise ValueError("walkthrough requires the documented seed-42 reference split")
    rows = np.array(splits['test'], dtype=int)
    pixels, labels = x[rows], y[rows]
    noisy = make_noise(pixels, sigma)
    clean_predictions = infer(artifact, pixels)
    noisy_predictions = infer(artifact, noisy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline, candidate, examples = [], [], []
    for i, row in enumerate(rows):
        common = {'id': f'digits-row-{row:04}',
                  'case_hash': case_hash(row, pixels[i]*16, labels[i]),
                  'suite': 'digits-seed42-test-v1', 'scorer': 'exact-match-v1',
                  'tags': [f'digit-{labels[i]}']}
        clean, stress = clean_predictions[i], noisy_predictions[i]
        for target, name, result in ((baseline, 'clean', clean), (candidate, 'noise-stress', stress)):
            target.append({**common, 'run': name,
                           'score': int(result['prediction'] == labels[i]),
                           'latency_ms': result['latency_ms']})
        examples.append({'row': int(row), 'label': int(labels[i]),
                         'clean': {**clean, 'pixels': (pixels[i]*16).tolist()},
                         'stress': {**stress, 'pixels': (noisy[i]*16).tolist()}})

    clean_accuracy = float(np.mean([r['score'] for r in baseline]))
    stress_accuracy = float(np.mean([r['score'] for r in candidate]))
    newly_wrong = [e for e in examples if e['clean']['prediction'] == e['label']
                   and e['stress']['prediction'] != e['label']]
    confident_wrong = [e for e in newly_wrong if e['stress']['confidence'] >= .95]
    selected = next(iter(confident_wrong or newly_wrong), None)
    # The animation also uses real forward passes, not interpolated confidence.
    sweep = []
    if selected:
        index = next(i for i, e in enumerate(examples) if e['row'] == selected['row'])
        for level in np.linspace(0, sigma, 41):
            image = make_noise(pixels, float(level))[index]
            result = infer(artifact, image[None, :])[0]
            sweep.append({'sigma': float(level), 'pixels': (image*16).tolist(), **result})
    summary = {'version': 1, 'seed': 42, 'noise_seed': 1042, 'sigma': sigma,
               'n': len(rows), 'dataset_sha256': ml.DATA_SHA256,
               'model_sha256': hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
               'clean_accuracy': clean_accuracy, 'stress_accuracy': stress_accuracy,
               'clean_errors': sum(1-r['score'] for r in baseline),
               'stress_errors': sum(1-r['score'] for r in candidate),
               'learned_threshold': artifact['threshold'],
               'newly_wrong_rows': [e['row'] for e in newly_wrong],
               'featured_row': selected['row'] if selected else None,
               'selection_rule': 'first newly wrong test row with stress confidence >= .95; otherwise first newly wrong row',
               'scope': 'Frozen seed-42 model, real public digit images, simulated Gaussian input corruption. Not a model upgrade, production benchmark, or OOD guarantee.',
               'timing': 'Single-example ml.predict wall time; includes validation and array conversion. Hardware dependent, not a model serving benchmark.',
               'pairing': 'Case hashes cover the original task input and answer. The candidate pipeline applies the documented corruption transform to that same task.',
               'noise_sweep': sweep,
               'examples': examples}
    for name, data in (('baseline.jsonl', baseline), ('candidate.jsonl', candidate)):
        (output / name).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in data))
    (output / 'trace.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/walkthrough')
    parser.add_argument('--sigma', type=float, default=.2)
    args = parser.parse_args()
    try:
        result = export(args.output, sigma=args.sigma)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'walkthrough: {error}\n')
    print(f"Actual inference on {result['n']} held-out examples")
    print(f"Clean: {result['clean_accuracy']:.2%} | noise stress: {result['stress_accuracy']:.2%}")
    print(f"Learned review threshold: {result['learned_threshold']}")
    print('Exported trace.json and paired EvalDelta JSONL files to', args.output)


if __name__ == '__main__':
    main()
