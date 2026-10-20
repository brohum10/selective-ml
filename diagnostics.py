"""Deterministic, descriptive diagnostics for completed Release Lab predictions.

The threshold sweep is a retrospective test-set exploration, NOT policy fitting.
It never changes trained models, frozen policy thresholds or release decisions.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

MAX_CASES = 5000


def validate_examples(examples):
    if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_CASES:
        raise ValueError('expected 1..5000 prediction examples')
    seen = set()
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError('each example must be an object')
        row, label = example.get('row'), example.get('label')
        if type(row) is not int or row < 0 or row in seen:
            raise ValueError('row IDs must be unique nonnegative integers')
        if type(label) is not int or not 0 <= label <= 9:
            raise ValueError('labels must be digits 0..9')
        seen.add(row)
        for name in ('baseline', 'candidate'):
            record = example.get(name)
            if not isinstance(record, dict):
                raise ValueError('both model predictions are required')
            predicted, confidence = record.get('prediction'), record.get('confidence')
            if type(predicted) is not int or not 0 <= predicted <= 9:
                raise ValueError('predictions must be digits 0..9')
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('confidence must be finite in [0, 1]')


def operating_point(examples, threshold):
    """Evaluate confidence >= threshold; empty acceptance has undefined accuracy."""
    validate_examples(examples)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite in [0, 1]')
    accepted = [e for e in examples if e['candidate']['confidence'] >= threshold]
    accepted_errors = sum(e['candidate']['prediction'] != e['label'] for e in accepted)
    total_errors = sum(e['candidate']['prediction'] != e['label'] for e in examples)
    deferred = len(examples)-len(accepted)
    deferred_errors = total_errors-accepted_errors
    return {'threshold': threshold, 'total': len(examples), 'accepted': len(accepted),
            'deferred': deferred, 'accepted_errors': accepted_errors,
            'deferred_errors': deferred_errors, 'deferred_correct': deferred-deferred_errors,
            'coverage': len(accepted)/len(examples),
            'accepted_accuracy': 1-accepted_errors/len(accepted) if accepted else None}


def threshold_frontier(examples):
    """O(n log n) exact attainable policies, grouping ties without breaking them.

    A confidence of exactly 1 is always accepted by thresholds in [0,1].
    Prefix sums make each operating point O(1) after sorting, not O(n).
    """
    validate_examples(examples)
    ordered = sorted(examples, key=lambda e: e['candidate']['confidence'])
    confidences = [e['candidate']['confidence'] for e in ordered]
    prefix_errors = [0]
    for e in ordered:
        prefix_errors.append(prefix_errors[-1]+int(e['candidate']['prediction'] != e['label']))
    thresholds = [0.0]+[math.nextafter(c, math.inf) for c in sorted(set(confidences)) if c < 1]
    points = []
    for threshold in thresholds:
        deferred = bisect.bisect_left(confidences, threshold)
        accepted = len(ordered)-deferred
        errors = prefix_errors[-1]-prefix_errors[deferred]
        points.append({'threshold': threshold, 'total': len(ordered), 'accepted': accepted,
                       'deferred': deferred, 'accepted_errors': errors,
                       'deferred_errors': prefix_errors[deferred],
                       'deferred_correct': deferred-prefix_errors[deferred],
                       'coverage': accepted/len(ordered),
                       'accepted_accuracy': 1-errors/accepted if accepted else None})
    return points


def reliability(examples, name):
    """Ten fixed equal-width bins; ECE is descriptive and bin-dependent."""
    bins = [{'lower': i/10, 'upper': (i+1)/10, 'count': 0,
             'correct': 0, 'confidence_sum': 0.0} for i in range(10)]
    for e in examples:
        confidence = e[name]['confidence']
        bucket = bins[min(9, int(confidence*10))]
        bucket['count'] += 1
        bucket['correct'] += int(e[name]['prediction'] == e['label'])
        bucket['confidence_sum'] += confidence
    ece = 0.0
    for bucket in bins:
        n = bucket['count']
        bucket['accuracy'] = bucket['correct']/n if n else None
        bucket['mean_confidence'] = bucket.pop('confidence_sum')/n if n else None
        if n:
            ece += n/len(examples)*abs(bucket['accuracy']-bucket['mean_confidence'])
    return {'bins': bins, 'ece': ece,
            'scope': 'Post-calibration test predictions; descriptive, bin-dependent, not a guarantee or fitting target.'}


def summarize_run(run):
    examples = run['examples']
    validate_examples(examples)
    if run['n'] != len(examples):
        raise ValueError('case count does not match predictions')
    matrices = {name: [[0]*10 for _ in range(10)] for name in ('baseline', 'candidate')}
    changes = {'both_correct': 0, 'improved': 0, 'regressed': 0, 'both_wrong': 0}
    rows = []
    for e in sorted(examples, key=lambda e: e['row']):
        a, b = e['baseline'], e['candidate']
        correct_a, correct_b = a['prediction'] == e['label'], b['prediction'] == e['label']
        category = 'both_correct' if correct_a and correct_b else 'improved' if correct_b else 'regressed' if correct_a else 'both_wrong'
        changes[category] += 1
        for name in matrices:
            matrices[name][e['label']][e[name]['prediction']] += 1
        rows.append({'row': e['row'], 'label': e['label'], 'category': category,
                     'baseline_prediction': a['prediction'], 'candidate_prediction': b['prediction'],
                     'candidate_confidence': b['confidence']})
    threshold = run['models']['candidate']['threshold']
    learned = operating_point(examples, threshold)
    return {'seed': run['seed'], 'n': len(examples), 'changes': changes,
            'confusion': matrices, 'rows': rows,
            'reliability': {name: reliability(examples, name) for name in matrices},
            'learned_policy': learned, 'frontier': threshold_frontier(examples)}


def summarize(report):
    if not isinstance(report, dict) or report.get('version') != 2 or not isinstance(report.get('runs'), list):
        raise ValueError('expected a complete version-2 Release Lab report')
    if not 1 <= len(report['runs']) <= 10:
        raise ValueError('expected 1..10 runs')
    seeds = [r['seed'] for r in report['runs']]
    if any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('run seeds must be distinct integers')
    return {'version': 1, 'source_created_utc': report.get('created_utc'),
            'diagnostics_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'Retrospective diagnostics only. Never fit a final policy on this test-set sweep. No models or release gates are changed. Seed results are not pooled.',
            'runs': [summarize_run(run) for run in report['runs']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.report.stat().st_size > 32_000_000:
            raise ValueError('report exceeds 32 MB')
        result = summarize(json.loads(args.report.read_text()))
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'diagnostics: {error}\n')
    print('Descriptive diagnostics saved:', args.output)


if __name__ == '__main__':
    main()
