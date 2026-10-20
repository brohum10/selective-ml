import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile

import diagnostics as d


def example(row, label, baseline, candidate, confidence):
    return {'row': row, 'label': label,
            'baseline': {'prediction': baseline, 'confidence': .8},
            'candidate': {'prediction': candidate, 'confidence': confidence}}


def fixture():
    return [example(8, 0, 0, 0, .9), example(2, 1, 0, 1, .7),
            example(5, 2, 2, 3, .7), example(1, 4, 5, 6, .3)]


def report():
    return {'version':2, 'runs':[{'seed':42, 'n':4, 'examples':fixture(),
                                 'models':{'candidate':{'threshold':.7}}}]}


class DiagnosticsTests(unittest.TestCase):
    def test_change_categories_partition_every_row_without_mutating_input(self):
        original = report()
        before = copy.deepcopy(original)
        result = d.summarize(original)['runs'][0]
        self.assertEqual(result['changes'], {'both_correct':1, 'improved':1, 'regressed':1, 'both_wrong':1})
        self.assertEqual([r['row'] for r in result['rows']], [1,2,5,8])
        self.assertEqual(original, before)

    def test_confusion_matrices_have_correct_orientation_and_totals(self):
        result = d.summarize(report())['runs'][0]
        self.assertEqual(result['confusion']['candidate'][2][3], 1)
        self.assertEqual(result['confusion']['baseline'][2][2], 1)
        for matrix in result['confusion'].values():
            self.assertEqual(sum(map(sum, matrix)), 4)

    def test_ties_move_together_and_threshold_equality_accepts(self):
        rows = fixture()
        at = d.operating_point(rows, .7)
        self.assertEqual((at['accepted'], at['accepted_errors']), (3,1))
        curve = d.threshold_frontier(rows)
        self.assertEqual([p['accepted'] for p in curve], [4,3,1,0])
        self.assertTrue(all(p['accepted'] != 2 for p in curve))

    def test_prefix_sum_frontier_matches_brute_force_at_every_boundary(self):
        rng = random.Random(42)
        rows = [example(i, i%10, i%10, rng.randrange(10), rng.choice([0,.2,.5,.8,1])) for i in range(200)]
        for point in d.threshold_frontier(rows):
            self.assertEqual(point, d.operating_point(rows, point['threshold']))

    def test_confidence_one_is_not_artificially_deferred(self):
        rows = [example(0, 1, 1, 2, 1)]
        self.assertEqual(d.threshold_frontier(rows)[-1]['accepted'], 1)
        self.assertEqual(d.operating_point(rows, 1)['accepted_errors'], 1)

    def test_empty_acceptance_does_not_claim_perfect_accuracy(self):
        point = d.operating_point(fixture(), 1)
        self.assertEqual(point['accepted'], 0)
        self.assertIsNone(point['accepted_accuracy'])
        self.assertEqual((point['deferred_errors'], point['deferred_correct']), (2,2))

    def test_reliability_bins_include_endpoints_and_preserve_empty_bins(self):
        rows = [example(0,0,0,0,0),example(1,0,0,1,.1),example(2,0,0,0,1)]
        result = d.reliability(rows, 'candidate')
        self.assertEqual([result['bins'][i]['count'] for i in [0,1,9]], [1,1,1])
        self.assertIsNone(result['bins'][2]['accuracy'])
        self.assertIsNone(result['bins'][2]['mean_confidence'])
        self.assertAlmostEqual(result['ece'], 1.1/3)

    def test_invalid_cases_rejected(self):
        for field, value in [('confidence',float('nan')),('confidence',float('inf')),('confidence',-1),('confidence',True),('prediction',10),('prediction',True)]:
            rows = fixture()
            rows[0]['candidate'][field] = value
            with self.assertRaises(ValueError): d.validate_examples(rows)
        for rows in [[],fixture()*2,[example(i,0,0,0,.5) for i in range(5001)]]:
            with self.assertRaises(ValueError): d.validate_examples(rows)

    def test_invalid_thresholds_and_run_contracts_rejected(self):
        for threshold in [-.1,1.1,float('nan'),True]:
            with self.assertRaises(ValueError): d.operating_point(fixture(),threshold)
        for mutate in [lambda r:r.update(version=1),lambda r:r['runs'][0].update(n=5),lambda r:r['runs'].append(copy.deepcopy(r['runs'][0]))]:
            value=report(); mutate(value)
            with self.assertRaises(ValueError): d.summarize(value)

    def test_actual_recorded_evidence_matches_frozen_policies(self):
        with zipfile.ZipFile(Path(__file__).parent/'media/release-lab-evidence.zip') as archive:
            recorded=json.loads(archive.read('summary.json'))
        result=d.summarize(recorded)
        for run, original in zip(result['runs'], recorded['runs'], strict=True):
            learned=run['learned_policy']
            for name in ['accepted','deferred','accepted_errors','deferred_errors','deferred_correct','coverage','accepted_accuracy']:
                self.assertEqual(learned[name], original['policy']['candidate'][name])
            self.assertEqual(sum(run['changes'].values()),275)
            self.assertEqual(run['changes']['improved']-run['changes']['regressed'],round((original['candidate_accuracy']-original['baseline_accuracy'])*275))

    def test_cli_emits_diagnostics_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source,target=Path(folder)/'report.json',Path(folder)/'diagnostics.json'
            source.write_text(json.dumps(report()))
            args=[sys.executable,str(Path(d.__file__)), '--report',str(source),'--output',str(target)]
            self.assertEqual(subprocess.run(args,capture_output=True).returncode,0)
            self.assertEqual(json.loads(target.read_text())['runs'][0]['n'],4)
            before=target.read_bytes()
            self.assertEqual(subprocess.run(args,capture_output=True).returncode,2)
            self.assertEqual(target.read_bytes(),before)


if __name__ == '__main__': unittest.main()
