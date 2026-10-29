"""Publication checks use invented rater metrics only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import publish_results


class PublicationMinimization(unittest.TestCase):
    def test_export_omits_raters_and_preserves_combined_values(self):
        original = {
            'annotators': ['invented-a', 'invented-b'],
            'pairwise': {'a_vs_model': {'soft_accuracy': .7}},
            'n_common': 10,
            'headline': {'human_vs_opus': {'soft_accuracy': .71}},
            'krippendorff': {'humans_only': {'interval_avg': .42}},
            'fleiss_kappa_all_raters': .25,
            'by_opus_confidence': {'low': {'n_users': 1}},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'private'
            source.mkdir()
            (source / 'ml_results.json').write_text('{"n_train":10}')
            (source / 'annotation_agreement.json').write_text(json.dumps(original))
            output = root / 'public'
            with patch.object(publish_results.config, 'RESULTS', source):
                self.assertEqual(publish_results.main(['--public-dir', str(output)]), 0)
            result = json.loads((output / 'annotation_agreement.json').read_text())
            self.assertNotIn('annotators', result)
            self.assertNotIn('pairwise', result)
            self.assertNotIn('by_opus_confidence', result)
            for key in ('n_common', 'headline', 'krippendorff', 'fleiss_kappa_all_raters'):
                self.assertEqual(result[key], original[key])
            self.assertEqual(json.loads((source / 'annotation_agreement.json').read_text()), original)

    def test_checker_rejects_individual_metrics_even_without_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'ml_results.json').write_text('{"n_train":10}')
            (root / 'annotation_agreement.json').write_text(
                '{"pairwise":{"1_vs_model":{"soft_accuracy":0.7}}}')
            errors = publish_results.audit_public_directory(root)
            self.assertTrue(any('annotation fields' in item for item in errors))

    def test_nested_rater_keys_are_removed_at_every_depth(self):
        record = {
            'n_common': 5,
            'headline': {
                'rater_1_vs_model': {'kappa': .1}, '1_vs_2': {'kappa': .3},
                'human_vs_opus': {'soft_accuracy': .7, '1': {'score': .9}},
            },
            'krippendorff': {'humans_only': {
                'interval_per_dim': {'bee': .1, 'rater_2': .9},
                'interval_avg': .2, '2': {'score': .4},
            }},
        }
        expected = {
            'n_common': 5, 'headline': {'human_vs_opus': {'soft_accuracy': .7}},
            'krippendorff': {'humans_only': {
                'interval_per_dim': {'bee': .1}, 'interval_avg': .2}},
        }
        self.assertEqual(publish_results.aggregate_annotation_agreement(record), expected)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'ml_results.json').write_text('{"n_train":5}')
            (root / 'annotation_agreement.json').write_text(json.dumps(record))
            self.assertTrue(publish_results.audit_public_directory(root))
            (root / 'annotation_agreement.json').write_text(json.dumps(expected))
            self.assertEqual(publish_results.audit_public_directory(root), [])

    def test_objects_and_nonfinite_values_cannot_hide_in_metric_slots(self):
        for value in ({'1': .7}, [.7], 'invented rater', True, float('nan'), float('inf')):
            with self.subTest(value=value):
                record = {'headline': {'human_vs_opus': {'soft_accuracy': value}}}
                with self.assertRaises(ValueError):
                    publish_results.aggregate_annotation_agreement(record)
        for value in (-1, 1.5, True, {'1': 5}):
            with self.assertRaises(ValueError):
                publish_results.aggregate_annotation_agreement({'n_common': value})

    def test_export_filters_nested_fields_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'private'
            source.mkdir()
            (source / 'ml_results.json').write_text('{"n_train":5}')
            record = {'n_common': 5, 'headline': {
                'rater_1_vs_model': {'kappa': .1},
                'human_vs_human': {'cohen_kappa': .3, '2': .9}}}
            (source / 'annotation_agreement.json').write_text(json.dumps(record))
            output = root / 'public'
            with patch.object(publish_results.config, 'RESULTS', source):
                self.assertEqual(publish_results.main(['--public-dir', str(output)]), 0)
            result = json.loads((output / 'annotation_agreement.json').read_text())
            self.assertEqual(result, {'n_common': 5, 'headline': {
                'human_vs_human': {'cohen_kappa': .3}}})


if __name__ == '__main__':
    unittest.main()
