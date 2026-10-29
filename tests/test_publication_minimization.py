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


if __name__ == '__main__':
    unittest.main()
