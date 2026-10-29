import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["SOCIALCOMPASS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import config
from label_validation import validate_distribution
from message_cleaning import is_system_body
from parse_whatsapp import parse_whatsapp
from features_v2 import build_interaction_graph, compute_interaction_metrics
from publish_results import audit_public_directory, main as check_export, aggregate_ml_results
from demo import run_demo
from train_ml import fusion_per_class_report


class ProbabilityTests(unittest.TestCase):
    def vector(self, first):
        return dict(zip(("bee", "ant", "butterfly", "capybara", "leech"),
                        (first, .5, .2, .2, .2)))

    def test_invalid_probability_cannot_be_hidden_by_total(self):
        for value in (-.1, 1.1, float("nan"), float("inf"), True, "0.2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_distribution(self.vector(value))

    def test_valid_distribution_is_normalized(self):
        row = dict.fromkeys(("bee", "ant", "butterfly", "capybara", "leech"), .198)
        self.assertAlmostEqual(sum(validate_distribution(row).values()), 1)


class ParserTests(unittest.TestCase):
    def test_normal_technical_prose_is_preserved(self):
        for text in ("I left a comment", "Fix the null pointer", "We added tests"):
            self.assertFalse(is_system_body(text))
        self.assertTrue(is_system_body("<Media omitted>"))
        self.assertTrue(is_system_body("This message was deleted"))

    def test_parser_deduplicates_without_dropping_prose(self):
        text = ("01/10/2026, 10:00 - Alpha: I left a comment\n"
                "01/10/2026, 10:01 - Bravo: Fix the null pointer\n"
                "01/10/2026, 10:01 - Bravo: Fix the null pointer\n")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invented.txt"
            path.write_text(text)
            with contextlib.redirect_stdout(io.StringIO()):
                frame = parse_whatsapp(path)
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame.body), {"I left a comment", "Fix the null pointer"})


class GraphTests(unittest.TestCase):
    def frame(self, channels):
        return pd.DataFrame({"author": ["Member A", "Member B"],
                             "body": ["Hello", "Thanks"], "channel": channels,
                             "datetime": pd.to_datetime([0, 60000], unit="ms")})

    def test_unrelated_channels_do_not_create_temporal_replies(self):
        self.assertEqual(build_interaction_graph(self.frame(["one", "two"])).number_of_edges(), 0)

    def test_graph_and_metrics_share_the_reply_definition(self):
        frame = self.frame(["one", "one"])
        graph = build_interaction_graph(frame)
        metrics = compute_interaction_metrics(frame).set_index("author")
        self.assertTrue(graph.has_edge("Member B", "Member A"))
        self.assertEqual(metrics.loc["Member B", "replies_sent"], 1)

    def test_conversation_ids_are_scoped_with_channels(self):
        frame = self.frame(["one", "two"])
        frame["conversation_id"] = "1"
        self.assertEqual(build_interaction_graph(frame).number_of_edges(), 0)

    def test_timezone_aware_timestamps_work(self):
        frame = self.frame(["one", "one"])
        frame["datetime"] = frame["datetime"].dt.tz_localize("UTC")
        self.assertEqual(build_interaction_graph(frame).number_of_edges(), 1)


class ExportTests(unittest.TestCase):
    def test_missing_directory_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(audit_public_directory(Path(temp) / "missing"))

    def test_aliases_do_not_authorize_individual_records(self):
        for field in ("per_person", "author", "user_id", "body", "edges"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                (folder / "ml_results.json").write_text('{"n_train":300}')
                (folder / "annotation_agreement.json").write_text(json.dumps({field: "Member 1"}))
                self.assertTrue(audit_public_directory(folder))

    def test_nested_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "ml_results.json").write_text('{"cv":{"fusion":{"per_person":[]}}}')
            self.assertTrue(audit_public_directory(folder))

    def test_only_training_summary_fields_are_selected(self):
        source = {"n_train": 300, "cv": {"score": .5113864895490434},
                  "domain_shift_d": {"x": 1}, "per_person": [{"author": "Member A"}]}
        result = aggregate_ml_results(source)
        self.assertEqual(result["cv"], source["cv"])
        self.assertNotIn("per_person", result)
        self.assertNotIn("domain_shift_d", result)

    def test_non_allowlisted_files_fail_even_with_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "ml_results.json").write_text('{"n_train":300}')
            self.assertEqual(audit_public_directory(folder), [])
            (folder / "wa_results.json").write_text('[]')
            self.assertTrue(audit_public_directory(folder))

    def test_actual_safe_repository_artifacts_pass(self):
        self.assertEqual(audit_public_directory(ROOT / "outputs/public"), [])


class ModelReportTests(unittest.TestCase):
    def test_per_class_report_uses_fusion_predictions(self):
        class PerfectFusion:
            def __init__(self, weight):
                self.weight = weight
            def fit(self, metrics, style, targets, sample_weight=None):
                return self
            def predict(self, metrics, style):
                return metrics
        values = np.tile(np.eye(5), (4, 1))
        with patch("train_ml.FusionModel", PerfectFusion):
            report = fusion_per_class_report(values, np.zeros((20, 2)), values,
                                             np.ones(20), values.argmax(1), .5,
                                             n_splits=2, n_repeats=1)
        self.assertTrue(all(r["f1_mean"] == 1 for r in report.values()))


class DemoTests(unittest.TestCase):
    def test_complete_offline_feature_demo(self):
        with tempfile.TemporaryDirectory() as temp, patch("socket.create_connection", side_effect=AssertionError("offline")):
            result = run_demo(temp)
            features = json.loads((Path(temp) / "features.json").read_text())
        self.assertEqual((result["messages"], result["members"], result["teams"]), (268, 53, 9))
        self.assertFalse(result["role_model_used"])
        self.assertTrue(all(row["author"].startswith("Member ") for row in features))


if __name__ == "__main__":
    unittest.main()
