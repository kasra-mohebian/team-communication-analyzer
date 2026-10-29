import os
import sys
import unittest
from pathlib import Path
os.environ["SOCIALCOMPASS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opus_labeler import format_for_opus

class PromptPrivacy(unittest.TestCase):
    def test_identifier_and_behaviour_metadata_are_not_in_prompt(self):
        text = format_for_opus("Invented Full Name", [
            {"body": "A fictional update", "is_reply": 1},
            {"body": "Another fictional update", "is_reply": 0}])
        self.assertNotIn("Invented Full Name", text)
        self.assertNotIn("Total messages", text)
        self.assertNotIn("[R]", text)
        self.assertNotIn("[I]", text)
        self.assertIn("U0001", text)
        self.assertIn("A fictional update", text)

    def test_selection_and_content_limits(self):
        text = format_for_opus("ignored", [{"body": "old"}, {"body": "x" * 400}], max_n=1)
        self.assertNotIn("old", text)
        self.assertIn("x" * 300, text)
        self.assertNotIn("x" * 301, text)
        with self.assertRaises(ValueError):
            format_for_opus("ignored", [], max_n=0)
