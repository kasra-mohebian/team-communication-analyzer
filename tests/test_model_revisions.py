"""Optional model loaders are checked without downloading models."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ['SOCIALCOMPASS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ml_features
import text_features
from model_revisions import EMBEDDING_MODEL, EMBEDDING_REVISION, EMOTION_REVISION, SENTIMENT_REVISION


class ModelRevisionTests(unittest.TestCase):
    def test_lazy_neural_loaders_pass_revision_and_disable_custom_code(self):
        constructor = Mock()
        pipeline = Mock()
        with patch.dict(sys.modules, {
            'sentence_transformers': SimpleNamespace(SentenceTransformer=constructor),
            'transformers': SimpleNamespace(pipeline=pipeline),
        }), patch.object(ml_features, '_sbert', None), patch.object(ml_features, '_emo_pipe', None), patch.object(ml_features, '_sent_pipe', None):
            ml_features._sbert_model()
            ml_features._emotion_pipe()
            ml_features._sentiment_pipe()
        constructor.assert_called_once_with(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, trust_remote_code=False)
        self.assertEqual([c.kwargs['revision'] for c in pipeline.call_args_list],
                         [EMOTION_REVISION, SENTIMENT_REVISION])
        self.assertTrue(all(c.kwargs['trust_remote_code'] is False for c in pipeline.call_args_list))

    def test_embedding_cache_changes_with_model_revision(self):
        original = text_features._signature([('fictional', '1')], [['Hello']])
        with patch.object(text_features, 'EMBEDDING_REVISION', 'a' * 40):
            revised = text_features._signature([('fictional', '1')], [['Hello']])
        self.assertNotEqual(original, revised)
