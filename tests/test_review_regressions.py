"""Regression cases from the public-edition review. All data are invented."""
import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
os.environ['SOCIALCOMPASS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from features import ALL_METRICS,extract_features
from features_v2 import _raw_mentions,_build_author_matcher,build_interaction_graph
from parse_slack import parse_slack
from parse_mbada import parse_mbada
from parse_nankani import parse_nankani
from text_features import _signature,build_features,STYLE_NAMES
from analyze_agreement import load_annotator_csvs
import train_ml

class ReviewTests(unittest.TestCase):
    def test_numbered_member_is_one_complete_mention(self):
        self.assertEqual(_raw_mentions('Ask @Member 12 about this.'),['Member 12'])
        frame=pd.DataFrame({'author':['Member 1','Member 12','Member 3'],
            'datetime':pd.to_datetime([0,60000,120000],unit='ms'),
            'body':['Hello','Ready','Please @Member 12 check']})
        graph=build_interaction_graph(frame)
        self.assertTrue(graph.has_edge('Member 3','Member 12'))
        self.assertFalse(graph.has_edge('Member 3','Member 1'))
    def test_wrapped_mentions_are_counted_once(self):
        self.assertEqual(_raw_mentions('@Jane\u2069 and <@Other>'),['Jane','Other'])
        self.assertEqual(_raw_mentions('Write jane@example.org'),[])
    def test_ambiguous_prefix_does_not_pick_first_person(self):
        match=_build_author_matcher(['Alex One','Alex Two'])
        self.assertIsNone(match('Alex'))
        self.assertEqual(match('Alex Two'),'Alex Two')
    def test_base_features_accept_timezone_aware_dates(self):
        frame=pd.DataFrame({'author':['Fictional 1','Fictional 2'],'body':['Hello','Thanks'],
            'datetime':pd.to_datetime([0,60000],unit='ms',utc=True)})
        self.assertEqual(len(extract_features(frame,verbose=False)),2)
    def test_initiation_does_not_use_another_channel_gap(self):
        frame=pd.DataFrame({'author':['Fictional 1','Fictional 2'],'body':['Hello','Thanks'],
            'datetime':pd.to_datetime([0,7200000],unit='ms'),'channel':['one','two']})
        self.assertTrue((extract_features(frame,verbose=False).initiation_rate==0).all())
    def test_slack_channel_scope_numeric_dates_and_dedup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for channel,author,time in [('one','Invented A','1'),('two','Invented B','2')]:
                path=root/channel/'threads'/'sample.xml';path.parent.mkdir(parents=True)
                msg=f'<message conversation_id="same"><user>{author}</user><ts>{time}</ts><text>Hello there</text></message>'
                path.write_text('<root>'+msg+msg+'</root>')
            frame=parse_slack(root)
        self.assertEqual(len(frame),2)
        self.assertEqual(frame.is_reply.sum(),0)
        self.assertEqual(frame.datetime.dt.year.tolist(),[1970,1970])
    def test_missing_slack_inputs_have_clear_error(self):
        with tempfile.TemporaryDirectory() as temp,self.assertRaisesRegex(ValueError,'No Slack XML'):
            parse_slack(temp)
    def test_mbada_preserves_technical_null_prose(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'invented.docx'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('word/document.xml','<w:document><w:body><w:p><w:r><w:t>8:28 PM - Anon 1: Fix the null pointer</w:t></w:r></w:p></w:body></w:document>')
            rows=parse_mbada(path)
        self.assertEqual(rows[0]['body'],'Fix the null pointer')
    def test_cache_key_changes_with_message_content(self):
        self.assertNotEqual(_signature(['set/user'],[['Old']]),_signature(['set/user'],[['New']]))
    def test_empty_text_input_preserves_dimensions(self):
        emb,style=build_features([],style_only=True)
        self.assertIsNone(emb)
        self.assertEqual(style.shape,(0,len(STYLE_NAMES)))
    def test_training_rejects_invalid_and_duplicate_labels(self):
        features=pd.DataFrame([{'author':'Invented','dataset':'synthetic',**dict.fromkeys(ALL_METRICS,.5)}])
        valid={'user_id':'Invented','dataset':'synthetic','confidence':'medium',**dict.fromkeys(train_ml.ARCH_KEYS,.2)}
        for labels in ([{**valid,'bee':-.2,'ant':.6}],[valid,valid]):
            with self.subTest(n=len(labels)),patch('train_ml.pd.read_json',side_effect=[features,pd.DataFrame(labels)]),self.assertRaises(ValueError):
                train_ml.load_training_matrix()
    def test_annotation_rejects_negative_zero_and_duplicates(self):
        for scores,duplicate in [([-10,40,30,20,20],False),([0]*5,False),([20]*5,True)]:
            with self.subTest(scores=scores,duplicate=duplicate),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'invented.csv';fields=['annotator','user_id','dataset']+train_ml.ARCH_KEYS
                row={'annotator':'Rater 1','user_id':'Invented','dataset':'invented',**dict(zip(train_ml.ARCH_KEYS,scores))}
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerow(row)
                    if duplicate:writer.writerow(row)
                with self.assertRaises(ValueError):load_annotator_csvs(temp)

    def test_phone_aliases_do_not_merge_matching_suffixes(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'invented.txt'
            path.write_text('26/01/2020, 4:19 pm - 10001234: Hello there\n'
                            '26/01/2020, 4:20 pm - 20001234: Thanks very much\n')
            with contextlib.redirect_stdout(io.StringIO()):frame=parse_nankani(path)
        self.assertEqual(frame.author.nunique(),2)
        self.assertEqual(set(frame.author),{'phone_user_001','phone_user_002'})

if __name__=='__main__':unittest.main()
