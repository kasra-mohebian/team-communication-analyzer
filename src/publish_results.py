"""Export a deliberately limited training-wide publication set.

No aliasing of individual profiles is offered here. Research outputs stay local.
The checker inspects actual files; it is not proof of legal clearance/anonymity.
"""
import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
import config

AGGREGATE_FILES = {'ml_results.json', 'annotation_agreement.json', 'eval_vs_opus.json',
                   'feature_weights.json', 'feature_weights_perarch.json'}
ML_FIELDS = {'n_train', 'class_counts', 'cv', 'best_model', 'fusion_metric_w',
             'fusion_w_source', 'fusion_w_cv_argmin', 'fusion_w_selection',
             'single_view_best', 'per_class_f1', 'per_class_f1_model',
             'single_view_per_class_f1', 'kmeans', 'feature_importance'}
FORBIDDEN_FIELDS = {'author', 'person', 'member', 'user', 'user_id', 'per_person',
                    'body', 'transcript', 'message_text', 'raw_text', 'messages',
                    'edges', 'nodes', 'holdout_reference', 'domain_shift_d',
                    'domain_shift_flagged', 'annotator', 'annotators', 'rater', 'raters'}
SCOPE = ('Selected training-wide summary. Individual/case-study fields omitted; '
         'retained numerical values unchanged.')

def aggregate_ml_results(record):
    """Select known training-wide fields; never alter their numerical values."""
    return {**{k: v for k, v in record.items() if k in ML_FIELDS},
            '_publication_scope': SCOPE}

AGREEMENT_METRICS = dict.fromkeys(
    ('soft_accuracy', 'cosine', 'jsd', 'hard_agreement', 'cohen_kappa'), 'number')
KRIPPENDORFF_METRICS = {
    'interval_per_dim': dict.fromkeys(('bee', 'ant', 'butterfly', 'capybara', 'leech'), 'number'),
    'interval_avg': 'number', 'nominal_argmax': 'number',
}
ANNOTATION_SCHEMA = {
    'n_common': 'count',
    'headline': {'human_vs_opus': AGREEMENT_METRICS, 'human_vs_human': AGREEMENT_METRICS},
    'krippendorff': {'with_opus': KRIPPENDORFF_METRICS, 'humans_only': KRIPPENDORFF_METRICS},
    'fleiss_kappa_all_raters': 'number',
}
ANNOTATION_FIELDS = set(ANNOTATION_SCHEMA)

def _select_annotation(node, schema, path='annotation_agreement'):
    if schema == 'count':
        if type(node) is not int or node < 0:
            raise ValueError(f'{path}: non-negative integer required')
        return node
    if schema == 'number':
        if type(node) not in (int, float) or not math.isfinite(node):
            raise ValueError(f'{path}: finite numeric metric required')
        return node
    if not isinstance(node, dict):
        raise ValueError(f'{path}: aggregate object required')
    return {key: _select_annotation(value, schema[key], f'{path}.{key}')
            for key, value in node.items() if key in schema}

def aggregate_annotation_agreement(record):
    """Retain combined agreement, never rater identifiers or individual scores."""
    return _select_annotation(record, ANNOTATION_SCHEMA)

def audit_public_directory(folder, known_names=()):
    folder = Path(folder)
    if not folder.is_dir():
        return ['public directory does not exist']
    errors = []
    if not (folder / 'ml_results.json').is_file():
        errors.append('missing required artifact: ml_results.json')
    contact = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|/(?:Users|home)/')
    def visit(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FORBIDDEN_FIELDS:
                    errors.append(f'{where}: individual/raw field {key}')
                visit(value, where)
        elif isinstance(node, list):
            for value in node:
                visit(value, where)
        elif isinstance(node, str) and re.search(r'\bMember\s+[A-Z0-9]+\b', node):
            errors.append(f'{where}: member-level narrative')
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(folder).as_posix()
        if name not in AGGREGATE_FILES | {'README.md'}:
            errors.append(f'{name}: not in publication allowlist')
            continue
        text = path.read_text(encoding='utf-8-sig')
        if contact.search(text):
            errors.append(f'{name}: contact/local-path pattern')
        for real in known_names:
            if re.search(r'\b' + re.escape(real) + r'\b', text, re.I):
                errors.append(f'{name}: configured private identifier')
        if path.suffix == '.json':
            try:
                obj = json.loads(text)
                if not isinstance(obj, dict):
                    errors.append(f'{name}: aggregate object required')
                else:
                    visit(obj, name)
                    if name == 'annotation_agreement.json':
                        try:
                            if aggregate_annotation_agreement(obj) != obj:
                                errors.append(f'{name}: individual or unexpected annotation fields')
                        except ValueError as error:
                            errors.append(f'{name}: invalid annotation schema: {error}')
                    if name == 'ml_results.json' and set(obj) - ML_FIELDS - {'_publication_scope'}:
                        errors.append(f'{name}: unexpected summary fields')
            except ValueError:
                errors.append(f'{name}: invalid JSON')
    return errors

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--public-dir', type=Path)
    args = parser.parse_args(argv)
    target = args.public_dir or (config.HISTORICAL_PUBLIC if args.check else config.PUBLIC)
    team = config.load_team()
    known = () if team['is_template'] else team['alias_of'].keys()
    if args.check:
        errors = audit_public_directory(target, known)
    else:
        source = Path(config.RESULTS)
        if not (source / 'ml_results.json').is_file():
            parser.error('missing training-wide ml_results.json')
        if target.exists():
            parser.error('destination exists; choose a fresh --public-dir')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.aggregate-', dir=target.parent) as temp:
            staging = Path(temp)
            for name in sorted(AGGREGATE_FILES):
                path = source / name
                if not path.is_file():
                    continue
                obj = json.loads(path.read_text())
                if name == 'ml_results.json':
                    obj = aggregate_ml_results(obj)
                elif name == 'annotation_agreement.json':
                    obj = aggregate_annotation_agreement(obj)
                (staging / name).write_text(json.dumps(obj, indent=2) + '\n')
            errors = audit_public_directory(staging, known)
            if not errors:
                shutil.copytree(staging, target)
    for error in errors:
        print('FAIL:', error)
    if not errors:
        print('Aggregate publication checks passed:', target)
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(main())
