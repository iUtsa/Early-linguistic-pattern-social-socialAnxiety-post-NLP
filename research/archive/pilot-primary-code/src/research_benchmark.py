"""Author-level baseline experiments with validation-only tuning and paired uncertainty.

This module does not certify clinical labels or turn community membership into diagnosis.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from .features import extract_features_batch, get_feature_names
from .preprocess import preprocess_corpus
from .research_validation import validate_author_metadata
from scripts.keyword_lists import LEAK_TERMS


FEATURE_SETS = {
    'length_only': ['char_count', 'word_count', 'avg_word_length'],
    'pronoun_only': ['fp_pronoun_rate', 'fp_pronoun_count'],
    'without_sentiment': get_feature_names()[6:],
    'linguistic_13': get_feature_names(),
}


def audit_posts(posts):
    """Reject invalid identifiers, mixed labels, overlap, and cross-split duplicate text."""
    posts = posts.copy()
    if not {'author', 'text', 'label', 'split'} <= set(posts.columns):
        raise ValueError('CSV requires author, text, label, split columns')
    if posts['author'].isna().any():
        raise ValueError('Null authors cannot be replaced with synthetic users')
    posts['author'] = posts['author'].astype(str).str.strip()
    posts['split'] = posts['split'].replace({'validation': 'val'})
    validate_author_metadata(posts)
    if posts['text'].isna().any() or not posts['text'].map(lambda x: isinstance(x, str)).all():
        raise ValueError('All texts must be non-null strings')
    normalized = posts['text'].map(lambda s: re.sub(r'\s+', ' ', s).strip().casefold())
    if normalized.isin(['', '[deleted]', '[removed]']).any():
        raise ValueError('Blank, deleted or removed text must be excluded by a documented policy')
    fingerprint = normalized.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    if (posts.groupby(fingerprint)['split'].nunique() > 1).any():
        raise ValueError('Identical normalized text occurs across splits; deduplicate before partitioning')
    if 'id' in posts and posts['id'].duplicated().any():
        raise ValueError('Repeated post IDs must be resolved before evaluation')
    users = posts.groupby('author', sort=True).agg(label=('label', 'first'), split=('split', 'first'))
    counts = {}
    for split in ['train', 'val', 'test']:
        subset = users[users['split'] == split]
        if set(subset['label']) != {0, 1}:
            raise ValueError(f'{split} needs independent authors in both classes')
        counts[split] = {'users': len(subset), 'positive_users': int(subset['label'].sum()),
                         'posts': int((posts['split'] == split).sum())}
    return posts, {'split_counts': counts, 'within_split_duplicate_text_rows': int(fingerprint.duplicated().sum()),
                   'mixed_author_labels': 0, 'author_overlap': 0, 'cross_split_duplicate_text': 0}


def metrics(y, probability):
    y = np.asarray(y)
    probability = np.asarray(probability)
    if set(y) != {0, 1}:
        raise ValueError('Both classes are required for comparative binary evaluation')
    if probability.shape != y.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid positive-class probabilities')
    predicted = (probability >= .5).astype(int)
    return {
        'n_users': len(y), 'positive_prevalence': float(y.mean()), 'threshold': .5,
        'accuracy': float(accuracy_score(y, predicted)),
        'macro_f1': float(f1_score(y, predicted, average='macro')),
        'positive_f1': float(f1_score(y, predicted)),
        'positive_precision': float(precision_score(y, predicted, zero_division=0)),
        'positive_recall': float(recall_score(y, predicted, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probability)),
        'average_precision': float(average_precision_score(y, probability)),
        'brier': float(brier_score_loss(y, probability)),
        'log_loss': float(log_loss(y, probability, labels=[0, 1])),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def bootstrap_indices(y, n_bootstrap, seed):
    """One independent author per row; stratification conditions on class prevalence."""
    if n_bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap replicates (1000+ for research)')
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(np.asarray(y) == label) for label in [0, 1]]
    if any(len(s) == 0 for s in strata):
        raise ValueError('Bootstrap requires both classes')
    for _ in range(n_bootstrap):
        yield np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])


def intervals(y, probability, n_bootstrap=1000, seed=42):
    values = {key: [] for key in ['macro_f1', 'positive_f1', 'roc_auc', 'average_precision', 'brier']}
    for idx in bootstrap_indices(y, n_bootstrap, seed):
        m = metrics(np.asarray(y)[idx], np.asarray(probability)[idx])
        for key in values:
            values[key].append(m[key])
    return {key: np.quantile(v, [.025, .975]).tolist() for key, v in values.items()}


def paired_comparison(y, prob_a, prob_b, n_bootstrap=1000, seed=42):
    """Report B minus A on exactly the same ordered test users."""
    y, prob_a, prob_b = np.asarray(y), np.asarray(prob_a), np.asarray(prob_b)
    point_a, point_b = metrics(y, prob_a), metrics(y, prob_b)
    deltas = {key: [] for key in ['macro_f1', 'positive_f1', 'average_precision']}
    for idx in bootstrap_indices(y, n_bootstrap, seed):
        ma, mb = metrics(y[idx], prob_a[idx]), metrics(y[idx], prob_b[idx])
        for key in deltas:
            deltas[key].append(mb[key] - ma[key])
    correct_a = (prob_a >= .5) == y
    correct_b = (prob_b >= .5) == y
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(correct_b & ~correct_a))
    p = float(binomtest(a_only, a_only + b_only, .5).pvalue) if a_only + b_only else 1.
    return {'direction': 'B minus A', 'delta': {k: point_b[k] - point_a[k] for k in deltas},
            'ci95': {k: np.quantile(v, [.025, .975]).tolist() for k, v in deltas.items()},
            'mcnemar_exact': {'a_correct_b_wrong': a_only, 'a_wrong_b_correct': b_only, 'pvalue': p}}


def chronological_posts(posts):
    column = next((c for c in ['created_utc', 'timestamp', 'created_at'] if c in posts), None)
    if column is None:
        raise ValueError('Prefix experiments require observed timestamps; posts_seen alone is insufficient')
    parsed = pd.to_datetime(posts[column], unit='s', utc=True, errors='coerce') if pd.api.types.is_numeric_dtype(posts[column]) else pd.to_datetime(posts[column], utc=True, errors='coerce')
    if parsed.isna().any():
        raise ValueError('All prefix-experiment timestamps must parse')
    result = posts.assign(_observed_time=parsed)
    if result.duplicated(['author', '_observed_time']).any():
        raise ValueError('Tied timestamps require a predefined deterministic ordering policy')
    result = result.sort_values(['author', '_observed_time']).reset_index(drop=True)
    result['_prefix_position'] = result.groupby('author').cumcount() + 1
    return result


def user_inputs(posts, mask_keywords=False):
    posts = posts.copy()
    texts = preprocess_corpus(posts['text'].tolist(), show_progress=False)
    if mask_keywords:
        # Neutral replacement, applied before ALL feature extraction.
        pattern = r'\b(?:' + '|'.join(re.escape(t) for t in sorted(LEAK_TERMS, key=len, reverse=True)) + r')\b'
        texts = [re.sub(pattern, 'term', text, flags=re.IGNORECASE) for text in texts]
    post_features = extract_features_batch(texts, show_progress=False)
    post_features['author'] = posts['author'].to_numpy()
    numeric = post_features.groupby('author', sort=True)[get_feature_names()].mean()
    metadata = posts.groupby('author', sort=True).agg(label=('label', 'first'), split=('split', 'first'))
    joined = pd.Series(texts, index=posts.index).groupby(posts['author'], sort=True).agg(' '.join)
    assert list(numeric.index) == list(metadata.index) == list(joined.index)
    return numeric, joined, metadata


def fit_baselines(numeric, texts, metadata, seed=42):
    """Select C on validation authors only; retain fitted preprocessing in each artifact."""
    train = metadata['split'].eq('train').to_numpy()
    val = metadata['split'].eq('val').to_numpy()
    test = metadata['split'].eq('test').to_numpy()
    labels = metadata['label'].to_numpy()
    results, models, predictions, selection = {}, {}, {}, {}
    for name in ['majority', *FEATURE_SETS, 'tfidf_lr']:
        print(f'Fitting baseline: {name}', flush=True)
        X = texts.to_numpy() if name == 'tfidf_lr' else numeric[FEATURE_SETS.get(name, ['word_count'])].to_numpy()
        candidates = []
        for c in ([None] if name == 'majority' else [.1, 1., 10.]):
            if name == 'majority':
                model = DummyClassifier(strategy='most_frequent', random_state=seed)
            elif name == 'tfidf_lr':
                model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000),
                                      LogisticRegression(C=c, max_iter=2000, random_state=seed))
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
            model.fit(X[train], labels[train])
            pred = model.predict(X[val])
            score = float(f1_score(labels[val], pred, average='macro'))
            candidates.append((score, model, c))
        # Grid order breaks ties toward smaller C. No refit with test or validation rows.
        score, model, c = max(candidates, key=lambda item: item[0])
        probability = model.predict_proba(X[test])[:, 1]
        result = metrics(labels[test], probability)
        if name == 'majority':
            # Dummy predictions are 0 or 1, so the common threshold reproduces predict().
            assert np.array_equal(model.predict(X[test]), (probability >= .5).astype(int))
        results[name], models[name], predictions[name] = result, model, probability
        selection[name] = {'C': c, 'validation_macro_f1': score}
    return results, models, predictions, selection


def validate_provenance(path):
    provenance = json.loads(Path(path).read_text())
    fields = ['dataset_source', 'dataset_version', 'permission_basis', 'label_definition',
              'ethics_status', 'author_id_origin', 'synthetic_data']
    missing = [k for k in fields if k not in provenance]
    if missing:
        raise ValueError(f'Provenance missing fields: {missing}')
    if any(not isinstance(provenance[k], str) or not provenance[k].strip() for k in fields[:-1]):
        raise ValueError('Document provenance fields with nonempty text')
    if not isinstance(provenance['synthetic_data'], bool):
        raise ValueError('synthetic_data must be a JSON boolean')
    if provenance['author_id_origin'] not in ['observed_stable', 'synthetic_grouping', 'generated_fixture']:
        raise ValueError('author_id_origin must be observed_stable, synthetic_grouping, or generated_fixture')
    if not provenance['synthetic_data'] and provenance['author_id_origin'] != 'observed_stable':
        raise ValueError('Real-data author-level evaluation requires observed_stable author identifiers')
    return provenance


def run(csv, provenance_path, output, ks=(), n_bootstrap=1000, seed=42):
    provenance = validate_provenance(provenance_path)
    posts, audit = audit_posts(pd.read_csv(csv, dtype={'author': str}))
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo):
        raise ValueError('Write private experiment artifacts outside the Git checkout')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'synthetic_software_validation' if provenance['synthetic_data'] else 'observational_proxy_experiment',
              'label_construct': provenance['label_definition'], 'provenance': provenance, 'data_audit': audit,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
              'input_sha256': hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
              'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','vaderSentiment','textblob','emoji']},
              'code_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), repo/'src/features.py', repo/'src/preprocess.py', repo/'scripts/keyword_lists.py']},
              'bootstrap': {'replicates': n_bootstrap, 'unit': 'author', 'method': 'stratified percentile',
                            'conditional_on': 'fitted models, observed split, and class counts'},
              'outcome_scope': 'Label prediction; no diagnosis, onset, screening utility, or causal interpretation',
              'analysis_notes': ['User features are post-level means; TF-IDF uses concatenated user text.',
                                 'Primary metric: macro-F1; primary comparator: TF-IDF LR versus linguistic_13.',
                                 'Paired comparisons and intervals are per-comparison, exploratory, and not multiplicity-adjusted.',
                                 'Keyword masking tests a fixed list, not independence from all topic vocabulary.'],
              'experiments': {}}
    try:
        report['git_commit'] = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        report['git_dirty'] = bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        report['git_commit'] = None
    cohort = posts
    if ks:
        if any(k < 1 for k in ks):
            raise ValueError('Prefix sizes must be positive')
        cohort = chronological_posts(posts)
        eligible = cohort.groupby('author').size().loc[lambda n: n >= max(ks)].index
        cohort = cohort[cohort['author'].isin(eligible)].copy()
        cohort, common_audit = audit_posts(cohort)
        report['common_cohort_audit'] = common_audit
        report['cohort_selection'] = 'Users in both classes with at least max(k) observed posts; future-activity selection limits population claims'
    test_authors = sorted(cohort.loc[cohort['split'].eq('test'), 'author'].unique())
    ids = {a: f'user_{i:06d}' for i,a in enumerate(test_authors)}
    all_predictions = {}
    for horizon in ['full', *sorted(set(ks))]:
        selected = cohort if horizon == 'full' else cohort[cohort['_prefix_position'] <= horizon]
        print(f'Extracting features: horizon={horizon}, posts={len(selected)}', flush=True)
        numeric, texts, metadata = user_inputs(selected)
        m, models, probabilities, selection = fit_baselines(numeric, texts, metadata, seed)
        y = metadata.loc[metadata['split'].eq('test'), 'label'].to_numpy()
        horizon_report = {'metrics': m, 'model_selection': selection, 'ci95': {}, 'paired_vs_linguistic_13': {}}
        for name, probability in probabilities.items():
            print(f'Author bootstrap: baseline={name}, replicates={n_bootstrap}', flush=True)
            horizon_report['ci95'][name] = intervals(y, probability, n_bootstrap, seed)
            if name != 'linguistic_13':
                horizon_report['paired_vs_linguistic_13'][name] = paired_comparison(y, probabilities['linguistic_13'], probability, n_bootstrap, seed)
            joblib.dump(models[name], output/f'{horizon}_{name}.joblib')
        # Frozen-model masked-test sensitivity: no retraining or test tuning.
        test = metadata['split'].eq('test').to_numpy()
        print('Frozen-model masked-test sensitivity', flush=True)
        masked, _, masked_meta = user_inputs(selected[selected['split'].eq('test')], mask_keywords=True)
        assert metadata.index[test].equals(masked_meta.index)
        masked_probability = models['linguistic_13'].predict_proba(masked.to_numpy())[:, 1]
        horizon_report['masked_test'] = {'metrics': metrics(y, masked_probability),
                                        'paired_delta': paired_comparison(y, probabilities['linguistic_13'], masked_probability, n_bootstrap, seed),
                                        'replacement': 'term', 'keyword_list': sorted(LEAK_TERMS)}
        ordered_test = metadata.index[test].tolist()
        assert ordered_test == test_authors
        frame = pd.DataFrame({'author_id': [ids[a] for a in ordered_test], 'label': y,
                              **probabilities, 'linguistic_13_masked_test': masked_probability})
        frame.to_csv(output/f'{horizon}_predictions.csv', index=False)
        report['experiments'][str(horizon)] = horizon_report
        all_predictions[horizon] = (y, probabilities)
    if ks:
        report['paired_prefix_vs_full'] = {}
        y, full = all_predictions['full']
        for k in sorted(set(ks)):
            prefix_y, prefix = all_predictions[k]
            assert np.array_equal(y, prefix_y)
            report['paired_prefix_vs_full'][str(k)] = paired_comparison(y, full['linguistic_13'], prefix['linguistic_13'], n_bootstrap, seed)
    report['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ks', nargs='*', type=int, default=[])
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        result = run(args.csv, args.provenance, args.output, args.ks, args.bootstrap, args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Completed {result['status']}; artifacts: {args.output}")


if __name__ == '__main__':
    main()
