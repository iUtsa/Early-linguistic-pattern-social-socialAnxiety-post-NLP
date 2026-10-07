"""Post-hoc threshold, comparison-transfer and verbatim-reuse checks.

This workflow uses retained primary models, never fits a new model on June data,
and releases aggregate evidence only. See research/REVIEWER_CHECK_PROTOCOL.md.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from .research_benchmark import bootstrap_indices, metrics, user_inputs


THRESHOLDS = np.arange(5, 96) / 100
MODEL_NAMES = ['linguistic_13', 'tfidf_lr']


def select_threshold(labels, probability):
    """Choose an operating point on validation labels only, with deterministic ties."""
    scores = [(float(f1_score(labels, probability >= t, average='macro')), float(t))
              for t in THRESHOLDS]
    score, threshold = max(scores, key=lambda x: (x[0], -abs(x[1] - .5), -x[1]))
    return {'threshold': threshold, 'validation_macro_f1': score}


def threshold_metrics(labels, probability, threshold):
    result = metrics(labels, probability)
    pred = np.asarray(probability) >= threshold
    result.update(threshold=float(threshold), accuracy=float(np.mean(pred == labels)),
                  macro_f1=float(f1_score(labels, pred, average='macro')),
                  positive_f1=float(f1_score(labels, pred)),
                  positive_precision=float(precision_score(labels, pred, zero_division=0)),
                  positive_recall=float(recall_score(labels, pred, zero_division=0)),
                  confusion_matrix=confusion_matrix(labels, pred, labels=[0, 1]).tolist())
    return result


def paired_thresholds(labels, a, ta, b, tb, bootstrap=1000):
    """Same authors in both arms; fixed thresholds throughout resampling."""
    pa, pb = np.asarray(a) >= ta, np.asarray(b) >= tb
    labels = np.asarray(labels)
    values = [f1_score(labels[i], pb[i], average='macro') -
              f1_score(labels[i], pa[i], average='macro')
              for i in bootstrap_indices(labels, bootstrap, 42)]
    return {'direction': 'B minus A',
            'macro_f1_delta': float(f1_score(labels, pb, average='macro') - f1_score(labels, pa, average='macro')),
            'macro_f1_ci95': np.quantile(values, [.025, .975]).tolist(),
            'bootstrap': bootstrap, 'seed': 42, 'threshold_a': float(ta), 'threshold_b': float(tb)}


def word_shingles(text):
    words = text.split()
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}


def near_duplicates(reference, query, similarity=.8):
    """Find exact five-word-shingle Jaccard >= similarity using prefix filtering.

    The ordering affects efficiency only. Full sets verify every candidate.
    Returns query positions, never raw text or matched author identities.
    """
    if not 0 < similarity <= 1:
        raise ValueError('Similarity must be in (0, 1]')
    refs = [word_shingles(x) for x in reference]
    queries = [word_shingles(x) for x in query]
    counts = Counter()
    for s in [*refs, *queries]:
        counts.update(s)
    inverted = defaultdict(list)
    order = lambda token: (counts[token], token)
    for i, s in enumerate(refs):
        if s:
            length = len(s) - math.ceil(similarity * len(s)) + 1
            for token in sorted(s, key=order)[:length]:
                inverted[token].append(i)
    found, pairs, verified = set(), 0, 0
    for q, s in enumerate(queries):
        if not s:
            continue
        length = len(s) - math.ceil(similarity * len(s)) + 1
        candidates = set()
        for token in sorted(s, key=order)[:length]:
            candidates.update(inverted.get(token, ()))
        for i in candidates:
            other = refs[i]
            if min(len(s), len(other)) < similarity * max(len(s), len(other)):
                continue
            verified += 1
            intersection = len(s & other)
            if intersection / (len(s) + len(other) - intersection) >= similarity:
                found.add(q)
                pairs += 1
    return found, {'matched_query_posts': len(found), 'matched_post_pairs': pairs,
                   'candidate_pairs_verified': verified, 'reference_posts': len(refs),
                   'query_posts': len(queries), 'similarity': similarity,
                   'definition': 'Set Jaccard of consecutive five-word shingles; no semantic-paraphrase coverage'}


def run(private_root, output, bootstrap=1000):
    root, output = Path(private_root).resolve(), Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo):
        raise ValueError('Individual input caches must remain outside Git')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {'status': 'post_hoc_exploratory_reviewer_checks', 'comparisons': {},
              'scope': 'Community classification, not diagnosis; June primary results already inspected',
              'protocol_sha256': hashlib.sha256((repo/'research/REVIEWER_CHECK_PROTOCOL.md').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'bootstrap': bootstrap, 'seed': 42}
    data = {}
    for name in ['mentalhealth', 'depression']:
        print(f'Reconstructing frozen inputs: {name}', flush=True)
        csv = root/f'pilot-{name}-data/posts.csv'
        posts = pd.read_csv(csv, dtype={'author': str})
        numeric, texts, meta = user_inputs(posts[~posts['split'].eq('train')])
        models = {m: joblib.load(root/f'pilot-{name}-seed42/full_{m}.joblib') for m in MODEL_NAMES}
        probabilities, chosen = {}, {}
        val, test = meta['split'].eq('val').to_numpy(), meta['split'].eq('test').to_numpy()
        saved = pd.read_csv(root/f'pilot-{name}-seed42/full_predictions.csv')
        if not np.array_equal(meta.loc[test, 'label'].to_numpy(), saved['label'].to_numpy()):
            raise ValueError('Saved labels do not align with reconstructed test authors')
        for model_name, model in models.items():
            X = texts.to_numpy() if model_name == 'tfidf_lr' else numeric.to_numpy()
            probabilities[model_name] = model.predict_proba(X)[:, 1]
            if not np.allclose(probabilities[model_name][test], saved[model_name].to_numpy(), atol=1e-12, rtol=0):
                raise ValueError('Frozen inference does not reproduce primary predictions')
            chosen[model_name] = select_threshold(meta.loc[val, 'label'].to_numpy(), probabilities[model_name][val])
        y = meta.loc[test, 'label'].to_numpy()
        tests = {m: threshold_metrics(y, probabilities[m][test], chosen[m]['threshold']) for m in MODEL_NAMES}
        result['comparisons'][name] = {
            'input_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(),
            'primary_report_sha256': hashlib.sha256((root/f'pilot-{name}-seed42/report.json').read_bytes()).hexdigest(),
            'frozen_inference_reproduced': True, 'threshold_selection': chosen,
            'validation_threshold_test': tests,
            'lexical_minus_linguistic': paired_thresholds(y, probabilities['linguistic_13'][test], chosen['linguistic_13']['threshold'],
                                                        probabilities['tfidf_lr'][test], chosen['tfidf_lr']['threshold'], bootstrap)}
        data[name] = dict(posts=posts, numeric=numeric, texts=texts, meta=meta,
                          models=models, probabilities=probabilities, chosen=chosen)
        joblib.dump({'numeric': numeric, 'texts': texts, 'metadata': meta, 'probabilities': probabilities}, output/f'{name}_private_inputs.joblib')
        (output/f'{name}_private_inputs.joblib').chmod(0o600)
        print(f'Inference and validation thresholds verified: {name}', flush=True)
    for target, source in [('mentalhealth', 'depression'), ('depression', 'mentalhealth')]:
        t, s = data[target], data[source]
        source_earlier = s['posts'][~s['posts']['split'].eq('test')]
        test_posts = t['posts'][t['posts']['split'].eq('test')]
        excluded = set(source_earlier['author']) & set(test_posts['author'])
        text_reuse = set(test_posts.loc[test_posts['text'].isin(set(source_earlier['text'])), 'author'])
        excluded |= text_reuse
        keep = t['meta']['split'].eq('test') & ~t['meta'].index.isin(excluded)
        y = t['meta'].loc[keep, 'label'].to_numpy()
        transfer = {'source_comparison': source, 'target_comparison': target,
                    'source_train_val_overlap_authors_excluded': len(excluded),
                    'source_train_val_exact_text_authors_excluded': len(text_reuse), 'models': {}}
        for model_name in MODEL_NAMES:
            X = t['texts'].loc[keep].to_numpy() if model_name == 'tfidf_lr' else t['numeric'].loc[keep].to_numpy()
            p = s['models'][model_name].predict_proba(X)[:, 1]
            native = t['probabilities'][model_name][keep.to_numpy()]
            st, tt = s['chosen'][model_name]['threshold'], t['chosen'][model_name]['threshold']
            transfer['models'][model_name] = {
                'native_at_05': threshold_metrics(y, native, .5), 'transfer_at_05': threshold_metrics(y, p, .5),
                'native_at_validation_threshold': threshold_metrics(y, native, tt),
                'transfer_at_source_validation_threshold': threshold_metrics(y, p, st),
                'transfer_minus_native': paired_thresholds(y, native, tt, p, st, bootstrap)}
        result['comparisons'][target]['cross_comparator_transfer'] = transfer
        print(f'Auditing near duplicates: {target}', flush=True)
        reference = t['posts'][~t['posts']['split'].eq('test')]['text'].tolist()
        query = test_posts['text'].tolist()
        positions, audit = near_duplicates(reference, query)
        removed = set(test_posts.iloc[sorted(positions)]['author'])
        retain = t['meta']['split'].eq('test') & ~t['meta'].index.isin(removed)
        labels = t['meta'].loc[retain, 'label'].to_numpy()
        audit['test_authors_excluded'] = len(removed)
        audit['filtered_primary_metrics'] = {m: metrics(labels, t['probabilities'][m][retain.to_numpy()]) for m in MODEL_NAMES}
        result['comparisons'][target]['near_duplicate_audit'] = audit
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (output/'report.json').chmod(0o600)
    print(json.dumps({k: {'thresholds': v['threshold_selection'], 'test_macro_f1':
          {m: n['macro_f1'] for m, n in v['validation_threshold_test'].items()}}
          for k, v in result['comparisons'].items()}, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    run(args.private_root, args.output, args.bootstrap)


if __name__ == '__main__':
    main()
