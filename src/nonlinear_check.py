"""Validation-tuned nonlinear control for the same 13 linguistic features."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
import pandas as pd

from .research_benchmark import user_inputs, metrics
from .reviewer_checks import select_threshold, threshold_metrics, paired_thresholds


def fit_validation_only(train_features, train_labels, val_features, val_labels):
    candidates = []
    for leaves in [15, 31]:
        for rate in [.05, .1]:
            model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=leaves,
                       learning_rate=rate, l2_regularization=1., early_stopping=False, random_state=42)
            model.fit(train_features, train_labels)
            probability = model.predict_proba(val_features)[:, 1]
            score = float(f1_score(val_labels, probability >= .5, average='macro'))
            candidates.append((score, model, probability, {'max_leaf_nodes': leaves, 'learning_rate': rate}))
    score, model, probability, params = max(candidates, key=lambda x: x[0])
    return model, {'parameters': params, 'validation_macro_f1_at_05': score,
                   **select_threshold(val_labels, probability)}


def run(private_root, reviewer_output, output, bootstrap=1000):
    root, cache, output = Path(private_root), Path(reviewer_output), Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo):
        raise ValueError('Keep fitted models and individual probabilities outside Git')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    reviewer = json.loads((cache/'report.json').read_text())
    result = {'status': 'post_hoc_exploratory_nonlinear_control', 'comparisons': {},
              'protocol_sha256': hashlib.sha256((repo/'research/REVIEWER_CHECK_PROTOCOL.md').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'bootstrap': bootstrap, 'seed': 42}
    for name in ['mentalhealth', 'depression']:
        print(f'Extracting March training features: {name}', flush=True)
        csv = root/f'pilot-{name}-data/posts.csv'
        if hashlib.sha256(csv.read_bytes()).hexdigest() != reviewer['comparisons'][name]['input_sha256']:
            raise ValueError('Input changed since frozen inference verification')
        posts = pd.read_csv(csv, dtype={'author': str})
        numeric, _, meta = user_inputs(posts[posts['split'].eq('train')])
        retained = joblib.load(cache/f'{name}_private_inputs.joblib')
        development = retained['metadata']['split'].eq('val')
        test = retained['metadata']['split'].eq('test')
        if set(meta.index) & set(retained['metadata'].index):
            raise ValueError('Training authors overlap validation/test authors')
        model, selection = fit_validation_only(numeric.to_numpy(), meta['label'].to_numpy(),
                        retained['numeric'].loc[development].to_numpy(),
                        retained['metadata'].loc[development, 'label'].to_numpy())
        p = model.predict_proba(retained['numeric'].loc[test].to_numpy())[:, 1]
        labels = retained['metadata'].loc[test, 'label'].to_numpy()
        threshold = selection['threshold']
        section = {'input_sha256': reviewer['comparisons'][name]['input_sha256'], 'selection': selection,
                   'test_at_05': metrics(labels, p), 'test_at_validation_threshold': threshold_metrics(labels, p, threshold)}
        for other in ['linguistic_13', 'tfidf_lr']:
            other_t = reviewer['comparisons'][name]['threshold_selection'][other]['threshold']
            section[f'nonlinear_minus_{other}'] = paired_thresholds(labels,
                        retained['probabilities'][other][test], other_t, p, threshold, bootstrap)
        joblib.dump(model, output/f'{name}_nonlinear.joblib')
        pd.DataFrame({'label': labels, 'nonlinear_probability': p}).to_csv(output/f'{name}_private_predictions.csv', index=False)
        section['model_sha256'] = hashlib.sha256((output/f'{name}_nonlinear.joblib').read_bytes()).hexdigest()
        section['private_predictions_sha256'] = hashlib.sha256((output/f'{name}_private_predictions.csv').read_bytes()).hexdigest()
        result['comparisons'][name] = section
        print(name, json.dumps({'selection': selection, 'test_macro_f1_at_05': section['test_at_05']['macro_f1'],
                                'test_macro_f1_at_validation_threshold': section['test_at_validation_threshold']['macro_f1']}), flush=True)
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for p in output.iterdir():
        p.chmod(0o600)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root', required=True)
    p.add_argument('--reviewer-output', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    run(a.private_root, a.reviewer_output, a.output, a.bootstrap)


if __name__ == '__main__':
    main()
