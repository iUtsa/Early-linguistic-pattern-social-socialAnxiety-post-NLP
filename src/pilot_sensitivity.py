"""Fixed exploratory matching and lexical keyword checks on frozen pilot models."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.keyword_lists import LEAK_TERMS
from .preprocess import preprocess_corpus
from .research_benchmark import intervals, metrics, paired_comparison


MODELS = ['majority', 'length_only', 'pronoun_only', 'without_sentiment', 'linguistic_13', 'tfidf_lr']
MASK_TERMS = LEAK_TERMS | {'depression', 'depressed', 'depressive', 'mentalhealth', 'suicidewatch', 'subreddit'}


def matched_positions(users, seed=42):
    if not users.index.is_unique:
        raise ValueError('Matching requires one row per observed author')
    users = users.copy()
    users['length_bin'] = pd.cut(users['mean_words'], [0,50,100,150,250,500,1000,np.inf], right=False, labels=False)
    users['activity_bin'] = users['posts'].clip(upper=3)
    users['_order'] = [hashlib.sha256(f'{seed}|{a}'.encode()).hexdigest() for a in users.index]
    users['_position'] = np.arange(len(users))
    positions, strata = [], []
    for (length, activity), group in users.groupby(['length_bin', 'activity_bin'], sort=True):
        counts = group['label'].value_counts()
        n = min(int(counts.get(0,0)), int(counts.get(1,0)))
        strata.append({'length_bin': int(length), 'activity_bin': int(activity),
                       'available_negative': int(counts.get(0,0)), 'available_positive': int(counts.get(1,0)), 'kept_per_class': n})
        for label in [0,1]:
            positions.extend(group[group['label'].eq(label)].sort_values('_order').head(n)['_position'].tolist())
    positions = np.sort(np.asarray(positions, dtype=int))
    if len(np.unique(positions)) != len(positions) or not len(positions):
        raise ValueError('Matched positions must be nonempty and unique')
    if users.iloc[positions]['label'].mean() != .5:
        raise ValueError('Matched cohort must contain equal class counts')
    return positions, strata


def standardized_difference(users, column):
    a = users.loc[users['label'].eq(1), column].to_numpy(dtype=float)
    b = users.loc[users['label'].eq(0), column].to_numpy(dtype=float)
    denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1))/2)
    return float((a.mean()-b.mean())/denominator) if denominator else None


def run(data_dir, run_dir, output, n_bootstrap=1000, seed=42):
    data_dir, run_dir, output = Path(data_dir), Path(run_dir), Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo):
        raise ValueError('Keep individual sensitivity artifacts outside Git')
    primary = json.loads((run_dir/'report.json').read_text())
    if primary['status'] != 'observational_proxy_experiment' or primary['seed'] != seed:
        raise ValueError('Use the specified empirical pilot and seed')
    if hashlib.sha256((data_dir/'posts.csv').read_bytes()).hexdigest() != primary['input_sha256']:
        raise ValueError('Prepared data changed since the primary run')
    for name in ['full_predictions.csv', 'full_tfidf_lr.joblib']:
        if hashlib.sha256((run_dir/name).read_bytes()).hexdigest() != primary['artifact_sha256'][name]:
            raise ValueError('Primary artifact checksum failed')
    posts = pd.read_csv(data_dir/'posts.csv', dtype={'author':str})
    posts = posts[posts['split'].eq('test')].copy()
    posts['_words'] = posts['text'].str.split().str.len()
    users = posts.groupby('author', sort=True).agg(label=('label','first'), posts=('text','size'), mean_words=('_words','mean'))
    predictions = pd.read_csv(run_dir/'full_predictions.csv')
    expected_ids = [f'user_{i:06d}' for i in range(len(users))]
    if predictions['author_id'].tolist() != expected_ids or not np.array_equal(predictions['label'], users['label']):
        raise ValueError('Primary predictions do not align with sorted test authors')
    positions, strata = matched_positions(users, seed)
    matched = users.iloc[positions]
    matched_predictions = predictions.iloc[positions].copy()
    y = matched_predictions['label'].to_numpy()
    matching = {'n_users': len(matched), 'source_test_users':len(users), 'positive_prevalence': float(y.mean()),
                'strata':strata, 'metrics':{}, 'ci95':{},
                'standardized_mean_differences': {stage:{column:standardized_difference(frame,column) for column in ['mean_words','posts']}
                                                for stage,frame in [('before',users),('after',matched)]}}
    for model in MODELS:
        matching['metrics'][model] = metrics(y, matched_predictions[model].to_numpy())
    for model in ['linguistic_13','tfidf_lr']:
        print(f'Matched author bootstrap: {model}', flush=True)
        matching['ci95'][model] = intervals(y, matched_predictions[model].to_numpy(), n_bootstrap, seed)
    matching['tfidf_minus_linguistic_13'] = paired_comparison(y, matched_predictions['linguistic_13'].to_numpy(),
                                                              matched_predictions['tfidf_lr'].to_numpy(), n_bootstrap, seed)
    print('Frozen TF-IDF keyword sensitivity', flush=True)
    texts = preprocess_corpus(posts['text'].tolist(), show_progress=False)
    pattern = r'\b(?:'+'|'.join(re.escape(t) for t in sorted(MASK_TERMS, key=len, reverse=True))+r')\b'
    texts = [re.sub(pattern,'term',text,flags=re.IGNORECASE) for text in texts]
    joined = pd.Series(texts,index=posts.index).groupby(posts['author'],sort=True).agg(' '.join)
    if not joined.index.equals(users.index):
        raise ValueError('Masked text alignment failed')
    model = joblib.load(run_dir/'full_tfidf_lr.joblib')
    probability = model.predict_proba(joined.to_numpy())[:,1]
    full_y = predictions['label'].to_numpy()
    lexical = {'mask_terms': sorted(MASK_TERMS), 'replacement':'term', 'frozen_model': True,
               'metrics': metrics(full_y,probability),
               'paired_delta':paired_comparison(full_y,predictions['tfidf_lr'].to_numpy(),probability,n_bootstrap,seed)}
    output.mkdir(parents=True,mode=0o700,exist_ok=False)
    matched_predictions.to_csv(output/'matched_predictions.csv',index=False)
    predictions[['author_id','label']].assign(tfidf_lr_masked=probability).to_csv(output/'lexical_masked_predictions.csv',index=False)
    report={'status':'exploratory_frozen_model_sensitivity','seed':seed,'bootstrap_replicates':n_bootstrap,
            'protocol_sha256':hashlib.sha256((repo/'research/UPLOADED_PILOT_SECONDARY_PROTOCOL.md').read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'primary_report_sha256':hashlib.sha256((run_dir/'report.json').read_bytes()).hexdigest(),
            'matched_test':matching,'lexical_masked_test':lexical,
            'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for p in output.iterdir():p.chmod(0o600)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=run(args.data_dir,args.run_dir,args.output)
    print(json.dumps({'matched_users':report['matched_test']['n_users'],
                      'masked_lexical_macro_f1':report['lexical_masked_test']['metrics']['macro_f1']}))


if __name__=='__main__':main()
