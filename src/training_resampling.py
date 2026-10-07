"""Stratified training-account resampling; no best-seed selection or primary refit."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .research_benchmark import user_inputs
from .reviewer_checks import select_threshold, threshold_metrics


def resample_training_indices(labels, seed):
    rng=np.random.default_rng(seed)
    labels=np.asarray(labels)
    strata=[np.flatnonzero(labels==k) for k in [0,1]]
    if any(len(s)==0 for s in strata):raise ValueError('Both training classes required')
    return np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])


def summarize(values):
    values=np.asarray(values,dtype=float)
    return {'mean':float(values.mean()),'sd':float(values.std(ddof=1)),
            'minimum':float(values.min()),'maximum':float(values.max()),
            'quantiles_025_50_975':np.quantile(values,[.025,.5,.975]).tolist(),
            'interpretation':'Descriptive distribution across 20 stratified training-account resamples, not a population confidence interval'}


def run(private_root, output):
    root,output=Path(private_root),Path(output).resolve()
    repo=Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo):raise ValueError('Keep individual probabilities and input caches outside Git')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    report={'status':'exploratory_training_account_resampling','seeds':list(range(100,120)),
            'protocol_sha256':hashlib.sha256((repo/'research/TEMPORAL_EXTENSION_PROTOCOL.md').read_bytes()).hexdigest(),
            'code_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),repo/'src/features.py',repo/'src/preprocess.py']},
            'scope':'Same March/May/June cohort and selected C; uncertainty in labels/source collection not addressed',
            'comparisons':{}}
    for comparison in ['mentalhealth','depression']:
        csv=root/f'pilot-{comparison}-data/posts.csv'
        posts=pd.read_csv(csv,dtype={'author':str})
        print(f'Preparing training-account resamples: {comparison}',flush=True)
        numeric,texts,metadata=user_inputs(posts[posts['split'].eq('train')])
        retained=joblib.load(root/f'reviewer-checks-seed42/{comparison}_private_inputs.joblib')
        val=retained['metadata']['split'].eq('val')
        test=retained['metadata']['split'].eq('test')
        if set(metadata.index)&set(retained['metadata'].index):raise ValueError('Training/validation/test account overlap')
        y=metadata['label'].to_numpy()
        yv=retained['metadata'].loc[val,'label'].to_numpy()
        yt=retained['metadata'].loc[test,'label'].to_numpy()
        primary={n:joblib.load(root/f'pilot-{comparison}-seed42/full_{n}.joblib') for n in ['linguistic_13','tfidf_lr']}
        joblib.dump({'numeric':numeric,'texts':texts,'metadata':metadata},output/f'{comparison}_private_training_inputs.joblib')
        rows=[]
        for seed in range(100,120):
            indices=resample_training_indices(y,seed)
            row={'seed':seed,'models':{}}
            predictions=pd.DataFrame({'label':yt})
            for name in ['linguistic_13','tfidf_lr']:
                model=clone(primary[name])
                model.set_params(logisticregression__random_state=seed)
                train_x=texts.to_numpy() if name=='tfidf_lr' else numeric.to_numpy()
                other_x=retained['texts'].to_numpy() if name=='tfidf_lr' else retained['numeric'].to_numpy()
                model.fit(train_x[indices],y[indices])
                p=model.predict_proba(other_x)[:,1]
                selection=select_threshold(yv,p[val])
                row['models'][name]={'C':float(model.get_params()['logisticregression__C']),**selection,
                                    'june':threshold_metrics(yt,p[test],selection['threshold'])}
                predictions[name]=p[test]
            row['lexical_minus_linguistic_macro_f1']=row['models']['tfidf_lr']['june']['macro_f1']-row['models']['linguistic_13']['june']['macro_f1']
            path=output/f'{comparison}_seed{seed}_private_predictions.csv'
            predictions.to_csv(path,index=False);path.chmod(0o600)
            row['private_predictions_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
            print(json.dumps({'comparison':comparison,'seed':seed,'macro_f1':{k:v['june']['macro_f1'] for k,v in row['models'].items()}}),flush=True)
        report['comparisons'][comparison]={'input_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
            'resamples':rows,'summary':{name:summarize([row['models'][name]['june']['macro_f1'] for row in rows]) for name in ['linguistic_13','tfidf_lr']},
            'lexical_advantage_summary':summarize([r['lexical_minus_linguistic_macro_f1'] for r in rows])}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for p in output.iterdir():p.chmod(0o600)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();run(a.private_root,a.output)


if __name__=='__main__':main()
