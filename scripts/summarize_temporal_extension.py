"""Verify saved later-period/resampling evidence and publish aggregate artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def verify_binary_metrics(frame, column, expected):
    y=frame['label'].to_numpy()
    p=frame[column].to_numpy()
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid saved probability')
    predicted=p>=expected['threshold']
    tn=int(np.sum((y==0)&~predicted));fp=int(np.sum((y==0)&predicted))
    fn=int(np.sum((y==1)&~predicted));tp=int(np.sum((y==1)&predicted))
    if min(np.sum(y==0),np.sum(y==1))==0:raise ValueError('Both test classes required')
    macro=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2
    if expected['confusion_matrix']!=[[tn,fp],[fn,tp]] or abs(macro-expected['macro_f1'])>1e-12 or len(y)!=expected['n_users']:
        raise ValueError('Saved predictions disagree with reported metrics')


def export(temporal_directory,resampling_directory,destination):
    temporal_directory,resampling_directory=Path(temporal_directory),Path(resampling_directory)
    destination=Path(destination)
    temporal=json.loads((temporal_directory/'report.json').read_text())
    resampling=json.loads((resampling_directory/'report.json').read_text())
    for comparison,section in temporal['comparisons'].items():
        for month,outcome in section['months'].items():
            path=temporal_directory/f'{comparison}_{month}_private_predictions.csv'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=outcome['private_predictions_sha256']:
                raise ValueError('Later-period probability checksum mismatch')
            posts=temporal_directory/f'{comparison}_{month}_posts.csv'
            if hashlib.sha256(posts.read_bytes()).hexdigest()!=outcome['prepared_posts_sha256']:
                raise ValueError('Later-period input checksum mismatch')
            frame=pd.read_csv(path)
            if frame['author_id'].duplicated().any():raise ValueError('Duplicated account prediction row')
            masks={'main':np.ones(len(frame),dtype=bool),'historical_absence_sensitivity':frame['historically_absent'].to_numpy(),
                   'near_duplicate_exclusion_sensitivity':frame['no_detected_reuse'].to_numpy()}
            for subset,mask in masks.items():
                for operating_point in ['metrics_at_validation_threshold','metrics_at_05']:
                    for name,expected in outcome[subset][operating_point].items():
                        verify_binary_metrics(frame.loc[mask],name,expected)
    for comparison,section in resampling['comparisons'].items():
        if [r['seed'] for r in section['resamples']]!=list(range(100,120)):
            raise ValueError('Missing/reordered training resamples')
        for row in section['resamples']:
            path=resampling_directory/f"{comparison}_seed{row['seed']}_private_predictions.csv"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['private_predictions_sha256']:
                raise ValueError('Resampled-model probability checksum mismatch')
            frame=pd.read_csv(path)
            for name,model in row['models'].items():verify_binary_metrics(frame,name,model['june'])
    destination.mkdir(parents=True,exist_ok=True)
    for directory,result,filename in [(temporal_directory,temporal,'temporal_extension_results.json'),
                                        (resampling_directory,resampling,'training_resampling_results.json')]:
        result['private_report_sha256']=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
        result['export_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['saved_probability_arithmetic_verified']=True
        (destination/filename).write_text(json.dumps(result,indent=2,allow_nan=False))
    figures=destination/'figures';figures.mkdir(exist_ok=True)
    names=['linguistic_13','linguistic_boosting','tfidf_lr']
    labels=['13-feature LR','13-feature boosting','TF-IDF LR']
    colors=['#176f9b','#6c8271','#c17122']
    fig,axes=plt.subplots(1,2,figsize=(9.8,4.8),sharey=True)
    for ax,comparison in zip(axes,['mentalhealth','depression']):
        for name,label,color in zip(names,labels,colors):
            outcomes=[temporal['comparisons'][comparison]['months'][m]['main'] for m in ['2022-07','2022-08']]
            points=np.array([r['metrics_at_validation_threshold'][name]['macro_f1'] for r in outcomes])
            intervals=np.array([r['macro_f1_ci95'][name] for r in outcomes])
            ax.errorbar(np.arange(2),points,yerr=[points-intervals[:,0],intervals[:,1]-points],
                        marker='o',capsize=4,color=color,label=label)
        ax.set_xticks(np.arange(2),['July 2022','August 2022'])
        ax.set_xlim(-.2,1.2);ax.set_ylim(0,1)
        ax.set_title(f'Anxiety vs {comparison}')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        for i,m in enumerate(['2022-07','2022-08']):
            n=temporal['comparisons'][comparison]['months'][m]['cohort']['retained_authors']
            ax.text(i,.06,f'n = {n:,}',ha='center',fontsize=9)
    axes[0].set_ylabel('Account-level macro-F1 at frozen May threshold')
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Frozen-model evaluation in two later RMHD months',fontsize=12)
    fig.text(.5,.012,'Intervals: 1,000 stratified account bootstraps. Different accounts across months; one source, community outcome.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.94])
    for extension in ['pdf','png']:fig.savefig(figures/f'temporal_extension.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    artifacts=[destination/'temporal_extension_results.json',destination/'training_resampling_results.json',
               figures/'temporal_extension.pdf',figures/'temporal_extension.png']
    (destination/'temporal_export_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},indent=2))
    print('Verified all four later-period cohorts and all 40 training-account resamples; aggregate export complete.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--temporal-directory',required=True)
    p.add_argument('--resampling-directory',required=True)
    p.add_argument('--destination',required=True)
    a=p.parse_args();export(a.temporal_directory,a.resampling_directory,a.destination)


if __name__=='__main__':main()
