"""Frozen July/August evaluation with full-release provenance and history audits."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .features import get_feature_names
from .preprocess import clean_text
from .research_benchmark import bootstrap_indices, user_inputs
from .reviewer_checks import threshold_metrics, paired_thresholds, near_duplicates
from .uploaded_pilot import SCHEMA, COMMUNITIES, PLACEHOLDERS


MONTHS = ['2022-07', '2022-08']
MODELS = ['linguistic_13', 'tfidf_lr', 'linguistic_boosting']


def valid_metadata(frame):
    d = frame.copy()
    d['_community'] = d['subreddit'].fillna('').astype(str).str.strip().str.casefold()
    d['author'] = d['author'].fillna('').astype(str).str.strip().str.casefold()
    author_ok = d['author'].str.fullmatch(r'[a-z0-9_-]{3,20}', na=False) & ~d['author'].isin(PLACEHOLDERS)
    d['created_utc'] = pd.to_numeric(d['created_utc'], errors='coerce')
    score = pd.to_numeric(d['score'], errors='coerce')
    finite = np.isfinite(d['created_utc']) & np.isfinite(score)
    time_ok = d['created_utc'].ge(pd.Timestamp('2019-01-01', tz='UTC').timestamp()) & d['created_utc'].lt(pd.Timestamp('2023-01-01', tz='UTC').timestamp())
    keep = author_ok & d['_community'].isin(COMMUNITIES) & finite & time_ok
    return d[keep].copy(), {'rows': len(d), 'excluded_metadata_rows': int((~keep).sum())}


def update_history(history, frame):
    for (community, author), timestamp in frame.groupby(['_community', 'author'])['created_utc'].min().items():
        history[community][author] = min(history[community].get(author, float('inf')), float(timestamp))


def pair_history(history, comparison):
    merged = dict(history['anxiety'])
    for author, time in history[comparison].items():
        merged[author] = min(merged.get(author, float('inf')), time)
    return merged


def inventory_release(manifest_path, upload_manifest_path, cache):
    manifest = json.loads(Path(manifest_path).read_text())
    original = {x['sha256'] for x in json.loads(Path(upload_manifest_path).read_text())}
    history, original_history, later_history = [defaultdict(dict) for _ in range(3)]
    files, later, seen = [], [], set()
    start = pd.Timestamp('2022-07-01', tz='UTC').timestamp()
    end = pd.Timestamp('2022-09-01', tz='UTC').timestamp()
    for item in manifest:
        if '/raw data/' not in item['name']:
            continue
        entry = {k: item[k] for k in ['name', 'sha256', 'size_bytes']}
        if not item['name'].lower().endswith('.csv'):
            files.append({**entry, 'status': 'unsupported_format'})
            continue
        path = Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Upstream extracted file checksum changed')
        if item['sha256'] in seen:
            files.append({**entry, 'status': 'byte_identical_file_skipped'})
            continue
        seen.add(item['sha256'])
        columns = pd.read_csv(path, nrows=0).columns
        if not SCHEMA <= set(columns):
            files.append({**entry, 'status': 'unsupported_schema'})
            continue
        raw = pd.read_csv(path, usecols=list(SCHEMA), low_memory=False)
        frame, counts = valid_metadata(raw)
        update_history(history, frame)
        if item['sha256'] in original:
            update_history(original_history, frame)
        in_later = frame['created_utc'].ge(start) & frame['created_utc'].lt(end)
        if in_later.any():
            subset = frame[in_later].copy()
            later.append(subset)
            update_history(later_history, subset)
        files.append({**entry, 'status': 'parsed', **counts,
                      'later_period_metadata_rows': int(in_later.sum())})
    if not later:
        raise ValueError('No July/August observations in release')
    result = {'files': files, 'raw_archive_paths': len(files),
              'parsed_distinct_file_rows': sum(f.get('rows', 0) for f in files),
              'metadata_valid_distinct_file_rows': sum(f.get('rows', 0)-f.get('excluded_metadata_rows', 0) for f in files),
              'unsupported_files': [f['name'] for f in files if f['status'].startswith('unsupported')],
              'duplicate_file_paths_skipped': sum(f['status']=='byte_identical_file_skipped' for f in files),
              'history_scope': 'Parseable metadata-valid raw CSV observations dated 2019 onward; unsupported format is not treated as observed'}
    state = {'later': pd.concat(later, ignore_index=True), 'history': dict(history),
             'original_history': dict(original_history), 'later_history': dict(later_history), 'inventory': result}
    joblib.dump(state, cache)
    Path(cache).chmod(0o600)
    return state


def monthly_cohort(later, comparison, month, original_history, later_history, reference_texts, earlier_texts):
    start = pd.Timestamp(month+'-01', tz='UTC')
    end = start + pd.offsets.MonthBegin(1)
    d = later[later['_community'].isin({'anxiety', comparison}) &
              later['created_utc'].ge(start.timestamp()) & later['created_utc'].lt(end.timestamp())].copy()
    report = {'metadata_posts': len(d), 'metadata_authors': int(d['author'].nunique())}
    mixed = d.groupby('author')['_community'].nunique().loc[lambda x: x>1].index
    report['ambiguous_authors'] = len(mixed)
    d = d[~d['author'].isin(mixed)].copy()
    previous = {a for h in [original_history, later_history] for a,t in h.items() if t < start.timestamp()}
    report['previously_observed_authors_excluded'] = int(d.loc[d['author'].isin(previous), 'author'].nunique())
    d = d[~d['author'].isin(previous)].copy()
    d['_body'] = d['selftext'].fillna('').map(clean_text)
    d['_title'] = d['title'].fillna('').map(clean_text)
    good = ~d['_body'].isin(['', '[deleted]', '[removed]']) & d['_body'].str.split().map(len).ge(10)
    report['body_quality_posts_excluded'] = int((~good).sum())
    d = d[good].copy()
    d['text'] = (d['_title']+' '+d['_body']).str.strip()
    record_duplicate = d.duplicated(['author', 'created_utc', '_community', 'text'])
    report['identical_record_posts_excluded'] = int(record_duplicate.sum())
    d = d[~record_duplicate].copy()
    shared = d.groupby('text')['author'].nunique().loc[lambda x: x>1].index
    report['shared_within_month_posts_excluded'] = int(d['text'].isin(shared).sum())
    d = d[~d['text'].isin(shared)].sort_values(['created_utc','author'],kind='stable')
    repeated = d['text'].duplicated()
    report['repeated_same_author_text_posts_excluded'] = int(repeated.sum())
    d = d[~repeated].copy()
    reference_reuse = set(d.loc[d['text'].isin(reference_texts), 'author'])
    report['training_validation_exact_reuse_authors_excluded'] = len(reference_reuse)
    d = d[~d['author'].isin(reference_reuse)].copy()
    earlier_reuse = d['text'].isin(earlier_texts)
    report['earlier_extension_exact_reuse_posts_excluded'] = int(earlier_reuse.sum())
    d = d[~earlier_reuse].copy()
    d['label'] = d['_community'].eq('anxiety').astype(int)
    d['split'] = 'test'
    report.update(retained_posts=len(d), retained_authors=int(d['author'].nunique()),
                  positive_authors=int(d.loc[d['label'].eq(1),'author'].nunique()),
                  utc_start=pd.to_datetime(d['created_utc'].min(),unit='s',utc=True).isoformat() if len(d) else None,
                  utc_end=pd.to_datetime(d['created_utc'].max(),unit='s',utc=True).isoformat() if len(d) else None)
    if d.groupby('author')['label'].nunique().gt(1).any():
        raise ValueError('Mixed community labels after cohort preparation')
    return d[['author','text','created_utc','label','split']].reset_index(drop=True), report


def macro_interval(labels, probability, threshold, bootstrap):
    pred = np.asarray(probability) >= threshold
    values = [f1_score(labels[i],pred[i],average='macro') for i in bootstrap_indices(labels,bootstrap,42)]
    return np.quantile(values,[.025,.975]).tolist()


def evaluate_subset(labels, probabilities, thresholds, keep, bootstrap):
    y = np.asarray(labels)[keep]
    if set(y) != {0,1}:
        return {'status':'not_estimable_both_classes_absent','n_users':len(y)}
    report = {'metrics_at_validation_threshold': {}, 'macro_f1_ci95': {},
              'metrics_at_05': {}, 'imprecision_flag': int(min(np.bincount(y))) < 100}
    for name in MODELS:
        p = probabilities[name][keep]
        report['metrics_at_validation_threshold'][name] = threshold_metrics(y,p,thresholds[name])
        report['macro_f1_ci95'][name] = macro_interval(y,p,thresholds[name],bootstrap)
        report['metrics_at_05'][name] = threshold_metrics(y,p,.5)
    for name in ['linguistic_13','linguistic_boosting']:
        report[f'lexical_minus_{name}'] = paired_thresholds(y,probabilities[name][keep],thresholds[name],
                        probabilities['tfidf_lr'][keep],thresholds['tfidf_lr'],bootstrap)
    return report


def run(release_manifest, upload_manifest, private_root, output, bootstrap=1000):
    root, output = Path(private_root), Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo):
        raise ValueError('Keep source history and individual predictions outside Git')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    print('Auditing full-release metadata and preparing later periods',flush=True)
    state = inventory_release(release_manifest,upload_manifest,output/'private_history.joblib')
    reviewer = json.loads((root/'reviewer-checks-seed42/report.json').read_text())
    nonlinear = json.loads((root/'nonlinear-check-seed42/report.json').read_text())
    result = {'status':'frozen_later_period_exploratory_extension','bootstrap':bootstrap,'seed':42,
              'protocol_sha256':hashlib.sha256((repo/'research/TEMPORAL_EXTENSION_PROTOCOL.md').read_bytes()).hexdigest(),
              'code_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),repo/'src/features.py',repo/'src/preprocess.py']},
              'release_manifest_sha256':hashlib.sha256(Path(release_manifest).read_bytes()).hexdigest(),
              'source_inventory':state['inventory'],'comparisons':{},
              'scope':'Two new months from the same collection, not independent-source or clinical validation'}
    for comparison in ['mentalhealth','depression']:
        primary = pd.read_csv(root/f'pilot-{comparison}-data/posts.csv',dtype={'author':str})
        reference = primary[~primary['split'].eq('test')]['text'].tolist()
        original_hist = pair_history(state['original_history'],comparison)
        later_hist = pair_history(state['later_history'],comparison)
        full_hist = pair_history(state['history'],comparison)
        models = {n:joblib.load(root/f'pilot-{comparison}-seed42/full_{n}.joblib') for n in ['linguistic_13','tfidf_lr']}
        models['linguistic_boosting'] = joblib.load(root/f'nonlinear-check-seed42/{comparison}_nonlinear.joblib')
        thresholds = {n:reviewer['comparisons'][comparison]['threshold_selection'][n]['threshold'] for n in ['linguistic_13','tfidf_lr']}
        thresholds['linguistic_boosting'] = nonlinear['comparisons'][comparison]['selection']['threshold']
        section = {'thresholds':thresholds,'months':{},'model_sha256':{}}
        for name in MODELS:
            path = root/f'nonlinear-check-seed42/{comparison}_nonlinear.joblib' if name=='linguistic_boosting' else root/f'pilot-{comparison}-seed42/full_{name}.joblib'
            section['model_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        earlier_texts, seen_authors = set(),set(primary['author'])
        for month in MONTHS:
            posts,audit = monthly_cohort(state['later'],comparison,month,original_hist,later_hist,set(reference),earlier_texts)
            if set(posts['author']) & seen_authors:
                raise ValueError('Observed model/evaluation author overlaps a later cohort')
            seen_authors.update(posts['author'])
            earlier_texts.update(posts['text'])
            print(f'Frozen inference: {comparison} {month}, {len(posts)} posts',flush=True)
            numeric,texts,meta = user_inputs(posts)
            labels = meta['label'].to_numpy()
            probabilities = {name:model.predict_proba(texts.to_numpy() if name=='tfidf_lr' else numeric[get_feature_names()].to_numpy())[:,1]
                             for name,model in models.items()}
            start = pd.Timestamp(month+'-01',tz='UTC').timestamp()
            historically_absent = np.array([full_hist.get(a,float('inf'))>=start for a in meta.index])
            near_positions,near_audit = near_duplicates(reference,posts['text'].tolist())
            reused_authors = set(posts.iloc[sorted(near_positions)]['author'])
            no_reuse = ~meta.index.isin(reused_authors)
            near_audit['affected_test_authors'] = len(reused_authors)
            outcome = {'cohort':audit,'main':evaluate_subset(labels,probabilities,thresholds,np.ones(len(labels),dtype=bool),bootstrap),
                       'historical_absence_sensitivity':evaluate_subset(labels,probabilities,thresholds,historically_absent,bootstrap),
                       'near_duplicate_audit':near_audit,
                       'near_duplicate_exclusion_sensitivity':evaluate_subset(labels,probabilities,thresholds,no_reuse,bootstrap)}
            predicted = pd.DataFrame({'author_id':[f'user_{i:06d}' for i in range(len(labels))],'label':labels,
                                      'historically_absent':historically_absent,'no_detected_reuse':no_reuse,**probabilities})
            prediction_path=output/f'{comparison}_{month}_private_predictions.csv'
            predicted.to_csv(prediction_path,index=False);prediction_path.chmod(0o600)
            posts_path=output/f'{comparison}_{month}_posts.csv'
            posts.to_csv(posts_path,index=False);posts_path.chmod(0o600)
            outcome['private_predictions_sha256']=hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            outcome['prepared_posts_sha256']=hashlib.sha256(posts_path.read_bytes()).hexdigest()
            section['months'][month]=outcome
            print(json.dumps({'comparison':comparison,'month':month,'authors':len(labels),
                    'main_macro_f1':{k:v['macro_f1'] for k,v in outcome['main']['metrics_at_validation_threshold'].items()},
                    'historically_absent_authors':int(historically_absent.sum())}),flush=True)
        result['comparisons'][comparison]=section
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));(output/'report.json').chmod(0o600)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-manifest',required=True)
    p.add_argument('--upload-manifest',required=True)
    p.add_argument('--private-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args();run(a.release_manifest,a.upload_manifest,a.private_root,a.output,a.bootstrap)


if __name__=='__main__':main()
