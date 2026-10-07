"""Audit uploaded PHQ-8 label files; emit aggregate evidence without participant IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FILES = ['labels.csv', 'train_split_Depression_AVEC2017.csv', 'dev_split_Depression_AVEC2017.csv',
         'test_split_Depression_AVEC2017.csv', 'full_test_split.csv']


def audit(manifest):
    rows = json.loads(Path(manifest).read_text())
    data, sources = {}, []
    for name in FILES:
        item = next(x for x in rows if x['name'] == name)
        path = Path(item['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Clinical input checksum failed')
        data[name] = pd.read_csv(path)
        sources.append({'name':name,'sha256':digest,'rows':len(data[name]),'columns':list(data[name])})
    labels = data['labels.csv']
    train, dev = data['train_split_Depression_AVEC2017.csv'], data['dev_split_Depression_AVEC2017.csv']
    test, blind = data['full_test_split.csv'], data['test_split_Depression_AVEC2017.csv']
    ids = {}
    for name, frame, column in [('labels',labels,'participant_id'),('train',train,'Participant_ID'),
                                ('dev',dev,'Participant_ID'),('test',test,'Participant_ID'),('blind',blind,'participant_ID')]:
        values = pd.to_numeric(frame[column],errors='coerce')
        if values.isna().any() or values.duplicated().any() or (values%1 != 0).any():
            raise ValueError('Observed participant identifiers must be present, unique integers')
        ids[name] = set(values.astype(int))
    combined = pd.concat([train.assign(_official_split='train'),dev.assign(_official_split='dev')]).set_index('Participant_ID')
    aligned = labels.set_index('participant_id').reindex(combined.index)
    label_values = pd.to_numeric(labels['anxiety_label'],errors='coerce')
    scores = pd.to_numeric(labels['phq8_score'],errors='coerce')
    if scores.isna().any() or not scores.between(0,24).all() or not label_values.isin([0,1]).all():
        raise ValueError('Invalid PHQ-8 score or binary label')
    threshold = [t for t in range(0,26) if (scores.ge(t).astype(int)==label_values).all()]
    report = {
        'status':'aggregate_label_provenance_audit', 'source_files':sources,
        'observed_participants_in_labels':len(ids['labels']),
        'unused_blank_participant_ID_column_rows':int(labels['participant_ID'].isna().sum()),
        'labels_equal_train_dev_id_union':ids['labels']==ids['train']|ids['dev'],
        'split_overlap_counts':{f'{a}_{b}':len(ids[a]&ids[b]) for a,b in [('train','dev'),('train','test'),('dev','test')]},
        'full_and_blind_test_ids_equal':ids['test']==ids['blind'],
        'total_distinct_participants':len(ids['train']|ids['dev']|ids['test']),
        'anxiety_label_exact_phq8_thresholds':threshold,
        'anxiety_label_counts':{str(k):int(v) for k,v in label_values.value_counts().items()},
        'derived_label_disagrees_with_supplied_depression_binary_rows':int((label_values!=labels['has_depression']).sum()),
        'merged_labels_match_original_scores':bool(aligned['phq8_score'].eq(combined['PHQ8_Score']).all()),
        'merged_labels_match_original_depression_binary':bool(aligned['has_depression'].eq(combined['PHQ8_Binary']).all()),
        'merged_labels_match_original_split':bool(aligned['split'].eq(combined['_official_split']).all()),
        'official_binary_counts_by_split':{name:{str(int(k)):int(v) for k,v in frame[column].value_counts().items()}
                                         for name,frame,column in [('train',train,'PHQ8_Binary'),('dev',dev,'PHQ8_Binary'),('test',test,'PHQ_Binary')]},
        'dev_derived_anxiety_label_counts':{str(k):int(v) for k,v in labels.loc[labels['participant_id'].isin(ids['dev']),'anxiety_label'].value_counts().items()},
        'interpretation':'The uploaded anxiety_label is derivable from a depression scale; no validated anxiety outcome was supplied.',
        'measurement_scope':'PHQ-8 depression symptoms; binary score thresholds are not clinician diagnoses.',
        'clinical_text_or_measured_linguistic_features_supplied':False,
        'clinical_effects_recomputed':False,
        'discrepancy_action':'Retain original supplied binary labels and document the one disagreement; verify authoritative version before choosing any correction.',
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = audit(args.manifest)
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({key:report[key] for key in ['total_distinct_participants','anxiety_label_exact_phq8_thresholds',
                                                'derived_label_disagrees_with_supplied_depression_binary_rows','official_binary_counts_by_split']},indent=2))


if __name__=='__main__':main()
