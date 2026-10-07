"""Export aggregate pilot evidence and a figure without exporting personal records."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest
from scipy.special import logsumexp


NAMES = ['majority', 'length_only', 'pronoun_only', 'without_sentiment', 'linguistic_13', 'tfidf_lr']
LABELS = ['Majority', 'Length only', 'Pronouns only', 'Without sentiment', '13 linguistic features', 'TF-IDF + LR']


def annotate_exact_tests(value):
    """Do not publish a numerical underflow as an exact probability of zero."""
    if isinstance(value, dict):
        if 'mcnemar_exact' in value:
            test = value['mcnemar_exact']
            a, b = test['a_correct_b_wrong'], test['a_wrong_b_correct']
            n = a+b
            p = float(binomtest(a,n,.5).pvalue) if n else 1.
            logp = min(0., float(np.log(2.)+logsumexp(binom.logpmf(np.arange(min(a,b)+1),n,.5)))) if n else 0.
            test.update({'pvalue':None if p==0. else p,'log10_pvalue':float(logp/np.log(10.)),'pvalue_underflow':p==0.})
        for item in value.values():annotate_exact_tests(item)
    elif isinstance(value,list):
        for item in value:annotate_exact_tests(item)


def verify_predictions(predictions, reported):
    """Independent arithmetic from saved predictions, not aggregate assertions."""
    if predictions['author_id'].duplicated().any():
        raise ValueError('Prediction identifiers are repeated')
    y = predictions['label'].to_numpy()
    if set(y) != {0, 1}:
        raise ValueError('Test labels require both classes')
    for name in [*NAMES, 'linguistic_13_masked_test']:
        probability = predictions[name].to_numpy()
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError('Invalid saved probabilities')
        p = probability >= .5
        tn = int(np.sum((y == 0) & ~p)); fp = int(np.sum((y == 0) & p))
        fn = int(np.sum((y == 1) & ~p)); tp = int(np.sum((y == 1) & p))
        f1_pos = 2*tp/(2*tp + fp + fn)
        f1_neg = 2*tn/(2*tn + fp + fn)
        expected = reported['masked_test']['metrics'] if name.endswith('masked_test') else reported['metrics'][name]
        if expected['confusion_matrix'] != [[tn, fp], [fn, tp]]:
            raise ValueError('Saved confusion matrix does not match predictions')
        if abs(expected['macro_f1'] - (f1_pos + f1_neg)/2) > 1e-12:
            raise ValueError('Saved macro-F1 does not match predictions')
        if expected['n_users'] != len(y):
            raise ValueError('Saved test sample size does not match predictions')


def summarize(private_root, output):
    private_root, output = Path(private_root), Path(output)
    result = {
        'date': '2026-10-07', 'status': 'exploratory_community_proxy_pilot',
        'scope': 'Selected uploaded examples; community-affiliation labels, not anxiety diagnosis or onset',
        'provenance_status': 'Kaggle RMHD version 1 and publisher-stated CC0 verified; all 15 authored uploads are byte-identical to downloaded upstream files. Secondary institutional determination remains outstanding; whole-release parsing coverage is documented separately.',
        'author_attributed_dataset_study': 'https://www.mdpi.com/2076-3417/14/4/1547',
        'source_study_documented_facts': {
            'source_pdf': 'https://mdpi-res.com/d_attachment/applsci/applsci-14-01547/article_deploy/applsci-14-01547.pdf',
            'source_pdf_sha256': '6ed5f1ad02f7a9ff9a9d3ad6be3c7074f2e5eca348e2902cb1c95f4f0fb6e1bb',
            'reading_scope': 'Methods, organization/annotation, ethics and availability sections',
            'reported_original_posts': 1494019, 'reported_collection_period': 'January 2019 to August 2022',
            'annotated_subset': '800 posts in four proposed root-cause categories, not anxiety diagnoses',
            'original_study_reported_approval': 'Victoria University HREC HRE23-005, 29 May 2023; does not establish coverage for current secondary analysis',
            'dataset_short_link': 'https://rb.gy/ewtjy',
            'resolved_dataset_url': 'https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset',
            'identified_version': 1, 'publisher_stated_license': 'CC0: Public Domain',
            'version_and_stated_license_metadata_verified': True,
            'upstream_content_checksum_identity_verified': True,
            'archive_sha256': '6078fe83304c266ca976973f8b1e553dc5d818c59810508151f9e6bc615bf9e4',
        },
        'comparison_dependence': 'The two comparisons share some Anxiety authors and are not independent replications',
        'comparisons': {},
    }
    result['export_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    upload_audit = json.loads((private_root/'upload-audit/aggregate_upload_audit.json').read_text())
    result['uploads'] = upload_audit
    result['additional_upload_audit'] = json.loads((private_root/'upload-audit/additional_aggregate_audit.json').read_text())
    result['clinical_label_audit'] = json.loads((private_root/'upload-audit/clinical_label_audit_reproducible.json').read_text())
    for comparison in ['mentalhealth', 'depression']:
        directory = private_root/f'pilot-{comparison}-seed42'
        report = json.loads((directory/'report.json').read_text())
        if report['status'] != 'observational_proxy_experiment':
            raise ValueError('An empirical pilot report is required')
        if report['experiments'].keys() != {'full'}:
            raise ValueError('This summary expects the fixed full-month pilot')
        for name, digest in report['artifact_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Experiment artifact checksum failed')
        verify_predictions(pd.read_csv(directory/'full_predictions.csv'), report['experiments']['full'])
        cohort = json.loads((private_root/f'pilot-{comparison}-data/cohort_audit.json').read_text())
        lengths = json.loads((private_root/f'pilot-{comparison}-data/length_activity_audit.json').read_text())
        # Only fixed, aggregate fields are exported. No raw CSV cells or fitted vocabulary.
        result['comparisons'][comparison] = {
            'cohort_audit': cohort, 'length_activity': lengths,
            'outcomes': report['experiments']['full'],
            'run_metadata': {key: report[key] for key in ['status', 'created_utc', 'seed', 'input_sha256', 'versions',
                             'code_sha256', 'git_commit', 'git_dirty', 'bootstrap', 'outcome_scope', 'analysis_notes', 'artifact_sha256']},
            'private_report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
            'saved_prediction_arithmetic_verified': True,
        }
        sensitivity_dir = private_root/f'pilot-{comparison}-sensitivity'
        sensitivity = json.loads((sensitivity_dir/'report.json').read_text())
        if sensitivity['primary_report_sha256'] != result['comparisons'][comparison]['private_report_sha256']:
            raise ValueError('Sensitivity uses a different primary run')
        for name, digest in sensitivity['artifact_sha256'].items():
            if hashlib.sha256((sensitivity_dir/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Sensitivity artifact checksum failed')
        matched_predictions = pd.read_csv(sensitivity_dir/'matched_predictions.csv')
        for name in NAMES:
            if len(matched_predictions) != sensitivity['matched_test']['metrics'][name]['n_users']:
                raise ValueError('Matched prediction sample count failed')
        result['comparisons'][comparison]['secondary_sensitivity'] = sensitivity
    annotate_exact_tests(result)
    result['statistical_format_note'] = 'Underflowed exact McNemar p-values are stored as null with finite log10 probability; private original reports remain unchanged.'
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(result, output.parent/'figures')
    result['figure_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [output.parent/'figures/pilot_baselines.pdf',output.parent/'figures/pilot_baselines.png']}
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def plot(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    y = np.arange(len(NAMES))
    for ax, comparison in zip(axes, ['mentalhealth', 'depression']):
        outcomes = result['comparisons'][comparison]['outcomes']
        point = np.array([outcomes['metrics'][name]['macro_f1'] for name in NAMES])
        lower = np.array([outcomes['ci95'][name]['macro_f1'][0] for name in NAMES])
        upper = np.array([outcomes['ci95'][name]['macro_f1'][1] for name in NAMES])
        colors = ['#9ba6ae']*4 + ['#176f9b', '#c17122']
        ax.barh(y, point, color=colors, height=.6)
        ax.errorbar(point, y, xerr=[point-lower, upper-point], fmt='none', color='#222222', capsize=3, linewidth=1)
        ax.set_yticks(y, LABELS)
        ax.set_xlim(0, 1)
        ax.set_xlabel('June author-level macro-F1')
        n = outcomes['metrics']['majority']['n_users']
        ax.set_title(f'Anxiety vs {comparison}\n{n:,} test authors')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(point):
            ax.text(min(value + .025, .94), i, f'{value:.3f}', va='center', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Temporal evaluation of community labels in uploaded examples', fontsize=12)
    fig.text(.5, .01, 'Bars: fixed models; intervals: 1,000 stratified author bootstraps. Community membership is not a diagnosis.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.94])
    fig.savefig(directory/'pilot_baselines.pdf', bbox_inches='tight')
    fig.savefig(directory/'pilot_baselines.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.private_root, args.output)
    print('Verified predictions and exported two aggregate community-proxy comparisons.')
    for name, comparison in result['comparisons'].items():
        print(name, json.dumps({model: value['macro_f1'] for model, value in comparison['outcomes']['metrics'].items()}))


if __name__ == '__main__':
    main()
