"""Verify private secondary outputs and export aggregate reviewer evidence/figure."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reviewer_checks import threshold_metrics


def export(reviewer_directory, nonlinear_directory, destination):
    reviewer_directory, nonlinear_directory = Path(reviewer_directory), Path(nonlinear_directory)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    reviewer = json.loads((reviewer_directory/'report.json').read_text())
    nonlinear = json.loads((nonlinear_directory/'report.json').read_text())
    for name in ['mentalhealth', 'depression']:
        section = nonlinear['comparisons'][name]
        if section['input_sha256'] != reviewer['comparisons'][name]['input_sha256']:
            raise ValueError('Reviewer and nonlinear runs use different inputs')
        for filename, key in [(f'{name}_nonlinear.joblib', 'model_sha256'),
                              (f'{name}_private_predictions.csv', 'private_predictions_sha256')]:
            if hashlib.sha256((nonlinear_directory/filename).read_bytes()).hexdigest() != section[key]:
                raise ValueError('Nonlinear artifact checksum failed')
        frame = pd.read_csv(nonlinear_directory/f'{name}_private_predictions.csv')
        observed = threshold_metrics(frame['label'].to_numpy(), frame['nonlinear_probability'].to_numpy(),
                                     section['selection']['threshold'])
        for key in ['macro_f1', 'roc_auc', 'accuracy', 'positive_recall']:
            if abs(observed[key]-section['test_at_validation_threshold'][key]) > 1e-12:
                raise ValueError('Saved nonlinear predictions do not reproduce report')
        if observed['confusion_matrix'] != section['test_at_validation_threshold']['confusion_matrix']:
            raise ValueError('Nonlinear confusion matrix disagrees with saved predictions')
    for directory, value, filename in [(reviewer_directory, reviewer, 'reviewer_check_results.json'),
                                        (nonlinear_directory, nonlinear, 'nonlinear_check_results.json')]:
        value['private_report_sha256'] = hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
        value['export_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (destination/filename).write_text(json.dumps(value, indent=2, allow_nan=False))
    figure_dir = destination/'figures'
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    labels = ['13-feature LR', '13-feature boosting', 'TF-IDF LR']
    for n, (name, color) in enumerate([('mentalhealth', '#176f9b'), ('depression', '#c17122')]):
        r, h = reviewer['comparisons'][name], nonlinear['comparisons'][name]
        values = [r['validation_threshold_test']['linguistic_13']['macro_f1'],
                  h['test_at_validation_threshold']['macro_f1'], r['validation_threshold_test']['tfidf_lr']['macro_f1']]
        axes[0].bar(np.arange(3) + (n-.5)*.32, values, width=.32, color=color, label=f'Anxiety vs {name}')
        for x, v in zip(np.arange(3)+(n-.5)*.32, values):
            axes[0].text(x, v+.013, f'{v:.3f}', ha='center', fontsize=8)
        contrasts = [r['lexical_minus_linguistic'], h['nonlinear_minus_tfidf_lr']]
        for j, c in enumerate(contrasts):
            sign = 1 if j==0 else -1
            point = sign*c['macro_f1_delta']
            bounds = sorted(sign*np.asarray(c['macro_f1_ci95']))
            y = n*2+j
            axes[1].errorbar(point, y, xerr=[[point-bounds[0]], [bounds[1]-point]],
                             fmt='o', color=color, capsize=4)
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('June account-level macro-F1')
    axes[0].set_title('May-selected operating points')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[1].set_yticks(np.arange(4), ['mentalhealth: vs LR', 'mentalhealth: vs boosting',
                                      'depression: vs LR', 'depression: vs boosting'])
    axes[1].invert_yaxis()
    axes[1].set_xlim(.2, .31)
    axes[1].set_xlabel('TF-IDF minus linguistic macro-F1')
    axes[1].set_title('Paired advantages with 95% intervals')
    for ax in axes:
        ax.grid(axis='y' if ax==axes[0] else 'x', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Exploratory reviewer controls: community affiliation, not diagnosis', fontsize=11)
    fig.text(.5, .015, 'Retained LR models; one validation-selected nonlinear control. Intervals: 1,000 stratified account bootstraps.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .045, 1, .93])
    for suffix in ['pdf', 'png']:
        fig.savefig(figure_dir/f'reviewer_controls.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [destination/'reviewer_check_results.json', destination/'nonlinear_check_results.json',
                          figure_dir/'reviewer_controls.pdf', figure_dir/'reviewer_controls.png']}
    (destination/'reviewer_export_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Verified nonlinear predictions/artifacts and exported aggregate evidence and PDF/PNG figure.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewer-directory', required=True)
    p.add_argument('--nonlinear-directory', required=True)
    p.add_argument('--destination', required=True)
    a = p.parse_args()
    export(a.reviewer_directory, a.nonlinear_directory, a.destination)


if __name__ == '__main__':
    main()
