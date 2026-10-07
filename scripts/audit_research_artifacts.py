"""Check published aggregates without pretending to reproduce the underlying experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, chi2


def audit(root):
    root = Path(root)
    result_path = root/'results/final_results_fixed.json'
    result = json.loads(result_path.read_text())
    (tn, fp), (fn, tp) = result['confusion_matrix']
    n = tn + fp + fn + tp
    recomputed = {'accuracy':(tn+tp)/n, 'precision':tp/(tp+fp),
                  'recall':tp/(tp+fn), 'f1':2*tp/(2*tp+fp+fn)}
    cross = pd.read_csv(root/'results/cross_domain_validation.csv')
    early = pd.read_csv(root/'results/early_slice_results.csv')
    evaluable = cross['daic_p'].notna()
    return {
        'scope': 'Arithmetic and artifact consistency only; experiments and data provenance are NOT reproduced',
        'input_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'test_confusion_matrix_n': n,
        'test_positive_prevalence': (fn+tp)/n,
        'confusion_matrix_metrics': recomputed,
        'json_matches_confusion_matrix': {k: abs(v-result['test'][k]) < 1e-12 for k,v in recomputed.items()},
        'roc_auc': 'Cannot recompute from a single confusion matrix; individual scores are missing',
        'cross_domain_direction_agreement': {
            'all_features': {'consistent':int(cross['consistent'].sum()), 'denominator':len(cross)},
            'non_missing_daic_p': {'consistent':int(cross.loc[evaluable,'consistent'].sum()), 'denominator':int(evaluable.sum())},
            'interpretation': 'Sign agreement is not external classifier validation or anxiety construct validation'},
        'early_history_cohort_sizes': early[['k','n_users','test_size','anxiety_pct']].to_dict(orient='records'),
        'readme_users': {'stated':155599, 'sum_of_split_rows':128140+22459+27459},
        'preprint_users': {'sum_of_split_rows':128140+27459+27459, 'stated_total':183058},
        'preprint_mcnemar_from_printed_discordant_pairs': {
            'b':52, 'c':76, 'uncorrected_chi_square':(52-76)**2/(52+76),
            'uncorrected_p':float(chi2.sf((52-76)**2/(52+76),1)),
            'continuity_corrected_chi_square':(abs(52-76)-1)**2/(52+76),
            'continuity_corrected_p':float(chi2.sf((abs(52-76)-1)**2/(52+76),1)),
            'exact_p':float(binomtest(52,128,.5).pvalue),
            'finding':'Paper prints an uncorrected formula but its statistic corresponds to continuity correction'},
        'model_sha256': hashlib.sha256((root/'models/logistic_regression_fixed.pkl').read_bytes()).hexdigest(),
        'experiment_reproduction': 'BLOCKED: no raw study CSV, individual predictions, split manifest, or clinical participant data',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    text=json.dumps(audit(args.root),indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')
    else:
        print(text)


if __name__=='__main__':
    main()
