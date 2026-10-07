"""Participant-level effect sizes and multiplicity correction from actual measurements."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln


def hedges_g(positive, control):
    """Positive minus control, using pooled sample SD and exact small-sample correction."""
    positive, control = np.asarray(positive, dtype=float), np.asarray(control, dtype=float)
    if min(len(positive), len(control)) < 2 or not np.isfinite(np.concatenate([positive, control])).all():
        raise ValueError('Effect sizes require >=2 finite observations in each group')
    degrees = len(positive) + len(control) - 2
    pooled = np.sqrt(((len(positive)-1)*positive.var(ddof=1) +
                      (len(control)-1)*control.var(ddof=1))/degrees)
    if pooled == 0:
        raise ValueError('Effect size is undefined when within-group variation is zero')
    correction = np.exp(gammaln(degrees/2) - .5*np.log(degrees/2) - gammaln((degrees-1)/2))
    return float(correction * (positive.mean()-control.mean()) / pooled)


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if np.any((pvalues < 0) | (pvalues > 1)) or not np.isfinite(pvalues).all():
        raise ValueError('FDR adjustment requires finite p-values in [0,1]')
    order = np.argsort(pvalues)
    sorted_adjusted = np.minimum.accumulate((pvalues[order]*len(pvalues)/np.arange(1,len(pvalues)+1))[::-1])[::-1]
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(sorted_adjusted, 1)
    return adjusted


def participant_feature_analysis(data, features, participant_col='participant_id', outcome_col='label'):
    """No imputed study effects; require one independent participant per row."""
    required = {participant_col, outcome_col, *features}
    if not required <= set(data):
        raise ValueError(f'Missing columns: {sorted(required-set(data))}')
    if data[participant_col].isna().any() or data[participant_col].duplicated().any():
        raise ValueError('Provide exactly one row per observed participant, not transcript turns')
    if data[outcome_col].isna().any() or set(data[outcome_col]) != {0,1}:
        raise ValueError('Both outcome groups must be observed and explicitly defined')
    results=[]
    for name in features:
        positive=data.loc[data[outcome_col].eq(1),name].to_numpy(dtype=float)
        control=data.loc[data[outcome_col].eq(0),name].to_numpy(dtype=float)
        g=hedges_g(positive,control)
        test=stats.ttest_ind(positive,control,equal_var=False)
        results.append({'feature':name,'positive_n':len(positive),'control_n':len(control),
                        'positive_mean':float(positive.mean()),'control_mean':float(control.mean()),
                        'hedges_g_positive_minus_control':g,'welch_p':float(test.pvalue)})
    result=pd.DataFrame(results)
    result['bh_q']=benjamini_hochberg(result['welch_p'])
    return result
