"""
Statistical testing utilities.
"""

import numpy as np
from scipy import stats


def welch_t_test(group1, group2):
    """
    Perform Welch's t-test (unequal variances).
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        dict: Test results
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    # Perform test
    statistic, pvalue = stats.ttest_ind(group1, group2, equal_var=False)
    
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': pvalue < 0.05,
        'mean1': group1.mean(),
        'mean2': group2.mean(),
        'std1': group1.std(),
        'std2': group2.std(),
        'n1': len(group1),
        'n2': len(group2)
    }


def mann_whitney_u_test(group1, group2):
    """
    Perform Mann-Whitney U test (non-parametric).
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        dict: Test results
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    # Perform test
    statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': pvalue < 0.05,
        'median1': np.median(group1),
        'median2': np.median(group2),
        'n1': len(group1),
        'n2': len(group2)
    }


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        float: Cohen's d
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def cliffs_delta(group1, group2):
    """
    Calculate Cliff's Delta effect size (non-parametric).
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        float: Cliff's Delta (range: -1 to 1)
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    n1, n2 = len(group1), len(group2)
    
    # Count comparisons
    greater = 0
    less = 0
    
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    
    delta = (greater - less) / (n1 * n2)
    
    return delta


def compare_groups(group1, group2, group1_name='Group 1', group2_name='Group 2'):
    """
    Comprehensive comparison of two groups.
    
    Args:
        group1: First group values
        group2: Second group values
        group1_name: Name of first group
        group2_name: Name of second group
        
    Returns:
        dict: Comparison results
    """
    # Parametric test
    t_test = welch_t_test(group1, group2)
    
    # Non-parametric test
    mw_test = mann_whitney_u_test(group1, group2)
    
    # Effect sizes
    d = cohens_d(group1, group2)
    delta = cliffs_delta(group1, group2)
    
    results = {
        'welch_t_test': t_test,
        'mann_whitney_u': mw_test,
        'cohens_d': d,
        'cliffs_delta': delta
    }
    
    # Print summary
    print(f"\nComparing {group1_name} vs. {group2_name}:")
    print(f"  {group1_name}: mean={t_test['mean1']:.4f}, std={t_test['std1']:.4f}, n={t_test['n1']}")
    print(f"  {group2_name}: mean={t_test['mean2']:.4f}, std={t_test['std2']:.4f}, n={t_test['n2']}")
    print(f"\n  Welch's t-test: t={t_test['statistic']:.4f}, p={t_test['pvalue']:.4f} {'*' if t_test['significant'] else ''}")
    print(f"  Mann-Whitney U: U={mw_test['statistic']:.0f}, p={mw_test['pvalue']:.4f} {'*' if mw_test['significant'] else ''}")
    print(f"  Cohen's d: {d:.4f} ({interpret_cohens_d(d)})")
    print(f"  Cliff's Delta: {delta:.4f} ({interpret_cliffs_delta(delta)})")
    
    return results


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_cliffs_delta(delta):
    """Interpret Cliff's Delta effect size."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def compare_multiple_features(data, features, group_col='label', 
                              group1_val=0, group2_val=1,
                              group1_name='Control', group2_name='Anxiety'):
    """
    Compare multiple features between two groups.
    
    Args:
        data: DataFrame
        features: List of feature names
        group_col: Column name for grouping
        group1_val: Value for first group
        group2_val: Value for second group
        group1_name: Name of first group
        group2_name: Name of second group
        
    Returns:
        dict: Results for each feature
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Comparing {group1_name} vs. {group2_name}")
    print("="*60)
    
    for feature in features:
        group1 = data[data[group_col] == group1_val][feature].dropna().values
        group2 = data[data[group_col] == group2_val][feature].dropna().values
        
        if len(group1) == 0 or len(group2) == 0:
            print(f"\nSkipping {feature}: insufficient data")
            continue
        
        print(f"\n{feature}:")
        results[feature] = compare_groups(group1, group2, group1_name, group2_name)
    
    return results