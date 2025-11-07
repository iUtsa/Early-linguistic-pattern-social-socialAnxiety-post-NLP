"""
Model evaluation utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def evaluate_model(model, X, y, split_name='test'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        split_name: Name of split for display
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating on {split_name} set...")
    
    # Predictions
    y_pred = model.predict(X)
    
    # Probabilities (if available)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X)
        # Normalize to [0, 1]
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
    else:
        y_proba = None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Print summary
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn:4d}  FP: {fp:4d}")
    print(f"    FN: {fn:4d}  TP: {tp:4d}")
    
    return metrics


def evaluate_calibration(model, X, y, n_bins=10):
    """
    Evaluate model calibration using Expected Calibration Error (ECE).
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        n_bins: Number of bins for calibration
        
    Returns:
        dict: Calibration metrics
    """
    if not hasattr(model, 'predict_proba'):
        print("Model does not support probability prediction")
        return {}
    
    y_proba = model.predict_proba(X)[:, 1]
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate ECE
    ece = 0.0
    bin_stats = []
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        
        bin_proba = y_proba[mask]
        bin_labels = y[mask]
        
        mean_proba = bin_proba.mean()
        mean_label = bin_labels.mean()
        bin_size = mask.sum()
        
        ece += (bin_size / len(y)) * abs(mean_proba - mean_label)
        
        bin_stats.append({
            'bin': bin_idx,
            'mean_proba': mean_proba,
            'mean_label': mean_label,
            'size': int(bin_size)
        })
    
    metrics = {
        'ece': ece,
        'n_bins': n_bins,
        'bin_stats': bin_stats
    }
    
    print(f"\nCalibration Metrics:")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    
    return metrics


def compare_models(models_dict, X, y, split_name='test'):
    """
    Compare multiple models on same dataset.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X: Features
        y: Labels
        split_name: Name of split
        
    Returns:
        dict: Results for each model
    """
    print(f"\n{'='*60}")
    print(f"Comparing Models on {split_name} set")
    print("="*60)
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"\n{name}:")
        metrics = evaluate_model(model, X, y, split_name)
        results[name] = metrics
    
    # Summary table
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Model':<15} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-"*60)
    
    for name, metrics in results.items():
        auc_str = f"{metrics['roc_auc']:.4f}" if 'roc_auc' in metrics else "N/A"
        print(f"{name:<15} {metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {auc_str:>8}")
    
    return results


def get_classification_report(model, X, y, target_names=None):
    """
    Get detailed classification report.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        target_names: List of class names
        
    Returns:
        str: Classification report
    """
    if target_names is None:
        target_names = ['Control', 'Anxiety']
    
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=target_names)
    
    return report