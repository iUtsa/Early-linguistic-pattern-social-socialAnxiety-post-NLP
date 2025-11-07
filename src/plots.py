"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_confusion_matrix(y_true, y_pred, classes=['Control', 'Anxiety'], 
                          normalize=False, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        normalize: Whether to normalize
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_calibration_curve(y_true, y_proba, n_bins=10, save_path=None):
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save figure
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate calibration
    bin_centers = []
    bin_true_freqs = []
    bin_counts = []
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        
        bin_centers.append(y_proba[mask].mean())
        bin_true_freqs.append(y_true[mask].mean())
        bin_counts.append(mask.sum())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calibration curve
    ax.plot(bin_centers, bin_true_freqs, 'o-', linewidth=2, 
            markersize=8, label='Model', color='#2ca02c')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved calibration curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_early_slice_results(results_df, metric='f1', save_path=None):
    """
    Plot performance vs. number of posts (early-slice analysis).
    
    Args:
        results_df: DataFrame with columns ['k', metric columns]
        metric: Metric to plot
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot train and val if available
    if f'train_{metric}' in results_df.columns:
        ax.plot(results_df['k'], results_df[f'train_{metric}'], 
                'o-', linewidth=2, markersize=8, label='Train', color='#1f77b4')
    
    if f'val_{metric}' in results_df.columns:
        ax.plot(results_df['k'], results_df[f'val_{metric}'], 
                's-', linewidth=2, markersize=8, label='Validation', color='#ff7f0e')
    
    if f'test_{metric}' in results_df.columns:
        ax.plot(results_df['k'], results_df[f'test_{metric}'], 
                '^-', linewidth=2, markersize=8, label='Test', color='#2ca02c')
    
    ax.set_xlabel('Number of Posts (k)', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} Score vs. Number of Posts', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(results_df['k'])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved early-slice plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_distribution(data, feature, hue='label', bins=30, 
                              save_path=None, title=None):
    """
    Plot feature distribution by class.
    
    Args:
        data: DataFrame with feature and label columns
        feature: Feature name
        hue: Column for grouping (default: 'label')
        bins: Number of histogram bins
        save_path: Path to save figure
        title: Custom title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE for each class
    for label in data[hue].unique():
        subset = data[data[hue] == label][feature].dropna()
        label_name = 'Anxiety' if label == 1 else 'Control'
        ax.hist(subset, bins=bins, alpha=0.5, label=label_name, density=True)
        subset.plot.kde(ax=ax, linewidth=2)
    
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title or f'Distribution of {feature}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()