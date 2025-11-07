"""
Model explainability using SHAP and LIME.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path


def explain_with_shap(model, X, feature_names=None, max_display=20, save_path=None):
    """
    Generate SHAP explanations for model.
    
    Args:
        model: Trained model
        X: Features (use a sample if large)
        feature_names: List of feature names
        max_display: Maximum features to display
        save_path: Path to save plot
        
    Returns:
        shap_values: SHAP values
    """
    print("\nGenerating SHAP explanations...")
    
    # Sample data if too large
    if len(X) > 1000:
        print(f"  Sampling 1000 instances from {len(X)} for SHAP...")
        indices = np.random.choice(len(X), size=1000, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Create explainer (handle calibrated models)
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        base_model = model.calibrated_classifiers_[0].base_estimator
    else:
        base_model = model
    
    # Use appropriate explainer
    if hasattr(base_model, 'coef_'):
        # Linear models
        explainer = shap.LinearExplainer(base_model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Tree-based or other models
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_sample)
        except:
            # Fallback to KernelExplainer (slower)
            print("  Using KernelExplainer (slower)...")
            explainer = shap.KernelExplainer(base_model.predict, shap.sample(X_sample, 100))
            shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class
    
    # Create summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  Saved SHAP plot to {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()
    
    return shap_values


def get_top_shap_features(shap_values, feature_names, top_k=10):
    """
    Get top features by mean absolute SHAP value.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        top_k: Number of top features
        
    Returns:
        list: Top features with their mean |SHAP| values
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
    
    # Sort by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    
    top_features = [
        (feature_names[i], mean_abs_shap[i])
        for i in sorted_indices
    ]
    
    print(f"\nTop {top_k} features by mean |SHAP|:")
    for i, (name, value) in enumerate(top_features):
        print(f"  {i+1:2d}. {name:30s} {value:.4f}")
    
    return top_features


def explain_with_lime(model, X_train, X_test, feature_names=None, num_samples=5, save_dir=None):
    """
    Generate LIME explanations for sample instances.
    
    Args:
        model: Trained model
        X_train: Training data (for LIME background)
        X_test: Test data to explain
        feature_names: List of feature names
        num_samples: Number of test samples to explain
        save_dir: Directory to save explanations
        
    Returns:
        list: LIME explanations
    """
    from lime.lime_tabular import LimeTabularExplainer
    
    print(f"\nGenerating LIME explanations for {num_samples} samples...")
    
    # Sample test instances
    if len(X_test) > num_samples:
        indices = np.random.choice(len(X_test), size=num_samples, replace=False)
        X_explain = X_test[indices]
    else:
        X_explain = X_test
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Control', 'Anxiety'],
        mode='classification'
    )
    
    explanations = []
    
    for i, instance in enumerate(X_explain):
        print(f"  Explaining instance {i+1}/{len(X_explain)}...")
        
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=10
        )
        
        explanations.append(exp)
        
        # Save if requested
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig = exp.as_pyplot_figure()
            fig.savefig(f"{save_dir}/lime_explanation_{i+1}.png", bbox_inches='tight', dpi=300)
            plt.close(fig)
    
    if save_dir:
        print(f"  Saved LIME explanations to {save_dir}")
    
    return explanations


def plot_feature_importance_bar(importances_dict, top_k=20, save_path=None):
    """
    Plot feature importances as horizontal bar chart.
    
    Args:
        importances_dict: Dictionary of {feature_name: importance}
        top_k: Number of top features to show
        save_path: Path to save plot
    """
    # Sort by absolute value
    sorted_items = sorted(importances_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in values]
    y_pos = np.arange(len(features))
    
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Coefficient / Importance', fontsize=12)
    ax.set_title(f'Top {top_k} Feature Importances', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Positive (↑ anxiety)'),
        Patch(facecolor='#d62728', label='Negative (↓ anxiety)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
    
    plt.close()