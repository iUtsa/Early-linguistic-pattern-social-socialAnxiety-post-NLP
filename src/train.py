"""
Model training utilities.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import numpy as np


def get_model(model_name='lr', seed=42):
    """
    Get model instance.
    
    Args:
        model_name: 'lr', 'svm', or 'lgbm'
        seed: Random seed
        
    Returns:
        sklearn-compatible model
    """
    if model_name == 'lr':
        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight='balanced',
            solver='lbfgs'
        )
    
    elif model_name == 'svm':
        return LinearSVC(
            max_iter=2000,
            random_state=seed,
            class_weight='balanced',
            dual=False
        )
    
    elif model_name == 'lgbm':
        return lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=seed,
            class_weight='balanced',
            verbose=-1
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(X_train, y_train, model_name='lr', seed=42, calibrate=False, X_val=None, y_val=None):
    """
    Train a model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Model type
        seed: Random seed
        calibrate: Whether to calibrate probabilities
        X_val: Validation features (for calibration)
        y_val: Validation labels (for calibration)
        
    Returns:
        Trained model
    """
    print(f"\nTraining {model_name.upper()} model...")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Train class distribution: {np.bincount(y_train)}")
    
    model = get_model(model_name, seed)
    model.fit(X_train, y_train)
    
    print(f"  Training complete")
    
    # Calibrate if requested
    if calibrate and X_val is not None and y_val is not None:
        print("  Calibrating probabilities...")
        model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
        model.fit(X_val, y_val)
        print("  Calibration complete")
    
    return model


def get_model_coefficients(model, feature_names=None):
    """
    Extract model coefficients (for linear models).
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        dict: Feature name -> coefficient mapping
    """
    # Handle calibrated models
    if isinstance(model, CalibratedClassifierCV):
        base_model = model.calibrated_classifiers_[0].base_estimator
    else:
        base_model = model
    
    # Get coefficients
    if hasattr(base_model, 'coef_'):
        coefs = base_model.coef_[0] if base_model.coef_.ndim > 1 else base_model.coef_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]
        
        coef_dict = {name: coef for name, coef in zip(feature_names, coefs)}
        
        # Sort by absolute value
        coef_dict = dict(sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return coef_dict
    
    return None


def print_top_features(model, feature_names, top_k=10):
    """
    Print top features by coefficient magnitude.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_k: Number of top features to show
    """
    coef_dict = get_model_coefficients(model, feature_names)
    
    if coef_dict is None:
        print("Model does not have interpretable coefficients")
        return
    
    print(f"\nTop {top_k} features by absolute coefficient:")
    for i, (name, coef) in enumerate(list(coef_dict.items())[:top_k]):
        direction = "↑ anxiety" if coef > 0 else "↓ anxiety"
        print(f"  {i+1:2d}. {name:30s} {coef:+.4f}  {direction}")


def save_model_checkpoint(model, path):
    """
    Save model checkpoint.
    
    Args:
        model: Trained model
        path: Save path
    """
    import pickle
    from pathlib import Path
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Saved model to {path}")


def load_model_checkpoint(path):
    """
    Load model checkpoint.
    
    Args:
        path: Path to model file
        
    Returns:
        Loaded model
    """
    import pickle
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded model from {path}")
    
    return model