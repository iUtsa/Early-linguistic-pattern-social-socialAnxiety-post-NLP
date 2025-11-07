"""
Utility functions for configuration, seeding, and I/O.
"""

import yaml
import random
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # If using torch: torch.manual_seed(seed)


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_metrics_csv(metrics_dict, output_path):
    """
    Save metrics dictionary to CSV.
    
    Args:
        metrics_dict: Dictionary of metrics (can be nested)
        output_path: Path to save CSV
    """
    ensure_dir(Path(output_path).parent)
    
    # Flatten nested dict if needed
    flat_dict = {}
    for key, val in metrics_dict.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat_dict[f"{key}_{subkey}"] = subval
        else:
            flat_dict[key] = val
    
    df = pd.DataFrame([flat_dict])
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")


def append_metrics_csv(metrics_dict, output_path):
    """
    Append metrics to existing CSV or create new one.
    
    Args:
        metrics_dict: Dictionary of metrics
        output_path: Path to CSV file
    """
    ensure_dir(Path(output_path).parent)
    
    # Flatten nested dict
    flat_dict = {}
    for key, val in metrics_dict.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat_dict[f"{key}_{subkey}"] = subval
        else:
            flat_dict[key] = val
    
    df_new = pd.DataFrame([flat_dict])
    
    # Append or create
    if Path(output_path).exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_path, index=False)
    else:
        df_new.to_csv(output_path, index=False)
    
    print(f"Appended metrics to {output_path}")


def print_dict(d, indent=0):
    """Pretty print nested dictionary."""
    for key, val in d.items():
        if isinstance(val, dict):
            print("  " * indent + f"{key}:")
            print_dict(val, indent + 1)
        else:
            print("  " * indent + f"{key}: {val}")


def format_metrics(metrics):
    """Format metrics dictionary for nice printing."""
    formatted = {}
    for key, val in metrics.items():
        if isinstance(val, float):
            formatted[key] = f"{val:.4f}"
        elif isinstance(val, dict):
            formatted[key] = format_metrics(val)
        else:
            formatted[key] = val
    return formatted