"""
Test trained models on external datasets (Kaggle Anxiety, Dreaddit).

This script loads a trained model and evaluates it on external test sets
for domain shift analysis and robustness checking.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import load_config, ensure_dir, save_metrics_csv
from src.preprocess import preprocess_corpus
from src.features import extract_features_batch, get_feature_names
from src.embeds import encode_texts
from src.train import load_model_checkpoint
from src.evaluate import evaluate_model
from sklearn.preprocessing import StandardScaler


def load_kaggle_anxiety(csv_path):
    """
    Load Kaggle Reddit Anxiety Posts dataset.
    
    Expected columns: text (or similar)
    All posts are anxiety class (label=1)
    
    Args:
        csv_path: Path to Kaggle CSV
        
    Returns:
        DataFrame with text and label columns
    """
    print(f"\nLoading Kaggle Anxiety dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Find text column (flexible naming)
    text_col = None
    for col in ['text', 'post', 'body', 'content', 'selftext']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"No text column found. Available columns: {df.columns.tolist()}")
    
    # Normalize
    df = df.rename(columns={text_col: 'text'})
    df = df[['text']].copy()
    df['label'] = 1  # All anxiety posts
    df['source'] = 'kaggle_anxiety'
    
    # Remove NaN
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip().str.len() > 10]
    
    print(f"  Loaded {len(df)} anxiety posts")
    
    return df


def load_dreaddit(csv_path):
    """
    Load Dreaddit stress dataset.
    
    Expected columns: text, label (0=no stress, 1=stress)
    
    Args:
        csv_path: Path to Dreaddit CSV
        
    Returns:
        DataFrame with text and label columns
    """
    print(f"\nLoading Dreaddit dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Find text and label columns
    text_col = None
    for col in ['text', 'post', 'body', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    label_col = None
    for col in ['label', 'stress', 'stressed']:
        if col in df.columns:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    # Normalize
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    df = df[['text', 'label']].copy()
    df['source'] = 'dreaddit'
    
    # Remove NaN
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip().str.len() > 10]
    
    print(f"  Loaded {len(df)} posts ({(df['label']==1).sum()} stress, {(df['label']==0).sum()} non-stress)")
    
    return df


def prepare_external_data(df, config, scaler=None):
    """
    Prepare external dataset for testing.
    
    Args:
        df: DataFrame with text and label
        config: Configuration dict
        scaler: Pre-fitted StandardScaler (from training data)
        
    Returns:
        tuple: (X, y) ready for prediction
    """
    print("\nPreparing external data...")
    
    # Preprocess
    df['text_clean'] = preprocess_corpus(df['text'].values, show_progress=True)
    
    # Extract features
    print("Extracting features...")
    features_df = extract_features_batch(df['text_clean'].values)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = encode_texts(
        df['text_clean'].values,
        model_name=config['training']['embedder'],
        batch_size=config['training']['batch_size'],
        use_cache=False  # Don't cache external data
    )
    
    # Reduce dimensions if needed
    from src.embeds import reduce_dimensions
    n_components = config['training'].get('pca_components', 300)
    if embeddings.shape[1] > n_components:
        embeddings, _ = reduce_dimensions(embeddings, n_components)
    
    # Combine
    X = np.hstack([features_df.values, embeddings])
    y = df['label'].values
    
    # Standardize using training scaler
    if scaler is not None:
        print("Applying training scaler...")
        X = scaler.transform(X)
    
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    return X, y


def test_on_external(model_path, external_csv, dataset_name, config, scaler_path=None):
    """
    Test trained model on external dataset.
    
    Args:
        model_path: Path to trained model
        external_csv: Path to external dataset CSV
        dataset_name: Name of dataset ('kaggle_anxiety' or 'dreaddit')
        config: Configuration dict
        scaler_path: Optional path to saved scaler
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print(f"Testing on {dataset_name.upper()}")
    print("="*60)
    
    # Load model
    model = load_model_checkpoint(model_path)
    
    # Load scaler if provided
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
    
    # Load external data
    if dataset_name == 'kaggle_anxiety':
        df = load_kaggle_anxiety(external_csv)
    elif dataset_name == 'dreaddit':
        df = load_dreaddit(external_csv)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Prepare data
    X, y = prepare_external_data(df, config, scaler)
    
    # Evaluate
    metrics = evaluate_model(model, X, y, split_name=dataset_name)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test on external datasets')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (e.g., models/checkpoints/lr_full.pkl)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['kaggle_anxiety', 'dreaddit'],
                        help='External dataset name')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to external dataset CSV')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to fitted scaler (optional)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results CSV')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Test on external dataset
    metrics = test_on_external(
        args.model,
        args.csv,
        args.dataset,
        config,
        args.scaler
    )
    
    # Save results
    if args.output:
        ensure_dir(Path(args.output).parent)
        save_metrics_csv(metrics, args.output)
    else:
        # Default save location
        output_path = f"models/results/external_{args.dataset}.csv"
        ensure_dir(Path(output_path).parent)
        save_metrics_csv(metrics, output_path)
    
    print("\n" + "="*60)
    print("External testing complete!")
    print("="*60)


if __name__ == '__main__':
    main()


"""
USAGE EXAMPLES:

# Test on Kaggle Anxiety dataset
python scripts/test_external.py \
  --model models/checkpoints/lr_full.pkl \
  --dataset kaggle_anxiety \
  --csv /path/to/kaggle_anxiety.csv \
  --output models/results/external_kaggle.csv

# Test on Dreaddit dataset
python scripts/test_external.py \
  --model models/checkpoints/lr_full.pkl \
  --dataset dreaddit \
  --csv /path/to/dreaddit.csv \
  --output models/results/external_dreaddit.csv

# With scaler (recommended for better performance)
python scripts/test_external.py \
  --model models/checkpoints/lr_full.pkl \
  --dataset kaggle_anxiety \
  --csv /path/to/kaggle_anxiety.csv \
  --scaler models/checkpoints/scaler_full.pkl

Note: You need to save the scaler separately. Add this to src/run.py:
  from src.train import save_model_checkpoint
  save_model_checkpoint(data['scaler'], 'models/checkpoints/scaler_full.pkl')
"""