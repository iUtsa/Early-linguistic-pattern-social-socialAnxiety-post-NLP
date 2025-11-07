"""
Dataset preparation and feature combination.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from features import extract_features_batch, aggregate_user_features, get_feature_names
from embeds import encode_texts, aggregate_user_embeddings, reduce_dimensions
from preprocess import preprocess_corpus


def prepare_full_dataset(posts, config, use_embeddings=True, use_features=True):
    """
    Prepare full dataset with all posts per user.
    
    Args:
        posts: Posts DataFrame
        config: Configuration dict
        use_embeddings: Include embeddings
        use_features: Include linguistic features
        
    Returns:
        dict: Prepared datasets with train/val/test splits
    """
    print("\n" + "="*60)
    print("Preparing Full Dataset")
    print("="*60)
    
    # Preprocess text
    print("Preprocessing text...")
    posts['text_clean'] = preprocess_corpus(posts['text'].values, show_progress=True)
    
    # Extract linguistic features
    if use_features:
        print("\nExtracting linguistic features...")
        features_df = extract_features_batch(posts['text_clean'].values)
        user_features, user_labels, user_splits = aggregate_user_features(features_df, posts)
    else:
        user_features = None
    
    # Generate embeddings
    if use_embeddings:
        print("\nGenerating embeddings...")
        embeddings = encode_texts(
            posts['text_clean'].values,
            model_name=config['training']['embedder'],
            batch_size=config['training']['batch_size']
        )
        
        # Aggregate to user level
        user_embeddings, user_labels, user_splits = aggregate_user_embeddings(
            embeddings, posts
        )
        
        # Reduce dimensions
        n_components = config['training'].get('pca_components', 300)
        user_embeddings, _ = reduce_dimensions(user_embeddings, n_components)
    else:
        user_embeddings = None
    
    # Combine features and embeddings
    if use_features and use_embeddings:
        print("\nCombining features and embeddings...")
        X = np.hstack([user_features.values, user_embeddings])
    elif use_features:
        X = user_features.values
    elif use_embeddings:
        X = user_embeddings
    else:
        raise ValueError("Must use at least features or embeddings")
    
    # Split data
    data = {}
    for split in ['train', 'val', 'test']:
        mask = user_splits == split
        data[f'{split}_X'] = X[mask]
        data[f'{split}_y'] = user_labels[mask]
        print(f"{split}: {data[f'{split}_X'].shape}")
    
    # Standardize features (fit on train, transform all)
    print("\nStandardizing features...")
    scaler = StandardScaler()
    data['train_X'] = scaler.fit_transform(data['train_X'])
    data['val_X'] = scaler.transform(data['val_X'])
    data['test_X'] = scaler.transform(data['test_X'])
    
    data['scaler'] = scaler
    data['feature_names'] = get_feature_names() if use_features else []
    
    return data


def prepare_early_slice_dataset(posts, k, config, use_embeddings=True, use_features=True):
    """
    Prepare dataset using only first k posts per user.
    
    Args:
        posts: Posts DataFrame
        k: Number of posts to use per user
        config: Configuration dict
        use_embeddings: Include embeddings
        use_features: Include linguistic features
        
    Returns:
        dict: Prepared datasets with train/val/test splits
    """
    print(f"\n{'='*60}")
    print(f"Preparing Early-Slice Dataset (k={k})")
    print("="*60)
    
    # Filter to first k posts
    posts_k = posts[posts['posts_seen'] <= k].copy()
    print(f"Using first {k} posts: {len(posts_k)} posts from {posts_k['author'].nunique()} users")
    
    # Check if all users have at least k posts
    user_post_counts = posts_k.groupby('author').size()
    users_with_k = (user_post_counts == k).sum()
    print(f"Users with exactly {k} posts: {users_with_k}/{len(user_post_counts)}")
    
    # Prepare dataset (same as full, but with filtered posts)
    return prepare_full_dataset(posts_k, config, use_embeddings, use_features)


def prepare_stage_dataset(posts, stage, config, use_embeddings=True, use_features=True):
    """
    Prepare dataset for a specific temporal stage.
    
    Args:
        posts: Posts DataFrame with temporal_stage column
        stage: Stage name ('pre_onset', 'post_onset', 'control')
        config: Configuration dict
        use_embeddings: Include embeddings
        use_features: Include linguistic features
        
    Returns:
        dict: Prepared datasets
    """
    print(f"\n{'='*60}")
    print(f"Preparing Stage Dataset: {stage}")
    print("="*60)
    
    # Filter to stage
    if 'temporal_stage' not in posts.columns:
        raise ValueError("Posts must have 'temporal_stage' column")
    
    stage_posts = posts[posts['temporal_stage'] == stage].copy()
    print(f"Stage '{stage}': {len(stage_posts)} posts from {stage_posts['author'].nunique()} users")
    
    if len(stage_posts) == 0:
        raise ValueError(f"No posts found for stage: {stage}")
    
    # Prepare dataset
    return prepare_full_dataset(stage_posts, config, use_embeddings, use_features)


def save_processed_features(posts, output_path='data/processed/posts_with_features.csv'):
    """
    Save posts with extracted features for later analysis.
    
    Args:
        posts: Posts DataFrame
        output_path: Output CSV path
    """
    from pathlib import Path
    from preprocess import preprocess_corpus
    from features import extract_features_batch
    
    print(f"\nSaving processed features to {output_path}...")
    
    # Preprocess if not already done
    if 'text_clean' not in posts.columns:
        posts['text_clean'] = preprocess_corpus(posts['text'].values, show_progress=True)
    
    # Extract features
    features_df = extract_features_batch(posts['text_clean'].values)
    
    # Combine with original data
    combined = pd.concat([posts.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    print(f"Saved {len(combined)} posts with features")
    
    return combined