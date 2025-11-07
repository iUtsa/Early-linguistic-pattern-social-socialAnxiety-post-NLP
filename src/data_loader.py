"""
Data loading utilities for RMHD dataset.
"""

import pandas as pd
from pathlib import Path


def load_posts(posts_csv, users_csv=None, stage_csv=None):
    """
    Load posts and optionally merge with user metadata.
    
    Args:
        posts_csv: Path to posts CSV file
        users_csv: Optional path to users CSV file
        stage_csv: Optional path to posts CSV with temporal_stage column
        
    Returns:
        tuple: (posts_df, users_df or None)
    """
    print(f"\nLoading posts from {posts_csv}...")
    
    # Load posts
    if stage_csv:
        posts = pd.read_csv(stage_csv)
        print(f"Loaded {len(posts)} posts with temporal stages from {stage_csv}")
    else:
        posts = pd.read_csv(posts_csv)
        print(f"Loaded {len(posts)} posts from {posts_csv}")
    
    # Load users if provided
    users = None
    if users_csv:
        users = pd.read_csv(users_csv)
        print(f"Loaded {len(users)} users from {users_csv}")
    
    # Basic info
    print(f"  Posts: {len(posts)}")
    print(f"  Users: {posts['author'].nunique()}")
    print(f"  Label distribution: {posts['label'].value_counts().to_dict()}")
    
    if 'split' in posts.columns:
        print(f"  Split distribution: {posts['split'].value_counts().to_dict()}")
    
    return posts, users


def validate_data(posts):
    """
    Validate posts DataFrame has required columns and structure.
    
    Args:
        posts: Posts DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    print("\nValidating data...")
    
    # Required columns
    required_cols = ['author', 'text', 'label']
    missing_cols = [col for col in required_cols if col not in posts.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for nulls
    null_counts = posts[required_cols].isnull().sum()
    if null_counts.any():
        print("Warning: Found null values:")
        print(null_counts[null_counts > 0])
    
    # Check labels
    unique_labels = posts['label'].unique()
    print(f"  Labels: {sorted(unique_labels)}")
    
    if len(unique_labels) != 2:
        print(f"Warning: Expected 2 classes, found {len(unique_labels)}")
    
    # Check for posts_seen column (temporal ordering)
    if 'posts_seen' not in posts.columns:
        print("Warning: 'posts_seen' column not found. Creating based on chronological order...")
        posts['posts_seen'] = posts.groupby('author').cumcount() + 1
    
    # Check for split column
    if 'split' not in posts.columns:
        print("Warning: 'split' column not found. You may need to create train/val/test splits.")
    
    print("✓ Data validation complete")
    
    return posts
