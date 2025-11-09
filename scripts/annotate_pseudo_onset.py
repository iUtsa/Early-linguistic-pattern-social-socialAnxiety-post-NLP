"""
Annotate pseudo-onset points for temporal drift analysis.

For anxiety users, identifies the first post containing anxiety keywords
and marks it as the onset point. Creates temporal stages:
- pre_onset: posts before onset
- at_onset: the onset post
- post_onset: posts after onset
- control: all control user posts
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('scripts')
from keyword_lists import is_anxiety_term


def annotate_onset(df):
    """
    Annotate onset indices and temporal stages.
    
    Args:
        df: DataFrame with posts
        
    Returns:
        DataFrame with onset_index and temporal_stage columns
    """
    print("Annotating pseudo-onset points...")
    
    # Initialize columns
    df['onset_index'] = np.nan
    df['temporal_stage'] = 'control'
    
    # Process anxiety users only
    anxiety_users = df[df['label'] == 1]['author'].unique()
    
    print(f"Processing {len(anxiety_users)} anxiety users...")
    
    onset_found = 0
    
    for user in tqdm(anxiety_users, desc="Finding onset points"):
        user_posts = df[df['author'] == user].sort_values('posts_seen')
        
        # Find first post with anxiety term
        onset_idx = None
        for idx, row in user_posts.iterrows():
            if is_anxiety_term(row['text']):
                onset_idx = row['posts_seen']
                break
        
        if onset_idx is not None:
            onset_found += 1
            # Mark onset index for all posts of this user
            df.loc[df['author'] == user, 'onset_index'] = onset_idx
            
            # Assign temporal stages
            user_mask = df['author'] == user
            df.loc[user_mask & (df['posts_seen'] < onset_idx), 'temporal_stage'] = 'pre_onset'
            df.loc[user_mask & (df['posts_seen'] == onset_idx), 'temporal_stage'] = 'at_onset'
            df.loc[user_mask & (df['posts_seen'] > onset_idx), 'temporal_stage'] = 'post_onset'
        else:
            # No onset found - mark all as post_onset (conservative)
            df.loc[df['author'] == user, 'temporal_stage'] = 'post_onset'
    
    print(f"\nOnset statistics:")
    print(f"  Anxiety users with detected onset: {onset_found}/{len(anxiety_users)} ({100*onset_found/len(anxiety_users):.1f}%)")
    print(f"\nTemporal stage distribution:")
    print(df['temporal_stage'].value_counts())
    
    return df


def add_relative_position(df):
    """
    Add relative position to onset (posts_seen - onset_index).
    Useful for onset-aligned visualizations.
    """
    print("Adding relative position to onset...")
    
    df['rel_to_onset'] = np.nan
    
    # Only for users with detected onset
    has_onset = df['onset_index'].notna()
    df.loc[has_onset, 'rel_to_onset'] = df.loc[has_onset, 'posts_seen'] - df.loc[has_onset, 'onset_index']
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Annotate pseudo-onset points')
    parser.add_argument('--in_csv', type=str, required=True,
                        help='Input posts CSV (from build_rmhd.py)')
    parser.add_argument('--out_csv', type=str, required=True,
                        help='Output path for annotated CSV')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Annotating Pseudo-Onset Points")
    print("="*60)
    
    # Load data
    print(f"Loading {args.in_csv}...")
    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df)} posts from {df['author'].nunique()} users")
    
    # Annotate onset
    df = annotate_onset(df)
    
    # Add relative position
    df = add_relative_position(df)
    
    # Save
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved annotated data to {args.out_csv}")
    
    print("="*60)
    print("Annotation complete!")
    print("="*60)


if __name__ == '__main__':
    main()