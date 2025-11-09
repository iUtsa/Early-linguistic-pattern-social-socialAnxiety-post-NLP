"""
Plot onset-aligned trajectories of features.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_onset_aligned(data, feature, window=(-6, 6), output_path=None):
    """
    Plot feature trajectory aligned to onset point.
    
    Args:
        data: DataFrame with rel_to_onset and feature columns
        feature: Feature name
        window: Tuple of (min_rel, max_rel) positions
        output_path: Save path
    """
    # Filter to anxiety users with detected onset
    anxiety_data = data[(data['label'] == 1) & (data['onset_index'].notna())].copy()
    
    if len(anxiety_data) == 0:
        print(f"No anxiety users with detected onset found")
        return
    
    # Filter to window
    anxiety_data = anxiety_data[
        (anxiety_data['rel_to_onset'] >= window[0]) & 
        (anxiety_data['rel_to_onset'] <= window[1])
    ]
    
    # Calculate mean and SEM by relative position
    grouped = anxiety_data.groupby('rel_to_onset')[feature].agg(['mean', 'sem', 'count'])
    grouped = grouped.sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line
    ax.plot(grouped.index, grouped['mean'], 'o-', linewidth=2.5, markersize=7, 
            color='#d62728', label='Anxiety users')
    
    # Add confidence interval (mean ± SEM)
    ax.fill_between(
        grouped.index,
        grouped['mean'] - grouped['sem'],
        grouped['mean'] + grouped['sem'],
        alpha=0.3,
        color='#d62728'
    )
    
    # Mark onset point
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Onset', alpha=0.7)
    
    ax.set_xlabel('Posts Relative to Onset', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{feature.replace("_", " ").title()} Trajectory Around Onset',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add sample sizes at bottom
    for pos, row in grouped.iterrows():
        if pos % 2 == 0:  # Show every other position to avoid crowding
            ax.text(pos, ax.get_ylim()[0], f'n={int(row["count"])}',
                   ha='center', va='top', fontsize=8, alpha=0.6)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved onset-aligned plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot onset-aligned trajectories')
    parser.add_argument('--data_csv', type=str,
                        default='data/processed/rmhd_posts_with_stages.csv',
                        help='Path to staged posts CSV')
    parser.add_argument('--output_dir', type=str,
                        default='data/reports',
                        help='Output directory')
    parser.add_argument('--window', type=int, nargs=2, default=[-6, 6],
                        help='Window around onset (min max)')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_csv}...")
    
    # Try loading from staged CSV first
    try:
        data = pd.read_csv(args.data_csv)
    except FileNotFoundError:
        print(f"Staged CSV not found: {args.data_csv}")
        print("Trying processed features CSV...")
        try:
            data = pd.read_csv('data/processed/posts_with_features.csv')
        except FileNotFoundError:
            print("Error: No data file found")
            print("Run: python scripts/annotate_pseudo_onset.py first")
            return
    
    if 'rel_to_onset' not in data.columns:
        print("Error: Data must have 'rel_to_onset' column")
        print("Run: python scripts/annotate_pseudo_onset.py first")
        return
    
    # Plot sentiment trajectory
    print("Plotting sentiment trajectory...")
    plot_onset_aligned(
        data,
        'vader_compound',
        window=tuple(args.window),
        output_path=f"{args.output_dir}/onset_aligned_sentiment.png"
    )
    
    # Plot pronoun trajectory
    print("Plotting pronoun trajectory...")
    plot_onset_aligned(
        data,
        'fp_pronoun_rate',
        window=tuple(args.window),
        output_path=f"{args.output_dir}/onset_aligned_pronouns.png"
    )
    
    print("✓ Onset-aligned plots complete!")


if __name__ == '__main__':
    main()