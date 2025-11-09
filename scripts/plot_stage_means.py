"""
Plot feature means across temporal stages.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_stage_means(data, features, output_path):
    """
    Plot feature means across stages.
    
    Args:
        data: DataFrame with features and temporal_stage
        features: List of feature names
        output_path: Save path
    """
    # Calculate means by stage
    stage_order = ['control', 'pre_onset', 'at_onset', 'post_onset']
    stage_means = data.groupby('temporal_stage')[features].mean()
    
    # Reorder if all stages present
    available_stages = [s for s in stage_order if s in stage_means.index]
    stage_means = stage_means.loc[available_stages]
    
    # Create subplots
    n_features = len(features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        values = stage_means[feature].values
        ax.plot(range(len(available_stages)), values, 'o-', linewidth=2, markersize=8)
        
        ax.set_xticks(range(len(available_stages)))
        ax.set_xticklabels([s.replace('_', '\n') for s in available_stages], fontsize=10)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{feature.replace("_", " ").title()} Across Stages', 
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved stage means plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot feature means by stage')
    parser.add_argument('--data_csv', type=str,
                        default='data/processed/posts_with_features.csv',
                        help='Path to posts with features CSV')
    parser.add_argument('--output', type=str,
                        default='data/reports/feature_means_by_stage.png',
                        help='Output plot path')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_csv}...")
    data = pd.read_csv(args.data_csv)
    
    if 'temporal_stage' not in data.columns:
        print("Error: Data must have 'temporal_stage' column")
        print("Run: python scripts/annotate_pseudo_onset.py first")
        return
    
    # Select key features
    features = ['vader_compound', 'fp_pronoun_rate', 'char_count', 'textblob_polarity']
    
    print(f"Plotting feature means across stages...")
    plot_stage_means(data, features, args.output)
    
    print(f"✓ Plot saved to {args.output}")


if __name__ == '__main__':
    main()