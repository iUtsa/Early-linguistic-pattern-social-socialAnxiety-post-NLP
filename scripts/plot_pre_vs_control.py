"""
Plot KDE distributions comparing pre_onset vs control for key features.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_kde_comparison(data, feature, stage1='control', stage2='pre_onset', output_path=None):
    """
    Plot KDE comparison between two stages.
    
    Args:
        data: DataFrame with features and temporal_stage
        feature: Feature name
        stage1: First stage name
        stage2: Second stage name
        output_path: Save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data
    data1 = data[data['temporal_stage'] == stage1][feature].dropna()
    data2 = data[data['temporal_stage'] == stage2][feature].dropna()
    
    # Plot KDE
    data1.plot.kde(ax=ax, linewidth=2.5, label=stage1.replace('_', ' ').title(), color='#1f77b4')
    data2.plot.kde(ax=ax, linewidth=2.5, label=stage2.replace('_', ' ').title(), color='#ff7f0e')
    
    # Add histograms with transparency
    ax.hist(data1, bins=30, alpha=0.2, density=True, color='#1f77b4')
    ax.hist(data2, bins=30, alpha=0.2, density=True, color='#ff7f0e')
    
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{feature.replace("_", " ").title()}: {stage1.replace("_", " ").title()} vs. {stage2.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add statistics
    mean1, mean2 = data1.mean(), data2.mean()
    std1, std2 = data1.std(), data2.std()
    
    stats_text = f'{stage1}: μ={mean1:.3f}, σ={std1:.3f}\n{stage2}: μ={mean2:.3f}, σ={std2:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved KDE plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot KDE comparisons')
    parser.add_argument('--data_csv', type=str,
                        default='data/processed/posts_with_features.csv',
                        help='Path to posts with features CSV')
    parser.add_argument('--output_dir', type=str,
                        default='data/reports',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_csv}...")
    data = pd.read_csv(args.data_csv)
    
    if 'temporal_stage' not in data.columns:
        print("Error: Data must have 'temporal_stage' column")
        return
    
    # Plot pronoun comparison
    print("Plotting pronoun KDE...")
    plot_kde_comparison(
        data, 
        'fp_pronoun_rate',
        'control',
        'pre_onset',
        f"{args.output_dir}/kde_pronoun_pre_vs_control.png"
    )
    
    # Plot sentiment comparison
    print("Plotting sentiment KDE...")
    plot_kde_comparison(
        data,
        'vader_compound',
        'control',
        'pre_onset',
        f"{args.output_dir}/kde_sentiment_pre_vs_control.png"
    )
    
    print("✓ KDE plots complete!")


if __name__ == '__main__':
    main()