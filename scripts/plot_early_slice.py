"""
Plot early-slice F1 scores vs. number of posts.
"""

import argparse
import pandas as pd
from src.plots import plot_early_slice_results
from src.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description='Plot early-slice results')
    parser.add_argument('--results_csv', type=str, 
                        default='models/results/early_slice.csv',
                        help='Path to early-slice results CSV')
    parser.add_argument('--output', type=str,
                        default='data/reports/early_slice_f1.png',
                        help='Output plot path')
    parser.add_argument('--metric', type=str, default='f1',
                        help='Metric to plot')
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.results_csv}...")
    results_df = pd.read_csv(args.results_csv)
    
    print(f"Plotting {args.metric} vs. k...")
    ensure_dir(args.output)
    plot_early_slice_results(results_df, metric=args.metric, save_path=args.output)
    
    print(f"✓ Plot saved to {args.output}")


if __name__ == '__main__':
    main()