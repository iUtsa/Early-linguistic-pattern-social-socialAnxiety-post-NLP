#!/bin/bash

# Run all plotting scripts

echo "=========================================="
echo "Generating all plots"
echo "=========================================="

# Check if data files exist
if [ ! -f "data/processed/posts_with_features.csv" ]; then
    echo "Error: data/processed/posts_with_features.csv not found"
    echo "Run experiments first: python src/run.py --mode full"
    exit 1
fi

# Plot early-slice results (if available)
if [ -f "models/results/early_slice.csv" ]; then
    echo ""
    echo "Plotting early-slice results..."
    python scripts/plot_early_slice.py
else
    echo "Skipping early-slice plot (no results found)"
fi

# Check if staged data exists
if [ -f "data/processed/rmhd_posts_with_stages.csv" ]; then
    STAGED_CSV="data/processed/rmhd_posts_with_stages.csv"
elif [ -f "data/processed/posts_with_features.csv" ]; then
    # Check if it has temporal_stage column
    STAGED_CSV="data/processed/posts_with_features.csv"
else
    echo ""
    echo "Warning: No staged data found"
    echo "Run: python scripts/annotate_pseudo_onset.py"
    echo "Skipping temporal analysis plots"
    exit 0
fi

# Plot stage means
echo ""
echo "Plotting feature means by stage..."
python scripts/plot_stage_means.py --data_csv "$STAGED_CSV"

# Plot pre vs control KDEs
echo ""
echo "Plotting KDE comparisons..."
python scripts/plot_pre_vs_control.py --data_csv "$STAGED_CSV"

# Plot onset-aligned trajectories
echo ""
echo "Plotting onset-aligned trajectories..."
python scripts/plot_onset_aligned.py --data_csv "$STAGED_CSV"

echo ""
echo "=========================================="
echo "All plots generated!"
echo "Check: data/reports/"
echo "=========================================="