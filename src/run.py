"""
Main orchestrator script for all experiments.
"""

import sys
import os
from pathlib import Path

# When run from src/ directory, go up to project root
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()

# Change to project root so all relative paths in config work
os.chdir(project_root)

# Add current directory to path for local imports
sys.path.insert(0, str(current_dir))

import argparse
import pandas as pd

from utils import load_config, set_seed, ensure_dir, save_metrics_csv, append_metrics_csv
from data_loader import load_posts, validate_data
from dataset_prep import (
    prepare_full_dataset, prepare_early_slice_dataset, 
    prepare_stage_dataset, save_processed_features
)
from train import train_model, print_top_features, save_model_checkpoint, get_model_coefficients
from evaluate import evaluate_model, evaluate_calibration
from explain import explain_with_shap, plot_feature_importance_bar
from plots import plot_early_slice_results
from stats import compare_groups, compare_multiple_features


def run_full_experiment(config, model_name='lr'):
    """
    Run full experiment with all posts per user.
    
    Args:
        config: Configuration dictionary
        model_name: Model type ('lr', 'svm', 'lgbm')
    """
    print("\n" + "="*60)
    print("FULL EXPERIMENT - All Posts Per User")
    print("="*60)
    
    # Load data
    posts, _ = load_posts(config['dataset']['posts_csv'], config['dataset']['users_csv'])
    validate_data(posts)
    
    # Save processed features for later plotting
    posts_with_features = save_processed_features(posts)
    
    # Prepare dataset
    data = prepare_full_dataset(posts, config, use_embeddings=True, use_features=True)
    
    # Train model
    model = train_model(
        data['train_X'], data['train_y'], 
        model_name=model_name,
        seed=config['training']['seed']
    )
    
    # Print top features
    print_top_features(model, data['feature_names'], top_k=15)
    
    # Evaluate on all splits
    results = {}
    for split in ['train', 'val', 'test']:
        metrics = evaluate_model(model, data[f'{split}_X'], data[f'{split}_y'], split)
        results[split] = metrics
    
    # Save results
    results_dir = config['training']['results_dir']
    ensure_dir(results_dir)
    
    save_metrics_csv(results, f"{results_dir}/val_metrics.csv")
    
    # SHAP explanations
    try:
        shap_path = f"{config['training']['reports_dir']}/shap_summary.png"
        explain_with_shap(model, data['val_X'], data['feature_names'], save_path=shap_path)
    except Exception as e:
        print(f"Warning: SHAP analysis failed: {e}")
    
    # Feature importance plot
    if model_name in ['lr', 'svm']:
        coef_dict = get_model_coefficients(model, data['feature_names'])
        if coef_dict:
            imp_path = f"{config['training']['reports_dir']}/feature_importance.png"
            plot_feature_importance_bar(coef_dict, top_k=20, save_path=imp_path)
    
    # Save model
    model_path = f"models/checkpoints/{model_name}_full.pkl"
    save_model_checkpoint(model, model_path)
    
    print("\n" + "="*60)
    print("Full experiment complete!")
    print("="*60)


def run_early_slice_experiment(config, model_name='lr'):
    """
    Run early-slice experiments at different k values.
    
    Args:
        config: Configuration dictionary
        model_name: Model type
    """
    print("\n" + "="*60)
    print("EARLY-SLICE EXPERIMENT")
    print("="*60)
    
    # Load data
    posts, _ = load_posts(config['dataset']['posts_csv'], config['dataset']['users_csv'])
    validate_data(posts)
    
    # K values to test
    k_values = config['training']['early_slice_ks']
    print(f"Testing k values: {k_values}")
    
    results_list = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Training with k={k} posts per user")
        print("="*60)
        
        # Prepare dataset
        data = prepare_early_slice_dataset(posts, k, config, use_embeddings=True, use_features=True)
        
        # Train model
        model = train_model(
            data['train_X'], data['train_y'],
            model_name=model_name,
            seed=config['training']['seed']
        )
        
        # Evaluate
        results = {'k': k}
        for split in ['train', 'val', 'test']:
            metrics = evaluate_model(model, data[f'{split}_X'], data[f'{split}_y'], split)
            for metric_name, metric_val in metrics.items():
                if isinstance(metric_val, (int, float)):
                    results[f'{split}_{metric_name}'] = metric_val
        
        results_list.append(results)
        
        # Save model checkpoint
        model_path = f"models/checkpoints/{model_name}_k{k}.pkl"
        save_model_checkpoint(model, model_path)
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_dir = config['training']['results_dir']
    ensure_dir(results_dir)
    
    results_path = f"{results_dir}/early_slice.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved early-slice results to {results_path}")
    
    # Plot results
    plot_path = f"{config['training']['reports_dir']}/early_slice_f1.png"
    ensure_dir(config['training']['reports_dir'])
    plot_early_slice_results(results_df, metric='f1', save_path=plot_path)
    
    # Print summary
    print("\n" + "="*60)
    print("Early-Slice Results Summary:")
    print("="*60)
    print(results_df[['k', 'val_f1', 'test_f1']].to_string(index=False))
    
    print("\n" + "="*60)
    print("Early-slice experiment complete!")
    print("="*60)


def run_stage_comparison(config, stage_csv, model_name='lr'):
    """
    Compare models trained on different temporal stages.
    
    Args:
        config: Configuration dictionary
        stage_csv: Path to CSV with temporal_stage annotations
        model_name: Model type
    """
    print("\n" + "="*60)
    print("TEMPORAL STAGE COMPARISON")
    print("="*60)
    
    # Load data
    posts, _ = load_posts(config['dataset']['posts_csv'], stage_csv=stage_csv)
    validate_data(posts)
    
    if 'temporal_stage' not in posts.columns:
        raise ValueError("Stage CSV must have 'temporal_stage' column")
    
    # Stage pairs to compare
    stage_pairs = config['training']['stage_pairs']
    
    results_text = []
    results_text.append("="*60)
    results_text.append("TEMPORAL STAGE COMPARISON RESULTS")
    results_text.append("="*60)
    
    for stage1, stage2 in stage_pairs:
        print(f"\n{'='*60}")
        print(f"Comparing: {stage1} vs. {stage2}")
        print("="*60)
        
        results_text.append(f"\n\n{stage1.upper()} vs. {stage2.upper()}")
        results_text.append("-"*60)
        
        # Prepare datasets for each stage
        try:
            data1 = prepare_stage_dataset(posts, stage1, config, use_embeddings=True, use_features=True)
            data2 = prepare_stage_dataset(posts, stage2, config, use_embeddings=True, use_features=True)
        except Exception as e:
            print(f"Error preparing stage data: {e}")
            continue
        
        # Train models
        model1 = train_model(data1['train_X'], data1['train_y'], model_name=model_name, seed=config['training']['seed'])
        model2 = train_model(data2['train_X'], data2['train_y'], model_name=model_name, seed=config['training']['seed'])
        
        # Evaluate
        metrics1 = evaluate_model(model1, data1['test_X'], data1['test_y'], stage1)
        metrics2 = evaluate_model(model2, data2['test_X'], data2['test_y'], stage2)
        
        results_text.append(f"\n{stage1} performance:")
        results_text.append(f"  F1: {metrics1['f1']:.4f}, Accuracy: {metrics1['accuracy']:.4f}")
        results_text.append(f"\n{stage2} performance:")
        results_text.append(f"  F1: {metrics2['f1']:.4f}, Accuracy: {metrics2['accuracy']:.4f}")
        
        # Statistical comparison of features
        print(f"\nComparing features between {stage1} and {stage2}...")
        stage1_posts = posts[posts['temporal_stage'] == stage1]
        stage2_posts = posts[posts['temporal_stage'] == stage2]
        
        # Compare key features
        key_features = ['vader_compound', 'fp_pronoun_rate', 'char_count', 'textblob_polarity']
        
        # Need to load posts with features
        if 'vader_compound' not in posts.columns:
            print("  (Feature comparison requires processed features - run full experiment first)")
        else:
            feature_comparison = compare_multiple_features(
                posts,
                key_features,
                group_col='temporal_stage',
                group1_val=stage1,
                group2_val=stage2,
                group1_name=stage1,
                group2_name=stage2
            )
            
            results_text.append(f"\nFeature comparison ({stage1} vs {stage2}):")
            for feat in key_features:
                if feat in feature_comparison:
                    comp = feature_comparison[feat]
                    results_text.append(f"  {feat}:")
                    results_text.append(f"    {stage1} mean: {comp['welch_t_test']['mean1']:.4f}")
                    results_text.append(f"    {stage2} mean: {comp['welch_t_test']['mean2']:.4f}")
                    results_text.append(f"    p-value: {comp['welch_t_test']['pvalue']:.4f}")
                    results_text.append(f"    Cohen's d: {comp['cohens_d']:.4f}")
    
    # Save results
    results_dir = config['training']['results_dir']
    ensure_dir(results_dir)
    
    results_path = f"{results_dir}/stage_compare.txt"
    with open(results_path, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"\nSaved stage comparison to {results_path}")
    
    print("\n" + "="*60)
    print("Stage comparison complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run NLP experiments')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['full', 'early_slice', 'stage_compare'],
                        help='Experiment mode')
    parser.add_argument('--model', type=str, default='lr',
                        choices=['lr', 'svm', 'lgbm'],
                        help='Model type')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--stage_csv', type=str, default=None,
                        help='Path to staged posts CSV (for stage_compare mode)')
    
    args = parser.parse_args()
    
    # Handle config path - if relative path doesn't exist, try from project root
    config_path = args.config
    if not Path(config_path).exists():
        # Try from project root (we already changed to project root earlier)
        alt_path = Path.cwd() / args.config
        if alt_path.exists():
            config_path = str(alt_path)
        else:
            # Try without the ../ prefix
            config_path = args.config.replace('../', '')
    
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Create output directories
    ensure_dir(config['training']['results_dir'])
    ensure_dir(config['training']['reports_dir'])
    
    # Run appropriate mode
    if args.mode == 'full':
        run_full_experiment(config, args.model)
    
    elif args.mode == 'early_slice':
        run_early_slice_experiment(config, args.model)
    
    elif args.mode == 'stage_compare':
        if args.stage_csv is None:
            raise ValueError("--stage_csv required for stage_compare mode")
        run_stage_comparison(config, args.stage_csv, args.model)
    
    print("\n✓ All experiments complete!")


if __name__ == '__main__':
    main()