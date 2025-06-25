#!/usr/bin/env python
"""
Script to run significance testing between baseline and extended models.
Uses Welch's t-test for ranks.
"""

import os
import numpy as np
import argparse
from scipy import stats
from saved_ranks_evaluator import get_triple_ranks
import matplotlib
# Set non-interactive backend for headless environments like HPC clusters
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

# Add numpy_to_python function for JSON serialization
def numpy_to_python(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Add wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Results will not be logged to Weights & Biases.")

# Default wandb configuration
WANDB_PROJECT = "recommender-server"
WANDB_ENTITY = "slaw-mecz-vrije-universiteit-amsterdam"

def run_significance_test(baseline_dir, extended_dir, output_file=None, use_wandb=False, run_name=None):
    """
    Run significance test between baseline and extended models.
    
    Args:
        baseline_dir: Directory with baseline model
        extended_dir: Directory with extended model
        output_file: File to write results to
        use_wandb: Whether to log results to wandb
        run_name: Name for the wandb run
    """
    print("\n=== Running Significance Testing ===")
    
    # Initialize wandb if requested
    if use_wandb and WANDB_AVAILABLE:
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"significance_test_{timestamp}"
            
        print(f"Initializing wandb run: {run_name}")
        try:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_name,
                job_type="significance_test",
                config={
                    "baseline_dir": baseline_dir,
                    "extended_dir": extended_dir,
                    "timestamp": timestamp
                }
            )
            wandb_initialized = True
        except Exception as e:
            print(f"Failed to initialize wandb: {str(e)}")
            print("Continuing without wandb logging")
            wandb_initialized = False
    else:
        wandb_initialized = False
    
    # Get ranks for baseline and extended models
    print(f"\nExtracting ranks from baseline model at {baseline_dir}")
    baseline_ranks = get_triple_ranks(baseline_dir)
    
    print(f"\nExtracting ranks from extended model at {extended_dir}")
    extended_ranks = get_triple_ranks(extended_dir)
    
    # Convert ranks to MRR (reciprocal ranks)
    baseline_mrr = 1.0 / baseline_ranks
    extended_mrr = 1.0 / extended_ranks
    
    # Run Welch's t-test for MRR
    t_stat_mrr, p_value_mrr = stats.ttest_ind(
        baseline_mrr, 
        extended_mrr, 
        equal_var=False  # Welch's t-test assumes unequal variances
    )
    
    # Calculate mean rank
    mean_baseline_rank = np.mean(baseline_ranks)
    mean_extended_rank = np.mean(extended_ranks)
    rank_improvement = (mean_baseline_rank - mean_extended_rank) / mean_baseline_rank * 100
    
    # Run Welch's t-test for ranks (lower is better)
    t_stat_rank, p_value_rank = stats.ttest_ind(
        baseline_ranks, 
        extended_ranks, 
        equal_var=False
    )
    
    # Calculate hits@k metrics
    k_values = [1, 3, 5, 10]
    baseline_hits = {}
    extended_hits = {}
    hits_improvement = {}
    hits_tstat = {}
    hits_pvalue = {}
    
    for k in k_values:
        # Calculate hits@k (binary: 1 if rank <= k, 0 otherwise)
        baseline_hits[k] = (baseline_ranks <= k).astype(float)
        extended_hits[k] = (extended_ranks <= k).astype(float)
        
        # Calculate mean hits@k
        mean_baseline_hits = np.mean(baseline_hits[k])
        mean_extended_hits = np.mean(extended_hits[k])
        
        # Calculate improvement (higher is better)
        hits_improvement[k] = (mean_extended_hits - mean_baseline_hits) / mean_baseline_hits * 100 if mean_baseline_hits > 0 else 0
        
        # Run Welch's t-test for hits@k
        hits_tstat[k], hits_pvalue[k] = stats.ttest_ind(
            baseline_hits[k],
            extended_hits[k],
            equal_var=False
        )
    
    # Prepare MRR results
    mean_baseline_mrr = np.mean(baseline_mrr)
    mean_extended_mrr = np.mean(extended_mrr)
    mrr_improvement = (mean_extended_mrr - mean_baseline_mrr) / mean_baseline_mrr * 100
    
    # Create a dictionary of all results for easy logging
    all_results = {
        "baseline_mrr": mean_baseline_mrr,
        "extended_mrr": mean_extended_mrr,
        "combined_mrr": mean_extended_mrr,  # Add combined_ version for consistency
        "mrr_improvement": mrr_improvement,
        "mrr_t_statistic": t_stat_mrr,
        "mrr_p_value": p_value_mrr,
        "mrr_significant": p_value_mrr < 0.05,
        
        "baseline_mean_rank": mean_baseline_rank,
        "extended_mean_rank": mean_extended_rank,
        "combined_mean_rank": mean_extended_rank,  # Add combined_ version for consistency
        "mean_rank_improvement": rank_improvement,
        "mean_rank_t_statistic": t_stat_rank,
        "mean_rank_p_value": p_value_rank,
        "mean_rank_significant": p_value_rank < 0.05,
    }
    
    # Add hits@k results to the dictionary
    for k in k_values:
        mean_baseline_hits_k = np.mean(baseline_hits[k])
        mean_extended_hits_k = np.mean(extended_hits[k])
        all_results.update({
            f"baseline_hits@{k}": mean_baseline_hits_k,
            f"extended_hits@{k}": mean_extended_hits_k,
            f"combined_hits@{k}": mean_extended_hits_k,  # Add combined_ version for consistency
            f"hits@{k}_improvement": hits_improvement[k],
            f"hits@{k}_t_statistic": hits_tstat[k],
            f"hits@{k}_p_value": hits_pvalue[k],
            f"hits@{k}_significant": hits_pvalue[k] < 0.05
        })
    
    # Determine overall significance
    if p_value_mrr < 0.05 or p_value_rank < 0.05 or any(p < 0.05 for p in hits_pvalue.values()):
        overall_significance = "At least one metric shows statistically significant differences (p < 0.05)."
    else:
        overall_significance = "No metrics show statistically significant differences (p >= 0.05)."
    
    all_results["overall_significance"] = overall_significance
    all_results["num_triples_evaluated"] = len(baseline_ranks)
    
    # Print results in a formatted way
    print("\n=== Significance Test Results ===")
    
    # MRR results
    print("\nMean Reciprocal Rank (MRR):")
    print(f"  Baseline: {mean_baseline_mrr:.6f}")
    print(f"  Extended: {mean_extended_mrr:.6f}")
    print(f"  Improvement: {mrr_improvement:.2f}%")
    print(f"  T-statistic: {t_stat_mrr:.6f}")
    print(f"  P-value: {p_value_mrr:.6f}")
    print(f"  Significance: {'Significant (p < 0.05)' if p_value_mrr < 0.05 else 'Not significant (p >= 0.05)'}")
    
    # Mean Rank results
    print("\nMean Rank:")
    print(f"  Baseline: {mean_baseline_rank:.2f}")
    print(f"  Extended: {mean_extended_rank:.2f}")
    print(f"  Improvement: {rank_improvement:.2f}% ({'better' if rank_improvement > 0 else 'worse'})")
    print(f"  T-statistic: {t_stat_rank:.6f}")
    print(f"  P-value: {p_value_rank:.6f}")
    print(f"  Significance: {'Significant (p < 0.05)' if p_value_rank < 0.05 else 'Not significant (p >= 0.05)'}")
    
    # Hits@k results
    print("\nHits@k:")
    for k in k_values:
        mean_baseline_hits_k = np.mean(baseline_hits[k])
        mean_extended_hits_k = np.mean(extended_hits[k])
        print(f"\n  Hits@{k}:")
        print(f"    Baseline: {mean_baseline_hits_k:.6f}")
        print(f"    Extended: {mean_extended_hits_k:.6f}")
        print(f"    Improvement: {hits_improvement[k]:.2f}%")
        print(f"    T-statistic: {hits_tstat[k]:.6f}")
        print(f"    P-value: {hits_pvalue[k]:.6f}")
        print(f"    Significance: {'Significant (p < 0.05)' if hits_pvalue[k] < 0.05 else 'Not significant (p >= 0.05)'}")
    
    print(f"\nOverall: {overall_significance}")
    print(f"\nTotal triples evaluated: {len(baseline_ranks)}")
    
    # Create visualizations for the results
    create_visualizations(baseline_ranks, extended_ranks, k_values, all_results)
    
    # Log results to wandb if initialized
    if wandb_initialized:
        # Log all metrics
        wandb.log(all_results)
        
        # Log histograms of ranks
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Limit ranks to 100 for better visualization in histogram
        max_rank_for_hist = 100
        hist_baseline = np.clip(baseline_ranks, 1, max_rank_for_hist)
        hist_extended = np.clip(extended_ranks, 1, max_rank_for_hist)
        
        ax1.hist(hist_baseline, bins=50, alpha=0.7, label=f'Baseline (Mean: {mean_baseline_rank:.1f})')
        ax1.hist(hist_extended, bins=50, alpha=0.7, label=f'Extended (Mean: {mean_extended_rank:.1f})')
        ax1.set_xlabel('Rank (clipped to 100)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Ranks')
        ax1.legend()
        
        # Create a bar chart for hits@k
        ks = list(k_values)
        baseline_values = [np.mean(baseline_hits[k]) for k in k_values]
        extended_values = [np.mean(extended_hits[k]) for k in k_values]
        
        x = np.arange(len(ks))
        width = 0.35
        
        ax2.bar(x - width/2, baseline_values, width, label='Baseline')
        ax2.bar(x + width/2, extended_values, width, label='Extended')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Hits@k')
        ax2.set_title('Hits@k Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ks)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('rank_distributions.png')
        
        # Log the plot to wandb
        wandb.log({"rank_distributions": wandb.Image('rank_distributions.png')})
        
        # Create a more comprehensive summary as a wandb Table
        data = []
        for metric in ["MRR", "Mean Rank"] + [f"Hits@{k}" for k in k_values]:
            if metric == "MRR":
                baseline_val = mean_baseline_mrr
                extended_val = mean_extended_mrr
                improvement_val = mrr_improvement
                p_value = p_value_mrr
                t_stat = t_stat_mrr
            elif metric == "Mean Rank":
                baseline_val = mean_baseline_rank
                extended_val = mean_extended_rank
                improvement_val = rank_improvement
                p_value = p_value_rank
                t_stat = t_stat_rank
            else:
                k = int(metric.split('@')[1])
                baseline_val = np.mean(baseline_hits[k])
                extended_val = np.mean(extended_hits[k])
                improvement_val = hits_improvement[k]
                p_value = hits_pvalue[k]
                t_stat = hits_tstat[k]
            
            data.append([
                metric,
                round(baseline_val, 6),
                round(extended_val, 6),
                f"{improvement_val:.2f}%",
                round(t_stat, 6),
                round(p_value, 6),
                "Yes" if p_value < 0.05 else "No"
            ])
        
        summary_table = wandb.Table(
            columns=["Metric", "Baseline", "Extended", "Improvement", "T-statistic", "P-value", "Significant"],
            data=data
        )
        wandb.log({"summary_table": summary_table})
        
        # Finish the wandb run
        wandb.finish()
    
    # Write results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("=== Significance Test Results ===\n\n")
            
            # MRR results
            f.write("Mean Reciprocal Rank (MRR):\n")
            f.write(f"  Baseline: {mean_baseline_mrr:.6f}\n")
            f.write(f"  Extended: {mean_extended_mrr:.6f}\n")
            f.write(f"  Improvement: {mrr_improvement:.2f}%\n")
            f.write(f"  T-statistic: {t_stat_mrr:.6f}\n")
            f.write(f"  P-value: {p_value_mrr:.6f}\n")
            f.write(f"  Significance: {'Significant (p < 0.05)' if p_value_mrr < 0.05 else 'Not significant (p >= 0.05)'}\n\n")
            
            # Mean Rank results
            f.write("Mean Rank:\n")
            f.write(f"  Baseline: {mean_baseline_rank:.2f}\n")
            f.write(f"  Extended: {mean_extended_rank:.2f}\n")
            f.write(f"  Improvement: {rank_improvement:.2f}% ({'better' if rank_improvement > 0 else 'worse'})\n")
            f.write(f"  T-statistic: {t_stat_rank:.6f}\n")
            f.write(f"  P-value: {p_value_rank:.6f}\n")
            f.write(f"  Significance: {'Significant (p < 0.05)' if p_value_rank < 0.05 else 'Not significant (p >= 0.05)'}\n\n")
            
            # Hits@k results
            f.write("Hits@k:\n")
            for k in k_values:
                mean_baseline_hits_k = np.mean(baseline_hits[k])
                mean_extended_hits_k = np.mean(extended_hits[k])
                f.write(f"\n  Hits@{k}:\n")
                f.write(f"    Baseline: {mean_baseline_hits_k:.6f}\n")
                f.write(f"    Extended: {mean_extended_hits_k:.6f}\n") 
                f.write(f"    Improvement: {hits_improvement[k]:.2f}%\n")
                f.write(f"    T-statistic: {hits_tstat[k]:.6f}\n")
                f.write(f"    P-value: {hits_pvalue[k]:.6f}\n")
                f.write(f"    Significance: {'Significant (p < 0.05)' if hits_pvalue[k] < 0.05 else 'Not significant (p >= 0.05)'}\n")
            
            f.write(f"\nOverall: {overall_significance}\n")
            
            f.write(f"\nTotal triples evaluated: {len(baseline_ranks)}\n")
        
        print(f"Results saved to {output_file}")
        
        # Also save results as JSON for easier programmatic access
        json_file = output_file.replace('.txt', '.json') if output_file.endswith('.txt') else f"{output_file}.json"
        with open(json_file, 'w') as f:
            # Convert all numpy types to Python types first
            serializable_results = {k: numpy_to_python(v) for k, v in all_results.items()}
            json.dump(serializable_results, f, indent=2)
        print(f"Results also saved in JSON format to {json_file}")
    
    return all_results

def create_visualizations(baseline_ranks, extended_ranks, k_values, all_results):
    """Create visualizations of the significance test results."""
    # Create a folder for visualizations if it doesn't exist
    viz_folder = "significance_viz"
    os.makedirs(viz_folder, exist_ok=True)
    
    # Plot 1: Distribution of ranks (clipped to 100 for better visualization)
    plt.figure(figsize=(10, 6))
    max_rank_for_hist = 100
    hist_baseline = np.clip(baseline_ranks, 1, max_rank_for_hist)
    hist_extended = np.clip(extended_ranks, 1, max_rank_for_hist)
    
    plt.hist(hist_baseline, bins=50, alpha=0.6, label=f'Baseline (Mean: {np.mean(baseline_ranks):.1f})')
    plt.hist(hist_extended, bins=50, alpha=0.6, label=f'Extended (Mean: {np.mean(extended_ranks):.1f})')
    plt.xlabel('Rank (clipped to 100)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ranks')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(viz_folder, 'rank_distribution.png'))
    
    # Plot 2: Hits@k comparison
    plt.figure(figsize=(10, 6))
    ks = list(k_values)
    baseline_values = [all_results[f"baseline_hits@{k}"] for k in k_values]
    extended_values = [all_results[f"extended_hits@{k}"] for k in k_values]
    
    x = np.arange(len(ks))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline')
    plt.bar(x + width/2, extended_values, width, label='Extended')
    plt.xlabel('k')
    plt.ylabel('Hits@k')
    plt.title('Hits@k Comparison')
    plt.xticks(x, ks)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(viz_folder, 'hits_at_k.png'))
    
    # Plot 3: Improvements across all metrics
    plt.figure(figsize=(12, 6))
    metrics = ['MRR', 'Mean Rank'] + [f'Hits@{k}' for k in k_values]
    improvements = [
        all_results['mrr_improvement'],
        all_results['mean_rank_improvement']
    ] + [all_results[f'hits@{k}_improvement'] for k in k_values]
    
    colors = ['green' if val > 0 else 'red' for val in improvements]
    
    plt.bar(metrics, improvements, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Metric')
    plt.ylabel('Improvement (%)')
    plt.title('Percentage Improvement in Each Metric')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, 'metric_improvements.png'))
    
    print(f"Visualizations saved to {viz_folder} directory")

def main():
    parser = argparse.ArgumentParser(description="Run significance testing between baseline and extended models")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline",
                        help="Directory with baseline model (default: models/baseline)")
    parser.add_argument("--extended-dir", type=str, default="models/extended",
                        help="Directory with extended model (default: models/extended)")
    parser.add_argument("--output-file", type=str, default="results.txt",
                        help="File to write results to (default: results.txt)")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--run-name", type=str, 
                        help="Name for the wandb run (default: auto-generated)")
    
    args = parser.parse_args()
    
    run_significance_test(
        baseline_dir=args.baseline_dir,
        extended_dir=args.extended_dir,
        output_file=args.output_file,
        use_wandb=args.use_wandb,
        run_name=args.run_name
    )

if __name__ == "__main__":
    main() 