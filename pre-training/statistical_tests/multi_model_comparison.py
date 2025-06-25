#!/usr/bin/env python
"""
Script to create comparison plots between multiple ComplEx models.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def parse_metrics_from_file(metrics_file):
    """Parse metrics from the metrics.txt file."""
    with open(metrics_file, 'r') as f:
        content = f.read()
    
    # Extract the metrics section
    lines = content.split('\n')
    
    # Find the line containing the metrics data
    metrics_data = None
    for line in lines:
        if line.startswith('both:'):
            # Extract the dictionary part
            dict_str = line[5:].strip()  # Remove 'both:'
            try:
                # Use eval carefully - this is a known good format
                metrics_data = eval(dict_str)
                break
            except:
                continue
    
    if metrics_data is None:
        raise ValueError(f"Could not parse metrics from {metrics_file}")
    
    # Extract realistic metrics (middle ground between optimistic and pessimistic)
    realistic_metrics = metrics_data.get('realistic', {})
    
    return {
        'hits@1': realistic_metrics.get('hits_at_1', 0),
        'hits@3': realistic_metrics.get('hits_at_3', 0),
        'hits@5': realistic_metrics.get('hits_at_5', 0),
        'hits@10': realistic_metrics.get('hits_at_10', 0),
        'mean_rank': realistic_metrics.get('arithmetic_mean_rank', 0),
        'mrr': realistic_metrics.get('inverse_harmonic_mean_rank', 0)
    }

def extract_model_info(metrics_file):
    """Extract model information from metrics file."""
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    info = {}
    for line in lines:
        if line.startswith('Model:'):
            info['model'] = line.split(':', 1)[1].strip()
        elif line.startswith('Dataset:'):
            info['dataset'] = line.split(':', 1)[1].strip()
        elif line.startswith('Embedding Dim:'):
            info['embedding_dim'] = line.split(':', 1)[1].strip()
        elif line.startswith('Probability Threshold:'):
            info['probability_threshold'] = line.split(':', 1)[1].strip()
    
    return info

def create_multi_model_comparison():
    """Create comparison plots between multiple models."""
    
    # Define model configurations
    models = {
        'Baseline': {
            'dir': Path("models/evaluation_pre_training/complex_baseline"),
            'color': '#2E86C1',
            'short_name': 'Baseline'
        },
        'Outgoing Extended': {
            'dir': Path("models/evaluation_pre_training/complex_extended_outgoing_prob25_19epoch"),
            'color': '#28B463',
            'short_name': 'Outgoing'
        },
        'Bidirectional Extended': {
            'dir': Path("models/evaluation_pre_training/complex_extended_bidirectional_prob25_19epoch"),
            'color': '#E74C3C',
            'short_name': 'Bidirectional'
        },
        'Bidirectional Typed': {
            'dir': Path("models/evaluation_pre_training/complex_extended_bidirectional_typed_prob25_19epoch"),
            'color': '#8E44AD',
            'short_name': 'Typed'
        }
    }
    
    # Parse metrics for all models
    model_metrics = {}
    model_info = {}
    
    for name, config in models.items():
        metrics_file = config['dir'] / "metrics.txt"
        
        if not metrics_file.exists():
            print(f"Metrics file not found for {name}: {metrics_file}")
            continue
            
        print(f"Parsing metrics for {name}...")
        model_metrics[name] = parse_metrics_from_file(metrics_file)
        model_info[name] = extract_model_info(metrics_file)
        
        # Print metrics for verification
        print(f"{name} Metrics:")
        for metric, value in model_metrics[name].items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    # Create output directory
    output_dir = Path("multi_model_comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plots
    create_hits_multi_comparison(model_metrics, models, output_dir)
    create_comprehensive_multi_comparison(model_metrics, model_info, models, output_dir)
    create_improvement_multi_comparison(model_metrics, models, output_dir)
    create_summary_table(model_metrics, model_info, output_dir)

def create_hits_multi_comparison(model_metrics, models, output_dir):
    """Create a comparison plot for Hits@k metrics across all models."""
    
    k_values = [1, 3, 5, 10]
    n_models = len(model_metrics)
    x = np.arange(len(k_values))
    width = 0.18  # Width of bars
    
    plt.figure(figsize=(14, 8))
    
    # Plot bars for each model
    for i, (name, metrics) in enumerate(model_metrics.items()):
        hits_values = [metrics[f'hits@{k}'] for k in k_values]
        color = models[name]['color']
        offset = (i - n_models/2 + 0.5) * width
        
        bars = plt.bar(x + offset, hits_values, width, label=name, 
                      color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    plt.xlabel('k', fontsize=12, fontweight='bold')
    plt.ylabel('Hits@k', fontsize=12, fontweight='bold')
    plt.title('Hits@k Comparison Across ComplEx Model Variants', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, [f'Hits@{k}' for k in k_values])
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Set y-axis limits to better show differences
    all_hits = [v for metrics in model_metrics.values() for k in k_values for v in [metrics[f'hits@{k}']]]
    max_val = max(all_hits)
    plt.ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hits_multi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-model Hits@k comparison plot to {output_dir / 'hits_multi_comparison.png'}")

def create_comprehensive_multi_comparison(model_metrics, model_info, models, output_dir):
    """Create a comprehensive comparison plot for all metrics across all models."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    model_names = list(model_metrics.keys())
    n_models = len(model_names)
    
    # Plot 1: Hits@k comparison
    k_values = [1, 3, 5, 10]
    x = np.arange(len(k_values))
    width = 0.18
    
    for i, name in enumerate(model_names):
        metrics = model_metrics[name]
        hits_values = [metrics[f'hits@{k}'] for k in k_values]
        color = models[name]['color']
        offset = (i - n_models/2 + 0.5) * width
        
        ax1.bar(x + offset, hits_values, width, label=name, color=color, alpha=0.8)
    
    ax1.set_xlabel('k')
    ax1.set_ylabel('Hits@k')
    ax1.set_title('Hits@k Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{k}' for k in k_values])
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: MRR comparison
    x_mrr = np.arange(1)
    width_mrr = 0.15
    
    for i, name in enumerate(model_names):
        metrics = model_metrics[name]
        mrr_value = [metrics['mrr']]
        color = models[name]['color']
        offset = (i - n_models/2 + 0.5) * width_mrr
        
        bars = ax2.bar(x_mrr + offset, mrr_value, width_mrr, label=name, color=color, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('MRR')
    ax2.set_title('Mean Reciprocal Rank Comparison')
    ax2.set_xticks(x_mrr)
    ax2.set_xticklabels(['MRR'])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Mean Rank comparison (lower is better)
    x_rank = np.arange(1)
    
    for i, name in enumerate(model_names):
        metrics = model_metrics[name]
        rank_value = [metrics['mean_rank']]
        color = models[name]['color']
        offset = (i - n_models/2 + 0.5) * width_mrr
        
        bars = ax3.bar(x_rank + offset, rank_value, width_mrr, label=name, color=color, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Mean Rank (lower is better)')
    ax3.set_title('Mean Rank Comparison')
    ax3.set_xticks(x_rank)
    ax3.set_xticklabels(['Mean Rank'])
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary information
    ax4.axis('off')
    
    # Create summary text for each model
    summary_text = "Model Summary:\n\n"
    for name, info in model_info.items():
        metrics = model_metrics[name]
        
        dataset_info = info.get('dataset', 'N/A')
        # Extract number of triples if available
        triples_count = "0"
        if '+' in dataset_info and 'artificial' in dataset_info:
            parts = dataset_info.split('+')
            if len(parts) > 1:
                triples_part = parts[1].strip()
                triples_count = triples_part.split()[0]
        
        summary_text += f"{name}:\n"
        summary_text += f"  • Dataset: {dataset_info}\n"
        summary_text += f"  • Artificial Triples: {triples_count}\n"
        summary_text += f"  • Hits@1: {metrics['hits@1']:.4f}\n"
        summary_text += f"  • Hits@10: {metrics['hits@10']:.4f}\n"
        summary_text += f"  • MRR: {metrics['mrr']:.4f}\n"
        summary_text += f"  • Mean Rank: {metrics['mean_rank']:.1f}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('ComplEx Model Variants: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_multi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive multi-model comparison plot to {output_dir / 'comprehensive_multi_comparison.png'}")

def create_improvement_multi_comparison(model_metrics, models, output_dir):
    """Create a plot showing percentage improvements relative to baseline."""
    
    # Use the first model as baseline (assuming it's the baseline)
    baseline_name = list(model_metrics.keys())[0]
    baseline_metrics = model_metrics[baseline_name]
    
    # Calculate improvements for each model relative to baseline
    improvements = {}
    for name, metrics in model_metrics.items():
        if name == baseline_name:
            continue  # Skip baseline itself
            
        model_improvements = {}
        
        # Calculate Hits@k improvements
        for k in [1, 3, 5, 10]:
            baseline_val = baseline_metrics[f'hits@{k}']
            current_val = metrics[f'hits@{k}']
            model_improvements[f'Hits@{k}'] = ((current_val - baseline_val) / baseline_val) * 100
        
        # MRR improvement
        baseline_mrr = baseline_metrics['mrr']
        current_mrr = metrics['mrr']
        model_improvements['MRR'] = ((current_mrr - baseline_mrr) / baseline_mrr) * 100
        
        # Mean Rank improvement (negative change is better for rank)
        baseline_rank = baseline_metrics['mean_rank']
        current_rank = metrics['mean_rank']
        model_improvements['Mean Rank'] = ((baseline_rank - current_rank) / baseline_rank) * 100
        
        improvements[name] = model_improvements
    
    # Create the plot
    metrics_list = ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'MRR', 'Mean Rank']
    n_metrics = len(metrics_list)
    n_models = len(improvements)
    
    x = np.arange(n_metrics)
    width = 0.25
    
    plt.figure(figsize=(16, 8))
    
    for i, (name, model_impr) in enumerate(improvements.items()):
        values = [model_impr[metric] for metric in metrics_list]
        color = models[name]['color']
        offset = (i - n_models/2 + 0.5) * width
        
        bars = plt.bar(x + offset, values, width, label=name, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.5 if height > 0 else -1.5),
                    f'{value:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance: Percentage Improvement over {baseline_name}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_list, rotation=45)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_multi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved improvement multi-model comparison plot to {output_dir / 'improvement_multi_comparison.png'}")

def create_summary_table(model_metrics, model_info, output_dir):
    """Create a summary table with all metrics."""
    
    # Create a text-based summary table
    with open(output_dir / 'metrics_summary.txt', 'w') as f:
        f.write("ComplEx Model Variants - Performance Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Header
        f.write(f"{'Model':<25} {'Hits@1':<8} {'Hits@3':<8} {'Hits@5':<8} {'Hits@10':<8} {'MRR':<8} {'Mean Rank':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Data rows
        for name, metrics in model_metrics.items():
            f.write(f"{name:<25} {metrics['hits@1']:<8.4f} {metrics['hits@3']:<8.4f} "
                   f"{metrics['hits@5']:<8.4f} {metrics['hits@10']:<8.4f} "
                   f"{metrics['mrr']:<8.4f} {metrics['mean_rank']:<10.1f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        
        # Model details
        f.write("\nModel Details:\n")
        for name, info in model_info.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Dataset: {info.get('dataset', 'N/A')}\n")
            f.write(f"  Embedding Dim: {info.get('embedding_dim', 'N/A')}\n")
            f.write(f"  Probability Threshold: {info.get('probability_threshold', 'N/A')}\n")
    
    print(f"Saved metrics summary table to {output_dir / 'metrics_summary.txt'}")

if __name__ == "__main__":
    print("Creating multi-model comparison plots...")
    create_multi_model_comparison()
    print("Done!") 