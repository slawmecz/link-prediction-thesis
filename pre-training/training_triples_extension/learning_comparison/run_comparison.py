#!/usr/bin/env python
"""
Runner script to train both baseline and bidirectional ComplEx models and generate comparison plots.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our training functions
from complex_baseline_with_callbacks import train_baseline_model_with_callbacks
from complex_bidirectional_with_callbacks import train_bidirectional_model_with_callbacks
from plot_training_metrics import create_training_comparison_plots

def run_training_comparison(
    baseline_output_dir="models/baseline_complex_with_callbacks",
    bidirectional_output_dir="models/bidirectional_complex_with_callbacks",
    plots_output_dir="metrics_comparison_plots",
    max_epochs=19,  # Reduced for research purposes
    max_entities=1000000,  # Limit entities for faster execution
    probability_threshold=0.5
):
    """Run complete training comparison pipeline."""
    
    print("=" * 60)
    print("ComplEx Training Comparison Pipeline")
    print("=" * 60)
    print(f"Baseline output: {baseline_output_dir}")
    print(f"Bidirectional output: {bidirectional_output_dir}")
    print(f"Plots output: {plots_output_dir}")
    print(f"Max epochs: {max_epochs}")
    print(f"Max entities for bidirectional: {max_entities}")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Step 1: Train baseline model
    print("\n STEP 1: Training Baseline ComplEx Model")
    print("-" * 50)
    
    baseline_start_time = time.time()
    try:
        baseline_model, baseline_dir, baseline_logs = train_baseline_model_with_callbacks(
            output_dir=baseline_output_dir,
            max_epochs=max_epochs
        )
        baseline_duration = time.time() - baseline_start_time
        print(f"Baseline training completed in {baseline_duration:.1f}s ({len(baseline_logs)} epochs)")
    except Exception as e:
        print(f"Baseline training failed: {str(e)}")
        baseline_duration = time.time() - baseline_start_time
        baseline_logs = []
    
    # Step 2: Train bidirectional model
    print("\nSTEP 2: Training Bidirectional ComplEx Model")
    print("-" * 50)
    
    bidirectional_start_time = time.time()
    try:
        bidirectional_model, bidirectional_dir, bidirectional_logs = train_bidirectional_model_with_callbacks(
            output_dir=bidirectional_output_dir,
            max_epochs=max_epochs,
            probability_threshold=probability_threshold,
            max_entities=max_entities
        )
        bidirectional_duration = time.time() - bidirectional_start_time
        print(f"Bidirectional training completed in {bidirectional_duration:.1f}s ({len(bidirectional_logs)} epochs)")
    except Exception as e:
        print(f"Bidirectional training failed: {str(e)}")
        bidirectional_duration = time.time() - bidirectional_start_time
        bidirectional_logs = []
    
    # Step 3: Generate comparison plots
    print("\nSTEP 3: Generating Comparison Plots")
    print("-" * 50)
    
    plots_start_time = time.time()
    try:
        baseline_csv = os.path.join(baseline_output_dir, "baseline_epoch_metrics.csv")
        bidirectional_csv = os.path.join(bidirectional_output_dir, "bidirectional_epoch_metrics.csv")
        
        create_training_comparison_plots(
            baseline_csv=baseline_csv,
            bidirectional_csv=bidirectional_csv,
            output_dir=plots_output_dir
        )
        plots_duration = time.time() - plots_start_time
        print(f"Plots generated in {plots_duration:.1f}s")
    except Exception as e:
        print(f"Plotting failed: {str(e)}")
        plots_duration = time.time() - plots_start_time
    
    # Summary
    total_duration = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Baseline training time:     {baseline_duration:.1f}s ({len(baseline_logs)} epochs)")
    print(f"Bidirectional training time: {bidirectional_duration:.1f}s ({len(bidirectional_logs)} epochs)")
    print(f"Plotting time:              {plots_duration:.1f}s")
    print(f"Total pipeline time:        {total_duration:.1f}s")
    
    print(f"\nOutput files:")
    print(f"  Baseline model: {baseline_output_dir}/")
    print(f"  Bidirectional model: {bidirectional_output_dir}/")
    print(f"  Comparison plots: {plots_output_dir}/")
    
    # Check if CSV files exist
    baseline_csv = os.path.join(baseline_output_dir, "baseline_epoch_metrics.csv")
    bidirectional_csv = os.path.join(bidirectional_output_dir, "bidirectional_epoch_metrics.csv")
    
    if os.path.exists(baseline_csv):
        print(f"Baseline metrics: {baseline_csv}")
    else:
        print(f"Baseline metrics: {baseline_csv} (not found)")
        
    if os.path.exists(bidirectional_csv):
        print(f"Bidirectional metrics: {bidirectional_csv}")
    else:
        print(f"Bidirectional metrics: {bidirectional_csv} (not found)")
    
    plots_file = os.path.join(plots_output_dir, "training_metrics_comparison.png")
    if os.path.exists(plots_file):
        print(f"Comparison plots: {plots_file}")
    else:
        print(f"Comparison plots: {plots_file} (not found)")
    
    print("\nTraining comparison pipeline completed!")
    return baseline_logs, bidirectional_logs

def main():
    """Parse command line arguments and run the comparison."""
    parser = argparse.ArgumentParser(description="Run ComplEx training comparison pipeline")
    
    parser.add_argument("--baseline-output", type=str, 
                        default="models/baseline_complex_with_callbacks",
                        help="Output directory for baseline model")
    parser.add_argument("--bidirectional-output", type=str, 
                        default="models/bidirectional_complex_with_callbacks",
                        help="Output directory for bidirectional model")
    parser.add_argument("--plots-output", type=str, 
                        default="metrics_comparison_plots",
                        help="Output directory for comparison plots")
    parser.add_argument("--max-epochs", type=int, default=5,
                        help="Maximum training epochs (default: 5 for testing)")
    parser.add_argument("--max-entities", type=int, default=100,
                        help="Maximum entities to process for bidirectional model (default: 100)")
    parser.add_argument("--probability-threshold", type=float, default=0.5,
                        help="Probability threshold for recommendations (default: 0.5)")
    
    args = parser.parse_args()
    
    # Run the comparison
    run_training_comparison(
        baseline_output_dir=args.baseline_output,
        bidirectional_output_dir=args.bidirectional_output,
        plots_output_dir=args.plots_output,
        max_epochs=args.max_epochs,
        max_entities=args.max_entities,
        probability_threshold=args.probability_threshold
    )

if __name__ == "__main__":
    main() 