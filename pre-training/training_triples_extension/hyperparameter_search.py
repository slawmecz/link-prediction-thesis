#!/usr/bin/env python
"""
Script to perform hyperparameter search for the extended model.
Focuses on probability_threshold, max_recommendations, and sampling_rate.
The custom implementation is for the bidirectional model.
"""

import os
import argparse
import itertools
import numpy as np
from datetime import datetime
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Results will not be logged to Weights & Biases.")

# Import train_extended_model functionality
import complex_extended_bidirectional as train_extended_model
from complex_extended_bidirectional import get_config


# Define default hyperparameter search grid. Change as needed
DEFAULT_PROBABILITY_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_MAX_RECOMMENDATIONS = [10]
DEFAULT_SAMPLING_RATES = [0.0]


# Default wandb configuration
WANDB_PROJECT = "recommender-server" 
WANDB_ENTITY = "slaw-mecz-vrije-universiteit-amsterdam"

def run_hyperparameter_search(
    baseline_dir,
    output_dir="models/hyperparameter_search", #change as needed
    dataset_name=None,
    model_type=None,
    embedding_dim=None,
    probability_thresholds=None,
    max_recommendations=None,
    sampling_rates=None,
    api_url=None,
    use_wandb=False,
    max_combinations=None
):
    """
    Run hyperparameter search for the extended model.
    
    Args:
        baseline_dir: Directory with baseline model
        output_dir: Base output directory for hyperparameter search results
        dataset_name: Name of the dataset (FB15k237 or CoDExSmall)
        model_type: Type of model (TransE, DistMult, etc.)
        embedding_dim: Dimension of entity/relation embeddings
        probability_thresholds: List of probability thresholds to try
        max_recommendations: List of max recommendations values to try
        sampling_rates: List of sampling rates to try
        api_url: URL of the recommendation API
        use_wandb: Whether to log results to wandb
        max_combinations: Maximum number of combinations to try
    
    Returns:
        DataFrame with results for all hyperparameter combinations
    """
    # Initialize hyperparameter grids with defaults if not provided
    probability_thresholds = probability_thresholds or DEFAULT_PROBABILITY_THRESHOLDS
    max_recommendations = max_recommendations or DEFAULT_MAX_RECOMMENDATIONS
    sampling_rates = sampling_rates or DEFAULT_SAMPLING_RATES
    
    # Generate all combinations of hyperparameters
    hyperparameter_grid = list(itertools.product(
        probability_thresholds,
        max_recommendations,
        sampling_rates
    ))
    
    # Limit the number of combinations if specified
    if max_combinations and max_combinations < len(hyperparameter_grid):
        print(f"Limiting to {max_combinations} combinations out of {len(hyperparameter_grid)} possible combinations")
        np.random.seed(42)  # For reproducibility
        hyperparameter_grid = np.random.choice(hyperparameter_grid, max_combinations, replace=False)
    
    # Initialize wandb flag
    wandb_initialized = False
    if not use_wandb or not WANDB_AVAILABLE:
        print("Weights & Biases logging disabled")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Prepare results storage
    results = []
    
    # Process each hyperparameter combination
    print(f"\n=== Running Hyperparameter Search with {len(hyperparameter_grid)} Combinations ===\n")
    
    for i, (prob_threshold, max_recs, samp_rate) in enumerate(hyperparameter_grid):
        # Create specific output directory for this combination
        combo_name = f"prob{prob_threshold}_maxrec{max_recs}_samp{samp_rate}"
        combo_dir = os.path.join(output_dir, combo_name)
        
        print(f"\n[{i+1}/{len(hyperparameter_grid)}] Testing: prob_threshold={prob_threshold}, "
              f"max_recommendations={max_recs}, sampling_rate={samp_rate}")
        
        # Create a separate wandb run for each combination
        if use_wandb and WANDB_AVAILABLE:
            if wandb_initialized:
                wandb.finish()  # End previous run if any
            
            # Create a descriptive name for this specific combination
            run_name = f"complex_bidirectional_prob{prob_threshold}_rec{max_recs}_samp{samp_rate}"
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_name,
                group="hyperparameter_search",
                job_type="combination_test",
                config={
                    "baseline_dir": baseline_dir,
                    "probability_threshold": prob_threshold,
                    "max_recommendations": max_recs, 
                    "sampling_rate": samp_rate,
                    "dataset": dataset_name or get_config('dataset.name'),
                    "model_type": model_type or get_config('model.type'),
                    "embedding_dim": embedding_dim or get_config('model.embedding_dim'),
                    "api_url": api_url or get_config('api.url')
                }
            )
            wandb_initialized = True
        
        # Update the max_recommendations and probability_threshold in environment
        os.environ["PROBABILITY_THRESHOLD"] = str(prob_threshold)
        os.environ["MAX_RECOMMENDATIONS"] = str(max_recs)
        os.environ["SAMPLING_RATE"] = str(samp_rate)
        
        # Configure the training
        # Override the default configs with our test values
        orig_configs = {}
        
        # Train the extended model with these hyperparameters
        try:
            # Save original configs for reference
            orig_configs['probability_threshold'] = get_config('probability_threshold')
            orig_configs['max_recommendations'] = get_config('max_recommendations')
            orig_configs['sampling_rate'] = get_config('sampling_rate')
            
            # Set up specialized configs for this run
            configs = {
                'probability_threshold': prob_threshold,
                'max_recommendations': max_recs,
                'sampling_rate': samp_rate
            }
            
            # Copy original get_config function
            original_get_config = get_config
            
            # Define a new get_config function that uses our configs
            def patched_get_config(key, default=None):
                if key in configs:
                    return configs[key]
                return original_get_config(key, default)
            
            # Train model with hyperparameters
            try:
                # Monkey patch the get_config function
                # This is a bit of a hack, but allows us to override configs without changing code
                #import train_extended_model
                #train_extended_model.get_config = patched_get_config
                
                # Train the model
                model, model_dir, direct_metrics = train_extended_model.train_extended_model(
                    output_dir=combo_dir,
                    baseline_model_dir=baseline_dir,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    embedding_dim=embedding_dim,
                    probability_threshold=prob_threshold,
                    sampling_rate=samp_rate
                )
                
                # Get metrics from the direct metrics returned by PyKEEN
                metrics = {}
                triples_info = {}
                
                if direct_metrics:
                    # The complex_extended_bidirectional.py returns hierarchical metrics with optimistic/realistic/pessimistic modes
                    print("\nUsing processed metrics from complex_extended_bidirectional:")
                    
                    print(f"Available metrics keys: {list(direct_metrics.keys())}")
                    
                    # Check for hierarchical metrics structure (new PyKEEN format)
                    hierarchical_metrics = False
                    if 'head' in direct_metrics and 'tail' in direct_metrics and 'both' in direct_metrics:
                        hierarchical_metrics = True
                        print("Detected hierarchical metrics structure (head/tail/both)")
                        
                        # Extract metrics from the 'both' section which contains averaged metrics
                        both_metrics = direct_metrics.get('both', {})
                        print(f"Available modes in 'both' section: {list(both_metrics.keys())}")
                        
                        # Get the evaluation mode - typically 'optimistic' or 'realistic'
                        eval_mode = 'optimistic'
                        if eval_mode in both_metrics:
                            mode_metrics = both_metrics[eval_mode]
                            print(f"Using '{eval_mode}' evaluation mode with metrics: {list(mode_metrics.keys())}")
                            
                            # Extract MRR (inverse_harmonic_mean_rank)
                            if 'inverse_harmonic_mean_rank' in mode_metrics:
                                value = mode_metrics['inverse_harmonic_mean_rank']
                                metrics["mrr"] = value
                                if pd.isna(value):
                                    print(f"  WARNING: MRR from 'both.{eval_mode}' is NaN")
                                else:
                                    print(f"  mrr: {value}")
                            
                            # Extract mean_rank
                            if 'arithmetic_mean_rank' in mode_metrics:
                                value = mode_metrics['arithmetic_mean_rank']
                                metrics["mean_rank"] = value
                                if pd.isna(value):
                                    print(f"  WARNING: mean_rank from 'both.{eval_mode}' is NaN")
                                else:
                                    print(f"  mean_rank: {value}")
                            
                            # Extract hits@k if available
                            for k in [1, 3, 5, 10]:
                                key = f'hits_at_{k}'
                                if key in mode_metrics:
                                    value = mode_metrics[key]
                                    metrics[f"hits@{k}"] = value
                                    if pd.isna(value):
                                        print(f"  WARNING: hits@{k} from 'both.{eval_mode}' is NaN")
                                    else:
                                        print(f"  hits@{k}: {value}")
                                else:
                                    print(f"  WARNING: hits@{k} not found in 'both.{eval_mode}'")
                        else:
                            print(f"  WARNING: '{eval_mode}' evaluation mode not found in 'both' section")
                            print(f"  Available modes: {list(both_metrics.keys())}")
                    # Check for optimistic/realistic/pessimistic structure (complex_extended_bidirectional format)
                    elif 'optimistic' in direct_metrics and 'realistic' in direct_metrics:
                        hierarchical_metrics = True
                        print("Detected optimistic/realistic/pessimistic metrics structure")
                        
                        # Use realistic metrics by default
                        eval_mode = 'realistic'
                        if eval_mode in direct_metrics:
                            mode_metrics = direct_metrics[eval_mode]
                            print(f"Using '{eval_mode}' evaluation mode with metrics: {list(mode_metrics.keys())}")
                            
                            # Extract hits@k metrics
                            for k in [1, 3, 5, 10]:
                                key = f'hits_at_{k}'
                                if key in mode_metrics:
                                    value = mode_metrics[key]
                                    metrics[f"hits@{k}"] = value
                                    if pd.isna(value):
                                        print(f"  WARNING: hits@{k} from '{eval_mode}' is NaN")
                                    else:
                                        print(f"  hits@{k}: {value}")
                                else:
                                    print(f"  WARNING: hits@{k} not found in '{eval_mode}'")
                            
                            # Extract mean rank (arithmetic_mean_rank)
                            if 'arithmetic_mean_rank' in mode_metrics:
                                value = mode_metrics['arithmetic_mean_rank']
                                metrics["mean_rank"] = value
                                if pd.isna(value):
                                    print(f"  WARNING: mean_rank from '{eval_mode}' is NaN")
                                else:
                                    print(f"  mean_rank: {value}")
                            else:
                                print(f"  WARNING: arithmetic_mean_rank not found in '{eval_mode}'")
                            
                            # Extract MRR (inverse_harmonic_mean_rank)
                            if 'inverse_harmonic_mean_rank' in mode_metrics:
                                value = mode_metrics['inverse_harmonic_mean_rank']
                                metrics["mrr"] = value
                                if pd.isna(value):
                                    print(f"  WARNING: MRR from '{eval_mode}' is NaN")
                                else:
                                    print(f"  mrr: {value}")
                            else:
                                print(f"  WARNING: inverse_harmonic_mean_rank not found in '{eval_mode}'")
                        else:
                            print(f"  WARNING: '{eval_mode}' evaluation mode not found")
                            print(f"  Available modes: {list(direct_metrics.keys())}")
                    else:
                        # Original flat metrics format - extract hits@k
                        for k in [1, 3, 5, 10]:
                            key = f'hits_at_{k}'
                            if key in direct_metrics:
                                value = direct_metrics[key]
                                metrics[f"hits@{k}"] = value
                                if pd.isna(value):
                                    print(f"  WARNING: hits@{k} is NaN")
                                else:
                                    print(f"  hits@{k}: {value}")
                            else:
                                print(f"  WARNING: hits@{k} metric (key '{key}') not found")
                        
                        # Extract mean rank
                        if 'mean_rank' in direct_metrics:
                            value = direct_metrics['mean_rank']
                            metrics["mean_rank"] = value
                            if pd.isna(value):
                                print(f"  WARNING: mean_rank is NaN")
                            else:
                                print(f"  mean_rank: {value}")
                        else:
                            # Check for alternative keys
                            alt_keys = [k for k in direct_metrics.keys() if "rank" in k.lower() and "mean" in k.lower()]
                            if alt_keys:
                                print(f"  Using alternative mean rank key: {alt_keys[0]}")
                                value = direct_metrics[alt_keys[0]]
                                metrics["mean_rank"] = value
                                print(f"  mean_rank: {value}")
                            else:
                                print(f"  WARNING: mean_rank metric not found")
                        
                        # Extract MRR
                        mrr_found = False
                        if 'inverse_harmonic_mean_rank' in direct_metrics:
                            value = direct_metrics['inverse_harmonic_mean_rank']
                            metrics["mrr"] = value
                            mrr_found = True
                            if pd.isna(value):
                                print(f"  WARNING: MRR (inverse_harmonic_mean_rank) is NaN")
                            else:
                                print(f"  mrr: {value}")
                        elif 'mean_reciprocal_rank' in direct_metrics:
                            value = direct_metrics['mean_reciprocal_rank']
                            metrics["mrr"] = value
                            mrr_found = True
                            if pd.isna(value):
                                print(f"  WARNING: MRR (mean_reciprocal_rank) is NaN")
                            else:
                                print(f"  mrr: {value}")
                        
                        if not mrr_found:
                            print(f"  WARNING: MRR metric not found")
                    
                    # Print all available metrics for debugging
                    print("\nAll available metrics from complex_extended_bidirectional:")
                    for key, value in direct_metrics.items():
                        if isinstance(value, dict):
                            print(f"  {key}: <dict with {len(value)} items>")
                        else:
                            value_status = "NaN" if pd.isna(value) else str(value)
                            print(f"  {key}: {value_status}")
                else:
                    print("WARNING: No metrics available from complex_extended_bidirectional results")
                    print("This could indicate that the evaluation did not complete successfully.")
                    print("Check the training log for errors or exceptions during evaluation.")
                
                # Still extract triples information from metrics.txt
                metrics_file = os.path.join(combo_dir, 'metrics.txt')
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_text = f.read()
                            
                            # Extract triples information
                            original_triples_match = re.search(r"Original training triples: ([\d,]+)", metrics_text)
                            if original_triples_match:
                                original_triples = int(original_triples_match.group(1).replace(',', ''))
                                triples_info["original_triples"] = original_triples
                            
                            new_triples_match = re.search(r"New triples added: ([\d,]+)", metrics_text)
                            if new_triples_match:
                                new_triples = int(new_triples_match.group(1).replace(',', ''))
                                triples_info["new_triples"] = new_triples
                            
                            total_triples_match = re.search(r"Total triples: ([\d,]+)", metrics_text)
                            if total_triples_match:
                                total_triples = int(total_triples_match.group(1).replace(',', ''))
                                triples_info["total_triples"] = total_triples
                            
                            # Check for entity information
                            original_entities_match = re.search(r"Original entities: ([\d,]+)", metrics_text)
                            if original_entities_match:
                                original_entities = int(original_entities_match.group(1).replace(',', ''))
                                triples_info["original_entities"] = original_entities
                            
                            new_entities_match = re.search(r"New entities: ([\d,]+)", metrics_text)
                            if new_entities_match:
                                new_entities = int(new_entities_match.group(1).replace(',', ''))
                                triples_info["new_entities"] = new_entities
                            
                            total_entities_match = re.search(r"Total entities: ([\d,]+)", metrics_text)
                            if total_entities_match:
                                total_entities = int(total_entities_match.group(1).replace(',', ''))
                                triples_info["total_entities"] = total_entities
                            
                            print("\nExtracted triples information:")
                            for key, value in triples_info.items():
                                print(f"  {key}: {value:,}")
                    
                    except Exception as e:
                        print(f"Error processing metrics file {metrics_file}: {str(e)}")
                else:
                    print(f"Warning: Metrics file not found at {metrics_file}")
                
                # Set the combined metrics for compatibility 
                metrics["combined_hits@1"] = metrics.get("hits@1")
                metrics["combined_hits@3"] = metrics.get("hits@3")
                metrics["combined_hits@5"] = metrics.get("hits@5")
                metrics["combined_hits@10"] = metrics.get("hits@10")
                metrics["combined_mean_rank"] = metrics.get("mean_rank")
                metrics["combined_mean_reciprocal_rank"] = metrics.get("mrr")
                
                # Store the results
                result_record = {
                    "probability_threshold": prob_threshold,
                    "max_recommendations": max_recs,
                    "sampling_rate": samp_rate,
                    "hits@1": metrics.get("hits@1"),
                    "hits@3": metrics.get("hits@3"),
                    "hits@5": metrics.get("hits@5"),
                    "hits@10": metrics.get("hits@10"),
                    "mean_rank": metrics.get("mean_rank"),
                    "mrr": metrics.get("mrr"),
                    "combined_hits@1": metrics.get("combined_hits@1"),
                    "combined_hits@3": metrics.get("combined_hits@3"),
                    "combined_hits@5": metrics.get("combined_hits@5"),
                    "combined_hits@10": metrics.get("combined_hits@10"),
                    "combined_mean_rank": metrics.get("combined_mean_rank"),
                    "combined_mean_reciprocal_rank": metrics.get("combined_mean_reciprocal_rank"),
                    "original_triples": triples_info.get("original_triples"),
                    "new_triples": triples_info.get("new_triples"),
                    "total_triples": triples_info.get("total_triples"),
                    "triples_increase_percent": (triples_info.get("new_triples", 0) / triples_info.get("original_triples", 1) * 100) if triples_info.get("original_triples") else None,
                    "original_entities": triples_info.get("original_entities"),
                    "new_entities": triples_info.get("new_entities"),
                    "total_entities": triples_info.get("total_entities"),
                    "output_dir": combo_dir
                }
                results.append(result_record)
                
                # Log result to wandb
                if wandb_initialized:
                    # Create a dict of metrics, filtering out NaN values
                    wandb_metrics = {
                        'probability_threshold': prob_threshold,
                        'max_recommendations': max_recs,
                        'sampling_rate': samp_rate,
                    }
                    
                    # Add metrics conditionally, handling NaN
                    metric_pairs = [
                        ('combined_hits@1', metrics.get('hits@1')),
                        ('combined_hits@3', metrics.get('hits@3')),
                        ('combined_hits@5', metrics.get('hits@5')),
                        ('combined_hits@10', metrics.get('hits@10')),
                        ('combined_mean_rank', metrics.get('mean_rank')),
                        ('combined_mrr', metrics.get('mrr')),
                        ('combined_mean_reciprocal_rank', metrics.get('mrr')),
                        ('hits@1', metrics.get('hits@1')),
                        ('hits@3', metrics.get('hits@3')),
                        ('hits@5', metrics.get('hits@5')),
                        ('hits@10', metrics.get('hits@10')),
                        ('mean_rank', metrics.get('mean_rank')),
                        ('mrr', metrics.get('mrr')),
                        ('mean_reciprocal_rank', metrics.get('mrr')),
                        ('combo_name', combo_name)
                    ]
                    
                    # Only add non-NaN metrics to wandb log
                    for key, value in metric_pairs:
                        if value is not None and not pd.isna(value):
                            wandb_metrics[key] = value
                        else:
                            # Log a message about the missing metric
                            print(f"Warning: Not logging {key} to wandb because it's None or NaN")
                    
                    # Log the metrics that we do have
                    if wandb_metrics:
                        wandb.log(wandb_metrics)
                
                print(f"Completed training and evaluation for combination {i+1}/{len(hyperparameter_grid)}")
                
            finally:
                # Restore original get_config function
                train_extended_model.get_config = original_get_config
                
        except Exception as e:
            print(f"Error training model with hyperparameters: {e}")
    
    # Convert results to DataFrame for easier analysis
    if results:
        df = pd.DataFrame(results)
        
        # Save results as CSV
        results_file = os.path.join(output_dir, 'hyperparameter_search_results.csv')
        df.to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")
        
        # Create visualizations
        create_visualizations(df, plots_dir)
        
        # Log summary to wandb
        if use_wandb and WANDB_AVAILABLE and df is not None:
            # Finish any previous runs
            if wandb_initialized:
                wandb.finish()
                
            # Create a summary run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_run_name = f"complex_bidirectional_summary_prob{min(probability_thresholds)}-{max(probability_thresholds)}_rec{min(max_recommendations)}-{max(max_recommendations)}_samp{min(sampling_rates)}-{max(sampling_rates)}_{timestamp}"
            
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=summary_run_name,
                job_type="hyperparameter_summary",
                config={
                    "baseline_dir": baseline_dir,
                    "dataset": dataset_name or get_config('dataset.name'),
                    "model_type": model_type or get_config('model.type'),
                    "probability_thresholds": probability_thresholds,
                    "max_recommendations": max_recommendations,
                    "sampling_rates": sampling_rates
                }
            )
                
            # Define a helper function to safely get the best combo for a metric
            def get_best_combo_safe(df, column, find_max=True):
                try:
                    # Filter out NaN values
                    df_valid = df.dropna(subset=[column])
                    if df_valid.empty:
                        return None
                            
                    # Get the index of the best value
                    idx = df_valid[column].idxmax() if find_max else df_valid[column].idxmin()
                    return df_valid.loc[idx]
                except Exception as e:
                    print(f"Error getting best combo for {column}: {str(e)}")
                    return None
            
            # Find the best combinations safely
            best_hits1_combo = get_best_combo_safe(df, 'hits@1')
            best_hits3_combo = get_best_combo_safe(df, 'hits@3')
            best_hits5_combo = get_best_combo_safe(df, 'hits@5')
            best_hits_combo = get_best_combo_safe(df, 'hits@10')
            best_rank_combo = get_best_combo_safe(df, 'mean_rank', False)
            best_mrr_combo = get_best_combo_safe(df, 'mrr')
            
            # Create a summary metrics dict
            summary_metrics = {}
            
            # Helper to safely add metrics
            def safe_add_metric(metrics_dict, name, combo, value_key):
                if combo is not None and value_key in combo and not pd.isna(combo[value_key]):
                    # Add the best_ prefixed metrics
                    metrics_dict[f'best_{name}'] = combo[value_key]
                    metrics_dict[f'best_{name}_threshold'] = combo['probability_threshold']
                    metrics_dict[f'best_{name}_max_recs'] = combo['max_recommendations']
                    metrics_dict[f'best_{name}_sampling'] = combo['sampling_rate']
                    
                    # Also add the regular metric name for compatibility with previous runs
                    metrics_dict[name] = combo[value_key]
            
            # Add all metrics safely
            safe_add_metric(summary_metrics, 'combined_hits@1', best_hits1_combo, 'hits@1')
            safe_add_metric(summary_metrics, 'combined_hits@3', best_hits3_combo, 'hits@3')
            safe_add_metric(summary_metrics, 'combined_hits@5', best_hits5_combo, 'hits@5')
            safe_add_metric(summary_metrics, 'combined_hits@10', best_hits_combo, 'hits@10')
            safe_add_metric(summary_metrics, 'combined_mean_rank', best_rank_combo, 'mean_rank')
            safe_add_metric(summary_metrics, 'combined_mrr', best_mrr_combo, 'mrr')
            safe_add_metric(summary_metrics, 'combined_mean_reciprocal_rank', best_mrr_combo, 'mrr')
            
            safe_add_metric(summary_metrics, 'hits@1', best_hits1_combo, 'hits@1')
            safe_add_metric(summary_metrics, 'hits@3', best_hits3_combo, 'hits@3')
            safe_add_metric(summary_metrics, 'hits@5', best_hits5_combo, 'hits@5')
            safe_add_metric(summary_metrics, 'hits@10', best_hits_combo, 'hits@10')
            safe_add_metric(summary_metrics, 'mean_rank', best_rank_combo, 'mean_rank')
            safe_add_metric(summary_metrics, 'mrr', best_mrr_combo, 'mrr')
            
            # Log the summary metrics
            if summary_metrics:
                wandb.log(summary_metrics)
            else:
                print("Warning: No valid metrics to log to wandb summary")
            
            # Log summary table
            summary_table = wandb.Table(dataframe=df)
            wandb.log({"results_table": summary_table})
            
            # Log plots
            for plot_file in os.listdir(plots_dir):
                if plot_file.endswith('.png'):
                    wandb.log({plot_file.replace('.png', ''): wandb.Image(os.path.join(plots_dir, plot_file))})
            
            # Finish wandb run
            wandb.finish()
        
        return df
    else:
        print("No results collected from hyperparameter search.")
        if wandb_initialized:
            wandb.finish()
        return None

def create_visualizations(df, plots_dir):
    """Create visualizations of hyperparameter search results."""
    # Create heatmaps for each sampling rate
    for sampling_rate in df['sampling_rate'].unique():
        # Filter data for this sampling rate
        df_filtered = df[df['sampling_rate'] == sampling_rate]
        
        # Pivot the data to create a heatmap
        pivot_hits = df_filtered.pivot_table(
            index='probability_threshold', 
            columns='max_recommendations',
            values='hits@10'
        )
        
        pivot_rank = df_filtered.pivot_table(
            index='probability_threshold', 
            columns='max_recommendations',
            values='mean_rank'
        )
        
        pivot_mrr = df_filtered.pivot_table(
            index='probability_threshold', 
            columns='max_recommendations',
            values='mrr'
        )
        
        # Plot hits@10 heatmap
        plt.figure(figsize=(10, 8))
        plt.title(f'Hits@10 for Sampling Rate = {sampling_rate}')
        plt.imshow(pivot_hits, cmap='viridis')
        plt.colorbar(label='Hits@10')
        plt.xlabel('Max Recommendations')
        plt.ylabel('Probability Threshold')
        plt.xticks(range(len(pivot_hits.columns)), pivot_hits.columns)
        plt.yticks(range(len(pivot_hits.index)), pivot_hits.index)
        for i in range(len(pivot_hits.index)):
            for j in range(len(pivot_hits.columns)):
                plt.text(j, i, f"{pivot_hits.iloc[i, j]:.4f}", ha="center", va="center", color="w")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'hits_heatmap_samp{sampling_rate}.png'))
        plt.close()
        
        # Plot mean rank heatmap (lower is better)
        plt.figure(figsize=(10, 8))
        plt.title(f'Mean Rank for Sampling Rate = {sampling_rate}')
        plt.imshow(pivot_rank, cmap='viridis_r')  # reversed colormap as lower is better
        plt.colorbar(label='Mean Rank')
        plt.xlabel('Max Recommendations')
        plt.ylabel('Probability Threshold')
        plt.xticks(range(len(pivot_rank.columns)), pivot_rank.columns)
        plt.yticks(range(len(pivot_rank.index)), pivot_rank.index)
        for i in range(len(pivot_rank.index)):
            for j in range(len(pivot_rank.columns)):
                plt.text(j, i, f"{pivot_rank.iloc[i, j]:.1f}", ha="center", va="center", color="w")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'rank_heatmap_samp{sampling_rate}.png'))
        plt.close()
        
        # Plot MRR heatmap
        plt.figure(figsize=(10, 8))
        plt.title(f'MRR for Sampling Rate = {sampling_rate}')
        plt.imshow(pivot_mrr, cmap='viridis')
        plt.colorbar(label='MRR')
        plt.xlabel('Max Recommendations')
        plt.ylabel('Probability Threshold')
        plt.xticks(range(len(pivot_mrr.columns)), pivot_mrr.columns)
        plt.yticks(range(len(pivot_mrr.index)), pivot_mrr.index)
        for i in range(len(pivot_mrr.index)):
            for j in range(len(pivot_mrr.columns)):
                plt.text(j, i, f"{pivot_mrr.iloc[i, j]:.6f}", ha="center", va="center", color="w")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'mrr_heatmap_samp{sampling_rate}.png'))
        plt.close()
    
    # Create line plots for each metric
    # Hits@10 vs probability_threshold for different max_recommendations
    plt.figure(figsize=(10, 6))
    for max_rec in df['max_recommendations'].unique():
        for samp_rate in df['sampling_rate'].unique():
            df_filtered = df[(df['max_recommendations'] == max_rec) & (df['sampling_rate'] == samp_rate)]
            if not df_filtered.empty:
                plt.plot(
                    df_filtered['probability_threshold'], 
                    df_filtered['hits@10'],
                    marker='o',
                    label=f'Max Rec={max_rec}, Samp={samp_rate}'
                )
    plt.xlabel('Probability Threshold')
    plt.ylabel('Hits@10')
    plt.title('Hits@10 vs Probability Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'hits_vs_threshold.png'))
    plt.close()
    
    # Find the best hyperparameter combinations with null safety checks
    # Helper function to safely get best combo
    def safely_get_best_combo(df, column, find_max=True):
        if df[column].isna().all():
            # Return first row if all values are NaN
            return df.iloc[0] if not df.empty else None
        else:
            # Drop NaN values and find max/min
            df_valid = df.dropna(subset=[column])
            if df_valid.empty:
                return df.iloc[0] if not df.empty else None
            else:
                if find_max:
                    return df_valid.loc[df_valid[column].idxmax()]
                else:
                    return df_valid.loc[df_valid[column].idxmin()]
    
    best_hits1_combo = safely_get_best_combo(df, 'hits@1', True)
    best_hits3_combo = safely_get_best_combo(df, 'hits@3', True)
    best_hits5_combo = safely_get_best_combo(df, 'hits@5', True)
    best_hits_combo = safely_get_best_combo(df, 'hits@10', True)
    best_rank_combo = safely_get_best_combo(df, 'mean_rank', False)  # lower is better
    best_mrr_combo = safely_get_best_combo(df, 'mrr', True)
    
    # Skip visualization if we don't have valid results
    if best_hits1_combo is None or best_hits_combo is None:
        print("Warning: Not enough valid data to create best combinations visualizations")
        return
    
    # Create a summary bar plot
    metrics = ['hits@1', 'hits@5', 'hits@10', 'mean_rank', 'mrr']
    best_combos = {
        'Best Hits@1': best_hits1_combo,
        'Best Hits@3': best_hits3_combo,
        'Best Hits@5': best_hits5_combo,
        'Best Hits@10': best_hits_combo,
        'Best Mean Rank': best_rank_combo,
        'Best MRR': best_mrr_combo
    }
    
    # Prepare data for summary plot
    combo_params = pd.DataFrame({
        'Combo': list(best_combos.keys()),
        'Probability Threshold': [combo['probability_threshold'] for combo in best_combos.values()],
        'Max Recommendations': [combo['max_recommendations'] for combo in best_combos.values()],
        'Sampling Rate': [combo['sampling_rate'] for combo in best_combos.values()]
    })
    
    # Plot the best combinations
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot probability thresholds
    axs[0].bar(combo_params['Combo'], combo_params['Probability Threshold'])
    axs[0].set_title('Best Probability Thresholds')
    axs[0].set_ylabel('Probability Threshold')
    axs[0].grid(axis='y')
    
    # Plot max recommendations
    axs[1].bar(combo_params['Combo'], combo_params['Max Recommendations'])
    axs[1].set_title('Best Max Recommendations')
    axs[1].set_ylabel('Max Recommendations')
    axs[1].grid(axis='y')
    
    # Plot sampling rates
    axs[2].bar(combo_params['Combo'], combo_params['Sampling Rate'])
    axs[2].set_title('Best Sampling Rates')
    axs[2].set_ylabel('Sampling Rate')
    axs[2].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_combinations.png'))
    plt.close()
    
    # Save the best combinations as text
    with open(os.path.join(plots_dir, 'best_combinations.txt'), 'w') as f:
        f.write("=== Best Hyperparameter Combinations ===\n\n")
        
        def safe_write_combo(f, name, combo, metric_name, metric_value):
            if combo is None:
                f.write(f"{name}:\n  No valid data available\n\n")
                return
                
            f.write(f"{name}:\n")
            f.write(f"  Probability Threshold: {combo['probability_threshold']}\n")
            f.write(f"  Max Recommendations: {combo['max_recommendations']}\n")
            f.write(f"  Sampling Rate: {combo['sampling_rate']}\n")
            
            if pd.isna(metric_value):
                f.write(f"  {metric_name}: N/A\n\n")
            else:
                f.write(f"  {metric_name}: {metric_value}\n\n")
        
        safe_write_combo(f, "Best for Hits@1", best_hits1_combo, "Hits@1", 
                          best_hits1_combo['hits@1'] if best_hits1_combo is not None else None)
                          
        safe_write_combo(f, "Best for Hits@3", best_hits3_combo, "Hits@3", 
                          best_hits3_combo['hits@3'] if best_hits3_combo is not None else None)
                          
        safe_write_combo(f, "Best for Hits@5", best_hits5_combo, "Hits@5", 
                          best_hits5_combo['hits@5'] if best_hits5_combo is not None else None)
                          
        safe_write_combo(f, "Best for Hits@10", best_hits_combo, "Hits@10", 
                          best_hits_combo['hits@10'] if best_hits_combo is not None else None)
                          
        safe_write_combo(f, "Best for Mean Rank", best_rank_combo, "Mean Rank", 
                          best_rank_combo['mean_rank'] if best_rank_combo is not None else None)
                          
        safe_write_combo(f, "Best for MRR", best_mrr_combo, "MRR", 
                          best_mrr_combo['mrr'] if best_mrr_combo is not None else None)

def format_value(value):
    """Format a value that might be None, a number, or 'N/A'."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return str(value)

def main():
    """Parse command line arguments and run hyperparameter search."""
    parser = argparse.ArgumentParser(description="Run hyperparameter search for extended model")
    parser.add_argument("--baseline-dir", type=str, default="models/baseline",
                        help="Directory with baseline model (default: models/baseline)")
    parser.add_argument("--output-dir", type=str, default="models/hyperparameter_search",
                        help="Output directory for search results (default: models/hyperparameter_search)")
    parser.add_argument("--dataset", type=str, choices=["FB15k237", "CoDExSmall"],
                        help="Dataset to use (default: from config)")
    parser.add_argument("--model", type=str,
                        help="Model type (default: from config)")
    parser.add_argument("--embedding-dim", type=int,
                        help="Embedding dimension (default: from config)")
    parser.add_argument("--probability-thresholds", type=float, nargs="+",
                        help=f"Probability thresholds to try (default: {DEFAULT_PROBABILITY_THRESHOLDS})")
    parser.add_argument("--max-recommendations", type=int, nargs="+",
                        help=f"Max recommendations values to try (default: {DEFAULT_MAX_RECOMMENDATIONS})")
    parser.add_argument("--sampling-rates", type=float, nargs="+",
                        help=f"Sampling rates to try (default: {DEFAULT_SAMPLING_RATES})")
    parser.add_argument("--api-url", type=str,
                        help="URL of the recommendation API (default: from config)")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--max-combinations", type=int,
                        help="Maximum number of combinations to try (default: all)")
    
    args = parser.parse_args()
    
    result_df = run_hyperparameter_search(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        model_type=args.model,
        embedding_dim=args.embedding_dim,
        probability_thresholds=args.probability_thresholds,
        max_recommendations=args.max_recommendations,
        sampling_rates=args.sampling_rates,
        api_url=args.api_url,
        use_wandb=args.use_wandb,
        max_combinations=args.max_combinations
    )
    
    if result_df is not None:
        print("\n=== Hyperparameter Search Complete ===")
        print("\nBest combinations for each metric:")
        
        # Helper function to safely print best combos
        def safely_print_best_combo(df, column, desc, format_str, find_max=True):
            if df[column].isna().all():
                print(f"\n{desc}: (No valid data)")
                return
                
            df_valid = df.dropna(subset=[column])
            if df_valid.empty:
                print(f"\n{desc}: (No valid data)")
                return
                
            if find_max:
                best_combo = df_valid.loc[df_valid[column].idxmax()]
                value = best_combo[column]
            else:
                best_combo = df_valid.loc[df_valid[column].idxmin()]
                value = best_combo[column]
                
            print(f"\n{desc} ({value:{format_str}}):")
            print(f"  Probability Threshold: {best_combo['probability_threshold']}")
            print(f"  Max Recommendations: {best_combo['max_recommendations']}")
            print(f"  Sampling Rate: {best_combo['sampling_rate']}")
        
        safely_print_best_combo(result_df, 'hits@1', "Best for Hits@1", ".4f")
        safely_print_best_combo(result_df, 'hits@3', "Best for Hits@3", ".4f")
        safely_print_best_combo(result_df, 'hits@5', "Best for Hits@5", ".4f")
        safely_print_best_combo(result_df, 'hits@10', "Best for Hits@10", ".4f")
        safely_print_best_combo(result_df, 'mean_rank', "Best for Mean Rank", ".1f", False)
        safely_print_best_combo(result_df, 'mrr', "Best for MRR", ".6f")
        
        # Create a more detailed summary table
        print("\nCreating detailed summary table...")
        
        # Select columns for the summary table
        summary_columns = [
            'probability_threshold', 'max_recommendations', 'sampling_rate',
            'original_triples', 'new_triples', 'total_triples', 'triples_increase_percent',
            'original_entities', 'new_entities', 'total_entities',
            'hits@1', 'hits@3', 'hits@5', 'hits@10', 'mean_rank', 'mrr'
        ]
        
        # Get just the columns we want in a specific order
        summary_df = result_df[summary_columns].copy()
        
        # Sort by hits@10 in descending order
        summary_df = summary_df.sort_values('hits@10', ascending=False)
        
        # Format percentage columns
        if 'triples_increase_percent' in summary_df.columns:
            summary_df['triples_increase_percent'] = summary_df['triples_increase_percent'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        
        # Save as CSV
        summary_file = os.path.join(args.output_dir, 'hyperparameter_search_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved detailed summary to {summary_file}")
        
        # Also save as human-readable text file
        text_summary_file = os.path.join(args.output_dir, 'hyperparameter_search_summary.txt')
        with open(text_summary_file, 'w') as f:
            f.write("=== Hyperparameter Search Summary ===\n\n")
            
            # Write the date and time
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write the best combinations
            f.write("=== Best Combinations ===\n\n")
            
            # Helper function to safely extract best combos for a metric
            def get_safe_combo_for_metric(df, column, find_max=True):
                if df[column].isna().all():
                    return None
                    
                df_valid = df.dropna(subset=[column])
                if df_valid.empty:
                    return None
                    
                if find_max:
                    return df_valid.loc[df_valid[column].idxmax()]
                else:
                    return df_valid.loc[df_valid[column].idxmin()]
            
            # Get the best combos safely
            safe_best_hits1 = get_safe_combo_for_metric(result_df, 'hits@1', True)
            safe_best_hits10 = get_safe_combo_for_metric(result_df, 'hits@10', True)
            safe_best_rank = get_safe_combo_for_metric(result_df, 'mean_rank', False)
            safe_best_mrr = get_safe_combo_for_metric(result_df, 'mrr', True)
            
            # Write best for Hits@1
            if safe_best_hits1 is None:
                f.write("Best for Hits@1: No valid data available\n\n")
            else:
                f.write(f"Best for Hits@1 ({safe_best_hits1['hits@1']:.4f}):\n")
                f.write(f"  Probability Threshold: {safe_best_hits1['probability_threshold']}\n")
                f.write(f"  Max Recommendations: {safe_best_hits1['max_recommendations']}\n")
                f.write(f"  Sampling Rate: {safe_best_hits1['sampling_rate']}\n")
                f.write(f"  New Triples: {format_value(safe_best_hits1.get('new_triples'))}\n")
                f.write(f"  Triples Increase: {format_value(safe_best_hits1.get('triples_increase_percent'))}\n\n")
            
            # Write best for Hits@10
            if safe_best_hits10 is None:
                f.write("Best for Hits@10: No valid data available\n\n")
            else:
                f.write(f"Best for Hits@10 ({safe_best_hits10['hits@10']:.4f}):\n")
                f.write(f"  Probability Threshold: {safe_best_hits10['probability_threshold']}\n")
                f.write(f"  Max Recommendations: {safe_best_hits10['max_recommendations']}\n")
                f.write(f"  Sampling Rate: {safe_best_hits10['sampling_rate']}\n")
                f.write(f"  New Triples: {format_value(safe_best_hits10.get('new_triples'))}\n")
                f.write(f"  Triples Increase: {format_value(safe_best_hits10.get('triples_increase_percent'))}\n\n")
            
            # Write best for Mean Rank
            if safe_best_rank is None:
                f.write("Best for Mean Rank: No valid data available\n\n")
            else:
                f.write(f"Best for Mean Rank ({safe_best_rank['mean_rank']:.1f}):\n")
                f.write(f"  Probability Threshold: {safe_best_rank['probability_threshold']}\n")
                f.write(f"  Max Recommendations: {safe_best_rank['max_recommendations']}\n")
                f.write(f"  Sampling Rate: {safe_best_rank['sampling_rate']}\n")
                f.write(f"  New Triples: {format_value(safe_best_rank.get('new_triples'))}\n")
                f.write(f"  Triples Increase: {format_value(safe_best_rank.get('triples_increase_percent'))}\n\n")
            
            # Write best for MRR
            if safe_best_mrr is None:
                f.write("Best for MRR: No valid data available\n\n")
            else:
                f.write(f"Best for MRR ({safe_best_mrr['mrr']:.6f}):\n")
                f.write(f"  Probability Threshold: {safe_best_mrr['probability_threshold']}\n")
                f.write(f"  Max Recommendations: {safe_best_mrr['max_recommendations']}\n")
                f.write(f"  Sampling Rate: {safe_best_mrr['sampling_rate']}\n")
                f.write(f"  New Triples: {format_value(safe_best_mrr.get('new_triples'))}\n")
                f.write(f"  Triples Increase: {format_value(safe_best_mrr.get('triples_increase_percent'))}\n\n")
            
            # Write a table of all results
            f.write("=== All Results (sorted by Hits@10) ===\n\n")
            f.write("Prob | MaxRec | Samp | New Triples | Triples % | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MeanRank | MRR\n")
            f.write("-" * 120 + "\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"{row['probability_threshold']:.1f} | ")
                f.write(f"{row['max_recommendations']:6d} | ")
                f.write(f"{row['sampling_rate']:.1f} | ")
                # Use the format_value function to safely handle None values
                new_triples = format_value(row.get('new_triples'))
                f.write(f"{new_triples:11} | ")
                triples_pct = row.get('triples_increase_percent', 'N/A')
                f.write(f"{triples_pct:9} | ")
                
                # Handle numeric metrics similarly
                hits1 = format_value(row.get('hits@1'))
                hits3 = format_value(row.get('hits@3'))
                hits5 = format_value(row.get('hits@5'))
                hits10 = format_value(row.get('hits@10'))
                mean_rank = format_value(row.get('mean_rank'))
                mrr = format_value(row.get('mrr'))
                
                f.write(f"{hits1:6} | ")
                f.write(f"{hits3:6} | ")
                f.write(f"{hits5:6} | ")
                f.write(f"{hits10:7} | ")
                f.write(f"{mean_rank:8} | ")
                f.write(f"{mrr:6}\n")
            
        print(f"Saved text summary to {text_summary_file}")
    
if __name__ == "__main__":
    main() 