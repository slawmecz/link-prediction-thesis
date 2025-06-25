#!/usr/bin/env python
"""
Script to train and save a baseline ComplEx model without any extended triples.
"""

import os
import os.path as osp
import logging
import torch
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import argparse
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config(key, default=None):
    """Get configuration from environment or use default."""
    configs = {
        'dataset.name': 'FB15k237',  # or CoDExSmall
        'model.type': 'ComplEx',
        'model.embedding_dim': 1000,  
        'model.max_epochs': 19,     
        'model.batch_size_train': 1000,  
        'model.batch_size_eval': 256,  
        'model.learning_rate': 0.1,  
        'sampling.negative_o': 1000,  
        'model.dropout': 0.5,  
        'model.regularize_weight': 0.05,  
        'model.relation_dropout': 0.22684140529516872,  
        'model.relation_regularize_weight': 8.266519211068944e-14,  
    }
    return configs.get(key, default)

def train_baseline_model(output_dir, dataset_name=None, model_type=None, embedding_dim=None, max_epochs=None):
    """
    Train a baseline ComplEx model and save it.
    
    Args:
        output_dir: Directory to save the model and triples
        dataset_name: Name of the dataset (FB15k237 or CoDExSmall)
        model_type: Type of model (ComplEx)
        embedding_dim: Dimension of entity/relation embeddings
        max_epochs: Maximum number of training epochs
    """
    print("\n=== Training Baseline ComplEx Model ===")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set device (GPU if available, CPU otherwise)
    device = torch.device('cuda' if cuda_available else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    os.environ["SEED"] = "42"
    
    # Load the dataset
    dataset_name = dataset_name or get_config('dataset.name')
    if dataset_name == "CoDExSmall":
        from pykeen.datasets import CoDExSmall
        dataset = CoDExSmall()
    elif dataset_name == "FB15k237":
        from pykeen.datasets import FB15k237
        dataset = FB15k237()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Print dataset info
    print(f"\nDataset: {dataset_name}")
    print(f"Entities: {len(dataset.entity_to_id)}")
    print(f"Relations: {len(dataset.relation_to_id)}")
    print(f"Training triples: {len(dataset.training.mapped_triples)}")
    print(f"Testing triples: {len(dataset.testing.mapped_triples)}")
    print(f"Validation triples: {len(dataset.validation.mapped_triples)}")
    
    # Setup model parameters
    model_type = model_type or get_config('model.type')
    embedding_dim = embedding_dim or get_config('model.embedding_dim')
    max_epochs = max_epochs or get_config('model.max_epochs')
    
    # Use optimal model parameters for ComplEx
    model_kwargs = {
        'embedding_dim': embedding_dim,
        'regularizer': 'LpRegularizer',
        'regularizer_kwargs': {
            'weight': get_config('model.regularize_weight'),
            'p': 3,  
        },
    }
    
    # Use a unique checkpoint path to avoid conflicts
    unique_checkpoint_name = f"checkpoint_{int(time.time())}.pt"
    custom_checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(custom_checkpoint_dir, exist_ok=True)
    
    training_kwargs = {
        'num_epochs': max_epochs,
        'batch_size': get_config('model.batch_size_train'),
        'use_tqdm': True,  # Show progress bars
        'use_tqdm_batch': True,  # Show batch progress
        'checkpoint_name': unique_checkpoint_name,  # Timestamp-based unique name
        'checkpoint_directory': custom_checkpoint_dir,  # Custom directory
        'checkpoint_frequency': 5,  # Save checkpoint every 5 epochs
        'tqdm_kwargs': {
            'mininterval': 2.0,  # Update progress bar at most every 2 seconds
            'miniters': 5,  # Update after at least 5 iterations
        }
    }
    
    print(f"Checkpoints will be saved to {custom_checkpoint_dir}/{unique_checkpoint_name}")
    
    optimizer_kwargs = {'lr': get_config('model.learning_rate')}
    
    lr_scheduler_kwargs = {
        'gamma': 0.95,  
    }
    
    stopper_kwargs = {
        'patience': 10,
        'frequency': 10,
        'metric': 'hits@10',
        'relative_delta': 0.0001
    }
    
    evaluation_kwargs = {
        'batch_size': get_config('model.batch_size_eval'),
    }
    
    # Train the model
    print(f"\nTraining {model_type} model with embedding_dim={embedding_dim}")
    result = pipeline(
        training=dataset.training,
        testing=dataset.testing,
        validation=dataset.validation,
        model=model_type,
        loss='crossentropy',
        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
        optimizer='Adam',
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler='ExponentialLR',
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        negative_sampler='basic',
        negative_sampler_kwargs={
            'num_negs_per_pos': get_config('sampling.negative_o'),
        },
        evaluation_kwargs=evaluation_kwargs,
        random_seed=int(os.environ.get("SEED", 42)),
        device=device,  # Add GPU support
        stopper='early',
        stopper_kwargs=stopper_kwargs
    )
    
    # Save the model
    model_file = osp.join(output_dir, 'trained_model.pkl')
    torch.save(result.model, model_file)
    print(f"Saved trained model to {model_file}")
    
    # Save training triples
    training_file = osp.join(output_dir, 'training_triples')
    try:
        # Try to save in binary format
        dataset.training.to_path_binary(training_file)
        print(f"Saved training triples in binary format to {training_file}")
    except Exception as e:
        print(f"Could not save in binary format, saving as CSV: {e}")
        # Fall back to CSV format
        csv_file = osp.join(output_dir, 'training_triples.csv')
        dataset.training.to_path(csv_file)
        print(f"Saved training triples in CSV format to {csv_file}")
    
    # Save testing triples
    testing_file = osp.join(output_dir, 'testing_triples')
    try:
        dataset.testing.to_path_binary(testing_file)
        print(f"Saved testing triples in binary format to {testing_file}")
    except Exception:
        csv_file = osp.join(output_dir, 'testing_triples.csv')
        dataset.testing.to_path(csv_file)
        print(f"Saved testing triples in CSV format to {csv_file}")
    
    # Save validation triples
    validation_file = osp.join(output_dir, 'validation_triples')
    try:
        dataset.validation.to_path_binary(validation_file)
        print(f"Saved validation triples in binary format to {validation_file}")
    except Exception:
        csv_file = osp.join(output_dir, 'validation_triples.csv')
        dataset.validation.to_path(csv_file)
        print(f"Saved validation triples in CSV format to {csv_file}")
    
    # Save metrics
    metrics_file = osp.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Embedding Dim: {embedding_dim}\n")
        f.write(f"Test Metrics:\n")
        
        # Format metrics, handling nested structures
        metrics_dict = result.metric_results.to_dict()
        for metric_name, metric_value in metrics_dict.items():
            f.write(f"{metric_name}: ")
            # Simple case: just a number
            if isinstance(metric_value, (int, float)):
                f.write(f"{metric_value:.6f}\n")
            # Dictionary or other complex structure
            else:
                f.write(f"{str(metric_value)}\n")
    
    print(f"Saved metrics to {metrics_file}")
    
    # Print key metrics in a clear format
    print("\n=== BASELINE COMPLEX MODEL EVALUATION METRICS ===")
    metrics = result.metric_results.to_dict()
    
    # Print hits@k metrics
    for k in [1, 3, 5, 10]:
        key = f'hits_at_{k}'
        if key in metrics:
            print(f"hits@{k}: {metrics[key]:.4f}")
    
    # Print mean rank metrics
    if 'mean_rank' in metrics:
        print(f"mean_rank: {metrics['mean_rank']:.4f}")
    
    # Print mean reciprocal rank
    if 'inverse_harmonic_mean_rank' in metrics:
        print(f"mean_reciprocal_rank: {metrics['inverse_harmonic_mean_rank']:.4f}")
    elif 'mean_reciprocal_rank' in metrics:
        print(f"mean_reciprocal_rank: {metrics['mean_reciprocal_rank']:.4f}")
        
    # Print raw metrics dict for debugging
    print("\nAll available metrics:")
    for key in sorted(metrics.keys()):
        print(f"  {key}: {metrics[key]}")
    
    print("\nBaseline ComplEx model training complete!")
    
    # Print final summary with separator lines
    print("\n" + "="*80)
    print("=== FINAL BASELINE MODEL SUMMARY ===")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Total training triples: {len(dataset.training.mapped_triples):,}")
    print(f"Total entities: {len(dataset.entity_to_id):,}")
    print(f"Total relations: {len(dataset.relation_to_id):,}")
    print("\nKey Metrics:")
    for k in [1, 3, 5, 10]:
        key = f'hits_at_{k}'
        if key in metrics:
            print(f"HITS@{k}: {metrics[key]:.4f}")
    if 'mean_rank' in metrics:
        print(f"MEAN RANK: {metrics['mean_rank']:.4f}")
    if 'inverse_harmonic_mean_rank' in metrics or 'mean_reciprocal_rank' in metrics:
        mrr = metrics.get('inverse_harmonic_mean_rank', metrics.get('mean_reciprocal_rank'))
        print(f"MEAN RECIPROCAL RANK: {mrr:.4f}")
    print("="*80)
    
    return result.model, output_dir

def main():
    """Parse command line arguments and train the model."""
    parser = argparse.ArgumentParser(description="Train a baseline ComplEx KG embedding model")
    parser.add_argument("--output-dir", type=str, default="models/baseline_complex",
                        help="Output directory for the trained model (default: models/baseline_complex)")
    parser.add_argument("--dataset", type=str, choices=["FB15k237", "CoDExSmall"],
                        help=f"Dataset to use (default: {get_config('dataset.name')})")
    parser.add_argument("--model", type=str,
                        help=f"Model type (default: {get_config('model.type')})")
    parser.add_argument("--embedding-dim", type=int,
                        help=f"Embedding dimension (default: {get_config('model.embedding_dim')})")
    parser.add_argument("--max-epochs", type=int,
                        help=f"Maximum training epochs (default: {get_config('model.max_epochs')})")
    
    args = parser.parse_args()
    
    train_baseline_model(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        model_type=args.model,
        embedding_dim=args.embedding_dim,
        max_epochs=args.max_epochs
    )

if __name__ == "__main__":
    main() 