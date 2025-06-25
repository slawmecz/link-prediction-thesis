#!/usr/bin/env python
"""
Script to train and save a baseline ComplEx model with epoch-by-epoch metrics logging.
Based on complex_baseline.py but uses PyKEEN callbacks for metric tracking.
"""

import os
import os.path as osp
import logging
import torch
import numpy as np
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.training.callbacks import TrainingCallback
import argparse
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config(key, default=None):
    """Get configuration from environment or use default."""
    configs = {
        'dataset.name': 'FB15k237',  # or CoDExSmall
        'model.type': 'ComplEx',
        'model.embedding_dim': 1000,  
        'model.max_epochs': 100,     
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

class MetricLoggerCallback(TrainingCallback):
    """Custom callback to log evaluation metrics after each epoch."""
    
    def __init__(self, output_dir, validation_triples_factory, training_triples_factory, test_triples_factory, eval_batch_size=256):
        super().__init__()
        self.output_dir = output_dir
        self.validation_triples_factory = validation_triples_factory
        self.training_triples_factory = training_triples_factory
        self.test_triples_factory = test_triples_factory
        self.eval_batch_size = eval_batch_size
        self.logs = []
        self.csv_path = osp.join(output_dir, 'baseline_epoch_metrics.csv')
        self.json_path = osp.join(output_dir, 'baseline_epoch_metrics.json')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs):
        """Called after each training epoch."""
        print(f"\nLogging metrics for epoch {epoch + 1}...")
        
        try:
            # Import evaluator here to avoid circular imports
            from pykeen.evaluation import RankBasedEvaluator
            
            # Create evaluator if not available
            evaluator = RankBasedEvaluator()
            
            # Evaluate model on validation set with proper filtering
            start_time = time.time()
            result = evaluator.evaluate(
                model=self.model,
                mapped_triples=self.validation_triples_factory.mapped_triples,
                batch_size=self.eval_batch_size,
                additional_filter_triples=[
                    self.training_triples_factory.mapped_triples,
                    self.validation_triples_factory.mapped_triples,
                    self.test_triples_factory.mapped_triples
                ]
            )
            eval_time = time.time() - start_time
            
            # Extract metrics from nested structure
            result_dict = result.to_dict()
            metrics = {'epoch': epoch + 1, 'loss': float(epoch_loss), 'eval_time': eval_time}
            
            # Navigate to realistic metrics
            if 'both' in result_dict and 'realistic' in result_dict['both']:
                realistic_metrics = result_dict['both']['realistic']
                
                # Extract specific metrics
                for k in [1, 3, 5, 10]:
                    key = f'hits_at_{k}'
                    if key in realistic_metrics:
                        metrics[f'Hits@{k}'] = float(realistic_metrics[key])
                
                # Extract MRR and Mean Rank
                if 'inverse_harmonic_mean_rank' in realistic_metrics:
                    metrics['MRR'] = float(realistic_metrics['inverse_harmonic_mean_rank'])
                
                if 'mean_rank' in realistic_metrics:
                    metrics['Mean_Rank'] = float(realistic_metrics['mean_rank'])
            
            self.logs.append(metrics)
            
            # Save to CSV and JSON after each epoch
            df = pd.DataFrame(self.logs)
            df.to_csv(self.csv_path, index=False)
            
            with open(self.json_path, 'w') as f:
                json.dump(self.logs, f, indent=2)
            
            # Print current metrics
            print(f"Epoch {epoch + 1} metrics (eval time: {eval_time:.1f}s):")
            for metric, value in metrics.items():
                if metric not in ['epoch', 'loss', 'eval_time']:
                    if metric == 'Mean_Rank':
                        print(f"  {metric}: {value:.1f}")
                    else:
                        print(f"  {metric}: {value:.4f}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Log error but continue training
            error_metrics = {
                'epoch': epoch + 1, 
                'loss': float(epoch_loss), 
                'error': str(e)
            }
            self.logs.append(error_metrics)

def train_baseline_model_with_callbacks(
    output_dir, 
    dataset_name=None, 
    model_type=None, 
    embedding_dim=None, 
    max_epochs=None
):
    """
    Train a baseline ComplEx model with epoch-by-epoch metric logging.
    
    Args:
        output_dir: Directory to save the model and metrics
        dataset_name: Name of the dataset (FB15k237 or CoDExSmall)
        model_type: Type of model (ComplEx)
        embedding_dim: Dimension of entity/relation embeddings
        max_epochs: Maximum number of training epochs
    """
    print("\n=== Training Baseline ComplEx Model with Callbacks ===")
    
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
    unique_checkpoint_name = f"baseline_checkpoint_{int(time.time())}.pt"
    custom_checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(custom_checkpoint_dir, exist_ok=True)
    
    # Create callback for metric logging
    callback = MetricLoggerCallback(output_dir, dataset.validation, dataset.training, dataset.testing, get_config('model.batch_size_eval'))
    
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
        },
        'callbacks': [callback],  # Add our custom callback
    }
    
    print(f"Checkpoints will be saved to {custom_checkpoint_dir}/{unique_checkpoint_name}")
    
    optimizer_kwargs = {'lr': get_config('model.learning_rate')}
    
    lr_scheduler_kwargs = {
        'gamma': 0.95,  # Decay rate for ExponentialLR
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
    model_file = osp.join(output_dir, 'baseline_trained_model.pkl')
    torch.save(result.model, model_file)
    print(f"Saved trained model to {model_file}")
    
    # Save final metrics
    metrics_file = osp.join(output_dir, 'final_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Model: {model_type} (Baseline)\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Embedding Dim: {embedding_dim}\n")
        f.write(f"Epochs trained: {len(callback.logs)}\n")
        f.write(f"Final Test Metrics:\n")
        
        # Format metrics, handling nested structures
        metrics_dict = result.metric_results.to_dict()
        for metric_name, metric_value in metrics_dict.items():
            f.write(f"{metric_name}: ")
            if isinstance(metric_value, (int, float)):
                f.write(f"{metric_value:.6f}\n")
            else:
                f.write(f"{str(metric_value)}\n")
    
    print(f"Saved metrics to {metrics_file}")
    print(f"Epoch-by-epoch metrics saved to {callback.csv_path}")
    print(f"Training completed with {len(callback.logs)} epochs logged!")
    
    return result.model, output_dir, callback.logs

def main():
    """Parse command line arguments and train the model."""
    parser = argparse.ArgumentParser(description="Train a baseline ComplEx model with epoch metrics")
    parser.add_argument("--output-dir", type=str, default="models/baseline_complex_with_callbacks",
                        help="Output directory for the trained model (default: models/baseline_complex_with_callbacks)")
    parser.add_argument("--dataset", type=str, choices=["FB15k237", "CoDExSmall"],
                        help=f"Dataset to use (default: {get_config('dataset.name')})")
    parser.add_argument("--model", type=str,
                        help=f"Model type (default: {get_config('model.type')})")
    parser.add_argument("--embedding-dim", type=int,
                        help=f"Embedding dimension (default: {get_config('model.embedding_dim')})")
    parser.add_argument("--max-epochs", type=int,
                        help=f"Maximum training epochs (default: {get_config('model.max_epochs')})")
    
    args = parser.parse_args()
    
    train_baseline_model_with_callbacks(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        model_type=args.model,
        embedding_dim=args.embedding_dim,
        max_epochs=args.max_epochs
    )

if __name__ == "__main__":
    main() 