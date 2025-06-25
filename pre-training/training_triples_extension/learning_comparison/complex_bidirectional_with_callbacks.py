#!/usr/bin/env python
"""
Script to train a bidirectional ComplEx model with epoch-by-epoch metrics logging.
Based on complex_extended_bidirectional.py but uses PyKEEN callbacks for metric tracking.
"""

import os
import os.path as osp
import logging
import torch
import numpy as np
import pandas as pd
import argparse
import requests
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.training.callbacks import TrainingCallback
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
        'api.url': 'http://localhost:8080/recommender',
        'api.timeout': 30,  # API request timeout in seconds
        'probability_threshold': 0.5,  # Probability threshold for recommendations
        'max_recommendations': 10,  
        'max_new_triples': 10000000,  
        'sampling_rate': 0.0,  # Rate at which to sample from new triples (0.0 = use all)
        'model.random_seed': 42,  
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
        self.csv_path = osp.join(output_dir, 'bidirectional_epoch_metrics.csv')
        self.json_path = osp.join(output_dir, 'bidirectional_epoch_metrics.json')
        
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
                
                # Extract specific metrics we care about
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

def get_entity_outgoing_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """Get all outgoing properties (relations) where the entity is the head."""
    head_triples = triples[triples[:, 0] == entity_id]
    return {f"O:{id_to_relation[rel_id.item()]}" for rel_id in head_triples[:, 1]}

def get_entity_incoming_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """Get all incoming properties (relations) where the entity is the tail."""
    tail_triples = triples[triples[:, 2] == entity_id]
    return {f"I:{id_to_relation[rel_id.item()]}" for rel_id in tail_triples[:, 1]}

def get_recommendations(properties: List[str], api_url: str = None) -> List[Dict[str, Any]]:
    """Get property recommendations from the API."""
    api_url = api_url or get_config('api.url')
    api_timeout = get_config('api.timeout')
    
    try:
        data = {
            "properties": properties,
            "types": []
        }
        
        print(f"Sending request to {api_url} with {len(properties)} properties")
        response = requests.post(api_url, json=data, timeout=api_timeout)
        response.raise_for_status()
        
        recommendations = response.json().get("recommendations", [])
        logger.info(f"Received {len(recommendations)} recommendations")
        return recommendations
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error processing recommendations: {str(e)}")
        return []

def process_recommendations(
    recommendations: List[Dict[str, Any]],
    threshold: float = None,
    max_recommendations: int = None
) -> List[Tuple[str, float]]:
    """Process and filter recommendations based on probability threshold."""
    threshold = threshold or get_config('probability_threshold')
    max_recommendations = max_recommendations or get_config('max_recommendations')
    
    filtered_recommendations = [
        (rec['property'], rec['probability'])
        for rec in recommendations
        if rec['probability'] >= threshold
    ]
    
    # Sort by probability in descending order
    filtered_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N recommendations
    result = filtered_recommendations[:max_recommendations]
    return result

def create_artificial_triples(
    dataset,
    probability_threshold: float = None,
    max_entities: int = 10000000
) -> Tuple[List[torch.Tensor], int]:
    """Create artificial triples based on recommendations using both incoming and outgoing properties."""
    probability_threshold = probability_threshold or get_config('probability_threshold')
    max_new_triples = get_config('max_new_triples')
    
    print("\n=== Creating Artificial Triples (Bidirectional) ===")
    
    # Get all unique entities (both head and tail positions)
    triples = dataset.training.mapped_triples
    all_entities = set(triples[:, 0].tolist()).union(set(triples[:, 2].tolist()))
    print(f"Number of unique entities (head + tail): {len(all_entities)}")
    
    # Create id_to_relation mapping
    id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}
    relation_to_id = dataset.relation_to_id.copy()
    
    # Set next entity ID
    next_entity_id = max(dataset.entity_to_id.values()) + 1
    
    # Limit entities for processing (for faster execution)
    entities_to_process = list(all_entities)[:max_entities]
    print(f"Processing first {len(entities_to_process)} entities (limited from {len(all_entities)} total)")
    
    # Group properties by entity (both incoming and outgoing)
    entity_properties = defaultdict(dict)
    for entity_id in entities_to_process:
        outgoing_props = get_entity_outgoing_properties(triples, entity_id, id_to_relation)
        incoming_props = get_entity_incoming_properties(triples, entity_id, id_to_relation)
        
        entity_properties[entity_id]['outgoing'] = outgoing_props
        entity_properties[entity_id]['incoming'] = incoming_props
        entity_properties[entity_id]['all'] = outgoing_props.union(incoming_props)
    
    new_triples = []
    triple_count = 0
    property_to_entity_id = {}
    
    # Process each entity and its properties
    for entity_id, props in entity_properties.items():
        if triple_count >= max_new_triples:
            break
        
        property_list = list(props['all'])
        if not property_list: 
            continue
        
        print(f"\nGetting recommendations for entity {entity_id} (has {len(property_list)} total properties)")
        recommendations = get_recommendations(property_list)
        filtered_recommendations = process_recommendations(recommendations, threshold=probability_threshold)
        filtered_recommendations = filtered_recommendations[:len(property_list)]
        
        # Create new triples for each recommendation
        for new_prop, probability in filtered_recommendations:
            if triple_count >= max_new_triples:
                break
            
            # Check if property has prefix and extract the actual property name
            is_incoming = False
            if new_prop.startswith("I:"):
                is_incoming = True
                prop_name = new_prop[2:]
            elif new_prop.startswith("O:"):
                prop_name = new_prop[2:]
            else:
                prop_name = new_prop
                
            # For FB15k237, we expect full paths in the actual property name
            if not prop_name.startswith('/'):
                print(f"Skipping non-path property {prop_name} for FB15k237 dataset")
                continue
            
            # Get the numeric relation ID for the property name (without prefix)
            if prop_name not in relation_to_id:
                print(f"Property not in known relations, skipping: {prop_name}")
                continue
            
            new_relation_id = relation_to_id[prop_name]
            
            # Get or create entity ID for this property
            if prop_name not in property_to_entity_id:
                property_to_entity_id[prop_name] = next_entity_id
                next_entity_id += 1
            
            # Create new triple with proper directionality based on prefix
            if is_incoming:
                new_triple = torch.tensor([property_to_entity_id[prop_name], new_relation_id, entity_id])
                print(f"Created incoming triple: ({property_to_entity_id[prop_name]}) --{prop_name}--> ({entity_id})")
            else:
                new_triple = torch.tensor([entity_id, new_relation_id, property_to_entity_id[prop_name]])
                print(f"Created outgoing triple: ({entity_id}) --{prop_name}--> ({property_to_entity_id[prop_name]})")
            
            new_triples.append(new_triple)
            triple_count += 1
    
    print(f"\nCreated {len(new_triples)} artificial triples")
    print(f"Final next_entity_id: {next_entity_id}")
    print(f"Number of unique property-specific entities: {len(property_to_entity_id)}")
    
    return new_triples, next_entity_id

def train_bidirectional_model_with_callbacks(
    output_dir, 
    dataset_name=None, 
    model_type=None, 
    embedding_dim=None, 
    max_epochs=None,
    probability_threshold=None,
    max_entities=1000000
):
    """
    Train a bidirectional ComplEx model with epoch-by-epoch metric logging.
    
    Args:
        output_dir: Directory to save the model and metrics
        dataset_name: Name of the dataset 
        model_type: Type of model (ComplEx)
        embedding_dim: Dimension of entity/relation embeddings
        max_epochs: Maximum number of training epochs
        probability_threshold: Threshold for recommendation filtering
        max_entities: Maximum number of entities to process for speed
    """
    print("\n=== Training Bidirectional ComplEx Model with Callbacks ===")
    
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
    
    # Create artificial triples
    probability_threshold = probability_threshold or get_config('probability_threshold')
    artificial_triples, next_entity_id = create_artificial_triples(
        dataset, 
        probability_threshold=probability_threshold,
        max_entities=max_entities
    )
    
    # Combine original and artificial triples
    if artificial_triples:
        print(f"\nCombining {len(dataset.training.mapped_triples)} original + {len(artificial_triples)} artificial triples")
        
        # Convert to labeled format for TriplesFactory
        id_to_entity = {v: k for k, v in dataset.training.entity_to_id.items()}
        id_to_relation = {v: k for k, v in dataset.training.relation_to_id.items()}
        
        # Convert original triples
        original_labeled_triples = []
        for triple in dataset.training.mapped_triples:
            head_id, rel_id, tail_id = triple.tolist()
            head_label = id_to_entity[head_id]
            rel_label = id_to_relation[rel_id]
            tail_label = id_to_entity[tail_id]
            original_labeled_triples.append((head_label, rel_label, tail_label))
        
        # Convert artificial triples (create new entity labels for artificial entities)
        artificial_labeled_triples = []
        for triple in artificial_triples:
            head_id, rel_id, tail_id = triple.tolist()
            
            # Handle head entity
            if head_id in id_to_entity:
                head_label = id_to_entity[head_id]
            else:
                head_label = f"artificial_entity_{head_id}"
                id_to_entity[head_id] = head_label
            
            # Handle relation (should always be original)
            rel_label = id_to_relation[rel_id]
            
            # Handle tail entity
            if tail_id in id_to_entity:
                tail_label = id_to_entity[tail_id]
            else:
                tail_label = f"artificial_entity_{tail_id}"
                id_to_entity[tail_id] = tail_label
            
            artificial_labeled_triples.append((head_label, rel_label, tail_label))
        
        # Combine all labeled triples
        all_labeled_triples = original_labeled_triples + artificial_labeled_triples
        
        # Create extended TriplesFactory
        extended_training = TriplesFactory.from_labeled_triples(
            np.array(all_labeled_triples, dtype=str),
            create_inverse_triples=True
        )
        
        print(f"Extended training triples (with inverses): {extended_training.num_triples}")
    else:
        print("No artificial triples created, using original dataset")
        extended_training = dataset.training
    
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
            'p': 3,  # This makes it equivalent to N3 regularization
        },
    }
    
    # Use a unique checkpoint path to avoid conflicts
    unique_checkpoint_name = f"bidirectional_checkpoint_{int(time.time())}.pt"
    custom_checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(custom_checkpoint_dir, exist_ok=True)
    
    # Create callback for metric logging
    callback = MetricLoggerCallback(output_dir, dataset.validation, extended_training, dataset.testing, get_config('model.batch_size_eval'))
    
    training_kwargs = {
        'num_epochs': max_epochs,
        'batch_size': get_config('model.batch_size_train'),
        'use_tqdm': True,
        'use_tqdm_batch': True,
        'checkpoint_name': unique_checkpoint_name,
        'checkpoint_directory': custom_checkpoint_dir,
        'checkpoint_frequency': 5,
        'tqdm_kwargs': {
            'mininterval': 2.0,
            'miniters': 5,
        },
        'callbacks': [callback],
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
    print(f"\nTraining {model_type} model with embedding_dim={embedding_dim} on extended dataset")
    result = pipeline(
        training=extended_training,
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
        device=device,
        stopper='early',
        stopper_kwargs=stopper_kwargs
    )
    
    # Save the model
    model_file = osp.join(output_dir, 'bidirectional_trained_model.pkl')
    torch.save(result.model, model_file)
    print(f"Saved trained model to {model_file}")
    
    # Save final metrics
    metrics_file = osp.join(output_dir, 'final_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Model: {model_type} (Bidirectional)\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Embedding Dim: {embedding_dim}\n")
        f.write(f"Probability Threshold: {probability_threshold}\n")
        f.write(f"Max Entities Processed: {max_entities}\n")
        f.write(f"Artificial Triples Created: {len(artificial_triples) if artificial_triples else 0}\n")
        f.write(f"Epochs trained: {len(callback.logs)}\n")
        f.write(f"Final Test Metrics:\n")
        
        # Format metrics
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
    parser = argparse.ArgumentParser(description="Train a bidirectional ComplEx model with epoch metrics")
    parser.add_argument("--output-dir", type=str, default="models/bidirectional_complex_with_callbacks",
                        help="Output directory for the trained model")
    parser.add_argument("--dataset", type=str, choices=["FB15k237", "CoDExSmall"],
                        help=f"Dataset to use (default: {get_config('dataset.name')})")
    parser.add_argument("--model", type=str,
                        help=f"Model type (default: {get_config('model.type')})")
    parser.add_argument("--embedding-dim", type=int,
                        help=f"Embedding dimension (default: {get_config('model.embedding_dim')})")
    parser.add_argument("--max-epochs", type=int,
                        help=f"Maximum training epochs (default: {get_config('model.max_epochs')})")
    parser.add_argument("--probability-threshold", type=float,
                        help=f"Probability threshold for recommendations (default: {get_config('probability_threshold')})")
    parser.add_argument("--max-entities", type=int, default=100,
                        help="Maximum number of entities to process for artificial triples (default: 100)")
    
    args = parser.parse_args()
    
    train_bidirectional_model_with_callbacks(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        model_type=args.model,
        embedding_dim=args.embedding_dim,
        max_epochs=args.max_epochs,
        probability_threshold=args.probability_threshold,
        max_entities=args.max_entities
    )

if __name__ == "__main__":
    main() 