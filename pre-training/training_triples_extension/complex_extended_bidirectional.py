#!/usr/bin/env python
"""
Script to train a model with extended triples from the recommender using ComplEx.
Adds both incoming and outgoing relations.
Train with FB15k237_bidirectional schema tree!!!
"""

import os
import os.path as osp
import logging
import torch
import numpy as np
import argparse
import requests
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import time
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config(key, default=None):
    """Get configuration from environment or use default."""
    configs = {
        'dataset.name': 'FB15k237',  
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
        'api.url': 'http://localhost:8080/recommender',
        'api.timeout': 30,  # API request timeout in seconds
        'probability_threshold': 0.5,  # Probability threshold for recommendations
        'max_recommendations': 10,  
        'max_new_triples': 10000000,  
        'sampling_rate': 0.0,  # Rate at which to sample from new triples (0.0 = use all)
        'model.random_seed': 42,  #
    }
    return configs.get(key, default)

def get_entity_outgoing_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """
    Get all outgoing properties (relations) where the entity is the head.
    
    Args:
        triples: Tensor of triples [head, relation, tail]
        entity_id: ID of the entity
        id_to_relation: Mapping from relation IDs to their labels
        
    Returns:
        Set of relation labels where the entity is the head with "O:" prefix
    """
    # Get all triples where entity is the head
    head_triples = triples[triples[:, 0] == entity_id]
    
    # Get relation labels with "O:" prefix
    return {f"O:{id_to_relation[rel_id.item()]}" for rel_id in head_triples[:, 1]}

def get_entity_incoming_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """
    Get all incoming properties (relations) where the entity is the tail.
    
    Args:
        triples: Tensor of triples [head, relation, tail]
        entity_id: ID of the entity
        id_to_relation: Mapping from relation IDs to their labels
        
    Returns:
        Set of relation labels where the entity is the tail with "I:" prefix
    """
    # Get all triples where entity is the tail
    tail_triples = triples[triples[:, 2] == entity_id]
    
    # Get relation labels with "I:" prefix
    return {f"I:{id_to_relation[rel_id.item()]}" for rel_id in tail_triples[:, 1]}

def get_recommendations(properties: List[str], api_url: str = None) -> List[Dict[str, Any]]:
    """
    Get property recommendations from the API.
    
    Args:
        properties: List of property names/IDs with "I:" or "O:" prefixes
        api_url: URL of the recommendation API
        
    Returns:
        List of recommended properties with their probabilities
    """
    api_url = api_url or get_config('api.url')
    api_timeout = get_config('api.timeout')
    
    
    
    try:
        # Prepare request data
        data = {
            "properties": properties,
            "types": []  # Empty list as we're not using types
        }
        
        print(f"Sending request to {api_url} with {len(properties)} properties:")
        print(f"First few properties: {properties[:5]}")
        
        # Make API request
        response = requests.post(
            api_url,
            json=data,
            timeout=api_timeout
        )
        response.raise_for_status() # Check the HTTP status
        
        # Parse response - get recommendations from the response
        recommendations = response.json().get("recommendations", [])
        logger.info(f"Received {len(recommendations)} recommendations")
        print(f"Received {len(recommendations)} recommendations")
        
        # Print first few recommendations for debugging
        if recommendations:
            print(f"First 3 recommendations: {recommendations[:3]}")
            # Check for duplicates in recommendations
            prop_set = set()
            duplicates = []
            for rec in recommendations:
                if rec['property'] in prop_set:
                    duplicates.append(rec['property'])
                else:
                    prop_set.add(rec['property'])
            if duplicates:
                print(f"Found {len(duplicates)} duplicate properties in recommendations")
                print(f"Example duplicates: {duplicates[:3]}")
        
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
    """
    Process and filter recommendations based on probability threshold.
    
    Args:
        recommendations: List of recommendations from API
        threshold: Minimum probability threshold
        max_recommendations: Maximum number of recommendations to return
        
    Returns:
        List of tuples (property_id, probability) for filtered recommendations
    """
    threshold = threshold or get_config('probability_threshold')
    max_recommendations = max_recommendations or get_config('max_recommendations')
    
    print(f"Processing {len(recommendations)} recommendations with threshold={threshold}")
    
    # Check for any abnormalities in the data structure
    if recommendations and 'property' not in recommendations[0]:
        print(f"WARNING: Unexpected recommendation format. Keys: {recommendations[0].keys()}")
    
    # Filter and sort recommendations
    filtered_recommendations = [
        (rec['property'], rec['probability'])
        for rec in recommendations
        if rec['probability'] >= threshold
    ]
    
    print(f"After threshold filtering: {len(filtered_recommendations)} recommendations remain")
    
    # Check probability distribution
    if filtered_recommendations:
        probs = [p[1] for p in filtered_recommendations]
        print(f"Probability range: min={min(probs):.4f}, max={max(probs):.4f}, avg={sum(probs)/len(probs):.4f}")
    
    # Sort by probability in descending order
    filtered_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N recommendations
    result = filtered_recommendations[:max_recommendations]
    print(f"Returning top {len(result)} recommendations (max_recommendations={max_recommendations})")
    
    return result

def create_artificial_triples(
    dataset,
    probability_threshold: float = None
) -> Tuple[List[torch.Tensor], int]:
    """
    Create artificial triples based on recommendations using both incoming and outgoing properties.
    
    Args:
        dataset: PyKEEN dataset
        probability_threshold: Threshold for recommendation filtering
        
    Returns:
        Tuple of (list of new triples, next entity ID)
    """
    probability_threshold = probability_threshold or get_config('probability_threshold')
    max_new_triples = get_config('max_new_triples')
    
    print("\n=== Creating Artificial Triples (Bidirectional) ===")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total number of entities: {len(dataset.entity_to_id)}")
    print(f"Total number of relations: {len(dataset.relation_to_id)}")
    print(f"Total number of triples in training set: {len(dataset.training.mapped_triples)}")
    
    # Get all unique entities (both head and tail positions)
    triples = dataset.training.mapped_triples
    all_entities = set(triples[:, 0].tolist()).union(set(triples[:, 2].tolist()))
    print(f"Number of unique entities (head + tail): {len(all_entities)}")
    
    # Create id_to_relation mapping
    id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}
    relation_to_id = dataset.relation_to_id.copy()  # Start with existing dataset mappings
    
    # Set next entity ID
    next_entity_id = max(dataset.entity_to_id.values()) + 1
    print(f"Initial next_entity_id: {next_entity_id}")
    
    # Group properties by entity (both incoming and outgoing)
    entity_properties = defaultdict(dict)
    for entity_id in all_entities:
        # Get outgoing properties (entity as head)
        outgoing_props = get_entity_outgoing_properties(triples, entity_id, id_to_relation)
        
        # Get incoming properties (entity as tail)
        incoming_props = get_entity_incoming_properties(triples, entity_id, id_to_relation)
        
        # Store both property types
        entity_properties[entity_id]['outgoing'] = outgoing_props
        entity_properties[entity_id]['incoming'] = incoming_props
        entity_properties[entity_id]['all'] = outgoing_props.union(incoming_props)
    
    print(f"\nNumber of entities with properties: {len(entity_properties)}")
    
    # Count distributions
    outgoing_counts = defaultdict(int)
    incoming_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    for entity_id, props in entity_properties.items():
        outgoing_count = len(props['outgoing'])
        incoming_count = len(props['incoming'])
        total_count = len(props['all'])
        
        outgoing_counts[outgoing_count] += 1
        incoming_counts[incoming_count] += 1
        total_counts[total_count] += 1
    
    print("\nOutgoing property distribution:")
    for count in sorted(outgoing_counts.keys()):
        print(f"Entities with {count} outgoing properties: {outgoing_counts[count]}")
    
    print("\nIncoming property distribution:")
    for count in sorted(incoming_counts.keys()):
        print(f"Entities with {count} incoming properties: {incoming_counts[count]}")
    
    print("\nTotal property distribution:")
    for count in sorted(total_counts.keys()):
        print(f"Entities with {count} total properties: {total_counts[count]}")
    
    new_triples = []
    triple_count = 0
    
    # Dictionary to store property-specific entity IDs
    property_to_entity_id = {}
    
    # Process each entity and its properties
    for entity_id, props in entity_properties.items():
        if triple_count >= max_new_triples:
            break
        
        # Get recommendations for all properties of this entity (both incoming and outgoing)
        property_list = list(props['all'])
        if not property_list: 
            continue  # Skip entities with no properties
        
        print(f"\nGetting recommendations for entity {entity_id} (has {len(property_list)} total properties)")
        recommendations = get_recommendations(property_list)
        filtered_recommendations = process_recommendations(recommendations, threshold=probability_threshold)
        # Limit recommendations to the number of original properties (both incoming and outgoing)-
        # you can skip that if no needed
        filtered_recommendations = filtered_recommendations[:len(property_list)]
        
        # Create new triples for each recommendation
        for new_prop, probability in filtered_recommendations:
            if triple_count >= max_new_triples:
                break
            
            # Check if property has prefix and extract the actual property name
            is_incoming = False
            if new_prop.startswith("I:"):
                is_incoming = True
                prop_name = new_prop[2:]  # Remove "I:" prefix
            elif new_prop.startswith("O:"):
                prop_name = new_prop[2:]  # Remove "O:" prefix
            else:
                prop_name = new_prop  # No prefix, use as is
                
            # Skip recommendations that don't match our dataset's format
            dataset_name = get_config('dataset.name')
            is_fb15k = dataset_name == "FB15k237"

            
            if is_fb15k:
                # For FB15k237, we expect full paths in the actual property name
                if not prop_name.startswith('/'):
                    print(f"Skipping non-path property {prop_name} for FB15k237 dataset")
                    continue
            else:
                # For CoDEx, we expect P-numbers in the actual property name
                if not prop_name.startswith('P'):
                    print(f"Skipping non-P-number property {prop_name} for CoDEx dataset")
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
                # For incoming properties: property → relation → entity
                # (entity is tail)
                new_triple = torch.tensor([property_to_entity_id[prop_name], new_relation_id, entity_id])
                print(f"Created incoming triple: ({property_to_entity_id[prop_name]}) --{prop_name}--> ({entity_id})")
            else:
                # For outgoing properties: entity → relation → property
                # (entity is head)
                new_triple = torch.tensor([entity_id, new_relation_id, property_to_entity_id[prop_name]])
                print(f"Created outgoing triple: ({entity_id}) --{prop_name}--> ({property_to_entity_id[prop_name]})")
            
            new_triples.append(new_triple)
            triple_count += 1
    
    print(f"\nCreated {len(new_triples)} artificial triples")
    print(f"Final next_entity_id: {next_entity_id}")
    print(f"Number of unique property-specific entities: {len(property_to_entity_id)}")
    
    return new_triples, next_entity_id

def sample_triples(extended_triples: torch.Tensor, sampling_rate: float = None) -> torch.Tensor:
    """
    Randomly sample triples from the extended dataset to create more diverse connections.
    
    Args:
        extended_triples: Tensor of shape (n, 3) containing the extended triples
        sampling_rate: Float between 0 and 1, indicating what fraction of triples to keep
    
    Returns:
        Sampled triples tensor
    """
    sampling_rate = sampling_rate or get_config('sampling_rate')
    
    # If sampling rate is 0, return all triples
    if sampling_rate <= 0:
        return extended_triples
    
    # Calculate number of triples to keep
    n_triples = len(extended_triples)
    n_keep = int(n_triples * (1 - sampling_rate))
    
    # Get random indices to keep
    indices = torch.randperm(n_triples)[:n_keep]
    
    # Return sampled triples
    return extended_triples[indices]

def train_extended_model(
    output_dir, 
    baseline_model_dir=None,
    dataset_name=None, 
    model_type=None, 
    embedding_dim=None, 
    max_epochs=None,
    probability_threshold=None,
    sampling_rate=None
):
    """
    Train a model with extended triples from the recommender using bidirectional properties.
    
    Args:
        output_dir: Directory to save the model and triples
        baseline_model_dir: Directory with the baseline model (optional)
        dataset_name: Name of the dataset (FB15k237 or CoDExSmall)
        model_type: Type of model (ComplEx)
        embedding_dim: Dimension of entity/relation embeddings
        max_epochs: Maximum number of training epochs
        probability_threshold: Threshold for recommendation filtering
        sampling_rate: Rate at which to sample from new triples
    """
    print("\n=== Training Extended ComplEx Model with Bidirectional Properties ===")
    
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
    new_triples, next_entity_id = create_artificial_triples(
        dataset,
        probability_threshold=probability_threshold
    )
    
    
    # Sample triples if needed
    sampling_rate = sampling_rate or get_config('sampling_rate')
    if sampling_rate > 0:
        extended_triples_tensor = torch.stack(new_triples)
        sampled_triples = sample_triples(extended_triples_tensor, sampling_rate)
    else:
        sampled_triples = torch.stack(new_triples)
    
    print(f"Using {len(sampled_triples)} artificial triples after sampling")
    
    # Create new entity and relation mappings that preserve the original indices
    extended_entity_to_id = dataset.entity_to_id.copy()
    extended_relation_to_id = dataset.relation_to_id.copy()
    
    # Add new entities to the mapping
    for i in range(len(dataset.entity_to_id), next_entity_id):
        extended_entity_to_id[f"NEW_{i}"] = i
    
    # Combine datasets
    combined_triples = torch.cat([
        dataset.training.mapped_triples,
        sampled_triples
    ])
    
    # Create combined factory
    combined_factory = TriplesFactory(
        mapped_triples=combined_triples,
        entity_to_id=extended_entity_to_id,
        relation_to_id=extended_relation_to_id,
    )
    
    print("\nExtended Dataset Statistics:")
    print(f"Original training triples: {len(dataset.training.mapped_triples):,}")
    print(f"New triples added: {len(sampled_triples):,}")
    print(f"Total triples: {len(combined_triples):,}")
    print(f"Original entities: {len(dataset.entity_to_id):,}")
    print(f"New entities: {len(extended_entity_to_id) - len(dataset.entity_to_id):,}")
    print(f"Total entities: {len(extended_entity_to_id):,}")
    
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
    
    # Set up checkpoint directory
    custom_checkpoint_dir = osp.join(output_dir, 'checkpoints')
    os.makedirs(custom_checkpoint_dir, exist_ok=True)
    
    # Create a unique checkpoint name to avoid conflicts
    unique_checkpoint_name = f"extended_complex_{int(time.time())}"
    
    # Set up training configuration
    training_kwargs = {
        'num_epochs': max_epochs or get_config('max_epochs'),
        'batch_size': get_config('model.batch_size_train'),
        'use_tqdm': True,  # Show progress bars
        'use_tqdm_batch': True,  # Show batch progress
        'checkpoint_name': unique_checkpoint_name,  # Unique name
        'checkpoint_directory': custom_checkpoint_dir,  # Custom directory
        'checkpoint_frequency': 10,  # Save every 10 epochs
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
    
    merged_file = os.path.join(output_dir, "merged_entities_metadata.tsv")
    with open(merged_file, 'w', encoding='utf-8') as f:
        f.write("entity_id\tentity_label\n")
        for entity, entity_id in extended_entity_to_id.items():
            f.write(f"{entity_id}\t{entity}\n")
    print(f"Saved merged entities metadata to {merged_file}")
    
    # Set up evaluation kwargs
    evaluation_kwargs = {
        'batch_size': get_config('model.batch_size_eval'),
    }
    
    # Print evaluation configuration for diagnostics
    print("\n=== EVALUATION CONFIGURATION ===")
    #print(f"Evaluator: {pipeline_kwargs.get('evaluator')}")
    print(f"Batch size: {evaluation_kwargs.get('batch_size')}")
    print(f"Test set size: {len(dataset.testing.mapped_triples):,}")
    
    # Verify the test set is not empty
    if len(dataset.testing.mapped_triples) == 0:
        print("WARNING: Test set is empty! This will result in NaN metrics.")
        print("Please check your dataset splitting configuration.")
    else:
        # Check for common evaluation edge cases
        num_entities = len(extended_entity_to_id)
        test_heads = set(dataset.testing.mapped_triples[:, 0].tolist())
        test_tails = set(dataset.testing.mapped_triples[:, 2].tolist())
        
        # Check if test set contains entities not in training
        print(f"Entities used as heads in test set: {len(test_heads):,}")
        print(f"Entities used as tails in test set: {len(test_tails):,}")
        
        if len(test_heads) < len(test_heads.union(test_tails)) / 10:
            print("WARNING: Very few entities used as heads in test set. This may cause evaluation issues.")
            
        if len(test_tails) < len(test_heads.union(test_tails)) / 10:
            print("WARNING: Very few entities used as tails in test set. This may cause evaluation issues.")
    
    # Train the model
    print(f"\nTraining {model_type} model with embedding_dim={embedding_dim} on extended dataset")
    result = pipeline(
        training=combined_factory,
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
        combined_factory.to_path_binary(training_file)
        print(f"Saved training triples in binary format to {training_file}")
    except Exception as e:
        print(f"Could not save in binary format, saving as CSV: {e}")
        # Fall back to CSV format
        csv_file = osp.join(output_dir, 'training_triples.csv')
        combined_factory.to_path(csv_file)
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
        f.write(f"Dataset: {dataset_name} + {len(sampled_triples)} artificial triples (bidirectional)\n")
        f.write(f"Embedding Dim: {embedding_dim}\n")
        f.write(f"Probability Threshold: {probability_threshold or get_config('probability_threshold')}\n")
        f.write(f"Sampling Rate: {sampling_rate or get_config('sampling_rate')}\n")
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
    print("\n=== EXTENDED COMPLEX MODEL EVALUATION METRICS ===")
    metrics = result.metric_results.to_dict()
    
    # Handle hierarchical metrics structure - use 'realistic' evaluation by default
    if 'realistic' in metrics:
        metrics = metrics['realistic']
        print("Using 'realistic' evaluation metrics")
    elif 'both' in metrics:
        metrics = metrics['both']  
        print("Using 'both' evaluation metrics")
    
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
    
    print("\nExtended ComplEx model training complete!")
    
    # Print final summary with separator lines
    print("\n" + "="*80)
    print("=== FINAL EXTENDED MODEL SUMMARY ===")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Original training triples: {len(dataset.training.mapped_triples):,}")
    print(f"Artificial triples added: {len(sampled_triples):,}")
    print(f"Total training triples: {len(combined_triples):,}")
    print(f"Triple increase: {(len(sampled_triples) / len(dataset.training.mapped_triples) * 100):.2f}%")
    print(f"Original entities: {len(dataset.entity_to_id):,}")
    print(f"New entities added: {len(extended_entity_to_id) - len(dataset.entity_to_id):,}")
    print(f"Total entities: {len(extended_entity_to_id):,}")
    
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
    
    # Return model, output directory and metrics
    return result.model, output_dir, metrics

def main():
    """Parse command line arguments and train the model."""
    parser = argparse.ArgumentParser(description="Train a model with extended triples using ComplEx")
    
    # Define a function to get default values for arguments
    def get_default(key):
        return get_config(key)
    
    parser.add_argument("--output-dir", type=str, default="models/extended_complex",
                        help="Output directory for the trained model (default: models/extended_complex)")
    parser.add_argument("--baseline-model-dir", type=str,
                        help="Directory with the baseline model (optional)")
    parser.add_argument("--dataset", type=str, choices=["FB15k237", "CoDExSmall"],
                        help=f"Dataset to use (default: {get_default('dataset.name')})")
    parser.add_argument("--model", type=str,
                        help=f"Model type (default: {get_default('model.type')})")
    parser.add_argument("--embedding-dim", type=int,
                        help=f"Embedding dimension (default: {get_default('model.embedding_dim')})")
    parser.add_argument("--max-epochs", type=int,
                        help=f"Maximum training epochs (default: {get_default('model.max_epochs')})")
    parser.add_argument("--probability-threshold", type=float,
                        help=f"Probability threshold for recommendations (default: {get_default('probability_threshold')})")
    parser.add_argument("--sampling-rate", type=float,
                        help=f"Sampling rate for new triples (default: {get_default('sampling_rate')})")
    parser.add_argument("--api-url", type=str,
                        help=f"URL of the recommendation API (default: {get_default('api.url')})")
    
    args = parser.parse_args()
    
    # Update API URL if provided
    if args.api_url:
        global get_config
        configs = {}
        configs['api.url'] = args.api_url
        # Monkey patch the get_config function to use our updated config
        original_get_config = get_config
        def patched_get_config(key, default=None):
            if key in configs:
                return configs[key]
            return original_get_config(key, default)
        get_config = patched_get_config
    
    train_extended_model(
        output_dir=args.output_dir,
        baseline_model_dir=args.baseline_model_dir,
        dataset_name=args.dataset,
        model_type=args.model,
        embedding_dim=args.embedding_dim,
        max_epochs=args.max_epochs,
        probability_threshold=args.probability_threshold,
        sampling_rate=args.sampling_rate
    )


if __name__ == "__main__":
    main() 