"""
ComplEx Model Training with Weighted Query-Aware Loss using PyKEEN Pipeline

This script trains a ComplEx model using PyKEEN pipeline with weighted training where each triple
is weighted based on Leave-One-Out scores computed from a recommender API.
It incorporates the I/O triples mechanism from complex_extended_bidirectional.
"""

import os
import sys
import json
import logging
import pickle
import requests
import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict

import torch
import wandb
import numpy as np
from pykeen.datasets import FB15k237
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator

# Import our custom modules
from weighted_training_loop import WeightedSLCWATrainingLoop
from leave_one_out_scoring import create_leave_one_out_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_config(key, default=None):
    """Get configuration values."""
    configs = {
        'dataset.name': 'FB15k237',
        'model.type': 'ComplEx',
        'model.embedding_dim': 1000,
        'model.max_epochs': 1,
        'model.batch_size_train': 1000,
        'model.batch_size_eval': 256,
        'model.learning_rate': 0.1,
        'sampling.negative_o': 1000,
        'model.dropout': 0.5,
        'model.regularize_weight': 0.05,
        'model.relation_dropout': 0.22684140529516872,
        'model.relation_regularize_weight': 8.266519211068944e-14,
        'api.url': 'http://localhost:8080/recommender',
        'api.timeout': 30,
        'probability_threshold': 0.25,
        'max_recommendations': 10,
        'max_new_triples': 100000,
        'sampling_rate': 0.0,
        'model.random_seed': 42,
        'model.evaluator': 'rankbased',
    }
    return configs.get(key, default)


def setup_wandb(config: Dict[str, Any]) -> None:
    """Initialize Weights & Biases logging."""
    wandb.init(
        project="complex-weighted-training-pipeline",
        config=config,
        name=f"weighted_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def get_entity_outgoing_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """
    Get all outgoing properties (relations) where the entity is the head.
    """
    head_triples = triples[triples[:, 0] == entity_id]
    return {f"O:{id_to_relation[rel_id.item()]}" for rel_id in head_triples[:, 1]}


def get_entity_incoming_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """
    Get all incoming properties (relations) where the entity is the tail.
    """
    tail_triples = triples[triples[:, 2] == entity_id]
    return {f"I:{id_to_relation[rel_id.item()]}" for rel_id in tail_triples[:, 1]}


def get_recommendations(properties: List[str], api_url: str = None) -> List[Dict[str, Any]]:
    """Get recommendations from the API."""
    api_url = api_url or get_config('api.url')
    timeout = get_config('api.timeout')
    
    # Use the correct API request structure (matching working examples)
    request_data = {
        "properties": properties,
        "types": []  # Empty list as we're not using types
    }
    
    try:
        response = requests.post(
            api_url,
            json=request_data,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        
        # Parse response - get recommendations from the response
        recommendations = response.json().get("recommendations", [])
        logger.info(f"Received {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.warning(f"API request failed for properties {properties}: {e}")
        return []


def process_recommendations(
    recommendations: List[Dict[str, Any]],
    threshold: float = None,
    max_recommendations: int = None
) -> List[Tuple[str, float]]:
    """Process and filter recommendations."""
    threshold = threshold or get_config('probability_threshold')
    max_recommendations = max_recommendations or get_config('max_recommendations')
    
    logger.info(f"Processing {len(recommendations)} recommendations with threshold={threshold}")
    
    # Check for any abnormalities in the data structure
    if recommendations and 'property' not in recommendations[0]:
        logger.warning(f"WARNING: Unexpected recommendation format. Keys: {recommendations[0].keys()}")
    
    # Use the correct key 'property' instead of 'recommendation'
    filtered = [
        (rec["property"], rec["probability"])
        for rec in recommendations
        if rec["probability"] >= threshold
    ]
    
    logger.info(f"After threshold filtering: {len(filtered)} recommendations remain")
    
    # Sort by probability in descending order
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    return filtered[:max_recommendations]


def create_artificial_triples(
    dataset,
    probability_threshold: float = None
) -> Tuple[List[torch.Tensor], int]:
    """
    Create artificial triples based on recommendations using both incoming and outgoing properties.
    """
    probability_threshold = probability_threshold or get_config('probability_threshold')
    max_new_triples = get_config('max_new_triples')
    
    logger.info("\n=== Creating Artificial Triples (Bidirectional) ===")
    
    # Get all unique entities
    triples = dataset.training.mapped_triples
    all_entities = set(triples[:, 0].tolist()).union(set(triples[:, 2].tolist()))
    logger.info(f"Number of unique entities: {len(all_entities)}")
    
    # Create mappings
    id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}
    relation_to_id = dataset.relation_to_id.copy()
    
    # Set next entity ID
    next_entity_id = max(dataset.entity_to_id.values()) + 1
    logger.info(f"Initial next_entity_id: {next_entity_id}")
    
    # Group properties by entity
    entity_properties = defaultdict(dict)
    
    for entity_id in all_entities:
        # Get incoming and outgoing properties
        incoming_props = get_entity_incoming_properties(triples, entity_id, id_to_relation)
        outgoing_props = get_entity_outgoing_properties(triples, entity_id, id_to_relation)
        
        all_props = incoming_props.union(outgoing_props)
        if all_props:
            entity_properties[entity_id] = {
                'properties': list(all_props),
                'incoming': incoming_props,
                'outgoing': outgoing_props
            }
    
    logger.info(f"Entities with properties: {len(entity_properties)}")
    
    # Get recommendations and create triples
    new_triples = []
    triple_count = 0
    property_to_entity_id = {}
    
    for entity_id, prop_data in entity_properties.items():
        if triple_count >= max_new_triples:
            break
            
        properties = prop_data['properties']
        if not properties:
            continue
        
        try:
            recommendations = get_recommendations(properties)
            filtered_recommendations = process_recommendations(
                recommendations, 
                threshold=probability_threshold
            )
            
            for prop_name, probability in filtered_recommendations:
                if triple_count >= max_new_triples:
                    break
                
                # Check if this is an incoming or outgoing property
                is_incoming = prop_name in prop_data['incoming']
                
                # Extract original relation name
                if prop_name.startswith("I:") or prop_name.startswith("O:"):
                    original_relation = prop_name[2:]  # Remove "I:" or "O:" prefix
                else:
                    original_relation = prop_name
                
                # Get or create relation ID
                if original_relation not in relation_to_id:
                    new_relation_id = len(relation_to_id)
                    relation_to_id[original_relation] = new_relation_id
                else:
                    new_relation_id = relation_to_id[original_relation]
                
                # Get or create entity ID for property
                if prop_name not in property_to_entity_id:
                    property_to_entity_id[prop_name] = next_entity_id
                    next_entity_id += 1
                
                # Create new triple with proper directionality
                if is_incoming:
                    # For incoming properties: property → relation → entity
                    new_triple = torch.tensor([property_to_entity_id[prop_name], new_relation_id, entity_id])
                else:
                    # For outgoing properties: entity → relation → property
                    new_triple = torch.tensor([entity_id, new_relation_id, property_to_entity_id[prop_name]])
                
                new_triples.append(new_triple)
                triple_count += 1
                
        except Exception as e:
            logger.warning(f"Failed to get recommendations for entity {entity_id}: {e}")
            continue
    
    logger.info(f"Created {len(new_triples)} artificial triples")
    logger.info(f"Final next_entity_id: {next_entity_id}")
    
    return new_triples, next_entity_id


def convert_triples_to_string_format(triples_factory) -> List[Tuple[str, str, str]]:
    """Convert PyKEEN triples factory to string format for API calls."""
    logger.info("Converting triples to string format...")
    
    string_triples = []
    mapped_triples = triples_factory.mapped_triples
    
    for triple in mapped_triples:
        head_id, relation_id, tail_id = triple.tolist()
        
        head_name = triples_factory.entity_id_to_label[head_id]
        relation_name = triples_factory.relation_id_to_label[relation_id]
        tail_name = triples_factory.entity_id_to_label[tail_id]
        
        string_triples.append((head_name, relation_name, tail_name))
    
    logger.info(f"Converted {len(string_triples)} triples to string format")
    return string_triples


def convert_string_weights_to_id_weights(
    string_weights: Dict[Tuple[str, str, str], float],
    triples_factory
) -> Dict[Tuple[int, int, int], float]:
    """Convert string-based triple weights to ID-based weights for PyKEEN."""
    logger.info("Converting string weights to ID weights...")
    
    id_weights = {}
    conversion_errors = 0
    
    for (head_name, relation_name, tail_name), weight in string_weights.items():
        try:
            head_id = triples_factory.entity_to_id[head_name]
            relation_id = triples_factory.relation_to_id[relation_name]
            tail_id = triples_factory.entity_to_id[tail_name]
            
            id_weights[(head_id, relation_id, tail_id)] = weight
            
        except KeyError as e:
            conversion_errors += 1
            if conversion_errors <= 10:  # Log first 10 errors
                logger.warning(f"Failed to convert triple ({head_name}, {relation_name}, {tail_name}): {e}")
    
    if conversion_errors > 0:
        logger.warning(f"Failed to convert {conversion_errors} triples to ID format")
    
    logger.info(f"Converted {len(id_weights)} string weights to ID weights")
    return id_weights


def save_weights_to_file(weights: Dict, filename: str) -> None:
    """Save weights dictionary to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    logger.info(f"Saved weights to {filename}")


def load_weights_from_file(filename: str) -> Dict:
    """Load weights dictionary from a pickle file."""
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    logger.info(f"Loaded weights from {filename}")
    return weights


def compute_and_save_triple_weights(
    triples_factory,
    config: Dict[str, Any],
    force_recompute: bool = False
) -> Dict[Tuple[int, int, int], float]:
    """Compute or load triple weights using Leave-One-Out scoring."""
    weights_filename = f"triple_weights_{config['dataset']}_averaged.pkl"
    
    # Check if weights file exists and we're not forcing recomputation
    if os.path.exists(weights_filename) and not force_recompute:
        logger.info(f"Loading existing weights from {weights_filename}")
        return load_weights_from_file(weights_filename)
    
    logger.info("Computing triple weights using Leave-One-Out scoring...")
    
    # Convert triples to string format
    string_triples = convert_triples_to_string_format(triples_factory)
    
    # Create Leave-One-Out scorer
    scorer = create_leave_one_out_scorer(
        api_url=config["api_url"],
        max_retries=config.get("api_max_retries", 3),
        retry_delay=config.get("api_retry_delay", 1.0),
        timeout=config.get("api_timeout", 30.0)
    )
    
    # Score all triples with averaging
    string_weights = scorer.score_all_triples(
        triples=string_triples,
        max_entities_to_score=config.get("max_entities_to_score", None),
        use_averaging=True
    )
    
    # Log scoring statistics
    scores = list(string_weights.values())
    logger.info(f"Weight statistics: min={min(scores):.4f}, max={max(scores):.4f}, "
                f"mean={sum(scores)/len(scores):.4f}")
    
    # Convert to ID-based weights
    id_weights = convert_string_weights_to_id_weights(string_weights, triples_factory)
    
    # Save weights for future use
    save_weights_to_file(id_weights, weights_filename)
    
    return id_weights


def train_weighted_complex_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a ComplEx model with weighted query-aware loss using PyKEEN pipeline.
    """
    logger.info("Starting weighted ComplEx model training with pipeline...")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Setup wandb if enabled
    if config.get("use_wandb", False):
        setup_wandb(config)
    
    # Load dataset
    logger.info(f"Loading {config['dataset']} dataset...")
    if config["dataset"] == "FB15k237":
        dataset = FB15k237()
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    
    # Create artificial triples (I/O mechanism)
    if config.get("create_artificial_triples", True):
        logger.info("Creating artificial triples using I/O mechanism...")
        new_triples, next_entity_id = create_artificial_triples(
            dataset,
            probability_threshold=config.get("probability_threshold", 0.25)
        )
        
        if new_triples:
            # Create extended entity mappings
            extended_entity_to_id = dataset.entity_to_id.copy()
            extended_relation_to_id = dataset.relation_to_id.copy()
            
            # Add new entities to the mapping
            for i in range(len(dataset.entity_to_id), next_entity_id):
                extended_entity_to_id[f"NEW_{i}"] = i
            
            # Combine datasets
            sampled_triples = torch.stack(new_triples)
            combined_triples = torch.cat([
                dataset.training.mapped_triples,
                sampled_triples
            ])
            
            # Create combined factory
            training_factory = TriplesFactory(
                mapped_triples=combined_triples,
                entity_to_id=extended_entity_to_id,
                relation_to_id=extended_relation_to_id,
            )
            
            logger.info(f"Extended training triples: {len(combined_triples):,}")
            logger.info(f"Artificial triples added: {len(sampled_triples):,}")
        else:
            logger.info("No artificial triples created, using original dataset")
            training_factory = dataset.training
    else:
        logger.info("Using original training dataset (no artificial triples)")
        training_factory = dataset.training
    
    # Compute triple weights
    triple_weights = compute_and_save_triple_weights(
        training_factory, 
        config,
        force_recompute=config.get("force_recompute_weights", False)
    )
    
    logger.info(f"Computed weights for {len(triple_weights)} triples")
    
    # Setup model parameters (matching complex_extended_bidirectional)
    model_kwargs = {
        'embedding_dim': config["embedding_dim"],
        'regularizer': 'LpRegularizer',
        'regularizer_kwargs': {
            'weight': config.get("regularize_weight", 0.05),
            'p': 3,
        },
        'random_seed': config.get("random_seed", 42)
    }
    
    training_kwargs = {
        'num_epochs': config["epochs"],
        'batch_size': config["batch_size"],
        'use_tqdm': True,
        'use_tqdm_batch': True,
    }
    
    optimizer_kwargs = {'lr': config["learning_rate"]}
    
    evaluation_kwargs = {
        'batch_size': config["eval_batch_size"],
        'use_tqdm': True
    }
    
    lr_scheduler_kwargs = {'gamma': 0.95}
    
    stopper_kwargs = {
        'patience': 10,
        'frequency': 10,
        'metric': 'hits@10',
        'relative_delta': 0.0001
    }
    
    # Train using PyKEEN pipeline with weighted training loop
    logger.info("Starting training with PyKEEN pipeline...")
    training_start_time = datetime.now()
    
    result = pipeline(
        training=training_factory,
        testing=dataset.testing,
        validation=dataset.validation,
        model='ComplEx',
        loss='crossentropy',  
        model_kwargs=model_kwargs,
        training_loop=WeightedSLCWATrainingLoop,
        training_loop_kwargs={
            'triple_weights': triple_weights,
            'weight_scale': config["weight_scale"],
        },
        training_kwargs=training_kwargs,
        optimizer='Adam',
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler='ExponentialLR',
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        negative_sampler='basic',
        evaluation_kwargs=evaluation_kwargs,
        random_seed=config.get("random_seed", 42),
        stopper='early',
        stopper_kwargs=stopper_kwargs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    training_duration = datetime.now() - training_start_time
    logger.info(f"Training completed in {training_duration}")
    
    # Extract final metrics
    metrics_dict = result.metric_results.to_dict()
    
    # Handle hierarchical metrics structure
    if 'both' in metrics_dict:
        test_metrics = metrics_dict['both']['realistic']
    elif 'realistic' in metrics_dict:
        test_metrics = metrics_dict['realistic']
    else:
        test_metrics = metrics_dict
    
    final_metrics = {
        "test_hits_at_1": test_metrics.get("hits_at_1", 0.0),
        "test_hits_at_3": test_metrics.get("hits_at_3", 0.0), 
        "test_hits_at_10": test_metrics.get("hits_at_10", 0.0),
        "test_mrr": test_metrics.get("inverse_harmonic_mean_rank", test_metrics.get("mean_reciprocal_rank", 0.0)),
        "test_mr": test_metrics.get("mean_rank", 0.0),
        "training_duration_seconds": training_duration.total_seconds(),
        "num_training_triples": len(training_factory.mapped_triples),
        "num_weighted_triples": len(triple_weights),
        "num_artificial_triples": len(training_factory.mapped_triples) - len(dataset.training.mapped_triples) if config.get("create_artificial_triples", True) else 0
    }
    
    logger.info(f"Final test metrics: {final_metrics}")
    
    # Log final results to wandb
    if config.get("use_wandb", False):
        wandb.log(final_metrics)
        wandb.finish()
    
    # Save model if requested
    if config.get("save_model", False):
        model_filename = f"complex_weighted_pipeline_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(result.model, model_filename)
        logger.info(f"Saved model to {model_filename}")
    
    return final_metrics


def main():
    """Main training function."""
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    config = {
        # Dataset and API
        "dataset": "FB15k237",
        "api_url": "http://localhost:8080/recommender",
        
        # Model hyperparameters 
        "embedding_dim": 1000,  
        "epochs": 1,          
        "batch_size": 1000,    
        "learning_rate": 0.1,  
        "regularize_weight": 0.05,  
        
        # Evaluation configuration
        "eval_batch_size": 256,
        
        # Weighted training parameters
        "weight_scale": 5.0,  # Amplification factor for weights
        "max_entities_to_score": 10,  # Score ALL entities
        "force_recompute_weights": False,  # Use cached weights if available
        
        # I/O triples mechanism
        "create_artificial_triples": False,
        "probability_threshold": 0.25,
        
        # Training configuration
        "random_seed": 42,
        
        # API configuration
        "api_max_retries": 3,
        "api_retry_delay": 1.0,
        "api_timeout": 30.0,
        
        # Logging and saving
        "use_wandb": True,
        "save_model": True,
    }
    
    try:
        results = train_weighted_complex_pipeline(config)
        
        # Save results to file
        results_filename = f"weighted_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_filename}")
        logger.info("Training completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()