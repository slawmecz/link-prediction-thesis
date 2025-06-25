#!/usr/bin/env python
"""
Script to analyze if triples created from recommendations appear in the test set.
"""

import os
import os.path as osp
import logging
import torch
import numpy as np
import requests
import argparse
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from pykeen.datasets import CoDExSmall, FB15k237

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config(key, default=None):
    """Get configuration from environment or use default."""
    configs = {
        'dataset.name': 'FB15k237',  # or CoDExSmall
        'api.url': 'http://localhost:8080/recommender',
        'api.timeout': 30,  # API request timeout in seconds
        'probability_threshold': 0.3,  # Probability threshold for recommendations
        'max_recommendations': 50,  # Maximum number of recommendations to use for each entity
        'num_entities': 1000000,  # Number of entities to analyze
    }
    return configs.get(key, default)

def get_recommendations(properties: List[str], api_url: str = None) -> List[Dict[str, Any]]:
    """Get property recommendations from the API."""
    api_url = api_url or get_config('api.url')
    api_timeout = get_config('api.timeout')
    
    try:
        data = {
            "properties": properties,
            "types": []  # Empty list as we're not using types
        }
        
        response = requests.post(
            api_url,
            json=data,
            timeout=api_timeout
        )
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
    
    # Filter and sort recommendations
    filtered_recommendations = [
        (rec['property'], rec['probability'])
        for rec in recommendations
        if rec['probability'] >= threshold
    ]
    
    # Sort by probability in descending order
    filtered_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N recommendations
    return filtered_recommendations[:max_recommendations]

def find_matching_tail_entities(
    head_id: int,
    relation: str,
    test_triples_by_hr: Dict[Tuple[int, str], List[int]],
    relation_to_id: Dict[str, int]
) -> List[int]:
    """Find existing tail entities in test set for a given head and relation."""
    relation_id = relation_to_id[relation]
    return test_triples_by_hr.get((head_id, relation), [])

def analyze_created_triples(num_entities: int = None, dataset_name: str = None, probability_threshold: float = None):
    """Analyze if triples created from recommendations appear in the test set."""
    print("\n=== Analyzing If Created Triples Appear In Test Set ===")
    
    # Get parameters from arguments or config
    num_entities = num_entities or get_config('num_entities')
    dataset_name = dataset_name or get_config('dataset.name')
    probability_threshold = probability_threshold or get_config('probability_threshold')
    
    print(f"\nWill analyze {num_entities} entities")
    print(f"Dataset: {dataset_name}")
    print(f"Probability threshold: {probability_threshold}")
    
    # Load the dataset
    if dataset_name == "CoDExSmall":
        dataset = CoDExSmall()
    elif dataset_name == "FB15k237":
        dataset = FB15k237()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Print dataset info
    print(f"\nEntities: {len(dataset.entity_to_id)}")
    print(f"Relations: {len(dataset.relation_to_id)}")
    print(f"Training triples: {len(dataset.training.mapped_triples)}")
    print(f"Testing triples: {len(dataset.testing.mapped_triples)}")
    
    # Create id_to_relation and relation_to_id mappings
    id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}
    relation_to_id = dataset.relation_to_id
    
    # Index test triples by (head, relation) for quick lookup
    test_triples_by_hr = defaultdict(list)
    for triple in dataset.testing.mapped_triples:
        head_id = triple[0].item()
        relation_id = triple[1].item()
        tail_id = triple[2].item()
        relation = id_to_relation[relation_id]
        test_triples_by_hr[(head_id, relation)].append(tail_id)
    
    # Group triples by head entity to get all properties for each entity
    entity_properties = defaultdict(set)
    for triple in dataset.training.mapped_triples:
        head_id = triple[0].item()
        relation_id = triple[1].item()
        full_property = id_to_relation[relation_id]
        entity_properties[head_id].add(full_property)
    
    # Sort entities by number of properties (descending) and take top N
    sorted_entities = sorted(
        entity_properties.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:num_entities]
    
    print(f"\nSelected {len(sorted_entities)} entities with most properties")
    
    # Statistics for analysis
    total_recommendations = 0
    total_potential_triples = 0
    triples_found_in_test = 0
    matched_relations = set()
    
    # Track triples that match
    matching_triples = []
    
    # Process each entity and its properties
    for head_id, properties in sorted_entities:
        # Get recommendations for all properties of this entity
        property_list = list(properties)
        print(f"\nAnalyzing entity {head_id} (has {len(properties)} properties)")
        recommendations = get_recommendations(property_list)
        filtered_recommendations = process_recommendations(
            recommendations, 
            threshold=probability_threshold
        )
        
        # Count recommendations
        total_recommendations += len(filtered_recommendations)
        
        # Check each recommendation 
        for new_prop, probability in filtered_recommendations:
            # Try to find matching tails in test set
            tail_entities = find_matching_tail_entities(
                head_id, 
                new_prop, 
                test_triples_by_hr,
                relation_to_id
            )
            
            num_matches = len(tail_entities)
            total_potential_triples += 1
            
            if num_matches > 0:
                triples_found_in_test += 1
                matched_relations.add(new_prop)
                
                # Save matching triples
                for tail_id in tail_entities:
                    matching_triples.append((head_id, new_prop, tail_id, probability))
    
    # Print results
    print("\n=== Results ===")
    print(f"Total entities analyzed: {len(sorted_entities)}")
    print(f"Total property recommendations: {total_recommendations}")
    print(f"Total recommended entity-relation pairs: {total_potential_triples}")
    print(f"Entity-relation pairs found in test set: {triples_found_in_test}")
    print(f"Percentage of recommended pairs found in test set: {(triples_found_in_test / total_potential_triples * 100):.2f}%")
    print(f"Number of unique relations matched: {len(matched_relations)}")
    
    # Print some examples of matching triples
    if matching_triples:
        print("\n=== Examples of Matching Entity-Relation Pairs ===")
        print("Head ID | Relation | Tail ID | Probability")
        print("-" * 60)
        
        # Sort by probability and print top 10
        for head, relation, tail, prob in sorted(matching_triples, key=lambda x: x[3], reverse=True)[:10]:
            print(f"{head:7} | {relation:8} | {tail:7} | {prob:.4f}")
    
    # Print final summary
    print("\n" + "="*60)
    print("=== FINAL SUMMARY ===")
    print("="*60)
    print(f"Total recommended entity-relation pairs: {total_potential_triples}")
    print(f"Entity-relation pairs found in test set: {triples_found_in_test}")
    print(f"Hit rate: {(triples_found_in_test / total_potential_triples * 100):.2f}%")
    print("="*60)

def main():
    """Parse command line arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Analyze if created triples appear in test set")
    parser.add_argument("--num-entities", type=int, default=None,
                      help="Number of entities to analyze (default: from config)")
    parser.add_argument("--dataset", type=str, choices=["FB15k237", "CoDExSmall"],
                      help="Dataset to use (default: from config)")
    parser.add_argument("--probability-threshold", type=float, default=None,
                      help="Probability threshold for recommendations (default: from config)")
    
    args = parser.parse_args()
    
    # Pass command line arguments directly to the analyze function
    analyze_created_triples(
        num_entities=args.num_entities,
        dataset_name=args.dataset,
        probability_threshold=args.probability_threshold
    )

if __name__ == "__main__":
    main() 