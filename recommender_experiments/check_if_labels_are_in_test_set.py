#!/usr/bin/env python
"""
Script to analyze if recommended properties appear in the test set.
"""

import os
import os.path as osp
import logging
import torch
import numpy as np
import requests
from typing import Dict, List, Tuple, Any
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
    """Process and filter recommendations based on thresholds."""
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

def analyze_recommendations(num_entities: int = None):
    """Analyze recommendations against test set."""
    print("\n=== Analyzing Recommendations Against Test Set ===")
    
    # Get number of entities to analyze
    num_entities = num_entities or get_config('num_entities')
    print(f"\nWill analyze {num_entities} entities")
    
    # Load the dataset
    dataset_name = get_config('dataset.name')
    if dataset_name == "CoDExSmall":
        dataset = CoDExSmall()
    elif dataset_name == "FB15k237":
        dataset = FB15k237()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Print dataset info
    print(f"\nDataset: {dataset_name}")
    print(f"Entities: {len(dataset.entity_to_id)}")
    print(f"Relations: {len(dataset.relation_to_id)}")
    print(f"Training triples: {len(dataset.training.mapped_triples)}")
    print(f"Testing triples: {len(dataset.testing.mapped_triples)}")
    
    # Create id_to_relation mapping
    id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}
    
    # Get all relations in test set with their frequencies
    test_relations = defaultdict(int)
    for triple in dataset.testing.mapped_triples:
        relation_id = triple[1].item()
        relation = id_to_relation[relation_id]
        test_relations[relation] += 1
    
    print(f"\nFound {len(test_relations)} unique relations in test set")
    
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
    found_in_test = []
    not_in_test = []
    entity_stats = defaultdict(lambda: {'total': 0, 'found': 0})
    
    # Keep track of unique recommended labels
    unique_recommended_labels = set()
    unique_labels_in_test = set()
    
    # Process each entity and its properties
    for head_id, properties in sorted_entities:
        # Get recommendations for all properties of this entity
        property_list = list(properties)
        print(f"\nGetting recommendations for entity {head_id} (has {len(properties)} properties)")
        recommendations = get_recommendations(property_list)
        filtered_recommendations = process_recommendations(recommendations)
        
        # Update entity statistics
        entity_stats[head_id]['total'] = len(filtered_recommendations)
        
        # Check each recommendation against test set
        for new_prop, probability in filtered_recommendations:
            total_recommendations += 1
            unique_recommended_labels.add(new_prop)
            
            if new_prop in test_relations:
                found_in_test.append((new_prop, probability, test_relations[new_prop]))
                entity_stats[head_id]['found'] += 1
                unique_labels_in_test.add(new_prop)
            else:
                not_in_test.append((new_prop, probability))
    
    # Print overall results
    print("\n=== Overall Results ===")
    print(f"Total recommendations: {total_recommendations}")
    print(f"Found in test set: {len(found_in_test)}")
    print(f"Not in test set: {len(not_in_test)}")
    
    if found_in_test:
        print("\nProperties found in test set (sorted by probability):")
        print("Property | Probability | Frequency in test set")
        print("-" * 50)
        for prop, prob, freq in sorted(found_in_test, key=lambda x: x[1], reverse=True):
            print(f"{prop:8} | {prob:.4f} | {freq}")
    
    # Calculate additional statistics
    if found_in_test:
        avg_prob = sum(prob for _, prob, _ in found_in_test) / len(found_in_test)
        avg_freq = sum(freq for _, _, freq in found_in_test) / len(found_in_test)
        print(f"\nAdditional Statistics:")
        print(f"Average probability of recommendations: {avg_prob:.4f}")
        print(f"Average frequency in test set: {avg_freq:.2f}")
        
        # Check if there's correlation between probability and frequency
        probs = [prob for _, prob, _ in found_in_test]
        freqs = [freq for _, _, freq in found_in_test]
        if len(probs) > 1:
            correlation = sum((p - avg_prob) * (f - avg_freq) for p, f in zip(probs, freqs)) / len(probs)
            print(f"Correlation between probability and frequency: {correlation:.4f}")
    
    # Print entity-level statistics
    print("\n=== Entity-level Statistics ===")
    print("Entity ID | Properties | Total Recs | Found in Test | Success Rate")
    print("-" * 75)
    for entity_id, stats in sorted(entity_stats.items()):
        if stats['total'] > 0:  # Only show entities that got recommendations
            success_rate = stats['found'] / stats['total'] * 100
            num_props = len(entity_properties[entity_id])
            print(f"{entity_id:9} | {num_props:9} | {stats['total']:10} | {stats['found']:12} | {success_rate:11.1f}%")
    
    # Print unique labels summary at the end
    print("\n" + "="*50)
    print("=== FINAL UNIQUE LABELS SUMMARY ===")
    print("="*50)
    print(f"Total unique labels recommended: {len(unique_recommended_labels)}")
    print(f"Unique labels found in test set: {len(unique_labels_in_test)}")
    print(f"Percentage of unique labels in test set: {(len(unique_labels_in_test) / len(unique_recommended_labels) * 100):.1f}%")
    print(f"Total number of unique labels in test set: {len(test_relations)}")
    print("="*50)

if __name__ == "__main__":
    analyze_recommendations() 