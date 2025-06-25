#!/usr/bin/env python
"""
Simple script to check if recommender suggests labels already in the input.
"""

import argparse
import requests
import random
import json
from collections import defaultdict
from pykeen.datasets import FB15k237, CoDExSmall

def get_config(key, default=None):
    """Get configuration from environment or use default."""
    configs = {
        'dataset.name': 'FB15k237',  # or CoDExSmall
        'api.url': 'http://localhost:8080/recommender',
        'api.timeout': 30,  # API request timeout in seconds
    }
    return configs.get(key, default)

def check_duplicates(dataset_name, api_url=None, sample_size=100000):
    # Use default API URL if not provided
    if api_url is None:
        api_url = get_config('api.url')
    
    # Load dataset
    if dataset_name.lower() == "fb15k237":
        dataset = FB15k237()
    elif dataset_name.lower() == "codexsmall":
        dataset = CoDExSmall()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get test triples and mappings
    test_triples = dataset.training.mapped_triples
    id_to_entity = {v: k for k, v in dataset.entity_to_id.items()}
    id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}
    
    # Extract entity-relation pairs from test set
    entity_relations = defaultdict(list)
    for head_id, rel_id, tail_id in test_triples:
        head = id_to_entity[head_id.item()]
        relation = id_to_relation[rel_id.item()]
        entity_relations[head].append(relation)
    
    # Sample entities
    entities = list(entity_relations.keys())
    if sample_size > 0 and sample_size < len(entities):
        entities = random.sample(entities, sample_size)
    
    # Stats
    total_recs = 0
    duplicate_recs = 0
    entities_with_dupes = 0
    
    # Example queries and responses
    num_examples_shown = 0
    max_examples = 5  # Show details for up to 5 examples
    
    # Check each entity
    print(f"Checking {len(entities)} entities...")
    print(f"Using recommender API at: {api_url}")
    for i, entity in enumerate(entities):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(entities)}")
            
        relations = entity_relations[entity]
        if not relations:
            continue
            
        # Query API
        try:
            # Prepare the payload
            payload = {
                "properties": relations,
                "types": []  # Empty list as we're not using types
            }
            
            # Show example of query for the first few entities
            show_example = num_examples_shown < max_examples
            
            if show_example:
                print("\n=== EXAMPLE QUERY ===")
                print(f"Entity: {entity}")
                print(f"Input relations: {relations}")
                print(f"Request payload: {json.dumps(payload, indent=2)}")
            
            # Make the API call
            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                response_data = response.json()
                recommendations = response_data.get("recommendations", [])
                
                # Show example of response
                if show_example:
                    print("\n=== EXAMPLE RESPONSE ===")
                    print(f"Status code: {response.status_code}")
                    print(f"Response data: {json.dumps(response_data, indent=2)}")
                    num_examples_shown += 1
                
                # Count duplicates
                duplicates = [r for r in recommendations if r in relations]
                
                total_recs += len(recommendations)
                duplicate_recs += len(duplicates)
                
                if duplicates:
                    entities_with_dupes += 1
                    print(f"\nEntity '{entity}' has {len(duplicates)} duplicate recommendations:")
                    print(f"  Input relations: {relations}")
                    print(f"  All recommendations: {recommendations}")
                    print(f"  Duplicate recommendations: {duplicates}")
        except Exception as e:
            print(f"Error querying API for {entity}: {str(e)}")
    
    # Print summary
    print("\n=== RESULTS ===")
    print(f"Entities checked: {len(entities)}")
    print(f"Total recommendations: {total_recs}")
    print(f"Duplicate recommendations: {duplicate_recs}")
    if total_recs > 0:
        print(f"Duplicate percentage: {duplicate_recs/total_recs*100:.2f}%")
    print(f"Entities with duplicates: {entities_with_dupes} ({entities_with_dupes/len(entities)*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Check recommender for duplicate suggestions")
    parser.add_argument("--dataset", type=str, required=True, choices=["FB15k237", "CoDExSmall"])
    parser.add_argument("--api-url", type=str, help=f"URL of the recommendation API (default: {get_config('api.url')})")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--examples", type=int, default=5, help="Number of detailed query/response examples to show")
    
    args = parser.parse_args()
    
    # Update API URL if provided
    api_url = args.api_url or get_config('api.url')
    
    check_duplicates(args.dataset, api_url, args.sample_size)

if __name__ == "__main__":
    main() 