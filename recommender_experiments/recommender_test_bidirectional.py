#!/usr/bin/env python
"""
Script to analyze precision-recall for bidirectional property recommendations (both incoming and outgoing).
"""

import logging
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from pykeen.datasets import CoDExSmall, FB15k237

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_entity_outgoing_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """Get all outgoing properties (relations) where the entity is the head."""
    head_triples = triples[triples[:, 0] == entity_id]
    return {f"O:{id_to_relation[rel_id.item()]}" for rel_id in head_triples[:, 1]}

def get_entity_incoming_properties(triples: torch.Tensor, entity_id: int, id_to_relation: Dict[int, str]) -> set:
    """Get all incoming properties (relations) where the entity is the tail."""
    tail_triples = triples[triples[:, 2] == entity_id]
    return {f"I:{id_to_relation[rel_id.item()]}" for rel_id in tail_triples[:, 1]}

def get_recommendations(properties: List[str]) -> List[Dict[str, Any]]:
    """Get property recommendations from the API."""
    try:
        data = {
            "properties": properties,
            "types": []
        }
        
        response = requests.post(
            "http://localhost:8080/recommender",
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        recommendations = response.json().get("recommendations", [])
        return recommendations
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return []

def calculate_precision_recall(
    entity_validation_props: Set[str], 
    recommendations: List[Tuple[str, float]],
    threshold: float
) -> Tuple[float, float]:
    """Calculate precision and recall for a single entity."""
    # Filter recommendations by threshold
    recommended_props = {prop for prop, prob in recommendations if prob >= threshold}
    
    if not recommended_props:
        return 0.0, 0.0
    
    # Calculate precision and recall
    true_positives = len(recommended_props & entity_validation_props)
    precision = true_positives / len(recommended_props) if recommended_props else 0.0
    recall = true_positives / len(entity_validation_props) if entity_validation_props else 0.0
    
    return precision, recall

def create_precision_recall_graph():
    """Create precision-recall graph for different probability thresholds."""
    print("Creating precision-recall graph for bidirectional properties...")
    
    # Load dataset
    dataset = FB15k237()
    print(f"Loaded FB15k237 dataset with {len(dataset.entity_to_id)} entities")
    
    # Create mappings
    id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}
    
    # Get entity properties from training and validation sets
    training_props = defaultdict(dict)
    validation_props = defaultdict(dict)
    
    print("Processing training triples...")
    for triple in dataset.training.mapped_triples:
        head_id = triple[0].item()
        tail_id = triple[2].item()
        relation = id_to_relation[triple[1].item()]
        
        # Add outgoing property for head
        if 'outgoing' not in training_props[head_id]:
            training_props[head_id]['outgoing'] = set()
        training_props[head_id]['outgoing'].add(f"O:{relation}")
        
        # Add incoming property for tail
        if 'incoming' not in training_props[tail_id]:
            training_props[tail_id]['incoming'] = set()
        training_props[tail_id]['incoming'].add(f"I:{relation}")
    
    print("Processing validation triples...")
    for triple in dataset.validation.mapped_triples:
        head_id = triple[0].item()
        tail_id = triple[2].item()
        relation = id_to_relation[triple[1].item()]
        
        # Add outgoing property for head
        if 'outgoing' not in validation_props[head_id]:
            validation_props[head_id]['outgoing'] = set()
        validation_props[head_id]['outgoing'].add(f"O:{relation}")
        
        # Add incoming property for tail
        if 'incoming' not in validation_props[tail_id]:
            validation_props[tail_id]['incoming'] = set()
        validation_props[tail_id]['incoming'].add(f"I:{relation}")
    
    # Select entities that appear in both training and validation
    common_entities = set(training_props.keys()) & set(validation_props.keys())
    print(f"Found {len(common_entities)} entities in both training and validation sets")
    
    # Take top entities with most total properties
    selected_entities = sorted(
        [(entity_id, len(training_props[entity_id].get('outgoing', set()) | training_props[entity_id].get('incoming', set()))) 
         for entity_id in common_entities],
        key=lambda x: x[1],
        reverse=True
    )[:14505] # limit number of entities; for FB15k237 the max number is 14505
    
    print(f"Selected {len(selected_entities)} entities for analysis")
    
    # Calculate average properties per entity
    avg_out_train = np.mean([len(training_props[eid]['outgoing']) for eid, _ in selected_entities if 'outgoing' in training_props[eid]])
    avg_in_train = np.mean([len(training_props[eid]['incoming']) for eid, _ in selected_entities if 'incoming' in training_props[eid]])
    avg_out_val = np.mean([len(validation_props[eid]['outgoing']) for eid, _ in selected_entities if 'outgoing' in validation_props[eid]])
    avg_in_val = np.mean([len(validation_props[eid]['incoming']) for eid, _ in selected_entities if 'incoming' in validation_props[eid]])
    
    print(f"\nAverage properties per entity:")
    print(f"Training - Outgoing: {avg_out_train:.1f}, Incoming: {avg_in_train:.1f}")
    print(f"Validation - Outgoing: {avg_out_val:.1f}, Incoming: {avg_in_val:.1f}")
    
    # Probability thresholds to test
    thresholds = np.arange(0.1, 0.95, 0.05)
    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        entity_precisions = []
        entity_recalls = []
        total_recommendations = 0
        total_true_positives = 0
        
        print(f"\nTesting threshold: {threshold:.2f}")
        
        for entity_id, _ in selected_entities:
            # Get all properties (both incoming and outgoing) for training
            all_train_props = (
                training_props[entity_id].get('outgoing', set()) |
                training_props[entity_id].get('incoming', set())
            )
            
            # Get all properties (both incoming and outgoing) for validation
            all_val_props = (
                validation_props[entity_id].get('outgoing', set()) |
                validation_props[entity_id].get('incoming', set())
            )
            
            if not all_train_props:
                continue
            
            # Get recommendations based on all training properties
            recommendations = get_recommendations(list(all_train_props))
            if not recommendations:
                continue
            
            # Convert to list of tuples
            rec_tuples = [(rec['property'], rec['probability']) for rec in recommendations]
            
            # Calculate precision and recall
            precision, recall = calculate_precision_recall(
                all_val_props,
                rec_tuples,
                threshold
            )
            
            # Count recommendations and true positives
            filtered_recs = [r for r in rec_tuples if r[1] >= threshold]
            total_recommendations += len(filtered_recs)
            true_positives = len(set(r[0] for r in filtered_recs) & all_val_props)
            total_true_positives += true_positives
            
            entity_precisions.append(precision)
            entity_recalls.append(recall)
        
        # Average across all entities
        avg_precision = np.mean(entity_precisions) if entity_precisions else 0.0
        avg_recall = np.mean(entity_recalls) if entity_recalls else 0.0
        
        precision_scores.append(avg_precision)
        recall_scores.append(avg_recall)
        
        print(f"Threshold {threshold:.2f}:")
        print(f"  Precision={avg_precision:.3f}")
        print(f"  Recall={avg_recall:.3f}")
        print(f"  Total recommendations: {total_recommendations}")
        print(f"  Total true positives: {total_true_positives}")
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.plot(precision_scores, recall_scores, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Precision', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title('Recall-Precision Curve for Bidirectional Property Recommendations', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(precision_scores) + 0.1)
    plt.ylim(0, max(recall_scores) + 0.1)
    
    # Add threshold annotations for some points
    for i in range(0, len(thresholds), 2):
        plt.annotate(f'{thresholds[i]:.2f}', 
                    (precision_scores[i], recall_scores[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curve_bidirectional_all_entities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nGraph saved as 'precision_recall_curve_bidirectional_all_entities.png'")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Max Precision: {max(precision_scores):.3f}")
    print(f"Max Recall: {max(recall_scores):.3f}")
    
    # Calculate F1 scores safely
    f1_scores = []
    for p, r in zip(precision_scores, recall_scores):
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0
        f1_scores.append(f1)
    
    best_f1_idx = np.argmax(f1_scores)
    print(f"Best F1 Score: {f1_scores[best_f1_idx]:.3f} at threshold {thresholds[best_f1_idx]:.2f}")

if __name__ == "__main__":
    create_precision_recall_graph() 