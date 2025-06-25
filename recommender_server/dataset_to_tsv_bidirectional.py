"""
Script that creates a TSV file, where each line represent a set of properties of one entity.
It adds both outging (with a prefix 'O:') and incoming (with a prefix 'I:') relations.
"""

import os
import logging
from typing import Dict, Set, List
import torch
from pykeen.datasets import CoDExSmall, FB15k237
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)

def get_entity_outgoing_properties(triples: torch.Tensor, entity_id: int) -> set:
    """
    Get all outgoing properties (relations) where the entity is the head.
    
    Args:
        triples: Tensor of triples [head, relation, tail]
        entity_id: ID of the entity
        
    Returns:
        Set of relation IDs where the entity is the head
    """
    # Get all triples where entity is the head
    head_triples = triples[triples[:, 0] == entity_id]
    head_props = set(head_triples[:, 1].tolist())
    
    return head_props

def get_entity_incoming_properties(triples: torch.Tensor, entity_id: int) -> set:
    """
    Get all incoming properties (relations) where the entity is the tail.
    
    Args:
        triples: Tensor of triples [head, relation, tail]
        entity_id: ID of the entity
        
    Returns:
        Set of relation IDs where the entity is the tail
    """
    # Get all triples where entity is the tail
    tail_triples = triples[triples[:, 2] == entity_id]
    tail_props = set(tail_triples[:, 1].tolist())
    
    return tail_props

def create_property_tsv(dataset, output_path: str, min_properties: int = 1) -> str:
    """
    Create a TSV file containing both incoming and outgoing property sets for each entity from a dataset.
    Each line corresponds to exactly one entity. Considers both triples where entities are in the head position (outgoing)
    and in the tail position (incoming).
    
    Prefixes:
    - "O:" for outgoing properties (entity is head)
    - "I:" for incoming properties (entity is tail)
    
    Args:
        dataset: Knowledge graph dataset (e.g., FB15k237, CoDExSmall)
        output_path: Path where to save the TSV file
        min_properties: Minimum number of total properties (incoming + outgoing) required for an entity
        
    Returns:
        Path to the created TSV file
    """
    print(f"Creating property TSV file at {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Ensured directory exists: {os.path.dirname(output_path)}")
    
    # Get training triples and relation mappings
    triples = dataset.training.mapped_triples
    id_to_relation = dataset.training.relation_id_to_label
    
    print(f"Number of triples: {len(triples)}")
    print(f"Number of relations: {len(id_to_relation)}")
    
    # Get all unique entities (both head and tail positions)
    all_entities = set(triples[:, 0].tolist()).union(set(triples[:, 2].tolist()))
    head_entities = set(triples[:, 0].tolist())
    tail_entities = set(triples[:, 2].tolist())
    
    print(f"Number of unique entities: {len(all_entities)}")
    print(f"Number of unique head entities: {len(head_entities)}")
    print(f"Number of unique tail entities: {len(tail_entities)}")
    print(f"Entities only as head: {len(head_entities - tail_entities)}")
    print(f"Entities only as tail: {len(tail_entities - head_entities)}")
    print(f"Entities as both head and tail: {len(head_entities.intersection(tail_entities))}")
    
    # Count entities by number of properties (both directions)
    entity_property_counts = {}
    outgoing_property_counts = {}
    incoming_property_counts = {}
    
    for entity_id in all_entities:
        # Get outgoing properties (entity as head)
        outgoing_props = get_entity_outgoing_properties(triples, entity_id)
        outgoing_labels = {id_to_relation[prop_id] for prop_id in outgoing_props}
        
        # Get incoming properties (entity as tail)
        incoming_props = get_entity_incoming_properties(triples, entity_id)
        incoming_labels = {id_to_relation[prop_id] for prop_id in incoming_props}
        
        # Count total properties
        total_props = len(outgoing_labels) + len(incoming_labels)
        entity_property_counts[total_props] = entity_property_counts.get(total_props, 0) + 1
        
        # Count outgoing properties
        num_outgoing = len(outgoing_labels)
        outgoing_property_counts[num_outgoing] = outgoing_property_counts.get(num_outgoing, 0) + 1
        
        # Count incoming properties
        num_incoming = len(incoming_labels)
        incoming_property_counts[num_incoming] = incoming_property_counts.get(num_incoming, 0) + 1
    
    print("\nEntity total property distribution (incoming + outgoing):")
    for num_props in sorted(entity_property_counts.keys()):
        print(f"Entities with {num_props} total properties: {entity_property_counts[num_props]}")
    
    print("\nEntity outgoing property distribution:")
    for num_props in sorted(outgoing_property_counts.keys()):
        print(f"Entities with {num_props} outgoing properties: {outgoing_property_counts[num_props]}")
    
    print("\nEntity incoming property distribution:")
    for num_props in sorted(incoming_property_counts.keys()):
        print(f"Entities with {num_props} incoming properties: {incoming_property_counts[num_props]}")
    
    # Write to TSV file
    print(f"\nWriting property sets for {len(all_entities)} entities to TSV...")
    entities_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for entity_id in all_entities:
            # Get outgoing properties for this entity (entity as head)
            outgoing_props = get_entity_outgoing_properties(triples, entity_id)
            outgoing_labels = {f"O:{id_to_relation[prop_id]}" for prop_id in outgoing_props}
            
            # Get incoming properties for this entity (entity as tail)
            incoming_props = get_entity_incoming_properties(triples, entity_id)
            incoming_labels = {f"I:{id_to_relation[prop_id]}" for prop_id in incoming_props}
            
            # Combine both property sets
            all_labels = outgoing_labels.union(incoming_labels)
            
            # If entity has enough total properties, write them to the file
            if len(all_labels) >= min_properties:
                # Convert set to sorted list for consistent output
                props_list = sorted(list(all_labels))
                
                # Write properties for this entity
                line = "\t".join(props_list) + "\n"
                f.write(line)
                entities_written += 1
    
    print(f"\nWrote {entities_written} entities to TSV file")
    print(f"Filtered out {len(all_entities) - entities_written} entities with fewer than {min_properties} total properties")
    print(f"Property TSV file created successfully at {output_path}")
    return output_path

def main():
    """
    Create the TSV conversion with both incoming and outgoing relations.
    """
    from pykeen.datasets import FB15k237, CoDExSmall
    import os

    print("Current working directory:", os.getcwd())
    
    # Load dataset
    print("Loading dataset...")
    dataset = FB15k237()
    print("Dataset loaded successfully")

    # Create TSV
    tsv_path = create_property_tsv(
        dataset,
        output_path="data/index_all_directions.tsv",
        min_properties=1
    )
    
    print(f"TSV file created at: {tsv_path}")
    print(f"File exists: {os.path.exists(tsv_path)}")
    if os.path.exists(tsv_path):
        print(f"File size: {os.path.getsize(tsv_path)} bytes")

if __name__ == "__main__":
    main() 