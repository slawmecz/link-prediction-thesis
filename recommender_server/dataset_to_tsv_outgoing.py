"""
Script that creates a TSV file, where each line represent a set of properties of one entity.
It adds only outging relations.
"""

import os
import logging
from typing import Dict, Set, List
import torch
from pykeen.datasets import CoDExSmall, FB15k237
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)

def get_entity_properties(triples: torch.Tensor, entity_id: int) -> set:
    """
    Get all properties (relations) where the entity is the head.
    
    Args:
        triples: Tensor of triples [head, relation, tail]
        entity_id: ID of the entity
        
    Returns:
        Set of relation IDs where the entity is the head
    """
    # Get all triples where entity is the head
    head_triples = triples[triples[:, 0] == entity_id]
    head_props = set(head_triples[:, 1].tolist())
    
    # Only return properties where entity is the head
    return head_props

#is not used anymore
def process_dataset_to_property_sets(dataset, min_properties: int = 1) -> List[Set[str]]:
    """
    Process a knowledge graph dataset to find properties for each entity.
    
    Args:
        dataset: Knowledge graph dataset (e.g., FB15k237, CoDExSmall)
        min_properties: Minimum number of properties required for an entity
        
    Returns:
        List of sets, where each set contains properties that appear together in an entity
    """
    print("Starting to process dataset...")
    
    # Get training triples and relation mappings
    triples = dataset.training.mapped_triples
    id_to_relation = dataset.training.relation_id_to_label
    
    print(f"Number of triples: {len(triples)}")
    print(f"Number of relations: {len(id_to_relation)}")
    
    # List to store property sets for each entity
    entity_property_sets = []
    
    # Get all unique entities
    unique_entities = set(triples[:, 0].tolist())
    print(f"Number of unique entities: {len(unique_entities)}")
    
    # For each entity, find its properties
    for entity_id in unique_entities:
        # Get all properties for this entity
        entity_props = get_entity_properties(triples, entity_id)
        print(f"Entity {entity_id} has {entity_props}")
        
        # If entity has enough properties, add them to the list
        if len(entity_props) >= min_properties:
            entity_property_sets.append(entity_props)
            if len(entity_property_sets) % 1000 == 0:
                print(f"Processed {len(entity_property_sets)} entities so far...")
    
    print(f"Found {len(entity_property_sets)} entities with at least {min_properties} properties")
    return entity_property_sets

def create_property_tsv(dataset, output_path: str, min_properties: int = 1) -> str:
    """
    Create a TSV file containing property sets for each entity from a dataset.
    Each line corresponds to exactly one entity. Only considers triples where
    entities are in the head position.
    
    Args:
        dataset: Knowledge graph dataset (e.g., FB15k237, CoDExSmall)
        output_path: Path where to save the TSV file
        min_properties: Minimum number of properties required for an entity
        
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
    
    # Get only unique head entities
    head_entities = set(triples[:, 0].tolist())
    
    print(f"Number of unique head entities: {len(head_entities)}")
    
    # Count entities by number of properties (head position only)
    entity_property_counts = {}
    for entity_id in head_entities:
        entity_props = get_entity_properties(triples, entity_id)
        # Use full relation paths instead of just the last word
        relation_labels = {id_to_relation[prop_id] for prop_id in entity_props}
        num_props = len(relation_labels)
        entity_property_counts[num_props] = entity_property_counts.get(num_props, 0) + 1
    
    print("\nEntity property distribution (head position only):")
    for num_props in sorted(entity_property_counts.keys()):
        print(f"Entities with {num_props} properties: {entity_property_counts[num_props]}")
    
    # Write to TSV file
    print(f"\nWriting property sets for {len(head_entities)} head entities to TSV...")
    entities_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for entity_id in head_entities:
            # Get all properties for this entity
            entity_props = get_entity_properties(triples, entity_id)
            
            # Use full relation paths instead of just the last word
            relation_labels = {id_to_relation[prop_id] for prop_id in entity_props}
            
            # If entity has enough properties, write them to the file
            if len(relation_labels) >= min_properties:
                # Convert set to sorted list for consistent output
                props_list = sorted(list(relation_labels))
                
                # Write properties for this entity
                line = "\t".join(props_list) + "\n"
                f.write(line)
                entities_written += 1
    
    print(f"\nWrote {entities_written} head entities to TSV file")
    print(f"Filtered out {len(head_entities) - entities_written} head entities with fewer than {min_properties} properties")
    print(f"Property TSV file created successfully at {output_path}")
    return output_path

def main():
    """
    Convert the dataset to TSV.
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
        output_path="data/index_heads.tsv",
        min_properties=1
    )
    
    print(f"TSV file created at: {tsv_path}")
    print(f"File exists: {os.path.exists(tsv_path)}")
    if os.path.exists(tsv_path):
        print(f"File size: {os.path.getsize(tsv_path)} bytes")

if __name__ == "__main__":
    main() 