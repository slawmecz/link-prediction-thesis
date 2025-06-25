#!/usr/bin/env python
"""
SavedRanksEvaluator implementation based on BioBLP's code.
This evaluator saves the individual ranks for significance testing.
"""

import os.path as osp
import numpy as np
import torch
from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.rank_based_evaluator import _iter_ranks
from pykeen.triples import TriplesFactory
from typing import Dict, List, Optional, Union, Any
import os
import pathlib

# Add safe globals for serialization
try:
    # For PyTorch 2.6+
    torch.serialization.add_safe_globals([pathlib.PureWindowsPath])
except (AttributeError, ImportError):
    # For older PyTorch versions that don't have this API
    pass

class SavedRanksEvaluator(RankBasedEvaluator):
    """
    Custom evaluator that saves ranks for statistical testing.
    
    This is based on the BioBLP implementation:
    https://github.com/elsevier-AI-Lab/BioBLP/blob/main/bioblp/evaluate.py
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_ranks = None
    
    def finalize(self) -> RankBasedMetricResults:
        """Finalize the evaluation and store the ranks."""
        if self.num_entities is None:
            raise ValueError("No entities provided for evaluation")

        # Create metrics using parent class method
        result = RankBasedMetricResults.from_ranks(
            metrics=self.metrics,
            rank_and_candidates=_iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates),
        )
        
        # Save the ranks before clearing them
        self.saved_ranks = self.ranks.copy()
        
        # Clear current ranks as the parent class would
        self.ranks.clear()
        self.num_candidates.clear()
        
        return result

def get_triple_ranks(model_path: str) -> np.ndarray:
    """
    Get ranks for all triples and save to CSV.
    
    Args:
        model_path: Path to the directory containing the model and triples
        
    Returns:
        Array of ranks for all triples
    """
    # Load the model
    model_file = osp.join(model_path, 'trained_model.pkl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set weights_only=False to allow loading the full PyKEEN model
    model = torch.load(model_file, map_location=device, weights_only=False)
    
    # Load training triples
    training_path = osp.join(model_path, 'training_triples')
    train = None
    
    # First try loading directly if it's a directory
    if osp.isdir(training_path):
        try:
            # Skip directory loading
            # Just log that we're skipping it
            print(f"Skipping directory loading for {training_path}, will try other methods")
        except Exception as e:
            print(f"Error loading from directory: {e}")
    
    # If that fails, try as a binary file
    if train is None:
        if osp.exists(training_path):
            try:
                train = TriplesFactory.from_path_binary(training_path)
                print(f"Loaded training triples from binary file: {training_path}")
            except Exception as e:
                print(f"Error loading binary file: {e}")
                
        # Try CSV file
        if train is None:
            csv_path = training_path + '.csv'
            if osp.exists(csv_path):
                try:
                    train = TriplesFactory.from_path(csv_path)
                    print(f"Loaded training triples from CSV file: {csv_path}")
                except Exception as e:
                    print(f"Error loading CSV file: {e}")
    
    # If still no triples, try direct loading with PyKEEN's dataset API
    if train is None:
        try:
            # Try to infer dataset from metrics.txt
            with open(osp.join(model_path, 'metrics.txt'), 'r') as f:
                metrics_content = f.read()
                if 'FB15k237' in metrics_content:
                    from pykeen.datasets import FB15k237
                    dataset = FB15k237()
                    train = dataset.training
                    print("Loaded FB15k237 dataset from PyKEEN")
                elif 'CoDExSmall' in metrics_content:
                    from pykeen.datasets import CoDExSmall
                    dataset = CoDExSmall()
                    train = dataset.training
                    print("Loaded CoDExSmall dataset from PyKEEN")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    if train is None:
        raise FileNotFoundError(f"Could not load training triples from {training_path}")
    
    # Load test triples with similar approach
    test_path = osp.join(model_path, 'testing_triples')
    test = None
    
    # First try loading directly if it's a directory
    if osp.isdir(test_path):
        try:
            # Skip directory loading 
            # Just log that we're skipping it
            print(f"Skipping directory loading for {test_path}, will try other methods")
        except Exception as e:
            print(f"Error loading from directory: {e}")
    
    # If that fails, try as file (old format)
    if test is None and osp.exists(test_path):
        try:
            # Check if from_path_binary accepts entity_to_id
            import inspect
            sig = inspect.signature(TriplesFactory.from_path_binary)
            if 'entity_to_id' in sig.parameters:
                test = TriplesFactory.from_path_binary(
                    test_path, 
                    entity_to_id=train.entity_to_id, 
                    relation_to_id=train.relation_to_id
                )
            else:
                # Older versions might not accept these arguments
                test = TriplesFactory.from_path_binary(test_path)
            print(f"Loaded testing triples from binary file: {test_path}")
        except Exception as e:
            print(f"Error loading binary file: {e}")
                
    # Try CSV file
    if test is None:
        csv_path = test_path + '.csv'
        if osp.exists(csv_path):
            try:
                test = TriplesFactory.from_path(csv_path, entity_to_id=train.entity_to_id, relation_to_id=train.relation_to_id)
                print(f"Loaded testing triples from CSV file: {csv_path}")
            except Exception as e:
                print(f"Error loading CSV file: {e}")
    
    # If still no triples, try direct loading with PyKEEN's dataset API
    if test is None:
        try:
            # Try to infer dataset from metrics.txt
            with open(osp.join(model_path, 'metrics.txt'), 'r') as f:
                metrics_content = f.read()
                if 'FB15k237' in metrics_content:
                    from pykeen.datasets import FB15k237
                    dataset = FB15k237()
                    test = dataset.testing
                    print("Loaded FB15k237 testing set from PyKEEN")
                elif 'CoDExSmall' in metrics_content:
                    from pykeen.datasets import CoDExSmall
                    dataset = CoDExSmall()
                    test = dataset.testing
                    print("Loaded CoDExSmall testing set from PyKEEN")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    if test is None:
        raise FileNotFoundError(f"Could not load testing triples from {test_path}")
    
    # Load validation triples if available
    valid = None
    validation_path = osp.join(model_path, 'validation_triples')
    
    # First try directory
    if osp.isdir(validation_path):
        try:
            # Skip directory loading 
            # Just log that we're skipping it
            print(f"Skipping directory loading for {validation_path}, will try other methods")
        except Exception as e:
            print(f"Error loading validation from directory: {e}")
    
    # Then try binary file
    if valid is None and osp.exists(validation_path):
        try:
            # Check if from_path_binary accepts entity_to_id
            import inspect
            sig = inspect.signature(TriplesFactory.from_path_binary)
            if 'entity_to_id' in sig.parameters:
                valid = TriplesFactory.from_path_binary(
                    validation_path, 
                    entity_to_id=train.entity_to_id, 
                    relation_to_id=train.relation_to_id
                )
            else:
                # Older versions might not accept these arguments
                valid = TriplesFactory.from_path_binary(validation_path)
            print(f"Loaded validation triples from binary file: {validation_path}")
        except Exception as e:
            print(f"Error loading validation binary file: {e}")
    
    # Finally try CSV
    if valid is None:
        csv_path = validation_path + '.csv'
        if osp.exists(csv_path):
            try:
                valid = TriplesFactory.from_path(csv_path, entity_to_id=train.entity_to_id, relation_to_id=train.relation_to_id)
                print(f"Loaded validation triples from CSV file: {csv_path}")
            except Exception as e:
                print(f"Error loading validation CSV file: {e}")
    
    # Prepare filter triples
    additional_filter_triples = [train.mapped_triples]
    if valid is not None:
        additional_filter_triples.append(valid.mapped_triples)
    
    # Evaluate with SavedRanksEvaluator
    evaluator = SavedRanksEvaluator(filtered=True)
    evaluator.evaluate(
        model=model,
        mapped_triples=test.mapped_triples,
        additional_filter_triples=additional_filter_triples
    )
    
    # Extract all ranks
    head_ranks = evaluator.saved_ranks.get(('head', 'realistic'), [])
    tail_ranks = evaluator.saved_ranks.get(('tail', 'realistic'), [])
    all_ranks = []
    
    # Concatenate head and tail ranks
    all_ranks.extend(head_ranks)
    all_ranks.extend(tail_ranks)
    ranks_array = np.concatenate(all_ranks) if all_ranks else np.array([])
    
    # Save ranks to CSV
    ranks_file = osp.join(model_path, 'ranks.csv')
    np.savetxt(ranks_file, ranks_array, fmt='%d')
    print(f"Saved {len(ranks_array)} ranks to {ranks_file}")
    
    return ranks_array 