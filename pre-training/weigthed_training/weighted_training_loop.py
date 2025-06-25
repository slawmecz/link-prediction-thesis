"""
Weighted Training Loop for PyKEEN ComplEx Model

This module implements a custom training loop that applies triple-specific weights
during training based on recommender system query scores.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from pykeen.training import SLCWATrainingLoop
from pykeen.typing import MappedTriples


class WeightedSLCWATrainingLoop(SLCWATrainingLoop):
    """
    Custom training loop that applies triple-specific weights during training.
    
    This extends PyKEEN's SLCWATrainingLoop to incorporate query-aware weights
    that emphasize triples involving entities with high recommender query scores.
    """
    
    def __init__(
        self,
        triple_weights: Dict[Tuple[int, int, int], float],
        weight_scale: float = 5.0,
        **kwargs
    ):
        """
        Initialize the weighted training loop.
        
        Args:
            triple_weights: Dictionary mapping (head_id, relation_id, tail_id) to weight
            weight_scale: Scale factor to amplify weight differences
            **kwargs: Additional arguments passed to parent SLCWATrainingLoop
        """
        super().__init__(**kwargs)
        self.triple_weights = triple_weights
        self.weight_scale = weight_scale
    
    def _get_batch_weights(self, positive_batch: torch.Tensor) -> torch.Tensor:
        """
        Get weights for a batch of positive triples.
        
        Args:
            positive_batch: Tensor of shape (batch_size, 3) with positive triples
            
        Returns:
            Tensor of shape (batch_size,) with weights for each triple
        """
        batch_size = positive_batch.shape[0]
        # Get device dynamically from the model
        device = next(self.model.parameters()).device
        weights = torch.ones(batch_size, device=device)
        
        # Apply weights for each triple in the batch
        for i, triple in enumerate(positive_batch):
            head_id, relation_id, tail_id = tuple(triple.cpu().numpy())
            triple_key = (int(head_id), int(relation_id), int(tail_id))
            
            if triple_key in self.triple_weights:
                base_weight = self.triple_weights[triple_key]
                # Apply scaling: weight_scale * base_weight
                scaled_weight = self.weight_scale * base_weight
                weights[i] = scaled_weight
            else:
                # Default weight for triples not in our weight dictionary
                weights[i] = 1.0
        
        return weights
    
    def _process_batch(
        self,
        batch: MappedTriples,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Process a batch with weighted loss calculation.
        
        This overrides the parent method to apply triple-specific weights.
        """
        # First call the parent method to get the standard loss
        loss = super()._process_batch(
            batch, start, stop, label_smoothing, slice_size
        )
        
        # Get the batch slice for weight calculation
        # MappedTriples is a tensor, so we can index it directly with integers
        try:
            # Create indices to access the batch slice
            end_idx = min(stop, len(batch))
            batch_indices = list(range(start, end_idx))
            
            # Get the actual triples for this batch slice
            if len(batch_indices) > 0:
                batch_slice = batch[batch_indices]  # Use integer list indexing instead of slice
            else:
                # Empty batch case
                device = next(self.model.parameters()).device
                batch_weights = torch.ones(0, device=device)
                return loss
                
        except (IndexError, AttributeError, TypeError) as e:
            # Fallback: create default weights if we can't process the batch
            batch_size = stop - start
            device = next(self.model.parameters()).device
            batch_weights = torch.ones(batch_size, device=device)
            mean_weight = batch_weights.mean()
            return loss * mean_weight
        
        # Get weights for this batch
        batch_weights = self._get_batch_weights(batch_slice)
        
        # Apply weights to the loss
        if loss.dim() > 0:
            # If loss is per-triple, apply weights directly
            if loss.shape[0] == batch_weights.shape[0]:
                weighted_loss = loss * batch_weights
            else:
                # If loss is aggregated, apply mean weight
                mean_weight = batch_weights.mean()
                weighted_loss = loss * mean_weight
        else:
            # Scalar loss - apply mean weight
            mean_weight = batch_weights.mean()
            weighted_loss = loss * mean_weight
        
        return weighted_loss


def create_weighted_training_loop(
    model,
    triples_factory,
    triple_weights: Dict[Tuple[int, int, int], float],
    weight_scale: float = 5.0,
    negative_sampler: str = "basic",
    negative_sampler_kwargs: Optional[Dict[str, Any]] = None,
    optimizer = None,
    **kwargs
) -> WeightedSLCWATrainingLoop:
    """
    Create a weighted training loop with proper configuration.
    
    Args:
        model: The PyKEEN model to train
        triples_factory: The triples factory containing training data
        triple_weights: Dictionary mapping triples to weights
        weight_scale: Scale factor for weights
        negative_sampler: Type of negative sampler to use
        negative_sampler_kwargs: Additional arguments for negative sampler
        optimizer: The optimizer to use for training
        **kwargs: Additional arguments for training loop
        
    Returns:
        Configured WeightedSLCWATrainingLoop instance
    """
    if negative_sampler_kwargs is None:
        negative_sampler_kwargs = {}
    
    # Set default negative sampler parameters if not provided
    if "num_negs_per_pos" not in negative_sampler_kwargs:
        negative_sampler_kwargs["num_negs_per_pos"] = 1000
    
    training_loop = WeightedSLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        triple_weights=triple_weights,
        weight_scale=weight_scale,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
        optimizer=optimizer,
        **kwargs
    )
    
    return training_loop 