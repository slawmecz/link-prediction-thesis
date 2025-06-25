"""
Leave-One-Out Scoring Utilities for Query-Aware Training

This module implements Leave-One-Out scoring where for each triple
we compute query scores from both the head and tail entity perspectives and average them.
"""

import json
import logging
import requests
import time
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeaveOneOutScorer:
    """
    Computes Leave-One-Out scores for knowledge graph triples using a recommender API.
    
    For each triple (head, relation, tail), this computes:
    1. Score from head perspective: remove relation from head's properties, query API, check if relation appears
    2. Score from tail perspective: remove inverse relation from tail's properties, query API
    3. Average the two scores
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8080/recommender",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        """
        Initialize the Leave-One-Out scorer.
        
        Args:
            api_url: URL of the recommender API
            max_retries: Maximum number of API call retries
            retry_delay: Delay between retries in seconds
            timeout: API call timeout in seconds
        """
        self.api_url = api_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Cache for entity properties and API responses
        self.entity_properties_cache: Dict[str, Set[str]] = {}
        self.api_response_cache: Dict[str, Dict[str, float]] = {}
    
    def _call_recommender_api(self, properties: List[str]) -> Optional[Dict[str, float]]:
        """
        Call the recommender API with retry logic.
        
        Args:
            properties: List of properties to include in the query
            
        Returns:
            Dictionary mapping recommended properties to their probabilities, or None if failed
        """
        # Create cache key
        cache_key = json.dumps(sorted(properties))
        if cache_key in self.api_response_cache:
            return self.api_response_cache[cache_key]
        
        for attempt in range(self.max_retries):
            try:
                # Use the same format as parameter_sharing.py
                request_data = {
                    "properties": properties,
                    "types": []
                }
                
                response = requests.post(
                    self.api_url,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # API returns {"recommendations": [...]} format
                    if "recommendations" in result:
                        recommendations = result["recommendations"]
                        
                        # Convert to our expected format: {property: probability}
                        property_scores = {}
                        for rec in recommendations:
                            if isinstance(rec, dict) and "property" in rec and "probability" in rec:
                                property_scores[rec["property"]] = rec["probability"]
                        
                        # Cache the result
                        self.api_response_cache[cache_key] = property_scores
                        return property_scores
                    else:
                        logger.warning(f"API response missing 'recommendations' field")
                        return {}
                else:
                    logger.warning(f"API call failed with status {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"Failed to get API response after {self.max_retries} attempts")
        return None
    
    def build_entity_properties_map(
        self,
        triples: List[Tuple[str, str, str]]
    ) -> Dict[str, Set[str]]:
        """
        Build a map from entities to their properties based on the training triples.
        
        Args:
            triples: List of (head, relation, tail) triples
            
        Returns:
            Dictionary mapping entity names to sets of their properties
        """
        logger.info("Building entity properties map...")
        entity_properties = defaultdict(set)
        
        for head, relation, tail in triples:
            # Add outgoing property for head entity
            entity_properties[head].add(f"O:{relation}")
            # Add incoming property for tail entity  
            entity_properties[tail].add(f"I:{relation}")
        
        # Convert to regular dict with frozen sets
        self.entity_properties_cache = {
            entity: set(props) for entity, props in entity_properties.items()
        }
        
        logger.info(f"Built properties map for {len(self.entity_properties_cache)} entities")
        return self.entity_properties_cache
    
    def get_triple_score_from_head(
        self,
        head: str,
        relation: str,
        tail: str
    ) -> float:
        """
        Get the query score for a triple from the head entity's perspective.
        
        This asks: "Given what we know about the head entity, how likely is it 
        that the head entity would have this outgoing relation?"
        
        Args:
            head: Head entity name
            relation: Relation name
            tail: Tail entity name (not used in scoring, just for context)
            
        Returns:
            Score between 0 and 1, or 0.05 if not found in recommendations
        """
        if head not in self.entity_properties_cache:
            return 0.05  # Default score for unknown entities

        # Get all properties of the head entity
        head_properties = self.entity_properties_cache[head].copy()
        
        # Remove the OUTGOING version of the target relation (leave-one-out)
        target_property = f"O:{relation}"
        if target_property in head_properties:
            head_properties.remove(target_property)

        if not head_properties:
            return 0.05  # No properties left to query with

        # Query the recommender API with remaining properties
        properties_list = list(head_properties)
        recommendations = self._call_recommender_api(properties_list)

        if recommendations is None:
            return 0.05  # API call failed

        # Look for the OUTGOING version of the relation in recommendations
        # This asks: "How likely is it that the head entity has this outgoing relation?"
        return recommendations.get(target_property, 0.05)
    
    def get_triple_score_from_tail(
        self,
        head: str,
        relation: str,
        tail: str
    ) -> float:
        """
        Get the query score for a triple from the tail entity's perspective.
        
        This asks: "Given what we know about the tail entity, how likely is it 
        that something would have this outgoing relation TO the tail entity?"
        
        Args:
            head: Head entity name (not used in scoring, just for context)
            relation: Relation name
            tail: Tail entity name
            
        Returns:
            Score between 0 and 1, or 0.05 if not found in recommendations
        """
        if tail not in self.entity_properties_cache:
            return 0.05  # Default score for unknown entities
        
        # Get all properties of the tail entity
        tail_properties = self.entity_properties_cache[tail].copy()
        
        # Remove the INCOMING version of the target relation (leave-one-out)
        incoming_property = f"I:{relation}"
        if incoming_property in tail_properties:
            tail_properties.remove(incoming_property)
        
        if not tail_properties:
            return 0.05  # No properties left to query with
        
        # Query the recommender API with remaining properties
        properties_list = list(tail_properties)
        recommendations = self._call_recommender_api(properties_list)
        
        if recommendations is None:
            return 0.05  # API call failed
        
        # Look for the OUTGOING version of the relation in recommendations
        # This asks: "How likely is it that something has this outgoing relation to the tail?"
        target_property = f"O:{relation}"  
        return recommendations.get(target_property, 0.05)
    
    def get_triple_score_averaged(
        self,
        head: str,
        relation: str,
        tail: str
    ) -> float:
        """
        Get the averaged query score for a triple from both entity perspectives.
        
        Both perspectives score the same outgoing relation (O:relation):
        - Head perspective: "How likely is head to have this outgoing relation?"
        - Tail perspective: "How likely is it that something has this outgoing relation to tail?"
        
        Args:
            head: Head entity name
            relation: Relation name
            tail: Tail entity name
            
        Returns:
            Average score from head and tail perspectives (both scoring O:relation)
        """
        head_score = self.get_triple_score_from_head(head, relation, tail)
        tail_score = self.get_triple_score_from_tail(head, relation, tail)
        
        # Average the two scores (both are scoring the same O:relation property)
        averaged_score = (head_score + tail_score) / 2.0
        return averaged_score
    
    def score_all_triples(
        self,
        triples: List[Tuple[str, str, str]],
        max_entities_to_score: Optional[int] = None,
        use_averaging: bool = True
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Score all triples using Leave-One-Out methodology.
        
        Args:
            triples: List of (head, relation, tail) triples to score
            max_entities_to_score: Maximum number of entities to score (None for all)
            use_averaging: Whether to average scores from both entity perspectives
            
        Returns:
            Dictionary mapping triples to their query scores
        """
        logger.info(f"Scoring {len(triples)} triples...")
        
        # Build entity properties map first
        self.build_entity_properties_map(triples)
        
        # Determine which entities to score
        all_entities = set()
        for head, relation, tail in triples:
            all_entities.add(head)
            all_entities.add(tail)
        
        entities_to_score = list(all_entities)
        if max_entities_to_score is not None:
            entities_to_score = entities_to_score[:max_entities_to_score]
            logger.info(f"Limiting scoring to {len(entities_to_score)} entities")
        
        # Score triples
        triple_scores = {}
        scored_count = 0
        
        for i, (head, relation, tail) in enumerate(triples):
            # Only score if both entities are in our scoring set
            if max_entities_to_score is not None:
                if head not in entities_to_score and tail not in entities_to_score:
                    triple_scores[(head, relation, tail)] = 0.05  # Default score
                    continue
            
            if use_averaging:
                score = self.get_triple_score_averaged(head, relation, tail)
            else:
                # Use only head perspective for compatibility
                score = self.get_triple_score_from_head(head, relation, tail)
            
            triple_scores[(head, relation, tail)] = score
            scored_count += 1
            
            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"Scored {i + 1}/{len(triples)} triples")
        
        logger.info(f"Completed scoring {scored_count} triples")
        return triple_scores
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached data."""
        return {
            "entity_properties_cached": len(self.entity_properties_cache),
            "api_responses_cached": len(self.api_response_cache)
        }


def create_leave_one_out_scorer(
    api_url: str = "http://localhost:8080/recommender",
    **kwargs
) -> LeaveOneOutScorer:
    """
    Create a Leave-One-Out scorer with default configuration.
    
    Args:
        api_url: URL of the recommender API
        **kwargs: Additional arguments for LeaveOneOutScorer
        
    Returns:
        Configured LeaveOneOutScorer instance
    """
    return LeaveOneOutScorer(api_url=api_url, **kwargs) 