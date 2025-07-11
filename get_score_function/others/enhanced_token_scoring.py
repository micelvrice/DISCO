import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math

class EnhancedTokenScorer:
    """
    Enhanced token importance scoring with multi-dimensional importance evaluation.
    
    Key improvements:
    1. Multi-scale attention analysis (local + global patterns)
    2. Temporal importance tracking with decay
    3. Semantic clustering for redundancy detection
    4. Adaptive scoring based on sequence characteristics
    5. Fast approximations for real-time inference
    """
    
    def __init__(self, window_size: int = 32, temporal_decay: float = 0.95, 
                 cluster_threshold: float = 0.8, fast_mode: bool = True):
        self.window_size = window_size
        self.temporal_decay = temporal_decay
        self.cluster_threshold = cluster_threshold
        self.fast_mode = fast_mode
        
        # Track token importance over time
        self.token_importance_history = []
        self.attention_pattern_cache = []
        self.semantic_clusters = []
        
    def calculate_multi_scale_attention_score(self, key_states: torch.Tensor, 
                                            value_states: torch.Tensor,
                                            query_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate importance scores using multi-scale attention analysis.
        
        Args:
            key_states: [bsz, num_heads, seq_len, head_dim]
            value_states: [bsz, num_heads, seq_len, head_dim]
            query_states: Optional query states for attention computation
            
        Returns:
            importance_scores: [bsz, num_heads, seq_len]
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        if self.fast_mode:
            return self._calculate_fast_importance_score(key_states, value_states)
        
        # 1. Local attention patterns (within recent window)
        local_scores = self._calculate_local_attention_score(key_states, value_states)
        
        # 2. Global attention patterns (across full sequence)
        global_scores = self._calculate_global_attention_score(key_states, value_states)
        
        # 3. Temporal decay adjustment
        temporal_scores = self._apply_temporal_decay(local_scores, global_scores, seq_len)
        
        # 4. Semantic clustering adjustment
        cluster_scores = self._apply_semantic_clustering_adjustment(key_states, value_states, temporal_scores)
        
        return cluster_scores
    
    def _calculate_fast_importance_score(self, key_states: torch.Tensor, 
                                       value_states: torch.Tensor) -> torch.Tensor:
        """
        Fast approximation of importance scoring for real-time inference.
        
        Uses efficient approximations:
        1. L2 norm variance as attention proxy
        2. Head-wise centroid distance for diversity
        3. Recent token emphasis for temporal importance
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # Exclude recent tokens from eviction consideration
        if seq_len <= self.window_size:
            # For short sequences, use uniform low scores
            return torch.ones(bsz, num_heads, seq_len, device=key_states.device) * 0.1
        
        scoreable_len = seq_len - self.window_size
        key_scoreable = key_states[:, :, :scoreable_len, :]
        value_scoreable = value_states[:, :, :scoreable_len, :]
        
        # 1. L2 distance from head-wise centroids (vectorized)
        key_centroids = key_scoreable.mean(dim=2, keepdim=True)  # [bsz, num_heads, 1, head_dim]
        value_centroids = value_scoreable.mean(dim=2, keepdim=True)
        
        key_distances = torch.norm(key_scoreable - key_centroids, dim=-1)  # [bsz, num_heads, scoreable_len]
        value_distances = torch.norm(value_scoreable - value_centroids, dim=-1)
        
        # 2. Normalize distances to [0, 1] per head
        key_distances_norm = self._normalize_per_head(key_distances)
        value_distances_norm = self._normalize_per_head(value_distances)
        
        # 3. Position-based temporal decay (recent tokens more important)
        position_weights = torch.linspace(1.0, 0.5, scoreable_len, device=key_states.device)
        position_weights = position_weights.view(1, 1, -1).expand(bsz, num_heads, -1)
        
        # 4. Combine scores with learned weights
        base_score = 0.6 * key_distances_norm + 0.4 * value_distances_norm
        temporal_adjusted = base_score * position_weights
        
        # 5. Add variance-based information content measure
        key_variance = torch.var(key_scoreable, dim=-1)  # [bsz, num_heads, scoreable_len]
        value_variance = torch.var(value_scoreable, dim=-1)
        
        # Normalize variance scores
        key_var_norm = self._normalize_per_head(key_variance)
        value_var_norm = self._normalize_per_head(value_variance)
        information_score = 0.5 * (key_var_norm + value_var_norm)
        
        # Final score: balance between distance and information content
        final_score = 0.7 * temporal_adjusted + 0.3 * information_score
        
        # Reshape to match expected output format
        if num_heads == 32:  # LLaMA-style grouped attention
            final_score = final_score.reshape(bsz, 8, 4, scoreable_len).mean(dim=2)
        
        return final_score
    
    def _normalize_per_head(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] per head dimension."""
        min_vals = tensor.amin(dim=-1, keepdim=True)
        max_vals = tensor.amax(dim=-1, keepdim=True)
        return (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    
    def _calculate_local_attention_score(self, key_states: torch.Tensor, 
                                       value_states: torch.Tensor) -> torch.Tensor:
        """Calculate attention scores within local windows."""
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # Use sliding window approach for local patterns
        window_scores = []
        
        for start_idx in range(0, seq_len - self.window_size + 1, self.window_size // 2):
            end_idx = min(start_idx + self.window_size, seq_len)
            
            window_keys = key_states[:, :, start_idx:end_idx, :]
            window_values = value_states[:, :, start_idx:end_idx, :]
            
            # Calculate self-attention within window
            attn_weights = torch.matmul(window_keys, window_keys.transpose(-2, -1))
            attn_weights = attn_weights / math.sqrt(head_dim)
            attn_probs = F.softmax(attn_weights, dim=-1)
            
            # Importance = sum of attention received from all positions
            local_importance = attn_probs.sum(dim=-2)  # [bsz, num_heads, window_len]
            window_scores.append(local_importance)
        
        # Combine overlapping windows (average overlaps)
        combined_scores = torch.zeros(bsz, num_heads, seq_len, device=key_states.device)
        overlap_counts = torch.zeros(seq_len, device=key_states.device)
        
        for i, (start_idx, score) in enumerate(zip(
            range(0, seq_len - self.window_size + 1, self.window_size // 2), 
            window_scores)):
            end_idx = min(start_idx + self.window_size, seq_len)
            combined_scores[:, :, start_idx:end_idx] += score
            overlap_counts[start_idx:end_idx] += 1
        
        # Average by overlap count
        overlap_counts = overlap_counts.clamp(min=1)
        combined_scores = combined_scores / overlap_counts.view(1, 1, -1)
        
        return combined_scores
    
    def _calculate_global_attention_score(self, key_states: torch.Tensor, 
                                        value_states: torch.Tensor) -> torch.Tensor:
        """Calculate global attention patterns across full sequence."""
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # For efficiency, use approximation for long sequences
        if seq_len > 512:
            # Sample representative positions
            sample_indices = torch.linspace(0, seq_len-1, 128, dtype=torch.long, device=key_states.device)
            sampled_keys = key_states[:, :, sample_indices, :]
            
            # Calculate attention from sampled positions to all positions
            global_attn = torch.matmul(sampled_keys, key_states.transpose(-2, -1))
            global_attn = global_attn / math.sqrt(head_dim)
            global_probs = F.softmax(global_attn, dim=-1)
            
            # Global importance = average attention from sampled positions
            global_importance = global_probs.mean(dim=-2)
        else:
            # Full global attention computation
            global_attn = torch.matmul(key_states, key_states.transpose(-2, -1))
            global_attn = global_attn / math.sqrt(head_dim)
            global_probs = F.softmax(global_attn, dim=-1)
            global_importance = global_probs.mean(dim=-2)
        
        return global_importance
    
    def _apply_temporal_decay(self, local_scores: torch.Tensor, 
                            global_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply temporal decay to emphasize recent tokens."""
        # Create decay weights (recent tokens have higher weights)
        decay_weights = torch.pow(self.temporal_decay, 
                                torch.arange(seq_len-1, -1, -1, device=local_scores.device))
        decay_weights = decay_weights.view(1, 1, -1)
        
        # Combine local and global with temporal weighting
        combined_scores = 0.6 * local_scores + 0.4 * global_scores
        temporal_adjusted = combined_scores * decay_weights
        
        return temporal_adjusted
    
    def _apply_semantic_clustering_adjustment(self, key_states: torch.Tensor, 
                                            value_states: torch.Tensor,
                                            base_scores: torch.Tensor) -> torch.Tensor:
        """
        Adjust scores based on semantic clustering to identify redundant tokens.
        
        Tokens in dense clusters are considered more redundant and get lower scores.
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # For efficiency, work with averaged representations across heads
        key_avg = key_states.mean(dim=1)  # [bsz, seq_len, head_dim]
        value_avg = value_states.mean(dim=1)
        
        # Calculate pairwise similarities
        key_similarities = F.cosine_similarity(
            key_avg.unsqueeze(2), key_avg.unsqueeze(1), dim=-1)  # [bsz, seq_len, seq_len]
        
        # Identify clusters (tokens with high similarity)
        cluster_mask = key_similarities > self.cluster_threshold
        cluster_density = cluster_mask.sum(dim=-1).float()  # [bsz, seq_len]
        
        # Redundancy penalty: higher density = higher redundancy = lower importance
        redundancy_penalty = 1.0 / (1.0 + cluster_density * 0.1)
        redundancy_penalty = redundancy_penalty.unsqueeze(1).expand(-1, num_heads, -1)
        
        # Apply penalty to base scores
        adjusted_scores = base_scores * redundancy_penalty
        
        return adjusted_scores
    
    def get_enhanced_token_scores(self, key_states: torch.Tensor, 
                                value_states: torch.Tensor,
                                attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main interface for enhanced token scoring.
        
        Args:
            key_states: [bsz, num_heads, seq_len, head_dim]
            value_states: [bsz, num_heads, seq_len, head_dim]
            attention_weights: Optional pre-computed attention weights
            
        Returns:
            token_scores: [bsz, num_heads, seq_len] importance scores
        """
        # Use multi-scale attention analysis
        importance_scores = self.calculate_multi_scale_attention_score(
            key_states, value_states)
        
        # If attention weights provided, incorporate them
        if attention_weights is not None:
            # Combine with actual attention patterns
            attn_importance = attention_weights.mean(dim=-2)  # Average across query positions
            importance_scores = 0.7 * importance_scores + 0.3 * attn_importance
        
        # Update history for temporal tracking
        self.token_importance_history.append(importance_scores.detach().cpu())
        
        # Keep only recent history
        if len(self.token_importance_history) > 10:
            self.token_importance_history.pop(0)
        
        return importance_scores
    
    def get_eviction_candidates(self, token_scores: torch.Tensor, 
                              num_evict: int, 
                              protected_recent: int = None) -> torch.Tensor:
        """
        Select tokens for eviction based on importance scores.
        
        Args:
            token_scores: [bsz, num_heads, seq_len] importance scores
            num_evict: Number of tokens to evict
            protected_recent: Number of recent tokens to protect (default: window_size)
            
        Returns:
            eviction_mask: [bsz, num_heads, seq_len] boolean mask for eviction
        """
        if protected_recent is None:
            protected_recent = self.window_size
        
        bsz, num_heads, seq_len = token_scores.shape
        
        # Create eviction mask
        eviction_mask = torch.zeros_like(token_scores, dtype=torch.bool)
        
        # Don't evict recent tokens
        if seq_len > protected_recent:
            scoreable_len = seq_len - protected_recent
            scoreable_scores = token_scores[:, :, :scoreable_len]
            
            # Select lowest scoring tokens for eviction
            num_evict_adjusted = min(num_evict, scoreable_len)
            
            # Get eviction indices per head
            _, evict_indices = torch.topk(scoreable_scores, 
                                        num_evict_adjusted, 
                                        dim=-1, 
                                        largest=False)  # Smallest scores
            
            # Set eviction mask
            for b in range(bsz):
                for h in range(num_heads):
                    eviction_mask[b, h, evict_indices[b, h]] = True
        
        return eviction_mask


def get_enhanced_scores_with_kv_fusion(key_states: torch.Tensor, 
                                     value_states: torch.Tensor, 
                                     scorer: Optional[EnhancedTokenScorer] = None,
                                     alpha: float = 1.0) -> torch.Tensor:
    """
    Enhanced version of the original get_scores_with_kv_fusion function.
    
    Args:
        key_states: Tensor of shape [bsz, num_heads, seq_len, head_dim]
        value_states: Tensor of shape [bsz, num_heads, seq_len, head_dim]
        scorer: Optional EnhancedTokenScorer instance
        alpha: Weight for value score (default: 1.0)
        
    Returns:
        final_score: Enhanced importance scores
    """
    if scorer is None:
        scorer = EnhancedTokenScorer(fast_mode=True)
    
    # Use enhanced scoring method
    enhanced_scores = scorer.get_enhanced_token_scores(key_states, value_states)
    
    return enhanced_scores