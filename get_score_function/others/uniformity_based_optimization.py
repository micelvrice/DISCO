"""
Uniformity-Based KV Cache Compression Optimization

Core Principle: The more uniform the distribution, the less redundancy there is.
This module implements layer budget allocation and token importance scoring
based on distribution uniformity measures.

Key Concepts:
1. Uniform distributions contain maximum information (high entropy)
2. Non-uniform distributions indicate redundancy (clustering, repetition)
3. Layer and token scoring should be correlated through uniformity measures
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.stats import entropy
from sklearn.decomposition import PCA
import math

class UniformityMeasures:
    """Collection of uniformity measurement functions."""
    
    @staticmethod
    def entropy_uniformity(distribution: torch.Tensor, base: str = 'e') -> torch.Tensor:
        """
        Calculate entropy-based uniformity measure.
        Higher entropy = more uniform = less redundancy.
        
        Args:
            distribution: Probability distribution tensor
            base: Logarithm base ('e', '2', '10')
        
        Returns:
            Entropy values (higher = more uniform)
        """
        # Ensure positive probabilities
        distribution = torch.clamp(distribution, min=1e-10)
        
        # Normalize to probability distribution
        distribution = distribution / distribution.sum(dim=-1, keepdim=True)
        
        if base == 'e':
            log_func = torch.log
        elif base == '2':
            log_func = torch.log2
        elif base == '10':
            log_func = torch.log10
        else:
            log_func = torch.log
        
        entropy = -torch.sum(distribution * log_func(distribution), dim=-1)
        return entropy
    
    @staticmethod
    def gini_uniformity(distribution: torch.Tensor) -> torch.Tensor:
        """
        Calculate Gini coefficient-based uniformity measure.
        Lower Gini = more uniform = less redundancy.
        
        Returns:
            1 - Gini coefficient (higher = more uniform)
        """
        # Sort values
        sorted_dist, _ = torch.sort(distribution, dim=-1)
        n = distribution.shape[-1]
        
        # Calculate Gini coefficient
        cumsum = torch.cumsum(sorted_dist, dim=-1)
        gini = (2 * torch.arange(1, n + 1, device=distribution.device).float() - n - 1)
        gini = gini.view(1, -1) if len(distribution.shape) > 1 else gini
        gini = torch.sum(gini * sorted_dist, dim=-1) / (n * torch.sum(sorted_dist, dim=-1))
        
        # Return uniformity measure (1 - Gini)
        return 1 - torch.abs(gini)
    
    @staticmethod
    def kl_divergence_from_uniform(distribution: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence from uniform distribution.
        Lower KL divergence = more uniform = less redundancy.
        
        Returns:
            -KL divergence (higher = more uniform)
        """
        # Normalize distribution
        distribution = distribution / distribution.sum(dim=-1, keepdim=True)
        
        # Uniform distribution
        uniform = torch.ones_like(distribution) / distribution.shape[-1]
        
        # KL divergence
        kl_div = torch.sum(distribution * torch.log(distribution / uniform + 1e-10), dim=-1)
        
        # Return negative KL divergence (higher = more uniform)
        return -kl_div
    
    @staticmethod
    def variance_uniformity(distribution: torch.Tensor) -> torch.Tensor:
        """
        Calculate variance-based uniformity measure.
        Lower variance = more uniform = less redundancy.
        
        Returns:
            -variance (higher = more uniform)
        """
        return -torch.var(distribution, dim=-1)
    
    @staticmethod
    def composite_uniformity(distribution: torch.Tensor, weights: List[float] = None) -> torch.Tensor:
        """
        Calculate composite uniformity measure combining multiple metrics.
        
        Args:
            distribution: Input distribution
            weights: Weights for [entropy, gini, kl_div, variance] measures
        
        Returns:
            Composite uniformity score (higher = more uniform)
        """
        if weights is None:
            weights = [0.4, 0.3, 0.2, 0.1]  # Emphasis on entropy
        
        # Calculate individual measures
        entropy_score = UniformityMeasures.entropy_uniformity(distribution)
        gini_score = UniformityMeasures.gini_uniformity(distribution)
        kl_score = UniformityMeasures.kl_divergence_from_uniform(distribution)
        var_score = UniformityMeasures.variance_uniformity(distribution)
        
        # Normalize scores to [0, 1] range
        def normalize_score(score):
            min_val = score.min()
            max_val = score.max()
            if max_val > min_val:
                return (score - min_val) / (max_val - min_val)
            return torch.ones_like(score) * 0.5
        
        entropy_norm = normalize_score(entropy_score)
        gini_norm = normalize_score(gini_score)
        kl_norm = normalize_score(kl_score)
        var_norm = normalize_score(var_score)
        
        # Weighted combination
        composite = (weights[0] * entropy_norm + 
                    weights[1] * gini_norm + 
                    weights[2] * kl_norm + 
                    weights[3] * var_norm)
        
        return composite

class UniformityBasedLayerBudgetAllocator:
    """
    Layer budget allocation based on distribution uniformity analysis.
    
    Core principle: Layers with more uniform KV distributions contain more information
    and should receive larger cache budgets.
    """
    
    def __init__(self, num_layers: int = 32, uniformity_method: str = 'composite'):
        self.num_layers = num_layers
        self.uniformity_method = uniformity_method
        self.layer_uniformity_history = []
        self.layer_preferences = None
        
    def calculate_layer_uniformity_scores(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Calculate uniformity scores for each layer based on KV cache distributions.
        
        Args:
            past_key_values: List of (key, value) tensors for each layer
            
        Returns:
            uniformity_scores: Tensor of shape [num_layers] with uniformity scores
        """
        uniformity_scores = []
        
        for layer_idx, (key_states, value_states) in enumerate(past_key_values):
            # Get key and value distributions
            key_dist = self._extract_distribution_features(key_states)
            value_dist = self._extract_distribution_features(value_states)
            
            # Calculate uniformity for both key and value
            if self.uniformity_method == 'entropy':
                key_uniformity = UniformityMeasures.entropy_uniformity(key_dist)
                value_uniformity = UniformityMeasures.entropy_uniformity(value_dist)
            elif self.uniformity_method == 'gini':
                key_uniformity = UniformityMeasures.gini_uniformity(key_dist)
                value_uniformity = UniformityMeasures.gini_uniformity(value_dist)
            elif self.uniformity_method == 'kl_divergence':
                key_uniformity = UniformityMeasures.kl_divergence_from_uniform(key_dist)
                value_uniformity = UniformityMeasures.kl_divergence_from_uniform(value_dist)
            elif self.uniformity_method == 'composite':
                key_uniformity = UniformityMeasures.composite_uniformity(key_dist)
                value_uniformity = UniformityMeasures.composite_uniformity(value_dist)
            else:
                raise ValueError(f"Unknown uniformity method: {self.uniformity_method}")
            
            # Combine key and value uniformity scores
            layer_uniformity = (key_uniformity + value_uniformity) / 2
            uniformity_scores.append(layer_uniformity.mean().item())
        
        return torch.tensor(uniformity_scores)
    
    def _extract_distribution_features(self, kv_states: torch.Tensor) -> torch.Tensor:
        """
        Extract distribution features from KV states for uniformity analysis.
        
        Args:
            kv_states: Tensor of shape [batch, num_heads, seq_len, head_dim]
            
        Returns:
            distribution_features: Tensor suitable for uniformity analysis
        """
        batch_size, num_heads, seq_len, head_dim = kv_states.shape
        
        # Method 1: Head-wise activation magnitudes
        head_magnitudes = torch.norm(kv_states, dim=-1)  # [batch, num_heads, seq_len]
        head_distributions = head_magnitudes.mean(dim=0)  # [num_heads, seq_len]
        
        # Method 2: Token-wise activation patterns
        token_patterns = kv_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        token_magnitudes = torch.norm(token_patterns, dim=-1)  # [batch, seq_len]
        
        # Method 3: Dimension-wise activation distributions
        dim_patterns = kv_states.permute(0, 3, 1, 2).reshape(batch_size, head_dim, -1)
        dim_magnitudes = torch.norm(dim_patterns, dim=-1)  # [batch, head_dim]
        
        # Combine features (use head-wise as primary feature for layer analysis)
        combined_features = head_distributions.flatten()  # [num_heads * seq_len]
        
        # Add positional encoding to capture sequence patterns
        positions = torch.arange(seq_len, device=kv_states.device).float()
        positional_weights = torch.sin(positions / 1000) + torch.cos(positions / 1000)
        
        # Weight by positional information
        weighted_features = []
        for head_idx in range(num_heads):
            head_feature = head_distributions[head_idx] * positional_weights
            weighted_features.append(head_feature)
        
        final_features = torch.cat(weighted_features)
        
        # Ensure positive values for uniformity analysis
        final_features = torch.abs(final_features) + 1e-8
        
        return final_features.unsqueeze(0) if len(final_features.shape) == 1 else final_features
    
    def allocate_layer_budgets(self, total_budget: int, uniformity_scores: torch.Tensor, 
                             min_budget_per_layer: int = 8) -> List[int]:
        """
        Allocate cache budgets to layers based on uniformity scores.
        
        Args:
            total_budget: Total cache budget to distribute
            uniformity_scores: Uniformity scores for each layer
            min_budget_per_layer: Minimum budget per layer
            
        Returns:
            budget_allocation: List of budget allocations per layer
        """
        # Ensure minimum budget for all layers
        min_total = min_budget_per_layer * self.num_layers
        if total_budget < min_total:
            raise ValueError(f"Total budget {total_budget} too small for minimum allocation {min_total}")
        
        # Available budget for distribution based on uniformity
        distributable_budget = total_budget - min_total
        
        # Normalize uniformity scores to probabilities
        uniformity_probs = torch.softmax(uniformity_scores, dim=0)
        
        # Allocate distributable budget proportionally to uniformity
        distributed_budgets = (uniformity_probs * distributable_budget).int()
        
        # Add minimum budget
        final_budgets = distributed_budgets + min_budget_per_layer
        
        # Handle rounding errors
        budget_sum = final_budgets.sum().item()
        if budget_sum != total_budget:
            diff = total_budget - budget_sum
            if diff > 0:
                # Add excess to highest uniformity layers
                _, top_indices = torch.topk(uniformity_scores, k=abs(diff))
                for idx in top_indices:
                    final_budgets[idx] += 1
            elif diff < 0:
                # Remove excess from lowest uniformity layers
                _, bottom_indices = torch.topk(uniformity_scores, k=abs(diff), largest=False)
                for idx in bottom_indices:
                    if final_budgets[idx] > min_budget_per_layer:
                        final_budgets[idx] -= 1
        
        return final_budgets.tolist()
    
    def update_layer_preferences(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Update layer preferences based on recent uniformity patterns."""
        current_uniformity = self.calculate_layer_uniformity_scores(past_key_values)
        self.layer_uniformity_history.append(current_uniformity)
        
        # Keep only recent history
        if len(self.layer_uniformity_history) > 10:
            self.layer_uniformity_history.pop(0)
        
        # Update preferences with exponential moving average
        if self.layer_preferences is None:
            self.layer_preferences = current_uniformity
        else:
            alpha = 0.3  # Learning rate
            self.layer_preferences = (1 - alpha) * self.layer_preferences + alpha * current_uniformity

class UniformityBasedTokenScorer:
    """
    Token importance scoring based on distribution uniformity analysis.
    
    Core principle: Tokens that contribute to uniform distributions contain more
    information and should be preserved in the cache.
    """
    
    def __init__(self, window_size: int = 32, uniformity_method: str = 'composite'):
        self.window_size = window_size
        self.uniformity_method = uniformity_method
        
    def calculate_token_uniformity_contributions(self, key_states: torch.Tensor, 
                                               value_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate how much each token contributes to distribution uniformity.
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            token_scores: [batch, num_heads, seq_len] - higher scores = more important
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Skip recent tokens (preserve window)
        if seq_len <= self.window_size:
            return torch.ones(batch_size, num_heads, seq_len, device=key_states.device) * 0.5
        
        scoreable_len = seq_len - self.window_size
        key_scoreable = key_states[:, :, :scoreable_len, :]
        value_scoreable = value_states[:, :, :scoreable_len, :]
        
        token_scores = []
        
        for token_idx in range(scoreable_len):
            # Calculate uniformity with and without this token
            full_uniformity = self._calculate_sequence_uniformity(key_scoreable, value_scoreable)
            
            # Create version without current token
            if token_idx == 0:
                reduced_key = key_scoreable[:, :, 1:, :]
                reduced_value = value_scoreable[:, :, 1:, :]
            elif token_idx == scoreable_len - 1:
                reduced_key = key_scoreable[:, :, :-1, :]
                reduced_value = value_scoreable[:, :, :-1, :]
            else:
                reduced_key = torch.cat([
                    key_scoreable[:, :, :token_idx, :],
                    key_scoreable[:, :, token_idx+1:, :]
                ], dim=2)
                reduced_value = torch.cat([
                    value_scoreable[:, :, :token_idx, :],
                    value_scoreable[:, :, token_idx+1:, :]
                ], dim=2)
            
            reduced_uniformity = self._calculate_sequence_uniformity(reduced_key, reduced_value)
            
            # Token contribution = how much uniformity decreases without it
            token_contribution = full_uniformity - reduced_uniformity
            token_scores.append(token_contribution)
        
        # Stack and normalize
        token_scores = torch.stack(token_scores, dim=-1)  # [batch, num_heads, scoreable_len]
        
        # Apply positional bias (prefer older tokens for eviction)
        position_bias = torch.linspace(1.0, 0.5, scoreable_len, device=key_states.device)
        position_bias = position_bias.view(1, 1, -1).expand(batch_size, num_heads, -1)
        
        # Combine uniformity contribution with position bias
        final_scores = token_scores * position_bias
        
        return final_scores
    
    def _calculate_sequence_uniformity(self, key_states: torch.Tensor, 
                                     value_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate uniformity of the entire sequence representation.
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            uniformity_score: [batch, num_heads] - higher = more uniform
        """
        # Combine key and value representations
        combined_states = torch.cat([key_states, value_states], dim=-1)
        
        # Calculate activation patterns across sequence
        activation_patterns = torch.norm(combined_states, dim=-1)  # [batch, num_heads, seq_len]
        
        # Calculate uniformity of activation patterns
        if self.uniformity_method == 'entropy':
            uniformity = UniformityMeasures.entropy_uniformity(activation_patterns)
        elif self.uniformity_method == 'gini':
            uniformity = UniformityMeasures.gini_uniformity(activation_patterns)
        elif self.uniformity_method == 'kl_divergence':
            uniformity = UniformityMeasures.kl_divergence_from_uniform(activation_patterns)
        elif self.uniformity_method == 'composite':
            uniformity = UniformityMeasures.composite_uniformity(activation_patterns)
        else:
            uniformity = UniformityMeasures.entropy_uniformity(activation_patterns)
        
        return uniformity
    
    def get_fast_uniformity_scores(self, key_states: torch.Tensor, 
                                 value_states: torch.Tensor) -> torch.Tensor:
        """
        Fast approximation of uniformity-based token scoring.
        
        Uses efficient approximations for real-time inference.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if seq_len <= self.window_size:
            return torch.ones(batch_size, num_heads, seq_len, device=key_states.device) * 0.5
        
        scoreable_len = seq_len - self.window_size
        
        # Fast approximation: token impact on distribution variance
        key_scoreable = key_states[:, :, :scoreable_len, :]
        value_scoreable = value_states[:, :, :scoreable_len, :]
        
        # Calculate token-wise activation magnitudes
        key_magnitudes = torch.norm(key_scoreable, dim=-1)  # [batch, num_heads, scoreable_len]
        value_magnitudes = torch.norm(value_scoreable, dim=-1)
        
        # Measure each token's contribution to overall distribution uniformity
        # Tokens that deviate from mean contribute less to uniformity
        key_mean = key_magnitudes.mean(dim=-1, keepdim=True)
        value_mean = value_magnitudes.mean(dim=-1, keepdim=True)
        
        key_deviations = torch.abs(key_magnitudes - key_mean)
        value_deviations = torch.abs(value_magnitudes - value_mean)
        
        # Lower deviation = better contribution to uniformity = higher importance
        uniformity_scores = 1.0 / (1.0 + key_deviations + value_deviations)
        
        # Apply position bias
        position_weights = torch.linspace(1.0, 0.6, scoreable_len, device=key_states.device)
        position_weights = position_weights.view(1, 1, -1).expand(batch_size, num_heads, -1)
        
        final_scores = uniformity_scores * position_weights
        
        return final_scores

class CorrelatedUniformityOptimizer:
    """
    Unified optimizer that correlates layer budget allocation and token scoring
    through shared uniformity measures.
    """
    
    def __init__(self, num_layers: int = 32, window_size: int = 32, 
                 uniformity_method: str = 'composite'):
        self.layer_allocator = UniformityBasedLayerBudgetAllocator(num_layers, uniformity_method)
        self.token_scorer = UniformityBasedTokenScorer(window_size, uniformity_method)
        self.correlation_weight = 0.3  # Weight for correlation between layers and tokens
        
    def get_correlated_allocations(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], 
                                 total_budget: int) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Get correlated layer budgets and token scores based on uniformity analysis.
        
        Args:
            past_key_values: KV cache states for all layers
            total_budget: Total cache budget
            
        Returns:
            layer_budgets: Budget allocation per layer
            token_scores: Token importance scores per layer
        """
        # Calculate layer uniformity scores
        layer_uniformity = self.layer_allocator.calculate_layer_uniformity_scores(past_key_values)
        
        # Calculate token scores for each layer
        token_scores = []
        for layer_idx, (key_states, value_states) in enumerate(past_key_values):
            layer_token_scores = self.token_scorer.get_fast_uniformity_scores(key_states, value_states)
            
            # Correlate with layer uniformity
            layer_weight = layer_uniformity[layer_idx].item()
            correlated_scores = layer_token_scores * (1.0 + self.correlation_weight * layer_weight)
            
            token_scores.append(correlated_scores)
        
        # Allocate budgets based on layer uniformity
        layer_budgets = self.layer_allocator.allocate_layer_budgets(total_budget, layer_uniformity)
        
        # Update preferences for future inference
        self.layer_allocator.update_layer_preferences(past_key_values)
        
        return layer_budgets, token_scores
    
    def get_uniformity_metrics(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """Get detailed uniformity metrics for analysis."""
        layer_uniformity = self.layer_allocator.calculate_layer_uniformity_scores(past_key_values)
        
        metrics = {
            'layer_uniformity_scores': layer_uniformity.tolist(),
            'overall_uniformity': layer_uniformity.mean().item(),
            'uniformity_variance': layer_uniformity.var().item(),
            'max_uniformity_layer': layer_uniformity.argmax().item(),
            'min_uniformity_layer': layer_uniformity.argmin().item()
        }
        
        return metrics