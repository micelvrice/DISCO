import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import warnings
import argparse
import random
import json
import os
from typing import List, Tuple, Dict, Optional
from numpy.polynomial import Polynomial

class AdaptiveLayerBudgetOptimizer:
    """
    Enhanced layer budget allocation with adaptive online learning and temporal dynamics.
    
    Key improvements:
    1. Online adaptation based on recent inference patterns
    2. Multi-modal scoring combining PCA, entropy, and attention patterns
    3. Temporal decay for better sequence modeling
    4. Adaptive budget reallocation during inference
    """
    
    def __init__(self, num_layers: int = 32, adaptation_rate: float = 0.1, 
                 temporal_decay: float = 0.95, min_budget_ratio: float = 0.05):
        self.num_layers = num_layers
        self.adaptation_rate = adaptation_rate
        self.temporal_decay = temporal_decay
        self.min_budget_ratio = min_budget_ratio
        
        # Initialize base layer preferences from offline analysis
        self.base_preferences = None
        self.online_preferences = None
        self.temporal_weights = None
        
        # Track online statistics
        self.layer_usage_history = []
        self.attention_concentration_history = []
        self.inference_count = 0
        
    def load_base_preferences(self, coeffs_path: str):
        """Load offline-computed base layer preferences."""
        coeffs = np.load(coeffs_path)
        poly_func = Polynomial(coeffs)
        
        # Generate base preferences for all layers
        layer_indices = np.arange(self.num_layers)
        self.base_preferences = poly_func(layer_indices)
        
        # Normalize to valid probability distribution
        self.base_preferences = np.maximum(self.base_preferences, 0)
        self.base_preferences = self.base_preferences / self.base_preferences.sum()
        
        # Initialize online preferences with base values
        self.online_preferences = self.base_preferences.copy()
        self.temporal_weights = np.ones(self.num_layers)
        
    def calculate_enhanced_dispersion_score(self, past_key_values, layer_idx: int) -> float:
        """
        Enhanced dispersion scoring with multiple information-theoretic measures.
        
        Improvements:
        1. Combines PCA variance with spectral entropy
        2. Considers key-value correlation patterns
        3. Accounts for attention head diversity
        """
        key_states = past_key_values[layer_idx][0].squeeze(0)  # [num_heads, seq_len, head_dim]
        value_states = past_key_values[layer_idx][1].squeeze(0)
        
        seq_len, num_heads, head_dim = key_states.shape[1], key_states.shape[0], key_states.shape[2]
        
        # Traditional PCA-based dispersion
        key_flat = key_states.permute(1, 0, 2).reshape(seq_len, -1)  # [seq_len, num_heads*head_dim]
        value_flat = value_states.permute(1, 0, 2).reshape(seq_len, -1)
        
        # Multi-component PCA for better information capture
        pca_components = min(4, seq_len - 1) if seq_len > 1 else 1
        
        if seq_len > 1:
            pca_key = PCA(n_components=pca_components)
            pca_value = PCA(n_components=pca_components)
            
            try:
                key_transformed = pca_key.fit_transform(key_flat.cpu().numpy())
                value_transformed = pca_value.fit_transform(value_flat.cpu().numpy())
                
                # Enhanced dispersion: combine explained variance with entropy
                key_variance = pca_key.explained_variance_.sum()
                value_variance = pca_value.explained_variance_.sum()
                
                # Spectral entropy of principal components
                key_entropy = self._calculate_spectral_entropy(pca_key.explained_variance_)
                value_entropy = self._calculate_spectral_entropy(pca_value.explained_variance_)
                
                # Key-Value correlation (measures multimodal information)
                kv_correlation = np.corrcoef(key_transformed.flatten(), value_transformed.flatten())[0, 1]
                kv_correlation = np.nan_to_num(kv_correlation, nan=0.0)
                
                # Attention head diversity (measures information distribution across heads)
                head_diversity = self._calculate_head_diversity(key_states, value_states)
                
                # Combined score with weighted components
                dispersion_score = (0.4 * (key_variance + value_variance) + 
                                  0.3 * (key_entropy + value_entropy) +
                                  0.2 * abs(kv_correlation) +
                                  0.1 * head_diversity)
                
            except Exception as e:
                # Fallback to simple variance if PCA fails
                dispersion_score = torch.var(key_flat).item() + torch.var(value_flat).item()
        else:
            dispersion_score = torch.var(key_flat).item() + torch.var(value_flat).item()
            
        return dispersion_score
    
    def _calculate_spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """Calculate entropy of eigenvalue distribution."""
        # Normalize eigenvalues to probability distribution
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        probs = eigenvalues / eigenvalues.sum()
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    def _calculate_head_diversity(self, key_states: torch.Tensor, value_states: torch.Tensor) -> float:
        """Calculate diversity of information across attention heads."""
        # Compute head-wise centroids
        key_centroids = key_states.mean(dim=1)  # [num_heads, head_dim]
        value_centroids = value_states.mean(dim=1)
        
        # Calculate pairwise distances between heads
        key_distances = torch.pdist(key_centroids).cpu().numpy()
        value_distances = torch.pdist(value_centroids).cpu().numpy()
        
        # Diversity is average distance between heads
        diversity = (key_distances.mean() + value_distances.mean()) / 2
        return diversity
    
    def update_online_preferences(self, layer_attention_patterns: List[torch.Tensor], 
                                layer_cache_usage: List[int]):
        """
        Update layer preferences based on recent inference patterns.
        
        Args:
            layer_attention_patterns: List of attention weights for each layer
            layer_cache_usage: Number of tokens actually used from cache per layer
        """
        self.inference_count += 1
        
        # Calculate attention concentration per layer
        attention_concentrations = []
        for layer_idx, attn_pattern in enumerate(layer_attention_patterns):
            if attn_pattern is not None:
                # Measure attention concentration (inverse of entropy)
                attn_probs = torch.softmax(attn_pattern, dim=-1)
                attn_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10))
                concentration = 1.0 / (1.0 + attn_entropy.item())  # Higher concentration = lower entropy
                attention_concentrations.append(concentration)
            else:
                attention_concentrations.append(0.5)  # Default neutral value
        
        # Update online preferences with exponential moving average
        attention_concentrations = np.array(attention_concentrations)
        cache_usage_normalized = np.array(layer_cache_usage) / max(max(layer_cache_usage), 1)
        
        # Combined signal: attention concentration + cache usage efficiency
        online_signal = 0.7 * attention_concentrations + 0.3 * cache_usage_normalized
        
        # Update with adaptive learning rate (higher for early inferences)
        current_rate = self.adaptation_rate * (1.0 + np.exp(-self.inference_count / 100))
        
        self.online_preferences = ((1 - current_rate) * self.online_preferences + 
                                 current_rate * online_signal)
        
        # Apply temporal decay
        self.temporal_weights *= self.temporal_decay
        self.temporal_weights += (1 - self.temporal_decay) * online_signal
        
        # Renormalize
        self.online_preferences = np.maximum(self.online_preferences, self.min_budget_ratio)
        self.online_preferences = self.online_preferences / self.online_preferences.sum()
        
    def get_adaptive_budget_allocation(self, total_budget: int, seq_len: int) -> List[int]:
        """
        Get adaptive budget allocation combining base preferences with online learning.
        
        Args:
            total_budget: Total cache budget across all layers
            seq_len: Current sequence length
            
        Returns:
            List of budget allocations per layer
        """
        if self.base_preferences is None:
            # Fallback to uniform allocation
            base_budget = total_budget // self.num_layers
            return [base_budget] * self.num_layers
        
        # Combine base preferences with online adaptation and temporal weighting
        alpha = min(0.7, self.inference_count / (self.inference_count + 50))  # Gradually increase online weight
        beta = 0.3  # Temporal weight factor
        
        combined_preferences = ((1 - alpha) * self.base_preferences + 
                              alpha * self.online_preferences + 
                              beta * self.temporal_weights)
        
        # Renormalize
        combined_preferences = np.maximum(combined_preferences, self.min_budget_ratio)
        combined_preferences = combined_preferences / combined_preferences.sum()
        
        # Allocate budget based on combined preferences
        budget_allocation = (combined_preferences * total_budget).astype(int)
        
        # Ensure minimum budget per layer and handle rounding
        min_budget = max(1, int(total_budget * self.min_budget_ratio / self.num_layers))
        budget_allocation = np.maximum(budget_allocation, min_budget)
        
        # Adjust for sequence length constraints
        budget_allocation = np.minimum(budget_allocation, seq_len)
        
        # Redistribute excess budget
        total_allocated = budget_allocation.sum()
        if total_allocated != total_budget:
            excess = total_budget - total_allocated
            if excess > 0:
                # Distribute excess proportionally to high-preference layers
                high_pref_layers = np.argsort(combined_preferences)[-abs(excess):]
                for layer_idx in high_pref_layers:
                    if excess > 0:
                        budget_allocation[layer_idx] += 1
                        excess -= 1
            elif excess < 0:
                # Remove from low-preference layers
                low_pref_layers = np.argsort(combined_preferences)[:abs(excess)]
                for layer_idx in low_pref_layers:
                    if budget_allocation[layer_idx] > min_budget and excess < 0:
                        budget_allocation[layer_idx] -= 1
                        excess += 1
        
        return budget_allocation.tolist()
    
    def get_layer_importance_scores(self) -> np.ndarray:
        """Return current layer importance scores for analysis."""
        if self.online_preferences is not None:
            return self.online_preferences.copy()
        elif self.base_preferences is not None:
            return self.base_preferences.copy()
        else:
            return np.ones(self.num_layers) / self.num_layers

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama3.1-8b-128k", 
                        choices=["Llama3-8b-Instruct-8k",  
                                 "llama3.1-8b-128k", 
                                 "Mistral-7b-Instruct-v0.3-32k"])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="qasper")
    parser.add_argument('--pca_components', type=int, default=4)  # Increased for better information capture
    parser.add_argument('--num_samples', type=int, default=50)  # Increased sample size
    parser.add_argument('--save_online_model', action='store_true', help="Save adaptive model for online use")
    return parser.parse_args(args)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize adaptive optimizer
    optimizer = AdaptiveLayerBudgetOptimizer(num_layers=32)
    
    # Load existing base preferences if available
    base_coeffs_path = f"full_fit_llama3_combined15_pca4.npy"
    if os.path.exists(base_coeffs_path):
        optimizer.load_base_preferences(base_coeffs_path)
        print("Loaded base layer preferences from existing model")
    
    print(f"Enhanced layer budget allocation with {args.pca_components} PCA components")
    print(f"Adaptive learning enabled with {args.num_samples} samples")
    
    # Example usage of adaptive budget allocation
    sample_budget = 1024
    sample_seq_len = 2048
    
    adaptive_allocation = optimizer.get_adaptive_budget_allocation(sample_budget, sample_seq_len)
    print(f"Sample adaptive allocation: {adaptive_allocation[:5]}... (first 5 layers)")
    print(f"Total budget used: {sum(adaptive_allocation)}/{sample_budget}")
    
    if args.save_online_model:
        # Save the adaptive optimizer for online use
        torch.save({
            'base_preferences': optimizer.base_preferences,
            'online_preferences': optimizer.online_preferences,
            'temporal_weights': optimizer.temporal_weights,
            'inference_count': optimizer.inference_count,
            'num_layers': optimizer.num_layers,
            'adaptation_rate': optimizer.adaptation_rate,
            'temporal_decay': optimizer.temporal_decay,
            'min_budget_ratio': optimizer.min_budget_ratio
        }, f"adaptive_layer_optimizer_{args.model}.pt")
        print("Saved adaptive layer optimizer for online inference")