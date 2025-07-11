import torch
import numpy as np
import torch.nn.functional as F
class CompressConfig:
    def __init__(self, compress=False, cascading=False, cache_size=1024, window_size=32, hyper=None, poly=False, 
                 fast_mode=True, info_theory_threshold=32):
        self.compress = compress
        self.cascading = cascading
        self.cache_size = cache_size
        self.window_size = window_size
        self.hyper = hyper
        self.poly = poly
        self.fast_mode = fast_mode  # Enable fast approximations
        self.info_theory_threshold = info_theory_threshold  # Min sequence length for enhanced scoring
    
    def __str__(self):
        return f"Config(cache_size={self.cache_size}, window_size={self.window_size}, hyper={self.hyper})"

def calculate_entropy(attention_scores):
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
    entropy= entropy.to(dtype=torch.float32)
    return entropy

def calculate_mutual_information_fast(attention_weights):
    """
    Fast approximation of mutual information using attention entropy.
    Optimized version that avoids softmax computation.
    """
    # Use attention variance as proxy for mutual information (much faster)
    # Higher variance = higher information content
    return attention_weights.var(dim=-1)

def calculate_key_entropy_fast(key_states):
    """
    Fast approximation of key entropy using norm variance.
    Avoids expensive matrix multiplication.
    """
    # Use norm variance as entropy proxy - much faster than full similarity matrix
    key_norms = torch.norm(key_states, dim=-1)  # [bsz, heads, seq_len]
    entropy_proxy = torch.var(key_norms, dim=-1).mean(dim=-1)  # [bsz]
    
    return entropy_proxy

# Keep original functions for fallback
def calculate_mutual_information(key_states, query_states, attention_weights):
    """
    Calculate mutual information between key and query states based on attention patterns.
    Uses information-theoretic approximation: I(K,Q) ≈ H(A) where A is attention distribution
    """
    # Normalize attention weights to probability distribution
    attention_probs = torch.softmax(attention_weights, dim=-1)
    
    # Calculate entropy of attention distribution (proxy for mutual information)
    attention_entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-10), dim=-1)
    
    return attention_entropy

def calculate_key_entropy(key_states):
    """
    Estimate information entropy of key representations using attention variance.
    Based on connection: H(K) ∝ Var(attention patterns from K)
    """
    # Calculate pairwise similarities (simplified information content measure)
    key_flat = key_states.flatten(-2)  # [bsz, heads, seq_len, head_dim*groups]
    key_norm = torch.nn.functional.normalize(key_flat, dim=-1)
    
    # Self-attention matrix as information content proxy
    sim_matrix = torch.matmul(key_norm, key_norm.transpose(-2, -1))
    
    # Entropy approximation: higher variance in similarity = higher information content
    entropy_proxy = torch.var(sim_matrix, dim=-1).mean(dim=-1)
    
    return entropy_proxy

def information_theoretic_token_score_fast(attention_scores, key_states, query_states, gamma=200.0):
    """
    Fast approximation of information-theoretic token scoring.
    Uses lightweight approximations to maintain speed while preserving benefits.
    """
    # Original attention statistics (fast)
    attn_mean = attention_scores.mean(dim=-2)  # [bsz, heads, key_len]
    
    # Handle variance calculation for single query case
    if attention_scores.shape[-2] > 1:
        attn_var = attention_scores.var(dim=-2)    # [bsz, heads, key_len]
    else:
        attn_var = torch.zeros_like(attn_mean)
    
    # Fast information-theoretic components
    mutual_info = calculate_mutual_information_fast(attention_scores)  # [bsz, heads, key_len]
    key_entropy = calculate_key_entropy_fast(key_states)  # [bsz]
    
    # Ensure mutual_info has same shape as attn_mean
    if mutual_info.shape[-1] != attn_mean.shape[-1]:
        # If shapes don't match, use attention variance as mutual info proxy
        mutual_info = attn_var
    
    # Expand key_entropy from [bsz] to [bsz, heads, key_len] 
    key_entropy_expanded = key_entropy.view(-1, 1, 1).expand(attn_mean.shape)
    
    # Lightweight information score with reduced weights for speed
    # α = 0.8 (attention weight), β = 0.15 (mutual info weight), γ = 0.05 (entropy weight)
    info_score = (0.8 * (attn_mean + gamma * attn_var) + 
                 0.15 * mutual_info + 
                 0.05 * key_entropy_expanded)
    
    return info_score

def information_theoretic_token_score(attention_scores, key_states, query_states, gamma=200.0, fast=True):
    """
    Enhanced token scoring using information-theoretic principles.
    
    Args:
        attention_scores: [bsz, heads, query_len, key_len] attention weights
        key_states: [bsz, heads, key_len, head_dim] key representations  
        query_states: [bsz, heads, query_len, head_dim] query representations
        gamma: weighting factor for attention variance
        fast: use fast approximations (default True for speed)
    """
    if fast:
        return information_theoretic_token_score_fast(attention_scores, key_states, query_states, gamma)
    
    # Original full computation (for reference/debugging)
    attn_mean = attention_scores.mean(dim=-2)  # [bsz, heads, key_len]
    
    if attention_scores.shape[-2] > 1:
        attn_var = attention_scores.var(dim=-2)    # [bsz, heads, key_len]
    else:
        attn_var = torch.zeros_like(attn_mean)
    
    # Information-theoretic components
    mutual_info = calculate_mutual_information(key_states, query_states, attention_scores)  # [bsz, heads, query_len]
    key_entropy = calculate_key_entropy(key_states)  # [bsz]
    
    # Align dimensions for combination
    mutual_info_avg = mutual_info.mean(dim=-1, keepdim=True).expand(-1, -1, attn_mean.shape[-1])
    key_entropy_expanded = key_entropy.view(-1, 1, 1).expand(attn_mean.shape)
    
    # Unified information score
    info_score = (0.6 * (attn_mean + gamma * attn_var) + 
                 0.3 * mutual_info_avg + 
                 0.1 * key_entropy_expanded)
    
    return info_score

def adjust_budgets(budget_list, total_budget, seq_len, layer_nums):

    budget_list = np.array(budget_list, dtype=int)
    # Limit the budget of all layers to not exceed seq_len
    excess = np.maximum(budget_list - seq_len, 0)
    budget_list = np.minimum(budget_list, seq_len)

    # Adjust excess budget
    total_excess = np.sum(excess)

    if total_excess > 0:

        valid_indices = budget_list < seq_len
        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            
            distribute_per_layer = total_excess // num_valid
            remainder = total_excess % num_valid

            budget_list[valid_indices] += distribute_per_layer
            budget_list[np.where(valid_indices)[0][:remainder]] += 1

    # Ensure total budget equals total_budget
    current_total_budget = np.sum(budget_list)
    budget_diff = total_budget - current_total_budget

    if budget_diff != 0:
        if budget_diff > 0:
            valid_indices = budget_list < seq_len  
        else:
            valid_indices = budget_list > 1  

        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            adjust_per_layer = abs(budget_diff) // num_valid
            remainder = abs(budget_diff) % num_valid

            if budget_diff > 0:
                budget_list[valid_indices] += adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] += 1
            else:
                budget_list[valid_indices] -= adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] -= 1

    return budget_list.tolist()


def get_scores_with_kv_fusion(key_states: torch.Tensor, value_states: torch.Tensor, alpha: float = 1.0, window_size=32) -> torch.Tensor:
    """
    DATI (Dispersion-Augmented Token Importance) - Optimized fast dispersion scoring
    
    High-performance implementation using vectorized operations and approximations:
    1. Fast centroid-based dispersion (O(n) instead of O(n²))
    2. Efficient statistical measures
    3. Reduced computational overhead while maintaining dispersion benefits
    
    Args:
        key_states: Tensor of shape [bsz, num_heads, seq_len, head_dim]
        value_states: Tensor of shape [bsz, num_heads, seq_len, head_dim] 
        alpha: weight for value score (default: 1.0)

    Returns:
        final_score: Tensor of shape [bsz, num_heads_grouped, seq_len_scoreable] — fast dispersion scores
    """
    bsz, num_heads, seq_len, head_dim = key_states.shape
    
    # Only score non-window tokens (exclude recent tokens from eviction)
    if seq_len <= window_size:
        return torch.zeros(bsz, 8, seq_len, device=key_states.device)
    
    scoreable_len = seq_len - window_size
    key_scoreable = key_states[:, :, :scoreable_len, :]
    value_scoreable = value_states[:, :, :scoreable_len, :]
    
    # ========== Fast Method 1: Multi-Scale Centroid Dispersion ==========
    def calculate_fast_dispersion(states: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast dispersion calculation using vectorized operations.
        O(n) complexity with good dispersion approximation.
        
        Args:
            states: [bsz, num_heads, seq_len, head_dim]
            
        Returns:
            dispersion_scores: [bsz, num_heads, seq_len]
        """
        bsz, num_heads, seq_len, head_dim = states.shape
        
        # Global centroid distance (very fast)
        global_centroid = states.mean(dim=2, keepdim=True)  # [bsz, num_heads, 1, head_dim]
        global_distances = torch.norm(states - global_centroid, dim=-1)  # [bsz, num_heads, seq_len]
        
        # Fast local dispersion using convolution-based sliding window
        # This is much faster than explicit loops
        if seq_len > 64:  # Only for longer sequences where it matters
            # Reshape for 1D convolution: [bsz * num_heads, head_dim, seq_len]
            states_conv = states.permute(0, 1, 3, 2).reshape(bsz * num_heads, head_dim, seq_len)
            
            # Local window size (adaptive)
            local_window = min(16, max(4, seq_len // 32))
            
            # Use unfold to create sliding windows efficiently
            if seq_len >= local_window:
                # Create sliding windows: [bsz * num_heads, head_dim, seq_len - window + 1, window]
                windowed = states_conv.unfold(2, local_window, 1)
                # Calculate local centroids: [bsz * num_heads, head_dim, seq_len - window + 1]
                local_centroids = windowed.mean(dim=-1)
                
                # Pad to match original length
                pad_left = local_window // 2
                pad_right = local_window - 1 - pad_left
                local_centroids = F.pad(local_centroids, (pad_left, pad_right), mode='replicate')
                
                # Reshape back and calculate distances
                local_centroids = local_centroids.reshape(bsz, num_heads, head_dim, seq_len).permute(0, 1, 3, 2)
                local_distances = torch.norm(states - local_centroids, dim=-1)
            else:
                local_distances = global_distances  # Fallback for very short sequences
        else:
            local_distances = global_distances
        
        # Fast statistical measures
        # Token variance (captures internal dispersion)
        token_variance = torch.var(states, dim=-1)  # [bsz, num_heads, seq_len]
        
        # Distance from running mean (temporal context)
        running_mean = torch.cumsum(states, dim=2) / torch.arange(1, seq_len + 1, device=states.device).view(1, 1, -1, 1)
        temporal_distances = torch.norm(states - running_mean, dim=-1)
        
        # Combine measures efficiently
        dispersion_score = (0.4 * global_distances + 
                           0.3 * local_distances + 
                           0.2 * token_variance + 
                           0.1 * temporal_distances)
        
        return dispersion_score
    
    # ========== Fast Method 2: Approximated k-NN via Sampling ==========
    def calculate_sampled_knn_dispersion(states: torch.Tensor, sample_ratio: float = 0.1) -> torch.Tensor:
        """
        Fast k-NN approximation using random sampling.
        Reduces complexity from O(n²) to O(n*s) where s << n.
        
        Args:
            states: [bsz, num_heads, seq_len, head_dim]
            sample_ratio: fraction of tokens to sample for distance calculation
            
        Returns:
            dispersion_scores: [bsz, num_heads, seq_len]
        """
        bsz, num_heads, seq_len, head_dim = states.shape
        
        # Number of samples (at least 5, at most seq_len // 2)
        num_samples = max(5, min(seq_len // 2, int(seq_len * sample_ratio)))
        
        # Random sampling indices
        sample_indices = torch.randperm(seq_len, device=states.device)[:num_samples]
        sampled_states = states[:, :, sample_indices, :]  # [bsz, num_heads, num_samples, head_dim]
        
        # Calculate distances to sampled tokens efficiently
        # Broadcasting: [bsz, num_heads, seq_len, 1, head_dim] - [bsz, num_heads, 1, num_samples, head_dim]
        states_expanded = states.unsqueeze(3)  # [bsz, num_heads, seq_len, 1, head_dim]
        sampled_expanded = sampled_states.unsqueeze(2)  # [bsz, num_heads, 1, num_samples, head_dim]
        
        distances = torch.norm(states_expanded - sampled_expanded, dim=-1)  # [bsz, num_heads, seq_len, num_samples]
        
        # Take mean distance to samples as dispersion proxy
        dispersion_scores = distances.mean(dim=-1)  # [bsz, num_heads, seq_len]
        
        return dispersion_scores
    
    # ========== Adaptive Method Selection ==========
    # Choose method based on sequence length for optimal speed/quality tradeoff
    if scoreable_len > 1024:
        # Very long sequences: use fastest method
        key_dispersion = calculate_fast_dispersion(key_scoreable)
        value_dispersion = calculate_fast_dispersion(value_scoreable)
    elif scoreable_len > 256:
        # Medium sequences: use sampling approximation
        key_dispersion = calculate_sampled_knn_dispersion(key_scoreable, sample_ratio=0.1)
        value_dispersion = calculate_sampled_knn_dispersion(value_scoreable, sample_ratio=0.1)
    else:
        # Short sequences: can afford more accurate computation
        key_dispersion = calculate_fast_dispersion(key_scoreable)
        value_dispersion = calculate_fast_dispersion(value_scoreable)
        
        # Add extra accuracy for short sequences
        key_sampled = calculate_sampled_knn_dispersion(key_scoreable, sample_ratio=0.3)
        value_sampled = calculate_sampled_knn_dispersion(value_scoreable, sample_ratio=0.3)
        
        # Blend for better accuracy
        key_dispersion = 0.7 * key_dispersion + 0.3 * key_sampled
        value_dispersion = 0.7 * value_dispersion + 0.3 * value_sampled
    
    # ========== Fast Normalization and Combination ==========
    # Efficient per-head normalization with fallback
    def fast_normalize(x: torch.Tensor) -> torch.Tensor:
        try:
            # Convert to float32 for quantile computation if needed
            x_float = x.float() if x.dtype in [torch.float16, torch.bfloat16] else x
            
            # Use quantile-based normalization for robustness and speed
            q10 = torch.quantile(x_float, 0.1, dim=-1, keepdim=True)
            q90 = torch.quantile(x_float, 0.9, dim=-1, keepdim=True)
            normalized = torch.clamp((x_float - q10) / (q90 - q10 + 1e-8), 0, 1)
            
            # Convert back to original dtype
            return normalized.to(x.dtype)
        except:
            # Fallback to min-max normalization if quantile fails
            min_val = x.amin(dim=-1, keepdim=True)
            max_val = x.amax(dim=-1, keepdim=True)
            return (x - min_val) / (max_val - min_val + 1e-8)
    
    key_norm = fast_normalize(key_dispersion)
    value_norm = fast_normalize(value_dispersion)
    
    # Fast consistency check (simplified)
    consistency = 1.0 - 0.5 * torch.abs(key_norm - value_norm)
    
    # Final combination
    final_dispersion = 0.45 * key_norm + 0.45 * value_norm + 0.1 * consistency
    
    # Fast position bias (vectorized)
    position_bias = torch.linspace(1.05, 0.95, scoreable_len, device=key_states.device)
    position_bias = position_bias.view(1, 1, -1)
    final_dispersion = final_dispersion * position_bias
    
    # Efficient head grouping for LLaMA
    if num_heads == 32:
        final_dispersion = final_dispersion.view(bsz, 8, 4, scoreable_len).mean(dim=2)
    
    return final_dispersion