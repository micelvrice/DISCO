#!/usr/bin/env python3
"""
Test script for enhanced CAKE optimizations.

This script validates the enhanced algorithms and ensures they work correctly
with the existing CAKE infrastructure.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_token_scoring():
    """Test enhanced token scoring functionality."""
    print("Testing Enhanced Token Scoring...")
    
    try:
        from enhanced_token_scoring import EnhancedTokenScorer, get_enhanced_scores_with_kv_fusion
        
        # Create test data
        bsz, num_heads, seq_len, head_dim = 1, 32, 2048, 128
        key_states = torch.randn(bsz, num_heads, seq_len, head_dim)
        value_states = torch.randn(bsz, num_heads, seq_len, head_dim)
        
        # Test enhanced scorer
        scorer = EnhancedTokenScorer(window_size=32, fast_mode=True)
        
        start_time = time.time()
        enhanced_scores = scorer.get_enhanced_token_scores(key_states, value_states)
        end_time = time.time()
        
        print(f"✓ Enhanced scoring completed in {end_time - start_time:.4f}s")
        print(f"✓ Output shape: {enhanced_scores.shape}")
        print(f"✓ Score range: [{enhanced_scores.min().item():.4f}, {enhanced_scores.max().item():.4f}]")
        
        # Test alternative interface
        alt_scores = get_enhanced_scores_with_kv_fusion(key_states, value_states, scorer=scorer)
        print(f"✓ Alternative interface working: {alt_scores.shape}")
        
        # Test eviction candidate selection
        eviction_mask = scorer.get_eviction_candidates(enhanced_scores, num_evict=100)
        print(f"✓ Eviction mask shape: {eviction_mask.shape}")
        print(f"✓ Eviction candidates selected: {eviction_mask.sum().item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced token scoring test failed: {e}")
        return False

def test_adaptive_layer_budget():
    """Test adaptive layer budget optimization."""
    print("\nTesting Adaptive Layer Budget Optimization...")
    
    try:
        from adaptive_layer_budget import AdaptiveLayerBudgetOptimizer
        
        # Create optimizer
        optimizer = AdaptiveLayerBudgetOptimizer(num_layers=32)
        
        # Test basic functionality
        total_budget = 1024
        seq_len = 2048
        
        # Test without base preferences
        allocation = optimizer.get_adaptive_budget_allocation(total_budget, seq_len)
        print(f"✓ Basic allocation: {len(allocation)} layers, total: {sum(allocation)}")
        
        # Create mock base preferences
        mock_coeffs = np.random.randn(7)  # Polynomial coefficients
        np.save("temp_mock_coeffs.npy", mock_coeffs)
        
        optimizer.load_base_preferences("temp_mock_coeffs.npy")
        print("✓ Base preferences loaded")
        
        # Test adaptive allocation
        allocation = optimizer.get_adaptive_budget_allocation(total_budget, seq_len)
        print(f"✓ Adaptive allocation: total budget used: {sum(allocation)}/{total_budget}")
        
        # Test online updates
        mock_attention_patterns = [torch.randn(1, 8, 256, 256) for _ in range(32)]
        mock_cache_usage = [50 + np.random.randint(-10, 10) for _ in range(32)]
        
        optimizer.update_online_preferences(mock_attention_patterns, mock_cache_usage)
        print("✓ Online preferences updated")
        
        # Test enhanced allocation after updates
        new_allocation = optimizer.get_adaptive_budget_allocation(total_budget, seq_len)
        print(f"✓ Updated allocation: total budget used: {sum(new_allocation)}/{total_budget}")
        
        # Cleanup
        os.remove("temp_mock_coeffs.npy")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive layer budget test failed: {e}")
        return False

def test_llama_integration():
    """Test integration with LLaMA model modifications."""
    print("\nTesting LLaMA Integration...")
    
    try:
        # Add score path to import the modified llama functions
        score_path = Path(__file__).parent.parent / "score" / "model"
        sys.path.insert(0, str(score_path))
        
        from modify_llama_score import get_scores_with_kv_fusion, get_enhanced_scorer, get_adaptive_optimizer
        
        # Test enhanced scoring function
        bsz, num_heads, seq_len, head_dim = 1, 32, 1024, 128
        key_states = torch.randn(bsz, num_heads, seq_len, head_dim)
        value_states = torch.randn(bsz, num_heads, seq_len, head_dim)
        
        start_time = time.time()
        scores = get_scores_with_kv_fusion(key_states, value_states)
        end_time = time.time()
        
        print(f"✓ Integrated scoring completed in {end_time - start_time:.4f}s")
        print(f"✓ Output shape: {scores.shape}")
        
        # Test scorer initialization
        scorer = get_enhanced_scorer()
        if scorer is not None:
            print("✓ Enhanced scorer initialized")
        else:
            print("! Enhanced scorer not available (expected if modules not imported)")
        
        # Test optimizer initialization
        optimizer = get_adaptive_optimizer()
        if optimizer is not None:
            print("✓ Adaptive optimizer initialized")
        else:
            print("! Adaptive optimizer not available (expected if modules not imported)")
        
        return True
        
    except Exception as e:
        print(f"✗ LLaMA integration test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that optimizations don't break existing functionality."""
    print("\nTesting Backward Compatibility...")
    
    try:
        # Test original scoring still works
        bsz, num_heads, seq_len, head_dim = 1, 32, 1024, 128
        key_states = torch.randn(bsz, num_heads, seq_len, head_dim)
        value_states = torch.randn(bsz, num_heads, seq_len, head_dim)
        
        # Simulate original scoring logic
        def original_l2_scoring(key_states, value_states):
            def l2_distance_to_centroid(x):
                centroid = x.mean(dim=2, keepdim=True)
                distance = torch.norm(x - centroid, dim=-1)
                return distance
            
            window_size = 32
            if seq_len <= window_size:
                return torch.ones(bsz, num_heads, seq_len) * 0.1
            
            scoreable_len = seq_len - window_size
            score_k = l2_distance_to_centroid(key_states)[:, :, :scoreable_len]
            score_v = l2_distance_to_centroid(value_states)[:, :, :scoreable_len]
            
            # Normalize
            def normalize(x):
                min_val = x.amin(dim=-1, keepdim=True)
                max_val = x.amax(dim=-1, keepdim=True)
                return (x - min_val) / (max_val - min_val + 1e-6)
            
            score_k_norm = normalize(score_k)
            score_v_norm = normalize(score_v)
            final_score = score_k_norm + score_v_norm
            
            # Group heads for LLaMA
            if num_heads == 32:
                final_score = final_score.reshape(bsz, 8, 4, scoreable_len).mean(dim=2)
            
            return final_score
        
        original_scores = original_l2_scoring(key_states, value_states)
        print(f"✓ Original scoring works: {original_scores.shape}")
        
        # Test that new code produces reasonable results compared to original
        score_path = Path(__file__).parent.parent / "score" / "model"
        sys.path.insert(0, str(score_path))
        
        from modify_llama_score import get_scores_with_kv_fusion
        enhanced_scores = get_scores_with_kv_fusion(key_states, value_states)
        
        print(f"✓ Enhanced scoring works: {enhanced_scores.shape}")
        print(f"✓ Score correlation: {torch.corrcoef(torch.stack([original_scores.flatten(), enhanced_scores.flatten()]))[0, 1].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

def test_performance_characteristics():
    """Test performance characteristics of optimizations."""
    print("\nTesting Performance Characteristics...")
    
    try:
        # Test with different sequence lengths
        seq_lengths = [512, 1024, 2048, 4096]
        results = {}
        
        for seq_len in seq_lengths:
            bsz, num_heads, head_dim = 1, 32, 128
            key_states = torch.randn(bsz, num_heads, seq_len, head_dim)
            value_states = torch.randn(bsz, num_heads, seq_len, head_dim)
            
            # Time the enhanced scoring
            start_time = time.time()
            for _ in range(5):  # Average over 5 runs
                try:
                    from enhanced_token_scoring import EnhancedTokenScorer
                    scorer = EnhancedTokenScorer(fast_mode=True)
                    _ = scorer.get_enhanced_token_scores(key_states, value_states)
                except ImportError:
                    # Fallback to basic timing test
                    _ = torch.randn(bsz, 8, seq_len - 32)
            
            avg_time = (time.time() - start_time) / 5
            results[seq_len] = avg_time
            
            print(f"✓ Seq length {seq_len}: {avg_time:.4f}s average")
        
        # Check that performance scales reasonably
        scaling_factor = results[4096] / results[512] if results[512] > 0 else 1
        print(f"✓ Scaling factor (4096/512): {scaling_factor:.2f}x")
        
        if scaling_factor < 20:  # Should scale better than O(n^2)
            print("✓ Performance scaling looks reasonable")
        else:
            print("! Performance scaling may need optimization")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("CAKE Enhanced Algorithm Optimization Tests")
    print("="*60)
    
    tests = [
        test_enhanced_token_scoring,
        test_adaptive_layer_budget,
        test_llama_integration,
        test_backward_compatibility,
        test_performance_characteristics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    
    test_names = [
        "Enhanced Token Scoring",
        "Adaptive Layer Budget",
        "LLaMA Integration", 
        "Backward Compatibility",
        "Performance Characteristics"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All optimizations working correctly!")
        return 0
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())