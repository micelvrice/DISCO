# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAKE (Cascading and Adaptive KV Cache Eviction with Layer Preferences) is a research implementation for optimizing Large Language Model inference efficiency. It implements sophisticated KV cache eviction strategies that analyze attention patterns across transformer layers and adaptively allocate cache memory, maintaining model performance while using only 3.2% of the original KV cache.

## Development Commands

### Installation and Setup
```bash
# Install the package and patch transformers
bash install.sh

# Manual installation (if needed)
pip install -e .
```

### Running Experiments

#### LongBench Evaluation
```bash
# Run prediction with CAKE compression
python experiments/LongBench/pred.py \
    --model llama3.1-8b-128k \
    --compress --cascading \
    --pred_name pred_result --device 0 \
    --cache_size 1024 \
    --window_size 32

# Run evaluation
python experiments/LongBench/eval.py \
    --model llama3.1-8b-128k \
    --cache_size 1024 \
    --eval_avg \
    --dir_path pred_result
```

#### Common Parameters
- `--model`: Model choice (llama3.1-8b-128k, mistral-0.3-7b-32k, qwen2.5-7b-instruct, etc.)
- `--compress`: Enable CAKE compression
- `--cascading`: Enable cascading cache management
- `--cache_size`: Total KV cache memory budget (default: 1024)
- `--window_size`: Recent token window size (default: 32)
- `--score`: Enable scoring functions

## Architecture Overview

### Core Components

**`/score/` - Main Implementation**
- `score_cache.py`: Core CAKE cache classes
  - `CakeCache`: Base cache with dynamic growth and eviction scoring
  - `CakeprefillKVCache`: Prefill-phase cache management  
  - `CakeDecodingKVCache_LayerWise`: Layer-wise decoding cache
- `utils.py`: Configuration (`CompressConfig`) and utility functions
- `monkeypatch.py`: Runtime patching of transformers attention mechanisms

**`/score/model/` - Model Integration**
- Model-specific implementations for LLaMA, Mistral, and Qwen2
- Each file modifies the attention mechanism for that specific model architecture

**`/experiments/LongBench/` - Evaluation Framework**
- `pred.py`: Generate predictions with CAKE compression
- `eval.py`: Evaluate predictions using task-specific metrics
- `metrics.py`: Implementation of evaluation metrics (F1, ROUGE, classification, etc.)
- `config/`: Model paths, prompts, and hyperparameter configurations

**`/get_score_function/` - Scoring Functions**
- Polynomial fitting for layer preference learning
- PCA analysis for efficient scoring
- Cross-dataset generalization training

### Key Technical Features

1. **Layer-aware Cache Allocation**: Different layers receive different cache budgets based on attention patterns
2. **Cascading Management**: Hierarchical cache eviction considering global memory constraints
3. **Information-Theoretic Scoring**: Uses entropy and mutual information for intelligent token selection
4. **Temporal Dynamics**: Eviction scoring considers how token importance changes over time

### Dependencies

- **Python 3.9+** required
- **transformers==4.43.3** (specific version requirement)
- PyTorch, NumPy, datasets, accelerate
- NLP evaluation libraries: rouge, rouge_score, tokenizers

### Installation Notes

The `install.sh` script patches specific lines in the transformers library to disable float conversion in model outputs. This is required for CAKE to function properly with the supported model architectures.

### Configuration

Main configuration is handled through `CompressConfig` class in `score/utils.py`:
- `cache_size`: Total memory budget
- `window_size`: Recent token window
- `compress`: Enable/disable CAKE compression
- `cascading`: Enable/disable cascading management
- Layer-specific parameters: `tau1`, `tau2`, `gamma`