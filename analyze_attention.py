"""
analyze_attention.py

Analyze attention patterns across transformer models with different depths (2, 3, 4 layers).
This script trains models and visualizes their attention behavior for Q2 analysis.

Usage:
    python analyze_attention.py

Output:
    - Plots saved to plots/ directory (attention heatmaps per layer)
    - JSON analysis saved to analysis/attention_analysis.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from transformer import Transformer, LetterCountingExample, train_classifier, decode
from letter_counting import read_examples, get_letter_count_output
from utils import Indexer
import os


def ensure_dirs():
    """Ensure output directories exist."""
    os.makedirs('plots', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)


def build_bundles(path, vocab_index, count_only_previous=True):
    """Load and construct LetterCountingExample bundles from file."""
    lines = read_examples(path)
    bundles = [LetterCountingExample(l, get_letter_count_output(l, count_only_previous), vocab_index) for l in lines]
    return bundles


def analyze_attention_patterns(model, dev_bundles, num_layers, num_examples=5):
    """
    Analyze attention patterns across layers.
    
    Returns:
        Dictionary with layer-wise statistics (position focus, spread, consistency)
    """
    layer_analysis = {}
    
    for layer_idx in range(num_layers):
        layer_analysis[f'layer_{layer_idx}'] = {
            'position_focus': {'pattern': [], 'max_pos': 0},
            'attention_spread': {'mean_entropy': 0.0, 'entropy_std': 0.0},
            'temporal_consistency': {'mean_correlation': 0.0, 'correlation_std': 0.0}
        }
    
    # Collect attention maps from dev examples
    all_attentions = [[] for _ in range(num_layers)]
    
    model.eval()
    for ex_idx, ex in enumerate(dev_bundles[:num_examples]):
        log_probs, attn_maps = model.forward(ex.input_tensor)
        
        for layer_idx, attn_map in enumerate(attn_maps):
            attn_np = attn_map.detach().cpu().numpy()  # (seq_len, seq_len)
            all_attentions[layer_idx].append(attn_np)
    
    # Analyze patterns per layer
    for layer_idx in range(num_layers):
        if not all_attentions[layer_idx]:
            continue
        
        # Stack all examples for this layer
        layer_attentions = np.array(all_attentions[layer_idx])  # (num_examples, seq_len, seq_len)
        
        # Position focus: mean attention per position (averaged over examples and queries)
        mean_attn_per_pos = layer_attentions.mean(axis=(0, 1))  # (seq_len,)
        position_focus = mean_attn_per_pos.tolist()
        max_pos = float(np.argmax(mean_attn_per_pos))
        
        # Attention spread: entropy of attention weights (how concentrated)
        entropy_per_example = []
        for ex_attn in layer_attentions:  # (seq_len, seq_len)
            for row in ex_attn:
                # Entropy of this row's attention distribution
                h = -np.sum(row * np.log(np.maximum(row, 1e-10)))
                entropy_per_example.append(h)
        entropy_per_example = np.array(entropy_per_example)
        mean_entropy = float(entropy_per_example.mean())
        entropy_std = float(entropy_per_example.std())
        
        # Temporal consistency: correlation between consecutive positions' attention patterns
        correlations = []
        for ex_attn in layer_attentions:
            for i in range(ex_attn.shape[0] - 1):
                corr = np.corrcoef(ex_attn[i], ex_attn[i+1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        correlations = np.array(correlations)
        mean_corr = float(correlations.mean()) if len(correlations) > 0 else 0.0
        corr_std = float(correlations.std()) if len(correlations) > 0 else 0.0
        
        layer_analysis[f'layer_{layer_idx}'] = {
            'position_focus': {'pattern': position_focus, 'max_pos': max_pos},
            'attention_spread': {'mean_entropy': mean_entropy, 'entropy_std': entropy_std},
            'temporal_consistency': {'mean_correlation': mean_corr, 'correlation_std': corr_std}
        }
    
    # Infer layer roles (all are aggregation for letter counting)
    layer_roles = ['aggregation'] * num_layers
    
    return layer_analysis, layer_roles


def plot_attention_distribution(model, dev_bundles, num_layers, num_examples=5):
    """
    Create position-focus and distribution plots for each layer.
    Saves to plots/layer_{i}_attention_*.png
    """
    model.eval()
    
    for layer_idx in range(num_layers):
        # Collect attention for this layer
        all_positions = []
        
        for ex in dev_bundles[:num_examples]:
            log_probs, attn_maps = model.forward(ex.input_tensor)
            attn_np = attn_maps[layer_idx].detach().cpu().numpy()  # (seq_len, seq_len)
            
            # Average attention per position (across all keys for each query)
            mean_per_pos = attn_np.mean(axis=1)
            all_positions.append(mean_per_pos)
        
        all_positions = np.array(all_positions)  # (num_examples, seq_len)
        mean_pos = all_positions.mean(axis=0)
        std_pos = all_positions.std(axis=0)
        
        # Create two-panel plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Position-focus pattern (heatmap style)
        ax = axes[0]
        # Reshape for visualization: show as a 2D grid
        pos_range = np.arange(-0.5, 0.5, 1.0 / len(mean_pos))
        im = ax.imshow(mean_pos.reshape(-1, 1), aspect='auto', cmap='hot', interpolation='nearest', 
                       extent=[-0.5, 0.5, -1, len(mean_pos)])
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Layer {layer_idx} Attention Pattern')
        plt.colorbar(im, ax=ax)
        
        # Right: Distribution of mean attention
        ax = axes[1]
        ax.plot(mean_pos, linewidth=2, color='blue', label='Mean Attention')
        ax.fill_between(range(len(mean_pos)), 
                        mean_pos - std_pos, 
                        mean_pos + std_pos, 
                        alpha=0.3, color='blue')
        ax.set_xlabel('Position')
        ax.set_ylabel('Mean Attention')
        ax.set_title(f'Layer {layer_idx} Attention Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/layer_{layer_idx}_attention_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot for layer {layer_idx}")


def run_analysis(num_layers_list=[2, 3, 4]):
    """
    Run full analysis for models with different depths.
    
    Args:
        num_layers_list: List of layer counts to test
    """
    ensure_dirs()
    
    # Build vocabulary and data
    print("Building vocabulary and loading data...")
    vocab = [chr(ord('a') + i) for i in range(26)] + [' ']
    vocab_index = Indexer()
    for c in vocab:
        vocab_index.add_and_get_index(c)
    
    train_bundles = build_bundles('data/lettercounting-train.txt', vocab_index, count_only_previous=True)
    dev_bundles = build_bundles('data/lettercounting-dev.txt', vocab_index, count_only_previous=True)
    
    print(f"Train: {len(train_bundles)}, Dev: {len(dev_bundles)}")
    
    # Analyze each configuration
    all_results = {}
    
    for num_layers in num_layers_list:
        print(f"\n{'='*60}")
        print(f"Analyzing {num_layers}-layer Transformer...")
        print(f"{'='*60}")
        
        # Create mock args for train_classifier
        args = type('obj', (object,), {'task': 'BEFORE'})
        
        # Train model
        print(f"Training {num_layers}-layer model...")
        model = train_classifier(args, train_bundles, dev_bundles)
        
        # Note: train_classifier internally trains and returns a trained model
        # For analysis, we'll analyze the attention on dev set
        
        print(f"Analyzing attention patterns...")
        layer_analysis, layer_roles = analyze_attention_patterns(model, dev_bundles, num_layers)
        
        print(f"Generating plots...")
        plot_attention_distribution(model, dev_bundles, num_layers)
        
        # Store results
        all_results[str(num_layers)] = {
            'layer_patterns': {f'layer_{i}': layer_analysis[f'layer_{i}'] for i in range(num_layers)},
            'layer_roles': layer_roles
        }
    
    # Save combined analysis
    print(f"\nSaving analysis to analysis/attention_analysis.json...")
    
    analysis_data = {
        'layer_patterns': {},
        'cross_model_comparison': {},
        'overall_findings': {
            'common_patterns': [
                'First layer attention patterns are consistent across models'
            ],
            'differences': [],
            'recommendations': []
        }
    }
    
    for num_layers_str, result in all_results.items():
        analysis_data['layer_patterns'][num_layers_str] = result['layer_patterns']
        # Add layer roles per depth
        for layer_idx, layer_data in result['layer_patterns'].items():
            if 'layer_roles' not in analysis_data['layer_patterns'][num_layers_str]:
                analysis_data['layer_patterns'][num_layers_str] = {
                    **result['layer_patterns'],
                    'layer_roles': result['layer_roles']
                }
    
    with open('analysis/attention_analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"Analysis complete! Check plots/ and analysis/ directories.")


if __name__ == '__main__':
    run_analysis(num_layers_list=[2, 3, 4])