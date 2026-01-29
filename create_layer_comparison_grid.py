#!/usr/bin/env python3
"""
Create a static grid comparison of all layers side-by-side.

This complements the animations by showing all layers at once,
making it easy to compare separation across layers.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from mechanistic.train_probes import LinearProbe
except ImportError:
    from src.mechanistic.train_probes import LinearProbe


def load_all_data(layers, activations_dir="data/activations"):
    """Load activations and probe directions."""
    activations = {}
    for layer in layers:
        path = Path(activations_dir) / f"layer_{layer}_activations.pt"
        if path.exists():
            data = torch.load(path)
            activations[layer] = {
                'faithful': data['faithful'].numpy(),
                'unfaithful': data['unfaithful'].numpy()
            }
    
    # Load probes
    probe_path = Path('results/probe_results/all_probe_results.pt')
    probe_directions = {}
    if probe_path.exists():
        results = torch.load(probe_path, weights_only=False)
        for layer in layers:
            layer_key = f'layer_{layer}'
            if layer_key in results:
                probe_directions[layer] = results[layer_key].direction.numpy()
    
    return activations, probe_directions


def create_comparison_grid(
    activations,
    probe_directions,
    layers,
    output_file="results/activation_visualizations/layer_comparison_grid.png"
):
    """Create a 2x2 grid showing all layers."""
    
    print("Creating layer comparison grid...")
    
    # Compute global PCA for consistent coordinates
    all_data = []
    for layer in layers:
        all_data.append(activations[layer]['faithful'])
        all_data.append(activations[layer]['unfaithful'])
    all_data = np.vstack(all_data)
    
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Compute global axis limits
    all_x, all_y = [], []
    for layer in layers:
        faithful = activations[layer]['faithful']
        unfaithful = activations[layer]['unfaithful']
        
        # Probe projection
        if layer in probe_directions:
            direction = probe_directions[layer]
            all_x.extend(faithful @ direction)
            all_x.extend(unfaithful @ direction)
        
        # PCA
        faithful_pca = pca.transform(faithful)
        unfaithful_pca = pca.transform(unfaithful)
        all_y.extend(faithful_pca[:, 0])
        all_y.extend(unfaithful_pca[:, 0])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    # Plot each layer
    for idx, layer in enumerate(layers):
        ax = axes[idx]
        
        faithful = activations[layer]['faithful']
        unfaithful = activations[layer]['unfaithful']
        
        # Get projections
        if layer in probe_directions:
            direction = probe_directions[layer]
            faithful_probe = faithful @ direction
            unfaithful_probe = unfaithful @ direction
        else:
            continue
        
        faithful_pca = pca.transform(faithful)[:, 0]
        unfaithful_pca = pca.transform(unfaithful)[:, 0]
        
        # Scatter plot
        ax.scatter(faithful_probe, faithful_pca, c='blue', alpha=0.6, s=100,
                  label='Faithful', edgecolors='black', linewidths=0.5)
        ax.scatter(unfaithful_probe, unfaithful_pca, c='red', alpha=0.6, s=100,
                  label='Unfaithful', edgecolors='black', linewidths=0.5)
        
        # Compute statistics
        f_mean = faithful_probe.mean()
        u_mean = unfaithful_probe.mean()
        separation = abs(f_mean - u_mean)
        
        # Styling
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel('Probe Direction (Faithfulness)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC1 (Maximum Variance)', fontsize=12, fontweight='bold')
        ax.set_title(f'Layer {layer}\nSeparation: {separation:.3f}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (
            f'Faithful: μ={f_mean:+.3f}\n'
            f'Unfaithful: μ={u_mean:+.3f}\n'
            f'Δ = {separation:.3f}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
               fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Layer Progression Comparison: Faithful vs Unfaithful Activations',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()
    return output_path


def create_separation_plot(
    activations,
    probe_directions,
    layers,
    output_file="results/activation_visualizations/separation_across_layers.png"
):
    """Create a line plot showing separation metrics across layers."""
    
    print("Creating separation plot...")
    
    separations = []
    
    for layer in layers:
        faithful = activations[layer]['faithful']
        unfaithful = activations[layer]['unfaithful']
        
        if layer in probe_directions:
            direction = probe_directions[layer]
            faithful_proj = faithful @ direction
            unfaithful_proj = unfaithful @ direction
            
            separation = abs(faithful_proj.mean() - unfaithful_proj.mean())
            separations.append(separation)
        else:
            separations.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(layers, separations, marker='o', linewidth=3, markersize=12,
           color='darkblue', label='Probe Direction Separation')
    
    # Highlight maximum
    max_idx = np.argmax(separations)
    max_layer = layers[max_idx]
    max_sep = separations[max_idx]
    
    ax.scatter([max_layer], [max_sep], s=300, c='red', marker='*',
              edgecolors='black', linewidths=2, zorder=5,
              label=f'Maximum (Layer {max_layer})')
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Separation (|μ_faithful - μ_unfaithful|)', fontsize=14, fontweight='bold')
    ax.set_title('Faithfulness Separation Across Layers', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    
    # Add value labels
    for layer, sep in zip(layers, separations):
        ax.text(layer, sep + 0.01, f'{sep:.3f}', ha='center', va='bottom',
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()
    return output_path


def main():
    """Main workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create static layer comparison visualizations')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 12, 18, 24],
                       help='Layers to visualize')
    parser.add_argument('--activations-dir', type=str, default='data/activations',
                       help='Directory containing activation files')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LAYER COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    activations, probe_directions = load_all_data(args.layers, args.activations_dir)
    
    if not activations:
        print("❌ No activations loaded. Exiting.")
        return
    
    print(f"✓ Loaded {len(activations)} layers")
    print(f"✓ Loaded {len(probe_directions)} probe directions")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    grid_path = create_comparison_grid(activations, probe_directions, args.layers)
    sep_path = create_separation_plot(activations, probe_directions, args.layers)
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"Created:")
    print(f"  - {grid_path}")
    print(f"  - {sep_path}")


if __name__ == "__main__":
    main()

