#!/usr/bin/env python3
"""
View and analyze Phase 3 activation data.

This script loads the cached activations and shows:
- Basic statistics
- Dimensionality reduction (PCA)
- Separation between faithful and unfaithful
- Which samples are most/least separable
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import LinearProbe for deserialization (must be at module level)
try:
    from mechanistic.train_probes import LinearProbe
except ImportError:
    try:
        from src.mechanistic.train_probes import LinearProbe
    except ImportError:
        print("⚠️  Warning: Could not import LinearProbe. Probe analysis may fail.")
        LinearProbe = None


def load_activations(layer, activations_dir="data/activations"):
    """Load activations for a specific layer."""
    
    path = Path(activations_dir) / f"layer_{layer}_activations.pt"
    if not path.exists():
        print(f"❌ File not found: {path}")
        return None, None
    
    data = torch.load(path)
    faithful = data['faithful'].numpy()
    unfaithful = data['unfaithful'].numpy()
    
    return faithful, unfaithful


def print_statistics(faithful, unfaithful, layer):
    """Print basic statistics about the activations."""
    
    print(f"\n{'='*60}")
    print(f"LAYER {layer} STATISTICS")
    print(f"{'='*60}")
    print()
    
    print(f"Faithful activations:")
    print(f"  Shape: {faithful.shape}")
    print(f"  Mean: {faithful.mean():.4f}")
    print(f"  Std:  {faithful.std():.4f}")
    print(f"  Min:  {faithful.min():.4f}")
    print(f"  Max:  {faithful.max():.4f}")
    print()
    
    print(f"Unfaithful activations:")
    print(f"  Shape: {unfaithful.shape}")
    print(f"  Mean: {unfaithful.mean():.4f}")
    print(f"  Std:  {unfaithful.std():.4f}")
    print(f"  Min:  {unfaithful.min():.4f}")
    print(f"  Max:  {unfaithful.max():.4f}")
    print()
    
    # Compute distance between means
    mean_faithful = faithful.mean(axis=0)
    mean_unfaithful = unfaithful.mean(axis=0)
    distance = np.linalg.norm(mean_faithful - mean_unfaithful)
    
    # Normalize by std
    pooled_std = np.sqrt((faithful.std()**2 + unfaithful.std()**2) / 2)
    normalized_distance = distance / pooled_std if pooled_std > 0 else 0
    
    print(f"Separation metrics:")
    print(f"  Distance between means: {distance:.4f}")
    print(f"  Normalized distance:    {normalized_distance:.4f}")
    print(f"  (Higher = better separation)")


def visualize_2d(faithful, unfaithful, layer, method='pca'):
    """Visualize activations in 2D using PCA or t-SNE."""
    
    print(f"\n{'='*60}")
    print(f"2D VISUALIZATION ({method.upper()}) - Layer {layer}")
    print(f"{'='*60}")
    
    # Combine data
    X = np.vstack([faithful, unfaithful])
    labels = np.array([1]*len(faithful) + [0]*len(unfaithful))
    
    # Reduce dimensions
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        explained = reducer.explained_variance_ratio_
        print(f"\nPCA explained variance: {explained[0]:.1%} + {explained[1]:.1%} = {explained.sum():.1%}")
    elif method.lower() == 'tsne':
        print("\nRunning t-SNE (this may take a minute)...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_2d = reducer.fit_transform(X)
    else:
        print(f"Unknown method: {method}")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot faithful (blue)
    faithful_2d = X_2d[labels == 1]
    plt.scatter(faithful_2d[:, 0], faithful_2d[:, 1], 
                c='blue', alpha=0.6, s=100, label='Faithful', edgecolors='black')
    
    # Plot unfaithful (red)
    unfaithful_2d = X_2d[labels == 0]
    plt.scatter(unfaithful_2d[:, 0], unfaithful_2d[:, 1], 
                c='red', alpha=0.6, s=100, label='Unfaithful', edgecolors='black')
    
    # Add labels to points
    for i, (x, y) in enumerate(faithful_2d):
        plt.annotate(f'F{i+1}', (x, y), fontsize=8, alpha=0.7)
    for i, (x, y) in enumerate(unfaithful_2d):
        plt.annotate(f'U{i+1}', (x, y), fontsize=8, alpha=0.7)
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(f'Layer {layer} Activations - {method.upper()} Projection', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path('results/activation_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'layer_{layer}_{method}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    
    plt.show()


def compute_probe_direction_projection(faithful, unfaithful, layer):
    """Load probe direction and project activations onto it."""
    
    print(f"\n{'='*60}")
    print(f"PROBE DIRECTION PROJECTION - Layer {layer}")
    print(f"{'='*60}")
    
    # Load probe results
    probe_path = Path('results/probe_results/all_probe_results.pt')
    if not probe_path.exists():
        print(f"\n⚠️  Probe results not found: {probe_path}")
        return
    
    if LinearProbe is None:
        print(f"\n❌ LinearProbe class not available. Cannot deserialize probe.")
        return
    
    try:
        results = torch.load(probe_path, weights_only=False)
        layer_key = f'layer_{layer}'
        
        if layer_key not in results:
            print(f"\n⚠️  No probe results for {layer_key}")
            return
        
        direction = results[layer_key].direction.numpy()
        accuracy = results[layer_key].accuracy
        
        print(f"\nProbe accuracy: {accuracy:.1%}")
        print(f"Direction shape: {direction.shape}")
        
        # Project activations onto direction
        faithful_proj = faithful @ direction
        unfaithful_proj = unfaithful @ direction
        
        print(f"\nProjections onto probe direction:")
        print(f"  Faithful:   mean={faithful_proj.mean():.4f}, std={faithful_proj.std():.4f}")
        print(f"  Unfaithful: mean={unfaithful_proj.mean():.4f}, std={unfaithful_proj.std():.4f}")
        
        # Compute separation
        separation = abs(faithful_proj.mean() - unfaithful_proj.mean())
        pooled_std = np.sqrt((faithful_proj.std()**2 + unfaithful_proj.std()**2) / 2)
        cohens_d = separation / pooled_std if pooled_std > 0 else 0
        
        print(f"\nSeparation:")
        print(f"  Absolute: {separation:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        
        if cohens_d < 0.2:
            print(f"  → Negligible separation")
        elif cohens_d < 0.5:
            print(f"  → Small separation")
        elif cohens_d < 0.8:
            print(f"  → Medium separation")
        else:
            print(f"  → Large separation")
        
        # Visualize projections
        plt.figure(figsize=(10, 6))
        
        # Histogram
        bins = np.linspace(
            min(faithful_proj.min(), unfaithful_proj.min()),
            max(faithful_proj.max(), unfaithful_proj.max()),
            20
        )
        
        plt.hist(faithful_proj, bins=bins, alpha=0.5, label='Faithful', color='blue')
        plt.hist(unfaithful_proj, bins=bins, alpha=0.5, label='Unfaithful', color='red')
        
        # Add vertical lines for means
        plt.axvline(faithful_proj.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Faithful mean: {faithful_proj.mean():.2f}')
        plt.axvline(unfaithful_proj.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Unfaithful mean: {unfaithful_proj.mean():.2f}')
        
        plt.xlabel('Projection Value', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Layer {layer} - Projection onto Probe Direction\n(Accuracy: {accuracy:.1%}, Cohen\'s d: {cohens_d:.2f})', 
                 fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_dir = Path('results/activation_visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'layer_{layer}_probe_projection.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved to: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error loading probe: {e}")


def analyze_all_layers():
    """Analyze all available layers."""
    
    layers = [6, 12, 18, 24]
    
    print("=" * 60)
    print("PHASE 3 ACTIVATION ANALYSIS")
    print("=" * 60)
    print()
    print("Available analyses:")
    print("  1. Statistics")
    print("  2. PCA visualization")
    print("  3. Probe direction projection")
    print("  4. All of the above")
    print()
    
    choice = input("Choose analysis (1-4) or layer number (6/12/18/24): ").strip()
    
    if choice in ['6', '12', '18', '24']:
        layer = int(choice)
        analyze_layer(layer, full_analysis=True)
    elif choice == '1':
        for layer in layers:
            faithful, unfaithful = load_activations(layer)
            if faithful is not None:
                print_statistics(faithful, unfaithful, layer)
    elif choice == '2':
        for layer in layers:
            faithful, unfaithful = load_activations(layer)
            if faithful is not None:
                visualize_2d(faithful, unfaithful, layer, method='pca')
    elif choice == '3':
        for layer in layers:
            faithful, unfaithful = load_activations(layer)
            if faithful is not None:
                compute_probe_direction_projection(faithful, unfaithful, layer)
    elif choice == '4':
        for layer in layers:
            analyze_layer(layer, full_analysis=True)
    else:
        print("Analyzing all layers (statistics only)...")
        for layer in layers:
            faithful, unfaithful = load_activations(layer)
            if faithful is not None:
                print_statistics(faithful, unfaithful, layer)


def analyze_layer(layer, full_analysis=False):
    """Analyze a specific layer."""
    
    faithful, unfaithful = load_activations(layer)
    if faithful is None:
        return
    
    # Always show statistics
    print_statistics(faithful, unfaithful, layer)
    
    if full_analysis:
        # PCA visualization
        visualize_2d(faithful, unfaithful, layer, method='pca')
        
        # Probe direction projection
        compute_probe_direction_projection(faithful, unfaithful, layer)


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='View and analyze Phase 3 activation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python view_activations.py
  
  # Analyze specific layer
  python view_activations.py --layer 12
  
  # Just statistics
  python view_activations.py --stats
  
  # PCA visualization only
  python view_activations.py --pca
  
  # Probe projection only
  python view_activations.py --probe
        """
    )
    
    parser.add_argument('--layer', type=int, choices=[6, 12, 18, 24],
                       help='Analyze specific layer')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics only')
    parser.add_argument('--pca', action='store_true',
                       help='Show PCA visualization only')
    parser.add_argument('--probe', action='store_true',
                       help='Show probe projection only')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    # If no arguments, run interactive mode
    if not any([args.layer, args.stats, args.pca, args.probe, args.all]):
        analyze_all_layers()
        return
    
    # Determine which layers to analyze
    layers = [args.layer] if args.layer else [6, 12, 18, 24]
    
    # Run analyses
    for layer in layers:
        faithful, unfaithful = load_activations(layer)
        if faithful is None:
            continue
        
        if args.stats or args.all:
            print_statistics(faithful, unfaithful, layer)
        
        if args.pca or args.all:
            visualize_2d(faithful, unfaithful, layer, method='pca')
        
        if args.probe or args.all:
            compute_probe_direction_projection(faithful, unfaithful, layer)
        
        # If no specific analysis chosen but layer specified, do full analysis
        if args.layer and not any([args.stats, args.pca, args.probe, args.all]):
            print_statistics(faithful, unfaithful, layer)
            visualize_2d(faithful, unfaithful, layer, method='pca')
            compute_probe_direction_projection(faithful, unfaithful, layer)


if __name__ == "__main__":
    main()

