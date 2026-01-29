#!/usr/bin/env python3
"""
Animate Layer Progression: Faithful vs Unfaithful Responses

This script creates an animation showing how activation representations
evolve across layers, with faithful and unfaithful responses as separate groups.

Visualization options:
1. Probe direction projection (1D → jittered to 2D)
2. PCA (2D projection)
3. Combined (probe direction + PCA component)
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import LinearProbe for deserialization
try:
    from mechanistic.train_probes import LinearProbe
except ImportError:
    from src.mechanistic.train_probes import LinearProbe


def save_animation(anim, output_path, fps=2):
    """
    Save animation, trying MP4 first (requires ffmpeg), falling back to GIF.
    
    Args:
        anim: matplotlib FuncAnimation object
        output_path: Path object or string
        fps: frames per second
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try MP4 first (better quality, smaller file)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"✓ Saved as MP4: {output_path}")
        return output_path
    except FileNotFoundError:
        # ffmpeg not installed, fallback to GIF
        warnings.warn(
            "ffmpeg not found. Falling back to GIF format. "
            "Install ffmpeg for better quality: brew install ffmpeg (Mac) or apt-get install ffmpeg (Linux)"
        )
        
        # Change extension to .gif
        gif_path = output_path.with_suffix('.gif')
        
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        print(f"✓ Saved as GIF: {gif_path}")
        return gif_path


def load_all_layer_activations(layers, activations_dir="data/activations"):
    """
    Load activations for all specified layers.
    
    Returns:
        dict: {layer: {'faithful': np.array, 'unfaithful': np.array}}
    """
    activations = {}
    
    for layer in layers:
        path = Path(activations_dir) / f"layer_{layer}_activations.pt"
        if not path.exists():
            print(f"⚠️  Warning: {path} not found, skipping layer {layer}")
            continue
        
        data = torch.load(path)
        activations[layer] = {
            'faithful': data['faithful'].numpy(),
            'unfaithful': data['unfaithful'].numpy()
        }
        
        print(f"✓ Loaded layer {layer}: {len(data['faithful'])} faithful, {len(data['unfaithful'])} unfaithful")
    
    return activations


def load_probe_directions(layers, probe_path="results/probe_results/all_probe_results.pt"):
    """
    Load probe directions for all layers.
    
    Returns:
        dict: {layer: direction_vector}
    """
    if not Path(probe_path).exists():
        print(f"⚠️  Warning: Probe results not found at {probe_path}")
        return None
    
    results = torch.load(probe_path, weights_only=False)
    
    directions = {}
    for layer in layers:
        layer_key = f'layer_{layer}'
        if layer_key in results:
            directions[layer] = results[layer_key].direction.numpy()
            print(f"✓ Loaded probe direction for layer {layer} (accuracy: {results[layer_key].accuracy:.3f})")
    
    return directions


def project_onto_probe_direction(activations, probe_directions, layers):
    """
    Project activations onto probe directions for each layer.
    
    Returns:
        dict: {layer: {'faithful_proj': np.array, 'unfaithful_proj': np.array}}
    """
    projections = {}
    
    for layer in layers:
        if layer not in activations or layer not in probe_directions:
            continue
        
        direction = probe_directions[layer]
        faithful = activations[layer]['faithful']
        unfaithful = activations[layer]['unfaithful']
        
        # Project: shape [n_samples, d_model] @ [d_model] = [n_samples]
        faithful_proj = faithful @ direction
        unfaithful_proj = unfaithful @ direction
        
        projections[layer] = {
            'faithful': faithful_proj,
            'unfaithful': unfaithful_proj
        }
    
    return projections


def compute_pca_per_layer(activations, layers, n_components=2):
    """
    Compute PCA independently for each layer.
    
    Returns:
        dict: {layer: {'faithful_pca': np.array, 'unfaithful_pca': np.array}}
    """
    pca_projections = {}
    
    for layer in layers:
        if layer not in activations:
            continue
        
        faithful = activations[layer]['faithful']
        unfaithful = activations[layer]['unfaithful']
        
        # Stack all data for PCA fitting
        all_data = np.vstack([faithful, unfaithful])
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(all_data)
        
        # Transform
        faithful_pca = pca.transform(faithful)
        unfaithful_pca = pca.transform(unfaithful)
        
        pca_projections[layer] = {
            'faithful': faithful_pca,
            'unfaithful': unfaithful_pca,
            'explained_variance': pca.explained_variance_ratio_
        }
        
        print(f"✓ PCA layer {layer}: explained variance = {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca_projections


def compute_global_pca(activations, layers, n_components=2):
    """
    Compute PCA on ALL layers concatenated (global coordinate system).
    
    This allows direct comparison across layers since all projections
    use the same PCA basis.
    
    Returns:
        dict: {layer: {'faithful_pca': np.array, 'unfaithful_pca': np.array}}
    """
    # Collect all activations across all layers
    all_data_list = []
    for layer in layers:
        if layer not in activations:
            continue
        all_data_list.append(activations[layer]['faithful'])
        all_data_list.append(activations[layer]['unfaithful'])
    
    # Stack into one big matrix
    all_data = np.vstack(all_data_list)
    
    print(f"\n📊 Fitting global PCA on {all_data.shape[0]} samples across {len(layers)} layers...")
    
    # Fit PCA once on all data
    pca = PCA(n_components=n_components)
    pca.fit(all_data)
    
    print(f"✓ Global PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Transform each layer's data
    pca_projections = {}
    for layer in layers:
        if layer not in activations:
            continue
        
        faithful = activations[layer]['faithful']
        unfaithful = activations[layer]['unfaithful']
        
        faithful_pca = pca.transform(faithful)
        unfaithful_pca = pca.transform(unfaithful)
        
        pca_projections[layer] = {
            'faithful': faithful_pca,
            'unfaithful': unfaithful_pca,
        }
    
    return pca_projections


def create_probe_projection_animation(
    probe_projections,
    layers,
    output_file="results/activation_visualizations/layer_progression_probe.mp4",
    fps=2
):
    """
    Create animation using probe direction projection (1D).
    
    Since probe projection is 1D, we'll:
    - Use probe projection as X-axis
    - Add random jitter on Y-axis for visibility
    """
    print(f"\n🎬 Creating probe direction animation...")
    
    # Prepare data
    n_faithful = len(probe_projections[layers[0]]['faithful'])
    n_unfaithful = len(probe_projections[layers[0]]['unfaithful'])
    
    # Generate consistent random jitter for each point
    np.random.seed(42)
    faithful_jitter = np.random.randn(n_faithful) * 0.05
    unfaithful_jitter = np.random.randn(n_unfaithful) * 0.05
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Initialize scatter plots (will be updated in animation)
    faithful_scatter = ax.scatter([], [], c='blue', alpha=0.6, s=100, 
                                  label=f'Faithful (n={n_faithful})', edgecolors='black', linewidths=0.5)
    unfaithful_scatter = ax.scatter([], [], c='red', alpha=0.6, s=100,
                                    label=f'Unfaithful (n={n_unfaithful})', edgecolors='black', linewidths=0.5)
    
    # Compute global X-axis limits (across all layers)
    all_projs = []
    for layer in layers:
        all_projs.extend(probe_projections[layer]['faithful'])
        all_projs.extend(probe_projections[layer]['unfaithful'])
    
    x_min, x_max = np.min(all_projs), np.max(all_projs)
    x_margin = (x_max - x_min) * 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('Projection onto Probe Direction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Random Jitter (for visibility)', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Title and text annotations
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', 
                   fontsize=16, fontweight='bold')
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', 
                        fontsize=10, family='monospace', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        """Initialize animation."""
        faithful_scatter.set_offsets(np.empty((0, 2)))
        unfaithful_scatter.set_offsets(np.empty((0, 2)))
        return faithful_scatter, unfaithful_scatter, title, stats_text
    
    def update(frame):
        """Update animation for each frame (layer)."""
        layer = layers[frame]
        
        # Get projections for this layer
        faithful_proj = probe_projections[layer]['faithful']
        unfaithful_proj = probe_projections[layer]['unfaithful']
        
        # Create 2D coordinates (projection + jitter)
        faithful_coords = np.column_stack([faithful_proj, faithful_jitter])
        unfaithful_coords = np.column_stack([unfaithful_proj, unfaithful_jitter])
        
        # Update scatter plots
        faithful_scatter.set_offsets(faithful_coords)
        unfaithful_scatter.set_offsets(unfaithful_coords)
        
        # Update title
        title.set_text(f'Activation Progression: Layer {layer}')
        
        # Compute statistics
        f_mean, f_std = faithful_proj.mean(), faithful_proj.std()
        u_mean, u_std = unfaithful_proj.mean(), unfaithful_proj.std()
        separation = abs(f_mean - u_mean)
        
        stats_text.set_text(
            f'Layer {layer}\n'
            f'Faithful:   μ={f_mean:+.3f}, σ={f_std:.3f}\n'
            f'Unfaithful: μ={u_mean:+.3f}, σ={u_std:.3f}\n'
            f'Separation: {separation:.3f}'
        )
        
        return faithful_scatter, unfaithful_scatter, title, stats_text
    
    # Create animation
    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(layers),
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save
    saved_path = save_animation(anim, output_file, fps=fps)
    plt.close()
    
    return saved_path


def create_pca_animation(
    pca_projections,
    layers,
    output_file="results/activation_visualizations/layer_progression_pca.mp4",
    fps=2,
    title_prefix="PCA Projection"
):
    """
    Create animation using 2D PCA projection.
    """
    print(f"\n🎬 Creating PCA animation...")
    
    # Prepare data
    n_faithful = len(pca_projections[layers[0]]['faithful'])
    n_unfaithful = len(pca_projections[layers[0]]['unfaithful'])
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Initialize scatter plots
    faithful_scatter = ax.scatter([], [], c='blue', alpha=0.6, s=100,
                                  label=f'Faithful (n={n_faithful})', edgecolors='black', linewidths=0.5)
    unfaithful_scatter = ax.scatter([], [], c='red', alpha=0.6, s=100,
                                    label=f'Unfaithful (n={n_unfaithful})', edgecolors='black', linewidths=0.5)
    
    # Compute global axis limits (across all layers)
    all_x, all_y = [], []
    for layer in layers:
        faithful_pca = pca_projections[layer]['faithful']
        unfaithful_pca = pca_projections[layer]['unfaithful']
        all_x.extend(faithful_pca[:, 0])
        all_x.extend(unfaithful_pca[:, 0])
        all_y.extend(faithful_pca[:, 1])
        all_y.extend(unfaithful_pca[:, 1])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Title and annotations
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center',
                   fontsize=16, fontweight='bold')
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top',
                        fontsize=10, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        """Initialize animation."""
        faithful_scatter.set_offsets(np.empty((0, 2)))
        unfaithful_scatter.set_offsets(np.empty((0, 2)))
        return faithful_scatter, unfaithful_scatter, title, stats_text
    
    def update(frame):
        """Update animation for each frame (layer)."""
        layer = layers[frame]
        
        # Get PCA projections for this layer
        faithful_pca = pca_projections[layer]['faithful']
        unfaithful_pca = pca_projections[layer]['unfaithful']
        
        # Update scatter plots
        faithful_scatter.set_offsets(faithful_pca)
        unfaithful_scatter.set_offsets(unfaithful_pca)
        
        # Update title
        title.set_text(f'{title_prefix}: Layer {layer}')
        
        # Compute statistics
        f_centroid = faithful_pca.mean(axis=0)
        u_centroid = unfaithful_pca.mean(axis=0)
        separation = np.linalg.norm(f_centroid - u_centroid)
        
        # Compute within-group spread
        f_spread = np.mean([np.linalg.norm(p - f_centroid) for p in faithful_pca])
        u_spread = np.mean([np.linalg.norm(p - u_centroid) for p in unfaithful_pca])
        
        stats_text.set_text(
            f'Layer {layer}\n'
            f'Separation: {separation:.3f}\n'
            f'Faithful spread: {f_spread:.3f}\n'
            f'Unfaithful spread: {u_spread:.3f}'
        )
        
        return faithful_scatter, unfaithful_scatter, title, stats_text
    
    # Create animation
    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(layers),
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save
    saved_path = save_animation(anim, output_file, fps=fps)
    plt.close()
    
    return saved_path


def create_combined_animation(
    probe_projections,
    pca_projections,
    layers,
    output_file="results/activation_visualizations/layer_progression_combined.mp4",
    fps=2
):
    """
    Create animation with probe projection as X-axis and first PCA component as Y-axis.
    
    This combines the most interpretable axis (probe direction = "faithfulness")
    with the most variance-capturing axis (PC1).
    """
    print(f"\n🎬 Creating combined (probe + PCA) animation...")
    
    # Prepare data
    n_faithful = len(probe_projections[layers[0]]['faithful'])
    n_unfaithful = len(probe_projections[layers[0]]['unfaithful'])
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Initialize scatter plots
    faithful_scatter = ax.scatter([], [], c='blue', alpha=0.6, s=100,
                                  label=f'Faithful (n={n_faithful})', edgecolors='black', linewidths=0.5)
    unfaithful_scatter = ax.scatter([], [], c='red', alpha=0.6, s=100,
                                    label=f'Unfaithful (n={n_unfaithful})', edgecolors='black', linewidths=0.5)
    
    # Compute global axis limits
    all_x, all_y = [], []
    for layer in layers:
        all_x.extend(probe_projections[layer]['faithful'])
        all_x.extend(probe_projections[layer]['unfaithful'])
        all_y.extend(pca_projections[layer]['faithful'][:, 0])
        all_y.extend(pca_projections[layer]['unfaithful'][:, 0])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel('Probe Direction (Faithfulness)', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC1 (Maximum Variance)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Title and annotations
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center',
                   fontsize=16, fontweight='bold')
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top',
                        fontsize=10, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        """Initialize animation."""
        faithful_scatter.set_offsets(np.empty((0, 2)))
        unfaithful_scatter.set_offsets(np.empty((0, 2)))
        return faithful_scatter, unfaithful_scatter, title, stats_text
    
    def update(frame):
        """Update animation for each frame (layer)."""
        layer = layers[frame]
        
        # Get projections for this layer
        faithful_probe = probe_projections[layer]['faithful']
        unfaithful_probe = probe_projections[layer]['unfaithful']
        faithful_pca = pca_projections[layer]['faithful'][:, 0]  # Use PC1
        unfaithful_pca = pca_projections[layer]['unfaithful'][:, 0]
        
        # Create 2D coordinates
        faithful_coords = np.column_stack([faithful_probe, faithful_pca])
        unfaithful_coords = np.column_stack([unfaithful_probe, unfaithful_pca])
        
        # Update scatter plots
        faithful_scatter.set_offsets(faithful_coords)
        unfaithful_scatter.set_offsets(unfaithful_coords)
        
        # Update title
        title.set_text(f'Activation Progression: Layer {layer}')
        
        # Compute statistics
        f_probe_mean = faithful_probe.mean()
        u_probe_mean = unfaithful_probe.mean()
        probe_separation = abs(f_probe_mean - u_probe_mean)
        
        f_centroid = faithful_coords.mean(axis=0)
        u_centroid = unfaithful_coords.mean(axis=0)
        euclidean_separation = np.linalg.norm(f_centroid - u_centroid)
        
        stats_text.set_text(
            f'Layer {layer}\n'
            f'Probe separation: {probe_separation:.3f}\n'
            f'Euclidean separation: {euclidean_separation:.3f}\n'
            f'Faithful probe: {f_probe_mean:+.3f}\n'
            f'Unfaithful probe: {u_probe_mean:+.3f}'
        )
        
        return faithful_scatter, unfaithful_scatter, title, stats_text
    
    # Create animation
    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(layers),
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save
    saved_path = save_animation(anim, output_file, fps=fps)
    plt.close()
    
    return saved_path


def main():
    """Main workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create layer progression animations')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 12, 18, 24],
                       help='Layers to visualize (default: 6 12 18 24)')
    parser.add_argument('--fps', type=int, default=2,
                       help='Animation speed in frames per second (default: 2)')
    parser.add_argument('--mode', choices=['probe', 'pca', 'global_pca', 'combined', 'all'],
                       default='all', help='Visualization mode (default: all)')
    parser.add_argument('--activations-dir', type=str, default='data/activations',
                       help='Directory containing activation files')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LAYER PROGRESSION ANIMATION")
    print("="*60)
    print(f"Layers: {args.layers}")
    print(f"FPS: {args.fps}")
    print(f"Mode: {args.mode}")
    print()
    
    # Step 1: Load activations
    print("[1/4] Loading activations...")
    activations = load_all_layer_activations(args.layers, args.activations_dir)
    
    if not activations:
        print("❌ No activations loaded. Exiting.")
        return
    
    # Step 2: Load probe directions
    print("\n[2/4] Loading probe directions...")
    probe_directions = load_probe_directions(args.layers)
    
    # Step 3: Compute projections
    print("\n[3/4] Computing projections...")
    
    probe_projections = None
    if probe_directions and args.mode in ['probe', 'combined', 'all']:
        print("  Computing probe direction projections...")
        probe_projections = project_onto_probe_direction(activations, probe_directions, args.layers)
    
    pca_projections = None
    if args.mode in ['pca', 'all']:
        print("  Computing per-layer PCA...")
        pca_projections = compute_pca_per_layer(activations, args.layers)
    
    global_pca_projections = None
    if args.mode in ['global_pca', 'combined', 'all']:
        print("  Computing global PCA (shared across layers)...")
        global_pca_projections = compute_global_pca(activations, args.layers)
    
    # Step 4: Create animations
    print("\n[4/4] Creating animations...")
    
    created_files = []
    
    if args.mode in ['probe', 'all'] and probe_projections:
        path = create_probe_projection_animation(
            probe_projections, args.layers, fps=args.fps
        )
        created_files.append(path)
    
    if args.mode in ['pca', 'all'] and pca_projections:
        path = create_pca_animation(
            pca_projections, args.layers,
            output_file="results/activation_visualizations/layer_progression_pca_per_layer.mp4",
            fps=args.fps,
            title_prefix="PCA Projection (Per-Layer)"
        )
        created_files.append(path)
    
    if args.mode in ['global_pca', 'all'] and global_pca_projections:
        path = create_pca_animation(
            global_pca_projections, args.layers,
            output_file="results/activation_visualizations/layer_progression_pca_global.mp4",
            fps=args.fps,
            title_prefix="PCA Projection (Global)"
        )
        created_files.append(path)
    
    if args.mode in ['combined', 'all'] and probe_projections and global_pca_projections:
        path = create_combined_animation(
            probe_projections, global_pca_projections, args.layers, fps=args.fps
        )
        created_files.append(path)
    
    # Summary
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"Created {len(created_files)} animation(s):")
    for path in created_files:
        print(f"  - {path}")
    print("\nTo view, open the MP4 files in your video player.")
    print("To change speed, use --fps flag (e.g., --fps 1 for slower, --fps 4 for faster)")


if __name__ == "__main__":
    main()

