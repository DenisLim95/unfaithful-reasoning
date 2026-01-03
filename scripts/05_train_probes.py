#!/usr/bin/env python3
"""
Script 05: Train Probes

Train linear probes to detect faithfulness from neural activations.

Usage:
    # Basic usage
    python scripts/05_train_probes.py
    
    # Custom activation directory and layers
    python scripts/05_train_probes.py \\
        --activations data/activations \\
        --layers 6 12 18 24
    
    # Custom output
    python scripts/05_train_probes.py \\
        --activations data/activations \\
        --output results/probe_results/my_probes.pt

Output:
    Saves trained probes to: results/probe_results/all_probe_results.pt
    File contains dict mapping 'layer_{N}' -> ProbeResult with:
        - layer: layer number
        - accuracy: test set accuracy
        - auc: AUC-ROC score
        - probe: trained LinearProbe model
        - direction: learned faithfulness direction vector

Dependencies:
    - Input: activation caches from 04_cache_activations.py
    - Output: trained probes for use in 06_test_probes.py
    - Requires: sklearn, torch
"""

import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.probes import train_probes_for_layers
from src.data import validate_activation_cache


def main():
    parser = argparse.ArgumentParser(
        description='Train linear probes for faithfulness detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--activations', type=str, default='data/activations',
                       help='Directory containing cached activations (default: data/activations)')
    parser.add_argument('--output', type=str, default='results/probe_results/all_probe_results.pt',
                       help='Output file for trained probes (default: results/probe_results/all_probe_results.pt)')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 12, 18, 24],
                       help='Layers to train probes for (default: 6 12 18 24)')
    
    args = parser.parse_args()
    
    # Validate activation caches
    print("Validating activation caches...")
    errors = validate_activation_cache(args.activations)
    
    if errors:
        print(f"❌ Activation cache validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease run 04_cache_activations.py first")
        return 1
    
    print(f"✓ Activation caches validated")
    
    # Train probes
    print(f"\nTraining probes for layers {args.layers}...")
    results = train_probes_for_layers(
        activation_dir=args.activations,
        layers=args.layers
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(results, output_path)
    print(f"\n✓ Saved trained probes to {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PROBE TRAINING SUMMARY")
    print(f"{'='*60}")
    for layer_key, result in results.items():
        print(f"\n{layer_key}:")
        print(f"  Accuracy: {result.accuracy*100:.1f}%")
        print(f"  AUC-ROC: {result.auc:.3f}")
    print(f"{'='*60}")
    
    # Find best layer
    best_layer = max(results.items(), key=lambda x: x[1].accuracy)
    print(f"\nBest performing layer: {best_layer[0]}")
    print(f"  Accuracy: {best_layer[1].accuracy*100:.1f}%")
    
    print(f"\nNext step: Run 06_test_probes.py to test generalization on new data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

