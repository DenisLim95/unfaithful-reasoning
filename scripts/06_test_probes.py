#!/usr/bin/env python3
"""
Script 06: Test Probes

Test trained probes on new test data to evaluate generalization.

Usage:
    # Basic usage (requires test activations)
    python scripts/06_test_probes.py \\
        --test-activations data/test_activations
    
    # Custom paths
    python scripts/06_test_probes.py \\
        --probes results/probe_results/all_probe_results.pt \\
        --test-activations data/test_activations

Output:
    Prints comparison of training vs test performance for each layer.
    Shows whether probes generalize to new data.

Dependencies:
    - Input: trained probes from 05_train_probes.py
    - Input: test activations (cache activations on test set first)
    - Requires: sklearn, torch
"""

import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.probes import evaluate_probe
from src.data import load_activations


def main():
    parser = argparse.ArgumentParser(
        description='Test probe generalization on new data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--probes', type=str, 
                       default='results/probe_results/all_probe_results.pt',
                       help='Path to trained probes (default: results/probe_results/all_probe_results.pt)')
    parser.add_argument('--test-activations', type=str, required=True,
                       help='Directory containing test activations')
    
    args = parser.parse_args()
    
    # Load trained probes
    probe_path = Path(args.probes)
    if not probe_path.exists():
        print(f"❌ Error: Trained probes not found: {probe_path}")
        print("\nPlease run 05_train_probes.py first")
        return 1
    
    print(f"Loading trained probes from {probe_path}...")
    probe_results = torch.load(probe_path, weights_only=False)
    print(f"✓ Loaded probes for {len(probe_results)} layers")
    
    # Test on each layer
    print(f"\n{'='*60}")
    print("TESTING PROBES ON NEW DATA")
    print(f"{'='*60}")
    
    test_results = {}
    
    for layer_key, probe_result in probe_results.items():
        layer_num = probe_result.layer
        
        # Load test activations
        test_act_path = Path(args.test_activations)
        try:
            print(f"\n{layer_key}:")
            test_data = load_activations(layer_num, str(test_act_path))
            
            print(f"  Test data: {len(test_data['faithful'])} faithful, "
                  f"{len(test_data['unfaithful'])} unfaithful")
            
            # Evaluate
            metrics = evaluate_probe(
                probe_result,
                test_data['faithful'],
                test_data['unfaithful']
            )
            
            test_results[layer_key] = metrics
            
            print(f"  Test Accuracy: {metrics['accuracy']*100:.1f}%")
            print(f"  Test AUC-ROC: {metrics['auc']:.3f}")
            
        except FileNotFoundError:
            print(f"  ⚠️  No test activations found for layer {layer_num}")
            continue
    
    # Summary comparison
    if test_results:
        print(f"\n{'='*60}")
        print("COMPARISON: Training vs Test Performance")
        print(f"{'='*60}")
        
        for layer_key in test_results:
            orig_acc = probe_results[layer_key].accuracy
            test_acc = test_results[layer_key]['accuracy']
            
            print(f"\n{layer_key}:")
            print(f"  Training accuracy: {orig_acc*100:.1f}%")
            print(f"  Test accuracy:     {test_acc*100:.1f}%")
            print(f"  Change:            {(test_acc - orig_acc)*100:+.1f} percentage points")
            
            # Interpretation
            if abs(test_acc - orig_acc) < 0.05:
                print(f"  → Probe generalizes well ✓")
            elif test_acc < 0.55:
                print(f"  → Probe performs at chance level (overfitting) ✗")
            else:
                print(f"  → Probe shows some generalization")
        
        print(f"{'='*60}")
        
        # Overall assessment
        avg_train = sum(probe_results[k].accuracy for k in test_results) / len(test_results)
        avg_test = sum(test_results[k]['accuracy'] for k in test_results) / len(test_results)
        
        print(f"\nOverall:")
        print(f"  Average training accuracy: {avg_train*100:.1f}%")
        print(f"  Average test accuracy:     {avg_test*100:.1f}%")
        print(f"  Generalization gap:        {(avg_train - avg_test)*100:.1f} percentage points")
        
        if avg_test < 0.55:
            print(f"\n⚠️  Warning: Probes do not generalize to new data")
            print(f"  This suggests:")
            print(f"  - Training set too small")
            print(f"  - Probes learned spurious patterns")
            print(f"  - No clear linear separation exists")
        elif avg_train - avg_test < 0.1:
            print(f"\n✓ Probes generalize reasonably well")
        else:
            print(f"\n⚠️  Moderate overfitting detected")
    else:
        print(f"\n❌ No test results available")
        print(f"Make sure test activations are cached in: {args.test_activations}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

