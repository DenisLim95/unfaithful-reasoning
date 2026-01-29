#!/usr/bin/env python3
"""
Diagnose why the probe fails to generalize.

This script investigates:
1. Distribution of projections (training vs test)
2. Optimal threshold
3. Individual example analysis
4. Potential distribution shift
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from mechanistic.train_probes import LinearProbe
except ImportError:
    from src.mechanistic.train_probes import LinearProbe


def load_data(layer, train_dir="data/activations", test_dir="data/test_activations"):
    """Load training and test activations for a layer."""
    
    # Training data
    train_path = Path(train_dir) / f"layer_{layer}_activations.pt"
    train_data = torch.load(train_path)
    
    X_train_f = train_data['faithful']
    X_train_u = train_data['unfaithful']
    X_train = torch.cat([X_train_f, X_train_u], dim=0)
    y_train = torch.cat([
        torch.ones(len(X_train_f)),
        torch.zeros(len(X_train_u))
    ])
    
    # Test data
    test_path = Path(test_dir) / f"layer_{layer}_activations.pt"
    test_data = torch.load(test_path)
    
    X_test_f = test_data['faithful']
    X_test_u = test_data['unfaithful']
    X_test = torch.cat([X_test_f, X_test_u], dim=0)
    y_test = torch.cat([
        torch.ones(len(X_test_f)),
        torch.zeros(len(X_test_u))
    ])
    
    return X_train, y_train, X_test, y_test


def analyze_projections(layer):
    """Analyze projection distributions for training vs test."""
    
    print(f"\n{'='*70}")
    print(f"LAYER {layer} ANALYSIS")
    print(f"{'='*70}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(layer)
    
    # Load probe
    probe_results = torch.load('results/probe_results/all_probe_results.pt', weights_only=False)
    direction = probe_results[f'layer_{layer}'].direction
    
    # Compute projections
    with torch.no_grad():
        if X_train.dtype != direction.dtype:
            direction = direction.to(X_train.dtype)
        
        train_proj = (X_train @ direction).numpy()
        test_proj = (X_test @ direction).numpy()
    
    # Split by label
    train_faithful = train_proj[y_train == 1]
    train_unfaithful = train_proj[y_train == 0]
    test_faithful = test_proj[y_test == 1]
    test_unfaithful = test_proj[y_test == 0]
    
    # Print statistics
    print(f"\nTRAINING DATA (n={len(train_proj)}):")
    print(f"  Faithful:   mean={train_faithful.mean():+.3f}, std={train_faithful.std():.3f}, range=[{train_faithful.min():+.3f}, {train_faithful.max():+.3f}]")
    print(f"  Unfaithful: mean={train_unfaithful.mean():+.3f}, std={train_unfaithful.std():.3f}, range=[{train_unfaithful.min():+.3f}, {train_unfaithful.max():+.3f}]")
    print(f"  Separation: {abs(train_faithful.mean() - train_unfaithful.mean()):.3f}")
    
    print(f"\nTEST DATA (n={len(test_proj)}):")
    print(f"  Faithful:   mean={test_faithful.mean():+.3f}, std={test_faithful.std():.3f}, range=[{test_faithful.min():+.3f}, {test_faithful.max():+.3f}]")
    print(f"  Unfaithful: mean={test_unfaithful.mean():+.3f}, std={test_unfaithful.std():.3f}, range=[{test_unfaithful.min():+.3f}, {test_unfaithful.max():+.3f}]")
    print(f"  Separation: {abs(test_faithful.mean() - test_unfaithful.mean()):.3f}")
    
    # Check for distribution shift
    print(f"\nDISTRIBUTION SHIFT CHECK:")
    print(f"  Training faithful mean: {train_faithful.mean():+.3f}")
    print(f"  Test faithful mean:     {test_faithful.mean():+.3f}")
    print(f"  Shift: {test_faithful.mean() - train_faithful.mean():+.3f}")
    print()
    print(f"  Training unfaithful mean: {train_unfaithful.mean():+.3f}")
    print(f"  Test unfaithful mean:     {test_unfaithful.mean():+.3f}")
    print(f"  Shift: {test_unfaithful.mean() - train_unfaithful.mean():+.3f}")
    
    # Find optimal threshold on test data
    print(f"\nTHRESHOLD ANALYSIS:")
    
    # Try different thresholds
    thresholds = np.linspace(test_proj.min(), test_proj.max(), 100)
    accuracies = []
    
    for thresh in thresholds:
        preds = (test_proj > thresh).astype(float)
        acc = accuracy_score(y_test.numpy(), preds)
        accuracies.append(acc)
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    print(f"  Test median threshold: {np.median(test_proj):+.3f} → accuracy: {accuracy_score(y_test.numpy(), (test_proj > np.median(test_proj)).astype(float)):.1%}")
    print(f"  Train median threshold: {np.median(train_proj):+.3f} → accuracy: {accuracy_score(y_test.numpy(), (test_proj > np.median(train_proj)).astype(float)):.1%}")
    print(f"  Optimal threshold: {best_threshold:+.3f} → accuracy: {best_accuracy:.1%}")
    
    # Compute AUC
    fpr, tpr, _ = roc_curve(y_test.numpy(), test_proj)
    auc_score = auc(fpr, tpr)
    print(f"  AUC-ROC: {auc_score:.3f}")
    
    if auc_score < 0.55:
        print(f"  ⚠️  AUC near 0.5 → probe has no discriminative power!")
    
    return {
        'train_faithful': train_faithful,
        'train_unfaithful': train_unfaithful,
        'test_faithful': test_faithful,
        'test_unfaithful': test_unfaithful,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'auc': auc_score
    }


def plot_distributions(layer, results):
    """Plot training vs test projection distributions."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Training distribution
    ax = axes[0]
    ax.hist(results['train_faithful'], bins=20, alpha=0.6, color='blue', label='Faithful', edgecolor='black')
    ax.hist(results['train_unfaithful'], bins=20, alpha=0.6, color='red', label='Unfaithful', edgecolor='black')
    ax.axvline(results['train_faithful'].mean(), color='blue', linestyle='--', linewidth=2, label=f'F mean={results["train_faithful"].mean():.3f}')
    ax.axvline(results['train_unfaithful'].mean(), color='red', linestyle='--', linewidth=2, label=f'U mean={results["train_unfaithful"].mean():.3f}')
    ax.set_xlabel('Projection onto Probe Direction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Layer {layer} - TRAINING DATA (n={len(results["train_faithful"]) + len(results["train_unfaithful"])})', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test distribution
    ax = axes[1]
    ax.hist(results['test_faithful'], bins=20, alpha=0.6, color='blue', label='Faithful', edgecolor='black')
    ax.hist(results['test_unfaithful'], bins=20, alpha=0.6, color='red', label='Unfaithful', edgecolor='black')
    ax.axvline(results['test_faithful'].mean(), color='blue', linestyle='--', linewidth=2, label=f'F mean={results["test_faithful"].mean():.3f}')
    ax.axvline(results['test_unfaithful'].mean(), color='red', linestyle='--', linewidth=2, label=f'U mean={results["test_unfaithful"].mean():.3f}')
    ax.axvline(results['best_threshold'], color='green', linestyle=':', linewidth=3, 
              label=f'Best threshold={results["best_threshold"]:.3f} (acc={results["best_accuracy"]:.1%})')
    ax.set_xlabel('Projection onto Probe Direction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Layer {layer} - TEST DATA (n={len(results["test_faithful"]) + len(results["test_unfaithful"])}) - AUC={results["auc"]:.3f}',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(f'results/probe_diagnostics/layer_{layer}_distribution_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_path}")
    plt.close()


def main():
    """Run diagnostics for all layers."""
    
    print("="*70)
    print("PROBE GENERALIZATION DIAGNOSTIC")
    print("="*70)
    print("\nInvestigating why the probe fails to generalize to test data...")
    
    layers = [6, 12, 18, 24]
    all_results = {}
    
    for layer in layers:
        results = analyze_projections(layer)
        all_results[layer] = results
        plot_distributions(layer, results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nTest Set Performance:")
    for layer in layers:
        print(f"  Layer {layer}: AUC={all_results[layer]['auc']:.3f}, "
              f"Best Acc={all_results[layer]['best_accuracy']:.1%}")
    
    print("\nKey Findings:")
    
    # Check if all AUCs are near 0.5
    all_auc = [all_results[layer]['auc'] for layer in layers]
    if max(all_auc) < 0.60:
        print("  🚨 All layers have AUC < 0.60 → Probe has minimal discriminative power!")
        print("  🚨 This suggests severe overfitting or the signal doesn't generalize.")
    
    # Check for distribution shift
    for layer in layers:
        train_f_mean = all_results[layer]['train_faithful'].mean()
        test_f_mean = all_results[layer]['test_faithful'].mean()
        shift = abs(test_f_mean - train_f_mean)
        if shift > 0.5:
            print(f"  ⚠️  Layer {layer}: Large distribution shift ({shift:.3f}) between train and test!")
    
    print("\nPlots saved to: results/probe_diagnostics/")
    print("\nRecommendation: Review the distribution plots to understand the failure mode.")


if __name__ == "__main__":
    main()



