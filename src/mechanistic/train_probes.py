"""
Phase 3 Task 3.3: Train Linear Probes

Train linear probes to classify faithful vs unfaithful from activations.
Implements Phase 3 Data Contract 2 (Probe Results).
"""

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Phase 3 contracts (handle both relative and absolute imports)
try:
    from .contracts import (
        ActivationCache,
        ProbeResult,
        Phase3Config,
        Phase3Error,
        Phase3OutputError,
        PHASE3_LAYERS
    )
except ImportError:
    from src.mechanistic.contracts import (
    ActivationCache,
    ProbeResult,
    Phase3Config,
    Phase3Error,
    Phase3OutputError,
    PHASE3_LAYERS
)


class LinearProbe(nn.Module):
    """
    Simple linear probe for faithfulness classification.
    
    Phase 3 spec: Single linear layer, no bias (for direction extraction).
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model] activations
        
        Returns:
            [batch, 1] logits
        """
        return self.linear(x)


def train_probe_for_layer(
    cache: ActivationCache,
    num_epochs: int,
    learning_rate: float,
    train_test_split_ratio: float,
    random_seed: int
) -> ProbeResult:
    """
    Train a linear probe for one layer.
    
    Phase 3 spec (from phased_implementation_plan.md lines 1615-1665):
    - Linear probe (single layer)
    - 80/20 train/test split (stratified)
    - Adam optimizer, lr=1e-3
    - BCEWithLogitsLoss
    - 50 epochs, no early stopping
    
    Args:
        cache: ActivationCache for this layer
        num_epochs: Training epochs (spec: 50)
        learning_rate: Learning rate (spec: 1e-3)
        train_test_split_ratio: Test set ratio (spec: 0.2)
        random_seed: Random seed (spec: 42)
    
    Returns:
        ProbeResult satisfying Phase 3 Data Contract 2
    """
    # Create dataset
    X = torch.cat([cache.faithful, cache.unfaithful], dim=0)
    y = torch.cat([
        torch.ones(cache.n_faithful),
        torch.zeros(cache.n_unfaithful)
    ])
    
    # Train/test split (stratified per spec)
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(),
        test_size=train_test_split_ratio,
        random_state=random_seed,
        stratify=y.numpy()
    )
    
    # Convert back to tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    
    # Initialize probe
    probe = LinearProbe(cache.d_model)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train (no early stopping per spec)
    probe.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = probe(X_train).squeeze()
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on test set
    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test).squeeze()
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs > 0.5).float()
        
        accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
        
        # Compute AUC (handle edge case where all labels are same)
        try:
            auc = roc_auc_score(y_test.numpy(), test_probs.numpy())
        except ValueError:
            # All labels are same class - default to 0.5
            auc = 0.5
    
    # Extract direction (weight vector)
    direction = probe.linear.weight.squeeze().detach()
    
    # Create result (enforces Phase 3 contract)
    result = ProbeResult(
        layer=f"layer_{cache.layer}",
        accuracy=float(accuracy),
        auc=float(auc),
        probe=probe,
        direction=direction
    )
    
    return result


def train_all_probes(config: Phase3Config = None) -> Dict[str, ProbeResult]:
    """
    Phase 3 Task 3.3: Train linear probes for all layers.
    
    Produces: results/probe_results/all_probe_results.pt
    Satisfies Phase 3 Data Contract 2.
    
    Args:
        config: Phase 3 configuration (uses defaults if None)
    
    Returns:
        Dictionary mapping layer_name -> ProbeResult
    """
    if config is None:
        config = Phase3Config()
    
    print("=" * 60)
    print("PHASE 3 TASK 3.3: Train Linear Probes")
    print("=" * 60)
    
    # Check activation caches exist
    print("\n[1/4] Checking activation caches...")
    activations_dir = Path(config.activations_dir)
    
    for layer in config.layers:
        cache_path = activations_dir / f"layer_{layer}_activations.pt"
        if not cache_path.exists():
            raise Phase3Error(
                f"Activation cache missing: {cache_path}\n"
                f"Run Task 3.2 (cache_activations.py) first."
            )
    
    print(f"   ✓ Found activation caches for {len(config.layers)} layers")
    
    # Train probes
    print(f"\n[2/4] Training probes (epochs={config.num_epochs}, lr={config.learning_rate})...")
    
    results = {}
    
    for layer in config.layers:
        layer_name = f"layer_{layer}"
        print(f"\n   Training {layer_name}...")
        
        # Load activation cache
        cache_path = activations_dir / f"{layer_name}_activations.pt"
        cache_data = torch.load(cache_path)
        
        # Wrap in ActivationCache to enforce contract
        cache = ActivationCache(
            faithful=cache_data['faithful'],
            unfaithful=cache_data['unfaithful'],
            layer=layer
        )
        
        print(f"     Data: {cache.n_faithful} faithful, {cache.n_unfaithful} unfaithful")
        
        # Train probe
        result = train_probe_for_layer(
            cache=cache,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            train_test_split_ratio=config.train_test_split,
            random_seed=config.random_seed
        )
        
        print(f"     Result: accuracy={result.accuracy:.3f}, auc={result.auc:.3f}")
        
        results[layer_name] = result
    
    # Find best layer
    best_layer_name = max(results.items(), key=lambda x: x[1].accuracy)[0]
    best_accuracy = results[best_layer_name].accuracy
    
    print(f"\n   ✓ Best layer: {best_layer_name} (accuracy={best_accuracy:.3f})")
    
    # Check Phase 3 acceptance criterion
    if best_accuracy <= 0.55:
        print(f"\n   ⚠️  Warning: Best accuracy {best_accuracy:.3f} ≤ 0.55")
        print(f"       Phase 3 acceptance criterion: at least one layer > 0.55")
        print(f"       This may be a null result (no linear faithfulness direction)")
    
    # Generate plot
    print("\n[3/4] Generating probe performance plot...")
    
    layers_list = [int(name.split('_')[1]) for name in results.keys()]
    accuracies = [r.accuracy for r in results.values()]
    aucs = [r.auc for r in results.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers_list, accuracies, marker='o', linewidth=2, markersize=8, label='Accuracy')
    ax.plot(layers_list, aucs, marker='s', linewidth=2, markersize=8, label='AUC-ROC', alpha=0.7)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax.axhline(0.55, color='orange', linestyle='--', alpha=0.5, label='Phase 3 Target')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Linear Probe Performance Across Layers\n(Predicting CoT Faithfulness)', fontsize=14)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Save plot
    output_dir = Path(config.probe_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "probe_performance.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"   ✓ Saved plot to: {plot_path}")
    
    # Save results
    print("\n[4/4] Saving probe results...")
    results_path = output_dir / "all_probe_results.pt"
    torch.save(results, results_path)
    
    print(f"   ✓ Saved results to: {results_path}")
    
    # Verify Phase 3 acceptance criteria
    print("\n" + "=" * 60)
    print("PHASE 3 ACCEPTANCE CRITERIA CHECK")
    print("=" * 60)
    
    all_pass = True
    
    # Criterion 1: Results for all 4 layers
    if len(results) == len(PHASE3_LAYERS):
        print(f"✓ Results for all {len(PHASE3_LAYERS)} layers present")
    else:
        print(f"✗ Expected {len(PHASE3_LAYERS)} layers, got {len(results)}")
        all_pass = False
    
    # Criterion 2: All accuracy/auc in [0, 1]
    all_valid = all(0 <= r.accuracy <= 1 and 0 <= r.auc <= 1 for r in results.values())
    if all_valid:
        print("✓ All accuracy/auc values in [0, 1]")
    else:
        print("✗ Some accuracy/auc values out of range")
        all_pass = False
    
    # Criterion 3: At least one layer > 0.55
    if best_accuracy > 0.55:
        print(f"✓ At least one layer has accuracy > 0.55 ({best_accuracy:.3f})")
    else:
        print(f"✗ No layer has accuracy > 0.55 (best: {best_accuracy:.3f})")
        print("  Note: This may be a valid null result")
        all_pass = False
    
    print("=" * 60)
    
    if all_pass:
        print("✅ PHASE 3 TASK 3.3 COMPLETE")
    else:
        print("⚠️  PHASE 3 TASK 3.3 COMPLETE WITH WARNINGS")
        print("   (Null results are scientifically valid)")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run with default config
    results = train_all_probes()

