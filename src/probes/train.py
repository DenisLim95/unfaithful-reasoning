"""
Probe training utilities.

This module provides functions for training and evaluating linear probes.
"""

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Results from probe training."""
    layer: int
    accuracy: float
    auc: float
    probe: 'LinearProbe'
    direction: torch.Tensor  # The learned faithfulness direction
    
    def __repr__(self):
        return f"ProbeResult(layer={self.layer}, accuracy={self.accuracy:.3f}, auc={self.auc:.3f})"


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)
    
    def get_direction(self):
        """Get the probe direction (weight vector)."""
        return self.linear.weight.squeeze(0)


def train_probe(
    faithful_acts: torch.Tensor,
    unfaithful_acts: torch.Tensor,
    test_size: float = 0.2,
    epochs: int = 100,
    lr: float = 0.01
) -> ProbeResult:
    """
    Train a linear probe to distinguish faithful from unfaithful activations.
    
    Args:
        faithful_acts: Tensor of shape [n_faithful, d_model]
        unfaithful_acts: Tensor of shape [n_unfaithful, d_model]
        test_size: Fraction of data to use for testing (default: 0.2)
        epochs: Number of training epochs (default: 100)
        lr: Learning rate (default: 0.01)
    
    Returns:
        ProbeResult with trained probe and metrics
    """
    # Prepare data
    X = torch.cat([faithful_acts, unfaithful_acts], dim=0)
    y = torch.cat([
        torch.ones(len(faithful_acts)),
        torch.zeros(len(unfaithful_acts))
    ])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(),
        test_size=test_size,
        random_state=42,
        stratify=y.numpy()
    )
    
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    
    # Initialize probe
    d_model = X.shape[1]
    probe = LinearProbe(d_model)
    
    # Training
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        
        loss.backward()
        optimizer.step()
    
    # Evaluate
    probe.eval()
    with torch.no_grad():
        logits_test = probe(X_test)
        probs = torch.sigmoid(logits_test)
        preds = (probs > 0.5).float()
        
        accuracy = accuracy_score(y_test.numpy(), preds.numpy())
        
        try:
            auc = roc_auc_score(y_test.numpy(), probs.numpy())
        except ValueError:
            auc = 0.5  # Fallback if only one class
    
    # Get direction
    direction = probe.get_direction()
    
    return ProbeResult(
        layer=-1,  # Will be set by caller
        accuracy=accuracy,
        auc=auc,
        probe=probe,
        direction=direction
    )


def evaluate_probe(
    probe_result: ProbeResult,
    test_faithful_acts: torch.Tensor,
    test_unfaithful_acts: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate a trained probe on new test data.
    
    Args:
        probe_result: Trained probe result
        test_faithful_acts: Test faithful activations
        test_unfaithful_acts: Test unfaithful activations
    
    Returns:
        Dict with 'accuracy' and 'auc' keys
    """
    # Prepare test data
    X_test = torch.cat([test_faithful_acts, test_unfaithful_acts], dim=0)
    y_test = torch.cat([
        torch.ones(len(test_faithful_acts)),
        torch.zeros(len(test_unfaithful_acts))
    ])
    
    # Ensure types match
    direction = probe_result.direction
    if X_test.dtype != direction.dtype:
        direction = direction.to(X_test.dtype)
    
    # Make predictions using direction
    with torch.no_grad():
        projections = X_test @ direction
        predictions = (projections > projections.median()).float()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
    
    try:
        auc = roc_auc_score(y_test.numpy(), projections.numpy())
    except ValueError:
        auc = 0.5
    
    return {
        'accuracy': accuracy,
        'auc': auc
    }


def train_probes_for_layers(
    activation_dir: str,
    layers: list = [6, 12, 18, 24]
) -> Dict[str, ProbeResult]:
    """
    Train probes for multiple layers.
    
    Args:
        activation_dir: Directory containing cached activations
        layers: List of layer numbers
    
    Returns:
        Dict mapping 'layer_{N}' -> ProbeResult
    """
    from src.data import load_activations
    
    results = {}
    
    for layer in layers:
        print(f"\nTraining probe for layer {layer}...")
        
        # Load activations
        data = load_activations(layer, activation_dir)
        faithful_acts = data['faithful']
        unfaithful_acts = data['unfaithful']
        
        print(f"  Data: {len(faithful_acts)} faithful, {len(unfaithful_acts)} unfaithful")
        print(f"  Dimension: {data['d_model']}")
        
        # Train probe
        result = train_probe(faithful_acts, unfaithful_acts)
        result.layer = layer
        
        print(f"  ✓ Accuracy: {result.accuracy*100:.1f}%")
        print(f"  ✓ AUC-ROC: {result.auc:.3f}")
        
        results[f"layer_{layer}"] = result
    
    return results

