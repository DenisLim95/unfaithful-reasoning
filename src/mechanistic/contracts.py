"""
Phase 3 Data Contracts - Type Definitions
These types encode the Phase 3 specification as executable constraints.
"""

from dataclasses import dataclass
from typing import Dict, Literal
import torch
from pathlib import Path


# Phase 3 Contract: Supported layers (fixed, not extensible)
PHASE3_LAYERS = [6, 12, 18, 24]
LayerID = Literal[6, 12, 18, 24]


# Phase 3 Contract: Minimum samples required
MIN_FAITHFUL_SAMPLES = 10
MIN_UNFAITHFUL_SAMPLES = 10


@dataclass
class ActivationCache:
    """
    Phase 3 Data Contract 1: Activation Cache File Format
    
    Represents: data/activations/layer_{N}_activations.pt
    
    Invariants (enforced):
    - faithful.shape = [n_faithful, d_model] where n_faithful >= 10
    - unfaithful.shape = [n_unfaithful, d_model] where n_unfaithful >= 10
    - faithful.shape[1] == unfaithful.shape[1] (same d_model)
    - dtype is float32 or float16
    - NO sequence dimension (must be pre-pooled)
    """
    faithful: torch.Tensor
    unfaithful: torch.Tensor
    layer: LayerID
    
    def __post_init__(self):
        """Enforce Phase 3 contracts."""
        # Check dimensions
        if self.faithful.ndim != 2:
            raise ValueError(
                f"Phase 3 Contract Violation: faithful tensor must be 2D "
                f"(got shape {self.faithful.shape}). "
                f"Activations must be mean-pooled over sequence."
            )
        
        if self.unfaithful.ndim != 2:
            raise ValueError(
                f"Phase 3 Contract Violation: unfaithful tensor must be 2D "
                f"(got shape {self.unfaithful.shape}). "
                f"Activations must be mean-pooled over sequence."
            )
        
        # Check minimum samples
        n_faithful = self.faithful.shape[0]
        if n_faithful < MIN_FAITHFUL_SAMPLES:
            raise ValueError(
                f"Phase 3 Contract Violation: Need at least {MIN_FAITHFUL_SAMPLES} "
                f"faithful samples, got {n_faithful}"
            )
        
        n_unfaithful = self.unfaithful.shape[0]
        if n_unfaithful < MIN_UNFAITHFUL_SAMPLES:
            raise ValueError(
                f"Phase 3 Contract Violation: Need at least {MIN_UNFAITHFUL_SAMPLES} "
                f"unfaithful samples, got {n_unfaithful}"
            )
        
        # Check d_model matches
        if self.faithful.shape[1] != self.unfaithful.shape[1]:
            raise ValueError(
                f"Phase 3 Contract Violation: d_model mismatch - "
                f"faithful has {self.faithful.shape[1]}, "
                f"unfaithful has {self.unfaithful.shape[1]}"
            )
        
        # Check dtype
        if self.faithful.dtype not in [torch.float32, torch.float16]:
            raise ValueError(
                f"Phase 3 Contract Violation: faithful tensor must be float32 or float16, "
                f"got {self.faithful.dtype}"
            )
        
        if self.unfaithful.dtype not in [torch.float32, torch.float16]:
            raise ValueError(
                f"Phase 3 Contract Violation: unfaithful tensor must be float32 or float16, "
                f"got {self.unfaithful.dtype}"
            )
        
        # Check layer is supported
        if self.layer not in PHASE3_LAYERS:
            raise ValueError(
                f"Phase 3 Contract Violation: Layer {self.layer} not supported. "
                f"Phase 3 only supports layers: {PHASE3_LAYERS}"
            )
    
    @property
    def d_model(self) -> int:
        """Return model dimension."""
        return self.faithful.shape[1]
    
    @property
    def n_faithful(self) -> int:
        """Return number of faithful samples."""
        return self.faithful.shape[0]
    
    @property
    def n_unfaithful(self) -> int:
        """Return number of unfaithful samples."""
        return self.unfaithful.shape[0]


@dataclass
class ProbeResult:
    """
    Phase 3 Data Contract 2: Probe Result Format
    
    Represents one layer's probe training result.
    
    Invariants (enforced):
    - accuracy in [0, 1]
    - auc in [0, 1]
    - direction.shape = [d_model]
    """
    layer: str  # "layer_6", "layer_12", etc.
    accuracy: float
    auc: float
    probe: torch.nn.Module
    direction: torch.Tensor
    
    def __post_init__(self):
        """Enforce Phase 3 contracts."""
        # Check accuracy range
        if not (0 <= self.accuracy <= 1):
            raise ValueError(
                f"Phase 3 Contract Violation: accuracy must be in [0, 1], "
                f"got {self.accuracy}"
            )
        
        # Check AUC range
        if not (0 <= self.auc <= 1):
            raise ValueError(
                f"Phase 3 Contract Violation: auc must be in [0, 1], "
                f"got {self.auc}"
            )
        
        # Check direction is 1D
        if self.direction.ndim != 1:
            raise ValueError(
                f"Phase 3 Contract Violation: direction must be 1D vector, "
                f"got shape {self.direction.shape}"
            )
        
        # Check layer format
        if not self.layer.startswith("layer_"):
            raise ValueError(
                f"Phase 3 Contract Violation: layer must be 'layer_N', "
                f"got '{self.layer}'"
            )


@dataclass
class Phase3Config:
    """
    Phase 3 Configuration
    
    Fixed parameters from specification (not user-configurable in Phase 3).
    """
    # Model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Layers to analyze (fixed in Phase 3)
    layers: list[int] = None
    
    # Sampling limits
    max_faithful: int = 30
    max_unfaithful: int = 20
    
    # Probe training
    num_epochs: int = 50
    learning_rate: float = 1e-3
    train_test_split: float = 0.2
    random_seed: int = 42
    
    # Paths (Phase 2 outputs -> Phase 3 inputs)
    responses_path: str = "data/responses/model_1.5B_responses.jsonl"
    faithfulness_path: str = "data/processed/faithfulness_scores.csv"
    
    # Phase 3 outputs
    activations_dir: str = "data/activations"
    probe_results_dir: str = "results/probe_results"
    
    def __post_init__(self):
        """Set default layers if not provided."""
        if self.layers is None:
            self.layers = PHASE3_LAYERS.copy()
        
        # Enforce Phase 3 layer constraint
        for layer in self.layers:
            if layer not in PHASE3_LAYERS:
                raise ValueError(
                    f"Phase 3 does not support layer {layer}. "
                    f"Supported layers: {PHASE3_LAYERS}"
                )


class Phase3Error(Exception):
    """Base exception for Phase 3 contract violations."""
    pass


class Phase3InputError(Phase3Error):
    """Phase 2 outputs are missing or invalid."""
    pass


class Phase3OutputError(Phase3Error):
    """Phase 3 failed to produce required outputs."""
    pass


def validate_phase2_outputs_exist(config: Phase3Config) -> None:
    """
    Check that Phase 2 outputs exist before starting Phase 3.
    
    Phase 3 REQUIRES Phase 2 to be complete.
    """
    if not Path(config.responses_path).exists():
        raise Phase3InputError(
            f"Phase 2 output missing: {config.responses_path}\n"
            f"Phase 3 requires Phase 2 to be completed first."
        )
    
    if not Path(config.faithfulness_path).exists():
        raise Phase3InputError(
            f"Phase 2 output missing: {config.faithfulness_path}\n"
            f"Phase 3 requires Phase 2 to be completed first."
        )

