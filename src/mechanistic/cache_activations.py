"""
Phase 3 Task 3.2: Cache Activations

Cache residual stream activations for faithful vs unfaithful responses.
Implements Phase 3 Data Contract 1 (Activation Cache Files).
"""

import json
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Phase 3 contracts (handle both relative and absolute imports)
try:
    from .contracts import (
        ActivationCache,
        Phase3Config,
        Phase3Error,
        validate_phase2_outputs_exist,
        PHASE3_LAYERS,
        MIN_FAITHFUL_SAMPLES,
        MIN_UNFAITHFUL_SAMPLES
    )
except ImportError:
    from src.mechanistic.contracts import (
    ActivationCache,
    Phase3Config,
    Phase3Error,
    validate_phase2_outputs_exist,
        PHASE3_LAYERS,
        MIN_FAITHFUL_SAMPLES,
        MIN_UNFAITHFUL_SAMPLES
)


def load_faithfulness_labels(faithfulness_path: str) -> tuple[List[str], List[str]]:
    """
    Load faithful and unfaithful pair IDs from Phase 2 outputs.
    
    Args:
        faithfulness_path: Path to faithfulness_scores.csv from Phase 2
    
    Returns:
        (faithful_ids, unfaithful_ids)
    """
    df = pd.read_csv(faithfulness_path)
    
    faithful_ids = df[df['is_faithful'] == True]['pair_id'].tolist()
    unfaithful_ids = df[df['is_faithful'] == False]['pair_id'].tolist()
    
    return faithful_ids, unfaithful_ids


def load_responses_map(responses_path: str) -> Dict[str, Dict]:
    """
    Load model responses from Phase 2 outputs.
    
    Args:
        responses_path: Path to model responses JSONL from Phase 2
    
    Returns:
        Dictionary mapping pair_id -> {"q1": response, "q2": response}
    """
    import jsonlines
    
    responses_by_pair = {}
    
    with jsonlines.open(responses_path) as reader:
        for response in reader:
            pair_id = response['pair_id']
            variant = response['variant']
            
            if pair_id not in responses_by_pair:
                responses_by_pair[pair_id] = {}
            
            responses_by_pair[pair_id][variant] = response
    
    return responses_by_pair


def cache_activations_for_pairs(
    model,
    responses_by_pair: Dict,
    pair_ids: List[str],
    layers: List[int],
    label: str
) -> Dict[int, torch.Tensor]:
    """
    Cache activations for a list of pair IDs.
    
    Args:
        model: HookedTransformer model
        responses_by_pair: Mapping of pair_id -> responses
        pair_ids: List of pair IDs to cache
        layers: List of layer numbers to cache
        label: Label for progress bar (e.g., "faithful")
    
    Returns:
        Dictionary mapping layer -> stacked activations [n_pairs, d_model]
    """
    # Initialize storage for each layer
    acts_by_layer = {layer: [] for layer in layers}
    
    for pair_id in tqdm(pair_ids, desc=f"Caching {label}"):
        # Use q1 response (per spec)
        if pair_id not in responses_by_pair:
            print(f"Warning: {pair_id} not found in responses, skipping")
            continue
        
        if 'q1' not in responses_by_pair[pair_id]:
            print(f"Warning: {pair_id} has no q1 response, skipping")
            continue
        
        response = responses_by_pair[pair_id]['q1']
        prompt = response['question']
        
        # Run model with cache
        with torch.no_grad():
            logits, cache = model.run_with_cache(prompt)
        
        # Extract activations at each layer
        for layer in layers:
            # Get residual stream after layer (per spec: blocks.{layer}.hook_resid_post)
            acts = cache[f"blocks.{layer}.hook_resid_post"]  # Shape: [1, seq_len, d_model]
            
            # Mean-pool over sequence (per Phase 3 contract)
            acts_pooled = acts.mean(dim=1)  # Shape: [1, d_model]
            
            acts_by_layer[layer].append(acts_pooled.cpu())
    
    # Stack into tensors
    stacked = {}
    for layer in layers:
        if len(acts_by_layer[layer]) > 0:
            stacked[layer] = torch.cat(acts_by_layer[layer], dim=0)  # [n_pairs, d_model]
        else:
            raise Phase3Error(f"No activations cached for {label} at layer {layer}")
    
    return stacked


def cache_activations(config: Phase3Config = None) -> None:
    """
    Phase 3 Task 3.2: Cache activations for faithful vs unfaithful responses.
    
    Produces: data/activations/layer_{N}_activations.pt (4 files)
    Each file satisfies Phase 3 Data Contract 1.
    
    Args:
        config: Phase 3 configuration (uses defaults if None)
    """
    if config is None:
        config = Phase3Config()
    
    print("=" * 60)
    print("PHASE 3 TASK 3.2: Cache Activations")
    print("=" * 60)
    
    # Check Phase 2 is complete
    print("\n[1/6] Checking Phase 2 outputs...")
    validate_phase2_outputs_exist(config)
    print("   ✓ Phase 2 outputs found")
    
    # Load faithfulness labels
    print("\n[2/6] Loading faithfulness labels...")
    faithful_ids, unfaithful_ids = load_faithfulness_labels(config.faithfulness_path)
    
    print(f"   Found {len(faithful_ids)} faithful pairs")
    print(f"   Found {len(unfaithful_ids)} unfaithful pairs")
    
    # Apply sampling limits
    faithful_ids = faithful_ids[:config.max_faithful]
    unfaithful_ids = unfaithful_ids[:config.max_unfaithful]
    
    print(f"   Using {len(faithful_ids)} faithful pairs")
    print(f"   Using {len(unfaithful_ids)} unfaithful pairs")
    
    # Check minimum samples (Phase 3 contract requirement)
    if len(faithful_ids) < MIN_FAITHFUL_SAMPLES:
        raise Phase3Error(
            f"Insufficient faithful samples: need {MIN_FAITHFUL_SAMPLES}, "
            f"got {len(faithful_ids)}. Phase 2 may have too high faithfulness rate."
        )
    
    if len(unfaithful_ids) < MIN_UNFAITHFUL_SAMPLES:
        raise Phase3Error(
            f"Insufficient unfaithful samples: need {MIN_UNFAITHFUL_SAMPLES}, "
            f"got {len(unfaithful_ids)}. Phase 2 may have too high faithfulness rate."
        )
    
    # Load responses
    print("\n[3/6] Loading model responses...")
    responses_by_pair = load_responses_map(config.responses_path)
    print(f"   ✓ Loaded {len(responses_by_pair)} response pairs")
    
    # Load model
    print(f"\n[4/6] Loading model: {config.model_name}")
    
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        raise Phase3Error(
            "TransformerLens not installed. Phase 3 requires:\n"
            "  pip install transformer-lens\n"
            "See phased_implementation_plan.md Task 3.1"
        )
    
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )
    print(f"   ✓ Model loaded on {model.cfg.device}")
    
    # Cache activations
    print(f"\n[5/6] Caching activations at layers {config.layers}...")
    
    print("\n   Caching faithful responses...")
    faithful_acts = cache_activations_for_pairs(
        model, responses_by_pair, faithful_ids, config.layers, "faithful"
    )
    
    print("\n   Caching unfaithful responses...")
    unfaithful_acts = cache_activations_for_pairs(
        model, responses_by_pair, unfaithful_ids, config.layers, "unfaithful"
    )
    
    # Save to disk (enforcing Phase 3 contract)
    print("\n[6/6] Saving activation caches...")
    output_dir = Path(config.activations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for layer in config.layers:
        # Create ActivationCache to enforce contract
        cache = ActivationCache(
            faithful=faithful_acts[layer],
            unfaithful=unfaithful_acts[layer],
            layer=layer
        )
        
        # Save in contract format
        output_path = output_dir / f"layer_{layer}_activations.pt"
        torch.save({
            'faithful': cache.faithful,
            'unfaithful': cache.unfaithful
        }, output_path)
        
        print(f"   ✓ layer_{layer}: {cache.n_faithful} faithful, "
              f"{cache.n_unfaithful} unfaithful, d_model={cache.d_model}")
        print(f"     Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ PHASE 3 TASK 3.2 COMPLETE")
    print("=" * 60)
    print(f"\nActivations cached for {len(config.layers)} layers")
    print(f"Output: {config.activations_dir}/layer_{{N}}_activations.pt")


if __name__ == "__main__":
    # Run with default config
    cache_activations()

