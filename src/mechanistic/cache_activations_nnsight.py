"""
Phase 3 Task 3.2: Cache Activations (HuggingFace version)

Alternative implementation using HuggingFace transformers directly.
Use this when TransformerLens doesn't support your model.
Works with ANY HuggingFace model!
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

# Import Phase 3 contracts
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
    """Load faithful and unfaithful pair IDs from Phase 2 outputs."""
    df = pd.read_csv(faithfulness_path)
    
    faithful_ids = df[df['is_faithful'] == True]['pair_id'].tolist()
    unfaithful_ids = df[df['is_faithful'] == False]['pair_id'].tolist()
    
    return faithful_ids, unfaithful_ids


def load_responses_map(responses_path: str) -> Dict:
    """Load all responses into a dictionary keyed by pair_id."""
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


def cache_activations_with_huggingface(
    model,
    tokenizer,
    responses_by_pair: Dict,
    pair_ids: List[str],
    layers: List[int],
    label: str
) -> Dict[int, torch.Tensor]:
    """
    Cache activations using HuggingFace transformers.
    
    Uses output_hidden_states=True which works with ANY model!
    
    Returns:
        Dict mapping layer -> activations tensor [n_pairs, d_model]
    """
    
    all_acts_by_layer = {layer: [] for layer in layers}
    
    for pair_id in tqdm(pair_ids, desc=f"Caching {label}"):
        # Use q1 response
        response = responses_by_pair[pair_id]['q1']
        prompt = response['question']
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Run with caching
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        
        # Extract activations at each layer and mean pool
        for layer_idx in layers:
            # hidden_states is tuple of (embed, layer_0, layer_1, ..., layer_N)
            # So layer_idx in model corresponds to hidden_states[layer_idx+1]
            acts = hidden_states[layer_idx + 1]  # [1, seq_len, d_model]
            acts_pooled = acts.mean(dim=1)  # [1, d_model]
            all_acts_by_layer[layer_idx].append(acts_pooled.cpu())
    
    # Stack into tensors
    for layer_idx in all_acts_by_layer:
        all_acts_by_layer[layer_idx] = torch.cat(all_acts_by_layer[layer_idx], dim=0)
    
    return all_acts_by_layer


def cache_activations(config: Phase3Config = None):
    """Main function to cache activations for Phase 3."""
    
    if config is None:
        config = Phase3Config()
    
    print("=" * 60)
    print("PHASE 3 TASK 3.2: Cache Activations (HuggingFace)")
    print("=" * 60)
    print()
    
    # Validate Phase 2 outputs exist
    print("[1/6] Checking Phase 2 outputs...")
    validate_phase2_outputs_exist(config)
    print("   ✓ Phase 2 outputs found")
    
    # Load faithful/unfaithful labels
    print("\n[2/6] Loading faithfulness labels...")
    faithful_ids, unfaithful_ids = load_faithfulness_labels(config.faithfulness_path)
    
    print(f"   Found {len(faithful_ids)} faithful pairs")
    print(f"   Found {len(unfaithful_ids)} unfaithful pairs")
    
    # Apply sampling limits
    faithful_ids = faithful_ids[:config.max_faithful]
    unfaithful_ids = unfaithful_ids[:config.max_unfaithful]
    
    print(f"   Using {len(faithful_ids)} faithful pairs")
    print(f"   Using {len(unfaithful_ids)} unfaithful pairs")
    
    # Check minimum samples
    if len(faithful_ids) < MIN_FAITHFUL_SAMPLES:
        raise Phase3Error(
            f"Insufficient faithful samples: need {MIN_FAITHFUL_SAMPLES}, "
            f"got {len(faithful_ids)}."
        )
    
    if len(unfaithful_ids) < MIN_UNFAITHFUL_SAMPLES:
        raise Phase3Error(
            f"Insufficient unfaithful samples: need {MIN_UNFAITHFUL_SAMPLES}, "
            f"got {len(unfaithful_ids)}."
        )
    
    # Load responses
    print("\n[3/6] Loading model responses...")
    responses_by_pair = load_responses_map(config.responses_path)
    print(f"   ✓ Loaded {len(responses_by_pair)} response pairs")
    
    # Load model with HuggingFace transformers
    print(f"\n[4/6] Loading model: {config.model_name}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    print(f"   ✓ Model loaded on {model.device}")
    
    # Cache activations
    print(f"\n[5/6] Caching activations at layers {config.layers}...")
    
    print("\n   Caching faithful responses...")
    faithful_acts = cache_activations_with_huggingface(
        model, tokenizer, responses_by_pair, faithful_ids, config.layers, "faithful"
    )
    
    print("\n   Caching unfaithful responses...")
    unfaithful_acts = cache_activations_with_huggingface(
        model, tokenizer, responses_by_pair, unfaithful_ids, config.layers, "unfaithful"
    )
    
    # Save to disk
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
        
        # Save
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
    cache_activations()

