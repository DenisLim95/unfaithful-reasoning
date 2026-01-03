"""
Activation caching utilities.

This module provides functions for caching model activations during inference.
"""

import torch
from typing import Dict, List, Tuple
from pathlib import Path


def cache_activations_for_prompt(
    model,
    tokenizer,
    prompt: str,
    layers: List[int]
) -> Dict[int, torch.Tensor]:
    """
    Cache activations for a single prompt at specified layers.
    
    Args:
        model: Loaded model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt
        layers: List of layer indices to cache (e.g., [6, 12, 18, 24])
    
    Returns:
        Dict mapping layer_num -> activation tensor (mean-pooled over sequence)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract activations at specified layers
    activations = {}
    for layer in layers:
        # Get hidden states at this layer: shape [batch, seq_len, d_model]
        hidden_states = outputs.hidden_states[layer]
        
        # Mean pool over sequence dimension: shape [batch, d_model]
        mean_pooled = hidden_states.mean(dim=1)
        
        # Move to CPU and squeeze batch dimension: shape [d_model]
        activations[layer] = mean_pooled.squeeze(0).cpu()
    
    return activations


def cache_activations_for_responses(
    model,
    tokenizer,
    responses: List[Dict],
    scores: List[Dict],
    layers: List[int] = [6, 12, 18, 24],
    output_dir: str = "data/activations"
) -> None:
    """
    Cache activations for a list of responses, grouped by faithfulness.
    
    Args:
        model: Loaded model
        tokenizer: Corresponding tokenizer
        responses: List of response dicts (from JSONL)
        scores: List of score dicts (from CSV)
        layers: Layers to cache (default: [6, 12, 18, 24])
        output_dir: Directory to save cached activations
    """
    # Create score lookup
    score_dict = {s['pair_id']: s for s in scores}
    
    # Group responses by faithfulness
    faithful_responses = []
    unfaithful_responses = []
    
    for resp in responses:
        pair_id = resp['pair_id']
        if pair_id in score_dict:
            if score_dict[pair_id]['faithful']:
                faithful_responses.append(resp)
            else:
                unfaithful_responses.append(resp)
    
    print(f"  Faithful responses: {len(faithful_responses)}")
    print(f"  Unfaithful responses: {len(unfaithful_responses)}")
    
    # Cache for each layer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for layer in layers:
        print(f"\n  Caching layer {layer}...")
        
        # Cache faithful activations
        faithful_acts = []
        for resp in faithful_responses:
            question = resp['question']
            acts = cache_activations_for_prompt(model, tokenizer, question, [layer])
            faithful_acts.append(acts[layer])
        
        # Cache unfaithful activations
        unfaithful_acts = []
        for resp in unfaithful_responses:
            question = resp['question']
            acts = cache_activations_for_prompt(model, tokenizer, question, [layer])
            unfaithful_acts.append(acts[layer])
        
        # Stack and save
        if faithful_acts and unfaithful_acts:
            faithful_tensor = torch.stack(faithful_acts)
            unfaithful_tensor = torch.stack(unfaithful_acts)
            
            cache_data = {
                'faithful': faithful_tensor,
                'unfaithful': unfaithful_tensor,
                'layer': layer,
                'd_model': faithful_tensor.shape[1]
            }
            
            output_file = output_path / f"layer_{layer}_activations.pt"
            torch.save(cache_data, output_file)
            
            print(f"    ✓ Saved: {len(faithful_acts)} faithful, {len(unfaithful_acts)} unfaithful")
            print(f"       Shape: {faithful_tensor.shape}")


def load_activations(layer: int, activation_dir: str = "data/activations") -> Dict:
    """
    Load cached activations for a specific layer.
    
    Args:
        layer: Layer number
        activation_dir: Directory containing cached activations
    
    Returns:
        Dict with 'faithful', 'unfaithful', 'layer', 'd_model' keys
    """
    cache_file = Path(activation_dir) / f"layer_{layer}_activations.pt"
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Activation cache not found: {cache_file}")
    
    return torch.load(cache_file, weights_only=False)


def validate_activation_cache(activation_dir: str = "data/activations") -> List[str]:
    """
    Validate activation cache files.
    
    Args:
        activation_dir: Directory containing cached activations
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    cache_dir = Path(activation_dir)
    
    if not cache_dir.exists():
        errors.append(f"Cache directory not found: {cache_dir}")
        return errors
    
    # Check for expected files
    expected_layers = [6, 12, 18, 24]
    for layer in expected_layers:
        cache_file = cache_dir / f"layer_{layer}_activations.pt"
        
        if not cache_file.exists():
            errors.append(f"Missing cache for layer {layer}")
            continue
        
        # Load and validate
        try:
            data = torch.load(cache_file, weights_only=False)
            
            # Check required keys
            required_keys = ['faithful', 'unfaithful', 'layer']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Layer {layer}: missing key '{key}'")
            
            # Check shapes match
            if 'faithful' in data and 'unfaithful' in data:
                faithful_shape = data['faithful'].shape
                unfaithful_shape = data['unfaithful'].shape
                
                if faithful_shape[1] != unfaithful_shape[1]:
                    errors.append(
                        f"Layer {layer}: dimension mismatch "
                        f"(faithful: {faithful_shape[1]}, unfaithful: {unfaithful_shape[1]})"
                    )
                
                # Check minimum samples
                if faithful_shape[0] < 10:
                    errors.append(f"Layer {layer}: too few faithful samples ({faithful_shape[0]})")
                if unfaithful_shape[0] < 10:
                    errors.append(f"Layer {layer}: too few unfaithful samples ({unfaithful_shape[0]})")
        
        except Exception as e:
            errors.append(f"Layer {layer}: error loading cache - {str(e)}")
    
    return errors

