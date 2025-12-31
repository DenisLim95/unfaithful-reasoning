#!/usr/bin/env python3
"""
Quick script to inspect model structure for Phase 3 planning.
Run this to understand your model's architecture before caching activations.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM
import argparse


def inspect_model_structure(model_name: str, load_full_model: bool = False):
    """
    Inspect model structure and print key information.
    
    Args:
        model_name: HuggingFace model identifier
        load_full_model: If True, load full model (slower but more detailed)
    """
    
    print("=" * 70)
    print(f"MODEL STRUCTURE INSPECTION: {model_name}")
    print("=" * 70)
    
    # Load config (fast)
    print("\n📋 Loading model config...")
    config = AutoConfig.from_pretrained(model_name)
    
    # Basic architecture info
    print("\n🏗️  ARCHITECTURE OVERVIEW")
    print("-" * 70)
    print(f"Model Type:              {config.model_type}")
    print(f"Number of Layers:        {config.num_hidden_layers}")
    print(f"Hidden Size (d_model):   {config.hidden_size}")
    print(f"Attention Heads:         {config.num_attention_heads}")
    print(f"Head Dimension:          {config.hidden_size // config.num_attention_heads}")
    print(f"Vocab Size:              {config.vocab_size}")
    print(f"Max Position Embeddings: {config.max_position_embeddings}")
    
    if hasattr(config, 'intermediate_size'):
        print(f"MLP Hidden Size:         {config.intermediate_size}")
    
    # Recommended layers for Phase 3
    n_layers = config.num_hidden_layers
    recommended_layers = [
        n_layers // 4,      # Early layer
        n_layers // 2,      # Middle layer  
        3 * n_layers // 4,  # Late layer
        n_layers - 1        # Final layer
    ]
    
    print("\n🎯 RECOMMENDED LAYERS FOR PHASE 3 ANALYSIS")
    print("-" * 70)
    print("For activation caching and probe training, test these layers:")
    for i, layer_idx in enumerate(recommended_layers, 1):
        position = ["Early", "Middle", "Late", "Final"][i-1]
        print(f"  Layer {layer_idx:2d}  ({position:6s} - {layer_idx/n_layers*100:.0f}% through model)")
    
    print(f"\nCommand for Phase 3:")
    print(f"  --layers {' '.join(map(str, recommended_layers))}")
    
    # Memory estimates
    print("\n💾 MEMORY ESTIMATES (FP16)")
    print("-" * 70)
    
    # Rough parameter count
    if hasattr(config, 'intermediate_size'):
        mlp_size = config.intermediate_size
    else:
        mlp_size = 4 * config.hidden_size  # Common default
    
    # Parameters per layer (rough estimate)
    params_per_layer = (
        4 * config.hidden_size ** 2 +  # QKV + O projections
        2 * config.hidden_size * mlp_size  # MLP up + down
    )
    total_params = params_per_layer * n_layers
    
    print(f"Estimated parameters:    ~{total_params/1e9:.2f}B")
    print(f"Model size (FP16):       ~{total_params * 2 / 1e9:.2f} GB")
    print(f"Activation cache (per layer): ~{config.hidden_size * 2048 * 2 / 1e6:.1f} MB (for 2048 tokens)")
    
    # If loading full model
    if load_full_model:
        print("\n🔍 LOADING FULL MODEL (this may take a minute)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"  # Load to CPU to avoid OOM
        )
        
        # Actual parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📊 ACTUAL PARAMETER COUNT: {total_params:,} ({total_params/1e9:.2f}B)")
        
        # Layer structure
        print("\n🏛️  LAYER STRUCTURE")
        print("-" * 70)
        print("First few layer names (use these for TransformerLens hooks):")
        
        layer_names = set()
        for name, _ in model.named_modules():
            # Extract layer number if present
            if 'layer' in name.lower() or 'block' in name.lower():
                parts = name.split('.')
                if len(parts) > 1:
                    layer_names.add('.'.join(parts[:2]))
        
        for name in sorted(list(layer_names))[:10]:
            print(f"  {name}")
        
        if len(layer_names) > 10:
            print(f"  ... and {len(layer_names) - 10} more")
    
    # TransformerLens compatibility
    print("\n🔧 TRANSFORMERLENS COMPATIBILITY")
    print("-" * 70)
    
    compatible_types = ['gpt2', 'gpt_neo', 'gpt_neox', 'llama', 'qwen', 'qwen2']
    if any(t in config.model_type.lower() for t in compatible_types):
        print("✅ This model type is likely compatible with TransformerLens")
        print(f"   Model type '{config.model_type}' is supported")
        print("\nTo load with TransformerLens:")
        print(f'  from transformer_lens import HookedTransformer')
        print(f'  model = HookedTransformer.from_pretrained("{model_name}")')
    else:
        print(f"⚠️  Model type '{config.model_type}' may not be directly supported")
        print("   You may need to use nnsight or raw HuggingFace hooks")
        print("\nAlternative (using nnsight):")
        print(f'  from nnsight import LanguageModel')
        print(f'  model = LanguageModel("{model_name}")')
    
    # Hook point reference
    print("\n📍 KEY ACTIVATION HOOK POINTS")
    print("-" * 70)
    print("For TransformerLens, use these hook names:")
    print("  • blocks.{N}.hook_resid_pre    - Input to layer N")
    print("  • blocks.{N}.hook_resid_mid    - After attention, before MLP")
    print("  • blocks.{N}.hook_resid_post   - Output of layer N (recommended)")
    print("  • blocks.{N}.attn.hook_pattern - Attention weights")
    print("  • blocks.{N}.attn.hook_q/k/v   - Query/Key/Value vectors")
    
    print("\n" + "=" * 70)
    print("✅ Inspection complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect model structure for mechanistic analysis"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load full model for detailed inspection (slower)"
    )
    
    args = parser.parse_args()
    
    inspect_model_structure(args.model_name, args.full)

