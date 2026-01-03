#!/usr/bin/env python3
"""
Script 04: Cache Activations

Cache neural activations for faithful vs unfaithful responses.

Usage:
    # Basic usage
    python scripts/04_cache_activations.py \\
        --responses data/responses/responses.jsonl \\
        --scores data/processed/faithfulness_scores.csv
    
    # Custom layers and output
    python scripts/04_cache_activations.py \\
        --responses data/responses/responses.jsonl \\
        --scores data/processed/faithfulness_scores.csv \\
        --layers 6 12 18 24 \\
        --output data/activations
    
    # Validate cached activations
    python scripts/04_cache_activations.py --validate

Output:
    Creates one file per layer: data/activations/layer_{N}_activations.pt
    Each file contains:
        - 'faithful': tensor of shape [n_faithful, d_model]
        - 'unfaithful': tensor of shape [n_unfaithful, d_model]
        - 'layer': layer number
        - 'd_model': model dimension

Dependencies:
    - Input: responses.jsonl from 02_generate_responses.py
    - Input: scores.csv from 03_score_faithfulness.py
    - Output: activation caches for use in 05_train_probes.py
    - Requires: GPU, transformers, torch
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model
from src.data import cache_activations_for_responses, validate_activation_cache


def load_responses(responses_file: str) -> list:
    """Load responses from JSONL file."""
    responses = []
    with open(responses_file, 'r') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def load_scores(scores_file: str) -> list:
    """Load scores from CSV file."""
    df = pd.read_csv(scores_file)
    return df.to_dict('records')


def main():
    parser = argparse.ArgumentParser(
        description='Cache neural activations for faithfulness analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--responses', type=str,
                       help='Path to responses JSONL file (from 02_generate_responses.py)')
    parser.add_argument('--scores', type=str,
                       help='Path to scores CSV file (from 03_score_faithfulness.py)')
    parser.add_argument('--output', type=str, default='data/activations',
                       help='Output directory for cached activations (default: data/activations)')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 12, 18, 24],
                       help='Layers to cache (default: 6 12 18 24)')
    parser.add_argument('--model', type=str,
                       default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='HuggingFace model identifier')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing activation caches and exit')
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        print("Validating activation caches...")
        errors = validate_activation_cache(args.output)
        
        if errors:
            print(f"\n❌ Found {len(errors)} validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print(f"✓ All activation caches are valid")
            
            # Show summary
            from src.data import load_activations
            print(f"\nCache summary:")
            for layer in [6, 12, 18, 24]:
                try:
                    data = load_activations(layer, args.output)
                    print(f"  Layer {layer}: {data['faithful'].shape[0]} faithful, "
                          f"{data['unfaithful'].shape[0]} unfaithful, "
                          f"d_model={data['d_model']}")
                except FileNotFoundError:
                    pass
        
        return 0
    
    # Check required arguments
    if not args.responses or not args.scores:
        print("❌ Error: --responses and --scores are required")
        print("\nUsage:")
        print("  python scripts/04_cache_activations.py \\")
        print("      --responses data/responses/responses.jsonl \\")
        print("      --scores data/processed/faithfulness_scores.csv")
        print("\nOr validate existing caches:")
        print("  python scripts/04_cache_activations.py --validate")
        return 1
    
    # Validate inputs
    if not Path(args.responses).exists():
        print(f"❌ Error: Responses file not found: {args.responses}")
        print("\nDid you run 02_generate_responses.py first?")
        return 1
    
    if not Path(args.scores).exists():
        print(f"❌ Error: Scores file not found: {args.scores}")
        print("\nDid you run 03_score_faithfulness.py first?")
        return 1
    
    # Load data
    print(f"Loading responses from {args.responses}...")
    responses = load_responses(args.responses)
    print(f"✓ Loaded {len(responses)} responses")
    
    print(f"Loading scores from {args.scores}...")
    scores = load_scores(args.scores)
    print(f"✓ Loaded {len(scores)} scores")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Cache activations
    print(f"\nCaching activations at layers {args.layers}...")
    print(f"Output directory: {args.output}")
    
    cache_activations_for_responses(
        model=model,
        tokenizer=tokenizer,
        responses=responses,
        scores=scores,
        layers=args.layers,
        output_dir=args.output
    )
    
    print(f"\n✓ Cached activations saved to {args.output}/")
    
    # Validate
    print(f"\nValidating caches...")
    errors = validate_activation_cache(args.output)
    
    if errors:
        print(f"⚠️  Found {len(errors)} validation warnings:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"✓ All caches validated successfully")
    
    print(f"\nNext step: Run 05_train_probes.py to train linear probes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

