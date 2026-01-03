#!/usr/bin/env python3
"""
Script 02: Generate Responses

Generate model responses for question pairs.

Usage:
    # Basic usage
    python scripts/02_generate_responses.py \\
        --questions data/raw/questions.json
    
    # Custom model and output
    python scripts/02_generate_responses.py \\
        --questions data/raw/questions.json \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --output data/responses/7b_responses.jsonl
    
    # Custom temperature
    python scripts/02_generate_responses.py \\
        --questions data/raw/questions.json \\
        --temperature 0.8

Output format (JSONL, one response per line):
    {
      "pair_id": "num_000",
      "variant": "q1",
      "question": "Is 900 larger than 795?",
      "response": "...",
      "reasoning": "900 has 9 in hundreds place...",
      "extracted_answer": "Yes",
      "expected_answer": "Yes",
      "is_valid_format": true,
      "timestamp": "2024-..."
    }

Dependencies:
    - Input: questions.json from 01_generate_questions.py
    - Output: responses.jsonl for use in 03_score_faithfulness.py
    - Requires: GPU, transformers, torch
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model, generate_response, parse_response


def load_questions(questions_file: str) -> list:
    """Load questions from JSON file."""
    with open(questions_file, 'r') as f:
        data = json.load(f)
    return data['pairs']


def save_response(response_dict: dict, output_file: str):
    """Append response to JSONL file."""
    with open(output_file, 'a') as f:
        f.write(json.dumps(response_dict) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Generate model responses for question pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--questions', type=str, required=True,
                       help='Path to questions JSON file (from 01_generate_questions.py)')
    parser.add_argument('--output', type=str, default='data/responses/responses.jsonl',
                       help='Output JSONL file path (default: data/responses/responses.jsonl)')
    parser.add_argument('--model', type=str, 
                       default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='HuggingFace model identifier')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature (default: 0.6)')
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='Maximum tokens to generate (default: 2048)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing responses file')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.questions).exists():
        print(f"❌ Error: Questions file not found: {args.questions}")
        print("\nDid you run 01_generate_questions.py first?")
        return 1
    
    # Load questions
    print(f"Loading questions from {args.questions}...")
    pairs = load_questions(args.questions)
    print(f"✓ Loaded {len(pairs)} question pairs")
    print(f"  Total prompts to generate: {len(pairs) * 2}")
    
    # Handle resume
    completed_ids = set()
    if args.resume and Path(args.output).exists():
        print(f"\nResuming from {args.output}...")
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    resp = json.loads(line)
                    completed_ids.add((resp['pair_id'], resp['variant']))
        print(f"✓ Found {len(completed_ids)} completed responses")
    else:
        # Create new file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear file if it exists
        open(args.output, 'w').close()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Generate responses
    print(f"\nGenerating responses...")
    print(f"  Model: {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    
    response_count = 0
    skipped_count = 0
    
    for pair in tqdm(pairs, desc="Processing pairs"):
        for variant in ['q1', 'q2']:
            # Skip if already completed
            if (pair['id'], variant) in completed_ids:
                skipped_count += 1
                continue
            
            question = pair[variant]
            expected_answer = pair[f'{variant}_answer']
            
            # Generate response
            response_text = generate_response(
                model=model,
                tokenizer=tokenizer,
                question=question,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            # Parse response
            parsed = parse_response(response_text, question)
            
            # Create response dict
            response_dict = {
                'pair_id': pair['id'],
                'variant': variant,
                'question': question,
                'response': parsed['raw_response'],
                'reasoning': parsed['reasoning'],
                'extracted_answer': parsed['answer'],
                'expected_answer': expected_answer,
                'is_valid_format': parsed['is_valid_format'],
                'timestamp': datetime.now().isoformat(),
                'generation_config': {
                    'model': args.model,
                    'temperature': args.temperature,
                    'max_tokens': args.max_tokens
                }
            }
            
            # Save immediately (streaming)
            save_response(response_dict, args.output)
            response_count += 1
    
    # Summary
    print(f"\n✓ Generated {response_count} new responses")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} already completed")
    print(f"✓ Saved to {args.output}")
    
    # Format compliance stats
    if response_count > 0:
        print(f"\nAnalyzing format compliance...")
        valid_count = 0
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    resp = json.loads(line)
                    if resp.get('is_valid_format'):
                        valid_count += 1
        
        total = len(pairs) * 2
        print(f"  Format compliance: {valid_count}/{total} ({valid_count/total*100:.1f}%)")
    
    print(f"\nNext step: Run 03_score_faithfulness.py to score faithfulness")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

