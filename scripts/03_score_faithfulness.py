#!/usr/bin/env python3
"""
Script 03: Score Faithfulness

Score model responses for faithfulness using either:
1. Answer correctness - checks if both Q1 and Q2 answers are correct
2. LLM judge - checks if reasoning is consistent with answer

Usage:
    # Score with answer correctness (fast, no API needed)
    python scripts/03_score_faithfulness.py \\
        --responses data/responses/responses.jsonl \\
        --method answer-correctness
    
    # Score with LLM judge (requires OpenAI API key)
    export OPENAI_API_KEY="sk-..."
    python scripts/03_score_faithfulness.py \\
        --responses data/responses/responses.jsonl \\
        --method llm-judge
    
    # Specify custom output
    python scripts/03_score_faithfulness.py \\
        --responses data/responses/responses.jsonl \\
        --method llm-judge \\
        --output results/custom_scores.csv

Input format (JSONL):
    Each line must have: pair_id, variant, question, reasoning, extracted_answer, expected_answer

Output format (CSV):
    Depends on method - see --help for details

Dependencies:
    - Input: responses.jsonl from 02_generate_responses.py
    - Output: scores.csv for use in 04_cache_activations.py
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

from src.faithfulness import (
    judge_reasoning_consistency,
    score_answer_correctness,
    estimate_cost
)


def load_responses(responses_file: str) -> list:
    """Load responses from JSONL file."""
    responses = []
    with open(responses_file, 'r') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def group_by_pairs(responses: list) -> dict:
    """Group responses by pair_id."""
    pairs_dict = {}
    for resp in responses:
        pair_id = resp['pair_id']
        if pair_id not in pairs_dict:
            pairs_dict[pair_id] = {}
        pairs_dict[pair_id][resp['variant']] = resp
    return pairs_dict


def score_with_answer_correctness(pairs_dict: dict) -> pd.DataFrame:
    """Score using answer correctness method."""
    print("Scoring with answer correctness method...")
    
    scores = []
    for pair_id, variants in tqdm(pairs_dict.items(), desc="Scoring pairs"):
        q1_resp = variants.get('q1', {})
        q2_resp = variants.get('q2', {})
        
        result = score_answer_correctness(
            q1_answer=q1_resp.get('extracted_answer', ''),
            q2_answer=q2_resp.get('extracted_answer', ''),
            q1_expected=q1_resp.get('expected_answer', ''),
            q2_expected=q2_resp.get('expected_answer', '')
        )
        
        scores.append({
            'pair_id': pair_id,
            'faithful': result['faithful'],
            'q1_correct': result['q1_correct'],
            'q2_correct': result['q2_correct'],
            'q1_answer': q1_resp.get('extracted_answer', ''),
            'q2_answer': q2_resp.get('extracted_answer', ''),
            'q1_expected': q1_resp.get('expected_answer', ''),
            'q2_expected': q2_resp.get('expected_answer', '')
        })
    
    return pd.DataFrame(scores)


def score_with_llm_judge(pairs_dict: dict, api_key: str = None) -> pd.DataFrame:
    """Score using LLM judge method."""
    print("Scoring with LLM judge method...")
    print("Using GPT-4o-mini to evaluate reasoning consistency...")
    
    scores = []
    for pair_id, variants in tqdm(pairs_dict.items(), desc="Judging pairs"):
        q1_resp = variants.get('q1', {})
        q2_resp = variants.get('q2', {})
        
        # Judge Q1
        q1_judgment = judge_reasoning_consistency(
            question=q1_resp.get('question', ''),
            reasoning=q1_resp.get('reasoning', ''),
            answer=q1_resp.get('extracted_answer', ''),
            api_key=api_key
        )
        
        # Judge Q2
        q2_judgment = judge_reasoning_consistency(
            question=q2_resp.get('question', ''),
            reasoning=q2_resp.get('reasoning', ''),
            answer=q2_resp.get('extracted_answer', ''),
            api_key=api_key
        )
        
        # Faithful if BOTH reasonings are consistent
        faithful = q1_judgment['is_consistent'] and q2_judgment['is_consistent']
        
        scores.append({
            'pair_id': pair_id,
            'faithful': faithful,
            'q1_reasoning_consistent': q1_judgment['is_consistent'],
            'q2_reasoning_consistent': q2_judgment['is_consistent'],
            'q1_confidence': q1_judgment['confidence'],
            'q2_confidence': q2_judgment['confidence'],
            'q1_explanation': q1_judgment['explanation'],
            'q2_explanation': q2_judgment['explanation'],
            'q1_answer': q1_resp.get('extracted_answer', ''),
            'q2_answer': q2_resp.get('extracted_answer', ''),
            'q1_reasoning': q1_resp.get('reasoning', '')[:200],  # Truncate for CSV
            'q2_reasoning': q2_resp.get('reasoning', '')[:200]
        })
    
    return pd.DataFrame(scores)


def print_summary(df: pd.DataFrame, method: str):
    """Print summary statistics."""
    faithful_count = df['faithful'].sum()
    total_count = len(df)
    
    print(f"\n{'='*70}")
    print(f"FAITHFULNESS SCORING SUMMARY ({method})")
    print(f"{'='*70}")
    print(f"Total pairs: {total_count}")
    print(f"Faithful: {faithful_count} ({faithful_count/total_count*100:.1f}%)")
    print(f"Unfaithful: {total_count - faithful_count} ({(total_count-faithful_count)/total_count*100:.1f}%)")
    
    if method == "LLM Judge":
        # Additional statistics for LLM judge
        if 'q1_confidence' in df.columns:
            high_conf = df[df['q1_confidence'] == 'high'].shape[0]
            med_conf = df[df['q1_confidence'] == 'medium'].shape[0]
            low_conf = df[df['q1_confidence'] == 'low'].shape[0]
            print(f"\nQ1 Confidence distribution:")
            print(f"  High: {high_conf} ({high_conf/total_count*100:.1f}%)")
            print(f"  Medium: {med_conf} ({med_conf/total_count*100:.1f}%)")
            print(f"  Low: {low_conf} ({low_conf/total_count*100:.1f}%)")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Score faithfulness of model responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--responses', type=str, required=True,
                       help='Path to responses JSONL file (from 02_generate_responses.py)')
    parser.add_argument('--output', type=str, default='data/processed/faithfulness_scores.csv',
                       help='Output CSV file path (default: data/processed/faithfulness_scores.csv)')
    parser.add_argument('--method', type=str, choices=['answer-correctness', 'llm-judge'],
                       default='llm-judge',
                       help='Scoring method (default: llm-judge)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--estimate-cost', action='store_true',
                       help='Estimate cost for LLM judge and exit')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.responses).exists():
        print(f"❌ Error: Responses file not found: {args.responses}")
        print("\nDid you run 02_generate_responses.py first?")
        return 1
    
    # Load responses
    print(f"Loading responses from {args.responses}...")
    responses = load_responses(args.responses)
    print(f"✓ Loaded {len(responses)} responses")
    
    # Group by pairs
    pairs_dict = group_by_pairs(responses)
    print(f"✓ Grouped into {len(pairs_dict)} pairs")
    
    # Estimate cost if requested
    if args.estimate_cost and args.method == 'llm-judge':
        cost_info = estimate_cost(len(pairs_dict))
        print(f"\n{'='*70}")
        print("COST ESTIMATE FOR LLM JUDGE")
        print(f"{'='*70}")
        print(f"Pairs to evaluate: {cost_info['num_pairs']}")
        print(f"Total judgments: {cost_info['num_judgments']} (2 per pair)")
        print(f"Estimated tokens: {cost_info['input_tokens']:,} input, {cost_info['output_tokens']:,} output")
        print(f"Estimated cost: ${cost_info['estimated_cost_usd']:.4f} USD")
        print(f"{'='*70}\n")
        return 0
    
    # Check API key for LLM judge
    if args.method == 'llm-judge':
        if args.openai_api_key is None:
            import os
            if 'OPENAI_API_KEY' not in os.environ:
                print("❌ Error: OpenAI API key required for LLM judge")
                print("Set OPENAI_API_KEY environment variable or use --openai-api-key")
                print("\nOr use --method answer-correctness instead")
                return 1
    
    # Score faithfulness
    if args.method == 'llm-judge':
        df = score_with_llm_judge(pairs_dict, api_key=args.openai_api_key)
        method_name = "LLM Judge"
    else:
        df = score_with_answer_correctness(pairs_dict)
        method_name = "Answer Correctness"
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved scores to {output_path}")
    
    # Print summary
    print_summary(df, method_name)
    
    print(f"Next step: Run 04_cache_activations.py to cache neural activations")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

