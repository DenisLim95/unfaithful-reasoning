#!/usr/bin/env python3
"""
Detailed analysis of unfaithful responses.
Shows full responses, not just excerpts.
"""

import json
import pandas as pd
from pathlib import Path
import sys


def analyze_unfaithful_responses(num_examples: int = 10):
    """Show detailed unfaithful response examples."""
    
    scores_file = Path("data/processed/test_faithfulness_scores.csv")
    responses_file = Path("data/responses/test_responses.jsonl")
    
    # Load data
    df = pd.read_csv(scores_file)
    
    responses = []
    with open(responses_file, 'r') as f:
        for line in f:
            responses.append(json.loads(line))
    
    # Group responses by pair_id
    pairs_dict = {}
    for r in responses:
        pair_id = r['pair_id']
        if pair_id not in pairs_dict:
            pairs_dict[pair_id] = []
        pairs_dict[pair_id].append(r)
    
    # Get unfaithful pairs
    unfaithful_pairs = df[~df['faithful']]
    
    print("="*80)
    print(f"DETAILED UNFAITHFUL RESPONSE ANALYSIS")
    print("="*80)
    print(f"\nTotal pairs: {len(df)}")
    print(f"Unfaithful: {len(unfaithful_pairs)} ({100*len(unfaithful_pairs)/len(df):.1f}%)")
    print(f"Showing first {min(num_examples, len(unfaithful_pairs))} examples...")
    
    # Categorize unfaithful reasons
    categories = {
        'both_wrong': [],
        'q1_wrong': [],
        'q2_wrong': [],
        'extraction_failed': []
    }
    
    for _, row in unfaithful_pairs.iterrows():
        pair_id = row['pair_id']
        pair_responses = pairs_dict.get(pair_id, [])
        if len(pair_responses) >= 2:
            r1, r2 = pair_responses[0], pair_responses[1]
            
            q1_correct = r1['extracted_answer'] == r1['expected_answer']
            q2_correct = r2['extracted_answer'] == r2['expected_answer']
            
            if r1['extracted_answer'] == 'Unknown' or r2['extracted_answer'] == 'Unknown':
                categories['extraction_failed'].append((r1, r2))
            elif not q1_correct and not q2_correct:
                categories['both_wrong'].append((r1, r2))
            elif not q1_correct:
                categories['q1_wrong'].append((r1, r2))
            elif not q2_correct:
                categories['q2_wrong'].append((r1, r2))
    
    print(f"\nBreakdown:")
    print(f"  - Both answers wrong: {len(categories['both_wrong'])}")
    print(f"  - Q1 wrong only: {len(categories['q1_wrong'])}")
    print(f"  - Q2 wrong only: {len(categories['q2_wrong'])}")
    print(f"  - Extraction failed: {len(categories['extraction_failed'])}")
    
    # Show examples from each category
    for category_name, category_pairs in categories.items():
        if not category_pairs:
            continue
        
        print("\n" + "="*80)
        print(f"CATEGORY: {category_name.upper().replace('_', ' ')}")
        print("="*80)
        
        for i, (r1, r2) in enumerate(category_pairs[:min(3, len(category_pairs))]):
            print(f"\n{'-'*80}")
            print(f"Example {i+1}")
            print(f"{'-'*80}")
            
            print(f"\n📝 Q1: {r1['question']}")
            print(f"   Expected: {r1['expected_answer']}")
            print(f"   Extracted: {r1['extracted_answer']} {'✓' if r1['extracted_answer'] == r1['expected_answer'] else '✗ WRONG'}")
            print(f"\n   Full Response:")
            print(f"   {'-'*76}")
            # Wrap text nicely
            response_lines = r1['response'].split('\n')
            for line in response_lines:
                print(f"   {line}")
            print(f"   {'-'*76}")
            
            print(f"\n📝 Q2: {r2['question']}")
            print(f"   Expected: {r2['expected_answer']}")
            print(f"   Extracted: {r2['extracted_answer']} {'✓' if r2['extracted_answer'] == r2['expected_answer'] else '✗ WRONG'}")
            print(f"\n   Full Response:")
            print(f"   {'-'*76}")
            response_lines = r2['response'].split('\n')
            for line in response_lines:
                print(f"   {line}")
            print(f"   {'-'*76}")
            
            # Analysis
            print(f"\n💡 Analysis:")
            if category_name == 'extraction_failed':
                print(f"   - Extraction failed to find clear Yes/No answer")
                print(f"   - Response may be cut off, looping, or ambiguous")
            elif category_name == 'both_wrong':
                print(f"   - Model gave wrong answer to BOTH questions")
                print(f"   - Could indicate computational error or confusion")
            elif category_name == 'q1_wrong':
                print(f"   - Model got Q1 wrong but Q2 right")
                print(f"   - Inconsistent reasoning between similar questions")
            elif category_name == 'q2_wrong':
                print(f"   - Model got Q2 wrong but Q1 right")
                print(f"   - Inconsistent reasoning between similar questions")


if __name__ == "__main__":
    num = 10
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
        except ValueError:
            pass
    
    analyze_unfaithful_responses(num)

