#!/usr/bin/env python3
"""
Extract example reasoning traces for MATS application.

Outputs:
- Console output of faithful and unfaithful examples
- Copy-paste ready format for write-up

Usage:
    python extract_examples.py
"""

import jsonlines
import pandas as pd
from pathlib import Path


def extract_examples():
    """Extract and display example reasoning traces."""
    
    print("=" * 80)
    print("EXTRACTING EXAMPLE REASONING TRACES")
    print("=" * 80)
    print()
    
    # Load data
    df = pd.read_csv('data/processed/faithfulness_scores.csv')
    
    # Get one faithful and one unfaithful example
    faithful_examples = df[df['is_faithful']]
    unfaithful_examples = df[~df['is_faithful']]
    
    if len(faithful_examples) == 0:
        print("✗ No faithful examples found!")
        return
    
    if len(unfaithful_examples) == 0:
        print("✗ No unfaithful examples found!")
        return
    
    # Pick good examples (both answers extracted with high confidence if possible)
    if 'extraction_confidence' in df.columns:
        faithful_examples = faithful_examples.sort_values('extraction_confidence', ascending=False)
        unfaithful_examples = unfaithful_examples.sort_values('extraction_confidence', ascending=False)
    
    faithful_id = faithful_examples.iloc[0]['pair_id']
    unfaithful_id = unfaithful_examples.iloc[0]['pair_id']
    
    print(f"Selected examples:")
    print(f"  Faithful:   {faithful_id}")
    print(f"  Unfaithful: {unfaithful_id}")
    print()
    
    # Load responses
    responses_path = Path('data/responses/model_1.5B_responses.jsonl')
    if not responses_path.exists():
        print(f"✗ Responses file not found: {responses_path}")
        return
    
    # Extract faithful example
    print("=" * 80)
    print("FAITHFUL EXAMPLE (for Appendix C)")
    print("=" * 80)
    print()
    
    faithful_responses = []
    with jsonlines.open(responses_path) as reader:
        for response in reader:
            if response['pair_id'] == faithful_id:
                faithful_responses.append(response)
    
    if len(faithful_responses) == 2:
        print("```")
        print(f"Pair ID: {faithful_id}")
        print()
        
        # Q1
        q1 = faithful_responses[0] if faithful_responses[0]['variant'] == 'q1' else faithful_responses[1]
        print(f"Q1: {q1['question']}")
        print()
        
        reasoning = q1.get('reasoning', q1.get('think_section', 'N/A'))
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + "..."
        print(f"Reasoning: {reasoning}")
        print()
        
        print(f"Final Answer: {q1.get('extracted_answer', 'N/A')}")
        print(f"Expected: {q1.get('expected_answer', 'N/A')}")
        print(f"Correct: {q1.get('is_correct', 'N/A')}")
        print()
        
        # Q2
        q2 = faithful_responses[1] if faithful_responses[1]['variant'] == 'q2' else faithful_responses[0]
        print(f"Q2: {q2['question']}")
        print()
        
        reasoning = q2.get('reasoning', q2.get('think_section', 'N/A'))
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + "..."
        print(f"Reasoning: {reasoning}")
        print()
        
        print(f"Final Answer: {q2.get('extracted_answer', 'N/A')}")
        print(f"Expected: {q2.get('expected_answer', 'N/A')}")
        print(f"Correct: {q2.get('is_correct', 'N/A')}")
        print()
        
        print("→ FAITHFUL: Both answers correct and consistent")
        print("```")
    
    # Extract unfaithful example
    print()
    print("=" * 80)
    print("UNFAITHFUL EXAMPLE (for Appendix C)")
    print("=" * 80)
    print()
    
    unfaithful_responses = []
    with jsonlines.open(responses_path) as reader:
        for response in reader:
            if response['pair_id'] == unfaithful_id:
                unfaithful_responses.append(response)
    
    if len(unfaithful_responses) == 2:
        print("```")
        print(f"Pair ID: {unfaithful_id}")
        print()
        
        # Q1
        q1 = unfaithful_responses[0] if unfaithful_responses[0]['variant'] == 'q1' else unfaithful_responses[1]
        print(f"Q1: {q1['question']}")
        print()
        
        reasoning = q1.get('reasoning', q1.get('think_section', 'N/A'))
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + "..."
        print(f"Reasoning: {reasoning}")
        print()
        
        print(f"Final Answer: {q1.get('extracted_answer', 'N/A')}")
        print(f"Expected: {q1.get('expected_answer', 'N/A')}")
        print(f"Correct: {q1.get('is_correct', 'N/A')}")
        print()
        
        # Q2
        q2 = unfaithful_responses[1] if unfaithful_responses[1]['variant'] == 'q2' else unfaithful_responses[0]
        print(f"Q2: {q2['question']}")
        print()
        
        reasoning = q2.get('reasoning', q2.get('think_section', 'N/A'))
        if len(reasoning) > 300:
            reasoning = reasoning[:300] + "..."
        print(f"Reasoning: {reasoning}")
        print()
        
        print(f"Final Answer: {q2.get('extracted_answer', 'N/A')}")
        print(f"Expected: {q2.get('expected_answer', 'N/A')}")
        print(f"Correct: {q2.get('is_correct', 'N/A')}")
        print()
        
        # Determine unfaithfulness type
        row = df[df['pair_id'] == unfaithful_id].iloc[0]
        if row['is_consistent'] == False:
            print("→ UNFAITHFUL: Inconsistent answers despite equivalent questions")
        elif not row['q1_correct'] or not row['q2_correct']:
            print("→ UNFAITHFUL: At least one answer incorrect")
        print("```")
    
    # Additional examples for diversity
    print()
    print("=" * 80)
    print("ADDITIONAL EXAMPLES (optional for diversity)")
    print("=" * 80)
    print()
    
    # Find a few more examples of each type
    print("Faithful pairs:")
    for i, row in faithful_examples.head(3).iterrows():
        print(f"  - {row['pair_id']}: Q1={row['q1_answer']}, Q2={row['q2_answer']} (both correct)")
    
    print("\nUnfaithful pairs:")
    for i, row in unfaithful_examples.head(3).iterrows():
        print(f"  - {row['pair_id']}: Q1={row['q1_answer']}, Q2={row['q2_answer']} "
              f"({'inconsistent' if not row['is_consistent'] else 'incorrect'})")
    
    print()
    print("=" * 80)
    print("INSTRUCTIONS")
    print("=" * 80)
    print()
    print("1. Copy the examples above into APPLICATION_EXECUTIVE_SUMMARY.md")
    print("   Section: Appendix C (Sample Examples)")
    print()
    print("2. For full reasoning traces, look up pair IDs in:")
    print("   data/responses/model_1.5B_responses.jsonl")
    print()
    print("3. Optional: Include additional examples for diversity")
    print("=" * 80)


if __name__ == '__main__':
    extract_examples()

