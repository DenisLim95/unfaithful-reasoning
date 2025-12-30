#!/usr/bin/env python3
"""
Show sample unfaithful results with full details.
"""
import pandas as pd
import jsonlines
import json


def show_unfaithful_samples(num_samples=3):
    """Display unfaithful pairs with questions, responses, and answers."""
    
    # Load scores
    scores = pd.read_csv('data/processed/faithfulness_scores.csv')
    
    # Load questions
    with open('data/raw/question_pairs.json') as f:
        questions = {p['id']: p for p in json.load(f)['pairs']}
    
    # Load responses
    responses = {}
    with jsonlines.open('data/responses/model_1.5B_responses.jsonl') as reader:
        for resp in reader:
            key = (resp['pair_id'], resp['variant'])
            responses[key] = resp
    
    # Get unfaithful pairs
    unfaithful = scores[~scores['is_faithful']].head(num_samples)
    
    print("=" * 80)
    print(f"UNFAITHFUL EXAMPLES ({len(unfaithful)} shown)")
    print("=" * 80)
    print()
    
    for idx, (_, row) in enumerate(unfaithful.iterrows(), 1):
        pair_id = row['pair_id']
        q = questions[pair_id]
        r1 = responses.get((pair_id, 'q1'), {})
        r2 = responses.get((pair_id, 'q2'), {})
        
        print(f"{'=' * 80}")
        print(f"EXAMPLE {idx}: {pair_id}")
        print(f"{'=' * 80}")
        print()
        
        # Q1
        print(f"QUESTION 1:")
        print(f"  {q['q1']}")
        print()
        print(f"  Expected Answer: {row['q1_correct_answer']}")
        print()
        
        # Show think section (first 200 chars)
        think1 = r1.get('think_section', '')
        if think1:
            print(f"  Model's Reasoning (first 200 chars):")
            print(f"    {think1[:200]}...")
            print()
        
        # Show final answer
        print(f"  Model's Final Answer:")
        final1 = r1.get('final_answer', 'N/A')
        # Show first 300 chars to see the actual answer
        print(f"    {final1[:300]}")
        if len(final1) > 300:
            print(f"    ... (truncated, {len(final1)} chars total)")
        print()
        
        print(f"  Extracted Answer: {row['q1_answer_normalized']}")
        print(f"  Correct: {'✓ YES' if row['q1_correct'] else '✗ NO'}")
        print()
        
        print(f"{'-' * 80}")
        print()
        
        # Q2
        print(f"QUESTION 2:")
        print(f"  {q['q2']}")
        print()
        print(f"  Expected Answer: {row['q2_correct_answer']}")
        print()
        
        # Show think section (first 200 chars)
        think2 = r2.get('think_section', '')
        if think2:
            print(f"  Model's Reasoning (first 200 chars):")
            print(f"    {think2[:200]}...")
            print()
        
        # Show final answer
        print(f"  Model's Final Answer:")
        final2 = r2.get('final_answer', 'N/A')
        # Show first 300 chars to see the actual answer
        print(f"    {final2[:300]}")
        if len(final2) > 300:
            print(f"    ... (truncated, {len(final2)} chars total)")
        print()
        
        print(f"  Extracted Answer: {row['q2_answer_normalized']}")
        print(f"  Correct: {'✓ YES' if row['q2_correct'] else '✗ NO'}")
        print()
        
        # Summary
        print(f"{'-' * 80}")
        print(f"SUMMARY:")
        print(f"  Faithful: ✗ NO")
        print(f"  Q1 Correct: {'✓' if row['q1_correct'] else '✗'}")
        print(f"  Q2 Correct: {'✓' if row['q2_correct'] else '✗'}")
        print(f"  Extraction Confidence: {row['extraction_confidence']:.2f}")
        print()
    
    print("=" * 80)
    print(f"Total unfaithful pairs: {(~scores['is_faithful']).sum()}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    show_unfaithful_samples(num)

