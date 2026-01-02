#!/usr/bin/env python3
"""
Comprehensive verification of test data quality.
Checks questions, answers, extraction logic, and faithfulness scoring.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter


def verify_questions():
    """Verify question pairs are correctly formatted."""
    print("="*60)
    print("1. VERIFYING QUESTION PAIRS")
    print("="*60)
    
    questions_file = Path("data/raw/test_question_pairs.json")
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    pairs = data['pairs']
    print(f"\n✓ Loaded {len(pairs)} question pairs")
    
    # Check format
    issues = []
    for i, pair in enumerate(pairs):
        pair_id = pair.get('id', f'pair_{i}')
        
        # Check required fields
        if 'q1' not in pair or 'q2' not in pair:
            issues.append(f"{pair_id}: Missing q1 or q2")
        if 'q1_answer' not in pair or 'q2_answer' not in pair:
            issues.append(f"{pair_id}: Missing q1_answer or q2_answer")
        
        # Check answers are Yes/No
        if pair.get('q1_answer') not in ['Yes', 'No']:
            issues.append(f"{pair_id}: q1_answer is '{pair.get('q1_answer')}' (not Yes/No)")
        if pair.get('q2_answer') not in ['Yes', 'No']:
            issues.append(f"{pair_id}: q2_answer is '{pair.get('q2_answer')}' (not Yes/No)")
        
        # Check q1_answer and q2_answer are opposite (for comparison questions)
        if pair.get('q1_answer') == pair.get('q2_answer'):
            issues.append(f"{pair_id}: q1 and q2 have SAME answer (should be opposite)")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\n✓ All question pairs correctly formatted")
    
    # Show 5 examples
    print(f"\n{'-'*60}")
    print("SAMPLE QUESTION PAIRS (First 5)")
    print(f"{'-'*60}")
    for i in range(min(5, len(pairs))):
        pair = pairs[i]
        print(f"\nPair {i+1} (ID: {pair['id']}):")
        print(f"  Q1: {pair['q1']}")
        print(f"  Q1 Answer: {pair['q1_answer']}")
        print(f"  Q2: {pair['q2']}")
        print(f"  Q2 Answer: {pair['q2_answer']}")
        
        # Manual verification
        import re
        nums = re.findall(r'\d+', pair['q1'])
        if len(nums) >= 2:
            a, b = int(nums[0]), int(nums[1])
            expected_q1 = "Yes" if a > b else "No"
            expected_q2 = "Yes" if b > a else "No"
            
            q1_correct = expected_q1 == pair['q1_answer']
            q2_correct = expected_q2 == pair['q2_answer']
            
            print(f"  Verification: {a} vs {b}")
            print(f"    Q1 answer: {pair['q1_answer']} {'✓' if q1_correct else '✗ WRONG (should be ' + expected_q1 + ')'}")
            print(f"    Q2 answer: {pair['q2_answer']} {'✓' if q2_correct else '✗ WRONG (should be ' + expected_q2 + ')'}")


def verify_responses_and_extraction():
    """Verify model responses and answer extraction."""
    print("\n" + "="*60)
    print("2. VERIFYING RESPONSES & EXTRACTION")
    print("="*60)
    
    responses_file = Path("data/responses/test_responses.jsonl")
    
    responses = []
    with open(responses_file, 'r') as f:
        for line in f:
            responses.append(json.loads(line))
    
    print(f"\n✓ Loaded {len(responses)} responses")
    
    # Count extraction results
    extracted_counts = Counter(r['extracted_answer'] for r in responses)
    expected_counts = Counter(r['expected_answer'] for r in responses)
    
    print(f"\nExpected Answers:")
    print(f"  Yes: {expected_counts['Yes']}")
    print(f"  No: {expected_counts['No']}")
    
    print(f"\nExtracted Answers:")
    print(f"  Yes: {extracted_counts['Yes']}")
    print(f"  No: {extracted_counts['No']}")
    print(f"  Unknown: {extracted_counts['Unknown']}")
    
    # Match rate
    matches = sum(1 for r in responses if r['extracted_answer'] == r['expected_answer'])
    match_rate = 100 * matches / len(responses)
    print(f"\nExtraction Match Rate: {match_rate:.1f}% ({matches}/{len(responses)})")
    
    # Show examples by category
    print(f"\n{'-'*60}")
    print("SAMPLE RESPONSES BY CATEGORY")
    print(f"{'-'*60}")
    
    # Category 1: Correct extraction
    correct = [r for r in responses if r['extracted_answer'] == r['expected_answer']]
    print(f"\n✓ CORRECT EXTRACTION (showing 3 of {len(correct)}):")
    for r in correct[:3]:
        print(f"\n  Question: {r['question']}")
        print(f"  Expected: {r['expected_answer']}")
        print(f"  Extracted: {r['extracted_answer']}")
        
        # Show end of response
        response_end = r['response'][-250:].strip()
        print(f"  Response END (last 250 chars):")
        print(f"  └─> ...{response_end}")
        
        # Check for format markers
        has_think = '</think>' in r['response'].lower()
        has_final = 'final answer:' in r['response'].lower()
        print(f"  Format: {'✓' if has_think else '✗'} <think> tags | {'✓' if has_final else '✗'} Final Answer:")
    
    # Category 2: Incorrect extraction
    incorrect = [r for r in responses if r['extracted_answer'] != r['expected_answer'] and r['extracted_answer'] != 'Unknown']
    print(f"\n✗ INCORRECT EXTRACTION (showing 5 of {len(incorrect)}):")
    for r in incorrect[:5]:
        print(f"\n  Question: {r['question']}")
        print(f"  Expected: {r['expected_answer']}")
        print(f"  Extracted: {r['extracted_answer']} ← WRONG")
        
        # Show end of response
        response_end = r['response'][-250:].strip()
        print(f"  Response END (last 250 chars):")
        print(f"  └─> ...{response_end}")
        
        # Check for format markers
        has_think = '</think>' in r['response'].lower()
        has_final = 'final answer:' in r['response'].lower()
        print(f"  Format: {'✓' if has_think else '✗'} <think> tags | {'✓' if has_final else '✗'} Final Answer:")
    
    # Category 3: Unknown extraction
    unknown = [r for r in responses if r['extracted_answer'] == 'Unknown']
    print(f"\n? UNKNOWN EXTRACTION (showing 5 of {len(unknown)}):")
    for r in unknown[:5]:
        print(f"\n  Question: {r['question']}")
        print(f"  Expected: {r['expected_answer']}")
        print(f"  Extracted: Unknown ← FAILED TO EXTRACT")
        
        # Show end of response
        response_end = r['response'][-250:].strip()
        print(f"  Response END (last 250 chars):")
        print(f"  └─> ...{response_end}")
        
        # Check for format markers
        has_think = '</think>' in r['response'].lower()
        has_final = 'final answer:' in r['response'].lower()
        print(f"  Format: {'✓' if has_think else '✗'} <think> tags | {'✓' if has_final else '✗'} Final Answer:")


def verify_faithfulness_scoring():
    """Verify faithfulness scoring logic."""
    print("\n" + "="*60)
    print("3. VERIFYING FAITHFULNESS SCORING")
    print("="*60)
    
    scores_file = Path("data/processed/test_faithfulness_scores.csv")
    responses_file = Path("data/responses/test_responses.jsonl")
    
    df = pd.read_csv(scores_file)
    print(f"\n✓ Loaded {len(df)} scored pairs")
    
    faithful_count = df['faithful'].sum()
    unfaithful_count = len(df) - faithful_count
    
    print(f"\nFaithfulness Distribution:")
    print(f"  Faithful: {faithful_count} ({100*faithful_count/len(df):.1f}%)")
    print(f"  Unfaithful: {unfaithful_count} ({100*unfaithful_count/len(df):.1f}%)")
    
    # Load responses for detailed checking
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
    
    # Show faithful examples
    faithful_pairs = df[df['faithful']].head(5)
    print(f"\n{'-'*60}")
    print(f"FAITHFUL EXAMPLES (showing 5 of {faithful_count})")
    print(f"{'-'*60}")
    
    for _, row in faithful_pairs.iterrows():
        pair_id = row['pair_id']
        pair_responses = pairs_dict.get(pair_id, [])
        if len(pair_responses) >= 2:
            r1, r2 = pair_responses[0], pair_responses[1]
            print(f"\nPair: {pair_id}")
            print(f"  Q1: {r1['question']}")
            print(f"    Expected: {r1['expected_answer']}, Extracted: {r1['extracted_answer']} {'✓' if r1['extracted_answer'] == r1['expected_answer'] else '✗'}")
            print(f"  Q2: {r2['question']}")
            print(f"    Expected: {r2['expected_answer']}, Extracted: {r2['extracted_answer']} {'✓' if r2['extracted_answer'] == r2['expected_answer'] else '✗'}")
            print(f"  → Faithful: Both correct")
    
    # Show unfaithful examples
    unfaithful_pairs = df[~df['faithful']].head(10)
    print(f"\n{'-'*60}")
    print(f"UNFAITHFUL EXAMPLES (showing 10 of {unfaithful_count})")
    print(f"{'-'*60}")
    
    for _, row in unfaithful_pairs.iterrows():
        pair_id = row['pair_id']
        pair_responses = pairs_dict.get(pair_id, [])
        if len(pair_responses) >= 2:
            r1, r2 = pair_responses[0], pair_responses[1]
            print(f"\nPair: {pair_id}")
            print(f"  Q1: {r1['question']}")
            print(f"    Expected: {r1['expected_answer']}, Extracted: {r1['extracted_answer']} {'✓' if r1['extracted_answer'] == r1['expected_answer'] else '✗'}")
            print(f"    Response END: ...{r1['response'][-200:]}")
            print(f"  Q2: {r2['question']}")
            print(f"    Expected: {r2['expected_answer']}, Extracted: {r2['extracted_answer']} {'✓' if r2['extracted_answer'] == r2['expected_answer'] else '✗'}")
            print(f"    Response END: ...{r2['response'][-200:]}")
            
            # Explain why unfaithful
            q1_correct = r1['extracted_answer'] == r1['expected_answer']
            q2_correct = r2['extracted_answer'] == r2['expected_answer']
            if r1['extracted_answer'] == 'Unknown' or r2['extracted_answer'] == 'Unknown':
                print(f"  → Unfaithful: Failed to extract answer(s)")
            elif not q1_correct and not q2_correct:
                print(f"  → Unfaithful: Both answers wrong")
            elif not q1_correct:
                print(f"  → Unfaithful: Q1 wrong")
            elif not q2_correct:
                print(f"  → Unfaithful: Q2 wrong")


if __name__ == "__main__":
    verify_questions()
    verify_responses_and_extraction()
    verify_faithfulness_scoring()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print("\nReview the output above to check:")
    print("  1. Question pairs have correct expected answers")
    print("  2. Answer extraction is working correctly")
    print("  3. Faithfulness scoring logic is sound")
    print("  4. Sample unfaithful responses make sense")

