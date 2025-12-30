"""
Manual review helper for Phase 1 question pairs.

Displays a random sample of question pairs for human verification.
"""

import json
import random
from pathlib import Path


def sample_for_review(n: int = 10):
    """
    Sample n random pairs for manual review.
    
    Args:
        n: Number of pairs to sample (default: 10)
    """
    
    file_path = Path('data/raw/question_pairs.json')
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("   Run generate_questions.py first")
        return
    
    with open(file_path) as f:
        data = json.load(f)
        pairs = data['pairs']
    
    sample = random.sample(pairs, min(n, len(pairs)))
    
    print("=" * 60)
    print(f"MANUAL REVIEW: {len(sample)} Random Pairs")
    print("=" * 60)
    print("\nFor each pair, verify:")
    print("  1. Questions are grammatically correct")
    print("  2. Correct answer is actually correct")
    print("  3. Both questions test the same fact\n")
    
    for i, pair in enumerate(sample, 1):
        print(f"\n{'='*60}")
        print(f"Pair {i}/{len(sample)}: {pair['id']}")
        print(f"{'='*60}")
        print(f"Difficulty: {pair['difficulty']}")
        print(f"\nQ1: {pair['q1']}")
        print(f"Q2: {pair['q2']}")
        print(f"\nCorrect Answer: {pair['correct_answer']}")
        print(f"\nMetadata: {pair['metadata']}")
        
        # For numerical, show computation
        if pair['difficulty'] == 'easy' and 'values' in pair['metadata']:
            vals = pair['metadata']['values']
            if 'a' in vals and 'b' in vals:
                print(f"\nVerification: max({vals['a']}, {vals['b']}) = {max(vals['a'], vals['b'])}")
        
        elif pair['difficulty'] == 'medium' and 'products' in pair['metadata']:
            prods = pair['metadata']['products']
            print(f"\nVerification: prod1={prods['prod1']}, prod2={prods['prod2']}")
            print(f"              max={max(prods['prod1'], prods['prod2'])}")
        
        elif pair['difficulty'] == 'hard' and 'powers' in pair['metadata']:
            pows = pair['metadata']['powers']
            print(f"\nVerification: pow1={pows['pow1']}, pow2={pows['pow2']}")
            print(f"              max={max(pows['pow1'], pows['pow2'])}")
    
    print(f"\n{'='*60}")
    print("✓ Review complete? (All pairs correct)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Set seed for reproducible sampling
    random.seed(42)
    
    sample_for_review(10)

