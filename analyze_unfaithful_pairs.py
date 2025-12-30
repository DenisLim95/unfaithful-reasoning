#!/usr/bin/env python3
"""
Analyze Unfaithful Pairs
Shows which question pairs resulted in unfaithful (inconsistent) responses.
"""
import pandas as pd
import json
import jsonlines
from pathlib import Path


def load_unfaithful_pairs():
    """Load and display unfaithful pairs with details."""
    
    # Load faithfulness scores
    scores_path = "data/processed/faithfulness_scores.csv"
    if not Path(scores_path).exists():
        print(f"❌ File not found: {scores_path}")
        print("Run Phase 2 first: ./run_phase2.sh")
        return
    
    df = pd.read_csv(scores_path)
    
    # Filter unfaithful pairs
    unfaithful = df[df['is_faithful'] == False].copy()
    
    print("=" * 80)
    print("UNFAITHFUL PAIRS ANALYSIS")
    print("=" * 80)
    print(f"\nTotal pairs: {len(df)}")
    print(f"Faithful pairs: {df['is_faithful'].sum()} ({df['is_faithful'].mean():.1%})")
    print(f"Unfaithful pairs: {len(unfaithful)} ({len(unfaithful)/len(df):.1%})")
    print()
    
    if len(unfaithful) == 0:
        print("✅ No unfaithful pairs found! Model is 100% faithful.")
        print("\n⚠️  This means you don't have examples for Phase 3 mechanistic analysis.")
        print("Consider generating more question pairs or analyzing why the model is so faithful.")
        return
    
    # Load questions for context
    questions_path = "data/raw/question_pairs.json"
    with open(questions_path) as f:
        questions = {p['id']: p for p in json.load(f)['pairs']}
    
    # Load responses for full context
    responses_path = "data/responses/model_1.5B_responses.jsonl"
    responses = {}
    with jsonlines.open(responses_path) as reader:
        for resp in reader:
            pair_id = resp['pair_id']
            if pair_id not in responses:
                responses[pair_id] = {}
            responses[pair_id][resp['variant']] = resp
    
    print("-" * 80)
    print("UNFAITHFUL PAIRS DETAILS")
    print("-" * 80)
    
    for idx, row in unfaithful.iterrows():
        pair_id = row['pair_id']
        question = questions.get(pair_id, {})
        pair_responses = responses.get(pair_id, {})
        
        print(f"\n{'='*80}")
        print(f"Pair {idx + 1}/{len(unfaithful)}: {pair_id}")
        print(f"{'='*80}")
        
        # Show questions
        print(f"\n📝 Questions:")
        print(f"  Q1: {question.get('q1', 'N/A')}")
        print(f"  Q2: {question.get('q2', 'N/A')}")
        print(f"  ✓ Correct Answer: {row['correct_answer']}")
        
        # Show model answers
        print(f"\n🤖 Model Answers:")
        print(f"  Q1 answered: {row['q1_answer']}")
        print(f"  Q2 answered: {row['q2_answer']}")
        print(f"  Q1 normalized: {row['q1_answer_normalized']}")
        print(f"  Q2 normalized: {row['q2_answer_normalized']}")
        
        # Show correctness
        print(f"\n✓ Correctness:")
        print(f"  Q1 correct: {'✅' if row['q1_correct'] else '❌'}")
        print(f"  Q2 correct: {'✅' if row['q2_correct'] else '❌'}")
        print(f"  Extraction confidence: {row['extraction_confidence']:.1f}")
        
        # Show inconsistency
        print(f"\n⚠️  Inconsistency:")
        print(f"  Model gave different answers: '{row['q1_answer_normalized']}' vs '{row['q2_answer_normalized']}'")
        
        # Show full responses (first 200 chars of each)
        if pair_responses:
            print(f"\n💭 Think Sections (preview):")
            if 'q1' in pair_responses:
                think_q1 = pair_responses['q1'].get('think_section', '')[:200]
                print(f"  Q1: {think_q1}...")
            if 'q2' in pair_responses:
                think_q2 = pair_responses['q2'].get('think_section', '')[:200]
                print(f"  Q2: {think_q2}...")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nUnfaithful pairs breakdown:")
    print(f"  Both answers wrong: {((~unfaithful['q1_correct']) & (~unfaithful['q2_correct'])).sum()}")
    print(f"  Q1 correct, Q2 wrong: {(unfaithful['q1_correct'] & (~unfaithful['q2_correct'])).sum()}")
    print(f"  Q1 wrong, Q2 correct: {((~unfaithful['q1_correct']) & unfaithful['q2_correct']).sum()}")
    print(f"  Both correct but inconsistent: {(unfaithful['q1_correct'] & unfaithful['q2_correct']).sum()}")
    
    print(f"\nAverage extraction confidence: {unfaithful['extraction_confidence'].mean():.2f}")
    
    # Phase 3 readiness
    print(f"\n{'='*80}")
    print("PHASE 3 READINESS")
    print(f"{'='*80}")
    if len(unfaithful) >= 10:
        print(f"✅ You have {len(unfaithful)} unfaithful pairs (need ≥10)")
        print("✅ Ready to proceed to Phase 3: Mechanistic Analysis")
    else:
        print(f"⚠️  You have {len(unfaithful)} unfaithful pairs (need ≥10)")
        print(f"⚠️  Need {10 - len(unfaithful)} more for Phase 3")
        print("\nOptions:")
        print("  1. Generate more question pairs in Phase 1")
        print("  2. Reframe Phase 3 analysis (attention patterns work with any split)")
        print("  3. Analyze why model is so faithful (also interesting!)")


if __name__ == "__main__":
    load_unfaithful_pairs()


