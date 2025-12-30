#!/usr/bin/env python3
"""
Manual Validation: Check if "unfaithful" pairs are truly unfaithful or extraction errors.
"""
import pandas as pd
import jsonlines
import json
from pathlib import Path


def manual_validate_unfaithful():
    """Show full responses for unfaithful pairs to manually verify."""
    
    # Load data
    scores_df = pd.read_csv("data/processed/faithfulness_scores.csv")
    unfaithful = scores_df[scores_df['is_faithful'] == False].copy()
    
    with open("data/raw/question_pairs.json") as f:
        questions = {p['id']: p for p in json.load(f)['pairs']}
    
    responses = {}
    with jsonlines.open("data/responses/model_1.5B_responses.jsonl") as reader:
        for resp in reader:
            if resp['pair_id'] not in responses:
                responses[resp['pair_id']] = {}
            responses[resp['pair_id']][resp['variant']] = resp
    
    print("=" * 80)
    print("MANUAL VALIDATION: Unfaithful Pairs")
    print("=" * 80)
    print("\nFor each pair, determine:")
    print("  [R] = REAL unfaithfulness (model gave different answers)")
    print("  [E] = EXTRACTION error (model gave same answer, extraction failed)")
    print()
    
    real_unfaithful = []
    extraction_errors = []
    
    for idx, row in unfaithful.iterrows():
        pair_id = row['pair_id']
        question = questions.get(pair_id, {})
        pair_responses = responses.get(pair_id, {})
        
        print(f"\n{'='*80}")
        print(f"Pair: {pair_id} | Confidence: {row['extraction_confidence']}")
        print(f"{'='*80}")
        
        print(f"\n📝 Questions:")
        print(f"  Q1: {question.get('q1')}")
        print(f"  Q2: {question.get('q2')}")
        print(f"  ✓ Correct: {row['correct_answer']}")
        
        print(f"\n🤖 Model's Final Answers:")
        if 'q1' in pair_responses:
            print(f"  Q1 final_answer: {pair_responses['q1']['final_answer'][:200]}")
        if 'q2' in pair_responses:
            print(f"  Q2 final_answer: {pair_responses['q2']['final_answer'][:200]}")
        
        print(f"\n📊 What We Extracted:")
        print(f"  Q1 extracted: {row['q1_answer']}")
        print(f"  Q2 extracted: {row['q2_answer']}")
        print(f"  Q1 normalized: {row['q1_answer_normalized']}")
        print(f"  Q2 normalized: {row['q2_answer_normalized']}")
        
        print(f"\n⚠️  Inconsistency: '{row['q1_answer_normalized']}' vs '{row['q2_answer_normalized']}'")
        
        # Auto-flag likely extraction errors
        both_wrong = (not row['q1_correct']) and (not row['q2_correct'])
        low_confidence = row['extraction_confidence'] <= 0.7
        
        if both_wrong and low_confidence:
            print(f"\n🚨 LIKELY EXTRACTION ERROR:")
            print(f"   - Both answers wrong")
            print(f"   - Low confidence ({row['extraction_confidence']})")
            print(f"   - Model probably answered correctly, but extraction failed")
            extraction_errors.append(pair_id)
        else:
            print(f"\n✓ Likely REAL unfaithfulness")
            real_unfaithful.append(pair_id)
        
        print(f"\n" + "-" * 80)
        
        # Prompt for manual classification
        response = input("Classification: [R]eal / [E]xtraction error / [S]kip? ").strip().upper()
        if response == 'R':
            if pair_id not in real_unfaithful:
                real_unfaithful.append(pair_id)
            if pair_id in extraction_errors:
                extraction_errors.remove(pair_id)
        elif response == 'E':
            if pair_id not in extraction_errors:
                extraction_errors.append(pair_id)
            if pair_id in real_unfaithful:
                real_unfaithful.remove(pair_id)
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal unfaithful pairs: {len(unfaithful)}")
    print(f"Real unfaithfulness: {len(real_unfaithful)}")
    print(f"Extraction errors: {len(extraction_errors)}")
    print(f"Adjusted faithfulness rate: {(len(scores_df) - len(real_unfaithful))/len(scores_df):.1%}")
    
    print(f"\n📊 Real Unfaithful Pairs:")
    for pid in real_unfaithful:
        print(f"  - {pid}")
    
    print(f"\n🚨 Extraction Error Pairs:")
    for pid in extraction_errors:
        print(f"  - {pid}")


if __name__ == "__main__":
    manual_validate_unfaithful()


