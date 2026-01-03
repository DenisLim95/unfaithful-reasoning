#!/usr/bin/env python3
"""
Compare faithfulness scores from different methods.

This script compares answer correctness vs LLM judge scores
and shows where they disagree.

Usage:
    python compare_scoring_methods.py \
      --correctness results/scores_correctness.csv \
      --llm-judge results/scores_llm_judge.csv
"""

import pandas as pd
import argparse
from pathlib import Path


def load_scores(filepath: str, method_name: str) -> pd.DataFrame:
    """Load scores and add method identifier."""
    df = pd.read_csv(filepath)
    df['method'] = method_name
    return df


def compare_methods(df_correctness: pd.DataFrame, df_llm: pd.DataFrame):
    """Compare two scoring methods and show disagreements."""
    
    print("\n" + "="*70)
    print("SCORING METHOD COMPARISON")
    print("="*70)
    
    # Overall rates
    correctness_rate = df_correctness['faithful'].mean() * 100
    llm_rate = df_llm['faithful'].mean() * 100
    
    print(f"\nOverall Faithfulness Rates:")
    print(f"  Answer Correctness: {correctness_rate:.1f}%")
    print(f"  LLM Judge:          {llm_rate:.1f}%")
    print(f"  Difference:         {llm_rate - correctness_rate:+.1f} percentage points")
    
    # Merge to find disagreements
    merged = df_correctness[['pair_id', 'faithful']].merge(
        df_llm[['pair_id', 'faithful']], 
        on='pair_id', 
        suffixes=('_correctness', '_llm')
    )
    
    # Classify disagreements
    agree_faithful = ((merged['faithful_correctness'] == True) & 
                      (merged['faithful_llm'] == True))
    agree_unfaithful = ((merged['faithful_correctness'] == False) & 
                        (merged['faithful_llm'] == False))
    disagree_llm_stricter = ((merged['faithful_correctness'] == True) & 
                             (merged['faithful_llm'] == False))
    disagree_correctness_stricter = ((merged['faithful_correctness'] == False) & 
                                     (merged['faithful_llm'] == True))
    
    print(f"\nAgreement Analysis (n={len(merged)} pairs):")
    print(f"  Both say faithful:     {agree_faithful.sum()} ({agree_faithful.sum()/len(merged)*100:.1f}%)")
    print(f"  Both say unfaithful:   {agree_unfaithful.sum()} ({agree_unfaithful.sum()/len(merged)*100:.1f}%)")
    print(f"  LLM stricter:          {disagree_llm_stricter.sum()} ({disagree_llm_stricter.sum()/len(merged)*100:.1f}%)")
    print(f"  Correctness stricter:  {disagree_correctness_stricter.sum()} ({disagree_correctness_stricter.sum()/len(merged)*100:.1f}%)")
    
    agreement_rate = (agree_faithful.sum() + agree_unfaithful.sum()) / len(merged) * 100
    print(f"\n  Total agreement: {agreement_rate:.1f}%")
    
    # Show examples of disagreements
    if disagree_llm_stricter.sum() > 0:
        print(f"\n" + "="*70)
        print("EXAMPLES: LLM Judge Stricter (Answer correct but reasoning inconsistent)")
        print("="*70)
        
        disagreement_ids = merged[disagree_llm_stricter]['pair_id'].head(5)
        for pair_id in disagreement_ids:
            llm_row = df_llm[df_llm['pair_id'] == pair_id].iloc[0]
            print(f"\nPair: {pair_id}")
            print(f"  Answer Correctness: ✓ Faithful (both answers correct)")
            print(f"  LLM Judge: ✗ Unfaithful")
            
            if 'q1_explanation' in llm_row:
                print(f"  Q1 explanation: {llm_row['q1_explanation'][:100]}...")
            if 'q2_explanation' in llm_row:
                print(f"  Q2 explanation: {llm_row['q2_explanation'][:100]}...")
    
    if disagree_correctness_stricter.sum() > 0:
        print(f"\n" + "="*70)
        print("EXAMPLES: Answer Correctness Stricter (Reasoning consistent but answer wrong)")
        print("="*70)
        
        disagreement_ids = merged[disagree_correctness_stricter]['pair_id'].head(5)
        for pair_id in disagreement_ids:
            correctness_row = df_correctness[df_correctness['pair_id'] == pair_id].iloc[0]
            print(f"\nPair: {pair_id}")
            print(f"  Answer Correctness: ✗ Unfaithful (wrong answer)")
            print(f"  LLM Judge: ✓ Faithful (reasoning consistent)")
            
            if 'q1_correct' in correctness_row:
                print(f"  Q1: {'Correct' if correctness_row['q1_correct'] else 'Wrong'}")
            if 'q2_correct' in correctness_row:
                print(f"  Q2: {'Correct' if correctness_row['q2_correct'] else 'Wrong'}")
    
    print("\n" + "="*70 + "\n")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Compare faithfulness scoring methods')
    parser.add_argument('--correctness', type=str, required=True,
                       help='CSV file with answer correctness scores')
    parser.add_argument('--llm-judge', type=str, required=True,
                       help='CSV file with LLM judge scores')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Save comparison results to CSV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.correctness).exists():
        print(f"❌ Error: File not found: {args.correctness}")
        return 1
    if not Path(args.llm_judge).exists():
        print(f"❌ Error: File not found: {args.llm_judge}")
        return 1
    
    # Load scores
    print(f"Loading answer correctness scores from {args.correctness}...")
    df_correctness = load_scores(args.correctness, 'correctness')
    print(f"✓ Loaded {len(df_correctness)} pairs")
    
    print(f"Loading LLM judge scores from {args.llm_judge}...")
    df_llm = load_scores(args.llm_judge, 'llm_judge')
    print(f"✓ Loaded {len(df_llm)} pairs")
    
    # Compare
    merged = compare_methods(df_correctness, df_llm)
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"✓ Saved comparison to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

