#!/usr/bin/env python3
"""
Quick Unfaithful Pairs Summary
Just the list of unfaithful pair IDs with basic info.
"""
import pandas as pd
from pathlib import Path


def quick_summary():
    """Quick summary of unfaithful pairs."""
    
    scores_path = "data/processed/faithfulness_scores.csv"
    if not Path(scores_path).exists():
        print(f"❌ File not found: {scores_path}")
        return
    
    df = pd.read_csv(scores_path)
    unfaithful = df[df['is_faithful'] == False]
    
    print(f"Faithfulness Rate: {df['is_faithful'].mean():.1%}")
    print(f"Unfaithful Pairs: {len(unfaithful)}/{len(df)}\n")
    
    if len(unfaithful) == 0:
        print("✅ No unfaithful pairs - model is 100% faithful!")
        return
    
    print("Unfaithful Pair IDs:")
    print("-" * 60)
    print(f"{'ID':<12} {'Q1 Answer':<15} {'Q2 Answer':<15} {'Correct':<10}")
    print("-" * 60)
    
    for _, row in unfaithful.iterrows():
        print(f"{row['pair_id']:<12} {str(row['q1_answer_normalized']):<15} "
              f"{str(row['q2_answer_normalized']):<15} {row['correct_answer']:<10}")
    
    print("-" * 60)
    print(f"\nTotal: {len(unfaithful)} unfaithful pairs")
    
    if len(unfaithful) >= 10:
        print("✅ Ready for Phase 3 (need ≥10)")
    else:
        print(f"⚠️  Need {10 - len(unfaithful)} more for Phase 3")


if __name__ == "__main__":
    quick_summary()


