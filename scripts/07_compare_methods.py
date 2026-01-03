#!/usr/bin/env python3
"""
Script 07: Compare Scoring Methods

Compare two different faithfulness scoring methods and analyze disagreements.

Usage:
    python scripts/07_compare_methods.py \\
        --method1-scores results/scores_correctness.csv \\
        --method1-name "Answer Correctness" \\
        --method2-scores results/scores_llm_judge.csv \\
        --method2-name "LLM Judge"
    
    # Save comparison
    python scripts/07_compare_methods.py \\
        --method1-scores results/scores_correctness.csv \\
        --method2-scores results/scores_llm_judge.csv \\
        --output results/comparison.csv

Dependencies:
    - Input: Two score CSV files from 03_score_faithfulness.py
    - Output: Comparison analysis and optional CSV
"""

import sys
import argparse
import pandas as pd
from pathlib import Path


def compare_methods(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    """Compare two scoring methods."""
    
    print("\n" + "="*70)
    print("SCORING METHOD COMPARISON")
    print("="*70)
    
    # Overall rates
    rate1 = df1['faithful'].mean() * 100
    rate2 = df2['faithful'].mean() * 100
    
    print(f"\nOverall Faithfulness Rates:")
    print(f"  {name1:20s}: {rate1:.1f}%")
    print(f"  {name2:20s}: {rate2:.1f}%")
    print(f"  Difference:         {rate2 - rate1:+.1f} percentage points")
    
    # Merge to find disagreements
    merged = df1[['pair_id', 'faithful']].merge(
        df2[['pair_id', 'faithful']], 
        on='pair_id', 
        suffixes=('_1', '_2')
    )
    
    # Classify
    agree_faithful = ((merged['faithful_1'] == True) & (merged['faithful_2'] == True))
    agree_unfaithful = ((merged['faithful_1'] == False) & (merged['faithful_2'] == False))
    method1_stricter = ((merged['faithful_1'] == False) & (merged['faithful_2'] == True))
    method2_stricter = ((merged['faithful_1'] == True) & (merged['faithful_2'] == False))
    
    print(f"\nAgreement Analysis (n={len(merged)} pairs):")
    print(f"  Both say faithful:     {agree_faithful.sum():4d} ({agree_faithful.sum()/len(merged)*100:.1f}%)")
    print(f"  Both say unfaithful:   {agree_unfaithful.sum():4d} ({agree_unfaithful.sum()/len(merged)*100:.1f}%)")
    print(f"  {name1} stricter:      {method1_stricter.sum():4d} ({method1_stricter.sum()/len(merged)*100:.1f}%)")
    print(f"  {name2} stricter:      {method2_stricter.sum():4d} ({method2_stricter.sum()/len(merged)*100:.1f}%)")
    
    agreement_rate = (agree_faithful.sum() + agree_unfaithful.sum()) / len(merged) * 100
    print(f"\n  Total agreement: {agreement_rate:.1f}%")
    
    print("\n" + "="*70 + "\n")
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Compare two faithfulness scoring methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--method1-scores', type=str, required=True,
                       help='First scoring method CSV file')
    parser.add_argument('--method1-name', type=str, default='Method 1',
                       help='Name for first method (default: Method 1)')
    parser.add_argument('--method2-scores', type=str, required=True,
                       help='Second scoring method CSV file')
    parser.add_argument('--method2-name', type=str, default='Method 2',
                       help='Name for second method (default: Method 2)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Save comparison to CSV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.method1_scores).exists():
        print(f"❌ Error: File not found: {args.method1_scores}")
        return 1
    if not Path(args.method2_scores).exists():
        print(f"❌ Error: File not found: {args.method2_scores}")
        return 1
    
    # Load scores
    print(f"Loading {args.method1_name} scores from {args.method1_scores}...")
    df1 = pd.read_csv(args.method1_scores)
    print(f"✓ Loaded {len(df1)} pairs")
    
    print(f"Loading {args.method2_name} scores from {args.method2_scores}...")
    df2 = pd.read_csv(args.method2_scores)
    print(f"✓ Loaded {len(df2)} pairs")
    
    # Compare
    merged = compare_methods(df1, df2, args.method1_name, args.method2_name)
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"✓ Saved comparison to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

