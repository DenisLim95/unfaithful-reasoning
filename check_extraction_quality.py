#!/usr/bin/env python3
"""
Check Extraction Quality
Automatically flags likely extraction errors vs real unfaithfulness.
"""
import pandas as pd


def check_extraction_quality():
    """Analyze extraction quality to separate real unfaithfulness from errors."""
    
    df = pd.read_csv("data/processed/faithfulness_scores.csv")
    unfaithful = df[df['is_faithful'] == False].copy()
    
    print("=" * 80)
    print("EXTRACTION QUALITY ANALYSIS")
    print("=" * 80)
    
    # Criteria for likely extraction errors
    unfaithful['likely_extraction_error'] = (
        (~unfaithful['q1_correct']) &  # Q1 wrong
        (~unfaithful['q2_correct']) &  # Q2 wrong
        (unfaithful['extraction_confidence'] <= 0.7)  # Low confidence
    )
    
    extraction_errors = unfaithful[unfaithful['likely_extraction_error']]
    likely_real = unfaithful[~unfaithful['likely_extraction_error']]
    
    print(f"\n📊 Summary:")
    print(f"  Total pairs: {len(df)}")
    print(f"  Labeled unfaithful: {len(unfaithful)} ({len(unfaithful)/len(df):.1%})")
    print(f"  Likely extraction errors: {len(extraction_errors)} ({len(extraction_errors)/len(df):.1%})")
    print(f"  Likely REAL unfaithfulness: {len(likely_real)} ({len(likely_real)/len(df):.1%})")
    
    print(f"\n🎯 Adjusted Faithfulness Rate:")
    adjusted_faithful = len(df) - len(likely_real)
    print(f"  (Excluding likely extraction errors)")
    print(f"  {adjusted_faithful}/{len(df)} = {adjusted_faithful/len(df):.1%}")
    
    print(f"\n{'='*80}")
    print("LIKELY EXTRACTION ERRORS")
    print(f"{'='*80}")
    print("(Both answers wrong + low confidence = extraction probably failed)\n")
    print(f"{'Pair ID':<12} {'Q1 Answer':<15} {'Q2 Answer':<15} {'Correct':<15} {'Conf'}")
    print("-" * 80)
    for _, row in extraction_errors.iterrows():
        print(f"{row['pair_id']:<12} {str(row['q1_answer_normalized'])[:15]:<15} "
              f"{str(row['q2_answer_normalized'])[:15]:<15} {str(row['correct_answer'])[:15]:<15} "
              f"{row['extraction_confidence']:.1f}")
    
    print(f"\n{'='*80}")
    print("LIKELY REAL UNFAITHFULNESS")
    print(f"{'='*80}")
    print("(At least one answer correct OR high confidence = extraction worked)\n")
    print(f"{'Pair ID':<12} {'Q1 Answer':<15} {'Q2 Answer':<15} {'Correct':<15} {'Q1✓':<5} {'Q2✓':<5} {'Conf'}")
    print("-" * 80)
    for _, row in likely_real.iterrows():
        q1_check = '✓' if row['q1_correct'] else '✗'
        q2_check = '✓' if row['q2_correct'] else '✗'
        print(f"{row['pair_id']:<12} {str(row['q1_answer_normalized'])[:15]:<15} "
              f"{str(row['q2_answer_normalized'])[:15]:<15} {str(row['correct_answer'])[:15]:<15} "
              f"{q1_check:<5} {q2_check:<5} {row['extraction_confidence']:.1f}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print(f"\nYour model's TRUE faithfulness rate is likely: {adjusted_faithful/len(df):.1%}")
    print(f"(vs. raw rate of {df['is_faithful'].mean():.1%})")
    
    if len(likely_real) >= 10:
        print(f"\n✅ You have {len(likely_real)} REAL unfaithful pairs")
        print("✅ Enough for Phase 3 mechanistic analysis!")
    else:
        print(f"\n⚠️  Only {len(likely_real)} REAL unfaithful pairs (need ≥10)")
        print(f"⚠️  You have {len(extraction_errors)} extraction errors inflating the count")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Review the 'LIKELY REAL UNFAITHFULNESS' list above")
    print(f"  2. For Phase 3, use only those pairs (not extraction errors)")
    print(f"  3. Consider improving extraction for power expressions (7^4, etc.)")


if __name__ == "__main__":
    check_extraction_quality()


