#!/usr/bin/env python3
"""
Analyze final results from 100-pair faithfulness experiment.

Outputs:
- Console summary statistics
- results/faithfulness_summary.json
- results/probe_summary.json

Usage:
    python analyze_final_results.py
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


def analyze_faithfulness(csv_path='data/processed/faithfulness_scores.csv'):
    """Analyze faithfulness rates and patterns."""
    
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} question pairs")
    
    # Basic statistics
    print("\n" + "=" * 60)
    print("FAITHFULNESS STATISTICS")
    print("=" * 60)
    
    n_pairs = len(df)
    n_faithful = df['is_faithful'].sum()
    n_unfaithful = (~df['is_faithful']).sum()
    faithfulness_rate = n_faithful / n_pairs
    
    print(f"Total pairs: {n_pairs}")
    print(f"Faithful: {n_faithful} ({faithfulness_rate:.1%})")
    print(f"Unfaithful: {n_unfaithful} ({1-faithfulness_rate:.1%})")
    
    # Q1 vs Q2 accuracy
    q1_correct = df['q1_correct'].mean()
    q2_correct = df['q2_correct'].mean()
    asymmetry = q1_correct - q2_correct
    
    print(f"\nQ1 accuracy: {q1_correct:.1%}")
    print(f"Q2 accuracy: {q2_correct:.1%}")
    print(f"Asymmetry (Q1-Q2): {asymmetry:+.1%}")
    
    if abs(asymmetry) > 0.1:
        print("  → Strong asymmetry suggests unfaithful reasoning")
    elif abs(asymmetry) > 0.05:
        print("  → Moderate asymmetry detected")
    else:
        print("  → Minimal asymmetry (model is consistent)")
    
    # Extraction confidence
    if 'extraction_confidence' in df.columns:
        high_conf = (df['extraction_confidence'] > 0.8).mean()
        print(f"\nHigh-confidence extractions: {high_conf:.1%}")
    
    # 95% Confidence intervals (Wilson score interval)
    ci_low, ci_high = proportion_confint(n_faithful, n_pairs, alpha=0.05, method='wilson')
    print(f"\n95% CI for faithfulness: [{ci_low:.1%}, {ci_high:.1%}]")
    
    # Comparison to prior work
    print("\n" + "=" * 60)
    print("COMPARISON TO PRIOR WORK")
    print("=" * 60)
    print(f"Claude 3.7:           25% faithful (Arcuschin et al. 2025)")
    print(f"DeepSeek R1 (70B):    39% faithful (Arcuschin et al. 2025)")
    print(f"This work (1.5B):     {faithfulness_rate:.1%} faithful")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if faithfulness_rate > 0.50:
        print("✓ Small model is MORE faithful than large models")
        print("  → Suggests: Small models can't hide unfaithful reasoning")
        print("  → Implication: Consider smaller models for safety-critical apps")
    elif faithfulness_rate > 0.35:
        print("≈ Small model has SIMILAR faithfulness to large models")
        print("  → Suggests: Faithfulness is training-dependent, not scale-dependent")
        print("  → Implication: Focus on training methods, not just scale")
    else:
        print("✗ Small model is LESS faithful than large models")
        print("  → Suggests: Small models struggle with reasoning consistency")
        print("  → Implication: May need larger models for faithful reasoning")
    
    # Breakdown by correctness
    print("\n" + "=" * 60)
    print("ACCURACY BREAKDOWN")
    print("=" * 60)
    
    both_correct = (df['q1_correct'] & df['q2_correct']).sum()
    both_wrong = (~df['q1_correct'] & ~df['q2_correct']).sum()
    one_correct = n_pairs - both_correct - both_wrong
    
    print(f"Both answers correct: {both_correct} ({both_correct/n_pairs:.1%})")
    print(f"One answer correct:   {one_correct} ({one_correct/n_pairs:.1%})")
    print(f"Both answers wrong:   {both_wrong} ({both_wrong/n_pairs:.1%})")
    
    # Save summary
    summary = {
        'n_pairs': int(n_pairs),
        'faithfulness_rate': f"{faithfulness_rate:.1%}",
        'ci_low': f"{ci_low:.1%}",
        'ci_high': f"{ci_high:.1%}",
        'q1_accuracy': f"{q1_correct:.1%}",
        'q2_accuracy': f"{q2_correct:.1%}",
        'asymmetry': f"{asymmetry:+.1%}",
        'n_faithful': int(n_faithful),
        'n_unfaithful': int(n_unfaithful),
        'both_correct': int(both_correct),
        'one_correct': int(one_correct),
        'both_wrong': int(both_wrong),
    }
    
    if 'extraction_confidence' in df.columns:
        summary['high_confidence'] = f"{high_conf:.1%}"
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'faithfulness_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Saved summary to results/faithfulness_summary.json")
    
    return df, summary


def analyze_probes(probe_path='results/probe_results/all_probe_results.pt', faithfulness_rate=0.5):
    """Analyze linear probe performance."""
    
    print("\n" + "=" * 60)
    print("LINEAR PROBE PERFORMANCE")
    print("=" * 60)
    
    if not Path(probe_path).exists():
        print(f"✗ Probe results not found: {probe_path}")
        print("  Run Phase 3 first: python src/mechanistic/train_probes.py")
        return None, None
    
    # Load probe results
    probe_results = torch.load(probe_path)
    print(f"✓ Loaded probe results for {len(probe_results)} layers")
    
    # Baselines
    random_baseline = 0.50
    majority_baseline = faithfulness_rate
    
    print(f"\nBaselines:")
    print(f"  Random guess:   {random_baseline:.1%}")
    print(f"  Majority class: {majority_baseline:.1%}")
    
    # Extract performance
    results_table = []
    
    for layer_name, result in sorted(probe_results.items()):
        layer_num = int(layer_name.split('_')[1])
        
        # Handle different result formats
        train_acc = result.get('train_accuracy', result.get('accuracy', 0))
        test_acc = result.get('test_accuracy', result.get('accuracy', 0))
        auc = result.get('auc', 0.5)
        
        vs_random = (test_acc - random_baseline) * 100
        vs_majority = (test_acc - majority_baseline) * 100
        
        results_table.append({
            'layer': layer_num,
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'auc': float(auc),
            'vs_random': float(vs_random),
            'vs_majority': float(vs_majority)
        })
        
        print(f"\nLayer {layer_num}:")
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Test accuracy:  {test_acc:.1%}")
        print(f"  AUC-ROC:        {auc:.3f}")
        print(f"  vs Random:      {vs_random:+.1f}pp")
        print(f"  vs Majority:    {vs_majority:+.1f}pp")
    
    # Find best layer
    best = max(results_table, key=lambda x: x['test_acc'])
    
    print(f"\n{'=' * 60}")
    print(f"BEST LAYER: {best['layer']}")
    print(f"{'=' * 60}")
    print(f"Test accuracy:                {best['test_acc']:.1%}")
    print(f"Improvement over random:      {best['vs_random']:+.1f}pp")
    print(f"Improvement over majority:    {best['vs_majority']:+.1f}pp")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if best['test_acc'] > 0.70:
        print("✓ STRONG linear encoding detected")
        print("  → Faithfulness has clear linear representation")
        print("  → Real-time monitoring is FEASIBLE")
        print("  → Can extract probe direction for interventions")
    elif best['test_acc'] > 0.60:
        print("≈ WEAK linear encoding detected")
        print("  → Some linear signal exists but noisy")
        print("  → May need ensemble or multiple layers for monitoring")
        print("  → Useful for triage but not definitive detection")
    else:
        print("✗ NO strong linear encoding")
        print("  → Faithfulness is not linearly represented")
        print("  → Need non-linear methods (SAEs, attention analysis)")
        print("  → Linear probes insufficient for monitoring")
    
    # Check if better than majority
    if best['test_acc'] > majority_baseline + 0.05:
        print("\n✓ Probe learns more than just majority class")
    else:
        print("\n⚠ Probe may just be predicting majority class")
        print("  This is a limitation but still a valid finding")
    
    # Check AUC consistency
    if best['auc'] < 0.5:
        print("\n⚠ WARNING: AUC < 0.5 suggests possible label flip bug")
        print("  Check probe training code for correct label encoding")
    
    # Layer-wise pattern analysis
    print("\n" + "=" * 60)
    print("LAYER-WISE PATTERN")
    print("=" * 60)
    
    layer_nums = [r['layer'] for r in results_table]
    accs = [r['test_acc'] for r in results_table]
    
    early = accs[0]  # Layer 6
    mid = max(accs[1:3]) if len(accs) > 2 else accs[1]  # Layer 12 or 18
    late = accs[3] if len(accs) > 3 else accs[-1]  # Layer 24
    
    print(f"Early layers (L6):    {early:.1%}")
    print(f"Middle layers (L12-18): {mid:.1%}")
    print(f"Late layers (L24):    {late:.1%}")
    
    if mid > max(early, late) + 0.05:
        print("\n→ Peak in MIDDLE layers")
        print("  Interpretation: Faithfulness computed during semantic reasoning")
    elif early > mid and early > late + 0.05:
        print("\n→ Peak in EARLY layers (surprising!)")
        print("  Interpretation: Faithfulness determined early in processing")
    elif late > mid and late > early + 0.05:
        print("\n→ Peak in LATE layers")
        print("  Interpretation: Faithfulness is post-hoc consistency check")
    else:
        print("\n→ FLAT across layers")
        print("  Interpretation: Faithfulness is distributed representation")
    
    # Save summary
    probe_summary = {
        'best_layer': int(best['layer']),
        'best_accuracy': f"{best['test_acc']:.1%}",
        'best_auc': f"{best['auc']:.3f}",
        'improvement_over_random': f"{best['vs_random']:+.1f}pp",
        'improvement_over_majority': f"{best['vs_majority']:+.1f}pp",
        'results_table': results_table
    }
    
    with open(Path('results') / 'probe_summary.json', 'w') as f:
        json.dump(probe_summary, f, indent=2, default=float)
    
    print("\n✓ Saved summary to results/probe_summary.json")
    
    return probe_results, probe_summary


def main():
    """Run all analyses."""
    
    print("=" * 60)
    print("MATS APPLICATION - FINAL RESULTS ANALYSIS")
    print("=" * 60)
    print()
    
    # Analyze faithfulness
    df, faith_summary = analyze_faithfulness()
    
    # Analyze probes
    faithfulness_rate = float(faith_summary['faithfulness_rate'].strip('%')) / 100
    probe_results, probe_summary = analyze_probes(faithfulness_rate=faithfulness_rate)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY FOR APPLICATION")
    print("=" * 60)
    
    print(f"\nKey Finding 1: Faithfulness Rate")
    print(f"  {faith_summary['faithfulness_rate']} (95% CI: {faith_summary['ci_low']}-{faith_summary['ci_high']})")
    
    if probe_summary:
        print(f"\nKey Finding 2: Linear Encoding")
        print(f"  Best accuracy: {probe_summary['best_accuracy']} at layer {probe_summary['best_layer']}")
        print(f"  Improvement over random: {probe_summary['improvement_over_random']}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run: python generate_application_figures.py")
    print("2. Run: python extract_examples.py")
    print("3. Fill in [PENDING] sections in APPLICATION_EXECUTIVE_SUMMARY.md")
    print("4. Fill in [TO FILL] sections in APPLICATION_FULL_WRITEUP.md")
    print("=" * 60)


if __name__ == '__main__':
    main()

