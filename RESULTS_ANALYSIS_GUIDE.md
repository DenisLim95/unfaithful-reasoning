# Results Analysis Guide: How to Complete Your Write-Up

**Purpose:** Step-by-step guide for analyzing your 100-pair results and filling in the [PENDING] sections of your application documents.

---

## Quick Start

Once your 100-pair run completes:

```bash
# 1. Check that all data is generated
ls data/processed/faithfulness_scores.csv
ls results/probe_results/all_probe_results.pt

# 2. Run analysis script (to be created below)
python analyze_final_results.py

# 3. Generate visualizations
python generate_application_figures.py

# 4. Fill in write-up templates with outputs
```

---

## Step 1: Analyze Faithfulness Rates

### Script: `analyze_final_results.py`

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('data/processed/faithfulness_scores.csv')

# Basic statistics
n_pairs = len(df)
n_faithful = df['is_faithful'].sum()
n_unfaithful = (~df['is_faithful']).sum()
faithfulness_rate = n_faithful / n_pairs

print("=" * 60)
print("FAITHFULNESS STATISTICS")
print("=" * 60)
print(f"Total pairs: {n_pairs}")
print(f"Faithful: {n_faithful} ({faithfulness_rate:.1%})")
print(f"Unfaithful: {n_unfaithful} ({1-faithfulness_rate:.1%})")

# Q1 vs Q2 accuracy
q1_correct = df['q1_correct'].mean()
q2_correct = df['q2_correct'].mean()
asymmetry = q1_correct - q2_correct

print(f"\nQ1 accuracy: {q1_correct:.1%}")
print(f"Q2 accuracy: {q2_correct:.1%}")
print(f"Asymmetry: {asymmetry:+.1%}")

# Extraction confidence
high_conf = (df['extraction_confidence'] > 0.8).mean()
print(f"\nHigh-confidence extractions: {high_conf:.1%}")

# 95% Confidence intervals (Wilson score interval)
from statsmodels.stats.proportion import proportion_confint

ci_low, ci_high = proportion_confint(n_faithful, n_pairs, alpha=0.05, method='wilson')
print(f"\n95% CI for faithfulness: [{ci_low:.1%}, {ci_high:.1%}]")

# Comparison to prior work
print("\n" + "=" * 60)
print("COMPARISON TO PRIOR WORK")
print("=" * 60)
print(f"Claude 3.7:           25% faithful (Arcuschin 2025)")
print(f"DeepSeek R1 (70B):    39% faithful (Arcuschin 2025)")
print(f"This work (1.5B):     {faithfulness_rate:.1%} faithful")

# Interpretation
if faithfulness_rate > 0.50:
    print("\n→ Small model is MORE faithful than large models")
    print("  Suggests: Small models can't hide unfaithful reasoning")
elif faithfulness_rate > 0.35:
    print("\n→ Small model has SIMILAR faithfulness to large models")
    print("  Suggests: Faithfulness is training-dependent, not scale-dependent")
else:
    print("\n→ Small model is LESS faithful than large models")
    print("  Suggests: Small models struggle with consistency")

# Save summary for write-up
summary = {
    'n_pairs': n_pairs,
    'faithfulness_rate': f"{faithfulness_rate:.1%}",
    'ci_low': f"{ci_low:.1%}",
    'ci_high': f"{ci_high:.1%}",
    'q1_accuracy': f"{q1_correct:.1%}",
    'q2_accuracy': f"{q2_correct:.1%}",
    'asymmetry': f"{asymmetry:+.1%}",
    'high_confidence': f"{high_conf:.1%}",
    'n_faithful': n_faithful,
    'n_unfaithful': n_unfaithful
}

import json
with open('results/faithfulness_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ Saved summary to results/faithfulness_summary.json")
```

**Run this:**
```bash
python analyze_final_results.py
```

**What to look for:**
1. Overall faithfulness rate (for Finding 1)
2. Q1 vs Q2 asymmetry (evidence of unfaithfulness)
3. Confidence interval width (precision of estimate)
4. Comparison to 39% (DeepSeek R1 70B)

---

## Step 2: Analyze Probe Performance

### Script: Add to `analyze_final_results.py`

```python
import torch

print("\n" + "=" * 60)
print("LINEAR PROBE PERFORMANCE")
print("=" * 60)

# Load probe results
probe_results = torch.load('results/probe_results/all_probe_results.pt')

# Baselines
random_baseline = 0.50
majority_baseline = faithfulness_rate  # % faithful

# Extract performance
results_table = []
for layer_name, result in probe_results.items():
    layer_num = int(layer_name.split('_')[1])
    train_acc = result.get('train_accuracy', result['accuracy'])
    test_acc = result['accuracy']
    auc = result['auc']
    
    vs_random = (test_acc - random_baseline) * 100
    vs_majority = (test_acc - majority_baseline) * 100
    
    results_table.append({
        'layer': layer_num,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'auc': auc,
        'vs_random': vs_random,
        'vs_majority': vs_majority
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
print(f"Test accuracy: {best['test_acc']:.1%}")
print(f"Improvement over random: {best['vs_random']:+.1f}pp")
print(f"Improvement over majority: {best['vs_majority']:+.1f}pp")
print(f"{'=' * 60}")

# Interpretation
if best['test_acc'] > 0.70:
    print("\n✓ STRONG linear encoding detected")
    print("  → Faithfulness has clear linear representation")
    print("  → Real-time monitoring is feasible")
elif best['test_acc'] > 0.60:
    print("\n≈ WEAK linear encoding detected")
    print("  → Some linear signal but noisy")
    print("  → May need ensemble or multiple layers")
else:
    print("\n✗ NO strong linear encoding")
    print("  → Faithfulness is not linearly represented")
    print("  → Need non-linear methods (SAEs, attention)")

# Check if better than majority
if best['test_acc'] > majority_baseline + 0.05:
    print("  → Probe learns more than just majority class ✓")
else:
    print("  → Probe may just predict majority class ✗")

# Layer-wise pattern analysis
layer_nums = [r['layer'] for r in results_table]
accs = [r['test_acc'] for r in results_table]

early = accs[0]  # Layer 6
mid = max(accs[1:3])  # Layer 12 or 18
late = accs[3]  # Layer 24

print(f"\nLayer-wise pattern:")
print(f"  Early (L6):   {early:.1%}")
print(f"  Middle (L12-18): {mid:.1%}")
print(f"  Late (L24):   {late:.1%}")

if mid > max(early, late) + 0.05:
    print("  → Peak in middle layers (semantic reasoning)")
elif early > mid and early > late:
    print("  → Peak in early layers (surprising!)")
elif late > mid and late > early:
    print("  → Peak in late layers (post-hoc consistency)")
else:
    print("  → Flat across layers (distributed representation)")

# Save for write-up
probe_summary = {
    'best_layer': best['layer'],
    'best_accuracy': f"{best['test_acc']:.1%}",
    'best_auc': f"{best['auc']:.3f}",
    'improvement_over_random': f"{best['vs_random']:+.1f}pp",
    'improvement_over_majority': f"{best['vs_majority']:+.1f}pp",
    'results_table': results_table
}

with open('results/probe_summary.json', 'w') as f:
    json.dump(probe_summary, f, indent=2, default=float)

print("\n✓ Saved summary to results/probe_summary.json")
```

**What to look for:**
1. Best layer accuracy (for Finding 2)
2. Improvement over baselines (evidence of learning)
3. Layer-wise pattern (for Finding 3)
4. AUC values (should be >0.5 and consistent with accuracy)

---

## Step 3: Generate Visualizations

### Script: `generate_application_figures.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

sns.set_style('whitegrid')

# Load summaries
with open('results/faithfulness_summary.json') as f:
    faith_summary = json.load(f)

with open('results/probe_summary.json') as f:
    probe_summary = json.load(f)

# Figure 1: Faithfulness Rate Comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Claude 3.7\n(~200B)', 'DeepSeek R1\n(70B)', 'This Work\n(1.5B)']
faithfulness = [0.25, 0.39, float(faith_summary['faithfulness_rate'].strip('%')) / 100]
colors = ['#d62728', '#ff7f0e', '#2ca02c']

bars = ax.bar(models, faithfulness, color=colors, alpha=0.7, edgecolor='black')

# Add percentage labels
for bar, val in zip(bars, faithfulness):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Faithfulness Rate', fontsize=14)
ax.set_title('CoT Faithfulness: Small vs Large Reasoning Models', fontsize=16, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% (random)')

ax.legend()
plt.tight_layout()
plt.savefig('results/figure1_faithfulness_comparison.png', dpi=300)
print("✓ Saved Figure 1: Faithfulness comparison")

# Figure 2: Probe Performance Across Layers
fig, ax = plt.subplots(figsize=(10, 6))

results = probe_summary['results_table']
layers = [r['layer'] for r in results]
test_accs = [r['test_acc'] for r in results]
aucs = [r['auc'] for r in results]

# Plot accuracy
ax.plot(layers, test_accs, marker='o', linewidth=3, markersize=10, 
        label='Test Accuracy', color='#2ca02c')

# Baselines
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Random (50%)', alpha=0.7)
majority = float(faith_summary['faithfulness_rate'].strip('%')) / 100
ax.axhline(majority, color='orange', linestyle='--', linewidth=2, 
           label=f'Majority ({majority:.0%})', alpha=0.7)

# Highlight best layer
best_layer = probe_summary['best_layer']
best_acc = float(probe_summary['best_accuracy'].strip('%')) / 100
ax.scatter([best_layer], [best_acc], s=300, color='gold', 
           edgecolor='black', linewidth=2, zorder=5, label='Best Layer')

ax.set_xlabel('Layer Number', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Linear Probe Performance Across Layers', fontsize=16, fontweight='bold')
ax.set_xticks(layers)
ax.set_ylim(0.4, 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('results/figure2_probe_performance.png', dpi=300)
print("✓ Saved Figure 2: Probe performance")

# Figure 3: Faithful vs Unfaithful Distribution
df = pd.read_csv('data/processed/faithfulness_scores.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Q1 accuracy
q1_faithful = df[df['is_faithful']]['q1_correct'].mean()
q1_unfaithful = df[~df['is_faithful']]['q1_correct'].mean()

axes[0].bar(['Faithful', 'Unfaithful'], [q1_faithful, q1_unfaithful], 
            color=['#2ca02c', '#d62728'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Q1 Accuracy', fontsize=12)
axes[0].set_title('Q1 Correctness by Faithfulness', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 1.0)
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

for i, (val, label) in enumerate(zip([q1_faithful, q1_unfaithful], ['Faithful', 'Unfaithful'])):
    axes[0].text(i, val + 0.03, f'{val:.0%}', ha='center', fontsize=12, fontweight='bold')

# Q2 accuracy
q2_faithful = df[df['is_faithful']]['q2_correct'].mean()
q2_unfaithful = df[~df['is_faithful']]['q2_correct'].mean()

axes[1].bar(['Faithful', 'Unfaithful'], [q2_faithful, q2_unfaithful], 
            color=['#2ca02c', '#d62728'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Q2 Accuracy', fontsize=12)
axes[1].set_title('Q2 Correctness by Faithfulness', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1.0)
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

for i, (val, label) in enumerate(zip([q2_faithful, q2_unfaithful], ['Faithful', 'Unfaithful'])):
    axes[1].text(i, val + 0.03, f'{val:.0%}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figure3_accuracy_by_faithfulness.png', dpi=300)
print("✓ Saved Figure 3: Accuracy by faithfulness")

# Figure 4: Sample Examples (text-based, to be manually created)
print("\n" + "=" * 60)
print("EXTRACT SAMPLE EXAMPLES")
print("=" * 60)

# Faithful example
faithful_pairs = df[df['is_faithful']].head(1)
print("\nFAITHFUL EXAMPLE:")
for _, row in faithful_pairs.iterrows():
    print(f"Pair ID: {row['pair_id']}")
    print(f"Q1 answer: {row['q1_answer']} (expected: {row['q1_correct_answer']})")
    print(f"Q2 answer: {row['q2_answer']} (expected: {row['q2_correct_answer']})")
    print(f"Consistent: {row['is_consistent']}")
    print(f"\n→ Look up full reasoning in data/responses/model_1.5B_responses.jsonl")

# Unfaithful example
unfaithful_pairs = df[~df['is_faithful']].head(1)
print("\nUNFAITHFUL EXAMPLE:")
for _, row in unfaithful_pairs.iterrows():
    print(f"Pair ID: {row['pair_id']}")
    print(f"Q1 answer: {row['q1_answer']} (expected: {row['q1_correct_answer']})")
    print(f"Q2 answer: {row['q2_answer']} (expected: {row['q2_correct_answer']})")
    print(f"Consistent: {row['is_consistent']}")
    print(f"\n→ Look up full reasoning in data/responses/model_1.5B_responses.jsonl")

print("\n✓ All figures generated in results/")
```

**Run this:**
```bash
python generate_application_figures.py
```

**Generated figures:**
1. `figure1_faithfulness_comparison.png` - For Executive Summary Finding 1
2. `figure2_probe_performance.png` - For Executive Summary Finding 2
3. `figure3_accuracy_by_faithfulness.png` - For Discussion section
4. Sample examples printed to console - Copy into write-up

---

## Step 4: Extract Example Reasoning Traces

### Script: `extract_examples.py`

```python
import jsonlines
import pandas as pd

# Load data
df = pd.read_csv('data/processed/faithfulness_scores.csv')

# Get one faithful and one unfaithful example
faithful_id = df[df['is_faithful']].iloc[0]['pair_id']
unfaithful_id = df[~df['is_faithful']].iloc[0]['pair_id']

print("=" * 80)
print("FAITHFUL EXAMPLE")
print("=" * 80)

# Load responses
with jsonlines.open('data/responses/model_1.5B_responses.jsonl') as reader:
    for response in reader:
        if response['pair_id'] == faithful_id:
            variant = response['variant']
            print(f"\n[{faithful_id}_{variant}]")
            print(f"Question: {response['question']}")
            print(f"\nReasoning:")
            print(response.get('reasoning', response.get('think_section', 'N/A'))[:500])
            print(f"\nFinal Answer: {response['extracted_answer']}")
            print(f"Expected: {response['expected_answer']}")
            print(f"Correct: {response['is_correct']}")

print("\n" + "=" * 80)
print("UNFAITHFUL EXAMPLE")
print("=" * 80)

with jsonlines.open('data/responses/model_1.5B_responses.jsonl') as reader:
    for response in reader:
        if response['pair_id'] == unfaithful_id:
            variant = response['variant']
            print(f"\n[{unfaithful_id}_{variant}]")
            print(f"Question: {response['question']}")
            print(f"\nReasoning:")
            print(response.get('reasoning', response.get('think_section', 'N/A'))[:500])
            print(f"\nFinal Answer: {response['extracted_answer']}")
            print(f"Expected: {response['expected_answer']}")
            print(f"Correct: {response['is_correct']}")

print("\n✓ Copy these examples into APPLICATION_EXECUTIVE_SUMMARY.md Appendix C")
```

---

## Step 5: Fill in Write-Up Templates

### Checklist for Executive Summary

Open `APPLICATION_EXECUTIVE_SUMMARY.md` and search for `[PENDING]` or `[TO FILL]`:

**Finding 1: Faithfulness Rates**
- [ ] Overall faithfulness percentage → From `faithfulness_summary.json`
- [ ] 95% CI → From `faithfulness_summary.json`
- [ ] Q1 vs Q2 accuracy → From `faithfulness_summary.json`
- [ ] Interpretation paragraph → Based on whether >50%, 35-50%, or <35%

**Finding 2: Linear Probe Detection**
- [ ] Best layer → From `probe_summary.json`
- [ ] Test accuracy → From `probe_summary.json`
- [ ] AUC-ROC → From `probe_summary.json`
- [ ] Improvement over random → From `probe_summary.json`
- [ ] Improvement over majority → From `probe_summary.json`
- [ ] Interpretation paragraph → Based on whether >70%, 60-70%, or <60%

**Finding 3: Layer-wise Analysis**
- [ ] Probe performance table → From `probe_summary.json['results_table']`
- [ ] Pattern description → Early/middle/late comparison
- [ ] Interpretation → Where faithfulness is computed

**Appendix A: Results Summary**
- [ ] Table A: Faithfulness statistics → From `faithfulness_summary.json`
- [ ] Table B: Probe performance → From `probe_summary.json['results_table']`
- [ ] Section C: Sample examples → From `extract_examples.py` output
- [ ] Section D: Figures → Reference the 3 generated PNG files

**Conclusion**
- [ ] Summary of findings (2-3 paragraphs)
- [ ] Why this matters for AI safety
- [ ] Connection to MATS research goals

### Checklist for Full Write-Up

Open `APPLICATION_FULL_WRITEUP.md` and search for `[PENDING]` or `[TO FILL]`:

**Section 6: Results**
- [ ] 6.1: Fill faithfulness rates table
- [ ] 6.2: Fill comparison table and interpretation
- [ ] 6.3: Fill probe performance table
- [ ] 6.4: Reference generated figures

**Section 7: Analysis**
- [ ] 7.1: Does linear encoding exist? → Based on probe accuracy
- [ ] 7.2: Layer-wise analysis → Based on best layer and pattern
- [ ] 7.3: Small vs large model → Based on faithfulness rate comparison

**Section 11: Conclusion**
- [ ] Summary of findings
- [ ] Contribution to field
- [ ] Personal reflection
- [ ] Why MATS

---

## Step 6: Final Checks

### Quality Checklist

**Executive Summary:**
- [ ] All [PENDING] filled in
- [ ] Figures referenced correctly
- [ ] Conclusion is clear and compelling
- [ ] Fits in ~5-7 pages
- [ ] No typos or formatting issues

**Full Write-Up:**
- [ ] All [PENDING] filled in
- [ ] Methods clearly explained
- [ ] Results match generated figures
- [ ] Discussion addresses implications
- [ ] Limitations are honest
- [ ] Future work is concrete
- [ ] ~20-25 pages total

**Figures:**
- [ ] Figure 1: Faithfulness comparison (publication quality)
- [ ] Figure 2: Probe performance (clear best layer)
- [ ] Figure 3: Accuracy by faithfulness (shows pattern)
- [ ] All figures have clear titles and labels
- [ ] All figures referenced in text

**Data Integrity:**
- [ ] All numbers in write-up match generated summaries
- [ ] No inconsistencies between tables
- [ ] Confidence intervals match point estimates
- [ ] Sample sizes add up correctly

### Final Export

```bash
# Create submission folder
mkdir mats_application_submission
cp APPLICATION_EXECUTIVE_SUMMARY.md mats_application_submission/
cp APPLICATION_FULL_WRITEUP.md mats_application_submission/
cp results/figure*.png mats_application_submission/
cp results/*_summary.json mats_application_submission/

# Optional: Include key code files
cp src/mechanistic/train_probes.py mats_application_submission/
cp src/mechanistic/cache_activations.py mats_application_submission/

# Create ZIP
zip -r mats_application_submission.zip mats_application_submission/

echo "✓ Application package ready: mats_application_submission.zip"
```

---

## Timeline Estimate

Once 100-pair run completes:

- **Hour 0.0-0.5:** Run analysis scripts
- **Hour 0.5-1.0:** Generate visualizations
- **Hour 1.0-1.5:** Extract examples and review
- **Hour 1.5-2.0:** Fill in executive summary
- **Hour 2.0-2.5:** Fill in full write-up
- **Hour 2.5-3.0:** Final checks and proofreading
- **Hour 3.0:** Submit!

**Total:** 3 hours for write-up completion (within 20+2 hour limit)

---

## Questions to Answer in Write-Up

Use your results to answer these key questions:

### For AI Safety

1. **Can we monitor faithfulness in real-time?**
   - If probe accuracy >70%: YES
   - If probe accuracy 60-70%: MAYBE (needs improvement)
   - If probe accuracy <60%: NO (need other methods)

2. **Are small models more faithful?**
   - Compare your % to 39% (DeepSeek R1 70B)
   - Explain implications for deployment

3. **Where is faithfulness computed?**
   - Best layer reveals processing stage
   - Informs where to intervene

### For Mechanistic Interpretability

1. **Are meta-cognitive properties linearly encoded?**
   - Extends representation engineering to second-order properties
   - If yes, suggests general principle
   - If no, faithfulness is special

2. **How does this inform our understanding of reasoning?**
   - Layer-wise pattern reveals processing stages
   - Comparison to truthfulness, refusal, etc.

3. **What are the practical applications?**
   - Real-time monitoring
   - Activation editing
   - Training improvements

---

## Common Issues and Troubleshooting

### Issue: Very high faithfulness (>80%)

**Possible causes:**
- Questions too easy
- Model very consistent
- Extraction errors (false positives)

**What to do:**
- Check manual examples
- Report as interesting finding
- Discuss why small models might be more faithful

### Issue: Very low probe accuracy (<55%)

**Possible causes:**
- No linear encoding (null result)
- Insufficient data
- Bug in probe training

**What to do:**
- Check if better than majority baseline
- Run on more data if possible
- Frame as null result (still publishable)

### Issue: AUC < 0.5

**This is a bug!** AUC should always be ≥ 0.5.

**Possible causes:**
- Labels flipped in probe training
- Incorrect AUC calculation

**What to do:**
- Check probe training code
- Verify label encoding (1=faithful, 0=unfaithful)

### Issue: Extraction confidence still low (<50% high-confidence)

**What to do:**
- Report honestly in limitations
- Do sensitivity analysis (results with/without low-confidence)
- Manual validation on subset

---

## Contact for Help

If you get stuck:

1. Check CURRENT_RESULTS_ANALYSIS.md for guidance
2. Review PHASE3_README.md for interpretation
3. Look at example papers (Arcuschin 2025, Zou 2023)

Good luck! 🚀


