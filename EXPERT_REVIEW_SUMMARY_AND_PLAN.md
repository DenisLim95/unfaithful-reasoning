# Expert Review: Key Findings and Implementation Plan

## Executive Summary

An expert senior researcher conducted a thorough review of the CoT faithfulness project. While the work is a valuable exploratory study with a solid framework, the reviewer identified critical methodological issues that limit the validity of conclusions. The key finding: **current results do not strongly support the claim that CoT faithfulness is linearly encoded** - the ~66.7% probe accuracy is essentially at baseline (majority class).

---

## Critical Issues Identified

### 1. **Fatal Flaw: Label-Activation Mismatch** 
**Severity: CRITICAL**

- **Problem**: Labels (faithful/unfaithful) were generated using stochastic sampling (temp=0.6), but activations were captured deterministically
- **Impact**: The same input could produce different labels across runs, but we're using fixed activations - this fundamentally undermines the probe
- **Quote**: "This is a critical flaw because it directly lowers the achievable probe accuracy and clouds interpretation"

### 2. **Inadequate Sample Size**
**Severity: CRITICAL**

- **Problem**: Only 50 question pairs, 44 for training, ~9 for testing
- **Impact**: Confidence intervals are ±30-40 percentage points - essentially meaningless
- **Quote**: "A single classification error changes accuracy by 11% on a 9-sample test set"

### 3. **Misleading Probe Performance**
**Severity: HIGH**

- **Problem**: 66.7% accuracy is compared to 50% random baseline, but should be compared to ~68% majority-class baseline
- **Impact**: The probe performed no better than always predicting "faithful" - it learned nothing
- **Evidence**: 66.7% = 6/9 correct, uniform across all layers suggests trivial solution

### 4. **Activation Capture Timing**
**Severity: HIGH**

- **Problem**: Activations captured BEFORE generation, not during reasoning
- **Impact**: Missing the actual reasoning process where faithfulness manifests
- **Quote**: "The probe wasn't looking at the model while it was reasoning, only before it started"

### 5. **Mean-Pooling Loss of Information**
**Severity: MEDIUM**

- **Problem**: Averaging across all tokens loses positional/token-specific signals
- **Impact**: Critical faithfulness signals (e.g., at comparison tokens) get diluted

### 6. **Conflated Faithfulness Definition**
**Severity: MEDIUM**

- **Problem**: Faithfulness = correctness + consistency (errors make pairs "unfaithful")
- **Impact**: Can't distinguish logical inconsistency from simple mistakes

---

## What The Reviewer Affirmed

### Strengths of the Work

1. **High faithfulness rate (72%) is well-supported** - this finding is solid
2. **Question-flipping methodology is sound** - creative and appropriate approach
3. **Transparent about limitations** - excellent scientific practice
4. **Good experimental infrastructure** - solid foundation for future work
5. **Appropriate research framing** - correctly labeled as exploratory/preliminary

---

## Key Statistical Concerns

### The Probe Results Are Not Reliable

1. **AUC values below 0.5** suggest classifier worse than random
2. **Uniform 66.7% across layers** is "suspicious" - indicates trivial solution
3. **No cross-validation** - single train-test split with high variance
4. **Overfitting risk** - 50 epochs on 35 samples with no early stopping
5. **No validation set** - can't assess generalization during training

### The "Late-Layer Emergence" Claim Is Questionable

- Separation metric (0.862 at layer 24) contradicts poor test performance
- Likely reflects training data overfitting, not real signal
- The reviewer is skeptical of this visualization metric given the contradictory results

---

## Interpretation Corrections

### What We Can Actually Conclude

| Claim in Report | Reviewer's Assessment |
|----------------|----------------------|
| "72% faithfulness in small model" | ✅ **Supported** - This is factual for this task |
| "Higher than large models" | ⚠️ **Speculative** - Different tasks, can't compare directly |
| "Modest linear signal exists" | ❌ **Not supported** - 66.7% is baseline, not signal |
| "Late-layer emergence" | ⚠️ **Suggestive but unproven** - Visualization unreliable |
| "Linear probes feasible for monitoring" | ❌ **No** - 66.7% accuracy is unreliable for practice |

### The Core Answer

> **"Is CoT faithfulness linearly encoded in small models?"**
> 
> The evidence leans toward **NO** - or at best, any linear encoding is extremely weak and not practically useful. If faithfulness is encoded, it's likely in a non-linear, complex manner.

---

## Implementation Plan

### Phase 1: Fix Critical Methodological Flaws (Immediate Priority)

#### Task 1.1: Eliminate Label-Activation Mismatch
**Goal**: Align labels with activations perfectly

**Options**:
- **Option A** (Recommended): Use deterministic generation (temperature=0) for labeling
  - Ensures same input → same output → same label
  - Captures activations from same deterministic run
  
- **Option B**: Capture activations during actual stochastic generation
  - More complex implementation
  - Requires hooking during autoregressive generation
  - Better reflects real model behavior

**Implementation Steps**:
1. Modify generation script to use temperature=0
2. Re-generate all question responses with fixed seed
3. Capture activations during the same deterministic run
4. Re-label all question pairs for faithfulness
5. Verify: Same input + seed always produces same label

**Time Estimate**: Should complete this first as it's foundational

---

#### Task 1.2: Dramatically Increase Sample Size
**Goal**: Achieve statistical reliability (target: 300-500 question pairs)

**Implementation Steps**:
1. Expand question generation script:
   - Generate 500 numerical comparison pairs (instead of 50)
   - Maintain same quality checks
   - Diversify number ranges and difficulties
   
2. Create proper data splits:
   - Training: 70% (~350 pairs)
   - Validation: 15% (~75 pairs) - for early stopping
   - Test: 15% (~75 pairs) - for final evaluation
   
3. Implement stratified sampling to maintain class balance

**Why This Matters**: 
- 75 test samples → ±11% confidence interval (vs ±30-40% currently)
- Can actually detect 5-10% accuracy improvements reliably
- Enables cross-validation and robust evaluation

---

### Phase 2: Improve Activation Capture (High Priority)

#### Task 2.1: Capture Activations During Reasoning
**Goal**: Get activations from the actual reasoning process

**Implementation Approaches**:

**Approach A: Last Token Before Answer**
- Capture activations at the final token of `<think>` content
- This is where the model has "decided" on an answer
- Easier to implement

**Approach B: Multi-Point Capture**
- Capture at several points: end of question, mid-reasoning, final token
- Provides richer temporal information
- Could train probes on concatenated features

**Implementation Steps**:
1. Modify activation caching to track generation step-by-step
2. Identify key positions:
   - After question tokens
   - After `<think>` tag
   - At last generated token before answer
3. Extract activations at each position
4. Create datasets with different timing captures
5. Compare probe performance across timing strategies

---

#### Task 2.2: Replace Mean-Pooling with Better Aggregation
**Goal**: Preserve important positional signals

**Options to Implement**:

1. **Last Token Representation** (Simplest)
   - Use only final token's hidden state
   - Common practice in classification tasks
   
2. **Attention-Weighted Pooling** (Better)
   - Learn attention weights over sequence
   - Focuses on important tokens (numbers, comparisons)
   
3. **Token-Specific Features** (Most Targeted)
   - Extract representations of the two numbers being compared
   - Extract representation of comparison operator
   - Concatenate these specific features
   
4. **Hierarchical Features** (Most Comprehensive)
   - Capture both token-level and sequence-level features
   - Use max-pooling + mean-pooling + last-token

**Recommended**: Start with last-token, then try attention-weighted

---

### Phase 3: Improve Probe Training (High Priority)

#### Task 3.1: Proper Training Regime
**Changes Needed**:

1. **Add bias term** to linear probe (standard practice)
2. **Implement early stopping** using validation set
3. **Add L2 regularization** to prevent overfitting
4. **Reduce epochs** to 20 (with early stopping)
5. **Use class weights** to handle imbalance if needed

**Code Changes**:
```python
# Current: probe = nn.Linear(d_model, 1, bias=False)
# New:
probe = nn.Linear(d_model, 1, bias=True)  # Add bias

# Add to training loop:
early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Add regularization to optimizer:
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
```

#### Task 3.2: Robust Evaluation
**Implement**:

1. **Cross-validation** (k-fold, k=5 or 10)
   - Get stable performance estimates
   - Compute mean and std of metrics
   
2. **Multiple evaluation metrics**:
   - Accuracy
   - Balanced accuracy (for class imbalance)
   - AUC-ROC
   - F1 score
   - Precision/Recall
   
3. **Confidence intervals** via bootstrapping
   - Resample test set 1000 times
   - Compute 95% CI on accuracy
   
4. **Baseline comparisons**:
   - Random baseline: 50%
   - Majority class baseline: % of most common class
   - Stratified random: weighted by class distribution

#### Task 3.3: Statistical Reporting
**Report for each layer**:
- Mean ± std accuracy (across CV folds)
- 95% confidence interval
- Comparison to all baselines
- AUC with confidence interval
- Confusion matrix

---

### Phase 4: Expand Task Diversity (Medium Priority)

#### Task 4.1: Add More Task Types
**Goal**: Test if findings generalize beyond simple numeric comparisons

**New Task Types to Implement**:

1. **Arithmetic Operations** (still math, slightly harder)
   - "Is 23 + 15 greater than 40?"
   - Requires calculation in reasoning

2. **Logical Comparisons** (different domain)
   - "If all A are B and all B are C, are all A necessarily C?"
   - Tests logical consistency

3. **Commonsense Reasoning** (non-mathematical)
   - "If it's raining, will the ground be wet?"
   - Flipped: "If the ground is wet, is it definitely raining?"

4. **Multi-hop Reasoning** (more complex)
   - Questions requiring 2-3 reasoning steps
   - More room for unfaithful reasoning

**Implementation**:
- Generate 300 pairs per task type
- Use same faithfulness labeling methodology
- Train separate probes per task type
- Compare: Is faithfulness easier to detect in some domains?

---

### Phase 5: Explore Non-Linear Methods (Medium Priority)

#### Task 5.1: Simple Non-Linear Probes
**Goal**: Test if faithfulness is there but non-linearly encoded

**Models to Try**:

1. **2-Layer MLP** (simple upgrade)
   ```python
   probe = nn.Sequential(
       nn.Linear(d_model, 256),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(256, 1)
   )
   ```

2. **Kernel SVM** (non-linear decision boundary)
   - Try RBF kernel
   - Compare to linear SVM

3. **Gradient Boosted Trees** (non-parametric)
   - XGBoost or LightGBM
   - Can handle non-linear patterns well
   - Provides feature importance

**Evaluation**: Compare against improved linear probe
- If non-linear >> linear: faithfulness is encoded non-linearly
- If non-linear ≈ linear: may not be encoded at all (or noise dominates)

---

#### Task 5.2: Attention Pattern Analysis
**Goal**: Look for faithfulness signals in attention heads

**Analysis to Perform**:

1. **Head-level correlation**:
   - For each attention head, compute average attention pattern
   - Compare patterns: faithful vs unfaithful examples
   - Use statistical tests to find heads with significant differences

2. **Attention to key tokens**:
   - Measure attention to number tokens
   - Measure attention to comparison operator tokens
   - Hypothesis: Faithful reasoning shows stronger attention to critical tokens

3. **Visualization**:
   - Create attention heatmaps for faithful/unfaithful cases
   - Look for interpretable patterns

**Tools**: Use TransformerLens' attention analysis utilities

---

### Phase 6: Causal Interventions (Lower Priority, High Impact)

#### Task 6.1: Activation Patching
**Goal**: Test causality, not just correlation

**Experiments to Run**:

1. **Faithful → Unfaithful patching**:
   - Take activations from a faithful example at layer L
   - Patch them into an unfaithful example's forward pass
   - Does this "fix" the unfaithful reasoning?

2. **Layer-specific effects**:
   - Patch each layer individually to find critical layers
   - Hypothesis: If layer 24 truly encodes faithfulness, patching it should have strongest effect

3. **Feature direction patching**:
   - Extract the "probe direction" (the learned weight vector)
   - Add/subtract multiples of this direction to activations
   - Does moving along this direction change faithfulness?

**Implementation**:
- Use TransformerLens' `hook` mechanism
- Requires generating both faithful and unfaithful examples
- Measure: Does answer flip? Does reasoning chain change?

---

### Phase 7: Compare Model Scales (Lower Priority)

#### Task 7.1: Evaluate Larger Models
**Goal**: Test the "small models more faithful" hypothesis directly

**Models to Test**:
- DeepSeek-R1-Distill-Qwen-7B (medium)
- DeepSeek-R1-Distill-Qwen-14B or 32B (if available, large)

**Controlled Comparison**:
- Use **exact same** 500 question pairs
- Use **same** prompting format
- Use **same** faithfulness labeling criteria
- Use **same** probe training procedure

**Analysis**:
- Does faithfulness rate actually decrease with size?
- Does probe accuracy change with model size?
- Are larger models' faithfulness patterns more/less linear?

**This addresses reviewer's key point**: Currently we can't claim small > large without controlled comparison

---

## Prioritized Roadmap

### Immediate (Week 1-2): Critical Fixes
- [ ] **Task 1.1**: Fix label-activation mismatch (use temp=0)
- [ ] **Task 1.2**: Generate 500 question pairs with proper splits
- [ ] **Task 3.1**: Implement proper probe training (bias, early stopping, regularization)
- [ ] **Task 3.2**: Implement robust evaluation (cross-validation, multiple metrics)

**Expected Outcome**: Clean, reliable baseline with proper statistics

---

### Short-term (Week 3-4): Improve Signal Detection
- [ ] **Task 2.1**: Capture activations during reasoning (last token approach)
- [ ] **Task 2.2**: Implement last-token representation (replace mean-pooling)
- [ ] **Task 3.3**: Comprehensive statistical reporting
- [ ] **Task 4.1**: Add 2-3 additional task types (300 pairs each)

**Expected Outcome**: Better chance of detecting linear signal if it exists

---

### Medium-term (Week 5-6): Explore Non-Linearity
- [ ] **Task 5.1**: Train non-linear probes (MLP, SVM, XGBoost)
- [ ] **Task 5.2**: Analyze attention patterns
- [ ] Compare linear vs non-linear performance

**Expected Outcome**: Determine if faithfulness is encoded non-linearly

---

### Long-term (Week 7-8): Causal & Comparative
- [ ] **Task 6.1**: Activation patching experiments
- [ ] **Task 7.1**: Evaluate larger models for comparison
- [ ] Final comprehensive analysis

**Expected Outcome**: Causal understanding and model-scale comparison

---

## Success Metrics

### How to Know We've Addressed the Review

**Minimum Bar** (fixes critical issues):
- ✅ Labels and activations perfectly aligned (deterministic)
- ✅ Test set size ≥ 75 examples (±11% CI or better)
- ✅ Proper statistical evaluation (CV, multiple metrics, CI)
- ✅ Probe performance compared to majority baseline, not 50%
- ✅ Clear reporting of what is/isn't supported by evidence

**Good Progress** (addresses all major concerns):
- ✅ All minimum bar items
- ✅ Activations captured during reasoning, not just before
- ✅ Token-specific or attention-weighted features (not mean-pooling)
- ✅ Multiple task types tested (3+)
- ✅ Non-linear probes explored

**Excellent Work** (publishable quality):
- ✅ All good progress items
- ✅ Causal interventions (activation patching)
- ✅ Controlled comparison with larger models
- ✅ Sample size 500+ pairs per task type
- ✅ Confidence intervals < ±10% on all claims

---

## Revised Research Questions

Based on the review, we should be asking:

### Primary Questions:
1. **Is CoT faithfulness encoded in model activations?** (don't assume linearly)
2. **If so, is it accessible via linear probes?** (separate question)
3. **At what point in computation does faithfulness manifest?** (during reasoning, not just after input)

### Secondary Questions:
4. **Does faithfulness encoding differ by task type?**
5. **Does model scale affect faithfulness rate and linear decodability?**
6. **What attention patterns correlate with faithful reasoning?**

### Causal Questions:
7. **Can we causally manipulate faithfulness by intervening on activations?**
8. **Which layers are causally responsible for faithful reasoning?**

---

## Updated Interpretation of Current Results

### What We Actually Showed

✅ **Confirmed**:
- This 1.5B model achieves 72% faithfulness on simple numeric comparisons
- This is a well-defined, measurable property of the model

⚠️ **Suggestive but Not Proven**:
- Small models might be more faithful than large models (need controlled comparison)
- Late layers might contain more faithfulness information (visualization unreliable)

❌ **Not Supported**:
- Faithfulness is linearly encoded (probe at baseline)
- Linear probes are feasible for real-time monitoring (66.7% is unreliable)
- There's a "modest linear signal" (this was misinterpreting baseline performance)

### Honest Conclusion from Current Work

> "We demonstrated that a 1.5B reasoning model can achieve 72% faithfulness on simple numeric comparisons. However, our initial attempt to detect faithfulness using linear probes on pre-generation activations was unsuccessful (66.7% accuracy, at majority-class baseline). This suggests either: (1) faithfulness is not linearly encoded in pre-generation activations, (2) our methodology had critical limitations (label noise, small sample, timing mismatch) that obscured the signal, or (3) non-linear methods are required. Future work with improved methodology is needed to distinguish these possibilities."

---

## Key Quotes from Reviewer

### On the Main Finding:
> "The probe's performance is so low and the test set so small that it's unclear any real signal was captured."

> "The results do not convincingly support a conclusion that faithfulness is linearly encoded in a readily detectable way – they rather indicate the opposite."

### On What to Do Next:
> "Expanding the dataset to hundreds of question pairs (e.g. 200-500 pairs) would greatly improve statistical power."

> "Use deterministic generation (temperature 0) when labeling faithfulness to remove randomness from the equation."

> "Capture activations at critical points during the chain-of-thought and at the moment just before the answer is produced."

> "Avoid blunt mean-pooling. Investigate token-level features or structured representations."

### On the Overall Work:
> "This research project is a valuable initial foray into mechanistically evaluating chain-of-thought faithfulness in a small model."

> "Denis Lim's project provides an insightful starting point and a solid experimental framework, but the evidence is too limited to make strong claims yet."

---

## Next Steps: Immediate Action Items

### This Week:
1. **Review this plan** with the team
2. **Prioritize** which fixes to tackle first (recommend: 1.1, 1.2, 3.1)
3. **Set up infrastructure** for larger dataset generation
4. **Design** deterministic generation pipeline

### Decision Point:
**Do we want to:**
- **Option A**: Fix everything and re-run (cleaner science, takes longer)
- **Option B**: Quick fix (temp=0, 200 pairs) to validate approach, then scale up
- **Option C**: Pivot to non-linear methods immediately (given linear seems weak)

**Recommendation**: Option B - quick validation with 200 deterministic pairs, then full scale-up if promising

---

## Conclusion

The expert review was thorough, fair, and ultimately very helpful. The reviewer identified that our current results don't support the strong claims we'd like to make, primarily due to methodological limitations (small sample, label-activation mismatch, timing issues) rather than fundamental problems with the approach.

**The good news**: The framework is sound, the faithfulness finding is solid, and we have a clear path forward.

**The challenge**: We need significantly more data and careful methodology to make reliable claims about linear encoding.

**The opportunity**: By addressing these issues systematically, we can produce much stronger evidence and potentially publishable results.

The review transforms this from "interesting preliminary finding" to "clear research agenda with specific, actionable improvements."
