# Phase 3: Mechanistic Analysis - Implementation Guide

## Overview

**Phase 3 Goal:** Find mechanistic explanation for faithfulness via linear probe analysis  
**Time:** 6-7 hours  
**Deliverables:**
- 4 activation cache files (`data/activations/layer_*_activations.pt`)
- 1 probe results file (`results/probe_results/all_probe_results.pt`)
- 1 performance plot (`results/probe_results/probe_performance.png`)

---

## Phase 3 Obligation Checklist

### Data Contract 1: Activation Cache Files ✓

Each `data/activations/layer_{N}_activations.pt` must contain:
- [x] `faithful` tensor: `[n_faithful, d_model]` where `n_faithful >= 10`
- [x] `unfaithful` tensor: `[n_unfaithful, d_model]` where `n_unfaithful >= 10`
- [x] dtype: `float32` or `float16`
- [x] No sequence dimension (mean-pooled)
- [x] Files for layers: 6, 12, 18, 24

### Data Contract 2: Probe Results ✓

`results/probe_results/all_probe_results.pt` must contain:
- [x] Results for all 4 layers
- [x] Each result has: `layer`, `accuracy`, `auc`, `probe`, `direction`
- [x] `accuracy` in [0, 1]
- [x] `auc` in [0, 1]
- [x] At least one layer with `accuracy > 0.55`

---

## Prerequisites

**Phase 3 REQUIRES Phase 2 to be complete:**
- ✓ `data/responses/model_1.5B_responses.jsonl` exists (100 responses)
- ✓ `data/processed/faithfulness_scores.csv` exists (50 pairs scored)

**Check Phase 2:**
```bash
python tests/validate_phase2.py
```

**Additional Requirements:**
```bash
pip install transformer-lens scikit-learn matplotlib
```

---

## Usage

### Step 1: Cache Activations (2-3 hours)

Caches residual stream activations for faithful vs unfaithful responses.

**Run on GPU pod (requires model loading):**
```bash
python src/mechanistic/cache_activations.py
```

**Expected output:**
```
PHASE 3 TASK 3.2: Cache Activations
====================================
[1/6] Checking Phase 2 outputs...
   ✓ Phase 2 outputs found
[2/6] Loading faithfulness labels...
   Found X faithful pairs
   Found Y unfaithful pairs
   Using 30 faithful pairs
   Using 20 unfaithful pairs
[3/6] Loading model responses...
   ✓ Loaded 50 response pairs
[4/6] Loading model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ✓ Model loaded on cuda
[5/6] Caching activations at layers [6, 12, 18, 24]...
   Caching faithful responses...
   Caching unfaithful responses...
[6/6] Saving activation caches...
   ✓ layer_6: 30 faithful, 20 unfaithful, d_model=1536
   ✓ layer_12: 30 faithful, 20 unfaithful, d_model=1536
   ✓ layer_18: 30 faithful, 20 unfaithful, d_model=1536
   ✓ layer_24: 30 faithful, 20 unfaithful, d_model=1536

✅ PHASE 3 TASK 3.2 COMPLETE
```

**Produces:**
- `data/activations/layer_6_activations.pt`
- `data/activations/layer_12_activations.pt`
- `data/activations/layer_18_activations.pt`
- `data/activations/layer_24_activations.pt`

---

### Step 2: Train Linear Probes (1-2 hours)

Trains linear classifiers to predict faithfulness from activations.

**Can run locally (uses cached activations):**
```bash
python src/mechanistic/train_probes.py
```

**Expected output:**
```
PHASE 3 TASK 3.3: Train Linear Probes
======================================
[1/4] Checking activation caches...
   ✓ Found activation caches for 4 layers
[2/4] Training probes (epochs=50, lr=0.001)...

   Training layer_6...
     Data: 30 faithful, 20 unfaithful
     Result: accuracy=0.XXX, auc=0.XXX

   Training layer_12...
     Data: 30 faithful, 20 unfaithful
     Result: accuracy=0.XXX, auc=0.XXX

   Training layer_18...
     Data: 30 faithful, 20 unfaithful
     Result: accuracy=0.XXX, auc=0.XXX

   Training layer_24...
     Data: 30 faithful, 20 unfaithful
     Result: accuracy=0.XXX, auc=0.XXX

   ✓ Best layer: layer_XX (accuracy=0.XXX)

[3/4] Generating probe performance plot...
   ✓ Saved plot to: results/probe_results/probe_performance.png
[4/4] Saving probe results...
   ✓ Saved results to: results/probe_results/all_probe_results.pt

PHASE 3 ACCEPTANCE CRITERIA CHECK
==================================
✓ Results for all 4 layers present
✓ All accuracy/auc values in [0, 1]
✓ At least one layer has accuracy > 0.55 (0.XXX)

✅ PHASE 3 TASK 3.3 COMPLETE
```

**Produces:**
- `results/probe_results/all_probe_results.pt`
- `results/probe_results/probe_performance.png`

---

### Step 3: Validate Phase 3 (5 minutes)

Verify all Phase 3 deliverables satisfy the specification.

```bash
python tests/validate_phase3.py
```

**Expected output (if passing):**
```
PHASE 3 VALIDATION: Mechanistic Analysis
=========================================

[Step 1/3] Checking output files...
✓ data/activations/layer_6_activations.pt
✓ data/activations/layer_12_activations.pt
✓ data/activations/layer_18_activations.pt
✓ data/activations/layer_24_activations.pt
✓ results/probe_results/all_probe_results.pt
✓ results/probe_results/probe_performance.png

[Step 2/3] Validating activation caches...
✓ layer_6: 30 faithful, 20 unfaithful, d_model=1536
✓ layer_12: 30 faithful, 20 unfaithful, d_model=1536
✓ layer_18: 30 faithful, 20 unfaithful, d_model=1536
✓ layer_24: 30 faithful, 20 unfaithful, d_model=1536
✓ d_model consistent across all layers: 1536

✅ Activation caches valid (Data Contract 1 satisfied)

[Step 3/3] Validating probe results...
✓ layer_6: accuracy=0.XXX, auc=0.XXX, direction_dim=1536
✓ layer_12: accuracy=0.XXX, auc=0.XXX, direction_dim=1536
✓ layer_18: accuracy=0.XXX, auc=0.XXX, direction_dim=1536
✓ layer_24: accuracy=0.XXX, auc=0.XXX, direction_dim=1536

Best layer: layer_XX with accuracy=0.XXX
✓ Exceeds Phase 3 threshold (>0.55)

✅ Probe results valid (Data Contract 2 satisfied)

PHASE 3 ACCEPTANCE CRITERIA SUMMARY
====================================
✓ Activation files exist for all 4 layers
✓ Each file has 'faithful' and 'unfaithful' tensors
✓ Minimum 10 examples in each class
✓ Probe results file exists
✓ Results for all 4 layers present
✓ All accuracy/auc values in [0, 1]
✓ At least one layer has accuracy > 0.55

✅✅✅ ALL PHASE 3 CHECKS PASSED ✅✅✅

✅ Ready to proceed to Phase 4 (Report)
```

---

## Interpreting Results

### Best Probe Accuracy > 0.65
**Finding:** Strong linear faithfulness direction exists  
**Interpretation:** Faithfulness is explicitly encoded in a linear subspace. Can potentially monitor CoT faithfulness using simple linear classifiers.

### Best Probe Accuracy 0.55-0.65
**Finding:** Weak linear faithfulness signal  
**Interpretation:** Some linear component exists but may be noisy or distributed across layers.

### Best Probe Accuracy < 0.55
**Finding:** Null result - no linear faithfulness direction  
**Interpretation:** Faithfulness is not linearly encoded. May emerge from complex non-linear interactions. This is also a scientifically valid result.

---

## Phase 3 Boundaries

### Phase 3 DOES:
- ✓ Cache activations for faithful/unfaithful responses
- ✓ Train linear probes at layers [6, 12, 18, 24]
- ✓ Generate probe performance plot
- ✓ Validate all outputs against specification

### Phase 3 does NOT:
- ✗ Perform attention analysis (not in Phase 3 spec)
- ✗ Generate executive summary (that's Phase 4)
- ✗ Test multiple models (1.5B only in Phase 3)
- ✗ Implement baselines/ablations (beyond scope)
- ✗ Create presentation slides (that's Phase 4)

**If you need these features, they belong in Phase 4 or future extensions.**

---

## Troubleshooting

### "Phase 2 output missing"
**Problem:** Phase 2 is not complete  
**Solution:** Run Phase 2 first:
```bash
python tests/validate_phase2.py  # Check Phase 2 status
python src/inference/batch_inference.py  # If needed
python src/evaluation/score_faithfulness.py  # If needed
```

### "Insufficient faithful/unfaithful samples"
**Problem:** Phase 2 has very high faithfulness rate (< 10 unfaithful examples)  
**Solution:** This is a valid finding! Options:
1. Generate more question pairs in Phase 1
2. Reframe as "small models are very faithful" (still interesting)
3. Continue with warning (validation may fail but result is valid)

### "TransformerLens not installed"
**Problem:** Missing dependency  
**Solution:**
```bash
pip install transformer-lens
```

### "No layer exceeds 0.55 accuracy"
**Problem:** No linear faithfulness direction found  
**Solution:** This is a **null result** - scientifically valid! Document it in Phase 4 as:
> "We did not find a strong linear faithfulness direction (best accuracy: X.XX). This suggests faithfulness may be encoded in a distributed or non-linear way."

---

## File Structure After Phase 3

```
mats-10.0/
├── data/
│   ├── activations/                    [NEW]
│   │   ├── layer_6_activations.pt      [NEW]
│   │   ├── layer_12_activations.pt     [NEW]
│   │   ├── layer_18_activations.pt     [NEW]
│   │   └── layer_24_activations.pt     [NEW]
│   ├── processed/
│   │   └── faithfulness_scores.csv     [from Phase 2]
│   └── responses/
│       └── model_1.5B_responses.jsonl  [from Phase 2]
├── results/
│   └── probe_results/                  [NEW]
│       ├── all_probe_results.pt        [NEW]
│       └── probe_performance.png       [NEW]
├── src/
│   └── mechanistic/                    [NEW]
│       ├── __init__.py                 [NEW]
│       ├── types.py                    [NEW]
│       ├── cache_activations.py        [NEW]
│       └── train_probes.py             [NEW]
└── tests/
    ├── test_phase3_contracts.py        [NEW]
    └── validate_phase3.py              [NEW]
```

---

## Next Steps

Once `python tests/validate_phase3.py` passes with exit code 0:

**Proceed to Phase 4:**
- Write executive summary (2-3 pages)
- Create 5-slide presentation
- Clean up code and documentation
- Run final validation

**See:** `phased_implementation_plan.md` lines 2043-2542 for Phase 4 details

---

## Time Tracking

- [x] Task 3.1: Install TransformerLens (~30 min)
- [x] Task 3.2: Cache Activations (~2-3 hours)
- [x] Task 3.3: Train Probes (~1-2 hours)
- [x] Task 3.4: Validation (~30 min)
- [x] Task 3.5: Interpret Results (~30 min)

**Total Phase 3 Time:** ~6-7 hours

---

## References

- **Specification:** `phased_implementation_plan.md` lines 1334-2041
- **Technical Details:** `technical_specification.md` lines 735-1189
- **Phase 3 Contracts:** `src/mechanistic/types.py`
- **Phase 3 Tests:** `tests/test_phase3_contracts.py`



