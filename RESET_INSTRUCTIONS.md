# How to Reset Test Data

## Quick Reset

**On your remote GPU pod:**

```bash
# Option 1: Use Python script (recommended)
python cleanup_test_data.py

# Option 2: Use bash script
chmod +x cleanup_test_data.sh
./cleanup_test_data.sh

# Option 3: Manual cleanup
rm -rf data/test_activations/
rm -f data/raw/test_question_pairs.json
rm -f data/responses/test_responses.jsonl
rm -f data/processed/test_faithfulness_scores.csv
```

---

## What Gets Deleted ❌

These are **test-specific files** that will be regenerated:

### Test Activations
```
data/test_activations/
├── layer_6_activations.pt    ❌ DELETED
├── layer_12_activations.pt   ❌ DELETED
├── layer_18_activations.pt   ❌ DELETED
└── layer_24_activations.pt   ❌ DELETED
```

### Test Questions
```
data/raw/test_question_pairs.json    ❌ DELETED
```

### Test Responses
```
data/responses/test_responses.jsonl  ❌ DELETED
```

### Test Scores
```
data/processed/test_faithfulness_scores.csv  ❌ DELETED
```

---

## What Gets Preserved ✅

These are your **Phase 3 training artifacts** (keep these!):

### Training Activations (44 samples)
```
data/activations/
├── layer_6_activations.pt     ✅ KEPT
├── layer_12_activations.pt    ✅ KEPT
├── layer_18_activations.pt    ✅ KEPT
└── layer_24_activations.pt    ✅ KEPT
```

### Trained Probes
```
results/probe_results/
├── all_probe_results.pt       ✅ KEPT
└── probe_performance.png      ✅ KEPT
```

### Training Questions (50 pairs)
```
data/raw/question_pairs.json   ✅ KEPT
```

### Training Responses
```
data/responses/model_1.5B_responses.jsonl  ✅ KEPT
```

### Training Faithfulness Scores
```
data/processed/faithfulness_scores.csv     ✅ KEPT
```

### Visualizations
```
results/activation_visualizations/         ✅ KEPT
```

---

## After Cleanup: Run Fresh Test

```bash
# Generate 200 new test question pairs and test the probe
python test_probe_on_new_data.py --num-questions 200
```

This will:
1. Generate 200 **new** question pairs (different from training)
2. Get model responses (~400 responses)
3. Score faithfulness
4. Cache activations to `data/test_activations/`
5. Test the probe and report accuracy

---

## Why Reset?

### When to Reset

- ✅ Before running a fresh test experiment
- ✅ If previous test data is corrupted/incomplete
- ✅ When you want different test questions
- ✅ After changing the test script

### When NOT to Reset

- ❌ If you want to re-test on the SAME test questions (for reproducibility)
- ❌ If you're comparing results across different thresholds on the same data

---

## Complete Reset (Including Training Data)

**⚠️ WARNING: This will delete EVERYTHING and require re-running Phase 3!**

Only do this if you want to start completely from scratch:

```bash
# DANGER ZONE: Delete training data too
rm -rf data/activations/
rm -rf data/test_activations/
rm -rf results/probe_results/
rm -f data/responses/test_responses.jsonl
rm -f data/raw/test_question_pairs.json

# You'll need to re-run Phase 3:
# python src/mechanistic/cache_activations_nnsight.py
# python src/mechanistic/train_probes.py
```

**Don't do this unless you want to retrain everything!**

---

## Verification After Cleanup

Check that test data is gone but training data remains:

```bash
# Should show "not found"
ls data/test_activations/

# Should show 4 files (training activations)
ls data/activations/

# Should show probe results
ls results/probe_results/

# Check file counts
find data/activations -name "*.pt" | wc -l    # Should be 4
find results/probe_results -name "*.pt" | wc -l  # Should be 1
```

---

## Summary

**Simple rule:**
- `test_*` files = temporary, can delete ❌
- Everything else = training data, keep ✅

**After cleanup, run:**
```bash
python test_probe_on_new_data.py --num-questions 200
```

To get fresh test results!



