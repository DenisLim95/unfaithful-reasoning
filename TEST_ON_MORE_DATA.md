# Testing Your Probe on More Data

## Current Situation

- **Original test set:** 9 samples (too small for reliable results)
- **Your probe:** Already trained on 35 samples
- **Goal:** Test on 50-100+ new samples for statistically significant results

---

## Quick Start (3 Options)

### Option 1: Quick Test (Test Only - Fastest) ⚡

If you already have cached test activations:

```bash
python test_probe_on_new_data.py --test-only
```

### Option 2: Generate & Test Locally (100 questions) 🖥️

**Warning:** Requires GPU and ~1-2 hours

```bash
# Full pipeline (generates questions, runs model, tests probe)
python test_probe_on_new_data.py --num-questions 100
```

### Option 3: Run on Remote GPU (Recommended) ☁️

**Best option:** More data, faster, reliable

```bash
# 1. Generate questions locally (fast, no GPU needed)
python generate_test_questions.py --num-pairs 200

# 2. Copy to remote machine
scp data/raw/test_question_pairs.json user@remote:/path/to/mats-10.0/data/raw/

# 3. On remote machine, run inference & caching
python test_probe_on_new_data.py --skip-generation

# 4. Copy results back
scp -r user@remote:/path/to/mats-10.0/data/test_activations ./data/
scp -r user@remote:/path/to/mats-10.0/data/processed/test_faithfulness_scores.csv ./data/processed/

# 5. Test locally (fast, CPU only)
python test_probe_on_new_data.py --test-only
```

---

## What This Does

### Step 1: Generate New Test Questions (CPU only)

```bash
python generate_test_questions.py --num-pairs 200
```

**Output:**
```
✓ Generated 200 question pairs
  - 80 easy
  - 80 medium
  - 40 hard
✓ Saved to: data/raw/test_question_pairs.json
```

**Creates:** 200 pairs × 2 variants = **400 prompts**

### Step 2: Run Full Test Pipeline (Requires GPU)

```bash
python test_probe_on_new_data.py --num-questions 100
```

**What it does:**
1. **Generates test questions** (if not skipped)
2. **Runs model inference** → 200 responses (100 pairs × 2)
3. **Scores faithfulness** → Identifies faithful/unfaithful pairs
4. **Caches activations** → Extracts activations at layers 6, 12, 18, 24
5. **Tests your trained probe** → Reports accuracy on new data

**Output:**
```
============================================================
TESTING EXISTING PROBE ON NEW DATA
============================================================

layer_12:
  Test samples: 100 (60 faithful, 40 unfaithful)
  Accuracy: 72.0%
  AUC-ROC: 0.689

============================================================
COMPARISON: Original vs New Test Set
============================================================

layer_12:
  Original test (9 samples):  66.7%
  New test (100 samples): 72.0%
  Change: +5.3 percentage points
```

---

## Command Line Options

### For `generate_test_questions.py`:

```bash
# Generate 50 pairs
python generate_test_questions.py --num-pairs 50

# Generate 500 pairs
python generate_test_questions.py --num-pairs 500

# Custom output location
python generate_test_questions.py --num-pairs 100 --output data/my_test_questions.json
```

### For `test_probe_on_new_data.py`:

```bash
# Skip steps (if already done):
python test_probe_on_new_data.py --skip-generation        # Use existing questions
python test_probe_on_new_data.py --skip-inference         # Use existing responses
python test_probe_on_new_data.py --skip-caching           # Use existing activations
python test_probe_on_new_data.py --test-only              # Just test probe

# Control test set size:
python test_probe_on_new_data.py --num-questions 50       # Smaller test set
python test_probe_on_new_data.py --num-questions 200      # Larger test set
```

---

## Recommended Workflow

### For Best Results (Remote GPU):

```bash
# ====================
# ON YOUR LOCAL MACHINE
# ====================

# 1. Generate 200 test questions (takes ~5 seconds)
python generate_test_questions.py --num-pairs 200

# 2. Copy to remote
scp data/raw/test_question_pairs.json user@remote-gpu:/path/to/mats-10.0/data/raw/

# ====================
# ON REMOTE GPU MACHINE
# ====================

cd /path/to/mats-10.0

# 3. Run inference & caching (takes ~1-2 hours)
python test_probe_on_new_data.py \
    --skip-generation \
    --num-questions 200

# ====================
# COPY BACK TO LOCAL
# ====================

# 4. Copy results back to local machine
scp -r user@remote-gpu:/path/to/mats-10.0/data/test_activations ./data/
scp user@remote-gpu:/path/to/mats-10.0/data/processed/test_faithfulness_scores.csv ./data/processed/
scp user@remote-gpu:/path/to/mats-10.0/data/responses/test_responses.jsonl ./data/responses/

# ====================
# ON YOUR LOCAL MACHINE
# ====================

# 5. Test the probe (takes ~1 second)
python test_probe_on_new_data.py --test-only
```

---

## What You'll Learn

### Statistical Significance

**Current:** 9 test samples → ±32% confidence interval  
**With 100 samples:** ±10% confidence interval  
**With 200 samples:** ±7% confidence interval

### Generalization Performance

Does your probe work on:
- ✅ **New numbers?** (e.g., "Is 734 larger than 921?" - never seen before)
- ✅ **Different difficulty?** (compare performance on easy vs hard)
- ✅ **Out-of-distribution?** (true test of learned patterns)

### Example Output:

```
COMPARISON: Original vs New Test Set
====================================

layer_12:
  Original test (9 samples):   66.7%  ← Small, unreliable
  New test (100 samples):      68.5%  ← Larger, more reliable
  Change: +1.8 percentage points

Interpretation:
  ✓ Probe generalizes reasonably well
  ✓ Performance consistent across datasets
  ⚠️ Still below 70% - weak signal confirmed
```

---

## Expected Timeline

| Dataset Size | Question Gen | Model Inference | Activation Cache | Testing | Total |
|--------------|--------------|-----------------|------------------|---------|-------|
| 50 pairs     | 5 sec        | 20 min          | 10 min           | 1 sec   | ~30 min |
| 100 pairs    | 5 sec        | 40 min          | 20 min           | 1 sec   | ~1 hour |
| 200 pairs    | 5 sec        | 80 min          | 40 min           | 1 sec   | ~2 hours |

**Note:** Model inference is the slowest step (requires GPU).

---

## Interpreting Results

### Good Signs ✅

- Accuracy on new data **within ±5%** of original
- AUC improves with more data
- Performance consistent across difficulty levels

### Bad Signs ⚠️

- Accuracy **drops >10%** on new data → Overfitting
- Accuracy **increases >10%** → Original test set was unrepresentative
- High variance across difficulty → Probe is brittle

### Example Interpretations:

**Scenario 1: Consistent Performance**
```
Original: 66.7% (9 samples)
New:      68.2% (100 samples)
→ Good! Probe generalizes well, original estimate was accurate
```

**Scenario 2: Overfitting**
```
Original: 66.7% (9 samples)
New:      52.3% (100 samples)
→ Bad! Probe memorized training data, doesn't generalize
```

**Scenario 3: Lucky Original Test**
```
Original: 66.7% (9 samples)
New:      78.9% (100 samples)
→ Interesting! Original test set was harder than typical
```

---

## Troubleshooting

### "CUDA out of memory"

```bash
# Reduce batch size (edit script line with max_new_tokens)
# Or process in smaller chunks
```

### "Model takes too long"

```bash
# Generate smaller test set
python test_probe_on_new_data.py --num-questions 50

# Or use faster sampling (edit temperature/max_tokens)
```

### "No test activations found"

```bash
# Make sure you ran caching step first
python test_probe_on_new_data.py  # Full pipeline
# Then
python test_probe_on_new_data.py --test-only
```

---

## Next Steps After Testing

### If probe generalizes well (new accuracy ≈ original):
- ✅ Report results with confidence
- ✅ Investigate which samples it gets wrong
- ✅ Analyze failure modes

### If probe overfits (new accuracy << original):
- ⚠️ Retrain with regularization
- ⚠️ Use more training data
- ⚠️ Try simpler probe architecture

### If probe does surprisingly well (new accuracy >> original):
- 🎉 Great news! Original test was just hard
- 🎉 Report the new, more reliable accuracy
- 🎉 Your findings are stronger than you thought

---

## Files Created

```
data/
├── raw/
│   └── test_question_pairs.json          # New test questions
├── responses/
│   └── test_responses.jsonl              # Model responses to test questions
├── processed/
│   └── test_faithfulness_scores.csv      # Faithfulness scores for test pairs
└── test_activations/                     # Test activations (separate from training)
    ├── layer_6_activations.pt
    ├── layer_12_activations.pt
    ├── layer_18_activations.pt
    └── layer_24_activations.pt
```

---

## Questions?

- **"Should I retrain the probe?"** → No! You want to test the EXISTING probe's generalization
- **"How many questions should I generate?"** → 100-200 for reliable results
- **"Can I run this without GPU?"** → Question generation: Yes. Model inference: No (too slow)
- **"Will this change my original results?"** → No, this is separate testing

