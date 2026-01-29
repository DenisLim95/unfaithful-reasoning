# Phase 2 Quickstart Guide

**Goal:** Generate model responses and compute faithfulness scores  
**Time:** 2-4 hours (mostly GPU inference)  
**Prerequisites:** Phase 1 complete, GPU available

---

## Option 1: Automated (Recommended)

Run everything at once:

```bash
./run_phase2.sh
```

This will:
1. Check Phase 1 dependency ✓
2. Run inference (2-3 hours) 🕐
3. Score faithfulness (5-10 min) 📊
4. Validate Phase 2 ✅

---

## Option 2: Manual (Step-by-Step)

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify Phase 1 complete
python tests/validate_questions.py  # Must exit 0
```

### Step 1: Run Inference (2-3 hours)

```bash
python src/inference/batch_inference.py
```

**Output:** `data/responses/model_1.5B_responses.jsonl` (100 responses)

### Step 2: Score Faithfulness (5-10 minutes)

```bash
python src/evaluation/score_faithfulness.py
```

**Output:** `data/processed/faithfulness_scores.csv` (50 scores)

### Step 3: Validate

```bash
python tests/validate_phase2.py
```

**Expected:** Exit code 0, "✅ ALL PHASE 2 CHECKS PASSED"

---

## What You Get

After Phase 2 completes:

1. **100 Model Responses**
   - File: `data/responses/model_1.5B_responses.jsonl`
   - Contains: Full responses with think sections and answers

2. **50 Faithfulness Scores**
   - File: `data/processed/faithfulness_scores.csv`
   - Contains: Consistency, faithfulness, correctness, confidence

3. **Summary Statistics**
   - Faithfulness rate (e.g., 45%)
   - Consistency rate
   - Q1/Q2 accuracy
   - Extraction confidence

---

## Interpreting Results

After validation, you'll see:

```
Summary Statistics:
  • Faithfulness rate: XX.X%
  • Consistency rate: XX.X%
  • Q1 accuracy: XX.X%
  • Q2 accuracy: XX.X%
  • High-confidence extractions: XX.X%
```

**Faithfulness rate:** Percentage of question pairs where model gave consistent answers

**Comparison to prior work:**
- DeepSeek R1 (70B): 39%
- Claude 3.7 Sonnet: 25%
- Your 1.5B model: ?

---

## Next Steps

### If Faithfulness Rate < 90%
✅ Good! You have unfaithful examples for Phase 3 mechanistic analysis

**Proceed if:** ≥10 unfaithful pairs  
**Go to:** Phase 3 - Mechanistic Analysis

### If Faithfulness Rate > 90%
⚠️ Model is very faithful (interesting finding!)

**Options:**
1. Generate more pairs to get ≥10 unfaithful examples
2. Skip Phase 3 or do attention analysis only
3. Reframe as "Why are small models faithful?"

---

## Troubleshooting

### "Phase 1 dependency not satisfied"
**Fix:** Run Phase 1 first: `python src/data_generation/generate_questions.py`

### "CUDA out of memory"
**Fix:** Model needs ~6GB VRAM. Options:
- Use smaller batch size (edit `batch_inference.py`)
- Use CPU (slow): edit script to remove `device_map="auto"`
- Use cloud GPU (Colab, RunPod, etc.)

### "Only X responses (expected 100)"
**Fix:** Inference crashed. Delete partial output and re-run:
```bash
rm data/responses/model_1.5B_responses.jsonl
python src/inference/batch_inference.py
```

### "Low extraction confidence"
**Review:** Inspect some responses manually:
```bash
head -20 data/responses/model_1.5B_responses.jsonl | jq
```
Model may use different answer phrasing than expected.

---

## Files Created

```
data/
├── responses/
│   └── model_1.5B_responses.jsonl      # 100 responses (Task 2.1)
└── processed/
    └── faithfulness_scores.csv         # 50 scores (Task 2.3)
```

---

## Time Breakdown

| Task | Description | Time |
|------|-------------|------|
| 2.1 | Model inference | 2-3 hours |
| 2.2 | Answer extraction | (embedded in 2.3) |
| 2.3 | Score faithfulness | 5-10 minutes |
| 2.4 | Validation | 1 minute |
| **Total** | | **~2-4 hours** |

---

## Help

- **Full docs:** See `PHASE2_README.md`
- **Implementation details:** See `PHASE2_IMPLEMENTATION_SUMMARY.md`
- **Technical spec:** See `technical_specification.md` § 4
- **Phased plan:** See `phased_implementation_plan.md` Phase 2

---

**Ready?** Run `./run_phase2.sh` to get started! 🚀

