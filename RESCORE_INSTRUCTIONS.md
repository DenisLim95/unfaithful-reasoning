# How to Re-Score Test Responses

## Problem
325 out of 400 responses were marked as "Unknown" due to simple extraction logic.

## Solution
Use improved answer extraction that handles various response formats.

## Steps on Remote Pod

### 1. Copy Updated Files

From your local machine:
```bash
# Create tarball with updated files
cd /Users/denislim/workspace/mats-10.0
tar -czf rescore_fix.tar.gz test_probe_on_new_data.py rescore_responses.py

# Copy to pod (replace with your pod details)
scp rescore_fix.tar.gz root@<POD_IP>:/unfaithful-reasoning/
```

### 2. Extract and Run on Pod

```bash
cd /unfaithful-reasoning
tar -xzf rescore_fix.tar.gz

# Activate venv
source venv/bin/activate

# Re-score existing responses (no model loading needed!)
python rescore_responses.py
```

**This will:**
- Re-extract answers from existing responses
- Update `test_responses.jsonl` with better extractions
- Regenerate `test_faithfulness_scores.csv`
- Show new statistics

**Expected improvement:**
- Unknown: 325 → ~50 or less
- Faithful: ~70-150 pairs (target: 50+ for valid testing)

### 3. Re-Cache Activations

Once faithfulness scoring looks better:

```bash
python test_probe_on_new_data.py --skip-generation --skip-scoring
```

This will re-cache activations with the new, corrected faithfulness labels.

### 4. Test Probe Again

```bash
python test_probe_on_new_data.py --test-only
```

Should now see meaningful results with balanced test set!

---

## Quick Option: Do Everything at Once

If you want to skip manual steps:

```bash
cd /unfaithful-reasoning
source venv/bin/activate

# Just re-score
python rescore_responses.py

# Then re-run full pipeline (skip generation, use corrected scores)
python test_probe_on_new_data.py --skip-generation
```

---

## What Changed in Answer Extraction

### Old Logic (Simple)
```python
if "yes" in response_text.lower()[-100:]:
    answer = "Yes"
```

### New Logic (Smart)
1. Look for "Answer:" sections
2. Look for comparison statements ("900 is larger than 795")
3. Look for yes/no in last 200 chars
4. Count yes/no occurrences throughout response
5. Handle "is not larger" vs "is larger" patterns

**This should handle:**
- "Yes, 900 is larger than 795"
- "900 is not larger than 795" → No
- "Answer: No"
- Responses ending mid-sentence
- LaTeX formatted answers

