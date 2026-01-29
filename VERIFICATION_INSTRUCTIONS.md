# Data Verification Instructions

## Purpose
Verify the test data quality before drawing conclusions about probe performance.

## What to Check
1. ✅ Question pairs have correct expected answers
2. ✅ Answer extraction logic is working properly
3. ✅ Faithfulness scoring is sound
4. 🔍 Examine unfaithful response patterns

---

## Copy Files to Pod

```bash
scp /Users/denislim/workspace/mats-10.0/verification_tools.tar.gz root@<POD_IP>:/unfaithful-reasoning/
```

---

## On Remote Pod

```bash
cd /unfaithful-reasoning
tar -xzf verification_tools.tar.gz
source venv/bin/activate

# Run comprehensive verification
python verify_test_data.py

# Analyze unfaithful responses in detail
python analyze_unfaithful.py
```

---

## What Each Script Does

### `verify_test_data.py`
Comprehensive verification in 3 sections:

**Section 1: Question Pairs**
- Checks all 200 pairs are correctly formatted
- Verifies q1_answer and q2_answer are opposite (Yes/No)
- Shows 5 example pairs with manual verification

**Section 2: Responses & Extraction**
- Shows extraction statistics (Yes/No/Unknown counts)
- Calculates match rate (extracted vs expected)
- Shows examples of:
  - ✓ Correct extractions
  - ✗ Incorrect extractions
  - ? Unknown extractions

**Section 3: Faithfulness Scoring**
- Shows faithful/unfaithful distribution
- Shows 5 faithful examples (both Q1 and Q2 correct)
- Shows 10 unfaithful examples with reasons

### `analyze_unfaithful.py`
Detailed analysis of unfaithful responses:

**Categories:**
1. Both answers wrong
2. Q1 wrong only
3. Q2 wrong only
4. Extraction failed

**For each category:**
- Shows 3 full examples with complete responses
- Includes analysis of why extraction failed
- Helps identify systematic issues

---

## What to Look For

### Red Flags (Data Issues):
❌ Expected answers are wrong (e.g., "Is 900 > 795?" expects "No")
❌ Extraction consistently wrong for correct responses
❌ Most unfaithful are due to extraction failures

### Green Flags (Valid Results):
✅ Expected answers are correct
✅ Extraction works well on clear Yes/No responses
✅ Unfaithful responses show genuine model errors or inconsistency

---

## Expected Output

If data is good, you should see:
- ~50% extraction match rate (model gets many wrong)
- Unfaithful responses have genuine reasoning errors
- Examples where model says contradictory things

If data has issues:
- Expected answers don't match ground truth
- Extraction fails on responses that clearly say "Yes" or "No"
- Most unfaithful are extraction failures, not model errors

---

## After Verification

Based on what you find:

**If data is valid:**
→ Probe results (50% accuracy) are real
→ Linear probes genuinely don't work
→ Document as negative result

**If data has issues:**
→ Fix the specific problem (questions, extraction, or scoring)
→ Re-run pipeline
→ Re-test probe

