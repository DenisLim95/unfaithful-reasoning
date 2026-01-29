# 🎯 Complete Toolkit - All Commands

## 📦 One-Time Setup

```bash
# Copy everything to pod
scp /Users/denislim/workspace/mats-10.0/complete_tools.tar.gz root@<POD_IP>:/unfaithful-reasoning/

# On pod
cd /unfaithful-reasoning
tar -xzf complete_tools.tar.gz
source venv/bin/activate
```

---

## 🚀 Main Workflow

### 1. Generate Responses (with fixed prompt)
```bash
python test_probe_on_new_data.py --num-questions 200
# Time: ~35 minutes
# Generates: 400 responses, scores faithfulness, caches activations, tests probe
```

---

## 🔍 Inspection Tools

### View Specific Responses
```bash
# First 5 responses
python view_response.py

# Search for specific question
python view_response.py "900"
python view_response.py "795 larger than 900"

# View more
python view_response.py --limit 20
python view_response.py --all  # All 400 responses
```

**Use this to:**
- ✅ Check if model follows "Final Answer: Yes/No" format
- ✅ Verify extraction is working
- ✅ Debug specific wrong extractions

---

### Verify Data Quality
```bash
python verify_test_data.py
# Time: ~5 seconds
# Shows: Question correctness, extraction stats, faithfulness breakdown
```

**Use this to:**
- ✅ Confirm question pairs are correct
- ✅ Check extraction match rate
- ✅ See sample faithful/unfaithful pairs

---

### Analyze Unfaithful Responses
```bash
python analyze_unfaithful.py
# Time: ~2 seconds
# Shows: Detailed breakdown of why pairs are unfaithful
```

**Use this to:**
- ✅ See full unfaithful examples with complete responses
- ✅ Categorize: both wrong, q1 wrong, q2 wrong, extraction failed
- ✅ Understand if unfaithfulness is genuine or extraction bug

---

### Re-Score Existing Responses
```bash
python rescore_responses.py
# Time: ~5 seconds
# Use if: Extraction logic changed, want to re-score without regenerating
```

**Use this to:**
- ✅ Update faithfulness scores after fixing extraction
- ✅ Avoid regenerating responses (saves 30 minutes)

---

## 📊 Typical Workflow

### After Regeneration
```bash
# 1. Check overall stats
python verify_test_data.py | tail -50

# 2. Look at unfaithful breakdown
python analyze_unfaithful.py | head -100

# 3. Inspect specific responses
python view_response.py "900"
python view_response.py "795 larger than 900"

# 4. Check probe results (from main script output)
# Scroll up to see "TESTING EXISTING PROBE ON NEW DATA" section
```

---

## ✅ What to Check

### Good Data Indicators:
- ✅ Faithfulness rate: 60-75%
- ✅ Unknown extractions: <10%
- ✅ Unfaithful examples show genuine errors (not extraction bugs)
- ✅ Responses have "Final Answer: Yes/No" format

### Bad Data Indicators:
- ❌ Faithfulness rate: <40% or >90%
- ❌ Unknown extractions: >20%
- ❌ Unfaithful are mostly extraction failures
- ❌ Responses don't follow format

---

## 🎯 Quick Reference

| Task | Command | Time |
|------|---------|------|
| Generate all data | `python test_probe_on_new_data.py --num-questions 200` | 35 min |
| View responses | `python view_response.py "search"` | instant |
| Verify quality | `python verify_test_data.py` | 5 sec |
| Analyze unfaithful | `python analyze_unfaithful.py` | 2 sec |
| Re-score only | `python rescore_responses.py` | 5 sec |

---

## 📝 Example Session

```bash
# Generate data
python test_probe_on_new_data.py --num-questions 200

# Wait 35 minutes...

# Verify results look good
python verify_test_data.py | tail -50

# Expected:
# - Faithful: 120-150 (60-75%)
# - Unknown: <40 (<10%)

# Look at unfaithful details
python analyze_unfaithful.py | head -200

# Check specific examples
python view_response.py "900"

# If everything looks good:
# - Faithful rate ~70% ✓
# - Extraction working ✓
# - Check probe accuracy from main script output

# Probe ~65-70%? → Success!
# Probe ~50%? → Linear probes don't work (valid negative result)
```

---

## 🆘 Troubleshooting

### Problem: Extraction still wrong
```bash
# View specific case
python view_response.py "795 larger than 900"

# Check if model uses "Final Answer:" format
# If not, prompt needs adjustment
```

### Problem: Low faithfulness rate (<50%)
```bash
# Check extraction stats
python verify_test_data.py | grep -A 10 "EXTRACTION"

# If many "Unknown", extraction is failing
# If many wrong extractions, logic needs fixing
```

### Problem: High faithfulness rate (>80%)
```bash
# Check if questions are too easy
python view_response.py --limit 20

# Model might be getting everything right
```

---

## 📚 Documentation

- `FIXED_PROMPT_INSTRUCTIONS.md` - What was fixed and why
- `BUG_FIX_SUMMARY.md` - Overview of bugs discovered
- `VIEW_RESPONSE_GUIDE.md` - Detailed guide for view_response.py
- `VERIFICATION_INSTRUCTIONS.md` - How to verify data quality

---

**All tools are now on your pod! Use them to verify and debug.** 🎉

