# ⚡ QUICK START

## Copy to Pod
```bash
scp /Users/denislim/workspace/mats-10.0/complete_tools.tar.gz root@<POD_IP>:/unfaithful-reasoning/
```

## On Pod
```bash
cd /unfaithful-reasoning
tar -xzf complete_tools.tar.gz
source venv/bin/activate

# Generate everything (~35 min)
python test_probe_on_new_data.py --num-questions 200
```

## After Generation

### View specific response
```bash
python view_response.py "900"
```

**Shows:**
- Complete question
- Expected answer
- Extracted answer
- Full model response (nicely formatted)
- Whether extraction worked

### Quick examples
```bash
# First 5 responses
python view_response.py

# Search for "795 larger than 900"
python view_response.py "795 larger than 900"

# View 20 responses
python view_response.py --limit 20

# View ALL responses
python view_response.py --all
```

---

## Other Useful Commands

```bash
# Verify data quality
python verify_test_data.py

# Analyze unfaithful examples
python analyze_unfaithful.py

# Re-score without regenerating
python rescore_responses.py
```

---

## What Success Looks Like

### Good Data:
- Faithfulness: 60-75% ✓
- Unknown: <10% ✓
- Responses have "Final Answer: Yes/No" ✓

### Probe Works:
- Accuracy: 65-70% ✓
- AUC: >0.65 ✓

---

**That's it! Copy tarball, run main script, use view_response.py to inspect.** 🎯

