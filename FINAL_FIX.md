# Final Fix: Chat Template + Better Verification

## 🎯 What's Fixed

### 1. **Chat Template (CRITICAL)**
DeepSeek-R1 is a **chat model**, not a base model. We now use proper chat formatting:

**Before:**
```python
prompt = "You are a helpful AI assistant..."
inputs = tokenizer(prompt, ...)
```

**After:**
```python
messages = [
    {"role": "system", "content": "You must follow exact format."},
    {"role": "user", "content": "Question: ... Format: <think>...</think> Final Answer: Yes/No"}
]
prompt = tokenizer.apply_chat_template(messages, ...)  # ← KEY!
```

**Also:**
- Lower temperature (0.1 instead of 0.6) = more deterministic
- Clearer instructions: "You MUST format..."
- Tells model not to add extra text

### 2. **Better Verification Output**
`verify_test_data.py` now shows:
- **Last 250 chars** of each response (not just excerpts)
- **Format check**: ✓/✗ for `<think>` tags and `Final Answer:`
- **Clearer labeling**: "Response END" instead of "Response excerpt"

---

## 📦 Deploy

```bash
# Copy to pod
scp /Users/denislim/workspace/mats-10.0/final_fixes.tar.gz root@<POD_IP>:/unfaithful-reasoning/

# On pod
cd /unfaithful-reasoning
tar -xzf final_fixes.tar.gz
source venv/bin/activate

# Regenerate with proper chat format
python test_probe_on_new_data.py --num-questions 200
```

**Time: ~35 minutes**

---

## 🔍 Verify Format After Generation

```bash
python verify_test_data.py | grep -A 5 "Response END"
```

**Look for:**
```
Response END (last 250 chars):
└─> ...<think>
900 has 9 in hundreds, 795 has 7.
</think>

Final Answer: Yes

Format: ✓ <think> tags | ✓ Final Answer:
```

**Good signs:**
- ✅ Responses end with "Final Answer: Yes" or "Final Answer: No"
- ✅ Both checkmarks are ✓
- ✅ No extra text after "Final Answer:"

**Bad signs:**
- ❌ "Format: ✗ <think> tags | ✗ Final Answer:"
- ❌ Responses end with rambling text
- ❌ Model says "Yes, X is smaller..." (contradictions)

---

## 📊 What Success Looks Like

### Response Format:
```
<think>
To compare 900 and 795:
- 900 has 9 in hundreds place
- 795 has 7 in hundreds place
- 9 > 7, so 900 is larger
</think>

Final Answer: Yes
```

### Verification Output:
```
✓ Loaded 400 responses

Extraction Match Rate: 85-95% (high!)

✓ CORRECT EXTRACTION (showing 3 of 380):
  Format: ✓ <think> tags | ✓ Final Answer:

Faithfulness: 120-150 pairs (60-75%)
```

### Probe Results:
```
layer_12:
  Test samples: 100 (50 faithful, 50 unfaithful)
  Accuracy: 65-70%  ← Success!
  AUC-ROC: 0.65-0.75
```

---

## 🎯 Expected Outcomes

**If chat template works:**
- Model follows format instructions
- Clean "Final Answer: Yes/No" at end
- Extraction works reliably (85%+ match rate)
- Balanced faithful/unfaithful split (60-75% faithful)
- Probe can finally be tested on GOOD data

**If still broken:**
- Model might not support chat templates well
- May need even stronger format constraints
- Could try few-shot examples

---

## Quick Commands

```bash
# Generate
python test_probe_on_new_data.py --num-questions 200

# Verify format
python verify_test_data.py | head -150

# Check specific responses
python view_response.py --limit 5
```

---

**This should be the REAL fix!** 🎯

Using the model's actual chat interface should make it follow instructions properly.
