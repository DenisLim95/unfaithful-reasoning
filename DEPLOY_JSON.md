# Deploy JSON Output Implementation

## ✅ All 6 Requirements Implemented

### 1. **Replaced `<think>` tags with JSON**
```json
{"answer": "Yes", "reasoning": "1. Step\\n2. Step\\n3. Step"}
```

### 2. **JSON-only output with strict schema**
- System prompt: "You output ONLY valid JSON"
- Exact schema specified
- No markdown, no extra text

### 3. **Fallback format available**
If JSON fails consistently, has fallback heuristic extraction

### 4. **Uses chat template properly (Qwen2)**
```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### 5. **Output validation + retry**
- Strict JSON parsing
- Answer must be "Yes" or "No"
- Retry once with format fix prompt
- Fallback extraction on double failure

### 6. **Cleanup & JSON extraction**
- Removes markdown code blocks
- Extracts JSON from surrounded text
- Finds first `{` and last `}`

---

## 🚀 Deploy Now

```bash
# Copy to pod
scp /Users/denislim/workspace/mats-10.0/json_output_final.tar.gz root@<POD_IP>:/unfaithful-reasoning/

# On pod
cd /unfaithful-reasoning
tar -xzf json_output_final.tar.gz
source venv/bin/activate

# Generate with JSON format
python test_probe_on_new_data.py --num-questions 200
```

**Time:** ~35 minutes

---

## 📊 What to Expect

### During Generation:
```
Generating responses: 100%|██████| 200/200 [28:30<00:00, 8.55s/it]

✓ Saved 400 responses to data/responses/test_responses.jsonl
  JSON validation: 382/400 (95.5%) valid
```

**✅ 90%+ valid = Success!**  
**⚠️ 80-90% valid = Acceptable (some fallbacks)**  
**❌ <80% valid = Problem (model not following)**

---

## 🔍 Verify Results

### 1. Check JSON compliance
```bash
python test_probe_on_new_data.py --num-questions 200 2>&1 | grep "JSON validation"
```

Look for:
```
JSON validation: 380/400 (95.0%) valid
```

### 2. Inspect sample responses
```bash
python view_response.py --limit 5
```

**Good:**
```
Response:
{"answer": "Yes", "reasoning": "1. 900 has 9 in hundreds\n2. 795 has 7\n3. 9 > 7"}
```

**Bad:**
```
Response:
Sure! Let me help you with that.
```json
{"answer": "yes"}
`` `
```

### 3. Check faithfulness rate
```bash
python verify_test_data.py | grep "Faithfulness Distribution"
```

**Expected:**
```
Faithfulness Distribution:
  Faithful: 140 (70.0%)
  Unfaithful: 60 (30.0%)
```

---

## 🎯 Expected Improvements

### Before (Free Text):
- ❌ Model ignores format
- ❌ Says "Yes, X is smaller..." (contradictions)
- ❌ Extraction match rate: 60-70%
- ❌ Faithfulness: 35% (wrong due to extraction bugs)

### After (JSON):
- ✅ Clean JSON structure
- ✅ No contradictions
- ✅ Extraction match rate: 95%+
- ✅ Faithfulness: 60-75% (realistic)

---

## 📈 Success Criteria

After running, you should see:

1. ✅ **JSON validation ≥90%**
2. ✅ **Extraction match rate ≥90%**
3. ✅ **Faithful rate 60-75%**
4. ✅ **Balanced test set (50/50)**
5. ✅ **Probe accuracy >50%** (if probe works)

---

## 🔧 If JSON Compliance is Low (<80%)

### Option A: Check sample responses
```bash
python view_response.py --limit 10
```

If you see markdown code blocks or extra text, the cleanup should handle it.

### Option B: More examples in prompt
Add few-shot examples:

```python
"Examples:\n"
'Question: Is 100 > 50?\n'
'{"answer": "Yes", "reasoning": "1. 100 is larger than 50"}\n\n'
'Question: Is 30 > 40?\n'
'{"answer": "No", "reasoning": "1. 30 is less than 40"}\n\n'
```

### Option C: Use fallback format
If JSON doesn't work at all, modify prompt to:

```
Reasoning:
1. [step]
2. [step]
3. [step]

Final Answer: Yes
```

---

## 📝 Quick Commands

```bash
# Full run
python test_probe_on_new_data.py --num-questions 200

# Just verify existing
python verify_test_data.py

# View specific responses
python view_response.py "900"

# Analyze unfaithful
python analyze_unfaithful.py
```

---

## 🎯 Bottom Line

**This JSON approach is the RIGHT way to do structured output.**

Benefits:
- ✅ Parseable
- ✅ Validated
- ✅ Retry logic
- ✅ Fallback extraction
- ✅ Clean data

**Deploy and run - this should finally give you reliable results!** 🚀

---

## 📚 Documentation

- `JSON_OUTPUT_README.md` - Full implementation details
- `FINAL_FIX.md` - What changed and why
- `TOOLKIT_GUIDE.md` - All available commands

**Everything you need is in the tarball!**



