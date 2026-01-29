# Refactoring Complete - Quick Start Guide

## What We Built

We've refactored the codebase from a confusing **phase-based** structure to a clean **workflow-based** structure.

### 🎯 Core Philosophy

**Before:** "What's Phase 2? Do I run Phase 3 after Phase 1?"  
**After:** "I want to score faithfulness → run `03_score_faithfulness.py`"

## 🚀 Quick Start

### Option 1: Full Pipeline (From Scratch)

```bash
# Run everything from scratch
bash workflows/full_pipeline.sh 100

# This will:
# 1. Generate 100 question pairs
# 2. Get model responses (requires GPU)
# 3. Score faithfulness (your choice of method)
# 4. Compare methods (if you chose both)
```

### Option 2: Step-by-Step

```bash
# Step 1: Generate questions
python scripts/01_generate_questions.py --num-pairs 100

# Step 2: Generate responses (requires GPU)
python scripts/02_generate_responses.py \\
    --questions data/raw/questions.json

# Step 3: Score faithfulness
python scripts/03_score_faithfulness.py \\
    --responses data/responses/responses.jsonl \\
    --method llm-judge

# Step 4: Compare methods (optional)
python scripts/07_compare_methods.py \\
    --method1-scores results/scores_correctness.csv \\
    --method2-scores results/scores_llm_judge.csv
```

### Option 3: Just Evaluation (You Have Responses)

```bash
# Score your existing responses with different methods
bash workflows/faithfulness_only.sh
```

## 📁 New Structure

```
scripts/                           # What you run
  01_generate_questions.py        → Generate question pairs
  02_generate_responses.py        → Get model responses
  03_score_faithfulness.py        → Score with LLM judge or correctness
  07_compare_methods.py           → Compare two methods

src/                               # What scripts import
  faithfulness/                   → Evaluation logic
    llm_judge.py                  → LLM as judge
    answer_correctness.py         → Answer correctness
  models/                         → Model inference
    inference.py                  → Load model, generate, parse
  data/                           → Data utilities
    questions.py                  → Question generation

workflows/                         # Composite workflows
  full_pipeline.sh                → Run everything
  faithfulness_only.sh            → Just evaluation
```

## 🎓 Key Improvements

### 1. No More Duplication
**Before:** LLM judge code in 3+ places  
**After:** ONE implementation in `src/faithfulness/llm_judge.py`

### 2. Clear Naming
**Before:** `run_phase2.sh`, `test_probe_on_new_data.py`  
**After:** `03_score_faithfulness.py`, `full_pipeline.sh`

### 3. Modularity
```python
# Now anyone can do this:
from src.faithfulness import judge_reasoning_consistency
from src.models import load_model, generate_response

# Use in notebooks, scripts, experiments...
```

### 4. Better UX
**Before:**
```bash
python score_faithfulness_llm.py --use-llm-judge --openai-api-key ...
```

**After:**
```bash
python scripts/03_score_faithfulness.py --method llm-judge
# API key from environment variable
```

## 📊 What Works Now

### ✅ Fully Functional
- Question generation (script 01)
- Response generation (script 02)
- Faithfulness scoring - both methods (script 03)
- Method comparison (script 07)
- Full pipeline workflow
- Faithfulness-only workflow

### 🚧 TODO (Scripts 04-06)
- Activation caching
- Probe training
- Probe testing

## 🧪 Test It Out

### With Your Existing Data

You already have responses in `data/responses/test_responses.jsonl`. Test the new scripts:

```bash
# Score with answer correctness
python scripts/03_score_faithfulness.py \\
    --responses data/responses/test_responses.jsonl \\
    --method answer-correctness \\
    --output results/new_scores_correctness.csv

# Score with LLM judge
export OPENAI_API_KEY="sk-..."
python scripts/03_score_faithfulness.py \\
    --responses data/responses/test_responses.jsonl \\
    --method llm-judge \\
    --output results/new_scores_llm.csv

# Compare
python scripts/07_compare_methods.py \\
    --method1-scores results/new_scores_correctness.csv \\
    --method2-scores results/new_scores_llm.csv
```

## 📖 Documentation

- **REFACTORED_STRUCTURE_README.md** - Complete guide
- **REFACTORING_PROGRESS.md** - What's done, what's next
- **This file** - Quick start
- `python scripts/XX_script.py --help` - Per-script help

## 🔄 Migration from Old Code

| Old File | New File | Status |
|----------|----------|--------|
| `score_faithfulness_llm.py` | `scripts/03_score_faithfulness.py` | ✅ Replaced |
| `compare_scoring_methods.py` | `scripts/07_compare_methods.py` | ✅ Replaced |
| `test_probe_on_new_data.py` | Scripts 02, 04, 06 | 🚧 Partial |
| `src/data_generation/generate_questions_yesno.py` | `scripts/01_generate_questions.py` | ✅ Replaced |
| `src/inference/batch_inference.py` | `scripts/02_generate_responses.py` | ✅ Replaced |

**Old files still work**, but use the new scripts for clarity!

## 💡 Examples

### Experiment with Different Temperatures

```bash
# Generate responses at different temperatures
for temp in 0.3 0.6 0.9; do
    python scripts/02_generate_responses.py \\
        --questions data/raw/questions.json \\
        --temperature $temp \\
        --output "data/responses/responses_temp_${temp}.jsonl"
done

# Score all of them
for temp in 0.3 0.6 0.9; do
    python scripts/03_score_faithfulness.py \\
        --responses "data/responses/responses_temp_${temp}.jsonl" \\
        --method llm-judge \\
        --output "results/scores_temp_${temp}.csv"
done
```

### Use in Notebooks

```python
# In Jupyter notebook
from src.faithfulness import judge_reasoning_consistency
from src.models import load_model, generate_response

# Load model once
model, tokenizer = load_model()

# Generate and judge
response = generate_response(model, tokenizer, "Is 100 larger than 50?")
judgment = judge_reasoning_consistency(
    question="Is 100 larger than 50?",
    reasoning=response,
    answer="Yes"
)

print(f"Consistent: {judgment['is_consistent']}")
print(f"Confidence: {judgment['confidence']}")
```

## 🎯 Next Steps

1. **Try the new scripts** with your existing data
2. **Use workflows** for common tasks
3. **Import libraries** in notebooks/analysis
4. **Finish remaining scripts** (04-06) when needed

## ❓ Questions?

```bash
# Get help for any script
python scripts/01_generate_questions.py --help
python scripts/02_generate_responses.py --help
python scripts/03_score_faithfulness.py --help
python scripts/07_compare_methods.py --help

# Check structure
cat REFACTORED_STRUCTURE_README.md

# Check progress
cat REFACTORING_PROGRESS.md
```

## 🎉 Benefits

### For Research
- ✅ Easy to iterate on evaluation methods
- ✅ Reproducible workflows
- ✅ Clear dependencies between steps

### For Development
- ✅ No code duplication
- ✅ Easy to add new features
- ✅ Testable components

### For Collaboration
- ✅ Self-documenting scripts
- ✅ Clear entry points
- ✅ Consistent interfaces

---

**The refactoring is 60% complete and fully usable for the core faithfulness evaluation workflow!**

