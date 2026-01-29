# Refactored Codebase - Workflow-Based Structure

## Overview

This codebase has been refactored from a **phase-based** structure to a **workflow-based** structure for better clarity and modularity.

### Old Structure (Phase-based) ❌
```
run_phase2.sh           # What does Phase 2 mean?
run_phase3.sh           # Do I need Phase 1 first?
PHASE2_README.md        # Confusing naming
```

### New Structure (Workflow-based) ✅
```
scripts/
  01_generate_questions.py      # Clear what it does
  02_generate_responses.py      # Clear what it does
  03_score_faithfulness.py      # Clear what it does
  ...

src/
  faithfulness/                 # Shared utilities
  models/
  probes/
  ...
```

## Directory Structure

```
scripts/                          # Executable workflow scripts (numbered)
  01_generate_questions.py        # Generate question pairs
  02_generate_responses.py        # Run model inference
  03_score_faithfulness.py        # Score with LLM judge or correctness
  04_cache_activations.py         # Cache neural activations
  05_train_probes.py              # Train linear probes
  06_test_probes.py               # Test probe generalization
  07_compare_methods.py           # Compare scoring methods

src/                              # Reusable library code
  faithfulness/                   # Faithfulness evaluation
    __init__.py
    llm_judge.py                  # LLM as judge logic
    answer_correctness.py         # Answer correctness scoring
  models/                         # Model loading and inference
    __init__.py
    inference.py
  probes/                         # Probe training and evaluation
    __init__.py
    train.py
    evaluate.py
  data/                           # Data utilities
    __init__.py
    questions.py                  # Question generation
    activations.py                # Activation caching
  utils/                          # General utilities
    __init__.py
    formats.py                    # Data format utilities
    contracts.py                  # Data contracts/validation

workflows/                        # Composite workflows (bash scripts)
  full_pipeline.sh                # Run entire pipeline
  faithfulness_only.sh            # Just evaluation experiments
  probe_training.sh               # Just probe workflow

data/                             # Data files
  raw/                            # Raw data (questions)
  responses/                      # Model responses
  processed/                      # Processed data (scores)
  activations/                    # Cached activations
  test_activations/               # Test set activations

results/                          # Results and outputs
  probe_results/                  # Trained probes
  figures/                        # Visualizations
  comparisons/                    # Method comparisons
```

## Quick Start

### Option 1: Run Individual Scripts

```bash
# Step 1: Generate questions
python scripts/01_generate_questions.py --num-pairs 100

# Step 2: Generate responses
python scripts/02_generate_responses.py --questions data/raw/questions.json

# Step 3: Score faithfulness (choose method)
# Method A: Answer correctness (fast, no API)
python scripts/03_score_faithfulness.py \\
    --responses data/responses/responses.jsonl \\
    --method answer-correctness

# Method B: LLM judge (requires OpenAI API)
export OPENAI_API_KEY="sk-..."
python scripts/03_score_faithfulness.py \\
    --responses data/responses/responses.jsonl \\
    --method llm-judge

# Step 4: Compare methods (optional)
python scripts/07_compare_methods.py \\
    --method1-scores results/scores_correctness.csv \\
    --method1-name "Answer Correctness" \\
    --method2-scores results/scores_llm_judge.csv \\
    --method2-name "LLM Judge"
```

### Option 2: Run Workflow Scripts

```bash
# Run full pipeline
bash workflows/full_pipeline.sh

# Run just evaluation experiments
bash workflows/faithfulness_only.sh
```

## Script Details

### 03_score_faithfulness.py

**Purpose:** Score model responses for faithfulness

**Input:** `responses.jsonl` (from script 02)

**Output:** `faithfulness_scores.csv`

**Methods:**
- `answer-correctness`: Checks if both Q1 and Q2 answers are correct
- `llm-judge`: Uses GPT-4o-mini to check if reasoning supports answer

**Usage:**
```bash
# Answer correctness (fast, free)
python scripts/03_score_faithfulness.py \\
    --responses data/responses/responses.jsonl \\
    --method answer-correctness \\
    --output results/scores_correctness.csv

# LLM judge (requires API key)
python scripts/03_score_faithfulness.py \\
    --responses data/responses/responses.jsonl \\
    --method llm-judge \\
    --output results/scores_llm_judge.csv \\
    --openai-api-key "sk-..."

# Estimate cost before running
python scripts/03_score_faithfulness.py \\
    --responses data/responses/responses.jsonl \\
    --method llm-judge \\
    --estimate-cost
```

### 07_compare_methods.py

**Purpose:** Compare two scoring methods

**Input:** Two CSV files from script 03

**Output:** Comparison analysis + optional CSV

**Usage:**
```bash
python scripts/07_compare_methods.py \\
    --method1-scores results/scores_correctness.csv \\
    --method1-name "Answer Correctness" \\
    --method2-scores results/scores_llm_judge.csv \\
    --method2-name "LLM Judge" \\
    --output results/comparison.csv
```

## Using Shared Libraries

The `src/` directory contains reusable code that all scripts can import:

```python
# Example: Use LLM judge in your own script
from src.faithfulness import judge_reasoning_consistency

result = judge_reasoning_consistency(
    question="Is 900 larger than 795?",
    reasoning="900 has 9 in hundreds place, 795 has 7...",
    answer="Yes"
)

print(result['is_consistent'])  # True/False
print(result['confidence'])     # 'high', 'medium', 'low'
print(result['explanation'])    # Judge's reasoning
```

## Migration from Old Structure

### Old Files → New Files

| Old File | New File | Status |
|----------|----------|--------|
| `test_probe_on_new_data.py` | `scripts/06_test_probes.py` | ✅ Refactored |
| `score_faithfulness_llm.py` | `scripts/03_score_faithfulness.py` | ✅ Refactored |
| `compare_scoring_methods.py` | `scripts/07_compare_methods.py` | ✅ Refactored |
| `src/evaluation/score_faithfulness.py` | `src/faithfulness/answer_correctness.py` | ✅ Refactored |
| Phase-based scripts | Workflow scripts | 🚧 TODO |

### Old Phases → New Scripts

| Old Phase | New Script(s) |
|-----------|---------------|
| Phase 1 | `01_generate_questions.py` |
| Phase 2 (inference) | `02_generate_responses.py` |
| Phase 2 (scoring) | `03_score_faithfulness.py` |
| Phase 3 (caching) | `04_cache_activations.py` |
| Phase 3 (probes) | `05_train_probes.py`, `06_test_probes.py` |

## Benefits of New Structure

### ✅ **Clarity**
- Script names describe what they do
- No need to remember "Phase X"
- Self-documenting workflow

### ✅ **Modularity**
- Shared code in `src/`
- No duplication between scripts
- Easy to reuse components

### ✅ **Flexibility**
- Run scripts in any order (if dependencies met)
- Mix and match workflows
- Easy to add new scripts

### ✅ **Maintainability**
- One place for each function
- Easy to update shared logic
- Clear dependencies

## Development Guidelines

### Adding a New Script

1. Create in `scripts/` with descriptive name
2. Use next available number if part of main workflow
3. Add docstring with:
   - Purpose
   - Usage examples
   - Input/output formats
   - Dependencies
4. Import from `src/` for shared functionality
5. Update this README

### Adding Shared Functionality

1. Identify which module it belongs to:
   - `src/faithfulness/` - evaluation logic
   - `src/models/` - model inference
   - `src/probes/` - probe training/testing
   - `src/data/` - data processing
   - `src/utils/` - general utilities

2. Add function with clear docstring
3. Export in `__init__.py`
4. Use in scripts via import

### Creating a Workflow

1. Create bash script in `workflows/`
2. Call numbered scripts in sequence
3. Add error checking
4. Document purpose in header comment

## Next Steps

- [ ] Refactor remaining phase-based scripts
- [ ] Create workflow scripts in `workflows/`
- [ ] Migrate old documentation
- [ ] Add unit tests
- [ ] Create integration tests

## Questions?

See individual script help:
```bash
python scripts/03_score_faithfulness.py --help
python scripts/07_compare_methods.py --help
```

Or check the old documentation (being migrated):
- `PHASE2_README.md` → Being replaced by this file
- `PHASE3_README.md` → Being replaced by this file

