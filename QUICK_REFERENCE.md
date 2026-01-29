# Quick Reference Guide

## All Scripts at a Glance

### Script 01: Generate Questions
```bash
python scripts/01_generate_questions.py --num-pairs 100 --output data/raw/questions.json
```
**Purpose**: Generate paired comparison questions  
**Output**: `data/raw/questions.json`  
**Dependencies**: None

### Script 02: Generate Responses
```bash
python scripts/02_generate_responses.py --questions data/raw/questions.json
```
**Purpose**: Get model responses for questions  
**Output**: `data/responses/responses.jsonl`  
**Dependencies**: Script 01, GPU, transformers  
**Note**: Uses 2048 max tokens (good for verbose models)

### Script 03: Score Faithfulness
```bash
# Default: LLM judge
export OPENAI_API_KEY="sk-..."
python scripts/03_score_faithfulness.py --responses data/responses/responses.jsonl

# Or answer correctness
python scripts/03_score_faithfulness.py \
    --responses data/responses/responses.jsonl \
    --method answer-correctness
```
**Purpose**: Score faithfulness (LLM judge or answer correctness)  
**Output**: `data/processed/faithfulness_scores.csv`  
**Dependencies**: Script 02, OpenAI API key (for LLM judge)

### Script 04: Cache Activations
```bash
python scripts/04_cache_activations.py \
    --responses data/responses/responses.jsonl \
    --scores data/processed/faithfulness_scores.csv
```
**Purpose**: Cache neural activations  
**Output**: `data/activations/layer_{N}_activations.pt`  
**Dependencies**: Scripts 02, 03, GPU, transformers

### Script 05: Train Probes
```bash
python scripts/05_train_probes.py
```
**Purpose**: Train linear probes  
**Output**: `results/probe_results/all_probe_results.pt`  
**Dependencies**: Script 04, sklearn

### Script 06: Test Probes
```bash
python scripts/06_test_probes.py --test-activations data/test_activations
```
**Purpose**: Test probe generalization  
**Output**: Console output with comparison  
**Dependencies**: Script 05, test activations

### Script 07: Compare Methods
```bash
python scripts/07_compare_methods.py \
    --method1-scores results/scores_correctness.csv \
    --method2-scores results/scores_llm.csv
```
**Purpose**: Compare two faithfulness scoring methods  
**Output**: Console output + optional CSV  
**Dependencies**: Two score files from script 03

## Workflows

### Full Pipeline
```bash
bash workflows/full_pipeline.sh 100
```
Runs: 01 → 02 → 03 → (optional) 07

### Faithfulness Only
```bash
bash workflows/faithfulness_only.sh
```
Runs: 03 with both methods → 07

### Probe Training
```bash
bash workflows/probe_training.sh
```
Runs: 04 (validate) → 05 → 06

## Common Patterns

### Generate New Data
```bash
python scripts/01_generate_questions.py --num-pairs 50
python scripts/02_generate_responses.py --questions data/raw/questions.json
```

### Re-score Existing Data
```bash
python scripts/03_score_faithfulness.py \
    --responses data/responses/test_responses.jsonl \
    --method llm-judge
```

### Validate Caches
```bash
python scripts/04_cache_activations.py --validate
```

### Get Help
```bash
python scripts/01_generate_questions.py --help
python scripts/02_generate_responses.py --help
python scripts/03_score_faithfulness.py --help
# ... etc
```

## File Locations

**Input Data:**
- Questions: `data/raw/questions.json`
- Responses: `data/responses/responses.jsonl`

**Processed Data:**
- Scores: `data/processed/faithfulness_scores.csv`
- Activations: `data/activations/layer_{N}_activations.pt`

**Results:**
- Probes: `results/probe_results/all_probe_results.pt`
- Comparisons: `results/method_comparison.csv`

## Import in Python

```python
# Faithfulness evaluation
from src.faithfulness import judge_reasoning_consistency, score_answer_correctness

# Model inference
from src.models import load_model, generate_response, parse_response

# Data utilities
from src.data import generate_question_set, cache_activations_for_responses

# Probes
from src.probes import train_probe, evaluate_probe
```

## Quick Start

**Test with existing data:**
```bash
python scripts/03_score_faithfulness.py \
    --responses data/responses/test_responses.jsonl
```

**Full pipeline (10 questions):**
```bash
bash workflows/full_pipeline.sh 10
```

**Documentation:**
- `REFACTORING_COMPLETE.md` - Overview
- `REFACTORED_STRUCTURE_README.md` - Detailed guide
- `REFACTORING_QUICKSTART.md` - Quick start
- `python scripts/XX_script.py --help` - Per-script help

