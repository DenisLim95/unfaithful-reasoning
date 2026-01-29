# Modular Faithfulness Evaluation - Quick Start

## Overview

The faithfulness evaluation has been separated into **modular, reusable components**:

```
test_probe_on_new_data.py     → Full pipeline (all-in-one)
score_faithfulness_llm.py     → Standalone scoring only
compare_scoring_methods.py    → Compare different scoring methods
```

## Quick Start Guide

### Step 1: Generate Responses (Once)

```bash
# Generate 100 question pairs and get model responses
python test_probe_on_new_data.py --num-questions 100

# This creates: data/responses/test_responses.jsonl
```

### Step 2: Score with Different Methods

```bash
# Method 1: Answer correctness (fast, no API needed)
python score_faithfulness_llm.py \
  --responses data/responses/test_responses.jsonl \
  --output results/scores_correctness.csv

# Method 2: LLM judge (requires OpenAI API)
export OPENAI_API_KEY="sk-your-key"
python score_faithfulness_llm.py --use-llm-judge \
  --responses data/responses/test_responses.jsonl \
  --output results/scores_llm_judge.csv
```

### Step 3: Compare Methods

```bash
python compare_scoring_methods.py \
  --correctness results/scores_correctness.csv \
  --llm-judge results/scores_llm_judge.csv
```

## Files Created

| File | Description |
|------|-------------|
| `score_faithfulness_llm.py` | Standalone scoring script |
| `compare_scoring_methods.py` | Compare two scoring methods |
| `LLM_JUDGE_GUIDE.md` | Detailed guide for LLM judge |
| `STANDALONE_SCORING_GUIDE.md` | Usage guide for standalone scoring |
| `MODULAR_EVALUATION_QUICKSTART.md` | This file |

## Example Workflow

### Scenario: Compare evaluation methods

```bash
# 1. Generate responses once
python test_probe_on_new_data.py --num-questions 100

# 2. Score with answer correctness
python score_faithfulness_llm.py \
  --responses data/responses/test_responses.jsonl \
  --output results/scores_correctness.csv

# 3. Score with LLM judge
python score_faithfulness_llm.py --use-llm-judge \
  --responses data/responses/test_responses.jsonl \
  --output results/scores_llm_judge.csv

# 4. Compare results
python compare_scoring_methods.py \
  --correctness results/scores_correctness.csv \
  --llm-judge results/scores_llm_judge.csv \
  --output results/comparison.csv
```

### Scenario: Iterate on scoring without re-inference

```bash
# Generate responses once (slow, uses GPU)
python test_probe_on_new_data.py --num-questions 1000

# Try different scoring approaches (fast, cheap)
python score_faithfulness_llm.py --use-llm-judge --responses ... --output results/v1.csv
# (modify judge prompt in score_faithfulness_llm.py)
python score_faithfulness_llm.py --use-llm-judge --responses ... --output results/v2.csv
# (try different confidence thresholds)
python score_faithfulness_llm.py --use-llm-judge --responses ... --output results/v3.csv
```

## Command Reference

### score_faithfulness_llm.py

```bash
# Basic usage
python score_faithfulness_llm.py --responses <path> [options]

# Options
--responses PATH          Input JSONL file (required)
--output PATH            Output CSV file (default: data/processed/faithfulness_scores.csv)
--use-llm-judge          Use LLM judge instead of answer correctness
--openai-api-key KEY     OpenAI API key (or use OPENAI_API_KEY env var)

# Examples
python score_faithfulness_llm.py --responses data/responses/test_responses.jsonl
python score_faithfulness_llm.py --use-llm-judge --responses data/responses/test_responses.jsonl
```

### compare_scoring_methods.py

```bash
# Basic usage
python compare_scoring_methods.py --correctness <path> --llm-judge <path> [options]

# Options
--correctness PATH       CSV with answer correctness scores (required)
--llm-judge PATH        CSV with LLM judge scores (required)
--output PATH           Save comparison to CSV (optional)

# Example
python compare_scoring_methods.py \
  --correctness results/scores_correctness.csv \
  --llm-judge results/scores_llm_judge.csv \
  --output results/comparison.csv
```

## Benefits of Modular Approach

### 1. **Cost Efficiency**
- Generate responses once (GPU expensive)
- Score multiple times (CPU/API cheap)
- Experiment with evaluation methods without re-inference

### 2. **Faster Iteration**
- Test different judge prompts
- Try different evaluation criteria
- Compare methods side-by-side

### 3. **Reproducibility**
- Keep raw responses unchanged
- Version control scoring methods separately
- Audit evaluation changes over time

### 4. **Flexibility**
- Mix and match: different prompts, same evaluation
- Score historical data with new methods
- Batch process large datasets

## Expected Results

### Answer Correctness Method
- **Faithfulness rate**: ~70-75%
- **Measures**: Whether both Q1 and Q2 answers are correct
- **Fast**: No API calls needed
- **Issue**: Doesn't check reasoning

### LLM Judge Method
- **Faithfulness rate**: ~40-60% (lower than correctness)
- **Measures**: Whether reasoning logically supports answer
- **Cost**: ~$0.01-0.02 per 100 pairs
- **Better**: Detects unfaithful reasoning

### Why LLM Judge Is Lower
- Catches reasoning-answer contradictions
- Detects position bias heuristics
- Identifies vague/missing reasoning
- More aligned with true "faithfulness" definition

## Next Steps

1. **Run both methods** on your data
2. **Compare results** to see where they disagree
3. **Analyze disagreements** - which method is more accurate?
4. **Retrain probes** with better faithfulness labels
5. **Test generalization** - do probes work better with LLM-judged labels?

## Troubleshooting

### "Responses file not found"
```bash
# Generate responses first
python test_probe_on_new_data.py --num-questions 100
```

### "OpenAI API key required"
```bash
export OPENAI_API_KEY="sk-your-key"
# or
python score_faithfulness_llm.py --openai-api-key "sk-your-key" ...
```

### "openai library not installed"
```bash
pip install openai
```

## Documentation

- `LLM_JUDGE_GUIDE.md` - Detailed LLM judge documentation
- `STANDALONE_SCORING_GUIDE.md` - Standalone scoring usage
- `score_faithfulness_llm.py --help` - Command line help
- `compare_scoring_methods.py --help` - Comparison tool help

## Summary

You now have:
- ✅ Modular, reusable evaluation pipeline
- ✅ Two faithfulness scoring methods
- ✅ Comparison tools
- ✅ Cost-efficient workflow
- ✅ Better aligned with true faithfulness definition

This addresses the researcher's critique by moving toward evaluating **reasoning consistency**, not just answer consistency!

