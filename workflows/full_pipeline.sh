#!/bin/bash
#
# Workflow: Full Pipeline
#
# Run the complete faithfulness evaluation pipeline from start to finish.
#
# Usage:
#   bash workflows/full_pipeline.sh [num_pairs]
#
# Example:
#   bash workflows/full_pipeline.sh 100
#

set -e  # Exit on error

# Configuration
NUM_PAIRS=${1:-50}  # Default: 50 pairs
QUESTIONS_FILE="data/raw/questions.json"
RESPONSES_FILE="data/responses/responses.jsonl"
SCORES_FILE="data/processed/faithfulness_scores.csv"

echo "============================================================"
echo "FULL PIPELINE: FAITHFULNESS EVALUATION"
echo "============================================================"
echo "Generating $NUM_PAIRS question pairs"
echo ""

# Step 1: Generate questions
echo "============================================================"
echo "STEP 1: Generate Questions"
echo "============================================================"
python scripts/01_generate_questions.py \
    --num-pairs "$NUM_PAIRS" \
    --output "$QUESTIONS_FILE" \
    --validate

echo ""

# Step 2: Generate responses
echo "============================================================"
echo "STEP 2: Generate Model Responses"
echo "============================================================"
echo "⚠️  This step requires GPU and will take time..."
echo ""
python scripts/02_generate_responses.py \
    --questions "$QUESTIONS_FILE" \
    --output "$RESPONSES_FILE"

echo ""

# Step 3: Score faithfulness
echo "============================================================"
echo "STEP 3: Score Faithfulness"
echo "============================================================"
echo "Choose scoring method:"
echo "  1) Answer correctness (fast, no API)"
echo "  2) LLM judge (requires OpenAI API key)"
echo "  3) Both"
echo ""
read -p "Enter choice (1/2/3): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[13]$ ]]; then
    echo "Scoring with answer correctness..."
    python scripts/03_score_faithfulness.py \
        --responses "$RESPONSES_FILE" \
        --method answer-correctness \
        --output "results/scores_correctness.csv"
fi

if [[ $REPLY =~ ^[23]$ ]]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ Error: OPENAI_API_KEY not set"
        echo "Set your API key:"
        echo "  export OPENAI_API_KEY='sk-...'"
        exit 1
    fi
    
    echo "Estimating cost for LLM judge..."
    python scripts/03_score_faithfulness.py \
        --responses "$RESPONSES_FILE" \
        --method llm-judge \
        --estimate-cost
    
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Scoring with LLM judge..."
        python scripts/03_score_faithfulness.py \
            --responses "$RESPONSES_FILE" \
            --method llm-judge \
            --output "results/scores_llm_judge.csv"
    fi
fi

# Step 4: Compare if both methods were used
if [[ -f "results/scores_correctness.csv" && -f "results/scores_llm_judge.csv" ]]; then
    echo ""
    echo "============================================================"
    echo "STEP 4: Compare Methods"
    echo "============================================================"
    python scripts/07_compare_methods.py \
        --method1-scores "results/scores_correctness.csv" \
        --method1-name "Answer Correctness" \
        --method2-scores "results/scores_llm_judge.csv" \
        --method2-name "LLM Judge" \
        --output "results/method_comparison.csv"
fi

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  Questions: $QUESTIONS_FILE"
echo "  Responses: $RESPONSES_FILE"
echo "  Scores: results/scores_*.csv"
echo ""
echo "Next steps:"
echo "  - Review scores: cat results/scores_*.csv"
echo "  - Cache activations: python scripts/04_cache_activations.py"
echo "  - Train probes: python scripts/05_train_probes.py"
echo ""

