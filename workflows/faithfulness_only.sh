#!/bin/bash
#
# Workflow: Faithfulness Evaluation Only
#
# This workflow focuses on faithfulness evaluation experiments.
# It assumes you already have model responses generated.
#
# Usage:
#   bash workflows/faithfulness_only.sh
#

set -e  # Exit on error

echo "============================================================"
echo "WORKFLOW: FAITHFULNESS EVALUATION"
echo "============================================================"

# Configuration
RESPONSES_FILE="data/responses/responses.jsonl"
OUTPUT_DIR="results"

# Check if responses exist
if [ ! -f "$RESPONSES_FILE" ]; then
    echo "❌ Error: Responses file not found: $RESPONSES_FILE"
    echo ""
    echo "Please run the full pipeline first:"
    echo "  bash workflows/full_pipeline.sh"
    echo ""
    echo "Or generate responses:"
    echo "  python scripts/02_generate_responses.py --questions data/raw/questions.json"
    exit 1
fi

echo ""
echo "Found responses file: $RESPONSES_FILE"
echo ""

# Step 1: Score with answer correctness
echo "============================================================"
echo "STEP 1: Score with Answer Correctness"
echo "============================================================"
python scripts/03_score_faithfulness.py \
    --responses "$RESPONSES_FILE" \
    --method answer-correctness \
    --output "$OUTPUT_DIR/scores_correctness.csv"

echo ""

# Step 2: Score with LLM judge
echo "============================================================"
echo "STEP 2: Score with LLM Judge"
echo "============================================================"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo ""
    echo "To use LLM judge, set your API key:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    echo "Skipping LLM judge scoring..."
    echo ""
else
    # Estimate cost first
    python scripts/03_score_faithfulness.py \
        --responses "$RESPONSES_FILE" \
        --method llm-judge \
        --estimate-cost
    
    echo ""
    read -p "Continue with LLM judge? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/03_score_faithfulness.py \
            --responses "$RESPONSES_FILE" \
            --method llm-judge \
            --output "$OUTPUT_DIR/scores_llm_judge.csv"
        
        echo ""
        
        # Step 3: Compare methods
        echo "============================================================"
        echo "STEP 3: Compare Methods"
        echo "============================================================"
        python scripts/07_compare_methods.py \
            --method1-scores "$OUTPUT_DIR/scores_correctness.csv" \
            --method1-name "Answer Correctness" \
            --method2-scores "$OUTPUT_DIR/scores_llm_judge.csv" \
            --method2-name "LLM Judge" \
            --output "$OUTPUT_DIR/method_comparison.csv"
    else
        echo "Skipped LLM judge scoring"
    fi
fi

echo ""
echo "============================================================"
echo "WORKFLOW COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Review your results:"
echo "  cat $OUTPUT_DIR/scores_correctness.csv"
if [ -f "$OUTPUT_DIR/scores_llm_judge.csv" ]; then
    echo "  cat $OUTPUT_DIR/scores_llm_judge.csv"
    echo "  cat $OUTPUT_DIR/method_comparison.csv"
fi
echo ""

