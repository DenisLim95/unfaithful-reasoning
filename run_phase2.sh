#!/bin/bash
# Phase 2 Execution Script
# Run all Phase 2 tasks in sequence

set -e  # Exit on error

echo "========================================"
echo "PHASE 2: Faithfulness Evaluation"
echo "========================================"

# Check Phase 1 dependency
echo ""
echo "[0/4] Checking Phase 1 dependency..."
if [ ! -f "data/raw/question_pairs.json" ]; then
    echo "❌ Phase 1 not complete: data/raw/question_pairs.json not found"
    echo "Run Phase 1 first: python src/data_generation/generate_questions.py"
    exit 1
fi
echo "✓ Phase 1 data found"

# Task 2.1: Run inference (THIS TAKES 2-3 HOURS)
echo ""
echo "[1/4] Running model inference (this will take 2-3 hours)..."
echo "Starting at: $(date)"
python src/inference/batch_inference.py
echo "Finished at: $(date)"

# Task 2.2 + 2.3: Score faithfulness (includes answer extraction)
echo ""
echo "[2/4] Scoring faithfulness..."
python src/evaluation/score_faithfulness_yesno.py

# Task 2.4: Validate Phase 2
echo ""
echo "[3/4] Validating Phase 2 contracts..."
python tests/validate_phase2.py

# Success
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ PHASE 2 COMPLETE"
    echo "========================================"
    echo ""
    echo "Deliverables:"
    echo "  • data/responses/model_1.5B_responses.jsonl (100 responses)"
    echo "  • data/processed/faithfulness_scores.csv (50 scores)"
    echo ""
    echo "Next steps:"
    echo "  1. Review faithfulness rate in validation output"
    echo "  2. Check if ≥10 unfaithful examples (needed for Phase 3)"
    echo "  3. If ready, proceed to Phase 3: Mechanistic Analysis"
    echo ""
else
    echo ""
    echo "❌ Phase 2 validation failed"
    echo "Fix errors before proceeding to Phase 3"
    exit 1
fi

