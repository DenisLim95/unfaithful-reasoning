#!/bin/bash
# Cleanup script to remove all test data while preserving training data

echo "========================================"
echo "CLEANING UP TEST DATA"
echo "========================================"
echo ""

# Define what to remove
TEST_DIRS=(
    "data/test_activations"
)

TEST_FILES=(
    "data/raw/test_question_pairs.json"
    "data/responses/test_responses.jsonl"
    "data/processed/test_faithfulness_scores.csv"
)

# Remove test directories
for dir in "${TEST_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing directory: $dir"
        rm -rf "$dir"
    else
        echo "Directory not found (skipping): $dir"
    fi
done

echo ""

# Remove test files
for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing file: $file"
        rm -f "$file"
    else
        echo "File not found (skipping): $file"
    fi
done

echo ""
echo "========================================"
echo "WHAT'S PRESERVED (Training Data)"
echo "========================================"
echo ""

# Show what's kept
echo "✓ Training activations: data/activations/"
ls -lh data/activations/ 2>/dev/null || echo "  (not found)"

echo ""
echo "✓ Trained probes: results/probe_results/"
ls -lh results/probe_results/ 2>/dev/null || echo "  (not found)"

echo ""
echo "✓ Training questions: data/raw/question_pairs.json"
ls -lh data/raw/question_pairs.json 2>/dev/null || echo "  (not found)"

echo ""
echo "✓ Training responses: data/responses/model_1.5B_responses.jsonl"
ls -lh data/responses/model_1.5B_responses.jsonl 2>/dev/null || echo "  (not found)"

echo ""
echo "========================================"
echo "✅ CLEANUP COMPLETE!"
echo "========================================"
echo ""
echo "You can now run a fresh test:"
echo "  python test_probe_on_new_data.py --num-questions 200"



