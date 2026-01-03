#!/bin/bash
#
# Workflow: Probe Training and Testing
#
# Train probes on cached activations and test generalization.
#
# Usage:
#   bash workflows/probe_training.sh
#

set -e  # Exit on error

echo "============================================================"
echo "WORKFLOW: PROBE TRAINING AND TESTING"
echo "============================================================"

# Configuration
ACTIVATIONS_DIR="data/activations"
TEST_ACTIVATIONS_DIR="data/test_activations"
PROBES_FILE="results/probe_results/all_probe_results.pt"

# Step 1: Validate activation caches
echo ""
echo "============================================================"
echo "STEP 1: Validate Activation Caches"
echo "============================================================"
python scripts/04_cache_activations.py --validate

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: Activation caches not found or invalid"
    echo ""
    echo "Please run activation caching first:"
    echo "  python scripts/04_cache_activations.py \\"
    echo "      --responses data/responses/responses.jsonl \\"
    echo "      --scores data/processed/faithfulness_scores.csv"
    exit 1
fi

echo ""

# Step 2: Train probes
echo "============================================================"
echo "STEP 2: Train Linear Probes"
echo "============================================================"
python scripts/05_train_probes.py \
    --activations "$ACTIVATIONS_DIR" \
    --output "$PROBES_FILE"

echo ""

# Step 3: Test probes (if test activations exist)
if [ -d "$TEST_ACTIVATIONS_DIR" ]; then
    echo "============================================================"
    echo "STEP 3: Test Probe Generalization"
    echo "============================================================"
    python scripts/06_test_probes.py \
        --probes "$PROBES_FILE" \
        --test-activations "$TEST_ACTIVATIONS_DIR"
else
    echo "============================================================"
    echo "STEP 3: Test Activations Not Found (Skipping)"
    echo "============================================================"
    echo ""
    echo "To test probe generalization, cache test activations first:"
    echo "  1. Generate test responses"
    echo "  2. Score them"
    echo "  3. Cache activations to $TEST_ACTIVATIONS_DIR"
fi

echo ""
echo "============================================================"
echo "WORKFLOW COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
echo "  Trained probes: $PROBES_FILE"
if [ -d "$TEST_ACTIVATIONS_DIR" ]; then
    echo "  Test results: see output above"
fi
echo ""

