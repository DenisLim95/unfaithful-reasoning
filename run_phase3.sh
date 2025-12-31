#!/bin/bash
#
# Phase 3: Mechanistic Analysis - Complete Runner
#
# This script runs all Phase 3 tasks in sequence.
# Total time: 6-7 hours
#

set -e  # Exit on error

echo "============================================================"
echo "PHASE 3: Mechanistic Analysis - Linear Probe Analysis"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Cache activations (2-3 hours)"
echo "  2. Train linear probes (1-2 hours)"
echo "  3. Validate Phase 3 deliverables (5 min)"
echo ""
echo "Prerequisites:"
echo "  - Phase 2 must be complete"
echo "  - transformer-lens must be installed"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Check Phase 2 is complete
echo ""
echo "[0/3] Checking Phase 2..."
if [ ! -f "data/responses/model_1.5B_responses.jsonl" ]; then
    echo "❌ Phase 2 output missing: data/responses/model_1.5B_responses.jsonl"
    echo "   Run Phase 2 first: python src/inference/batch_inference.py"
    exit 1
fi

if [ ! -f "data/processed/faithfulness_scores.csv" ]; then
    echo "❌ Phase 2 output missing: data/processed/faithfulness_scores.csv"
    echo "   Run Phase 2 first: python src/evaluation/score_faithfulness.py"
    exit 1
fi

echo "✓ Phase 2 outputs found"

# Task 3.2: Cache Activations
echo ""
echo "============================================================"
echo "[1/3] Task 3.2: Caching Activations"
echo "============================================================"
echo ""
echo "⏱️  Estimated time: 2-3 hours"
echo "🖥️  Requires: GPU (model loading)"
echo ""
echo "Using HuggingFace directly (TransformerLens doesn't support DeepSeek)"
python src/mechanistic/cache_activations_nnsight.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Task 3.2 failed. Check error above."
    exit 1
fi

# Task 3.3: Train Probes
echo ""
echo "============================================================"
echo "[2/3] Task 3.3: Training Linear Probes"
echo "============================================================"
echo ""
echo "⏱️  Estimated time: 1-2 hours"
echo "🖥️  Can run on CPU (uses cached activations)"
echo ""
python src/mechanistic/train_probes.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Task 3.3 failed. Check error above."
    exit 1
fi

# Task 3.4: Validation
echo ""
echo "============================================================"
echo "[3/3] Task 3.4: Validating Phase 3 Deliverables"
echo "============================================================"
echo ""
python tests/validate_phase3.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "🎉 PHASE 3 COMPLETE!"
    echo "============================================================"
    echo ""
    echo "Deliverables:"
    echo "  ✓ 4 activation cache files (data/activations/)"
    echo "  ✓ 1 probe results file (results/probe_results/)"
    echo "  ✓ 1 performance plot (results/probe_results/probe_performance.png)"
    echo ""
    echo "Next steps:"
    echo "  • Review probe_performance.png"
    echo "  • Interpret results (see PHASE3_README.md)"
    echo "  • Proceed to Phase 4 (Report & Polish)"
    echo ""
    echo "See: phased_implementation_plan.md lines 2043-2542 for Phase 4"
    echo ""
    exit 0
else
    echo ""
    echo "============================================================"
    echo "❌ PHASE 3 VALIDATION FAILED"
    echo "============================================================"
    echo ""
    echo "Check errors above and re-run:"
    echo "  python tests/validate_phase3.py"
    echo ""
    exit 1
fi

