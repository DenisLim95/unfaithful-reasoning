#!/bin/bash
# Regenerate questions in Yes/No format

set -e

echo "========================================"
echo "REGENERATING QUESTIONS (Yes/No Format)"
echo "========================================"

# Backup old questions
if [ -f "data/raw/question_pairs.json" ]; then
    echo ""
    echo "Backing up old questions..."
    mv data/raw/question_pairs.json data/raw/question_pairs_old.json
    echo "✓ Backed up to data/raw/question_pairs_old.json"
fi

# Generate new Yes/No questions
echo ""
echo "Generating new Yes/No questions..."
python src/data_generation/generate_questions_yesno.py

# Validate
echo ""
echo "Validating new questions..."
python tests/validate_questions.py

echo ""
echo "========================================"
echo "✅ QUESTIONS REGENERATED"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review the new question format above"
echo "  2. If satisfied, run Phase 2 with new questions:"
echo "     ./run_phase2.sh"
echo ""
echo "To restore old questions:"
echo "  mv data/raw/question_pairs_old.json data/raw/question_pairs.json"
echo ""


