"""
Phase 2 Validation Tests
Contract: Executable encoding of Phase 2 acceptance criteria.
Must pass before Phase 3 can begin.
"""
import json
import jsonlines
import pandas as pd
import sys
from pathlib import Path
from collections import Counter


# Phase 2 file paths (from spec)
PHASE1_INPUT = "data/raw/question_pairs.json"
PHASE2_RESPONSES = "data/responses/model_1.5B_responses.jsonl"
PHASE2_SCORES = "data/processed/faithfulness_scores.csv"


class Phase2ValidationError(Exception):
    """Raised when Phase 2 validation fails."""
    pass


def validate_responses(file_path: str = PHASE2_RESPONSES):
    """
    Validate model responses against Phase 2 contract.
    
    Phase 2 Acceptance Criteria:
    1. File exists
    2. Contains exactly 100 lines (valid JSON)
    3. Each pair_id appears exactly twice
    4. All responses are non-empty
    5. All required fields present
    6. All variant fields are "q1" or "q2"
    
    Returns:
        (is_valid: bool, errors: list)
    """
    errors = []
    
    # Check 1: File exists
    if not Path(file_path).exists():
        return False, [f"Phase 2 responses file not found: {file_path}"]
    
    # Load all responses
    responses = []
    pair_ids = []
    
    try:
        with jsonlines.open(file_path) as reader:
            for i, response in enumerate(reader):
                responses.append(response)
                
                # Check 5: Required fields
                required = [
                    'pair_id', 'variant', 'question', 'response',
                    'think_section', 'final_answer', 'timestamp', 'generation_config'
                ]
                for field in required:
                    if field not in response:
                        errors.append(f"Line {i}: Missing required field '{field}'")
                
                # Check 6: Variant is q1 or q2
                if response.get('variant') not in ['q1', 'q2']:
                    errors.append(f"Line {i}: Invalid variant '{response.get('variant')}'")
                
                # Check 4: Response is non-empty
                if not response.get('response'):
                    errors.append(f"Line {i}: Empty response field")
                
                # Check generation_config structure
                gen_config = response.get('generation_config', {})
                if 'temperature' not in gen_config:
                    errors.append(f"Line {i}: Missing generation_config.temperature")
                if 'model' not in gen_config:
                    errors.append(f"Line {i}: Missing generation_config.model")
                
                pair_ids.append(response.get('pair_id'))
    
    except Exception as e:
        return False, [f"Error reading JSONL: {e}"]
    
    # Check 2: Exactly 100 responses
    if len(responses) != 100:
        errors.append(
            f"Phase 2 contract violation: Expected exactly 100 responses, got {len(responses)}"
        )
    
    # Check 3: Each pair_id appears exactly twice
    counts = Counter(pair_ids)
    for pair_id, count in counts.items():
        if count != 2:
            errors.append(
                f"Phase 2 contract violation: Pair {pair_id} has {count} responses (expected 2)"
            )
    
    # Check that we have 50 unique pairs
    if len(counts) != 50:
        errors.append(
            f"Phase 2 contract violation: Expected 50 unique pair_ids, got {len(counts)}"
        )
    
    return len(errors) == 0, errors


def validate_scores(file_path: str = PHASE2_SCORES):
    """
    Validate faithfulness scores against Phase 2 contract.
    
    Phase 2 Acceptance Criteria:
    6. File exists
    7. Contains exactly 50 rows
    8. All required columns present
    9. Faithfulness rate is between 0% and 100%
    10. At least 80% of extractions have confidence > 0.5
    11. is_consistent = (q1_norm == q2_norm) for all rows
    12. is_faithful = is_consistent for all rows (Phase 2 only)
    
    Returns:
        (is_valid: bool, errors: list)
    """
    errors = []
    
    # Check 6: File exists
    if not Path(file_path).exists():
        return False, [f"Phase 2 scores file not found: {file_path}"]
    
    # Load CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return False, [f"Error reading CSV: {e}"]
    
    # Check 7: Exactly 50 rows
    if len(df) != 50:
        errors.append(
            f"Phase 2 contract violation: Expected exactly 50 rows, got {len(df)}"
        )
    
    # Check 8: Required columns
    required_cols = [
        'pair_id', 'category', 'q1_answer', 'q2_answer',
        'q1_answer_normalized', 'q2_answer_normalized',
        'correct_answer', 'is_consistent', 'is_faithful',
        'q1_correct', 'q2_correct', 'extraction_confidence'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Phase 2 contract violation: Missing required column '{col}'")
    
    if errors:
        return False, errors
    
    # Check 9: Faithfulness rate in [0, 1]
    faithful_rate = df['is_faithful'].mean()
    if not (0 <= faithful_rate <= 1):
        errors.append(
            f"Phase 2 contract violation: Faithfulness rate {faithful_rate} not in [0, 1]"
        )
    
    # Check extraction confidence range
    if df['extraction_confidence'].min() < 0 or df['extraction_confidence'].max() > 1:
        errors.append(
            "Phase 2 contract violation: extraction_confidence values outside [0, 1]"
        )
    
    # Check 10: High-confidence extractions
    high_conf_pct = (df['extraction_confidence'] > 0.5).mean()
    if high_conf_pct < 0.8:
        errors.append(
            f"Phase 2 contract violation: Only {high_conf_pct:.1%} have confidence > 0.5 "
            f"(expected >= 80%)"
        )
    
    # Check 11: Consistency logic
    for idx, row in df.iterrows():
        expected_consistent = (row['q1_answer_normalized'] == row['q2_answer_normalized'])
        if row['is_consistent'] != expected_consistent:
            errors.append(
                f"Phase 2 contract violation: Pair {row['pair_id']}: "
                f"is_consistent={row['is_consistent']} but normalized answers "
                f"{'match' if expected_consistent else 'differ'}"
            )
    
    # Check 12: Phase 2 faithfulness logic (is_faithful = is_consistent)
    for idx, row in df.iterrows():
        if row['is_faithful'] != row['is_consistent']:
            errors.append(
                f"Phase 2 contract violation: Pair {row['pair_id']}: "
                f"is_faithful={row['is_faithful']} but is_consistent={row['is_consistent']}. "
                f"Phase 2 requires is_faithful == is_consistent."
            )
    
    return len(errors) == 0, errors


def validate_phase1_dependency():
    """
    Validate that Phase 1 is complete before running Phase 2.
    
    Returns:
        (is_valid: bool, errors: list)
    """
    errors = []
    
    if not Path(PHASE1_INPUT).exists():
        errors.append(
            f"Phase 1 dependency not satisfied: {PHASE1_INPUT} not found. "
            f"Run Phase 1 validation first."
        )
        return False, errors
    
    try:
        with open(PHASE1_INPUT) as f:
            data = json.load(f)
            pairs = data.get('pairs', [])
            
            if len(pairs) != 50:
                errors.append(
                    f"Phase 1 → Phase 2 contract violation: "
                    f"Expected 50 pairs from Phase 1, found {len(pairs)}"
                )
    except Exception as e:
        errors.append(f"Error validating Phase 1 input: {e}")
        return False, errors
    
    return len(errors) == 0, errors


def main():
    """
    Run Phase 2 validation.
    
    Exit codes:
        0: All Phase 2 acceptance criteria passed
        1: Phase 2 validation failed
    """
    print("=" * 60)
    print("PHASE 2 VALIDATION: Faithfulness Evaluation")
    print("=" * 60)
    
    all_pass = True
    
    # Check Phase 1 dependency
    print("\n0. Checking Phase 1 dependency...")
    valid_phase1, phase1_errors = validate_phase1_dependency()
    if valid_phase1:
        print("   ✅ Phase 1 dependency satisfied")
    else:
        print(f"   ❌ {len(phase1_errors)} error(s):")
        for err in phase1_errors:
            print(f"      • {err}")
        all_pass = False
        # Stop here if Phase 1 not complete
        print("\n" + "=" * 60)
        print("❌ PHASE 2 VALIDATION FAILED")
        print("\n❌ Complete Phase 1 before running Phase 2")
        print("=" * 60)
        return 1
    
    # Validate responses
    print("\n1. Validating model responses...")
    valid_responses, response_errors = validate_responses()
    if valid_responses:
        print("   ✅ Responses valid")
    else:
        print(f"   ❌ {len(response_errors)} error(s):")
        for err in response_errors[:10]:  # Show first 10
            print(f"      • {err}")
        if len(response_errors) > 10:
            print(f"      ... and {len(response_errors) - 10} more errors")
        all_pass = False
    
    # Validate scores
    print("\n2. Validating faithfulness scores...")
    valid_scores, score_errors = validate_scores()
    if valid_scores:
        print("   ✅ Scores valid")
        
        # Print summary stats
        try:
            df = pd.read_csv(PHASE2_SCORES)
            print(f"\n   Summary Statistics:")
            print(f"     • Faithfulness rate: {df['is_faithful'].mean():.1%}")
            print(f"     • Consistency rate: {df['is_consistent'].mean():.1%}")
            print(f"     • Q1 accuracy: {df['q1_correct'].mean():.1%}")
            print(f"     • Q2 accuracy: {df['q2_correct'].mean():.1%}")
            print(f"     • High-confidence extractions: {(df['extraction_confidence'] > 0.5).mean():.1%}")
        except Exception as e:
            print(f"   ⚠ Could not print summary: {e}")
    else:
        print(f"   ❌ {len(score_errors)} error(s):")
        for err in score_errors[:10]:
            print(f"      • {err}")
        if len(score_errors) > 10:
            print(f"      ... and {len(score_errors) - 10} more errors")
        all_pass = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL PHASE 2 CHECKS PASSED")
        print("\nPhase 2 Deliverables:")
        print(f"  ✓ {PHASE2_RESPONSES} (100 responses)")
        print(f"  ✓ {PHASE2_SCORES} (50 scored pairs)")
        print("\n✅ Ready to proceed to Phase 3")
        return 0
    else:
        print("❌ PHASE 2 VALIDATION FAILED")
        print("\n❌ Fix errors before proceeding to Phase 3")
        return 1


if __name__ == "__main__":
    sys.exit(main())

