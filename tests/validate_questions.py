"""
Validation script for Phase 1 question pairs.

Checks all acceptance criteria:
1. File exists
2. Valid JSON
3. Exactly 50 pairs
4. All required fields present
5. No duplicate IDs
6. All q1 != q2
7. Difficulty distribution: 20/20/10
8. All correct_answer fields non-empty
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def validate_question_pairs(
    file_path: str = "data/raw/question_pairs.json"
) -> Tuple[bool, List[str]]:
    """
    Validate question pairs against spec.
    
    Args:
        file_path: Path to question pairs JSON file
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check 1: File exists
    if not Path(file_path).exists():
        return False, [f"File not found: {file_path}"]
    
    # Check 2: Valid JSON
    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    
    # Check 3: Has 'pairs' key
    if 'pairs' not in data:
        return False, ["Missing 'pairs' key in root object"]
    
    pairs = data['pairs']
    
    # Check 4: Exactly 50 pairs
    if len(pairs) != 50:
        errors.append(f"Expected 50 pairs, got {len(pairs)}")
    
    # Check 5-11: Validate each pair
    ids = set()
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    
    for i, pair in enumerate(pairs):
        pair_id = pair.get('id', f'<missing_{i}>')
        
        # Check required fields
        required_fields = ['id', 'category', 'difficulty', 'q1', 'q2', 'correct_answer', 'metadata']
        for field in required_fields:
            if field not in pair:
                errors.append(f"Pair {pair_id}: Missing required field '{field}'")
        
        # Check duplicate IDs
        if pair_id in ids:
            errors.append(f"Duplicate ID: {pair_id}")
        ids.add(pair_id)
        
        # Check q1 != q2
        if pair.get('q1') == pair.get('q2'):
            errors.append(f"Pair {pair_id}: q1 and q2 are identical")
        
        # Check correct_answer is non-empty
        if not pair.get('correct_answer'):
            errors.append(f"Pair {pair_id}: Empty correct_answer")
        
        # Check difficulty is valid
        difficulty = pair.get('difficulty')
        if difficulty not in ['easy', 'medium', 'hard']:
            errors.append(f"Pair {pair_id}: Invalid difficulty '{difficulty}'")
        else:
            difficulty_counts[difficulty] += 1
        
        # Check category
        if pair.get('category') != 'numerical_comparison':
            errors.append(
                f"Pair {pair_id}: Expected category 'numerical_comparison', "
                f"got '{pair.get('category')}'"
            )
    
    # Check difficulty distribution
    if difficulty_counts['easy'] != 20:
        errors.append(f"Expected 20 easy pairs, got {difficulty_counts['easy']}")
    if difficulty_counts['medium'] != 20:
        errors.append(f"Expected 20 medium pairs, got {difficulty_counts['medium']}")
    if difficulty_counts['hard'] != 10:
        errors.append(f"Expected 10 hard pairs, got {difficulty_counts['hard']}")
    
    return len(errors) == 0, errors


def main():
    """Run validation and print results."""
    print("=" * 60)
    print("PHASE 1 VALIDATION: Question Pairs")
    print("=" * 60)
    
    is_valid, errors = validate_question_pairs()
    
    if is_valid:
        print("\n✅ ALL CHECKS PASSED")
        print("\nPhase 1 acceptance criteria met:")
        print("  ✓ File exists and is valid JSON")
        print("  ✓ Contains 50 pairs")
        print("  ✓ All required fields present")
        print("  ✓ No duplicate IDs")
        print("  ✓ All q1 != q2")
        print("  ✓ Correct difficulty distribution (20/20/10)")
        print("  ✓ All correct_answer fields non-empty")
        print("\n✅ Ready to proceed to Phase 2")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED: {len(errors)} error(s)\n")
        for error in errors:
            print(f"  • {error}")
        print("\n❌ Fix errors before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())

