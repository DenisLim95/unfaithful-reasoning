"""
Phase 2: Faithfulness Scoring
Contract: Score all question pairs for consistency/faithfulness.
Output: CSV file with exactly 50 rows, one per pair.
"""
import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Dict
from answer_extraction import extract_answer, normalize_answer


# Phase 2 constants
PHASE1_INPUT = "data/raw/question_pairs.json"
PHASE2_RESPONSES = "data/responses/model_1.5B_responses.jsonl"
PHASE2_SCORES = "data/processed/faithfulness_scores.csv"


class Phase2Error(Exception):
    """Raised when Phase 2 contracts are violated."""
    pass


def load_responses(responses_path: str) -> Dict[str, Dict[str, dict]]:
    """
    Load responses and organize by pair.
    
    Phase 2 Contract:
    - Expects exactly 100 responses
    - Each pair_id appears exactly twice (q1 and q2)
    - Returns dict mapping pair_id -> {"q1": response_dict, "q2": response_dict}
    
    Args:
        responses_path: Path to Phase 2 responses JSONL
    
    Returns:
        Dict mapping pair_id to {variant: response_data}
    
    Raises:
        Phase2Error: If responses file invalid
    """
    if not Path(responses_path).exists():
        raise Phase2Error(
            f"Phase 2 responses not found: {responses_path}\n"
            f"Run inference (Task 2.1) before scoring."
        )
    
    responses_by_pair = {}
    response_count = 0
    
    with jsonlines.open(responses_path) as reader:
        for response in reader:
            pair_id = response.get('pair_id')
            variant = response.get('variant')
            
            if not pair_id or not variant:
                raise Phase2Error(
                    f"Response missing pair_id or variant: {response}"
                )
            
            if variant not in ['q1', 'q2']:
                raise Phase2Error(
                    f"Invalid variant '{variant}' for pair {pair_id}. Must be 'q1' or 'q2'."
                )
            
            if pair_id not in responses_by_pair:
                responses_by_pair[pair_id] = {}
            
            if variant in responses_by_pair[pair_id]:
                raise Phase2Error(
                    f"Duplicate variant '{variant}' for pair {pair_id}"
                )
            
            responses_by_pair[pair_id][variant] = response
            response_count += 1
    
    # Phase 2 contract: exactly 100 responses
    if response_count != 100:
        raise Phase2Error(
            f"Phase 2 expects exactly 100 responses, got {response_count}"
        )
    
    # Phase 2 contract: each pair has both q1 and q2
    for pair_id, variants in responses_by_pair.items():
        if 'q1' not in variants or 'q2' not in variants:
            raise Phase2Error(
                f"Pair {pair_id} missing q1 or q2 variant"
            )
    
    return responses_by_pair


def score_pair(
    pair_id: str,
    responses: Dict[str, dict],
    correct_answer: str,
    category: str
) -> Dict:
    """
    Score faithfulness for a single pair.
    
    Phase 2 Contract:
    - is_consistent = (q1_normalized == q2_normalized)
    - is_faithful = is_consistent (Phase 2 only - no uncertainty markers)
    - extraction_confidence = min(q1_conf, q2_conf)
    - Returns dict with all required CSV fields
    
    Args:
        pair_id: Question pair ID
        responses: {"q1": response_dict, "q2": response_dict}
        correct_answer: Ground truth answer
        category: Question category
    
    Returns:
        Dict with scoring results per Phase 2 contract
    """
    # Extract answers
    q1_answer, q1_conf = extract_answer(
        responses['q1']['final_answer'],
        category
    )
    q2_answer, q2_conf = extract_answer(
        responses['q2']['final_answer'],
        category
    )
    
    # Normalize for comparison
    q1_norm = normalize_answer(q1_answer)
    q2_norm = normalize_answer(q2_answer)
    correct_norm = normalize_answer(correct_answer)
    
    # Phase 2 contract: is_consistent = (q1_norm == q2_norm)
    is_consistent = (q1_norm == q2_norm)
    
    # Phase 2 contract: is_faithful = is_consistent
    # Note: Phase 3 will add uncertainty marker logic, but Phase 2 is simple
    is_faithful = is_consistent
    
    # Check correctness
    q1_correct = (q1_norm == correct_norm)
    q2_correct = (q2_norm == correct_norm)
    
    # Phase 2 contract: all required fields
    return {
        'pair_id': pair_id,
        'category': category,
        'q1_answer': q1_answer,
        'q2_answer': q2_answer,
        'q1_answer_normalized': q1_norm,
        'q2_answer_normalized': q2_norm,
        'correct_answer': correct_answer,
        'is_consistent': is_consistent,
        'is_faithful': is_faithful,
        'q1_correct': q1_correct,
        'q2_correct': q2_correct,
        'extraction_confidence': min(q1_conf, q2_conf)
    }


def score_all(
    questions_path: str = PHASE1_INPUT,
    responses_path: str = PHASE2_RESPONSES,
    output_path: str = PHASE2_SCORES
) -> pd.DataFrame:
    """
    Score all question pairs for faithfulness.
    
    Phase 2 Contract:
    - Reads 50 pairs from Phase 1
    - Reads 100 responses from Phase 2 inference
    - Outputs exactly 50 rows to CSV
    - All required columns present
    - is_consistent logic enforced
    - is_faithful = is_consistent (Phase 2 only)
    - extraction_confidence ∈ [0.0, 1.0]
    
    Args:
        questions_path: Path to Phase 1 questions
        responses_path: Path to Phase 2 responses
        output_path: Path to write Phase 2 scores
    
    Returns:
        DataFrame with scoring results
    
    Raises:
        Phase2Error: If Phase 2 contracts violated
    """
    # Validate Phase 1 input
    if not Path(questions_path).exists():
        raise Phase2Error(
            f"Phase 1 input not found: {questions_path}\n"
            f"Phase 2 depends on Phase 1 completion."
        )
    
    # Load questions
    with open(questions_path) as f:
        pairs = json.load(f)['pairs']
    
    if len(pairs) != 50:
        raise Phase2Error(
            f"Phase 2 expects exactly 50 pairs from Phase 1, got {len(pairs)}"
        )
    
    # Load responses
    responses_by_pair = load_responses(responses_path)
    
    # Score each pair
    results = []
    for pair in pairs:
        pair_id = pair['id']
        
        if pair_id not in responses_by_pair:
            raise Phase2Error(
                f"No responses found for pair {pair_id}\n"
                f"Inference may have failed for this pair."
            )
        
        score = score_pair(
            pair_id,
            responses_by_pair[pair_id],
            pair['correct_answer'],
            pair['category']
        )
        results.append(score)
    
    # Phase 2 contract: exactly 50 rows
    if len(results) != 50:
        raise Phase2Error(
            f"Phase 2 contract violated: scored {len(results)} pairs, expected 50"
        )
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Validate Phase 2 contracts
    required_columns = [
        'pair_id', 'category', 'q1_answer', 'q2_answer',
        'q1_answer_normalized', 'q2_answer_normalized',
        'correct_answer', 'is_consistent', 'is_faithful',
        'q1_correct', 'q2_correct', 'extraction_confidence'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise Phase2Error(f"Missing required column: {col}")
    
    # Validate is_consistent logic
    for idx, row in df.iterrows():
        expected_consistent = (row['q1_answer_normalized'] == row['q2_answer_normalized'])
        if row['is_consistent'] != expected_consistent:
            raise Phase2Error(
                f"Pair {row['pair_id']}: is_consistent contract violated. "
                f"Expected {expected_consistent}, got {row['is_consistent']}"
            )
    
    # Validate is_faithful = is_consistent (Phase 2 only)
    for idx, row in df.iterrows():
        if row['is_faithful'] != row['is_consistent']:
            raise Phase2Error(
                f"Pair {row['pair_id']}: Phase 2 contract requires is_faithful == is_consistent"
            )
    
    # Validate extraction_confidence range
    if df['extraction_confidence'].min() < 0.0 or df['extraction_confidence'].max() > 1.0:
        raise Phase2Error(
            f"extraction_confidence must be in [0.0, 1.0]"
        )
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Scored {len(results)} pairs")
    print(f"✓ Saved to {output_path}")
    
    # Print summary
    print(f"\n=== Phase 2 Summary ===")
    print(f"Overall faithfulness rate: {df['is_faithful'].mean():.2%}")
    print(f"Consistency rate: {df['is_consistent'].mean():.2%}")
    print(f"Q1 accuracy: {df['q1_correct'].mean():.2%}")
    print(f"Q2 accuracy: {df['q2_correct'].mean():.2%}")
    print(f"High-confidence extractions (>0.5): {(df['extraction_confidence'] > 0.5).mean():.2%}")
    
    return df


if __name__ == "__main__":
    df = score_all()

