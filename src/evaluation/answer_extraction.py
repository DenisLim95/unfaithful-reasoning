"""
Phase 2: Answer Extraction
Contract: Extract answers from model responses with confidence scores.
Scope: ONLY numerical_comparison category (Phase 1 constraint)
"""
import re
from typing import Tuple


# Phase 2 contract: Only numerical_comparison supported
SUPPORTED_CATEGORIES = {"numerical_comparison"}


class Phase2Error(Exception):
    """Raised when trying to use Phase 3+ features in Phase 2."""
    pass


def extract_answer(final_answer: str, category: str = "numerical_comparison") -> Tuple[str, float]:
    """
    Extract the model's answer from final_answer section.
    
    Phase 2 Contract:
    - Returns (answer: str, confidence: float)
    - confidence ∈ [0.0, 1.0]
    - answer is never empty
    - Only supports category="numerical_comparison"
    
    Args:
        final_answer: Text after </think> tag
        category: Must be "numerical_comparison" in Phase 2
    
    Returns:
        (answer, confidence) where confidence indicates extraction reliability
    
    Raises:
        Phase2Error: If category is not "numerical_comparison"
        ValueError: If final_answer is empty
    """
    if not final_answer or not final_answer.strip():
        raise ValueError("final_answer cannot be empty")
    
    if category not in SUPPORTED_CATEGORIES:
        raise Phase2Error(
            f"Category '{category}' not supported in Phase 2. "
            f"Only {SUPPORTED_CATEGORIES} are implemented. "
            f"This is a Phase 2 boundary enforcement."
        )
    
    final_answer = final_answer.strip()
    
    # Strategy 1: Look for explicit answer patterns (confidence: 0.9)
    patterns = [
        r"(?:answer is|final answer:|answer:)\s*(?:\*\*)?(.+?)(?:\*\*)?(?:\.|$)",
        r"(?:therefore|thus|so),?\s*(?:\*\*)?(.+?)(?:\*\*)?(?:\s+is|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, final_answer, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer, 0.9
    
    # Strategy 2: For numerical_comparison, extract first number (confidence: 0.7)
    if category == "numerical_comparison":
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', final_answer)
        if numbers:
            return numbers[0], 0.7
    
    # Strategy 3: Take first sentence (confidence: 0.4)
    sentences = final_answer.split('.')
    if sentences and sentences[0].strip():
        return sentences[0].strip(), 0.4
    
    # Strategy 4: Return entire final answer (confidence: 0.2)
    return final_answer, 0.2


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Phase 2 Contract:
    - Returns normalized lowercase string
    - Deterministic (same input → same output)
    - Extracts numbers for numerical comparisons
    
    Args:
        answer: Raw answer string
    
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    # Convert to lowercase, remove punctuation, normalize whitespace
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
    answer = re.sub(r'\s+', ' ', answer)      # Normalize whitespace
    
    # Extract just numbers if present (for numerical_comparison)
    numbers = re.findall(r'\d+', answer)
    if numbers:
        return numbers[0]
    
    return answer.strip()


# Phase 2 test helper
if __name__ == "__main__":
    # Test extraction strategies
    test_cases = [
        ("The answer is 847.", "numerical_comparison"),
        ("Therefore, 839 is larger.", "numerical_comparison"),
        ("847", "numerical_comparison"),
        ("I think it's 847 because...", "numerical_comparison"),
    ]
    
    print("=== Phase 2 Answer Extraction Tests ===")
    for answer_text, category in test_cases:
        extracted, conf = extract_answer(answer_text, category)
        normalized = normalize_answer(extracted)
        print(f"Input: {answer_text}")
        print(f"  Extracted: {extracted} (confidence: {conf:.1f})")
        print(f"  Normalized: {normalized}\n")
    
    # Test Phase 2 boundary enforcement
    print("=== Phase 2 Boundary Test ===")
    try:
        extract_answer("Paris", "factual_comparison")
        print("ERROR: Should have raised Phase2Error!")
    except Phase2Error as e:
        print(f"✓ Correctly rejected non-Phase-2 category: {e}")

