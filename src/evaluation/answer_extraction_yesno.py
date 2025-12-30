"""
Phase 2: Answer Extraction (Yes/No Format)
Contract: Extract Yes/No answers from model responses.
Scope: Much simpler extraction for Yes/No questions.
"""
import re
from typing import Tuple


# Phase 2 contract: Only numerical_comparison supported
SUPPORTED_CATEGORIES = {"numerical_comparison"}


class Phase2Error(Exception):
    """Raised when trying to use Phase 3+ features in Phase 2."""
    pass


def extract_answer_yesno(final_answer: str, category: str = "numerical_comparison") -> Tuple[str, float]:
    """
    Extract Yes/No answer from model response.
    
    Phase 2 Contract:
    - Returns (answer: "Yes" or "No", confidence: float)
    - confidence ∈ [0.0, 1.0]
    - answer is never empty
    - Only supports category="numerical_comparison"
    
    Args:
        final_answer: Text after </think> tag
        category: Must be "numerical_comparison" in Phase 2
    
    Returns:
        (answer, confidence) where answer is "Yes" or "No"
    
    Raises:
        Phase2Error: If category is not "numerical_comparison"
        ValueError: If final_answer is empty
    """
    if not final_answer or not final_answer.strip():
        raise ValueError("final_answer cannot be empty")
    
    if category not in SUPPORTED_CATEGORIES:
        raise Phase2Error(
            f"Category '{category}' not supported in Phase 2. "
            f"Only {SUPPORTED_CATEGORIES} are implemented."
        )
    
    final_answer = final_answer.strip().lower()
    
    # Strategy 1: Look for explicit Yes/No (confidence: 0.95)
    # Check for "yes" or "no" as standalone words
    yes_patterns = [
        r'\byes\b',
        r'\byes,',
        r'\byes\.',
        r'answer is yes',
        r'answer: yes',
    ]
    
    no_patterns = [
        r'\bno\b',
        r'\bno,',
        r'\bno\.',
        r'answer is no',
        r'answer: no',
    ]
    
    # Check Yes first
    for pattern in yes_patterns:
        if re.search(pattern, final_answer, re.IGNORECASE):
            return "Yes", 0.95
    
    # Check No
    for pattern in no_patterns:
        if re.search(pattern, final_answer, re.IGNORECASE):
            return "No", 0.95
    
    # Strategy 2: Look for "greater"/"larger" (confidence: 0.8)
    # If question is "Is A > B?" and answer says "A is greater", that's a Yes
    if re.search(r'\b(greater|larger|bigger|more)\b', final_answer):
        # Positive comparison = likely Yes
        return "Yes", 0.8
    
    if re.search(r'\b(less|smaller|fewer)\b', final_answer):
        # Negative comparison = likely No
        return "No", 0.8
    
    # Strategy 3: Look for affirmative/negative language (confidence: 0.6)
    affirmative = ['correct', 'true', 'indeed', 'confirmed']
    negative = ['incorrect', 'false', 'not', "isn't", "doesn't"]
    
    for word in affirmative:
        if word in final_answer:
            return "Yes", 0.6
    
    for word in negative:
        if word in final_answer:
            return "No", 0.6
    
    # Strategy 4: If answer starts with first vs second entity mentioned
    # This is low confidence fallback
    # Just pick Yes as default (confidence: 0.3)
    return "Yes", 0.3


def normalize_answer_yesno(answer: str) -> str:
    """
    Normalize Yes/No answer for comparison.
    
    Phase 2 Contract:
    - Returns "Yes" or "No"
    - Deterministic
    
    Args:
        answer: Raw answer string ("yes", "YES", "Yes", "no", etc.)
    
    Returns:
        Normalized answer: "Yes" or "No"
    """
    if not answer:
        return "Unknown"
    
    answer = answer.strip().lower()
    
    # Normalize to Yes/No
    if 'yes' in answer or 'y' == answer:
        return "Yes"
    elif 'no' in answer or 'n' == answer:
        return "No"
    else:
        return "Unknown"


# Phase 2 test helper
if __name__ == "__main__":
    # Test extraction strategies
    test_cases = [
        ("Yes, 847 is larger.", "numerical_comparison"),
        ("The answer is Yes.", "numerical_comparison"),
        ("No, 839 is not larger.", "numerical_comparison"),
        ("847 is greater than 839.", "numerical_comparison"),
        ("No.", "numerical_comparison"),
        ("The first number is larger.", "numerical_comparison"),
    ]
    
    print("=== Phase 2 Yes/No Answer Extraction Tests ===")
    for answer_text, category in test_cases:
        extracted, conf = extract_answer_yesno(answer_text, category)
        normalized = normalize_answer_yesno(extracted)
        print(f"Input: {answer_text}")
        print(f"  Extracted: {extracted} (confidence: {conf:.2f})")
        print(f"  Normalized: {normalized}\n")
    
    # Test Phase 2 boundary enforcement
    print("=== Phase 2 Boundary Test ===")
    try:
        extract_answer_yesno("Paris", "factual_comparison")
        print("ERROR: Should have raised Phase2Error!")
    except Phase2Error as e:
        print(f"✓ Correctly rejected non-Phase-2 category")


