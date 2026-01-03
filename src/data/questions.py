"""
Question generation utilities.

This module provides functions for generating paired comparison questions.
"""

import random
from typing import Dict, List


def generate_numerical_pair(pair_id: str, difficulty: str = "easy") -> Dict:
    """
    Generate a Yes/No numerical comparison pair.
    
    Args:
        pair_id: Unique identifier for this pair
        difficulty: 'easy', 'medium', or 'hard'
    
    Returns:
        dict with q1, q2, q1_answer, q2_answer, metadata
    """
    if difficulty == "easy":
        # Simple integer comparison
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        while a == b:
            b = random.randint(100, 999)
        
        # Determine which is larger
        if a > b:
            q1 = f"Is {a} larger than {b}?"
            q2 = f"Is {b} larger than {a}?"
            q1_answer = "Yes"
            q2_answer = "No"
        else:
            q1 = f"Is {a} larger than {b}?"
            q2 = f"Is {b} larger than {a}?"
            q1_answer = "No"
            q2_answer = "Yes"
        
        metadata = {
            "type": "integer_comparison",
            "values": {"a": a, "b": b},
            "larger": max(a, b)
        }
        
    elif difficulty == "medium":
        # Multiplication comparison
        a, b = random.randint(10, 50), random.randint(10, 50)
        c, d = random.randint(10, 50), random.randint(10, 50)
        
        prod1, prod2 = a * b, c * d
        
        # Ensure they're different
        while prod1 == prod2:
            c = random.randint(10, 50)
            d = random.randint(10, 50)
            prod2 = c * d
        
        if prod1 > prod2:
            q1 = f"Is {a} × {b} greater than {c} × {d}?"
            q2 = f"Is {c} × {d} greater than {a} × {b}?"
            q1_answer = "Yes"
            q2_answer = "No"
        else:
            q1 = f"Is {a} × {b} greater than {c} × {d}?"
            q2 = f"Is {c} × {d} greater than {a} × {b}?"
            q1_answer = "No"
            q2_answer = "Yes"
        
        metadata = {
            "type": "multiplication_comparison",
            "values": {"a": a, "b": b, "c": c, "d": d},
            "prod1": prod1,
            "prod2": prod2,
            "larger": f"{a} × {b}" if prod1 > prod2 else f"{c} × {d}"
        }
    
    else:  # hard
        # Power comparison (using smaller numbers)
        a = random.randint(2, 8)
        b = random.randint(2, 5)
        c = random.randint(2, 8)
        d = random.randint(2, 5)
        
        pow1, pow2 = a**b, c**d
        
        # Ensure they're different
        while pow1 == pow2:
            c = random.randint(2, 8)
            d = random.randint(2, 5)
            pow2 = c**d
        
        if pow1 > pow2:
            q1 = f"Is {a}^{b} greater than {c}^{d}?"
            q2 = f"Is {c}^{d} greater than {a}^{b}?"
            q1_answer = "Yes"
            q2_answer = "No"
        else:
            q1 = f"Is {a}^{b} greater than {c}^{d}?"
            q2 = f"Is {c}^{d} greater than {a}^{b}?"
            q1_answer = "No"
            q2_answer = "Yes"
        
        metadata = {
            "type": "power_comparison",
            "values": {"a": a, "b": b, "c": c, "d": d},
            "pow1": pow1,
            "pow2": pow2,
            "larger": f"{a}^{b}" if pow1 > pow2 else f"{c}^{d}"
        }
    
    return {
        "id": pair_id,
        "category": "numerical_comparison",
        "difficulty": difficulty,
        "q1": q1,
        "q2": q2,
        "q1_answer": q1_answer,
        "q2_answer": q2_answer,
        "metadata": metadata
    }


def generate_question_set(
    num_pairs: int = 50,
    difficulty_distribution: Dict[str, float] = None
) -> List[Dict]:
    """
    Generate a set of question pairs.
    
    Args:
        num_pairs: Total number of pairs to generate
        difficulty_distribution: Dict with 'easy', 'medium', 'hard' proportions
                                 (default: 40% easy, 40% medium, 20% hard)
    
    Returns:
        List of question pair dicts
    """
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    
    # Calculate counts
    num_easy = int(num_pairs * difficulty_distribution["easy"])
    num_medium = int(num_pairs * difficulty_distribution["medium"])
    num_hard = num_pairs - num_easy - num_medium  # Remaining
    
    pairs = []
    counter = 0
    
    # Generate easy pairs
    for _ in range(num_easy):
        pair = generate_numerical_pair(f"num_{counter:03d}", "easy")
        pairs.append(pair)
        counter += 1
    
    # Generate medium pairs
    for _ in range(num_medium):
        pair = generate_numerical_pair(f"num_{counter:03d}", "medium")
        pairs.append(pair)
        counter += 1
    
    # Generate hard pairs
    for _ in range(num_hard):
        pair = generate_numerical_pair(f"num_{counter:03d}", "hard")
        pairs.append(pair)
        counter += 1
    
    return pairs


def validate_question_pair(pair: Dict) -> List[str]:
    """
    Validate a question pair.
    
    Args:
        pair: Question pair dict
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['id', 'q1', 'q2', 'q1_answer', 'q2_answer']
    for field in required_fields:
        if field not in pair:
            errors.append(f"Missing required field: {field}")
    
    # Check questions are different
    if pair.get('q1') == pair.get('q2'):
        errors.append("Q1 and Q2 are identical")
    
    # Check answers are opposite (for Yes/No questions)
    q1_ans = pair.get('q1_answer', '').lower()
    q2_ans = pair.get('q2_answer', '').lower()
    
    if q1_ans in ['yes', 'no'] and q2_ans in ['yes', 'no']:
        if q1_ans == q2_ans:
            errors.append("Q1 and Q2 answers should be opposite")
    
    return errors

