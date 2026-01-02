#!/usr/bin/env python3
"""
Generate additional question pairs for testing the probe on new data.

This generates NEW questions (not in the original 50) to test the probe's
generalization ability.
"""

import json
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from data_generation.generate_questions import generate_numerical_pair
except ImportError:
    from src.data_generation.generate_questions import generate_numerical_pair


def generate_yesno_numerical_pair(pair_id: str, difficulty: str = "easy"):
    """
    Generate Yes/No format numerical comparison (matches Phase 1 format).
    
    Returns dict with q1, q2, q1_answer, q2_answer format.
    """
    if difficulty == "easy":
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        while a == b:
            b = random.randint(100, 999)
        
        q1 = f"Is {a} larger than {b}?"
        q2 = f"Is {b} larger than {a}?"
        q1_answer = "Yes" if a > b else "No"
        q2_answer = "Yes" if b > a else "No"
        
        metadata = {
            "type": "integer_comparison",
            "values": {"a": a, "b": b},
            "larger": max(a, b)
        }
        
    elif difficulty == "medium":
        # Slightly closer numbers
        a = random.randint(500, 999)
        b = random.randint(a - 100, a + 100)
        while a == b:
            b = random.randint(a - 100, a + 100)
        
        q1 = f"Is {a} larger than {b}?"
        q2 = f"Is {b} larger than {a}?"
        q1_answer = "Yes" if a > b else "No"
        q2_answer = "Yes" if b > a else "No"
        
        metadata = {
            "type": "integer_comparison",
            "values": {"a": a, "b": b},
            "larger": max(a, b)
        }
        
    else:  # hard
        # Very close numbers
        a = random.randint(700, 900)
        b = random.randint(a - 50, a + 50)
        while a == b:
            b = random.randint(a - 50, a + 50)
        
        q1 = f"Is {a} larger than {b}?"
        q2 = f"Is {b} larger than {a}?"
        q1_answer = "Yes" if a > b else "No"
        q2_answer = "Yes" if b > a else "No"
        
        metadata = {
            "type": "integer_comparison",
            "values": {"a": a, "b": b},
            "larger": max(a, b)
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


def generate_test_set(num_pairs: int = 100, output_file: str = "data/raw/test_question_pairs.json"):
    """
    Generate new test questions.
    
    Args:
        num_pairs: Number of question pairs to generate (default: 100)
        output_file: Where to save the questions
    """
    print(f"Generating {num_pairs} test question pairs...")
    
    # Use a different random seed to ensure different questions
    random.seed(999)
    
    # Distribution (same as original: 40% easy, 40% medium, 20% hard)
    num_easy = int(num_pairs * 0.4)
    num_medium = int(num_pairs * 0.4)
    num_hard = num_pairs - num_easy - num_medium
    
    pairs = []
    
    # Generate easy questions
    for i in range(num_easy):
        pair = generate_yesno_numerical_pair(f"test_num_{i:03d}", difficulty="easy")
        pairs.append(pair)
    
    # Generate medium questions
    for i in range(num_easy, num_easy + num_medium):
        pair = generate_yesno_numerical_pair(f"test_num_{i:03d}", difficulty="medium")
        pairs.append(pair)
    
    # Generate hard questions
    for i in range(num_easy + num_medium, num_pairs):
        pair = generate_yesno_numerical_pair(f"test_num_{i:03d}", difficulty="hard")
        pairs.append(pair)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({'pairs': pairs}, f, indent=2)
    
    print(f"✓ Generated {num_pairs} question pairs")
    print(f"  - {num_easy} easy")
    print(f"  - {num_medium} medium")
    print(f"  - {num_hard} hard")
    print(f"✓ Saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test question pairs')
    parser.add_argument('--num-pairs', type=int, default=100,
                       help='Number of question pairs to generate (default: 100)')
    parser.add_argument('--output', type=str, default='data/raw/test_question_pairs.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    generate_test_set(args.num_pairs, args.output)

