"""
Phase 1: Question Generation (Yes/No Format)
Generates question pairs with Yes/No answers for reliable extraction.
"""
import json
import random
from pathlib import Path
from typing import Dict, List


def generate_numerical_pair_yesno(pair_id: str, difficulty: str = "easy") -> Dict:
    """Generate a Yes/No numerical comparison pair."""
    
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


def generate_all_questions_yesno(
    num_pairs: int = 50,
    output_path: str = "data/raw/question_pairs.json"
):
    """
    Generate question pairs in Yes/No format for Phase 1.
    
    Distribution: 20 easy, 20 medium, 10 hard
    """
    pairs = []
    
    # Distribution: 20 easy, 20 medium, 10 hard
    difficulties = ["easy"] * 20 + ["medium"] * 20 + ["hard"] * 10
    
    for i, difficulty in enumerate(difficulties):
        pair = generate_numerical_pair_yesno(f"num_{i:03d}", difficulty)
        pairs.append(pair)
    
    # Save
    output = {"pairs": pairs}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Generated {len(pairs)} question pairs (Yes/No format)")
    print(f"✓ Distribution: 20 easy, 20 medium, 10 hard")
    print(f"✓ Saved to {output_path}")
    
    # Show examples
    print(f"\n=== Example Questions ===")
    for diff in ["easy", "medium", "hard"]:
        example = next(p for p in pairs if p["difficulty"] == diff)
        print(f"\n{diff.upper()}:")
        print(f"  Q1: {example['q1']}")
        print(f"  Q1 Answer: {example['q1_answer']}")
        print(f"  Q2: {example['q2']}")
        print(f"  Q2 Answer: {example['q2_answer']}")
    
    return pairs


if __name__ == "__main__":
    pairs = generate_all_questions_yesno()


