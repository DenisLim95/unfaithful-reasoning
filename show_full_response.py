#!/usr/bin/env python3
"""
View full model output for a specific pair ID.
"""
import jsonlines
import json
import sys


def show_full_response(pair_id):
    """Display complete model response for a given pair."""
    
    # Load question
    with open('data/raw/question_pairs.json') as f:
        questions = {p['id']: p for p in json.load(f)['pairs']}
    
    if pair_id not in questions:
        print(f"Error: Pair ID '{pair_id}' not found")
        print(f"Available IDs: num_000 to num_049")
        return
    
    q = questions[pair_id]
    
    # Load responses
    responses = {}
    with jsonlines.open('data/responses/model_1.5B_responses.jsonl') as reader:
        for resp in reader:
            if resp['pair_id'] == pair_id:
                responses[resp['variant']] = resp
    
    if not responses:
        print(f"Error: No responses found for pair '{pair_id}'")
        return
    
    # Display
    print("=" * 100)
    print(f"FULL MODEL OUTPUT FOR: {pair_id}")
    print("=" * 100)
    print()
    
    # Metadata
    print(f"Category: {q['category']}")
    print(f"Difficulty: {q['difficulty']}")
    if 'metadata' in q:
        print(f"Metadata: {q['metadata']}")
    print()
    
    # Q1
    print("=" * 100)
    print("QUESTION 1")
    print("=" * 100)
    print()
    print(f"Question: {q['q1']}")
    print(f"Expected Answer: {q['q1_answer']}")
    print()
    
    if 'q1' in responses:
        r1 = responses['q1']
        
        print("-" * 100)
        print("THINK SECTION (Model's Reasoning):")
        print("-" * 100)
        print(r1['think_section'])
        print()
        
        print("-" * 100)
        print("FINAL ANSWER (After </think>):")
        print("-" * 100)
        print(r1['final_answer'])
        print()
    else:
        print("No response found for Q1")
        print()
    
    # Q2
    print()
    print("=" * 100)
    print("QUESTION 2")
    print("=" * 100)
    print()
    print(f"Question: {q['q2']}")
    print(f"Expected Answer: {q['q2_answer']}")
    print()
    
    if 'q2' in responses:
        r2 = responses['q2']
        
        print("-" * 100)
        print("THINK SECTION (Model's Reasoning):")
        print("-" * 100)
        print(r2['think_section'])
        print()
        
        print("-" * 100)
        print("FINAL ANSWER (After </think>):")
        print("-" * 100)
        print(r2['final_answer'])
        print()
    else:
        print("No response found for Q2")
        print()
    
    print("=" * 100)


def list_available_pairs():
    """Show all available pair IDs."""
    try:
        with open('data/raw/question_pairs.json') as f:
            pairs = json.load(f)['pairs']
        
        print("Available pair IDs:")
        for p in pairs:
            print(f"  {p['id']}: {p['q1'][:60]}...")
    except FileNotFoundError:
        print("Error: data/raw/question_pairs.json not found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_full_response.py <pair_id>")
        print()
        print("Examples:")
        print("  python show_full_response.py num_000")
        print("  python show_full_response.py num_010")
        print("  python show_full_response.py num_042")
        print()
        print("To see all available IDs:")
        print("  python show_full_response.py --list")
        sys.exit(1)
    
    pair_id = sys.argv[1]
    
    if pair_id == "--list":
        list_available_pairs()
    else:
        show_full_response(pair_id)

