#!/usr/bin/env python3
"""
View complete model responses for specific questions.
Useful for debugging extraction and verifying model behavior.
"""

import json
import sys
from pathlib import Path


def view_responses(search_term=None, show_all=False, limit=5):
    """
    View complete model responses.
    
    Args:
        search_term: Search for questions containing this text (e.g., "900" or "795")
        show_all: If True, show all responses (ignores limit)
        limit: Max number of responses to show
    """
    
    responses_file = Path("data/responses/test_responses.jsonl")
    
    if not responses_file.exists():
        print(f"❌ {responses_file} not found!")
        print("Run 'python test_probe_on_new_data.py' first to generate responses.")
        return
    
    # Load all responses
    responses = []
    with open(responses_file, 'r') as f:
        for line in f:
            responses.append(json.loads(line))
    
    print(f"✓ Loaded {len(responses)} responses")
    
    # Filter if search term provided
    if search_term:
        responses = [r for r in responses if search_term.lower() in r['question'].lower()]
        print(f"✓ Found {len(responses)} matching '{search_term}'")
    
    # Limit results
    if not show_all:
        responses = responses[:limit]
    
    # Display responses
    print("\n" + "="*80)
    print("MODEL RESPONSES")
    print("="*80)
    
    for i, resp in enumerate(responses):
        print(f"\n{'─'*80}")
        print(f"Response {i+1}")
        print(f"{'─'*80}")
        
        # Question and metadata
        print(f"\n📝 Question: {resp['question']}")
        print(f"   Pair ID: {resp['pair_id']}")
        print(f"   Variant: {resp['variant']}")
        
        # Expected vs extracted
        expected = resp['expected_answer']
        extracted = resp['extracted_answer']
        match = "✓" if expected == extracted else "✗"
        
        print(f"\n   Expected Answer: {expected}")
        print(f"   Extracted Answer: {extracted} {match}")
        
        # Full response
        print(f"\n📄 Complete Response:")
        print(f"   {'┌' + '─'*76 + '┐'}")
        
        response_text = resp['response']
        
        # Split by lines and indent
        lines = response_text.split('\n')
        for line in lines:
            # Wrap long lines
            if len(line) > 74:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 74:
                        current_line += (word + " ")
                    else:
                        print(f"   │ {current_line:<74} │")
                        current_line = word + " "
                if current_line:
                    print(f"   │ {current_line:<74} │")
            else:
                print(f"   │ {line:<74} │")
        
        print(f"   {'└' + '─'*76 + '┘'}")
        
        # Analysis
        print(f"\n💡 Analysis:")
        if extracted == "Unknown":
            print(f"   ⚠️  Failed to extract answer")
            print(f"   → Check if response contains 'Final Answer:' or clear Yes/No")
        elif expected != extracted:
            print(f"   ✗ Wrong extraction")
            print(f"   → Model may have given wrong answer OR extraction failed")
        else:
            print(f"   ✓ Correct extraction")
            print(f"   → Model answered correctly and extraction worked")
        
        # Show key patterns
        if "final answer:" in response_text.lower():
            import re
            match = re.search(r'final\s+answer:\s*(\w+)', response_text.lower())
            if match:
                print(f"   → Found 'Final Answer: {match.group(1)}'")
        
        if "</think>" in response_text.lower():
            print(f"   → Contains </think> tag")


def show_usage():
    print("""
Usage: python view_response.py [OPTIONS]

Options:
  (no args)              Show first 5 responses
  "search term"          Search for questions containing text
  --all                  Show all responses (no limit)
  --limit N              Show N responses (default: 5)

Examples:
  python view_response.py                    # First 5 responses
  python view_response.py "900"              # Questions about 900
  python view_response.py "795 larger"       # Specific comparison
  python view_response.py --all              # All responses
  python view_response.py "900" --limit 10   # First 10 matching "900"
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
        sys.exit(0)
    
    # Parse arguments
    search_term = None
    show_all = False
    limit = 5
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--all":
            show_all = True
        elif arg == "--limit":
            i += 1
            limit = int(sys.argv[i])
        elif not arg.startswith("--"):
            search_term = arg
        i += 1
    
    view_responses(search_term, show_all, limit)

