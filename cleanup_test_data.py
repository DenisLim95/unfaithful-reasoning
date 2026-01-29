#!/usr/bin/env python3
"""
Cleanup script to remove all test data while preserving training data.

This resets your experiment to allow fresh test runs without affecting
your Phase 3 training data (activations, probes, etc.).
"""

import shutil
from pathlib import Path


def cleanup_test_data():
    """Remove all test-related data."""
    
    print("=" * 60)
    print("CLEANING UP TEST DATA")
    print("=" * 60)
    print()
    
    # Define what to remove
    test_dirs = [
        Path("data/test_activations"),
    ]
    
    test_files = [
        Path("data/raw/test_question_pairs.json"),
        Path("data/responses/test_responses.jsonl"),
        Path("data/processed/test_faithfulness_scores.csv"),
    ]
    
    # Remove test directories
    for dir_path in test_dirs:
        if dir_path.exists():
            print(f"✓ Removing directory: {dir_path}")
            shutil.rmtree(dir_path)
        else:
            print(f"  (not found, skipping): {dir_path}")
    
    print()
    
    # Remove test files
    for file_path in test_files:
        if file_path.exists():
            print(f"✓ Removing file: {file_path}")
            file_path.unlink()
        else:
            print(f"  (not found, skipping): {file_path}")
    
    print()
    print("=" * 60)
    print("WHAT'S PRESERVED (Training Data)")
    print("=" * 60)
    print()
    
    # Show what's kept
    preserved = {
        "Training activations": Path("data/activations"),
        "Trained probes": Path("results/probe_results"),
        "Training questions": Path("data/raw/question_pairs.json"),
        "Training responses": Path("data/responses/model_1.5B_responses.jsonl"),
        "Faithfulness scores": Path("data/processed/faithfulness_scores.csv"),
    }
    
    for name, path in preserved.items():
        if path.exists():
            if path.is_dir():
                num_files = len(list(path.glob("*")))
                print(f"✓ {name}: {path} ({num_files} files)")
            else:
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"⚠ {name}: {path} (not found)")
    
    print()
    print("=" * 60)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 60)
    print()
    print("You can now run a fresh test:")
    print("  python test_probe_on_new_data.py --num-questions 200")
    print()


if __name__ == "__main__":
    cleanup_test_data()



