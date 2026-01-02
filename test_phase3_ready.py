#!/usr/bin/env python3
"""
Quick test to verify Phase 3 is ready to run.
Run this before starting Phase 3 to catch any issues.
"""

import sys
from pathlib import Path

def test_phase3_ready():
    """Run all pre-flight checks for Phase 3."""
    
    print("=" * 60)
    print("PHASE 3 PRE-FLIGHT CHECKS")
    print("=" * 60)
    print()
    
    all_pass = True
    
    # Test 1: Python version
    print("[1/8] Checking Python version...")
    if sys.version_info >= (3, 10):
        print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.10+)")
        all_pass = False
    
    # Test 2: Critical packages
    print("\n[2/8] Checking packages...")
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'pandas': 'Pandas',
        'transformer_lens': 'TransformerLens',
        'typeguard': 'TypeGuard'
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - Run: pip install {module.replace('_', '-')}")
            all_pass = False
    
    # Test 3: CUDA
    print("\n[3/8] Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available")
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  No CUDA (will use CPU - very slow!)")
    except:
        print("   ✗ Cannot check CUDA")
        all_pass = False
    
    # Test 4: Phase 2 outputs
    print("\n[4/8] Checking Phase 2 outputs...")
    if Path("data/responses/model_1.5B_responses.jsonl").exists():
        print("   ✓ model_1.5B_responses.jsonl")
    else:
        print("   ✗ model_1.5B_responses.jsonl missing")
        all_pass = False
    
    if Path("data/processed/faithfulness_scores.csv").exists():
        print("   ✓ faithfulness_scores.csv")
    else:
        print("   ✗ faithfulness_scores.csv missing")
        all_pass = False
    
    # Test 5: Contracts import
    print("\n[5/8] Checking Phase 3 contracts...")
    try:
        from src.mechanistic.contracts import (
            Phase3Config, ActivationCache, 
            MIN_FAITHFUL_SAMPLES, MIN_UNFAITHFUL_SAMPLES
        )
        print("   ✓ contracts.py imports OK")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        all_pass = False
    
    # Test 6: Main script imports
    print("\n[6/8] Checking cache_activations.py...")
    try:
        from src.mechanistic.cache_activations import cache_activations
        print("   ✓ cache_activations.py imports OK")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        all_pass = False
    
    # Test 7: Probe script imports
    print("\n[7/8] Checking train_probes.py...")
    try:
        from src.mechanistic.train_probes import train_all_probes
        print("   ✓ train_probes.py imports OK")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        all_pass = False
    
    # Test 8: Disk space
    print("\n[8/8] Checking disk space...")
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    if free_gb >= 5:
        print(f"   ✓ {free_gb:.1f} GB free")
    else:
        print(f"   ⚠️  Only {free_gb:.1f} GB free (recommend 5+ GB)")
    
    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL CHECKS PASSED")
        print("\nYou're ready to run Phase 3!")
        print("  bash run_phase3.sh")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nFix the issues above before running Phase 3.")
        return 1

if __name__ == "__main__":
    sys.exit(test_phase3_ready())

