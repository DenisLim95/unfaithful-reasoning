#!/usr/bin/env python3
"""
Verify if there's data leakage between training and test sets.

Run this on your pod to check if test_probe_on_new_data.py
accidentally tested on the training data.
"""

import torch
import numpy as np
from pathlib import Path


def check_data_leakage():
    """Check if test activations are identical to training activations."""
    
    print("="*60)
    print("DATA LEAKAGE VERIFICATION")
    print("="*60)
    
    # Check if both directories exist
    train_dir = Path("data/activations")
    test_dir = Path("data/test_activations")
    
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        return
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print(f"   This means you never ran the test data caching!")
        print(f"   You probably tested on TRAINING data by accident.")
        return
    
    print(f"\n1. Checking data sizes...")
    print(f"-" * 60)
    
    for layer in [6, 12, 18, 24]:
        train_file = train_dir / f"layer_{layer}_activations.pt"
        test_file = test_dir / f"layer_{layer}_activations.pt"
        
        if not train_file.exists():
            print(f"  Layer {layer}: Training file missing")
            continue
        
        if not test_file.exists():
            print(f"  Layer {layer}: Test file missing")
            continue
        
        train_data = torch.load(train_file)
        test_data = torch.load(test_file)
        
        train_f_count = train_data['faithful'].shape[0]
        train_u_count = train_data['unfaithful'].shape[0]
        test_f_count = test_data['faithful'].shape[0]
        test_u_count = test_data['unfaithful'].shape[0]
        
        print(f"\n  Layer {layer}:")
        print(f"    Training: {train_f_count} faithful, {train_u_count} unfaithful")
        print(f"    Test:     {test_f_count} faithful, {test_u_count} unfaithful")
        
        # Red flag 1: Exact same counts
        if train_f_count == test_f_count and train_u_count == test_u_count:
            print(f"    ⚠️  IDENTICAL COUNTS - High risk of data leakage!")
        
        # Red flag 2: Test is much smaller (might be training split)
        if test_f_count < train_f_count and test_u_count < train_u_count:
            print(f"    ⚠️  Test set smaller - might be training test split")
    
    # Detailed check on layer 24
    print(f"\n2. Checking for identical samples (Layer 24)...")
    print(f"-" * 60)
    
    train_file = train_dir / "layer_24_activations.pt"
    test_file = test_dir / "layer_24_activations.pt"
    
    if train_file.exists() and test_file.exists():
        train_data = torch.load(train_file)
        test_data = torch.load(test_file)
        
        train_f = train_data['faithful'].numpy()
        test_f = test_data['faithful'].numpy()
        
        # Check if ANY test samples are in training set
        matches_found = 0
        samples_to_check = min(10, len(test_f))
        
        print(f"  Checking first {samples_to_check} test samples...")
        
        for i in range(samples_to_check):
            test_sample = test_f[i]
            
            # Check against all training samples
            for j in range(len(train_f)):
                if np.allclose(test_sample, train_f[j], atol=1e-6):
                    matches_found += 1
                    print(f"    Sample {i}: ⚠️  MATCHES training sample {j}")
                    break
            else:
                print(f"    Sample {i}: ✓ Unique")
        
        print(f"\n  Summary:")
        print(f"    Matches found: {matches_found}/{samples_to_check}")
        
        if matches_found > 0:
            print(f"    ❌ DATA LEAKAGE DETECTED!")
            print(f"    You tested on training data.")
        else:
            print(f"    ✓ No exact matches - test data appears to be different")
    
    # Check timestamps
    print(f"\n3. Checking file timestamps...")
    print(f"-" * 60)
    
    import os
    from datetime import datetime
    
    train_file = train_dir / "layer_24_activations.pt"
    test_file = test_dir / "layer_24_activations.pt"
    
    if train_file.exists():
        train_time = os.path.getmtime(train_file)
        train_dt = datetime.fromtimestamp(train_time)
        print(f"  Training data created: {train_dt}")
    
    if test_file.exists():
        test_time = os.path.getmtime(test_file)
        test_dt = datetime.fromtimestamp(test_time)
        print(f"  Test data created:     {test_dt}")
        
        if train_file.exists():
            if abs(train_time - test_time) < 60:  # Within 1 minute
                print(f"  ⚠️  Files created within 1 minute - suspicious!")
            elif test_time < train_time:
                print(f"  ⚠️  Test data OLDER than training - definitely wrong!")
    
    print(f"\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    if not test_dir.exists():
        print("\n❌ PROBLEM IDENTIFIED:")
        print("   - No test_activations/ directory found")
        print("   - You likely ran --test-only without generating new test data")
        print("   - This means you tested on the TRAINING data")
        print("\n✅ FIX:")
        print("   Run the full pipeline to generate NEW test data:")
        print("   python test_probe_on_new_data.py --num-questions 100")
    else:
        print("\n✓ Test directory exists")
        print("  Check the detailed results above for data leakage")


if __name__ == "__main__":
    check_data_leakage()

