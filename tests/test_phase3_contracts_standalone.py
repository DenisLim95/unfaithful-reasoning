"""
Phase 3 Contract Tests (Standalone - No pytest required)
Demonstrates that Phase 3 contracts enforce specification.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mechanistic.contracts import (
    ActivationCache,
    ProbeResult,
    Phase3Config,
    PHASE3_LAYERS,
    MIN_FAITHFUL_SAMPLES,
    MIN_UNFAITHFUL_SAMPLES,
)


def test_activation_cache_contract():
    """Test that ActivationCache enforces Phase 3 specification."""
    
    print("\n" + "="*60)
    print("TEST: ActivationCache Contract Enforcement")
    print("="*60)
    
    # Test 1: Valid cache passes
    print("\n[Test 1] Valid activation cache...")
    try:
        cache = ActivationCache(
            faithful=torch.randn(20, 1536, dtype=torch.float16),
            unfaithful=torch.randn(15, 1536, dtype=torch.float16),
            layer=12
        )
        print(f"✓ PASS: Valid cache accepted")
        print(f"  - n_faithful={cache.n_faithful}, n_unfaithful={cache.n_unfaithful}")
        print(f"  - d_model={cache.d_model}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    
    # Test 2: Reject 3D tensors (not mean-pooled)
    print("\n[Test 2] Reject 3D tensors (must be mean-pooled)...")
    try:
        cache = ActivationCache(
            faithful=torch.randn(20, 100, 1536),  # Has sequence dim!
            unfaithful=torch.randn(15, 1536),
            layer=12
        )
        print(f"✗ FAIL: 3D tensor was accepted (should be rejected)")
        return False
    except ValueError as e:
        if "must be 2D" in str(e) and "mean-pooled" in str(e):
            print(f"✓ PASS: 3D tensor correctly rejected")
            print(f"  Error: {str(e)[:100]}...")
        else:
            print(f"✗ FAIL: Wrong error: {e}")
            return False
    
    # Test 3: Reject too few samples
    print("\n[Test 3] Reject too few faithful samples...")
    try:
        cache = ActivationCache(
            faithful=torch.randn(5, 1536),  # Too few!
            unfaithful=torch.randn(15, 1536),
            layer=12
        )
        print(f"✗ FAIL: Too few samples accepted (should be rejected)")
        return False
    except ValueError as e:
        if f"at least {MIN_FAITHFUL_SAMPLES}" in str(e):
            print(f"✓ PASS: Too few samples correctly rejected")
            print(f"  Error: {str(e)[:100]}...")
        else:
            print(f"✗ FAIL: Wrong error: {e}")
            return False
    
    # Test 4: Reject d_model mismatch
    print("\n[Test 4] Reject d_model mismatch...")
    try:
        cache = ActivationCache(
            faithful=torch.randn(20, 1536),
            unfaithful=torch.randn(15, 2048),  # Different d_model!
            layer=12
        )
        print(f"✗ FAIL: d_model mismatch accepted (should be rejected)")
        return False
    except ValueError as e:
        if "d_model mismatch" in str(e):
            print(f"✓ PASS: d_model mismatch correctly rejected")
            print(f"  Error: {str(e)[:100]}...")
        else:
            print(f"✗ FAIL: Wrong error: {e}")
            return False
    
    # Test 5: Reject unsupported layer
    print("\n[Test 5] Reject unsupported layer...")
    try:
        cache = ActivationCache(
            faithful=torch.randn(20, 1536),
            unfaithful=torch.randn(15, 1536),
            layer=10  # Not in PHASE3_LAYERS!
        )
        print(f"✗ FAIL: Unsupported layer accepted (should be rejected)")
        return False
    except ValueError as e:
        if "Phase 3" in str(e) and "not supported" in str(e):
            print(f"✓ PASS: Unsupported layer correctly rejected")
            print(f"  Error: {str(e)[:100]}...")
        else:
            print(f"✗ FAIL: Wrong error: {e}")
            return False
    
    print("\n✅ All ActivationCache contract tests passed!")
    return True


def test_probe_result_contract():
    """Test that ProbeResult enforces Phase 3 specification."""
    
    print("\n" + "="*60)
    print("TEST: ProbeResult Contract Enforcement")
    print("="*60)
    
    # Test 1: Valid result passes
    print("\n[Test 1] Valid probe result...")
    try:
        probe = torch.nn.Linear(1536, 1)
        result = ProbeResult(
            layer="layer_12",
            accuracy=0.75,
            auc=0.82,
            probe=probe,
            direction=torch.randn(1536)
        )
        print(f"✓ PASS: Valid result accepted")
        print(f"  - accuracy={result.accuracy}, auc={result.auc}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    
    # Test 2: Reject accuracy out of range
    print("\n[Test 2] Reject accuracy > 1...")
    try:
        probe = torch.nn.Linear(1536, 1)
        result = ProbeResult(
            layer="layer_12",
            accuracy=1.5,  # Out of range!
            auc=0.82,
            probe=probe,
            direction=torch.randn(1536)
        )
        print(f"✗ FAIL: Out of range accuracy accepted (should be rejected)")
        return False
    except ValueError as e:
        if "accuracy must be in [0, 1]" in str(e):
            print(f"✓ PASS: Out of range accuracy correctly rejected")
            print(f"  Error: {str(e)[:100]}...")
        else:
            print(f"✗ FAIL: Wrong error: {e}")
            return False
    
    # Test 3: Reject 2D direction
    print("\n[Test 3] Reject 2D direction (must be 1D)...")
    try:
        probe = torch.nn.Linear(1536, 1)
        result = ProbeResult(
            layer="layer_12",
            accuracy=0.75,
            auc=0.82,
            probe=probe,
            direction=torch.randn(10, 1536)  # 2D!
        )
        print(f"✗ FAIL: 2D direction accepted (should be rejected)")
        return False
    except ValueError as e:
        if "direction must be 1D" in str(e):
            print(f"✓ PASS: 2D direction correctly rejected")
            print(f"  Error: {str(e)[:100]}...")
        else:
            print(f"✗ FAIL: Wrong error: {e}")
            return False
    
    print("\n✅ All ProbeResult contract tests passed!")
    return True


def test_phase3_config():
    """Test Phase 3 configuration."""
    
    print("\n" + "="*60)
    print("TEST: Phase3Config")
    print("="*60)
    
    # Test 1: Default config
    print("\n[Test 1] Default config uses Phase 3 specification...")
    config = Phase3Config()
    
    checks = [
        (config.layers == PHASE3_LAYERS, f"layers={config.layers}"),
        (config.num_epochs == 50, f"num_epochs={config.num_epochs}"),
        (config.learning_rate == 1e-3, f"learning_rate={config.learning_rate}"),
        (config.train_test_split == 0.2, f"train_test_split={config.train_test_split}"),
    ]
    
    all_pass = True
    for passed, desc in checks:
        if passed:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} (wrong value!)")
            all_pass = False
    
    if not all_pass:
        return False
    
    # Test 2: Reject unsupported layers
    print("\n[Test 2] Reject unsupported layers...")
    try:
        config = Phase3Config(layers=[1, 2, 3])
        print(f"✗ FAIL: Unsupported layers accepted")
        return False
    except ValueError as e:
        if "Phase 3 does not support" in str(e):
            print(f"✓ PASS: Unsupported layers correctly rejected")
            print(f"  Error: {str(e)[:100]}...")
        else:
            print(f"✗ FAIL: Wrong error: {e}")
            return False
    
    print("\n✅ All Phase3Config tests passed!")
    return True


def test_phase3_boundaries():
    """Test that Phase 3 respects its boundaries."""
    
    print("\n" + "="*60)
    print("TEST: Phase 3 Boundaries")
    print("="*60)
    
    print("\n[Test 1] Phase 3 supports exactly 4 layers...")
    if PHASE3_LAYERS == [6, 12, 18, 24] and len(PHASE3_LAYERS) == 4:
        print(f"✓ PASS: PHASE3_LAYERS = {PHASE3_LAYERS}")
    else:
        print(f"✗ FAIL: PHASE3_LAYERS = {PHASE3_LAYERS} (expected [6, 12, 18, 24])")
        return False
    
    print("\n[Test 2] Phase 3 has no attention analysis...")
    try:
        from mechanistic import attention_analysis
        print(f"✗ FAIL: attention_analysis module exists (shouldn't in Phase 3)")
        return False
    except ImportError:
        print(f"✓ PASS: attention_analysis correctly not in Phase 3")
    
    print("\n[Test 3] Phase 3 has no report generation...")
    try:
        from mechanistic import report_generator
        print(f"✗ FAIL: report_generator module exists (shouldn't in Phase 3)")
        return False
    except ImportError:
        print(f"✓ PASS: report_generator correctly not in Phase 3")
    
    print("\n✅ All Phase 3 boundary tests passed!")
    return True


def main():
    """Run all Phase 3 contract tests."""
    
    print("="*60)
    print("PHASE 3 CONTRACT TESTS")
    print("="*60)
    print("\nThese tests verify that Phase 3 implementation enforces")
    print("the specification from phased_implementation_plan.md")
    
    tests = [
        ("ActivationCache Contract", test_activation_cache_contract),
        ("ProbeResult Contract", test_probe_result_contract),
        ("Phase3Config", test_phase3_config),
        ("Phase 3 Boundaries", test_phase3_boundaries),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} raised exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅✅✅ ALL PHASE 3 CONTRACT TESTS PASSED ✅✅✅")
        print("="*60)
        print("\nPhase 3 implementation correctly enforces specification!")
        return 0
    else:
        print("❌ SOME PHASE 3 CONTRACT TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

