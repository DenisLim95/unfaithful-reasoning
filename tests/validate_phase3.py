"""
Phase 3 Task 3.4: Automated Validation Script

Validates Phase 3 deliverables against specification.
This script encodes Phase 3 acceptance criteria as executable tests.
"""

import torch
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mechanistic.contracts import (
    ActivationCache,
    ProbeResult,
    PHASE3_LAYERS,
    MIN_FAITHFUL_SAMPLES,
    MIN_UNFAITHFUL_SAMPLES
)

# Import LinearProbe so PyTorch can deserialize it
try:
    from mechanistic.train_probes import LinearProbe
except ImportError:
    # If import fails, try alternative paths
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    try:
        from src.mechanistic.train_probes import LinearProbe
    except ImportError:
        pass  # Will be caught during validation


def validate_activations(
    layers: List[int] = None,
    activations_dir: str = "data/activations"
) -> Tuple[bool, List[str]]:
    """
    Validate Phase 3 Data Contract 1: Activation Cache Files
    
    Acceptance criteria (from phased_implementation_plan.md lines 1428-1437):
    1. Activation files exist for all 4 layers
    2. Each file has 'faithful' and 'unfaithful' tensors
    3. Minimum 10 examples in each class
    4. d_model consistent across layers
    5. Tensors are 2D (mean-pooled)
    6. dtype is float32 or float16
    """
    if layers is None:
        layers = PHASE3_LAYERS
    
    errors = []
    d_models = []
    
    print("\n" + "=" * 60)
    print("VALIDATING ACTIVATION CACHES (Data Contract 1)")
    print("=" * 60)
    
    for layer in layers:
        layer_name = f"layer_{layer}"
        file_path = Path(activations_dir) / f"{layer_name}_activations.pt"
        
        # Check 1: File exists
        if not file_path.exists():
            errors.append(f"Missing activation file: {file_path}")
            continue
        
        # Check 2: Load and validate structure
        try:
            data = torch.load(file_path)
        except Exception as e:
            errors.append(f"{layer_name}: Error loading file: {e}")
            continue
        
        # Check 3: Has required keys
        if 'faithful' not in data:
            errors.append(f"{layer_name}: Missing 'faithful' key")
            continue
        
        if 'unfaithful' not in data:
            errors.append(f"{layer_name}: Missing 'unfaithful' key")
            continue
        
        # Validate using ActivationCache contract
        try:
            cache = ActivationCache(
                faithful=data['faithful'],
                unfaithful=data['unfaithful'],
                layer=layer
            )
            
            # Track d_model for consistency check
            d_models.append(cache.d_model)
            
            print(f"✓ {layer_name}: {cache.n_faithful} faithful, "
                  f"{cache.n_unfaithful} unfaithful, d_model={cache.d_model}")
            
        except ValueError as e:
            errors.append(f"{layer_name}: {str(e)}")
            continue
    
    # Check 4: d_model consistent across layers
    if len(set(d_models)) > 1:
        errors.append(f"d_model inconsistent across layers: {d_models}")
    elif len(d_models) > 0:
        print(f"\n✓ d_model consistent across all layers: {d_models[0]}")
    
    return len(errors) == 0, errors


def validate_probe_results(
    results_file: str = "results/probe_results/all_probe_results.pt"
) -> Tuple[bool, List[str]]:
    """
    Validate Phase 3 Data Contract 2: Probe Results
    
    Acceptance criteria (from phased_implementation_plan.md lines 1428-1437):
    4. Probe results file exists
    5. Results for all 4 layers present
    6. All accuracy/auc values in [0, 1]
    7. At least one layer has accuracy > 0.55
    """
    errors = []
    
    print("\n" + "=" * 60)
    print("VALIDATING PROBE RESULTS (Data Contract 2)")
    print("=" * 60)
    
    # Check 1: File exists
    if not Path(results_file).exists():
        return False, [f"Probe results file not found: {results_file}"]
    
    # Check 2: Load results
    try:
        # Add project root to path for module imports
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        results = torch.load(results_file, weights_only=False)
    except Exception as e:
        return False, [f"Error loading results: {e}"]
    
    # Check 3: All layers present
    expected_layers = [f"layer_{l}" for l in PHASE3_LAYERS]
    for layer in expected_layers:
        if layer not in results:
            errors.append(f"Missing results for {layer}")
    
    if len(results) != len(PHASE3_LAYERS):
        errors.append(f"Expected {len(PHASE3_LAYERS)} layers, got {len(results)}")
    
    # Check 4: Validate each layer's results
    best_acc = 0.0
    best_layer = None
    
    for layer_name, result in results.items():
        # Validate using ProbeResult contract
        try:
            # Note: We can't re-create ProbeResult easily (needs probe object)
            # So we check fields directly
            required = ['layer', 'accuracy', 'auc', 'probe', 'direction']
            for field in required:
                if field not in result.__dict__:
                    errors.append(f"{layer_name}: Missing field '{field}'")
            
            acc = result.accuracy
            auc = result.auc
            
            # Check ranges
            if not (0 <= acc <= 1):
                errors.append(f"{layer_name}: Accuracy {acc} not in [0, 1]")
            
            if not (0 <= auc <= 1):
                errors.append(f"{layer_name}: AUC {auc} not in [0, 1]")
            
            # Check direction is 1D
            if result.direction.ndim != 1:
                errors.append(f"{layer_name}: Direction must be 1D, got shape {result.direction.shape}")
            
            # Track best
            if acc > best_acc:
                best_acc = acc
                best_layer = layer_name
            
            print(f"✓ {layer_name}: accuracy={acc:.3f}, auc={auc:.3f}, "
                  f"direction_dim={len(result.direction)}")
            
        except Exception as e:
            errors.append(f"{layer_name}: Validation error: {e}")
    
    # Check 5: At least one layer beats threshold
    print(f"\n{'='*60}")
    print(f"Best layer: {best_layer} with accuracy={best_acc:.3f}")
    
    if best_acc > 0.55:
        print(f"✓ Exceeds Phase 3 threshold (>0.55)")
    else:
        errors.append(f"No layer exceeds 0.55 accuracy (best: {best_acc:.3f})")
        print(f"✗ Below Phase 3 threshold (≤0.55)")
        print(f"  Note: This may indicate a null result (no linear direction)")
    
    return len(errors) == 0, errors


def validate_outputs_exist() -> Tuple[bool, List[str]]:
    """
    Validate Phase 3 output files exist.
    
    Phase 3 must produce:
    - 4 activation cache files
    - 1 probe results file
    - 1 probe performance plot
    """
    errors = []
    
    print("\n" + "=" * 60)
    print("VALIDATING PHASE 3 OUTPUT FILES")
    print("=" * 60)
    
    # Check activation files
    for layer in PHASE3_LAYERS:
        path = Path(f"data/activations/layer_{layer}_activations.pt")
        if path.exists():
            print(f"✓ {path}")
        else:
            errors.append(f"Missing: {path}")
            print(f"✗ Missing: {path}")
    
    # Check probe results file
    results_path = Path("results/probe_results/all_probe_results.pt")
    if results_path.exists():
        print(f"✓ {results_path}")
    else:
        errors.append(f"Missing: {results_path}")
        print(f"✗ Missing: {results_path}")
    
    # Check plot
    plot_path = Path("results/probe_results/probe_performance.png")
    if plot_path.exists():
        print(f"✓ {plot_path}")
    else:
        errors.append(f"Missing: {plot_path}")
        print(f"✗ Missing: {plot_path}")
    
    return len(errors) == 0, errors


def main():
    """
    Run Phase 3 validation.
    
    Exit code 0 = all checks passed
    Exit code 1 = some checks failed
    """
    print("=" * 60)
    print("PHASE 3 VALIDATION: Mechanistic Analysis")
    print("=" * 60)
    print("\nThis script validates Phase 3 deliverables against the specification:")
    print("  - phased_implementation_plan.md (lines 1334-2041)")
    print("  - technical_specification.md (lines 735-1189)")
    
    all_pass = True
    
    # Step 1: Check output files exist
    print("\n[Step 1/3] Checking output files...")
    outputs_valid, output_errors = validate_outputs_exist()
    if not outputs_valid:
        print(f"\n❌ {len(output_errors)} missing output file(s):")
        for err in output_errors:
            print(f"   • {err}")
        all_pass = False
        
        # If outputs don't exist, can't proceed with validation
        print("\n" + "=" * 60)
        print("❌ PHASE 3 VALIDATION FAILED")
        print("=" * 60)
        print("\nRun Phase 3 tasks first:")
        print("  1. python src/mechanistic/cache_activations.py")
        print("  2. python src/mechanistic/train_probes.py")
        return 1
    
    # Step 2: Validate activations
    print("\n[Step 2/3] Validating activation caches...")
    valid_acts, act_errors = validate_activations()
    if valid_acts:
        print("\n✅ Activation caches valid (Data Contract 1 satisfied)")
    else:
        print(f"\n❌ {len(act_errors)} activation error(s):")
        for err in act_errors[:10]:
            print(f"   • {err}")
        all_pass = False
    
    # Step 3: Validate probe results
    print("\n[Step 3/3] Validating probe results...")
    valid_probes, probe_errors = validate_probe_results()
    if valid_probes:
        print("\n✅ Probe results valid (Data Contract 2 satisfied)")
    else:
        print(f"\n❌ {len(probe_errors)} probe error(s):")
        for err in probe_errors[:10]:
            print(f"   • {err}")
        all_pass = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 3 ACCEPTANCE CRITERIA SUMMARY")
    print("=" * 60)
    
    criteria = [
        ("Activation files exist for all 4 layers", outputs_valid),
        ("Each file has 'faithful' and 'unfaithful' tensors", valid_acts),
        ("Minimum 10 examples in each class", valid_acts),
        ("Probe results file exists", outputs_valid),
        ("Results for all 4 layers present", valid_probes),
        ("All accuracy/auc values in [0, 1]", valid_probes),
        ("At least one layer has accuracy > 0.55", valid_probes),
    ]
    
    for criterion, passed in criteria:
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {criterion}")
    
    print("=" * 60)
    
    if all_pass:
        print("✅✅✅ ALL PHASE 3 CHECKS PASSED ✅✅✅")
        print("\n✅ Ready to proceed to Phase 4 (Report)")
        print("\nPhase 3 deliverables:")
        print("  • 4 activation cache files")
        print("  • 1 probe results file")
        print("  • 1 probe performance plot")
        return 0
    else:
        print("❌ PHASE 3 VALIDATION FAILED")
        print("\n❌ Fix errors before proceeding to Phase 4")
        return 1


if __name__ == "__main__":
    sys.exit(main())

