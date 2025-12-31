#!/usr/bin/env python3
"""
Quick script to view Phase 3 results.
Works from anywhere - handles imports properly.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import LinearProbe so torch.load can deserialize it
try:
    from src.mechanistic.train_probes import LinearProbe
except ImportError:
    print("⚠️  Warning: Could not import LinearProbe")
    print("   Probe objects won't be accessible, but metrics will work")

def view_results(results_file="results/probe_results/all_probe_results.pt"):
    """Display Phase 3 probe results."""
    
    print("=" * 60)
    print("PHASE 3 RESULTS: Linear Probe Analysis")
    print("=" * 60)
    print()
    
    # Check file exists
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("\nMake sure you're running this from the project root:")
        print("  cd /path/to/mats-10.0")
        print("  python view_phase3_results.py")
        return
    
    # Load results
    print(f"Loading: {results_file}")
    try:
        results = torch.load(results_file, weights_only=False)
        print(f"✓ Loaded results for {len(results)} layers\n")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Display results
    print("=" * 60)
    print("PROBE PERFORMANCE BY LAYER")
    print("=" * 60)
    print()
    
    best_acc = 0
    best_layer = None
    
    for layer_name in sorted(results.keys()):
        result = results[layer_name]
        acc = result.accuracy
        auc = result.auc
        
        print(f"{layer_name:10s}  Accuracy: {acc:6.1%}  AUC: {auc:.3f}")
        
        if acc > best_acc:
            best_acc = acc
            best_layer = layer_name
    
    print()
    print("=" * 60)
    print(f"BEST LAYER: {best_layer} with {best_acc:.1%} accuracy")
    print("=" * 60)
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    print("-" * 60)
    
    if best_acc > 0.70:
        print("🌟 STRONG SIGNAL")
        print("   A clear linear faithfulness direction exists!")
        print("   Faithfulness is explicitly encoded in the model.")
        interpretation = "strong"
    elif best_acc > 0.60:
        print("✅ MODERATE SIGNAL")
        print("   A linear faithfulness direction was found.")
        print("   Faithfulness has clear linear structure.")
        interpretation = "moderate"
    elif best_acc > 0.55:
        print("〰️  WEAK SIGNAL")
        print("   Some linear structure detected.")
        print("   Faithfulness may be partially linearly encoded.")
        interpretation = "weak"
    else:
        print("❌ NULL RESULT")
        print("   No linear faithfulness direction found.")
        print("   Faithfulness is not simply linearly encoded.")
        interpretation = "null"
    
    print()
    print("IMPLICATIONS:")
    print("-" * 60)
    
    if interpretation in ["strong", "moderate"]:
        print("• Faithfulness can be detected using linear probes")
        print("• Real-time monitoring may be feasible")
        print("• Interventions along this direction are possible")
        print(f"• Layer {best_layer.split('_')[1]} contains the key information")
    elif interpretation == "weak":
        print("• Faithfulness has some linear structure")
        print("• Non-linear methods may work better")
        print("• More complex monitoring needed")
    else:
        print("• Faithfulness is not linearly separable")
        print("• Distributed or non-linear encoding")
        print("• This null result rules out simple approaches")
    
    print()
    print("=" * 60)
    print("COMPARISON TO BASELINES")
    print("=" * 60)
    print()
    print(f"Random guess:        50.0%")
    print(f"Your probe:         {best_acc:6.1%}")
    print(f"Improvement:        +{(best_acc - 0.5)*100:.1f} percentage points")
    print()
    
    # Summary for report
    print("=" * 60)
    print("FOR YOUR PHASE 4 REPORT")
    print("=" * 60)
    print()
    print("Key Finding:")
    print(f'  "We trained linear probes to predict CoT faithfulness from')
    print(f'   activations. The best probe achieved {best_acc:.1%} accuracy at')
    print(f'   {best_layer}, indicating that faithfulness {"is" if best_acc > 0.55 else "is not"} linearly')
    print(f'   encoded in the model\'s representations."')
    print()
    
    if best_acc > 0.55:
        print("Implication:")
        print('  "This suggests CoT faithfulness could be monitored in')
        print('   real-time using simple linear classifiers, with potential')
        print('   applications for AI safety and alignment."')
    else:
        print("Implication:")
        print('  "This null result suggests faithfulness emerges from')
        print('   complex non-linear interactions, making simple monitoring')
        print('   approaches insufficient."')
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View Phase 3 probe results")
    parser.add_argument(
        "--results",
        default="results/probe_results/all_probe_results.pt",
        help="Path to probe results file"
    )
    
    args = parser.parse_args()
    view_results(args.results)

