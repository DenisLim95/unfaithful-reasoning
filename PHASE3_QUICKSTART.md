# Phase 3 Quick Start Guide

## 30-Second Summary

**What:** Linear probe analysis for CoT faithfulness detection  
**Input:** Phase 2 outputs (responses + faithfulness scores)  
**Output:** Activation caches + probe results + performance plot  
**Time:** 6-7 hours  

---

## Prerequisites Check

```bash
# Must pass before starting Phase 3
python tests/validate_phase2.py

# Install dependencies
pip install transformer-lens scikit-learn matplotlib
```

---

## Run Phase 3

```bash
# All tasks in one command
./run_phase3.sh

# OR step-by-step:
python src/mechanistic/cache_activations.py    # 2-3 hours, GPU
python src/mechanistic/train_probes.py         # 1-2 hours, CPU ok
python tests/validate_phase3.py                # 5 min
```

---

## Expected Output

### Files Created

```
data/activations/
  ├── layer_6_activations.pt
  ├── layer_12_activations.pt
  ├── layer_18_activations.pt
  └── layer_24_activations.pt

results/probe_results/
  ├── all_probe_results.pt
  └── probe_performance.png
```

### Success Criteria

```bash
python tests/validate_phase3.py
# Exit code 0 = SUCCESS ✅
# Exit code 1 = FAILED ❌
```

---

## Interpreting Results

Open `results/probe_results/probe_performance.png`

### Best Accuracy > 0.65
**Finding:** Strong linear faithfulness direction  
**Meaning:** Can monitor faithfulness with linear classifiers  

### Best Accuracy 0.55-0.65
**Finding:** Weak linear signal  
**Meaning:** Some linear component, may need non-linear methods  

### Best Accuracy < 0.55
**Finding:** Null result (no linear direction)  
**Meaning:** Faithfulness not linearly encoded - valid scientific result!  

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Phase 2 output missing" | Run Phase 2 first |
| "Insufficient samples" | Valid result! Document in Phase 4 |
| "No layer > 0.55" | Null result - scientifically valid |
| "TransformerLens not installed" | `pip install transformer-lens` |

---

## Next Steps

After validation passes:

1. Review `probe_performance.png`
2. Interpret findings (see above)
3. Proceed to Phase 4: `phased_implementation_plan.md` lines 2043+

---

## Documentation

- **User Guide:** `PHASE3_README.md` (detailed usage)
- **Implementation:** `PHASE3_IMPLEMENTATION_SUMMARY.md` (technical details)
- **Specification:** `phased_implementation_plan.md` lines 1334-2041

---

## Phase 3 Scope

✓ Cache activations at 4 layers  
✓ Train linear probes  
✓ Generate performance plot  
✗ Attention analysis (not in Phase 3)  
✗ Report writing (that's Phase 4)  

---

**Quick Status Check:**
```bash
ls data/activations/*.pt | wc -l  # Should be 4
ls results/probe_results/*        # Should be 2 files
python tests/validate_phase3.py   # Should exit 0
```



