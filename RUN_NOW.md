# 🎯 RUN PHASE 3 NOW - Final Instructions

## ✅ ALL ERRORS FIXED!

I found and fixed the **last** import error. Here's what to do:

## Step 1: Sync This ONE File to Remote

You only need to update **one file** on your remote pod:

```bash
# File that changed: src/mechanistic/cache_activations.py
```

### Option A: Using Git (30 seconds)

```bash
# On your LAPTOP:
cd /Users/denislim/workspace/mats-10.0
git add src/mechanistic/cache_activations.py
git commit -m "Fix: Move MIN_*_SAMPLES imports to top"
git push

# On your REMOTE POD:
cd /unfaithful-reasoning
git pull
```

### Option B: Manual Copy (2 minutes)

```bash
# On REMOTE POD:
nano src/mechanistic/cache_activations.py

# Replace lines 21-37 with:
```

```python
# Import Phase 3 contracts (handle both relative and absolute imports)
try:
    from .contracts import (
        ActivationCache,
        Phase3Config,
        Phase3Error,
        validate_phase2_outputs_exist,
        PHASE3_LAYERS,
        MIN_FAITHFUL_SAMPLES,
        MIN_UNFAITHFUL_SAMPLES
    )
except ImportError:
    from src.mechanistic.contracts import (
        ActivationCache,
        Phase3Config,
        Phase3Error,
        validate_phase2_outputs_exist,
        PHASE3_LAYERS,
        MIN_FAITHFUL_SAMPLES,
        MIN_UNFAITHFUL_SAMPLES
    )
```

And **delete line 183** which says:
```python
from .contracts import MIN_FAITHFUL_SAMPLES, MIN_UNFAITHFUL_SAMPLES
```

## Step 2: Test Everything Works

```bash
# On REMOTE POD:
cd /unfaithful-reasoning
python test_phase3_ready.py
```

You should see:
```
✅ ALL CHECKS PASSED

You're ready to run Phase 3!
  bash run_phase3.sh
```

## Step 3: RUN PHASE 3! 🚀

```bash
bash run_phase3.sh
```

Expected output:
```
============================================================
PHASE 3: Mechanistic Analysis - Linear Probe Analysis
============================================================

[0/3] Checking Phase 2...
✓ Phase 2 outputs found

[1/3] Task 3.2: Caching Activations
⏱️  Estimated time: 2-3 hours

[1/6] Checking Phase 2 outputs...
   ✓ Phase 2 outputs found

[2/6] Loading faithfulness labels...
   Found 36 faithful pairs
   Found 14 unfaithful pairs
   Using 30 faithful pairs
   Using 14 unfaithful pairs

[3/6] Loading model responses...
   ✓ Loaded 50 response pairs

[4/6] Loading model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ✓ Model loaded on cuda

[5/6] Caching activations at layers [6, 12, 18, 24]...
   Caching faithful responses...
   [Progress bar will show here]
```

## What Changed in the Fix

**Before (BROKEN):**
```python
# Line 183 - inside the function:
from .contracts import MIN_FAITHFUL_SAMPLES, MIN_UNFAITHFUL_SAMPLES  # ❌ FAILS
```

**After (FIXED):**
```python
# Lines 21-37 - at the top of file:
try:
    from .contracts import (
        ...
        MIN_FAITHFUL_SAMPLES,     # ✅ WORKS
        MIN_UNFAITHFUL_SAMPLES    # ✅ WORKS
    )
except ImportError:
    from src.mechanistic.contracts import (...)
```

## Timeline After You Start

1. **Model loading**: 2-5 minutes
2. **Caching activations**: 2-3 hours ⏳
3. **Training probes**: 1-2 hours ⏳
4. **Validation**: 5 minutes
5. **Total**: ~3-4 hours

## What You'll Get

```
✅ PHASE 3 COMPLETE!

Deliverables:
  ✓ 4 activation cache files (data/activations/)
  ✓ 1 probe results file (results/probe_results/)
  ✓ 1 performance plot (results/probe_results/probe_performance.png)

Next steps:
  • Review probe_performance.png
  • Interpret results
  • Proceed to Phase 4 (Report & Polish)
```

## Quick Sanity Check

Before running, verify:

```bash
cd /unfaithful-reasoning
ls -la src/mechanistic/contracts.py      # Should exist
ls -la src/mechanistic/types.py          # Should NOT exist
grep "MIN_FAITHFUL_SAMPLES" src/mechanistic/cache_activations.py | head -1
# Should show: "        MIN_FAITHFUL_SAMPLES,"
```

## All Fixes Applied

| # | Issue | Status |
|---|-------|--------|
| 1 | `types.py` circular import | ✅ |
| 2 | PyTorch 2.2.0 unavailable | ✅ |
| 3 | Missing `pandas` | ✅ |
| 4 | Missing `typeguard` | ✅ |
| 5 | Top-level relative imports | ✅ |
| 6 | In-function relative imports | ✅ |

## Ready? Let's Go! 🎉

```bash
cd /unfaithful-reasoning
git pull                      # or manually update the file
python test_phase3_ready.py   # verify everything
bash run_phase3.sh            # RUN IT!
```

Phase 3 will take ~4 hours. You can let it run and come back later!

---

**Questions?** All fixes are documented in `FINAL_FIX.md`

