# Fixed: Relative Import Error

## Problem
```
ImportError: attempted relative import with no known parent package
```

This happened when running:
```bash
python src/mechanistic/cache_activations.py
```

## Root Cause

When you run a Python script directly (not as a module), Python doesn't recognize it as part of a package, so relative imports like `from .contracts import` fail.

## Solution Applied

Updated both Phase 3 scripts to handle imports robustly:

### Files Updated:
1. ✅ `src/mechanistic/cache_activations.py`
2. ✅ `src/mechanistic/train_probes.py`

### What Changed:

Added smart import logic that works both ways:

```python
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try relative import first, fall back to absolute
try:
    from .contracts import (...)
except ImportError:
    from src.mechanistic.contracts import (...)
```

This makes the scripts work when run:
- ✅ Directly: `python src/mechanistic/cache_activations.py`
- ✅ As module: `python -m src.mechanistic.cache_activations`
- ✅ From any directory

## What To Do Now

**On your remote GPU pod**, sync the updated files:

### Option 1: Quick Git Update
```bash
# On laptop
cd /Users/denislim/workspace/mats-10.0
git add -A
git commit -m "Fix: Make imports work for direct script execution"
git push

# On remote pod
cd /unfaithful-reasoning
git pull
```

### Option 2: Manual Sync
```bash
# From laptop
rsync -av /Users/denislim/workspace/mats-10.0/src/ \
    user@your-gpu-pod:/unfaithful-reasoning/src/
```

### Then Run Phase 3:
```bash
# On remote pod
cd /unfaithful-reasoning
bash run_phase3.sh
```

## Summary of All Fixes So Far

| Issue | Status |
|-------|--------|
| 1. Circular import (`types.py` vs stdlib) | ✅ Fixed (renamed to `contracts.py`) |
| 2. PyTorch 2.2.0 not available | ✅ Fixed (updated to `>=2.0.0`) |
| 3. Missing `pandas` | ✅ Fixed (`pip install pandas`) |
| 4. Missing `typeguard` | ✅ Fixed (`pip install typeguard`) |
| 5. Relative import error | ✅ Fixed (smart import logic) |
| **Ready to run Phase 3?** | ✅ **YES!** |

## Files You Need to Sync

Make sure these are up to date on your remote pod:
- ✅ `requirements.txt` (relaxed versions)
- ✅ `src/mechanistic/contracts.py` (renamed from types.py)
- ✅ `src/mechanistic/cache_activations.py` (fixed imports)
- ✅ `src/mechanistic/train_probes.py` (fixed imports)

## Verification

After syncing, verify the imports work:

```bash
# On remote pod
cd /unfaithful-reasoning
python -c "from src.mechanistic.contracts import Phase3Config; print('✓ Imports work!')"
python -c "import sys; sys.path.insert(0, '.'); from src.mechanistic.cache_activations import load_faithfulness_labels; print('✓ Scripts import!')"
```

## Alternative: Run as Module (Optional)

If you prefer, you can also update `run_phase3.sh` to run scripts as modules:

Change:
```bash
python src/mechanistic/cache_activations.py
```

To:
```bash
python -m src.mechanistic.cache_activations
```

But with our fix, both ways work! ✅

