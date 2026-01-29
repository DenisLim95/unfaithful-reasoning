# Fixed: Circular Import Error

## Problem
The error you encountered was:
```
ImportError: cannot import name 'GenericAlias' from partially initialized module 'types' 
(most likely due to a circular import) (/unfaithful-reasoning/src/mechanistic/types.py)
```

## Root Cause
You had a file named `types.py` in `src/mechanistic/` that was conflicting with Python's built-in `types` module. When Python tried to import standard library modules, it found your local file first, causing a circular import.

## Solution Applied

### 1. Renamed the file
```bash
src/mechanistic/types.py → src/mechanistic/contracts.py
```

This is actually a better name since the file contains Phase 3 data contracts!

### 2. Updated all imports
Updated 6 files that imported from the old name:

**Source files:**
- `src/mechanistic/cache_activations.py` (2 occurrences)
- `src/mechanistic/train_probes.py` (1 occurrence)

**Test files:**
- `tests/validate_phase3.py` (1 occurrence)
- `tests/test_phase3_contracts_standalone.py` (1 occurrence)
- `tests/test_phase3_contracts.py` (2 occurrences)

All changed from:
```python
from .types import (...)
# or
from mechanistic.types import (...)
```

To:
```python
from .contracts import (...)
# or
from mechanistic.contracts import (...)
```

## How to Proceed

On your **remote GPU pod** (where you're running Phase 3), sync the updated code:

```bash
# From your local laptop, rsync to the remote pod
rsync -av --exclude='data/' --exclude='results/' --exclude='venv/' \
    /Users/denislim/workspace/mats-10.0/ \
    user@your-gpu-pod:/unfaithful-reasoning/

# Or use git if you prefer
cd /Users/denislim/workspace/mats-10.0
git add -A
git commit -m "Fix: Rename types.py to contracts.py to avoid circular import"
git push

# Then on remote:
cd /unfaithful-reasoning
git pull
```

Then run Phase 3 again:

```bash
# On the remote GPU pod
cd /unfaithful-reasoning
bash run_phase3.sh
```

## Verification

The fix is complete when:
- ✅ File `src/mechanistic/contracts.py` exists
- ✅ File `src/mechanistic/types.py` does NOT exist
- ✅ All 6 files import from `contracts` instead of `types`
- ✅ No `ImportError` about `GenericAlias`

## Why This Happened

Python's import system searches for modules in this order:
1. Current directory
2. PYTHONPATH directories
3. Standard library
4. Site packages

Since you had `types.py` in the project, it was found before Python's built-in `types` module, causing the conflict.

**Lesson:** Never name your files the same as Python built-in modules!

Common names to avoid:
- `types.py` ❌
- `string.py` ❌
- `os.py` ❌
- `sys.py` ❌
- `json.py` ❌
- `math.py` ❌

Better alternatives:
- `contracts.py` ✅
- `schemas.py` ✅
- `data_types.py` ✅
- `custom_types.py` ✅

## Status

✅ **FIXED** - Ready to run Phase 3 on your GPU pod!

