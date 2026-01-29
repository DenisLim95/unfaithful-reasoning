# 🚀 Sync & Run Phase 3 - Complete Guide

## All Issues Fixed! ✅

Here's everything that was wrong and has been fixed:

1. ✅ **Circular import** - Renamed `types.py` → `contracts.py`
2. ✅ **PyTorch version** - Updated `requirements.txt` for newer versions
3. ✅ **Missing packages** - Added `typeguard` and relaxed version constraints
4. ✅ **Import errors** - Made scripts work when run directly

## Quick Start: 3 Commands

**On your remote GPU pod**, run these 3 commands:

```bash
# 1. Sync the fixed files (choose one method below)

# 2. Install missing packages
pip install typeguard

# 3. Run Phase 3
cd /unfaithful-reasoning
bash run_phase3.sh
```

## Step 1: Sync Files (Choose One Method)

### Method A: Using Git (Recommended)

```bash
# On your LAPTOP
cd /Users/denislim/workspace/mats-10.0
git add -A
git commit -m "Fix Phase 3 import and dependency issues"
git push

# On your REMOTE GPU POD
cd /unfaithful-reasoning
git pull
```

### Method B: Using rsync

```bash
# On your LAPTOP
rsync -av --exclude='data/' --exclude='venv/' \
    /Users/denislim/workspace/mats-10.0/ \
    user@your-gpu-pod:/unfaithful-reasoning/
```

### Method C: Manual File Copy

Copy these specific files from laptop to remote:

1. `requirements.txt`
2. `src/mechanistic/contracts.py` (renamed from types.py)
3. `src/mechanistic/cache_activations.py`
4. `src/mechanistic/train_probes.py`
5. `tests/validate_phase3.py`
6. `tests/test_phase3_contracts.py`
7. `tests/test_phase3_contracts_standalone.py`

## Step 2: Install Packages

**On your remote GPU pod:**

```bash
cd /unfaithful-reasoning
conda activate cot-unfaith  # or stay in base if you prefer

# Install the one missing package
pip install typeguard

# Verify everything is installed
python -c "import torch, pandas, transformer_lens, typeguard; print('✓ All packages OK')"
```

## Step 3: Verify Imports Work

```bash
# Test the fixed imports
python -c "from src.mechanistic.contracts import Phase3Config; print('✓ Imports working!')"
```

If you see `✓ Imports working!`, you're ready to go!

## Step 4: Run Phase 3

```bash
bash run_phase3.sh
```

You should see:
```
============================================================
PHASE 3: Mechanistic Analysis - Linear Probe Analysis
============================================================

This script will:
  1. Cache activations (2-3 hours)
  2. Train linear probes (1-2 hours)
  3. Validate Phase 3 deliverables (5 min)

Prerequisites:
  - Phase 2 must be complete
  - transformer-lens must be installed

Continue? (y/n) y

[0/3] Checking Phase 2...
✓ Phase 2 outputs found

[1/3] Task 3.2: Caching Activations
⏱️  Estimated time: 2-3 hours
🖥️  Requires: GPU (model loading)

Loading model...
```

## Expected Timeline

- **Activation caching**: 2-3 hours (GPU-intensive)
- **Probe training**: 1-2 hours (can run on CPU)
- **Validation**: 5 minutes
- **Total**: ~3-4 hours

## Troubleshooting

### Still Getting Import Errors?

```bash
# Make sure you're in the project root
cd /unfaithful-reasoning
pwd  # Should show /unfaithful-reasoning

# Check files exist
ls -la src/mechanistic/contracts.py  # Should exist
ls -la src/mechanistic/types.py      # Should NOT exist (deleted)
```

### Out of Memory?

Edit `src/mechanistic/cache_activations.py` and reduce:
```python
max_faithful: int = 20  # (default was 30)
max_unfaithful: int = 15  # (default was 20)
```

### CUDA Not Available?

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, check:
```bash
nvidia-smi  # Should show your GPU
```

### Want to Test First?

Run just the imports without starting Phase 3:
```bash
python -c "
from src.mechanistic.cache_activations import cache_activations_for_phase3
from src.mechanistic.train_probes import train_all_probes
print('✓ All Phase 3 functions importable!')
"
```

## What Phase 3 Will Do

1. **Load your Phase 2 results** (faithfulness scores)
2. **Load the model** (DeepSeek-R1-Distill-Qwen-1.5B)
3. **Re-run prompts** to cache activations at layers [6, 12, 18, 24]
4. **Save activation files** (~500 MB total)
5. **Train linear probes** (4 probes, one per layer)
6. **Generate results**:
   - Probe accuracy scores
   - Best layer identification
   - Direction vectors
   - Performance plot

## After Phase 3 Completes

You'll have:
- ✅ 4 activation cache files in `data/activations/`
- ✅ Probe results in `results/probe_results/`
- ✅ Performance plot: `results/probe_results/probe_performance.png`

Then you can move to **Phase 4: Report & Polish**!

## Quick Reference: All Commands

```bash
# Complete workflow for remote pod:
cd /unfaithful-reasoning
git pull                          # Sync files
conda activate cot-unfaith        # Activate env
pip install typeguard             # Install missing package
python -c "from src.mechanistic.contracts import Phase3Config; print('✓ OK')"  # Verify
bash run_phase3.sh                # Run Phase 3 (3-4 hours)
```

## Need Help?

Check these documentation files:
- `FIX_CIRCULAR_IMPORT.md` - Details on the types.py → contracts.py fix
- `FIX_TORCH_VERSION.md` - Details on PyTorch version issue
- `FIX_IMPORT_ERROR.md` - Details on relative import fix
- `REMOTE_SETUP_GUIDE.md` - Full setup guide

---

**You're all set!** Just sync, install `typeguard`, and run Phase 3! 🎉

