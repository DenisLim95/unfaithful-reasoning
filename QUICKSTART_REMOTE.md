# 🚀 Quick Start: Remote GPU Pod Setup

## TL;DR - Copy/Paste This

```bash
# On your remote GPU pod
cd /unfaithful-reasoning
conda activate cot-unfaith

# Install PyTorch with CUDA
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install everything else
pip install -r requirements.txt

# Verify
python -c "import torch, pandas, transformer_lens; print('✓ Ready!')"

# Run Phase 3
bash run_phase3.sh
```

## If That Doesn't Work

See `REMOTE_SETUP_GUIDE.md` for detailed troubleshooting.

## Current Issue

You're missing `pandas` and probably other packages. The `requirements.txt` file is correct, you just need to install it on the remote machine.

## What Changed Recently

1. ✅ Fixed circular import (`types.py` → `contracts.py`)
2. ✅ Added setup scripts for remote environment
3. ⏳ Need to install dependencies on remote pod

## Time Estimate

- Setup: 5-8 minutes
- Phase 3: 2-3 hours
- **Total: ~3 hours**

---

**Ready?** Just run those 5 commands above on your GPU pod! 🎯

