# Remote GPU Pod Setup Guide for Phase 3

## Problem
You're getting `ModuleNotFoundError: No module named 'pandas'` on your remote GPU pod.

## Why This Happened
The conda environment was created, but dependencies weren't fully installed.

## Solution: 2 Options

### **Option 1: Quick Fix (Use requirements.txt)**

On your remote GPU pod:

```bash
# 1. SSH into your pod
ssh user@your-gpu-pod

# 2. Navigate to project
cd /unfaithful-reasoning

# 3. Activate conda environment
conda activate cot-unfaith

# 4. Install PyTorch with CUDA first (important!)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 5. Install all other requirements
pip install -r requirements.txt

# 6. Verify critical packages
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import transformer_lens; print('TransformerLens: OK')"
```

### **Option 2: Use Setup Script (Recommended)**

I've created a comprehensive setup script:

```bash
# On your remote GPU pod
cd /unfaithful-reasoning

# Activate environment first!
conda activate cot-unfaith

# Run setup script
bash setup_remote_env.sh
```

The script will:
- ✅ Install PyTorch with CUDA support
- ✅ Install all Phase 3 dependencies
- ✅ Verify all installations
- ✅ Test Phase 3 imports

## What Gets Installed

### Core ML (Phase 2, 3)
- `torch==2.2.0` (with CUDA 11.8)
- `transformers==4.39.0`
- `accelerate==0.27.0`

### Mechanistic Interpretability (Phase 3)
- `transformer-lens==1.17.0`
- `nnsight==0.2.6`

### Data & Analysis (All Phases)
- `pandas==2.2.0` ← **This is what was missing!**
- `numpy>=1.20.0,<2.0.0`
- `scipy==1.12.0`
- `scikit-learn==1.4.0`
- `jsonlines==4.0.0`
- `pyyaml==6.0.1`

### Visualization (Phase 2, 4)
- `matplotlib==3.8.2`
- `seaborn==0.13.2`
- `plotly==5.18.0`

### Utilities
- `tqdm==4.66.1`
- `pytest==8.0.0`
- `pytest-cov==4.1.0`

## After Installation

Verify everything is ready:

```bash
# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"

# Check Phase 3 dependencies
python -c "from transformer_lens import HookedTransformer; print('✓ TransformerLens OK')"
python -c "from src.mechanistic.contracts import Phase3Config; print('✓ Phase 3 imports OK')"
```

## Then Run Phase 3

```bash
bash run_phase3.sh
```

## Troubleshooting

### If PyTorch doesn't detect CUDA
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

### If TransformerLens fails to install
```bash
# Install from source
pip install git+https://github.com/neelnanda-io/TransformerLens.git
```

### If you get memory errors during Phase 3
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Or reduce batch size in cache_activations.py
# (edit the file and set max_faithful=20, max_unfaithful=15)
```

## Common Mistakes to Avoid

❌ **Don't do this:**
```bash
# Running pip install outside conda environment
pip install -r requirements.txt  # Wrong shell!
```

✅ **Do this:**
```bash
# Always activate first
conda activate cot-unfaith
pip install -r requirements.txt  # Correct!
```

❌ **Don't do this:**
```bash
# Installing CPU-only PyTorch on GPU pod
pip install torch  # Gets CPU version by default!
```

✅ **Do this:**
```bash
# Explicitly install CUDA version
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

## Storage Space Check

Phase 3 will download the model and cache activations:
- Model size: ~3-4 GB
- Activation cache: ~500 MB
- **Total needed: ~5 GB**

Check available space:
```bash
df -h .
```

## Expected Installation Time

- PyTorch with CUDA: 2-3 minutes
- All other packages: 3-5 minutes
- **Total: ~5-8 minutes**

## After Setup Is Complete

You should see:
```
✅ SETUP COMPLETE!

You can now run Phase 3:
  bash run_phase3.sh
```

Then Phase 3 will run for 2-3 hours to cache activations.

