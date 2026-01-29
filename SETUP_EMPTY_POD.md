# Complete Setup Guide for Empty RunPod

**For pods with nothing installed - no conda, no Python packages, nothing.**

---

## Step 1: Verify Basic System (On Pod)

```bash
# Check if Python 3 is installed (usually pre-installed on RunPod)
python3 --version

# If Python 3 is NOT installed, install it:
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Check GPU is accessible
nvidia-smi

# Check disk space (need ~10GB free)
df -h
```

**Expected output:**
- Python 3.8+ (usually 3.10 or 3.11 on RunPod)
- GPU visible via `nvidia-smi`
- At least 10GB free disk space

---

## Step 2: Transfer Code from Local Machine

**On your LOCAL machine** (new terminal):

```bash
cd /Users/denislim/workspace/mats-10.0

# Create tarball with all essential files
tar -czf mats-complete.tar.gz \
    src/ \
    scripts/ \
    workflows/ \
    requirements.txt \
    setup_remote_pod.sh \
    *.py \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.md' \
    --exclude='data/activations' \
    --exclude='results'

# Copy to pod (replace YOUR_POD_IP with actual IP)
scp mats-complete.tar.gz root@YOUR_POD_IP:/root/

# Optional: Also copy any existing data you want to keep
# scp -r data/ root@YOUR_POD_IP:/root/mats-10.0/
```

---

## Step 3: Extract and Navigate (On Pod)

```bash
# Extract the tarball
cd /root
tar -xzf mats-complete.tar.gz

# This should create a directory structure
# Navigate to project root
cd /root/workspace/mats-10.0  # or wherever it extracted

# Verify files are there
ls -la
ls -la src/
ls -la scripts/
```

---

## Step 4: Run Automated Setup Script

```bash
# Make setup script executable
chmod +x setup_remote_pod.sh

# Run the setup (this will take 5-10 minutes)
./setup_remote_pod.sh
```

**What this script does:**
1. ✅ Checks Python version
2. ✅ Creates virtual environment (`venv/`)
3. ✅ Creates directory structure
4. ✅ Installs all dependencies (PyTorch, transformers, etc.)
5. ✅ Verifies GPU access
6. ✅ Tests model loading (downloads model from HuggingFace automatically)

**Expected output:**
```
==========================================
Setting Up Remote Pod Environment
==========================================

[1/7] Checking Python version...
Python 3.10.12
✓ Python 3 found

[2/7] Creating virtual environment...
✓ Virtual environment created and activated

[3/7] Creating directory structure...
✓ Directories created

[4/7] Installing Python dependencies...
Installing from requirements.txt...
[lots of pip install output...]
✓ Dependencies installed

[5/7] Verifying GPU access...
CUDA available: True
GPU count: 1
GPU name: NVIDIA A100-SXM4-40GB

[6/7] Testing model loading...
Loading model...
Downloading tokenizer...
Downloading model files...
✓ Tokenizer loaded
✓ Model loaded
✓ Model inference works
✓ Model device: cuda:0

[7/7] Setup Summary
==========================================
✓ Python installed
✓ Virtual environment created (venv/)
✓ Directory structure created
✓ Dependencies installed
✓ GPU verified
✓ Model loading tested

==========================================
SETUP COMPLETE!
==========================================
```

---

## Step 5: Activate Environment and Verify

```bash
# IMPORTANT: Always activate venv before running scripts
source venv/bin/activate

# You should see (venv) prefix in your prompt
# Verify everything works
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check critical packages
python -c "import transformers, pandas, sklearn, numpy, matplotlib; print('✓ All packages imported')"

# Quick model test
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('✓ Model ready (will download on first use)')
"
```

---

## Step 6: You're Ready to Run!

**Common commands:**

```bash
# Always activate first!
source venv/bin/activate

# Generate questions
python scripts/01_generate_questions.py --num-pairs 100

# Generate responses (requires GPU, downloads model if needed)
python scripts/02_generate_responses.py --questions data/raw/questions.json

# Score faithfulness
python scripts/03_score_faithfulness.py --responses data/responses/responses.jsonl

# Or run full pipeline
bash workflows/full_pipeline.sh 100
```

---

## Troubleshooting

### Python 3 Not Found

```bash
# Install Python 3
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Verify
python3 --version
```

### Setup Script Fails

**If `setup_remote_pod.sh` fails, do manual setup:**

```bash
# 1. Create venv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Create directories
mkdir -p data/{raw,responses,processed,activations,test_activations}
mkdir -p results/{probe_results,activation_visualizations}

# 6. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### CUDA Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# If nvidia-smi fails, GPU might not be properly configured
# Check RunPod dashboard for GPU status

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Disk Space

```bash
# Check space
df -h

# Clean up if needed
pip cache purge
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai*  # Only if you need to re-download

# Model is ~3-4GB, activations can be ~500MB-2GB
```

### Module Not Found Errors

```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt

# If specific package fails, install individually
pip install transformers pandas scikit-learn numpy matplotlib
```

### Model Download Fails

The model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) downloads automatically from HuggingFace on first use. If it fails:

```bash
# Check internet connection
ping huggingface.co

# Set HuggingFace cache location if needed
export HF_HOME=/root/.cache/huggingface

# Try manual download
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='float16', device_map='auto')
print('✓ Model downloaded successfully')
"
```

---

## Quick Reference Card

```bash
# === SETUP (ONE TIME) ===
cd /root
tar -xzf mats-complete.tar.gz
cd /root/workspace/mats-10.0  # or wherever extracted
chmod +x setup_remote_pod.sh
./setup_remote_pod.sh

# === EVERY TIME YOU SSH IN ===
cd /root/workspace/mats-10.0  # or your project path
source venv/bin/activate

# === VERIFY SETUP ===
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# === RUN SCRIPTS ===
python scripts/01_generate_questions.py --num-pairs 100
python scripts/02_generate_responses.py --questions data/raw/questions.json
bash workflows/full_pipeline.sh 100
```

---

## What Gets Installed

- **Python packages:** ~2GB
  - PyTorch with CUDA: ~1.5GB
  - Transformers: ~200MB
  - Other packages: ~300MB

- **Model (downloaded on first use):** ~3-4GB
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

- **Total disk space needed:** ~10GB

---

## Expected Setup Time

- **Transfer code:** 1-2 minutes (depends on connection)
- **Extract tarball:** 30 seconds
- **Run setup script:** 5-10 minutes
  - Creating venv: 10 seconds
  - Installing PyTorch: 2-3 minutes
  - Installing other packages: 2-3 minutes
  - Testing model (downloads model): 3-5 minutes
- **Total:** ~10-15 minutes

---

## After Setup

Your pod is now ready! The model will be cached in `~/.cache/huggingface/` so subsequent runs won't need to download it again.

**Remember:** Always activate the venv before running scripts:
```bash
source venv/bin/activate
```

---

## Need Help?

If setup fails:
1. Check `nvidia-smi` works
2. Check `python3 --version` shows 3.8+
3. Check disk space with `df -h`
4. Try manual setup (see Troubleshooting section)
5. Check RunPod logs/dashboard for GPU issues
