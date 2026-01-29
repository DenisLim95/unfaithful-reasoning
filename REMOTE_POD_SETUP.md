# Remote Pod Setup Guide

## Quick Start (Copy-Paste Commands)

### On Your Local Machine

```bash
# 1. Generate test questions locally (if you haven't already)
python generate_test_questions.py --num-pairs 200

# 2. Create a tarball of essential files
cd /Users/denislim/workspace/mats-10.0
tar -czf mats-project.tar.gz \
    src/ \
    test_probe_on_new_data.py \
    generate_test_questions.py \
    setup_remote_pod.sh \
    use_original_env.sh \
    requirements.txt \
    data/raw/test_question_pairs.json \
    results/probe_results/all_probe_results.pt

# 3. Copy to remote pod
scp mats-project.tar.gz root@your-pod-ip:/root/

# 4. SSH into the pod
ssh root@your-pod-ip
```

### On the Remote Pod

```bash
# 1. Extract files
cd /root
tar -xzf mats-project.tar.gz
cd workspace/mats-10.0  # or wherever it extracted

# 2. Make setup script executable and run it
chmod +x setup_remote_pod.sh
./setup_remote_pod.sh

# 3. Run the test pipeline
python test_probe_on_new_data.py --skip-generation --num-questions 200

# This will:
# - Generate responses for 200 test question pairs (400 prompts)
# - Score faithfulness
# - Cache activations
# - Test your existing probe
# Expected time: ~2 hours
```

### Copy Results Back to Local

```bash
# On your local machine
cd /Users/denislim/workspace/mats-10.0

# Copy test results back
scp -r root@your-pod-ip:/root/workspace/mats-10.0/data/test_activations ./data/
scp root@your-pod-ip:/root/workspace/mats-10.0/data/processed/test_faithfulness_scores.csv ./data/processed/
scp root@your-pod-ip:/root/workspace/mats-10.0/data/responses/test_responses.jsonl ./data/responses/

# Now test locally (fast, no GPU needed)
python test_probe_on_new_data.py --test-only
```

---

## Step-by-Step Guide

### Step 1: Prepare Files on Local Machine

#### Option A: Minimal Transfer (Fastest)

Transfer only what's needed:

```bash
cd /Users/denislim/workspace/mats-10.0

# Create archive of essential files only
tar -czf mats-minimal.tar.gz \
    src/ \
    test_probe_on_new_data.py \
    generate_test_questions.py \
    setup_remote_pod.sh \
    data/raw/test_question_pairs.json \
    results/probe_results/all_probe_results.pt
```

**Size:** ~10-50 MB

#### Option B: Full Transfer (Includes Training Data)

Transfer everything:

```bash
cd /Users/denislim/workspace/mats-10.0

# Create archive of all project files
tar -czf mats-full.tar.gz \
    --exclude='venv' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pt' \
    --exclude='data/activations' \
    src/ \
    test_probe_on_new_data.py \
    generate_test_questions.py \
    setup_remote_pod.sh \
    data/ \
    results/probe_results/all_probe_results.pt
```

**Size:** ~100-200 MB

### Step 2: Copy to Remote Pod

```bash
# Get your pod IP from RunPod/Vast.ai dashboard
POD_IP="your.pod.ip.address"

# Copy the archive
scp mats-minimal.tar.gz root@${POD_IP}:/root/

# Or if you created the full archive:
scp mats-full.tar.gz root@${POD_IP}:/root/
```

### Step 3: SSH into Pod

```bash
ssh root@${POD_IP}
```

### Step 4: Extract and Setup

```bash
# Extract
cd /root
tar -xzf mats-minimal.tar.gz

# Navigate to project (adjust path if needed)
cd workspace/mats-10.0  # or whatever directory it extracted to
# If it extracted to /root directly, then:
# cd /root

# Run setup script
chmod +x setup_remote_pod.sh
./setup_remote_pod.sh
```

**Expected output:**
```
==========================================
Setting Up Remote Pod Environment
==========================================

[1/6] Checking Python version...
Python 3.10.12
✓ Python 3 found

[2/6] Creating directory structure...
✓ Directories created

[3/6] Installing Python dependencies...
✓ Dependencies installed

[4/6] Verifying GPU access...
CUDA available: True
GPU count: 1
GPU name: NVIDIA A100-SXM4-40GB

[5/6] Testing model loading...
✓ Tokenizer loaded
✓ Model loaded
✓ Model inference works
✓ Model device: cuda:0

[6/6] Setup Summary
==========================================
✓ Python installed
✓ Directory structure created
✓ Dependencies installed
✓ GPU verified
✓ Model loading tested

==========================================
SETUP COMPLETE!
==========================================
```

### Step 5: Run Test Pipeline

```bash
# Run the full test pipeline
python test_probe_on_new_data.py --skip-generation --num-questions 200

# Or with nohup to keep running if you disconnect:
nohup python test_probe_on_new_data.py --skip-generation --num-questions 200 > test_output.log 2>&1 &

# Monitor progress:
tail -f test_output.log
```

**Expected timeline:**
- Question parsing: 5 seconds
- Model inference: ~80 minutes (200 pairs × 2 = 400 responses)
- Faithfulness scoring: 10 seconds
- Activation caching: ~40 minutes
- Probe testing: 5 seconds
- **Total: ~2 hours**

### Step 6: Copy Results Back

Once complete, on your **local machine**:

```bash
POD_IP="your.pod.ip.address"

cd /Users/denislim/workspace/mats-10.0

# Copy test results
scp -r root@${POD_IP}:/root/workspace/mats-10.0/data/test_activations ./data/
scp root@${POD_IP}:/root/workspace/mats-10.0/data/processed/test_faithfulness_scores.csv ./data/processed/
scp root@${POD_IP}:/root/workspace/mats-10.0/data/responses/test_responses.jsonl ./data/responses/

# If you want the logs too:
scp root@${POD_IP}:/root/workspace/mats-10.0/test_output.log ./
```

### Step 7: Test Probe Locally

```bash
# On your local machine
python test_probe_on_new_data.py --test-only
```

---

## Troubleshooting

### "Permission denied (publickey)"

```bash
# Use password authentication
scp -o PreferredAuthentications=password mats-minimal.tar.gz root@${POD_IP}:/root/

# Or set up SSH key
ssh-copy-id root@${POD_IP}
```

### "tar: Cannot open: No such file or directory"

```bash
# Make sure you're in the right directory
cd /Users/denislim/workspace/mats-10.0

# Check files exist
ls -la test_probe_on_new_data.py
ls -la results/probe_results/all_probe_results.pt
```

### "CUDA out of memory"

```bash
# Edit test_probe_on_new_data.py and reduce the limit on lines 144 and 151:
# Change `[:50]` to `[:25]` to process fewer samples at once
```

### "No such file: test_question_pairs.json"

```bash
# Generate questions on the remote pod
python generate_test_questions.py --num-pairs 200
```

### Setup script fails at model loading

```bash
# Check GPU is accessible
nvidia-smi

# Check CUDA version
nvcc --version

# May need to install different PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Alternative: Manual Setup

If the automatic script doesn't work, here's manual setup:

```bash
# 1. Install dependencies manually
pip install torch transformers pandas scikit-learn matplotlib seaborn tqdm accelerate

# 2. Create directories
mkdir -p data/{raw,responses,processed,activations,test_activations}
mkdir -p results/{probe_results,activation_visualizations}

# 3. Test GPU
python -c "import torch; print(torch.cuda.is_available())"

# 4. Run pipeline
python test_probe_on_new_data.py --skip-generation
```

---

## Files That Need to Be on Remote Pod

### Essential (Must Have):
- `src/` directory (all Python code)
- `test_probe_on_new_data.py` (main test script)
- `data/raw/test_question_pairs.json` (test questions)
- `results/probe_results/all_probe_results.pt` (trained probe)

### Optional (Can Generate on Remote):
- `generate_test_questions.py` (can generate questions remotely)
- `requirements.txt` (for pip install)

### NOT Needed on Remote:
- `venv/` (create fresh env on remote)
- `.git/` (not needed for inference)
- `data/activations/` (training data, not needed for testing)
- `*.md` documentation files

---

## Quick Reference: All Commands

```bash
# === LOCAL ===
cd /Users/denislim/workspace/mats-10.0
python generate_test_questions.py --num-pairs 200
tar -czf mats-minimal.tar.gz src/ test_probe_on_new_data.py setup_remote_pod.sh data/raw/test_question_pairs.json results/probe_results/all_probe_results.pt
scp mats-minimal.tar.gz root@POD_IP:/root/
ssh root@POD_IP

# === REMOTE ===
cd /root
tar -xzf mats-minimal.tar.gz
chmod +x setup_remote_pod.sh
./setup_remote_pod.sh
nohup python test_probe_on_new_data.py --skip-generation --num-questions 200 > test_output.log 2>&1 &
tail -f test_output.log
# [wait ~2 hours]

# === LOCAL ===
scp -r root@POD_IP:/root/workspace/mats-10.0/data/test_activations ./data/
scp root@POD_IP:/root/workspace/mats-10.0/data/processed/test_faithfulness_scores.csv ./data/processed/
python test_probe_on_new_data.py --test-only
```

---

## Cost Estimation

**RunPod/Vast.ai A100 40GB:**
- ~$1.50/hour
- Expected runtime: 2 hours
- **Total cost: ~$3.00**

**Alternative (cheaper): RTX 4090**
- ~$0.40/hour
- Expected runtime: 3-4 hours
- **Total cost: ~$1.50**

---

Ready to go! Do you need help with any specific step?

