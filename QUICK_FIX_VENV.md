# Quick Fix for Current Pod (With Virtual Environment)

You're on your remote pod and got the `register_pytree_node` error. Here's how to fix it:

---

## Option 1: Use the Fix Script (Easiest)

```bash
# Run the fix script
chmod +x fix_compatibility.sh
./fix_compatibility.sh

# The script will:
# 1. Create/activate venv
# 2. Install correct package versions
# 3. Test everything works

# After it completes, you're ready to run:
python test_probe_on_new_data.py --skip-generation --num-questions 200
```

---

## Option 2: Manual Fix

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies with correct versions
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install pandas scikit-learn matplotlib seaborn tqdm accelerate

# 4. Verify it works
python -c "import torch; from transformers import AutoModelForCausalLM; print('✓ Success')"

# 5. Run your test
python test_probe_on_new_data.py --skip-generation --num-questions 200
```

---

## Important: Always Activate venv

**Every time you SSH into the pod**, you need to activate the virtual environment:

```bash
cd /root  # or wherever your project is
source venv/bin/activate

# Now you can run Python scripts
python test_probe_on_new_data.py --skip-generation
```

---

## Running with nohup (Recommended)

To keep the process running after you disconnect:

```bash
# Activate venv first
source venv/bin/activate

# Run with nohup
nohup python test_probe_on_new_data.py \
    --skip-generation \
    --num-questions 200 \
    > test_output.log 2>&1 &

# Monitor progress
tail -f test_output.log

# Press Ctrl+C to stop monitoring (process keeps running)
# You can now disconnect from SSH safely
```

---

## Check if Process is Running

```bash
# See if Python is running
ps aux | grep python

# Check the log
tail -f test_output.log

# See how long it's been running
ps -eo pid,etime,cmd | grep python
```

---

## For Future: Fresh Setup with venv

If you spin up a NEW pod in the future:

```bash
# After extracting your files
cd /root  # or project directory

# Run setup script (now creates venv automatically)
chmod +x setup_remote_pod.sh
./setup_remote_pod.sh

# Activate venv
source venv/bin/activate

# Run your code
python test_probe_on_new_data.py --skip-generation
```

The updated `setup_remote_pod.sh` now:
1. ✅ Creates virtual environment
2. ✅ Installs dependencies in venv
3. ✅ Uses correct package versions
4. ✅ Tests everything

---

## Troubleshooting

### "bash: venv/bin/activate: No such file or directory"

```bash
# Create venv
python3 -m venv venv

# Then activate
source venv/bin/activate
```

### "Command 'python' not found"

Use `python3` instead:

```bash
python3 test_probe_on_new_data.py --skip-generation
```

Or create an alias in venv (after activating):

```bash
# In activated venv, python should already work
which python  # Should show /root/venv/bin/python
```

### "Package not found"

Make sure venv is activated (you should see `(venv)` in your prompt):

```bash
source venv/bin/activate
pip list  # Check what's installed
```

---

## Quick Reference

```bash
# === EVERY TIME YOU SSH IN ===
cd /root  # or your project dir
source venv/bin/activate

# === RUN YOUR CODE ===
python test_probe_on_new_data.py --skip-generation --num-questions 200

# === RUN IN BACKGROUND ===
nohup python test_probe_on_new_data.py --skip-generation --num-questions 200 > test.log 2>&1 &
tail -f test.log

# === CHECK STATUS ===
ps aux | grep python
tail test.log
```

---

**Right now on your pod, run this:**

```bash
./fix_compatibility.sh
```

That's it! The fix script will handle everything with the virtual environment. 🚀

