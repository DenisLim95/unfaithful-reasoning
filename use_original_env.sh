#!/bin/bash
# Use the EXACT same environment from Phase 1-3
# This installs from requirements.txt to ensure consistency

echo "=========================================="
echo "Installing Original Environment"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Created venv"
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt (your original environment)..."
    pip install -r requirements.txt
    echo "✓ Installed all dependencies"
else
    echo "❌ requirements.txt not found!"
    echo "Please copy requirements.txt to this directory"
    exit 1
fi

echo ""
echo "Verifying installation..."
python3 << 'EOF'
import torch
import transformers
import pandas
import sklearn
import matplotlib

print(f"✓ PyTorch:      {torch.__version__}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ Pandas:       {pandas.__version__}")
print(f"✓ Scikit-learn: {sklearn.__version__}")
print(f"✓ CUDA:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU:          {torch.cuda.get_device_name(0)}")

print("\nTesting model loading...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✓ Model loaded")
print(f"✓ Device: {model.device}")

# Quick test
inputs = tokenizer("Test", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
print("✓ Inference works")

print("\n✅ Environment matches Phase 1-3!")
EOF

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "✅ Same environment as Phase 1-3"
echo "✅ Installed from: requirements.txt"
echo ""
echo "You can now run:"
echo "  python test_probe_on_new_data.py --skip-generation --num-questions 200"
echo ""
echo "Or with nohup:"
echo "  nohup python test_probe_on_new_data.py --skip-generation --num-questions 200 > test.log 2>&1 &"
echo ""

