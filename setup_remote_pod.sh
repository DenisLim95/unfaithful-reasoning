#!/bin/bash
# Setup script for new remote GPU pod
# Run this on the remote machine after copying your code

set -e  # Exit on error

echo "=========================================="
echo "Setting Up Remote Pod Environment"
echo "=========================================="
echo ""

# 1. Check Python version
echo "[1/7] Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found!"
    exit 1
fi
echo "✓ Python 3 found"
echo ""

# 2. Create virtual environment
echo "[2/7] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists, removing old one..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created and activated"
echo ""

# 3. Create directory structure
echo "[3/7] Creating directory structure..."
mkdir -p data/raw
mkdir -p data/responses
mkdir -p data/processed
mkdir -p data/activations
mkdir -p data/test_activations
mkdir -p results/probe_results
mkdir -p results/activation_visualizations
echo "✓ Directories created"
echo ""

# 4. Install Python dependencies
echo "[4/7] Installing Python dependencies..."
pip install --upgrade pip

# Install from requirements.txt to match original environment
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing manually..."
    # Fallback to manual install
    pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers>=4.39.0
    pip install pandas>=2.0.0
    pip install scikit-learn>=1.3.0
    pip install matplotlib>=3.7.0
    pip install seaborn>=0.12.0
    pip install tqdm>=4.65.0
    pip install accelerate>=0.27.0
fi

echo "✓ Dependencies installed"
echo ""

# 5. Verify GPU access
echo "[5/7] Verifying GPU access..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 6. Test model loading
echo "[6/7] Testing model loading..."
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ Model loaded")
    
    # Quick test
    inputs = tokenizer("Test", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    print("✓ Model inference works")
    print(f"✓ Model device: {model.device}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
EOF

echo ""

# 7. Summary
echo "[7/7] Setup Summary"
echo "=========================================="
echo "✓ Python installed"
echo "✓ Virtual environment created (venv/)"
echo "✓ Directory structure created"
echo "✓ Dependencies installed"
echo "✓ GPU verified"
echo "✓ Model loading tested"
echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: Activate venv before running scripts:"
echo "   source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. python test_probe_on_new_data.py --skip-generation"
echo ""

