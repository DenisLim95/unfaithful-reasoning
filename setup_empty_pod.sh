#!/bin/bash
# Complete setup script for empty RunPod
# Run this AFTER transferring code to the pod

set -e  # Exit on error

echo "=========================================="
echo "Complete Setup for Empty RunPod"
echo "=========================================="
echo ""

# Check Python 3
echo "[1/8] Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python 3 not found. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-pip python3-venv
fi
python3 --version
echo "✓ Python 3 found"
echo ""

# Check GPU
echo "[2/8] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo "✓ GPU detected"
else
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available."
fi
echo ""

# Check disk space
echo "[3/8] Checking disk space..."
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE" -lt 10 ]; then
    echo "⚠️  WARNING: Less than 10GB available. You may run out of space."
else
    echo "✓ Disk space OK (${AVAILABLE}GB available)"
fi
echo ""

# Find project directory
echo "[4/8] Finding project directory..."
if [ -f "setup_remote_pod.sh" ]; then
    PROJECT_DIR=$(pwd)
elif [ -f "requirements.txt" ]; then
    PROJECT_DIR=$(pwd)
else
    # Try common locations
    if [ -d "/root/workspace/mats-10.0" ]; then
        PROJECT_DIR="/root/workspace/mats-10.0"
    elif [ -d "/root/mats-10.0" ]; then
        PROJECT_DIR="/root/mats-10.0"
    else
        echo "❌ ERROR: Cannot find project directory."
        echo "   Please cd into the project directory first."
        exit 1
    fi
fi

cd "$PROJECT_DIR"
echo "✓ Project directory: $PROJECT_DIR"
echo ""

# Create virtual environment
echo "[5/8] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  venv already exists, removing old one..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created"
echo ""

# Upgrade pip
echo "[6/8] Upgrading pip..."
pip install --upgrade pip -q
echo "✓ Pip upgraded"
echo ""

# Install dependencies
echo "[7/8] Installing dependencies..."
echo "   This will take 5-10 minutes..."
echo ""

# Install PyTorch with CUDA first
echo "   Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# Install other dependencies
if [ -f "requirements.txt" ]; then
    echo "   Installing from requirements.txt..."
    pip install -r requirements.txt -q
else
    echo "   ⚠️  requirements.txt not found, installing core packages..."
    pip install transformers accelerate pandas scikit-learn numpy matplotlib seaborn tqdm -q
fi

echo "✓ Dependencies installed"
echo ""

# Create directories
echo "[8/8] Creating directory structure..."
mkdir -p data/{raw,responses,processed,activations,test_activations}
mkdir -p results/{probe_results,activation_visualizations}
echo "✓ Directories created"
echo ""

# Verify installation
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "❌ PyTorch failed"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')" || echo "❌ CUDA check failed"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
fi

python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" || echo "❌ Transformers failed"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')" || echo "❌ Pandas failed"
python -c "import sklearn; print(f'✓ Scikit-learn {sklearn.__version__}')" || echo "❌ Scikit-learn failed"

echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: Always activate venv before running scripts:"
echo "   source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. python scripts/01_generate_questions.py --num-pairs 100"
echo "3. python scripts/02_generate_responses.py --questions data/raw/questions.json"
echo ""
echo "The model will download automatically on first use (~3-4GB)."
echo ""
