#!/bin/bash
#
# Installation script for Remote GPU Pod
# Use this to install all dependencies with compatible versions
#

set -e

echo "=========================================="
echo "Installing Dependencies for Phase 3"
echo "=========================================="
echo ""

# Check we're in conda environment
if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    echo "⚠️  You're in the base environment!"
    echo "   It's better to create a new environment:"
    echo ""
    echo "   conda create -n cot-unfaith python=3.10 -y"
    echo "   conda activate cot-unfaith"
    echo "   bash INSTALL_ON_REMOTE.sh"
    echo ""
    read -p "Continue anyway in base environment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

# Check for CUDA
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU found:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    HAS_GPU=true
else
    echo "⚠️  No GPU detected - will install CPU-only PyTorch"
    HAS_GPU=false
fi
echo ""

# Install PyTorch first (most critical)
echo "=========================================="
echo "Step 1: Installing PyTorch"
echo "=========================================="
echo ""

if [ "$HAS_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    # Let pip find the best CUDA version automatically
    pip install torch torchvision torchaudio
else
    echo "Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
if [ "$HAS_GPU" = true ]; then
    python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')" || echo "  (CUDA info not available)"
fi
echo ""

# Install all other dependencies
echo "=========================================="
echo "Step 2: Installing Other Dependencies"
echo "=========================================="
echo ""

pip install -r requirements.txt

# Verify critical packages
echo ""
echo "=========================================="
echo "Step 3: Verification"
echo "=========================================="
echo ""

python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" || echo "❌ Transformers failed"
python -c "import transformer_lens; print('✓ TransformerLens OK')" || echo "❌ TransformerLens failed"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')" || echo "❌ Pandas failed"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" || echo "❌ NumPy failed"
python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')" || echo "❌ Matplotlib failed"
python -c "import sklearn; print(f'✓ Scikit-learn {sklearn.__version__}')" || echo "❌ Scikit-learn failed"

# Test Phase 3 imports
echo ""
python -c "from src.mechanistic.contracts import Phase3Config; print('✓ Phase 3 imports OK')" || echo "❌ Phase 3 imports failed"

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "You can now run Phase 3:"
echo "  bash run_phase3.sh"
echo ""

