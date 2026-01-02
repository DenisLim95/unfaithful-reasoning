#!/bin/bash
#
# Setup script for Remote GPU Pod Environment
# Run this on your GPU pod to install all Phase 3 dependencies
#

set -e  # Exit on error

echo "============================================================"
echo "REMOTE ENVIRONMENT SETUP - Phase 3 Dependencies"
echo "============================================================"
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "cot-unfaith"; then
    echo "Creating conda environment: cot-unfaith"
    conda create -n cot-unfaith python=3.10 -y
else
    echo "✓ Conda environment 'cot-unfaith' already exists"
fi

echo ""
echo "Activating environment..."
# Note: This doesn't work in scripts, user must manually activate
echo "⚠️  Please activate the environment manually:"
echo "    conda activate cot-unfaith"
echo ""
echo "Then run the rest of this script..."
echo ""

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "cot-unfaith" ]]; then
    echo "❌ ERROR: Please activate the conda environment first:"
    echo "    conda activate cot-unfaith"
    echo "    bash setup_remote_env.sh"
    exit 1
fi

echo "============================================================"
echo "Installing PyTorch with CUDA Support"
echo "============================================================"
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    
    # Install PyTorch with CUDA 11.8 support
    echo "Installing PyTorch 2.2.0 with CUDA 11.8..."
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️  No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "============================================================"
echo "Installing Other Dependencies"
echo "============================================================"
echo ""

# Install transformers and other core dependencies
pip install transformers==4.39.0
pip install accelerate==0.27.0

# Mechanistic interpretability tools (Phase 3)
echo "Installing mechanistic interpretability tools..."
pip install transformer-lens==1.17.0
pip install nnsight==0.2.6

# Data processing
echo "Installing data processing libraries..."
pip install pandas==2.2.0
pip install numpy>=1.20.0,<2.0.0
pip install scipy==1.12.0
pip install scikit-learn==1.4.0

# Data formats
pip install jsonlines==4.0.0
pip install pyyaml==6.0.1

# Visualization
echo "Installing visualization libraries..."
pip install matplotlib==3.8.2
pip install seaborn==0.13.2
pip install plotly==5.18.0

# Utilities
pip install tqdm==4.66.1

# Testing
pip install pytest==8.0.0
pip install pytest-cov==4.1.0

echo ""
echo "============================================================"
echo "Verification"
echo "============================================================"
echo ""

# Verify critical imports
echo "Verifying installations..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "❌ PyTorch failed"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')" || echo "❌ CUDA check failed"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" || echo "❌ Transformers failed"
python -c "import transformer_lens; print('✓ TransformerLens OK')" || echo "❌ TransformerLens failed"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')" || echo "❌ Pandas failed"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" || echo "❌ NumPy failed"
python -c "import sklearn; print(f'✓ Scikit-learn {sklearn.__version__}')" || echo "❌ Scikit-learn failed"
python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')" || echo "❌ Matplotlib failed"
python -c "import jsonlines; print('✓ jsonlines OK')" || echo "❌ jsonlines failed"

echo ""
echo "============================================================"
echo "Testing Phase 3 Imports"
echo "============================================================"
echo ""

# Test that Phase 3 scripts can import
python -c "from src.mechanistic.contracts import Phase3Config; print('✓ Phase 3 contracts import OK')" || echo "❌ Phase 3 imports failed"

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "You can now run Phase 3:"
echo "  bash run_phase3.sh"
echo ""
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

