#!/bin/bash
# Fix torch version mismatch
# Run this to align torch, torchvision, and torchaudio versions

echo "=========================================="
echo "Fixing PyTorch Version Mismatch"
echo "=========================================="
echo ""

# Activate venv
source venv/bin/activate

echo "Current versions:"
python -c "import torch; print(f'  torch:       {torch.__version__}')" 2>/dev/null || echo "  torch: not installed"
python -c "import torchvision; print(f'  torchvision: {torchvision.__version__}')" 2>/dev/null || echo "  torchvision: not installed"
python -c "import torchaudio; print(f'  torchaudio:  {torchaudio.__version__}')" 2>/dev/null || echo "  torchaudio: not installed"

echo ""
echo "Reinstalling PyTorch packages to match versions..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Installing matching versions..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "New versions:"
python -c "import torch; print(f'  torch:       {torch.__version__}')"
python -c "import torchvision; print(f'  torchvision: {torchvision.__version__}')"
python -c "import torchaudio; print(f'  torchaudio:  {torchaudio.__version__}')"

echo ""
echo "Verifying CUDA..."
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Verifying Qwen2 support..."
python -c "from transformers import Qwen2Config; print('  ✅ Qwen2 supported')"

echo ""
echo "Testing model loading..."
python << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"  Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("  ✓ Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("  ✓ Model loaded")
print(f"  ✓ Device: {model.device}")

# Quick test
inputs = tokenizer("Test", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5)
print("  ✓ Inference works")
EOF

echo ""
echo "=========================================="
echo "✅ FIXED!"
echo "=========================================="
echo ""
echo "All PyTorch packages are now compatible"
echo "You can now run:"
echo "  python test_probe_on_new_data.py --skip-generation --num-questions 200"
echo ""

