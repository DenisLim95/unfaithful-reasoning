#!/bin/bash
# Quick fix for PyTorch/Transformers compatibility issue
# Run this on the remote pod if you got the register_pytree_node error

echo "=========================================="
echo "Fixing PyTorch/Transformers Compatibility"
echo "=========================================="
echo ""

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

echo "Uninstalling incompatible versions..."
pip uninstall -y torch torchvision torchaudio transformers

echo ""
echo "Installing compatible versions..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.0  # Supports Qwen2 model

echo ""
echo "Verifying installation..."
python3 << 'EOF'
import torch
import transformers

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Transformers version: {transformers.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# Test model loading
from transformers import AutoModelForCausalLM, AutoTokenizer

print("\nTesting model loading...")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✓ Model loaded")
print(f"✓ Model device: {model.device}")

# Quick inference test
inputs = tokenizer("Test", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
print("✓ Model inference works")

print("\n✅ Everything is working correctly!")
EOF

echo ""
echo "=========================================="
echo "FIX COMPLETE!"
echo "=========================================="
echo ""
echo "⚠️  Virtual environment is active"
echo "To use it in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "You can now run:"
echo "python test_probe_on_new_data.py --skip-generation"
echo ""
echo "Or with nohup (to keep running after disconnect):"
echo "nohup python test_probe_on_new_data.py --skip-generation --num-questions 200 > test.log 2>&1 &"


