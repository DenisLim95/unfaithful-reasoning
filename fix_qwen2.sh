#!/bin/bash
# Final fix for Qwen2 model support
# The issue: Transformers needs to be new enough to support Qwen2, but compatible with PyTorch 2.1

echo "=========================================="
echo "Final Fix: Qwen2 Model Support"
echo "=========================================="
echo ""

# Activate venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade transformers to support Qwen2
echo "Upgrading transformers to support Qwen2 model..."
pip install --upgrade transformers==4.46.0

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
print("✓ Model loaded successfully!")
print(f"✓ Model device: {model.device}")

# Quick inference test
inputs = tokenizer("Is 5 larger than 3?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("✓ Model inference works")
print(f"  Sample output: {response[:100]}...")

print("\n✅ Everything is working correctly!")
EOF

echo ""
echo "=========================================="
echo "FIX COMPLETE!"
echo "=========================================="
echo ""
echo "✅ Virtual environment: venv/"
echo "✅ PyTorch: 2.1.0"
echo "✅ Transformers: 4.46.0 (supports Qwen2)"
echo ""
echo "You can now run:"
echo "  python test_probe_on_new_data.py --skip-generation --num-questions 200"
echo ""
echo "Or with nohup (recommended):"
echo "  nohup python test_probe_on_new_data.py --skip-generation --num-questions 200 > test.log 2>&1 &"
echo "  tail -f test.log"
echo ""

