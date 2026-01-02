#!/bin/bash
# Quick fix for answer extraction on remote pod

echo "============================================================"
echo "STEP 1: Copy files to pod"
echo "============================================================"
echo ""
echo "Run this command (replace <POD_IP> with your pod's IP):"
echo ""
echo "  scp /Users/denislim/workspace/mats-10.0/rescore_fix.tar.gz root@<POD_IP>:/unfaithful-reasoning/"
echo ""
echo "Then SSH to your pod and run:"
echo ""
echo "  cd /unfaithful-reasoning"
echo "  tar -xzf rescore_fix.tar.gz"
echo "  source venv/bin/activate"
echo "  python rescore_responses.py"
echo ""
echo "This will re-extract answers from existing responses."
echo "No model loading needed - runs in ~5 seconds!"
echo ""
echo "============================================================"

