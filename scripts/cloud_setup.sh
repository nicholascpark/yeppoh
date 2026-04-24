#!/bin/bash
# Quick setup on a fresh cloud GPU instance (Vast.ai, RunPod, Lambda, etc.)
#
# Usage:
#   ssh root@<cloud-ip>
#   curl -sSL https://raw.githubusercontent.com/nicholascpark/yeppoh/main/scripts/cloud_setup.sh | bash
#
# Or if you've already cloned:
#   bash scripts/cloud_setup.sh

set -e

echo "═══ YEPPOH CLOUD SETUP ═══"

# Clone if not already in repo
if [ ! -f "pyproject.toml" ]; then
    echo "Cloning repo..."
    git clone https://github.com/nicholascpark/yeppoh.git
    cd yeppoh
fi

# Install deps
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Verify GPU
echo ""
echo "GPU check:"
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Verify Genesis
echo ""
echo "Genesis check:"
python -c "import genesis; print(f'  Genesis version: {genesis.__version__}')" 2>/dev/null || echo "  Genesis import failed — may need: pip install genesis-world"

echo ""
echo "═══ SETUP COMPLETE ═══"
echo ""
echo "Run training:"
echo "  python experiments/01_basic_blob.py"
echo ""
echo "Run in background (survives SSH disconnect):"
echo "  nohup python experiments/02_locomotion.py > train.log 2>&1 &"
echo "  tail -f train.log"
echo ""
echo "Or use tmux:"
echo "  tmux new -s train"
echo "  python experiments/02_locomotion.py"
echo "  # Ctrl+B, D to detach. 'tmux attach -t train' to reconnect."
