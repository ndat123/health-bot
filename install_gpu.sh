#!/bin/bash

echo "========================================"
echo "CAI DAT GPU CHO PYTHON"
echo "========================================"
echo ""

echo "[1/3] Go PyTorch CPU version..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "[2/3] Cai dat PyTorch voi CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[3/3] Kiem tra cai dat..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "========================================"
echo "HOAN THANH!"
echo "========================================"
echo ""
echo "Bay gio ban co the chay:"
echo "  python resume_training_90_95_percent.py"
echo ""





