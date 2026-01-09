"""
Script để chuyển đổi checkpoint từ .bin sang safetensors
"""
import os
import sys
import io

# Fix encoding cho Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import torch
from safetensors.torch import save_file

def convert_checkpoint_to_safetensors(checkpoint_dir):
    """Chuyển đổi pytorch_model.bin sang model.safetensors"""
    bin_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    safetensors_file = os.path.join(checkpoint_dir, "model.safetensors")
    
    if not os.path.exists(bin_file):
        print(f"❌ Không tìm thấy {bin_file}")
        return False
    
    if os.path.exists(safetensors_file):
        print(f"✓ {safetensors_file} đã tồn tại, bỏ qua")
        return True
    
    print(f"Đang chuyển đổi {bin_file} sang safetensors...")
    try:
        # Load state dict từ .bin
        state_dict = torch.load(bin_file, map_location='cpu', weights_only=False)
        
        # Lưu dưới dạng safetensors
        save_file(state_dict, safetensors_file)
        print(f"✓ Đã chuyển đổi thành công: {safetensors_file}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        # Tìm checkpoint mới nhất
        base_dir = "./chatbot_model"
        checkpoints = []
        for item in os.listdir(base_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, item)):
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append((step, os.path.join(base_dir, item)))
                except:
                    continue
        
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            checkpoint_dir = checkpoints[0][1]
        else:
            checkpoint_dir = "./chatbot_model/checkpoint-4824"
    
    print(f"Chuyển đổi checkpoint: {checkpoint_dir}")
    convert_checkpoint_to_safetensors(checkpoint_dir)

