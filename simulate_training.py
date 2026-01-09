"""
Simulate Training - Giả lập quá trình training với metrics giả
Hữu ích để test UI, demo, hoặc khi train thật quá lâu
"""

import os
import json
import time
import random
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers import AutoConfig
from safetensors.torch import save_file
import sys
import io

# Set UTF-8 encoding cho Windows console
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# ==================== CẤU HÌNH ====================
OUTPUT_DIR = "./chatbot_model"
DATA_FILE = "./ViMedical_Disease.csv"
NUM_EPOCHS = 25
STEPS_PER_EPOCH = 450  # Giả lập số steps mỗi epoch
TOTAL_STEPS = NUM_EPOCHS * STEPS_PER_EPOCH

# Metrics mục tiêu (sẽ đạt được ở cuối training)
TARGET_ACCURACY = 0.92  # 92%
TARGET_F1 = 0.90  # 90%
TARGET_PRECISION = 0.91  # 91%
TARGET_RECALL = 0.90  # 90%

# Metrics ban đầu (ở epoch 1)
INITIAL_ACCURACY = 0.15  # 15%
INITIAL_F1 = 0.10  # 10%
INITIAL_PRECISION = 0.12  # 12%
INITIAL_RECALL = 0.10  # 10%
INITIAL_LOSS = 4.5

# ==================== HELPER FUNCTIONS ====================
def load_or_create_mapping():
    """Load hoặc tạo disease mapping"""
    mapping_path = os.path.join(OUTPUT_DIR, "disease_mapping.json")
    
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"✓ Đã load mapping: {len(mapping['id_to_disease'])} classes")
        return mapping
    else:
        # Tạo mapping giả từ data
        import pandas as pd
        df = pd.read_csv(DATA_FILE, encoding='utf-8')
        diseases = df['Disease'].unique()
        
        disease_to_id = {disease: idx for idx, disease in enumerate(diseases)}
        id_to_disease = {idx: disease for disease, idx in disease_to_id.items()}
        
        mapping = {
            'disease_to_id': disease_to_id,
            'id_to_disease': {str(k): v for k, v in id_to_disease.items()}
        }
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Đã tạo mapping: {len(disease_to_id)} classes")
        return mapping

def create_fake_model(num_labels):
    """Tạo model giả (chỉ để lưu checkpoint)"""
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.num_labels = num_labels
    
    model = AutoModelForSequenceClassification.from_config(config)
    return model

def calculate_metrics(epoch, total_epochs):
    """Tính toán metrics giả dựa trên epoch"""
    # Progress từ 0.0 đến 1.0
    progress = epoch / total_epochs
    
    # Sử dụng cosine curve để tạo progress mượt mà
    # Bắt đầu nhanh, sau đó chậm dần
    smooth_progress = 1 - np.cos(progress * np.pi / 2)
    
    # Thêm một chút noise để tự nhiên hơn
    noise = random.uniform(-0.02, 0.02)
    
    # Tính metrics
    accuracy = INITIAL_ACCURACY + (TARGET_ACCURACY - INITIAL_ACCURACY) * smooth_progress + noise
    f1 = INITIAL_F1 + (TARGET_F1 - INITIAL_F1) * smooth_progress + noise
    precision = INITIAL_PRECISION + (TARGET_PRECISION - INITIAL_PRECISION) * smooth_progress + noise
    recall = INITIAL_RECALL + (TARGET_RECALL - INITIAL_RECALL) * smooth_progress + noise
    
    # Loss giảm dần
    loss = INITIAL_LOSS * (1 - smooth_progress * 0.9) + random.uniform(-0.1, 0.1)
    
    # Top-K accuracy
    top3_acc = accuracy + random.uniform(0.05, 0.15)
    top5_acc = accuracy + random.uniform(0.10, 0.25)
    
    # Đảm bảo metrics trong khoảng hợp lý
    accuracy = max(0.0, min(1.0, accuracy))
    f1 = max(0.0, min(1.0, f1))
    precision = max(0.0, min(1.0, precision))
    recall = max(0.0, min(1.0, recall))
    loss = max(0.1, loss)
    top3_acc = max(0.0, min(1.0, top3_acc))
    top5_acc = max(0.0, min(1.0, top5_acc))
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
    }

def simulate_epoch(epoch, total_epochs, num_labels):
    """Giả lập một epoch"""
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch}/{total_epochs}")
    print(f"{'='*70}")
    
    # Tính metrics
    metrics = calculate_metrics(epoch, total_epochs)
    
    # Giả lập training steps
    print(f"\nĐang train...")
    for step in range(0, STEPS_PER_EPOCH, 50):
        # Progress bar giả
        progress = (step / STEPS_PER_EPOCH) * 100
        bar_length = 40
        filled = int(bar_length * step / STEPS_PER_EPOCH)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Loss giảm dần trong epoch
        step_loss = metrics['loss'] * (1 - (step / STEPS_PER_EPOCH) * 0.1)
        
        print(f"  Step {step:4d}/{STEPS_PER_EPOCH} [{bar}] {progress:5.1f}% - Loss: {step_loss:.4f}", end='\r')
        time.sleep(0.01)  # Giả lập thời gian train
    
    print(f"  Step {STEPS_PER_EPOCH:4d}/{STEPS_PER_EPOCH} [{'█' * bar_length}] 100.0% - Loss: {metrics['loss']:.4f}")
    
    # Hiển thị metrics
    print(f"\n📊 Metrics:")
    print(f"  Loss:           {metrics['loss']:.4f}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:      {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:         {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:       {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    
    # Lưu checkpoint
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch * STEPS_PER_EPOCH}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tạo model giả và lưu
    model = create_fake_model(num_labels)
    model_state = model.state_dict()
    
    # Lưu safetensors với retry logic
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)  # Đợi một chút để giải phóng file locks
            save_file(model_state, safetensors_path)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Đợi lâu hơn
            else:
                print(f"  ⚠️ Không thể lưu checkpoint: {str(e)[:50]}")
                # Tiếp tục với các bước khác
    
    # Lưu config với retry logic
    try:
        config = model.config
        config.save_pretrained(checkpoint_dir)
    except Exception as e:
        print(f"  ⚠️ Không thể lưu config: {str(e)[:50]}")
    
    # Lưu training state giả
    trainer_state = {
        'epoch': float(epoch),
        'global_step': epoch * STEPS_PER_EPOCH,
        'total_flos': 0,
        'log_history': [],
        'best_metric': metrics['f1'],
        'best_model_checkpoint': checkpoint_dir if metrics['f1'] > 0.85 else None,
    }
    
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(trainer_state_path, 'w', encoding='utf-8') as f:
        json.dump(trainer_state, f, indent=2)
    
    print(f"✓ Checkpoint saved: {checkpoint_dir}")
    
    return metrics

def simulate_training():
    """Giả lập toàn bộ quá trình training"""
    print("="*70)
    print("SIMULATE TRAINING - GIẢ LẬP TRAINING")
    print("="*70)
    print("⚠️  Đây là giả lập - không train thật!")
    print("   Metrics được tạo tự động để test/demo")
    print("="*70)
    
    # Load mapping
    mapping = load_or_create_mapping()
    num_labels = len(mapping['id_to_disease'])
    
    print(f"\n📊 Thông tin:")
    print(f"  Số classes: {num_labels}")
    print(f"  Số epochs: {NUM_EPOCHS}")
    print(f"  Steps per epoch: {STEPS_PER_EPOCH}")
    print(f"  Tổng số steps: {TOTAL_STEPS}")
    print(f"  Mục tiêu Accuracy: {TARGET_ACCURACY*100:.1f}%")
    print(f"  Mục tiêu F1: {TARGET_F1*100:.1f}%")
    
    # Tạo output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Giả lập từng epoch
    all_metrics = []
    best_f1 = 0.0
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        metrics = simulate_epoch(epoch, NUM_EPOCHS, num_labels)
        all_metrics.append(metrics)
        
        # Track best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_epoch = epoch
        
        # Early stopping giả (nếu đạt mục tiêu)
        if metrics['accuracy'] >= TARGET_ACCURACY and metrics['f1'] >= TARGET_F1:
            print(f"\n🎉 Đã đạt mục tiêu ở epoch {epoch}!")
            print(f"   Accuracy: {metrics['accuracy']*100:.2f}% (mục tiêu: {TARGET_ACCURACY*100:.1f}%)")
            print(f"   F1: {metrics['f1']*100:.2f}% (mục tiêu: {TARGET_F1*100:.1f}%)")
            break
        
        time.sleep(0.5)  # Giả lập thời gian giữa các epoch
    
    elapsed_time = time.time() - start_time
    
    # Lưu model cuối cùng
    print(f"\n{'='*70}")
    print("LƯU MODEL CUỐI CÙNG")
    print(f"{'='*70}")
    
    final_model = create_fake_model(num_labels)
    final_model_state = final_model.state_dict()
    
    # Lưu vào OUTPUT_DIR với retry logic
    safetensors_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    
    # Retry logic để tránh lỗi file lock trên Windows
    max_retries = 5
    saved_successfully = False
    for attempt in range(max_retries):
        try:
            time.sleep(2)  # Đợi một chút để giải phóng file locks
            
            # Kiểm tra xem file có đang được sử dụng không
            if os.path.exists(safetensors_path):
                try:
                    # Thử mở file để kiểm tra
                    with open(safetensors_path, 'rb') as f:
                        pass
                except PermissionError:
                    print(f"  ⚠️ File đang được sử dụng, đợi thêm...")
                    time.sleep(3)
                    continue
            
            save_file(final_model_state, safetensors_path)
            saved_successfully = True
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Tăng dần thời gian đợi
                print(f"  ⚠️ Lỗi khi lưu (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}")
                print(f"  → Đợi {wait_time} giây rồi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ Không thể lưu model cuối cùng sau {max_retries} lần thử")
                print(f"  → Lỗi: {str(e)[:100]}")
                print(f"  → Model đã được lưu ở checkpoint trước đó (checkpoint-{best_epoch * STEPS_PER_EPOCH})")
    
    if saved_successfully:
        print(f"  ✓ Model safetensors đã được lưu")
    
    # Lưu config và tokenizer
    try:
        time.sleep(1)
        config = final_model.config
        config.save_pretrained(OUTPUT_DIR)
        print(f"  ✓ Config đã được lưu")
    except Exception as e:
        print(f"  ⚠️ Lỗi khi lưu config: {str(e)[:100]}")
    
    try:
        time.sleep(1)
        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  ✓ Tokenizer đã được lưu")
    except Exception as e:
        print(f"  ⚠️ Lỗi khi lưu tokenizer: {str(e)[:100]}")
    
    if saved_successfully:
        print(f"\n✓ Model đã được lưu tại: {OUTPUT_DIR}")
    else:
        print(f"\n⚠️ Model không thể lưu vào {OUTPUT_DIR}, nhưng checkpoints đã được lưu")
    
    # Tóm tắt kết quả
    print(f"\n{'='*70}")
    print("TÓM TẮT KẾT QUẢ")
    print(f"{'='*70}")
    
    final_metrics = all_metrics[-1]
    print(f"\n📊 Metrics cuối cùng:")
    print(f"  Loss:           {final_metrics['loss']:.4f}")
    print(f"  Accuracy:       {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:      {final_metrics['precision']:.4f} ({final_metrics['precision']*100:.2f}%)")
    print(f"  Recall:         {final_metrics['recall']:.4f} ({final_metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:       {final_metrics['f1']:.4f} ({final_metrics['f1']*100:.2f}%)")
    print(f"  Top-3 Accuracy: {final_metrics['top3_accuracy']:.4f} ({final_metrics['top3_accuracy']*100:.2f}%)")
    print(f"  Top-5 Accuracy: {final_metrics['top5_accuracy']:.4f} ({final_metrics['top5_accuracy']*100:.2f}%)")
    
    print(f"\n🏆 Best Model:")
    print(f"  Epoch: {best_epoch}")
    print(f"  F1: {best_f1:.4f} ({best_f1*100:.2f}%)")
    
    print(f"\n⏱️  Thời gian: {elapsed_time:.2f} giây (giả lập)")
    
    # Phân tích
    print(f"\n{'='*70}")
    print("PHÂN TÍCH KẾT QUẢ")
    print(f"{'='*70}")
    
    if final_metrics['accuracy'] >= 0.90:
        print("🎉 XUẤT SẮC! Model đã đạt mục tiêu 90-95%!")
    elif final_metrics['accuracy'] >= 0.80:
        print("✓ Rất tốt! Model đã đạt > 80% accuracy")
    elif final_metrics['accuracy'] >= 0.50:
        print("✓ Tốt! Model đã học được nhiều")
    else:
        print("⚠️ Model cần train thêm")
    
    print(f"\n{'='*70}")
    print("HOÀN THÀNH!")
    print(f"{'='*70}")
    print("\n💡 Lưu ý:")
    print("  - Đây là giả lập, model không thực sự được train")
    print("  - Để train thật, sử dụng: resume_training_90_95_percent.py")
    print("  - Model giả lập có thể được sử dụng để test UI/demo")

if __name__ == "__main__":
    try:
        simulate_training()
    except KeyboardInterrupt:
        print("\n\n⚠️ Giả lập bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

