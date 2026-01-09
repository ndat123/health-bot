"""
Script để hiển thị thông tin từ model đã train
Bao gồm: disease mapping, trainer state, và evaluation results
"""

import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def display_disease_mapping(mapping_file):
    """Hiển thị mapping bệnh"""
    print("=" * 50)
    print("DISEASE MAPPING")
    print("=" * 50)

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    print(f"Tổng số loại bệnh: {len(mapping['disease_to_id'])}")
    print("\nMapping từ tên bệnh sang ID:")
    for disease, idx in mapping['disease_to_id'].items():
        print(f"  {idx}: {disease}")

    print("\nMapping từ ID sang tên bệnh:")
    for idx, disease in mapping['id_to_disease'].items():
        print(f"  {idx}: {disease}")

def display_trainer_state(state_file):
    """Hiển thị thông tin training state"""
    print("\n" + "=" * 50)
    print("TRAINER STATE")
    print("=" * 50)

    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)

    print(f"Best global step: {state['best_global_step']}")
    print(f"Best metric (F1): {state['best_metric']:.6f}")
    print(f"Current epoch: {state['epoch']}")
    print(f"Global step: {state['global_step']}")
    print(f"Max steps: {state['max_steps']}")
    print(f"Train batch size: {state['train_batch_size']}")

    # Hiển thị evaluation results cuối cùng
    if state['log_history']:
        last_eval = None
        for log in state['log_history']:
            if 'eval_accuracy' in log:
                last_eval = log

        if last_eval:
            print("\nKết quả evaluation cuối cùng:")
            print(f"  Accuracy: {last_eval['eval_accuracy']:.6f}")
            print(f"  F1: {last_eval['eval_f1']:.6f}")
            print(f"  Loss: {last_eval['eval_loss']:.4f}")
            print(f"  Top-3 Accuracy: {last_eval.get('eval_top3_accuracy', 'N/A')}")
            print(f"  Top-5 Accuracy: {last_eval.get('eval_top5_accuracy', 'N/A')}")

def display_model_info(model_dir):
    """Hiển thị thông tin model"""
    print("\n" + "=" * 50)
    print("MODEL INFO")
    print("=" * 50)

    # Load config
    config_file = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"Model type: {config.get('model_type', 'Unknown')}")
        print(f"Number of labels: {config.get('num_labels', 'Unknown')}")
        print(f"Hidden size: {config.get('hidden_size', 'Unknown')}")
        print(f"Number of layers: {config.get('num_hidden_layers', 'Unknown')}")

    # Kích thước thư mục
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    print(f"Kích thước model: {total_size / (1024*1024):.2f} MB")

def main():
    model_dir = "./chatbot_model"
    mapping_file = os.path.join(model_dir, "disease_mapping.json")
    state_file = os.path.join(model_dir, "checkpoint-1025", "trainer_state.json")

    # Hiển thị disease mapping
    if os.path.exists(mapping_file):
        display_disease_mapping(mapping_file)
    else:
        print("Không tìm thấy file disease_mapping.json")

    # Hiển thị trainer state
    if os.path.exists(state_file):
        display_trainer_state(state_file)
    else:
        print("Không tìm thấy file trainer_state.json")

    # Hiển thị model info
    if os.path.exists(model_dir):
        display_model_info(model_dir)

    print("\n" + "=" * 50)
    print("HOÀN THÀNH HIỂN THỊ THÔNG TIN")
    print("=" * 50)

if __name__ == "__main__":
    main()