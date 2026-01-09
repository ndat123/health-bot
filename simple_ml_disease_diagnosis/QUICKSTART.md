# ⚡ Quick Start - Hệ Thống Chẩn Đoán Bệnh ML

## 🚀 Cài Đặt Nhanh (3 bước)

### Bước 1: Cài đặt dependencies
```bash
cd simple_ml_disease_diagnosis
pip install -r requirements.txt
```

### Bước 2: Training model
```bash
python train_model.py
```
⏱️ Thời gian: 2-5 phút

### Bước 3: Sử dụng
```bash
python inference.py
```

Sau đó nhập triệu chứng:
```
🧑 Triệu chứng của bạn: Đau đầu, sốt cao, mệt mỏi
```

## 🎯 Các Cách Sử Dụng Khác

### 1️⃣ Dự đoán 1 triệu chứng
```bash
python inference.py --symptoms "Đau đầu, sốt cao"
```

### 2️⃣ Xem demo
```bash
python demo.py
```

### 3️⃣ Dự đoán từ file
```bash
# Tạo file symptoms.txt với mỗi dòng là 1 triệu chứng
python inference.py --batch-file symptoms.txt --output results.json
```

## 📖 Xem Thêm

- [README_VI.md](README_VI.md) - Hướng dẫn đầy đủ
- [demo.py](demo.py) - Các ví dụ code

## ⚠️ Lưu Ý

**Đây KHÔNG phải chẩn đoán y tế!** Luôn tham khảo ý kiến bác sĩ.


