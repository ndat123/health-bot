# 📊 Kết Quả Training - Hệ Thống Chẩn Đoán Bệnh ML

## 🎯 Tổng Quan

Training đã hoàn thành thành công cho 4 models ML trên dataset ViMedical_Disease.

**Thời gian training:** 08/01/2026 00:12:45  
**Dataset:** 12,066 mẫu, 603 loại bệnh  
**Train/Test split:** 80% / 20%  
**Số features (TF-IDF):** 5,000

---

## 📈 Kết Quả Chi Tiết

### 🏆 Model Tốt Nhất: SVM (Support Vector Machine)

| Metric | Giá Trị |
|--------|---------|
| **Accuracy** | **49.34%** |
| **Precision** | 47.34% |
| **Recall** | 49.34% |
| **F1-Score** | 46.53% |
| **Top-3 Accuracy** | **62.77%** |
| **Top-5 Accuracy** | **67.70%** |
| **Training Time** | 4.55s |

### So Sánh Tất Cả Models

| Model | Accuracy | F1-Score | Top-3 Acc | Top-5 Acc | Training Time |
|-------|----------|----------|-----------|-----------|---------------|
| **SVM** | **49.34%** | **46.53%** | **62.77%** | **67.70%** | 4.55s |
| **Logistic Regression** | 42.83% | 37.15% | 59.12% | 64.39% | 15.88s |
| **Naive Bayes** | 41.75% | 36.86% | 58.29% | 64.01% | 0.17s |
| **Random Forest** | 32.21% | 29.74% | 41.96% | 45.48% | 2.08s |

---

## 💡 Phân Tích Kết Quả

### Tại sao Accuracy ~40-50%?

**Lý do chính:**
1. **Dataset khó:** 603 classes (rất nhiều!) - trong ML thông thường 2-50 classes
2. **Nhiều bệnh có triệu chứng tương tự:** Ví dụ: Cúm vs Viêm họng vs Sốt virus
3. **Mỗi bệnh có ít mẫu:** 12,066 mẫu / 603 bệnh = ~20 mẫu/bệnh (ít!)
4. **TF-IDF limitations:** Không nắm bắt được ngữ nghĩa sâu như BERT

### Nhưng hệ thống vẫn hữu ích vì:

✅ **Top-3 Accuracy 62.77%** = Bệnh đúng có trong top 3 dự đoán  
✅ **Top-5 Accuracy 67.70%** = Bệnh đúng có trong top 5 dự đoán  
✅ Đủ để "gợi ý" các khả năng, không phải chẩn đoán chính thức  
✅ Rất nhanh (inference < 10ms)  
✅ Nhẹ (model < 50MB)

---

## 🎯 Khuyến Nghị Sử Dụng

### 1. Luôn hiển thị Top-K predictions (k=3 hoặc 5)
```python
result = assistant.diagnose(symptoms, top_k=5)
# Hiển thị 5 khả năng, không chỉ 1
```

### 2. Sử dụng SVM (model tốt nhất)
```bash
python inference.py --model-type svm --model-dir ./saved_models/svm
```

### 3. Giảm confidence threshold
```bash
# Mặc định: 0.15 (15%)
# Nên giảm xuống: 0.10 (10%) vì dataset khó
python inference.py --threshold 0.10
```

### 4. Yêu cầu user mô tả chi tiết
- Càng nhiều triệu chứng = càng chính xác
- Tốt: "Đau đầu, sốt cao 39 độ, mệt mỏi, buồn nôn, đau cơ"
- Không tốt: "Đau đầu"

---

## 🔄 Cải Thiện Trong Tương Lai

### Cách tăng Accuracy:

#### 1. Data Augmentation
- Tạo thêm mẫu bằng paraphrasing
- Synonym replacement
- Back-translation

#### 2. Feature Engineering
- Thêm character n-grams (2-4 chars)
- Tăng max_features lên 10,000-20,000
- Thử n-gram range (1, 3) thay vì (1, 2)

#### 3. Model Improvements
- **Ensemble:** Voting của nhiều models
- **Stacking:** Stack các models
- **Hyperparameter tuning:** GridSearchCV

#### 4. Advanced Models
- **BERT/PhoBERT:** ~92-95% accuracy (đã có trong project gốc!)
- **BiLSTM + Attention**
- **CNN for text**

---

## 📊 Benchmark với BERT

| Metric | ML Simple (SVM) | BERT (Project gốc) |
|--------|-----------------|-------------------|
| Accuracy | ~49% | ~92-95% |
| Top-3 Acc | ~63% | ~98%+ |
| Training | 5s | 2-4 giờ |
| Inference | <10ms | ~50-100ms (GPU) |
| Model size | <50MB | ~500MB |
| Hardware | CPU OK | Cần GPU |

**Kết luận:** 
- ML Simple: Tốt cho demo, học tập, tài nguyên hạn chế
- BERT: Cần cho production, accuracy cao

---

## ✅ Checklist Sử Dụng

Khi sử dụng hệ thống này:

- ✅ Luôn hiển thị Top-3 hoặc Top-5 predictions
- ✅ Không tuyên bố "chẩn đoán chính xác", chỉ "gợi ý"
- ✅ Hiển thị rõ disclaimer
- ✅ Khuyến khích user gặp bác sĩ
- ✅ Yêu cầu mô tả triệu chứng chi tiết
- ✅ Sử dụng model SVM (tốt nhất)
- ✅ Giảm threshold xuống 0.10

---

## 🎓 Kết Luận

**Hệ thống đã hoàn thành và sẵn sàng sử dụng!**

Mặc dù accuracy ~49% có vẻ thấp, nhưng:
- ✅ Đây là bài toán RẤT KHÓ (603 classes)
- ✅ Top-3/Top-5 accuracy khá tốt (~63-68%)
- ✅ Phù hợp cho mục đích học tập và demo
- ✅ Đủ để "gợi ý" các khả năng bệnh
- ✅ Rất nhanh và nhẹ

**Nếu cần accuracy cao hơn:**
- → Sử dụng hệ thống BERT có sẵn trong project gốc
- → Hoặc implement các cải tiến đề xuất ở trên

---

**Hệ thống sẵn sàng! Happy coding! 🚀**


