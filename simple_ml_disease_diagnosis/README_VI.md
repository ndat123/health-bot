# 🏥 Hệ Thống Chẩn Đoán Bệnh Dựa Trên Machine Learning

## 📋 Mô Tả

Hệ thống chẩn đoán bệnh tự động dựa trên triệu chứng, sử dụng Machine Learning truyền thống với TF-IDF và các thuật toán phân loại.

### ✨ Tính Năng

- ✅ **Tiền xử lý văn bản tiếng Việt**: Chuẩn hóa, làm sạch text
- ✅ **Nhiều models ML**: Logistic Regression, Naive Bayes, Random Forest, SVM
- ✅ **TF-IDF Vectorization**: Chuyển triệu chứng thành features
- ✅ **Dự đoán Top-K**: Trả về nhiều khả năng bệnh
- ✅ **Confidence Score**: Đánh giá độ tin cậy dự đoán
- ✅ **Chế độ Interactive**: Chat với AI để chẩn đoán
- ✅ **Batch Processing**: Xử lý nhiều ca cùng lúc

### ⚠️ LƯU Ý QUAN TRỌNG

**Đây KHÔNG phải là chẩn đoán y tế chính thức!**

- Chỉ là công cụ hỗ trợ dự đoán sơ bộ
- Luôn tham khảo ý kiến bác sĩ chuyên khoa
- Không tự ý điều trị dựa trên kết quả này

---

## 🚀 Cài Đặt

### 1. Yêu Cầu Hệ Thống

- Python 3.8+
- pip

### 2. Cài Đặt Dependencies

```bash
cd simple_ml_disease_diagnosis
pip install -r requirements.txt
```

### 3. Chuẩn Bị Dữ Liệu

Đảm bảo file `ViMedical_Disease.csv` nằm trong thư mục gốc dự án:

```
ViMedical_Disease/
├── simple_ml_disease_diagnosis/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── inference.py
│   ├── demo.py
│   └── requirements.txt
└── ViMedical_Disease.csv  ← File dữ liệu ở đây
```

---

## 📚 Hướng Dẫn Sử Dụng

### Bước 1: Training Models

Training tất cả các models (khuyến nghị):

```bash
cd simple_ml_disease_diagnosis
python train_model.py
```

Quá trình này sẽ:
- ✅ Load và tiền xử lý dữ liệu từ CSV
- ✅ Training 4 models: Logistic Regression, Naive Bayes, Random Forest, SVM
- ✅ Đánh giá và so sánh các models
- ✅ Lưu models vào `./saved_models/`

**Thời gian:** Khoảng 2-5 phút (tùy cấu hình máy)

**Output:**
```
saved_models/
├── logistic_regression/
│   ├── logistic_regression_model.pkl
│   ├── vectorizer.pkl
│   ├── disease_mapping.json
│   └── training_history.json
├── naive_bayes/
├── random_forest/
├── svm/
└── comparison_results.json
```

### Bước 2: Sử Dụng Hệ Thống

#### 🎯 Chế độ 1: Interactive Chat (Khuyến nghị)

Chạy chế độ chat tương tác:

```bash
python inference.py
```

Sau đó nhập triệu chứng của bạn:

```
🧑 Triệu chứng của bạn: Đau đầu, sốt cao, mệt mỏi, buồn nôn

🏥 KẾT QUẢ DỰ ĐOÁN
======================================================================
💡 Bạn có thể đang mắc:

1. Sốt xuất huyết
   🟢 Độ tin cậy: RẤT CAO (82.45%)
   [████████████████████████████████████░░░░]

2. Cúm
   🟡 Độ tin cậy: CAO (65.20%)
   [██████████████████████████░░░░░░░░░░░░░░]
...
```

**Commands trong chat:**
- `quit` / `exit` / `thoát`: Thoát chương trình
- `history`: Xem lịch sử chẩn đoán
- `clear`: Xóa lịch sử

#### 🎯 Chế độ 2: Dự Đoán Đơn

Dự đoán cho 1 triệu chứng cụ thể:

```bash
python inference.py --symptoms "Đau đầu, sốt cao, mệt mỏi"
```

#### 🎯 Chế độ 3: Batch Processing

Dự đoán cho nhiều ca từ file:

1. Tạo file `symptoms.txt`:
```
Đau đầu, sốt cao, mệt mỏi
Ho, sổ mũi, đau họng
Đau bụng, tiêu chảy
...
```

2. Chạy batch prediction:
```bash
python inference.py --batch-file symptoms.txt --output results.json
```

#### 🎯 Chế độ 4: Sử dụng Model Khác

Mặc định sử dụng Logistic Regression. Để dùng model khác:

```bash
# Sử dụng Random Forest
python inference.py --model-type random_forest --model-dir ./saved_models/random_forest

# Sử dụng Naive Bayes
python inference.py --model-type naive_bayes --model-dir ./saved_models/naive_bayes

# Sử dụng SVM
python inference.py --model-type svm --model-dir ./saved_models/svm
```

#### 🎯 Chế độ 5: Điều Chỉnh Confidence Threshold

```bash
# Ngưỡng thấp hơn (chấp nhận dự đoán ít tin cậy hơn)
python inference.py --threshold 0.10

# Ngưỡng cao hơn (chỉ chấp nhận dự đoán rất tin cậy)
python inference.py --threshold 0.30
```

### Bước 3: Chạy Demo

Xem tất cả các ví dụ sử dụng:

```bash
python demo.py
```

Demo bao gồm:
1. ✅ Sử dụng cơ bản
2. ✅ Output chi tiết
3. ✅ Dự đoán hàng loạt
4. ✅ So sánh các models
5. ✅ Chế độ interactive

---

## 🔧 API Usage (Sử dụng trong code)

### Ví dụ 1: Dự đoán đơn giản

```python
from inference import MedicalDiagnosisAssistant

# Khởi tạo
assistant = MedicalDiagnosisAssistant(
    model_dir='./saved_models/logistic_regression',
    model_type='logistic_regression'
)

# Dự đoán
result = assistant.diagnose("Đau đầu, sốt cao, mệt mỏi", top_k=3)

# Kết quả
if result['success']:
    print(f"Bệnh: {result['top_prediction']['disease']}")
    print(f"Độ tin cậy: {result['top_prediction']['confidence_percent']}")
```

### Ví dụ 2: Training custom model

```python
from train_model import DiseaseClassifier
from data_preprocessing import DiseaseDataLoader
from sklearn.model_selection import train_test_split

# Load data
loader = DiseaseDataLoader("../ViMedical_Disease.csv")
df = loader.prepare_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Question_Processed'], 
    df['label'], 
    test_size=0.2,
    random_state=42
)

# Training
classifier = DiseaseClassifier(model_type='logistic_regression')
classifier.train(X_train, y_train, loader.disease_mapping, loader.reverse_mapping)

# Evaluate
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']*100:.2f}%")

# Save
classifier.save_model('./my_custom_model')
```

### Ví dụ 3: Load model đã lưu

```python
from train_model import DiseaseClassifier

# Load
classifier = DiseaseClassifier.load_model(
    output_dir='./saved_models/logistic_regression',
    model_type='logistic_regression'
)

# Predict
predictions = classifier.predict("Đau đầu, sốt cao", top_k=5)

for disease, confidence in predictions:
    print(f"{disease}: {confidence*100:.2f}%")
```

---

## 📊 Hiệu Suất Models

Kết quả trên dataset ViMedical_Disease (603 loại bệnh, ~12K mẫu):

| Model | Accuracy | F1-Score | Training Time | Inference Speed |
|-------|----------|----------|---------------|-----------------|
| **Logistic Regression** | **~85-90%** | **~84-88%** | 10-20s | Rất nhanh |
| Naive Bayes | ~75-80% | ~73-78% | 5-10s | Rất nhanh |
| Random Forest | ~80-85% | ~78-83% | 30-60s | Nhanh |
| SVM | ~85-90% | ~83-87% | 20-40s | Nhanh |

**Khuyến nghị:** Sử dụng **Logistic Regression** để có sự cân bằng tốt nhất giữa độ chính xác và tốc độ.

---

## 📁 Cấu Trúc Thư Mục

```
simple_ml_disease_diagnosis/
├── README_VI.md              # Hướng dẫn tiếng Việt (file này)
├── requirements.txt          # Dependencies
├── __init__.py              # Package init
│
├── data_preprocessing.py    # Module tiền xử lý dữ liệu
├── train_model.py           # Module training models
├── inference.py             # Module dự đoán
├── demo.py                  # Script demo
│
└── saved_models/            # Models đã train (tự động tạo)
    ├── logistic_regression/
    ├── naive_bayes/
    ├── random_forest/
    ├── svm/
    └── comparison_results.json
```

---

## 🐛 Xử Lý Lỗi Thường Gặp

### Lỗi 1: "Không tìm thấy file ViMedical_Disease.csv"

**Nguyên nhân:** File dữ liệu không đúng vị trí

**Giải pháp:**
```bash
# Đảm bảo cấu trúc thư mục:
ViMedical_Disease/
├── simple_ml_disease_diagnosis/
└── ViMedical_Disease.csv  ← Phải ở đây
```

### Lỗi 2: "ModuleNotFoundError"

**Nguyên nhân:** Thiếu dependencies

**Giải pháp:**
```bash
pip install -r requirements.txt
```

### Lỗi 3: "Không tìm thấy model"

**Nguyên nhân:** Chưa training model

**Giải pháp:**
```bash
python train_model.py
```

### Lỗi 4: Confidence thấp cho tất cả dự đoán

**Nguyên nhân:** 
- Triệu chứng không rõ ràng
- Model chưa được train tốt

**Giải pháp:**
- Mô tả triệu chứng chi tiết hơn
- Retrain model với nhiều epochs hơn
- Giảm confidence threshold: `--threshold 0.10`

---

## 🔬 Chi Tiết Kỹ Thuật

### Tiền Xử Lý Dữ Liệu

1. **Làm sạch văn bản:**
   - Chuyển về lowercase
   - Loại bỏ ký tự đặc biệt
   - Chuẩn hóa khoảng trắng

2. **Xử lý missing values:**
   - Loại bỏ các dòng có dữ liệu thiếu

3. **Label encoding:**
   - Chuyển tên bệnh thành ID số

### Feature Engineering

**TF-IDF Vectorization:**
- Max features: 5000
- N-gram range: (1, 2) - unigrams và bigrams
- Min document frequency: 2
- Max document frequency: 0.8
- Sublinear TF scaling

### Models

1. **Logistic Regression:**
   - Regularization: L2 (C=1.0)
   - Max iterations: 1000
   - Multi-class: One-vs-Rest

2. **Naive Bayes:**
   - Algorithm: Multinomial NB
   - Alpha (smoothing): 1.0

3. **Random Forest:**
   - Trees: 100
   - Max depth: 30
   - Parallel: n_jobs=-1

4. **SVM:**
   - Kernel: Linear
   - C: 1.0
   - Max iterations: 2000

### Evaluation Metrics

- **Accuracy:** Tỷ lệ dự đoán đúng
- **Precision:** Độ chính xác của dự đoán
- **Recall:** Khả năng tìm ra đúng
- **F1-Score:** Trung bình điều hòa của Precision và Recall
- **Top-K Accuracy:** Label đúng có trong top K dự đoán không

---

## 🎓 Ví Dụ Đầu Vào/Đầu Ra

### Ví dụ 1: Sốt xuất huyết

**Input:**
```
Đau đầu, sốt cao, mệt mỏi, buồn nôn
```

**Output:**
```
Bệnh dự đoán: Sốt xuất huyết
Độ tin cậy: 82%
Khuyến nghị: Đến gặp bác sĩ để kiểm tra
```

### Ví dụ 2: Cúm

**Input:**
```
Ho, sổ mũi, đau họng, sốt nhẹ, mệt mỏi
```

**Output:**
```
Bệnh dự đoán: Cúm
Độ tin cậy: 78%
Khuyến nghị: Đến gặp bác sĩ để kiểm tra
```

### Ví dụ 3: Triệu chứng không rõ

**Input:**
```
Mệt mỏi
```

**Output:**
```
⚠️ CẢNH BÁO: Độ tin cậy thấp!
Vui lòng cung cấp thêm thông tin chi tiết về triệu chứng.
```

---

## 📝 License

Dataset ViMedical_Disease: CC BY-NC-SA 4.0

---

## 👥 Liên Hệ & Đóng Góp

Nếu có thắc mắc hoặc muốn đóng góp, vui lòng:
- Mở Issue trên GitHub
- Gửi Pull Request
- Liên hệ tác giả dataset gốc

---

## 🙏 Lời Cảm Ơn

- Dataset: [ViMedical_Disease](https://github.com/PB3002/ViMedical_Disease) by PB3002
- Scikit-learn team
- Python community

---

**Chúc bạn sử dụng hệ thống hiệu quả! 🏥**

*Nhớ rằng: Đây chỉ là công cụ hỗ trợ, luôn tham khảo ý kiến bác sĩ!*


