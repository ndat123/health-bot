# 🏥 Medical Disease Diagnosis System - Simple ML

A complete disease diagnosis system based on symptoms using traditional Machine Learning (TF-IDF + Classifiers).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (if not trained yet)
python train_model.py

# 3. Use the system
python inference.py
```

**Interactive mode:**
```
🧑 Your symptoms: Headache, high fever, fatigue, nausea

🏥 PREDICTION RESULTS
💡 You may have:
1. Disease A (0.37%)
2. Disease B (0.30%)
...

⚠️ WARNING: This is NOT a medical diagnosis!
   Please consult a doctor for accurate diagnosis.
```

---

## 📋 Features

✅ **Vietnamese text preprocessing**: Normalization, cleaning  
✅ **4 ML models**: Logistic Regression, Naive Bayes, Random Forest, SVM  
✅ **TF-IDF vectorization**: Convert symptoms to features  
✅ **Top-K predictions**: Multiple disease possibilities  
✅ **Confidence scores**: Probability for each prediction  
✅ **Interactive chat**: Talk to AI for diagnosis  
✅ **Batch processing**: Process multiple cases  
✅ **Always warns**: NOT a medical diagnosis, see a doctor  

---

## 📊 Performance

Dataset: **ViMedical_Disease** (603 diseases, ~12K samples)

| Model | Accuracy | Top-3 Acc | Top-5 Acc | Time |
|-------|----------|-----------|-----------|------|
| **SVM** ⭐ | **49.34%** | **62.77%** | **67.70%** | 4.5s |
| Logistic Regression | 42.83% | 59.12% | 64.39% | 15.9s |
| Naive Bayes | 41.75% | 58.29% | 64.01% | 0.2s |
| Random Forest | 32.21% | 41.96% | 45.48% | 2.1s |

**Note:** With 603 classes, ~50% accuracy is reasonable. Top-K accuracy is more important!

---

## 📖 Documentation

- 🇻🇳 [README_VI.md](README_VI.md) - Full guide in Vietnamese
- ⚡ [QUICKSTART.md](QUICKSTART.md) - 3-step quick start
- 📊 [KET_QUA_TRAINING.md](KET_QUA_TRAINING.md) - Training results
- 🔧 [GIAI_THICH_FIX_CONFIDENCE.md](GIAI_THICH_FIX_CONFIDENCE.md) - Confidence fix explanation

---

## 💻 Usage Examples

### 1. Interactive Chat
```bash
python inference.py
```

### 2. Single Prediction
```bash
python inference.py --symptoms "Headache, high fever, fatigue"
```

### 3. Batch Processing
```bash
python inference.py --batch-file symptoms.txt --output results.json
```

### 4. Use Different Model
```bash
python inference.py --model-type svm --model-dir ./saved_models/svm
```

### 5. Python Code
```python
from inference import MedicalDiagnosisAssistant

assistant = MedicalDiagnosisAssistant(
    model_dir='./saved_models/svm',
    model_type='svm'
)

result = assistant.diagnose("Headache, fever, fatigue", top_k=3)
print(f"Disease: {result['top_prediction']['disease']}")
print(f"Confidence: {result['top_prediction']['confidence_percent']}")
```

---

## ⚠️ Important Notice

**THIS IS NOT A MEDICAL DIAGNOSIS!**

- ✅ This is an AI-based prediction tool for educational purposes
- ✅ Always consult a medical professional for accurate diagnosis
- ✅ Do not self-medicate based on these results
- ✅ If symptoms are severe, seek medical attention immediately

---

## 🔧 Technical Details

### Data Preprocessing
- Vietnamese text normalization
- Missing value handling
- Optional accent removal
- Stopword removal

### Vectorization
- **TF-IDF**: Max features 5000, n-grams (1,2)
- Min DF: 2, Max DF: 0.8
- Sublinear TF scaling

### Models
1. **Logistic Regression**: C=1.0, max_iter=1000
2. **Naive Bayes**: Multinomial, alpha=1.0
3. **Random Forest**: 100 trees, max_depth=30
4. **SVM**: Linear kernel, C=1.0

---

## 📁 Project Structure

```
simple_ml_disease_diagnosis/
├── README.md                 # This file
├── README_VI.md             # Vietnamese guide
├── requirements.txt         # Dependencies
│
├── data_preprocessing.py    # Data preprocessing
├── train_model.py          # Model training
├── inference.py            # Inference (MAIN)
├── demo.py                 # Demo script
├── example_usage.py        # Code examples
│
└── saved_models/           # Trained models
    ├── logistic_regression/
    ├── naive_bayes/
    ├── random_forest/
    └── svm/
```

---

## 🆚 Comparison with BERT

| | Simple ML | BERT (In project) |
|---|---|---|
| Accuracy | ~49% | ~92-95% |
| Training | 5s | 2-4 hours |
| Inference | <10ms | ~100ms |
| Model size | <50MB | ~500MB |
| Hardware | CPU OK | GPU needed |
| Best for | Learning, demo | Production |

---

## 📝 License

- **Code**: Free for educational use
- **Dataset**: [ViMedical_Disease](https://github.com/PB3002/ViMedical_Disease) - CC BY-NC-SA 4.0

---

## 🙏 Credits

- **Dataset**: [ViMedical_Disease by PB3002](https://github.com/PB3002/ViMedical_Disease)
- **Libraries**: Scikit-learn, Pandas, NumPy

---

## 🚀 Get Started

```bash
python inference.py
```

**Happy coding! 🎉**

*Remember: This is just a support tool, ALWAYS consult a doctor!*


