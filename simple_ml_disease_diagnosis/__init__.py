"""
Simple ML Disease Diagnosis System
Hệ thống chẩn đoán bệnh đơn giản dựa trên Machine Learning

Modules:
- data_preprocessing: Tiền xử lý dữ liệu tiếng Việt
- train_model: Training các models ML (TF-IDF + Logistic Regression, Naive Bayes, etc.)
- inference: Dự đoán bệnh từ triệu chứng
- demo: Script demo và ví dụ sử dụng
"""

__version__ = "1.0.0"
__author__ = "Medical AI Assistant"

from .data_preprocessing import VietnameseTextPreprocessor, DiseaseDataLoader
from .train_model import DiseaseClassifier
from .inference import MedicalDiagnosisAssistant

__all__ = [
    'VietnameseTextPreprocessor',
    'DiseaseDataLoader',
    'DiseaseClassifier',
    'MedicalDiagnosisAssistant'
]


