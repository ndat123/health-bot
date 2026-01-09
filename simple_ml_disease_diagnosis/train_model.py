"""
Module training model ML cho chẩn đoán bệnh
Hỗ trợ các model:
- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes
- TF-IDF + Random Forest
- TF-IDF + SVM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import pickle
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DiseaseDataLoader


class DiseaseClassifier:
    """Class training và đánh giá model phân loại bệnh"""
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Khởi tạo classifier
        
        Args:
            model_type: Loại model ('logistic_regression', 'naive_bayes', 
                       'random_forest', 'svm')
        """
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.disease_mapping = None
        self.reverse_mapping = None
        self.training_history = {}
        
        # Khởi tạo model theo loại
        self.model = self._create_model(model_type)
    
    def _create_model(self, model_type: str):
        """
        Tạo model theo loại
        
        Args:
            model_type: Loại model
            
        Returns:
            Model đã khởi tạo
        """
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=30,
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'svm': LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
                verbose=0
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Model type không hợp lệ. Chọn: {list(models.keys())}")
        
        return models[model_type]
    
    def prepare_features(self, texts: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Chuyển văn bản thành vector TF-IDF
        
        Args:
            texts: Series chứa văn bản
            fit: Có fit vectorizer không (chỉ True khi training)
            
        Returns:
            Ma trận TF-IDF features
        """
        if fit:
            print("🔧 Đang tạo TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,  # Giới hạn số features
                ngram_range=(1, 2),  # Unigrams và bigrams
                min_df=2,  # Bỏ qua terms xuất hiện < 2 documents
                max_df=0.8,  # Bỏ qua terms xuất hiện > 80% documents
                sublinear_tf=True  # Scale TF sublinearly
            )
            features = self.vectorizer.fit_transform(texts)
            print(f"✓ Đã tạo {features.shape[1]} TF-IDF features")
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer chưa được fit!")
            features = self.vectorizer.transform(texts)
        
        return features
    
    def train(self, X_train: pd.Series, y_train: pd.Series, 
              disease_mapping: dict, reverse_mapping: dict) -> Dict[str, Any]:
        """
        Training model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            disease_mapping: Mapping disease name -> ID
            reverse_mapping: Mapping ID -> disease name
            
        Returns:
            Dictionary chứa kết quả training
        """
        print(f"\n🚀 BẮT ĐẦU TRAINING - {self.model_type.upper()}")
        print("="*70)
        
        self.disease_mapping = disease_mapping
        self.reverse_mapping = reverse_mapping
        
        # Tạo features
        X_train_features = self.prepare_features(X_train, fit=True)
        
        # Training
        print(f"📚 Đang training {self.model_type}...")
        start_time = datetime.now()
        
        self.model.fit(X_train_features, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Hoàn thành training trong {training_time:.2f} giây")
        
        # Lưu lịch sử
        self.training_history = {
            'model_type': self.model_type,
            'training_time': training_time,
            'num_samples': len(X_train),
            'num_classes': len(disease_mapping),
            'num_features': X_train_features.shape[1],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return self.training_history
    
    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Đánh giá model trên test set
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary chứa metrics
        """
        print(f"\n📊 ĐÁNH GIÁ MODEL")
        print("="*70)
        
        # Transform features
        X_test_features = self.prepare_features(X_test, fit=False)
        
        # Dự đoán
        y_pred = self.model.predict(X_test_features)
        
        # Tính các metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Top-K accuracy (nếu model hỗ trợ predict_proba)
        top_k_acc = {}
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test_features)
            for k in [1, 3, 5]:
                top_k_acc[f'top_{k}_accuracy'] = self._calculate_top_k_accuracy(
                    y_test, y_proba, k
                )
        elif hasattr(self.model, 'decision_function'):
            y_scores = self.model.decision_function(X_test_features)
            for k in [1, 3, 5]:
                top_k_acc[f'top_{k}_accuracy'] = self._calculate_top_k_accuracy(
                    y_test, y_scores, k
                )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            **top_k_acc
        }
        
        # In kết quả
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        
        if top_k_acc:
            print("\nTop-K Accuracy:")
            for k, acc in top_k_acc.items():
                print(f"  {k}: {acc*100:.2f}%")
        
        return metrics
    
    def _calculate_top_k_accuracy(self, y_true: np.ndarray, 
                                   y_scores: np.ndarray, k: int) -> float:
        """
        Tính Top-K accuracy
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            k: K value
            
        Returns:
            Top-K accuracy
        """
        # Lấy top K predictions
        top_k_preds = np.argsort(y_scores, axis=1)[:, -k:]
        
        # Kiểm tra label thật có trong top K không
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def predict(self, text: str, top_k: int = 3) -> list:
        """
        Dự đoán bệnh từ triệu chứng
        
        Args:
            text: Mô tả triệu chứng
            top_k: Số lượng dự đoán hàng đầu
            
        Returns:
            List các tuple (disease_name, confidence)
        """
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model chưa được training!")
        
        # Transform text
        text_features = self.vectorizer.transform([text])
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_features)[0]
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(text_features)[0]
            # Convert scores to probabilities using softmax
            exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
            probabilities = exp_scores / np.sum(exp_scores)
        else:
            # Fallback: chỉ trả về prediction
            pred_label = self.model.predict(text_features)[0]
            return [(self.reverse_mapping[pred_label], 1.0)]
        
        # Lấy top K
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            disease_name = self.reverse_mapping[idx]
            confidence = probabilities[idx]
            results.append((disease_name, confidence))
        
        return results
    
    def save_model(self, output_dir: str = './saved_models'):
        """
        Lưu model và các thông tin liên quan
        
        Args:
            output_dir: Thư mục lưu model
        """
        print(f"\n💾 Đang lưu model...")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu model
        model_path = os.path.join(output_dir, f'{self.model_type}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Đã lưu model: {model_path}")
        
        # Lưu vectorizer
        vectorizer_path = os.path.join(output_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ Đã lưu vectorizer: {vectorizer_path}")
        
        # Lưu disease mapping
        mapping_path = os.path.join(output_dir, 'disease_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'disease_to_id': self.disease_mapping,
                'id_to_disease': {str(k): v for k, v in self.reverse_mapping.items()},
                'num_classes': len(self.disease_mapping)
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã lưu disease mapping: {mapping_path}")
        
        # Lưu training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã lưu training history: {history_path}")
        
        print(f"\n✅ Model đã được lưu tại: {output_dir}")
    
    @classmethod
    def load_model(cls, output_dir: str = './saved_models', 
                   model_type: str = 'logistic_regression'):
        """
        Load model đã lưu
        
        Args:
            output_dir: Thư mục chứa model
            model_type: Loại model
            
        Returns:
            Instance của DiseaseClassifier đã load model
        """
        print(f"📂 Đang load model từ {output_dir}...")
        
        # Tạo instance
        classifier = cls(model_type=model_type)
        
        # Load model
        model_path = os.path.join(output_dir, f'{model_type}_model.pkl')
        with open(model_path, 'rb') as f:
            classifier.model = pickle.load(f)
        print(f"✓ Đã load model")
        
        # Load vectorizer
        vectorizer_path = os.path.join(output_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            classifier.vectorizer = pickle.load(f)
        print(f"✓ Đã load vectorizer")
        
        # Load mapping
        mapping_path = os.path.join(output_dir, 'disease_mapping.json')
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            classifier.disease_mapping = mapping_data['disease_to_id']
            classifier.reverse_mapping = {int(k): v for k, v in mapping_data['id_to_disease'].items()}
        print(f"✓ Đã load mapping")
        
        print(f"✅ Model đã sẵn sàng!")
        
        return classifier


def train_all_models(data_file: str, output_base_dir: str = './saved_models',
                     test_size: float = 0.2, remove_accents: bool = False):
    """
    Training tất cả các models và so sánh
    
    Args:
        data_file: Đường dẫn file CSV
        output_base_dir: Thư mục lưu models
        test_size: Tỷ lệ test set
        remove_accents: Có loại bỏ dấu không
    """
    print("🌟 TRAINING TẤT CẢ CÁC MODELS")
    print("="*70)
    
    # Load và prepare data
    loader = DiseaseDataLoader(data_file, remove_accents=remove_accents)
    df = loader.prepare_data(remove_stopwords=False)
    
    # Split data
    print(f"\n✂️  Chia dữ liệu: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    X = df['Question_Processed']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {len(X_train)} mẫu")
    print(f"✓ Test set: {len(X_test)} mẫu")
    
    # Training các models
    model_types = ['logistic_regression', 'naive_bayes', 'random_forest', 'svm']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'='*70}")
        
        # Tạo output directory cho model này
        output_dir = os.path.join(output_base_dir, model_type)
        
        # Training
        classifier = DiseaseClassifier(model_type=model_type)
        train_history = classifier.train(
            X_train, y_train, 
            loader.disease_mapping, 
            loader.reverse_mapping
        )
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        
        # Save
        classifier.save_model(output_dir)
        
        # Lưu kết quả
        results[model_type] = {
            'training': train_history,
            'evaluation': metrics
        }
    
    # So sánh các models
    print(f"\n{'='*70}")
    print("📊 SO SÁNH CÁC MODELS")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':>12} {'F1-Score':>12} {'Time (s)':>12}")
    print("-"*70)
    
    for model_type, result in results.items():
        acc = result['evaluation']['accuracy'] * 100
        f1 = result['evaluation']['f1_score'] * 100
        time = result['training']['training_time']
        print(f"{model_type:<25} {acc:>11.2f}% {f1:>11.2f}% {time:>11.2f}s")
    
    # Tìm model tốt nhất
    best_model = max(results.items(), key=lambda x: x[1]['evaluation']['accuracy'])
    print(f"\n🏆 MODEL TỐT NHẤT: {best_model[0].upper()}")
    print(f"   Accuracy: {best_model[1]['evaluation']['accuracy']*100:.2f}%")
    print(f"   F1-Score: {best_model[1]['evaluation']['f1_score']*100:.2f}%")
    
    # Lưu tổng hợp kết quả
    summary_path = os.path.join(output_base_dir, 'comparison_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Kết quả so sánh đã lưu tại: {summary_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Kiểm tra file data có tồn tại không
    data_file = "../ViMedical_Disease.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Không tìm thấy file dữ liệu: {data_file}")
        print("Vui lòng đặt file ViMedical_Disease.csv trong thư mục gốc!")
        sys.exit(1)
    
    # Training tất cả models
    results = train_all_models(
        data_file=data_file,
        output_base_dir='./saved_models',
        test_size=0.2,
        remove_accents=False
    )
    
    print("\n✅ HOÀN THÀNH!")

