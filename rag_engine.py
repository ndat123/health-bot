"""
RAG Engine - Sử dụng Vector Embeddings cho tìm kiếm ngữ nghĩa
Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cấu hình
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_FILE = 'disease_embeddings.pkl'
DATA_FILE = 'ViMedical_Disease.csv'

class MedicalRAG:
    def __init__(self):
        print("⏳ Initializing RAG Engine...")
        self.model = None
        self.df = None
        self.embeddings = None
        self.load_data()
        self.load_model()
        
    def load_data(self):
        """Load data từ CSV"""
        if os.path.exists(DATA_FILE):
            self.df = pd.read_csv(DATA_FILE)
            # Ensure columns exist
            if 'Question' not in self.df.columns or 'Disease' not in self.df.columns:
                raise ValueError("CSV must have 'Question' and 'Disease' columns")
            print(f"✓ Loaded CSV: {len(self.df)} records")
        else:
            raise FileNotFoundError(f"Could not find {DATA_FILE}")

    def load_model(self):
        """Load embedding model và embeddings đã cache"""
        print(f"⏳ Loading Embedding Model ({MODEL_NAME})...")
        # Force CPU to avoid CUDA OOM issues during web serving
        self.model = SentenceTransformer(MODEL_NAME, device='cpu')
        
        # Check cache
        if os.path.exists(EMBEDDING_FILE):
            print("✓ Loading cached embeddings...")
            with open(EMBEDDING_FILE, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                # Verify length
                if len(self.embeddings) != len(self.df):
                    print("⚠️ Cache outdated. Re-generating embeddings...")
                    self.generate_embeddings()
        else:
            print("Status: Generating new embeddings (This happens only once)...")
            self.generate_embeddings()
            
    def generate_embeddings(self):
        """Tạo vector embeddings cho toàn bộ database"""
        texts = self.df['Question'].tolist()
        # Encode in batches to show progress
        print(f"⏳ Encoding {len(texts)} symptoms...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Save cache
        with open(EMBEDDING_FILE, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings}, f)
        print("✓ Embeddings saved to cache")

    def search(self, query, top_k=10):
        """Tìm kiếm semantic similarity"""
        # Encode query
        query_vector = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        scores = []
        
        for idx in top_indices:
            score = similarities[idx]
            row = self.df.iloc[idx]
            results.append({
                'disease': row['Disease'],
                'symptom': row['Question'],
                'score': float(score)
            })
            
        return results

    def predict_disease(self, query, top_k=5):
        """Dự đoán bệnh dựa trên RAG vote"""
        search_results = self.search(query, top_k=20) # Lấy top 20 symptoms khớp nhất
        
        disease_scores = {}
        disease_details = {}
        
        for item in search_results:
            d = item['disease']
            s = item['score']
            
            if d not in disease_scores:
                disease_scores[d] = 0
                disease_details[d] = []
            
            # Weighted vote: Score càng cao càng có giá trị
            disease_scores[d] += s * 100 
            disease_details[d].append(item['symptom'])
            
        # Sort diseases by total score
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        predictions = []
        for disease, score in sorted_diseases:
            predictions.append({
                'disease': disease,
                'confidence': score, # Score này là tổng hợp, ko phải % direct
                'matched_symptoms': disease_details[disease][:3]
            })
            
        return predictions

# Singleton instance
rag_engine = None

def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        rag_engine = MedicalRAG()
    return rag_engine

if __name__ == "__main__":
    # Test script
    print("\n🧪 Testing RAG Engine...")
    engine = get_rag_engine()
    
    test_query = "Tôi bị đau đầu như búa bổ và buồn nôn"
    print(f"\nQuery: {test_query}")
    
    preds = engine.predict_disease(test_query)
    
    print("\n🔍 Top Predictions:")
    for p in preds:
        print(f"- {p['disease']} (Score: {p['confidence']:.2f})")
        print(f"  Matches: {p['matched_symptoms']}")
