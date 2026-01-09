"""
Script để sử dụng chatbot y tế đã được train
Cho phép người dùng nhập câu hỏi về triệu chứng và nhận dự đoán bệnh
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

class MedicalChatbot:
    def __init__(self, model_path="./chatbot_model", max_length=256):
        """Khởi tạo chatbot"""
        self.model_path = model_path
        self.max_length = max_length
        
        print("Đang load model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Load disease mapping
        with open(f"{model_path}/disease_mapping.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)
            self.id_to_disease = {int(k): v for k, v in mapping["id_to_disease"].items()}
        
        print(f"Đã load model với {len(self.id_to_disease)} loại bệnh")
    
    def predict(self, question, top_k=3):
        """
        Dự đoán bệnh từ câu hỏi về triệu chứng
        
        Args:
            question: Câu hỏi về triệu chứng
            top_k: Số lượng bệnh có khả năng cao nhất để trả về
        
        Returns:
            List các tuple (tên_bệnh, xác_suất)
        """
        # Tokenize
        inputs = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        
        # Lấy top_k bệnh có xác suất cao nhất
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.id_to_disease)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            disease_name = self.id_to_disease[idx.item()]
            results.append((disease_name, prob.item()))
        
        return results
    
    def chat(self):
        """Chế độ chat tương tác"""
        print("\n" + "=" * 60)
        print("CHATBOT Y TẾ TIẾNG VIỆT")
        print("=" * 60)
        print("Nhập câu hỏi về triệu chứng của bạn.")
        print("Gõ 'quit' hoặc 'exit' để thoát.")
        print("=" * 60 + "\n")
        
        while True:
            question = input("Bạn: ").strip()
            
            if question.lower() in ['quit', 'exit', 'thoát']:
                print("\nCảm ơn bạn đã sử dụng chatbot!")
                break
            
            if not question:
                continue
            
            # Dự đoán
            results = self.predict(question, top_k=3)
            
            # Hiển thị kết quả
            print("\nChatbot:")
            print("Dựa trên các triệu chứng bạn mô tả, bạn có thể đang mắc:")
            for i, (disease, prob) in enumerate(results, 1):
                print(f"  {i}. {disease} (Xác suất: {prob*100:.2f}%)")
            
            print("\n⚠️ LƯU Ý: Đây chỉ là dự đoán sơ bộ dựa trên triệu chứng.")
            print("Bạn nên tham khảo ý kiến bác sĩ để được chẩn đoán chính xác.\n")
            print("-" * 60 + "\n")

def main():
    """Hàm main"""
    model_path = "./chatbot_model"
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại {model_path}")
        print("Vui lòng chạy train_chatbot.py trước để train model.")
        return
    
    # Khởi tạo chatbot
    chatbot = MedicalChatbot(model_path)
    
    # Chạy chế độ chat
    chatbot.chat()

if __name__ == "__main__":
    main()

