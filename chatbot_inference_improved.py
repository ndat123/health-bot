"""
Script cải tiến để sử dụng chatbot y tế đã được train
Bao gồm các cải thiện:
- Confidence threshold
- Giải thích dự đoán (attention visualization)
- Lưu lịch sử hội thoại
- Xử lý câu hỏi không rõ ràng
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from datetime import datetime
import numpy as np

class ImprovedMedicalChatbot:
    def __init__(self, model_path="./chatbot_model_improved", max_length=256, 
                 confidence_threshold=0.15, out_of_domain_threshold=0.10):
        """
        Khởi tạo chatbot với các tính năng cải tiến
        
        Args:
            model_path: Đường dẫn đến model
            max_length: Độ dài tối đa của input
            confidence_threshold: Ngưỡng confidence tối thiểu (0-1)
            out_of_domain_threshold: Ngưỡng để phát hiện câu hỏi không liên quan
        """
        self.model_path = model_path
        self.max_length = max_length
        self.dynamic_max_length = True  # Tự động điều chỉnh
        self.confidence_threshold = confidence_threshold
        self.out_of_domain_threshold = out_of_domain_threshold
        self.conversation_history = []
        
        # Từ khóa y tế để validate
        self.medical_keywords = [
            'triệu chứng', 'bệnh', 'đau', 'mệt', 'sốt', 'ho', 'khó thở',
            'buồn nôn', 'chóng mặt', 'nhức đầu', 'mất ngủ', 'ngứa', 'sưng'
        ]
        
        print("Đang load model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Load disease mapping
        mapping_path = f"{model_path}/disease_mapping.json"
        if not os.path.exists(mapping_path):
            # Fallback to original model path
            mapping_path = "./chatbot_model/disease_mapping.json"
        
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            self.id_to_disease = {int(k): v for k, v in mapping["id_to_disease"].items()}
        
        print(f"✓ Đã load model với {len(self.id_to_disease)} loại bệnh")
        print(f"✓ Confidence threshold: {confidence_threshold:.2%}")
    
    def validate_question(self, question):
        """Validate câu hỏi đầu vào"""
        question = str(question).strip()
        
        if not question or len(question) < 10:
            return False, "Câu hỏi quá ngắn. Vui lòng mô tả chi tiết hơn về triệu chứng."
        
        if len(question) > 2000:
            return False, "Câu hỏi quá dài. Vui lòng rút gọn lại."
        
        # Kiểm tra có chứa từ khóa y tế không
        question_lower = question.lower()
        has_medical_keyword = any(keyword in question_lower for keyword in self.medical_keywords)
        
        if not has_medical_keyword:
            return False, "Câu hỏi không liên quan đến triệu chứng y tế."
        
        return True, None
    
    def is_out_of_domain(self, probabilities):
        """Phát hiện câu hỏi không liên quan đến bệnh"""
        max_prob = float(torch.max(probabilities).item())
        return max_prob < self.out_of_domain_threshold, max_prob
    
    def predict(self, question, top_k=3, return_details=False):
        """
        Dự đoán bệnh từ câu hỏi về triệu chứng với validation và out-of-domain detection
        
        Args:
            question: Câu hỏi về triệu chứng
            top_k: Số lượng bệnh có khả năng cao nhất để trả về
            return_details: Có trả về thông tin chi tiết không
        
        Returns:
            List các tuple (tên_bệnh, xác_suất) hoặc dict nếu return_details=True
        """
        # Validate question
        is_valid, error_msg = self.validate_question(question)
        if not is_valid:
            if return_details:
                return {
                    'error': error_msg,
                    'is_valid': False,
                    'question': question,
                    'is_confident': False
                }
            return [], False
        
        # Tính max_length động nếu cần
        if self.dynamic_max_length:
            temp_tokenized = self.tokenizer(question, truncation=False, padding=False)
            actual_length = len(temp_tokenized['input_ids'])
            effective_max_length = min(actual_length + 20, self.max_length, 512)
        else:
            effective_max_length = self.max_length
        
        # Tokenize
        inputs = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=effective_max_length,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            
            # Lấy attention weights nếu có
            attentions = None
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions
        
        # Kiểm tra out-of-domain trước
        is_ood, max_prob_ood = self.is_out_of_domain(probabilities)
        
        # Lấy top_k bệnh có xác suất cao nhất
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.id_to_disease)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            disease_name = self.id_to_disease[idx.item()]
            results.append((disease_name, prob.item()))
        
        # Kiểm tra confidence
        max_confidence = results[0][1] if results else 0.0
        is_confident = max_confidence >= self.confidence_threshold
        
        if return_details:
            return {
                'predictions': results,
                'is_confident': is_confident,
                'is_out_of_domain': is_ood,
                'max_confidence': max_confidence,
                'confidence_threshold': self.confidence_threshold,
                'out_of_domain_threshold': self.out_of_domain_threshold,
                'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                'input_text': question,
                'is_valid': True
            }
        
        return results, is_confident
    
    def get_confidence_level(self, confidence):
        """Chuyển đổi confidence thành mức độ tin cậy"""
        if confidence >= 0.7:
            return "RẤT CAO", "🟢"
        elif confidence >= 0.4:
            return "CAO", "🟡"
        elif confidence >= 0.2:
            return "TRUNG BÌNH", "🟠"
        else:
            return "THẤP", "🔴"
    
    def format_prediction_output(self, results, is_confident):
        """Format output dự đoán với màu sắc và emoji"""
        output = []
        
        if not is_confident:
            output.append("\n⚠️  CẢNH BÁO: Độ tin cậy thấp!")
            output.append(f"Xác suất cao nhất chỉ {results[0][1]*100:.2f}% (ngưỡng: {self.confidence_threshold*100:.1f}%)")
            output.append("Triệu chứng có thể không rõ ràng hoặc không đủ thông tin.\n")
        
        output.append("Dựa trên các triệu chứng bạn mô tả, bạn có thể đang mắc:\n")
        
        for i, (disease, prob) in enumerate(results, 1):
            level, emoji = self.get_confidence_level(prob)
            bar_length = int(prob * 30)  # Progress bar
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            output.append(f"{i}. {disease}")
            output.append(f"   {emoji} Xác suất: {prob*100:.2f}% - Độ tin cậy: {level}")
            output.append(f"   [{bar}]\n")
        
        return "\n".join(output)
    
    def suggest_more_info(self, results):
        """Đề xuất thông tin cần bổ sung"""
        suggestions = []
        
        # Nếu confidence thấp, đề xuất thêm thông tin
        if results[0][1] < self.confidence_threshold:
            suggestions.append("\n💡 ĐỀ XUẤT: Vui lòng cung cấp thêm thông tin:")
            suggestions.append("   - Triệu chứng cụ thể hơn (vị trí, mức độ, thời gian)")
            suggestions.append("   - Các triệu chứng kèm theo khác")
            suggestions.append("   - Thời gian xuất hiện triệu chứng")
            suggestions.append("   - Yếu tố làm tăng/giảm triệu chứng")
        
        # Nếu top 2 predictions gần nhau, cảnh báo
        if len(results) >= 2:
            diff = results[0][1] - results[1][1]
            if diff < 0.1:  # Chênh lệch < 10%
                suggestions.append(f"\n⚠️  LƯU Ý: Xác suất giữa 2 bệnh hàng đầu rất gần nhau")
                suggestions.append(f"   ({results[0][0]}: {results[0][1]*100:.1f}% vs {results[1][0]}: {results[1][1]*100:.1f}%)")
                suggestions.append("   Cần thêm thông tin để phân biệt chính xác.")
        
        return "\n".join(suggestions) if suggestions else ""
    
    def save_conversation(self, filename="conversation_history.json"):
        """Lưu lịch sử hội thoại"""
        filepath = os.path.join(self.model_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        return filepath
    
    def chat(self, save_history=True):
        """Chế độ chat tương tác cải tiến"""
        print("\n" + "=" * 70)
        print("CHATBOT Y TẾ TIẾNG VIỆT - PHIÊN BẢN CẢI TIẾN")
        print("=" * 70)
        print("Nhập câu hỏi về triệu chứng của bạn.")
        print("Gõ 'quit' hoặc 'exit' để thoát.")
        print("Gõ 'history' để xem lịch sử hội thoại.")
        print("Gõ 'clear' để xóa lịch sử hội thoại.")
        print("=" * 70)
        print(f"\n📊 Thông tin:")
        print(f"   - Confidence threshold: {self.confidence_threshold*100:.1f}%")
        print(f"   - Số loại bệnh: {len(self.id_to_disease)}")
        print(f"   - Model: {self.model_path}")
        print("\n" + "-" * 70 + "\n")
        
        session_start = datetime.now()
        
        while True:
            try:
                question = input("🧑 Bạn: ").strip()
                
                # Xử lý commands
                if question.lower() in ['quit', 'exit', 'thoát', 'q']:
                    if save_history and self.conversation_history:
                        filepath = self.save_conversation()
                        print(f"\n💾 Lịch sử hội thoại đã được lưu tại: {filepath}")
                    print("\n👋 Cảm ơn bạn đã sử dụng chatbot! Hẹn gặp lại!")
                    break
                
                if question.lower() == 'history':
                    self.show_history()
                    continue
                
                if question.lower() == 'clear':
                    self.conversation_history = []
                    print("\n✓ Đã xóa lịch sử hội thoại.\n")
                    continue
                
                if not question:
                    continue
                
                # Dự đoán với details để kiểm tra validation
                timestamp = datetime.now()
                result_details = self.predict(question, top_k=5, return_details=True)
                
                # Kiểm tra nếu có lỗi validation
                if isinstance(result_details, dict) and not result_details.get('is_valid', True):
                    print(f"\n⚠️  Chatbot: {result_details.get('error', 'Câu hỏi không hợp lệ')}\n")
                    print("-" * 70 + "\n")
                    continue
                
                # Lấy results từ details
                if isinstance(result_details, dict):
                    results = result_details['predictions']
                    is_ood = result_details.get('is_out_of_domain', False)
                    is_confident = result_details.get('is_confident', False)
                else:
                    results = result_details
                    is_ood = False
                    is_confident = False
                
                # Lưu vào lịch sử
                self.conversation_history.append({
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'question': question,
                    'predictions': [
                        {'disease': disease, 'probability': float(prob)}
                        for disease, prob in results
                    ],
                    'is_confident': is_confident,
                    'is_out_of_domain': is_ood,
                    'max_confidence': float(results[0][1]) if results else 0.0
                })
                
                # Cảnh báo nếu out-of-domain
                if is_ood:
                    print("\n⚠️  CẢNH BÁO: Câu hỏi có thể không liên quan đến bệnh!")
                    if results:
                        print(f"Độ tin cậy rất thấp ({results[0][1]*100:.2f}%).")
                    print("Vui lòng mô tả các triệu chứng y tế cụ thể.\n")
                
                # Hiển thị kết quả
                if results:
                    print("\n🤖 Chatbot:")
                    print(self.format_prediction_output(results[:3], is_confident))
                    
                    # Đề xuất thêm thông tin nếu cần
                    suggestions = self.suggest_more_info(results)
                    if suggestions:
                        print(suggestions)
                    
                    # Hiển thị thêm 2 bệnh tiếp theo nếu có
                    if len(results) > 3:
                        print("\n📋 Các khả năng khác (xác suất thấp hơn):")
                        for i, (disease, prob) in enumerate(results[3:5], 4):
                            print(f"   {i}. {disease} ({prob*100:.2f}%)")
                    
                    print("\n" + "⚠️" * 35)
                    print("⚠️  LƯU Ý QUAN TRỌNG:")
                    print("   - Đây chỉ là dự đoán sơ bộ dựa trên AI, KHÔNG PHẢI chẩn đoán y tế")
                    print("   - Bạn NÊN tham khảo ý kiến bác sĩ để được chẩn đoán chính xác")
                    print("   - Không tự ý điều trị dựa trên kết quả này")
                    print("⚠️" * 35)
                    print("\n" + "-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Đã nhận Ctrl+C. Đang thoát...")
                if save_history and self.conversation_history:
                    filepath = self.save_conversation()
                    print(f"💾 Lịch sử hội thoại đã được lưu tại: {filepath}")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {str(e)}")
                print("Vui lòng thử lại.\n")
    
    def show_history(self):
        """Hiển thị lịch sử hội thoại"""
        if not self.conversation_history:
            print("\n📭 Chưa có lịch sử hội thoại.\n")
            return
        
        print("\n" + "=" * 70)
        print(f"LỊCH SỬ HỘI THOẠI ({len(self.conversation_history)} câu hỏi)")
        print("=" * 70)
        
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{i}. [{entry['timestamp']}]")
            print(f"   Câu hỏi: {entry['question']}")
            print(f"   Dự đoán hàng đầu: {entry['predictions'][0]['disease']}")
            print(f"   Xác suất: {entry['predictions'][0]['probability']*100:.2f}%")
            confidence_status = "✓ Tin cậy" if entry['is_confident'] else "⚠ Không tin cậy"
            print(f"   Trạng thái: {confidence_status}")
        
        print("\n" + "=" * 70 + "\n")
    
    def batch_predict(self, questions, output_file=None):
        """
        Dự đoán cho nhiều câu hỏi cùng lúc
        
        Args:
            questions: List các câu hỏi
            output_file: File để lưu kết quả (optional)
        
        Returns:
            List kết quả dự đoán
        """
        results = []
        
        print(f"\n🔄 Đang xử lý {len(questions)} câu hỏi...")
        
        for i, question in enumerate(questions, 1):
            print(f"   [{i}/{len(questions)}] Processing...", end='\r')
            preds, is_confident = self.predict(question, top_k=3)
            results.append({
                'question': question,
                'predictions': [
                    {'disease': disease, 'probability': float(prob)}
                    for disease, prob in preds
                ],
                'is_confident': is_confident
            })
        
        print(f"\n✓ Hoàn thành xử lý {len(questions)} câu hỏi!")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Kết quả đã được lưu tại: {output_file}")
        
        return results

def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chatbot Y Tế Tiếng Việt - Phiên bản cải tiến')
    parser.add_argument('--model', type=str, default='./chatbot_model_improved',
                        help='Đường dẫn đến model')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--batch', type=str, default=None,
                        help='File chứa danh sách câu hỏi để xử lý batch')
    parser.add_argument('--output', type=str, default=None,
                        help='File output cho batch processing')
    
    args = parser.parse_args()
    
    # Kiểm tra model path
    if not os.path.exists(args.model):
        # Thử fallback sang model cũ
        if os.path.exists('./chatbot_model'):
            print(f"⚠️  Không tìm thấy model tại {args.model}")
            print("   Sử dụng model cũ tại ./chatbot_model")
            args.model = './chatbot_model'
        else:
            print(f"❌ Không tìm thấy model tại {args.model}")
            print("Vui lòng chạy train_chatbot_improved.py trước để train model.")
            return
    
    # Khởi tạo chatbot
    chatbot = ImprovedMedicalChatbot(
        model_path=args.model,
        confidence_threshold=args.threshold
    )
    
    # Batch processing hoặc interactive chat
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ Không tìm thấy file: {args.batch}")
            return
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        output_file = args.output or 'batch_predictions.json'
        chatbot.batch_predict(questions, output_file)
    else:
        # Chạy chế độ chat
        chatbot.chat()

if __name__ == "__main__":
    main()

