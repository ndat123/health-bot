"""
Module inference - Dự đoán bệnh từ triệu chứng
Cung cấp interface thân thiện cho người dùng
"""

import os
import json
from typing import List, Tuple, Optional
from datetime import datetime
from train_model import DiseaseClassifier
from data_preprocessing import VietnameseTextPreprocessor


class MedicalDiagnosisAssistant:
    """
    Trợ lý chẩn đoán y tế dựa trên AI
    
    LƯU Ý: Đây KHÔNG phải là chẩn đoán y tế chính thức!
    """
    
    def __init__(self, model_dir: str, model_type: str = 'logistic_regression',
                 confidence_threshold: float = 0.15):
        """
        Khởi tạo trợ lý chẩn đoán
        
        Args:
            model_dir: Thư mục chứa model
            model_type: Loại model
            confidence_threshold: Ngưỡng confidence tối thiểu
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print("🔄 Đang khởi tạo trợ lý chẩn đoán...")
        self.classifier = DiseaseClassifier.load_model(model_dir, model_type)
        
        # Preprocessor
        self.preprocessor = VietnameseTextPreprocessor(remove_accents=False)
        
        # Conversation history
        self.history = []
        
        print("✅ Trợ lý đã sẵn sàng!")
        print(f"📊 Model: {model_type}")
        print(f"📊 Số loại bệnh: {len(self.classifier.disease_mapping)}")
        print(f"📊 Ngưỡng confidence: {confidence_threshold*100:.1f}%")
    
    def preprocess_symptoms(self, symptoms: str) -> str:
        """
        Tiền xử lý mô tả triệu chứng
        
        Args:
            symptoms: Mô tả triệu chứng từ người dùng
            
        Returns:
            Triệu chứng đã xử lý
        """
        return self.preprocessor.preprocess(symptoms, remove_stopwords=False)
    
    def diagnose(self, symptoms: str, top_k: int = 3, 
                 return_details: bool = False) -> dict:
        """
        Chẩn đoán bệnh từ triệu chứng
        
        Args:
            symptoms: Mô tả triệu chứng
            top_k: Số lượng bệnh có khả năng cao nhất
            return_details: Trả về thông tin chi tiết
            
        Returns:
            Dictionary chứa kết quả chẩn đoán
        """
        # Validate input
        if not symptoms or len(symptoms.strip()) < 10:
            return {
                'success': False,
                'error': 'Vui lòng mô tả triệu chứng chi tiết hơn (ít nhất 10 ký tự)',
                'symptoms': symptoms
            }
        
        # Preprocess
        processed_symptoms = self.preprocess_symptoms(symptoms)
        
        if not processed_symptoms:
            return {
                'success': False,
                'error': 'Không thể xử lý mô tả triệu chứng. Vui lòng thử lại.',
                'symptoms': symptoms
            }
        
        # Predict
        try:
            predictions = self.classifier.predict(processed_symptoms, top_k=top_k)
        except Exception as e:
            return {
                'success': False,
                'error': f'Lỗi khi dự đoán: {str(e)}',
                'symptoms': symptoms
            }
        
        if not predictions:
            return {
                'success': False,
                'error': 'Không thể dự đoán bệnh từ triệu chứng này.',
                'symptoms': symptoms
            }
        
        # Check confidence
        max_confidence = predictions[0][1]
        is_confident = max_confidence >= self.confidence_threshold
        
        # Prepare result
        result = {
            'success': True,
            'symptoms': symptoms,
            'processed_symptoms': processed_symptoms,
            'predictions': [
                {
                    'disease': disease,
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence*100:.2f}%"
                }
                for disease, confidence in predictions
            ],
            'top_prediction': {
                'disease': predictions[0][0],
                'confidence': float(predictions[0][1]),
                'confidence_percent': f"{predictions[0][1]*100:.2f}%"
            },
            'is_confident': is_confident,
            'confidence_threshold': self.confidence_threshold,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if return_details:
            result['model_type'] = self.model_type
            result['model_dir'] = self.model_dir
        
        # Save to history
        self.history.append(result)
        
        return result
    
    def format_diagnosis_output(self, result: dict) -> str:
        """
        Format kết quả chẩn đoán thành text đẹp
        
        Args:
            result: Dictionary kết quả từ diagnose()
            
        Returns:
            String đã format
        """
        if not result['success']:
            return f"❌ Lỗi: {result['error']}"
        
        output = []
        output.append("\n" + "="*70)
        output.append("🏥 KẾT QUẢ DỰ ĐOÁN")
        output.append("="*70)
        
        # Cảnh báo nếu confidence thấp
        if not result['is_confident']:
            output.append("\n⚠️  CẢNH BÁO: Độ tin cậy thấp!")
            output.append(f"   Xác suất cao nhất chỉ {result['top_prediction']['confidence_percent']}")
            output.append(f"   (ngưỡng tối thiểu: {self.confidence_threshold*100:.1f}%)")
            output.append("   → Triệu chứng có thể không rõ ràng hoặc cần thêm thông tin.\n")
        
        output.append("\n📋 Dựa trên các triệu chứng bạn mô tả:")
        output.append(f'   "{result["symptoms"]}"')
        output.append("\n💡 Bạn có thể đang mắc:")
        
        # Hiển thị predictions
        for i, pred in enumerate(result['predictions'], 1):
            confidence = pred['confidence']
            disease = pred['disease']
            
            # Emoji và mức độ tin cậy
            if confidence >= 0.7:
                emoji, level = "🟢", "RẤT CAO"
            elif confidence >= 0.4:
                emoji, level = "🟡", "CAO"
            elif confidence >= 0.2:
                emoji, level = "🟠", "TRUNG BÌNH"
            else:
                emoji, level = "🔴", "THẤP"
            
            # Progress bar
            bar_length = int(confidence * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            output.append(f"\n{i}. {disease}")
            output.append(f"   {emoji} Độ tin cậy: {level} ({pred['confidence_percent']})")
            output.append(f"   [{bar}]")
        
        # Đề xuất
        if not result['is_confident']:
            output.append("\n💡 ĐỀ XUẤT:")
            output.append("   • Mô tả chi tiết hơn về triệu chứng")
            output.append("   • Cung cấp thông tin về thời gian xuất hiện")
            output.append("   • Nêu các triệu chứng kèm theo khác")
        
        # Warning
        output.append("\n" + "⚠️ "*35)
        output.append("⚠️  LƯU Ý QUAN TRỌNG:")
        output.append("   • Đây chỉ là DỰ ĐOÁN SƠ BỘ dựa trên AI")
        output.append("   • KHÔNG PHẢI là chẩn đoán y tế chính thức")
        output.append("   • Bạn NÊN đến gặp bác sĩ để được khám và chẩn đoán chính xác")
        output.append("   • KHÔNG tự ý điều trị dựa trên kết quả này")
        output.append("   • Nếu triệu chứng nghiêm trọng, hãy đi khám NGAY!")
        output.append("⚠️ "*35)
        output.append("="*70 + "\n")
        
        return "\n".join(output)
    
    def interactive_chat(self):
        """Chế độ chat tương tác"""
        print("\n" + "="*70)
        print("🏥 TRỢ LÝ CHẨN ĐOÁN BỆNH TIẾNG VIỆT")
        print("="*70)
        print("\nCách sử dụng:")
        print("  • Nhập mô tả triệu chứng của bạn")
        print("  • Gõ 'quit', 'exit' hoặc 'thoát' để kết thúc")
        print("  • Gõ 'history' để xem lịch sử")
        print("  • Gõ 'clear' để xóa lịch sử")
        print("\n" + "-"*70)
        print(f"Model: {self.model_type}")
        print(f"Số loại bệnh: {len(self.classifier.disease_mapping)}")
        print("-"*70 + "\n")
        
        while True:
            try:
                # Nhập triệu chứng
                symptoms = input("🧑 Triệu chứng của bạn: ").strip()
                
                # Xử lý commands
                if symptoms.lower() in ['quit', 'exit', 'thoát', 'q']:
                    print("\n👋 Cảm ơn bạn đã sử dụng! Hãy chăm sóc sức khỏe!")
                    print("💡 Nhớ đến gặp bác sĩ nếu triệu chứng kéo dài!\n")
                    break
                
                if symptoms.lower() == 'history':
                    self.show_history()
                    continue
                
                if symptoms.lower() == 'clear':
                    self.history = []
                    print("\n✓ Đã xóa lịch sử\n")
                    continue
                
                if not symptoms:
                    continue
                
                # Chẩn đoán
                result = self.diagnose(symptoms, top_k=5)
                
                # Hiển thị kết quả
                print(self.format_diagnosis_output(result))
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Đã nhận Ctrl+C")
                print("👋 Cảm ơn bạn đã sử dụng!\n")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {str(e)}")
                print("Vui lòng thử lại.\n")
    
    def show_history(self):
        """Hiển thị lịch sử chẩn đoán"""
        if not self.history:
            print("\n📭 Chưa có lịch sử chẩn đoán\n")
            return
        
        print("\n" + "="*70)
        print(f"📜 LỊCH SỬ CHẨN ĐOÁN ({len(self.history)} lần)")
        print("="*70)
        
        for i, entry in enumerate(self.history, 1):
            if not entry['success']:
                continue
            
            print(f"\n{i}. [{entry['timestamp']}]")
            print(f"   Triệu chứng: {entry['symptoms'][:60]}...")
            print(f"   Dự đoán: {entry['top_prediction']['disease']}")
            print(f"   Độ tin cậy: {entry['top_prediction']['confidence_percent']}")
            status = "✓ Tin cậy" if entry['is_confident'] else "⚠ Cần thêm thông tin"
            print(f"   Trạng thái: {status}")
        
        print("\n" + "="*70 + "\n")
    
    def batch_diagnose(self, symptoms_list: List[str]) -> List[dict]:
        """
        Chẩn đoán hàng loạt
        
        Args:
            symptoms_list: Danh sách các mô tả triệu chứng
            
        Returns:
            List kết quả chẩn đoán
        """
        results = []
        
        print(f"\n🔄 Đang xử lý {len(symptoms_list)} ca...")
        
        for i, symptoms in enumerate(symptoms_list, 1):
            print(f"   [{i}/{len(symptoms_list)}]", end='\r')
            result = self.diagnose(symptoms, top_k=3)
            results.append(result)
        
        print(f"\n✓ Hoàn thành {len(symptoms_list)} ca!")
        
        # Thống kê
        successful = sum(1 for r in results if r['success'])
        confident = sum(1 for r in results if r.get('is_confident', False))
        
        print(f"\n📊 Thống kê:")
        print(f"   Thành công: {successful}/{len(symptoms_list)}")
        print(f"   Độ tin cậy cao: {confident}/{successful}")
        
        return results


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Trợ lý chẩn đoán bệnh dựa trên triệu chứng'
    )
    parser.add_argument(
        '--model-dir', 
        type=str, 
        default='./saved_models/logistic_regression',
        help='Thư mục chứa model'
    )
    parser.add_argument(
        '--model-type', 
        type=str, 
        default='logistic_regression',
        choices=['logistic_regression', 'naive_bayes', 'random_forest', 'svm'],
        help='Loại model'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.15,
        help='Ngưỡng confidence (0-1)'
    )
    parser.add_argument(
        '--symptoms', 
        type=str, 
        default=None,
        help='Mô tả triệu chứng (nếu không dùng chế độ interactive)'
    )
    parser.add_argument(
        '--batch-file', 
        type=str, 
        default=None,
        help='File chứa danh sách triệu chứng (mỗi dòng 1 case)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='File lưu kết quả (cho batch mode)'
    )
    
    args = parser.parse_args()
    
    # Kiểm tra model có tồn tại không
    if not os.path.exists(args.model_dir):
        print(f"❌ Không tìm thấy model tại: {args.model_dir}")
        print("💡 Vui lòng chạy train_model.py trước để train model!")
        return
    
    # Khởi tạo assistant
    assistant = MedicalDiagnosisAssistant(
        model_dir=args.model_dir,
        model_type=args.model_type,
        confidence_threshold=args.threshold
    )
    
    # Xử lý theo mode
    if args.batch_file:
        # Batch mode
        if not os.path.exists(args.batch_file):
            print(f"❌ Không tìm thấy file: {args.batch_file}")
            return
        
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            symptoms_list = [line.strip() for line in f if line.strip()]
        
        results = assistant.batch_diagnose(symptoms_list)
        
        # Lưu kết quả
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Kết quả đã lưu tại: {args.output}")
    
    elif args.symptoms:
        # Single prediction mode
        result = assistant.diagnose(args.symptoms, top_k=5)
        print(assistant.format_diagnosis_output(result))
    
    else:
        # Interactive mode
        assistant.interactive_chat()


if __name__ == "__main__":
    main()


