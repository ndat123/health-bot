"""
Hệ thống chẩn đoán bệnh sử dụng Google Gemini API
Dựa trên dataset ViMedical_Disease với 603 loại bệnh
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import google.generativeai as genai

class GeminiDiseaseDiagnosis:
    """
    Chatbot chẩn đoán bệnh sử dụng Gemini API
    """
    
    def __init__(self, api_key: str = None, dataset_path: str = "ViMedical_Disease.csv"):
        """
        Khởi tạo Gemini Diagnosis System
        
        Args:
            api_key: Google API Key (nếu None, sẽ lấy từ env GOOGLE_API_KEY)
            dataset_path: Đường dẫn đến file dataset
        """
        # Setup API
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError(
                    "Vui lòng cung cấp GOOGLE_API_KEY!\n"
                    "Cách 1: Truyền vào api_key parameter\n"
                    "Cách 2: Set environment variable GOOGLE_API_KEY"
                )
        
        genai.configure(api_key=api_key)
        
        # Load model
        print("🔄 Đang khởi tạo Gemini model...")
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Load dataset để hiểu về các bệnh
        print(f"📂 Đang load dataset từ {dataset_path}...")
        self.df = pd.read_csv(dataset_path, encoding='utf-8')
        self.diseases = sorted(self.df['Disease'].unique().tolist())
        
        print(f"✅ Đã load {len(self.diseases)} loại bệnh")
        
        # Tạo knowledge base cho model
        self._build_knowledge_base()
        
        # History
        self.conversation_history = []
        
        print("✅ Gemini Disease Diagnosis System sẵn sàng!\n")
    
    def _build_knowledge_base(self):
        """Xây dựng knowledge base từ dataset"""
        print("🔨 Đang xây dựng knowledge base...")
        
        # Tạo mapping disease -> symptoms
        self.disease_symptoms = {}
        for disease in self.diseases:
            symptoms = self.df[self.df['Disease'] == disease]['Question'].tolist()
            self.disease_symptoms[disease] = symptoms[:10]  # Lấy 10 mẫu đầu
        
        # Tạo system instruction
        self.system_instruction = self._create_system_instruction()
        print("✅ Knowledge base đã sẵn sàng")
    
    def _create_system_instruction(self) -> str:
        """Tạo system instruction cho Gemini"""
        diseases_list = "\n".join([f"- {d}" for d in self.diseases[:50]])  # Top 50 bệnh
        
        return f"""Bạn là một trợ lý y tế AI chuyên nghiệp, được đào tạo để hỗ trợ chẩn đoán bệnh dựa trên triệu chứng.

NHIỆM VỤ:
1. Phân tích triệu chứng người dùng mô tả (bằng tiếng Việt)
2. Dự đoán 3-5 bệnh có khả năng cao nhất từ database
3. Giải thích lý do và đưa ra lời khuyên

DATABASE: Bạn có kiến thức về {len(self.diseases)} loại bệnh phổ biến, bao gồm:
{diseases_list}
... và {len(self.diseases) - 50} bệnh khác

ĐỊNH DẠNG TRẢ LỜI:
```
🔍 PHÂN TÍCH TRIỆU CHỨNG:
[Tóm tắt các triệu chứng chính]

💡 DỰ ĐOÁN BỆNH (Top 3-5):

1. [Tên bệnh]
   📊 Độ tin cậy: [Cao/Trung bình/Thấp] (~[%]%)
   📝 Lý do: [Giải thích ngắn gọn tại sao]
   
2. [Tên bệnh]
   📊 Độ tin cậy: [Cao/Trung bình/Thấp] (~[%]%)
   📝 Lý do: [Giải thích]
   
3. [Tên bệnh]
   📊 Độ tin cậy: [Cao/Trung bình/Thấp] (~[%]%)
   📝 Lý do: [Giải thích]

💊 KHUYẾN NGHỊ:
- [Lời khuyên 1]
- [Lời khuyên 2]
- [Lời khuyên 3]

⚠️ LƯU Ý QUAN TRỌNG:
- Đây chỉ là dự đoán sơ bộ dựa trên AI, KHÔNG PHẢI chẩn đoán y tế
- Bạn NÊN đến gặp bác sĩ để được khám và chẩn đoán chính xác
- KHÔNG tự ý điều trị dựa trên kết quả này
```

QUY TẮC:
1. Luôn trả lời bằng tiếng Việt
2. Chỉ dự đoán bệnh có trong database
3. Nếu triệu chứng không rõ ràng, yêu cầu thêm thông tin
4. Luôn nhắc nhở người dùng đi khám bác sĩ
5. Đánh giá độ tin cậy dựa trên mức độ khớp triệu chứng:
   - Cao (70-90%): Triệu chứng rất khớp
   - Trung bình (40-70%): Triệu chứng khớp một phần
   - Thấp (<40%): Triệu chứng mơ hồ hoặc không điển hình
6. Nếu câu hỏi không liên quan y tế, lịch sự từ chối và hướng dẫn đúng cách
"""
    
    def predict(self, symptoms: str, context: str = None) -> Dict:
        """
        Dự đoán bệnh từ triệu chứng
        
        Args:
            symptoms: Mô tả triệu chứng
            context: Thông tin bổ sung (tuổi, giới tính, tiền sử...)
        
        Returns:
            Dict chứa kết quả dự đoán
        """
        print(f"\n{'='*70}")
        print(f"🔍 ĐANG PHÂN TÍCH TRIỆU CHỨNG...")
        print(f"{'='*70}")
        
        # Tạo prompt
        prompt = self._create_prompt(symptoms, context)
        
        try:
            # Gọi Gemini API
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Giảm creativity, tăng accuracy
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                )
            )
            
            result = {
                'success': True,
                'symptoms': symptoms,
                'response': response.text,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'Gemini 2.0 Flash'
            }
            
            # Lưu vào history
            self.conversation_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'symptoms': symptoms,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return error_result
    
    def _create_prompt(self, symptoms: str, context: str = None) -> str:
        """Tạo prompt cho Gemini"""
        prompt = f"{self.system_instruction}\n\n"
        prompt += f"TRIỆU CHỨNG CỦA NGƯỜI DÙNG:\n{symptoms}\n"
        
        if context:
            prompt += f"\nTHÔNG TIN BỔ SUNG:\n{context}\n"
        
        prompt += "\nHãy phân tích và đưa ra dự đoán theo định dạng đã hướng dẫn."
        
        return prompt
    
    def chat(self, save_history: bool = True):
        """
        Chế độ chat tương tác
        
        Args:
            save_history: Có lưu lịch sử không
        """
        print("\n" + "="*70)
        print("💬 GEMINI DISEASE DIAGNOSIS CHATBOT")
        print("="*70)
        print("Mô tả triệu chứng của bạn để được tư vấn.")
        print("Gõ 'quit' hoặc 'exit' để thoát.")
        print("Gõ 'history' để xem lịch sử.")
        print("="*70)
        print(f"\n📊 Hệ thống:")
        print(f"   - Model: Gemini 2.0 Flash")
        print(f"   - Database: {len(self.diseases)} loại bệnh")
        print(f"   - Powered by: Google AI")
        print("\n" + "-"*70 + "\n")
        
        while True:
            try:
                # Input
                symptoms = input("🧑 Bạn: ").strip()
                
                # Commands
                if symptoms.lower() in ['quit', 'exit', 'thoát', 'q']:
                    if save_history and self.conversation_history:
                        self._save_history()
                    print("\n👋 Cảm ơn bạn đã sử dụng! Hẹn gặp lại!")
                    break
                
                if symptoms.lower() == 'history':
                    self._show_history()
                    continue
                
                if not symptoms:
                    continue
                
                # Predict
                result = self.predict(symptoms)
                
                # Display result
                if result['success']:
                    print(f"\n{'='*70}")
                    print("🤖 GEMINI AI:")
                    print(f"{'='*70}")
                    print(result['response'])
                    print(f"{'='*70}\n")
                else:
                    print(f"\n❌ Lỗi: {result['error']}\n")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Đã nhận Ctrl+C. Đang thoát...")
                if save_history and self.conversation_history:
                    self._save_history()
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {str(e)}\n")
    
    def _show_history(self):
        """Hiển thị lịch sử"""
        if not self.conversation_history:
            print("\n📭 Chưa có lịch sử.\n")
            return
        
        print("\n" + "="*70)
        print(f"📜 LỊCH SỬ ({len(self.conversation_history)} lượt)")
        print("="*70)
        
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{i}. [{entry['timestamp']}]")
            print(f"   Triệu chứng: {entry['symptoms'][:80]}...")
            if entry['success']:
                print(f"   ✅ Đã phân tích thành công")
            else:
                print(f"   ❌ Lỗi: {entry.get('error', 'Unknown')}")
        
        print("\n" + "="*70 + "\n")
    
    def _save_history(self):
        """Lưu lịch sử"""
        filename = f"gemini_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Lịch sử đã được lưu: {filename}")
    
    def batch_predict(self, symptoms_list: List[str], output_file: str = None) -> List[Dict]:
        """
        Dự đoán hàng loạt
        
        Args:
            symptoms_list: Danh sách triệu chứng
            output_file: File lưu kết quả
        
        Returns:
            List kết quả
        """
        print(f"\n🔄 Đang xử lý {len(symptoms_list)} trường hợp...")
        
        results = []
        for i, symptoms in enumerate(symptoms_list, 1):
            print(f"   [{i}/{len(symptoms_list)}] Đang phân tích...", end='\r')
            result = self.predict(symptoms)
            results.append(result)
        
        print(f"\n✅ Hoàn thành xử lý {len(symptoms_list)} trường hợp!")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Kết quả đã được lưu: {output_file}")
        
        return results
    
    def get_disease_info(self, disease_name: str) -> Dict:
        """
        Lấy thông tin về một bệnh cụ thể
        
        Args:
            disease_name: Tên bệnh
        
        Returns:
            Dict chứa thông tin bệnh
        """
        if disease_name not in self.diseases:
            return {
                'success': False,
                'error': f'Không tìm thấy bệnh "{disease_name}" trong database'
            }
        
        symptoms = self.disease_symptoms.get(disease_name, [])
        
        prompt = f"""Cung cấp thông tin chi tiết về bệnh "{disease_name}" theo định dạng sau:

📋 THÔNG TIN VỀ {disease_name.upper()}

🔍 MÔ TẢ:
[Mô tả ngắn gọn về bệnh]

⚠️ TRIỆU CHỨNG ĐIỂN HÌNH:
- [Triệu chứng 1]
- [Triệu chứng 2]
- [Triệu chứng 3]

🏥 NGUYÊN NHÂN:
- [Nguyên nhân 1]
- [Nguyên nhân 2]

💊 ĐIỀU TRỊ:
- [Phương pháp điều trị]

🛡️ PHÒNG NGỪA:
- [Cách phòng ngừa]

⚠️ KHI NÀO CẦN ĐI KHÁM GẤP:
- [Dấu hiệu nguy hiểm]
"""
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'disease': disease_name,
                'info': response.text
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemini Disease Diagnosis System')
    parser.add_argument('--api-key', type=str, help='Google API Key')
    parser.add_argument('--dataset', type=str, default='ViMedical_Disease.csv',
                        help='Đường dẫn dataset')
    parser.add_argument('--batch', type=str, help='File chứa danh sách triệu chứng')
    parser.add_argument('--output', type=str, help='File output cho batch processing')
    parser.add_argument('--info', type=str, help='Xem thông tin về một bệnh cụ thể')
    
    args = parser.parse_args()
    
    # Kiểm tra API key
    if not args.api_key and not os.getenv('GOOGLE_API_KEY'):
        print("❌ Lỗi: Chưa có Google API Key!")
        print("\nCách lấy API Key:")
        print("1. Truy cập: https://makersuite.google.com/app/apikey")
        print("2. Tạo API key mới")
        print("3. Copy API key")
        print("\nCách sử dụng:")
        print("  python gemini_disease_diagnosis.py --api-key YOUR_API_KEY")
        print("  hoặc")
        print("  set GOOGLE_API_KEY=YOUR_API_KEY")
        print("  python gemini_disease_diagnosis.py")
        return
    
    try:
        # Khởi tạo system
        system = GeminiDiseaseDiagnosis(
            api_key=args.api_key,
            dataset_path=args.dataset
        )
        
        # Batch processing
        if args.batch:
            with open(args.batch, 'r', encoding='utf-8') as f:
                symptoms_list = [line.strip() for line in f if line.strip()]
            
            output_file = args.output or 'gemini_batch_results.json'
            system.batch_predict(symptoms_list, output_file)
        
        # Disease info
        elif args.info:
            result = system.get_disease_info(args.info)
            if result['success']:
                print(result['info'])
            else:
                print(f"❌ {result['error']}")
        
        # Interactive chat
        else:
            system.chat()
    
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

