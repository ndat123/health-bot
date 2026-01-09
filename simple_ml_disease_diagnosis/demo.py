"""
Script demo - Ví dụ sử dụng hệ thống chẩn đoán bệnh
"""

import os
import sys

# Thêm thư mục hiện tại vào Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import MedicalDiagnosisAssistant


def demo_basic_usage():
    """Demo sử dụng cơ bản"""
    print("\n" + "="*70)
    print("DEMO 1: SỬ DỤNG CƠ BẢN")
    print("="*70)
    
    # Khởi tạo assistant
    assistant = MedicalDiagnosisAssistant(
        model_dir='./saved_models/logistic_regression',
        model_type='logistic_regression',
        confidence_threshold=0.15
    )
    
    # Các ví dụ triệu chứng
    test_cases = [
        "Đau đầu, sốt cao, mệt mỏi, buồn nôn",
        "Ho khan, khó thở, đau ngực, sốt nhẹ",
        "Đau bụng, tiêu chảy, buồn nôn, mệt mỏi",
        "Ngứa da, phát ban đỏ, sưng",
        "Đau khớp, sưng khớp, khó cử động"
    ]
    
    print("\n📋 Đang test với các triệu chứng mẫu...\n")
    
    for i, symptoms in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}")
        print(f"{'='*70}")
        print(f"Triệu chứng: {symptoms}")
        
        result = assistant.diagnose(symptoms, top_k=3)
        
        if result['success']:
            print(f"\n🏥 Dự đoán hàng đầu:")
            top = result['top_prediction']
            print(f"   Bệnh: {top['disease']}")
            print(f"   Độ tin cậy: {top['confidence_percent']}")
            
            print(f"\n📊 Top 3 khả năng:")
            for j, pred in enumerate(result['predictions'], 1):
                print(f"   {j}. {pred['disease']} - {pred['confidence_percent']}")
        else:
            print(f"❌ Lỗi: {result['error']}")
        
        print()


def demo_detailed_output():
    """Demo với output chi tiết"""
    print("\n" + "="*70)
    print("DEMO 2: OUTPUT CHI TIẾT")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant(
        model_dir='./saved_models/logistic_regression',
        model_type='logistic_regression',
        confidence_threshold=0.15
    )
    
    symptoms = "Tôi đang cảm thấy đau đầu dữ dội, sốt cao 39 độ, mệt mỏi và buồn nôn. Triệu chứng xuất hiện từ 2 ngày nay."
    
    print(f"\nTriệu chứng: {symptoms}")
    
    result = assistant.diagnose(symptoms, top_k=5, return_details=True)
    
    # In output đẹp
    print(assistant.format_diagnosis_output(result))


def demo_batch_prediction():
    """Demo dự đoán hàng loạt"""
    print("\n" + "="*70)
    print("DEMO 3: DỰ ĐOÁN HÀNG LOẠT")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant(
        model_dir='./saved_models/logistic_regression',
        model_type='logistic_regression',
        confidence_threshold=0.15
    )
    
    # Danh sách triệu chứng
    symptoms_list = [
        "Sốt cao, đau đầu, đau cơ, buồn nôn",
        "Ho, sổ mũi, đau họng, sốt nhẹ",
        "Đau bụng, tiêu chảy, buồn nôn",
        "Ngứa ngáy da, nổi mẩn đỏ",
        "Khó thở, đau ngực, tim đập nhanh",
        "Chóng mặt, hoa mắt, yếu người",
        "Đau lưng, tê chân tay",
        "Mất ngủ, lo âu, căng thẳng"
    ]
    
    print(f"\n📋 Test với {len(symptoms_list)} trường hợp...\n")
    
    results = assistant.batch_diagnose(symptoms_list)
    
    # Hiển thị tổng hợp
    print("\n" + "="*70)
    print("KẾT QUẢ TỔNG HỢP")
    print("="*70)
    
    for i, (symptoms, result) in enumerate(zip(symptoms_list, results), 1):
        if result['success']:
            top = result['top_prediction']
            status = "✓" if result['is_confident'] else "⚠"
            print(f"\n{i}. {status} {symptoms[:50]}...")
            print(f"   → {top['disease']} ({top['confidence_percent']})")
        else:
            print(f"\n{i}. ❌ {symptoms[:50]}...")
            print(f"   → Lỗi: {result['error']}")


def demo_compare_models():
    """Demo so sánh các models"""
    print("\n" + "="*70)
    print("DEMO 4: SO SÁNH CÁC MODELS")
    print("="*70)
    
    symptoms = "Đau đầu, sốt cao, mệt mỏi, buồn nôn, đau cơ"
    
    models = [
        ('logistic_regression', 'Logistic Regression'),
        ('naive_bayes', 'Naive Bayes'),
        ('random_forest', 'Random Forest'),
        ('svm', 'Support Vector Machine')
    ]
    
    print(f"\nTriệu chứng: {symptoms}\n")
    print(f"{'Model':<25} {'Dự đoán':<30} {'Confidence':>15}")
    print("-"*70)
    
    for model_type, model_name in models:
        model_dir = f'./saved_models/{model_type}'
        
        # Kiểm tra model có tồn tại không
        if not os.path.exists(model_dir):
            print(f"{model_name:<25} {'Model chưa được train':<30} {'-':>15}")
            continue
        
        try:
            assistant = MedicalDiagnosisAssistant(
                model_dir=model_dir,
                model_type=model_type,
                confidence_threshold=0.15
            )
            
            result = assistant.diagnose(symptoms, top_k=1)
            
            if result['success']:
                top = result['top_prediction']
                disease = top['disease'][:28]  # Giới hạn độ dài
                confidence = top['confidence_percent']
                print(f"{model_name:<25} {disease:<30} {confidence:>15}")
            else:
                print(f"{model_name:<25} {'Lỗi':<30} {'-':>15}")
        
        except Exception as e:
            print(f"{model_name:<25} {'Error: ' + str(e)[:20]:<30} {'-':>15}")


def demo_interactive():
    """Demo chế độ interactive"""
    print("\n" + "="*70)
    print("DEMO 5: CHẾ ĐỘ INTERACTIVE")
    print("="*70)
    
    print("\nBạn có muốn thử chế độ chat tương tác không?")
    print("(Gõ 'y' để tiếp tục, Enter để bỏ qua)")
    
    choice = input("Lựa chọn: ").strip().lower()
    
    if choice == 'y':
        assistant = MedicalDiagnosisAssistant(
            model_dir='./saved_models/logistic_regression',
            model_type='logistic_regression',
            confidence_threshold=0.15
        )
        
        assistant.interactive_chat()
    else:
        print("Đã bỏ qua demo interactive.")


def main():
    """Main function"""
    print("\n" + "🏥"*35)
    print("HỆ THỐNG CHẨN ĐOÁN BỆNH DỰA TRÊN TRIỆU CHỨNG")
    print("MEDICAL DISEASE DIAGNOSIS SYSTEM")
    print("🏥"*35)
    
    # Kiểm tra models có tồn tại không
    if not os.path.exists('./saved_models'):
        print("\n❌ CẢNH BÁO: Chưa có models được train!")
        print("💡 Vui lòng chạy train_model.py trước:")
        print("   python train_model.py")
        print("\nHoặc để train tất cả models:")
        print("   cd simple_ml_disease_diagnosis")
        print("   python train_model.py")
        return
    
    try:
        # Demo 1: Basic usage
        demo_basic_usage()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        # Demo 2: Detailed output
        demo_detailed_output()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        # Demo 3: Batch prediction
        demo_batch_prediction()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        # Demo 4: Compare models
        demo_compare_models()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        # Demo 5: Interactive
        demo_interactive()
        
        print("\n" + "="*70)
        print("✅ HOÀN THÀNH TẤT CẢ DEMO!")
        print("="*70)
        print("\n💡 Để sử dụng chế độ interactive:")
        print("   python inference.py")
        print("\n💡 Để dự đoán 1 triệu chứng:")
        print('   python inference.py --symptoms "đau đầu, sốt cao"')
        print("\n💡 Để xem tất cả options:")
        print("   python inference.py --help")
        print()
        
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


