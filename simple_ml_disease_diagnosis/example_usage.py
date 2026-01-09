"""
Ví dụ sử dụng hệ thống trong code Python
Example usage in Python code
"""

import sys
import os

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import MedicalDiagnosisAssistant


def example_1_basic_prediction():
    """Ví dụ 1: Dự đoán cơ bản"""
    print("\n" + "="*70)
    print("VÍ DỤ 1: DỰ ĐOÁN CÔ BẢN")
    print("="*70)
    
    # Khởi tạo assistant
    assistant = MedicalDiagnosisAssistant(
        model_dir='./saved_models/logistic_regression',
        model_type='logistic_regression',
        confidence_threshold=0.15
    )
    
    # Dự đoán
    symptoms = "Đau đầu, sốt cao, mệt mỏi, buồn nôn"
    result = assistant.diagnose(symptoms, top_k=3)
    
    # Hiển thị kết quả
    if result['success']:
        print(f"\n✅ Dự đoán thành công!")
        print(f"Triệu chứng: {symptoms}")
        print(f"\nTop prediction:")
        print(f"  Bệnh: {result['top_prediction']['disease']}")
        print(f"  Độ tin cậy: {result['top_prediction']['confidence_percent']}")
        print(f"\nTop 3 khả năng:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['disease']} - {pred['confidence_percent']}")
    else:
        print(f"❌ Lỗi: {result['error']}")


def example_2_multiple_predictions():
    """Ví dụ 2: Nhiều dự đoán"""
    print("\n" + "="*70)
    print("VÍ DỤ 2: NHIỀU DỰ ĐOÁN")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant(
        model_dir='./saved_models/logistic_regression',
        model_type='logistic_regression'
    )
    
    test_cases = [
        "Đau đầu, sốt cao",
        "Ho, sổ mũi, đau họng",
        "Đau bụng, tiêu chảy",
    ]
    
    print(f"\n📋 Dự đoán cho {len(test_cases)} trường hợp:\n")
    
    for i, symptoms in enumerate(test_cases, 1):
        result = assistant.diagnose(symptoms, top_k=1)
        if result['success']:
            top = result['top_prediction']
            print(f"{i}. Triệu chứng: {symptoms}")
            print(f"   → {top['disease']} ({top['confidence_percent']})\n")


def example_3_batch_processing():
    """Ví dụ 3: Xử lý hàng loạt"""
    print("\n" + "="*70)
    print("VÍ DỤ 3: XỬ LÝ HÀNG LOẠT")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant(
        model_dir='./saved_models/logistic_regression',
        model_type='logistic_regression'
    )
    
    symptoms_list = [
        "Sốt cao, đau đầu, đau cơ",
        "Ho, khó thở, đau ngực",
        "Đau bụng, buồn nôn, tiêu chảy",
        "Ngứa da, nổi mẩn đỏ",
        "Chóng mặt, hoa mắt"
    ]
    
    results = assistant.batch_diagnose(symptoms_list)
    
    print(f"\n📊 Kết quả:")
    successful = sum(1 for r in results if r['success'])
    print(f"  Thành công: {successful}/{len(results)}")


def example_4_using_different_models():
    """Ví dụ 4: Sử dụng các models khác nhau"""
    print("\n" + "="*70)
    print("VÍ DỤ 4: SO SÁNH CÁC MODELS")
    print("="*70)
    
    symptoms = "Đau đầu, sốt cao, mệt mỏi"
    
    models = [
        ('logistic_regression', 'Logistic Regression'),
        ('naive_bayes', 'Naive Bayes'),
        ('random_forest', 'Random Forest'),
        ('svm', 'SVM')
    ]
    
    print(f"\nTriệu chứng: {symptoms}\n")
    print(f"{'Model':<25} {'Dự đoán':<30} {'Confidence':>12}")
    print("-"*70)
    
    for model_type, model_name in models:
        model_dir = f'./saved_models/{model_type}'
        
        if not os.path.exists(model_dir):
            print(f"{model_name:<25} {'Chưa được train':<30} {'-':>12}")
            continue
        
        try:
            assistant = MedicalDiagnosisAssistant(
                model_dir=model_dir,
                model_type=model_type
            )
            
            result = assistant.diagnose(symptoms, top_k=1)
            
            if result['success']:
                top = result['top_prediction']
                disease = top['disease'][:28]
                conf = top['confidence_percent']
                print(f"{model_name:<25} {disease:<30} {conf:>12}")
        except Exception as e:
            print(f"{model_name:<25} {'Error':<30} {'-':>12}")


def example_5_detailed_result():
    """Ví dụ 5: Kết quả chi tiết"""
    print("\n" + "="*70)
    print("VÍ DỤ 5: KẾT QUẢ CHI TIẾT")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant(
        model_dir='./saved_models/logistic_regression',
        model_type='logistic_regression'
    )
    
    symptoms = "Đau đầu, sốt cao, mệt mỏi, buồn nôn"
    result = assistant.diagnose(symptoms, top_k=5, return_details=True)
    
    # Hiển thị output đẹp
    print(assistant.format_diagnosis_output(result))


def main():
    """Main function"""
    
    # Kiểm tra models
    if not os.path.exists('./saved_models'):
        print("\n❌ Chưa có models!")
        print("💡 Chạy: python train_model.py")
        return
    
    print("\n" + "🏥"*35)
    print("CÁC VÍ DỤ SỬ DỤNG HỆ THỐNG CHẨN ĐOÁN")
    print("🏥"*35)
    
    try:
        # Chạy các ví dụ
        example_1_basic_prediction()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        example_2_multiple_predictions()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        example_3_batch_processing()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        example_4_using_different_models()
        input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        example_5_detailed_result()
        
        print("\n" + "="*70)
        print("✅ HOÀN THÀNH TẤT CẢ VÍ DỤ!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


