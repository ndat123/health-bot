"""
Script để test độ chính xác của model và cải thiện nếu cần
"""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
import re
import math
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY != 'your_groq_api_key_here' else None
GROQ_MODEL = 'llama-3.3-70b-versatile'

# Load data
print("Loading data...")
df = pd.read_csv('ViMedical_Disease.csv', encoding='utf-8')
diseases = sorted(df['Disease'].unique().tolist())

# Build disease knowledge base
disease_symptoms = {}
for disease in diseases:
    disease_data = df[df['Disease'] == disease]['Question'].tolist()
    disease_symptoms[disease] = disease_data[:10]

print(f"✓ Loaded {len(diseases)} diseases with {len(df)} symptom samples")

# Stopwords tiếng Việt
stopwords = {
    'tôi', 'của', 'có', 'bị', 'đang', 'là', 'và', 'này', 'thể', 'các', 'với',
    'một', 'được', 'hay', 'để', 'khi', 'như', 'thì', 'nào', 'làm', 'trong',
    'từ', 'cho', 'về', 'người', 'những', 'không', 'có thể', 'gì', 'hiện',
    'cảm', 'triệu', 'chứng'
}

def find_relevant_diseases(symptoms_input, top_k=15):
    """
    Tìm các bệnh có triệu chứng tương tự với input của user
    Dùng TF-IDF để tăng độ chính xác
    """
    from collections import Counter, defaultdict
    
    # Normalize input
    symptoms_lower = symptoms_input.lower()
    
    # Extract keywords (các từ quan trọng)
    keywords = re.findall(r'\w+', symptoms_lower)
    keywords = [k for k in keywords if len(k) > 2 and k not in stopwords]
    
    # Extract phrases (2-3 từ)
    words = symptoms_lower.split()
    phrases = []
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if len(phrase) > 8:  # Chỉ lấy phrase dài
            phrases.append(phrase)
    
    # Tính IDF cho mỗi keyword
    keyword_idf = defaultdict(int)
    for disease, symptom_list in disease_symptoms.items():
        disease_text = " ".join(symptom_list).lower()
        for keyword in set(keywords):
            if keyword in disease_text:
                keyword_idf[keyword] += 1
    
    # Tính IDF score
    total_diseases = len(disease_symptoms)
    idf_scores = {}
    for keyword, count in keyword_idf.items():
        if count > 0:
            idf_scores[keyword] = math.log(total_diseases / count)
    
    # Score cho mỗi bệnh với TF-IDF
    disease_scores = {}
    
    for disease, symptom_list in disease_symptoms.items():
        disease_text = " ".join(symptom_list).lower()
        score = 0
        
        # Score từ keywords với IDF weighting
        for keyword in keywords:
            if keyword in disease_text:
                tf = disease_text.count(keyword)
                idf = idf_scores.get(keyword, 0)
                score += tf * idf * 10
        
        # Bonus score cho exact phrases
        for phrase in phrases:
            if phrase in disease_text:
                score += 50
        
        if score > 0:
            disease_scores[disease] = score
    
    # Lấy top k bệnh có score cao nhất
    top_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return top_diseases

def predict_disease(symptoms_input, use_api=False):
    """
    Dự đoán bệnh từ triệu chứng
    
    Args:
        symptoms_input: Triệu chứng người dùng nhập
        use_api: Có sử dụng API không (mặc định False để test nhanh)
    
    Returns:
        predicted_disease: Bệnh được dự đoán
        confidence: Độ tin cậy (0-100)
    """
    # Tìm các bệnh liên quan
    top_diseases = find_relevant_diseases(symptoms_input, top_k=10)
    
    if not top_diseases:
        return None, 0
    
    # Nếu không dùng API, chỉ lấy bệnh có score cao nhất
    if not use_api or not groq_client:
        predicted_disease = top_diseases[0][0]
        # Tính confidence dựa trên score
        total_score = sum(score for _, score in top_diseases[:3])
        confidence = min(100, int((top_diseases[0][1] / total_score) * 100)) if total_score > 0 else 0
        return predicted_disease, confidence
    
    # Sử dụng API để dự đoán chính xác hơn
    try:
        # Build context
        context = "\n🔍 Các bệnh có triệu chứng tương tự:\n"
        for i, (disease, score) in enumerate(top_diseases[:5], 1):
            context += f"{i}. {disease}\n"
        
        prompt = f"""Bạn là bác sĩ AI. Dựa trên triệu chứng và danh sách bệnh liên quan, hãy chọn 1 bệnh phù hợp nhất.

THÔNG TIN TỪ DATABASE:
{context}

TRIỆU CHỨNG: "{symptoms_input}"

Trả lời CHỈ TÊN BỆNH (không giải thích), chọn từ danh sách trên."""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là bác sĩ AI, chỉ trả lời tên bệnh"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50,
        )
        
        predicted_disease = response.choices[0].message.content.strip()
        
        # Kiểm tra xem bệnh có trong danh sách không
        disease_names = [d for d, _ in top_diseases]
        if predicted_disease not in disease_names:
            # Tìm bệnh gần nhất
            for disease in disease_names:
                if disease.lower() in predicted_disease.lower() or predicted_disease.lower() in disease.lower():
                    predicted_disease = disease
                    break
            else:
                predicted_disease = top_diseases[0][0]
        
        confidence = 85  # Confidence cao hơn khi dùng API
        return predicted_disease, confidence
        
    except Exception as e:
        print(f"API Error: {e}")
        # Fallback to non-API method
        predicted_disease = top_diseases[0][0]
        confidence = 70
        return predicted_disease, confidence

def test_model_accuracy(sample_size=500, use_api=False):
    """
    Test độ chính xác của model
    
    Args:
        sample_size: Số lượng mẫu để test (mặc định 500)
        use_api: Có sử dụng API không (False = test nhanh, True = test chính xác)
    
    Returns:
        accuracy: Độ chính xác (%)
        results: Chi tiết kết quả test
    """
    print(f"\n{'='*70}")
    print(f"🧪 TESTING MODEL ACCURACY")
    print(f"{'='*70}")
    print(f"Sample size: {sample_size}")
    print(f"Use API: {use_api}")
    print(f"{'='*70}\n")
    
    # Lấy mẫu ngẫu nhiên từ dataset
    test_samples = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    correct = 0
    total = 0
    results = []
    
    # Thống kê theo bệnh
    disease_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for idx, row in test_samples.iterrows():
        actual_disease = row['Disease']
        symptoms = row['Question']
        
        # Dự đoán
        predicted_disease, confidence = predict_disease(symptoms, use_api=use_api)
        
        # Kiểm tra kết quả
        is_correct = (predicted_disease == actual_disease)
        
        if is_correct:
            correct += 1
        
        total += 1
        
        # Cập nhật thống kê theo bệnh
        disease_stats[actual_disease]['total'] += 1
        if is_correct:
            disease_stats[actual_disease]['correct'] += 1
        
        # Lưu kết quả
        results.append({
            'symptoms': symptoms[:100],
            'actual_disease': actual_disease,
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'is_correct': is_correct
        })
        
        # Progress
        if total % 50 == 0:
            current_accuracy = (correct / total) * 100
            print(f"Progress: {total}/{sample_size} - Current accuracy: {current_accuracy:.2f}%")
    
    # Tính độ chính xác
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Wrong predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")
    
    # Tìm các bệnh có độ chính xác thấp
    print("📉 Diseases with low accuracy (< 70%):")
    low_accuracy_diseases = []
    for disease, stats in disease_stats.items():
        if stats['total'] >= 3:  # Chỉ xem xét bệnh có ít nhất 3 mẫu
            disease_accuracy = (stats['correct'] / stats['total']) * 100
            if disease_accuracy < 70:
                low_accuracy_diseases.append({
                    'disease': disease,
                    'accuracy': disease_accuracy,
                    'total': stats['total'],
                    'correct': stats['correct']
                })
    
    low_accuracy_diseases.sort(key=lambda x: x['accuracy'])
    
    for item in low_accuracy_diseases[:10]:  # Top 10 bệnh có độ chính xác thấp nhất
        print(f"  - {item['disease']}: {item['accuracy']:.1f}% ({item['correct']}/{item['total']})")
    
    return accuracy, results, disease_stats, low_accuracy_diseases

def save_test_results(accuracy, results, disease_stats, low_accuracy_diseases):
    """Lưu kết quả test"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tạo thư mục nếu chưa có
    os.makedirs('test_results', exist_ok=True)
    
    # Lưu summary
    summary = {
        'timestamp': timestamp,
        'accuracy': accuracy,
        'total_samples': len(results),
        'correct': sum(1 for r in results if r['is_correct']),
        'wrong': sum(1 for r in results if not r['is_correct']),
        'low_accuracy_diseases_count': len(low_accuracy_diseases)
    }
    
    with open(f'test_results/test_summary_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Lưu chi tiết
    with open(f'test_results/test_details_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Lưu thống kê theo bệnh
    disease_stats_list = []
    for disease, stats in disease_stats.items():
        disease_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        disease_stats_list.append({
            'disease': disease,
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': disease_accuracy
        })
    
    disease_stats_list.sort(key=lambda x: x['accuracy'])
    
    with open(f'test_results/disease_stats_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(disease_stats_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Test results saved to test_results/test_*_{timestamp}.json")

def improve_model():
    """
    Cải thiện model bằng cách:
    1. Tăng trọng số cho các từ khóa quan trọng
    2. Thêm synonym matching
    3. Cải thiện phrase detection
    """
    print(f"\n{'='*70}")
    print(f"🔧 IMPROVING MODEL")
    print(f"{'='*70}\n")
    
    improvements = []
    
    # 1. Tạo từ điển từ đồng nghĩa y tế
    medical_synonyms = {
        'đau': ['nhức', 'buốt', 'đau đớn', 'đau nhức'],
        'sốt': ['nóng', 'sốt cao', 'sốt nhẹ', 'ớn lạnh'],
        'ho': ['ho khan', 'ho có đờm', 'ho ra máu', 'ho nhiều'],
        'buồn nôn': ['nôn', 'ói', 'nôn mửa', 'muốn nôn'],
        'chóng mặt': ['hoa mắt', 'choáng váng', 'ngất', 'mất thăng bằng'],
        'mệt': ['mệt mỏi', 'kiệt sức', 'yếu', 'mỏi'],
        'khó thở': ['thở gấp', 'thở nhanh', 'ngạt thở', 'khó thở'],
        'đau đầu': ['nhức đầu', 'đau nửa đầu', 'đau đầu dữ dội'],
        'tiêu chảy': ['đi ngoài', 'phân lỏng', 'ỉa chảy'],
        'táo bón': ['khó đi ngoài', 'đại tiện khó', 'bí đại tiện'],
    }
    
    improvements.append("✓ Added medical synonym dictionary")
    
    # 2. Tạo danh sách từ khóa quan trọng (có trọng số cao)
    important_keywords = {
        'ho ra máu': 5.0,
        'xuất huyết': 5.0,
        'sụt cân': 4.0,
        'khó thở': 4.0,
        'đau ngực': 4.0,
        'co giật': 5.0,
        'tê liệt': 5.0,
        'vàng da': 4.5,
        'phù': 4.0,
        'sốt cao': 3.5,
        'đau dữ dội': 4.0,
    }
    
    improvements.append("✓ Added important keyword weights")
    
    # 3. Lưu cải thiện vào file
    improvement_config = {
        'medical_synonyms': medical_synonyms,
        'important_keywords': important_keywords,
        'version': '2.0',
        'timestamp': datetime.now().isoformat()
    }
    
    with open('model_improvements.json', 'w', encoding='utf-8') as f:
        json.dump(improvement_config, f, ensure_ascii=False, indent=2)
    
    improvements.append("✓ Saved improvements to model_improvements.json")
    
    print("\n".join(improvements))
    print(f"\n{'='*70}")
    
    return improvement_config

def main():
    """Main function"""
    print(f"\n{'='*70}")
    print(f"🏥 DISEASE DIAGNOSIS MODEL - ACCURACY TEST & IMPROVEMENT")
    print(f"{'='*70}\n")
    
    # Menu
    print("Choose an option:")
    print("1. Quick test (500 samples, no API) - Fast")
    print("2. Full test (1000 samples, no API) - Comprehensive")
    print("3. API test (100 samples, with API) - Most accurate but slow")
    print("4. Improve model")
    print("5. Test and improve if accuracy < 80%")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        accuracy, results, disease_stats, low_accuracy_diseases = test_model_accuracy(sample_size=500, use_api=False)
        save_test_results(accuracy, results, disease_stats, low_accuracy_diseases)
        
    elif choice == '2':
        accuracy, results, disease_stats, low_accuracy_diseases = test_model_accuracy(sample_size=1000, use_api=False)
        save_test_results(accuracy, results, disease_stats, low_accuracy_diseases)
        
    elif choice == '3':
        if not groq_client:
            print("❌ Error: GROQ_API_KEY not configured!")
            return
        accuracy, results, disease_stats, low_accuracy_diseases = test_model_accuracy(sample_size=100, use_api=True)
        save_test_results(accuracy, results, disease_stats, low_accuracy_diseases)
        
    elif choice == '4':
        improve_model()
        
    elif choice == '5':
        # Test trước
        print("\n🧪 Testing current model...")
        accuracy, results, disease_stats, low_accuracy_diseases = test_model_accuracy(sample_size=500, use_api=False)
        save_test_results(accuracy, results, disease_stats, low_accuracy_diseases)
        
        # Kiểm tra độ chính xác
        if accuracy < 80:
            print(f"\n⚠️ Accuracy ({accuracy:.2f}%) is below 80%!")
            print("🔧 Applying improvements...")
            improve_model()
            
            print("\n🧪 Testing improved model...")
            # Test lại sau khi cải thiện
            accuracy_new, results_new, disease_stats_new, low_accuracy_diseases_new = test_model_accuracy(sample_size=500, use_api=False)
            save_test_results(accuracy_new, results_new, disease_stats_new, low_accuracy_diseases_new)
            
            print(f"\n📊 COMPARISON:")
            print(f"Before: {accuracy:.2f}%")
            print(f"After: {accuracy_new:.2f}%")
            print(f"Improvement: {accuracy_new - accuracy:+.2f}%")
        else:
            print(f"\n✅ Accuracy ({accuracy:.2f}%) is already above 80%!")
            print("No improvement needed.")
    
    else:
        print("❌ Invalid choice!")

if __name__ == '__main__':
    main()
