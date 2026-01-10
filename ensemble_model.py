"""
Ensemble Model - Kết hợp nhiều methods để đạt 75-80% accuracy
KHÔNG CẦN FINE-TUNE
"""
import sys
sys.path.insert(0, '.')

from web_app_gemini import find_relevant_diseases, df, groq_client, GROQ_MODEL
import pandas as pd
from collections import Counter, defaultdict
import re

print(f"\n{'='*70}")
print(f"🎯 ENSEMBLE MODEL - NO FINE-TUNING NEEDED")
print(f"{'='*70}\n")

# Medical rules
MEDICAL_RULES = {
    # Triệu chứng đặc trưng → Bệnh
    'ho ra máu': {
        'Ung Thư Phổi': 20,
        'Lao Phổi': 15,
        'Viêm Phổi Nặng': 10
    },
    'xuất huyết': {
        'Sốt Xuất Huyết': 20,
        'Xuất Huyết Não': 15
    },
    'sụt cân': {
        'Ung Thư': 15,
        'Lao': 10,
        'Đái Tháo Đường': 8
    },
    'vàng da': {
        'Viêm Gan': 20,
        'Sỏi Mật': 15
    },
    'co giật': {
        'Động Kinh': 20,
        'Viêm Màng Não': 15,
        'Sốt Cao': 10
    },
}

# Tổ hợp triệu chứng
SYMPTOM_COMBINATIONS = {
    ('sốt cao', 'đau đầu', 'buồn nôn'): {
        'Sốt Xuất Huyết': 15,
        'Viêm Màng Não': 12,
        'Cúm': 8
    },
    ('ho', 'sốt', 'đau ngực'): {
        'Viêm Phổi': 15,
        'Cúm': 10
    },
}

def apply_medical_rules(symptoms_input, disease_scores):
    """
    Áp dụng medical rules để boost/reduce scores
    """
    symptoms_lower = symptoms_input.lower()
    
    # Rule 1: Triệu chứng đặc trưng
    for symptom, disease_boosts in MEDICAL_RULES.items():
        if symptom in symptoms_lower:
            for disease, boost in disease_boosts.items():
                # Tìm bệnh tương tự trong list
                for d in disease_scores.keys():
                    if disease.lower() in d.lower() or d.lower() in disease.lower():
                        disease_scores[d] += boost
                        break
    
    # Rule 2: Tổ hợp triệu chứng
    for symptom_combo, disease_boosts in SYMPTOM_COMBINATIONS.items():
        if all(s in symptoms_lower for s in symptom_combo):
            for disease, boost in disease_boosts.items():
                for d in disease_scores.keys():
                    if disease.lower() in d.lower() or d.lower() in disease.lower():
                        disease_scores[d] += boost
                        break
    
    return disease_scores

def ensemble_predict(symptoms_input, use_api=False):
    """
    Ensemble prediction kết hợp:
    1. Enhanced TF-IDF
    2. Medical Rules
    3. API (optional)
    """
    # Method 1: Enhanced TF-IDF
    _, top_diseases_list, _, top_diseases_with_scores = find_relevant_diseases(symptoms_input, top_k=10)
    
    if not top_diseases_with_scores:
        return None, 0
    
    # Convert to dict
    disease_scores = {disease: score for disease, score in top_diseases_with_scores}
    
    # Method 2: Apply Medical Rules
    disease_scores = apply_medical_rules(symptoms_input, disease_scores)
    
    # Method 3: API Refinement (optional)
    if use_api and groq_client:
        try:
            # Chỉ dùng API nếu top 2 scores gần nhau (không chắc chắn)
            sorted_scores = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_scores) >= 2:
                top1_score = sorted_scores[0][1]
                top2_score = sorted_scores[1][1]
                score_diff = (top1_score - top2_score) / top1_score if top1_score > 0 else 0
                
                if score_diff < 0.15:  # Gần nhau → không chắc → dùng API
                    # Lấy top 3 để API chọn
                    top3_diseases = [d for d, s in sorted_scores[:3]]
                    context = "\n".join([f"{i+1}. {d}" for i, d in enumerate(top3_diseases)])
                    
                    prompt = f"""Bạn là bác sĩ chuyên khoa với 20 năm kinh nghiệm.

DATABASE: 603 bệnh tiếng Việt đã được phân tích.

TOP 3 BỆNH KHẢ NĂNG CAO NHẤT (từ database):
{context}

TRIỆU CHỨNG CỦA BỆNH NHÂN:
"{symptoms_input}"

NHIỆM VỤ:
Chọn 1 bệnh phù hợp NHẤT từ top 3 trên.
Chỉ trả lời TÊN BỆNH, không giải thích.

LƯU Ý:
- Phải chọn từ top 3 trên
- Tên bệnh phải CHÍNH XÁC như trong danh sách"""

                    response = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "Bạn là bác sĩ AI, chỉ trả lời tên bệnh"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=50,
                    )
                    
                    api_prediction = response.choices[0].message.content.strip()
                    
                    # Validate
                    if api_prediction in top3_diseases:
                        # Boost API prediction
                        disease_scores[api_prediction] += 100
        
        except Exception as e:
            pass  # Nếu API lỗi, dùng TF-IDF + Rules
    
    # Final ranking
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_diseases:
        predicted_disease = sorted_diseases[0][0]
        confidence = min(100, int(sorted_diseases[0][1] / max(1, sorted_diseases[0][1]) * 100))
        return predicted_disease, confidence
    
    return None, 0

def test_ensemble_model(sample_size=200, use_api=False):
    """Test ensemble model"""
    print(f"Testing ensemble model...")
    print(f"Sample size: {sample_size}")
    print(f"Use API: {use_api}")
    print(f"{'='*70}\n")
    
    # Test samples
    test_samples = df.sample(n=sample_size, random_state=42)
    
    correct = 0
    total = 0
    
    for idx, row in test_samples.iterrows():
        actual_disease = row['Disease']
        symptoms = row['Question']
        
        # Predict
        predicted_disease, confidence = ensemble_predict(symptoms, use_api=use_api)
        
        if predicted_disease == actual_disease:
            correct += 1
        
        total += 1
        
        # Progress
        if total % 50 == 0:
            acc = (correct / total) * 100
            print(f"Progress: {total}/{sample_size} - Accuracy: {acc:.1f}%")
    
    # Results
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"📊 ENSEMBLE MODEL RESULTS")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")
    
    # Comparison
    print(f"📈 COMPARISON:")
    print(f"TF-IDF only: 52.5%")
    print(f"Ensemble (no API): {accuracy:.1f}%")
    print(f"Improvement: {accuracy - 52.5:+.1f}%")
    
    if accuracy >= 75:
        print(f"\n✅ SUCCESS! Accuracy >= 75% without fine-tuning!")
    elif accuracy >= 70:
        print(f"\n✅ GOOD! Accuracy >= 70%")
        print(f"💡 Try with API to reach 75%+")
    else:
        print(f"\n⚠️ Accuracy < 70%")
        print(f"💡 Try with API or add more rules")
    
    return accuracy

if __name__ == '__main__':
    print("Choose test mode:")
    print("1. Fast test (200 samples, no API)")
    print("2. API test (100 samples, with API)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == '1':
        accuracy = test_ensemble_model(sample_size=200, use_api=False)
    elif choice == '2':
        if not groq_client:
            print("❌ Error: GROQ_API_KEY not configured!")
        else:
            accuracy = test_ensemble_model(sample_size=100, use_api=True)
    else:
        print("Invalid choice!")
