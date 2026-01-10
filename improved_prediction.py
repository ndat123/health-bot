"""
Improved Disease Prediction với độ chính xác cao hơn
Sử dụng:
1. Enhanced TF-IDF với synonym matching
2. N-gram matching (2-gram, 3-gram)
3. Weighted keywords
4. Disease-specific pattern matching
"""
import pandas as pd
import json
import os
import re
import math
from collections import defaultdict, Counter
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Load data
df = pd.read_csv('ViMedical_Disease.csv', encoding='utf-8')
diseases = sorted(df['Disease'].unique().tolist())

# Build disease knowledge base với nhiều thông tin hơn
disease_symptoms = {}
disease_keywords = {}  # Keywords đặc trưng cho mỗi bệnh

for disease in diseases:
    disease_data = df[df['Disease'] == disease]['Question'].tolist()
    disease_symptoms[disease] = disease_data
    
    # Extract keywords cho mỗi bệnh
    all_text = " ".join(disease_data).lower()
    words = re.findall(r'\w+', all_text)
    disease_keywords[disease] = Counter(words)

print(f"✓ Loaded {len(diseases)} diseases with {len(df)} symptom samples")

# Medical synonyms (từ đồng nghĩa y tế)
MEDICAL_SYNONYMS = {
    'đau': ['nhức', 'buốt', 'đau đớn', 'đau nhức', 'đau rát'],
    'sốt': ['nóng', 'sốt cao', 'sốt nhẹ', 'ớn lạnh', 'rét run'],
    'ho': ['ho khan', 'ho có đờm', 'ho ra máu', 'ho nhiều', 'ho dai dẳng'],
    'buồn nôn': ['nôn', 'ói', 'nôn mửa', 'muốn nôn', 'ọe'],
    'chóng mặt': ['hoa mắt', 'choáng váng', 'ngất', 'mất thăng bằng', 'váng đầu'],
    'mệt': ['mệt mỏi', 'kiệt sức', 'yếu', 'mỏi', 'uể oải'],
    'khó thở': ['thở gấp', 'thở nhanh', 'ngạt thở', 'khó thở', 'thở dốc'],
    'đau đầu': ['nhức đầu', 'đau nửa đầu', 'đau đầu dữ dội', 'đau đầu nhiều'],
    'tiêu chảy': ['đi ngoài', 'phân lỏng', 'ỉa chảy', 'đi ngoài nhiều'],
    'táo bón': ['khó đi ngoài', 'đại tiện khó', 'bí đại tiện', 'khó đi cầu'],
    'ngứa': ['ngứa ngáy', 'ngứa rát', 'ngứa nhiều', 'ngứa da'],
    'sưng': ['phù', 'sưng to', 'sưng phù', 'phù nề'],
    'đỏ': ['đỏ da', 'đỏ bừng', 'ửng đỏ'],
    'khàn': ['khàn tiếng', 'khàn giọng', 'mất tiếng'],
    'chảy máu': ['xuất huyết', 'chảy máu cam', 'chảy máu chân răng'],
}

# Important keywords với trọng số
IMPORTANT_KEYWORDS = {
    'ho ra máu': 10.0,
    'xuất huyết': 10.0,
    'sụt cân': 8.0,
    'khó thở': 7.0,
    'đau ngực': 7.0,
    'co giật': 10.0,
    'tê liệt': 10.0,
    'vàng da': 9.0,
    'phù': 7.0,
    'sốt cao': 6.0,
    'đau dữ dội': 7.0,
    'mất ý thức': 10.0,
    'ngất xiu': 9.0,
    'đau bụng dữ dội': 8.0,
    'nôn ra máu': 10.0,
    'phân đen': 9.0,
    'tiểu ra máu': 9.0,
}

# Stopwords
STOPWORDS = {
    'tôi', 'của', 'có', 'bị', 'đang', 'là', 'và', 'này', 'thể', 'các', 'với',
    'một', 'được', 'hay', 'để', 'khi', 'như', 'thì', 'nào', 'làm', 'trong',
    'từ', 'cho', 'về', 'người', 'những', 'không', 'có thể', 'gì', 'hiện',
    'cảm', 'triệu', 'chứng', 'bệnh', 'nhân', 'đang', 'cảm', 'thấy'
}

def expand_with_synonyms(text):
    """Mở rộng text với các từ đồng nghĩa"""
    expanded_terms = [text]
    text_lower = text.lower()
    
    for key, synonyms in MEDICAL_SYNONYMS.items():
        if key in text_lower:
            for syn in synonyms:
                expanded_terms.append(text_lower.replace(key, syn))
        for syn in synonyms:
            if syn in text_lower:
                expanded_terms.append(text_lower.replace(syn, key))
    
    return expanded_terms

def extract_ngrams(text, n=2):
    """Extract n-grams từ text"""
    words = text.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        if len(ngram) > 5:  # Chỉ lấy ngram dài
            ngrams.append(ngram)
    return ngrams

def calculate_disease_score(symptoms_input, disease, disease_symptom_list):
    """
    Tính score cho một bệnh dựa trên nhiều yếu tố
    """
    score = 0
    
    # Normalize input
    symptoms_lower = symptoms_input.lower()
    
    # 1. Exact phrase matching (trọng số cao nhất)
    disease_text = " ".join(disease_symptom_list).lower()
    
    # Extract important phrases từ input
    for phrase, weight in IMPORTANT_KEYWORDS.items():
        if phrase in symptoms_lower:
            if phrase in disease_text:
                score += weight * 100  # Bonus rất cao cho important keywords
    
    # 2. N-gram matching (2-gram, 3-gram)
    input_bigrams = extract_ngrams(symptoms_input, 2)
    input_trigrams = extract_ngrams(symptoms_input, 3)
    
    for trigram in input_trigrams:
        if trigram in disease_text:
            score += 80  # Trigram match = rất tốt
    
    for bigram in input_bigrams:
        if bigram in disease_text:
            score += 40  # Bigram match = tốt
    
    # 3. Keyword matching với TF-IDF
    keywords = re.findall(r'\w+', symptoms_lower)
    keywords = [k for k in keywords if len(k) > 2 and k not in STOPWORDS]
    
    # Tính IDF cho keywords
    total_diseases = len(diseases)
    for keyword in keywords:
        # Đếm số bệnh có keyword này
        disease_count = sum(1 for d in diseases if keyword in " ".join(disease_symptoms[d]).lower())
        
        if disease_count > 0:
            idf = math.log(total_diseases / disease_count)
            
            # TF trong disease này
            tf = disease_text.count(keyword)
            
            if tf > 0:
                score += tf * idf * 15
    
    # 4. Synonym matching
    expanded_inputs = expand_with_synonyms(symptoms_input)
    for expanded in expanded_inputs:
        if expanded != symptoms_lower:  # Không tính lại input gốc
            # Đếm số từ khớp
            expanded_words = set(re.findall(r'\w+', expanded))
            disease_words = set(re.findall(r'\w+', disease_text))
            common_words = expanded_words & disease_words
            score += len(common_words) * 5
    
    # 5. Disease-specific keywords (từ thống kê)
    # Lấy top keywords của disease này
    if disease in disease_keywords:
        top_disease_keywords = [word for word, count in disease_keywords[disease].most_common(20)]
        for keyword in keywords:
            if keyword in top_disease_keywords:
                score += 20  # Bonus cho keyword đặc trưng của bệnh
    
    return score

def predict_disease_improved(symptoms_input, top_k=10):
    """
    Dự đoán bệnh với thuật toán cải tiến
    
    Returns:
        top_diseases: List of (disease, score) tuples
    """
    disease_scores = {}
    
    for disease in diseases:
        disease_symptom_list = disease_symptoms[disease]
        score = calculate_disease_score(symptoms_input, disease, disease_symptom_list)
        
        if score > 0:
            disease_scores[disease] = score
    
    # Sort by score
    top_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return top_diseases

def test_improved_model(sample_size=500):
    """Test model cải tiến"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING IMPROVED MODEL")
    print(f"{'='*70}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*70}\n")
    
    # Lấy mẫu ngẫu nhiên
    test_samples = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    correct = 0
    total = 0
    top3_correct = 0  # Đúng trong top 3
    top5_correct = 0  # Đúng trong top 5
    
    results = []
    disease_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'top3': 0, 'top5': 0})
    
    for idx, row in test_samples.iterrows():
        actual_disease = row['Disease']
        symptoms = row['Question']
        
        # Dự đoán
        top_predictions = predict_disease_improved(symptoms, top_k=10)
        
        if not top_predictions:
            total += 1
            continue
        
        predicted_disease = top_predictions[0][0]
        top3_diseases = [d for d, s in top_predictions[:3]]
        top5_diseases = [d for d, s in top_predictions[:5]]
        
        # Kiểm tra kết quả
        is_correct = (predicted_disease == actual_disease)
        is_top3 = (actual_disease in top3_diseases)
        is_top5 = (actual_disease in top5_diseases)
        
        if is_correct:
            correct += 1
        if is_top3:
            top3_correct += 1
        if is_top5:
            top5_correct += 1
        
        total += 1
        
        # Cập nhật thống kê
        disease_stats[actual_disease]['total'] += 1
        if is_correct:
            disease_stats[actual_disease]['correct'] += 1
        if is_top3:
            disease_stats[actual_disease]['top3'] += 1
        if is_top5:
            disease_stats[actual_disease]['top5'] += 1
        
        # Lưu kết quả
        results.append({
            'symptoms': symptoms[:100],
            'actual_disease': actual_disease,
            'predicted_disease': predicted_disease,
            'top3_diseases': top3_diseases,
            'is_correct': is_correct,
            'is_top3': is_top3,
            'is_top5': is_top5
        })
        
        # Progress
        if total % 50 == 0:
            current_accuracy = (correct / total) * 100
            current_top3 = (top3_correct / total) * 100
            print(f"Progress: {total}/{sample_size} - Accuracy: {current_accuracy:.2f}% | Top-3: {current_top3:.2f}%")
    
    # Kết quả
    accuracy = (correct / total) * 100 if total > 0 else 0
    top3_accuracy = (top3_correct / total) * 100 if total > 0 else 0
    top5_accuracy = (top5_correct / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"📊 IMPROVED MODEL RESULTS")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}% ({top3_correct}/{total})")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total})")
    print(f"{'='*70}\n")
    
    # Lưu kết quả
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs('test_results', exist_ok=True)
    
    summary = {
        'timestamp': timestamp,
        'model': 'improved',
        'top1_accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': total,
        'correct': correct,
        'top3_correct': top3_correct,
        'top5_correct': top5_correct
    }
    
    with open(f'test_results/improved_model_summary_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Results saved to test_results/improved_model_summary_{timestamp}.json\n")
    
    return accuracy, top3_accuracy, top5_accuracy, results

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"🚀 IMPROVED DISEASE PREDICTION MODEL")
    print(f"{'='*70}\n")
    
    # Test model
    accuracy, top3_acc, top5_acc, results = test_improved_model(sample_size=500)
    
    if accuracy >= 80:
        print(f"✅ SUCCESS! Accuracy ({accuracy:.2f}%) is above 80%!")
    else:
        print(f"⚠️ Accuracy ({accuracy:.2f}%) is still below 80%")
        print(f"💡 Consider using API-based prediction for better results")
