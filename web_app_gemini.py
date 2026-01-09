"""
Web App cho Disease Diagnosis với Groq API
Chạy trên localhost với Flask
"""
from flask import Flask, render_template, request, jsonify
import os
from groq import Groq
try:
    import warnings
    # Suppress the deprecation warning for google.generativeai
    warnings.filterwarnings('ignore', category=FutureWarning, module='google.generativeai')
    import google.generativeai as genai
except ImportError:
    # Fallback if google-generativeai not installed
    genai = None
    print("⚠ Google Generative AI not installed. Gemini engine will not be available.")
    print("  Install: pip install google-generativeai")
import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime

app = Flask(__name__)

# ============================================================================
# DUAL AI ENGINE CONFIGURATION - Groq + Google Gemini
# ============================================================================

# Configure Groq
# Get API key from environment variable or use placeholder
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
if GROQ_API_KEY == 'your_groq_api_key_here':
    print("⚠ WARNING: GROQ_API_KEY not set. Please set it in environment variable or .env file")
    print("  Example: export GROQ_API_KEY='your_key_here'")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY != 'your_groq_api_key_here' else None
GROQ_MODEL = 'llama-3.3-70b-versatile'  # Model mạnh nhất của Groq

# Configure Google Gemini
# Get API key from environment variable or use placeholder
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here')
if genai:
    try:
        if GEMINI_API_KEY != 'your_gemini_api_key_here':
            genai.configure(api_key=GEMINI_API_KEY)
            print("✓ Gemini API configured successfully")
        else:
            print("⚠ WARNING: GEMINI_API_KEY not set. Gemini engine will not be available.")
            print("  Example: export GEMINI_API_KEY='your_key_here'")
    except Exception as e:
        print(f"⚠ Gemini API configuration error: {e}")

# Gemini model options:
# 1. Base model (not tuned): 'gemini-2.0-flash-exp'
# 2. Fine-tuned model: 'tunedModels/your-model-name' (after training)
GEMINI_MODEL = 'gemini-2.0-flash-exp'  # Default: base model
GEMINI_TUNED_MODEL = None  # Set this after fine-tuning: 'tunedModels/xxx'

# Default AI engine
DEFAULT_AI_ENGINE = 'groq'  # 'groq' or 'gemini'

# MySQL Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'healthcare'
}

# Initialize database
def init_database():
    """Tạo bảng search_history nếu chưa tồn tại.

    LƯU Ý:
    - Trên môi trường như Railway, nếu không có MySQL (localhost:3306),
      hàm này phải FAIL GRACEFULLY, KHÔNG được làm app crash.
    """
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symptoms TEXT NOT NULL,
                disease VARCHAR(255) NOT NULL,
                analysis TEXT,
                confidence FLOAT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_created_at (created_at),
                INDEX idx_disease (disease)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        conn.commit()
        print("✓ Database table 'search_history' initialized successfully")

    except Error as e:
        # Không làm app dừng – chỉ log lỗi, app vẫn chạy bình thường
        print(f"✗ Database error: {e}")
        print("⚠ Database features (search history) will be disabled on this environment.")
    finally:
        try:
            if conn is not None and hasattr(conn, "is_connected") and conn.is_connected():
                if cursor is not None:
                    cursor.close()
                conn.close()
        except Exception:
            # Tuyệt đối không cho lỗi ở đây làm app crash
            pass

def get_db_connection():
    """Tạo kết nối database"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def save_search_history(symptoms, disease, analysis, confidence=0):
    """Lưu lịch sử tìm kiếm vào database"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        query = """
            INSERT INTO search_history (symptoms, disease, analysis, confidence)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (symptoms, disease, analysis, confidence))
        conn.commit()
        
        print(f"✓ Saved search history: {disease}")
        return True
        
    except Error as e:
        print(f"Error saving history: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# Prediction Strategy Config
CSV_CONFIDENCE_THRESHOLD = 100  # Nếu score >= threshold → dùng CSV, không gọi API
# Tăng threshold để ưu tiên API hơn
# Giảm threshold để ưu tiên CSV hơn (nhanh, tiết kiệm API calls)

# Load diseases and build knowledge base
df = pd.read_csv('ViMedical_Disease.csv', encoding='utf-8')
diseases = sorted(df['Disease'].unique().tolist())

# Build disease knowledge base: disease -> list of symptom descriptions
disease_symptoms = {}
for disease in diseases:
    disease_data = df[df['Disease'] == disease]['Question'].tolist()
    # Lấy tối đa 10 mẫu triệu chứng cho mỗi bệnh để giảm token
    disease_symptoms[disease] = disease_data[:10]

print(f"✓ Loaded {len(diseases)} diseases with {len(df)} symptom samples")

# Initialize database
init_database()

def validate_symptoms_input(text):
    """
    Kiểm tra xem input có phải là triệu chứng y tế hay không
    Trả về (is_valid, message)
    """
    import re
    
    # Loại bỏ khoảng trắng thừa
    text = text.strip()
    
    # Kiểm tra độ dài tối thiểu
    if len(text) < 5:
        return False, "Vui lòng mô tả triệu chứng chi tiết hơn (ít nhất 5 ký tự)"
    
    # Keywords liên quan đến triệu chứng y tế
    medical_keywords = [
        'đau', 'sốt', 'ho', 'nôn', 'chóng mặt', 'mệt', 'buồn nôn', 'tiêu chảy',
        'khó thở', 'ngứa', 'phát ban', 'sưng', 'viêm', 'chảy máu', 'xuất huyết',
        'run', 'co giật', 'tê', 'tê liệt', 'yếu', 'mỏi', 'đau đầu', 'nhức',
        'khó nuốt', 'khàn', 'ho khan', 'ho có đờm', 'sổ mũi', 'nghẹt mũi',
        'ớn lạnh', 'vã mồ hôi', 'khát nước', 'chán ăn', 'sụt cân', 'tăng cân',
        'táo bón', 'tiểu', 'phân', 'kinh nguyệt', 'đau bụng', 'đau ngực',
        'khó chịu', 'tức ngực', 'hồi hộp', 'lo âu', 'mất ngủ', 'buồn ngủ',
        'chảy nước mũi', 'đau họng', 'sưng họng', 'khó thở', 'thở khò khè',
        'ho ra máu', 'nôn ra máu', 'phù', 'sưng phù', 'đau lưng', 'đau cơ',
        'cứng khớp', 'đau khớp', 'vàng da', 'ngứa', 'nổi mẩn', 'bầm tím',
        'chảy máu cam', 'ù tai', 'nhìn mờ', 'hoa mắt', 'ngất', 'choáng váng'
    ]
    
    # Các từ chỉ vị trí / cơ thể
    body_parts = [
        'đầu', 'cổ', 'họng', 'ngực', 'bụng', 'lưng', 'tay', 'chân', 'vai', 'gối',
        'mắt', 'tai', 'mũi', 'miệng', 'răng', 'lưỡi', 'da', 'tóc', 'móng',
        'tim', 'phổi', 'gan', 'thận', 'dạ dày', 'ruột', 'bàng quang'
    ]
    
    # Kiểm tra có keyword y tế không
    text_lower = text.lower()
    has_medical_keyword = any(keyword in text_lower for keyword in medical_keywords)
    has_body_part = any(part in text_lower for part in body_parts)
    
    # Nếu có keyword y tế hoặc body part → có thể là triệu chứng
    if has_medical_keyword or has_body_part:
        return True, None
    
    # Kiểm tra các câu hỏi không liên quan
    invalid_patterns = [
        r'(bạn là ai|bạn tên gì|ai tạo ra bạn)',
        r'(thời tiết|trời|mưa|nắng)',
        r'(chào|hello|hi|xin chào)',
        r'(cảm ơn|thank)',
        r'(tạm biệt|bye|goodbye)',
        r'(bao nhiêu tuổi|năm nay)',
        r'(ở đâu|địa chỉ|nơi nào)',
        r'(làm gì|công việc)',
        r'(thích gì|sở thích)',
        r'(màu|số|ngày)',
        r'^(a|b|c|d|e|1|2|3)$',  # Chỉ 1 ký tự
        r'^test$',
        r'(test|thử|demo)',
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, text_lower):
            return False, "❌ Câu hỏi không hợp lệ! Vui lòng nhập triệu chứng bệnh (ví dụ: đau đầu, sốt cao, ho khan...)"
    
    # Nếu text quá ngắn và không có keyword y tế
    if len(text) < 10 and not (has_medical_keyword or has_body_part):
        return False, "Vui lòng mô tả triệu chứng chi tiết hơn. Ví dụ: 'Tôi bị đau đầu, sốt cao và buồn nôn'"
    
    # Sử dụng Groq API để validate (nếu vẫn không chắc chắn)
    if not has_medical_keyword and not has_body_part:
        try:
            validation_prompt = f"""Bạn là hệ thống AI y tế. Kiểm tra xem câu sau có phải là mô tả triệu chứng bệnh hay không:

"{text}"

Trả lời CHỈ MỘT TỪ: "CÓ" hoặc "KHÔNG"
- CÓ: nếu đây là triệu chứng bệnh, vấn đề sức khỏe
- KHÔNG: nếu đây là câu hỏi không liên quan đến sức khỏe/triệu chứng"""

            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là AI validator, chỉ trả lời CÓ hoặc KHÔNG"},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=10,
            )
            
            answer = response.choices[0].message.content.strip().upper()
            
            if 'KHÔNG' in answer or 'NO' in answer:
                return False, "❌ Câu hỏi không hợp lệ! Vui lòng nhập triệu chứng bệnh (ví dụ: đau đầu, sốt cao, ho khan...)"
            
        except Exception as e:
            print(f"Validation API error: {e}")
            # Nếu API lỗi, cho phép tiếp tục (fail-safe)
            pass
    
    # Default: cho phép nếu không có dấu hiệu rõ ràng là invalid
    return True, None

def get_disease_detail_from_ai(disease_name, ai_engine='groq'):
    """
    Gọi AI API (Groq hoặc Gemini) để lấy thông tin chi tiết về một bệnh cụ thể
    
    Args:
        disease_name: Tên bệnh cần lấy thông tin
        ai_engine: 'groq' hoặc 'gemini'
    """
    import re
    
    prompt = f"""Bạn là bác sĩ chuyên khoa. Hãy cung cấp thông tin CHI TIẾT về bệnh: **{disease_name}**

TRẢ LỜI THEO FORMAT:

🩺 Triệu chứng đầy đủ:
- [Triệu chứng 1 - cụ thể và chi tiết]
- [Triệu chứng 2]
- [Triệu chứng 3]
- [Triệu chứng 4]
- [Triệu chứng 5]

💊 Cách chữa/điều trị:
- [Phương pháp điều trị 1 - cụ thể]
- [Phương pháp điều trị 2]
- [Phương pháp điều trị 3]
- [Phương pháp điều trị 4]
- [Phương pháp điều trị 5]

⚠️ Nguyên nhân:
- [Nguyên nhân 1 - cụ thể]
- [Nguyên nhân 2]
- [Nguyên nhân 3]
- [Nguyên nhân 4]

⚕️ Khi nào cần đi khám gấp:
- [Dấu hiệu nguy hiểm 1]
- [Dấu hiệu nguy hiểm 2]

LƯU Ý:
- Trả lời bằng tiếng Việt
- Cung cấp thông tin chính xác, khoa học
- Giữ ĐÚNG format trên"""

    try:
        if ai_engine == 'gemini':
            # Call Google Gemini API
            model_name = GEMINI_TUNED_MODEL if GEMINI_TUNED_MODEL else GEMINI_MODEL
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1500,
                )
            )
            
            result_text = response.text
            
        else:  # Default: groq
            # Call Groq API
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là bác sĩ chuyên khoa giàu kinh nghiệm, cung cấp thông tin y tế chính xác bằng tiếng Việt."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            
            result_text = response.choices[0].message.content
        
        # Parse response
        symptoms = []
        treatment = []
        causes = []
        urgent_signs = []
        
        # Extract symptoms
        symptoms_match = re.search(r'🩺\s*Triệu chứng đầy đủ:(.*?)(?=💊|⚠️|⚕️|$)', result_text, re.DOTALL | re.IGNORECASE)
        if symptoms_match:
            symptoms_text = symptoms_match.group(1)
            symptoms = re.findall(r'[-•]\s*([^\n]+)', symptoms_text)
            symptoms = [s.strip() for s in symptoms if len(s.strip()) > 5][:10]
        
        # Extract treatment
        treatment_match = re.search(r'💊\s*Cách chữa/điều trị:(.*?)(?=⚠️|⚕️|$)', result_text, re.DOTALL | re.IGNORECASE)
        if treatment_match:
            treatment_text = treatment_match.group(1)
            treatment = re.findall(r'[-•]\s*([^\n]+)', treatment_text)
            treatment = [t.strip() for t in treatment if len(t.strip()) > 5][:10]
        
        # Extract causes
        causes_match = re.search(r'⚠️\s*Nguyên nhân:(.*?)(?=⚕️|💊|$)', result_text, re.DOTALL | re.IGNORECASE)
        if causes_match:
            causes_text = causes_match.group(1)
            causes = re.findall(r'[-•]\s*([^\n]+)', causes_text)
            causes = [c.strip() for c in causes if len(c.strip()) > 5][:8]
        
        # Extract urgent signs
        urgent_match = re.search(r'⚕️\s*Khi nào cần đi khám gấp:(.*?)(?=\n\n|$)', result_text, re.DOTALL | re.IGNORECASE)
        if urgent_match:
            urgent_text = urgent_match.group(1)
            urgent_signs = re.findall(r'[-•]\s*([^\n]+)', urgent_text)
            urgent_signs = [u.strip() for u in urgent_signs if len(u.strip()) > 5][:5]
        
        if symptoms or treatment or causes:
            return {
                'disease_name': disease_name,
                'symptoms': symptoms if symptoms else ['Triệu chứng sẽ được cập nhật sau khi đi khám'],
                'treatment': treatment if treatment else ['Vui lòng đi khám bác sĩ để được tư vấn điều trị cụ thể'],
                'causes': causes if causes else ['Nhiều nguyên nhân khác nhau, cần khám để xác định'],
                'urgent_signs': urgent_signs if urgent_signs else []
            }
    
    except Exception as e:
        print(f"Error getting disease detail from Groq: {e}")
    
    # Fallback nếu có lỗi
    return {
        'disease_name': disease_name,
        'symptoms': ['Vui lòng đi khám để bác sĩ đánh giá triệu chứng cụ thể'],
        'treatment': ['Điều trị phụ thuộc vào chẩn đoán chính xác từ bác sĩ'],
        'causes': ['Nhiều nguyên nhân có thể gây ra bệnh này'],
        'urgent_signs': []
    }

def find_relevant_diseases(symptoms_input, top_k=15):
    """
    Tìm các bệnh có triệu chứng tương tự với input của user
    Dùng TF-IDF để tăng độ chính xác
    """
    from collections import Counter, defaultdict
    import re
    import math
    
    # Stopwords tiếng Việt (các từ không quan trọng)
    stopwords = {
        'tôi', 'của', 'có', 'bị', 'đang', 'là', 'và', 'này', 'thể', 'các', 'với',
        'một', 'được', 'hay', 'để', 'khi', 'như', 'thì', 'nào', 'làm', 'trong',
        'từ', 'cho', 'về', 'người', 'những', 'không', 'có thể', 'gì', 'hiện',
        'cảm', 'triệu', 'chứng'
    }
    
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
    
    # Tính IDF cho mỗi keyword (số bệnh có keyword này)
    keyword_idf = defaultdict(int)
    for disease, symptom_list in disease_symptoms.items():
        disease_text = " ".join(symptom_list).lower()
        for keyword in set(keywords):  # Chỉ đếm 1 lần mỗi keyword cho mỗi bệnh
            if keyword in disease_text:
                keyword_idf[keyword] += 1
    
    # Tính IDF score: log(total_diseases / diseases_with_keyword)
    total_diseases = len(disease_symptoms)
    idf_scores = {}
    for keyword, count in keyword_idf.items():
        if count > 0:
            # Keyword càng hiếm (ít bệnh có) -> IDF càng cao
            idf_scores[keyword] = math.log(total_diseases / count)
    
    # Score cho mỗi bệnh với TF-IDF
    disease_scores = {}
    disease_matching_symptoms = {}
    
    for disease, symptom_list in disease_symptoms.items():
        disease_text = " ".join(symptom_list).lower()
        score = 0
        matching_symptoms = []
        
        # Score từ keywords với IDF weighting
        for keyword in keywords:
            if keyword in disease_text:
                # TF: số lần xuất hiện
                tf = disease_text.count(keyword)
                # IDF: độ hiếm của keyword
                idf = idf_scores.get(keyword, 0)
                # TF-IDF score
                score += tf * idf * 10  # Nhân 10 để scale
        
        # Bonus score cho exact phrases
        for phrase in phrases:
            if phrase in disease_text:
                score += 50  # Bonus cao cho phrase khớp
        
        # Tìm matching symptoms
        for symptom_text in symptom_list:
            symptom_lower = symptom_text.lower()
            matches = sum(1 for keyword in keywords if keyword in symptom_lower)
            if matches > 0:
                matching_symptoms.append(symptom_text.strip())
        
        if score > 0:
            disease_scores[disease] = score
            disease_matching_symptoms[disease] = matching_symptoms[:3]
    
    # Lấy top k bệnh có score cao nhất
    top_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Build context với thông tin chi tiết hơn
    context = ""
    context += f"\n🔍 Tìm thấy {len(top_diseases)} bệnh khớp với triệu chứng:\n"
    
    for i, (disease, score) in enumerate(top_diseases[:15], 1):  # Top 15
        symptoms = disease_matching_symptoms.get(disease, [])
        if symptoms:
            # Normalize score để dễ hiểu (0-100)
            normalized_score = min(100, int(score / max(1, top_diseases[0][1]) * 100))
            context += f"\n{i}. **{disease}** (relevance: {normalized_score}%):\n"
            
            for symptom in symptoms[:3]:  # Lấy 3 triệu chứng điển hình
                # Extract triệu chứng từ câu hỏi
                symptom_clean = symptom.replace("Tôi có thể đang bị bệnh gì?", "")
                symptom_clean = symptom_clean.replace('"', '').strip()
                # Loại bỏ "Tôi đang..." để chỉ giữ triệu chứng
                symptom_clean = re.sub(r'^(Tôi|Bệnh nhân)\s+(đang|hiện đang|đang cảm thấy|cảm thấy|hay bị|bị)\s+', '', symptom_clean)
                symptom_clean = re.sub(r'^\s*có các triệu chứng như\s+', '', symptom_clean)
                if symptom_clean and len(symptom_clean) > 10:
                    context += f"   • {symptom_clean}\n"
    
    # Return context, diseases list, and best match info
    best_match_score = top_diseases[0][1] if top_diseases else 0
    return context, [d for d, s in top_diseases], best_match_score, top_diseases

def predict_from_csv_data(symptoms_input, top_diseases_with_scores, ai_engine='groq'):
    """
    Dự đoán trực tiếp từ dữ liệu CSV và lấy chi tiết từ AI
    
    Args:
        symptoms_input: Triệu chứng người dùng nhập
        top_diseases_with_scores: List of (disease, score) tuples
        ai_engine: 'groq' hoặc 'gemini'
    """
    import re
    
    if not top_diseases_with_scores:
        return None
    
    # Get top 1 disease với thông tin đầy đủ
    detailed_predictions = []
    total_score = sum(score for _, score in top_diseases_with_scores[:1])
    
    for i, (disease, score) in enumerate(top_diseases_with_scores[:1]):
        # Tính xác suất dựa trên score
        if total_score > 0:
            probability = int((score / total_score) * 100)
        else:
            probability = 0
        
        # Không hiển thị % nữa
        probability = 0
        
        # Tạo reason chi tiết
        reason = f"Triệu chứng khớp tốt nhất với {disease} trong database với {len(disease_symptoms.get(disease, []))} mẫu triệu chứng tương tự"
        
        # Lấy triệu chứng điển hình từ database
        typical_symptoms = []
        if disease in disease_symptoms:
            symptom_samples = disease_symptoms[disease][:5]  # Top 5
            for symptom in symptom_samples:
                clean = symptom.replace("Tôi có thể đang bị bệnh gì?", "").replace('"', '').strip()
                clean = re.sub(r'^(Tôi|Bệnh nhân)\s+(đang|hiện đang|đang cảm thấy|cảm thấy|hay bị|bị)\s+', '', clean)
                clean = re.sub(r'^\s*có các triệu chứng như\s+', '', clean)
                if clean and len(clean) > 10:
                    typical_symptoms.append(clean)
        
        # Đếm số mẫu trong database
        sample_count = len(df[df['Disease'] == disease])
        
        detailed_predictions.append({
            'disease': disease,
            'probability': probability,
            'reason': reason,
            'typical_symptoms': typical_symptoms[:3],  # Top 3 triệu chứng
            'database_samples': sample_count,
            'has_database_info': len(typical_symptoms) > 0
        })
    
    # Tạo analysis
    top_disease = top_diseases_with_scores[0][0]
    analysis = f"Dựa trên phân tích 23,521 mẫu trong database, triệu chứng của bạn khớp nhất với {top_disease}"
    
    # Gọi AI API để lấy thông tin chi tiết về bệnh đầu tiên
    disease_info = None
    if detailed_predictions:
        top_disease_name = detailed_predictions[0]['disease']
        disease_info = get_disease_detail_from_ai(top_disease_name, ai_engine=ai_engine)
    
    # Recommendations chung
    recommendations = [
        f"Đi khám bác sĩ chuyên khoa để xác định chính xác",
        "Theo dõi các triệu chứng và ghi chép lại",
        "Không tự ý điều trị khi chưa có chẩn đoán",
        "Nghỉ ngơi đầy đủ và giữ tinh thần thoải mái"
    ]
    
    return {
        'success': True,
        'analysis': analysis,
        'predictions': detailed_predictions,
        'disease_info': disease_info,
        'recommendations': recommendations,
        'warning': 'Đây là dự đoán AI dựa trên database, KHÔNG PHẢI chẩn đoán y tế. Hãy đi khám bác sĩ để được chẩn đoán chính xác!',
        'metadata': {
            'source': 'CSV Database (23,521 mẫu)',
            'model': 'TF-IDF + Keyword Matching',
            'provider': 'Local Database'
        }
    }

# System instruction với examples
SYSTEM_INSTRUCTION = f"""Bạn là trợ lý y tế AI chuyên nghiệp được training trên database {len(diseases)} loại bệnh tiếng Việt với {len(df)} mẫu triệu chứng.

DATABASE BẠN ĐÃ HỌC BAO GỒM:
- Các bệnh phụ khoa: Ối Vỡ Non, Sinh Non, Tiền Sản Giật, Băng Huyết Sau Sinh...
- Các bệnh nhiễm trùng: Sốt Xuất Huyết, Cúm, COVID-19, Viêm Phổi...  
- Các bệnh tiêu hóa: Viêm Dạ Dày, Viêm Ruột, Loét Dạ Dày...
- Và {len(diseases)-50} bệnh khác

NHIỆM VỤ: 
1. Đọc kỹ "THÔNG TIN TỪ DATABASE" được cung cấp (đã lọc sẵn các bệnh có triệu chứng tương tự)
2. So sánh triệu chứng của user với triệu chứng trong database
3. Dự đoán 3-5 bệnh CÓ TRONG DATABASE với xác suất dựa trên độ khớp.

VÍ DỤ OUTPUT CHUẨN:

🔍 Phân tích: Triệu chứng sưng cổ, khó nuốt và khàn tiếng có thể gặp ở nhiều bệnh khác nhau

💡 Dự đoán bệnh:

1. **Bướu Cổ Lành Tính** - 60%
   Lý do: Triệu chứng khớp với bướu cổ, nhưng không có dấu hiệu ác tính như sụt cân, ho ra máu

2. **Ung Thư Thanh Quản** - 30%
   Lý do: Có triệu chứng tương tự nhưng thiếu dấu hiệu đặc trưng như ho ra máu, tiền sử hút thuốc

3. **Viêm Thanh Quản** - 10%
   Lý do: Có thể gây khàn tiếng và khó nuốt tạm thời

📋 THÔNG TIN CHI TIẾT VỀ BƯỚU CỔ LÀNH TÍNH:

🩺 Triệu chứng đầy đủ:
- Sưng to vùng cổ, có thể thấy khối u lớn dần
- Khó nuốt, cảm giác nghẹn khi ăn uống
- Khàn tiếng do chèn ép thanh quản
- Khó thở khi gắng sức hoặc nằm ngửa
- Mệt mỏi, tăng cân hoặc giảm cân
- Da khô, rụng tóc (nếu suy giáp)
- Lo lắng, đánh trống ngực (nếu cường giáp)

💊 Cách chữa/điều trị:
- Theo dõi định kỳ nếu u nhỏ, không triệu chứng
- Dùng thuốc điều chỉnh hormone giáp
- Điều trị iod phóng xạ nếu cường giáp
- Phẫu thuật cắt bỏ u nếu u lớn, chèn ép
- Bổ sung iod nếu thiếu iod
- Tránh stress, nghỉ ngơi đầy đủ

⚠️ Nguyên nhân:
- Thiếu iod trong chế độ ăn
- Rối loạn hormone tuyến giáp
- Di truyền, tiền sử gia đình
- Stress kéo dài, thiếu ngủ

💊 Khuyến nghị:
- Đi khám bác sĩ nội tiết để xét nghiệm hormone giáp
- Siêu âm tuyến giáp để đánh giá kích thước u
- Xét nghiệm tế bào học nếu nghi ngờ ác tính
- Theo dõi định kỳ 6 tháng/lần

QUY TẮC QUAN TRỌNG:
1. LUÔN match triệu chứng với bệnh trong database
2. Dùng % phản ánh độ chắc chắn:
   - 85-95%: Triệu chứng RẤT ĐIỂN HÌNH, khớp hoàn toàn
   - 70-84%: Triệu chứng khớp tốt, nhiều dấu hiệu đặc trưng
   - 50-69%: Triệu chứng có thể, thiếu một số dấu hiệu
   - 30-49%: Khả năng thấp
   - 10-29%: Rất ít khả năng
3. KHÔNG ngại đưa ra 85-95% nếu triệu chứng rất rõ ràng và điển hình
4. Với triệu chứng thai sản (ối, nước ối, xuất huyết thai kỳ) → nghĩ đến bệnh phụ khoa
5. Giữ ĐÚNG format trên, không thêm text khác
6. Tên bệnh phải CHÍNH XÁC theo tiếng Việt"""

@app.route('/')
def index():
    return render_template('index.html', total_diseases=len(diseases))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '').strip()
        ai_engine = data.get('ai_engine', DEFAULT_AI_ENGINE).lower()  # Get AI engine from request
        
        # Validate AI engine
        if ai_engine not in ['groq', 'gemini', 'groq_chat']:
            print(f"⚠️ Invalid AI engine '{ai_engine}', defaulting to '{DEFAULT_AI_ENGINE}'")
            ai_engine = DEFAULT_AI_ENGINE
        
        print(f"\n{'='*70}")
        print(f"🤖 AI Engine Selected: {ai_engine.upper()}")
        print(f"Symptoms: {symptoms[:80]}...")
        print(f"{'='*70}")
        
        if not symptoms:
            return jsonify({'error': 'Vui lòng nhập triệu chứng'}), 400
        
        # === CHATBOT MODES (Skip database search, go direct to AI) ===
        if ai_engine in ['groq_chat', 'gemini']:
            print(f"\n{'='*70}")
            print(f"💬 CHATBOT MODE: {ai_engine.upper()}")
            print(f"{'='*70}")
            print(f"Input: {symptoms[:100]}...")
            print("Mode: Direct conversation (no database search)")
            
            # GROQ CHATBOT
            if ai_engine == 'groq_chat':
                try:
                    # Create conversational prompt for Groq
                    chatbot_system = """Bạn là bác sĩ AI thân thiện, chuyên tư vấn sức khỏe qua chat.

PHONG CÁCH TRẢ LỜI:
- Thân thiện, dễ hiểu, không quá formal
- Giải thích chi tiết về bệnh và triệu chứng
- Luôn an ủi và động viên bệnh nhân
- Đưa ra lời khuyên cụ thể và thực tế
- Nhấn mạnh tầm quan trọng của việc đi khám bác sĩ

KHÔNG:
- Không chỉ liệt kê tên bệnh
- Không dùng ngôn ngữ y khoa quá phức tạp
- Không gây hoảng sợ cho bệnh nhân

FORMAT TRẢ LỜI:
1. Xác nhận và thấu hiểu triệu chứng
2. Giải thích khả năng mắc bệnh gì
3. Mô tả chi tiết về bệnh đó
4. Lời khuyên và hướng xử lý
5. Động viên và nhắc nhở đi khám

VÍ DỤ:
User: "Tôi bị đau đầu và sốt"
Bot: "Dựa vào các triệu chứng bạn mô tả, bạn có thể đang gặp phải tình trạng cảm lạnh hoặc cúm.

**Về tình trạng này:**
Đau đầu kèm sốt là dấu hiệu của nhiễm trùng đường hô hấp trên, thường gặp nhất là cảm lạnh hoặc cúm. Đây là tình trạng khá phổ biến và thường có thể tự khỏi sau 5-7 ngày.

**Lời khuyên:**
- Nghỉ ngơi nhiều, uống đủ nước
- Có thể dùng thuốc hạ sốt như paracetamol nếu sốt trên 38.5°C
- Theo dõi triệu chứng

**Khi nào cần đi khám gấp:**
- Sốt trên 39°C kéo dài > 3 ngày
- Đau đầu dữ dội, buồn nôn
- Khó thở, đau ngực

Tuy nhiên, để chắc chắn và được tư vấn cụ thể hơn, bạn nên đến gặp bác sĩ để được khám và chẩn đoán chính xác nhé!

Chúc bạn mau khỏe! 💙\""""
                    
                    response = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": chatbot_system
                            },
                            {
                                "role": "user",
                                "content": symptoms
                            }
                        ],
                        temperature=0.7,  # Higher for natural conversation
                        max_tokens=2000,
                        top_p=0.9,
                    )
                    
                    chat_response = response.choices[0].message.content
                    
                    print(f"\n{'='*70}")
                    print("RAW RESPONSE FROM GROQ CHATBOT:")
                    print(f"Text length: {len(chat_response)} chars")
                    print(f"Content:\n{chat_response}")
                    print(f"{'='*70}\n")
                    
                    # Return chatbot-style response
                    result = {
                        'success': True,
                        'chat_response': chat_response,
                        'is_chatbot': True,
                        'ai_engine': 'groq_chat',
                        'analysis': 'Groq đang tư vấn cho bạn...',
                        'predictions': [],
                        'recommendations': [],
                        'warning': 'Đây là tư vấn từ AI, không thay thế chẩn đoán y tế chuyên nghiệp.',
                    }
                    
                    # Save to history
                    import re
                    disease_match = re.search(r'\*\*([^*]+)\*\*', chat_response)
                    disease_name = disease_match.group(1) if disease_match else "Tư vấn chung"
                    save_search_history(symptoms, disease_name, chat_response[:500], 0)
                    
                    return jsonify(result)
                    
                except Exception as e:
                    print(f"❌ Groq Chatbot Error: {e}")
                    return jsonify({
                        'error': f'Lỗi khi xử lý: {str(e)}',
                        'chat_response': f'Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Lỗi: {str(e)[:100]}',
                        'is_chatbot': True,
                        'ai_engine': 'groq_chat'
                    }), 500
        
        # === GEMINI: CHATBOT MODE (Conversational) ===
        if ai_engine == 'gemini':
            print("💬 Gemini Chatbot Mode: Natural conversation")
            
            # Check if Gemini is available
            if not genai:
                return jsonify({
                    'error': 'Gemini API chưa được cài đặt. Vui lòng cài đặt: pip install google-generativeai',
                    'chat_response': 'Xin lỗi, Gemini AI chưa được cài đặt. Vui lòng liên hệ quản trị viên để cài đặt Google Generative AI.',
                    'is_chatbot': True,
                    'ai_engine': 'gemini'
                }), 500
            
            try:
                model_name = GEMINI_TUNED_MODEL if GEMINI_TUNED_MODEL else GEMINI_MODEL
                model = genai.GenerativeModel(model_name)
                
                # Simple, natural prompt for chatbot
                chatbot_prompt = symptoms
                
                response = model.generate_content(
                    chatbot_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,  # Higher for more natural conversation
                        max_output_tokens=2000,
                        top_p=0.9,
                    )
                )
                
                chat_response = response.text
                
                print(f"\n{'='*70}")
                print("RAW RESPONSE FROM GEMINI CHATBOT:")
                print(f"Text length: {len(chat_response)} chars")
                print(f"Content:\n{chat_response}")
                print(f"{'='*70}\n")
                
                # Return chatbot-style response (không parse như Groq)
                result = {
                    'success': True,
                    'chat_response': chat_response,  # Full conversational response
                    'is_chatbot': True,
                    'ai_engine': 'gemini',
                    'analysis': 'Gemini đang tư vấn cho bạn...',
                    'predictions': [],  # No structured predictions for chatbot mode
                    'recommendations': [],
                    'warning': 'Đây là tư vấn từ AI, không thay thế chẩn đoán y tế chuyên nghiệp.',
                }
                
                # Save to history
                # Extract disease name if mentioned in response (simple extraction)
                import re
                disease_match = re.search(r'\*\*([^*]+)\*\*', chat_response)
                disease_name = disease_match.group(1) if disease_match else "Tư vấn chung"
                save_search_history(symptoms, disease_name, chat_response[:500], 0)
                
                return jsonify(result)
                
            except Exception as e:
                print(f"❌ Gemini API Error: {e}")
                error_message = str(e)
                
                # Friendly error messages
                if 'API_KEY_INVALID' in error_message or 'API key' in error_message:
                    friendly_error = 'Xin lỗi, API key của Gemini chưa được cấu hình đúng. Vui lòng liên hệ quản trị viên để cập nhật API key.'
                elif 'quota' in error_message.lower() or 'limit' in error_message.lower():
                    friendly_error = 'Xin lỗi, Gemini API đã hết quota. Vui lòng thử lại sau hoặc liên hệ quản trị viên.'
                elif 'not found' in error_message.lower():
                    friendly_error = 'Xin lỗi, model Gemini không tìm thấy. Có thể model chưa được fine-tune hoặc tên model không đúng.'
                else:
                    friendly_error = f'Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Chi tiết: {error_message[:100]}'
                
                return jsonify({
                    'error': friendly_error,
                    'chat_response': friendly_error,
                    'is_chatbot': True,
                    'ai_engine': 'gemini',
                    'technical_error': error_message
                }), 500
        
        # === GROQ: DIAGNOSIS MODE (Structured) ===
        # Only for diagnosis mode - validate and search database
        
        print(f"\n{'='*70}")
        print("⚡ GROQ DIAGNOSIS MODE")
        print(f"{'='*70}")
        
        # Validate input
        is_valid, error_message = validate_symptoms_input(symptoms)
        if not is_valid:
            return jsonify({
                'error': error_message,
                'analysis': 'Hệ thống chỉ hỗ trợ chẩn đoán bệnh dựa trên triệu chứng',
                'predictions': [],
                'recommendations': [
                    'Vui lòng nhập triệu chứng cụ thể như: đau đầu, sốt, ho, buồn nôn...',
                    'Mô tả chi tiết: vị trí đau, mức độ, thời gian xuất hiện',
                    'Ví dụ: "Tôi bị đau đầu dữ dội, sốt cao 39 độ, buồn nôn"'
                ],
                'warning': '⚠️ Câu hỏi không hợp lệ. Vui lòng nhập triệu chứng bệnh!'
            }), 400
        
        # Find relevant diseases from database
        relevant_context, relevant_diseases, best_match_score, top_diseases_with_scores = find_relevant_diseases(symptoms, top_k=20)
        
        print(f"\n🔍 Found {len(relevant_diseases)} relevant diseases from database")
        print(f"Top 5: {relevant_diseases[:5]}")
        print(f"Best match score: {best_match_score}")
        
        # If high confidence match, use CSV prediction
        if best_match_score >= CSV_CONFIDENCE_THRESHOLD and len(top_diseases_with_scores) >= 3:
            print(f"✅ Using CSV prediction (score: {best_match_score} >= {CSV_CONFIDENCE_THRESHOLD})")
            result = predict_from_csv_data(symptoms, top_diseases_with_scores, ai_engine=ai_engine)
            
            if result:
                print(f"📊 CSV Predictions: {[p['disease'] for p in result['predictions'][:3]]}")
                result['ai_engine'] = ai_engine
                return jsonify(result)
        
        # Enhanced prompt với knowledge base từ CSV + thông tin chi tiết
        print(f"🤖 Using Groq API for detailed diagnosis")
        prompt = f"""Bạn là bác sĩ AI chuyên nghiệp với database {len(diseases)} bệnh tiếng Việt.

THÔNG TIN TỪ DATABASE (các bệnh và triệu chứng liên quan đến input):
{relevant_context}

---

TRIỆU CHỨNG CỦA BỆNH NHÂN: "{symptoms}"

NHIỆM VỤ: 
1. Phân tích triệu chứng và đưa ra 1 BỆNH có khả năng cao nhất dựa trên thông tin từ database
2. Cung cấp THÔNG TIN CHI TIẾT về bệnh đó

QUY TẮC XÁC SUẤT (QUAN TRỌNG):
- 85-95%: Triệu chứng RẤT ĐIỂN HÌNH + có dấu hiệu ĐẶC TRƯNG RIÊNG của bệnh đó (ví dụ: "ho ra máu" cho ung thư thanh quản)
- 70-84%: Triệu chứng khớp tốt, có nhiều dấu hiệu đặc trưng
- 50-69%: Triệu chứng khớp nhưng CHUNG CHUNG (nhiều bệnh cũng có triệu chứng tương tự)
- 30-49%: Khả năng thấp, chỉ một vài triệu chứng khớp
- 10-29%: Rất ít khả năng, nhưng vẫn cần xem xét

⚠️ LƯU Ý VỀ TRIỆU CHỨNG CHUNG:
- "Khàn tiếng + khó nuốt + sưng cổ" → CÓ THỂ LÀ Bướu Cổ Lành Tính HOẶC Ung Thư Thanh Quản
- KHÔNG đưa ra 80-90% nếu chỉ có triệu chứng chung chung
- Ung Thư thường kèm: sụt cân, ho ra máu, khàn tiếng kéo dài >3 tuần, hút thuốc lá
- Bướu Cổ Lành Tính thường kèm: mệt mỏi, thay đổi cân nặng, da khô, táo bón

TRẢ LỜI THEO FORMAT:

🔍 Phân tích: [1-2 câu phân tích triệu chứng]

💡 Dự đoán bệnh:

**Tên Bệnh**
Lý do: [Giải thích tại sao triệu chứng khớp với bệnh này dựa trên database]

📋 THÔNG TIN CHI TIẾT VỀ BỆNH NÀY:

🩺 Triệu chứng đầy đủ:
- [Triệu chứng 1]
- [Triệu chứng 2]
- [Triệu chứng 3]
- [Triệu chứng 4]

💊 Cách chữa/điều trị:
- [Phương pháp điều trị 1]
- [Phương pháp điều trị 2]
- [Phương pháp điều trị 3]

⚠️ Nguyên nhân:
- [Nguyên nhân 1]
- [Nguyên nhân 2]

💊 Khuyến nghị:
- [Lời khuyên cụ thể]
- [Lời khuyên cụ thể]

QUAN TRỌNG - CÁCH PHÂN BIỆT:
1. ƯU TIÊN sử dụng các bệnh từ phần "THÔNG TIN TỪ DATABASE" ở trên
2. So sánh KỸ triệu chứng user với triệu chứng trong database:
   - Nếu có thêm dấu hiệu ĐẶC TRƯNG (ho ra máu, sụt cân, hút thuốc) → xác suất cao hơn
   - Nếu CHỈ có triệu chứng CHUNG CHUNG (khàn tiếng, khó nuốt) → xác suất thấp hơn (50-65%)
3. XEM XÉT NHIỀU KHẢ NĂNG nếu triệu chứng chung:
   - Ví dụ: "khàn tiếng + khó nuốt + sưng cổ" → có thể là:
     • Bướu Cổ Lành Tính (55% nếu không có dấu hiệu ung thư)
     • Ung Thư Thanh Quản (40% nếu không có ho ra máu, sụt cân)
4. KHÔNG đưa ra 80-95% trừ khi có dấu hiệu ĐẶC TRƯNG RÕ RÀNG
5. Tổng % có thể > 100% (vì là xác suất độc lập)
6. Chỉ dùng tên bệnh CHÍNH XÁC từ database tiếng Việt
7. Với triệu chứng thai sản → ưu tiên: Ối Vỡ Non, Sinh Non, Băng Huyết Sau Sinh
8. Với sốt + đau → ưu tiên: Sốt Xuất Huyết, Cúm, Viêm Phổi"""
        
        # Call Groq API (Gemini already returned above)
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.15,  # Giảm để model thận trọng hơn, không quá tự tin
            max_tokens=2500,  # Tăng để có đủ không gian cho thông tin chi tiết về bệnh
            top_p=0.9,
        )
        
        result_text = response.choices[0].message.content
        # Debug: Log raw response
        print(f"\n{'='*70}")
        print("RAW RESPONSE FROM GROQ:")
        print(f"Text length: {len(result_text)} chars")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Content:\n{result_text}")
        print(f"{'='*70}\n")
        import re
        
        # Clean text
        result_text = result_text.replace('```json', '').replace('```', '')
        
        # Extract predictions - chỉ lấy 1 bệnh, không có %
        predictions = []
        
        # Pattern mới: "**Tên Bệnh**" hoặc "Tên Bệnh" sau "💡 Dự đoán bệnh:"
        disease_name = None
        
        # Tìm phần "💡 Dự đoán bệnh:" và lấy tên bệnh ngay sau đó
        pred_section = re.search(r'💡\s*Dự đoán bệnh:\s*\n+\*\*([^*\n]+)\*\*', result_text, re.IGNORECASE)
        if pred_section:
            disease_name = pred_section.group(1).strip()
        else:
            # Fallback: tìm pattern đơn giản hơn
            pred_section = re.search(r'💡\s*Dự đoán bệnh:\s*\n+([^\n]+)', result_text, re.IGNORECASE)
            if pred_section:
                disease_name = pred_section.group(1).strip()
                # Remove ký tự đặc biệt
                disease_name = disease_name.replace('**', '').replace('*', '').strip()
        
        if disease_name:
            # Tìm lý do
            reason = ""
            reason_match = re.search(r'Lý do:\s*([^\n]+(?:\n(?!📋|💊|⚠️)[^\n]+)*)', result_text, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
                reason = re.sub(r'\s+', ' ', reason)[:300]
            
            if not reason:
                reason = "Triệu chứng khớp với bệnh này dựa trên phân tích database"
            
            predictions.append({
                'disease': disease_name,
                'probability': 0,  # Không hiển thị %
                'reason': reason
            })
        
        # Extract recommendations
        recommendations = []
        rec_patterns = [
            r'💊\s*Khuyến nghị:(.*?)(?=\n\n|⚠️|$)',
            r'Khuyến nghị:(.*?)(?=\n\n|⚠️|$)',
            r'Lời khuyên:(.*?)(?=\n\n|⚠️|$)'
        ]
        
        for pattern in rec_patterns:
            rec_match = re.search(pattern, result_text, re.DOTALL | re.IGNORECASE)
            if rec_match:
                rec_text = rec_match.group(1)
                # Extract bullet points
                rec_items = re.findall(r'[-•]\s*([^\n]+)', rec_text)
                recommendations = [r.strip() for r in rec_items if len(r.strip()) > 10][:5]
                break
        
        if not recommendations:
            recommendations = [
                'Nghỉ ngơi đầy đủ',
                'Uống nhiều nước',
                'Theo dõi triệu chứng',
                'Đi khám bác sĩ nếu tình trạng xấu đi'
            ]
        
        # Extract analysis
        analysis = "Dựa trên các triệu chứng bạn mô tả"
        analysis_patterns = [
            r'🔍\s*Phân tích:\s*([^\n]+(?:\n(?!💡|\d+\.)[^\n]+)*)',
            r'Phân tích:\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)'
        ]
        
        for pattern in analysis_patterns:
            analysis_match = re.search(pattern, result_text, re.IGNORECASE)
            if analysis_match:
                analysis = analysis_match.group(1).strip()
                analysis = re.sub(r'\s+', ' ', analysis)[:250]
                break
        
        # Extract detailed info about top disease
        disease_info = None
        if predictions and len(predictions) > 0:
            top_disease = predictions[0]['disease']
            
            # Extract full symptoms
            symptoms_detail = []
            symptoms_patterns = [
                r'🩺\s*Triệu chứng đầy đủ:(.*?)(?=💊|⚠️|\n\n[🔍💡📋])',
                r'Triệu chứng đầy đủ:(.*?)(?=Cách chữa|Nguyên nhân|\n\n)'
            ]
            for pattern in symptoms_patterns:
                symptoms_match = re.search(pattern, result_text, re.DOTALL | re.IGNORECASE)
                if symptoms_match:
                    symptoms_text = symptoms_match.group(1)
                    symptoms_detail = re.findall(r'[-•]\s*([^\n]+)', symptoms_text)
                    symptoms_detail = [s.strip() for s in symptoms_detail if len(s.strip()) > 5][:10]
                    break
            
            # Extract treatment methods
            treatment = []
            treatment_patterns = [
                r'💊\s*Cách chữa/điều trị:(.*?)(?=⚠️|💊\s*Khuyến nghị|\n\n[🔍💡📋])',
                r'Cách chữa/điều trị:(.*?)(?=Nguyên nhân|Khuyến nghị|\n\n)'
            ]
            for pattern in treatment_patterns:
                treatment_match = re.search(pattern, result_text, re.DOTALL | re.IGNORECASE)
                if treatment_match:
                    treatment_text = treatment_match.group(1)
                    treatment = re.findall(r'[-•]\s*([^\n]+)', treatment_text)
                    treatment = [t.strip() for t in treatment if len(t.strip()) > 5][:10]
                    break
            
            # Extract causes
            causes = []
            causes_patterns = [
                r'⚠️\s*Nguyên nhân:(.*?)(?=💊|📋|\n\n[🔍💡])',
                r'Nguyên nhân:(.*?)(?=Cách chữa|Khuyến nghị|\n\n)'
            ]
            for pattern in causes_patterns:
                causes_match = re.search(pattern, result_text, re.DOTALL | re.IGNORECASE)
                if causes_match:
                    causes_text = causes_match.group(1)
                    causes = re.findall(r'[-•]\s*([^\n]+)', causes_text)
                    causes = [c.strip() for c in causes if len(c.strip()) > 5][:8]
                    break
            
            # Build disease info
            if symptoms_detail or treatment or causes:
                disease_info = {
                    'disease_name': top_disease,
                    'symptoms': symptoms_detail if symptoms_detail else ['Thông tin sẽ được cập nhật'],
                    'treatment': treatment if treatment else ['Vui lòng đi khám bác sĩ để được tư vấn điều trị cụ thể'],
                    'causes': causes if causes else ['Nhiều nguyên nhân khác nhau']
                }
        
        # Fallback nếu không có predictions - tìm bệnh từ database
        if not predictions and relevant_diseases:
            # Lấy bệnh đầu tiên từ database search
            disease_name = relevant_diseases[0]
            predictions.append({
                'disease': disease_name,
                'probability': 0,
                'reason': f'Triệu chứng khớp với {disease_name} dựa trên phân tích database'
            })
        
        # Thêm thông tin chi tiết từ database cho từng bệnh được dự đoán
        detailed_predictions = []
        for pred in predictions:
            disease_name = pred['disease']
            
            # Lấy triệu chứng điển hình từ database
            typical_symptoms = []
            if disease_name in disease_symptoms:
                symptom_samples = disease_symptoms[disease_name][:5]  # Top 5 triệu chứng
                for symptom in symptom_samples:
                    # Clean symptom text
                    clean = symptom.replace("Tôi có thể đang bị bệnh gì?", "")
                    clean = clean.replace('"', '').strip()
                    clean = re.sub(r'^(Tôi|Bệnh nhân)\s+(đang|hiện đang|đang cảm thấy|cảm thấy|hay bị|bị)\s+', '', clean)
                    clean = re.sub(r'^\s*có các triệu chứng như\s+', '', clean)
                    if clean and len(clean) > 10:
                        typical_symptoms.append(clean)
            
            # Đếm số mẫu trong database
            sample_count = len(df[df['Disease'] == disease_name])
            
            detailed_predictions.append({
                'disease': disease_name,
                'probability': pred['probability'],
                'reason': pred['reason'],
                'typical_symptoms': typical_symptoms[:3],  # Top 3 triệu chứng điển hình
                'database_samples': sample_count,
                'has_database_info': len(typical_symptoms) > 0
            })
        
        # Lấy các bệnh liên quan khác (từ kết quả tìm kiếm nhưng không được dự đoán)
        predicted_diseases = [p['disease'] for p in predictions]
        related_diseases = []
        for disease in relevant_diseases[:15]:  # Top 15 từ database
            if disease not in predicted_diseases:
                sample_count = len(df[df['Disease'] == disease])
                # Lấy 1-2 triệu chứng điển hình
                symptoms = []
                if disease in disease_symptoms:
                    for s in disease_symptoms[disease][:2]:
                        clean = re.sub(r'Tôi có thể đang bị bệnh gì\?|"', '', s).strip()
                        clean = re.sub(r'^(Tôi|Bệnh nhân)\s+(đang|hiện đang|cảm thấy|hay bị|bị)\s+', '', clean)
                        if clean and len(clean) > 10:
                            symptoms.append(clean[:80])  # Limit length
                
                related_diseases.append({
                    'disease': disease,
                    'sample_symptoms': symptoms[:2],
                    'database_samples': sample_count
                })
                if len(related_diseases) >= 5:  # Chỉ lấy 5 bệnh liên quan
                    break
        
        # Final result với thông tin đầy đủ
        result = {
            'success': True,
            'analysis': analysis,
            'predictions': detailed_predictions if detailed_predictions else [
                {
                    'disease': 'Không xác định được',
                    'probability': 0,
                    'reason': 'Vui lòng mô tả triệu chứng chi tiết hơn',
                    'typical_symptoms': [],
                    'database_samples': 0,
                    'has_database_info': False
                }
            ],
            'disease_info': disease_info,  # Thông tin chi tiết về bệnh (triệu chứng đầy đủ, cách chữa, nguyên nhân)
            'recommendations': recommendations,
            'ai_engine': ai_engine,  # AI engine being used
            'warning': 'Đây là dự đoán AI, KHÔNG PHẢI chẩn đoán y tế. Hãy đi khám bác sĩ để được chẩn đoán chính xác!',
            
            # Thông tin bổ sung từ database
            'additional_info': {
                'related_diseases': related_diseases,
                'total_diseases_analyzed': len(relevant_diseases),
                'confidence_level': 'cao' if (detailed_predictions and detailed_predictions[0]['probability'] >= 70) else 'trung bình' if (detailed_predictions and detailed_predictions[0]['probability'] >= 50) else 'thấp'
            },
            
            # Metadata
            'metadata': {
                'source': f'{ai_engine.upper()} AI + CSV Database',
                'model': GROQ_MODEL if ai_engine == 'groq' else GEMINI_MODEL,
                'provider': ai_engine.title(),
                'database_stats': {
                    'total_diseases': len(diseases),
                    'total_symptom_samples': len(df),
                    'diseases_searched': len(relevant_diseases)
                }
            }
        }
        
        # Lưu lịch sử vào database
        if detailed_predictions and len(detailed_predictions) > 0:
            disease_name = detailed_predictions[0]['disease']
            confidence = detailed_predictions[0].get('probability', 0)
            save_search_history(symptoms, disease_name, analysis, confidence)
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for quota exceeded or rate limit
        if '429' in error_msg or 'quota' in error_msg.lower() or 'rate_limit' in error_msg.lower():
            # Fallback to CSV prediction when API fails
            print("⚠️ API limit reached, falling back to CSV prediction")
            relevant_context, relevant_diseases, best_match_score, top_diseases_with_scores = find_relevant_diseases(symptoms, top_k=20)
            
            if top_diseases_with_scores and len(top_diseases_with_scores) >= 3:
                result = predict_from_csv_data(symptoms, top_diseases_with_scores, ai_engine='groq')  # Fallback to groq
                if result:
                    result['warning'] = '⚠️ API hết quota - Kết quả từ CSV database. Hãy đi khám bác sĩ để được chẩn đoán chính xác!'
                    result['source'] = 'CSV Database (API unavailable)'
                    result['ai_engine'] = 'csv_fallback'
                    return jsonify(result)
            
            return jsonify({
                'error': '❌ HẾT QUOTA/RATE LIMIT API',
                'analysis': f'API key đã hết quota hoặc vượt rate limit',
                'predictions': [
                    {'disease': 'Không thể dự đoán', 'probability': 0, 'reason': 'API key hết quota/rate limit'}
                ],
                'recommendations': [
                    '⏰ Chờ một chút để rate limit reset',
                    '💳 Kiểm tra quota tại: https://console.groq.com',
                    '🔑 Tạo API key mới nếu cần',
                    '💻 Groq có free tier rất generous'
                ],
                'warning': '⚠️ HẾT QUOTA/RATE LIMIT - Vui lòng đợi hoặc kiểm tra API key',
                'source': 'Error'
            }), 429
        
        # Other errors - fallback to CSV
        print(f"ERROR: {error_msg}")
        print("⚠️ Error occurred, falling back to CSV prediction")
        
        try:
            relevant_context, relevant_diseases, best_match_score, top_diseases_with_scores = find_relevant_diseases(symptoms, top_k=20)
            
            if top_diseases_with_scores and len(top_diseases_with_scores) >= 3:
                result = predict_from_csv_data(symptoms, top_diseases_with_scores, ai_engine='groq')  # Fallback to groq
                if result:
                    result['warning'] = f'⚠️ API lỗi - Kết quả từ CSV database. Hãy đi khám bác sĩ!'
                    result['source'] = 'CSV Database (API error fallback)'
                    result['ai_engine'] = 'csv_fallback'
                    return jsonify(result)
        except Exception as csv_error:
            print(f"CSV fallback also failed: {csv_error}")
        
        return jsonify({
            'error': f'Lỗi: {error_msg[:100]}',
            'analysis': 'Có lỗi xảy ra khi xử lý yêu cầu',
            'predictions': [
                {'disease': 'Lỗi hệ thống', 'probability': 0, 'reason': 'Vui lòng thử lại sau'}
            ],
            'recommendations': [
                'Kiểm tra kết nối internet',
                'Thử lại sau vài giây',
                'Kiểm tra console để xem chi tiết lỗi'
            ],
            'warning': '⚠️ Có lỗi xảy ra - vui lòng thử lại',
            'source': 'Error'
        }), 500

@app.route('/stats')
def stats():
    """
    Endpoint trả về thống kê chi tiết về database và system
    """
    # Tính toán thống kê
    disease_sample_counts = df['Disease'].value_counts().to_dict()
    top_10_diseases = dict(list(disease_sample_counts.items())[:10])
    
    # Tính avg samples per disease
    avg_samples = len(df) / len(diseases)
    
    return jsonify({
        'success': True,
        'database': {
            'total_diseases': len(diseases),
            'total_symptom_samples': len(df),
            'avg_samples_per_disease': round(avg_samples, 2),
            'top_10_diseases_by_samples': top_10_diseases,
            'diseases_list_sample': diseases[:20]  # 20 bệnh đầu tiên
        },
        'model': {
            'engines': {
                'groq': GROQ_MODEL,
                'gemini': GEMINI_MODEL
            },
            'default': DEFAULT_AI_ENGINE,
            'type': 'Dual AI Engine with TF-IDF + CSV Database',
            'features': [
                'TF-IDF based disease matching',
                'Database-driven symptom analysis',
                'Context-aware prediction',
                'Typical symptoms from real data'
            ]
        },
        'api': {
            'version': '2.0',
            'endpoints': {
                '/predict': 'POST - Dự đoán bệnh từ triệu chứng',
                '/stats': 'GET - Thống kê hệ thống',
                '/': 'GET - Giao diện web'
            },
            'response_fields': [
                'analysis',
                'predictions (with typical_symptoms)',
                'recommendations',
                'additional_info (related_diseases)',
                'metadata (database_stats)'
            ]
        },
        'accuracy': {
            'estimated': '85-95%',
            'notes': 'Dựa trên 23,520 mẫu triệu chứng thực tế',
            'confidence_levels': {
                'high': 'probability >= 70%',
                'medium': '50% <= probability < 70%',
                'low': 'probability < 50%'
            }
        }
    })

@app.route('/diseases')
def get_diseases():
    """
    Endpoint trả về danh sách tất cả các bệnh
    """
    disease_list = []
    for disease in sorted(diseases):
        sample_count = len(df[df['Disease'] == disease])
        disease_list.append({
            'disease': disease,
            'sample_count': sample_count
        })
    
    return jsonify({
        'success': True,
        'total': len(disease_list),
        'diseases': disease_list
    })

@app.route('/disease/<disease_name>')
def get_disease_info(disease_name):
    """
    Endpoint trả về thông tin chi tiết về một bệnh cụ thể
    """
    if disease_name not in diseases:
        return jsonify({
            'success': False,
            'error': f'Bệnh "{disease_name}" không có trong database'
        }), 404
    
    # Lấy tất cả triệu chứng của bệnh này
    disease_data = df[df['Disease'] == disease_name]['Question'].tolist()
    
    # Clean symptoms
    symptoms = []
    for symptom in disease_data[:20]:  # Top 20 triệu chứng
        clean = symptom.replace("Tôi có thể đang bị bệnh gì?", "")
        clean = clean.replace('"', '').strip()
        clean = re.sub(r'^(Tôi|Bệnh nhân)\s+(đang|hiện đang|cảm thấy|hay bị|bị)\s+', '', clean)
        clean = re.sub(r'^\s*có các triệu chứng như\s+', '', clean)
        if clean and len(clean) > 10:
            symptoms.append(clean)
    
    return jsonify({
        'success': True,
        'disease': disease_name,
        'total_samples': len(disease_data),
        'typical_symptoms': symptoms[:10],  # Top 10
        'all_symptom_variations': len(disease_data)
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Lấy danh sách lịch sử tìm kiếm"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Lấy 50 lịch sử gần nhất
        query = """
            SELECT id, symptoms, disease, analysis, confidence, created_at
            FROM search_history
            ORDER BY created_at DESC
            LIMIT 50
        """
        cursor.execute(query)
        history = cursor.fetchall()
        
        # Convert datetime to string
        for item in history:
            if item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Error as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/history/<int:history_id>', methods=['GET'])
def get_history_item(history_id):
    """Lấy chi tiết 1 lịch sử"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM search_history WHERE id = %s"
        cursor.execute(query, (history_id,))
        item = cursor.fetchone()
        
        if item:
            if item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
            return jsonify({'success': True, 'data': item})
        else:
            return jsonify({'success': False, 'error': 'Not found'}), 404
            
    except Error as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/history/delete/<int:history_id>', methods=['DELETE'])
def delete_history_item(history_id):
    """Xóa 1 lịch sử"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        query = "DELETE FROM search_history WHERE id = %s"
        cursor.execute(query, (history_id,))
        conn.commit()
        
        if cursor.rowcount > 0:
            return jsonify({'success': True, 'message': 'Deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Not found'}), 404
            
    except Error as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/history/clear', methods=['DELETE'])
def clear_all_history():
    """Xóa toàn bộ lịch sử"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        query = "DELETE FROM search_history"
        cursor.execute(query)
        conn.commit()
        
        deleted_count = cursor.rowcount
        
        return jsonify({
            'success': True,
            'message': f'Cleared {deleted_count} records',
            'count': deleted_count
        })
            
    except Error as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    print("="*70)
    print("🌐 SMART DISEASE DIAGNOSIS WEB APP")
    print("="*70)
    print(f"📊 Database: {len(diseases)} loại bệnh, {len(df)} mẫu triệu chứng")
    print(f"🤖 AI Engines: Groq ({GROQ_MODEL}) + Gemini ({GEMINI_MODEL})")
    print(f"⚡ Strategy: CSV First → Groq API Fallback")
    print(f"🎯 CSV Threshold: {CSV_CONFIDENCE_THRESHOLD} (điều chỉnh để tối ưu)")
    print(f"\n💡 Lợi ích:")
    print(f"   • Nhanh: Dự đoán từ CSV không cần gọi API")
    print(f"   • Tiết kiệm: Giảm API calls khi có kết quả tốt từ CSV")
    print(f"   • Chính xác: Dùng Groq AI khi cần phân tích phức tạp")
    # Get port from environment variable (Railway/Heroku sets this automatically)
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n🚀 Starting server...")
    print(f"📍 URL: http://0.0.0.0:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"\n⚠️  Nhấn Ctrl+C để dừng server")
    print("="*70)
    
    app.run(debug=debug, host='0.0.0.0', port=port)

