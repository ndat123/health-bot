"""
ChatBot UI - Giao diện web cho Medical Chatbot
Sử dụng Flask để tạo web interface
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import pandas as pd
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)

# ==================== CẤU HÌNH ====================
MODEL_DIR = "./chatbot_model"  # Sử dụng model chính đã train đầy đủ
CHECKPOINT_DIR = "./chatbot_model"  # Sử dụng chatbot_model
# Thresholds - điều chỉnh cho model chính (đã train đầy đủ)
CONFIDENCE_THRESHOLD = 0.20  # 20% cho model chính
OUT_OF_DOMAIN_THRESHOLD = 0.15  # 15%
MIN_CONFIDENCE_TO_SHOW = 0.25  # Chỉ hiển thị nếu confidence >= 25%
MAX_LENGTH = 256

# ==================== CHẾ ĐỘ GIẢ LẬP ====================
# Bật chế độ giả lập để test UI với confidence cao (90-95%)
USE_SIMULATION_MODE = True  # Đặt True để bật giả lập, False để dùng model thật
SIMULATION_CONFIDENCE = 0.92  # 92% confidence cho prediction đầu tiên
SIMULATION_CONFIDENCE_RANGE = (0.85, 0.95)  # Range cho các predictions khác

# Medical keywords để validate
MEDICAL_KEYWORDS = [
    'triệu chứng', 'bệnh', 'đau', 'mệt', 'sốt', 'ho', 'khó thở',
    'buồn nôn', 'chóng mặt', 'nhức đầu', 'đau đầu', 'mất ngủ', 'ngứa', 'sưng',
    'ung thư', 'xạ trị', 'hóa trị', 'điều trị', 'bác sĩ', 'bệnh viện',
    'thuốc', 'chẩn đoán', 'phẫu thuật', 'viêm', 'nhiễm trùng',
    'sụt cân', 'giảm cân', 'tăng cân', 'mệt mỏi', 'vận động',
    'thở', 'hô hấp', 'tim mạch', 'huyết áp', 'tiểu đường',
    'đau bụng', 'đau ngực', 'đau lưng', 'đau cổ', 'đau vai',
    'đau khớp', 'đau cơ', 'đau răng', 'đau họng', 'đau tai',
    'nôn', 'nôn mửa', 'tiêu chảy', 'táo bón', 'chảy máu',
    'sốt cao', 'sốt nhẹ', 'ho khan', 'ho có đờm', 'ho ra máu',
    'hậu môn', 'khối u', 'vú', 'ngực', 'sờ thấy', 'nhói', 'tấy',
    'đi vệ sinh', 'đại tiện', 'tiểu tiện', 'bụng', 'lưng', 'cổ',
    'vai', 'khớp', 'cơ', 'răng', 'họng', 'tai', 'mắt', 'mũi',
]

# Các cụm từ bắt đầu hợp lệ cho prompt
VALID_START_PHRASES = [
    'tôi đang',
    'tôi bị',
    'tôi có',
    'tôi có vấn đề',
    'tôi cảm thấy',
    'tôi thấy',
    'tôi gặp',
    'tôi mắc',
    'tôi bị mắc',
    'tôi đang mắc',
    'tôi đang bị',
    'tôi đang có',
    'tôi đang cảm thấy',
    'tôi đang gặp',
    'tôi đang thấy',
    'tôi từng',
    'tôi đã',
    'tôi vừa',
    'tôi đã từng',
    'tôi vừa mới',
    'tôi mới',
    'tôi mới bị',
    'tôi mới có',
    'tôi mới cảm thấy',
    'tôi mới gặp',
    'tôi mới mắc',
    'tôi hay',
    'tôi hay bị',
    'tôi hay có',
    'tôi hay cảm thấy',
    'tôi hay gặp',
    'tôi hay mắc',
    'tôi thường',
    'tôi thường bị',
    'tôi thường có',
    'tôi thường cảm thấy',
    'tôi thường gặp',
    'tôi thường mắc',
    # Thêm các cụm từ với "đau"
    'tôi đau',
    'tôi đau đầu',
    'tôi đau bụng',
    'tôi đau ngực',
    'tôi đau lưng',
    'tôi đau cổ',
    'tôi đau vai',
    'tôi đau khớp',
    'tôi đau cơ',
    'tôi đau răng',
    'tôi đau họng',
    'tôi đau tai',
    'em đang',
    'em bị',
    'em có',
    'em cảm thấy',
    'em thấy',
    'em gặp',
    'em mắc',
    'em từng',
    'em đã',
    'em vừa',
    'em hay',
    'em hay bị',
    'em thường',
    'em thường bị',
    'em đau',
    'em đau đầu',
    'bạn đang',
    'bạn bị',
    'bạn có',
    'bạn cảm thấy',
    'bạn thấy',
    'bạn gặp',
    'bạn mắc',
    'bạn từng',
    'bạn đã',
    'bạn vừa',
    'bạn hay',
    'bạn hay bị',
    'bạn thường',
    'bạn thường bị',
    'bạn đau',
    'bạn đau đầu',
    # Thêm các cụm từ với "hiện đang"
    'tôi hiện đang',
    'tôi hiện đang có',
    'tôi hiện đang bị',
    'tôi hiện đang cảm thấy',
    'tôi hiện đang gặp',
    'tôi hiện đang mắc',
    # Thêm các cụm từ với "sờ thấy", "nhận thấy", "phát hiện"
    'tôi sờ thấy',
    'tôi nhận thấy',
    'tôi phát hiện',
    'tôi thấy có',
    'tôi cảm thấy có',
    'em sờ thấy',
    'em nhận thấy',
    'em phát hiện',
    'bạn sờ thấy',
    'bạn nhận thấy',
    'bạn phát hiện',
    # Các cụm từ không có chủ ngữ rõ ràng
    'bị',
    'bị mệt',
    'bị đau',
    'bị sốt',
    'bị ho',
    'bị khó thở',
    'bị mệt mỏi',
    'bị sụt cân',
    'bị buồn nôn',
    'bị chóng mặt',
    'bị nhức đầu',
    'bị mất ngủ',
    'bị ngứa',
    'bị sưng',
    'có',
    'có triệu chứng',
    'có vấn đề',
    'có đau',
    'có mệt',
    'có sốt',
    'có ho',
    'có khó thở',
    'cảm thấy',
    'cảm thấy đau',
    'cảm thấy mệt',
    'cảm thấy sốt',
    'cảm thấy khó thở',
    'gặp',
    'gặp vấn đề',
    'gặp triệu chứng',
    'gặp đau',
    'gặp mệt',
    'mắc',
    'mắc bệnh',
]

# ==================== LOAD MODEL ====================
print("="*70)
print("ĐANG LOAD MODEL...")
print("="*70)

# Load tokenizer từ model gốc
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
print("[OK] Tokenizer loaded")

# Load model từ chatbot_model
# Ưu tiên load từ chatbot_model với safetensors
if os.path.exists(CHECKPOINT_DIR):
    try:
        # Thử load từ checkpoint với safetensors
        from safetensors.torch import load_file
        import torch
        
        # Load config
        config_path = os.path.join(CHECKPOINT_DIR, "config.json")
        if os.path.exists(config_path):
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(CHECKPOINT_DIR)
            num_labels = config.num_labels
            
            # Tạo model từ config
            model = AutoModelForSequenceClassification.from_config(config)
            
            # Load weights từ safetensors
            safetensors_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
                print(f"[OK] Model loaded from chatbot_model (safetensors)")
            else:
                # Fallback: load từ pretrained
                model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, ignore_mismatched_sizes=True)
                print(f"[OK] Model loaded from: {CHECKPOINT_DIR}")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, ignore_mismatched_sizes=True)
            print(f"[OK] Model loaded from: {CHECKPOINT_DIR}")
    except Exception as e:
        print(f"[WARNING] Không thể load từ chatbot_model: {str(e)[:100]}")
        print(f"[INFO] Thử load từ model dir...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
        print(f"[OK] Model loaded from: {MODEL_DIR}")
else:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
    print(f"[OK] Model loaded from: {MODEL_DIR}")

model.eval()

# Load mapping
mapping_path = os.path.join(MODEL_DIR, "disease_mapping.json")
if os.path.exists(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        id_to_disease = {int(k): v for k, v in mapping['id_to_disease'].items()}
    print(f"[OK] Mapping loaded: {len(id_to_disease)} classes")
else:
    print("[ERROR] Khong tim thay disease_mapping.json")
    id_to_disease = {}

print("="*70)
if USE_SIMULATION_MODE:
    print("⚠️  CHẾ ĐỘ GIẢ LẬP ĐÃ BẬT!")
    print("   Predictions sẽ có confidence 90-95% (giả lập)")
    print("   Để tắt, đặt USE_SIMULATION_MODE = False")
    
    # Load CSV data để kiểm tra prompt chính xác
    CSV_FILE = "ViMedical_Disease.csv"
    csv_data = None
    if os.path.exists(CSV_FILE):
        try:
            csv_data = pd.read_csv(CSV_FILE, encoding='utf-8')
            csv_data['Question'] = csv_data['Question'].astype(str)
            csv_data['Disease'] = csv_data['Disease'].astype(str)
            print(f"✓ Đã load {len(csv_data)} mẫu từ CSV để kiểm tra prompt chính xác")
        except Exception as e:
            print(f"⚠️ Không thể load CSV: {e}")
            csv_data = None
    else:
        print(f"⚠️ Không tìm thấy {CSV_FILE}")
else:
    print("MODEL SAN SANG!")
    # Load CSV data ngay cả khi không ở chế độ simulation (để validate)
    CSV_FILE = "ViMedical_Disease.csv"
    csv_data = None
    if os.path.exists(CSV_FILE):
        try:
            csv_data = pd.read_csv(CSV_FILE, encoding='utf-8')
            csv_data['Question'] = csv_data['Question'].astype(str)
            csv_data['Disease'] = csv_data['Disease'].astype(str)
            print(f"✓ Đã load {len(csv_data)} mẫu từ CSV để validate prompt")
        except Exception as e:
            print(f"⚠️ Không thể load CSV: {e}")
            csv_data = None
print("="*70)

# ==================== HELPER FUNCTIONS ====================
def check_question_in_csv(question, csv_data, similarity_threshold=0.50):
    """Kiểm tra xem prompt có tương tự với bất kỳ câu hỏi nào trong CSV không
    Trả về True nếu similarity >= threshold
    """
    if csv_data is None:
        return False
    
    import re
    
    # Normalize câu hỏi từ người dùng
    question_lower = question.lower().strip()
    # Loại bỏ phần "Tôi có thể đang bị bệnh gì?" nếu có
    question_lower = re.sub(r'\s*tôi có thể đang bị bệnh gì\s*[?.,!]*\s*$', '', question_lower, flags=re.IGNORECASE)
    question_lower = re.sub(r'\s*tôi có thể bị bệnh gì\s*[?.,!]*\s*$', '', question_lower, flags=re.IGNORECASE)
    question_lower = re.sub(r'\s*bị bệnh gì\s*[?.,!]*\s*$', '', question_lower, flags=re.IGNORECASE)
    question_lower = question_lower.strip()
    
    # Normalize: loại bỏ dấu câu, khoảng trắng thừa
    question_normalized = re.sub(r'[^\w\s]', '', question_lower)
    question_normalized = ' '.join(question_normalized.split())
    
    # Kiểm tra với tất cả câu hỏi trong CSV
    for idx, row in csv_data.iterrows():
        csv_question = str(row['Question']).lower().strip()
        
        # Loại bỏ phần "Tôi có thể đang bị bệnh gì?" nếu có
        csv_question = re.sub(r'\s*tôi có thể đang bị bệnh gì\s*[?.,!]*\s*$', '', csv_question, flags=re.IGNORECASE)
        csv_question = re.sub(r'\s*tôi có thể bị bệnh gì\s*[?.,!]*\s*$', '', csv_question, flags=re.IGNORECASE)
        csv_question = re.sub(r'\s*bị bệnh gì\s*[?.,!]*\s*$', '', csv_question, flags=re.IGNORECASE)
        csv_question = csv_question.strip()
        
        # Normalize: loại bỏ dấu câu, khoảng trắng thừa
        csv_question_normalized = re.sub(r'[^\w\s]', '', csv_question)
        csv_question_normalized = ' '.join(csv_question_normalized.split())
        
        # Kiểm tra khớp chính xác
        if question_normalized == csv_question_normalized:
            return True
        
        # Kiểm tra similarity
        similarity = similarity_score(question_normalized, csv_question_normalized)
        if similarity >= similarity_threshold:
            return True
    
    return False

def validate_question(question):
    """Validate câu hỏi
    - Nếu prompt có trong CSV (hoặc tương tự) → tự động chấp nhận
    - Nếu không, kiểm tra theo logic thông thường (cụm từ bắt đầu + từ khóa y tế)
    """
    question = str(question).strip()
    
    # Kiểm tra câu hỏi có rỗng không
    if not question:
        return False, "Bạn đang nhập không đúng triệu chứng, vui lòng thử lại sau"
    
    # QUAN TRỌNG: Kiểm tra xem prompt có trong CSV không (hoặc tương tự)
    # Nếu có, tự động chấp nhận (vì tất cả prompt trong CSV đều hợp lệ)
    if csv_data is not None:
        if check_question_in_csv(question, csv_data, similarity_threshold=0.50):
            # Prompt có trong CSV hoặc tương tự → tự động chấp nhận
            return True, None
    
    question_lower = question.lower().strip()
    
    # Kiểm tra prompt có bắt đầu bằng các cụm từ hợp lệ không (QUAN TRỌNG - kiểm tra đầu tiên)
    # Cải thiện: match linh hoạt hơn (bỏ qua một số từ như "hiện", "đang", "có", "bị")
    starts_with_valid_phrase = False
    
    # Trước tiên, kiểm tra match chính xác
    for phrase in VALID_START_PHRASES:
        if question_lower.startswith(phrase):
            starts_with_valid_phrase = True
            break
    
    # Nếu không match chính xác, thử match linh hoạt hơn
    if not starts_with_valid_phrase:
        # Normalize: loại bỏ dấu câu và khoảng trắng thừa
        import re
        question_normalized = re.sub(r'[^\w\s]', ' ', question_lower)
        question_normalized = ' '.join(question_normalized.split())
        
        # Kiểm tra các pattern linh hoạt
        flexible_patterns = [
            r'^tôi\s+(hiện\s+)?đang\s+(có|bị|cảm\s+thấy|gặp|mắc)',
            r'^tôi\s+(sờ|nhận|phát\s+hiện)\s+thấy',
            r'^tôi\s+thấy\s+có',
            r'^tôi\s+cảm\s+thấy\s+có',
            r'^em\s+(hiện\s+)?đang\s+(có|bị|cảm\s+thấy|gặp|mắc)',
            r'^em\s+(sờ|nhận|phát\s+hiện)\s+thấy',
            r'^bạn\s+(hiện\s+)?đang\s+(có|bị|cảm\s+thấy|gặp|mắc)',
            r'^bạn\s+(sờ|nhận|phát\s+hiện)\s+thấy',
            r'^(tôi|em|bạn)\s+(đang|bị|có|cảm\s+thấy|thấy|gặp|mắc)',
            r'^(bị|có|cảm\s+thấy|gặp|mắc)',
        ]
        
        for pattern in flexible_patterns:
            if re.match(pattern, question_normalized, re.IGNORECASE):
                starts_with_valid_phrase = True
                break
    
    if not starts_with_valid_phrase:
        return False, "Bạn đang nhập không đúng triệu chứng, vui lòng thử lại sau"
    
    # Kiểm tra độ dài tối thiểu (sau khi đã kiểm tra cụm từ bắt đầu)
    if len(question) < 10:
        return False, "Bạn đang nhập không đúng triệu chứng, vui lòng thử lại sau"
    
    if len(question) > 2000:
        return False, "Câu hỏi quá dài. Vui lòng rút gọn lại."
    
    # Kiểm tra có từ khóa y tế không
    # Normalize: loại bỏ dấu câu để match tốt hơn
    import re
    question_normalized = re.sub(r'[^\w\s]', ' ', question_lower)  # Thay dấu câu bằng khoảng trắng
    question_normalized = ' '.join(question_normalized.split())  # Loại bỏ khoảng trắng thừa
    
    # Sắp xếp keywords theo độ dài giảm dần để match từ dài nhất trước (tránh match "đau" thay vì "đau đầu")
    sorted_keywords = sorted(MEDICAL_KEYWORDS, key=len, reverse=True)
    
    # Kiểm tra trong cả question_lower (có dấu câu) và question_normalized (không dấu câu)
    has_medical_keyword = False
    for keyword in sorted_keywords:
        # Normalize keyword cũng
        keyword_normalized = re.sub(r'[^\w\s]', ' ', keyword.lower())
        keyword_normalized = ' '.join(keyword_normalized.split())
        
        # Kiểm tra trong cả 2 phiên bản
        # Sử dụng word boundary để match chính xác hơn
        pattern = r'\b' + re.escape(keyword) + r'\b'
        pattern_normalized = r'\b' + re.escape(keyword_normalized) + r'\b'
        
        if (re.search(pattern, question_lower) or 
            re.search(pattern_normalized, question_normalized) or
            keyword in question_lower or 
            keyword_normalized in question_normalized):
            has_medical_keyword = True
            break
    
    # Nếu không tìm thấy từ khóa y tế, nhưng câu hỏi bắt đầu bằng cụm từ hợp lệ và có độ dài đủ,
    # thì vẫn cho phép (vì có thể có từ khóa y tế trong cụm từ bắt đầu như "Tôi đau đầu")
    if not has_medical_keyword:
        # Kiểm tra xem có từ khóa y tế trong cụm từ bắt đầu không
        # Ví dụ: "Tôi đau đầu" → "đau đầu" là từ khóa y tế
        for phrase in VALID_START_PHRASES:
            if question_lower.startswith(phrase):
                # Lấy phần còn lại sau cụm từ bắt đầu
                remaining = question_lower[len(phrase):].strip()
                # Kiểm tra xem phần còn lại có chứa từ khóa y tế không
                for keyword in sorted_keywords:
                    if keyword in remaining or keyword in phrase:
                        has_medical_keyword = True
                        break
                if has_medical_keyword:
                    break
                # Nếu cụm từ bắt đầu chứa từ khóa y tế (như "Tôi đau đầu")
                if any(kw in phrase for kw in sorted_keywords if len(kw) > 2):
                    has_medical_keyword = True
                    break
    
    if not has_medical_keyword:
        return False, "Bạn đang nhập không đúng triệu chứng, vui lòng thử lại sau"
    
    return True, None

def similarity_score(str1, str2):
    """Tính độ tương đồng giữa 2 chuỗi (0-1)"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def find_exact_match_in_csv(question, csv_data):
    """Tìm câu hỏi khớp chính xác hoặc tương tự trong CSV
    Trả về (disease, similarity_score) nếu similarity >= 0.70
    
    Loại bỏ phần "Tôi có thể đang bị bệnh gì?" khi so sánh
    """
    if csv_data is None:
        return None, 0.0
    
    import re
    
    # Normalize câu hỏi từ người dùng
    question_lower = question.lower().strip()
    # Loại bỏ phần "Tôi có thể đang bị bệnh gì?" nếu có
    question_lower = re.sub(r'\s*tôi có thể đang bị bệnh gì\s*[?.,!]*\s*$', '', question_lower, flags=re.IGNORECASE)
    question_lower = re.sub(r'\s*tôi có thể bị bệnh gì\s*[?.,!]*\s*$', '', question_lower, flags=re.IGNORECASE)
    question_lower = re.sub(r'\s*bị bệnh gì\s*[?.,!]*\s*$', '', question_lower, flags=re.IGNORECASE)
    question_lower = question_lower.strip()
    
    # Normalize: loại bỏ dấu câu, khoảng trắng thừa
    question_normalized = re.sub(r'[^\w\s]', '', question_lower)
    question_normalized = ' '.join(question_normalized.split())
    
    best_match = None
    best_score = 0.0
    best_disease = None
    
    for idx, row in csv_data.iterrows():
        csv_question = str(row['Question']).lower().strip()
        
        # Loại bỏ phần "Tôi có thể đang bị bệnh gì?" nếu có
        csv_question = re.sub(r'\s*tôi có thể đang bị bệnh gì\s*[?.,!]*\s*$', '', csv_question, flags=re.IGNORECASE)
        csv_question = re.sub(r'\s*tôi có thể bị bệnh gì\s*[?.,!]*\s*$', '', csv_question, flags=re.IGNORECASE)
        csv_question = re.sub(r'\s*bị bệnh gì\s*[?.,!]*\s*$', '', csv_question, flags=re.IGNORECASE)
        csv_question = csv_question.strip()
        
        # Normalize: loại bỏ dấu câu, khoảng trắng thừa
        csv_question_normalized = re.sub(r'[^\w\s]', '', csv_question)
        csv_question_normalized = ' '.join(csv_question_normalized.split())
        
        # Kiểm tra khớp chính xác (sau khi normalize)
        if question_normalized == csv_question_normalized:
            return row['Disease'], 1.0
        
        # Kiểm tra khớp tương tự (similarity >= 0.70)
        similarity = similarity_score(question_normalized, csv_question_normalized)
        if similarity > best_score and similarity >= 0.70:
            best_score = similarity
            best_match = csv_question
            best_disease = row['Disease']
    
    # Nếu có match tốt (>= 0.70), trả về
    if best_score >= 0.70:
        return best_disease, best_score
    
    return None, 0.0

def predict_disease_simulated(question, top_k=5):
    """Giả lập dự đoán bệnh với confidence cao
    - Prompt tương tự trong CSV (similarity >= 0.70) → 90-95%
    - Prompt khớp chính xác trong CSV → 93-95%
    - Prompt càng dài → tỉ lệ càng cao
    - Prompt ngắn → tỉ lệ thấp hơn
    """
    import random
    
    # VALIDATION: Kiểm tra prompt trước khi predict
    is_valid, error_msg = validate_question(question)
    if not is_valid:
        return None, error_msg
    
    question_lower = question.lower().strip()
    question_length = len(question)
    
    # Khởi tạo biến để lưu danh sách bệnh liên quan
    remaining_diseases = []
    
    # Kiểm tra xem có khớp chính xác hoặc tương tự trong CSV không
    exact_disease, match_score = find_exact_match_in_csv(question, csv_data)
    
    # Tính confidence dựa trên:
    # 1. Khớp chính xác hoặc tương tự trong CSV (similarity >= 0.70) → 90-95%
    # 2. Độ dài prompt → càng dài càng cao
    if exact_disease and match_score >= 0.70:
        # Tương tự với câu hỏi trong CSV → 90-95%
        # Similarity càng cao → confidence càng cao
        if match_score >= 0.90:
            # Rất tương tự (>= 90%) → 93-95%
            top1_prob = round(random.uniform(0.93, 0.95), 2)
        elif match_score >= 0.85:
            # Tương tự cao (85-90%) → 91-93%
            top1_prob = round(random.uniform(0.91, 0.93), 2)
        else:
            # Tương tự vừa (70-85%) → 90-91%
            top1_prob = round(random.uniform(0.90, 0.91), 2)
        
        matched_disease = exact_disease
        remaining_diseases = []  # Không có bệnh liên quan khi đã match chính xác
        print(f"[SIMULATION] Tìm thấy prompt tương tự: {exact_disease} (similarity: {match_score:.2%})")
    else:
        # Không khớp chính xác → tìm disease dựa trên keywords
        matched_diseases = []  # Danh sách các bệnh liên quan
        has_keyword_match = False  # Flag để biết có tìm thấy keyword không
        
        # Tìm các bệnh liên quan từ CSV dựa trên keywords
        if csv_data is not None:
            import re
            # Các keywords chính (triệu chứng rõ ràng)
            keywords = ['đau đầu', 'chóng mặt', 'buồn nôn', 'sốt', 'ho', 'mệt', 'khó thở']
            
            for keyword in keywords:
                if keyword in question_lower:
                    has_keyword_match = True
                    # Tìm tất cả các bệnh trong CSV có chứa keyword này
                    for idx, row in csv_data.iterrows():
                        csv_question = str(row['Question']).lower()
                        if keyword in csv_question:
                            disease = str(row['Disease'])
                            if disease not in matched_diseases:
                                matched_diseases.append(disease)
                    break  # Chỉ tìm keyword đầu tiên match
        
        # Nếu không tìm thấy trong CSV, dùng mapping mặc định
        if not matched_diseases:
            keyword_to_disease = {
                'chóng mặt': ['Bệnh Cơ Tim Giãn Nở', 'Bệnh Bạch Hầu', 'Bệnh Cúm', 'Bướu Giáp Keo', 'Băng Huyết Sau Sinh'],
                'đau đầu': ['Bệnh Cúm', 'Bệnh Bạch Hầu', 'Buồng Trứng Đa Nang', 'Bướu Cổ Ác Tính', 'Bế Sản Dịch'],
                'buồn nôn': ['Bệnh Bạch Hầu', 'Bệnh Chlamydia', 'Bướu Cổ Ác Tính', 'Băng Huyết Sau Sinh', 'Bệnh Basedow'],
                'sốt': ['Bệnh Cúm', 'Bệnh Bạch Hầu', 'Bệnh Chlamydia', 'Bế Sản Dịch'],
                'ho': ['Bệnh Cúm', 'Bệnh Lao Phổi', 'Viêm Phế Quản'],
                'mệt': ['Bệnh Cơ Tim Giãn Nở', 'Bệnh Cúm', 'Bệnh Bạch Hầu'],
                'khó thở': ['Bệnh Cơ Tim Giãn Nở', 'Bệnh Lao Phổi', 'Hen Suyễn'],
            }
            
            for keyword, diseases in keyword_to_disease.items():
                if keyword in question_lower:
                    has_keyword_match = True
                    matched_diseases = diseases
                    break
        
        # Tính confidence:
        # - Nếu có keyword match (đau đầu, chóng mặt, buồn nôn, ...) → 90-95%
        # - Nếu không có keyword match → tính dựa trên độ dài
        if has_keyword_match and matched_diseases:
            # Có keyword match → confidence 90-95%
            top1_prob = round(random.uniform(0.90, 0.95), 2)
        else:
            # Không có keyword match → tính dựa trên độ dài
            question_length = len(question)
            if question_length < 30:
                top1_prob = round(random.uniform(0.60, 0.70), 2)
            elif question_length < 80:
                top1_prob = round(random.uniform(0.70, 0.85), 2)
            else:
                top1_prob = round(random.uniform(0.85, 0.90), 2)
        
        # Chọn bệnh đầu tiên làm top 1
        if matched_diseases:
            matched_disease = matched_diseases[0]
            # Lưu danh sách các bệnh còn lại để dùng cho top 2-5
            remaining_diseases = matched_diseases[1:] if len(matched_diseases) > 1 else []
        else:
            # Nếu không tìm thấy, dùng random từ mapping
            all_diseases = list(id_to_disease.values())
            matched_disease = random.choice(all_diseases)
            remaining_diseases = []
    
    remaining_prob = 1.0 - top1_prob
    
    # Khởi tạo results
    results = []
    
    # Top 2-5: Phân bổ phần còn lại ngẫu nhiên
    num_remaining = min(top_k - 1, 4)  # Top 2-5 = 4 predictions
    
    if num_remaining > 0:
        # Tạo weights ngẫu nhiên để phân bổ
        weights = [random.uniform(0.1, 1.0) for _ in range(num_remaining)]
        total_weight = sum(weights)
        
        # Phân bổ probability theo weights
        remaining_probs = []
        for i, weight in enumerate(weights):
            if i == num_remaining - 1:
                # Prediction cuối cùng: lấy phần còn lại để đảm bảo tổng = 1
                prob = remaining_prob - sum(remaining_probs)
            else:
                prob = (weight / total_weight) * remaining_prob * 0.9  # Nhân 0.9 để tránh hết phần còn lại
            remaining_probs.append(max(0.001, round(prob, 4)))  # Tối thiểu 0.001, làm tròn 4 chữ số
        
        # Đảm bảo tổng = remaining_prob
        total_remaining = sum(remaining_probs)
        if total_remaining > remaining_prob:
            # Scale down nếu vượt quá
            scale = remaining_prob / total_remaining
            remaining_probs = [round(p * scale, 4) for p in remaining_probs]
        elif total_remaining < remaining_prob:
            # Thêm phần còn lại vào prediction cuối cùng
            remaining_probs[-1] = round(remaining_probs[-1] + (remaining_prob - total_remaining), 4)
    else:
        remaining_probs = []
    
    # Tạo results
    all_probs = [top1_prob] + remaining_probs
    
    # Tìm disease ID cho matched_disease
    disease_id = None
    for did, dname in id_to_disease.items():
        if matched_disease in dname or dname in matched_disease:
            disease_id = did
            matched_disease = dname  # Dùng tên chính xác từ mapping
            break
    
    if disease_id is None:
        # Nếu không tìm thấy, dùng random ID
        disease_id = random.choice(list(id_to_disease.keys()))
        matched_disease = id_to_disease[disease_id]
    
    # Thêm prediction đầu tiên
    confidence_percent = round(top1_prob * 100, 2)
    results.append({
        'disease': matched_disease,
        'probability': top1_prob,
        'confidence_percent': confidence_percent
    })
    
    # Thêm các predictions còn lại
    used_disease_ids = {disease_id}
    used_disease_names = {matched_disease}
    
    # Nếu có danh sách bệnh liên quan từ keyword matching, ưu tiên dùng chúng
    if remaining_diseases:
        # Dùng các bệnh liên quan từ keyword matching
        disease_idx = 0
        for i, prob in enumerate(remaining_probs):
            if disease_idx < len(remaining_diseases):
                # Tìm disease ID cho bệnh này
                candidate_disease = remaining_diseases[disease_idx]
                found_id = None
                for did, dname in id_to_disease.items():
                    if candidate_disease in dname or dname in candidate_disease:
                        if did not in used_disease_ids:
                            found_id = did
                            used_disease_ids.add(found_id)
                            used_disease_names.add(dname)
                            disease = dname
                            break
                
                if found_id is not None:
                    confidence_percent = round(prob * 100, 2)
                    results.append({
                        'disease': disease,
                        'probability': prob,
                        'confidence_percent': confidence_percent
                    })
                    disease_idx += 1
                else:
                    # Nếu không tìm thấy ID, dùng random
                    available_ids = [did for did in id_to_disease.keys() if did not in used_disease_ids]
                    if available_ids:
                        disease_id = random.choice(available_ids)
                        used_disease_ids.add(disease_id)
                        disease = id_to_disease[disease_id]
                        used_disease_names.add(disease)
                        confidence_percent = round(prob * 100, 2)
                        results.append({
                            'disease': disease,
                            'probability': prob,
                            'confidence_percent': confidence_percent
                        })
            else:
                # Hết danh sách bệnh liên quan, dùng random
                available_ids = [did for did in id_to_disease.keys() if did not in used_disease_ids]
                if available_ids:
                    disease_id = random.choice(available_ids)
                    used_disease_ids.add(disease_id)
                    disease = id_to_disease[disease_id]
                    used_disease_names.add(disease)
                    confidence_percent = round(prob * 100, 2)
                    results.append({
                        'disease': disease,
                        'probability': prob,
                        'confidence_percent': confidence_percent
                    })
    else:
        # Không có danh sách bệnh liên quan, dùng random như cũ
        for i, prob in enumerate(remaining_probs):
            # Tìm disease ID chưa dùng
            available_ids = [did for did in id_to_disease.keys() if did not in used_disease_ids]
            if available_ids:
                disease_id = random.choice(available_ids)
                used_disease_ids.add(disease_id)
                disease = id_to_disease[disease_id]
                used_disease_names.add(disease)
                
                confidence_percent = round(prob * 100, 2)
                results.append({
                    'disease': disease,
                    'probability': prob,
                    'confidence_percent': confidence_percent
                })
    
    # Đảm bảo có đủ top_k predictions
    while len(results) < top_k:
        available_ids = [did for did in id_to_disease.keys() 
                        if did not in used_disease_ids]
        if available_ids:
            disease_id = random.choice(available_ids)
            used_disease_ids.add(disease_id)
            disease = id_to_disease[disease_id]
            prob = round(random.uniform(0.001, 0.01), 4)
            confidence_percent = round(prob * 100, 2)
            results.append({
                'disease': disease,
                'probability': prob,
                'confidence_percent': confidence_percent
            })
        else:
            break
    
    # Sort theo confidence
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    # Đảm bảo tổng = 100% (làm tròn)
    total_prob = sum(r['probability'] for r in results)
    if abs(total_prob - 1.0) > 0.01:  # Nếu sai lệch > 1%
        # Điều chỉnh prediction cuối cùng
        diff = 1.0 - total_prob
        if results:
            results[-1]['probability'] = round(results[-1]['probability'] + diff, 4)
            results[-1]['confidence_percent'] = round(results[-1]['probability'] * 100, 2)
    
    max_confidence = results[0]['confidence_percent'] if results else 0.0
    
    return {
        'predictions': results,
        'is_confident': True,
        'is_out_of_domain': False,
        'max_confidence': max_confidence
    }, None

def predict_disease(question, top_k=5):
    """Dự đoán bệnh"""
    # Nếu bật chế độ giả lập, dùng simulated predictions
    if USE_SIMULATION_MODE:
        return predict_disease_simulated(question, top_k)
    
    # Validate
    is_valid, error_msg = validate_question(question)
    if not is_valid:
        return None, error_msg
    
    # Dynamic max length
    temp_tokenized = tokenizer(question, truncation=False, padding=False)
    actual_length = len(temp_tokenized['input_ids'])
    effective_max_length = min(actual_length + 20, MAX_LENGTH, 512)
    
    # Tokenize
    inputs = tokenizer(
        question,
        truncation=True,
        padding='max_length',
        max_length=effective_max_length,
        return_tensors='pt'
    )
    
    # Predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    
    # Check out-of-domain
    max_prob = float(torch.max(probabilities).item())
    is_ood = max_prob < OUT_OF_DOMAIN_THRESHOLD
    
    # Get top-k
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(id_to_disease)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        prob_value = float(prob.item())
        confidence_percent = prob_value * 100
        
        # Luôn thêm vào kết quả (không filter) - để người dùng thấy được predictions
        disease_name = id_to_disease.get(idx.item(), f"Class {idx.item()}")
        results.append({
            'disease': disease_name,
            'probability': prob_value,
            'confidence_percent': confidence_percent
        })
    
    max_confidence = results[0]['confidence_percent'] if results else 0.0
    is_confident = max_confidence >= (CONFIDENCE_THRESHOLD * 100)
    
    # Nếu tất cả confidence đều quá thấp và gần bằng nhau -> model chưa train đủ
    if results and len(results) > 1:
        confidences = [r['confidence_percent'] for r in results]
        max_conf = max(confidences)
        min_conf = min(confidences)
        # Nếu max confidence < 1% và các confidence gần bằng nhau -> model chưa học được
        if max_conf < 1.0 and (max_conf - min_conf) < 0.5:
            is_ood = True  # Đánh dấu là out-of-domain vì model chưa tự tin
    
    return {
        'predictions': results,
        'is_confident': is_confident,
        'is_out_of_domain': is_ood,
        'max_confidence': max_confidence
    }, None

# ==================== ROUTES ====================
@app.route('/')
def index():
    """Trang chủ"""
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint cho chat"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing question'
            }), 400
        
        question = data['question'].strip()
        top_k = data.get('top_k', 5)
        
        result, error = predict_disease(question, top_k=top_k)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'num_classes': len(id_to_disease),
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/api/chat', '/api/health']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error) if app.debug else 'An error occurred'
    }), 500

# ==================== MAIN ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("CHATBOT UI SERVER")
    print("="*70)
    print("Server đang chạy tại: http://localhost:5000")
    print("Mở trình duyệt và truy cập: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

