"""
Flask API Server để expose model AI cho PHP
Endpoints:
- POST /predict - Dự đoán bệnh từ câu hỏi
- POST /predict_batch - Dự đoán nhiều câu hỏi cùng lúc
- GET /health - Kiểm tra server status
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Cho phép CORS để PHP có thể gọi

# Cấu hình
MODEL_PATH = "./chatbot_model"  # Đường dẫn đến model
MAX_LENGTH = 256
DYNAMIC_MAX_LENGTH = True  # Tự động điều chỉnh max_length cho câu hỏi dài
CONFIDENCE_THRESHOLD = 0.15
OUT_OF_DOMAIN_THRESHOLD = 0.10  # Ngưỡng để phát hiện câu hỏi không liên quan
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Từ khóa y tế để validate
MEDICAL_KEYWORDS = [
    'triệu chứng', 'bệnh', 'đau', 'mệt', 'sốt', 'ho', 'khó thở',
    'buồn nôn', 'chóng mặt', 'nhức đầu', 'mất ngủ', 'ngứa', 'sưng',
    'nôn', 'tiêu chảy', 'táo bón', 'đau bụng', 'khó chịu', 'yếu',
    'suy nhược', 'phù', 'đỏ', 'nóng', 'lạnh', 'run', 'co giật'
]

# Global variables
tokenizer = None
model = None
id_to_disease = None

def load_model():
    """Load model và mapping"""
    global tokenizer, model, id_to_disease
    
    try:
        logger.info(f"Đang load model từ: {MODEL_PATH}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        model.to(DEVICE)
        
        # Load disease mapping
        mapping_path = os.path.join(MODEL_PATH, "disease_mapping.json")
        if not os.path.exists(mapping_path):
            # Fallback
            mapping_path = "./chatbot_model/disease_mapping.json"
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            id_to_disease = {int(k): v for k, v in mapping["id_to_disease"].items()}
        
        logger.info(f"✓ Model loaded: {len(id_to_disease)} classes")
        logger.info(f"✓ Device: {DEVICE}")
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi load model: {str(e)}")
        return False

def validate_question(question):
    """
    Validate câu hỏi đầu vào
    
    Returns:
        (is_valid, error_message)
    """
    question = str(question).strip()
    
    if not question or len(question) < 10:
        return False, "Câu hỏi quá ngắn. Vui lòng mô tả chi tiết hơn về triệu chứng."
    
    if len(question) > 2000:
        return False, "Câu hỏi quá dài. Vui lòng rút gọn lại."
    
    # Kiểm tra có chứa từ khóa y tế không
    question_lower = question.lower()
    has_medical_keyword = any(keyword in question_lower for keyword in MEDICAL_KEYWORDS)
    
    if not has_medical_keyword:
        return False, "Câu hỏi không liên quan đến triệu chứng y tế. Vui lòng mô tả các triệu chứng bạn đang gặp phải."
    
    return True, None

def is_out_of_domain(probabilities, threshold=OUT_OF_DOMAIN_THRESHOLD):
    """
    Phát hiện câu hỏi không liên quan đến bệnh (out-of-domain)
    
    Args:
        probabilities: Probability distribution
        threshold: Ngưỡng để xác định out-of-domain
    
    Returns:
        (is_out_of_domain, max_prob)
    """
    max_prob = float(torch.max(probabilities).item())
    is_ood = max_prob < threshold
    
    return is_ood, max_prob

def extract_important_tokens(question, tokenizer, model, top_n=10):
    """
    Trích xuất các từ quan trọng nhất (feature importance)
    Sử dụng gradient-based importance
    
    Returns:
        list of (token, importance_score)
    """
    try:
        # Tokenize
        inputs = tokenizer(question, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Enable gradient
        inputs['input_ids'].requires_grad = True
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get max probability
        probs = torch.softmax(logits, dim=-1)
        max_prob, max_idx = torch.max(probs, dim=1)
        
        # Backward pass
        max_prob.backward()
        
        # Get gradients
        gradients = inputs['input_ids'].grad
        if gradients is None:
            return []
        
        # Calculate importance (absolute gradient)
        importance = torch.abs(gradients[0])
        
        # Get tokens
        token_ids = inputs['input_ids'][0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Combine and sort
        token_importance = list(zip(tokens, importance.cpu().numpy()))
        token_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out special tokens and get top N
        important_tokens = []
        for token, score in token_importance:
            if token not in ['[CLS]', '[SEP]', '[PAD]', '<unk>'] and not token.startswith('##'):
                important_tokens.append({
                    'token': token.replace('##', ''),
                    'importance': float(score)
                })
                if len(important_tokens) >= top_n:
                    break
        
        return important_tokens
    
    except Exception as e:
        logger.warning(f"Không thể extract important tokens: {str(e)}")
        return []

def predict_disease(question, top_k=TOP_K, confidence_threshold=CONFIDENCE_THRESHOLD, 
                   include_explanation=False):
    """
    Dự đoán bệnh từ câu hỏi với các cải tiến
    
    Args:
        question: Câu hỏi về triệu chứng
        top_k: Số lượng bệnh trả về
        confidence_threshold: Ngưỡng confidence
        include_explanation: Có trả về explanation không
    
    Returns:
        dict với predictions và metadata
    """
    if tokenizer is None or model is None:
        raise Exception("Model chưa được load")
    
    # Validate question
    is_valid, error_msg = validate_question(question)
    if not is_valid:
        return {
            'error': error_msg,
            'is_valid': False,
            'question': question
        }
    
    # Tính max_length động nếu cần
    if DYNAMIC_MAX_LENGTH:
        # Tokenize để tính độ dài
        temp_tokenized = tokenizer(question, truncation=False, padding=False)
        actual_length = len(temp_tokenized['input_ids'])
        effective_max_length = min(actual_length + 20, MAX_LENGTH, 512)  # Tối đa 512
    else:
        effective_max_length = MAX_LENGTH
    
    # Tokenize
    inputs = tokenizer(
        question,
        truncation=True,
        padding='max_length',
        max_length=effective_max_length,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    
    # Check out-of-domain
    is_ood, max_prob = is_out_of_domain(probabilities)
    
    # Get top-k
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(id_to_disease)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        disease_name = id_to_disease[idx.item()]
        predictions.append({
            'disease': disease_name,
            'probability': float(prob.item()),
            'confidence_percent': float(prob.item() * 100)
        })
    
    max_confidence = predictions[0]['probability']
    is_confident = max_confidence >= confidence_threshold
    
    result = {
        'predictions': predictions,
        'is_confident': is_confident,
        'is_out_of_domain': is_ood,
        'max_confidence': float(max_confidence),
        'confidence_threshold': confidence_threshold,
        'question': question,
        'is_valid': True,
        'timestamp': datetime.now().isoformat()
    }
    
    # Thêm explanation nếu được yêu cầu
    if include_explanation:
        important_tokens = extract_important_tokens(question, tokenizer, model)
        result['explanation'] = {
            'important_tokens': important_tokens,
            'message': f"Các từ khóa quan trọng nhất: {', '.join([t['token'] for t in important_tokens[:5]])}"
        }
    
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Kiểm tra server status"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': DEVICE,
        'num_classes': len(id_to_disease) if id_to_disease else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Dự đoán bệnh từ câu hỏi với validation và explanation
    
    Request body (JSON):
    {
        "question": "Tôi đang cảm thấy mệt mỏi, chóng mặt...",
        "top_k": 3,  // optional, default 3
        "confidence_threshold": 0.15,  // optional, default 0.15
        "include_explanation": false  // optional, default false
    }
    
    Response (JSON):
    {
        "success": true,
        "data": {
            "predictions": [...],
            "is_confident": true,
            "is_out_of_domain": false,
            "explanation": {...},  // nếu include_explanation=true
            ...
        }
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: question'
            }), 400
        
        question = data['question'].strip()
        
        # Get optional parameters
        top_k = data.get('top_k', TOP_K)
        confidence_threshold = data.get('confidence_threshold', CONFIDENCE_THRESHOLD)
        include_explanation = data.get('include_explanation', False)
        
        # Predict (validation được thực hiện bên trong)
        result = predict_disease(question, top_k, confidence_threshold, include_explanation)
        
        # Kiểm tra nếu có lỗi validation
        if not result.get('is_valid', True):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Invalid question'),
                'data': result
            }), 400
        
        # Kiểm tra out-of-domain
        if result.get('is_out_of_domain', False):
            return jsonify({
                'success': True,
                'warning': 'Câu hỏi có thể không liên quan đến bệnh. Độ tin cậy rất thấp.',
                'data': result
            })
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        logger.error(f"Lỗi khi predict: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Dự đoán nhiều câu hỏi cùng lúc
    
    Request body (JSON):
    {
        "questions": [
            "Câu hỏi 1",
            "Câu hỏi 2",
            ...
        ],
        "top_k": 3,  // optional
        "confidence_threshold": 0.15  // optional
    }
    
    Response (JSON):
    {
        "success": true,
        "data": [
            {
                "question": "Câu hỏi 1",
                "predictions": [...],
                ...
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'questions' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: questions'
            }), 400
        
        questions = data['questions']
        
        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({
                'success': False,
                'error': 'questions must be a non-empty array'
            }), 400
        
        if len(questions) > 100:  # Limit batch size
            return jsonify({
                'success': False,
                'error': 'Maximum 100 questions per batch'
            }), 400
        
        top_k = data.get('top_k', TOP_K)
        confidence_threshold = data.get('confidence_threshold', CONFIDENCE_THRESHOLD)
        
        results = []
        for question in questions:
            try:
                result = predict_disease(question.strip(), top_k, confidence_threshold)
                results.append(result)
            except Exception as e:
                results.append({
                    'question': question,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'success': True,
            'data': results,
            'total': len(results)
        })
    
    except Exception as e:
        logger.error(f"Lỗi khi predict batch: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/info', methods=['GET'])
def get_info():
    """Lấy thông tin về model"""
    return jsonify({
        'model_path': MODEL_PATH,
        'device': DEVICE,
        'max_length': MAX_LENGTH,
        'num_classes': len(id_to_disease) if id_to_disease else 0,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'default_top_k': TOP_K
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Load model khi start server
    if not load_model():
        logger.error("Không thể load model. Server sẽ không hoạt động.")
        exit(1)
    
    # Start server
    logger.info("="*70)
    logger.info("API SERVER ĐÃ SẴN SÀNG")
    logger.info("="*70)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Endpoints:")
    logger.info(f"  - GET  /health - Health check")
    logger.info(f"  - POST /predict - Single prediction")
    logger.info(f"  - POST /predict_batch - Batch prediction")
    logger.info(f"  - GET  /info - Model info")
    logger.info("="*70)
    
    # Run server
    # host='0.0.0.0' để có thể truy cập từ ngoài
    # port=5000 là mặc định, có thể thay đổi
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

