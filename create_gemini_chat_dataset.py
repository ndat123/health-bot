"""
Tạo dataset cho Gemini Chatbot - Format hội thoại tự nhiên
"""
import pandas as pd
import json
import sys
import io

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_conversational_dataset():
    """
    Tạo dataset chat tự nhiên cho Gemini fine-tuning
    """
    print("=" * 70)
    print("Creating Gemini Chatbot Dataset (Conversational Style)")
    print("=" * 70)
    
    # Load CSV
    df = pd.read_csv('ViMedical_Disease.csv', encoding='utf-8')
    print(f"\nLoaded {len(df)} rows from CSV")
    
    # System instruction for chatbot behavior
    system_instruction = """Bạn là bác sĩ AI thân thiện, chuyên tư vấn sức khỏe qua chat.

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
5. Động viên và nhắc nhở đi khám"""
    
    # Create conversational training examples
    training_examples = []
    
    # Nhóm theo bệnh để tạo response phong phú hơn
    diseases_grouped = df.groupby('Disease')
    
    for disease, group in diseases_grouped:
        # Lấy tất cả triệu chứng của bệnh này
        symptoms = group['Question'].tolist()
        
        # Tạo response theo phong cách chat tự nhiên
        for symptom in symptoms[:10]:  # Limit to 10 samples per disease
            # Remove "Tôi có thể đang bị bệnh gì?" từ câu hỏi
            clean_symptom = symptom.replace("Tôi có thể đang bị bệnh gì?", "").strip()
            
            # Tạo response tự nhiên
            response = f"""Dựa vào các triệu chứng bạn mô tả, bạn có thể đang gặp phải Hội chứng **{disease}**.

**Về bệnh này:**
{disease} là một tình trạng sức khỏe mà các triệu chứng bạn đang gặp phải là khá điển hình. Tình trạng này cần được chẩn đoán và điều trị đúng cách để tránh các biến chứng không mong muốn.

**Các triệu chứng thường gặp:**
{clean_symptom}

**Lời khuyên:**
Tuy nhiên, để xác định chính xác bạn đang mắc bệnh gì, bạn cần đến gặp bác sĩ để được chẩn đoán. Bác sĩ sẽ hỏi bạn về các triệu chứng, khám sức khỏe cho bạn và có thể yêu cầu bạn làm một số xét nghiệm để xác định chính xác nguyên nhân của các triệu chứng.

**Hãy nhớ rằng:**
Việc tự chẩn đoán bệnh là điều không nên. Luôn luôn tham khảo ý kiến bác sĩ để có được chẩn đoán và phương pháp điều trị phù hợp nhất cho bạn.

Chúc bạn sớm khỏe lại! 💙"""

            # Format for Google AI Studio
            example = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": clean_symptom}]
                    },
                    {
                        "role": "model",
                        "parts": [{"text": response}]
                    }
                ]
            }
            
            training_examples.append(example)
    
    # Save to JSONL file
    output_file = 'gemini_chatbot_dataset.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\nCreated {len(training_examples)} training examples")
    print(f"Saved to: {output_file}")
    
    # Statistics
    diseases = df['Disease'].unique()
    print(f"\nDataset Statistics:")
    print(f"  - Total examples: {len(training_examples)}")
    print(f"  - Unique diseases: {len(diseases)}")
    print(f"  - Format: Conversational (Chat-style)")
    
    # Show sample
    print(f"\nSample training example:")
    print("-" * 70)
    sample = training_examples[0]
    print(f"User: {sample['contents'][0]['parts'][0]['text'][:100]}...")
    print(f"\nModel: {sample['contents'][1]['parts'][0]['text'][:200]}...")
    print("-" * 70)
    
    # Save system instruction
    with open('system_instruction_chatbot.txt', 'w', encoding='utf-8') as f:
        f.write(system_instruction)
    
    print(f"\nSystem instruction saved to: system_instruction_chatbot.txt")
    
    return output_file

def validate_dataset():
    """
    Validate JSONL format
    """
    print("\n" + "=" * 70)
    print("Validating Dataset Format")
    print("=" * 70)
    
    with open('gemini_chatbot_dataset.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    valid = 0
    invalid = 0
    
    for i, line in enumerate(lines[:10]):  # Check first 10
        try:
            data = json.loads(line)
            assert 'contents' in data
            assert len(data['contents']) == 2
            assert data['contents'][0]['role'] == 'user'
            assert data['contents'][1]['role'] == 'model'
            valid += 1
        except Exception as e:
            print(f"Invalid line {i+1}: {e}")
            invalid += 1
    
    print(f"\nValidation Result:")
    print(f"  - Valid: {valid}/10")
    print(f"  - Invalid: {invalid}/10")
    
    if invalid == 0:
        print("\n✓ Dataset format is CORRECT")
    else:
        print("\n✗ Dataset has errors, please fix")

if __name__ == '__main__':
    print("\n")
    output_file = create_conversational_dataset()
    validate_dataset()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Go to: https://aistudio.google.com/")
    print("2. Click 'Create new' > 'Tuned model'")
    print("3. Upload: gemini_chatbot_dataset.jsonl")
    print("4. Add system instruction from: system_instruction_chatbot.txt")
    print("5. Configure:")
    print("   - Model: Gemini 2.0 Flash")
    print("   - Epochs: 5")
    print("   - Batch size: 16")
    print("   - Temperature: 0.7 (more natural)")
    print("6. Start tuning (~20-30 minutes)")
    print("7. Copy the model name and add to web_app_gemini.py")
    print("=" * 70)
    print("\n")


