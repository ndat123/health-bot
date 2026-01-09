"""
Script để tạo fine-tuning dataset cho Google AI Studio
Format: JSONL (JSON Lines) - mỗi line là một training example
"""
import pandas as pd
import json
import sys
import io

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_gemini_training_dataset():
    """
    Tạo dataset cho Google AI Studio fine-tuning
    Format: JSONL với messages array (user + model)
    """
    print("=" * 70)
    print("Creating Google AI Studio Fine-tuning Dataset")
    print("=" * 70)
    
    # Load CSV
    df = pd.read_csv('ViMedical_Disease.csv', encoding='utf-8')
    print(f"\nLoaded {len(df)} rows from CSV")
    
    # System instruction for model behavior
    system_instruction = """Bạn là bác sĩ AI chuyên chẩn đoán bệnh dựa trên triệu chứng.
Nhiệm vụ: Phân tích triệu chứng và trả về TÊN BỆNH chính xác.
Chỉ trả về tên bệnh, không giải thích thêm."""
    
    # Create JSONL format
    training_examples = []
    
    for idx, row in df.iterrows():
        disease = row['Disease'].strip()
        question = row['Question'].strip()
        
        # Format: Google AI Studio expects "contents" with "role" and "parts"
        example = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": question}]
                },
                {
                    "role": "model",
                    "parts": [{"text": disease}]
                }
            ]
        }
        
        training_examples.append(example)
    
    # Save to JSONL file
    output_file = 'gemini_training_dataset.jsonl'
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
    print(f"  - Avg examples per disease: {len(training_examples) / len(diseases):.1f}")
    
    # Show sample
    print(f"\nSample training example:")
    print("-" * 70)
    sample = training_examples[0]
    print(f"User: {sample['contents'][0]['parts'][0]['text'][:100]}...")
    print(f"Model: {sample['contents'][1]['parts'][0]['text']}")
    print("-" * 70)
    
    return output_file

def create_system_instruction_file():
    """
    Tạo file system instruction riêng cho Google AI Studio
    """
    system_instruction = """Bạn là bác sĩ AI chuyên khoa Tai Mũi Họng (ENT) với khả năng chẩn đoán bệnh dựa trên triệu chứng.

NHIỆM VỤ:
- Phân tích triệu chứng người dùng mô tả
- Chẩn đoán bệnh chính xác nhất
- Đưa ra lời khuyên y tế phù hợp

QUY TẮC:
1. Chỉ chẩn đoán các bệnh đã được training (35 bệnh ENT)
2. Trả lời bằng tiếng Việt
3. Nếu không chắc chắn, khuyên người dùng đi khám bác sĩ
4. Không đưa ra chẩn đoán sai lệch hoặc nguy hiểm

FORMAT TRẢ LỜI:
- Tên bệnh
- Triệu chứng chính
- Nguyên nhân có thể
- Cách điều trị/chữa
- Khi nào cần đi khám GẤP

AN TOÀN:
- Không thay thế bác sĩ thực
- Khuyến khích đi khám nếu triệu chứng nghiêm trọng
- Không tư vấn thuốc cụ thể (chỉ nói chung)"""
    
    with open('system_instruction.txt', 'w', encoding='utf-8') as f:
        f.write(system_instruction)
    
    print(f"\nSystem instruction saved to: system_instruction.txt")

def validate_dataset():
    """
    Validate JSONL format
    """
    print("\n" + "=" * 70)
    print("Validating Dataset Format")
    print("=" * 70)
    
    with open('gemini_training_dataset.jsonl', 'r', encoding='utf-8') as f:
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
    output_file = create_gemini_training_dataset()
    create_system_instruction_file()
    validate_dataset()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Go to: https://aistudio.google.com/")
    print("2. Click 'Create new' > 'Tuned model'")
    print("3. Upload: gemini_training_dataset.jsonl")
    print("4. Add system instruction from: system_instruction.txt")
    print("5. Configure:")
    print("   - Model: Gemini 2.0 Flash")
    print("   - Epochs: 5")
    print("   - Batch size: 16")
    print("6. Start tuning (~20-30 minutes)")
    print("7. Copy the model name and add to web_app_gemini.py")
    print("=" * 70)
    print("\n")


