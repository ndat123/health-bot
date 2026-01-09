"""
Script để thực hiện data augmentation cho dataset ViMedical_Disease
Các kỹ thuật:
- Synonym replacement (thay thế từ đồng nghĩa)
- Random insertion (chèn từ ngẫu nhiên)
- Random swap (hoán đổi vị trí từ)
- Random deletion (xóa từ ngẫu nhiên)
"""

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import json

# Từ điển đồng nghĩa tiếng Việt cho các triệu chứng y tế phổ biến
SYNONYMS = {
    'đau': ['nhức', 'đau nhức', 'đau đớn', 'khó chịu'],
    'mệt': ['mệt mỏi', 'uể oải', 'kiệt sức', 'mỏi mệt'],
    'chóng mặt': ['hoa mắt', 'choáng váng', 'váng đầu', 'đầu óc quay cuồng'],
    'sốt': ['nóng', 'sốt cao', 'bị sốt', 'có sốt'],
    'ho': ['ho khan', 'ho có đờm', 'bị ho', 'hay ho'],
    'đau đầu': ['nhức đầu', 'đau ở đầu', 'đau vùng đầu'],
    'buồn nôn': ['nôn', 'muốn nôn', 'cảm giác buồn nôn'],
    'khó thở': ['thở khó', 'khó hít thở', 'thở gấp', 'khó thở'],
    'mất ngủ': ['khó ngủ', 'không ngủ được', 'ngủ không được'],
    'đau bụng': ['nhức bụng', 'đau ở bụng', 'bụng đau'],
    'tiêu chảy': ['đi ngoài nhiều', 'phân lỏng', 'ỉa chảy'],
    'táo bón': ['khó đi ngoài', 'không đi được', 'đại tiện khó'],
    'ngứa': ['ngứa ngáy', 'swelling', 'bị ngứa'],
    'sưng': ['phù', 'phù nề', 'sưng lên'],
    'đỏ': ['đỏ lên', 'ửng đỏ', 'bị đỏ'],
    'yếu': ['suy nhược', 'mệt mỏi', 'không có sức'],
    'khó chịu': ['bất tiện', 'không thoải mái', 'cảm thấy khó chịu'],
}

# Các cụm từ bắt đầu câu hỏi
QUESTION_STARTERS = [
    'Tôi đang',
    'Tôi hay',
    'Tôi thường',
    'Tôi cảm thấy',
    'Gần đây tôi',
    'Dạo này tôi',
    'Hiện tại tôi',
    'Tôi bị',
]

# Các cụm từ kết thúc câu hỏi
QUESTION_ENDERS = [
    'Tôi có thể đang bị bệnh gì?',
    'Tôi bị bệnh gì?',
    'Đây có thể là bệnh gì?',
    'Có thể tôi đang mắc bệnh gì?',
    'Tôi đang mắc bệnh gì?',
]

def synonym_replacement(text, n=1):
    """
    Thay thế n từ trong câu bằng từ đồng nghĩa
    
    Args:
        text: Câu gốc
        n: Số từ cần thay thế
    
    Returns:
        Câu đã được thay thế
    """
    words = text.split()
    
    # Tìm các từ có thể thay thế
    replaceable_words = []
    for i, word in enumerate(words):
        word_lower = word.lower()
        for key in SYNONYMS.keys():
            if key in word_lower:
                replaceable_words.append((i, key))
    
    if not replaceable_words:
        return text
    
    # Chọn ngẫu nhiên n từ để thay thế
    n = min(n, len(replaceable_words))
    words_to_replace = random.sample(replaceable_words, n)
    
    new_words = words.copy()
    for idx, original_word in words_to_replace:
        synonyms = SYNONYMS[original_word]
        new_word = random.choice(synonyms)
        # Thay thế trong câu
        new_words[idx] = new_words[idx].lower().replace(original_word, new_word)
    
    return ' '.join(new_words)

def random_insertion(text, n=1):
    """
    Chèn ngẫu nhiên n từ đồng nghĩa vào câu
    
    Args:
        text: Câu gốc
        n: Số từ cần chèn
    
    Returns:
        Câu đã được chèn thêm từ
    """
    words = text.split()
    
    # Lấy danh sách tất cả các từ đồng nghĩa
    all_synonyms = []
    for synonyms_list in SYNONYMS.values():
        all_synonyms.extend(synonyms_list)
    
    if not all_synonyms:
        return text
    
    for _ in range(n):
        synonym = random.choice(all_synonyms)
        random_idx = random.randint(0, len(words))
        words.insert(random_idx, synonym)
    
    return ' '.join(words)

def random_swap(text, n=1):
    """
    Hoán đổi ngẫu nhiên vị trí của n cặp từ
    
    Args:
        text: Câu gốc
        n: Số cặp từ cần hoán đổi
    
    Returns:
        Câu đã được hoán đổi
    """
    words = text.split()
    
    if len(words) < 2:
        return text
    
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return ' '.join(words)

def random_deletion(text, p=0.1):
    """
    Xóa ngẫu nhiên các từ với xác suất p
    
    Args:
        text: Câu gốc
        p: Xác suất xóa mỗi từ
    
    Returns:
        Câu đã được xóa bớt từ
    """
    words = text.split()
    
    if len(words) == 1:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    # Đảm bảo ít nhất còn 1 từ
    if len(new_words) == 0:
        return random.choice(words)
    
    return ' '.join(new_words)

def change_question_format(text):
    """
    Thay đổi cách đặt câu hỏi (đầu và cuối câu)
    
    Args:
        text: Câu hỏi gốc
    
    Returns:
        Câu hỏi với format mới
    """
    # Tìm phần giữa (triệu chứng)
    # Giả sử format: "Starter ... symptoms ... Ender"
    
    # Tìm starter
    starter_found = None
    for starter in QUESTION_STARTERS:
        if text.startswith(starter):
            starter_found = starter
            break
    
    # Tìm ender
    ender_found = None
    for ender in QUESTION_ENDERS:
        if text.endswith(ender):
            ender_found = ender
            break
    
    if starter_found and ender_found:
        # Lấy phần giữa
        middle = text[len(starter_found):len(text)-len(ender_found)].strip()
        
        # Chọn starter và ender mới
        new_starter = random.choice(QUESTION_STARTERS)
        new_ender = random.choice(QUESTION_ENDERS)
        
        return f"{new_starter} {middle} {new_ender}"
    
    return text

def augment_text(text, num_aug=1, techniques=['synonym', 'format']):
    """
    Tạo các phiên bản augmented của text
    
    Args:
        text: Text gốc
        num_aug: Số phiên bản augmented cần tạo
        techniques: Danh sách các kỹ thuật sử dụng
    
    Returns:
        List các text đã augmented
    """
    augmented_texts = []
    
    for _ in range(num_aug):
        aug_text = text
        
        # Áp dụng các kỹ thuật ngẫu nhiên
        if 'synonym' in techniques and random.random() > 0.3:
            aug_text = synonym_replacement(aug_text, n=random.randint(1, 2))
        
        if 'format' in techniques and random.random() > 0.5:
            aug_text = change_question_format(aug_text)
        
        if 'swap' in techniques and random.random() > 0.7:
            aug_text = random_swap(aug_text, n=1)
        
        if 'delete' in techniques and random.random() > 0.8:
            aug_text = random_deletion(aug_text, p=0.05)
        
        # Chỉ thêm nếu khác với text gốc
        if aug_text != text and aug_text not in augmented_texts:
            augmented_texts.append(aug_text)
    
    return augmented_texts

def augment_dataset(input_file, output_file, augment_per_sample=2, 
                    min_samples_per_class=30):
    """
    Augment toàn bộ dataset, tập trung vào các classes có ít mẫu
    
    Args:
        input_file: File CSV đầu vào
        output_file: File CSV đầu ra
        augment_per_sample: Số mẫu augmented cho mỗi mẫu gốc
        min_samples_per_class: Số mẫu tối thiểu cho mỗi class
    """
    print("="*70)
    print("DATA AUGMENTATION")
    print("="*70)
    
    # Đọc dữ liệu
    print("\n1. Đọc dữ liệu...")
    df = pd.read_csv(input_file)
    print(f"   ✓ Đã đọc {len(df)} mẫu")
    
    # Thống kê
    print("\n2. Thống kê ban đầu:")
    disease_counts = df['Disease'].value_counts()
    print(f"   - Số loại bệnh: {len(disease_counts)}")
    print(f"   - Trung bình: {disease_counts.mean():.1f} mẫu/bệnh")
    print(f"   - Min: {disease_counts.min()} mẫu")
    print(f"   - Max: {disease_counts.max()} mẫu")
    
    # Tìm các classes cần augment
    classes_need_aug = disease_counts[disease_counts < min_samples_per_class]
    print(f"\n3. Classes cần augmentation: {len(classes_need_aug)}")
    print(f"   (Các bệnh có < {min_samples_per_class} mẫu)")
    
    # Augment
    print("\n4. Đang thực hiện augmentation...")
    augmented_data = []
    
    for disease in tqdm(classes_need_aug.index, desc="Augmenting"):
        disease_samples = df[df['Disease'] == disease]
        current_count = len(disease_samples)
        needed = min_samples_per_class - current_count
        
        # Tính số lần augment cho mỗi mẫu
        aug_per_sample = max(1, needed // current_count + 1)
        
        for _, row in disease_samples.iterrows():
            original_question = row['Question']
            
            # Tạo augmented versions
            aug_questions = augment_text(
                original_question,
                num_aug=aug_per_sample,
                techniques=['synonym', 'format', 'swap']
            )
            
            # Thêm vào danh sách
            for aug_q in aug_questions[:needed]:  # Chỉ lấy đủ số lượng cần
                augmented_data.append({
                    'Disease': disease,
                    'Question': aug_q,
                    'is_augmented': True,
                    'original_question': original_question
                })
                needed -= 1
                
                if needed <= 0:
                    break
            
            if needed <= 0:
                break
    
    print(f"   ✓ Đã tạo {len(augmented_data)} mẫu augmented")
    
    # Thêm cột is_augmented cho dữ liệu gốc
    df['is_augmented'] = False
    df['original_question'] = df['Question']
    
    # Kết hợp dữ liệu
    print("\n5. Kết hợp dữ liệu...")
    augmented_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Lưu file
    print(f"\n6. Lưu dữ liệu...")
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"   ✓ Đã lưu tại: {output_file}")
    
    # Thống kê sau augmentation
    print("\n7. Thống kê sau augmentation:")
    new_disease_counts = combined_df['Disease'].value_counts()
    print(f"   - Tổng số mẫu: {len(combined_df)} (+{len(augmented_data)})")
    print(f"   - Trung bình: {new_disease_counts.mean():.1f} mẫu/bệnh")
    print(f"   - Min: {new_disease_counts.min()} mẫu")
    print(f"   - Max: {new_disease_counts.max()} mẫu")
    
    # Lưu thống kê
    stats = {
        'original_samples': len(df),
        'augmented_samples': len(augmented_data),
        'total_samples': len(combined_df),
        'num_classes': len(disease_counts),
        'classes_augmented': len(classes_need_aug),
        'original_stats': {
            'mean': float(disease_counts.mean()),
            'min': int(disease_counts.min()),
            'max': int(disease_counts.max()),
            'std': float(disease_counts.std())
        },
        'augmented_stats': {
            'mean': float(new_disease_counts.mean()),
            'min': int(new_disease_counts.min()),
            'max': int(new_disease_counts.max()),
            'std': float(new_disease_counts.std())
        }
    }
    
    stats_file = output_file.replace('.csv', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"   ✓ Thống kê đã lưu tại: {stats_file}")
    
    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)
    
    return combined_df

def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Augmentation cho ViMedical_Disease')
    parser.add_argument('--input', type=str, default='ViMedical_Disease.csv',
                        help='File CSV đầu vào')
    parser.add_argument('--output', type=str, default='ViMedical_Disease_augmented.csv',
                        help='File CSV đầu ra')
    parser.add_argument('--min-samples', type=int, default=30,
                        help='Số mẫu tối thiểu cho mỗi class')
    parser.add_argument('--aug-per-sample', type=int, default=2,
                        help='Số mẫu augmented cho mỗi mẫu gốc')
    
    args = parser.parse_args()
    
    # Kiểm tra file input
    import os
    if not os.path.exists(args.input):
        print(f"❌ Không tìm thấy file: {args.input}")
        return
    
    # Thực hiện augmentation
    augment_dataset(
        input_file=args.input,
        output_file=args.output,
        augment_per_sample=args.aug_per_sample,
        min_samples_per_class=args.min_samples
    )

if __name__ == "__main__":
    main()









