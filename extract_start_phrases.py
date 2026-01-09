"""
Script để trích xuất các cụm từ bắt đầu phổ biến từ CSV
"""
import pandas as pd
import re
import sys
import io
from collections import Counter

# Fix encoding cho Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Đọc CSV
df = pd.read_csv('ViMedical_Disease.csv', encoding='utf-8')
questions = df['Question'].astype(str).tolist()

# Trích xuất các cụm từ bắt đầu (2-5 từ đầu tiên)
starters = []
for q in questions:
    q_lower = q.lower().strip()
    # Loại bỏ dấu câu ở đầu
    q_lower = re.sub(r'^[^\w]+', '', q_lower)
    words = q_lower.split()
    
    # Lấy 2-5 từ đầu tiên
    for i in range(2, min(6, len(words) + 1)):
        starter = ' '.join(words[:i])
        if 5 <= len(starter) <= 40:  # Độ dài hợp lý
            starters.append(starter)

# Đếm tần suất
counter = Counter(starters)

# Lấy top 200 cụm từ phổ biến nhất
top_starters = [phrase for phrase, count in counter.most_common(200)]

# In ra để copy vào code
print("# Các cụm từ bắt đầu từ CSV (200 cụm từ phổ biến nhất):")
for i, phrase in enumerate(top_starters, 1):
    print(f"    '{phrase}',")
    if i % 10 == 0:
        print()

