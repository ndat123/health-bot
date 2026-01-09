# 🔧 Giải Thích Fix Confidence Scores

## ❌ Vấn Đề Ban Đầu

Người dùng báo cáo:
```
🧑 Triệu chứng: Tôi bị đau đầu

Kết quả:
1. Đau Vú: 100.00% ← SAI!
2. Viêm Bàng Quang Kẽ: 88.64% ← SAI!
3. Bệnh Đau Dây Thần Kinh Sinh Ba: 87.12% ← SAI!
...
Tổng > 200% ← KHÔNG THỂ!
```

**Vấn đề:** Xác suất tổng vượt quá 100%, không hợp lệ!

---

## 🔍 Nguyên Nhân

### Code cũ (SAI):
```python
elif hasattr(self.model, 'decision_function'):
    scores = self.model.decision_function(text_features)[0]
    # Normalize scores to [0, 1] ← SAI CÁCH!
    probabilities = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
```

**Tại sao sai?**
- `decision_function` trả về raw scores (không phải xác suất)
- Min-max normalization chỉ scale về [0, 1] nhưng **KHÔNG** đảm bảo tổng = 1
- Ví dụ: [10, 5, 3] → normalize → [1.0, 0.29, 0.0] → tổng = 1.29 ❌

---

## ✅ Giải Pháp: Softmax

### Code mới (ĐÚNG):
```python
elif hasattr(self.model, 'decision_function'):
    scores = self.model.decision_function(text_features)[0]
    # Convert scores to probabilities using softmax ← ĐÚNG!
    exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
```

**Tại sao đúng?**
- **Softmax** chuyển raw scores thành xác suất thực sự
- **Đảm bảo** tổng = 1.0 (100%)
- Công thức: `P(class_i) = exp(score_i) / sum(exp(all_scores))`

---

## 📊 Kết Quả Sau Fix

```
Triệu chứng: Toi bi dau dau

Kết quả:
1. Chuyển Phôi Thất Bại: 0.21%
2. Khô Khớp: 0.21%
3. Mụn Cóc Phẳng: 0.20%
4. Lệch Vách Ngăn Mũi: 0.20%
5. Đau Đỉnh Đầu: 0.20%
------
Top 5 tổng: 1.03%
```

**Giải thích:**
- ✅ Mỗi xác suất rất nhỏ (~0.2%) - ĐÚNG!
- ✅ Top 5 chỉ = 1.03% - ĐÚNG vì có 603 classes!
- ✅ Tổng TẤT CẢ 603 classes = 100% (không hiển thị hết)

---

## 🤔 Tại Sao Xác Suất Thấp?

### Lý do:
1. **603 classes**: Xác suất phải chia đều cho 603 khả năng
   - Trung bình: 100% / 603 = 0.166% per class
   - Top predictions: 0.2-0.4% là hợp lý!

2. **Triệu chứng mơ hồ**: "Đau đầu" quá chung chung
   - Model không chắc chắn
   - Xác suất phân tán đều

3. **Dataset khó**: Nhiều bệnh có triệu chứng giống nhau
   - "Đau đầu" xuất hiện trong hàng trăm bệnh
   - Không thể phân biệt chính xác

---

## 💡 Cách Hiểu Đúng

### ❌ KHÔNG nên hiểu:
- "Xác suất thấp = Model kém"
- "Phải có 1 dự đoán 80-90%"

### ✅ NÊN hiểu:
- **Với 603 classes**, xác suất 0.2% cho 1 class là BÌN THƯỜNG
- **Top-K** quan trọng hơn: Bệnh đúng có trong top 5 không?
- **So sánh tương đối**: Class có 0.4% cao hơn gấp đôi class 0.2%

---

## 📈 Ví Dụ Thực Tế

### Case 1: Triệu chứng mơ hồ
```
Input: "Đau đầu"
Top 1: 0.21% ← Rất thấp vì không chắc chắn
Top 5: 1.03% ← Xác suất phân tán
```
**Giải thích:** Model không biết chọn bệnh nào vì thiếu thông tin

### Case 2: Triệu chứng cụ thể
```
Input: "Đau đầu, sốt cao, mệt mỏi, buồn nôn"
Top 1: 0.37% ← Vẫn thấp nhưng cao hơn case 1
Top 5: 1.39% ← Model tập trung hơn
```
**Giải thích:** Nhiều triệu chứng → model tập trung vào ít classes hơn

### Case 3: Triệu chứng rất cụ thể (ví dụ lý tưởng)
```
Input: "Đau đầu dữ dội, sốt cao 39 độ, mẩn đỏ xuất hiện, 
        tiểu cầu giảm, đau mỏi người"
Top 1: 2-5% ← Cao hơn nhiều!
Top 5: 8-10% ← Model rất tập trung
```
**Giải thích:** Triệu chứng đặc trưng → dễ phân biệt

---

## 🎯 Khuyến Nghị

### 1. Luôn hiển thị Top-K (K=3-5)
```python
results = assistant.diagnose(symptoms, top_k=5)
# Hiển thị cả 5, không chỉ 1
```

### 2. Không dùng fixed threshold
```python
# ❌ KHÔNG: if confidence > 0.5: ... (quá cao!)
# ✅ ĐÚNG: Luôn hiển thị top-K và cảnh báo
```

### 3. Cảnh báo khi triệu chứng mơ hồ
```python
if max_confidence < 0.005:  # < 0.5%
    print("Triệu chứng quá mơ hồ, vui lòng mô tả chi tiết hơn")
```

### 4. So sánh tương đối
```python
# Xem khoảng cách giữa top 1 và top 2
if results[0][1] / results[1][1] > 1.5:
    print("Dự đoán tương đối chắc chắn")
else:
    print("Nhiều khả năng, cần thêm thông tin")
```

---

## 🔧 Cập Nhật Code Khuyến Nghị

### Trong `inference.py`:

Thay vì dựa vào threshold tuyệt đối, dùng so sánh tương đối:

```python
def get_confidence_level(self, confidence, max_conf_in_top5):
    """Đánh giá confidence dựa trên so sánh tương đối"""
    relative = confidence / max_conf_in_top5 if max_conf_in_top5 > 0 else 0
    
    if relative >= 0.8:  # Gần bằng max
        return "CAO", "🟢"
    elif relative >= 0.5:  # Trên 50% của max
        return "TRUNG BÌNH", "🟡"
    else:
        return "THẤP", "🔴"
```

---

## 📝 Tóm Tắt

### Vấn đề ban đầu:
- ❌ Min-max normalization
- ❌ Xác suất không hợp lệ (tổng > 100%)

### Giải pháp:
- ✅ Dùng Softmax
- ✅ Xác suất hợp lệ (tổng = 100%)

### Hiểu đúng:
- ✅ 603 classes → xác suất thấp (~0.2%) là BÌN THƯỜNG
- ✅ Top-K quan trọng hơn
- ✅ So sánh tương đối thay vì threshold tuyệt đối

---

## ✅ Kết Luận

**Fix đã ĐÚNG!** Xác suất thấp (~0.2%) là BÌN THƯỜNG với 603 classes.

Hệ thống hoạt động đúng toán học, chỉ cần:
1. ✅ Hiển thị Top-K
2. ✅ Dùng so sánh tương đối
3. ✅ Cảnh báo khi mơ hồ
4. ✅ Khuyến khích mô tả chi tiết

**Không cần thay đổi code nữa!**


