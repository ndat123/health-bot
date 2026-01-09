# 🔐 Cấu Hình Environment Variables

## ⚠️ QUAN TRỌNG: API Keys

**KHÔNG BAO GIỜ commit API keys lên GitHub!**

File này hướng dẫn cách cấu hình API keys một cách an toàn.

---

## 📋 Cách 1: Environment Variables (Khuyến nghị)

### Windows (PowerShell):
```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

### Windows (CMD):
```cmd
set GROQ_API_KEY=your_groq_api_key_here
set GEMINI_API_KEY=your_gemini_api_key_here
```

### Linux/Mac:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
```

---

## 📋 Cách 2: File .env (Tự động load)

1. Tạo file `.env` trong thư mục gốc:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

2. Cài đặt `python-dotenv`:
```bash
pip install python-dotenv
```

3. Thêm vào đầu file `web_app_gemini.py`:
```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file
```

**Lưu ý:** File `.env` đã có trong `.gitignore`, sẽ không bị commit.

---

## 🔑 Lấy API Keys

### Groq API Key:
1. Đăng ký tại: https://console.groq.com/
2. Tạo API key mới
3. Copy và set vào environment variable

### Google Gemini API Key:
1. Đăng ký tại: https://aistudio.google.com/
2. Tạo API key mới
3. Copy và set vào environment variable

---

## ✅ Kiểm Tra

Sau khi set environment variables, chạy:
```bash
python web_app_gemini.py
```

Nếu thấy:
- ✅ `✓ Gemini API configured successfully` → OK
- ⚠️ `WARNING: API_KEY not set` → Chưa set đúng

---

## 🚨 Security Best Practices

1. ✅ **Dùng environment variables** thay vì hardcode
2. ✅ **Không commit** `.env` file
3. ✅ **Không share** API keys trong code
4. ✅ **Rotate keys** định kỳ nếu bị lộ
5. ✅ **Sử dụng** `.gitignore` để bảo vệ

---

## 📝 Quick Start

```bash
# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
$env:GEMINI_API_KEY="AIzaSy_your_key_here"
python web_app_gemini.py

# Linux/Mac
export GROQ_API_KEY="gsk_your_key_here"
export GEMINI_API_KEY="AIzaSy_your_key_here"
python web_app_gemini.py
```

---

**🔒 Bảo vệ API keys của bạn!**

