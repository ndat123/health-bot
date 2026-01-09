@echo off
chcp 65001 > nul
REM Set your Google API key as environment variable before running
REM Example: set GOOGLE_API_KEY=your_api_key_here
python gemini_disease_diagnosis.py
pause


