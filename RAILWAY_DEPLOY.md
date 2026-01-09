# 🚂 Railway Deployment Guide

## 📋 Files Created for Railway

1. **`Procfile`** - Tells Railway how to start your app
2. **`railway.json`** - Railway configuration
3. **`runtime.txt`** - Python version
4. **`requirements.txt`** - Updated with gunicorn

## 🔧 Configuration

### Procfile
```
web: gunicorn --bind 0.0.0.0:$PORT web_app_gemini:app
```

### Environment Variables

Set these in Railway dashboard:

1. **GROQ_API_KEY** - Your Groq API key
2. **GEMINI_API_KEY** - Your Google Gemini API key (optional)
3. **DB_HOST** - MySQL host (if using database)
4. **DB_USER** - MySQL username
5. **DB_PASSWORD** - MySQL password
6. **DB_NAME** - Database name (default: healthcare)
7. **FLASK_DEBUG** - Set to `False` for production

## 🚀 Deploy Steps

1. **Connect Repository**
   - Go to Railway dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Set Environment Variables**
   - Go to Variables tab
   - Add all required environment variables
   - Save

3. **Deploy**
   - Railway will automatically detect Python
   - It will run: `pip install -r requirements.txt`
   - Then start with: `gunicorn --bind 0.0.0.0:$PORT web_app_gemini:app`

4. **Check Logs**
   - Go to Deployments tab
   - Click on latest deployment
   - Check logs for errors

## ✅ Verification

After deployment, check:
- ✅ App starts without errors
- ✅ Environment variables are set
- ✅ API keys are working
- ✅ Database connection (if using)

## 🐛 Troubleshooting

### Error: "No start command found"
- ✅ **Fixed**: Created `Procfile` with gunicorn command

### Error: "Module not found"
- Check `requirements.txt` has all dependencies
- Railway will install from requirements.txt

### Error: "Port already in use"
- Railway automatically sets `$PORT` environment variable
- App now reads from `os.getenv('PORT', 5000)`

### Error: "API key not found"
- Set environment variables in Railway dashboard
- Check variable names match exactly

## 📝 Notes

- **Gunicorn** is used for production (better than Flask's dev server)
- **Port** is automatically set by Railway
- **Debug mode** is disabled in production (set via FLASK_DEBUG)
- **Database** connection should work if MySQL is accessible from Railway

---

**🚂 Your app should now deploy successfully on Railway!**

