"""
Quick script to check Gemini quota status
"""
import sys
import io
from datetime import datetime

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_quota():
    """Check Gemini API quota and suggest actions"""
    print("=" * 70)
    print("Gemini Quota Checker")
    print("=" * 70)
    
    print("\nChecking your configuration...")
    
    # Read model from config
    try:
        with open('web_app_gemini.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        model_match = re.search(r"GEMINI_MODEL = ['\"]([^'\"]+)['\"]", content)
        
        if model_match:
            model = model_match.group(1)
            print(f"Current model: {model}")
            
            # Show quota info for each model
            if model == 'gemini-1.5-flash':
                print("\n✅ GOOD CHOICE! High quota model")
                print("\nFree Tier Limits:")
                print("  📊 15 requests per minute (RPM)")
                print("  📊 1,500 requests per day (RPD)")
                print("  📊 1,000,000 tokens per minute (TPM)")
                print("\n💡 This is the BEST model for free tier")
                
            elif model == 'gemini-2.0-flash-exp':
                print("\n⚠️  WARNING: Low quota model (experimental)")
                print("\nFree Tier Limits:")
                print("  📊 2 requests per minute (RPM) - VERY LOW!")
                print("  📊 50 requests per day (RPD) - VERY LOW!")
                print("  📊 Limited tokens")
                print("\n🔧 RECOMMENDATION: Switch to gemini-1.5-flash")
                print("   Edit web_app_gemini.py:")
                print("   GEMINI_MODEL = 'gemini-1.5-flash'")
                
            else:
                print(f"\n⚠️  Unknown model: {model}")
                print("   Please use: gemini-1.5-flash")
        
    except Exception as e:
        print(f"Error reading config: {e}")
    
    # Check for quota errors in logs
    print("\n" + "-" * 70)
    print("Quota Status Check:")
    print("-" * 70)
    
    try:
        # Try to import and configure
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import google.generativeai as genai
        
        # Read API key from environment variable
        import os
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment variables")
            print("   Please set it: export GEMINI_API_KEY='your_key_here'")
            print("   Or on Windows: set GEMINI_API_KEY=your_key_here")
            return
        
        genai.configure(api_key=api_key)
        
        # Try simple request with gemini-1.5-flash (better quota)
        print("\nTesting API with gemini-1.5-flash...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            "Test",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=10,
            )
        )
        
        if response.text:
            print("✅ API is working! No quota issues detected.")
            print(f"\n💚 You're good to go!")
            print(f"   Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        
    except Exception as e:
        error_str = str(e)
        
        if '429' in error_str or 'quota' in error_str.lower():
            print("❌ QUOTA EXCEEDED")
            
            # Extract wait time if available
            import re
            wait_match = re.search(r'retry in ([\d.]+)s', error_str)
            if wait_match:
                wait_time = float(wait_match.group(1))
                print(f"\n⏰ Wait time: {wait_time:.0f} seconds (~{wait_time/60:.1f} minutes)")
            
            print("\n🔧 Actions you can take:")
            print("   1. ⏳ Wait for quota reset (shown above)")
            print("   2. 🔄 Switch to gemini-1.5-flash if not already")
            print("   3. ⚡ Use Groq mode instead (no quota issues)")
            print("   4. 💰 Upgrade to paid tier (if urgent)")
            
            print("\n📊 Check your usage at:")
            print("   https://ai.google.dev/rate-limit")
            print("   https://console.cloud.google.com/")
            
            return False
        else:
            print(f"❌ Error: {error_str[:200]}")
            return False
    
    print("\n" + "=" * 70)

def show_recommendations():
    """Show quota optimization tips"""
    print("\n" + "=" * 70)
    print("💡 Tips to Avoid Quota Issues:")
    print("=" * 70)
    
    print("""
1. ✅ Use gemini-1.5-flash (not gemini-2.0-flash-exp)
   - 15x higher quota
   - More stable
   - Production-ready

2. ⏱️  Add delays between requests
   - Wait 1-2 seconds between API calls
   - Prevents hitting rate limits

3. 🔄 Use Groq as primary
   - Set DEFAULT_AI_ENGINE = 'groq'
   - Use Gemini only when needed
   - Groq has generous free tier

4. 📊 Monitor usage
   - Check: https://ai.google.dev/rate-limit
   - Track daily usage
   - Plan accordingly

5. 💰 Consider paid tier (if needed)
   - Much higher quotas
   - Better for production
   - Still affordable
    """)

if __name__ == '__main__':
    print("\n")
    success = check_quota()
    show_recommendations()
    print("\n")
    
    if success:
        print("✅ All good! You can use Gemini now.")
    else:
        print("⚠️  Quota issues detected. Follow recommendations above.")
    
    print("\n")
