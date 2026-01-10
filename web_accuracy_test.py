from flask import Flask, render_template, jsonify
import sys
import threading
import time
import pandas as pd

# Import logic from main app
# This assumes web_app_gemini.py is in the same directory
try:
    from web_app_gemini import find_relevant_diseases, df, GROQ_MODEL, GEMINI_MODEL, DEFAULT_AI_ENGINE
except ImportError:
    # Handle case where imports might fail or require setup
    print("Warning: Could not import from web_app_gemini. using mock data for development if needed.")
    df = pd.DataFrame()
    GROQ_MODEL = "Unknown" 
    GEMINI_MODEL = "Unknown"

app = Flask(__name__)

# Global state for test progress
test_state = {
    'status': 'idle', # idle, running, completed
    'total': 200,
    'current': 0,
    'correct_top1': 0,
    'correct_top3': 0,
    'top1_acc': 0.0,
    'top3_acc': 0.0,
    'log': ''
}

def run_accuracy_test_thread():
    global test_state
    
    # Reset state
    test_state['status'] = 'running'
    test_state['current'] = 0
    test_state['correct_top1'] = 0
    test_state['correct_top3'] = 0
    test_state['log'] = 'Initialization...'
    
    sample_size = test_state.get('total', 200)
    search_method = test_state.get('method', 'hybrid')
    
    # Get random samples
    try:
        if df.empty:
            test_state['log'] = "Error: Dataset is empty"
            test_state['status'] = 'completed'
            return

        # Use random_state=None to get different samples each time
        # Handle case where sample_size > len(df)
        actual_sample_size = min(sample_size, len(df))
        test_samples = df.sample(n=actual_sample_size, random_state=None)
        
        start_time = time.time()
        
        print(f"Running test with size={actual_sample_size}, method={search_method}")
        
        for idx, row in test_samples.iterrows():
            actual_disease = row['Disease']
            symptoms = row['Question']
            
            # Predict
            try:
                # Use find_relevant_diseases from main app
                # Pass the selected search method
                # Note: 'llama' option acts like hybrid/full pipeline, 
                # but for retrieval accuracy we benchmark the underlying search
                algo_mode = 'hybrid'
                if search_method == 'tfidf': algo_mode = 'tfidf'
                elif search_method == 'rag': algo_mode = 'rag'
                elif search_method == 'bm25': algo_mode = 'bm25'
                
                _, top_diseases_list, _, _ = find_relevant_diseases(symptoms, top_k=5, search_method=algo_mode)
                
                if top_diseases_list:
                    predicted_disease = top_diseases_list[0]
                    top_3 = top_diseases_list[:3]
                    
                    if predicted_disease == actual_disease:
                        test_state['correct_top1'] += 1
                        
                    if actual_disease in top_3:
                        test_state['correct_top3'] += 1
            except Exception as e:
                print(f"Error processing sample: {e}")
            
            # Update progress
            test_state['current'] += 1
            
            # Calculate accuracy on the fly
            if test_state['current'] > 0:
                test_state['top1_acc'] = (test_state['correct_top1'] / test_state['current']) * 100
                test_state['top3_acc'] = (test_state['correct_top3'] / test_state['current']) * 100
            
            # More frequent updates at the start
            if test_state['current'] <= 10 or test_state['current'] % 10 == 0:
                msg = f"Processed {test_state['current']}/{actual_sample_size} samples..."
                test_state['log'] = msg
                print(msg) 
        
        test_state['status'] = 'completed'
        test_state['log'] = f"Test Finished in {time.time() - start_time:.2f}s"
        
    except Exception as e:
        test_state['status'] = 'error'
        test_state['log'] = f"Error: {str(e)}"

@app.route('/')
def index():
    # Determine which model is primarily active
    model_name = GROQ_MODEL if DEFAULT_AI_ENGINE == 'groq' else GEMINI_MODEL
    return render_template('accuracy.html', 
                          model_name=model_name,
                          dataset_size=len(df))

from flask import request

@app.route('/api/start_test', methods=['POST'])
def start_test():
    if test_state['status'] == 'running':
        return jsonify({'status': 'already_running'})
        
    data = request.get_json() or {}
    sample_size = int(data.get('sample_size', 200))
    search_method = data.get('search_method', 'hybrid') # hybrid, tfidf, rag, llama
    
    print(f"Received request to start test: Size={sample_size}, Method={search_method}")
    
    # Update global state configuration
    test_state['total'] = sample_size
    test_state['method'] = search_method
    
    # Start thread
    thread = threading.Thread(target=run_accuracy_test_thread)
    thread.daemon = True
    thread.start()
    
    print("Test thread started successfully.")
    
    return jsonify({'status': 'started'})

@app.route('/api/progress')
def get_progress():
    return jsonify(test_state)

if __name__ == '__main__':
    print("Starting Accuracy Test Web Server on port 5001...")
    # Run on a different port than the main app
    app.run(debug=False, port=5001)
