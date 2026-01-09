"""
Database module for storing search history
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_FILE = Path(__file__).parent / 'healthcare.db'

def init_db():
    """Initialize database and create tables"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create search_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT NOT NULL,
            disease TEXT NOT NULL,
            analysis TEXT,
            predictions TEXT,
            disease_info TEXT,
            recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_created_at 
        ON search_history(created_at DESC)
    ''')
    
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized: {DB_FILE}")

def save_search_history(symptoms, disease, analysis=None, predictions=None, 
                       disease_info=None, recommendations=None):
    """Save search to history"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO search_history 
        (symptoms, disease, analysis, predictions, disease_info, recommendations)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        symptoms,
        disease,
        analysis,
        json.dumps(predictions, ensure_ascii=False) if predictions else None,
        json.dumps(disease_info, ensure_ascii=False) if disease_info else None,
        json.dumps(recommendations, ensure_ascii=False) if recommendations else None
    ))
    
    conn.commit()
    history_id = cursor.lastrowid
    conn.close()
    
    return history_id

def get_search_history(limit=50):
    """Get recent search history"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, symptoms, disease, analysis, created_at
        FROM search_history
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to list of dicts
    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'symptoms': row['symptoms'],
            'disease': row['disease'],
            'analysis': row['analysis'],
            'created_at': row['created_at']
        })
    
    return history

def get_search_detail(search_id):
    """Get full detail of a search"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM search_history WHERE id = ?
    ''', (search_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    # Parse JSON fields
    return {
        'id': row['id'],
        'symptoms': row['symptoms'],
        'disease': row['disease'],
        'analysis': row['analysis'],
        'predictions': json.loads(row['predictions']) if row['predictions'] else None,
        'disease_info': json.loads(row['disease_info']) if row['disease_info'] else None,
        'recommendations': json.loads(row['recommendations']) if row['recommendations'] else None,
        'created_at': row['created_at']
    }

def delete_search_history(search_id):
    """Delete a search from history"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM search_history WHERE id = ?', (search_id,))
    
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return deleted

def clear_all_history():
    """Clear all search history"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM search_history')
    
    count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return count

def get_statistics():
    """Get database statistics"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Total searches
    cursor.execute('SELECT COUNT(*) FROM search_history')
    total_searches = cursor.fetchone()[0]
    
    # Most common diseases
    cursor.execute('''
        SELECT disease, COUNT(*) as count
        FROM search_history
        GROUP BY disease
        ORDER BY count DESC
        LIMIT 5
    ''')
    top_diseases = [{'disease': row[0], 'count': row[1]} for row in cursor.fetchall()]
    
    # Recent activity (last 7 days)
    cursor.execute('''
        SELECT COUNT(*) FROM search_history
        WHERE created_at >= datetime('now', '-7 days')
    ''')
    recent_activity = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_searches': total_searches,
        'top_diseases': top_diseases,
        'recent_activity': recent_activity
    }

if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database ready!")

