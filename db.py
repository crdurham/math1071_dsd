import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).parent
DB_FILE = ROOT / "data" / "comments.db"

# Make folder if it doesn't exist
DB_FILE.parent.mkdir(exist_ok=True, parents=True)

def init_comments_table():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        timestamp TEXT,
        comment TEXT
    )
    """)
    conn.commit()
    conn.close()

init_comments_table()

def save_comment(name, comment):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO comments (name, timestamp, comment) 
        VALUES (?, ?, ?)
    """, (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), comment))
    conn.commit()
    conn.close()   

def load_comments():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM comments", conn)
    conn.close()
    return df

def clear_comments():
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM comments")  # delete all rows
    conn.commit()
    conn.close()