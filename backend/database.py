import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "users.db")

def connect_db():
    return sqlite3.connect(DB_PATH)

def create_users_table():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def create_analysis_table():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            ats_score INTEGER,
            date TEXT,
            pdf_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def register_user(username, email, password):
    """Register a new user with hashed password. Returns True on success."""
    conn = connect_db()
    cur = conn.cursor()
    try:
        hashed = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    """Verify user credentials. Transparently migrates plaintext passwords to hashed.
    Returns (username, email) on success, None on failure.
    """
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT username, email, password FROM users WHERE username=?",
        (username,)
    )
    user = cur.fetchone()

    if not user:
        conn.close()
        return None

    stored_password = user[2]

    # Check if password is already hashed (werkzeug hashes start with known prefixes)
    is_hashed = (
        stored_password.startswith("pbkdf2:") or
        stored_password.startswith("scrypt:") or
        stored_password.startswith("$")
    )

    if is_hashed:
        # Normal hashed password verification
        if check_password_hash(stored_password, password):
            conn.close()
            return (user[0], user[1])
        else:
            conn.close()
            return None
    else:
        # Legacy plaintext password — verify and migrate
        if stored_password == password:
            # Transparently migrate to hashed password
            new_hash = generate_password_hash(password)
            cur.execute(
                "UPDATE users SET password=? WHERE username=?",
                (new_hash, username)
            )
            conn.commit()
            conn.close()
            return (user[0], user[1])
        else:
            conn.close()
            return None
