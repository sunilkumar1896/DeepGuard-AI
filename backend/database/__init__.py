"""
backend/database/__init__.py
----------------------------
SQLite database manager for DeepGuard AI Deepfake Detection Platform.
Handles persistent storage for users, authentication credentials, and analysis history.
"""
import os
import sys
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "deepguard.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with dictionary-like row access."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes database schema and default seeds."""
    conn = get_db_connection()
    try:
        with conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT,
                    role TEXT DEFAULT 'analyst',
                    created_at TEXT NOT NULL
                )
            """)

            # Analyses table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    details_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)

                    # Login / Registration events table (admin monitoring)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    username TEXT NOT NULL,
                    email TEXT NOT NULL,
                    full_name TEXT,
                    event_type TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)

            # Contact / Enterprise Inquiries table ("Get in Touch")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contact_inquiries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    organization TEXT,
                    interest TEXT,
                    message TEXT NOT NULL,
                    ip_address TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            # Password Reset Links table (email reset flow)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS password_resets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    email TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    reset_url TEXT NOT NULL,
                    is_used INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Helpful indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON login_events(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contact_timestamp ON contact_inquiries(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_resets_token ON password_resets(token)")
    finally:
        conn.close()


def log_auth_event(
    user_id: int,
    username: str,
    email: str,
    full_name: Optional[str],
    event_type: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """Logs a login or registration event for admin monitoring."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO login_events (user_id, username, email, full_name, event_type, ip_address, user_agent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, username, email, full_name or username, event_type, ip_address or "unknown", user_agent or "", timestamp),
            )
    finally:
        conn.close()


def get_all_users() -> List[Dict[str, Any]]:
    """Returns all registered users (admin only). Excludes password hashes and salts."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT id, username, email, full_name, role, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_login_events(limit: int = 200) -> List[Dict[str, Any]]:
    """Returns recent login/register events for admin dashboard."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, username, email, full_name, event_type, ip_address, user_agent, timestamp
            FROM login_events
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_contact_inquiry(
    full_name: str,
    email: str,
    organization: Optional[str],
    interest: Optional[str],
    message: str,
    ip_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Saves a 'Get in Touch' inquiry for admin monitoring."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO contact_inquiries (full_name, email, organization, interest, message, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    full_name.strip(),
                    email.strip().lower(),
                    (organization or "").strip(),
                    (interest or "").strip(),
                    message.strip(),
                    ip_address or "unknown",
                    timestamp,
                ),
            )
            return {
                "id": cursor.lastrowid,
                "full_name": full_name.strip(),
                "email": email.strip().lower(),
                "organization": organization,
                "interest": interest,
                "message": message.strip(),
                "timestamp": timestamp,
            }
    finally:
        conn.close()


def get_contact_inquiries(limit: int = 100) -> List[Dict[str, Any]]:
    """Returns all contact inquiries for the admin dashboard."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, full_name, email, organization, interest, message, ip_address, timestamp
            FROM contact_inquiries
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_password_reset(
    user_id: int,
    email: str,
    token: str,
    reset_url: str,
    expires_at: str,
) -> Dict[str, Any]:
    """Records a generated password reset token & link."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO password_resets (user_id, email, token, reset_url, is_used, created_at, expires_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (user_id, email.strip().lower(), token, reset_url, created_at, expires_at),
            )
            return {
                "id": cursor.lastrowid,
                "user_id": user_id,
                "email": email.strip().lower(),
                "token": token,
                "reset_url": reset_url,
                "created_at": created_at,
                "expires_at": expires_at,
            }
    finally:
        conn.close()


def get_password_reset_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Retrieves a password reset record by token string."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM password_resets WHERE token = ?",
            (token,),
        ).fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def mark_password_reset_used(token: str) -> bool:
    """Marks a reset token as used so it cannot be re-used."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                "UPDATE password_resets SET is_used = 1 WHERE token = ?",
                (token,),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def get_recent_password_resets(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent password reset tokens for admin monitoring."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT pr.*, u.username
            FROM password_resets pr
            LEFT JOIN users u ON pr.user_id = u.id
            ORDER BY pr.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Converts a sqlite3.Row object to a plain Python dict."""
    if row is None:
        return None
    return dict(row)


# ── User Operations ─────────────────────────────────────────────

def create_user(
    username: str,
    email: str,
    password_hash: str,
    salt: str,
    full_name: Optional[str] = None,
    role: str = "analyst",
) -> Dict[str, Any]:
    """Creates a new user record."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, full_name, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username.strip(), email.strip().lower(), password_hash, salt, full_name or username, role, created_at),
            )
            user_id = cursor.lastrowid
            return {
                "id": user_id,
                "username": username.strip(),
                "email": email.strip().lower(),
                "full_name": full_name or username,
                "role": role,
                "created_at": created_at,
            }
    finally:
        conn.close()


def update_user_password(user_id: int, password_hash: str, salt: str) -> bool:
    """Updates a user's password hash and salt."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                (password_hash, salt, user_id),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetches user by primary key ID."""
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Fetches user by case-insensitive username."""
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)).fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetches user by case-insensitive email."""
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip().lower(),)).fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def get_user_by_identifier(identifier: str) -> Optional[Dict[str, Any]]:
    """Fetches user by either username or email."""
    ident = identifier.strip()
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE",
            (ident, ident.lower()),
        ).fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


# ── Analysis Log Operations ─────────────────────────────────────

def save_analysis(
    record_id: str,
    user_id: Optional[int],
    filename: str,
    media_type: str,
    label: str,
    confidence: float,
    inference_time_ms: float,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Saves a forensic analysis scan record to the database."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details_json = json.dumps(details) if details is not None else None

    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analyses
                (id, user_id, filename, media_type, label, confidence, inference_time_ms, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    user_id,
                    filename,
                    media_type,
                    label,
                    confidence,
                    inference_time_ms,
                    details_json,
                    created_at,
                ),
            )
        return {
            "id": record_id,
            "user_id": user_id,
            "filename": filename,
            "media_type": media_type,
            "label": label,
            "confidence": confidence,
            "inference_time_ms": inference_time_ms,
            "created_at": created_at,
            "details": details,
        }
    finally:
        conn.close()


def get_history(user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves analysis records.
    If user_id is provided, returns analyses for that user.
    If user_id is None, returns recent public/guest analyses.
    """
    conn = get_db_connection()
    try:
        if user_id is not None:
            rows = conn.execute(
                """
                SELECT a.*, u.username as owner_username
                FROM analyses a
                LEFT JOIN users u ON a.user_id = u.id
                WHERE a.user_id = ?
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT a.*, u.username as owner_username
                FROM analyses a
                LEFT JOIN users u ON a.user_id = u.id
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        items = []
        for r in rows:
            item = dict(r)
            if item.get("details_json"):
                try:
                    item["details"] = json.loads(item["details_json"])
                except Exception:
                    item["details"] = None
            else:
                item["details"] = None
            item.pop("details_json", None)
            item["timestamp"] = item.get("created_at")
            items.append(item)
        return items
    finally:
        conn.close()


def get_analysis_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single analysis record by ID."""
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (record_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        if item.get("details_json"):
            try:
                item["details"] = json.loads(item["details_json"])
            except Exception:
                item["details"] = None
        else:
            item["details"] = None
        item.pop("details_json", None)
        item["timestamp"] = item.get("created_at")
        return item
    finally:
        conn.close()


def delete_analysis(record_id: str, user_id: Optional[int] = None, is_admin: bool = False) -> bool:
    """Deletes an analysis record. Ensures ownership check unless user is admin."""
    conn = get_db_connection()
    try:
        with conn:
            if is_admin or user_id is None:
                cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
            else:
                cursor = conn.execute(
                    "DELETE FROM analyses WHERE id = ? AND user_id = ?",
                    (record_id, user_id),
                )
            return cursor.rowcount > 0
    finally:
        conn.close()


def get_dashboard_stats(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Computes aggregate metrics for analyses."""
    conn = get_db_connection()
    try:
        if user_id is not None:
            query = "SELECT label, confidence FROM analyses WHERE user_id = ?"
            rows = conn.execute(query, (user_id,)).fetchall()
        else:
            query = "SELECT label, confidence FROM analyses"
            rows = conn.execute(query).fetchall()

        total = len(rows)
        real_count = sum(1 for r in rows if r["label"] == "REAL")
        fake_count = sum(1 for r in rows if r["label"] in ("DEEPFAKE", "FAKE"))
        avg_conf = (sum(r["confidence"] for r in rows) / total) if total > 0 else 0.0

        return {
            "total_analyses": total,
            "real_detected": real_count,
            "fake_detected": fake_count,
            "avg_confidence": round(avg_conf, 1),
        }
    finally:
        conn.close()
