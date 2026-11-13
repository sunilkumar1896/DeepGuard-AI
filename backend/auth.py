"""
backend/auth.py
---------------
Authentication & Security module for DeepGuard AI Platform.
Provides PBKDF2 password hashing, JWT token generation & verification (RFC 7519),
and FastAPI user dependencies.
"""
import os
import sys
import time
import json
import base64
import hmac
import hashlib
import secrets
from typing import Optional, Dict, Any, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database import (
    get_user_by_id,
    get_user_by_identifier,
    get_user_by_username,
    get_user_by_email,
    create_user,
    init_db,
)

# Configuration
SECRET_KEY = os.getenv("DEEPGUARD_SECRET_KEY", "deepguard-ai-forensic-secret-key-2024-jwt-token")
TOKEN_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days

security_scheme = HTTPBearer(auto_error=False)


# ── Password Hashing ────────────────────────────────────────────

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns (hex_hash, salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)
    hashed_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return hashed_bytes.hex(), salt


def verify_password(plain_password: str, password_hash: str, salt: str) -> bool:
    """Verifies a plain password against the stored hash and salt."""
    expected_hash, _ = hash_password(plain_password, salt)
    return hmac.compare_digest(expected_hash, password_hash)


# ── JWT Token Implementation (RFC 7519 compliant) ───────────────

def _base64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _base64url_decode(data: str) -> bytes:
    """Base64url decode with padding restoration."""
    padding = "=" * (4 - (len(data) % 4)) if len(data) % 4 != 0 else ""
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(payload_data: Dict[str, Any], expires_in: int = TOKEN_EXPIRY_SECONDS) -> str:
    """
    Generates an RFC 7519 HMAC-SHA256 signed JSON Web Token (JWT).
    """
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = payload_data.copy()
    payload["iat"] = now
    payload["exp"] = now + expires_in

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    header_b64 = _base64url_encode(header_json)
    payload_b64 = _base64url_encode(payload_json)

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifies and decodes an HMAC-SHA256 JWT. Returns payload dict if valid, else None.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        actual_sig = _base64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload_bytes = _base64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        if payload.get("exp") and time.time() > payload["exp"]:
            return None  # Token expired

        return payload
    except Exception:
        return None


def create_password_reset_token(user_id: int, email: str, expires_in: int = 1800) -> str:
    """
    Generates a cryptographically signed password reset token valid for 30 minutes.
    """
    return create_access_token(
        {"sub": str(user_id), "email": email, "purpose": "pwd_reset"},
        expires_in=expires_in,
    )


def verify_password_reset_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifies that a token is a valid, unexpired password reset token.
    """
    payload = decode_access_token(token)
    if not payload or payload.get("purpose") != "pwd_reset":
        return None
    return payload


# ── Seed Default Accounts ───────────────────────────────────────

def seed_default_accounts():
    """Initializes tables and seeds default demo & admin accounts if not already present."""
    init_db()

    # Demo Analyst
    if not get_user_by_username("demo"):
        pwd_hash, salt = hash_password("demo12345")
        create_user(
            username="demo",
            email="demo@deepguard.ai",
            password_hash=pwd_hash,
            salt=salt,
            full_name="Demo Analyst",
            role="analyst",
        )

    # System Admin
    if not get_user_by_username("admin"):
        pwd_hash, salt = hash_password("admin12345")
        create_user(
            username="admin",
            email="admin@deepguard.ai",
            password_hash=pwd_hash,
            salt=salt,
            full_name="Security Lead",
            role="admin",
        )


# Run seed check on module load
seed_default_accounts()


# ── FastAPI Dependencies ────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Dict[str, Any]:
    """
    Dependency that enforces valid authentication.
    Raises HTTP 401 if token is missing, invalid, or expired.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in to access this resource.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return safe user dictionary without hash and salt
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[Dict[str, Any]]:
    """
    Dependency that extracts the user if a valid token is provided,
    or returns None if unauthenticated. Never raises 401.
    """
    if not credentials:
        return None

    try:
        payload = decode_access_token(credentials.credentials)
        if not payload or "sub" not in payload:
            return None

        user = get_user_by_id(int(payload["sub"]))
        if not user:
            return None

        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "created_at": user["created_at"],
        }
    except Exception:
        return None


async def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency that restricts access to admin accounts."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required.",
        )
    return current_user
