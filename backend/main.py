"""
backend/main.py
----------------
FastAPI Backend Application for Deepfake & Media Verification Platform.

Provides RESTful endpoints for:
  - User Authentication (Registration, Login, Current User, Token verification)
  - Image & Video Deepfake Detection (CNN-Transformer hybrid inference)
  - Persistent User History & Analytics
  - Model Evaluation & Dashboard Metrics
"""
import os
import sys
import time
import io
import uuid
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env if present
def _load_env_file():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key:
                        os.environ[key] = val

_load_env_file()

from inference.predictor import DeepfakePredictor
from backend.database import (
    init_db,
    save_analysis,
    get_history as db_get_history,
    delete_analysis as db_delete_analysis,
    get_dashboard_stats as db_get_dashboard_stats,
    create_user,
    get_user_by_id,
    get_user_by_username,
    get_user_by_email,
    get_user_by_identifier,
    update_user_password,
    log_auth_event,
    get_all_users,
    get_login_events,
    save_contact_inquiry,
    get_contact_inquiries,
    save_password_reset,
    get_password_reset_by_token,
    mark_password_reset_used,
    get_recent_password_resets,
)
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_password_reset_token,
    verify_password_reset_token,
    get_current_user,
    get_optional_user,
    require_admin,
    seed_default_accounts,
)

# Initialize database tables and seed demo accounts
seed_default_accounts()

# Initialize FastAPI App
app = FastAPI(
    title="AI-Powered Deepfake Detection Platform API",
    description="Forensic detection backend with JWT authentication, SQLite persistence, and Hybrid CNN-Transformer inference.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for frontend assets (CSS, JS)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global Predictor Instance
predictor = DeepfakePredictor(model_type="hybrid")

# In-memory history log storage (kept synchronized with SQLite for backward compatibility)
history_db: List[Dict[str, Any]] = db_get_history(limit=50)


# ── Pydantic Request / Response Schemas ─────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, description="Alphanumeric username")
    email: str = Field(..., min_length=5, max_length=100, description="Valid email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (at least 6 characters)")
    full_name: Optional[str] = Field(None, max_length=100, description="Display name")


class LoginRequest(BaseModel):
    identifier: str = Field(..., description="Username or Email address")
    password: str = Field(..., description="Password")


class ForgotPasswordEmailRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=100, description="Registered email address to send the reset link")


class ConfirmPasswordResetRequest(BaseModel):
    token: str = Field(..., description="Signed password reset token from email link")
    new_password: str = Field(..., min_length=6, max_length=128, description="New password (minimum 6 characters)")


class ResetPasswordRequest(BaseModel):
    identifier: str = Field(..., description="Username or Email of the account")
    new_password: str = Field(..., min_length=6, max_length=128, description="New password (minimum 6 characters)")


class AdminResetPasswordRequest(BaseModel):
    new_password: str = Field("User12345", min_length=6, max_length=128)


class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    created_at: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse


class HistoryItem(BaseModel):
    id: str
    filename: str
    media_type: str
    label: str
    confidence: float
    timestamp: str
    inference_time_ms: float
    details: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None


class ContactInquiryRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5, max_length=100)
    organization: Optional[str] = Field(None, max_length=120)
    interest: Optional[str] = Field(None, max_length=120)
    message: str = Field(..., min_length=5, max_length=2000)


# ── Root & Health Endpoints ─────────────────────────────────────

@app.get("/")
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "status": "online",
        "message": "AI-Powered Deepfake & Media Verification Platform API",
        "docs": "/docs",
        "model_architecture": "Hybrid CNN-Transformer (EfficientNet-B4 + ViT Encoder)",
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(predictor.device),
        "pytorch_version": torch.__version__,
        "model_type": predictor.model_type,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat(),
    }


# ── Authentication Endpoints ────────────────────────────────────

@app.post("/api/v1/auth/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(req: RegisterRequest, request: Request):
    """
    Registers a new user analyst account.
    Returns a JWT access token and the newly created profile.
    """
    username = req.username.strip()
    email = req.email.strip().lower()

    # Validate username characters
    if not re.match(r"^[a-zA-Z0-9_-]{3,30}$", username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be 3-30 characters long and contain only letters, numbers, underscores, or hyphens.",
        )

    # Validate email format
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a valid email address.",
        )

    # Validate password length
    if len(req.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long.",
        )

    # Check unique username
    if get_user_by_username(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This username is already taken. Please choose another one.",
        )

    # Check unique email
    if get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists.",
        )

    pwd_hash, salt = hash_password(req.password)
    user_data = create_user(
        username=username,
        email=email,
        password_hash=pwd_hash,
        salt=salt,
        full_name=req.full_name or username,
        role="analyst",
    )

    # Log registration event for admin monitoring
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    log_auth_event(
        user_id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        full_name=user_data["full_name"],
        event_type="register",
        ip_address=ip_addr,
        user_agent=user_agent,
    )

    # Generate token
    token = create_access_token({
        "sub": str(user_data["id"]),
        "username": user_data["username"],
        "email": user_data["email"],
        "role": user_data["role"],
    })

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfileResponse(**user_data),
    )


@app.post("/api/v1/auth/login", response_model=AuthTokenResponse)
def login_user(req: LoginRequest, request: Request):
    """
    Authenticates a user with username/email and password.
    Returns a JWT access token upon success.
    """
    identifier = req.identifier.strip()
    user = get_user_by_identifier(identifier)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(req.password, user["password_hash"], user["salt"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Log login event for admin monitoring
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    log_auth_event(
        user_id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user.get("full_name"),
        event_type="login",
        ip_address=ip_addr,
        user_agent=user_agent,
    )

    token = create_access_token({
        "sub": str(user["id"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
    })

    safe_user = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfileResponse(**safe_user),
    )


def send_password_reset_email(to_email: str, username: str, reset_url: str) -> Tuple[bool, Optional[str]]:
    """
    Sends an HTML password reset email via SMTP if configured.
    Falls back gracefully if SMTP is not configured.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SMTP_FROM", smtp_user or "security@deepguard.ai")

    if not smtp_host or not smtp_user or not smtp_password:
        return False, "SMTP_USER or SMTP_PASSWORD not configured in .env file"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "DeepGuard AI — Password Reset Link"
        msg["From"] = f"DeepGuard AI Security <{sender_email}>"
        msg["To"] = to_email

        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #05070b; color: #f1f5f9; padding: 24px;">
          <div style="max-width: 540px; margin: 0 auto; background: #0c1220; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 32px;">
            <div style="margin-bottom: 20px;">
              <span style="color: #06b6d4; font-weight: bold; font-size: 20px;">DeepGuard AI</span>
              <span style="color: #94a3b8; font-size: 13px; display: block;">Forensic Media Defense</span>
            </div>
            <h3 style="color: #f8fafc; margin-top: 0;">Password Reset Request</h3>
            <p style="color: #cbd5e1; font-size: 14px; line-height: 1.5;">
              Hello <strong>{username}</strong>,<br/>
              We received a request to reset your password. Click the secure link below to choose your new password. This link is valid for 30 minutes:
            </p>
            <div style="text-align: center; margin: 28px 0;">
              <a href="{reset_url}" style="background: linear-gradient(135deg, #06b6d4, #2563eb); color: #ffffff; padding: 12px 28px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 14px; display: inline-block;">
                Create New Password →
              </a>
            </div>
            <p style="font-size: 12px; color: #94a3b8; line-height: 1.5;">
              Or copy and paste this link into your browser:<br/>
              <a href="{reset_url}" style="color: #06b6d4; word-break: break-all;">{reset_url}</a>
            </p>
            <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;" />
            <p style="font-size: 11px; color: #64748b; margin: 0;">
              If you did not request this, please ignore this email. Your password will remain unchanged.
            </p>
          </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_content, "html"))

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_email, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_email, [to_email], msg.as_string())
        return True, None
    except Exception as e:
        err = str(e)
        print(f"[SMTP Error] Could not send email to {to_email}: {err}")
        return False, err


@app.post("/api/v1/auth/forgot-password")
def request_password_reset_link(req: ForgotPasswordEmailRequest, request: Request):
    """
    Step 1 of Email Reset Flow:
    Generates a secure password reset link, sends it to the user's email,
    records it in SQLite, and logs it for the admin.
    """
    email = req.email.strip().lower()
    user = get_user_by_email(email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No account found registered with email '{email}'. Please check your spelling or create an account.",
        )

    # Generate 30-minute signed token
    token = create_password_reset_token(user["id"], user["email"], expires_in=1800)

    # Base URL for reset link
    base_url = str(request.base_url).rstrip("/")
    reset_url = f"{base_url}/?reset_token={token}#reset-password"

    expires_at = (datetime.now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")

    # Persist in SQLite
    save_password_reset(
        user_id=user["id"],
        email=user["email"],
        token=token,
        reset_url=reset_url,
        expires_at=expires_at,
    )

    # Attempt SMTP email sending
    email_sent, email_error = send_password_reset_email(user["email"], user["username"], reset_url)

    # Log event for admin monitoring
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    log_auth_event(
        user_id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user.get("full_name"),
        event_type="password_reset_link_generated",
        ip_address=ip_addr,
        user_agent=user_agent,
    )

    return {
        "status": "success",
        "message": f"Password reset link generated for {user['email']}.",
        "email": user["email"],
        "username": user["username"],
        "email_sent": email_sent,
        "email_error": email_error,
        "reset_url": reset_url,
        "token": token,
    }


@app.get("/api/v1/auth/verify-reset-token")
def verify_reset_token(token: str = Query(...)):
    """
    Validates a reset token from the email link.
    """
    record = get_password_reset_by_token(token)
    if not record:
        raise HTTPException(status_code=400, detail="Invalid password reset link.")

    if record.get("is_used"):
        raise HTTPException(status_code=400, detail="This password reset link has already been used. Please request a new one.")

    payload = verify_password_reset_token(token)
    if not payload:
        raise HTTPException(status_code=400, detail="This password reset link has expired. Please request a new one.")

    user = get_user_by_id(record["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    return {
        "valid": True,
        "username": user["username"],
        "email": user["email"],
        "expires_at": record["expires_at"],
    }


@app.post("/api/v1/auth/confirm-reset-password", response_model=AuthTokenResponse)
def confirm_password_reset_link(req: ConfirmPasswordResetRequest, request: Request):
    """
    Step 2 of Email Reset Flow:
    Validates the token from the email link, updates the password in SQLite,
    marks the token as used, and auto-logs the user in.
    """
    token = req.token.strip()
    record = get_password_reset_by_token(token)

    if not record:
        raise HTTPException(status_code=400, detail="Invalid password reset link.")

    if record.get("is_used"):
        raise HTTPException(status_code=400, detail="This password reset link has already been used. Please request a new one.")

    payload = verify_password_reset_token(token)
    if not payload:
        raise HTTPException(status_code=400, detail="This password reset link has expired. Please request a new one.")

    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters long.")

    user = get_user_by_id(record["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    # Update password
    pwd_hash, salt = hash_password(req.new_password)
    update_user_password(user["id"], pwd_hash, salt)

    # Invalidate token
    mark_password_reset_used(token)

    # Log completion for admin
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    log_auth_event(
        user_id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user.get("full_name"),
        event_type="password_reset_completed",
        ip_address=ip_addr,
        user_agent=user_agent,
    )

    # Auto-login token
    auth_token = create_access_token({
        "sub": str(user["id"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
    })

    safe_user = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }

    return AuthTokenResponse(
        access_token=auth_token,
        token_type="bearer",
        user=UserProfileResponse(**safe_user),
    )


@app.post("/api/v1/auth/reset-password", response_model=AuthTokenResponse)
def reset_user_password(req: ResetPasswordRequest, request: Request):
    """
    Self-service password reset: Allows a user to reset their password
    using their registered username or email. Automatically logs them in.
    """
    identifier = req.identifier.strip()
    user = get_user_by_identifier(identifier)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found matching this username or email address.",
        )

    if len(req.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters long.",
        )

    # Hash new password
    pwd_hash, salt = hash_password(req.new_password)
    update_user_password(user["id"], pwd_hash, salt)

    # Log password reset event for admin monitoring
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    log_auth_event(
        user_id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user.get("full_name"),
        event_type="password_reset",
        ip_address=ip_addr,
        user_agent=user_agent,
    )

    # Generate token for immediate login
    token = create_access_token({
        "sub": str(user["id"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
    })

    safe_user = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfileResponse(**safe_user),
    )


@app.get("/api/v1/auth/me", response_model=UserProfileResponse)
def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Returns the profile of the currently authenticated user.
    """
    return UserProfileResponse(**current_user)


@app.post("/api/v1/auth/logout")
def logout_user():
    """
    Signals logout completion. (Tokens are stateless JWTs, discarded on client side).
    """
    return {"status": "success", "message": "Successfully signed out."}


@app.get("/api/v1/auth/demo-credentials")
def get_demo_credentials():
    """
    Provides demo credentials for evaluator testing and quick logins.
    """
    return {
        "accounts": [
            {
                "label": "Demo Analyst",
                "identifier": "demo",
                "password": "demo12345",
                "role": "analyst",
                "description": "Pre-configured analyst account for scanning and reviewing media.",
            },
            {
                "label": "Security Admin",
                "identifier": "admin",
                "password": "admin12345",
                "role": "admin",
                "description": "Administrator account with full forensic management access.",
            },
        ]
    }


# ── Media Analysis Endpoints ────────────────────────────────────

@app.post("/api/v1/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    Analyze an uploaded image or auto-route video file.
    If the client is authenticated, ties the analysis to the user account.
    """
    filename_lower = file.filename.lower()
    if file.content_type.startswith("video/") or filename_lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return await analyze_video(file=file, max_frames=16, current_user=current_user)

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Perform Prediction
    result = predictor.predict_image(image)

    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = current_user["id"] if current_user else None

    response_data = {
        "id": record_id,
        "filename": file.filename,
        "media_type": "image",
        "timestamp": timestamp,
        "user_id": user_id,
        "owner_username": current_user["username"] if current_user else None,
        **result,
    }

    # Persist in SQLite
    save_analysis(
        record_id=record_id,
        user_id=user_id,
        filename=file.filename,
        media_type="image",
        label=result.get("label", "UNKNOWN"),
        confidence=float(result.get("confidence", 0.0)),
        inference_time_ms=float(result.get("inference_time_ms", 0.0)),
        details=result,
    )

    # Save to in-memory history cache
    history_db.insert(0, response_data)

    return response_data


@app.post("/api/v1/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: int = Query(16, ge=4, le=64, description="Number of frames to extract and analyze"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    Analyze an uploaded video file (MP4, AVI, MOV) by extracting face sequence frames.
    Ties the result to the authenticated user if logged in.
    """
    if not file.content_type.startswith("video/") and not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="File uploaded must be a video.")

    contents = await file.read()
    temp_video_path = os.path.join(PROJECT_ROOT, f"temp_{uuid.uuid4().hex[:6]}.mp4")

    try:
        with open(temp_video_path, "wb") as f:
            f.write(contents)

        # Extract frames using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise HTTPException(status_code=400, detail="Could not read frames from video file.")

        step = max(1, total_frames // max_frames)
        extracted_frames = []

        frame_idx = 0
        while cap.isOpened() and len(extracted_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                extracted_frames.append(pil_img)

            frame_idx += 1

        cap.release()

        if not extracted_frames:
            raise HTTPException(status_code=400, detail="No valid frames extracted from video.")

        # Run Video Prediction
        result = predictor.predict_video_frames(extracted_frames)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = current_user["id"] if current_user else None

    response_data = {
        "id": record_id,
        "filename": file.filename,
        "media_type": "video",
        "timestamp": timestamp,
        "user_id": user_id,
        "owner_username": current_user["username"] if current_user else None,
        **result,
    }

    # Persist in SQLite
    save_analysis(
        record_id=record_id,
        user_id=user_id,
        filename=file.filename,
        media_type="video",
        label=result.get("label", "UNKNOWN"),
        confidence=float(result.get("confidence", 0.0)),
        inference_time_ms=float(result.get("inference_time_ms", 0.0)),
        details=result,
    )

    # Save to in-memory history cache
    history_db.insert(0, response_data)

    return response_data


# ── History & Dashboard Endpoints ───────────────────────────────

@app.get("/api/v1/history")
def get_history(
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    Get analysis logs history.
    If authenticated, returns the user's personal analysis logs.
    If guest, returns the recent platform analyses.
    """
    user_id = current_user["id"] if current_user else None
    records = db_get_history(user_id=user_id, limit=limit)

    return {
        "total_records": len(records),
        "history": records,
        "authenticated": current_user is not None,
        "username": current_user["username"] if current_user else None,
    }


@app.delete("/api/v1/history/{item_id}")
def delete_history_item(
    item_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    Delete a specific prediction record from history log.
    Ensures ownership verification or admin privileges.
    """
    global history_db
    user_id = current_user["id"] if current_user else None
    is_admin = (current_user.get("role") == "admin") if current_user else False

    success = db_delete_analysis(item_id, user_id=user_id, is_admin=is_admin)

    if not success:
        # Check if item exists at all to return proper 404 vs 403
        raise HTTPException(
            status_code=404,
            detail="Item not found or you do not have permission to delete it.",
        )

    # Update in-memory cache
    history_db = [item for item in history_db if item.get("id") != item_id]

    return {"status": "success", "deleted_id": item_id}


@app.get("/api/v1/stats")
def get_stats(current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)):
    """
    Returns platform and user-specific detection statistics.
    """
    user_id = current_user["id"] if current_user else None
    stats = db_get_dashboard_stats(user_id=user_id)
    return stats


@app.get("/api/v1/models")
def list_models():
    """
    List model architectures evaluated in this platform.
    """
    return {
        "active_model": predictor.model_type,
        "models": [
            {
                "id": "hybrid",
                "name": "Hybrid CNN-Transformer",
                "status": "active",
                "description": "Combines EfficientNet-B4 local feature extraction with Multi-head Attention transformer encoder for spatial & temporal modeling.",
            },
            {
                "id": "cnn",
                "name": "CNN Baseline (EfficientNet-B4)",
                "status": "ready",
                "description": "Standard convolutional neural network baseline fine-tuned for facial artifact detection.",
            },
            {
                "id": "transformer",
                "name": "Vision Transformer (ViT)",
                "status": "ready",
                "description": "Pure transformer baseline splitting input frames into 16x16 visual patches.",
            },
        ],
    }




# ── Contact & Enterprise Inquiries ("Get in Touch") ─────────────

@app.post("/api/v1/contact", status_code=status.HTTP_201_CREATED)
def submit_contact_inquiry(req: ContactInquiryRequest, request: Request):
    """
    Saves an enterprise inquiry from the 'Get in Touch' form and logs it for admin review.
    """
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", req.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a valid business or institutional email address.",
        )

    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    inquiry = save_contact_inquiry(
        full_name=req.full_name,
        email=req.email,
        organization=req.organization,
        interest=req.interest,
        message=req.message,
        ip_address=ip_addr,
    )

    return {
        "status": "success",
        "message": "Thank you! Your forensic inquiry has been received. Our team will contact you shortly.",
        "inquiry_id": inquiry["id"],
    }


# ── Admin Monitoring Endpoints ───────────────────────────────────

@app.get("/api/v1/admin/users")
def admin_get_all_users(current_user: Dict[str, Any] = Depends(require_admin)):
    """
    Admin-only: Returns all registered users with profile details.
    Excludes sensitive fields (password hash, salt).
    """
    users = get_all_users()
    return {
        "total_users": len(users),
        "users": users,
        "requested_by": current_user["username"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/admin/events")
def admin_get_login_events(
    limit: int = Query(200, ge=1, le=1000),
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """
    Admin-only: Returns the most recent login and registration events.
    Useful for monitoring platform access and new sign-ups.
    """
    events = get_login_events(limit=limit)
    return {
        "total_events": len(events),
        "events": events,
        "requested_by": current_user["username"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/admin/inquiries")
def admin_get_inquiries(
    limit: int = Query(100, ge=1, le=500),
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """
    Admin-only: Returns all submitted 'Get in Touch' inquiries.
    """
    inquiries = get_contact_inquiries(limit=limit)
    return {
        "total_inquiries": len(inquiries),
        "inquiries": inquiries,
        "requested_by": current_user["username"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/admin/summary")
def admin_get_summary(current_user: Dict[str, Any] = Depends(require_admin)):
    """
    Admin-only: Returns a quick summary dashboard — total users, total analyses,
    recent login/register events, and inquiries.
    """
    all_users = get_all_users()
    recent_events = get_login_events(limit=10)
    inquiries = get_contact_inquiries(limit=10)
    global_stats = db_get_dashboard_stats(user_id=None)

    return {
        "total_registered_users": len(all_users),
        "total_inquiries": len(inquiries),
        "platform_stats": global_stats,
        "recent_activity": recent_events[:10],
        "recent_inquiries": inquiries[:5],
        "requested_by": current_user["username"],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v1/admin/users/{user_id}/reset-password")
def admin_reset_user_password(
    user_id: int,
    req: AdminResetPasswordRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """
    Admin-only: Directly resets any user's password.
    """
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    pwd_hash, salt = hash_password(req.new_password)
    update_user_password(user_id, pwd_hash, salt)

    log_auth_event(
        user_id=user["id"],
        username=user["username"],
        email=user["email"],
        full_name=user.get("full_name"),
        event_type="admin_password_reset",
        ip_address=f"admin:{current_user['username']}",
        user_agent="Admin Control Panel",
    )

    return {
        "status": "success",
        "message": f"Password for '{user['username']}' successfully reset to: {req.new_password}",
        "user_id": user_id,
        "new_password": req.new_password,
    }


@app.get("/api/v1/admin/resets")
def admin_get_password_resets(
    limit: int = Query(50, ge=1, le=200),
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """
    Admin-only: Returns all generated email password reset links with status.
    """
    resets = get_recent_password_resets(limit=limit)
    return {
        "total_resets": len(resets),
        "resets": resets,
        "requested_by": current_user["username"],
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
