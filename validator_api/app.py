from fastapi import FastAPI, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pydantic import BaseModel
import uuid
import sys
import os
import random
import time
import hashlib
import ipaddress
import requests
import json
from collections import defaultdict

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env from project root (parent of validator_api)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    env_path = os.path.join(parent_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
except ImportError:
    # python-dotenv not installed, skip (env vars can still be set manually)
    pass

# Add parent directory to path to import quasar
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from . import models
from . import auth
from .database import engine, get_db

from fastapi.middleware.cors import CORSMiddleware

# Create database tables
models.Base.metadata.create_all(bind=engine)

# IP Banning Configuration
MAX_FAILURES_BEFORE_BAN = 5
BAN_DURATION_HOURS = 24

# TPS Sanity Bounds
MIN_PLAUSIBLE_TPS = float(os.environ.get("MIN_PLAUSIBLE_TPS", "10.0"))
MAX_PLAUSIBLE_TPS = float(os.environ.get("MAX_PLAUSIBLE_TPS", "50000000.0"))


def get_client_ip(request: Request) -> Optional[str]:
    """
    Extract the real client IP from a request
    """
    forwarded = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded:
        ip = forwarded.split(",")[0].strip()
        if ip:
            return ip

    real_ip = request.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip

    if hasattr(request, "client") and request.client:
        return request.client.host

    return None


def is_private_ip(ip: Optional[str]) -> bool:
    """Return True if the IP is a private/internal address (RFC 1918, loopback, etc.)."""
    if not ip:
        return True
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def check_ip_ban(ip_address: str, db: Session) -> tuple[bool, Optional[str]]:
    """
    Check if IP is banned.
    Returns: (is_banned, reason)
    """
    if not ip_address:
        return False, None
    
    ip_ban = db.query(models.IPBan).filter(
        models.IPBan.ip_address == ip_address
    ).first()
    
    if not ip_ban:
        return False, None
    
    # Check if ban has expired
    if ip_ban.is_banned and ip_ban.banned_until:
        if datetime.utcnow() < ip_ban.banned_until:
            remaining = (ip_ban.banned_until - datetime.utcnow()).total_seconds() / 3600
            return True, f"IP banned for {remaining:.1f} more hours"
        else:
            # Ban expired, reset
            ip_ban.is_banned = False
            ip_ban.banned_until = None
            ip_ban.failure_count = 0
            db.commit()
            return False, None
    
    return False, None

def record_failure(ip_address: str, db: Session):
    """Record a failed submission for IP tracking."""
    if not ip_address:
        return
    
    ip_ban = db.query(models.IPBan).filter(
        models.IPBan.ip_address == ip_address
    ).first()
    
    if not ip_ban:
        ip_ban = models.IPBan(
            ip_address=ip_address,
            failure_count=1,
            last_failure_time=datetime.utcnow()
        )
        db.add(ip_ban)
    else:
        ip_ban.failure_count += 1
        ip_ban.last_failure_time = datetime.utcnow()
        
        # Ban if exceeds threshold
        if ip_ban.failure_count >= MAX_FAILURES_BEFORE_BAN:
            from datetime import timedelta
            ip_ban.is_banned = True
            ip_ban.banned_until = datetime.utcnow() + timedelta(hours=BAN_DURATION_HOURS)
            print(f"🚫 [IP_BAN] Banned IP {ip_address} for {BAN_DURATION_HOURS} hours "
                  f"(failures: {ip_ban.failure_count})")
    
    db.commit()

def record_success(ip_address: str, db: Session):
    """Reset failure count on successful submission."""
    if not ip_address:
        return
    
    ip_ban = db.query(models.IPBan).filter(
        models.IPBan.ip_address == ip_address
    ).first()
    
    if ip_ban and ip_ban.failure_count > 0:
        ip_ban.failure_count = 0
        db.commit()


def normalize_network(network: Optional[str]) -> str:
    """Return 'finney' or 'test'. Defaults to 'finney'."""
    if not network or not str(network).strip():
        return "finney"
    n = str(network).strip().lower()
    return n if n in ("finney", "test") else "finney"


# Helper function for solution hash calculation
def calculate_solution_hash(tokens_per_sec: float, target_sequence_length: int, 
                          benchmarks: Optional[Dict] = None) -> str:
    """
    Calculate hash of solution to detect identical results.
    Used for first-submission-wins logic.
    """
    # Normalize to 2 decimal places to account for minor variations
    normalized_tps = round(tokens_per_sec, 2)
    
    # Create hashable representation
    solution_data = {
        "tokens_per_sec": normalized_tps,
        "target_sequence_length": target_sequence_length,
        "benchmarks": benchmarks or {}
    }
    
    # Sort benchmarks for consistent hashing
    if benchmarks:
        solution_data["benchmarks"] = dict(sorted(benchmarks.items()))
    
    # Create hash
    solution_str = json.dumps(solution_data, sort_keys=True)
    return hashlib.sha256(solution_str.encode()).hexdigest()[:16]


def _github_username_from_fork_url(fork_url: Optional[str]) -> Optional[str]:
    """Extract GitHub username from fork URL, e.g. https://github.com/username/repo -> username."""
    if not fork_url:
        return None
    import re
    m = re.match(r"https?://(?:www\.)?github\.com/([^/]+)", (fork_url or "").strip())
    return m.group(1) if m else None


# REWARD_DISTRIBUTION

# Add new columns if they don't exist (migration)
from sqlalchemy import text
try:
    # Check database type
    is_postgresql = "postgresql" in str(engine.url)
    
    with engine.connect() as conn:
        if is_postgresql:
            # PostgreSQL: use information_schema
            # Check speed_submissions columns
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'speed_submissions'
            """))
            columns = [row[0] for row in result]
            
            # Add missing columns
            if "vram_mb" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN vram_mb REAL"))
                conn.commit()
            if "benchmarks" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN benchmarks TEXT"))
                conn.commit()
            if "validated" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN validated BOOLEAN DEFAULT FALSE"))
                conn.commit()
            if "round_id" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN round_id INTEGER"))
                conn.commit()
            if "ip_address" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN ip_address VARCHAR"))
                conn.commit()
            if "is_baseline" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN is_baseline BOOLEAN DEFAULT FALSE"))
                conn.commit()
            if "solution_hash" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN solution_hash VARCHAR"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # CONTEXT BUILDER COLUMN (Phase 5: repo_hash for consistency tracking)
            # ═══════════════════════════════════════════════════════════════════════════
            if "repo_hash" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN repo_hash VARCHAR"))
                conn.commit()
                # Create index for repo_hash
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_speed_submissions_repo_hash ON speed_submissions(repo_hash)"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # COMMIT-REVEAL COLUMNS (from const's qllm architecture)
            # ═══════════════════════════════════════════════════════════════════════════
            if "commitment_hash" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN commitment_hash VARCHAR"))
                conn.commit()
            if "commitment_salt" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN commitment_salt VARCHAR"))
                conn.commit()
            if "reveal_block" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN reveal_block INTEGER"))
                conn.commit()
            if "is_revealed" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN is_revealed BOOLEAN DEFAULT TRUE"))
                conn.commit()
            if "docker_image" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN docker_image VARCHAR"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # LOGIT VERIFICATION COLUMNS (from const's qllm architecture)
            # ═══════════════════════════════════════════════════════════════════════════
            if "logit_verification_passed" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN logit_verification_passed BOOLEAN"))
                conn.commit()
            if "cosine_similarity" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN cosine_similarity REAL"))
                conn.commit()
            if "max_abs_diff" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN max_abs_diff REAL"))
                conn.commit()
            if "verification_reason" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN verification_reason VARCHAR"))
                conn.commit()
            if "throughput_verified" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN throughput_verified REAL"))
                conn.commit()
            if "validated_tokens_per_sec" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN validated_tokens_per_sec REAL"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # SCORE COLUMN (for storing validation scores)
            # ═══════════════════════════════════════════════════════════════════════════
            if "score" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN score REAL"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # NETWORK COLUMN (testnet vs mainnet separation) - PostgreSQL
            # ═══════════════════════════════════════════════════════════════════════════
            if "network" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN network VARCHAR(32) DEFAULT 'finney' NOT NULL"))
                conn.commit()
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_speed_submissions_network ON speed_submissions(network)"))
                conn.commit()
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'competition_rounds'
            """))
            cr_cols = [row[0] for row in result] if result else []
            if cr_cols and "network" not in cr_cols:
                conn.execute(text("ALTER TABLE competition_rounds ADD COLUMN network VARCHAR(32) DEFAULT 'finney' NOT NULL"))
                conn.commit()
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'miner_scores'
            """))
            ms_cols = [row[0] for row in result] if result else []
            if ms_cols and "network" not in ms_cols:
                conn.execute(text("ALTER TABLE miner_scores ADD COLUMN network VARCHAR(32) DEFAULT 'finney' NOT NULL"))
                conn.commit()
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'miner_registrations'
            """))
            mr_cols = [row[0] for row in result] if result else []
            if mr_cols and "network" not in mr_cols:
                conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN network VARCHAR(32) DEFAULT 'finney' NOT NULL"))
                conn.commit()
            if mr_cols and "coldkey" not in mr_cols:
                conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN coldkey VARCHAR"))
                conn.commit()
            if mr_cols and "is_flagged" not in mr_cols:
                conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN is_flagged BOOLEAN DEFAULT FALSE"))
                conn.commit()
            if mr_cols and "flag_reason" not in mr_cols:
                conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN flag_reason VARCHAR"))
                conn.commit()
            if mr_cols and "github_username" not in mr_cols:
                conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN github_username VARCHAR"))
                conn.commit()
            if mr_cols and "registration_ip" not in mr_cols:
                conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN registration_ip VARCHAR"))
                conn.commit()
            
            # Check if competition_rounds table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'competition_rounds'
                )
            """))
            if not result.scalar():
                # Table doesn't exist, will be created by create_all
                pass
            
            # Check if ip_bans table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'ip_bans'
                )
            """))
            if not result.scalar():
                # Table doesn't exist, will be created by create_all
                pass
        else:
            # SQLite: use PRAGMA
            result = conn.execute(text("PRAGMA table_info(speed_submissions)"))
            columns = [row[1] for row in result]
            
            # Add missing columns
            if "vram_mb" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN vram_mb REAL"))
                conn.commit()
            if "benchmarks" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN benchmarks TEXT"))
                conn.commit()
            if "validated" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN validated BOOLEAN DEFAULT 0"))
                conn.commit()
            if "round_id" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN round_id INTEGER"))
                conn.commit()
            if "ip_address" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN ip_address TEXT"))
                conn.commit()
            if "is_baseline" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN is_baseline BOOLEAN DEFAULT 0"))
                conn.commit()
            if "solution_hash" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN solution_hash TEXT"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # CONTEXT BUILDER COLUMN (Phase 5: repo_hash for consistency tracking)
            # ═══════════════════════════════════════════════════════════════════════════
            if "repo_hash" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN repo_hash TEXT"))
                conn.commit()
                # Create index for repo_hash
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_speed_submissions_repo_hash ON speed_submissions(repo_hash)"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # COMMIT-REVEAL COLUMNS (from const's qllm architecture)
            # ═══════════════════════════════════════════════════════════════════════════
            if "commitment_hash" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN commitment_hash TEXT"))
                conn.commit()
            if "commitment_salt" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN commitment_salt TEXT"))
                conn.commit()
            if "reveal_block" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN reveal_block INTEGER"))
                conn.commit()
            if "is_revealed" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN is_revealed BOOLEAN DEFAULT 1"))
                conn.commit()
            if "docker_image" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN docker_image TEXT"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # LOGIT VERIFICATION COLUMNS (from const's qllm architecture)
            # ═══════════════════════════════════════════════════════════════════════════
            if "logit_verification_passed" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN logit_verification_passed BOOLEAN"))
                conn.commit()
            if "cosine_similarity" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN cosine_similarity REAL"))
                conn.commit()
            if "max_abs_diff" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN max_abs_diff REAL"))
                conn.commit()
            if "verification_reason" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN verification_reason TEXT"))
                conn.commit()
            if "throughput_verified" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN throughput_verified REAL"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # SCORE COLUMN (for storing validation scores)
            # ═══════════════════════════════════════════════════════════════════════════
            if "score" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN score REAL"))
                conn.commit()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # NETWORK COLUMN (testnet vs mainnet separation) - SQLite
            # ═══════════════════════════════════════════════════════════════════════════
            if "network" not in columns:
                conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN network TEXT DEFAULT 'finney'"))
                conn.commit()
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_speed_submissions_network ON speed_submissions(network)"))
                conn.commit()
            try:
                result = conn.execute(text("PRAGMA table_info(competition_rounds)"))
                cr_cols = [row[1] for row in result]
                if "network" not in cr_cols:
                    conn.execute(text("ALTER TABLE competition_rounds ADD COLUMN network TEXT DEFAULT 'finney'"))
                    conn.commit()
            except Exception:
                pass
            try:
                result = conn.execute(text("PRAGMA table_info(miner_scores)"))
                ms_cols = [row[1] for row in result]
                if "network" not in ms_cols:
                    conn.execute(text("ALTER TABLE miner_scores ADD COLUMN network TEXT DEFAULT 'finney'"))
                    conn.commit()
            except Exception:
                pass
            try:
                result = conn.execute(text("PRAGMA table_info(miner_registrations)"))
                mr_cols = [row[1] for row in result]
                if "network" not in mr_cols:
                    conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN network TEXT DEFAULT 'finney'"))
                    conn.commit()
                if "coldkey" not in mr_cols:
                    conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN coldkey TEXT"))
                    conn.commit()
                if "is_flagged" not in mr_cols:
                    conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN is_flagged INTEGER DEFAULT 0"))
                    conn.commit()
                if "flag_reason" not in mr_cols:
                    conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN flag_reason TEXT"))
                    conn.commit()
                if "github_username" not in mr_cols:
                    conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN github_username TEXT"))
                    conn.commit()
                if "registration_ip" not in mr_cols:
                    conn.execute(text("ALTER TABLE miner_registrations ADD COLUMN registration_ip TEXT"))
                    conn.commit()
            except Exception:
                pass
            
            # Commit any pending changes
            conn.commit()
except Exception as e:
    print(f"⚠️  Database migration warning: {e}")
    print("   This is normal if the database is already up-to-date.")
    # Continue anyway - the app will work if columns already exist

app = FastAPI(title="Quasar Validator API")

_cors_origins_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()] if _cors_origins_raw else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Hotkey", "Signature", "Timestamp", "Authorization", "X-API-Key"],
    max_age=600,
)

# Rate limiting for DDOS protection
# Simple in-memory rate limiter: {hotkey: [timestamp1, timestamp2, ...]}
rate_limit_store = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 10  # max requests per window

def check_rate_limit(hotkey: str):
    """Check if hotkey has exceeded rate limit."""
    now = time.time()
    # Remove old timestamps outside the window
    rate_limit_store[hotkey] = [t for t in rate_limit_store[hotkey] if now - t < RATE_LIMIT_WINDOW]

    if len(rate_limit_store[hotkey]) >= RATE_LIMIT_MAX_REQUESTS:
        print(f"⚠️ [RATE_LIMIT] Hotkey {hotkey[:12]}... exceeded rate limit")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )

    # Add current timestamp
    rate_limit_store[hotkey].append(now)

# League configuration
LEAGUES = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "1M"]
LEAGUE_MULTIPLIERS = {
    "100k": 0.5,
    "200k": 0.75,
    "300k": 1.0,
    "400k": 1.25,
    "500k": 1.5,
    "600k": 1.75,
    "700k": 2.0,
    "800k": 2.25,
    "900k": 2.5,
    "1M": 3.0
}

def get_league(context_length: int) -> str:
    """Determine league based on context length."""
    for i, league in enumerate(LEAGUES):
        max_tokens = (i + 1) * 100_000
        if context_length <= max_tokens:
            return league
    return "1M"  # Fallback to highest league

def get_league_for_seq_len(seq_len: int) -> str:
    """Get league based on sequence length."""
    if seq_len >= 1_000_000:
        return "1M"
    elif seq_len >= 900_000:
        return "900k"
    elif seq_len >= 800_000:
        return "800k"
    elif seq_len >= 700_000:
        return "700k"
    elif seq_len >= 600_000:
        return "600k"
    elif seq_len >= 500_000:
        return "500k"
    elif seq_len >= 400_000:
        return "400k"
    elif seq_len >= 300_000:
        return "300k"
    elif seq_len >= 200_000:
        return "200k"
    else:
        return "100k"

class WeightEntry(BaseModel):
    uid: int
    hotkey: str
    weight: float
    tokens_per_sec: Optional[float] = None
    github_username: Optional[str] = None

class GetWeightsResponse(BaseModel):
    epoch: int
    weights: List[WeightEntry]
    round_id: Optional[int] = None
    round_number: Optional[int] = None
    round_status: Optional[str] = None
    winner_hotkey: Optional[str] = None

@app.post("/submit_kernel")
def submit_kernel(
    req: models.SpeedSubmissionRequest,
    request: Request,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Submit kernel optimization results from miners.
    Stores fork URL, commit hash, performance metrics, and signature.
    """
    import traceback
    try:
        print(f"📥 [SUBMIT_KERNEL] Miner: {req.miner_hotkey[:8]} | Fork: {req.fork_url}")
        print(f"📥 [SUBMIT_KERNEL] Commit: {req.commit_hash[:12]}... | Performance: {req.tokens_per_sec:.2f} tokens/sec")
        if req.repo_hash:
            print(f"📥 [SUBMIT_KERNEL] Repo Hash: {req.repo_hash} (context consistency)")
        if req.vram_mb is not None:
            print(f"📥 [SUBMIT_KERNEL] VRAM_MB: {req.vram_mb:.2f}")
        if req.benchmarks is not None:
            try:
                print(f"📥 [SUBMIT_KERNEL] Benchmarks: {len(req.benchmarks)} seq lengths")
            except Exception:
                print(f"📥 [SUBMIT_KERNEL] Benchmarks: (unprintable)")

        # Verify the hotkey matches the authenticated miner
        if req.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="Hotkey mismatch")

        # Logit verification required: only verified submissions can rank
        
        if req.tokens_per_sec <= 0:
            raise HTTPException(
                status_code=400,
                detail="tokens_per_sec must be a positive number."
            )
        if req.tokens_per_sec < MIN_PLAUSIBLE_TPS or req.tokens_per_sec > MAX_PLAUSIBLE_TPS:
            raise HTTPException(
                status_code=400,
                detail=f"tokens_per_sec={req.tokens_per_sec:.2f} is outside plausible range "
                       f"Rejected as spam."
            )

        network = normalize_network(getattr(req, "network", None))

        # Check if miner is registered for this network, auto-register if not
        miner_reg = db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey == hotkey,
            models.MinerRegistration.network == network
        ).first()

        if not miner_reg:
            miner_reg = models.MinerRegistration(
                hotkey=hotkey,
                network=network,
                uid=0
            )
            db.add(miner_reg)
            db.commit()
            print(f"✅ [SUBMIT_KERNEL] Auto-registered miner {hotkey[:8]}... on {network} (UID will be synced from metagraph)")

        client_ip = get_client_ip(request)
        
        # Check IP ban BEFORE processing submission
        is_banned, ban_reason = check_ip_ban(client_ip, db)
        if is_banned:
            raise HTTPException(
                status_code=403,
                detail=f"IP address banned: {ban_reason}"
            )

        # ── Anti-spam: flagged miner check ──
        if miner_reg and miner_reg.is_flagged:
            print(f"🚫 Flagged miner {hotkey[:12]}... reason: {miner_reg.flag_reason}")
            raise HTTPException(status_code=403, detail=f"Miner flagged: {miner_reg.flag_reason or 'anti-spam'}")

        # ── Anti-spam: GitHub username dedup ──
        gh_user = _github_username_from_fork_url(req.fork_url)
        if gh_user:
            existing_owner = (
                db.query(models.MinerRegistration)
                .filter(
                    models.MinerRegistration.github_username == gh_user,
                    models.MinerRegistration.network == network,
                    models.MinerRegistration.hotkey != hotkey,
                )
                .first()
            )
            if existing_owner:
                print(f"🚫 GitHub user '{gh_user}' already claimed by hotkey {existing_owner.hotkey[:12]}... — rejecting {hotkey[:12]}...")
                if miner_reg and not miner_reg.is_flagged:
                    miner_reg.is_flagged = True
                    miner_reg.flag_reason = f"Duplicate GitHub username '{gh_user}' (original: {existing_owner.hotkey[:12]}...)"
                    db.commit()
                raise HTTPException(
                    status_code=403,
                    detail=f"GitHub username '{gh_user}' is already registered to a different hotkey. One hotkey per GitHub account."
                )
            if miner_reg and not miner_reg.github_username:
                miner_reg.github_username = gh_user
                db.commit()

        # IP registration dedup (skip for private/proxy IPs to avoid false positives)
        if client_ip and miner_reg and not is_private_ip(client_ip):
            if not miner_reg.registration_ip:
                miner_reg.registration_ip = client_ip
                db.commit()
            other_ip_reg = (
                db.query(models.MinerRegistration)
                .filter(
                    models.MinerRegistration.registration_ip == client_ip,
                    models.MinerRegistration.network == network,
                    models.MinerRegistration.hotkey != hotkey,
                )
                .first()
            )
            if other_ip_reg:
                print(f"🚫 IP {client_ip} already used by hotkey {other_ip_reg.hotkey[:12]}... — rejecting {hotkey[:12]}...")
                if miner_reg and not miner_reg.is_flagged:
                    miner_reg.is_flagged = True
                    miner_reg.flag_reason = f"Duplicate IP {client_ip} (original: {other_ip_reg.hotkey[:12]}...)"
                    db.commit()
                raise HTTPException(
                    status_code=403,
                    detail="This IP address is already associated with a different hotkey. One hotkey per IP."
                )

        # ── Anti-spam: coldkey dedup (one hotkey per coldkey) ──
        if miner_reg and miner_reg.coldkey:
            other_coldkey_reg = (
                db.query(models.MinerRegistration)
                .filter(
                    models.MinerRegistration.coldkey == miner_reg.coldkey,
                    models.MinerRegistration.network == network,
                    models.MinerRegistration.hotkey != hotkey,
                )
                .first()
            )
            if other_coldkey_reg:
                print(f"🚫 Coldkey {miner_reg.coldkey[:12]}... already used by hotkey {other_coldkey_reg.hotkey[:12]}... — rejecting {hotkey[:12]}...")
                if not miner_reg.is_flagged:
                    miner_reg.is_flagged = True
                    miner_reg.flag_reason = f"Duplicate coldkey (original: {other_coldkey_reg.hotkey[:12]}...)"
                    db.commit()
                raise HTTPException(
                    status_code=403,
                    detail="This coldkey already has a registered hotkey. One hotkey per coldkey."
                )

        # Rate-limiting
        # Prevent spam submissions from the same miner (max 1 submission per 5 minutes)
        SUBMISSION_COOLDOWN_SECONDS = int(os.environ.get("SUBMISSION_COOLDOWN_SECONDS", "300"))  # 5 minutes
        recent_submission = (
            db.query(models.SpeedSubmission)
            .filter(
                models.SpeedSubmission.miner_hotkey == hotkey,
                models.SpeedSubmission.network == network,
                models.SpeedSubmission.created_at >= datetime.utcnow() - timedelta(seconds=SUBMISSION_COOLDOWN_SECONDS)
            )
            .order_by(models.SpeedSubmission.created_at.desc())
            .first()
        )
        if recent_submission:
            seconds_ago = (datetime.utcnow() - recent_submission.created_at).total_seconds()
            remaining = SUBMISSION_COOLDOWN_SECONDS - seconds_ago
            print(f"🚫 [SUBMIT_KERNEL] Rate limit: {hotkey[:12]}... submitted {seconds_ago:.0f}s ago (cooldown: {SUBMISSION_COOLDOWN_SECONDS}s)")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {remaining:.0f} seconds before submitting again. "
                       f"Maximum 1 submission per {SUBMISSION_COOLDOWN_SECONDS // 60} minutes."
            )
        
        # Calculate solution hash for duplicate detection
        solution_hash = calculate_solution_hash(
            req.tokens_per_sec,
            req.target_sequence_length,
            req.benchmarks
        )
        
        # Get current round for this network and assign to submission (auto-expires finished rounds)
        current_round = ensure_current_round(db, network)

        # Create new speed submission
        new_submission = models.SpeedSubmission(
            network=network,
            miner_hotkey=req.miner_hotkey,
            miner_uid=miner_reg.uid,
            fork_url=req.fork_url,
            commit_hash=req.commit_hash,
            repo_hash=req.repo_hash,  # Store repository context hash
            target_sequence_length=req.target_sequence_length,
            tokens_per_sec=req.tokens_per_sec,
            vram_mb=req.vram_mb,
            benchmarks=json.dumps(req.benchmarks) if req.benchmarks else None,
            signature=req.signature,
            round_id=current_round.id,
            solution_hash=solution_hash,
            ip_address=client_ip
        )

        db.add(new_submission)
        db.commit()
        db.refresh(new_submission)

        # Record successful submission (reset failure count)
        if client_ip:
            record_success(client_ip, db)

        print(f"✅ [SUBMIT_OPT] Submission saved with ID: {new_submission.id}")
        return models.SpeedSubmissionResponse(
            submission_id=new_submission.id,
            miner_hotkey=new_submission.miner_hotkey,
            fork_url=new_submission.fork_url,
            commit_hash=new_submission.commit_hash,
            target_sequence_length=new_submission.target_sequence_length,
            tokens_per_sec=new_submission.tokens_per_sec,
            vram_mb=new_submission.vram_mb,
            benchmarks=json.loads(new_submission.benchmarks) if new_submission.benchmarks else None,
            created_at=new_submission.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [SUBMIT_KERNEL] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    return models.SpeedSubmissionResponse(
        submission_id=new_submission.id,
        miner_hotkey=new_submission.miner_hotkey,
        fork_url=new_submission.fork_url,
        commit_hash=new_submission.commit_hash,
        repo_hash=new_submission.repo_hash,
        target_sequence_length=new_submission.target_sequence_length,
        tokens_per_sec=new_submission.tokens_per_sec,
        created_at=new_submission.created_at
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# COMMIT-REVEAL ENDPOINTS (from qllm architecture)
# Prevents validators from copying miner code before evaluation
# ═══════════════════════════════════════════════════════════════════════════════════

# Commit-reveal configuration
BLOCKS_UNTIL_REVEAL = int(os.environ.get("BLOCKS_UNTIL_REVEAL", 100))  # ~20 minutes
BLOCK_TIME_SECONDS = 12  # Bittensor block time

def get_current_block() -> int:
    """Get current Bittensor block number (approximation based on time)."""
    # In production, this should query the actual chain
    # For now, use time-based approximation (genesis + seconds/12)
    import time
    # Approximate genesis time for Bittensor mainnet
    GENESIS_TIME = 1609459200  # Jan 1, 2021 UTC
    current_time = int(time.time())
    return (current_time - GENESIS_TIME) // BLOCK_TIME_SECONDS


@app.post("/commit_submission", response_model=models.CommitSubmissionResponse)
def commit_submission(
    req: models.CommitSubmissionRequest,
    request: Request,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Phase 1 of commit-reveal: Submit a commitment hash.
    
    The commitment hash is SHA256(salt + docker_image or fork_url).
    The actual submission data will be revealed after BLOCKS_UNTIL_REVEAL blocks.
    
    This prevents other validators from copying miner code before evaluation.
    """
    import traceback
    try:
        print(f"🔐 [COMMIT] Miner: {req.miner_hotkey[:8]} | Commitment: {req.commitment_hash[:16]}...")
        
        # Verify the hotkey matches the authenticated miner
        if req.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="Hotkey mismatch")

        network = normalize_network(getattr(req, "network", None))
        
        # Check if miner is registered for this network, auto-register if not
        miner_reg = db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey == hotkey,
            models.MinerRegistration.network == network
        ).first()
        
        if not miner_reg:
            miner_reg = models.MinerRegistration(
                hotkey=hotkey,
                network=network,
                uid=0
            )
            db.add(miner_reg)
            db.commit()
            print(f"✅ [COMMIT] Auto-registered miner {hotkey[:8]}... on {network} (UID will be synced from metagraph)")
        
        # Extract IP address (correctly handle reverse proxies)
        client_ip = get_client_ip(request)
        
        # Check IP ban
        is_banned, ban_reason = check_ip_ban(client_ip, db)
        if is_banned:
            raise HTTPException(status_code=403, detail=f"IP address banned: {ban_reason}")
        
        # Check for duplicate commitment hash
        existing = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.commitment_hash == req.commitment_hash
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=409,
                detail="Commitment hash already exists. Use a different salt."
            )
        
        # Calculate reveal block
        current_block = get_current_block()
        reveal_block = current_block + BLOCKS_UNTIL_REVEAL
        
        # Get current round for this network (auto-expires finished rounds)
        current_round = ensure_current_round(db, network)
        
        # Create commitment entry (not revealed yet)
        new_submission = models.SpeedSubmission(
            network=network,
            miner_hotkey=req.miner_hotkey,
            miner_uid=miner_reg.uid,
            target_sequence_length=req.target_sequence_length,
            tokens_per_sec=0.0,  # Will be set on reveal
            signature=req.signature,
            commitment_hash=req.commitment_hash,
            reveal_block=reveal_block,
            is_revealed=False,
            round_id=current_round.id,
            ip_address=client_ip
        )
        
        db.add(new_submission)
        db.commit()
        db.refresh(new_submission)
        
        # Calculate estimated reveal time
        reveal_seconds = BLOCKS_UNTIL_REVEAL * BLOCK_TIME_SECONDS
        estimated_reveal = datetime.utcnow() + timedelta(seconds=reveal_seconds)
        
        print(f"✅ [COMMIT] Commitment saved. ID: {new_submission.id}, Reveal at block: {reveal_block}")
        
        return models.CommitSubmissionResponse(
            submission_id=new_submission.id,
            commitment_hash=req.commitment_hash,
            reveal_block=reveal_block,
            estimated_reveal_time=estimated_reveal.isoformat(),
            message=f"Commitment accepted. Reveal after block {reveal_block} (~{reveal_seconds // 60} minutes)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [COMMIT] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reveal_submission", response_model=models.RevealSubmissionResponse)
def reveal_submission(
    req: models.RevealSubmissionRequest,
    request: Request,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Phase 2 of commit-reveal: Reveal the actual submission data.
    
    The salt + data must hash to the previously committed hash.
    This can only be called after the reveal block has passed.
    """
    import traceback
    try:
        print(f"🔓 [REVEAL] Miner: {req.miner_hotkey[:8]} | Submission ID: {req.submission_id}")
        
        # Verify the hotkey matches
        if req.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="Hotkey mismatch")
        
        # Get the commitment
        submission = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.id == req.submission_id,
            models.SpeedSubmission.miner_hotkey == req.miner_hotkey
        ).first()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        if submission.is_revealed:
            raise HTTPException(status_code=409, detail="Submission already revealed")
        
        # Check if reveal block has passed
        current_block = get_current_block()
        if current_block < submission.reveal_block:
            blocks_remaining = submission.reveal_block - current_block
            seconds_remaining = blocks_remaining * BLOCK_TIME_SECONDS
            raise HTTPException(
                status_code=425,  # Too Early
                detail=f"Cannot reveal yet. Wait {blocks_remaining} blocks (~{seconds_remaining // 60} minutes)"
            )
        
        # Verify the commitment hash
        # Hash should be: SHA256(salt + fork_url) or SHA256(salt + docker_image)
        reveal_data = req.docker_image if req.docker_image else req.fork_url
        expected_hash = hashlib.sha256(
            (req.commitment_salt + reveal_data).encode()
        ).hexdigest()
        
        if expected_hash != submission.commitment_hash:
            # Record failure for IP banning
            client_ip = get_client_ip(request)
            if client_ip:
                record_failure(client_ip, db)
            
            raise HTTPException(
                status_code=400,
                detail="Commitment verification failed. Hash mismatch."
            )
        
        # GitHub username dedup (at reveal time)
        network = normalize_network(getattr(submission, "network", None))
        gh_user = _github_username_from_fork_url(req.fork_url)
        if gh_user:
            existing_owner = (
                db.query(models.MinerRegistration)
                .filter(
                    models.MinerRegistration.github_username == gh_user,
                    models.MinerRegistration.network == network,
                    models.MinerRegistration.hotkey != hotkey,
                )
                .first()
            )
            if existing_owner:
                print(f"🚫 GitHub user '{gh_user}' already claimed by hotkey {existing_owner.hotkey[:12]}... — rejecting reveal from {hotkey[:12]}...")
                raise HTTPException(
                    status_code=403,
                    detail=f"GitHub username '{gh_user}' is already registered to a different hotkey. One hotkey per GitHub account."
                )
            miner_reg = db.query(models.MinerRegistration).filter(
                models.MinerRegistration.hotkey == hotkey,
                models.MinerRegistration.network == network
            ).first()
            if miner_reg and not miner_reg.github_username:
                miner_reg.github_username = gh_user
                db.commit()

        # Calculate solution hash for duplicate detection
        solution_hash = calculate_solution_hash(
            req.tokens_per_sec,
            submission.target_sequence_length,
            req.benchmarks
        )
        
        # Validate TPS bounds before accepting reveal
        if req.tokens_per_sec is not None and req.tokens_per_sec > 0:
            if req.tokens_per_sec < MIN_PLAUSIBLE_TPS or req.tokens_per_sec > MAX_PLAUSIBLE_TPS:
                raise HTTPException(
                    status_code=400,
                    detail=f"tokens_per_sec={req.tokens_per_sec:.2f} is outside plausible range "
                           f"Rejected as spam."
                )

        # Update submission with revealed data
        submission.fork_url = req.fork_url
        submission.commit_hash = req.commit_hash
        submission.tokens_per_sec = req.tokens_per_sec
        submission.vram_mb = req.vram_mb
        submission.benchmarks = json.dumps(req.benchmarks) if req.benchmarks else None
        submission.docker_image = req.docker_image
        submission.commitment_salt = req.commitment_salt
        submission.is_revealed = True
        submission.solution_hash = solution_hash
        submission.signature = req.signature
        
        db.commit()
        db.refresh(submission)
        
        print(f"✅ [REVEAL] Submission {req.submission_id} revealed successfully")
        
        return models.RevealSubmissionResponse(
            submission_id=submission.id,
            miner_hotkey=submission.miner_hotkey,
            fork_url=submission.fork_url,
            commit_hash=submission.commit_hash,
            is_revealed=True,
            message="Submission revealed successfully. Awaiting validation."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [REVEAL] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pending_reveals")
def get_pending_reveals(
    db: Session = Depends(get_db)
):
    """
    Get submissions that are pending reveal (commitment made but not yet revealed).
    """
    current_block = get_current_block()
    
    pending = db.query(models.SpeedSubmission).filter(
        models.SpeedSubmission.is_revealed == False,
        models.SpeedSubmission.commitment_hash != None
    ).all()
    
    return {
        "current_block": current_block,
        "pending_reveals": [
            {
                "submission_id": s.id,
                "miner_hotkey": s.miner_hotkey,
                "commitment_hash": s.commitment_hash[:16] + "...",
                "reveal_block": s.reveal_block,
                "blocks_remaining": max(0, s.reveal_block - current_block) if s.reveal_block else 0,
                "can_reveal": s.reveal_block <= current_block if s.reveal_block else False,
                "created_at": s.created_at.isoformat()
            }
            for s in pending
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# LOGIT VERIFICATION ENDPOINTS (from const's qllm architecture)
# Verifies miners are running the actual model, not returning bogus values
# ═══════════════════════════════════════════════════════════════════════════════════

@app.post("/record_verification")
def record_verification(
    submission_id: int,
    verified: bool,
    cosine_similarity: Optional[float] = None,
    max_abs_diff: Optional[float] = None,
    throughput: Optional[float] = None,
    reason: Optional[str] = None,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_validator_signature)
):
    """
    Record logit verification result for a submission.
    Called by validator after running logit verification.
    Requires validator authentication.
    """
    submission = db.query(models.SpeedSubmission).filter(
        models.SpeedSubmission.id == submission_id
    ).first()
    
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # Validate cosine_similarity range [0.0, 1.0]
    if cosine_similarity is not None and (cosine_similarity < 0.0 or cosine_similarity > 1.0):
        raise HTTPException(status_code=400, detail=f"cosine_similarity must be in [0.0, 1.0], got {cosine_similarity}")
    # Validate max_abs_diff is non-negative
    if max_abs_diff is not None and max_abs_diff < 0.0:
        raise HTTPException(status_code=400, detail=f"max_abs_diff must be >= 0, got {max_abs_diff}")
    # Validate throughput if provided
    if throughput is not None and (throughput < 0 or throughput > MAX_PLAUSIBLE_TPS):
        raise HTTPException(status_code=400, detail=f"throughput={throughput} outside plausible range")

    # Update verification fields
    submission.logit_verification_passed = verified
    submission.cosine_similarity = cosine_similarity
    submission.max_abs_diff = max_abs_diff
    submission.throughput_verified = throughput
    submission.verification_reason = reason
    
    db.commit()
    
    status = "✅ PASSED" if verified else "❌ FAILED"
    print(f"🔍 [VERIFY] Submission {submission_id}: {status}")
    if cosine_similarity is not None:
        print(f"    Cosine similarity: {cosine_similarity:.4f}")
    if max_abs_diff is not None:
        print(f"    Max abs diff: {max_abs_diff:.4f}")
    if reason:
        print(f"    Reason: {reason}")
    
    return {"status": "recorded", "verified": verified}


@app.get("/verification_stats")
def get_verification_stats(
    round_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get verification statistics for submissions.
    """
    query = db.query(models.SpeedSubmission).filter(
        models.SpeedSubmission.is_revealed == True
    )
    
    if round_id:
        query = query.filter(models.SpeedSubmission.round_id == round_id)
    
    submissions = query.all()
    
    total = len(submissions)
    verified_passed = sum(1 for s in submissions if s.logit_verification_passed == True)
    verified_failed = sum(1 for s in submissions if s.logit_verification_passed == False)
    pending_verification = sum(1 for s in submissions if s.logit_verification_passed is None)
    
    avg_cosine = None
    avg_max_diff = None
    
    verified_submissions = [s for s in submissions if s.cosine_similarity is not None]
    if verified_submissions:
        avg_cosine = sum(s.cosine_similarity for s in verified_submissions) / len(verified_submissions)
        diff_submissions = [s for s in verified_submissions if s.max_abs_diff is not None]
        if diff_submissions:
            avg_max_diff = sum(s.max_abs_diff for s in diff_submissions) / len(diff_submissions)
    
    return {
        "total_submissions": total,
        "verified_passed": verified_passed,
        "verified_failed": verified_failed,
        "pending_verification": pending_verification,
        "pass_rate": verified_passed / (verified_passed + verified_failed) if (verified_passed + verified_failed) > 0 else None,
        "avg_cosine_similarity": avg_cosine,
        "avg_max_abs_diff": avg_max_diff
    }


@app.get("/get_submission_stats")
def get_submission_stats(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    limit = min(limit, 200)
    """
    Get submission statistics for the system.
    Returns recent submissions with performance metrics.
    Only returns unvalidated submissions for validator to process.
    """
    import traceback
    try:
        # Get recent unvalidated submissions
        recent_submissions = (
            db.query(models.SpeedSubmission)
            .filter(models.SpeedSubmission.validated == False)
            .order_by(models.SpeedSubmission.created_at.desc())
            .limit(limit)
            .all()
        )

        # Calculate stats
        total_submissions = db.query(models.SpeedSubmission).count()

        if recent_submissions:
            avg_tokens_per_sec = sum(s.tokens_per_sec for s in recent_submissions) / len(recent_submissions)
            max_tokens_per_sec = max(s.tokens_per_sec for s in recent_submissions)
            min_tokens_per_sec = min(s.tokens_per_sec for s in recent_submissions)
        else:
            avg_tokens_per_sec = max_tokens_per_sec = min_tokens_per_sec = 0.0

        # Get top performers
        top_submissions = (
            db.query(models.SpeedSubmission)
            .order_by(models.SpeedSubmission.tokens_per_sec.desc())
            .limit(10)
            .all()
        )

        def parse_benchmarks(benchmarks_str):
            """Parse benchmarks JSON string with error handling."""
            if not benchmarks_str:
                return None
            try:
                return json.loads(benchmarks_str)
            except Exception as e:
                print(f"Error parsing benchmarks: {e}")
                return None

        return {
            "total_submissions": total_submissions,
            "recent_submissions": [
                {
                    "id": s.id,
                    "miner_hotkey": s.miner_hotkey,
                    "fork_url": "[HIDDEN]",
                    "commit_hash": "[HIDDEN]",
                    "repo_hash": s.repo_hash,
                    "target_sequence_length": s.target_sequence_length,
                    "tokens_per_sec": s.tokens_per_sec,
                    "vram_mb": s.vram_mb,
                    "benchmarks": parse_benchmarks(s.benchmarks),
                    "validated": s.validated,
                    "created_at": s.created_at.isoformat()
                }
                for s in recent_submissions
            ],
            "stats": {
                "avg_tokens_per_sec": round(avg_tokens_per_sec, 2),
                "max_tokens_per_sec": round(max_tokens_per_sec, 2),
                "min_tokens_per_sec": round(min_tokens_per_sec, 2),
                "total_submissions": total_submissions
            },
            "top_performers": [
                {
                    "id": s.id,
                    "miner_hotkey": s.miner_hotkey,
                    "tokens_per_sec": s.tokens_per_sec,
                    "target_sequence_length": s.target_sequence_length,
                    "created_at": s.created_at.isoformat()
                }
                for s in top_submissions
            ]
        }
    except Exception as e:
        print(f"Error in get_submission_stats: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/get_pending_validations")
def get_pending_validations(
    limit: int = 10,
    network: Optional[str] = None,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_validator_signature)
):
    """
    Validator-only endpoint: Get unvalidated submissions with full details (including fork_url) for the given network.
    Requires validator authentication. fork_url is never exposed to public endpoints.
    """
    limit = min(limit, 100)
    network = normalize_network(network)

    # Exclude submissions from flagged miners
    flagged_hotkeys_q = (
        db.query(models.MinerRegistration.hotkey)
        .filter(models.MinerRegistration.is_flagged == True)
    )
    flagged_hotkeys = {r.hotkey for r in flagged_hotkeys_q.all()}

    submissions_q = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.network == network)
        .filter(models.SpeedSubmission.validated == False)
        .filter(models.SpeedSubmission.is_revealed == True)
        .order_by(models.SpeedSubmission.created_at.desc())
    )
    if flagged_hotkeys:
        submissions_q = submissions_q.filter(
            ~models.SpeedSubmission.miner_hotkey.in_(flagged_hotkeys)
        )
    submissions = submissions_q.limit(limit).all()

    return {
        "pending_count": len(submissions),
        "submissions": [
            {
                "id": s.id,
                "miner_hotkey": s.miner_hotkey,
                "fork_url": s.fork_url,
                "commit_hash": s.commit_hash,
                "repo_hash": s.repo_hash,
                "docker_image": s.docker_image,
                "target_sequence_length": s.target_sequence_length,
                "tokens_per_sec": s.tokens_per_sec,
                "vram_mb": s.vram_mb,
                "benchmarks": s.benchmarks,
                "validated": s.validated,
                "is_revealed": s.is_revealed,
                "created_at": s.created_at.isoformat()
            }
            for s in submissions
        ]
    }

@app.post("/mark_validated")
def mark_validated(
    req: dict,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_validator_signature)
):
    """
    Mark a submission as validated and record its score.
    Used by validators to avoid re-evaluating the same submission.
    Requires validator authentication.
    """
    submission_id = req.get("submission_id")
    if not submission_id:
        raise HTTPException(status_code=400, detail="submission_id required")

    submission = db.query(models.SpeedSubmission).filter(models.SpeedSubmission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    submission.validated = True
    
    # Update score if provided (clamp to [0.0, 1.0])
    score = req.get("score")
    if score is not None:
        score = max(0.0, min(1.0, float(score)))
        submission.score = score
        print(f"📊 [MARK_VALIDATED] Submission {submission_id}: score={score:.4f}")

    actual_tps = req.get("actual_tokens_per_sec")
    if actual_tps is not None:
        actual_tps = float(actual_tps)
        if actual_tps < MIN_PLAUSIBLE_TPS or actual_tps > MAX_PLAUSIBLE_TPS:
            raise HTTPException(
                status_code=400,
                detail=f"actual_tokens_per_sec={actual_tps:.2f} is outside plausible range "
                       f"Rejected."
            )
        old_tps = submission.tokens_per_sec
        submission.validated_tokens_per_sec = actual_tps
    
        if old_tps and actual_tps > 0:
            discrepancy_pct = abs(old_tps - actual_tps) / max(old_tps, 1) * 100
            discrepancy_abs = abs(old_tps - actual_tps)
            
            # Server-side thresholds (miners cannot bypass)
            DISCREPANCY_PCT_THRESHOLD = float(os.environ.get("DISCREPANCY_PCT_THRESHOLD", "50.0"))  # 50% difference
            DISCREPANCY_ABS_THRESHOLD = float(os.environ.get("DISCREPANCY_ABS_THRESHOLD", "1000.0"))  # 1000 tok/s minimum
            
            should_flag = (
                discrepancy_pct > DISCREPANCY_PCT_THRESHOLD and 
                discrepancy_abs > DISCREPANCY_ABS_THRESHOLD
            )
            
            if should_flag:
                print(f"🚨 [MARK_VALIDATED] TPS MISMATCH for {submission.miner_hotkey[:12]}...: "
                      f"claimed={old_tps:.2f}, actual={actual_tps:.2f} "
                      f"(diff: {discrepancy_pct:.1f}%, abs: {discrepancy_abs:.0f} tok/s)")
                
                # Auto-flag miner for spam (claiming unrealistic values)
                miner_reg = db.query(models.MinerRegistration).filter(
                    models.MinerRegistration.hotkey == submission.miner_hotkey,
                    models.MinerRegistration.network == (getattr(submission, "network", None) or DEFAULT_NETWORK)
                ).first()
                
                if miner_reg and not miner_reg.is_flagged:
                    miner_reg.is_flagged = True
                    miner_reg.flag_reason = (
                        f"Large TPS discrepancy: claimed {old_tps:.0f} tok/s, "
                        f"actual {actual_tps:.0f} tok/s (diff: {discrepancy_pct:.1f}%, abs: {discrepancy_abs:.0f} tok/s)"
                    )
                    db.commit()
                    print(f"🚫 [MARK_VALIDATED] Auto-flagged miner {submission.miner_hotkey[:12]}... for spam")
            else:
                print(f"📊 [MARK_VALIDATED] Submission {submission_id}: validated_tps={actual_tps:.2f} "
                      f"(claimed={old_tps:.2f}, diff={discrepancy_pct:.1f}%, abs={discrepancy_abs:.0f} tok/s)")
        else:
            print(f"📊 [MARK_VALIDATED] Submission {submission_id}: validated_tps={actual_tps:.2f}")

    db.commit()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Aggregate scores per miner per league
    # ═══════════════════════════════════════════════════════════════════════════
    if score is not None and submission.miner_hotkey:
        try:
            # Determine league from target_sequence_length
            league = get_league_for_seq_len(submission.target_sequence_length)
            sub_network = getattr(submission, "network", None) or DEFAULT_NETWORK
            
            # Find existing MinerScore by hotkey + league + network
            miner_score = db.query(models.MinerScore).filter(
                models.MinerScore.hotkey == submission.miner_hotkey,
                models.MinerScore.league == league,
                models.MinerScore.network == sub_network,
            ).first()
            
            if miner_score:
                # Update existing score using exponential moving average
                # Alpha = 0.3 means new score has 30% weight, old score has 70% weight
                alpha = 0.3
                miner_score.score = alpha * float(score) + (1 - alpha) * miner_score.score
                miner_score.tasks_completed += 1
                miner_score.last_updated = datetime.utcnow()
                print(f"📊 [MINER_SCORES] Updated {submission.miner_hotkey[:8]}... in {league}: score={miner_score.score:.4f}, tasks={miner_score.tasks_completed}")
            else:
                # Create new MinerScore entry
                # First, ensure MinerRegistration exists for this network
                registration = db.query(models.MinerRegistration).filter(
                    models.MinerRegistration.hotkey == submission.miner_hotkey,
                    models.MinerRegistration.network == sub_network
                ).first()
                
                if not registration:
                    registration = models.MinerRegistration(
                        hotkey=submission.miner_hotkey,
                        network=sub_network,
                        uid=submission.miner_uid or 0
                    )
                    db.add(registration)
                
                # Create MinerScore (model_name not in SpeedSubmission; use "Unknown")
                miner_score = models.MinerScore(
                    hotkey=submission.miner_hotkey,
                    model_name="Unknown",
                    league=league,
                    network=sub_network,
                    score=float(score),
                    tasks_completed=1
                )
                db.add(miner_score)
                print(f"📊 [MINER_SCORES] Created {submission.miner_hotkey[:8]}... in {league}: score={score:.4f}")
            
            db.commit()
        except Exception as e:
            print(f"⚠️ [MINER_SCORES] Failed to update miner_scores: {e}")
            db.rollback()
            # Don't fail the request if miner_scores update fails
    
    # Record successful validation (reset failure count for IP)
    if submission.ip_address:
        record_success(submission.ip_address, db)

    return {"status": "ok", "submission_id": submission_id, "score": submission.score}

@app.post("/record_failure")
def record_failure_endpoint(
    req: dict,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_validator_signature)
):
    """Record a failed submission for IP tracking. Requires validator authentication."""
    ip_address = req.get("ip_address")
    if ip_address:
        record_failure(ip_address, db)
        return {"status": "ok", "ip_address": ip_address, "message": "Failure recorded"}
    return {"status": "ok", "message": "No IP address provided"}

@app.post("/register_miner")
def register_miner(
    req: models.RegisterMinerRequest,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Register a miner with a specific model and league.
    Miners can register multiple times for different (model, league) combinations.
    """
    if req.league not in LEAGUES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid league. Must be one of: {', '.join(LEAGUES)}"
        )

    network = normalize_network(getattr(req, "network", None))

    # Check if already registered for this (hotkey, model, league, network) combo
    existing = db.query(models.MinerScore).filter(
        models.MinerScore.hotkey == hotkey,
        models.MinerScore.model_name == req.model_name,
        models.MinerScore.league == req.league,
        models.MinerScore.network == network
    ).first()

    miner_uid = req.uid if req.uid is not None else 0

    if existing:
        # Check if MinerRegistration exists for this network, create or update
        registration = db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey == hotkey,
            models.MinerRegistration.network == network
        ).first()

        if not registration:
            new_registration = models.MinerRegistration(
                hotkey=hotkey,
                network=network,
                uid=miner_uid
            )
            db.add(new_registration)
            db.commit()
            print(f"✅ [REGISTER] Created missing MinerRegistration for {hotkey[:8]} on {network} with UID={miner_uid}")
        elif registration.uid == 0 and miner_uid > 0:
            registration.uid = miner_uid
            db.commit()
            print(f"✅ [REGISTER] Updated UID for {hotkey[:8]} on {network}: 0 -> {miner_uid}")

        print(f"ℹ️ [REGISTER] Miner {hotkey[:8]} already registered for {req.model_name} in {req.league} on {network}")
        return {
            "status": "already_registered",
            "hotkey": hotkey,
            "model_name": req.model_name,
            "league": req.league,
            "network": network,
            "current_score": existing.score,
            "uid": registration.uid if registration else miner_uid
        }

    # Create new registration
    new_score = models.MinerScore(
        hotkey=hotkey,
        model_name=req.model_name,
        league=req.league,
        network=network,
        score=0.0,
        tasks_completed=0
    )
    db.add(new_score)

    # Create MinerRegistration entry with actual UID for this network
    new_registration = models.MinerRegistration(
        hotkey=hotkey,
        network=network,
        uid=miner_uid
    )
    db.add(new_registration)

    db.commit()

    print(f"✅ [REGISTER] Miner {hotkey[:8]} registered for {req.model_name} in {req.league} on {network} with UID={miner_uid}")
    return {
        "status": "registered",
        "hotkey": hotkey,
        "model_name": req.model_name,
        "league": req.league,
        "network": network,
        "uid": miner_uid
    }

@app.post("/sync_uids")
def sync_uids(
    req: dict,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_validator_signature)
):
    """
    Bulk-update miner UIDs and coldkeys from the metagraph for a given network.
    """
    network = normalize_network(req.get("network"))
    uid_map = req.get("uid_map") or req
    if isinstance(uid_map, dict) and "network" in uid_map and "uid_map" in uid_map:
        uid_map = uid_map.get("uid_map", {})
    if not isinstance(uid_map, dict):
        raise HTTPException(status_code=400, detail="uid_map required (dict of hotkey -> uid)")
    coldkey_map = req.get("coldkey_map") or {}

    if not uid_map:
        return {"status": "ok", "updated": 0, "total_entries": 0, "network": network}
    
    # OPTIMIZATION: Bulk fetch all existing registrations in one query instead of N queries
    hotkeys_list = list(uid_map.keys())
    existing_regs = {
        reg.hotkey: reg
        for reg in db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey.in_(hotkeys_list),
            models.MinerRegistration.network == network
        ).all()
    }
    
    updated = 0
    new_regs = []
    
    for hotkey, uid in uid_map.items():
        registration = existing_regs.get(hotkey)
        if registration:
            changed = False
            if registration.uid != uid:
                old_uid = registration.uid
                registration.uid = uid
                changed = True
                if updated < 5:
                    print(f"[SYNC_UIDS] Updated {hotkey[:12]}... on {network}: UID {old_uid} -> {uid}")
            ck = coldkey_map.get(hotkey)
            if ck and registration.coldkey != ck:
                registration.coldkey = ck
                changed = True
                if updated < 5:
                    print(f"[SYNC_UIDS] Updated coldkey for {hotkey[:12]}... on {network}")
            if changed:
                updated += 1
        else:
            new_reg = models.MinerRegistration(
                hotkey=hotkey, network=network, uid=uid,
                coldkey=coldkey_map.get(hotkey)
            )
            new_regs.append(new_reg)
            updated += 1
            if len(new_regs) <= 5:
                print(f"[SYNC_UIDS] Created registration for {hotkey[:12]}... on {network} with UID={uid}")
    
    if new_regs:
        db.bulk_save_objects(new_regs)
    
    stale_count = (
        db.query(models.MinerRegistration)
        .filter(
            models.MinerRegistration.network == network,
            ~models.MinerRegistration.hotkey.in_(hotkeys_list),
        )
        .delete(synchronize_session="fetch")
    )
    if stale_count > 0:
        print(f"[SYNC_UIDS] Removed {stale_count} deregistered miners from {network}")
    
    db.commit()
    print(f"[SYNC_UIDS] Synced {updated} UIDs for network={network} "
          f"({len(new_regs)} new, {updated - len(new_regs)} updated, {stale_count} removed)")
    return {
        "status": "ok", "updated": updated, "removed": stale_count,
        "total_entries": len(uid_map), "network": network,
    }


@app.get("/flagged_miners")
def get_flagged_miners(
    network: Optional[str] = None,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_validator_signature)
):
    """List all miners flagged by spam checks. Requires validator authentication.
    """
    network = normalize_network(network)
    flagged = (
        db.query(models.MinerRegistration)
        .filter(
            models.MinerRegistration.network == network,
            models.MinerRegistration.is_flagged == True,
        )
        .all()
    )
    return {
        "network": network,
        "count": len(flagged),
        "flagged": [
            {
                "hotkey": r.hotkey,
                "uid": r.uid,
                "flag_reason": r.flag_reason,
            }
            for r in flagged
        ],
    }


@app.post("/flag_miner")
def flag_miner(
    req: dict,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(auth.verify_validator_signature)
):
    """Manually flag or unflag a miner. Requires validator authentication."""
    hotkey = req.get("hotkey")
    if not hotkey:
        raise HTTPException(status_code=400, detail="hotkey required")
    network = normalize_network(req.get("network"))
    flag = req.get("flag", True)
    reason = req.get("reason", "Manual flag")
    reg = db.query(models.MinerRegistration).filter(
        models.MinerRegistration.hotkey == hotkey,
        models.MinerRegistration.network == network,
    ).first()
    if not reg:
        raise HTTPException(status_code=404, detail="Miner not found")
    reg.is_flagged = flag
    reg.flag_reason = reason if flag else None
    db.commit()
    action = "flagged" if flag else "unflagged"
    print(f"[FLAG] Miner {hotkey[:12]}... {action} on {network}: {reason}")
    return {"status": action, "hotkey": hotkey, "network": network}


# Updated reward distribution for top 4
REWARD_DISTRIBUTION = [0.60, 0.25, 0.10, 0.05]  # 60%, 25%, 10%, 5%
TOP_N_MINERS = 4

@app.get("/get_weights")
def get_weights(
    round_id: Optional[int] = None,
    network: Optional[str] = None,
    completed_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get weights for top 4 performers in a specific round for the given network.
    
    Reward distribution:
    - 1st place: 60%
    - 2nd place: 25%
    - 3rd place: 10%
    - 4th place: 5%
    
    Args:
        round_id: Optional round ID. If not specified, uses the most recent completed round.
        network: "finney" (mainnet) or "test" (testnet). Default finney.
        completed_only: If True and round_id not specified, only use a completed round (never active). Default True.
    """
    network = normalize_network(network)
    if round_id is None:
        round_obj = (
            db.query(models.CompetitionRound)
            .filter(models.CompetitionRound.network == network)
            .filter(models.CompetitionRound.status == "completed")
            .order_by(models.CompetitionRound.round_number.desc())
            .first()
        )
        if not round_obj and not completed_only:
            round_obj = (
                db.query(models.CompetitionRound)
                .filter(models.CompetitionRound.network == network)
                .filter(models.CompetitionRound.status == "active")
                .order_by(models.CompetitionRound.round_number.desc())
                .first()
            )
            if round_obj:
                print(f"[WEIGHTS] No completed round yet, using active round {round_obj.round_number} for network={network}")
    else:
        round_obj = db.query(models.CompetitionRound).filter(
            models.CompetitionRound.id == round_id,
            models.CompetitionRound.network == network
        ).first()
    
    if not round_obj:
        print("[WEIGHTS] No round found (completed or active)")
        return GetWeightsResponse(epoch=int(time.time()), weights=[], round_id=None, round_number=None, round_status=None, winner_hotkey=None)
    
    # Get all validated submissions for this round
    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_obj.id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )
    
    if not submissions:
        print(f"[WEIGHTS] No validated submissions in round {round_obj.round_number}")
        return GetWeightsResponse(epoch=int(time.time()), weights=[], round_id=round_obj.id, round_number=round_obj.round_number, round_status=round_obj.status, winner_hotkey=round_obj.winner_hotkey)
    
    # Calculate rankings with first-submission-wins logic
    # Note: baseline_submission_id on a round is the baseline for the NEXT round
    # When calculating weights for a round, we need the baseline that was active DURING that round
    # For the first round (lowest round_number): no baseline (None)
    # For subsequent rounds: use previous round's baseline_submission_id
    
    # Find the first round (lowest round_number) for this network
    first_round = (
        db.query(models.CompetitionRound)
        .filter(models.CompetitionRound.network == round_obj.network)
        .order_by(models.CompetitionRound.round_number.asc())
        .first()
    )
    
    if first_round and round_obj.round_number == first_round.round_number:
        # This is the first round - no baseline
        baseline_id = None
    else:
        # Get previous round's baseline (same network)
        prev_round = db.query(models.CompetitionRound).filter(
            models.CompetitionRound.network == round_obj.network,
            models.CompetitionRound.round_number == round_obj.round_number - 1
        ).first()
        baseline_id = prev_round.baseline_submission_id if prev_round else None
    
    rankings = calculate_rankings(submissions, baseline_id, db)
    
    if not rankings:
        print(f"[WEIGHTS] No valid rankings for round {round_obj.round_number}")
        return GetWeightsResponse(epoch=int(time.time()), weights=[], round_id=round_obj.id, round_number=round_obj.round_number, round_status=round_obj.status, winner_hotkey=round_obj.winner_hotkey)
    
    # Distribute weights to top 4
    weights = []
    print(f"[WEIGHTS] Distributing rewards for round {round_obj.round_number}:")
    
    for i, ranking in enumerate(rankings[:TOP_N_MINERS]):
        weight = REWARD_DISTRIBUTION[i] if i < len(REWARD_DISTRIBUTION) else 0.0
        
        # Get UID from MinerRegistration for this network
        miner_reg = db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey == ranking["miner_hotkey"],
            models.MinerRegistration.network == round_obj.network
        ).first()
        uid = miner_reg.uid if miner_reg else 0
        
        tokens_per_sec = ranking.get("tokens_per_sec")
        github_username = None
        sub = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.id == ranking["submission_id"]
        ).first()
        if sub and sub.fork_url:
            github_username = _github_username_from_fork_url(sub.fork_url)
        
        weights.append(WeightEntry(
            uid=uid,
            hotkey=ranking["miner_hotkey"],
            weight=weight,
            tokens_per_sec=tokens_per_sec,
            github_username=github_username
        ))
        
        print(f"  #{ranking['rank']}: {ranking['miner_hotkey'][:12]}... - "
              f"weight={weight:.2%} "
              f"(weighted_score={ranking['weighted_score']:.0f})")
    
    return GetWeightsResponse(
        epoch=int(time.time()), 
        weights=weights, 
        round_id=round_obj.id, 
        round_number=round_obj.round_number, 
        round_status=round_obj.status,
        winner_hotkey=round_obj.winner_hotkey
    )

# ==================== ROUND MANAGEMENT ENDPOINTS ====================

def _finalize_round_impl(round_obj, db):
    if round_obj.status != "active":
        return
    round_obj.status = "evaluating"
    db.commit()
    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_obj.id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )
    if not submissions:
        round_obj.status = "completed"
        db.commit()
        return
    rankings = calculate_rankings(submissions, round_obj.baseline_submission_id, db)
    if rankings:
        winner = rankings[0]
        round_obj.winner_hotkey = winner["miner_hotkey"]
        round_obj.baseline_submission_id = winner["submission_id"]
        db.commit()
        winner_submission = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.id == winner["submission_id"]
        ).first()
        if winner_submission:
            winner_submission.is_baseline = True
            db.commit()
        print(f"✅ [FINALIZE] Round {round_obj.round_number} winner: {round_obj.winner_hotkey}")
    else:
        print(f"⚠️ [FINALIZE] Round {round_obj.round_number} has no valid rankings (no submissions passed all filters)")
        print(f"   Validated submissions: {len(submissions)}")
        print(f"   Possible reasons:")
        print(f"   - No submissions passed logit verification")
        print(f"   - No submissions have validated_tokens_per_sec")
        print(f"   - All submissions below baseline")
    round_obj.status = "completed"
    db.commit()


def ensure_current_round(db, network: Optional[str] = None):
    network = normalize_network(network)
    now = datetime.utcnow()
    expired = (
        db.query(models.CompetitionRound)
        .filter(models.CompetitionRound.network == network)
        .filter(models.CompetitionRound.status == "active")
        .filter(models.CompetitionRound.end_time < now)
        .order_by(models.CompetitionRound.round_number.asc())
        .all()
    )
    for r in expired:
        _finalize_round_impl(r, db)
    current_round = (
        db.query(models.CompetitionRound)
        .filter(models.CompetitionRound.network == network)
        .filter(models.CompetitionRound.status == "active")
        .order_by(models.CompetitionRound.round_number.desc())
        .first()
    )
    if not current_round:
        last_round = (
            db.query(models.CompetitionRound)
            .filter(models.CompetitionRound.network == network)
            .order_by(models.CompetitionRound.round_number.desc())
            .first()
        )
        next_round_number = (last_round.round_number + 1) if last_round else 1
        current_round = models.CompetitionRound(
            round_number=next_round_number,
            network=network,
            start_time=now,
            end_time=now + timedelta(hours=48),
            status="active"
        )
        db.add(current_round)
        db.commit()
        db.refresh(current_round)
        print(f"✅ [ROUND] Created new round #{current_round.round_number} for network={network}")
    return current_round


@app.get("/get_current_round", response_model=models.RoundResponse)
def get_current_round(
    network: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get the current active round for the given network. Expires finished rounds and creates a new one if needed."""
    try:
        network = normalize_network(network)
        current_round = ensure_current_round(db, network)
        now = datetime.utcnow()
        time_remaining = max(0, int((current_round.end_time - now).total_seconds()))
        
        # Count submissions in this round
        submission_count = (
            db.query(models.SpeedSubmission)
            .filter(models.SpeedSubmission.round_id == current_round.id)
            .count()
        )
        
        # Create response with proper datetime handling
        response_data = {
            "id": current_round.id,
            "round_number": current_round.round_number,
            "start_time": current_round.start_time.isoformat() if isinstance(current_round.start_time, datetime) else str(current_round.start_time),
            "end_time": current_round.end_time.isoformat() if isinstance(current_round.end_time, datetime) else str(current_round.end_time),
            "status": current_round.status,
            "time_remaining_seconds": time_remaining,
            "baseline_submission_id": current_round.baseline_submission_id,
            "winner_hotkey": current_round.winner_hotkey,
            "total_submissions": submission_count
        }
        
        print(f"✅ [GET_CURRENT_ROUND] Returning round {current_round.round_number}")
        return models.RoundResponse(**response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"❌ [GET_CURRENT_ROUND] Error: {e}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting current round: {str(e)}"
        )

@app.post("/create_round", response_model=models.RoundResponse)
def create_round(
    req: models.CreateRoundRequest,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_validator_signature)
):
    """Create a new competition round. Requires validator authentication.
    
    Request body (can send empty {} to use defaults):
    - duration_hours: int (default: 48)
    - baseline_submission_id: Optional[int] (default: None)
    
    Example:
    - Empty body: {}
    - With duration: {"duration_hours": 24}
    - With baseline: {"duration_hours": 48, "baseline_submission_id": 123}
    """
    try:
        # Get last round number
        last_round = (
            db.query(models.CompetitionRound)
            .order_by(models.CompetitionRound.round_number.desc())
            .first()
        )
        
        next_round_number = (last_round.round_number + 1) if last_round else 1
        
        # Mark previous round as completed
        if last_round and last_round.status == "active":
            last_round.status = "completed"
            db.commit()
        
        # Handle baseline_submission_id: convert 0 to None, validate if provided
        baseline_submission_id = req.baseline_submission_id
        if baseline_submission_id == 0 or baseline_submission_id is None:
            baseline_submission_id = None
        else:
            # Validate that the baseline submission exists
            baseline_submission = db.query(models.SpeedSubmission).filter(
                models.SpeedSubmission.id == baseline_submission_id
            ).first()
            if not baseline_submission:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Baseline submission {baseline_submission_id} not found"
                )
        
        # Create new round
        from datetime import timedelta
        now = datetime.utcnow()
        new_round = models.CompetitionRound(
            round_number=next_round_number,
            network=network,
            start_time=now,
            end_time=now + timedelta(hours=req.duration_hours),
            status="active",
            baseline_submission_id=baseline_submission_id
        )
        
        db.add(new_round)
        db.commit()
        db.refresh(new_round)
        
        # Count submissions in this round (should be 0 for new round)
        submission_count = (
            db.query(models.SpeedSubmission)
            .filter(models.SpeedSubmission.round_id == new_round.id)
            .count()
        )
        
        print(f"✅ [ROUND] Created round #{next_round_number} (baseline: {req.baseline_submission_id})")
        
        # Format response with proper datetime handling
        time_remaining = max(0, int((new_round.end_time - datetime.utcnow()).total_seconds()))
        
        return models.RoundResponse(
            id=new_round.id,
            round_number=new_round.round_number,
            start_time=new_round.start_time.isoformat() if isinstance(new_round.start_time, datetime) else str(new_round.start_time),
            end_time=new_round.end_time.isoformat() if isinstance(new_round.end_time, datetime) else str(new_round.end_time),
            status=new_round.status,
            time_remaining_seconds=time_remaining,
            baseline_submission_id=new_round.baseline_submission_id,
            winner_hotkey=new_round.winner_hotkey,
            total_submissions=submission_count
        )
        
    except Exception as e:
        import traceback
        error_msg = f"❌ [CREATE_ROUND] Error: {e}"
        print(error_msg)
        print(traceback.format_exc())
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating round: {str(e)}"
        )

def calculate_rankings(
    submissions: List[models.SpeedSubmission],
    baseline_submission_id: Optional[int],
    db: Session
) -> List[Dict]:
    """
    Calculate rankings with first-submission-wins logic.
    
    Ranking criteria (in order):
    1. Logit verification must pass (filter out failed/pending)
    2. Weighted score (tokens_per_sec * league_multiplier) - DESC
    3. Created timestamp (first submission wins) - ASC
    4. Submission ID (tiebreaker) - ASC
    
    Note: Submissions that failed logit verification are excluded from rankings.
    This prevents miners from returning bogus values quickly.
    """
    # Get baseline if exists
    baseline = None
    if baseline_submission_id:
        baseline = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.id == baseline_submission_id
        ).first()
    
    # Pre-fetch flagged hotkeys so we can exclude them from rankings.
    # Flagged miners' existing submissions must not win rounds or become baselines.
    flagged_hotkeys = {
        r.hotkey
        for r in db.query(models.MinerRegistration.hotkey).filter(
            models.MinerRegistration.is_flagged == True
        ).all()
    }

    # Calculate weighted scores
    ranked_submissions = []
    for sub in submissions:
        # ═══════════════════════════════════════════════════════════════════════
        # FLAGGED MINER FILTER
        # ═══════════════════════════════════════════════════════════════════════
        if sub.miner_hotkey in flagged_hotkeys:
            print(f"[RANKING] Skipping {sub.miner_hotkey[:8]}...: Miner is flagged")
            continue

        # ═══════════════════════════════════════════════════════════════════════
        # LOGIT VERIFICATION FILTER (from qllm architecture)
        # Skip submissions that failed logit verification or are not revealed
        # ═══════════════════════════════════════════════════════════════════════
        
        # Skip unrevealed submissions (commit-reveal mechanism)
        if sub.is_revealed == False:
            print(f"[RANKING] Skipping {sub.miner_hotkey[:8]}...: Not revealed yet")
            continue
        
        # Logit verification is MANDATORY. Only submissions that explicitly
        # passed (True) can participate in rankings. Submissions that failed
        # (False) or were never verified (None) are excluded.
        if sub.logit_verification_passed != True:
            reason = "failed" if sub.logit_verification_passed == False else "not verified"
            print(f"[RANKING] Skipping {sub.miner_hotkey[:8]}...: Logit verification {reason}")
            continue
        
        # ── ANTI-SPAM: Only use validator-measured TPS ──
        # Never fall back to miner-claimed value. This prevents spam submissions
        # from ranking high before validation. Unvalidated submissions are excluded above.
        if sub.validated_tokens_per_sec is None:
            print(f"[RANKING] Skipping {sub.miner_hotkey[:8]}...: No validated_tokens_per_sec (not yet validated)")
            continue
        
        effective_tps = sub.validated_tokens_per_sec

        # Skip if below baseline (for round 2+)
        if baseline:
            baseline_tps = baseline.validated_tokens_per_sec or baseline.tokens_per_sec
            if baseline_tps is not None:
                baseline_league = get_league_for_seq_len(baseline.target_sequence_length)
                baseline_multiplier = LEAGUE_MULTIPLIERS.get(baseline_league, 1.0)
                baseline_weighted = baseline_tps * baseline_multiplier

                sub_league = get_league_for_seq_len(sub.target_sequence_length)
                sub_multiplier = LEAGUE_MULTIPLIERS.get(sub_league, 1.0)
                sub_weighted = effective_tps * sub_multiplier

                if sub_weighted <= baseline_weighted:
                    continue

        league = get_league_for_seq_len(sub.target_sequence_length)
        multiplier = LEAGUE_MULTIPLIERS.get(league, 1.0)
        weighted_score = effective_tps * multiplier

        ranked_submissions.append({
            "submission_id": sub.id,
            "miner_hotkey": sub.miner_hotkey,
            "tokens_per_sec": effective_tps,
            "target_sequence_length": sub.target_sequence_length,
            "league": league,
            "multiplier": multiplier,
            "weighted_score": weighted_score,
            "created_at": sub.created_at,
            "solution_hash": sub.solution_hash,
            # Verification info (from qllm architecture)
            "logit_verification_passed": sub.logit_verification_passed,
            "cosine_similarity": sub.cosine_similarity,
            "max_abs_diff": sub.max_abs_diff,
            "throughput_verified": sub.throughput_verified
        })
    
    # Sort by: weighted_score DESC, created_at ASC, submission_id ASC
    ranked_submissions.sort(
        key=lambda x: (
            -x["weighted_score"],  # Negative for descending
            x["created_at"],  # Ascending (first wins)
            x["submission_id"]  # Tiebreaker
        )
    )
    
    # Add rank numbers
    for i, sub in enumerate(ranked_submissions, start=1):
        sub["rank"] = i
    
    return ranked_submissions

@app.get("/get_submission_rate")
def get_submission_rate(
    window_minutes: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get current submission rate (submissions per minute).
    Used by validators to adjust polling frequency dynamically.
    
    Args:
        window_minutes: Time window in minutes to calculate rate (default: 10)
    
    Returns:
        Dictionary with submissions_per_minute, recent_submissions, window_minutes
    """
    window_minutes = max(1, min(window_minutes, 1440))
    cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
    
    recent_submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.created_at >= cutoff_time)
        .count()
    )
    
    submissions_per_minute = recent_submissions / window_minutes if window_minutes > 0 else 0
    
    return {
        "submissions_per_minute": round(submissions_per_minute, 2),
        "recent_submissions": recent_submissions,
        "window_minutes": window_minutes
    }

@app.post("/finalize_round/{round_id}")
def finalize_round(round_id: int, db: Session = Depends(get_db), hotkey: str = Depends(auth.verify_validator_signature)):
    """
    Finalize a round: evaluate all submissions and determine winners.
    Called at round deadline. Requires validator authentication.
    """
    round_obj = db.query(models.CompetitionRound).filter(
        models.CompetitionRound.id == round_id
    ).first()
    
    if not round_obj:
        raise HTTPException(status_code=404, detail="Round not found")
    
    if round_obj.status != "active":
        raise HTTPException(status_code=400, detail="Round already finalized")
    
    # CRITICAL: Only allow finalization if round has actually expired
    now = datetime.utcnow()
    if round_obj.end_time > now:
        time_remaining = (round_obj.end_time - now).total_seconds()
        hours_remaining = time_remaining / 3600
        raise HTTPException(
            status_code=400,
            detail=f"Round has not expired yet. {hours_remaining:.2f} hours remaining. Rounds must run for full 48 hours."
        )
    
    # Mark as evaluating
    round_obj.status = "evaluating"
    db.commit()
    
    # Get all validated submissions for this round
    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )
    
    if not submissions:
        print(f"[ROUND] No validated submissions in round {round_id}")
        round_obj.status = "completed"
        db.commit()
        return {"status": "completed", "winners": []}
    
    # Calculate rankings (with first-submission-wins logic)
    rankings = calculate_rankings(submissions, round_obj.baseline_submission_id, db)
    
    # Update round with winner
    if rankings:
        winner = rankings[0]
        round_obj.winner_hotkey = winner["miner_hotkey"]
        round_obj.baseline_submission_id = winner["submission_id"]
        db.commit()
        
        # Mark winning submission as baseline for next round
        winner_submission = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.id == winner["submission_id"]
        ).first()
        if winner_submission:
            winner_submission.is_baseline = True
            db.commit()
    
    round_obj.status = "completed"
    db.commit()
    
    print(f"✅ [ROUND] Round {round_id} finalized. Winner: {round_obj.winner_hotkey}")
    
    return {
        "status": "completed",
        "round_id": round_id,
        "winner": round_obj.winner_hotkey,
        "rankings": rankings[:4]  # Top 4
    }

@app.get("/get_completed_rounds")
def get_completed_rounds(
    network: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get list of completed rounds with their winners.
    Useful for viewing round history and checking if winners were set.
    """
    limit = min(limit, 100)
    network = normalize_network(network)
    
    rounds = (
        db.query(models.CompetitionRound)
        .filter(models.CompetitionRound.network == network)
        .filter(models.CompetitionRound.status == "completed")
        .order_by(models.CompetitionRound.round_number.desc())
        .limit(limit)
        .all()
    )
    
    result = []
    for round_obj in rounds:
        # Count submissions
        submission_count = (
            db.query(models.SpeedSubmission)
            .filter(models.SpeedSubmission.round_id == round_obj.id)
            .count()
        )
        
        validated_count = (
            db.query(models.SpeedSubmission)
            .filter(models.SpeedSubmission.round_id == round_obj.id)
            .filter(models.SpeedSubmission.validated == True)
            .count()
        )
        
        result.append({
            "round_id": round_obj.id,
            "round_number": round_obj.round_number,
            "start_time": round_obj.start_time.isoformat() if isinstance(round_obj.start_time, datetime) else str(round_obj.start_time),
            "end_time": round_obj.end_time.isoformat() if isinstance(round_obj.end_time, datetime) else str(round_obj.end_time),
            "status": round_obj.status,
            "winner_hotkey": round_obj.winner_hotkey,
            "baseline_submission_id": round_obj.baseline_submission_id,
            "total_submissions": submission_count,
            "validated_submissions": validated_count,
            "has_winner": round_obj.winner_hotkey is not None
        })
    
    return {
        "network": network,
        "completed_rounds": result,
        "total": len(result)
    }

@app.post("/refinalize_round/{round_id}")
def refinalize_round(round_id: int, db: Session = Depends(get_db), hotkey: str = Depends(auth.verify_validator_signature)):
    """
    Re-finalize a completed round that doesn't have a winner_hotkey set.
    Useful if a round was completed but winner wasn't set due to timing issues.
    Requires validator authentication.
    """
    round_obj = db.query(models.CompetitionRound).filter(
        models.CompetitionRound.id == round_id
    ).first()
    
    if not round_obj:
        raise HTTPException(status_code=404, detail="Round not found")
    
    if round_obj.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Round is not completed (status: {round_obj.status}). Only completed rounds can be re-finalized."
        )
    
    if round_obj.winner_hotkey:
        return {
            "status": "already_has_winner",
            "round_id": round_id,
            "winner_hotkey": round_obj.winner_hotkey,
            "message": "Round already has a winner. No need to re-finalize."
        }
    
    # Get all validated submissions for this round
    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )
    
    if not submissions:
        return {
            "status": "no_submissions",
            "round_id": round_id,
            "message": "No validated submissions found in this round."
        }
    
    # Calculate rankings
    rankings = calculate_rankings(submissions, round_obj.baseline_submission_id, db)
    
    if not rankings:
        # Check why rankings are empty
        total_submissions = len(submissions)
        revealed_count = sum(1 for s in submissions if s.is_revealed == True)
        verified_count = sum(1 for s in submissions if s.logit_verification_passed == True)
        validated_tps_count = sum(1 for s in submissions if s.validated_tokens_per_sec is not None)
        
        return {
            "status": "no_valid_rankings",
            "round_id": round_id,
            "message": "No submissions passed all ranking filters.",
            "diagnostics": {
                "total_validated_submissions": total_submissions,
                "revealed_submissions": revealed_count,
                "passed_logit_verification": verified_count,
                "have_validated_tokens_per_sec": validated_tps_count
            }
        }
    
    # Set winner
    winner = rankings[0]
    round_obj.winner_hotkey = winner["miner_hotkey"]
    round_obj.baseline_submission_id = winner["submission_id"]
    db.commit()
    
    # Mark winning submission as baseline for next round
    winner_submission = db.query(models.SpeedSubmission).filter(
        models.SpeedSubmission.id == winner["submission_id"]
    ).first()
    if winner_submission:
        winner_submission.is_baseline = True
        db.commit()
    
    print(f"✅ [REFINALIZE] Round {round_id} winner set: {round_obj.winner_hotkey}")
    
    return {
        "status": "success",
        "round_id": round_id,
        "winner_hotkey": round_obj.winner_hotkey,
        "rankings": rankings[:4]  # Top 4
    }


# ==================== DASHBOARD ENDPOINTS ====================
# Authenticated read-only endpoints for frontend dashboard.
# Accepts: API key (Authorization: Bearer <key>) or validator signature.

@app.get("/dashboard/overview")
def dashboard_overview(
    network: Optional[str] = None,
    db: Session = Depends(get_db),
    role: str = Depends(auth.verify_dashboard_read),
):
    """Aggregated dashboard overview: current round, recent rounds, top miners, stats."""
    network = normalize_network(network)

    current_round = ensure_current_round(db, network)
    now = datetime.utcnow()
    time_remaining = max(0, int((current_round.end_time - now).total_seconds()))

    current_round_submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == current_round.id)
        .count()
    )

    recent_rounds = (
        db.query(models.CompetitionRound)
        .filter(models.CompetitionRound.network == network)
        .filter(models.CompetitionRound.status == "completed")
        .order_by(models.CompetitionRound.round_number.desc())
        .limit(5)
        .all()
    )

    recent_rounds_data = []
    for r in recent_rounds:
        sub_count = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.round_id == r.id
        ).count()
        recent_rounds_data.append({
            "round_id": r.id,
            "round_number": r.round_number,
            "start_time": r.start_time.isoformat(),
            "end_time": r.end_time.isoformat(),
            "winner_hotkey": r.winner_hotkey,
            "total_submissions": sub_count,
        })

    total_miners = db.query(models.MinerRegistration).filter(
        models.MinerRegistration.network == network,
        models.MinerRegistration.is_flagged == False,
    ).count()

    total_submissions = db.query(models.SpeedSubmission).filter(
        models.SpeedSubmission.network == network,
    ).count()

    validated_submissions = db.query(models.SpeedSubmission).filter(
        models.SpeedSubmission.network == network,
        models.SpeedSubmission.validated == True,
    ).count()

    cutoff = now - timedelta(hours=24)
    submissions_24h = db.query(models.SpeedSubmission).filter(
        models.SpeedSubmission.network == network,
        models.SpeedSubmission.created_at >= cutoff,
    ).count()

    return {
        "network": network,
        "current_round": {
            "round_id": current_round.id,
            "round_number": current_round.round_number,
            "status": current_round.status,
            "start_time": current_round.start_time.isoformat(),
            "end_time": current_round.end_time.isoformat(),
            "time_remaining_seconds": time_remaining,
            "total_submissions": current_round_submissions,
        },
        "recent_completed_rounds": recent_rounds_data,
        "stats": {
            "total_registered_miners": total_miners,
            "total_submissions": total_submissions,
            "validated_submissions": validated_submissions,
            "submissions_last_24h": submissions_24h,
        },
    }


@app.get("/dashboard/leaderboard")
def dashboard_leaderboard(
    network: Optional[str] = None,
    round_id: Optional[int] = None,
    db: Session = Depends(get_db),
    role: str = Depends(auth.verify_dashboard_read),
):
    """Leaderboard for a given round (defaults to latest completed round)."""
    network = normalize_network(network)

    if round_id:
        round_obj = db.query(models.CompetitionRound).filter(
            models.CompetitionRound.id == round_id,
            models.CompetitionRound.network == network,
        ).first()
    else:
        round_obj = (
            db.query(models.CompetitionRound)
            .filter(models.CompetitionRound.network == network)
            .filter(models.CompetitionRound.status == "completed")
            .order_by(models.CompetitionRound.round_number.desc())
            .first()
        )

    if not round_obj:
        return {"round": None, "leaderboard": []}

    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_obj.id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )

    rankings = calculate_rankings(submissions, round_obj.baseline_submission_id, db)

    leaderboard = []
    for entry in rankings[:20]:
        leaderboard.append({
            "rank": entry["rank"],
            "miner_hotkey": entry["miner_hotkey"],
            "tokens_per_sec": entry["tokens_per_sec"],
            "target_sequence_length": entry["target_sequence_length"],
            "league": entry["league"],
            "weighted_score": entry["weighted_score"],
            "logit_verified": entry.get("logit_verification_passed"),
            "cosine_similarity": entry.get("cosine_similarity"),
        })

    return {
        "round": {
            "round_id": round_obj.id,
            "round_number": round_obj.round_number,
            "status": round_obj.status,
            "winner_hotkey": round_obj.winner_hotkey,
            "start_time": round_obj.start_time.isoformat(),
            "end_time": round_obj.end_time.isoformat(),
        },
        "leaderboard": leaderboard,
    }


@app.get("/dashboard/miner/{hotkey}")
def dashboard_miner_detail(
    hotkey: str,
    network: Optional[str] = None,
    db: Session = Depends(get_db),
    role: str = Depends(auth.verify_dashboard_read),
):
    """Per-miner detail view for dashboard."""
    network = normalize_network(network)

    reg = db.query(models.MinerRegistration).filter(
        models.MinerRegistration.hotkey == hotkey,
        models.MinerRegistration.network == network,
    ).first()

    if not reg:
        raise HTTPException(status_code=404, detail="Miner not found")

    recent_submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.miner_hotkey == hotkey)
        .filter(models.SpeedSubmission.network == network)
        .order_by(models.SpeedSubmission.created_at.desc())
        .limit(20)
        .all()
    )

    submissions_data = []
    for s in recent_submissions:
        submissions_data.append({
            "id": s.id,
            "round_id": s.round_id,
            "tokens_per_sec": s.tokens_per_sec,
            "validated_tokens_per_sec": s.validated_tokens_per_sec,
            "target_sequence_length": s.target_sequence_length,
            "validated": s.validated,
            "score": s.score,
            "logit_verification_passed": s.logit_verification_passed,
            "cosine_similarity": s.cosine_similarity,
            "is_baseline": s.is_baseline,
            "created_at": s.created_at.isoformat(),
        })

    return {
        "hotkey": reg.hotkey,
        "uid": reg.uid,
        "network": network,
        "is_flagged": reg.is_flagged,
        "flag_reason": reg.flag_reason if reg.is_flagged else None,
        "registered_at": reg.registered_at,
        "last_seen": reg.last_seen,
        "recent_submissions": submissions_data,
    }
