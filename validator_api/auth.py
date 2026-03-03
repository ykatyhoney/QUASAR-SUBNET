import os
import time
import bittensor as bt
from fastapi import Request, HTTPException, Depends
from typing import Optional, Set, Dict

# ── Validator hotkey allowlist ──
_VALIDATOR_HOTKEYS_RAW = os.environ.get("VALIDATOR_HOTKEYS", "").strip()
AUTHORIZED_VALIDATOR_HOTKEYS: Set[str] = {
    hk.strip() for hk in _VALIDATOR_HOTKEYS_RAW.split(",") if hk.strip()
}

if AUTHORIZED_VALIDATOR_HOTKEYS:
    print(f"[AUTH] Loaded {len(AUTHORIZED_VALIDATOR_HOTKEYS)} authorized validator hotkey(s)")
else:
    print("[AUTH] WARNING: VALIDATOR_HOTKEYS env var is empty. "
          "Validator-only endpoints will reject all requests.")

# ── Signature replay protection ──
SIGNATURE_MAX_AGE_SECONDS = int(os.environ.get("SIGNATURE_MAX_AGE_SECONDS", "120"))

# ── API keys for team members (dashboard devs, admins, etc.) ──
# Format: comma-separated "key:role" pairs.
#   role = "read"  -> can access read-only dashboard endpoints
#   role = "admin" -> full access (same as validator)
# If no role suffix, defaults to "read".
# Example: API_KEYS="sk-abc123:read,sk-xyz789:admin"
_API_KEYS_RAW = os.environ.get("API_KEYS", "").strip()
API_KEY_ROLES: Dict[str, str] = {}
for entry in _API_KEYS_RAW.split(","):
    entry = entry.strip()
    if not entry:
        continue
    if ":" in entry:
        key, role = entry.rsplit(":", 1)
        API_KEY_ROLES[key.strip()] = role.strip().lower()
    else:
        API_KEY_ROLES[entry] = "read"

if API_KEY_ROLES:
    print(f"[AUTH] Loaded {len(API_KEY_ROLES)} API key(s) "
          f"(read={sum(1 for r in API_KEY_ROLES.values() if r == 'read')}, "
          f"admin={sum(1 for r in API_KEY_ROLES.values() if r == 'admin')})")


def verify_signature(request: Request) -> str:
    """
    Verify a Bittensor hotkey signature with replay protection.
    """
    hotkey = request.headers.get("Hotkey")
    signature = request.headers.get("Signature")

    if not hotkey or not signature:
        raise HTTPException(status_code=401, detail="Missing Hotkey or Signature headers")

    try:
        try:
            signature_bytes = bytes.fromhex(signature.removeprefix("0x"))
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid signature format. Expected hex string")

        timestamp_str = request.headers.get("Timestamp")
        if timestamp_str:
            try:
                ts = int(timestamp_str)
            except ValueError:
                raise HTTPException(status_code=401, detail="Invalid Timestamp header")
            age = abs(int(time.time()) - ts)
            if age > SIGNATURE_MAX_AGE_SECONDS:
                raise HTTPException(
                    status_code=401,
                    detail=f"Signature expired ({age}s old, max {SIGNATURE_MAX_AGE_SECONDS}s)"
                )
            message = f"{hotkey}:{timestamp_str}".encode()
        else:
            # Legacy mode: signed message is just the hotkey
            message = hotkey.encode()

        keypair = bt.Keypair(ss58_address=hotkey)
        if not keypair.verify(message, signature_bytes):
            raise HTTPException(status_code=401, detail="Invalid signature")

        return hotkey

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


def verify_validator_signature(request: Request) -> str:
    """
    Verify that the caller is an authorized validator.

    1. Verifies the Bittensor signature (with replay protection).
    2. Checks the hotkey against the VALIDATOR_HOTKEYS allowlist.
    """
    hotkey = verify_signature(request)

    if not AUTHORIZED_VALIDATOR_HOTKEYS:
        raise HTTPException(
            status_code=503,
            detail="Validator authentication is not configured. "
                   "Set the VALIDATOR_HOTKEYS environment variable."
        )

    if hotkey not in AUTHORIZED_VALIDATOR_HOTKEYS:
        raise HTTPException(
            status_code=403,
            detail="Hotkey is not an authorized validator. "
                   "Only registered validator hotkeys can call this endpoint."
        )

    return hotkey


def _extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from Authorization header (Bearer token) or X-API-Key header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-API-Key", "").strip() or None


def verify_api_key(request: Request, required_role: str = "read") -> str:
    """
    Verify an API key from the request headers.

    Accepts: Authorization: Bearer <key>  OR  X-API-Key: <key>

    Returns the role string on success. Raises 401/403 on failure.
    """
    key = _extract_api_key(request)
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key. Use 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header.")

    role = API_KEY_ROLES.get(key)
    if role is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    role_hierarchy = {"read": 0, "admin": 1}
    if role_hierarchy.get(role, 0) < role_hierarchy.get(required_role, 0):
        raise HTTPException(
            status_code=403,
            detail=f"API key has '{role}' role but '{required_role}' is required"
        )

    return role


def verify_dashboard_read(request: Request) -> str:
    """Verify the caller has at least 'read' access (API key OR validator signature)."""
    # Try API key first (cheaper check)
    key = _extract_api_key(request)
    if key:
        return verify_api_key(request, required_role="read")

    # Fall back to Bittensor validator signature
    try:
        verify_validator_signature(request)
        return "admin"
    except HTTPException:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Use 'Authorization: Bearer <api_key>' or Bittensor validator signature."
        )


def verify_dashboard_admin(request: Request) -> str:
    """Verify the caller has 'admin' access (admin API key OR validator signature)."""
    key = _extract_api_key(request)
    if key:
        return verify_api_key(request, required_role="admin")

    try:
        verify_validator_signature(request)
        return "admin"
    except HTTPException:
        raise HTTPException(
            status_code=401,
            detail="Admin authentication required. Use admin API key or Bittensor validator signature."
        )
