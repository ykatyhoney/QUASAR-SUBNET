import os
import time
import bittensor as bt
from fastapi import Request, HTTPException, Depends
from typing import Optional, Set

_VALIDATOR_HOTKEYS_RAW = os.environ.get("VALIDATOR_HOTKEYS", "").strip()
AUTHORIZED_VALIDATOR_HOTKEYS: Set[str] = {
    hk.strip() for hk in _VALIDATOR_HOTKEYS_RAW.split(",") if hk.strip()
}

# Maximum age (seconds) of a signed timestamp
SIGNATURE_MAX_AGE_SECONDS = int(os.environ.get("SIGNATURE_MAX_AGE_SECONDS", "120"))


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
