import os
import bittensor as bt
from fastapi import Request, HTTPException, Depends
from typing import Optional, Set

_VALIDATOR_HOTKEYS_RAW = os.environ.get("VALIDATOR_HOTKEYS", "").strip()
AUTHORIZED_VALIDATOR_HOTKEYS: Set[str] = {
    hk.strip() for hk in _VALIDATOR_HOTKEYS_RAW.split(",") if hk.strip()
}

def verify_signature(request: Request) -> str:
    """
    Verify a Bittensor hotkey signature.
    """
    hotkey = request.headers.get("Hotkey")
    signature = request.headers.get("Signature")

    if not hotkey or not signature:
        raise HTTPException(status_code=401, detail="Missing Hotkey or Signature headers")

    try:
        try:
            signature_bytes = bytes.fromhex(signature)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid signature format. Expected hex string")

        keypair = bt.Keypair(ss58_address=hotkey)
        if not keypair.verify(hotkey.encode(), signature_bytes):
            raise HTTPException(status_code=401, detail="Invalid signature")

        return hotkey

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


def verify_validator_signature(request: Request) -> str:
    """
    Verify that the caller is an authorized validator.
    """
    hotkey = verify_signature(request)

    if not AUTHORIZED_VALIDATOR_HOTKEYS:
        raise HTTPException(
            status_code=503,
            detail="Validator authentication is not configured."
        )

    if hotkey not in AUTHORIZED_VALIDATOR_HOTKEYS:
        raise HTTPException(
            status_code=403,
            detail="Hotkey is not an authorized validator. "
                   "Only registered validator hotkeys can call this endpoint."
        )

    return hotkey
