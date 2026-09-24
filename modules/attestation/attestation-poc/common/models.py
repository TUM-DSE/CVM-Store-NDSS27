from typing import Dict, Optional

from pydantic import BaseModel


class NonceRequest(BaseModel):
    public_key: str  # hex, uncompressed secp256k1


class NonceResponse(BaseModel):
    nonce: str  # hex, 64 random server bytes
    timings: Optional[Dict[str, float]] = None


class AttestRequest(BaseModel):
    evidence: str  # base64, the ratsd evidence token
    public_key: str  # hex


class AttestResponse(BaseModel):
    encrypted_config: str  # base64, ECIES ciphertext
    timings: Optional[Dict[str, float]] = None
