import base64
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Dict, Optional

import jwt
import requests
import urllib3
from ecies import encrypt
from fastapi import FastAPI, HTTPException

from common.models import AttestRequest, AttestResponse, NonceRequest, NonceResponse
from common.timing import TimingLogger

DEFAULT_VERIFIER_HOST = "localhost"
DEFAULT_VERIFIER_PORT = 8443

# The harness's own docker deployment, self-signed.
VERIFY_VERAISON_TLS = False

NONCE_TTL_SECONDS = 600

EVIDENCE_CONTENT_TYPE = (
    'application/eat+cwt; eat_profile="tag:github.com,2025:veraison/ratsd/cmw"'
)

if not VERIFY_VERAISON_TLS:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Attestation Server")

# bound nonce (base64url) -> {pubkey, server_nonce, expires}
nonce_store: Dict[str, Dict] = {}
config_data: bytes = b""
output_dir: Optional[Path] = None
verifier_url: str = f"https://{DEFAULT_VERIFIER_HOST}:{DEFAULT_VERIFIER_PORT}"


def configure(data: bytes, out_dir: Optional[Path], verifier: str):
    global config_data, output_dir, verifier_url
    config_data = data
    output_dir = out_dir
    verifier_url = verifier
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)


def parse_size(size_str: str) -> int:
    size_str = size_str.upper().strip()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    try:
        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-1]) * multiplier)
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}")


def generate_config(size: int) -> bytes:
    return os.urandom(size)


@app.post("/nonce", response_model=NonceResponse)
async def create_nonce(request: NonceRequest):
    timings = TimingLogger()

    try:
        with timings.measure("nonce_generation"):
            server_nonce = secrets.token_bytes(64)

            try:
                pubkey_bytes = bytes.fromhex(request.public_key)
            except ValueError:
                raise HTTPException(400, "Invalid public key format (expected hex)")

            # Binds the evidence to the client's key; comes back as eat_nonce.
            pubkey_hash = hashlib.sha512(pubkey_bytes).digest()
            binding = hashlib.sha512(server_nonce + pubkey_hash).digest()
            expected_nonce = base64.urlsafe_b64encode(binding).decode()

            nonce_store[expected_nonce] = {
                "pubkey": request.public_key,
                "server_nonce": server_nonce.hex(),
                "expires": time.time() + NONCE_TTL_SECONDS,
            }

            logger.info(f"Generated nonce for pubkey {request.public_key[:16]}...")
            logger.info(f"Nonce (base64url): {expected_nonce}")

            if output_dir:
                (output_dir / "nonce.txt").write_text(
                    f"nonce (base64url): {expected_nonce}\n"
                )

        return NonceResponse(nonce=server_nonce.hex(), timings=timings.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating nonce")
        raise HTTPException(500, f"Internal server error: {e}")


@app.post("/attest", response_model=AttestResponse)
async def attest(request: AttestRequest):
    timings = TimingLogger()

    try:
        with timings.measure("total_attestation"):
            nonce_key = None
            for stored_nonce, info in nonce_store.items():
                if info["pubkey"] == request.public_key:
                    if time.time() > info["expires"]:
                        del nonce_store[stored_nonce]
                        raise HTTPException(400, "Nonce expired")
                    nonce_key = stored_nonce
                    break
            if nonce_key is None:
                raise HTTPException(400, "No nonce found for this public key")

            with timings.measure("veraison_create_session"):
                session_response = requests.post(
                    f"{verifier_url}/challenge-response/v1/newSession",
                    params={"nonce": nonce_key},
                    verify=VERIFY_VERAISON_TLS,
                )
                if session_response.status_code != 201:
                    raise HTTPException(
                        503,
                        f"Veraison session creation failed: {session_response.text}",
                    )
                session_id = session_response.headers["Location"].split("/")[-1]
                logger.info(f"Created Veraison session: {session_id}")

            with timings.measure("veraison_submit_evidence"):
                verification_response = requests.post(
                    f"{verifier_url}/challenge-response/v1/session/{session_id}",
                    data=base64.b64decode(request.evidence),
                    headers={
                        "accept": "application/vnd.veraison.challenge-response-session+json",
                        "Content-Type": EVIDENCE_CONTENT_TYPE,
                    },
                    verify=VERIFY_VERAISON_TLS,
                )
                if verification_response.status_code != 200:
                    raise HTTPException(
                        401, f"Evidence submission failed: {verification_response.text}"
                    )

            with timings.measure("jwt_verification"):
                jwt_token = verification_response.json().get("result")
                if not jwt_token:
                    raise HTTPException(401, "No JWT result in verification response")

                # Own verifier over localhost; its signing key is deliberately not checked.
                decoded = jwt.decode(jwt_token, options={"verify_signature": False})

                sevsnp_result = decoded.get("submods", {}).get("SEVSNP", {})
                status = sevsnp_result.get("ear.status")
                if status != "affirming":
                    logger.warning(f"Attestation failed with status: {status}")
                    policy_claims = sevsnp_result.get("ear.veraison.policy-claims", {})
                    raise HTTPException(
                        403,
                        f"Attestation verification failed: {status}, {policy_claims}",
                    )

                if decoded.get("eat_nonce") != nonce_key:
                    raise HTTPException(
                        403, "Nonce verification failed: eat_nonce mismatch"
                    )
                logger.info("Attestation verification succeeded")

                if output_dir:
                    (output_dir / "jwt_result.json").write_text(
                        json.dumps(decoded, indent=2)
                    )

            with timings.measure("config_encryption"):
                try:
                    encrypted_config = encrypt(request.public_key, config_data)
                except Exception as e:
                    logger.exception("ECIES encryption failed")
                    raise HTTPException(500, f"Encryption failed: {e}")

            del nonce_store[nonce_key]
            logger.info(f"Successfully provisioned config ({len(config_data)} bytes)")

        if output_dir:
            (output_dir / "server_times.json").write_text(
                json.dumps(timings.to_dict(), indent=2)
            )

        return AttestResponse(
            encrypted_config=base64.b64encode(encrypted_config).decode(),
            timings=timings.to_dict(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in attest endpoint")
        raise HTTPException(500, f"Internal server error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "nonces_active": len(nonce_store)}
