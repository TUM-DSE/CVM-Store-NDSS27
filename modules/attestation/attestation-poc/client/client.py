import base64
import hashlib
import json
import logging
import secrets
from pathlib import Path
from typing import Tuple

import coincurve
import requests
from ecies import decrypt

from common.timing import TimingLogger

DEFAULT_RATSD_HOST = "localhost"
DEFAULT_RATSD_PORT = 8895

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_keypair() -> Tuple[str, str]:
    private_key_hex = secrets.token_hex(32)
    private_key = coincurve.PrivateKey(bytes.fromhex(private_key_hex))
    public_key_hex = private_key.public_key.format(compressed=False).hex()
    return private_key_hex, public_key_hex


def request_nonce(server_url: str, public_key_hex: str) -> bytes:
    response = requests.post(
        f"{server_url}/nonce", json={"public_key": public_key_hex}, timeout=10
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Nonce request failed: {response.status_code} {response.text}"
        )
    return bytes.fromhex(response.json()["nonce"])


def get_evidence_from_ratsd(ratsd_url: str, nonce: bytes, public_key_hex: str) -> bytes:
    # Same binding the server computes: SHA512(server_nonce || SHA512(pubkey)).
    key_hash = hashlib.sha512(bytes.fromhex(public_key_hex)).digest()
    bound_nonce = hashlib.sha512(nonce + key_hash).digest()
    nonce_b64 = base64.urlsafe_b64encode(bound_nonce).decode().rstrip("=")

    evidence_request = {
        "nonce": nonce_b64,
        "attester-selection": {
            "tsm-report": {
                "content-type": "application/vnd.veraison.configfs-tsm+json",
                "privilege_level": "1",
            }
        },
    }
    response = requests.post(
        f"{ratsd_url}/ratsd/chares",
        json=evidence_request,
        headers={"Content-Type": "application/vnd.veraison.chares+json"},
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"ratsd request failed: {response.status_code} {response.text}"
        )
    return response.content


def send_attestation(server_url: str, evidence: bytes, public_key_hex: str) -> bytes:
    response = requests.post(
        f"{server_url}/attest",
        json={
            "evidence": base64.b64encode(evidence).decode(),
            "public_key": public_key_hex,
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Attestation failed: {response.status_code} {response.text}"
        )
    return base64.b64decode(response.json()["encrypted_config"])


def run_attestation_flow(server_url: str, output_dir: Path, ratsd_url: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = TimingLogger()

    logger.info("Generating ECC key pair...")
    with timings.measure("key_generation"):
        private_key_hex, public_key_hex = generate_keypair()
    logger.info(f"Public key: {public_key_hex[:32]}...")
    (output_dir / "public_key.txt").write_text(public_key_hex)

    logger.info(f"Requesting nonce from {server_url}...")
    with timings.measure("nonce_request"):
        nonce = request_nonce(server_url, public_key_hex)
    logger.info(f"Received nonce: {nonce.hex()}")

    logger.info(f"Requesting evidence from ratsd at {ratsd_url}...")
    with timings.measure("ratsd_evidence_collection"):
        evidence = get_evidence_from_ratsd(ratsd_url, nonce, public_key_hex)
    logger.info(f"Received evidence: {len(evidence)} bytes")
    (output_dir / "evidence_token.bin").write_bytes(evidence)

    logger.info("Sending attestation to server...")
    with timings.measure("server_attest_request"):
        encrypted_config = send_attestation(server_url, evidence, public_key_hex)
    logger.info(f"Received encrypted config: {len(encrypted_config)} bytes")
    (output_dir / "encrypted_config.bin").write_bytes(encrypted_config)

    logger.info("Decrypting config...")
    with timings.measure("config_decryption"):
        config = decrypt(private_key_hex, encrypted_config)
    logger.info(f"Decrypted config: {len(config)} bytes")
    (output_dir / "decrypted_config.bin").write_bytes(config)

    timings.add("total_client_time", sum(timings.to_dict().values()))
    logger.info("Attestation flow completed successfully")

    (output_dir / "client_times.json").write_text(
        json.dumps(timings.to_dict(), indent=2)
    )
    return timings.to_dict()
