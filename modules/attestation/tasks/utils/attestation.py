"""Parsing of attestation run timings into the summary table."""

import json
import statistics
from pathlib import Path

CLIENT_TIMES = "client_times.json"
SERVER_TIMES = "server_times.json"

SUMMARY_HEADER = [
    "Component",
    "Evidence Collection (ms)",
    "Evidence Appraisal (ms)",
    "Attestation Result Appraisal (ms)",
    "Seal/Unseal (ms)",
    "Total (ms)",
]


def latest_runs(result_dir: Path, n: int) -> list[str]:
    """Timestamps of the n most recent runs that have both timing files."""
    if not result_dir.exists():
        return []
    stamps = [
        p.name.removesuffix(f"_{CLIENT_TIMES}")
        for p in result_dir.glob(f"*_{CLIENT_TIMES}")
        if (result_dir / p.name.replace(CLIENT_TIMES, SERVER_TIMES)).exists()
    ]
    return sorted(stamps, reverse=True)[:n]


def load_runs(result_dir: Path, timestamps: list[str]) -> tuple[list[dict], list[dict]]:
    client = [
        json.loads((result_dir / f"{ts}_{CLIENT_TIMES}").read_text())
        for ts in timestamps
    ]
    server = [
        json.loads((result_dir / f"{ts}_{SERVER_TIMES}").read_text())
        for ts in timestamps
    ]
    return client, server


def _mean_ms(values: list[float]) -> float:
    return statistics.mean(values) * 1000


def average_times(
    client: list[dict], server: list[dict]
) -> dict[str, dict[str, float]]:
    """Per-component means in milliseconds; the timing files are in seconds."""
    return {
        "cvm": {
            "seal_unseal": _mean_ms(
                [c["key_generation"] + c["config_decryption"] for c in client]
            ),
            "evidence_collection": _mean_ms(
                [c["ratsd_evidence_collection"] for c in client]
            ),
            "total": _mean_ms([c["total_client_time"] for c in client]),
        },
        "trusted_party": {
            "seal_unseal": _mean_ms([s["config_encryption"] for s in server]),
            "evidence_verification": _mean_ms(
                [
                    s["veraison_create_session"] + s["veraison_submit_evidence"]
                    for s in server
                ]
            ),
            "appraisal": _mean_ms([s["jwt_verification"] for s in server]),
            "total": _mean_ms(
                [
                    s["veraison_create_session"]
                    + s["veraison_submit_evidence"]
                    + s["jwt_verification"]
                    + s["config_encryption"]
                    for s in server
                ]
            ),
        },
    }


def summary_rows(averages: dict[str, dict[str, float]]) -> list[list[str]]:
    cvm = averages["cvm"]
    tp = averages["trusted_party"]
    return [
        SUMMARY_HEADER,
        [
            "CVM",
            f"{cvm['evidence_collection']:.2f}",
            "-",
            "-",
            f"{cvm['seal_unseal']:.2f}",
            f"{cvm['total']:.2f}",
        ],
        [
            "Trusted Party",
            "-",
            f"{tp['evidence_verification']:.2f}",
            f"{tp['appraisal']:.2f}",
            f"{tp['seal_unseal']:.2f}",
            f"{tp['total']:.2f}",
        ],
    ]
