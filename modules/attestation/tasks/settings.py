"""Attestation constants, overridable through config.toml [defaults]."""

import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Optional

from core.tasks.config import PROJECT_ROOT

ATTESTATION_SERVER_PORT = 8000
# QEMU user networking: the guest reaches the host at this address.
QEMU_HOST_GATEWAY = "10.0.2.2"

PROCESSOR_MODELS = ["milan", "genoa", "bergamo", "siena", "turin"]
DEFAULT_PROCESSOR_MODEL = "genoa"


@lru_cache(maxsize=1)
def _defaults() -> dict:
    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        return tomllib.load(f).get("defaults", {})


def server_port() -> int:
    return int(_defaults().get("attestation_server_port", ATTESTATION_SERVER_PORT))


def host_gateway() -> str:
    return str(_defaults().get("attestation_host_gateway", QEMU_HOST_GATEWAY))


def _host_processor_model() -> Optional[str]:
    # EPYC model numbers: 7xx3 Milan, 9xx4 Genoa, 97x4 Bergamo, 8xx4 Siena, 9xx5 Turin.
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None
    match = re.search(r"EPYC (\d)(\d)\d(\d)", cpuinfo)
    if not match:
        return None
    series, sub, gen = match.groups()
    if (series, gen) == ("7", "3"):
        return "milan"
    if (series, gen) == ("8", "4"):
        return "siena"
    if (series, gen) == ("9", "4"):
        return "bergamo" if sub == "7" else "genoa"
    if (series, gen) == ("9", "5"):
        return "turin"
    return None


def processor_model() -> str:
    model = (
        _defaults().get("attestation_processor_model")
        or _host_processor_model()
        or DEFAULT_PROCESSOR_MODEL
    )
    if model not in PROCESSOR_MODELS:
        raise ValueError(
            f"Unknown processor model {model!r}, expected one of {PROCESSOR_MODELS}"
        )
    return model
