"""Veraison verification service: docker deployment and CoRIM provisioning."""

import json
import os
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from invoke import task

from core.tasks.config import BUILD_DIR
from core.tasks.procs import run
from core.tasks.qemu_builder import get_vm_config

from modules.attestation.tasks import settings

VERAISON_REPO_URL = "https://github.com/THS-on/services.git"
VERAISON_REF = "sev-snp"
# `git ls-remote` of VERAISON_REF, 2026-09-09.
VERAISON_REV = "3e25827514b3b5663e756772d429e19db2163ba4"
VERAISON_REPO = BUILD_DIR / "veraison-services"
VERAISON_ENV = VERAISON_REPO / "deployments" / "docker" / "env.bash"
VERAISON_BIN = VERAISON_REPO / "deployments" / "docker" / "veraison"

CORIM_DIR = BUILD_DIR / "attestation" / "corims"

CORIM_PROFILE = "https://amd.com/ark"
CORIM_META = {
    "corim-id": "5c57e8f4-46cd-421b-91c9-08cf93e13cfc",
    "profile": CORIM_PROFILE,
}
PROVISIONING_MEDIA_TYPE = f'application/corim-unsigned+cbor; profile="{CORIM_PROFILE}"'


def require_deployment() -> None:
    if not VERAISON_BIN.exists():
        raise RuntimeError(
            f"Veraison deployment not found at {VERAISON_REPO}; "
            "run `oma-arvio veraison.setup` first"
        )


def _veraison(*args: str) -> None:
    require_deployment()
    # env.bash exports the deployment settings the wrapper needs.
    run(
        ["bash", "-c", 'source "$0" && exec "$@"', str(VERAISON_ENV), str(VERAISON_BIN)]
        + list(args),
        cwd=VERAISON_REPO,
        stdout=None,
    )


def start_services(service: Optional[str] = None) -> None:
    _veraison("start", *([service] if service else []))


def stop_services(service: Optional[str] = None) -> None:
    _veraison("stop", *([service] if service else []))


def reset_stores() -> None:
    _veraison("clear-stores")


def submit_corim(corim: Path) -> None:
    _veraison(
        "cocli",
        "corim",
        "submit",
        "-i",
        "--corim-file",
        str(corim),
        "--media-type",
        PROVISIONING_MEDIA_TYPE,
    )


def _corim_dir() -> Path:
    CORIM_DIR.mkdir(parents=True, exist_ok=True)
    # cocli runs as the invoking user; under oma-arvio-root the dir is handed over.
    if os.geteuid() == 0 and "SUDO_UID" in os.environ:
        uid = int(os.environ["SUDO_UID"])
        os.chown(CORIM_DIR, uid, int(os.environ.get("SUDO_GID", -1)))
    return CORIM_DIR


def _comid_template(model: str, cert_pem: str) -> dict:
    return {
        "lang": "en-GB",
        "tag-identity": {"id": str(uuid.uuid4()), "version": 0},
        "triples": {
            "attester-verification-keys": [
                {
                    "environment": {
                        "class": {"vendor": "AMD", "model": model.capitalize()}
                    },
                    "verification-keys": [
                        {"type": "pkix-base64-cert", "value": cert_pem}
                    ],
                }
            ]
        },
    }


def _trust_anchor_corim(model: str) -> Path:
    return CORIM_DIR / f"ta-{model}-corim.cbor"


def build_trust_anchor_corims(models: list[str]) -> list[Path]:
    out = _corim_dir()
    meta = out / "corim-meta.json"
    meta.write_text(json.dumps(CORIM_META, indent=2))
    corims = []
    for model in models:
        with tempfile.TemporaryDirectory(prefix="snpguest-certs-") as certs:
            run(["snpguest", "fetch", "ca", "pem", certs, model], stdout=None)
            ark_pem = (Path(certs) / "ark.pem").read_text()

        template = out / f"ta-{model}.json"
        template.write_text(json.dumps(_comid_template(model, ark_pem), indent=2))
        comid = out / f"ta-{model}.cbor"
        corim = _trust_anchor_corim(model)
        for stale in (comid, corim):
            stale.unlink(missing_ok=True)

        _veraison(
            "cocli",
            "comid",
            "create",
            "--template",
            str(template),
            "--output-dir",
            str(out),
        )
        _veraison(
            "cocli",
            "corim",
            "create",
            "-m",
            str(comid),
            "-t",
            str(meta),
            "-o",
            str(corim),
        )
        if not corim.exists():
            raise FileNotFoundError(f"cocli did not produce {corim}")
        corims.append(corim)
    return corims


def trust_anchor_corims(models: list[str]) -> list[Path]:
    missing = [m for m in models if not _trust_anchor_corim(m).exists()]
    if missing:
        build_trust_anchor_corims(missing)
    return [_trust_anchor_corim(m) for m in models]


def provision_trust_anchor_corims(models: list[str]) -> None:
    for corim in trust_anchor_corims(models):
        submit_corim(corim)


def provision_reference_corim(
    report: Path, maxvcpus: int, cpu_model: str, output: Path
) -> None:
    ovmf = get_vm_config("snp", attestation=True).ovmf
    if not ovmf.exists():
        raise FileNotFoundError(f"OVMF not found at {ovmf}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as vmconfig:
        vmconfig.write(f'maxvcpus: {maxvcpus}\nmodel: "{cpu_model}"\n')
        vmconfig.flush()
        run(
            [
                "go-gen-ref",
                "sevsnp",
                "--report",
                str(report),
                "--vmconfig",
                vmconfig.name,
                "--ovmf",
                str(ovmf),
                "--corim",
                str(output),
            ],
            stdout=None,
        )
    submit_corim(output)


@task
def clone(c: Any, force: bool = False) -> None:
    """Clone the Veraison services fork (VERAISON_REV) into the build dir."""
    if VERAISON_REPO.exists():
        if not force:
            print(f"{VERAISON_REPO} exists, use --force to re-clone")
            return
        shutil.rmtree(VERAISON_REPO)
    VERAISON_REPO.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--branch",
            VERAISON_REF,
            VERAISON_REPO_URL,
            str(VERAISON_REPO),
        ],
        stdout=None,
    )
    run(
        ["git", "-C", str(VERAISON_REPO), "checkout", "--detach", VERAISON_REV],
        stdout=None,
    )


@task
def build(c: Any) -> None:
    """Build the docker deployment and start it (make docker-deploy)."""
    require_deployment()
    run(["make", "docker-deploy"], cwd=VERAISON_REPO, stdout=None)


@task
def setup(c: Any, force: bool = False) -> None:
    """clone + build."""
    clone(c, force=force)
    build(c)


@task
def start(c: Any, service: Optional[str] = None) -> None:
    """Start all services, or one of vts/provisioning/verification/management/keycloak."""
    start_services(service)


@task
def stop(c: Any, service: Optional[str] = None) -> None:
    """Stop all services, or one of them."""
    stop_services(service)


@task
def status(c: Any) -> None:
    _veraison("status")


@task
def logs(c: Any, service: str) -> None:
    _veraison("logs", service)


@task
def stores(c: Any) -> None:
    """Show provisioned endorsements, trust anchors and policies."""
    _veraison("stores")


@task
def clear_stores(c: Any) -> None:
    reset_stores()


@task
def cocli(c: Any, args: str = "") -> None:
    """Run cocli with the given argument string, e.g. --args "corim submit --help"."""
    _veraison("cocli", *(shlex.split(args) or ["--help"]))


@task
def generate_trust_anchor_corims(c: Any, all_models: bool = False) -> None:
    """Fetch the AMD ARK and build trust-anchor CoRIMs into the build dir."""
    models = settings.PROCESSOR_MODELS if all_models else [settings.processor_model()]
    for corim in build_trust_anchor_corims(models):
        print(corim)


@task
def provision_trust_anchors(c: Any, all_models: bool = False) -> None:
    """Submit the trust-anchor CoRIMs (built if missing) to the running deployment."""
    models = settings.PROCESSOR_MODELS if all_models else [settings.processor_model()]
    provision_trust_anchor_corims(models)


@task
def provision_reference_value(
    c: Any,
    report_path: str,
    maxvcpus: int,
    cpu_model: str,
    output: str = "/tmp/ref_value.cbor",
) -> None:
    """Generate the reference-value CoRIM for a guest report with go-gen-ref and submit it."""
    provision_reference_corim(Path(report_path), maxvcpus, cpu_model, Path(output))
