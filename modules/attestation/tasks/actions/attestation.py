"""run-attestation: provision Veraison from the guest's report, then attest."""

import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from core.tasks.actions import ActionContext
from core.tasks.actions.registry import register_action
from core.tasks.qemu_builder import CpuFeature
from core.tasks.resources import VMResource

from modules.attestation.tasks import settings, veraison


def _wait_for_server(port: int, timeout: float = 30) -> None:
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"attestation-server not ready on port {port} after {timeout}s")


@contextmanager
def _attestation_server(
    output_dir: Path, log: Path, config_size: str, port: int
) -> Iterator[None]:
    cmd = [
        "attestation-server",
        "--port",
        str(port),
        "--config-size",
        config_size,
        "--output",
        str(output_dir),
    ]
    with open(log, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        try:
            _wait_for_server(port)
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _qemu_cpu_model(resource: VMResource) -> str:
    """Return the `-cpu` model core boots the guest with, as go-gen-ref wants it."""
    args = CpuFeature(resource).qemu_args()
    return args[args.index("-cpu") + 1].split(",")[0]


@register_action("attestation")
def run_attestation(
    ctx: ActionContext,
    config_size: str = "1M",
    processor_model: Optional[str] = None,
) -> None:
    """Provision Veraison from this guest's SNP report, then run one attestation.

    Args:
        ctx: Action context (VM/host runner, output dirs, timestamp)
        config_size: attestation-server --config-size (default "1M")
        processor_model: CoRIM trust-anchor processor model, defaulting to
            ``settings.processor_model()``

    Every output file in the action's output dir is prefixed with the run's
    timestamp: ``_report.bin``, ``_request.bin``, ``_ref_value.cbor``,
    ``_server.log``, and whatever the server and client wrote.
    """
    if ctx.is_host or "snp" not in ctx.name:
        raise ValueError(
            "Attestation needs an SEV-SNP guest (vm.start --type snp --attestation)"
        )
    ts = ctx.timestamp
    host_out, guest_out = ctx.outputdir_host, ctx.outputdir_guest

    veraison.start_services()
    # Reference values from an earlier guest image would otherwise still appraise.
    veraison.reset_stores()
    veraison.provision_trust_anchor_corims(
        [processor_model or settings.processor_model()]
    )

    report = f"{ts}_report.bin"
    ctx.vm.ssh_cmd(
        [
            "snpguest",
            "report",
            "-r",
            str(guest_out / report),
            str(guest_out / f"{ts}_request.bin"),
        ]
    )
    resource = ctx.config["resource"]
    veraison.provision_reference_corim(
        host_out / report,
        resource.cpu,
        _qemu_cpu_model(resource),
        host_out / f"{ts}_ref_value.cbor",
    )

    # Both sides write fixed file names: stage per run, flatten with the timestamp.
    staging = host_out / ts
    staging.mkdir(exist_ok=True)
    port = settings.server_port()
    server_url = f"http://{settings.host_gateway()}:{port}"
    with _attestation_server(staging, host_out / f"{ts}_server.log", config_size, port):
        ctx.vm.ssh_cmd(
            [
                "attestation-client",
                "--server-url",
                server_url,
                "--output",
                str(guest_out / ts),
            ]
        )
    for staged in staging.iterdir():
        staged.rename(host_out / f"{ts}_{staged.name}")
    staging.rmdir()
