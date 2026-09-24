#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import uuid
from pathlib import Path
from typing import Optional

from core.tasks.actions import ActionContext
from core.tasks.actions.registry import register_action

from modules.accelstore.tasks.ablation import Ablation
from modules.accelstore.tasks.crypto_backend import CryptoBackend
from modules.accelstore.tasks.io_path import IoPath
from modules.accelstore.tasks import job_shape
from modules.accelstore.tasks.pipeline_opts import PipelineOpts
from modules.accelstore.tasks.sync_modes import SyncMode
from modules.accelstore.tasks.actions.storage_env import (
    precondition_device,
    spdk_device,
)

JOBS_DIR = Path("/shared/modules/accelstore")  # packaged by the module flake

VALID_CLIENTS = ["kvs_client", "log_client", "fs_client"]

# Wrapper exit code: the client finished but no usable trace dump was produced.
TRACE_PIPELINE_FAILED = 42

# Above the client's 8M default, which the full-trace suite has overrun.
DEFAULT_TRACE_ENTRIES = 16777216


def _get_job_path_component(job):
    """Return the result-path component for a job name or job sequence."""
    if isinstance(job, list):
        if len(job) == 1:
            return Path(job[0]).stem
        else:
            job_names = [Path(j).stem for j in job]
            return "#".join(job_names)
    return Path(job).stem if isinstance(job, (str, Path)) else job


def _accelstore_path(name: str, cfg: dict) -> tuple:
    backend = CryptoBackend.parse(cfg.get("crypto") or CryptoBackend.AEGIS)
    components = [
        "accelstore",
        name,
        backend.client_binary(cfg.get("client", "")),
        _get_job_path_component(cfg.get("job", "")),
    ]
    # Above the sync and ablation slugs, so an older parser skips a kernel arm.
    io_slug = IoPath.from_config(cfg).path_component()
    if io_slug:
        components.append(io_slug)
    sync = SyncMode.from_config(cfg).path_component()
    if sync:
        components.append(sync)
    # Separate component, not a suffix: "sync-abl-*" parses as a sync slug.
    ablation_slug = Ablation.from_config(cfg).path_component()
    if ablation_slug:
        components.append(ablation_slug)
    pipeline_slug = PipelineOpts.from_config(cfg).path_component()
    if pipeline_slug:
        components.append(pipeline_slug)
    return tuple(components)


def _check_jobs_against_chunk(jobs: list, pipe: PipelineOpts) -> None:
    """Refuse an explicit chunk size that would amplify a job's reads."""
    if pipe.chunk_size is None:
        return
    for name in jobs:
        path = JOBS_DIR / name
        try:
            text = path.read_text()
        except OSError as exc:
            print(f"NOTE: chunk check skipped for {name}: {exc}")
            continue
        shape = job_shape.parse_job(text)
        problem = job_shape.check_chunk_for_job(shape, pipe.chunk_size)
        if problem:
            raise ValueError(
                f"--chunk-size {pipe.chunk_size} amplifies {name}: {problem}."
                " Pass a chunk that satisfies the rule, or drop the argument to"
                " run at the library default"
            )
        print(f"chunk geometry: {name}: {job_shape.describe(shape, pipe.chunk_size)}")


@register_action("accelstore", path_fn=_accelstore_path)
def run_accelstore(
    ctx: ActionContext,
    client: str,
    job: str | list[str],
    config: str,
    pci_device: str,
    crypto: str = CryptoBackend.AEGIS.value,
    dev_path: str = None,
    device: str = "Nvme1n1",
    tracing: Optional[str] = None,
    trace_entries: Optional[int] = DEFAULT_TRACE_ENTRIES,
    silent: bool = True,
    skip_precondition: bool = False,
    precondition_scheme: str = None,
    ablation: Optional[str] = None,
    io_path: Optional[str] = None,
    trigger: Optional[str] = None,
    every_n: Optional[int] = None,
    interval_us: Optional[int] = None,
    coupling: Optional[str] = None,
    commit_every: Optional[int] = None,
    blocking: bool = False,
    metadata_only: bool = False,
    tc_backend: Optional[str] = None,
    tc_rtt_us: Optional[int] = None,
    tc_rtt_model: Optional[str] = None,
    tc_rtt_interval: Optional[int] = None,
    chunk_size: Optional[int] = None,
    pipeline_bytes: Optional[int] = None,
    io_depth: Optional[int] = None,
    reorder_window: Optional[int] = None,
    burst_bytes: Optional[str] = None,
    burst_adaptive: bool = False,
    io_span_bytes: Optional[int] = None,
    cluster_size: Optional[int] = None,
    md_pages: Optional[int] = None,
):
    """Run an AccelStore client benchmark, optionally under SPDK tracing.

    Args:
        ctx: Action context (VM, output dirs, timestamp)
        client: Client name, unprefixed: kvs_client, log_client or fs_client
        job: Workload file name, or a list run as a sequence sharing one
             trusted counter file
        config: SPDK bdev configuration as a JSON string
        pci_device: PCI address of the device the bdev config attaches
        crypto: Build variant to run: aegis (default), aesgcm or none
        dev_path: Kernel device node used for pre-conditioning
        device: Bdev name created by the config
        tracing: "tracing", "small-trace", or None to disable
        trace_entries: Trace ring entries per lcore, a power of two in
                       [32768, 67108864]. None uses the client's own default.
                       Ignored when tracing is None.
        silent: Pass --silent to the client
        skip_precondition: Skip the secure erase and pre-conditioning
        precondition_scheme: Pre-conditioning scheme name
        ablation: Comma-separated subset of no-overlap|bounce-copy|no-encrypt|
                  no-verify|zero-copy, "none" for the unablated sweep arm, or
                  None for a run outside a sweep
        io_path: "spdk" or "kernel" arm of a bypass sweep, None outside one.
                 Whether vfio-pci is bound is read off ``config`` itself.
        trigger: Epoch trigger: on-shutdown (default), on-barrier, every-n,
                 every-t, per-op
        every_n: Mutating operations between epochs (operand of every-n)
        interval_us: Durability window in microseconds (operand of every-t)
        coupling: Counter commit against the metadata write: ordered
                  (default) or parallel
        commit_every: Counter commit every K epochs (K > 1 is refused)
        blocking: Hold mutators from arming an epoch until it is durable.
                  Only every-n and every-t have an epoch to block on.
        metadata_only: Persist metadata without committing a counter value
        tc_backend: Counter service: local (default) or mock-remote
        tc_rtt_us: Round trip a paying commit waits out under the mock
        tc_rtt_model: Which commits pay it: every (default), periodic, elapsed
        tc_rtt_interval: Operand of the periodic/elapsed models
        chunk_size: --crypto-chunk-size in bytes
        pipeline_bytes: --pipeline-bytes, the in-flight crypto byte budget
        io_depth: --io-depth, chunk I/Os in flight per operation
        reorder_window: --reorder-window, chunks in progress (KVS/LOG reads)
        burst_bytes: --burst-bytes, crypto per submit/drain pass, or "max"
        burst_adaptive: --burst-adaptive
        io_span_bytes: --io-span-bytes, the coalesced device command span
        cluster_size: --cluster-size of a freshly created blobstore
        md_pages: --md-pages reserved at init

        The geometry arguments form one PipelineOpts and the sync/tc arguments
        one SyncMode, each of which validates them and names the result
        directory.

    Output structure:
        bench-result/accelstore/{name}/{client}/{job}/{timestamp}.log
        bench-result/accelstore/{name}/{client}/{job}/{timestamp}_trace.dump (if tracing)
        bench-result/accelstore/{name}/{client}/{seal}#{unseal}/{timestamp}_seq{N}.log
        bench-result/accelstore/{name}/{client}/{seal}#{unseal}/{timestamp}_tc.bin
    """
    backend = CryptoBackend.parse(crypto)

    sync = SyncMode(
        trigger=trigger,
        every_n=every_n,
        interval_us=interval_us,
        coupling=coupling,
        commit_every=commit_every,
        blocking=blocking,
        metadata_only=metadata_only,
        tc_backend=tc_backend,
        tc_rtt_us=tc_rtt_us,
        tc_rtt_model=tc_rtt_model,
        tc_rtt_interval=tc_rtt_interval,
    )
    sync.validate()
    for warning in sync.warnings():
        print(f"WARNING: {warning}")

    abl = Ablation.parse(ablation)
    abl.validate(backend)

    io = IoPath.parse(io_path)
    pipe = PipelineOpts.from_task_kwargs(
        chunk_size=chunk_size,
        pipeline_bytes=pipeline_bytes,
        io_depth=io_depth,
        reorder_window=reorder_window,
        burst_bytes=burst_bytes,
        burst_adaptive=burst_adaptive,
        io_span_bytes=io_span_bytes,
        cluster_size=cluster_size,
        md_pages=md_pages,
    )
    pipe.validate()
    for warning in pipe.warnings():
        print(f"WARNING: {warning}")

    if tracing is not None and tracing not in ["tracing", "small-trace"]:
        raise ValueError(
            f"Invalid tracing mode '{tracing}'. Must be 'tracing', 'small-trace', or None"
        )

    # Validated even when tracing is off, so a bad value is not silently dropped.
    if trace_entries is not None:
        trace_entries = int(trace_entries)
        if not 32768 <= trace_entries <= 67108864:
            raise ValueError(
                f"trace_entries {trace_entries} out of range [32768, 67108864]"
            )
        if trace_entries & (trace_entries - 1):
            raise ValueError(f"trace_entries {trace_entries} is not a power of two")

    jobs = job if isinstance(job, list) else [job]
    is_sequence = len(jobs) > 1

    if len(jobs) == 0:
        raise ValueError("Job sequence cannot be empty")

    if client not in VALID_CLIENTS:
        raise ValueError(
            f"Invalid client '{client}'. Must be one of: {', '.join(VALID_CLIENTS)}"
        )

    client_binary = backend.client_binary(client)

    _check_jobs_against_chunk(jobs, pipe)

    try:
        parsed_config = json.loads(config)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {e}")

    # Read off the config, not the arm name, so the two cannot disagree.
    bdev_methods = {
        entry.get("method")
        for subsystem in parsed_config.get("subsystems", [])
        if subsystem.get("subsystem") == "bdev"
        for entry in subsystem.get("config", [])
    }
    binds_vfio = "bdev_nvme_attach_controller" in bdev_methods
    if io.kernel and binds_vfio:
        raise ValueError(
            "io_path 'kernel' with a bdev_nvme config: the kernel arm runs on"
            " bdev_aio over the namespace node, and SPDK's own driver cannot"
            " claim the device the kernel driver is serving"
        )

    vm = ctx.vm
    outputdir_guest = ctx.outputdir_guest
    date = ctx.timestamp

    tc_file = outputdir_guest / f"{date}_tc.bin" if is_sequence else None

    # Before spdk-setup binds vfio-pci: nvme-cli needs the kernel driver.
    precondition_device(
        ctx, vm, dev_path, skip=skip_precondition, scheme=precondition_scheme
    )

    with spdk_device(
        vm,
        config,
        pci_device,
        device,
        destroy_silent=silent,
        bind=binds_vfio,
        bind_note=(
            f"Leaving {pci_device} on the kernel driver"
            f" ({io.label()}); hugepages only"
        ),
    ) as config_file:
        for idx, current_job in enumerate(jobs):
            # The store is destroyed only on entry, so later jobs see its state.
            if idx != 0:
                print(f"Skipping blobstore destroyer for job {idx+1}/{len(jobs)}")

            if is_sequence:
                output_log = outputdir_guest / f"{date}_seq{idx}.log"
                trace_output = (
                    (outputdir_guest / f"{date}_seq{idx}_trace.dump")
                    if tracing
                    else None
                )
            else:
                output_log = outputdir_guest / f"{date}.log"
                trace_output = (
                    (outputdir_guest / f"{date}_trace.dump") if tracing else None
                )

            job_path = str(JOBS_DIR / current_job)

            cmd = [client_binary, job_path, "--config", config_file, "--device", device]

            if tc_file:
                cmd.extend(["--tc-path", str(tc_file)])

            if silent:
                cmd.append("--silent")

            cmd.extend(sync.client_args())
            cmd.extend(abl.client_args())
            cmd.extend(pipe.client_args())

            arms = [a.path_component() for a in (io, abl) if not a.is_default()]
            arm = f" [{' '.join(arms)}]" if arms else ""

            if tracing:
                ring = (
                    f", ring {trace_entries} entries "
                    f"({trace_entries * 32 // (1024 * 1024)} MiB)"
                    if trace_entries is not None
                    else ""
                )
                print(
                    f"Running {client_binary} (job {idx+1}/{len(jobs)}){arm}"
                    f" with {tracing} enabled{ring}"
                )
                if tracing == "tracing":
                    cmd.append("--tracing")
                elif tracing == "small-trace":
                    cmd.append("--small-trace")
                if trace_entries is not None:
                    cmd.extend(["--trace-entries", str(trace_entries)])
                _run_client_with_tracing(vm, cmd, output_log, trace_output, client)
            else:
                print(f"Running {client_binary} (job {idx+1}/{len(jobs)}){arm}")
                result = vm.ssh_cmd(cmd, check=False)
                _save_output_to_file(vm, output_log, result.stdout, result.stderr)


def _run_client_with_tracing(
    vm, cmd: list, output_log: Path, trace_output: Path, app_name: str
) -> None:
    """Run the client under spdk_trace_record and convert the dump to JSON.

    ``app_name`` names the SPDK shm trace file, which every build variant
    registers under the plain client name.
    """
    wrapper_script_path = f"/tmp/accelstore_run_{uuid.uuid4().hex[:8]}.sh"

    cmd_str = " ".join(cmd)

    wrapper_script = f"""#!/usr/bin/env bash
set -e

{cmd_str} > {output_log} 2>&1 &
CLIENT_PID=$!
echo $CLIENT_PID

SHM_FILE="/dev/shm/{app_name}_trace.pid$CLIENT_PID"

TIMEOUT=3000
ELAPSED=0
while [ ! -e "$SHM_FILE" ] && [ $ELAPSED -lt $TIMEOUT ]; do
    if ! kill -0 $CLIENT_PID 2>/dev/null; then
        echo "ERROR: client exited before creating $SHM_FILE" >&2
        wait $CLIENT_PID || true
        exit 1
    fi
    sleep 0.01
    ELAPSED=$((ELAPSED + 1))
done

if [ ! -e "$SHM_FILE" ]; then
    echo "ERROR: Timeout waiting for trace file $SHM_FILE to be created" >&2
    kill $CLIENT_PID 2>/dev/null || true
    exit 1
fi

# Attaching before spdk_trace_init() has filled the header fails with "Invalid tsc_rate 0".
READY_TIMEOUT=1000
ELAPSED=0
while [ $ELAPSED -lt $READY_TIMEOUT ]; do
    if grep -q "for offline analysis/debug" {output_log} 2>/dev/null; then
        break
    fi
    if ! kill -0 $CLIENT_PID 2>/dev/null; then
        break
    fi
    sleep 0.01
    ELAPSED=$((ELAPSED + 1))
done

# spdk_trace_record exits rather than waiting when the header is not ready yet.
TRACE_PID=""
for attempt in 1 2 3; do
    spdk_trace_record -s {app_name} -p $CLIENT_PID -f {trace_output} -q &
    CANDIDATE=$!
    sleep 0.5
    if kill -0 $CANDIDATE 2>/dev/null; then
        TRACE_PID=$CANDIDATE
        break
    fi
    wait $CANDIDATE 2>/dev/null || true
    echo "WARNING: spdk_trace_record exited immediately (attempt $attempt), retrying" >&2
    rm -f {trace_output} {trace_output}-*
    sleep 0.5
done

if [ -z "$TRACE_PID" ]; then
    echo "ERROR: could not attach spdk_trace_record to $CLIENT_PID" >&2
    kill $CLIENT_PID 2>/dev/null || true
    wait $CLIENT_PID 2>/dev/null || true
    rm -f "$SHM_FILE"
    exit {TRACE_PIPELINE_FAILED}
fi

CLIENT_EXIT=0
wait $CLIENT_PID || CLIENT_EXIT=$?  # set -e must not abort the teardown below

sleep 1  # let spdk_trace_record flush

kill $TRACE_PID 2>/dev/null || true
wait $TRACE_PID 2>/dev/null || true

rm -f "$SHM_FILE"

if [ ! -s {trace_output} ]; then
    echo "ERROR: trace dump {trace_output} is missing or empty" >&2
    exit {TRACE_PIPELINE_FAILED}
fi

echo "converting to JSON"
if ! spdk_trace -f {trace_output} -j > {trace_output}.json; then
    echo "ERROR: spdk_trace failed to convert {trace_output}" >&2
    rm -f {trace_output}.json
    exit {TRACE_PIPELINE_FAILED}
fi

if [ ! -s {trace_output}.json ]; then
    echo "ERROR: {trace_output}.json is empty" >&2
    rm -f {trace_output}.json
    exit {TRACE_PIPELINE_FAILED}
fi

exit $CLIENT_EXIT
"""

    vm.create_file(wrapper_script_path, wrapper_script)
    vm.ssh_cmd(["chmod", "+x", wrapper_script_path], check=True)

    print("Running client with tracing via wrapper script")
    result = vm.ssh_cmd([wrapper_script_path], check=False)

    print(f"result: {result.stdout}")

    if result.returncode == TRACE_PIPELINE_FAILED:
        print("=" * 72)
        print(f"TRACE FAILURE: no usable trace produced for {trace_output}")
        print("=" * 72)
        vm.ssh_cmd(
            ["sh", "-c", f"echo 'trace pipeline failed' > {trace_output}.FAILED"],
            check=False,
        )
    elif result.returncode != 0:
        print(f"Client exited with code {result.returncode}")

    vm.ssh_cmd(["rm", "-f", wrapper_script_path], check=False)


def _save_output_to_file(
    vm,
    output_file: Path,
    stdout: str,
    stderr: str,
    returncode: Optional[int] = None,
) -> None:
    """Write the client's stdout, stderr and exit code to a file in the guest.

    `stderr` is only non-empty when the caller asked ssh_cmd for it, and the
    exit code is recorded only when `returncode` is given.
    """
    combined_output = ""
    if stdout:
        combined_output += f"=== STDOUT ===\n{stdout}\n"
    if stderr:
        combined_output += f"=== STDERR ===\n{stderr}\n"

    if not combined_output:
        combined_output = "(no output)\n"

    if returncode is not None:
        combined_output += f"=== EXIT: {returncode} ===\n"

    vm.create_file(str(output_file), combined_output)
    print(f"Saved output to: {output_file}")
