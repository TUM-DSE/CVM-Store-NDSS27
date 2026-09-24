"""The paper's figures and tables, one task each: ``oma-arvio-root paper.<x>``.

:data:`FIGURES` writes each figure's run, parse and plot steps down as plain
data, and every ``paper.*`` task runs one of them in that order. A step that
raises is recorded and the next one runs, so a partial night still parses and
plots before the task exits non-zero.
"""

import inspect
import time

from invoke import task

# Modules, not task objects: Collection.from_module would register a bound Task.
from modules.accelstore.tasks import accelstore as acc
from modules.accelstore.tasks import ycsb_tasks
from modules.attestation.tasks import attestation as att
from modules.attestation.tasks import veraison as veraison_tasks

KINDS = ("run", "parse", "plot")
SETUPS = ["snp"]  # every run step boots the SNP guest
STACKS = ("rollbaccine", "rollbaccine-f1m")  # arms that need the stack guest

# A step is (kind, task, kwargs). FIGURES: name -> (description, output, steps).
_ARMS = "ext4,rollbaccine,accelstore"
_BENCH = dict(backends=_ARMS, crypto="none,aegis")
_YCSB = dict(arms=_ARMS, crypto="none,aegis")
_THREADS = ((1, {}), (8, dict(split_loops=True)))

_MLPERF = dict(
    variants="ckpt-llama3-8b-subset",
    arms="accelstore-log,accelstore-fs,posix-direct",
    stacks="ext4,rollbaccine-f1m",
    sizes="large",
)

# The two ends of every client's size axis, traced, on the medium SNP guest.
_AMPLIFICATION = dict(
    setups="snp",
    sizes="medium",
    clients="kvs_client,log_client,fs_client",
    crypto="aegis",
    tracing="tracing",
    precondition_scheme="null",
    jobs=",".join(
        [
            "kvs_workload_A_64m.txt",
            "kvs_workload_A_4g.txt",
            "log_workload_64MB_write_read_seq.txt",
            "log_workload_4GB_write_read_seq.txt",
            "fs_workload_write_read_64m.txt",
            "fs_workload_write_read_4g.txt",
        ]
    ),
)

FIGURES = {
    "fio-iops": (
        "The fio IOPS figure",
        "bench-result/plots/fio-iops_iops.pdf",
        [("run", acc.run_fio_iops, {}), ("plot", acc.plot_fio_iops, {})],
    ),
    "fio-stacks": (
        "The block-stack figure",
        "bench-result/parsed/fio-bandwidth.csv + bench-result/plots/fio-luks_iops.pdf",
        [
            ("run", acc.run_fio_iops, {}),
            ("run", acc.run_fio_bandwidth, {}),
            ("parse", acc.parse_fio_bandwidth, {}),
            ("plot", acc.plot_fio_luks, {}),
        ],
    ),
    "crypto": (
        "The tcrypt/openssl table",
        "bench-result/parsed/crypto-summary.csv",
        [("run", acc.run_crypto, {}), ("parse", acc.parse_crypto, {})],
    ),
    "attestation": (
        "The SNP attestation table",
        "bench-result/attestation-summary.csv",
        [("run", att.run, dict(iterations=3)), ("parse", att.parse, dict(runs=3))],
    ),
    "amplification": (
        "The R/W amplification table",
        "bench-result/parsed/amplification.csv",
        [
            ("run", acc.run_all, _AMPLIFICATION),
            (
                "run",
                acc.run_all,
                dict(_AMPLIFICATION, sync_trigger="every-t", sync_interval_us=100000),
            ),
            ("parse", acc.parse_amplification, {}),
        ],
    ),
    "epoch": (
        "The metadata epoch figure",
        "bench-result/plots/epoch-overhead.pdf",
        [("run", acc.run_epoch_sweep, {}), ("plot", acc.plot_epoch, {})],
    ),
    "ablation": (
        "The ablation figure",
        "bench-result/plots/ablation.pdf",
        [
            ("run", acc.run_ablation, dict(io_path="spdk")),
            ("run", acc.run_ablation, dict(io_path="kernel", arms="aegis:none")),
            ("parse", acc.parse_ablation, {}),
            ("plot", acc.plot_ablation, {}),
        ],
    ),
    "seal-unseal": (
        "The seal->unseal recovery table",
        "bench-result/parsed/seal-unseal-summary.csv",
        [("run", acc.run_seal_unseal, {}), ("parse", acc.parse_seal_unseal, {})],
    ),
    "synthetic": (
        "The synthetic figure",
        "bench-result/plots/synthetic.pdf",
        [
            ("run", acc.run_bench, dict(suite="random-read", **_BENCH, threads=t, **s))
            for t, s in _THREADS
        ]
        + [
            ("run", acc.run_ycsb, dict(points=p, **_YCSB, **s))
            for p in ("large", "small")
            for s in ({}, dict(threads=8))
        ]
        + [
            ("run", acc.run_bench, dict(suite=suite, **_BENCH, threads=t, **s))
            for suite in ("log-small", "log-large", "fs-large")
            for t, s in _THREADS
        ]
        + [
            ("parse", acc.parse_bench, {}),
            ("parse", acc.parse_ycsb, {}),
            ("plot", acc.plot_synthetic, {}),
        ],
    ),
    "apps": (
        "The application figure",
        "bench-result/plots/apps.pdf",
        [
            ("run", acc.run_mlperf, _MLPERF),
            ("run", acc.run_mlperf, dict(_MLPERF, arms="posix-buffered")),
            (
                "run",
                acc.run_bench,
                dict(
                    suite="apps",
                    backends="ext4,rollbaccine-f1m,accelstore",
                    crypto="aegis",
                    split_loops=True,
                ),
            ),
            (
                "run",
                acc.run_bench,
                dict(
                    suite="apps",
                    backends="ext4,rollbaccine-f1m",
                    split_loops=True,
                    direct=False,
                ),
            ),
            (
                "run",
                acc.run_redis,
                dict(arms="accelstore,ext4,rollbaccine-f1m"),
            ),
            ("parse", acc.parse_mlperf, {}),
            ("parse", acc.parse_bench, {}),
            ("parse", acc.parse_redis, {}),
            ("plot", acc.plot_apps, {}),
        ],
    ),
}


# `--quick` is the same figures over fewer points. A figure absent here is unreduced.
QUICK_SKIP = frozenset({"fio-iops", "fio-stacks"})

QUICK_EPOCH_MS_MAX = 70
QUICK_SEAL_UNSEAL_SIZES = "100mb,10gb,1000gb"
QUICK_PERSONALITY_DIR = "/shared/modules/accelstore/personalities-quick"


def _quick_logs(*names):
    """The reduced LOG personalities, as the guest paths run-bench takes."""
    return ",".join(f"{QUICK_PERSONALITY_DIR}/{name}.wl" for name in names)


_QUICK_ABL = dict(
    jobs=",".join(
        job for jobs in acc.ACCELSTORE_ABLATION_JOBS_QUICK.values() for job in jobs
    )
)
_BENCH_QUICK = dict(_BENCH, quick=True)

# (personalities, the run flags naming them by hand does not infer)
_BENCH_QUICK_SETS = (
    ("fs_random_read_4k,fs_random_read_64k,fs_random_read_1m", {}),
    # --personalities infers no geometry from a suite, and log-small needs one.
    (_quick_logs("log_small_16k", "log_small_1m"), dict(derive_store_geometry=True)),
    (_quick_logs("log_write_read_256mb", "log_write_read_1gb"), {}),
    ("fs_write_read_64m,fs_write_read_256m,fs_write_read_1g", {}),
)

# The A sweep's two halves, each at half its byte target, in FIGURES' order.
_YCSB_QUICK_HALVES = (
    ("A_256m,A_1g", dict(large_total_gb=128)),
    ("A_16k,A_1m", dict(direction_bytes_gb=64)),
)

QUICK = {
    "epoch": [
        ("run", acc.run_epoch_sweep, dict(max_epoch_ms=QUICK_EPOCH_MS_MAX)),
        ("plot", acc.plot_epoch, {}),
    ],
    "seal-unseal": [
        ("run", acc.run_seal_unseal, dict(data_sizes=QUICK_SEAL_UNSEAL_SIZES)),
        ("parse", acc.parse_seal_unseal, {}),
    ],
    "ablation": [
        ("run", acc.run_ablation, dict(io_path="spdk", **_QUICK_ABL)),
        (
            "run",
            acc.run_ablation,
            dict(io_path="kernel", arms="aegis:none", **_QUICK_ABL),
        ),
        ("parse", acc.parse_ablation, _QUICK_ABL),
        ("plot", acc.plot_ablation, _QUICK_ABL),
    ],
    "synthetic": [
        (
            "run",
            acc.run_ycsb,
            dict(points=points, **_YCSB, quick=True, **half, **threading),
        )
        for points, half in _YCSB_QUICK_HALVES
        for threading in ({}, dict(threads=8))
    ]
    + [
        (
            "run",
            acc.run_bench,
            dict(personalities=pers, **_BENCH_QUICK, threads=t, **flags, **s),
        )
        for pers, flags in _BENCH_QUICK_SETS
        for t, s in _THREADS
    ]
    + [
        ("parse", acc.parse_bench, {}),
        ("parse", acc.parse_ycsb, {}),
        ("plot", acc.plot_synthetic, dict(quick=True)),
    ],
}


def command(fn, kwargs, hostname=None):
    """``oma-arvio-root <module>.<task> --key value ...`` for one step.

    ``<module>`` is read off the package of the function that defines the task.
    ``key_with_underscores`` becomes ``--key-with-underscores``, ``True`` the
    bare flag, ``False`` its ``--no-`` form, ``None`` is left out.
    """
    if hostname and "hostname" in inspect.signature(fn.body).parameters:
        kwargs = dict(kwargs, hostname=hostname)
    namespace = fn.body.__module__.split(".")[1]  # modules.<module>.tasks...
    parts = ["oma-arvio-root", f"{namespace}.{fn.name}"]
    for key, value in kwargs.items():
        flag = key.replace("_", "-")
        if value is True:
            parts.append(f"--{flag}")
        elif value is False:
            parts.append(f"--no-{flag}")
        elif value is not None:
            parts += [f"--{flag}", str(value)]
    return " ".join(parts)


def plan(names, plot_only=False, plot=True, quick=False):
    """The steps of the named figures in execution order: every run step, then
    the parse steps, then the plot steps; a step two figures share once."""
    kinds = [
        k for k in KINDS if not (plot_only and k == "run") and (plot or k != "plot")
    ]
    if quick:
        names = [name for name in names if name not in QUICK_SKIP]
    steps, seen = [], set()
    for kind in kinds:
        for name in names:
            for step in (QUICK if quick else {}).get(name, FIGURES[name][2]):
                if step[0] not in KINDS:
                    raise ValueError(f"step kind must be one of {KINDS}: {step}")
                line = command(step[1], step[2])
                if step[0] == kind and line not in seen:
                    seen.add(line)
                    steps.append(step)
    return steps


def execute(
    ctx,
    names,
    hostname=None,
    print_only=False,
    plot_only=False,
    plot=True,
    quick=False,
):
    """Run the plan of the named figures, then print a summary; exit non-zero
    if any step raised or any sweep reported a failed point."""
    skipped = [name for name in names if name in QUICK_SKIP] if quick else []
    if skipped:
        print(f"# --quick skips {', '.join(skipped)}: not part of the quick suite")
    steps = plan(names, plot_only, plot, quick)
    if print_only:
        for kind in KINDS:
            lines = [command(fn, kw, hostname) for k, fn, kw in steps if k == kind]
            if lines:
                print(f"# {kind}", *lines, sep="\n")
        return

    runs = [(fn, kw) for kind, fn, kw in steps if kind == "run"]
    if any(fn is acc.run_mlperf for fn, _ in runs):
        acc._require_mlperf_guest(SETUPS)
    if any(fn is att.run for fn, _ in runs):
        veraison_tasks.require_deployment()
    stacks = [
        s
        for s in STACKS
        if any(s in str(v).split(",") for _, kw in runs for v in kw.values())
    ]
    if stacks:
        ycsb_tasks._require_stack_guest(SETUPS, stacks)

    rows = []
    for kind, fn, kwargs in steps:
        line = command(fn, kwargs, hostname)
        print(f"\n### {line}")
        params = inspect.signature(fn.body).parameters
        call = dict(kwargs)
        if hostname and "hostname" in params:
            call["hostname"] = hostname
        if kind == "run" and "continue_on_error" in params:
            call["continue_on_error"] = True
        start, error, result = time.time(), None, None
        try:
            result = fn(ctx, **call)
        except SystemExit as exc:
            error = f"exit {exc.code}"
        except Exception as exc:  # noqa: BLE001 - one step is not the run
            error = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: {error}")
        statuses = [r.get("status") for r in result] if isinstance(result, list) else []
        rows.append((line, statuses, int(time.time() - start), error))

    print("\n" + "=" * 72)
    for line, statuses, seconds, error in rows:
        ok, failed = statuses.count("SUCCESS"), statuses.count("FAILED")
        counts = f"{ok:>4} ok {failed:>4} failed" if statuses else " " * 18
        print(
            f"{counts} {seconds // 60:>5}m  {line}" + (f"  [{error}]" if error else "")
        )
    print("=" * 72)
    if any(error or "FAILED" in statuses for _, statuses, _, error in rows):
        raise SystemExit(1)


FLAGS_DOC = """
    Args:
        hostname: Host to run on (default: current host)
        print_only: Print the commands this task runs, and run nothing
        plot_only: Skip the run steps; parse and plot what bench-result/ holds
        plot: --no-plot runs and parses, draws nothing
        quick: Fewer points and less data per point (skips the fio figures)
"""


def _figure_task(name, description, output):
    def body(
        ctx,
        hostname: str = None,
        print_only: bool = False,
        plot_only: bool = False,
        plot: bool = True,
        quick: bool = False,
    ):
        execute(ctx, [name], hostname, print_only, plot_only, plot, quick)

    body.__name__ = name.replace("-", "_")
    body.__doc__ = f"{description} -> {output}\n{FLAGS_DOC}"
    return task(name=name)(body)


for _name, (_description, _output, _) in FIGURES.items():
    globals()[_name.replace("-", "_")] = _figure_task(_name, _description, _output)


@task(name="all")
def all_figures(
    ctx,
    only: str = None,
    hostname: str = None,
    print_only: bool = False,
    plot_only: bool = False,
    plot: bool = True,
    quick: bool = False,
):
    """Every figure and table of the paper, cheapest first.

    fio-iops, fio-stacks, crypto, attestation, amplification, epoch, ablation,
    seal-unseal, synthetic, apps -- every run step of the selection, then every
    parse step, then every plot step, a step two figures share once.

    Args:
        only: Comma-separated figure names (default: all ten)
        hostname: Host to run on (default: current host)
        print_only: Print the commands this task runs, and run nothing
        plot_only: Skip the run steps; parse and plot what bench-result/ holds
        plot: --no-plot runs and parses, draws nothing
        quick: Fewer points and less data per point (skips the fio figures)
    """
    names = [n.strip() for n in only.split(",") if n.strip()] if only else list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise ValueError(
            f"Unknown figure(s) {', '.join(unknown)}. Figures: {', '.join(FIGURES)}"
        )
    names = [n for n in FIGURES if n in names]
    execute(ctx, names, hostname, print_only, plot_only, plot, quick)
