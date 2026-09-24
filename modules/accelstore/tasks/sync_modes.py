#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metadata persistence mode of an AccelStore client run.

The epoch trigger, whether a mutation may be answered before its epoch is
durable, how the counter commit is scheduled, and which counter service answers
it -- the clients' ``--sync-*`` and ``--tc-*`` flags as one object. Validation
mirrors ``lib/as_sync_config.c`` and ``clients/client_common.c``.
"""

from dataclasses import dataclass, fields
from typing import Optional

# lib/as_sync_config.c, g_trigger_tokens / g_coupling_tokens.
TRIGGERS = ("on-shutdown", "on-barrier", "every-n", "every-t", "per-op")
COUPLINGS = ("ordered", "parallel", "independent")

# The triggers --sync-blocking accepts: only a scheduled epoch can be blocked on.
BLOCKING_TRIGGERS = ("every-n", "every-t")

# The library defaults, i.e. what a client with no --sync-*/--tc-* flag runs.
DEFAULT_TRIGGER = "on-shutdown"
DEFAULT_COUPLING = "ordered"
DEFAULT_TC_BACKEND = "local"
DEFAULT_TC_RTT_MODEL = "every"

TC_BACKENDS = ("local", "mock-remote")
TC_RTT_MODELS = ("every", "periodic", "elapsed")

# include/tc_remote_mock.h TC_MOCK_RTT_US_DEFAULT.
DEFAULT_TC_RTT_US = 2000

# One hour in us: AS_SYNC_INTERVAL_US_MAX and TC_MOCK_OPERAND_MAX.
INTERVAL_US_MAX = 3600 * 1000 * 1000
TC_OPERAND_MAX = 3600 * 1000 * 1000

NUMERIC_FIELDS = (
    "every_n",
    "interval_us",
    "commit_every",
    "tc_rtt_us",
    "tc_rtt_interval",
)

# Couplings the library parses but refuses, with the reason it gives.
REFUSED_COUPLINGS = {
    "independent": (
        "needs the K-generation freshness window:"
        " LOG's != and KVS's <= counter checks are not K-aware yet"
    ),
}


def _epoch_label_us(interval_us: int) -> str:
    """Render an epoch interval: 100000 -> '100ms'."""
    if interval_us % 1000 == 0:
        ms = interval_us // 1000
        if ms % 1000 == 0:
            return f"{ms // 1000}s"
        return f"{ms}ms"
    return f"{interval_us}us"


@dataclass(frozen=True)
class SyncMode:
    """One metadata-persistence configuration; ``None`` means client default."""

    trigger: Optional[str] = None
    every_n: Optional[int] = None
    interval_us: Optional[int] = None
    coupling: Optional[str] = None
    commit_every: Optional[int] = None
    blocking: bool = False
    metadata_only: bool = False
    tc_backend: Optional[str] = None
    tc_rtt_us: Optional[int] = None
    tc_rtt_model: Optional[str] = None
    tc_rtt_interval: Optional[int] = None

    def __post_init__(self) -> None:
        """Coerce the numeric knobs, which invoke passes in as strings."""
        for name in NUMERIC_FIELDS:
            value = getattr(self, name)
            if value is None or isinstance(value, int):
                continue
            try:
                coerced = int(str(value).strip())
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be an integer, got {value!r}") from None
            object.__setattr__(self, name, coerced)

    @classmethod
    def from_config(cls, cfg: dict) -> "SyncMode":
        """Pick the sync keys out of an action config, ignoring the rest."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names and v is not None})

    def as_config(self) -> dict:
        """The keys to merge into an action_config; omits unset knobs."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) not in (None, False)
        }

    @classmethod
    def from_task_kwargs(cls, **kwargs) -> "SyncMode":
        """Build from the ``--sync-*``/``--tc-*`` names ``run-all`` exposes."""
        renamed = {
            k[len("sync_") :] if k.startswith("sync_") else k: v
            for k, v in kwargs.items()
        }
        names = {f.name for f in fields(cls)}
        unknown = set(renamed) - names
        if unknown:
            raise ValueError(f"not sync arguments: {', '.join(sorted(unknown))}")
        return cls(**{k: v for k, v in renamed.items() if v is not None})

    def as_task_kwargs(self) -> dict:
        """The ``run-all`` arguments for this mode, for a sweep to splat."""
        prefixed = {
            "trigger",
            "every_n",
            "interval_us",
            "coupling",
            "commit_every",
            "blocking",
            "metadata_only",
        }
        return {
            (f"sync_{k}" if k in prefixed else k): v
            for k, v in self.as_config().items()
        }

    def is_default(self) -> bool:
        """True for the run the library does with no flags at all."""
        return self.path_component() is None

    @property
    def effective_trigger(self) -> str:
        return self.trigger or DEFAULT_TRIGGER

    @property
    def effective_coupling(self) -> str:
        return self.coupling or DEFAULT_COUPLING

    @property
    def effective_tc_backend(self) -> str:
        return self.tc_backend or DEFAULT_TC_BACKEND

    @property
    def rtt_us(self) -> int:
        """Round trip a paying commit waits out, 0 under the local counter."""
        if self.effective_tc_backend != "mock-remote":
            return 0
        return self.tc_rtt_us if self.tc_rtt_us is not None else DEFAULT_TC_RTT_US

    @property
    def epoch_us(self) -> Optional[int]:
        """The durability window in us, which only ``every-t`` states."""
        return self.interval_us if self.effective_trigger == "every-t" else None

    @property
    def epoch_ops(self) -> Optional[int]:
        """The durability window in mutating operations, for ``every-n``."""
        return self.every_n if self.effective_trigger == "every-n" else None

    def validate(self) -> None:
        """Reject what the library and the clients reject, before a VM boots."""
        if self.trigger is not None and self.trigger not in TRIGGERS:
            raise ValueError(
                f"sync trigger '{self.trigger}' is unknown."
                f" Accepted: {'|'.join(TRIGGERS)}"
            )
        if self.coupling is not None and self.coupling not in COUPLINGS:
            raise ValueError(
                f"sync coupling '{self.coupling}' is unknown."
                f" Accepted: {'|'.join(COUPLINGS)}"
            )

        trigger = self.effective_trigger
        if trigger == "every-n":
            if not self.every_n:
                raise ValueError("sync trigger every-n needs an operation count")
        elif self.every_n:
            raise ValueError(
                "an operation count only has a meaning for trigger every-n"
                f" (got trigger {trigger})"
            )

        if trigger == "every-t":
            if not self.interval_us:
                raise ValueError("sync trigger every-t needs an interval")
        elif self.interval_us:
            raise ValueError(
                "an interval only has a meaning for trigger every-t"
                f" (got trigger {trigger})"
            )

        if self.interval_us is not None and not 0 < self.interval_us <= INTERVAL_US_MAX:
            raise ValueError(
                f"the durability window is bounded at {INTERVAL_US_MAX} us (one hour)"
            )
        if self.every_n is not None and self.every_n <= 0:
            raise ValueError("an operation count must be > 0")

        coupling = self.effective_coupling
        if coupling in REFUSED_COUPLINGS:
            raise ValueError(f"coupling {coupling} {REFUSED_COUPLINGS[coupling]}")

        if self.commit_every is not None:
            if self.commit_every <= 0:
                raise ValueError("a commit ratio must be > 0")
            if self.commit_every > 1:
                raise ValueError(
                    "a commit ratio > 1 only has a meaning under coupling independent,"
                    " which needs the K-generation freshness window first"
                )

        if self.blocking and trigger not in BLOCKING_TRIGGERS:
            raise ValueError(
                "blocking only has a meaning for trigger every-n or every-t:"
                " on-shutdown and on-barrier have no scheduled epoch to block on,"
                " and per-op already delivers at durability"
                f" (got trigger {trigger})"
            )

        if self.metadata_only and coupling == "parallel":
            raise ValueError(
                "metadata-only issues no counter commit, so coupling parallel has"
                " nothing to overlap"
            )

        # The RTT knobs describe the mock's timer, so they need mock-remote.
        backend = self.effective_tc_backend
        if backend not in TC_BACKENDS:
            raise ValueError(
                f"tc backend '{backend}' is unknown. Accepted: {'|'.join(TC_BACKENDS)}"
            )
        rtt_flags_given = any(
            v is not None
            for v in (self.tc_rtt_us, self.tc_rtt_model, self.tc_rtt_interval)
        )
        if backend == "local" and rtt_flags_given:
            raise ValueError(
                "tc_rtt_us/tc_rtt_model/tc_rtt_interval need tc_backend mock-remote"
            )
        if self.tc_rtt_model is not None and self.tc_rtt_model not in TC_RTT_MODELS:
            raise ValueError(
                f"tc rtt model '{self.tc_rtt_model}' is unknown."
                f" Accepted: {'|'.join(TC_RTT_MODELS)}"
            )
        for name, value in (
            ("tc_rtt_us", self.tc_rtt_us),
            ("tc_rtt_interval", self.tc_rtt_interval),
        ):
            if value is not None and not 0 < value <= TC_OPERAND_MAX:
                raise ValueError(f"{name} must be > 0 and <= {TC_OPERAND_MAX}")

        if backend == "mock-remote":
            model = self.tc_rtt_model or DEFAULT_TC_RTT_MODEL
            if model != "every" and not self.tc_rtt_interval:
                raise ValueError(
                    f"tc rtt model {model} needs tc_rtt_interval"
                    " (periodic: commits between paying ones; elapsed: us between them)"
                )
            if model == "every" and self.tc_rtt_interval:
                raise ValueError("tc_rtt_interval has no meaning for rtt model every")

        # The local counter commits inline, leaving no round trip to overlap.
        if coupling == "parallel" and backend == "local":
            raise ValueError(
                "coupling parallel needs a counter service with a round trip;"
                " pass tc_backend mock-remote"
            )

    def warnings(self) -> list[str]:
        """Configurations that run but probably do not measure what they say."""
        out = []
        if self.effective_coupling == "parallel" and self.rtt_us:
            out.append(
                f"coupling parallel at a {self.rtt_us} us round trip only pays where"
                " the metadata blob write (D) is shorter than that round trip;"
                " on FS and LOG D is ~260 ms (full node-map snapshot per epoch),"
                " so every epoch loses the race and the run measures ordered's cost"
                " (read stats.commits_race_lost, not the epoch count)"
            )
        if self.metadata_only:
            out.append(
                "metadata-only commits no counter value: the run measures the"
                " durability cost of an epoch alone and has no freshness"
                " guarantee at all"
            )
        return out

    def client_args(self) -> list[str]:
        """The client command-line flags for this mode."""
        args: list[str] = []
        for flag, value in (
            ("--sync-trigger", self.trigger),
            ("--sync-every-n", self.every_n),
            ("--sync-interval-us", self.interval_us),
            ("--sync-coupling", self.coupling),
            ("--sync-commit-every", self.commit_every),
        ):
            if value is not None:
                args += [flag, str(value)]
        if self.blocking:
            args.append("--sync-blocking")
        if self.metadata_only:
            args.append("--sync-metadata-only")
        for flag, value in (
            ("--tc-backend", self.tc_backend),
            ("--tc-rtt-us", self.tc_rtt_us),
            ("--tc-rtt-model", self.tc_rtt_model),
            ("--tc-rtt-interval", self.tc_rtt_interval),
        ):
            if value is not None:
                args += [flag, str(value)]
        return args

    def path_component(self) -> Optional[str]:
        """Stable slug for this mode in a result path, ``None`` for the default."""
        parts = []
        if self.trigger:
            if self.trigger == "every-n":
                parts.append(f"sync-every-n-{self.every_n}")
            elif self.trigger == "every-t":
                parts.append(f"sync-every-t-{self.interval_us}us")
            else:
                parts.append(f"sync-{self.trigger}")
        if self.coupling:
            parts.append(f"coupling-{self.coupling}")
        if self.commit_every:
            parts.append(f"k-{self.commit_every}")
        if self.blocking:
            parts.append("blocking")
        if self.metadata_only:
            parts.append("metadata-only")
        if self.tc_backend:
            parts.append(f"tc-{self.tc_backend}")
        if self.tc_rtt_us:
            parts.append(f"rtt-{self.tc_rtt_us}us")
        if self.tc_rtt_model:
            parts.append(f"rttmodel-{self.tc_rtt_model}")
        if self.tc_rtt_interval:
            parts.append(f"rttint-{self.tc_rtt_interval}")
        return "_".join(parts) if parts else None

    def label(self) -> str:
        """Short human-readable name for a table row or a plot legend."""
        if self.is_default():
            return "baseline (on-shutdown, local counter)"
        parts = []
        trigger = self.effective_trigger
        if trigger == "every-t" and self.interval_us:
            parts.append(f"every {_epoch_label_us(self.interval_us)}")
        elif trigger == "every-n" and self.every_n:
            parts.append(f"every {self.every_n} ops")
        else:
            parts.append(trigger)
        if self.blocking:
            parts.append("blocking")
        if self.coupling and self.coupling != DEFAULT_COUPLING:
            parts.append(self.coupling)
        if self.metadata_only:
            parts.append("metadata-only")
        elif self.rtt_us:
            parts.append(f"RTT {_epoch_label_us(self.rtt_us)}")
        return ", ".join(parts)


def parse_path_component(slug: Optional[str]) -> SyncMode:
    """Decode a result-path slug back into a :class:`SyncMode`.

    The inverse of :meth:`SyncMode.path_component`. ``None`` or an empty slug
    is the default mode.

    >>> parse_path_component("sync-every-t-100000us_tc-mock-remote_rtt-5000us")
    SyncMode(trigger='every-t', every_n=None, interval_us=100000, coupling=None, commit_every=None, blocking=False, metadata_only=False, tc_backend='mock-remote', tc_rtt_us=5000, tc_rtt_model=None, tc_rtt_interval=None)

    Raises ``ValueError`` on a token this module did not write.
    """
    if not slug:
        return SyncMode()

    kwargs: dict = {}
    for token in slug.split("_"):
        if token.startswith("sync-"):
            rest = token[len("sync-") :]
            if rest.startswith("every-n-"):
                kwargs["trigger"] = "every-n"
                kwargs["every_n"] = int(rest[len("every-n-") :])
            elif rest.startswith("every-t-"):
                kwargs["trigger"] = "every-t"
                kwargs["interval_us"] = int(rest[len("every-t-") :].removesuffix("us"))
            elif rest in TRIGGERS:
                kwargs["trigger"] = rest
            else:
                raise ValueError(f"unknown sync trigger token '{token}' in '{slug}'")
        elif token.startswith("coupling-"):
            kwargs["coupling"] = token[len("coupling-") :]
        elif token.startswith("k-"):
            kwargs["commit_every"] = int(token[len("k-") :])
        elif token == "blocking":
            kwargs["blocking"] = True
        elif token == "metadata-only":
            kwargs["metadata_only"] = True
        elif token.startswith("tc-"):
            kwargs["tc_backend"] = token[len("tc-") :]
        elif token.startswith("rttmodel-"):
            kwargs["tc_rtt_model"] = token[len("rttmodel-") :]
        elif token.startswith("rttint-"):
            kwargs["tc_rtt_interval"] = int(token[len("rttint-") :])
        elif token.startswith("rtt-"):
            kwargs["tc_rtt_us"] = int(token[len("rtt-") :].removesuffix("us"))
        else:
            raise ValueError(f"unknown sync token '{token}' in '{slug}'")

    return SyncMode(**kwargs)


def is_sync_component(name: str) -> bool:
    """Whether a directory name is (meant to be) one of this module's slugs.

    A name starting with ``sync-`` counts even if the rest does not parse, so a
    malformed slug is reported by the loader rather than quietly skipped.
    """
    if name.startswith("sync-"):
        return True
    try:
        return not parse_path_component(name).is_default()
    except ValueError:
        return False


# The submitted paper's epoch ladder (§VII-B), kept so the figure keeps its x axis.
EPOCH_MS_MIN = 1
EPOCH_MS_MAX = 281
EPOCH_MS_STEP = 20

# The RTT the submitted evaluation models, not the client's own 2 ms default.
PAPER_RTT_US = 5000

# Mutating operations between epochs, powers of ten since the range spans four.
EVERY_N_LADDER = (1, 10, 100, 1000, 10000)


def epoch_ladder_ms(
    min_ms: int = EPOCH_MS_MIN,
    max_ms: int = EPOCH_MS_MAX,
    step_ms: int = EPOCH_MS_STEP,
) -> list[int]:
    """The every-t ladder, in milliseconds."""
    if min_ms <= 0 or step_ms <= 0 or max_ms < min_ms:
        raise ValueError(
            f"epoch ladder [{min_ms}, {max_ms}] step {step_ms} is empty or negative"
        )
    return list(range(min_ms, max_ms + 1, step_ms))
