#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-path ablation switches of an AccelStore client run.

``--ablation`` takes away one property of the crypto data path at a time.
Validation mirrors ``as_ablation_validate`` (``lib/as_ablation.c``), whose
rules depend on the build (``#ifdef DISABLE_CIF``).
"""

from dataclasses import dataclass
from typing import Optional

from modules.accelstore.tasks.crypto_backend import CryptoBackend

# as_ablation_to_string() order, so slugs do not depend on caller order.
ABLATION_TOKENS = (
    "no-overlap",
    "bounce-copy",
    "no-encrypt",
    "no-verify",
    "zero-copy",
)

# Not a client token: a "none" run passes no --ablation flag at all.
NO_ABLATION = "none"

SLUG_PREFIX = "abl-"


@dataclass(frozen=True)
class Ablation:
    """The ablation arm of one run.

    ``tokens`` is in :data:`ABLATION_TOKENS` order. ``explicit`` separates a
    run outside any ablation sweep (keeps today's result path) from the
    sweep's own unablated arm (gets the ``abl-none`` directory).
    """

    tokens: tuple[str, ...] = ()
    explicit: bool = False

    @classmethod
    def parse(cls, spec) -> "Ablation":
        """Coerce a task argument to an arm: ``None`` (outside a sweep),
        ``"none"`` (unablated arm) or a comma-separated token list."""
        if isinstance(spec, cls):
            return spec
        if spec is None:
            return cls()

        text = str(spec).strip()
        if not text:
            raise ValueError(
                "ablation: an empty token list is a typo, not 'no ablation'"
                f" -- omit the argument, or pass '{NO_ABLATION}' for the"
                " unablated arm of a sweep"
            )
        if text == NO_ABLATION:
            return cls(explicit=True)

        listed = [t.strip() for t in text.split(",")]
        if not all(listed):
            raise ValueError(f"ablation: empty token in '{spec}'")
        unknown = [t for t in listed if t not in ABLATION_TOKENS]
        if unknown:
            raise ValueError(
                f"ablation: unknown token(s) {unknown}."
                f" Accepted: {'|'.join(ABLATION_TOKENS)}, or '{NO_ABLATION}'"
            )
        # Canonical order, duplicates collapsed: the client ORs the bits.
        return cls(tuple(t for t in ABLATION_TOKENS if t in listed), True)

    @classmethod
    def from_config(cls, cfg: dict) -> "Ablation":
        """Pick the ablation key out of an action config, ignoring the rest."""
        return cls.parse(cfg.get("ablation"))

    @property
    def spec(self) -> Optional[str]:
        """The task/action argument this arm round-trips through."""
        if self.tokens:
            return ",".join(self.tokens)
        return NO_ABLATION if self.explicit else None

    def as_config(self) -> dict:
        """The key to merge into an action_config, empty when not an arm."""
        return {} if self.spec is None else {"ablation": self.spec}

    def as_task_kwargs(self) -> dict:
        """The ``run-all`` argument for this arm, for a sweep to splat."""
        return self.as_config()

    def is_default(self) -> bool:
        """True for a run that is not part of an ablation sweep at all."""
        return not self.tokens and not self.explicit

    def validate(self, backend: CryptoBackend = CryptoBackend.AEGIS) -> None:
        """Reject what ``as_ablation_validate`` would, ``backend`` standing in
        for the build (``#ifdef DISABLE_CIF``)."""
        if backend.encrypts:
            if "zero-copy" in self.tokens:
                raise ValueError(
                    "ablation 'zero-copy' requires the DISABLE_CIF build"
                    " (--crypto none); it hands the caller's buffer to the"
                    " device, which leaves nothing for an AEAD to do"
                )
        else:
            implied = [t for t in self.tokens if t in ("no-encrypt", "no-verify")]
            if implied:
                raise ValueError(
                    f"ablation {implied} is already implied by the DISABLE_CIF"
                    f" build (--crypto {backend.value}); the client rejects the"
                    " combination"
                )

        conflicting = [t for t in self.tokens if t in ("bounce-copy", "no-overlap")]
        if "zero-copy" in self.tokens and conflicting:
            raise ValueError(
                f"ablation 'zero-copy' is mutually exclusive with {conflicting}:"
                " it bypasses the staging buffers those two act on"
            )

    def client_args(self) -> list[str]:
        """The client flags for this arm, empty for the unablated arm."""
        return ["--ablation", ",".join(self.tokens)] if self.tokens else []

    def path_component(self) -> Optional[str]:
        """Stable slug identifying this arm in a result path.

        ``None`` outside a sweep. The unablated arm gets its own ``abl-none``
        slug so a sweep's baseline comes from the sweep's own session.
        """
        if self.is_default():
            return None
        return SLUG_PREFIX + ("+".join(self.tokens) if self.tokens else NO_ABLATION)

    def label(self) -> str:
        if not self.tokens:
            return "unablated"
        return "+".join(self.tokens)


def parse_path_component(slug: str) -> Ablation:
    """Decode a result-directory slug back into an arm.

    >>> parse_path_component("abl-no-encrypt").tokens
    ('no-encrypt',)
    >>> parse_path_component("abl-none").is_default()
    False
    """
    if not slug or not slug.startswith(SLUG_PREFIX):
        raise ValueError(f"not an ablation slug: '{slug}'")
    body = slug[len(SLUG_PREFIX) :]
    if body == NO_ABLATION:
        return Ablation(explicit=True)
    return Ablation.parse(",".join(body.split("+")))


def is_ablation_component(name: str) -> bool:
    """Whether a directory name is (meant to be) one of this module's slugs."""
    return name.startswith(SLUG_PREFIX)
