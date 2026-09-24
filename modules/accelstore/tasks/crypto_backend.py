#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The AEAD backend an AccelStore client build was compiled with."""

from enum import Enum


class CryptoBackend(Enum):
    """A crypto build variant of the AccelStore binaries."""

    AEGIS = "aegis"
    AESGCM = "aesgcm"
    NONE = "none"

    @property
    def binary_prefix(self) -> str:
        return "" if self is CryptoBackend.AEGIS else _PREFIXES[self]

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def encrypts(self) -> bool:
        """False for the DISABLE_CIF build, which runs no AEAD at all."""
        return self is not CryptoBackend.NONE

    def client_binary(self, base_client: str) -> str:
        """Binary name of ``base_client`` in this build variant.

        >>> CryptoBackend.AESGCM.client_binary("kvs_client")
        'aesgcm_kvs_client'
        """
        return self.binary_prefix + base_client

    @classmethod
    def parse(cls, value) -> "CryptoBackend":
        """Coerce a task argument (an id, or a member already) to a member."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown crypto backend '{value}'."
                f" Valid: {', '.join(b.value for b in cls)}"
            ) from None

    @classmethod
    def parse_list(cls, spec) -> list["CryptoBackend"]:
        """Parse a ``--crypto`` argument: comma-separated ids, ``"all"``, or
        ``None`` for :data:`DEFAULT_BACKENDS`."""
        if spec is None:
            return list(DEFAULT_BACKENDS)
        if isinstance(spec, str):
            spec = [s.strip() for s in spec.split(",") if s.strip()]
        if list(spec) == ["all"]:
            return list(cls)
        return [cls.parse(s) for s in spec]

    @classmethod
    def split_client_binary(cls, binary: str) -> tuple[str, "CryptoBackend"]:
        """Decode a binary name back into ``(base_client, backend)``."""
        for backend in _PREFIXED_LONGEST_FIRST:
            if binary.startswith(backend.binary_prefix):
                return binary.removeprefix(backend.binary_prefix), backend
        return binary, cls.AEGIS


_PREFIXES = {
    CryptoBackend.AESGCM: "aesgcm_",
    CryptoBackend.NONE: "no_",
}

_LABELS = {
    CryptoBackend.AEGIS: "AEGIS",
    CryptoBackend.AESGCM: "AES-GCM",
    CryptoBackend.NONE: "w/o CIF",
}

# The default build has no prefix, so it is the fallback rather than a match.
_PREFIXED_LONGEST_FIRST = sorted(
    (b for b in CryptoBackend if b.binary_prefix), key=lambda b: -len(b.binary_prefix)
)

DEFAULT_BACKENDS = [CryptoBackend.AEGIS, CryptoBackend.NONE]
