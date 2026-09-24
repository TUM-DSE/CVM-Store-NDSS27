import time
from contextlib import contextmanager
from typing import Dict


class TimingLogger:
    """Collects named wall-clock durations in seconds."""

    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = time.perf_counter() - start

    def add(self, name: str, value: float):
        self.timings[name] = value

    def to_dict(self) -> Dict[str, float]:
        return self.timings.copy()
