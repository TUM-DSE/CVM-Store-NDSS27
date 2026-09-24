"""Register accelstore actions."""

from . import accelstore  # noqa: F401 - side effect: @register_action calls
from . import bench  # noqa: F401 - side effect: @register_action calls
from . import mlperf  # noqa: F401 - side effect: @register_action calls
from . import redis  # noqa: F401 - side effect: @register_action calls
from . import ycsb  # noqa: F401 - side effect: @register_action calls
