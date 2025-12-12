import logging
import time
from functools import wraps
from typing import Any

logger = logging.getLogger("latency_monitor")


class LatencyMonitor:
    """
    Tools to track execution latency.
    """

    @staticmethod
    def time_execution(log_threshold_ms: float = 100.0):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000.0

                if duration_ms > log_threshold_ms:
                    logger.warning(
                        f"🐢 Slow execution: {func.__name__} took {duration_ms:.2f}ms"
                    )
                else:
                    logger.debug(f"⚡ {func.__name__} took {duration_ms:.2f}ms")

                return result

            return wrapper

        return decorator
