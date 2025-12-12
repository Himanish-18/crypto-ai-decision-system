
import functools
import logging
import time

logger = logging.getLogger("fault_isolation")

class CircuitBreakerOpen(Exception):
    pass

class FaultIsolationZones:
    """
    Circuit Breaker decoration to isolate failures.
    """
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failures = 0
        self.last_failure_time = 0
        self.threshold = failure_threshold
        self.timeout = recovery_timeout
        self.state = "CLOSED" # OPEN, CLOSED, HALF-OPEN
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF-OPEN"
                else:
                    raise CircuitBreakerOpen(f"Zone {func.__name__} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                logger.error(f"Failure in {func.__name__}: {e}")
                
                if self.failures >= self.threshold:
                    self.state = "OPEN"
                    logger.critical(f"Circuit Breaker TRIPPED for {func.__name__}")
                raise e
        return wrapper
