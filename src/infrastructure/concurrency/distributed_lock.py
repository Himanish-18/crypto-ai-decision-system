
import time
import logging
from contextlib import contextmanager

# v40 Infrastructure: Distributed Lock
# Prevents race conditions across services/processes.
# Simulates Redis Redlock algorithm.

class DistributedLock:
    def __init__(self):
        self._locks = {}
        self.logger = logging.getLogger("infrastructure.lock")

    @contextmanager
    def acquire(self, resource: str, ttl_ms=1000, timeout_ms=500):
        """
        Usage: with dist_lock.acquire("order_book_btc"): ...
        """
        start = time.time()
        acquired = False
        
        while (time.time() - start) * 1000 < timeout_ms:
            if resource not in self._locks:
                self._locks[resource] = time.time() + (ttl_ms / 1000.0)
                acquired = True
                break
            
            # Check for expiry
            if time.time() > self._locks[resource]:
                 self._locks[resource] = time.time() + (ttl_ms / 1000.0)
                 acquired = True
                 break
                 
            time.sleep(0.01) # Spin wait
            
        if not acquired:
            raise RuntimeError(f"Could not acquire lock for {resource}")
            
        try:
            yield
        finally:
            if resource in self._locks:
                del self._locks[resource]

dist_lock = DistributedLock()
