
import logging
from typing import Any, Optional
import time

logger = logging.getLogger("redis_cache_stub")

class RedisCacheLayer:
    """
    In-memory simulation of Redis for microsecond feature lookup.
    """
    
    _INSTANCE = None
    
    def __init__(self):
        self.store = {}
        self.ttls = {}
        
    @classmethod
    def get_instance(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = RedisCacheLayer()
        return cls._INSTANCE

    def set(self, key: str, value: Any, ttl_sec: int = 60):
        self.store[key] = value
        self.ttls[key] = time.time() + ttl_sec

    def get(self, key: str) -> Optional[Any]:
        if key not in self.store: return None
        
        if time.time() > self.ttls.get(key, 0):
            del self.store[key]
            del self.ttls[key]
            return None
            
        return self.store[key]
