
import uuid
import time
import logging
from contextlib import contextmanager
from typing import Optional

# v40 Infrastructure: Distributed Tracing
# Generates Trace IDs and Spans for debugging distributed flows.

class Span:
    def __init__(self, trace_id: str, span_id: str, name: str, parent_id: Optional[str] = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.parent_id = parent_id
        self.start_time = time.time()
        self.duration = 0.0

    def finish(self):
        self.duration = (time.time() - self.start_time) * 1000 # ms

class Tracer:
    def __init__(self):
        self.logger = logging.getLogger("infrastructure.trace")
        self._active_span = None

    @contextmanager
    def start_span(self, name: str, trace_id: Optional[str] = None):
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())[:8]
        parent_id = self._active_span.span_id if self._active_span else None
        
        span = Span(trace_id, span_id, name, parent_id)
        self._active_span = span
        
        try:
            yield span
        finally:
            span.finish()
            self.logger.info(f"🕵️ Trace: [{span.trace_id}] {span.name} took {span.duration:.2f}ms")
            self._active_span = None # Simple stack (assumes single threaded for logic flow)

tracer = Tracer()
