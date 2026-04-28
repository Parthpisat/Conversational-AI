import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Represents a single streaming event in the pipeline."""
    step_number: int
    event_name: str  # e.g., "nlp_translation_start", "nlp_translation_complete"
    name: str  # Human-readable name: "Translating Natural Language to SQL"
    status: str  # "pending", "running", "complete", "error"
    timestamp: str  # ISO format
    duration_ms: Optional[int] = None  # Time taken for this step
    details: Optional[Dict[str, Any]] = None  # Step-specific data
    error: Optional[str] = None  # Error message if status == "error"

    def to_json(self) -> str:
        """Convert to JSON string for SSE streaming."""
        return json.dumps(asdict(self))


class StreamingEventEmitter:
    """Manages pipeline events and emits them for streaming to frontend."""
    
    def __init__(self):
        self.events = []
        self.step_timings = {}  # Track timing for each step
        self.callbacks = []  # List of callbacks to call when event emitted
    
    def on_event(self, callback: Callable):
        """Register a callback to be called when any event is emitted."""
        self.callbacks.append(callback)
    
    def _emit(self, event: StreamEvent):
        """Internal: emit an event and call all registered callbacks."""
        self.events.append(event)
        logger.info(f"📡 EVENT: {event.event_name} - {event.name} - {event.status}")
        
        # Call all registered callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def start_step(self, step_number: int, step_name: str, description: str):
        """Mark the start of a pipeline step."""
        self.step_timings[step_name] = time.time()
        event = StreamEvent(
            step_number=step_number,
            event_name=f"{step_name}_start",
            name=description,
            status="running",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        self._emit(event)
        return event
    
    def complete_step(self, step_number: int, step_name: str, description: str, 
                     details: Optional[Dict[str, Any]] = None):
        """Mark the completion of a pipeline step."""
        duration_ms = None
        if step_name in self.step_timings:
            duration_ms = int((time.time() - self.step_timings[step_name]) * 1000)
        
        event = StreamEvent(
            step_number=step_number,
            event_name=f"{step_name}_complete",
            name=description,
            status="complete",
            timestamp=datetime.utcnow().isoformat() + "Z",
            duration_ms=duration_ms,
            details=details or {},
        )
        self._emit(event)
        return event
    
    def error_step(self, step_number: int, step_name: str, description: str, error_msg: str):
        """Mark a pipeline step as errored."""
        duration_ms = None
        if step_name in self.step_timings:
            duration_ms = int((time.time() - self.step_timings[step_name]) * 1000)
        
        event = StreamEvent(
            step_number=step_number,
            event_name=f"{step_name}_error",
            name=description,
            status="error",
            timestamp=datetime.utcnow().isoformat() + "Z",
            duration_ms=duration_ms,
            error=error_msg,
        )
        self._emit(event)
        return event
    
    def emit_final_result(self, result_data: Dict[str, Any]):
        """Emit the final result event."""
        event = StreamEvent(
            step_number=999,  # Special number for final
            event_name="response_complete",
            name="Response Complete",
            status="complete",
            timestamp=datetime.utcnow().isoformat() + "Z",
            details=result_data,
        )
        self._emit(event)
        return event
    
    def get_events_as_sse(self) -> str:
        """Convert all events to SSE format for streaming."""
        sse_lines = []
        for event in self.events:
            sse_lines.append(f"event: {event.event_name}")
            sse_lines.append(f"data: {event.to_json()}")
            sse_lines.append("")  # Blank line between events
        return "\n".join(sse_lines)


# Global instance
streaming_emitter = StreamingEventEmitter()
