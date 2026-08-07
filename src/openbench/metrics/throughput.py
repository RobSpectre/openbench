"""Token throughput tracking for live tokens/sec display during evals.

A process-wide singleton records every completed model generation via
inspect_ai's hooks API (`on_model_usage`), keeping a rolling window of
(output_tokens, call_duration) samples plus lifetime totals.

Two rates are exposed:
- instantaneous: aggregated generation rate over a trailing window
- average: lifetime aggregate generation rate (total output tokens /
  total generation time)

"Generation time" uses inspect's `call_duration` (time within a successful
model call, excluding retry waits), so the rate reflects effective decode
speed rather than sample-level wall-clock which includes tool execution,
solver logic, and queueing.

The tracker is thread-safe: model calls complete on many concurrent async
tasks and hook emissions are awaited inline, so a simple lock suffices.
"""

import time
from collections import deque

# Rolling window (seconds) over which the instantaneous rate is computed.
_TPS_WINDOW_SECONDS = 30.0
_TPS_MAX_EVENTS = 2000


class ThroughputTracker:
    """Tracks model generation throughput using inspect hook events."""

    def __init__(self) -> None:
        # Rolling window events: (timestamp, output_tokens, call_duration)
        self._events: deque[tuple[float, int, float]] = deque(maxlen=_TPS_MAX_EVENTS)
        # Lifetime totals
        self._total_output_tokens = 0
        self._total_generation_time = 0.0
        self._total_calls = 0

    def record(
        self,
        output_tokens: int,
        call_duration: float,
        timestamp: float | None = None,
    ) -> None:
        """Record a completed model generation.

        Args:
            output_tokens: Number of tokens generated in this call.
            call_duration: Duration of the model call in seconds (from
                inspect's ModelUsageData.call_duration).
            timestamp: Event time (defaults to time.monotonic()).
        """
        if output_tokens <= 0:
            return

        ts = timestamp if timestamp is not None else time.monotonic()
        duration = max(0.0, call_duration)

        # Mutations happen on async tasks within the event loop; use a lock
        # only if contention becomes an issue (e.g. threaded workers).
        self._events.append((ts, output_tokens, duration))
        self._total_output_tokens += output_tokens
        self._total_generation_time += duration
        self._total_calls += 1

        # Prune events outside the rolling window
        cutoff = ts - _TPS_WINDOW_SECONDS
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def instantaneous_tps(self, now: float | None = None) -> float:
        """Tokens/sec over the trailing window (aggregated generation rate).

        Computed as total output tokens / total generation time within the
        window. With parallel requests, spans overlap and this correctly
        reflects aggregate decode throughput.
        """
        if not self._events:
            return 0.0

        current = now if now is not None else time.monotonic()
        cutoff = current - _TPS_WINDOW_SECONDS

        total_tokens = 0
        total_time = 0.0
        for ts, tokens, duration in self._events:
            if ts >= cutoff:
                total_tokens += tokens
                total_time += duration

        if total_time <= 0:
            return 0.0
        return total_tokens / total_time

    def average_tps(self) -> float:
        """Lifetime aggregate generation rate in output tokens/sec."""
        if self._total_generation_time <= 0:
            return 0.0
        return self._total_output_tokens / self._total_generation_time

    def total_output_tokens(self) -> int:
        """Total output tokens generated across all model calls."""
        return self._total_output_tokens

    def total_calls(self) -> int:
        """Total number of model generations recorded."""
        return self._total_calls

    def reset(self) -> None:
        """Clear all recorded events and lifetime totals."""
        self._events.clear()
        self._total_output_tokens = 0
        self._total_generation_time = 0.0
        self._total_calls = 0


_tracker = ThroughputTracker()


def get_tracker() -> ThroughputTracker:
    """Return the process-wide throughput tracker singleton."""
    return _tracker


def format_tps(tps: float) -> str:
    """Format a tokens/sec rate for display."""
    if tps >= 10_000:
        return f"{tps / 1000:.1f}k"
    if tps >= 1000:
        return f"{tps / 1000:.2f}k"
    return f"{tps:.1f}"
