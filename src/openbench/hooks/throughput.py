"""Live tokens/sec tracking via Inspect AI's hooks API.

Registers a hook that receives every successful model generation
(`on_model_usage`), records it in the shared throughput tracker, and
publishes a `tok/s` counter through `inspect_ai.util.display_counter`
(the documented public API for live display counters).

Since counters are displayed by every built-in display mode (textual,
rich) via `display_counter`, no monkey patching is required.
"""

from inspect_ai.hooks import Hooks, ModelUsageData, TaskEnd, hooks

from openbench.metrics.throughput import format_tps, get_tracker


@hooks(
    name="openbench-tps",
    description="Track generation throughput and publish tok/s to the live display",
)
class ThroughputHook(Hooks):
    """Record model generations and mirror throughput into the display."""

    def __init__(self) -> None:
        self._tracker = get_tracker()

    async def on_model_usage(self, data: ModelUsageData) -> None:
        """Handle inspect's per-generation model_usage event."""
        output_tokens = data.usage.output_tokens or 0
        if output_tokens > 0:
            self._tracker.record(
                output_tokens=output_tokens,
                call_duration=data.call_duration,
            )

        # Publish the instantaneous rate to inspect's live display counters.
        self._publish_counter("tok/s", self._tracker.instantaneous_tps())

    async def on_task_end(self, data: TaskEnd) -> None:
        """Publish the run's average at task end so it appears in the footer."""
        self._publish_counter("tok/s avg", self._tracker.average_tps())

    @staticmethod
    def _publish_counter(caption: str, value: float) -> None:
        try:
            from inspect_ai.util import display_counter

            display_counter(caption, format_tps(value))
        except Exception:
            # Display may be unavailable in headless/log modes; non-fatal.
            pass
