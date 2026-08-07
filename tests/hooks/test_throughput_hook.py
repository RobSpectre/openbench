"""Tests for `openbench.hooks.throughput` (Inspect AI hooks integration)."""

from __future__ import annotations

import asyncio

import pytest
from inspect_ai.hooks import Hooks, ModelUsageData
from inspect_ai.hooks._hooks import TaskEnd
from inspect_ai.model._model_output import ModelUsage

from openbench.metrics.throughput import get_tracker
from openbench.hooks.throughput import ThroughputHook


@pytest.fixture(autouse=True)
def reset_tracker():
    tracker = get_tracker()
    tracker.reset()
    yield
    tracker.reset()


def wait_single(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_throughput_hook_is_subclass_of_hooks():
    assert issubclass(ThroughputHook, Hooks)


def test_hook_registered_via_decorator():
    """Importing the module must register the hook so inspect emits events."""
    from inspect_ai.hooks._hooks import get_all_hooks

    hook_types = [h.__class__.__name__ for h in get_all_hooks()]
    assert "ThroughputHook" in hook_types


def test_on_model_usage_records_and_shows_counter():
    hook = ThroughputHook()
    usage = ModelUsage(input_tokens=100, output_tokens=50, total_tokens=150)

    published = []

    def fake_display_counter(caption: str, value: str) -> None:
        published.append((caption, value))

    with pytest.MonkeyPatch.context() as m:
        m.setattr("inspect_ai.util.display_counter", fake_display_counter)
        data = ModelUsageData(model_name="mock/model", usage=usage, call_duration=2.5)
        asyncio.run(hook.on_model_usage(data))

    assert get_tracker().total_output_tokens() == 50
    assert get_tracker().total_calls() == 1
    # average = total tokens / duration = 50 / 2.5 = 20 tok/s
    assert get_tracker().average_tps() == 20.0
    assert published == [("tok/s", "20.0")]  # one event: 50 tokens / 2.5s = 20 tok/s


def test_on_model_usage_zero_tokens_no_reset_but_still_shows_zero():
    hook = ThroughputHook()
    usage = ModelUsage(input_tokens=100, output_tokens=0, total_tokens=100)

    published = []
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "inspect_ai.util.display_counter", lambda c, v: published.append((c, v))
        )
        data = ModelUsageData(model_name="mock/model", usage=usage, call_duration=1.0)
        asyncio.run(hook.on_model_usage(data))

    # no tokens recorded, but counter still displayed
    assert get_tracker().total_output_tokens() == 0
    assert get_tracker().total_calls() == 0
    # display happened even though nothing was recorded
    assert len(published) == 1


def test_on_task_end_shows_average_rate():
    hook = ThroughputHook()
    usage = ModelUsage(input_tokens=100, output_tokens=200, total_tokens=300)

    published = []

    def fake_display_counter(caption: str, value: str) -> None:
        published.append((caption, value))

    with pytest.MonkeyPatch.context() as m:
        m.setattr("inspect_ai.util.display_counter", fake_display_counter)
        data = ModelUsageData(model_name="mock/model", usage=usage, call_duration=10.0)
        asyncio.run(hook.on_model_usage(data))

        asyncio.run(
            hook.on_task_end(
                TaskEnd(run_id="r", eval_set_id="s", eval_id="e", log=None)  # type: ignore[arg-type]
            )
        )

    assert ("tok/s avg", "20.0") in published
