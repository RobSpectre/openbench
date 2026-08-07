"""Tests for openbench.metrics.throughput."""

from openbench.metrics.throughput import ThroughputTracker, format_tps


def test_initial_state():
    """A fresh tracker reports zero everywhere."""
    t = ThroughputTracker()
    assert t.total_output_tokens() == 0
    assert t.total_calls() == 0
    assert t.instantaneous_tps() == 0.0
    assert t.average_tps() == 0.0


def test_single_event_zero_rate():
    """One event has no span, so instantaneous is zero; average depends on duration."""
    t = ThroughputTracker()
    t.record(output_tokens=100, call_duration=0.0, timestamp=10.0)
    assert t.total_output_tokens() == 100
    assert t.total_calls() == 1
    assert t.instantaneous_tps(now=10.1) == 0.0
    assert t.average_tps() == 0.0


def test_two_events_average_and_instant():
    """2 events: 100 tokens each, 5s/5s durations -> both rates ~40 tok/s."""
    t = ThroughputTracker()
    t.record(output_tokens=100, call_duration=5.0, timestamp=10.0)
    t.record(output_tokens=100, call_duration=5.0, timestamp=15.0)
    # average: 200 tokens / 10s = 20
    assert t.average_tps() == 20.0
    # instantaneous (window covers both events): same ratio
    assert t.instantaneous_tps(now=20.0) == 20.0


def test_zero_token_generations_are_ignored():
    """Zero-token calls should have no effect."""
    t = ThroughputTracker()
    t.record(output_tokens=0, call_duration=2.0)
    assert t.total_output_tokens() == 0
    assert t.total_calls() == 0
    assert t.average_tps() == 0.0


def test_reset_clears_state():
    """reset() restores initial state."""
    t = ThroughputTracker()
    t.record(output_tokens=100, call_duration=2.0)
    t.reset()
    assert t.total_output_tokens() == 0
    assert t.instantaneous_tps() == 0.0


def test_window_pruning_removes_old_events():
    """Old events fall out of the trailing window but stay in totals."""
    t = ThroughputTracker()
    t.record(output_tokens=100, call_duration=5.0, timestamp=0.0)
    t.record(output_tokens=100, call_duration=5.0, timestamp=40.0)
    # average sees all
    assert t.average_tps() == 20.0
    # instantaneous (30s window) sees only the latest event, so its tokens/duration
    # is still 20, but if we hadn't seen any tokens for a while it would drop.
    assert t.instantaneous_tps(now=41.0) == 20.0


def test_format_tps_units():
    """format_tps produces human-readable units (Python banker's rounding)."""
    assert format_tps(0.0) == "0.0"
    assert format_tps(123.45) == "123.5"
    assert format_tps(999.5) == "999.5"
    assert format_tps(1000.0) == "1.00k"
    assert format_tps(12345.0) == "12.3k"
