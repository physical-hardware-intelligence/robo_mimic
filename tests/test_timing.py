"""Per-stage timing. Pure, so it is tested properly rather than eyeballed."""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robo_mimic.timing import Budget, Clock, StageStats

STAGES = ("capture", "detect", "solve")


def test_an_unknown_stage_is_refused() -> None:
    """A typo would otherwise vanish into the dict and never be reported."""
    budget = Budget(STAGES)
    with pytest.raises(KeyError, match="unknown stage"):
        budget.add("capature", 1.0)


def test_stages_report_in_declared_order_not_insertion_order() -> None:
    """The live readout and the printed table must agree, always."""
    budget = Budget(STAGES)
    for stage in reversed(STAGES):
        budget.add(stage, 1.0)
    assert [row.stage for row in budget.table()] == list(STAGES)


def test_a_stage_that_never_ran_is_omitted_not_zeroed() -> None:
    budget = Budget(STAGES)
    budget.add("detect", 5.0)
    assert [row.stage for row in budget.table()] == ["detect"]
    assert budget.mean_ms("capture") == 0.0


def test_the_window_forgets_old_samples() -> None:
    budget = Budget(STAGES, window=10)
    for _ in range(10):
        budget.add("solve", 100.0)
    for _ in range(10):
        budget.add("solve", 1.0)
    assert budget.mean_ms("solve") == pytest.approx(1.0)
    assert budget.table()[0].samples == 10


def test_total_is_the_sum_of_the_means() -> None:
    budget = Budget(STAGES)
    budget.add("capture", 1.0)
    budget.add("detect", 2.0)
    budget.add("detect", 4.0)
    budget.add("solve", 0.5)
    assert budget.total_ms() == pytest.approx(1.0 + 3.0 + 0.5)


def test_headroom_goes_negative_when_it_does_not_fit() -> None:
    budget = Budget(STAGES)
    budget.add("detect", 40.0)
    assert budget.headroom_ms(31.0) < 0.0
    assert budget.headroom_ms(10.0) > 0.0
    assert budget.headroom_ms(25.0) == pytest.approx(0.0, abs=1e-9)


def test_stats_are_ordered_mean_le_p95_le_max() -> None:
    budget = Budget(STAGES)
    for value in (1.0, 2.0, 3.0, 50.0):
        budget.add("solve", value)
    row = budget.table()[0]
    assert row.mean_ms <= row.p95_ms <= row.max_ms
    assert row.max_ms == 50.0


# --- the Clock ----------------------------------------------------------------
def test_the_clock_records_roughly_the_right_duration() -> None:
    budget = Budget(STAGES)
    with budget.measure("solve"):
        time.sleep(0.01)
    measured = budget.mean_ms("solve")
    assert 8.0 < measured < 60.0, f"{measured:.2f} ms for a 10 ms sleep"


def test_the_clock_records_even_when_the_body_raises() -> None:
    """A stage that failed still consumed time. Dropping the sample would
    silently flatter the budget, and the failing stage is the one you most want
    to see in the table."""
    budget = Budget(STAGES)
    with pytest.raises(ValueError, match="boom"), budget.measure("detect"):
        raise ValueError("boom")
    assert budget.table()[0].stage == "detect"
    assert budget.table()[0].samples == 1


def test_clocks_nest_without_interfering() -> None:
    budget = Budget(STAGES)
    with budget.measure("capture"), budget.measure("solve"):
        pass
    assert {row.stage for row in budget.table()} == {"capture", "solve"}


@given(values=st.lists(st.floats(0.0, 1000.0), min_size=1, max_size=200))
@settings(max_examples=200, deadline=None)
def test_stats_never_lie_about_their_inputs(values: list[float]) -> None:
    budget = Budget(STAGES, window=1000)
    for value in values:
        budget.add("solve", value)
    row = budget.table()[0]
    assert isinstance(row, StageStats)
    assert row.samples == len(values)
    # Tolerance, because the mean is a sum-then-divide: hypothesis found
    # mean([1.69, 1.69, 1.69]) == 1.6900000000000002, which exceeds the max by
    # 2e-16. The bound is mathematically exact and numerically is not.
    slack = 1e-9 * max(1.0, abs(max(values)))
    assert min(values) - slack <= row.mean_ms <= max(values) + slack
    assert row.max_ms == max(values)
    assert row.p95_ms <= max(values) + slack


def test_clock_is_reusable_via_measure() -> None:
    budget = Budget(STAGES)
    for _ in range(3):
        with budget.measure("solve"):
            pass
    assert budget.table()[0].samples == 3


def test_a_bare_clock_still_works() -> None:
    budget = Budget(STAGES)
    with Clock(budget, "capture"):
        pass
    assert budget.table()[0].stage == "capture"
