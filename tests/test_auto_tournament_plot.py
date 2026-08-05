"""The dual-axis fitness/sigma trace.

Fitness climbs through roughly 0.05-0.5 while sigma decays from ~0.5 toward 0.
On one shared axis the larger range flattens the other curve, so each series is
normalised against its OWN min/max and the real values are shown in the legend.
"""
import pytest

from ui.auto_tournament_window import normalize_series


def test_empty_series_normalises_to_nothing():
    assert normalize_series([]) == []


def test_series_is_mapped_onto_zero_to_one():
    assert normalize_series([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_lowest_point_is_zero_and_highest_is_one():
    out = normalize_series([0.2, 0.9, 0.35, 0.5])
    assert min(out) == 0.0
    assert max(out) == 1.0


def test_a_flat_series_sits_in_the_middle_rather_than_dividing_by_zero():
    """A single generation, or a sigma that has not moved yet, must not blow up."""
    assert normalize_series([0.4, 0.4, 0.4]) == [0.5, 0.5, 0.5]
    assert normalize_series([0.7]) == [0.5]


def test_a_decaying_series_keeps_its_shape():
    """Sigma decays; normalising must not flip it."""
    out = normalize_series([0.5, 0.4, 0.3, 0.2])
    assert out[0] == 1.0
    assert out[-1] == 0.0
    assert out == sorted(out, reverse=True)


def test_non_finite_values_do_not_poison_the_plot():
    out = normalize_series([0.1, float("nan"), 0.5])
    assert all(0.0 <= v <= 1.0 for v in out)
    assert len(out) == 3


@pytest.mark.parametrize("vals", [[1e-9, 2e-9], [1e6, 2e6], [-0.5, 0.5]])
def test_scale_and_sign_do_not_matter(vals):
    assert normalize_series(vals) == [0.0, 1.0]
