"""Every state field is either undoable or explicitly excluded, with a reason.

Cases come from __dataclass_fields__ rather than a hand-written list. A
hand-written list is what let the previous 'declared but never read' defects
through: the field is added, nothing references it, and no test fails.
"""
import pytest

from state import sim_state, preferences_state

CONTAINERS = [
    pytest.param(sim_state.SimState, sim_state.UNDOABLE_FIELDS,
                 sim_state.NOT_UNDOABLE, id="SimState"),
    pytest.param(preferences_state.PreferencesState,
                 preferences_state.UNDOABLE_FIELDS,
                 preferences_state.NOT_UNDOABLE, id="PreferencesState"),
]


@pytest.mark.parametrize("cls,undoable,excluded", CONTAINERS)
def test_every_field_is_classified(cls, undoable, excluded):
    fields = set(cls.__dataclass_fields__)
    classified = set(undoable) | set(excluded)
    assert fields - classified == set(), (
        "unclassified field(s) - add to UNDOABLE_FIELDS or NOT_UNDOABLE")


@pytest.mark.parametrize("cls,undoable,excluded", CONTAINERS)
def test_no_field_is_classified_twice(cls, undoable, excluded):
    assert set(undoable) & set(excluded) == set()


@pytest.mark.parametrize("cls,undoable,excluded", CONTAINERS)
def test_no_phantom_names(cls, undoable, excluded):
    fields = set(cls.__dataclass_fields__)
    assert set(undoable) - fields == set(), "UNDOABLE_FIELDS names a dead field"
    assert set(excluded) - fields == set(), "NOT_UNDOABLE names a dead field"


@pytest.mark.parametrize("cls,undoable,excluded", CONTAINERS)
def test_every_exclusion_carries_a_reason(cls, undoable, excluded):
    for name, reason in excluded.items():
        assert isinstance(reason, str) and reason.strip(), name
