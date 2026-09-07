"""The per-generation refresh sweep runs on the scoring worker, not in tell().

It is a kNN of a tenth of the archive against all of it: measured 178 ms at
11514 entries, once a generation, on the frame loop. The plan is taken on the
main thread (rows and a COPY of the vectors), scored beside the CLIP pass, and
written back in tell() to the rows that still hold the entries it scored.
"""
import threading

import numpy as np
import pytest

from services.archive import Archive
from tests.test_archive import cand, fresh


def _clustered(n=8, **kw):
    a = fresh(liveness_min=0.0, k=1, **kw)
    for i in range(n):
        a.consider(cand([1.0, 0.05 * i, 0.0, 0.0]), novelty=1.0)
    return a


def test_a_planned_refresh_lands_exactly_where_the_inline_one_would():
    a, b = _clustered(), _clustered()
    plan = a.refresh_plan(3)
    nov = Archive.refresh_score(plan)
    assert a.refresh_apply(plan, nov) == 3
    b.refresh(3)
    assert [e.novelty for e in a.entries] == pytest.approx(
        [e.novelty for e in b.entries])
    assert a._refresh_cursor == b._refresh_cursor


def test_the_plan_holds_a_copy_so_the_worker_never_reads_live_arrays():
    a = _clustered()
    plan = a.refresh_plan(3)
    assert not np.shares_memory(plan.embeddings, a._emb)


def test_a_row_that_changed_hands_between_plan_and_apply_is_skipped():
    a = _clustered(6)
    plan = a.refresh_plan(3)                 # rows 0, 1, 2
    nov = Archive.refresh_score(plan)
    a._remove(1)                             # the last entry moves into row 1
    assert a.entries[1].id != int(plan.ids[1])
    assert a.refresh_apply(plan, nov) == 2
    assert a.entries[1].novelty == 1.0, "row 1 holds another entry now"


def test_an_empty_plan_is_none_and_apply_survives_a_shrunken_archive():
    assert fresh().refresh_plan(5) is None
    a = _clustered(3)
    plan = a.refresh_plan(3)
    nov = Archive.refresh_score(plan)
    a._remove(2)
    a._remove(1)
    assert a.refresh_apply(plan, nov) == 1


# ---- through the driver and the service --------------------------------

def test_tell_applies_the_worker_scored_refresh_instead_of_sweeping_inline():
    from tests.test_imgep_driver import make, moving

    d, arc, ts = make(grid=2, seed_n=1)
    d.refresh_sweep_gens = 1
    for _ in range(3):
        d.tell(d.ask(4), moving(4))
    for e in arc.entries:
        e.novelty = 1.0
    calls = {"inline": 0}
    real = arc.refresh

    def counting(n):
        calls["inline"] += 1
        return real(n)
    arc.refresh = counting

    z = d.ask(4)
    snaps = moving(4)
    d.precompute_plan()
    pre = d.precompute(snaps)
    d.tell(z, snaps, pre)
    assert calls["inline"] == 0
    assert any(e.novelty != 1.0 for e in arc.entries), "the sweep never landed"


def test_the_service_takes_the_plan_on_the_main_thread_before_submitting():
    from tests.test_async_scoring import SlowScorer, finish, make, run_to_score

    scorer = SlowScorer()
    scorer.release.set()
    svc, ts = make(scorer=scorer)
    seen = {}
    real_pre = svc.driver.precompute

    def plan():
        seen["plan_thread"] = threading.current_thread().name

    def pre(snapshots):
        seen["pre_thread"] = threading.current_thread().name
        return real_pre(snapshots)

    svc.driver.precompute_plan = plan
    svc.driver.precompute = pre
    run_to_score(svc)
    finish(svc)
    assert seen["plan_thread"] == "MainThread"
    assert seen["pre_thread"] != "MainThread"
