"""Scoring runs off the frame loop.

CLIP is 92-97% of a generation's main-thread cost - measured 2026-08-10 at
grid 8 with 6 snapshots: 1.1 s at 1 view and 4.2 s at 3, against ~50-130 ms for
descriptor, admission, thumbnails, refresh and prune put together. Blocking the
frame on it froze the app once a generation for the whole of archive growth.

The split is deliberately narrow: only the driver's precompute() leaves the
main thread. Everything after it mutates the archive, which the UI reads every
frame to draw the gallery, the map and the status, so moving THAT off-thread
would need a lock around every one of those reads.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from services.auto_tournament_service import Action, AutoTournamentService, Phase
from services.tournament_service import TournamentService


class SlowScorer:
    """A scorer that blocks until released, so 'is it off the main thread' is
    an assertion rather than a stopwatch reading."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread_names = []
        self.calls = 0

    def set_prompt(self, text, distractors=None):
        pass

    def score(self, images):
        self.calls += 1
        self.thread_names.append(threading.current_thread().name)
        self.entered.set()
        self.release.wait(5.0)
        return np.linspace(0.1, 0.6, len(images)).astype(np.float32)


def make(grid=2, scorer=None):
    ts = TournamentService(grid=grid)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=scorer or SlowScorer(), logger=None)
    svc.configure(steps_per_gen=20, snapshots_per_gen=1, sim_steps_per_frame=20)
    return svc, ts


def crops(svc, value=128):
    return np.full((svc.tournament.tiles, 224, 224, 3), value, dtype=np.uint8)


def run_to_score(svc, limit=200):
    """Drive update() up to the first SCORE action, without taking it."""
    svc.start("coral")
    for _ in range(limit):
        a = svc.update()
        if a is Action.CAPTURE:
            svc.submit_frames(crops(svc))
        elif a is Action.SCORE:
            return
    raise AssertionError("never reached SCORE")


def finish(svc, limit=500):
    """Poll SCORE until a fitness comes back, as main.py does."""
    for _ in range(limit):
        fit = svc.score_and_tell()
        if fit is not None:
            return fit
        time.sleep(0.001)
        assert svc.update() is Action.SCORE, "should stay in SCORE while pending"
    raise AssertionError("scoring never completed")


def test_the_first_score_frame_returns_without_scoring():
    """The whole point: the frame comes back immediately, so the UI keeps
    drawing while CLIP runs."""
    svc, _ = make()
    run_to_score(svc)
    assert svc.score_and_tell() is None
    assert svc.generation == 0, "the generation must not have advanced"
    svc.driver.scorer.release.set()
    finish(svc)
    assert svc.generation == 1


def test_clip_runs_on_a_worker_thread():
    svc, _ = make()
    run_to_score(svc)
    svc.score_and_tell()
    assert svc.driver.scorer.entered.wait(5.0), "the worker never started"
    main = threading.current_thread().name
    assert svc.driver.scorer.thread_names[0] != main
    svc.driver.scorer.release.set()
    finish(svc)


def test_update_keeps_returning_score_until_it_lands():
    """No new action and no new phase: the pending state is the SCORE phase
    with a future outstanding, which is why nothing else had to change."""
    svc, _ = make()
    run_to_score(svc)
    svc.score_and_tell()
    for _ in range(5):
        assert svc.update() is Action.SCORE
        assert svc.phase is Phase.SCORE
    svc.driver.scorer.release.set()
    finish(svc)


def test_a_pending_pass_is_dropped_when_the_generation_is_aborted():
    """abort_generation clears the frame buffer. A result computed from it
    describes a rollout that is no longer going to be scored, and the buffer
    it came from is gone."""
    svc, _ = make()
    run_to_score(svc)
    svc.score_and_tell()
    assert svc.driver.scorer.entered.wait(5.0)
    svc.abort_generation()
    assert svc._pre_future is None
    svc.driver.scorer.release.set()
    # The next score submits a FRESH pass rather than reusing the dropped one.
    calls = svc.driver.scorer.calls
    svc.driver.scorer.release.set()
    run_to_score(svc)
    finish(svc)
    assert svc.driver.scorer.calls > calls


def test_reset_drops_a_pending_pass():
    svc, _ = make()
    run_to_score(svc)
    svc.score_and_tell()
    assert svc.driver.scorer.entered.wait(5.0)
    svc.reset()
    assert svc._pre_future is None
    svc.driver.scorer.release.set()
    assert svc.generation == 0


def test_the_worker_does_not_see_the_buffer_cleared_underneath_it():
    """submit() takes a COPY of the list. Passing self._buffer itself would let
    abort_generation().clear() empty it inside the worker, which scores zero
    snapshots and returns a fitness for nothing."""
    svc, _ = make()
    run_to_score(svc)
    n_before = len(svc._buffer)
    svc.score_and_tell()
    assert svc.driver.scorer.entered.wait(5.0)
    svc._buffer.clear()
    svc.driver.scorer.release.set()
    time.sleep(0.05)
    assert svc.driver.scorer.calls == n_before


def test_the_worker_error_surfaces_on_the_main_thread():
    """A future that raised must not be swallowed - a silently dead search
    that keeps returning None would look exactly like a slow one."""

    class Broken:
        def set_prompt(self, text, distractors=None):
            pass

        def score(self, images):
            raise RuntimeError("clip exploded")

    svc, _ = make(scorer=Broken())
    run_to_score(svc)
    svc.score_and_tell()
    with pytest.raises(RuntimeError, match="clip exploded"):
        for _ in range(500):
            if svc.score_and_tell() is not None:
                break
            time.sleep(0.001)


def test_synchronous_mode_still_works():
    """tools/ and any caller that is not a frame loop must be able to turn the
    thread off and get a fitness from a single call."""
    scorer = SlowScorer()
    scorer.release.set()
    svc, _ = make(scorer=scorer)
    svc.async_scoring = False
    run_to_score(svc)
    fit = svc.score_and_tell()
    assert fit is not None and len(fit) == 4
    assert scorer.thread_names == [threading.current_thread().name]


def test_close_is_safe_with_nothing_in_flight():
    svc, _ = make()
    svc.close()
    svc.close()
