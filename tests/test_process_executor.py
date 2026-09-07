"""The UMAP fit runs in another PROCESS, because a thread cannot keep numba's
JIT off the frame loop: the compile runs in Python and holds the GIL.
Measured in the real app, a refit on a worker thread stalled 233 frames over
45 seconds. See CLAUDE.md.
"""
import os
import pickle
import time

import numpy as np

from utilities.process_executor import ProcessExecutor


def _boom():
    raise ValueError("no")


def _wait(job, seconds=30.0):
    t0 = time.time()
    while not job.done():
        assert time.time() - t0 < seconds, "job never finished"
        time.sleep(0.02)


def test_a_job_runs_in_another_process_and_its_result_comes_back():
    ex = ProcessExecutor()
    try:
        job = ex.submit(os.getpid)
        _wait(job)
        assert job.result() != os.getpid()
    finally:
        ex.shutdown(wait=True)


def test_the_child_is_reused_so_a_jit_paid_once_stays_paid():
    ex = ProcessExecutor()
    try:
        a = ex.submit(os.getpid); _wait(a)
        b = ex.submit(os.getpid); _wait(b)
        assert a.result() == b.result()
    finally:
        ex.shutdown(wait=True)


def test_a_raising_job_surfaces_on_result_and_leaves_the_child_usable():
    ex = ProcessExecutor()
    try:
        job = ex.submit(_boom)
        _wait(job)
        try:
            job.result()
        except RuntimeError as exc:
            assert "ValueError: no" in str(exc)
        else:
            raise AssertionError("the exception was swallowed")
        again = ex.submit(os.getpid)
        _wait(again)
        assert again.result() != os.getpid()
    finally:
        ex.shutdown(wait=True)


def test_shutdown_kills_the_child():
    ex = ProcessExecutor()
    job = ex.submit(os.getpid)
    _wait(job)
    proc = ex._proc
    ex.shutdown(wait=True)
    assert not proc.is_alive()
    assert not ex.alive


def test_submit_returns_before_a_fresh_child_has_read_the_job():
    """The pipe send waits for the child; the frame loop must not."""
    ex = ProcessExecutor(prime=("json",))
    try:
        payload = np.zeros((6000, 512), np.float32)      # a real archive's size
        t = time.perf_counter()
        job = ex.submit(np.shape, payload)
        assert time.perf_counter() - t < 0.2
        _wait(job)
        assert job.result() == (6000, 512)
    finally:
        ex.shutdown(wait=True)


def test_the_map_fit_is_a_module_level_function_that_pickles():
    """What crosses the pipe must pickle by reference; a closure cannot."""
    from services.map_layout import make_layout
    from services.map_layout_service import fit_layout

    pickle.dumps((fit_layout, (make_layout, "pca", np.zeros((4, 3), np.float32),
                               [], None)))


def test_the_service_fits_through_a_process_by_default():
    from services.map_layout_service import MapLayoutService

    svc = MapLayoutService()
    assert isinstance(svc.executor, ProcessExecutor)
    assert not svc.executor.alive          # spawned on first use, not here
    svc.shutdown()
