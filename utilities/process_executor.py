"""A one-job-at-a-time executor backed by a persistent child PROCESS.

A worker thread cannot keep a numba compile off the frame loop: the JIT runs
in Python and holds the GIL. A process can. The child is spawned on first use,
serves one job after another so a JIT paid once stays paid, and is terminated
by shutdown(). Jobs and results cross a pipe by pickle, so a job must be a
module-level function and its arguments plain data.
"""
from __future__ import annotations

import multiprocessing as mp
import threading


def _serve(conn, prime) -> None:
    for name in prime:
        try:
            __import__(name)
        except Exception:               # noqa: BLE001 - a job will say why
            pass
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        fn, args = msg
        try:
            out = (True, fn(*args))
        except BaseException as exc:    # noqa: BLE001
            # Not every exception pickles; the message always does.
            out = (False, RuntimeError(f"{type(exc).__name__}: {exc}"))
        try:
            conn.send(out)
        except (EOFError, OSError, ValueError):
            return


def _send(conn, msg) -> None:
    try:
        conn.send(msg)
    except (EOFError, OSError, ValueError):
        pass                            # shut down under it; the job reports


class ProcessJob:
    """What submit() hands back: future-shaped, polled from the frame loop."""

    def __init__(self, conn, proc):
        self._conn = conn
        self._proc = proc
        self._got = False
        self._ok = False
        self._val = None

    def _take(self, block: bool) -> None:
        if self._got:
            return
        try:
            if not block and not self._conn.poll():
                if self._proc.is_alive():
                    return
            self._ok, self._val = self._conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            self._ok, self._val = False, RuntimeError("the worker process died")
        self._got = True

    def done(self) -> bool:
        self._take(block=False)
        return self._got

    def result(self):
        self._take(block=True)
        if not self._ok:
            raise self._val
        return self._val

    def cancel(self) -> bool:
        return False


class ProcessExecutor:
    """submit(fn, *args) -> ProcessJob. One job in flight at a time.

    `prime` names modules the child imports before its first job, for a
    package whose import order matters.
    """

    def __init__(self, prime: tuple[str, ...] = ()):
        self._prime = tuple(prime)
        self._proc = None
        self._conn = None
        self._job: ProcessJob | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _ensure(self) -> None:
        if self.alive:
            return
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_serve, args=(child, self._prime),
                           daemon=True, name="fluoddity-worker")
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent

    def submit(self, fn, *args) -> ProcessJob:
        """Never blocks the caller: a pipe send waits for the child to read,
        and a freshly spawned child is still importing."""
        if self._job is not None and not self._job.done():
            raise RuntimeError("one job at a time")
        self._ensure()
        conn = self._conn
        threading.Thread(target=_send, args=(conn, (fn, args)),
                         daemon=True, name="fluoddity-worker-send").start()
        self._job = ProcessJob(conn, self._proc)
        return self._job

    def shutdown(self, wait: bool = False, cancel_futures: bool = False) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = self._job = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None and proc.is_alive():
            proc.terminate()
            if wait:
                proc.join(5)
