"""Line-oriented progress with heartbeats for non-interactive Slurm logs."""
from threading import Event, Lock, Thread
import time


class StageProgress:
    def __init__(self, stage, total, *, context="", every=10, interval=30.0):
        self.stage, self.total, self.context = stage, total, context or ""
        self.every, self.interval = every, interval
        self.completed = self.unscored = self.unverified = 0
        self.started = time.monotonic()
        self._stop = Event()
        self._lock = Lock()
        self._thread = None

    def _emit(self, status):
        with self._lock:
            elapsed = time.monotonic() - self.started
            eta = f"{elapsed * (self.total - self.completed) / self.completed:.1f}s" if self.completed else "unknown"
            print(f"[hierarchical-rema][progress] {self.context} stage={self.stage} "
                  f"status={status} completed={self.completed}/{self.total} "
                  f"unscored={self.unscored} unverified={self.unverified} "
                  f"elapsed={elapsed:.1f}s eta={eta}", flush=True)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self._emit("running")

    def __enter__(self):
        self._emit("start")
        self._thread = Thread(target=self._heartbeat, daemon=True, name="rema-progress")
        self._thread.start()
        return self

    def advance(self, *, unscored=False, unverified=False):
        with self._lock:
            self.completed += 1
            self.unscored += int(unscored)
            self.unverified += int(unverified)
        if self.every and self.completed % self.every == 0 and self.completed < self.total:
            self._emit("running")

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self._emit("failed" if exc_type else "complete")
