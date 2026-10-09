"""Quiet live feature progress without changing the checkpoint-hashed builder.

Only logging bindings are wrapped. Context-local suppression leaves concurrent
training/audit calls and all numerical feature construction unchanged.
"""
import builtins
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

from . import features

_quiet = ContextVar("graphprm_quiet_feature_progress", default=False)
_install_lock = Lock()
_installed = False


def _install():
    global _installed
    with _install_lock:
        if _installed:
            return
        original_tqdm = features.tqdm
        original_print = getattr(features, "print", builtins.print)

        def progress_bar(*args, **kwargs):
            if _quiet.get():
                kwargs["disable"] = True
            return original_tqdm(*args, **kwargs)

        def progress_print(*args, **kwargs):
            if _quiet.get() and args and str(args[0]).startswith("[features] graphs="):
                return
            return original_print(*args, **kwargs)

        features.tqdm = progress_bar
        features.print = progress_print
        _installed = True


@contextmanager
def feature_progress(*, visible=False):
    _install()
    token = _quiet.set(not visible)
    try:
        yield
    finally:
        _quiet.reset(token)

