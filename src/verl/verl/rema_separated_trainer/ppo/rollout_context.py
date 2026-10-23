"""Manage physical rollout pools, which can be shared by several logical roles."""

from contextlib import contextmanager
import logging


@contextmanager
def generation_context(worker_groups, batch):
    groups = {id(group): group for group in worker_groups}.values()
    entered = []
    failed = False
    try:
        for group in groups:
            # Include partially entered groups in cleanup after an RPC failure.
            entered.append(group)
            group.enter_generate_context(batch)
        yield
    except BaseException:
        failed = True
        raise
    finally:
        cleanup_error = None
        for group in reversed(entered):
            try:
                group.exit_generate_context(batch)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
                logging.getLogger(__name__).exception('Could not exit rollout context')
        if cleanup_error is not None and not failed:
            raise cleanup_error


def frozen_pool_flags(rollout_roles, train_roles, decomposer_role, *, enabled=True):
    """Only permanently frozen pools qualify, never a temporarily inactive role."""
    if not enabled:
        return False, False
    pools = ({decomposer_role}, set(rollout_roles) - {decomposer_role})
    return tuple(bool(roles) and not roles.intersection(train_roles) for roles in pools)
