"""Choose a shared-worker training role from plans, before observing outcomes."""


WORKER_FUNCTIONS = ('nonterminal', 'terminal')


def worker_function_ids(role, terminal_roles, action_present, group_ids, worker_roles):
    """Classify actual focal actions: -1 absent/nonworker, 0 local, 1 terminal."""
    if not (len(terminal_roles) == len(action_present) == len(group_ids)):
        raise ValueError('Worker function metadata must have one value per trajectory')
    if role not in worker_roles:
        return [-1] * len(group_ids)
    result, terminals_by_group = [], {}
    for terminal, present, uid in zip(terminal_roles, action_present, group_ids):
        if uid in terminals_by_group and terminals_by_group[uid] != terminal:
            raise ValueError('Worker function selection requires a shared terminal role within each UID')
        terminals_by_group[uid] = terminal
        if present and terminal not in worker_roles:
            raise ValueError(f'Invalid terminal_stage_role for a present worker: {terminal!r}')
        result.append(int(terminal == role) if present else -1)
    return result


def select_present_worker_role(stages_by_index, group_ids, candidate_roles, rng,
                               target_function=None):
    """Sample in proportion to UID coverage, counting 8x4 replicas only once."""
    if target_function is not None and target_function not in WORKER_FUNCTIONS:
        raise ValueError(f'Unknown worker function: {target_function!r}')
    counts = dict.fromkeys(candidate_roles, 0)
    seen = {}
    for index, stages in stages_by_index.items():
        uid = group_ids[index]
        ordered_roles = tuple(stage[0] for stage in stages)
        if uid in seen:
            if seen[uid] != ordered_roles:
                raise ValueError('C3 worker selection requires a shared plan within each UID')
            continue
        seen[uid] = ordered_roles
        if target_function == 'terminal':
            roles = ordered_roles[-1:]
        elif target_function == 'nonterminal':
            roles = ordered_roles[:-1]
        else:
            roles = ordered_roles
        for role in roles:
            if role in counts:
                counts[role] += 1
    available = [role for role, count in counts.items() if count]
    total = sum(counts.values())
    selected = (str(rng.choice(available, p=[counts[role] / total for role in available]))
                if available else None)
    return selected, counts
