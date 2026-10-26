"""Choose a shared-worker training role from plans, before observing outcomes."""


def select_present_worker_role(stages_by_index, group_ids, candidate_roles, rng):
    """Sample in proportion to UID coverage, counting 8x4 replicas only once."""
    counts = dict.fromkeys(candidate_roles, 0)
    seen = {}
    for index, stages in stages_by_index.items():
        uid = group_ids[index]
        roles = frozenset(stage[0] for stage in stages if stage[0] in counts)
        if uid in seen:
            if seen[uid] != roles:
                raise ValueError('C3 worker selection requires a shared plan within each UID')
            continue
        seen[uid] = roles
        for role in roles:
            counts[role] += 1
    available = [role for role, count in counts.items() if count]
    total = sum(counts.values())
    selected = (str(rng.choice(available, p=[counts[role] / total for role in available]))
                if available else None)
    return selected, counts
