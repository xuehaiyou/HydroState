"""Fixed-total hierarchical stratified draws with capacity-aware balanced quotas."""

from collections import defaultdict, deque

from hydrostate.data.gee.contracts import fingerprint

LEVELS = ("landcover", "climate_class", "year", "month")
POLICY = "fixed_total_capacity_balanced_hierarchy_v1"


def allocate(capacities, count, order):
    """Integer max-min allocation: exhaust scarce children, equalize the rest."""
    count = min(count, sum(capacities.values()))
    low, high = 0, max(capacities.values(), default=0) + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if sum(min(c, middle) for c in capacities.values()) <= count:
            low = middle
        else:
            high = middle
    result = {k: min(c, low) for k, c in capacities.items()}
    remainder = count - sum(result.values())
    for key in [k for k in order if result[k] < capacities[k]][:remainder]:
        result[key] += 1
    assert sum(result.values()) == count
    return result


def balanced_draw(rows, *, seed, target_total):
    """Draw to a fixed total, redistributing only when a child lacks candidates.

    Nonzero frequency candidates have priority. Within each frequency, max-min
    quotas apply recursively to land cover, climate, year and month. Failed
    candidates are removed before replanning; no quality threshold is relaxed.
    """
    if target_total <= 0:
        raise ValueError("Target count must be positive")

    def rank(value):
        return fingerprint([seed, value])

    def build(items, depth, path):
        if depth == len(LEVELS):
            sites = defaultdict(list)
            for row in items:
                sites[row["physical_group_id"]].append(row["candidate_id"])
            queues = [deque(sorted(sites[s], key=rank)) for s in sorted(sites, key=rank)]
            ordered = []
            while queues:
                for queue in queues:
                    ordered.append(queue.popleft())
                queues = [q for q in queues if q]
            return dict(capacity=len(ordered), ids=ordered, path=path)
        groups = defaultdict(list)
        for row in items:
            groups[row[LEVELS[depth]]].append(row)
        children = {key: build(value, depth + 1, path + [key]) for key, value in groups.items()}
        capacity = sum(child["capacity"] for child in children.values())
        return dict(capacity=capacity, children=children, path=path, level=LEVELS[depth])

    audit, selected = [], []

    def take(node, count):
        assert 0 <= count <= node["capacity"]
        if "ids" in node:
            selected.extend(node["ids"][:count])
            return
        children = node["children"]
        capacity = {k: child["capacity"] for k, child in children.items()}
        order = sorted(children, key=lambda k: rank(node["path"] + [k]))
        counts = allocate(capacity, count, order)
        unsaturated = [counts[k] for k in children if counts[k] < capacity[k]]
        difference = max(unsaturated) - min(unsaturated) if unsaturated else 0
        assert difference <= 1
        assert all(
            counts[k] == capacity[k] or counts[k] >= max(counts.values()) - 1 for k in children
        )
        audit.append(
            dict(
                path=node["path"],
                level=node["level"],
                counts=counts,
                candidate_capacity=capacity,
                unsaturated_difference=difference,
                capacity_limited=[k for k in children if counts[k] == capacity[k]],
            )
        )
        for key in sorted(children, key=lambda k: rank(node["path"] + [k])):
            take(children[key], counts[key])

    frequencies = defaultdict(list)
    for row in rows:
        frequencies[row["frequency_bin"]].append(row)
    trees = {frequency: build(items, 0, [frequency]) for frequency, items in frequencies.items()}
    nonzero = {b: tree["capacity"] for b, tree in trees.items() if b != 0}
    quotas = allocate(nonzero, target_total, sorted(nonzero))
    if 0 in trees:
        quotas[0] = min(trees[0]["capacity"], target_total - sum(quotas.values()))
    for frequency in sorted(trees, key=lambda b: (b == 0, b)):
        take(trees[frequency], quotas[frequency])
    return selected, dict(
        policy=POLICY,
        target_total=target_total,
        by_frequency=quotas,
        strata=audit,
        candidate_shortfall=max(0, target_total - len(selected)),
        max_unsaturated_difference=max((a["unsaturated_difference"] for a in audit), default=0),
    )
