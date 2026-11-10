"""Maximum hierarchical integer-balanced draws, independent for each water bin."""

from collections import defaultdict, deque

from hydrostate.data.gee.contracts import fingerprint

LEVELS = ("landcover", "climate_class", "year", "month")
POLICY = "within_frequency_hierarchical_integer_balance_v1"


def balanced_draw(rows, *, seed):
    """Return the largest feasible draw with sibling counts differing by at most one.

    Each child admits every integer size from zero through its capacity. Thus a
    parent with n children has capacity n*min(capacity) + count(capacity>min).
    This recursive bound is attainable; no water bin caps another water bin.
    Failed candidates must be removed before calling; an empty branch is recorded
    by the caller, rather than supplied with duplicate or fictitious observations.
    """

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
        minimum = min(child["capacity"] for child in children.values())
        capacity = len(children) * minimum + sum(c["capacity"] > minimum for c in children.values())
        return dict(capacity=capacity, children=children, path=path, level=LEVELS[depth])

    audit, selected = [], []

    def take(node, count):
        assert 0 <= count <= node["capacity"]
        if "ids" in node:
            selected.extend(node["ids"][:count])
            return
        children = node["children"]
        base, remainder = divmod(count, len(children))
        extra = sorted(
            (k for k, c in children.items() if c["capacity"] > base),
            key=lambda k: rank(node["path"] + [k]),
        )[:remainder]
        assert len(extra) == remainder
        counts = {k: base + int(k in extra) for k in children}
        audit.append(
            dict(
                path=node["path"],
                level=node["level"],
                counts=counts,
                maximum_difference=max(counts.values()) - min(counts.values()),
            )
        )
        for key in sorted(children, key=lambda k: rank(node["path"] + [k])):
            take(children[key], counts[key])

    frequencies = defaultdict(list)
    for row in rows:
        frequencies[row["frequency_bin"]].append(row)
    capacities = {}
    for frequency in sorted(frequencies, key=lambda b: (b == 0, b)):
        tree = build(frequencies[frequency], 0, [frequency])
        capacities[frequency] = tree["capacity"]
        take(tree, tree["capacity"])
    return selected, dict(
        policy=POLICY,
        by_frequency=capacities,
        strata=audit,
        max_sibling_difference=max((a["maximum_difference"] for a in audit), default=0),
    )
