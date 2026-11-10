"""Bounded smoke collection before auditing nonempty station samples."""


def collect_until_auditable(read_report, collect_batch, audit, *, budget=240, batch_size=24):
    """Resume committed attempts; never audit empty data or loop without progress.

    The budget bounds total smoke attempts across restarts, not each invocation.
    Exhaustion is an explicit outcome; collection/audit errors propagate.
    """
    while True:
        report = read_report()
        attempted = report.get("attempted", 0)
        if report.get("successful", 0) > 0:
            audit()
            return "passed"
        if attempted >= budget:
            return "smoke_budget_exhausted"
        collect_batch(min(batch_size, budget - attempted))
        if read_report().get("attempted", 0) <= attempted:
            return "smoke_pool_exhausted"
