import pytest

from hydrostate.data.site.smoke import collect_until_auditable


def test_empty_first_batch_continues_then_audits():
    report = dict(attempted=12, successful=0)
    batches, audits = [], []

    def collect(n):
        batches.append(n)
        report["attempted"] += n
        report["successful"] = int(len(batches) == 2)

    assert collect_until_auditable(lambda: report, collect, lambda: audits.append(True)) == "passed"
    assert batches == [24, 24]
    assert audits == [True]


def test_budget_is_total_across_restarts_and_empty_is_not_audited():
    report = dict(attempted=235, successful=0)

    def collect(n):
        assert n == 5
        report["attempted"] += n

    def audit():
        pytest.fail("Empty data must not be audited")

    assert collect_until_auditable(lambda: report, collect, audit) == "smoke_budget_exhausted"
    assert collect_until_auditable(lambda: report, collect, audit) == "smoke_budget_exhausted"


def test_exhausted_pool_and_real_audit_failure():
    assert (
        collect_until_auditable(lambda: {}, lambda n: None, lambda: None) == "smoke_pool_exhausted"
    )

    def audit():
        raise ValueError("Actual invalid sample")

    with pytest.raises(ValueError, match="Actual invalid"):
        collect_until_auditable(lambda: dict(successful=1), lambda n: None, audit)
