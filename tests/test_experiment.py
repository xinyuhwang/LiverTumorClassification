from __future__ import annotations

from liver_tumor.experiment import make_run_id


def test_run_id_contains_phase_round_task_and_git_sha() -> None:
    run_id = make_run_id("P01", "R02", "Data Audit", "abcdef123456")

    assert run_id.startswith("p01-r02-data-audit-")
    assert run_id.endswith("abcdef12")
