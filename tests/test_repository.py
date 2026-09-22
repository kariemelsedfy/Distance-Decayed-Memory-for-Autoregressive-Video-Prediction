from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hpc_credentials_are_gitignored() -> None:
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env.hpc.local" in patterns


def test_active_plans_live_in_docs() -> None:
    assert (ROOT / "docs" / "PROJECT_PLAN.md").is_file()
    assert (ROOT / "docs" / "TRACK_A_PLAN.md").is_file()
    assert not (ROOT / "PROJECT_PLAN.md").exists()
    assert not (ROOT / "TRACK_A_PLAN.md").exists()


def test_extracted_appendices_are_linked() -> None:
    project_plan = (ROOT / "docs" / "PROJECT_PLAN.md").read_text(encoding="utf-8")
    assert "[Read Appendix C: original design proposal](proposal.md)" in project_plan
    assert (
        "[Read Appendix D: Bowdoin HPC reference](hpc/bowdoin-hpc.md)" in project_plan
    )
