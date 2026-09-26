from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDK_COMMIT = "a459b328f79fbc55663d4f0c041d3805ccde21ce"


def test_cross_interpreter_constraint_is_applied_to_generated_lock() -> None:
    constraints = (ROOT / "requirements-constraints.txt").read_text(encoding="utf-8")
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "rpds-py<0.31" in constraints
    assert "--constraint=requirements-constraints.txt" in lock
    assert "rpds-py==0.30.0" in lock


def test_qdk_dependency_is_pinned_consistently() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    expected = f"quant-data-kit.git@{QDK_COMMIT}"
    assert expected in project
    assert expected in lock
    assert QDK_COMMIT in readme
