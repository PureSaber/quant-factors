from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cross_interpreter_constraint_is_applied_to_generated_lock() -> None:
    constraints = (ROOT / "requirements-constraints.txt").read_text(encoding="utf-8")
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "rpds-py<0.31" in constraints
    assert "--constraint=requirements-constraints.txt" in lock
    assert "rpds-py==0.30.0" in lock
