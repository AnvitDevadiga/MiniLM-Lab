def test_evaluation_script_exists() -> None:
    from pathlib import Path

    assert Path("scripts/evaluate.py").exists()
