import json
from pathlib import Path

import pytest

from minilm_lab.provenance import environment, summarize, write_record


def test_summary_statistics() -> None:
    stats = summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert stats["median"] == 3.0
    assert stats["min"] == 1.0 and stats["max"] == 5.0
    assert stats["iqr"] == 2.0
    assert stats["p95"] == pytest.approx(4.8)


def test_records_are_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"environment": {"git": {"commit": "abcdef0123"}}, "results": [1]}
    first = write_record(tmp_path, "demo", payload)
    assert first.name.endswith("_demo_abcdef0.json")
    assert json.loads(first.read_text())["results"] == [1]
    # Same second, same name: the second write must fail rather than replace.
    monkeypatch.setattr("minilm_lab.provenance.datetime", type("Frozen", (), {"now": staticmethod(lambda tz: _stamp(first))}))
    with pytest.raises(FileExistsError):
        write_record(tmp_path, "demo", {**payload, "results": [2]})
    assert json.loads(first.read_text())["results"] == [1]


def _stamp(path: Path):
    from datetime import datetime, timezone

    return datetime.strptime(path.name.split("_")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def test_environment_records_versions_and_git() -> None:
    info = environment("cpu")
    assert info["torch"] and info["python"] and info["device"] == "cpu"
    assert set(info["git"]) == {"commit", "branch", "dirty"}
