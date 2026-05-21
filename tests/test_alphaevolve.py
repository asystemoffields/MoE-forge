from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "sonnet-evolve" / "alphaevolve.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("alphaevolve", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["alphaevolve"] = module  # dataclass(slots=True) resolution needs this
    spec.loader.exec_module(module)
    return module


ae = _load_module()


def test_extract_code_handles_fenced_and_bare() -> None:
    fenced = "blah\n```python\ndef group():\n    return 1\n```\ntrailing"
    assert ae.extract_code(fenced) == "def group():\n    return 1\n"
    bare = "def group():\n    return 2"
    assert ae.extract_code(bare) == "def group():\n    return 2\n"
    # First block wins when there are several.
    multi = "```python\nA\n```\n```python\nB\n```"
    assert ae.extract_code(multi).strip() == "A"


def test_last_json_object_ignores_surrounding_noise() -> None:
    text = "warning: foo\n{\"a\": 1}\nmore log\n{\"mean_error\": 0.5, \"ok\": true}\n"
    assert ae._last_json_object(text) == {"mean_error": 0.5, "ok": True}
    assert ae._last_json_object("no json here") is None


def test_last_json_object_returns_top_level_not_nested() -> None:
    # A pretty-printed result with a nested per-item array must return the OUTER object.
    text = '{\n  "mean_error": 0.41,\n  "per_layer": [\n    {"layer": "a", "error": 0.4},\n    {"layer": "b", "error": 0.42}\n  ]\n}\n'
    assert ae._last_json_object(text)["mean_error"] == 0.41


def _task(tmp_path: Path, evaluator: Path, *, direction: str = "min") -> "ae.TaskConfig":
    return ae.TaskConfig(
        name="t",
        evaluator_cmd=[sys.executable, str(evaluator), "--candidate", "{candidate}"],
        score_key="score",
        seed_candidate=tmp_path / "seed.py",
        prompt="evolve a value",
        direction=direction,
    )


def _write_evaluator(tmp_path: Path) -> Path:
    # Dummy evaluator: exec the candidate (which sets `value`), print {"score": value}.
    evaluator = tmp_path / "score.py"
    evaluator.write_text(
        "import argparse, json\n"
        "p = argparse.ArgumentParser(); p.add_argument('--candidate'); a = p.parse_args()\n"
        "ns = {}\n"
        "exec(open(a.candidate).read(), ns)\n"
        "print(json.dumps({'score': ns['value']}))\n",
        encoding="utf-8",
    )
    return evaluator


def test_run_evaluator_parses_score(tmp_path: Path) -> None:
    task = _task(tmp_path, _write_evaluator(tmp_path))
    candidate = tmp_path / "cand.py"
    candidate.write_text("value = 0.25\n", encoding="utf-8")
    score, raw = ae.run_evaluator(candidate, task)
    assert score == pytest.approx(0.25)
    assert raw["score"] == pytest.approx(0.25)


def test_run_evaluator_failed_candidate_scores_worst(tmp_path: Path) -> None:
    task = _task(tmp_path, _write_evaluator(tmp_path))
    candidate = tmp_path / "bad.py"
    candidate.write_text("raise ValueError('boom')\n", encoding="utf-8")  # exec fails in evaluator
    score, raw = ae.run_evaluator(candidate, task)
    assert score == float("inf")  # worst for direction="min"


def test_top_selection_respects_direction(tmp_path: Path) -> None:
    task = _task(tmp_path, _write_evaluator(tmp_path))
    pop = [{"score": 0.5, "id": "a"}, {"score": 0.2, "id": "b"}, {"score": 0.9, "id": "c"}]
    assert [p["id"] for p in ae._top(pop, task, 2)] == ["b", "a"]
    task_max = _task(tmp_path, _write_evaluator(tmp_path), direction="max")
    assert [p["id"] for p in ae._top(pop, task_max, 2)] == ["c", "a"]


def test_load_task_reads_prompt_file(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("the contract", encoding="utf-8")
    (tmp_path / "seed.py").write_text("value = 1\n", encoding="utf-8")
    config = {
        "name": "demo",
        "evaluator_cmd": ["python", "score.py", "--candidate", "{candidate}"],
        "score_key": "score",
        "direction": "min",
        "seed_candidate": "seed.py",
        "prompt_file": "prompt.md",
    }
    config_path = tmp_path / "task.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    task = ae.load_task(config_path)
    assert task.name == "demo"
    assert task.prompt == "the contract"
    assert task.seed_candidate == tmp_path / "seed.py"
    assert task.is_better(0.1, 0.2) is True
