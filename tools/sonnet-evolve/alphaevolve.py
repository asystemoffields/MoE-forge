"""sonnet-evolve — a small, reproducible, task-agnostic AlphaEvolve loop.

A *candidate* is a code file. A *task* is defined by (1) an evaluator command that scores a
candidate file and prints JSON containing a score, (2) a seed candidate, and (3) a generation
prompt describing the contract. The loop uses claude-sonnet to mutate the current best
candidates into new ones, scores each with the evaluator subprocess, keeps the best, and
repeats — logging every candidate, score, and prompt to a run directory for reproducibility.

To adopt it to a new task you write three things (no framework changes):
  1. an evaluator: `<cmd> --candidate <file>` that prints JSON with your score key,
  2. a seed candidate file,
  3. a task JSON config (paths + prompt + score key + direction).

Run:
  python tools/sonnet-evolve/alphaevolve.py --task tools/sonnet-evolve/tasks/carve_grouping.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

MODEL = "claude-sonnet-4-6"
_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class AlphaEvolveError(RuntimeError):
    """Raised when a run cannot proceed (bad config, evaluator missing, etc.)."""


@dataclass(slots=True)
class TaskConfig:
    name: str
    evaluator_cmd: list[str]          # tokens; "{candidate}" is replaced with the candidate path
    score_key: str                    # key in the evaluator's JSON output to optimize
    seed_candidate: Path
    prompt: str                       # generation prompt (the stable, cached task spec)
    direction: str = "min"            # "min" or "max"
    candidate_filename: str = "candidate.py"
    generations: int = 3
    candidates_per_gen: int = 3
    parents_kept: int = 2
    model: str = MODEL
    max_tokens: int = 8000
    temperature: float = 1.0
    evaluator_env: dict[str, str] = field(default_factory=dict)

    @property
    def worst_score(self) -> float:
        return float("inf") if self.direction == "min" else float("-inf")

    def is_better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "min" else a > b


def load_task(path: Path) -> TaskConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    prompt = data.get("prompt")
    if "prompt_file" in data:
        prompt = (base / data["prompt_file"]).read_text(encoding="utf-8")
    if not prompt:
        raise AlphaEvolveError("task config needs 'prompt' or 'prompt_file'")
    seed = Path(data["seed_candidate"])
    if not seed.is_absolute():
        seed = base / seed
    return TaskConfig(
        name=data["name"],
        evaluator_cmd=list(data["evaluator_cmd"]),
        score_key=data["score_key"],
        seed_candidate=seed,
        prompt=prompt,
        direction=data.get("direction", "min"),
        candidate_filename=data.get("candidate_filename", "candidate.py"),
        generations=int(data.get("generations", 3)),
        candidates_per_gen=int(data.get("candidates_per_gen", 3)),
        parents_kept=int(data.get("parents_kept", 2)),
        model=data.get("model", MODEL),
        max_tokens=int(data.get("max_tokens", 8000)),
        temperature=float(data.get("temperature", 1.0)),
        evaluator_env=dict(data.get("evaluator_env", {})),
    )


def extract_code(text: str) -> str:
    """Pull the first python code block from a model response; fall back to the whole text."""
    matches = _CODE_FENCE.findall(text)
    if matches:
        return matches[0].strip() + "\n"
    return text.strip() + "\n"


def run_evaluator(candidate_path: Path, task: TaskConfig) -> tuple[float, dict[str, Any]]:
    """Run the task evaluator on a candidate; return (score, raw_json). Bad candidates score worst."""
    cmd = [token.replace("{candidate}", str(candidate_path)) for token in task.evaluator_cmd]
    env = {**os.environ, **task.evaluator_env}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        completed = subprocess.run(
            cmd, env=env, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise AlphaEvolveError(f"could not run evaluator: {exc}") from exc
    payload = _last_json_object(completed.stdout)
    if payload is None:
        return task.worst_score, {"error": "evaluator produced no JSON", "stderr": completed.stderr[-500:]}
    if "error" in payload or task.score_key not in payload:
        return task.worst_score, payload
    try:
        return float(payload[task.score_key]), payload
    except (TypeError, ValueError):
        return task.worst_score, payload


def generate(task: TaskConfig, parents: list[dict[str, Any]], count: int, *, client: Any) -> list[str]:
    """Ask sonnet for `count` improved candidates given the current best parents."""
    nudges = [
        "Improve on the best parent.",
        "Try a meaningfully different algorithmic approach than the parents.",
        "Combine ideas from two parents, or refine the best one with a local search.",
        "Take a bold, unconventional approach grounded in the task signal.",
    ]
    parent_block = _format_parents(parents, task)
    candidates: list[str] = []
    for index in range(count):
        nudge = nudges[index % len(nudges)]
        user_text = (
            f"{parent_block}\n\n{nudge}\n\n"
            "Return ONLY a single ```python code block defining the required entrypoint. "
            "No prose before or after."
        )
        message = client.messages.create(
            model=task.model,
            max_tokens=task.max_tokens,
            temperature=task.temperature,
            system=[{"type": "text", "text": task.prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        candidates.append(extract_code(text))
    return candidates


def evolve(task: TaskConfig, *, client: Any, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    seed_code = task.seed_candidate.read_text(encoding="utf-8")
    seed_path = run_dir / "gen0" / task.candidate_filename
    _write(seed_path, seed_code)
    seed_score, seed_raw = run_evaluator(seed_path, task)
    population = [{"generation": 0, "id": "seed", "path": str(seed_path), "code": seed_code,
                  "score": seed_score, "raw": seed_raw}]
    history.append({"generation": 0, "id": "seed", "score": seed_score})
    print(f"gen 0  seed                 score={_fmt(seed_score)}")

    for generation in range(1, task.generations + 1):
        parents = _top(population, task, task.parents_kept)
        codes = generate(task, parents, task.candidates_per_gen, client=client)
        for index, code in enumerate(codes):
            candidate_id = f"g{generation}_c{index}"
            path = run_dir / f"gen{generation}" / f"{candidate_id}_{task.candidate_filename}"
            _write(path, code)
            score, raw = run_evaluator(path, task)
            population.append({"generation": generation, "id": candidate_id, "path": str(path),
                               "code": code, "score": score, "raw": raw})
            history.append({"generation": generation, "id": candidate_id, "score": score})
            print(f"gen {generation}  {candidate_id:<18} score={_fmt(score)}")

    best = _top(population, task, 1)[0]
    manifest = {
        "format": "sonnet_evolve_run",
        "task": task.name,
        "model": task.model,
        "direction": task.direction,
        "generations": task.generations,
        "candidates_per_gen": task.candidates_per_gen,
        "seed_score": seed_score,
        "best": {"id": best["id"], "generation": best["generation"], "score": best["score"],
                 "path": best["path"]},
        "history": history,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write(run_dir / "best_candidate.py", best["code"])
    return manifest


def _format_parents(parents: list[dict[str, Any]], task: TaskConfig) -> str:
    lines = [f"Current best candidates ({task.direction}imize the score; "
             f"score key = '{task.score_key}'):"]
    for rank, parent in enumerate(parents):
        lines.append(f"\n## Parent {rank + 1} — score {_fmt(parent['score'])}\n```python\n{parent['code'].strip()}\n```")
    return "\n".join(lines)


def _top(population: list[dict[str, Any]], task: TaskConfig, n: int) -> list[dict[str, Any]]:
    ordered = sorted(population, key=lambda item: item["score"], reverse=(task.direction == "max"))
    return ordered[: max(1, n)]


def _last_json_object(text: str) -> dict[str, Any] | None:
    """Return the last *top-level* JSON object in text (skips past parsed spans so it does
    not descend into nested objects like a result's per-item array)."""
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    index = 0
    length = len(text)
    while index < length:
        if text[index] != "{":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(value, dict):
            found = value
        index += end  # skip the whole parsed object; don't rescan its nested braces
    return found


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fmt(score: float) -> str:
    if score in (float("inf"), float("-inf")):
        return "FAILED"
    return f"{score:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Task JSON config.")
    parser.add_argument("--generations", type=int, help="Override generations.")
    parser.add_argument("--candidates-per-gen", type=int, help="Override candidates per generation.")
    parser.add_argument("--run-dir", type=Path, help="Output directory (default tools/sonnet-evolve/runs/<task>-<ts>).")
    args = parser.parse_args()

    task = load_task(args.task)
    if args.generations is not None:
        task.generations = args.generations
    if args.candidates_per_gen is not None:
        task.candidates_per_gen = args.candidates_per_gen

    run_dir = args.run_dir or (
        args.task.parent.parent / "runs" / f"{task.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    import anthropic

    client = anthropic.Anthropic()
    manifest = evolve(task, client=client, run_dir=run_dir)
    print(f"\nbest: {manifest['best']['id']} score={_fmt(manifest['best']['score'])} "
          f"(seed {_fmt(manifest['seed_score'])}) -> {run_dir}")


if __name__ == "__main__":
    main()
