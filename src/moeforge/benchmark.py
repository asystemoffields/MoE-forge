from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COSMOPEDIA_TASKS_REVISION = "38789cac6b7a15047cec96ffd14d4c6dfd9cdf4c"
COSMOPEDIA_TASKS_URL = (
    "https://raw.githubusercontent.com/huggingface/cosmopedia/"
    f"{COSMOPEDIA_TASKS_REVISION}/evaluation/lighteval_tasks.py"
)
COSMOPEDIA_TASKS_V010_PATCH = (
    "python -c \"from pathlib import Path; p=Path('lighteval_tasks.py'); "
    "text=p.read_text(); "
    "text=text.replace('        output_regex=None,\\n',''); "
    "text=text.replace('        frozen=False,\\n',''); "
    "text=text.replace('            output_regex=output_regex,\\n',''); "
    "text=text.replace('            frozen=frozen,\\n',''); "
    "text += '\\nfor _moeforge_task in TASKS_TABLE:\\n"
    "    _moeforge_task.trust_dataset = True\\n"
    "    if isinstance(_moeforge_task.prompt_function, str):\\n"
    "        _moeforge_task.prompt_function = globals()[_moeforge_task.prompt_function]\\n'; "
    "p.write_text(text)\""
)


SMOLLM_BASE_TASKS: list[dict[str, Any]] = [
    {
        "id": "hellaswag",
        "label": "HellaSwag",
        "task_spec": "custom|hellaswag|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": 41.2,
        "priority": "core",
    },
    {
        "id": "arc:easy",
        "label": "ARC Easy",
        "task_spec": "custom|arc:easy|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": None,
        "priority": "core",
    },
    {
        "id": "arc:challenge",
        "label": "ARC Challenge",
        "task_spec": "custom|arc:challenge|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": None,
        "priority": "core",
    },
    {
        "id": "piqa",
        "label": "PIQA",
        "task_spec": "custom|piqa|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": 68.4,
        "priority": "core",
    },
    {
        "id": "commonsense_qa",
        "label": "CommonsenseQA",
        "task_spec": "custom|commonsense_qa|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": 32.7,
        "priority": "core",
    },
    {
        "id": "winogrande",
        "label": "Winogrande",
        "task_spec": "custom|winogrande|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": 51.3,
        "priority": "core",
    },
    {
        "id": "openbookqa",
        "label": "OpenBookQA",
        "task_spec": "custom|openbookqa|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": 34.0,
        "priority": "core",
    },
    {
        "id": "trivia_qa",
        "label": "TriviaQA",
        "task_spec": "custom|trivia_qa|0|1",
        "metric_keys": ["quasi_exact_match_triviaqa", "exact_match", "accuracy"],
        "reported_smollm_135m_score": 4.3,
        "priority": "core",
    },
    {
        "id": "mmlu_cloze",
        "label": "MMLU Cloze",
        "task_spec": "custom|mmlu_cloze:*|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_score": 30.2,
        "priority": "core",
        "aggregate_prefix": "mmlu_cloze:",
    },
    {
        "id": "gsm8k",
        "label": "GSM8K",
        "task_spec": "custom|gsm8k|5|1",
        "metric_keys": ["quasi_exact_match_gsm8k", "exact_match", "accuracy"],
        "reported_smollm_135m_score": 1.0,
        "priority": "diagnostic",
    },
]

SMOLLM_INSTRUCT_TASKS: list[dict[str, Any]] = [
    {
        "id": "ifeval",
        "label": "IFEval",
        "task_spec": "lighteval|ifeval|0|0",
        "metric_keys": [
            "prompt_level_strict_acc",
            "inst_level_strict_acc",
            "average_prompt_inst",
            "accuracy",
        ],
        "reported_smollm_135m_instruct_score": 17.2,
        "priority": "chat_core",
    },
    {
        "id": "mt_bench",
        "label": "MT-Bench",
        "task_spec": "lighteval|mt_bench|0|0",
        "metric_keys": ["score", "average", "judge_score"],
        "reported_smollm_135m_instruct_score": 1.68,
        "priority": "chat_core",
        "requires_judge": True,
        "normalization_denominator": 10.0,
    },
    {
        "id": "hellaswag",
        "label": "HellaSwag",
        "task_spec": "custom|hellaswag|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_instruct_score": 38.9,
        "priority": "supporting",
    },
    {
        "id": "arc:easy",
        "label": "ARC Easy",
        "task_spec": "custom|arc:easy|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_instruct_score": None,
        "priority": "supporting",
    },
    {
        "id": "arc:challenge",
        "label": "ARC Challenge",
        "task_spec": "custom|arc:challenge|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_instruct_score": None,
        "priority": "supporting",
    },
    {
        "id": "piqa",
        "label": "PIQA",
        "task_spec": "custom|piqa|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_instruct_score": 64.0,
        "priority": "supporting",
    },
    {
        "id": "mmlu_cloze",
        "label": "MMLU Cloze",
        "task_spec": "custom|mmlu_cloze:*|0|1",
        "metric_keys": ["loglikelihood_acc_norm_nospace", "acc_norm", "accuracy"],
        "reported_smollm_135m_instruct_score": 28.3,
        "priority": "supporting",
        "aggregate_prefix": "mmlu_cloze:",
    },
    {
        "id": "bbh",
        "label": "BBH",
        "task_spec": "lighteval|bbh|3|0",
        "metric_keys": ["exact_match", "accuracy"],
        "reported_smollm_135m_instruct_score": 25.2,
        "priority": "diagnostic",
    },
    {
        "id": "gsm8k",
        "label": "GSM8K",
        "task_spec": "custom|gsm8k|5|1",
        "metric_keys": ["quasi_exact_match_gsm8k", "exact_match", "accuracy"],
        "reported_smollm_135m_instruct_score": 1.4,
        "priority": "diagnostic",
    },
]

SMOLLM_MMLU_CLOZE_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

SMOLLM_BASE_REFERENCES = [
    {
        "title": "SmolLM model card",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM-135M",
        "note": "Source model summary, intended usage, training data, limitations, and Transformers loading path.",
    },
    {
        "title": "SmolLM blog",
        "url": "https://huggingface.co/blog/smollm",
        "note": "Introduces the model family and frames evaluation around common-sense reasoning and world knowledge.",
    },
    {
        "title": "SmolLM2-135M model card",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M",
        "note": "Reports SmolLM-135M baseline scores and states that evaluations are zero-shot with LightEval.",
    },
    {
        "title": "SmolLM-135M-Instruct model card",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct",
        "note": "Documents chat-template usage, instruct training mix, and intended local chat usage.",
    },
    {
        "title": "Cosmopedia evaluation scripts",
        "url": "https://github.com/huggingface/cosmopedia/tree/main/evaluation",
        "note": "LightEval custom task recipe used by the SmolLM training/evaluation stack.",
    },
    {
        "title": "LightEval docs",
        "url": "https://huggingface.co/docs/lighteval/index",
        "note": "Evaluation framework with Transformers, Accelerate, vLLM, endpoint, and custom-task backends.",
    },
]


@dataclass(slots=True)
class BenchmarkPlanOptions:
    source_model: str
    moe_model: str
    output_path: Path
    suite: str = "smollm-base"
    output_dir: Path = Path("benchmarks/smollm-base")
    max_samples: int = 1000
    batch_size: int = 16
    backend: str = "lighteval"
    custom_tasks_path: str = "lighteval_tasks.py"
    include_gsm8k: bool = True
    use_chat_template: bool = False


@dataclass(slots=True)
class BenchmarkCompareOptions:
    dense_report: Path
    moe_report: Path
    output_path: Path
    suite: str = "smollm-base"
    max_average_drop: float = 0.05
    max_core_drop: float = 0.08
    min_average_retention: float = 0.95
    min_core_retention: float = 0.90


def write_benchmark_plan(options: BenchmarkPlanOptions) -> dict[str, Any]:
    tasks = _suite_tasks(options.suite, include_gsm8k=options.include_gsm8k)
    task_spec = ",".join(_expanded_task_specs(tasks))
    dense_output = options.output_dir / "dense"
    moe_output = options.output_dir / "moe"
    compare_output = options.output_dir / "benchmark-compare.json"
    use_chat_template = options.use_chat_template or options.suite.endswith("instruct")
    agent_notes = [
        "Run dense and MoE benchmark commands with the same harness checkout, task file, max_samples, batch size, dtype, and device class.",
        "For local wrappers, use trust_remote_code=True because MoE Forge exports a custom AutoModel package.",
        "Treat GSM8K as diagnostic for 135M models; the release gate focuses on tasks the source checkpoint reports as meaningful.",
        "Archive raw harness outputs plus benchmark-compare.json with the model card.",
    ]
    if options.suite.endswith("instruct"):
        agent_notes.insert(2, "Use the checkpoint chat template for both dense and MoE runs.")
        agent_notes.insert(3, "MT-Bench requires judge-model configuration; record judge model, seed, temperature, and raw judgments.")

    plan = {
        "format": "moeforge_benchmark_plan",
        "suite": options.suite,
        "backend": options.backend,
        "source_model": options.source_model,
        "moe_model": options.moe_model,
        "output_dir": str(options.output_dir),
        "backend_version_hint": "LightEval v0.10.x for compatibility with the SmolLM/Cosmopedia custom task file.",
        "custom_tasks_revision": COSMOPEDIA_TASKS_REVISION,
        "max_samples": options.max_samples,
        "batch_size": options.batch_size,
        "custom_tasks_path": options.custom_tasks_path,
        "tasks": tasks,
        "task_spec": task_spec,
        "commands": {
            "install": [
                "git clone --branch v0.10.0 https://github.com/huggingface/lighteval.git",
                "cd lighteval && pip install '.[accelerate,quantization,adapters]'",
                f"curl -L {COSMOPEDIA_TASKS_URL} -o lighteval_tasks.py",
                COSMOPEDIA_TASKS_V010_PATCH,
            ],
            "dense": _lighteval_command(
                model=options.source_model,
                output_dir=dense_output,
                custom_tasks_path=options.custom_tasks_path,
                task_spec=task_spec,
                max_samples=options.max_samples,
                batch_size=options.batch_size,
                use_chat_template=use_chat_template,
            ),
            "moe": _lighteval_command(
                model=options.moe_model,
                output_dir=moe_output,
                custom_tasks_path=options.custom_tasks_path,
                task_spec=task_spec,
                max_samples=options.max_samples,
                batch_size=options.batch_size,
                trust_remote_code=True,
                use_chat_template=use_chat_template,
            ),
            "compare": (
                "moe-forge benchmark-compare "
                f"--dense-report {dense_output / 'results.json'} "
                f"--moe-report {moe_output / 'results.json'} "
                f"--output {compare_output}"
            ),
        },
        "release_gate": {
            "comparison_command": "moe-forge benchmark-compare",
            "max_average_drop": 0.05,
            "max_core_drop": 0.08,
            "min_average_retention": 0.95,
            "min_core_retention": 0.90,
            "required_artifact": str(compare_output),
        },
        "agent_notes": agent_notes,
        "references": SMOLLM_BASE_REFERENCES,
    }
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def compare_benchmark_reports(options: BenchmarkCompareOptions) -> dict[str, Any]:
    dense_payload = _read_json(options.dense_report)
    moe_payload = _read_json(options.moe_report)
    tasks = _suite_tasks(options.suite, include_gsm8k=True)
    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for task in tasks:
        dense_score = _extract_task_score(dense_payload, task)
        moe_score = _extract_task_score(moe_payload, task)
        comparable = dense_score is not None and moe_score is not None
        if not comparable:
            warnings.append(f"missing benchmark metric for {task['id']}")
        delta = None if not comparable else moe_score - dense_score
        retention = None
        if comparable and dense_score not in (None, 0.0):
            retention = moe_score / dense_score
        results.append(
            {
                "task": task["id"],
                "label": task["label"],
                "priority": task["priority"],
                "dense_score": dense_score,
                "moe_score": moe_score,
                "absolute_delta": delta,
                "retention": retention,
                "reported_smollm_135m_score": task.get("reported_smollm_135m_score"),
                "reported_smollm_135m_instruct_score": task.get("reported_smollm_135m_instruct_score"),
                "metric_keys": task["metric_keys"],
            }
        )

    comparable = [row for row in results if row["dense_score"] is not None and row["moe_score"] is not None]
    core = [row for row in comparable if row["priority"] in {"core", "chat_core"}]
    average_dense = _average(row["dense_score"] for row in comparable)
    average_moe = _average(row["moe_score"] for row in comparable)
    average_drop = None if average_dense is None or average_moe is None else average_dense - average_moe
    average_retention = None
    if average_dense not in (None, 0.0) and average_moe is not None:
        average_retention = average_moe / average_dense

    worst_core_drop = _max_drop(core)
    worst_core_retention = _min_retention(core)
    checks = [
        _threshold_check(
            name="benchmark.average_drop",
            value=average_drop,
            passed=average_drop is not None and average_drop <= options.max_average_drop,
            message=f"average drop={average_drop}",
            threshold=f"<= {options.max_average_drop}",
        ),
        _threshold_check(
            name="benchmark.average_retention",
            value=average_retention,
            passed=average_retention is not None and average_retention >= options.min_average_retention,
            message=f"average retention={average_retention}",
            threshold=f">= {options.min_average_retention}",
        ),
        _threshold_check(
            name="benchmark.core_drop",
            value=worst_core_drop,
            passed=worst_core_drop is not None and worst_core_drop <= options.max_core_drop,
            message=f"worst core drop={worst_core_drop}",
            threshold=f"<= {options.max_core_drop}",
        ),
        _threshold_check(
            name="benchmark.core_retention",
            value=worst_core_retention,
            passed=worst_core_retention is not None and worst_core_retention >= options.min_core_retention,
            message=f"worst core retention={worst_core_retention}",
            threshold=f">= {options.min_core_retention}",
        ),
    ]
    failed = [check for check in checks if check["status"] == "fail"]
    report = {
        "format": "moeforge_benchmark_comparison",
        "suite": options.suite,
        "status": "passed" if not failed else "blocked",
        "passed": not failed,
        "dense_report": str(options.dense_report),
        "moe_report": str(options.moe_report),
        "task_count": len(results),
        "comparable_task_count": len(comparable),
        "summary": {
            "average_dense_score": average_dense,
            "average_moe_score": average_moe,
            "average_drop": average_drop,
            "average_retention": average_retention,
            "worst_core_drop": worst_core_drop,
            "worst_core_retention": worst_core_retention,
        },
        "thresholds": {
            "max_average_drop": options.max_average_drop,
            "max_core_drop": options.max_core_drop,
            "min_average_retention": options.min_average_retention,
            "min_core_retention": options.min_core_retention,
        },
        "checks": checks,
        "results": results,
        "warnings": warnings,
        "next_actions": _next_actions(failed=failed, warnings=warnings),
        "references": SMOLLM_BASE_REFERENCES,
    }
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _suite_tasks(suite: str, *, include_gsm8k: bool) -> list[dict[str, Any]]:
    if suite == "smollm-base":
        tasks = [dict(task) for task in SMOLLM_BASE_TASKS]
    elif suite == "smollm-instruct":
        tasks = [dict(task) for task in SMOLLM_INSTRUCT_TASKS]
    else:
        raise ValueError(f"unknown benchmark suite: {suite}")
    if not include_gsm8k:
        tasks = [task for task in tasks if task["id"] != "gsm8k"]
    return tasks


def _expanded_task_specs(tasks: list[dict[str, Any]]) -> list[str]:
    specs: list[str] = []
    for task in tasks:
        if task["id"] == "mmlu_cloze":
            specs.extend(f"custom|mmlu_cloze:{subset}|0|1" for subset in SMOLLM_MMLU_CLOZE_SUBSETS)
        else:
            specs.append(str(task["task_spec"]))
    return specs


def _lighteval_command(
    *,
    model: str,
    output_dir: Path,
    custom_tasks_path: str,
    task_spec: str,
    max_samples: int,
    batch_size: int,
    trust_remote_code: bool = False,
    use_chat_template: bool = False,
) -> str:
    model_args = f"model_name={model},batch_size={batch_size}"
    if trust_remote_code:
        model_args += ",trust_remote_code=True"
    command = (
        "lighteval accelerate "
        f'"{model_args}" '
        f'"{task_spec}" '
        f"--custom-tasks {custom_tasks_path} "
        f"--output-dir {output_dir} "
        f"--max-samples {max_samples} "
        "--save-details"
    )
    if use_chat_template:
        command += " --use-chat-template"
    return command


def _extract_task_score(payload: dict[str, Any], task: dict[str, Any]) -> float | None:
    if task.get("aggregate_prefix"):
        scores = []
        for key, value in _walk_metric_dicts(payload):
            normalized = _normalize_task_name(key)
            if normalized.startswith(str(task["aggregate_prefix"])):
                score = _metric_from_dict(value, task)
                if score is not None:
                    scores.append(score)
        return _average(scores)

    candidates = {str(task["id"]), _normalize_task_name(str(task["task_spec"]))}
    for key, value in _walk_metric_dicts(payload):
        normalized = _normalize_task_name(key)
        if normalized in candidates or normalized.endswith(f":{task['id']}"):
            score = _metric_from_dict(value, task)
            if score is not None:
                return score
    return None


def _walk_metric_dicts(value: Any, *, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if any(_looks_numeric(metric_value) for metric_value in value.values()):
            found.append((path, value))
        for key, nested in value.items():
            next_path = str(key) if not path else f"{path}.{key}"
            found.extend(_walk_metric_dicts(nested, path=next_path))
    return found


def _metric_from_dict(metrics: dict[str, Any], task: dict[str, Any]) -> float | None:
    keys = task["metric_keys"]
    normalized = {_normalize_metric_name(key): value for key, value in metrics.items()}
    for key in keys:
        value = normalized.get(_normalize_metric_name(key))
        if _looks_numeric(value):
            return _score_scale(float(value), task=task)
    for key, value in normalized.items():
        if any(name in key for name in ("acc", "exact_match", "f1")) and _looks_numeric(value):
            return _score_scale(float(value), task=task)
    return None


def _normalize_task_name(value: str) -> str:
    if "|" in value:
        parts = value.split("|")
        if len(parts) >= 2:
            value = parts[1]
    return value.removeprefix("results.").removeprefix("tasks.").strip()


def _normalize_metric_name(value: str) -> str:
    return value.replace(",none", "").replace("|none", "").strip()


def _score_scale(value: float, *, task: dict[str, Any]) -> float:
    denominator = task.get("normalization_denominator")
    if denominator:
        return value / float(denominator) if value > 1.0 else value
    return value / 100.0 if value > 1.0 else value


def _max_drop(rows: list[dict[str, Any]]) -> float | None:
    drops = []
    for row in rows:
        dense = row.get("dense_score")
        moe = row.get("moe_score")
        if dense is not None and moe is not None:
            drops.append(float(dense) - float(moe))
    return max(drops) if drops else None


def _min_retention(rows: list[dict[str, Any]]) -> float | None:
    retentions = [float(row["retention"]) for row in rows if row.get("retention") is not None]
    return min(retentions) if retentions else None


def _threshold_check(*, name: str, value: float | None, passed: bool, message: str, threshold: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "value": value,
        "message": message,
        "threshold": threshold,
    }


def _next_actions(*, failed: list[dict[str, Any]], warnings: list[str]) -> list[str]:
    if failed:
        return [
            "Improve sparse router recovery and rerun the same benchmark plan.",
            "Compare all-expert, default-pool, and learned-router packages to isolate routing loss from carving loss.",
        ]
    if warnings:
        return ["Fill missing task metrics before treating this as release evidence."]
    return ["Attach benchmark-compare.json and raw harness outputs to the model card before HF publication."]


def _average(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _looks_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload
