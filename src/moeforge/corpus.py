from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable


class CorpusBuildError(RuntimeError):
    """Raised when a recovery corpus cannot be built."""


@dataclass(slots=True)
class CorpusBuildOptions:
    output_path: Path
    manifest_path: Path | None = None
    sources: tuple[str, ...] = ("smollm-benchmix",)
    max_samples_per_source: int = 128
    seed: int = 13
    include_answers: bool = False
    split: str = "train"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    candidates: tuple[str, ...]
    formatter: str
    config: str | None = None
    split: str | None = None


@dataclass(slots=True)
class CorpusSample:
    text: str
    source: dict[str, Any]


SMOLLM_BENCHMIX: tuple[DatasetSource, ...] = (
    DatasetSource(name="piqa", candidates=("piqa", "ybisk/piqa"), formatter="piqa"),
    DatasetSource(
        name="arc_easy",
        candidates=("ai2_arc", "allenai/ai2_arc"),
        config="ARC-Easy",
        formatter="arc",
    ),
    DatasetSource(
        name="arc_challenge",
        candidates=("ai2_arc", "allenai/ai2_arc"),
        config="ARC-Challenge",
        formatter="arc",
    ),
    DatasetSource(name="hellaswag", candidates=("hellaswag", "Rowan/hellaswag"), formatter="hellaswag"),
    DatasetSource(
        name="commonsense_qa",
        candidates=("commonsense_qa", "tau/commonsense_qa"),
        formatter="commonsense_qa",
    ),
    DatasetSource(
        name="winogrande",
        candidates=("winogrande", "allenai/winogrande"),
        config="winogrande_debiased",
        formatter="winogrande",
    ),
    DatasetSource(
        name="openbookqa",
        candidates=("openbookqa", "allenai/openbookqa"),
        config="main",
        formatter="openbookqa",
    ),
)


BUILTIN_SMOKE_TEXTS: tuple[str, ...] = (
    "Question: Which object is used to write on a chalkboard?\nChoices:\nA. spoon\nB. chalk\nC. blanket\nD. river",
    "Question: If rain falls all night, what is likely to be wet in the morning?\nChoices:\nA. sidewalk\nB. flame\nC. desert map\nD. candle",
    "Question: What tool is best for cutting paper?\nChoices:\nA. scissors\nB. pillow\nC. cloud\nD. bottle",
    "A scientist writes a hypothesis, collects evidence, and revises the claim.",
    "A recipe lists ingredients and steps for cooking food.",
)


FORMATTERS: dict[str, Callable[[dict[str, Any], bool], str | None]] = {}


def build_recovery_corpus(options: CorpusBuildOptions) -> dict[str, Any]:
    if options.max_samples_per_source <= 0:
        raise CorpusBuildError("max_samples_per_source must be positive")

    sources = _expand_sources(options.sources, default_split=options.split)
    rng = random.Random(options.seed)
    samples: list[CorpusSample] = []
    source_reports: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in sources:
        if source.name == "builtin-smoke":
            source_samples = _builtin_samples(max_samples=options.max_samples_per_source, rng=rng)
            samples.extend(source_samples)
            source_reports.append(
                {
                    "name": source.name,
                    "kind": "builtin",
                    "status": "loaded",
                    "sample_count": len(source_samples),
                    "include_answers": False,
                }
            )
            continue

        try:
            source_samples, report = _dataset_samples(
                source,
                max_samples=options.max_samples_per_source,
                seed=options.seed,
                include_answers=options.include_answers,
            )
        except CorpusBuildError as exc:
            warnings.append(str(exc))
            source_reports.append(
                {
                    "name": source.name,
                    "kind": "hf_dataset",
                    "status": "skipped",
                    "dataset_candidates": list(source.candidates),
                    "config": source.config,
                    "split": source.split or options.split,
                    "reason": str(exc),
                }
            )
            continue
        samples.extend(source_samples)
        source_reports.append(report)

    if not samples:
        warnings.append("all configured dataset sources failed; using builtin smoke corpus")
        samples = _builtin_samples(max_samples=options.max_samples_per_source, rng=rng)
        source_reports.append(
            {
                "name": "builtin-smoke",
                "kind": "builtin",
                "status": "loaded",
                "sample_count": len(samples),
                "include_answers": False,
            }
        )

    text = "\n\n".join(sample.text.strip() for sample in samples if sample.text.strip()) + "\n"
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(text, encoding="utf-8")
    data = options.output_path.read_bytes()
    manifest_path = options.manifest_path or options.output_path.with_suffix(options.output_path.suffix + ".manifest.json")
    manifest = {
        "format": "moeforge_recovery_corpus",
        "output_path": str(options.output_path),
        "manifest_path": str(manifest_path),
        "byte_count": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sample_count": len(samples),
        "sample_sha256": [hashlib.sha256(sample.text.encode("utf-8")).hexdigest() for sample in samples],
        "max_samples_per_source": options.max_samples_per_source,
        "seed": options.seed,
        "include_answers": options.include_answers,
        "split": options.split,
        "sources": source_reports,
        "warnings": warnings,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _expand_sources(raw_sources: tuple[str, ...], *, default_split: str) -> list[DatasetSource]:
    expanded: list[DatasetSource] = []
    for raw in raw_sources:
        name = raw.strip()
        if not name:
            continue
        if name == "smollm-benchmix":
            expanded.extend(
                DatasetSource(
                    name=source.name,
                    candidates=source.candidates,
                    formatter=source.formatter,
                    config=source.config,
                    split=source.split or default_split,
                )
                for source in SMOLLM_BENCHMIX
            )
        elif name == "builtin-smoke":
            expanded.append(DatasetSource(name="builtin-smoke", candidates=(), formatter="builtin", split=default_split))
        else:
            matched = [source for source in SMOLLM_BENCHMIX if source.name == name]
            if not matched:
                raise CorpusBuildError(f"unknown corpus source: {name}")
            source = matched[0]
            expanded.append(
                DatasetSource(
                    name=source.name,
                    candidates=source.candidates,
                    formatter=source.formatter,
                    config=source.config,
                    split=source.split or default_split,
                )
            )
    if not expanded:
        raise CorpusBuildError("at least one corpus source is required")
    return expanded


def _dataset_samples(
    source: DatasetSource,
    *,
    max_samples: int,
    seed: int,
    include_answers: bool,
) -> tuple[list[CorpusSample], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise CorpusBuildError("datasets is required for HF-backed corpus sources") from exc

    errors: list[str] = []
    dataset = None
    loaded_candidate = None
    loaded_via = None
    for candidate in source.candidates:
        try:
            dataset = load_dataset(candidate, source.config, split=source.split)
            loaded_candidate = candidate
            loaded_via = "load_dataset"
            break
        except Exception as exc:  # pragma: no cover - depends on remote datasets
            try:
                dataset = _load_converted_parquet_dataset(
                    load_dataset=load_dataset,
                    dataset_id=candidate,
                    config=source.config,
                    split=str(source.split or "train"),
                )
                loaded_candidate = candidate
                loaded_via = "converted_parquet"
                break
            except Exception as parquet_exc:  # pragma: no cover - depends on remote datasets
                errors.append(f"{candidate}: {exc}; converted parquet: {parquet_exc}")
    if dataset is None or loaded_candidate is None:
        raise CorpusBuildError(f"could not load {source.name}: {'; '.join(errors)}")

    try:
        dataset = dataset.shuffle(seed=seed)
    except Exception:
        pass

    formatter = _formatter(source.formatter)
    samples: list[CorpusSample] = []
    checked = 0
    limit = _dataset_length(dataset)
    while len(samples) < max_samples and checked < limit:
        row = dict(dataset[checked])
        checked += 1
        text = formatter(row, include_answers)
        if not text:
            continue
        samples.append(
            CorpusSample(
                text=text,
                source={
                    "source": source.name,
                    "dataset": loaded_candidate,
                    "config": source.config,
                    "split": source.split,
                    "row_index": checked - 1,
                },
            )
        )

    if not samples:
        raise CorpusBuildError(f"{source.name} produced no usable text samples")

    return samples, {
        "name": source.name,
        "kind": "hf_dataset",
        "status": "loaded",
        "dataset": loaded_candidate,
        "loaded_via": loaded_via,
        "dataset_candidates": list(source.candidates),
        "config": source.config,
        "split": source.split,
        "formatter": source.formatter,
        "sample_count": len(samples),
        "include_answers": include_answers,
    }


def _load_converted_parquet_dataset(
    *,
    load_dataset: Any,
    dataset_id: str,
    config: str | None,
    split: str,
) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - bundled with datasets in normal HF installs
        raise CorpusBuildError("huggingface_hub is required for converted parquet fallback") from exc

    split_name = split.split("[", 1)[0]
    files = HfApi().list_repo_files(
        repo_id=dataset_id,
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    parquet_files = [path for path in files if path.endswith(".parquet") and f"/{split_name}/" in path]
    if config:
        config_prefix = f"{config}/"
        config_files = [path for path in parquet_files if path.startswith(config_prefix)]
        if config_files:
            parquet_files = config_files
    if not parquet_files:
        raise CorpusBuildError(f"no converted parquet files found for split {split_name}")
    urls = [
        f"hf://datasets/{dataset_id}@refs/convert/parquet/{path}"
        for path in sorted(parquet_files)
    ]
    return load_dataset("parquet", data_files={split_name: urls}, split=split_name)


def _dataset_length(dataset: Any) -> int:
    try:
        return len(dataset)
    except TypeError as exc:  # pragma: no cover - normal load_dataset returns sized datasets
        raise CorpusBuildError("streaming datasets are not supported for corpus-build") from exc


def _builtin_samples(*, max_samples: int, rng: random.Random) -> list[CorpusSample]:
    texts = list(BUILTIN_SMOKE_TEXTS)
    rng.shuffle(texts)
    return [
        CorpusSample(text=text, source={"source": "builtin-smoke", "row_index": index})
        for index, text in enumerate(texts[:max_samples])
    ]


def _formatter(name: str) -> Callable[[dict[str, Any], bool], str | None]:
    if not FORMATTERS:
        FORMATTERS.update(
            {
                "piqa": _format_piqa,
                "arc": _format_arc,
                "hellaswag": _format_hellaswag,
                "commonsense_qa": _format_commonsense_qa,
                "winogrande": _format_winogrande,
                "openbookqa": _format_openbookqa,
            }
        )
    try:
        return FORMATTERS[name]
    except KeyError as exc:
        raise CorpusBuildError(f"unknown corpus formatter: {name}") from exc


def _format_piqa(row: dict[str, Any], include_answer: bool) -> str | None:
    goal = _clean(row.get("goal"))
    choices = [_clean(row.get("sol1")), _clean(row.get("sol2"))]
    if not goal or not all(choices):
        return None
    answer = _answer_from_index(row.get("label"), choices)
    return _question_block(question=goal, choices=choices, answer=answer if include_answer else None)


def _format_arc(row: dict[str, Any], include_answer: bool) -> str | None:
    question = _clean(row.get("question"))
    labels, choices = _choice_dict(row.get("choices"))
    if not question or not choices:
        return None
    answer = _answer_from_label(row.get("answerKey"), labels, choices)
    return _question_block(question=question, choices=choices, labels=labels, answer=answer if include_answer else None)


def _format_hellaswag(row: dict[str, Any], include_answer: bool) -> str | None:
    context = " ".join(
        item
        for item in (_clean(row.get("activity_label")), _clean(row.get("ctx")), _clean(row.get("ctx_a")), _clean(row.get("ctx_b")))
        if item
    )
    choices = [_clean(item) for item in _as_list(row.get("endings")) if _clean(item)]
    if not context or not choices:
        return None
    answer = _answer_from_index(row.get("label"), choices)
    return _question_block(
        question=f"Choose the most plausible continuation: {context}",
        choices=choices,
        answer=answer if include_answer else None,
    )


def _format_commonsense_qa(row: dict[str, Any], include_answer: bool) -> str | None:
    question = _clean(row.get("question"))
    labels, choices = _choice_dict(row.get("choices"))
    if not question or not choices:
        return None
    answer = _answer_from_label(row.get("answerKey"), labels, choices)
    return _question_block(question=question, choices=choices, labels=labels, answer=answer if include_answer else None)


def _format_winogrande(row: dict[str, Any], include_answer: bool) -> str | None:
    sentence = _clean(row.get("sentence"))
    choices = [_clean(row.get("option1")), _clean(row.get("option2"))]
    if not sentence or not all(choices):
        return None
    answer = _answer_from_index(_one_based_index(row.get("answer")), choices)
    return _question_block(
        question=f"Fill the blank in this sentence: {sentence}",
        choices=choices,
        answer=answer if include_answer else None,
    )


def _format_openbookqa(row: dict[str, Any], include_answer: bool) -> str | None:
    question = _clean(row.get("question_stem") or row.get("question"))
    labels, choices = _choice_dict(row.get("choices"))
    if not question or not choices:
        return None
    answer = _answer_from_label(row.get("answerKey"), labels, choices)
    return _question_block(question=question, choices=choices, labels=labels, answer=answer if include_answer else None)


def _question_block(
    *,
    question: str,
    choices: list[str],
    labels: list[str] | None = None,
    answer: str | None = None,
) -> str:
    resolved_labels = labels or [_letter(index) for index in range(len(choices))]
    lines = [f"Question: {question}", "Choices:"]
    lines.extend(f"{label}. {choice}" for label, choice in zip(resolved_labels, choices, strict=False))
    if answer:
        lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def _choice_dict(value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, dict):
        return [], []
    labels = [_clean(item) for item in _as_list(value.get("label"))]
    choices = [_clean(item) for item in _as_list(value.get("text"))]
    pairs = [(label or _letter(index), choice) for index, (label, choice) in enumerate(zip(labels, choices, strict=False)) if choice]
    if not pairs:
        return [], []
    return [label for label, _ in pairs], [choice for _, choice in pairs]


def _answer_from_index(value: Any, choices: list[str]) -> str | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= index < len(choices):
        return choices[index]
    return None


def _one_based_index(value: Any) -> int | None:
    try:
        return int(value) - 1
    except (TypeError, ValueError):
        return None


def _answer_from_label(value: Any, labels: list[str], choices: list[str]) -> str | None:
    key = _clean(value)
    if not key:
        return None
    for label, choice in zip(labels, choices, strict=False):
        if label == key:
            return choice
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else list(value) if isinstance(value, tuple) else [value]


def _clean(value: Any) -> str:
    return " ".join(str(value).replace("\n", " ").split()) if value is not None else ""


def _letter(index: int) -> str:
    return chr(ord("A") + index)
