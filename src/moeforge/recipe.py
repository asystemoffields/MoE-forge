from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Stage:
    name: str
    description: str


@dataclass(slots=True)
class RouterPlan:
    kind: str
    load_balance_weight: float
    entropy_weight: float
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CalibrationPlan:
    samples: int
    sequence_length: int
    dataset: str


@dataclass(slots=True)
class EvalPlan:
    perplexity: bool
    teacher_kl: bool
    smoke_prompts: bool
    speed: bool
    memory: bool


@dataclass(slots=True)
class ExportPlan:
    target: str
    quantize_after_export: bool = False


@dataclass(slots=True)
class ConversionRecipe:
    source_model: str
    source_format: str
    architecture: str | None
    adapter_family: str | None
    goal: str
    target: str
    hardware: str
    strategy: str
    experts: int
    top_k: int
    shared_ratio: float
    moe_layers: list[int] | str
    router: RouterPlan
    calibration: CalibrationPlan
    recovery_steps: int
    eval: EvalPlan
    export: ExportPlan
    layout: dict[str, Any]
    stages: list[Stage]
    warnings: list[str] = field(default_factory=list)
    source_summary: dict[str, Any] = field(default_factory=dict)


def recipe_to_dict(recipe: ConversionRecipe) -> dict[str, Any]:
    return asdict(recipe)
