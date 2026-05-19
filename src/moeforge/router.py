from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


class RouterPlanError(RuntimeError):
    """Raised when router metadata cannot be built or applied."""


@dataclass(slots=True)
class DocumentPool:
    document_index: int
    text_sha256: str
    experts: list[int]
    scores: list[float]
    method: str


@dataclass(slots=True)
class RouterPlan:
    strategy: str
    source_profile: str
    adapter_family: str | None
    expert_count: int
    pool_size: int
    default_pool: list[int]
    documents: list[DocumentPool] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_router_plan(
    *,
    profile_path: Path,
    pool_size: int | None = None,
) -> RouterPlan:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expert_count = _infer_expert_count(profile)
    if expert_count <= 0:
        raise RouterPlanError("profile does not expose a positive expert count")

    resolved_pool_size = max(1, min(pool_size or min(2, expert_count), expert_count))
    warnings = list(profile.get("warnings", []))
    documents = []
    score_totals = [0.0 for _ in range(expert_count)]

    for raw_doc in profile.get("documents", []):
        if not isinstance(raw_doc, dict):
            continue
        raw_pool = raw_doc.get("expert_pool")
        if not isinstance(raw_pool, dict):
            warnings.append(f"document {raw_doc.get('index')} has no expert_pool; default pool will be used")
            continue
        scores = [float(item) for item in raw_pool.get("scores", [])]
        scores = _pad_scores(scores, expert_count)
        experts = _top_experts(scores, resolved_pool_size)
        for index, score in enumerate(scores):
            score_totals[index] += score
        documents.append(
            DocumentPool(
                document_index=int(raw_doc.get("index", len(documents))),
                text_sha256=str(raw_doc.get("text_sha256", "")),
                experts=experts,
                scores=scores,
                method=str(raw_pool.get("method", "profile_document_pool")),
            )
        )

    if not documents:
        warnings.append("profile contained no document expert pools; router uses default pool only")

    default_pool = _top_experts(score_totals, resolved_pool_size)
    return RouterPlan(
        strategy="document_pool_then_token_router",
        source_profile=str(profile_path),
        adapter_family=profile.get("adapter_family"),
        expert_count=expert_count,
        pool_size=resolved_pool_size,
        default_pool=default_pool,
        documents=documents,
        warnings=warnings,
        references=[
            "https://allenai.org/blog/emo",
            "https://arxiv.org/abs/2605.06663",
        ],
    )


def select_expert_pool(
    router_plan: dict[str, Any],
    *,
    text: str | None = None,
    text_sha256: str | None = None,
    document_index: int | None = None,
) -> list[int]:
    lookup_hash = text_sha256 or (_hash_text(text) if text is not None else None)
    documents = router_plan.get("documents", [])
    if isinstance(documents, list):
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            if lookup_hash and doc.get("text_sha256") == lookup_hash:
                return [int(item) for item in doc.get("experts", [])]
            if document_index is not None and int(doc.get("document_index", -1)) == int(document_index):
                return [int(item) for item in doc.get("experts", [])]

    default_pool = router_plan.get("default_pool", [])
    if not default_pool:
        raise RouterPlanError("router plan has no matching document pool and no default_pool")
    return [int(item) for item in default_pool]


def _infer_expert_count(profile: dict[str, Any]) -> int:
    for module in profile.get("modules", {}).values():
        if not isinstance(module, dict):
            continue
        assignment = module.get("assignment")
        if isinstance(assignment, dict):
            experts = assignment.get("experts")
            if isinstance(experts, list) and experts:
                return len(experts)
    for doc in profile.get("documents", []):
        if not isinstance(doc, dict):
            continue
        pool = doc.get("expert_pool")
        if isinstance(pool, dict) and isinstance(pool.get("scores"), list):
            return len(pool["scores"])
    return 0


def _top_experts(scores: list[float], pool_size: int) -> list[int]:
    if not scores:
        return []
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:pool_size]


def _pad_scores(scores: list[float], expert_count: int) -> list[float]:
    if len(scores) >= expert_count:
        return scores[:expert_count]
    return scores + [0.0] * (expert_count - len(scores))


def _hash_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

