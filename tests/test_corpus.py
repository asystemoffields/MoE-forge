from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from moeforge.cli import main
from moeforge.corpus import CorpusBuildError, CorpusBuildOptions, _load_dataset_noninteractive, build_recovery_corpus


def test_build_recovery_corpus_builtin_records_manifest(tmp_path: Path) -> None:
    output = tmp_path / "train.txt"
    manifest_path = tmp_path / "corpus.json"

    manifest = build_recovery_corpus(
        CorpusBuildOptions(
            output_path=output,
            manifest_path=manifest_path,
            sources=("builtin-smoke",),
            max_samples_per_source=2,
            seed=7,
        )
    )

    text = output.read_text(encoding="utf-8")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "moeforge_recovery_corpus"
    assert manifest["sample_count"] == 2
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert saved["sources"][0]["name"] == "builtin-smoke"
    assert saved["sources"][0]["sample_count"] == 2
    assert "Question:" in text or "scientist" in text


def test_corpus_build_cli_writes_builtin_corpus(tmp_path: Path) -> None:
    output = tmp_path / "train.txt"
    manifest = tmp_path / "manifest.json"

    status = main(
        [
            "corpus-build",
            "--source",
            "builtin-smoke",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--max-samples-per-source",
            "1",
        ]
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert status == 0
    assert output.exists()
    assert payload["output_path"] == str(output)
    assert payload["sample_count"] == 1


def test_corpus_build_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(CorpusBuildError, match="unknown corpus source"):
        build_recovery_corpus(
            CorpusBuildOptions(
                output_path=tmp_path / "train.txt",
                sources=("not-a-source",),
            )
        )


def test_load_dataset_noninteractive_sets_trust_remote_code() -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(path: str, name: str | None, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, "name": name, **kwargs})
        return {"loaded": True}

    dataset = _load_dataset_noninteractive(fake_load_dataset, "demo/dataset", "config", split="train")

    assert dataset == {"loaded": True}
    assert calls == [
        {
            "path": "demo/dataset",
            "name": "config",
            "split": "train",
            "trust_remote_code": True,
        }
    ]


def test_load_dataset_noninteractive_falls_back_for_old_datasets() -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(path: str, name: str | None, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, "name": name, **kwargs})
        if "trust_remote_code" in kwargs:
            raise TypeError("unexpected keyword argument 'trust_remote_code'")
        return {"loaded": True}

    dataset = _load_dataset_noninteractive(fake_load_dataset, "demo/dataset", None, split="validation")

    assert dataset == {"loaded": True}
    assert calls == [
        {
            "path": "demo/dataset",
            "name": None,
            "split": "validation",
            "trust_remote_code": True,
        },
        {
            "path": "demo/dataset",
            "name": None,
            "split": "validation",
        },
    ]
