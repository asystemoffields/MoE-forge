from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HFRefError(ValueError):
    """Raised when a Hugging Face reference cannot be resolved."""


_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class HFModelRef:
    repo_id: str
    revision: str = "main"

    @property
    def display(self) -> str:
        if self.revision == "main":
            return self.repo_id
        return f"{self.repo_id}@{self.revision}"


def parse_hf_model_ref(raw: str) -> HFModelRef | None:
    value = raw.strip()
    if value.startswith("hf:"):
        value = value[3:]

    if "://" in value:
        return None

    value = value.replace("\\", "/")
    if value.startswith("/") or value.startswith("./") or value.startswith("../"):
        return None
    if re.match(r"^[A-Za-z]:/", value):
        return None

    repo_id, revision = _split_revision(value)
    if not _HF_REPO_RE.match(repo_id):
        return None
    return HFModelRef(repo_id=repo_id, revision=revision)


def download_hf_config(ref: HFModelRef, *, cache_dir: Path | None = None, timeout: float = 30.0) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    target = hf_config_cache_path(ref, cache_dir=cache_dir)
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{ref.repo_id}/resolve/{ref.revision}/config.json"
    request = Request(url, headers={"User-Agent": "moe-forge/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        raise HFRefError(f"could not fetch {ref.display}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise HFRefError(f"could not fetch {ref.display}: {exc.reason}") from exc

    target.write_bytes(payload)
    return target


def hf_config_cache_path(ref: HFModelRef, *, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    safe_repo = ref.repo_id.replace("/", "--")
    safe_revision = re.sub(r"[^A-Za-z0-9._-]+", "_", ref.revision)
    return cache_dir / "hf" / safe_repo / safe_revision / "config.json"


def default_cache_dir() -> Path:
    override = os.environ.get("MOEFORGE_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "moe-forge"


def _split_revision(value: str) -> tuple[str, str]:
    if "@" not in value:
        return value, "main"
    repo_id, revision = value.rsplit("@", 1)
    return repo_id, revision or "main"

