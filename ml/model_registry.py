from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_BASENAME = "distilbert_clause_classifier"
PRODUCTION_POINTER_DIR = MODELS_DIR / MODEL_BASENAME
PRODUCTION_POINTER_PATH = PRODUCTION_POINTER_DIR / "production.json"


@dataclass
class ModelVersionInfo:
    model_name: str
    version: int
    artifact_path: str
    created_at: str
    stage: str = "staging"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_versioned_model_dir(version: int) -> Path:
    return MODELS_DIR / f"{MODEL_BASENAME}_v{version}"


def list_existing_versions() -> list[int]:
    if not MODELS_DIR.exists():
        return []

    pattern = re.compile(rf"^{re.escape(MODEL_BASENAME)}_v(\d+)$")
    versions: list[int] = []

    for path in MODELS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            versions.append(int(match.group(1)))

    return sorted(versions)


def get_next_version() -> int:
    versions = list_existing_versions()
    return 1 if not versions else versions[-1] + 1


def get_latest_version() -> int | None:
    versions = list_existing_versions()
    if not versions:
        return None
    return versions[-1]


def save_version_metadata(version_info: ModelVersionInfo) -> Path:
    model_dir = Path(version_info.artifact_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_dir / "version.json"
    metadata_path.write_text(
        json.dumps(asdict(version_info), indent=2),
        encoding="utf-8",
    )
    return metadata_path


def load_version_metadata(model_dir: Path) -> ModelVersionInfo:
    metadata_path = model_dir / "version.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"version.json not found at: {metadata_path}")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ModelVersionInfo(**data)


def mark_as_production(version_info: ModelVersionInfo) -> Path:
    PRODUCTION_POINTER_DIR.mkdir(parents=True, exist_ok=True)

    payload = asdict(version_info)
    payload["stage"] = "production"

    PRODUCTION_POINTER_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return PRODUCTION_POINTER_PATH


def get_production_model_info() -> ModelVersionInfo | None:
    if not PRODUCTION_POINTER_PATH.exists():
        return None

    data = json.loads(PRODUCTION_POINTER_PATH.read_text(encoding="utf-8"))
    return ModelVersionInfo(**data)


def resolve_production_model_dir() -> Path | None:
    info = get_production_model_info()
    if info is None:
        return None

    model_dir = Path(info.artifact_path)
    if model_dir.exists():
        return model_dir

    return None


def resolve_latest_model_dir() -> Path | None:
    latest_version = get_latest_version()
    if latest_version is None:
        return None

    model_dir = get_versioned_model_dir(latest_version)
    if model_dir.exists():
        return model_dir

    return None


def resolve_model_dir() -> Path:
    production_dir = resolve_production_model_dir()
    if production_dir is not None:
        return production_dir

    latest_dir = resolve_latest_model_dir()
    if latest_dir is not None:
        return latest_dir

    raise FileNotFoundError(
        "No versioned model directory found. Train a model first."
    )