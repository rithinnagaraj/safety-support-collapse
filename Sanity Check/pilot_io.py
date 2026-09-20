"""Section 1 paths, manifest validation, and artifact I/O."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

SECTION_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SECTION_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from protocol_core import content_hash, read_json, read_jsonl, require_keys, write_json  # noqa: E402
from benign_tasks import LOGIC_WORDING_BANKS, MATH_WORDING_BANKS, generate_benign_split  # noqa: E402
from safety_harness import PARSER_SCHEMA_VERSION, SYSTEM_PROMPT, _score  # noqa: E402
from safety_tasks import (  # noqa: E402
    CURRENT_FILE_TEMPLATES,
    CURRENT_RECIPIENT_TEMPLATES,
    FUTURE_TEMPLATES,
    TOOL_SCHEMA,
    generate_safety_split,
)

CONFIG_PATH = SECTION_DIR / "pilot_config.json"
DEFAULT_DATA_DIR = SECTION_DIR / "data"
DEFAULT_ARTIFACT_DIR = REPOSITORY_ROOT / "artifacts" / "sanity-check"


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=False
    )
    head = result.stdout.strip() if result.returncode == 0 else "uncommitted"
    excluded_parts = {
        ".git",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "checkpoints",
        "data",
        "wandb",
    }
    source_files = sorted(
        path for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file()
        and excluded_parts.isdisjoint(path.parts)
        and path.suffix.lower() in {".py", ".json", ".txt", ".md"}
    )
    source_digest = content_hash(
        [{"path": str(path.relative_to(REPOSITORY_ROOT)), "sha256": content_hash(path.read_text(encoding="utf-8"))} for path in source_files]
    )
    return f"{head}+source-{source_digest}"


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for distribution in (
        "numpy",
        "torch",
        "transformers",
        "peft",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
        "accelerate",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def tokenizer_fingerprint(model_id: str, revision: str) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers must be installed before freezing or validating the manifest") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("The frozen tokenizer does not provide a native chat template")
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "chat_template_sha256": content_hash(tokenizer.chat_template),
        "special_tokens_sha256": content_hash(tokenizer.special_tokens_map),
        "padding_side": tokenizer.padding_side,
    }


def device_inventory() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "cuda_version": None, "devices": []}
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {"cuda_available": available, "cuda_version": torch.version.cuda, "devices": devices}


def component_hashes() -> dict[str, str]:
    return {
        "reward_extractor_hash": content_hash(inspect.getsource(_score)),
        "generator_template_hash": content_hash(
            {
                "safety_generator": inspect.getsource(generate_safety_split),
                "benign_generator": inspect.getsource(generate_benign_split),
                "math_wording_banks": MATH_WORDING_BANKS,
                "logic_wording_banks": LOGIC_WORDING_BANKS,
                "current_file_templates": CURRENT_FILE_TEMPLATES,
                "current_recipient_templates": CURRENT_RECIPIENT_TEMPLATES,
                "future_templates": FUTURE_TEMPLATES,
            }
        ),
    }


def validate_manifest(
    manifest_path: Path, *, require_ml: bool = True, execution_device: str | None = None
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    require_keys(
        manifest,
        {
            "protocol_version", "model", "code_revision", "dependencies", "device",
            "parser_schema_hash", "reward_extractor_hash", "generator_template_hash",
            "tokenizer", "splits", "seed_mapping", "frozen_config",
        },
        "manifest",
    )
    require_keys(manifest["model"], {"id", "revision", "chat_template"}, "manifest.model")
    require_keys(
        manifest["tokenizer"],
        {"class", "vocab_size", "chat_template_sha256", "special_tokens_sha256", "padding_side"},
        "manifest.tokenizer",
    )
    require_keys(
        manifest["frozen_config"],
        {
            "protocol_version", "model", "sampling", "adapter", "optimizer", "training",
            "calibration_smoke", "gates", "seeds", "safety_splits", "benign_splits",
        },
        "manifest.frozen_config",
    )
    if manifest["seed_mapping"] != manifest["frozen_config"]["seeds"]:
        raise ValueError("Manifest seed mapping differs from the frozen configuration")
    if manifest["model"] != manifest["frozen_config"]["model"]:
        raise ValueError("Manifest model metadata differs from the frozen configuration")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(manifest["model"]["revision"])):
        raise ValueError("Manifest must freeze the model to a full 40-character commit SHA")
    if not manifest["device"].get("description"):
        raise ValueError("Manifest must include a concrete device description")
    if not manifest["device"].get("execution_device"):
        raise ValueError("Manifest must freeze the execution device selector")
    if execution_device is not None and execution_device != manifest["device"]["execution_device"]:
        raise ValueError(
            f"Requested device {execution_device!r} differs from frozen device "
            f"{manifest['device']['execution_device']!r}"
        )
    current_revision = git_revision()
    if manifest["code_revision"] != current_revision:
        raise ValueError("Source tree changed after the data manifest was frozen; regenerate all splits")
    current_schema_hash = content_hash(
        {
            "system_prompt": SYSTEM_PROMPT,
            "tool_schema": TOOL_SCHEMA,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
        }
    )
    if manifest["parser_schema_hash"] != current_schema_hash:
        raise ValueError("Parser/action schema changed after the data manifest was frozen")
    for name, expected in component_hashes().items():
        if manifest[name] != expected:
            raise ValueError(f"Frozen component changed after manifest creation: {name}")
    if require_ml:
        missing = [name for name in ("torch", "transformers", "peft") if manifest["dependencies"].get(name) in {None, "not-installed"}]
        if missing:
            raise ValueError(f"Manifest records missing ML dependencies: {', '.join(missing)}")
        current_dependencies = dependency_versions()
        changed = {
            name: {"frozen": version, "current": current_dependencies.get(name)}
            for name, version in manifest["dependencies"].items()
            if current_dependencies.get(name) != version
        }
        if changed:
            raise ValueError(f"Dependency versions differ from the frozen manifest: {changed}")
        if manifest["device"].get("detected") != device_inventory():
            raise ValueError("Detected accelerator configuration differs from the frozen manifest")
        current_tokenizer = tokenizer_fingerprint(
            manifest["model"]["id"], manifest["model"]["revision"]
        )
        if manifest["tokenizer"] != current_tokenizer:
            raise ValueError("Tokenizer or native chat template differs from the frozen manifest")
    data_dir = manifest_path.parent
    for split_name, metadata in manifest["splits"].items():
        path = data_dir / metadata["file"]
        if not path.is_file():
            raise ValueError(f"Manifest split file is missing: {path}")
        rows = list(read_jsonl(path))
        if len(rows) != metadata["count"]:
            raise ValueError(f"Split {split_name} count changed: {len(rows)} != {metadata['count']}")
        if content_hash(rows) != metadata["sha256"]:
            raise ValueError(f"Split {split_name} hash mismatch")
    return manifest


def record_run_metadata(output_dir: Path, manifest: dict[str, Any], command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run-metadata.json",
        {
            "command": command,
            "manifest_hash": content_hash(manifest),
            "protocol_version": manifest["protocol_version"],
            "code_revision": git_revision(),
        },
    )
    deviations = output_dir / "deviations.json"
    if not deviations.exists():
        write_json(deviations, {"deviations": []})
    write_json(
        output_dir / "run-status.json",
        {"complete": False, "manifest_hash": content_hash(manifest)},
    )


def mark_run_complete(output_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(
        output_dir / "run-status.json",
        {"complete": True, "manifest_hash": content_hash(manifest)},
    )


def require_matching_run(output_dir: Path, manifest: dict[str, Any], label: str) -> None:
    metadata_path = output_dir / "run-metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"{label} run metadata is missing: {metadata_path}")
    metadata = read_json(metadata_path)
    expected = content_hash(manifest)
    if metadata.get("manifest_hash") != expected:
        raise ValueError(f"{label} was produced under a different frozen manifest")
    status_path = output_dir / "run-status.json"
    if not status_path.is_file():
        raise ValueError(f"{label} completion status is missing: {status_path}")
    status = read_json(status_path)
    if status.get("manifest_hash") != expected or status.get("complete") is not True:
        raise ValueError(f"{label} is incomplete or belongs to a different frozen manifest")
