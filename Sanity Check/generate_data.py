"""Generate and freeze every Section 1 split before any training."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import sys

SECTION_DIR = Path(__file__).resolve().parent
ROOT = SECTION_DIR.parent
for path in (ROOT, SECTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benign_tasks import generate_benign_split  # noqa: E402
from protocol_core import content_hash, write_json, write_jsonl  # noqa: E402
from safety_harness import PARSER_SCHEMA_VERSION, SYSTEM_PROMPT  # noqa: E402
from safety_tasks import TOOL_SCHEMA, generate_safety_split, validate_split  # noqa: E402
from pilot_io import (  # noqa: E402
    DEFAULT_DATA_DIR,
    component_hashes,
    dependency_versions,
    device_inventory,
    git_revision,
    load_config,
    tokenizer_fingerprint,
)


def build_data(
    output_dir: Path, model_revision: str, device_description: str, execution_device: str = "cuda"
) -> dict:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", model_revision):
        raise ValueError("--model-revision must be a full 40-character immutable commit SHA")
    if not device_description.strip():
        raise ValueError("--device-description must identify the execution hardware")
    config = deepcopy(load_config())
    config["model"]["revision"] = model_revision
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata: dict[str, dict] = {}
    safety_latents: set[str] = set()
    safety_semantics: set[str] = set()
    benign_semantics: set[str] = set()
    templates_by_namespace: dict[str, set[str]] = {}

    for name, spec in config["safety_splits"].items():
        rows = generate_safety_split(
            split=name,
            prompt_count=spec["prompts"],
            horizon=spec["horizon"],
            seed=config["seeds"]["data"],
            template_namespace=spec["template_namespace"],
        )
        validate_split(rows, spec["prompts"])
        overlap = safety_latents.intersection(row.latent_id for row in rows)
        if overlap:
            raise ValueError(f"Latent instances overlap across splits: {sorted(overlap)[:3]}")
        safety_latents.update(row.latent_id for row in rows)
        for row in rows:
            if row.surface_variant_id != 0:
                continue
            signature = content_hash(
                {
                    "family": row.family,
                    "horizon": row.horizon,
                    "files": [record.__dict__ for record in row.files],
                    "authorized_recipients": row.authorized_recipients,
                    "unauthorized_recipients": row.unauthorized_recipients,
                    "expected_answer": row.expected_answer,
                    "expected_recipient": row.expected_recipient,
                    "expected_content": row.expected_content,
                }
            )
            if signature in safety_semantics:
                raise ValueError(f"Duplicate safety latent content detected at {row.latent_id}")
            safety_semantics.add(signature)
        templates_by_namespace.setdefault(spec["template_namespace"], set()).update(row.template_id for row in rows)
        payload = [row.to_dict() for row in rows]
        filename = f"{name}.jsonl"
        write_jsonl(output_dir / filename, payload)
        split_metadata[name] = {"kind": "safety", "file": filename, "count": len(payload), "sha256": content_hash(payload)}

    namespace_names = sorted(templates_by_namespace)
    for index, first in enumerate(namespace_names):
        for second in namespace_names[index + 1 :]:
            overlap = templates_by_namespace[first].intersection(templates_by_namespace[second])
            if overlap:
                raise ValueError(f"Template IDs overlap between {first} and {second}: {sorted(overlap)[:3]}")

    for name, spec in config["benign_splits"].items():
        benign_seeds = (
            [config["seeds"]["discovery_benign"], *config["seeds"]["replication_benign"]]
            if spec["per_benign_seed"]
            else [config["seeds"]["data"]]
        )
        for benign_seed in benign_seeds:
            split_name = f"{name}_seed_{benign_seed}" if spec["per_benign_seed"] else name
            rows = generate_benign_split(
                split=split_name,
                domain=spec["domain"],
                count=spec["prompts"],
                seed=benign_seed,
                wording_bank=spec["wording_bank"],
            )
            for row in rows:
                signature = row.latent_signature
                if signature in benign_semantics:
                    raise ValueError(f"Duplicate benign latent content detected at {row.instance_id}")
                benign_semantics.add(signature)
            payload = [row.to_dict() for row in rows]
            filename = f"{split_name}.jsonl"
            write_jsonl(output_dir / filename, payload)
            split_metadata[split_name] = {
                "kind": "benign",
                "file": filename,
                "count": len(payload),
                "sha256": content_hash(payload),
                "benign_seed": benign_seed if spec["per_benign_seed"] else None,
            }

    schema_material = {
        "system_prompt": SYSTEM_PROMPT,
        "tool_schema": TOOL_SCHEMA,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
    }
    manifest = {
        "protocol_version": config["protocol_version"],
        "model": config["model"],
        "tokenizer": tokenizer_fingerprint(config["model"]["id"], model_revision),
        "code_revision": git_revision(),
        "dependencies": dependency_versions(),
        "device": {
            "description": device_description,
            "execution_device": execution_device,
            "detected": device_inventory(),
        },
        "parser_schema_hash": content_hash(schema_material),
        "splits": split_metadata,
        "seed_mapping": config["seeds"],
        "frozen_config": config,
        **component_hashes(),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-revision", required=True, help="Full 40-character Hugging Face model commit SHA")
    parser.add_argument("--device-description", required=True, help="Concrete accelerator and topology")
    parser.add_argument("--execution-device", default="cuda", help="Frozen torch device selector, e.g. cuda or cuda:1")
    args = parser.parse_args()
    manifest = build_data(
        args.output_dir.resolve(), args.model_revision, args.device_description, args.execution_device
    )
    print(f"Wrote {len(manifest['splits'])} frozen splits to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
