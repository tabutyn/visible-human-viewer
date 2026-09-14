#!/usr/bin/env python3
"""Export only current, allowlisted JSON metadata; never modify the source data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


SUBJECTS = ("male", "female")
METADATA_FILES = (
    "manifest.json",
    "corrections.json",
    "overrides.json",
    "alignment.json",
    "alignment-candidate-v2.json",
    "reviewed-alignment.json",
    "manual-alignment.json",
    "medical-view.json",
    "alignment-qc.json",
    "qc.json",
)
VOLUME_METADATA_FILES = (
    "volume-v1/manifest.json",
    "rgb-volume-v1/manifest.json",
)
SUBJECT_STATUS = {
    "male": {
        "status": "experimental",
        "warnings": [
            "Experimental alignment; not clinically validated.",
            "Saved human corrections and automated candidates are distinct evidence, not proof of anatomical accuracy.",
        ],
    },
    "female": {
        "status": "stale-geometry-unvalidated",
        "warnings": [
            "Treat female alignment as stale until GE pixel spacing/FOV metadata and derived CT geometry are verified and rebuilt.",
            "Not clinically validated; do not promote these profiles merely because they are archived here.",
        ],
    },
}


def export_checkpoint(processed_root, output, subjects=SUBJECTS, include_volume_manifests=False):
    """Atomically publish a new checkpoint while preserving each file's exact bytes.

    An exclusive sibling lock serializes cooperating exporters. Existing outputs,
    source symlinks, malformed JSON, and outputs inside the source tree are refused.
    """
    processed_root = Path(processed_root).resolve(strict=True)
    output = Path(output).absolute()
    if not processed_root.is_dir():
        raise ValueError("Processed root must be a directory")
    if output.resolve().is_relative_to(processed_root):
        raise ValueError("Checkpoint output must be outside the source data tree")
    if os.path.lexists(output):
        raise FileExistsError(f"Checkpoint already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.export.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(lock_fd)
        if os.path.lexists(output):
            raise FileExistsError(f"Checkpoint already exists: {output}")
        index = {
            "schema_version": 1,
            "kind": "visible-human-metadata-checkpoint",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contents": "allowlisted-current-json-metadata-only",
            "source_layout": "Processed/v1/{subject}/{file}",
            "files_preserved_byte_for_byte": True,
            "warnings": [
                "No photographs, CT pixels, volume bricks, caches, or edit history are included.",
                "This is an archival snapshot, not an anatomical validation or an automatic restore procedure.",
                "Restore only against the exact matching source inputs; do not rewrite input hashes or mark stale profiles current.",
                "Stop editing while exporting if a transactionally consistent snapshot across multiple files is required.",
            ],
            "subjects": {},
        }
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.export-", dir=output.parent) as temporary:
            staging = Path(temporary) / "checkpoint"
            staging.mkdir()
            selected = tuple(subjects)
            unknown = set(selected) - set(SUBJECTS)
            if unknown or not selected:
                raise ValueError(f"Unsupported checkpoint subjects: {', '.join(sorted(unknown)) or 'none'}")
            for subject in selected:
                source_dir = processed_root / subject
                if source_dir.is_symlink():
                    raise ValueError(f"Source subject may not be a symlink: {subject}")
                if not source_dir.is_dir():
                    continue
                record = {**SUBJECT_STATUS[subject], "files": [], "missing_optional_files": []}
                names = METADATA_FILES + (VOLUME_METADATA_FILES if include_volume_manifests else ())
                for name in names:
                    source = source_dir / name
                    if source.is_symlink():
                        raise ValueError(f"Source metadata may not be a symlink: {subject}/{name}")
                    if not source.exists():
                        record["missing_optional_files"].append(name)
                        continue
                    if not source.is_file():
                        raise ValueError(f"Source metadata must be a regular file: {subject}/{name}")
                    payload = source.read_bytes()
                    json.loads(payload)  # Validate, but never reserialize source bytes.
                    relative = Path(subject) / name
                    destination = staging / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    record["files"].append({
                        "path": relative.as_posix(),
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    })
                if record["files"]:
                    index["subjects"][subject] = record
            if not index["subjects"]:
                raise ValueError("No allowlisted subject metadata was found")
            records = [entry for subject in index["subjects"].values() for entry in subject["files"]]
            index["metadata_file_count"] = len(records)
            index["metadata_bytes"] = sum(entry["size_bytes"] for entry in records)
            (staging / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
            # Detect a file changed during export instead of publishing mixed versions.
            for entry in records:
                current = (processed_root / entry["path"]).read_bytes()
                if hashlib.sha256(current).hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"Source changed during export: {entry['path']}; retry after editing stops")
            if os.path.lexists(output):
                raise FileExistsError(f"Checkpoint already exists: {output}")
            os.rename(staging, output)
        return index
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True, help="Existing Processed/v1 directory, read only")
    parser.add_argument("--output", type=Path, required=True, help="New checkpoint directory outside the processed data tree")
    parser.add_argument("--subject", action="append", choices=SUBJECTS, dest="subjects", help="Subject to include; repeat as needed (default: both)")
    parser.add_argument("--include-volume-manifests", action="store_true", help="Include CT/RGB geometry and brick manifests, but never brick payloads")
    args = parser.parse_args()
    try:
        index = export_checkpoint(
            args.processed_root, args.output, args.subjects or SUBJECTS,
            include_volume_manifests=args.include_volume_manifests,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Checkpoint not written: {error}\n")
    print(f"Exported {index['metadata_file_count']} JSON files ({index['metadata_bytes']:,} bytes) to {args.output}")


if __name__ == "__main__":
    main()
