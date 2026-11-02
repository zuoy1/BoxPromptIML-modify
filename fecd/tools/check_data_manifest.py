"""Read-only integrity audit for the reconstructed BoxPromptIML splits.

The only path rewrite is the explicitly declared historical root mapping
``/hy-tmp/datasets`` -> ``/home/datasets``. Every rewritten source image is
checked against the SHA256 recorded in the existing manifest; rows are joined
by dataset and stable sample id, never by sorted directory position or stem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np


PATH_MAPPING_FILE = Path(__file__).resolve().parents[1] / "configs/stage2a_path_mapping.json"
PATH_MAPPING = json.loads(PATH_MAPPING_FILE.read_text())
_PREFIX_MAPPING = PATH_MAPPING["prefix_mappings"][0]
OLD_ROOT = _PREFIX_MAPPING["from"]
NEW_ROOT = Path(_PREFIX_MAPPING["to"])
DEFAULT_SPLIT_ROOT = NEW_ROOT / "processed/boxprompt_paper_scale_20261016"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "assets/stage2a_data_audit.json"


def resolve_path(value: str) -> Path:
    if value == OLD_ROOT:
        return NEW_ROOT
    if value.startswith(OLD_ROOT + "/"):
        return NEW_ROOT / value[len(OLD_ROOT) + 1 :]
    path = Path(value)
    if path.is_absolute():
        return path
    raise ValueError(f"Manifest path is neither absolute nor covered by mapping: {value}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_image(path: Path, mode: str | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        image.load()
        if mode:
            image = image.convert(mode)
        array = np.asarray(image)
        size = image.size
    return array, size


def sample_key(dataset: str, split: str, sample_id: str, image_sha: str) -> str:
    body = "\0".join((dataset, split, sample_id, image_sha)).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.split_root.resolve()

    root_rows: list[dict[str, Any]] = json.loads((root / "manifest.json").read_text())
    train_rows = read_csv(root / "train_manifest.csv")
    test_rows = read_csv(root / "test_manifest.csv")
    c1_rows: list[dict[str, Any]] = json.loads((root / "test/C1/manifest.json").read_text())

    root_index = {(row["dataset"], row["id"]): row for row in root_rows}
    if len(root_index) != len(root_rows):
        raise RuntimeError("Root manifest has duplicate (dataset, id) keys")

    audit: dict[str, Any] = {
        "status": "running",
        "split_root": str(root),
        "path_mapping": {"from": OLD_ROOT, "to": str(NEW_ROOT), "mode": "explicit prefix mapping; no filename guessing"},
        "protocol_note": "Local reconstructed split; not the unavailable author sample-identity protocol.",
        "splits": {},
        "dataset_counts": {},
        "duplicate_content": {},
        "group_overlap": {},
        "nist16": {},
        "errors": [],
    }

    expected_main = {"train": train_rows, "test": test_rows}
    seen_sample_keys: set[str] = set()
    split_hashes: dict[str, dict[str, set[str]]] = {
        "train": {"image_sha256": set(), "pixel_sha256": set()},
        "test": {"image_sha256": set(), "pixel_sha256": set()},
    }
    per_dataset: dict[str, Counter] = defaultdict(Counter)
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"train": set(), "test": set()})
    ids_by_group: dict[tuple[str, str, str], dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
    image_hash_to_ids: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
    pixel_hash_to_ids: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})

    for split, csv_rows in expected_main.items():
        counters = Counter()
        csv_index = {(row["dataset"], row["id"]): row for row in csv_rows}
        if len(csv_index) != len(csv_rows):
            audit["errors"].append(f"{split}: duplicate CSV identity")
        for csv_row in csv_rows:
            dataset, sample_id = csv_row["dataset"], csv_row["id"]
            key = (dataset, sample_id)
            record = root_index.get(key)
            if record is None:
                audit["errors"].append(f"{split}: missing root manifest identity {key}")
                counters["missing_manifest_record"] += 1
                continue
            if record.get("split") != split or csv_row.get("split") != split:
                audit["errors"].append(f"{split}: split mismatch for {key}")
            if csv_row.get("split_image") != record.get("split_image") or csv_row.get("split_mask") != record.get("split_mask"):
                audit["errors"].append(f"{split}: split path does not match root manifest for {key}")

            image_path = resolve_path(record["image"])
            source_mask_path = resolve_path(record["mask"])
            prepared_mask_path = resolve_path(record["prepared_mask"])
            declared_image_link = resolve_path(record["split_image"])
            declared_mask_link = resolve_path(record["split_mask"])

            # The historical processed image is an explicit symlink to prepared_image;
            # follow its recorded source identity, then verify bytes using manifest SHA.
            split_image_path = resolve_path(record["split_image"])
            try:
                image_link_target = split_image_path.readlink()
                if str(image_link_target) != record["prepared_image"]:
                    audit["errors"].append(f"{split}: image symlink target differs from manifest for {key}")
            except OSError:
                # On C1 no split image links are readable in this runtime; the exact
                # source image path and SHA are instead declared in its own manifest.
                pass
            try:
                mask_link_target = declared_mask_link.readlink()
                if str(mask_link_target) != record["prepared_mask"]:
                    audit["errors"].append(f"{split}: mask symlink target differs from manifest for {key}")
            except OSError:
                pass

            try:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                image_array, image_size = decode_image(image_path, "RGB")
                counters["image_decoded"] += 1
                image_hash = sha256(image_path)
                if image_hash != record["image_sha256"]:
                    audit["errors"].append(f"{split}: source image SHA mismatch for {key}")
                if image_size != (record["width"], record["height"]):
                    audit["errors"].append(f"{split}: image dimension metadata mismatch for {key}: {image_size}")
                image_hash_to_ids[image_hash][split].append(f"{dataset}/{sample_id}")
                if record.get("pixel_sha256"):
                    split_hashes[split]["pixel_sha256"].add(record["pixel_sha256"])
                    pixel_hash_to_ids[record["pixel_sha256"]][split].append(f"{dataset}/{sample_id}")
                split_hashes[split]["image_sha256"].add(image_hash)
            except Exception as exc:
                counters["image_missing_or_invalid"] += 1
                audit["errors"].append(f"{split}: image failure for {key}: {type(exc).__name__}: {exc}")
                image_array = None
                image_size = None

            try:
                if not source_mask_path.is_file():
                    raise FileNotFoundError(source_mask_path)
                source_mask_hash = sha256(source_mask_path)
                if source_mask_hash != record["mask_sha256"]:
                    audit["errors"].append(f"{split}: raw/source mask SHA mismatch for {key}")
                source_mask, source_mask_size = decode_image(source_mask_path)
                counters["source_mask_decoded"] += 1
                if source_mask_size != (record["mask_size"][0], record["mask_size"][1]):
                    audit["errors"].append(f"{split}: source mask metadata size mismatch for {key}")
            except Exception as exc:
                counters["source_mask_missing_or_invalid"] += 1
                audit["errors"].append(f"{split}: source mask failure for {key}: {type(exc).__name__}: {exc}")

            try:
                if not prepared_mask_path.is_file():
                    raise FileNotFoundError(prepared_mask_path)
                mask_array, mask_size = decode_image(prepared_mask_path, "L")
                counters["prepared_gt_decoded"] += 1
                if image_size and mask_size != image_size:
                    audit["errors"].append(f"{split}: image/GT dimensions differ for {key}: {image_size} vs {mask_size}")
                unique_values = np.unique(mask_array)
                if not set(map(int, unique_values)).issubset({0, 255}):
                    audit["errors"].append(f"{split}: non-binary prepared GT for {key}: {unique_values[:16].tolist()}")
                counters["gt_binary"] += int(set(map(int, unique_values)).issubset({0, 255}))
            except Exception as exc:
                counters["gt_missing_or_invalid"] += 1
                audit["errors"].append(f"{split}: prepared GT failure for {key}: {type(exc).__name__}: {exc}")

            sample_id_key = sample_key(dataset, split, sample_id, record["image_sha256"])
            if sample_id_key in seen_sample_keys:
                audit["errors"].append(f"Duplicate stable sample key: {dataset}/{split}/{sample_id}")
            seen_sample_keys.add(sample_id_key)
            group = str(record.get("group", ""))
            groups[dataset][split].add(group)
            ids_by_group[(dataset, group, split)][split].append(sample_id)
            per_dataset[dataset][split] += 1
            counters["records"] += 1

        expected_files_by_dir: dict[Path, set[str]] = defaultdict(set)
        for csv_row in csv_rows:
            for field in ("split_image", "split_mask"):
                expected_path = resolve_path(csv_row[field])
                expected_files_by_dir[expected_path.parent].add(expected_path.name)
        directory_entries = {}
        for directory, expected_names in expected_files_by_dir.items():
            actual_names = {
                item.name for item in directory.iterdir()
                if item.is_file() or item.is_symlink()
            }
            missing_names = sorted(expected_names - actual_names)
            extra_names = sorted(actual_names - expected_names)
            directory_entries[str(directory)] = {
                "expected": len(expected_names),
                "actual": len(actual_names),
                "missing": missing_names,
                "extra": extra_names,
            }
            if missing_names or extra_names:
                audit["errors"].append(
                    f"{split}: directory membership mismatch at {directory}: "
                    f"missing={len(missing_names)} extra={len(extra_names)}"
                )

        audit["splits"][split] = {
            "manifest_records": len(csv_rows),
            "checks": dict(counters),
            "directory_entries": directory_entries,
        }

    # C1 is a separate 920-image locked test set and must never appear in train.
    c1_counts = Counter()
    c1_image_hashes: set[str] = set()
    c1_pixel_hashes: set[str] = set()
    c1_id_seen: set[str] = set()
    for record in c1_rows:
        sample_id = record["id"]
        if sample_id in c1_id_seen:
            audit["errors"].append(f"C1: duplicate id {sample_id}")
        c1_id_seen.add(sample_id)
        image_path = resolve_path(record["image"])
        source_mask_path = resolve_path(record["mask"])
        split_mask_path = resolve_path(record["split_mask"])
        identity = ("CASIAv1", "test", sample_id)
        try:
            image_array, image_size = decode_image(image_path, "RGB")
            image_hash = sha256(image_path)
            if image_hash != record["image_sha256"]:
                audit["errors"].append(f"C1: image SHA mismatch for {sample_id}")
            if image_size != (record["width"], record["height"]):
                audit["errors"].append(f"C1: image dimensions mismatch for {sample_id}")
            c1_image_hashes.add(image_hash)
            if record.get("pixel_sha256"):
                c1_pixel_hashes.add(record["pixel_sha256"])
            c1_counts["image_decoded"] += 1
        except Exception as exc:
            c1_counts["image_missing_or_invalid"] += 1
            audit["errors"].append(f"C1: image failure for {sample_id}: {type(exc).__name__}: {exc}")
            image_size = None
        try:
            source_mask, source_mask_size = decode_image(source_mask_path, "L")
            split_mask, split_mask_size = decode_image(split_mask_path, "L")
            expected_mask = np.where(source_mask > 127, 255, 0).astype(np.uint8)
            if not np.array_equal(expected_mask, split_mask):
                audit["errors"].append(f"C1: split GT differs from manifest threshold rule for {sample_id}")
            if image_size and split_mask_size != image_size:
                audit["errors"].append(f"C1: image/GT dimensions differ for {sample_id}")
            if not set(map(int, np.unique(split_mask))).issubset({0, 255}):
                audit["errors"].append(f"C1: non-binary GT for {sample_id}")
            c1_counts["source_gt_decoded"] += 1
            c1_counts["split_gt_decoded"] += 1
        except Exception as exc:
            c1_counts["gt_missing_or_invalid"] += 1
            audit["errors"].append(f"C1: GT failure for {sample_id}: {type(exc).__name__}: {exc}")
    c1_expected_files: dict[Path, set[str]] = defaultdict(set)
    for record in c1_rows:
        for field in ("split_image", "split_mask"):
            expected_path = resolve_path(record[field])
            c1_expected_files[expected_path.parent].add(expected_path.name)
    c1_directory_entries = {}
    for directory, expected_names in c1_expected_files.items():
        actual_names = {
            item.name for item in directory.iterdir()
            if item.is_file() or item.is_symlink()
        }
        missing_names = sorted(expected_names - actual_names)
        extra_names = sorted(actual_names - expected_names)
        c1_directory_entries[str(directory)] = {
            "expected": len(expected_names),
            "actual": len(actual_names),
            "missing": missing_names,
            "extra": extra_names,
        }
        if missing_names or extra_names:
            audit["errors"].append(
                f"C1: directory membership mismatch at {directory}: "
                f"missing={len(missing_names)} extra={len(extra_names)}"
            )

    train_image_hashes = split_hashes["train"]["image_sha256"]
    train_pixel_hashes = split_hashes["train"]["pixel_sha256"]
    c1_train_image_overlap = sorted(train_image_hashes & c1_image_hashes)
    c1_train_pixel_overlap = sorted(train_pixel_hashes & c1_pixel_hashes)
    audit["splits"]["C1_test"] = {
        "manifest_records": len(c1_rows),
        "checks": dict(c1_counts),
        "directory_entries": c1_directory_entries,
    }

    for dataset in sorted(set(row["dataset"] for row in root_rows)):
        audit["dataset_counts"][dataset] = {
            "train": per_dataset[dataset]["train"],
            "test": per_dataset[dataset]["test"],
        }
    audit["dataset_counts"]["CASIAv1"] = {"train": 0, "test": len(c1_rows)}

    exact_image_dupes = [
        {"sha256": digest, "train_ids": ids["train"], "test_ids": ids["test"]}
        for digest, ids in image_hash_to_ids.items() if ids["train"] and ids["test"]
    ]
    exact_pixel_dupes = [
        {"sha256": digest, "train_ids": ids["train"], "test_ids": ids["test"]}
        for digest, ids in pixel_hash_to_ids.items() if ids["train"] and ids["test"]
    ]
    all_records = root_rows + c1_rows
    all_image_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_pixel_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        all_image_groups[row["image_sha256"]].append(row)
        all_pixel_groups[row["pixel_sha256"]].append(row)

    def duplicate_summaries(groups_by_hash: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_duplicates = []
        within_split_duplicates = []
        for digest, records in groups_by_hash.items():
            if len(records) < 2:
                continue
            identities = [
                f"{row['dataset']}/{row.get('split', 'test')}/{row['id']}"
                for row in records
            ]
            item = {"sha256": digest, "samples": identities}
            all_duplicates.append(item)
            if len({row.get("split", "test") for row in records}) == 1:
                within_split_duplicates.append(item)
        return all_duplicates, within_split_duplicates

    all_image_duplicates, within_split_image_duplicates = duplicate_summaries(all_image_groups)
    all_pixel_duplicates, within_split_pixel_duplicates = duplicate_summaries(all_pixel_groups)
    audit["duplicate_content"] = {
        "train_test_exact_byte_image_overlap": len(exact_image_dupes),
        "train_test_decoded_pixel_overlap": len(exact_pixel_dupes),
        "exact_image_examples": exact_image_dupes[:20],
        "exact_pixel_examples": exact_pixel_dupes[:20],
        "all_splits_exact_image_duplicate_groups": len(all_image_duplicates),
        "all_splits_pixel_duplicate_groups": len(all_pixel_duplicates),
        "within_split_exact_image_duplicate_groups": len(within_split_image_duplicates),
        "within_split_pixel_duplicate_groups": len(within_split_pixel_duplicates),
        "within_split_exact_image_examples": within_split_image_duplicates[:20],
        "within_split_pixel_examples": within_split_pixel_duplicates[:20],
        "C1_test_vs_train_exact_image_overlap": len(c1_train_image_overlap),
        "C1_test_vs_train_pixel_sha_overlap": len(c1_train_pixel_overlap),
    }

    group_overlaps = {}
    for dataset, values in groups.items():
        overlap = sorted(values["train"] & values["test"])
        if overlap:
            group_overlaps[dataset] = overlap
    audit["group_overlap"] = group_overlaps
    nist_rows = [row for row in root_rows if row["dataset"] == "NIST16"]
    nist_train_ids = {row["id"] for row in nist_rows if row["split"] == "train"}
    nist_test_ids = {row["id"] for row in nist_rows if row["split"] == "test"}
    nist_cross_groups = sorted(groups.get("NIST16", {"train": set(), "test": set()})["train"] & groups.get("NIST16", {"train": set(), "test": set()})["test"])
    nist_group_details = []
    for group in nist_cross_groups:
        nist_group_details.append({
            "group": group,
            "train_ids": sorted(row["id"] for row in nist_rows if row["split"] == "train" and row.get("group") == group),
            "test_ids": sorted(row["id"] for row in nist_rows if row["split"] == "test" and row.get("group") == group),
        })
    audit["nist16"] = {
        "train_records": len(nist_train_ids),
        "test_records": len(nist_test_ids),
        "cross_split_source_group_count": len(nist_cross_groups),
        "cross_split_source_groups": nist_group_details,
        "exact_image_overlap": sum(1 for item in exact_image_dupes if any(x.startswith("NIST16/") for x in item["train_ids"] + item["test_ids"])),
        "recommendation": "Keep the historical local split locked for comparison and report its known base-source overlap. Create any group-isolated research split as a separately named manifest with a new experiment identity; do not rewrite this split and do not claim exact author-protocol identity.",
    }
    audit["protocol_distinction"] = {
        "author_protocol": "Paper counts are known, but author sample identity lists are unavailable locally.",
        "local_reconstruction": "Existing 5737/230 split plus separately added 920-image CASIAv1 test manifest; one NIST16 source group crosses train/test.",
        "corrected_isolation_protocol": "Not created in stage 2A; would require a new, separately versioned group-disjoint manifest and must not replace the comparison split.",
    }

    expected_counts = {"train": 5737, "test": 230, "C1_test": 920}
    for split, expected in expected_counts.items():
        actual = audit["splits"][split]["manifest_records"] if split in audit["splits"] else len(c1_rows)
        if actual != expected:
            audit["errors"].append(f"{split}: expected {expected} manifest entries, got {actual}")
    audit["status"] = "passed" if not audit["errors"] else "blocked"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": audit["status"],
        "output": str(args.output),
        "splits": audit["splits"],
        "dataset_counts": audit["dataset_counts"],
        "duplicate_content": audit["duplicate_content"],
        "group_overlap": audit["group_overlap"],
        "nist16": audit["nist16"],
        "error_count": len(audit["errors"]),
        "errors_sample": audit["errors"][:30],
    }, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
