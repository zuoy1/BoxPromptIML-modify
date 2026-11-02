#!/usr/bin/env python3
"""Verify the restored A-server project layout on the B server.

Run with the project conda environment:
    /path/to/envs/boxpromptiml/bin/python migration/verify_paths.py

Default checks are fast (stat/count only). Add --hash to verify large SHA256
artifacts (SAM 2.4G, TinyViT 22M, old checkpoint-20, boxes, and the four
stage-4F1 final.pt files). The 4F1 run outputs are expected under
/hy-tmp/BoxPromtIML/fecd_runs by default; override with --fecd-runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_FECD_RUNS = Path("/hy-tmp/BoxPromtIML/fecd_runs")

EXPECTED = {
    "sam": 2564550879,
    "tinyvit": 21712177,
    "checkpoint20": 23491814,
    "boxes": 4577020,
    "stage4f1_final": 70991310,
    "stage4p_final": 70991246,
    "stage4f0_epoch1": 70991054,
    "stage4f1_partial": 70991310,
}
SHA256 = {
    "sam": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    "tinyvit": "e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d",
    "checkpoint20": "7e54d4c155014fa7289b945f32fea6989583dd7dc58952348c75248a3e35f892",
    "boxes": "2461ca0b6cb8a16496947773797c71954b61ccf7540ef1549e68afb69b38493c",
}
STAGE4F1_SHA256 = {
    "P0": "9ea55e01e5e354d252fb6470c7a3aa04712840eb657917e694c19f56650bd1ea",
    "P1": "b0e56a4eb4ffc361f80476af580b30c2351ad885bea10f251b08ae119cdf4d89",
    "P2F": "1d1997cd71b8ff82b5e42debdf5e1e9e8ea659b8c1194c231481c3e0088a9404",
    "P3F": "8c6dc3512bdebaf0885b85dbe9b65af737b804f90e21238405073cbe4fa11ea6",
}
SPLITS = {
    "train/Tp": ("/home/datasets/processed/boxprompt_paper_scale_20261016/train/Tp", 5737),
    "train/Gt": ("/home/datasets/processed/boxprompt_paper_scale_20261016/train/Gt", 5737),
    "test/NC16/Tp": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/NC16/Tp", 150),
    "test/NC16/Gt": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/NC16/Gt", 150),
    "test/C1/Tp": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/C1/Tp", 920),
    "test/C1/Gt": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/C1/Gt", 920),
    "test/Columbia/Tp": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/Columbia/Tp", 50),
    "test/Columbia/Gt": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/Columbia/Gt", 50),
    "test/Coverage/Tp": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/Coverage/Tp", 30),
    "test/Coverage/Gt": ("/home/datasets/processed/boxprompt_paper_scale_20261016/test/Coverage/Gt", 30),
}

failures: list[str] = []
warnings: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("PASS  " if ok else "FAIL  ") + message)
    if not ok:
        failures.append(message)


def warn(ok: bool, message: str) -> None:
    if not ok:
        print("WARN  " + message)
        warnings.append(message)


def count_dir(path: str, expected: int, label: str) -> None:
    p = Path(path)
    if not p.is_dir():
        check(False, f"{label}: missing directory {path}")
        return
    entries = list(os.scandir(p))
    check(len(entries) == expected, f"{label}: {len(entries)} entries (expected {expected})")
    broken = [e.name for e in entries if e.is_symlink() and not e.is_file()]
    check(not broken, f"{label}: no broken symlinks" + ("" if not broken else f" ({broken[:3]})"))
    links = sum(1 for e in entries if e.is_symlink())
    warn(links > 0 or expected == 0, f"{label}: symlinks were preserved ({links} links)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_checkpoint(path: Path, expected_size: int, label: str) -> None:
    check(path.is_file(), f"{label} exists: {path}")
    if path.is_file():
        check(path.stat().st_size == expected_size,
              f"{label} size ({path.stat().st_size}, expected {expected_size})")


def check_optional_checkpoint(path: Path, expected_size: int, label: str) -> None:
    if not path.is_file():
        warn(False, f"{label} is absent (optional P1 asset): {path}")
        return
    check(path.stat().st_size == expected_size,
          f"{label} size ({path.stat().st_size}, expected {expected_size})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="verify SHA256 of large artifacts")
    parser.add_argument("--fecd-runs", type=Path, default=DEFAULT_FECD_RUNS,
                        help=f"root containing stage4f1/ and stage5a/ (default: {DEFAULT_FECD_RUNS})")
    args = parser.parse_args()

    print(f"project: {PROJECT}")
    check(PROJECT.is_dir(), "project directory exists")

    for rel in ("train.py", "eval.py", "boxpromptiml.py", "utils/data_val.py",
                "TinyViT/models/tiny_vit.py", "fecd/tests/run_stage2b_tests.py",
                "fecd/tools/run_stage4f1_autoshutdown.py",
                "fecd/tools/check_stage4f1_v2.py",
                "fecd/training/f1_runner.py",
                "fecd/diagnostics/stage5a_io.py",
                "fecd/docs/stage4f1_training_report_20261030.md",
                "fecd/docs/stage5a_failure_diagnosis.md"):
        check((PROJECT / rel).is_file(), f"code file exists: {rel}")

    check((PROJECT / "fecd").is_dir(), "fecd/ exists")
    check((PROJECT / "MIGRATION.md").is_file(), "MIGRATION.md exists")

    # Path aliases: both historical prefixes must resolve to the restored roots.
    for alias, real in (
        ("/hy-tmp/datasets", "/home/datasets"),
        ("/hy-tmp/weights", "/home/weights"),
        ("/hy-tmp/runs", "/home/runs"),
    ):
        a, r = Path(alias), Path(real)
        check(a.exists(), f"alias exists: {alias} -> {real}")
        try:
            check(a.exists() and r.exists() and os.path.samefile(a, r),
                  f"alias points to the same data: {alias} -> {real}")
        except OSError as exc:
            check(False, f"alias comparison failed: {alias} ({exc})")
    warn(Path("/hy-tmp/projects").exists(), "optional alias /hy-tmp/projects is absent")

    # Dataset split membership and symlink health.
    for label, (path, expected) in SPLITS.items():
        count_dir(path, expected, label)
    dataset_links = [p for p in Path("/home/datasets").rglob("*") if p.is_symlink()]
    check(len(dataset_links) == 30787,
          f"dataset symlink total ({len(dataset_links)}, expected 30787)")
    broken_dataset = [str(p) for p in dataset_links if not p.exists()]
    check(not broken_dataset,
          "all dataset symlinks resolve" + ("" if not broken_dataset else f" ({broken_dataset[:3]})"))

    # Weights and old checkpoints.
    sam = PROJECT / "segment-anything/sam_vit_h_4b8939.pth"
    check(sam.exists(), f"SAM checkpoint resolves: {sam}")
    if sam.exists():
        check(sam.stat().st_size == EXPECTED["sam"], f"SAM checkpoint size ({sam.stat().st_size})")
        warn(sam.is_symlink(), "SAM checkpoint is a real file, not the historical symlink")

    tiny = Path("/home/weights/tiny_vit_5m_22kto1k_distill.pth")
    check(tiny.is_file(), f"TinyViT checkpoint exists: {tiny}")
    if tiny.is_file():
        check(tiny.stat().st_size == EXPECTED["tinyvit"], f"TinyViT checkpoint size ({tiny.stat().st_size})")
    hub = Path.home() / ".cache/torch/hub/checkpoints/tiny_vit_5m_22kto1k_distill.pth"
    check(hub.exists(), f"TinyViT torch-hub cache resolves: {hub}")
    if hub.exists():
        check(hub.stat().st_size == EXPECTED["tinyvit"], "TinyViT torch-hub cache size")

    ckpt_dir = Path("/home/runs/boxprompt_formal_20261016_122447/checkpoints")
    check(ckpt_dir.is_dir(), f"formal checkpoint directory exists: {ckpt_dir}")
    check(len(list(ckpt_dir.glob("checkpoint-*.pth"))) == 20, "20 formal checkpoints are present")
    ckpt20 = ckpt_dir / "checkpoint-20.pth"
    check_checkpoint(ckpt20, EXPECTED["checkpoint20"], "checkpoint-20")

    # FECD boxes and stage-2C caches.
    boxes = PROJECT / "fecd/assets/boxes_train.json"
    check(boxes.is_file(), f"FECD boxes file exists: {boxes}")
    if boxes.is_file():
        payload = json.loads(boxes.read_text())
        records = payload.get("records", [])
        check(len(records) == 5737, f"FECD boxes records ({len(records)})")
        missing = [r["sample_id"] for r in records if not Path(r["image_path"]).exists()]
        check(not missing, "FECD box image paths resolve" + ("" if not missing else f" ({missing[:3]})"))

    expected_hashes = {
        "k1": "d7ccb769414fd8beb943888c81239e77e14e799c6300377673bbc2b6680f159b",
        "k8": "2bff688546a79079dcbac4ae29bb2c5169823704025c69b746a8cb57df99aafd",
    }
    for k, expected_hash in expected_hashes.items():
        roots = list((PROJECT / "fecd/assets/stage2c_cache" / k).glob("*/config.json"))
        check(len(roots) == 1, f"FECD {k} cache config is present")
        if not roots:
            continue
        cfg = json.loads(roots[0].read_text())
        check(cfg.get("config_hash") == expected_hash, f"FECD {k} config hash matches")
        npz = list(roots[0].parent.glob("*.npz"))
        check(len(npz) == 5737, f"FECD {k} cache files ({len(npz)})")
        index = roots[0].parent / "dataset_index.json"
        check(index.is_file(), f"FECD {k} dataset index exists")
        if index.is_file():
            rows = json.loads(index.read_text())
            check(len(rows) == 5737, f"FECD {k} dataset index rows ({len(rows)})")
            bad = [r["cache_path"] for r in rows if not (PROJECT / r["cache_path"]).exists()]
            check(not bad, f"FECD {k} index cache paths resolve" + ("" if not bad else f" ({bad[:2]})"))

    # Stage 3/4 assets. These are optional for running 4F1 but required to
    # reproduce the complete research record.
    evidence = PROJECT / "fecd/assets/stage3_evidence"
    warn(evidence.is_dir(), f"stage3 evidence cache is absent: {evidence}")
    if evidence.is_dir():
        audit = evidence / "stage3_evidence_cache_audit.json"
        check(audit.is_file(), "stage3 evidence audit exists")
        if audit.is_file():
            data = json.loads(audit.read_text())
            check(data.get("succeeded") == 5737 and data.get("failed") == [],
                  "stage3 evidence audit passed (5737/5737, no failures)")
        npz = list((evidence / "raw").glob("*/*.npz"))
        check(len(npz) == 5737, f"stage3 evidence NPZ files ({len(npz)})")

    split_audit = PROJECT / "fecd/assets/stage3ra_split_audit.json"
    check(split_audit.is_file(), f"stage3ra split audit exists: {split_audit}")
    if split_audit.is_file():
        data = json.loads(split_audit.read_text())
        check(data.get("status") == "passed", "stage3ra split audit status")
        check(data.get("counts") == {"student_train": 3530, "new_dev": 530, "new_confirm": 530},
              "stage3ra split counts (3530/530/530)")

    for variant, size in (("P0", "stage4p_final"), ("P1", "stage4p_final"),
                          ("P2", "stage4p_final"), ("P3", "stage4p_final")):
        check_optional_checkpoint(PROJECT / f"fecd/assets/stage4p/{variant}/final.pt",
                                  EXPECTED[size], f"stage4p {variant} final")
    for variant in ("P0", "P1", "P2F", "P3F"):
        check_optional_checkpoint(PROJECT / f"fecd/assets/stage4f0/{variant}/epoch1.pt",
                                  EXPECTED["stage4f0_epoch1"], f"stage4f0 {variant} epoch1")
    check_optional_checkpoint(PROJECT / "fecd/assets/stage4f1/P0/last.pt",
                              EXPECTED["stage4f1_partial"], "stage4f1 partial P0 last")

    revalidation = PROJECT / "fecd/assets/stage4f1_v/revalidation_20261030_v2/validation_v2.json"
    warn(revalidation.is_file(), f"stage4f1-v validation is absent (optional P1): {revalidation}")
    if revalidation.is_file():
        data = json.loads(revalidation.read_text())
        check(data.get("status") == "PASS" and data.get("engineering") == "PASS",
              "stage4f1-v engineering status PASS")
        check(data.get("research_conclusion") == "NO CLEAR GAIN",
              "stage4f1-v research conclusion is NO CLEAR GAIN")

    # Stage 4F1 formal run and stage 5A diagnostics live on the data disk.
    formal = args.fecd_runs / "stage4f1/formal20_seed20261029_noshutdown_20261029a"
    check(formal.is_dir(), f"stage4f1 formal run exists: {formal}")
    for variant, expected_hash in STAGE4F1_SHA256.items():
        p = formal / f"artifacts/{variant}/final.pt"
        check_checkpoint(p, EXPECTED["stage4f1_final"], f"stage4f1 {variant} final")
        if args.hash and p.is_file():
            print(f"INFO  hashing stage4f1 {variant}: {p}")
            check(sha256(p) == expected_hash, f"stage4f1 {variant} final SHA256 matches")
    status = formal / "status.json"
    check(status.is_file(), f"stage4f1 status.json exists: {status}")
    if status.is_file():
        data = json.loads(status.read_text())
        check(data.get("status") == "FAILED",
              "stage4f1 original status is still FAILED (expected; see 4F1-V revalidation)")

    stage5a = args.fecd_runs / "stage5a/diagnosis_20261030"
    warn(stage5a.is_dir(), f"stage5a diagnosis is absent: {stage5a}")
    if stage5a.is_dir():
        validation = stage5a / "validation_final.json"
        check(validation.is_file(), "stage5a validation_final.json exists")
        if validation.is_file():
            data = json.loads(validation.read_text())
            check(data.get("status") == "PASS", "stage5a validation status PASS")

    # Python environment (run this script with the project interpreter).
    print(f"python: {sys.version.split()[0]}")
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import cv2  # noqa: F401
        import timm  # noqa: F401
        import albumentations  # noqa: F401
        import sklearn  # noqa: F401
        import tensorboardX  # noqa: F401
        import thop  # noqa: F401
        import segment_anything  # noqa: F401
        check(True, "import torch/torchvision/cv2/timm/albumentations/sklearn/tensorboardX/thop/segment_anything")
        print(f"torch: {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    except Exception as exc:  # noqa: BLE001
        check(False, f"import runtime dependencies: {type(exc).__name__}: {exc}")

    if args.hash:
        for label, path, expected in (
            ("SAM", sam, SHA256["sam"]),
            ("TinyViT", tiny, SHA256["tinyvit"]),
            ("checkpoint-20", ckpt20, SHA256["checkpoint20"]),
            ("boxes", boxes, SHA256["boxes"]),
            ("stage4f1-v validation", revalidation,
             "97af9005e6f611bfdc7df31f859babfe32dea8b2b256fa3c8c9828e4b35d7a42"),
        ):
            if path.is_file():
                print(f"INFO  hashing {label}: {path}")
                check(sha256(path) == expected, f"{label} SHA256 matches")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} checks failed)")
        return 1
    print(f"RESULT: PASS (warnings: {len(warnings)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
