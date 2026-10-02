"""Read-only validation of reconstructed datasets and stored MD5 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

from coldddi.benchmark_data import pair_keys, validate_dataset


def artifact_paths(root: Path) -> list[Path]:
    """Files protected by a reconstruction checksum manifest (raw XML excluded)."""
    return sorted(p for directory in (root / "filtered", root / "enriched", root / "splits",
                                     root.parent / "outputs_full/annotations", root.parent / "annotations_sample")
                  for p in directory.rglob("*")
                  if p.is_file() and p.suffix in {".csv", ".parquet", ".json"})


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: Path) -> Path:
    """Record this reconstruction's files, not a claim of paper-reference identity."""
    paths = artifact_paths(root)
    if not paths or not any((root / "splits").glob("seed*/manifest.json")):
        raise ValueError(f"Cannot record checksums without completed splits: {root}")
    manifest = root / "splits/checksums.txt"
    manifest.write_text("".join(f"{file_md5(p)}  {p.relative_to(root.parent).as_posix()}\n" for p in paths),
                        encoding="utf-8")
    return manifest


def verify_checksums(root: Path, manifest: Path) -> int:
    """Reject changed, missing, extra or unsafe manifest entries; never rewrite it."""
    root = root.resolve()
    expected = {p.relative_to(root.parent).as_posix() for p in artifact_paths(root)}
    recorded = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{32})  (.+)", line)
        if not match:
            raise ValueError(f"Invalid checksum line in {manifest}")
        digest, name = match.groups()
        target = (root.parent / name).resolve()
        if not target.is_relative_to(root.parent) or name not in expected or name in recorded:
            raise ValueError(f"Unsafe, missing or duplicate checksum target: {name}")
        recorded.add(name)
        if file_md5(target) != digest.lower():
            raise ValueError(f"MD5 mismatch: {name}")
    if recorded != expected:
        raise ValueError(f"Checksum manifest is incomplete: {sorted(expected - recorded)}")
    return len(recorded)


def check_dataset(root: Path, ab_path: Path, *, strict: bool = False,
                  profile: str = "custom", reference_checksums: Path | None = None) -> dict:
    """Check every seed without regenerating data or loading a model."""
    root = root.resolve()
    manifests = sorted((root / "splits").glob("seed*/manifest.json"))
    if not manifests:
        raise ValueError(f"No per-seed split manifests found under {root}")
    checked = []
    summary = {}
    for manifest in manifests:
        match = re.fullmatch(r"seed(\d+)", manifest.parent.name)
        if not match:
            raise ValueError(f"Invalid seed directory: {manifest.parent}")
        seed = int(match.group(1))
        ds, ab, report = validate_dataset(root, ab_path, seed)
        positives = set(pair_keys(ds.edges, "ddi_edges.csv"))
        combined = set().union(*(set(pair_keys(frame, name)) for name, frame in ds.splits.items()))
        if combined != positives:
            raise ValueError(f"seed{seed}: positive splits do not cover the full DDI edge table")
        for name, counts in report["splits"].items():
            if counts["negative"] != counts["positive"]:
                raise ValueError(f"seed{seed}/{name}: expected 1:1 positive/negative counts")
        g1 = set(ds.splits.g1_drugs)
        for epoch in sorted((manifest.parent / "train_negatives").glob("epoch_*.parquet")):
            keys = set(pair_keys(pd.read_parquet(epoch), str(epoch)))
            if len(keys) != len(ds.splits.train) or keys & positives:
                raise ValueError(f"Invalid training negatives: {epoch}")
            if any(a not in g1 or b not in g1 for a, b in keys):
                raise ValueError(f"Training negatives outside G1: {epoch}")
        summary = {"drugs": len(ds.drugs), "positive_pairs": len(ds.edges),
                   "ddi_types": int(ds.edges.ddi_type.nunique()), "type_a_pairs": int(ab.has_key_entity.sum())}
        stats_file = root / "filtered/stats.json"
        if stats_file.exists():
            last = json.loads(stats_file.read_text(encoding="utf-8"))[-1]
            if (last["n_drugs"], last["n_edges"], last["n_types"]) != (
                summary["drugs"], summary["positive_pairs"], summary["ddi_types"]
            ):
                raise ValueError("filtered/stats.json does not match the actual tables")
        checked.append({"seed": seed, "splits": report["splits"], "warnings": report["warnings"]})
    expected = {"toy": (86, 1383, 24), "drugbank-5.1.13": (1900, 565731, 215)}
    if profile != "custom":
        actual = (summary["drugs"], summary["positive_pairs"], summary["ddi_types"])
        if actual != expected[profile]:
            raise ValueError(f"{profile} counts mismatch: expected {expected[profile]}, got {actual}")
    checksum_files = None
    if strict or reference_checksums is not None:
        if ab_path.resolve() not in {p.resolve() for p in artifact_paths(root)}:
            raise ValueError("Strict checks require the selected A/B annotations to be inside the checksummed release")
        checksum_files = verify_checksums(root, reference_checksums or root / "splits/checksums.txt")
    return {"data": str(root), **summary, "seeds": checked, "checksum_files": checksum_files,
            "paper_identity_verified": False}


def find_annotations(root: Path) -> Path:
    candidates = [path for path in (root.parent / "outputs_full/annotations/ab.parquet",
                                   root.parent / "annotations_sample/ab_sample.parquet") if path.is_file()]
    if len(candidates) > 1:
        raise ValueError(f"Multiple annotation releases beside {root}; use --ab-parquet to select one")
    if candidates:
        return candidates[0]
    raise ValueError(f"No matching A/B annotations found beside {root}; use --ab-parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Reconstruction output root or intermediate/ directory.")
    parser.add_argument("--ab-parquet", type=Path, help="Explicit matching annotations for a single dataset.")
    parser.add_argument("--strict", action="store_true", help="Also verify the stored MD5 manifest; never creates one.")
    parser.add_argument("--reference-checksums", type=Path, help="Independent MD5 manifest, for a single dataset.")
    parser.add_argument("--profile", choices=("custom", "toy", "drugbank-5.1.13"), default="custom",
                        help="Optional reference count check. Custom datasets need not match paper counts.")
    args = parser.parse_args(argv)
    try:
        base = args.data.resolve()
        index = base / "reconstruction.json"
        if index.is_file():
            entries = json.loads(index.read_text(encoding="utf-8"))["datasets"]
            roots = [(base / entry).resolve() for entry in entries]
            if not roots or len(roots) != len(set(roots)) or any(not p.is_relative_to(base) for p in roots):
                raise ValueError("Invalid dataset paths in reconstruction.json")
        else:
            roots = [base if (base / "filtered").is_dir() else base / "intermediate"]
        if len(roots) > 1 and (args.ab_parquet or args.reference_checksums or args.profile != "custom"):
            raise ValueError("Explicit annotations, profiles and reference checksums require a single dataset directory")
        for root in roots:
            report = check_dataset(root, args.ab_parquet or find_annotations(root), strict=args.strict,
                                   profile=args.profile, reference_checksums=args.reference_checksums)
            print(json.dumps(report, indent=2))
        print("Sanity checks passed. Stored checksums verify artifact integrity, not historical paper identity.")
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print(f"sanity_check: {exc}", file=sys.stderr)
        return 1
    return 0
