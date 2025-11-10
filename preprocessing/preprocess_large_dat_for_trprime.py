#!/usr/bin/env python3
"""
Utility to convert large OTA IQ captures into TPrime-compatible datasets.

Example:
    python preprocess_large_dat_for_trprime.py \
        --src /mnt/dm-3/DroneDetect/DroneDetectV2_byModel \
        --condition BLUE \
        --mode FY \
        --samples-per-file 8192 \
        --num-examples-per-class 500
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.io import savemat


VALID_MODES = {"ON", "HO", "FY"}
VALID_CONDITIONS = {"BLUE", "WIFI", "BOTH", "CLEAN"}
SUPPORTED_EXT = {".dat", ".bin"}


@dataclass
class Segment:
    samples: np.ndarray
    src_name: str
    segment_idx: int
    start_idx: int
    end_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk OTA IQ files into TPrime dataset folders."
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="Root directory containing class subfolders (e.g., /mnt/.../DroneDetectV2_byModel).")
    parser.add_argument("--mode", required=True, nargs='+', choices=sorted(VALID_MODES),
                            help="Acquisition modes to include (ON, HO, FY).")
    parser.add_argument("--condition", required=True, nargs='+', choices=sorted(VALID_CONDITIONS),
                        help="Capture conditions to include (BLUE, WIFI, BOTH, CLEAN).")
    parser.add_argument("--samples-per-file", required=True, type=int,
                        help="Number of complex IQ samples per exported .mat example.")
    parser.add_argument("--num-examples-per-class", required=True, type=int,
                        help="Training examples to write per class (evaluation set gets ceil(10%%) extra).")
    parser.add_argument("--out-root", type=Path, default=Path("data"),
                        help="Output root where CONDITION_MODE_NUM datasets are created (default: ./data).")
    parser.add_argument("--seed", type=int, default=4389, help="Seed used when shuffling segments.")
    return parser.parse_args()


def _normalize_case(token: str) -> str:
    return token.upper()


def gather_class_file_map(src_root: Path, conditions: Sequence[str], modes: Sequence[str]) -> Dict[str, List[Path]]:
    """
    Returns a mapping {class_name: [file_paths]} for every (condition, mode) pair provided.
    Supports folder layouts like:
        /SRC/CONDITION/CLASS_MODE/<files>
    as well as the legacy layout where classes live directly under SRC.
    """
    norm_conditions = [_normalize_case(c) for c in conditions]
    norm_modes = [_normalize_case(m) for m in modes]
    file_map: Dict[str, List[Path]] = defaultdict(list)

    found_new_layout = False
    for condition in norm_conditions:
        condition_dir = src_root / condition
        if not condition_dir.is_dir():
            continue
        found_new_layout = True
        for subdir in sorted(condition_dir.rglob("*")):
            if not subdir.is_dir():
                continue
            name_up = subdir.name.upper()
            for mode in norm_modes:
                mode_token = f"_{mode}"
                if not name_up.endswith(mode_token):
                    continue
                class_name = subdir.name[: -len(mode_token)].rstrip("_")
                files = [
                    f for f in sorted(subdir.rglob("*"))
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
                ]
                if files:
                    file_map[class_name].extend(files)
                break  # avoid double counting this subdir

    if not found_new_layout:
        for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            files = []
            for f in class_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXT:
                    continue
                name = f.name.upper()
                if not any(name.startswith(f"{cond}__") for cond in norm_conditions):
                    continue
                if not any(f"_{mode}__" in name for mode in norm_modes):
                    continue
                files.append(f)
            if files:
                file_map[class_dir.name].extend(sorted(files))

    if not file_map:
        raise RuntimeError(
            f"Could not locate any files for conditions {conditions} and modes {modes} under {src_root}"
        )

    # Deduplicate paths while keeping order
    deduped_map: Dict[str, List[Path]] = {}
    for class_name, paths in file_map.items():
        seen = set()
        ordered = []
        for path in paths:
            if path not in seen:
                ordered.append(path)
                seen.add(path)
        deduped_map[class_name] = ordered
    return deduped_map


def _complex_memmap(path: Path) -> np.memmap:
    if path.suffix.lower() == ".dat":
        floats = np.memmap(path, dtype=np.float32, mode="r")
        usable = (floats.size // 2) * 2
        if usable == 0:
            raise ValueError(f"{path} does not contain interleaved IQ samples.")
        return floats[:usable].view(np.complex64)
    if path.suffix.lower() == ".bin":
        return np.memmap(path, dtype=np.complex128, mode="r")
    raise ValueError(f"Unsupported extension {path.suffix} in {path}")


def _segments_from_file(
    file_path: Path,
    samples_per_file: int,
    max_segments: int,
) -> List[Segment]:
    if max_segments <= 0:
        return []
    complex_stream = _complex_memmap(file_path)
    available = complex_stream.shape[0] // samples_per_file
    take = min(available, max_segments)
    segs: List[Segment] = []
    for seg_idx in range(take):
        start = seg_idx * samples_per_file
        end = start + samples_per_file
        window = np.array(complex_stream[start:end], dtype=np.complex64)
        segs.append(Segment(window, file_path.stem, seg_idx, start, end))
    return segs


def _allocate_counts(total: int, n_files: int) -> List[int]:
    if n_files == 0:
        return []
    base = total // n_files
    remainder = total % n_files
    counts = [base] * n_files
    for i in range(remainder):
        counts[i] += 1
    return counts


def split_train_eval_segments(
    files: Sequence[Path],
    samples_per_file: int,
    train_total: int,
    eval_total: int,
) -> Tuple[List[Segment], List[Segment]]:
    n_files = len(files)
    train_counts = _allocate_counts(train_total, n_files)
    eval_counts = _allocate_counts(eval_total, n_files)
    train_segments: List[Segment] = []
    eval_segments: List[Segment] = []
    for file_path, tr_quota, ev_quota in zip(files, train_counts, eval_counts):
        total_quota = tr_quota + ev_quota
        segments = _segments_from_file(file_path, samples_per_file, total_quota)
        if len(segments) < total_quota:
            raise RuntimeError(
                f"File {file_path} only provided {len(segments)} segments "
                f"(needed {total_quota}) — consider reducing samples-per-file or quotas."
            )
        if tr_quota:
            train_segments.extend(segments[:tr_quota])
        if ev_quota:
            eval_segments.extend(segments[tr_quota:tr_quota + ev_quota])
    return train_segments, eval_segments


def save_segments(segments: Sequence[Segment], dest_dir: Path, class_name: str) -> None:
    class_dir = dest_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        out_name = (
            f"{seg.src_name}__seg{seg.segment_idx:05d}"
            f"__s{seg.start_idx}_e{seg.end_idx}.mat"
        )
        out_path = class_dir / out_name
        savemat(out_path, {"waveform": seg.samples[:, None]}, do_compression=False)


def main() -> None:
    args = parse_args()
    src_root = args.src.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source directory {src_root} does not exist.")
    if args.samples_per_file <= 0:
        raise ValueError("--samples-per-file must be > 0")
    if args.num_examples_per_class <= 0:
        raise ValueError("--num-examples-per-class must be > 0")

    cond_tag = "-".join(sorted([_normalize_case(c) for c in args.condition]))
    mode_tag = "-".join(sorted([_normalize_case(m) for m in args.mode]))
    dataset_prefix = f"{cond_tag}_{mode_tag}_{args.num_examples_per_class}"
    train_root = out_root / f"{dataset_prefix}_train"
    eval_count = int(math.ceil(args.num_examples_per_class * 0.1))
    eval_root = out_root / f"{dataset_prefix}_eval"
    rng = random.Random(args.seed)

    # Reset output directories if they already exist
    for root in (train_root, eval_root):
        if root.exists():
            print(f"[INFO] Removing existing dataset directory: {root}")
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    class_files = gather_class_file_map(src_root, args.condition, args.mode)
    print(f"Discovered {len(class_files)} classes with conditions={args.condition}, modes={args.mode}")
    print(f"Building datasets at {train_root} and {eval_root}")

    total_written = {"train": 0, "eval": 0}
    for class_name, candidate_files in class_files.items():
        print(f"\nProcessing class {class_name} ({len(candidate_files)} files)")
        need_total = args.num_examples_per_class + eval_count
        print(f"  Found {len(candidate_files)} matching files. Collecting {need_total} segments...")
        train_segments, eval_segments = split_train_eval_segments(
            candidate_files,
            args.samples_per_file,
            args.num_examples_per_class,
            eval_count,
        )
        save_segments(train_segments, train_root, class_name)
        save_segments(eval_segments, eval_root, class_name)
        total_written["train"] += len(train_segments)
        total_written["eval"] += len(eval_segments)
        print(f"  Saved {len(train_segments)} train and {len(eval_segments)} eval segments for {class_name}.")
        if len(train_segments) != args.num_examples_per_class:
            print(
                f"[WARN] Class {class_name}: requested {args.num_examples_per_class} train segments "
                f"but saved {len(train_segments)}."
            )
        if len(eval_segments) != eval_count:
            print(
                f"[WARN] Class {class_name}: requested {eval_count} eval segments "
                f"but saved {len(eval_segments)}."
            )

    print("\nDone.")
    print(f"Total train segments: {total_written['train']}")
    print(f"Total eval segments:  {total_written['eval']}")
    print("Datasets ready to pass via --raw_path in TPrime_transformer_train.py")


if __name__ == "__main__":
    main()
