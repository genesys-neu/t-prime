#!/usr/bin/env python3
from pathlib import Path
import re, math, shutil, sys
from collections import defaultdict

SRC_ROOT = Path("/mnt/dm-3/DroneDetect/processed_bursts")
DST_ROOT = Path("/mnt/dm-3/DroneDetect/eval_bursts")
DRY_RUN  = False  # set True to preview

# Example name:
# CLEAN__AIR_ON__AIR_0000_04_burst0237_part00_cf-14.305MHz_bw13.945MHz_9899S.npy
RGX = re.compile(
    # Accept either ..._burst####_partNN_... or ..._burst####_... (no part)
    r"""^(?P<stem>.+?)_burst(?P<burst>\d{4})(?:_part(?P<part>\d{2}))?_.+\.(?:npy|mat)$""",
    re.VERBOSE
)

def move_last_5_percent_in_leaf(leaf_dir: Path):
    # Group files by "source stem" (everything before _burst####)
    files = sorted([p for p in list(leaf_dir.glob("*.npy")) + list(leaf_dir.glob("*.mat")) if p.is_file()])
    if not files:
        return 0, 0

    groups = defaultdict(list)
    for f in files:
        m = RGX.match(f.name)
        if not m:
            # skip unexpected names
            continue
        stem = m.group("stem")              # e.g., CLEAN__AIR_ON__AIR_0000_04
        burst = int(m.group("burst"))       # e.g., 0237
        part_str = m.group("part")
        part = int(part_str) if part_str is not None else 0
        groups[stem].append((burst, part, f))

    moved_count = 0
    seen_count  = 0

    # mirror dest dir
    rel = leaf_dir.relative_to(SRC_ROOT)    # MODEL/NOISE/OP
    out_dir = (DST_ROOT / rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem, entries in groups.items():
        # Collect distinct bursts and sort
        bursts = sorted({b for (b, _, _) in entries})
        n_bursts = len(bursts)
        if n_bursts == 0:
            continue
        k = max(1, math.floor(0.05 * n_bursts))  # last 5% bursts, at least 1
        tail_bursts = set(bursts[-k:])           # e.g., {243, 244, 245, ...}

        # Move all parts belonging to those tail bursts
        for b, part, f in entries:
            seen_count += 1
            if b in tail_bursts:
                dst = out_dir / f.name
                if DRY_RUN:
                    print(f"[DRY] {f} -> {dst}")
                else:
                    shutil.move(str(f), str(dst))
                moved_count += 1

    return seen_count, moved_count

def main():
    if not SRC_ROOT.exists():
        print(f"Source root not found: {SRC_ROOT}", file=sys.stderr)
        sys.exit(1)
    total_seen = total_moved = 0
    # We want to process grouped by interference (noise) first, then operational mode, then model.
    models = [d.name for d in sorted(SRC_ROOT.iterdir()) if d.is_dir()]
    # collect all noise names across models
    noise_names = set()
    for m in models:
        mpath = SRC_ROOT / m
        for n in mpath.iterdir():
            if n.is_dir():
                noise_names.add(n.name)

    for noise in sorted(noise_names):
        # collect op names across models for this noise
        op_names = set()
        for m in models:
            nd = SRC_ROOT / m / noise
            if not nd.exists() or not nd.is_dir():
                continue
            for opd in nd.iterdir():
                if opd.is_dir():
                    op_names.add(opd.name)

        for op in sorted(op_names):
            for m in models:
                leaf = SRC_ROOT / m / noise / op
                if not leaf.exists() or not leaf.is_dir():
                    continue
                seen, moved = move_last_5_percent_in_leaf(leaf)
                total_seen += seen
                total_moved += moved
                print(f"{leaf} : moved {moved}/{seen} files (tail 5% per source)")
    print(f"\nDONE. Total moved: {total_moved} of {total_seen} files.")
    print(f"Eval root: {DST_ROOT}")

if __name__ == "__main__":
    main()
