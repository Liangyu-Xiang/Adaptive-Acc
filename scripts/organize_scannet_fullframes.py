#!/usr/bin/env python3
"""Create a model-agnostic ScanNet frame layout from SensReader output.

Large RGB/depth/pose files stay in the SensReader directory; the standard
``color/``, ``depth/`` and ``pose/`` trees contain relative symlinks only.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def index(path: Path) -> int:
    match = re.search(r"frame-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot obtain frame index from {path}")
    return int(match.group(1))


def link(source: Path, target: Path) -> None:
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw, output = args.raw_dir, args.output_dir
    colors = {index(path): path for path in raw.glob("frame-*.color.jpg")}
    depths = {index(path): path for path in raw.glob("frame-*.depth.png")}
    poses = {index(path): path for path in raw.glob("frame-*.pose.txt")}
    ids = sorted(set(colors) & set(depths) & set(poses))
    if not ids:
        raise RuntimeError(f"No complete RGB/depth/pose triplets in {raw}")
    for kind in ("color", "depth", "pose"):
        (output / kind).mkdir(parents=True, exist_ok=True)
    for frame_id in ids:
        name = f"{frame_id:06d}"
        link(colors[frame_id], output / "color" / f"{name}.jpg")
        link(depths[frame_id], output / "depth" / f"{name}.png")
        link(poses[frame_id], output / "pose" / f"{name}.txt")
    intrinsic = raw / "intrinsic"
    for source_name, target_name in (("intrinsic_color.txt", "intrinsics_color.txt"), ("intrinsic_depth.txt", "intrinsics_depth.txt")):
        source = intrinsic / source_name
        if source.is_file():
            link(source, output / target_name)
    print(f"{output.name}: linked {len(ids)} complete frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
