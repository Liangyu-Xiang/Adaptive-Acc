#!/usr/bin/env python3
"""Render readable, non-pooled views from a full-token attention TIFF.

The numeric TIFF is preserved.  The renderer writes a 4096px log-scale whole
matrix and 10x10 native-resolution frame-pair tiles.  Tiles have the same
global log color mapping, so brightness remains comparable across frame pairs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("tiff", type=Path)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--overview-size", type=int, default=4096)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--tiles", action="store_true", help="Also write 100 native-resolution frame-pair tiles.")
    return p.parse_args()


def main() -> None:
    a = parse()
    raw = np.asarray(Image.open(a.tiff), dtype=np.float32) / 65535.0
    n = raw.shape[0]
    if raw.ndim != 2 or n != raw.shape[1] or n % a.frames:
        raise ValueError(f"expected square matrix divisible by frames, got {raw.shape}")
    # A fixed global log transform retains all token pairs and preserves
    # cross-tile comparability.  The 99.9 percentile avoids a few diagonal
    # peaks flattening all visible structure.
    vmax = float(np.quantile(raw, 0.999))
    display = np.log1p(raw / max(vmax, 1e-12)) / np.log(2.0)
    display = np.clip(display, 0.0, 1.0)
    image = Image.fromarray((display * 255).astype(np.uint8), mode="L")
    target_dir = a.output_dir or a.tiff.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    base = target_dir / a.tiff.with_suffix("").name
    overview = image.copy()
    overview.thumbnail((a.overview_size, a.overview_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(overview)
    scale = overview.width / n
    per_frame = n // a.frames
    for i in range(1, a.frames):
        pos = round(i * per_frame * scale)
        draw.line((pos, 0, pos, overview.height), fill=255, width=2)
        draw.line((0, pos, overview.width, pos), fill=255, width=2)
    overview.save(target_dir / f"{base.name}_log_overview.png")
    if not a.tiles:
        return
    tiles = target_dir / f"{base.name}_frame_tiles"
    tiles.mkdir(exist_ok=True)
    for query_frame in range(a.frames):
        for key_frame in range(a.frames):
            y0, y1 = query_frame * per_frame, (query_frame + 1) * per_frame
            x0, x1 = key_frame * per_frame, (key_frame + 1) * per_frame
            image.crop((x0, y0, x1, y1)).save(tiles / f"query_F{query_frame:02d}_key_F{key_frame:02d}.png")
    (tiles / "README.txt").write_text(
        "Each PNG is one query-frame x key-frame block at original per-token resolution. "
        "All tiles share the same log color scale within this global layer; brightness is comparable.\n"
    )


if __name__ == "__main__":
    main()
