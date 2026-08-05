from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def resolve_reference_frame_index(index: int, num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    resolved = int(index)
    if resolved < 0:
        resolved += num_frames
    if resolved < 0 or resolved >= num_frames:
        raise ValueError(
            "reference_frame_index must select one sampled frame; "
            f"got {index} for {num_frames} frames"
        )
    return resolved


def reference_first_order(num_frames: int, reference_frame_index: int) -> list[int]:
    reference = resolve_reference_frame_index(reference_frame_index, num_frames)
    return [reference] + [index for index in range(num_frames) if index != reference]


def reorder_reference_first(
    items: Sequence[T],
    reference_frame_index: int,
) -> list[T]:
    return [items[index] for index in reference_first_order(len(items), reference_frame_index)]


def parse_frame_index_spec(spec: str, num_frames: int) -> tuple[int, ...]:
    text = spec.strip().lower()
    if not text:
        raise ValueError("frame index spec must not be empty")
    if text == "all":
        return tuple(range(num_frames))
    if text.startswith("uniform:"):
        count = int(text.split(":", 1)[1])
        if count <= 0:
            raise ValueError(f"uniform frame count must be positive, got {count}")
        count = min(count, num_frames)
        return tuple((index * num_frames) // count for index in range(count))

    indices: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item and not item.startswith("-"):
            start_text, end_text = item.split("-", 1)
            start = resolve_reference_frame_index(int(start_text), num_frames)
            end = resolve_reference_frame_index(int(end_text), num_frames)
            if end < start:
                raise ValueError(f"Invalid frame-index range {item!r}: end < start")
            indices.extend(range(start, end + 1))
        else:
            indices.append(resolve_reference_frame_index(int(item), num_frames))
    if not indices:
        raise ValueError("frame index spec did not contain any indices")
    return tuple(dict.fromkeys(indices))


def resolve_first_frame_token_indices(
    spec: str | Sequence[int],
    num_frames: int,
) -> tuple[int, ...]:
    if isinstance(spec, str):
        indices = parse_frame_index_spec(spec, num_frames)
    else:
        indices = tuple(
            resolve_reference_frame_index(index, num_frames) for index in spec
        )
    if not indices:
        raise ValueError("first-frame token indices must not be empty")
    if 0 not in indices:
        raise ValueError("first-frame token indices must include input position 0")
    return tuple(dict.fromkeys(indices))
