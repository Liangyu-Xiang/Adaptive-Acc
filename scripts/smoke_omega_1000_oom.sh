#!/usr/bin/env bash
# Check whether a 1000-frame 7Scenes forward pass succeeds or OOMs.
# OOM is recorded in summary.json and does not fail the shell script unless
# FAIL_ON_OOM=1 is set.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

CONDA_BIN=${CONDA_BIN:-conda}
CONDA_ENV=${CONDA_ENV:-omega_pro6000}
GPU=${GPU:-${1:-0}}
CHECKPOINT=${CHECKPOINT:-/data/mmc_lyxiang/3D/ckpts/vggt_omega_1b_512.pt}
DATA_ROOT=${DATA_ROOT:-/data/mmc_lyxiang/dataset/7scenes}
SEQUENCE=${SEQUENCE:-chess/seq-03}
NUM_FRAMES=${NUM_FRAMES:-1000}
SAMPLING_STRIDE=${SAMPLING_STRIDE:-1}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
METHODS=${METHODS:-baseline,u-m}
RETAIN_ONLY_CACHED_INTERMEDIATES=${RETAIN_ONLY_CACHED_INTERMEDIATES:-1}
VGGT_SDPA_BACKEND=${VGGT_SDPA_BACKEND:-flash}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/smoke_omega_1000_oom}
FAIL_ON_OOM=${FAIL_ON_OOM:-0}

mkdir -p "${OUTPUT_DIR}"

echo "repo=${repo_root}"
echo "conda_env=${CONDA_ENV}"
echo "gpu=${GPU}"
echo "checkpoint=${CHECKPOINT}"
echo "data_root=${DATA_ROOT}"
echo "sequence=${SEQUENCE}"
echo "num_frames=${NUM_FRAMES}"
echo "methods=${METHODS}"
echo "retain_only_cached_intermediates=${RETAIN_ONLY_CACHED_INTERMEDIATES}"
echo "vggt_sdpa_backend=${VGGT_SDPA_BACKEND}"
echo "fail_on_oom=${FAIL_ON_OOM}"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" \
VGGT_UM_TRITON=1 \
VGGT_SDPA_BACKEND="${VGGT_SDPA_BACKEND}" \
CHECKPOINT="${CHECKPOINT}" \
DATA_ROOT="${DATA_ROOT}" \
SEQUENCE="${SEQUENCE}" \
NUM_FRAMES="${NUM_FRAMES}" \
SAMPLING_STRIDE="${SAMPLING_STRIDE}" \
IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
METHODS="${METHODS}" \
RETAIN_ONLY_CACHED_INTERMEDIATES="${RETAIN_ONLY_CACHED_INTERMEDIATES}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FAIL_ON_OOM="${FAIL_ON_OOM}" \
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.sdpa import sdpa_runtime_status
from vggt_omega.utils.load_fn import load_and_preprocess_images


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def frame_index(path: Path) -> int:
    return int(path.name.split("-", 1)[1].split(".", 1)[0])


def select_images(sequence_dir: Path, num_frames: int, stride: int) -> list[Path]:
    images = sorted(sequence_dir.glob("frame-*.color.png"), key=frame_index)
    if stride <= 0:
        raise ValueError("SAMPLING_STRIDE must be positive")
    selected = images[::stride][:num_frames]
    if len(selected) < num_frames:
        raise RuntimeError(
            f"{sequence_dir} has only {len(selected)} frames after stride {stride}; "
            f"need {num_frames}"
        )
    return selected


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state = torch.load(checkpoint, mmap=True, **kwargs)
    except TypeError:
        state = torch.load(checkpoint, **kwargs)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def model_kwargs(method: str, retain_only_cached_intermediates: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "merge_ratio": 0.0,
        "retain_only_cached_intermediates": retain_only_cached_intermediates,
    }
    if method == "fastvggt":
        kwargs["merge_ratio"] = 0.9
    elif method == "u-m":
        kwargs.update(
            frame_fusion_mode="u-m",
            frame_fusion_recompute_layers="0,10,17",
            frame_fusion_lambda_cost=0.03,
            frame_fusion_temporal_window=4,
            frame_fusion_spatial_radius=2,
            frame_fusion_attention_variant="representative",
        )
    elif method != "baseline":
        raise ValueError(f"Unsupported method: {method}")
    return kwargs


def run_one(
    method: str,
    checkpoint: Path,
    image_paths: list[Path],
    image_resolution: int,
    retain_only_cached_intermediates: bool,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = images = predictions = None
    try:
        state = load_state_dict(checkpoint)
        model = VGGTOmega(**model_kwargs(method, retain_only_cached_intermediates)).eval()
        model.load_state_dict(state, strict=True)
        del state
        model = model.to(device)
        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode="max_size",
            image_resolution=image_resolution,
        ).to(device, non_blocking=True)
        with torch.inference_mode():
            predictions = model(images)
        torch.cuda.synchronize(device)
        debug = getattr(model.aggregator, "last_frame_fusion_debug", {}) or {}
        first_batch = (debug.get("batches") or [{}])[0]
        return {
            "method": method,
            "success": True,
            "oom": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
            "images_shape": tuple(images.shape),
            "output_keys": sorted(predictions),
            "pose_enc_shape": tuple(predictions["pose_enc"].shape),
            "depth_shape": tuple(predictions["depth"].shape),
            "frame_fusion_mode": debug.get("mode"),
            "frame_fusion_recompute_layers": debug.get("recompute_layers"),
            "frame_fusion_edge_score_backend": first_batch.get("edge_score_backend"),
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.synchronize(device)
        return {
            "method": method,
            "success": False,
            "oom": True,
            "error": "CUDA out of memory",
            "detail": str(exc),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
        }
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            torch.cuda.synchronize(device)
            return {
                "method": method,
                "success": False,
                "oom": True,
                "error": "CUDA out of memory",
                "detail": message,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
            }
        raise
    finally:
        del predictions, images, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = Path(os.environ["CHECKPOINT"])
    data_root = Path(os.environ["DATA_ROOT"])
    sequence = os.environ["SEQUENCE"]
    sequence_dir = data_root / sequence
    num_frames = int(os.environ["NUM_FRAMES"])
    stride = int(os.environ["SAMPLING_STRIDE"])
    image_resolution = int(os.environ["IMAGE_RESOLUTION"])
    methods = [item.strip() for item in os.environ["METHODS"].split(",") if item.strip()]
    retain_only_cached_intermediates = env_bool("RETAIN_ONLY_CACHED_INTERMEDIATES", True)
    fail_on_oom = env_bool("FAIL_ON_OOM", False)
    output_dir = Path(os.environ["OUTPUT_DIR"])

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)

    image_paths = select_images(sequence_dir, num_frames, stride)
    results = []
    fatal_error = None
    for method in methods:
        try:
            results.append(
                run_one(method, checkpoint, image_paths, image_resolution, retain_only_cached_intermediates)
            )
        except Exception as exc:
            fatal_error = {
                "method": method,
                "success": False,
                "oom": False,
                "error": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            }
            results.append(fatal_error)
            break

    summary = {
        "checkpoint": str(checkpoint),
        "sequence": str(sequence_dir),
        "num_frames": len(image_paths),
        "sampling_stride": stride,
        "first_frame": image_paths[0].name,
        "last_frame": image_paths[-1].name,
        "image_resolution": image_resolution,
        "retain_only_cached_intermediates": retain_only_cached_intermediates,
        "fail_on_oom": fail_on_oom,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "sdpa": sdpa_runtime_status(),
        "triton": {
            "package_available": importlib.util.find_spec("triton") is not None,
            "vggt_um_triton": os.environ.get("VGGT_UM_TRITON", "1"),
        },
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"summary_json={summary_path.resolve()}", flush=True)

    if fatal_error is not None:
        return 2
    if fail_on_oom and any(result.get("oom") for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  echo "1000-frame OOM smoke test exited with status ${status}" >&2
fi
exit "${status}"
