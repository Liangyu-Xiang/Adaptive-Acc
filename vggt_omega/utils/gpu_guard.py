from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ComputeApp:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int


def _run_nvidia_smi(query: str) -> list[str]:
    command = [
        "nvidia-smi",
        f"--query-{query}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "nvidia-smi failed"
        raise RuntimeError(message)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].lower() == "no running processes found":
        return []
    return lines


def query_gpu_inventory() -> tuple[dict[str, str], dict[str, int]]:
    uuid_by_index: dict[str, str] = {}
    memory_by_index: dict[str, int] = {}
    for line in _run_nvidia_smi("gpu=index,uuid,memory.used"):
        index_text, uuid_text, memory_text = [part.strip() for part in line.split(",", 2)]
        uuid_by_index[index_text] = uuid_text
        memory_by_index[index_text] = int(memory_text)
    return uuid_by_index, memory_by_index


def query_compute_apps() -> list[ComputeApp]:
    apps: list[ComputeApp] = []
    for line in _run_nvidia_smi("compute-apps=gpu_uuid,pid,process_name,used_memory"):
        gpu_uuid, pid_text, process_name, used_memory_text = [part.strip() for part in line.split(",", 3)]
        apps.append(
            ComputeApp(
                gpu_uuid=gpu_uuid,
                pid=int(pid_text),
                process_name=process_name,
                used_memory_mib=int(used_memory_text),
            )
        )
    return apps


def assert_exclusive_gpu(
    physical_gpu_index: int | str,
    *,
    allowed_pids: Iterable[int] = (),
    max_other_memory_mib: int = 512,
) -> None:
    index_text = str(physical_gpu_index)
    uuid_by_index, memory_by_index = query_gpu_inventory()
    if index_text not in uuid_by_index:
        known = ", ".join(sorted(uuid_by_index)) or "<none>"
        raise RuntimeError(f"GPU index {index_text} not found; visible indices: {known}")

    allowed = {int(pid) for pid in allowed_pids}
    target_uuid = uuid_by_index[index_text]
    target_apps = [app for app in query_compute_apps() if app.gpu_uuid == target_uuid]
    foreign_apps = [app for app in target_apps if app.pid not in allowed]
    if foreign_apps:
        details = "; ".join(
            f"pid={app.pid} name={app.process_name} mem={app.used_memory_mib}MiB"
            for app in foreign_apps
        )
        raise RuntimeError(f"GPU {index_text} is not exclusive: {details}")

    total_used_mib = memory_by_index[index_text]
    own_used_mib = sum(app.used_memory_mib for app in target_apps if app.pid in allowed)
    other_used_mib = max(0, total_used_mib - own_used_mib)
    if other_used_mib > max_other_memory_mib:
        raise RuntimeError(
            f"GPU {index_text} has {other_used_mib}MiB residual memory outside allowed PIDs "
            f"(total={total_used_mib}MiB, own={own_used_mib}MiB, limit={max_other_memory_mib}MiB)"
        )


def wait_for_exclusive_gpu(
    physical_gpu_index: int | str,
    *,
    allowed_pids: Iterable[int] = (),
    max_other_memory_mib: int = 512,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
) -> None:
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while True:
        try:
            assert_exclusive_gpu(
                physical_gpu_index,
                allowed_pids=allowed_pids,
                max_other_memory_mib=max_other_memory_mib,
            )
            return
        except RuntimeError as error:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(str(error)) from error
            time.sleep(poll_seconds)
