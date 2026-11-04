"""Read-only physical GPU checks; never creates a CUDA context."""
from __future__ import annotations

import os
import subprocess

FORBIDDEN_INDEX = 3
FORBIDDEN_UUID = "GPU-e8be9cff-d196-2c04-597d-6ed91e4d6468"
ALLOWED_INDICES = (0, 1, 2, 4, 5, 6)


def gpu_inventory():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    )
    result = {}
    for line in raw.splitlines():
        index, uuid, memory, utilization = (part.strip() for part in line.split(","))
        result[int(index)] = {
            "uuid": uuid, "memory_MiB": int(memory), "utilization": int(utilization),
        }
    if FORBIDDEN_INDEX not in result:
        raise RuntimeError("Cannot verify physical GPU 3 mapping; refusing launch")
    return result


def compute_processes():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [tuple(part.strip() for part in line.split(",")) for line in raw.splitlines() if line.strip()]


def gpu_environment(index, *, require_idle=True, inventory=None, processes=None):
    index = int(index)
    if index not in ALLOWED_INDICES:
        raise ValueError(f"Physical GPU {index} is not allowed; GPU 3 is always forbidden")
    state = gpu_inventory() if inventory is None else inventory
    row = state[index]
    forbidden = {FORBIDDEN_UUID, state[FORBIDDEN_INDEX]["uuid"]}
    if row["uuid"] in forbidden:
        raise RuntimeError("Forbidden physical GPU/UUID mapping")
    if require_idle:
        jobs = compute_processes() if processes is None else processes
        if any(uuid == row["uuid"] for uuid, _ in jobs) or row["memory_MiB"] > 256 or row["utilization"] > 5:
            raise RuntimeError(f"Physical GPU {index} is occupied; refusing launch")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=row["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return environment
