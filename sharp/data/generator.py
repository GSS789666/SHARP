"""Synthetic task instance generator for SHARP.

Mirrors the data style used by CAHT (aisle-structured warehouse, time windows)
but extends with pedestrian density tags so each instance describes a complete
chance-constrained scenario.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

from sharp.env.warehouse import Task


def generate_tasks(
    n: int,
    world_size: float,
    horizon: float = 600.0,
    rng: np.random.Generator | None = None,
) -> List[Task]:
    """Generate ``n`` pickup-delivery tasks within a square warehouse.

    - pickup/delivery uniform in [0.5, world_size-0.5]
    - payloads ~ U[0.2, 1.0]
    - time windows: tw_start ~ U[0, horizon/2], width ~ U[60, 240] s
    - priority: 80% = 1, 15% = 2, 5% = 3
    """
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = 0.5, world_size - 0.5
    tasks: List[Task] = []
    for tid in range(n):
        pickup = rng.uniform(lo, hi, size=2).astype(np.float32)
        # ensure delivery is at least 5 m away
        for _ in range(8):
            delivery = rng.uniform(lo, hi, size=2).astype(np.float32)
            if np.linalg.norm(delivery - pickup) > 5.0:
                break
        payload = float(rng.uniform(0.2, 1.0))
        tw_start = float(rng.uniform(0, horizon * 0.5))
        tw_width = float(rng.uniform(60.0, 240.0))
        priority = int(rng.choice([1, 2, 3], p=[0.80, 0.15, 0.05]))
        tasks.append(
            Task(
                tid=tid,
                pickup=pickup,
                delivery=delivery,
                payload=payload,
                tw_start=tw_start,
                tw_end=tw_start + tw_width,
                priority=priority,
            )
        )
    return tasks


def sample_instance(cfg: dict, seed: int = 0) -> List[Task]:
    """Top-level convenience wrapper using cfg['env'].

    The horizon is derived from ``max_steps * dt`` to keep time windows
    inside the simulation window (otherwise most ``tw_start`` would fall
    outside the rollout and reward signals collapse).
    """
    rng = np.random.default_rng(seed)
    n = cfg["env"]["num_tasks"]
    world = cfg["env"]["grid_size"] * cfg["env"]["cell_meters"]
    horizon = cfg["env"]["max_steps"] * cfg["pedestrian"]["dt"]
    return generate_tasks(n=n, world_size=world, horizon=horizon, rng=rng)
