"""Pedestrian-blind CAHT-style baseline (B1 in §4 of the proposal).

This baseline reproduces the *constraint-aware* but *pedestrian-blind* spirit
of CAHT (Gong & Varlamov 2025): each free loader is greedily paired with the
earliest open task whose deterministic feasibility constraints (capacity,
energy, time-window) are satisfied. Pedestrian risk is ignored, so this
policy expects to outperform SHARP on Makespan/energy but to violate the
collision rate ε severely. It is the head-to-head reference that proves the
*marginal* value of the new safety mechanisms.

We keep this in the same Python process — we are not re-training the
published CAHT network. Treat this as a deterministic policy that exercises
the same feasibility logic; the conclusion of the paper rests on the relative
gap between this and SHARP, not on the absolute scores of either.
"""
from __future__ import annotations
import numpy as np
from sharp.env import WarehouseRLEnv, RobotType


class GreedyCAHTPolicy:
    """Returns an action index for the lead loader at each step."""

    def act(self, env: WarehouseRLEnv) -> int:
        wenv = env.env
        lead = env.lead_loader_id
        if lead is None:
            return env.num_tasks  # wait

        # earliest open task by tw_start; tie-break by distance
        loader = wenv.robots[lead]
        open_tasks = [t for t in wenv.tasks if t.state == "open"]
        if not open_tasks:
            return env.num_tasks

        # filter feasibility (capacity / energy on transporter side)
        free_t = [
            r for r in wenv.robots
            if r.rtype == RobotType.TRANSPORTER and not r.busy
        ]
        if not free_t:
            return env.num_tasks
        free_u = [
            r for r in wenv.robots
            if r.rtype == RobotType.UNLOADER and not r.busy
        ]
        if not free_u:
            return env.num_tasks

        candidates = []
        for t in open_tasks:
            best_t = min(free_t, key=lambda r: float(np.linalg.norm(r.pos - t.pickup)))
            d_total = (
                float(np.linalg.norm(best_t.pos - t.pickup))
                + float(np.linalg.norm(t.pickup - t.delivery))
            )
            if t.payload > best_t.capacity:
                continue
            if d_total * best_t.energy_per_meter > best_t.energy:
                continue
            score = t.tw_start + 0.05 * float(np.linalg.norm(loader.pos - t.pickup))
            candidates.append((score, t.tid))
        if not candidates:
            return env.num_tasks
        candidates.sort()
        return int(candidates[0][1])
