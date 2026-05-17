"""Warehouse environment for SHARP.

Models the РП / РТ / РР heterogeneous fleet executing pickup-delivery tasks
under chance-constrained safety w.r.t. a moving pedestrian crowd.

Coordinate system: the world is a continuous 2D plane in metres. A grid of
shape ``(grid_size, grid_size)`` with cell side ``cell_meters`` lays on top
purely for shelf placement and aisle structure.

The env exposes Manhattan path planning between waypoints (sufficient for
warehouse aisles) and a step-by-step simulation that interleaves robot motion
with crowd updates so we can score collision risk online.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .pedestrian import PedestrianCrowd


class RobotType(str, Enum):
    LOADER = "loader"            # РП
    TRANSPORTER = "transporter"  # РТ
    UNLOADER = "unloader"        # РР


@dataclass
class Robot:
    rid: int
    rtype: RobotType
    pos: np.ndarray           # (2,) metres
    speed: float              # m/s
    capacity: int
    energy: float             # remaining
    energy_max: float
    energy_per_meter: float
    radius: float             # collision footprint, metres
    # runtime state
    busy: bool = False
    task_id: int = -1
    waypoints: List[np.ndarray] = field(default_factory=list)
    distance_travelled: float = 0.0


@dataclass
class Task:
    tid: int
    pickup: np.ndarray        # (2,) metres
    delivery: np.ndarray      # (2,) metres
    payload: float            # kg-equivalent (used in capacity check)
    tw_start: float           # earliest service time (s)
    tw_end: float             # latest service time (s)
    priority: int = 1
    state: str = "open"       # open | assigned | picking | transit | done
    assigned_to: Tuple[int, int, int] = (-1, -1, -1)  # (loader, transporter, unloader)


class WarehouseEnv:
    """Continuous-time, discrete-step warehouse simulator.

    Each call to :meth:`step` advances ``dt`` seconds (default 0.25 s, taken
    from pedestrian config). Robots follow Manhattan paths between waypoints;
    pedestrians follow the Social Force Model.
    """

    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        e = cfg["env"]
        self.grid_size = e["grid_size"]
        self.cell = e["cell_meters"]
        self.world_size = self.grid_size * self.cell  # metres
        self.dt = cfg["pedestrian"]["dt"]
        self.max_steps = e["max_steps"]
        self.t = 0.0
        self.step_idx = 0
        self.collisions = 0
        self.collision_events: List[dict] = []

        self.robots: List[Robot] = []
        self.tasks: List[Task] = []
        self._spawn_robots()

        ped_n = 5 if e["pedestrian_density"] == "low" else 15
        self.crowd = PedestrianCrowd(
            n=ped_n,
            bounds=(self.world_size, self.world_size),
            cfg=cfg["pedestrian"],
            rng=self.rng,
        )

    # =================================================================== init

    def _spawn_robots(self) -> None:
        e = self.cfg["env"]
        rcfg = self.cfg["robot"]
        rid = 0

        def _add(rtype: RobotType, n: int, key: str):
            nonlocal rid
            for _ in range(n):
                pos = self.rng.uniform(1.0, self.world_size - 1.0, size=2).astype(np.float32)
                self.robots.append(
                    Robot(
                        rid=rid,
                        rtype=rtype,
                        pos=pos,
                        speed=rcfg[key]["speed"],
                        capacity=rcfg[key]["capacity"],
                        energy=rcfg[key]["energy_max"],
                        energy_max=rcfg[key]["energy_max"],
                        energy_per_meter=rcfg[key]["energy_per_meter"],
                        radius=self.cfg["risk"]["collision_radius"][key],
                    )
                )
                rid += 1

        _add(RobotType.LOADER, e["num_loaders"], "loader")
        _add(RobotType.TRANSPORTER, e["num_transporters"], "transporter")
        _add(RobotType.UNLOADER, e["num_unloaders"], "unloader")

    def reset_tasks(self, tasks: Sequence[Task]) -> None:
        self.tasks = list(tasks)

    # =================================================================== api

    def assign(
        self,
        task_id: int,
        loader_id: int,
        transporter_id: int,
        unloader_id: int,
    ) -> bool:
        """Atomically commit a (loader, transporter, unloader) assignment.

        Returns False if any robot is already busy or any constraint fails.
        """
        task = self.tasks[task_id]
        if task.state != "open":
            return False
        rl = self.robots[loader_id]
        rt = self.robots[transporter_id]
        ru = self.robots[unloader_id]
        if (
            rl.rtype != RobotType.LOADER
            or rt.rtype != RobotType.TRANSPORTER
            or ru.rtype != RobotType.UNLOADER
        ):
            return False
        if rl.busy or rt.busy or ru.busy:
            return False
        if task.payload > rt.capacity:
            return False
        # rough energy feasibility: round trip pickup -> delivery
        d_total = (
            self._manhattan(rt.pos, task.pickup) + self._manhattan(task.pickup, task.delivery)
        )
        if d_total * rt.energy_per_meter > rt.energy:
            return False

        task.state = "assigned"
        task.assigned_to = (loader_id, transporter_id, unloader_id)
        rl.busy = rt.busy = ru.busy = True
        rl.task_id = rt.task_id = ru.task_id = task_id

        # Waypoint plans. Transporter only sees the *current* leg; the
        # delivery leg is added once the loader finishes handover. This
        # prevents the transporter from blowing past pickup before the
        # loader arrives.
        rl.waypoints = [task.pickup.copy()]   # to pickup
        rt.waypoints = [task.pickup.copy()]   # to pickup; delivery appended later
        ru.waypoints = [task.delivery.copy()] # to delivery (waits there)
        return True

    def step(self) -> dict:
        """Advance one ``dt`` step. Returns info dict for logging."""
        self.crowd.step(obstacles=self._robot_positions())
        for r in self.robots:
            self._move_robot(r)
        col_now = self._check_collisions()
        self.collisions += col_now

        # task progress checks
        self._update_task_states()

        self.t += self.dt
        self.step_idx += 1

        return {
            "t": self.t, "step": self.step_idx, "collisions_step": col_now,
            "collisions_total": self.collisions,
            "tasks_done": sum(1 for k in self.tasks if k.state == "done"),
        }

    def done(self) -> bool:
        if self.step_idx >= self.max_steps:
            return True
        return all(k.state == "done" for k in self.tasks) and len(self.tasks) > 0

    # ============================================================== mechanics

    def _move_robot(self, r: Robot) -> None:
        if not r.waypoints:
            return
        target = r.waypoints[0]
        delta = target - r.pos
        dist = np.linalg.norm(delta)
        max_step = r.speed * self.dt
        if dist <= max_step:
            r.pos = target.copy()
            r.distance_travelled += dist
            r.energy -= dist * r.energy_per_meter
            r.waypoints.pop(0)
        else:
            r.pos = r.pos + delta / (dist + 1e-9) * max_step
            r.distance_travelled += max_step
            r.energy -= max_step * r.energy_per_meter

    def _update_task_states(self) -> None:
        for task in self.tasks:
            if task.state in ("open", "done"):
                continue
            l_id, t_id, u_id = task.assigned_to
            rl, rt, ru = self.robots[l_id], self.robots[t_id], self.robots[u_id]

            if task.state == "assigned":
                # both loader and transporter must arrive at pickup;
                # they will idle (empty waypoints) once they get there.
                rl_here = np.linalg.norm(rl.pos - task.pickup) < 0.6
                rt_here = np.linalg.norm(rt.pos - task.pickup) < 0.6
                if rl_here and rt_here:
                    task.state = "transit"
                    rl.busy = False              # loader released after handover
                    rl.task_id = -1
                    rl.waypoints = []
                    # arm the transporter for the delivery leg
                    rt.waypoints = [task.delivery.copy()]

            elif task.state == "transit":
                rt_here = np.linalg.norm(rt.pos - task.delivery) < 0.6
                ru_here = np.linalg.norm(ru.pos - task.delivery) < 0.6
                if rt_here and ru_here:
                    task.state = "done"
                    rt.busy = ru.busy = False
                    rt.task_id = ru.task_id = -1
                    rt.waypoints = []
                    ru.waypoints = []

    def _check_collisions(self) -> int:
        """Count robot–pedestrian overlap events this step."""
        if self.crowd.n == 0:
            return 0
        rpos = self._robot_positions()
        rrad = np.array([r.radius for r in self.robots], dtype=np.float32)
        ppos = self.crowd.pos
        prad = self.crowd.radius
        # pairwise distances
        diff = rpos[:, None, :] - ppos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        thresh = rrad[:, None] + prad[None, :]
        hits = dist < thresh
        n_hits = int(hits.sum())
        if n_hits > 0:
            ri, pi = np.where(hits)
            for r_idx, p_idx in zip(ri, pi):
                self.collision_events.append(
                    {"t": self.t, "robot": int(r_idx), "ped": int(p_idx)}
                )
        return n_hits

    # =============================================================== helpers

    def _robot_positions(self) -> np.ndarray:
        return np.stack([r.pos for r in self.robots]).astype(np.float32)

    @staticmethod
    def _manhattan(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).sum())

    # ------------------------------------------------------------- snapshots

    def snapshot(self) -> dict:
        return {
            "t": self.t, "step": self.step_idx,
            "robots": [
                {
                    "rid": r.rid, "type": r.rtype.value,
                    "pos": r.pos.tolist(), "energy": float(r.energy),
                    "busy": r.busy, "task": r.task_id,
                }
                for r in self.robots
            ],
            "tasks_open": sum(1 for k in self.tasks if k.state == "open"),
            "tasks_done": sum(1 for k in self.tasks if k.state == "done"),
            "collisions": self.collisions,
            "ped": self.crowd.snapshot(),
        }
