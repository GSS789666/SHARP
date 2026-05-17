"""Lightweight Social Force Model for warehouse pedestrians.

Reference: Helbing & Molnar (1995). We use a vectorised batch implementation
with three pedestrian classes (adult, elderly, child) that differ in desired
speed and footprint radius. This module is independent of PySocialForce so it
runs out of the box on Windows with only numpy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List, Sequence, Tuple


@dataclass
class Pedestrian:
    pos: np.ndarray              # shape (2,) in metres
    vel: np.ndarray              # shape (2,) in m/s
    goal: np.ndarray             # shape (2,) target waypoint
    desired_speed: float
    radius: float
    klass: str                   # 'adult' | 'elderly' | 'child'

    def to_dict(self):
        return {
            "pos": self.pos.tolist(), "vel": self.vel.tolist(),
            "goal": self.goal.tolist(), "klass": self.klass,
        }


class PedestrianCrowd:
    """Vectorised Social Force crowd simulator.

    State arrays (N pedestrians):
        pos   (N, 2)  positions
        vel   (N, 2)  velocities
        goal  (N, 2)  current goal waypoints
    """

    CLASS_RADIUS = {"adult": 0.30, "elderly": 0.32, "child": 0.25}

    def __init__(
        self,
        n: int,
        bounds: Tuple[float, float],
        cfg: dict,
        rng: np.random.Generator | None = None,
    ):
        self.n = n
        self.bounds = bounds                        # (W, H) in metres
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self):
        cfg = self.cfg
        W, H = self.bounds
        klasses = self.rng.choice(
            cfg["classes"], size=self.n, p=cfg["class_probs"]
        )
        self.klasses = klasses
        self.radius = np.array(
            [self.CLASS_RADIUS[k] for k in klasses], dtype=np.float32
        )
        self.desired_speed = np.array(
            [cfg["desired_speed"][k] for k in klasses], dtype=np.float32
        )

        self.pos = self.rng.uniform([0.5, 0.5], [W - 0.5, H - 0.5], (self.n, 2)).astype(np.float32)
        self.goal = self.rng.uniform([0.5, 0.5], [W - 0.5, H - 0.5], (self.n, 2)).astype(np.float32)
        self.vel = np.zeros((self.n, 2), dtype=np.float32)

    # ------------------------------------------------------------------ step

    def step(self, obstacles: np.ndarray | None = None) -> None:
        """Advance one Social Force step.

        Parameters
        ----------
        obstacles : (M, 2) array of obstacle (x, y) positions or None.
        """
        cfg = self.cfg
        dt = cfg["dt"]
        tau = cfg["relaxation_time"]

        # 1) goal-attraction force
        goal_dir = self.goal - self.pos
        gd_norm = np.linalg.norm(goal_dir, axis=1, keepdims=True) + 1e-6
        f_goal = (self.desired_speed[:, None] * goal_dir / gd_norm - self.vel) / tau

        # 2) pedestrian-pedestrian repulsion
        f_ped = self._pairwise_repulsion(
            self.pos, A=cfg["social_A"], B=cfg["social_B"]
        )

        # 3) obstacle repulsion
        f_obs = np.zeros_like(self.pos)
        if obstacles is not None and len(obstacles) > 0:
            f_obs = self._obstacle_repulsion(
                obstacles, A=cfg["obstacle_A"], B=cfg["obstacle_B"]
            )

        # integrate
        accel = f_goal + f_ped + f_obs
        self.vel = self.vel + accel * dt

        # cap speed at 1.3x desired
        sp = np.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-6
        cap = 1.3 * self.desired_speed[:, None]
        scale = np.minimum(1.0, cap / sp)
        self.vel = self.vel * scale

        self.pos = self.pos + self.vel * dt

        # reflect at walls
        W, H = self.bounds
        for i in range(self.n):
            if self.pos[i, 0] < 0 or self.pos[i, 0] > W:
                self.vel[i, 0] *= -0.5
                self.pos[i, 0] = np.clip(self.pos[i, 0], 0.0, W)
            if self.pos[i, 1] < 0 or self.pos[i, 1] > H:
                self.vel[i, 1] *= -0.5
                self.pos[i, 1] = np.clip(self.pos[i, 1], 0.0, H)

        # resample goals when reached
        d2g = np.linalg.norm(self.pos - self.goal, axis=1)
        reached = d2g < cfg["goal_radius"]
        if np.any(reached):
            new_goals = self.rng.uniform(
                [0.5, 0.5], [W - 0.5, H - 0.5], (reached.sum(), 2)
            ).astype(np.float32)
            self.goal[reached] = new_goals

    # -------------------------------------------------------------- forces

    @staticmethod
    def _pairwise_repulsion(
        pos: np.ndarray, A: float, B: float
    ) -> np.ndarray:
        """Vectorised exponential pedestrian repulsion."""
        n = pos.shape[0]
        if n < 2:
            return np.zeros_like(pos)
        diff = pos[:, None, :] - pos[None, :, :]      # (n, n, 2)
        dist = np.linalg.norm(diff, axis=2) + 1e-6    # (n, n)
        # zero out self interaction
        np.fill_diagonal(dist, np.inf)
        magn = A * np.exp(-dist / B)                  # (n, n)
        unit = diff / dist[:, :, None]                # (n, n, 2)
        f = (magn[:, :, None] * unit).sum(axis=1)     # (n, 2)
        return f.astype(np.float32)

    @staticmethod
    def _obstacle_repulsion(
        pos_or_self,
        A: float = 5.0,
        B: float = 0.2,
    ):
        # Note: this static method is called as instance method below.
        raise NotImplementedError("call via _obstacle_repulsion_impl")

    def _obstacle_repulsion(
        self, obstacles: np.ndarray, A: float, B: float
    ) -> np.ndarray:
        """Repulsion from a set of obstacle points (M, 2).

        Each pedestrian sees only the *nearest* obstacle to keep things cheap.
        """
        diff = self.pos[:, None, :] - obstacles[None, :, :]   # (n, M, 2)
        d = np.linalg.norm(diff, axis=2) + 1e-6                # (n, M)
        idx = np.argmin(d, axis=1)                             # (n,)
        nd = d[np.arange(self.n), idx]                         # (n,)
        nu = diff[np.arange(self.n), idx] / nd[:, None]
        magn = A * np.exp(-nd / B)
        return (magn[:, None] * nu).astype(np.float32)

    # ------------------------------------------------------------ utilities

    def snapshot(self) -> dict:
        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "goal": self.goal.copy(),
            "klasses": self.klasses.copy(),
            "radius": self.radius.copy(),
        }

    def predict(self, horizon: int, dt: float | None = None) -> np.ndarray:
        """Constant-velocity rollout for the next `horizon` steps.

        Returns array of shape (horizon, n, 2). Used as the *mean* of the
        Gaussian belief in Theorem 1; the variance grows linearly with horizon.
        """
        if dt is None:
            dt = self.cfg["dt"]
        steps = np.arange(1, horizon + 1, dtype=np.float32)
        return self.pos[None, :, :] + steps[:, None, None] * self.vel[None, :, :] * dt
