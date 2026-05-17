"""Gym-style RL wrapper around :class:`WarehouseEnv`.

Decision interface — at each decision step the env exposes:
    - the **lead loader** (the free loader with the smallest id), which is the
      robot whose decision the policy must make this step
    - the full set of open tasks
    - pre-computed feasibility mask, CVaR mask and normalised margins for
      every (robot, task) pair so the policy can also reason about cross-robot
      compatibility through its encoder

The action is a single integer in ``[0, M]`` where:
    - ``a in [0, M-1]`` means: assign task ``a`` to the lead loader, pairing
      it with the nearest free transporter and the nearest-to-delivery free
      unloader (greedy choices for the auxiliary roles).
    - ``a == M`` is the *wait* action (no new assignment this step).

Between two decision steps the underlying simulator advances ``decision_dt /
sim_dt`` ticks. Reward and cost are accumulated over those ticks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .warehouse import WarehouseEnv, RobotType
from sharp.data import sample_instance
from sharp.model.cvar_mask import cvar_safety_mask
from sharp.model.ped_features import compute_ped_features
from sharp.model.sharp import SHARPInput


@dataclass
class StepInfo:
    reward: float
    cost: float                    # collision cost (= num collisions this step)
    done: bool
    truncated: bool
    info: dict


class WarehouseRLEnv:
    """Single-agent RL view of the warehouse simulator.

    The "agent" controls the lead loader at each decision point. Auxiliary
    roles (one transporter + one unloader) are filled by a greedy heuristic
    so the model only has to learn the high-impact loader-task pairing.
    """

    def __init__(
        self,
        cfg: dict,
        seed: int = 0,
        decision_period: int = 4,
        safety_mode: str = "cvar",
    ):
        """
        safety_mode :
            ``cvar``    -- use the Theorem 1 CVaR mask (default; SHARP).
            ``off``     -- always-True safety mask. Used by the Lagrangian-PPO
                           baseline (B2), where safety is enforced *only* by
                           the dual variable, not the architecture.
        """
        # YAML 1.1 parses bare `off` as Python False; coerce defensively so
        # downstream code can always assume a string.
        if safety_mode is False or safety_mode is None:
            safety_mode = "off"
        elif safety_mode is True:
            safety_mode = "cvar"
        assert safety_mode in {"cvar", "off"}, safety_mode
        self.cfg = cfg
        self.seed = seed
        self.decision_period = decision_period   # sim steps between decisions
        self.safety_mode = safety_mode
        self.env: Optional[WarehouseEnv] = None
        self.num_tasks = cfg["env"]["num_tasks"]
        self.action_dim = self.num_tasks + 1     # +1 for wait
        self._reset_state()

    # =================================================================== api

    def reset(self, seed: Optional[int] = None) -> SHARPInput:
        if seed is not None:
            self.seed = seed
        self.env = WarehouseEnv(self.cfg, seed=self.seed)
        self.env.reset_tasks(sample_instance(self.cfg, seed=self.seed))
        self._reset_state()
        return self._observation()

    def _reset_state(self):
        self.last_done_count = 0
        self.last_collisions = 0
        self.episode_reward = 0.0
        self.episode_cost = 0.0
        self.steps = 0

    def step(self, action: int) -> Tuple[SHARPInput, StepInfo]:
        """Apply the chosen action, advance ``decision_period`` ticks, return."""
        assert self.env is not None, "call reset() first"
        # 1) apply action
        if action < self.num_tasks:
            self._try_assign(int(action))
        # action == self.num_tasks → wait, no-op

        # 2) advance simulator by decision_period sim ticks
        col_before = self.env.collisions
        done_before_ids = {t.tid for t in self.env.tasks if t.state == "done"}
        for _ in range(self.decision_period):
            self.env.step()
            if self.env.done():
                break
        col_after = self.env.collisions
        done_after_ids = {t.tid for t in self.env.tasks if t.state == "done"}
        new_done_ids = done_after_ids - done_before_ids

        # 3) reward / cost
        new_col = col_after - col_before
        reward = self._reward(new_done_ids, new_col)
        cost = float(new_col)
        new_done = len(new_done_ids)
        done_after = len(done_after_ids)
        self.episode_reward += reward
        self.episode_cost += cost
        self.steps += 1

        truncated = self.steps * self.decision_period >= self.cfg["env"]["max_steps"]
        done = self.env.done() or truncated

        info = {
            "tasks_done": done_after,
            "collisions_total": col_after,
            "episode_reward": self.episode_reward,
            "episode_cost": self.episode_cost,
            "sim_step": self.env.step_idx,
        }
        return self._observation(), StepInfo(reward, cost, done, truncated, info)

    # ============================================================ observation

    def _observation(self) -> SHARPInput:
        env = self.env
        cfg = self.cfg

        # --------------- robot tensors
        R = len(env.robots)
        robot_pos = np.stack([r.pos for r in env.robots]).astype(np.float32)
        # Raw robot feature: pos(2) + energy_norm + capacity_norm + type_oh(3) = 7
        type_oh = np.zeros((R, 3), dtype=np.float32)
        for i, r in enumerate(env.robots):
            type_oh[i, list(RobotType).index(r.rtype)] = 1.0
        robot_raw = np.concatenate(
            [
                robot_pos,
                np.array([[r.energy / r.energy_max] for r in env.robots], dtype=np.float32),
                np.array([[1.0 - float(r.busy)] for r in env.robots], dtype=np.float32),
                type_oh,
            ],
            axis=1,
        )                                                    # (R, 7)

        # --------------- task tensors (only open tasks expose pickup / etc.)
        M = self.num_tasks
        task_pickup = np.zeros((M, 2), dtype=np.float32)
        task_deliv = np.zeros((M, 2), dtype=np.float32)
        task_raw = np.zeros((M, 8), dtype=np.float32)
        open_mask = np.zeros((M,), dtype=bool)
        for j, t in enumerate(env.tasks):
            task_pickup[j] = t.pickup
            task_deliv[j] = t.delivery
            task_raw[j] = [
                t.pickup[0], t.pickup[1], t.delivery[0], t.delivery[1],
                t.payload, t.tw_start, t.tw_end, float(t.priority),
            ]
            open_mask[j] = (t.state == "open")

        # --------------- pedestrian features per node
        ped_pos = env.crowd.pos
        ped_vel = env.crowd.vel
        ped_rad = env.crowd.radius
        rfeat = compute_ped_features(robot_pos, ped_pos, ped_vel)
        tfeat = compute_ped_features(task_pickup, ped_pos, ped_vel)

        robot_feat = np.concatenate([robot_raw, rfeat], axis=1).astype(np.float32)   # (R, 11)
        task_feat = np.concatenate([task_raw, tfeat], axis=1).astype(np.float32)     # (M, 12)

        # --------------- masks
        # feasibility: only free loaders * open tasks are eligible
        feas = np.zeros((R, M), dtype=bool)
        loader_ids = [
            i for i, r in enumerate(env.robots)
            if r.rtype == RobotType.LOADER and not r.busy
        ]
        for i in loader_ids:
            feas[i, open_mask] = True

        # CVaR mask only against task pickups
        robot_radius = np.array([r.radius for r in env.robots], dtype=np.float32)
        robot_speed = np.array([r.speed for r in env.robots], dtype=np.float32)
        if self.safety_mode == "cvar" and ped_pos.shape[0] > 0:
            # collect free transporter / unloader info so the mask covers the
            # full assignment chain, not just the loader-to-pickup leg.
            free_t_pos = np.array(
                [r.pos for r in env.robots
                 if r.rtype == RobotType.TRANSPORTER and not r.busy],
                dtype=np.float32,
            )
            free_t_spd = np.array(
                [r.speed for r in env.robots
                 if r.rtype == RobotType.TRANSPORTER and not r.busy],
                dtype=np.float32,
            )
            free_u_pos = np.array(
                [r.pos for r in env.robots
                 if r.rtype == RobotType.UNLOADER and not r.busy],
                dtype=np.float32,
            )
            free_u_spd = np.array(
                [r.speed for r in env.robots
                 if r.rtype == RobotType.UNLOADER and not r.busy],
                dtype=np.float32,
            )
            full_check_ok = (
                free_t_pos.shape[0] > 0 and free_u_pos.shape[0] > 0
            )
            cvar_m, cvar_margin = cvar_safety_mask(
                robot_pos=robot_pos, robot_radius=robot_radius,
                robot_speed=robot_speed, task_pickup=task_pickup,
                ped_pos=ped_pos, ped_vel=ped_vel, ped_radius=ped_rad,
                dt=cfg["pedestrian"]["dt"],
                horizon=cfg["risk"]["prediction_horizon"],
                alpha=cfg["risk"]["cvar_alpha"],
                task_delivery=task_deliv if full_check_ok else None,
                transporter_pos=free_t_pos if full_check_ok else None,
                transporter_speed=free_t_spd if full_check_ok else None,
                unloader_pos=free_u_pos if full_check_ok else None,
                unloader_speed=free_u_spd if full_check_ok else None,
            )
            avg_rho = robot_radius[:, None] + ped_rad.mean()
            margin_norm = (cvar_margin / np.maximum(avg_rho, 1e-3)).astype(np.float32)
        else:
            cvar_m = np.ones((R, M), dtype=bool)
            margin_norm = np.full((R, M), 1e3, dtype=np.float32)

        # to torch tensors with batch dim 1
        return SHARPInput(
            robot_feat=torch.from_numpy(robot_feat).unsqueeze(0),
            robot_pos=torch.from_numpy(robot_pos).unsqueeze(0),
            task_feat=torch.from_numpy(task_feat).unsqueeze(0),
            task_pos=torch.from_numpy(task_pickup).unsqueeze(0),
            feas_mask=torch.from_numpy(feas).unsqueeze(0),
            cvar_mask=torch.from_numpy(cvar_m).unsqueeze(0),
            margin_norm=torch.from_numpy(margin_norm).unsqueeze(0),
        )

    # ============================================================ helpers

    @property
    def lead_loader_id(self) -> Optional[int]:
        """Smallest-id free loader (the row whose action is consumed)."""
        for i, r in enumerate(self.env.robots):
            if r.rtype == RobotType.LOADER and not r.busy:
                return i
        return None

    def _try_assign(self, task_id: int) -> bool:
        env = self.env
        loader_id = self.lead_loader_id
        if loader_id is None:
            return False
        task = env.tasks[task_id]
        if task.state != "open":
            return False
        # nearest free transporter to pickup
        free_t = [
            r for r in env.robots
            if r.rtype == RobotType.TRANSPORTER and not r.busy
        ]
        if not free_t:
            return False
        t_id = min(free_t, key=lambda r: float(np.linalg.norm(r.pos - task.pickup))).rid
        # nearest free unloader to delivery
        free_u = [
            r for r in env.robots
            if r.rtype == RobotType.UNLOADER and not r.busy
        ]
        if not free_u:
            return False
        u_id = min(free_u, key=lambda r: float(np.linalg.norm(r.pos - task.delivery))).rid
        return env.assign(task_id, loader_id, t_id, u_id)

    def _reward(self, new_done_ids: set, new_col: int) -> float:
        """Per-decision-step reward.

        Sums priority-weighted completion bonuses minus a TW-late linear
        decay, minus collision penalty (kept 0 by default — handled by dual),
        minus a constant step penalty that prevents the trivial wait policy.
        """
        rcfg = self.cfg["reward"]
        r = 0.0
        for tid in new_done_ids:
            t = self.env.tasks[tid]
            base = rcfg["task_completion"] * float(t.priority)
            late = max(0.0, self.env.t - t.tw_end)
            base = max(0.0, base - rcfg["tw_late_slope"] * late)
            r += base
        r -= rcfg["collision_penalty"] * new_col
        r -= rcfg["step_penalty"]
        return float(r)

    # ============================================================ debug

    def render_text(self) -> str:
        e = self.env
        return (
            f"step={self.steps:4d}  sim_t={e.t:6.1f}s  "
            f"open={sum(1 for x in e.tasks if x.state=='open'):3d}  "
            f"done={sum(1 for x in e.tasks if x.state=='done'):3d}  "
            f"col={e.collisions:4d}  R={self.episode_reward:8.1f}  C={self.episode_cost:5.1f}"
        )
