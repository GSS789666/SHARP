"""Primal-Dual PPO for SHARP (CMDP training).

Implements Algorithm 1 of method draft §3.7:

    for k = 1..T:
        roll out τ ~ π_k
        compute Â (advantage on r̃ = r − λ·c)
        θ ← θ + η_π · ∇L_clip(θ)
        λ ← max(0, λ + η_λ · (Ĵ_c − ε))

The implementation deliberately stays minimal: a single-agent env, a single
rollout buffer, and one optimisation pass per rollout. It is a faithful
reference of Theorem 2 with no algorithmic embellishments so we can clearly
attribute any empirical gain to the architectural contributions of SHARP.
"""
from __future__ import annotations
import os
import math
import time
import json
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange

from sharp.env import WarehouseRLEnv
from sharp.model import SHARP
from sharp.model.sharp import SHARPInput


# =================================================================== buffer

@dataclass
class RolloutBuffer:
    obs: List[SHARPInput] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logp: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    lead_loader: List[int] = field(default_factory=list)

    def __len__(self):
        return len(self.actions)

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.logp.clear()
        self.rewards.clear(); self.costs.clear(); self.values.clear()
        self.dones.clear(); self.lead_loader.clear()


# =================================================================== utils

def gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
    last_value: float, gamma: float = 0.99, lam: float = 0.95,
):
    T = rewards.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nt = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * nv * nt - values[t]
        last = delta + gamma * lam * nt * last
        adv[t] = last
    ret = adv + values
    return adv, ret


def collect_rollout(
    env: WarehouseRLEnv,
    model: SHARP,
    n_steps: int,
    device: str = "cpu",
) -> Tuple[RolloutBuffer, dict]:
    buf = RolloutBuffer()
    model.eval()
    obs = env.reset() if env.env is None else env._observation()
    if env.env is None or env.env.done() or env.steps == 0:
        obs = env.reset(seed=int(np.random.randint(0, 1e6)))
    eps_returns = []
    eps_costs = []
    eps_done_count = []
    cur_R = 0.0; cur_C = 0.0; cur_done = 0
    with torch.no_grad():
        for _ in range(n_steps):
            obs_dev = _to_device(obs, device)
            logits, value = model(obs_dev)         # (1, R, M+1)
            lead = env.lead_loader_id
            if lead is None:
                action = env.num_tasks
                row = logits[0, 0]                 # placeholder
                logp = torch.tensor(0.0)
            else:
                row = logits[0, lead]
                if not torch.isfinite(row).any():
                    action = env.num_tasks
                    logp = torch.tensor(0.0)
                else:
                    dist = Categorical(logits=row)
                    a = dist.sample()
                    action = int(a.item())
                    logp = dist.log_prob(a)
            buf.obs.append(_to_device(obs, "cpu"))
            buf.actions.append(int(action))
            buf.logp.append(float(logp.item()))
            buf.values.append(float(value.item()))
            buf.lead_loader.append(int(lead) if lead is not None else -1)
            obs, info = env.step(action)
            buf.rewards.append(float(info.reward))
            buf.costs.append(float(info.cost))
            buf.dones.append(bool(info.done))
            cur_R += info.reward; cur_C += info.cost
            cur_done = info.info["tasks_done"]
            if info.done:
                eps_returns.append(cur_R)
                eps_costs.append(cur_C)
                eps_done_count.append(cur_done)
                cur_R = cur_C = 0.0
                obs = env.reset(seed=int(np.random.randint(0, 1e6)))
        # bootstrap value of next state
        obs_dev = _to_device(obs, device)
        _, last_v = model(obs_dev)
        last_value = float(last_v.item())
    diag = {
        "ep_return_mean": float(np.mean(eps_returns)) if eps_returns else cur_R,
        "ep_cost_mean": float(np.mean(eps_costs)) if eps_costs else cur_C,
        "ep_done_mean": float(np.mean(eps_done_count)) if eps_done_count else cur_done,
        "n_episodes": len(eps_returns),
        "last_value": last_value,
    }
    return buf, diag


def _to_device(x: SHARPInput, device: str) -> SHARPInput:
    return SHARPInput(
        robot_feat=x.robot_feat.to(device),
        robot_pos=x.robot_pos.to(device),
        task_feat=x.task_feat.to(device),
        task_pos=x.task_pos.to(device),
        feas_mask=x.feas_mask.to(device),
        cvar_mask=x.cvar_mask.to(device),
        margin_norm=x.margin_norm.to(device),
    )


# ==================================================================== algo

class PrimalDualPPO:
    def __init__(
        self,
        model: SHARP,
        env: WarehouseRLEnv,
        cfg: dict,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.env = env
        self.cfg = cfg
        self.device = device
        t = cfg["training"]
        self.optim = torch.optim.Adam(model.parameters(), lr=t["lr_policy"])
        self.lambda_ = 0.0
        self.lr_dual = t["lr_dual"]
        # Hard cap on lambda — see config/default.yaml for the rationale.
        self.lambda_max = float(t.get("lambda_max", float("inf")))
        self.eps_safety = cfg["risk"]["epsilon_safety"]

    def update(self, buf: RolloutBuffer, last_value: float):
        t = self.cfg["training"]
        gamma, lam = t["gamma"], t["gae_lambda"]
        clip = t["clip_eps"]

        rewards = np.array(buf.rewards, dtype=np.float32)
        costs = np.array(buf.costs, dtype=np.float32)
        values = np.array(buf.values, dtype=np.float32)
        dones = np.array(buf.dones, dtype=bool)
        # Lagrangian-shaped reward
        r_hat = rewards - self.lambda_ * costs
        adv, ret = gae(r_hat, values, dones, last_value, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(buf.logp, dtype=torch.float32, device=self.device)
        actions = torch.tensor(buf.actions, dtype=torch.long, device=self.device)
        leads = buf.lead_loader

        self.model.train()
        # one full epoch over the buffer
        # (small buffers + small networks ⇒ no minibatching)
        new_logp_list = []
        new_v_list = []
        ent_list = []
        for i, obs in enumerate(buf.obs):
            obs_d = _to_device(obs, self.device)
            logits, v = self.model(obs_d)
            new_v_list.append(v)
            if leads[i] < 0:
                # no decision was made at this step; use a deterministic 0
                new_logp_list.append(torch.tensor(0.0, device=self.device))
                ent_list.append(torch.tensor(0.0, device=self.device))
                continue
            row = logits[0, leads[i]]
            if not torch.isfinite(row).any():
                new_logp_list.append(torch.tensor(0.0, device=self.device))
                ent_list.append(torch.tensor(0.0, device=self.device))
                continue
            dist = Categorical(logits=row)
            new_logp_list.append(dist.log_prob(actions[i]))
            ent_list.append(dist.entropy())

        new_logp = torch.stack(new_logp_list)
        new_v = torch.stack(new_v_list).squeeze(-1)
        entropy = torch.stack(ent_list)

        ratio = torch.exp(new_logp - old_logp)
        s1 = ratio * adv_t
        s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
        loss_pi = -torch.min(s1, s2).mean()
        loss_v = F.mse_loss(new_v, ret_t)
        loss_ent = -entropy.mean()
        loss = loss_pi + t["value_coef"] * loss_v + t["entropy_coef"] * loss_ent

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()

        # dual update: episode-averaged cost − ε  (per decision step proxy)
        cost_per_step = costs.mean()
        new_lam = self.lambda_ + self.lr_dual * (cost_per_step - self.eps_safety)
        # clip to [0, lambda_max]: prevents the safety penalty from
        # overwhelming the task-completion reward.
        self.lambda_ = float(np.clip(new_lam, 0.0, self.lambda_max))

        return {
            "loss": float(loss.item()),
            "loss_pi": float(loss_pi.item()),
            "loss_v": float(loss_v.item()),
            "entropy": float((-loss_ent).item()),
            "lambda": float(self.lambda_),
            "cost_step_mean": float(cost_per_step),
            "ratio_mean": float(ratio.mean().item()),
        }


# =================================================================== driver

def train(
    cfg: dict,
    save_dir: str,
    n_iters: int = 50,
    n_steps_per_iter: int = 256,
    device: str = "cpu",
    log_every: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    env = WarehouseRLEnv(cfg, seed=cfg["training"]["seed"])
    model = SHARP()
    algo = PrimalDualPPO(model, env, cfg, device=device)

    log = []
    pbar = trange(n_iters, dynamic_ncols=True)
    for it in pbar:
        buf, diag = collect_rollout(env, model, n_steps_per_iter, device=device)
        upd = algo.update(buf, diag["last_value"])
        row = {"iter": it, **diag, **upd}
        log.append(row)
        if it % log_every == 0:
            pbar.set_postfix({
                "ret": f"{diag['ep_return_mean']:.0f}",
                "cost": f"{diag['ep_cost_mean']:.1f}",
                "λ": f"{algo.lambda_:.3f}",
                "L": f"{upd['loss']:.3f}",
            })
        with open(os.path.join(save_dir, "train_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if (it + 1) % 10 == 0 or it == n_iters - 1:
            torch.save(
                {"model": model.state_dict(), "lambda": algo.lambda_,
                 "iter": it, "cfg": cfg},
                os.path.join(save_dir, f"ckpt_{it+1:04d}.pt"),
            )
    return model, log
