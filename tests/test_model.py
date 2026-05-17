"""Smoke test the SHARP model on a synthetic batch."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sharp.model import SHARP
from sharp.model.sharp import SHARPInput
from sharp.model import compute_ped_features, cvar_safety_mask


def make_synth_batch(B=2, R=12, M=20, H=10, world=20.0, seed=0):
    rng = np.random.default_rng(seed)
    robot_pos = rng.uniform(0.5, world - 0.5, (B, R, 2)).astype(np.float32)
    task_pickup = rng.uniform(0.5, world - 0.5, (B, M, 2)).astype(np.float32)
    task_deliv = rng.uniform(0.5, world - 0.5, (B, M, 2)).astype(np.float32)
    ped_pos = rng.uniform(0.5, world - 0.5, (H, 2)).astype(np.float32)
    ped_vel = rng.normal(0, 0.5, (H, 2)).astype(np.float32)
    ped_rad = np.full((H,), 0.3, dtype=np.float32)
    robot_rad = np.full((R,), 0.5, dtype=np.float32)
    robot_speed = np.full((R,), 1.5, dtype=np.float32)

    # raw 7-dim robot feature: pos(2)+energy(1)+capacity(1)+type_oh(3)
    robot_raw = np.concatenate(
        [robot_pos,
         rng.uniform(0.5, 1.0, (B, R, 1)).astype(np.float32),       # energy/max
         rng.uniform(0.0, 1.0, (B, R, 1)).astype(np.float32),       # capacity used
         np.eye(3)[rng.integers(0, 3, (B, R))].astype(np.float32),  # type one-hot
        ], axis=-1,
    )                                                               # (B, R, 7)
    # raw 8-dim task feature: pickup(2)+deliv(2)+payload+tw_start+tw_end+priority
    task_raw = np.concatenate(
        [task_pickup, task_deliv,
         rng.uniform(0.2, 1.0, (B, M, 1)).astype(np.float32),
         rng.uniform(0.0, 100, (B, M, 1)).astype(np.float32),
         rng.uniform(100, 300, (B, M, 1)).astype(np.float32),
         rng.integers(1, 4, (B, M, 1)).astype(np.float32),
        ], axis=-1,
    )                                                               # (B, M, 8)
    # pedestrian features (per batch element)
    rfeat = np.zeros((B, R, 4), np.float32)
    tfeat = np.zeros((B, M, 4), np.float32)
    for b in range(B):
        rfeat[b] = compute_ped_features(robot_pos[b], ped_pos, ped_vel)
        tfeat[b] = compute_ped_features(task_pickup[b], ped_pos, ped_vel)

    robot_feat = np.concatenate([robot_raw, rfeat], axis=-1)        # (B, R, 11)
    task_feat = np.concatenate([task_raw, tfeat], axis=-1)          # (B, M, 12)

    feas = np.ones((B, R, M), bool)
    cvar = np.ones((B, R, M), bool)
    margin = np.full((B, R, M), 5.0, np.float32)
    for b in range(B):
        m, mg = cvar_safety_mask(
            robot_pos=robot_pos[b], robot_radius=robot_rad,
            robot_speed=robot_speed,
            task_pickup=task_pickup[b],
            ped_pos=ped_pos, ped_vel=ped_vel, ped_radius=ped_rad,
            dt=0.25, horizon=4, alpha=0.05,
        )
        cvar[b] = m
        margin[b] = mg / (robot_rad[:, None] + ped_rad[None, :].mean())
    return SHARPInput(
        robot_feat=torch.from_numpy(robot_feat),
        robot_pos=torch.from_numpy(robot_pos),
        task_feat=torch.from_numpy(task_feat),
        task_pos=torch.from_numpy(task_pickup),
        feas_mask=torch.from_numpy(feas),
        cvar_mask=torch.from_numpy(cvar),
        margin_norm=torch.from_numpy(margin),
    )


def main():
    torch.manual_seed(0)
    model = SHARP()
    inp = make_synth_batch()
    print(f"params: {model.num_params:,}")
    with torch.no_grad():
        logits, v = model(inp)
    print(f"logits  shape={tuple(logits.shape)}  finite={torch.isfinite(logits).any().item()}  -inf%={(~torch.isfinite(logits)).float().mean().item():.2%}")
    print(f"value   shape={tuple(v.shape)}  v={v.tolist()}")
    # softmax over tasks per (batch, robot)
    p = torch.softmax(logits, dim=-1)
    print(f"row sums (should be 1 or NaN if all-masked): "
          f"{p.sum(-1).tolist()}")
    # sample one assignment per row
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    print(f"sampled actions shape={tuple(a.shape)} examples row0={a[0].tolist()[:5]}")
    print("OK")


if __name__ == "__main__":
    main()
