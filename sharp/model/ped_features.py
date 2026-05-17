"""4-dimensional pedestrian context features for each robot/task node.

Following §3.5 of the method draft, every robot or task token receives a
4-vector summarising the local pedestrian situation in a 5 m disc:
    [count, mean_distance, mean_speed, fraction_approaching]

These features are concatenated to the raw node features before encoding so
the Transformer can attend to safety-critical context without changing its
internal architecture.
"""
from __future__ import annotations
import numpy as np


def compute_ped_features(
    node_pos: np.ndarray,            # (N, 2) m
    ped_pos: np.ndarray,             # (H, 2) m
    ped_vel: np.ndarray,             # (H, 2) m/s
    radius: float = 5.0,
) -> np.ndarray:
    """Compute the 4-dim feature for every node.

    Returns array of shape (N, 4). A node with zero pedestrians in range
    receives a zero vector (no context).
    """
    N = node_pos.shape[0]
    H = ped_pos.shape[0]
    out = np.zeros((N, 4), dtype=np.float32)
    if H == 0:
        return out

    diff = ped_pos[None, :, :] - node_pos[:, None, :]   # (N, H, 2)
    dist = np.linalg.norm(diff, axis=2)                 # (N, H)
    in_range = dist < radius                            # (N, H)
    counts = in_range.sum(axis=1).astype(np.float32)    # (N,)

    # protect against division by zero on nodes with empty neighbourhoods
    safe_counts = np.where(counts > 0, counts, 1.0)

    # mean distance (within range)
    masked_dist = np.where(in_range, dist, 0.0)
    mean_dist = masked_dist.sum(axis=1) / safe_counts

    # mean speed of the in-range pedestrians
    speed = np.linalg.norm(ped_vel, axis=1)             # (H,)
    masked_speed = in_range.astype(np.float32) * speed[None, :]
    mean_speed = masked_speed.sum(axis=1) / safe_counts

    # fraction approaching: pedestrian whose velocity has positive component
    # *toward* the node, i.e.  (-diff)·v > 0  ⇔  diff·v < 0
    inner = (diff * ped_vel[None, :, :]).sum(axis=2)    # (N, H)
    approaching = (in_range & (inner < 0)).astype(np.float32)
    frac_approach = approaching.sum(axis=1) / safe_counts

    out[:, 0] = counts
    out[:, 1] = mean_dist
    out[:, 2] = mean_speed
    out[:, 3] = frac_approach

    # zero rows where there were no neighbours at all
    out[counts == 0] = 0.0
    return out
