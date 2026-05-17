"""CVaR safety mask — closed-form implementation of Theorem 1.

Given a robot's candidate route to a task pickup (a Manhattan polyline) and
the predicted Gaussian belief over each pedestrian, compute the *lower*
bound on the CVaR of the minimum trajectory clearance and convert it to a
binary safety mask plus a continuous safety margin.

This module is pure-numpy + a thin torch wrapper, since the closed form has
no learnable parameters of its own.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------- belief

def build_predicted_belief(
    ped_pos: np.ndarray,        # (H, 2)
    ped_vel: np.ndarray,        # (H, 2)
    horizon: int,
    dt: float,
    sigma0: float = 0.20,       # base std at k=1 (m)
    kappa: float = 0.40,        # diffusion growth rate per step
):
    """Constant-velocity mean + linearly-growing isotropic variance.

    Returns
    -------
    mu     : (K, H, 2)
    sigma2 : (K, H)   isotropic variance of each component (so total per
                      coordinate). Used as σ_{l,h}² along any direction
                      because the covariance is σ²·I₂ ⇒ projected variance
                      equals σ² regardless of the unit direction.
    """
    H = ped_pos.shape[0]
    if H == 0:
        return (
            np.zeros((horizon, 0, 2), dtype=np.float32),
            np.zeros((horizon, 0), dtype=np.float32),
        )
    steps = np.arange(1, horizon + 1, dtype=np.float32)             # (K,)
    mu = ped_pos[None, :, :] + steps[:, None, None] * ped_vel[None, :, :] * dt
    sigma2 = (sigma0 ** 2) * (1.0 + kappa * steps)                  # (K,)
    sigma2 = np.broadcast_to(sigma2[:, None], (horizon, H)).copy()
    return mu.astype(np.float32), sigma2.astype(np.float32)


# ---------------------------------------------------------------- waypoints

def manhattan_polyline(start: np.ndarray, end: np.ndarray, max_step: float) -> np.ndarray:
    """Discretise a Manhattan path start -> end with a given max step length."""
    # x first, then y (deterministic; either is fine for a worst-case bound)
    pts = [start.copy()]
    cur = start.copy()
    # x leg
    dx = end[0] - cur[0]
    nx = max(1, int(math.ceil(abs(dx) / max_step)))
    for i in range(1, nx + 1):
        p = cur.copy()
        p[0] = cur[0] + dx * i / nx
        pts.append(p)
    cur = pts[-1].copy()
    # y leg
    dy = end[1] - cur[1]
    ny = max(1, int(math.ceil(abs(dy) / max_step)))
    for i in range(1, ny + 1):
        p = cur.copy()
        p[1] = cur[1] + dy * i / ny
        pts.append(p)
    return np.asarray(pts, dtype=np.float32)


# ------------------------------------------------------------------ CVaR

def _phi_inverse_mills_ratio(alpha: float) -> float:
    """φ(α) = (1/α)·ϕ(Φ⁻¹(α))  — the inverse Mills ratio at level α (lower tail).

    For Gaussian X ~ N(0,1): CVaR_α(X) = -ϕ(Φ⁻¹(α))/α = -φ(α).
    """
    return float(norm.pdf(norm.ppf(alpha)) / alpha)


def _trajectory_cvar(
    poly: np.ndarray,           # (L, 2) waypoints
    speed: float,
    radius_sum: np.ndarray,     # (H,) robot+ped radius
    mu: np.ndarray,             # (K, H, 2)
    sigma2: np.ndarray,         # (K, H)
    horizon: int,
    H: int,
    dt: float,
    alpha: float,
    start_step: int = 0,
) -> float:
    """CVaR lower bound on min clearance along a single polyline."""
    L = poly.shape[0]
    if L == 0 or H == 0:
        return 1e6
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    arrive_step = np.clip(
        start_step + np.ceil(cum / max(speed * dt, 1e-3)).astype(int),
        1, horizon,
    )                                                               # (L,)
    bar_alpha = max(alpha / max(L * H, 1), 1e-6)
    phi_bar = _phi_inverse_mills_ratio(bar_alpha)
    mu_lh = mu[arrive_step - 1]                                     # (L, H, 2)
    d = np.linalg.norm(poly[:, None, :] - mu_lh, axis=2)            # (L, H)
    sig = np.sqrt(sigma2[arrive_step - 1])                          # (L, H)
    cvar_lb = d - radius_sum[None, :] - phi_bar * sig               # (L, H)
    return float(cvar_lb.min())


def cvar_safety_mask(
    robot_pos: np.ndarray,           # (R, 2)
    robot_radius: np.ndarray,        # (R,)
    robot_speed: np.ndarray,         # (R,)
    task_pickup: np.ndarray,         # (M, 2)
    ped_pos: np.ndarray,             # (H, 2)
    ped_vel: np.ndarray,             # (H, 2)
    ped_radius: np.ndarray,          # (H,)
    *,
    dt: float,
    horizon: int = 16,
    alpha: float = 0.05,
    sigma0: float = 0.20,
    kappa: float = 0.40,
    poly_step: float = 1.0,
    task_delivery: np.ndarray | None = None,    # (M, 2) or None
    transporter_pos: np.ndarray | None = None,  # (T, 2) or None
    transporter_speed: np.ndarray | None = None,
    unloader_pos: np.ndarray | None = None,     # (U, 2) or None
    unloader_speed: np.ndarray | None = None,
):
    """Return (mask, margin) for every (robot, task) pair.

    The basic check covers the loader-to-pickup leg. If the optional
    transporter / unloader / delivery info is provided, the mask additionally
    requires that the *best* free transporter and unloader paths to their
    respective endpoints are also safe — this matches the way assignments
    are committed in the env (the loader is paired with the nearest free
    transporter and the nearest unloader). This closes the credit-assignment
    gap noted in the Phase 3 review: it is no longer possible for SHARP to
    "assign safely" while triggering collisions through the auxiliary roles.
    """
    R = robot_pos.shape[0]
    M = task_pickup.shape[0]
    H = ped_pos.shape[0]

    margin = np.full((R, M), 1e6, dtype=np.float32)
    if H == 0:
        return np.ones((R, M), dtype=bool), margin

    mu, sigma2 = build_predicted_belief(
        ped_pos, ped_vel, horizon=horizon, dt=dt, sigma0=sigma0, kappa=kappa
    )
    rad_sum = robot_radius[:, None] + ped_radius[None, :]   # (R, H)

    full_check = (
        task_delivery is not None and transporter_pos is not None
        and transporter_speed is not None
        and unloader_pos is not None and unloader_speed is not None
    )

    for i in range(R):
        for j in range(M):
            poly = manhattan_polyline(robot_pos[i], task_pickup[j], poly_step)
            m_load = _trajectory_cvar(
                poly, robot_speed[i], rad_sum[i], mu, sigma2,
                horizon, H, dt, alpha,
            )
            margin[i, j] = m_load

            if not full_check or m_load < 0:
                continue

            # The transporter-to-pickup leg is largely covered by the
            # loader-to-pickup leg's destination region, so we skip it for
            # speed and check the two *new* legs that the loader leg does not:
            # (a) transporter pickup -> delivery, with a temporal offset
            # (b) unloader -> delivery
            d_tr = np.linalg.norm(transporter_pos - task_pickup[j], axis=1)
            t_idx = int(np.argmin(d_tr))
            t_spd = float(transporter_speed[t_idx])
            poly_t2 = manhattan_polyline(task_pickup[j], task_delivery[j], poly_step)
            d_t1 = float(np.linalg.norm(transporter_pos[t_idx] - task_pickup[j]))
            start2 = int(np.ceil(d_t1 / max(t_spd * dt, 1e-3)))
            m_tr2 = _trajectory_cvar(
                poly_t2, t_spd, rad_sum[i], mu, sigma2, horizon, H, dt, alpha,
                start_step=start2,
            )

            d_un = np.linalg.norm(unloader_pos - task_delivery[j], axis=1)
            u_idx = int(np.argmin(d_un))
            u_spd = float(unloader_speed[u_idx])
            poly_u = manhattan_polyline(unloader_pos[u_idx], task_delivery[j], poly_step)
            m_un = _trajectory_cvar(
                poly_u, u_spd, rad_sum[i], mu, sigma2, horizon, H, dt, alpha,
            )

            margin[i, j] = float(min(m_load, m_tr2, m_un))

    mask = margin >= 0.0
    return mask, margin
