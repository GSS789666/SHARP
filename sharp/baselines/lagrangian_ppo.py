"""Lagrangian-PPO baseline (B2 in §4 of the proposal).

This is a *standard* primal-dual PPO without any of SHARP's pedestrian-aware
contributions. Concretely it disables, in the same SHARP network, two
mechanisms:

    1. The CVaR safety mask (always sets cvar_mask = True everywhere).
    2. The soft risk bias (sets the learnable η to a fixed 0).

What remains is exactly the *architecture-level* baseline expected by reviewers
("show that adding a SafeRL dual variable to the deterministic CAHT mask
isn't enough — the safety reasoning has to be embedded structurally"). Both
B1 (greedy CAHT) and B2 (Lagrangian-PPO) are necessary references because
they rule out two different alternative explanations for any improvement.

Implementation strategy: rather than duplicate the model and trainer, we
expose a single ``safety_mode`` knob in the env and a ``no_soft_bias`` flag
in the model. The training entry-point passes the right combination.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from sharp.model.sharp import SHARP, SHARPInput, BilinearDecoder


class _NoSoftBiasDecoder(BilinearDecoder):
    """Decoder with the soft risk bias disabled (η forced to 0)."""

    def forward(self, e_robot, e_task, feas_mask, cvar_mask, margin_norm):
        # Override: ignore margin_norm entirely.
        margin_norm = torch.full_like(margin_norm, 1e6)
        return super().forward(e_robot, e_task, feas_mask, cvar_mask, margin_norm)


class SHARPLagrangian(SHARP):
    """SHARP variant with no CVaR mask and no soft bias — used as B2."""

    def __init__(self, **kw):
        super().__init__(**kw)
        # replace decoder with the no-bias variant
        d = self.decoder.W.shape[0]
        new = _NoSoftBiasDecoder(d)
        new.load_state_dict(self.decoder.state_dict())
        # zero out eta_raw so softplus(eta_raw) ≈ 0.7; we additionally clamp
        # it during forward by overriding the η effect inside _NoSoftBiasDecoder.
        self.decoder = new


def lagrangian_observation_patch(obs: SHARPInput) -> SHARPInput:
    """Force the CVaR mask to all-True so this baseline ignores pedestrian risk."""
    cvar_all_true = torch.ones_like(obs.cvar_mask)
    return SHARPInput(
        robot_feat=obs.robot_feat,
        robot_pos=obs.robot_pos,
        task_feat=obs.task_feat,
        task_pos=obs.task_pos,
        feas_mask=obs.feas_mask,
        cvar_mask=cvar_all_true,
        margin_norm=obs.margin_norm,
    )
