"""SHARP: pedestrian-aware extension of CAHT.

Architecture (per §3.5 of method draft):
    - Per-token feature: raw entity features ⊕ 4-dim pedestrian context.
    - Spatial-bias Transformer encoder (4 layers, d=128, 8 heads).
    - Bilinear assignment decoder with two-layer mask:
          mask_total = feasibility_mask & cvar_safety_mask
      and a learnable soft risk bias (eq. 3 of method draft).
    - Value head for PPO.

The architecture is intentionally close to CAHT so its theoretical claims
about constraint satisfaction transfer; the *new* mechanisms are confined
to the input pipeline (pedestrian features) and the decoder's masking +
soft bias logic.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================== encoder

class SpatialBiasMHA(nn.Module):
    """Multi-head self-attention with a learned scalar bias per pair distance.

    The original CAHT paper computes b_{ij} = MLP(||x_i - x_j||) and adds it
    to the QK^T logits. We use the same recipe with a tiny 2-layer MLP.
    """

    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.d = d
        self.h = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, heads),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x:   (B, N, d)
        # pos: (B, N, 2) for spatial bias
        B, N, _ = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.dh)
        q, k, v = qkv.unbind(dim=2)                # (B, N, H, dh) each
        # logits
        logits = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.dh)
        # spatial bias
        d_ij = torch.linalg.norm(
            pos[:, :, None, :] - pos[:, None, :, :], dim=-1, keepdim=True
        )                                          # (B, N, N, 1)
        bias = self.dist_mlp(d_ij).permute(0, 3, 1, 2)   # (B, H, N, N)
        logits = logits + bias
        attn = logits.softmax(dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(B, N, self.d)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.attn = SpatialBiasMHA(d, heads)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d),
        )

    def forward(self, x, pos):
        x = x + self.attn(self.ln1(x), pos)
        x = x + self.mlp(self.ln2(x))
        return x


class SpatialBiasEncoder(nn.Module):
    def __init__(self, d_in: int, d: int = 128, layers: int = 4, heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList(
            [EncoderLayer(d, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x, pos)
        return x


# =============================================================== decoder

class BilinearDecoder(nn.Module):
    """Bilinear robot-task scorer with two-layer mask, soft risk bias, and a
    *wait* column that is never masked.

    The wait column corresponds to action index ``M`` and has a learnable
    scalar logit shared across robots. Including it guarantees every row has
    at least one feasible action, which both prevents NaN gradients and
    realises the ``delay_and_wait`` mode used in the Mivar-style discussion
    of §1.
    """

    def __init__(self, d: int = 128):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W)
        # learnable temperature for the soft risk bias (clamped via softplus)
        self.eta_raw = nn.Parameter(torch.tensor(0.5))   # softplus(0.5)≈0.97
        self.tau = 2.0  # margin threshold in robot-radii (fixed hyperparam)
        # learnable wait logit (per robot, independent of state)
        self.wait_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.eta_raw)

    def forward(
        self,
        e_robot: torch.Tensor,        # (B, R, d)
        e_task:  torch.Tensor,        # (B, M, d)
        feas_mask: torch.Tensor,      # (B, R, M) bool, True = feasible
        cvar_mask: torch.Tensor,      # (B, R, M) bool, True = safe
        margin_norm: torch.Tensor,    # (B, R, M) float, normalised by ρ
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, R, _ = e_robot.shape
        # bilinear logits
        logits = torch.einsum("brd,de,bme->brm", e_robot, self.W, e_task)
        # soft risk bias (eq. 3): only where positive but small margin
        soft = torch.clamp(self.tau - margin_norm, min=0.0)
        logits = logits - self.eta * soft
        # two-layer hard mask
        mask = feas_mask & cvar_mask
        logits = logits.masked_fill(~mask, float("-inf"))
        # append the always-feasible wait column
        wait = self.wait_logit.expand(B, R, 1)
        logits = torch.cat([logits, wait], dim=-1)         # (B, R, M+1)
        wait_mask = torch.ones(B, R, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask, wait_mask], dim=-1)
        return logits, mask


# =============================================================== top-level

@dataclass
class SHARPInput:
    """Container for one decision step."""
    robot_feat: torch.Tensor    # (B, R, F_r)
    robot_pos:  torch.Tensor    # (B, R, 2)
    task_feat:  torch.Tensor    # (B, M, F_t)
    task_pos:   torch.Tensor    # (B, M, 2)
    feas_mask:  torch.Tensor    # (B, R, M)
    cvar_mask:  torch.Tensor    # (B, R, M)
    margin_norm: torch.Tensor   # (B, R, M)


class SHARP(nn.Module):
    """End-to-end pedestrian-aware allocation policy + value."""

    def __init__(
        self,
        robot_feat_dim: int = 11,    # raw 7 + ped 4
        task_feat_dim: int = 12,     # raw 8 + ped 4
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        # separate projections for robots and tasks (two token types)
        self.robot_proj = nn.Linear(robot_feat_dim, d_model)
        self.task_proj = nn.Linear(task_feat_dim, d_model)
        self.encoder = SpatialBiasEncoder(d_model, d_model, n_layers, n_heads)
        self.decoder = BilinearDecoder(d_model)
        # value head: pool encoder output -> scalar
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, inp: SHARPInput) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate robots and tasks into one token stream so the encoder
        # can attend across types, but project separately first to respect
        # type-specific feature dimensionality.
        e_r = self.robot_proj(inp.robot_feat)          # (B, R, d)
        e_t = self.task_proj(inp.task_feat)            # (B, M, d)
        x = torch.cat([e_r, e_t], dim=1)               # (B, R+M, d)
        pos = torch.cat([inp.robot_pos, inp.task_pos], dim=1)
        x = self.encoder(x, pos)
        R = inp.robot_feat.shape[1]
        e_r2, e_t2 = x[:, :R], x[:, R:]
        logits, _ = self.decoder(
            e_r2, e_t2, inp.feas_mask, inp.cvar_mask, inp.margin_norm,
        )
        # value: average over all tokens
        v = self.value_head(x.mean(dim=1)).squeeze(-1)   # (B,)
        return logits, v
