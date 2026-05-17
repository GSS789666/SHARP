"""Matplotlib visualisation for SHARP simulation snapshots."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches


ROBOT_COLORS = {
    "loader": "#1f77b4",       # blue
    "transporter": "#2ca02c",  # green
    "unloader": "#d62728",     # red
}
PED_COLORS = {
    "adult": "#7f7f7f",
    "elderly": "#9467bd",
    "child": "#ff7f0e",
}


def render_frame(env, ax=None, title: str = "") -> plt.Figure:
    """Draw the current env state. Returns the figure.

    The figure is created via ``plt.figure`` *without* registering it on the
    pyplot global state to avoid leaks when called repeatedly inside a loop.
    """
    own_fig = ax is None
    if own_fig:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure
    W = env.world_size
    ax.set_xlim(0, W)
    ax.set_ylim(0, W)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")

    # tasks (open / done)
    for task in env.tasks:
        if task.state == "open":
            ax.plot(*task.pickup, marker="^", color="black", markersize=5, alpha=0.4)
            ax.plot(*task.delivery, marker="v", color="black", markersize=5, alpha=0.4)
        elif task.state == "done":
            ax.plot(*task.delivery, marker="x", color="green", markersize=4, alpha=0.6)

    # pedestrians
    for i in range(env.crowd.n):
        kls = str(env.crowd.klasses[i])
        c = PED_COLORS.get(kls, "gray")
        circ = patches.Circle(
            env.crowd.pos[i], radius=env.crowd.radius[i], color=c, alpha=0.55
        )
        ax.add_patch(circ)
        v = env.crowd.vel[i]
        ax.arrow(
            env.crowd.pos[i, 0], env.crowd.pos[i, 1],
            v[0] * 0.6, v[1] * 0.6,
            color=c, head_width=0.15, alpha=0.7, length_includes_head=True,
        )

    # robots
    for r in env.robots:
        c = ROBOT_COLORS[r.rtype.value]
        circ = patches.Circle(r.pos, radius=r.radius, color=c, alpha=0.9)
        ax.add_patch(circ)
        if r.waypoints:
            tx, ty = r.waypoints[0]
            ax.plot([r.pos[0], tx], [r.pos[1], ty], color=c, linestyle="--",
                    alpha=0.6, linewidth=0.8)
        ax.text(r.pos[0] + 0.3, r.pos[1] + 0.3, str(r.rid),
                fontsize=7, color=c, weight="bold")

    if not title:
        title = (
            f"t={env.t:5.1f}s  step={env.step_idx:4d}  "
            f"done={sum(1 for k in env.tasks if k.state == 'done')}/{len(env.tasks)}  "
            f"col={env.collisions}"
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2, linewidth=0.4)
    return fig


def save_animation(snapshots, save_path: str, fps: int = 6) -> None:
    """Render a sequence of snapshots to a .mp4 (or .gif fallback)."""
    import matplotlib.animation as animation
    from PIL import Image
    import io

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    images = []
    for snap in snapshots:
        fig = snap  # snap is a pre-rendered Figure (not pyplot-attached)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    if save_path.lower().endswith(".gif"):
        images[0].save(
            save_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
    else:
        # fall back to gif if ffmpeg missing
        save_path = save_path.rsplit(".", 1)[0] + ".gif"
        images[0].save(
            save_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
    print(f"[viz] saved {save_path}")
