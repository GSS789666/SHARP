"""End-to-end smoke test for the SHARP simulator.

Loads default config, generates one task instance, runs a *greedy nearest*
allocator, and saves a sequence of frames + an animated GIF.

Usage:
    python scripts/demo_simulator.py --steps 200 --gif results/demo.gif
"""
from __future__ import annotations
import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow ``python scripts/...`` from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sharp.env import WarehouseEnv, RobotType
from sharp.data import sample_instance
from sharp.utils import render_frame, save_animation


def greedy_assign(env: WarehouseEnv) -> int:
    """Assign as many open tasks as possible using nearest free robots.

    Returns the number of new assignments this call.
    """
    new_assigns = 0
    free_loaders = [r for r in env.robots if r.rtype == RobotType.LOADER and not r.busy]
    free_trans = [r for r in env.robots if r.rtype == RobotType.TRANSPORTER and not r.busy]
    free_unl = [r for r in env.robots if r.rtype == RobotType.UNLOADER and not r.busy]
    if not (free_loaders and free_trans and free_unl):
        return 0
    open_tasks = [t for t in env.tasks if t.state == "open"]
    open_tasks.sort(key=lambda t: t.tw_start)

    for task in open_tasks:
        if not (free_loaders and free_trans and free_unl):
            break
        l = min(free_loaders, key=lambda r: float(np.linalg.norm(r.pos - task.pickup)))
        t = min(free_trans, key=lambda r: float(np.linalg.norm(r.pos - task.pickup)))
        u = min(free_unl, key=lambda r: float(np.linalg.norm(r.pos - task.delivery)))
        if env.assign(task.tid, l.rid, t.rid, u.rid):
            new_assigns += 1
            free_loaders.remove(l)
            free_trans.remove(t)
            free_unl.remove(u)
    return new_assigns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--gif", default="results/demo.gif")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames-every", type=int, default=4,
                    help="capture a frame every N sim steps")
    args = ap.parse_args()

    cfg_path = os.path.join(ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = WarehouseEnv(cfg=cfg, seed=args.seed)
    env.reset_tasks(sample_instance(cfg, seed=args.seed))

    print(f"[demo] world {env.world_size:.1f} m, robots={len(env.robots)}, "
          f"tasks={len(env.tasks)}, peds={env.crowd.n}")

    figs = []
    capture = args.gif and args.gif.lower() != "none"
    for step in range(args.steps):
        greedy_assign(env)
        info = env.step()
        if capture and step % args.frames_every == 0:
            figs.append(render_frame(env))
        if step % 50 == 0:
            print(f"  step={step:4d} done={info['tasks_done']:3d}  "
                  f"col_total={info['collisions_total']}")
        if env.done():
            print(f"[demo] all tasks done at step {step}")
            break

    print(f"[demo] finished. tasks_done={sum(1 for k in env.tasks if k.state=='done')}/"
          f"{len(env.tasks)}, collisions={env.collisions}, "
          f"sim_time={env.t:.1f}s")

    if capture and figs:
        out = os.path.join(ROOT, args.gif)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_animation(figs, out, fps=6)


if __name__ == "__main__":
    main()
