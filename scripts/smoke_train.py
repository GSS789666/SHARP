"""Phase 3 smoke test — short SHARP training run.

This trains for only ~30 minutes on CPU at default scale; the goal is to
exercise the full RL loop (model + env + primal-dual PPO) end-to-end and
catch any structural bugs before the user launches the real overnight runs.
"""
import argparse
import os
import sys
import yaml
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sharp.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", default="results/smoke")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # tiny problem for smoke test
    cfg["env"]["num_tasks"] = 20
    cfg["env"]["max_steps"] = 400
    cfg["training"]["seed"] = 0

    save_dir = os.path.join(ROOT, args.out)
    print(f"[smoke] iters={args.iters}  steps/iter={args.steps}  save_dir={save_dir}")
    t0 = time.time()
    model, log = train(
        cfg=cfg, save_dir=save_dir,
        n_iters=args.iters, n_steps_per_iter=args.steps,
        device="cpu", log_every=1,
    )
    print(f"[smoke] done in {time.time()-t0:.1f}s")
    if log:
        first, last = log[0], log[-1]
        print(f"  iter {first['iter']}: return={first.get('ep_return_mean',0):.1f} "
              f"cost={first.get('ep_cost_mean',0):.1f} λ={first.get('lambda',0):.3f}")
        print(f"  iter {last['iter']}: return={last.get('ep_return_mean',0):.1f} "
              f"cost={last.get('ep_cost_mean',0):.1f} λ={last.get('lambda',0):.3f}")


if __name__ == "__main__":
    main()
