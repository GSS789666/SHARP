"""Single-run launcher for a (algorithm, scenario, seed) triple.

This is the unit of work the user fires from cmd. Each invocation reads the
shared :file:`config/default.yaml`, layers a scenario override from
:file:`config/scenarios.yaml`, then trains for a fixed number of iterations
and saves all artefacts under :file:`results/main/<scenario>/<algo>/seed_<k>/`.
"""
import argparse
import json
import os
import sys
import time
import yaml
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sharp.train import train


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    help="key under scenarios: in scenarios.yaml")
    ap.add_argument("--algo", required=True,
                    help="key under algorithms: in scenarios.yaml or "
                         "ablations: in ablations.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out-root", default="results/main")
    ap.add_argument("--ablation", action="store_true",
                    help="resolve --algo against ablations.yaml instead")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="override risk.epsilon_safety (per-step collision "
                         "rate budget). When provided, also tightens "
                         "risk.cvar_alpha proportionally so the mask scales "
                         "with the deployment safety target.")
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda — auto-detects cuda if available")
    args = ap.parse_args()

    # load + merge configs
    with open(os.path.join(ROOT, "config/default.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "config/scenarios.yaml"), encoding="utf-8") as f:
        scn = yaml.safe_load(f)
    if args.scenario not in scn["scenarios"]:
        sys.exit(f"unknown scenario {args.scenario}; "
                 f"available: {list(scn['scenarios'])}")
    deep_update(cfg, scn["scenarios"][args.scenario])
    cfg["training"]["seed"] = args.seed

    # epsilon override (and proportional CVaR alpha tightening). The default
    # mask uses cvar_alpha = 0.20 with epsilon_safety = 0.05; we keep the
    # ratio cvar_alpha = 4 × epsilon_safety with a floor of 0.02 so the mask
    # never collapses to "all unsafe" during training.
    if args.epsilon is not None:
        cfg["risk"]["epsilon_safety"] = float(args.epsilon)
        cfg["risk"]["cvar_alpha"] = max(4.0 * float(args.epsilon), 0.02)

    if args.ablation:
        with open(os.path.join(ROOT, "config/ablations.yaml"), encoding="utf-8") as f:
            abl = yaml.safe_load(f)
        if args.algo not in abl["ablations"]:
            sys.exit(f"unknown ablation {args.algo}; "
                     f"available: {list(abl['ablations'])}")
        algo_cfg = abl["ablations"][args.algo]
    else:
        if args.algo not in scn["algorithms"]:
            sys.exit(f"unknown algo {args.algo}; "
                     f"available: {list(scn['algorithms'])}")
        algo_cfg = scn["algorithms"][args.algo]

    # fix all RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # safety mode for the env wrapper
    safety_mode = algo_cfg["safety_mode"]
    fix_lambda = algo_cfg.get("fix_lambda", None)
    no_soft_bias = algo_cfg.get("no_soft_bias", False)

    save_dir = os.path.join(
        ROOT, args.out_root, args.scenario, args.algo, f"seed_{args.seed}",
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print(f"[run] scenario={args.scenario}  algo={args.algo}  seed={args.seed}")
    print(f"[run] safety_mode={safety_mode}  iters={args.iters}  steps={args.steps}")
    print(f"[run] epsilon={cfg['risk']['epsilon_safety']:.4f}  "
          f"cvar_alpha={cfg['risk']['cvar_alpha']:.3f}")
    print(f"[run] save_dir={save_dir}")

    t0 = time.time()
    # the trainer always builds the full env; we monkey-patch its safety_mode
    # via an internal hook.
    from sharp.env import WarehouseRLEnv
    from sharp.train.ppo_primal_dual import PrimalDualPPO, collect_rollout
    from sharp.model import SHARP
    from tqdm import trange

    # auto-pick device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[run] WARN: --device cuda requested but CUDA not available, "
              "falling back to cpu")
        device = "cpu"
    if device == "cpu" and torch.cuda.is_available() and args.device == "cpu":
        # silent: user explicitly asked cpu
        pass
    print(f"[run] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    env = WarehouseRLEnv(cfg, seed=args.seed, safety_mode=safety_mode)
    model = SHARP()
    if no_soft_bias:
        # disable the soft risk bias by clamping eta_raw to a very negative
        # value (softplus(-1e3) ≈ 0) and freezing it
        with torch.no_grad():
            model.decoder.eta_raw.fill_(-1e3)
        model.decoder.eta_raw.requires_grad_(False)
    algo = PrimalDualPPO(model, env, cfg, device=device)
    if fix_lambda is not None:
        algo.lambda_ = float(fix_lambda)
        algo.lr_dual = 0.0   # freeze dual

    log = []
    pbar = trange(args.iters, dynamic_ncols=True)
    for it in pbar:
        buf, diag = collect_rollout(env, model, args.steps, device=device)
        upd = algo.update(buf, diag["last_value"])
        row = {"iter": it, **diag, **upd}
        log.append(row)
        pbar.set_postfix({
            "ret": f"{diag['ep_return_mean']:.1f}",
            "cost": f"{diag['ep_cost_mean']:.1f}",
            "λ": f"{algo.lambda_:.3f}",
        })
        with open(os.path.join(save_dir, "train_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if (it + 1) % 25 == 0 or it == args.iters - 1:
            torch.save(
                {"model": model.state_dict(), "lambda": algo.lambda_,
                 "iter": it, "cfg": cfg, "args": vars(args)},
                os.path.join(save_dir, f"ckpt_{it+1:04d}.pt"),
            )
    elapsed = time.time() - t0
    summary = {
        "scenario": args.scenario, "algo": args.algo, "seed": args.seed,
        "iters": args.iters, "steps": args.steps,
        "elapsed_s": elapsed,
        "first": log[0] if log else {},
        "last": log[-1] if log else {},
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[run] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
