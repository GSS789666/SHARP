"""Run a non-learning baseline policy on the RL env, log metrics."""
import argparse, json, os, sys, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sharp.env import WarehouseRLEnv
from sharp.baselines import GreedyCAHTPolicy, MILPPolicy

POLICIES = {"caht": GreedyCAHTPolicy, "milp": MILPPolicy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--policy", choices=list(POLICIES.keys()), default="caht")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--scenario", default=None,
                    help="optional scenario key from config/scenarios.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.scenario:
        with open(os.path.join(ROOT, "config/scenarios.yaml"), encoding="utf-8") as f:
            scn = yaml.safe_load(f)
        if args.scenario not in scn["scenarios"]:
            sys.exit(f"unknown scenario {args.scenario}")
        for k, v in scn["scenarios"][args.scenario].items():
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v
    if args.num_tasks is not None:
        cfg["env"]["num_tasks"] = args.num_tasks
    if args.max_steps is not None:
        cfg["env"]["max_steps"] = args.max_steps

    policy = POLICIES[args.policy]()
    base_seed = args.seed
    env = WarehouseRLEnv(cfg, seed=0)
    rows = []
    for ep in range(args.episodes):
        env.reset(seed=base_seed * 100 + ep)
        while True:
            a = policy.act(env)
            _, info = env.step(a)
            if info.done:
                break
        rows.append({
            "episode": ep,
            "tasks_done": info.info["tasks_done"],
            "collisions": info.info["collisions_total"],
            "reward": info.info["episode_reward"],
            "cost": info.info["episode_cost"],
        })
        print(f"ep {ep}: done={rows[-1]['tasks_done']:3d}/{args.num_tasks} "
              f"col={rows[-1]['collisions']:3d}  R={rows[-1]['reward']:8.1f}")

    if args.out is None:
        args.out = f"results/baseline_{args.policy}.json"
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    avg_done = sum(r["tasks_done"] for r in rows) / len(rows)
    avg_col = sum(r["collisions"] for r in rows) / len(rows)
    print(f"\navg tasks_done={avg_done:.1f}/{args.num_tasks}  avg collisions={avg_col:.1f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
