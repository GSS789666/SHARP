"""Smoke test the RL environment wrapper with a uniform-random policy."""
import os, sys, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sharp.env import WarehouseRLEnv
from sharp.model import SHARP

with open(os.path.join(ROOT, "config/default.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# small problem for a fast smoke test
cfg["env"]["num_tasks"] = 20
cfg["env"]["max_steps"] = 400

torch.manual_seed(0)
env = WarehouseRLEnv(cfg, seed=0, decision_period=4)
model = SHARP(robot_feat_dim=11, task_feat_dim=12)
model.eval()

obs = env.reset()
print(f"params={model.num_params:,}  R={obs.robot_feat.shape[1]}  M={obs.task_feat.shape[1]}")
print(f"action_dim = {env.action_dim}  (= num_tasks + 1 wait)")

n_decisions = 0
n_assigns = 0
n_waits = 0
n_unfeasible = 0
while True:
    with torch.no_grad():
        logits, value = model(obs)            # logits: (1, R, M+1)
    lead = env.lead_loader_id
    if lead is None:
        # nobody to act this step → wait
        action = env.num_tasks
    else:
        row = logits[0, lead]                 # (M+1,)
        # mask check
        if torch.isfinite(row).any():
            dist = torch.distributions.Categorical(logits=row)
            action = int(dist.sample().item())
        else:
            action = env.num_tasks
    obs, info = env.step(action)
    n_decisions += 1
    if action == env.num_tasks:
        n_waits += 1
    else:
        # success only counted by side effect via env._try_assign return val
        # we just track whether action was wait
        n_assigns += 1
    if n_decisions % 25 == 0:
        print(env.render_text(), f"  v={value.item():.2f}  a={action}")
    if info.done:
        print(env.render_text(), f"  v={value.item():.2f}  a={action}")
        break

print(f"\nepisode: decisions={n_decisions}  assigns={n_assigns}  waits={n_waits}")
print(f"total reward={info.info['episode_reward']:.1f}  cost={info.info['episode_cost']:.1f}")
print(f"tasks done={info.info['tasks_done']}/{cfg['env']['num_tasks']}  "
      f"collisions={info.info['collisions_total']}")
