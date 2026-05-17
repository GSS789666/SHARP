"""Profile observation/CVaR cost."""
import os, sys, time, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sharp.env import WarehouseRLEnv

with open(os.path.join(ROOT, "config/default.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["env"]["num_tasks"] = 20

env = WarehouseRLEnv(cfg, seed=0, safety_mode="cvar")
env.reset()
N = 20
t0 = time.time()
for _ in range(N):
    env._observation()
print(f"cvar-on  avg: {(time.time()-t0)/N*1000:.1f} ms")

env2 = WarehouseRLEnv(cfg, seed=0, safety_mode="off")
env2.reset()
t0 = time.time()
for _ in range(N):
    env2._observation()
print(f"cvar-off avg: {(time.time()-t0)/N*1000:.1f} ms")
