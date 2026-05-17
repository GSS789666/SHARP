"""Quick debug script: trace task state transitions."""
import os, sys, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sharp.env import WarehouseEnv, RobotType
from sharp.data import sample_instance
from scripts.demo_simulator import greedy_assign

with open(os.path.join(ROOT, "config/default.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

env = WarehouseEnv(cfg, seed=0)
env.reset_tasks(sample_instance(cfg, seed=0))
print(f"world={env.world_size:.1f}m  tasks={len(env.tasks)}  robots={len(env.robots)}")

prev_states = ["open"] * len(env.tasks)
transitions = []
for step in range(400):
    greedy_assign(env)
    env.step()
    for i, t in enumerate(env.tasks):
        if t.state != prev_states[i]:
            transitions.append((step, i, prev_states[i], t.state))
            prev_states[i] = t.state
    if step % 40 == 0 or step == 399:
        ldr = sum(1 for r in env.robots if r.rtype == RobotType.LOADER and r.busy)
        trn = sum(1 for r in env.robots if r.rtype == RobotType.TRANSPORTER and r.busy)
        unl = sum(1 for r in env.robots if r.rtype == RobotType.UNLOADER and r.busy)
        print(f"step={step:3d} t={env.t:5.1f}  busy L/T/U={ldr}/{trn}/{unl}  "
              f"open={sum(1 for x in env.tasks if x.state=='open'):3d}  "
              f"assigned={sum(1 for x in env.tasks if x.state=='assigned'):3d}  "
              f"transit={sum(1 for x in env.tasks if x.state=='transit'):3d}  "
              f"done={sum(1 for x in env.tasks if x.state=='done'):3d}")

print("\nfirst 20 transitions:")
for tr in transitions[:20]:
    print(f"  step={tr[0]}  task={tr[1]}  {tr[2]} -> {tr[3]}")

# look at one assigned task
ex = next((t for t in env.tasks if t.state in ("assigned","transit")), None)
if ex:
    print(f"\nexample stuck task tid={ex.tid} state={ex.state} assigned_to={ex.assigned_to}")
    for rid in ex.assigned_to:
        r = env.robots[rid]
        wp = ", ".join([f"({w[0]:.1f},{w[1]:.1f})" for w in r.waypoints])
        print(f"  robot {rid} ({r.rtype.value})  pos=({r.pos[0]:.2f},{r.pos[1]:.2f})  "
              f"busy={r.busy}  waypoints=[{wp}]")
    print(f"  pickup=({ex.pickup[0]:.2f},{ex.pickup[1]:.2f})  delivery=({ex.delivery[0]:.2f},{ex.delivery[1]:.2f})")
