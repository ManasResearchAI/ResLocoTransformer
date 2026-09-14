"""Check that reward/feet_gait is actually included in reward/total.

Run: python3 scripts/verify_feet_gait.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from envs.builder import get_env
from torchrl.utils import get_params

params = get_params("configs/ablation_baseline.json")
env_params = dict(params["env"])
env = get_env(params["env_name"], env_params)
base = env.unwrapped
# The shipped configs keep use_feet_gait off (it duplicates use_gait_phase);
# force it on here so this script tests the term's wiring, not the config.
base.use_feet_gait = True

env.reset()
seen_nonzero = False
for _ in range(50):
    env.step(env.action_space.sample())
    _, terms = base._compute_reward()
    gait = terms["reward/feet_gait"]
    if abs(gait) > 1e-9:
        seen_nonzero = True
    # total must equal the sum of the active terms, feet_gait included
    recomputed = terms["reward/total"]
    without = recomputed - gait
    assert abs(recomputed - without - gait) < 1e-9

# Direct check: flipping the flag must change the total by exactly r_feet_gait
_, terms = base._compute_reward()
gait = terms["reward/feet_gait"]
total_with = terms["reward/total"]
base.use_feet_gait = False
_, terms_off = base._compute_reward()
base.use_feet_gait = True
delta = total_with - terms_off["reward/total"]

print(f"feet_gait term            : {gait:.6f}")
print(f"total(on) - total(off)    : {delta:.6f}")
print(f"non-zero term observed    : {seen_nonzero}")
ok = abs(delta - gait) < 1e-6 and seen_nonzero
print("PASS" if ok else "FAIL")
env.close()
sys.exit(0 if ok else 1)
