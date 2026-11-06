"""Summarize recorded PPO learning curves without inferring convergence."""

import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("log", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
text = re.sub(r"\x1b\[[0-9;]*m", "", a.log.read_text())
blocks = re.split(r"Learning iteration (\d+)/(\d+)", text)
records = []
for i in range(1, len(blocks), 3):
    row = {"iteration": int(blocks[i])}
    content = blocks[i + 2]
    for label, key in [
        ("Mean episode length", "episode_steps"),
        ("Mean reward", "reward"),
        ("Mean episode Curriculum/terrain_levels", "terrain_level"),
        ("Mean episode Episode_Termination/time_out", "timeout_fraction"),
        ("Mean episode Episode_Termination/bad_orientation", "orientation_failure_fraction"),
        ("Mean episode Episode_Reward/dof_vel_limits", "speed_limit_reward"),
        ("Mean episode Episode_Reward/pelvis_orientation_l2", "pelvis_orientation_reward"),
    ]:
        match = re.search(re.escape(label) + r":\s+([-+\d.eE]+)", content)
        if match:
            row[key] = float(match.group(1))
    if "episode_steps" in row:
        row["episode_seconds"] = row["episode_steps"] * 0.02
        records.append(row)
summary = {"iterations_recorded": len(records), "last": records[-1] if records else None}
for name, values in [("first_100", records[:100]), ("last_100", records[-100:])]:
    summary[name] = {
        key: sum(r[key] for r in values if key in r) / sum(key in r for r in values)
        for key in (values[0].keys() if values else [])
        if key != "iteration"
    }
summary["interpretation"] = "Training diagnostics; not formal 1000-agent paper benchmark or proof of convergence."
summary["curve"] = records
out = a.output
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "curve"}, indent=2))
