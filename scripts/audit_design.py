"""Compare experiment definitions against a fetched official source snapshot."""

import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
upstream = root / "audit/upstream"


def nodes(path):
    return ast.parse(path.read_text()).body


def find(path, name):
    for node in nodes(path):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise KeyError(name)


def equal(left, right, name):
    a, b = find(left, name), find(right, name)
    # Descriptive docstrings are not experiment parameters.
    for node in (a, b):
        if isinstance(node, ast.ClassDef):
            node.body = [n for n in node.body if not isinstance(n, ast.Expr)]
    return ast.dump(a) == ast.dump(b)


checks = {
    "terrain_all_parameters": equal(
        upstream / "parkour_env_cfg.py", root / "hikinglab3/terrain_cfg.py", "ROUGH_TERRAINS_CFG"
    )
}
for name in ["ActionsCfg", "CommandsCfg", "G1Rewards", "CurriculumCfg"]:
    checks[name] = equal(upstream / "parkour_env_cfg.py", root / "hikinglab3/task_terms_cfg.py", name)
for name in ["DepthEncoderConv2dCfg", "EncoderConfigs", "MoEPolicyCfg", "AmpAlgoCfg", "G1ParkourPPORunnerCfg"]:
    checks[name] = equal(upstream / "instinct_rl_amp_cfg.py", root / "hikinglab3/agent_cfg.py", name)
(root / "audit/design_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
assert all(checks.values()), "Experiment definitions differ from official snapshot"
