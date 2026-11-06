"""Measure the unchanged training workload in a bounded foreground child.

Device memory is sampled, so its observed peak is a lower bound on the true peak.
This is a capacity/throughput check, not a policy-quality evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time


def memory_sample():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0]) / 1024
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=5,
    ).splitlines()[0]
    used, free = map(float, gpu.split(","))
    return {"gpu_used_mib": used, "gpu_free_mib": free,
            "ram_available_mib": fields["MemAvailable"],
            "swap_used_mib": fields["SwapTotal"] - fields["SwapFree"]}


def children(pid):
    try:
        direct = [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []
    return [descendant for child in direct for descendant in [*children(child), child]]


def stop_owned_process(proc):
    # Never signal unrelated processes or the user's terminal/process group.
    owned = children(proc.pid) + [proc.pid]
    for pid in owned:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for pid in children(proc.pid) + [proc.pid]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--backend", choices=("newton", "physx"), default="newton")
    parser.add_argument("--iterations", default=12, type=int)
    parser.add_argument("--warmup-updates", default=3, type=int)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--reserve-vram-mib", default=768, type=int)
    parser.add_argument("--reserve-ram-mib", default=4096, type=int)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= args.warmup_updates:
        parser.error("positive environments and iterations > warmup required")
    root = Path(__file__).resolve().parents[1]
    log = args.log_dir.resolve()
    log.mkdir(parents=True, exist_ok=False)
    command = [str(root / "run.sh"), "--mode", "train", "--profile", "original",
               "--num-envs", str(args.num_envs), "--iterations", str(args.iterations),
               "--log-dir", str(log)]
    child_cwd = root
    child_env = os.environ.copy()
    child_env["OMP_NUM_THREADS"] = "8"
    child_env["PYTHONUNBUFFERED"] = "1"
    if args.backend == "physx":
        child_cwd = Path("/home/ye/hiking_in_the_wild/agibot_x2_hiking_in_the_wild")
        conda = Path("/home/ye/miniconda3/envs/Instinct")
        motion_dir = root / "data/official/data&model/parkour_motion_reference"
        child_env["CONDA_PREFIX"] = str(conda)
        child_env["PATH"] = str(conda / "bin") + os.pathsep + child_env.get("PATH", "")
        command = [str(conda / "bin/python"), "scripts/instinct_rl/train.py", "--headless",
                   "--task", "Instinct-Parkour-Target-Amp-G1-v0",
                   "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
                   "--seed", "42", "--logroot", str(log / "experiments"),
                   "env.scene.motion_reference.motion_buffers.run_walk.path=" + json.dumps(str(motion_dir)),
                   "env.scene.motion_reference.motion_buffers.run_walk.filtered_motion_selection_filepath="
                   + json.dumps(str(motion_dir / "parkour_motion_without_run.yaml")),
                   "agent.save_interval=" + str(args.iterations),
                   "env.rewards.rewards.freeze_upper_body.params.asset_cfg.joint_names="
                   + json.dumps([".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"])]
    (log / "benchmark_command.json").write_text(json.dumps(command, indent=2))
    samples = []
    baseline = memory_sample()
    if (baseline["gpu_free_mib"] < args.reserve_vram_mib or
            baseline["ram_available_mib"] < args.reserve_ram_mib):
        raise RuntimeError("Insufficient headroom before starting the workload")
    started = time.monotonic()
    reason = None
    with (log / "console.log").open("w") as output:
        proc = subprocess.Popen(command, cwd=child_cwd, env=child_env, stdout=output, stderr=subprocess.STDOUT)
        try:
            next_status = 0
            while proc.poll() is None:
                sample = {"elapsed_s": round(time.monotonic() - started, 3), **memory_sample()}
                samples.append(sample)
                if sample["elapsed_s"] >= next_status:
                    print(json.dumps(sample), flush=True)
                    next_status = sample["elapsed_s"] + 30
                if sample["gpu_free_mib"] < args.reserve_vram_mib:
                    reason = "stopped_at_vram_reserve"
                elif sample["ram_available_mib"] < args.reserve_ram_mib:
                    reason = "stopped_at_ram_reserve"
                if reason:
                    stop_owned_process(proc)
                    break
                time.sleep(0.5)
        finally:
            if proc.poll() is None:
                stop_owned_process(proc)
    elapsed = time.monotonic() - started
    (log / "resource_samples.json").write_text(json.dumps(samples, indent=2))
    content = (log / "console.log").read_text(errors="replace")
    timings = re.findall(r"Computation:\s+(\d+) steps/s \(collection: ([\d.]+)s, learning ([\d.]+)s\)", content)
    # Handle historical logger variants without changing its timing instrumentation.
    if not timings:
        timings = re.findall(r"Computation:\s+(\d+) steps/s \(collection: ([\d.]+)s, learning: ([\d.]+)s\)", content)
    steady = [(int(s), float(c), float(l)) for s, c, l in timings[args.warmup_updates:]]
    result_path = log / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if args.backend == "physx" and len(timings) == args.iterations and proc.returncode == 0:
        result = {"mode": "train", "iterations": len(timings), "num_envs": args.num_envs,
                  "seconds_from_rounded_iteration_timings": sum(float(c) + float(l) for _, c, l in timings)}
    report = {
        "backend": args.backend,
        "num_envs": args.num_envs, "requested_updates": args.iterations,
        "completed": proc.returncode == 0 and result is not None,
        "returncode": proc.returncode, "stop_reason": reason,
        "foreground_wall_seconds": elapsed, "training_result": result,
        "baseline": baseline,
        "observed_device_peak_mib": max(s["gpu_used_mib"] for s in samples),
        "minimum_device_free_mib": min(s["gpu_free_mib"] for s in samples),
        "minimum_host_available_mib": min(s["ram_available_mib"] for s in samples),
        "maximum_swap_used_mib": max(s["swap_used_mib"] for s in samples),
        "memory_note": "Device-wide sampled readings, including other GPU users; peaks are lower bounds. Host available includes reclaimable cache. Not a long-run guarantee.",
        "warmup_updates_omitted": args.warmup_updates,
        "steady_updates_measured": len(steady),
        "mean_logged_transitions_per_second": statistics.mean(x[0] for x in steady) if steady else None,
        "mean_collection_seconds": statistics.mean(x[1] for x in steady) if steady else None,
        "mean_learning_seconds": statistics.mean(x[2] for x in steady) if steady else None,
    }
    (log / "capacity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
