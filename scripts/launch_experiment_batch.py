#!/usr/bin/env python3
"""Assign experiment bundles to execution hosts and build launch commands."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def load_execution_hosts(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assign_trials_to_devices(
    bundles: list[dict[str, Any]],
    hosts_config: dict[str, Any],
) -> list[dict[str, Any]]:
    devices: list[tuple[str, int]] = []
    for host in hosts_config.get("hosts", []):
        for gpu in host.get("gpus", []):
            devices.append((host["name"], gpu))

    if not devices:
        raise ValueError("execution host config must provide at least one GPU")

    assignments: list[dict[str, Any]] = []
    for index, bundle in enumerate(bundles):
        server, gpu = devices[index % len(devices)]
        assignments.append(
            {
                "experiment_id": bundle["experiment_id"],
                "server": server,
                "gpu": gpu,
                "slot_index": index // len(devices),
            }
        )
    return assignments


def build_remote_launch_plan(
    bundle_spec: dict[str, Any],
    host_config: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    workspace_root = host_config["workspace_root"].rstrip("/")
    snapshot_dirname = host_config["snapshot_dirname"].strip("/")
    snapshot_id = bundle_spec.get("snapshot_id", bundle_spec["git_commit"])
    remote_snapshot_root = f"{workspace_root}/{snapshot_dirname}/{snapshot_id}"
    remote_repo_root = f"{remote_snapshot_root}/repo"
    remote_experiment_dir = f"{remote_repo_root}/{bundle_spec['relative_experiment_path']}"
    remote_command_path = f"{remote_experiment_dir}/command.txt"
    stdout_path = f"{remote_experiment_dir}/logs/stdout.log"
    stderr_path = f"{remote_experiment_dir}/logs/stderr.log"
    launch_steps = []
    conda_root = host_config.get("conda_root")
    conda_env = host_config.get("conda_env")
    pythonpath_prepend = host_config.get("pythonpath_prepend")
    if conda_root and conda_env:
        launch_steps.extend(
            [
                "source",
                shlex.quote(f"{conda_root.rstrip('/')}/etc/profile.d/conda.sh"),
                "&&",
                "conda",
                "activate",
                shlex.quote(conda_env),
                "&&",
            ]
        )
    if pythonpath_prepend:
        launch_steps.extend(
            [
                "export",
                f"PYTHONPATH={shlex.quote(pythonpath_prepend)}:$PYTHONPATH",
                "&&",
            ]
        )
    launch_steps.extend(
        [
            "cd",
            shlex.quote(remote_repo_root),
            "&&",
            f"CUDA_VISIBLE_DEVICES={assignment['gpu']}",
            "bash",
            shlex.quote(remote_command_path),
            ">>",
            shlex.quote(stdout_path),
            "2>>",
            shlex.quote(stderr_path),
        ]
    )
    remote_command = " ".join(
        [
            "mkdir",
            "-p",
            shlex.quote(f"{remote_experiment_dir}/logs"),
            "&&",
            "bash",
            "-lc",
            shlex.quote(" ".join(launch_steps)),
        ]
    )

    return {
        "experiment_id": bundle_spec["experiment_id"],
        "server": host_config["name"],
        "ssh_target": host_config["ssh_target"],
        "gpu": assignment["gpu"],
        "slot_index": assignment["slot_index"],
        "remote_snapshot_root": remote_snapshot_root,
        "remote_repo_root": remote_repo_root,
        "remote_experiment_dir": remote_experiment_dir,
        "remote_command": remote_command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign experiment manifests to execution hosts and print launch plans."
    )
    parser.add_argument(
        "--hosts-config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "servers"
        / "execution_hosts.json",
    )
    parser.add_argument(
        "bundle_specs",
        nargs="*",
        type=Path,
        help="JSON files containing bundle specs",
    )
    args = parser.parse_args()

    hosts_config = load_execution_hosts(args.hosts_config)
    bundles = []
    for spec_path in args.bundle_specs:
        with spec_path.open("r", encoding="utf-8") as handle:
            bundles.append(json.load(handle))

    assignments = assign_trials_to_devices(bundles, hosts_config)
    hosts_by_name = {host["name"]: host for host in hosts_config.get("hosts", [])}
    plans = [
        build_remote_launch_plan(bundle, hosts_by_name[assignment["server"]], assignment)
        for bundle, assignment in zip(bundles, assignments)
    ]
    print(json.dumps(plans, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
