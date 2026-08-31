"""Run explicit processed world-space pick/place waypoints in headless MuJoCo."""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

from mimic.robot.factory import build_executor, read_waypoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--waypoints",
        type=Path,
        required=True,
        help="JSON of processed world tool poses and object goal; not raw tracking",
    )
    parser.add_argument("--log", type=Path, required=True, help="New JSONL diagnostics file")
    args = parser.parse_args()
    task = read_waypoints(json.loads(args.waypoints.read_text()))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x") as stream:

        def record(event):
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()

        executor = build_executor(args.config, record)
        record(
            {
                "event": "metadata",
                "config": args.config.read_text(),
                "waypoints_sha256": hashlib.sha256(args.waypoints.read_bytes()).hexdigest(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("mink", "mujoco", "qpsolvers", "daqp")
                },
            }
        )
        report = executor.run(task)
    print(json.dumps(asdict(report), indent=2, allow_nan=False))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
