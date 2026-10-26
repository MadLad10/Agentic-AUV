#!/usr/bin/env python3
"""Exercise the existing application against an isolated standalone simulator wheel.

Source scripts/env.sh first. --site is a separate target installation, not the
application's normal environment. No dependencies or source files are changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib import metadata, util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "core": [
        "src/auv_marine_sim/test/test_marine_sim.py",
        "src/auv_onboard/test/test_onboard.py",
        "src/auv_onboard/test/test_estimators.py",
        "src/auv_onboard/test/test_sonar_mapmatch.py",
        "src/auv_onboard/test/test_tracking.py",
        "src/auv_agents/test/test_graph.py",
    ],
    "vision": ["src/auv_onboard/test/test_yolo_detector.py"],
    "ros": ["tests/integration/test_ros2_vehicle.py"],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, default="core")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    site = args.site.resolve()
    wheel = args.wheel.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be an empty directory")
    if not site.is_dir() or not wheel.is_file():
        parser.error("--site must be an installed target directory and --wheel a local wheel")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir()
    os.chdir(ROOT)
    # Prepend the target for both this process and subprocesses, including ROS.
    if any(name == "auv_marine_sim" or name.startswith("auv_marine_sim.") for name in sys.modules):
        raise RuntimeError("Simulator was imported before target isolation")
    sys.path.insert(0, str(site))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(site), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    )
    # The local Torch build must load before a MuJoCo EGL context is created.
    if util.find_spec("torch") is not None:
        import torch  # noqa: F401
    from auv_marine_sim.sim import MarineSim
    from auv_marine_sim.world import arena_path, default_config_path

    paths = {
        "simulator": Path(inspect.getfile(MarineSim)).resolve(),
        "config": Path(default_config_path()).resolve(),
        "default_arena": Path(arena_path("default")).resolve(),
        "alternative_arena": Path(arena_path("buoy_hidden")).resolve(),
    }
    if not all(path.is_relative_to(site) for path in paths.values()):
        raise RuntimeError(f"A simulator import or resource escaped the isolated target: {paths}")
    with zipfile.ZipFile(wheel) as archive:
        payload = {name: archive.read(name) for name in archive.namelist()
                   if name.startswith("auv_marine_sim/") and not name.endswith("/")}
    if not payload or any((site / name).read_bytes() != data for name, data in payload.items()):
        raise RuntimeError("Installed simulator payload does not match the supplied wheel")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    provenance = {
        "started_at": datetime.now(ZoneInfo("Asia/Dhaka")).isoformat(),
        "scope": "Standalone simulator compatibility check; normal application environment unchanged",
        "suite": args.suite,
        "python": sys.executable,
        "application_revision": revision,
        "target_site": str(site),
        "wheel": {"path": str(wheel), "sha256": digest(wheel)},
        "installed_payload_files_verified": len(payload),
        "import_and_resource_paths": {name: str(path) for name, path in paths.items()},
        "resource_sha256": {name: digest(path) for name, path in paths.items()},
        "versions": {name: metadata.version(name) for name in
                     ["mujoco-marine-sim", "mujoco", "numpy", "pytest", "langgraph", "gtsam"]},
        "render_backend": os.environ.get("MUJOCO_GL"),
        "test_sources": {name: digest(ROOT / name) for name in SUITES[args.suite]},
        "application_runtime_modified": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"suite": args.suite, "verified_simulator": str(paths["simulator"]),
                      "verified_config": str(paths["config"])}), flush=True)
    import pytest
    start = time.monotonic()
    code = int(pytest.main(["-v", "--tb=short", "--color=no",
                           f"--junitxml={output / 'junit.xml'}",
                           f"--basetemp={output / 'raw/pytest_tmp'}", *SUITES[args.suite]]))
    loaded = {name: str(Path(module.__file__).resolve())
              for name, module in sys.modules.items()
              if (name == "auv_marine_sim" or name.startswith("auv_marine_sim."))
              and getattr(module, "__file__", None)}
    imports_valid = all(Path(path).is_relative_to(site) for path in loaded.values())
    cases = []
    if (output / "junit.xml").is_file():
        for case in ET.parse(output / "junit.xml").getroot().iter("testcase"):
            issue = next((child for child in case if child.tag in {"failure", "error", "skipped"}), None)
            cases.append({"name": case.get("name"), "class": case.get("classname"),
                          "seconds": float(case.get("time", "0")),
                          "status": issue.tag if issue is not None else "passed",
                          **({"message": issue.get("message", "")} if issue is not None else {})})
    counts = {key: sum(case["status"] == key for case in cases)
              for key in ["passed", "failure", "error", "skipped"]}
    # Skipped required checks do not establish compatibility for this suite.
    verified = code == 0 and imports_valid and bool(cases) and not counts["skipped"]
    result = {"suite": args.suite, "verified": verified, "pytest_exit_code": code,
              "wall_seconds": round(time.monotonic() - start, 3), "counts": counts,
              "simulator_imports_remained_isolated": imports_valid,
              "loaded_simulator_modules": loaded, "cases": cases}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["suite", "verified", "pytest_exit_code", "counts"]}), flush=True)
    return 0 if verified else (code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
