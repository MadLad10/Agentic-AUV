# Standalone simulator integration checks — Step 2

Date: 2026-10-27, Asia/Dhaka.

**The standalone simulator passes the application's compatibility checks: 54 tests passed, with no failures, errors or skips in the final suites.** The simulator wheel was installed into an isolated target directory and loaded ahead of the application's existing ROS overlay. The application's normal dependency setup and simulator sources were preserved.

| Final suite | Result | Scope | Receipt |
|---|---|---|---|
| Core | 52 passed | Dynamics, sensor payloads, all 11 fault types, camera/sonar rendering, referee, EKF/factor-graph missions, learned velocity, map matching, tracking and mission-agent behavior | [Result](core_verified/result.json), [JUnit](core_verified/junit.xml) |
| Vision | 1 passed | Existing YOLO weights detect the gate in a rendered camera image with plausible range and bearing | [Result](vision/result.json), [JUnit](vision/junit.xml) |
| ROS 2 | 1 passed | A separate vehicle-node process accepts high-level navigation/surface actions; the referee confirms gate passage, surfacing and zero collisions | [Result](ros/result.json), [JUnit](ros/junit.xml) |

[Machine-readable summary](summary.json). [Rendering/sensor preflight](preflight.json). [Normal application environment after testing](normal_environment.json).

## What was checked

The runner verifies that `MarineSim`, `sim.yaml` and both packaged arenas resolve inside the isolated installation. It compares every installed simulator payload file against the supplied wheel before running tests, then checks the paths of all loaded simulator modules again after pytest. The wheel's runtime/configuration payload also matches the current standalone source files.

The tests ran with the application's NumPy 1.26.4, MuJoCo 3.16.0 and pytest 7.4.4, using EGL rendering. Per-suite provenance records the wheel SHA-256, application revision, dependency versions, test-file hashes and import/resource paths:

- [Core provenance](core_verified/provenance.json).
- [Vision provenance](vision/provenance.json).
- [ROS provenance](ros/provenance.json).

An additional [negative check](negative_guard.json) supplies an empty target directory while leaving the original ROS-overlay simulator available. The runner rejects that import before starting pytest. This guards against accidentally reporting the original simulator's results as standalone integration results.

The mission checks include gate/buoy/surface execution, DVL-loss diagnosis, blackout-driven replanning using sonar, turbidity-driven inspection, leak-triggered safe surfacing, operator approval/rejection and planned abort completion. Agent plans use the existing scripted `MockProvider`; no live model service is called. The core mission tests use the configured ground-truth camera-detection adapter. The vision suite separately exercises YOLO on rendered pixels.

## Initial runner error and correction

The [first core attempt](core/result.json) passed 50 tests; two tests could not start because the parent directory for pytest's temporary files had not been created. This was a test-runner setup error. Creating that parent directory fixed the harness; the complete [second core run](core_verified/result.json) passed all 52 tests. The original result and JUnit report remain available, and no simulator, mission algorithm or test assertion was changed.

## Reproduce

The application workspace must already be built with its normal dependencies and model artifacts. Build the standalone wheel in its repository if `dist/` is absent. The wheel is installed with `--no-deps` into a temporary directory so the application keeps its existing dependency versions.

From the Agentic AUV workspace:

```bash
source scripts/env.sh
sim_check_dir="$(mktemp -d /tmp/auv-standalone-check.XXXXXX)"
sim_check_wheel="$(realpath ../mujoco_marine_sim/dist/mujoco_marine_sim-0.1.0-py3-none-any.whl)"

UV_CACHE_DIR="$sim_check_dir/cache" "$HOME/.local/bin/uv" pip install \
  --python .venv/bin/python --no-index --no-deps \
  --target "$sim_check_dir/site-packages" "$sim_check_wheel"

sim_check_status=0
for suite in core vision ros; do
  ROS_LOG_DIR="$sim_check_dir/ros-logs" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/check_standalone_sim.py \
    --site "$sim_check_dir/site-packages" \
    --wheel "$sim_check_wheel" \
    --suite "$suite" --output "$sim_check_dir/results/$suite" || { sim_check_status=$?; break; }
done
test "$sim_check_status" -eq 0
```

Each suite produces `provenance.json`, `result.json` and `junit.xml`. Use a fresh output directory for each invocation. Skipped checks cause the runner to return an unsuccessful verification status. Detailed logs and pytest/ROS temporary files from this recorded run remain local under the suites' ignored `raw/` directories.

## Migration status

- Step 2, compatibility verification against the standalone package: **passed**.
- Step 1, changing the normal application installation to depend on that package: **pending**.
- Retiring the application's duplicate simulator implementation: **pending**, after the normal dependency setup is migrated and checked.

The normal application still resolves `auv_marine_sim` through its existing build overlay; the standalone distribution was installed only into the test target. These results establish compatibility for the recorded conditions and provide a baseline for the dependency migration.
