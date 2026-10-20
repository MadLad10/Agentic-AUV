# Agentic AUV: multimodal multi-agent autonomy for an underwater robot

**v2 of [AUV_LLM_planner](https://github.com/MadLad10/AUV_LLM_planner).** v1 was a single LLM agent
that turned plain-English missions into safe, validated plans for an autonomous underwater vehicle
(AUV). v2 keeps v1's tools, plan schema, deterministic safety validator and 25-mission evaluation,
and adds:

- a **LangGraph multi-agent system** (supervisor, perception, navigation, health & safety,
  diagnostics) with **human-in-the-loop** plan approval and **checkpointing**,
- a new **MuJoCo marine simulator** with noisy, failing sensors (camera with turbidity,
  forward-looking sonar, DVL, IMU, compass, depth, battery, thruster telemetry, leak sensor) and
  a **fault injector**,
- **EKF navigation** (visual-inertial-acoustic sensor fusion) scored against ground truth,
- a **YOLO detector trained from scratch on synthetic, simulator-auto-labelled images**,
- a client for on-demand **vision-language model (VLM)** scene queries (wired in and tested with a
  mock; **no real VLM has been evaluated yet**: needs a Groq key or a local model).

> **Status: work in progress.** Milestones M1–M3 and M6 are done and tested; M4 (YOLO) is
> training; MCP, observability, the v1-vs-v2 evaluation and media are next. See [PLAN.md](PLAN.md).
> Every number below comes from a test or a logged run in this repository.

## How it works

```mermaid
flowchart LR
  OP[Operator] -->|mission text, approval| SUP[Supervisor LLM]
  SUP --> VAL{{Safety validator<br/>rules, not AI}}
  VAL -->|approved plan| EXE[Executor]
  EXE -->|v1 typed tools| VEH[Onboard software<br/>EKF, guidance, PID]
  VEH --> SIM[(MuJoCo sim)]
  SIM --> PER[Perception agent<br/>camera + sonar + VLM]
  SIM --> NAV[Navigation agent<br/>EKF health]
  SIM --> HLT{{Health & safety<br/>veto}}
  HLT -->|fault| DIA[Diagnostics LLM]
  DIA --> SUP
  PER --> SUP
  NAV --> SUP
```

- The **LLMs never touch the thrusters.** They plan and explain; every motion goes through v1's
  typed tools and the deterministic safety validator. Only the onboard PID controller produces
  thruster commands.
- The **health & safety agent is deterministic and has veto power**: a leak, a frozen depth
  sensor, a battery below the floor or a lost depth axis makes the vehicle surface immediately,
  whatever the plan or the LLM says. The diagnostics LLM cannot downgrade a veto (tested).
- A **referee** that only sees simulator ground truth scores every run; the agents' own claims
  are never trusted.

## What already works (tested)

| Camera blackout: diagnosed, replanned, gate found by **sonar alone** | DVL dropout: EKF falls back to IMU dead reckoning + visual landmarks |
|---|---|
| ![blackout](docs/media/v2_blackout_sonar_gate.gif) | ![dvl dropout](docs/media/v2_dvl_dropout.gif) |

*Each panel: third-person view, onboard camera (with detector boxes), forward sonar, top-down map
(white = true path, blue = EKF estimate), live health status. Left: the full LangGraph agent loop
with the LLM's plans replayed from a file for reproducibility (agents, safety checks, sim and referee
all live). Right: a scripted tool sequence (the same tools the agents call).*

| Capability | Evidence |
|---|---|
| Gate → red buoy → surface on **EKF navigation** | gate passed, buoy touched, surfaced, 0 collisions in every seed; localisation error about 0.1 m (20-seed study in progress: `eval/results/nav_study_20seeds/`) |
| DVL dropout → IMU dead reckoning | drift is highly variable (0.4–1.7 m RMSE across seeds), so a 20-seed study is running; in a 5-seed pilot a **learned velocity model** (GRU on IMU + thrust history) cut the error from ~1 m to ~0.15 m. The referee also caught runs where the tools reported success but the buoy was missed |
| Gyro bias / compass disturbance | absorbed by the EKF (bias estimated, bad compass readings gated out) |
| Leak / frozen depth sensor | health veto in ≤ 3 s, emergency surface confirmed by the referee |
| Dead thruster | detected from thruster current; the allocator re-distributes thrust; mission still completes |
| Camera blackout | diagnosed → replanned → gate found by sonar alone → gate passed |
| Turbidity (2.5 m visibility) | camera–sonar fusion tracker + sonar-cued search: task **5/5 vs 0/5** without it, 70 s vs 417 s ([perception study](eval/results/perception_study/summary.md)); the VLM query in this path was tested with a **mock** VLM only |
| YOLO from scratch on synthetic data | mAP@0.5 0.944 on held-out synthetic scenes; one closed-loop gate → buoy → surface run with YOLO instead of the simulated detector |
| Human-in-the-loop | LangGraph `interrupt` before execution; resume from a **SQLite checkpoint**; dashboard Approve/Reject |
| MCP tool server | unsafe calls (e.g. 25 m depth) refused by the validator over the MCP protocol; no low-level tool exists |
| Observability | OpenTelemetry spans (mission → agent node → LLM call / tool call) in a self-hosted Jaeger |
| v1 regression | all 130 v1 unit tests still pass |

## Hardware and software

One laptop: Intel i7-13620H, 15 GB RAM shared with other work, Intel integrated graphics (no GPU
compute). Ubuntu 22.04, MuJoCo 3.16 (headless EGL rendering), LangGraph 1.2,
Ultralytics YOLOv8 (CPU), local LLMs through Ollama, Groq (optional, OpenAI-compatible API).

**Why MuJoCo?** We tried Stonefish, Gazebo Harmonic + DAVE and MuJoCo on this machine (details in
[PLAN.md §1.1](PLAN.md)). MuJoCo was the only one that ran natively next to ROS 2 Humble with
GPU rendering on the integrated graphics (camera at ~110 fps, physics 600× real time, 140 MB RAM).
Its limits: we wrote the buoyancy, drag and sensor models ourselves, and the sonar is ray-cast.

## Run it

```bash
pip install --user uv && scripts/setup_venv.sh          # Python env (no sudo)
source /opt/ros/humble/setup.bash && colcon build --symlink-install
source scripts/env.sh
python -m pytest -m "not integration and not llm"        # tests
python scripts/scripted_mission.py --nav ekf --fault dvl_dropout:10      # no LLM
python -m auv_agents.run "pass through the gate then surface" --provider ollama
python -m auv_agents.run "pass through the gate then surface" --approval interrupt   # you approve
```

## Limitations (so far)

- **ROS 2:** v1's ROS 2 packages are kept and still build, but **v2's agents, sim and onboard stack
  run in one Python process without ROS 2** (a ROS 2 bridge is not built yet).
- **Most results use a simulated camera detector,** not image-based detection: a camera model that
  uses simulator geometry plus noise, field of view, occlusion, water visibility and dropouts. The
  YOLO detector is trained and tested, but the 72-run evaluation and the studies did not use it.
- **Navigation knows the gate's position** from a course map (with 15 cm error), as RoboSub teams do;
  the sonar-only gate pass also takes the gate's depth from that map (a 2D sonar has no elevation).
- **All data is synthetic** (generated by our simulator); nothing is validated on real water.
- **The v1-vs-v2 evaluation ran each mission once,** on our MuJoCo sim (not v1's original Gazebo world).
- **Health thresholds were calibrated on the same simulator** they are tested in.

- Simulation only; hand-written hydrodynamics (linear + quadratic damping, no added-mass coupling).
- The sonar is a ray-cast fan with beam-width-aware object returns, not an acoustic model.
- Actions run faster than real time (lockstep, capped). Between actions the sim keeps running in
  real time while the agents think, so LLM latency costs battery as it would on a real vehicle;
  unit tests switch that idle clock off for speed.
- The YOLO detector is trained only on synthetic images.

**Keywords:** agentic AI · multi-agent systems · LangGraph · human-in-the-loop · checkpointing ·
vision-language models (VLM) · sensor fusion · extended Kalman filter (EKF) · synthetic data ·
YOLO · fault injection · AI safety guardrails · LLM evaluation · MuJoCo · marine robotics
