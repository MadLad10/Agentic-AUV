# Perception A/B: camera-sonar fusion tracking OFF vs ON (v2 agent graph, replayed plans)

10 seeds per arm; referee-scored. "task" = mission objective met per ground truth.

| Condition | Tracking | task | surfaced | runs with collisions | wrong-buoy touches | false target flagged | inspections (mean) | sim s (mean) |
|---|---|---|---|---|---|---|---|---|
| turbidity_touch_red | off | 0/10 | 10/10 | 0/10 | 0 | 0/10 | 0.0 | 417 |
| turbidity_touch_red | ON | 10/10 | 10/10 | 0/10 | 0 | 0/10 | 2.8 | 78 |
| false_target_gate | off | 10/10 | 10/10 | 0/10 | 0 | 0/10 | 0.0 | 40 |
| false_target_gate | ON | 10/10 | 10/10 | 0/10 | 0 | 10/10 | 0.0 | 43 |
| nominal_gate_red_green | off | 10/10 | 10/10 | 0/10 | 0 | 10/10 | 0.0 | 74 |
| nominal_gate_red_green | ON | 10/10 | 10/10 | 0/10 | 0 | 0/10 | 0.0 | 74 |
