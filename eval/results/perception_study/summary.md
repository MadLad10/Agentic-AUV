# Perception A/B: camera-sonar fusion tracking OFF vs ON (v2 agent graph, replayed plans)

5 seeds per arm; referee-scored. "task" = mission objective met per ground truth.

| Condition | Tracking | task | surfaced | wrong-buoy touches | false target flagged | inspections (mean) | sim s (mean) |
|---|---|---|---|---|---|---|---|
| turbidity_touch_red | off | 0/5 | 5/5 | 0 | 0/5 | 0.0 | 417 |
| turbidity_touch_red | ON | 5/5 | 5/5 | 0 | 0/5 | 2.4 | 70 |
| false_target_gate | off | 5/5 | 5/5 | 0 | 0/5 | 0.0 | 40 |
| false_target_gate | ON | 5/5 | 5/5 | 0 | 5/5 | 0.0 | 40 |
| nominal_gate_red_green | off | 5/5 | 5/5 | 0 | 5/5 | 0.0 | 74 |
| nominal_gate_red_green | ON | 5/5 | 5/5 | 0 | 0/5 | 0.0 | 74 |
