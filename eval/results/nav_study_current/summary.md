# Localisation study (navigation estimate vs ground truth)

Scripted gate -> red buoy -> surface, no LLM; 20 seeds per row; faults injected at t = 10 s.

"tools ok" = every tool call reported success (what the vehicle believes); "referee: task" = gate passed AND red buoy touched according to simulator ground truth.

| Condition | Estimator | tools ok | referee: task | RMSE xy (m) mean ± sd | max xy (m) mean | worst max xy (m) | RMSE depth (m) |
|---|---|---|---|---|---|---|---|
| dvl_dropout | ekf | 20/20 | 11/20 | 0.771 ± 0.425 | 1.782 | 5.388 | 0.008 |
| dvl_dropout | ekf+learned_vel | 20/20 | 20/20 | 0.207 ± 0.100 | 0.377 | 1.860 | 0.026 |
| dvl_dropout | ekf+learned_vel+current | 20/20 | 20/20 | 0.222 ± 0.116 | 0.477 | 1.554 | 0.019 |
| dvl_dropout_current | ekf | 20/20 | 8/20 | 0.793 ± 0.279 | 1.903 | 3.355 | 0.008 |
| dvl_dropout_current | ekf+learned_vel | 20/20 | 0/20 | 1.635 ± 0.238 | 3.229 | 4.966 | 0.023 |
| dvl_dropout_current | ekf+learned_vel+current | 20/20 | 20/20 | 0.646 ± 0.139 | 1.249 | 2.199 | 0.024 |
