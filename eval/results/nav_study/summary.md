# Localisation study (EKF vs ground truth)

Scripted gate -> red buoy -> surface, no LLM; 5 seeds per condition; faults injected at t = 10 s.

"tools ok" = every tool call reported success (what the vehicle believes); "referee: task" = gate passed AND red buoy touched according to simulator ground truth.

| Condition | tools ok | referee: task | RMSE xy (m) mean ± sd | max xy (m) mean | worst max xy (m) | RMSE depth (m) |
|---|---|---|---|---|---|---|
| nominal | 5/5 | 5/5 | 0.122 ± 0.063 | 0.165 | 0.290 | 0.007 |
| dvl_dropout | 5/5 | 5/5 | 0.439 ± 0.148 | 0.953 | 1.223 | 0.008 |
| imu_bias | 5/5 | 5/5 | 0.117 ± 0.062 | 0.166 | 0.277 | 0.007 |
| compass_disturbance | 5/5 | 5/5 | 0.145 ± 0.069 | 0.216 | 0.358 | 0.007 |
| dvl_dropout_no_landmarks | 5/5 | 1/5 | 1.695 ± 0.395 | 3.947 | 4.758 | 0.008 |
