# Localisation study (navigation estimate vs ground truth)

Scripted gate -> red buoy -> surface, no LLM; 20 seeds per row; faults injected at t = 10 s.

"tools ok" = every tool call reported success (what the vehicle believes); "referee: task" = gate passed AND red buoy touched according to simulator ground truth.

| Condition | Estimator | tools ok | referee: task | RMSE xy (m) mean ± sd | max xy (m) mean | worst max xy (m) | RMSE depth (m) |
|---|---|---|---|---|---|---|---|
| nominal | ekf | 20/20 | 20/20 | 0.160 ± 0.073 | 0.205 | 0.381 | 0.007 |
| nominal | ekf+learned_vel | 20/20 | 20/20 | 0.160 ± 0.073 | 0.205 | 0.381 | 0.007 |
| nominal | fgo | 20/20 | 20/20 | 0.191 ± 0.068 | 0.295 | 0.433 | 0.019 |
| nominal | fgo+learned_vel | 20/20 | 20/20 | 0.191 ± 0.068 | 0.295 | 0.433 | 0.019 |
| dvl_dropout | ekf | 20/20 | 10/20 | 0.768 ± 0.427 | 1.770 | 5.388 | 0.008 |
| dvl_dropout | ekf+learned_vel | 20/20 | 20/20 | 0.189 ± 0.057 | 0.289 | 0.472 | 0.025 |
| dvl_dropout | fgo | 19/20 | 0/20 | 4.541 ± 11.752 | 9.502 | 108.736 | 0.020 |
| dvl_dropout | fgo+learned_vel | 20/20 | 15/20 | 1.057 ± 0.300 | 1.758 | 3.166 | 0.022 |
| dvl_dropout_no_landmarks | ekf | 20/20 | 4/20 | 1.323 ± 0.571 | 2.900 | 5.547 | 0.008 |
| dvl_dropout_no_landmarks | ekf+learned_vel | 20/20 | 20/20 | 0.223 ± 0.196 | 0.606 | 2.051 | 0.016 |
| dvl_dropout_no_landmarks | fgo | 14/20 | 1/20 | 10.841 ± 15.487 | 19.775 | 96.811 | 0.018 |
| dvl_dropout_no_landmarks | fgo+learned_vel | 20/20 | 20/20 | 0.649 ± 0.168 | 1.129 | 2.273 | 0.022 |
| compass_disturbance | ekf | 20/20 | 20/20 | 0.173 ± 0.074 | 0.229 | 0.383 | 0.007 |
| compass_disturbance | ekf+learned_vel | 20/20 | 20/20 | 0.173 ± 0.074 | 0.229 | 0.383 | 0.007 |
| compass_disturbance | fgo | 20/20 | 20/20 | 1.826 ± 0.069 | 3.173 | 3.489 | 0.019 |
| compass_disturbance | fgo+learned_vel | 20/20 | 20/20 | 1.826 ± 0.069 | 3.173 | 3.489 | 0.019 |
