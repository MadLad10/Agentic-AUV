# Localisation study (navigation estimate vs ground truth)

Scripted gate -> red buoy -> surface, no LLM; 20 seeds per row; faults injected at t = 10 s.

"tools ok" = every tool call reported success (what the vehicle believes); "referee: task" = gate passed AND red buoy touched according to simulator ground truth.

| Condition | Estimator | tools ok | referee: task | RMSE xy (m) mean ± sd | max xy (m) mean | worst max xy (m) | RMSE depth (m) |
|---|---|---|---|---|---|---|---|
| nominal | ekf | 20/20 | 20/20 | 0.161 ± 0.073 | 0.206 | 0.381 | 0.007 |
| nominal | ekf+sonar_walls | 20/20 | 20/20 | 0.094 ± 0.034 | 0.183 | 0.296 | 0.008 |
| nominal | ekf+learned_vel+current | 20/20 | 20/20 | 0.161 ± 0.073 | 0.206 | 0.381 | 0.007 |
| nominal | ekf+sonar_walls+learned_vel+current | 20/20 | 20/20 | 0.094 ± 0.034 | 0.183 | 0.296 | 0.008 |
| dvl_dropout | ekf | 20/20 | 11/20 | 0.771 ± 0.425 | 1.782 | 5.388 | 0.008 |
| dvl_dropout | ekf+sonar_walls | 20/20 | 19/20 | 0.238 ± 0.294 | 0.666 | 2.797 | 0.008 |
| dvl_dropout | ekf+learned_vel+current | 20/20 | 20/20 | 0.222 ± 0.116 | 0.477 | 1.554 | 0.019 |
| dvl_dropout | ekf+sonar_walls+learned_vel+current | 20/20 | 20/20 | 0.129 ± 0.091 | 0.299 | 1.651 | 0.025 |
| compass_disturbance | ekf | 20/20 | 20/20 | 0.175 ± 0.074 | 0.232 | 0.383 | 0.007 |
| compass_disturbance | ekf+sonar_walls | 20/20 | 20/20 | 0.099 ± 0.036 | 0.193 | 0.328 | 0.008 |
| compass_disturbance | ekf+learned_vel+current | 20/20 | 20/20 | 0.175 ± 0.074 | 0.232 | 0.383 | 0.007 |
| compass_disturbance | ekf+sonar_walls+learned_vel+current | 20/20 | 20/20 | 0.099 ± 0.036 | 0.193 | 0.328 | 0.008 |
| compass_dvl_long_holds | ekf | 9/20 | 1/20 | 35.864 ± 42.322 | 90.828 | 423.739 | 0.602 |
| compass_dvl_long_holds | ekf+sonar_walls | 14/20 | 12/20 | 3.651 ± 2.761 | 8.233 | 21.742 | 0.005 |
| compass_dvl_long_holds | ekf+learned_vel+current | 20/20 | 0/20 | 2.520 ± 0.351 | 4.024 | 5.716 | 0.012 |
| compass_dvl_long_holds | ekf+sonar_walls+learned_vel+current | 20/20 | 20/20 | 0.445 ± 0.323 | 0.942 | 2.277 | 0.012 |
| dvl_dropout_current | ekf | 20/20 | 8/20 | 0.793 ± 0.279 | 1.903 | 3.355 | 0.008 |
| dvl_dropout_current | ekf+sonar_walls | 20/20 | 20/20 | 0.159 ± 0.093 | 0.487 | 1.204 | 0.008 |
| dvl_dropout_current | ekf+learned_vel+current | 20/20 | 20/20 | 0.646 ± 0.139 | 1.249 | 2.199 | 0.024 |
| dvl_dropout_current | ekf+sonar_walls+learned_vel+current | 20/20 | 20/20 | 0.584 ± 0.152 | 1.152 | 2.084 | 0.023 |
