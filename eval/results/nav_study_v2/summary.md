# Localisation study (navigation estimate vs ground truth)

Scripted gate -> red buoy -> surface, no LLM; 5 seeds per row; faults injected at t = 10 s.

"tools ok" = every tool call reported success (what the vehicle believes); "referee: task" = gate passed AND red buoy touched according to simulator ground truth.

| Condition | Estimator | tools ok | referee: task | RMSE xy (m) mean ± sd | max xy (m) mean | worst max xy (m) | RMSE depth (m) |
|---|---|---|---|---|---|---|---|
| nominal | ekf | 5/5 | 5/5 | 0.127 ± 0.059 | 0.172 | 0.268 | 0.008 |
| nominal | fgo | 5/5 | 5/5 | 0.134 ± 0.045 | 0.241 | 0.320 | 0.019 |
| nominal | ekf+learned_vel | 5/5 | 5/5 | 0.127 ± 0.059 | 0.172 | 0.268 | 0.008 |
| nominal | fgo+learned_vel | 5/5 | 5/5 | 0.134 ± 0.045 | 0.241 | 0.320 | 0.019 |
| dvl_dropout | ekf | 5/5 | 0/5 | 0.986 ± 0.191 | 2.363 | 3.375 | 0.008 |
| dvl_dropout | fgo | 5/5 | 0/5 | 2.590 ± 1.732 | 6.037 | 11.442 | 0.021 |
| dvl_dropout | ekf+learned_vel | 5/5 | 5/5 | 0.145 ± 0.039 | 0.254 | 0.346 | 0.020 |
| dvl_dropout | fgo+learned_vel | 5/5 | 3/5 | 0.944 ± 0.349 | 1.733 | 3.166 | 0.022 |
| dvl_dropout_no_landmarks | ekf | 5/5 | 2/5 | 1.184 ± 0.492 | 2.522 | 3.845 | 0.008 |
| dvl_dropout_no_landmarks | fgo | 3/5 | 1/5 | 8.127 ± 11.891 | 15.205 | 55.930 | 0.017 |
| dvl_dropout_no_landmarks | ekf+learned_vel | 5/5 | 5/5 | 0.182 ± 0.155 | 0.470 | 1.578 | 0.017 |
| dvl_dropout_no_landmarks | fgo+learned_vel | 5/5 | 5/5 | 0.662 ± 0.198 | 1.044 | 1.489 | 0.022 |
| imu_bias | ekf | 5/5 | 5/5 | 0.117 ± 0.057 | 0.170 | 0.264 | 0.007 |
| imu_bias | fgo | 5/5 | 5/5 | 0.146 ± 0.053 | 0.246 | 0.324 | 0.019 |
| imu_bias | ekf+learned_vel | 5/5 | 5/5 | 0.117 ± 0.057 | 0.170 | 0.264 | 0.007 |
| imu_bias | fgo+learned_vel | 5/5 | 5/5 | 0.146 ± 0.053 | 0.246 | 0.324 | 0.019 |
| compass_disturbance | ekf | 5/5 | 5/5 | 0.150 ± 0.072 | 0.215 | 0.348 | 0.008 |
| compass_disturbance | fgo | 5/5 | 5/5 | 1.815 ± 0.043 | 3.163 | 3.292 | 0.019 |
| compass_disturbance | ekf+learned_vel | 5/5 | 5/5 | 0.150 ± 0.072 | 0.215 | 0.348 | 0.008 |
| compass_disturbance | fgo+learned_vel | 5/5 | 5/5 | 1.815 ± 0.043 | 3.163 | 3.292 | 0.019 |
