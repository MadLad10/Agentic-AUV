# Detector study (all trained from scratch on synthetic, auto-labelled frames; CPU)

| Model | Params (M) | Train data | mAP@0.5 | mAP@0.5:0.95 | CPU ms/img | recall @ 2 m | recall @ 3 m | recall @ 4 m | recall @ 6 m | recall @ 9 m | recall @ 12 m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n_base | 3.01 | synthetic | 0.944 | 0.818 | 11.57 | 0.09 | 0.20 | 0.32 | 0.55 | 0.83 | 0.88 |
| yolo11n_base | 2.59 | synthetic | 0.924 | 0.795 | 18.96 | 0.09 | 0.21 | 0.32 | 0.56 | 0.83 | 0.87 |
| yolo12n_base | 2.57 | synthetic | 0.928 | 0.793 | 16.84 | 0.08 | 0.19 | 0.31 | 0.56 | 0.80 | 0.89 |
| yolov8n_turbid | 3.01 | synthetic_turbid | 0.940 | 0.803 | 12.69 | 0.09 | 0.20 | 0.32 | 0.59 | 0.85 | 0.87 |

Recall @ visibility: same scenes and poses at each water visibility, labelled with the clear-water labels (objects visible at 12 m), so lower visibility hides far objects for every model alike.
