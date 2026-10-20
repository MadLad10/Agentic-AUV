# VLM benchmark: moondream (local, CPU) vs YOLOv8n on the same simulator frames

Frames: {'clear': 40, 'murky': 10, 'black': 10}. Open questions scored by keyword matching (moondream returns empty answers to yes/no questions); YOLO = our detector trained from scratch (conf 0.35). Raw answers in results.json.

| Object | VLM precision | VLM recall | YOLO precision | YOLO recall | support |
|---|---|---|---|---|---|
| gate | 0.333 | 0.545 | 0.9 | 0.818 | 11 |
| red_buoy | 1.0 | 0.667 | 1.0 | 1.0 | 6 |
| green_buoy | 0.467 | 1.0 | 0.75 | 0.857 | 7 |
| yellow_buoy | 0.4 | 0.222 | 0.889 | 0.889 | 9 |

Blackout detection ("black"/"dark" in the description): {'black_frames_called_dark': 1, 'normal_frames_called_dark': 0.5}
Empty VLM answers: 33 of 180 questions
Latency per VLM call: median 2.19 s, p90 15.04 s (180 calls, CPU, machine shared with other jobs)
