# Assignment 3 - UAV Drone Detection and Tracking

## Output Videos

[![Drone Tracking Video 1](https://img.youtube.com/vi/17G8AW9ev8A/0.jpg)](https://youtu.be/17G8AW9ev8A)

[![Drone Tracking Video 2](https://img.youtube.com/vi/3A2ncN0b2ss/0.jpg)](https://youtu.be/3A2ncN0b2ss)

## Dataset and Detector

I used YOLOv8n pretrained on COCO. COCO doesn't have a drone class so I filtered detections to anything matching `drone`, `uav`, or `airplane` at a confidence threshold of 0.3.

Both videos were split into frames at 5 fps - 828 frames from video 1, 2,580 from video 2. Video 1 had 266 frames with detections. Video 2 only had 7, which makes sense given how small and distant the drone is in that footage. The base COCO model wasn't trained on drone-specific data so it struggles there - fine-tuning on a Roboflow drone dataset would fix this.

Detection frames are uploaded here: [rolandchi/introai-drone-detections-assignment3](https://huggingface.co/datasets/rolandchi/introai-drone-detections-assignment3)

## Kalman Filter Design

State vector is `[cx, cy, vx, vy]` - bounding box center plus velocity. Constant velocity motion model. The filter only observes position `[cx, cy]` and infers velocity over time.

| Matrix | Value | Description |
|--------|-------|-------------|
| F (transition) | constant velocity | predicts next position from current position + velocity |
| H (measurement) | observes cx, cy only | velocity is not directly observed |
| R (measurement noise) | 10 * I | accounts for detector jitter |
| Q (process noise) | 0.1 * I | assumes smooth drone motion |
| P (initial covariance) | 500 * I | high initial uncertainty before first update |

Each frame: predict first, then update if a detection is present.

## Missed Detections and Failure Cases

The tracker keeps predicting for up to 5 consecutive missed frames before dropping the track and waiting to reinitialize. This handles short occlusions or frames where the detector just misses.

The main failure case is video 2 - 7 detections out of 2,580 frames means the tracker barely gets to run. The drone is small and far away and the base model wasn't built for that. The other edge case is that the pipeline only handles one drone per frame (highest confidence box wins), so it wouldn't generalize to multi-drone scenarios without adding proper assignment logic.
