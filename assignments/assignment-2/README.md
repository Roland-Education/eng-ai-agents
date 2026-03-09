# Assignment 2: Image-to-Video Semantic Retrieval via Object Detection

## Files
- assignment2_colab.ipynb — Full pipeline notebook (run on Google Colab with T4 GPU)
- report.md — Written report

## HuggingFace Dataset
Detection results (Parquet): https://huggingface.co/datasets/rolandchi/car-parts-detections

---
language:
- en
license: mit
task_categories:
- object-detection
tags:
- car-parts
- yolov8
- video
- bounding-boxes
size_categories:
- 1K<n<10K
---

# Car Parts Video Detections

Object detection results from a car exterior review video, used for image-to-video semantic retrieval.

## Source

- **Video:** YouTube ID `YcvECxtXoxQ`
- **Detector:** `yolov8n-seg` fine-tuned on the [Ultralytics car-parts-seg dataset](https://docs.ultralytics.com/datasets/segment/carparts-seg/)
- **Sampling rate:** 1 frame per second

## Schema

| Column | Type | Description |
|---|---|---|
| `video_id` | string | YouTube video ID of the source video |
| `frame_index` | int | 1-indexed frame number within the video |
| `timestamp_sec` | int | Elapsed seconds into the video (equals `frame_index` at 1 fps) |
| `class_label` | string | Detected car part label (e.g. `"hood"`, `"wheel"`, `"front_bumper"`) |
| `x_min` | float | Bounding box left edge, in pixels |
| `y_min` | float | Bounding box top edge, in pixels |
| `x_max` | float | Bounding box right edge, in pixels |
| `y_max` | float | Bounding box bottom edge, in pixels |
| `confidence_score` | float | Detection confidence score, range 0.0–1.0 |
| `detector_name` | string | Model identifier string (`yolov8n-seg-carparts`) |

Bounding box coordinates are in absolute pixel space relative to the original video frame resolution (1920 × 1080).

## Stats

| Metric | Value |
|---|---|
| Total detections | 1,064 |
| Frames processed | 359 |
| Frames with ≥1 detection | 300 |
| Distinct class labels | 20 |

### Detected Classes

`back_bumper`, `back_glass`, `back_left_door`, `back_left_light`, `back_right_door`, `back_right_light`, `front_bumper`, `front_glass`, `front_left_door`, `front_left_light`, `front_right_door`, `front_right_light`, `hood`, `left_mirror`, `right_mirror`, `tailgate`, `trunk`, `wheel`

## Usage

```python
import pandas as pd

df = pd.read_parquet("video_detections.parquet")

# Filter to a specific part
hood_frames = df[df["class_label"] == "hood"]

# High-confidence detections only
high_conf = df[df["confidence_score"] >= 0.75]

# All detections for a given frame
frame_5 = df[df["frame_index"] == 5]
```

Or with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("rolandchi/car-parts-detections")
df = ds["train"].to_pandas()
```

## Assignment

CS-GY-6613-INET2 · Intro to Artificial Intelligence · Spring 2026 · Assignment 2
