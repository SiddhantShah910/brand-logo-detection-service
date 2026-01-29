# Model and tracking notes

## Model choice
- Considered YOLOv5s, YOLOv5m, and YOLOv8
- YOLOv5m caught small logos better than v5s without the latency of YOLOv5l/x

## Tracking choice
- ByteTrack keeps IDs stable across frames and reduces flicker in the CSV
