# Metrics and outputs

## What we track
- Detection confidence
- Spot-check precision (false positives)
- Processing FPS

## CSV format
One row per completed track (first frame of the track):
`frame_id`, `timestamp`, `brand`, `confidence`, `x1`, `y1`, `x2`, `y2`.

