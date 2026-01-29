# Training notes

- YOLOv5m was used to balance accuracy with inference speed and model size.
- Training ran on a remote GPU because there is no local GPU available.

Command used:
```bash
python train.py \
  --img 640 \
  --batch 8 \
  --epochs 300 \
  --data dataset.yaml \
  --weights yolov5m.pt \
  --name logo_detection_model
```
