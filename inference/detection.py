import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import cv2
import pandas as pd
import numpy as np
import yolov5
import supervision as sv

# Allow loading Windows-trained weights on Unix
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

# Load model and set confidence threshold
model = yolov5.load('../models/logo_detection_model.pt')
model.conf = 0.1  # Confidence threshold

# Video I/O
input_path = "../data/input_video.mp4"
output_path = "../outputs/labeled_output.mp4"
FRAME_SKIP = 5
cap = cv2.VideoCapture(input_path)

# Video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ByteTrack tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.1,
    lost_track_buffer=30,           
    minimum_matching_threshold=0.8,
    frame_rate=fps
)

# Annotators for visualization
box_annotator = sv.BoxAnnotator(
    thickness=2,
    color=sv.ColorPalette.from_hex(['#00FF00'])
)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_padding=5,
    color=sv.ColorPalette.from_hex(['#00FF00'])
)

# Get class names from model
class_names = model.names

# Track logger (only logs when a track ends)
class TrackLogger:
    def __init__(self, fps, lost_threshold=30):
        self.active_tracks = {}  
        self.completed_tracks = []
        self.fps = fps
        self.lost_threshold = lost_threshold
    
    def update(self, tracked_detections, class_names, current_frame):
        """Update tracks with current frame's detections"""
        current_track_ids = set()
        
        for i in range(len(tracked_detections)):
            track_id = int(tracked_detections.tracker_id[i])
            class_id = tracked_detections.class_id[i]
            conf = float(tracked_detections.confidence[i])
            bbox = tracked_detections.xyxy[i]
            brand = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            current_track_ids.add(track_id)
            
            if track_id not in self.active_tracks:
                # New track: store first appearance data
                self.active_tracks[track_id] = {
                    'frame_id': current_frame,
                    'brand': brand,
                    'confidence': conf,
                    'bbox': bbox,  # First bounding box
                    'last_frame': current_frame
                }
            else:
                # Update last seen frame
                self.active_tracks[track_id]['last_frame'] = current_frame
        
        # Close tracks that have been lost long enough
        ended_tracks = []
        for track_id, track_data in self.active_tracks.items():
            if track_id not in current_track_ids:
                if current_frame - track_data['last_frame'] >= self.lost_threshold:
                    ended_tracks.append(track_id)
        
        for track_id in ended_tracks:
            self._finalize_track(track_id)
    
    def _finalize_track(self, track_id):
        """Move track to completed list with simplified fields"""
        track_data = self.active_tracks.pop(track_id)
        bbox = track_data['bbox']
        
        self.completed_tracks.append({
            'frame_id': track_data['frame_id'],
            'timestamp': round(track_data['frame_id'] / self.fps, 3),
            'brand': track_data['brand'],
            'confidence': round(track_data['confidence'], 3),
            'x1': round(float(bbox[0]), 1),
            'y1': round(float(bbox[1]), 1),
            'x2': round(float(bbox[2]), 1),
            'y2': round(float(bbox[3]), 1)
        })
    
    def finalize_all(self):
        """Finalize all remaining active tracks"""
        for track_id in list(self.active_tracks.keys()):
            self._finalize_track(track_id)
    
    def get_results(self):
        """Get all completed tracks as DataFrame"""
        return pd.DataFrame(self.completed_tracks)

# Initialize track logger
track_logger = TrackLogger(fps=fps, lost_threshold=30)

# Main processing loop
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Cache for last detection results (used for skipped frames)
last_tracked_detections = sv.Detections.empty()
last_labels = []

print("Press 'q' to stop.\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only run inference on every Nth frame
        if frame_count % FRAME_SKIP == 0:
            # Run inference
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, size=640)
            detections_df = results.pandas().xyxy[0]
            
            # Convert to supervision Detections format
            if len(detections_df) > 0:
                xyxy = detections_df[['xmin', 'ymin', 'xmax', 'ymax']].values
                confidence = detections_df['confidence'].values
                class_id = detections_df['class'].values.astype(int)
                
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id
                )
            else:
                detections = sv.Detections.empty()
            
            # Update tracker
            tracked_detections = tracker.update_with_detections(detections)
            
            # Update track logger (logs when tracks end)
            track_logger.update(tracked_detections, class_names, frame_count)
            
            # Create labels from class names
            labels = []
            for i in range(len(tracked_detections)):
                track_id = tracked_detections.tracker_id[i]
                class_id = tracked_detections.class_id[i]
                conf = tracked_detections.confidence[i]
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                labels.append(f"{class_name}")
            
            # Cache results for skipped frames
            last_tracked_detections = tracked_detections
            last_labels = labels
        else:
            # Use cached detections for skipped frames
            tracked_detections = last_tracked_detections
            labels = last_labels
        
        # Draw annotations
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, tracked_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, tracked_detections, labels=labels)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Show live
        cv2.imshow("Logo Detection Live", annotated_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user.")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")
except Exception as e:
    print(f"\nError during processing: {e}")
finally:
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save remaining tracks
    track_logger.finalize_all()
    df = track_logger.get_results()
    df.to_csv("../outputs/detections.csv", index=False)
