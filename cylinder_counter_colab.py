# Cylinder Counting Solution
# Detects and counts gas cylinders from top-down video using YOLOv8 and ByteTrack.

import os
import shutil
import cv2
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
DATASET_ZIP = "Annotate Cyinders.v2i.yolov8.zip"
VIDEO_PATH = "Task.mp4"
OUTPUT_VIDEO_PATH = "output_tracking.mp4"
DATASET_DIR = "Annotate Cyinders.v2i.yolov8" 

def setup_environment():
    """Installs necessary libraries."""
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    import ultralytics
    ultralytics.checks()

def prepare_dataset():
    """Unzips the dataset."""
    if not os.path.exists(DATASET_ZIP):
        print(f"Error: {DATASET_ZIP} not found.")
        return False

    if os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' already exists. Skipping unzip.")
    else:
        print(f"Unzipping {DATASET_ZIP}...")
        os.system(f"unzip -q '{DATASET_ZIP}'")
        print("Unzip complete.")
    return True

def train_model():
    """Trains the YOLOv8 model."""
    print("Starting training...")
    model = YOLO("yolov8n.pt") 
    data_yaml_path = os.path.join(os.getcwd(), DATASET_DIR, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
         # Fallback if unzipped structure is slightly different
         possible_yaml = os.path.join(os.getcwd(), DATASET_DIR, "**", "data.yaml")
         import glob
         found = glob.glob(possible_yaml, recursive=True)
         if found:
             data_yaml_path = found[0]
         else:
             print(f"Error: data.yaml not found in {DATASET_DIR}.")
             return None

    # Train for 100 epochs
    results = model.train(data=data_yaml_path, epochs=100, imgsz=640)
    print("Training complete.")
    return model

def process_video(model_path):
    print(f"Processing video: {VIDEO_PATH}...")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
        return

    # Load model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Error reading video file"
    
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --- COUNTING CONFIGURATION ---
    # Define a counting zone (middle 10% of screen)
    zone_y_min = int(h * 0.45)
    zone_y_max = int(h * 0.55)
    
    counted_ids = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run tracking (but don't plot automatically to avoid boxes)
        results = model.track(frame, persist=True, verbose=False)
        
        # Start with clean frame (or original frame) for output
        # User doesn't want boxes, so we just use the original frame
        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cy = float((y1 + y2) / 2)

                # Check if centroid is in the counting zone
                if zone_y_min < cy < zone_y_max:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)

        # Draw just the Count (Cleaner output)
        cv2.putText(annotated_frame, f"Total Count: {len(counted_ids)}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        video_writer.write(annotated_frame)
    
    cap.release()
    video_writer.release()
    print(f"Processing complete. Saved to {OUTPUT_VIDEO_PATH}")
    print(f"Total Cylinders Counted: {len(counted_ids)}")

if __name__ == "__main__":
    setup_environment()
    if prepare_dataset():
        trained_model = train_model()
        if trained_model:
            import glob
            # Get latest run
            runs = glob.glob("runs/detect/train*")
            if runs:
                # best.pt might be in 'weights' folder
                latest_run = max(runs, key=os.path.getmtime)
                best_model = os.path.join(latest_run, "weights", "best.pt")
                if os.path.exists(best_model):
                     print(f"Using model: {best_model}")
                     process_video(best_model)
                else:
                     print("Model weights not found. Check training output.")
