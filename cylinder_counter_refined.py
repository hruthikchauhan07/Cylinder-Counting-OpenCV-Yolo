import os
import shutil
import cv2
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
DATASET_ZIP = "Annotate Cyinders.v2i.yolov8.zip"
VIDEO_PATH = "Task.mp4"
OUTPUT_VIDEO_PATH = "output_tracking_refined.mp4" # Changed output name
DATASET_DIR = "Annotate Cyinders.v2i.yolov8"

def setup_environment():
    """Installs necessary libraries."""
    try:
        import ultralytics
        ultralytics.checks()
    except ImportError:
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
    # Using 'yolov8n.pt' as per original notebook, or 'yolov8m.pt' from v2?
    # Notebook used 'n', v2 used 'm'. User asked for modification of notebook which had 'n'.
    # I will stick to 'n' to match the notebook unless v2 was preferred. 
    # Actually, v2 used 'm' and 50 epochs. Notebook used 'n' and 100 epochs.
    # The user asked to modify the notebook code. I will use 'n'.
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
    # Define a counting zone (middle 20% of screen height)
    # The cylinders move through this zone to be counted.
    ZONE_Y_MIN = int(h * 0.40)
    ZONE_Y_MAX = int(h * 0.60)
    
    # --- TRUCK ROI CONFIGURATION ---
    # Only cylinders with centers INSIDE this box will be drawn/counted.
    # This filters out detections on the ground/street.
    # Adjust these 0.0-1.0 ratios to fit the truck in your video.
    TRUCK_ROI_X_MIN = int(w * 0.15) 
    TRUCK_ROI_X_MAX = int(w * 0.85)
    TRUCK_ROI_Y_MIN = int(h * 0.05)
    TRUCK_ROI_Y_MAX = int(h * 0.95)

    counted_ids = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run tracking
        results = model.track(frame, persist=True, verbose=False)
        
        annotated_frame = frame.copy()
        
        # Optional: Draw Valid Region Box (Debug) - Commented out to reduce noise
        # cv2.rectangle(annotated_frame, (TRUCK_ROI_X_MIN, TRUCK_ROI_Y_MIN), (TRUCK_ROI_X_MAX, TRUCK_ROI_Y_MAX), (0, 0, 0), 1)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 1. TRUCK ROI FILTER: Check if center is inside the truck area
                if (TRUCK_ROI_X_MIN < cx < TRUCK_ROI_X_MAX) and (TRUCK_ROI_Y_MIN < cy < TRUCK_ROI_Y_MAX):
                    
                    # 2. DRAW LIGHTER BOXES
                    # Color: Light Cyan (255, 255, 200) - BGR
                    # Thickness: 1 (Small/Thin)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 200), 1)
                    
                    # Draw ID (Small and unobtrusive)
                    cv2.putText(annotated_frame, f"{track_id}", (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)

                    # 3. COUNTING LOGIC (Zone based)
                    if ZONE_Y_MIN < cy < ZONE_Y_MAX:
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)

        # 4. DRAW BIG HIGHLIGHTED COUNT
        count_str = f"Total: {len(counted_ids)}"
        font_scale = 2.5
        thickness = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(count_str, font, font_scale, thickness)
        
        # Draw background rectangle
        # Position: Top-left cornere (approx (30, 80) for text)
        # Box coords: (20, 20) to (20+w+pad, 20+h+pad)
        box_x1, box_y1 = 20, 20
        box_x2, box_y2 = 20 + text_w + 20, 80 + 20 # 80 is baseline y roughly
        
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        
        # Draw Text
        # Color: Yellow-ish (0, 255, 255)
        cv2.putText(annotated_frame, count_str, (30, 80), 
                    font, font_scale, (0, 255, 255), thickness)

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
                latest_run = max(runs, key=os.path.getmtime)
                best_model = os.path.join(latest_run, "weights", "best.pt")
                if os.path.exists(best_model):
                        print(f"Using model: {best_model}")
                        process_video(best_model)
                else:
                        print("Model weights not found. Check training output.")
