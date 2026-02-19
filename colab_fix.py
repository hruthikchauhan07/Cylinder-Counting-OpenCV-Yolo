import yaml
import os
from ultralytics import YOLO

DATASET_ZIP = "Annotate Cyinders.v2i.yolov8.zip"
DATASET_DIR = "Annotate Cyinders.v2i.yolov8"

# 1. Unzip if needed
if os.path.exists(DATASET_ZIP):
    # Just unzip if we haven't already seen the folder
    if not os.path.isdir(DATASET_DIR) and not os.path.exists("data.yaml"):
        print("Unzipping dataset...")
        !unzip -q "{DATASET_ZIP}"
        print("Unzip complete.")
    else:
        print("Dataset files found.")

# 2. Find data.yaml and DETERMINE ROOT
data_yaml_path = None
dataset_root = None

if os.path.exists("data.yaml"):
    data_yaml_path = os.path.abspath("data.yaml")
    dataset_root = os.getcwd() # It's in the current folder
elif os.path.exists(os.path.join(DATASET_DIR, "data.yaml")):
    data_yaml_path = os.path.abspath(os.path.join(DATASET_DIR, "data.yaml"))
    dataset_root = os.path.abspath(DATASET_DIR)

if data_yaml_path:
    print(f"Found data.yaml at: {data_yaml_path}")
    print(f"Dataset root appears to be: {dataset_root}")
    
    # 3. FIX: Rewrite data.yaml with CORRECT RELATIVE PATHS
    # YOLO expects paths relative to where it runs OR absolute paths.
    # Absolute paths are safer.
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Check if 'train' folder exists in root or dataset_dir
    # We need to find where 'train/images' actually IS.
    
    images_dir = None
    if os.path.exists(os.path.join(dataset_root, "train", "images")):
         images_dir = dataset_root
    elif os.path.exists(os.path.join(dataset_root, "images", "train")):
         images_dir = dataset_root
    
    if images_dir:
        # Update to ABSOLUTE paths
        data['path'] = images_dir # YOLOv8 usage: root path
        data['train'] = "train/images"
        data['val'] = "valid/images"
        data['test'] = "test/images"
        
        # Save updated yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data, f)
        print(f"Updated data.yaml. Root set to: {images_dir}")

        # 4. Train
        print("Starting Training...")
        model = YOLO("yolov8n.pt")
        results = model.train(data=data_yaml_path, epochs=100, imgsz=640)
        print("Training Complete.")
    else:
        print(f"ERROR: Could not locate 'train/images' inside {dataset_root}. Check directory structure.")
        # Debug listing
        print("Listing directory:")
        print(os.listdir(dataset_root))
else:
    print("ERROR: data.yaml not found! Please check if the zip file was uploaded and unzipped correctly.")
