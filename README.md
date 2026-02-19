# Cylinder Counter

Automated system for counting gas cylinders on a moving truck using YOLOv8 and ByteTrack.

## Overview
This project processes top-down operational footage to detect and count gas cylinders. It utilizes a fine-tuned YOLOv8 model for detection and ByteTrack for robust object tracking across frames. A bidirectional line-crossing algorithms ensures accurate counts even with jittery motion.

## Features
- **Object Detection**: Custom fine-tuned YOLOv8 Nano model.
- **Tracking**: Integrated ByteTrack for ID persistence.
- **Counting**: Configurable ROI (Region of Interest) zone-based counting.
- ** visualization**: Generates annotated video with bounding boxes and live count.

## Prerequisites
- Python 3.8+
- GPU support (Recommended for training)
- Libraries: `ultralytics`, `opencv-python`, `numpy`

## Usage (Google Colab / Local)
1. Ensure the dataset `Annotate Cyinders.v2i.yolov8.zip` and input video `Task.mp4` are in the project root.
2. Run the main script:
   ```bash
   python cylinder_counter_colab.py
   ```
   This will:
   - Install dependencies (if missing)
   - Unzip the dataset
   - Train the YOLOv8 model for 100 epochs
   - Run inference on `Task.mp4`
   - Save output to `output_tracking.mp4`

## Configuration
Key parameters in `cylinder_counter_colab.py`:
- `DATASET_ZIP`: Path to dataset archive
- `VIDEO_PATH`: Input video path
- `line_y`: Vertical position of the counting line (default: 50% height)
