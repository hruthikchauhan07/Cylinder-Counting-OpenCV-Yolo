# Cylinder Counting System 🧪

A computer vision solution to detect and count gas cylinders from a top-down video using **YOLOv8** (Object Detection) and **ByteTrack** (Object Tracking).

## Project Output

You can find the project output at the following link:
[Project Output](https://drive.google.com/file/d/1pQi74DQNRkitk535ZPZG9Tm6QJF_ITmc/view?usp=sharing)

## 🚀 Features

*   **Robust Detection:** Fine-tuned YOLOv8n model on custom cylinder dataset.
*   **Accurate Tracking:** Uses ByteTrack to maintain object IDs across frames.
*   **Smart Counting:** 
    *   **ROI Filtering:** Ignores objects outside the truck bed (e.g., street noise).
    *   **Counting Zone:** Increments count only when cylinders pass through the central validation zone.
*   **Optimized Visualization:** Clean, unobtrusive bounding boxes and a clear heads-up display for the count.

## 📂 Project Structure

*   `cylinder_counter_refined.py`: **Main Script.** The refined, standalone Python script for local execution.
*   `Cylinder_Counting_Colab.ipynb`: **Recommended.** Optimized notebook for Google Colab (handles GPU training & path fixes automatically).
*   `requirements.txt`: Python dependencies.
*   `Annotate Cyinders.v2i.yolov8.zip`: The dataset used for training/fine-tuning.
*   `Task.mp4`: Input video file.

## 🛠️ Setup & Usage

### Option 1: Google Colab (Recommended)
**Use this if you don't have a powerful GPU locally.**

1.  Upload `Cylinder_Counting_Colab.ipynb` to Google Colab.
2.  Enable GPU runtime (`Runtime` > `Change runtime type` > `T4 GPU`).
3.  Upload `Annotate Cyinders.v2i.yolov8.zip` and `Task.mp4` to the file session.
4.  Run all cells. The notebook will handle environment setup, dataset preparation, training, and inference automatically.

### Option 2: Local Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cylinder-counting.git
    cd cylinder-counting
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the refined counter:**
    ```bash
    python cylinder_counter_refined.py
    ```

## 🧠 Methodology

1.  **Object Detection (YOLOv8n)**:
    *   The model identifies gas cylinders in each frame.
    *   It was trained for 100 epochs on the provided dataset to handle the specific top-down view.

2.  **Object Tracking (ByteTrack)**:
    *   Assigns a unique ID to each detected cylinder.
    *   Ensures that a cylinder is not counted multiple times as it moves through the frame.

3.  **Counting Logic (ROI + Zone)**:
    *   **Truck ROI:** A bounding box `(x_min, y_min, x_max, y_max)` filters out detections that are not inside the truck.
    *   **Counting Line/Zone:** A horizontal zone (40%-60% height) detects when a cylinder cleanly passes the middle of the screen.

## 📊 Results

*   **Accuracy:** High precision in distinguishing individual cylinders, even when tightly packed.
*   **Performance:** Real-time processing capability on standard GPUs (T4/RTX series).

## 👤 Author

[Hruthik M]
