
import cv2
import numpy as np

def analyze_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Accumulate flow
    total_dx = 0
    total_dy = 0
    frame_count = 0
    
    # Check first 50 frames
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Mean flow excluding small noise
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = mag > 2
        
        if np.any(mask):
            dx = np.mean(flow[..., 0][mask])
            dy = np.mean(flow[..., 1][mask])
            total_dx += dx
            total_dy += dy
        
        prev_gray = gray
        frame_count += 1
        
    cap.release()
    
    print(f"Average DX: {total_dx/frame_count:.2f}")
    print(f"Average DY: {total_dy/frame_count:.2f}")
    
    if abs(total_dy) > abs(total_dx):
        print("Dominant Motion: VERTICAL")
    else:
        print("Dominant Motion: HORIZONTAL")

analyze_motion("Task.mp4")
