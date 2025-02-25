import cv2
import numpy as np
from pose_tracker import PoseTracker
import sys
import platform
import os

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def main():
    # Check if running on Snapdragon
    is_snapdragon = "Linux" in platform.system() and os.uname().machine == "aarch64"
    if is_snapdragon:
        print("Running on Snapdragon device")
    
    model_path = "/Users/soham/qualhack/mediapipe_pose-mediapipeposedetector.onnx"
    
    # Initialize video capture with Snapdragon optimizations
    cap = cv2.VideoCapture(0)
    if is_snapdragon:
        # Optimize for Snapdragon camera pipeline
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        sys.exit(1)

    pose_tracker = PoseTracker(model_path, is_snapdragon=is_snapdragon)
    counter = 0
    stage = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame, landmarks = pose_tracker.detect_pose(frame)
        
        try:
            # Get coordinates
            shoulder = [landmarks[12][0], landmarks[12][1]]
            elbow = [landmarks[14][0], landmarks[14][1]]
            wrist = [landmarks[16][0], landmarks[16][1]]
            
            # Calculate angle using the more reliable method
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            elbow_px = tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int))
            cv2.putText(frame, f'{int(angle)}°', 
                       (elbow_px[0] + 10, elbow_px[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Use the working thresholds
            if angle > 155:
                stage = "down"
            if angle < 40 and stage == "down":
                stage = "up"
                counter += 1
            
            # Status box
            cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
            
            # Rep counter
            cv2.putText(frame, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage display
            cv2.putText(frame, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(frame, stage or "none", (85,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        cv2.imshow('Qualcomm AI Exercise Counter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 