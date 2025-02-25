import cv2
import numpy as np
import onnxruntime as ort

class PoseTracker:
    def __init__(self, model_path, is_snapdragon=False):
        # Initialize ONNX Runtime session with the Qualcomm-optimized model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.is_snapdragon = is_snapdragon
        
    def preprocess_image(self, frame):
        input_size = (128, 128)
        image = cv2.resize(frame, input_size)
        # Normalize to [0,1] and convert to float32
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        # Add batch dimension and transpose to NCHW format
        image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)
        return image
        
    def detect_pose(self, frame):
        preprocessed = self.preprocess_image(frame)
        # Get model output
        outputs = self.session.run(None, {self.input_name: preprocessed})
        landmarks = outputs[0][0]  # Take first set of landmarks
        
        # Draw landmarks
        self.draw_arm_landmarks(frame, landmarks)
        return frame, landmarks
        
    def draw_arm_landmarks(self, frame, landmarks):
        height, width = frame.shape[:2]
        
        # Right arm indices (based on MediaPipe model)
        key_points = [(12, 14), (14, 16)]  # (shoulder-elbow, elbow-wrist)
        
        # Draw landmarks and connections
        for start_idx, end_idx in key_points:
            start_point = (int(landmarks[start_idx][0] * width), int(landmarks[start_idx][1] * height))
            end_point = (int(landmarks[end_idx][0] * width), int(landmarks[end_idx][1] * height))
            
            # Draw points
            cv2.circle(frame, start_point, 5, (245, 117, 66), -1)
            cv2.circle(frame, end_point, 5, (245, 117, 66), -1)
            
            # Draw lines
            cv2.line(frame, start_point, end_point, (245, 117, 66), 2) 