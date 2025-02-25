import numpy as np

class ExerciseCounter:
    def __init__(self):
        self.counter = 0
        self.stage = None
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def count_bicep_curls(self, keypoints):
        if len(keypoints) < 1:
            return self.counter
            
        # Extract relevant keypoints (adjust indices based on model output)
        shoulder = np.array([keypoints[5][0], keypoints[5][1]])  # Right shoulder
        elbow = np.array([keypoints[6][0], keypoints[6][1]])    # Right elbow
        wrist = np.array([keypoints[7][0], keypoints[7][1]])    # Right wrist
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Count reps
        if angle > 160:
            self.stage = "down"
        elif angle < 30 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            
        return self.counter 