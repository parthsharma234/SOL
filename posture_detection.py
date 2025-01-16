import cv2
import mediapipe as mp
import numpy as np
from fer import FER  

# Initialize mediapipe for face and body landmark detection
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()

# Initialize FER for emotion detection
emotion_detector = FER()

# Function to draw landmarks
def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(frame, (x, y), 5, color, -1)

# Analyze body posture for slouching
def analyze_body_language(frame, landmarks):
    if landmarks:
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0],
        ]
        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0],
        ]

        # Vertical difference 
        vertical_diff = abs(left_shoulder[1] - right_shoulder[1])

        # Define thresholds for slouching
        if vertical_diff < 10:  # Adjust this threshold based on your setup
            suggestion = "Good posture: Shoulders are aligned"
            color = (0, 255, 0)  # Green
        else:
            suggestion = "Slouching detected: Shoulders are misaligned"
            color = (0, 0, 255)  # Red

        # Draw landmarks on the shoulders
        draw_landmarks(frame, [left_shoulder, right_shoulder], color)
        return suggestion, color
    return None, (255, 255, 255)

# Analyze facial expressions for emotional confidence
def analyze_facial_expressions(frame):
    emotions = emotion_detector.detect_emotions(frame)
    if emotions:
        main_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        if main_emotion in ["neutral", "happy"]:
            return "Confident facial expression", (0, 255, 0)  # Green
        else:
            return "Low-confidence expression detected", (0, 0, 255)  # Red
    return "No face detected", (255, 255, 255)  # White

# Main function
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

       
        frame = cv2.flip(frame, 1)

      
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        body_suggestion = ""
        facial_suggestion = ""

       
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_suggestion, body_color = analyze_body_language(frame, landmarks)
            cv2.putText(frame, body_suggestion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, body_color, 2)

        
        facial_suggestion, facial_color = analyze_facial_expressions(frame)
        cv2.putText(frame, facial_suggestion, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, facial_color, 2)

       
        cv2.imshow("Confidence and Posture Analysis", frame)

        # Break on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
