import cv2
import mediapipe as mp
import numpy as np
from fer import FER

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()
emotion_detector = FER()

def draw_lines_and_landmarks(frame, points, color=(0, 255, 0), thickness=2):
    for i in range(len(points) - 1):
        start = (int(points[i][0]), int(points[i][1]))
        end = (int(points[i + 1][0]), int(points[i + 1][1]))
        cv2.line(frame, start, end, color, thickness)
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, color, -1)

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
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0],
        ]
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0],
        ]
        left_hand = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0],
        ]
        right_hand = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0],
        ]

        body_points = [left_shoulder, right_shoulder, right_hip, left_hip, left_shoulder]
        arms_points = [left_shoulder, left_hand, right_hand, right_shoulder]

        draw_lines_and_landmarks(frame, body_points, color=(0, 255, 0))
        draw_lines_and_landmarks(frame, arms_points, color=(255, 0, 0))

        vertical_diff = abs(left_shoulder[1] - right_shoulder[1])
        if vertical_diff < 10:
            return "Good posture: Shoulders aligned", (0, 255, 0)
        return "Slouching detected: Shoulders misaligned", (0, 0, 255)

    return None, (255, 255, 255)

def analyze_facial_expressions(frame):
    emotions = emotion_detector.detect_emotions(frame)
    if emotions:
        main_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        if main_emotion in ["neutral", "happy"]:
            return "Confident expression", (0, 255, 0)
        return "Low-confidence expression", (0, 0, 255)
    return "No face detected", (255, 255, 255)

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        body_suggestion, body_color = "", (255, 255, 255)
        facial_suggestion, facial_color = "", (255, 255, 255)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_suggestion, body_color = analyze_body_language(frame, landmarks)
        
        facial_suggestion, facial_color = analyze_facial_expressions(frame)

        combined_suggestion = f"{body_suggestion} | {facial_suggestion}"
        cv2.putText(frame, combined_suggestion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(frame, combined_suggestion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Posture and Emotion Analysis", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
