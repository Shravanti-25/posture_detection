from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
import threading

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize pyttsx3 for voice-over
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of the speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to generate video feed for the frontend
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Initialize feedback text
        neck_feedback = ""
        torso_feedback = ""

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant points
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angles
            neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)
            torso_angle = calculate_angle(left_shoulder, left_hip, left_ankle)

            # Feedback logic for neck
            if neck_angle > 30:
                neck_feedback = "Neck Bend Detected"
                color_neck = (0, 0, 255)  # Red
                # Trigger text-to-speech alert
                threading.Thread(target=speak_posture, args=("Neck bend detected!",)).start()
            else:
                neck_feedback = "Neck Aligned"
                color_neck = (0, 255, 0)  # Green

            # Feedback logic for torso
            if torso_angle > 10:
                torso_feedback = "Torso Misaligned"
                color_torso = (0, 0, 255)  # Red
            else:
                torso_feedback = "Torso Aligned"
                color_torso = (0, 255, 0)  # Green

            # Display angles and feedback
            cv2.putText(frame, f"Neck Angle: {int(neck_angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_neck, 2)
            cv2.putText(frame, f"Torso Angle: {int(torso_angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_torso, 2)
            cv2.putText(frame, neck_feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_neck, 2)
            cv2.putText(frame, torso_feedback, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_torso, 2)

            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to handle text-to-speech in a separate thread
def speak_posture(posture_message):
    engine.say(posture_message)
    engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_posture', methods=['POST'])
def detect_posture():
    posture_message = "Posture detected! Adjust your posture."
    threading.Thread(target=speak_posture, args=(posture_message,)).start()
    return jsonify({"message": "Posture detected!"})

if __name__ == '__main__':
    app.run(debug=True)
