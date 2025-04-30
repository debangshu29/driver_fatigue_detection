import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import threading
import pyttsx3
import time
from collections import deque

# Initialize pyttsx3 engine once
engine = pyttsx3.init()
eye_alarm_on = False
yawn_alarm_on = False
hazard_alarm_on = False  # NEW

# Alarm for eyes
def play_eye_alarm():
    while eye_alarm_on:
        engine.say("Wake up! Wake up! You are sleeping!")
        engine.runAndWait()
        time.sleep(1)

# Alarm for yawning
def play_yawn_alarm():
    while yawn_alarm_on:
        engine.say("Take rest, you are yawning!")
        engine.runAndWait()
        time.sleep(1)

# Alarm for hazard
def play_hazard_alarm():
    while hazard_alarm_on:
        engine.say("Hazard lights activated! Driver not responding!")
        engine.runAndWait()

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Eye and mouth landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [13, 14]  # Upper and lower lips

def calculate_EAR(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.20
EYE_FRAME_LIMIT = 25
CLOSED_FRAMES = 0

MOUTH_OPEN_THRESHOLD = 25
YAWN_FRAMES = 0
YAWN_FRAME_LIMIT = 15

HAZARD_TRIGGER_LIMIT = 100
hazard_on = False
hazard_blink_state = False
hazard_last_toggle_time = time.time()

ear_history = deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            left_eye, right_eye = [], []

            for idx in LEFT_EYE_IDX:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))
            for idx in RIGHT_EYE_IDX:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))

            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            ear_history.append(avg_ear)
            smoothed_ear = sum(ear_history) / len(ear_history)

            if smoothed_ear < EAR_THRESHOLD:
                CLOSED_FRAMES += 1
            else:
                CLOSED_FRAMES = 0
                eye_alarm_on = False

            if CLOSED_FRAMES > EYE_FRAME_LIMIT and not eye_alarm_on:
                eye_alarm_on = True
                threading.Thread(target=play_eye_alarm, daemon=True).start()

            # Hazard trigger
            if CLOSED_FRAMES > HAZARD_TRIGGER_LIMIT:
                if not hazard_on:
                    hazard_on = True
                    if not hazard_alarm_on:
                        hazard_alarm_on = True
                        threading.Thread(target=play_hazard_alarm, daemon=True).start()
            else:
                hazard_on = False
                hazard_alarm_on = False

            # Enhanced mouth tracking
            MOUTH_TRACK_IDX = [13, 14, 78, 308, 82, 87, 317, 312]  # key points around lips
            mouth_points = []

            for idx in MOUTH_TRACK_IDX:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                mouth_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # draw yellow dots

            # Calculate vertical mouth distance using top and bottom lip (13 and 14)
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]
            mouth_open_dist = abs(top_lip.y - bottom_lip.y) * h

            if mouth_open_dist > MOUTH_OPEN_THRESHOLD:
                YAWN_FRAMES += 1
            else:
                YAWN_FRAMES = 0
                yawn_alarm_on = False

            if YAWN_FRAMES > YAWN_FRAME_LIMIT and not yawn_alarm_on:
                yawn_alarm_on = True
                threading.Thread(target=play_yawn_alarm, daemon=True).start()

            eye_state = "Closed" if smoothed_ear < EAR_THRESHOLD else "Open"
            color = (0, 0, 255) if eye_state == "Closed" else (0, 255, 0)
            cv2.putText(frame, f'Eye State: {eye_state}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f'EAR: {smoothed_ear:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if CLOSED_FRAMES > EYE_FRAME_LIMIT:
                cv2.putText(frame, "SLEEPING DETECTED", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

            if YAWN_FRAMES > YAWN_FRAME_LIMIT:
                cv2.putText(frame, "YAWNING DETECTED", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

            # Hazard blinking effect
            if hazard_on:
                current_time = time.time()
                if current_time - hazard_last_toggle_time > 0.5:
                    hazard_blink_state = not hazard_blink_state
                    hazard_last_toggle_time = current_time
                if hazard_blink_state:
                    cv2.putText(frame, " HAZARD LIGHTS ACTIVATED ", (50, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

    cv2.imshow('Driver Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
