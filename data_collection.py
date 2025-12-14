import cv2
import mediapipe as mp
import numpy as np
import csv
import os

GESTURE_NAME = input().strip()
SAVE_IMAGES = True
DATA_DIR = "dataset"
IMG_DIR = os.path.join(DATA_DIR, "images", GESTURE_NAME)
CSV_PATH = os.path.join(DATA_DIR, f"{GESTURE_NAME}.csv")

os.makedirs(IMG_DIR, exist_ok=True)

existing_count = 0
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r") as f:
        existing_count = sum(1 for _ in f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.65)

def extract_69_features(hand):
    lm = np.array([[p.x, p.y, p.z] for p in hand.landmark])
    lm -= lm[0]
    flat = lm.flatten()
    extra = np.array([
        np.linalg.norm(lm[4] - lm[8]),
        np.linalg.norm(lm[8] - lm[12]),
        np.linalg.norm(lm[12] - lm[16]),
        np.linalg.norm(lm[16] - lm[20]),
        np.linalg.norm(lm[5] - lm[17]),
        np.linalg.norm(lm[0] - lm[9])
    ])
    return np.concatenate([flat, extra])

def crop_hand(frame, hand):
    h, w, _ = frame.shape
    xs = [p.x * w for p in hand.landmark]
    ys = [p.y * h for p in hand.landmark]
    x1, x2 = max(0, int(min(xs)) - 20), min(w, int(max(xs)) + 20)
    y1, y2 = max(0, int(min(ys)) - 20), min(h, int(max(ys)) + 20)
    return frame[y1:y2, x1:x2]

cap = cv2.VideoCapture(1)
count = 0
MAX_SAMPLES = 300

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    while count < MAX_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            writer.writerow(extract_69_features(hand))

            if SAVE_IMAGES:
                img = crop_hand(frame, hand)
                cv2.imwrite(os.path.join(IMG_DIR, f"{GESTURE_NAME}_{existing_count + count}.jpg"), img)

            count += 1

        cv2.imshow("collect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
