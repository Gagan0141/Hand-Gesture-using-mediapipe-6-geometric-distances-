import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

MODEL_PATH = "models/tf_gesture_model.keras"
LABELS_PATH = "models/labels.txt"

model = load_model(MODEL_PATH)
labels = [l.strip() for l in open(LABELS_PATH)]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

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

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = extract_69_features(hand).reshape(1, 69)
        pred = model.predict(features, verbose=0)[0]
        idx = np.argmax(pred)

        cv2.putText(frame, f"{labels[idx]} {pred[idx]:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
