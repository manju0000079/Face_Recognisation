# recognize_lbph_live.py
import cv2
import json
import time
import os

# ensure model exists
if not os.path.exists("models/lbph_model.yml") or not os.path.exists("models/labels.json"):
    print("Model files not found. Please run train_lbph.py first.")
    raise SystemExit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_model.yml")

with open("models/labels.json", "r") as f:
    label_id = json.load(f)
id_label = {v: k for k, v in label_id.items()}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam. Try closing other apps or change index (0 -> 1).")
    raise SystemExit(1)

CONFIDENCE_THRESHOLD = 70 # lower = stricter

print("Starting live recognition. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in rects:
        roi = gray[y:y+h, x:x+w]
        try:
            roi_resized = cv2.resize(roi, (200, 200))
        except:
            continue
        label, conf = recognizer.predict(roi_resized)
        name = id_label.get(label, "Unknown")
        display_name = name if conf <= CONFIDENCE_THRESHOLD else "Unknown"
        text = f"{display_name} ({conf:.1f})"
        color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("LBPH Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
